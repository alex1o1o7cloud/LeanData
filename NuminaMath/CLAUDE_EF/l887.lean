import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_reciprocal_l887_88786

/-- The function f(x) = x^2 / (1 + x^2) -/
noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

/-- Theorem: f(x) + f(1/x) = 1 for all x ≠ 0 -/
theorem f_sum_reciprocal (x : ℝ) (hx : x ≠ 0) : f x + f (1/x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_reciprocal_l887_88786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_composites_l887_88758

def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬Nat.Prime n

theorem consecutive_composites
  (a t d r : ℕ)
  (ha : a > 1)
  (ht : t > 1)
  (hd : d > 1)
  (hr : r > 1)
  (hac : isComposite a)
  (htc : isComposite t)
  (hdc : isComposite d)
  (hrc : isComposite r) :
  ∃ x : ℕ, ∀ j ∈ Finset.range r, isComposite ((a * t^(x + j)) + d) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_composites_l887_88758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_resistance_change_approx_l887_88783

/-- The change in total resistance when switch K is closed -/
noncomputable def resistance_change (R₁ R₂ R₃ : ℝ) : ℝ :=
  let R₀ := R₁
  let R_K := 1 / (1 / R₁ + 1 / R₂ + 1 / R₃)
  R_K - R₀

/-- Theorem stating the change in total resistance for given values -/
theorem resistance_change_approx :
  let R₁ : ℝ := 4
  let R₂ : ℝ := 8
  let R₃ : ℝ := 16
  abs (resistance_change R₁ R₂ R₃ + 1.7) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_resistance_change_approx_l887_88783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_class_drunk_drivers_l887_88717

theorem traffic_class_drunk_drivers :
  ∃ (d : ℕ), 
    (d + (7 * d - 3) + 2 * d = 105) ∧ 
    (d = 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_class_drunk_drivers_l887_88717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_theorem_l887_88734

-- Define the variables
variable (x y z : ℝ)

-- Define the relationship between x and y
def relation (x y : ℝ) : Prop := y = -2 * x + 1

-- Define positive correlation
def positively_correlated (a b : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → a x₁ < a x₂ ∧ b x₁ < b x₂

-- Define negative correlation
def negatively_correlated (a b : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → a x₁ > a x₂ ∧ b x₁ < b x₂

-- Theorem statement
theorem correlation_theorem 
  (h1 : relation x y)
  (h2 : positively_correlated (λ t ↦ y) (λ t ↦ z)) :
  negatively_correlated (λ t ↦ x) (λ t ↦ y) ∧
  negatively_correlated (λ t ↦ x) (λ t ↦ z) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_theorem_l887_88734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borya_wins_l887_88746

/-- A game played on a circle with 33 points. -/
structure CircleGame where
  numPoints : Nat
  numPoints_eq : numPoints = 33

/-- The possible colors for painting points. -/
inductive Color
  | Blue
  | Red

/-- A player in the game. -/
inductive Player
  | Anya
  | Borya
deriving Repr, DecidableEq

/-- The state of the game board. -/
def GameState := List Color

/-- Check if a move is legal (doesn't create two adjacent same-color points). -/
def isLegalMove (state : GameState) (position : Nat) (color : Color) : Bool :=
  sorry

/-- The next player to move. -/
def nextPlayer (current : Player) : Player :=
  match current with
  | Player.Anya => Player.Borya
  | Player.Borya => Player.Anya

/-- A winning strategy for a player. -/
def WinningStrategy (player : Player) :=
  ∀ (game : CircleGame) (state : GameState),
    ∃ (move : Nat), isLegalMove state move (if player = Player.Anya then Color.Blue else Color.Red) ∧
      ¬∃ (opponentMove : Nat), isLegalMove (sorry) opponentMove (if player = Player.Anya then Color.Red else Color.Blue)

/-- The main theorem stating that Borya has a winning strategy. -/
theorem borya_wins (game : CircleGame) : WinningStrategy Player.Borya := by
  sorry

#check borya_wins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borya_wins_l887_88746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_suit_cost_l887_88704

/-- The cost of Paul's suit given his shopping trip details -/
noncomputable def suit_cost (shirt_price : ℝ) (shirt_count : ℕ) 
              (pants_price : ℝ) (pants_count : ℕ)
              (sweater_price : ℝ) (sweater_count : ℕ)
              (store_discount : ℝ) (coupon_discount : ℝ)
              (total_spent : ℝ) : ℝ :=
  let other_items_cost := shirt_price * (shirt_count : ℝ) + 
                          pants_price * (pants_count : ℝ) + 
                          sweater_price * (sweater_count : ℝ)
  let suit_cost := (total_spent / ((1 - store_discount) * (1 - coupon_discount))) - other_items_cost
  suit_cost

theorem paul_suit_cost :
  suit_cost 15 4 40 2 30 2 0.2 0.1 252 = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_suit_cost_l887_88704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_S_l887_88720

/-- The function f(x) = x^2 - 6x + 5 -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The set of points (x, y) satisfying the given conditions -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 + f p.2 ≤ 0 ∧ f p.1 - f p.2 ≥ 0}

/-- The circle (x - 3)^2 + (y - 3)^2 ≤ 8 -/
def circleSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 3)^2 ≤ 8}

/-- The region defined by y ≤ x and y ≥ 6 - x -/
def region1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≤ p.1 ∧ p.2 ≥ 6 - p.1}

/-- The region defined by y ≥ x and y ≤ 6 - x -/
def region2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≥ p.1 ∧ p.2 ≤ 6 - p.1}

/-- The theorem stating that S is a subset of the intersection of the circle and the union of region1 and region2 -/
theorem characterization_of_S : S ⊆ circleSet ∩ (region1 ∪ region2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_S_l887_88720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_length_l887_88714

/-- Given two intersecting lines AC and BD with point of intersection O,
    prove that if OA = 5, OC = 12, OD = 5, OB = 6, and BD = 10, then AC = 13 -/
theorem intersecting_lines_length (A B C D O : EuclideanSpace ℝ (Fin 2)) : 
  ‖A - O‖ = 5 →
  ‖C - O‖ = 12 →
  ‖D - O‖ = 5 →
  ‖B - O‖ = 6 →
  ‖B - D‖ = 10 →
  ‖A - C‖ = 13 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_length_l887_88714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_circle_l887_88784

theorem points_in_circle (points : Finset (ℝ × ℝ)) : 
  (points.card = 51) → 
  (∀ p ∈ points, p.1 ∈ Set.Icc (0 : ℝ) 1 ∧ p.2 ∈ Set.Icc (0 : ℝ) 1) →
  ∃ (center : ℝ × ℝ), (center.1 ∈ Set.Icc (0 : ℝ) 1 ∧ center.2 ∈ Set.Icc (0 : ℝ) 1) ∧
    3 ≤ (points.filter (λ p => Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) ≤ 1/7)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_in_circle_l887_88784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lace_length_for_circular_tablecloth_l887_88756

/-- The length of lace needed for a circular tablecloth border -/
theorem lace_length_for_circular_tablecloth (area : ℝ) (pi_approx : ℝ) (extra_percentage : ℝ) : 
  area = 616 → 
  pi_approx = 22 / 7 → 
  extra_percentage = 0.1 → 
  ∃ (radius circumference lace_length : ℝ),
    area = pi_approx * radius^2 ∧
    circumference = 2 * pi_approx * radius ∧
    lace_length = circumference * (1 + extra_percentage) ∧
    Int.floor (lace_length + 0.5) = 97 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lace_length_for_circular_tablecloth_l887_88756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_may_savings_l887_88737

/-- Calculates the savings for a given month, starting from January (month 1) -/
def savings (month : Nat) : Nat :=
  match month with
  | 0 => 0  -- Handle the zero case
  | 1 => 20  -- January savings
  | n + 1 => 3 * savings n  -- Each subsequent month

theorem may_savings : savings 5 = 1620 := by
  -- Expand the definition of savings for each month
  have h1 : savings 1 = 20 := rfl
  have h2 : savings 2 = 60 := by simp [savings, h1]
  have h3 : savings 3 = 180 := by simp [savings, h2]
  have h4 : savings 4 = 540 := by simp [savings, h3]
  have h5 : savings 5 = 1620 := by simp [savings, h4]
  exact h5

#eval savings 5  -- This will evaluate to 1620

end NUMINAMATH_CALUDE_ERRORFEEDBACK_may_savings_l887_88737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_weavers_total_mats_l887_88727

/-- Represents a weaver's production rate -/
structure WeaverRate where
  mats : ℕ
  days : ℕ

/-- Calculates the daily rate of a weaver -/
def dailyRate (wr : WeaverRate) : ℚ :=
  wr.mats / wr.days

/-- Calculates the total mats produced by a group of weavers in a given number of days -/
def totalMats (rates : List WeaverRate) (projectDays : ℕ) : ℕ :=
  Int.toNat ⌊(rates.map dailyRate).sum * projectDays⌋

/-- Theorem stating the total number of mats woven by four weavers in 10 days -/
theorem four_weavers_total_mats :
  let weaverA : WeaverRate := { mats := 4, days := 6 }
  let weaverB : WeaverRate := { mats := 5, days := 7 }
  let weaverC : WeaverRate := { mats := 3, days := 4 }
  let weaverD : WeaverRate := { mats := 6, days := 9 }
  let weavers : List WeaverRate := [weaverA, weaverB, weaverC, weaverD]
  totalMats weavers 10 = 28 := by
  sorry

#eval totalMats [{ mats := 4, days := 6 }, { mats := 5, days := 7 }, { mats := 3, days := 4 }, { mats := 6, days := 9 }] 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_weavers_total_mats_l887_88727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_divided_rectangle_exists_l887_88767

/-- A rectangle that can potentially be divided into a regular hexagon and right-angled triangles -/
structure DividedRectangle where
  x : ℝ  -- Length of the rectangle
  y : ℝ  -- Width of the rectangle
  n : ℕ  -- Number of right-angled triangles

/-- The area of a regular hexagon with side length 1 -/
noncomputable def hexagonArea : ℝ := 3 * Real.sqrt 3 / 2

/-- The area of a right-angled triangle with legs 1 and √3 -/
noncomputable def triangleArea : ℝ := Real.sqrt 3 / 2

/-- Helper function to check if a real number is in ℤ[√3] -/
def isInZSqrt3 (x : ℝ) : Prop :=
  ∃ (a b : ℤ), x = a + b * Real.sqrt 3

/-- The theorem stating that no such rectangle exists -/
theorem no_divided_rectangle_exists : ¬ ∃ (r : DividedRectangle), 
  r.x * r.y = hexagonArea + r.n * triangleArea ∧
  isInZSqrt3 r.x ∧
  isInZSqrt3 r.y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_divided_rectangle_exists_l887_88767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_complement_when_a_is_3_intersection_equals_N_iff_a_in_range_l887_88785

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 18 ≤ 0}
def N (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 2*a + 1}

-- Part 1
theorem intersection_and_complement_when_a_is_3 :
  (M ∩ N 3 = Set.Icc (-2) 6) ∧ 
  ((Set.univ : Set ℝ) \ N 3 = Set.Iio (-2) ∪ Set.Ioi 7) := by sorry

-- Part 2
theorem intersection_equals_N_iff_a_in_range :
  ∀ a : ℝ, M ∩ N a = N a ↔ a ∈ Set.Iic (5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_complement_when_a_is_3_intersection_equals_N_iff_a_in_range_l887_88785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Y_to_AB_l887_88774

-- Define the square
def Square (s : ℝ) : Set (ℝ × ℝ) := {p | 
  p = (0, 0) ∨ p = (s, 0) ∨ p = (s, s) ∨ p = (0, s)}

-- Define the semicircle from A
def SemicircleA (s : ℝ) : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | p.1^2 + (p.2 - s)^2 = s^2 ∧ p.2 ≥ s}

-- Define the semicircle from D
def SemicircleD (s : ℝ) : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | (p.1 - s)^2 + p.2^2 = s^2 ∧ p.1 ≥ s}

-- Define the intersection point Y
def Y (s : ℝ) : ℝ × ℝ := (s, s)

-- Theorem statement
theorem distance_Y_to_AB (s : ℝ) (h : s > 0) :
  let square := Square s
  let semicircleA := SemicircleA s
  let semicircleD := SemicircleD s
  let y := Y s
  y ∈ semicircleA ∧ y ∈ semicircleD ∧ y.2 = s :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Y_to_AB_l887_88774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_triangle_perimeter_squared_l887_88723

/-- Given a triangle ABC inscribed in a circle of radius r, where AB is a chord
    (not a diameter) of length 2r * sin(θ), and s = AC + BC, 
    the maximum value of s^2 is 4r^2. -/
theorem max_inscribed_triangle_perimeter_squared (r : ℝ) (θ : ℝ) 
  (h_r_pos : r > 0) (h_θ_pos : θ > 0) (h_θ_lt_pi : θ < π) : 
  ∃ (s : ℝ), s = 2 * r * Real.sin θ ∧ ∀ (s' : ℝ), s' = 2 * r * Real.sin θ → s'^2 ≤ 4 * r^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_triangle_perimeter_squared_l887_88723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_no_green_3x3_value_l887_88749

/-- Represents a 4x4 grid where each cell can be either green or yellow -/
def Grid := Fin 4 → Fin 4 → Bool

/-- The probability of a cell being green -/
noncomputable def p_green : ℝ := 1/2

/-- Checks if a 3x3 subgrid starting at (i,j) is all green -/
def is_green_3x3 (g : Grid) (i j : Fin 2) : Prop :=
  ∀ (x y : Fin 3), g (i + x) (j + y)

/-- The probability of not having a 3x3 green square in a 4x4 grid -/
noncomputable def p_no_green_3x3 : ℝ :=
  1 - (4 * (1 - (1 - p_green)^9) - 2 * (1 - (1 - p_green)^12))

theorem p_no_green_3x3_value : p_no_green_3x3 = 2033 / 2048 := by
  sorry

#eval (2033 : Nat) + 2048

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_no_green_3x3_value_l887_88749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_function_l887_88781

/-- A function is symmetric to y = 1/(x+1) about (1,0) if for every point (x, f(x)) on its graph,
    the point (2-x, -f(x)) lies on the graph of y = 1/(x+1) -/
def IsSymmetricTo (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 1 / ((2 - x) + 1) = -f x

/-- If f is symmetric to y = 1/(x+1) about (1,0), then f(x) = 1/(x-3) -/
theorem symmetry_implies_function (f : ℝ → ℝ) (h : IsSymmetricTo f) :
  ∀ x : ℝ, f x = 1 / (x - 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_function_l887_88781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_perpendicular_criterion_l887_88719

/-- A type representing a plane in 3D space -/
structure Plane where

/-- A type representing a line in 3D space -/
structure Line where

/-- Perpendicular relation between a line and a plane -/
def perpendicular (l : Line) (p : Plane) : Prop := sorry

/-- Parallel relation between a line and a plane -/
def parallel (l : Line) (p : Plane) : Prop := sorry

/-- Perpendicular relation between two planes -/
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry

theorem plane_perpendicular_criterion (α β : Plane) : 
  (∃ l : Line, perpendicular l α ∧ parallel l β) → perpendicular_planes α β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_perpendicular_criterion_l887_88719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_l887_88730

/-- The parabola y² = 4x -/
def Parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

/-- The line 2x - y + 3 = 0 -/
def Line (P : ℝ × ℝ) : Prop := 2 * P.1 - P.2 + 3 = 0

/-- Distance from a point to the y-axis -/
noncomputable def DistanceToYAxis (P : ℝ × ℝ) : ℝ := abs P.1

/-- Distance from a point to the line 2x - y + 3 = 0 -/
noncomputable def DistanceToLine (P : ℝ × ℝ) : ℝ := abs (2 * P.1 - P.2 + 3) / Real.sqrt 5

/-- The sum of distances from P to the line and the y-axis -/
noncomputable def SumOfDistances (P : ℝ × ℝ) : ℝ := DistanceToLine P + DistanceToYAxis P

/-- The minimum value of the sum of distances -/
theorem min_sum_of_distances :
  ∀ P : ℝ × ℝ, Parabola P → SumOfDistances P ≥ Real.sqrt 5 - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_l887_88730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirty_degree_same_side_angles_l887_88750

def sameSideAngles (baseAngle : ℝ) : Set ℝ :=
  {θ | ∃ k : ℤ, θ = baseAngle + k * 360}

theorem thirty_degree_same_side_angles :
  sameSideAngles 30 = {θ : ℝ | ∃ k : ℤ, θ = 30 + k * 360} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirty_degree_same_side_angles_l887_88750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l887_88705

-- Define the vector type
def Vec2 := ℝ × ℝ

-- Define vector operations
def vec_add (v w : Vec2) : Vec2 := (v.1 + w.1, v.2 + w.2)
def vec_smul (r : ℝ) (v : Vec2) : Vec2 := (r * v.1, r * v.2)

-- Define the given vectors
def a : Vec2 := (1, 2)
def b : Vec2 := (-3, 0)

-- Define parallelism for Vec2
def parallel (v w : Vec2) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Main theorem
theorem parallel_vectors_m_value :
  ∃ (m : ℝ), parallel (vec_add (vec_smul 2 a) b) (vec_add a (vec_smul (-m) b)) ∧ m = -1/2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l887_88705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phoenix_number_theorem_l887_88713

def is_phoenix_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  List.Pairwise (·≠·) digits ∧ 
  digits.all (λ d => d ≠ 0) ∧
  digits[0]! + digits[2]! = digits[1]! + digits[3]! ∧
  digits[0]! + digits[2]! = 9

def K (n : ℕ) : ℚ := n / 99

def swap_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  c * 1000 + d * 100 + a * 10 + b

theorem phoenix_number_theorem :
  ∀ n : ℕ, 
    is_phoenix_number n →
    Even n →
    is_phoenix_number (swap_digits n) →
    (3 * K n + 2 * K (swap_digits n)).den = 1 →
    (3 * K n + 2 * K (swap_digits n)).num % 9 = 0 →
    n / 1000 ≥ (n / 100) % 10 →
    (n = 8514 ∨ n = 3168) :=
by sorry

#eval K 5841

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phoenix_number_theorem_l887_88713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equation_solution_l887_88741

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := ((x + 3) / 4) ^ (1/3)

-- State the theorem
theorem g_equation_solution :
  ∃ (x : ℝ), g (2 * x) = 2 * g x ∧ x = -7/2 :=
by
  -- Introduce the witness
  use -7/2
  
  -- Split the goal into two parts
  constructor
  
  -- Prove the equation holds for x = -7/2
  · sorry
  
  -- Prove x = -7/2 (trivial)
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equation_solution_l887_88741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_l887_88797

/-- The function representing the curve -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x + Real.log x

/-- The derivative of the function f -/
noncomputable def f_derivative (k : ℝ) (x : ℝ) : ℝ := k + 1 / x

theorem tangent_parallel_to_x_axis (k : ℝ) :
  f_derivative k 1 = 0 → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_l887_88797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_and_b_l887_88764

noncomputable def y (a b x : ℝ) : ℝ := a * Real.sin (2 * x - Real.pi / 3) + b

theorem find_a_and_b :
  ∀ a b : ℝ,
  (∀ x ∈ Set.Icc (-Real.pi/3) (2*Real.pi/3), y a b x ≤ 4) ∧
  (∀ x ∈ Set.Icc (-Real.pi/3) (2*Real.pi/3), y a b x ≥ -2) ∧
  (∃ x₁ ∈ Set.Icc (-Real.pi/3) (2*Real.pi/3), y a b x₁ = 4) ∧
  (∃ x₂ ∈ Set.Icc (-Real.pi/3) (2*Real.pi/3), y a b x₂ = -2) →
  ((a = 3 ∧ b = 1) ∨ (a = -3 ∧ b = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_and_b_l887_88764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l887_88739

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.medianCM : Line := { a := 2, b := -1, c := -5 }
def Triangle.altitudeBH : Line := { a := 1, b := -2, c := -5 }

theorem triangle_properties (t : Triangle) 
  (h1 : t.A = (5, 1))
  (h2 : Triangle.medianCM = { a := 2, b := -1, c := -5 })
  (h3 : Triangle.altitudeBH = { a := 1, b := -2, c := -5 }) :
  (∃ (l : Line), l = { a := 2, b := 1, c := -11 } ∧ 
    (l.a * t.A.1 + l.b * t.A.2 + l.c = 0 ∧ l.a * t.C.1 + l.b * t.C.2 + l.c = 0)) ∧
  t.B = (-1, -3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l887_88739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l887_88715

/-- The function f(x) defined as (7x^5 + 2x^3 + 3x^2 + 8x + 4) / (8x^5 + 5x^3 + 4x^2 + 6x + 2) -/
noncomputable def f (x : ℝ) : ℝ := (7*x^5 + 2*x^3 + 3*x^2 + 8*x + 4) / (8*x^5 + 5*x^3 + 4*x^2 + 6*x + 2)

/-- The horizontal asymptote of f(x) is 7/8 -/
theorem horizontal_asymptote_of_f : 
  ∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, |x| > N → |f x - 7/8| < ε := by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l887_88715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisor_problem_l887_88744

theorem common_divisor_problem (n : ℕ) (hn : n < 50) :
  (∃ d : ℕ, d > 1 ∧ d ∣ (4 * n + 5) ∧ d ∣ (7 * n + 6)) ↔ n ∈ ({7, 18, 29, 40} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisor_problem_l887_88744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l887_88793

-- Define the circles and line
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*a*y = 0
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def line (x y : ℝ) : Prop := x + y = 0

-- Define the chord length
noncomputable def chord_length (a : ℝ) : ℝ := 2 * Real.sqrt 2

-- State the theorem
theorem circles_intersect (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ (x y : ℝ), circle_M a x y ∧ line x y) 
  (h3 : chord_length a = 2 * Real.sqrt 2) :
  ∃ (x y : ℝ), circle_M a x y ∧ circle_N x y := by
  sorry

-- Additional helper lemmas (if needed)
lemma circle_M_center_radius (a : ℝ) (h : a > 0) :
  ∃ (c : ℝ × ℝ) (r : ℝ), ∀ (x y : ℝ), circle_M a x y ↔ (x - c.1)^2 + (y - c.2)^2 = r^2 := by
  sorry

lemma distance_between_centers (a : ℝ) (h : a > 0) :
  Real.sqrt ((0 - 1)^2 + (a - 1)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l887_88793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_to_line_distance_constraint_l887_88771

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 is given by this formula -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The problem statement -/
theorem point_to_line_distance_constraint (a : ℝ) :
  distance_point_to_line 4 a 4 (-3) (-1) ≤ 3 → 0 ≤ a ∧ a ≤ 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_to_line_distance_constraint_l887_88771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_area_is_two_l887_88760

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  (1/2) * abs (t.p1.x * (t.p2.y - t.p3.y) + t.p2.x * (t.p3.y - t.p1.y) + t.p3.x * (t.p1.y - t.p2.y))

/-- Defines the grid spacing -/
def gridSpacing : ℝ := 2

/-- Defines the first triangle -/
def triangle1 : Triangle :=
  { p1 := { x := 0, y := 0 },
    p2 := { x := 2, y := 1 },
    p3 := { x := 0, y := 2 } }

/-- Defines the second triangle -/
def triangle2 : Triangle :=
  { p1 := { x := 2, y := 2 },
    p2 := { x := 0, y := 1 },
    p3 := { x := 2, y := 0 } }

/-- Theorem: The area of the overlapping region of the two triangles is 2 square units -/
theorem overlapping_area_is_two :
  ∃ (overlapArea : ℝ), overlapArea = 2 ∧
  overlapArea ≤ triangleArea triangle1 ∧
  overlapArea ≤ triangleArea triangle2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_area_is_two_l887_88760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_elements_count_l887_88722

def S : Finset ℕ := Finset.range 100

def multiples_of_2 : Finset ℕ := S.filter (λ n => n % 2 = 0)
def multiples_of_5 : Finset ℕ := S.filter (λ n => n % 5 = 0)

def remaining_elements : Finset ℕ := S \ (multiples_of_2 ∪ multiples_of_5)

theorem remaining_elements_count : Finset.card remaining_elements = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_elements_count_l887_88722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_smallest_four_digit_in_pascal_l887_88751

/-- Pascal's triangle is a triangular array of binomial coefficients -/
def pascal_triangle : ℕ → ℕ → ℕ := sorry

/-- Every positive integer appears in Pascal's triangle -/
axiom pascal_contains_all_positive : ∀ n : ℕ, n > 0 → ∃ i j : ℕ, pascal_triangle i j = n

/-- The first row of Pascal's triangle starts with 1 -/
axiom pascal_first_row : pascal_triangle 0 0 = 1

/-- Each row of Pascal's triangle starts and ends with 1 -/
axiom pascal_row_ends : ∀ n : ℕ, pascal_triangle n 0 = 1 ∧ pascal_triangle n n = 1

/-- Each number inside a row is the sum of the two numbers directly above it -/
axiom pascal_sum_rule : ∀ n k : ℕ, k > 0 → k < n → 
  pascal_triangle n k = pascal_triangle (n-1) (k-1) + pascal_triangle (n-1) k

/-- The third smallest four-digit number in Pascal's triangle is 1002 -/
theorem third_smallest_four_digit_in_pascal : 
  ∃ i j : ℕ, pascal_triangle i j = 1002 ∧ 
  (∀ x y : ℕ, pascal_triangle x y ≥ 1000 → pascal_triangle x y = 1000 ∨ pascal_triangle x y = 1001 ∨ pascal_triangle x y ≥ 1002) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_smallest_four_digit_in_pascal_l887_88751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_consumption_l887_88718

-- Define constants
def fuel_efficiency : ℝ := 56 -- km per liter
def travel_time : ℝ := 5.7 -- hours
def speed_mph : ℝ := 91 -- miles per hour
def gallon_to_liter : ℝ := 3.8 -- liters per gallon
def mile_to_km : ℝ := 1.6 -- km per mile

-- Define the theorem
theorem car_fuel_consumption :
  let speed_kph : ℝ := speed_mph * mile_to_km
  let distance : ℝ := speed_kph * travel_time
  let fuel_consumed_liters : ℝ := distance / fuel_efficiency
  let fuel_consumed_gallons : ℝ := fuel_consumed_liters / gallon_to_liter
  ∃ ε > 0, |fuel_consumed_gallons - 3.9| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_consumption_l887_88718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_A_to_ellipse_l887_88799

/-- The ellipse on which point B lies -/
def ellipse (x y : ℝ) : Prop := x^2 + 6 * y^2 = 6

/-- Point A -/
def A : ℝ × ℝ := (0, 2)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The maximum distance between A and any point B on the ellipse -/
theorem max_distance_A_to_ellipse :
  (∃ (B : ℝ × ℝ), ellipse B.1 B.2 ∧
    ∀ (C : ℝ × ℝ), ellipse C.1 C.2 → distance A B ≥ distance A C) ∧
  (∀ (B : ℝ × ℝ), ellipse B.1 B.2 → distance A B ≤ 3 * Real.sqrt 30 / 5) ∧
  (∃ (B : ℝ × ℝ), ellipse B.1 B.2 ∧ distance A B = 3 * Real.sqrt 30 / 5) := by
  sorry

#check max_distance_A_to_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_A_to_ellipse_l887_88799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_negative_x_l887_88709

-- Define an odd function f
noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then -x * Real.log (1 + x) else -(-x * Real.log (1 + (-x)))

-- Define OddFunction
def Function.OddFunction {X : Type*} [Neg X] (f : X → X) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_negative_x (x : ℝ) :
  Function.OddFunction f → (x < 0 → f x = -x * Real.log (1 - x)) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_negative_x_l887_88709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_maximum_l887_88707

/-- Function f₁ -/
noncomputable def f₁ (x : ℝ) : ℝ := x^2 / (1 + x^12)

/-- Function f₂ -/
noncomputable def f₂ (x : ℝ) : ℝ := x^3 / (1 + x^11)

/-- Function f₃ -/
noncomputable def f₃ (x : ℝ) : ℝ := x^4 / (1 + x^10)

/-- Function f₄ -/
noncomputable def f₄ (x : ℝ) : ℝ := x^5 / (1 + x^9)

/-- Function f₅ -/
noncomputable def f₅ (x : ℝ) : ℝ := x^6 / (1 + x^8)

/-- The maximum value of f₄ on positive reals -/
noncomputable def max_f₄ : ℝ := (4/9) * ((5/4) ^ (5/9))

theorem smallest_maximum (x : ℝ) (hx : x > 0) :
  (∀ y > 0, f₄ y ≤ max_f₄) ∧
  (∃ z > 0, f₁ z > max_f₄) ∧
  (∃ z > 0, f₂ z > max_f₄) ∧
  (∃ z > 0, f₃ z > max_f₄) ∧
  (∃ z > 0, f₅ z > max_f₄) :=
by sorry

#check smallest_maximum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_maximum_l887_88707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_different_suits_l887_88747

/-- A standard deck of cards --/
def standard_deck : ℕ := 52

/-- Number of suits in a standard deck --/
def num_suits : ℕ := 4

/-- Number of cards per suit --/
def cards_per_suit : ℕ := standard_deck / num_suits

/-- The probability of selecting three cards of different suits --/
def prob_diff_suits : ℚ := 91 / 170

/-- The probability of selecting three cards of different suits
    is equal to 91/170 --/
theorem probability_three_different_suits :
  prob_diff_suits = 91 / 170 := by
  -- Proof goes here
  sorry

#eval prob_diff_suits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_different_suits_l887_88747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l887_88755

noncomputable def Circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 25}

noncomputable def α : ℝ := Real.arccos (-3/5)

noncomputable def A : ℝ × ℝ := (-3, 4)

noncomputable def B : ℝ × ℝ := (5 * Real.cos (2*α), 5 * Real.sin (2*α))

theorem tangent_line_equation :
  ∃ (sign : Bool), 
    7 * B.1 + (if sign then 24 else -24) * B.2 + 125 = 0 ∧
    ∀ (x y : ℝ), (x, y) ∈ Circle → 
      (7 * x + (if sign then 24 else -24) * y + 125 = 0 ↔ (x, y) = B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l887_88755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_weekly_earnings_l887_88736

/-- Calculates James' weekly earnings given his job rates and hours worked --/
noncomputable def weekly_earnings (main_job_rate : ℝ) (main_job_hours : ℝ) : ℝ :=
  let second_job_rate := main_job_rate * 0.8
  let second_job_hours := main_job_hours / 2
  main_job_rate * main_job_hours + second_job_rate * second_job_hours

/-- Proves that James' weekly earnings are $840 --/
theorem james_weekly_earnings :
  weekly_earnings 20 30 = 840 := by
  -- Unfold the definition of weekly_earnings
  unfold weekly_earnings
  -- Simplify the arithmetic
  simp [mul_add, add_mul, mul_assoc]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_weekly_earnings_l887_88736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_curve_and_line_l887_88708

/-- The curve C in parametric form -/
noncomputable def curve_C (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, 1 + Real.sin α ^ 2)

/-- The line l in rectangular form -/
def line_l (x : ℝ) : ℝ := x

/-- The intersection point -/
def intersection_point : ℝ × ℝ := (1, 1)

/-- Theorem stating that the intersection_point lies on both the curve C and the line l -/
theorem intersection_point_on_curve_and_line :
  ∃ α : ℝ,
    curve_C α = intersection_point ∧
    line_l (intersection_point.fst) = intersection_point.snd := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_curve_and_line_l887_88708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l887_88791

-- Define the inverse proportion function
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

-- Define the set of points in the first and third quadrants
def first_and_third_quadrants : Set (ℝ × ℝ) :=
  {p | (p.1 > 0 ∧ p.2 > 0) ∨ (p.1 < 0 ∧ p.2 < 0)}

-- Theorem statement
theorem inverse_proportion_quadrants (k : ℝ) :
  (∃ x, inverse_proportion k x = -1 ∧ x = -2) →
  ∀ x ≠ 0, (x, inverse_proportion k x) ∈ first_and_third_quadrants := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l887_88791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l887_88710

noncomputable def f (x : ℝ) : ℝ := 1 - 3 * Real.sin (x + Real.pi / 4) ^ 2

theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l887_88710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l887_88770

-- Define the solution set type
def SolutionSet (a : ℝ) : Set ℝ :=
  if a > 1 ∨ a < 0 then
    Set.Iic a ∪ Set.Ici (a^2)
  else if a = 1 ∨ a = 0 then
    Set.univ
  else if 0 < a ∧ a < 1 then
    Set.Iic (a^2) ∪ Set.Ici a
  else
    ∅

-- State the theorem
theorem inequality_solution_set (a : ℝ) :
  {x : ℝ | x^2 - (a^2 + a)*x + a^3 ≥ 0} = SolutionSet a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l887_88770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_four_axes_symmetry_l887_88759

-- Define a grid of cells
def Grid := List (List Bool)

-- Define a function to check if a figure has axes of symmetry
def hasAxesOfSymmetry (g : Grid) : Nat := sorry

-- Define a function to shade a cell in the grid
def shadeCell (g : Grid) (row col : Nat) : Grid := sorry

-- Theorem statement
theorem possible_four_axes_symmetry (F : Grid) :
  hasAxesOfSymmetry F = 0 →
  ∃ (row col : Nat), hasAxesOfSymmetry (shadeCell F row col) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_four_axes_symmetry_l887_88759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_largest_pandigital_multiples_of_693_l887_88780

def is_pandigital (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), 
    digits.length = 10 ∧ 
    digits.toFinset = Finset.range 10 ∧
    digits.foldl (λ acc d => acc * 10 + d) 0 = n

def is_valid_multiple (n : ℕ) : Prop :=
  is_pandigital n ∧ 
  n % 693 = 0 ∧
  n ≥ 1000000000

theorem smallest_largest_pandigital_multiples_of_693 :
  (∀ m : ℕ, is_valid_multiple m → m ≥ 1024375968) ∧
  (∀ m : ℕ, is_valid_multiple m → m ≤ 9876523041) ∧
  is_valid_multiple 1024375968 ∧
  is_valid_multiple 9876523041 :=
by
  sorry

#check smallest_largest_pandigital_multiples_of_693

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_largest_pandigital_multiples_of_693_l887_88780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l887_88773

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x / Real.sqrt (5 + 4 * Real.cos x)

-- State the theorem
theorem f_range :
  ∀ x ∈ Set.Icc 0 (2 * Real.pi),
    ∃ y ∈ Set.Icc (-1/2) (1/2), f x = y ∧
    ∀ z, f x = z → z ∈ Set.Icc (-1/2) (1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l887_88773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_is_right_angled_l887_88728

-- Define the points
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (3, 2)
def C : ℝ × ℝ := (-1, 4)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the lengths of the sides
noncomputable def AB : ℝ := distance A B
noncomputable def BC : ℝ := distance B C
noncomputable def AC : ℝ := distance A C

-- Theorem statement
theorem triangle_ABC_is_right_angled :
  AB^2 + AC^2 = BC^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_is_right_angled_l887_88728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l887_88754

/-- The volume of a regular triangular pyramid with a perpendicular of length p drawn from the base
to a lateral edge and a dihedral angle α between lateral faces. -/
noncomputable def pyramidVolume (p : ℝ) (α : ℝ) : ℝ :=
  (9 * p^3 * Real.tan (α/2)^3) / (4 * Real.sqrt (3 * Real.tan (α/2)^2 - 1))

/-- Theorem stating the volume of a regular triangular pyramid with given parameters. -/
theorem regular_triangular_pyramid_volume (p : ℝ) (α : ℝ) (h1 : p > 0) (h2 : α > 0) (h3 : α < π) :
  pyramidVolume p α = (9 * p^3 * Real.tan (α/2)^3) / (4 * Real.sqrt (3 * Real.tan (α/2)^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l887_88754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l887_88721

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x, g (-x) = -g x := by
  intro x
  -- Expand the definition of g
  simp [g, f]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l887_88721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l887_88790

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (r^n - 1) / (r - 1)

theorem geometric_series_sum :
  let a : ℝ := 2
  let r : ℝ := -2
  let n : ℕ := 10
  geometric_sum a r n = -682 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l887_88790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_example_l887_88732

theorem complex_magnitude_example : Complex.abs (-3 + (5/2) * Complex.I) = Real.sqrt 61 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_example_l887_88732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_equality_l887_88776

noncomputable def a (n : ℕ) : ℝ := Real.sin (2^n * Real.pi / 30) ^ 2

theorem smallest_n_for_equality : 
  (∃ (n : ℕ), n > 0 ∧ a n = a 0) ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < 4 → a m ≠ a 0) ∧
  a 4 = a 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_equality_l887_88776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_distance_l887_88743

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9

-- Define point P
def P : ℝ × ℝ := (1, 1)

-- Define the line l passing through P
noncomputable def l (θ : ℝ) (t : ℝ) : ℝ × ℝ := (P.1 + t * Real.cos θ, P.2 + t * Real.sin θ)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem curve_intersection_distance :
  ∀ θ t₁ t₂,
  C (l θ t₁).1 (l θ t₁).2 →
  C (l θ t₂).1 (l θ t₂).2 →
  distance P (l θ t₁) = 2 * distance P (l θ t₂) →
  distance (l θ t₁) (l θ t₂) = 3 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_distance_l887_88743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_triangle_area_l887_88775

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2 = 1

-- Define the asymptotes of the hyperbola
def asymptote (x y : ℝ) : Prop := x = 2*y ∨ x = -2*y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1)*(p3.2 - p1.2) - (p3.1 - p1.1)*(p2.2 - p1.2))

theorem parabola_hyperbola_triangle_area 
  (P : ℝ × ℝ) 
  (h1 : asymptote P.1 P.2)
  (h2 : Real.sqrt 2 * distance origin focus = distance P focus) :
  triangleArea P focus origin = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_triangle_area_l887_88775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_solution_exists_l887_88745

theorem zero_solution_exists : ∃ (a b c d : ℕ), 
  ((4:ℤ)^a + (4:ℤ)^b = (9:ℤ)^c + (9:ℤ)^d) ∧ 
  (a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_solution_exists_l887_88745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l887_88735

/-- The area of the shaded region between two concentric circles -/
noncomputable def shaded_area (r : ℝ) : ℝ := Real.pi * (25 * r^2 - r^2)

theorem shaded_area_calculation :
  let r : ℝ := 3
  shaded_area r = 216 * Real.pi := by
  -- Unfold the definition of shaded_area
  unfold shaded_area
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l887_88735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l887_88738

def b : ℕ → ℕ
  | 0 => 5  -- Adding the base case for 0
  | 1 => 5
  | n + 1 => b n + 3 * n

theorem b_50_value : b 50 = 3680 := by
  -- The proof will be skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l887_88738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l887_88796

noncomputable section

-- Define the curve C
def curve_C (α : Real) : Real × Real :=
  (2 + Real.sqrt 5 * Real.cos α, 1 + Real.sqrt 5 * Real.sin α)

-- Define the polar equation of line l
def line_l (θ : Real) (ρ : Real) : Prop :=
  ρ * (Real.sin θ + Real.cos θ) = 1

-- State the theorem
theorem curve_C_properties :
  -- 1. The polar equation of curve C
  ∃ (f : Real → Real), ∀ θ, f θ = 4 * Real.cos θ + 2 * Real.sin θ ∧
    (∀ α, ∃ ρ θ, curve_C α = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ ρ = f θ) ∧
  -- 2. The length of the chord intercepted by line l on curve C
  ∃ (chord_length : Real), chord_length = 2 * Real.sqrt 3 ∧
    ∃ (p q : Real × Real),
      (∃ α₁, curve_C α₁ = p) ∧
      (∃ α₂, curve_C α₂ = q) ∧
      (∃ θ₁ ρ₁, line_l θ₁ ρ₁ ∧ p = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁)) ∧
      (∃ θ₂ ρ₂, line_l θ₂ ρ₂ ∧ q = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂)) ∧
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l887_88796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l887_88724

-- Define the curves
noncomputable def curve1 (x : ℝ) : ℝ := x^2
noncomputable def curve2 (x : ℝ) : ℝ := Real.sqrt x

-- Define the area of the enclosed figure
noncomputable def enclosedArea : ℝ := ∫ x in Set.Icc 0 1, curve2 x - curve1 x

-- Theorem statement
theorem area_enclosed_by_curves : enclosedArea = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l887_88724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_inequality_l887_88766

theorem cos_inequality : Real.cos (-2/5 * Real.pi) < Real.cos (-1/4 * Real.pi) := by
  have h1 : -Real.pi < -2/5 * Real.pi := by sorry
  have h2 : -2/5 * Real.pi < -1/4 * Real.pi := by sorry
  have h3 : -1/4 * Real.pi < 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_inequality_l887_88766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l887_88725

noncomputable def initial_height : ℝ := 800
noncomputable def bounce_ratio : ℝ := 2/3
noncomputable def height_threshold : ℝ := 2

noncomputable def height_after_bounces (k : ℕ) : ℝ := initial_height * (bounce_ratio ^ k)

theorem ball_bounce_count :
  ∃ k : ℕ, (∀ n < k, height_after_bounces n ≥ height_threshold) ∧
           (height_after_bounces k < height_threshold) ∧
           k = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l887_88725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l887_88703

theorem triangle_cosine_inequality (A B C : ℝ) (hAcute : 0 < A ∧ A < π/2) 
  (hBcute : 0 < B ∧ B < π/2) (hCcute : 0 < C ∧ C < π/2) (hSum : A + B + C = π) :
  (Real.cos A / (Real.cos B * Real.cos C) + Real.cos B / (Real.cos A * Real.cos C) + Real.cos C / (Real.cos A * Real.cos B)) ≥ 
  3 * (1 / (1 + Real.cos A) + 1 / (1 + Real.cos B) + 1 / (1 + Real.cos C)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l887_88703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_30cm_forms_valid_triangle_l887_88798

/-- Triangle inequality theorem checker -/
def satisfiesTriangleInequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a length can form a valid triangle with given sides -/
def canFormTriangle (side1 side2 side3 : ℝ) : Prop :=
  satisfiesTriangleInequality side1 side2 side3

/-- Given lengths -/
def givenLength1 : ℝ := 30
def givenLength2 : ℝ := 50

/-- Possible third lengths -/
def possibleLengths : List ℝ := [20, 30, 80, 90]

/-- Theorem: Only 30cm among the possible lengths can form a valid triangle -/
theorem only_30cm_forms_valid_triangle :
  ∃! x, x ∈ possibleLengths ∧ canFormTriangle givenLength1 givenLength2 x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_30cm_forms_valid_triangle_l887_88798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_hundred_digit_numbers_l887_88731

/-- A 100-digit natural number. -/
def HundredDigitNumber : Type := { n : ℕ // n ≥ 10^99 ∧ n < 10^100 }

/-- Function that replaces a digit with zero in a number. -/
def replaceDigitWithZero (n : HundredDigitNumber) (pos : Fin 100) : ℕ :=
  sorry

/-- Theorem stating the condition for the special 100-digit numbers. -/
theorem special_hundred_digit_numbers (N : HundredDigitNumber) :
  (∃ (pos : Fin 100), (replaceDigitWithZero N pos : ℕ) = (N.val / 13)) →
  ∃ (b : Fin 4), b.val ≠ 0 ∧ N.val = 325 * b.val * 10^97 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_hundred_digit_numbers_l887_88731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l887_88757

-- Define the constants
noncomputable def a : ℝ := 2^(1/5 : ℝ)
noncomputable def b : ℝ := Real.log 2
noncomputable def c : ℝ := Real.log 2 / Real.log 0.3

-- State the theorem
theorem abc_relationship : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l887_88757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_value_is_correct_l887_88752

/-- Represents the types of coins --/
inductive CoinType
  | Dollar
  | HalfDollar
  | Quarter
  | Dime
  | Nickel
  | Penny

/-- The number of rolls for each coin type --/
def numRolls : CoinType → Nat
  | CoinType.Dollar => 6
  | CoinType.HalfDollar => 5
  | CoinType.Quarter => 7
  | CoinType.Dime => 4
  | CoinType.Nickel => 3
  | CoinType.Penny => 2

/-- The number of coins in each roll for each coin type --/
def coinsPerRoll : CoinType → Nat
  | CoinType.Dollar => 20
  | CoinType.HalfDollar => 25
  | CoinType.Quarter => 40
  | CoinType.Dime => 50
  | CoinType.Nickel => 40
  | CoinType.Penny => 50

/-- The value of each coin in dollars --/
def coinValue : CoinType → Rat
  | CoinType.Dollar => 1
  | CoinType.HalfDollar => 1/2
  | CoinType.Quarter => 1/4
  | CoinType.Dime => 1/10
  | CoinType.Nickel => 1/20
  | CoinType.Penny => 1/100

/-- The total value of all coins --/
def totalValue : Rat :=
  List.sum (List.map
    (fun ct => (numRolls ct : Rat) * (coinsPerRoll ct : Rat) * coinValue ct)
    [CoinType.Dollar, CoinType.HalfDollar, CoinType.Quarter, CoinType.Dime, CoinType.Nickel, CoinType.Penny])

theorem total_value_is_correct : totalValue = 279.5 := by
  sorry

#eval totalValue

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_value_is_correct_l887_88752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l887_88740

theorem problem_solution :
  ∀ a b c : ℤ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  a ≥ b ∧ b ≥ c →
  a^2 - b^2 - c^2 + a*b = 2011 →
  a^2 + 3*b^2 + 3*c^2 - 3*a*b - 2*a*c - 2*b*c = -1997 →
  a = 253 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l887_88740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_diagonal_l887_88706

/-- The distance from point (-C, 0) to the line y = x is C²/2 --/
theorem distance_to_diagonal (C : ℝ) : 
  let P : ℝ × ℝ := (-C, 0)
  let line (x : ℝ) : ℝ := x
  let dist := Real.sqrt ((line (-C) - 0)^2 + (C - (-C))^2) / Real.sqrt 2
  dist^2 = C^2 / 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_diagonal_l887_88706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l887_88711

noncomputable def f (x : ℝ) : ℝ := 
  (Real.log (-x^2 + 2*x + 3)) / Real.sqrt (1 - x) + x^0

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l887_88711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diffeq_l887_88748

/-- The differential equation y'' + 2y' + 5y = e^(-x) cos(2x) -/
def DiffEq (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv y)) x + 2 * (deriv y) x + 5 * y x = Real.exp (-x) * Real.cos (2 * x)

/-- The general solution of the differential equation -/
noncomputable def GeneralSolution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  (C₁ * Real.cos (2 * x) + C₂ * Real.sin (2 * x)) * Real.exp (-x) + 
  (1 / 4) * x * Real.exp (-x) * Real.sin (2 * x)

/-- Theorem stating that the GeneralSolution satisfies the DiffEq -/
theorem general_solution_satisfies_diffeq (C₁ C₂ : ℝ) : 
  DiffEq (GeneralSolution C₁ C₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_satisfies_diffeq_l887_88748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_plaque_weight_l887_88787

/-- Represents a triangular plaque with a circular cut-out -/
structure Plaque where
  side : ℝ
  cut_out_diameter : ℝ

/-- Calculates the weight of a plaque given its parameters -/
noncomputable def plaque_weight (p : Plaque) (reference_side : ℝ) (reference_weight : ℝ) : ℝ :=
  let triangle_area (s : ℝ) := s^2 * Real.sqrt 3 / 4
  let cut_out_area := Real.pi * (p.cut_out_diameter / 2)^2
  let plaque_area := triangle_area p.side - cut_out_area
  (plaque_area * reference_weight) / (triangle_area reference_side)

theorem second_plaque_weight :
  let first_plaque_side := (6 : ℝ)
  let first_plaque_weight := (24 : ℝ)
  let second_plaque := Plaque.mk 8 (4 * Real.sqrt 3)
  let second_plaque_weight := plaque_weight second_plaque first_plaque_side first_plaque_weight
  ∃ ε > 0, abs (second_plaque_weight - 27) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_plaque_weight_l887_88787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_cone_example_l887_88768

/-- The lateral surface area of a cone with base radius r and slant height l is π * r * l -/
theorem cone_lateral_surface_area (r l : ℝ) (hr : r > 0) (hl : l > 0) :
  π * r * l = (1 / 2) * (2 * π * r) * l := by
  ring

/-- Given a cone with base radius 3cm and slant height 5cm, its lateral surface area is 15π cm² -/
theorem cone_example :
  let r : ℝ := 3
  let l : ℝ := 5
  (1 / 2) * (2 * π * r) * l = 15 * π := by
  simp
  ring

#check cone_lateral_surface_area
#check cone_example

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_cone_example_l887_88768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_arithmetic_sequence_l887_88700

-- Define a polynomial type
def MyPolynomial (α : Type*) [Semiring α] := ℕ → α

-- Define the degree of a polynomial
noncomputable def degree {α : Type*} [Semiring α] (P : MyPolynomial α) : ℕ := sorry

-- Define the k-th derivative of a polynomial
noncomputable def derivative {α : Type*} [Semiring α] (P : MyPolynomial α) (k : ℕ) : MyPolynomial α := sorry

-- Define the number of roots of a polynomial
noncomputable def num_roots {α : Type*} [Field α] (P : MyPolynomial α) : ℕ := sorry

-- Define the sum of roots of a polynomial
noncomputable def sum_roots {α : Type*} [Field α] (P : MyPolynomial α) : α := sorry

-- Main theorem
theorem root_sum_arithmetic_sequence 
  {α : Type*} [Field α]
  (P : MyPolynomial α) 
  (n : ℕ) 
  (h1 : degree P = n)
  (h2 : ∀ k : ℕ, k ≤ n → num_roots (derivative P k) = n - k) :
  ∃ (a d : α), ∀ k : ℕ, k ≤ n → sum_roots (derivative P k) = a + k * d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_arithmetic_sequence_l887_88700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l887_88769

-- Define the power function as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_properties :
  ∃ α : ℝ, 
    (f α 3 = Real.sqrt 3) ∧ 
    (¬ ∀ x, f α x = f α (-x)) ∧ 
    (¬ ∀ x, f α (-x) = -(f α x)) ∧
    (∀ x y, 0 < x ∧ x < y → f α x < f α y) := by
  -- Provide the value of α
  use 1/2
  
  -- Split the goal into four parts
  constructor
  · -- Prove f α 3 = Real.sqrt 3
    sorry
  constructor
  · -- Prove ¬ ∀ x, f α x = f α (-x)
    sorry
  constructor
  · -- Prove ¬ ∀ x, f α (-x) = -(f α x)
    sorry
  · -- Prove ∀ x y, 0 < x ∧ x < y → f α x < f α y
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l887_88769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_fraction_outside_circle_l887_88795

/-- An isosceles triangle with a specific base and height, and an inscribed circle. -/
structure IsoscelesTriangleWithCircle where
  -- Base of the triangle
  base : ℝ
  -- Height of the triangle
  height : ℝ
  -- The triangle is isosceles
  isIsosceles : True
  -- The base is 10 units
  baseIs10 : base = 10
  -- The height is 12 units
  heightIs12 : height = 12
  -- There is an inscribed circle touching all three sides
  hasInscribedCircle : True

/-- The fraction of the triangle's area outside the inscribed circle. -/
noncomputable def areaFractionOutsideCircle (t : IsoscelesTriangleWithCircle) : ℝ :=
  1 - (5 * Real.pi / 27)

/-- Theorem stating that the fraction of the area outside the circle is 1 - (5π/27). -/
theorem area_fraction_outside_circle (t : IsoscelesTriangleWithCircle) :
  areaFractionOutsideCircle t = 1 - (5 * Real.pi / 27) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_fraction_outside_circle_l887_88795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l887_88765

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 - 1) / (x^2 + 2)

-- State the theorem about the range of the function
theorem range_of_f :
  Set.range f = Set.Icc (-1/2 : ℝ) 3 \ {3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l887_88765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l887_88702

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def axis_of_symmetry (a b : ℝ) : ℝ := -b / (2 * a)

theorem quadratic_properties (a b c : ℝ) (h1 : a < 0) (h2 : axis_of_symmetry a b = 2) :
  (-4 * a * c < 0) ∧ 
  (∀ k : ℝ, k ≠ -1) ∧
  (b = -4 * a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l887_88702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_residue_mod_2011_l887_88788

def T : ℤ := (List.range 1005).foldl (λ acc i => acc + (2*i + 1) - (2*i + 3)) 0

theorem T_residue_mod_2011 : T ≡ 1009 [ZMOD 2011] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_residue_mod_2011_l887_88788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_l887_88782

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then -x - 4 else x^2 - 5

theorem f_solution : 
  {a : ℝ | f a - 11 = 0} = {-15, 4} :=
by
  -- The proof goes here
  sorry

#check f_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_l887_88782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_ratio_is_four_l887_88712

/-- Represents the ratio of Mindy's income to Mork's income -/
noncomputable def income_ratio (mork_tax_rate mindy_tax_rate combined_tax_rate : ℝ) : ℝ :=
  (combined_tax_rate - mork_tax_rate) / (mindy_tax_rate - combined_tax_rate)

/-- Theorem stating that given the tax rates, the income ratio is 4 -/
theorem income_ratio_is_four :
  income_ratio 0.40 0.25 0.28 = 4 := by
  -- Unfold the definition of income_ratio
  unfold income_ratio
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_ratio_is_four_l887_88712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_squared_range_l887_88729

theorem y_squared_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 4) ^ (1/3) = 4) :
  15 < y^2 ∧ y^2 < 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_squared_range_l887_88729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_height_at_10_l887_88779

/-- Represents a parabolic arch -/
structure ParabolicArch where
  height : ℝ
  span : ℝ

/-- Calculates the height of a parabolic arch at a given distance from the center -/
noncomputable def archHeight (arch : ParabolicArch) (x : ℝ) : ℝ :=
  -((4 * arch.height) / (arch.span ^ 2)) * x^2 + arch.height

/-- Theorem: For a parabolic arch with height 20 inches and span 50 inches,
    the height at 10 inches from the center is 16.8 inches -/
theorem parabolic_arch_height_at_10 :
  let arch : ParabolicArch := { height := 20, span := 50 }
  archHeight arch 10 = 16.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_height_at_10_l887_88779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debate_competition_results_l887_88772

/-- The number of students from School A -/
def school_a_students : ℕ := 3

/-- The number of boys from School A -/
def school_a_boys : ℕ := 2

/-- The number of girls from School A -/
def school_a_girls : ℕ := 1

/-- The number of students from School B -/
def school_b_students : ℕ := 5

/-- The number of boys from School B -/
def school_b_boys : ℕ := 3

/-- The number of girls from School B -/
def school_b_girls : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := school_a_students + school_b_students

/-- The total number of boys -/
def total_boys : ℕ := school_a_boys + school_b_boys

/-- The total number of girls -/
def total_girls : ℕ := school_a_girls + school_b_girls

theorem debate_competition_results :
  (∃ (arrangements : ℕ), arrangements = 14400 ∧ 
    arrangements = (total_students - 2) * (total_students - 1) * Nat.factorial (total_students - 2)) ∧
  (∃ (prob_at_least_one_girl : ℚ), prob_at_least_one_girl = 13/14 ∧
    prob_at_least_one_girl = 1 - (Nat.choose total_boys 4 / Nat.choose total_students 4)) ∧
  (∃ (prob_two_boys_same_school : ℚ), prob_two_boys_same_school = 6/35 ∧
    prob_two_boys_same_school = (Nat.choose school_a_boys 2 * Nat.choose total_girls 2 + 
      Nat.choose school_b_boys 2 * Nat.choose total_girls 2) / Nat.choose total_students 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debate_competition_results_l887_88772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l887_88753

/-- Triangle ABC with given properties -/
structure TriangleABC where
  /-- Cosine of angle A -/
  cos_A : ℝ
  /-- Angle C is twice angle A -/
  C_eq_2A : ℝ
  /-- Property that cos A = 3/4 -/
  cos_A_eq : cos_A = 3/4
  /-- Property that C = 2A -/
  C_eq_2A_prop : C_eq_2A = 2

/-- Theorem about sin B and area of triangle ABC -/
theorem triangle_ABC_properties (t : TriangleABC) :
  ∃ (sin_B : ℝ), sin_B = 5 * Real.sqrt 7 / 16 ∧
  ∃ (a b S : ℝ), a = 4 → S = 15 * Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l887_88753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_years_of_investment_l887_88701

/-- Given a principal amount and the difference in interest for a higher rate,
    calculate the number of years the sum was invested at simple interest. -/
noncomputable def calculateYears (principal : ℝ) (interestDifference : ℝ) : ℝ :=
  (100 * interestDifference) / (5 * principal)

/-- Theorem stating that for a principal of 200 and an interest difference of 100,
    the number of years is 100. -/
theorem years_of_investment (principal : ℝ) (interestDifference : ℝ)
    (h1 : principal = 200)
    (h2 : interestDifference = 100) :
    calculateYears principal interestDifference = 100 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculateYears 200 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_years_of_investment_l887_88701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circle_area_eq_pi_over_four_l887_88716

/-- The area of a quarter of a unit circle with center at (1, 0) and radius 1 -/
noncomputable def quarter_circle_area : ℝ := ∫ x in (0:ℝ)..(1:ℝ), Real.sqrt (1 - (x - 1)^2)

/-- The area of a quarter of a unit circle with center at (1, 0) and radius 1 is equal to π/4 -/
theorem quarter_circle_area_eq_pi_over_four : quarter_circle_area = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circle_area_eq_pi_over_four_l887_88716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_equations_l887_88733

-- Define the eccentricity and focal distance
noncomputable def eccentricity : ℝ := 4/5
noncomputable def focal_distance : ℝ := 2 * Real.sqrt 34

-- Define the semi-major and semi-minor axes
def a : ℝ := 5
def b : ℝ := 3

-- Define the ellipse and hyperbola equations
def ellipse_equation (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1
def hyperbola_equation (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

theorem ellipse_hyperbola_equations :
  (∀ x y, ellipse_equation x y ↔ x^2/25 + y^2/9 = 1) ∧
  (∀ x y, hyperbola_equation x y ↔ x^2/25 - y^2/9 = 1) := by
  sorry

#check ellipse_hyperbola_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_equations_l887_88733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_sauce_lasts_21_days_l887_88763

/-- Calculates the number of full days a jar of hot sauce will last -/
def hot_sauce_duration (jar_volume : ℚ) (serving_size : ℚ) (daily_servings : ℕ) : ℕ :=
  (jar_volume / (serving_size * daily_servings : ℚ)).floor.toNat

/-- Proves that a jar of hot sauce with given parameters lasts 21 days -/
theorem hot_sauce_lasts_21_days :
  hot_sauce_duration 950 15 3 = 21 := by
  sorry

#eval hot_sauce_duration 950 15 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_sauce_lasts_21_days_l887_88763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equality_iff_angle_equality_l887_88777

/-- In a triangle ABC, sinA = sinB is equivalent to A = B -/
theorem sin_equality_iff_angle_equality (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) :
  Real.sin A = Real.sin B ↔ A = B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equality_iff_angle_equality_l887_88777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l887_88762

/-- An arithmetic sequence with first term 0 and non-zero common difference -/
structure ArithmeticSequence where
  d : ℚ
  hd : d ≠ 0

variable (a : ArithmeticSequence)

/-- The nth term of the arithmetic sequence -/
def a_n (n : ℕ) : ℚ := (n - 1 : ℚ) * a.d

/-- The sum of the first n terms of the arithmetic sequence -/
def S_n (n : ℕ) : ℚ := (n : ℚ) * (a_n a 1 + a_n a n) / 2

/-- Theorem: If a_k equals the sum of the first 7 terms, then k = 22 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) : 
  ∃ k : ℕ, a_n a k = S_n a 7 ∧ k = 22 := by
  use 22
  constructor
  · sorry -- Proof that a_n a 22 = S_n a 7
  · rfl

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l887_88762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_k_l887_88778

/-- A function y = k/x where k > 0 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k / x

/-- The theorem stating that if f has a minimum value of 5 on [2,4], then k = 20 -/
theorem minimum_value_implies_k (k : ℝ) :
  k > 0 →
  (∀ x ∈ Set.Icc 2 4, f k x ≥ 5) →
  (∃ x ∈ Set.Icc 2 4, f k x = 5) →
  k = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_k_l887_88778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_zero_count_condition_l887_88742

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

-- Theorem for the tangent line
theorem tangent_line_at_zero :
  ∃ (m : ℝ), m = 2 ∧ 
  (∀ x, m * x = (f 1 x - f 1 0) / (x - 0)) := by sorry

-- Theorem for the range of a
theorem zero_count_condition (a : ℝ) :
  ((∃! x, x ∈ Set.Ioo (-1 : ℝ) 0 ∧ f a x = 0) ∧
   (∃! x, x ∈ Set.Ioi 0 ∧ f a x = 0)) ↔
  a < -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_zero_count_condition_l887_88742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_isosceles_trapezoid_l887_88789

/-- The area of an isosceles trapezoid with given dimensions -/
noncomputable def isosceles_trapezoid_area (leg : ℝ) (base1 : ℝ) (base2 : ℝ) : ℝ :=
  let height := Real.sqrt (leg^2 - ((base2 - base1) / 2)^2)
  (base1 + base2) * height / 2

/-- Theorem stating that the area of the specified isosceles trapezoid is 40 -/
theorem area_of_specific_isosceles_trapezoid :
  isosceles_trapezoid_area 5 7 13 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_isosceles_trapezoid_l887_88789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_constant_term_l887_88726

/-- A polynomial of degree 4 that is a perfect square --/
def is_perfect_square (a b : ℚ) : Prop :=
  ∃ (p q : ℚ), ∀ (x : ℚ), x^4 + x^3 - x^2 + a*x + b = (x^2 + p*x + q)^2

/-- If x^4 + x^3 - x^2 + ax + b is a perfect square, then b = 25/64 --/
theorem perfect_square_constant_term (a b : ℚ) :
  is_perfect_square a b → b = 25/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_constant_term_l887_88726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l887_88761

/-- The time (in seconds) it takes for a train to pass a person moving in the opposite direction. -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) : ℝ :=
  train_length / ((train_speed + person_speed) * (1000 / 3600))

/-- Theorem stating that the time for a 385m train moving at 60 kmph to pass a person moving at 6 kmph
    in the opposite direction is approximately 21.005 seconds. -/
theorem train_passing_time_approx :
  let t := train_passing_time 385 60 6
  ∃ ε > 0, abs (t - 21.005) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l887_88761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coverage_ring_area_proof_l887_88792

/-- The number of radars --/
def n : ℕ := 9

/-- The radius of each radar's coverage circle in km --/
noncomputable def r : ℝ := 37

/-- The width of the coverage ring in km --/
noncomputable def w : ℝ := 24

/-- The central angle of the regular polygon formed by the radars --/
noncomputable def θ : ℝ := 2 * Real.pi / n

/-- The area of the coverage ring --/
noncomputable def coverage_ring_area : ℝ := 1680 * Real.pi / Real.tan (20 * Real.pi / 180)

theorem coverage_ring_area_proof :
  let OB := r * Real.sin (θ / 2) / Real.sin (Real.pi / n)
  let OD := OB - (r * Real.cos (θ / 2))
  let OC := OD - w / 2
  let OE := OD + w / 2
  Real.pi * (OE^2 - OC^2) = coverage_ring_area := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coverage_ring_area_proof_l887_88792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l887_88794

/-- The area of a trapezium with parallel sides of lengths a and b,
    and a distance h between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of lengths 20 cm and 18 cm,
    and a distance of 13 cm between them, is equal to 247 square centimeters. -/
theorem trapezium_area_example : trapeziumArea 20 18 13 = 247 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic expression
  simp [add_mul, mul_div_assoc]
  -- Check that the result is equal to 247
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l887_88794
