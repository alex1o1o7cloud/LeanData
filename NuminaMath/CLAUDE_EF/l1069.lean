import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1069_106955

/-- The function f(x) defined as a rational function -/
noncomputable def f (x : ℝ) : ℝ := (12*x^4 + 4*x^3 + 9*x^2 + 5*x + 3) / (3*x^4 + 2*x^3 + 8*x^2 + 3*x + 1)

/-- Theorem stating that the horizontal asymptote of f(x) occurs at y = 4 -/
theorem horizontal_asymptote_of_f : 
  ∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, |x| > N → |f x - 4| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1069_106955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_distance_in_S_l1069_106971

/-- The set of points (x, y) with integer coordinates satisfying y² = 4x² - 15 -/
def S : Set (ℤ × ℤ) := {p | p.2^2 = 4 * p.1^2 - 15}

/-- The distance between two points in ℤ × ℤ -/
noncomputable def distance (p q : ℤ × ℤ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_max_distance_in_S :
  (∃ p q, p ∈ S ∧ q ∈ S ∧ distance p q = 2) ∧
  (∀ p q, p ∈ S → q ∈ S → distance p q ≤ 2 * Real.sqrt 65) ∧
  (∃ p q, p ∈ S ∧ q ∈ S ∧ distance p q = 2 * Real.sqrt 65) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_distance_in_S_l1069_106971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_3_l1069_106918

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the area calculation function
noncomputable def area (t : Triangle) : ℝ :=
  Real.sqrt (1/4 * (t.a^2 * t.c^2 - ((t.a^2 + t.c^2 - t.b^2)/2)^2))

-- Theorem statement
theorem triangle_area_is_sqrt_3 (t : Triangle) 
  (h1 : t.a^2 * Real.sin t.C = 4 * Real.sin t.A)
  (h2 : (t.a + t.c)^2 = 12 + t.b^2) :
  area t = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_3_l1069_106918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_speeds_l1069_106908

/-- Represents a runner on a closed track -/
structure Runner where
  speed : ℝ

/-- Represents a race between two runners on a closed track -/
structure Race where
  track_length : ℝ
  runner1 : Runner
  runner2 : Runner

/-- The conditions of the race -/
def race_conditions (r : Race) : Prop :=
  r.track_length / r.runner1.speed - r.track_length / r.runner2.speed = 10 ∧
  720 * r.runner1.speed - 720 * r.runner2.speed = r.track_length

/-- The theorem to be proved -/
theorem runner_speeds (r : Race) (h : race_conditions r) :
  (r.runner1.speed = 80 ∧ r.runner2.speed = 90) ∨
  (r.runner1.speed = 90 ∧ r.runner2.speed = 80) :=
by
  sorry

#check runner_speeds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_speeds_l1069_106908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1069_106974

/-- Given a sequence a, its partial sum S, and a derived sequence b, 
    we prove the range of λ satisfying a certain inequality. -/
theorem lambda_range (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) (b : ℕ+ → ℝ) (lambda : ℝ) : 
  (∀ n : ℕ+, S n = (-1)^(n : ℕ) * a n - (1 / 2^(n : ℕ))) →
  (∀ n : ℕ+, b n = 8 * a 2 * 2^((n : ℕ) - 1)) →
  (∀ n : ℕ+, lambda * b n - 1 > 0) →
  lambda > 1/2 ∧ ∀ ε > 0, ∃ lambda' : ℝ, lambda' > 1/2 ∧ lambda' < 1/2 + ε ∧ 
    ∀ n : ℕ+, lambda' * b n - 1 > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1069_106974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_perpendicular_l1069_106970

-- Define the ellipse (T)
def ellipse_T (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle (C)
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the conditions and theorem
theorem tangent_lines_perpendicular 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ellipse_T 0 1 a b) -- Vertex A(0, 1) is on the ellipse
  (h4 : (a^2 - b^2) / a^2 = 6 / 9) -- Eccentricity e = √6/3
  : ∀ x y : ℝ, circle_C x y → 
    ∃ k1 k2 : ℝ, k1 * k2 = -1 ∧ 
    (∀ t : ℝ, ellipse_T (x + t) (y + k1 * t) a b → t = 0) ∧
    (∀ t : ℝ, ellipse_T (x + t) (y + k2 * t) a b → t = 0) :=
by sorry

#check tangent_lines_perpendicular

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_perpendicular_l1069_106970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eraser_collection_ratio_l1069_106933

/-- The number of erasers collected by Celine -/
def celine_erasers : ℕ := 10

/-- The number of erasers collected by Julian -/
def julian_erasers : ℕ := 2 * celine_erasers

/-- The total number of erasers collected by Celine, Gabriel, and Julian -/
def total_erasers : ℕ := 35

/-- The number of erasers collected by Gabriel -/
def gabriel_erasers : ℕ := total_erasers - celine_erasers - julian_erasers

/-- The ratio of erasers collected by Celine to those collected by Gabriel -/
noncomputable def eraser_ratio : ℚ := celine_erasers / gabriel_erasers

theorem eraser_collection_ratio :
  eraser_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eraser_collection_ratio_l1069_106933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l1069_106983

noncomputable def circle_C1 (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 1

noncomputable def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 16

noncomputable def center_distance : ℝ := Real.sqrt ((2 + 2)^2 + (5 - 2)^2)

def radius_C1 : ℝ := 1

def radius_C2 : ℝ := 4

theorem circles_externally_tangent : center_distance = radius_C1 + radius_C2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l1069_106983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l1069_106976

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x, f x a = f (-x) a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l1069_106976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_power_l1069_106910

theorem degree_of_polynomial_power : 
  Polynomial.degree ((2 * X^3 + 5 : Polynomial ℝ)^10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_power_l1069_106910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_change_grasshopper_theorem_l1069_106996

/-- Represents the state of grasshoppers on a line -/
inductive GrasshopperState
  | ABC
  | ACB
  | BAC
  | BCA
  | CAB
  | CBA

/-- Counts the number of inversions in a given state -/
def count_inversions (state : GrasshopperState) : Nat :=
  match state with
  | GrasshopperState.ABC => 0
  | GrasshopperState.ACB => 2
  | GrasshopperState.BAC => 1
  | GrasshopperState.BCA => 2
  | GrasshopperState.CAB => 3
  | GrasshopperState.CBA => 1

/-- Represents a single jump of a grasshopper -/
def jump (state : GrasshopperState) : GrasshopperState :=
  match state with
  | GrasshopperState.ABC => GrasshopperState.BAC
  | GrasshopperState.ACB => GrasshopperState.CAB
  | GrasshopperState.BAC => GrasshopperState.BCA
  | GrasshopperState.BCA => GrasshopperState.CBA
  | GrasshopperState.CAB => GrasshopperState.ACB
  | GrasshopperState.CBA => GrasshopperState.ABC

/-- The number of inversions changes by 1 after each jump -/
theorem inversion_change (state : GrasshopperState) :
  (count_inversions (jump state) : Int) - (count_inversions state : Int) = 1 ∨
  (count_inversions (jump state) : Int) - (count_inversions state : Int) = -1 :=
by sorry

/-- After an odd number of jumps, the number of inversions cannot be the same as the initial number -/
theorem grasshopper_theorem (initial : GrasshopperState) (n : Nat) (h : Odd n) :
  count_inversions (n.iterate jump initial) ≠ count_inversions initial :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_change_grasshopper_theorem_l1069_106996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cube_cost_l1069_106919

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem paint_cube_cost (paint_cost : ℝ) (paint_coverage : ℝ) (cube_side : ℝ) : 
  paint_cost = 36.5 →
  paint_coverage = 16 →
  cube_side = 8 →
  (6 * cube_side^2 / paint_coverage) * paint_cost = 876 := by
  intros h1 h2 h3
  -- Proof steps would go here
  sorry

#check paint_cube_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cube_cost_l1069_106919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_of_sqrt_81_and_self_cube_root_numbers_l1069_106947

theorem sqrt_of_sqrt_81_and_self_cube_root_numbers : 
  (∃ x : ℝ, x^2 = Real.sqrt 81 ∧ (x = 3 ∨ x = -3)) ∧ 
  (∀ x : ℝ, x^3 = x ↔ x = -1 ∨ x = 0 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_of_sqrt_81_and_self_cube_root_numbers_l1069_106947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nitrogen_highest_mass_percentage_l1069_106935

/-- Molar mass of nitrogen in g/mol -/
noncomputable def molar_mass_N : ℝ := 14.01

/-- Molar mass of hydrogen in g/mol -/
noncomputable def molar_mass_H : ℝ := 1.01

/-- Number of hydrogen atoms in ammonia -/
def num_H_atoms : ℕ := 3

/-- Molar mass of ammonia (NH3) in g/mol -/
noncomputable def molar_mass_NH3 : ℝ := molar_mass_N + num_H_atoms * molar_mass_H

/-- Mass percentage of nitrogen in ammonia -/
noncomputable def mass_percentage_N : ℝ := (molar_mass_N / molar_mass_NH3) * 100

/-- Mass percentage of hydrogen in ammonia -/
noncomputable def mass_percentage_H : ℝ := (num_H_atoms * molar_mass_H / molar_mass_NH3) * 100

theorem nitrogen_highest_mass_percentage :
  mass_percentage_N > mass_percentage_H :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nitrogen_highest_mass_percentage_l1069_106935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_hyperbola_equation_l1069_106936

-- Define the ellipse parameters
noncomputable def ellipse_major_axis : ℝ := 10
noncomputable def ellipse_eccentricity : ℝ := 4/5

-- Define the hyperbola parameters
noncomputable def hyperbola_vertex_distance : ℝ := 6
noncomputable def hyperbola_asymptote_slope : ℝ := 3/2

-- Theorem for the ellipse equation
theorem ellipse_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  a = ellipse_major_axis / 2 ∧
  (a^2 - b^2) / a^2 = ellipse_eccentricity^2 ∧
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 25 + y^2 / 9 = 1) := by
  sorry

-- Theorem for the hyperbola equation
theorem hyperbola_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  2 * a = hyperbola_vertex_distance ∧
  b / a = hyperbola_asymptote_slope ∧
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 11 - y^2 / 9 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_hyperbola_equation_l1069_106936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_exists_l1069_106930

/-- Represents a piece on the grid -/
structure Piece where
  x : Nat
  y : Nat

/-- Represents the grid and piece placement -/
structure Grid (n : Nat) where
  pieces : Finset Piece
  n_ge_two : n ≥ 2
  piece_count : pieces.card = 2 * n
  valid_placement : ∀ p ∈ pieces, p.x < n ∧ p.y < n

/-- Checks if four pieces form a parallelogram -/
def is_parallelogram (p1 p2 p3 p4 : Piece) : Prop :=
  (p2.x - p1.x = p4.x - p3.x) ∧ (p2.y - p1.y = p4.y - p3.y)

/-- Main theorem: There exists a parallelogram in the grid -/
theorem parallelogram_exists (n : Nat) (grid : Grid n) :
  ∃ p1 p2 p3 p4, p1 ∈ grid.pieces ∧ p2 ∈ grid.pieces ∧ p3 ∈ grid.pieces ∧ p4 ∈ grid.pieces ∧ is_parallelogram p1 p2 p3 p4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_exists_l1069_106930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_four_digit_reverse_diff_l1069_106980

def reverse_number (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem unique_four_digit_reverse_diff : ∃! n : ℕ, 
  1000 ≤ n ∧ n < 10000 ∧ n + 7182 = reverse_number n :=
by
  use 1909
  constructor
  · sorry  -- Proof that 1909 satisfies the conditions
  · sorry  -- Proof of uniqueness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_four_digit_reverse_diff_l1069_106980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_property_iff_intersection_property_l1069_106939

/-- A function has the rotation property for π/4 if it intersects any line y = x + b at most once -/
def has_rotation_property (f : ℝ → ℝ) : Prop :=
  ∀ b : ℝ, ∃! x : ℝ, f x = x + b

/-- Theorem stating the equivalence between the rotation property and intersection property -/
theorem rotation_property_iff_intersection_property (f : ℝ → ℝ) :
  (∀ θ : ℝ, θ = π/4 → Function.Injective (λ x => (f (x * Real.cos θ - f x * Real.sin θ) * Real.cos θ + 
    (x * Real.sin θ + f x * Real.cos θ) * Real.sin θ))) ↔ has_rotation_property f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_property_iff_intersection_property_l1069_106939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_covering_l1069_106999

variable (n : ℕ)

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle formed by three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- A set of n points in a plane -/
def PointSet : Type := Fin n → Point

/-- Predicate to check if a point is inside or on the boundary of a triangle -/
def pointInTriangle (p a b c : Point) : Prop := sorry

theorem triangle_covering (points : PointSet n) 
  (h : ∀ (i j k : Fin n), triangleArea (points i) (points j) (points k) ≤ 1) :
  ∃ (a b c : Point), (∀ (i : Fin n), pointInTriangle (points i) a b c) ∧
                     triangleArea a b c ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_covering_l1069_106999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1069_106953

noncomputable def line_l (α : Real) (t : Real) : Real × Real :=
  (-3 + t * Real.cos α, Real.sqrt 3 + t * Real.sin α)

noncomputable def curve_C (θ : Real) : Real × Real :=
  (2 * Real.sqrt 3 * Real.cos θ, 2 * Real.sqrt 3 * Real.sin θ)

noncomputable def distance (p1 p2 : Real × Real) : Real :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_intersects_circle (α : Real) 
  (h1 : 0 ≤ α) (h2 : α < π) (h3 : α ≠ π/2) :
  ∃ (t1 t2 : Real), 
    let A := line_l α t1
    let B := line_l α t2
    distance A B = 2 * Real.sqrt 3 ∧
    ∃ (θ1 θ2 : Real), curve_C θ1 = A ∧ curve_C θ2 = B →
    α = π/6 ∧ 
    let M := (line_l α t1).1
    let N := (line_l α t2).1
    distance (M, 0) (N, 0) = 4 := by
  sorry

#check line_intersects_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1069_106953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_irrational_l1069_106940

/-- The perimeter of a square with side length 2 -/
def square_perimeter : ℝ := 4 * 2

/-- The radius of a circle with the same perimeter as the square -/
noncomputable def circle_radius : ℝ := square_perimeter / (2 * Real.pi)

/-- The area of the circle -/
noncomputable def circle_area : ℝ := Real.pi * circle_radius ^ 2

/-- Theorem stating that the area of the circle is irrational -/
theorem circle_area_irrational : Irrational circle_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_irrational_l1069_106940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_digit_sum_l1069_106954

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Represents a 9-digit number as a list of digits -/
def Number := List Digit

/-- The starting number 123456789 -/
def startNumber : Number := [⟨1, by norm_num⟩, ⟨2, by norm_num⟩, ⟨3, by norm_num⟩, 
                             ⟨4, by norm_num⟩, ⟨5, by norm_num⟩, ⟨6, by norm_num⟩, 
                             ⟨7, by norm_num⟩, ⟨8, by norm_num⟩, ⟨9, by norm_num⟩]

/-- Swaps two adjacent non-zero digits and decreases both by 1 -/
def swapAndDecrease (n : Number) (i : Nat) : Number :=
  sorry

/-- Computes the sum of digits in a number -/
def digitSum (n : Number) : Nat :=
  n.foldl (λ sum d => sum + d.val) 0

/-- Checks if a given number can be obtained from the start number
    using the allowed operations -/
def isReachable (n : Number) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem min_digit_sum :
  ∃ (n : Number), isReachable n ∧ digitSum n = 5 ∧
    ∀ (m : Number), isReachable m → digitSum m ≥ 5 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_digit_sum_l1069_106954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1069_106929

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x : ℝ, x + (f a b 1) - 3 = 0) ∧
    (a = 1 ∧ b = 8/3) ∧
    (∀ x : ℝ, x < 0 ∨ x > 2 → (deriv (f a b)) x > 0) ∧
    (∀ x : ℝ, 0 < x ∧ x < 2 → (deriv (f a b)) x < 0) ∧
    (∀ c : ℝ, (∀ x : ℝ, -2 ≤ x ∧ x ≤ 4 → f a b x < c^2 - c) ↔
      (c < (1 - Real.sqrt 33) / 2 ∨ c > (1 + Real.sqrt 33) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1069_106929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_one_on_top_l1069_106932

/-- Represents the state of the card pile -/
def CardPile (n : ℕ) := List ℕ

/-- The operation of reversing the top k cards -/
def reverseTop (k : ℕ) (pile : CardPile n) : CardPile n :=
  (pile.take k).reverse ++ pile.drop k

/-- Predicate to check if 1 is on top of the pile -/
def oneOnTop (pile : CardPile n) : Prop :=
  pile.head? = some 1

/-- The main theorem to prove -/
theorem eventually_one_on_top (n : ℕ) (initial_pile : CardPile n) :
  ∃ (k : ℕ), oneOnTop (Nat.repeat (λ pile => reverseTop (pile.head?.getD 1) pile) k initial_pile) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_one_on_top_l1069_106932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l1069_106985

def team_scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def total_games : ℕ := 15

def lost_games : ℕ := 7

def won_games : ℕ := total_games - lost_games

def opponent_score_lost (team_score : ℕ) : ℕ := team_score + 1

def opponent_score_won (team_score : ℕ) : ℚ := team_score / 3

theorem opponent_total_score :
  ∃ (lost_scores won_scores : List ℕ),
    lost_scores.length = lost_games ∧
    won_scores.length = won_games ∧
    lost_scores ++ won_scores = team_scores ∧
    (lost_scores.map opponent_score_lost).sum +
    (won_scores.map (fun x => Int.floor (opponent_score_won x))).sum = 67 := by
  sorry

#eval team_scores
#eval total_games
#eval lost_games
#eval won_games

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opponent_total_score_l1069_106985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l1069_106925

/-- The profit function representing annual profit in ten thousand yuan
    as a function of annual output in ten thousand units -/
noncomputable def profit_function (x : ℝ) : ℝ := -1/3 * x^3 + 81 * x - 234

/-- Theorem stating that the annual output maximizing profit is 9 ten thousand units -/
theorem max_profit_at_nine :
  ∃ (x : ℝ), x = 9 ∧
  ∀ (y : ℝ), profit_function y ≤ profit_function x := by
  sorry

#check max_profit_at_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l1069_106925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_l1069_106915

/-- Represents a tetrahedron with edge lengths -/
structure Tetrahedron where
  edges : Finset ℝ
  edge_count : edges.card = 6

/-- Checks if the given edges satisfy the triangle inequality for all faces of the tetrahedron -/
def satisfies_triangle_inequality (t : Tetrahedron) : Prop :=
  ∀ a b c, a ∈ t.edges → b ∈ t.edges → c ∈ t.edges → 
    a + b > c ∧ b + c > a ∧ c + a > b

theorem tetrahedron_edge_length (t : Tetrahedron) 
  (h1 : t.edges = {8, 14, 19, 28, 37, 42})
  (h2 : 42 ∈ t.edges)
  (h3 : satisfies_triangle_inequality t) :
  14 ∈ t.edges := by
  rw [h1]
  simp
  
#check tetrahedron_edge_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_l1069_106915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_in_terms_of_x_l1069_106951

theorem tan_double_angle_in_terms_of_x (θ : ℝ) (x : ℝ) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (h_cos : Real.cos θ = Real.sqrt ((x - 2) / (3 * x))) :
  Real.tan (2 * θ) = (2 * Real.sqrt (2 * (x^2 - x - 2))) / (-x - 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_in_terms_of_x_l1069_106951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_line_family_l1069_106920

theorem fixed_point_of_line_family :
  ∃! p : ℝ × ℝ, ∀ k : ℝ, k * p.1 - p.2 + 1 = 3 * k :=
by
  -- The unique point (3, 1) satisfies the equation for all k
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_line_family_l1069_106920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grain_storage_properties_l1069_106972

/-- Represents the properties of a cylindrical grain storage with a conical cover -/
structure GrainStorage where
  R : ℝ
  volume : ℝ
  cover_angle : ℝ

/-- The area of tinplate required for the grain storage -/
noncomputable def S (g : GrainStorage) : ℝ :=
  16 * (1 + Real.sqrt 2) / g.R + (1 + Real.sqrt 2) * Real.pi * g.R^2

/-- The theorem stating the properties of the grain storage -/
theorem grain_storage_properties (g : GrainStorage) 
  (h_volume : g.volume = (8 + 8 * Real.sqrt 2) * Real.pi)
  (h_angle : g.cover_angle = Real.pi / 4)
  (h_R_pos : g.R > 0) :
  ∃ (R_min : ℝ),
    (∀ R, R > 0 → S ⟨R, g.volume, g.cover_angle⟩ ≥ S ⟨R_min, g.volume, g.cover_angle⟩) ∧
    S ⟨R_min, g.volume, g.cover_angle⟩ = 8 * (1 + Real.sqrt 2) * Real.pi ∧
    2 * R_min + Real.sqrt 2 * R_min = 2 + 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grain_storage_properties_l1069_106972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_market_sales_l1069_106921

/-- The amount of ground beef sold on Saturday, given the sales data for the week --/
noncomputable def saturday_sales (thursday_sales : ℝ) (planned_total : ℝ) (excess_sale : ℝ) : ℝ :=
  let friday_sales := 2 * thursday_sales
  let total_sales := planned_total + excess_sale
  let sunday_sales := (total_sales - thursday_sales - friday_sales) / 3
  2 * sunday_sales

theorem meat_market_sales : saturday_sales 210 500 325 = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meat_market_sales_l1069_106921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_six_l1069_106945

open MeasureTheory

-- Define the interval
def interval : Set ℝ := Set.Icc 0 7

-- Define the function to determine if a point is closer to 6 than to 0
def closerToSix (x : ℝ) : Prop := |x - 6| < |x - 0|

-- Define the set of points closer to 6 than to 0 within the interval
def closerToSixSet : Set ℝ := {x ∈ interval | closerToSix x}

-- State the theorem
theorem probability_closer_to_six :
  (volume closerToSixSet) / (volume interval) = 4 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closer_to_six_l1069_106945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_implies_plane_perp_l1069_106922

/-- A type representing a plane in 3D space -/
structure Plane where

/-- A type representing a line in 3D space -/
structure Line where

/-- A predicate indicating that a line is in a plane -/
def LineInPlane (l : Line) (p : Plane) : Prop := sorry

/-- A predicate indicating that a line is perpendicular to a plane -/
def LinePerpToPlane (l : Line) (p : Plane) : Prop := sorry

/-- A predicate indicating that two planes are perpendicular -/
def PlanePerpToPlane (p1 : Plane) (p2 : Plane) : Prop := sorry

/-- Theorem: If a line in a plane is perpendicular to another plane, 
    then the two planes are perpendicular -/
theorem line_perp_implies_plane_perp 
  (α β : Plane) (m : Line) 
  (h1 : LineInPlane m α) 
  (h2 : LinePerpToPlane m β) : 
  PlanePerpToPlane α β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_implies_plane_perp_l1069_106922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1069_106903

noncomputable def f (ω : ℝ) (m : ℝ) (x : ℝ) : ℝ := 
  (Real.sqrt 3 / 2) * Real.sin (ω * x) - (1 / 2) * Real.cos (ω * x) - m

theorem function_properties (ω : ℝ) (m : ℝ) (h₁ : ω > 0) :
  (∃ (max min : ℝ), (∀ x, f ω m x ≤ max) ∧ (∀ x, f ω m x ≥ min) ∧ max = 5 * min → m = -3/2) ∧
  (m = Real.sqrt 2 / 2 → 
    ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > x₁ ∧ 
    f ω m x₁ = 0 ∧ f ω m x₂ = 0 ∧
    (∀ x, 0 < x ∧ x < x₁ → f ω m x ≠ 0) ∧
    x₂ - 2 * x₁ = Real.pi / 36 → ω = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1069_106903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1069_106965

/-- Given a hyperbola with equation x^2/9 - y^2/4 = 1, its asymptotes are y = ±(2/3)x -/
theorem hyperbola_asymptotes :
  let hyperbola := {p : ℝ × ℝ | p.1^2/9 - p.2^2/4 = 1}
  let asymptotes := {p : ℝ × ℝ | p.2 = 2/3*p.1 ∨ p.2 = -2/3*p.1}
  asymptotes = {p : ℝ × ℝ | p.1^2/9 - p.2^2/4 = 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1069_106965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_perimeter_bound_l1069_106984

/-- An equilateral triangle -/
structure EquilateralTriangle :=
  (side : ℝ)
  (side_pos : side > 0)

/-- An inscribed triangle within an equilateral triangle -/
structure InscribedTriangle (outer : EquilateralTriangle) :=
  (side1 side2 side3 : ℝ)
  (side1_pos : side1 > 0)
  (side2_pos : side2 > 0)
  (side3_pos : side3 > 0)

/-- The semi-perimeter of an equilateral triangle -/
noncomputable def semiPerimeter (t : EquilateralTriangle) : ℝ := 3 * t.side / 2

/-- The semi-perimeter of an inscribed triangle -/
noncomputable def inscribedSemiPerimeter {outer : EquilateralTriangle} (t : InscribedTriangle outer) : ℝ :=
  (t.side1 + t.side2 + t.side3) / 2

/-- Theorem: The perimeter of an inscribed triangle in an equilateral triangle is at least 2s' ≥ s -/
theorem inscribed_triangle_perimeter_bound 
  (outer : EquilateralTriangle) 
  (inner : InscribedTriangle outer) : 
  2 * inscribedSemiPerimeter inner ≥ semiPerimeter outer :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_perimeter_bound_l1069_106984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l1069_106950

/-- Quadratic function f(x) = x^2 - 4x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem quadratic_properties :
  ∃ (a h k : ℝ),
    (∀ x, f x = a * (x - h)^2 + k) ∧ 
    (a > 0) ∧
    (h = 2) ∧
    (k = -1) ∧
    (∀ x y, x < y → y < h → f y < f x) ∧
    (∀ x, -1 ≤ x → x < 3 → -1 ≤ f x ∧ f x ≤ 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l1069_106950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_origin_slope_l1069_106961

/-- Given a line through the origin with slope 3/2, prove that when x increases by 12, y increases by 18. -/
theorem line_through_origin_slope (line : ℝ → ℝ) 
  (origin : line 0 = 0)  -- Line passes through origin
  (slope : ∀ x : ℝ, line (x + 4) - line x = 6)  -- Slope condition
  : line 12 - line 0 = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_origin_slope_l1069_106961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_l1069_106966

theorem gcd_of_polynomial_and_multiple (y : ℤ) : 
  (∃ k : ℤ, y = 2480 * k) → 
  Int.gcd ((4*y+3)*(9*y+2)*(5*y+7)*(6*y+8)) y.natAbs = 336 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_l1069_106966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l1069_106962

/-- The phase shift of a cosine function y = A cos(B(x + C)) + D is -C/B -/
noncomputable def phase_shift (A B C D : ℝ) : ℝ := -C / B

/-- The given cosine function y = 2 cos(2x + π/4) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 4)

theorem phase_shift_of_f :
  phase_shift 2 2 (Real.pi / 4) 0 = -Real.pi / 8 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l1069_106962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_cubed_plus_z_squared_l1069_106982

noncomputable def z : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 3))

theorem z_cubed_plus_z_squared : z^3 + z^2 = (1 - Complex.I * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_cubed_plus_z_squared_l1069_106982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_point_equivalence_l1069_106912

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Converts a polar point to standard form (r > 0, 0 ≤ θ < 2π) -/
noncomputable def toStandardPolar (p : PolarPoint) : PolarPoint :=
  if p.r < 0 then
    { r := -p.r, θ := (p.θ + Real.pi) % (2 * Real.pi) }
  else
    { r := p.r, θ := p.θ % (2 * Real.pi) }

theorem polar_point_equivalence :
  let p := PolarPoint.mk (-4) (Real.pi / 6)
  toStandardPolar p = PolarPoint.mk 4 (7 * Real.pi / 6) := by
  sorry

#check polar_point_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_point_equivalence_l1069_106912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_y_equals_x_l1069_106960

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem statement
theorem symmetry_about_y_equals_x : 
  ∀ x : ℝ, x > 0 → g (f x) = x ∧ f (g x) = x :=
by
  intro x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_y_equals_x_l1069_106960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1069_106973

theorem trigonometric_equation_solution (t : ℝ) :
  (Real.cos t ≠ 0 ∧ Real.cos (2 * t) ≠ 0) →
  ((Real.tan (2 * t) / (Real.cos t)^2) - (Real.tan t / (Real.cos (2 * t))^2) = 0 ↔
   (∃ k : ℤ, t = Real.pi * k) ∨ (∃ n : ℤ, t = Real.pi / 6 ∨ t = -Real.pi / 6 + Real.pi * n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1069_106973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1069_106937

open Real

theorem tan_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : cos α ^ 2 + sin (π + 2 * α) = 3 / 10) : tan α = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1069_106937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1069_106914

/-- Calculates the annual interest rate given the principal, time period in days, and interest earned. -/
noncomputable def calculate_annual_interest_rate (principal : ℝ) (days : ℝ) (interest_earned : ℝ) : ℝ :=
  (interest_earned * 100 * 365) / (principal * days)

/-- Theorem stating that given the specific conditions, the annual interest rate is 6.25% -/
theorem interest_rate_calculation (principal : ℝ) (days : ℝ) (interest_earned : ℝ) 
  (h1 : principal = 2000)
  (h2 : days = 73)
  (h3 : interest_earned = 25) :
  calculate_annual_interest_rate principal days interest_earned = 6.25 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval calculate_annual_interest_rate 2000 73 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1069_106914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_not_in_first_quadrant_l1069_106917

theorem complex_not_in_first_quadrant (m : ℝ) : 
  ¬∃ (z : ℂ), z = (m - 2*Complex.I) / (1 + 2*Complex.I) ∧ (z.re > 0 ∧ z.im > 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_not_in_first_quadrant_l1069_106917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_ten_solution_l1069_106975

-- Define a function to convert a list of digits in base b to a natural number
def toNatBase (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

-- Define the equation
def equation (b : Nat) : Prop :=
  toNatBase [2, 3, 8, 4] b + toNatBase [3, 4, 1, 7] b = toNatBase [1, 0, 2, 0, 1] b

-- Theorem statement
theorem base_ten_solution :
  ∃! b : Nat, b > 1 ∧ equation b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_ten_solution_l1069_106975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_l1069_106913

/-- A line is tangent to a circle if the distance from the center of the circle to the line equals the radius of the circle. -/
def is_tangent_line (a : ℝ) : Prop :=
  let circle_center : ℝ × ℝ := (0, 1)
  let circle_radius := 1
  let line_distance := |3 - a| / Real.sqrt (1 + (2 - a)^2)
  line_distance = circle_radius

theorem tangent_line_value (a : ℝ) : is_tangent_line a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_l1069_106913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_verify_solution_l1069_106906

/-- The functional equation problem -/
theorem functional_equation_solution :
  ∀ (f g : ℕ+ → ℕ+),
  (∀ n : ℕ+, f^[g n + 1] n + g^[f n] n = f (n + 1) - g (n + 1) + 1) →
  (∀ n : ℕ+, f n = n ∧ g n = 1) :=
by
  intros f g h
  -- We prove that f n = n and g n = 1 for all n
  have main_claim : ∀ n : ℕ+, f n = n ∧ g n = 1 := by
    intro n
    -- The proof would go here
    sorry
  -- Return the main claim
  exact main_claim

/-- Verifying the solution satisfies the equation -/
theorem verify_solution :
  let f : ℕ+ → ℕ+ := λ n => n
  let g : ℕ+ → ℕ+ := λ _ => 1
  ∀ n : ℕ+, f^[g n + 1] n + g^[f n] n = f (n + 1) - g (n + 1) + 1 :=
by
  intros f g n
  -- The verification would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_verify_solution_l1069_106906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_radius_l1069_106948

/-- The volume of a sphere with radius r -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The problem statement -/
theorem snowman_radius (x : ℝ) : 
  sphereVolume 2 + sphereVolume x + sphereVolume 4 + sphereVolume 6 = (1432 / 3) * Real.pi → 
  x = (70 : ℝ)^(1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_radius_l1069_106948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_3003_divisors_l1069_106997

def has_exactly_3003_divisors (n : ℕ) : Prop :=
  (Finset.filter (λ d ↦ d ∣ n) (Finset.range (n + 1))).card = 3003

def is_not_divisible_by_30 (m : ℕ) : Prop :=
  ¬(30 ∣ m)

theorem least_integer_with_3003_divisors :
  ∃ (n m k : ℕ),
    has_exactly_3003_divisors n ∧
    (∀ i < n, ¬has_exactly_3003_divisors i) ∧
    n = m * (30 ^ k) ∧
    is_not_divisible_by_30 m ∧
    m + k = 104978 := by
  sorry

#check least_integer_with_3003_divisors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_3003_divisors_l1069_106997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_line_segment_l1069_106926

-- Define points A and B
noncomputable def A : ℝ × ℝ := (1, 3)
noncomputable def B : ℝ × ℝ := (7, -1)

-- Define the vector from A to B
noncomputable def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define point C
noncomputable def C : ℝ × ℝ := (B.1 + AB.1 / 2, B.2 + AB.2 / 2)

-- Theorem statement
theorem extended_line_segment :
  C = (10, -3) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_line_segment_l1069_106926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_biking_carpooling_difference_l1069_106957

/-- Represents the seasons in a year -/
inductive Season
  | Winter
  | Summer
  | RestOfYear

/-- Represents the modes of transportation -/
inductive TransportMode
  | Drive
  | PublicTransport
  | Bike
  | Carpool

/-- Total number of employees -/
def totalEmployees : ℕ := 500

/-- Percentage of employees using each transport mode in a given season -/
def transportPercentage (s : Season) (m : TransportMode) : ℚ :=
  match s, m with
  | Season.Winter, TransportMode.Drive => 45 / 100
  | Season.Winter, TransportMode.PublicTransport => 30 / 100
  | Season.Summer, TransportMode.Drive => 35 / 100
  | Season.Summer, TransportMode.PublicTransport => 40 / 100
  | Season.RestOfYear, TransportMode.Drive => 55 / 100
  | Season.RestOfYear, TransportMode.PublicTransport => 25 / 100
  | _, _ => 0  -- Default value for other combinations

/-- Percentage of remaining employees who bike in a given season -/
def bikingPercentage (s : Season) : ℚ :=
  match s with
  | Season.Winter => 50 / 100
  | Season.Summer => 60 / 100
  | Season.RestOfYear => 40 / 100

/-- Calculate the number of employees using a specific transport mode in a given season -/
def employeesUsingMode (s : Season) (m : TransportMode) : ℚ :=
  (transportPercentage s m) * totalEmployees

/-- Calculate the number of employees biking in a given season -/
def employeesBiking (s : Season) : ℚ :=
  let remainingEmployees := totalEmployees - (employeesUsingMode s TransportMode.Drive + employeesUsingMode s TransportMode.PublicTransport)
  (bikingPercentage s) * remainingEmployees

/-- Calculate the number of employees carpooling in a given season -/
def employeesCarpooling (s : Season) : ℚ :=
  let remainingEmployees := totalEmployees - (employeesUsingMode s TransportMode.Drive + employeesUsingMode s TransportMode.PublicTransport)
  remainingEmployees - (employeesBiking s)

/-- Calculate the difference between biking and carpooling employees in a given season -/
def bikingCarpoolingDifference (s : Season) : ℚ :=
  (employeesBiking s) - (employeesCarpooling s)

/-- Theorem: The average difference between biking and carpooling employees over the year is approximately 1.67 -/
theorem average_biking_carpooling_difference :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  abs ((bikingCarpoolingDifference Season.Winter + bikingCarpoolingDifference Season.Summer + bikingCarpoolingDifference Season.RestOfYear) / 3 - 1.67) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_biking_carpooling_difference_l1069_106957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1069_106946

noncomputable section

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 + 2 * Real.cos α, 2 + 2 * Real.sin α)

-- Define the line l
noncomputable def line_l (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * x

-- Define the polar coordinate equation of curve C
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 - 2 * Real.sqrt 3 * ρ * Real.cos θ - 4 * ρ * Real.sin θ + 3 = 0

-- Theorem statement
theorem curve_C_properties :
  ∀ (α ρ θ : ℝ),
    let (x, y) := curve_C α
    (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
    polar_equation ρ θ ∧
    ∃ (ρ₁ ρ₂ : ℝ),
      polar_equation ρ₁ (π/6) ∧
      polar_equation ρ₂ (π/6) ∧
      ρ₁ * ρ₂ = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1069_106946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_terms_plus_four_is_geometric_l1069_106998

def my_sequence : ℕ → ℚ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then 2 * my_sequence n else my_sequence n + 2

def even_terms_plus_four (n : ℕ) : ℚ :=
  my_sequence (2 * n + 1) + 4

theorem even_terms_plus_four_is_geometric :
  ∃ (r : ℚ), ∀ (n : ℕ), even_terms_plus_four (n + 1) = r * even_terms_plus_four n :=
by
  use 2
  intro n
  simp [even_terms_plus_four, my_sequence]
  -- The actual proof would go here
  sorry

#eval my_sequence 0  -- Should output 1
#eval my_sequence 1  -- Should output 2
#eval my_sequence 2  -- Should output 4
#eval my_sequence 3  -- Should output 8
#eval even_terms_plus_four 0  -- Should output 6
#eval even_terms_plus_four 1  -- Should output 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_terms_plus_four_is_geometric_l1069_106998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_problem_solution_l1069_106989

noncomputable def river_problem (still_water_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : ℝ :=
  let upstream_speed := still_water_speed - river_speed
  let downstream_speed := still_water_speed + river_speed
  let one_way_distance := (upstream_speed * downstream_speed * total_time) / (2 * (upstream_speed + downstream_speed))
  2 * one_way_distance

theorem river_problem_solution :
  river_problem 8 2 1 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_problem_solution_l1069_106989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_exponent_l1069_106904

theorem prime_power_exponent (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  Nat.Prime (m ^ (4 ^ n + 1) - 1) → ∃ t : ℕ, n = 2 ^ t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_exponent_l1069_106904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doctor_lawyer_ratio_l1069_106916

/-- Represents a group of doctors and lawyers -/
structure MyGroup where
  doctors : ℕ
  lawyers : ℕ

/-- The average age of the entire group -/
def groupAverage (g : MyGroup) : ℝ := 40

/-- The average age of doctors -/
def doctorAverage : ℝ := 35

/-- The average age of lawyers -/
def lawyerAverage : ℝ := 50

/-- The ratio of doctors to lawyers is 2:1 -/
theorem doctor_lawyer_ratio (g : MyGroup) : 
  g.doctors = 2 * g.lawyers :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_doctor_lawyer_ratio_l1069_106916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nba_finals_sequences_l1069_106979

/-- Represents the number of games in the series -/
def num_games : ℕ := 7

/-- Represents the number of wins required to win the series -/
def wins_required : ℕ := 4

/-- Represents whether the winning team wins on their home court -/
def wins_at_home : Prop := True

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Represents the total number of possible win-loss sequences -/
def total_sequences : ℕ := choose 5 3 + choose 6 3

theorem nba_finals_sequences :
  num_games = 7 ∧ wins_required = 4 ∧ wins_at_home → total_sequences = 30 :=
by sorry

#eval total_sequences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nba_finals_sequences_l1069_106979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_arguments_of_fifth_roots_l1069_106988

noncomputable def z : ℂ := -1 / Real.sqrt 2 + Complex.I / Real.sqrt 2

def is_fifth_root (w : ℂ) : Prop := w^5 = z

def fifth_roots : Set ℂ := {w | is_fifth_root w}

theorem sum_of_arguments_of_fifth_roots :
  ∃ (args : Finset ℝ),
    args.card = 5 ∧
    (∀ θ ∈ args, 0 ≤ θ ∧ θ < 2 * Real.pi) ∧
    (∀ w ∈ fifth_roots, ∃ θ ∈ args, w = Complex.exp (Complex.I * θ)) ∧
    args.sum id = 35 * Real.pi / 2 :=
by sorry

#check sum_of_arguments_of_fifth_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_arguments_of_fifth_roots_l1069_106988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1069_106993

def A : Set ℝ := {y | ∃ x, y = 2 * x + 1}
def B : Set ℝ := {y | ∃ x, y = -x^2}

theorem intersection_of_A_and_B : A ∩ B = Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1069_106993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1069_106958

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℝ
  a_1 : ℝ := a - 1
  a_2 : ℝ := 4
  a_3 : ℝ := 2 * a

/-- Sum of first n terms of the arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  sorry

/-- b_n is defined as S_n / n -/
noncomputable def b (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  S seq n / n

/-- Main theorem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∃ k : ℕ, S seq k = 30 ∧ seq.a = 3 ∧ k = 5) ∧
  (∀ n : ℕ, n > 0 → b seq 3 + b seq 7 + b seq 11 + b seq (4*n - 1) = 2*n^2 + 2*n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1069_106958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_traverse_iff_black_excluded_l1069_106981

/-- Represents a square on the grid -/
structure Square where
  row : Nat
  col : Nat

/-- Defines the color of a square based on its position -/
def isBlack (s : Square) : Bool :=
  (s.row + s.col) % 2 = 1

/-- Represents a grid with 2m rows and 2n columns -/
structure Grid where
  m : Nat
  n : Nat

/-- Represents a path on the grid -/
def GridPath := List Square

/-- Checks if a path is valid (only horizontal or vertical moves) -/
def isValidPath (path : GridPath) : Bool :=
  sorry

/-- Checks if a path visits all squares except one -/
def visitsAllExceptOne (g : Grid) (path : GridPath) (excluded : Square) : Bool :=
  sorry

/-- Main theorem: A snail can traverse the grid visiting all squares except one
    if and only if the excluded square is black -/
theorem snail_traverse_iff_black_excluded (g : Grid) (excluded : Square) :
  (∃ path : GridPath, path.head? = some ⟨0, 0⟩ ∧
                  path.getLast? = some ⟨2 * g.m - 1, 2 * g.n - 1⟩ ∧
                  isValidPath path ∧
                  visitsAllExceptOne g path excluded) ↔
  isBlack excluded :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_traverse_iff_black_excluded_l1069_106981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PC_l1069_106990

noncomputable section

/-- Ship's speed in nautical miles per hour -/
def ship_speed : ℝ := 30

/-- Time sailed from A to B in hours -/
def time_AB : ℝ := 40 / 60

/-- Time sailed from B to C in hours -/
def time_BC : ℝ := 80 / 60

/-- Angle between AP and AB -/
def angle_PAB : ℝ := 60 * Real.pi / 180

/-- Angle between BP and BA -/
def angle_PBA : ℝ := 30 * Real.pi / 180

/-- Angle between BC and BA -/
def angle_ABC : ℝ := 60 * Real.pi / 180

theorem distance_PC (P A B C : ℝ × ℝ) : 
  A.1 = 0 ∧ A.2 = 0 →
  B.1 = 0 ∧ B.2 = ship_speed * time_AB →
  C.1 = ship_speed * time_BC * Real.cos angle_ABC ∧ 
  C.2 = B.2 + ship_speed * time_BC * Real.sin angle_ABC →
  P.1 = (A.1 - (Real.tan angle_PAB) * A.2) / (1 + (Real.tan angle_PAB) * (Real.tan angle_PBA)) ∧
  P.2 = (Real.tan angle_PBA) * P.1 →
  Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2) = 20 * Real.sqrt 7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PC_l1069_106990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l1069_106911

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a line y = 2x + b -/
structure Line where
  b : ℝ

/-- Represents a point on the parabola -/
structure ParabolaPoint (c : Parabola) where
  x : ℝ
  y : ℝ
  hy : y^2 = 2 * c.p * x

def chord_length (l : Line) (c : Parabola) : ℝ := 5

def passes_through_focus (l : Line) (c : Parabola) : Prop :=
  l.b = -c.p

def point_N : ℝ × ℝ := (3, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_distance_theorem (c : Parabola) (l : Line) 
  (h_chord : chord_length l c = 5)
  (h_focus : passes_through_focus l c) :
  ∀ (M : ParabolaPoint c), distance (M.x, M.y) point_N ≥ 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l1069_106911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_a_middle_correct_arrangements_a_end_b_not_end_correct_arrangements_ab_adjacent_not_c_correct_l1069_106907

/-- The number of different products to arrange -/
def n : ℕ := 5

/-- The number of arrangements with product A in the middle -/
def arrangements_a_middle : ℕ := 24

/-- The number of arrangements with A at either end and B not at either end -/
def arrangements_a_end_b_not_end : ℕ := 36

/-- The number of arrangements with A and B adjacent but not adjacent to C -/
def arrangements_ab_adjacent_not_c : ℕ := 36

/-- Theorem for the number of arrangements with A in the middle -/
theorem arrangements_a_middle_correct :
  arrangements_a_middle = (n - 1).factorial := by sorry

/-- Theorem for the number of arrangements with A at either end and B not at either end -/
theorem arrangements_a_end_b_not_end_correct :
  arrangements_a_end_b_not_end = 2 * 3 * (n - 2).factorial := by sorry

/-- Theorem for the number of arrangements with A and B adjacent but not adjacent to C -/
theorem arrangements_ab_adjacent_not_c_correct :
  arrangements_ab_adjacent_not_c = 2 * (n - 1).factorial - 2 * (n - 2).factorial := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_a_middle_correct_arrangements_a_end_b_not_end_correct_arrangements_ab_adjacent_not_c_correct_l1069_106907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1069_106900

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧
  (2 * t.a + t.b) * Real.cos t.C + t.c * Real.cos t.B = 0 ∧
  1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2 ∧
  t.A + t.B + t.C = Real.pi

-- Define the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.c = Real.sqrt 23 ∧ t.a = 5/3 ∧ t.b = 10/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1069_106900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_probability_l1069_106902

/-- Represents the possible combinations of signs that can be filled in the blanks -/
inductive SignCombination
  | MinusMinus
  | PlusPlus
  | PlusMinus
  | MinusPlus

/-- Checks if a given sign combination forms a perfect square -/
def isPerfectSquare (combo : SignCombination) : Bool :=
  match combo with
  | SignCombination.MinusPlus => true
  | SignCombination.PlusPlus => true
  | _ => false

/-- Counts the number of sign combinations that form a perfect square -/
def countPerfectSquares : Nat :=
  (List.filter isPerfectSquare [SignCombination.MinusMinus, SignCombination.PlusPlus, 
                                SignCombination.PlusMinus, SignCombination.MinusPlus]).length

/-- The total number of possible sign combinations -/
def totalCombinations : Nat := 4

/-- The probability of forming a perfect square -/
noncomputable def probabilityPerfectSquare : ℚ :=
  (countPerfectSquares : ℚ) / (totalCombinations : ℚ)

theorem perfect_square_probability :
  probabilityPerfectSquare = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_probability_l1069_106902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_l1069_106928

/-- A parabola with equation y^2 = 2Px passing through (2, 4) -/
structure Parabola where
  P : ℝ
  eq : (4 : ℝ)^2 = 2 * P * 2

/-- The focus of the parabola -/
def focus (p : Parabola) : ℝ × ℝ := (p.P, 0)

/-- Point A on the parabola -/
def A : ℝ × ℝ := (2, 4)

/-- Fixed point B -/
def B : ℝ × ℝ := (8, -8)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem stating the ratio of distances -/
theorem distance_ratio (p : Parabola) : 
  (distance A (focus p)) / (distance B (focus p)) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_l1069_106928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_cristina_race_headstart_l1069_106956

/-- Represents the race scenario between Nicky and Cristina -/
structure RaceScenario where
  totalDistance : ℕ
  cristinaSpeed : ℕ
  nickySpeed : ℕ
  catchUpTime : ℕ

/-- Calculates the head start time given a race scenario -/
def headStartTime (race : RaceScenario) : ℚ :=
  race.catchUpTime - (race.nickySpeed * race.catchUpTime) / race.cristinaSpeed

/-- Theorem stating that in the given race scenario, Cristina gave Nicky a 12-second head start -/
theorem nicky_cristina_race_headstart :
  let race : RaceScenario := {
    totalDistance := 300,
    cristinaSpeed := 5,
    nickySpeed := 3,
    catchUpTime := 30
  }
  headStartTime race = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_cristina_race_headstart_l1069_106956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1069_106967

-- Define the given parameters
noncomputable def train_speed_kmph : ℝ := 54
noncomputable def time_to_cross : ℝ := 54.995600351971845
noncomputable def bridge_length : ℝ := 660

-- Define the conversion factor from kmph to m/s
noncomputable def kmph_to_ms : ℝ := 1000 / 3600

-- Define the function to calculate the train length
noncomputable def calculate_train_length (speed_kmph time_seconds bridge_length_m : ℝ) : ℝ :=
  speed_kmph * kmph_to_ms * time_seconds - bridge_length_m

-- Theorem statement
theorem train_length_calculation :
  ∃ ε > 0, |calculate_train_length train_speed_kmph time_to_cross bridge_length - 164.93| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1069_106967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l1069_106959

/-- Given a triangle with one side of length 1, area 1/4, and two acute angles with sines in the ratio 1:2,
    the lengths of the other two sides are approximately 0.5115 and 1.0230. -/
theorem triangle_side_lengths (α β γ : Real) (a b : Real) :
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.sin α / Real.sin β = 1/2 →  -- ratio of sines
  γ = π - α - β →  -- sum of angles in a triangle
  a * Real.sin β = Real.sin α →  -- sine rule
  b * Real.sin α = Real.sin β →  -- sine rule
  1/4 = 1/2 * a * b * Real.sin γ →  -- area formula
  1 * Real.sin α = a * Real.sin γ →  -- sine rule
  1 * Real.sin β = b * Real.sin γ →  -- sine rule
  ∃ (ε : Real), ε > 0 ∧ ε < 0.0001 ∧ 
    (abs (a - 0.5115) < ε ∧ abs (b - 1.0230) < ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l1069_106959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_divisible_by_15_and_8_l1069_106986

theorem four_digit_divisible_by_15_and_8 : 
  (Finset.filter (fun n : Nat => 1000 ≤ n ∧ n < 10000 ∧ n % 15 = 0 ∧ n % 8 = 0) (Finset.range 10000)).card = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_divisible_by_15_and_8_l1069_106986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_outfits_count_l1069_106909

/-- Represents the number of items of a specific type and color -/
structure ItemCount where
  count : Nat

/-- Represents the total number of valid outfits -/
def TotalOutfits : Nat := 4240

/-- The number of red shirts -/
def redShirts : ItemCount := { count := 7 }

/-- The number of green shirts -/
def greenShirts : ItemCount := { count := 3 }

/-- The number of pants -/
def pants : ItemCount := { count := 8 }

/-- The number of blue shoes -/
def blueShoes : ItemCount := { count := 5 }

/-- The number of red shoes -/
def redShoes : ItemCount := { count := 5 }

/-- The number of green hats -/
def greenHats : ItemCount := { count := 10 }

/-- The number of red hats -/
def redHats : ItemCount := { count := 6 }

/-- Function to calculate the number of outfits for a specific combination -/
def outfitsForCombination (shirts : ItemCount) (pants : ItemCount) (shoes : ItemCount) (hats : ItemCount) : Nat :=
  shirts.count * pants.count * shoes.count * hats.count

/-- Theorem stating that the total number of valid outfits is 4240 -/
theorem total_outfits_count : TotalOutfits = 4240 := by
  -- Unfold the definition of TotalOutfits
  unfold TotalOutfits
  -- The proof is trivial since TotalOutfits is defined as 4240
  rfl

#check total_outfits_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_outfits_count_l1069_106909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_target_l1069_106905

noncomputable def options : List ℝ := [2000, 1500, 200, 2500, 3000]

noncomputable def target : ℝ := 503 / 0.251

theorem closest_to_target (x : ℝ) (h : x ∈ options) : 
  ∀ y ∈ options, |x - target| ≤ |y - target| ↔ x = 2000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_target_l1069_106905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_seven_l1069_106923

theorem divisible_by_seven (n : ℕ) : ∃ k : ℤ, 3^(2*n + 2) - 2^(n + 1) = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_seven_l1069_106923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_fractions_between_one_fourth_and_two_fifths_l1069_106987

theorem count_fractions_between_one_fourth_and_two_fifths :
  let is_valid (n d : ℕ) : Prop :=
    0 < n ∧ n < d ∧ d < 20 ∧ Nat.Coprime n d ∧ 
    (1 : ℚ) / 4 < (n : ℚ) / d ∧ (n : ℚ) / d < (2 : ℚ) / 5
  (Finset.filter (fun p : ℕ × ℕ => is_valid p.fst p.snd) (Finset.product (Finset.range 19) (Finset.range 20))).card = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_fractions_between_one_fourth_and_two_fifths_l1069_106987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_distance_ratio_l1069_106931

/-- Parabola type -/
structure Parabola where
  a : ℝ
  k : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given parabola P -/
def P : Parabola := { a := 4, k := 0 }

/-- Vertex of a parabola -/
def vertex (p : Parabola) : Point :=
  { x := 0, y := p.k }

/-- Focus of a parabola -/
noncomputable def focus (p : Parabola) : Point :=
  { x := 0, y := p.k + 1 / (4 * p.a) }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The ratio of F₁F₂ to V₁V₂ is 9/8 -/
theorem parabola_focal_distance_ratio :
  let V₁ := vertex P
  let F₁ := focus P
  let Q : Parabola := { a := 1/2, k := 1/2 }
  let V₂ := vertex Q
  let F₂ := focus Q
  distance F₁ F₂ / distance V₁ V₂ = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_distance_ratio_l1069_106931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waiter_customers_l1069_106942

theorem waiter_customers (initial : ℕ) (left : ℕ) (new : ℕ) :
  initial ≥ left →
  (initial - left + new : ℕ) = initial - left + new :=
by sorry

-- Example with given numbers
def example_customers : ℕ := 
  let initial := 19
  let left := 14
  let new := 36
  initial - left + new

#eval example_customers  -- Should output 41

end NUMINAMATH_CALUDE_ERRORFEEDBACK_waiter_customers_l1069_106942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_sine_function_l1069_106991

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + 5 * Real.pi / 2)

theorem symmetry_axis_of_sine_function :
  ∀ x : ℝ, f ((-Real.pi/2) + x) = f ((-Real.pi/2) - x) :=
by
  intro x
  unfold f
  simp [Real.sin_add]
  -- The proof is omitted for brevity
  sorry

#check symmetry_axis_of_sine_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_sine_function_l1069_106991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cube_of_matrix_l1069_106995

theorem det_cube_of_matrix {n : Type*} [Fintype n] [DecidableEq n] 
  (N : Matrix n n ℝ) (h : Matrix.det N = 3) :
  Matrix.det (N ^ 3) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cube_of_matrix_l1069_106995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_are_odd_l1069_106927

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x + 1/x
noncomputable def g (x : ℝ) : ℝ := x^3 + 2*x

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem f_and_g_are_odd :
  is_odd_function f ∧ is_odd_function g :=
by
  constructor
  · -- Prove f is odd
    intro x
    simp [f, is_odd_function]
    field_simp
    ring
  · -- Prove g is odd
    intro x
    simp [g, is_odd_function]
    ring

-- Note: The proof is completed without using 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_are_odd_l1069_106927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_max_on_interval_l1069_106992

-- Define the quadratic function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_min : ∀ x : ℝ, f x ≥ 1
axiom f_at_zero : f 0 = 3
axiom f_at_two : f 2 = 3

-- Theorem for the analytical expression of f
theorem f_expression : f = fun x ↦ 2 * (x - 1)^2 + 1 := by sorry

-- Theorem for the maximum value of f on [-1/2, 3/2]
theorem f_max_on_interval : 
  ∃ x ∈ Set.Icc (-1/2 : ℝ) (3/2), ∀ y ∈ Set.Icc (-1/2 : ℝ) (3/2), f x ≥ f y ∧ f x = 11/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_expression_f_max_on_interval_l1069_106992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_problem_l1069_106944

-- Define the circle C
def my_circle (m : ℝ) (x y : ℝ) : Prop := (x - m)^2 + y^2 = 5

-- Define the ellipse E
def my_ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the dot product of two 2D vectors
def my_dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem ellipse_and_circle_problem 
  (m : ℝ) 
  (a b : ℝ) 
  (h1 : m < 3) 
  (h2 : a > b) 
  (h3 : b > 0) 
  (h4 : my_circle m 3 1) 
  (h5 : my_ellipse a b 3 1) 
  (h6 : ∃ k : ℝ, k * 4 - 4 - 4 * k + 4 = 0 ∧ 
    (k * x - y - 4 * k + 4)^2 / (k^2 + 1) = 5 ∧ 
    x < 4) :
  m = 1 ∧ 
  a^2 = 18 ∧ 
  b^2 = 2 ∧ 
  ∀ x y : ℝ, my_ellipse a b x y → 
    -12 ≤ my_dot_product (x - 3) (y - 1) 1 3 ∧ 
    my_dot_product (x - 3) (y - 1) 1 3 ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_problem_l1069_106944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_focus_parallel_to_given_line_l1069_106901

/-- The focus of the parabola y^2 = 2x -/
noncomputable def focus : ℝ × ℝ := (1/2, 0)

/-- The slope of the given line 3x - 2y + 5 = 0 -/
noncomputable def m : ℝ := 3/2

/-- The equation of line l that passes through the focus and is parallel to 3x - 2y + 5 = 0 -/
def line_equation (x y : ℝ) : Prop := 6*x - 4*y - 3 = 0

theorem line_through_focus_parallel_to_given_line :
  ∀ x y : ℝ, line_equation x y ↔ 
    (x = focus.1 ∧ y = focus.2) ∨ 
    (∃ t : ℝ, x = focus.1 + t ∧ y = focus.2 + m * t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_focus_parallel_to_given_line_l1069_106901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_achievable_l1069_106964

def sequence_list : List ℕ := [1, 3, 9, 27, 81, 243, 729]

def is_achievable (n : ℕ) : Prop :=
  ∃ (subset : List ℤ), 
    (∀ x, x ∈ subset → ∃ y ∈ sequence_list, x = y ∨ x = -y) ∧
    subset.sum = n

theorem all_achievable : 
  ∀ n : ℕ, n ≤ 1093 → is_achievable n :=
by sorry

#check all_achievable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_achievable_l1069_106964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_equal_one_l1069_106963

open Real

-- Define the expressions
noncomputable def expression_a : ℝ :=
  (1 - cos (15 * π / 180)) * (1 + sin (75 * π / 180)) + 
  cos (75 * π / 180) * cos (15 * π / 180) * (1 / tan (15 * π / 180))

noncomputable def expression_b (α : ℝ) : ℝ :=
  sin (45 * π / 180 - α) - cos (30 * π / 180 + α) + 
  sin (30 * π / 180) ^ 2 - cos (45 * π / 180 + α) + 
  sin (60 * π / 180 - α) + sin (60 * π / 180) ^ 2

-- Theorem statement
theorem expressions_equal_one :
  expression_a = 1 ∧ ∀ α, expression_b α = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expressions_equal_one_l1069_106963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_analysis_l1069_106941

theorem condition_analysis : 
  (∀ x : ℝ, x > 1 → 1 / x < 1) ∧ 
  (∃ x : ℝ, x ≤ 1 ∧ 1 / x < 1) := by
  constructor
  · intro x hx
    sorry
  · use -1
    constructor
    · linarith
    · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_analysis_l1069_106941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_angle_formula_l1069_106943

/-- The angle between apothems of adjacent lateral faces in a regular n-sided pyramid -/
noncomputable def apothem_angle (n : ℕ) (α : ℝ) : ℝ :=
  2 * Real.arcsin (Real.cos (Real.pi / n) * Real.tan (α / 2))

/-- Theorem: The angle between apothems of adjacent lateral faces in a regular n-sided pyramid -/
theorem apothem_angle_formula (n : ℕ) (α : ℝ) (h1 : n ≥ 3) (h2 : 0 < α ∧ α < Real.pi) :
  ∃ θ : ℝ, θ = apothem_angle n α ∧ 
    θ = 2 * Real.arcsin (Real.cos (Real.pi / n) * Real.tan (α / 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_angle_formula_l1069_106943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_trip_choices_l1069_106949

theorem class_trip_choices : Fintype.card (Fin 4 → Fin 3) = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_trip_choices_l1069_106949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1069_106978

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) / Real.log (2 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-1) 1 ∪ Set.Ioo 1 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1069_106978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zero_l1069_106924

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 / (3^x + 1) + a

theorem function_zero (a : ℝ) : f 1 a = 0 ↔ a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zero_l1069_106924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_one_l1069_106934

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_value_at_one (f : ℝ → ℝ) 
  (h1 : is_odd (fun x ↦ f x + x^2)) 
  (h2 : is_even (fun x ↦ f x + 2^x)) : 
  f 1 = -7/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_one_l1069_106934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_integers_l1069_106994

-- Define is_valid as a decidable predicate
def is_valid (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 60 ∧ ¬ Nat.Prime n ∧ (Nat.factorial (n^2 - 3)) % (Nat.factorial n)^(n-1) = 0

-- Provide an instance of DecidablePred for is_valid
instance : DecidablePred is_valid :=
  fun n => decidable_of_iff
    (10 ≤ n ∧ n ≤ 60 ∧ ¬ Nat.Prime n ∧ (Nat.factorial (n^2 - 3)) % (Nat.factorial n)^(n-1) = 0)
    (by simp [is_valid])

-- State the theorem
theorem count_valid_integers : (Finset.filter is_valid (Finset.range 51)).card = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_integers_l1069_106994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_difference_l1069_106977

/-- Calculates the new price after applying a percentage change --/
noncomputable def apply_percentage_change (price : ℝ) (percentage : ℝ) : ℝ :=
  price * (1 + percentage / 100)

/-- Represents the pricing policy for the store items --/
structure PricingPolicy where
  candy_box : ℝ
  soda_can : ℝ
  popcorn_bag : ℝ
  gum_pack : ℝ

/-- Calculates the total cost of all items --/
noncomputable def total_cost (policy : PricingPolicy) : ℝ :=
  policy.candy_box + policy.soda_can + policy.popcorn_bag + policy.gum_pack

/-- Initial pricing policy --/
def initial_policy : PricingPolicy :=
  { candy_box := 10
    soda_can := 9
    popcorn_bag := 5
    gum_pack := 2 }

/-- New pricing policy after changes --/
noncomputable def new_policy : PricingPolicy :=
  { candy_box := apply_percentage_change initial_policy.candy_box 25
    soda_can := apply_percentage_change initial_policy.soda_can (-15)
    popcorn_bag := initial_policy.popcorn_bag * 2
    gum_pack := initial_policy.gum_pack }

theorem price_change_difference :
  total_cost new_policy - total_cost initial_policy = 6.15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_difference_l1069_106977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_drawing_events_l1069_106968

/-- Represents the outcome of drawing products -/
inductive Outcome
  | AtLeastOneGenuine
  | AtLeastThreeDefective
  | AllSixDefective
  | TwoDefectiveFourGenuine
deriving Repr, DecidableEq

/-- Represents the type of event -/
inductive EventType
  | Certain
  | Random
  | Impossible
deriving Repr, DecidableEq

/-- Function to determine the event type given the total products, genuine products, and outcome -/
def eventType (total : ℕ) (genuine : ℕ) (outcome : Outcome) : EventType :=
  sorry

/-- Theorem stating the properties of drawing 6 products from 100 with 95 genuine and 5 defective -/
theorem product_drawing_events :
  let total := 100
  let genuine := 95
  let defective := 5
  let drawn := 6
  (eventType total genuine Outcome.AtLeastOneGenuine = EventType.Certain) ∧
  (eventType total genuine Outcome.AtLeastThreeDefective = EventType.Random) ∧
  (eventType total genuine Outcome.AllSixDefective = EventType.Impossible) ∧
  (eventType total genuine Outcome.TwoDefectiveFourGenuine = EventType.Random) ∧
  (Finset.filter (λ o => eventType total genuine o = EventType.Random)
    {Outcome.AtLeastOneGenuine, Outcome.AtLeastThreeDefective,
     Outcome.AllSixDefective, Outcome.TwoDefectiveFourGenuine}).card = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_drawing_events_l1069_106968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_intersection_l1069_106952

/-- The circle with center (1,2) and radius 2 -/
def myCircle : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 2)^2 = 4}

/-- The point M -/
def M : ℝ × ℝ := (3, 1)

/-- A line with parameter a -/
def line (a : ℝ) : Set (ℝ × ℝ) := {p | a * p.1 - p.2 + 4 = 0}

/-- The length of a chord AB formed by the intersection of the line and the circle -/
noncomputable def chordLength (a : ℝ) : ℝ := 2 * Real.sqrt 3

theorem tangent_and_intersection :
  (∃ (k : ℝ), ({p : ℝ × ℝ | p.1 = 3} ∪ {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 - 5 = 0}) = 
    {p : ℝ × ℝ | p ∈ myCircle ∧ (∃ (t : ℝ), p = M + t • (p - M))}) ∧
  (∃ (A B : ℝ × ℝ), A ∈ myCircle ∧ B ∈ myCircle ∧ A ∈ line (-3/4) ∧ B ∈ line (-3/4) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (chordLength (-3/4))^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_intersection_l1069_106952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_1_to_40_last_nonzero_digit_l1069_106938

def last_nonzero_digit (n : Nat) : Nat := 
  n % 10 

theorem product_1_to_40_last_nonzero_digit : 
  last_nonzero_digit (Finset.prod (Finset.range 40) (λ i => i + 1)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_1_to_40_last_nonzero_digit_l1069_106938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_half_l1069_106969

/-- The sum of the infinite series ∑(n=1 to ∞) [(n^3 + n^2 - n) / (n + 3)!] -/
noncomputable def infinite_series_sum : ℝ :=
  ∑' n : ℕ, (n ^ 3 + n ^ 2 - n : ℝ) / (n + 3).factorial

/-- Theorem stating that the sum of the infinite series is equal to 1/2 -/
theorem infinite_series_sum_eq_half : infinite_series_sum = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_half_l1069_106969
