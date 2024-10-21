import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_m3n6_in_expansion_l531_53170

theorem coefficient_m3n6_in_expansion (m n : ℕ) : 
  (Finset.range 10).sum (λ k ↦ (Nat.choose 9 k) * m^k * n^(9-k)) =
  84 * m^3 * n^6 + (Finset.range 10).sum (λ k ↦ if k ≠ 3 then (Nat.choose 9 k) * m^k * n^(9-k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_m3n6_in_expansion_l531_53170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_black_cards_probability_l531_53140

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of black cards in a standard deck -/
def blackCardCount : ℕ := 26

/-- The number of cards drawn -/
def drawnCardCount : ℕ := 4

/-- The probability of drawing four black cards from the top of a standard deck -/
def probabilityFourBlackCards : ℚ := 276 / 4998

theorem four_black_cards_probability :
  (Nat.choose blackCardCount drawnCardCount : ℚ) / (Nat.choose standardDeckSize drawnCardCount) = probabilityFourBlackCards :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_black_cards_probability_l531_53140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_BC_l531_53130

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the angles and side lengths
noncomputable def angle_A : ℝ := 40 * Real.pi / 180
noncomputable def angle_B : ℝ := 70 * Real.pi / 180
def side_AC : ℝ := 7

-- Define the theorem
theorem triangle_side_BC (t : Triangle) : 
  let BC := side_AC * (Real.sin angle_A) / (Real.sin angle_B)
  ∃ ε > 0, abs (BC - 4.78) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_BC_l531_53130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_exists_l531_53154

/-- Represents the price reduction in yuan -/
def x : ℝ := sorry

/-- Initial average daily sale -/
def initial_sales : ℝ := 20

/-- Initial profit per piece in yuan -/
def initial_profit_per_piece : ℝ := 40

/-- Increase in sales per yuan of price reduction -/
def sales_increase_rate : ℝ := 2

/-- Target average daily profit in yuan -/
def target_daily_profit : ℝ := 1200

/-- New profit per piece after price reduction -/
def new_profit_per_piece (x : ℝ) : ℝ := initial_profit_per_piece - x

/-- New daily sales after price reduction -/
def new_daily_sales (x : ℝ) : ℝ := initial_sales + sales_increase_rate * x

/-- Theorem stating the existence of a price reduction that meets the requirements -/
theorem price_reduction_exists : 
  ∃ x : ℝ, new_profit_per_piece x * new_daily_sales x = target_daily_profit := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_exists_l531_53154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_when_f_has_zero_points_l531_53139

open Real MeasureTheory

/-- The function f(x) with parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * sin x + (3 * Real.sqrt 3 / π) * x + m

/-- The domain of x -/
def domain : Set ℝ := {x | -π/3 ≤ x ∧ x ≤ π/3}

/-- Theorem stating the range of m when f has zero points in the domain -/
theorem range_of_m_when_f_has_zero_points (m : ℝ) (h : ∃ x ∈ domain, f m x = 0) :
  m ∈ Set.Icc (-2 * Real.sqrt 3) (2 * Real.sqrt 3) := by
  sorry

/-- The range of m -/
def range_of_m : Set ℝ := Set.Icc (-2 * Real.sqrt 3) (2 * Real.sqrt 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_when_f_has_zero_points_l531_53139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_savings_difference_l531_53148

/-- Represents the savings from a coupon --/
def Savings (price : ℝ) : ℕ → ℝ
| 0 => 0.20 * price  -- Coupon A
| 1 => 40            -- Coupon B
| 2 => 0.30 * (price - 100)  -- Coupon C
| _ => 0

/-- The problem statement --/
theorem coupon_savings_difference :
  ∃ (x y : ℝ),
    x > 100 ∧ y > 100 ∧
    (∀ p : ℝ, x ≤ p ∧ p ≤ y →
      Savings p 0 ≥ max (Savings p 1) (Savings p 2)) ∧
    (∀ p : ℝ, p < x ∨ p > y →
      Savings p 0 < max (Savings p 1) (Savings p 2)) ∧
    y - x = 100 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_savings_difference_l531_53148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_chord_length_l531_53113

/-- Represents an octagon inscribed in a circle -/
structure InscribedOctagon where
  -- Four consecutive sides of length 4
  short_sides : Fin 4 → ℝ
  short_sides_length : ∀ i, short_sides i = 4
  -- Four consecutive sides of length 6
  long_sides : Fin 4 → ℝ
  long_sides_length : ∀ i, long_sides i = 6
  -- The chord dividing the octagon into two quadrilaterals
  chord : ℝ

/-- The theorem statement -/
theorem octagon_chord_length (O : InscribedOctagon) :
  ∃ (p q : ℕ), 
    O.chord = 4 * Real.sqrt 3 ∧
    O.chord = p / q ∧
    Nat.Coprime p q ∧
    p + q = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_chord_length_l531_53113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_diagonal_division_l531_53195

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cuboid in 3D space -/
structure Cuboid where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D
  a' : Point3D
  b' : Point3D
  c' : Point3D
  d' : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  d : ℝ

/-- Function to check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane) : Prop :=
  plane.normal.x * p.x + plane.normal.y * p.y + plane.normal.z * p.z = plane.d

/-- Function to calculate the intersection point of a line and a plane -/
noncomputable def lineIntersectPlane (p1 p2 : Point3D) (plane : Plane) : Point3D :=
  sorry

/-- Theorem stating that the plane passing through B, C, D divides AA' in the ratio 1:2 -/
theorem cuboid_diagonal_division (cube : Cuboid) :
  let plane : Plane := ⟨⟨sorry, sorry, sorry⟩, sorry⟩
  let t := lineIntersectPlane cube.a cube.a' plane
  pointOnPlane cube.b plane ∧ 
  pointOnPlane cube.c plane ∧ 
  pointOnPlane cube.d plane →
  (t.x - cube.a.x) / (cube.a'.x - t.x) = 1 / 2 ∧
  (t.y - cube.a.y) / (cube.a'.y - t.y) = 1 / 2 ∧
  (t.z - cube.a.z) / (cube.a'.z - t.z) = 1 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_diagonal_division_l531_53195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_suits_in_six_draws_exact_l531_53138

/-- The probability of drawing at least one card from each suit when drawing 6 cards with replacement from a standard 52-card deck -/
noncomputable def prob_all_suits_in_six_draws : ℝ :=
  let num_suits : ℕ := 4
  let num_draws : ℕ := 6
  let prob_suit : ℝ := 1 / num_suits
  let prob_not_suit : ℝ := 1 - prob_suit
  let prob_miss_any_suit : ℝ := num_suits * prob_not_suit ^ num_draws -
                                 (num_suits.choose 2) * (1 - 2 * prob_suit) ^ num_draws +
                                 (num_suits.choose 3) * (1 - 3 * prob_suit) ^ num_draws -
                                 (num_suits.choose 4) * (1 - 4 * prob_suit) ^ num_draws
  1 - prob_miss_any_suit

/-- The probability of drawing at least one card from each suit when drawing 6 cards with replacement from a standard 52-card deck is 1260/4096 -/
theorem prob_all_suits_in_six_draws_exact : prob_all_suits_in_six_draws = 1260 / 4096 := by
  sorry

-- Note: The #eval command is removed as it might not work with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_suits_in_six_draws_exact_l531_53138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_critical_points_l531_53184

/-- Given a function f(x) = xe^x - ae^(2x) where a is a real number,
    if f(x) has exactly two critical points x₁ and x₂ with x₁ < x₂,
    then a is in the open interval (0, 1/2). -/
theorem range_of_a_for_two_critical_points
  (f : ℝ → ℝ)
  (a : ℝ)
  (h_f : ∀ x, f x = x * Real.exp x - a * Real.exp (2 * x))
  (h_critical : ∃ x₁ x₂, x₁ < x₂ ∧
    (∀ x, HasDerivAt f (0 : ℝ) x ↔ x = x₁ ∨ x = x₂)) :
  0 < a ∧ a < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_critical_points_l531_53184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kayak_rental_cost_l531_53173

/-- Represents the rental business scenario -/
structure RentalBusiness where
  canoe_cost : ℕ
  canoe_kayak_ratio : Rat
  total_revenue : ℕ
  canoe_kayak_difference : ℕ

/-- Theorem stating the cost of a kayak rental per day -/
theorem kayak_rental_cost (rb : RentalBusiness)
  (h1 : rb.canoe_cost = 14)
  (h2 : rb.canoe_kayak_ratio = 3 / 2)
  (h3 : rb.total_revenue = 288)
  (h4 : rb.canoe_kayak_difference = 4) :
  ∃ (kayak_cost : ℕ), kayak_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kayak_rental_cost_l531_53173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_circle_l531_53123

/-- The maximum value of |PM| + |PF₁| for an ellipse and circle -/
theorem max_distance_ellipse_circle : ∃ (max : ℝ),
  let ellipse := {P : ℝ × ℝ | (P.1^2 / 25) + (P.2^2 / 9) = 1}
  let circle := {M : ℝ × ℝ | M.1^2 + (M.2 - 2 * Real.sqrt 5)^2 = 1}
  let F₁ : ℝ × ℝ := (-4, 0)
  (∀ (P : ℝ × ℝ) (M : ℝ × ℝ), P ∈ ellipse → M ∈ circle →
    Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) + Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) ≤ max) ∧
  max = 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_circle_l531_53123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_subset_l531_53192

/-- The set of all lattice points in ℤ³ -/
def T : Set (ℤ × ℤ × ℤ) := Set.univ

/-- Two lattice points are neighbors if they have two coordinates the same and the third differs by 1 -/
def isNeighbor (p q : ℤ × ℤ × ℤ) : Prop :=
  (p.1 = q.1 ∧ p.2.1 = q.2.1 ∧ (p.2.2 = q.2.2 + 1 ∨ p.2.2 = q.2.2 - 1)) ∨
  (p.1 = q.1 ∧ p.2.2 = q.2.2 ∧ (p.2.1 = q.2.1 + 1 ∨ p.2.1 = q.2.1 - 1)) ∨
  (p.2.1 = q.2.1 ∧ p.2.2 = q.2.2 ∧ (p.1 = q.1 + 1 ∨ p.1 = q.1 - 1))

/-- The theorem stating the existence of subset S with the required properties -/
theorem exists_special_subset : ∃ (S : Set (ℤ × ℤ × ℤ)),
  (∀ x ∈ S, ∀ y ∈ T, isNeighbor x y → y ∉ S) ∧
  (∀ x ∈ T, x ∉ S → ∃! y, y ∈ T ∧ isNeighbor x y ∧ y ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_subset_l531_53192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arrangement_l531_53164

-- Define the type for table entries
inductive TableEntry
| One
| NegOne

-- Define the table type
def Table := Matrix (Fin 600) (Fin 600) TableEntry

-- Function to calculate the sum of a rectangle
def rectangleSum (t : Table) (row_start col_start row_size col_size : Nat) : Int :=
  sorry

-- Function to check if a table satisfies the rectangle condition
def satisfiesRectangleCondition (t : Table) : Prop :=
  ∀ (row col : Fin 597) (is_4x6 : Bool),
    let sum := if is_4x6 then
                 rectangleSum t row.val col.val 4 6
               else
                 rectangleSum t row.val col.val 6 4
    abs sum > 4

-- Function to calculate the sum of the entire table
def tableSum (t : Table) : Int :=
  sorry

-- The main theorem
theorem impossible_arrangement : ¬ ∃ (t : Table),
  satisfiesRectangleCondition t ∧ abs (tableSum t) < 90000 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_arrangement_l531_53164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_power_sum_l531_53122

theorem last_digit_of_power_sum (m : ℕ) : (2^(m+2006) + 2^m) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_power_sum_l531_53122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poster_minimal_area_l531_53175

/-- The minimal area of wall covered by posters -/
def minimal_area (m n : ℕ) : ℕ :=
  m * (n * (n + 1) / 2)

/-- Theorem stating the minimal area of wall covered by posters -/
theorem poster_minimal_area (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m ≥ n) :
  ∃ (arrangement : Finset (ℕ × ℕ)),
    (∀ (k l : ℕ), (k, l) ∈ arrangement → 1 ≤ k ∧ k ≤ m ∧ 1 ≤ l ∧ l ≤ n) ∧
    (arrangement.card = m * n) ∧
    (∀ (cover : Finset (ℕ × ℕ)), 
      (∀ (p : ℕ × ℕ), p ∈ cover → p ∈ arrangement) →
      (∀ (p q : ℕ × ℕ), p ∈ cover ∧ q ∈ cover ∧ p ≠ q → 
        p.1 ≤ q.1 ∧ p.2 ≤ q.2 → p.1 = q.1 ∧ p.2 = q.2) →
      (Finset.sum cover (λ (p : ℕ × ℕ) => p.1 * p.2) ≥ minimal_area m n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poster_minimal_area_l531_53175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_sectors_theorem_l531_53100

/-- The area of the figure formed by the overlap of two 90° sectors of a circle with radius 15 -/
noncomputable def overlapping_sectors_area (r : ℝ) (θ : ℝ) : ℝ :=
  2 * (θ / (2 * Real.pi)) * (Real.pi * r^2) - ((θ / 2) / (2 * Real.pi)) * (Real.pi * r^2)

theorem overlapping_sectors_theorem :
  overlapping_sectors_area 15 (Real.pi / 2) = 84.375 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_sectors_theorem_l531_53100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_line_problem_l531_53162

-- Define the coordinates of points A and B
noncomputable def A : ℝ := -8
noncomputable def B : ℝ := A + 20

-- Define the folding point C
noncomputable def C : ℝ := (B + (-4)) / 2

-- Define the possible coordinates of point D
noncomputable def D₁ : ℝ := -10.5
noncomputable def D₂ : ℝ := 14.5

-- Define the time when A and B are equidistant from the origin
noncomputable def t : ℝ := 4/3

-- Theorem statement
theorem number_line_problem :
  B = 12 ∧
  C = 4 ∧
  (|D₁ - A| + |B - D₁| = 25 ∨ |D₂ - A| + |B - D₂| = 25) ∧
  |A - t| = |B - 2*t| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_line_problem_l531_53162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_guaranteed_matches_l531_53197

/-- Represents a circular arrangement of items -/
def CircularArrangement := List Bool

/-- Checks if two circular arrangements have at least n items in the same position -/
def hasAtLeastNMatches (arr1 arr2 : CircularArrangement) (n : Nat) : Prop :=
  ((List.zip arr1 arr2).filter (fun (x, y) => x = y)).length ≥ n

/-- Rotates a circular arrangement by k positions -/
def rotate (arr : CircularArrangement) (k : Nat) : CircularArrangement :=
  let n := arr.length
  (List.drop (k % n) arr) ++ (List.take (k % n) arr)

theorem guaranteed_matches (arr : CircularArrangement) :
  arr.length = 15 →
  (arr.filter id).length = 8 →
  ∀ k, hasAtLeastNMatches arr (rotate arr k) 7 :=
by
  intro h_length h_white_count k
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_guaranteed_matches_l531_53197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_in_expansion_l531_53133

theorem constant_term_in_expansion (n : ℕ) (x : ℝ) : 
  (2 : ℝ)^n = 512 → 
  (∃ (k : ℕ), (Nat.choose n k) * (-1)^k * x^((n - 3*k)/2) = -84 ∧ 
               (n - 3*k)/2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_in_expansion_l531_53133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_of_roots_l531_53178

theorem tan_sum_of_roots (α β : ℝ) : 
  (Real.tan α)^2 + 3*(Real.tan α) - 2 = 0 → 
  (Real.tan β)^2 + 3*(Real.tan β) - 2 = 0 → 
  Real.tan (α + β) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_of_roots_l531_53178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_inverse_l531_53158

theorem power_equality_implies_inverse (x : ℝ) : (81 : ℝ)^4 = (27 : ℝ)^x → (3 : ℝ)^(-x) = 1 / (3 : ℝ)^(16/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_inverse_l531_53158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l531_53168

theorem tan_double_angle (θ : ℝ) (h1 : Real.cos θ = -3/5) (h2 : θ ∈ Set.Ioo 0 Real.pi) : 
  Real.tan (2 * θ) = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l531_53168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_log10_l531_53193

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem inverse_of_log10 :
  ∀ x : ℝ, x > 0 → f (10^x) = x ∧ 10^(f x) = x :=
by
  -- Introduce variables and assumption
  intro x hx
  -- Split the conjunction
  constructor
  
  -- Prove f (10^x) = x
  · sorry
  
  -- Prove 10^(f x) = x
  · sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_log10_l531_53193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l531_53171

theorem parity_of_expression (a b c : ℕ) (ha : Odd a) (hb : Odd b) :
  Odd c → Even (3^a + (b+2)^2*c) ∧ Even c → Odd (3^a + (b+2)^2*c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l531_53171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_of_expressions_l531_53126

theorem positive_difference_of_expressions : 
  |((6^2 + 6^2) / 6 : ℚ) - ((6^2 * 6^2) / 6 : ℚ)| = 204 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_of_expressions_l531_53126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l531_53174

structure FabricSquare where
  length : Nat
  width : Nat
  color : String

def redSquare : FabricSquare := { length := 8, width := 5, color := "red" }
def blueSquare : FabricSquare := { length := 10, width := 7, color := "blue" }
def greenSquare : FabricSquare := { length := 5, width := 5, color := "green" }
def yellowSquare : FabricSquare := { length := 6, width := 4, color := "yellow" }
def whiteSquare : FabricSquare := { length := 12, width := 8, color := "white" }

def fabricSquares : List FabricSquare := [redSquare, blueSquare, greenSquare, yellowSquare, whiteSquare]

def flagLength : Nat := 20
def flagHeight : Nat := 12

def isValidArrangement (arrangement : List FabricSquare) : Prop :=
  arrangement.length = 5 ∧
  (arrangement.map (λ s => s.length)).sum ≤ flagLength ∧
  (arrangement.map (λ s => s.width)).maximum?.getD 0 ≤ flagHeight ∧
  (arrangement.map (λ s => s.color)) = ["red", "blue", "green", "yellow", "white"]

theorem no_valid_arrangement :
  ¬∃ arrangement : List FabricSquare, arrangement.Sublist fabricSquares ∧ isValidArrangement arrangement := by
  sorry

#eval fabricSquares.map (λ s => s.length)
#eval (fabricSquares.map (λ s => s.length)).sum
#eval fabricSquares.map (λ s => s.width)
#eval (fabricSquares.map (λ s => s.width)).maximum?.getD 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l531_53174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_length_l531_53131

-- Define a triangle with specific properties
structure SpecialTriangle where
  -- The ratio of interior angles is 1:5:6
  angle_ratio : Fin 3 → ℕ
  angle_ratio_valid : angle_ratio = ![1, 5, 6]
  -- The longest side has length 12
  longest_side : ℝ
  longest_side_length : longest_side = 12

-- Helper function (not part of the problem, but needed for the theorem statement)
noncomputable def altitude_to_longest_side (t : SpecialTriangle) : ℝ :=
  sorry

-- Theorem statement
theorem altitude_length (t : SpecialTriangle) : 
  ∃ (h : ℝ), h = 3 ∧ h = altitude_to_longest_side t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_length_l531_53131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_form_l531_53146

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ
deriving Inhabited

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Represents the quadrilateral -/
def quad : List Point := [
  ⟨1, 2⟩,
  ⟨4, 5⟩,
  ⟨5, 4⟩,
  ⟨2, 1⟩
]

/-- Calculates the perimeter of the quadrilateral -/
noncomputable def perimeter : ℝ :=
  distance (quad[0]) (quad[1]) +
  distance (quad[1]) (quad[2]) +
  distance (quad[2]) (quad[3]) +
  distance (quad[3]) (quad[0])

theorem perimeter_form :
  ∃ (c d : ℤ), perimeter = c * Real.sqrt 2 + d * Real.sqrt 10 ∧ c + d = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_form_l531_53146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_part_distance_approx_12_l531_53106

/-- Represents a two-part bicycle trip -/
structure BicycleTrip where
  distance1 : ℝ  -- Distance of the first part in km
  speed1 : ℝ     -- Average speed of the first part in km/hr
  speed2 : ℝ     -- Average speed of the second part in km/hr
  avgSpeed : ℝ   -- Average speed for the entire trip in km/hr

/-- Calculates the distance of the second part of the trip -/
noncomputable def secondPartDistance (trip : BicycleTrip) : ℝ :=
  let time1 := trip.distance1 / trip.speed1
  let totalTime := trip.distance1 / trip.avgSpeed
  let time2 := totalTime - time1
  trip.speed2 * time2

/-- Theorem stating that for the given conditions, the second part distance is approximately 12 km -/
theorem second_part_distance_approx_12 (trip : BicycleTrip)
    (h1 : trip.distance1 = 10)
    (h2 : trip.speed1 = 12)
    (h3 : trip.speed2 = 10)
    (h4 : trip.avgSpeed = 10.82) :
    ∃ ε > 0, abs (secondPartDistance trip - 12) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_part_distance_approx_12_l531_53106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_above_1999_l531_53132

/-- Represents the number of cards above a given card in the original stack -/
def cards_above (n : ℕ) (stack : List ℕ) : ℕ :=
  sorry

/-- The process of removing the top card and moving the next to the bottom -/
def process (stack : List ℕ) : List ℕ :=
  sorry

/-- Checks if a list is in ascending order -/
def is_ascending (lst : List ℕ) : Prop :=
  sorry

/-- The theorem to be proved -/
theorem cards_above_1999 (initial_stack : List ℕ) :
  initial_stack.length = 2000 ∧
  (∀ i, i ∈ initial_stack → 1 ≤ i ∧ i ≤ 2000) ∧
  (∀ i j, i ≠ j → initial_stack.get? i ≠ initial_stack.get? j) ∧
  is_ascending (process initial_stack) →
  cards_above 1999 initial_stack = 927 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_above_1999_l531_53132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l531_53115

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := x < -1 ∨ x > 3

-- Theorem statement
theorem f_strictly_increasing :
  ∀ x y, domain x → domain y → x < y → x < -1 → y < -1 → f x < f y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l531_53115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunday_visitors_l531_53155

/-- Proves that the average number of visitors on Sundays is 630 given the conditions -/
theorem sunday_visitors (total_days : ℕ) (non_sunday_visitors : ℕ) (avg_visitors : ℕ) (sunday_visitors : ℕ) : 
  total_days = 30 →
  non_sunday_visitors = 240 →
  avg_visitors = 305 →
  (5 * sunday_visitors + 25 * non_sunday_visitors) / total_days = avg_visitors →
  sunday_visitors = 630 :=
by
  sorry

#check sunday_visitors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunday_visitors_l531_53155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_maps_D_to_circle_l531_53142

-- Define the complex plane
variable (z w : ℂ)

-- Define the original region D
def D (z : ℂ) : Prop :=
  (z.re)^2 + (z.im)^2 - 2*(z.re) = 0

-- Define the transformation function
def f (z : ℂ) : ℂ :=
  3 * z + (1 : ℂ)

-- Define the transformed region
def TransformedRegion (w : ℂ) : Prop :=
  (w.re - 3)^2 + (w.im - 1)^2 ≤ 9

-- Theorem statement
theorem transformation_maps_D_to_circle :
  ∀ z : ℂ, D z → TransformedRegion (f z) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_maps_D_to_circle_l531_53142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l531_53112

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.sin (2 * x + 5)

-- State the theorem
theorem derivative_of_f (x : ℝ) :
  deriv f x = Real.sin (2 * x + 5) + 2 * x * Real.cos (2 * x + 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l531_53112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_row_15_fifth_number_l531_53121

theorem pascal_triangle_row_15_fifth_number :
  let row := λ k => Nat.choose 15 k
  row 0 = 1 ∧ row 1 = 15 → row 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_row_15_fifth_number_l531_53121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_marble_sum_l531_53189

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def pairs : Finset (ℕ × ℕ) := 
  (marbles.product marbles).filter (λ p => p.1 < p.2)

def pair_sum (p : ℕ × ℕ) : ℕ := p.1 + p.2

theorem expected_value_of_marble_sum : 
  (pairs.sum pair_sum) / pairs.card = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_marble_sum_l531_53189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_semicircle_radius_is_24_over_5_l531_53166

/-- Represents a right triangle with an inscribed semicircle -/
structure RightTriangleWithSemicircle where
  /-- Length of side XZ -/
  xz : ℝ
  /-- Length of side YZ -/
  yz : ℝ
  /-- The triangle is right-angled at Z -/
  is_right_angle : True
  /-- A semicircle is inscribed touching XZ and YZ at their midpoints and the hypotenuse XY -/
  has_inscribed_semicircle : True

/-- The radius of the inscribed semicircle in the given right triangle -/
noncomputable def semicircle_radius (t : RightTriangleWithSemicircle) : ℝ :=
  24 / 5

/-- Theorem stating that for a right triangle with sides 15 and 8, 
    the radius of the inscribed semicircle is 24/5 -/
theorem inscribed_semicircle_radius_is_24_over_5 
  (t : RightTriangleWithSemicircle) 
  (h1 : t.xz = 15) 
  (h2 : t.yz = 8) : 
  semicircle_radius t = 24 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_semicircle_radius_is_24_over_5_l531_53166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_areas_l531_53147

theorem circle_triangle_areas (X Y Z : ℝ) : 
  -- Circle circumscribed about a triangle with sides 15, 20, and 25
  15^2 + 20^2 = 25^2 →
  -- X, Y, Z are areas of non-triangular regions
  X > 0 ∧ Y > 0 ∧ Z > 0 →
  -- Z is the largest area
  Z ≥ X ∧ Z ≥ Y →
  -- The circle's diameter is 25
  let r : ℝ := 25 / 2
  -- Z is the area of a semicircle
  Z = π * r^2 / 2 →
  -- X + Y + triangle area = Z
  X + Y + (15 * 20 / 2) = Z :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_areas_l531_53147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_and_extreme_value_l531_53186

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b * Real.log x

noncomputable def g (x : ℝ) : ℝ := (x - 10) / (x - 4)

theorem parallel_tangents_and_extreme_value :
  ∃ (b : ℝ),
    (∀ x : ℝ, x > 0 → HasDerivAt (f b) ((deriv (f b)) x) x) ∧
    (∀ x : ℝ, x ≠ 4 → HasDerivAt g ((deriv g) x) x) ∧
    ((deriv (f b)) 5 = (deriv g) 5) ∧
    (b = -20) ∧
    (∀ x : ℝ, x > 0 → f b x ≥ f b (Real.sqrt 10)) ∧
    (f b (Real.sqrt 10) = 10 - 10 * Real.log 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_and_extreme_value_l531_53186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_exp_inequality_l531_53161

theorem cos_exp_inequality (θ₁ θ₂ : Real) (h1 : 0 < θ₁) (h2 : θ₁ < θ₂) (h3 : θ₂ < π/2) :
  Real.cos θ₂ * Real.exp (Real.cos θ₁) < Real.cos θ₁ * Real.exp (Real.cos θ₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_exp_inequality_l531_53161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_m_value_l531_53145

/-- Given two points A(-2,m) and B(m,4), if the line passing through these points
    is perpendicular to the line 2x+y-1=0, then m = 2. -/
theorem perpendicular_line_m_value (m : ℝ) : 
  (((4 - m) / (m + 2)) * (-2) = -1) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_m_value_l531_53145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l531_53102

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) 
    is 3/2, given that one of its asymptotic lines has the equation y = -√5/2 * x. -/
theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) 
  (h_asymptote : ∀ x : ℝ, ∃ y : ℝ, y = -(Real.sqrt 5)/2 * x ∧ x^2 / a^2 - y^2 / b^2 = 1) : 
  (Real.sqrt (a^2 + b^2)) / a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l531_53102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_four_ninths_l531_53167

/-- The fraction of a semicircle of radius 1 that is shaded when an equilateral triangle
    of height 1 is inscribed in it, and a circle is inscribed in the triangle. -/
noncomputable def shaded_fraction (semicircle_radius : ℝ) (triangle_height : ℝ) : ℝ :=
  let triangle_side := 2 * triangle_height / Real.sqrt 3
  let triangle_area := Real.sqrt 3 / 4 * triangle_side ^ 2
  let inscribed_circle_radius := triangle_side * Real.sqrt 3 / 6
  let inscribed_circle_area := Real.pi * inscribed_circle_radius ^ 2
  let semicircle_area := Real.pi * semicircle_radius ^ 2 / 2
  let shaded_area := semicircle_area - triangle_area - inscribed_circle_area
  shaded_area / semicircle_area

/-- Theorem stating that the shaded fraction is 4/9 when the semicircle radius and triangle height are both 1. -/
theorem shaded_fraction_is_four_ninths :
  shaded_fraction 1 1 = 4 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_four_ninths_l531_53167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_range_l531_53125

/-- Inverse proportion function -/
noncomputable def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := (5 * m - 2) / x

theorem inverse_proportion_m_range
  (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ)
  (h_x : x₁ < x₂ ∧ x₂ < 0)
  (h_y : y₁ < y₂)
  (h_A : inverse_proportion m x₁ = y₁)
  (h_B : inverse_proportion m x₂ = y₂) :
  m < 2/5 := by
  sorry

#check inverse_proportion_m_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_range_l531_53125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_AB_l531_53128

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + (3/5) * t, (4/5) * t)

-- Define the curve C
def curve_C (k : ℝ) : ℝ × ℝ := (4 * k^2, 4 * k)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t k, line_l t = p ∧ curve_C k = p}

-- Theorem statement
theorem length_of_segment_AB :
  let A : ℝ × ℝ := (4, 4)
  let B : ℝ × ℝ := (1/4, -1)
  (A ∈ intersection_points) ∧ (B ∈ intersection_points) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 25/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_AB_l531_53128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l531_53196

theorem right_triangle_hypotenuse (a : ℝ) (θ : ℝ) :
  a > 0 →
  θ = 30 * π / 180 →
  ∃ (b c : ℝ),
    a^2 + b^2 = c^2 ∧
    Real.sin θ = a / c ∧
    a = 12 →
    c = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l531_53196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_translation_range_l531_53159

open Real Set

theorem cosine_translation_range (f g : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = cos (2 * x)) →
  (∀ x, g x = f (x + φ)) →
  (0 < φ) →
  (φ < π / 2) →
  (∀ x ∈ Icc (-π/6) (π/6), StrictMonoOn g (Icc (-π/6) (π/6))) →
  (∃ x ∈ Ioo (-π/6) 0, g x = 0 ∧ ∀ y ∈ Ioo x 0, g y > 0) →
  φ ∈ Ioo (π/4) (π/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_translation_range_l531_53159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l531_53141

theorem integral_inequality (x : ℝ) (hx : x ∈ Set.Ioo 0 1) :
  ∫ y in Set.Icc 0 1, Real.sqrt (1 + (Real.cos y)^2) ≥ Real.sqrt (x^2 + (Real.sin x)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l531_53141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_roots_quadratic_l531_53181

theorem sine_cosine_roots_quadratic (α : Real) (k : Real) : 
  (2 * (Real.sin α)^2 - 4 * k * Real.sin α - 3 * k = 0) ∧ 
  (2 * (Real.cos α)^2 - 4 * k * Real.cos α - 3 * k = 0) → 
  k = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_roots_quadratic_l531_53181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l531_53110

-- Define the points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the circle
def myCircle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Theorem statement
theorem circle_equation :
  myCircle ((M.1 + N.1) / 2, (M.2 + N.2) / 2) (((N.1 - M.1)^2 + (N.2 - M.2)^2).sqrt / 2) =
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l531_53110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l531_53165

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ := 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = 2 * Real.pi / 3)
  (h2 : a = (1, Real.sqrt 3))
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 1) :
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l531_53165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l531_53118

/-- The circle C in the problem -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

/-- A line with slope 1 -/
def line_with_slope_1 (b : ℝ) (x y : ℝ) : Prop := y = x + b

/-- Point O is the origin -/
def origin : ℝ × ℝ := (0, 0)

/-- Two points are perpendicular with respect to the origin -/
def perpendicular_to_origin (A B : ℝ × ℝ) : Prop :=
  (A.1 * B.1 + A.2 * B.2 = 0) ∧ (A ≠ origin) ∧ (B ≠ origin)

/-- Main theorem -/
theorem circle_intersection_theorem :
  ∃! (b₁ b₂ : ℝ), b₁ ≠ b₂ ∧
  (∃ (A₁ B₁ A₂ B₂ : ℝ × ℝ),
    (circle_C A₁.1 A₁.2 ∧ circle_C B₁.1 B₁.2 ∧
     line_with_slope_1 b₁ A₁.1 A₁.2 ∧ line_with_slope_1 b₁ B₁.1 B₁.2 ∧
     perpendicular_to_origin A₁ B₁) ∧
    (circle_C A₂.1 A₂.2 ∧ circle_C B₂.1 B₂.2 ∧
     line_with_slope_1 b₂ A₂.1 A₂.2 ∧ line_with_slope_1 b₂ B₂.1 B₂.2 ∧
     perpendicular_to_origin A₂ B₂)) ∧
  b₁ = -4 ∧ b₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l531_53118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l531_53185

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.exp x - Real.cos x 
  else Real.exp (-x) - Real.cos (-x)

-- State the theorem
theorem solution_set_of_inequality 
  (h_even : ∀ x, f x = f (-x))
  (h_def : ∀ x ≥ 0, f x = Real.exp x - Real.cos x) :
  {x : ℝ | f (x - 1) - 1 < Real.exp Real.pi} = Set.Ioo (1 - Real.pi) (1 + Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l531_53185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_fourth_l531_53109

/-- Given an angle θ whose terminal side passes through the point (1,2) in a Cartesian coordinate system,
    prove that tan(θ + π/4) = -3. -/
theorem tan_theta_plus_pi_fourth (θ : ℝ) (h : Real.tan θ = 2) : Real.tan (θ + Real.pi/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_fourth_l531_53109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vitya_has_winning_strategy_l531_53104

/-- Represents the state of the game as a list of natural numbers, where each number is the size of a pile --/
def GameState := List Nat

/-- Represents a player in the game --/
inductive Player
| Kolya
| Vitya

/-- Represents a move in the game --/
structure Move where
  pileToSplit : Nat
  newPile1 : Nat
  newPile2 : Nat

/-- Checks if a move is valid given the current game state --/
def isValidMove (state : GameState) (move : Move) : Prop :=
  ∃ (i : Nat), i < state.length ∧ 
    state.get? i = some move.pileToSplit ∧
    move.pileToSplit > 1 ∧
    move.newPile1 + move.newPile2 = move.pileToSplit

/-- Applies a move to the current game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over (all piles have size 1) --/
def isGameOver (state : GameState) : Prop :=
  state.all (· = 1)

/-- Represents a strategy for a player --/
def Strategy := GameState → Move

/-- Simulates the game given strategies --/
def gamePlay (initialState : GameState) (kolyaStrategy vityaStrategy : Strategy) (startingPlayer : Player) : Player :=
  sorry

/-- Theorem: Vitya has a winning strategy in the stone splitting game --/
theorem vitya_has_winning_strategy :
  ∃ (strategy : Strategy), 
    ∀ (kolyaStrategy : Strategy),
      let initialState : GameState := [31]
      gamePlay initialState kolyaStrategy strategy Player.Kolya = Player.Vitya :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vitya_has_winning_strategy_l531_53104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_over_cos_l531_53117

theorem sin_double_over_cos (α : ℝ) : 
  Real.sin (π + α) = -1/3 → Real.sin (2*α) / Real.cos α = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_over_cos_l531_53117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l531_53160

noncomputable def f (ω m x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 3) + m - Real.sqrt 3

theorem function_properties (ω m : ℝ) (h_ω_pos : ω > 0) 
  (h_f_0 : f ω m 0 = 2) 
  (h_period : ∃ T > 0, ∀ x, f ω m (x + T) = f ω m x ∧ T = Real.pi) :
  ω = 2 ∧ m = 2 ∧ 
  f ω m (Real.pi / 6) = 2 ∧
  (∃ a : ℝ, a = Real.pi / 12 ∧ ∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a → f ω m x₁ < f ω m x₂) ∧
  (∀ a : ℝ, a > Real.pi / 12 → ∃ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a ∧ f ω m x₁ ≥ f ω m x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l531_53160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l531_53199

/-- The function g defined for positive real numbers x, y, and z -/
noncomputable def g (x y z : ℝ) : ℝ := x^2 / (x^2 + y^2) + y^2 / (y^2 + z^2) + z^2 / (z^2 + x^2)

/-- Theorem stating that g(x,y,z) is always between 1 and 2 for positive real x, y, and z -/
theorem g_bounds (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  1 < g x y z ∧ g x y z < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l531_53199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l531_53119

noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 4) / (4 * x^2 + 7 * x + 3)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, |x| > N → |f x - 3/2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l531_53119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l531_53172

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - 3 * x
def g (x : ℝ) : ℝ := x^2 - 2 * x - 1

-- Theorem statement
theorem problem_solution :
  -- Part 1: If x = 1 is an extremum of f, then a = 1
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 → (x = 1 → (deriv (f a)) x = 0)) →
  (∃ (a : ℝ), a = 1 ∧ ∀ (x : ℝ), x > 0 → (x = 1 → (deriv (f a)) x = 0)) ∧
  -- Part 2: g(x) ≤ f(x) for all x > 0
  (∀ (x : ℝ), x > 0 → g x ≤ f 1 x) ∧
  -- Part 3: (2n+1)² > 4ln(n!) for all natural numbers n
  (∀ (n : ℕ), (2 * ↑n + 1)^2 > 4 * Real.log (Nat.factorial n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l531_53172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l531_53198

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := -x^2 + 5*x - 6

-- Define points A, B, and C
noncomputable def A : ℝ × ℝ := (2, 0)
noncomputable def B : ℝ × ℝ := (3, 0)
noncomputable def C : ℝ × ℝ := (5/2, 1/4)

-- Theorem statement
theorem parabola_properties :
  -- A and B are roots of the parabola
  parabola A.1 = 0 ∧ parabola B.1 = 0 ∧
  -- A is to the left of B
  A.1 < B.1 ∧
  -- C is the vertex of the parabola
  C.1 = 5/2 ∧ C.2 = 1/4 ∧
  -- The area of triangle ABC is 1/8
  (1/2) * (B.1 - A.1) * C.2 = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l531_53198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_function_l531_53143

theorem min_value_trig_function (x : ℝ) :
  x ∈ Set.Icc π (2 * π) →
  ∃ y : ℝ, y = Real.sqrt 3 * Real.sin (x / 2) + Real.cos (x / 2) ∧
    y ≥ -1 ∧
    ∀ z : ℝ, z ∈ Set.Icc π (2 * π) →
      Real.sqrt 3 * Real.sin (z / 2) + Real.cos (z / 2) ≥ -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_function_l531_53143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_swimming_both_days_l531_53108

theorem students_swimming_both_days 
  (total_students : ℕ) 
  (swimming_per_day : ℕ) 
  (football_per_day : ℕ) 
  (swimming_today_swam_yesterday : ℕ) 
  (swimming_today_played_football_yesterday : ℕ) 
  (football_today_swam_yesterday : ℕ) 
  (football_today_played_football_yesterday : ℕ) 
  (set_of_students : Finset ℕ)
  (set_of_swimmers_today : Finset ℕ)
  (set_of_football_players_today : Finset ℕ)
  (set_of_swimmers_yesterday : Finset ℕ)
  (set_of_football_players_yesterday : Finset ℕ) :
  total_students = 33 →
  swimming_per_day = 22 →
  football_per_day = 22 →
  swimming_today_swam_yesterday = 15 →
  swimming_today_played_football_yesterday = 15 →
  football_today_swam_yesterday = 15 →
  football_today_played_football_yesterday = 15 →
  Finset.card set_of_students = total_students →
  Finset.card set_of_swimmers_today = swimming_per_day →
  Finset.card set_of_football_players_today = football_per_day →
  (∀ student, student ∈ set_of_students → 
    (student ∈ set_of_swimmers_today ∨ student ∈ set_of_football_players_today)) →
  (∀ student, student ∈ set_of_students → 
    (student ∈ set_of_swimmers_yesterday ∨ student ∈ set_of_football_players_yesterday)) →
  Finset.card (set_of_swimmers_today ∩ set_of_swimmers_yesterday) = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_swimming_both_days_l531_53108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_extrema_l531_53163

def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 4*x

theorem f_monotonicity_and_extrema :
  ∃ (min_val max_val : ℝ),
    (∀ x < -2, (3*x^2 + 4*x - 4) > 0) ∧
    (∀ x ∈ Set.Ioo (-2 : ℝ) (2/3), (3*x^2 + 4*x - 4) < 0) ∧
    (∀ x > 2/3, (3*x^2 + 4*x - 4) > 0) ∧
    (∀ x ∈ Set.Icc (-5 : ℝ) 0, f x ≥ min_val) ∧
    (∀ x ∈ Set.Icc (-5 : ℝ) 0, f x ≤ max_val) ∧
    min_val = -55 ∧
    max_val = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_extrema_l531_53163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_m_zero_l531_53183

/-- The function f(x) with parameter m -/
noncomputable def f (x m : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + m

/-- The theorem stating that if the maximum value of f(x) on [0, π/2] is 3, then m = 0 -/
theorem max_value_implies_m_zero (m : ℝ) :
  (∃ (max_val : ℝ), max_val = 3 ∧ 
    ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x m ≤ max_val) →
  m = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_m_zero_l531_53183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l531_53144

/-- Represents an infinite sequence of equilateral triangles where each subsequent triangle
    is formed by joining the midpoints of the sides of the previous triangle. -/
noncomputable def TriangleSequence (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => TriangleSequence a n / 2

/-- The sum of the perimeters of all triangles in the sequence. -/
noncomputable def PerimeterSum (a : ℝ) : ℝ :=
  3 * ∑' n, TriangleSequence a n

theorem equilateral_triangle_side_length (a : ℝ) (h : PerimeterSum a = 180) :
  a = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l531_53144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l531_53135

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 7 / (2 * Real.log 2)
noncomputable def c : ℝ := (0.3 : ℝ)^(-(3/2 : ℝ))

-- State the theorem
theorem relationship_abc : c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l531_53135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_percentage_equal_iff_x_eq_6_l531_53137

/-- Atomic mass of Carbon in g/mol -/
noncomputable def C_mass : ℝ := 12.01

/-- Atomic mass of Hydrogen in g/mol -/
noncomputable def H_mass : ℝ := 1.008

/-- Atomic mass of Oxygen in g/mol -/
noncomputable def O_mass : ℝ := 16.00

/-- Mass percentage of Carbon in C6H8O6 -/
noncomputable def C_percentage_C6H8O6 : ℝ := 40.91

/-- Formula for calculating mass percentage of Carbon in CxH8O6 -/
noncomputable def C_percentage_CxH8O6 (x : ℝ) : ℝ :=
  (x * C_mass) / (x * C_mass + 8 * H_mass + 6 * O_mass) * 100

/-- Theorem stating that the mass percentage of C in CxH8O6 is 40.91% iff x = 6 -/
theorem C_percentage_equal_iff_x_eq_6 :
  ∀ x, C_percentage_CxH8O6 x = C_percentage_C6H8O6 ↔ x = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_percentage_equal_iff_x_eq_6_l531_53137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_obtuse_angle_A_l531_53116

/-- Triangle ABC with vertices A(2,1), B(-1,4), and C(5,3) has an obtuse angle at A -/
theorem triangle_ABC_obtuse_angle_A :
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (-1, 4)
  let C : ℝ × ℝ := (5, 3)
  let AB : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC : ℝ := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC : ℝ := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB^2 + AC^2 < BC^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_obtuse_angle_A_l531_53116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_parabola_l531_53129

theorem axis_of_symmetry_parabola (a : ℝ) (h : a ≠ 0) :
  ∃ y : ℝ → ℝ, (y = λ x => a * x^2) ∧
  (∀ x : ℝ, y x = y (-x)) ↔ (λ x : ℝ => -1 / (4 * a)) = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_parabola_l531_53129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_geq_cos_probability_l531_53134

/-- The probability that sin x ≥ cos x when x is uniformly distributed in [0, π] is 3/4 -/
theorem sin_geq_cos_probability : 
  ∀ (P : Set ℝ → ℝ), 
    (∀ a b, a < b → P (Set.Icc a b) = (b - a) / π) →
    P {x ∈ Set.Icc 0 π | Real.sin x ≥ Real.cos x} = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_geq_cos_probability_l531_53134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sixes_three_dice_l531_53182

noncomputable def roll_dice (n : ℕ) : ℝ := n

noncomputable def prob_not_six : ℝ := 5/6

noncomputable def prob_six : ℝ := 1/6

noncomputable def expected_sixes (n : ℕ) : ℝ :=
  (0 : ℝ) * (prob_not_six ^ n) +
  (1 : ℝ) * n * (prob_six * prob_not_six ^ (n-1)) +
  (2 : ℝ) * (n.choose 2) * (prob_six ^ 2 * prob_not_six ^ (n-2)) +
  (3 : ℝ) * (prob_six ^ n)

theorem expected_sixes_three_dice :
  expected_sixes 3 = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sixes_three_dice_l531_53182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l531_53114

/-- The length of the chord formed by the intersection of a line and a circle -/
noncomputable def chord_length (a b c : ℝ) (x₀ y₀ r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - (abs (a*x₀ + b*y₀ + c) / Real.sqrt (a^2 + b^2))^2)

/-- Theorem: The length of the chord formed by the intersection of the line x - 2y - 3 = 0
    and the circle (x-2)² + (y+3)² = 9 is equal to 4 -/
theorem intersection_chord_length :
  chord_length 1 (-2) (-3) 2 (-3) 3 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l531_53114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_length_l531_53177

/-- A rectangular prism with dimensions 2m x 1m x 1m -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_eq : length = 2
  width_eq : width = 1
  height_eq : height = 1

/-- A path in the rectangular prism -/
structure FlyPath (prism : RectangularPrism) where
  length : ℝ
  starts_at_corner : Bool
  visits_all_corners : Bool
  returns_to_start : Bool
  straight_lines : Bool

/-- The theorem stating the maximum path length -/
theorem max_path_length (prism : RectangularPrism) :
  ∃ (path : FlyPath prism),
    path.length = 4 * Real.sqrt 6 + 4 * Real.sqrt 5 ∧
    path.starts_at_corner ∧
    path.visits_all_corners ∧
    path.returns_to_start ∧
    path.straight_lines ∧
    ∀ (other_path : FlyPath prism),
      other_path.starts_at_corner →
      other_path.visits_all_corners →
      other_path.returns_to_start →
      other_path.straight_lines →
      other_path.length ≤ path.length :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_length_l531_53177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_properties_l531_53169

theorem roots_properties (a b : ℝ) (h1 : a^2 - 6*a + 4 = 0) (h2 : b^2 - 6*b + 4 = 0) (h3 : a > b) :
  (a > 0 ∧ b > 0) ∧ (Real.sqrt a - Real.sqrt b) / (Real.sqrt a + Real.sqrt b) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_properties_l531_53169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_parabola_to_line_l531_53191

/-- The parabola y² = 6x -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = 6 * p.1}

/-- The line 3x - 4y + 12 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p | 3 * p.1 - 4 * p.2 + 12 = 0}

/-- The shortest distance from a point to a line -/
noncomputable def shortestDistance (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem shortest_distance_parabola_to_line :
  ∀ M ∈ Parabola, shortestDistance M Line = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_parabola_to_line_l531_53191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l531_53156

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x) / Real.sqrt (9 - x^2)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (-x^2 - 6*x - 5)

-- Theorem for the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -3 < x ∧ x < 0 ∨ 2 < x ∧ x < 3} :=
by sorry

-- Theorem for the range of g
theorem range_of_g :
  Set.range g = Set.Icc 0 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l531_53156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_pyramid_height_l531_53105

/-- Represents a pyramid with an isosceles right triangle base -/
structure IsoscelesRightPyramid where
  leg : ℝ
  height : ℝ
  volume : ℝ

/-- Calculates the volume of an isosceles right pyramid -/
noncomputable def pyramidVolume (p : IsoscelesRightPyramid) : ℝ :=
  (1/3) * (1/2 * p.leg * p.leg) * p.height

theorem isosceles_right_pyramid_height
  (p : IsoscelesRightPyramid)
  (h_leg : p.leg = 3)
  (h_volume : p.volume = 6)
  (h_volume_calc : p.volume = pyramidVolume p) :
  p.height = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_pyramid_height_l531_53105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_seat_number_l531_53120

def systematic_sample (total : ℕ) (sample_size : ℕ) (known_seats : List ℕ) : Prop :=
  sample_size > 0 ∧
  total > 0 ∧
  sample_size ≤ total ∧
  known_seats.length < sample_size ∧
  ∀ i j, i < j → j < known_seats.length → known_seats.get! i < known_seats.get! j

theorem fourth_seat_number
  (total : ℕ)
  (sample_size : ℕ)
  (known_seats : List ℕ)
  (h_systematic : systematic_sample total sample_size known_seats)
  (h_total : total = 52)
  (h_sample_size : sample_size = 4)
  (h_known_seats : known_seats = [6, 32, 45]) :
  ∃ (fourth_seat : ℕ), fourth_seat = 19 ∧ (known_seats ++ [fourth_seat]).Perm [6, 19, 32, 45] := by
  sorry

#check fourth_seat_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_seat_number_l531_53120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_solutions_l531_53111

def vector_a : ℝ × ℝ := (2, -1)

theorem vector_b_solutions (b : ℝ × ℝ) :
  (‖vector_a + b‖ = 1) ∧  -- magnitude of sum is 1
  ((vector_a + b).1 = 0)  -- sum is parallel to y-axis (x-component is 0)
  →
  b = (-2, 2) ∨ b = (-2, 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_solutions_l531_53111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_sum_T_bound_l531_53180

noncomputable def sequence_a (n : ℕ+) : ℝ := 3^(n.val - 1)

noncomputable def sum_S (n : ℕ+) : ℝ := (3/2) * sequence_a n - 1/2

noncomputable def sequence_b (n : ℕ+) : ℝ := (2 * n.val : ℝ) / (sequence_a (n + 2) - sequence_a (n + 1))

noncomputable def sum_T (n : ℕ+) : ℝ := (Finset.range n.val).sum (λ i => sequence_b ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_a_formula (n : ℕ+) : sequence_a n = 3^(n.val - 1) := by
  -- The proof is trivial since it's the definition
  rfl

theorem sum_T_bound (n : ℕ+) : sum_T n < 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_sum_T_bound_l531_53180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l531_53150

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Check if a point lies on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Check if a point lies on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.c

/-- Check if a point lies on a circle -/
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The right focus of an ellipse -/
noncomputable def right_focus (e : Ellipse) : Point :=
  { x := Real.sqrt (e.a^2 - e.b^2), y := 0 }

theorem ellipse_eccentricity_special_case (e : Ellipse) 
  (l : Line) (A B : Point) (c : Circle) :
  l.m = -Real.sqrt 3 ∧ 
  on_ellipse A e ∧ 
  on_ellipse B e ∧
  on_line A l ∧
  on_line B l ∧
  on_circle (right_focus e) c ∧
  c.radius = Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) / 2 →
  eccentricity e = Real.sqrt 3 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l531_53150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_at_12_l531_53149

/-- An arithmetic sequence with properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  a1_positive : a 1 > 0
  condition : 3 * a 5 = 5 * a 8

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem sum_max_at_12 (seq : ArithmeticSequence) :
  ∀ n : ℕ, S seq 12 ≥ S seq n := by
  sorry

#check sum_max_at_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_at_12_l531_53149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l531_53187

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Ioo 1 2, x^2 - abs a * x + a - 1 > 0) → a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l531_53187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_factor_formula_l531_53107

/-- The shape factor of a cylindrical building -/
noncomputable def shape_factor (R H : ℝ) : ℝ :=
  let F₀ : ℝ := Real.pi * R^2 + 2 * Real.pi * R * H  -- exposed area
  let V₀ : ℝ := Real.pi * R^2 * H              -- volume
  F₀ / V₀

/-- Theorem: The shape factor of a cylindrical building equals (2H + R) / (HR) -/
theorem shape_factor_formula (R H : ℝ) (h₁ : R > 0) (h₂ : H > 0) :
  shape_factor R H = (2 * H + R) / (H * R) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_factor_formula_l531_53107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l531_53157

-- Define the triangle
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  -- Z is a right angle (90°)
  (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0 ∧
  -- Angle X is 60°
  ((Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2)) / 
    (Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) * Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2)) = 1/2 ∧
  -- Length of XZ is 6
  Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2) = 6

-- Theorem statement
theorem triangle_side_length (X Y Z : ℝ × ℝ) :
  Triangle X Y Z → Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) = 12 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l531_53157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_closest_to_one_l531_53101

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

theorem ratio_closest_to_one :
  let largest_element := (2 : ℝ)^12
  let sum_others := geometric_sum 2 2 11
  let ratio := largest_element / sum_others
  ∀ i ∈ ({0, 1, 2, 3, 4} : Set ℕ), |ratio - 1| ≤ |ratio - (i : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_closest_to_one_l531_53101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_droplet_size_in_scientific_notation_l531_53194

noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

theorem droplet_size_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    0.000005 = scientific_notation a n ∧ 
    1 ≤ |a| ∧ 
    |a| < 10 ∧
    a = 5 ∧
    n = -6 := by
  use 5, -6
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_droplet_size_in_scientific_notation_l531_53194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_implies_y_neg_one_l531_53152

/-- The distance between two points in 3D space -/
noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- Theorem: If a point M(0, y, 0) on the y-axis has equal distances to 
    points P(1, 0, 2) and Q(1, -3, 1), then y = -1 -/
theorem equal_distance_implies_y_neg_one :
  ∀ y : ℝ, distance 0 y 0 1 0 2 = distance 0 y 0 1 (-3) 1 → y = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_implies_y_neg_one_l531_53152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_f_g_l531_53153

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem min_difference_f_g :
  ∃ (min_val : ℝ), min_val = 1/2 + 1/2 * Real.log 2 ∧
  ∀ (x : ℝ), x > 0 → |f x - g x| ≥ min_val :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_f_g_l531_53153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l531_53124

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.log (a * x + b) + x^2

-- State the theorem
theorem tangent_line_and_inequality (a b : ℝ) (h1 : a ≠ 0) :
  (∀ x, f a b x ≤ x^2 + x) →
  (∃ k, ∀ x, f a b x = f a b 1 + (x - 1) * k) →
  (a = -1 ∧ b = 2) ∧ (∀ a b : ℝ, a > 0 → b ≤ a - a * Real.log a → a * b ≤ Real.exp 1 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l531_53124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_distance_in_hexagon_l531_53136

/-- A regular hexagon with side length 2 -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : ∀ i : Fin 6, dist (vertices i) (vertices ((i + 1) % 6)) = 2

/-- A parabola passing through four points of the hexagon -/
structure Parabola (hexagon : RegularHexagon) where
  equation : ℝ → ℝ → Prop
  passes_through : ∀ (p : ℝ × ℝ) (i : Fin 4), equation p.1 p.2 ↔ p = (hexagon.vertices i)

/-- The distance from the focus of a parabola to its directrix -/
noncomputable def focal_distance (hexagon : RegularHexagon) (p : Parabola hexagon) : ℝ := sorry

theorem parabola_focal_distance_in_hexagon 
  (hexagon : RegularHexagon) 
  (parabola : Parabola hexagon) : 
  focal_distance hexagon parabola = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_distance_in_hexagon_l531_53136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_morning_emails_l531_53188

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 5

/-- The total number of emails Jack received during the day -/
def total_emails : ℕ := 8

/-- Theorem stating that the number of emails Jack received in the morning is 3 -/
theorem jack_morning_emails : morning_emails = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_morning_emails_l531_53188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radians_to_degrees_conversion_l531_53127

/-- Conversion factor from radians to degrees -/
noncomputable def π_in_degrees : ℝ := 180

/-- The given angle in radians -/
noncomputable def angle_in_radians : ℝ := 4 * Real.pi / 3

/-- The angle converted to degrees -/
def angle_in_degrees : ℝ := 240

/-- Theorem stating that the conversion from radians to degrees is correct -/
theorem radians_to_degrees_conversion :
  (angle_in_radians * π_in_degrees) / Real.pi = angle_in_degrees := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radians_to_degrees_conversion_l531_53127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l531_53190

noncomputable def f (x : ℝ) : ℝ := (3*x + 4) / (x + 3)

theorem f_properties :
  let R := {y : ℝ | ∃ x : ℝ, x ≥ 0 ∧ f x = y}
  ∃ N n : ℝ,
    (∀ y ∈ R, y ≤ N) ∧
    (∀ y ∈ R, n ≤ y) ∧
    (n ∈ R) ∧
    (N ∉ R) ∧
    N = 3 ∧
    n = 4/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l531_53190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mooncake_min_avg_sales_l531_53151

/-- The sales function for mooncakes -/
noncomputable def f (t : ℝ) : ℝ := t^2 + 10*t + 16

/-- The average sales function for mooncakes -/
noncomputable def avg_sales (t : ℝ) : ℝ := f t / t

/-- The minimum average sales of mooncakes -/
def min_avg_sales : ℝ := 16

/-- Theorem stating that the minimum average sales is 16 -/
theorem mooncake_min_avg_sales :
  ∀ t : ℝ, 0 < t → t ≤ 30 → avg_sales t ≥ min_avg_sales :=
by
  sorry

#check mooncake_min_avg_sales

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mooncake_min_avg_sales_l531_53151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_route_time_l531_53103

/-- Represents a point in the postman's route -/
inductive Point
| PostOffice
| A | B | C | D | E | F | G | H | I | J | K | L
deriving DecidableEq

/-- The time in minutes to travel between adjacent points -/
def segment_time : ℕ := 10

/-- The total number of points including the post office -/
def total_points : ℕ := 13

/-- A route is a list of points -/
def Route := List Point

/-- Calculate the time taken for a given route -/
def route_time (r : Route) : ℕ :=
  (r.length - 1) * segment_time

/-- A valid route starts and ends at the post office and visits all points -/
def is_valid_route (r : Route) : Prop :=
  r.head? = some Point.PostOffice ∧
  r.getLast? = some Point.PostOffice ∧
  r.toFinset.card = total_points

theorem min_route_time :
  ∃ (r : Route), is_valid_route r ∧ 
    (∀ (r' : Route), is_valid_route r' → route_time r ≤ route_time r') ∧
    route_time r = 4 * 60 := by
  sorry

#check min_route_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_route_time_l531_53103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mean_BC_l531_53176

/-- Represents a pile of rocks -/
structure RockPile where
  weight : ℝ  -- Total weight of the pile
  count : ℝ   -- Number of rocks in the pile (using ℝ for simplicity)

/-- Calculates the mean weight of a pile of rocks -/
noncomputable def mean_weight (pile : RockPile) : ℝ := pile.weight / pile.count

/-- Combines two piles of rocks -/
def combine_piles (pile1 pile2 : RockPile) : RockPile :=
  { weight := pile1.weight + pile2.weight,
    count := pile1.count + pile2.count }

/-- The theorem to be proved -/
theorem max_mean_BC (pile_A pile_B pile_C : RockPile)
  (hA : mean_weight pile_A = 30)
  (hB : mean_weight pile_B = 70)
  (hAB : mean_weight (combine_piles pile_A pile_B) = 50)
  (hAC : mean_weight (combine_piles pile_A pile_C) = 40) :
  ∃ (n : ℕ), n ≤ 80 ∧ 
    ∀ (m : ℕ), mean_weight (combine_piles pile_B pile_C) ≤ ↑m → m ≤ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mean_BC_l531_53176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_increase_approximation_l531_53179

/-- The length of the side of the first equilateral triangle -/
def initial_side_length : ℝ := 3

/-- The scale factor for each subsequent triangle -/
def scale_factor : ℝ := 1.2

/-- The number of triangles -/
def num_triangles : ℕ := 5

/-- The area of an equilateral triangle given its side length -/
noncomputable def triangle_area (side_length : ℝ) : ℝ := (Real.sqrt 3 / 4) * side_length^2

/-- The side length of the nth triangle -/
noncomputable def nth_triangle_side (n : ℕ) : ℝ := initial_side_length * scale_factor^(n - 1)

/-- The percent increase in area from the first to the fifth triangle -/
noncomputable def percent_increase : ℝ :=
  let first_area := triangle_area initial_side_length
  let fifth_area := triangle_area (nth_triangle_side num_triangles)
  (fifth_area - first_area) / first_area * 100

theorem area_increase_approximation :
  abs (percent_increase - 329.98) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_increase_approximation_l531_53179
