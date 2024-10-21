import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_l385_38583

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 15 * x + 8 * y = 120

-- Define the triangle vertices
def triangle_vertices : Set (ℝ × ℝ) := {(0, 0), (8, 0), (0, 15)}

-- Define the function to calculate the altitude from (0,0) to the line
noncomputable def altitude_to_line (a b c : ℝ) : ℝ := |c| / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem sum_of_altitudes :
  ∀ (triangle : Set (ℝ × ℝ)),
    triangle = triangle_vertices →
    (∀ (x y : ℝ), (x, y) ∈ triangle → line_equation x y) →
    (8 : ℝ) + 15 + altitude_to_line 15 8 120 = 511 / 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_l385_38583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_isosceles_l385_38515

/-- A triangle with the given properties is isosceles -/
theorem triangle_is_isosceles (a b c : ℝ) (angle_A : ℝ) :
  angle_A = 30 * π / 180 →  -- 30° angle in radians
  a + b + c = 36 →  -- perimeter is 36 cm
  a = 9 →  -- side opposite to 30° angle is 9 cm
  b = c :=  -- two sides are equal, making it isosceles
by
  sorry  -- Proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_isosceles_l385_38515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_approx_21_88_percent_l385_38576

-- Define the revenue amounts in billions of dollars
noncomputable def transaction_last_year : ℝ := 40.0
noncomputable def transaction_this_year : ℝ := 28.8
noncomputable def data_last_year : ℝ := 25.0
noncomputable def data_this_year : ℝ := 20.0
noncomputable def cross_border_last_year : ℝ := 20.0
noncomputable def cross_border_this_year : ℝ := 17.6

-- Define the total revenue for last year and this year
noncomputable def total_last_year : ℝ := transaction_last_year + data_last_year + cross_border_last_year
noncomputable def total_this_year : ℝ := transaction_this_year + data_this_year + cross_border_this_year

-- Define the percentage decrease
noncomputable def percentage_decrease : ℝ := (total_last_year - total_this_year) / total_last_year * 100

-- Theorem statement
theorem revenue_decrease_approx_21_88_percent :
  abs (percentage_decrease - 21.88) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_decrease_approx_21_88_percent_l385_38576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l385_38536

noncomputable def f (x : ℝ) : ℝ := (3 + 5 * Real.sin x) / Real.sqrt (5 + 4 * Real.cos x + 3 * Real.sin x)

theorem f_range : ∀ x : ℝ, -4/5 * Real.sqrt 10 < f x ∧ f x ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l385_38536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_d_positive_l385_38595

/-- Given a geometric sequence {c_n} with positive terms and common ratio r,
    we define d_n as n * (c_1 * r^((n-1)/2))^n.
    This theorem states that {d_n} is also a geometric sequence. -/
theorem geometric_sequence_property (c : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, c n > 0) 
  (h_geom : ∀ n, c (n + 1) = c n * r) :
  ∃ q : ℝ, ∀ n : ℕ, 
    n * (c 1 * r^((n - 1) / 2))^n = 
    (1 : ℕ) * (c 1 * r^((1 - 1) / 2))^1 * q^(n - 1) :=
sorry

/-- This function defines d_n as specified in the problem -/
def d (c : ℕ → ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  n * (c 1 * r^((n - 1) / 2))^n

/-- This theorem states that d_n is positive when c_n is positive -/
theorem d_positive (c : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, c n > 0) (n : ℕ) :
  d c r n > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_d_positive_l385_38595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_problem_l385_38588

def is_systematic_sample (s : List Nat) : Prop :=
  s.length = 6 ∧ 
  s.all (· ≤ 60) ∧
  ∃ k, ∀ i, i < s.length - 1 → s[i + 1]! = s[i]! + k

def option_a : List Nat := [6, 12, 18, 24, 30, 36]
def option_b : List Nat := [2, 4, 8, 16, 32, 60]
def option_c : List Nat := [3, 12, 23, 34, 43, 53]
def option_d : List Nat := [5, 15, 25, 35, 45, 55]

theorem systematic_sampling_problem :
  is_systematic_sample option_d ∧
  ¬is_systematic_sample option_a ∧
  ¬is_systematic_sample option_b ∧
  ¬is_systematic_sample option_c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_problem_l385_38588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_is_pythagorean_triple_l385_38507

-- Define a function to check if three numbers form a Pythagorean triple
def isPythagoreanTriple (a b c : ℚ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

-- Define the given groups of numbers
def groupA : List ℚ := [12, 15, 18]
def groupB : List ℚ := [3/10, 4/10, 5/10]
def groupC : List ℚ := [3/2, 3, 5/2]
def groupD : List ℚ := [12, 16, 20]

-- Theorem stating that only groupD is a Pythagorean triple
theorem only_D_is_pythagorean_triple :
  (¬ isPythagoreanTriple groupA[0]! groupA[1]! groupA[2]!) ∧
  (¬ isPythagoreanTriple groupB[0]! groupB[1]! groupB[2]!) ∧
  (¬ isPythagoreanTriple groupC[0]! groupC[2]! groupC[1]!) ∧
  isPythagoreanTriple groupD[0]! groupD[1]! groupD[2]! :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_is_pythagorean_triple_l385_38507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progressions_do_not_exhaust_naturals_l385_38577

/-- A geometric progression with integer terms -/
structure GeometricProgression where
  first_term : ℤ
  common_ratio : ℤ

/-- The set of all terms in a geometric progression -/
def GeometricProgression.terms (gp : GeometricProgression) : Set ℤ :=
  {n : ℤ | ∃ k : ℕ, n = gp.first_term * gp.common_ratio ^ k}

theorem geometric_progressions_do_not_exhaust_naturals 
  (progressions : Finset GeometricProgression)
  (h_count : progressions.card = 1975) :
  ∃ n : ℕ, ∀ gp ∈ progressions, (n : ℤ) ∉ gp.terms :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progressions_do_not_exhaust_naturals_l385_38577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l385_38541

def arithmeticSequence (n : ℕ) : ℝ := 2 * n + 5

theorem arithmetic_sequence_proof :
  ∀ n : ℕ, arithmeticSequence (n + 1) - arithmeticSequence n = 2 :=
by
  intro n
  simp [arithmeticSequence]
  ring

#check arithmetic_sequence_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l385_38541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l385_38571

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega_value (ω : ℝ) (φ : ℝ) (T : ℝ) :
  ω > 0 →
  -π / 2 < φ →
  φ < π / 2 →
  T > 0 →
  T = 2 * π / ω →
  f ω φ T = Real.sqrt 3 / 2 →
  (∀ x : ℝ, f ω φ (x - π / 6) = f ω φ (-x - π / 6)) →
  ω ≥ 5 ∧ (∀ ω' : ℝ, ω' ≥ 5 → ω ≤ ω') :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l385_38571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_not_in_first_quadrant_l385_38559

theorem function_not_in_first_quadrant (x : ℝ) : x > 0 → (1/2 : ℝ)^x - 2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_not_in_first_quadrant_l385_38559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l385_38586

theorem triangle_angle_proof (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < Real.pi) ∧ (0 < B) ∧ (B < Real.pi) ∧ (0 < C) ∧ (C < Real.pi) →  -- Angle conditions
  (A + B + C = Real.pi) →  -- Sum of angles in a triangle
  (a > 0) ∧ (b > 0) ∧ (c > 0) →  -- Side length conditions
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →  -- Law of sines
  (2 * b * Real.cos B = a * Real.cos C + c * Real.cos A) →  -- Given condition
  B = Real.pi / 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l385_38586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_circumference_l385_38561

/-- The circumference of the smaller circle given two circles where:
    1. The larger circle has a circumference of 704 meters
    2. The difference between the areas of the larger and smaller circles is 17254.942310250928 square meters
-/
theorem smaller_circle_circumference :
  ∀ (r R : ℝ),
    R > r ∧ 
    2 * Real.pi * R = 704 ∧
    Real.pi * R^2 - Real.pi * r^2 = 17254.942310250928 →
    |2 * Real.pi * r - 527.08| < 0.01 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_circumference_l385_38561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_ratio_proof_l385_38546

/-- The daily wage of a man in rupees -/
def man_wage : ℕ := 350

/-- The daily wage of a woman in rupees -/
def woman_wage : ℕ := 200

/-- The total number of men in the first case -/
def men_count1 : ℕ := 24

/-- The total number of women in the first case -/
def women_count1 : ℕ := 16

/-- The total number of women in the second case -/
def women_count2 : ℕ := 37

/-- The total daily wages in rupees -/
def total_wages : ℕ := 11600

/-- The ratio of men to women in the second case -/
def men_women_ratio : ℚ := 12 / 37

theorem wage_ratio_proof :
  (men_count1 * man_wage + women_count1 * woman_wage = total_wages) →
  (∃ (men_count2 : ℕ), men_count2 * man_wage + women_count2 * woman_wage = total_wages) →
  (∃ (men_count2 : ℕ), (men_count2 : ℚ) / women_count2 = men_women_ratio) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_ratio_proof_l385_38546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_proof_l385_38522

/-- Given a trapezoid ABCD with the following properties:
  - A is at (3, 0)
  - E(6, -1) is the midpoint of AB
  - F(7, 2) is the midpoint of CD
  - BC is parallel to the y-axis
Prove that:
  1. The trapezoid is isosceles (BC = AD)
  2. The angle DAB = arccos(1/√10)
-/
theorem isosceles_trapezoid_proof (A B C D E F : ℝ × ℝ) : 
  A = (3, 0) →
  E = (6, -1) →
  F = (7, 2) →
  E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  F = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  C.1 = B.1 →
  (B.1 - A.1 : ℝ) * (D.2 - C.2) = (D.1 - C.1) * (B.2 - A.2) →
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AD := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let angle_DAB := Real.arccos ((D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2)) / (AD * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))
  BC = AD ∧ angle_DAB = Real.arccos (1 / Real.sqrt 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_proof_l385_38522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l385_38529

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, (2 : ℝ)^x = 5) ↔ (∃ x : ℝ, (2 : ℝ)^x ≠ 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l385_38529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l385_38506

theorem no_valid_n : ¬∃ n : ℕ, (1000 ≤ n / 5 ∧ n / 5 ≤ 9999) ∧ (1000 ≤ 5 * n ∧ 5 * n ≤ 9999) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_n_l385_38506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_return_probability_l385_38526

/-- Probability of returning to the starting vertex after n moves -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/2 * (1 - P n)

/-- The probability of returning to the starting vertex after 18 moves on an equilateral triangle -/
theorem bug_return_probability : ∃ (a b : ℕ), 
  (Nat.Coprime a b) ∧ 
  (P 18 = a / b) := by
  sorry

#eval P 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_return_probability_l385_38526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l385_38528

theorem integral_inequality : 
  let S₁ := ∫ x in (1:ℝ)..2, x
  let S₂ := ∫ x in (1:ℝ)..2, Real.exp x
  let S₃ := ∫ x in (1:ℝ)..2, x^2
  S₁ < S₃ ∧ S₃ < S₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l385_38528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_band_arrangement_possibilities_l385_38544

theorem band_arrangement_possibilities : 
  ∃! n : ℕ, n = (Finset.filter (λ x => 4 ≤ x ∧ x ≤ 15 ∧ 90 % x = 0) (Finset.range 16)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_band_arrangement_possibilities_l385_38544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l385_38511

def A : Finset Nat := {1, 2, 3}
def B : Finset Nat := {2, 4, 5}

theorem union_cardinality : Finset.card (A ∪ B) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_cardinality_l385_38511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l385_38518

-- Define the basic types
variable (Point : Type) -- Type for points
variable (Line : Type) -- Type for lines
variable (Plane : Type) -- Type for planes

-- Define the relations
variable (parallel_lines : Line → Line → Prop) -- Parallel relation for lines
variable (parallel_line_plane : Line → Plane → Prop) -- Parallel relation between line and plane
variable (contained_in_plane : Line → Plane → Prop) -- Containment relation of line in plane

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (α : Plane)
  (h1 : parallel_lines a b)
  (h2 : parallel_line_plane b α) :
  parallel_line_plane a α ∨ contained_in_plane a α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relationship_l385_38518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_is_nine_l385_38574

/-- Represents a 3x3 grid of integers -/
def Grid := Matrix (Fin 3) (Fin 3) ℕ

/-- Checks if two positions in the grid share an edge -/
def shares_edge (i j i' j' : Fin 3) : Prop :=
  (i = i' ∧ |j - j'| = 1) ∨ (j = j' ∧ |i - i'| = 1)

/-- Defines a valid grid configuration according to the problem conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ n : ℕ, n ∈ Finset.range 9 → ∃ i j, g i j = n + 1) ∧
  (∀ i j i' j', g i j + 1 = g i' j' → shares_edge i j i' j') ∧
  (g 0 0 + g 0 2 + g 2 0 + g 2 2 = 24) ∧
  (g 0 0 + g 1 1 + g 2 2 = 24) ∧
  (g 0 2 + g 1 1 + g 2 0 = 24)

theorem center_is_nine (g : Grid) (h : valid_grid g) : g 1 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_is_nine_l385_38574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_golden_ratio_width_l385_38578

/-- The golden ratio's conjugate -/
noncomputable def goldenRatioConjugate : ℝ := (Real.sqrt 5 - 1) / 2

/-- Calculates the width of a book given its length and the golden ratio's conjugate -/
noncomputable def bookWidth (length : ℝ) : ℝ := goldenRatioConjugate * length

theorem book_golden_ratio_width :
  let length : ℝ := 14
  bookWidth length = 7 * Real.sqrt 5 - 7 := by
  sorry

#check book_golden_ratio_width

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_golden_ratio_width_l385_38578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l385_38527

-- Define the constant
noncomputable def α : ℝ := 2 + Real.sqrt 3

-- Define x
noncomputable def x : ℝ := α ^ 1000

-- Define n as the floor of x
noncomputable def n : ℤ := ⌊x⌋

-- Define f
noncomputable def f : ℝ := x - n

-- Theorem statement
theorem x_times_one_minus_f_equals_one : x * (1 - f) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l385_38527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l385_38520

/-- Calculates the speed of a train in km/h given its length, the platform length, and the time to cross. -/
noncomputable def train_speed (train_length platform_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / time
  3.6 * speed_ms

/-- Theorem stating that a train with given parameters has a specific speed. -/
theorem train_speed_calculation :
  let train_length : ℝ := 200
  let platform_length : ℝ := 288.928
  let crossing_time : ℝ := 22
  let calculated_speed := train_speed train_length platform_length crossing_time
  ∃ ε > 0, |calculated_speed - 79.9664| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l385_38520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosing_polygons_sides_l385_38570

/-- The number of sides of the inner regular polygon -/
def m : ℕ := 8

/-- The number of sides of each enclosing regular polygon -/
def n : ℕ := 8

/-- The interior angle of a regular polygon with k sides -/
noncomputable def interiorAngle (k : ℕ) : ℝ :=
  (k - 2 : ℝ) * 180 / k

/-- The exterior angle of a regular polygon with k sides -/
noncomputable def exteriorAngle (k : ℕ) : ℝ :=
  360 / k

/-- The theorem stating that for a regular octagon enclosed by 8 regular n-sided polygons,
    where the vertices match precisely, n must equal 8 -/
theorem enclosing_polygons_sides : exteriorAngle m = exteriorAngle n → n = m := by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosing_polygons_sides_l385_38570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_springfield_market_deal_savings_l385_38594

/-- Represents the hat deal at Springfield Market -/
structure HatDeal where
  regularPrice : ℚ
  secondHatDiscount : ℚ
  thirdHatDiscount : ℚ

/-- Calculates the total cost for three hats under the deal -/
def totalCostWithDeal (deal : HatDeal) : ℚ :=
  deal.regularPrice + 
  (deal.regularPrice * (1 - deal.secondHatDiscount)) + 
  (deal.regularPrice * (1 - deal.thirdHatDiscount))

/-- Calculates the percentage saved under the deal -/
def percentageSaved (deal : HatDeal) : ℚ :=
  (3 * deal.regularPrice - totalCostWithDeal deal) / (3 * deal.regularPrice) * 100

/-- Theorem: The percentage saved when buying three hats under the Springfield Market deal is 25% -/
theorem springfield_market_deal_savings : 
  let deal : HatDeal := { 
    regularPrice := 60,
    secondHatDiscount := 1/4,
    thirdHatDiscount := 1/2
  }
  percentageSaved deal = 25 := by
  sorry

#eval percentageSaved { regularPrice := 60, secondHatDiscount := 1/4, thirdHatDiscount := 1/2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_springfield_market_deal_savings_l385_38594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cart_max_speed_l385_38547

/-- The maximum speed of a cart on a circular track -/
noncomputable def max_speed (R a : ℝ) : ℝ :=
  ((16 * Real.pi ^ 2 * R ^ 2 * a ^ 2) / (16 * Real.pi ^ 2 + 1)) ^ (1/4)

/-- Theorem: The maximum speed of the cart under given conditions -/
theorem cart_max_speed :
  let R : ℝ := 4
  let a : ℝ := 2
  let v_max := max_speed R a
  0 < v_max ∧ v_max < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cart_max_speed_l385_38547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l385_38517

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 - 8 + 2 * x

-- State the theorem
theorem zero_in_interval :
  (f 3 < 0) → (f 4 > 0) → ∃ x : ℝ, x ∈ Set.Ioo 3 4 ∧ f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l385_38517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_formula_l385_38549

/-- Represents a rectangular parallelepiped with given properties -/
structure RectParallelepiped where
  α : ℝ  -- angle AMB
  β : ℝ  -- angle BMB₁
  b : ℝ  -- length of B₁M

/-- The lateral surface area of the rectangular parallelepiped -/
noncomputable def lateral_surface_area (p : RectParallelepiped) : ℝ :=
  2 * p.b^2 * Real.sqrt 2 * Real.sin (2 * p.β) * Real.sin ((2 * p.α + Real.pi) / 4)

/-- Theorem stating the lateral surface area of the rectangular parallelepiped -/
theorem lateral_surface_area_formula (p : RectParallelepiped) :
  lateral_surface_area p = 2 * p.b^2 * Real.sqrt 2 * Real.sin (2 * p.β) * Real.sin ((2 * p.α + Real.pi) / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_formula_l385_38549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l385_38539

-- Define the circle
def our_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define the property of P being the midpoint of chord AB
def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define a line equation
def line_equation (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Theorem statement
theorem chord_equation (A B : ℝ × ℝ) :
  (∃ (x y : ℝ), our_circle x y ∧ ((x, y) = A ∨ (x, y) = B)) →
  is_midpoint P A B →
  (∃ (x y : ℝ), line_equation 1 (-1) (-3) x y ∧ ((x, y) = A ∨ (x, y) = B)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l385_38539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_arithmetic_progression_inradius_l385_38519

/-- A right triangle with sides in arithmetic progression has the common difference
    equal to the radius of its inscribed circle. -/
theorem right_triangle_arithmetic_progression_inradius 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_arithmetic : ∃ (d : ℝ), b = a + d ∧ c = a + 2*d) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  (c - a) / 2 = (a + b - c) / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_arithmetic_progression_inradius_l385_38519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_exponential_l385_38590

-- Define the exponential function
def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = a^x

-- Define the specific function f(x) = 3^x
noncomputable def f (x : ℝ) : ℝ := 3^x

-- Theorem statement
theorem f_is_exponential : is_exponential_function f := by
  -- Provide the value of a
  use 3
  -- Prove the three conditions
  constructor
  · -- Prove 3 > 0
    norm_num
  constructor
  · -- Prove 3 ≠ 1
    norm_num
  · -- Prove ∀ x, f x = 3^x
    intro x
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_exponential_l385_38590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_catch_ace_l385_38503

/-- The distance Flash must run to catch Ace -/
noncomputable def catch_up_distance (v_f : ℝ) (z w d : ℝ) : ℝ :=
  (d + w * z) / (z - 1)

/-- Theorem stating the distance Flash must run to catch Ace -/
theorem flash_catch_ace 
  (v_f : ℝ) -- Flash's speed
  (z : ℝ) -- Factor by which Ace is slower than Flash
  (w : ℝ) -- Ace's head start distance
  (d : ℝ) -- Flash's waiting time
  (h_z : z > 1) -- Condition that z > 1
  (h_v_f : v_f > 0) -- Assumption that Flash's speed is positive
  : 
  let v_a := v_f / z -- Ace's speed
  let t := (d + w * z) / (v_f * (z - 1)) -- Time for Flash to catch Ace
  v_f * t = catch_up_distance v_f z w d ∧ -- Flash's distance equals the catch-up distance
  v_a * (t + d) + w = v_f * t -- Ace's total distance equals Flash's distance
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_catch_ace_l385_38503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equidistant_point_l385_38592

/-- A parabola with equation y^2 = 4x -/
structure Parabola where
  C : Set (ℝ × ℝ)
  eq : ∀ (x y : ℝ), (x, y) ∈ C ↔ y^2 = 4*x

/-- The focus of a parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: For a parabola y^2 = 4x, if a point A on the parabola is equidistant
    from the focus F and the point B(3,0), then |AB| = 2√2 -/
theorem parabola_equidistant_point (p : Parabola) (A : ℝ × ℝ) 
    (h1 : A ∈ p.C) 
    (h2 : distance A focus = distance focus (3, 0)) :
  distance A (3, 0) = Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equidistant_point_l385_38592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l385_38530

theorem matrix_scalar_multiplication (u : Fin 2 → ℝ) : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 0], ![0, 3]]
  N.mulVec u = (3 : ℝ) • u := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l385_38530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_travel_time_relation_l385_38501

/-- Represents the speed of a canoe in still water -/
noncomputable def canoe_speed : ℝ → ℝ := λ _ => 1

/-- Represents the speed of the river current -/
noncomputable def river_speed : ℝ → ℝ := λ _ => 0.5

/-- Represents the distance between villages A and B -/
noncomputable def distance : ℝ → ℝ := λ _ => 1

/-- The time to travel from A to B -/
noncomputable def time_A_to_B (v r d : ℝ) : ℝ := d / (v + r)

/-- The time to travel from B to A -/
noncomputable def time_B_to_A (v r d : ℝ) : ℝ := d / (v - r)

/-- The time to travel from B to A without paddles -/
noncomputable def time_B_to_A_no_paddles (r d : ℝ) : ℝ := d / r

theorem canoe_travel_time_relation (v r d : ℝ) 
  (h1 : v > 0) 
  (h2 : r > 0) 
  (h3 : d > 0) 
  (h4 : v > r) 
  (h5 : time_A_to_B v r d = 3 * time_B_to_A v r d) 
  (h6 : v = 2 * r) : 
  time_B_to_A_no_paddles r d = 3 * time_B_to_A v r d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_travel_time_relation_l385_38501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_constant_value_l385_38510

/-- A function satisfying the given conditions -/
noncomputable def f (c : ℝ) : ℝ → ℝ := sorry

/-- The domain of f is [1, +∞) -/
axiom f_domain (c x : ℝ) : x ≥ 1 → f c x ≠ 0

/-- f(2x) = cf(x) for all x in the domain -/
axiom f_scaling (c x : ℝ) : x ≥ 1 → f c (2 * x) = c * f c x

/-- f(x) = 1 - |x - 3| for 2 ≤ x ≤ 4 -/
axiom f_definition (c x : ℝ) : 2 ≤ x ∧ x ≤ 4 → f c x = 1 - |x - 3|

/-- c is a positive constant -/
axiom c_positive (c : ℝ) : c > 0

/-- All local maximum points of f fall on the same line -/
axiom local_max_on_line (c : ℝ) : ∃ (m b : ℝ), ∀ (x : ℝ), x ≥ 1 → IsLocalMax (f c) x → f c x = m * x + b

/-- The main theorem: if f satisfies all conditions, then c = 1 or c = 2 -/
theorem f_constant_value (c : ℝ) : (∀ x, x ≥ 1 → f c x ≠ 0) ∧ 
  (∀ x, x ≥ 1 → f c (2 * x) = c * f c x) ∧ 
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f c x = 1 - |x - 3|) ∧ 
  c > 0 ∧
  (∃ (m b : ℝ), ∀ (x : ℝ), x ≥ 1 → IsLocalMax (f c) x → f c x = m * x + b) →
  c = 1 ∨ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_constant_value_l385_38510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurrence_properties_l385_38556

def recurrence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 5 * recurrence (n + 1) - recurrence n - 1

theorem recurrence_properties :
  ∀ n : ℕ,
    (recurrence n > 0 ∧ recurrence (n + 1) > 0) ∧
    (recurrence n ≤ recurrence (n + 1)) ∧
    (∃ k : ℕ, k * (recurrence n * recurrence (n + 1)) = 
      recurrence n^2 + recurrence (n + 1)^2 + recurrence n + recurrence (n + 1) + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurrence_properties_l385_38556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l385_38538

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + 6 / x - 3
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * x + 7 - 3 * m

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, (1 < x ∧ x < b) ↔ x * f a x < 4

-- Define the relationship between f and g
def f_g_relation (a m : ℝ) : Prop :=
  ∀ x₁, x₁ ∈ Set.Icc 2 3 → ∃ x₂, x₂ ∈ Set.Ioc 1 4 ∧ f a x₁ = g m x₂ / x₁

-- State the theorem
theorem function_properties :
  ∃ a b : ℝ, solution_set a b ∧
  (a = 1 ∧ b = 2) ∧
  ∀ m : ℝ, f_g_relation a m ↔ (m ≤ -3 ∨ m > 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l385_38538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deceased_income_l385_38524

theorem deceased_income (initial_members final_members : ℕ) 
  (initial_average final_average : ℝ) 
  (h1 : initial_members = 4)
  (h2 : final_members = 3)
  (h3 : initial_average = 735)
  (h4 : final_average = 650)
  : initial_members * initial_average - final_members * final_average = 990 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deceased_income_l385_38524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_power_two_inequality_implies_alpha_value_l385_38582

theorem cos_power_two_inequality_implies_alpha_value (α : ℝ) :
  (∀ n : ℕ, Real.cos (2^n * α) < -1/3) →
  ∃ k : ℤ, α = 2 * k * Real.pi + 2 * Real.pi / 3 ∨ α = 2 * k * Real.pi - 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_power_two_inequality_implies_alpha_value_l385_38582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_iff_not_prime_power_l385_38584

/-- A function from positive integers to positive integers is Canadian if it satisfies
    gcd(f(f(x)), f(x+y)) = gcd(x, y) for all positive integers x and y -/
def Canadian (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, Nat.gcd (f (f x)).val (f (x + y)).val = Nat.gcd x.val y.val

/-- A positive integer m is magical if f(m) = m for all Canadian functions f -/
def Magical (m : ℕ+) : Prop :=
  ∀ f : ℕ+ → ℕ+, Canadian f → f m = m

/-- A positive integer n is a prime power if there exists a prime p and a positive integer k
    such that n = p^k -/
def IsPrimePower (n : ℕ+) : Prop :=
  ∃ p k : ℕ, Nat.Prime p ∧ n.val = p ^ k

theorem magical_iff_not_prime_power (m : ℕ+) :
  Magical m ↔ ¬ IsPrimePower m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_iff_not_prime_power_l385_38584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l385_38558

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - log x

theorem problem_solution :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x ≤ 2) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x = 2) ∧
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x ≥ log 2 + 5/4) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x = log 2 + 5/4) ∧
  (∃ b : ℝ, b > 0 ∧
    (∀ x ∈ Set.Ioo (0 : ℝ) (exp 1), f 0 b x ≥ 3) ∧
    (∃ x ∈ Set.Ioo (0 : ℝ) (exp 1), f 0 b x = 3) ∧
    b = exp 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l385_38558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_through_point_perpendicular_to_vector_l385_38591

/-- The equation of a plane passing through a point and perpendicular to a vector -/
theorem plane_equation_through_point_perpendicular_to_vector 
  (A B C : ℝ × ℝ × ℝ) : 
  let BC := (C.fst - B.fst, C.snd.fst - B.snd.fst, C.snd.snd - B.snd.snd)
  (∀ (x y z : ℝ), x + 2*y + 2*z - 3 = 0 ↔ 
    -- The plane passes through point A
    BC.fst * (x - A.fst) + BC.snd.fst * (y - A.snd.fst) + BC.snd.snd * (z - A.snd.snd) = 0) →
  -- Given points
  A = (-3, 5, -2) →
  B = (-4, 0, 3) →
  C = (-3, 2, 5) →
  -- The equation represents the plane
  ∀ (x y z : ℝ), x + 2*y + 2*z - 3 = 0 ↔ 
    BC.fst * (x - A.fst) + BC.snd.fst * (y - A.snd.fst) + BC.snd.snd * (z - A.snd.snd) = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_through_point_perpendicular_to_vector_l385_38591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l385_38569

/-- The area of a trapezoid with bases 4h and 5h, and height h, is 9h^2/2 -/
theorem trapezoid_area (h : ℝ) : 
  (1 / 2) * ((4 * h) + (5 * h)) * h = (9 * h^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l385_38569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acceleration_inverse_square_distance_l385_38554

/-- The position function of a point moving in a straight line -/
noncomputable def s (t : ℝ) : ℝ := Real.rpow t (2/3)

/-- The velocity function, which is the derivative of the position function -/
noncomputable def v (t : ℝ) : ℝ := (2/3) * Real.rpow t (-1/3)

/-- The acceleration function, which is the derivative of the velocity function -/
noncomputable def a (t : ℝ) : ℝ := -(2/9) * Real.rpow t (-4/3)

/-- Theorem stating that the acceleration is inversely proportional to the square of the distance traveled -/
theorem acceleration_inverse_square_distance :
  ∃ (k : ℝ), ∀ (t : ℝ), t > 0 → a t = k / (s t)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acceleration_inverse_square_distance_l385_38554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_region_area_is_six_l385_38581

-- Define the region in the XY-plane
def region_xy (x y : ℝ) : Prop := |4*x - 12| + |3*y - 9| ≤ 6

-- Define the area of the region
def area_of_region : ℝ := 6

-- Theorem statement
theorem region_area_is_six :
  ∃ (A : Set (ℝ × ℝ)), (∀ (x y : ℝ), (x, y) ∈ A ↔ region_xy x y) ∧ 
  MeasureTheory.volume A = ENNReal.ofReal area_of_region :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_region_area_is_six_l385_38581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_triangle_area_l385_38580

noncomputable section

-- Define the function f(x)
def f (x : Real) : Real := Real.sin (2 * x - Real.pi / 6)

-- Theorem for the monotonically decreasing interval
theorem monotonic_decreasing_interval :
  ∀ x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 6), 
    StrictAntiOn f (Set.Icc (Real.pi / 3) (5 * Real.pi / 6)) :=
by sorry

-- Define the triangle
structure Triangle :=
  (A B C : Real)
  (a b c : Real)
  (acute_A : 0 < A ∧ A < Real.pi / 2)
  (right_C : C = Real.pi / 2)
  (side_a : a = 2 * Real.sqrt 3)
  (side_c : c = 4)
  (angle_sum : A + B + C = Real.pi)

-- Theorem for the area of the triangle
theorem triangle_area (t : Triangle) (h : t.A = Real.pi / 3) : 
  (1 / 2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_triangle_area_l385_38580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_exponentials_and_logarithm_l385_38502

theorem compare_exponentials_and_logarithm : (0.7 : Real)^6 < Real.log 6 / Real.log 7 ∧ Real.log 6 / Real.log 7 < 6^(0.7 : Real) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_exponentials_and_logarithm_l385_38502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tractor_price_is_5000_l385_38542

/-- Represents the price of a single red tractor -/
def red_tractor_price : ℝ → Prop := sorry

/-- Commission rate for red tractors -/
def red_commission : ℝ := 0.1

/-- Commission rate for green tractors -/
def green_commission : ℝ := 0.2

/-- Number of red tractors sold -/
def red_tractors_sold : ℕ := 2

/-- Number of green tractors sold -/
def green_tractors_sold : ℕ := 3

/-- Tobias's total salary for the week -/
def total_salary : ℝ := 7000

/-- The price of a green tractor is twice the price of a red tractor -/
axiom green_tractor_price (p : ℝ) : red_tractor_price p → red_tractor_price (2 * p)

/-- Theorem stating that the price of a single red tractor is $5000 -/
theorem red_tractor_price_is_5000 :
  ∃ p : ℝ, red_tractor_price p ∧ 
  (red_tractors_sold : ℝ) * red_commission * p + 
  (green_tractors_sold : ℝ) * green_commission * (2 * p) = total_salary ∧
  p = 5000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tractor_price_is_5000_l385_38542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_digit_sum_of_successor_l385_38598

/-- Sum of digits of a natural number in decimal representation -/
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

/-- The minimal sum of digits possible for a given natural number -/
def minimal_sum_of_digits (n : ℕ) : ℕ :=
  sorry

/-- Given a natural number n whose decimal digits sum to 2017,
    the smallest possible sum of digits of n+1 is 2. -/
theorem smallest_digit_sum_of_successor (n : ℕ) 
    (h : sum_of_digits n = 2017) : minimal_sum_of_digits (n + 1) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_digit_sum_of_successor_l385_38598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_satisfying_equation_l385_38535

noncomputable def floor (x : ℝ) := ⌊x⌋

theorem smallest_n_satisfying_equation :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    floor ((Real.sqrt (m + 1) - Real.sqrt m) / 
           (Real.sqrt (4 * m^2 + 4 * m + 1) - Real.sqrt (4 * m^2 + 4 * m))) ≠ 
    floor (Real.sqrt (4 * m + 2018))) ∧
  floor ((Real.sqrt (n + 1) - Real.sqrt n) / 
         (Real.sqrt (4 * n^2 + 4 * n + 1) - Real.sqrt (4 * n^2 + 4 * n))) = 
  floor (Real.sqrt (4 * n + 2018)) ∧
  n = 252253 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_satisfying_equation_l385_38535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_identification_l385_38531

structure Coin where
  weight : ℝ

structure WeighingResult where
  left : List Coin
  right : List Coin
  result : Ordering

def isGenuine (c : Coin) (genuineWeight : ℝ) : Prop :=
  c.weight = genuineWeight

def isHeavy (c : Coin) (genuineWeight : ℝ) (difference : ℝ) : Prop :=
  c.weight = genuineWeight + difference

def isLight (c : Coin) (genuineWeight : ℝ) (difference : ℝ) : Prop :=
  c.weight = genuineWeight - difference

noncomputable def performWeighing (left right : List Coin) : WeighingResult :=
  { left := left
  , right := right
  , result := 
      if (left.map Coin.weight).sum > (right.map Coin.weight).sum then
        Ordering.gt
      else if (left.map Coin.weight).sum < (right.map Coin.weight).sum then
        Ordering.lt
      else
        Ordering.eq }

theorem counterfeit_coin_identification
  (a b c d e : Coin)
  (genuineWeight difference : ℝ)
  (hGenuine : ∃ x y z, x ∈ [a, b, c, d, e] ∧ y ∈ [a, b, c, d, e] ∧ z ∈ [a, b, c, d, e] ∧
               isGenuine x genuineWeight ∧ isGenuine y genuineWeight ∧ isGenuine z genuineWeight)
  (hHeavy : ∃ x, x ∈ [a, b, c, d, e] ∧ isHeavy x genuineWeight difference)
  (hLight : ∃ x, x ∈ [a, b, c, d, e] ∧ isLight x genuineWeight difference)
  (hDifference : difference > 0) :
  ∃ (w₁ w₂ w₃ : WeighingResult),
    (w₁ = performWeighing [a] [b]) ∧
    (w₂ = performWeighing [c] [d]) ∧
    (w₃ = performWeighing [a, b] [c, d]) ∧
    (∃ (heavy light : Coin),
      heavy ∈ [a, b, c, d, e] ∧
      light ∈ [a, b, c, d, e] ∧
      heavy ≠ light ∧
      isHeavy heavy genuineWeight difference ∧
      isLight light genuineWeight difference) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_identification_l385_38531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_l385_38553

/-- The area of a regular octagon formed by removing four unit squares from the corners of a square with side length 2 + √2 -/
theorem octagon_area : ℝ := by
  -- Define the side length of the square
  let square_side : ℝ := 2 + Real.sqrt 2
  
  -- Define the area of the square
  let square_area : ℝ := square_side ^ 2
  
  -- Define the area of four unit squares (to be removed)
  let removed_area : ℝ := 4 * 1

  -- Define the area of the octagon
  let octagon_area : ℝ := square_area - removed_area

  -- State that the octagon area is equal to 4 + 4√2
  have h : octagon_area = 4 + 4 * Real.sqrt 2 := by sorry

  -- Return the result
  exact octagon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_l385_38553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_circles_l385_38575

theorem shaded_area_of_circles (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) : 
  let chord_length : ℝ := 6
  let large_circle_radius : ℝ := r₁ + r₂
  let small_circles_touch : Prop := r₁ * r₂ = (chord_length / 2) ^ 2
  let shaded_area : ℝ := π * ((r₁ + r₂)^2 - r₁^2 - r₂^2)
  small_circles_touch → shaded_area = 9 * π / 2 := by
  intros
  sorry

#check shaded_area_of_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_circles_l385_38575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_B_union_A_eq_le_three_C_subset_A_iff_a_in_range_l385_38500

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Theorem 1: (∁ᵤB) ∪ A = {x | x ≤ 3}
theorem complement_B_union_A_eq_le_three :
  (Set.univ \ B) ∪ A = {x : ℝ | x ≤ 3} := by sorry

-- Theorem 2: C ⊆ A if and only if 1 < a ≤ 3
theorem C_subset_A_iff_a_in_range (a : ℝ) :
  C a ⊆ A ↔ 1 < a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_B_union_A_eq_le_three_C_subset_A_iff_a_in_range_l385_38500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percent_approx_l385_38513

/-- Calculate the profit percent from buying, repairing, and selling a car -/
noncomputable def profit_percent (purchase_price repair_cost selling_price : ℝ) : ℝ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem: The profit percent for the given scenario is approximately 13.86% -/
theorem car_profit_percent_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |profit_percent 42000 15000 64900 - 13.86| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_profit_percent_approx_l385_38513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l385_38587

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt 3 * Real.sin (2 * x - Real.pi / 3) - 2 * Real.sin (x - Real.pi / 4) * Real.sin (x + Real.pi / 4)

theorem f_properties : 
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/12) (Real.pi/2) → f x ≤ 1) ∧
  (f (Real.pi/3) = 1) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/12) (Real.pi/2) → f x ≥ -Real.sqrt 3 / 2) ∧
  (f (-Real.pi/12) = -Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l385_38587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_beads_count_l385_38552

theorem white_beads_count (total_black : ℕ) (total_white : ℕ) 
  (black_fraction : ℚ) (white_fraction : ℚ) (pulled_out : ℕ) : 
  total_black = 90 → 
  black_fraction = 1/6 → 
  white_fraction = 1/3 → 
  pulled_out = 32 → 
  (black_fraction * total_black + white_fraction * total_white : ℚ) = pulled_out →
  total_white = 51 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_beads_count_l385_38552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_percent_decrease_correct_optimal_selling_price_correct_l385_38509

-- Define the initial and final factory prices
noncomputable def initialPrice : ℝ := 144
noncomputable def finalPrice : ℝ := 100

-- Define the number of years
def years : ℕ := 2

-- Define the average percentage decrease
noncomputable def avgPercentDecrease : ℝ := 1 / 6

-- Define the initial mall selling price
noncomputable def initialMallPrice : ℝ := 140

-- Define the initial daily sales
noncomputable def initialDailySales : ℝ := 20

-- Define the price reduction effect
noncomputable def priceReductionEffect : ℝ := 10 / 5

-- Define the target daily profit
noncomputable def targetDailyProfit : ℝ := 1250

-- Define the optimal selling price
noncomputable def optimalSellingPrice : ℝ := 125

-- Theorem for the average percentage decrease
theorem avg_percent_decrease_correct :
  initialPrice * (1 - avgPercentDecrease) ^ years = finalPrice := by sorry

-- Theorem for the optimal selling price
theorem optimal_selling_price_correct :
  let salesQuantity := initialDailySales + priceReductionEffect * (initialMallPrice - optimalSellingPrice)
  (optimalSellingPrice - finalPrice) * salesQuantity = targetDailyProfit := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_percent_decrease_correct_optimal_selling_price_correct_l385_38509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_divisor_div_by_five_l385_38505

theorem prob_divisor_div_by_five (n : ℕ) (hn : n = 15) :
  let factorial := n.factorial
  let divisors := Finset.filter (· ∣ factorial) (Finset.range (factorial + 1))
  let divisors_div_by_five := Finset.filter (λ d => 5 ∣ d) divisors
  (divisors_div_by_five.card : ℚ) / divisors.card = 3/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_divisor_div_by_five_l385_38505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frank_money_problem_l385_38568

theorem frank_money_problem (initial_money : ℝ) : 
  (18 / 35 : ℝ) * initial_money = 480 → 
  abs (initial_money - 933.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frank_money_problem_l385_38568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_covered_by_unit_circle_l385_38564

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance function between two points
noncomputable def distance (p q : Point2D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Theorem statement
theorem centers_covered_by_unit_circle 
  (X A B C : Point2D)
  (hA : distance X A < 1)
  (hB : distance X B < 1)
  (hC : distance X C < 1) :
  ∃ (center : Point2D), 
    distance center A ≤ 1 ∧ 
    distance center B ≤ 1 ∧ 
    distance center C ≤ 1 := by
  -- Use X as the center of the covering circle
  use X
  constructor
  · -- Prove distance X A ≤ 1
    exact le_of_lt hA
  constructor
  · -- Prove distance X B ≤ 1
    exact le_of_lt hB
  -- Prove distance X C ≤ 1
  exact le_of_lt hC


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_covered_by_unit_circle_l385_38564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_distance_range_l385_38560

noncomputable section

-- Define the curve C
def curve_C (θ : ℝ) : ℝ × ℝ :=
  (1 + Real.sqrt 3 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define the line l
def line_l (m : ℝ) (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi/3) = (Real.sqrt 3 / 2) * m

-- Define the distance from a point to a line
def distance_point_to_line (x y m : ℝ) : ℝ :=
  abs (y + Real.sqrt 3 * x - Real.sqrt 3 * m) / 2

-- Theorem 1: Line l is tangent to curve C when m = 3
theorem line_tangent_to_curve : 
  ∃ θ, distance_point_to_line (curve_C θ).1 (curve_C θ).2 3 = Real.sqrt 3 := by sorry

-- Theorem 2: Range of m for which there exists a point on C with distance √3/2 from l
theorem distance_range :
  ∀ m, (∃ θ, distance_point_to_line (curve_C θ).1 (curve_C θ).2 m = Real.sqrt 3 / 2) ↔ 
    -2 ≤ m ∧ m ≤ 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_distance_range_l385_38560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_intervals_l385_38589

-- Define the function f(x) = |sin(x + π/3)|
noncomputable def f (x : ℝ) : ℝ := |Real.sin (x + Real.pi/3)|

-- Define the property of being strictly increasing on an interval
def StrictlyIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem f_strictly_increasing_intervals :
  ∀ k : ℤ, StrictlyIncreasingOn f (k * Real.pi - Real.pi/3) (k * Real.pi + Real.pi/6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_intervals_l385_38589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_time_to_dance_class_l385_38572

def alice_steps_per_minute : ℝ := 110
def alice_step_length : ℝ := 50
def alice_time : ℝ := 20
def bella_steps_per_minute : ℝ := 120
def bella_step_length : ℝ := 47

theorem bella_time_to_dance_class :
  let alice_speed := alice_steps_per_minute * alice_step_length
  let distance := alice_speed * alice_time
  let bella_speed := bella_steps_per_minute * bella_step_length
  bella_speed > 0 →
  Int.floor (distance / bella_speed + 0.5) = 20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_time_to_dance_class_l385_38572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equations_represent_hyperbola_l385_38557

-- Define the constant θ
variable (θ : ℝ)

-- Define the parameter t
variable (t : ℝ)

-- Define x and y as functions of t
noncomputable def x (t : ℝ) : ℝ := (1/2) * (Real.exp t + Real.exp (-t)) * Real.cos θ
noncomputable def y (t : ℝ) : ℝ := (1/2) * (Real.exp t - Real.exp (-t)) * Real.sin θ

-- State the theorem
theorem equations_represent_hyperbola (h : ∀ n : ℤ, θ ≠ n * π / 2) :
  ∃ a b : ℝ, ∀ t : ℝ, (x θ t / a)^2 - (y θ t / b)^2 = 1 := by
  -- We'll use cos θ as a and sin θ as b
  let a := Real.cos θ
  let b := Real.sin θ
  
  -- Existential introduction
  use a, b
  
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equations_represent_hyperbola_l385_38557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_53_to_base5_l385_38523

def to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

def is_non_consecutive (digits : List ℕ) : Prop :=
  ∀ i : ℕ, i + 1 < digits.length → digits[i]?.isSome → digits[i+1]?.isSome →
    (digits[i]?.getD 0) + 1 ≠ (digits[i+1]?.getD 0)

theorem decimal_53_to_base5 :
  let base5_digits := to_base5 53
  base5_digits.length = 3 ∧ is_non_consecutive base5_digits :=
by
  -- Define base5_digits
  let base5_digits := to_base5 53
  
  -- Split the goal into two parts
  have h1 : base5_digits.length = 3 := by sorry
  have h2 : is_non_consecutive base5_digits := by sorry
  
  -- Combine the two parts to prove the theorem
  exact ⟨h1, h2⟩

#eval to_base5 53  -- This will print the result of converting 53 to base 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_53_to_base5_l385_38523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_factorial_plus_l385_38565

theorem prime_divisors_factorial_plus (n : ℕ) : ∃ f : ℕ → ℕ, 
  (∀ k, k < n → Prime (f k)) ∧ 
  (∀ k, k < n → f k ∣ n.factorial + k + 1) ∧
  (∀ k i, k < n → i < n → k ≠ i → ¬(f k ∣ n.factorial + i + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_factorial_plus_l385_38565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_real_roots_2010_poly_l385_38534

/-- A polynomial of degree 2010 with real coefficients -/
def Polynomial2010 : Type := Polynomial ℝ

/-- The roots of a polynomial -/
noncomputable def roots (p : Polynomial2010) : Finset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
noncomputable def distinctAbsValues (p : Polynomial2010) : ℕ := sorry

/-- The number of real roots of a polynomial -/
noncomputable def realRootCount (p : Polynomial2010) : ℕ := sorry

/-- Theorem: If a polynomial of degree 2010 with real coefficients has 1005 distinct absolute values among its roots, then it has at least 3 real roots -/
theorem min_real_roots_2010_poly (p : Polynomial2010) 
  (h_degree : p.degree = 2010)
  (h_distinct_abs : distinctAbsValues p = 1005) :
  realRootCount p ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_real_roots_2010_poly_l385_38534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_l385_38579

theorem exam_average (total_boys : ℕ) (passed_boys : ℕ) (overall_avg : ℚ) (failed_avg : ℚ) :
  total_boys = 120 →
  overall_avg = 35 →
  failed_avg = 15 →
  passed_boys = 100 →
  (let failed_boys := total_boys - passed_boys
   let total_marks := overall_avg * total_boys
   let failed_marks := failed_avg * failed_boys
   let passed_marks := total_marks - failed_marks
   (passed_marks / passed_boys) = 39) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_l385_38579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l385_38562

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define point A
def point_A : ℝ × ℝ := (5, 2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (min_val : ℝ), min_val = 7 ∧
  ∀ (p : ℝ × ℝ), parabola p.1 p.2 →
    distance p point_A + distance p focus ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l385_38562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l385_38525

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℤ := {x | 0 < x ∧ x < 5}

theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l385_38525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_of_parametric_function_l385_38533

-- Define the parametric function
noncomputable def x (t : ℝ) := Real.tan t
noncomputable def y (t : ℝ) := 1 / Real.sin (2 * t)

-- Define the second derivative
noncomputable def y_xx_second_derivative (t : ℝ) : ℝ := 
  -2 * (Real.cos t)^3 / (Real.sin t * Real.cos (2 * t))

-- State the theorem
theorem second_derivative_of_parametric_function (t : ℝ) :
  y_xx_second_derivative t = -2 * (Real.cos t)^3 / (Real.sin t * Real.cos (2 * t)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_of_parametric_function_l385_38533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l385_38540

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_property (t : Triangle) (h : t.a^2 = t.b^2 + t.c^2 - t.b * t.c) :
  (t.a * Real.sin t.B) / t.b = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l385_38540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_BC_l385_38585

theorem length_of_BC (AB AD DC : ℝ) (h1 : AB = 20) (h2 : AD = 16) (h3 : DC = 5) : 
  ∃ BC : ℝ, BC = 13 ∧ BC^2 = (Real.sqrt (AB^2 - AD^2))^2 + DC^2 := by
  let BD := Real.sqrt (AB^2 - AD^2)
  let BC := Real.sqrt (BD^2 + DC^2)
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_BC_l385_38585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_distance_l385_38555

noncomputable section

/-- Parabola passing through (4,4) with focus on x-axis -/
def Parabola : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = 4*x}

/-- Line x - y + 4 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {(x, y) | x - y + 4 = 0}

/-- Distance function between a point and the line -/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 + 4| / Real.sqrt 2

theorem parabola_and_distance :
  (∀ p ∈ Parabola, p.2^2 = 4*p.1) ∧
  (∃ d : ℝ, d = 3 * Real.sqrt 2 / 2 ∧
    ∀ p ∈ Parabola, distance_to_line p ≥ d ∧
    ∃ q ∈ Parabola, distance_to_line q = d) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_distance_l385_38555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_tournament_theorem_l385_38516

/-- Represents a volleyball tournament. -/
structure Tournament where
  n : ℕ  -- number of teams
  games : Fin n → Fin n → Bool  -- games i j is true if team i lost to team j
  valid : ∀ i j, i ≠ j → (games i j = !games j i)  -- each game has one winner and one loser

/-- A property that holds for a tournament if in any group of k teams, 
    there is at least one team that lost to no more than four others in that group. -/
def HasLowLossProperty (t : Tournament) (k : ℕ) : Prop :=
  ∀ (group : Finset (Fin t.n)), group.card = k → 
    ∃ i ∈ group, (group.filter (fun j ↦ t.games i j)).card ≤ 4

/-- The main theorem statement. -/
theorem volleyball_tournament_theorem (t : Tournament) (h : t.n = 110) :
  HasLowLossProperty t 55 → 
  ∃ i : Fin t.n, (Finset.univ.filter (fun j ↦ t.games i j)).card ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_tournament_theorem_l385_38516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_fourth_rods_l385_38508

def rod_lengths : List ℕ := List.range 30

def given_rods : List ℕ := [5, 10, 20]

def valid_fourth_rod (x : ℕ) : Bool :=
  x ∈ rod_lengths ∧ 
  x ∉ given_rods ∧ 
  x > 5 ∧ 
  x < 30 ∧
  x + 5 + 10 > 20 ∧
  x + 5 + 20 > 10 ∧
  x + 10 + 20 > 5

theorem count_valid_fourth_rods : 
  (rod_lengths.filter valid_fourth_rod).length = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_fourth_rods_l385_38508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_difference_l385_38514

noncomputable def score_distribution : Finset (ℝ × ℝ) := 
  {(60, 0.15), (75, 0.20), (80, 0.30), (85, 0.25), (90, 0.10)}

noncomputable def mean_score : ℝ := 
  (score_distribution.sum (λ (score, freq) => score * freq)) / 
  (score_distribution.sum (λ (_, freq) => freq))

def median_score : ℝ := 80

theorem score_difference : 
  |mean_score - median_score| = 1.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_difference_l385_38514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_l385_38596

noncomputable def f (x : ℝ) : ℝ := sorry

theorem derivative_at_two :
  ∃ (f' : ℝ → ℝ), (∀ x, HasDerivAt f (f' x) x) ∧ f' 2 = -9/4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_l385_38596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l385_38504

theorem equation_solutions : 
  {(n, m) : ℕ × ℕ | 3^n + 55 = m^2} = {(2, 8), (6, 28)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l385_38504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l385_38566

-- Define the function f(x) = |cos(2x)|
noncomputable def f (x : ℝ) := abs (Real.cos (2 * x))

-- State the theorem
theorem f_properties :
  -- f has a period of π/2
  (∀ x, f (x + Real.pi/2) = f x) ∧
  -- f is monotonically increasing in the interval (π/4, π/2)
  (∀ x y, Real.pi/4 < x ∧ x < y ∧ y < Real.pi/2 → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l385_38566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_combinations_eq_360_l385_38563

/-- Represents the number of choices for each day of the week -/
def daily_choices : Fin 6 → ℕ
  | 0 => 2  -- Monday
  | 1 => 3  -- Tuesday
  | 2 => 6  -- Wednesday
  | 3 => 5  -- Thursday
  | 4 => 2  -- Friday
  | 5 => 1  -- Saturday

/-- The total number of combinations for the week-long event -/
def total_combinations : ℕ := (List.range 6).foldl (fun acc i => acc * daily_choices i) 1

/-- Theorem stating that the total number of combinations is 360 -/
theorem total_combinations_eq_360 : total_combinations = 360 := by
  rw [total_combinations]
  simp [daily_choices]
  norm_num
  rfl

#eval total_combinations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_combinations_eq_360_l385_38563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l385_38537

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 2 = 1

-- Define the right focus of the hyperbola
noncomputable def right_focus (a : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 + 2), 0)

-- Define the asymptotes of the hyperbola
def asymptote (a : ℝ) (x y : ℝ) : Prop :=
  Real.sqrt 2 * x = a * y ∨ Real.sqrt 2 * x = -a * y

-- Define circle C
noncomputable def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  (x - Real.sqrt (a^2 + 2))^2 + y^2 = 2

-- Define line l
def line_l (x y : ℝ) : Prop :=
  x = Real.sqrt 3 * y

-- Theorem statement
theorem hyperbola_circle_intersection (a : ℝ) :
  (a > 0) →
  (∀ x y, asymptote a x y → (∃ x' y', circle_C a x' y' ∧ (x = x' ∨ y = y'))) →
  (∃ x₁ y₁ x₂ y₂, circle_C a x₁ y₁ ∧ circle_C a x₂ y₂ ∧ 
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ 
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 4)) →
  a = Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l385_38537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_OBEC_l385_38543

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := (10, 0)
def D : ℝ × ℝ := sorry
def E : ℝ × ℝ := (5, 5)

-- Define the lines
def line1 : ℝ → ℝ := λ x => -3 * x + 20
def line2 : ℝ → ℝ := λ x => 5

-- Helper function for area calculation
def area_quadrilateral (p1 p2 p3 p4 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_OBEC : 
  (line1 5 = 5) →  -- E is on line1
  (line2 5 = 5) →  -- E is on line2
  (line2 10 = 0) → -- C is on line2
  (∃ x, line1 x = 0 ∧ x > 0) → -- A exists
  (∃ y, line1 0 = y ∧ y > 0) → -- B exists
  (∃ y, line2 0 = y) → -- D exists
  area_quadrilateral O B E C = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_OBEC_l385_38543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l385_38550

/-- The circle with equation x^2 + y^2 = 5 -/
def myCircle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 5}

/-- The point (2, -1) -/
def myPoint : ℝ × ℝ := (2, -1)

/-- The line with equation 2x - y - 5 = 0 -/
def myLine : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 - 5 = 0}

theorem tangent_line_to_circle : 
  (∀ p ∈ myCircle, p ∉ myLine ∨ p = myPoint) ∧ 
  myPoint ∈ myLine ∧
  (∃ q ∈ myCircle, q ≠ myPoint ∧ q ∈ myLine) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l385_38550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l385_38593

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (-x^2 - 3*x + 4)}
def B : Set ℝ := {x | ∃ y, y = Real.exp (Real.log 2 * (2 - x^2))}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Set.Ioc (-4) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l385_38593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ana_investment_interest_l385_38573

/-- Calculate compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Ana's investment scenario --/
theorem ana_investment_interest :
  let principal : ℝ := 1500
  let rate : ℝ := 0.08
  let time : ℕ := 4
  ∃ x : ℝ, abs (compound_interest principal rate time - x) < 0.001 ∧ x = 540.735 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ana_investment_interest_l385_38573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_books_per_student_l385_38548

theorem average_books_per_student (total_students : ℕ) 
  (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 40)
  (h2 : zero_books = 2)
  (h3 : one_book = 12)
  (h4 : two_books = 10)
  (h5 : zero_books + one_book + two_books < total_students) :
  (0 * zero_books + 1 * one_book + 2 * two_books + 
   3 * (total_students - (zero_books + one_book + two_books))) / total_students ≥ 2 := by
  sorry

#check average_books_per_student

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_books_per_student_l385_38548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_width_from_circumference_difference_l385_38532

/-- The width of a circular road around a circular ground -/
noncomputable def road_width (outer_radius inner_radius : ℝ) : ℝ := outer_radius - inner_radius

/-- The circumference of a circle given its radius -/
noncomputable def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius

theorem road_width_from_circumference_difference 
  (outer_radius inner_radius : ℝ) 
  (h : circumference outer_radius - circumference inner_radius = 66) :
  road_width outer_radius inner_radius = 66 / (2 * Real.pi) := by
  sorry

#check road_width_from_circumference_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_width_from_circumference_difference_l385_38532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_theorem_l385_38551

/-- The measure of an angle in degrees, given that its supplement is eight times its complement. -/
def angle_measure : ℝ :=
  77.14

/-- Theorem stating that if an angle's supplement is eight times its complement, 
    then its measure is approximately 77.14 degrees. -/
theorem angle_measure_theorem (x : ℝ) 
    (h : 180 - x = 8 * (90 - x)) : 
    abs (x - angle_measure) < 0.01 := by
  sorry

#eval angle_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_theorem_l385_38551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisor_of_eight_probability_divisor_of_eight_is_half_l385_38597

/-- A fair 8-sided die is rolled. The probability that the number rolled is a divisor of 8 is 1/2. -/
theorem probability_divisor_of_eight (n : ℕ) : 
  n ∈ Finset.range 8 → (n ∣ 8 ↔ n ∈ Finset.range 9 ∩ {1, 2, 4, 8}) :=
by sorry

theorem probability_divisor_of_eight_is_half : 
  (Finset.filter (· ∣ 8) (Finset.range 8)).card / (Finset.range 8).card = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisor_of_eight_probability_divisor_of_eight_is_half_l385_38597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_swept_by_specific_triangle_l385_38545

/-- Calculates the area swept by a triangle moving upward -/
noncomputable def area_swept_by_moving_triangle (BC AB AD : ℝ) (speed : ℝ) (time : ℝ) : ℝ :=
  let distance := speed * time
  let rectangle_area := BC * distance
  let triangle_area := (1/2) * BC * AD
  let small_triangles_area := (1/2) * BC * distance
  rectangle_area + triangle_area + small_triangles_area

/-- Theorem stating the area swept by the specific triangle -/
theorem area_swept_by_specific_triangle :
  area_swept_by_moving_triangle 6 5 4 3 2 = 66 := by
  -- Unfold the definition and simplify
  unfold area_swept_by_moving_triangle
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_swept_by_specific_triangle_l385_38545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_statements_l385_38521

noncomputable def reciprocal (n : ℝ) : ℝ := 1 / n

theorem reciprocal_statements :
  (¬(reciprocal 4 + reciprocal 8 = reciprocal 12)) ∧
  (¬(reciprocal 9 - reciprocal 3 = reciprocal 6)) ∧
  (¬((reciprocal 3 + reciprocal 4) * reciprocal 12 = reciprocal 7)) ∧
  ((reciprocal 15) / (reciprocal 5) = reciprocal 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_statements_l385_38521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_of_square_l385_38567

def vertex1 : ℂ := 2 + 3*Complex.I
def vertex2 : ℂ := -1 + 4*Complex.I
def vertex3 : ℂ := -3 - 2*Complex.I

def is_square (v1 v2 v3 v4 : ℂ) : Prop :=
  let s1 := v2 - v1
  let s2 := v3 - v2
  let s3 := v4 - v3
  let s4 := v1 - v4
  (s1 * Complex.I = s2) ∧ (s2 * Complex.I = -s1) ∧ (s3 * Complex.I = s4) ∧ (s4 * Complex.I = -s3)

theorem fourth_vertex_of_square :
  ∃ (v4 : ℂ), is_square vertex1 vertex2 vertex3 v4 ∧ v4 = -4 - 5*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_of_square_l385_38567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doughnuts_left_l385_38512

theorem doughnuts_left (total_doughnuts : ℕ) (total_staff : ℕ) 
  (staff_eating_three : ℕ) (staff_eating_two : ℕ) 
  (h1 : total_doughnuts = 120)
  (h2 : total_staff = 35)
  (h3 : staff_eating_three = 15)
  (h4 : staff_eating_two = 10)
  (h5 : staff_eating_three + staff_eating_two < total_staff) : 
  total_doughnuts - 
  (staff_eating_three * 3 + staff_eating_two * 2 + 
   (total_staff - staff_eating_three - staff_eating_two) * 4) = 15 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doughnuts_left_l385_38512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_august_temp_l385_38599

/-- Temperature function -/
noncomputable def temp_func (a b x : ℝ) : ℝ :=
  a + b * Real.sin ((Real.pi / 6) * x + Real.pi / 6)

/-- Theorem: Given the temperature function and conditions, prove the temperature in August -/
theorem august_temp (a b : ℝ) :
  temp_func a b 6 = 22 →
  temp_func a b 12 = 4 →
  temp_func a b 8 = 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_august_temp_l385_38599
