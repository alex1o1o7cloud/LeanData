import Mathlib

namespace NUMINAMATH_CALUDE_collinear_points_k_l3503_350370

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_k (k : ℝ) : 
  collinear ⟨1, -4⟩ ⟨3, 2⟩ ⟨6, k/3⟩ → k = 33 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_k_l3503_350370


namespace NUMINAMATH_CALUDE_simplify_expression_l3503_350340

theorem simplify_expression (x : ℝ) : 
  3*x + 6*x + 9*x + 12*x + 15*x + 18 + 24 = 45*x + 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3503_350340


namespace NUMINAMATH_CALUDE_A_neither_sufficient_nor_necessary_l3503_350318

-- Define propositions A and B
def proposition_A (a b : ℝ) : Prop := a + b ≠ 4
def proposition_B (a b : ℝ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem stating that A is neither sufficient nor necessary for B
theorem A_neither_sufficient_nor_necessary :
  ¬(∀ a b : ℝ, proposition_A a b → proposition_B a b) ∧
  ¬(∀ a b : ℝ, proposition_B a b → proposition_A a b) :=
sorry

end NUMINAMATH_CALUDE_A_neither_sufficient_nor_necessary_l3503_350318


namespace NUMINAMATH_CALUDE_prove_n_equals_two_l3503_350377

def a (n k : ℕ) : ℕ := (n * k + 1) ^ k

theorem prove_n_equals_two (n : ℕ) : a n (a n (a n 0)) = 343 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_prove_n_equals_two_l3503_350377


namespace NUMINAMATH_CALUDE_sweets_expenditure_l3503_350309

theorem sweets_expenditure (initial_amount : ℚ) (friends_count : ℕ) (amount_per_friend : ℚ) (final_amount : ℚ) 
  (h1 : initial_amount = 71/10)
  (h2 : friends_count = 2)
  (h3 : amount_per_friend = 1)
  (h4 : final_amount = 405/100) :
  initial_amount - friends_count * amount_per_friend - final_amount = 21/20 := by
  sorry

end NUMINAMATH_CALUDE_sweets_expenditure_l3503_350309


namespace NUMINAMATH_CALUDE_kaleb_final_score_l3503_350320

/-- Calculates Kaleb's final adjusted score in a trivia game -/
theorem kaleb_final_score : 
  let first_half_score : ℝ := 43
  let first_half_bonus1 : ℝ := 0.20
  let first_half_bonus2 : ℝ := 0.05
  let second_half_score : ℝ := 23
  let second_half_penalty1 : ℝ := 0.10
  let second_half_penalty2 : ℝ := 0.08
  
  let first_half_adjusted := first_half_score * (1 + first_half_bonus1 + first_half_bonus2)
  let second_half_adjusted := second_half_score * (1 - second_half_penalty1 - second_half_penalty2)
  
  first_half_adjusted + second_half_adjusted = 72.61
  := by sorry

end NUMINAMATH_CALUDE_kaleb_final_score_l3503_350320


namespace NUMINAMATH_CALUDE_complex_coordinate_l3503_350351

theorem complex_coordinate (z : ℂ) (h : Complex.I * z = 2 + 4 * Complex.I) : 
  z = 4 - 2 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_coordinate_l3503_350351


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l3503_350322

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (bp : BatsmanPerformance) (runsInLastInning : ℕ) : ℚ :=
  (bp.totalRuns + runsInLastInning) / (bp.innings + 1)

/-- Theorem: If a batsman's average increases by 2 after scoring 50 in the 17th inning, 
    then the new average is 18 -/
theorem batsman_average_after_17th_inning 
  (bp : BatsmanPerformance)
  (h1 : bp.innings = 16)
  (h2 : newAverage bp 50 = bp.average + 2)
  : newAverage bp 50 = 18 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l3503_350322


namespace NUMINAMATH_CALUDE_triangle_property_l3503_350327

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  a = 3 →
  b = 2 * Real.sqrt 6 →
  B = 2 * A →
  0 < A →
  A < π →
  0 < B →
  B < π →
  0 < C →
  C < π →
  A + B + C = π →
  a = 2 * (Real.sin (A / 2)) * (Real.sin (C / 2)) / Real.sin B →
  b = 2 * (Real.sin (B / 2)) * (Real.sin (C / 2)) / Real.sin A →
  c = 2 * (Real.sin (A / 2)) * (Real.sin (B / 2)) / Real.sin C →
  Real.cos A = Real.sqrt 6 / 3 ∧ c = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3503_350327


namespace NUMINAMATH_CALUDE_probability_neighboring_points_l3503_350393

/-- The probability of choosing neighboring points on a circle -/
theorem probability_neighboring_points (n : ℕ) (h : n ≥ 3) :
  (2 : ℚ) / (n - 1) = (n : ℚ) / (n.choose 2) := by
  sorry

end NUMINAMATH_CALUDE_probability_neighboring_points_l3503_350393


namespace NUMINAMATH_CALUDE_product_equals_square_l3503_350339

theorem product_equals_square : 10 * 9.99 * 0.999 * 100 = (99.9 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l3503_350339


namespace NUMINAMATH_CALUDE_inverse_function_theorem_l3503_350349

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2 - 1

theorem inverse_function_theorem (x : ℝ) (h : x > 0) :
  f (f_inv x) = x ∧ f_inv (f x) = x :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_theorem_l3503_350349


namespace NUMINAMATH_CALUDE_equation_positive_roots_l3503_350383

theorem equation_positive_roots (b : ℝ) : 
  ∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧ r₁ ≠ r₂ ∧
  (∀ x : ℝ, x > 0 → ((x - b) * (x - 2) * (x + 1) = 3 * (x - b) * (x + 1)) ↔ (x = r₁ ∨ x = r₂)) :=
sorry

end NUMINAMATH_CALUDE_equation_positive_roots_l3503_350383


namespace NUMINAMATH_CALUDE_no_function_pair_exists_l3503_350316

theorem no_function_pair_exists : ¬∃ (f g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^3 := by
  sorry

end NUMINAMATH_CALUDE_no_function_pair_exists_l3503_350316


namespace NUMINAMATH_CALUDE_ten_mile_taxi_cost_l3503_350334

/-- Calculates the cost of a taxi ride -/
def taxiCost (baseFare mileCharge flatCharge thresholdMiles miles : ℚ) : ℚ :=
  baseFare + mileCharge * miles + if miles > thresholdMiles then flatCharge else 0

/-- Theorem: The cost of a 10-mile taxi ride is $5.50 -/
theorem ten_mile_taxi_cost :
  taxiCost 2 0.3 0.5 8 10 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_ten_mile_taxi_cost_l3503_350334


namespace NUMINAMATH_CALUDE_only_C_not_like_terms_l3503_350305

-- Define what it means for two terms to be like terms
def are_like_terms (term1 term2 : ℕ → ℕ → ℝ) : Prop :=
  ∀ a b, ∃ k, term1 a b = k * term2 a b ∨ term2 a b = k * term1 a b

-- Define the terms from the problem
def term_A1 (_ _ : ℕ) : ℝ := -2
def term_A2 (_ _ : ℕ) : ℝ := 12

def term_B1 (a b : ℕ) : ℝ := -2 * a^2 * b
def term_B2 (a b : ℕ) : ℝ := a^2 * b

def term_C1 (m _ : ℕ) : ℝ := 2 * m
def term_C2 (_ n : ℕ) : ℝ := 2 * n

def term_D1 (x y : ℕ) : ℝ := -1 * x^2 * y^2
def term_D2 (x y : ℕ) : ℝ := 12 * x^2 * y^2

-- Theorem stating that only C is not like terms
theorem only_C_not_like_terms :
  are_like_terms term_A1 term_A2 ∧
  are_like_terms term_B1 term_B2 ∧
  ¬(are_like_terms term_C1 term_C2) ∧
  are_like_terms term_D1 term_D2 :=
sorry

end NUMINAMATH_CALUDE_only_C_not_like_terms_l3503_350305


namespace NUMINAMATH_CALUDE_min_integer_solution_inequality_l3503_350313

theorem min_integer_solution_inequality :
  ∀ x : ℤ, (4 * (x + 1) + 2 > x - 1) ↔ (x ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_min_integer_solution_inequality_l3503_350313


namespace NUMINAMATH_CALUDE_absolute_value_difference_l3503_350306

theorem absolute_value_difference (x p : ℝ) (h1 : |x - 5| = p) (h2 : x > 5) : x - p = 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_difference_l3503_350306


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3503_350380

/-- Proves that if an item is sold for 1260 with a 16% loss, its cost price was 1500 --/
theorem cost_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1260)
  (h2 : loss_percentage = 16) : 
  (selling_price / (1 - loss_percentage / 100)) = 1500 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3503_350380


namespace NUMINAMATH_CALUDE_range_of_mu_l3503_350375

theorem range_of_mu (a b μ : ℝ) (ha : a > 0) (hb : b > 0) (hμ : μ > 0) 
  (h : 1/a + 9/b = 1) : 
  (∀ a b, a > 0 → b > 0 → 1/a + 9/b = 1 → a + b ≥ μ) ↔ μ ∈ Set.Ioc 0 16 :=
by sorry

end NUMINAMATH_CALUDE_range_of_mu_l3503_350375


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3503_350386

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def linesParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (A : Point)
  (l : Line)
  (h_A : A.x = 1 ∧ A.y = 1)
  (h_l : l.a = 3 ∧ l.b = -2 ∧ l.c = 1) :
  ∃ (result : Line),
    result.a = 3 ∧ result.b = -2 ∧ result.c = -1 ∧
    pointOnLine A result ∧
    linesParallel result l :=
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3503_350386


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3503_350394

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (prod_eq : x * y = 162) : 
  |x - y| = 6 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3503_350394


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l3503_350365

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) = (4 * Real.pi * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l3503_350365


namespace NUMINAMATH_CALUDE_minimum_days_to_plant_trees_l3503_350310

def trees_planted (n : ℕ) : ℕ := 2^n - 1

theorem minimum_days_to_plant_trees : 
  (∀ k : ℕ, k < 7 → trees_planted k < 100) ∧ 
  trees_planted 7 ≥ 100 := by
sorry

end NUMINAMATH_CALUDE_minimum_days_to_plant_trees_l3503_350310


namespace NUMINAMATH_CALUDE_mrs_crabapple_gift_sequences_l3503_350384

/-- Represents Mrs. Crabapple's class setup -/
structure ClassSetup where
  num_students : ℕ
  meetings_per_week : ℕ
  alternating_gifts : Bool
  starts_with_crabapple : Bool

/-- Calculates the number of different gift recipient sequences for a given class setup -/
def num_gift_sequences (setup : ClassSetup) : ℕ :=
  setup.num_students ^ setup.meetings_per_week

/-- Theorem stating the number of different gift recipient sequences for Mrs. Crabapple's class -/
theorem mrs_crabapple_gift_sequences :
  let setup : ClassSetup := {
    num_students := 11,
    meetings_per_week := 4,
    alternating_gifts := true,
    starts_with_crabapple := true
  }
  num_gift_sequences setup = 14641 := by
  sorry

end NUMINAMATH_CALUDE_mrs_crabapple_gift_sequences_l3503_350384


namespace NUMINAMATH_CALUDE_system_solution_l3503_350360

theorem system_solution :
  let x : ℝ := -13
  let y : ℝ := -1
  let z : ℝ := 2
  (x + y + 16 * z = 18) ∧
  (x - 3 * y + 8 * z = 6) ∧
  (2 * x - y - 4 * z = -33) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3503_350360


namespace NUMINAMATH_CALUDE_alex_age_l3503_350389

theorem alex_age (alex_age precy_age : ℕ) : 
  (alex_age + 3 = 3 * (precy_age + 3)) →
  (alex_age - 1 = 7 * (precy_age - 1)) →
  alex_age = 15 := by
sorry

end NUMINAMATH_CALUDE_alex_age_l3503_350389


namespace NUMINAMATH_CALUDE_A_div_B_eq_37_l3503_350307

-- Define the series A
def A : ℝ := sorry

-- Define the series B
def B : ℝ := sorry

-- Theorem stating the relationship between A and B
theorem A_div_B_eq_37 : A / B = 37 := by sorry

end NUMINAMATH_CALUDE_A_div_B_eq_37_l3503_350307


namespace NUMINAMATH_CALUDE_triangle_abc_isosceles_l3503_350332

/-- Given a triangle ABC where 2sin(A) * cos(B) = sin(C), prove that the triangle is isosceles -/
theorem triangle_abc_isosceles (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_condition : 2 * Real.sin A * Real.cos B = Real.sin C) :
  A = B ∨ B = C ∨ A = C :=
sorry

end NUMINAMATH_CALUDE_triangle_abc_isosceles_l3503_350332


namespace NUMINAMATH_CALUDE_f_abs_g_is_odd_l3503_350314

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem f_abs_g_is_odd 
  (hf : is_odd f) 
  (hg : is_even g) : 
  is_odd (λ x ↦ f x * |g x|) := by
  sorry

end NUMINAMATH_CALUDE_f_abs_g_is_odd_l3503_350314


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3503_350358

theorem largest_n_satisfying_inequality : 
  ∀ n : ℤ, (1 : ℚ) / 3 + (n : ℚ) / 7 < 1 ↔ n ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3503_350358


namespace NUMINAMATH_CALUDE_possible_total_students_l3503_350363

/-- Represents the possible total number of students -/
inductive TotalStudents
  | seventySix
  | eighty

/-- Checks if a number is a valid group size given the constraints -/
def isValidGroupSize (size : ℕ) : Prop :=
  size = 12 ∨ size = 13 ∨ size = 14

/-- Represents the distribution of students into groups -/
structure StudentDistribution where
  groupSizes : Fin 6 → ℕ
  validSizes : ∀ i, isValidGroupSize (groupSizes i)
  fourGroupsOf13 : (Finset.filter (fun i => groupSizes i = 13) Finset.univ).card = 4
  totalStudents : TotalStudents

/-- The main theorem stating the possible total number of students -/
theorem possible_total_students (d : StudentDistribution) :
    d.totalStudents = TotalStudents.seventySix ∨
    d.totalStudents = TotalStudents.eighty :=
  sorry

end NUMINAMATH_CALUDE_possible_total_students_l3503_350363


namespace NUMINAMATH_CALUDE_not_prime_two_pow_plus_one_l3503_350343

theorem not_prime_two_pow_plus_one (n : ℕ) (d : ℕ) (h_odd : Odd d) (h_div : d ∣ n) :
  ¬ Nat.Prime (2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_two_pow_plus_one_l3503_350343


namespace NUMINAMATH_CALUDE_expression_bounds_l3503_350302

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) : 
  2 * Real.sqrt 2 + 2 ≤ 
    Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt ((b+1)^2 + (2 - c)^2) + 
    Real.sqrt ((c-1)^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2) ∧
  Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt ((b+1)^2 + (2 - c)^2) + 
  Real.sqrt ((c-1)^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l3503_350302


namespace NUMINAMATH_CALUDE_quadratic_decreasing_parameter_range_l3503_350341

/-- Given a quadratic function f(x) = -x^2 - 2ax - 3 that is decreasing on the interval (-2, +∞),
    prove that the parameter a is in the range [2, +∞). -/
theorem quadratic_decreasing_parameter_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = -x^2 - 2*a*x - 3) 
  (h2 : ∀ x y, x > -2 → y > x → f y < f x) : 
  a ∈ Set.Ici 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_parameter_range_l3503_350341


namespace NUMINAMATH_CALUDE_negative_number_identification_l3503_350356

theorem negative_number_identification :
  let numbers : List ℚ := [1, 0, 1/2, -2]
  ∀ x ∈ numbers, x < 0 ↔ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_negative_number_identification_l3503_350356


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3503_350348

theorem complex_equation_solution (z : ℂ) : (z - 1) * I = 1 + I → z = 2 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3503_350348


namespace NUMINAMATH_CALUDE_cube_root_square_l3503_350364

theorem cube_root_square (y : ℝ) : (y + 5) ^ (1/3 : ℝ) = 3 → (y + 5)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_square_l3503_350364


namespace NUMINAMATH_CALUDE_cube_packing_percentage_l3503_350371

/-- Calculates the number of whole cubes that can fit along a dimension -/
def cubesFit (dimension : ℕ) (cubeSize : ℕ) : ℕ :=
  dimension / cubeSize

/-- Calculates the volume of a rectangular box -/
def boxVolume (length width height : ℕ) : ℕ :=
  length * width * height

/-- Calculates the volume of a cube -/
def cubeVolume (size : ℕ) : ℕ :=
  size * size * size

/-- Theorem: The percentage of volume occupied by 4-inch cubes in a box with 
    dimensions 8 × 5 × 14 inches is 24/35 * 100% -/
theorem cube_packing_percentage :
  let boxLength : ℕ := 8
  let boxWidth : ℕ := 5
  let boxHeight : ℕ := 14
  let cubeSize : ℕ := 4
  let cubesAlongLength := cubesFit boxLength cubeSize
  let cubesAlongWidth := cubesFit boxWidth cubeSize
  let cubesAlongHeight := cubesFit boxHeight cubeSize
  let totalCubes := cubesAlongLength * cubesAlongWidth * cubesAlongHeight
  let volumeOccupied := totalCubes * cubeVolume cubeSize
  let totalVolume := boxVolume boxLength boxWidth boxHeight
  (volumeOccupied : ℚ) / totalVolume * 100 = 24 / 35 * 100 := by
  sorry

end NUMINAMATH_CALUDE_cube_packing_percentage_l3503_350371


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l3503_350346

theorem arithmetic_geometric_mean_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h_mean : (a + b) / 2 = 3 * Real.sqrt (a * b)) : 
  ∃ (n : ℤ), n = 34 ∧ ∀ (m : ℤ), |a / b - n| ≤ |a / b - m| :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l3503_350346


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3503_350362

theorem arithmetic_sequence_sum (k : ℕ) : 
  let a : ℕ → ℕ := λ n => 1 + 2 * (n - 1)
  let S : ℕ → ℕ := λ n => n * (2 * a 1 + (n - 1) * 2) / 2
  S (k + 2) - S k = 24 → k = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3503_350362


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3503_350323

/-- The asymptotes of the hyperbola (y²/16) - (x²/9) = 1 are y = ±(4/3)x -/
theorem hyperbola_asymptotes :
  let hyperbola := (fun (x y : ℝ) => (y^2 / 16) - (x^2 / 9) = 1)
  ∃ (m : ℝ), m > 0 ∧
    (∀ (x y : ℝ), hyperbola x y → (y = m*x ∨ y = -m*x)) ∧
    m = 4/3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3503_350323


namespace NUMINAMATH_CALUDE_two_numbers_property_l3503_350382

theorem two_numbers_property : ∃ x y : ℕ, 
  x ∈ Finset.range 38 ∧ 
  y ∈ Finset.range 38 ∧ 
  x ≠ y ∧
  (Finset.sum (Finset.range 38) id) - x - y = x * y + 1 ∧
  y - x = 20 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_property_l3503_350382


namespace NUMINAMATH_CALUDE_circle_properties_l3503_350376

/-- Given a circle with equation x^2 - 24x + y^2 - 4y = -36, 
    prove its center, radius, and the sum of center coordinates and radius. -/
theorem circle_properties : 
  let D : Set (ℝ × ℝ) := {p | (p.1^2 - 24*p.1 + p.2^2 - 4*p.2 = -36)}
  ∃ (a b r : ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ D ↔ (x - a)^2 + (y - b)^2 = r^2) ∧ 
    a = 12 ∧ 
    b = 2 ∧ 
    r = 4 * Real.sqrt 7 ∧
    a + b + r = 14 + 4 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3503_350376


namespace NUMINAMATH_CALUDE_john_newspaper_profit_l3503_350366

/-- Calculates the profit made by John selling newspapers --/
theorem john_newspaper_profit :
  let total_newspapers : ℕ := 500
  let selling_price : ℚ := 2
  let sold_percentage : ℚ := 80 / 100
  let discount_percentage : ℚ := 75 / 100
  let profit : ℚ := (total_newspapers : ℚ) * sold_percentage * selling_price - 
                    total_newspapers * (selling_price * (1 - discount_percentage))
  profit = 550
  := by sorry

end NUMINAMATH_CALUDE_john_newspaper_profit_l3503_350366


namespace NUMINAMATH_CALUDE_average_age_calculation_l3503_350379

theorem average_age_calculation (fifth_graders : ℕ) (fifth_graders_avg : ℝ)
  (parents : ℕ) (parents_avg : ℝ) (teachers : ℕ) (teachers_avg : ℝ) :
  fifth_graders = 40 ∧ fifth_graders_avg = 10 ∧
  parents = 60 ∧ parents_avg = 35 ∧
  teachers = 10 ∧ teachers_avg = 45 →
  let total_age := fifth_graders * fifth_graders_avg + parents * parents_avg + teachers * teachers_avg
  let total_people := fifth_graders + parents + teachers
  abs ((total_age / total_people) - 26.82) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_age_calculation_l3503_350379


namespace NUMINAMATH_CALUDE_nth_monomial_formula_l3503_350324

/-- Represents the coefficient of the nth monomial in the sequence -/
def coefficient (n : ℕ) : ℕ := 3 * n + 2

/-- Represents the exponent of 'a' in the nth monomial of the sequence -/
def exponent (n : ℕ) : ℕ := n

/-- Represents the nth monomial in the sequence as a function of 'a' -/
def nthMonomial (n : ℕ) (a : ℝ) : ℝ := (coefficient n : ℝ) * a ^ (exponent n)

/-- The sequence of monomials follows the pattern 5a, 8a^2, 11a^3, 14a^4, ... -/
axiom sequence_pattern (n : ℕ) (a : ℝ) : 
  n ≥ 1 → nthMonomial n a = (3 * n + 2 : ℝ) * a ^ n

/-- Theorem: The nth monomial in the sequence is equal to (3n+2)a^n -/
theorem nth_monomial_formula (n : ℕ) (a : ℝ) : 
  n ≥ 1 → nthMonomial n a = (3 * n + 2 : ℝ) * a ^ n := by
  sorry

end NUMINAMATH_CALUDE_nth_monomial_formula_l3503_350324


namespace NUMINAMATH_CALUDE_race_car_count_l3503_350354

theorem race_car_count (p_x p_y p_z p_combined : ℚ) (h1 : p_x = 1 / 7)
    (h2 : p_y = 1 / 3) (h3 : p_z = 1 / 5)
    (h4 : p_combined = p_x + p_y + p_z)
    (h5 : p_combined = 71 / 105) : ∃ n : ℕ, n = 105 ∧ p_x = 1 / n := by
  sorry

end NUMINAMATH_CALUDE_race_car_count_l3503_350354


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3503_350325

theorem complex_fraction_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(4/3))^(3/(2*5)) / (a^4)^(3/5) / ((a * (a^2 * b)^(1/3))^(1/2))^4 * ((a * b^(1/2))^(1/4))^6 = 1 / (a^2 * b)^(1/12) :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3503_350325


namespace NUMINAMATH_CALUDE_circle_properties_l3503_350312

-- Define the set of points (x, y) satisfying the equation
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.2 + 1 = 0}

-- Theorem statement
theorem circle_properties (p : ℝ × ℝ) (h : p ∈ S) :
  (∃ (z : ℝ), ∀ (q : ℝ × ℝ), q ∈ S → q.1 + q.2 ≤ z ∧ z = 2 + Real.sqrt 6) ∧
  (∀ (q : ℝ × ℝ), q ∈ S → q.1 ≠ 0 → -Real.sqrt 2 ≤ (q.2 + 1) / q.1 ∧ (q.2 + 1) / q.1 ≤ Real.sqrt 2) ∧
  (∀ (q : ℝ × ℝ), q ∈ S → 8 - 2*Real.sqrt 15 ≤ q.1^2 - 2*q.1 + q.2^2 + 1 ∧ 
                         q.1^2 - 2*q.1 + q.2^2 + 1 ≤ 8 + 2*Real.sqrt 15) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l3503_350312


namespace NUMINAMATH_CALUDE_bucket_weight_l3503_350392

/-- Given a bucket with unknown empty weight and full water weight,
    if it weighs c kilograms when three-quarters full and b kilograms when half-full,
    then its weight when one-third full is (5/3)b - (2/3)c kilograms. -/
theorem bucket_weight (b c : ℝ) : 
  (∃ x y : ℝ, x + 3/4 * y = c ∧ x + 1/2 * y = b) → 
  (∃ z : ℝ, z = 5/3 * b - 2/3 * c ∧ 
    ∀ x y : ℝ, x + 3/4 * y = c → x + 1/2 * y = b → x + 1/3 * y = z) :=
by sorry

end NUMINAMATH_CALUDE_bucket_weight_l3503_350392


namespace NUMINAMATH_CALUDE_cube_surface_area_from_volume_l3503_350388

-- Define the volume of the cube
def cube_volume : ℝ := 125

-- Theorem stating the relationship between volume and surface area of one side
theorem cube_surface_area_from_volume :
  ∃ (side_length : ℝ), 
    side_length^3 = cube_volume ∧ 
    side_length^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_volume_l3503_350388


namespace NUMINAMATH_CALUDE_divisibility_problem_l3503_350367

theorem divisibility_problem (n m k : ℕ) (h1 : n = 172835) (h2 : m = 136) (h3 : k = 21) :
  (n + k) % m = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3503_350367


namespace NUMINAMATH_CALUDE_five_people_four_rooms_l3503_350385

/-- The number of ways to assign n people to k rooms, where any number of people can be in a room -/
def room_assignments (n k : ℕ) : ℕ := sorry

/-- The specific case for 5 people and 4 rooms -/
theorem five_people_four_rooms : room_assignments 5 4 = 61 := by sorry

end NUMINAMATH_CALUDE_five_people_four_rooms_l3503_350385


namespace NUMINAMATH_CALUDE_division_simplification_l3503_350399

theorem division_simplification (a b c d e f : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) / (e * f) = (a * d) / (b * c * e * f) := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l3503_350399


namespace NUMINAMATH_CALUDE_division_equality_l3503_350344

theorem division_equality (h : 29.94 / 1.45 = 17.3) : 2994 / 14.5 = 173 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l3503_350344


namespace NUMINAMATH_CALUDE_train_speed_l3503_350374

/-- The speed of a train given the time to cross an electric pole and a platform -/
theorem train_speed (pole_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  pole_time = 12 →
  platform_length = 320 →
  platform_time = 44 →
  ∃ (train_length : ℝ) (speed_mps : ℝ),
    train_length = speed_mps * pole_time ∧
    train_length + platform_length = speed_mps * platform_time ∧
    speed_mps * 3.6 = 36 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3503_350374


namespace NUMINAMATH_CALUDE_probability_of_two_white_balls_l3503_350331

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def black_balls : ℕ := 2

def probability_two_white : ℚ := 3 / 10

theorem probability_of_two_white_balls :
  (Nat.choose white_balls 2) / (Nat.choose total_balls 2) = probability_two_white :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_white_balls_l3503_350331


namespace NUMINAMATH_CALUDE_oldest_child_age_l3503_350353

theorem oldest_child_age (a b c d : ℕ) : 
  a = 6 ∧ b = 9 ∧ c = 12 ∧ (a + b + c + d : ℚ) / 4 = 9 → d = 9 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l3503_350353


namespace NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l3503_350369

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- n has exactly 12 positive integer divisors -/
def has_twelve_divisors (n : ℕ+) : Prop := num_divisors n = 12

theorem smallest_with_twelve_divisors :
  ∃ (n : ℕ+), has_twelve_divisors n ∧ ∀ (m : ℕ+), has_twelve_divisors m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l3503_350369


namespace NUMINAMATH_CALUDE_ellipse_equation_l3503_350372

/-- The standard equation of an ellipse passing through (3,0) with eccentricity √6/3 -/
theorem ellipse_equation (x y : ℝ) :
  let e : ℝ := Real.sqrt 6 / 3
  let passes_through : ℝ × ℝ := (3, 0)
  (∃ (a b : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ∧
                 (3^2 / a^2 + 0^2 / b^2 = 1) ∧
                 e^2 = 1 - (min a b)^2 / (max a b)^2)) →
  (x^2 / 9 + y^2 / 3 = 1) ∨ (x^2 / 9 + y^2 / 27 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3503_350372


namespace NUMINAMATH_CALUDE_hayley_sticker_distribution_l3503_350317

theorem hayley_sticker_distribution (total_stickers : ℕ) (num_friends : ℕ) 
  (h1 : total_stickers = 72) 
  (h2 : num_friends = 9) 
  (h3 : total_stickers % num_friends = 0) : 
  total_stickers / num_friends = 8 := by
sorry

end NUMINAMATH_CALUDE_hayley_sticker_distribution_l3503_350317


namespace NUMINAMATH_CALUDE_probability_of_purple_l3503_350395

def die_sides : ℕ := 6
def red_sides : ℕ := 3
def yellow_sides : ℕ := 2
def blue_sides : ℕ := 1

def prob_red : ℚ := red_sides / die_sides
def prob_blue : ℚ := blue_sides / die_sides

theorem probability_of_purple (h1 : die_sides = red_sides + yellow_sides + blue_sides)
  (h2 : prob_red = red_sides / die_sides)
  (h3 : prob_blue = blue_sides / die_sides) :
  prob_red * prob_blue + prob_blue * prob_red = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_probability_of_purple_l3503_350395


namespace NUMINAMATH_CALUDE_max_unit_digit_of_2015_divisor_power_l3503_350391

def unit_digit (n : ℕ) : ℕ := n % 10

def is_divisor (d n : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem max_unit_digit_of_2015_divisor_power :
  ∃ (d : ℕ), is_divisor d 2015 ∧
  unit_digit (d^(2015 / d)) = 7 ∧
  ∀ (k : ℕ), is_divisor k 2015 → unit_digit (k^(2015 / k)) ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_unit_digit_of_2015_divisor_power_l3503_350391


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l3503_350329

theorem divisible_by_twelve (n : Nat) : n ≤ 9 → 5148 = 514 * 10 + n ↔ (514 * 10 + n) % 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l3503_350329


namespace NUMINAMATH_CALUDE_saramago_readers_l3503_350359

/-- Represents the number of workers in Palabras bookstore who have read certain books. -/
structure BookReaders where
  total : ℕ
  saramago : ℕ
  kureishi : ℕ
  both : ℕ
  neither : ℕ

/-- The conditions given in the problem. -/
def palabras_conditions (r : BookReaders) : Prop :=
  r.total = 150 ∧
  r.kureishi = r.total / 6 ∧
  r.both = 12 ∧
  r.neither = r.saramago - r.both - 1 ∧
  r.saramago - r.both + r.kureishi - r.both + r.both + r.neither = r.total

/-- The theorem to be proved. -/
theorem saramago_readers (r : BookReaders) 
  (h : palabras_conditions r) : r.saramago = 75 := by
  sorry

#check saramago_readers

end NUMINAMATH_CALUDE_saramago_readers_l3503_350359


namespace NUMINAMATH_CALUDE_system_solution_l3503_350336

theorem system_solution (c₁ c₂ c₃ : ℝ) :
  let x₁ := -2 * c₁ - c₂ + 2
  let x₂ := c₁ + 1
  let x₃ := c₂ + 3
  let x₄ := 2 * c₂ + 2 * c₃ - 2
  let x₅ := c₃ + 1
  (x₁ + 2 * x₂ - x₃ + x₄ - 2 * x₅ = -3) ∧
  (x₁ + 2 * x₂ + 3 * x₃ - x₄ + 2 * x₅ = 17) ∧
  (2 * x₁ + 4 * x₂ + 2 * x₃ = 14) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3503_350336


namespace NUMINAMATH_CALUDE_travelers_checks_denomination_l3503_350335

theorem travelers_checks_denomination 
  (total_checks : ℕ) 
  (total_worth : ℚ) 
  (spent_checks : ℕ) 
  (remaining_average : ℚ) 
  (h1 : total_checks = 30)
  (h2 : total_worth = 1800)
  (h3 : spent_checks = 6)
  (h4 : remaining_average = 62.5)
  (h5 : (total_checks - spent_checks : ℚ) * remaining_average + spent_checks * x = total_worth) :
  x = 50 := by
  sorry

end NUMINAMATH_CALUDE_travelers_checks_denomination_l3503_350335


namespace NUMINAMATH_CALUDE_problem_solution_l3503_350308

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.sin y = 2008)
  (h2 : x + 2008 * Real.cos y = 2007)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3503_350308


namespace NUMINAMATH_CALUDE_james_teaching_years_l3503_350319

theorem james_teaching_years (james partner : ℕ) 
  (h1 : james = partner + 10)
  (h2 : james + partner = 70) : 
  james = 40 := by
sorry

end NUMINAMATH_CALUDE_james_teaching_years_l3503_350319


namespace NUMINAMATH_CALUDE_person_on_throne_l3503_350301

-- Define the possible characteristics of the person
inductive PersonType
| Liar
| Monkey
| Knight

-- Define the statement made by the person
def statement (p : PersonType) : Prop :=
  p = PersonType.Liar ∨ p = PersonType.Monkey

-- Theorem to prove
theorem person_on_throne (p : PersonType) (h : statement p) : 
  p = PersonType.Monkey ∧ p ≠ PersonType.Liar :=
sorry

end NUMINAMATH_CALUDE_person_on_throne_l3503_350301


namespace NUMINAMATH_CALUDE_passenger_ticket_probability_l3503_350321

/-- The probability of a passenger getting a ticket at three counters -/
theorem passenger_ticket_probability
  (p₁ p₂ p₃ p₄ p₅ p₆ : ℝ)
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1)
  (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1)
  (h₃ : 0 ≤ p₃ ∧ p₃ ≤ 1)
  (h₄ : 0 ≤ p₄ ∧ p₄ ≤ 1)
  (h₅ : 0 ≤ p₅ ∧ p₅ ≤ 1)
  (h₆ : 0 ≤ p₆ ∧ p₆ ≤ 1)
  (h_sum : p₁ + p₂ + p₃ = 1) :
  let prob_get_ticket := p₁ * (1 - p₄) + p₂ * (1 - p₅) + p₃ * (1 - p₆)
  0 ≤ prob_get_ticket ∧ prob_get_ticket ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_passenger_ticket_probability_l3503_350321


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_neg_two_l3503_350300

theorem at_least_one_not_less_than_neg_two
  (a b c : ℝ)
  (ha : a < 0)
  (hb : b < 0)
  (hc : c < 0) :
  (a + 1/b ≥ -2) ∨ (b + 1/c ≥ -2) ∨ (c + 1/a ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_neg_two_l3503_350300


namespace NUMINAMATH_CALUDE_f_nested_application_l3503_350333

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 1

theorem f_nested_application : f (f (f (f (f 1)))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_nested_application_l3503_350333


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3503_350398

theorem polynomial_simplification (q : ℝ) : 
  (5 * q^4 + 3 * q^3 - 7 * q + 8) + (6 - 9 * q^3 + 4 * q - 3 * q^4) = 
  2 * q^4 - 6 * q^3 - 3 * q + 14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3503_350398


namespace NUMINAMATH_CALUDE_some_number_value_l3503_350350

theorem some_number_value (some_number : ℝ) : 
  (3 * 10^2) * (4 * some_number) = 12 → some_number = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3503_350350


namespace NUMINAMATH_CALUDE_hybrid_cars_with_full_headlights_l3503_350390

/-- Given a car dealership with the following properties:
  * There are 600 cars in total
  * 60% of cars are hybrids
  * 40% of hybrids have only one headlight
  Prove that the number of hybrids with full headlights is 216 -/
theorem hybrid_cars_with_full_headlights 
  (total_cars : ℕ) 
  (hybrid_percentage : ℚ) 
  (one_headlight_percentage : ℚ) 
  (h1 : total_cars = 600)
  (h2 : hybrid_percentage = 60 / 100)
  (h3 : one_headlight_percentage = 40 / 100) :
  ↑total_cars * hybrid_percentage - ↑total_cars * hybrid_percentage * one_headlight_percentage = 216 := by
  sorry

end NUMINAMATH_CALUDE_hybrid_cars_with_full_headlights_l3503_350390


namespace NUMINAMATH_CALUDE_derivative_greater_than_average_rate_of_change_l3503_350373

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (1 - 2*a) * x - Real.log x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2*a*x + (1 - 2*a) - 1/x

-- Theorem statement
theorem derivative_greater_than_average_rate_of_change 
  (a : ℝ) (x0 x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : x0 ≠ (x1 + x2) / 2) :
  f' a x0 > (f a x1 - f a x2) / (x1 - x2) := by
  sorry

end

end NUMINAMATH_CALUDE_derivative_greater_than_average_rate_of_change_l3503_350373


namespace NUMINAMATH_CALUDE_basketball_tournament_l3503_350303

theorem basketball_tournament (n : ℕ) (h_pos : n > 0) : 
  let total_players := 5 * n
  let total_matches := (total_players * (total_players - 1)) / 2
  let women_wins := 3 * total_matches / 7
  let men_wins := 4 * total_matches / 7
  (women_wins + men_wins = total_matches) → 
  (n ≠ 2 ∧ n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 5) :=
by sorry

end NUMINAMATH_CALUDE_basketball_tournament_l3503_350303


namespace NUMINAMATH_CALUDE_parabola_directrix_l3503_350304

/-- Given a parabola defined by x = -1/8 * y^2, its directrix is x = 1/2 -/
theorem parabola_directrix (y : ℝ) : 
  let x := -1/8 * y^2
  let a := -1/8
  let focus_x := 1 / (4 * a)
  let directrix_x := -focus_x
  directrix_x = 1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3503_350304


namespace NUMINAMATH_CALUDE_certain_number_problem_l3503_350352

theorem certain_number_problem (x y : ℝ) : 
  0.12 / x * y = 12 ∧ x = 0.1 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3503_350352


namespace NUMINAMATH_CALUDE_pokemon_card_difference_l3503_350397

-- Define the initial number of cards for Sally and Dan
def sally_initial : ℕ := 27
def dan_cards : ℕ := 41

-- Define the number of cards Sally bought
def sally_bought : ℕ := 20

-- Define Sally's total cards after buying
def sally_total : ℕ := sally_initial + sally_bought

-- Theorem to prove
theorem pokemon_card_difference : sally_total - dan_cards = 6 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_card_difference_l3503_350397


namespace NUMINAMATH_CALUDE_smallest_positive_integer_2010m_44000n_l3503_350338

theorem smallest_positive_integer_2010m_44000n : 
  (∃ (k : ℕ), k > 0 ∧ ∃ (m n : ℤ), k = 2010 * m + 44000 * n) ∧ 
  (∀ (k : ℕ), k > 0 → (∃ (m n : ℤ), k = 2010 * m + 44000 * n) → k ≥ 10) ∧
  (∃ (m n : ℤ), 10 = 2010 * m + 44000 * n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_2010m_44000n_l3503_350338


namespace NUMINAMATH_CALUDE_students_at_start_correct_l3503_350387

/-- The number of students at the start of the year in fourth grade -/
def students_at_start : ℕ := 10

/-- The number of students added during the year -/
def students_added : ℝ := 4.0

/-- The number of new students who came to school -/
def new_students : ℝ := 42.0

/-- The total number of students at the end of the year -/
def students_at_end : ℕ := 56

/-- Theorem stating that the number of students at the start of the year is correct -/
theorem students_at_start_correct :
  students_at_start + (students_added + new_students) = students_at_end := by
  sorry

end NUMINAMATH_CALUDE_students_at_start_correct_l3503_350387


namespace NUMINAMATH_CALUDE_middle_group_frequency_l3503_350326

/-- Represents a frequency distribution histogram -/
structure FrequencyHistogram where
  rectangles : Fin 5 → ℝ
  total_sample_size : ℝ
  middle_rectangle_condition : rectangles 2 = (1/3) * (rectangles 0 + rectangles 1 + rectangles 3 + rectangles 4)
  total_area_condition : rectangles 0 + rectangles 1 + rectangles 2 + rectangles 3 + rectangles 4 = total_sample_size

/-- The theorem stating that the frequency of the middle group is 25 -/
theorem middle_group_frequency (h : FrequencyHistogram) (h_sample_size : h.total_sample_size = 100) :
  h.rectangles 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_middle_group_frequency_l3503_350326


namespace NUMINAMATH_CALUDE_probability_odd_limit_l3503_350378

/-- Represents the probability of getting an odd number after n button presses -/
def probability_odd (n : ℕ) : ℝ := sorry

/-- The recurrence relation for the probability of getting an odd number -/
axiom probability_recurrence (n : ℕ) : 
  probability_odd (n + 1) = probability_odd n - (1/2) * (probability_odd n)^2

/-- The initial probability (after one button press) is not exactly 1/3 -/
axiom initial_probability : probability_odd 1 ≠ 1/3

/-- Theorem: The probability of getting an odd number converges to 1/3 -/
theorem probability_odd_limit : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |probability_odd n - 1/3| < ε :=
sorry

end NUMINAMATH_CALUDE_probability_odd_limit_l3503_350378


namespace NUMINAMATH_CALUDE_equidistant_points_on_line_l3503_350342

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 4 * x - 3 * y = 12

-- Define the equidistant condition
def equidistant (x y : ℝ) : Prop := abs x = abs y

-- Define quadrants
def quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0
def quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0
def quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0
def quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem equidistant_points_on_line :
  (∃ x y : ℝ, line_equation x y ∧ equidistant x y ∧ quadrant_I x y) ∧
  (∃ x y : ℝ, line_equation x y ∧ equidistant x y ∧ quadrant_IV x y) ∧
  (¬∃ x y : ℝ, line_equation x y ∧ equidistant x y ∧ quadrant_II x y) ∧
  (¬∃ x y : ℝ, line_equation x y ∧ equidistant x y ∧ quadrant_III x y) :=
sorry

end NUMINAMATH_CALUDE_equidistant_points_on_line_l3503_350342


namespace NUMINAMATH_CALUDE_existence_of_nondivisible_power_l3503_350337

theorem existence_of_nondivisible_power (a b c : ℕ+) (h : Nat.gcd a b.val = 1 ∧ Nat.gcd (Nat.gcd a b.val) c.val = 1) :
  ∃ n : ℕ+, ∀ k : ℕ+, ¬(2^n.val ∣ a^k.val + b^k.val + c^k.val) :=
sorry

end NUMINAMATH_CALUDE_existence_of_nondivisible_power_l3503_350337


namespace NUMINAMATH_CALUDE_log_equation_solution_l3503_350361

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  4 * (Real.log x / Real.log 3) = Real.log (4 * x^2) / Real.log 3 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3503_350361


namespace NUMINAMATH_CALUDE_combined_tax_rate_l3503_350311

/-- Given two individuals with different tax rates and incomes, calculate their combined tax rate -/
theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.40) 
  (h2 : mindy_rate = 0.25) 
  (h3 : income_ratio = 4) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l3503_350311


namespace NUMINAMATH_CALUDE_expected_blue_correct_without_replacement_more_reliable_l3503_350368

-- Define the population
def total_population : ℕ := 200
def blue_population : ℕ := 120
def pink_population : ℕ := 80

-- Define the sample sizes
def sample_size_small : ℕ := 2
def sample_size_large : ℕ := 10

-- Define the true proportion of blue items
def true_proportion : ℚ := blue_population / total_population

-- Part 1: Expected number of blue items in small sample
def expected_blue_small_sample : ℚ := 6/5

-- Part 2: Probabilities for large sample
def prob_within_error_with_replacement : ℚ := 66647/100000
def prob_within_error_without_replacement : ℚ := 67908/100000

-- Theorem statements
theorem expected_blue_correct : 
  ∀ (sampling_method : String), 
  (sampling_method = "with_replacement" ∨ sampling_method = "without_replacement") → 
  expected_blue_small_sample = sample_size_small * true_proportion :=
sorry

theorem without_replacement_more_reliable :
  prob_within_error_without_replacement > prob_within_error_with_replacement :=
sorry

end NUMINAMATH_CALUDE_expected_blue_correct_without_replacement_more_reliable_l3503_350368


namespace NUMINAMATH_CALUDE_inequality_proof_l3503_350357

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  2 ≤ (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ∧ 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≤ (1 + a) * (1 + b) * (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3503_350357


namespace NUMINAMATH_CALUDE_smallest_towel_sets_l3503_350396

def hand_towels_per_set : ℕ := 23
def bath_towels_per_set : ℕ := 29

def total_towels (sets : ℕ) : ℕ :=
  sets * hand_towels_per_set + sets * bath_towels_per_set

theorem smallest_towel_sets :
  ∃ (sets : ℕ),
    (500 ≤ total_towels sets) ∧
    (total_towels sets ≤ 700) ∧
    (∀ (other_sets : ℕ),
      (500 ≤ total_towels other_sets) ∧
      (total_towels other_sets ≤ 700) →
      sets ≤ other_sets) ∧
    sets * hand_towels_per_set = 230 ∧
    sets * bath_towels_per_set = 290 :=
by sorry

end NUMINAMATH_CALUDE_smallest_towel_sets_l3503_350396


namespace NUMINAMATH_CALUDE_redistribution_result_l3503_350381

/-- Represents the number of marbles each person has -/
structure MarbleCount where
  tyrone : ℚ
  eric : ℚ

/-- The initial distribution of marbles -/
def initial : MarbleCount := { tyrone := 125, eric := 25 }

/-- The number of marbles Tyrone gives to Eric -/
def marbles_given : ℚ := 12.5

/-- The final distribution of marbles after Tyrone gives some to Eric -/
def final : MarbleCount :=
  { tyrone := initial.tyrone - marbles_given,
    eric := initial.eric + marbles_given }

/-- Theorem stating that after redistribution, Tyrone has three times as many marbles as Eric -/
theorem redistribution_result :
  final.tyrone = 3 * final.eric := by sorry

end NUMINAMATH_CALUDE_redistribution_result_l3503_350381


namespace NUMINAMATH_CALUDE_sin_period_l3503_350315

theorem sin_period (x : ℝ) : 
  let f : ℝ → ℝ := fun x => Real.sin ((1/2) * x + 3)
  ∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sin_period_l3503_350315


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3503_350330

theorem division_remainder_problem (j : ℕ+) (h : ∃ b : ℕ, 120 = b * j^2 + 12) :
  ∃ k : ℕ, 180 = k * j + 0 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3503_350330


namespace NUMINAMATH_CALUDE_vector_properties_l3503_350355

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 2]

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (c : ℝ), ∀ i, v i = c * w i

def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0 * w 0 + v 1 * w 1) = 0

theorem vector_properties :
  (∃ k : ℝ, parallel (fun i => k * (a i) + b i) (fun i => a i - 2 * (b i))) ∧
  perpendicular (fun i => (25/3) * (a i) + b i) (fun i => a i - 2 * (b i)) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l3503_350355


namespace NUMINAMATH_CALUDE_annual_earnings_difference_l3503_350347

/-- Calculates the difference in annual earnings between a new job and an old job -/
theorem annual_earnings_difference
  (new_wage : ℝ)
  (new_hours : ℝ)
  (old_wage : ℝ)
  (old_hours : ℝ)
  (weeks_per_year : ℝ)
  (h1 : new_wage = 20)
  (h2 : new_hours = 40)
  (h3 : old_wage = 16)
  (h4 : old_hours = 25)
  (h5 : weeks_per_year = 52) :
  new_wage * new_hours * weeks_per_year - old_wage * old_hours * weeks_per_year = 20800 := by
  sorry

#check annual_earnings_difference

end NUMINAMATH_CALUDE_annual_earnings_difference_l3503_350347


namespace NUMINAMATH_CALUDE_factory_uses_systematic_sampling_l3503_350345

/-- Represents a sampling method used in quality control --/
inductive SamplingMethod
| Systematic
| Random
| Stratified
| Cluster

/-- Represents a factory with a conveyor belt and quality inspection process --/
structure Factory where
  /-- The interval between product inspections in minutes --/
  inspection_interval : ℕ
  /-- Whether the inspection position on the conveyor belt is fixed --/
  fixed_position : Bool

/-- Determines if a given factory uses systematic sampling --/
def uses_systematic_sampling (f : Factory) : Prop :=
  f.inspection_interval > 0 ∧ f.fixed_position

/-- The factory described in the problem --/
def problem_factory : Factory :=
  { inspection_interval := 10
  , fixed_position := true }

/-- Theorem stating that the factory in the problem uses systematic sampling --/
theorem factory_uses_systematic_sampling :
  uses_systematic_sampling problem_factory :=
sorry

end NUMINAMATH_CALUDE_factory_uses_systematic_sampling_l3503_350345


namespace NUMINAMATH_CALUDE_angle_terminal_side_l3503_350328

/-- Two angles have the same terminal side if they differ by a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α - β = k * 360

/-- The expression for angles with the same terminal side as -463° -/
def angle_expression (k : ℤ) : ℝ := k * 360 + 257

theorem angle_terminal_side :
  ∀ k : ℤ, same_terminal_side (angle_expression k) (-463) :=
by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l3503_350328
