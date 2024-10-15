import Mathlib

namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l2038_203844

/-- The equation of a conic section -/
def conicEquation (x y : ℝ) : Prop :=
  (x - 3)^2 = 4 * (y + 2)^2 + 25

/-- Definition of a hyperbola -/
def isHyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
    ∀ x y, f x y ↔ a * x^2 + b * y^2 + c * x + d * y + e = 0

/-- Theorem: The given conic equation represents a hyperbola -/
theorem conic_is_hyperbola : isHyperbola conicEquation :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l2038_203844


namespace NUMINAMATH_CALUDE_system_solution_l2038_203826

theorem system_solution (a b c x y z : ℝ) 
  (eq1 : x * (x + y + z) = a^2)
  (eq2 : y * (x + y + z) = b^2)
  (eq3 : z * (x + y + z) = c^2) :
  (x = a^2 / Real.sqrt (a^2 + b^2 + c^2) ∧ 
   y = b^2 / Real.sqrt (a^2 + b^2 + c^2) ∧ 
   z = c^2 / Real.sqrt (a^2 + b^2 + c^2)) ∨
  (x = -a^2 / Real.sqrt (a^2 + b^2 + c^2) ∧ 
   y = -b^2 / Real.sqrt (a^2 + b^2 + c^2) ∧ 
   z = -c^2 / Real.sqrt (a^2 + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2038_203826


namespace NUMINAMATH_CALUDE_rectangle_width_l2038_203842

theorem rectangle_width (area : ℝ) (perimeter : ℝ) (width : ℝ) (length : ℝ) :
  area = 50 →
  perimeter = 30 →
  area = length * width →
  perimeter = 2 * (length + width) →
  width = 5 ∨ width = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l2038_203842


namespace NUMINAMATH_CALUDE_intersection_point_l2038_203879

/-- The x-coordinate of the intersection point of y = 2x - 1 and y = x + 1 -/
def x : ℝ := 2

/-- The y-coordinate of the intersection point of y = 2x - 1 and y = x + 1 -/
def y : ℝ := 3

/-- The first linear function -/
def f (x : ℝ) : ℝ := 2 * x - 1

/-- The second linear function -/
def g (x : ℝ) : ℝ := x + 1

theorem intersection_point :
  f x = y ∧ g x = y ∧ f x = g x :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l2038_203879


namespace NUMINAMATH_CALUDE_line_x_axis_intersection_l2038_203856

/-- The line equation 5y - 6x = 15 intersects the x-axis at the point (-2.5, 0) -/
theorem line_x_axis_intersection :
  ∃! (x : ℝ), 5 * 0 - 6 * x = 15 ∧ x = -2.5 := by
  sorry

end NUMINAMATH_CALUDE_line_x_axis_intersection_l2038_203856


namespace NUMINAMATH_CALUDE_mother_age_twice_alex_l2038_203838

/-- Alex's birth year -/
def alexBirthYear : ℕ := 2000

/-- The year when Alex's mother's age was five times his age -/
def referenceYear : ℕ := 2010

/-- Alex's mother's age is five times Alex's age in the reference year -/
axiom mother_age_five_times (y : ℕ) : y - alexBirthYear = 10 → y = referenceYear → 
  ∃ (motherAge : ℕ), motherAge = 5 * (y - alexBirthYear)

/-- The year when Alex's mother's age will be twice his age -/
def targetYear : ℕ := 2040

theorem mother_age_twice_alex :
  ∃ (motherAge alexAge : ℕ),
    motherAge = 2 * alexAge ∧
    alexAge = targetYear - alexBirthYear ∧
    motherAge = (referenceYear - alexBirthYear) * 5 + (targetYear - referenceYear) :=
sorry

end NUMINAMATH_CALUDE_mother_age_twice_alex_l2038_203838


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2038_203840

theorem sum_of_coefficients (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (1 - 2*x)^5 = a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5 →
  a₀ + a₁ + a₂ + a₃ + a₄ = 33 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2038_203840


namespace NUMINAMATH_CALUDE_no_positive_integer_pairs_l2038_203818

theorem no_positive_integer_pairs : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x > y ∧ (x^2 : ℝ) + y^2 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_pairs_l2038_203818


namespace NUMINAMATH_CALUDE_fraction_five_times_seven_over_ten_l2038_203845

theorem fraction_five_times_seven_over_ten : (5 * 7) / 10 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_five_times_seven_over_ten_l2038_203845


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2038_203892

theorem quadratic_one_solution_sum (a : ℝ) : 
  (∃ (a₁ a₂ : ℝ), 
    (∀ x : ℝ, 3 * x^2 + a₁ * x + 6 * x + 7 = 0 ↔ x = -((a₁ + 6) / 6)) ∧
    (∀ x : ℝ, 3 * x^2 + a₂ * x + 6 * x + 7 = 0 ↔ x = -((a₂ + 6) / 6)) ∧
    a₁ ≠ a₂ ∧ 
    (∀ a' : ℝ, a' ≠ a₁ ∧ a' ≠ a₂ → 
      ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 3 * x₁^2 + a' * x₁ + 6 * x₁ + 7 = 0 ∧ 
                     3 * x₂^2 + a' * x₂ + 6 * x₂ + 7 = 0)) →
  a₁ + a₂ = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l2038_203892


namespace NUMINAMATH_CALUDE_isosceles_minimizes_perimeter_l2038_203827

/-- Given a base length and area, the isosceles triangle minimizes the sum of the other two sides -/
theorem isosceles_minimizes_perimeter (a S : ℝ) (ha : a > 0) (hS : S > 0) :
  ∃ (h : ℝ), h > 0 ∧
  ∀ (b c : ℝ), b > 0 → c > 0 →
  (a * h / 2 = S) →
  (a * (b^2 - h^2).sqrt / 2 = S) →
  (a * (c^2 - h^2).sqrt / 2 = S) →
  b + c ≥ 2 * (4 * S^2 / a^2 + a^2 / 4).sqrt :=
sorry

end NUMINAMATH_CALUDE_isosceles_minimizes_perimeter_l2038_203827


namespace NUMINAMATH_CALUDE_pizza_theorem_l2038_203841

def pizza_problem (total_slices : ℕ) (slices_per_person : ℕ) : ℕ :=
  (total_slices / slices_per_person) - 1

theorem pizza_theorem :
  pizza_problem 12 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l2038_203841


namespace NUMINAMATH_CALUDE_student_number_calculation_l2038_203846

theorem student_number_calculation (x : ℕ) (h : x = 129) : 2 * x - 148 = 110 := by
  sorry

end NUMINAMATH_CALUDE_student_number_calculation_l2038_203846


namespace NUMINAMATH_CALUDE_chi_square_test_error_probability_l2038_203858

/-- Represents the chi-square statistic -/
def chi_square : ℝ := 15.02

/-- Represents the critical value -/
def critical_value : ℝ := 6.635

/-- Represents the p-value -/
def p_value : ℝ := 0.01

/-- Represents the sample size -/
def sample_size : ℕ := 1000

/-- Represents the probability of error in rejecting the null hypothesis -/
def error_probability : ℝ := p_value

theorem chi_square_test_error_probability :
  error_probability = p_value :=
sorry

end NUMINAMATH_CALUDE_chi_square_test_error_probability_l2038_203858


namespace NUMINAMATH_CALUDE_binomial_20_9_l2038_203863

theorem binomial_20_9 (h1 : Nat.choose 18 7 = 31824)
                      (h2 : Nat.choose 18 8 = 43758)
                      (h3 : Nat.choose 18 9 = 43758) :
  Nat.choose 20 9 = 163098 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_9_l2038_203863


namespace NUMINAMATH_CALUDE_min_value_of_y_l2038_203839

theorem min_value_of_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → 1/x + 4/y ≥ 9/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ 1/x + 4/y = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_y_l2038_203839


namespace NUMINAMATH_CALUDE_positive_intervals_l2038_203876

theorem positive_intervals (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_intervals_l2038_203876


namespace NUMINAMATH_CALUDE_jerrys_cartridge_cost_l2038_203806

/-- The total cost of printer cartridges for Jerry -/
def total_cost (color_cartridge_cost : ℕ) (bw_cartridge_cost : ℕ) (color_cartridge_count : ℕ) (bw_cartridge_count : ℕ) : ℕ :=
  color_cartridge_cost * color_cartridge_count + bw_cartridge_cost * bw_cartridge_count

/-- Theorem: Jerry's total cost for printer cartridges is $123 -/
theorem jerrys_cartridge_cost :
  total_cost 32 27 3 1 = 123 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_cartridge_cost_l2038_203806


namespace NUMINAMATH_CALUDE_f_g_f_1_equals_102_l2038_203832

def f (x : ℝ) : ℝ := 5 * x + 2

def g (x : ℝ) : ℝ := 3 * x - 1

theorem f_g_f_1_equals_102 : f (g (f 1)) = 102 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_1_equals_102_l2038_203832


namespace NUMINAMATH_CALUDE_periodic_function_value_l2038_203885

def periodic_function (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x : ℝ, f (x + period) = f x

theorem periodic_function_value 
  (f : ℝ → ℝ) 
  (h_periodic : periodic_function f (π / 2))
  (h_value : f (π / 3) = 1) : 
  f (17 * π / 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l2038_203885


namespace NUMINAMATH_CALUDE_onion_weight_problem_l2038_203894

theorem onion_weight_problem (total_weight : Real) (total_count : Nat) (removed_count : Nat) (removed_avg : Real) (remaining_count : Nat) :
  total_weight = 7.68 →
  total_count = 40 →
  removed_count = 5 →
  removed_avg = 0.206 →
  remaining_count = total_count - removed_count →
  let remaining_weight := total_weight - (removed_count * removed_avg)
  let remaining_avg := remaining_weight / remaining_count
  remaining_avg = 0.190 := by
sorry

end NUMINAMATH_CALUDE_onion_weight_problem_l2038_203894


namespace NUMINAMATH_CALUDE_parallel_lines_interior_alternate_angles_l2038_203813

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- A line intersects two other lines -/
def intersects (l : Line) (l1 l2 : Line) : Prop := sorry

/-- Interior alternate angles between two lines and a transversal -/
def interior_alternate_angles (l1 l2 l : Line) (α β : Angle) : Prop := sorry

/-- The proposition about parallel lines and interior alternate angles -/
theorem parallel_lines_interior_alternate_angles 
  (l1 l2 l : Line) (α β : Angle) :
  parallel l1 l2 → 
  intersects l l1 l2 → 
  interior_alternate_angles l1 l2 l α β → 
  α = β := 
sorry

end NUMINAMATH_CALUDE_parallel_lines_interior_alternate_angles_l2038_203813


namespace NUMINAMATH_CALUDE_total_groceries_l2038_203824

def cookies : ℕ := 12
def noodles : ℕ := 16

theorem total_groceries : cookies + noodles = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_groceries_l2038_203824


namespace NUMINAMATH_CALUDE_b_bounded_a_value_l2038_203820

/-- A quadratic function f(x) = ax^2 + bx + c with certain properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  bounded : ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1

/-- The coefficient b of a QuadraticFunction is bounded by 1 -/
theorem b_bounded (f : QuadraticFunction) : |f.b| ≤ 1 := by sorry

/-- If f(0) = -1 and f(1) = 1, then a = 2 -/
theorem a_value (f : QuadraticFunction) 
  (h0 : f.c = -1) 
  (h1 : f.a + f.b + f.c = 1) : 
  f.a = 2 := by sorry

end NUMINAMATH_CALUDE_b_bounded_a_value_l2038_203820


namespace NUMINAMATH_CALUDE_machine_work_time_l2038_203805

theorem machine_work_time (x : ℝ) : 
  (x > 0) →
  (1 / (x + 4) + 1 / (x + 2) + 1 / (2 * x + 2) = 1 / x) →
  x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_machine_work_time_l2038_203805


namespace NUMINAMATH_CALUDE_independence_test_most_appropriate_l2038_203866

/-- Represents the survey data in a 2x2 contingency table --/
structure SurveyData where
  male_total : ℕ
  male_doping : ℕ
  female_total : ℕ
  female_framed : ℕ

/-- Represents different statistical methods --/
inductive StatMethod
  | MeanVariance
  | RegressionAnalysis
  | IndependenceTest
  | Probability

/-- Checks if a method is most appropriate for analyzing the given survey data --/
def is_most_appropriate (method : StatMethod) (data : SurveyData) : Prop :=
  method = StatMethod.IndependenceTest

/-- The main theorem stating that the Independence Test is the most appropriate method --/
theorem independence_test_most_appropriate (data : SurveyData) :
  is_most_appropriate StatMethod.IndependenceTest data :=
sorry

end NUMINAMATH_CALUDE_independence_test_most_appropriate_l2038_203866


namespace NUMINAMATH_CALUDE_product_difference_square_equals_negative_one_l2038_203833

theorem product_difference_square_equals_negative_one :
  2021 * 2023 - 2022^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_square_equals_negative_one_l2038_203833


namespace NUMINAMATH_CALUDE_rectangle_width_l2038_203870

/-- Given a rectangle with perimeter 6a + 4b and length 2a + b, prove its width is a + b -/
theorem rectangle_width (a b : ℝ) : 
  let perimeter := 6*a + 4*b
  let length := 2*a + b
  let width := (perimeter / 2) - length
  width = a + b := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l2038_203870


namespace NUMINAMATH_CALUDE_numbers_close_to_zero_not_set_l2038_203807

-- Define the property of being "close to 0"
def CloseToZero (x : ℝ) : Prop := sorry

-- Define the criteria for set formation
structure SetCriteria :=
  (definiteness : Prop)
  (distinctness : Prop)
  (unorderedness : Prop)

-- Define a function to check if a collection satisfies set criteria
def SatisfiesSetCriteria (S : Set ℝ) (criteria : SetCriteria) : Prop := sorry

-- Theorem stating that "numbers close to 0" cannot form a set
theorem numbers_close_to_zero_not_set :
  ¬ ∃ (S : Set ℝ) (criteria : SetCriteria), 
    (∀ x ∈ S, CloseToZero x) ∧ 
    SatisfiesSetCriteria S criteria :=
sorry

end NUMINAMATH_CALUDE_numbers_close_to_zero_not_set_l2038_203807


namespace NUMINAMATH_CALUDE_first_four_seeds_l2038_203861

/-- Represents a row in the random number table -/
def RandomTableRow := List Nat

/-- The random number table -/
def randomTable : List RandomTableRow := [
  [78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279],
  [43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820],
  [61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636],
  [63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421],
  [42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983]
]

/-- The starting position in the random number table -/
def startPosition : Nat × Nat := (2, 5)

/-- The total number of seeds -/
def totalSeeds : Nat := 850

/-- Function to get the next valid seed number -/
def getNextValidSeed (table : List RandomTableRow) (pos : Nat × Nat) (maxSeed : Nat) : Option (Nat × (Nat × Nat)) :=
  sorry

/-- Theorem stating that the first 4 valid seed numbers are 390, 737, 220, and 372 -/
theorem first_four_seeds :
  let seedNumbers := [390, 737, 220, 372]
  ∃ (pos1 pos2 pos3 pos4 : Nat × Nat),
    getNextValidSeed randomTable startPosition totalSeeds = some (seedNumbers[0], pos1) ∧
    getNextValidSeed randomTable pos1 totalSeeds = some (seedNumbers[1], pos2) ∧
    getNextValidSeed randomTable pos2 totalSeeds = some (seedNumbers[2], pos3) ∧
    getNextValidSeed randomTable pos3 totalSeeds = some (seedNumbers[3], pos4) :=
  sorry

end NUMINAMATH_CALUDE_first_four_seeds_l2038_203861


namespace NUMINAMATH_CALUDE_max_sum_with_length_constraint_l2038_203810

-- Define the length of an integer as the number of prime factors
def length (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem max_sum_with_length_constraint :
  ∀ x y : ℕ,
    x > 1 →
    y > 1 →
    length x + length y ≤ 16 →
    x + 3 * y ≤ 98306 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_length_constraint_l2038_203810


namespace NUMINAMATH_CALUDE_point_A_on_line_l_l2038_203815

/-- A line passing through the origin with slope -2 -/
def line_l (x y : ℝ) : Prop := y = -2 * x

/-- The point (1, -2) -/
def point_A : ℝ × ℝ := (1, -2)

/-- Theorem: The point (1, -2) lies on the line l -/
theorem point_A_on_line_l : line_l point_A.1 point_A.2 := by sorry

end NUMINAMATH_CALUDE_point_A_on_line_l_l2038_203815


namespace NUMINAMATH_CALUDE_square_root_squared_specific_square_root_squared_l2038_203854

theorem square_root_squared (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by sorry

theorem specific_square_root_squared : (Real.sqrt 978121) ^ 2 = 978121 := by
  apply square_root_squared
  norm_num


end NUMINAMATH_CALUDE_square_root_squared_specific_square_root_squared_l2038_203854


namespace NUMINAMATH_CALUDE_right_triangular_pyramid_relation_l2038_203851

/-- Right triangular pyramid with pairwise perpendicular side edges -/
structure RightTriangularPyramid where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  h_pos : 0 < h

/-- The relationship between side edges and altitude in a right triangular pyramid -/
theorem right_triangular_pyramid_relation (p : RightTriangularPyramid) :
  1 / p.a ^ 2 + 1 / p.b ^ 2 + 1 / p.c ^ 2 = 1 / p.h ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_pyramid_relation_l2038_203851


namespace NUMINAMATH_CALUDE_polygonal_chain_circle_cover_l2038_203899

/-- A planar closed polygonal chain -/
structure ClosedPolygonalChain where
  vertices : Set (ℝ × ℝ)
  is_closed : True  -- This is a placeholder for the closure property
  perimeter : ℝ

/-- Theorem: For any closed polygonal chain with perimeter 1, 
    there exists a point such that all points on the chain 
    are within distance 1/4 from it -/
theorem polygonal_chain_circle_cover 
  (chain : ClosedPolygonalChain) 
  (h_perimeter : chain.perimeter = 1) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ chain.vertices → 
    Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_polygonal_chain_circle_cover_l2038_203899


namespace NUMINAMATH_CALUDE_B_necessary_not_sufficient_l2038_203830

def A (x : ℝ) : Prop := 0 < x ∧ x < 5

def B (x : ℝ) : Prop := |x - 2| < 3

theorem B_necessary_not_sufficient :
  (∀ x, A x → B x) ∧ (∃ x, B x ∧ ¬A x) := by
  sorry

end NUMINAMATH_CALUDE_B_necessary_not_sufficient_l2038_203830


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l2038_203817

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (x + 1/x) ≤ Real.sqrt 15 ∧ ∃ y : ℝ, y + 1/y = Real.sqrt 15 ∧ 13 = y^2 + 1/y^2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l2038_203817


namespace NUMINAMATH_CALUDE_smallest_cookie_count_l2038_203864

theorem smallest_cookie_count : ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → 4*m - 4 = (m^2)/2 → m ≥ n) ∧ 4*n - 4 = (n^2)/2 ∧ n^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_count_l2038_203864


namespace NUMINAMATH_CALUDE_min_a_value_l2038_203872

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x * (x^3 - 3*x + 3) - a * exp x - x

theorem min_a_value :
  ∀ a : ℝ, (∃ x : ℝ, x ≥ -2 ∧ f a x ≤ 0) → a ≥ 1 - 1/exp 1 :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l2038_203872


namespace NUMINAMATH_CALUDE_number_difference_equation_l2038_203873

theorem number_difference_equation (x : ℝ) : 
  0.62 * x - 0.20 * 250 = 43 → x = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_equation_l2038_203873


namespace NUMINAMATH_CALUDE_janous_inequality_l2038_203809

theorem janous_inequality (α x y z : ℝ) (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = α) :
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ 16 * (x / z + z / x + 2) := by
  sorry

end NUMINAMATH_CALUDE_janous_inequality_l2038_203809


namespace NUMINAMATH_CALUDE_original_number_proof_l2038_203883

theorem original_number_proof : ∃ x : ℝ, 16 * x = 3408 ∧ x = 213 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2038_203883


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l2038_203821

theorem midpoint_sum_equals_vertex_sum (a b : ℝ) 
  (h : a + b + (a + 5) = 15) : 
  (a + b) / 2 + (2 * a + 5) / 2 + (b + a + 5) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l2038_203821


namespace NUMINAMATH_CALUDE_english_score_l2038_203857

theorem english_score (korean math : ℕ) (h1 : (korean + math) / 2 = 88) 
  (h2 : (korean + math + 94) / 3 = 90) : 94 = 94 := by
  sorry

end NUMINAMATH_CALUDE_english_score_l2038_203857


namespace NUMINAMATH_CALUDE_students_taking_none_in_high_school_l2038_203889

/-- The number of students taking neither music, nor art, nor science in a high school -/
def students_taking_none (total : ℕ) (music art science : ℕ) (music_and_art music_and_science art_and_science : ℕ) (all_three : ℕ) : ℕ :=
  total - (music + art + science - music_and_art - music_and_science - art_and_science + all_three)

/-- Theorem stating the number of students taking neither music, nor art, nor science -/
theorem students_taking_none_in_high_school :
  students_taking_none 800 80 60 50 30 25 20 15 = 670 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_none_in_high_school_l2038_203889


namespace NUMINAMATH_CALUDE_negation_of_exponential_inequality_l2038_203867

theorem negation_of_exponential_inequality :
  (¬ (∀ x : ℝ, Real.exp x ≥ 1)) ↔ (∃ x : ℝ, Real.exp x < 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_exponential_inequality_l2038_203867


namespace NUMINAMATH_CALUDE_parrots_per_cage_l2038_203825

/-- Given a pet store with birds, calculate the number of parrots per cage. -/
theorem parrots_per_cage
  (num_cages : ℕ)
  (parakeets_per_cage : ℕ)
  (total_birds : ℕ)
  (h1 : num_cages = 6)
  (h2 : parakeets_per_cage = 7)
  (h3 : total_birds = 54) :
  (total_birds - num_cages * parakeets_per_cage) / num_cages = 2 :=
by sorry

end NUMINAMATH_CALUDE_parrots_per_cage_l2038_203825


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2038_203828

theorem geometric_sequence_sixth_term 
  (a : ℕ → ℝ) 
  (n : ℕ) 
  (h1 : n = 9) 
  (h2 : a 1 = 9) 
  (h3 : a n = 26244) 
  (h4 : ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ n → a j / a i = a (i + 1) / a i) : 
  a 6 = 2187 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2038_203828


namespace NUMINAMATH_CALUDE_square_difference_equality_l2038_203897

theorem square_difference_equality : 1013^2 - 991^2 - 1007^2 + 997^2 = 24048 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2038_203897


namespace NUMINAMATH_CALUDE_limit_at_infinity_limit_at_point_l2038_203814

-- Part 1
theorem limit_at_infinity (ε : ℝ) (hε : ε > 0) :
  ∃ M : ℝ, ∀ x : ℝ, x > M → |(2*x + 3)/(3*x) - 2/3| < ε :=
sorry

-- Part 2
theorem limit_at_point (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x - 3| ∧ |x - 3| < δ → |(2*x + 1) - 7| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_at_infinity_limit_at_point_l2038_203814


namespace NUMINAMATH_CALUDE_favorite_movies_total_duration_l2038_203802

/-- Given the durations of four people's favorite movies with specific relationships,
    prove that the total duration of all movies is 76 hours. -/
theorem favorite_movies_total_duration
  (michael_duration : ℝ)
  (joyce_duration : ℝ)
  (nikki_duration : ℝ)
  (ryn_duration : ℝ)
  (h1 : joyce_duration = michael_duration + 2)
  (h2 : nikki_duration = 3 * michael_duration)
  (h3 : ryn_duration = 4/5 * nikki_duration)
  (h4 : nikki_duration = 30) :
  joyce_duration + michael_duration + nikki_duration + ryn_duration = 76 := by
sorry


end NUMINAMATH_CALUDE_favorite_movies_total_duration_l2038_203802


namespace NUMINAMATH_CALUDE_product_with_zero_is_zero_l2038_203882

theorem product_with_zero_is_zero :
  (-2.5) * 0.37 * 1.25 * (-4) * (-8) * 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_with_zero_is_zero_l2038_203882


namespace NUMINAMATH_CALUDE_birds_in_tree_l2038_203811

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29) 
  (h2 : final_birds = 42) : 
  final_birds - initial_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l2038_203811


namespace NUMINAMATH_CALUDE_parabola_coefficient_l2038_203862

/-- A quadratic function of the form y = mx^2 + 2 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 2

/-- The condition for a downward-opening parabola -/
def is_downward_opening (m : ℝ) : Prop := m < 0

theorem parabola_coefficient :
  ∀ m : ℝ, is_downward_opening m → m = -2 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l2038_203862


namespace NUMINAMATH_CALUDE_system_inequality_solution_range_l2038_203877

theorem system_inequality_solution_range (x y m : ℝ) : 
  x - 2*y = 1 → 
  2*x + y = 4*m → 
  x + 3*y < 6 → 
  m < 7/4 := by
sorry

end NUMINAMATH_CALUDE_system_inequality_solution_range_l2038_203877


namespace NUMINAMATH_CALUDE_divisibility_property_l2038_203886

theorem divisibility_property (n : ℕ) (hn : n > 1) : 
  ∃ k : ℤ, n^(n-1) - 1 = (n-1)^2 * k := by sorry

end NUMINAMATH_CALUDE_divisibility_property_l2038_203886


namespace NUMINAMATH_CALUDE_tangent_when_zero_discriminant_l2038_203893

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic function -/
def discriminant (f : QuadraticFunction) : ℝ :=
  f.b^2 - 4 * f.a * f.c

/-- Determines if a quadratic function's graph is tangent to the x-axis -/
def is_tangent_to_x_axis (f : QuadraticFunction) : Prop :=
  ∃ x : ℝ, f.a * x^2 + f.b * x + f.c = 0 ∧
    ∀ y : ℝ, y ≠ x → f.a * y^2 + f.b * y + f.c > 0

/-- The main theorem: if the discriminant is zero, the graph is tangent to the x-axis -/
theorem tangent_when_zero_discriminant (k : ℝ) :
  let f : QuadraticFunction := ⟨3, 9, k⟩
  discriminant f = 0 → is_tangent_to_x_axis f :=
by sorry

end NUMINAMATH_CALUDE_tangent_when_zero_discriminant_l2038_203893


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l2038_203884

-- Define the problem
theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l2038_203884


namespace NUMINAMATH_CALUDE_production_cost_decrease_rate_l2038_203896

theorem production_cost_decrease_rate : ∃ x : ℝ, 
  (400 * (1 - x)^2 = 361) ∧ (x = 0.05) := by sorry

end NUMINAMATH_CALUDE_production_cost_decrease_rate_l2038_203896


namespace NUMINAMATH_CALUDE_smallest_five_digit_palindrome_div_by_6_l2038_203871

/-- A function that checks if a number is a five-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The theorem stating that 20002 is the smallest five-digit palindrome divisible by 6 -/
theorem smallest_five_digit_palindrome_div_by_6 :
  ∀ n : ℕ, is_five_digit_palindrome n ∧ n % 6 = 0 → n ≥ 20002 :=
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_palindrome_div_by_6_l2038_203871


namespace NUMINAMATH_CALUDE_ashley_age_is_8_l2038_203880

-- Define Ashley's and Mary's ages as natural numbers
variable (ashley_age mary_age : ℕ)

-- Define the conditions
def age_ratio : Prop := ashley_age * 7 = mary_age * 4
def age_sum : Prop := ashley_age + mary_age = 22

-- State the theorem
theorem ashley_age_is_8 
  (h1 : age_ratio ashley_age mary_age) 
  (h2 : age_sum ashley_age mary_age) : 
  ashley_age = 8 := by
sorry

end NUMINAMATH_CALUDE_ashley_age_is_8_l2038_203880


namespace NUMINAMATH_CALUDE_unacceptable_weight_l2038_203835

def acceptable_range (x : ℝ) : Prop := 49.7 ≤ x ∧ x ≤ 50.3

theorem unacceptable_weight : ¬(acceptable_range 49.6) := by
  sorry

end NUMINAMATH_CALUDE_unacceptable_weight_l2038_203835


namespace NUMINAMATH_CALUDE_halloween_candy_problem_l2038_203812

/-- Given Katie's candy count, her sister's candy count, and the number of pieces eaten,
    calculate the remaining candy pieces. -/
theorem halloween_candy_problem (katie_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) :
  katie_candy = 10 →
  sister_candy = 6 →
  eaten_candy = 9 →
  katie_candy + sister_candy - eaten_candy = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_problem_l2038_203812


namespace NUMINAMATH_CALUDE_average_of_quadratic_solutions_l2038_203874

theorem average_of_quadratic_solutions (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 4 * x₁ + 1 = 0) → 
  (3 * x₂^2 - 4 * x₂ + 1 = 0) → 
  x₁ ≠ x₂ → 
  (x₁ + x₂) / 2 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_average_of_quadratic_solutions_l2038_203874


namespace NUMINAMATH_CALUDE_problem_statement_l2038_203869

theorem problem_statement (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 4 * Real.sqrt 2) = Q) :
  10 * (6 * x + 8 * Real.sqrt 2 - Real.sqrt 2) = 4 * Q - 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2038_203869


namespace NUMINAMATH_CALUDE_supplementary_angle_measure_l2038_203898

-- Define the angle x
def x : ℝ := 10

-- Define the complementary angle
def complementary_angle (x : ℝ) : ℝ := 90 - x

-- Define the supplementary angle
def supplementary_angle (x : ℝ) : ℝ := 180 - x

-- Theorem statement
theorem supplementary_angle_measure :
  (x / complementary_angle x = 1 / 8) →
  supplementary_angle x = 170 := by
  sorry

end NUMINAMATH_CALUDE_supplementary_angle_measure_l2038_203898


namespace NUMINAMATH_CALUDE_different_color_probability_l2038_203853

/-- The set of colors for shorts -/
def shorts_colors : Finset String := {"black", "gold", "silver"}

/-- The set of colors for jerseys -/
def jersey_colors : Finset String := {"black", "white", "gold"}

/-- The probability of selecting different colors for shorts and jerseys -/
theorem different_color_probability : 
  (shorts_colors.card * jersey_colors.card - (shorts_colors ∩ jersey_colors).card) / 
  (shorts_colors.card * jersey_colors.card : ℚ) = 7/9 := by
sorry

end NUMINAMATH_CALUDE_different_color_probability_l2038_203853


namespace NUMINAMATH_CALUDE_old_lamp_height_is_one_foot_l2038_203804

-- Define the height of the new lamp
def new_lamp_height : ℝ := 2.3333333333333335

-- Define the difference in height between the new and old lamps
def height_difference : ℝ := 1.3333333333333333

-- Theorem to prove
theorem old_lamp_height_is_one_foot :
  new_lamp_height - height_difference = 1 := by sorry

end NUMINAMATH_CALUDE_old_lamp_height_is_one_foot_l2038_203804


namespace NUMINAMATH_CALUDE_stream_speed_proof_l2038_203801

/-- Proves that the speed of the stream is 21 kmph given the conditions of the rowing problem. -/
theorem stream_speed_proof (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 63 →
  (1 / (boat_speed - stream_speed)) = (2 / (boat_speed + stream_speed)) →
  stream_speed = 21 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_proof_l2038_203801


namespace NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l2038_203822

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 3*x - 4 > 0}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | x ≤ 3 ∨ x > 4} := by sorry

-- Theorem for A ∩ (U \ B)
theorem intersection_A_complement_B : A ∩ (U \ B) = {x | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l2038_203822


namespace NUMINAMATH_CALUDE_common_tangents_possible_values_l2038_203852

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- The number of common tangent lines between two circles -/
def num_common_tangents (c1 c2 : Circle) : ℕ := sorry

/-- Theorem stating the possible values for the number of common tangents -/
theorem common_tangents_possible_values (c1 c2 : Circle) (h : c1 ≠ c2) :
  ∃ n : ℕ, num_common_tangents c1 c2 = n ∧ n ∈ ({0, 1, 2, 3, 4} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_common_tangents_possible_values_l2038_203852


namespace NUMINAMATH_CALUDE_sum_of_M_subset_products_l2038_203895

def M : Set ℚ := {-2/3, 5/4, 1, 4}

def f (x : ℚ) : ℚ := (x + 2/3) * (x - 5/4) * (x - 1) * (x - 4)

def sum_of_subset_products (S : Set ℚ) : ℚ :=
  (f 1) - 1

theorem sum_of_M_subset_products :
  sum_of_subset_products M = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_M_subset_products_l2038_203895


namespace NUMINAMATH_CALUDE_odd_perfect_square_theorem_l2038_203834

/-- 
The sum of divisors function σ(n) is the sum of all positive divisors of n, including n itself.
-/
def sum_of_divisors (n : ℕ+) : ℕ := sorry

/-- 
A number is a perfect square if it is the product of an integer with itself.
-/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem odd_perfect_square_theorem (n : ℕ+) : 
  sum_of_divisors n = 2 * n.val + 1 → Odd n.val ∧ is_perfect_square n.val :=
sorry

end NUMINAMATH_CALUDE_odd_perfect_square_theorem_l2038_203834


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2038_203823

/-- Given a geometric sequence {a_n} with positive terms where a₁, (1/2)a₃, and 2a₂ form an arithmetic sequence,
    prove that (a₉ + a₁₀) / (a₇ + a₈) = 3 + 2√2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∃ q : ℝ, ∀ n, a (n + 1) = q * a n)
  (h_arith : a 1 + 2 * a 2 = a 3) :
  (a 9 + a 10) / (a 7 + a 8) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2038_203823


namespace NUMINAMATH_CALUDE_equal_time_per_style_l2038_203855

-- Define the swimming styles
inductive SwimmingStyle
| FrontCrawl
| Breaststroke
| Backstroke
| Butterfly

-- Define the problem parameters
def totalDistance : ℝ := 600
def totalTime : ℝ := 15
def numStyles : ℕ := 4

-- Define the speed for each style (yards per minute)
def speed (style : SwimmingStyle) : ℝ :=
  match style with
  | SwimmingStyle.FrontCrawl => 45
  | SwimmingStyle.Breaststroke => 35
  | SwimmingStyle.Backstroke => 40
  | SwimmingStyle.Butterfly => 30

-- Theorem to prove
theorem equal_time_per_style :
  ∀ (style : SwimmingStyle),
  (totalTime / numStyles : ℝ) = 3.75 ∧
  (totalDistance / numStyles : ℝ) / speed style ≤ totalTime / numStyles :=
by sorry

end NUMINAMATH_CALUDE_equal_time_per_style_l2038_203855


namespace NUMINAMATH_CALUDE_katie_has_more_games_l2038_203803

/-- The number of games Katie has -/
def katie_games : ℕ := 81

/-- The number of games Katie's friends have -/
def friends_games : ℕ := 59

/-- The difference in games between Katie and her friends -/
def game_difference : ℕ := katie_games - friends_games

theorem katie_has_more_games : game_difference = 22 := by
  sorry

end NUMINAMATH_CALUDE_katie_has_more_games_l2038_203803


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2038_203891

/-- Given a polynomial equation, prove the sum of specific coefficients -/
theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = 
    a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + 
    a₉*(x+1)^9 + a₁₀*(x+1)^10 + a₁₁*(x+1)^11) →
  a₁ + a₂ + a₁₁ = 781 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2038_203891


namespace NUMINAMATH_CALUDE_not_prime_qt_plus_q_plus_t_l2038_203831

theorem not_prime_qt_plus_q_plus_t (q t : ℕ+) (h : q > 1 ∨ t > 1) : 
  ¬ Nat.Prime (q * t + q + t) := by
sorry

end NUMINAMATH_CALUDE_not_prime_qt_plus_q_plus_t_l2038_203831


namespace NUMINAMATH_CALUDE_cube_edge_sum_exists_l2038_203829

/-- Represents the edges of a cube --/
def CubeEdges := Fin 12

/-- Represents the faces of a cube --/
def CubeFaces := Fin 6

/-- A function that assigns numbers to the edges of a cube --/
def EdgeAssignment := CubeEdges → Fin 12

/-- A function that returns the edges that make up a face --/
def FaceEdges : CubeFaces → Finset CubeEdges := sorry

/-- The sum of numbers on a face given an edge assignment --/
def FaceSum (assignment : EdgeAssignment) (face : CubeFaces) : ℕ :=
  (FaceEdges face).sum (fun edge => (assignment edge).val + 1)

/-- Theorem stating that there exists an assignment of numbers from 1 to 12
    to the edges of a cube such that the sum of numbers on each face is equal --/
theorem cube_edge_sum_exists : 
  ∃ (assignment : EdgeAssignment), 
    (∀ (face1 face2 : CubeFaces), FaceSum assignment face1 = FaceSum assignment face2) ∧ 
    (∀ (edge1 edge2 : CubeEdges), edge1 ≠ edge2 → assignment edge1 ≠ assignment edge2) := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_sum_exists_l2038_203829


namespace NUMINAMATH_CALUDE_percentage_relationships_l2038_203860

theorem percentage_relationships (a b c d e f g : ℝ) 
  (h1 : d = 0.22 * b) 
  (h2 : d = 0.35 * f) 
  (h3 : e = 0.27 * a) 
  (h4 : e = 0.60 * f) 
  (h5 : c = 0.14 * a) 
  (h6 : c = 0.40 * b) 
  (h7 : d = 2 * c) 
  (h8 : g = 3 * e) : 
  g = 0.81 * a ∧ b = 0.7 * a ∧ f = 0.45 * a := by
  sorry


end NUMINAMATH_CALUDE_percentage_relationships_l2038_203860


namespace NUMINAMATH_CALUDE_exactly_two_approvals_probability_l2038_203881

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 5

/-- The number of desired successes -/
def k : ℕ := 2

/-- The probability of exactly k successes in n independent trials with probability p -/
def binomial_probability (p : ℝ) (n k : ℕ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_approvals_probability :
  binomial_probability p n k = 0.3648 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_approvals_probability_l2038_203881


namespace NUMINAMATH_CALUDE_two_green_marbles_probability_l2038_203847

/-- The probability of drawing two green marbles consecutively without replacement -/
theorem two_green_marbles_probability 
  (red green white blue : ℕ) 
  (h_red : red = 3)
  (h_green : green = 4)
  (h_white : white = 8)
  (h_blue : blue = 5) : 
  (green : ℚ) / (red + green + white + blue) * 
  ((green - 1) : ℚ) / (red + green + white + blue - 1) = 3 / 95 := by
sorry

end NUMINAMATH_CALUDE_two_green_marbles_probability_l2038_203847


namespace NUMINAMATH_CALUDE_pyarelals_loss_is_1800_l2038_203888

/-- Represents the loss incurred by Pyarelal in a business partnership with Ashok -/
def pyarelals_loss (pyarelals_capital : ℚ) (total_loss : ℚ) : ℚ :=
  (pyarelals_capital / (pyarelals_capital + pyarelals_capital / 9)) * total_loss

/-- Theorem stating that Pyarelal's loss is 1800 given the conditions of the problem -/
theorem pyarelals_loss_is_1800 (pyarelals_capital : ℚ) (h : pyarelals_capital > 0) :
  pyarelals_loss pyarelals_capital 2000 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_pyarelals_loss_is_1800_l2038_203888


namespace NUMINAMATH_CALUDE_variance_best_stability_measure_l2038_203887

/-- A measure of stability for a set of test scores -/
class StabilityMeasure where
  measure : List ℝ → ℝ

/-- Average as a stability measure -/
def average : StabilityMeasure := sorry

/-- Median as a stability measure -/
def median : StabilityMeasure := sorry

/-- Variance as a stability measure -/
def variance : StabilityMeasure := sorry

/-- Mode as a stability measure -/
def mode : StabilityMeasure := sorry

/-- A function that determines if a stability measure is the best for test scores -/
def isBestStabilityMeasure (m : StabilityMeasure) : Prop := sorry

theorem variance_best_stability_measure : isBestStabilityMeasure variance := by
  sorry

end NUMINAMATH_CALUDE_variance_best_stability_measure_l2038_203887


namespace NUMINAMATH_CALUDE_partition_S_l2038_203837

def S : Set ℚ := {-5/6, 0, -7/2, 6/5, 6}

theorem partition_S :
  (∃ (A B : Set ℚ), A ∪ B = S ∧ A ∩ B = ∅ ∧
    A = {x ∈ S | x < 0} ∧
    B = {x ∈ S | x ≥ 0} ∧
    A = {-5/6, -7/2} ∧
    B = {0, 6/5, 6}) :=
by sorry

end NUMINAMATH_CALUDE_partition_S_l2038_203837


namespace NUMINAMATH_CALUDE_brigade_plowing_rates_l2038_203875

/-- Represents the daily plowing rate and work duration of a brigade --/
structure Brigade where
  daily_rate : ℝ
  days_worked : ℝ

/-- Proves that given the problem conditions, the brigades' daily rates are 24 and 27 hectares --/
theorem brigade_plowing_rates 
  (first_brigade second_brigade : Brigade)
  (h1 : first_brigade.daily_rate * first_brigade.days_worked = 240)
  (h2 : second_brigade.daily_rate * second_brigade.days_worked = 240 * 1.35)
  (h3 : second_brigade.daily_rate = first_brigade.daily_rate + 3)
  (h4 : second_brigade.days_worked = first_brigade.days_worked + 2)
  (h5 : first_brigade.daily_rate > 20)
  (h6 : second_brigade.daily_rate > 20)
  : first_brigade.daily_rate = 24 ∧ second_brigade.daily_rate = 27 := by
  sorry

#check brigade_plowing_rates

end NUMINAMATH_CALUDE_brigade_plowing_rates_l2038_203875


namespace NUMINAMATH_CALUDE_equation_solution_l2038_203890

theorem equation_solution (x : ℚ) : 
  5 * x - 6 = 15 * x + 21 → 3 * (x + 5)^2 = 2523 / 100 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2038_203890


namespace NUMINAMATH_CALUDE_geometric_sequence_2010th_term_l2038_203849

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  p : ℝ
  q : ℝ
  first_term : ℝ
  second_term : ℝ
  third_term : ℝ
  fourth_term : ℝ
  h1 : first_term = p
  h2 : second_term = 9
  h3 : third_term = 3 * p / q
  h4 : fourth_term = 3 * p * q

/-- The 2010th term of the geometric sequence is 9 -/
theorem geometric_sequence_2010th_term (seq : GeometricSequence) :
  let r := seq.second_term / seq.first_term
  seq.first_term * r^(2009 : ℕ) = 9 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_2010th_term_l2038_203849


namespace NUMINAMATH_CALUDE_first_train_speed_l2038_203848

/-- Given two trains with a speed ratio of 7:8, where the second train travels 400 km in 4 hours,
    prove that the speed of the first train is 87.5 km/h. -/
theorem first_train_speed
  (speed_ratio : ℚ) -- Ratio of speeds between the two trains
  (distance : ℝ) -- Distance traveled by the second train
  (time : ℝ) -- Time taken by the second train
  (h1 : speed_ratio = 7 / 8) -- The ratio of speeds is 7:8
  (h2 : distance = 400) -- The second train travels 400 km
  (h3 : time = 4) -- The second train takes 4 hours
  : ∃ (speed1 : ℝ), speed1 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_first_train_speed_l2038_203848


namespace NUMINAMATH_CALUDE_vertex_locus_is_partial_parabola_l2038_203816

/-- The locus of points (x_t, y_t) where x_t = -t / (t^2 + 1) and y_t = c - t^2 / (t^2 + 1),
    as t ranges over all real numbers, forms part, but not all, of a parabola. -/
theorem vertex_locus_is_partial_parabola (c : ℝ) (h : c > 0) :
  ∃ (a b d : ℝ), ∀ (t : ℝ),
    ∃ (x y : ℝ), x = -t / (t^2 + 1) ∧ y = c - t^2 / (t^2 + 1) ∧
    (y = a * x^2 + b * x + d ∨ y < a * x^2 + b * x + d) :=
sorry

end NUMINAMATH_CALUDE_vertex_locus_is_partial_parabola_l2038_203816


namespace NUMINAMATH_CALUDE_painted_rectangle_ratio_l2038_203859

/-- Given a rectangle with length 2s and width s, and a paint brush of width w,
    if half the area of the rectangle is painted when the brush is swept along both diagonals,
    then the ratio of the length of the rectangle to the brush width is 6. -/
theorem painted_rectangle_ratio (s w : ℝ) (h_pos_s : 0 < s) (h_pos_w : 0 < w) :
  w^2 + 2*(s-w)^2 = s^2 → (2*s) / w = 6 := by sorry

end NUMINAMATH_CALUDE_painted_rectangle_ratio_l2038_203859


namespace NUMINAMATH_CALUDE_huangshan_temperature_difference_l2038_203808

def temperature_difference (lowest highest : ℤ) : ℤ :=
  highest - lowest

theorem huangshan_temperature_difference :
  let lowest : ℤ := -13
  let highest : ℤ := 11
  temperature_difference lowest highest = 24 := by
  sorry

end NUMINAMATH_CALUDE_huangshan_temperature_difference_l2038_203808


namespace NUMINAMATH_CALUDE_expression_evaluation_l2038_203865

theorem expression_evaluation :
  let x : ℚ := 6
  let y : ℚ := -1/6
  let expr := 7 * x^2 * y - (3*x*y - 2*(x*y - 7/2*x^2*y + 1) + 1/2*x*y)
  expr = 7/2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2038_203865


namespace NUMINAMATH_CALUDE_expression_simplification_l2038_203819

theorem expression_simplification (x y : ℝ) (hx : x = -1) (hy : y = 3) :
  5 * (3 * x^2 * y - 2 * x * y^2) - 2 * (3 * x^2 * y - 5 * x * y^2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2038_203819


namespace NUMINAMATH_CALUDE_equation_equivalence_l2038_203868

theorem equation_equivalence (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 4) (hy1 : y ≠ 0) (hy2 : y ≠ 6) :
  (2 / x + 3 / y = 1 / 2) ↔ (4 * y / (y - 6) = x) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2038_203868


namespace NUMINAMATH_CALUDE_pentadecagon_diagonals_l2038_203878

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentadecagon is a polygon with 15 sides -/
def pentadecagon_sides : ℕ := 15

theorem pentadecagon_diagonals :
  num_diagonals pentadecagon_sides = 90 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_diagonals_l2038_203878


namespace NUMINAMATH_CALUDE_equation_solution_l2038_203800

theorem equation_solution (x : ℝ) : 
  (x / 3) / 5 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2038_203800


namespace NUMINAMATH_CALUDE_sin_cos_product_zero_l2038_203850

theorem sin_cos_product_zero (θ : Real) (h : Real.sin θ + Real.cos θ = -1) : 
  Real.sin θ * Real.cos θ = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_zero_l2038_203850


namespace NUMINAMATH_CALUDE_simplify_expression_l2038_203836

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12 + 15*x + 18 = 33*x + 30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2038_203836


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_pow_215_l2038_203843

theorem last_three_digits_of_7_pow_215 : 7^215 ≡ 447 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_pow_215_l2038_203843
