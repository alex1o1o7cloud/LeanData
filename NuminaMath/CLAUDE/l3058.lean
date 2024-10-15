import Mathlib

namespace NUMINAMATH_CALUDE_max_profit_at_50_l3058_305813

/-- Profit function given the price increase x -/
def profit (x : ℕ) : ℤ := -5 * x^2 + 500 * x + 20000

/-- The maximum allowed price increase -/
def max_increase : ℕ := 200

/-- Theorem stating the maximum profit and the price increase that achieves it -/
theorem max_profit_at_50 :
  ∃ (x : ℕ), x ≤ max_increase ∧ 
  profit x = 32500 ∧ 
  ∀ (y : ℕ), y ≤ max_increase → profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_50_l3058_305813


namespace NUMINAMATH_CALUDE_mason_car_nuts_l3058_305853

/-- The number of nuts in Mason's car after squirrels stockpile for a given number of days -/
def nutsInCar (busySquirrels sleepySquirrels : ℕ) (busyNutsPerDay sleepyNutsPerDay days : ℕ) : ℕ :=
  (busySquirrels * busyNutsPerDay + sleepySquirrels * sleepyNutsPerDay) * days

/-- Theorem stating the number of nuts in Mason's car given the problem conditions -/
theorem mason_car_nuts :
  nutsInCar 2 1 30 20 40 = 3200 := by
  sorry

#eval nutsInCar 2 1 30 20 40

end NUMINAMATH_CALUDE_mason_car_nuts_l3058_305853


namespace NUMINAMATH_CALUDE_line_separate_from_circle_l3058_305890

/-- Given a point (a, b) within the unit circle, prove that the line ax + by = 1 is separate from the circle -/
theorem line_separate_from_circle (a b : ℝ) (h : a^2 + b^2 < 1) :
  ∀ x y : ℝ, x^2 + y^2 = 1 → a*x + b*y ≠ 1 := by sorry

end NUMINAMATH_CALUDE_line_separate_from_circle_l3058_305890


namespace NUMINAMATH_CALUDE_least_number_with_remainder_5_l3058_305836

/-- The least number that leaves a remainder of 5 when divided by 8, 12, 15, and 20 -/
def leastNumber : ℕ := 125

/-- Checks if a number leaves a remainder of 5 when divided by the given divisor -/
def hasRemainder5 (n : ℕ) (divisor : ℕ) : Prop :=
  n % divisor = 5

theorem least_number_with_remainder_5 :
  (∀ divisor ∈ [8, 12, 15, 20], hasRemainder5 leastNumber divisor) ∧
  (∀ m < leastNumber, ∃ divisor ∈ [8, 12, 15, 20], ¬hasRemainder5 m divisor) :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_5_l3058_305836


namespace NUMINAMATH_CALUDE_equal_sum_sequence_2011_sum_l3058_305877

/-- Definition of an equal sum sequence -/
def IsEqualSumSequence (a : ℕ → ℤ) (sum : ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = sum

/-- Definition of the sum of the first n terms of a sequence -/
def SequenceSum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum a

theorem equal_sum_sequence_2011_sum
  (a : ℕ → ℤ)
  (h_equal_sum : IsEqualSumSequence a 1)
  (h_first_term : a 1 = -1) :
  SequenceSum a 2011 = 1004 := by
sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_2011_sum_l3058_305877


namespace NUMINAMATH_CALUDE_prime_sum_equation_l3058_305821

theorem prime_sum_equation (p q s : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime s →
  p + q = s + 4 →
  1 < p →
  p < q →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_equation_l3058_305821


namespace NUMINAMATH_CALUDE_total_angle_extrema_l3058_305830

/-- A sequence of k positive real numbers -/
def PositiveSequence (k : ℕ) := { seq : Fin k → ℝ // ∀ i, seq i > 0 }

/-- The total angle of rotation for a given sequence of segment lengths -/
noncomputable def TotalAngle (k : ℕ) (seq : Fin k → ℝ) : ℝ := sorry

/-- A permutation of indices -/
def Permutation (k : ℕ) := { perm : Fin k → Fin k // Function.Bijective perm }

theorem total_angle_extrema (k : ℕ) (a : PositiveSequence k) :
  ∃ (max_perm min_perm : Permutation k),
    (∀ i j : Fin k, i ≤ j → (max_perm.val i).val ≤ (max_perm.val j).val) ∧
    (∀ i j : Fin k, i ≤ j → (min_perm.val i).val ≥ (min_perm.val j).val) ∧
    (∀ p : Permutation k,
      TotalAngle k (a.val ∘ p.val) ≤ TotalAngle k (a.val ∘ max_perm.val) ∧
      TotalAngle k (a.val ∘ p.val) ≥ TotalAngle k (a.val ∘ min_perm.val)) :=
sorry

end NUMINAMATH_CALUDE_total_angle_extrema_l3058_305830


namespace NUMINAMATH_CALUDE_dans_minimum_spending_l3058_305841

/-- Given Dan's purchases and spending information, prove he spent at least $9 -/
theorem dans_minimum_spending (chocolate_cost candy_cost difference : ℕ) 
  (h1 : chocolate_cost = 7)
  (h2 : candy_cost = 2)
  (h3 : chocolate_cost = candy_cost + difference)
  (h4 : difference = 5) : 
  chocolate_cost + candy_cost ≥ 9 := by
  sorry

#check dans_minimum_spending

end NUMINAMATH_CALUDE_dans_minimum_spending_l3058_305841


namespace NUMINAMATH_CALUDE_last_number_proof_l3058_305887

theorem last_number_proof (A B C D : ℝ) 
  (h1 : (A + B + C) / 3 = 6)
  (h2 : (B + C + D) / 3 = 3)
  (h3 : A + D = 13) :
  D = 2 := by
sorry

end NUMINAMATH_CALUDE_last_number_proof_l3058_305887


namespace NUMINAMATH_CALUDE_collinear_points_imply_a_value_l3058_305897

/-- Given three points A, B, and C in the plane, 
    this function returns true if they are collinear. -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  (C.2 - A.2) * (B.1 - A.1) = (B.2 - A.2) * (C.1 - A.1)

theorem collinear_points_imply_a_value : 
  ∀ a : ℝ, collinear (3, 2) (-2, a) (8, 12) → a = -8 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_imply_a_value_l3058_305897


namespace NUMINAMATH_CALUDE_sequence_a_closed_form_l3058_305834

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 5
  | (n + 2) => (2 * (sequence_a (n + 1))^2 - 3 * sequence_a (n + 1) - 9) / (2 * sequence_a n)

theorem sequence_a_closed_form (n : ℕ) : sequence_a n = 2^(n + 2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_closed_form_l3058_305834


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3058_305895

def set_A : Set ℤ := {x | |x| < 3}
def set_B : Set ℤ := {x | |x| > 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3058_305895


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3058_305891

/-- Represents a stratified sampling scenario by gender -/
structure StratifiedSample where
  total_population : ℕ
  male_population : ℕ
  male_sample : ℕ
  total_sample : ℕ

/-- Theorem stating that given the conditions, the total sample size is 36 -/
theorem stratified_sample_size
  (s : StratifiedSample)
  (h1 : s.total_population = 120)
  (h2 : s.male_population = 80)
  (h3 : s.male_sample = 24)
  (h4 : s.male_sample / s.total_sample = s.male_population / s.total_population) :
  s.total_sample = 36 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l3058_305891


namespace NUMINAMATH_CALUDE_right_angle_vector_proof_l3058_305823

/-- Given two vectors OA and OB in a 2D Cartesian coordinate system, 
    where OA forms a right angle with AB, prove that the y-coordinate of OA is 5. -/
theorem right_angle_vector_proof (t : ℝ) : 
  let OA : ℝ × ℝ := (-1, t)
  let OB : ℝ × ℝ := (2, 2)
  let AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)
  (AB.1 * OB.1 + AB.2 * OB.2 = 0) → t = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_angle_vector_proof_l3058_305823


namespace NUMINAMATH_CALUDE_mean_variance_relationship_l3058_305874

-- Define the sample size
def sample_size : Nat := 50

-- Define the original mean and variance
def original_mean : Real := 70
def original_variance : Real := 75

-- Define the incorrect and correct data points
def incorrect_point1 : Real := 60
def incorrect_point2 : Real := 90
def correct_point1 : Real := 80
def correct_point2 : Real := 70

-- Define the new mean and variance after correction
def new_mean : Real := original_mean
noncomputable def new_variance : Real := original_variance - 8

-- Theorem statement
theorem mean_variance_relationship :
  new_mean = original_mean ∧ new_variance < original_variance :=
by sorry

end NUMINAMATH_CALUDE_mean_variance_relationship_l3058_305874


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3058_305849

theorem perfect_square_condition (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - (m+1)*x + 1 = k^2) → (m = 1 ∨ m = -3) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3058_305849


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3058_305835

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) * a m = a n * a (m + 1)

/-- The problem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_a5 : a 5 = 2) 
    (h_a7 : a 7 = 8) : 
    a 6 = 4 ∨ a 6 = -4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l3058_305835


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3058_305845

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 4 + a 8 = 16) :
  a 2 + a 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3058_305845


namespace NUMINAMATH_CALUDE_unique_divisor_property_l3058_305884

theorem unique_divisor_property (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_5 : p > 5) :
  ∃! x : ℕ, x ≠ 0 ∧ ∀ n : ℕ, n > 0 → (5 * p + x) ∣ (5 * p^n + x^n) ∧ x = p := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_property_l3058_305884


namespace NUMINAMATH_CALUDE_bus_occupancy_problem_l3058_305871

/-- Given an initial number of people on a bus, and the number of people who get on and off,
    calculate the final number of people on the bus. -/
def final_bus_occupancy (initial : ℕ) (got_on : ℕ) (got_off : ℕ) : ℕ :=
  initial + got_on - got_off

/-- Theorem stating that with 32 people initially on the bus, 19 getting on, and 13 getting off,
    the final number of people on the bus is 38. -/
theorem bus_occupancy_problem :
  final_bus_occupancy 32 19 13 = 38 := by
  sorry

end NUMINAMATH_CALUDE_bus_occupancy_problem_l3058_305871


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_two_l3058_305824

theorem logarithm_expression_equals_two :
  (Real.log 243 / Real.log 3) / (Real.log 3 / Real.log 81) -
  (Real.log 729 / Real.log 3) / (Real.log 3 / Real.log 27) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_two_l3058_305824


namespace NUMINAMATH_CALUDE_clara_age_problem_l3058_305806

theorem clara_age_problem : ∃! x : ℕ+, 
  (∃ n : ℕ+, (x - 2 : ℤ) = n^2) ∧ 
  (∃ m : ℕ+, (x + 3 : ℤ) = m^3) ∧ 
  x = 123 := by
  sorry

end NUMINAMATH_CALUDE_clara_age_problem_l3058_305806


namespace NUMINAMATH_CALUDE_coffee_shop_total_sales_l3058_305805

/-- Calculates the total money made by a coffee shop given the number of coffee and tea orders and their respective prices. -/
def coffee_shop_sales (coffee_orders : ℕ) (coffee_price : ℕ) (tea_orders : ℕ) (tea_price : ℕ) : ℕ :=
  coffee_orders * coffee_price + tea_orders * tea_price

/-- Theorem stating that the coffee shop made $67 given the specified orders and prices. -/
theorem coffee_shop_total_sales :
  coffee_shop_sales 7 5 8 4 = 67 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_total_sales_l3058_305805


namespace NUMINAMATH_CALUDE_sunset_duration_l3058_305886

/-- Proves that a sunset with 12 color changes occurring every 10 minutes lasts 2 hours. -/
theorem sunset_duration (color_change_interval : ℕ) (total_changes : ℕ) (minutes_per_hour : ℕ) :
  color_change_interval = 10 →
  total_changes = 12 →
  minutes_per_hour = 60 →
  (color_change_interval * total_changes) / minutes_per_hour = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_sunset_duration_l3058_305886


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3058_305838

theorem triangle_abc_properties (a b c A B C : ℝ) : 
  a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = (1/2) * b →
  a > b →
  b = Real.sqrt 13 →
  a + c = 4 →
  B = π / 6 ∧ 
  (1/2) * a * c * Real.sin B = (6 - 3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3058_305838


namespace NUMINAMATH_CALUDE_intersection_parallel_line_l3058_305822

/-- The equation of a line passing through the intersection of two lines and parallel to a third line -/
theorem intersection_parallel_line (x y : ℝ) : 
  (y = 2*x + 1) → 
  (y = 3*x - 1) → 
  (∃ m : ℝ, 2*x + y - m = 0 ∧ (∀ x y : ℝ, 2*x + y - m = 0 ↔ 2*x + y - 3 = 0)) →
  (2*x + y - 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_l3058_305822


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3058_305857

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 5*x + 4) * (x^2 + 3*x + 2) + (x^2 + 4*x - 3) = 
  (x^2 + 4*x + 2) * (x^2 + 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3058_305857


namespace NUMINAMATH_CALUDE_license_plate_increase_l3058_305846

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  new_plates / old_plates = 26^2 / 10 := by
sorry

end NUMINAMATH_CALUDE_license_plate_increase_l3058_305846


namespace NUMINAMATH_CALUDE_curve_properties_l3058_305840

/-- The curve equation -/
def curve (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0

theorem curve_properties :
  (∀ x y : ℝ, curve x y 1 ↔ x = 1 ∧ y = 1) ∧
  (∀ a : ℝ, a ≠ 1 → curve 1 1 a) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l3058_305840


namespace NUMINAMATH_CALUDE_triangle_shape_l3058_305863

theorem triangle_shape (a b : ℝ) (A B : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) 
  (h : a * Real.cos A = b * Real.cos B) :
  A = B ∨ A + B = π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_shape_l3058_305863


namespace NUMINAMATH_CALUDE_total_pears_l3058_305837

-- Define the number of pears sold
def sold : ℕ := 20

-- Define the number of pears poached in terms of sold
def poached : ℕ := sold / 2

-- Define the number of pears canned in terms of poached
def canned : ℕ := poached + poached / 5

-- Theorem statement
theorem total_pears : sold + poached + canned = 42 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_l3058_305837


namespace NUMINAMATH_CALUDE_cubic_factorization_l3058_305858

theorem cubic_factorization (a : ℝ) : a^3 + 2*a^2 + a = a*(a+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3058_305858


namespace NUMINAMATH_CALUDE_total_weight_is_675_l3058_305812

/-- The total weight Tom is moving with, given his weight, the weight he holds in each hand, and the weight of his vest. -/
def total_weight_moved (tom_weight : ℝ) (hand_weight_multiplier : ℝ) (vest_weight_multiplier : ℝ) : ℝ :=
  tom_weight + (vest_weight_multiplier * tom_weight) + (2 * hand_weight_multiplier * tom_weight)

/-- Theorem stating that the total weight Tom is moving with is 675 kg -/
theorem total_weight_is_675 :
  total_weight_moved 150 1.5 0.5 = 675 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_675_l3058_305812


namespace NUMINAMATH_CALUDE_centroid_coordinates_specific_triangle_centroid_l3058_305804

/-- The centroid of a triangle is located at the arithmetic mean of its vertices. -/
theorem centroid_coordinates (A B C : ℝ × ℝ) :
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  A = (-1, 3) → B = (1, 2) → C = (2, -5) → G = (2/3, 0) := by
  sorry

/-- The centroid of the specific triangle ABC is at (2/3, 0). -/
theorem specific_triangle_centroid :
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (1, 2)
  let C : ℝ × ℝ := (2, -5)
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  G = (2/3, 0) := by
  sorry

end NUMINAMATH_CALUDE_centroid_coordinates_specific_triangle_centroid_l3058_305804


namespace NUMINAMATH_CALUDE_digit_2007_in_2003_digit_number_l3058_305817

/-- The sequence of digits formed by concatenating positive integers -/
def digit_sequence : ℕ → ℕ := sorry

/-- The function G(n) that calculates the number of digits preceding 10^n in the sequence -/
def G (n : ℕ) : ℕ := sorry

/-- The function f(n) that returns the number of digits in the number where the 10^n-th digit occurs -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 10^2007-th digit occurs in a 2003-digit number -/
theorem digit_2007_in_2003_digit_number : f 2007 = 2003 := by sorry

end NUMINAMATH_CALUDE_digit_2007_in_2003_digit_number_l3058_305817


namespace NUMINAMATH_CALUDE_glass_volume_l3058_305869

/-- Given a bottle and a glass, proves that the volume of the glass is 0.5 L 
    when water is poured from a full 1.5 L bottle into an empty glass 
    until both are 3/4 full. -/
theorem glass_volume (bottle_initial : ℝ) (glass : ℝ) : 
  bottle_initial = 1.5 →
  (3/4) * bottle_initial + (3/4) * glass = bottle_initial →
  glass = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_glass_volume_l3058_305869


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3058_305855

/-- A line passing through point (1, 1) with equal intercepts on both coordinate axes -/
def equal_intercept_line (x y : ℝ) : Prop :=
  (x = 1 ∧ y = 1) ∨ 
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a * x + b * y = a * b ∧ a = b)

/-- The equation of the line is x - y = 0 or x + y - 2 = 0 -/
theorem equal_intercept_line_equation (x y : ℝ) :
  equal_intercept_line x y ↔ (x - y = 0 ∨ x + y - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3058_305855


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l3058_305826

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 18 * x + c = 0) →  -- exactly one solution
  (a + c = 26) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 13 + 2 * Real.sqrt 22 ∧ c = 13 - 2 * Real.sqrt 22) := by
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l3058_305826


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3058_305803

theorem sqrt_fraction_simplification :
  Real.sqrt (25 / 36 + 16 / 9) = Real.sqrt 89 / 6 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3058_305803


namespace NUMINAMATH_CALUDE_geometric_sequence_a1_l3058_305881

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_a1 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  a 3 = 1 →
  a 5 = 1/2 →
  a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a1_l3058_305881


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l3058_305832

theorem product_remainder_mod_five : (1234 * 1987 * 2013 * 2021) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l3058_305832


namespace NUMINAMATH_CALUDE_highest_seat_number_is_44_l3058_305811

/-- Calculates the highest seat number in a systematic sample -/
def highest_seat_number (total_students : ℕ) (sample_size : ℕ) (first_student : ℕ) : ℕ :=
  let interval := total_students / sample_size
  first_student + (sample_size - 1) * interval

/-- Theorem: The highest seat number in the sample is 44 -/
theorem highest_seat_number_is_44 :
  highest_seat_number 56 4 2 = 44 := by
  sorry

#eval highest_seat_number 56 4 2

end NUMINAMATH_CALUDE_highest_seat_number_is_44_l3058_305811


namespace NUMINAMATH_CALUDE_jisoo_drank_least_l3058_305885

-- Define the amount of juice each person drank
def jennie_juice : ℚ := 9/5

-- Define Jisoo's juice amount in terms of Jennie's
def jisoo_juice : ℚ := jennie_juice - 1/5

-- Define Rohee's juice amount in terms of Jisoo's
def rohee_juice : ℚ := jisoo_juice + 3/10

-- Theorem statement
theorem jisoo_drank_least : 
  jisoo_juice < jennie_juice ∧ jisoo_juice < rohee_juice := by
  sorry


end NUMINAMATH_CALUDE_jisoo_drank_least_l3058_305885


namespace NUMINAMATH_CALUDE_horror_movie_tickets_l3058_305870

theorem horror_movie_tickets (romance_tickets : ℕ) (horror_tickets : ℕ) : 
  romance_tickets = 25 →
  horror_tickets = 3 * romance_tickets + 18 →
  horror_tickets = 93 := by
sorry

end NUMINAMATH_CALUDE_horror_movie_tickets_l3058_305870


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3058_305818

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the circle equation
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- State the theorem
theorem circle_equation_proof :
  ∃ (c : Circle),
    (∃ (x : ℝ), line1 x 0 ∧ c.center = (x, 0)) ∧
    (∀ (x y : ℝ), line2 x y → ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2)) →
    ∀ (x y : ℝ), circleEquation c x y ↔ (x + 1)^2 + y^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3058_305818


namespace NUMINAMATH_CALUDE_min_omega_l3058_305816

theorem min_omega (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = 2 * Real.sin (ω * x)) →
  ω > 0 →
  (∀ x ∈ Set.Icc (-π/3) (π/4), f x ≥ -2) →
  (∃ x ∈ Set.Icc (-π/3) (π/4), f x = -2) →
  ω ≥ 3/2 ∧ ∀ ω' ≥ 3/2, ∃ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω' * x) = -2 :=
by sorry

end NUMINAMATH_CALUDE_min_omega_l3058_305816


namespace NUMINAMATH_CALUDE_internship_arrangements_l3058_305879

/-- The number of ways to arrange 4 distinct objects into 2 indistinguishable pairs 
    and then assign these pairs to 2 distinct locations -/
theorem internship_arrangements (n : Nat) (m : Nat) : n = 4 ∧ m = 2 → 
  (Nat.choose n 2 / 2) * m.factorial = 6 := by
  sorry

end NUMINAMATH_CALUDE_internship_arrangements_l3058_305879


namespace NUMINAMATH_CALUDE_remainder_theorem_l3058_305899

theorem remainder_theorem (x : ℤ) : 
  (x^15 + 3) % (x + 2) = (-2)^15 + 3 :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3058_305899


namespace NUMINAMATH_CALUDE_distance_AB_is_7_l3058_305819

/-- Represents the distance between two points A and B, given the conditions of the pedestrian problem. -/
def distance_AB : ℝ :=
  let v1 : ℝ := 4  -- Speed of the first pedestrian in km/hr
  let v2 : ℝ := 3  -- Speed of the second pedestrian in km/hr
  let t_meet : ℝ := 1.5  -- Time until meeting in hours
  let d1_before : ℝ := v1 * t_meet  -- Distance covered by first pedestrian before meeting
  let d2_before : ℝ := v2 * t_meet  -- Distance covered by second pedestrian before meeting
  let d1_after : ℝ := v1 * 0.75  -- Distance covered by first pedestrian after meeting
  let d2_after : ℝ := v2 * (4/3)  -- Distance covered by second pedestrian after meeting
  d1_before + d2_before  -- Total distance

/-- Theorem stating that the distance between points A and B is 7 km, given the conditions of the pedestrian problem. -/
theorem distance_AB_is_7 : distance_AB = 7 := by
  sorry  -- Proof is omitted as per instructions

#eval distance_AB  -- This will evaluate to 7

end NUMINAMATH_CALUDE_distance_AB_is_7_l3058_305819


namespace NUMINAMATH_CALUDE_acute_angles_insufficient_for_congruence_l3058_305872

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- leg
  b : ℝ  -- leg
  c : ℝ  -- hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Define congruence for right-angled triangles
def congruent (t1 t2 : RightTriangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define equality of acute angles
def equal_acute_angles (t1 t2 : RightTriangle) : Prop :=
  Real.arctan (t1.a / t1.b) = Real.arctan (t2.a / t2.b) ∧
  Real.arctan (t1.b / t1.a) = Real.arctan (t2.b / t2.a)

-- Theorem statement
theorem acute_angles_insufficient_for_congruence :
  ∃ (t1 t2 : RightTriangle), equal_acute_angles t1 t2 ∧ ¬congruent t1 t2 :=
sorry

end NUMINAMATH_CALUDE_acute_angles_insufficient_for_congruence_l3058_305872


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l3058_305854

theorem divisibility_of_sum_of_squares (p a b : ℕ) : 
  Prime p → 
  (∃ n : ℕ, p = 4 * n + 3) → 
  p ∣ (a^2 + b^2) → 
  (p ∣ a ∧ p ∣ b) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l3058_305854


namespace NUMINAMATH_CALUDE_tan_double_angle_l3058_305808

theorem tan_double_angle (α : ℝ) (h : Real.sin α - 2 * Real.cos α = 0) : 
  Real.tan (2 * α) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l3058_305808


namespace NUMINAMATH_CALUDE_faster_walking_speed_l3058_305839

theorem faster_walking_speed 
  (actual_distance : ℝ) 
  (original_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_distance = 60) 
  (h2 : original_speed = 12) 
  (h3 : additional_distance = 20) : 
  let time := actual_distance / original_speed
  let total_distance := actual_distance + additional_distance
  let faster_speed := total_distance / time
  faster_speed = 16 := by sorry

end NUMINAMATH_CALUDE_faster_walking_speed_l3058_305839


namespace NUMINAMATH_CALUDE_unique_solution_2m_minus_1_eq_3n_l3058_305878

theorem unique_solution_2m_minus_1_eq_3n :
  ∀ m n : ℕ+, 2^(m : ℕ) - 1 = 3^(n : ℕ) ↔ m = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_2m_minus_1_eq_3n_l3058_305878


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l3058_305880

theorem missing_fraction_sum (x : ℚ) : x = -11/60 →
  1/3 + 1/2 + (-5/6) + 1/5 + 1/4 + (-2/15) + x = 0.13333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l3058_305880


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_to_fifth_l3058_305802

theorem imaginary_part_of_one_plus_i_to_fifth (i : ℂ) : i * i = -1 → Complex.im ((1 + i)^5) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_to_fifth_l3058_305802


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3058_305894

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c > 0 ∧
    c / a = Real.sqrt 6 / 2 ∧
    b * c / Real.sqrt (a^2 + b^2) = 1 ∧
    c^2 = a^2 + b^2) →
  a^2 = 2 ∧ b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3058_305894


namespace NUMINAMATH_CALUDE_fourth_pillar_height_17_l3058_305892

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space using the general form Ax + By + Cz = D -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the height of the fourth pillar in a square arrangement -/
def calculateFourthPillarHeight (a b c : ℝ) : ℝ :=
  sorry

theorem fourth_pillar_height_17 :
  calculateFourthPillarHeight 15 10 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_fourth_pillar_height_17_l3058_305892


namespace NUMINAMATH_CALUDE_frog_game_result_l3058_305842

def frog_A_jump : ℕ := 10
def frog_B_jump : ℕ := 15
def trap_interval : ℕ := 12

def first_trap (jump_distance : ℕ) : ℕ :=
  (trap_interval / jump_distance) * jump_distance

theorem frog_game_result :
  let first_frog_trap := min (first_trap frog_A_jump) (first_trap frog_B_jump)
  let other_frog_distance := if first_frog_trap = first_trap frog_B_jump
                             then (first_frog_trap / frog_B_jump) * frog_A_jump
                             else (first_frog_trap / frog_A_jump) * frog_B_jump
  (trap_interval - (other_frog_distance % trap_interval)) % trap_interval = 8 := by
  sorry

end NUMINAMATH_CALUDE_frog_game_result_l3058_305842


namespace NUMINAMATH_CALUDE_max_value_of_g_l3058_305815

/-- The function g(x) = 4x - x^4 -/
def g (x : ℝ) : ℝ := 4*x - x^4

/-- The theorem stating that the maximum value of g(x) on [0, 2] is 3 -/
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3058_305815


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l3058_305851

theorem halfway_between_fractions : 
  let a := (1 : ℚ) / 8
  let b := (1 : ℚ) / 3
  (a + b) / 2 = 11 / 48 := by sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l3058_305851


namespace NUMINAMATH_CALUDE_g_of_three_eq_seventeen_sixths_l3058_305828

/-- Given a function g satisfying the equation for all x ≠ 1/2, prove that g(3) = 17/6 -/
theorem g_of_three_eq_seventeen_sixths 
  (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 1/2 → g x + g ((x + 2) / (2 - 4*x)) = 2*x) : 
  g 3 = 17/6 := by
sorry

end NUMINAMATH_CALUDE_g_of_three_eq_seventeen_sixths_l3058_305828


namespace NUMINAMATH_CALUDE_max_sum_square_roots_l3058_305868

/-- Given a positive real number k, the function f(x) = x + √(k - x^2) 
    reaches its maximum value of √(2k) when x = √(k/2) -/
theorem max_sum_square_roots (k : ℝ) (h : k > 0) :
  ∃ (x : ℝ), x ≥ 0 ∧ x^2 ≤ k ∧
    (∀ (y : ℝ), y ≥ 0 → y^2 ≤ k → x + Real.sqrt (k - x^2) ≥ y + Real.sqrt (k - y^2)) ∧
    x + Real.sqrt (k - x^2) = Real.sqrt (2 * k) ∧
    x = Real.sqrt (k / 2) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_square_roots_l3058_305868


namespace NUMINAMATH_CALUDE_product_125_sum_31_l3058_305856

theorem product_125_sum_31 :
  ∃ (a b c : ℕ+), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a * b * c : ℕ) = 125 →
    (a + b + c : ℕ) = 31 := by
sorry

end NUMINAMATH_CALUDE_product_125_sum_31_l3058_305856


namespace NUMINAMATH_CALUDE_prob_no_red_3x3_is_170_171_l3058_305847

/-- Represents a 4x4 grid where each cell can be either red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a 3x3 subgrid starting at (i, j) is all red -/
def is_red_3x3 (g : Grid) (i j : Fin 2) : Prop :=
  ∀ x y, g (i + x) (j + y) = true

/-- The probability of a random 4x4 grid not containing a 3x3 red square -/
def prob_no_red_3x3 : ℚ :=
  170 / 171

/-- The main theorem stating the probability of a 4x4 grid not containing a 3x3 red square -/
theorem prob_no_red_3x3_is_170_171 :
  prob_no_red_3x3 = 170 / 171 := by sorry

end NUMINAMATH_CALUDE_prob_no_red_3x3_is_170_171_l3058_305847


namespace NUMINAMATH_CALUDE_shooting_probability_l3058_305898

theorem shooting_probability (accuracy : ℝ) (two_shots : ℝ) :
  accuracy = 9/10 →
  two_shots = 1/2 →
  (two_shots / accuracy) = 5/9 :=
by sorry

end NUMINAMATH_CALUDE_shooting_probability_l3058_305898


namespace NUMINAMATH_CALUDE_louisa_average_speed_l3058_305859

/-- Proves that given the travel conditions, Louisa's average speed was 40 miles per hour -/
theorem louisa_average_speed :
  ∀ (v : ℝ),
  v > 0 →
  (280 / v) = (160 / v) + 3 →
  v = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_louisa_average_speed_l3058_305859


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3058_305850

def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3058_305850


namespace NUMINAMATH_CALUDE_sqrt_nine_equals_three_l3058_305866

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_nine_equals_three_l3058_305866


namespace NUMINAMATH_CALUDE_power_relation_l3058_305864

theorem power_relation (x a : ℝ) (h : x^(-a) = 3) : x^(2*a) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l3058_305864


namespace NUMINAMATH_CALUDE_method_of_continued_proportion_is_correct_l3058_305831

-- Define the possible methods
inductive AncientChineseMathMethod
| CircleCutting
| ContinuedProportion
| SuJiushaoAlgorithm
| SunTzuRemainder

-- Define a property for methods that can find GCD
def canFindGCD (method : AncientChineseMathMethod) : Prop := sorry

-- Define a property for methods from Song and Yuan dynasties
def fromSongYuanDynasties (method : AncientChineseMathMethod) : Prop := sorry

-- Define a property for methods comparable to Euclidean algorithm
def comparableToEuclidean (method : AncientChineseMathMethod) : Prop := sorry

-- Theorem stating that the Method of Continued Proportion is the correct answer
theorem method_of_continued_proportion_is_correct :
  ∃ (method : AncientChineseMathMethod),
    method = AncientChineseMathMethod.ContinuedProportion ∧
    canFindGCD method ∧
    fromSongYuanDynasties method ∧
    comparableToEuclidean method ∧
    (∀ (other : AncientChineseMathMethod),
      other ≠ AncientChineseMathMethod.ContinuedProportion →
      ¬(canFindGCD other ∧ fromSongYuanDynasties other ∧ comparableToEuclidean other)) :=
sorry

end NUMINAMATH_CALUDE_method_of_continued_proportion_is_correct_l3058_305831


namespace NUMINAMATH_CALUDE_triangle_area_l3058_305825

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define the length of a side
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two sides
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def area (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area (A B C : ℝ × ℝ) :
  Triangle A B C →
  length A B = 6 →
  angle B A C = 30 * π / 180 →
  angle A B C = 120 * π / 180 →
  area A B C = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3058_305825


namespace NUMINAMATH_CALUDE_crayons_per_row_l3058_305801

/-- Given that Faye has 16 rows of crayons and pencils, with a total of 96 crayons,
    prove that there are 6 crayons in each row. -/
theorem crayons_per_row (total_rows : ℕ) (total_crayons : ℕ) (h1 : total_rows = 16) (h2 : total_crayons = 96) :
  total_crayons / total_rows = 6 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_row_l3058_305801


namespace NUMINAMATH_CALUDE_necklace_stand_capacity_l3058_305883

-- Define the given constants
def current_necklaces : ℕ := 5
def ring_capacity : ℕ := 30
def current_rings : ℕ := 18
def bracelet_capacity : ℕ := 15
def current_bracelets : ℕ := 8
def necklace_price : ℕ := 4
def ring_price : ℕ := 10
def bracelet_price : ℕ := 5
def total_cost : ℕ := 183

-- Theorem to prove
theorem necklace_stand_capacity : ∃ (total_necklaces : ℕ), 
  total_necklaces = current_necklaces + 
    ((total_cost - (ring_price * (ring_capacity - current_rings) + 
                    bracelet_price * (bracelet_capacity - current_bracelets))) / necklace_price) :=
by sorry

end NUMINAMATH_CALUDE_necklace_stand_capacity_l3058_305883


namespace NUMINAMATH_CALUDE_point_ordering_on_reciprocal_function_l3058_305829

/-- Given points on the graph of y = k/x where k > 0, prove a < c < b -/
theorem point_ordering_on_reciprocal_function (k a b c : ℝ) : 
  k > 0 → 
  a * (-2) = k → 
  b * 2 = k → 
  c * 3 = k → 
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_point_ordering_on_reciprocal_function_l3058_305829


namespace NUMINAMATH_CALUDE_collatz_eighth_term_one_l3058_305888

def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def collatzSequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => collatz (collatzSequence n k)

def validStartingNumbers : Set ℕ :=
  {n | n > 0 ∧ collatzSequence n 7 = 1}

theorem collatz_eighth_term_one :
  validStartingNumbers = {2, 3, 16, 20, 21, 128} :=
sorry

end NUMINAMATH_CALUDE_collatz_eighth_term_one_l3058_305888


namespace NUMINAMATH_CALUDE_sin_330_degrees_l3058_305848

theorem sin_330_degrees : Real.sin (330 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l3058_305848


namespace NUMINAMATH_CALUDE_root_polynomial_relation_l3058_305867

theorem root_polynomial_relation : ∃ (b c : ℤ), 
  (∀ r : ℝ, r^2 - 2*r - 1 = 0 → r^5 - b*r - c = 0) ∧ b*c = 348 := by
  sorry

end NUMINAMATH_CALUDE_root_polynomial_relation_l3058_305867


namespace NUMINAMATH_CALUDE_third_side_length_equal_to_altitude_l3058_305882

/-- Given an acute-angled triangle with two sides of lengths √13 and √10 cm,
    if the third side is equal to the altitude drawn to it,
    then the length of the third side is 3 cm. -/
theorem third_side_length_equal_to_altitude
  (a b c : ℝ) -- sides of the triangle
  (h : ℝ) -- altitude to side c
  (acute : 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) -- acute-angled triangle
  (side1 : a = Real.sqrt 13)
  (side2 : b = Real.sqrt 10)
  (altitude_eq_side : h = c)
  (pythagorean1 : a^2 = (c - h)^2 + h^2)
  (pythagorean2 : b^2 = h^2 + h^2) :
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_equal_to_altitude_l3058_305882


namespace NUMINAMATH_CALUDE_extraneous_root_value_l3058_305810

theorem extraneous_root_value (x m : ℝ) : 
  ((x + 7) / (x - 1) + 2 = (m + 5) / (x - 1)) ∧ 
  (x = 1) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_extraneous_root_value_l3058_305810


namespace NUMINAMATH_CALUDE_andrew_stamping_rate_l3058_305873

/-- Andrew's work schedule and permit stamping rate -/
def andrew_schedule (appointments : ℕ) (appointment_duration : ℕ) (workday_length : ℕ) (total_permits : ℕ) : ℕ :=
  let time_in_appointments := appointments * appointment_duration
  let time_stamping := workday_length - time_in_appointments
  total_permits / time_stamping

/-- Theorem stating Andrew's permit stamping rate given his schedule -/
theorem andrew_stamping_rate :
  andrew_schedule 2 3 8 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_andrew_stamping_rate_l3058_305873


namespace NUMINAMATH_CALUDE_train_length_l3058_305843

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 24 → speed * time * (1000 / 3600) = 420 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3058_305843


namespace NUMINAMATH_CALUDE_voldemort_cake_calories_l3058_305809

/-- Calculates the calories of a cake given daily calorie limit, consumed calories, and remaining allowed calories. -/
def cake_calories (daily_limit : ℕ) (breakfast : ℕ) (lunch : ℕ) (chips : ℕ) (coke : ℕ) (remaining : ℕ) : ℕ :=
  daily_limit - (breakfast + lunch + chips + coke) - remaining

/-- Proves that the cake has 110 calories given Voldemort's calorie intake information. -/
theorem voldemort_cake_calories :
  cake_calories 2500 560 780 310 215 525 = 110 := by
  sorry

#eval cake_calories 2500 560 780 310 215 525

end NUMINAMATH_CALUDE_voldemort_cake_calories_l3058_305809


namespace NUMINAMATH_CALUDE_polygon_sides_l3058_305875

/-- Theorem: A polygon with 1080° as the sum of its interior angles has 8 sides. -/
theorem polygon_sides (n : ℕ) : (180 * (n - 2) = 1080) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3058_305875


namespace NUMINAMATH_CALUDE_conference_handshakes_l3058_305844

/-- Represents a conference with a fixed number of attendees and handshakes per person -/
structure Conference where
  attendees : ℕ
  handshakes_per_person : ℕ

/-- Calculates the minimum number of unique handshakes in a conference -/
def min_handshakes (conf : Conference) : ℕ :=
  conf.attendees * conf.handshakes_per_person / 2

/-- Theorem stating that in a conference of 30 people where each person shakes hands
    with exactly 3 others, the minimum number of unique handshakes is 45 -/
theorem conference_handshakes :
  let conf : Conference := { attendees := 30, handshakes_per_person := 3 }
  min_handshakes conf = 45 := by
  sorry


end NUMINAMATH_CALUDE_conference_handshakes_l3058_305844


namespace NUMINAMATH_CALUDE_student_calculation_difference_l3058_305896

theorem student_calculation_difference : 
  let number : ℝ := 80.00000000000003
  let correct_answer := number * (4/5 : ℝ)
  let student_answer := number / (4/5 : ℝ)
  student_answer - correct_answer = 36.0000000000000175 := by
sorry

end NUMINAMATH_CALUDE_student_calculation_difference_l3058_305896


namespace NUMINAMATH_CALUDE_plastic_rings_weight_sum_l3058_305827

theorem plastic_rings_weight_sum :
  let orange_ring : Float := 0.08333333333333333
  let purple_ring : Float := 0.3333333333333333
  let white_ring : Float := 0.4166666666666667
  orange_ring + purple_ring + white_ring = 0.8333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_plastic_rings_weight_sum_l3058_305827


namespace NUMINAMATH_CALUDE_product_of_fractions_l3058_305861

theorem product_of_fractions : (2 : ℚ) / 9 * (5 : ℚ) / 4 = (5 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3058_305861


namespace NUMINAMATH_CALUDE_truck_capacity_l3058_305852

/-- The total fuel capacity of Donny's truck -/
def total_capacity : ℕ := 150

/-- The amount of fuel already in the truck -/
def initial_fuel : ℕ := 38

/-- The amount of money Donny started with -/
def initial_money : ℕ := 350

/-- The amount of change Donny received -/
def change : ℕ := 14

/-- The cost of fuel per liter -/
def cost_per_liter : ℕ := 3

/-- Theorem stating that the total capacity of Donny's truck is 150 liters -/
theorem truck_capacity : 
  total_capacity = initial_fuel + (initial_money - change) / cost_per_liter := by
  sorry

end NUMINAMATH_CALUDE_truck_capacity_l3058_305852


namespace NUMINAMATH_CALUDE_greatest_four_digit_number_l3058_305807

theorem greatest_four_digit_number (n : ℕ) : n ≤ 9999 ∧ n ≥ 1000 ∧ 
  ∃ k₁ k₂ : ℕ, n = 11 * k₁ + 2 ∧ n = 7 * k₂ + 4 → n ≤ 9973 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_number_l3058_305807


namespace NUMINAMATH_CALUDE_factor_polynomial_l3058_305800

theorem factor_polynomial (x : ℝ) : 98 * x^7 - 266 * x^13 = 14 * x^7 * (7 - 19 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3058_305800


namespace NUMINAMATH_CALUDE_triangle_altitude_segment_l3058_305876

theorem triangle_altitude_segment (a b c h x : ℝ) : 
  a = 40 → b = 90 → c = 100 → 
  a^2 = x^2 + h^2 → b^2 = (c - x)^2 + h^2 → 
  c - x = 82.5 := by sorry

end NUMINAMATH_CALUDE_triangle_altitude_segment_l3058_305876


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l3058_305893

/-- A type representing the scientists -/
def Scientist : Type := Fin 17

/-- A type representing the topics -/
inductive Topic
| A
| B
| C

/-- A function representing the correspondence between scientists on topics -/
def correspondence : Scientist → Scientist → Topic := sorry

/-- The main theorem stating that there exists a monochromatic triangle -/
theorem monochromatic_triangle_exists :
  ∃ (s1 s2 s3 : Scientist) (t : Topic),
    s1 ≠ s2 ∧ s1 ≠ s3 ∧ s2 ≠ s3 ∧
    correspondence s1 s2 = t ∧
    correspondence s1 s3 = t ∧
    correspondence s2 s3 = t :=
  sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l3058_305893


namespace NUMINAMATH_CALUDE_x_intercepts_difference_l3058_305865

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the x-intercepts
variable (x₁ x₂ x₃ x₄ : ℝ)

-- State the conditions
axiom g_def : ∀ x, g x = 2 * f (200 - x)
axiom vertex_condition : ∃ v, f v = 0 ∧ g v = 0
axiom x_intercepts_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄
axiom x_intercepts_f : f x₁ = 0 ∧ f x₄ = 0
axiom x_intercepts_g : g x₂ = 0 ∧ g x₃ = 0
axiom x_diff : x₃ - x₂ = 300

-- Theorem to prove
theorem x_intercepts_difference : x₄ - x₁ = 600 := by sorry

end NUMINAMATH_CALUDE_x_intercepts_difference_l3058_305865


namespace NUMINAMATH_CALUDE_factory_uses_systematic_sampling_l3058_305833

/-- Represents a sampling method used in quality control -/
inductive SamplingMethod
| Systematic
| Random
| Stratified
| Cluster

/-- Represents a factory's product inspection process -/
structure InspectionProcess where
  productsOnConveyor : Bool
  fixedSamplingPosition : Bool
  regularInterval : Bool

/-- Determines the sampling method based on the inspection process -/
def determineSamplingMethod (process : InspectionProcess) : SamplingMethod :=
  if process.productsOnConveyor && process.fixedSamplingPosition && process.regularInterval then
    SamplingMethod.Systematic
  else
    SamplingMethod.Random  -- Default to Random for simplicity

/-- Theorem: The given inspection process uses Systematic Sampling -/
theorem factory_uses_systematic_sampling (process : InspectionProcess) 
  (h1 : process.productsOnConveyor = true)
  (h2 : process.fixedSamplingPosition = true)
  (h3 : process.regularInterval = true) :
  determineSamplingMethod process = SamplingMethod.Systematic :=
by sorry

end NUMINAMATH_CALUDE_factory_uses_systematic_sampling_l3058_305833


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l3058_305862

/-- Prove that the difference between the average age of the whole cricket team
and the average age of the remaining players is 3 years. -/
theorem cricket_team_age_difference 
  (team_size : ℕ) 
  (team_avg_age : ℝ) 
  (wicket_keeper_age_diff : ℝ) 
  (remaining_avg_age : ℝ) 
  (h1 : team_size = 11)
  (h2 : team_avg_age = 26)
  (h3 : wicket_keeper_age_diff = 3)
  (h4 : remaining_avg_age = 23) :
  team_avg_age - remaining_avg_age = 3 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l3058_305862


namespace NUMINAMATH_CALUDE_smallest_integer_gcd_with_18_l3058_305889

theorem smallest_integer_gcd_with_18 : ∃ n : ℕ, n > 100 ∧ n.gcd 18 = 6 ∧ ∀ m : ℕ, m > 100 ∧ m.gcd 18 = 6 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_gcd_with_18_l3058_305889


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3058_305814

theorem min_value_quadratic (x y : ℝ) : x^2 + x*y + y^2 + 7 ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3058_305814


namespace NUMINAMATH_CALUDE_cos_one_third_solutions_l3058_305860

theorem cos_one_third_solutions (α : Real) (h1 : α ∈ Set.Icc 0 (2 * Real.pi)) (h2 : Real.cos α = 1/3) :
  α = Real.arccos (1/3) ∨ α = 2 * Real.pi - Real.arccos (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cos_one_third_solutions_l3058_305860


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3058_305820

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The x-axis -/
def xAxis : Set Point :=
  {p : Point | p.y = 0}

theorem point_on_x_axis (a : ℝ) :
  let P : Point := ⟨4, 2*a + 6⟩
  P ∈ xAxis → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3058_305820
