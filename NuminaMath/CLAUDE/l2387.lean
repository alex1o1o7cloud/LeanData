import Mathlib

namespace NUMINAMATH_CALUDE_cosine_sum_pentagon_l2387_238705

theorem cosine_sum_pentagon : 
  Real.cos (5 * π / 180) + Real.cos (77 * π / 180) + Real.cos (149 * π / 180) + 
  Real.cos (221 * π / 180) + Real.cos (293 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_pentagon_l2387_238705


namespace NUMINAMATH_CALUDE_infinite_primes_quadratic_equation_l2387_238760

theorem infinite_primes_quadratic_equation :
  ∀ (S : Finset Nat), ∃ (p : Nat), Prime p ∧ p ∉ S ∧ ∃ (x y : ℤ), x^2 + x + 1 = p * y := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_quadratic_equation_l2387_238760


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l2387_238763

/-- Given two lines that intersect at a specific point, find the value of k -/
theorem intersection_point_k_value (k : ℝ) : 
  (∃ y : ℝ, -4 * (-6) + y = k ∧ 1.5 * (-6) + y = 20) → k = 53 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l2387_238763


namespace NUMINAMATH_CALUDE_athletes_division_l2387_238761

theorem athletes_division (n : ℕ) (k : ℕ) : n = 10 ∧ k = 5 → (n.choose k) / 2 = 126 := by
  sorry

end NUMINAMATH_CALUDE_athletes_division_l2387_238761


namespace NUMINAMATH_CALUDE_optimal_walking_distance_ratio_l2387_238708

-- Define the problem setup
structure TravelProblem where
  totalDistance : ℝ
  speedA : ℝ
  speedB : ℝ
  speedC : ℝ
  mk_travel_problem : totalDistance > 0 ∧ speedA > 0 ∧ speedB > 0 ∧ speedC > 0

-- Define the optimal solution
def OptimalSolution (p : TravelProblem) :=
  ∃ (x : ℝ),
    0 < x ∧ x < p.totalDistance ∧
    (p.totalDistance - x) / p.speedA = x / (2 * p.speedC) + (p.totalDistance - x) / p.speedC

-- Theorem statement
theorem optimal_walking_distance_ratio 
  (p : TravelProblem) 
  (h_speeds : p.speedA = 4 ∧ p.speedB = 5 ∧ p.speedC = 12) 
  (h_optimal : OptimalSolution p) : 
  ∃ (distA distB : ℝ),
    distA > 0 ∧ distB > 0 ∧
    distA / distB = 17 / 10 ∧
    distA + distB = p.totalDistance :=
  sorry

end NUMINAMATH_CALUDE_optimal_walking_distance_ratio_l2387_238708


namespace NUMINAMATH_CALUDE_expression_evaluation_l2387_238759

theorem expression_evaluation :
  let a : ℤ := -2
  3 * a * (2 * a^2 - 4 * a + 3) - 2 * a^2 * (3 * a + 4) = -98 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2387_238759


namespace NUMINAMATH_CALUDE_mothers_age_l2387_238757

/-- Given a person and their mother, with the following conditions:
  1. The person's present age is two-fifths of the age of his mother.
  2. After 10 years, the person will be one-half of the age of his mother.
  This theorem proves that the mother's present age is 50 years. -/
theorem mothers_age (person_age mother_age : ℕ) 
  (h1 : person_age = (2 * mother_age) / 5)
  (h2 : person_age + 10 = (mother_age + 10) / 2) : 
  mother_age = 50 := by
  sorry

end NUMINAMATH_CALUDE_mothers_age_l2387_238757


namespace NUMINAMATH_CALUDE_gcd_n4_plus_27_and_n_plus_3_l2387_238734

theorem gcd_n4_plus_27_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^4 + 27) (n + 3) = if n % 3 = 0 then 3 else 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_n4_plus_27_and_n_plus_3_l2387_238734


namespace NUMINAMATH_CALUDE_factors_and_product_l2387_238788

-- Define a multiplication equation
def multiplication_equation (a b c : ℕ) : Prop := a * b = c

-- Define factors and product
def is_factor (a b c : ℕ) : Prop := multiplication_equation a b c
def is_product (a b c : ℕ) : Prop := multiplication_equation a b c

-- Theorem statement
theorem factors_and_product (a b c : ℕ) :
  multiplication_equation a b c → (is_factor a b c ∧ is_factor b a c ∧ is_product a b c) :=
by sorry

end NUMINAMATH_CALUDE_factors_and_product_l2387_238788


namespace NUMINAMATH_CALUDE_first_cat_weight_l2387_238742

theorem first_cat_weight (total_weight second_cat_weight third_cat_weight : ℕ) 
  (h1 : total_weight = 13)
  (h2 : second_cat_weight = 7)
  (h3 : third_cat_weight = 4)
  : total_weight - second_cat_weight - third_cat_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_cat_weight_l2387_238742


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l2387_238770

theorem line_slope_intercept_product (m b : ℚ) : 
  m = 3/4 → b = 2 → m * b > 1 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l2387_238770


namespace NUMINAMATH_CALUDE_reunion_attendance_overlap_l2387_238701

theorem reunion_attendance_overlap (total_guests : ℕ) (oates_attendees : ℕ) (hall_attendees : ℕ) (brown_attendees : ℕ)
  (h_total : total_guests = 200)
  (h_oates : oates_attendees = 60)
  (h_hall : hall_attendees = 90)
  (h_brown : brown_attendees = 80)
  (h_all_attend : total_guests ≤ oates_attendees + hall_attendees + brown_attendees) :
  let min_overlap := oates_attendees + hall_attendees + brown_attendees - total_guests
  let max_overlap := min oates_attendees (min hall_attendees brown_attendees)
  (min_overlap = 30 ∧ max_overlap = 60) :=
by sorry

end NUMINAMATH_CALUDE_reunion_attendance_overlap_l2387_238701


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l2387_238755

/-- Proves that given a father who is currently 45 years old, and after 15 years
    will be twice as old as his son, the current ratio of the father's age to
    the son's age is 3:1. -/
theorem father_son_age_ratio :
  ∀ (father_age son_age : ℕ),
    father_age = 45 →
    father_age + 15 = 2 * (son_age + 15) →
    father_age / son_age = 3 :=
by sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l2387_238755


namespace NUMINAMATH_CALUDE_solution_in_quadrant_III_l2387_238745

/-- 
Given a system of equations x - y = 4 and cx + y = 5, where c is a constant,
this theorem states that the solution (x, y) is in Quadrant III 
(i.e., x < 0 and y < 0) if and only if c < -1.
-/
theorem solution_in_quadrant_III (c : ℝ) :
  (∃ x y : ℝ, x - y = 4 ∧ c * x + y = 5 ∧ x < 0 ∧ y < 0) ↔ c < -1 :=
by sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_III_l2387_238745


namespace NUMINAMATH_CALUDE_tangency_condition_l2387_238735

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 6

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m*x^2 = 6

/-- The tangency condition -/
def are_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, parabola x y ∧ hyperbola m x y ∧
  ∀ x' y' : ℝ, parabola x' y' ∧ hyperbola m x' y' → (x = x' ∧ y = y')

/-- The theorem stating the condition for tangency -/
theorem tangency_condition :
  ∀ m : ℝ, are_tangent m ↔ (m = 12 + 10 * Real.sqrt 6 ∨ m = 12 - 10 * Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_tangency_condition_l2387_238735


namespace NUMINAMATH_CALUDE_no_egyptian_fraction_for_seven_seventeenths_l2387_238719

theorem no_egyptian_fraction_for_seven_seventeenths :
  ¬ ∃ (a b : ℕ+), (7 : ℚ) / 17 = 1 / (a : ℚ) + 1 / (b : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_no_egyptian_fraction_for_seven_seventeenths_l2387_238719


namespace NUMINAMATH_CALUDE_largest_fraction_sum_inequality_l2387_238709

theorem largest_fraction_sum_inequality 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≥ b) (hac : a ≥ c)
  (h_eq : a / b = c / d) : 
  a + d > b + c := by
sorry

end NUMINAMATH_CALUDE_largest_fraction_sum_inequality_l2387_238709


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2387_238791

theorem complex_number_quadrant (z : ℂ) (h : z = 1 - 2*I) : 
  (z.re > 0 ∧ z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2387_238791


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l2387_238732

theorem quadratic_inequality_no_solution :
  ∀ x : ℝ, 2 * x^2 - 3 * x + 4 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l2387_238732


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2387_238720

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2387_238720


namespace NUMINAMATH_CALUDE_unique_natural_solution_l2387_238722

theorem unique_natural_solution :
  ∀ n : ℕ, n ≠ 0 → (2 * n - 1 / (n^5 : ℚ) = 3 - 2 / (n : ℚ)) ↔ n = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_natural_solution_l2387_238722


namespace NUMINAMATH_CALUDE_inequality_proof_l2387_238774

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x * y + y * z + z * x = 1) : 
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2387_238774


namespace NUMINAMATH_CALUDE_range_of_a_l2387_238790

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2387_238790


namespace NUMINAMATH_CALUDE_cryptarithm_solution_exists_l2387_238729

theorem cryptarithm_solution_exists : ∃ (Φ E B P A J : ℕ), 
  Φ < 10 ∧ E < 10 ∧ B < 10 ∧ P < 10 ∧ A < 10 ∧ J < 10 ∧
  Φ ≠ E ∧ Φ ≠ B ∧ Φ ≠ P ∧ Φ ≠ A ∧ Φ ≠ J ∧
  E ≠ B ∧ E ≠ P ∧ E ≠ A ∧ E ≠ J ∧
  B ≠ P ∧ B ≠ A ∧ B ≠ J ∧
  P ≠ A ∧ P ≠ J ∧
  A ≠ J ∧
  E ≠ 0 ∧ A ≠ 0 ∧ J ≠ 0 ∧
  (Φ : ℚ) / E + (B * 10 + P : ℚ) / (A * J) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_exists_l2387_238729


namespace NUMINAMATH_CALUDE_equation_solutions_l2387_238764

theorem equation_solutions :
  (∃ x : ℝ, x^2 = 7) ∧
  (∃ x : ℝ, x^2 + 8*x = 0) ∧
  (∃ x : ℝ, x^2 - 4*x - 3 = 0) ∧
  (∃ x : ℝ, x*(x - 2) = 2 - x) ∧
  (∀ x : ℝ, x^2 = 7 → x = Real.sqrt 7 ∨ x = -Real.sqrt 7) ∧
  (∀ x : ℝ, x^2 + 8*x = 0 → x = 0 ∨ x = -8) ∧
  (∀ x : ℝ, x^2 - 4*x - 3 = 0 → x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) ∧
  (∀ x : ℝ, x*(x - 2) = 2 - x → x = 2 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2387_238764


namespace NUMINAMATH_CALUDE_find_b_value_l2387_238765

theorem find_b_value (b : ℝ) : (5 : ℝ)^2 + b * 5 - 35 = 0 → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l2387_238765


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2387_238728

/-- An arithmetic sequence with positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_positive : d > 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- The sum of the first three terms of the sequence is 15 -/
def sum_first_three (seq : ArithmeticSequence) : Prop :=
  seq.a 1 + seq.a 2 + seq.a 3 = 15

/-- The product of the first three terms of the sequence is 80 -/
def product_first_three (seq : ArithmeticSequence) : Prop :=
  seq.a 1 * seq.a 2 * seq.a 3 = 80

/-- Theorem: If the sum of the first three terms is 15 and their product is 80,
    then the sum of the 11th, 12th, and 13th terms is 135 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
  (h_sum : sum_first_three seq) (h_product : product_first_three seq) :
  seq.a 11 + seq.a 12 + seq.a 13 = 135 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2387_238728


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2387_238778

theorem arithmetic_mean_after_removal (S : Finset ℝ) (a b c : ℝ) :
  S.card = 60 →
  a = 48 ∧ b = 52 ∧ c = 56 →
  a ∈ S ∧ b ∈ S ∧ c ∈ S →
  (S.sum id) / S.card = 42 →
  ((S.sum id) - (a + b + c)) / (S.card - 3) = 41.47 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2387_238778


namespace NUMINAMATH_CALUDE_problem_solution_l2387_238737

theorem problem_solution : (((3⁻¹ : ℚ) + 7^3 - 2)⁻¹ * 7 : ℚ) = 21 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2387_238737


namespace NUMINAMATH_CALUDE_divisors_of_1200_l2387_238726

theorem divisors_of_1200 : Finset.card (Nat.divisors 1200) = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_1200_l2387_238726


namespace NUMINAMATH_CALUDE_no_integer_solution_l2387_238777

theorem no_integer_solution : ¬ ∃ (a b c d : ℤ),
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2387_238777


namespace NUMINAMATH_CALUDE_skirt_cut_amount_l2387_238738

/-- The amount cut off the pants in inches -/
def pants_cut : ℝ := 0.5

/-- The additional amount cut off the skirt compared to the pants in inches -/
def additional_skirt_cut : ℝ := 0.25

/-- The total amount cut off the skirt in inches -/
def skirt_cut : ℝ := pants_cut + additional_skirt_cut

theorem skirt_cut_amount : skirt_cut = 0.75 := by sorry

end NUMINAMATH_CALUDE_skirt_cut_amount_l2387_238738


namespace NUMINAMATH_CALUDE_student_average_age_l2387_238707

theorem student_average_age (n : ℕ) (teacher_age : ℕ) (new_average : ℝ) :
  n = 50 ∧ teacher_age = 65 ∧ new_average = 15 →
  (n : ℝ) * (((n : ℝ) * new_average - teacher_age) / n) + teacher_age = (n + 1 : ℝ) * new_average →
  ((n : ℝ) * new_average - teacher_age) / n = 14 := by
  sorry

end NUMINAMATH_CALUDE_student_average_age_l2387_238707


namespace NUMINAMATH_CALUDE_compute_expression_l2387_238797

theorem compute_expression : 9 + 4 * (5 - 2 * 3)^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2387_238797


namespace NUMINAMATH_CALUDE_million_to_scientific_notation_l2387_238711

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem million_to_scientific_notation :
  toScientificNotation (42.39 * 1000000) = ScientificNotation.mk 4.239 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_million_to_scientific_notation_l2387_238711


namespace NUMINAMATH_CALUDE_unique_number_property_l2387_238727

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_property_l2387_238727


namespace NUMINAMATH_CALUDE_fifteenth_term_geometric_sequence_l2387_238706

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

theorem fifteenth_term_geometric_sequence :
  geometric_sequence 12 (1/3) 15 = 4/1594323 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_geometric_sequence_l2387_238706


namespace NUMINAMATH_CALUDE_mark_fish_problem_l2387_238762

/-- Calculates the total number of young fish given the number of tanks, 
    pregnant fish per tank, and young per fish. -/
def total_young_fish (num_tanks : ℕ) (fish_per_tank : ℕ) (young_per_fish : ℕ) : ℕ :=
  num_tanks * fish_per_tank * young_per_fish

/-- Theorem stating that with 3 tanks, 4 pregnant fish per tank, and 20 young per fish, 
    the total number of young fish is 240. -/
theorem mark_fish_problem :
  total_young_fish 3 4 20 = 240 := by
  sorry

end NUMINAMATH_CALUDE_mark_fish_problem_l2387_238762


namespace NUMINAMATH_CALUDE_correct_formula_l2387_238731

def f (x : ℝ) : ℝ := 5 * x^2 + x

theorem correct_formula : 
  (f 0 = 0) ∧ 
  (f 1 = 20) ∧ 
  (f 2 = 60) ∧ 
  (f 3 = 120) ∧ 
  (f 4 = 200) := by
  sorry

end NUMINAMATH_CALUDE_correct_formula_l2387_238731


namespace NUMINAMATH_CALUDE_car_trade_profit_percentage_l2387_238739

/-- Calculates the profit percentage on the original price when a car is bought at a discount and sold at an increase. -/
theorem car_trade_profit_percentage 
  (original_price : ℝ) 
  (discount_percentage : ℝ) 
  (increase_percentage : ℝ) 
  (h1 : discount_percentage = 20) 
  (h2 : increase_percentage = 50) 
  : (((1 - discount_percentage / 100) * (1 + increase_percentage / 100) - 1) * 100 = 20) := by
sorry

end NUMINAMATH_CALUDE_car_trade_profit_percentage_l2387_238739


namespace NUMINAMATH_CALUDE_truck_filling_problem_l2387_238768

/-- A problem about filling a truck with stone blocks -/
theorem truck_filling_problem 
  (truck_capacity : ℕ) 
  (initial_workers : ℕ) 
  (work_rate : ℕ) 
  (initial_work_time : ℕ) 
  (total_time : ℕ)
  (h1 : truck_capacity = 6000)
  (h2 : initial_workers = 2)
  (h3 : work_rate = 250)
  (h4 : initial_work_time = 4)
  (h5 : total_time = 6)
  : ∃ (joined_workers : ℕ),
    (initial_workers * work_rate * initial_work_time) + 
    ((initial_workers + joined_workers) * work_rate * (total_time - initial_work_time)) = 
    truck_capacity ∧ joined_workers = 6 := by
  sorry


end NUMINAMATH_CALUDE_truck_filling_problem_l2387_238768


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2387_238798

/-- The line y - 1 = k(x - 1) always intersects the circle x² + y² - 2y = 0 for any real number k. -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  (y - 1 = k * (x - 1)) ∧ (x^2 + y^2 - 2*y = 0) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2387_238798


namespace NUMINAMATH_CALUDE_general_inequality_l2387_238789

theorem general_inequality (x n : ℝ) (h1 : x > 0) (h2 : n > 0) 
  (h3 : ∃ (a : ℝ), a > 0 ∧ x + a / x^n ≥ n + 1) :
  ∃ (a : ℝ), a = n^n ∧ x + a / x^n ≥ n + 1 :=
sorry

end NUMINAMATH_CALUDE_general_inequality_l2387_238789


namespace NUMINAMATH_CALUDE_tangent_line_to_sqrt_curve_l2387_238724

theorem tangent_line_to_sqrt_curve (x y : ℝ) :
  (∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧
    (a * 1 + b * 2 + c = 0) ∧
    (∃ (x₀ : ℝ), x₀ > 0 ∧ 
      a * x₀ + b * Real.sqrt x₀ + c = 0 ∧
      a + b * (1 / (2 * Real.sqrt x₀)) = 0)) ↔
  ((x - (4 + 2 * Real.sqrt 3) * y + (7 + 4 * Real.sqrt 3) = 0) ∨
   (x - (4 - 2 * Real.sqrt 3) * y + (7 - 4 * Real.sqrt 3) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_sqrt_curve_l2387_238724


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2387_238740

-- Define the polynomial
def p (x : ℝ) : ℝ := (x^2 - 5*x + 6) * x * (x - 4) * (x - 6)

-- State the theorem
theorem roots_of_polynomial : 
  {x : ℝ | p x = 0} = {0, 2, 3, 4, 6} := by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2387_238740


namespace NUMINAMATH_CALUDE_sibling_age_sum_l2387_238704

/-- Given the ages of four siblings with specific relationships, prove that the sum of three of their ages is 25. -/
theorem sibling_age_sum : 
  ∀ (juliet maggie ralph nicky : ℕ),
  juliet = maggie + 3 →
  ralph = juliet + 2 →
  2 * nicky = ralph →
  juliet = 10 →
  maggie + ralph + nicky = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_sibling_age_sum_l2387_238704


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2387_238746

theorem sqrt_product_equality : 2 * Real.sqrt 3 * (3 * Real.sqrt 2) = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2387_238746


namespace NUMINAMATH_CALUDE_classroom_seating_l2387_238795

/-- Given a classroom with 53 students seated in rows of either 6 or 7 students,
    with all seats occupied, prove that the number of rows seating exactly 7 students is 5. -/
theorem classroom_seating (total_students : ℕ) (rows_with_seven : ℕ) : 
  total_students = 53 →
  (∃ (rows_with_six : ℕ), total_students = 7 * rows_with_seven + 6 * rows_with_six) →
  rows_with_seven = 5 := by
  sorry

end NUMINAMATH_CALUDE_classroom_seating_l2387_238795


namespace NUMINAMATH_CALUDE_bianca_books_total_l2387_238776

theorem bianca_books_total (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 8)
  (h2 : mystery_shelves = 5)
  (h3 : picture_shelves = 4) :
  books_per_shelf * (mystery_shelves + picture_shelves) = 72 :=
by sorry

end NUMINAMATH_CALUDE_bianca_books_total_l2387_238776


namespace NUMINAMATH_CALUDE_games_in_own_group_l2387_238781

/-- Represents a baseball league with two groups of teams. -/
structure BaseballLeague where
  n : ℕ  -- Number of games played against each team in own group
  m : ℕ  -- Number of games played against each team in other group

/-- Theorem about the number of games played within a team's own group. -/
theorem games_in_own_group (league : BaseballLeague)
  (h1 : league.n > 2 * league.m)
  (h2 : league.m > 4)
  (h3 : 3 * league.n + 4 * league.m = 76) :
  3 * league.n = 48 := by
sorry

end NUMINAMATH_CALUDE_games_in_own_group_l2387_238781


namespace NUMINAMATH_CALUDE_third_class_proportion_l2387_238773

theorem third_class_proportion (first_class second_class third_class : ℕ) 
  (h1 : first_class = 30)
  (h2 : second_class = 50)
  (h3 : third_class = 20) :
  (third_class : ℚ) / (first_class + second_class + third_class : ℚ) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_third_class_proportion_l2387_238773


namespace NUMINAMATH_CALUDE_min_value_complex_sum_l2387_238754

theorem min_value_complex_sum (a b c d : ℤ) (ζ : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_fourth_root : ζ^4 = 1)
  (h_not_one : ζ ≠ 1) :
  ∃ (m : ℝ), m = 2 ∧ ∀ (x y z w : ℤ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
    Complex.abs (x + y*ζ + z*ζ^2 + w*ζ^3) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_sum_l2387_238754


namespace NUMINAMATH_CALUDE_all_solutions_are_scalar_multiples_of_base_l2387_238733

/-- A quadruple of real numbers satisfying the given equation -/
structure Quadruple where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h : a * (b + c) = b * (c + d) ∧ b * (c + d) = c * (d + a) ∧ c * (d + a) = d * (a + b)

/-- The set of base solutions -/
def BaseSolutions : Set Quadruple := {
  ⟨1, 0, 0, 0, sorry⟩,
  ⟨0, 1, 0, 0, sorry⟩,
  ⟨0, 0, 1, 0, sorry⟩,
  ⟨0, 0, 0, 1, sorry⟩,
  ⟨1, 1, 1, 1, sorry⟩,
  ⟨1, -1, 1, -1, sorry⟩,
  ⟨1, -1 + Real.sqrt 2, -1, 1 - Real.sqrt 2, sorry⟩,
  ⟨1, -1 - Real.sqrt 2, -1, 1 + Real.sqrt 2, sorry⟩
}

/-- Main theorem: All solutions are scalar multiples of base solutions -/
theorem all_solutions_are_scalar_multiples_of_base (q : Quadruple) :
  ∃ (k : ℝ) (b : Quadruple), b ∈ BaseSolutions ∧
    q.a = k * b.a ∧ q.b = k * b.b ∧ q.c = k * b.c ∧ q.d = k * b.d :=
  sorry

end NUMINAMATH_CALUDE_all_solutions_are_scalar_multiples_of_base_l2387_238733


namespace NUMINAMATH_CALUDE_expression_simplification_l2387_238730

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 3 * x + y / 3 ≠ 0) :
  (3 * x + y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹) = (3 * x * y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2387_238730


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l2387_238767

/-- A triangle with an inscribed rectangle -/
structure InscribedRectangle where
  /-- The height of the triangle -/
  triangle_height : ℝ
  /-- The base of the triangle -/
  triangle_base : ℝ
  /-- The width of the inscribed rectangle -/
  rectangle_width : ℝ
  /-- The length of the inscribed rectangle -/
  rectangle_length : ℝ
  /-- The width of the rectangle is one-third of its length -/
  width_is_third_of_length : rectangle_width = rectangle_length / 3
  /-- The rectangle is inscribed in the triangle -/
  rectangle_inscribed : rectangle_length ≤ triangle_base

/-- The area of the inscribed rectangle given the triangle's dimensions -/
def rectangle_area (r : InscribedRectangle) : ℝ :=
  r.rectangle_width * r.rectangle_length

/-- Theorem: The area of the inscribed rectangle is 675/64 square inches -/
theorem inscribed_rectangle_area (r : InscribedRectangle)
    (h1 : r.triangle_height = 9)
    (h2 : r.triangle_base = 15) :
    rectangle_area r = 675 / 64 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l2387_238767


namespace NUMINAMATH_CALUDE_square_floor_tiles_l2387_238792

theorem square_floor_tiles (s : ℕ) (h1 : s > 0) : 
  (2 * s - 1 : ℝ) / (s^2 : ℝ) = 0.41 → s^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l2387_238792


namespace NUMINAMATH_CALUDE_stan_run_time_l2387_238710

/-- Calculates the total run time given the number of 3-minute songs, 2-minute songs, and additional time needed. -/
def total_run_time (three_min_songs : ℕ) (two_min_songs : ℕ) (additional_time : ℕ) : ℕ :=
  three_min_songs * 3 + two_min_songs * 2 + additional_time

/-- Proves that given 10 3-minute songs, 15 2-minute songs, and 40 minutes of additional time, the total run time is 100 minutes. -/
theorem stan_run_time :
  total_run_time 10 15 40 = 100 := by
  sorry

end NUMINAMATH_CALUDE_stan_run_time_l2387_238710


namespace NUMINAMATH_CALUDE_f_value_at_negative_pi_third_l2387_238743

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (Real.cos x)^2 - b * Real.sin x * Real.cos x - a / 2

theorem f_value_at_negative_pi_third (a b : ℝ) :
  (∃ (x : ℝ), f a b x ≤ 1/2) ∧
  (f a b (π/3) = Real.sqrt 3 / 4) →
  (f a b (-π/3) = 0 ∨ f a b (-π/3) = -Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_f_value_at_negative_pi_third_l2387_238743


namespace NUMINAMATH_CALUDE_wednesday_savings_l2387_238784

/-- Represents Donny's savings throughout the week -/
structure WeekSavings where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Calculates the total savings before Thursday -/
def total_savings (s : WeekSavings) : ℕ :=
  s.monday + s.tuesday + s.wednesday

theorem wednesday_savings (s : WeekSavings) 
  (h1 : s.monday = 15)
  (h2 : s.tuesday = 28)
  (h3 : total_savings s / 2 = 28) : 
  s.wednesday = 13 := by
sorry

end NUMINAMATH_CALUDE_wednesday_savings_l2387_238784


namespace NUMINAMATH_CALUDE_min_value_expression_l2387_238787

theorem min_value_expression (x : ℝ) (h : x > 10) : 
  x^2 / (x - 10) ≥ 40 ∧ ∃ x₀ > 10, x₀^2 / (x₀ - 10) = 40 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2387_238787


namespace NUMINAMATH_CALUDE_dorchester_daily_pay_l2387_238715

/-- Represents Dorchester's earnings at the puppy wash -/
structure PuppyWashEarnings where
  dailyPay : ℝ
  puppyWashRate : ℝ
  puppiesWashed : ℕ
  totalEarnings : ℝ

/-- Dorchester's earnings satisfy the given conditions -/
def dorchesterEarnings : PuppyWashEarnings where
  dailyPay := 40
  puppyWashRate := 2.25
  puppiesWashed := 16
  totalEarnings := 76

/-- Theorem: Dorchester's daily pay is $40 given the conditions -/
theorem dorchester_daily_pay :
  dorchesterEarnings.dailyPay = 40 ∧
  dorchesterEarnings.totalEarnings = dorchesterEarnings.dailyPay +
    dorchesterEarnings.puppyWashRate * dorchesterEarnings.puppiesWashed :=
by sorry

end NUMINAMATH_CALUDE_dorchester_daily_pay_l2387_238715


namespace NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l2387_238721

/-- Given two vectors that are normal vectors of parallel planes, prove that their specific components multiply to -3 -/
theorem parallel_planes_normal_vectors (m n : ℝ) : 
  let a : Fin 3 → ℝ := ![0, 1, m]
  let b : Fin 3 → ℝ := ![0, n, -3]
  (∃ (k : ℝ), a = k • b) →  -- Parallel planes condition
  m * n = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l2387_238721


namespace NUMINAMATH_CALUDE_difference_of_squares_value_l2387_238713

theorem difference_of_squares_value (x y : ℤ) (hx : x = -5) (hy : y = -10) :
  (y - x) * (y + x) = 75 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_value_l2387_238713


namespace NUMINAMATH_CALUDE_sum_of_unique_decimals_sum_of_unique_decimals_proof_l2387_238772

/-- The sum of all unique decimals formed by 4 distinct digit cards and 1 decimal point card -/
theorem sum_of_unique_decimals : ℝ :=
  let digit_sum := (0 : ℕ) + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
  let num_permutations := 24
  let num_decimal_positions := 4
  666.6

/-- The number of unique decimals that can be formed -/
def num_unique_decimals : ℕ := 72

theorem sum_of_unique_decimals_proof :
  sum_of_unique_decimals = 666.6 ∧ num_unique_decimals = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_unique_decimals_sum_of_unique_decimals_proof_l2387_238772


namespace NUMINAMATH_CALUDE_shane_sandwiches_l2387_238756

/-- The number of slices in each package of sliced ham -/
def slices_per_ham_package (
  bread_slices_per_package : ℕ)
  (num_bread_packages : ℕ)
  (num_ham_packages : ℕ)
  (leftover_bread_slices : ℕ)
  (bread_slices_per_sandwich : ℕ) : ℕ :=
  let total_bread_slices := bread_slices_per_package * num_bread_packages
  let used_bread_slices := total_bread_slices - leftover_bread_slices
  let num_sandwiches := used_bread_slices / bread_slices_per_sandwich
  num_sandwiches / num_ham_packages

theorem shane_sandwiches :
  slices_per_ham_package 20 2 2 8 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_shane_sandwiches_l2387_238756


namespace NUMINAMATH_CALUDE_vowel_initial_probability_theorem_l2387_238748

/-- The number of students in the class -/
def total_students : ℕ := 26

/-- The number of vowels (including 'Y') -/
def vowel_count : ℕ := 6

/-- The probability of selecting a student with vowel initials -/
def vowel_initial_probability : ℚ := 3 / 13

/-- Theorem stating the probability of selecting a student with vowel initials -/
theorem vowel_initial_probability_theorem :
  (vowel_count : ℚ) / total_students = vowel_initial_probability := by
  sorry

end NUMINAMATH_CALUDE_vowel_initial_probability_theorem_l2387_238748


namespace NUMINAMATH_CALUDE_leo_current_weight_l2387_238747

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 80

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 140 - leo_weight

/-- The combined weight of Leo and Kendra in pounds -/
def combined_weight : ℝ := 140

theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = combined_weight) ∧
  (leo_weight = 80) :=
by sorry

end NUMINAMATH_CALUDE_leo_current_weight_l2387_238747


namespace NUMINAMATH_CALUDE_climb_8_stairs_l2387_238751

/-- The number of ways to climb n stairs, taking 1, 2, 3, or 4 steps at a time. -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | n + 4 => climbStairs n + climbStairs (n + 1) + climbStairs (n + 2) + climbStairs (n + 3)

/-- Theorem stating that there are 108 ways to climb 8 stairs -/
theorem climb_8_stairs : climbStairs 8 = 108 := by
  sorry

end NUMINAMATH_CALUDE_climb_8_stairs_l2387_238751


namespace NUMINAMATH_CALUDE_distance_AB_is_5360_l2387_238700

/-- Represents a person in the problem -/
inductive Person
| A
| B
| C

/-- Represents a point on the path -/
structure Point where
  x : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  A : Point
  B : Point
  C : Point
  D : Point
  initialSpeed : Person → ℝ
  returnSpeed : Person → ℝ
  distanceTraveled : Person → Point → ℝ

/-- The main theorem to be proved -/
theorem distance_AB_is_5360 (setup : ProblemSetup) : 
  setup.B.x - setup.A.x = 5360 :=
sorry

end NUMINAMATH_CALUDE_distance_AB_is_5360_l2387_238700


namespace NUMINAMATH_CALUDE_sum_of_q_p_equals_negative_twenty_l2387_238752

def p (x : ℝ) : ℝ := x^2 - 4

def q (x : ℝ) : ℝ := -abs x

def xValues : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_q_p_equals_negative_twenty :
  (xValues.map (λ x => q (p x))).sum = -20 := by sorry

end NUMINAMATH_CALUDE_sum_of_q_p_equals_negative_twenty_l2387_238752


namespace NUMINAMATH_CALUDE_election_winner_margin_l2387_238785

theorem election_winner_margin (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (992 : ℚ) / total_votes = 62 / 100) : 
  992 - (total_votes - 992) = 384 := by
sorry

end NUMINAMATH_CALUDE_election_winner_margin_l2387_238785


namespace NUMINAMATH_CALUDE_function_inequality_l2387_238779

/-- Given a function f : ℝ → ℝ with derivative f', prove that if f'(x) < f(x) for all x,
    then f(2) < e^2 * f(0) and f(2012) < e^2012 * f(0) -/
theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (hf' : ∀ x, deriv f x = f' x) (h : ∀ x, f' x < f x) : 
    f 2 < Real.exp 2 * f 0 ∧ f 2012 < Real.exp 2012 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2387_238779


namespace NUMINAMATH_CALUDE_smallest_angle_in_ratio_triangle_l2387_238723

theorem smallest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  (a : ℝ) / 4 = (b : ℝ) / 5 →
  (a : ℝ) / 4 = (c : ℝ) / 7 →
  a + b + c = 180 →
  a = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_ratio_triangle_l2387_238723


namespace NUMINAMATH_CALUDE_complex_calculation_l2387_238753

theorem complex_calculation (A M S : ℂ) (P : ℝ) 
  (hA : A = 5 - 2*I) 
  (hM : M = -3 + 2*I) 
  (hS : S = 2*I) 
  (hP : P = 3) : 
  2 * (A - M + S - P) = 10 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_calculation_l2387_238753


namespace NUMINAMATH_CALUDE_union_M_N_l2387_238718

def M : Set ℝ := {x | 1 / x > 1}
def N : Set ℝ := {x | x^2 + 2*x - 3 < 0}

theorem union_M_N : M ∪ N = Set.Ioo (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_union_M_N_l2387_238718


namespace NUMINAMATH_CALUDE_g_sum_theorem_l2387_238717

def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + d * x^2 + 5

theorem g_sum_theorem (a b c d : ℝ) (h : g a b c d 2 = 4) :
  g a b c d 2 + g a b c d (-2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_theorem_l2387_238717


namespace NUMINAMATH_CALUDE_simplify_expression_l2387_238741

theorem simplify_expression (x : ℝ) : ((3 * x + 8) - 5 * x) / 2 = -x + 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2387_238741


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2387_238749

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x + 4| :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2387_238749


namespace NUMINAMATH_CALUDE_fraction_equality_l2387_238714

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hyx : y - x^2 ≠ 0) :
  (x^2 - 1/y) / (y - x^2) = (x^2 * y - 1) / (y^2 - x^2 * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2387_238714


namespace NUMINAMATH_CALUDE_marks_reading_increase_l2387_238736

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Mark's current daily reading time in hours -/
def current_daily_reading : ℕ := 2

/-- Mark's desired weekly reading time in hours -/
def desired_weekly_reading : ℕ := 18

/-- Calculate the increase in Mark's weekly reading time -/
def reading_time_increase : ℕ :=
  desired_weekly_reading - (current_daily_reading * days_in_week)

/-- Theorem stating that Mark's weekly reading time increase is 4 hours -/
theorem marks_reading_increase : reading_time_increase = 4 := by
  sorry

end NUMINAMATH_CALUDE_marks_reading_increase_l2387_238736


namespace NUMINAMATH_CALUDE_concatenation_puzzle_l2387_238716

theorem concatenation_puzzle :
  ∃! (a b : ℕ),
    100 ≤ a ∧ a ≤ 999 ∧
    1000 ≤ b ∧ b ≤ 9999 ∧
    10000 * a + b = 11 * a * b ∧
    a + b = 1093 := by
  sorry

end NUMINAMATH_CALUDE_concatenation_puzzle_l2387_238716


namespace NUMINAMATH_CALUDE_arrangement_and_selection_theorem_l2387_238750

def girls : ℕ := 3
def boys : ℕ := 4
def total_people : ℕ := girls + boys

def arrangements_no_adjacent_girls : ℕ := (Nat.factorial boys) * (Nat.choose (boys + 1) girls)

def selections_with_at_least_one_girl : ℕ := Nat.choose total_people 3 - Nat.choose boys 3

theorem arrangement_and_selection_theorem :
  (arrangements_no_adjacent_girls = 1440) ∧
  (selections_with_at_least_one_girl = 31) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_and_selection_theorem_l2387_238750


namespace NUMINAMATH_CALUDE_quadratic_value_l2387_238783

-- Define the quadratic function
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- State the theorem
theorem quadratic_value (p q : ℝ) :
  f p q 1 = 3 → f p q (-3) = 7 → f p q (-5) = 21 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_l2387_238783


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2387_238771

theorem simplify_and_rationalize : 
  (Real.sqrt 8 / Real.sqrt 3) * (Real.sqrt 25 / Real.sqrt 30) * (Real.sqrt 16 / Real.sqrt 21) = 
  4 * Real.sqrt 14 / 63 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2387_238771


namespace NUMINAMATH_CALUDE_local_maximum_at_two_l2387_238775

/-- The function f(x) = x(x-c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) --/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

theorem local_maximum_at_two (c : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), f c x ≤ f c 2) →
  (f_derivative c 2 = 0) →
  (∀ x ∈ Set.Ioo (2 - δ) 2, f_derivative c x > 0) →
  (∀ x ∈ Set.Ioo 2 (2 + δ), f_derivative c x < 0) →
  c = 6 := by
  sorry

end NUMINAMATH_CALUDE_local_maximum_at_two_l2387_238775


namespace NUMINAMATH_CALUDE_transaction_handling_l2387_238744

/-- Problem: Transaction Handling --/
theorem transaction_handling 
  (mabel_transactions : ℕ)
  (anthony_percentage : ℚ)
  (cal_fraction : ℚ)
  (jade_additional : ℕ)
  (h1 : mabel_transactions = 90)
  (h2 : anthony_percentage = 11/10)
  (h3 : cal_fraction = 2/3)
  (h4 : jade_additional = 18) :
  let anthony_transactions := mabel_transactions * anthony_percentage
  let cal_transactions := anthony_transactions * cal_fraction
  let jade_transactions := cal_transactions + jade_additional
  jade_transactions = 84 := by
sorry

end NUMINAMATH_CALUDE_transaction_handling_l2387_238744


namespace NUMINAMATH_CALUDE_exists_k_undecided_tournament_l2387_238712

/-- A tournament is represented as a function that takes two players and returns true if the first player defeats the second, and false otherwise. -/
def Tournament (n : ℕ) := Fin n → Fin n → Bool

/-- A tournament is k-undecided if for any set of k players, there exists a player who has defeated all of them. -/
def IsKUndecided (k : ℕ) (n : ℕ) (t : Tournament n) : Prop :=
  ∀ (A : Finset (Fin n)), A.card = k →
    ∃ (p : Fin n), ∀ (a : Fin n), a ∈ A → t p a = true

/-- For any positive integer k, there exists a k-undecided tournament with more than k players. -/
theorem exists_k_undecided_tournament (k : ℕ+) :
  ∃ (n : ℕ), n > k ∧ ∃ (t : Tournament n), IsKUndecided k n t :=
sorry

end NUMINAMATH_CALUDE_exists_k_undecided_tournament_l2387_238712


namespace NUMINAMATH_CALUDE_simplified_ratio_of_stickers_l2387_238769

theorem simplified_ratio_of_stickers (kate_stickers : ℕ) (jenna_stickers : ℕ) 
  (h1 : kate_stickers = 21) (h2 : jenna_stickers = 12) : 
  (kate_stickers / Nat.gcd kate_stickers jenna_stickers : ℚ) / 
  (jenna_stickers / Nat.gcd kate_stickers jenna_stickers : ℚ) = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplified_ratio_of_stickers_l2387_238769


namespace NUMINAMATH_CALUDE_equation_solutions_l2387_238796

theorem equation_solutions : ∃ (x₁ x₂ x₃ : ℝ),
  (x₁ = 3 ∧ x₂ = (2 + Real.sqrt 1121) / 14 ∧ x₃ = (2 - Real.sqrt 1121) / 14) ∧
  (∀ x : ℝ, (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 60 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2387_238796


namespace NUMINAMATH_CALUDE_point_line_distance_l2387_238794

/-- Given a point (4, 3) and a line 3x - 4y + a = 0, if the distance from the point to the line is 1, then a = ±5 -/
theorem point_line_distance (a : ℝ) : 
  let point : ℝ × ℝ := (4, 3)
  let line_equation (x y : ℝ) := 3 * x - 4 * y + a
  let distance := |line_equation point.1 point.2| / Real.sqrt (3^2 + (-4)^2)
  distance = 1 → a = 5 ∨ a = -5 := by
sorry

end NUMINAMATH_CALUDE_point_line_distance_l2387_238794


namespace NUMINAMATH_CALUDE_remaining_distance_l2387_238758

-- Define the total distance to the concert
def total_distance : ℕ := 78

-- Define the distance already driven
def distance_driven : ℕ := 32

-- Theorem to prove the remaining distance
theorem remaining_distance : total_distance - distance_driven = 46 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l2387_238758


namespace NUMINAMATH_CALUDE_jerry_candy_count_jerry_candy_count_proof_l2387_238786

theorem jerry_candy_count : ℕ → Prop :=
  fun total_candy : ℕ =>
    ∃ (candy_per_bag : ℕ),
      -- Total number of bags
      (9 : ℕ) * candy_per_bag = total_candy ∧
      -- Number of non-chocolate bags
      (9 - 2 - 3 : ℕ) * candy_per_bag = 28 ∧
      -- The result we want to prove
      total_candy = 63

-- The proof of the theorem
theorem jerry_candy_count_proof : jerry_candy_count 63 := by
  sorry

end NUMINAMATH_CALUDE_jerry_candy_count_jerry_candy_count_proof_l2387_238786


namespace NUMINAMATH_CALUDE_m_range_l2387_238725

/-- A function that represents f(x) = -x^2 - tx + 3t --/
def f (t : ℝ) (x : ℝ) : ℝ := -x^2 - t*x + 3*t

/-- Predicate that checks if f has only one zero in the interval (0, 2) --/
def has_one_zero_in_interval (t : ℝ) : Prop :=
  ∃! x, 0 < x ∧ x < 2 ∧ f t x = 0

/-- The sufficient but not necessary condition --/
def sufficient_condition (m : ℝ) : Prop :=
  ∀ t, 0 < t ∧ t < m → has_one_zero_in_interval t

/-- The theorem stating the range of m --/
theorem m_range :
  ∀ m, (m > 0 ∧ sufficient_condition m ∧ ¬(∀ t, has_one_zero_in_interval t → (0 < t ∧ t < m))) ↔
       (0 < m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_m_range_l2387_238725


namespace NUMINAMATH_CALUDE_triangle_division_theorem_l2387_238780

-- Define a triangle
structure Triangle :=
  (A B C : Point)

-- Define a point inside a triangle
def PointInside (T : Triangle) (P : Point) : Prop :=
  -- Placeholder for the condition that P is inside triangle T
  sorry

-- Define a point on a side of a triangle
def PointOnSide (T : Triangle) (Q : Point) : Prop :=
  -- Placeholder for the condition that Q is on a side of triangle T
  sorry

-- Define the property of not sharing an entire side
def NotShareEntireSide (T1 T2 : Triangle) : Prop :=
  -- Placeholder for the condition that T1 and T2 do not share an entire side
  sorry

theorem triangle_division_theorem (T : Triangle) :
  ∃ (P Q : Point) (T1 T2 T3 T4 : Triangle),
    PointInside T P ∧
    PointOnSide T Q ∧
    NotShareEntireSide T1 T2 ∧
    NotShareEntireSide T1 T3 ∧
    NotShareEntireSide T1 T4 ∧
    NotShareEntireSide T2 T3 ∧
    NotShareEntireSide T2 T4 ∧
    NotShareEntireSide T3 T4 :=
  sorry

end NUMINAMATH_CALUDE_triangle_division_theorem_l2387_238780


namespace NUMINAMATH_CALUDE_equidistant_point_y_axis_l2387_238702

/-- The y-coordinate of the point on the y-axis that is equidistant from A(-3, -2) and B(2, 3) is 0 -/
theorem equidistant_point_y_axis : ∃ y : ℝ, 
  (y = 0) ∧ 
  ((-3 - 0)^2 + (-2 - y)^2 = (2 - 0)^2 + (3 - y)^2) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_axis_l2387_238702


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2387_238766

/-- Two lines y = ax - 2 and y = x + 1 are perpendicular if and only if a = -1 -/
theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, y = a * x - 2 ∧ y = x + 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2387_238766


namespace NUMINAMATH_CALUDE_problem_solution_l2387_238799

-- Define the absolute value function
def f (x : ℝ) : ℝ := abs x

-- Define the function g
def g (t : ℝ) (x : ℝ) : ℝ := 3 * f x - f (x - t)

theorem problem_solution :
  -- Part I
  (∀ a : ℝ, a > 0 → (∀ x : ℝ, 2 * f (x - 1) + f (2 * x - a) ≥ 1) →
    (a ∈ Set.Ioo 0 1 ∪ Set.Ici 3)) ∧
  -- Part II
  (∀ t : ℝ, t ≠ 0 →
    (∫ x, abs (g t x)) = 3 →
    t = 2 * Real.sqrt 2 ∨ t = -2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2387_238799


namespace NUMINAMATH_CALUDE_a_range_when_f_decreasing_l2387_238782

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Define the property of f being decreasing on (-∞, 6)
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → x < 6 → y < 6 → f a x > f a y

-- State the theorem
theorem a_range_when_f_decreasing (a : ℝ) :
  is_decreasing_on_interval a → a ∈ Set.Ici 6 :=
by
  sorry

#check a_range_when_f_decreasing

end NUMINAMATH_CALUDE_a_range_when_f_decreasing_l2387_238782


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2387_238703

theorem decimal_to_fraction :
  (35 : ℚ) / 100 = 7 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2387_238703


namespace NUMINAMATH_CALUDE_triangle_count_difference_l2387_238793

/-- The number of distinct, incongruent, integer-sided triangles with perimeter n -/
def t (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem triangle_count_difference (n : ℕ) (h : n ≥ 3) :
  (t (2 * n - 1) - t (2 * n) = ⌊(6 : ℚ) / n⌋) ∨
  (t (2 * n - 1) - t (2 * n) = ⌊(6 : ℚ) / n⌋ + 1) :=
sorry

end NUMINAMATH_CALUDE_triangle_count_difference_l2387_238793
