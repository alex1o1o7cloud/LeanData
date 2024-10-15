import Mathlib

namespace NUMINAMATH_CALUDE_mother_younger_by_two_l3188_318882

/-- A family consisting of a father, mother, brother, sister, and Kaydence. -/
structure Family where
  total_age : ℕ
  father_age : ℕ
  brother_age : ℕ
  sister_age : ℕ
  kaydence_age : ℕ

/-- The age difference between the father and mother in the family. -/
def age_difference (f : Family) : ℕ :=
  f.father_age - (f.total_age - (f.father_age + f.brother_age + f.sister_age + f.kaydence_age))

/-- Theorem stating the age difference between the father and mother is 2 years. -/
theorem mother_younger_by_two (f : Family) 
    (h1 : f.total_age = 200)
    (h2 : f.father_age = 60)
    (h3 : f.brother_age = f.father_age / 2)
    (h4 : f.sister_age = 40)
    (h5 : f.kaydence_age = 12) :
    age_difference f = 2 := by
  sorry

end NUMINAMATH_CALUDE_mother_younger_by_two_l3188_318882


namespace NUMINAMATH_CALUDE_optimal_soap_cost_l3188_318853

/-- Represents the discount percentage based on the number of bars purchased -/
def discount (bars : ℕ) : ℚ :=
  if bars ≥ 8 then 15/100
  else if bars ≥ 6 then 10/100
  else if bars ≥ 4 then 5/100
  else 0

/-- Calculates the cost of soap for a year -/
def soap_cost (price_per_bar : ℚ) (months_per_bar : ℕ) (months_in_year : ℕ) : ℚ :=
  let bars_needed := months_in_year / months_per_bar
  let total_cost := price_per_bar * bars_needed
  total_cost * (1 - discount bars_needed)

theorem optimal_soap_cost :
  soap_cost 8 2 12 = 432/10 :=
sorry

end NUMINAMATH_CALUDE_optimal_soap_cost_l3188_318853


namespace NUMINAMATH_CALUDE_other_number_proof_l3188_318827

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 2310)
  (h2 : Nat.gcd a b = 61)
  (h3 : a = 210) : 
  b = 671 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l3188_318827


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_c_coords_l3188_318817

/-- An isosceles right triangle in 2D space -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isIsosceles : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2
  isRight : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0

/-- Theorem: The coordinates of C in the given isosceles right triangle -/
theorem isosceles_right_triangle_c_coords :
  ∀ t : IsoscelesRightTriangle,
  t.A = (1, 0) → t.B = (3, 1) →
  t.C = (2, 3) ∨ t.C = (4, -1) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_c_coords_l3188_318817


namespace NUMINAMATH_CALUDE_problem_solvers_equal_girls_l3188_318869

/-- Given a class of students, prove that the number of students who solved a problem
    is equal to the total number of girls, given that the number of boys who solved
    the problem is equal to the number of girls who did not solve it. -/
theorem problem_solvers_equal_girls (total : ℕ) (boys girls : ℕ) 
    (boys_solved girls_solved : ℕ) : 
    boys + girls = total →
    boys_solved = girls - girls_solved →
    boys_solved + girls_solved = girls := by
  sorry

end NUMINAMATH_CALUDE_problem_solvers_equal_girls_l3188_318869


namespace NUMINAMATH_CALUDE_tensor_inequality_range_l3188_318860

def tensor (x y : ℝ) : ℝ := x * (1 - y)

theorem tensor_inequality_range (a : ℝ) : 
  (∀ x : ℝ, tensor (x - a) (x + a) < 1) ↔ a ∈ Set.Ioo (-1/2) (3/2) :=
sorry

end NUMINAMATH_CALUDE_tensor_inequality_range_l3188_318860


namespace NUMINAMATH_CALUDE_coin_fraction_missing_l3188_318872

theorem coin_fraction_missing (x : ℚ) (h : x > 0) : 
  let lost := (2 : ℚ) / 3 * x
  let found := (3 : ℚ) / 4 * lost
  (lost - found) / x = (1 : ℚ) / 6 :=
by sorry

end NUMINAMATH_CALUDE_coin_fraction_missing_l3188_318872


namespace NUMINAMATH_CALUDE_set_membership_implies_x_values_l3188_318843

theorem set_membership_implies_x_values (x : ℝ) :
  let A : Set ℝ := {2, 4, x^2 - x}
  6 ∈ A → x = 3 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_x_values_l3188_318843


namespace NUMINAMATH_CALUDE_initial_daily_production_l3188_318801

/-- The number of days the company worked after the initial 3 days -/
def additional_days : ℕ := 20

/-- The total number of parts produced -/
def total_parts : ℕ := 675

/-- The number of extra parts produced beyond the plan -/
def extra_parts : ℕ := 100

/-- The daily increase in parts production after the initial 3 days -/
def daily_increase : ℕ := 5

theorem initial_daily_production :
  ∃ (x : ℕ),
    x > 0 ∧
    3 * x + additional_days * (x + daily_increase) = total_parts + extra_parts ∧
    x = 29 := by
  sorry

end NUMINAMATH_CALUDE_initial_daily_production_l3188_318801


namespace NUMINAMATH_CALUDE_second_quadrant_condition_l3188_318826

/-- Given a complex number z = i(i-a) where a is real, if z corresponds to a point in the second 
    quadrant of the complex plane, then a < 0. -/
theorem second_quadrant_condition (a : ℝ) : 
  let z : ℂ := Complex.I * (Complex.I - a)
  (z.re < 0 ∧ z.im > 0) → a < 0 := by
sorry

end NUMINAMATH_CALUDE_second_quadrant_condition_l3188_318826


namespace NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l3188_318805

/-- 
Given a geometric sequence with positive terms, where:
- a₁ is the first term
- q is the common ratio
- S is the sum of the first 4 terms
- P is the product of the first 4 terms
- M is the sum of the reciprocals of the first 4 terms

Prove that if S = 9 and P = 81/4, then M = 2
-/
theorem geometric_sequence_reciprocal_sum 
  (a₁ q : ℝ) 
  (h_positive : a₁ > 0 ∧ q > 0) 
  (h_sum : a₁ * (1 - q^4) / (1 - q) = 9) 
  (h_product : a₁^4 * q^6 = 81/4) : 
  (1/a₁) * (1 - (1/q)^4) / (1 - 1/q) = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l3188_318805


namespace NUMINAMATH_CALUDE_pr_equals_21_l3188_318873

/-- Triangle PQR with given side lengths -/
structure Triangle where
  PQ : ℝ
  QR : ℝ
  PR : ℕ

/-- The triangle inequality theorem holds for the given triangle -/
def satisfies_triangle_inequality (t : Triangle) : Prop :=
  t.PQ + t.PR > t.QR ∧ t.QR + t.PQ > t.PR ∧ t.PR + t.QR > t.PQ

/-- The theorem stating that PR = 21 satisfies the conditions -/
theorem pr_equals_21 (t : Triangle) 
  (h1 : t.PQ = 7) 
  (h2 : t.QR = 20) 
  (h3 : t.PR = 21) : 
  satisfies_triangle_inequality t :=
sorry

end NUMINAMATH_CALUDE_pr_equals_21_l3188_318873


namespace NUMINAMATH_CALUDE_waiter_tip_problem_l3188_318864

/-- Calculates the tip amount per customer given the total customers, non-tipping customers, and total tips. -/
def tip_per_customer (total_customers : ℕ) (non_tipping_customers : ℕ) (total_tips : ℚ) : ℚ :=
  total_tips / (total_customers - non_tipping_customers)

/-- Proves that given 10 total customers, 5 non-tipping customers, and $15 total tips, 
    the amount each tipping customer gave is $3. -/
theorem waiter_tip_problem :
  tip_per_customer 10 5 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tip_problem_l3188_318864


namespace NUMINAMATH_CALUDE_subtraction_problem_l3188_318844

theorem subtraction_problem : 
  (888.88 : ℝ) - (444.44 : ℝ) = (444.44 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3188_318844


namespace NUMINAMATH_CALUDE_total_selection_methods_is_eight_l3188_318866

/-- The number of students who can only use the synthetic method -/
def synthetic_students : Nat := 5

/-- The number of students who can only use the analytical method -/
def analytical_students : Nat := 3

/-- The total number of ways to select a student to prove the problem -/
def total_selection_methods : Nat := synthetic_students + analytical_students

/-- Theorem stating that the total number of selection methods is 8 -/
theorem total_selection_methods_is_eight : total_selection_methods = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_selection_methods_is_eight_l3188_318866


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3188_318824

theorem decimal_to_fraction : 
  (3.56 : ℚ) = 89 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3188_318824


namespace NUMINAMATH_CALUDE_rotationally_invariant_unique_fixed_point_l3188_318857

/-- A function whose graph remains unchanged after rotation by π/2 around the origin -/
def RotationallyInvariant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y ↔ f (-y) = x

/-- The main theorem stating that a rotationally invariant function
    has exactly one fixed point at the origin -/
theorem rotationally_invariant_unique_fixed_point
  (f : ℝ → ℝ) (h : RotationallyInvariant f) :
  (∃! x : ℝ, f x = x) ∧ (∀ x : ℝ, f x = x → x = 0) :=
by sorry


end NUMINAMATH_CALUDE_rotationally_invariant_unique_fixed_point_l3188_318857


namespace NUMINAMATH_CALUDE_overlapping_area_l3188_318818

theorem overlapping_area (total_length : ℝ) (left_length right_length : ℝ) 
  (left_only_area right_only_area : ℝ) : 
  total_length = 16 →
  left_length = 9 →
  right_length = 7 →
  left_only_area = 27 →
  right_only_area = 18 →
  ∃ (overlap_area : ℝ), 
    overlap_area = 13.5 ∧
    left_length / right_length = (left_only_area + overlap_area) / (right_only_area + overlap_area) :=
by sorry

end NUMINAMATH_CALUDE_overlapping_area_l3188_318818


namespace NUMINAMATH_CALUDE_f_is_linear_l3188_318802

/-- Defines a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function representing the equation 3y + 1 = 6 -/
def f (y : ℝ) : ℝ := 3 * y + 1

/-- Theorem stating that f is a linear equation -/
theorem f_is_linear : is_linear_equation f := by
  sorry

#check f_is_linear

end NUMINAMATH_CALUDE_f_is_linear_l3188_318802


namespace NUMINAMATH_CALUDE_property_holds_iff_one_or_two_l3188_318889

-- Define the property for a given k
def has_property (k : ℕ) : Prop :=
  k ≥ 1 ∧
  ∀ (coloring : ℤ → Fin k),
  ∃ (a : ℕ → ℤ),
    (∀ i < 2023, a i < a (i + 1)) ∧
    (∀ i < 2023, ∃ n : ℕ, a (i + 1) - a i = 2^n) ∧
    (∀ i < 2023, coloring (a i) = coloring (a 0))

-- State the theorem
theorem property_holds_iff_one_or_two :
  ∀ k : ℕ, has_property k ↔ k = 1 ∨ k = 2 := by sorry

end NUMINAMATH_CALUDE_property_holds_iff_one_or_two_l3188_318889


namespace NUMINAMATH_CALUDE_equal_distribution_contribution_l3188_318829

def earnings : List ℕ := [10, 15, 20, 25, 30, 50]

theorem equal_distribution_contribution :
  let total := earnings.sum
  let equal_share := total / earnings.length
  let max_earner := earnings.maximum?
  max_earner.map (λ m => m - equal_share) = some 25 := by sorry

end NUMINAMATH_CALUDE_equal_distribution_contribution_l3188_318829


namespace NUMINAMATH_CALUDE_difference_of_squares_l3188_318807

theorem difference_of_squares : 550^2 - 450^2 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3188_318807


namespace NUMINAMATH_CALUDE_f_composition_of_three_l3188_318852

def f (x : ℝ) : ℝ := -3 * x + 5

theorem f_composition_of_three : f (f (f 3)) = -46 := by sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l3188_318852


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3188_318885

/-- Given vectors a and b in ℝ², prove that k = -1 makes k*a - b perpendicular to a -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (-3, 1)) :
  ∃ k : ℝ, k = -1 ∧ (k • a - b) • a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3188_318885


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3188_318894

theorem smallest_solution_of_equation (x : ℝ) :
  x > 0 ∧ x / 4 + 2 / (3 * x) = 5 / 6 →
  x ≥ 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3188_318894


namespace NUMINAMATH_CALUDE_midpoint_square_area_l3188_318806

/-- Given a square with area 100, prove that a smaller square formed by 
    connecting the midpoints of the sides of the larger square has an area of 25. -/
theorem midpoint_square_area (large_square : Real × Real → Real × Real) 
  (h_area : (large_square (1, 1) - large_square (0, 0)).1 ^ 2 = 100) :
  let small_square := fun (t : Real × Real) => 
    ((large_square (t.1, t.2) + large_square (t.1 + 1, t.2 + 1)) : Real × Real) / 2
  (small_square (1, 1) - small_square (0, 0)).1 ^ 2 = 25 := by
  sorry


end NUMINAMATH_CALUDE_midpoint_square_area_l3188_318806


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3188_318893

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - I) / (3 + 4*I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3188_318893


namespace NUMINAMATH_CALUDE_difference_of_squares_253_247_l3188_318877

theorem difference_of_squares_253_247 : 253^2 - 247^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_253_247_l3188_318877


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l3188_318880

theorem quadratic_root_implies_coefficient (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1 - Complex.I) ^ 2 + a * (1 - Complex.I) + 2 = 0 →
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l3188_318880


namespace NUMINAMATH_CALUDE_job_completion_time_l3188_318825

/-- The number of days it takes for a given number of machines to complete a job -/
def days_to_complete (num_machines : ℕ) : ℝ := sorry

/-- The rate at which each machine works (jobs per day) -/
def machine_rate : ℝ := sorry

theorem job_completion_time :
  -- Five machines working at the same rate
  (days_to_complete 5 * 5 * machine_rate = 1) →
  -- Ten machines can complete the job in 10 days
  (10 * 10 * machine_rate = 1) →
  -- The initial five machines take 20 days to complete the job
  days_to_complete 5 = 20 := by sorry

end NUMINAMATH_CALUDE_job_completion_time_l3188_318825


namespace NUMINAMATH_CALUDE_subtract_fractions_l3188_318836

theorem subtract_fractions : (2 : ℚ) / 3 - 5 / 12 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l3188_318836


namespace NUMINAMATH_CALUDE_secret_spread_day_l3188_318896

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ :=
  (3^(n+1) - 1) / 2

/-- The day of the week, represented as a number from 0 (Sunday) to 6 (Saturday) -/
def day_of_week (n : ℕ) : Fin 7 :=
  n % 7

theorem secret_spread_day : ∃ n : ℕ, secret_spread n ≥ 3280 ∧ day_of_week n = 6 :=
sorry

end NUMINAMATH_CALUDE_secret_spread_day_l3188_318896


namespace NUMINAMATH_CALUDE_farmers_market_spending_l3188_318862

/-- Given Sandi's initial amount and Gillian's total spending, prove that Gillian spent $150 more than three times Sandi's spending. -/
theorem farmers_market_spending (sandi_initial : ℕ) (gillian_total : ℕ)
  (h1 : sandi_initial = 600)
  (h2 : gillian_total = 1050) :
  gillian_total - 3 * (sandi_initial / 2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_farmers_market_spending_l3188_318862


namespace NUMINAMATH_CALUDE_list_length_difference_l3188_318813

/-- 
Given two lists of integers, where the second list contains all elements of the first list 
plus one additional element, prove that the difference in their lengths is 1.
-/
theorem list_length_difference (list1 list2 : List Int) (h : ∀ x, x ∈ list1 → x ∈ list2) 
  (h_additional : ∃ y, y ∈ list2 ∧ y ∉ list1) : 
  list2.length - list1.length = 1 := by
  sorry

end NUMINAMATH_CALUDE_list_length_difference_l3188_318813


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_dormitory_to_city_distance_proof_l3188_318859

theorem dormitory_to_city_distance : ℝ → Prop :=
  fun total_distance =>
    (1/6 : ℝ) * total_distance +
    (1/4 : ℝ) * total_distance +
    (1/3 : ℝ) * total_distance +
    10 +
    (1/12 : ℝ) * total_distance = total_distance →
    total_distance = 60

-- The proof is omitted
theorem dormitory_to_city_distance_proof : dormitory_to_city_distance 60 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_dormitory_to_city_distance_proof_l3188_318859


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l3188_318823

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, |3 * y + 4| ≤ 21 → y ≥ x) → x = -8 ∧ |3 * x + 4| ≤ 21 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l3188_318823


namespace NUMINAMATH_CALUDE_line_parameterization_l3188_318835

/-- Given a line y = 2x - 40 parameterized by (x, y) = (f(t), 20t - 14),
    prove that f(t) = 10t + 13 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 20 * t - 14 = 2 * (f t) - 40) → 
  (∀ t : ℝ, f t = 10 * t + 13) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3188_318835


namespace NUMINAMATH_CALUDE_dividend_calculation_l3188_318810

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 7) :
  divisor * quotient + remainder = 160 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3188_318810


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l3188_318804

def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_23 :
  ∃ (p : Nat), is_prime p ∧ digit_sum p = 23 ∧
  ∀ (q : Nat), is_prime q ∧ digit_sum q = 23 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l3188_318804


namespace NUMINAMATH_CALUDE_bridge_distance_l3188_318842

theorem bridge_distance (a b c : ℝ) (ha : a = 7) (hb : b = 8) (hc : c = 9) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * A)
  let r := A / s
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  let O₁O₂ := Real.sqrt (R^2 + 2 * R * r * cos_C + r^2)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs (O₁O₂ - 5.75) < ε :=
sorry

end NUMINAMATH_CALUDE_bridge_distance_l3188_318842


namespace NUMINAMATH_CALUDE_intersection_line_l3188_318847

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := (x+4)^2 + (y+3)^2 = 8

-- Define the line
def line (x y : ℝ) : Prop := 4*x + 3*y + 13 = 0

-- Theorem statement
theorem intersection_line :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_l3188_318847


namespace NUMINAMATH_CALUDE_coefficient_x2y2_is_70_l3188_318808

/-- The coefficient of x^2 * y^2 in the expansion of (x/√y - y/√x)^8 -/
def coefficient_x2y2 : ℕ :=
  let expression := (fun x y => x / Real.sqrt y - y / Real.sqrt x) ^ 8
  70  -- Placeholder for the actual coefficient

/-- The coefficient of x^2 * y^2 in the expansion of (x/√y - y/√x)^8 is 70 -/
theorem coefficient_x2y2_is_70 : coefficient_x2y2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y2_is_70_l3188_318808


namespace NUMINAMATH_CALUDE_infinite_geometric_series_second_term_l3188_318850

theorem infinite_geometric_series_second_term 
  (r : ℝ) (S : ℝ) (h_r : r = 1/4) (h_S : S = 40) :
  let a := S * (1 - r)
  (a * r) = 15/2 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_second_term_l3188_318850


namespace NUMINAMATH_CALUDE_divisor_count_l3188_318845

def n : ℕ := 2028
def k : ℕ := 2004

theorem divisor_count (h : n = 2^2 * 3^2 * 13^2) : 
  (Finset.filter (fun d => (Finset.filter (fun x => x ∣ d) (Finset.range (n^k + 1))).card = n) 
   (Finset.filter (fun x => x ∣ n^k) (Finset.range (n^k + 1)))).card = 216 := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_l3188_318845


namespace NUMINAMATH_CALUDE_purely_imaginary_m_value_l3188_318809

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_m_value (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - m - 2) (m + 1)
  is_purely_imaginary z → m = 2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_m_value_l3188_318809


namespace NUMINAMATH_CALUDE_population_change_l3188_318883

theorem population_change (P : ℝ) : 
  (P * 1.05 * 0.95 = 9975) → P = 10000 := by
  sorry

end NUMINAMATH_CALUDE_population_change_l3188_318883


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l3188_318822

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ+), S.card = 11 ∧ (∀ x, x ∈ S ↔ ∃ c d : ℕ+, (Nat.gcd c d * Nat.lcm c d = 360) ∧ Nat.gcd c d = x)) :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l3188_318822


namespace NUMINAMATH_CALUDE_equation_solution_l3188_318863

theorem equation_solution : 
  let f (x : ℂ) := (4 * x^3 + 4 * x^2 + 3 * x + 2) / (x - 2)
  let g (x : ℂ) := 4 * x^2 + 5 * x + 4
  let sol₁ : ℂ := (-9 + Complex.I * Real.sqrt 79) / 8
  let sol₂ : ℂ := (-9 - Complex.I * Real.sqrt 79) / 8
  (∀ x : ℂ, x ≠ 2 → f x = g x) → (f sol₁ = g sol₁ ∧ f sol₂ = g sol₂) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3188_318863


namespace NUMINAMATH_CALUDE_milk_cost_l3188_318839

theorem milk_cost (banana_cost : ℝ) (tax_rate : ℝ) (total_spent : ℝ) 
  (h1 : banana_cost = 2)
  (h2 : tax_rate = 0.2)
  (h3 : total_spent = 6) :
  ∃ milk_cost : ℝ, milk_cost = 3 ∧ 
    total_spent = (milk_cost + banana_cost) * (1 + tax_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_milk_cost_l3188_318839


namespace NUMINAMATH_CALUDE_sqrt_equality_condition_l3188_318861

theorem sqrt_equality_condition (x : ℝ) : 
  Real.sqrt ((x + 1)^2 + (x - 1)^2) = (x + 1) - (x - 1) ↔ x = 1 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equality_condition_l3188_318861


namespace NUMINAMATH_CALUDE_square_root_problem_l3188_318831

theorem square_root_problem (x : ℝ) : (3/5) * x^2 = 126.15 → x = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l3188_318831


namespace NUMINAMATH_CALUDE_investment_interest_calculation_l3188_318849

/-- Calculates the total interest earned from an investment split between two interest rates -/
def total_interest (total_investment : ℝ) (rate1 rate2 : ℝ) (amount_at_rate2 : ℝ) : ℝ :=
  let amount_at_rate1 := total_investment - amount_at_rate2
  let interest1 := amount_at_rate1 * rate1
  let interest2 := amount_at_rate2 * rate2
  interest1 + interest2

/-- Theorem stating that the total interest is $660 given the specified conditions -/
theorem investment_interest_calculation :
  total_interest 18000 0.03 0.05 6000 = 660 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_calculation_l3188_318849


namespace NUMINAMATH_CALUDE_joan_gave_two_balloons_l3188_318874

/-- The number of blue balloons Joan gave to Jessica --/
def balloons_given_to_jessica (initial : ℕ) (received : ℕ) (final : ℕ) : ℕ :=
  initial + received - final

/-- Proof that Joan gave 2 balloons to Jessica --/
theorem joan_gave_two_balloons : 
  balloons_given_to_jessica 9 5 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_joan_gave_two_balloons_l3188_318874


namespace NUMINAMATH_CALUDE_classroom_students_l3188_318838

theorem classroom_students (n : ℕ) : 
  n < 50 ∧ n % 6 = 4 ∧ n % 4 = 2 ↔ n ∈ ({10, 22, 34} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_classroom_students_l3188_318838


namespace NUMINAMATH_CALUDE_unique_number_between_cubes_l3188_318855

theorem unique_number_between_cubes : ∃! (n : ℕ), 
  n > 0 ∧ 
  24 ∣ n ∧ 
  (9 : ℝ) < n^(1/3) ∧ 
  n^(1/3) < (9.1 : ℝ) ∧ 
  n = 744 := by sorry

end NUMINAMATH_CALUDE_unique_number_between_cubes_l3188_318855


namespace NUMINAMATH_CALUDE_final_number_is_88_or_94_l3188_318803

/-- Represents the two allowed operations on the number -/
inductive Operation
| replace_with_diff
| increase_decrease

/-- The initial number with 98 eights -/
def initial_number : Nat := 88888888  -- Simplified representation

/-- Applies a single operation to a number -/
def apply_operation (n : Nat) (op : Operation) : Nat :=
  match op with
  | Operation.replace_with_diff => sorry
  | Operation.increase_decrease => sorry

/-- Applies a sequence of operations to a number -/
def apply_operations (n : Nat) (ops : List Operation) : Nat :=
  match ops with
  | [] => n
  | op :: rest => apply_operations (apply_operation n op) rest

/-- The theorem stating that the final two-digit number must be 88 or 94 -/
theorem final_number_is_88_or_94 (ops : List Operation) :
  ∃ (result : Nat), apply_operations initial_number ops = result ∧ (result = 88 ∨ result = 94) :=
sorry

end NUMINAMATH_CALUDE_final_number_is_88_or_94_l3188_318803


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l3188_318833

theorem last_two_digits_sum (n : ℕ) : n = 23 →
  (7^n + 13^n) % 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l3188_318833


namespace NUMINAMATH_CALUDE_total_watermelon_seeds_l3188_318881

theorem total_watermelon_seeds (bom gwi yeon : ℕ) : 
  bom = 300 →
  gwi = bom + 40 →
  yeon = 3 * gwi →
  bom + gwi + yeon = 1660 := by
sorry

end NUMINAMATH_CALUDE_total_watermelon_seeds_l3188_318881


namespace NUMINAMATH_CALUDE_arithmetic_sequence_s_value_l3188_318812

/-- An arithmetic sequence with 7 terms -/
structure ArithmeticSequence :=
  (a : Fin 7 → ℚ)
  (is_arithmetic : ∀ i j k : Fin 7, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
                   a j - a i = a k - a j)

/-- The theorem statement -/
theorem arithmetic_sequence_s_value 
  (seq : ArithmeticSequence)
  (first_term : seq.a 0 = 20)
  (last_term : seq.a 6 = 40)
  (second_to_last : seq.a 5 = seq.a 4 + 10) :
  seq.a 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_s_value_l3188_318812


namespace NUMINAMATH_CALUDE_quadratic_value_at_4_l3188_318814

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_value_at_4 
  (a b c : ℝ) 
  (h_max : ∃ (k : ℝ), quadratic a b c k = 5 ∧ ∀ x, quadratic a b c x ≤ 5)
  (h_max_at_3 : quadratic a b c 3 = 5)
  (h_at_0 : quadratic a b c 0 = -13) :
  quadratic a b c 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_at_4_l3188_318814


namespace NUMINAMATH_CALUDE_pentagon_divisible_hexagon_divisible_heptagon_not_divisible_l3188_318887

/-- A polygon is a closed shape with a certain number of sides and vertices. -/
structure Polygon where
  sides : ℕ
  vertices : ℕ

/-- A triangle is a polygon with 3 sides and 3 vertices. -/
def Triangle : Polygon := ⟨3, 3⟩

/-- A pentagon is a polygon with 5 sides and 5 vertices. -/
def Pentagon : Polygon := ⟨5, 5⟩

/-- A hexagon is a polygon with 6 sides and 6 vertices. -/
def Hexagon : Polygon := ⟨6, 6⟩

/-- A heptagon is a polygon with 7 sides and 7 vertices. -/
def Heptagon : Polygon := ⟨7, 7⟩

/-- A polygon can be divided into two triangles if there exists a way to combine two triangles to form that polygon. -/
def CanBeDividedIntoTwoTriangles (p : Polygon) : Prop :=
  ∃ (t1 t2 : Polygon), t1 = Triangle ∧ t2 = Triangle ∧ p.sides = t1.sides + t2.sides - 2 ∧ p.vertices = t1.vertices + t2.vertices - 2

theorem pentagon_divisible : CanBeDividedIntoTwoTriangles Pentagon := by sorry

theorem hexagon_divisible : CanBeDividedIntoTwoTriangles Hexagon := by sorry

theorem heptagon_not_divisible : ¬CanBeDividedIntoTwoTriangles Heptagon := by sorry

end NUMINAMATH_CALUDE_pentagon_divisible_hexagon_divisible_heptagon_not_divisible_l3188_318887


namespace NUMINAMATH_CALUDE_percent_of_a_is_4b_l3188_318819

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.8 * b) : 
  (4 * b) / a * 100 = 222.22222222222223 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_is_4b_l3188_318819


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l3188_318898

/-- Jessie's weight loss problem -/
theorem jessie_weight_loss (current_weight lost_weight : ℕ) 
  (h1 : current_weight = 34)
  (h2 : lost_weight = 35) : 
  current_weight + lost_weight = 69 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l3188_318898


namespace NUMINAMATH_CALUDE_all_ones_satisfy_l3188_318828

def satisfies_inequalities (a : Fin 100 → ℝ) : Prop :=
  ∀ i : Fin 100, a i - 4 * a (i.succ) + 3 * a (i.succ.succ) ≥ 0

theorem all_ones_satisfy (a : Fin 100 → ℝ) 
  (h : satisfies_inequalities a) (h1 : a 0 = 1) : 
  ∀ i : Fin 100, a i = 1 :=
sorry

end NUMINAMATH_CALUDE_all_ones_satisfy_l3188_318828


namespace NUMINAMATH_CALUDE_wax_left_after_detailing_l3188_318878

/-- The amount of wax needed to detail Kellan's car in ounces. -/
def car_wax : ℕ := 3

/-- The amount of wax needed to detail Kellan's SUV in ounces. -/
def suv_wax : ℕ := 4

/-- The amount of wax in the bottle Kellan bought in ounces. -/
def bought_wax : ℕ := 11

/-- The amount of wax Kellan spilled in ounces. -/
def spilled_wax : ℕ := 2

/-- The theorem states that given the above conditions, 
    the amount of wax Kellan has left after waxing his car and SUV is 2 ounces. -/
theorem wax_left_after_detailing : 
  bought_wax - spilled_wax - (car_wax + suv_wax) = 2 := by
  sorry

end NUMINAMATH_CALUDE_wax_left_after_detailing_l3188_318878


namespace NUMINAMATH_CALUDE_interest_problem_l3188_318834

/-- Given a principal amount and an interest rate, prove that they satisfy the conditions for simple and compound interest over 2 years -/
theorem interest_problem (P R : ℝ) : 
  (P * R * 2 / 100 = 20) →  -- Simple interest condition
  (P * ((1 + R/100)^2 - 1) = 22) →  -- Compound interest condition
  (P = 50 ∧ R = 20) := by
sorry

end NUMINAMATH_CALUDE_interest_problem_l3188_318834


namespace NUMINAMATH_CALUDE_fraction_inverse_addition_l3188_318886

theorem fraction_inverse_addition (a b : ℚ) (h : a ≠ b) :
  let c := -(a + b)
  (a + c) / (b + c) = b / a :=
by sorry

end NUMINAMATH_CALUDE_fraction_inverse_addition_l3188_318886


namespace NUMINAMATH_CALUDE_parabola_transformation_l3188_318832

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := -x^2

/-- The transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := -(x - 2)^2 - 3

/-- Theorem stating that the transformed parabola is equivalent to
    shifting the original parabola 2 units right and 3 units down -/
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 2) - 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l3188_318832


namespace NUMINAMATH_CALUDE_jacks_sock_purchase_l3188_318884

/-- Represents the number of pairs of socks at each price point --/
structure SockPurchase where
  two_dollar : ℕ
  three_dollar : ℕ
  four_dollar : ℕ

/-- Checks if the given SockPurchase satisfies all conditions --/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.two_dollar + p.three_dollar + p.four_dollar = 15 ∧
  2 * p.two_dollar + 3 * p.three_dollar + 4 * p.four_dollar = 36 ∧
  p.two_dollar ≥ 1 ∧ p.three_dollar ≥ 1 ∧ p.four_dollar ≥ 1

/-- Theorem stating that the only valid purchase has 10 pairs of $2 socks --/
theorem jacks_sock_purchase :
  ∀ p : SockPurchase, is_valid_purchase p → p.two_dollar = 10 := by
  sorry

end NUMINAMATH_CALUDE_jacks_sock_purchase_l3188_318884


namespace NUMINAMATH_CALUDE_bianca_deleted_pictures_l3188_318840

theorem bianca_deleted_pictures (total_files songs text_files : ℕ) 
  (h1 : total_files = 17)
  (h2 : songs = 8)
  (h3 : text_files = 7)
  : total_files = songs + text_files + 2 := by
  sorry

end NUMINAMATH_CALUDE_bianca_deleted_pictures_l3188_318840


namespace NUMINAMATH_CALUDE_ratio_expression_value_l3188_318848

theorem ratio_expression_value (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by sorry

end NUMINAMATH_CALUDE_ratio_expression_value_l3188_318848


namespace NUMINAMATH_CALUDE_stamps_per_binder_l3188_318888

-- Define the number of notebooks and stamps per notebook
def num_notebooks : ℕ := 4
def stamps_per_notebook : ℕ := 20

-- Define the number of binders
def num_binders : ℕ := 2

-- Define the fraction of stamps kept
def fraction_kept : ℚ := 1/4

-- Define the number of stamps given away
def stamps_given_away : ℕ := 135

-- Theorem to prove
theorem stamps_per_binder :
  ∃ (x : ℕ), 
    (3/4 : ℚ) * (num_notebooks * stamps_per_notebook + num_binders * x) = stamps_given_away ∧
    x = 50 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_binder_l3188_318888


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_3_m_value_when_intersection_given_l3188_318867

-- Define sets A and B
def A : Set ℝ := {x | x > 1}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Part 1
theorem intersection_complement_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x < 5} :=
sorry

-- Part 2
theorem m_value_when_intersection_given :
  A ∩ B m = {x | -1 < x ∧ x < 4} → m = 8 :=
sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_3_m_value_when_intersection_given_l3188_318867


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3188_318858

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane 
  (m : Line) (α β : Plane) 
  (h1 : contained_in m α) 
  (h2 : parallel α β) : 
  line_parallel_to_plane m β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3188_318858


namespace NUMINAMATH_CALUDE_hexadecagon_diagonals_l3188_318811

/-- Number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexadecagon is a 16-sided polygon -/
def hexadecagon_sides : ℕ := 16

theorem hexadecagon_diagonals :
  num_diagonals hexadecagon_sides = 104 := by sorry

end NUMINAMATH_CALUDE_hexadecagon_diagonals_l3188_318811


namespace NUMINAMATH_CALUDE_moremom_arrangements_count_l3188_318879

/-- The number of unique arrangements of letters in MOREMOM -/
def moremom_arrangements : ℕ := 420

/-- The total number of letters in MOREMOM -/
def total_letters : ℕ := 7

/-- The number of M's in MOREMOM -/
def m_count : ℕ := 3

/-- The number of O's in MOREMOM -/
def o_count : ℕ := 2

/-- Theorem stating that the number of unique arrangements of letters in MOREMOM is 420 -/
theorem moremom_arrangements_count :
  moremom_arrangements = Nat.factorial total_letters /(Nat.factorial m_count * Nat.factorial o_count) :=
by sorry

end NUMINAMATH_CALUDE_moremom_arrangements_count_l3188_318879


namespace NUMINAMATH_CALUDE_first_product_of_98_l3188_318854

/-- The first product of the digits of a two-digit number -/
def first_digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- Theorem: The first product of the digits of 98 is 72 -/
theorem first_product_of_98 : first_digit_product 98 = 72 := by
  sorry

end NUMINAMATH_CALUDE_first_product_of_98_l3188_318854


namespace NUMINAMATH_CALUDE_emily_walks_farther_l3188_318876

def troy_base_distance : ℕ := 75
def emily_base_distance : ℕ := 98

def troy_detours : List ℕ := [15, 20, 10, 0, 5]
def emily_detours : List ℕ := [10, 25, 10, 15, 10]

def total_distance (base : ℕ) (detours : List ℕ) : ℕ :=
  (base * 10 + detours.sum) * 2

theorem emily_walks_farther :
  total_distance emily_base_distance emily_detours -
  total_distance troy_base_distance troy_detours = 270 := by
  sorry

end NUMINAMATH_CALUDE_emily_walks_farther_l3188_318876


namespace NUMINAMATH_CALUDE_min_value_theorem_l3188_318821

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (8^a * 2^b)) : 
  ∃ (min_val : ℝ), min_val = 5 + 2 * Real.sqrt 3 ∧ 
    ∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt 2 = Real.sqrt (8^x * 2^y) → 
      1/x + 2/y ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3188_318821


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l3188_318890

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  total_members : Nat
  captain_age : Nat
  wicket_keeper_age : Nat
  team_average_age : Nat

/-- The difference between the average age of remaining players and the whole team -/
def age_difference (team : CricketTeam) : Rat :=
  let remaining_members := team.total_members - 2
  let total_age := team.team_average_age * team.total_members
  let remaining_age := total_age - team.captain_age - team.wicket_keeper_age
  let remaining_average := remaining_age / remaining_members
  team.team_average_age - remaining_average

/-- Theorem stating the age difference for a specific cricket team -/
theorem cricket_team_age_difference :
  ∃ (team : CricketTeam),
    team.total_members = 11 ∧
    team.captain_age = 26 ∧
    team.wicket_keeper_age = team.captain_age + 5 ∧
    team.team_average_age = 24 ∧
    age_difference team = 1 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l3188_318890


namespace NUMINAMATH_CALUDE_no_integer_solution_l3188_318875

theorem no_integer_solution : ¬ ∃ (a k : ℤ), 2 * a^2 - 7 * k + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3188_318875


namespace NUMINAMATH_CALUDE_derivative_of_f_l3188_318897

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x^2 - x + 1)

-- State the theorem
theorem derivative_of_f : 
  deriv f = λ x => 3 * x^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3188_318897


namespace NUMINAMATH_CALUDE_max_value_theorem_l3188_318846

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^2 + b^2 - Real.sqrt 3 * a * b = 1) : 
  Real.sqrt 3 * a^2 - a * b ≤ 2 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3188_318846


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3188_318871

theorem binomial_coefficient_equality : Nat.choose 10 8 = Nat.choose 10 2 ∧ Nat.choose 10 8 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3188_318871


namespace NUMINAMATH_CALUDE_women_no_traits_l3188_318891

/-- Represents the number of women in the population -/
def total_population : ℕ := 200

/-- Probability of having only one specific trait -/
def prob_one_trait : ℚ := 1/20

/-- Probability of having precisely two specific traits -/
def prob_two_traits : ℚ := 2/25

/-- Probability of having all three traits, given a woman has X and Y -/
def prob_all_given_xy : ℚ := 1/4

/-- Number of women with only one trait -/
def women_one_trait : ℕ := 10

/-- Number of women with exactly two traits -/
def women_two_traits : ℕ := 16

/-- Number of women with all three traits -/
def women_all_traits : ℕ := 5

/-- Theorem stating the number of women with none of the three traits -/
theorem women_no_traits : 
  total_population - 3 * women_one_trait - 3 * women_two_traits - women_all_traits = 117 := by
  sorry

end NUMINAMATH_CALUDE_women_no_traits_l3188_318891


namespace NUMINAMATH_CALUDE_intersection_product_range_l3188_318837

/-- Sphere S centered at origin with radius √6 -/
def S (x y z : ℝ) : Prop := x^2 + y^2 + z^2 = 6

/-- Plane α passing through (4, 0, 0), (0, 4, 0), (0, 0, 4) -/
def α (x y z : ℝ) : Prop := x + y + z = 4

theorem intersection_product_range :
  ∀ x y z : ℝ, S x y z → α x y z → 50/27 ≤ x*y*z ∧ x*y*z ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_product_range_l3188_318837


namespace NUMINAMATH_CALUDE_boat_current_rate_l3188_318820

/-- Proves that given a boat with a speed of 20 km/hr in still water,
    traveling 10 km downstream in 24 minutes, the rate of the current is 5 km/hr. -/
theorem boat_current_rate :
  let boat_speed : ℝ := 20 -- km/hr
  let downstream_distance : ℝ := 10 -- km
  let downstream_time : ℝ := 24 / 60 -- hr (24 minutes converted to hours)
  ∃ current_rate : ℝ,
    (boat_speed + current_rate) * downstream_time = downstream_distance ∧
    current_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_boat_current_rate_l3188_318820


namespace NUMINAMATH_CALUDE_x_value_when_s_reaches_15000_l3188_318868

/-- The function that calculates S for a given n -/
def S (n : ℕ) : ℕ := n * (n + 3)

/-- The function that calculates X for a given n -/
def X (n : ℕ) : ℕ := 4 + 2 * (n - 1)

/-- The theorem to prove -/
theorem x_value_when_s_reaches_15000 :
  ∃ n : ℕ, S n ≥ 15000 ∧ ∀ m : ℕ, m < n → S m < 15000 ∧ X n = 244 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_s_reaches_15000_l3188_318868


namespace NUMINAMATH_CALUDE_smallest_multiple_l3188_318870

theorem smallest_multiple (x : ℕ) : x = 432 ↔ 
  (x > 0 ∧ 500 * x % 864 = 0 ∧ ∀ y : ℕ, y > 0 → 500 * y % 864 = 0 → x ≤ y) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3188_318870


namespace NUMINAMATH_CALUDE_quadratic_negative_root_range_l3188_318830

/-- Given a quadratic function f(x) = (m-2)x^2 - 4mx + 2m - 6,
    this theorem states the range of m for which f(x) has at least one negative root. -/
theorem quadratic_negative_root_range (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (m - 2) * x^2 - 4 * m * x + 2 * m - 6 = 0) ↔
  (1 ≤ m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_range_l3188_318830


namespace NUMINAMATH_CALUDE_gcd_105_90_l3188_318895

theorem gcd_105_90 : Nat.gcd 105 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_90_l3188_318895


namespace NUMINAMATH_CALUDE_symmetric_sine_cosine_value_l3188_318800

/-- Given a function f(x) = 3sin(ωx + φ) that is symmetric about x = π/3,
    prove that g(π/3) = 1 where g(x) = 3cos(ωx + φ) + 1 -/
theorem symmetric_sine_cosine_value 
  (ω φ : ℝ) 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (hf : f = fun x ↦ 3 * Real.sin (ω * x + φ))
  (hg : g = fun x ↦ 3 * Real.cos (ω * x + φ) + 1)
  (h_sym : ∀ x : ℝ, f (π / 3 + x) = f (π / 3 - x)) : 
  g (π / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sine_cosine_value_l3188_318800


namespace NUMINAMATH_CALUDE_yard_area_l3188_318856

/-- The area of a rectangular yard with a square cut-out -/
theorem yard_area (length width cut_side : ℝ) (h1 : length = 20) (h2 : width = 18) (h3 : cut_side = 4) :
  length * width - cut_side * cut_side = 344 :=
by sorry

end NUMINAMATH_CALUDE_yard_area_l3188_318856


namespace NUMINAMATH_CALUDE_haley_has_35_marbles_l3188_318841

/-- The number of marbles Haley has, given the number of boys and marbles per boy -/
def haley_marbles (num_boys : ℕ) (marbles_per_boy : ℕ) : ℕ :=
  num_boys * marbles_per_boy

/-- Theorem stating that Haley has 35 marbles -/
theorem haley_has_35_marbles :
  haley_marbles 5 7 = 35 := by
  sorry

end NUMINAMATH_CALUDE_haley_has_35_marbles_l3188_318841


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l3188_318899

-- Define the operation ⊗ (using ⊗ instead of ⭐ as it's more readily available)
noncomputable def bowtie (x y : ℝ) : ℝ := x + Real.sqrt (y + Real.sqrt (y + Real.sqrt y))

-- State the theorem
theorem bowtie_equation_solution (h : ℝ) : bowtie 3 h = 5 → h = 2 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l3188_318899


namespace NUMINAMATH_CALUDE_hundredth_row_sum_l3188_318815

def triangular_array_sum (n : ℕ) : ℕ :=
  2^(n+1) - 4

theorem hundredth_row_sum : 
  triangular_array_sum 100 = 2^101 - 4 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_row_sum_l3188_318815


namespace NUMINAMATH_CALUDE_storm_rain_difference_l3188_318851

/-- Amount of rain in the first hour -/
def first_hour_rain : ℝ := 5

/-- Total amount of rain in the first two hours -/
def total_rain : ℝ := 22

/-- Amount of rain in the second hour -/
def second_hour_rain : ℝ := total_rain - first_hour_rain

/-- The difference between the amount of rain in the second hour and twice the amount of rain in the first hour -/
def rain_difference : ℝ := second_hour_rain - 2 * first_hour_rain

theorem storm_rain_difference : rain_difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_storm_rain_difference_l3188_318851


namespace NUMINAMATH_CALUDE_base_height_ratio_l3188_318865

/-- Represents a triangular field with specific properties -/
structure TriangularField where
  base : ℝ
  height : ℝ
  cultivation_cost : ℝ
  cost_per_hectare : ℝ
  base_multiple_of_height : ∃ k : ℝ, base = k * height
  total_cost : cultivation_cost = 333.18
  cost_rate : cost_per_hectare = 24.68
  base_value : base = 300
  height_value : height = 300

/-- Theorem stating that the ratio of base to height is 1:1 for the given triangular field -/
theorem base_height_ratio (field : TriangularField) : field.base / field.height = 1 := by
  sorry

#check base_height_ratio

end NUMINAMATH_CALUDE_base_height_ratio_l3188_318865


namespace NUMINAMATH_CALUDE_simplest_radical_form_among_options_l3188_318892

def is_simplest_radical_form (x : ℝ) : Prop :=
  ∃ n : ℕ, x = Real.sqrt n ∧ 
  (∀ m : ℕ, m ^ 2 ∣ n → m = 1) ∧
  (∀ a b : ℕ, n ≠ a / b)

theorem simplest_radical_form_among_options : 
  is_simplest_radical_form (Real.sqrt 10) ∧
  ¬is_simplest_radical_form (Real.sqrt 9) ∧
  ¬is_simplest_radical_form (Real.sqrt 20) ∧
  ¬is_simplest_radical_form (Real.sqrt (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_simplest_radical_form_among_options_l3188_318892


namespace NUMINAMATH_CALUDE_triangle_area_solutions_l3188_318816

theorem triangle_area_solutions : 
  let vertex_A : ℝ × ℝ := (-5, 0)
  let vertex_B : ℝ × ℝ := (5, 0)
  let vertex_C (θ : ℝ) : ℝ × ℝ := (5 * Real.cos θ, 5 * Real.sin θ)
  let triangle_area (θ : ℝ) : ℝ := 
    abs ((vertex_B.1 - vertex_A.1) * (vertex_C θ).2 - (vertex_B.2 - vertex_A.2) * (vertex_C θ).1 - 
         (vertex_A.1 * vertex_B.2 - vertex_A.2 * vertex_B.1)) / 2
  ∃! (solutions : Finset ℝ), 
    (∀ θ ∈ solutions, 0 ≤ θ ∧ θ < 2 * Real.pi) ∧ 
    (∀ θ ∈ solutions, triangle_area θ = 10) ∧ 
    solutions.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_solutions_l3188_318816
