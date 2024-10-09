import Mathlib

namespace cost_equation_l878_87869

variables (x y z : ℝ)

theorem cost_equation (h1 : 2 * x + y + 3 * z = 24) (h2 : 3 * x + 4 * y + 2 * z = 36) : x + y + z = 12 := by
  -- proof steps would go here, but are omitted as per instruction
  sorry

end cost_equation_l878_87869


namespace find_angle_A_find_area_l878_87889

noncomputable def triangle_area (a b c : ℝ) (A : ℝ) : ℝ :=
  0.5 * b * c * Real.sin A

theorem find_angle_A (a b c A : ℝ)
  (h1: ∀ x, 4 * Real.cos x * Real.sin (x - π/6) ≤ 4 * Real.cos A * Real.sin (A - π/6))
  (h2: a = b^2 + c^2 - 2 * b * c * Real.cos A) : 
  A = π / 3 := by
  sorry

theorem find_area (a b c : ℝ)
  (A : ℝ) (hA : A = π / 3)
  (ha : a = Real.sqrt 7) (hb : b = 2) 
  : triangle_area a b c A = (3 * Real.sqrt 3) / 2 := by
  sorry

end find_angle_A_find_area_l878_87889


namespace simplify_999_times_neg13_simplify_complex_expr_correct_division_calculation_l878_87894

-- Part 1: Proving the simplified form of arithmetic operations
theorem simplify_999_times_neg13 : 999 * (-13) = -12987 := by
  sorry

theorem simplify_complex_expr :
  999 * (118 + 4 / 5) + 333 * (-3 / 5) - 999 * (18 + 3 / 5) = 99900 := by
  sorry

-- Part 2: Proving the correct calculation of division
theorem correct_division_calculation : 6 / (-1 / 2 + 1 / 3) = -36 := by
  sorry

end simplify_999_times_neg13_simplify_complex_expr_correct_division_calculation_l878_87894


namespace intersection_A_B_intersection_CR_A_B_l878_87867

noncomputable def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
noncomputable def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
noncomputable def CR_A : Set ℝ := {x : ℝ | x < 3} ∪ {x : ℝ | 7 ≤ x}

theorem intersection_A_B :
  A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 7} :=
by
  sorry

theorem intersection_CR_A_B :
  CR_A ∩ B = ({x : ℝ | 2 < x ∧ x < 3} ∪ {x : ℝ | 7 ≤ x ∧ x < 10}) :=
by
  sorry

end intersection_A_B_intersection_CR_A_B_l878_87867


namespace triple_divisor_sum_6_l878_87863

-- Summarize the definition of the divisor sum function excluding the number itself
def divisorSumExcluding (n : ℕ) : ℕ :=
  (Finset.filter (λ x => x ≠ n) (Finset.range (n + 1))).sum id

-- This is the main statement that we need to prove
theorem triple_divisor_sum_6 : divisorSumExcluding (divisorSumExcluding (divisorSumExcluding 6)) = 6 := 
by sorry

end triple_divisor_sum_6_l878_87863


namespace add_neg_two_eq_zero_l878_87866

theorem add_neg_two_eq_zero :
  (-2) + 2 = 0 :=
by
  sorry

end add_neg_two_eq_zero_l878_87866


namespace f_is_odd_l878_87822

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.sqrt (1 + x^2))

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end f_is_odd_l878_87822


namespace emails_received_in_afternoon_l878_87874

theorem emails_received_in_afternoon (A : ℕ) 
  (h1 : 4 + (A - 3) = 9) : 
  A = 8 :=
by
  sorry

end emails_received_in_afternoon_l878_87874


namespace translation_proof_l878_87899

-- Define the points and the translation process
def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (1, 2)
def point_C : ℝ × ℝ := (1, -2)

-- Translation from point A to point C
def translation_vector : ℝ × ℝ :=
  (point_C.1 - point_A.1, point_C.2 - point_A.2)

-- Define point D using the translation vector applied to point B
def point_D : ℝ × ℝ :=
  (point_B.1 + translation_vector.1, point_B.2 + translation_vector.2)

-- Statement to prove point D has the expected coordinates
theorem translation_proof : 
  point_D = (3, 0) :=
by 
  -- The exact proof is omitted, presented here for completion
  sorry

end translation_proof_l878_87899


namespace binom_2023_2_eq_l878_87834

theorem binom_2023_2_eq : Nat.choose 2023 2 = 2045323 := by
  sorry

end binom_2023_2_eq_l878_87834


namespace quadratic_two_distinct_roots_l878_87802

theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k*x^2 - 6*x + 9 = 0) ∧ (k*y^2 - 6*y + 9 = 0)) ↔ (k < 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_two_distinct_roots_l878_87802


namespace smallest_possible_difference_l878_87877

theorem smallest_possible_difference :
  ∃ (x y z : ℕ), 
    x + y + z = 1801 ∧ x < y ∧ y ≤ z ∧ x + y > z ∧ y + z > x ∧ z + x > y ∧ (y - x = 1) := 
by
  sorry

end smallest_possible_difference_l878_87877


namespace subtraction_solution_l878_87838

noncomputable def x : ℝ := 47.806

theorem subtraction_solution :
  (3889 : ℝ) + 12.808 - x = 3854.002 :=
by
  sorry

end subtraction_solution_l878_87838


namespace least_positive_integer_divisors_l878_87812

theorem least_positive_integer_divisors (n m k : ℕ) (h₁ : (∀ d : ℕ, d ∣ n ↔ d ≤ 2023))
(h₂ : n = m * 6^k) (h₃ : (∀ d : ℕ, d ∣ 6 → ¬(d ∣ m))) : m + k = 80 :=
sorry

end least_positive_integer_divisors_l878_87812


namespace boxes_given_to_mom_l878_87872

theorem boxes_given_to_mom 
  (sophie_boxes : ℕ) 
  (donuts_per_box : ℕ) 
  (donuts_to_sister : ℕ) 
  (donuts_left_for_her : ℕ) 
  (H1 : sophie_boxes = 4) 
  (H2 : donuts_per_box = 12) 
  (H3 : donuts_to_sister = 6) 
  (H4 : donuts_left_for_her = 30)
  : sophie_boxes * donuts_per_box - donuts_to_sister - donuts_left_for_her = donuts_per_box := 
by
  sorry

end boxes_given_to_mom_l878_87872


namespace perfect_square_divisors_of_240_l878_87824

theorem perfect_square_divisors_of_240 : 
  (∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, 0 < k ∧ k < n → ¬(k = 1 ∨ k = 4 ∨ k = 16)) := 
sorry

end perfect_square_divisors_of_240_l878_87824


namespace lanies_salary_l878_87882

variables (hours_worked_per_week : ℚ) (hourly_rate : ℚ)

namespace Lanie
def salary (fraction_of_weekly_hours : ℚ) : ℚ :=
  (fraction_of_weekly_hours * hours_worked_per_week) * hourly_rate

theorem lanies_salary : 
  hours_worked_per_week = 40 ∧
  hourly_rate = 15 ∧
  fraction_of_weekly_hours = 4 / 5 →
  salary fraction_of_weekly_hours = 480 :=
by
  -- Proof steps go here
  sorry
end Lanie

end lanies_salary_l878_87882


namespace sum_areas_of_tangent_circles_l878_87870

theorem sum_areas_of_tangent_circles : 
  ∃ r s t : ℝ, 
    (r + s = 6) ∧ 
    (r + t = 8) ∧ 
    (s + t = 10) ∧ 
    (π * (r^2 + s^2 + t^2) = 36 * π) :=
by
  sorry

end sum_areas_of_tangent_circles_l878_87870


namespace swimming_speed_in_still_water_l878_87873

/-- The speed (in km/h) of a man swimming in still water given the speed of the water current
    and the time taken to swim a certain distance against the current. -/
theorem swimming_speed_in_still_water (v : ℝ) (speed_water : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed_water = 12) (h2 : time = 5) (h3 : distance = 40)
  (h4 : time = distance / (v - speed_water)) : v = 20 :=
by
  sorry

end swimming_speed_in_still_water_l878_87873


namespace area_of_each_small_concave_quadrilateral_l878_87803

noncomputable def inner_diameter : ℝ := 8
noncomputable def outer_diameter : ℝ := 10
noncomputable def total_area_covered_by_annuli : ℝ := 112.5
noncomputable def pi : ℝ := 3.14

theorem area_of_each_small_concave_quadrilateral (inner_diameter outer_diameter total_area_covered_by_annuli pi: ℝ)
    (h1 : inner_diameter = 8)
    (h2 : outer_diameter = 10)
    (h3 : total_area_covered_by_annuli = 112.5)
    (h4 : pi = 3.14) :
    (π * (outer_diameter / 2) ^ 2 - π * (inner_diameter / 2) ^ 2) * 5 - total_area_covered_by_annuli / 4 = 7.2 := 
sorry

end area_of_each_small_concave_quadrilateral_l878_87803


namespace money_returned_l878_87815

theorem money_returned (individual group taken : ℝ)
  (h1 : individual = 12000)
  (h2 : group = 16000)
  (h3 : taken = 26400) :
  (individual + group - taken) = 1600 :=
by
  -- The proof has been omitted
  sorry

end money_returned_l878_87815


namespace gianna_saved_for_365_days_l878_87898

-- Define the total amount saved and the amount saved each day
def total_amount_saved : ℕ := 14235
def amount_saved_each_day : ℕ := 39

-- Define the problem statement to prove the number of days saved
theorem gianna_saved_for_365_days :
  (total_amount_saved / amount_saved_each_day) = 365 :=
sorry

end gianna_saved_for_365_days_l878_87898


namespace extraMaterialNeeded_l878_87897

-- Box dimensions
def smallBoxLength (a : ℝ) : ℝ := a
def smallBoxWidth (b : ℝ) : ℝ := 1.5 * b
def smallBoxHeight (c : ℝ) : ℝ := c

def largeBoxLength (a : ℝ) : ℝ := 1.5 * a
def largeBoxWidth (b : ℝ) : ℝ := 2 * b
def largeBoxHeight (c : ℝ) : ℝ := 2 * c

-- Volume calculations
def volumeSmallBox (a b c : ℝ) : ℝ := a * (1.5 * b) * c
def volumeLargeBox (a b c : ℝ) : ℝ := (1.5 * a) * (2 * b) * (2 * c)

-- Surface area calculations
def surfaceAreaSmallBox (a b c : ℝ) : ℝ := 2 * (a * (1.5 * b)) + 2 * (a * c) + 2 * ((1.5 * b) * c)
def surfaceAreaLargeBox (a b c : ℝ) : ℝ := 2 * ((1.5 * a) * (2 * b)) + 2 * ((1.5 * a) * (2 * c)) + 2 * ((2 * b) * (2 * c))

-- Proof statement
theorem extraMaterialNeeded (a b c : ℝ) :
  (volumeSmallBox a b c = 1.5 * a * b * c) ∧ (volumeLargeBox a b c = 6 * a * b * c) ∧ 
  (surfaceAreaLargeBox a b c - surfaceAreaSmallBox a b c = 3 * a * b + 4 * a * c + 5 * b * c) :=
by
  sorry

end extraMaterialNeeded_l878_87897


namespace tan_angle_sum_l878_87808

noncomputable def tan_sum (θ : ℝ) : ℝ := Real.tan (θ + (Real.pi / 4))

theorem tan_angle_sum :
  let x := 1
  let y := 2
  let θ := Real.arctan (y / x)
  tan_sum θ = -3 := by
  sorry

end tan_angle_sum_l878_87808


namespace analytical_expression_of_f_range_of_f_on_interval_l878_87865

noncomputable def f (x : ℝ) (a c : ℝ) : ℝ := a * x^3 + c * x

theorem analytical_expression_of_f
  (a c : ℝ)
  (h1 : a > 0)
  (h2 : ∀ x, f x a c = a * x^3 + c * x) 
  (h3 : 3 * a + c = -6)
  (h4 : ∀ x, (3 * a * x ^ 2 + c) ≥ -12) :
    a = 2 ∧ c = -12 :=
by
  sorry

theorem range_of_f_on_interval
  (h1 : ∃ a c, a = 2 ∧ c = -12)
  (h2 : ∀ x, f x 2 (-12) = 2 * x^3 - 12 * x)
  :
    Set.range (fun x => f x 2 (-12)) = Set.Icc (-8 * Real.sqrt 2) (8 * Real.sqrt 2) :=
by
  sorry

end analytical_expression_of_f_range_of_f_on_interval_l878_87865


namespace find_other_endpoint_l878_87843

def other_endpoint (midpoint endpoint: ℝ × ℝ) : ℝ × ℝ :=
  let (mx, my) := midpoint
  let (ex, ey) := endpoint
  (2 * mx - ex, 2 * my - ey)

theorem find_other_endpoint :
  other_endpoint (3, 1) (7, -4) = (-1, 6) :=
by
  -- Midpoint formula to find other endpoint
  sorry

end find_other_endpoint_l878_87843


namespace inverse_contrapositive_l878_87860

theorem inverse_contrapositive (a b c : ℝ) (h : a > b → a + c > b + c) :
  a + c ≤ b + c → a ≤ b :=
sorry

end inverse_contrapositive_l878_87860


namespace candies_per_child_rounded_l878_87817

/-- There are 15 pieces of candy divided equally among 7 children. The number of candies per child, rounded to the nearest tenth, is 2.1. -/
theorem candies_per_child_rounded :
  let candies := 15
  let children := 7
  Float.round (candies / children * 10) / 10 = 2.1 :=
by
  sorry

end candies_per_child_rounded_l878_87817


namespace inequality_solution_set_l878_87835

theorem inequality_solution_set (x : ℝ) :
  (3 * (x + 2) - x > 4) ∧ ((1 + 2 * x) / 3 ≥ x - 1) ↔ (-1 < x ∧ x ≤ 4) :=
by
  sorry

end inequality_solution_set_l878_87835


namespace complete_square_l878_87891

theorem complete_square {x : ℝ} (h : x^2 + 10 * x - 3 = 0) : (x + 5)^2 = 28 :=
sorry

end complete_square_l878_87891


namespace part1_solution_set_a_eq_1_part2_range_of_values_a_l878_87818

def f (x a : ℝ) : ℝ := |(2 * x - a)| + |(x - 3 * a)|

theorem part1_solution_set_a_eq_1 :
  ∀ x : ℝ, f x 1 ≤ 4 ↔ 0 ≤ x ∧ x ≤ 2 :=
by sorry

theorem part2_range_of_values_a :
  ∀ a : ℝ, (∀ x : ℝ, f x a ≥ |(x - a / 2)| + a^2 + 1) ↔
    ((-2 : ℝ) ≤ a ∧ a ≤ -1 / 2) ∨ (1 / 2 ≤ a ∧ a ≤ 2) :=
by sorry

end part1_solution_set_a_eq_1_part2_range_of_values_a_l878_87818


namespace sum_of_interior_angles_l878_87841

noncomputable def exterior_angle (n : ℕ) := 360 / n

theorem sum_of_interior_angles (n : ℕ) (h : exterior_angle n = 45) :
  180 * (n - 2) = 1080 :=
by
  sorry

end sum_of_interior_angles_l878_87841


namespace max_min_difference_l878_87831

noncomputable def difference_max_min_z (x y z : ℝ) : ℝ :=
  if h₁ : x + y + z = 3 ∧ x^2 + y^2 + z^2 = 18 then 6 else 0

theorem max_min_difference (x y z : ℝ) (h₁ : x + y + z = 3) (h₂ : x^2 + y^2 + z^2 = 18) :
  difference_max_min_z x y z = 6 :=
by sorry

end max_min_difference_l878_87831


namespace least_number_remainder_l878_87878

theorem least_number_remainder (n : ℕ) (h1 : n % 34 = 4) (h2 : n % 5 = 4) : n = 174 :=
  sorry

end least_number_remainder_l878_87878


namespace larger_number_is_42_l878_87816

theorem larger_number_is_42 (x y : ℕ) (h1 : x + y = 77) (h2 : 5 * x = 6 * y) : x = 42 :=
by
  sorry

end larger_number_is_42_l878_87816


namespace quadratic_roots_l878_87819

theorem quadratic_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*m = 0) ∧ (x2^2 + 2*x2 + 2*m = 0)) ↔ m < 1/2 :=
by sorry

end quadratic_roots_l878_87819


namespace nine_digit_number_conditions_l878_87814

def nine_digit_number := 900900000

def remove_second_digit (n : ℕ) : ℕ := n / 100000000 * 10000000 + n % 10000000
def remove_third_digit (n : ℕ) : ℕ := n / 10000000 * 1000000 + n % 1000000
def remove_ninth_digit (n : ℕ) : ℕ := n / 10

theorem nine_digit_number_conditions :
  (remove_second_digit nine_digit_number) % 2 = 0 ∧
  (remove_third_digit nine_digit_number) % 3 = 0 ∧
  (remove_ninth_digit nine_digit_number) % 9 = 0 :=
by
  -- Proof steps would be included here.
  sorry

end nine_digit_number_conditions_l878_87814


namespace time_for_5x5_grid_l878_87893

-- Definitions based on the conditions
def total_length_3x7 : ℕ := 4 * 7 + 8 * 3
def time_for_3x7 : ℕ := 26
def time_per_unit_length : ℚ := time_for_3x7 / total_length_3x7
def total_length_5x5 : ℕ := 6 * 5 + 6 * 5
def expected_time_for_5x5 : ℚ := total_length_5x5 * time_per_unit_length

-- Theorem statement to prove the total time for 5x5 grid
theorem time_for_5x5_grid : expected_time_for_5x5 = 30 := by
  sorry

end time_for_5x5_grid_l878_87893


namespace total_money_made_l878_87821

def num_coffee_customers : ℕ := 7
def price_per_coffee : ℕ := 5
def num_tea_customers : ℕ := 8
def price_per_tea : ℕ := 4

theorem total_money_made (h1 : num_coffee_customers = 7) (h2 : price_per_coffee = 5) 
  (h3 : num_tea_customers = 8) (h4 : price_per_tea = 4) : 
  (num_coffee_customers * price_per_coffee + num_tea_customers * price_per_tea) = 67 :=
by
  sorry

end total_money_made_l878_87821


namespace determine_k_values_parallel_lines_l878_87859

theorem determine_k_values_parallel_lines :
  ∀ k : ℝ, ((k - 3) * x + (4 - k) * y + 1 = 0 ∧ 2 * (k - 3) * x - 2 * y + 3 = 0)
  → k = 2 ∨ k = 3 ∨ k = 6 :=
by
  sorry

end determine_k_values_parallel_lines_l878_87859


namespace solve_equation_l878_87811

theorem solve_equation (x : ℝ) :
  (x + 1)^2 = (2 * x - 1)^2 ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_equation_l878_87811


namespace linear_equation_a_is_the_only_one_l878_87845

-- Definitions for each equation
def equation_a (x y : ℝ) : Prop := x + y = 2
def equation_b (x : ℝ) : Prop := x + 1 = -10
def equation_c (x y : ℝ) : Prop := x - 1/y = 6
def equation_d (x y : ℝ) : Prop := x^2 = 2 * y

-- Proof that equation_a is the only linear equation with two variables
theorem linear_equation_a_is_the_only_one (x y : ℝ) : 
  equation_a x y ∧ ¬equation_b x ∧ ¬(∃ y, equation_c x y) ∧ ¬(∃ y, equation_d x y) :=
by
  sorry

end linear_equation_a_is_the_only_one_l878_87845


namespace three_digit_numbers_satisfying_condition_l878_87851

theorem three_digit_numbers_satisfying_condition :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000) →
    ∃ (a b c : ℕ),
      (N = 100 * a + 10 * b + c) ∧ (N = 11 * (a^2 + b^2 + c^2)) 
    ↔ (N = 550 ∨ N = 803) :=
by
  sorry

end three_digit_numbers_satisfying_condition_l878_87851


namespace cannot_be_written_as_square_l878_87861

theorem cannot_be_written_as_square (A B : ℤ) : 
  99999 + 111111 * Real.sqrt 3 ≠ (A + B * Real.sqrt 3) ^ 2 :=
by
  -- Here we would provide the actual mathematical proof
  sorry

end cannot_be_written_as_square_l878_87861


namespace line_equation_l878_87836

def line_through (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let m := (y₂ - y₁) / (x₂ - x₁)
  y - y₁ = m * (x - x₁)

noncomputable def is_trisection_point (A B QR : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (qx, qy) := QR
  (qx = (2 * x₂ + x₁) / 3 ∧ qy = (2 * y₂ + y₁) / 3) ∨
  (qx = (x₂ + 2 * x₁) / 3 ∧ qy = (y₂ + 2 * y₁) / 3)

theorem line_equation (A B P Q : ℝ × ℝ)
  (hA : A = (3, 4))
  (hB : B = (-4, 5))
  (hP : is_trisection_point B A P)
  (hQ : is_trisection_point B A Q) :
  line_through A P 1 3 ∨ line_through A P 2 1 → 
  (line_through A P 3 4 → P = (1, 3)) ∧ 
  (line_through A P 2 1 → P = (2, 1)) ∧ 
  (line_through A P x y → x - 4 * y + 13 = 0) := 
by 
  sorry

end line_equation_l878_87836


namespace find_first_number_l878_87844

theorem find_first_number (x : ℤ) (k : ℤ) :
  (29 > 0) ∧ (x % 29 = 8) ∧ (1490 % 29 = 11) → x = 29 * k + 8 :=
by
  intros h
  sorry

end find_first_number_l878_87844


namespace x_minus_y_eq_neg3_l878_87826

theorem x_minus_y_eq_neg3 (x y : ℝ) (i : ℂ) (h1 : x * i + 2 = y - i) (h2 : i^2 = -1) : x - y = -3 := 
  sorry

end x_minus_y_eq_neg3_l878_87826


namespace base_for_195₁₀_four_digit_even_final_digit_l878_87810

theorem base_for_195₁₀_four_digit_even_final_digit :
  ∃ b : ℕ, (b^3 ≤ 195 ∧ 195 < b^4) ∧ (∃ d : ℕ, 195 % b = d ∧ d % 2 = 0) ∧ b = 5 :=
by {
  sorry
}

end base_for_195₁₀_four_digit_even_final_digit_l878_87810


namespace total_amount_shared_l878_87855

noncomputable def z : ℝ := 300
noncomputable def y : ℝ := 1.2 * z
noncomputable def x : ℝ := 1.25 * y

theorem total_amount_shared (z y x : ℝ) (hz : z = 300) (hy : y = 1.2 * z) (hx : x = 1.25 * y) :
  x + y + z = 1110 :=
by
  simp [hx, hy, hz]
  -- Add intermediate steps here if necessary
  sorry

end total_amount_shared_l878_87855


namespace simplify_3_375_to_fraction_l878_87896

def simplified_fraction_of_3_375 : ℚ := 3.375

theorem simplify_3_375_to_fraction : simplified_fraction_of_3_375 = 27 / 8 := 
by
  sorry

end simplify_3_375_to_fraction_l878_87896


namespace proof_theorem_l878_87842

noncomputable def proof_problem : Prop :=
  let a := 6
  let b := 15
  let c := 7
  let lhs := a * b * c
  let rhs := (Real.sqrt ((a^2) + (2 * a) + (b^3) - (b^2) + (3 * b))) / (c^2 + c + 1) + 629.001
  lhs = rhs

theorem proof_theorem : proof_problem :=
  by
  sorry

end proof_theorem_l878_87842


namespace greg_age_is_16_l878_87864

-- Define the ages of Cindy, Jan, Marcia, and Greg based on the conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem statement: Prove that Greg's age is 16
theorem greg_age_is_16 : greg_age = 16 :=
by
  -- Proof would go here
  sorry

end greg_age_is_16_l878_87864


namespace find_bags_l878_87887

theorem find_bags (x : ℕ) : 10 + x + 7 = 20 → x = 3 :=
by
  sorry

end find_bags_l878_87887


namespace percent_germinated_is_31_l878_87840

-- Define given conditions
def seeds_first_plot : ℕ := 300
def seeds_second_plot : ℕ := 200
def germination_rate_first_plot : ℝ := 0.25
def germination_rate_second_plot : ℝ := 0.40

-- Calculate the number of germinated seeds in each plot
def germinated_first_plot : ℝ := germination_rate_first_plot * seeds_first_plot
def germinated_second_plot : ℝ := germination_rate_second_plot * seeds_second_plot

-- Calculate total number of seeds and total number of germinated seeds
def total_seeds : ℕ := seeds_first_plot + seeds_second_plot
def total_germinated : ℝ := germinated_first_plot + germinated_second_plot

-- Prove the percentage of the total number of seeds that germinated
theorem percent_germinated_is_31 :
  ((total_germinated / total_seeds) * 100) = 31 := 
by
  sorry

end percent_germinated_is_31_l878_87840


namespace pos_diff_is_multiple_of_9_l878_87890

theorem pos_diff_is_multiple_of_9 
  (q r : ℕ) 
  (h_qr : 10 ≤ q ∧ q < 100 ∧ 10 ≤ r ∧ r < 100 ∧ (q % 10) * 10 + (q / 10) = r)
  (h_max_diff : q - r = 63) : 
  ∃ k : ℕ, q - r = 9 * k :=
by
  sorry

end pos_diff_is_multiple_of_9_l878_87890


namespace james_bought_dirt_bikes_l878_87895

variable (D : ℕ)

-- Definitions derived from conditions
def cost_dirt_bike := 150
def cost_off_road_vehicle := 300
def registration_fee := 25
def num_off_road_vehicles := 4
def total_paid := 1825

-- Auxiliary definitions
def total_cost_dirt_bike := cost_dirt_bike + registration_fee
def total_cost_off_road_vehicle := cost_off_road_vehicle + registration_fee
def total_cost_off_road_vehicles := num_off_road_vehicles * total_cost_off_road_vehicle
def total_cost_dirt_bikes := total_paid - total_cost_off_road_vehicles

-- The final statement we need to prove
theorem james_bought_dirt_bikes : D = total_cost_dirt_bikes / total_cost_dirt_bike ↔ D = 3 := by
  sorry

end james_bought_dirt_bikes_l878_87895


namespace find_uv_non_integer_l878_87858

def p (b : Fin 14 → ℚ) (x y : ℚ) : ℚ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3 + 
  b 10 * x^4 + b 11 * y^4 + b 12 * x^3 * y^2 + b 13 * y^3 * x^2

variables (b : Fin 14 → ℚ)
variables (u v : ℚ)

def zeros_at_specific_points :=
  p b 0 0 = 0 ∧ p b 1 0 = 0 ∧ p b (-1) 0 = 0 ∧
  p b 0 1 = 0 ∧ p b 0 (-1) = 0 ∧ p b 1 1 = 0 ∧
  p b (-1) (-1) = 0 ∧ p b 2 2 = 0 ∧ 
  p b 2 (-2) = 0 ∧ p b (-2) 2 = 0

theorem find_uv_non_integer
  (h : zeros_at_specific_points b) :
  p b (5/19) (16/19) = 0 :=
sorry

end find_uv_non_integer_l878_87858


namespace point_in_first_quadrant_l878_87850

theorem point_in_first_quadrant (x y : ℝ) (hx : x = 6) (hy : y = 2) : x > 0 ∧ y > 0 :=
by
  rw [hx, hy]
  exact ⟨by norm_num, by norm_num⟩

end point_in_first_quadrant_l878_87850


namespace problem_solution_l878_87871

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - 3)

theorem problem_solution : S = (13 / 4) + (3 / 4) * Real.sqrt 13 :=
by sorry

end problem_solution_l878_87871


namespace victoria_gym_sessions_l878_87837

-- Define the initial conditions
def starts_on_monday := true
def sessions_per_two_week_cycle := 6
def total_sessions := 30

-- Define the sought day of the week when all gym sessions are completed
def final_day := "Thursday"

-- The theorem stating the problem
theorem victoria_gym_sessions : 
  starts_on_monday →
  sessions_per_two_week_cycle = 6 →
  total_sessions = 30 →
  final_day = "Thursday" := 
by
  intros
  exact sorry

end victoria_gym_sessions_l878_87837


namespace option_C_correct_l878_87857

theorem option_C_correct (a b : ℝ) : ((a^2 * b)^3) / ((-a * b)^2) = a^4 * b := by
  sorry

end option_C_correct_l878_87857


namespace positive_y_percent_y_eq_16_l878_87809

theorem positive_y_percent_y_eq_16 (y : ℝ) (hy : 0 < y) (h : 0.01 * y * y = 16) : y = 40 :=
by
  sorry

end positive_y_percent_y_eq_16_l878_87809


namespace original_number_l878_87876

theorem original_number (x : ℝ) (h : 20 = 0.4 * (x - 5)) : x = 55 :=
sorry

end original_number_l878_87876


namespace calculate_total_marks_l878_87892

theorem calculate_total_marks 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (marks_per_correct : ℕ) 
  (marks_per_wrong : ℤ) 
  (total_attempted : total_questions = 60) 
  (correct_attempted : correct_answers = 44)
  (marks_per_correct_is_4 : marks_per_correct = 4)
  (marks_per_wrong_is_neg1 : marks_per_wrong = -1) : 
  total_questions * marks_per_correct - (total_questions - correct_answers) * (abs marks_per_wrong) = 160 := 
by 
  sorry

end calculate_total_marks_l878_87892


namespace side_ratio_triangle_square_pentagon_l878_87800

-- Define the conditions
def perimeter_triangle (t : ℝ) := 3 * t = 18
def perimeter_square (s : ℝ) := 4 * s = 16
def perimeter_pentagon (p : ℝ) := 5 * p = 20

-- Statement to be proved
theorem side_ratio_triangle_square_pentagon 
  (t s p : ℝ)
  (ht : perimeter_triangle t)
  (hs : perimeter_square s)
  (hp : perimeter_pentagon p) : 
  (t / s = 3 / 2) ∧ (t / p = 3 / 2) := 
sorry

end side_ratio_triangle_square_pentagon_l878_87800


namespace range_of_m_l878_87888

variable (m : ℝ)

def p : Prop := m + 1 ≤ 0
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (h : ¬ (p m ∧ q m)) : m ≤ -2 ∨ m > -1 := 
by
  sorry

end range_of_m_l878_87888


namespace evaluate_f_g_f_l878_87839

def f (x: ℝ) : ℝ := 5 * x + 4
def g (x: ℝ) : ℝ := 3 * x + 5

theorem evaluate_f_g_f :
  f (g (f 3)) = 314 :=
by
  sorry

end evaluate_f_g_f_l878_87839


namespace find_lowest_temperature_l878_87856

noncomputable def lowest_temperature 
(T1 T2 T3 T4 T5 : ℝ) : ℝ :=
if h : T1 + T2 + T3 + T4 + T5 = 200 ∧ max (max (max T1 T2) (max T3 T4)) T5 - min (min (min T1 T2) (min T3 T4)) T5 = 50 then
   min (min (min T1 T2) (min T3 T4)) T5
else 
  0

theorem find_lowest_temperature (T1 T2 T3 T4 T5 : ℝ) 
  (h_avg : T1 + T2 + T3 + T4 + T5 = 200)
  (h_range : max (max (max T1 T2) (max T3 T4)) T5 - min (min (min T1 T2) (min T3 T4)) T5 ≤ 50) : 
  lowest_temperature T1 T2 T3 T4 T5 = 30 := 
sorry

end find_lowest_temperature_l878_87856


namespace zach_cookies_total_l878_87853

theorem zach_cookies_total :
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  cookies_monday + cookies_tuesday + cookies_wednesday = 92 :=
by
  let cookies_monday := 32
  let cookies_tuesday := cookies_monday / 2
  let cookies_wednesday := cookies_tuesday * 3 - 4
  sorry

end zach_cookies_total_l878_87853


namespace alan_tickets_l878_87884

variables (A M : ℕ)

def condition1 := A + M = 150
def condition2 := M = 5 * A - 6

theorem alan_tickets : A = 26 :=
by
  have h1 : condition1 A M := sorry
  have h2 : condition2 A M := sorry
  sorry

end alan_tickets_l878_87884


namespace problem_statement_l878_87848

noncomputable def alpha : ℝ := 3 + Real.sqrt 8
noncomputable def beta : ℝ := 3 - Real.sqrt 8
noncomputable def x : ℝ := alpha^(500)
noncomputable def N : ℝ := alpha^(500) + beta^(500)
noncomputable def n : ℝ := N - 1
noncomputable def f : ℝ := x - n
noncomputable def one_minus_f : ℝ := 1 - f

theorem problem_statement : x * one_minus_f = 1 :=
by
  -- Insert the proof here
  sorry

end problem_statement_l878_87848


namespace likes_spinach_not_music_lover_l878_87886

universe u

variable (Person : Type u)
variable (likes_spinach is_pearl_diver is_music_lover : Person → Prop)

theorem likes_spinach_not_music_lover :
  (∃ x, likes_spinach x ∧ ¬ is_pearl_diver x) →
  (∀ x, is_music_lover x → (is_pearl_diver x ∨ ¬ likes_spinach x)) →
  (∀ x, (¬ is_pearl_diver x → is_music_lover x) ∨ (is_pearl_diver x → ¬ is_music_lover x)) →
  (∀ x, likes_spinach x → ¬ is_music_lover x) :=
by
  sorry

end likes_spinach_not_music_lover_l878_87886


namespace sam_drove_200_miles_l878_87801

theorem sam_drove_200_miles
  (distance_m: ℝ)
  (time_m: ℝ)
  (distance_s: ℝ)
  (time_s: ℝ)
  (rate_m: ℝ)
  (rate_s: ℝ)
  (h1: distance_m = 150)
  (h2: time_m = 3)
  (h3: rate_m = distance_m / time_m)
  (h4: time_s = 4)
  (h5: rate_s = rate_m)
  (h6: distance_s = rate_s * time_s):
  distance_s = 200 :=
by
  sorry

end sam_drove_200_miles_l878_87801


namespace average_age_of_three_l878_87813

theorem average_age_of_three (Kimiko_age : ℕ) (Omi_age : ℕ) (Arlette_age : ℕ) 
  (h1 : Omi_age = 2 * Kimiko_age) 
  (h2 : Arlette_age = (3 * Kimiko_age) / 4) 
  (h3 : Kimiko_age = 28) : 
  (Kimiko_age + Omi_age + Arlette_age) / 3 = 35 := 
  by sorry

end average_age_of_three_l878_87813


namespace Gina_tip_is_5_percent_l878_87829

noncomputable def Gina_tip_percentage : ℝ := 5

theorem Gina_tip_is_5_percent (bill_amount : ℝ) (good_tipper_percentage : ℝ)
    (good_tipper_extra_tip_cents : ℝ) (good_tipper_tip : ℝ) 
    (Gina_tip_extra_cents : ℝ):
    bill_amount = 26 ∧
    good_tipper_percentage = 20 ∧
    Gina_tip_extra_cents = 390 ∧
    good_tipper_tip = (20 / 100) * 26 ∧
    Gina_tip_extra_cents = 390 ∧
    (Gina_tip_percentage / 100) * bill_amount + (Gina_tip_extra_cents / 100) = good_tipper_tip
    → Gina_tip_percentage = 5 :=
by
  sorry

end Gina_tip_is_5_percent_l878_87829


namespace selling_prices_maximize_profit_l878_87828

-- Definitions for the conditions
def total_items : ℕ := 200
def budget : ℤ := 5000
def cost_basketball : ℤ := 30
def cost_volleyball : ℤ := 24
def selling_price_ratio : ℚ := 3 / 2
def school_purchase_basketballs_value : ℤ := 1800
def school_purchase_volleyballs_value : ℤ := 1500
def basketballs_fewer_than_volleyballs : ℤ := 10

-- Part 1: Proof of selling prices
theorem selling_prices (x : ℚ) :
  (school_purchase_volleyballs_value / x - school_purchase_basketballs_value / (x * selling_price_ratio) = basketballs_fewer_than_volleyballs)
  → ∃ (basketball_price volleyball_price : ℚ), basketball_price = 45 ∧ volleyball_price = 30 :=
by
  sorry

-- Part 2: Proof of maximizing profit
theorem maximize_profit (a : ℕ) :
  (cost_basketball * a + cost_volleyball * (total_items - a) ≤ budget)
  → ∃ optimal_a : ℕ, (optimal_a = 33 ∧ total_items - optimal_a = 167) :=
by
  sorry

end selling_prices_maximize_profit_l878_87828


namespace break_even_price_l878_87852

noncomputable def initial_investment : ℝ := 1500
noncomputable def cost_per_tshirt : ℝ := 3
noncomputable def num_tshirts_break_even : ℝ := 83
noncomputable def total_cost_equipment_tshirts : ℝ := initial_investment + (cost_per_tshirt * num_tshirts_break_even)
noncomputable def price_per_tshirt := total_cost_equipment_tshirts / num_tshirts_break_even

theorem break_even_price : price_per_tshirt = 21.07 := by
  sorry

end break_even_price_l878_87852


namespace simplify_expression_l878_87833

theorem simplify_expression (y : ℝ) : (3 * y + 4 * y + 5 * y + 7) = (12 * y + 7) :=
by
  sorry

end simplify_expression_l878_87833


namespace gcd_8917_4273_l878_87832

theorem gcd_8917_4273 : Int.gcd 8917 4273 = 1 :=
by
  sorry

end gcd_8917_4273_l878_87832


namespace simplify_expression_l878_87827

theorem simplify_expression : (0.4 * 0.5 + 0.3 * 0.2) = 0.26 := by
  sorry

end simplify_expression_l878_87827


namespace calculate_expression_l878_87849

theorem calculate_expression:
  500 * 4020 * 0.0402 * 20 = 1616064000 := by
  sorry

end calculate_expression_l878_87849


namespace common_ratio_l878_87875

theorem common_ratio (a1 a2 a3 : ℚ) (S3 q : ℚ)
  (h1 : a3 = 3 / 2)
  (h2 : S3 = 9 / 2)
  (h3 : a1 + a2 + a3 = S3)
  (h4 : a1 = a3 / q^2)
  (h5 : a2 = a3 / q):
  q = 1 ∨ q = -1/2 :=
by sorry

end common_ratio_l878_87875


namespace stationery_sales_calculation_l878_87804

-- Definitions
def total_sales : ℕ := 120
def fabric_percentage : ℝ := 0.30
def jewelry_percentage : ℝ := 0.20
def knitting_percentage : ℝ := 0.15
def home_decor_percentage : ℝ := 0.10
def stationery_percentage := 1 - (fabric_percentage + jewelry_percentage + knitting_percentage + home_decor_percentage)
def stationery_sales := stationery_percentage * total_sales

-- Statement to prove
theorem stationery_sales_calculation : stationery_sales = 30 := by
  -- Providing the initial values and assumptions to the context
  have h1 : total_sales = 120 := rfl
  have h2 : fabric_percentage = 0.30 := rfl
  have h3 : jewelry_percentage = 0.20 := rfl
  have h4 : knitting_percentage = 0.15 := rfl
  have h5 : home_decor_percentage = 0.10 := rfl
  
  -- Calculating the stationery percentage and sales
  have h_stationery_percentage : stationery_percentage = 1 - (fabric_percentage + jewelry_percentage + knitting_percentage + home_decor_percentage) := rfl
  have h_stationery_sales : stationery_sales = stationery_percentage * total_sales := rfl

  -- The calculated value should match the proof's requirement
  sorry

end stationery_sales_calculation_l878_87804


namespace marcus_goal_points_value_l878_87847

-- Definitions based on conditions
def marcus_goals_first_type := 5
def marcus_goals_second_type := 10
def second_type_goal_points := 2
def team_total_points := 70
def marcus_percentage_points := 50

-- Theorem statement
theorem marcus_goal_points_value : 
  ∃ (x : ℕ), 5 * x + 10 * 2 = 35 ∧ 35 = 50 * team_total_points / 100 := 
sorry

end marcus_goal_points_value_l878_87847


namespace secretary_worked_longest_l878_87868

theorem secretary_worked_longest
  (h1 : ∀ (x : ℕ), 3 * x + 5 * x + 7 * x + 11 * x = 2080)
  (h2 : ∀ (a b c d : ℕ), a = 3 * x ∧ b = 5 * x ∧ c = 7 * x ∧ d = 11 * x → d = 11 * x):
  ∃ y : ℕ, y = 880 :=
by
  sorry

end secretary_worked_longest_l878_87868


namespace correct_fraction_l878_87881

theorem correct_fraction (x y : ℕ) (h1 : 480 * 5 / 6 = 480 * x / y + 250) : x / y = 5 / 16 :=
by
  sorry

end correct_fraction_l878_87881


namespace necessary_but_not_sufficient_condition_for_ellipse_l878_87830

theorem necessary_but_not_sufficient_condition_for_ellipse (m : ℝ) :
  (2 < m ∧ m < 6) ↔ ((∃ m, 2 < m ∧ m < 6 ∧ m ≠ 4) ∧ (∀ m, (2 < m ∧ m < 6) → ¬(m = 4))) := 
sorry

end necessary_but_not_sufficient_condition_for_ellipse_l878_87830


namespace determine_p_l878_87880

-- Define the quadratic equation
def quadratic_eq (p x : ℝ) : ℝ := 3 * x^2 - 5 * (p - 1) * x + (p^2 + 2)

-- Define the conditions for the roots x1 and x2
def conditions (p x1 x2 : ℝ) : Prop :=
  quadratic_eq p x1 = 0 ∧
  quadratic_eq p x2 = 0 ∧
  x1 + 4 * x2 = 14

-- Define the theorem to prove the correct values of p
theorem determine_p (p : ℝ) (x1 x2 : ℝ) :
  conditions p x1 x2 → p = 742 / 127 ∨ p = 4 :=
by
  sorry

end determine_p_l878_87880


namespace larry_daily_dog_time_l878_87879

-- Definitions from the conditions
def half_hour_in_minutes : ℕ := 30
def twice_a_day (minutes : ℕ) : ℕ := 2 * minutes
def one_fifth_hour_in_minutes : ℕ := 60 / 5

-- Hypothesis resulting from the conditions
def time_walking_and_playing : ℕ := twice_a_day half_hour_in_minutes
def time_feeding : ℕ := one_fifth_hour_in_minutes

-- The theorem to prove
theorem larry_daily_dog_time : time_walking_and_playing + time_feeding = 72 := by
  show time_walking_and_playing + time_feeding = 72
  sorry

end larry_daily_dog_time_l878_87879


namespace pipe_a_fills_cistern_l878_87805

theorem pipe_a_fills_cistern :
  ∀ (x : ℝ), (1 / x + 1 / 120 - 1 / 120 = 1 / 60) → x = 60 :=
by
  intro x
  intro h
  sorry

end pipe_a_fills_cistern_l878_87805


namespace roots_square_sum_l878_87825

theorem roots_square_sum (r s t p q : ℝ) 
  (h1 : r + s + t = p) 
  (h2 : r * s + r * t + s * t = q) : 
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
by 
  -- proof skipped
  sorry

end roots_square_sum_l878_87825


namespace isosceles_triangle_largest_angle_l878_87820

theorem isosceles_triangle_largest_angle (a b c : ℝ) (h1 : a = b) (h2 : b_angle = 50) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) 
  (h6 : a + b + c = 180) : c ≥ a ∨ c ≥ b → c = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l878_87820


namespace no_integer_solutions_l878_87806

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), x^4 + y^2 = 6 * y - 3 :=
by
  sorry

end no_integer_solutions_l878_87806


namespace compare_diff_functions_l878_87883

variable {R : Type*} [LinearOrderedField R]
variable {f g : R → R}
variable (h_fg : ∀ x, f' x > g' x)
variable {x1 x2 : R}

theorem compare_diff_functions (h : x1 < x2) : f x1 - f x2 < g x1 - g x2 :=
  sorry

end compare_diff_functions_l878_87883


namespace necessary_condition_l878_87846

theorem necessary_condition (x : ℝ) (h : (x-1) * (x-2) ≤ 0) : x^2 - 3 * x ≤ 0 :=
sorry

end necessary_condition_l878_87846


namespace solve_for_x_l878_87885

noncomputable def x : ℚ := 45^2 / (7 - (3 / 4))

theorem solve_for_x : x = 324 := by
  sorry

end solve_for_x_l878_87885


namespace six_cube_2d_faces_count_l878_87823

open BigOperators

theorem six_cube_2d_faces_count :
    let vertices := 64
    let edges_1d := 192
    let edges_2d := 240
    let small_cubes := 46656
    let faces_per_plane := 36
    let planes_count := 15 * 7^4
    faces_per_plane * planes_count = 1296150 := by
  sorry

end six_cube_2d_faces_count_l878_87823


namespace circle_center_l878_87854

theorem circle_center 
    (x y : ℝ)
    (h : x^2 + y^2 - 4 * x + 6 * y = 0) :
    (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = (x^2 - 4*x + 4) + (y^2 + 6*y + 9) 
    → (x, y) = (2, -3)) :=
sorry

end circle_center_l878_87854


namespace sum_of_first_6033_terms_l878_87807

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := 
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_6033_terms (a r : ℝ) (h1 : geometric_sum a r 2011 = 200) 
  (h2 : geometric_sum a r 4022 = 380) : 
  geometric_sum a r 6033 = 542 :=
sorry

end sum_of_first_6033_terms_l878_87807


namespace math_proof_problem_l878_87862

noncomputable def f (x : ℝ) := Real.log (Real.sin x) * Real.log (Real.cos x)

def domain (k : ℤ) : Set ℝ := { x | 2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi / 2 }

def is_even_shifted : Prop :=
  ∀ x, f (x + Real.pi / 4) = f (- (x + Real.pi / 4))

def has_unique_maximum : Prop :=
  ∃! x, 0 < x ∧ x < Real.pi / 2 ∧ ∀ y, 0 < y ∧ y < Real.pi / 2 → f y ≤ f x

theorem math_proof_problem (k : ℤ) :
  (∀ x, x ∈ domain k → f x ∈ domain k) ∧
  ¬ (∀ x, f (-x) = f x) ∧
  is_even_shifted ∧
  has_unique_maximum :=
by
  sorry

end math_proof_problem_l878_87862
