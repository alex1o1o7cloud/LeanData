import Mathlib

namespace NUMINAMATH_GPT_A_number_is_35_l356_35612

theorem A_number_is_35 (A B : ℕ) 
  (h_sum_digits : A + B = 8) 
  (h_diff_numbers : 10 * B + A = 10 * A + B + 18) :
  10 * A + B = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_A_number_is_35_l356_35612


namespace NUMINAMATH_GPT_road_signs_at_first_intersection_l356_35640

theorem road_signs_at_first_intersection (x : ℕ) 
    (h1 : x + (x + x / 4) + 2 * (x + x / 4) + (2 * (x + x / 4) - 20) = 270) : 
    x = 40 := 
sorry

end NUMINAMATH_GPT_road_signs_at_first_intersection_l356_35640


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_l356_35682

-- Defining the given repeating decimals as fractions
def rep_decimal1 : ℚ := 2 / 9
def rep_decimal2 : ℚ := 2 / 99
def rep_decimal3 : ℚ := 2 / 9999

-- Stating the theorem to prove the given sum equals the correct answer
theorem sum_of_repeating_decimals :
  rep_decimal1 + rep_decimal2 + rep_decimal3 = 224422 / 9999 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_repeating_decimals_l356_35682


namespace NUMINAMATH_GPT_selling_price_of_cycle_l356_35651

theorem selling_price_of_cycle (cost_price : ℕ) (gain_percent : ℕ) (cost_price_eq : cost_price = 1500) (gain_percent_eq : gain_percent = 8) :
  ∃ selling_price : ℕ, selling_price = 1620 := 
by
  sorry

end NUMINAMATH_GPT_selling_price_of_cycle_l356_35651


namespace NUMINAMATH_GPT_cost_of_berries_and_cheese_l356_35636

variables (b m l c : ℕ)

theorem cost_of_berries_and_cheese (h1 : b + m + l + c = 25)
                                  (h2 : m = 2 * l)
                                  (h3 : c = b + 2) : 
                                  b + c = 10 :=
by {
  -- proof omitted, this is just the statement
  sorry
}

end NUMINAMATH_GPT_cost_of_berries_and_cheese_l356_35636


namespace NUMINAMATH_GPT_mad_hatter_must_secure_at_least_70_percent_l356_35693

theorem mad_hatter_must_secure_at_least_70_percent :
  ∀ (N : ℕ) (uM uH uD : ℝ) (α : ℝ),
    uM = 0.2 ∧ uH = 0.25 ∧ uD = 0.3 → 
    uM + α * 0.25 ≥ 0.25 + (1 - α) * 0.25 ∧
    uM + α * 0.25 ≥ 0.3 + (1 - α) * 0.25 →
    α ≥ 0.7 :=
by
  intros N uM uH uD α h hx
  sorry 

end NUMINAMATH_GPT_mad_hatter_must_secure_at_least_70_percent_l356_35693


namespace NUMINAMATH_GPT_math_problem_l356_35635

noncomputable def problem_statement (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (6 - a) * (6 - b) * (6 - c) * (6 - d) = 9

theorem math_problem
  (a b c d : ℕ)
  (h1 : a ≠ b)
  (h2 : a ≠ c)
  (h3 : a ≠ d)
  (h4 : b ≠ c)
  (h5 : b ≠ d)
  (h6 : c ≠ d)
  (h7 : (6 - a) * (6 - b) * (6 - c) * (6 - d) = 9) :
  a + b + c + d = 24 :=
sorry

end NUMINAMATH_GPT_math_problem_l356_35635


namespace NUMINAMATH_GPT_solve_system_l356_35608

theorem solve_system :
  ∃ (x y : ℚ), (4 * x - 35 * y = -1) ∧ (3 * y - x = 5) ∧ (x = -172 / 23) ∧ (y = -19 / 23) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l356_35608


namespace NUMINAMATH_GPT_distinguishes_conditional_from_sequential_l356_35634

variable (C P S I D : Prop)

-- Conditions
def conditional_structure_includes_processing_box  : Prop := C = P
def conditional_structure_includes_start_end_box   : Prop := C = S
def conditional_structure_includes_io_box          : Prop := C = I
def conditional_structure_includes_decision_box    : Prop := C = D
def sequential_structure_excludes_decision_box     : Prop := ¬S = D

-- Proof problem statement
theorem distinguishes_conditional_from_sequential : C → S → I → D → P → 
    (conditional_structure_includes_processing_box C P) ∧ 
    (conditional_structure_includes_start_end_box C S) ∧ 
    (conditional_structure_includes_io_box C I) ∧ 
    (conditional_structure_includes_decision_box C D) ∧ 
    sequential_structure_excludes_decision_box S D → 
    (D = true) :=
by sorry

end NUMINAMATH_GPT_distinguishes_conditional_from_sequential_l356_35634


namespace NUMINAMATH_GPT_complement_intersection_l356_35686

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection :
  U \ (A ∩ B) = {1, 4, 5} := by
    sorry

end NUMINAMATH_GPT_complement_intersection_l356_35686


namespace NUMINAMATH_GPT_multiplication_identity_l356_35696

theorem multiplication_identity : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ 29 := by
  sorry

end NUMINAMATH_GPT_multiplication_identity_l356_35696


namespace NUMINAMATH_GPT_initial_number_is_correct_l356_35641

theorem initial_number_is_correct (x : ℝ) (h : 8 * x - 4 = 2.625) : x = 0.828125 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_is_correct_l356_35641


namespace NUMINAMATH_GPT_floor_floor_3x_eq_floor_x_plus_1_l356_35662

theorem floor_floor_3x_eq_floor_x_plus_1 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1⌋ = ⌊x + 1⌋) ↔ (2 / 3 ≤ x ∧ x < 4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_floor_floor_3x_eq_floor_x_plus_1_l356_35662


namespace NUMINAMATH_GPT_bag_of_potatoes_weight_l356_35658

variable (W : ℝ)

-- Define the condition given in the problem.
def condition : Prop := W = 12 / (W / 2)

-- Define the statement we want to prove.
theorem bag_of_potatoes_weight : condition W → W = 24 := by
  intro h
  sorry

end NUMINAMATH_GPT_bag_of_potatoes_weight_l356_35658


namespace NUMINAMATH_GPT_train_travel_distance_l356_35688

theorem train_travel_distance (m : ℝ) (h : 3 * 60 * 1 = m) : m = 180 :=
by
  sorry

end NUMINAMATH_GPT_train_travel_distance_l356_35688


namespace NUMINAMATH_GPT_atomic_weight_Ba_l356_35665

-- Definitions for conditions
def atomic_weight_O : ℕ := 16
def molecular_weight_compound : ℕ := 153

-- Theorem statement
theorem atomic_weight_Ba : ∃ bw, molecular_weight_compound = bw + atomic_weight_O ∧ bw = 137 :=
by {
  -- Skip the proof
  sorry
}

end NUMINAMATH_GPT_atomic_weight_Ba_l356_35665


namespace NUMINAMATH_GPT_intersection_sum_zero_l356_35663

-- Definitions from conditions:
def lineA (x : ℝ) : ℝ := -x
def lineB (x : ℝ) : ℝ := 5 * x - 10

-- Declaration of the theorem:
theorem intersection_sum_zero : ∃ a b : ℝ, lineA a = b ∧ lineB a = b ∧ a + b = 0 := sorry

end NUMINAMATH_GPT_intersection_sum_zero_l356_35663


namespace NUMINAMATH_GPT_range_of_a_l356_35643

noncomputable def A (a : ℝ) := {x : ℝ | a < x ∧ x < 2 * a + 1}
def B := {x : ℝ | abs (x - 1) > 2}

theorem range_of_a (a : ℝ) (h : A a ⊆ B) : a ≤ -1 ∨ a ≥ 3 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l356_35643


namespace NUMINAMATH_GPT_ellipse_parametric_form_l356_35652

theorem ellipse_parametric_form :
  (∃ A B C D E F : ℤ,
    ((∀ t : ℝ, (3 * (Real.sin t - 2)) / (3 - Real.cos t) = x ∧ 
     (2 * (Real.cos t - 4)) / (3 - Real.cos t) = y) → 
    (A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0)) ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.gcd (Int.natAbs D) (Int.gcd (Int.natAbs E) (Int.natAbs F))))) = 1 ∧
    (Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E + Int.natAbs F = 1846)) := 
sorry

end NUMINAMATH_GPT_ellipse_parametric_form_l356_35652


namespace NUMINAMATH_GPT_students_passed_l356_35697

noncomputable def total_students : ℕ := 360
noncomputable def bombed : ℕ := (5 * total_students) / 12
noncomputable def not_bombed : ℕ := total_students - bombed
noncomputable def no_show : ℕ := (7 * not_bombed) / 15
noncomputable def remaining_after_no_show : ℕ := not_bombed - no_show
noncomputable def less_than_D : ℕ := 45
noncomputable def remaining_after_less_than_D : ℕ := remaining_after_no_show - less_than_D
noncomputable def technical_issues : ℕ := remaining_after_less_than_D / 8
noncomputable def passed_students : ℕ := remaining_after_less_than_D - technical_issues

theorem students_passed : passed_students = 59 := by
  sorry

end NUMINAMATH_GPT_students_passed_l356_35697


namespace NUMINAMATH_GPT_matthew_younger_than_freddy_l356_35632

variables (M R F : ℕ)

-- Define the conditions
def sum_of_ages : Prop := M + R + F = 35
def matthew_older_than_rebecca : Prop := M = R + 2
def freddy_age : Prop := F = 15

-- Prove the statement "Matthew is 4 years younger than Freddy."
theorem matthew_younger_than_freddy (h1 : sum_of_ages M R F) (h2 : matthew_older_than_rebecca M R) (h3 : freddy_age F) :
    F - M = 4 := by
  sorry

end NUMINAMATH_GPT_matthew_younger_than_freddy_l356_35632


namespace NUMINAMATH_GPT_tan_theta_expr_l356_35611

theorem tan_theta_expr (θ : ℝ) (h : Real.tan θ = 4) : 
  (Real.sin θ + Real.cos θ) / (17 * Real.sin θ) + (Real.sin θ ^ 2) / 4 = 21 / 68 := 
by sorry

end NUMINAMATH_GPT_tan_theta_expr_l356_35611


namespace NUMINAMATH_GPT_problem_min_value_l356_35659

theorem problem_min_value {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + a * b + a * c + b * c = 4) : 
  (2 * a + b + c) ≥ 4 := 
  sorry

end NUMINAMATH_GPT_problem_min_value_l356_35659


namespace NUMINAMATH_GPT_trigonometric_identity_l356_35684

theorem trigonometric_identity (theta : ℝ) (h : Real.cos ((5 * Real.pi)/12 - theta) = 1/3) :
  Real.sin ((Real.pi)/12 + theta) = 1/3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l356_35684


namespace NUMINAMATH_GPT_sum_of_fractions_l356_35661

theorem sum_of_fractions :
  (3 / 20 : ℝ) +  (7 / 200) + (8 / 2000) + (3 / 20000) = 0.1892 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l356_35661


namespace NUMINAMATH_GPT_train_arrival_day_l356_35668

-- Definitions for the start time and journey duration
def start_time : ℕ := 0  -- early morning (0 hours) on Tuesday
def journey_duration : ℕ := 28  -- 28 hours

-- Proving the arrival time
theorem train_arrival_day (start_time journey_duration : ℕ) :
  journey_duration == 28 → 
  start_time == 0 → 
  (journey_duration / 24, journey_duration % 24) == (1, 4) → 
  true := 
by
  intros
  sorry

end NUMINAMATH_GPT_train_arrival_day_l356_35668


namespace NUMINAMATH_GPT_mimi_spent_on_clothes_l356_35670

theorem mimi_spent_on_clothes : 
  let A := 800
  let N := 2 * A
  let S := 4 * A
  let P := 1 / 2 * N
  let total_spending := 10000
  let total_sneaker_spending := A + N + S + P
  let amount_spent_on_clothes := total_spending - total_sneaker_spending
  amount_spent_on_clothes = 3600 := 
by
  sorry

end NUMINAMATH_GPT_mimi_spent_on_clothes_l356_35670


namespace NUMINAMATH_GPT_wade_customers_sunday_l356_35680

theorem wade_customers_sunday :
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let total_tips := 296
  let tips_friday := customers_friday * tips_per_customer
  let tips_saturday := customers_saturday * tips_per_customer
  let tips_fri_sat := tips_friday + tips_saturday
  let tips_sunday := total_tips - tips_fri_sat
  let customers_sunday := tips_sunday / tips_per_customer
  customers_sunday = 36 :=
by
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let total_tips := 296
  let tips_friday := customers_friday * tips_per_customer
  let tips_saturday := customers_saturday * tips_per_customer
  let tips_fri_sat := tips_friday + tips_saturday
  let tips_sunday := total_tips - tips_fri_sat
  let customers_sunday := tips_sunday / tips_per_customer
  have h : customers_sunday = 36 := by sorry
  exact h

end NUMINAMATH_GPT_wade_customers_sunday_l356_35680


namespace NUMINAMATH_GPT_symmetric_point_coordinates_l356_35699

theorem symmetric_point_coordinates :
  ∀ (M N : ℝ × ℝ), M = (3, -4) ∧ M.fst = -N.fst ∧ M.snd = N.snd → N = (-3, -4) :=
by
  intro M N h
  sorry

end NUMINAMATH_GPT_symmetric_point_coordinates_l356_35699


namespace NUMINAMATH_GPT_range_of_a_l356_35669

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 2 - 1

def is_fixed_point (a x : ℝ) : Prop := f a x = x

def is_stable_point (a x : ℝ) : Prop := f a (f a x) = x

def are_equal_sets (a : ℝ) : Prop :=
  {x : ℝ | is_fixed_point a x} = {x : ℝ | is_stable_point a x}

theorem range_of_a (a : ℝ) (h : are_equal_sets a) : - (1 / 4) ≤ a ∧ a ≤ 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l356_35669


namespace NUMINAMATH_GPT_monkey_bananas_max_l356_35671

noncomputable def max_bananas_home : ℕ :=
  let total_bananas := 100
  let distance := 50
  let carry_capacity := 50
  let consumption_rate := 1
  let distance_each_way := distance / 2
  let bananas_eaten_each_way := distance_each_way * consumption_rate
  let bananas_left_midway := total_bananas / 2 - bananas_eaten_each_way
  let bananas_picked_midway := bananas_left_midway * 2
  let bananas_left_home := bananas_picked_midway - distance_each_way * consumption_rate
  bananas_left_home

theorem monkey_bananas_max : max_bananas_home = 25 :=
  sorry

end NUMINAMATH_GPT_monkey_bananas_max_l356_35671


namespace NUMINAMATH_GPT_value_of_m_l356_35685

theorem value_of_m (m : ℝ) (h : ∀ x : ℝ, 0 < x → x < 2 → - (1 / 2) * x^2 + 2 * x ≤ m * x) :
  m = 1 :=
sorry

end NUMINAMATH_GPT_value_of_m_l356_35685


namespace NUMINAMATH_GPT_greatest_abs_solution_l356_35674

theorem greatest_abs_solution :
  (∃ x : ℝ, x^2 + 18 * x + 81 = 0 ∧ ∀ y : ℝ, y^2 + 18 * y + 81 = 0 → |x| ≥ |y| ∧ |x| = 9) :=
sorry

end NUMINAMATH_GPT_greatest_abs_solution_l356_35674


namespace NUMINAMATH_GPT_sum_x1_x2_eq_five_l356_35624

theorem sum_x1_x2_eq_five {x1 x2 : ℝ} 
  (h1 : 2^x1 = 5 - x1)
  (h2 : x2 + Real.log x2 / Real.log 2 = 5) : 
  x1 + x2 = 5 := 
sorry

end NUMINAMATH_GPT_sum_x1_x2_eq_five_l356_35624


namespace NUMINAMATH_GPT_maximize_c_l356_35650

theorem maximize_c (c d e : ℤ) (h1 : 5 * c + (d - 12)^2 + e^3 = 235) (h2 : c < d) : c ≤ 22 :=
sorry

end NUMINAMATH_GPT_maximize_c_l356_35650


namespace NUMINAMATH_GPT_one_element_in_A_inter_B_range_m_l356_35660

theorem one_element_in_A_inter_B_range_m (m : ℝ) :
  let A := {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = -x^2 + m * x - 1}
  let B := {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = 3 - x ∧ 0 ≤ x ∧ x ≤ 3}
  (∃! p, p ∈ A ∧ p ∈ B) → (m = 3 ∨ m > 10 / 3) :=
by
  sorry

end NUMINAMATH_GPT_one_element_in_A_inter_B_range_m_l356_35660


namespace NUMINAMATH_GPT_compute_k_l356_35610

noncomputable def tan_inverse (k : ℝ) : ℝ := Real.arctan k

theorem compute_k (x k : ℝ) (hx1 : Real.tan x = 2 / 3) (hx2 : Real.tan (3 * x) = 3 / 5) : k = 2 / 3 := sorry

end NUMINAMATH_GPT_compute_k_l356_35610


namespace NUMINAMATH_GPT_houses_in_lawrence_county_l356_35606

theorem houses_in_lawrence_county 
  (houses_before_boom : ℕ := 1426) 
  (houses_built_during_boom : ℕ := 574) 
  : houses_before_boom + houses_built_during_boom = 2000 := 
by 
  sorry

end NUMINAMATH_GPT_houses_in_lawrence_county_l356_35606


namespace NUMINAMATH_GPT_count_ball_distribution_l356_35695

theorem count_ball_distribution (A B C D : ℕ) (balls : ℕ) :
  (A + B > C + D ∧ A + B + C + D = balls) → 
  (balls = 30) →
  (∃ n, n = 2600) :=
by
  intro h_ball_dist h_balls
  sorry

end NUMINAMATH_GPT_count_ball_distribution_l356_35695


namespace NUMINAMATH_GPT_find_integer_n_l356_35644

theorem find_integer_n (n : ℤ) (h : (⌊n^2 / 4⌋ - (⌊n / 2⌋)^2) = 3) : n = 7 :=
sorry

end NUMINAMATH_GPT_find_integer_n_l356_35644


namespace NUMINAMATH_GPT_expression_for_f_when_x_lt_0_l356_35666

noncomputable section

variable (f : ℝ → ℝ)

theorem expression_for_f_when_x_lt_0
  (hf_neg : ∀ x : ℝ, f (-x) = -f x)
  (hf_pos : ∀ x : ℝ, x > 0 → f x = x * abs (x - 2)) :
  ∀ x : ℝ, x < 0 → f x = x * abs (x + 2) :=
by
  sorry

end NUMINAMATH_GPT_expression_for_f_when_x_lt_0_l356_35666


namespace NUMINAMATH_GPT_selling_price_correct_l356_35637

namespace Shopkeeper

def costPrice : ℝ := 1500
def profitPercentage : ℝ := 20
def expectedSellingPrice : ℝ := 1800

theorem selling_price_correct
  (cp : ℝ := costPrice)
  (pp : ℝ := profitPercentage) :
  cp * (1 + pp / 100) = expectedSellingPrice :=
by
  sorry

end Shopkeeper

end NUMINAMATH_GPT_selling_price_correct_l356_35637


namespace NUMINAMATH_GPT_fraction_defined_iff_l356_35618

theorem fraction_defined_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (|x| - 6)) ↔ (x ≠ 6 ∧ x ≠ -6) :=
by 
  sorry

end NUMINAMATH_GPT_fraction_defined_iff_l356_35618


namespace NUMINAMATH_GPT_initial_days_planned_l356_35600

-- We define the variables and conditions given in the problem.
variables (men_original men_absent men_remaining days_remaining days_initial : ℕ)
variable (work_equivalence : men_original * days_initial = men_remaining * days_remaining)

-- Conditions from the problem
axiom men_original_cond : men_original = 48
axiom men_absent_cond : men_absent = 8
axiom men_remaining_cond : men_remaining = men_original - men_absent
axiom days_remaining_cond : days_remaining = 18

-- Theorem to be proved
theorem initial_days_planned : days_initial = 15 :=
by
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_initial_days_planned_l356_35600


namespace NUMINAMATH_GPT_quadratic_has_real_roots_l356_35656

theorem quadratic_has_real_roots (k : ℝ) (h : k > 0) : ∃ x : ℝ, x^2 + 2 * x - k = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_l356_35656


namespace NUMINAMATH_GPT_number_of_valid_strings_l356_35619

def count_valid_strings (n : ℕ) : ℕ :=
  4^n - 3 * 3^n + 3 * 2^n - 1

theorem number_of_valid_strings (n : ℕ) :
  count_valid_strings n = 4^n - 3 * 3^n + 3 * 2^n - 1 :=
by sorry

end NUMINAMATH_GPT_number_of_valid_strings_l356_35619


namespace NUMINAMATH_GPT_John_traded_in_car_money_back_l356_35605

-- First define the conditions provided in the problem.
def UberEarnings : ℝ := 30000
def CarCost : ℝ := 18000
def UberProfit : ℝ := 18000

-- We need to prove that John got $6000 back when trading in the car.
theorem John_traded_in_car_money_back : 
  UberEarnings - UberProfit = CarCost - 6000 := 
by
  -- provide the detailed steps inside the proof block if needed
  sorry

end NUMINAMATH_GPT_John_traded_in_car_money_back_l356_35605


namespace NUMINAMATH_GPT_two_digit_numbers_l356_35694

theorem two_digit_numbers (n m : ℕ) (Hn : 1 ≤ n ∧ n ≤ 9) (Hm : n < m ∧ m ≤ 9) :
  ∃ (count : ℕ), count = 36 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_numbers_l356_35694


namespace NUMINAMATH_GPT_erasers_left_in_the_box_l356_35657

-- Conditions expressed as definitions
def E0 : ℕ := 320
def E1 : ℕ := E0 - 67
def E2 : ℕ := E1 - 126
def E3 : ℕ := E2 + 30

-- Proof problem statement
theorem erasers_left_in_the_box : E3 = 157 := 
by sorry

end NUMINAMATH_GPT_erasers_left_in_the_box_l356_35657


namespace NUMINAMATH_GPT_problem_condition_holds_l356_35678

theorem problem_condition_holds (x y : ℝ) (h₁ : x + 0.35 * y - (x + y) = 200) : y = -307.69 :=
sorry

end NUMINAMATH_GPT_problem_condition_holds_l356_35678


namespace NUMINAMATH_GPT_contrapositive_iff_l356_35664

theorem contrapositive_iff (a b : ℤ) : (a > b → a - 5 > b - 5) ↔ (a - 5 ≤ b - 5 → a ≤ b) :=
by sorry

end NUMINAMATH_GPT_contrapositive_iff_l356_35664


namespace NUMINAMATH_GPT_part1_sales_increase_part2_price_reduction_l356_35691

-- Part 1: If the price is reduced by 4 yuan, the new average daily sales will be 28 items.
theorem part1_sales_increase (initial_sales : ℕ) (increase_per_yuan : ℕ) (reduction : ℕ) :
  initial_sales = 20 → increase_per_yuan = 2 → reduction = 4 →
  initial_sales + increase_per_yuan * reduction = 28 :=
by sorry

-- Part 2: By how much should the price of each item be reduced for a daily profit of 1050 yuan.
theorem part2_price_reduction (initial_sales : ℕ) (increase_per_yuan : ℕ) (initial_profit : ℕ) 
  (target_profit : ℕ) (min_profit_per_item : ℕ) (x : ℕ) :
  initial_sales = 20 → increase_per_yuan = 2 → initial_profit = 40 → target_profit = 1050 
  → min_profit_per_item = 25 → (40 - x) * (20 + 2 * x) = 1050 → (40 - x) ≥ 25 → x = 5 :=
by sorry

end NUMINAMATH_GPT_part1_sales_increase_part2_price_reduction_l356_35691


namespace NUMINAMATH_GPT_part1_part2_l356_35626

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x)
  else (Real.log x / Real.log 3 - 1) * (Real.log x / Real.log 3 - 2)

theorem part1 : f (Real.log 3 / Real.log 2 - Real.log 2 / Real.log 2) = 2 / 3 := by
  sorry

theorem part2 : ∃ x : ℝ, f x = -1 / 4 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l356_35626


namespace NUMINAMATH_GPT_area_EPHQ_l356_35677

theorem area_EPHQ {EFGH : Type} 
  (rectangle_EFGH : EFGH) 
  (length_EF : Real) (width_EG : Real) 
  (P_point : Real) (Q_point : Real) 
  (area_EFGH : Real) 
  (area_EFP : Real) 
  (area_EHQ : Real) : 
  length_EF = 12 → width_EG = 6 → P_point = 4 → Q_point = 3 → 
  area_EFGH = length_EF * width_EG →
  area_EFP = (1 / 2) * width_EG * P_point →
  area_EHQ = (1 / 2) * length_EF * Q_point → 
  (area_EFGH - area_EFP - area_EHQ) = 42 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end NUMINAMATH_GPT_area_EPHQ_l356_35677


namespace NUMINAMATH_GPT_different_language_classes_probability_l356_35628

theorem different_language_classes_probability :
  let total_students := 40
  let french_students := 28
  let spanish_students := 26
  let german_students := 15
  let french_and_spanish_students := 10
  let french_and_german_students := 6
  let spanish_and_german_students := 8
  let all_three_languages_students := 3
  let total_pairs := Nat.choose total_students 2
  let french_only := french_students - (french_and_spanish_students + french_and_german_students - all_three_languages_students) - all_three_languages_students
  let spanish_only := spanish_students - (french_and_spanish_students + spanish_and_german_students - all_three_languages_students) - all_three_languages_students
  let german_only := german_students - (french_and_german_students + spanish_and_german_students - all_three_languages_students) - all_three_languages_students
  let french_only_pairs := Nat.choose french_only 2
  let spanish_only_pairs := Nat.choose spanish_only 2
  let german_only_pairs := Nat.choose german_only 2
  let single_language_pairs := french_only_pairs + spanish_only_pairs + german_only_pairs
  let different_classes_probability := 1 - (single_language_pairs / total_pairs)
  different_classes_probability = (34 / 39) :=
by
  sorry

end NUMINAMATH_GPT_different_language_classes_probability_l356_35628


namespace NUMINAMATH_GPT_ratio_of_red_to_blue_marbles_l356_35602

theorem ratio_of_red_to_blue_marbles (total_marbles yellow_marbles : ℕ) (green_marbles blue_marbles red_marbles : ℕ) 
  (odds_blue : ℚ) 
  (h1 : total_marbles = 60) 
  (h2 : yellow_marbles = 20) 
  (h3 : green_marbles = yellow_marbles / 2) 
  (h4 : red_marbles + blue_marbles = total_marbles - (yellow_marbles + green_marbles)) 
  (h5 : odds_blue = 0.25) 
  (h6 : blue_marbles = odds_blue * (red_marbles + blue_marbles)) : 
  red_marbles / blue_marbles = 11 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_red_to_blue_marbles_l356_35602


namespace NUMINAMATH_GPT_coordinates_of_AC_l356_35689

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p1 p2 : Point3D) : Point3D :=
  { x := p1.x - p2.x,
    y := p1.y - p2.y,
    z := p1.z - p2.z }

def scalar_mult (k : ℝ) (v : Point3D) : Point3D :=
  { x := k * v.x,
    y := k * v.y,
    z := k * v.z }

noncomputable def A : Point3D := { x := 1, y := 2, z := 3 }
noncomputable def B : Point3D := { x := 4, y := 5, z := 9 }

theorem coordinates_of_AC : vector_sub B A = { x := 3, y := 3, z := 6 } →
  scalar_mult (1 / 3) (vector_sub B A) = { x := 1, y := 1, z := 2 } :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_AC_l356_35689


namespace NUMINAMATH_GPT_base_b_square_l356_35630

theorem base_b_square (b : ℕ) (h : b > 2) : ∃ k : ℕ, 121 = k ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_base_b_square_l356_35630


namespace NUMINAMATH_GPT_red_balls_estimation_l356_35679

noncomputable def numberOfRedBalls (x : ℕ) : ℝ := x / (x + 3)

theorem red_balls_estimation {x : ℕ} (h : numberOfRedBalls x = 0.85) : x = 17 :=
by
  sorry

end NUMINAMATH_GPT_red_balls_estimation_l356_35679


namespace NUMINAMATH_GPT_magnitude_of_two_a_minus_b_l356_35675

namespace VectorMagnitude

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (3, -2)

-- Function to calculate the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Vector operation 2a - b
def two_a_minus_b : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

-- The theorem to prove
theorem magnitude_of_two_a_minus_b : magnitude two_a_minus_b = Real.sqrt 17 := by
  sorry

end VectorMagnitude

end NUMINAMATH_GPT_magnitude_of_two_a_minus_b_l356_35675


namespace NUMINAMATH_GPT_relation_between_x_and_y_l356_35609

noncomputable def x : ℝ := 2 + Real.sqrt 3
noncomputable def y : ℝ := 1 / (2 - Real.sqrt 3)

theorem relation_between_x_and_y : x = y := sorry

end NUMINAMATH_GPT_relation_between_x_and_y_l356_35609


namespace NUMINAMATH_GPT_PatriciaHighlightFilmTheorem_l356_35642

def PatriciaHighlightFilmProblem : Prop :=
  let point_guard_seconds := 130
  let shooting_guard_seconds := 145
  let small_forward_seconds := 85
  let power_forward_seconds := 60
  let center_seconds := 180
  let total_seconds := point_guard_seconds + shooting_guard_seconds + small_forward_seconds + power_forward_seconds + center_seconds
  let num_players := 5
  let average_seconds := total_seconds / num_players
  let average_minutes := average_seconds / 60
  average_minutes = 2

theorem PatriciaHighlightFilmTheorem : PatriciaHighlightFilmProblem :=
  by
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_PatriciaHighlightFilmTheorem_l356_35642


namespace NUMINAMATH_GPT_problem1_problem2_l356_35672

-- Define the variables for the two problems
variables (a b x : ℝ)

-- The first problem
theorem problem1 :
  (a + 2 * b) ^ 2 - 4 * b * (a + b) = a ^ 2 :=
by 
  -- Proof goes here
  sorry

-- The second problem
theorem problem2 :
  ((x ^ 2 - 2 * x) / (x ^ 2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x ^ 2 - 4)) = x + 2 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_problem1_problem2_l356_35672


namespace NUMINAMATH_GPT_directrix_of_parabola_l356_35639

theorem directrix_of_parabola (p m : ℝ) (hp : p > 0)
  (hM_on_parabola : (4, m).fst ^ 2 = 2 * p * (4, m).snd)
  (hM_to_focus : dist (4, m) (p / 2, 0) = 6) :
  -p/2 = -2 :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l356_35639


namespace NUMINAMATH_GPT_base_case_of_interior_angle_sum_l356_35645

-- Definitions consistent with conditions: A convex polygon with at least n sides where n >= 3.
def convex_polygon (n : ℕ) : Prop := n ≥ 3

-- Proposition: If w the sum of angles for convex polygons, we start checking from n = 3.
theorem base_case_of_interior_angle_sum (n : ℕ) (h : convex_polygon n) :
  n = 3 := 
by
  sorry

end NUMINAMATH_GPT_base_case_of_interior_angle_sum_l356_35645


namespace NUMINAMATH_GPT_find_value_of_a3_plus_a5_l356_35627

variable {a : ℕ → ℝ}
variable {r : ℝ}

noncomputable def geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem find_value_of_a3_plus_a5 (h_geom : geometric_seq a r) (h_pos: ∀ n, 0 < a n)
  (h_eq: a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) :
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_GPT_find_value_of_a3_plus_a5_l356_35627


namespace NUMINAMATH_GPT_statue_selling_price_l356_35623

/-- Problem conditions -/
def original_cost : ℤ := 550
def profit_percentage : ℝ := 0.20

/-- Proof problem statement -/
theorem statue_selling_price : original_cost + profit_percentage * original_cost = 660 := by
  sorry

end NUMINAMATH_GPT_statue_selling_price_l356_35623


namespace NUMINAMATH_GPT_percentage_of_total_is_sixty_l356_35690

def num_boys := 600
def diff_boys_girls := 400
def num_girls := num_boys + diff_boys_girls
def total_people := num_boys + num_girls
def target_number := 960
def target_percentage := (target_number / total_people) * 100

theorem percentage_of_total_is_sixty :
  target_percentage = 60 := by
  sorry

end NUMINAMATH_GPT_percentage_of_total_is_sixty_l356_35690


namespace NUMINAMATH_GPT_locus_of_M_l356_35681

theorem locus_of_M (k : ℝ) (A B M : ℝ × ℝ) (hA : A.1 ≥ 0 ∧ A.2 = 0) (hB : B.2 ≥ 0 ∧ B.1 = 0) (h_sum : A.1 + B.2 = k) :
    ∃ (M : ℝ × ℝ), (M.1 - k / 2)^2 + (M.2 - k / 2)^2 = k^2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_locus_of_M_l356_35681


namespace NUMINAMATH_GPT_solution_inequality_equivalence_l356_35698

-- Define the inequality to be proved
def inequality (x : ℝ) : Prop :=
  (x + 1 / 2) * (3 / 2 - x) ≥ 0

-- Define the set of solutions such that -1/2 ≤ x ≤ 3/2
def solution_set (x : ℝ) : Prop :=
  -1 / 2 ≤ x ∧ x ≤ 3 / 2

-- The statement to be proved: the solution set of the inequality is {x | -1/2 ≤ x ≤ 3/2}
theorem solution_inequality_equivalence :
  {x : ℝ | inequality x} = {x : ℝ | solution_set x} :=
by 
  sorry

end NUMINAMATH_GPT_solution_inequality_equivalence_l356_35698


namespace NUMINAMATH_GPT_power_inequality_l356_35638

theorem power_inequality (a b c d : ℝ) (ha : 0 < a) (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
sorry

end NUMINAMATH_GPT_power_inequality_l356_35638


namespace NUMINAMATH_GPT_inequality_relationship_l356_35621

noncomputable def a : ℝ := Real.sin (4 / 5)
noncomputable def b : ℝ := Real.cos (4 / 5)
noncomputable def c : ℝ := Real.tan (4 / 5)

theorem inequality_relationship : c > a ∧ a > b := sorry

end NUMINAMATH_GPT_inequality_relationship_l356_35621


namespace NUMINAMATH_GPT_number_of_hours_sold_l356_35655

def packs_per_hour_peak := 6
def packs_per_hour_low := 4
def price_per_pack := 60
def extra_revenue := 1800

def revenue_per_hour_peak := packs_per_hour_peak * price_per_pack
def revenue_per_hour_low := packs_per_hour_low * price_per_pack
def revenue_diff_per_hour := revenue_per_hour_peak - revenue_per_hour_low

theorem number_of_hours_sold (h : ℕ) 
  (h_eq : revenue_diff_per_hour * h = extra_revenue) : 
  h = 15 :=
by
  -- skip proof
  sorry

end NUMINAMATH_GPT_number_of_hours_sold_l356_35655


namespace NUMINAMATH_GPT_remainder_when_dividing_150_l356_35604

theorem remainder_when_dividing_150 (k : ℕ) (hk1 : k > 0) (hk2 : 80 % k^2 = 8) : 150 % k = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_dividing_150_l356_35604


namespace NUMINAMATH_GPT_find_random_discount_l356_35667

theorem find_random_discount
  (initial_price : ℝ) (final_price : ℝ) (autumn_discount : ℝ) (loyalty_discount : ℝ) (random_discount : ℝ) :
  initial_price = 230 ∧ final_price = 69 ∧ autumn_discount = 0.25 ∧ loyalty_discount = 0.20 ∧ 
  final_price = initial_price * (1 - autumn_discount) * (1 - loyalty_discount) * (1 - random_discount / 100) →
  random_discount = 50 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_random_discount_l356_35667


namespace NUMINAMATH_GPT_volume_in_cubic_yards_l356_35648

-- Definition: A box with a specific volume in cubic feet.
def volume_in_cubic_feet (v : ℝ) : Prop :=
  v = 200

-- Definition: Conversion factor from cubic feet to cubic yards.
def cubic_feet_per_cubic_yard : ℝ := 27

-- Theorem: The volume of the box in cubic yards given the volume in cubic feet.
theorem volume_in_cubic_yards (v_cubic_feet : ℝ) 
    (h : volume_in_cubic_feet v_cubic_feet) : 
    v_cubic_feet / cubic_feet_per_cubic_yard = 200 / 27 :=
  by
    rw [h]
    sorry

end NUMINAMATH_GPT_volume_in_cubic_yards_l356_35648


namespace NUMINAMATH_GPT_find_a_plus_b_l356_35617

-- Given conditions
variable (a b : ℝ)

-- The imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Condition equation
def equation := (a + i) * i = b - 2 * i

-- Define the lean statement
theorem find_a_plus_b (h : equation a b) : a + b = -3 :=
by sorry

end NUMINAMATH_GPT_find_a_plus_b_l356_35617


namespace NUMINAMATH_GPT_h_at_2_l356_35601

noncomputable def f (x : ℝ) : ℝ := 3 * x - 4
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (3 * f x) - 3
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_at_2 : h 2 = 3 * Real.sqrt 6 - 13 := 
by 
  sorry -- We skip the proof steps.

end NUMINAMATH_GPT_h_at_2_l356_35601


namespace NUMINAMATH_GPT_remainder_mod_500_l356_35603

theorem remainder_mod_500 :
  ( 5^(5^(5^5)) ) % 500 = 125 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_remainder_mod_500_l356_35603


namespace NUMINAMATH_GPT_last_two_digits_of_7_pow_2023_l356_35649

theorem last_two_digits_of_7_pow_2023 : (7 ^ 2023) % 100 = 43 := by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_7_pow_2023_l356_35649


namespace NUMINAMATH_GPT_last_10_digits_repeat_periodically_l356_35633

theorem last_10_digits_repeat_periodically :
  ∃ (p : ℕ) (n₀ : ℕ), p = 4 * 10^9 ∧ n₀ = 10 ∧ 
  ∀ n, (2^(n + p) % 10^10 = 2^n % 10^10) :=
by sorry

end NUMINAMATH_GPT_last_10_digits_repeat_periodically_l356_35633


namespace NUMINAMATH_GPT_katy_brownies_l356_35614

theorem katy_brownies :
  ∃ (n : ℤ), (n = 5 + 2 * 5) :=
by
  sorry

end NUMINAMATH_GPT_katy_brownies_l356_35614


namespace NUMINAMATH_GPT_greatest_x_lcm_l356_35625

theorem greatest_x_lcm (x : ℕ) (h1 : Nat.lcm x 15 = Nat.lcm 90 15) (h2 : Nat.lcm x 18 = Nat.lcm 90 18) : x = 90 := 
sorry

end NUMINAMATH_GPT_greatest_x_lcm_l356_35625


namespace NUMINAMATH_GPT_real_part_of_one_over_one_minus_z_l356_35673

open Complex

noncomputable def real_part_fraction {z : ℂ} (hz1 : norm z = 1) (hz2 : ¬(z.im = 0)) : ℝ :=
  re (1 / (1 - z))

theorem real_part_of_one_over_one_minus_z (z : ℂ) (hz1 : norm z = 1) (hz2 : ¬(z.im = 0)) :
  real_part_fraction hz1 hz2 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_real_part_of_one_over_one_minus_z_l356_35673


namespace NUMINAMATH_GPT_log_abs_monotone_decreasing_l356_35622

open Real

theorem log_abs_monotone_decreasing {a : ℝ} (h : ∀ x y, 0 < x ∧ x < y ∧ y ≤ a → |log x| ≥ |log y|) : 0 < a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_log_abs_monotone_decreasing_l356_35622


namespace NUMINAMATH_GPT_exists_initial_value_l356_35654

theorem exists_initial_value (x : ℤ) : ∃ y : ℤ, x + 49 = y^2 :=
sorry

end NUMINAMATH_GPT_exists_initial_value_l356_35654


namespace NUMINAMATH_GPT_smallest_even_integer_l356_35607

theorem smallest_even_integer (n : ℕ) (h_even : n % 2 = 0)
  (h_2digit : 10 ≤ n ∧ n ≤ 98)
  (h_property : (n - 2) * n * (n + 2) = 5 * ((n - 2) + n + (n + 2))) :
  n = 86 :=
by
  sorry

end NUMINAMATH_GPT_smallest_even_integer_l356_35607


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l356_35692

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h₀ : b > a) (h₁ : a > 0) :
  (1 / (a ^ 2) > 1 / (b ^ 2)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l356_35692


namespace NUMINAMATH_GPT_problem_statement_l356_35620

-- Definitions of the operations △ and ⊗
def triangle (a b : ℤ) : ℤ := a + b + a * b - 1
def otimes (a b : ℤ) : ℤ := a * a - a * b + b * b

-- The theorem statement
theorem problem_statement : triangle 3 (otimes 2 4) = 50 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l356_35620


namespace NUMINAMATH_GPT_range_of_m_local_odd_function_l356_35631

def is_local_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f (-x) = -f x

noncomputable def f (x m : ℝ) : ℝ :=
  9^x - m * 3^x - 3

theorem range_of_m_local_odd_function :
  (∀ m : ℝ, is_local_odd_function (λ x => f x m) ↔ m ∈ Set.Ici (-2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_local_odd_function_l356_35631


namespace NUMINAMATH_GPT_corner_cell_revisit_l356_35683

theorem corner_cell_revisit
    (M N : ℕ)
    (hM : M = 101)
    (hN : N = 200)
    (initial_position : ℕ × ℕ)
    (h_initial : initial_position = (0, 0) ∨ initial_position = (0, 200) ∨ initial_position = (101, 0) ∨ initial_position = (101, 200)) :
    ∃ final_position : ℕ × ℕ, 
      final_position = initial_position ∧ (final_position = (0, 0) ∨ final_position = (0, 200) ∨ final_position = (101, 0) ∨ final_position = (101, 200)) :=
by
  sorry

end NUMINAMATH_GPT_corner_cell_revisit_l356_35683


namespace NUMINAMATH_GPT_prime_root_condition_l356_35676

theorem prime_root_condition (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℤ, x ≠ y ∧ (x^2 + 2 * p * x - 240 * p = 0) ∧ (y^2 + 2 * p * y - 240 * p = 0) ∧ x*y = -240*p) → p = 5 :=
by sorry

end NUMINAMATH_GPT_prime_root_condition_l356_35676


namespace NUMINAMATH_GPT_range_of_f_l356_35629

noncomputable def f : ℝ → ℝ := sorry -- Define f appropriately

theorem range_of_f : Set.range f = {y : ℝ | 0 < y} :=
sorry

end NUMINAMATH_GPT_range_of_f_l356_35629


namespace NUMINAMATH_GPT_daily_egg_count_per_female_emu_l356_35687

noncomputable def emus_per_pen : ℕ := 6
noncomputable def pens : ℕ := 4
noncomputable def total_eggs_per_week : ℕ := 84

theorem daily_egg_count_per_female_emu :
  (total_eggs_per_week / ((pens * emus_per_pen) / 2 * 7) = 1) :=
by
  sorry

end NUMINAMATH_GPT_daily_egg_count_per_female_emu_l356_35687


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l356_35647

-- Problem 1
theorem problem1 : -2.8 + (-3.6) + 3 - (-3.6) = 0.2 := 
by
  sorry

-- Problem 2
theorem problem2 : (-4) ^ 2010 * (-0.25) ^ 2009 + (-12) * (1 / 3 - 3 / 4 + 5 / 6) = -9 := 
by
  sorry

-- Problem 3
theorem problem3 : 13 * (16/60 : ℝ) * 5 - 19 * (12/60 : ℝ) / 6 = 13 * (8/60 : ℝ) + 50 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l356_35647


namespace NUMINAMATH_GPT_line_equation_passing_through_and_perpendicular_l356_35613

theorem line_equation_passing_through_and_perpendicular :
  ∃ A B C : ℝ, (∀ x y : ℝ, 2 * x - 4 * y + 5 = 0 → -2 * x + y + 1 = 0 ∧ 
(x = 2 ∧ y = -1) → 2 * x + y - 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_passing_through_and_perpendicular_l356_35613


namespace NUMINAMATH_GPT_substring_012_appears_148_times_l356_35646

noncomputable def count_substring_012_in_base_3_concat (n : ℕ) : ℕ :=
  -- The function that counts the "012" substrings in the concatenated base-3 representations
  sorry

theorem substring_012_appears_148_times :
  count_substring_012_in_base_3_concat 728 = 148 :=
  sorry

end NUMINAMATH_GPT_substring_012_appears_148_times_l356_35646


namespace NUMINAMATH_GPT_pq_necessary_not_sufficient_l356_35653

theorem pq_necessary_not_sufficient (p q : Prop) : (p ∨ q) → (p ∧ q) ↔ false :=
by sorry

end NUMINAMATH_GPT_pq_necessary_not_sufficient_l356_35653


namespace NUMINAMATH_GPT_rope_length_after_knots_l356_35615

def num_ropes : ℕ := 64
def length_per_rope : ℕ := 25
def length_reduction_per_knot : ℕ := 3
def num_knots : ℕ := num_ropes - 1
def initial_total_length : ℕ := num_ropes * length_per_rope
def total_reduction : ℕ := num_knots * length_reduction_per_knot
def final_rope_length : ℕ := initial_total_length - total_reduction

theorem rope_length_after_knots :
  final_rope_length = 1411 := by
  sorry

end NUMINAMATH_GPT_rope_length_after_knots_l356_35615


namespace NUMINAMATH_GPT_john_paid_more_l356_35616

theorem john_paid_more 
  (original_price : ℝ)
  (discount_percentage : ℝ) 
  (tip_percentage : ℝ) 
  (discounted_price : ℝ)
  (john_tip : ℝ) 
  (john_total : ℝ)
  (jane_tip : ℝ)
  (jane_total : ℝ) 
  (difference : ℝ) :
  original_price = 42.00000000000004 →
  discount_percentage = 0.10 →
  tip_percentage = 0.15 →
  discounted_price = original_price - (discount_percentage * original_price) →
  john_tip = tip_percentage * original_price →
  john_total = original_price + john_tip →
  jane_tip = tip_percentage * discounted_price →
  jane_total = discounted_price + jane_tip →
  difference = john_total - jane_total →
  difference = 4.830000000000005 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_john_paid_more_l356_35616
