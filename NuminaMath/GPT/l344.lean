import Mathlib

namespace g_value_at_50_l344_34438

noncomputable def g : ℝ → ℝ :=
sorry

theorem g_value_at_50 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, 0 < x → 0 < y → x * g y - y ^ 2 * g x = g (x / y)) :
  g 50 = 0 :=
by
  sorry

end g_value_at_50_l344_34438


namespace p_squared_plus_one_over_p_squared_plus_six_l344_34490

theorem p_squared_plus_one_over_p_squared_plus_six (p : ℝ) (h : p + 1/p = 10) : p^2 + 1/p^2 + 6 = 104 :=
by {
  sorry
}

end p_squared_plus_one_over_p_squared_plus_six_l344_34490


namespace tiffany_bags_found_day_after_next_day_l344_34465

noncomputable def tiffany_start : Nat := 10
noncomputable def tiffany_next_day : Nat := 3
noncomputable def tiffany_total : Nat := 20
noncomputable def tiffany_day_after_next_day : Nat := 20 - (tiffany_start + tiffany_next_day)

theorem tiffany_bags_found_day_after_next_day : tiffany_day_after_next_day = 7 := by
  sorry

end tiffany_bags_found_day_after_next_day_l344_34465


namespace final_price_correct_l344_34464

noncomputable def final_price_per_litre : Real :=
  let cost_1 := 70 * 43 * (1 - 0.15)
  let cost_2 := 50 * 51 * (1 + 0.10)
  let cost_3 := 15 * 60 * (1 - 0.08)
  let cost_4 := 25 * 62 * (1 + 0.12)
  let cost_5 := 40 * 67 * (1 - 0.05)
  let cost_6 := 10 * 75 * (1 - 0.18)
  let total_cost := cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6
  let total_volume := 70 + 50 + 15 + 25 + 40 + 10
  total_cost / total_volume

theorem final_price_correct : final_price_per_litre = 52.80 := by
  sorry

end final_price_correct_l344_34464


namespace point_in_fourth_quadrant_l344_34491

theorem point_in_fourth_quadrant (m : ℝ) (h : m < 0) : (-m + 1 > 0 ∧ -1 < 0) :=
by
  sorry

end point_in_fourth_quadrant_l344_34491


namespace solution_interval_l344_34429

def check_solution (b : ℝ) (x : ℝ) : ℝ :=
  x^2 - b * x - 5

theorem solution_interval (b x : ℝ) :
  (check_solution b (-2) = 5) ∧
  (check_solution b (-1) = -1) ∧
  (check_solution b (4) = -1) ∧
  (check_solution b (5) = 5) →
  (∃ x, -2 < x ∧ x < -1 ∧ check_solution b x = 0) ∨
  (∃ x, 4 < x ∧ x < 5 ∧ check_solution b x = 0) :=
by
  sorry

end solution_interval_l344_34429


namespace time_taken_by_Arun_to_cross_train_B_l344_34425

structure Train :=
  (length : ℕ)
  (speed_kmh : ℕ)

def to_m_per_s (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 1000) / 3600

def relative_speed (trainA trainB : Train) : ℕ :=
  to_m_per_s trainA.speed_kmh + to_m_per_s trainB.speed_kmh

def total_length (trainA trainB : Train) : ℕ :=
  trainA.length + trainB.length

def time_to_cross (trainA trainB : Train) : ℕ :=
  total_length trainA trainB / relative_speed trainA trainB

theorem time_taken_by_Arun_to_cross_train_B :
  time_to_cross (Train.mk 175 54) (Train.mk 150 36) = 13 :=
by
  sorry

end time_taken_by_Arun_to_cross_train_B_l344_34425


namespace factorization_1_min_value_l344_34415

-- Problem 1: Prove that m² - 4mn + 3n² = (m - 3n)(m - n)
theorem factorization_1 (m n : ℤ) : m^2 - 4*m*n + 3*n^2 = (m - 3*n)*(m - n) :=
by
  sorry

-- Problem 2: Prove that the minimum value of m² - 3m + 2015 is 2012 3/4
theorem min_value (m : ℝ) : ∃ x : ℝ, x = m^2 - 3*m + 2015 ∧ x = 2012 + 3/4 :=
by
  sorry

end factorization_1_min_value_l344_34415


namespace prove_sum_is_12_l344_34445

theorem prove_sum_is_12 (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := 
by 
  sorry

end prove_sum_is_12_l344_34445


namespace percentage_of_x_l344_34472

theorem percentage_of_x (x : ℝ) (h : x > 0) : ((x / 5 + x / 25) / x) * 100 = 24 := 
by 
  sorry

end percentage_of_x_l344_34472


namespace students_passed_both_tests_l344_34428

theorem students_passed_both_tests
    (total_students : ℕ)
    (passed_long_jump : ℕ)
    (passed_shot_put : ℕ)
    (failed_both : ℕ)
    (h_total : total_students = 50)
    (h_long_jump : passed_long_jump = 40)
    (h_shot_put : passed_shot_put = 31)
    (h_failed_both : failed_both = 4) : 
    (total_students - failed_both = passed_long_jump + passed_shot_put - 25) :=
by 
  sorry

end students_passed_both_tests_l344_34428


namespace problem_statement_l344_34467

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h1 : ∀ x : ℝ, f (x + 2016) = f (-x + 2016))
    (h2 : ∀ x1 x2 : ℝ, 2016 ≤ x1 ∧ 2016 ≤ x2 ∧ x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0) :
    f 2019 < f 2014 ∧ f 2014 < f 2017 :=
sorry

end problem_statement_l344_34467


namespace ratio_of_w_to_y_l344_34482

theorem ratio_of_w_to_y (w x y z : ℚ)
  (h1 : w / x = 5 / 4)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 4) :
  w / y = 10 / 3 :=
sorry

end ratio_of_w_to_y_l344_34482


namespace parabola_shifts_down_decrease_c_real_roots_l344_34435

-- The parabolic function and conditions
variables {a b c k : ℝ}

-- Assumption that a is positive
axiom ha : a > 0

-- Parabola shifts down when constant term c is decreased
theorem parabola_shifts_down (c : ℝ) (k : ℝ) (hk : k > 0) :
  ∀ x, (a * x^2 + b * x + (c - k)) = (a * x^2 + b * x + c) - k :=
by sorry

-- Discriminant of quadratic equation ax^2 + bx + c = 0
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- If the discriminant is negative, decreasing c can result in real roots
theorem decrease_c_real_roots (b c : ℝ) (hb : b^2 < 4 * a * c) (k : ℝ) (hk : k > 0) :
  discriminant a b (c - k) ≥ 0 :=
by sorry

end parabola_shifts_down_decrease_c_real_roots_l344_34435


namespace algebraic_expression_value_l344_34447

variable {a b c : ℝ}

theorem algebraic_expression_value
  (h1 : (a + b) * (b + c) * (c + a) = 0)
  (h2 : a * b * c < 0) :
  (a / |a|) + (b / |b|) + (c / |c|) = 1 := by
  sorry

end algebraic_expression_value_l344_34447


namespace monotone_range_of_f_l344_34462

theorem monotone_range_of_f {f : ℝ → ℝ} (a : ℝ) 
  (h : ∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≤ y → f x ≤ f y) : a ≤ 0 :=
sorry

end monotone_range_of_f_l344_34462


namespace fraction_addition_simplified_form_l344_34419

theorem fraction_addition_simplified_form :
  (7 / 8) + (3 / 5) = 59 / 40 := 
by sorry

end fraction_addition_simplified_form_l344_34419


namespace solve_equation_l344_34417

theorem solve_equation (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) (h : a^2 = b * (b + 7)) : 
  (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
by sorry

end solve_equation_l344_34417


namespace interest_years_eq_three_l344_34411

theorem interest_years_eq_three :
  ∀ (x y : ℝ),
    (x + 1720 = 2795) →
    (x * (3 / 100) * 8 = 1720 * (5 / 100) * y) →
    y = 3 :=
by
  intros x y hsum heq
  sorry

end interest_years_eq_three_l344_34411


namespace oliver_baths_per_week_l344_34437

-- Define all the conditions given in the problem
def bucket_capacity : ℕ := 120
def num_buckets_to_fill_tub : ℕ := 14
def num_buckets_removed : ℕ := 3
def weekly_water_usage : ℕ := 9240

-- Calculate total water to fill bathtub, water removed, water used per bath, and baths per week
def total_tub_capacity : ℕ := num_buckets_to_fill_tub * bucket_capacity
def water_removed : ℕ := num_buckets_removed * bucket_capacity
def water_per_bath : ℕ := total_tub_capacity - water_removed
def baths_per_week : ℕ := weekly_water_usage / water_per_bath

theorem oliver_baths_per_week : baths_per_week = 7 := by
  sorry

end oliver_baths_per_week_l344_34437


namespace percentage_error_in_area_l344_34443

noncomputable def side_with_error (s : ℝ) : ℝ := 1.04 * s

noncomputable def actual_area (s : ℝ) : ℝ := s ^ 2

noncomputable def calculated_area (s : ℝ) : ℝ := (side_with_error s) ^ 2

noncomputable def percentage_error (actual : ℝ) (calculated : ℝ) : ℝ :=
  ((calculated - actual) / actual) * 100

theorem percentage_error_in_area (s : ℝ) :
  percentage_error (actual_area s) (calculated_area s) = 8.16 := by
  sorry

end percentage_error_in_area_l344_34443


namespace math_problem_l344_34496

noncomputable def f : ℝ → ℝ := sorry

theorem math_problem (h_decreasing : ∀ x y : ℝ, 2 < x → x < y → f y < f x)
  (h_even : ∀ x : ℝ, f (-x + 2) = f (x + 2)) :
  f 2 < f 3 ∧ f 3 < f 0 ∧ f 0 < f (-1) :=
by
  sorry

end math_problem_l344_34496


namespace least_value_y_l344_34461

theorem least_value_y
  (h : ∀ y : ℝ, 5 * y ^ 2 + 7 * y + 3 = 6 → -3 ≤ y) : 
  ∃ y : ℝ, 5 * y ^ 2 + 7 * y + 3 = 6 ∧ y = -3 :=
by
  sorry

end least_value_y_l344_34461


namespace payment_to_N_l344_34460

variable (x : ℝ)

/-- Conditions stating the total payment and the relationship between M and N's payment --/
axiom total_payment : x + 1.20 * x = 550

/-- Statement to prove the amount paid to N per week --/
theorem payment_to_N : x = 250 :=
by
  sorry

end payment_to_N_l344_34460


namespace olympic_triathlon_total_distance_l344_34407

theorem olympic_triathlon_total_distance (x : ℝ) (L S : ℝ)
  (hL : L = 4 * x)
  (hS : S = (3 / 80) * x)
  (h_diff : L - S = 8.5) :
  x + L + S = 51.5 := by
  sorry

end olympic_triathlon_total_distance_l344_34407


namespace min_value_fraction_l344_34454

theorem min_value_fraction (x : ℝ) (h : x > 0) : ∃ y, y = 4 ∧ (∀ z, z = (x + 5) / Real.sqrt (x + 1) → y ≤ z) := sorry

end min_value_fraction_l344_34454


namespace total_lives_l344_34485

theorem total_lives (initial_friends : ℕ) (initial_lives_per_friend : ℕ) (additional_players : ℕ) (lives_per_new_player : ℕ) :
  initial_friends = 7 →
  initial_lives_per_friend = 7 →
  additional_players = 2 →
  lives_per_new_player = 7 →
  (initial_friends * initial_lives_per_friend + additional_players * lives_per_new_player) = 63 :=
by
  intros
  sorry

end total_lives_l344_34485


namespace unique_integral_solution_l344_34422

noncomputable def positiveInt (x : ℤ) : Prop := x > 0

theorem unique_integral_solution (m n : ℤ) (hm : positiveInt m) (hn : positiveInt n) (unique_sol : ∃! (x y : ℤ), x + y^2 = m ∧ x^2 + y = n) : 
  ∃ (k : ℕ), m - n = 2^k ∨ m - n = -2^k :=
sorry

end unique_integral_solution_l344_34422


namespace trig_expression_eval_l344_34430

open Real

-- Declare the main theorem
theorem trig_expression_eval (θ : ℝ) (k : ℤ) 
  (h : sin (θ + k * π) = -2 * cos (θ + k * π)) :
  (4 * sin θ - 2 * cos θ) / (5 * cos θ + 3 * sin θ) = 10 :=
  sorry

end trig_expression_eval_l344_34430


namespace compute_cos_2_sum_zero_l344_34495

theorem compute_cos_2_sum_zero (x y z : ℝ)
  (h1 : Real.cos (x + Real.pi / 4) + Real.cos (y + Real.pi / 4) + Real.cos (z + Real.pi / 4) = 0)
  (h2 : Real.sin (x + Real.pi / 4) + Real.sin (y + Real.pi / 4) + Real.sin (z + Real.pi / 4) = 0) :
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 0 :=
by
  sorry

end compute_cos_2_sum_zero_l344_34495


namespace find_2023rd_letter_l344_34466

def seq : List Char := ['A', 'B', 'C', 'D', 'D', 'C', 'B', 'A']

theorem find_2023rd_letter : seq.get! ((2023 % seq.length) - 1) = 'B' :=
by
  sorry

end find_2023rd_letter_l344_34466


namespace water_evaporation_problem_l344_34468

theorem water_evaporation_problem 
  (W : ℝ) 
  (evaporation_rate : ℝ := 0.01) 
  (evaporation_days : ℝ := 20) 
  (total_evaporation : ℝ := evaporation_rate * evaporation_days) 
  (evaporation_percentage : ℝ := 0.02) 
  (evaporation_amount : ℝ := evaporation_percentage * W) :
  evaporation_amount = total_evaporation → W = 10 :=
by
  sorry

end water_evaporation_problem_l344_34468


namespace shopkeeper_discount_problem_l344_34452

theorem shopkeeper_discount_problem (CP SP_with_discount SP_without_discount Discount : ℝ)
  (h1 : SP_with_discount = CP + 0.273 * CP)
  (h2 : SP_without_discount = CP + 0.34 * CP) :
  Discount = SP_without_discount - SP_with_discount →
  (Discount / SP_without_discount) * 100 = 5 := 
sorry

end shopkeeper_discount_problem_l344_34452


namespace eliza_height_is_68_l344_34440

-- Define the known heights of the siblings
def height_sibling_1 : ℕ := 66
def height_sibling_2 : ℕ := 66
def height_sibling_3 : ℕ := 60

-- The total height of all 5 siblings combined
def total_height : ℕ := 330

-- Eliza is 2 inches shorter than the last sibling
def height_difference : ℕ := 2

-- Define the heights of the siblings
def height_remaining_siblings := total_height - (height_sibling_1 + height_sibling_2 + height_sibling_3)

-- The height of the last sibling
def height_last_sibling := (height_remaining_siblings + height_difference) / 2

-- Eliza's height
def height_eliza := height_last_sibling - height_difference

-- We need to prove that Eliza's height is 68 inches
theorem eliza_height_is_68 : height_eliza = 68 := by
  sorry

end eliza_height_is_68_l344_34440


namespace trigonometric_comparison_l344_34427

noncomputable def a : ℝ := Real.sin (3 * Real.pi / 5)
noncomputable def b : ℝ := Real.cos (2 * Real.pi / 5)
noncomputable def c : ℝ := Real.tan (2 * Real.pi / 5)

theorem trigonometric_comparison :
  b < a ∧ a < c :=
by {
  -- Use necessary steps to demonstrate b < a and a < c
  sorry
}

end trigonometric_comparison_l344_34427


namespace cost_of_one_dozen_pens_l344_34410

noncomputable def cost_of_one_pen_and_one_pencil_ratio := 5

theorem cost_of_one_dozen_pens
  (cost_pencil : ℝ)
  (cost_3_pens_5_pencils : 3 * (cost_of_one_pen_and_one_pencil_ratio * cost_pencil) + 5 * cost_pencil = 200) :
  12 * (cost_of_one_pen_and_one_pencil_ratio * cost_pencil) = 600 :=
by
  sorry

end cost_of_one_dozen_pens_l344_34410


namespace largest_integer_x_l344_34401

theorem largest_integer_x (x : ℤ) : (8:ℚ)/11 > (x:ℚ)/15 → x ≤ 10 :=
by
  intro h
  sorry

end largest_integer_x_l344_34401


namespace c_10_eq_3_pow_89_l344_34444

section sequence
  open Nat

  -- Define the sequence c
  def c : ℕ → ℕ
  | 0     => 3  -- Note: Typically Lean sequences start from 0, not 1
  | 1     => 9
  | (n+2) => c n.succ * c n

  -- Define the auxiliary sequence d
  def d : ℕ → ℕ
  | 0     => 1  -- Note: Typically Lean sequences start from 0, not 1
  | 1     => 2
  | (n+2) => d n.succ + d n

  -- The theorem we need to prove
  theorem c_10_eq_3_pow_89 : c 9 = 3 ^ d 9 :=    -- Note: c_{10} in the original problem is c(9) in Lean
  sorry   -- Proof omitted
end sequence

end c_10_eq_3_pow_89_l344_34444


namespace cyclist_speed_l344_34424

theorem cyclist_speed 
  (course_length : ℝ)
  (second_cyclist_speed : ℝ)
  (meeting_time : ℝ)
  (total_distance : ℝ)
  (condition1 : course_length = 45)
  (condition2 : second_cyclist_speed = 16)
  (condition3 : meeting_time = 1.5)
  (condition4 : total_distance = meeting_time * (second_cyclist_speed + 14))
  : (meeting_time * 14 + meeting_time * second_cyclist_speed = course_length) :=
by
  sorry

end cyclist_speed_l344_34424


namespace andrew_age_l344_34416

variables (a g : ℝ)

theorem andrew_age (h1 : g = 15 * a) (h2 : g - a = 60) : a = 30 / 7 :=
by sorry

end andrew_age_l344_34416


namespace min_abs_sum_l344_34439

theorem min_abs_sum : ∃ x : ℝ, (|x + 1| + |x + 2| + |x + 6|) = 5 :=
sorry

end min_abs_sum_l344_34439


namespace S_equals_x4_l344_34492

-- Define the expression for S
def S (x : ℝ) : ℝ := (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * x - 3

-- State the theorem to be proved
theorem S_equals_x4 (x : ℝ) : S x = x^4 :=
by
  sorry

end S_equals_x4_l344_34492


namespace exists_integer_a_l344_34448

theorem exists_integer_a (p : ℕ) (hp : p ≥ 5) [Fact (Nat.Prime p)] : 
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧ (¬ p^2 ∣ a^(p-1) - 1) ∧ (¬ p^2 ∣ (a+1)^(p-1) - 1) :=
by
  sorry

end exists_integer_a_l344_34448


namespace Nickel_ate_3_chocolates_l344_34456

-- Definitions of the conditions
def Robert_chocolates : ℕ := 12
def extra_chocolates : ℕ := 9
def Nickel_chocolates (N : ℕ) : Prop := Robert_chocolates = N + extra_chocolates

-- The proof goal
theorem Nickel_ate_3_chocolates : ∃ N : ℕ, Nickel_chocolates N ∧ N = 3 :=
by
  sorry

end Nickel_ate_3_chocolates_l344_34456


namespace trapezium_perimeters_l344_34405

theorem trapezium_perimeters (AB BC AD AF : ℝ)
  (h1 : AB = 30) (h2 : BC = 30) (h3 : AD = 25) (h4 : AF = 24) :
  ∃ p : ℝ, (p = 90 ∨ p = 104) :=
by
  sorry

end trapezium_perimeters_l344_34405


namespace quadratic_solution_l344_34426

def quadratic_rewrite (x b c : ℝ) : ℝ := (x + b) * (x + b) + c

theorem quadratic_solution (b c : ℝ)
  (h1 : ∀ x, x^2 + 2100 * x + 4200 = quadratic_rewrite x b c)
  (h2 : c = -b^2 + 4200) :
  c / b = -1034 :=
by
  sorry

end quadratic_solution_l344_34426


namespace seq_inequality_l344_34432

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 6 ∧ ∀ n, n > 3 → a n = 3 * a (n - 1) - a (n - 2) - 2 * a (n - 3)

theorem seq_inequality (a : ℕ → ℕ) (h : seq a) : ∀ n, n > 3 → a n > 3 * 2 ^ (n - 2) :=
  sorry

end seq_inequality_l344_34432


namespace minimum_knights_l344_34470

-- Definitions based on the conditions
def total_people := 1001
def is_knight (person : ℕ) : Prop := sorry -- Assume definition of knight
def is_liar (person : ℕ) : Prop := sorry    -- Assume definition of liar

-- Conditions
axiom next_to_each_knight_is_liar : ∀ (p : ℕ), is_knight p → is_liar (p + 1) ∨ is_liar (p - 1)
axiom next_to_each_liar_is_knight : ∀ (p : ℕ), is_liar p → is_knight (p + 1) ∨ is_knight (p - 1)

-- Proving the minimum number of knights
theorem minimum_knights : ∃ (k : ℕ), k ≤ total_people ∧ k ≥ 502 ∧ (∀ (n : ℕ), n ≥ k → is_knight n) :=
  sorry

end minimum_knights_l344_34470


namespace find_a_range_l344_34409

noncomputable def f (x : ℝ) := (x - 1) / Real.exp x

noncomputable def condition_holds (a : ℝ) : Prop :=
∀ t ∈ (Set.Icc (1/2 : ℝ) 2), f t > t

theorem find_a_range (a : ℝ) (h : condition_holds a) : a > Real.exp 2 + 1/2 := sorry

end find_a_range_l344_34409


namespace derivative_f_cos2x_l344_34453

variable {f : ℝ → ℝ} {x : ℝ}

theorem derivative_f_cos2x :
  f (Real.cos (2 * x)) = 1 - 2 * (Real.sin x) ^ 2 →
  deriv f x = -2 * Real.sin (2 * x) :=
by sorry

end derivative_f_cos2x_l344_34453


namespace max_c_l344_34488

theorem max_c (c : ℝ) : 
  (∀ x y : ℝ, x > y ∧ y > 0 → x^2 - 2 * y^2 ≤ c * x * (y - x)) 
  → c ≤ 2 * Real.sqrt 2 - 4 := 
by
  sorry

end max_c_l344_34488


namespace circle_center_and_radius_l344_34457

theorem circle_center_and_radius:
  ∀ x y : ℝ, 
  (x + 1) ^ 2 + (y - 3) ^ 2 = 36 
  → ∃ C : (ℝ × ℝ), C = (-1, 3) ∧ ∃ r : ℝ, r = 6 := sorry

end circle_center_and_radius_l344_34457


namespace ratio_of_Patrick_to_Joseph_l344_34499

def countries_traveled_by_George : Nat := 6
def countries_traveled_by_Joseph : Nat := countries_traveled_by_George / 2
def countries_traveled_by_Zack : Nat := 18
def countries_traveled_by_Patrick : Nat := countries_traveled_by_Zack / 2

theorem ratio_of_Patrick_to_Joseph : countries_traveled_by_Patrick / countries_traveled_by_Joseph = 3 :=
by
  -- The definition conditions have already been integrated above
  sorry

end ratio_of_Patrick_to_Joseph_l344_34499


namespace min_value_fraction_l344_34442

theorem min_value_fraction (x : ℝ) (h : x > 6) : 
  (∃ x_min, x_min = 12 ∧ (∀ x > 6, (x * x) / (x - 6) ≥ 18) ∧ (x * x) / (x - 6) = 18) :=
sorry

end min_value_fraction_l344_34442


namespace speed_equation_l344_34498

theorem speed_equation
  (dA dB : ℝ)
  (sB : ℝ)
  (sA : ℝ)
  (time_difference : ℝ)
  (h1 : dA = 800)
  (h2 : dB = 400)
  (h3 : sA = 1.2 * sB)
  (h4 : time_difference = 4) :
  (dA / sA - dB / sB = time_difference) :=
by
  sorry

end speed_equation_l344_34498


namespace percentage_reduction_l344_34403

theorem percentage_reduction (original reduced : ℝ) (h_original : original = 253.25) (h_reduced : reduced = 195) : 
  ((original - reduced) / original) * 100 = 22.99 :=
by
  sorry

end percentage_reduction_l344_34403


namespace factorize_expression_l344_34404

theorem factorize_expression (x y : ℝ) : x^2 + x * y + x = x * (x + y + 1) := 
by
  sorry

end factorize_expression_l344_34404


namespace no_real_roots_iff_no_positive_discriminant_l344_34431

noncomputable def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

theorem no_real_roots_iff_no_positive_discriminant (m : ℝ) 
  (h : discriminant m (-2*(m+2)) (m+5) < 0) : 
  (discriminant (m-5) (-2*(m+2)) m < 0 ∨ discriminant (m-5) (-2*(m+2)) m > 0 ∨ m - 5 = 0) :=
by 
  sorry

end no_real_roots_iff_no_positive_discriminant_l344_34431


namespace rotary_club_extra_omelets_l344_34487

theorem rotary_club_extra_omelets
  (small_children_tickets : ℕ)
  (older_children_tickets : ℕ)
  (adult_tickets : ℕ)
  (senior_tickets : ℕ)
  (eggs_total : ℕ)
  (omelet_for_small_child : ℝ)
  (omelet_for_older_child : ℝ)
  (omelet_for_adult : ℝ)
  (omelet_for_senior : ℝ)
  (eggs_per_omelet : ℕ)
  (extra_omelets : ℕ) :
  small_children_tickets = 53 →
  older_children_tickets = 35 →
  adult_tickets = 75 →
  senior_tickets = 37 →
  eggs_total = 584 →
  omelet_for_small_child = 0.5 →
  omelet_for_older_child = 1 →
  omelet_for_adult = 2 →
  omelet_for_senior = 1.5 →
  eggs_per_omelet = 2 →
  extra_omelets = (eggs_total - (2 * (small_children_tickets * omelet_for_small_child +
                                      older_children_tickets * omelet_for_older_child +
                                      adult_tickets * omelet_for_adult +
                                      senior_tickets * omelet_for_senior))) / eggs_per_omelet →
  extra_omelets = 25 :=
by
  intros hsmo_hold hsoc_hold hat_hold hsnt_hold htot_hold
        hosm_hold hocc_hold hact_hold hsen_hold hepom_hold hres_hold
  sorry

end rotary_club_extra_omelets_l344_34487


namespace division_problem_l344_34489

theorem division_problem :
  0.045 / 0.0075 = 6 :=
sorry

end division_problem_l344_34489


namespace seunghwa_express_bus_distance_per_min_l344_34420

noncomputable def distance_per_min_on_express_bus (total_distance : ℝ) (total_time : ℝ) (time_on_general : ℝ) (gasoline_general : ℝ) (distance_per_gallon : ℝ) (gasoline_used : ℝ) : ℝ :=
  let distance_general := (gasoline_used * distance_per_gallon) / gasoline_general
  let distance_express := total_distance - distance_general
  let time_express := total_time - time_on_general
  (distance_express / time_express)

theorem seunghwa_express_bus_distance_per_min :
  distance_per_min_on_express_bus 120 110 (70) 6 (40.8) 14 = 0.62 :=
by
  sorry

end seunghwa_express_bus_distance_per_min_l344_34420


namespace max_ab_l344_34475

theorem max_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 4 * b = 1) : ab ≤ 1 / 16 :=
sorry

end max_ab_l344_34475


namespace square_area_from_circle_area_l344_34478

variable (square_area : ℝ) (circle_area : ℝ)

theorem square_area_from_circle_area 
  (h1 : circle_area = 9 * Real.pi) 
  (h2 : square_area = (2 * Real.sqrt (circle_area / Real.pi))^2) : 
  square_area = 36 := 
by
  sorry

end square_area_from_circle_area_l344_34478


namespace total_cost_paid_l344_34463

-- Definition of the given conditions
def number_of_DVDs : ℕ := 4
def cost_per_DVD : ℝ := 1.2

-- The theorem to be proven
theorem total_cost_paid : number_of_DVDs * cost_per_DVD = 4.8 := by
  sorry

end total_cost_paid_l344_34463


namespace polar_line_eq_l344_34479

theorem polar_line_eq (ρ θ : ℝ) : (ρ * Real.cos θ = 1) ↔ (ρ = Real.cos θ ∨ ρ = Real.sin θ ∨ 1 / Real.cos θ = ρ) := by
  sorry

end polar_line_eq_l344_34479


namespace sum_lent_eq_1100_l344_34493

def interest_rate : ℚ := 6 / 100

def period : ℕ := 8

def interest_amount (P : ℚ) : ℚ :=
  period * interest_rate * P

def total_interest_eq_principal_minus_572 (P: ℚ) : Prop :=
  interest_amount P = P - 572

theorem sum_lent_eq_1100 : ∃ P : ℚ, total_interest_eq_principal_minus_572 P ∧ P = 1100 :=
by
  use 1100
  sorry

end sum_lent_eq_1100_l344_34493


namespace perimeter_of_square_land_is_36_diagonal_of_square_land_is_27_33_l344_34459

def square_land (A P D : ℝ) :=
  (5 * A = 10 * P + 45) ∧
  (3 * D = 2 * P + 10)

theorem perimeter_of_square_land_is_36 (A P D : ℝ) (h1 : 5 * A = 10 * P + 45) (h2 : 3 * D = 2 * P + 10) :
  P = 36 :=
sorry

theorem diagonal_of_square_land_is_27_33 (A P D : ℝ) (h1 : P = 36) (h2 : 3 * D = 2 * P + 10) :
  D = 82 / 3 :=
sorry

end perimeter_of_square_land_is_36_diagonal_of_square_land_is_27_33_l344_34459


namespace evaluate_Q_at_2_and_neg2_l344_34418

-- Define the polynomial Q and the conditions
variable {Q : ℤ → ℤ}
variable {m : ℤ}

-- The given conditions
axiom cond1 : Q 0 = m
axiom cond2 : Q 1 = 3 * m
axiom cond3 : Q (-1) = 4 * m

-- The proof goal
theorem evaluate_Q_at_2_and_neg2 : Q 2 + Q (-2) = 22 * m :=
sorry

end evaluate_Q_at_2_and_neg2_l344_34418


namespace probability_event_a_without_replacement_independence_of_events_with_replacement_l344_34450

open ProbabilityTheory MeasureTheory Set

-- Definitions corresponding to the conditions
def BallLabeled (i : ℕ) : Prop := i ∈ Finset.range 10

def EventA (second_ball : ℕ) : Prop := second_ball = 2

def EventB (first_ball second_ball : ℕ) (m : ℕ) : Prop := first_ball + second_ball = m

-- First Part: Probability without replacement
theorem probability_event_a_without_replacement :
  ∃ P_A : ℝ, P_A = 1 / 10 := sorry

-- Second Part: Independence with replacement
theorem independence_of_events_with_replacement (m : ℕ) :
  (EventA 2 → (∀ first_ball : ℕ, BallLabeled first_ball → EventB first_ball 2 m) ↔ m = 9) := sorry

end probability_event_a_without_replacement_independence_of_events_with_replacement_l344_34450


namespace n_times_2pow_nplus1_plus_1_is_square_l344_34469

theorem n_times_2pow_nplus1_plus_1_is_square (n : ℕ) (h : 0 < n) :
  ∃ m : ℤ, n * 2 ^ (n + 1) + 1 = m * m ↔ n = 3 := 
by
  sorry

end n_times_2pow_nplus1_plus_1_is_square_l344_34469


namespace total_wheels_l344_34414

def regular_bikes := 7
def children_bikes := 11
def tandem_bikes_4_wheels := 5
def tandem_bikes_6_wheels := 3
def unicycles := 4
def tricycles := 6
def bikes_with_training_wheels := 8

def wheels_regular := 2
def wheels_children := 4
def wheels_tandem_4 := 4
def wheels_tandem_6 := 6
def wheel_unicycle := 1
def wheels_tricycle := 3
def wheels_training := 4

theorem total_wheels : 
  (regular_bikes * wheels_regular) +
  (children_bikes * wheels_children) + 
  (tandem_bikes_4_wheels * wheels_tandem_4) + 
  (tandem_bikes_6_wheels * wheels_tandem_6) + 
  (unicycles * wheel_unicycle) + 
  (tricycles * wheels_tricycle) + 
  (bikes_with_training_wheels * wheels_training) 
  = 150 := by
  sorry

end total_wheels_l344_34414


namespace num_license_plates_l344_34449

-- Let's state the number of letters in the alphabet, vowels, consonants, and digits.
def num_letters : ℕ := 26
def num_vowels : ℕ := 5  -- A, E, I, O, U and Y is not a vowel
def num_consonants : ℕ := 21  -- Remaining letters including Y
def num_digits : ℕ := 10  -- 0 through 9

-- Prove the number of five-character license plates
theorem num_license_plates : 
  (num_consonants * num_consonants * num_vowels * num_vowels * num_digits) = 110250 :=
  by 
  sorry

end num_license_plates_l344_34449


namespace initial_shells_l344_34481

theorem initial_shells (x : ℕ) (h : x + 23 = 28) : x = 5 :=
by
  sorry

end initial_shells_l344_34481


namespace ascending_function_k_ge_2_l344_34473

open Real

def is_ascending (f : ℝ → ℝ) (k : ℝ) (M : Set ℝ) : Prop :=
  ∀ x ∈ M, f (x + k) ≥ f x

theorem ascending_function_k_ge_2 :
  ∀ (k : ℝ), (∀ x : ℝ, x ≥ -1 → (x + k) ^ 2 ≥ x ^ 2) → k ≥ 2 :=
by
  intros k h
  sorry

end ascending_function_k_ge_2_l344_34473


namespace theta_in_fourth_quadrant_l344_34412

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin (2 * θ) < 0) : 
  (∃ k : ℤ, θ = 2 * π * k + 7 * π / 4 ∨ θ = 2 * π * k + π / 4) ∧ θ = 2 * π * k + 7 * π / 4 :=
sorry

end theta_in_fourth_quadrant_l344_34412


namespace friends_cant_go_to_movies_l344_34402

theorem friends_cant_go_to_movies (total_friends : ℕ) (friends_can_go : ℕ) (H1 : total_friends = 15) (H2 : friends_can_go = 8) : (total_friends - friends_can_go) = 7 :=
by
  sorry

end friends_cant_go_to_movies_l344_34402


namespace power_of_fraction_to_decimal_l344_34480

theorem power_of_fraction_to_decimal : ∃ x : ℕ, (1 / 9 : ℚ) ^ x = 1 / 81 ∧ x = 2 :=
by
  use 2
  simp
  sorry

end power_of_fraction_to_decimal_l344_34480


namespace last_even_distribution_l344_34494

theorem last_even_distribution (n : ℕ) (h : n = 590490) :
  ∃ k : ℕ, (k ≤ n ∧ (n = 3^k + 3^k + 3^k) ∧ (∀ m : ℕ, m < k → ¬(n = 3^m + 3^m + 3^m))) ∧ k = 1 := 
by 
  sorry

end last_even_distribution_l344_34494


namespace drum_oil_capacity_l344_34483

theorem drum_oil_capacity (C : ℝ) (Y : ℝ) 
  (hX : DrumX_Oil = 0.5 * C) 
  (hY : DrumY_Cap = 2 * C) 
  (hY_filled : Y + 0.5 * C = 0.65 * (2 * C)) :
  Y = 0.8 * C :=
by
  sorry

end drum_oil_capacity_l344_34483


namespace loss_percentage_is_75_l344_34408

-- Given conditions
def cost_price_one_book (C : ℝ) : Prop := C > 0
def selling_price_one_book (S : ℝ) : Prop := S > 0
def cost_price_5_equals_selling_price_20 (C S : ℝ) : Prop := 5 * C = 20 * S

-- Proof goal
theorem loss_percentage_is_75 (C S : ℝ) (h1 : cost_price_one_book C) (h2 : selling_price_one_book S) (h3 : cost_price_5_equals_selling_price_20 C S) : 
  ((C - S) / C) * 100 = 75 :=
by
  sorry

end loss_percentage_is_75_l344_34408


namespace first_year_after_2020_with_digit_sum_18_l344_34484

theorem first_year_after_2020_with_digit_sum_18 : 
  ∃ (y : ℕ), y > 2020 ∧ (∃ a b c : ℕ, (2 + a + b + c = 18 ∧ y = 2000 + 100 * a + 10 * b + c)) ∧ y = 2799 := 
sorry

end first_year_after_2020_with_digit_sum_18_l344_34484


namespace remainder_modulo_seven_l344_34451

theorem remainder_modulo_seven (n : ℕ)
  (h₁ : n^2 % 7 = 1)
  (h₂ : n^3 % 7 = 6) :
  n % 7 = 6 :=
sorry

end remainder_modulo_seven_l344_34451


namespace fraction_simplification_l344_34413

theorem fraction_simplification :
  (1/2 * 1/3 * 1/4 * 1/5 + 3/2 * 3/4 * 3/5) / (1/2 * 2/3 * 2/5) = 41/8 :=
by
  sorry

end fraction_simplification_l344_34413


namespace perfect_square_trinomial_l344_34436

theorem perfect_square_trinomial (a b c : ℤ) (f : ℤ → ℤ) (h : ∀ x : ℤ, f x = a * x^2 + b * x + c) :
  ∃ d e : ℤ, ∀ x : ℤ, f x = (d * x + e) ^ 2 :=
sorry

end perfect_square_trinomial_l344_34436


namespace proof_ab_greater_ac_l344_34433

theorem proof_ab_greater_ac (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : 
  a * b > a * c :=
by sorry

end proof_ab_greater_ac_l344_34433


namespace joshua_share_is_30_l344_34446

-- Definitions based on the conditions
def total_amount_shared : ℝ := 40
def ratio_joshua_justin : ℝ := 3

-- Proposition to prove
theorem joshua_share_is_30 (J : ℝ) (Joshua_share : ℝ) :
  J + ratio_joshua_justin * J = total_amount_shared → 
  Joshua_share = ratio_joshua_justin * J → 
  Joshua_share = 30 :=
sorry

end joshua_share_is_30_l344_34446


namespace Mario_savings_percentage_l344_34497

theorem Mario_savings_percentage 
  (P : ℝ) -- Normal price of a single ticket 
  (h_campaign : 5 * P = 3 * P) -- Campaign condition: 5 tickets for the price of 3
  : (2 * P) / (5 * P) * 100 = 40 := 
by
  -- Below this, we would write the actual automated proof, but we leave it as sorry.
  sorry

end Mario_savings_percentage_l344_34497


namespace intersection_point_l344_34455

theorem intersection_point (a b d x y : ℝ) (h1 : a = b + d) (h2 : a * x + b * y = b + 2 * d) :
    (x, y) = (-1, 1) :=
by
  sorry

end intersection_point_l344_34455


namespace largest_even_whole_number_l344_34476

theorem largest_even_whole_number (x : ℕ) (h1 : 9 * x < 150) (h2 : x % 2 = 0) : x ≤ 16 :=
by
  sorry

end largest_even_whole_number_l344_34476


namespace fred_earnings_over_weekend_l344_34406

-- Fred's earning from delivering newspapers
def earnings_from_newspapers : ℕ := 16

-- Fred's earning from washing cars
def earnings_from_cars : ℕ := 74

-- Fred's total earnings over the weekend
def total_earnings : ℕ := earnings_from_newspapers + earnings_from_cars

-- Proof that total earnings is 90
theorem fred_earnings_over_weekend : total_earnings = 90 :=
by 
  -- sorry statement to skip the proof steps
  sorry

end fred_earnings_over_weekend_l344_34406


namespace find_general_term_find_sum_of_b_l344_34421

variables {n : ℕ} (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Given conditions
axiom a5 : a 5 = 10
axiom S7 : S 7 = 56

-- Definition of S (Sum of first n terms of an arithmetic sequence)
def S_def (a : ℕ → ℕ) (n : ℕ) : ℕ := n * (a 1 + a n) / 2

-- Definition of the arithmetic sequence
def a_arith_seq (n : ℕ) : ℕ := 2 * n

-- Assuming the axiom for the arithmetic sequence sum
axiom S_is_arith : S 7 = S_def a 7

theorem find_general_term : a = a_arith_seq := 
by sorry

-- Sequence b
def b (n : ℕ) : ℕ := 2 + 9 ^ n

-- Sum of first n terms of sequence b
def T (n : ℕ) : ℕ := (Finset.range n).sum b

-- Prove T_n formula
theorem find_sum_of_b : ∀ n, T n = 2 * n + 9 / 8 * (9 ^ n - 1) :=
by sorry

end find_general_term_find_sum_of_b_l344_34421


namespace oak_trees_initial_count_l344_34458

theorem oak_trees_initial_count (x : ℕ) (cut_down : ℕ) (remaining : ℕ) (h_cut : cut_down = 2) (h_remaining : remaining = 7)
  (h_equation : (x - cut_down) = remaining) : x = 9 := by
  -- We are given that cut_down = 2
  -- and remaining = 7
  -- and we need to show that the initial count x = 9
  sorry

end oak_trees_initial_count_l344_34458


namespace sum_of_angles_x_y_l344_34474

theorem sum_of_angles_x_y :
  let num_arcs := 15
  let angle_per_arc := 360 / num_arcs
  let central_angle_x := 3 * angle_per_arc
  let central_angle_y := 5 * angle_per_arc
  let inscribed_angle (central_angle : ℝ) := central_angle / 2
  let angle_x := inscribed_angle central_angle_x
  let angle_y := inscribed_angle central_angle_y
  angle_x + angle_y = 96 := 
  sorry

end sum_of_angles_x_y_l344_34474


namespace abs_diff_of_solutions_eq_5_point_5_l344_34471

theorem abs_diff_of_solutions_eq_5_point_5 (x y : ℝ)
  (h1 : ⌊x⌋ + (y - ⌊y⌋) = 3.7)
  (h2 : (x - ⌊x⌋) + ⌊y⌋ = 8.2) :
  |x - y| = 5.5 :=
sorry

end abs_diff_of_solutions_eq_5_point_5_l344_34471


namespace ducks_in_the_marsh_l344_34486

-- Define the conditions
def number_of_geese : ℕ := 58
def total_number_of_birds : ℕ := 95
def number_of_ducks : ℕ := total_number_of_birds - number_of_geese

-- Prove the conclusion
theorem ducks_in_the_marsh : number_of_ducks = 37 := by
  -- subtraction to find number_of_ducks
  sorry

end ducks_in_the_marsh_l344_34486


namespace overall_percent_decrease_l344_34434

theorem overall_percent_decrease (trouser_price_italy : ℝ) (jacket_price_italy : ℝ) 
(trouser_price_uk : ℝ) (trouser_discount_uk : ℝ) (jacket_price_uk : ℝ) 
(jacket_discount_uk : ℝ) (exchange_rate : ℝ) 
(h1 : trouser_price_italy = 200) (h2 : jacket_price_italy = 150) 
(h3 : trouser_price_uk = 150) (h4 : trouser_discount_uk = 0.20) 
(h5 : jacket_price_uk = 120) (h6 : jacket_discount_uk = 0.30) 
(h7 : exchange_rate = 0.85) : 
((trouser_price_italy + jacket_price_italy) - 
 ((trouser_price_uk * (1 - trouser_discount_uk) / exchange_rate) + 
 (jacket_price_uk * (1 - jacket_discount_uk) / exchange_rate))) / 
 (trouser_price_italy + jacket_price_italy) * 100 = 31.43 := 
by 
  sorry

end overall_percent_decrease_l344_34434


namespace truck_tank_capacity_l344_34441

-- Definitions based on conditions
def truck_tank (T : ℝ) : Prop := true
def car_tank : Prop := true
def truck_half_full (T : ℝ) : Prop := true
def car_third_full : Prop := true
def add_fuel (T : ℝ) : Prop := T / 2 + 8 = 18

-- Theorem statement
theorem truck_tank_capacity (T : ℝ) (ht : truck_tank T) (hc : car_tank) 
  (ht_half : truck_half_full T) (hc_third : car_third_full) (hf_add : add_fuel T) : T = 20 :=
  sorry

end truck_tank_capacity_l344_34441


namespace range_of_d_l344_34423

variable {S : ℕ → ℝ} -- S is the sum of the series
variable {a : ℕ → ℝ} -- a is the arithmetic sequence

theorem range_of_d (d : ℝ) (h1 : a 3 = 12) (h2 : S 12 > 0) (h3 : S 13 < 0) :
  -24 / 7 < d ∧ d < -3 := sorry

end range_of_d_l344_34423


namespace integer_solutions_l344_34477

theorem integer_solutions :
  { (x, y) : ℤ × ℤ | x^2 = 1 + 4 * y^3 * (y + 2) } = {(1, 0), (1, -2), (-1, 0), (-1, -2)} :=
by
  sorry

end integer_solutions_l344_34477


namespace second_number_is_three_l344_34400

theorem second_number_is_three (x y : ℝ) (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) : y = 3 :=
by
  -- To be proved: sorry for now
  sorry

end second_number_is_three_l344_34400
