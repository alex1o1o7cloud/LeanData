import Mathlib

namespace vanya_correct_answers_l54_5434

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end vanya_correct_answers_l54_5434


namespace amoeba_doubling_time_l54_5479

theorem amoeba_doubling_time (H1 : ∀ t : ℕ, t = 60 → 2^(t / 3) = 2^20) :
  ∀ t : ℕ, 2 * 2^(t / 3) = 2^20 → t = 57 :=
by
  intro t
  intro H2
  sorry

end amoeba_doubling_time_l54_5479


namespace simplify_expression_l54_5431

variable (a b c : ℝ)

theorem simplify_expression :
  (-32 * a^4 * b^5 * c) / ((-2 * a * b)^3) * (-3 / 4 * a * c) = -3 * a^2 * b^2 * c^2 :=
  by
    sorry

end simplify_expression_l54_5431


namespace soccer_club_girls_l54_5488

theorem soccer_club_girls (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : (1 / 3 : ℚ) * G + B = 18) : 
  G = 18 := 
  by sorry

end soccer_club_girls_l54_5488


namespace arithmetic_seq_problem_l54_5451

theorem arithmetic_seq_problem (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a n = a 1 + (n - 1) * d)
  (h_cond : a 1 + 3 * a 8 + a 15 = 60) :
  2 * a 9 - a 10 = 12 := 
sorry

end arithmetic_seq_problem_l54_5451


namespace weekly_earnings_l54_5429

theorem weekly_earnings :
  let hours_Monday := 2
  let minutes_Tuesday := 75
  let start_Thursday := (15, 10) -- 3:10 PM in (hour, minute) format
  let end_Thursday := (17, 45) -- 5:45 PM in (hour, minute) format
  let minutes_Saturday := 45

  let pay_rate_weekday := 4 -- \$4 per hour
  let pay_rate_weekend := 5 -- \$5 per hour

  -- Convert time to hours
  let hours_Tuesday := minutes_Tuesday / 60.0
  let Thursday_work_minutes := (end_Thursday.1 * 60 + end_Thursday.2) - (start_Thursday.1 * 60 + start_Thursday.2)
  let hours_Thursday := Thursday_work_minutes / 60.0
  let hours_Saturday := minutes_Saturday / 60.0

  -- Calculate earnings
  let earnings_Monday := hours_Monday * pay_rate_weekday
  let earnings_Tuesday := hours_Tuesday * pay_rate_weekday
  let earnings_Thursday := hours_Thursday * pay_rate_weekday
  let earnings_Saturday := hours_Saturday * pay_rate_weekend

  -- Total earnings
  let total_earnings := earnings_Monday + earnings_Tuesday + earnings_Thursday + earnings_Saturday

  total_earnings = 27.08 := by sorry

end weekly_earnings_l54_5429


namespace inverse_function_property_l54_5423

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a ^ x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_property (a : ℝ) (h : g a 2 = 4) : f a 2 = 1 := by
  have g_inverse_f : g a (f a 2) = 2 := by sorry
  have a_value : a = 2 := by sorry
  rw [a_value]
  sorry

end inverse_function_property_l54_5423


namespace additional_people_needed_l54_5411

theorem additional_people_needed
  (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ)
  (h_initial : initial_people * initial_time = 24)
  (h_time : new_time = 2)
  (h_initial_people : initial_people = 8)
  (h_initial_time : initial_time = 3) :
  (24 / new_time) - initial_people = 4 :=
by
  sorry

end additional_people_needed_l54_5411


namespace unique_real_solution_l54_5480

-- Define the variables
variables (x y : ℝ)

-- State the condition
def equation (x y : ℝ) : Prop :=
  (2^(4*x + 2)) * (4^(2*x + 3)) = (8^(3*x + 4)) * y

-- State the theorem
theorem unique_real_solution (y : ℝ) (h_y : 0 < y) : ∃! x : ℝ, equation x y :=
sorry

end unique_real_solution_l54_5480


namespace sarahs_packages_l54_5424

def num_cupcakes_before : ℕ := 60
def num_cupcakes_ate : ℕ := 22
def cupcakes_per_package : ℕ := 10

theorem sarahs_packages : (num_cupcakes_before - num_cupcakes_ate) / cupcakes_per_package = 3 :=
by
  sorry

end sarahs_packages_l54_5424


namespace prob_yellow_is_3_over_5_required_red_balls_is_8_l54_5478

-- Defining the initial conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 4
def yellow_balls : ℕ := 6

-- Part 1: Prove the probability of drawing a yellow ball is 3/5
theorem prob_yellow_is_3_over_5 :
  (yellow_balls : ℚ) / (total_balls : ℚ) = 3 / 5 := sorry

-- Part 2: Prove that adding 8 red balls makes the probability of drawing a red ball 2/3
theorem required_red_balls_is_8 (x : ℕ) :
  (red_balls + x : ℚ) / (total_balls + x : ℚ) = 2 / 3 → x = 8 := sorry

end prob_yellow_is_3_over_5_required_red_balls_is_8_l54_5478


namespace ab_value_l54_5445

theorem ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 30) (h4 : 2 * a * b + 12 * a = 3 * b + 240) :
  a * b = 255 :=
sorry

end ab_value_l54_5445


namespace solve_m_l54_5465

theorem solve_m (m : ℝ) : 
  (m - 3) * x^2 - 3 * x + m^2 = 9 → m^2 - 9 = 0 → m = -3 :=
by
  sorry

end solve_m_l54_5465


namespace angle_in_second_quadrant_l54_5426

open Real

-- Define the fourth quadrant condition
def isFourthQuadrant (α : ℝ) (k : ℤ) : Prop :=
  2 * k * π - π / 2 < α ∧ α < 2 * k * π

-- Define the second quadrant condition
def isSecondQuadrant (β : ℝ) (k : ℤ) : Prop :=
  2 * k * π + π / 2 < β ∧ β < 2 * k * π + π

-- The main theorem to prove
theorem angle_in_second_quadrant (α : ℝ) (k : ℤ) :
  isFourthQuadrant α k → isSecondQuadrant (π + α) k :=
sorry

end angle_in_second_quadrant_l54_5426


namespace apex_angle_of_quadrilateral_pyramid_l54_5461

theorem apex_angle_of_quadrilateral_pyramid :
  ∃ (α : ℝ), α = Real.arccos ((Real.sqrt 5 - 1) / 2) :=
sorry

end apex_angle_of_quadrilateral_pyramid_l54_5461


namespace ratio_area_rectangle_triangle_l54_5409

noncomputable def area_rectangle (L W : ℝ) : ℝ :=
  L * W

noncomputable def area_triangle (L W : ℝ) : ℝ :=
  (1 / 2) * L * W

theorem ratio_area_rectangle_triangle (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  area_rectangle L W / area_triangle L W = 2 :=
by
  -- sorry will be replaced by the actual proof
  sorry

end ratio_area_rectangle_triangle_l54_5409


namespace quad_common_root_l54_5443

theorem quad_common_root (a b c d : ℝ) :
  (∃ α : ℝ, α^2 + a * α + b = 0 ∧ α^2 + c * α + d = 0) ↔ (a * d - b * c) * (c - a) = (b - d)^2 ∧ (a ≠ c) := 
sorry

end quad_common_root_l54_5443


namespace cashier_adjustment_l54_5447

-- Define the conditions
variables {y : ℝ}

-- Error calculation given the conditions
def half_dollar_error (y : ℝ) : ℝ := 0.50 * y
def five_dollar_error (y : ℝ) : ℝ := 5 * y
def total_error (y : ℝ) : ℝ := half_dollar_error y + five_dollar_error y

-- Theorem statement
theorem cashier_adjustment (y : ℝ) : total_error y = 5.50 * y :=
sorry

end cashier_adjustment_l54_5447


namespace flask_forces_l54_5487

theorem flask_forces (r : ℝ) (ρ g h_A h_B h_C V : ℝ) (A : ℝ) (FA FB FC : ℝ) (h1 : r = 2)
  (h2 : A = π * r^2)
  (h3 : V = A * h_A ∧ V = A * h_B ∧ V = A * h_C)
  (h4 : FC = ρ * g * h_C * A)
  (h5 : FA = ρ * g * h_A * A)
  (h6 : FB = ρ * g * h_B * A)
  (h7 : h_C > h_A ∧ h_A > h_B) : FC > FA ∧ FA > FB := 
sorry

end flask_forces_l54_5487


namespace cylinder_dimensions_l54_5484

theorem cylinder_dimensions (r_sphere : ℝ) (r_cylinder h d : ℝ)
  (h_d_eq : h = d) (r_sphere_val : r_sphere = 6) 
  (sphere_area_eq : 4 * Real.pi * r_sphere^2 = 2 * Real.pi * r_cylinder * h) :
  h = 12 ∧ d = 12 :=
by 
  sorry

end cylinder_dimensions_l54_5484


namespace original_plan_months_l54_5496

theorem original_plan_months (x : ℝ) (h : 1 / (x - 6) = 1.4 * (1 / x)) : x = 21 :=
by
  sorry

end original_plan_months_l54_5496


namespace add_base8_numbers_l54_5490

def fromBase8 (n : Nat) : Nat :=
  Nat.digits 8 n |> Nat.ofDigits 8

theorem add_base8_numbers : 
  fromBase8 356 + fromBase8 672 + fromBase8 145 = fromBase8 1477 :=
by
  sorry

end add_base8_numbers_l54_5490


namespace andrew_stickers_now_l54_5425

-- Defining the conditions
def total_stickers : Nat := 1500
def ratio_susan : Nat := 1
def ratio_andrew : Nat := 1
def ratio_sam : Nat := 3
def total_ratio : Nat := ratio_susan + ratio_andrew + ratio_sam
def part : Nat := total_stickers / total_ratio
def susan_share : Nat := ratio_susan * part
def andrew_share_initial : Nat := ratio_andrew * part
def sam_share : Nat := ratio_sam * part
def sam_to_andrew : Nat := (2 * sam_share) / 3

-- Andrew's final stickers count
def andrew_share_final : Nat :=
  andrew_share_initial + sam_to_andrew

-- The theorem to prove
theorem andrew_stickers_now : andrew_share_final = 900 :=
by
  -- Proof would go here
  sorry

end andrew_stickers_now_l54_5425


namespace quadratic_solution_property_l54_5402

theorem quadratic_solution_property (p q : ℝ)
  (h : ∀ x, 2 * x^2 + 8 * x - 42 = 0 → x = p ∨ x = q) :
  (p - q + 2) ^ 2 = 144 :=
sorry

end quadratic_solution_property_l54_5402


namespace total_marbles_l54_5459

theorem total_marbles (jars clay_pots total_marbles jars_marbles pots_marbles : ℕ)
  (h1 : jars = 16)
  (h2 : jars = 2 * clay_pots)
  (h3 : jars_marbles = 5)
  (h4 : pots_marbles = 3 * jars_marbles)
  (h5 : total_marbles = jars * jars_marbles + clay_pots * pots_marbles) :
  total_marbles = 200 := by
  sorry

end total_marbles_l54_5459


namespace oldest_child_age_l54_5489

def arithmeticProgression (a d : ℕ) (n : ℕ) : ℕ := 
  a + (n - 1) * d

theorem oldest_child_age (a : ℕ) (d : ℕ) (n : ℕ) 
  (average : (arithmeticProgression a d 1 + arithmeticProgression a d 2 + arithmeticProgression a d 3 + arithmeticProgression a d 4 + arithmeticProgression a d 5) / 5 = 10)
  (distinct : ∀ i j, i ≠ j → arithmeticProgression a d i ≠ arithmeticProgression a d j)
  (constant_difference : d = 3) :
  arithmeticProgression a d 5 = 16 :=
by
  sorry

end oldest_child_age_l54_5489


namespace correct_calculation_l54_5421

theorem correct_calculation (x : ℤ) (h1 : x + 65 = 125) : x + 95 = 155 :=
by sorry

end correct_calculation_l54_5421


namespace domain_of_h_l54_5427

noncomputable def h (x : ℝ) : ℝ := (x^4 - 5 * x + 6) / (|x - 4| + |x + 2| - 1)

theorem domain_of_h : ∀ x : ℝ, |x - 4| + |x + 2| - 1 ≠ 0 := by
  intro x
  sorry

end domain_of_h_l54_5427


namespace original_cost_price_l54_5416

theorem original_cost_price (C : ℝ) : 
  (0.89 * C * 1.20 = 54000) → C = 50561.80 :=
by
  sorry

end original_cost_price_l54_5416


namespace satisify_absolute_value_inequality_l54_5415

theorem satisify_absolute_value_inequality :
  ∃ (t : Finset ℤ), t.card = 2 ∧ ∀ y ∈ t, |7 * y + 4| ≤ 10 :=
by
  sorry

end satisify_absolute_value_inequality_l54_5415


namespace find_tire_price_l54_5452

def regular_price_of_tire (x : ℝ) : Prop :=
  3 * x + 0.75 * x = 270

theorem find_tire_price (x : ℝ) (h1 : regular_price_of_tire x) : x = 72 :=
by
  sorry

end find_tire_price_l54_5452


namespace daisies_given_l54_5474

theorem daisies_given (S : ℕ) (h : (5 + S) / 2 = 7) : S = 9 := by
  sorry

end daisies_given_l54_5474


namespace find_five_digit_number_l54_5481

theorem find_five_digit_number :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∃ rev_n : ℕ, rev_n = (n % 10) * 10000 + (n / 10 % 10) * 1000 + (n / 100 % 10) * 100 + (n / 1000 % 10) * 10 + (n / 10000) ∧ 9 * n = rev_n) ∧ n = 10989 :=
  sorry

end find_five_digit_number_l54_5481


namespace log_sum_nine_l54_5435

-- Define that {a_n} is a geometric sequence and satisfies the given conditions.
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a n = a 1 * r ^ (n - 1)

-- Given conditions
axiom a_pos (a : ℕ → ℝ) : (∀ n, a n > 0)      -- All terms are positive
axiom a2a8_eq_4 (a : ℕ → ℝ) : a 2 * a 8 = 4    -- a₂a₈ = 4

theorem log_sum_nine (a : ℕ → ℝ) 
  (geo_seq : geometric_sequence a) 
  (pos : ∀ n, a n > 0)
  (eq4 : a 2 * a 8 = 4) :
  (Real.logb 2 (a 1) + Real.logb 2 (a 2) + Real.logb 2 (a 3) + Real.logb 2 (a 4)
  + Real.logb 2 (a 5) + Real.logb 2 (a 6) + Real.logb 2 (a 7) + Real.logb 2 (a 8)
  + Real.logb 2 (a 9)) = 9 :=
by
  sorry

end log_sum_nine_l54_5435


namespace elizabeth_net_profit_l54_5457

theorem elizabeth_net_profit :
  let cost_per_bag := 3.00
  let num_bags := 20
  let price_first_15_bags := 6.00
  let price_last_5_bags := 4.00
  let total_cost := cost_per_bag * num_bags
  let revenue_first_15 := 15 * price_first_15_bags
  let revenue_last_5 := 5 * price_last_5_bags
  let total_revenue := revenue_first_15 + revenue_last_5
  let net_profit := total_revenue - total_cost
  net_profit = 50.00 :=
by
  sorry

end elizabeth_net_profit_l54_5457


namespace percentage_increase_l54_5449

theorem percentage_increase (P : ℝ) (x : ℝ) 
(h1 : 1.17 * P = 0.90 * P * (1 + x / 100)) : x = 33.33 :=
by sorry

end percentage_increase_l54_5449


namespace apple_juice_production_l54_5498

noncomputable def apple_usage 
  (total_apples : ℝ) 
  (mixed_percentage : ℝ) 
  (juice_percentage : ℝ) 
  (sold_fresh_percentage : ℝ) : ℝ := 
  let mixed_apples := total_apples * mixed_percentage / 100
  let remainder_apples := total_apples - mixed_apples
  let juice_apples := remainder_apples * juice_percentage / 100
  juice_apples

theorem apple_juice_production :
  apple_usage 6 20 60 40 = 2.9 := 
by
  sorry

end apple_juice_production_l54_5498


namespace seashell_count_l54_5486

theorem seashell_count (Sam Mary Lucy : Nat) (h1 : Sam = 18) (h2 : Mary = 47) (h3 : Lucy = 32) : 
  Sam + Mary + Lucy = 97 :=
by 
  sorry

end seashell_count_l54_5486


namespace jamal_books_remaining_l54_5405

variable (initial_books : ℕ := 51)
variable (history_books : ℕ := 12)
variable (fiction_books : ℕ := 19)
variable (children_books : ℕ := 8)
variable (misplaced_books : ℕ := 4)

theorem jamal_books_remaining : 
  initial_books - history_books - fiction_books - children_books + misplaced_books = 16 := by
  sorry

end jamal_books_remaining_l54_5405


namespace find_positive_integer_N_l54_5440

theorem find_positive_integer_N (N : ℕ) (h₁ : 33^2 * 55^2 = 15^2 * N^2) : N = 121 :=
by {
  sorry
}

end find_positive_integer_N_l54_5440


namespace men_dropped_out_l54_5418

theorem men_dropped_out (x : ℕ) : 
  (∀ (days_half days_full men men_remaining : ℕ),
    days_half = 15 ∧ days_full = 25 ∧ men = 5 ∧ men_remaining = men - x ∧ 
    (men * (2 * days_half)) = ((men_remaining) * days_full)) -> x = 1 :=
by
  intros h
  sorry

end men_dropped_out_l54_5418


namespace value_of_Z_4_3_l54_5473

def Z (a b : ℤ) : ℤ := a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3

theorem value_of_Z_4_3 : Z 4 3 = 1 := by
  sorry

end value_of_Z_4_3_l54_5473


namespace find_integer_triples_l54_5468

theorem find_integer_triples (x y z : ℤ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 2003 ↔ 
  (x = 668 ∧ y = 668 ∧ z = 667) ∨ 
  (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
  (x = 667 ∧ y = 668 ∧ z = 668) :=
by sorry

end find_integer_triples_l54_5468


namespace determine_machines_in_first_group_l54_5420

noncomputable def machines_in_first_group (x r : ℝ) : Prop :=
  (x * r * 6 = 1) ∧ (12 * r * 4 = 1)

theorem determine_machines_in_first_group (x r : ℝ) (h : machines_in_first_group x r) :
  x = 8 :=
by
  sorry

end determine_machines_in_first_group_l54_5420


namespace avg_weight_a_b_l54_5422

theorem avg_weight_a_b (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 60)
  (h2 : (B + C) / 2 = 50)
  (h3 : B = 60) :
  (A + B) / 2 = 70 := 
sorry

end avg_weight_a_b_l54_5422


namespace projected_revenue_increase_is_20_percent_l54_5476

noncomputable def projected_percentage_increase_of_revenue (R : ℝ) (actual_revenue : ℝ) (projected_revenue : ℝ) : ℝ :=
  (projected_revenue / R - 1) * 100

theorem projected_revenue_increase_is_20_percent (R : ℝ) (actual_revenue : ℝ) :
  actual_revenue = R * 0.75 →
  actual_revenue = (R * (1 + 20 / 100)) * 0.625 →
  projected_percentage_increase_of_revenue R ((R * (1 + 20 / 100))) = 20 :=
by
  intros h1 h2
  sorry

end projected_revenue_increase_is_20_percent_l54_5476


namespace value_of_a_minus_b_l54_5439

theorem value_of_a_minus_b (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 13) (h3 : a * b > 0) : a - b = -10 ∨ a - b = 10 :=
sorry

end value_of_a_minus_b_l54_5439


namespace fraction_of_shaded_circle_l54_5412

theorem fraction_of_shaded_circle (total_regions shaded_regions : ℕ) (h1 : total_regions = 4) (h2 : shaded_regions = 1) :
  shaded_regions / total_regions = 1 / 4 := by
  sorry

end fraction_of_shaded_circle_l54_5412


namespace remainder_3_pow_1000_mod_7_l54_5410

theorem remainder_3_pow_1000_mod_7 : 3 ^ 1000 % 7 = 4 := by
  sorry

end remainder_3_pow_1000_mod_7_l54_5410


namespace factor_polynomial_l54_5400

-- Define the polynomial expression
def polynomial (x : ℝ) : ℝ := 60 * x + 45 + 9 * x ^ 2

-- Define the factored form of the polynomial
def factored_form (x : ℝ) : ℝ := 3 * (3 * x + 5) * (x + 3)

-- The statement of the problem to prove equivalence of the forms
theorem factor_polynomial : ∀ x : ℝ, polynomial x = factored_form x :=
by
  -- The actual proof is omitted and replaced by sorry
  sorry

end factor_polynomial_l54_5400


namespace speed_of_current_l54_5403

theorem speed_of_current (c r : ℝ) 
  (h1 : 12 = (c - r) * 6) 
  (h2 : 12 = (c + r) * 0.75) : 
  r = 7 := 
by
  sorry

end speed_of_current_l54_5403


namespace eval_expression_l54_5417

theorem eval_expression : 15 * 30 + 45 * 15 - 15 * 10 = 975 :=
by 
  sorry

end eval_expression_l54_5417


namespace vehicles_count_l54_5494

theorem vehicles_count (T : ℕ) : 
    2 * T + 3 * (2 * T) + (T / 2) + T = 180 → 
    T = 19 ∧ 2 * T = 38 ∧ 3 * (2 * T) = 114 ∧ (T / 2) = 9 := 
by 
    intros h
    sorry

end vehicles_count_l54_5494


namespace find_n_that_makes_vectors_collinear_l54_5485

theorem find_n_that_makes_vectors_collinear (n : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, n)) (h_collinear : ∃ k : ℝ, 2 • a - b = k • b) : n = 9 :=
sorry

end find_n_that_makes_vectors_collinear_l54_5485


namespace zoey_holidays_in_a_year_l54_5438

-- Definitions based on the conditions
def holidays_per_month := 2
def months_in_year := 12

-- Lean statement representing the proof problem
theorem zoey_holidays_in_a_year : (holidays_per_month * months_in_year) = 24 :=
by sorry

end zoey_holidays_in_a_year_l54_5438


namespace consecutive_numbers_count_l54_5456

theorem consecutive_numbers_count (n : ℕ) 
(avg : ℝ) 
(largest : ℕ) 
(h_avg : avg = 20) 
(h_largest : largest = 23) 
(h_eq : (largest + (largest - (n - 1))) / 2 = avg) : 
n = 7 := 
by 
  sorry

end consecutive_numbers_count_l54_5456


namespace remaining_volume_of_cube_l54_5444

theorem remaining_volume_of_cube :
  let s := 6
  let r := 3
  let h := 6
  let V_cube := s^3
  let V_cylinder := Real.pi * (r^2) * h
  V_cube - V_cylinder = 216 - 54 * Real.pi :=
by
  sorry

end remaining_volume_of_cube_l54_5444


namespace sphere_volume_in_cone_l54_5491

theorem sphere_volume_in_cone :
  let d := 24
  let theta := 90
  let r := 24 * (Real.sqrt 2 - 1)
  let V := (4 / 3) * Real.pi * r^3
  ∃ (R : ℝ), r = R ∧ V = (4 / 3) * Real.pi * R^3 := by
  sorry

end sphere_volume_in_cone_l54_5491


namespace a_star_b_value_l54_5493

theorem a_star_b_value (a b : ℤ) (h1 : a + b = 12) (h2 : a * b = 32) (h3 : b = 8) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 3 / 8 := by
sorry

end a_star_b_value_l54_5493


namespace recurring_decimal_division_l54_5460

noncomputable def recurring_decimal_fraction : ℚ :=
  let frac_81 := (81 : ℚ) / 99
  let frac_36 := (36 : ℚ) / 99
  frac_81 / frac_36

theorem recurring_decimal_division :
  recurring_decimal_fraction = 9 / 4 :=
by
  sorry

end recurring_decimal_division_l54_5460


namespace shortest_distance_parabola_line_l54_5436

theorem shortest_distance_parabola_line :
  ∃ (P Q : ℝ × ℝ), P.2 = P.1^2 - 6 * P.1 + 15 ∧ Q.2 = 2 * Q.1 - 7 ∧
  ∀ (p q : ℝ × ℝ), p.2 = p.1^2 - 6 * p.1 + 15 → q.2 = 2 * q.1 - 7 → 
  dist p q ≥ dist P Q :=
sorry

end shortest_distance_parabola_line_l54_5436


namespace negation_of_p_l54_5432

theorem negation_of_p (p : Prop) :
  (¬ (∀ (a : ℝ), a ≥ 0 → a^4 + a^2 ≥ 0)) ↔ (∃ (a : ℝ), a ≥ 0 ∧ a^4 + a^2 < 0) := 
by
  sorry

end negation_of_p_l54_5432


namespace max_n_for_Sn_neg_l54_5419

noncomputable def Sn (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n * (a 1 + a n)) / 2

theorem max_n_for_Sn_neg (a : ℕ → ℝ) (h1 : ∀ n : ℕ, (n + 1) * Sn n a < n * Sn (n + 1) a)
  (h2 : a 8 / a 7 < -1) :
  ∀ n : ℕ, S_13 < 0 ∧ S_14 > 0 →
  ∀ m : ℕ, m > 13 → Sn m a ≥ 0 :=
sorry

end max_n_for_Sn_neg_l54_5419


namespace total_numbers_l54_5482

theorem total_numbers (N : ℕ) (sum_total : ℝ) (avg_total : ℝ) (avg1 : ℝ) (avg2 : ℝ) (avg3 : ℝ) :
  avg_total = 6.40 → avg1 = 6.2 → avg2 = 6.1 → avg3 = 6.9 →
  sum_total = 2 * avg1 + 2 * avg2 + 2 * avg3 →
  N = sum_total / avg_total →
  N = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_numbers_l54_5482


namespace percentage_students_with_same_grade_l54_5413

def total_students : ℕ := 50
def students_with_same_grade : ℕ := 3 + 6 + 8 + 2 + 1

theorem percentage_students_with_same_grade :
  (students_with_same_grade / total_students : ℚ) * 100 = 40 :=
by
  sorry

end percentage_students_with_same_grade_l54_5413


namespace rows_seating_l54_5462

theorem rows_seating (x y : ℕ) (h : 7 * x + 6 * y = 52) : x = 4 :=
by
  sorry

end rows_seating_l54_5462


namespace batsman_average_is_18_l54_5466
noncomputable def average_after_18_innings (score_18th: ℕ) (average_17th: ℕ) (innings: ℕ) : ℕ :=
  let total_runs_17 := average_17th * 17
  let total_runs_18 := total_runs_17 + score_18th
  total_runs_18 / innings

theorem batsman_average_is_18 {score_18th: ℕ} {average_17th: ℕ} {expected_average: ℕ} :
  score_18th = 1 → average_17th = 19 → expected_average = 18 →
  average_after_18_innings score_18th average_17th 18 = expected_average := by
  sorry

end batsman_average_is_18_l54_5466


namespace area_of_parallelogram_l54_5499

theorem area_of_parallelogram (b h : ℕ) (hb : b = 60) (hh : h = 16) : b * h = 960 := by
  -- Here goes the proof
  sorry

end area_of_parallelogram_l54_5499


namespace andrew_eggs_count_l54_5414

def cost_of_toast (num_toasts : ℕ) : ℕ :=
  num_toasts * 1

def cost_of_eggs (num_eggs : ℕ) : ℕ :=
  num_eggs * 3

def total_cost (num_toasts : ℕ) (num_eggs : ℕ) : ℕ :=
  cost_of_toast num_toasts + cost_of_eggs num_eggs

theorem andrew_eggs_count (E : ℕ) (H1 : total_cost 2 2 = 8)
                       (H2 : total_cost 1 E + 8 = 15) : E = 2 := by
  sorry

end andrew_eggs_count_l54_5414


namespace numPythagoreanTriples_l54_5446

def isPythagoreanTriple (x y z : ℕ) : Prop :=
  x < y ∧ y < z ∧ x^2 + y^2 = z^2

theorem numPythagoreanTriples (n : ℕ) : ∃! T : (ℕ × ℕ × ℕ) → Prop, 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (T (2^(n+1))) :=
sorry

end numPythagoreanTriples_l54_5446


namespace money_taken_l54_5483

def total_people : ℕ := 6
def cost_per_soda : ℝ := 0.5
def cost_per_pizza : ℝ := 1.0

theorem money_taken (total_people cost_per_soda cost_per_pizza : ℕ × ℝ × ℝ ) :
  total_people * cost_per_soda + total_people * cost_per_pizza = 9 := by
  sorry

end money_taken_l54_5483


namespace smallest_solution_x4_minus_40x2_plus_400_eq_zero_l54_5437

theorem smallest_solution_x4_minus_40x2_plus_400_eq_zero :
  ∃ x : ℝ, (x^4 - 40 * x^2 + 400 = 0) ∧ (∀ y : ℝ, (y^4 - 40 * y^2 + 400 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_40x2_plus_400_eq_zero_l54_5437


namespace certain_amount_l54_5428

theorem certain_amount (n : ℤ) (x : ℤ) : n = 5 ∧ 7 * n - 15 = 2 * n + x → x = 10 :=
by
  sorry

end certain_amount_l54_5428


namespace ratio_of_socks_l54_5464

-- Conditions:
variable (B : ℕ) (W : ℕ) (L : ℕ)
-- B = number of black socks
-- W = initial number of white socks
-- L = number of white socks lost

-- Setting given conditions:
axiom hB : B = 6
axiom hL : L = W / 2
axiom hCond : W / 2 = B + 6

-- Prove the ratio of white socks to black socks is 4:1
theorem ratio_of_socks : B = 6 → W / 2 = B + 6 → (W / 2) + (W / 2) = 24 → (B : ℚ) / (W : ℚ) = 1 / 4 :=
by intros hB hCond hW
   sorry

end ratio_of_socks_l54_5464


namespace exists_x_eq_28_l54_5495

theorem exists_x_eq_28 : ∃ x : Int, 45 - (x - (37 - (15 - 16))) = 55 ↔ x = 28 := 
by
  sorry

end exists_x_eq_28_l54_5495


namespace ice_cream_cost_l54_5433

variable {x F M : ℤ}

theorem ice_cream_cost (h1 : F = x - 7) (h2 : M = x - 1) (h3 : F + M < x) : x = 7 :=
by
  sorry

end ice_cream_cost_l54_5433


namespace problem_statement_l54_5477

-- Definitions for the given conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- The main statement that needs to be proved
theorem problem_statement (f : ℝ → ℝ) (h_odd : odd_function f) (h_monotone : monotone_decreasing f) : f (-1) > f 3 :=
by 
  sorry

end problem_statement_l54_5477


namespace sum_of_x_and_reciprocal_eq_3_5_l54_5401

theorem sum_of_x_and_reciprocal_eq_3_5
    (x : ℝ)
    (h : x^2 + (1 / x^2) = 10.25) :
    x + (1 / x) = 3.5 := 
by
  sorry

end sum_of_x_and_reciprocal_eq_3_5_l54_5401


namespace not_enough_pharmacies_l54_5407

theorem not_enough_pharmacies : 
  ∀ (n m : ℕ), n = 10 ∧ m = 10 →
  ∃ (intersections : ℕ), intersections = n * m ∧ 
  ∀ (d : ℕ), d = 3 →
  ∀ (coverage : ℕ), coverage = (2 * d + 1) * (2 * d + 1) →
  ¬ (coverage * 12 ≥ intersections * 2) :=
by sorry

end not_enough_pharmacies_l54_5407


namespace smallest_five_digit_divisible_by_53_and_3_l54_5448

/-- The smallest five-digit positive integer divisible by 53 and 3 is 10062 -/
theorem smallest_five_digit_divisible_by_53_and_3 : ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 53 = 0 ∧ n % 3 = 0 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 53 = 0 ∧ m % 3 = 0 → n ≤ m ∧ n = 10062 :=
by
  sorry

end smallest_five_digit_divisible_by_53_and_3_l54_5448


namespace print_shop_x_charge_l54_5472

theorem print_shop_x_charge :
  ∃ (x : ℝ), 60 * x + 90 = 60 * 2.75 ∧ x = 1.25 :=
by
  sorry

end print_shop_x_charge_l54_5472


namespace polynomial_roots_l54_5471

theorem polynomial_roots (α : ℝ) : 
  (α^2 + α - 1 = 0) → (α^3 - 2 * α + 1 = 0) :=
by sorry

end polynomial_roots_l54_5471


namespace geom_series_first_term_l54_5467

theorem geom_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = 1/4) (h2 : S = 80) (h3 : S = a / (1 - r)) : a = 60 :=
by
  sorry -- proof goes here

end geom_series_first_term_l54_5467


namespace find_radius_of_smaller_circles_l54_5450

noncomputable def smaller_circle_radius (r : ℝ) : Prop :=
  ∃ sin72 : ℝ, sin72 = Real.sin (72 * Real.pi / 180) ∧
  r = (2 * sin72) / (1 - sin72)

theorem find_radius_of_smaller_circles (r : ℝ) :
  (smaller_circle_radius r) ↔
  r = (2 * Real.sin (72 * Real.pi / 180)) / (1 - Real.sin (72 * Real.pi / 180)) :=
by
  sorry

end find_radius_of_smaller_circles_l54_5450


namespace largest_n_l54_5463

theorem largest_n (x y z n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12 → n ≤ 6 :=
by
  sorry

end largest_n_l54_5463


namespace area_of_triangle_PQR_l54_5455

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 2, y := 5 }

-- Define the lines using their slopes and the point P
def line1 (x : ℝ) : ℝ := -x + 7
def line2 (x : ℝ) : ℝ := -2 * x + 9

-- Definitions of points Q and R, which are the x-intercepts
def Q : Point := { x := 7, y := 0 }
def R : Point := { x := 4.5, y := 0 }

-- Theorem statement
theorem area_of_triangle_PQR : 
  let base := 7 - 4.5
  let height := 5
  (1 / 2) * base * height = 6.25 := by
  sorry

end area_of_triangle_PQR_l54_5455


namespace moving_circle_trajectory_l54_5475

-- Define the two given circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 169
def C₂ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 9

-- The theorem statement
theorem moving_circle_trajectory :
  (∀ x y : ℝ, (exists r : ℝ, r > 0 ∧ ∃ M : ℝ × ℝ, 
  (C₁ M.1 M.2 ∧ ((M.1 - 4)^2 + M.2^2 = (13 - r)^2) ∧
  C₂ M.1 M.2 ∧ ((M.1 + 4)^2 + M.2^2 = (r + 3)^2)) ∧
  ((x = M.1) ∧ (y = M.2))) ↔ (x^2 / 64 + y^2 / 48 = 1)) := sorry

end moving_circle_trajectory_l54_5475


namespace compare_solutions_l54_5406

variables (p q r s : ℝ)
variables (hp : p ≠ 0) (hr : r ≠ 0)

theorem compare_solutions :
  ((-q / p) > (-s / r)) ↔ (s * r > q * p) :=
by sorry

end compare_solutions_l54_5406


namespace min_value_of_abs_diff_l54_5470
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x

theorem min_value_of_abs_diff (x1 x2 x : ℝ) (h1 : f x1 ≤ f x) (h2: f x ≤ f x2) : |x1 - x2| = π := by
  sorry

end min_value_of_abs_diff_l54_5470


namespace remainder_when_divided_by_x_minus_2_l54_5404

def p (x : ℤ) : ℤ := x^5 + x^3 + x + 3

theorem remainder_when_divided_by_x_minus_2 :
  p 2 = 45 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l54_5404


namespace opera_house_rows_l54_5458

variable (R : ℕ)
variable (SeatsPerRow : ℕ)
variable (TicketPrice : ℕ)
variable (TotalEarnings : ℕ)
variable (SeatsTakenPercent : ℝ)

-- Conditions
axiom num_seats_per_row : SeatsPerRow = 10
axiom ticket_price : TicketPrice = 10
axiom total_earnings : TotalEarnings = 12000
axiom seats_taken_percent : SeatsTakenPercent = 0.8

-- Main theorem statement
theorem opera_house_rows
  (h1 : SeatsPerRow = 10)
  (h2 : TicketPrice = 10)
  (h3 : TotalEarnings = 12000)
  (h4 : SeatsTakenPercent = 0.8) :
  R = 150 :=
sorry

end opera_house_rows_l54_5458


namespace percentage_decrease_in_area_l54_5453

noncomputable def original_radius (r : ℝ) : ℝ := r
noncomputable def new_radius (r : ℝ) : ℝ := 0.5 * r
noncomputable def original_area (r : ℝ) : ℝ := Real.pi * r ^ 2
noncomputable def new_area (r : ℝ) : ℝ := Real.pi * (0.5 * r) ^ 2

theorem percentage_decrease_in_area (r : ℝ) (hr : 0 ≤ r) :
  ((original_area r - new_area r) / original_area r) * 100 = 75 :=
by
  sorry

end percentage_decrease_in_area_l54_5453


namespace bald_eagle_pairs_l54_5442

theorem bald_eagle_pairs (n_1963 : ℕ) (increase : ℕ) (h1 : n_1963 = 417) (h2 : increase = 6649) :
  (n_1963 + increase = 7066) :=
by
  sorry

end bald_eagle_pairs_l54_5442


namespace scout_troop_profit_l54_5430

-- Defining the basic conditions as Lean definitions
def num_bars : ℕ := 1500
def cost_rate : ℚ := 3 / 4 -- rate in dollars per bar
def sell_rate : ℚ := 2 / 3 -- rate in dollars per bar

-- Calculate total cost, total revenue, and profit
def total_cost : ℚ := num_bars * cost_rate
def total_revenue : ℚ := num_bars * sell_rate
def profit : ℚ := total_revenue - total_cost

-- The final theorem to be proved
theorem scout_troop_profit : profit = -125 := by
  sorry

end scout_troop_profit_l54_5430


namespace Jungkook_red_balls_count_l54_5454

-- Define the conditions
def red_balls_per_box : ℕ := 3
def boxes_Jungkook_has : ℕ := 2

-- Statement to prove
theorem Jungkook_red_balls_count : red_balls_per_box * boxes_Jungkook_has = 6 :=
by sorry

end Jungkook_red_balls_count_l54_5454


namespace prove_statement_II_l54_5441

variable (digit : ℕ)

def statement_I : Prop := (digit = 2)
def statement_II : Prop := (digit ≠ 3)
def statement_III : Prop := (digit = 5)
def statement_IV : Prop := (digit ≠ 6)

/- The main proposition that three statements are true and one is false. -/
def three_true_one_false (s1 s2 s3 s4 : Prop) : Prop :=
  (s1 ∧ s2 ∧ s3 ∧ ¬s4) ∨ (s1 ∧ s2 ∧ ¬s3 ∧ s4) ∨ 
  (s1 ∧ ¬s2 ∧ s3 ∧ s4) ∨ (¬s1 ∧ s2 ∧ s3 ∧ s4)

theorem prove_statement_II : 
  (three_true_one_false (statement_I digit) (statement_II digit) (statement_III digit) (statement_IV digit)) → 
  statement_II digit :=
sorry

end prove_statement_II_l54_5441


namespace parabola_above_line_l54_5408

variable (a b c : ℝ) (h : (b - c)^2 - 4 * a * c < 0)

theorem parabola_above_line : (b - c)^2 - 4 * a * c < 0 → (b - c)^2 - 4 * c * (a + b) < 0 :=
by sorry

end parabola_above_line_l54_5408


namespace parallelepiped_diagonal_inequality_l54_5469

theorem parallelepiped_diagonal_inequality 
  (a b c d : ℝ) 
  (h_d : d = Real.sqrt (a^2 + b^2 + c^2)) : 
  a^2 + b^2 + c^2 ≥ d^2 / 3 := 
by 
  sorry

end parallelepiped_diagonal_inequality_l54_5469


namespace spike_crickets_hunted_morning_l54_5497

def crickets_hunted_in_morning (C : ℕ) (total_daily_crickets : ℕ) : Prop :=
  4 * C = total_daily_crickets

theorem spike_crickets_hunted_morning (C : ℕ) (total_daily_crickets : ℕ) :
  total_daily_crickets = 20 → crickets_hunted_in_morning C total_daily_crickets → C = 5 :=
by
  intros h1 h2
  sorry

end spike_crickets_hunted_morning_l54_5497


namespace odd_function_domain_real_l54_5492

theorem odd_function_domain_real
  (a : ℤ)
  (h_condition : a = -1 ∨ a = 1 ∨ a = 3) :
  (∀ x : ℝ, ∃ y : ℝ, x ≠ 0 → y = x^a) →
  (∀ x : ℝ, x ≠ 0 → (x^a = (-x)^a)) →
  (a = 1 ∨ a = 3) :=
sorry

end odd_function_domain_real_l54_5492
