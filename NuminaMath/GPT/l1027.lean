import Mathlib

namespace solution_set_a_eq_1_no_positive_a_for_all_x_l1027_102726

-- Define the original inequality for a given a.
def inequality (a x : ℝ) : Prop := |a * x - 1| + |a * x - a| ≥ 2

-- Part 1: For a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | inequality 1 x } = {x : ℝ | x ≤ 0 ∨ x ≥ 2} :=
sorry

-- Part 2: There is no positive a such that the inequality holds for all x ∈ ℝ
theorem no_positive_a_for_all_x :
  ¬ ∃ a > 0, ∀ x : ℝ, inequality a x :=
sorry

end solution_set_a_eq_1_no_positive_a_for_all_x_l1027_102726


namespace range_of_a_l1027_102743

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 - 3 * a * x^2 + (2 * a + 1) * x

noncomputable def f' (a x : ℝ) : ℝ :=
  3 * x^2 - 6 * a * x + (2 * a + 1)

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f' a x = 0 ∧ ∀ y : ℝ, f' a y ≠ 0) →
  (a > 1 ∨ a < -1 / 3) :=
sorry

end range_of_a_l1027_102743


namespace find_base_of_denominator_l1027_102722

theorem find_base_of_denominator 
  (some_base : ℕ)
  (h1 : (1/2)^16 * (1/81)^8 = 1 / some_base^16) : 
  some_base = 18 :=
sorry

end find_base_of_denominator_l1027_102722


namespace mila_hours_to_match_agnes_monthly_earnings_l1027_102702

-- Definitions based on given conditions
def hourly_rate_mila : ℕ := 10
def hourly_rate_agnes : ℕ := 15
def weekly_hours_agnes : ℕ := 8
def weeks_in_month : ℕ := 4

-- Target statement to prove: Mila needs to work 48 hours to earn as much as Agnes in a month
theorem mila_hours_to_match_agnes_monthly_earnings :
  ∃ (h : ℕ), h = 48 ∧ (h * hourly_rate_mila) = (hourly_rate_agnes * weekly_hours_agnes * weeks_in_month) :=
by
  sorry

end mila_hours_to_match_agnes_monthly_earnings_l1027_102702


namespace turtle_population_2002_l1027_102794

theorem turtle_population_2002 (k : ℝ) (y : ℝ)
  (h1 : 58 + k * 92 = y)
  (h2 : 179 - 92 = k * y) 
  : y = 123 :=
by
  sorry

end turtle_population_2002_l1027_102794


namespace ladder_length_l1027_102708

variable (x y : ℝ)

theorem ladder_length :
  (x^2 = 15^2 + y^2) ∧ (x^2 = 24^2 + (y - 13)^2) → x = 25 := by
  sorry

end ladder_length_l1027_102708


namespace average_yield_per_tree_l1027_102742

theorem average_yield_per_tree :
  let t1 := 3
  let t2 := 2
  let t3 := 1
  let nuts1 := 60
  let nuts2 := 120
  let nuts3 := 180
  let total_nuts := t1 * nuts1 + t2 * nuts2 + t3 * nuts3
  let total_trees := t1 + t2 + t3
  let average_yield := total_nuts / total_trees
  average_yield = 100 := 
by
  sorry

end average_yield_per_tree_l1027_102742


namespace nearest_integer_to_sum_l1027_102785

theorem nearest_integer_to_sum (x y : ℝ) (h1 : |x| - y = 1) (h2 : |x| * y + x^2 = 2) : Int.ceil (x + y) = 2 :=
sorry

end nearest_integer_to_sum_l1027_102785


namespace find_max_marks_l1027_102772

theorem find_max_marks (M : ℝ) (h1 : 0.60 * M = 80 + 100) : M = 300 := 
by
  sorry

end find_max_marks_l1027_102772


namespace folding_positions_l1027_102759

theorem folding_positions (positions : Finset ℕ) (h_conditions: positions = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}) : 
  ∃ valid_positions : Finset ℕ, valid_positions = {1, 2, 3, 4, 9, 10, 11, 12} ∧ valid_positions.card = 8 :=
by
  sorry

end folding_positions_l1027_102759


namespace minimal_ab_l1027_102734

theorem minimal_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
(h : 1 / (a : ℝ) + 1 / (3 * b : ℝ) = 1 / 9) : a * b = 60 :=
sorry

end minimal_ab_l1027_102734


namespace Gwen_still_has_money_in_usd_l1027_102727

open Real

noncomputable def exchange_rate : ℝ := 0.85
noncomputable def usd_gift : ℝ := 5.00
noncomputable def eur_gift : ℝ := 20.00
noncomputable def usd_spent_on_candy : ℝ := 3.25
noncomputable def eur_spent_on_toy : ℝ := 5.50

theorem Gwen_still_has_money_in_usd :
  let eur_conversion_to_usd := eur_gift / exchange_rate
  let total_usd_received := usd_gift + eur_conversion_to_usd
  let usd_spent_on_toy := eur_spent_on_toy / exchange_rate
  let total_usd_spent := usd_spent_on_candy + usd_spent_on_toy
  total_usd_received - total_usd_spent = 18.81 :=
by
  sorry

end Gwen_still_has_money_in_usd_l1027_102727


namespace election_result_l1027_102778

theorem election_result (Vx Vy Vz : ℝ) (Pz : ℝ)
  (h1 : Vx = 3 * (Vx / 3)) (h2 : Vy = 2 * (Vy / 2)) (h3 : Vz = 1 * (Vz / 1))
  (h4 : 0.63 * (Vx + Vy + Vz) = 0.74 * Vx + 0.67 * Vy + Pz * Vz) :
  Pz = 0.22 :=
by
  -- proof steps would go here
  -- sorry to keep the proof incomplete
  sorry

end election_result_l1027_102778


namespace h_oplus_h_op_h_equals_h_l1027_102776

def op (x y : ℝ) : ℝ := x^3 - y

theorem h_oplus_h_op_h_equals_h (h : ℝ) : op h (op h h) = h := by
  sorry

end h_oplus_h_op_h_equals_h_l1027_102776


namespace first_number_percentage_of_second_l1027_102723

theorem first_number_percentage_of_second {X : ℝ} (H1 : ℝ) (H2 : ℝ) 
  (H1_def : H1 = 0.05 * X) (H2_def : H2 = 0.25 * X) : 
  (H1 / H2) * 100 = 20 :=
by
  sorry

end first_number_percentage_of_second_l1027_102723


namespace length_of_EC_l1027_102739

variable (AC : ℝ) (AB : ℝ) (CD : ℝ) (EC : ℝ)

def is_trapezoid (AB CD : ℝ) : Prop := AB = 3 * CD
def perimeter (AB CD AC : ℝ) : Prop := AB + CD + AC + (AC / 3) = 36

theorem length_of_EC
  (h1 : is_trapezoid AB CD)
  (h2 : AC = 18)
  (h3 : perimeter AB CD AC) :
  EC = 9 / 2 :=
  sorry

end length_of_EC_l1027_102739


namespace total_outcomes_l1027_102786

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of events
def num_events : ℕ := 3

-- Theorem statement: asserting the total number of different outcomes
theorem total_outcomes : num_students ^ num_events = 125 :=
by
  sorry

end total_outcomes_l1027_102786


namespace age_group_caloric_allowance_l1027_102784

theorem age_group_caloric_allowance
  (average_daily_allowance : ℕ)
  (daily_reduction : ℕ)
  (reduced_weekly_allowance : ℕ)
  (week_days : ℕ)
  (h1 : daily_reduction = 500)
  (h2 : week_days = 7)
  (h3 : reduced_weekly_allowance = 10500)
  (h4 : reduced_weekly_allowance = (average_daily_allowance - daily_reduction) * week_days) :
  average_daily_allowance = 2000 :=
sorry

end age_group_caloric_allowance_l1027_102784


namespace factorize_expression_l1027_102782

theorem factorize_expression (m n : ℤ) : 
  4 * m^2 * n - 4 * n^3 = 4 * n * (m + n) * (m - n) :=
by
  sorry

end factorize_expression_l1027_102782


namespace student_factor_l1027_102745

theorem student_factor (x : ℤ) : (121 * x - 138 = 104) → x = 2 :=
by
  intro h
  sorry

end student_factor_l1027_102745


namespace initial_mean_corrected_l1027_102704

theorem initial_mean_corrected
  (M : ℝ)
  (h : 30 * M + 10 = 30 * 140.33333333333334) :
  M = 140 :=
by
  sorry

end initial_mean_corrected_l1027_102704


namespace find_base_l1027_102762

theorem find_base (b : ℕ) (h : (3 * b + 2) ^ 2 = b ^ 3 + b + 4) : b = 8 :=
sorry

end find_base_l1027_102762


namespace terminating_decimals_count_l1027_102777

noncomputable def int_counts_terminating_decimals : ℕ :=
  let n_limit := 500
  let denominator := 2100
  Nat.floor (n_limit / 21)

theorem terminating_decimals_count :
  int_counts_terminating_decimals = 23 :=
by
  /- Proof will be here eventually -/
  sorry

end terminating_decimals_count_l1027_102777


namespace abs_has_min_at_zero_l1027_102737

def f (x : ℝ) : ℝ := abs x

theorem abs_has_min_at_zero : ∃ m, (∀ x : ℝ, f x ≥ m) ∧ f 0 = m := by
  sorry

end abs_has_min_at_zero_l1027_102737


namespace correct_result_l1027_102706

-- Define the original number
def original_number := 51 + 6

-- Define the correct calculation using multiplication
def correct_calculation (x : ℕ) : ℕ := x * 6

-- Theorem to prove the correct calculation
theorem correct_result : correct_calculation original_number = 342 := by
  -- Skip the actual proof steps
  sorry

end correct_result_l1027_102706


namespace natalie_needs_10_bushes_l1027_102724

-- Definitions based on the conditions
def bushes_to_containers (bushes : ℕ) := bushes * 10
def containers_to_zucchinis (containers : ℕ) := (containers * 3) / 4

-- The proof statement
theorem natalie_needs_10_bushes :
  ∃ bushes : ℕ, containers_to_zucchinis (bushes_to_containers bushes) ≥ 72 ∧ bushes = 10 :=
sorry

end natalie_needs_10_bushes_l1027_102724


namespace snake_length_difference_l1027_102780

theorem snake_length_difference :
  ∀ (jake_len penny_len : ℕ), 
    jake_len > penny_len →
    jake_len + penny_len = 70 →
    jake_len = 41 →
    jake_len - penny_len = 12 :=
by
  intros jake_len penny_len h1 h2 h3
  sorry

end snake_length_difference_l1027_102780


namespace loss_percentage_l1027_102775

-- Definitions of cost price (C) and selling price (S)
def cost_price : ℤ := sorry
def selling_price : ℤ := sorry

-- Given condition: Cost price of 40 articles equals selling price of 25 articles
axiom condition : 40 * cost_price = 25 * selling_price

-- Statement to prove: The merchant made a loss of 20%
theorem loss_percentage (C S : ℤ) (h : 40 * C = 25 * S) : 
  ((S - C) * 100) / C = -20 := 
sorry

end loss_percentage_l1027_102775


namespace parabola_symmetric_points_l1027_102796

theorem parabola_symmetric_points (a : ℝ) (x1 y1 x2 y2 m : ℝ) 
  (h_parabola : ∀ x, y = a * x^2)
  (h_a_pos : a > 0)
  (h_focus_directrix : 1 / (2 * a) = 1 / 4)
  (h_symmetric : y1 = a * x1^2 ∧ y2 = a * x2^2 ∧ ∃ m, y1 = m + (x1 - m))
  (h_product : x1 * x2 = -1 / 2) :
  m = 3 / 2 := 
sorry

end parabola_symmetric_points_l1027_102796


namespace sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100_l1027_102725

theorem sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100 : 
  (15^25 + 5^25) % 100 = 0 := 
by
  sorry

end sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100_l1027_102725


namespace parabola_ratio_l1027_102751

-- Define the conditions and question as a theorem statement
theorem parabola_ratio
  (V₁ V₃ : ℝ × ℝ)
  (F₁ F₃ : ℝ × ℝ)
  (hV₁ : V₁ = (0, 0))
  (hF₁ : F₁ = (0, 1/8))
  (hV₃ : V₃ = (0, -1/2))
  (hF₃ : F₃ = (0, -1/4)) :
  dist F₁ F₃ / dist V₁ V₃ = 3 / 4 :=
  by
  sorry

end parabola_ratio_l1027_102751


namespace sum_first_n_terms_of_arithmetic_sequence_l1027_102711

def arithmetic_sequence_sum (a1 d n: ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_first_n_terms_of_arithmetic_sequence :
  arithmetic_sequence_sum 2 2 n = n * (n + 1) / 2 :=
by sorry

end sum_first_n_terms_of_arithmetic_sequence_l1027_102711


namespace intersection_M_N_l1027_102770

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := { -1, 0, 1, 2, 3 }

theorem intersection_M_N : M ∩ N = {0, 1, 2} := 
by
  sorry

end intersection_M_N_l1027_102770


namespace function_g_l1027_102767

theorem function_g (g : ℝ → ℝ) (t : ℝ) :
  (∀ t, (20 * t - 14) = 2 * (g t) - 40) → (g t = 10 * t + 13) :=
by
  intro h
  have h1 : 20 * t - 14 = 2 * (g t) - 40 := h t
  sorry

end function_g_l1027_102767


namespace min_value_expression_l1027_102740

open Real

theorem min_value_expression 
  (a : ℝ) 
  (b : ℝ) 
  (hb : 0 < b) 
  (e : ℝ) 
  (he : e = 2.718281828459045) :
  ∃ x : ℝ, 
  (x = 2 * (1 - log 2)^2) ∧
  ∀ a b, 
    0 < b → 
    ((1 / 2) * exp a - log (2 * b))^2 + (a - b)^2 ≥ x :=
sorry

end min_value_expression_l1027_102740


namespace truthful_dwarfs_count_l1027_102781

def number_of_dwarfs := 10
def vanilla_ice_cream := number_of_dwarfs
def chocolate_ice_cream := number_of_dwarfs / 2
def fruit_ice_cream := 1

theorem truthful_dwarfs_count (T L : ℕ) (h1 : T + L = 10)
  (h2 : vanilla_ice_cream = T + (L * 2))
  (h3 : chocolate_ice_cream = T / 2 + (L / 2 * 2))
  (h4 : fruit_ice_cream = 1)
  : T = 4 :=
sorry

end truthful_dwarfs_count_l1027_102781


namespace odd_function_has_specific_a_l1027_102760

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = - f x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
x / ((2 * x + 1) * (x - a))

theorem odd_function_has_specific_a :
  ∀ a, is_odd (f a) → a = 1 / 2 :=
by sorry

end odd_function_has_specific_a_l1027_102760


namespace quadratic_inequality_solution_l1027_102729

theorem quadratic_inequality_solution (x : ℝ) : (2 * x^2 - 5 * x - 3 < 0) ↔ (-1/2 < x ∧ x < 3) :=
by
  sorry

end quadratic_inequality_solution_l1027_102729


namespace abs_x_minus_one_iff_x_in_interval_l1027_102753

theorem abs_x_minus_one_iff_x_in_interval (x : ℝ) :
  |x - 1| < 2 ↔ (x + 1) * (x - 3) < 0 :=
by
  sorry

end abs_x_minus_one_iff_x_in_interval_l1027_102753


namespace tangent_line_at_neg_ln_2_range_of_a_inequality_range_of_a_zero_point_l1027_102788

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

/-- Problem 1 -/
theorem tangent_line_at_neg_ln_2 :
  let x := -Real.log 2
  let y := f x
  ∃ k b : ℝ, (y - b) = k * (x - (-Real.log 2)) ∧ k = (Real.exp x - 1) ∧ b = Real.log 2 + 1/2 :=
sorry

/-- Problem 2 -/
theorem range_of_a_inequality :
  ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x > a * x) ↔ a ∈ Set.Iio (Real.exp 1 - 1) :=
sorry

/-- Problem 3 -/
theorem range_of_a_zero_point :
  ∀ a : ℝ, (∃! x : ℝ, f x - a * x = 0) ↔ a ∈ (Set.Iio (-1) ∪ Set.Ioi (Real.exp 1 - 1)) :=
sorry

end tangent_line_at_neg_ln_2_range_of_a_inequality_range_of_a_zero_point_l1027_102788


namespace geometric_sequence_product_bound_l1027_102701

theorem geometric_sequence_product_bound {a1 a2 a3 m q : ℝ} (h_sum : a1 + a2 + a3 = 3 * m) (h_m_pos : 0 < m) (h_q_pos : 0 < q) (h_geom : a1 = a2 / q ∧ a3 = a2 * q) : 
  0 < a1 * a2 * a3 ∧ a1 * a2 * a3 ≤ m^3 := 
sorry

end geometric_sequence_product_bound_l1027_102701


namespace no_rectangle_with_five_distinct_squares_no_rectangle_with_six_distinct_squares_l1027_102718

-- Part (a): Prove that it is impossible to arrange five distinct-sized squares to form a rectangle.
theorem no_rectangle_with_five_distinct_squares (s1 s2 s3 s4 s5 : ℕ) 
  (dist : s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧ s3 ≠ s4 ∧ s3 ≠ s5 ∧ s4 ≠ s5) :
  ¬ ∃ (l w : ℕ), (s1 ≤ l ∧ s1 ≤ w) ∧ (s2 ≤ l ∧ s2 ≤ w) ∧ (s3 ≤ l ∧ s3 ≤ w) ∧ (s4 ≤ l ∧ s4 ≤ w) ∧ (s5 ≤ l ∧ s5 ≤ w) ∧
  (l * w = (s1 + s2 + s3 + s4 + s5)) :=
by
  -- Proof placeholder
  sorry

-- Part (b): Prove that it is impossible to arrange six distinct-sized squares to form a rectangle.
theorem no_rectangle_with_six_distinct_squares (s1 s2 s3 s4 s5 s6 : ℕ) 
  (dist : s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧ s1 ≠ s6 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧ s2 ≠ s6 ∧ s3 ≠ s4 ∧ s3 ≠ s5 ∧ s3 ≠ s6 ∧ s4 ≠ s5 ∧ s4 ≠ s6 ∧ s5 ≠ s6) :
  ¬ ∃ (l w : ℕ), (s1 ≤ l ∧ s1 ≤ w) ∧ (s2 ≤ l ∧ s2 ≤ w) ∧ (s3 ≤ l ∧ s3 ≤ w) ∧ (s4 ≤ l ∧ s4 ≤ w) ∧ (s5 ≤ l ∧ s5 ≤ w) ∧ (s6 ≤ l ∧ s6 ≤ w) ∧
  (l * w = (s1 + s2 + s3 + s4 + s5 + s6)) :=
by
  -- Proof placeholder
  sorry

end no_rectangle_with_five_distinct_squares_no_rectangle_with_six_distinct_squares_l1027_102718


namespace find_n_l1027_102736

theorem find_n (n k : ℕ) (a b : ℝ) (h_pos : k > 0) (h_n : n ≥ 2) (h_ab_neq : a ≠ 0 ∧ b ≠ 0) (h_a : a = (k + 1) * b) : n = 2 * k + 2 :=
by sorry

end find_n_l1027_102736


namespace sum_transformed_roots_l1027_102709

theorem sum_transformed_roots :
  ∀ (a b c : ℝ),
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
  (45 * a^3 - 75 * a^2 + 33 * a - 2 = 0) →
  (45 * b^3 - 75 * b^2 + 33 * b - 2 = 0) →
  (45 * c^3 - 75 * c^2 + 33 * c - 2 = 0) →
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 60) :=
by
  intros a b c h_bounds h_poly_a h_poly_b h_poly_c h_distinct
  sorry

end sum_transformed_roots_l1027_102709


namespace students_and_swimmers_l1027_102703

theorem students_and_swimmers (N : ℕ) (x : ℕ) 
  (h1 : x = N / 4) 
  (h2 : x / 2 = 4) : 
  N = 32 ∧ N - x = 24 := 
by 
  sorry

end students_and_swimmers_l1027_102703


namespace walking_running_ratio_l1027_102744

theorem walking_running_ratio (d_w d_r : ℝ) (h1 : d_w / 4 + d_r / 8 = 3) (h2 : d_w + d_r = 16) :
  d_w / d_r = 1 := by
  sorry

end walking_running_ratio_l1027_102744


namespace min_value_l1027_102700

theorem min_value : ∀ (a b : ℝ), a + b^2 = 2 → (∀ x y : ℝ, x = a^2 + 6 * y^2 → y = b) → (∃ c : ℝ, c = 3) :=
by
  intros a b h₁ h₂
  sorry

end min_value_l1027_102700


namespace convex_100gon_distinct_numbers_l1027_102715

theorem convex_100gon_distinct_numbers :
  ∀ (vertices : Fin 100 → (ℕ × ℕ)),
  (∀ i, (vertices i).1 ≠ (vertices i).2) →
  ∃ (erase_one_number : ∀ (i : Fin 100), ℕ),
  (∀ i, erase_one_number i = (vertices i).1 ∨ erase_one_number i = (vertices i).2) ∧
  (∀ i j, i ≠ j → (i = j + 1 ∨ (i = 0 ∧ j = 99)) → erase_one_number i ≠ erase_one_number j) :=
by sorry

end convex_100gon_distinct_numbers_l1027_102715


namespace weight_ratio_l1027_102733

-- Conditions
def initial_weight : ℕ := 99
def initial_loss : ℕ := 12
def weight_added_back (x : ℕ) : Prop := x = 81 + 30 - initial_weight
def times_lost : ℕ := 3 * initial_loss
def final_gain : ℕ := 6
def final_weight : ℕ := 81

-- Question
theorem weight_ratio (x : ℕ)
  (H1 : weight_added_back x)
  (H2 : initial_weight - initial_loss + x - times_lost + final_gain = final_weight) :
  x / initial_loss = 2 := by
  sorry

end weight_ratio_l1027_102733


namespace handmade_ornaments_l1027_102792

noncomputable def handmade_more_than_1_sixth(O : ℕ) (h1 : (1 / 3 : ℚ) * O = 20) (h2 : (1 / 2 : ℚ) * (handmade : ℕ) = 20) : Prop :=
  handmade - (1 / 6 * O) = 20

theorem handmade_ornaments (O handmade : ℕ) (h1 : (1 / 3 : ℚ) * O = 20) (h2 : (1 / 2 : ℚ) * handmade = 20) :
  handmade_more_than_1_sixth O h1 h2 :=
by
  sorry

end handmade_ornaments_l1027_102792


namespace geometric_series_common_ratio_l1027_102799

theorem geometric_series_common_ratio (a r : ℝ) (h : a / (1 - r) = 64 * (a * r^4 / (1 - r))) : r = 1/2 :=
by {
  sorry
}

end geometric_series_common_ratio_l1027_102799


namespace abs_ineq_range_m_l1027_102735

theorem abs_ineq_range_m :
  ∀ m : ℝ, (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) ↔ m ≤ 3 :=
by
  sorry

end abs_ineq_range_m_l1027_102735


namespace even_function_solution_l1027_102730

theorem even_function_solution :
  ∀ (m : ℝ), (∀ x : ℝ, (m+1) * x^2 + (m-2) * x = (m+1) * x^2 - (m-2) * x) → (m = 2 ∧ ∀ x : ℝ, (2+1) * x^2 + (2-2) * x = 3 * x^2) :=
by
  sorry

end even_function_solution_l1027_102730


namespace temperature_max_time_l1027_102783

theorem temperature_max_time (t : ℝ) (h : 0 ≤ t) : 
  (-t^2 + 10 * t + 60 = 85) → t = 15 := 
sorry

end temperature_max_time_l1027_102783


namespace sugar_theft_problem_l1027_102790

-- Define the statements by Gercoginya and the Cook
def gercoginya_statement := "The cook did not steal the sugar"
def cook_statement := "The sugar was stolen by Gercoginya"

-- Define the thief and truth/lie conditions
def thief_lies (x: String) : Prop := x = "The cook stole the sugar"
def other_truth_or_lie (x y: String) : Prop := x = "The sugar was stolen by Gercoginya" ∨ x = "The sugar was not stolen by Gercoginya"

-- The main proof problem to be solved
theorem sugar_theft_problem : 
  ∃ thief : String, 
    (thief = "cook" ∧ thief_lies gercoginya_statement ∧ other_truth_or_lie cook_statement gercoginya_statement) ∨ 
    (thief = "gercoginya" ∧ thief_lies cook_statement ∧ other_truth_or_lie gercoginya_statement cook_statement) :=
sorry

end sugar_theft_problem_l1027_102790


namespace lcm_of_8_9_5_10_l1027_102789

theorem lcm_of_8_9_5_10 : Nat.lcm (Nat.lcm 8 9) (Nat.lcm 5 10) = 360 := by
  sorry

end lcm_of_8_9_5_10_l1027_102789


namespace find_larger_number_l1027_102750

theorem find_larger_number (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : y = 33 :=
by
  sorry

end find_larger_number_l1027_102750


namespace polynomial_simplification_l1027_102752

theorem polynomial_simplification (x : ℝ) :
    (3 * x - 2) * (5 * x^12 - 3 * x^11 + 4 * x^9 - 2 * x^8)
    = 15 * x^13 - 19 * x^12 + 6 * x^11 + 12 * x^10 - 14 * x^9 - 4 * x^8 := by
  sorry

end polynomial_simplification_l1027_102752


namespace wizard_elixir_combinations_l1027_102793

def roots : ℕ := 4
def minerals : ℕ := 5
def incompatible_pairs : ℕ := 3
def total_combinations : ℕ := roots * minerals
def valid_combinations : ℕ := total_combinations - incompatible_pairs

theorem wizard_elixir_combinations : valid_combinations = 17 := by
  sorry

end wizard_elixir_combinations_l1027_102793


namespace grace_earnings_l1027_102749

noncomputable def weekly_charge : ℕ := 300
noncomputable def payment_interval : ℕ := 2
noncomputable def target_weeks : ℕ := 6
noncomputable def target_amount : ℕ := 1800

theorem grace_earnings :
  (target_weeks * weekly_charge = target_amount) → 
  (target_weeks / payment_interval) * (payment_interval * weekly_charge) = target_amount :=
by
  sorry

end grace_earnings_l1027_102749


namespace intersection_point_polar_coords_l1027_102758

open Real

def curve_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

def curve_C2 (t x y : ℝ) : Prop :=
  (x = 2 - t) ∧ (y = t)

theorem intersection_point_polar_coords :
  ∃ (ρ θ : ℝ), (ρ = sqrt 2) ∧ (θ = π / 4) ∧
  ∃ (x y t : ℝ), curve_C2 t x y ∧ curve_C1 x y ∧
  (ρ = sqrt (x^2 + y^2)) ∧ (tan θ = y / x) :=
by
  sorry

end intersection_point_polar_coords_l1027_102758


namespace intersection_area_two_circles_l1027_102748

theorem intersection_area_two_circles :
  let r : ℝ := 3
  let center1 : ℝ × ℝ := (3, 0)
  let center2 : ℝ × ℝ := (0, 3)
  let intersection_area := (9 * Real.pi - 18) / 2
  (∃ x y : ℝ, (x - center1.1)^2 + y^2 = r^2 ∧ x^2 + (y - center2.2)^2 = r^2) →
  (∃ (a : ℝ), a = intersection_area) :=
by
  sorry

end intersection_area_two_circles_l1027_102748


namespace negation_of_proposition_l1027_102797

theorem negation_of_proposition :
  (∀ x : ℝ, 2^x + x^2 > 0) → (∃ x0 : ℝ, 2^x0 + x0^2 ≤ 0) :=
sorry

end negation_of_proposition_l1027_102797


namespace sampling_method_is_stratified_l1027_102717

/-- There are 500 boys and 400 girls in the high school senior year.
The total population consists of 900 students.
A random sample of 25 boys and 20 girls was taken.
Prove that the sampling method used is stratified sampling method. -/
theorem sampling_method_is_stratified :
    let boys := 500
    let girls := 400
    let total_students := 900
    let sample_boys := 25
    let sample_girls := 20
    let sampling_method := "Stratified sampling"
    sample_boys < boys ∧ sample_girls < girls → sampling_method = "Stratified sampling"
:=
sorry

end sampling_method_is_stratified_l1027_102717


namespace currency_conversion_l1027_102757

variable (a : ℚ)

theorem currency_conversion
  (h1 : (0.5 / 100) * a = 75 / 100) -- 0.5% of 'a' = 75 paise
  (rate_usd : ℚ := 0.012)          -- Conversion rate (USD/INR)
  (rate_eur : ℚ := 0.010)          -- Conversion rate (EUR/INR)
  (rate_gbp : ℚ := 0.009)          -- Conversion rate (GBP/INR)
  (paise_to_rupees : ℚ := 1 / 100) -- 1 Rupee = 100 paise
  : (a * paise_to_rupees * rate_usd = 1.8) ∧
    (a * paise_to_rupees * rate_eur = 1.5) ∧
    (a * paise_to_rupees * rate_gbp = 1.35) :=
by
  sorry

end currency_conversion_l1027_102757


namespace bill_due_in_9_months_l1027_102773

-- Define the conditions
def true_discount : ℝ := 240
def face_value : ℝ := 2240
def interest_rate : ℝ := 0.16

-- Define the present value calculated from the true discount and face value
def present_value := face_value - true_discount

-- Define the time in months required to match the conditions
noncomputable def time_in_months : ℝ := 12 * ((face_value / present_value - 1) / interest_rate)

-- State the theorem that the bill is due in 9 months
theorem bill_due_in_9_months : time_in_months = 9 :=
by
  sorry

end bill_due_in_9_months_l1027_102773


namespace sufficient_not_necessary_condition_l1027_102707

noncomputable def setA (x : ℝ) : Prop := 
  (Real.log x / Real.log 2 - 1) * (Real.log x / Real.log 2 - 3) ≤ 0

noncomputable def setB (x : ℝ) (a : ℝ) : Prop := 
  (2 * x - a) / (x + 1) > 1

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, setA x → setB x a) ∧ (¬ ∀ x, setB x a → setA x) ↔ 
  -2 < a ∧ a < 1 := 
  sorry

end sufficient_not_necessary_condition_l1027_102707


namespace range_of_a_l1027_102714

noncomputable def P (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 + a * x + 1 > 0

noncomputable def Q (a : ℝ) : Prop :=
(∃ (x y : ℝ), (x^2 / a + y^2 / (a - 3) = 1)) ∧ ∀ (x y : ℝ), (x^2 / a + y^2 / (a - 3) = 1) → (a * (a - 3) < 0)

theorem range_of_a (a : ℝ) (h1 : P a ∨ Q a) (h2 : ¬ (P a ∧ Q a)) : a = 0 ∨ (3 ≤ a ∧ a < 4) := 
sorry

end range_of_a_l1027_102714


namespace ivar_total_water_needed_l1027_102712

-- Define the initial number of horses
def initial_horses : ℕ := 3

-- Define the added horses
def added_horses : ℕ := 5

-- Define the total number of horses
def total_horses : ℕ := initial_horses + added_horses

-- Define water consumption per horse per day for drinking
def water_consumption_drinking : ℕ := 5

-- Define water consumption per horse per day for bathing
def water_consumption_bathing : ℕ := 2

-- Define total water consumption per horse per day
def total_water_consumption_per_horse_per_day : ℕ := 
    water_consumption_drinking + water_consumption_bathing

-- Define total daily water consumption for all horses
def daily_water_consumption_all_horses : ℕ := 
    total_horses * total_water_consumption_per_horse_per_day

-- Define total water consumption over 28 days
def total_water_consumption_28_days : ℕ := 
    daily_water_consumption_all_horses * 28

-- State the theorem
theorem ivar_total_water_needed : 
    total_water_consumption_28_days = 1568 := 
by
  sorry

end ivar_total_water_needed_l1027_102712


namespace equation1_unique_solutions_equation2_unique_solutions_l1027_102766

noncomputable def solve_equation1 : ℝ → Prop :=
fun x => x ^ 2 - 4 * x + 1 = 0

noncomputable def solve_equation2 : ℝ → Prop :=
fun x => 2 * x ^ 2 - 3 * x + 1 = 0

theorem equation1_unique_solutions :
  ∀ x, solve_equation1 x ↔ (x = 2 + Real.sqrt 3) ∨ (x = 2 - Real.sqrt 3) := by
  sorry

theorem equation2_unique_solutions :
  ∀ x, solve_equation2 x ↔ (x = 1) ∨ (x = 1 / 2) := by
  sorry

end equation1_unique_solutions_equation2_unique_solutions_l1027_102766


namespace unique_zero_of_f_l1027_102710

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2*x + a * (Real.exp (x - 1) + Real.exp (-x + 1))

theorem unique_zero_of_f (a : ℝ) : (∃! x, f x a = 0) ↔ a = 1 / 2 := sorry

end unique_zero_of_f_l1027_102710


namespace p_sufficient_not_necessary_l1027_102746

theorem p_sufficient_not_necessary:
  (∀ a b : ℝ, a > b ∧ b > 0 → (1 / a^2 < 1 / b^2)) ∧ 
  (∃ a b : ℝ, (1 / a^2 < 1 / b^2) ∧ ¬ (a > b ∧ b > 0)) :=
sorry

end p_sufficient_not_necessary_l1027_102746


namespace domain_of_g_eq_l1027_102795

noncomputable def g (x : ℝ) : ℝ := (x + 2) / (Real.sqrt (x^2 - 5 * x + 6))

theorem domain_of_g_eq : 
  {x : ℝ | 0 < x^2 - 5 * x + 6} = {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x} :=
by
  sorry

end domain_of_g_eq_l1027_102795


namespace A_share_of_profit_l1027_102721

theorem A_share_of_profit
  (A_investment : ℤ) (B_investment : ℤ) (C_investment : ℤ)
  (A_profit_share : ℚ) (B_profit_share : ℚ) (C_profit_share : ℚ)
  (total_profit : ℤ) :
  A_investment = 6300 ∧ B_investment = 4200 ∧ C_investment = 10500 ∧
  A_profit_share = 0.45 ∧ B_profit_share = 0.3 ∧ C_profit_share = 0.25 ∧ 
  total_profit = 12200 →
  A_profit_share * total_profit = 5490 :=
by sorry

end A_share_of_profit_l1027_102721


namespace convex_hexagon_possibilities_l1027_102756

noncomputable def hexagon_side_lengths : List ℕ := [1, 2, 3, 4, 5, 6]

theorem convex_hexagon_possibilities : 
  ∃ (hexagons : List (List ℕ)), 
    (∀ h ∈ hexagons, 
      (h.length = 6) ∧ 
      (∀ a ∈ h, a ∈ hexagon_side_lengths)) ∧ 
      (hexagons.length = 3) := 
sorry

end convex_hexagon_possibilities_l1027_102756


namespace reasoning_is_wrong_l1027_102774

-- Definitions of the conditions
def some_rationals_are_proper_fractions := ∃ q : ℚ, ∃ f : ℚ, q = f ∧ f.den ≠ 1
def integers_are_rationals := ∀ z : ℤ, ∃ q : ℚ, q = z

-- Proof that the form of reasoning is wrong given the conditions
theorem reasoning_is_wrong 
  (h₁ : some_rationals_are_proper_fractions) 
  (h₂ : integers_are_rationals) :
  ¬ (∀ z : ℤ, ∃ f : ℚ, z = f ∧ f.den ≠ 1) := 
sorry

end reasoning_is_wrong_l1027_102774


namespace min_le_one_fourth_sum_max_ge_four_ninths_sum_l1027_102716

variable (a b c : ℝ)

theorem min_le_one_fourth_sum
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots : b^2 - 4 * a * c ≥ 0) :
  min a (min b c) ≤ 1 / 4 * (a + b + c) :=
sorry

theorem max_ge_four_ninths_sum
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_roots : b^2 - 4 * a * c ≥ 0) :
  max a (max b c) ≥ 4 / 9 * (a + b + c) :=
sorry

end min_le_one_fourth_sum_max_ge_four_ninths_sum_l1027_102716


namespace union_A_B_intersection_complement_A_B_l1027_102720

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | 4 < x ∧ x < 10}

theorem union_A_B :
  A ∪ B = {x : ℝ | 3 ≤ x ∧ x < 10} :=
sorry

def complement_A := {x : ℝ | x < 3 ∨ x ≥ 7}

theorem intersection_complement_A_B :
  (complement_A ∩ B) = {x : ℝ | 7 ≤ x ∧ x < 10} :=
sorry

end union_A_B_intersection_complement_A_B_l1027_102720


namespace range_of_first_term_l1027_102761

-- Define the arithmetic sequence and its common difference.
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Define the sum of the first n terms of the sequence.
def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a + (n - 1) * d)) / 2

-- Prove the range of the first term a1 given the conditions.
theorem range_of_first_term (a d : ℤ) (S : ℕ → ℤ) (h1 : d = -2)
  (h2 : ∀ n, S n = sum_of_first_n_terms a d n)
  (h3 : S 7 = S 7)
  (h4 : ∀ n, n ≠ 7 → S n < S 7) :
  12 < a ∧ a < 14 :=
by
  sorry

end range_of_first_term_l1027_102761


namespace fraction_d_can_be_zero_l1027_102791

theorem fraction_d_can_be_zero :
  ∃ x : ℝ, (x + 1) / (x - 1) = 0 :=
by {
  sorry
}

end fraction_d_can_be_zero_l1027_102791


namespace inverse_B_squared_l1027_102768

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

def B_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -3, 2],
    ![  1, -1 ]]

theorem inverse_B_squared :
  B⁻¹ = B_inv →
  (B^2)⁻¹ = B_inv * B_inv :=
by sorry

end inverse_B_squared_l1027_102768


namespace days_of_harvest_l1027_102747

-- Conditions
def ripeOrangesPerDay : ℕ := 82
def totalRipeOranges : ℕ := 2050

-- Problem statement: Prove the number of days of harvest
theorem days_of_harvest : (totalRipeOranges / ripeOrangesPerDay) = 25 :=
by
  sorry

end days_of_harvest_l1027_102747


namespace chandra_valid_pairings_l1027_102728

def valid_pairings (num_bowls : ℕ) (num_glasses : ℕ) : ℕ :=
  num_bowls * num_glasses

theorem chandra_valid_pairings : valid_pairings 6 6 = 36 :=
  by sorry

end chandra_valid_pairings_l1027_102728


namespace only_n1_makes_n4_plus4_prime_l1027_102705

theorem only_n1_makes_n4_plus4_prime (n : ℕ) (h : n > 0) : (n = 1) ↔ Prime (n^4 + 4) :=
sorry

end only_n1_makes_n4_plus4_prime_l1027_102705


namespace sum_mod_9_equal_6_l1027_102798

theorem sum_mod_9_equal_6 :
  ((1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888) % 9) = 6 :=
by
  sorry

end sum_mod_9_equal_6_l1027_102798


namespace initial_percentage_proof_l1027_102754

noncomputable def initialPercentageAntifreeze (P : ℝ) : Prop :=
  let initial_fluid : ℝ := 4
  let drained_fluid : ℝ := 2.2857
  let added_antifreeze_fluid : ℝ := 2.2857 * 0.8
  let final_percentage : ℝ := 0.5
  let final_fluid : ℝ := 4
  
  let initial_antifreeze : ℝ := initial_fluid * P
  let drained_antifreeze : ℝ := drained_fluid * P
  let total_antifreeze_after_replacement : ℝ := initial_antifreeze - drained_antifreeze + added_antifreeze_fluid
  
  total_antifreeze_after_replacement = final_fluid * final_percentage

-- Prove that the initial percentage is 0.1
theorem initial_percentage_proof : initialPercentageAntifreeze 0.1 :=
by
  dsimp [initialPercentageAntifreeze]
  simp
  exact sorry

end initial_percentage_proof_l1027_102754


namespace circumscribed_circle_radius_l1027_102719

theorem circumscribed_circle_radius (h8 h15 h17 : ℝ) (h_triangle : h8 = 8 ∧ h15 = 15 ∧ h17 = 17) : 
  ∃ R : ℝ, R = 17 := 
sorry

end circumscribed_circle_radius_l1027_102719


namespace paint_left_after_two_coats_l1027_102732

theorem paint_left_after_two_coats :
  let initial_paint := 3 -- liters
  let first_coat_paint := initial_paint / 2
  let paint_after_first_coat := initial_paint - first_coat_paint
  let second_coat_paint := (2 / 3) * paint_after_first_coat
  let paint_after_second_coat := paint_after_first_coat - second_coat_paint
  (paint_after_second_coat * 1000) = 500 := by
  sorry

end paint_left_after_two_coats_l1027_102732


namespace polynomial_not_factorizable_l1027_102763

theorem polynomial_not_factorizable
  (n m : ℕ)
  (hnm : n > m)
  (hm1 : m > 1)
  (hn_odd : n % 2 = 1)
  (hm_odd : m % 2 = 1) :
  ¬ ∃ (g h : Polynomial ℤ), g.degree > 0 ∧ h.degree > 0 ∧ (x^n + x^m + x + 1 = g * h) :=
by
  sorry

end polynomial_not_factorizable_l1027_102763


namespace solve_triangle_l1027_102769

theorem solve_triangle (a b m₁ m₂ k₃ : ℝ) (h1 : a = m₂ / Real.sin γ) (h2 : b = m₁ / Real.sin γ) : 
  a = m₂ / Real.sin γ ∧ b = m₁ / Real.sin γ := 
  by 
  sorry

end solve_triangle_l1027_102769


namespace sufficient_not_necessary_condition_l1027_102779

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x = -1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = -1) :=
by
  sorry

end sufficient_not_necessary_condition_l1027_102779


namespace dan_destroyed_l1027_102738

def balloons_initial (fred: ℝ) (sam: ℝ) : ℝ := fred + sam

theorem dan_destroyed (fred: ℝ) (sam: ℝ) (final_balloons: ℝ) (destroyed_balloons: ℝ) :
  fred = 10.0 →
  sam = 46.0 →
  final_balloons = 40.0 →
  destroyed_balloons = (balloons_initial fred sam) - final_balloons →
  destroyed_balloons = 16.0 := by
  intros h1 h2 h3 h4
  sorry

end dan_destroyed_l1027_102738


namespace find_ordered_pairs_l1027_102755

theorem find_ordered_pairs (a b : ℕ) (h1 : 2 * a + 1 ∣ 3 * b - 1) (h2 : 2 * b + 1 ∣ 3 * a - 1) : 
  (a = 2 ∧ b = 2) ∨ (a = 12 ∧ b = 17) ∨ (a = 17 ∧ b = 12) :=
by {
  sorry -- proof omitted
}

end find_ordered_pairs_l1027_102755


namespace vojta_correct_sum_l1027_102787

theorem vojta_correct_sum (S A B C : ℕ)
  (h1 : S + (10 * B + C) = 2224)
  (h2 : S + (10 * A + B) = 2198)
  (h3 : S + (10 * A + C) = 2204)
  (A_digit : 0 ≤ A ∧ A < 10)
  (B_digit : 0 ≤ B ∧ B < 10)
  (C_digit : 0 ≤ C ∧ C < 10) :
  S + 100 * A + 10 * B + C = 2324 := 
sorry

end vojta_correct_sum_l1027_102787


namespace cos_neg245_l1027_102741

-- Define the given condition and declare the theorem to prove the required equality
variable (a : ℝ)
def cos_25_eq_a : Prop := (Real.cos 25 * Real.pi / 180 = a)

theorem cos_neg245 :
  cos_25_eq_a a → Real.cos (-245 * Real.pi / 180) = -Real.sqrt (1 - a^2) :=
by
  intro h
  sorry

end cos_neg245_l1027_102741


namespace lines_intersecting_sum_a_b_l1027_102765

theorem lines_intersecting_sum_a_b 
  (a b : ℝ) 
  (hx : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ x = 3 * y + a)
  (hy : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ y = 3 * x + b)
  : a + b = -10 :=
by
  sorry

end lines_intersecting_sum_a_b_l1027_102765


namespace tangent_lines_to_circle_l1027_102713

theorem tangent_lines_to_circle 
  (x y : ℝ) 
  (circle : (x - 2) ^ 2 + (y + 1) ^ 2 = 1) 
  (point : x = 3 ∧ y = 3) : 
  (x = 3 ∨ 15 * x - 8 * y - 21 = 0) :=
sorry

end tangent_lines_to_circle_l1027_102713


namespace symmetric_about_line_5pi12_l1027_102731

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem symmetric_about_line_5pi12 :
  ∀ x : ℝ, f (5 * Real.pi / 12 - x) = f (5 * Real.pi / 12 + x) :=
by
  intros x
  sorry

end symmetric_about_line_5pi12_l1027_102731


namespace percentage_decrease_l1027_102771

theorem percentage_decrease (x y z : ℝ) (h1 : x = 1.30 * y) (h2 : x = 0.65 * z) : 
  ((z - y) / z) * 100 = 50 :=
by
  sorry

end percentage_decrease_l1027_102771


namespace half_angle_in_second_quadrant_l1027_102764

theorem half_angle_in_second_quadrant 
  {θ : ℝ} (k : ℤ)
  (hθ_quadrant4 : 2 * k * Real.pi + (3 / 2) * Real.pi ≤ θ ∧ θ ≤ 2 * k * Real.pi + 2 * Real.pi)
  (hcos : abs (Real.cos (θ / 2)) = - Real.cos (θ / 2)) : 
  ∃ m : ℤ, (m * Real.pi + (Real.pi / 2) ≤ θ / 2 ∧ θ / 2 ≤ m * Real.pi + Real.pi) :=
sorry

end half_angle_in_second_quadrant_l1027_102764
