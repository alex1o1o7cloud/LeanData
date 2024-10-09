import Mathlib

namespace sum_geometric_sequence_first_10_terms_l2243_224325

theorem sum_geometric_sequence_first_10_terms :
  let a₁ : ℚ := 12
  let r : ℚ := 1 / 3
  let S₁₀ : ℚ := 12 * (1 - (1 / 3)^10) / (1 - 1 / 3)
  S₁₀ = 1062864 / 59049 := by
  sorry

end sum_geometric_sequence_first_10_terms_l2243_224325


namespace monotonic_intervals_range_of_c_l2243_224345

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := c * Real.log x + (1 / 2) * x ^ 2 + b * x

lemma extreme_point_condition {b c : ℝ} (h1 : c ≠ 0) (h2 : f 1 b c = 0) : b + c + 1 = 0 :=
sorry

theorem monotonic_intervals (b c : ℝ) (h1 : c ≠ 0) (h2 : f 1 b c = 0) (h3 : c > 1) :
  (∀ x, 0 < x ∧ x < 1 → f 1 b c < f x b c) ∧ 
  (∀ x, 1 < x ∧ x < c → f 1 b c > f x b c) ∧ 
  (∀ x, x > c → f 1 b c < f x b c) :=
sorry

theorem range_of_c (b c : ℝ) (h1 : c ≠ 0) (h2 : f 1 b c = 0) (h3 : (f 1 b c < 0)) :
  -1 / 2 < c ∧ c < 0 :=
sorry

end monotonic_intervals_range_of_c_l2243_224345


namespace exists_n_geq_k_l2243_224312

theorem exists_n_geq_k (a : ℕ → ℕ) (h_distinct : ∀ i j : ℕ, i ≠ j → a i ≠ a j) 
    (h_positive : ∀ i : ℕ, a i > 0) :
    ∀ k : ℕ, ∃ n : ℕ, n > k ∧ a n ≥ n :=
by
  intros k
  sorry

end exists_n_geq_k_l2243_224312


namespace smallest_number_of_students_l2243_224339

-- Define the conditions as given in the problem
def eight_to_six_ratio : ℕ × ℕ := (5, 3) -- ratio of 8th-graders to 6th-graders
def eight_to_nine_ratio : ℕ × ℕ := (7, 4) -- ratio of 8th-graders to 9th-graders

theorem smallest_number_of_students (a b c : ℕ)
  (h1 : a = 5 * b) (h2 : b = 3 * c) (h3 : a = 7 * c) : a + b + c = 76 := 
sorry

end smallest_number_of_students_l2243_224339


namespace inequality_and_equality_condition_l2243_224394

theorem inequality_and_equality_condition (a b : ℝ) (h : a < b) :
  a^3 - 3 * a ≤ b^3 - 3 * b + 4 ∧ (a = -1 ∧ b = 1 → a^3 - 3 * a = b^3 - 3 * b + 4) :=
sorry

end inequality_and_equality_condition_l2243_224394


namespace eggs_in_box_l2243_224306

theorem eggs_in_box (initial_count : ℝ) (added_count : ℝ) (total_count : ℝ) 
  (h_initial : initial_count = 47.0) 
  (h_added : added_count = 5.0) : total_count = 52.0 :=
by 
  sorry

end eggs_in_box_l2243_224306


namespace balloon_highest_elevation_l2243_224348

theorem balloon_highest_elevation 
  (lift_rate : ℕ)
  (descend_rate : ℕ)
  (pull_time1 : ℕ)
  (release_time : ℕ)
  (pull_time2 : ℕ) :
  lift_rate = 50 →
  descend_rate = 10 →
  pull_time1 = 15 →
  release_time = 10 →
  pull_time2 = 15 →
  (lift_rate * pull_time1 - descend_rate * release_time + lift_rate * pull_time2) = 1400 :=
by
  sorry

end balloon_highest_elevation_l2243_224348


namespace parallel_lines_value_of_a_l2243_224341

theorem parallel_lines_value_of_a (a : ℝ) : 
  (∀ x y : ℝ, ax + (a+2)*y + 2 = 0 → x + a*y + 1 = 0 → ∀ m n : ℝ, ax + (a + 2)*n + 2 = 0 → x + a*n + 1 = 0) →
  a = -1 := 
sorry

end parallel_lines_value_of_a_l2243_224341


namespace parents_without_fulltime_jobs_l2243_224365

theorem parents_without_fulltime_jobs (total : ℕ) (mothers fathers full_time_mothers full_time_fathers : ℕ) 
(h1 : mothers = 2 * fathers / 3)
(h2 : full_time_mothers = 9 * mothers / 10)
(h3 : full_time_fathers = 3 * fathers / 4)
(h4 : mothers + fathers = total) :
(100 * (total - (full_time_mothers + full_time_fathers))) / total = 19 :=
by
  sorry

end parents_without_fulltime_jobs_l2243_224365


namespace func_inequality_l2243_224303

noncomputable def f (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Given function properties
variables {a b c : ℝ} (h_a : a > 0) (symmetry : ∀ x : ℝ, f a b c (2 + x) = f a b c (2 - x))

theorem func_inequality : f a b c 2 < f a b c 1 ∧ f a b c 1 < f a b c 4 :=
by
  sorry

end func_inequality_l2243_224303


namespace integer_sided_triangle_with_60_degree_angle_exists_l2243_224311

theorem integer_sided_triangle_with_60_degree_angle_exists 
  (m n t : ℤ) : 
  ∃ (x y z : ℤ), (x = (m^2 - n^2) * t) ∧ 
                  (y = m * (m - 2 * n) * t) ∧ 
                  (z = (m^2 - m * n + n^2) * t) := by
  sorry

end integer_sided_triangle_with_60_degree_angle_exists_l2243_224311


namespace complement_U_M_l2243_224327

theorem complement_U_M :
  let U := {x : ℤ | ∃ k : ℤ, x = 2 * k}
  let M := {x : ℤ | ∃ k : ℤ, x = 4 * k}
  {x | x ∈ U ∧ x ∉ M} = {x : ℤ | ∃ k : ℤ, x = 4 * k - 2} :=
by
  sorry

end complement_U_M_l2243_224327


namespace smallest_perimeter_even_integer_triangl_l2243_224398

theorem smallest_perimeter_even_integer_triangl (n : ℕ) (h : n > 2) :
  let a := 2 * n - 2
  let b := 2 * n
  let c := 2 * n + 2
  2 * n - 2 + 2 * n > 2 * n + 2 ∧
  2 * n - 2 + 2 * n + 2 > 2 * n ∧
  2 * n + 2 * n + 2 > 2 * n - 2 ∧ 
  2 * 3 - 2 + 2 * 3 + 2 * 3 + 2 = 18 :=
by
  { sorry }

end smallest_perimeter_even_integer_triangl_l2243_224398


namespace exponent_product_to_sixth_power_l2243_224347

theorem exponent_product_to_sixth_power :
  ∃ n : ℤ, 3^(12) * 3^(18) = n^6 ∧ n = 243 :=
by
  use 243
  sorry

end exponent_product_to_sixth_power_l2243_224347


namespace correct_option_B_l2243_224371

theorem correct_option_B (a b : ℝ) : (-a^2 * b^3)^2 = a^4 * b^6 := 
  sorry

end correct_option_B_l2243_224371


namespace remaining_work_hours_l2243_224333

theorem remaining_work_hours (initial_hours_per_week initial_weeks total_earnings first_weeks first_week_hours : ℝ) 
  (hourly_wage remaining_weeks remaining_earnings total_hours_required : ℝ) : 
  15 = initial_hours_per_week →
  15 = initial_weeks →
  4500 = total_earnings →
  3 = first_weeks →
  5 = first_week_hours →
  hourly_wage = total_earnings / (initial_hours_per_week * initial_weeks) →
  remaining_earnings = total_earnings - (first_week_hours * hourly_wage * first_weeks) →
  remaining_weeks = initial_weeks - first_weeks →
  total_hours_required = remaining_earnings / (hourly_wage * remaining_weeks) →
  total_hours_required = 17.5 :=
by
  intros
  sorry

end remaining_work_hours_l2243_224333


namespace find_quadratic_expression_l2243_224381

-- Define the quadratic function
def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

-- Define conditions
def intersects_x_axis_at_A (a b c : ℝ) : Prop :=
  quadratic a b c (-2) = 0

def intersects_x_axis_at_B (a b c : ℝ) : Prop :=
  quadratic a b c (1) = 0

def has_maximum_value (a : ℝ) : Prop :=
  a < 0

-- Define the target function
def f_expr (x : ℝ) : ℝ := -x^2 - x + 2

-- The theorem to be proved
theorem find_quadratic_expression :
  ∃ a b c, 
    intersects_x_axis_at_A a b c ∧
    intersects_x_axis_at_B a b c ∧
    has_maximum_value a ∧
    ∀ x, quadratic a b c x = f_expr x :=
sorry

end find_quadratic_expression_l2243_224381


namespace find_pairs_l2243_224383

theorem find_pairs (a b : ℕ) :
  (1111 * a) % (11 * b) = 11 * (a - b) →
  140 ≤ (1111 * a) / (11 * b) ∧ (1111 * a) / (11 * b) ≤ 160 →
  (a, b) = (3, 2) ∨ (a, b) = (6, 4) ∨ (a, b) = (7, 5) ∨ (a, b) = (9, 6) :=
by
  sorry

end find_pairs_l2243_224383


namespace precision_tens_place_l2243_224374

-- Given
def given_number : ℝ := 4.028 * (10 ^ 5)

-- Prove that the precision of the given_number is to the tens place.
theorem precision_tens_place : true := by
  -- Proof goes here
  sorry

end precision_tens_place_l2243_224374


namespace greatest_drop_in_june_l2243_224321

def monthly_changes := [("January", 1.50), ("February", -2.25), ("March", 0.75), ("April", -3.00), ("May", 1.00), ("June", -4.00)]

theorem greatest_drop_in_june : ∀ months : List (String × Float), (months = monthly_changes) → 
  (∃ month : String, 
    month = "June" ∧ 
    ∀ m p, m ≠ "June" → (m, p) ∈ months → p ≥ -4.00) :=
by
  sorry

end greatest_drop_in_june_l2243_224321


namespace number_of_pieces_l2243_224396

def length_piece : ℝ := 0.40
def total_length : ℝ := 47.5

theorem number_of_pieces : ⌊total_length / length_piece⌋ = 118 := by
  sorry

end number_of_pieces_l2243_224396


namespace mark_candy_bars_consumption_l2243_224338

theorem mark_candy_bars_consumption 
  (recommended_intake : ℕ := 150)
  (soft_drink_calories : ℕ := 2500)
  (soft_drink_added_sugar_percent : ℕ := 5)
  (candy_bar_added_sugar_calories : ℕ := 25)
  (exceeded_percentage : ℕ := 100)
  (actual_intake := recommended_intake + (recommended_intake * exceeded_percentage / 100))
  (soft_drink_added_sugar := soft_drink_calories * soft_drink_added_sugar_percent / 100)
  (candy_bars_added_sugar := actual_intake - soft_drink_added_sugar)
  (number_of_bars := candy_bars_added_sugar / candy_bar_added_sugar_calories) : 
  number_of_bars = 7 := 
by
  sorry

end mark_candy_bars_consumption_l2243_224338


namespace Timmy_needs_to_go_faster_l2243_224316

-- Define the trial speeds and the required speed
def s1 : ℕ := 36
def s2 : ℕ := 34
def s3 : ℕ := 38
def s_req : ℕ := 40

-- Statement of the theorem
theorem Timmy_needs_to_go_faster :
  s_req - (s1 + s2 + s3) / 3 = 4 :=
by
  sorry

end Timmy_needs_to_go_faster_l2243_224316


namespace Jonas_needs_to_buy_35_pairs_of_socks_l2243_224324

theorem Jonas_needs_to_buy_35_pairs_of_socks
  (socks : ℕ)
  (shoes : ℕ)
  (pants : ℕ)
  (tshirts : ℕ)
  (double_items : ℕ)
  (needed_items : ℕ)
  (pairs_of_socks_needed : ℕ) :
  socks = 20 →
  shoes = 5 →
  pants = 10 →
  tshirts = 10 →
  double_items = 2 * (2 * socks + 2 * shoes + pants + tshirts) →
  needed_items = double_items - (2 * socks + 2 * shoes + pants + tshirts) →
  pairs_of_socks_needed = needed_items / 2 →
  pairs_of_socks_needed = 35 :=
by sorry

end Jonas_needs_to_buy_35_pairs_of_socks_l2243_224324


namespace are_naptime_l2243_224366

def flight_duration := 11 * 60 + 20  -- in minutes

def time_spent_reading := 2 * 60      -- in minutes
def time_spent_watching_movies := 4 * 60  -- in minutes
def time_spent_eating_dinner := 30    -- in minutes
def time_spent_listening_to_radio := 40   -- in minutes
def time_spent_playing_games := 1 * 60 + 10   -- in minutes

def total_time_spent_on_activities := 
  time_spent_reading + 
  time_spent_watching_movies + 
  time_spent_eating_dinner + 
  time_spent_listening_to_radio + 
  time_spent_playing_games

def remaining_time := (flight_duration - total_time_spent_on_activities) / 60  -- in hours

theorem are_naptime : remaining_time = 3 := by
  sorry

end are_naptime_l2243_224366


namespace reading_hours_l2243_224388

theorem reading_hours (h : ℕ) (lizaRate suzieRate : ℕ) (lizaPages suziePages : ℕ) 
  (hliza : lizaRate = 20) (hsuzie : suzieRate = 15) 
  (hlizaPages : lizaPages = lizaRate * h) (hsuziePages : suziePages = suzieRate * h) 
  (h_diff : lizaPages = suziePages + 15) : h = 3 :=
by {
  sorry
}

end reading_hours_l2243_224388


namespace days_before_reinforcement_l2243_224317

theorem days_before_reinforcement
    (garrison_1 : ℕ)
    (initial_days : ℕ)
    (reinforcement : ℕ)
    (additional_days : ℕ)
    (total_men_after_reinforcement : ℕ)
    (man_days_initial : ℕ)
    (man_days_after : ℕ)
    (x : ℕ) :
    garrison_1 * (initial_days - x) = total_men_after_reinforcement * additional_days →
    garrison_1 = 2000 →
    initial_days = 54 →
    reinforcement = 1600 →
    additional_days = 20 →
    total_men_after_reinforcement = garrison_1 + reinforcement →
    man_days_initial = garrison_1 * initial_days →
    man_days_after = total_men_after_reinforcement * additional_days →
    x = 18 :=
by
  intros h_eq g_1 i_days r_f a_days total_men m_days_i m_days_a
  sorry

end days_before_reinforcement_l2243_224317


namespace true_if_a_gt_1_and_b_gt_1_then_ab_gt_1_l2243_224322

theorem true_if_a_gt_1_and_b_gt_1_then_ab_gt_1 (a b : ℝ) (ha : a > 1) (hb : b > 1) : ab > 1 :=
sorry

end true_if_a_gt_1_and_b_gt_1_then_ab_gt_1_l2243_224322


namespace vector_perpendicular_iff_l2243_224378

theorem vector_perpendicular_iff (k : ℝ) :
  let a := (Real.sqrt 3, 1)
  let b := (0, 1)
  let c := (k, Real.sqrt 3)
  let ab := (Real.sqrt 3, 3)  -- a + 2b
  a.1 * c.1 + ab.2 * c.2 = 0 → k = -3 :=
by
  let a := (Real.sqrt 3, 1)
  let b := (0, 1)
  let c := (k, Real.sqrt 3)
  let ab := (Real.sqrt 3, 3)  -- a + 2b
  intro h
  sorry

end vector_perpendicular_iff_l2243_224378


namespace arithmetic_geometric_seq_proof_l2243_224367

theorem arithmetic_geometric_seq_proof
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : a1 - a2 = -1)
  (h2 : 1 * (b2 * b2) = 4)
  (h3 : b2 > 0) :
  (a1 - a2) / b2 = -1 / 2 :=
by
  sorry

end arithmetic_geometric_seq_proof_l2243_224367


namespace transmission_time_is_128_l2243_224384

def total_time (blocks chunks_per_block rate : ℕ) : ℕ :=
  (blocks * chunks_per_block) / rate

theorem transmission_time_is_128 :
  total_time 80 256 160 = 128 :=
  by
  sorry

end transmission_time_is_128_l2243_224384


namespace evaluate_expression_at_zero_l2243_224356

theorem evaluate_expression_at_zero :
  ∀ x : ℝ, (x ≠ -1) ∧ (x ≠ 3) →
  ( (3 * x^2 - 2 * x + 1) / ((x + 1) * (x - 3)) - (5 + 2 * x) / ((x + 1) * (x - 3)) ) = 2 :=
by
  sorry

end evaluate_expression_at_zero_l2243_224356


namespace hockey_players_count_l2243_224353

theorem hockey_players_count (cricket_players : ℕ) (football_players : ℕ) (softball_players : ℕ) (total_players : ℕ) 
(h_cricket : cricket_players = 16) 
(h_football : football_players = 18) 
(h_softball : softball_players = 13) 
(h_total : total_players = 59) : 
  total_players - (cricket_players + football_players + softball_players) = 12 := 
by sorry

end hockey_players_count_l2243_224353


namespace tan_neg_five_pi_div_four_l2243_224389

theorem tan_neg_five_pi_div_four : Real.tan (- (5 * Real.pi / 4)) = -1 := 
sorry

end tan_neg_five_pi_div_four_l2243_224389


namespace center_of_circle_l2243_224360

theorem center_of_circle (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (3, 8)) (h2 : (x2, y2) = (11, -4)) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (7, 2) := by
  sorry

end center_of_circle_l2243_224360


namespace horizontal_asymptote_of_rational_function_l2243_224376

theorem horizontal_asymptote_of_rational_function :
  (∃ y, y = (10 * x ^ 4 + 3 * x ^ 3 + 7 * x ^ 2 + 6 * x + 4) / (2 * x ^ 4 + 5 * x ^ 3 + 4 * x ^ 2 + 2 * x + 1) → y = 5) := sorry

end horizontal_asymptote_of_rational_function_l2243_224376


namespace houses_with_dogs_l2243_224368

theorem houses_with_dogs (C B Total : ℕ) (hC : C = 30) (hB : B = 10) (hTotal : Total = 60) :
  ∃ D, D = 40 :=
by
  -- The overall proof would go here
  sorry

end houses_with_dogs_l2243_224368


namespace math_problem_l2243_224343

theorem math_problem :
  ( (1 / 3 * 9) ^ 2 * (1 / 27 * 81) ^ 2 * (1 / 243 * 729) ^ 2) = 729 := by
  sorry

end math_problem_l2243_224343


namespace prove_a_range_l2243_224308

-- Defining the propositions p and q
def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1 : ℝ) 1, a^2 * x^2 + a * x - 2 = 0
def q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- The proposition to prove
theorem prove_a_range (a : ℝ) (hpq : ¬(p a ∨ q a)) : a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 :=
by
  sorry

end prove_a_range_l2243_224308


namespace product_base9_l2243_224304

open Nat

noncomputable def base9_product (a b : ℕ) : ℕ := 
  let a_base10 := 3*9^2 + 6*9^1 + 2*9^0
  let b_base10 := 7
  let product_base10 := a_base10 * b_base10
  -- converting product_base10 from base 10 to base 9
  2 * 9^3 + 8 * 9^2 + 7 * 9^1 + 5 * 9^0 -- which simplifies to 2875 in base 9

theorem product_base9: base9_product 362 7 = 2875 :=
by
  -- Here should be the proof or a computational check
  sorry

end product_base9_l2243_224304


namespace point_in_fourth_quadrant_l2243_224330

variable (a : ℝ)

theorem point_in_fourth_quadrant (h : a < -1) : 
    let x := a^2 - 2*a - 1
    let y := (a + 1) / abs (a + 1)
    (x > 0) ∧ (y < 0) := 
by
  let x := a^2 - 2*a - 1
  let y := (a + 1) / abs (a + 1)
  sorry

end point_in_fourth_quadrant_l2243_224330


namespace aiden_nap_is_15_minutes_l2243_224392

def aiden_nap_duration_in_minutes (nap_in_hours : ℚ) (minutes_per_hour : ℕ) : ℚ :=
  nap_in_hours * minutes_per_hour

theorem aiden_nap_is_15_minutes :
  aiden_nap_duration_in_minutes (1/4) 60 = 15 := by
  sorry

end aiden_nap_is_15_minutes_l2243_224392


namespace find_square_sum_of_xy_l2243_224350

theorem find_square_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h1 : x * y + x + y = 83) (h2 : x^2 * y + x * y^2 = 1056) : x^2 + y^2 = 458 :=
sorry

end find_square_sum_of_xy_l2243_224350


namespace Sandy_phone_bill_expense_l2243_224358
noncomputable def Sandy_age_now : ℕ := 34
noncomputable def Kim_age_now : ℕ := 10
noncomputable def Sandy_phone_bill : ℕ := 10 * Sandy_age_now

theorem Sandy_phone_bill_expense :
  (Sandy_age_now - 2 = 36 - 2) ∧ (Kim_age_now + 2 = 12) ∧ (36 = 3 * 12) ∧ (Sandy_phone_bill = 340) := by
sorry

end Sandy_phone_bill_expense_l2243_224358


namespace cubic_common_roots_l2243_224335

theorem cubic_common_roots:
  ∃ (c d : ℝ), 
  (∀ r s : ℝ,
    r ≠ s ∧ 
    (r ∈ {x : ℝ | x^3 + c * x^2 + 16 * x + 9 = 0}) ∧
    (s ∈ {x : ℝ | x^3 + c * x^2 + 16 * x + 9 = 0}) ∧ 
    (r ∈ {x : ℝ | x^3 + d * x^2 + 20 * x + 12 = 0}) ∧
    (s ∈ {x : ℝ | x^3 + d * x^2 + 20 * x + 12 = 0})) → 
  c = 8 ∧ d = 9 := 
by
  sorry

end cubic_common_roots_l2243_224335


namespace find_numbers_l2243_224362

theorem find_numbers (a b c : ℕ) (h₁ : 10 ≤ b ∧ b < 100) (h₂ : 10 ≤ c ∧ c < 100)
    (h₃ : 10^4 * a + 100 * b + c = (a + b + c)^3) : (a = 9 ∧ b = 11 ∧ c = 25) :=
by
  sorry

end find_numbers_l2243_224362


namespace div_eq_frac_l2243_224336

theorem div_eq_frac : 250 / (5 + 12 * 3^2) = 250 / 113 :=
by
  sorry

end div_eq_frac_l2243_224336


namespace combined_tennis_percentage_l2243_224331

variable (totalStudentsNorth totalStudentsSouth : ℕ)
variable (percentTennisNorth percentTennisSouth : ℕ)

def studentsPreferringTennisNorth : ℕ := totalStudentsNorth * percentTennisNorth / 100
def studentsPreferringTennisSouth : ℕ := totalStudentsSouth * percentTennisSouth / 100

def totalStudentsBothSchools : ℕ := totalStudentsNorth + totalStudentsSouth
def studentsPreferringTennisBothSchools : ℕ := studentsPreferringTennisNorth totalStudentsNorth percentTennisNorth
                                            + studentsPreferringTennisSouth totalStudentsSouth percentTennisSouth

def combinedPercentTennis : ℕ := studentsPreferringTennisBothSchools totalStudentsNorth totalStudentsSouth percentTennisNorth percentTennisSouth
                                 * 100 / totalStudentsBothSchools totalStudentsNorth totalStudentsSouth

theorem combined_tennis_percentage :
  (totalStudentsNorth = 1800) →
  (totalStudentsSouth = 2700) →
  (percentTennisNorth = 25) →
  (percentTennisSouth = 35) →
  combinedPercentTennis totalStudentsNorth totalStudentsSouth percentTennisNorth percentTennisSouth = 31 :=
by
  intros
  sorry

end combined_tennis_percentage_l2243_224331


namespace find_A_l2243_224385

theorem find_A :
  ∃ A B : ℕ, A < 10 ∧ B < 10 ∧ 5 * 100 + A * 10 + 8 - (B * 100 + 1 * 10 + 4) = 364 ∧ A = 7 :=
by
  sorry

end find_A_l2243_224385


namespace max_single_player_salary_l2243_224354

theorem max_single_player_salary (n : ℕ) (m : ℕ) (T : ℕ) (n_pos : n = 18) (m_pos : m = 20000) (T_pos : T = 800000) :
  ∃ x : ℕ, (∀ y : ℕ, y ≤ x → y ≤ 460000) ∧ (17 * m + x ≤ T) :=
by
  sorry

end max_single_player_salary_l2243_224354


namespace no_real_m_perpendicular_l2243_224302

theorem no_real_m_perpendicular (m : ℝ) : 
  ¬ ∃ m, ((m - 2) * m = -3) := 
sorry

end no_real_m_perpendicular_l2243_224302


namespace greatest_prime_factor_of_15_l2243_224382

-- Definitions based on conditions
def factorial (n : ℕ) : ℕ := Nat.factorial n

def expr : ℕ := factorial 15 + factorial 17

-- Statement to prove
theorem greatest_prime_factor_of_15!_plus_17! :
  ∃ p : ℕ, Nat.Prime p ∧ (∀ q: ℕ, Nat.Prime q ∧ q ∣ expr → q ≤ 13) ∧ p = 13 :=
sorry

end greatest_prime_factor_of_15_l2243_224382


namespace wholesale_price_of_milk_l2243_224364

theorem wholesale_price_of_milk (W : ℝ) 
  (h1 : ∀ p : ℝ, p = 1.25 * W) 
  (h2 : ∀ q : ℝ, q = 0.95 * (1.25 * W)) 
  (h3 : q = 4.75) :
  W = 4 :=
by
  sorry

end wholesale_price_of_milk_l2243_224364


namespace ratio_of_speeds_l2243_224357

theorem ratio_of_speeds (L V : ℝ) (R : ℝ) (h1 : L > 0) (h2 : V > 0) (h3 : R ≠ 0)
  (h4 : (1.48 * L) / (R * V) = (1.40 * L) / V) : R = 37 / 35 :=
by
  -- Proof would be inserted here
  sorry

end ratio_of_speeds_l2243_224357


namespace isosceles_trapezoid_height_l2243_224397

/-- Given an isosceles trapezoid with area 100 and diagonals that are mutually perpendicular,
    we want to prove that the height of the trapezoid is 10. -/
theorem isosceles_trapezoid_height (BC AD h : ℝ) 
    (area_eq_100 : 100 = (1 / 2) * (BC + AD) * h)
    (height_eq_half_sum : h = (1 / 2) * (BC + AD)) :
    h = 10 :=
by
  sorry

end isosceles_trapezoid_height_l2243_224397


namespace license_plate_count_correct_l2243_224344

-- Define the number of choices for digits and letters
def num_digit_choices : ℕ := 10^3
def num_letter_block_choices : ℕ := 26^3
def num_position_choices : ℕ := 4

-- Compute the total number of distinct license plates
def total_license_plates : ℕ := num_position_choices * num_digit_choices * num_letter_block_choices

-- The proof statement
theorem license_plate_count_correct : total_license_plates = 70304000 := by
  -- This proof is left as an exercise
  sorry

end license_plate_count_correct_l2243_224344


namespace only_zero_and_one_square_equal_themselves_l2243_224323

theorem only_zero_and_one_square_equal_themselves (x: ℝ) : (x^2 = x) ↔ (x = 0 ∨ x = 1) :=
by sorry

end only_zero_and_one_square_equal_themselves_l2243_224323


namespace vectors_orthogonal_x_value_l2243_224377

theorem vectors_orthogonal_x_value :
  (∀ x : ℝ, (3 * x + 4 * (-7) = 0) → (x = 28 / 3)) := 
by 
  sorry

end vectors_orthogonal_x_value_l2243_224377


namespace trivia_team_members_l2243_224373

theorem trivia_team_members (n p s x y : ℕ) (h1 : n = 12) (h2 : p = 64) (h3 : s = 8) (h4 : x = p / s) (h5 : y = n - x) : y = 4 :=
by
  sorry

end trivia_team_members_l2243_224373


namespace average_number_of_visitors_is_25_l2243_224313

-- Define the sequence parameters
def a : ℕ := 10  -- First term
def d : ℕ := 5   -- Common difference
def n : ℕ := 7   -- Number of days

-- Define the sequence for the number of visitors on each day
def visitors (i : ℕ) : ℕ := a + (i - 1) * d

-- Define the average number of visitors
def avg_visitors : ℕ := (List.sum (List.map visitors [1, 2, 3, 4, 5, 6, 7])) / n

-- Prove the average
theorem average_number_of_visitors_is_25 : avg_visitors = 25 :=
by
  -- Placeholder for the actual proof
  sorry

end average_number_of_visitors_is_25_l2243_224313


namespace reach_any_position_l2243_224379

/-- We define a configuration of marbles in terms of a finite list of natural numbers, which corresponds to the number of marbles in each hole. A configuration transitions to another by moving marbles from one hole to subsequent holes in a circular manner. -/
def configuration (n : ℕ) := List ℕ 

/-- Define the operation of distributing marbles from one hole to subsequent holes. -/
def redistribute (l : configuration n) (i : ℕ) : configuration n :=
  sorry -- The exact redistribution function would need to be implemented based on the conditions.

theorem reach_any_position (n : ℕ) (m : ℕ) (init_config final_config : configuration n)
  (h_num_marbles : init_config.sum = m)
  (h_final_marbles : final_config.sum = m) :
  ∃ steps, final_config = (steps : List ℕ).foldl redistribute init_config :=
sorry

end reach_any_position_l2243_224379


namespace triangle_smallest_angle_l2243_224395

theorem triangle_smallest_angle (a b c : ℝ) (h1 : a + b + c = 180) (h2 : a = 5 * c) (h3 : b = 3 * c) : c = 20 :=
by
  sorry

end triangle_smallest_angle_l2243_224395


namespace probability_second_third_different_colors_l2243_224386

def probability_different_colors (blue_chips : ℕ) (red_chips : ℕ) (yellow_chips : ℕ) : ℚ :=
  let total_chips := blue_chips + red_chips + yellow_chips
  let prob_diff :=
    ((blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips)) +
    ((red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips)) +
    ((yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips))
  prob_diff

theorem probability_second_third_different_colors :
  probability_different_colors 7 6 5 = 107 / 162 :=
by
  sorry

end probability_second_third_different_colors_l2243_224386


namespace probability_of_sum_being_6_l2243_224351

noncomputable def prob_sum_6 : ℚ :=
  let total_outcomes := 6 * 6
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes

theorem probability_of_sum_being_6 :
  prob_sum_6 = 5 / 36 :=
by
  sorry

end probability_of_sum_being_6_l2243_224351


namespace exists_two_natural_pairs_satisfying_equation_l2243_224309

theorem exists_two_natural_pairs_satisfying_equation :
  ∃ (x1 y1 x2 y2 : ℕ), (2 * x1^3 = y1^4) ∧ (2 * x2^3 = y2^4) ∧ ¬(x1 = x2 ∧ y1 = y2) :=
sorry

end exists_two_natural_pairs_satisfying_equation_l2243_224309


namespace intersection_nonempty_implies_range_l2243_224359

namespace ProofProblem

def M (x y : ℝ) : Prop := x + y + 1 ≥ Real.sqrt (2 * (x^2 + y^2))
def N (a x y : ℝ) : Prop := |x - a| + |y - 1| ≤ 1

theorem intersection_nonempty_implies_range (a : ℝ) :
  (∃ x y : ℝ, M x y ∧ N a x y) → (1 - Real.sqrt 6 ≤ a ∧ a ≤ 3 + Real.sqrt 10) :=
by
  sorry

end ProofProblem

end intersection_nonempty_implies_range_l2243_224359


namespace puppy_weight_l2243_224387

variable (a b c : ℝ)

theorem puppy_weight :
  (a + b + c = 30) →
  (a + c = 3 * b) →
  (a + b = c) →
  a = 7.5 := by
  intros h1 h2 h3
  sorry

end puppy_weight_l2243_224387


namespace arithmetic_sequence_problem_l2243_224352

variable {a₁ d : ℝ} (S : ℕ → ℝ)

axiom Sum_of_terms (n : ℕ) : S n = n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_problem
  (h : S 10 = 4 * S 5) :
  (a₁ / d) = 1 / 2 :=
by
  -- definitional expansion and algebraic simplification would proceed here
  sorry

end arithmetic_sequence_problem_l2243_224352


namespace youngest_person_age_l2243_224370

theorem youngest_person_age (total_age_now : ℕ) (total_age_when_born : ℕ) (Y : ℕ) (h1 : total_age_now = 210) (h2 : total_age_when_born = 162) : Y = 48 :=
by
  sorry

end youngest_person_age_l2243_224370


namespace remaining_digits_product_l2243_224346

theorem remaining_digits_product (a b c : ℕ)
  (h1 : (a + b) % 10 = c % 10)
  (h2 : (b + c) % 10 = a % 10)
  (h3 : (c + a) % 10 = b % 10) :
  ((a * b * c) % 1000 = 0 ∨
   (a * b * c) % 1000 = 250 ∨
   (a * b * c) % 1000 = 500 ∨
   (a * b * c) % 1000 = 750) :=
sorry

end remaining_digits_product_l2243_224346


namespace ratio_problem_l2243_224318

theorem ratio_problem
  (w x y z : ℝ)
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 2 / 3)
  (h3 : w / z = 3 / 5) :
  (x + y) / z = 27 / 10 :=
by
  sorry

end ratio_problem_l2243_224318


namespace not_always_true_inequality_l2243_224375

variable {x y z : ℝ} {k : ℤ}

theorem not_always_true_inequality :
  x > 0 → y > 0 → x > y → z ≠ 0 → k ≠ 0 → ¬ ( ∀ z, (x / (z^k) > y / (z^k)) ) :=
by
  intro hx hy hxy hz hk
  sorry

end not_always_true_inequality_l2243_224375


namespace geometric_to_arithmetic_sequence_l2243_224320

theorem geometric_to_arithmetic_sequence {a : ℕ → ℝ} (q : ℝ) 
    (h_gt0 : 0 < q) (h_pos : ∀ n, 0 < a n)
    (h_geom_seq : ∀ n, a (n + 1) = a n * q)
    (h_arith_seq : 2 * (1 / 2 * a 3) = a 1 + 2 * a 2) :
    a 10 / a 8 = 3 + 2 * Real.sqrt 2 := 
by
  sorry

end geometric_to_arithmetic_sequence_l2243_224320


namespace a_2008_lt_5_l2243_224334

theorem a_2008_lt_5 :
  ∃ a b : ℕ → ℝ, 
    a 1 = 1 ∧ 
    b 1 = 2 ∧ 
    (∀ n, a (n + 1) = (1 + a n + a n * b n) / (b n)) ∧ 
    (∀ n, b (n + 1) = (1 + b n + a n * b n) / (a n)) ∧ 
    a 2008 < 5 := 
sorry

end a_2008_lt_5_l2243_224334


namespace teal_more_blue_l2243_224369

theorem teal_more_blue (total : ℕ) (green : ℕ) (both_green_blue : ℕ) (neither_green_blue : ℕ)
  (h1 : total = 150) (h2 : green = 90) (h3 : both_green_blue = 40) (h4 : neither_green_blue = 25) :
  ∃ (blue : ℕ), blue = 75 :=
by
  sorry

end teal_more_blue_l2243_224369


namespace number_of_valid_pairs_l2243_224337

theorem number_of_valid_pairs (m n : ℕ) (h1 : n > m) (h2 : 3 * (m - 4) * (n - 4) = m * n) : 
  (m, n) = (7, 18) ∨ (m, n) = (8, 12) ∨ (m, n) = (9, 10) ∨ (m-6) * (n-6) = 12 := sorry

end number_of_valid_pairs_l2243_224337


namespace x_squared_plus_y_squared_geq_five_l2243_224319

theorem x_squared_plus_y_squared_geq_five (x y : ℝ) (h : abs (x - 2 * y) = 5) : x^2 + y^2 ≥ 5 := 
sorry

end x_squared_plus_y_squared_geq_five_l2243_224319


namespace imaginary_part_of_z_l2243_224328

open Complex -- open complex number functions

theorem imaginary_part_of_z (z : ℂ) (h : (z + 1) * (2 - I) = 5 * I) :
  z.im = 2 :=
sorry

end imaginary_part_of_z_l2243_224328


namespace fraction_add_eq_l2243_224355

theorem fraction_add_eq (x y : ℝ) (hx : y / x = 3 / 7) : (x + y) / x = 10 / 7 :=
by
  sorry

end fraction_add_eq_l2243_224355


namespace exists_four_integers_multiple_1984_l2243_224305

theorem exists_four_integers_multiple_1984 (a : Fin 97 → ℕ) (h_distinct : Function.Injective a) :
  ∃ i j k l : Fin 97, i ≠ j ∧ k ≠ l ∧ 1984 ∣ (a i - a j) * (a k - a l) :=
sorry

end exists_four_integers_multiple_1984_l2243_224305


namespace volume_of_prism_l2243_224390

variable (l w h : ℝ)

def area1 (l w : ℝ) : ℝ := l * w
def area2 (w h : ℝ) : ℝ := w * h
def area3 (l h : ℝ) : ℝ := l * h
def volume (l w h : ℝ) : ℝ := l * w * h

axiom cond1 : area1 l w = 15
axiom cond2 : area2 w h = 20
axiom cond3 : area3 l h = 30

theorem volume_of_prism : volume l w h = 30 * Real.sqrt 10 :=
by
  sorry

end volume_of_prism_l2243_224390


namespace smallest_whole_number_l2243_224361

theorem smallest_whole_number (m : ℕ) :
  m % 2 = 1 ∧
  m % 3 = 1 ∧
  m % 4 = 1 ∧
  m % 5 = 1 ∧
  m % 6 = 1 ∧
  m % 8 = 1 ∧
  m % 11 = 0 → 
  m = 1801 :=
by
  intros h
  sorry

end smallest_whole_number_l2243_224361


namespace time_difference_l2243_224349

-- Definitions
def time_chinese : ℕ := 5
def time_english : ℕ := 7

-- Statement to prove
theorem time_difference : time_english - time_chinese = 2 := by
  -- Proof goes here
  sorry

end time_difference_l2243_224349


namespace total_milks_taken_l2243_224332

def total_milks (chocolateMilk strawberryMilk regularMilk : Nat) : Nat :=
  chocolateMilk + strawberryMilk + regularMilk

theorem total_milks_taken :
  total_milks 2 15 3 = 20 :=
by
  sorry

end total_milks_taken_l2243_224332


namespace frac_subtraction_simplified_l2243_224326

-- Definitions of the fractions involved.
def frac1 : ℚ := 8 / 19
def frac2 : ℚ := 5 / 57

-- The primary goal is to prove the equality.
theorem frac_subtraction_simplified : frac1 - frac2 = 1 / 3 :=
by {
  -- Proof of the statement.
  sorry
}

end frac_subtraction_simplified_l2243_224326


namespace log_base_problem_l2243_224393

noncomputable def log_of_base (base value : ℝ) : ℝ := Real.log value / Real.log base

theorem log_base_problem (x : ℝ) (h : log_of_base 16 (x - 3) = 1 / 4) : 1 / log_of_base (x - 3) 2 = 1 := 
by
  sorry

end log_base_problem_l2243_224393


namespace mean_value_of_quadrilateral_interior_angles_l2243_224391

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l2243_224391


namespace min_value_expression_l2243_224363

theorem min_value_expression (x : ℝ) : 
  ∃ y : ℝ, (y = (x+2)*(x+3)*(x+4)*(x+5) + 3033) ∧ y ≥ 3032 ∧ 
  (∀ z : ℝ, (z = (x+2)*(x+3)*(x+4)*(x+5) + 3033) → z ≥ 3032) := 
sorry

end min_value_expression_l2243_224363


namespace find_years_l2243_224329

def sum_interest_years (P R : ℝ) (T : ℝ) : Prop :=
  (P * (R + 5) / 100 * T = P * R / 100 * T + 300) ∧ P = 600

theorem find_years {R : ℝ} {T : ℝ} (h1 : sum_interest_years 600 R T) : T = 10 :=
by
  -- proof omitted
  sorry

end find_years_l2243_224329


namespace find_numbers_l2243_224340

theorem find_numbers (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
                     (hxy_mul : 2000 ≤ x * y ∧ x * y < 3000) (hxy_add : 100 ≤ x + y ∧ x + y < 1000)
                     (h_digit_relation : x * y = 2000 + x + y) : 
                     (x = 24 ∧ y = 88) ∨ (x = 88 ∧ y = 24) ∨ (x = 30 ∧ y = 70) ∨ (x = 70 ∧ y = 30) :=
by
  -- The proof will go here
  sorry

end find_numbers_l2243_224340


namespace john_duck_price_l2243_224301

theorem john_duck_price
  (n_ducks : ℕ)
  (cost_per_duck : ℕ)
  (weight_per_duck : ℕ)
  (total_profit : ℕ)
  (total_cost : ℕ)
  (total_weight : ℕ)
  (total_revenue : ℕ)
  (price_per_pound : ℕ)
  (h1 : n_ducks = 30)
  (h2 : cost_per_duck = 10)
  (h3 : weight_per_duck = 4)
  (h4 : total_profit = 300)
  (h5 : total_cost = n_ducks * cost_per_duck)
  (h6 : total_weight = n_ducks * weight_per_duck)
  (h7 : total_revenue = total_cost + total_profit)
  (h8 : price_per_pound = total_revenue / total_weight) :
  price_per_pound = 5 := 
sorry

end john_duck_price_l2243_224301


namespace PQ_parallel_to_AB_3_times_l2243_224380

-- Definitions for the problem
structure Rectangle :=
  (A B C D : Type)
  (AB AD : ℝ)
  (P Q : ℝ → ℝ)
  (P_speed Q_speed : ℝ)
  (time : ℝ)

noncomputable def rectangle_properties (R : Rectangle) : Prop :=
  R.AB = 4 ∧
  R.AD = 12 ∧
  ∀ t, 0 ≤ t → t ≤ 12 → R.P t = t ∧  -- P moves from A to D at 1 cm/s
  R.Q_speed = 3 ∧                     -- Q moves at 3 cm/s
  ∀ t, R.Q t = R.Q_speed * t ∧             -- Q moves from C to B and back
  ∃ s1 s2 s3, R.P s1 = 4 ∧ R.P s2 = 8 ∧ R.P s3 = 12 ∧
  (R.Q s1 = 3 ∨ R.Q s1 = 1) ∧
  (R.Q s2 = 6 ∨ R.Q s2 = 2) ∧
  (R.Q s3 = 9 ∨ R.Q s3 = 0)

theorem PQ_parallel_to_AB_3_times : 
  ∀ (R : Rectangle), rectangle_properties R → 
  ∃ (times : ℕ), times = 3 :=
by
  sorry

end PQ_parallel_to_AB_3_times_l2243_224380


namespace monthly_sales_fraction_l2243_224315

theorem monthly_sales_fraction (V S_D T : ℝ) 
  (h1 : S_D = 6 * V) 
  (h2 : S_D = 0.35294117647058826 * T) 
  : V = (1 / 17) * T :=
sorry

end monthly_sales_fraction_l2243_224315


namespace anie_days_to_finish_task_l2243_224300

def extra_hours : ℕ := 5
def normal_work_hours : ℕ := 10
def total_project_hours : ℕ := 1500

theorem anie_days_to_finish_task : (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end anie_days_to_finish_task_l2243_224300


namespace linear_function_solution_l2243_224372

theorem linear_function_solution (f : ℝ → ℝ) (h1 : ∀ x, f (f x) = 16 * x - 15) :
  (∀ x, f x = 4 * x - 3) ∨ (∀ x, f x = -4 * x + 5) :=
sorry

end linear_function_solution_l2243_224372


namespace hal_battery_change_25th_time_l2243_224342

theorem hal_battery_change_25th_time (months_in_year : ℕ) 
    (battery_interval : ℕ) 
    (first_change_month : ℕ) 
    (change_count : ℕ) : 
    (battery_interval * (change_count-1)) % months_in_year + first_change_month % months_in_year = first_change_month % months_in_year :=
by
    have h1 : months_in_year = 12 := by sorry
    have h2 : battery_interval = 5 := by sorry
    have h3 : first_change_month = 5 := by sorry -- May is represented by 5 (0 = January, 1 = February, ..., 4 = April, 5 = May, ...)
    have h4 : change_count = 25 := by sorry
    sorry

end hal_battery_change_25th_time_l2243_224342


namespace find_a_l2243_224310

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end find_a_l2243_224310


namespace smallest_x_l2243_224307

-- Define 450 and provide its factorization.
def n1 := 450
def n1_factors := 2^1 * 3^2 * 5^2

-- Define 675 and provide its factorization.
def n2 := 675
def n2_factors := 3^3 * 5^2

-- State the theorem that proves the smallest x for the condition
theorem smallest_x (x : ℕ) (hx : 450 * x % 675 = 0) : x = 3 := sorry

end smallest_x_l2243_224307


namespace proof_problem_l2243_224399

-- Definitions of parallel and perpendicular relationships for lines and planes
def parallel (α β : Type) : Prop := sorry
def perpendicular (α β : Type) : Prop := sorry
def contained_in (m : Type) (α : Type) : Prop := sorry

-- Variables representing lines and planes
variables (l m n : Type) (α β : Type)

-- Assumptions from the conditions in step a)
variables 
  (h1 : m ≠ l)
  (h2 : α ≠ β)
  (h3 : parallel m n)
  (h4 : perpendicular m α)
  (h5 : perpendicular n β)

-- The goal is to prove that the planes α and β are parallel under the given conditions
theorem proof_problem : parallel α β :=
sorry

end proof_problem_l2243_224399


namespace num_monic_quadratic_trinomials_l2243_224314

noncomputable def count_monic_quadratic_trinomials : ℕ :=
  4489

theorem num_monic_quadratic_trinomials :
  count_monic_quadratic_trinomials = 4489 :=
by
  sorry

end num_monic_quadratic_trinomials_l2243_224314
