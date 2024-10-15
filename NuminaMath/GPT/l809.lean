import Mathlib

namespace NUMINAMATH_GPT_area_of_pentagon_m_n_l809_80938

noncomputable def m : ℤ := 12
noncomputable def n : ℤ := 11

theorem area_of_pentagon_m_n :
  let pentagon_area := (Real.sqrt m) + (Real.sqrt n)
  m + n = 23 :=
by
  have m_pos : m > 0 := by sorry
  have n_pos : n > 0 := by sorry
  sorry

end NUMINAMATH_GPT_area_of_pentagon_m_n_l809_80938


namespace NUMINAMATH_GPT_water_overflowed_calculation_l809_80971

/-- The water supply rate is 200 kilograms per hour. -/
def water_supply_rate : ℕ := 200

/-- The water tank capacity is 4000 kilograms. -/
def tank_capacity : ℕ := 4000

/-- The water runs for 24 hours. -/
def running_time : ℕ := 24

/-- Calculation for the kilograms of water that overflowed. -/
theorem water_overflowed_calculation :
  water_supply_rate * running_time - tank_capacity = 800 :=
by
  -- calculation skipped
  sorry

end NUMINAMATH_GPT_water_overflowed_calculation_l809_80971


namespace NUMINAMATH_GPT_simplify_complex_fraction_l809_80900

theorem simplify_complex_fraction : 
  (6 - 3 * Complex.I) / (-2 + 5 * Complex.I) = (-27 / 29) - (24 / 29) * Complex.I := 
by 
  sorry

end NUMINAMATH_GPT_simplify_complex_fraction_l809_80900


namespace NUMINAMATH_GPT_exchange_rate_lire_l809_80954

theorem exchange_rate_lire (x : ℕ) (h : 2500 / 2 = x / 5) : x = 6250 :=
by
  sorry

end NUMINAMATH_GPT_exchange_rate_lire_l809_80954


namespace NUMINAMATH_GPT_HVAC_cost_per_vent_l809_80979

/-- 
The cost of Joe's new HVAC system is $20,000. It includes 2 conditioning zones, each with 5 vents.
Prove that the cost of the system per vent is $2,000.
-/
theorem HVAC_cost_per_vent
    (cost : ℕ := 20000)
    (zones : ℕ := 2)
    (vents_per_zone : ℕ := 5)
    (total_vents : ℕ := zones * vents_per_zone) :
    (cost / total_vents) = 2000 := by
  sorry

end NUMINAMATH_GPT_HVAC_cost_per_vent_l809_80979


namespace NUMINAMATH_GPT_find_k_l809_80945

theorem find_k (x y k : ℝ) (h1 : x + y = 5 * k) (h2 : x - 2 * y = -k) (h3 : 2 * x - y = 8) : k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l809_80945


namespace NUMINAMATH_GPT_base_conversion_sum_l809_80957

-- Definition of conversion from base 13 to base 10
def base13_to_base10 (n : ℕ) : ℕ :=
  3 * (13^2) + 4 * (13^1) + 5 * (13^0)

-- Definition of conversion from base 14 to base 10 where C = 12 and D = 13
def base14_to_base10 (m : ℕ) : ℕ :=
  4 * (14^2) + 12 * (14^1) + 13 * (14^0)

theorem base_conversion_sum :
  base13_to_base10 345 + base14_to_base10 (4 * 14^2 + 12 * 14 + 13) = 1529 := 
by
  sorry -- proof to be provided

end NUMINAMATH_GPT_base_conversion_sum_l809_80957


namespace NUMINAMATH_GPT_original_money_in_wallet_l809_80961

-- Definitions based on the problem's conditions
def grandmother_gift : ℕ := 20
def aunt_gift : ℕ := 25
def uncle_gift : ℕ := 30
def cost_per_game : ℕ := 35
def number_of_games : ℕ := 3
def money_left : ℕ := 20

-- Calculations as specified in the solution
def birthday_money := grandmother_gift + aunt_gift + uncle_gift
def total_game_cost := cost_per_game * number_of_games
def total_money_before_purchase := total_game_cost + money_left

-- Proof that the original amount of money in Geoffrey's wallet
-- was €50 before he got the birthday money and made the purchase.
theorem original_money_in_wallet : total_money_before_purchase - birthday_money = 50 := by
  sorry

end NUMINAMATH_GPT_original_money_in_wallet_l809_80961


namespace NUMINAMATH_GPT_slope_range_l809_80946

open Real

theorem slope_range (k : ℝ) :
  (∃ b : ℝ, 
    ∃ x1 x2 x3 : ℝ,
      (x1 + x2 + x3 = 0) ∧
      (x1 ≥ 0) ∧ (x2 ≥ 0) ∧ (x3 < 0) ∧
      ((kx1 + b) = ((x1 + 1) / (|x1| + 1))) ∧
      ((kx2 + b) = ((x2 + 1) / (|x2| + 1))) ∧
      ((kx3 + b) = ((x3 + 1) / (|x3| + 1)))) →
  (0 < k ∧ k < (2 / 9)) :=
sorry

end NUMINAMATH_GPT_slope_range_l809_80946


namespace NUMINAMATH_GPT_number_of_female_workers_l809_80948

theorem number_of_female_workers (M F : ℕ) (M_no F_no : ℝ) 
  (hM : M = 112)
  (h1 : M_no = 0.40 * M)
  (h2 : F_no = 0.25 * F)
  (h3 : M_no / (M_no + F_no) = 0.30)
  (h4 : F_no / (M_no + F_no) = 0.70)
  : F = 420 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_female_workers_l809_80948


namespace NUMINAMATH_GPT_cost_per_acre_proof_l809_80982

def cost_of_land (tac tl : ℕ) (hc hcc hcp heq : ℕ) (ttl : ℕ) : ℕ := ttl - (hc + hcc + hcp + heq)

def cost_per_acre (total_land : ℕ) (cost_land : ℕ) : ℕ := cost_land / total_land

theorem cost_per_acre_proof (tac tl hc hcc hcp heq ttl epl : ℕ) 
  (h1 : tac = 30)
  (h2 : hc = 120000)
  (h3 : hcc = 20 * 1000)
  (h4 : hcp = 100 * 5)
  (h5 : heq = 6 * 100 + 6000)
  (h6 : ttl = 147700) :
  cost_per_acre tac (cost_of_land tac tl hc hcc hcp heq ttl) = epl := by
  sorry

end NUMINAMATH_GPT_cost_per_acre_proof_l809_80982


namespace NUMINAMATH_GPT_fred_blue_marbles_l809_80992

theorem fred_blue_marbles (tim_marbles : ℕ) (fred_marbles : ℕ) (h1 : tim_marbles = 5) (h2 : fred_marbles = 22 * tim_marbles) : fred_marbles = 110 :=
by
  sorry

end NUMINAMATH_GPT_fred_blue_marbles_l809_80992


namespace NUMINAMATH_GPT_remainder_b96_div_50_l809_80917

theorem remainder_b96_div_50 (b : ℕ → ℕ) (h : ∀ n, b n = 7^n + 9^n) : b 96 % 50 = 2 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_remainder_b96_div_50_l809_80917


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l809_80925

theorem quadratic_inequality_solution
  (a : ℝ) :
  (∀ x : ℝ, (a-2)*x^2 + 2*(a-2)*x - 4 ≤ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l809_80925


namespace NUMINAMATH_GPT_men_per_table_l809_80960

theorem men_per_table 
  (num_tables : ℕ) 
  (women_per_table : ℕ) 
  (total_customers : ℕ) 
  (h1 : num_tables = 9) 
  (h2 : women_per_table = 7) 
  (h3 : total_customers = 90)
  : (total_customers - num_tables * women_per_table) / num_tables = 3 :=
by
  sorry

end NUMINAMATH_GPT_men_per_table_l809_80960


namespace NUMINAMATH_GPT_min_abs_expr1_min_abs_expr2_l809_80928

theorem min_abs_expr1 (x : ℝ) : |x - 4| + |x + 2| ≥ 6 := sorry

theorem min_abs_expr2 (x : ℝ) : |(5 / 6) * x - 1| + |(1 / 2) * x - 1| + |(2 / 3) * x - 1| ≥ 1 / 2 := sorry

end NUMINAMATH_GPT_min_abs_expr1_min_abs_expr2_l809_80928


namespace NUMINAMATH_GPT_tour_groups_and_savings_minimum_people_for_savings_l809_80910

theorem tour_groups_and_savings (x y : ℕ) (m : ℕ):
  (x + y = 102) ∧ (45 * x + 50 * y - 40 * 102 = 730) → 
  (x = 58 ∧ y = 44) :=
by
  sorry

theorem minimum_people_for_savings (m : ℕ):
  (∀ m, m < 50 → 50 * m > 45 * 51) → 
  (m ≥ 46) :=
by
  sorry

end NUMINAMATH_GPT_tour_groups_and_savings_minimum_people_for_savings_l809_80910


namespace NUMINAMATH_GPT_minimum_value_x3_plus_y3_minus_5xy_l809_80908

theorem minimum_value_x3_plus_y3_minus_5xy (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  x^3 + y^3 - 5 * x * y ≥ -125 / 27 := 
sorry

end NUMINAMATH_GPT_minimum_value_x3_plus_y3_minus_5xy_l809_80908


namespace NUMINAMATH_GPT_find_n_l809_80944

theorem find_n (n : ℕ) :
  (∀ k : ℕ, k > 0 → k^2 + (n / k^2) ≥ 1991) ∧ (∃ k : ℕ, k > 0 ∧ k^2 + (n / k^2) < 1992) ↔ 967 * 1024 ≤ n ∧ n < 968 * 1024 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l809_80944


namespace NUMINAMATH_GPT_ratio_length_to_breadth_l809_80942

theorem ratio_length_to_breadth (l b : ℕ) (h1 : b = 14) (h2 : l * b = 588) : l / b = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_length_to_breadth_l809_80942


namespace NUMINAMATH_GPT_count_words_200_l809_80989

theorem count_words_200 : 
  let single_word_numbers := 29
  let compound_words_21_to_99 := 144
  let compound_words_100_to_199 := 54 + 216
  single_word_numbers + compound_words_21_to_99 + compound_words_100_to_199 = 443 :=
by
  sorry

end NUMINAMATH_GPT_count_words_200_l809_80989


namespace NUMINAMATH_GPT_sum_of_consecutive_numbers_with_lcm_168_l809_80924

theorem sum_of_consecutive_numbers_with_lcm_168 (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : Nat.lcm a (Nat.lcm b c) = 168) : a + b + c = 21 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_numbers_with_lcm_168_l809_80924


namespace NUMINAMATH_GPT_smallest_n_satisfying_condition_l809_80983

theorem smallest_n_satisfying_condition :
  ∃ n : ℕ, (n > 1) ∧ (∀ i : ℕ, i ≥ 1 → i < n → (∃ k : ℕ, i + (i+1) = k^2)) ∧ n = 8 :=
sorry

end NUMINAMATH_GPT_smallest_n_satisfying_condition_l809_80983


namespace NUMINAMATH_GPT_max_value_of_expression_l809_80922

noncomputable def maximum_value (x y z : ℝ) := 8 * x + 3 * y + 10 * z

theorem max_value_of_expression :
  ∀ (x y z : ℝ), 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 → maximum_value x y z ≤ (Real.sqrt 481) / 6 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l809_80922


namespace NUMINAMATH_GPT_id_tags_divided_by_10_l809_80935

def uniqueIDTags (chars : List Char) (counts : Char → Nat) : Nat :=
  let permsWithoutRepetition := 
    Nat.factorial 7 / Nat.factorial (7 - 5)
  let repeatedCharTagCount := 10 * 10 * 6
  permsWithoutRepetition + repeatedCharTagCount

theorem id_tags_divided_by_10 :
  uniqueIDTags ['M', 'A', 'T', 'H', '2', '0', '3'] (fun c =>
    if c = 'M' then 1 else
    if c = 'A' then 1 else
    if c = 'T' then 1 else
    if c = 'H' then 1 else
    if c = '2' then 2 else
    if c = '0' then 1 else
    if c = '3' then 1 else 0) / 10 = 312 :=
by
  sorry

end NUMINAMATH_GPT_id_tags_divided_by_10_l809_80935


namespace NUMINAMATH_GPT_range_of_k_intersection_l809_80903

theorem range_of_k_intersection (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (k^2 - 1) * x1^2 + 4 * k * x1 + 10 = 0 ∧ (k^2 - 1) * x2^2 + 4 * k * x2 + 10 = 0) ↔ (-1 < k ∧ k < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_intersection_l809_80903


namespace NUMINAMATH_GPT_box_contains_1_8_grams_child_ingests_0_1_grams_l809_80934

-- Define the conditions
def packet_weight : ℝ := 0.2
def packets_in_box : ℕ := 9
def half_a_packet : ℝ := 0.5

-- Prove that a box contains 1.8 grams of "acetaminophen"
theorem box_contains_1_8_grams : packets_in_box * packet_weight = 1.8 :=
by
  sorry

-- Prove that a child will ingest 0.1 grams of "acetaminophen" if they take half a packet
theorem child_ingests_0_1_grams : half_a_packet * packet_weight = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_box_contains_1_8_grams_child_ingests_0_1_grams_l809_80934


namespace NUMINAMATH_GPT_no_valid_six_digit_palindrome_years_l809_80953

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def is_six_digit_palindrome (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ is_palindrome n

noncomputable def is_four_digit_prime_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ is_palindrome n ∧ is_prime n

noncomputable def is_two_digit_prime_palindrome (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ is_palindrome n ∧ is_prime n

theorem no_valid_six_digit_palindrome_years :
  ∀ N : ℕ, is_six_digit_palindrome N →
  ¬ ∃ (p q : ℕ), is_four_digit_prime_palindrome p ∧ is_two_digit_prime_palindrome q ∧ N = p * q := 
sorry

end NUMINAMATH_GPT_no_valid_six_digit_palindrome_years_l809_80953


namespace NUMINAMATH_GPT_circle_equation_is_correct_l809_80997

def center : Int × Int := (-3, 4)
def radius : Int := 3
def circle_standard_equation (x y : Int) : Int :=
  (x + 3)^2 + (y - 4)^2

theorem circle_equation_is_correct :
  circle_standard_equation x y = 9 :=
sorry

end NUMINAMATH_GPT_circle_equation_is_correct_l809_80997


namespace NUMINAMATH_GPT_prob_A_wins_match_expected_games_won_variance_games_won_l809_80930

-- Definitions of probabilities
def prob_A_win := 0.6
def prob_B_win := 0.4

-- Prove that the probability of A winning the match is 0.648
theorem prob_A_wins_match : 
  prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win = 0.648 :=
  sorry

-- Define the expected number of games won by A
noncomputable def expected_games_won_by_A := 
  0 * (prob_B_win * prob_B_win) + 1 * (2 * prob_A_win * prob_B_win * prob_B_win) + 
  2 * (prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win)

-- Prove the expected number of games won by A is 1.5
theorem expected_games_won : 
  expected_games_won_by_A = 1.5 :=
  sorry

-- Define the variance of the number of games won by A
noncomputable def variance_games_won_by_A := 
  (prob_B_win * prob_B_win) * (0 - 1.5)^2 + 
  (2 * prob_A_win * prob_B_win * prob_B_win) * (1 - 1.5)^2 + 
  (prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win) * (2 - 1.5)^2

-- Prove the variance of the number of games won by A is 0.57
theorem variance_games_won : 
  variance_games_won_by_A = 0.57 :=
  sorry

end NUMINAMATH_GPT_prob_A_wins_match_expected_games_won_variance_games_won_l809_80930


namespace NUMINAMATH_GPT_symmetrical_point_wrt_x_axis_l809_80999

theorem symmetrical_point_wrt_x_axis (x y : ℝ) (P_symmetrical : (ℝ × ℝ)) (hx : x = -1) (hy : y = 2) : 
  P_symmetrical = (x, -y) → P_symmetrical = (-1, -2) :=
by
  intros h
  rw [hx, hy] at h
  exact h

end NUMINAMATH_GPT_symmetrical_point_wrt_x_axis_l809_80999


namespace NUMINAMATH_GPT_cubic_sum_l809_80932

theorem cubic_sum (a b c : ℤ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 11) (h3 : a * b * c = -6) :
  a^3 + b^3 + c^3 = 223 :=
by
  sorry

end NUMINAMATH_GPT_cubic_sum_l809_80932


namespace NUMINAMATH_GPT_inequality_proof_l809_80921

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + a + 1) * (b^2 + b + 1) * (c^2 + c + 1)) / (a * b * c) ≥ 27 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l809_80921


namespace NUMINAMATH_GPT_rectangle_diagonal_l809_80987

theorem rectangle_diagonal (l w : ℝ) (hl : l = 40) (hw : w = 40 * Real.sqrt 2) :
  Real.sqrt (l^2 + w^2) = 40 * Real.sqrt 3 :=
by
  rw [hl, hw]
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_l809_80987


namespace NUMINAMATH_GPT_asymptote_of_hyperbola_l809_80998

theorem asymptote_of_hyperbola (x y : ℝ) (h : (x^2 / 16) - (y^2 / 25) = 1) : 
  y = (5 / 4) * x :=
sorry

end NUMINAMATH_GPT_asymptote_of_hyperbola_l809_80998


namespace NUMINAMATH_GPT_aarti_bina_work_l809_80904

theorem aarti_bina_work (days_aarti : ℚ) (days_bina : ℚ) (D : ℚ)
  (ha : days_aarti = 5) (hb : days_bina = 8)
  (rate_aarti : 1 / days_aarti = 1/5) 
  (rate_bina : 1 / days_bina = 1/8)
  (combine_rate : (1 / days_aarti) + (1 / days_bina) = 13 / 40) :
  3 / (13 / 40) = 120 / 13 := 
by
  sorry

end NUMINAMATH_GPT_aarti_bina_work_l809_80904


namespace NUMINAMATH_GPT_find_x_l809_80966

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
sorry

end NUMINAMATH_GPT_find_x_l809_80966


namespace NUMINAMATH_GPT_trigonometric_identity_l809_80920

open Real

theorem trigonometric_identity (θ : ℝ) (h : π / 4 < θ ∧ θ < π / 2) :
  2 * cos θ + sqrt (1 - 2 * sin (π - θ) * cos θ) = sin θ + cos θ :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l809_80920


namespace NUMINAMATH_GPT_total_slides_used_l809_80984

theorem total_slides_used (duration : ℕ) (initial_slides : ℕ) (initial_time : ℕ) (constant_rate : ℕ) (total_time: ℕ)
  (H1 : duration = 50)
  (H2 : initial_slides = 4)
  (H3 : initial_time = 2)
  (H4 : constant_rate = initial_slides / initial_time)
  (H5 : total_time = duration) 
  : (constant_rate * total_time) = 100 := 
by
  sorry

end NUMINAMATH_GPT_total_slides_used_l809_80984


namespace NUMINAMATH_GPT_find_D_l809_80955

noncomputable def Point : Type := ℝ × ℝ

-- Given points A, B, and C
def A : Point := (-2, 0)
def B : Point := (6, 8)
def C : Point := (8, 6)

-- Condition: AB parallel to DC and AD parallel to BC, which means it is a parallelogram
def is_parallelogram (A B C D : Point) : Prop :=
  ((B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2)) ∧
  ((C.1 - B.1, C.2 - B.2) = (D.1 - A.1, D.2 - A.2))

-- Proves that with given A, B, and C, D should be (0, -2)
theorem find_D : ∃ D : Point, is_parallelogram A B C D ∧ D = (0, -2) :=
  by sorry

end NUMINAMATH_GPT_find_D_l809_80955


namespace NUMINAMATH_GPT_max_students_distribute_pens_pencils_l809_80993

noncomputable def gcd_example : ℕ :=
  Nat.gcd 1340 1280

theorem max_students_distribute_pens_pencils : gcd_example = 20 :=
sorry

end NUMINAMATH_GPT_max_students_distribute_pens_pencils_l809_80993


namespace NUMINAMATH_GPT_vertex_of_parabola_on_x_axis_l809_80977

theorem vertex_of_parabola_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, (x^2 - 6*x + c = 0)) ↔ c = 9 :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_on_x_axis_l809_80977


namespace NUMINAMATH_GPT_general_term_of_c_l809_80913

theorem general_term_of_c (a b : ℕ → ℕ) (c : ℕ → ℕ) : 
  (∀ n, a n = 2 ^ n) →
  (∀ n, b n = 3 * n + 2) →
  (∀ n, ∃ m k, a n = b m ∧ n = 2 * k + 1 → c k = a n) →
  ∀ n, c n = 2 ^ (2 * n + 1) :=
by
  intros ha hb hc n
  have h' := hc n
  sorry

end NUMINAMATH_GPT_general_term_of_c_l809_80913


namespace NUMINAMATH_GPT_option_b_correct_l809_80975

variables {m n : Line} {α β : Plane}

-- Define the conditions as per the problem.
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular_to_plane (p1 p2 : Plane) : Prop := sorry
def lines_perpendicular (l1 l2 : Line) : Prop := sorry

theorem option_b_correct (h1 : line_perpendicular_to_plane m α)
                         (h2 : line_perpendicular_to_plane n β)
                         (h3 : lines_perpendicular m n) :
                         plane_perpendicular_to_plane α β :=
sorry

end NUMINAMATH_GPT_option_b_correct_l809_80975


namespace NUMINAMATH_GPT_condition_sufficiency_l809_80918

theorem condition_sufficiency (x : ℝ) :
  (2 ≤ x ∧ x ≤ 3) → (x < -3 ∨ x ≥ 1) ∧ (∃ x : ℝ, (x < -3 ∨ x ≥ 1) ∧ ¬(2 ≤ x ∧ x ≤ 3)) :=
by
  sorry

end NUMINAMATH_GPT_condition_sufficiency_l809_80918


namespace NUMINAMATH_GPT_average_of_numbers_eq_x_l809_80907

theorem average_of_numbers_eq_x (x : ℝ) (h : (2 + x + 10) / 3 = x) : x = 6 := 
by sorry

end NUMINAMATH_GPT_average_of_numbers_eq_x_l809_80907


namespace NUMINAMATH_GPT_solve_rational_eq_l809_80943

theorem solve_rational_eq (x : ℝ) :
  (1 / (x^2 + 14*x - 36)) + (1 / (x^2 + 5*x - 14)) + (1 / (x^2 - 16*x - 36)) = 0 ↔ 
  x = 9 ∨ x = -4 ∨ x = 12 ∨ x = 3 :=
sorry

end NUMINAMATH_GPT_solve_rational_eq_l809_80943


namespace NUMINAMATH_GPT_number_less_than_neg_two_l809_80940

theorem number_less_than_neg_two : ∃ x : Int, x = -2 - 1 := 
by
  use -3
  sorry

end NUMINAMATH_GPT_number_less_than_neg_two_l809_80940


namespace NUMINAMATH_GPT_blue_part_length_l809_80990

variable (total_length : ℝ) (black_part white_part blue_part : ℝ)

-- Conditions
axiom h1 : black_part = 1 / 8 * total_length
axiom h2 : white_part = 1 / 2 * (total_length - black_part)
axiom h3 : total_length = 8

theorem blue_part_length : blue_part = total_length - black_part - white_part :=
by
  sorry

end NUMINAMATH_GPT_blue_part_length_l809_80990


namespace NUMINAMATH_GPT_min_elements_of_B_l809_80927

def A (k : ℝ) : Set ℝ :=
if k < 0 then {x | (k / 4 + 9 / (4 * k) + 3) < x ∧ x < 11 / 2}
else if k = 0 then {x | x < 11 / 2}
else if 0 < k ∧ k < 1 ∨ k > 9 then {x | x < 11 / 2 ∨ x > k / 4 + 9 / (4 * k) + 3}
else if 1 ≤ k ∧ k ≤ 9 then {x | x < k / 4 + 9 / (4 * k) + 3 ∨ x > 11 / 2}
else ∅

def B (k : ℝ) : Set ℤ := {x : ℤ | ↑x ∈ A k}

theorem min_elements_of_B (k : ℝ) (hk : k < 0) : 
  B k = {2, 3, 4, 5} :=
sorry

end NUMINAMATH_GPT_min_elements_of_B_l809_80927


namespace NUMINAMATH_GPT_pumac_grader_remainder_l809_80902

/-- A PUMaC grader is grading the submissions of forty students s₁, s₂, ..., s₄₀ for the
    individual finals round, which has three problems.
    After grading a problem of student sᵢ, the grader either:
    * grades another problem of the same student, or
    * grades the same problem of the student sᵢ₋₁ or sᵢ₊₁ (if i > 1 and i < 40, respectively).
    He grades each problem exactly once, starting with the first problem of s₁
    and ending with the third problem of s₄₀.
    Let N be the number of different orders the grader may grade the students’ problems in this way.
    Prove: N ≡ 78 [MOD 100] -/

noncomputable def grading_orders_mod : ℕ := 2 * (3 ^ 38) % 100

theorem pumac_grader_remainder :
  grading_orders_mod = 78 :=
by
  sorry

end NUMINAMATH_GPT_pumac_grader_remainder_l809_80902


namespace NUMINAMATH_GPT_rachel_hw_diff_l809_80909

-- Definitions based on the conditions of the problem
def math_hw_pages := 15
def reading_hw_pages := 6

-- The statement we need to prove, including the conditions
theorem rachel_hw_diff : 
  math_hw_pages - reading_hw_pages = 9 := 
by
  sorry

end NUMINAMATH_GPT_rachel_hw_diff_l809_80909


namespace NUMINAMATH_GPT_tank_empty_time_when_inlet_open_l809_80965

-- Define the conditions
def leak_empty_time : ℕ := 6
def tank_capacity : ℕ := 4320
def inlet_rate_per_minute : ℕ := 6

-- Calculate rates from conditions
def leak_rate_per_hour : ℕ := tank_capacity / leak_empty_time
def inlet_rate_per_hour : ℕ := inlet_rate_per_minute * 60

-- Proof Problem: Prove the time for the tank to empty when both leak and inlet are open
theorem tank_empty_time_when_inlet_open :
  tank_capacity / (leak_rate_per_hour - inlet_rate_per_hour) = 12 :=
by
  sorry

end NUMINAMATH_GPT_tank_empty_time_when_inlet_open_l809_80965


namespace NUMINAMATH_GPT_factorize_expr1_factorize_expr2_l809_80985

open BigOperators

/-- Given m and n, prove that m^3 n - 9 m n can be factorized as mn(m + 3)(m - 3). -/
theorem factorize_expr1 (m n : ℤ) : m^3 * n - 9 * m * n = n * m * (m + 3) * (m - 3) :=
sorry

/-- Given a, prove that a^3 + a - 2a^2 can be factorized as a(a - 1)^2. -/
theorem factorize_expr2 (a : ℤ) : a^3 + a - 2 * a^2 = a * (a - 1)^2 :=
sorry

end NUMINAMATH_GPT_factorize_expr1_factorize_expr2_l809_80985


namespace NUMINAMATH_GPT_B_finish_in_54_days_l809_80901

-- Definitions based on conditions
variables (A B : ℝ) -- A and B are the amount of work done in one day
axiom h1 : A = 2 * B -- A is twice as good as workman as B
axiom h2 : (A + B) * 18 = 1 -- Together, A and B finish the piece of work in 18 days

-- Prove that B alone will finish the work in 54 days.
theorem B_finish_in_54_days : (1 / B) = 54 :=
by 
  sorry

end NUMINAMATH_GPT_B_finish_in_54_days_l809_80901


namespace NUMINAMATH_GPT_find_s_when_t_is_64_l809_80967

theorem find_s_when_t_is_64 (s : ℝ) (t : ℝ) (h1 : t = 8 * s^3) (h2 : t = 64) : s = 2 :=
by
  -- Proof will be written here
  sorry

end NUMINAMATH_GPT_find_s_when_t_is_64_l809_80967


namespace NUMINAMATH_GPT_num_balls_picked_l809_80952

-- Definitions based on the conditions
def numRedBalls : ℕ := 4
def numBlueBalls : ℕ := 3
def numGreenBalls : ℕ := 2
def totalBalls : ℕ := numRedBalls + numBlueBalls + numGreenBalls
def probFirstRed : ℚ := numRedBalls / totalBalls
def probSecondRed : ℚ := (numRedBalls - 1) / (totalBalls - 1)

-- Theorem stating the problem
theorem num_balls_picked :
  probFirstRed * probSecondRed = 1 / 6 → 
  (∃ (n : ℕ), n = 2) :=
by 
  sorry

end NUMINAMATH_GPT_num_balls_picked_l809_80952


namespace NUMINAMATH_GPT_intersection_correct_l809_80949

open Set

def M := {x : ℝ | x^2 + x - 6 < 0}
def N := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def intersection := (M ∩ N) = {x : ℝ | 1 ≤ x ∧ x < 2}

theorem intersection_correct : intersection := by
  sorry

end NUMINAMATH_GPT_intersection_correct_l809_80949


namespace NUMINAMATH_GPT_solution_set_of_inequality_l809_80936

variable (a b x : ℝ)
variable (h1 : a < 0)

theorem solution_set_of_inequality (h : a * x + b < 0) : x > -b / a :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l809_80936


namespace NUMINAMATH_GPT_part_one_retail_wholesale_l809_80958

theorem part_one_retail_wholesale (x : ℕ) (wholesale : ℕ) : 
  70 * x + 40 * wholesale = 4600 ∧ x + wholesale = 100 → x = 20 ∧ wholesale = 80 :=
by
  sorry

end NUMINAMATH_GPT_part_one_retail_wholesale_l809_80958


namespace NUMINAMATH_GPT_total_gallons_of_seed_l809_80995

-- Condition (1): The area of the football field is 8000 square meters.
def area_football_field : ℝ := 8000

-- Condition (2): Each square meter needs 4 times as much seed as fertilizer.
def seed_to_fertilizer_ratio : ℝ := 4

-- Condition (3): Carson uses 240 gallons of seed and fertilizer combined for every 2000 square meters.
def combined_usage_per_2000sqm : ℝ := 240
def area_unit : ℝ := 2000

-- Target: Prove that the total gallons of seed Carson uses for the entire field is 768 gallons.
theorem total_gallons_of_seed : seed_to_fertilizer_ratio * area_football_field / area_unit / (seed_to_fertilizer_ratio + 1) * combined_usage_per_2000sqm * (area_football_field / area_unit) = 768 :=
sorry

end NUMINAMATH_GPT_total_gallons_of_seed_l809_80995


namespace NUMINAMATH_GPT_find_k_l809_80951

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 8
def g (x : ℝ) (k : ℝ) : ℝ := x ^ 2 - k * x + 3

theorem find_k : 
  (f 5 - g 5 k = 12) → k = -53 / 5 :=
by
  intro hyp
  sorry

end NUMINAMATH_GPT_find_k_l809_80951


namespace NUMINAMATH_GPT_pounds_of_oranges_l809_80939

noncomputable def price_of_pounds_oranges (E O : ℝ) (P : ℕ) : Prop :=
  let current_total_price := E
  let increased_total_price := 1.09 * E + 1.06 * (O * P)
  (increased_total_price - current_total_price) = 15

theorem pounds_of_oranges (E O : ℝ) (P : ℕ): 
  E = O * P ∧ 
  (price_of_pounds_oranges E O P) → 
  P = 100 := 
by
  sorry

end NUMINAMATH_GPT_pounds_of_oranges_l809_80939


namespace NUMINAMATH_GPT_range_of_a_l809_80926

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a * x - 1 else a / x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

def func_increasing_on_R (a : ℝ) : Prop :=
  is_increasing_on (f a) Set.univ

theorem range_of_a (a : ℝ) : func_increasing_on_R a ↔ a < -2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l809_80926


namespace NUMINAMATH_GPT_original_number_l809_80969

theorem original_number (x : ℤ) (h : 5 * x - 9 = 51) : x = 12 :=
sorry

end NUMINAMATH_GPT_original_number_l809_80969


namespace NUMINAMATH_GPT_odd_operations_l809_80962

theorem odd_operations (a b : ℤ) (ha : ∃ k : ℤ, a = 2 * k + 1) (hb : ∃ j : ℤ, b = 2 * j + 1) :
  (∃ k : ℤ, (a * b) = 2 * k + 1) ∧ (∃ m : ℤ, a^2 = 2 * m + 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_odd_operations_l809_80962


namespace NUMINAMATH_GPT_determine_all_functions_l809_80968

-- Define the natural numbers (ℕ) as positive integers
def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, k * k = x

theorem determine_all_functions (g : ℕ → ℕ) :
  (∀ m n : ℕ, is_perfect_square ((g m + n) * (m + g n))) →
  ∃ c : ℕ, ∀ n : ℕ, g n = n + c :=
by
  sorry

end NUMINAMATH_GPT_determine_all_functions_l809_80968


namespace NUMINAMATH_GPT_range_of_a_l809_80933

noncomputable def y (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * x ^ 2 - 2 * x

noncomputable def y' (x : ℝ) (a : ℝ) : ℝ := 1 / x + 2 * a * x - 2

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → y' x a ≥ 0) ↔ a ≥ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l809_80933


namespace NUMINAMATH_GPT_sum_even_less_100_correct_l809_80974

-- Define the sequence of even, positive integers less than 100
def even_seq (n : ℕ) : Prop := n % 2 = 0 ∧ 0 < n ∧ n < 100

-- Sum of the first n positive integers
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Sum of the even, positive integers less than 100
def sum_even_less_100 : ℕ := 2 * sum_n 49

theorem sum_even_less_100_correct : sum_even_less_100 = 2450 := by
  sorry

end NUMINAMATH_GPT_sum_even_less_100_correct_l809_80974


namespace NUMINAMATH_GPT_probability_same_color_is_27_over_100_l809_80923

def num_sides_die1 := 20
def num_sides_die2 := 20

def maroon_die1 := 5
def teal_die1 := 6
def cyan_die1 := 7
def sparkly_die1 := 1
def silver_die1 := 1

def maroon_die2 := 4
def teal_die2 := 6
def cyan_die2 := 7
def sparkly_die2 := 1
def silver_die2 := 2

noncomputable def probability_same_color : ℚ :=
  (maroon_die1 * maroon_die2 + teal_die1 * teal_die2 + cyan_die1 * cyan_die2 + sparkly_die1 * sparkly_die2 + silver_die1 * silver_die2) /
  (num_sides_die1 * num_sides_die2)

theorem probability_same_color_is_27_over_100 :
  probability_same_color = 27 / 100 := 
sorry

end NUMINAMATH_GPT_probability_same_color_is_27_over_100_l809_80923


namespace NUMINAMATH_GPT_average_TV_sets_in_shops_l809_80911

def shop_a := 20
def shop_b := 30
def shop_c := 60
def shop_d := 80
def shop_e := 50
def total_shops := 5

theorem average_TV_sets_in_shops : (shop_a + shop_b + shop_c + shop_d + shop_e) / total_shops = 48 :=
by
  have h1 : shop_a + shop_b + shop_c + shop_d + shop_e = 240
  { sorry }
  have h2 : 240 / total_shops = 48
  { sorry }
  exact Eq.trans (congrArg (fun x => x / total_shops) h1) h2

end NUMINAMATH_GPT_average_TV_sets_in_shops_l809_80911


namespace NUMINAMATH_GPT_unique_positive_integer_n_l809_80956

theorem unique_positive_integer_n (n x : ℕ) (hx : x > 0) (hn : n = 2 ^ (2 * x - 1) - 5 * x - 3 ∧ n = (2 ^ (x-1) - 1) * (2 ^ x + 1)) : n = 2015 := by
  sorry

end NUMINAMATH_GPT_unique_positive_integer_n_l809_80956


namespace NUMINAMATH_GPT_obtuse_triangle_k_values_l809_80937

theorem obtuse_triangle_k_values (k : ℕ) (h : k > 0) :
  (∃ k, (5 < k ∧ k ≤ 12) ∨ (21 ≤ k ∧ k < 29)) → ∃ n : ℕ, n = 15 :=
by
  sorry

end NUMINAMATH_GPT_obtuse_triangle_k_values_l809_80937


namespace NUMINAMATH_GPT_circle_equation_exists_l809_80978

-- Define the necessary conditions
def tangent_to_x_axis (r b : ℝ) : Prop :=
  r^2 = b^2

def center_on_line (a b : ℝ) : Prop :=
  3 * a - b = 0

def intersects_formula (a b r : ℝ) : Prop :=
  2 * r^2 = (a - b)^2 + 14

-- Main theorem combining the conditions and proving the circles' equations
theorem circle_equation_exists (a b r : ℝ) :
  tangent_to_x_axis r b →
  center_on_line a b →
  intersects_formula a b r →
  ((x - 1)^2 + (y - 3)^2 = 9 ∨ (x + 1)^2 + (y + 3)^2 = 9) :=
by
  intros h_tangent h_center h_intersects
  sorry

end NUMINAMATH_GPT_circle_equation_exists_l809_80978


namespace NUMINAMATH_GPT_length_of_PB_l809_80991

theorem length_of_PB 
  (AB BC : ℝ) 
  (PA PD PC PB : ℝ)
  (h1 : AB = 2 * BC) 
  (h2 : PA = 5) 
  (h3 : PD = 12) 
  (h4 : PC = 13) 
  (h5 : PA^2 + PB^2 = (AB^2 + BC^2) / 5) -- derived from question
  (h6 : PB^2 = ((2 * BC)^2) - PA^2) : 
  PB = 10.5 :=
by 
  -- We would insert proof steps here (not required as per instructions)
  sorry

end NUMINAMATH_GPT_length_of_PB_l809_80991


namespace NUMINAMATH_GPT_f_1982_eq_660_l809_80963

def f : ℕ → ℕ := sorry

axiom h1 : ∀ m n : ℕ, f (m + n) - f m - f n = 0 ∨ f (m + n) - f m - f n = 1
axiom h2 : f 2 = 0
axiom h3 : f 3 > 0
axiom h4 : f 9999 = 3333

theorem f_1982_eq_660 : f 1982 = 660 := sorry

end NUMINAMATH_GPT_f_1982_eq_660_l809_80963


namespace NUMINAMATH_GPT_find_angle_l809_80914

theorem find_angle (x : ℝ) (h : 180 - x = 6 * (90 - x)) : x = 72 := 
by 
    sorry

end NUMINAMATH_GPT_find_angle_l809_80914


namespace NUMINAMATH_GPT_find_coefficients_l809_80964

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem find_coefficients 
  (A B Q C P : V) 
  (hQ : Q = (5 / 7 : ℝ) • A + (2 / 7 : ℝ) • B)
  (hC : C = A + 2 • B)
  (hP : P = Q + C) : 
  ∃ s v : ℝ, P = s • A + v • B ∧ s = 12 / 7 ∧ v = 16 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_coefficients_l809_80964


namespace NUMINAMATH_GPT_initial_ratio_men_to_women_l809_80973

theorem initial_ratio_men_to_women (M W : ℕ) (h1 : (W - 3) * 2 = 24) (h2 : 14 - 2 = M) : M / gcd M W = 4 ∧ W / gcd M W = 5 := by 
  sorry

end NUMINAMATH_GPT_initial_ratio_men_to_women_l809_80973


namespace NUMINAMATH_GPT_evaluate_difference_floor_squares_l809_80986

theorem evaluate_difference_floor_squares (x : ℝ) (h : x = 15.3) : ⌊x^2⌋ - ⌊x⌋^2 = 9 := by
  sorry

end NUMINAMATH_GPT_evaluate_difference_floor_squares_l809_80986


namespace NUMINAMATH_GPT_find_varphi_intervals_of_increase_l809_80929

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem find_varphi (φ : ℝ) (h1 : -Real.pi < φ) (h2 : φ < 0)
  (h3 : ∃ k : ℤ, 2 * (Real.pi / 8) + φ = (Real.pi / 2) + k * Real.pi) :
  φ = -3 * Real.pi / 4 :=
sorry

theorem intervals_of_increase (m : ℤ) :
  ∀ x : ℝ, (π / 8 + m * π ≤ x ∧ x ≤ 5 * π / 8 + m * π) ↔
  Real.sin (2 * x - 3 * π / 4) > 0 :=
sorry

end NUMINAMATH_GPT_find_varphi_intervals_of_increase_l809_80929


namespace NUMINAMATH_GPT_max_value_of_x_plus_2y_l809_80976

theorem max_value_of_x_plus_2y {x y : ℝ} (h : |x| + |y| ≤ 1) : (x + 2 * y) ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_x_plus_2y_l809_80976


namespace NUMINAMATH_GPT_avg_price_of_returned_tshirts_l809_80994

-- Define the conditions as Lean definitions
def avg_price_50_tshirts := 750
def num_tshirts := 50
def num_returned_tshirts := 7
def avg_price_remaining_43_tshirts := 720

-- The correct price of the 7 returned T-shirts
def correct_avg_price_returned := 6540 / 7

-- The proof statement
theorem avg_price_of_returned_tshirts :
  (num_tshirts * avg_price_50_tshirts - (num_tshirts - num_returned_tshirts) * avg_price_remaining_43_tshirts) / num_returned_tshirts = correct_avg_price_returned :=
by
  sorry

end NUMINAMATH_GPT_avg_price_of_returned_tshirts_l809_80994


namespace NUMINAMATH_GPT_xyz_value_l809_80959

-- We define the constants from the problem
variables {x y z : ℂ}

-- Here's the theorem statement in Lean 4.
theorem xyz_value :
  (x * y + 5 * y = -20) →
  (y * z + 5 * z = -20) →
  (z * x + 5 * x = -20) →
  x * y * z = 100 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_xyz_value_l809_80959


namespace NUMINAMATH_GPT_convert_mps_to_kmph_l809_80947

theorem convert_mps_to_kmph (v_mps : ℝ) (conversion_factor : ℝ) : v_mps = 22 → conversion_factor = 3.6 → v_mps * conversion_factor = 79.2 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_convert_mps_to_kmph_l809_80947


namespace NUMINAMATH_GPT_domain_of_function_l809_80906

section
variable (x : ℝ)

def condition_1 := x + 4 ≥ 0
def condition_2 := x + 2 ≠ 0
def domain := { x : ℝ | x ≥ -4 ∧ x ≠ -2 }

theorem domain_of_function : (condition_1 x ∧ condition_2 x) ↔ (x ∈ domain) :=
by
  sorry
end

end NUMINAMATH_GPT_domain_of_function_l809_80906


namespace NUMINAMATH_GPT_proof_problem_l809_80988

variables (a b c : Line) (alpha beta gamma : Plane)

-- Define perpendicular relationship between line and plane
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define parallel relationship between lines
def parallel_lines (l1 l2 : Line) : Prop := sorry

-- Define parallel relationship between planes
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Main theorem statement
theorem proof_problem 
  (h1 : perp_line_plane a alpha) 
  (h2 : perp_line_plane b beta) 
  (h3 : parallel_planes alpha beta) : 
  parallel_lines a b :=
sorry

end NUMINAMATH_GPT_proof_problem_l809_80988


namespace NUMINAMATH_GPT_solution_is_thirteen_over_nine_l809_80970

noncomputable def check_solution (x : ℝ) : Prop :=
  (3 * x^2 / (x - 2) - (3 * x + 9) / 4 + (6 - 9 * x) / (x - 2) + 2 = 0) ∧
  (x^3 ≠ 3 * x + 1)

theorem solution_is_thirteen_over_nine :
  check_solution (13 / 9) :=
by
  sorry

end NUMINAMATH_GPT_solution_is_thirteen_over_nine_l809_80970


namespace NUMINAMATH_GPT_kristin_runs_around_l809_80931

-- Definitions of the conditions.
def kristin_runs_faster (v_k v_s : ℝ) : Prop := v_k = 3 * v_s
def sarith_runs_times (S : ℕ) : Prop := S = 8
def field_length (c_field a_field : ℝ) : Prop := c_field = a_field / 2

-- The question is to prove Kristin runs around the field 12 times.
def kristin_runs_times (K : ℕ) : Prop := K = 12

-- The main theorem statement combining conditions to prove the question.
theorem kristin_runs_around :
  ∀ (v_k v_s c_field a_field : ℝ) (S K : ℕ),
    kristin_runs_faster v_k v_s →
    sarith_runs_times S →
    field_length c_field a_field →
    K = (S : ℝ) * (3 / 2) →
    kristin_runs_times K :=
by sorry

end NUMINAMATH_GPT_kristin_runs_around_l809_80931


namespace NUMINAMATH_GPT_find_diameter_l809_80919

noncomputable def cost_per_meter : ℝ := 2
noncomputable def total_cost : ℝ := 188.49555921538757
noncomputable def circumference (c : ℝ) (p : ℝ) : ℝ := c / p
noncomputable def diameter (c : ℝ) : ℝ := c / Real.pi

theorem find_diameter :
  diameter (circumference total_cost cost_per_meter) = 30 := by
  sorry

end NUMINAMATH_GPT_find_diameter_l809_80919


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l809_80905

noncomputable def P := {x : ℝ | 0 < x ∧ x < 3}
noncomputable def Q := {x : ℝ | -3 < x ∧ x < 3}

theorem sufficient_but_not_necessary_condition :
  (∀ x, x ∈ P → x ∈ Q) ∧ ¬(∀ x, x ∈ Q → x ∈ P) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l809_80905


namespace NUMINAMATH_GPT_tablecloth_overhang_l809_80950

theorem tablecloth_overhang (d r l overhang1 overhang2 : ℝ) (h1 : d = 0.6) (h2 : r = d / 2) (h3 : l = 1) 
  (h4 : overhang1 = 0.5) (h5 : overhang2 = 0.3) :
  ∃ overhang3 overhang4 : ℝ, overhang3 = 0.33 ∧ overhang4 = 0.52 := 
sorry

end NUMINAMATH_GPT_tablecloth_overhang_l809_80950


namespace NUMINAMATH_GPT_white_area_correct_l809_80915

def total_sign_area : ℕ := 8 * 20
def black_area_C : ℕ := 8 * 1 + 2 * (1 * 3)
def black_area_A : ℕ := 2 * (8 * 1) + 2 * (1 * 2)
def black_area_F : ℕ := 8 * 1 + 2 * (1 * 4)
def black_area_E : ℕ := 3 * (1 * 4)

def total_black_area : ℕ := black_area_C + black_area_A + black_area_F + black_area_E
def white_area : ℕ := total_sign_area - total_black_area

theorem white_area_correct : white_area = 98 :=
  by 
    sorry -- State the theorem without providing the proof.

end NUMINAMATH_GPT_white_area_correct_l809_80915


namespace NUMINAMATH_GPT_statement1_statement2_statement3_l809_80981

variable (a b c m : ℝ)

-- Given condition
def quadratic_eq (a b c : ℝ) : Prop := a ≠ 0

-- Statement 1
theorem statement1 (h0 : quadratic_eq a b c) (h1 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = 2) : 2 * a - c = 0 :=
sorry

-- Statement 2
theorem statement2 (h0 : quadratic_eq a b c) (h2 : b = 2 * a + c) : (b^2 - 4 * a * c) > 0 :=
sorry

-- Statement 3
theorem statement3 (h0 : quadratic_eq a b c) (h3 : a * m^2 + b * m + c = 0) : b^2 - 4 * a * c = (2 * a * m + b)^2 :=
sorry

end NUMINAMATH_GPT_statement1_statement2_statement3_l809_80981


namespace NUMINAMATH_GPT_elderly_teachers_in_sample_l809_80980

-- Definitions based on the conditions
def numYoungTeachersSampled : ℕ := 320
def ratioYoungToElderly : ℚ := 16 / 9

-- The theorem that needs to be proved
theorem elderly_teachers_in_sample :
  ∃ numElderlyTeachersSampled : ℕ, 
    numYoungTeachersSampled * (9 / 16) = numElderlyTeachersSampled := 
by
  use 180
  sorry

end NUMINAMATH_GPT_elderly_teachers_in_sample_l809_80980


namespace NUMINAMATH_GPT_closest_integer_to_sqrt_11_l809_80996

theorem closest_integer_to_sqrt_11 : 
  ∀ (x : ℝ), (3 : ℝ) ≤ x → x ≤ 3.5 → x = 3 :=
by
  intro x hx h3_5
  sorry

end NUMINAMATH_GPT_closest_integer_to_sqrt_11_l809_80996


namespace NUMINAMATH_GPT_value_of_expression_l809_80941

theorem value_of_expression (m n : ℤ) (hm : |m| = 3) (hn : |n| = 2) (hmn : m < n) :
  m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l809_80941


namespace NUMINAMATH_GPT_archer_expected_hits_l809_80916

noncomputable def binomial_expected_value (n : ℕ) (p : ℝ) : ℝ :=
  n * p

theorem archer_expected_hits :
  binomial_expected_value 10 0.9 = 9 :=
by
  sorry

end NUMINAMATH_GPT_archer_expected_hits_l809_80916


namespace NUMINAMATH_GPT_jerry_age_proof_l809_80912

variable (J : ℝ)

/-- Mickey's age is 4 years less than 400% of Jerry's age. Mickey is 18 years old. Prove that Jerry is 5.5 years old. -/
theorem jerry_age_proof (h : 18 = 4 * J - 4) : J = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_jerry_age_proof_l809_80912


namespace NUMINAMATH_GPT_jerry_pick_up_trays_l809_80972

theorem jerry_pick_up_trays : 
  ∀ (trays_per_trip trips trays_from_second total),
  trays_per_trip = 8 →
  trips = 2 →
  trays_from_second = 7 →
  total = (trays_per_trip * trips) →
  (total - trays_from_second) = 9 :=
by
  intros trays_per_trip trips trays_from_second total
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_jerry_pick_up_trays_l809_80972
