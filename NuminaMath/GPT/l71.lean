import Mathlib

namespace typing_and_editing_time_l71_71045

-- Definitions for typing and editing times for consultants together and for Mary and Jim individually
def combined_typing_time := 12.5
def combined_editing_time := 7.5
def mary_typing_time := 30.0
def jim_editing_time := 12.0

-- The total time when Jim types and Mary edits
def total_time := 42.0

-- Proof statement
theorem typing_and_editing_time :
  (combined_typing_time = 12.5) ∧ 
  (combined_editing_time = 7.5) ∧ 
  (mary_typing_time = 30.0) ∧ 
  (jim_editing_time = 12.0) →
  total_time = 42.0 := 
by
  intro h
  -- Proof to be filled later
  sorry

end typing_and_editing_time_l71_71045


namespace milan_rate_per_minute_l71_71013

-- Definitions based on the conditions
def monthly_fee : ℝ := 2.0
def total_bill : ℝ := 23.36
def total_minutes : ℕ := 178
def expected_rate_per_minute : ℝ := 0.12

-- Theorem statement based on the question
theorem milan_rate_per_minute :
  (total_bill - monthly_fee) / total_minutes = expected_rate_per_minute := 
by 
  sorry

end milan_rate_per_minute_l71_71013


namespace no_k_for_linear_function_not_in_second_quadrant_l71_71099

theorem no_k_for_linear_function_not_in_second_quadrant :
  ¬∃ k : ℝ, ∀ x < 0, (k-1)*x + k ≤ 0 :=
by
  sorry

end no_k_for_linear_function_not_in_second_quadrant_l71_71099


namespace small_triangles_count_l71_71910

theorem small_triangles_count
  (sL sS : ℝ)  -- side lengths of large (sL) and small (sS) triangles
  (hL : sL = 15)  -- condition for the large triangle's side length
  (hS : sS = 3)   -- condition for the small triangle's side length
  : sL^2 / sS^2 = 25 := 
by {
  -- Definitions to skip the proof body
  -- Further mathematical steps would usually go here
  -- but 'sorry' is used to indicate the skipped proof.
  sorry
}

end small_triangles_count_l71_71910


namespace leopards_count_l71_71066

theorem leopards_count (L : ℕ) (h1 : 100 + 80 + L + 10 * L + 50 + 2 * (80 + L) = 670) : L = 20 :=
by
  sorry

end leopards_count_l71_71066


namespace smallest_n_l71_71416

-- Define the conditions as properties of integers
def connected (a b : ℕ): Prop := sorry -- Assume we have a definition for connectivity

def condition1 (a b n : ℕ) : Prop :=
  ¬connected a b → Nat.gcd (a^2 + b^2) n = 1

def condition2 (a b n : ℕ) : Prop :=
  connected a b → Nat.gcd (a^2 + b^2) n > 1

theorem smallest_n : ∃ n, n = 65 ∧ ∀ (a b : ℕ), condition1 a b n ∧ condition2 a b n := by
  sorry

end smallest_n_l71_71416


namespace solve_for_k_l71_71519

theorem solve_for_k : ∃ k : ℤ, 2^5 - 10 = 5^2 + k ∧ k = -3 := by
  sorry

end solve_for_k_l71_71519


namespace nth_equation_l71_71673

theorem nth_equation (n : ℕ) : 
  1 - (1 / ((n + 1)^2)) = (n / (n + 1)) * ((n + 2) / (n + 1)) :=
by sorry

end nth_equation_l71_71673


namespace max_a_for_necessary_not_sufficient_condition_l71_71333

theorem max_a_for_necessary_not_sufficient_condition {x a : ℝ} (h : ∀ x, x^2 > 1 → x < a) : a = -1 :=
by sorry

end max_a_for_necessary_not_sufficient_condition_l71_71333


namespace persistence_of_2_persistence_iff_2_l71_71251

def is_persistent (T : ℝ) : Prop :=
  ∀ (a b c d : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
                    a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ d ≠ 1) →
    (a + b + c + d = T) →
    (1 / a + 1 / b + 1 / c + 1 / d = T) →
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) + 1 / (1 - d) = T)

theorem persistence_of_2 : is_persistent 2 :=
by
  -- The proof is omitted as per instructions
  sorry

theorem persistence_iff_2 (T : ℝ) : is_persistent T ↔ T = 2 :=
by
  -- The proof is omitted as per instructions
  sorry

end persistence_of_2_persistence_iff_2_l71_71251


namespace inverse_proportion_quadrants_l71_71145

theorem inverse_proportion_quadrants (k : ℝ) : (∀ x, x ≠ 0 → ((x < 0 → (2 - k) / x > 0) ∧ (x > 0 → (2 - k) / x < 0))) → k > 2 :=
by sorry

end inverse_proportion_quadrants_l71_71145


namespace points_for_level_completion_l71_71640

-- Condition definitions
def enemies_defeated : ℕ := 6
def points_per_enemy : ℕ := 9
def total_points : ℕ := 62

-- Derived definitions (based on the problem steps):
def points_from_enemies : ℕ := enemies_defeated * points_per_enemy
def points_for_completing_level : ℕ := total_points - points_from_enemies

-- Theorem statement
theorem points_for_level_completion : points_for_completing_level = 8 := by
  sorry

end points_for_level_completion_l71_71640


namespace least_subtraction_divisible_by13_l71_71781

theorem least_subtraction_divisible_by13 (n : ℕ) (h : n = 427398) : ∃ k : ℕ, k = 2 ∧ (n - k) % 13 = 0 := by
  sorry

end least_subtraction_divisible_by13_l71_71781


namespace inequality_hold_l71_71577

theorem inequality_hold (n : ℕ) (h1 : n > 1) : 1 + n * 2^((n - 1 : ℕ) / 2) < 2^n :=
by
  sorry

end inequality_hold_l71_71577


namespace complex_sum_l71_71647

noncomputable def omega : ℂ := sorry
axiom h1 : omega^11 = 1
axiom h2 : omega ≠ 1

theorem complex_sum 
: omega^10 + omega^14 + omega^18 + omega^22 + omega^26 + omega^30 + omega^34 + omega^38 + omega^42 + omega^46 + omega^50 + omega^54 + omega^58 
= -omega^10 :=
sorry

end complex_sum_l71_71647


namespace chess_match_probability_l71_71756

theorem chess_match_probability (p : ℝ) (h0 : 0 < p) (h1 : p < 1) :
  (3 * p^3 * (1 - p) ≤ 6 * p^3 * (1 - p)^2) → (p ≤ 1/2) :=
by
  sorry

end chess_match_probability_l71_71756


namespace ratio_of_newspapers_l71_71942

theorem ratio_of_newspapers (C L : ℕ) (h1 : C = 42) (h2 : L = C + 23) : C / (C + 23) = 42 / 65 := by
  sorry

end ratio_of_newspapers_l71_71942


namespace evaluate_expression_l71_71810

theorem evaluate_expression (m n : ℤ) (hm : m = 2) (hn : n = -3) : (m + n) ^ 2 - 2 * m * (m + n) = 5 := by
  -- Proof skipped
  sorry

end evaluate_expression_l71_71810


namespace interval_intersection_l71_71298

open Set

-- Define the conditions
def condition_3x (x : ℝ) : Prop := 2 < 3 * x ∧ 3 * x < 3
def condition_4x (x : ℝ) : Prop := 2 < 4 * x ∧ 4 * x < 3

-- Define the problem statement
theorem interval_intersection :
  {x : ℝ | condition_3x x} ∩ {x | condition_4x x} = Ioo (2 / 3) (3 / 4) :=
by sorry

end interval_intersection_l71_71298


namespace arith_seq_general_formula_geom_seq_sum_l71_71200

-- Problem 1
theorem arith_seq_general_formula (a : ℕ → ℕ) (d : ℕ) (h_d : d = 3) (h_a1 : a 1 = 4) :
  a n = 3 * n + 1 :=
sorry

-- Problem 2
theorem geom_seq_sum (b : ℕ → ℚ) (S : ℕ → ℚ) (h_b1 : b 1 = 1 / 3) (r : ℚ) (h_r : r = 1 / 3) :
  S n = (1 / 2) * (1 - (1 / 3 ^ n)) :=
sorry

end arith_seq_general_formula_geom_seq_sum_l71_71200


namespace James_uses_150_sheets_of_paper_l71_71766

-- Define the conditions
def number_of_books := 2
def pages_per_book := 600
def pages_per_side := 4
def sides_per_sheet := 2

-- Statement to prove
theorem James_uses_150_sheets_of_paper :
  number_of_books * pages_per_book / (pages_per_side * sides_per_sheet) = 150 :=
by sorry

end James_uses_150_sheets_of_paper_l71_71766


namespace q_investment_l71_71373

theorem q_investment (p_investment : ℝ) (profit_ratio_p : ℝ) (profit_ratio_q : ℝ) (q_investment : ℝ) 
  (h1 : p_investment = 40000) 
  (h2 : profit_ratio_p / profit_ratio_q = 2 / 3) 
  : q_investment = 60000 := 
sorry

end q_investment_l71_71373


namespace shells_collected_by_savannah_l71_71832

def num_shells_jillian : ℕ := 29
def num_shells_clayton : ℕ := 8
def total_shells_distributed : ℕ := 54

theorem shells_collected_by_savannah (S : ℕ) :
  num_shells_jillian + S + num_shells_clayton = total_shells_distributed → S = 17 :=
by
  sorry

end shells_collected_by_savannah_l71_71832


namespace ratio_of_a_to_b_l71_71532

variable (a b c d : ℝ)

theorem ratio_of_a_to_b (h1 : c = 0.20 * a) (h2 : c = 0.10 * b) : a = (1 / 2) * b :=
by
  sorry

end ratio_of_a_to_b_l71_71532


namespace sam_has_12_nickels_l71_71599

theorem sam_has_12_nickels (n d : ℕ) (h1 : n + d = 30) (h2 : 5 * n + 10 * d = 240) : n = 12 :=
sorry

end sam_has_12_nickels_l71_71599


namespace People_Distribution_l71_71604

theorem People_Distribution 
  (total_people : ℕ) 
  (total_buses : ℕ) 
  (equal_distribution : ℕ) 
  (h1 : total_people = 219) 
  (h2 : total_buses = 3) 
  (h3 : equal_distribution = total_people / total_buses) : 
  equal_distribution = 73 :=
by 
  intros 
  sorry

end People_Distribution_l71_71604


namespace page_number_added_twice_l71_71148

theorem page_number_added_twice (n p : ℕ) (Hn : 1 ≤ n) (Hsum : (n * (n + 1)) / 2 + p = 2630) : 
  p = 2 :=
sorry

end page_number_added_twice_l71_71148


namespace union_complements_eq_l71_71332

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_complements_eq :
  U = {0, 1, 3, 5, 6, 8} →
  A = {1, 5, 8} →
  B = {2} →
  (U \ A) ∪ B = {0, 2, 3, 6} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  -- Prove that (U \ A) ∪ B = {0, 2, 3, 6}
  sorry

end union_complements_eq_l71_71332


namespace harriet_smallest_stickers_l71_71482

theorem harriet_smallest_stickers 
  (S : ℕ) (a b c : ℕ)
  (h1 : S = 5 * a + 3)
  (h2 : S = 11 * b + 3)
  (h3 : S = 13 * c + 3)
  (h4 : S > 3) :
  S = 718 :=
by
  sorry

end harriet_smallest_stickers_l71_71482


namespace find_speed_of_goods_train_l71_71073

noncomputable def speed_of_goods_train (v_man : ℝ) (t_pass : ℝ) (d_goods : ℝ) : ℝ := 
  let v_man_mps := v_man * (1000 / 3600)
  let v_relative := d_goods / t_pass
  let v_goods_mps := v_relative - v_man_mps
  v_goods_mps * (3600 / 1000)

theorem find_speed_of_goods_train :
  speed_of_goods_train 45 8 340 = 108 :=
by sorry

end find_speed_of_goods_train_l71_71073


namespace gcf_7fact_8fact_l71_71536

theorem gcf_7fact_8fact : 
  let f_7 := Nat.factorial 7
  let f_8 := Nat.factorial 8
  Nat.gcd f_7 f_8 = f_7 := 
by
  sorry

end gcf_7fact_8fact_l71_71536


namespace prime_gt_five_condition_l71_71730

theorem prime_gt_five_condition (p : ℕ) [Fact (Nat.Prime p)] (h : p > 5) :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 1 < p - a^2 ∧ p - a^2 < p - b^2 ∧ (p - a^2) ∣ (p - b)^2 := 
sorry

end prime_gt_five_condition_l71_71730


namespace min_odd_is_1_l71_71161

def min_odd_integers (a b c d e f : ℤ) : ℤ :=
  if (a + b) % 2 = 0 ∧ 
     (a + b + c + d) % 2 = 1 ∧ 
     (a + b + c + d + e + f) % 2 = 0 then
    1
  else
    sorry -- This should be replaced by a calculation of the true minimum based on conditions.

def satisfies_conditions (a b c d e f : ℤ) :=
  a + b = 30 ∧ 
  a + b + c + d = 47 ∧ 
  a + b + c + d + e + f = 65

theorem min_odd_is_1 (a b c d e f : ℤ) (h : satisfies_conditions a b c d e f) : 
  min_odd_integers a b c d e f = 1 := 
sorry

end min_odd_is_1_l71_71161


namespace last_digit_of_a_power_b_l71_71410

-- Define the constants from the problem
def a : ℕ := 954950230952380948328708
def b : ℕ := 470128749397540235934750230

-- Define a helper function to calculate the last digit of a natural number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the main statement to be proven
theorem last_digit_of_a_power_b : last_digit ((last_digit a) ^ (b % 4)) = 4 :=
by
  -- Here go the proof steps if we were to provide them
  sorry

end last_digit_of_a_power_b_l71_71410


namespace bc_lt_3ad_l71_71593

theorem bc_lt_3ad {a b c d x1 x2 x3 : ℝ}
    (h1 : a ≠ 0)
    (h2 : x1 > 0 ∧ x2 > 0 ∧ x3 > 0)
    (h3 : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)
    (h4 : x1 + x2 + x3 = -b / a)
    (h5 : x1 * x2 + x2 * x3 + x1 * x3 = c / a)
    (h6 : x1 * x2 * x3 = -d / a) : 
    b * c < 3 * a * d := 
sorry

end bc_lt_3ad_l71_71593


namespace bob_distance_when_they_meet_l71_71222

-- Define the conditions
def distance_XY : ℝ := 10
def yolanda_rate : ℝ := 3
def bob_rate : ℝ := 4
def yolanda_start_time : ℝ := 0
def bob_start_time : ℝ := 1

-- The statement we want to prove
theorem bob_distance_when_they_meet : 
  ∃ t : ℝ, (yolanda_rate * (t + 1) + bob_rate * t = distance_XY) ∧ (bob_rate * t = 4) :=
sorry

end bob_distance_when_they_meet_l71_71222


namespace amount_A_l71_71913

theorem amount_A (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 527) : A = 62 := by
  sorry

end amount_A_l71_71913


namespace ratio_second_third_l71_71018

theorem ratio_second_third (S T : ℕ) (h_sum : 200 + S + T = 500) (h_third : T = 100) : S / T = 2 := by
  sorry

end ratio_second_third_l71_71018


namespace calculate_3_pow_5_mul_6_pow_5_l71_71429

theorem calculate_3_pow_5_mul_6_pow_5 :
  3^5 * 6^5 = 34012224 := 
by 
  sorry

end calculate_3_pow_5_mul_6_pow_5_l71_71429


namespace coefficient_of_y_in_first_equation_is_minus_1_l71_71257

variable (x y z : ℝ)

def equation1 : Prop := 6 * x - y + 3 * z = 22 / 5
def equation2 : Prop := 4 * x + 8 * y - 11 * z = 7
def equation3 : Prop := 5 * x - 6 * y + 2 * z = 12
def sum_xyz : Prop := x + y + z = 10

theorem coefficient_of_y_in_first_equation_is_minus_1 :
  equation1 x y z → equation2 x y z → equation3 x y z → sum_xyz x y z → (-1 : ℝ) = -1 :=
by
  sorry

end coefficient_of_y_in_first_equation_is_minus_1_l71_71257


namespace p_sufficient_not_necessary_for_q_l71_71260

variable (x : ℝ)

def p : Prop := x > 0
def q : Prop := x > -1

theorem p_sufficient_not_necessary_for_q : (p x → q x) ∧ ¬ (q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l71_71260


namespace part_I_part_II_l71_71667

noncomputable def f (x a : ℝ) : ℝ := |x - a| - 2 * |x - 1|

-- Part I
theorem part_I (x : ℝ) : (f x 3) ≥ 1 ↔ (0 ≤ x ∧ x ≤ 4 / 3) :=
by sorry

-- Part II
theorem part_II (a : ℝ) : (∀ x, 1 ≤ x ∧ x ≤ 2 → f x a - |2 * x - 5| ≤ 0) ↔ (-1 ≤ a ∧ a ≤ 4) :=
by sorry

end part_I_part_II_l71_71667


namespace determine_ab_l71_71895

theorem determine_ab :
  ∃ a b : ℝ, 
  (3 + 8 * a = 2 - 3 * b) ∧ 
  (-1 - 6 * a = 4 * b) → 
  a = -1 / 14 ∧ b = -1 / 14 := 
by 
sorry

end determine_ab_l71_71895


namespace plain_chips_count_l71_71478

theorem plain_chips_count (total_chips : ℕ) (BBQ_chips : ℕ)
  (hyp1 : total_chips = 9) (hyp2 : BBQ_chips = 5)
  (hyp3 : (5 * 4 / (2 * 1) : ℚ) / ((9 * 8 * 7) / (3 * 2 * 1)) = 0.11904761904761904) :
  total_chips - BBQ_chips = 4 := by
sorry

end plain_chips_count_l71_71478


namespace correct_average_l71_71998

theorem correct_average :
  let avg_incorrect := 15
  let num_numbers := 20
  let read_incorrect1 := 42
  let read_correct1 := 52
  let read_incorrect2 := 68
  let read_correct2 := 78
  let read_incorrect3 := 85
  let read_correct3 := 95
  let incorrect_sum := avg_incorrect * num_numbers
  let diff1 := read_correct1 - read_incorrect1
  let diff2 := read_correct2 - read_incorrect2
  let diff3 := read_correct3 - read_incorrect3
  let total_diff := diff1 + diff2 + diff3
  let correct_sum := incorrect_sum + total_diff
  let correct_avg := correct_sum / num_numbers
  correct_avg = 16.5 :=
by
  sorry

end correct_average_l71_71998


namespace max_piece_length_l71_71921

theorem max_piece_length (L1 L2 L3 L4 : ℕ) (hL1 : L1 = 48) (hL2 : L2 = 72) (hL3 : L3 = 120) (hL4 : L4 = 144) 
  (h_min_pieces : ∀ L k, L = 48 ∨ L = 72 ∨ L = 120 ∨ L = 144 → k > 0 → L / k ≥ 5) : 
  ∃ k, k = 8 ∧ ∀ L, (L = L1 ∨ L = L2 ∨ L = L3 ∨ L = L4) → L % k = 0 :=
by
  sorry

end max_piece_length_l71_71921


namespace red_cards_pick_ordered_count_l71_71917

theorem red_cards_pick_ordered_count :
  let deck_size := 36
  let suits := 3
  let suit_size := 12
  let red_suits := 2
  let red_cards := red_suits * suit_size
  (red_cards * (red_cards - 1) = 552) :=
by
  let deck_size := 36
  let suits := 3
  let suit_size := 12
  let red_suits := 2
  let red_cards := red_suits * suit_size
  show (red_cards * (red_cards - 1) = 552)
  sorry

end red_cards_pick_ordered_count_l71_71917


namespace pump_fills_tank_without_leak_l71_71546

theorem pump_fills_tank_without_leak (T : ℝ) (h1 : 1 / 12 = 1 / T - 1 / 12) : T = 6 :=
sorry

end pump_fills_tank_without_leak_l71_71546


namespace vertex_of_quadratic_l71_71039

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem vertex_of_quadratic :
  ∃ (h k : ℝ), (∀ x : ℝ, f x = (x - h)^2 + k) ∧ (h = 1) ∧ (k = -2) :=
by
  sorry

end vertex_of_quadratic_l71_71039


namespace triangle_angle_contradiction_l71_71242

-- Define the condition: all internal angles of the triangle are less than 60 degrees.
def condition (α β γ : ℝ) (h: α + β + γ = 180): Prop :=
  α < 60 ∧ β < 60 ∧ γ < 60

-- The proof statement
theorem triangle_angle_contradiction (α β γ : ℝ) (h_sum : α + β + γ = 180) (h: condition α β γ h_sum) : false :=
sorry

end triangle_angle_contradiction_l71_71242


namespace find_min_max_value_l71_71337

open Real

theorem find_min_max_value (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) (h_det : b^2 - 4 * a * c < 0) :
  ∃ (min_val max_val: ℝ),
    min_val = (2 * d * sqrt (a * c)) / (b + 2 * sqrt (a * c)) ∧ 
    max_val = (2 * d * sqrt (a * c)) / (b - 2 * sqrt (a * c)) ∧
    (∀ x y : ℝ, a * x^2 + c * y^2 ≥ min_val ∧ a * x^2 + c * y^2 ≤ max_val) :=
by
  -- Proof goes here
  sorry

end find_min_max_value_l71_71337


namespace rectangular_board_area_l71_71381

variable (length width : ℕ)

theorem rectangular_board_area
  (h1 : length = 2 * width)
  (h2 : 2 * length + 2 * width = 84) :
  length * width = 392 := 
by
  sorry

end rectangular_board_area_l71_71381


namespace which_is_negative_l71_71484

theorem which_is_negative
    (A : ℤ := 2023)
    (B : ℤ := -2023)
    (C : ℚ := 1/2023)
    (D : ℤ := 0) :
    B < 0 :=
by
  sorry

end which_is_negative_l71_71484


namespace abc_sum_seven_l71_71610

theorem abc_sum_seven (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) : a + b + c = 7 :=
sorry

end abc_sum_seven_l71_71610


namespace numberOfTrucks_l71_71023

-- Conditions
def numberOfTanksPerTruck : ℕ := 3
def capacityPerTank : ℕ := 150
def totalWaterCapacity : ℕ := 1350

-- Question and proof goal
theorem numberOfTrucks : 
  (totalWaterCapacity / (numberOfTanksPerTruck * capacityPerTank) = 3) := 
by 
  sorry

end numberOfTrucks_l71_71023


namespace triangle_perimeter_l71_71031

theorem triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 6)
  (c1 c2 : ℝ) (h3 : (c1 - 2) * (c1 - 4) = 0) (h4 : (c2 - 2) * (c2 - 4) = 0) :
  c1 = 2 ∨ c1 = 4 → c2 = 2 ∨ c2 = 4 → 
  (c1 ≠ 2 ∧ c1 = 4 ∨ c2 ≠ 2 ∧ c2 = 4) → 
  (a + b + c1 = 13 ∨ a + b + c2 = 13) :=
by
  sorry

end triangle_perimeter_l71_71031


namespace div_condition_positive_integers_l71_71747

theorem div_condition_positive_integers 
  (a b d : ℕ) 
  (h1 : a + b ≡ 0 [MOD d]) 
  (h2 : a * b ≡ 0 [MOD d^2]) 
  (h3 : 0 < a) 
  (h4 : 0 < b) 
  (h5 : 0 < d) : 
  d ∣ a ∧ d ∣ b :=
sorry

end div_condition_positive_integers_l71_71747


namespace find_k_l71_71249

-- The function that computes the sum of the digits for the known form of the product (9 * 999...9) with k digits.
def sum_of_digits (k : ℕ) : ℕ :=
  8 + 9 * (k - 1) + 1

theorem find_k (k : ℕ) : sum_of_digits k = 2000 ↔ k = 222 := by
  sorry

end find_k_l71_71249


namespace halogens_have_solid_liquid_gas_l71_71267

def at_25C_and_1atm (element : String) : String :=
  match element with
  | "Li" | "Na" | "K" | "Rb" | "Cs" => "solid"
  | "N" => "gas"
  | "P" | "As" | "Sb" | "Bi" => "solid"
  | "O" => "gas"
  | "S" | "Se" | "Te" => "solid"
  | "F" | "Cl" => "gas"
  | "Br" => "liquid"
  | "I" | "At" => "solid"
  | _ => "unknown"

def family_has_solid_liquid_gas (family : List String) : Prop :=
  "solid" ∈ family.map at_25C_and_1atm ∧
  "liquid" ∈ family.map at_25C_and_1atm ∧
  "gas" ∈ family.map at_25C_and_1atm

theorem halogens_have_solid_liquid_gas :
  family_has_solid_liquid_gas ["F", "Cl", "Br", "I", "At"] :=
by
  sorry

end halogens_have_solid_liquid_gas_l71_71267


namespace chocolates_remaining_l71_71900

def chocolates := 24
def chocolates_first_day := 4
def chocolates_eaten_second_day := (2 * chocolates_first_day) - 3
def chocolates_eaten_third_day := chocolates_first_day - 2
def chocolates_eaten_fourth_day := chocolates_eaten_third_day - 1

theorem chocolates_remaining :
  chocolates - (chocolates_first_day + chocolates_eaten_second_day + chocolates_eaten_third_day + chocolates_eaten_fourth_day) = 12 := by
  sorry

end chocolates_remaining_l71_71900


namespace increasing_digits_count_l71_71027

theorem increasing_digits_count : 
  ∃ n, n = 120 ∧ ∀ x : ℕ, x ≤ 1000 → (∀ i j : ℕ, i < j → ((x / 10^i % 10) < (x / 10^j % 10)) → 
  x ≤ 1000 ∧ (x / 10^i % 10) ≠ (x / 10^j % 10)) :=
sorry

end increasing_digits_count_l71_71027


namespace wine_cost_increase_l71_71932

noncomputable def additional_cost (initial_price : ℝ) (num_bottles : ℕ) (month1_rate : ℝ) (month2_tariff : ℝ) (month2_discount : ℝ) (month3_tariff : ℝ) (month3_rate : ℝ) : ℝ := 
  let price_month1 := initial_price * (1 + month1_rate) 
  let cost_month1 := num_bottles * price_month1
  let price_month2 := (initial_price * (1 + month2_tariff)) * (1 - month2_discount)
  let cost_month2 := num_bottles * price_month2
  let price_month3 := (initial_price * (1 + month3_tariff)) * (1 - month3_rate)
  let cost_month3 := num_bottles * price_month3
  (cost_month1 + cost_month2 + cost_month3) - (3 * num_bottles * initial_price)

theorem wine_cost_increase : 
  additional_cost 20 5 0.05 0.25 0.15 0.35 0.03 = 42.20 :=
by sorry

end wine_cost_increase_l71_71932


namespace find_a_range_l71_71303

open Real

-- Define the points A, B, and C
def A : (ℝ × ℝ) := (4, 1)
def B : (ℝ × ℝ) := (-1, -6)
def C : (ℝ × ℝ) := (-3, 2)

-- Define the system of inequalities representing the region D
def region_D (x y : ℝ) : Prop :=
  7 * x - 5 * y - 23 ≤ 0 ∧
  x + 7 * y - 11 ≤ 0 ∧
  4 * x + y + 10 ≥ 0

-- Define the inequality condition for points B and C on opposite sides of the line 4x - 3y - a = 0
def opposite_sides (a : ℝ) : Prop :=
  (14 - a) * (-18 - a) < 0

-- Lean statement to prove the given problem
theorem find_a_range : 
  ∃ a : ℝ, region_D 0 0 ∧ opposite_sides a → -18 < a ∧ a < 14 :=
by 
  sorry

end find_a_range_l71_71303


namespace combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400_l71_71581

noncomputable def num_combinations_4_blocks_no_same_row_col :=
  (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)

theorem combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400 :
  num_combinations_4_blocks_no_same_row_col = 5400 := 
by
  sorry

end combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400_l71_71581


namespace quadratic_has_distinct_real_roots_l71_71787

def discriminant (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

theorem quadratic_has_distinct_real_roots :
  let a := 5
  let b := -2
  let c := -7
  discriminant a b c > 0 :=
by
  sorry

end quadratic_has_distinct_real_roots_l71_71787


namespace james_drive_time_to_canada_l71_71112

theorem james_drive_time_to_canada : 
  ∀ (distance speed stop_time : ℕ), 
    speed = 60 → 
    distance = 360 → 
    stop_time = 1 → 
    (distance / speed) + stop_time = 7 :=
by
  intros distance speed stop_time h1 h2 h3
  sorry

end james_drive_time_to_canada_l71_71112


namespace three_digit_multiples_of_seven_l71_71562

theorem three_digit_multiples_of_seven : 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  (last_multiple - first_multiple + 1) = 128 :=
by 
  let min_bound := 100
  let max_bound := 999
  let first_multiple := Nat.ceil (min_bound / 7)
  let last_multiple := Nat.floor (max_bound / 7)
  have calc_first_multiple : first_multiple = 15 := sorry
  have calc_last_multiple : last_multiple = 142 := sorry
  have count_multiples : (last_multiple - first_multiple + 1) = 128 := sorry
  exact count_multiples

end three_digit_multiples_of_seven_l71_71562


namespace shaded_area_percentage_l71_71557

theorem shaded_area_percentage (total_area shaded_area : ℕ) (h_total : total_area = 49) (h_shaded : shaded_area = 33) : 
  (shaded_area : ℚ) / total_area = 33 / 49 := 
by
  sorry

end shaded_area_percentage_l71_71557


namespace fixed_point_tangent_circle_l71_71830

theorem fixed_point_tangent_circle (x y a b t : ℝ) :
  (x ^ 2 + (y - 2) ^ 2 = 16) ∧ (a * 0 + b * 2 - 12 = 0) ∧ (y = -6) ∧ 
  (t * x - 8 * y = 0) → 
  (0, 0) = (0, 0) :=
by 
  sorry

end fixed_point_tangent_circle_l71_71830


namespace unit_vector_norm_diff_l71_71765

noncomputable def sqrt42_sqrt3_div_2 : ℝ := (Real.sqrt 42 * Real.sqrt 3) / 2
noncomputable def sqrt17_div_sqrt2 : ℝ := (Real.sqrt 17) / Real.sqrt 2

theorem unit_vector_norm_diff {x1 y1 z1 x2 y2 z2 : ℝ}
  (h1 : x1^2 + y1^2 + z1^2 = 1)
  (h2 : 3*x1 + y1 + 2*z1 = sqrt42_sqrt3_div_2)
  (h3 : 2*x1 + 2*y1 + 3*z1 = sqrt17_div_sqrt2)
  (h4 : x2^2 + y2^2 + z2^2 = 1)
  (h5 : 3*x2 + y2 + 2*z2 = sqrt42_sqrt3_div_2)
  (h6 : 2*x2 + 2*y2 + 3*z2 = sqrt17_div_sqrt2)
  (h_distinct : (x1, y1, z1) ≠ (x2, y2, z2)) :
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2) = Real.sqrt 2 :=
by
  sorry

end unit_vector_norm_diff_l71_71765


namespace compute_series_sum_l71_71226

noncomputable def term (n : ℕ) : ℝ := (5 * n - 2) / (3 ^ n)

theorem compute_series_sum : 
  ∑' n, term n = 11 / 4 := 
sorry

end compute_series_sum_l71_71226


namespace find_f_2008_l71_71522

variable (f : ℝ → ℝ) 
variable (g : ℝ → ℝ) -- g is the inverse of f

def satisfies_conditions (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (g x) = x) ∧ (∀ y : ℝ, g (f y) = y) ∧ 
  (f 9 = 18) ∧ (∀ x : ℝ, g (x + 1) = (f (x + 1)))

theorem find_f_2008 (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (h : satisfies_conditions f g) : f 2008 = -1981 :=
sorry

end find_f_2008_l71_71522


namespace complement_A_in_U_l71_71977

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}

theorem complement_A_in_U : (U \ A) = {x | -1 <= x ∧ x <= 3} :=
by
  sorry

end complement_A_in_U_l71_71977


namespace evaluate_sum_l71_71096

variable {a b c : ℝ}

theorem evaluate_sum 
  (h : a / (30 - a) + b / (75 - b) + c / (55 - c) = 8) :
  6 / (30 - a) + 15 / (75 - b) + 11 / (55 - c) = 187 / 30 :=
by
  sorry

end evaluate_sum_l71_71096


namespace race_outcomes_210_l71_71825

-- Define the participants
def participants : List String := ["Abe", "Bobby", "Charles", "Devin", "Edwin", "Fern", "Grace"]

-- The question is to prove the number of different 1st-2nd-3rd place outcomes is 210.
theorem race_outcomes_210 (h : participants.length = 7) : (7 * 6 * 5 = 210) :=
  by sorry

end race_outcomes_210_l71_71825


namespace distance_to_first_museum_l71_71361

theorem distance_to_first_museum (x : ℝ) 
  (dist_second_museum : ℝ) 
  (total_distance : ℝ) 
  (h1 : dist_second_museum = 15) 
  (h2 : total_distance = 40) 
  (h3 : 2 * x + 2 * dist_second_museum = total_distance) : x = 5 :=
by 
  sorry

end distance_to_first_museum_l71_71361


namespace danny_initial_wrappers_l71_71537

def initial_wrappers (total_wrappers: ℕ) (found_wrappers: ℕ): ℕ :=
  total_wrappers - found_wrappers

theorem danny_initial_wrappers : initial_wrappers 57 30 = 27 :=
by
  exact rfl

end danny_initial_wrappers_l71_71537


namespace third_quadrant_condition_l71_71671

-- Define the conditions for the third quadrant
def in_third_quadrant (p: ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

-- Translate the problem statement to a Lean theorem
theorem third_quadrant_condition (a b : ℝ) (h1 : a + b < 0) (h2 : a * b > 0) : in_third_quadrant (a, b) :=
sorry

end third_quadrant_condition_l71_71671


namespace probability_two_digit_between_21_and_30_l71_71188

theorem probability_two_digit_between_21_and_30 (dice1 dice2 : ℤ) (h1 : 1 ≤ dice1 ∧ dice1 ≤ 6) (h2 : 1 ≤ dice2 ∧ dice2 ≤ 6) :
∃ (p : ℚ), p = 11 / 36 := 
sorry

end probability_two_digit_between_21_and_30_l71_71188


namespace statement_A_statement_B_statement_C_l71_71070

variables {p : ℝ} (hp : p > 0) (x0 y0 x1 y1 x2 y2 : ℝ)
variables (h_parabola : ∀ x y, y^2 = 2*p*x) 
variables (h_point_P : ∀ k m, y0 ≠ 0 ∧ x0 = k*y0 + m)

-- Statement A
theorem statement_A (hy0 : y0 = 0) : y1 * y2 = -2 * p * x0 :=
sorry

-- Statement B
theorem statement_B (hx0 : x0 = 0) : 1 / y1 + 1 / y2 = 1 / y0 :=
sorry

-- Statement C
theorem statement_C : (y0 - y1) * (y0 - y2) = y0^2 - 2 * p * x0 :=
sorry

end statement_A_statement_B_statement_C_l71_71070


namespace quadratic_root_l71_71678

theorem quadratic_root (k : ℝ) (h : (1 : ℝ)^2 + k * 1 - 3 = 0) : k = 2 := 
sorry

end quadratic_root_l71_71678


namespace number_of_customers_l71_71981

-- Definitions based on conditions
def popularity (p : ℕ) (c w : ℕ) (k : ℝ) : Prop :=
  p = k * (w / c)

-- Given values
def given_values : Prop :=
  ∃ k : ℝ, popularity 15 500 1000 k

-- Problem statement
theorem number_of_customers:
  given_values →
  popularity 15 600 1200 7.5 :=
by
  intro h
  -- Proof omitted
  sorry

end number_of_customers_l71_71981


namespace eating_contest_l71_71820

variables (hotdog_weight burger_weight pie_weight : ℕ)
variable (noah_burgers jacob_pies mason_hotdogs : ℕ)
variable (total_weight_mason_hotdogs : ℕ)

theorem eating_contest :
  hotdog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  noah_burgers = 8 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  total_weight_mason_hotdogs = mason_hotdogs * hotdog_weight →
  total_weight_mason_hotdogs = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end eating_contest_l71_71820


namespace factorize_1_factorize_2_factorize_3_solve_system_l71_71174

-- Proving the factorization identities
theorem factorize_1 (y : ℝ) : 5 * y - 10 * y^2 = 5 * y * (1 - 2 * y) :=
by
  sorry

theorem factorize_2 (m : ℝ) : (3 * m - 1)^2 - 9 = (3 * m + 2) * (3 * m - 4) :=
by
  sorry

theorem factorize_3 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2)^2 :=
by
  sorry

-- Proving the solution to the system of equations
theorem solve_system (x y : ℝ) (h1 : x - y = 3) (h2 : x - 3 * y = -1) : x = 5 ∧ y = 2 :=
by
  sorry

end factorize_1_factorize_2_factorize_3_solve_system_l71_71174


namespace reflect_P_y_axis_l71_71312

def P : ℝ × ℝ := (2, 1)

def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

theorem reflect_P_y_axis :
  reflect_y_axis P = (-2, 1) :=
by
  sorry

end reflect_P_y_axis_l71_71312


namespace sum_x_coords_Q3_is_132_l71_71644

noncomputable def sum_x_coords_Q3 (x_coords: Fin 44 → ℝ) (sum_x1: ℝ) : ℝ :=
  sum_x1 -- given sum_x1 is the sum of x-coordinates of Q1 i.e., 132

theorem sum_x_coords_Q3_is_132 (x_coords: Fin 44 → ℝ) (sum_x1: ℝ) (h: sum_x1 = 132) :
  sum_x_coords_Q3 x_coords sum_x1 = 132 :=
by
  sorry

end sum_x_coords_Q3_is_132_l71_71644


namespace determine_n_from_average_l71_71271

-- Definitions derived from conditions
def total_cards (n : ℕ) : ℕ := n * (n + 1) / 2
def sum_of_values (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6
def average_value (n : ℕ) : ℚ := sum_of_values n / total_cards n

-- Main statement for proving equivalence
theorem determine_n_from_average :
  (∃ n : ℕ, average_value n = 2023) ↔ (n = 3034) :=
by
  sorry

end determine_n_from_average_l71_71271


namespace find_x1_plus_x2_l71_71109

def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

theorem find_x1_plus_x2 (x1 x2 : ℝ) (hneq : x1 ≠ x2) (h1 : f x1 = 101) (h2 : f x2 = 101) : x1 + x2 = 2 := 
by 
  -- proof or sorry can be used; let's assume we use sorry to skip proof
  sorry

end find_x1_plus_x2_l71_71109


namespace min_val_proof_l71_71108

noncomputable def minimum_value (x y z: ℝ) := 9 / x + 4 / y + 1 / z

theorem min_val_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + 2 * y + 3 * z = 12) :
  minimum_value x y z ≥ 49 / 12 :=
by {
  sorry
}

end min_val_proof_l71_71108


namespace sum_of_remainders_mod_13_l71_71538

theorem sum_of_remainders_mod_13 
  (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end sum_of_remainders_mod_13_l71_71538


namespace find_a1_l71_71755

-- Given an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Arithmetic sequence is monotonically increasing
def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

-- First condition: sum of first three terms
def sum_first_three_terms (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 12

-- Second condition: product of first three terms
def product_first_three_terms (a : ℕ → ℝ) : Prop :=
  a 0 * a 1 * a 2 = 48

-- Proving that a_1 = 2 given the conditions
theorem find_a1 (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : is_monotonically_increasing a)
  (h3 : sum_first_three_terms a) (h4 : product_first_three_terms a) : a 0 = 2 :=
by
  -- Proof will be filled in here
  sorry

end find_a1_l71_71755


namespace pow_eq_of_pow_sub_eq_l71_71507

theorem pow_eq_of_pow_sub_eq (a : ℝ) (m n : ℕ) (h1 : a^m = 6) (h2 : a^(m-n) = 2) : a^n = 3 := 
by
  sorry

end pow_eq_of_pow_sub_eq_l71_71507


namespace exists_real_root_iff_l71_71838

theorem exists_real_root_iff {m : ℝ} :
  (∃x : ℝ, 25 - abs (x + 1) - 4 * 5 - abs (x + 1) - m = 0) ↔ (-3 < m ∧ m < 0) :=
by
  sorry

end exists_real_root_iff_l71_71838


namespace fraction_multiplication_l71_71176

theorem fraction_multiplication : (1 / 3) * (1 / 4) * (1 / 5) * 60 = 1 := by
  sorry

end fraction_multiplication_l71_71176


namespace decimal_equivalent_of_fraction_squared_l71_71149

theorem decimal_equivalent_of_fraction_squared : (1 / 4 : ℝ) ^ 2 = 0.0625 :=
by sorry

end decimal_equivalent_of_fraction_squared_l71_71149


namespace find_quad_function_l71_71924

-- Define the quadratic function with the given conditions
def quad_function (a b c : ℝ) (f : ℝ → ℝ) :=
  ∀ x, f x = a * x^2 + b * x + c

-- Define the values y(-2) = -3, y(-1) = -4, y(0) = -3, y(2) = 5
def given_points (f : ℝ → ℝ) :=
  f (-2) = -3 ∧ f (-1) = -4 ∧ f 0 = -3 ∧ f 2 = 5

-- Prove that y = x^2 + 2x - 3 satisfies the given points
theorem find_quad_function : ∃ f : ℝ → ℝ, (quad_function 1 2 (-3) f) ∧ (given_points f) :=
by
  sorry

end find_quad_function_l71_71924


namespace largest_B_div_by_4_l71_71535

-- Given conditions
def is_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- The seven-digit integer is 4B6792X
def number (B X : ℕ) : ℕ := 4000000 + B * 100000 + 60000 + 7000 + 900 + 20 + X

-- Problem statement: Prove that the largest digit B so that the seven-digit integer 4B6792X is divisible by 4
theorem largest_B_div_by_4 
(B X : ℕ) 
(hX : is_digit X)
(div_4 : divisible_by_4 (number B X)) : 
B = 9 := sorry

end largest_B_div_by_4_l71_71535


namespace f_leq_binom_l71_71212

-- Define the function f with given conditions
def f (m n : ℕ) : ℕ := if m = 1 ∨ n = 1 then 1 else sorry

-- State the property to be proven
theorem f_leq_binom (m n : ℕ) (h2 : 2 ≤ m) (h2' : 2 ≤ n) :
  f m n ≤ Nat.choose (m + n) n := 
sorry

end f_leq_binom_l71_71212


namespace problem_abc_value_l71_71244

theorem problem_abc_value 
  (a b c : ℤ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > 0)
  (h4 : Int.gcd b c = 1)
  (h5 : (b + c) % a = 0)
  (h6 : (a + c) % b = 0) :
  a * b * c = 6 :=
sorry

end problem_abc_value_l71_71244


namespace cos_half_pi_minus_2alpha_l71_71473

open Real

theorem cos_half_pi_minus_2alpha (α : ℝ) (h : sin α - cos α = 1 / 3) : cos (π / 2 - 2 * α) = 8 / 9 :=
sorry

end cos_half_pi_minus_2alpha_l71_71473


namespace tan_20_plus_4sin_20_eq_sqrt3_l71_71239

theorem tan_20_plus_4sin_20_eq_sqrt3 :
  (Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180)) = Real.sqrt 3 := by
  sorry

end tan_20_plus_4sin_20_eq_sqrt3_l71_71239


namespace trig_identity_l71_71517

theorem trig_identity (α m : ℝ) (h : Real.tan α = m) :
  (Real.sin (π / 4 + α))^2 - (Real.sin (π / 6 - α))^2 - Real.cos (5 * π / 12) * Real.sin (5 * π / 12 - 2 * α) = 2 * m / (1 + m^2) :=
by
  sorry

end trig_identity_l71_71517


namespace max_profit_l71_71775

noncomputable def total_cost (Q : ℝ) : ℝ := 5 * Q^2

noncomputable def demand_non_slytherin (P : ℝ) : ℝ := 26 - 2 * P

noncomputable def demand_slytherin (P : ℝ) : ℝ := 10 - P

noncomputable def combined_demand (P : ℝ) : ℝ :=
  if P >= 13 then demand_non_slytherin P else demand_non_slytherin P + demand_slytherin P

noncomputable def inverse_demand (Q : ℝ) : ℝ :=
  if Q <= 6 then 13 - Q / 2 else 12 - Q / 3

noncomputable def revenue (Q : ℝ) : ℝ :=
  if Q <= 6 then Q * (13 - Q / 2) else Q * (12 - Q / 3)

noncomputable def marginal_revenue (Q : ℝ) : ℝ :=
  if Q <= 6 then 13 - Q else 12 - 2 * Q / 3

noncomputable def marginal_cost (Q : ℝ) : ℝ := 10 * Q

theorem max_profit :
  ∃ Q P TR TC π,
    P = inverse_demand Q ∧
    TR = P * Q ∧
    TC = total_cost Q ∧
    π = TR - TC ∧
    π = 7.69 :=
sorry

end max_profit_l71_71775


namespace fraction_simplification_l71_71506

/-- Given x and y, under the conditions x ≠ 3y and x ≠ -3y, 
we want to prove that (2 * x) / (x ^ 2 - 9 * y ^ 2) - 1 / (x - 3 * y) = 1 / (x + 3 * y). -/
theorem fraction_simplification (x y : ℝ) (h1 : x ≠ 3 * y) (h2 : x ≠ -3 * y) :
  (2 * x) / (x ^ 2 - 9 * y ^ 2) - 1 / (x - 3 * y) = 1 / (x + 3 * y) :=
by
  sorry

end fraction_simplification_l71_71506


namespace age_of_eldest_child_l71_71114

-- Define the conditions as hypotheses
def child_ages_sum_equals_50 (x : ℕ) : Prop :=
  x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50

-- Define the main theorem to prove the age of the eldest child
theorem age_of_eldest_child (x : ℕ) (h : child_ages_sum_equals_50 x) : x + 8 = 14 :=
sorry

end age_of_eldest_child_l71_71114


namespace curve_is_circle_l71_71626

theorem curve_is_circle : ∀ (θ : ℝ), ∃ r : ℝ, r = 3 * Real.cos θ → ∃ (x y : ℝ), x^2 + y^2 = (3/2)^2 :=
by
  intro θ
  use 3 * Real.cos θ
  sorry

end curve_is_circle_l71_71626


namespace toll_booth_ratio_l71_71702

theorem toll_booth_ratio (total_cars : ℕ) (monday_cars tuesday_cars friday_cars saturday_cars sunday_cars : ℕ)
  (x : ℕ) (h1 : total_cars = 450) (h2 : monday_cars = 50) (h3 : tuesday_cars = 50) (h4 : friday_cars = 50)
  (h5 : saturday_cars = 50) (h6 : sunday_cars = 50) (h7 : monday_cars + tuesday_cars + x + x + friday_cars + saturday_cars + sunday_cars = total_cars) :
  x = 100 ∧ x / monday_cars = 2 :=
by
  sorry

end toll_booth_ratio_l71_71702


namespace find_f_l71_71004

-- Define the function space and conditions
def func (f : ℕ+ → ℝ) :=
  (∀ m n : ℕ+, f (m * n) = f m + f n) ∧
  (∀ n : ℕ+, f (n + 1) ≥ f n)

-- Define the theorem statement
theorem find_f (f : ℕ+ → ℝ) (hf : func f) : ∀ n : ℕ+, f n = 0 :=
sorry

end find_f_l71_71004


namespace numberOfCubesWithNoMoreThanFourNeighbors_l71_71290

def unitCubesWithAtMostFourNeighbors (a b c : ℕ) (h1 : a > 4) (h2 : b > 4) (h3 : c > 4) 
(h4 : (a - 2) * (b - 2) * (c - 2) = 836) : ℕ := 
  4 * (a - 2 + b - 2 + c - 2) + 8

theorem numberOfCubesWithNoMoreThanFourNeighbors (a b c : ℕ) 
(h1 : a > 4) (h2 : b > 4) (h3 : c > 4)
(h4 : (a - 2) * (b - 2) * (c - 2) = 836) :
  unitCubesWithAtMostFourNeighbors a b c h1 h2 h3 h4 = 144 :=
sorry

end numberOfCubesWithNoMoreThanFourNeighbors_l71_71290


namespace slices_per_pizza_l71_71531

def num_pizzas : ℕ := 2
def total_slices : ℕ := 16

theorem slices_per_pizza : total_slices / num_pizzas = 8 := by
  sorry

end slices_per_pizza_l71_71531


namespace max_problems_to_miss_to_pass_l71_71870

theorem max_problems_to_miss_to_pass (total_problems : ℕ) (pass_percentage : ℝ) :
  total_problems = 50 → pass_percentage = 0.85 → 7 = ↑total_problems * (1 - pass_percentage) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end max_problems_to_miss_to_pass_l71_71870


namespace sneakers_cost_l71_71401

theorem sneakers_cost (rate_per_yard : ℝ) (num_yards_cut : ℕ) (total_earnings : ℝ) :
  rate_per_yard = 2.15 ∧ num_yards_cut = 6 ∧ total_earnings = rate_per_yard * num_yards_cut → 
  total_earnings = 12.90 :=
by
  sorry

end sneakers_cost_l71_71401


namespace engineers_percentage_calculation_l71_71754

noncomputable def percentageEngineers (num_marketers num_engineers num_managers total_salary: ℝ) : ℝ := 
  let num_employees := num_marketers + num_engineers + num_managers 
  if num_employees = 0 then 0 else num_engineers / num_employees * 100

theorem engineers_percentage_calculation : 
  let marketers_percentage := 0.7 
  let engineers_salary := 80000
  let average_salary := 80000
  let marketers_salary_total := 50000 * marketers_percentage 
  let managers_total_percent := 1 - marketers_percentage - x / 100
  let managers_salary := 370000 * managers_total_percent 
  marketers_salary_total + engineers_salary * x / 100 + managers_salary = average_salary -> 
  x = 22.76 
:= 
sorry

end engineers_percentage_calculation_l71_71754


namespace calculate_number_of_models_l71_71028

-- Define the constants and conditions
def time_per_set : ℕ := 2  -- time per set in minutes
def sets_bathing_suits : ℕ := 2  -- number of bathing suit sets each model wears
def sets_evening_wear : ℕ := 3  -- number of evening wear sets each model wears
def total_show_time : ℕ := 60  -- total show time in minutes

-- Calculate the total time each model takes
def model_time : ℕ := 
  (sets_bathing_suits + sets_evening_wear) * time_per_set

-- Proof problem statement
theorem calculate_number_of_models : 
  (total_show_time / model_time) = 6 := by
  sorry

end calculate_number_of_models_l71_71028


namespace totalUniqueStudents_l71_71689

-- Define the club memberships and overlap
variable (mathClub scienceClub artClub overlap : ℕ)

-- Conditions based on the problem
def mathClubSize : Prop := mathClub = 15
def scienceClubSize : Prop := scienceClub = 10
def artClubSize : Prop := artClub = 12
def overlapSize : Prop := overlap = 5

-- Main statement to prove
theorem totalUniqueStudents : 
  mathClubSize mathClub → 
  scienceClubSize scienceClub →
  artClubSize artClub →
  overlapSize overlap →
  mathClub + scienceClub + artClub - overlap = 32 := by
  intros
  sorry

end totalUniqueStudents_l71_71689


namespace values_of_a_l71_71760

theorem values_of_a (a : ℝ) : 
  ∃a1 a2 : ℝ, 
  (∀ x y : ℝ, (y = 3 * x + a) ∧ (y = x^3 + 3 * a^2) → (x = 0) → (y = 3 * a^2)) →
  ((a = 0) ∨ (a = 1/3)) ∧ 
  ((a1 = 0) ∨ (a1 = 1/3)) ∧
  ((a2 = 0) ∨ (a2 = 1/3)) ∧ 
  (a ≠ a1 ∨ a ≠ a2) ∧ 
  (∃ n : ℤ, n = 2) :=
by sorry

end values_of_a_l71_71760


namespace sam_distance_l71_71326

theorem sam_distance (miles_marguerite : ℕ) (hours_marguerite : ℕ) (hours_sam : ℕ) 
  (speed_increase : ℚ) (avg_speed_marguerite : ℚ) (speed_sam : ℚ) (distance_sam : ℚ) :
  miles_marguerite = 120 ∧ hours_marguerite = 3 ∧ hours_sam = 4 ∧ speed_increase = 1.20 ∧
  avg_speed_marguerite = miles_marguerite / hours_marguerite ∧ 
  speed_sam = avg_speed_marguerite * speed_increase ∧
  distance_sam = speed_sam * hours_sam →
  distance_sam = 192 :=
by
  intros h
  sorry

end sam_distance_l71_71326


namespace no_integer_solutions_for_inequality_l71_71795

open Int

theorem no_integer_solutions_for_inequality : ∀ x : ℤ, (x - 4) * (x - 5) < 0 → False :=
by
  sorry

end no_integer_solutions_for_inequality_l71_71795


namespace area_of_right_triangle_l71_71850

theorem area_of_right_triangle (A B C : ℝ) (hA : A = 64) (hB : B = 36) (hC : C = 100) : 
  (1 / 2) * (Real.sqrt A) * (Real.sqrt B) = 24 :=
by
  sorry

end area_of_right_triangle_l71_71850


namespace fabric_nguyen_needs_l71_71405

-- Definitions for conditions
def fabric_per_pant : ℝ := 8.5
def total_pants : ℝ := 7
def yards_to_feet (yards : ℝ) : ℝ := yards * 3
def fabric_nguyen_has_yards : ℝ := 3.5

-- The proof we need to establish
theorem fabric_nguyen_needs : (total_pants * fabric_per_pant) - (yards_to_feet fabric_nguyen_has_yards) = 49 :=
by
  sorry

end fabric_nguyen_needs_l71_71405


namespace probability_of_7_successes_l71_71935

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_of_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * p^k * (1 - p)^(n - k)

theorem probability_of_7_successes :
  probability_of_successes 7 7 (2/7) = 128 / 823543 :=
by
  sorry

end probability_of_7_successes_l71_71935


namespace faster_train_speed_l71_71685

theorem faster_train_speed (dist_between_stations : ℕ) (extra_distance : ℕ) (slower_speed : ℕ) 
  (dist_between_stations_eq : dist_between_stations = 444)
  (extra_distance_eq : extra_distance = 60) 
  (slower_speed_eq : slower_speed = 16) :
  ∃ (faster_speed : ℕ), faster_speed = 21 := by
  sorry

end faster_train_speed_l71_71685


namespace project_completion_time_l71_71650

theorem project_completion_time (m n : ℝ) (hm : m > 0) (hn : n > 0):
  (1 / (1 / m + 1 / n)) = (m * n) / (m + n) :=
by
  sorry

end project_completion_time_l71_71650


namespace cubic_yard_to_cubic_feet_l71_71852

theorem cubic_yard_to_cubic_feet (h : 1 = 3) : 1 = 27 := 
by
  sorry

end cubic_yard_to_cubic_feet_l71_71852


namespace math_problem_l71_71169

variable {a b : ℕ → ℕ}

-- Condition 1: a_n is an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + a m

-- Condition 2: 2a₂ - a₇² + 2a₁₂ = 0
def satisfies_equation (a : ℕ → ℕ) : Prop :=
  2 * a 2 - (a 7)^2 + 2 * a 12 = 0

-- Condition 3: b_n is a geometric sequence
def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, b (n + m) = b n * b m

-- Condition 4: b₇ = a₇
def b7_eq_a7 (a b : ℕ → ℕ) : Prop :=
  b 7 = a 7

-- To prove: b₅ * b₉ = 16
theorem math_problem (a b : ℕ → ℕ)
  (h₁ : is_arithmetic_sequence a)
  (h₂ : satisfies_equation a)
  (h₃ : is_geometric_sequence b)
  (h₄ : b7_eq_a7 a b) :
  b 5 * b 9 = 16 :=
sorry

end math_problem_l71_71169


namespace gcd_256_180_720_l71_71443

theorem gcd_256_180_720 : Int.gcd (Int.gcd 256 180) 720 = 36 := by
  sorry

end gcd_256_180_720_l71_71443


namespace a_eq_b_pow_n_l71_71022

variables (a b n : ℕ)
variable (h : ∀ (k : ℕ), k ≠ b → b - k ∣ a - k^n)

theorem a_eq_b_pow_n : a = b^n := 
by
  sorry

end a_eq_b_pow_n_l71_71022


namespace math_problem_l71_71600

   theorem math_problem :
     6 * (-1 / 2) + Real.sqrt 3 * Real.sqrt 8 + (-15 : ℝ)^0 = 2 * Real.sqrt 6 - 2 :=
   by
     sorry
   
end math_problem_l71_71600


namespace count_negative_values_correct_l71_71989

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l71_71989


namespace binomial_sixteen_twelve_eq_l71_71050

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem to prove
theorem binomial_sixteen_twelve_eq : binomial 16 12 = 43680 := by
  sorry

end binomial_sixteen_twelve_eq_l71_71050


namespace min_units_for_profitability_profitability_during_epidemic_l71_71619

-- Conditions
def assembly_line_cost : ℝ := 1.8
def selling_price_per_product : ℝ := 0.1
def max_annual_output : ℕ := 100

noncomputable def production_cost (x : ℕ) : ℝ := 5 + 135 / (x + 1)

-- Part 1: Prove Minimum x for profitability
theorem min_units_for_profitability (x : ℕ) :
  (10 - (production_cost x)) * x - assembly_line_cost > 0 ↔ x ≥ 63 := sorry

-- Part 2: Profitability and max profit output during epidemic
theorem profitability_during_epidemic (x : ℕ) :
  (60 < x ∧ x ≤ max_annual_output) → 
  ((10 - (production_cost x)) * 60 - (x - 60) - assembly_line_cost > 0) ↔ x = 89 := sorry

end min_units_for_profitability_profitability_during_epidemic_l71_71619


namespace rowing_problem_l71_71377

theorem rowing_problem (R S x y : ℝ) 
  (h1 : R = y + x) 
  (h2 : S = y - x) : 
  x = (R - S) / 2 ∧ y = (R + S) / 2 :=
by
  sorry

end rowing_problem_l71_71377


namespace find_value_of_E_l71_71216

variables (Q U I E T Z : ℤ)

theorem find_value_of_E (hZ : Z = 15) (hQUIZ : Q + U + I + Z = 60) (hQUIET : Q + U + I + E + T = 75) (hQUIT : Q + U + I + T = 50) : E = 25 :=
by
  have hQUIZ_val : Q + U + I = 45 := by linarith [hZ, hQUIZ]
  have hQUIET_val : E + T = 30 := by linarith [hQUIZ_val, hQUIET]
  have hQUIT_val : T = 5 := by linarith [hQUIZ_val, hQUIT]
  linarith [hQUIET_val, hQUIT_val]

end find_value_of_E_l71_71216


namespace cost_per_steak_knife_l71_71713

theorem cost_per_steak_knife :
  ∀ (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℝ),
  sets = 2 →
  knives_per_set = 4 →
  cost_per_set = 80 →
  (cost_per_set * sets) / (sets * knives_per_set) = 20 :=
by
  intros sets knives_per_set cost_per_set sets_eq knives_per_set_eq cost_per_set_eq
  rw [sets_eq, knives_per_set_eq, cost_per_set_eq]
  sorry

end cost_per_steak_knife_l71_71713


namespace x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1_l71_71652

theorem x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1 (x : ℝ) : (x > 1 → |x| > 1) ∧ (¬(x > 1 ↔ |x| > 1)) :=
by
  sorry

end x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1_l71_71652


namespace find_sum_2017_l71_71025

-- Define the arithmetic sequence and its properties
def is_arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a n) / 2

-- Given conditions
variables (a : ℕ → ℤ)
axiom h1 : is_arithmetic_sequence a
axiom h2 : sum_first_n_terms a 2011 = -2011
axiom h3 : a 1012 = 3

-- Theorem to be proven
theorem find_sum_2017 : sum_first_n_terms a 2017 = 2017 :=
by sorry

end find_sum_2017_l71_71025


namespace minimum_positive_difference_contains_amounts_of_numbers_on_strips_l71_71492

theorem minimum_positive_difference_contains_amounts_of_numbers_on_strips (a b c d e f : ℕ) 
  (h1 : a + f = 7) (h2 : b + e = 7) (h3 : c + d = 7) :
  ∃ (min_diff : ℕ), min_diff = 1 :=
by {
  -- The problem guarantees the minimum difference given the conditions.
  sorry
}

end minimum_positive_difference_contains_amounts_of_numbers_on_strips_l71_71492


namespace profit_percentage_l71_71985

theorem profit_percentage (initial_cost_per_pound : ℝ) (ruined_percent : ℝ) (selling_price_per_pound : ℝ) (desired_profit_percent : ℝ) : 
  initial_cost_per_pound = 0.80 ∧ ruined_percent = 0.10 ∧ selling_price_per_pound = 0.96 → desired_profit_percent = 8 := by
  sorry

end profit_percentage_l71_71985


namespace number_of_seeds_per_row_l71_71612

-- Define the conditions as variables
def rows : ℕ := 6
def total_potatoes : ℕ := 54
def seeds_per_row : ℕ := 9

-- State the theorem
theorem number_of_seeds_per_row :
  total_potatoes / rows = seeds_per_row :=
by
-- We ignore the proof here, it will be provided later
sorry

end number_of_seeds_per_row_l71_71612


namespace first_part_lent_years_l71_71964

theorem first_part_lent_years (P P1 P2 : ℝ) (rate1 rate2 : ℝ) (years2 : ℝ) (interest1 interest2 : ℝ) (t : ℝ) 
  (h1 : P = 2717)
  (h2 : P2 = 1672)
  (h3 : P1 = P - P2)
  (h4 : rate1 = 0.03)
  (h5 : rate2 = 0.05)
  (h6 : years2 = 3)
  (h7 : interest1 = P1 * rate1 * t)
  (h8 : interest2 = P2 * rate2 * years2)
  (h9 : interest1 = interest2) :
  t = 8 :=
sorry

end first_part_lent_years_l71_71964


namespace diophantine_equation_solvable_l71_71371

theorem diophantine_equation_solvable (a : ℕ) (ha : 0 < a) : 
  ∃ (x y : ℤ), x^2 - y^2 = a^3 :=
by
  let x := (a * (a + 1)) / 2
  let y := (a * (a - 1)) / 2
  have hx : x^2 = (a * (a + 1) / 2 : ℤ)^2 := sorry
  have hy : y^2 = (a * (a - 1) / 2 : ℤ)^2 := sorry
  use x
  use y
  sorry

end diophantine_equation_solvable_l71_71371


namespace meal_combinations_l71_71425

theorem meal_combinations (MenuA_items : ℕ) (MenuB_items : ℕ) : MenuA_items = 15 ∧ MenuB_items = 12 → MenuA_items * MenuB_items = 180 :=
by
  sorry

end meal_combinations_l71_71425


namespace abs_neg_frac_l71_71419

theorem abs_neg_frac : abs (-3 / 2) = 3 / 2 := 
by sorry

end abs_neg_frac_l71_71419


namespace proof_problem_l71_71375

theorem proof_problem
  (x y : ℤ)
  (hx : ∃ m : ℤ, x = 6 * m)
  (hy : ∃ n : ℤ, y = 12 * n) :
  (x + y) % 2 = 0 ∧ (x + y) % 6 = 0 ∧ ¬ (x + y) % 12 = 0 → ¬ (x + y) % 12 = 0 :=
  sorry

end proof_problem_l71_71375


namespace vector_addition_and_scalar_multiplication_l71_71113

-- Specify the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 1)

-- Define the theorem we want to prove
theorem vector_addition_and_scalar_multiplication :
  a + 2 • b = (-3, 4) :=
sorry

end vector_addition_and_scalar_multiplication_l71_71113


namespace evaluate_expression_l71_71878

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l71_71878


namespace average_score_girls_l71_71814

theorem average_score_girls (num_boys num_girls : ℕ) (avg_boys avg_class : ℕ) : 
  num_boys = 12 → 
  num_girls = 4 → 
  avg_boys = 84 → 
  avg_class = 86 → 
  ∃ avg_girls : ℕ, avg_girls = 92 :=
by
  intros h1 h2 h3 h4
  sorry

end average_score_girls_l71_71814


namespace geometric_seq_ad_eq_2_l71_71707

open Real

def geometric_sequence (a b c d : ℝ) : Prop :=
∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r 

def is_max_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
f x = y ∧ ∀ z : ℝ, z ≠ x → f x ≥ f z

theorem geometric_seq_ad_eq_2 (a b c d : ℝ) :
  geometric_sequence a b c d →
  is_max_point (λ x => 3 * x - x ^ 3) b c →
  a * d = 2 :=
by
  sorry

end geometric_seq_ad_eq_2_l71_71707


namespace value_of_M_l71_71285

noncomputable def a : ℝ := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2)
noncomputable def b : ℝ := Real.sqrt (5 - 2 * Real.sqrt 6)
noncomputable def M : ℝ := a - b

theorem value_of_M : M = 4 :=
by
  sorry

end value_of_M_l71_71285


namespace evaluate_expression_l71_71624

theorem evaluate_expression :
  (1 / (-8^2)^4) * (-8)^9 = -8 := by
  sorry

end evaluate_expression_l71_71624


namespace largest_odd_digit_multiple_of_5_lt_10000_l71_71896

def is_odd_digit (n : ℕ) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), is_odd_digit d

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem largest_odd_digit_multiple_of_5_lt_10000 :
  ∃ n, n < 10000 ∧ all_odd_digits n ∧ is_multiple_of_5 n ∧
        ∀ m, m < 10000 → all_odd_digits m → is_multiple_of_5 m → m ≤ n :=
  sorry

end largest_odd_digit_multiple_of_5_lt_10000_l71_71896


namespace number_of_cats_l71_71495

theorem number_of_cats (c d : ℕ) (h1 : c = 20 + d) (h2 : c + d = 60) : c = 40 :=
sorry

end number_of_cats_l71_71495


namespace division_quotient_l71_71418

theorem division_quotient (dividend divisor remainder quotient : ℕ) 
  (h₁ : dividend = 95) (h₂ : divisor = 15) (h₃ : remainder = 5)
  (h₄ : dividend = divisor * quotient + remainder) : quotient = 6 :=
by
  sorry

end division_quotient_l71_71418


namespace reserve_bird_percentage_l71_71767

theorem reserve_bird_percentage (total_birds hawks paddyfield_warbler_percentage kingfisher_percentage woodpecker_percentage owl_percentage : ℕ) 
  (h1 : total_birds = 5000)
  (h2 : hawks = 30 * total_birds / 100)
  (h3 : paddyfield_warbler_percentage = 40)
  (h4 : kingfisher_percentage = 25)
  (h5 : woodpecker_percentage = 15)
  (h6 : owl_percentage = 15) :
  let non_hawks := total_birds - hawks
  let paddyfield_warblers := paddyfield_warbler_percentage * non_hawks / 100
  let kingfishers := kingfisher_percentage * paddyfield_warblers / 100
  let woodpeckers := woodpecker_percentage * non_hawks / 100
  let owls := owl_percentage * non_hawks / 100
  let specified_non_hawks := paddyfield_warblers + kingfishers + woodpeckers + owls
  let unspecified_non_hawks := non_hawks - specified_non_hawks
  let percentage_unspecified := unspecified_non_hawks * 100 / total_birds
  percentage_unspecified = 14 := by
  sorry

end reserve_bird_percentage_l71_71767


namespace find_x_l71_71440

noncomputable def a : ℝ × ℝ := (3, 4)
noncomputable def b : ℝ × ℝ := (2, 1)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem find_x (x : ℝ) (h : dot_product (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2) = 0) : x = -3 :=
  sorry

end find_x_l71_71440


namespace fishing_tomorrow_l71_71143

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l71_71143


namespace rabbit_time_2_miles_l71_71742

def rabbit_travel_time (distance : ℕ) (rate : ℕ) : ℕ :=
  (distance * 60) / rate

theorem rabbit_time_2_miles : rabbit_travel_time 2 5 = 24 := by
  sorry

end rabbit_time_2_miles_l71_71742


namespace total_oranges_picked_l71_71773

-- Defining the number of oranges picked by Mary, Jason, and Sarah
def maryOranges := 122
def jasonOranges := 105
def sarahOranges := 137

-- The theorem to prove that the total number of oranges picked is 364
theorem total_oranges_picked : maryOranges + jasonOranges + sarahOranges = 364 := by
  sorry

end total_oranges_picked_l71_71773


namespace solve_abs_inequality_l71_71993

theorem solve_abs_inequality (x : ℝ) :
  2 ≤ |3 * x - 6| ∧ |3 * x - 6| ≤ 15 ↔ (-3 ≤ x ∧ x ≤ 4 / 3) ∨ (8 / 3 ≤ x ∧ x ≤ 7) := 
sorry

end solve_abs_inequality_l71_71993


namespace W_3_7_eq_13_l71_71411

-- Define the operation W
def W (x y : ℤ) : ℤ := y + 5 * x - x^2

-- State the theorem
theorem W_3_7_eq_13 : W 3 7 = 13 := by
  sorry

end W_3_7_eq_13_l71_71411


namespace height_of_parabolic_arch_l71_71736

theorem height_of_parabolic_arch (a : ℝ) (x : ℝ) (k : ℝ) (h : ℝ) (s : ℝ) :
  k = 20 →
  s = 30 →
  a = - 4 / 45 →
  x = 3 →
  k = h →
  y = a * x^2 + k →
  h = 20 → 
  y = 19.2 :=
by
  -- Given the conditions, we'll prove using provided Lean constructs
  sorry

end height_of_parabolic_arch_l71_71736


namespace person_B_reads_more_than_A_l71_71323

-- Assuming people are identifiers for Person A and Person B.
def pages_read_A (days : ℕ) (daily_read : ℕ) : ℕ := days * daily_read

def pages_read_B (days : ℕ) (daily_read : ℕ) (rest_cycle : ℕ) : ℕ := 
  let full_cycles := days / rest_cycle
  let remainder_days := days % rest_cycle
  let active_days := days - full_cycles
  active_days * daily_read

-- Given conditions
def daily_read_A := 8
def daily_read_B := 13
def rest_cycle_B := 3
def total_days := 7

-- The main theorem to prove
theorem person_B_reads_more_than_A : 
  (pages_read_B total_days daily_read_B rest_cycle_B) - (pages_read_A total_days daily_read_A) = 9 :=
by
  sorry

end person_B_reads_more_than_A_l71_71323


namespace option_B_shares_asymptotes_l71_71195

-- Define the given hyperbola equation
def given_hyperbola (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1

-- The asymptotes for the given hyperbola
def asymptotes_of_given_hyperbola (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Define the hyperbola for option B
def option_B_hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 16) = 1

-- The asymptotes for option B hyperbola
def asymptotes_of_option_B_hyperbola (x y : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Theorem stating that the hyperbola in option B shares the same asymptotes as the given hyperbola
theorem option_B_shares_asymptotes :
  (∀ x y : ℝ, given_hyperbola x y → asymptotes_of_given_hyperbola x y) →
  (∀ x y : ℝ, option_B_hyperbola x y → asymptotes_of_option_B_hyperbola x y) :=
by
  intros h₁ h₂
  -- Here should be the proof to show they have the same asymptotes
  sorry

end option_B_shares_asymptotes_l71_71195


namespace fruits_eaten_total_l71_71399

variable (oranges_per_day : ℕ) (grapes_per_day : ℕ) (days : ℕ)

def total_fruits (oranges_per_day grapes_per_day days : ℕ) : ℕ :=
  (oranges_per_day * days) + (grapes_per_day * days)

theorem fruits_eaten_total 
  (h1 : oranges_per_day = 20)
  (h2 : grapes_per_day = 40) 
  (h3 : days = 30) : 
  total_fruits oranges_per_day grapes_per_day days = 1800 := 
by 
  sorry

end fruits_eaten_total_l71_71399


namespace min_side_length_l71_71659

def table_diagonal (w h : ℕ) : ℕ :=
  Nat.sqrt (w * w + h * h)

theorem min_side_length (w h : ℕ) (S : ℕ) (dw : w = 9) (dh : h = 12) (dS : S = 15) :
  S >= table_diagonal w h :=
by
  sorry

end min_side_length_l71_71659


namespace sum_of_integers_l71_71539

theorem sum_of_integers {n : ℤ} (h : n + 2 = 9) : n + (n + 1) + (n + 2) = 24 := by
  sorry

end sum_of_integers_l71_71539


namespace geometric_sequence_general_term_arithmetic_sequence_sum_l71_71289

noncomputable def a (n : ℕ) : ℕ := 2^n

def b (n : ℕ) : ℕ := 4 * (n - 1)

def S (n : ℕ) : ℕ := 2 * n^2 - 2 * n

theorem geometric_sequence_general_term
    (a1 : ℕ := 2)
    (a4 : ℕ := 16)
    (h1 : a 1 = a1)
    (h2 : a 4 = a4)
    : ∀ n : ℕ, a n = a 1 * 2^(n-1) :=
by
  sorry

theorem arithmetic_sequence_sum
    (a2 : ℕ := 4)
    (a5 : ℕ := 32)
    (b2 : ℕ := a 2)
    (b9 : ℕ := a 5)
    (h1 : b 2 = b2)
    (h2 : b 9 = b9)
    : ∀ n : ℕ, S n = n * (n - 1) * 2 :=
by
  sorry

end geometric_sequence_general_term_arithmetic_sequence_sum_l71_71289


namespace least_times_to_eat_l71_71210

theorem least_times_to_eat (A B C : ℕ) (h1 : A = (9 * B) / 5) (h2 : B = C / 8) : 
  A = 2 ∧ B = 1 ∧ C = 8 :=
sorry

end least_times_to_eat_l71_71210


namespace find_roots_and_m_l71_71404

theorem find_roots_and_m (m a : ℝ) (h_root : (-2)^2 - 4 * (-2) + m = 0) :
  m = -12 ∧ a = 6 :=
by
  sorry

end find_roots_and_m_l71_71404


namespace rectangle_area_l71_71618

theorem rectangle_area (y : ℝ) (h : y > 0) 
    (h_area : ∃ (E F G H : ℝ × ℝ), 
        E = (0, 0) ∧ 
        F = (0, 5) ∧ 
        G = (y, 5) ∧ 
        H = (y, 0) ∧ 
        5 * y = 45) : 
    y = 9 := 
by
    sorry

end rectangle_area_l71_71618


namespace commute_days_l71_71344

-- Definitions of the variables
variables (a b c x : ℕ)

-- Given conditions
def condition1 : Prop := a + c = 12
def condition2 : Prop := b + c = 20
def condition3 : Prop := a + b = 14

-- The theorem to prove
theorem commute_days (h1 : condition1 a c) (h2 : condition2 b c) (h3 : condition3 a b) : a + b + c = 23 :=
sorry

end commute_days_l71_71344


namespace dozens_of_golf_balls_l71_71662

theorem dozens_of_golf_balls (total_balls : ℕ) (dozen_size : ℕ) (h1 : total_balls = 156) (h2 : dozen_size = 12) : total_balls / dozen_size = 13 :=
by
  have h_total : total_balls = 156 := h1
  have h_size : dozen_size = 12 := h2
  sorry

end dozens_of_golf_balls_l71_71662


namespace trapezoid_midsegment_inscribed_circle_l71_71542

theorem trapezoid_midsegment_inscribed_circle (P : ℝ) (hP : P = 40) 
    (inscribed : Π (a b c d : ℝ), a + b = c + d) : 
    (∃ (c d : ℝ), (c + d) / 2 = 10) :=
by
  sorry

end trapezoid_midsegment_inscribed_circle_l71_71542


namespace fraction_irreducible_l71_71994

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_irreducible_l71_71994


namespace possible_third_side_l71_71005

theorem possible_third_side (x : ℝ) : (3 + 4 > x) ∧ (abs (4 - 3) < x) → (x = 2) :=
by 
  sorry

end possible_third_side_l71_71005


namespace intersection_lines_l71_71698

theorem intersection_lines (c d : ℝ) :
    (∃ x y, x = (1/3) * y + c ∧ y = (1/3) * x + d ∧ x = 3 ∧ y = -1) →
    c + d = 4 / 3 :=
by
  sorry

end intersection_lines_l71_71698


namespace sum_of_cubes_is_nine_l71_71946

def sum_of_cubes_of_consecutive_integers (n : ℤ) : ℤ :=
  n^3 + (n + 1)^3

theorem sum_of_cubes_is_nine :
  ∃ n : ℤ, sum_of_cubes_of_consecutive_integers n = 9 :=
by
  sorry

end sum_of_cubes_is_nine_l71_71946


namespace picture_distance_from_right_end_l71_71448

def distance_from_right_end_of_wall (wall_width picture_width position_from_left : ℕ) : ℕ := 
  wall_width - (position_from_left + picture_width)

theorem picture_distance_from_right_end :
  ∀ (wall_width picture_width position_from_left : ℕ), 
  wall_width = 24 -> 
  picture_width = 4 -> 
  position_from_left = 5 -> 
  distance_from_right_end_of_wall wall_width picture_width position_from_left = 15 :=
by
  intros wall_width picture_width position_from_left hw hp hp_left
  rw [hw, hp, hp_left]
  sorry

end picture_distance_from_right_end_l71_71448


namespace concentric_circles_false_statement_l71_71012

theorem concentric_circles_false_statement
  (a b c : ℝ)
  (h1 : a < b)
  (h2 : b < c) :
  ¬ (b + a = c + b) :=
sorry

end concentric_circles_false_statement_l71_71012


namespace problem_l71_71002

open Real 

noncomputable def sqrt_log_a (a : ℝ) : ℝ := sqrt (log a / log 10)
noncomputable def sqrt_log_b (b : ℝ) : ℝ := sqrt (log b / log 10)

theorem problem (a b : ℝ) 
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (condition1 : sqrt_log_a a + 2 * sqrt_log_b b + 2 * log (sqrt a) / log 10 + log (sqrt b) / log 10 = 150)
  (int_sqrt_log_a : ∃ (m : ℕ), sqrt_log_a a = m)
  (int_sqrt_log_b : ∃ (n : ℕ), sqrt_log_b b = n)
  (condition2 : a^2 * b = 10^81) :
  a * b = 10^85 :=
sorry

end problem_l71_71002


namespace desired_line_equation_l71_71912

-- Define the center of the circle and the equation of the given line
def center : (ℝ × ℝ) := (-1, 0)
def line1 (x y : ℝ) : Prop := x + y = 0

-- Define the desired line passing through the center of the circle and perpendicular to line1
def line2 (x y : ℝ) : Prop := x + y + 1 = 0

-- The theorem stating that the desired line equation is x + y + 1 = 0
theorem desired_line_equation : ∀ (x y : ℝ),
  (center = (-1, 0)) → (∀ x y, line1 x y → line2 x y) :=
by
  sorry

end desired_line_equation_l71_71912


namespace probability_both_selected_l71_71986

theorem probability_both_selected 
  (p_jamie : ℚ) (p_tom : ℚ) 
  (h1 : p_jamie = 2/3) 
  (h2 : p_tom = 5/7) : 
  (p_jamie * p_tom = 10/21) :=
by
  sorry

end probability_both_selected_l71_71986


namespace min_value_of_a_for_inverse_l71_71857

theorem min_value_of_a_for_inverse (a : ℝ) : 
  (∀ x y : ℝ, x ≥ a → y ≥ a → (x^2 + 4*x ≤ y^2 + 4*y ↔ x ≤ y)) → a = -2 :=
by
  sorry

end min_value_of_a_for_inverse_l71_71857


namespace inequality_max_k_l71_71693

theorem inequality_max_k (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2 * d)^5) ≥ 174960 * a * b * c * d^3 :=
sorry

end inequality_max_k_l71_71693


namespace angle_B_in_triangle_l71_71727

/-- In triangle ABC, if BC = √3, AC = √2, and ∠A = π/3,
then ∠B = π/4. -/
theorem angle_B_in_triangle
  (BC AC : ℝ) (A B : ℝ)
  (hBC : BC = Real.sqrt 3)
  (hAC : AC = Real.sqrt 2)
  (hA : A = Real.pi / 3) :
  B = Real.pi / 4 :=
sorry

end angle_B_in_triangle_l71_71727


namespace ratio_of_r_l71_71534

theorem ratio_of_r
  (total : ℕ) (r_amount : ℕ) (pq_amount : ℕ)
  (h_total : total = 7000 )
  (h_r_amount : r_amount = 2800 )
  (h_pq_amount : pq_amount = total - r_amount) :
  (r_amount / Nat.gcd r_amount pq_amount, pq_amount / Nat.gcd r_amount pq_amount) = (2, 3) :=
by
  sorry

end ratio_of_r_l71_71534


namespace smallest_n_exists_square_smallest_n_exists_cube_l71_71439

open Nat

-- Statement for part (a)
theorem smallest_n_exists_square (n x y : ℕ) : (∀ n x y, (x * (x + n) = y^2) → (∃ (x y : ℕ), n = 3 ∧ (x * (x + 3) = y^2))) := sorry

-- Statement for part (b)
theorem smallest_n_exists_cube (n x y : ℕ) : (∀ n x y, (x * (x + n) = y^3) → (∃ (x y : ℕ), n = 2 ∧ (x * (x + 2) = y^3))) := sorry

end smallest_n_exists_square_smallest_n_exists_cube_l71_71439


namespace not_enough_money_l71_71403

-- Define the prices of the books
def price_animal_world : Real := 21.8
def price_fairy_tale_stories : Real := 19.5

-- Define the total amount of money Xiao Ming has
def xiao_ming_money : Real := 40.0

-- Define the statement we want to prove
theorem not_enough_money : (price_animal_world + price_fairy_tale_stories) > xiao_ming_money := by
  sorry

end not_enough_money_l71_71403


namespace suitable_communication_l71_71034

def is_suitable_to_communicate (beijing_time : Nat) (sydney_difference : Int) (los_angeles_difference : Int) : Bool :=
  let sydney_time := beijing_time + sydney_difference
  let los_angeles_time := beijing_time - los_angeles_difference
  sydney_time >= 8 ∧ sydney_time <= 22 -- let's assume suitable time is between 8:00 to 22:00

theorem suitable_communication:
  let beijing_time := 18
  let sydney_difference := 2
  let los_angeles_difference := 15
  is_suitable_to_communicate beijing_time sydney_difference los_angeles_difference = true :=
by
  sorry

end suitable_communication_l71_71034


namespace volume_of_prism_l71_71355

theorem volume_of_prism (l w h : ℝ) (hlw : l * w = 15) (hwh : w * h = 20) (hlh : l * h = 24) : l * w * h = 60 := 
sorry

end volume_of_prism_l71_71355


namespace car_value_decrease_per_year_l71_71396

theorem car_value_decrease_per_year 
  (initial_value : ℝ) (final_value : ℝ) (years : ℝ) (decrease_per_year : ℝ)
  (h1 : initial_value = 20000)
  (h2 : final_value = 14000)
  (h3 : years = 6)
  (h4 : initial_value - final_value = 6 * decrease_per_year) : 
  decrease_per_year = 1000 :=
sorry

end car_value_decrease_per_year_l71_71396


namespace product_of_fraction_l71_71893

theorem product_of_fraction (x : ℚ) (h : x = 17 / 999) : 17 * 999 = 16983 := by sorry

end product_of_fraction_l71_71893


namespace incorrect_pair_l71_71470

def roots_of_polynomial (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

theorem incorrect_pair : ¬ ∃ x : ℝ, (y = x - 1 ∧ y = x + 1 ∧ roots_of_polynomial x) :=
by
  sorry

end incorrect_pair_l71_71470


namespace inequality_3a3_2b3_3a2b_2ab2_l71_71217

theorem inequality_3a3_2b3_3a2b_2ab2 (a b : ℝ) (h₁ : a ≥ b) (h₂ : b > 0) : 
  3 * a ^ 3 + 2 * b ^ 3 ≥ 3 * a ^ 2 * b + 2 * a * b ^ 2 :=
by
  sorry

end inequality_3a3_2b3_3a2b_2ab2_l71_71217


namespace find_f_7_5_l71_71939

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom f_range_0_1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7_5 : f 7.5 = -0.5 :=
by
  -- The proof goes here
  sorry

end find_f_7_5_l71_71939


namespace evaluate_division_l71_71124

theorem evaluate_division : 64 / 0.08 = 800 := by
  sorry

end evaluate_division_l71_71124


namespace Ray_has_4_nickels_left_l71_71545

def Ray_initial_cents := 95
def Ray_cents_to_Peter := 25
def Ray_cents_to_Randi := 2 * Ray_cents_to_Peter

-- There are 5 cents in each nickel
def cents_per_nickel := 5

-- Nickels Ray originally has
def Ray_initial_nickels := Ray_initial_cents / cents_per_nickel
-- Nickels given to Peter
def Ray_nickels_to_Peter := Ray_cents_to_Peter / cents_per_nickel
-- Nickels given to Randi
def Ray_nickels_to_Randi := Ray_cents_to_Randi / cents_per_nickel
-- Total nickels given away
def Ray_nickels_given_away := Ray_nickels_to_Peter + Ray_nickels_to_Randi
-- Nickels left with Ray
def Ray_nickels_left := Ray_initial_nickels - Ray_nickels_given_away

theorem Ray_has_4_nickels_left :
  Ray_nickels_left = 4 :=
by
  sorry

end Ray_has_4_nickels_left_l71_71545


namespace find_B_l71_71952

variables {a b c A B C : ℝ}

-- Conditions
axiom given_condition_1 : (c - b) / (c - a) = (Real.sin A) / (Real.sin C + Real.sin B)

-- Law of Sines
axiom law_of_sines_1 : (c - b) / (c - a) = a / (c + b)

-- Law of Cosines
axiom law_of_cosines_1 : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c)

-- Target
theorem find_B : B = Real.pi / 3 := 
sorry

end find_B_l71_71952


namespace shadow_length_minor_fullness_l71_71084

/-
An arithmetic sequence {a_n} where the length of shadows a_i decreases by the same amount, the conditions are:
1. The sum of the shadows on the Winter Solstice (a_1), the Beginning of Spring (a_4), and the Vernal Equinox (a_7) is 315 cun.
2. The sum of the shadows on the first nine solar terms is 855 cun.

We need to prove that the shadow length on Minor Fullness day (a_11) is 35 cun (i.e., 3 chi and 5 cun).
-/
theorem shadow_length_minor_fullness 
  (a : ℕ → ℕ) 
  (d : ℤ)
  (h1 : a 1 + a 4 + a 7 = 315) 
  (h2 : 9 * a 1 + 36 * d = 855) 
  (seq : ∀ n : ℕ, a n = a 1 + (n - 1) * d) :
  a 11 = 35 := 
by 
  sorry

end shadow_length_minor_fullness_l71_71084


namespace percentage_female_on_duty_l71_71965

-- Definition of conditions
def on_duty_officers : ℕ := 152
def female_on_duty : ℕ := on_duty_officers / 2
def total_female_officers : ℕ := 400

-- Proof goal
theorem percentage_female_on_duty : (female_on_duty * 100) / total_female_officers = 19 := by
  -- We would complete the proof here
  sorry

end percentage_female_on_duty_l71_71965


namespace product_of_largest_integer_digits_l71_71261

theorem product_of_largest_integer_digits (u v : ℕ) :
  u^2 + v^2 = 45 ∧ u < v → u * v = 18 :=
sorry

end product_of_largest_integer_digits_l71_71261


namespace min_fraction_l71_71743

theorem min_fraction (x A C : ℝ) (hx : x > 0) (hA : A = x^2 + 1/x^2) (hC : C = x + 1/x) :
  ∃ m, m = 2 * Real.sqrt 2 ∧ ∀ B, B > 0 → x^2 + 1/x^2 = B → x + 1/x = C → B / C ≥ m :=
by
  sorry

end min_fraction_l71_71743


namespace problem_l71_71126

theorem problem (f : ℕ → ℕ → ℕ) (h0 : f 1 1 = 1) (h1 : ∀ m n, f m n ∈ {x | x > 0}) 
  (h2 : ∀ m n, f m (n + 1) = f m n + 2) (h3 : ∀ m, f (m + 1) 1 = 2 * f m 1) : 
  f 1 5 = 9 ∧ f 5 1 = 16 ∧ f 5 6 = 26 :=
sorry

end problem_l71_71126


namespace school_robes_l71_71068

theorem school_robes (total_singers robes_needed : ℕ) (robe_cost total_spent existing_robes : ℕ) 
  (h1 : total_singers = 30)
  (h2 : robe_cost = 2)
  (h3 : total_spent = 36)
  (h4 : total_singers - total_spent / robe_cost = existing_robes) :
  existing_robes = 12 :=
by sorry

end school_robes_l71_71068


namespace cyclic_quadrilaterals_count_l71_71307

theorem cyclic_quadrilaterals_count :
  ∃ n : ℕ, n = 568 ∧
  ∀ (a b c d : ℕ), 
    a + b + c + d = 32 ∧
    a ≤ b ∧ b ≤ c ∧ c ≤ d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a + b + c > d) ∧ (b + c + d > a) ∧ (c + d + a > b) ∧ (d + a + b > c) ∧
    (c - a)^2 + (d - b)^2 = (c + a)^2 + (d + b)^2
      → n = 568 := 
sorry

end cyclic_quadrilaterals_count_l71_71307


namespace trigonometric_identity_l71_71859

theorem trigonometric_identity 
  (x : ℝ)
  (h : Real.sin (2 * x + Real.pi / 5) = Real.sqrt 3 / 3) : 
  Real.sin (4 * Real.pi / 5 - 2 * x) + Real.sin (3 * Real.pi / 10 - 2 * x)^2 = (2 + Real.sqrt 3) / 3 :=
by
  sorry

end trigonometric_identity_l71_71859


namespace division_remainder_l71_71666

/-- The remainder when 3572 is divided by 49 is 44. -/
theorem division_remainder :
  3572 % 49 = 44 :=
by
  sorry

end division_remainder_l71_71666


namespace min_value_proof_l71_71867

noncomputable def min_value (x y : ℝ) : ℝ :=
  (y / x) + (1 / y)

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  (min_value x y) ≥ 4 :=
by
  sorry

end min_value_proof_l71_71867


namespace perpendicular_lines_l71_71663

theorem perpendicular_lines :
  ∃ y x : ℝ, (4 * y - 3 * x = 15) ∧ (3 * y + 4 * x = 12) :=
by
  sorry

end perpendicular_lines_l71_71663


namespace solve_system_eqns_l71_71902

theorem solve_system_eqns :
  ∀ x y z : ℝ, 
  (x * y + 5 * y * z - 6 * x * z = -2 * z) ∧
  (2 * x * y + 9 * y * z - 9 * x * z = -12 * z) ∧
  (y * z - 2 * x * z = 6 * z) →
  x = -2 ∧ y = 2 ∧ z = 1 / 6 ∨
  y = 0 ∧ z = 0 ∨
  x = 0 ∧ z = 0 :=
by
  sorry

end solve_system_eqns_l71_71902


namespace probability_at_least_one_woman_l71_71153

theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ)
  (h1 : total_people = 10) (h2 : men = 5) (h3 : women = 5) (h4 : selected = 3) :
  (1 - (men / total_people) * ((men - 1) / (total_people - 1)) * ((men - 2) / (total_people - 2))) = 5 / 6 :=
by
  sorry

end probability_at_least_one_woman_l71_71153


namespace eval_f_neg2_l71_71081

-- Define the function f
def f (x : ℤ) : ℤ := x^2 - 3*x + 1

-- Theorem statement
theorem eval_f_neg2 : f (-2) = 11 := by
  sorry

end eval_f_neg2_l71_71081


namespace math_problem_l71_71968

theorem math_problem : 1003^2 - 997^2 - 1001^2 + 999^2 = 8000 := by
  sorry

end math_problem_l71_71968


namespace rotational_homothety_commutes_l71_71140

-- Definitions for our conditions
variable (H1 H2 : Point → Point)

-- Definition of rotational homothety. 
-- You would define it based on your bespoke library/formalization.
axiom is_rot_homothety : ∀ (H : Point → Point), Prop

-- Main theorem statement
theorem rotational_homothety_commutes (H1 H2 : Point → Point) (A : Point) 
    (h1_rot : is_rot_homothety H1) (h2_rot : is_rot_homothety H2) : 
    (H1 ∘ H2 = H2 ∘ H1) ↔ (H1 (H2 A) = H2 (H1 A)) :=
sorry

end rotational_homothety_commutes_l71_71140


namespace first_part_is_7613_l71_71469

theorem first_part_is_7613 :
  ∃ (n : ℕ), ∃ (d : ℕ), d = 3 ∧ (761 * 10 + d) * 1000 + 829 = n ∧ (n % 9 = 0) ∧ (761 * 10 + d = 7613) := 
by
  sorry

end first_part_is_7613_l71_71469


namespace sum_of_distinct_prime_factors_of_number_is_10_l71_71266

-- Define the constant number 9720
def number : ℕ := 9720

-- Define the distinct prime factors of 9720
def distinct_prime_factors_of_number : List ℕ := [2, 3, 5]

-- Sum function for the list of distinct prime factors
def sum_of_distinct_prime_factors (lst : List ℕ) : ℕ :=
  lst.foldr (.+.) 0

-- The main theorem to prove
theorem sum_of_distinct_prime_factors_of_number_is_10 :
  sum_of_distinct_prime_factors distinct_prime_factors_of_number = 10 := by
  sorry

end sum_of_distinct_prime_factors_of_number_is_10_l71_71266


namespace square_area_from_triangle_perimeter_l71_71129

noncomputable def perimeter_triangle (a b c : ℝ) : ℝ := a + b + c

noncomputable def side_length_square (perimeter : ℝ) : ℝ := perimeter / 4

noncomputable def area_square (side_length : ℝ) : ℝ := side_length * side_length

theorem square_area_from_triangle_perimeter 
  (a b c : ℝ) 
  (h₁ : a = 5.5) 
  (h₂ : b = 7.5) 
  (h₃ : c = 11) 
  (h₄ : perimeter_triangle a b c = 24) 
  : area_square (side_length_square (perimeter_triangle a b c)) = 36 := 
by 
  simp [perimeter_triangle, side_length_square, area_square, h₁, h₂, h₃, h₄]
  sorry

end square_area_from_triangle_perimeter_l71_71129


namespace total_trees_after_planting_l71_71181

-- Definitions based on conditions
def initial_trees : ℕ := 34
def trees_to_plant : ℕ := 49

-- Statement to prove the total number of trees after planting
theorem total_trees_after_planting : initial_trees + trees_to_plant = 83 := 
by 
  sorry

end total_trees_after_planting_l71_71181


namespace tan_product_identity_l71_71831

theorem tan_product_identity : (1 + Real.tan (Real.pi / 180 * 17)) * (1 + Real.tan (Real.pi / 180 * 28)) = 2 := by
  sorry

end tan_product_identity_l71_71831


namespace oak_trees_initially_in_park_l71_71957

def initialOakTrees (new_oak_trees total_oak_trees_after: ℕ) : ℕ :=
  total_oak_trees_after - new_oak_trees

theorem oak_trees_initially_in_park (new_oak_trees total_oak_trees_after initial_oak_trees : ℕ) 
  (h_new_trees : new_oak_trees = 2) 
  (h_total_after : total_oak_trees_after = 11) 
  (h_correct : initial_oak_trees = 9) : 
  initialOakTrees new_oak_trees total_oak_trees_after = initial_oak_trees := 
by 
  rw [h_new_trees, h_total_after, h_correct]
  sorry

end oak_trees_initially_in_park_l71_71957


namespace odd_prime_2wy_factors_l71_71089

theorem odd_prime_2wy_factors (w y : ℕ) (h1 : Nat.Prime w) (h2 : Nat.Prime y) (h3 : ¬ Even w) (h4 : ¬ Even y) (h5 : w < y) (h6 : Nat.totient (2 * w * y) = 8) :
  w = 3 :=
sorry

end odd_prime_2wy_factors_l71_71089


namespace number_of_red_balls_l71_71805

theorem number_of_red_balls
    (black_balls : ℕ)
    (frequency : ℝ)
    (total_balls : ℕ)
    (red_balls : ℕ) 
    (h_black : black_balls = 5)
    (h_frequency : frequency = 0.25)
    (h_total : total_balls = black_balls / frequency) :
    red_balls = total_balls - black_balls → red_balls = 15 :=
by
  intros h_red
  sorry

end number_of_red_balls_l71_71805


namespace customer_payment_probability_l71_71280

theorem customer_payment_probability :
  let total_customers := 100
  let age_40_50_non_mobile := 13
  let age_50_60_non_mobile := 27
  let total_40_60_non_mobile := age_40_50_non_mobile + age_50_60_non_mobile
  let probability := (total_40_60_non_mobile : ℚ) / total_customers
  probability = 2 / 5 := by
sorry

end customer_payment_probability_l71_71280


namespace rectangle_square_overlap_l71_71036

theorem rectangle_square_overlap (ABCD EFGH : Type) (s x y : ℝ)
  (h1 : 0.3 * s^2 = 0.6 * x * y)
  (h2 : AB = 2 * s)
  (h3 : AD = y)
  (h4 : x * y = 0.5 * s^2) :
  x / y = 8 :=
sorry

end rectangle_square_overlap_l71_71036


namespace largest_alpha_l71_71146

theorem largest_alpha (a b : ℕ) (h1 : a < b) (h2 : b < 2 * a) (N : ℕ) :
  ∃ (α : ℝ), α = 1 / (2 * a^2 - 2 * a * b + b^2) ∧
  (∃ marked_cells : ℕ, marked_cells ≥ α * (N:ℝ)^2) :=
by
  sorry

end largest_alpha_l71_71146


namespace find_x2_times_sum_roots_l71_71219

noncomputable def sqrt2015 := Real.sqrt 2015

theorem find_x2_times_sum_roots
  (x1 x2 x3 : ℝ)
  (h_eq : ∀ x : ℝ, sqrt2015 * x^3 - 4030 * x^2 + 2 = 0 → x = x1 ∨ x = x2 ∨ x = x3)
  (h_ineq : x1 < x2 ∧ x2 < x3) :
  x2 * (x1 + x3) = 2 := by
  sorry

end find_x2_times_sum_roots_l71_71219


namespace find_a_b_l71_71995

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem find_a_b : 
  (∀ x : ℝ, f (g x a b) = 9 * x^2 + 6 * x + 1) ↔ ((a = 3 ∧ b = 1) ∨ (a = -3 ∧ b = -1)) :=
by
  sorry

end find_a_b_l71_71995


namespace roger_left_money_correct_l71_71706

noncomputable def roger_left_money (P : ℝ) (q : ℝ) (E : ℝ) (r1 : ℝ) (C : ℝ) (r2 : ℝ) : ℝ :=
  let feb_expense := q * P
  let after_feb := P - feb_expense
  let mar_expense := E * r1
  let after_mar := after_feb - mar_expense
  let mom_gift := C * r2
  after_mar + mom_gift

theorem roger_left_money_correct :
  roger_left_money 45 0.35 20 1.2 46 0.8 = 42.05 :=
by
  sorry

end roger_left_money_correct_l71_71706


namespace fraction_simplify_l71_71603

theorem fraction_simplify : (7 + 21) / (14 + 42) = 1 / 2 := by
  sorry

end fraction_simplify_l71_71603


namespace sum_a_b_eq_neg2_l71_71255

theorem sum_a_b_eq_neg2 (a b : ℝ) (h : (a - 2)^2 + |b + 4| = 0) : a + b = -2 := 
by 
  sorry

end sum_a_b_eq_neg2_l71_71255


namespace common_solution_l71_71635

-- Define the conditions of the equations as hypotheses
variables (x y : ℝ)

-- First equation
def eq1 := x^2 + y^2 = 4

-- Second equation
def eq2 := x^2 = 4*y - 8

-- Proof statement: If there exists real numbers x and y such that both equations hold,
-- then y must be equal to 2.
theorem common_solution (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) : y = 2 :=
sorry

end common_solution_l71_71635


namespace number_of_men_for_2km_road_l71_71105

noncomputable def men_for_1km_road : ℕ := 30
noncomputable def days_for_1km_road : ℕ := 12
noncomputable def hours_per_day_for_1km_road : ℕ := 8
noncomputable def length_of_1st_road : ℕ := 1
noncomputable def length_of_2nd_road : ℕ := 2
noncomputable def working_hours_per_day_2nd_road : ℕ := 14
noncomputable def days_for_2km_road : ℝ := 20.571428571428573

theorem number_of_men_for_2km_road (total_man_hours_1km : ℕ := men_for_1km_road * days_for_1km_road * hours_per_day_for_1km_road):
  (men_for_1km_road * length_of_2nd_road * days_for_1km_road * hours_per_day_for_1km_road = 5760) →
  ∃ (men_for_2nd_road : ℕ), men_for_1km_road * 2 * days_for_1km_road * hours_per_day_for_1km_road = 5760 ∧  men_for_2nd_road * days_for_2km_road * working_hours_per_day_2nd_road = 5760 ∧ men_for_2nd_road = 20 :=
by {
  sorry
}

end number_of_men_for_2km_road_l71_71105


namespace rational_solutions_for_k_l71_71459

theorem rational_solutions_for_k :
  ∀ (k : ℕ), k > 0 → 
  (∃ x : ℚ, k * x^2 + 16 * x + k = 0) ↔ k = 8 :=
by
  sorry

end rational_solutions_for_k_l71_71459


namespace set_intersection_nonempty_l71_71844

theorem set_intersection_nonempty {a : ℕ} (h : ({0, a} ∩ {1, 2} : Set ℕ) ≠ ∅) :
  a = 1 ∨ a = 2 := by
  sorry

end set_intersection_nonempty_l71_71844


namespace cos_beta_value_cos_2alpha_plus_beta_value_l71_71214

-- Definitions of the conditions
variables (α β : ℝ)
variable (condition1 : 0 < α ∧ α < π / 2)
variable (condition2 : π / 2 < β ∧ β < π)
variable (condition3 : Real.cos (α + π / 4) = 1 / 3)
variable (condition4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3)

-- Proof problem (1)
theorem cos_beta_value :
  ∀ α β, (0 < α ∧ α < π / 2) →
  (π / 2 < β ∧ β < π) →
  (Real.cos (α + π / 4) = 1 / 3) →
  (Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) →
  Real.cos β = - 4 * Real.sqrt 2 / 9 :=
by
  intros α β condition1 condition2 condition3 condition4
  sorry

-- Proof problem (2)
theorem cos_2alpha_plus_beta_value :
  ∀ α β, (0 < α ∧ α < π / 2) →
  (π / 2 < β ∧ β < π) →
  (Real.cos (α + π / 4) = 1 / 3) →
  (Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) →
  Real.cos (2 * α + β) = -1 :=
by
  intros α β condition1 condition2 condition3 condition4
  sorry

end cos_beta_value_cos_2alpha_plus_beta_value_l71_71214


namespace repeating_decimal_to_fraction_l71_71438

theorem repeating_decimal_to_fraction :
  (0.3 + 0.206) = (5057 / 9990) :=
sorry

end repeating_decimal_to_fraction_l71_71438


namespace sum_of_digits_B_l71_71888

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_B (n : ℕ) (h : n = 4444^4444) : digit_sum (digit_sum (digit_sum n)) = 7 :=
by
  sorry

end sum_of_digits_B_l71_71888


namespace intersection_is_singleton_zero_l71_71806

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {-2, 0}

-- Define the theorem to be proved
theorem intersection_is_singleton_zero : M ∩ N = {0} :=
by
  -- Proof is provided by the steps above but not needed here
  sorry

end intersection_is_singleton_zero_l71_71806


namespace initial_bacteria_count_l71_71877

theorem initial_bacteria_count 
  (double_every_30_seconds : ∀ n : ℕ, n * 2^(240 / 30) = 262144) : 
  ∃ n : ℕ, n = 1024 :=
by
  -- Define the initial number of bacteria.
  let n := 262144 / (2^8)
  -- Assert that the initial number is 1024.
  use n
  -- To skip the proof.
  sorry

end initial_bacteria_count_l71_71877


namespace cos_theta_plus_pi_over_3_l71_71762

theorem cos_theta_plus_pi_over_3 {θ : ℝ} (h : Real.sin (θ / 2 + π / 6) = 2 / 3) :
  Real.cos (θ + π / 3) = 1 / 9 :=
by
  sorry

end cos_theta_plus_pi_over_3_l71_71762


namespace inequality_solution_l71_71483

theorem inequality_solution (x : ℝ) : 
  (x ∈ Set.Iio (-3/4) ∪ Set.Ioc 4 5 ∪ Set.Ioi 5) ↔ 
  (x+2) ≠ 0 ∧ (x-2) ≠ 0 ∧ (4 * (x^2 - 1) * (x-2) - (x+2) * (7 * x - 6)) / (4 * (x+2) * (x-2)) ≥ 0 := 
by
  sorry

end inequality_solution_l71_71483


namespace number_conversion_l71_71284

theorem number_conversion (a b c d : ℕ) : 
  4090000 = 409 * 10000 ∧ (a = 800000) ∧ (b = 5000) ∧ (c = 20) ∧ (d = 4) → 
  (a + b + c + d = 805024) :=
by
  sorry

end number_conversion_l71_71284


namespace ram_account_balance_first_year_l71_71243

theorem ram_account_balance_first_year :
  let initial_deposit := 1000
  let interest_first_year := 100
  initial_deposit + interest_first_year = 1100 :=
by
  sorry

end ram_account_balance_first_year_l71_71243


namespace total_donation_l71_71777

-- Define the conditions in the problem
def Barbara_stuffed_animals : ℕ := 9
def Trish_stuffed_animals : ℕ := 2 * Barbara_stuffed_animals
def Barbara_sale_price : ℝ := 2
def Trish_sale_price : ℝ := 1.5

-- Define the goal as a theorem to be proven
theorem total_donation : Barbara_sale_price * Barbara_stuffed_animals + Trish_sale_price * Trish_stuffed_animals = 45 := by
  sorry

end total_donation_l71_71777


namespace exists_n_consecutive_numbers_l71_71949

theorem exists_n_consecutive_numbers:
  ∃ n : ℕ, n % 5 = 0 ∧ (n + 1) % 4 = 0 ∧ (n + 2) % 3 = 0 := sorry

end exists_n_consecutive_numbers_l71_71949


namespace salary_increase_l71_71798

theorem salary_increase (S P : ℝ) (h1 : 0.70 * S + P * (0.70 * S) = 0.91 * S) : P = 0.30 :=
by
  have eq1 : 0.70 * S * (1 + P) = 0.91 * S := by sorry
  have eq2 : S * (0.70 + 0.70 * P) = 0.91 * S := by sorry
  have eq3 : 0.70 + 0.70 * P = 0.91 := by sorry
  have eq4 : 0.70 * P = 0.21 := by sorry
  have eq5 : P = 0.21 / 0.70 := by sorry
  have eq6 : P = 0.30 := by sorry
  exact eq6

end salary_increase_l71_71798


namespace range_of_a_l71_71088

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a <= x ∧ x < y ∧ y <= b → f y <= f x

theorem range_of_a (f : ℝ → ℝ) :
  odd_function f →
  decreasing_on_interval f (-1) 1 →
  (∀ a : ℝ, 0 < a ∧ a < 1 → f (1 - a) + f (2 * a - 1) < 0) →
  (∀ a : ℝ, 0 < a ∧ a < 1) :=
sorry

end range_of_a_l71_71088


namespace abs_val_eq_two_l71_71171

theorem abs_val_eq_two (x : ℝ) (h : |x| = 2) : x = 2 ∨ x = -2 := 
sorry

end abs_val_eq_two_l71_71171


namespace candy_profit_l71_71258

theorem candy_profit :
  let num_bars := 800
  let cost_per_4_bars := 3
  let sell_per_3_bars := 2
  let cost_price := (cost_per_4_bars / 4) * num_bars
  let sell_price := (sell_per_3_bars / 3) * num_bars
  let profit := sell_price - cost_price
  profit = -66.67 :=
by
  sorry

end candy_profit_l71_71258


namespace john_annual_payment_l71_71843

open Real

-- Definitions extracted from the problem:
def epipen_cost : ℝ := 500
def insurance_coverage : ℝ := 0.75
def epipen_frequency_per_year : ℕ := 2
def john_payment_per_epipen : ℝ := epipen_cost * (1 - insurance_coverage)

-- The statement to be proved:
theorem john_annual_payment :
  john_payment_per_epipen * epipen_frequency_per_year = 250 :=
by
  sorry

end john_annual_payment_l71_71843


namespace arithmetic_sequence_conditions_l71_71987

open Nat

theorem arithmetic_sequence_conditions (S : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : S 6 > S 7) (h2 : S 7 > S 5) :
  d < 0 ∧ S 11 > 0 := 
sorry

end arithmetic_sequence_conditions_l71_71987


namespace train_speed_l71_71464

theorem train_speed (L : ℝ) (T : ℝ) (V_m : ℝ) (V_t : ℝ) : (L = 500) → (T = 29.997600191984642) → (V_m = 5 / 6) → (V_t = (L / T) + V_m) → (V_t * 3.6 = 63) :=
by
  intros hL hT hVm hVt
  simp at hL hT hVm hVt
  sorry

end train_speed_l71_71464


namespace seating_arrangements_l71_71306

/-- 
Given seven seats in a row, with four people sitting such that exactly two adjacent seats are empty,
prove that the number of different seating arrangements is 480.
-/
theorem seating_arrangements (seats people : ℕ) (adj_empty : ℕ) : 
  seats = 7 → people = 4 → adj_empty = 2 → 
  (∃ count : ℕ, count = 480) :=
by
  sorry

end seating_arrangements_l71_71306


namespace number_of_students_playing_soccer_l71_71423

variable (total_students boys playing_soccer_girls not_playing_soccer_girls : ℕ)
variable (percentage_boys_playing_soccer : ℕ)

-- Conditions
axiom h1 : total_students = 470
axiom h2 : boys = 300
axiom h3 : not_playing_soccer_girls = 135
axiom h4 : percentage_boys_playing_soccer = 86
axiom h5 : playing_soccer_girls = 470 - 300 - not_playing_soccer_girls

-- Question: Prove that the number of students playing soccer is 250
theorem number_of_students_playing_soccer : 
  (playing_soccer_girls * 100) / (100 - percentage_boys_playing_soccer) = 250 :=
sorry

end number_of_students_playing_soccer_l71_71423


namespace points_lie_on_hyperbola_l71_71180

def point_on_hyperbola (t : ℝ) : Prop :=
  let x := 2 * Real.exp t - 2 * Real.exp (-t)
  let y := 4 * (Real.exp t + Real.exp (-t))
  (y^2) / 16 - (x^2) / 4 = 1

theorem points_lie_on_hyperbola : ∀ t : ℝ, point_on_hyperbola t :=
by
  intro t
  sorry

end points_lie_on_hyperbola_l71_71180


namespace box_volume_l71_71277

-- Given conditions
variables (a b c : ℝ)
axiom ab_eq : a * b = 30
axiom bc_eq : b * c = 18
axiom ca_eq : c * a = 45

-- Prove that the volume of the box (a * b * c) equals 90 * sqrt(3)
theorem box_volume : a * b * c = 90 * Real.sqrt 3 :=
by
  sorry

end box_volume_l71_71277


namespace highest_student_id_in_sample_l71_71972

variable (n : ℕ) (start : ℕ) (interval : ℕ)

theorem highest_student_id_in_sample :
  start = 5 → n = 54 → interval = 9 → 6 = n / interval → start = 5 →
  5 + (interval * (6 - 1)) = 50 :=
by
  sorry

end highest_student_id_in_sample_l71_71972


namespace bianca_total_books_l71_71567

theorem bianca_total_books (shelves_mystery shelves_picture books_per_shelf : ℕ) 
  (h1 : shelves_mystery = 5) 
  (h2 : shelves_picture = 4) 
  (h3 : books_per_shelf = 8) : 
  (shelves_mystery * books_per_shelf + shelves_picture * books_per_shelf) = 72 := 
by 
  sorry

end bianca_total_books_l71_71567


namespace circle_area_with_radius_8_l71_71444

noncomputable def circle_radius : ℝ := 8
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circle_area_with_radius_8 :
  circle_area circle_radius = 64 * Real.pi :=
by
  sorry

end circle_area_with_radius_8_l71_71444


namespace square_diagonal_l71_71953

theorem square_diagonal (A : ℝ) (s : ℝ) (d : ℝ) (hA : A = 338) (hs : s^2 = A) (hd : d^2 = 2 * s^2) : d = 26 :=
by
  -- Proof goes here
  sorry

end square_diagonal_l71_71953


namespace find_t2_l71_71818

variable {P A1 A2 t1 r t2 : ℝ}
def conditions (P A1 A2 t1 r t2 : ℝ) :=
  P = 650 ∧
  A1 = 815 ∧
  A2 = 870 ∧
  t1 = 3 ∧
  A1 = P + (P * r * t1) / 100 ∧
  A2 = P + (P * r * t2) / 100

theorem find_t2
  (P A1 A2 t1 r t2 : ℝ)
  (hc : conditions P A1 A2 t1 r t2) :
  t2 = 4 :=
by
  sorry

end find_t2_l71_71818


namespace time_for_Q_l71_71199

-- Definitions of conditions
def time_for_P := 252
def meet_time := 2772

-- Main statement to prove
theorem time_for_Q : (∃ T : ℕ, lcm time_for_P T = meet_time) ∧ (lcm time_for_P meet_time = meet_time) :=
    by 
    sorry

end time_for_Q_l71_71199


namespace general_term_of_arithmetic_seq_sum_of_first_n_terms_b_n_l71_71194

theorem general_term_of_arithmetic_seq
  (a_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (a_2_eq_3 : a_n 2 = 3)
  (S_4_eq_16 : S_n 4 = 16) :
  (∀ n, a_n n = 2 * n - 1) :=
sorry

theorem sum_of_first_n_terms_b_n
  (a_n : ℕ → ℕ)
  (b_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (general_formula_a_n : ∀ n, a_n n = 2 * n - 1)
  (b_n_definition : ∀ n, b_n n = 1 / (a_n n * a_n (n + 1))) :
  (∀ n, T_n n = n / (2 * n + 1)) :=
sorry

end general_term_of_arithmetic_seq_sum_of_first_n_terms_b_n_l71_71194


namespace smallest_natural_number_B_l71_71299

theorem smallest_natural_number_B (A : ℕ) (h : A % 2 = 0 ∧ A % 3 = 0) :
    ∃ B : ℕ, (360 / (A^3 / B) = 5) ∧ B = 3 :=
by
  sorry

end smallest_natural_number_B_l71_71299


namespace emily_purchased_9_wall_prints_l71_71788

/-
  Given the following conditions:
  - cost_of_each_pair_of_curtains = 30
  - num_of_pairs_of_curtains = 2
  - installation_cost = 50
  - cost_of_each_wall_print = 15
  - total_order_cost = 245

  Prove that Emily purchased 9 wall prints
-/
noncomputable def num_wall_prints_purchased 
  (cost_of_each_pair_of_curtains : ℝ) 
  (num_of_pairs_of_curtains : ℝ) 
  (installation_cost : ℝ) 
  (cost_of_each_wall_print : ℝ) 
  (total_order_cost : ℝ) 
  : ℝ :=
  (total_order_cost - (num_of_pairs_of_curtains * cost_of_each_pair_of_curtains + installation_cost)) / cost_of_each_wall_print

theorem emily_purchased_9_wall_prints
  (cost_of_each_pair_of_curtains : ℝ := 30) 
  (num_of_pairs_of_curtains : ℝ := 2) 
  (installation_cost : ℝ := 50) 
  (cost_of_each_wall_print : ℝ := 15) 
  (total_order_cost : ℝ := 245) :
  num_wall_prints_purchased cost_of_each_pair_of_curtains num_of_pairs_of_curtains installation_cost cost_of_each_wall_print total_order_cost = 9 :=
sorry

end emily_purchased_9_wall_prints_l71_71788


namespace find_passing_marks_l71_71790

-- Defining the conditions as Lean statements
def condition1 (T P : ℝ) : Prop := 0.30 * T = P - 50
def condition2 (T P : ℝ) : Prop := 0.45 * T = P + 25

-- The theorem to prove
theorem find_passing_marks (T P : ℝ) (h1 : condition1 T P) (h2 : condition2 T P) : P = 200 :=
by
  -- Placeholder proof
  sorry

end find_passing_marks_l71_71790


namespace roots_k_m_l71_71304

theorem roots_k_m (k m : ℝ) 
  (h1 : ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b + c = 11 ∧ a * b + b * c + c * a = k ∧ a * b * c = m)
  : k + m = 52 :=
sorry

end roots_k_m_l71_71304


namespace inverse_of_matrix_l71_71383

open Matrix

def mat : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 9], ![2, 5]]

def inv_mat : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![5/2, -9/2], ![-1, 2]]

theorem inverse_of_matrix :
  ∃ (inv : Matrix (Fin 2) (Fin 2) ℚ), 
    inv * mat = 1 ∧ mat * inv = 1 :=
  ⟨inv_mat, by
    -- Providing the proof steps here is beyond the scope
    sorry⟩

end inverse_of_matrix_l71_71383


namespace fourth_number_second_set_l71_71453

theorem fourth_number_second_set :
  (∃ (x y : ℕ), (28 + x + 42 + 78 + 104) / 5 = 90 ∧ (128 + 255 + 511 + y + x) / 5 = 423 ∧ x = 198) →
  (y = 1023) :=
by
  sorry

end fourth_number_second_set_l71_71453


namespace plates_used_l71_71687

theorem plates_used (P : ℕ) (h : 3 * 2 * P + 4 * 8 = 38) : P = 1 := by
  sorry

end plates_used_l71_71687


namespace smallest_period_of_f_l71_71350

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x + Real.cos x) ^ 2 + 1

theorem smallest_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
by
  sorry

end smallest_period_of_f_l71_71350


namespace neg_prop_p_l71_71633

-- Define the function f as a real-valued function
variable (f : ℝ → ℝ)

-- Definitions for the conditions in the problem
def prop_p := ∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0

-- Theorem stating the negation of proposition p
theorem neg_prop_p : ¬prop_p f ↔ ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
by 
  sorry

end neg_prop_p_l71_71633


namespace floor_of_sqrt_sum_eq_floor_of_sqrt_expr_l71_71330

theorem floor_of_sqrt_sum_eq_floor_of_sqrt_expr (n : ℤ): 
  Int.floor (Real.sqrt n + Real.sqrt (n + 1)) = Int.floor (Real.sqrt (4 * n + 2)) := 
sorry

end floor_of_sqrt_sum_eq_floor_of_sqrt_expr_l71_71330


namespace row_speed_with_stream_l71_71861

theorem row_speed_with_stream (v : ℝ) (s : ℝ) (h1 : s = 2) (h2 : v - s = 12) : v + s = 16 := by
  -- Placeholder for the proof
  sorry

end row_speed_with_stream_l71_71861


namespace fish_eaten_by_new_fish_l71_71579

def initial_original_fish := 14
def added_fish := 2
def exchange_new_fish := 3
def total_fish_now := 11

theorem fish_eaten_by_new_fish : initial_original_fish - (total_fish_now - exchange_new_fish) = 6 := by
  -- This is where the proof would go
  sorry

end fish_eaten_by_new_fish_l71_71579


namespace Mina_stops_in_D_or_A_l71_71808

-- Define the relevant conditions and problem statement
def circumference := 60
def total_distance := 6000
def quarters := ["A", "B", "C", "D"]
def start_position := "S"
def stop_position := if (total_distance % circumference) == 0 then "S" else ""

theorem Mina_stops_in_D_or_A : stop_position = start_position → start_position = "D" ∨ start_position = "A" :=
by
  sorry

end Mina_stops_in_D_or_A_l71_71808


namespace power_first_digits_l71_71586

theorem power_first_digits (n : ℕ) (h1 : ∀ k : ℕ, n ≠ 10^k) : ∃ j k : ℕ, 1973 ≤ n^j / 10^k ∧ n^j / 10^k < 1974 := by
  sorry

end power_first_digits_l71_71586


namespace shirts_before_buying_l71_71696

-- Define the conditions
variable (new_shirts : ℕ)
variable (total_shirts : ℕ)

-- Define the statement where we need to prove the number of shirts Sarah had before buying the new ones
theorem shirts_before_buying (h₁ : new_shirts = 8) (h₂ : total_shirts = 17) : total_shirts - new_shirts = 9 :=
by
  -- Proof goes here
  sorry

end shirts_before_buying_l71_71696


namespace types_of_problems_l71_71772

def bill_problems : ℕ := 20
def ryan_problems : ℕ := 2 * bill_problems
def frank_problems : ℕ := 3 * ryan_problems
def problems_per_type : ℕ := 30

theorem types_of_problems : (frank_problems / problems_per_type) = 4 := by
  sorry

end types_of_problems_l71_71772


namespace circle_tangent_to_line_iff_m_eq_zero_l71_71786

theorem circle_tangent_to_line_iff_m_eq_zero (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m^2 ∧ x - y = m) ↔ m = 0 :=
by 
  sorry

end circle_tangent_to_line_iff_m_eq_zero_l71_71786


namespace no_real_solution_l71_71314

theorem no_real_solution (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : 1 / a + 1 / b = 1 / (a + b)) : False :=
by
  sorry

end no_real_solution_l71_71314


namespace simplify_expression_l71_71554

theorem simplify_expression (x : ℝ) :
  (x-1)^4 + 4*(x-1)^3 + 6*(x-1)^2 + 4*(x-1) = x^4 - 1 :=
  by 
    sorry

end simplify_expression_l71_71554


namespace original_paint_intensity_l71_71927

theorem original_paint_intensity
  (I : ℝ) -- Original intensity of the red paint
  (f : ℝ) -- Fraction of the original paint replaced
  (new_intensity : ℝ) -- Intensity of the new paint
  (replacement_intensity : ℝ) -- Intensity of the replacement red paint
  (hf : f = 2 / 3)
  (hreplacement_intensity : replacement_intensity = 0.30)
  (hnew_intensity : new_intensity = 0.40)
  : I = 0.60 := 
sorry

end original_paint_intensity_l71_71927


namespace work_together_days_l71_71782

theorem work_together_days (A_rate B_rate : ℝ) (x B_alone_days : ℝ)
  (hA : A_rate = 1 / 5)
  (hB : B_rate = 1 / 15)
  (h_total_work : (A_rate + B_rate) * x + B_rate * B_alone_days = 1) :
  x = 2 :=
by
  -- Set up the equation based on given rates and solving for x.
  sorry

end work_together_days_l71_71782


namespace johns_share_l71_71186

theorem johns_share
  (total_amount : ℕ)
  (ratio_john : ℕ)
  (ratio_jose : ℕ)
  (ratio_binoy : ℕ)
  (total_parts : ℕ)
  (value_per_part : ℕ)
  (johns_parts : ℕ)
  (johns_share : ℕ)
  (h1 : total_amount = 4800)
  (h2 : ratio_john = 2)
  (h3 : ratio_jose = 4)
  (h4 : ratio_binoy = 6)
  (h5 : total_parts = ratio_john + ratio_jose + ratio_binoy)
  (h6 : value_per_part = total_amount / total_parts)
  (h7 : johns_parts = ratio_john)
  (h8 : johns_share = value_per_part * johns_parts) :
  johns_share = 800 := by
  sorry

end johns_share_l71_71186


namespace sum_series_eq_3_div_4_l71_71139

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l71_71139


namespace selection_count_l71_71313

theorem selection_count (word : String) (vowels : Finset Char) (consonants : Finset Char)
  (hword : word = "УЧЕБНИК")
  (hvowels : vowels = {'У', 'Е', 'И'})
  (hconsonants : consonants = {'Ч', 'Б', 'Н', 'К'})
  :
  vowels.card * consonants.card = 12 :=
by {
  sorry
}

end selection_count_l71_71313


namespace expression_is_integer_l71_71966

theorem expression_is_integer (n : ℕ) : 
  (3 ^ (2 * n) / 112 - 4 ^ (2 * n) / 63 + 5 ^ (2 * n) / 144) = (k : ℤ) :=
sorry

end expression_is_integer_l71_71966


namespace sachin_age_l71_71347

theorem sachin_age (S R : ℕ) (h1 : R = S + 18) (h2 : S * 9 = R * 7) : S = 63 := 
by
  sorry

end sachin_age_l71_71347


namespace fixed_point_exists_l71_71715

theorem fixed_point_exists (m : ℝ) :
  ∀ (x y : ℝ), (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0 → x = 3 ∧ y = 1 :=
by
  sorry

end fixed_point_exists_l71_71715


namespace child_growth_l71_71118

-- Define variables for heights
def current_height : ℝ := 41.5
def previous_height : ℝ := 38.5

-- Define the problem statement in Lean 4
theorem child_growth :
  current_height - previous_height = 3 :=
by 
  sorry

end child_growth_l71_71118


namespace value_of_a_plus_c_l71_71395

variables {a b c : ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

def f_inv (x : ℝ) : ℝ := c * x^2 + b * x + a

theorem value_of_a_plus_c : a + c = -1 :=
sorry

end value_of_a_plus_c_l71_71395


namespace ratio_second_to_first_l71_71943

noncomputable def ratio_of_second_to_first (x y z : ℕ) (k : ℕ) : ℕ := sorry

theorem ratio_second_to_first
    (x y z : ℕ)
    (h1 : z = 2 * y)
    (h2 : y = k * x)
    (h3 : (x + y + z) / 3 = 78)
    (h4 : x = 18)
    (k_val : k = 4):
  ratio_of_second_to_first x y z k = 4 := sorry

end ratio_second_to_first_l71_71943


namespace margie_change_l71_71179

theorem margie_change : 
  let cost_per_apple := 0.30
  let cost_per_orange := 0.40
  let number_of_apples := 5
  let number_of_oranges := 4
  let total_money := 10.00
  let total_cost_of_apples := cost_per_apple * number_of_apples
  let total_cost_of_oranges := cost_per_orange * number_of_oranges
  let total_cost_of_fruits := total_cost_of_apples + total_cost_of_oranges
  let change_received := total_money - total_cost_of_fruits
  change_received = 6.90 :=
by
  sorry

end margie_change_l71_71179


namespace clowns_attended_l71_71785

-- Definition of the problem's conditions
def num_children : ℕ := 30
def initial_candies : ℕ := 700
def candies_sold_per_person : ℕ := 20
def remaining_candies : ℕ := 20

-- Main theorem stating that 4 clowns attended the carousel
theorem clowns_attended (num_clowns : ℕ) (candies_left: num_clowns * candies_sold_per_person + num_children * candies_sold_per_person = initial_candies - remaining_candies) : num_clowns = 4 := by
  sorry

end clowns_attended_l71_71785


namespace less_than_subtraction_l71_71380

-- Define the numbers as real numbers
def a : ℝ := 47.2
def b : ℝ := 0.5

-- Theorem statement
theorem less_than_subtraction : a - b = 46.7 :=
by
  sorry

end less_than_subtraction_l71_71380


namespace profit_function_and_optimal_price_l71_71548

variable (cost selling base_units additional_units: ℝ)
variable (x: ℝ) (y: ℝ)

def profit (x: ℝ): ℝ := -20 * x^2 + 100 * x + 6000

theorem profit_function_and_optimal_price:
  (cost = 40) →
  (selling = 60) →
  (base_units = 300) →
  (additional_units = 20) →
  (0 ≤ x) →
  (x < 20) →
  (y = profit x) →
  exists x_max y_max: ℝ, (x_max = 2.5) ∧ (y_max = 6125) :=
by 
  sorry

end profit_function_and_optimal_price_l71_71548


namespace largest_circle_center_is_A_l71_71017

-- Define the given lengths of the pentagon's sides
def AB : ℝ := 16
def BC : ℝ := 14
def CD : ℝ := 17
def DE : ℝ := 13
def AE : ℝ := 14

-- Define the radii of the circles centered at points A, B, C, D, E
variables (R_A R_B R_C R_D R_E : ℝ)

-- Conditions based on the problem statement
def radius_conditions : Prop :=
  R_A + R_B = AB ∧
  R_B + R_C = BC ∧
  R_C + R_D = CD ∧
  R_D + R_E = DE ∧
  R_E + R_A = AE

-- The main theorem to prove
theorem largest_circle_center_is_A (h : radius_conditions R_A R_B R_C R_D R_E) :
  10 ≥ R_A ∧ R_A ≥ R_B ∧ R_A ≥ R_C ∧ R_A ≥ R_D ∧ R_A ≥ R_E :=
by sorry

end largest_circle_center_is_A_l71_71017


namespace rational_linear_function_l71_71547

theorem rational_linear_function (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x :=
sorry

end rational_linear_function_l71_71547


namespace f_g_5_l71_71162

def g (x : ℕ) : ℕ := 4 * x + 10

def f (x : ℕ) : ℕ := 6 * x - 12

theorem f_g_5 : f (g 5) = 168 := by
  sorry

end f_g_5_l71_71162


namespace positive_difference_l71_71413

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 := by
  sorry

end positive_difference_l71_71413


namespace blood_drops_per_liter_l71_71959

def mosquito_drops : ℕ := 20
def fatal_blood_loss_liters : ℕ := 3
def mosquitoes_to_kill : ℕ := 750

theorem blood_drops_per_liter (D : ℕ) (total_drops : ℕ) : 
  (total_drops = mosquitoes_to_kill * mosquito_drops) → 
  (fatal_blood_loss_liters * D = total_drops) → 
  D = 5000 := 
  by 
    intros h1 h2
    sorry

end blood_drops_per_liter_l71_71959


namespace train_speed_in_km_per_hr_l71_71292

-- Definitions from the problem conditions
def length_of_train : ℝ := 50
def time_to_cross_pole : ℝ := 3

-- Conversion factor from the problem 
def meter_per_sec_to_km_per_hr : ℝ := 3.6

-- Lean theorem statement based on problem conditions and solution
theorem train_speed_in_km_per_hr : 
  (length_of_train / time_to_cross_pole) * meter_per_sec_to_km_per_hr = 60 := by
  sorry

end train_speed_in_km_per_hr_l71_71292


namespace martin_initial_spending_l71_71976

theorem martin_initial_spending :
  ∃ (x : ℝ), 
    ∀ (a b : ℝ), 
      a = x - 100 →
      b = a - 0.20 * a →
      x - b = 280 →
      x = 1000 :=
by
  sorry

end martin_initial_spending_l71_71976


namespace election_margin_of_victory_l71_71933

theorem election_margin_of_victory (T : ℕ) (H_winning_votes : T * 58 / 100 = 1044) :
  1044 - (T * 42 / 100) = 288 :=
by
  sorry

end election_margin_of_victory_l71_71933


namespace solve_system_of_equations_l71_71358

theorem solve_system_of_equations :
  ∀ (x1 x2 x3 x4 x5: ℝ), 
  (x3 + x4 + x5)^5 = 3 * x1 ∧ 
  (x4 + x5 + x1)^5 = 3 * x2 ∧ 
  (x5 + x1 + x2)^5 = 3 * x3 ∧ 
  (x1 + x2 + x3)^5 = 3 * x4 ∧ 
  (x2 + x3 + x4)^5 = 3 * x5 →
  (x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨ 
  (x1 = 1/3 ∧ x2 = 1/3 ∧ x3 = 1/3 ∧ x4 = 1/3 ∧ x5 = 1/3) ∨ 
  (x1 = -1/3 ∧ x2 = -1/3 ∧ x3 = -1/3 ∧ x4 = -1/3 ∧ x5 = -1/3) := 
by 
  sorry

end solve_system_of_equations_l71_71358


namespace substance_same_number_of_atoms_l71_71339

def molecule (kind : String) (atom_count : ℕ) := (kind, atom_count)

def H3PO4 := molecule "H₃PO₄" 8
def H2O2 := molecule "H₂O₂" 4
def H2SO4 := molecule "H₂SO₄" 7
def NaCl := molecule "NaCl" 2 -- though it consists of ions, let's denote it as 2 for simplicity
def HNO3 := molecule "HNO₃" 5

def mol_atoms (mol : ℝ) (molecule : ℕ) : ℝ := mol * molecule

theorem substance_same_number_of_atoms :
  mol_atoms 0.2 H3PO4.2 = mol_atoms 0.4 H2O2.2 :=
by
  unfold H3PO4 H2O2 mol_atoms
  sorry

end substance_same_number_of_atoms_l71_71339


namespace min_visible_pairs_l71_71472

-- Define the problem conditions
def bird_circle_flock (P : ℕ) : Prop :=
  P = 155

def mutual_visibility_condition (θ : ℝ) : Prop :=
  θ ≤ 10

-- Define the minimum number of mutually visible pairs
def min_mutual_visible_pairs (P_pairs : ℕ) : Prop :=
  P_pairs = 270

-- The main theorem statement
theorem min_visible_pairs (n : ℕ) (θ : ℝ) (P_pairs : ℕ)
  (H1 : bird_circle_flock n)
  (H2 : mutual_visibility_condition θ) :
  min_mutual_visible_pairs P_pairs :=
by
  sorry

end min_visible_pairs_l71_71472


namespace compare_sqrt_l71_71803

noncomputable def a : ℝ := 3 * Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 15

theorem compare_sqrt : a > b :=
by
  sorry

end compare_sqrt_l71_71803


namespace Louie_monthly_payment_l71_71184

noncomputable def monthly_payment (P : ℕ) (r : ℚ) (n t : ℕ) : ℚ :=
  (P : ℚ) * (1 + r / n)^(n * t) / t

theorem Louie_monthly_payment : 
  monthly_payment 2000 0.10 1 3 = 887 := 
by
  sorry

end Louie_monthly_payment_l71_71184


namespace math_problem_l71_71093

variables (x y : ℝ)

noncomputable def question_value (x y : ℝ) : ℝ := (2 * x - 5 * y) / (5 * x + 2 * y)

theorem math_problem 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (cond : (5 * x - 2 * y) / (2 * x + 3 * y) = 1) : 
  question_value x y = -5 / 31 :=
sorry

end math_problem_l71_71093


namespace subsets_containing_six_l71_71609

theorem subsets_containing_six :
  ∃ s : Finset (Fin 6), s = {1, 2, 3, 4, 5, 6} ∧ (∃ n : ℕ, n = 32 ∧ n = 2 ^ 5) := by
  sorry

end subsets_containing_six_l71_71609


namespace binom_18_6_eq_18564_l71_71172

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binom_18_6_eq_18564 : binomial 18 6 = 18564 := by
  sorry

end binom_18_6_eq_18564_l71_71172


namespace smallest_four_digit_mod_8_l71_71201

theorem smallest_four_digit_mod_8 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 8 = 5 ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 8 = 5 → n ≤ m) → n = 1005 :=
by
  sorry

end smallest_four_digit_mod_8_l71_71201


namespace incorrect_proposition3_l71_71764

open Real

-- Definitions from the problem
def prop1 (x : ℝ) := 2 * sin (2 * x - π / 3) = 2
def prop2 (x y : ℝ) := tan x + tan (π - x) = 0
def prop3 (x1 x2 : ℝ) (k : ℤ) := x1 - x2 = (k : ℝ) * π → k % 2 = 1
def prop4 (x : ℝ) := cos x ^ 2 + sin x >= -1

-- Incorrect proposition proof
theorem incorrect_proposition3 (x1 x2 : ℝ) (k : ℤ) :
  sin (2 * x1 - π / 4) = 0 →
  sin (2 * x2 - π / 4) = 0 →
  x1 - x2 ≠ (k : ℝ) * π := sorry

end incorrect_proposition3_l71_71764


namespace find_k_square_binomial_l71_71247

theorem find_k_square_binomial (k : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 - 16 * x + k = (x + b)^2) ↔ k = 64 :=
by
  sorry

end find_k_square_binomial_l71_71247


namespace ratio_x_y_l71_71428

theorem ratio_x_y (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : x / y = 11 / 6 :=
by
  sorry

end ratio_x_y_l71_71428


namespace find_missing_score_l71_71768

theorem find_missing_score
  (scores : List ℕ)
  (h_scores : scores = [73, 83, 86, 73, x])
  (mean : ℚ)
  (h_mean : mean = 79.2)
  (h_length : scores.length = 5)
  : x = 81 := by
  sorry

end find_missing_score_l71_71768


namespace chicken_bucket_feeds_l71_71979

theorem chicken_bucket_feeds :
  ∀ (cost_per_bucket : ℝ) (total_cost : ℝ) (total_people : ℕ),
  cost_per_bucket = 12 →
  total_cost = 72 →
  total_people = 36 →
  (total_people / (total_cost / cost_per_bucket)) = 6 :=
by
  intros cost_per_bucket total_cost total_people h1 h2 h3
  sorry

end chicken_bucket_feeds_l71_71979


namespace solution_set_abs_inequality_l71_71817

theorem solution_set_abs_inequality (x : ℝ) :
  (|2 - x| ≥ 1) ↔ (x ≤ 1 ∨ x ≥ 3) :=
by
  sorry

end solution_set_abs_inequality_l71_71817


namespace division_result_l71_71009

theorem division_result:
    35 / 0.07 = 500 := by
  sorry

end division_result_l71_71009


namespace num_packs_blue_tshirts_l71_71649

def num_white_tshirts_per_pack : ℕ := 6
def num_packs_white_tshirts : ℕ := 5
def num_blue_tshirts_per_pack : ℕ := 9
def total_num_tshirts : ℕ := 57

theorem num_packs_blue_tshirts : (total_num_tshirts - num_white_tshirts_per_pack * num_packs_white_tshirts) / num_blue_tshirts_per_pack = 3 := by
  sorry

end num_packs_blue_tshirts_l71_71649


namespace train_speed_is_correct_l71_71229

-- Definitions based on the conditions
def length_of_train : ℝ := 120       -- Train is 120 meters long
def time_to_cross : ℝ := 16          -- The train takes 16 seconds to cross the post

-- Conversion constants
def seconds_to_hours : ℝ := 3600
def meters_to_kilometers : ℝ := 1000

-- The speed of the train in km/h
noncomputable def speed_of_train (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (seconds_to_hours / meters_to_kilometers)

-- Theorem: The speed of the train is 27 km/h
theorem train_speed_is_correct : speed_of_train length_of_train time_to_cross = 27 :=
by
  -- This is where the proof should be, but we leave it as sorry as instructed
  sorry

end train_speed_is_correct_l71_71229


namespace f_2015_value_l71_71664

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom f_is_odd : odd_function f
axiom f_periodicity : ∀ x : ℝ, f (x + 2) = -f x
axiom f_definition_in_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 3^x - 1

theorem f_2015_value : f 2015 = -2 :=
by
  sorry

end f_2015_value_l71_71664


namespace arithmetic_progression_a6_l71_71329

theorem arithmetic_progression_a6 (a1 d : ℤ) (h1 : a1 + (a1 + d) + (a1 + 2 * d) = 168) (h2 : (a1 + 4 * d) - (a1 + d) = 42) : 
  a1 + 5 * d = 3 := 
sorry

end arithmetic_progression_a6_l71_71329


namespace motorboat_time_to_C_l71_71525

variables (r s p t_B : ℝ)

-- Condition declarations
def kayak_speed := r + s
def motorboat_speed := p
def meeting_time := 12

-- Problem statement: to prove the time it took for the motorboat to reach dock C before turning back
theorem motorboat_time_to_C :
  (2 * r + s) * t_B = r * 12 + s * 6 → t_B = (r * 12 + s * 6) / (2 * r + s) := 
by
  intros h
  sorry

end motorboat_time_to_C_l71_71525


namespace paul_account_balance_after_transactions_l71_71572

theorem paul_account_balance_after_transactions :
  let transfer1 := 90
  let transfer2 := 60
  let service_charge_percent := 2 / 100
  let initial_balance := 400
  let service_charge1 := service_charge_percent * transfer1
  let service_charge2 := service_charge_percent * transfer2
  let total_deduction1 := transfer1 + service_charge1
  let total_deduction2 := transfer2 + service_charge2
  let reversed_transfer2 := total_deduction2 - transfer2
  let balance_after_transfer1 := initial_balance - total_deduction1
  let final_balance := balance_after_transfer1 - reversed_transfer2
  (final_balance = 307) :=
by
  sorry

end paul_account_balance_after_transactions_l71_71572


namespace algebra_or_drafting_not_both_l71_71735

theorem algebra_or_drafting_not_both {A D : Finset ℕ} (h1 : (A ∩ D).card = 10) (h2 : A.card = 24) (h3 : D.card - (A ∩ D).card = 11) : (A ∪ D).card - (A ∩ D).card = 25 := by
  sorry

end algebra_or_drafting_not_both_l71_71735


namespace smallest_perimeter_of_square_sides_l71_71836

/-
  Define a predicate for the triangle inequality condition for squares of integers.
-/
def triangle_ineq_squares (a b c : ℕ) : Prop :=
  (a < b) ∧ (b < c) ∧ (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2)

/-
  Statement that proves the smallest possible perimeter given the conditions.
-/
theorem smallest_perimeter_of_square_sides : 
  ∃ a b c : ℕ, a < b ∧ b < c ∧ triangle_ineq_squares a b c ∧ a^2 + b^2 + c^2 = 77 :=
sorry

end smallest_perimeter_of_square_sides_l71_71836


namespace solution_l71_71614

def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem solution (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by 
  -- Here we will skip the actual proof by using sorry
  sorry

end solution_l71_71614


namespace parrots_count_l71_71501

theorem parrots_count (p r : ℕ) : 2 * p + 4 * r = 26 → p + r = 10 → p = 7 := by
  intros h1 h2
  sorry

end parrots_count_l71_71501


namespace largest_value_of_n_l71_71950

theorem largest_value_of_n (A B n : ℤ) (h1 : A * B = 72) (h2 : n = 6 * B + A) : n = 433 :=
sorry

end largest_value_of_n_l71_71950


namespace general_formula_l71_71763

open Nat

def a (n : ℕ) : ℚ :=
  if n = 0 then 7/6 else 0 -- Recurrence initialization with dummy else condition

-- Defining the recurrence relation as a function
lemma recurrence_relation {n : ℕ} (h : n > 0) : 
    a n = (1 / 2) * a (n - 1) + (1 / 3) := 
sorry

-- Proof of the general formula
theorem general_formula (n : ℕ) : a n = (1 / (2^n : ℚ)) + (2 / 3) :=
sorry

end general_formula_l71_71763


namespace parabola_equation_l71_71505

theorem parabola_equation (d : ℝ) (p : ℝ) (x y : ℝ) (h1 : d = 2) (h2 : y = 2) (h3 : x = 1) :
  y^2 = 4 * x :=
sorry

end parabola_equation_l71_71505


namespace num_three_digit_integers_with_zero_in_units_place_divisible_by_30_l71_71183

noncomputable def countThreeDigitMultiplesOf30WithZeroInUnitsPlace : ℕ :=
  let a := 120
  let d := 30
  let l := 990
  (l - a) / d + 1

theorem num_three_digit_integers_with_zero_in_units_place_divisible_by_30 :
  countThreeDigitMultiplesOf30WithZeroInUnitsPlace = 30 := by
  sorry

end num_three_digit_integers_with_zero_in_units_place_divisible_by_30_l71_71183


namespace cherries_eaten_l71_71513

-- Define the number of cherries Oliver had initially
def initial_cherries : ℕ := 16

-- Define the number of cherries Oliver had left after eating some
def left_cherries : ℕ := 6

-- Prove that the difference between the initial and left cherries is 10
theorem cherries_eaten : initial_cherries - left_cherries = 10 := by
  sorry

end cherries_eaten_l71_71513


namespace diameter_of_double_area_square_l71_71918

-- Define the given conditions and the problem to be solved
theorem diameter_of_double_area_square (d₁ : ℝ) (d₁_eq : d₁ = 4 * Real.sqrt 2) :
  ∃ d₂ : ℝ, d₂ = 8 :=
by
  -- Define the conditions
  let s₁ := d₁ / Real.sqrt 2
  have s₁_sq : s₁ ^ 2 = (d₁ ^ 2) / 2 := by sorry -- Pythagorean theorem

  let A₁ := s₁ ^ 2
  have A₁_eq : A₁ = 16 := by sorry -- Given diagonal, thus area

  let A₂ := 2 * A₁
  have A₂_eq : A₂ = 32 := by sorry -- Double the area

  let s₂ := Real.sqrt A₂
  have s₂_eq : s₂ = 4 * Real.sqrt 2 := by sorry -- Side length of second square

  let d₂ := s₂ * Real.sqrt 2
  have d₂_eq : d₂ = 8 := by sorry -- Diameter of the second square

  -- Prove the theorem
  existsi d₂
  exact d₂_eq

end diameter_of_double_area_square_l71_71918


namespace find_value_of_a5_l71_71865

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a_1 d : ℝ), ∀ n, a n = a_1 + (n - 1) * d

variable (h_arith : is_arithmetic_sequence a)
variable (h : a 2 + a 8 = 12)

theorem find_value_of_a5 : a 5 = 6 :=
by
  sorry

end find_value_of_a5_l71_71865


namespace expression_value_correct_l71_71623

theorem expression_value_correct (a b : ℤ) (h1 : a = -3) (h2 : b = 2) : -a - b^3 + a * b = -11 := by
  sorry

end expression_value_correct_l71_71623


namespace new_deck_card_count_l71_71549

-- Define the conditions
def cards_per_time : ℕ := 30
def times_per_week : ℕ := 3
def weeks : ℕ := 11
def decks : ℕ := 18
def total_cards_tear_per_week : ℕ := cards_per_time * times_per_week
def total_cards_tear : ℕ := total_cards_tear_per_week * weeks
def total_cards_in_decks (cards_per_deck : ℕ) : ℕ := decks * cards_per_deck

-- Define the theorem we need to prove
theorem new_deck_card_count :
  ∃ (x : ℕ), total_cards_in_decks x = total_cards_tear ↔ x = 55 := by
  sorry

end new_deck_card_count_l71_71549


namespace prob_both_students_female_l71_71037

-- Define the conditions
def total_students : ℕ := 5
def male_students : ℕ := 2
def female_students : ℕ := 3
def selected_students : ℕ := 2

-- Define the function to compute binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function to select 2 female students
def probability_both_female : ℚ := 
  (binomial female_students selected_students : ℚ) / (binomial total_students selected_students : ℚ)

-- The actual theorem to be proved
theorem prob_both_students_female : probability_both_female = 0.3 := by
  sorry

end prob_both_students_female_l71_71037


namespace neg_p_neither_sufficient_nor_necessary_l71_71514

-- Definitions of p and q as described
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

-- Proving that ¬p is neither a sufficient nor necessary condition for q
theorem neg_p_neither_sufficient_nor_necessary (x : ℝ) : 
  ( ¬ p x → q x ) = false ∧ ( q x → ¬ p x ) = false := by
  sorry

end neg_p_neither_sufficient_nor_necessary_l71_71514


namespace find_rth_term_l71_71364

def S (n : ℕ) : ℕ := 2 * n + 3 * (n^3)

def a (r : ℕ) : ℕ := S r - S (r - 1)

theorem find_rth_term (r : ℕ) : a r = 9 * r^2 - 9 * r + 5 := by
  sorry

end find_rth_term_l71_71364


namespace probability_full_house_after_rerolling_l71_71398

theorem probability_full_house_after_rerolling
  (a b c : ℕ)
  (h0 : a ≠ b)
  (h1 : c ≠ a)
  (h2 : c ≠ b) :
  (2 / 6 : ℚ) = (1 / 3 : ℚ) :=
by
  sorry

end probability_full_house_after_rerolling_l71_71398


namespace third_set_candies_l71_71311

-- Define the types of candies in each set
variables {L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ}

-- Equal total candies condition
axiom total_candies : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3

-- Conditions for the first set
axiom first_set_conditions : S1 = M1 ∧ L1 = S1 + 7

-- Conditions for the second set
axiom second_set_conditions : L2 = S2 ∧ M2 = L2 - 15

-- Condition for the third set
axiom no_hard_candies_in_third_set : L3 = 0

-- The objective to be proved
theorem third_set_candies : L3 + S3 + M3 = 29 :=
  by
    sorry

end third_set_candies_l71_71311


namespace sum_1026_is_2008_l71_71556

def sequence_sum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let groups_sum : ℕ := (n * n)
    let extra_2s := (2008 - groups_sum) / 2
    (n * (n + 1)) / 2 + extra_2s

theorem sum_1026_is_2008 : sequence_sum 1026 = 2008 :=
  sorry

end sum_1026_is_2008_l71_71556


namespace distinct_valid_sets_count_l71_71872

-- Define non-negative powers of 2 and 3
def is_non_neg_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a ∨ n = 3^b

-- Define the condition for sum of elements in set S to be 2014
def valid_sets (S : Finset ℕ) : Prop :=
  (∀ x ∈ S, is_non_neg_power x) ∧ (S.sum id = 2014)

theorem distinct_valid_sets_count : ∃ (number_of_distinct_sets : ℕ), number_of_distinct_sets = 64 :=
  sorry

end distinct_valid_sets_count_l71_71872


namespace determine_ABC_l71_71799

theorem determine_ABC : 
  ∀ (A B C : ℝ), 
    A = 2 * B - 3 * C ∧ 
    B = 2 * C - 5 ∧ 
    A + B + C = 100 → 
    A = 18.75 ∧ B = 52.5 ∧ C = 28.75 :=
by
  intro A B C h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end determine_ABC_l71_71799


namespace find_f_of_3_l71_71110

theorem find_f_of_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/x + 2) = x) : f 3 = 1 := 
sorry

end find_f_of_3_l71_71110


namespace red_balloon_count_l71_71963

theorem red_balloon_count (total_balloons : ℕ) (green_balloons : ℕ) (red_balloons : ℕ) :
  total_balloons = 17 →
  green_balloons = 9 →
  red_balloons = total_balloons - green_balloons →
  red_balloons = 8 := by
  sorry

end red_balloon_count_l71_71963


namespace find_prime_pairs_l71_71103

open Nat

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem find_prime_pairs :
  ∀ m n : ℕ,
  is_prime m → is_prime n → (m < n ∧ n < 5 * m) → is_prime (m + 3 * n) →
  (m = 2 ∧ (n = 3 ∨ n = 5 ∨ n = 7)) :=
by
  sorry

end find_prime_pairs_l71_71103


namespace Q_over_P_l71_71256

theorem Q_over_P :
  (∀ (x : ℝ), x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 6 → 
    (P / (x + 6) + Q / (x^2 - 6*x) = (x^2 - 3*x + 12) / (x^3 + x^2 - 24*x))) →
  Q / P = 5 / 3 :=
by
  sorry

end Q_over_P_l71_71256


namespace first_term_of_geometric_sequence_l71_71097

theorem first_term_of_geometric_sequence
  (a r : ℚ) -- where a is the first term and r is the common ratio
  (h1 : a * r^4 = 45) -- fifth term condition
  (h2 : a * r^5 = 60) -- sixth term condition
  : a = 1215 / 256 := 
sorry

end first_term_of_geometric_sequence_l71_71097


namespace complement_of_M_l71_71322

noncomputable def U : Set ℝ := Set.univ
noncomputable def M : Set ℝ := {a | a ^ 2 - 2 * a > 0}
noncomputable def C_U_M : Set ℝ := U \ M

theorem complement_of_M :
  C_U_M = {a | 0 ≤ a ∧ a ≤ 2} :=
by
  sorry

end complement_of_M_l71_71322


namespace probability_sum_of_two_dice_is_4_l71_71250

noncomputable def fair_dice_probability_sum_4 : ℚ :=
  let total_outcomes := 6 * 6 -- Total outcomes for two dice
  let favorable_outcomes := 3 -- Outcomes that sum to 4: (1, 3), (3, 1), (2, 2)
  favorable_outcomes / total_outcomes

theorem probability_sum_of_two_dice_is_4 : fair_dice_probability_sum_4 = 1 / 12 := 
by
  sorry

end probability_sum_of_two_dice_is_4_l71_71250


namespace speed_of_first_car_l71_71116

-- Define the conditions
def t : ℝ := 3.5
def v : ℝ := sorry -- (To be solved in the proof)
def speed_second_car : ℝ := 58
def total_distance : ℝ := 385

-- The distance each car travels after t hours
def distance_first_car : ℝ := v * t
def distance_second_car : ℝ := speed_second_car * t

-- The equation representing the total distance between the two cars after 3.5 hours
def equation := distance_first_car + distance_second_car = total_distance

-- The main theorem stating the speed of the first car
theorem speed_of_first_car : v = 52 :=
by
  -- The important proof steps would go here solving the equation "equation".
  sorry

end speed_of_first_car_l71_71116


namespace cistern_total_wet_surface_area_l71_71390

-- Define the length, width, and depth of water in the cistern
def length : ℝ := 9
def width : ℝ := 4
def depth : ℝ := 1.25

-- Define the bottom surface area
def bottom_surface_area : ℝ := length * width

-- Define the longer side surface area
def longer_side_surface_area_each : ℝ := depth * length

-- Define the shorter end surface area
def shorter_end_surface_area_each : ℝ := depth * width

-- Calculate the total wet surface area
def total_wet_surface_area : ℝ := bottom_surface_area + 2 * longer_side_surface_area_each + 2 * shorter_end_surface_area_each

-- The theorem to be proved
theorem cistern_total_wet_surface_area :
  total_wet_surface_area = 68.5 :=
by
  -- since bottom_surface_area = 36,
  -- 2 * longer_side_surface_area_each = 22.5, and
  -- 2 * shorter_end_surface_area_each = 10
  -- the total will be equal to 68.5
  sorry

end cistern_total_wet_surface_area_l71_71390


namespace division_result_l71_71265

theorem division_result : 3486 / 189 = 18.444444444444443 := 
by sorry

end division_result_l71_71265


namespace coin_difference_l71_71077

variables (x y : ℕ)

theorem coin_difference (h1 : x + y = 15) (h2 : 2 * x + 5 * y = 51) : x - y = 1 := by
  sorry

end coin_difference_l71_71077


namespace fraction_simplification_addition_l71_71461

theorem fraction_simplification_addition :
  (∃ a b : ℕ, 0.4375 = (a : ℚ) / b ∧ Nat.gcd a b = 1 ∧ a + b = 23) :=
by
  sorry

end fraction_simplification_addition_l71_71461


namespace intersection_A_B_l71_71504

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end intersection_A_B_l71_71504


namespace calculate_Y_payment_l71_71530

-- Define the known constants
def total_payment : ℝ := 590
def x_to_y_ratio : ℝ := 1.2

-- Main theorem statement, asserting the value of Y's payment
theorem calculate_Y_payment (Y : ℝ) (X : ℝ) 
  (h1 : X = x_to_y_ratio * Y) 
  (h2 : X + Y = total_payment) : 
  Y = 268.18 :=
by
  sorry

end calculate_Y_payment_l71_71530


namespace Panikovsky_share_l71_71936

theorem Panikovsky_share :
  ∀ (horns hooves weight : ℕ) 
    (k δ : ℝ),
    horns = 17 →
    hooves = 2 →
    weight = 1 →
    (∀ h, h = k + δ) →
    (∀ wt, wt = k + 2 * δ) →
    (20 * k + 19 * δ) / 2 = 10 * k + 9.5 * δ →
    9 * k + 7.5 * δ = (9 * (k + δ) + 2 * k) →
    ∃ (Panikov_hearts Panikov_hooves : ℕ), 
    Panikov_hearts = 9 ∧ Panikov_hooves = 2 := 
by
  intros
  sorry

end Panikovsky_share_l71_71936


namespace inequality_true_l71_71349

noncomputable def f : ℝ → ℝ := sorry -- f is a function defined on (0, +∞)

axiom f_derivative (x : ℝ) (hx : 0 < x) : ∃ f'' : ℝ → ℝ, f'' x * x + 2 * f x = 1 / x^2

theorem inequality_true : (f 2) / 9 < (f 3) / 4 :=
  sorry

end inequality_true_l71_71349


namespace find_least_integer_l71_71973

theorem find_least_integer (x : ℤ) : (3 * |x| - 4 < 20) → (x ≥ -7) :=
by
  sorry

end find_least_integer_l71_71973


namespace domain_of_function_l71_71485

-- Definitions of the conditions

def sqrt_condition (x : ℝ) : Prop := -x^2 - 3*x + 4 ≥ 0
def log_condition (x : ℝ) : Prop := x + 1 > 0 ∧ x + 1 ≠ 1

-- Statement of the problem

theorem domain_of_function :
  {x : ℝ | sqrt_condition x ∧ log_condition x} = { x | -1 < x ∧ x < 0 ∨ 0 < x ∧ x ≤ 1 } :=
sorry

end domain_of_function_l71_71485


namespace total_price_all_art_l71_71082

-- Define the conditions
def total_price_first_three_pieces : ℕ := 45000
def price_next_piece := (total_price_first_three_pieces / 3) * 3 / 2 

-- Statement to prove
theorem total_price_all_art : total_price_first_three_pieces + price_next_piece = 67500 :=
by
  sorry -- Proof is omitted

end total_price_all_art_l71_71082


namespace probability_two_students_same_school_l71_71704

/-- Definition of the problem conditions -/
def total_students : ℕ := 3
def total_schools : ℕ := 4
def total_basic_events : ℕ := total_schools ^ total_students
def favorable_events : ℕ := 36

/-- Theorem stating the probability of exactly two students choosing the same school -/
theorem probability_two_students_same_school : 
  favorable_events / (total_schools ^ total_students) = 9 / 16 := 
  sorry

end probability_two_students_same_school_l71_71704


namespace determine_k_l71_71631

theorem determine_k (k : ℝ) (h1 : ∃ x y : ℝ, y = 4 * x + 3 ∧ y = -2 * x - 25 ∧ y = 3 * x + k) : k = -5 / 3 := by
  sorry

end determine_k_l71_71631


namespace teamA_teamB_repair_eq_l71_71938

-- conditions
def teamADailyRepair (x : ℕ) := x -- represent Team A repairing x km/day
def teamBDailyRepair (x : ℕ) := x + 3 -- represent Team B repairing x + 3 km/day
def timeTaken (distance rate: ℕ) := distance / rate -- time = distance / rate

-- Proof problem statement
theorem teamA_teamB_repair_eq (x : ℕ) (hx : x > 0) (hx_plus_3 : x + 3 > 0) :
  timeTaken 6 (teamADailyRepair x) = timeTaken 8 (teamBDailyRepair x) → (6 / x = 8 / (x + 3)) :=
by
  intros h
  sorry

end teamA_teamB_repair_eq_l71_71938


namespace maximum_ab_ac_bc_l71_71117

theorem maximum_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 5) : 
  ab + ac + bc ≤ 25 / 6 :=
sorry

end maximum_ab_ac_bc_l71_71117


namespace total_flag_distance_moved_l71_71095

def flagpole_length : ℕ := 60

def initial_raise_distance : ℕ := flagpole_length

def lower_to_half_mast_distance : ℕ := flagpole_length / 2

def raise_from_half_mast_distance : ℕ := flagpole_length / 2

def final_lower_distance : ℕ := flagpole_length

theorem total_flag_distance_moved :
  initial_raise_distance + lower_to_half_mast_distance + raise_from_half_mast_distance + final_lower_distance = 180 :=
by
  sorry

end total_flag_distance_moved_l71_71095


namespace pair_ab_l71_71899

def students_activities_ways (n_students n_activities : Nat) : Nat :=
  n_activities ^ n_students

def championships_outcomes (n_championships n_students : Nat) : Nat :=
  n_students ^ n_championships

theorem pair_ab (a b : Nat) :
  a = students_activities_ways 4 3 ∧ b = championships_outcomes 3 4 →
  (a, b) = (3^4, 4^3) := by
  sorry

end pair_ab_l71_71899


namespace average_price_of_fruit_l71_71771

theorem average_price_of_fruit 
  (price_apple price_orange : ℝ)
  (total_fruits initial_fruits kept_oranges kept_fruits : ℕ)
  (average_price_kept average_price_initial : ℝ)
  (h1 : price_apple = 40)
  (h2 : price_orange = 60)
  (h3 : initial_fruits = 10)
  (h4 : kept_oranges = initial_fruits - 6)
  (h5 : average_price_kept = 50) :
  average_price_initial = 56 := 
sorry

end average_price_of_fruit_l71_71771


namespace nina_shoe_payment_l71_71493

theorem nina_shoe_payment :
  let first_pair_original := 22
  let first_pair_discount := 0.10 * first_pair_original
  let first_pair_discounted := first_pair_original - first_pair_discount
  let first_pair_tax := 0.05 * first_pair_discounted
  let first_pair_final := first_pair_discounted + first_pair_tax

  let second_pair_original := first_pair_original * 1.50
  let second_pair_discount := 0.15 * second_pair_original
  let second_pair_discounted := second_pair_original - second_pair_discount
  let second_pair_tax := 0.07 * second_pair_discounted
  let second_pair_final := second_pair_discounted + second_pair_tax

  let total_payment := first_pair_final + second_pair_final
  total_payment = 50.80 :=
by 
  sorry

end nina_shoe_payment_l71_71493


namespace find_range_of_a_l71_71433

-- Definitions and conditions
def pointA : ℝ × ℝ := (0, 3)
def lineL (x : ℝ) : ℝ := 2 * x - 4
def circleCenter (a : ℝ) : ℝ × ℝ := (a, 2 * a - 4)
def circleRadius : ℝ := 1

-- The range to prove
def valid_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 12 / 5

-- Main theorem
theorem find_range_of_a (a : ℝ) (M : ℝ × ℝ)
  (on_circle : (M.1 - (circleCenter a).1)^2 + (M.2 - (circleCenter a).2)^2 = circleRadius^2)
  (condition_MA_MD : (M.1 - pointA.1)^2 + (M.2 - pointA.2)^2 = 4 * M.1^2 + 4 * M.2^2) :
  valid_range a :=
sorry

end find_range_of_a_l71_71433


namespace solution_set_inequality_l71_71254

theorem solution_set_inequality (m : ℝ) (h : 3 - m < 0) :
  { x : ℝ | (2 - m) * x + 2 > m } = { x : ℝ | x < -1 } :=
sorry

end solution_set_inequality_l71_71254


namespace fill_tank_in_18_minutes_l71_71741

-- Define the conditions
def rate_pipe_A := 1 / 9  -- tanks per minute
def rate_pipe_B := - (1 / 18) -- tanks per minute (negative because it's emptying)

-- Define the net rate of both pipes working together
def net_rate := rate_pipe_A + rate_pipe_B

-- Define the time to fill the tank when both pipes are working
def time_to_fill_tank := 1 / net_rate

theorem fill_tank_in_18_minutes : time_to_fill_tank = 18 := 
    by
    -- Sorry to skip the actual proof
    sorry

end fill_tank_in_18_minutes_l71_71741


namespace mass_percentage_H_correct_l71_71500

noncomputable def mass_percentage_H_in_CaH2 : ℝ :=
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_H : ℝ := 1.01
  let molar_mass_CaH2 : ℝ := molar_mass_Ca + 2 * molar_mass_H
  (2 * molar_mass_H / molar_mass_CaH2) * 100

theorem mass_percentage_H_correct :
  |mass_percentage_H_in_CaH2 - 4.80| < 0.01 :=
by
  sorry

end mass_percentage_H_correct_l71_71500


namespace correct_article_usage_l71_71155

def sentence : String :=
  "While he was at ____ college, he took part in the march, and was soon thrown into ____ prison."

def rules_for_articles (context : String) (noun : String) : String → Bool
| "the" => noun ≠ "college" ∨ context = "specific"
| ""    => noun = "college" ∨ noun = "prison"
| _     => false

theorem correct_article_usage : 
  rules_for_articles "general" "college" "" ∧ 
  rules_for_articles "general" "prison" "" :=
by
  sorry

end correct_article_usage_l71_71155


namespace average_speed_of_car_l71_71000

noncomputable def average_speed (total_distance total_time : ℕ) : ℕ :=
  total_distance / total_time

theorem average_speed_of_car :
  let d1 := 80
  let d2 := 40
  let d3 := 60
  let d4 := 50
  let d5 := 90
  let d6 := 100
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  average_speed total_distance total_time = 70 :=
by
  let d1 := 80
  let d2 := 40
  let d3 := 60
  let d4 := 50
  let d5 := 90
  let d6 := 100
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  exact sorry

end average_speed_of_car_l71_71000


namespace semicircle_inequality_l71_71107

-- Define the points on the semicircle
variables (A B C D E : ℝ)
-- Define the length function
def length (X Y : ℝ) : ℝ := abs (X - Y)

-- This is the main theorem statement
theorem semicircle_inequality {A B C D E : ℝ} :
  length A B ^ 2 + length B C ^ 2 + length C D ^ 2 + length D E ^ 2 +
  length A B * length B C * length C D + length B C * length C D * length D E < 4 :=
sorry

end semicircle_inequality_l71_71107


namespace a_plus_b_values_l71_71884

theorem a_plus_b_values (a b : ℤ) (h1 : |a + 1| = 0) (h2 : b^2 = 9) :
  a + b = 2 ∨ a + b = -4 :=
by
  have ha : a = -1 := by sorry
  have hb1 : b = 3 ∨ b = -3 := by sorry
  cases hb1 with
  | inl b_pos =>
    left
    rw [ha, b_pos]
    exact sorry
  | inr b_neg =>
    right
    rw [ha, b_neg]
    exact sorry

end a_plus_b_values_l71_71884


namespace min_value_of_quadratic_l71_71391

theorem min_value_of_quadratic (x y : ℝ) : (x^2 + 2*x*y + y^2) ≥ 0 ∧ ∃ x y, x = -y ∧ x^2 + 2*x*y + y^2 = 0 := by
  sorry

end min_value_of_quadratic_l71_71391


namespace greatest_divisor_arithmetic_sum_l71_71881

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l71_71881


namespace percentage_women_no_french_speak_spanish_german_l71_71670

variable (total_workforce : Nat)
variable (men_percentage women_percentage : ℕ)
variable (men_only_french men_only_spanish men_only_german : ℕ)
variable (men_both_french_spanish men_both_french_german men_both_spanish_german : ℕ)
variable (men_all_three_languages women_only_french women_only_spanish : ℕ)
variable (women_only_german women_both_french_spanish women_both_french_german : ℕ)
variable (women_both_spanish_german women_all_three_languages : ℕ)

-- Conditions
axiom h1 : men_percentage = 60
axiom h2 : women_percentage = 40
axiom h3 : women_only_french = 30
axiom h4 : women_only_spanish = 25
axiom h5 : women_only_german = 20
axiom h6 : women_both_french_spanish = 10
axiom h7 : women_both_french_german = 5
axiom h8 : women_both_spanish_german = 5
axiom h9 : women_all_three_languages = 5

theorem percentage_women_no_french_speak_spanish_german:
  women_only_spanish + women_only_german + women_both_spanish_german = 50 := by
  sorry

end percentage_women_no_french_speak_spanish_german_l71_71670


namespace fraction_division_l71_71601

variable {x : ℝ}
variable (hx : x ≠ 0)

theorem fraction_division (hx : x ≠ 0) : (3 / 8) / (5 * x / 12) = 9 / (10 * x) := 
by
  sorry

end fraction_division_l71_71601


namespace friends_attended_l71_71263

theorem friends_attended (total_guests bride_couples groom_couples : ℕ)
                         (bride_guests groom_guests family_guests friends : ℕ)
                         (h1 : total_guests = 300)
                         (h2 : bride_couples = 30)
                         (h3 : groom_couples = 30)
                         (h4 : bride_guests = bride_couples * 2)
                         (h5 : groom_guests = groom_couples * 2)
                         (h6 : family_guests = bride_guests + groom_guests)
                         (h7 : friends = total_guests - family_guests) :
  friends = 180 :=
by sorry

end friends_attended_l71_71263


namespace cindy_age_l71_71497

-- Define the ages involved
variables (C J M G : ℕ)

-- Define the conditions
def jan_age_condition : Prop := J = C + 2
def marcia_age_condition : Prop := M = 2 * J
def greg_age_condition : Prop := G = M + 2
def greg_age_known : Prop := G = 16

-- The statement we need to prove
theorem cindy_age : 
  jan_age_condition C J → 
  marcia_age_condition J M → 
  greg_age_condition M G → 
  greg_age_known G → 
  C = 5 := 
by 
  -- Sorry is used here to skip the proof
  sorry

end cindy_age_l71_71497


namespace total_monkeys_l71_71886

theorem total_monkeys (x : ℕ) (h : (1 / 8 : ℝ) * x ^ 2 + 12 = x) : x = 48 :=
sorry

end total_monkeys_l71_71886


namespace maximize_x2y5_l71_71669

theorem maximize_x2y5 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 50) : 
  x = 100 / 7 ∧ y = 250 / 7 :=
sorry

end maximize_x2y5_l71_71669


namespace rhombus_perimeter_l71_71732

theorem rhombus_perimeter
  (d1 d2 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 16) :
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 8 * Real.sqrt 41 := 
  sorry

end rhombus_perimeter_l71_71732


namespace calculate_expression_l71_71568

theorem calculate_expression :
  |1 - Real.sqrt 2| + (1/2)^(-2 : ℤ) - (Real.pi - 2023)^0 = Real.sqrt 2 + 2 := 
by
  sorry

end calculate_expression_l71_71568


namespace smallest_fraction_gt_five_sevenths_l71_71192

theorem smallest_fraction_gt_five_sevenths (a b : ℕ) (h1 : 10 ≤ a ∧ a ≤ 99) (h2 : 10 ≤ b ∧ b ≤ 99) (h3 : 7 * a > 5 * b) : a = 68 ∧ b = 95 :=
sorry

end smallest_fraction_gt_five_sevenths_l71_71192


namespace total_chairs_l71_71276

def numIndoorTables := 9
def numOutdoorTables := 11
def chairsPerIndoorTable := 10
def chairsPerOutdoorTable := 3

theorem total_chairs :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 :=
by
  sorry

end total_chairs_l71_71276


namespace annual_income_correct_l71_71159

-- Define the principal amounts and interest rates
def principal_1 : ℝ := 3000
def rate_1 : ℝ := 0.085

def principal_2 : ℝ := 5000
def rate_2 : ℝ := 0.064

-- Define the interest calculations for each investment
def interest_1 : ℝ := principal_1 * rate_1
def interest_2 : ℝ := principal_2 * rate_2

-- Define the total annual income
def total_annual_income : ℝ := interest_1 + interest_2

-- Proof statement
theorem annual_income_correct : total_annual_income = 575 :=
by
  sorry

end annual_income_correct_l71_71159


namespace tan_identity_15_eq_sqrt3_l71_71224

theorem tan_identity_15_eq_sqrt3 :
  (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by
  sorry

end tan_identity_15_eq_sqrt3_l71_71224


namespace range_of_a_l71_71784

theorem range_of_a (a : ℝ) (h : 2 * a - 1 ≤ 11) : a < 6 :=
by
  sorry

end range_of_a_l71_71784


namespace points_per_enemy_l71_71207

theorem points_per_enemy (total_enemies : ℕ) (destroyed_enemies : ℕ) (total_points : ℕ) 
  (h1 : total_enemies = 7)
  (h2 : destroyed_enemies = total_enemies - 2)
  (h3 : destroyed_enemies = 5)
  (h4 : total_points = 40) :
  total_points / destroyed_enemies = 8 :=
by
  sorry

end points_per_enemy_l71_71207


namespace initial_apples_l71_71951

theorem initial_apples (A : ℕ) 
  (H1 : A - 2 + 4 + 5 = 14) : 
  A = 7 := 
by 
  sorry

end initial_apples_l71_71951


namespace ones_digit_of_73_pow_351_l71_71559

theorem ones_digit_of_73_pow_351 : 
  (73 ^ 351) % 10 = 7 := 
by 
  sorry

end ones_digit_of_73_pow_351_l71_71559


namespace sum_of_29_12_23_is_64_sixtyfour_is_two_to_six_l71_71144

theorem sum_of_29_12_23_is_64: 29 + 12 + 23 = 64 := sorry

theorem sixtyfour_is_two_to_six:
  64 = 2^6 := sorry

end sum_of_29_12_23_is_64_sixtyfour_is_two_to_six_l71_71144


namespace find_g_5_l71_71297

noncomputable def g : ℝ → ℝ := sorry

axiom g_property (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0

theorem find_g_5 : g 5 = 1 :=
by
  sorry

end find_g_5_l71_71297


namespace complex_fraction_identity_l71_71079

theorem complex_fraction_identity (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 1)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1 / 3 :=
by 
  sorry

end complex_fraction_identity_l71_71079


namespace least_positive_n_for_reducible_fraction_l71_71701

theorem least_positive_n_for_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℤ, k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (6 * n + 7)) ∧ n = 126 :=
by
  sorry

end least_positive_n_for_reducible_fraction_l71_71701


namespace find_a5_of_geom_seq_l71_71157

theorem find_a5_of_geom_seq 
  (a : ℕ → ℝ) (q : ℝ)
  (hgeom : ∀ n, a (n + 1) = a n * q)
  (S : ℕ → ℝ)
  (hS3 : S 3 = a 0 * (1 - q ^ 3) / (1 - q))
  (hS6 : S 6 = a 0 * (1 - q ^ 6) / (1 - q))
  (hS9 : S 9 = a 0 * (1 - q ^ 9) / (1 - q))
  (harith : S 3 + S 6 = 2 * S 9)
  (a8 : a 8 = 3) :
  a 5 = -6 :=
by
  sorry

end find_a5_of_geom_seq_l71_71157


namespace three_pos_reals_inequality_l71_71486

open Real

theorem three_pos_reals_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_eq : a + b + c = a^2 + b^2 + c^2) :
  ((a^2) / (a^2 + b * c) + (b^2) / (b^2 + c * a) + (c^2) / (c^2 + a * b)) ≥ (a + b + c) / 2 :=
by
  sorry

end three_pos_reals_inequality_l71_71486


namespace train_initial_speed_l71_71955

theorem train_initial_speed (x : ℝ) (h : 3 * 25 * (x / V + (2 * x / 20)) = 3 * x) : V = 50 :=
  by
  sorry

end train_initial_speed_l71_71955


namespace bejgli_slices_l71_71151

theorem bejgli_slices (x : ℕ) (hx : x ≤ 58) 
    (h1 : x * (x - 1) * (x - 2) = 3 * (58 - x) * (57 - x) * x) : 
    58 - x = 21 :=
by
  have hpos1 : 0 < x := sorry  -- x should be strictly positive since it's a count
  have hpos2 : 0 < 58 - x := sorry  -- the remaining slices should be strictly positive
  sorry

end bejgli_slices_l71_71151


namespace solve_inequality_l71_71227

theorem solve_inequality (x : ℝ) :
  abs (x + 3) + abs (2 * x - 1) < 7 ↔ -3 ≤ x ∧ x < 5 / 3 :=
by
  sorry

end solve_inequality_l71_71227


namespace whiteboard_ink_cost_l71_71376

/-- 
There are 5 classes: A, B, C, D, E
Class A: 3 whiteboards
Class B: 2 whiteboards
Class C: 4 whiteboards
Class D: 1 whiteboard
Class E: 3 whiteboards
The ink usage per whiteboard in each class:
Class A: 20ml per whiteboard
Class B: 25ml per whiteboard
Class C: 15ml per whiteboard
Class D: 30ml per whiteboard
Class E: 20ml per whiteboard
The cost of ink is 50 cents per ml
-/
def total_cost_in_dollars : ℕ :=
  let ink_usage_A := 3 * 20
  let ink_usage_B := 2 * 25
  let ink_usage_C := 4 * 15
  let ink_usage_D := 1 * 30
  let ink_usage_E := 3 * 20
  let total_ink_usage := ink_usage_A + ink_usage_B + ink_usage_C + ink_usage_D + ink_usage_E
  let total_cost_in_cents := total_ink_usage * 50
  total_cost_in_cents / 100

theorem whiteboard_ink_cost : total_cost_in_dollars = 130 := 
  by 
    sorry -- Proof needs to be implemented

end whiteboard_ink_cost_l71_71376


namespace proof_problem_l71_71750

theorem proof_problem
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2010)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2009)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2010)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2009)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2010)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2009) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = -1 :=
by
  sorry

end proof_problem_l71_71750


namespace solve_k_l71_71911

theorem solve_k (t s : ℤ) : (∃ k m, 8 * k + 4 = 7 * m ∧ k = -4 + 7 * t ∧ m = -4 + 8 * t) →
  (∃ k m, 12 * k - 8 = 7 * m ∧ k = 3 + 7 * s ∧ m = 4 + 12 * s) →
  7 * t - 4 = 7 * s + 3 →
  ∃ k, k = 3 + 7 * s :=
by
  sorry

end solve_k_l71_71911


namespace corey_needs_more_golf_balls_l71_71971

-- Defining the constants based on the conditions
def goal : ℕ := 48
def found_on_saturday : ℕ := 16
def found_on_sunday : ℕ := 18

-- The number of golf balls Corey has found over the weekend
def total_found : ℕ := found_on_saturday + found_on_sunday

-- The number of golf balls Corey still needs to find to reach his goal
def remaining : ℕ := goal - total_found

-- The desired theorem statement
theorem corey_needs_more_golf_balls : remaining = 14 := 
by 
  sorry

end corey_needs_more_golf_balls_l71_71971


namespace least_number_to_subtract_l71_71789

theorem least_number_to_subtract (n : ℕ) : (n = 5) → (5000 - n) % 37 = 0 :=
by sorry

end least_number_to_subtract_l71_71789


namespace largest_of_numbers_l71_71710

theorem largest_of_numbers (a b c d : ℝ) (hₐ : a = 0) (h_b : b = -1) (h_c : c = -2) (h_d : d = Real.sqrt 3) :
  d = Real.sqrt 3 ∧ d > a ∧ d > b ∧ d > c :=
by
  -- Using sorry to skip the proof
  sorry

end largest_of_numbers_l71_71710


namespace midpoint_product_l71_71956

theorem midpoint_product (x y : ℝ) :
  (∃ B : ℝ × ℝ, B = (x, y) ∧ 
  (4, 6) = ( (2 + B.1) / 2, (9 + B.2) / 2 )) → x * y = 18 :=
by
  -- Placeholder for the proof
  sorry

end midpoint_product_l71_71956


namespace rational_solutions_equation_l71_71168

theorem rational_solutions_equation :
  ∃ x : ℚ, (|x - 19| + |x - 93| = 74 ∧ x ∈ {y : ℚ | 19 ≤ y ∨ 19 < y ∧ y < 93 ∨ y ≥ 93}) :=
sorry

end rational_solutions_equation_l71_71168


namespace shooting_game_system_l71_71111

theorem shooting_game_system :
  ∃ x y : ℕ, (x + y = 20 ∧ 3 * x = y) :=
by
  sorry

end shooting_game_system_l71_71111


namespace geometry_problem_l71_71315

-- Definitions for geometrical entities
variable {Point : Type} -- type representing points

variable (Line : Type) -- type representing lines
variable (Plane : Type) -- type representing planes

-- Parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop) 
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Given entities
variables (m : Line) (n : Line) (α : Plane) (β : Plane)

-- Given conditions
axiom condition1 : perpendicular α β
axiom condition2 : perpendicular_line_plane m β
axiom condition3 : ¬ contained_in m α

-- Statement of the problem in Lean 4
theorem geometry_problem : parallel m α :=
by
  -- proof will involve using the axioms and definitions
  sorry

end geometry_problem_l71_71315


namespace solve_exp_equation_l71_71630

theorem solve_exp_equation (e : ℝ) (x : ℝ) (h_e : e = Real.exp 1) :
  e^x + 2 * e^(-x) = 3 ↔ x = 0 ∨ x = Real.log 2 :=
sorry

end solve_exp_equation_l71_71630


namespace total_bills_inserted_l71_71776

theorem total_bills_inserted (x y : ℕ) (h1 : x = 175) (h2 : x + 5 * y = 300) : 
  x + y = 200 :=
by {
  -- Since we focus strictly on the statement per instruction, the proof is omitted
  sorry 
}

end total_bills_inserted_l71_71776


namespace remainder_of_3x_plus_5y_l71_71020

-- Conditions and parameter definitions
def x (k : ℤ) := 13 * k + 7
def y (m : ℤ) := 17 * m + 11

-- Proof statement
theorem remainder_of_3x_plus_5y (k m : ℤ) : (3 * x k + 5 * y m) % 221 = 76 := by
  sorry

end remainder_of_3x_plus_5y_l71_71020


namespace quadratic_function_expression_quadratic_function_inequality_l71_71083

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

theorem quadratic_function_expression (a b c : ℝ) (h₀ : a ≠ 0) 
  (h₁ : ∀ x : ℝ, f (x + 1) - f x = 2 * x) 
  (h₂ : f 0 = 1) : 
  (f x = x^2 - x + 1) := 
by {
  sorry
}

theorem quadratic_function_inequality (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x > 2 * x + m) ↔ m < -1 := 
by {
  sorry
}

end quadratic_function_expression_quadratic_function_inequality_l71_71083


namespace range_of_phi_l71_71528

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + 2 * φ)

theorem range_of_phi :
  ∀ φ : ℝ,
  (0 < φ) ∧ (φ < π / 2) →
  (∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π/6 → g x φ ≤ g (x + π/6) φ) →
  (∃ x : ℝ, -π/6 < x ∧ x < 0 ∧ g x φ = 0) →
  φ ∈ Set.Ioc (π / 4) (π / 3) := 
by
  intros φ h1 h2 h3
  sorry

end range_of_phi_l71_71528


namespace interest_difference_l71_71714

noncomputable def principal := 63100
noncomputable def rate := 10 / 100
noncomputable def time := 2

noncomputable def simple_interest := principal * rate * time
noncomputable def compound_interest := principal * (1 + rate)^time - principal

theorem interest_difference :
  (compound_interest - simple_interest) = 671 := by
  sorry

end interest_difference_l71_71714


namespace system_of_equations_solution_l71_71190

theorem system_of_equations_solution (x y : ℚ) :
  (3 * x^2 + 2 * y^2 + 2 * x + 3 * y = 0 ∧ 4 * x^2 - 3 * y^2 - 3 * x + 4 * y = 0) ↔ 
  ((x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1)) :=
by
  sorry

end system_of_equations_solution_l71_71190


namespace negation_of_exists_l71_71001

theorem negation_of_exists (p : Prop) : 
  (∃ (x₀ : ℝ), x₀ > 0 ∧ |x₀| ≤ 2018) ↔ 
  ¬(∀ (x : ℝ), x > 0 → |x| > 2018) :=
by sorry

end negation_of_exists_l71_71001


namespace volume_of_figure_eq_half_l71_71054

-- Define a cube data structure and its properties
structure Cube where
  edge_length : ℝ
  h_el : edge_length = 1

-- Define a function to calculate volume of the figure
noncomputable def volume_of_figure (c : Cube) : ℝ := sorry

-- Example cube
def example_cube : Cube := { edge_length := 1, h_el := rfl }

-- Theorem statement
theorem volume_of_figure_eq_half (c : Cube) : volume_of_figure c = 1 / 2 := by
  sorry

end volume_of_figure_eq_half_l71_71054


namespace trapezium_hole_perimeter_correct_l71_71930

variable (a b : ℝ)

def trapezium_hole_perimeter (a b : ℝ) : ℝ :=
  6 * a - 3 * b

theorem trapezium_hole_perimeter_correct (a b : ℝ) :
  trapezium_hole_perimeter a b = 6 * a - 3 * b :=
by
  sorry

end trapezium_hole_perimeter_correct_l71_71930


namespace rabbit_clearing_10_square_yards_per_day_l71_71552

noncomputable def area_cleared_by_one_rabbit_per_day (length width : ℕ) (rabbits : ℕ) (days : ℕ) : ℕ :=
  (length * width) / (3 * 3 * rabbits * days)

theorem rabbit_clearing_10_square_yards_per_day :
  area_cleared_by_one_rabbit_per_day 200 900 100 20 = 10 :=
by sorry

end rabbit_clearing_10_square_yards_per_day_l71_71552


namespace cannot_have_1970_minus_signs_in_grid_l71_71584

theorem cannot_have_1970_minus_signs_in_grid :
  ∀ (k l : ℕ), k ≤ 100 → l ≤ 100 → (k+l)*50 - k*l ≠ 985 :=
by
  intros k l hk hl
  sorry

end cannot_have_1970_minus_signs_in_grid_l71_71584


namespace police_officers_on_duty_l71_71283

theorem police_officers_on_duty
  (female_officers : ℕ)
  (percent_female_on_duty : ℚ)
  (total_female_on_duty : ℕ)
  (total_officers_on_duty : ℕ)
  (H1 : female_officers = 1000)
  (H2 : percent_female_on_duty = 15 / 100)
  (H3 : total_female_on_duty = percent_female_on_duty * female_officers)
  (H4 : 2 * total_female_on_duty = total_officers_on_duty) :
  total_officers_on_duty = 300 :=
by
  sorry

end police_officers_on_duty_l71_71283


namespace MoneyDivision_l71_71744

theorem MoneyDivision (w x y z : ℝ)
  (hw : y = 0.5 * w)
  (hx : x = 0.7 * w)
  (hz : z = 0.3 * w)
  (hy : y = 90) :
  w + x + y + z = 450 := by
  sorry

end MoneyDivision_l71_71744


namespace solve_inequality_l71_71975

theorem solve_inequality (a x : ℝ) : 
  (a = 0 ∧ x ≤ -1) ∨ 
  (a > 0 ∧ (x ≥ 2 / a ∨ x ≤ -1)) ∨ 
  (-2 < a ∧ a < 0 ∧ 2 / a ≤ x ∧ x ≤ -1) ∨ 
  (a = -2 ∧ x = -1) ∨
  (a < -2 ∧ -1 ≤ x ∧ x ≤ 2 / a) ↔ 
  a * x ^ 2 + (a - 2) * x - 2 ≥ 0 := 
sorry

end solve_inequality_l71_71975


namespace triangle_inradius_is_2_5_l71_71177

variable (A : ℝ) (p : ℝ) (r : ℝ)

def triangle_has_given_inradius (A p : ℝ) : Prop :=
  A = r * p / 2

theorem triangle_inradius_is_2_5 (h₁ : A = 25) (h₂ : p = 20) :
  triangle_has_given_inradius A p r → r = 2.5 := sorry

end triangle_inradius_is_2_5_l71_71177


namespace kevin_marbles_l71_71275

theorem kevin_marbles (M : ℕ) (h1 : 40 * 3 = 120) (h2 : 4 * M = 320 - 120) :
  M = 50 :=
by {
  sorry
}

end kevin_marbles_l71_71275


namespace Heechul_has_most_books_l71_71570

namespace BookCollection

variables (Heejin Heechul Dongkyun : ℕ)

theorem Heechul_has_most_books (h_h : ℕ) (h_j : ℕ) (d : ℕ) 
  (h_h_eq : h_h = h_j + 2) (d_lt_h_j : d < h_j) : 
  h_h > h_j ∧ h_h > d := 
by
  sorry

end BookCollection

end Heechul_has_most_books_l71_71570


namespace percentage_parents_agree_l71_71853

def total_parents : ℕ := 800
def disagree_parents : ℕ := 640

theorem percentage_parents_agree : 
  ((total_parents - disagree_parents) / total_parents : ℚ) * 100 = 20 := 
by 
  sorry

end percentage_parents_agree_l71_71853


namespace number_of_students_is_four_l71_71296

-- Definitions from the conditions
def average_weight_decrease := 8
def replaced_student_weight := 96
def new_student_weight := 64
def weight_decrease := replaced_student_weight - new_student_weight

-- Goal: Prove that the number of students is 4
theorem number_of_students_is_four
  (average_weight_decrease: ℕ)
  (replaced_student_weight new_student_weight: ℕ)
  (weight_decrease: ℕ) :
  weight_decrease / average_weight_decrease = 4 := 
by
  sorry

end number_of_students_is_four_l71_71296


namespace drying_time_l71_71393

theorem drying_time
  (time_short : ℕ := 10) -- Time to dry a short-haired dog in minutes
  (time_full : ℕ := time_short * 2) -- Time to dry a full-haired dog in minutes, which is twice as long
  (num_short : ℕ := 6) -- Number of short-haired dogs
  (num_full : ℕ := 9) -- Number of full-haired dogs
  : (time_short * num_short + time_full * num_full) / 60 = 4 := 
by
  sorry

end drying_time_l71_71393


namespace equation_solution_l71_71797

noncomputable def solve_equation (x : ℝ) : Prop :=
  (1/4) * x^(1/2 * Real.log x / Real.log 2) = 2^(1/4 * (Real.log x / Real.log 2)^2)

theorem equation_solution (x : ℝ) (hx : 0 < x) : solve_equation x → (x = 2^(2*Real.sqrt 2) ∨ x = 2^(-2*Real.sqrt 2)) :=
  by
  intro h
  sorry

end equation_solution_l71_71797


namespace solve_for_t_l71_71187

theorem solve_for_t (s t : ℚ) (h1 : 8 * s + 6 * t = 160) (h2 : s = t + 3) : t = 68 / 7 :=
by
  sorry

end solve_for_t_l71_71187


namespace max_profit_l71_71637

noncomputable def profit (x : ℕ) : ℝ := -0.15 * (x : ℝ)^2 + 3.06 * (x : ℝ) + 30

theorem max_profit :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 15 ∧ profit x = 45.6 ∧ ∀ y : ℕ, 0 ≤ y ∧ y ≤ 15 → profit y ≤ profit x :=
by
  sorry

end max_profit_l71_71637


namespace berries_from_fourth_bush_l71_71452

def number_of_berries (n : ℕ) : ℕ :=
  match n with
  | 1 => 3
  | 2 => 4
  | 3 => 7
  | 5 => 19
  | _ => sorry  -- Assume the given pattern

theorem berries_from_fourth_bush : number_of_berries 4 = 12 :=
by sorry

end berries_from_fourth_bush_l71_71452


namespace right_triangle_smaller_angle_l71_71676

theorem right_triangle_smaller_angle (x : ℝ) (h_right_triangle : 0 < x ∧ x < 90)
  (h_double_angle : ∃ y : ℝ, y = 2 * x)
  (h_angle_sum : x + 2 * x = 90) :
  x = 30 :=
  sorry

end right_triangle_smaller_angle_l71_71676


namespace max_distance_circle_to_line_l71_71868

-- Definitions for the circle and line
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 2 * y = 0
def line_eq (x y : ℝ) : Prop := x + y + 2 = 0

-- Proof statement
theorem max_distance_circle_to_line 
  (x y : ℝ)
  (h_circ : circle_eq x y)
  (h_line : ∀ (x y : ℝ), line_eq x y → true) :
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 :=
sorry

end max_distance_circle_to_line_l71_71868


namespace smallest_number_of_coins_l71_71317

theorem smallest_number_of_coins :
  ∃ pennies nickels dimes quarters half_dollars : ℕ,
    pennies + nickels + dimes + quarters + half_dollars = 6 ∧
    (∀ amount : ℕ, amount < 100 →
      ∃ p n d q h : ℕ,
        p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧ q ≤ quarters ∧ h ≤ half_dollars ∧
        1 * p + 5 * n + 10 * d + 25 * q + 50 * h = amount) :=
sorry

end smallest_number_of_coins_l71_71317


namespace max_problems_missed_to_pass_l71_71197

theorem max_problems_missed_to_pass (total_problems : ℕ) (min_percentage : ℚ) 
  (h_total_problems : total_problems = 40) 
  (h_min_percentage : min_percentage = 0.85) : 
  ∃ max_missed : ℕ, max_missed = total_problems - ⌈total_problems * min_percentage⌉₊ ∧ max_missed = 6 :=
by
  sorry

end max_problems_missed_to_pass_l71_71197


namespace original_visual_range_l71_71901

theorem original_visual_range
  (V : ℝ)
  (h1 : 2.5 * V = 150) :
  V = 60 :=
by
  sorry

end original_visual_range_l71_71901


namespace two_digit_number_conditions_l71_71665

-- Definitions for two-digit number and its conditions
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

-- The proof problem statement in Lean 4
theorem two_digit_number_conditions (N : ℕ) (c d : ℕ) :
  is_two_digit_number N ∧ N = 10 * c + d ∧ N' = N + 7 ∧ 
  N = 6 * sum_of_digits (N + 7) →
  N = 24 ∨ N = 78 :=
by
  sorry

end two_digit_number_conditions_l71_71665


namespace find_subtracted_value_l71_71885

theorem find_subtracted_value (N V : ℕ) (hN : N = 12) (h : 4 * N - V = 9 * (N - 7)) : V = 3 :=
by
  sorry

end find_subtracted_value_l71_71885


namespace Zilla_savings_l71_71389

-- Define the conditions
def rent_expense (E : ℝ) (R : ℝ) := R = 0.07 * E
def other_expenses (E : ℝ) := 0.5 * E
def amount_saved (E : ℝ) (R : ℝ) (S : ℝ) := S = E - (R + other_expenses E)

-- Define the main problem statement
theorem Zilla_savings (E R S: ℝ) 
    (hR : rent_expense E R)
    (hR_val : R = 133)
    (hS : amount_saved E R S) : 
    S = 817 := by
  sorry

end Zilla_savings_l71_71389


namespace circle_equation_l71_71967

theorem circle_equation :
  ∃ (r : ℝ), ∀ (x y : ℝ), (x - 3) ^ 2 + (y - 1) ^ 2 = r ↔ (x = 0 ∧ y = 0) → ((x - 3) ^ 2 + (y - 1) ^ 2 = 10) :=
by
  sorry

end circle_equation_l71_71967


namespace infinite_solutions_eq_a_l71_71387

variable (a x y: ℝ)

-- Define the two equations
def eq1 : Prop := a * x + y - 1 = 0
def eq2 : Prop := 4 * x + a * y - 2 = 0

theorem infinite_solutions_eq_a (h : ∃ x y, eq1 a x y ∧ eq2 a x y) :
  a = 2 := 
sorry

end infinite_solutions_eq_a_l71_71387


namespace valid_combinations_count_l71_71658

theorem valid_combinations_count : 
  let wrapping_paper_count := 10
  let ribbon_count := 3
  let gift_card_count := 5
  let invalid_combinations := 1 -- red ribbon with birthday card
  let total_combinations := wrapping_paper_count * ribbon_count * gift_card_count
  total_combinations - invalid_combinations = 149 := 
by 
  sorry

end valid_combinations_count_l71_71658


namespace range_of_m_l71_71367

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (6 - 3 * (x + 1) < x - 9) ∧ (x - m > -1) ↔ (x > 3)) → (m ≤ 4) :=
by
  sorry

end range_of_m_l71_71367


namespace math_problem_l71_71675

theorem math_problem (c d : ℝ) (hc : c^2 - 6 * c + 15 = 27) (hd : d^2 - 6 * d + 15 = 27) (h_cd : c ≥ d) : 
  3 * c + 2 * d = 15 + Real.sqrt 21 :=
by
  sorry

end math_problem_l71_71675


namespace parallel_lines_k_value_l71_71269

-- Define the lines and the condition of parallelism
def line1 (x y : ℝ) := x + 2 * y - 1 = 0
def line2 (k x y : ℝ) := k * x - y = 0

-- Define the parallelism condition
def lines_parallel (k : ℝ) := (1 / k) = (2 / -1)

-- Prove that given the parallelism condition, k equals -1/2
theorem parallel_lines_k_value (k : ℝ) (h : lines_parallel k) : k = (-1 / 2) :=
by
  sorry

end parallel_lines_k_value_l71_71269


namespace max_value_sin_sin2x_l71_71014

open Real

/-- Given x is an acute angle, find the maximum value of the function y = sin x * sin (2 * x). -/
theorem max_value_sin_sin2x (x : ℝ) (hx : 0 < x ∧ x < π / 2) :
    ∃ max_y : ℝ, ∀ y : ℝ, y = sin x * sin (2 * x) -> y ≤ max_y ∧ max_y = 4 * sqrt 3 / 9 :=
by
  -- To be completed
  sorry

end max_value_sin_sin2x_l71_71014


namespace range_of_a_l71_71555

theorem range_of_a (a : ℝ) : 1 ∉ {x : ℝ | x^2 - 2 * x + a > 0} → a ≤ 1 :=
by
  sorry

end range_of_a_l71_71555


namespace expand_polynomial_l71_71342

theorem expand_polynomial (x : ℝ) : (5 * x + 3) * (6 * x ^ 2 + 2) = 30 * x ^ 3 + 18 * x ^ 2 + 10 * x + 6 :=
by
  sorry

end expand_polynomial_l71_71342


namespace max_value_x_plus_2y_l71_71700

theorem max_value_x_plus_2y (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x * y = 4) :
  x + 2 * y ≤ 4 :=
sorry

end max_value_x_plus_2y_l71_71700


namespace hyewon_painted_colors_l71_71816

def pentagonal_prism := 
  let num_rectangular_faces := 5 
  let num_pentagonal_faces := 2
  num_rectangular_faces + num_pentagonal_faces

theorem hyewon_painted_colors : pentagonal_prism = 7 := 
by
  sorry

end hyewon_painted_colors_l71_71816


namespace betty_total_oranges_l71_71858

-- Definitions for the given conditions
def boxes : ℝ := 3.0
def oranges_per_box : ℝ := 24

-- Theorem statement to prove the correct answer to the problem
theorem betty_total_oranges : boxes * oranges_per_box = 72 := by
  sorry

end betty_total_oranges_l71_71858


namespace necessary_and_sufficient_condition_l71_71931

theorem necessary_and_sufficient_condition (t : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
    (∀ n, S n = n^2 + 5*n + t) →
    (t = 0 ↔ (∀ n, a n = 2*n + 4 ∧ (n > 0 → a n = S n - S (n - 1)))) :=
by
  sorry

end necessary_and_sufficient_condition_l71_71931


namespace common_area_of_equilateral_triangles_in_unit_square_l71_71476

theorem common_area_of_equilateral_triangles_in_unit_square
  (unit_square_side_length : ℝ)
  (triangle_side_length : ℝ)
  (common_area : ℝ)
  (h_unit_square : unit_square_side_length = 1)
  (h_triangle_side : triangle_side_length = 1) :
  common_area = -1 :=
by
  sorry

end common_area_of_equilateral_triangles_in_unit_square_l71_71476


namespace circle_center_and_radius_l71_71694

-- Define the given conditions
variable (a : ℝ) (h : a^2 = a + 2 ∧ a ≠ 0)

-- Define the equation
noncomputable def circle_equation (x y : ℝ) : ℝ := a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a

-- Lean definition to represent the problem
theorem circle_center_and_radius :
  (∃a : ℝ, a ≠ 0 ∧ a^2 = a + 2 ∧
    (∃x y : ℝ, circle_equation a x y = 0) ∧
    ((a = -1) → ((∃x y : ℝ, (x + 2)^2 + (y + 4)^2 = 25) ∧
                 (center_x = -2) ∧ (center_y = -4) ∧ (radius = 5)))) :=
by
  sorry

end circle_center_and_radius_l71_71694


namespace solve_for_x_l71_71851

theorem solve_for_x : (42 / (7 - 3 / 7) = 147 / 23) :=
by
  sorry

end solve_for_x_l71_71851


namespace negation_of_exists_is_forall_l71_71804

theorem negation_of_exists_is_forall :
  (¬ ∃ x : ℝ, x^3 + 1 = 0) ↔ ∀ x : ℝ, x^3 + 1 ≠ 0 :=
by 
  sorry

end negation_of_exists_is_forall_l71_71804


namespace tickets_needed_l71_71529

variable (rides_rollercoaster : ℕ) (tickets_rollercoaster : ℕ)
variable (rides_catapult : ℕ) (tickets_catapult : ℕ)
variable (rides_ferris_wheel : ℕ) (tickets_ferris_wheel : ℕ)

theorem tickets_needed 
    (hRides_rollercoaster : rides_rollercoaster = 3)
    (hTickets_rollercoaster : tickets_rollercoaster = 4)
    (hRides_catapult : rides_catapult = 2)
    (hTickets_catapult : tickets_catapult = 4)
    (hRides_ferris_wheel : rides_ferris_wheel = 1)
    (hTickets_ferris_wheel : tickets_ferris_wheel = 1) :
    rides_rollercoaster * tickets_rollercoaster +
    rides_catapult * tickets_catapult +
    rides_ferris_wheel * tickets_ferris_wheel = 21 :=
by {
    sorry
}

end tickets_needed_l71_71529


namespace find_a_l71_71282

-- Conditions as definitions:
variable (a : ℝ) (b : ℝ)
variable (A : ℝ × ℝ := (0, 0)) (B : ℝ × ℝ := (a, 0)) (C : ℝ × ℝ := (0, b))
noncomputable def area (a b : ℝ) : ℝ := (1 / 2) * a * b

-- Given conditions:
axiom h1 : b = 4
axiom h2 : area a b = 28
axiom h3 : a > 0

-- The proof goal:
theorem find_a : a = 14 := by
  -- proof omitted
  sorry

end find_a_l71_71282


namespace locus_of_M_equation_of_l_l71_71828
open Real

-- Step 1: Define the given circles
def circle_F1 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def circle_F2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 36

-- Step 2: Define the condition of tangency for the moving circle M
def external_tangent_F1 (cx cy r : ℝ) : Prop := (cx + 2)^2 + cy^2 = (2 + r)^2
def internal_tangent_F2 (cx cy r : ℝ) : Prop := (cx - 2)^2 + cy^2 = (6 - r)^2

-- Step 4: Prove the locus C is an ellipse with the equation excluding x = -4
theorem locus_of_M (cx cy : ℝ) : 
  (∃ r : ℝ, external_tangent_F1 cx cy r ∧ internal_tangent_F2 cx cy r) ↔
  (cx ≠ -4 ∧ (cx^2) / 16 + (cy^2) / 12 = 1) :=
sorry

-- Step 5: Define the conditions for the midpoint of segment AB
def midpoint_Q (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = -1

-- Step 6: Prove the equation of line l
theorem equation_of_l (x1 y1 x2 y2 : ℝ) (h1 : midpoint_Q x1 y1 x2 y2) 
  (h2 : (x1^2 / 16 + y1^2 / 12 = 1) ∧ (x2^2 / 16 + y2^2 / 12 = 1)) :
  3 * (x1 - x2) - 2 * (y1 - y2) = 8 :=
sorry

end locus_of_M_equation_of_l_l71_71828


namespace time_required_painting_rooms_l71_71991

-- Definitions based on the conditions
def alice_rate := 1 / 4
def bob_rate := 1 / 6
def charlie_rate := 1 / 8
def combined_rate := 13 / 24
def required_time : ℚ := 74 / 13

-- Proof problem statement
theorem time_required_painting_rooms (t : ℚ) :
  (combined_rate) * (t - 2) = 2 ↔ t = required_time :=
by
  sorry

end time_required_painting_rooms_l71_71991


namespace common_chord_length_of_two_circles_l71_71524

noncomputable def common_chord_length (r : ℝ) : ℝ :=
  if r = 10 then 10 * Real.sqrt 3 else sorry

theorem common_chord_length_of_two_circles (r : ℝ) (h : r = 10) :
  common_chord_length r = 10 * Real.sqrt 3 :=
by
  rw [h]
  sorry

end common_chord_length_of_two_circles_l71_71524


namespace roots_order_l71_71167

theorem roots_order {a b m n : ℝ} (h1 : m < n) (h2 : a < b)
  (hm : 1 - (m - a) * (m - b) = 0) (hn : 1 - (n - a) * (n - b) = 0) :
  m < a ∧ a < b ∧ b < n :=
sorry

end roots_order_l71_71167


namespace log3_of_7_eq_ab_l71_71400

noncomputable def log3_of_2_eq_a (a : ℝ) : Prop := Real.log 2 / Real.log 3 = a
noncomputable def log2_of_7_eq_b (b : ℝ) : Prop := Real.log 7 / Real.log 2 = b

theorem log3_of_7_eq_ab (a b : ℝ) (h1 : log3_of_2_eq_a a) (h2 : log2_of_7_eq_b b) :
  Real.log 7 / Real.log 3 = a * b :=
sorry

end log3_of_7_eq_ab_l71_71400


namespace intersection_points_count_l71_71264

theorem intersection_points_count (B : ℝ) (hB : 0 < B) :
  ∃ p : ℕ, p = 4 ∧ (∀ x y : ℝ, (y = B * x^2 ∧ y^2 + 4 * y - 2 = x^2 + 5 * y) ↔ p = 4) := by
sorry

end intersection_points_count_l71_71264


namespace discount_percentage_l71_71048

/-
  A retailer buys 80 pens at the market price of 36 pens from a wholesaler.
  He sells these pens giving a certain discount and his profit is 120%.
  What is the discount percentage he gave on the pens?
-/
theorem discount_percentage
  (P : ℝ)
  (CP SP D DP : ℝ) 
  (h1 : CP = 36 * P)
  (h2 : SP = 2.2 * CP)
  (h3 : D = P - (SP / 80))
  (h4 : DP = (D / P) * 100) :
  DP = 1 := 
sorry

end discount_percentage_l71_71048


namespace travel_remaining_distance_l71_71357

-- Definitions of given conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Define the distances each person traveled
def amoli_distance := amoli_speed * amoli_time
def anayet_distance := anayet_speed * anayet_time

-- Define the total distance covered
def total_covered := amoli_distance + anayet_distance

-- Define the remaining distance
def remaining_distance := total_distance - total_covered

-- Prove the remaining distance is 121 miles
theorem travel_remaining_distance : remaining_distance = 121 := by
  sorry

end travel_remaining_distance_l71_71357


namespace range_of_a_l71_71067

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x - a ≤ -3) → a ∈ Set.Iic (-6) ∪ Set.Ici 2 :=
by
  intro h
  sorry

end range_of_a_l71_71067


namespace isosceles_obtuse_triangle_l71_71811

theorem isosceles_obtuse_triangle (A B C : ℝ) (h_isosceles: A = B)
  (h_obtuse: A + B + C = 180) 
  (h_max_angle: C = 157.5): A = 11.25 :=
by
  sorry

end isosceles_obtuse_triangle_l71_71811


namespace bjorn_cannot_prevent_vakha_l71_71309

-- Define the primary settings and objects involved
def n_points : ℕ := 99
inductive Color
| red 
| blue 

structure GameState :=
  (turn : ℕ)
  (points : Fin n_points → Option Color)

-- Define the valid states of the game where turn must be within the range of points
def valid_state (s : GameState) : Prop :=
  s.turn ≤ n_points ∧ ∀ p, s.points p ≠ none

-- Define what it means for an equilateral triangle to be monochromatically colored
def monochromatic_equilateral_triangle (state : GameState) : Prop :=
  ∃ (p1 p2 p3 : Fin n_points), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    (p1.val + (n_points/3) % n_points) = p2.val ∧
    (p2.val + (n_points/3) % n_points) = p3.val ∧
    (p3.val + (n_points/3) % n_points) = p1.val ∧
    (state.points p1 = state.points p2) ∧ 
    (state.points p2 = state.points p3)

-- Vakha's winning condition
def vakha_wins (state : GameState) : Prop := 
  monochromatic_equilateral_triangle state

-- Bjorn's winning condition prevents Vakha from winning
def bjorn_can_prevent_vakha (initial_state : GameState) : Prop :=
  ¬ vakha_wins initial_state

-- Main theorem stating Bjorn cannot prevent Vakha from winning
theorem bjorn_cannot_prevent_vakha : ∀ (initial_state : GameState),
  valid_state initial_state → ¬ bjorn_can_prevent_vakha initial_state :=
sorry

end bjorn_cannot_prevent_vakha_l71_71309


namespace mary_regular_hours_l71_71164

theorem mary_regular_hours (x y : ℕ) (h1 : 8 * x + 10 * y = 560) (h2 : x + y = 60) : x = 20 :=
by
  sorry

end mary_regular_hours_l71_71164


namespace pump_without_leak_time_l71_71907

theorem pump_without_leak_time :
  ∃ T : ℝ, (1/T - 1/5.999999999999999 = 1/3) ∧ T = 2 :=
by 
  sorry

end pump_without_leak_time_l71_71907


namespace find_m_over_n_l71_71231

variable (a b : ℝ × ℝ)
variable (m n : ℝ)
variable (n_nonzero : n ≠ 0)

axiom a_def : a = (1, 2)
axiom b_def : b = (-2, 3)
axiom collinear : ∃ k : ℝ, m • a - n • b = k • (a + 2 • b)

theorem find_m_over_n : m / n = -1 / 2 := by
  sorry

end find_m_over_n_l71_71231


namespace yen_checking_account_l71_71643

theorem yen_checking_account (savings : ℕ) (total : ℕ) (checking : ℕ) (h1 : savings = 3485) (h2 : total = 9844) (h3 : checking = total - savings) :
  checking = 6359 :=
by
  rw [h1, h2] at h3
  exact h3

end yen_checking_account_l71_71643


namespace maximum_sum_of_O_and_square_l71_71745

theorem maximum_sum_of_O_and_square 
(O square : ℕ) (h1 : (O > 0) ∧ (square > 0)) 
(h2 : (O : ℚ) / 11 < (7 : ℚ) / (square))
(h3 : (7 : ℚ) / (square) < (4 : ℚ) / 5) : 
O + square = 18 :=
sorry

end maximum_sum_of_O_and_square_l71_71745


namespace domain_of_f_l71_71100

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (2 - x)) + Real.log (x+1)

theorem domain_of_f : {x : ℝ | (2 - x) > 0 ∧ (x + 1) > 0} = {x : ℝ | -1 < x ∧ x < 2} := 
by
  ext x
  simp
  sorry

end domain_of_f_l71_71100


namespace total_pennies_donated_l71_71098

def cassandra_pennies : ℕ := 5000
def james_pennies : ℕ := cassandra_pennies - 276
def total_pennies : ℕ := cassandra_pennies + james_pennies

theorem total_pennies_donated : total_pennies = 9724 := by
  sorry

end total_pennies_donated_l71_71098


namespace least_k_for_168_l71_71826

theorem least_k_for_168 (k : ℕ) :
  (k^3 % 168 = 0) ↔ k ≥ 42 :=
sorry

end least_k_for_168_l71_71826


namespace slope_of_line_l71_71359

theorem slope_of_line (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (1, 2)) (h2 : (x2, y2) = (4, 8)) :
  (y2 - y1) / (x2 - x1) = 2 := 
by
  sorry

end slope_of_line_l71_71359


namespace multiplicative_inverse_exists_and_is_correct_l71_71928

theorem multiplicative_inverse_exists_and_is_correct :
  ∃ N : ℤ, N > 0 ∧ (123456 * 171717) * N % 1000003 = 1 :=
sorry

end multiplicative_inverse_exists_and_is_correct_l71_71928


namespace candy_problem_l71_71402

-- Define the given conditions
def numberOfStudents : Nat := 43
def piecesOfCandyPerStudent : Nat := 8

-- Formulate the problem statement
theorem candy_problem : numberOfStudents * piecesOfCandyPerStudent = 344 := by
  sorry

end candy_problem_l71_71402


namespace carlton_outfits_l71_71761

theorem carlton_outfits (button_up_shirts sweater_vests : ℕ) 
  (h1 : sweater_vests = 2 * button_up_shirts)
  (h2 : button_up_shirts = 3) :
  sweater_vests * button_up_shirts = 18 :=
by
  sorry

end carlton_outfits_l71_71761


namespace optimal_ticket_price_l71_71147

noncomputable def revenue (x : ℕ) : ℤ :=
  if x < 6 then -5750
  else if x ≤ 10 then 1000 * (x : ℤ) - 5750
  else if x ≤ 38 then -30 * (x : ℤ)^2 + 1300 * (x : ℤ) - 5750
  else -5750

theorem optimal_ticket_price :
  revenue 22 = 8330 :=
by
  sorry

end optimal_ticket_price_l71_71147


namespace intersecting_diagonals_probability_l71_71442

def probability_of_intersecting_diagonals_inside_dodecagon : ℚ :=
  let total_points := 12
  let total_segments := (total_points.choose 2)
  let sides := 12
  let diagonals := total_segments - sides
  let ways_to_choose_2_diagonals := (diagonals.choose 2)
  let ways_to_choose_4_points := (total_points.choose 4)
  let probability := (ways_to_choose_4_points : ℚ) / (ways_to_choose_2_diagonals : ℚ)
  probability

theorem intersecting_diagonals_probability (H : probability_of_intersecting_diagonals_inside_dodecagon = 165 / 477) : 
  probability_of_intersecting_diagonals_inside_dodecagon = 165 / 477 :=
  by
  sorry

end intersecting_diagonals_probability_l71_71442


namespace tangent_expression_l71_71334

theorem tangent_expression :
  (Real.tan (10 * Real.pi / 180) + Real.tan (50 * Real.pi / 180) + Real.tan (120 * Real.pi / 180))
  / (Real.tan (10 * Real.pi / 180) * Real.tan (50 * Real.pi / 180)) = -Real.sqrt 3 := by
  sorry

end tangent_expression_l71_71334


namespace price_of_each_rose_l71_71189

def number_of_roses_started (roses : ℕ) : Prop := roses = 9
def number_of_roses_left (roses : ℕ) : Prop := roses = 4
def amount_earned (money : ℕ) : Prop := money = 35
def selling_price_per_rose (price : ℕ) : Prop := price = 7

theorem price_of_each_rose 
  (initial_roses sold_roses left_roses total_money price_per_rose : ℕ)
  (h1 : number_of_roses_started initial_roses)
  (h2 : number_of_roses_left left_roses)
  (h3 : amount_earned total_money)
  (h4 : initial_roses - left_roses = sold_roses)
  (h5 : total_money / sold_roses = price_per_rose) :
  selling_price_per_rose price_per_rose := 
by
  sorry

end price_of_each_rose_l71_71189


namespace find_base_l71_71208

noncomputable def log_base (a x : ℝ) := Real.log x / Real.log a

theorem find_base (a : ℝ) (h : 1 < a) :
  (log_base a (2 * a) - log_base a a = 1 / 2) → a = 4 :=
by
  -- skipping the proof
  sorry

end find_base_l71_71208


namespace geometric_sequence_property_l71_71898

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n m : ℕ), a (n + 1) * a (m + 1) = a n * a m

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
(h_condition : a 2 * a 4 = 1/2) :
  a 1 * a 3 ^ 2 * a 5 = 1/4 :=
by
  sorry

end geometric_sequence_property_l71_71898


namespace correct_operation_l71_71060

-- Define that m and n are elements of an arbitrary commutative ring
variables {R : Type*} [CommRing R] (m n : R)

theorem correct_operation : (m * n) ^ 2 = m ^ 2 * n ^ 2 := by
  sorry

end correct_operation_l71_71060


namespace correct_difference_is_nine_l71_71875

-- Define the conditions
def misunderstood_number : ℕ := 35
def actual_number : ℕ := 53
def incorrect_difference : ℕ := 27

-- Define the two-digit number based on Yoongi's incorrect calculation
def original_number : ℕ := misunderstood_number + incorrect_difference

-- State the theorem
theorem correct_difference_is_nine : (original_number - actual_number) = 9 :=
by
  -- Proof steps go here
  sorry

end correct_difference_is_nine_l71_71875


namespace smallest_n_proof_l71_71015

-- Given conditions and the problem statement in Lean 4
noncomputable def smallest_n : ℕ := 11

theorem smallest_n_proof :
  ∃ (x y z : ℕ), (x > 0 ∧ y > 0 ∧ z > 0) ∧ (smallest_n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 11) :=
sorry

end smallest_n_proof_l71_71015


namespace triangle_area_is_18_l71_71498

noncomputable def area_of_triangle (y_8 y_2_2x y_2_minus_2x : ℝ) : ℝ :=
  let intersect1 : ℝ × ℝ := (3, 8)
  let intersect2 : ℝ × ℝ := (-3, 8)
  let intersect3 : ℝ × ℝ := (0, 2)
  let base := 3 - -3
  let height := 8 - 2
  (1 / 2 ) * base * height

theorem triangle_area_is_18 : 
  area_of_triangle (8) (2 + 2 * x) (2 - 2 * x) = 18 := 
  by
    sorry

end triangle_area_is_18_l71_71498


namespace sam_total_money_spent_l71_71447

def value_of_pennies (n : ℕ) : ℝ := n * 0.01
def value_of_nickels (n : ℕ) : ℝ := n * 0.05
def value_of_dimes (n : ℕ) : ℝ := n * 0.10
def value_of_quarters (n : ℕ) : ℝ := n * 0.25

def total_money_spent : ℝ :=
  (value_of_pennies 5 + value_of_nickels 3) +  -- Monday
  (value_of_dimes 8 + value_of_quarters 4) +   -- Tuesday
  (value_of_nickels 7 + value_of_dimes 10 + value_of_quarters 2) +  -- Wednesday
  (value_of_pennies 20 + value_of_nickels 15 + value_of_dimes 12 + value_of_quarters 6) +  -- Thursday
  (value_of_pennies 45 + value_of_nickels 20 + value_of_dimes 25 + value_of_quarters 10)  -- Friday

theorem sam_total_money_spent : total_money_spent = 14.05 :=
by
  sorry

end sam_total_money_spent_l71_71447


namespace total_wire_length_l71_71069

theorem total_wire_length
  (A B C D E : ℕ)
  (hA : A = 16)
  (h_ratio : 4 * A = 5 * B ∧ 4 * A = 7 * C ∧ 4 * A = 3 * D ∧ 4 * A = 2 * E)
  (hC : C = B + 8) :
  (A + B + C + D + E) = 84 := 
sorry

end total_wire_length_l71_71069


namespace simplify_expression_is_3_l71_71223

noncomputable def simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) : ℝ :=
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)

theorem simplify_expression_is_3 (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) :
  simplify_expression x y z hx hy hz h = 3 :=
  sorry

end simplify_expression_is_3_l71_71223


namespace interval_of_expression_l71_71580

theorem interval_of_expression (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
  1 < (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) ∧ 
  (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) < 2 :=
by sorry

end interval_of_expression_l71_71580


namespace min_value_of_function_l71_71455

theorem min_value_of_function (x : ℝ) (hx : x > 3) :
  (x + (1 / (x - 3))) ≥ 5 :=
sorry

end min_value_of_function_l71_71455


namespace T_five_three_l71_71738

def T (a b : ℤ) : ℤ := 4 * a + 6 * b + 2

theorem T_five_three : T 5 3 = 40 := by
  sorry

end T_five_three_l71_71738


namespace intersection_of_sets_l71_71228

def SetA : Set ℝ := { x | |x| ≤ 1 }
def SetB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_of_sets : (SetA ∩ SetB) = { x | 0 ≤ x ∧ x ≤ 1 } := 
by
  sorry

end intersection_of_sets_l71_71228


namespace test_question_total_l71_71479

theorem test_question_total
  (total_points : ℕ)
  (points_2q : ℕ)
  (points_4q : ℕ)
  (num_2q : ℕ)
  (num_4q : ℕ)
  (H1 : total_points = 100)
  (H2 : points_2q = 2)
  (H3 : points_4q = 4)
  (H4 : num_2q = 30)
  (H5 : total_points = num_2q * points_2q + num_4q * points_4q) :
  num_2q + num_4q = 40 := 
sorry

end test_question_total_l71_71479


namespace ball_first_less_than_25_cm_l71_71954

theorem ball_first_less_than_25_cm (n : ℕ) :
  ∀ n, (200 : ℝ) * (3 / 4) ^ n < 25 ↔ n ≥ 6 := by sorry

end ball_first_less_than_25_cm_l71_71954


namespace simplify_and_evaluate_l71_71578

theorem simplify_and_evaluate : 
  ∀ (x y : ℚ), x = 1 / 2 → y = 2 / 3 →
  ((x - 2 * y)^2 + (x - 2 * y) * (x + 2 * y) - 3 * x * (2 * x - y)) / (2 * x) = -4 / 3 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end simplify_and_evaluate_l71_71578


namespace gwendolyn_reading_time_l71_71712

/--
Gwendolyn can read 200 sentences in 1 hour. 
Each paragraph has 10 sentences. 
There are 20 paragraphs per page. 
The book has 50 pages. 
--/
theorem gwendolyn_reading_time : 
  let sentences_per_hour := 200
  let sentences_per_paragraph := 10
  let paragraphs_per_page := 20
  let pages := 50
  let sentences_per_page := sentences_per_paragraph * paragraphs_per_page
  let total_sentences := sentences_per_page * pages
  (total_sentences / sentences_per_hour) = 50 := 
by
  let sentences_per_hour : ℕ := 200
  let sentences_per_paragraph : ℕ := 10
  let paragraphs_per_page : ℕ := 20
  let pages : ℕ := 50
  let sentences_per_page : ℕ := sentences_per_paragraph * paragraphs_per_page
  let total_sentences : ℕ := sentences_per_page * pages
  have h : (total_sentences / sentences_per_hour) = 50 := by sorry
  exact h

end gwendolyn_reading_time_l71_71712


namespace compute_product_l71_71657

variable (x1 y1 x2 y2 x3 y3 : ℝ)

def condition1 (x y : ℝ) : Prop := x^3 - 3 * x * y^2 = 2010
def condition2 (x y : ℝ) : Prop := y^3 - 3 * x^2 * y = 2000

theorem compute_product (h1 : condition1 x1 y1) (h2 : condition2 x1 y1)
    (h3 : condition1 x2 y2) (h4 : condition2 x2 y2)
    (h5 : condition1 x3 y3) (h6 : condition2 x3 y3) :
    (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 100 := 
    sorry

end compute_product_l71_71657


namespace average_hours_per_day_l71_71974

theorem average_hours_per_day (h : ℝ) :
  (3 * h * 12 + 2 * h * 9 = 108) → h = 2 :=
by 
  intro h_condition
  sorry

end average_hours_per_day_l71_71974


namespace dealer_sold_BMWs_l71_71594

theorem dealer_sold_BMWs (total_cars : ℕ) (ford_pct toyota_pct nissan_pct bmw_pct : ℝ)
  (h_total_cars : total_cars = 300)
  (h_ford_pct : ford_pct = 0.1)
  (h_toyota_pct : toyota_pct = 0.2)
  (h_nissan_pct : nissan_pct = 0.3)
  (h_bmw_pct : bmw_pct = 1 - (ford_pct + toyota_pct + nissan_pct)) :
  total_cars * bmw_pct = 120 := by
  sorry

end dealer_sold_BMWs_l71_71594


namespace calculate_weight_5_moles_Al2O3_l71_71947

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def molecular_weight_Al2O3 : ℝ := (2 * atomic_weight_Al) + (3 * atomic_weight_O)
def moles_Al2O3 : ℝ := 5
def weight_5_moles_Al2O3 : ℝ := moles_Al2O3 * molecular_weight_Al2O3

theorem calculate_weight_5_moles_Al2O3 :
  weight_5_moles_Al2O3 = 509.8 :=
by sorry

end calculate_weight_5_moles_Al2O3_l71_71947


namespace number_of_subjects_l71_71449

variable (P C M : ℝ)

-- Given conditions
def conditions (P C M : ℝ) : Prop :=
  (P + C + M) / 3 = 75 ∧
  (P + M) / 2 = 90 ∧
  (P + C) / 2 = 70 ∧
  P = 95

-- Proposition with given conditions and the conclusion
theorem number_of_subjects (P C M : ℝ) (h : conditions P C M) : 
  (∃ n : ℕ, n = 3) :=
by
  sorry

end number_of_subjects_l71_71449


namespace total_area_rectangle_l71_71052

theorem total_area_rectangle (BF CF : ℕ) (A1 A2 x : ℕ) (h1 : BF = 3 * CF) (h2 : A1 = 3 * A2) (h3 : 2 * x = 96) (h4 : 48 = x) (h5 : A1 = 3 * 48) (h6 : A2 = 48) : A1 + A2 = 192 :=
  by sorry

end total_area_rectangle_l71_71052


namespace solve_for_k_l71_71422

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k-1) * x + 2

theorem solve_for_k (k : ℝ) :
  (∀ x : ℝ, f k x = f k (-x)) ↔ k = 1 :=
by
  sorry

end solve_for_k_l71_71422


namespace neither_sufficient_nor_necessary_l71_71006

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem neither_sufficient_nor_necessary (a : ℝ) :
  (a ∈ M → a ∈ N) = false ∧ (a ∈ N → a ∈ M) = false := by
  sorry

end neither_sufficient_nor_necessary_l71_71006


namespace composite_product_division_l71_71138

noncomputable def firstFiveCompositeProduct : ℕ := 4 * 6 * 8 * 9 * 10
noncomputable def nextFiveCompositeProduct : ℕ := 12 * 14 * 15 * 16 * 18

theorem composite_product_division : firstFiveCompositeProduct / nextFiveCompositeProduct = 1 / 42 := by
  sorry

end composite_product_division_l71_71138


namespace find_x_from_expression_l71_71123

theorem find_x_from_expression
  (y : ℚ)
  (h1 : y = -3/2)
  (h2 : -2 * (x : ℚ) - y^2 = 0.25) : 
  x = -5/4 := 
by 
  sorry

end find_x_from_expression_l71_71123


namespace polynomial_roots_l71_71503

theorem polynomial_roots (d e : ℤ) :
  (∀ r, r^2 - 2 * r - 1 = 0 → r^5 - d * r - e = 0) ↔ (d = 29 ∧ e = 12) := by
  sorry

end polynomial_roots_l71_71503


namespace and_15_and_l71_71835

def x_and (x : ℝ) : ℝ := 8 - x
def and_x (x : ℝ) : ℝ := x - 8

theorem and_15_and : and_x (x_and 15) = -15 :=
by
  sorry

end and_15_and_l71_71835


namespace simplify_and_evaluate_expression_l71_71560

theorem simplify_and_evaluate_expression (x : ℚ) (h : x = 3) :
  ((2 * x^2 + 2 * x) / (x^2 - 1) - (x^2 - x) / (x^2 - 2 * x + 1)) / (x / (x + 1)) = 2 :=
by
  sorry

end simplify_and_evaluate_expression_l71_71560


namespace eliminate_alpha_l71_71064

theorem eliminate_alpha (α x y : ℝ) (h1 : x = Real.tan α ^ 2) (h2 : y = Real.sin α ^ 2) : 
  x - y = x * y := 
by
  sorry

end eliminate_alpha_l71_71064


namespace Leroy_min_bail_rate_l71_71508

noncomputable def min_bailing_rate
    (distance_to_shore : ℝ)
    (leak_rate : ℝ)
    (max_tolerable_water : ℝ)
    (rowing_speed : ℝ)
    : ℝ :=
  let time_to_shore := distance_to_shore / rowing_speed * 60
  let total_water_intake := leak_rate * time_to_shore
  let required_bailing := total_water_intake - max_tolerable_water
  required_bailing / time_to_shore

theorem Leroy_min_bail_rate
    (distance_to_shore : ℝ := 2)
    (leak_rate : ℝ := 15)
    (max_tolerable_water : ℝ := 60)
    (rowing_speed : ℝ := 4)
    : min_bailing_rate 2 15 60 4 = 13 := 
by
  simp [min_bailing_rate]
  sorry

end Leroy_min_bail_rate_l71_71508


namespace david_average_marks_l71_71414

-- Define the individual marks
def english_marks : ℕ := 74
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 90
def total_marks : ℕ := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

-- Define the average marks
def average_marks : ℚ := total_marks / num_subjects

-- Assert the average marks calculation
theorem david_average_marks : average_marks = 75.6 := by
  sorry

end david_average_marks_l71_71414


namespace solve_for_x_l71_71746

theorem solve_for_x (x : ℝ) (h : (6 * x ^ 2 + 111 * x + 1) / (2 * x + 37) = 3 * x + 1) : x = -18 :=
sorry

end solve_for_x_l71_71746


namespace probability_of_triangle_l71_71674

/-- There are 12 figures in total: 4 squares, 5 triangles, and 3 rectangles.
    Prove that the probability of choosing a triangle is 5/12. -/
theorem probability_of_triangle (total_figures : ℕ) (num_squares : ℕ) (num_triangles : ℕ) (num_rectangles : ℕ)
  (h1 : total_figures = 12)
  (h2 : num_squares = 4)
  (h3 : num_triangles = 5)
  (h4 : num_rectangles = 3) :
  num_triangles / total_figures = 5 / 12 :=
sorry

end probability_of_triangle_l71_71674


namespace arithmetic_sequence_difference_l71_71016

theorem arithmetic_sequence_difference 
  (a b c : ℝ) 
  (h1: 2 + (7 / 4) = a)
  (h2: 2 + 2 * (7 / 4) = b)
  (h3: 2 + 3 * (7 / 4) = c)
  (h4: 2 + 4 * (7 / 4) = 9):
  c - a = 3.5 :=
by sorry

end arithmetic_sequence_difference_l71_71016


namespace trigonometric_expression_l71_71551

theorem trigonometric_expression
  (α : ℝ)
  (h : (Real.tan α - 3) * (Real.sin α + Real.cos α + 3) = 0) :
  2 + (2 / 3) * Real.sin α ^ 2 + (1 / 4) * Real.cos α ^ 2 = 21 / 8 := 
by sorry

end trigonometric_expression_l71_71551


namespace find_b_l71_71940

theorem find_b (b : ℤ) :
  (∃ x : ℤ, x^2 + b * x - 36 = 0 ∧ x = -9) → b = 5 :=
by
  sorry

end find_b_l71_71940


namespace part_I_part_II_l71_71874

noncomputable def f (a x : ℝ) : ℝ := |x - 1| + a * |x - 2|

theorem part_I (a : ℝ) (h_min : ∃ m, ∀ x, f a x ≥ m) : -1 ≤ a ∧ a ≤ 1 :=
sorry

theorem part_II (a : ℝ) (h_bound : ∀ x, f a x ≥ 1/2) : a = 1/3 :=
sorry

end part_I_part_II_l71_71874


namespace exists_y_less_than_half_p_l71_71351

theorem exists_y_less_than_half_p (p : ℕ) (hp : Nat.Prime p) (hp_gt3 : p > 3) :
  ∃ (y : ℕ), y < p / 2 ∧ ∀ (a b : ℕ), p * y + 1 ≠ a * b ∨ a ≤ y ∨ b ≤ y :=
by sorry

end exists_y_less_than_half_p_l71_71351


namespace Zhukov_birth_year_l71_71075

-- Define the conditions
def years_lived_total : ℕ := 78
def years_lived_20th_more_than_19th : ℕ := 70

-- Define the proof problem
theorem Zhukov_birth_year :
  ∃ y19 y20 : ℕ, y19 + y20 = years_lived_total ∧ y20 = y19 + years_lived_20th_more_than_19th ∧ (1900 - y19) = 1896 :=
by
  sorry

end Zhukov_birth_year_l71_71075


namespace steven_owes_jeremy_l71_71288

-- Define the payment per room
def payment_per_room : ℚ := 13 / 3

-- Define the number of rooms cleaned
def rooms_cleaned : ℚ := 5 / 2

-- Calculate the total amount owed
def total_amount_owed : ℚ := payment_per_room * rooms_cleaned

-- The theorem statement to prove
theorem steven_owes_jeremy :
  total_amount_owed = 65 / 6 :=
by
  sorry

end steven_owes_jeremy_l71_71288


namespace students_journals_l71_71127

theorem students_journals :
  ∃ u v : ℕ, 
    u + v = 75000 ∧ 
    (7 * u + 2 * v = 300000) ∧ 
    (∃ b g : ℕ, b = u * 7 / 300 ∧ g = v * 2 / 300 ∧ b = 700 ∧ g = 300) :=
by {
  -- The proving steps will go here
  sorry
}

end students_journals_l71_71127


namespace scientific_notation_500_billion_l71_71465

theorem scientific_notation_500_billion :
  ∃ (a : ℝ), 500000000000 = a * 10 ^ 10 ∧ 1 ≤ a ∧ a < 10 :=
by
  sorry

end scientific_notation_500_billion_l71_71465


namespace inequality_proof_l71_71051

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ((b + c - a)^2) / (a^2 + (b + c)^2) + ((c + a - b)^2) / (b^2 + (c + a)^2) + ((a + b - c)^2) / (c^2 + (a + b)^2) ≥ 3 / 5 :=
  sorry

end inequality_proof_l71_71051


namespace vasya_wins_l71_71646

-- Definition of the game and players
inductive Player
| Vasya : Player
| Petya : Player

-- Define the problem conditions
structure Game where
  initial_piles : ℕ := 1      -- Initially, there is one pile
  players_take_turns : Bool := true
  take_or_divide : Bool := true
  remove_last_wins : Bool := true
  vasya_first_but_cannot_take_initially : Bool := true

-- Define the function to determine the winner
def winner_of_game (g : Game) : Player :=
  if g.initial_piles = 1 ∧ g.vasya_first_but_cannot_take_initially then Player.Vasya else Player.Petya

-- Define the theorem stating Vasya will win given the game conditions
theorem vasya_wins : ∀ (g : Game), g = {
    initial_piles := 1,
    players_take_turns := true,
    take_or_divide := true,
    remove_last_wins := true,
    vasya_first_but_cannot_take_initially := true
} → winner_of_game g = Player.Vasya := by
  -- Insert proof here
  sorry

end vasya_wins_l71_71646


namespace max_sum_cos_l71_71629

theorem max_sum_cos (a b c : ℝ) (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x) ≥ -1) : a + b + c ≤ 3 := by
  sorry

end max_sum_cos_l71_71629


namespace arrange_numbers_l71_71608

theorem arrange_numbers :
  (2 : ℝ) ^ 1000 < (5 : ℝ) ^ 500 ∧ (5 : ℝ) ^ 500 < (3 : ℝ) ^ 750 :=
by
  sorry

end arrange_numbers_l71_71608


namespace find_f_half_l71_71906

variable {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

def f_condition (f : R → R) : Prop := ∀ x : R, x < 0 → f x = 1 / (x + 1)

theorem find_f_half (f : R → R) (h_odd : odd_function f) (h_condition : f_condition f) : f (1 / 2) = -2 := by
  sorry

end find_f_half_l71_71906


namespace binom_even_if_power_of_two_binom_odd_if_not_power_of_two_l71_71175

-- Definition of power of two
def is_power_of_two (n : ℕ) := ∃ m : ℕ, n = 2^m

-- Theorems to be proven
theorem binom_even_if_power_of_two (n : ℕ) (h : is_power_of_two n) :
  ∀ k : ℕ, 1 ≤ k ∧ k < n → Nat.choose n k % 2 = 0 := sorry

theorem binom_odd_if_not_power_of_two (n : ℕ) (h : ¬ is_power_of_two n) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ Nat.choose n k % 2 = 1 := sorry

end binom_even_if_power_of_two_binom_odd_if_not_power_of_two_l71_71175


namespace find_marks_in_chemistry_l71_71679

theorem find_marks_in_chemistry
  (marks_english : ℕ)
  (marks_math : ℕ)
  (marks_physics : ℕ)
  (marks_biology : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (marks_english_eq : marks_english = 86)
  (marks_math_eq : marks_math = 85)
  (marks_physics_eq : marks_physics = 92)
  (marks_biology_eq : marks_biology = 95)
  (average_marks_eq : average_marks = 89)
  (num_subjects_eq : num_subjects = 5) : 
  ∃ marks_chemistry : ℕ, marks_chemistry = 87 :=
by
  sorry

end find_marks_in_chemistry_l71_71679


namespace gcd_176_88_l71_71543

theorem gcd_176_88 : Nat.gcd 176 88 = 88 :=
by
  sorry

end gcd_176_88_l71_71543


namespace range_of_sum_l71_71021

theorem range_of_sum (a b c : ℝ) (h1: a > b) (h2 : b > c) (h3 : a + b + c = 1) (h4 : a^2 + b^2 + c^2 = 3) :
-2/3 < b + c ∧ b + c < 0 := 
by 
  sorry

end range_of_sum_l71_71021


namespace decrease_in_radius_l71_71173

theorem decrease_in_radius
  (dist_summer : ℝ)
  (dist_winter : ℝ)
  (radius_summer : ℝ) 
  (mile_to_inch : ℝ)
  (π : ℝ) 
  (δr : ℝ) :
  dist_summer = 560 →
  dist_winter = 570 →
  radius_summer = 20 →
  mile_to_inch = 63360 →
  π = Real.pi →
  δr = 0.33 :=
sorry

end decrease_in_radius_l71_71173


namespace swans_count_l71_71731

def numberOfSwans : Nat := 12

theorem swans_count (y : Nat) (x : Nat) (h1 : y = 5) (h2 : ∃ n m : Nat, x = 2 * n + 2 ∧ x = 3 * m - 3) : x = numberOfSwans := 
  by 
    sorry

end swans_count_l71_71731


namespace infinite_solutions_l71_71421

theorem infinite_solutions (x y : ℝ) : ∃ x y : ℝ, x^3 + y^2 * x - 6 * x + 5 * y + 1 = 0 :=
sorry

end infinite_solutions_l71_71421


namespace correct_calculation_l71_71156

variable (a b : ℚ)

theorem correct_calculation :
  (a / b) ^ 4 = a ^ 4 / b ^ 4 := 
by
  sorry

end correct_calculation_l71_71156


namespace inequality_holds_l71_71426

-- Given conditions
variables {a b x y : ℝ}
variables (pos_a : 0 < a) (pos_b : 0 < b) (pos_x : 0 < x) (pos_y : 0 < y)
variable (h : a + b = 1)

-- Goal/Question
theorem inequality_holds : (a * x + b * y) * (b * x + a * y) ≥ x * y :=
by sorry

end inequality_holds_l71_71426


namespace m_div_x_eq_4_div_5_l71_71846

variable (a b : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_ratio : a / b = 4 / 5)

def x := a * 1.25

def m := b * 0.80

theorem m_div_x_eq_4_div_5 : m / x = 4 / 5 :=
by
  sorry

end m_div_x_eq_4_div_5_l71_71846


namespace cost_of_natural_seedless_raisins_l71_71680

theorem cost_of_natural_seedless_raisins
  (cost_golden: ℝ) (n_golden: ℕ) (n_natural: ℕ) (cost_mixture: ℝ) (cost_per_natural: ℝ) :
  cost_golden = 2.55 ∧ n_golden = 20 ∧ n_natural = 20 ∧ cost_mixture = 3
  → cost_per_natural = 3.45 :=
by
  sorry

end cost_of_natural_seedless_raisins_l71_71680


namespace power_function_increasing_is_3_l71_71281

theorem power_function_increasing_is_3 (m : ℝ) :
  (∀ x : ℝ, x > 0 → (m^2 - m - 5) * (x^(m)) > 0) ∧ (m^2 - m - 5 = 1) → m = 3 :=
by
  sorry

end power_function_increasing_is_3_l71_71281


namespace fuel_consumption_l71_71753

def initial_volume : ℕ := 3000
def volume_jan_1 : ℕ := 180
def volume_may_1 : ℕ := 1238
def refill_volume : ℕ := 3000

theorem fuel_consumption :
  (initial_volume - volume_jan_1) + (refill_volume - volume_may_1) = 4582 := by
  sorry

end fuel_consumption_l71_71753


namespace A_knit_time_l71_71697

def rate_A (x : ℕ) : ℚ := 1 / x
def rate_B : ℚ := 1 / 6

def combined_rate_two_pairs_in_4_days (x : ℕ) : Prop :=
  rate_A x + rate_B = 1 / 2

theorem A_knit_time : ∃ x : ℕ, combined_rate_two_pairs_in_4_days x ∧ x = 3 :=
by
  existsi 3
  -- (Formal proof would go here)
  sorry

end A_knit_time_l71_71697


namespace green_faction_lies_more_l71_71441

theorem green_faction_lies_more (r1 r2 r3 l1 l2 l3 : ℕ) 
  (h1 : r1 + r2 + r3 + l1 + l2 + l3 = 2016) 
  (h2 : r1 + l2 + l3 = 1208) 
  (h3 : r2 + l1 + l3 = 908) 
  (h4 : r3 + l1 + l2 = 608) :
  l3 - r3 = 100 :=
by
  sorry

end green_faction_lies_more_l71_71441


namespace number_of_terms_ap_l71_71891

variables (a d n : ℤ) 

def sum_of_first_thirteen_terms := (13 / 2) * (2 * a + 12 * d)
def sum_of_last_thirteen_terms := (13 / 2) * (2 * a + (2 * n - 14) * d)

def sum_excluding_first_three := ((n - 3) / 2) * (2 * a + (n - 4) * d)
def sum_excluding_last_three := ((n - 3) / 2) * (2 * a + (n - 1) * d)

theorem number_of_terms_ap (h1 : sum_of_first_thirteen_terms a d = (1 / 2) * sum_of_last_thirteen_terms a d)
  (h2 : sum_excluding_first_three a d / sum_excluding_last_three a d = 5 / 4) : n = 22 :=
sorry

end number_of_terms_ap_l71_71891


namespace probability_at_least_one_white_l71_71305

def total_number_of_pairs : ℕ := 10
def number_of_pairs_with_at_least_one_white_ball : ℕ := 7

theorem probability_at_least_one_white :
  (number_of_pairs_with_at_least_one_white_ball : ℚ) / (total_number_of_pairs : ℚ) = 7 / 10 :=
by
  sorry

end probability_at_least_one_white_l71_71305


namespace mod_mult_congruence_l71_71672

theorem mod_mult_congruence (n : ℤ) (h1 : 215 ≡ 65 [ZMOD 75])
  (h2 : 789 ≡ 39 [ZMOD 75]) (h3 : 215 * 789 ≡ n [ZMOD 75]) (hn : 0 ≤ n ∧ n < 75) :
  n = 60 :=
by
  sorry

end mod_mult_congruence_l71_71672


namespace exist_coprime_sums_l71_71879

theorem exist_coprime_sums (n k : ℕ) (h1 : 0 < n) (h2 : Even (k * (n - 1))) :
  ∃ x y : ℕ, Nat.gcd x n = 1 ∧ Nat.gcd y n = 1 ∧ (x + y) % n = k % n :=
  sorry

end exist_coprime_sums_l71_71879


namespace product_xyz_l71_71682

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 2) (h3 : x + 1 / z = 3) : x * y * z = 2 := 
by sorry

end product_xyz_l71_71682


namespace initial_bottles_calculation_l71_71980

theorem initial_bottles_calculation (maria_bottles : ℝ) (sister_bottles : ℝ) (left_bottles : ℝ) 
  (H₁ : maria_bottles = 14.0) (H₂ : sister_bottles = 8.0) (H₃ : left_bottles = 23.0) :
  maria_bottles + sister_bottles + left_bottles = 45.0 :=
by
  sorry

end initial_bottles_calculation_l71_71980


namespace fraction_simplification_l71_71982

theorem fraction_simplification (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) : 
  (x^2 + x) / (x^2 - 1) = x / (x - 1) :=
by
  -- Hint of expected development environment setting
  sorry

end fraction_simplification_l71_71982


namespace complaint_online_prob_l71_71122

/-- Define the various probability conditions -/
def prob_online := 4 / 5
def prob_store := 1 / 5
def qual_rate_online := 17 / 20
def qual_rate_store := 9 / 10
def non_qual_rate_online := 1 - qual_rate_online
def non_qual_rate_store := 1 - qual_rate_store
def prob_complaint_online := prob_online * non_qual_rate_online
def prob_complaint_store := prob_store * non_qual_rate_store
def total_prob_complaint := prob_complaint_online + prob_complaint_store

/-- The theorem states that given the conditions, the probability of an online purchase given a complaint is 6/7 -/
theorem complaint_online_prob : 
    (prob_complaint_online / total_prob_complaint) = 6 / 7 := 
by
    sorry

end complaint_online_prob_l71_71122


namespace find_factor_l71_71032

theorem find_factor (x f : ℝ) (h1 : x = 6)
    (h2 : (2 * x + 9) * f = 63) : f = 3 :=
sorry

end find_factor_l71_71032


namespace parabola_intersects_x_axis_l71_71996

theorem parabola_intersects_x_axis {p q x₀ x₁ x₂ : ℝ} (h : ∀ (x : ℝ), x ^ 2 + p * x + q ≠ 0)
    (M_below_x_axis : x₀ ^ 2 + p * x₀ + q < 0)
    (M_at_1_neg2 : x₀ = 1 ∧ (1 ^ 2 + p * 1 + q = -2)) :
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₀ < x₁ → x₁ < x₂) ∧ x₁ = -1 ∧ x₂ = 2 ∨ x₁ = 0 ∧ x₂ = 3) :=
by
  sorry

end parabola_intersects_x_axis_l71_71996


namespace range_of_f_x_lt_1_l71_71937

theorem range_of_f_x_lt_1 (x : ℝ) (f : ℝ → ℝ) (h : f x = x^3) : f x < 1 ↔ x < 1 := by
  sorry

end range_of_f_x_lt_1_l71_71937


namespace probability_of_all_selected_l71_71234

theorem probability_of_all_selected :
  let p_x := 1 / 7
  let p_y := 2 / 9
  let p_z := 3 / 11
  p_x * p_y * p_z = 1 / 115.5 :=
by
  let p_x := 1 / 7
  let p_y := 2 / 9
  let p_z := 3 / 11
  sorry

end probability_of_all_selected_l71_71234


namespace square_area_side4_l71_71660

theorem square_area_side4
  (s : ℕ)
  (A : ℕ)
  (P : ℕ)
  (h_s : s = 4)
  (h_A : A = s * s)
  (h_P : P = 4 * s)
  (h_eqn : (A + s) - P = 4) : A = 16 := sorry

end square_area_side4_l71_71660


namespace value_of_m_l71_71233

theorem value_of_m (x m : ℝ) (h_positive_root : x > 0) (h_eq : x / (x - 1) - m / (1 - x) = 2) : m = -1 := by
  sorry

end value_of_m_l71_71233


namespace no_such_function_exists_l71_71639

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ m n : ℕ, (m + f n)^2 ≥ 3 * (f m)^2 + n^2 :=
by 
  sorry

end no_such_function_exists_l71_71639


namespace conversion_bah_rah_yah_l71_71043

theorem conversion_bah_rah_yah (bahs rahs yahs : ℝ) 
  (h1 : 10 * bahs = 16 * rahs) 
  (h2 : 6 * rahs = 10 * yahs) :
  (10 / 16) * (6 / 10) * 500 * yahs = 187.5 * bahs :=
by sorry

end conversion_bah_rah_yah_l71_71043


namespace fraction_problem_l71_71237

theorem fraction_problem
    (q r s u : ℚ)
    (h1 : q / r = 8)
    (h2 : s / r = 4)
    (h3 : s / u = 1 / 3) :
    u / q = 3 / 2 :=
  sorry

end fraction_problem_l71_71237


namespace circle_equation_through_points_l71_71588

theorem circle_equation_through_points (A B: ℝ × ℝ) (C : ℝ × ℝ)
  (hA : A = (1, -1)) (hB : B = (-1, 1)) (hC : C.1 + C.2 = 2)
  (hAC : dist A C = dist B C) :
  (x - C.1) ^ 2 + (y - C.2) ^ 2 = 4 :=
by
  sorry

end circle_equation_through_points_l71_71588


namespace evaluate_expression_l71_71914

theorem evaluate_expression (a b c : ℤ) (ha : a = 3) (hb : b = 2) (hc : c = 1) :
  ((a^2 + b + c)^2 - (a^2 - b - c)^2) = 108 :=
by
  sorry

end evaluate_expression_l71_71914


namespace probability_diff_by_three_l71_71915

theorem probability_diff_by_three (r1 r2 : ℕ) (h1 : 1 ≤ r1 ∧ r1 ≤ 6) (h2 : 1 ≤ r2 ∧ r2 ≤ 6) :
  (∃ (rolls : List (ℕ × ℕ)), 
    rolls = [ (2, 5), (5, 2), (3, 6), (4, 1) ] ∧ 
    (r1, r2) ∈ rolls) →
  (4 : ℚ) / 36 = (1 / 9 : ℚ) :=
by sorry

end probability_diff_by_three_l71_71915


namespace gerald_paid_l71_71849

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 := by
  sorry

end gerald_paid_l71_71849


namespace min_value_pq_l71_71115

theorem min_value_pq (p q : ℝ) (hp : 0 < p) (hq : 0 < q)
  (h1 : p^2 - 8 * q ≥ 0)
  (h2 : 4 * q^2 - 4 * p ≥ 0) :
  p + q ≥ 6 :=
sorry

end min_value_pq_l71_71115


namespace leopards_arrangement_l71_71749

theorem leopards_arrangement :
  let total_leopards := 9
  let ends_leopards := 2
  let middle_leopard := 1
  let remaining_leopards := total_leopards - ends_leopards - middle_leopard
  (2 * 1 * (Nat.factorial remaining_leopards) = 1440) := by
  sorry

end leopards_arrangement_l71_71749


namespace eq_squares_diff_l71_71792

theorem eq_squares_diff {x y z : ℝ} :
  x = (y - z)^2 ∧ y = (x - z)^2 ∧ z = (x - y)^2 →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 0) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 1) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 1) :=
by
  sorry

end eq_squares_diff_l71_71792


namespace sin_right_triangle_l71_71203

theorem sin_right_triangle (FG GH : ℝ) (h1 : FG = 13) (h2 : GH = 12) (h3 : FG^2 = FH^2 + GH^2) : 
  sin_H = 5 / 13 :=
by sorry

end sin_right_triangle_l71_71203


namespace log_negative_l71_71845

open Real

theorem log_negative (a : ℝ) (h : a > 0) : log (-a) = log a := sorry

end log_negative_l71_71845


namespace difference_is_correct_l71_71518

-- Definition of the given numbers
def numbers : List ℕ := [44, 16, 2, 77, 241]

-- Define the sum of the numbers
def sum_numbers := numbers.sum

-- Define the average of the numbers
def average := sum_numbers / numbers.length

-- Define the difference between sum and average
def difference := sum_numbers - average

-- The theorem we need to prove
theorem difference_is_correct : difference = 304 := by
  sorry

end difference_is_correct_l71_71518


namespace find_number_l71_71336

theorem find_number (x : ℕ) (h : 5 + 2 * (8 - x) = 15) : x = 3 :=
sorry

end find_number_l71_71336


namespace not_product_24_pair_not_24_l71_71592

theorem not_product_24 (a b : ℤ) : 
  (a, b) = (-4, -6) ∨ (a, b) = (-2, -12) ∨ (a, b) = (2, 12) ∨ (a, b) = (3/4, 32) → a * b = 24 :=
sorry

theorem pair_not_24 :
  ¬(1/3 * -72 = 24) :=
sorry

end not_product_24_pair_not_24_l71_71592


namespace power_function_no_origin_l71_71295

theorem power_function_no_origin (m : ℝ) : 
  (m^2 - m - 1 <= 0) ∧ (m^2 - 3 * m + 3 = 1) → m = 1 :=
by
  intros
  sorry

end power_function_no_origin_l71_71295


namespace quadratic_b_value_l71_71268

theorem quadratic_b_value (b : ℝ) (n : ℝ) (h_b_neg : b < 0) 
  (h_equiv : ∀ x : ℝ, (x + n)^2 + 1 / 16 = x^2 + b * x + 1 / 4) : 
  b = - (Real.sqrt 3) / 2 := 
sorry

end quadratic_b_value_l71_71268


namespace average_minutes_run_is_44_over_3_l71_71386

open BigOperators

def average_minutes_run (s : ℕ) : ℚ :=
  let sixth_graders := 3 * s
  let seventh_graders := s
  let eighth_graders := s / 2
  let total_students := sixth_graders + seventh_graders + eighth_graders
  let total_minutes_run := 20 * sixth_graders + 12 * eighth_graders
  total_minutes_run / total_students

theorem average_minutes_run_is_44_over_3 (s : ℕ) (h1 : 0 < s) : 
  average_minutes_run s = 44 / 3 := 
by
  sorry

end average_minutes_run_is_44_over_3_l71_71386


namespace bus_initial_count_l71_71385

theorem bus_initial_count (x : ℕ) (got_off : ℕ) (remained : ℕ) (h1 : got_off = 47) (h2 : remained = 43) (h3 : x - got_off = remained) : x = 90 :=
by
  rw [h1, h2] at h3
  sorry

end bus_initial_count_l71_71385


namespace find_current_l71_71607

theorem find_current (R Q t : ℝ) (hR : R = 8) (hQ : Q = 72) (ht : t = 2) :
  ∃ I : ℝ, Q = I^2 * R * t ∧ I = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end find_current_l71_71607


namespace inv_prop_x_y_l71_71300

theorem inv_prop_x_y (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x = 4) (h3 : y = 2) (h4 : y = 10) : x = 4 / 5 :=
by
  sorry

end inv_prop_x_y_l71_71300


namespace total_students_in_class_l71_71059

def current_students : ℕ := 6 * 3
def students_bathroom : ℕ := 5
def students_canteen : ℕ := 5 * 5
def group1_students : ℕ := 4
def group2_students : ℕ := 5
def group3_students : ℕ := 6
def group4_students : ℕ := 3
def new_group_students : ℕ := group1_students + group2_students + group3_students + group4_students
def germany_students : ℕ := 3
def france_students : ℕ := 4
def norway_students : ℕ := 3
def italy_students : ℕ := 2
def spain_students : ℕ := 2
def australia_students : ℕ := 1
def foreign_exchange_students : ℕ :=
  germany_students + france_students + norway_students + italy_students + spain_students + australia_students

def total_students : ℕ :=
  current_students + students_bathroom + students_canteen + new_group_students + foreign_exchange_students

theorem total_students_in_class : total_students = 81 := by
  rfl  -- Reflective equality since total_students already sums to 81 based on the definitions

end total_students_in_class_l71_71059


namespace car_rental_cost_l71_71198

theorem car_rental_cost (daily_rent : ℕ) (rent_duration : ℕ) (mileage_rate : ℚ) (mileage : ℕ) (total_cost : ℕ) :
  daily_rent = 30 → rent_duration = 5 → mileage_rate = 0.25 → mileage = 500 → total_cost = 275 :=
by
  intros hd hr hm hl
  sorry

end car_rental_cost_l71_71198


namespace factorize_x4_plus_16_l71_71019

theorem factorize_x4_plus_16: ∀ (x : ℝ), x^4 + 16 = (x^2 + 2 * x + 2) * (x^2 - 2 * x + 2) :=
by
  intro x
  sorry

end factorize_x4_plus_16_l71_71019


namespace polygonal_number_8_8_l71_71574

-- Definitions based on conditions
def triangular_number (n : ℕ) : ℕ := (n^2 + n) / 2
def square_number (n : ℕ) : ℕ := n^2
def pentagonal_number (n : ℕ) : ℕ := (3 * n^2 - n) / 2
def hexagonal_number (n : ℕ) : ℕ := (4 * n^2 - 2 * n) / 2

-- General formula for k-sided polygonal number
def polygonal_number (n k : ℕ) : ℕ := ((k - 2) * n^2 + (4 - k) * n) / 2

-- The proposition to be proved
theorem polygonal_number_8_8 : polygonal_number 8 8 = 176 := by
  sorry

end polygonal_number_8_8_l71_71574


namespace infinite_series_sum_l71_71204

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l71_71204


namespace arctan_sum_l71_71063

theorem arctan_sum : 
  Real.arctan (1/2) + Real.arctan (1/5) + Real.arctan (1/8) = Real.pi / 4 := 
by 
  sorry

end arctan_sum_l71_71063


namespace family_ages_l71_71565

theorem family_ages 
  (youngest : ℕ)
  (middle : ℕ := youngest + 2)
  (eldest : ℕ := youngest + 4)
  (mother : ℕ := 3 * youngest + 16)
  (father : ℕ := 4 * youngest + 18)
  (total_sum : youngest + middle + eldest + mother + father = 90) :
  youngest = 5 ∧ middle = 7 ∧ eldest = 9 ∧ mother = 31 ∧ father = 38 := 
by 
  sorry

end family_ages_l71_71565


namespace problem_l71_71436

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem problem {a α b β : ℝ} (h : f 2001 a α b β = 3) : f 2012 a α b β = -3 := by
  sorry

end problem_l71_71436


namespace exists_zero_in_interval_minus3_minus2_l71_71920

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x - x

theorem exists_zero_in_interval_minus3_minus2 : 
  ∃ x ∈ Set.Icc (-3 : ℝ) (-2), f x = 0 :=
by
  sorry

end exists_zero_in_interval_minus3_minus2_l71_71920


namespace arithmetic_mean_three_fractions_l71_71648

theorem arithmetic_mean_three_fractions :
  let a := (5 : ℚ) / 8
  let b := (7 : ℚ) / 8
  let c := (3 : ℚ) / 4
  (a + b) / 2 = c :=
by
  sorry

end arithmetic_mean_three_fractions_l71_71648


namespace monster_perimeter_l71_71494

theorem monster_perimeter (r : ℝ) (theta : ℝ) (h₁ : r = 2) (h₂ : theta = 90 * π / 180) :
  2 * r + (3 / 4) * (2 * π * r) = 3 * π + 4 := by
  -- Sorry to skip the proof.
  sorry

end monster_perimeter_l71_71494


namespace Vasya_birthday_on_Thursday_l71_71480

-- Define the days of the week enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define the conditions
def Sunday_is_the_day_after_tomorrow (today : DayOfWeek) : Prop :=
  match today with
  | Friday => true
  | _      => false

def Vasya_birthday_not_on_Sunday (birthday : DayOfWeek) : Prop :=
  birthday ≠ Sunday

def Today_is_next_day_after_birthday (today birthday : DayOfWeek) : Prop :=
  match (today, birthday) with
  | (Friday, Thursday) => true
  | _                  => false

-- The main theorem to prove Vasya's birthday was on Thursday
theorem Vasya_birthday_on_Thursday (birthday today : DayOfWeek) 
  (H1: Sunday_is_the_day_after_tomorrow today) 
  (H2: Vasya_birthday_not_on_Sunday birthday) 
  (H3: Today_is_next_day_after_birthday today birthday) : 
  birthday = Thursday :=
by
  sorry

end Vasya_birthday_on_Thursday_l71_71480


namespace jeff_can_store_songs_l71_71104

def gbToMb (gb : ℕ) : ℕ := gb * 1000

def newAppsStorage : ℕ :=
  5 * 450 + 5 * 300 + 5 * 150

def newPhotosStorage : ℕ :=
  300 * 4 + 50 * 8

def newVideosStorage : ℕ :=
  15 * 400 + 30 * 200

def newPDFsStorage : ℕ :=
  25 * 20

def totalNewStorage : ℕ :=
  newAppsStorage + newPhotosStorage + newVideosStorage + newPDFsStorage

def existingStorage : ℕ :=
  gbToMb 7

def totalUsedStorage : ℕ :=
  existingStorage + totalNewStorage

def totalStorage : ℕ :=
  gbToMb 32

def remainingStorage : ℕ :=
  totalStorage - totalUsedStorage

def numSongs (storage : ℕ) (avgSongSize : ℕ) : ℕ :=
  storage / avgSongSize

theorem jeff_can_store_songs : 
  numSongs remainingStorage 20 = 320 :=
by
  sorry

end jeff_can_store_songs_l71_71104


namespace two_f_x_eq_8_over_4_plus_x_l71_71512

variable (f : ℝ → ℝ)
variable (x : ℝ)
variables (hx : 0 < x)
variable (h : ∀ x, 0 < x → f (2 * x) = 2 / (2 + x))

theorem two_f_x_eq_8_over_4_plus_x : 2 * f x = 8 / (4 + x) :=
by sorry

end two_f_x_eq_8_over_4_plus_x_l71_71512


namespace telephone_number_problem_l71_71992

theorem telephone_number_problem
  (digits : Finset ℕ)
  (A B C D E F G H I J : ℕ)
  (h_digits : digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h_distinct : [A, B, C, D, E, F, G, H, I, J].Nodup)
  (h_ABC : A > B ∧ B > C)
  (h_DEF : D > E ∧ E > F)
  (h_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_DEF_consecutive_odd : D = E + 2 ∧ E = F + 2 ∧ (D % 2 = 1) ∧ (E % 2 = 1) ∧ (F % 2 = 1))
  (h_GHIJ_consecutive_even : G = H + 2 ∧ H = I + 2 ∧ I = J + 2 ∧ (G % 2 = 0) ∧ (H % 2 = 0) ∧ (I % 2 = 0) ∧ (J % 2 = 0))
  (h_sum_ABC : A + B + C = 15) :
  A = 9 :=
by
  sorry

end telephone_number_problem_l71_71992


namespace area_difference_l71_71356

-- Definitions of the given conditions
structure Triangle :=
(base : ℝ)
(height : ℝ)

def area (t : Triangle) : ℝ :=
  0.5 * t.base * t.height

-- Conditions of the problem
def EFG : Triangle := {base := 8, height := 4}
def EFG' : Triangle := {base := 4, height := 2}

-- Proof statement
theorem area_difference :
  area EFG - area EFG' = 12 :=
by
  sorry

end area_difference_l71_71356


namespace geom_seq_sum_l71_71511

variable {a : ℕ → ℝ}

theorem geom_seq_sum (h : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) : a 5 + a 7 = 6 ∨ a 5 + a 7 = -6 := by
  sorry

end geom_seq_sum_l71_71511


namespace probability_all_same_room_probability_at_least_two_same_room_l71_71566

/-- 
  Given that there are three people and each person is assigned to one of four rooms with equal probability,
  let P1 be the probability that all three people are assigned to the same room,
  and let P2 be the probability that at least two people are assigned to the same room.
  We need to prove:
  1. P1 = 1 / 16
  2. P2 = 5 / 8
-/
noncomputable def P1 : ℚ := sorry

noncomputable def P2 : ℚ := sorry

theorem probability_all_same_room :
  P1 = 1 / 16 :=
sorry

theorem probability_at_least_two_same_room :
  P2 = 5 / 8 :=
sorry

end probability_all_same_room_probability_at_least_two_same_room_l71_71566


namespace shark_feed_l71_71415

theorem shark_feed (S : ℝ) (h1 : S + S/2 + 5 * S = 26) : S = 4 := 
by sorry

end shark_feed_l71_71415


namespace variance_of_yield_l71_71796

/-- Given a data set representing annual average yields,
    prove that the variance of this data set is approximately 171. --/
theorem variance_of_yield {yields : List ℝ} 
  (h_yields : yields = [450, 430, 460, 440, 450, 440, 470, 460]) :
  let mean := (yields.sum / yields.length : ℝ)
  let squared_diffs := (yields.map (fun x => (x - mean)^2))
  let variance := (squared_diffs.sum / (yields.length - 1 : ℝ))
  abs (variance - 171) < 1 :=
by
  sorry

end variance_of_yield_l71_71796


namespace smallest_n_not_prime_l71_71962

theorem smallest_n_not_prime : ∃ n, n = 4 ∧ ∀ m : ℕ, m < 4 → Prime (2 * m + 1) ∧ ¬ Prime (2 * 4 + 1) :=
by
  sorry

end smallest_n_not_prime_l71_71962


namespace smaller_investment_value_l71_71489

theorem smaller_investment_value :
  ∃ (x : ℝ), 0.07 * x + 0.27 * 1500 = 0.22 * (x + 1500) ∧ x = 500 :=
by
  sorry

end smaller_investment_value_l71_71489


namespace spherical_to_rectangular_coordinates_l71_71092

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ),
  ρ = 5 → θ = π / 6 → φ = π / 3 →
  let x := ρ * (Real.sin φ * Real.cos θ)
  let y := ρ * (Real.sin φ * Real.sin θ)
  let z := ρ * Real.cos φ
  x = 15 / 4 ∧ y = 5 * Real.sqrt 3 / 4 ∧ z = 2.5 :=
by
  intros ρ θ φ hρ hθ hφ
  sorry

end spherical_to_rectangular_coordinates_l71_71092


namespace count_integers_congruent_mod_l71_71723

theorem count_integers_congruent_mod (n : ℕ) (h₁ : n < 1200) (h₂ : n ≡ 3 [MOD 7]) : 
  ∃ (m : ℕ), (m = 171) :=
by
  sorry

end count_integers_congruent_mod_l71_71723


namespace arithmetic_seq_a7_value_l71_71709

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ): Prop := 
  ∀ n : ℕ, a (n+1) = a n + d

theorem arithmetic_seq_a7_value
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : a 4 = 4)
  (h3 : a 3 + a 8 = 5) :
  a 7 = 1 := 
sorry

end arithmetic_seq_a7_value_l71_71709


namespace max_n_perfect_cube_l71_71721

-- Definition for sum of squares
def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- Definition for sum of squares from (n+1) to 2n
def sum_of_squares_segment (n : ℕ) : ℕ :=
  2 * n * (2 * n + 1) * (4 * n + 1) / 6 - n * (n + 1) * (2 * n + 1) / 6

-- Definition for the product of the sums
def product_of_sums (n : ℕ) : ℕ :=
  (sum_of_squares n) * (sum_of_squares_segment n)

-- Predicate for perfect cube
def is_perfect_cube (x : ℕ) : Prop :=
  ∃ y : ℕ, y ^ 3 = x

-- The main theorem to be proved
theorem max_n_perfect_cube : ∃ (n : ℕ), n ≤ 2050 ∧ is_perfect_cube (product_of_sums n) ∧ ∀ m : ℕ, (m ≤ 2050 ∧ is_perfect_cube (product_of_sums m)) → m ≤ 2016 := 
sorry

end max_n_perfect_cube_l71_71721


namespace number_of_students_in_first_group_l71_71563

def total_students : ℕ := 24
def second_group : ℕ := 8
def third_group : ℕ := 7
def fourth_group : ℕ := 4
def summed_other_groups : ℕ := second_group + third_group + fourth_group
def students_first_group : ℕ := total_students - summed_other_groups

theorem number_of_students_in_first_group :
  students_first_group = 5 :=
by
  -- proof required here
  sorry

end number_of_students_in_first_group_l71_71563


namespace number_of_students_l71_71119

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N)
  (h2 : 95 * (N - 5) = T - 100) : N = 25 :=
by
  sorry

end number_of_students_l71_71119


namespace trapezoid_area_l71_71783

noncomputable def area_trapezoid : ℝ :=
  let x1 := 10
  let x2 := -10
  let y1 := 10
  let h := 10
  let a := 20  -- length of top side at y = 10
  let b := 10  -- length of lower side
  (a + b) * h / 2

theorem trapezoid_area : area_trapezoid = 150 := by
  sorry

end trapezoid_area_l71_71783


namespace maximum_cars_quotient_l71_71182

theorem maximum_cars_quotient
  (car_length : ℕ) (m_speed : ℕ) (half_hour_distance : ℕ) 
  (unit_length : ℕ) (max_units : ℕ) (N : ℕ) :
  (car_length = 5) →
  (half_hour_distance = 10000) →
  (unit_length = 5 * (m_speed + 1)) →
  (max_units = half_hour_distance / unit_length) →
  (N = max_units) →
  (N / 10 = 200) :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end maximum_cars_quotient_l71_71182


namespace area_of_tangency_triangle_l71_71824

theorem area_of_tangency_triangle 
  (r1 r2 r3 : ℝ) 
  (h1 : r1 = 2)
  (h2 : r2 = 3)
  (h3 : r3 = 4) 
  (mutually_tangent : ∀ {c1 c2 c3 : ℝ}, c1 + c2 = r1 + r2 ∧ c2 + c3 = r2 + r3 ∧ c1 + c3 = r1 + r3 ) :
  ∃ area : ℝ, area = 3 * (Real.sqrt 6) / 2 :=
by
  sorry

end area_of_tangency_triangle_l71_71824


namespace compute_expression_l71_71855

theorem compute_expression : -8 * 4 - (-6 * -3) + (-10 * -5) = 0 := by sorry

end compute_expression_l71_71855


namespace sales_discount_l71_71474

theorem sales_discount
  (P N : ℝ)  -- original price and number of items sold
  (H1 : (1 - D / 100) * 1.3 = 1.17) -- condition when discount D is applied
  (D : ℝ)  -- sales discount percentage
  : D = 10 := by
  sorry

end sales_discount_l71_71474


namespace quadratic_square_binomial_l71_71388

theorem quadratic_square_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 18 * x + k = (x + b) ^ 2) ↔ k = 81 := by
  sorry

end quadratic_square_binomial_l71_71388


namespace lcm_10_to_30_l71_71677

def list_of_ints := [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

def lcm_of_list (l : List Nat) : Nat :=
  l.foldr Nat.lcm 1

theorem lcm_10_to_30 : lcm_of_list list_of_ints = 232792560 :=
  sorry

end lcm_10_to_30_l71_71677


namespace find_a_value_l71_71294

noncomputable def solve_for_a (y : ℝ) (a : ℝ) : Prop :=
  0 < y ∧ (a * y) / 20 + (3 * y) / 10 = 0.6499999999999999 * y 

theorem find_a_value (y : ℝ) (a : ℝ) (h : solve_for_a y a) : a = 7 := 
by 
  sorry

end find_a_value_l71_71294


namespace find_pairs_l71_71642

theorem find_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0)
  (cond1 : (m^2 - n) ∣ (m + n^2))
  (cond2 : (n^2 - m) ∣ (n + m^2)) :
  (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) := 
sorry

end find_pairs_l71_71642


namespace triangle_angle_contradiction_l71_71378

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180)
(h2 : α > 60) (h3 : β > 60) (h4 : γ > 60) : false :=
sorry

end triangle_angle_contradiction_l71_71378


namespace lindsay_dolls_l71_71699

theorem lindsay_dolls (B B_b B_k : ℕ) 
  (h1 : B_b = 4 * B)
  (h2 : B_k = 4 * B - 2)
  (h3 : B_b + B_k = B + 26) : B = 4 :=
by
  sorry

end lindsay_dolls_l71_71699


namespace simplify_expression_l71_71708

variable {a b c : ℝ}

-- Assuming the conditions specified in the problem
def valid_conditions (a b c : ℝ) : Prop := (1 - a * b ≠ 0) ∧ (1 + c * a ≠ 0)

theorem simplify_expression (h : valid_conditions a b c) :
  (a + b) / (1 - a * b) + (c - a) / (1 + c * a) / 
  (1 - ((a + b) / (1 - a * b) * (c - a) / (1 + c * a))) = 
  (b + c) / (1 - b * c) := 
sorry

end simplify_expression_l71_71708


namespace ascorbic_acid_molecular_weight_l71_71302

theorem ascorbic_acid_molecular_weight (C H O : ℕ → ℝ)
  (C_weight : C 6 = 6 * 12.01)
  (H_weight : H 8 = 8 * 1.008)
  (O_weight : O 6 = 6 * 16.00)
  (total_mass_given : 528 = 6 * 12.01 + 8 * 1.008 + 6 * 16.00) :
  6 * 12.01 + 8 * 1.008 + 6 * 16.00 = 176.124 := 
by 
  sorry

end ascorbic_acid_molecular_weight_l71_71302


namespace radius_of_inscribed_circle_l71_71617

theorem radius_of_inscribed_circle (a b c r : ℝ) (h : a^2 + b^2 = c^2) :
  r = a + b - c :=
sorry

end radius_of_inscribed_circle_l71_71617


namespace person_age_in_1893_l71_71597

theorem person_age_in_1893 
    (x y : ℕ)
    (h1 : 0 ≤ x ∧ x < 10)
    (h2 : 0 ≤ y ∧ y < 10)
    (h3 : 1 + 8 + x + y = 93 - 10 * x - y) : 
    1893 - (1800 + 10 * x + y) = 24 :=
by
  sorry

end person_age_in_1893_l71_71597


namespace total_pizzas_two_days_l71_71487

-- Declare variables representing the number of pizzas made by Craig and Heather
variables (c1 c2 h1 h2 : ℕ)

-- Given conditions as definitions
def condition_1 : Prop := h1 = 4 * c1
def condition_2 : Prop := h2 = c2 - 20
def condition_3 : Prop := c1 = 40
def condition_4 : Prop := c2 = c1 + 60

-- Prove that the total number of pizzas made in two days equals 380
theorem total_pizzas_two_days (c1 c2 h1 h2 : ℕ)
  (h_cond1 : condition_1 c1 h1) (h_cond2 : condition_2 h2 c2)
  (h_cond3 : condition_3 c1) (h_cond4 : condition_4 c1 c2) :
  (h1 + c1) + (h2 + c2) = 380 :=
by
  -- As we don't need to provide the proof here, just insert sorry
  sorry

end total_pizzas_two_days_l71_71487


namespace average_price_of_goat_l71_71596

theorem average_price_of_goat (total_cost_goats_hens : ℕ) (num_goats num_hens : ℕ) (avg_price_hen : ℕ)
  (h1 : total_cost_goats_hens = 2500) (h2 : num_hens = 10) (h3 : avg_price_hen = 50) (h4 : num_goats = 5) :
  (total_cost_goats_hens - num_hens * avg_price_hen) / num_goats = 400 :=
sorry

end average_price_of_goat_l71_71596


namespace height_of_old_lamp_l71_71926

theorem height_of_old_lamp (height_new_lamp : ℝ) (height_difference : ℝ) (h : height_new_lamp = 2.33) (h_diff : height_difference = 1.33) : 
  (height_new_lamp - height_difference) = 1.00 :=
by
  have height_new : height_new_lamp = 2.33 := h
  have height_diff : height_difference = 1.33 := h_diff
  sorry

end height_of_old_lamp_l71_71926


namespace inequality_proof_l71_71125

theorem inequality_proof (a b : ℝ) : 
  (a^4 + a^2 * b^2 + b^4) / 3 ≥ (a^3 * b + b^3 * a) / 2 :=
by
  sorry

end inequality_proof_l71_71125


namespace surface_area_of_tunneled_cube_l71_71800

-- Definition of the initial cube and its properties.
def cube (side_length : ℕ) := side_length * side_length * side_length

-- Initial side length of the large cube
def large_cube_side : ℕ := 12

-- Each small cube side length
def small_cube_side : ℕ := 3

-- Number of small cubes that fit into the large cube
def num_small_cubes : ℕ := (cube large_cube_side) / (cube small_cube_side)

-- Number of cubes removed initially
def removed_cubes : ℕ := 27

-- Number of remaining cubes after initial removal
def remaining_cubes : ℕ := num_small_cubes - removed_cubes

-- Surface area of each unmodified small cube
def small_cube_surface : ℕ := 54

-- Additional surface area due to removal of center units
def additional_surface : ℕ := 24

-- Surface area of each modified small cube
def modified_cube_surface : ℕ := small_cube_surface + additional_surface

-- Total surface area before adjustment for shared faces
def total_surface_before_adjustment : ℕ := remaining_cubes * modified_cube_surface

-- Shared surface area to be subtracted
def shared_surface : ℕ := 432

-- Final surface area of the resulting figure
def final_surface_area : ℕ := total_surface_before_adjustment - shared_surface

-- Theorem statement
theorem surface_area_of_tunneled_cube : final_surface_area = 2454 :=
by {
  -- Proof required here
  sorry
}

end surface_area_of_tunneled_cube_l71_71800


namespace sara_ate_16_apples_l71_71970

theorem sara_ate_16_apples (S : ℕ) (h1 : ∃ (A : ℕ), A = 4 * S ∧ S + A = 80) : S = 16 :=
by
  obtain ⟨A, h2, h3⟩ := h1
  sorry

end sara_ate_16_apples_l71_71970


namespace solve_fraction_equation_l71_71274

theorem solve_fraction_equation (t : ℝ) (h₀ : t ≠ 6) (h₁ : t ≠ -4) :
  (t = -2 ∨ t = -5) ↔ (t^2 - 3 * t - 18) / (t - 6) = 2 / (t + 4) := 
by
  sorry

end solve_fraction_equation_l71_71274


namespace complement_union_l71_71854

open Set

universe u

variable {U : Type u} [Fintype U] [DecidableEq U]
variable {A B : Set U}

def complement (s : Set U) : Set U := {x | x ∉ s}

theorem complement_union {U : Set ℕ} (A B : Set ℕ) 
  (h1 : complement A ∩ B = {1})
  (h2 : A ∩ B = {3})
  (h3 : complement A ∩ complement B = {2}) :
  complement (A ∪ B) = {2} :=
by sorry

end complement_union_l71_71854


namespace arithmetic_sequence_fifth_term_l71_71030

theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℕ), (∀ n, a n.succ = a n + 2) → a 1 = 2 → a 5 = 10 :=
by
  intros a h1 h2
  sorry

end arithmetic_sequence_fifth_term_l71_71030


namespace num_triangles_in_n_gon_l71_71160

-- Definitions for the problem in Lean based on provided conditions
def n_gon (n : ℕ) : Type := sorry  -- Define n-gon as a polygon with n sides
def non_intersecting_diagonals (n : ℕ) : Prop := sorry  -- Define the property of non-intersecting diagonals in an n-gon
def num_triangles (n : ℕ) : ℕ := sorry  -- Define a function to calculate the number of triangles formed by the diagonals in an n-gon

-- Statement of the theorem to prove
theorem num_triangles_in_n_gon (n : ℕ) (h : non_intersecting_diagonals n) : num_triangles n = n - 2 :=
by
  sorry

end num_triangles_in_n_gon_l71_71160


namespace initial_books_calculation_l71_71620

-- Definitions based on conditions
def total_books : ℕ := 77
def additional_books : ℕ := 23

-- Statement of the problem
theorem initial_books_calculation : total_books - additional_books = 54 :=
by
  sorry

end initial_books_calculation_l71_71620


namespace group_interval_eq_l71_71691

noncomputable def group_interval (a b m h : ℝ) : ℝ := abs (a - b)

theorem group_interval_eq (a b m h : ℝ) 
  (h1 : h = m / abs (a - b)) :
  abs (a - b) = m / h := 
by 
  sorry

end group_interval_eq_l71_71691


namespace sine_theorem_l71_71978

theorem sine_theorem (a b c α β γ : ℝ) 
  (h1 : a / Real.sin α = b / Real.sin β) 
  (h2 : b / Real.sin β = c / Real.sin γ) 
  (h3 : α + β + γ = Real.pi) :
  a = b * Real.cos γ + c * Real.cos β ∧
  b = c * Real.cos α + a * Real.cos γ ∧
  c = a * Real.cos β + b * Real.cos α :=
by
  sorry

end sine_theorem_l71_71978


namespace min_knights_proof_l71_71770

-- Noncomputable theory as we are dealing with existence proofs
noncomputable def min_knights (n : ℕ) : ℕ :=
  -- Given the table contains 1001 people
  if n = 1001 then 502 else 0

-- The proof problem statement, we need to ensure that minimum number of knights is 502
theorem min_knights_proof : min_knights 1001 = 502 := 
  by
    -- Sketch of proof: Deriving that the minimum number of knights must be 502 based on the problem constraints
    sorry

end min_knights_proof_l71_71770


namespace price_of_one_liter_l71_71945

theorem price_of_one_liter
  (total_cost : ℝ) (num_bottles : ℝ) (liters_per_bottle : ℝ)
  (H : total_cost = 12 ∧ num_bottles = 6 ∧ liters_per_bottle = 2) :
  total_cost / (num_bottles * liters_per_bottle) = 1 :=
by
  sorry

end price_of_one_liter_l71_71945


namespace find_integer_n_l71_71232

theorem find_integer_n :
  ∃ n : ℕ, 0 ≤ n ∧ n < 201 ∧ 200 * n ≡ 144 [MOD 101] ∧ n = 29 := 
by
  sorry

end find_integer_n_l71_71232


namespace solution_of_inequality_l71_71166

theorem solution_of_inequality (x : ℝ) : x * (x - 1) < 2 ↔ -1 < x ∧ x < 2 :=
sorry

end solution_of_inequality_l71_71166


namespace jack_jill_meet_distance_l71_71793

theorem jack_jill_meet_distance : 
  ∀ (total_distance : ℝ) (uphill_distance : ℝ) (headstart : ℝ) 
  (jack_speed_up : ℝ) (jack_speed_down : ℝ)
  (jill_speed_up : ℝ) (jill_speed_down : ℝ), 
  total_distance = 12 → 
  uphill_distance = 6 → 
  headstart = 1 / 4 → 
  jack_speed_up = 12 → 
  jack_speed_down = 18 → 
  jill_speed_up = 14 → 
  jill_speed_down = 20 → 
  ∃ meet_position : ℝ, meet_position = 15.75 :=
by
  sorry

end jack_jill_meet_distance_l71_71793


namespace length_of_larger_cuboid_l71_71003

theorem length_of_larger_cuboid
  (n : ℕ)
  (l_small : ℝ) (w_small : ℝ) (h_small : ℝ)
  (w_large : ℝ) (h_large : ℝ)
  (V_large : ℝ)
  (n_eq : n = 56)
  (dim_small : l_small = 5 ∧ w_small = 3 ∧ h_small = 2)
  (dim_large : w_large = 14 ∧ h_large = 10)
  (V_large_eq : V_large = n * (l_small * w_small * h_small)) :
  ∃ l_large : ℝ, l_large = V_large / (w_large * h_large) ∧ l_large = 12 := by
  sorry

end length_of_larger_cuboid_l71_71003


namespace product_of_roots_l71_71887

theorem product_of_roots :
  (let a := 36
   let b := -24
   let c := -120
   a ≠ 0) →
  let roots_product := c / a
  roots_product = -10/3 :=
by
  sorry

end product_of_roots_l71_71887


namespace triangle_property_l71_71424

theorem triangle_property
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a > b)
  (h2 : a = 5)
  (h3 : c = 6)
  (h4 : Real.sin B = 3 / 5) :
  (b = Real.sqrt 13 ∧ Real.sin A = 3 * Real.sqrt 13 / 13) →
  Real.sin (2 * A + π / 4) = 7 * Real.sqrt 2 / 26 :=
sorry

end triangle_property_l71_71424


namespace ratio_of_sold_phones_to_production_l71_71437

def last_years_production : ℕ := 5000
def this_years_production : ℕ := 2 * last_years_production
def phones_left_in_factory : ℕ := 7500
def sold_phones : ℕ := this_years_production - phones_left_in_factory

theorem ratio_of_sold_phones_to_production : 
  (sold_phones : ℚ) / this_years_production = 1 / 4 := 
by
  sorry

end ratio_of_sold_phones_to_production_l71_71437


namespace combined_salaries_correct_l71_71847

noncomputable def combined_salaries_BCDE (A B C D E : ℕ) : Prop :=
  (A = 8000) →
  ((A + B + C + D + E) / 5 = 8600) →
  (B + C + D + E = 35000)

theorem combined_salaries_correct 
  (A B C D E : ℕ) 
  (hA : A = 8000) 
  (havg : (A + B + C + D + E) / 5 = 8600) : 
  B + C + D + E = 35000 :=
sorry

end combined_salaries_correct_l71_71847


namespace previous_job_salary_is_correct_l71_71807

-- Define the base salary and commission structure.
def base_salary_new_job : ℝ := 45000
def commission_rate : ℝ := 0.15
def sale_amount : ℝ := 750
def minimum_sales : ℝ := 266.67

-- Define the total salary from the new job with the minimum sales.
def new_job_total_salary : ℝ :=
  base_salary_new_job + (commission_rate * sale_amount * minimum_sales)

-- Define Tom's previous job's salary.
def previous_job_salary : ℝ := 75000

-- Prove that Tom's previous job salary matches the new job total salary with the minimum sales.
theorem previous_job_salary_is_correct :
  (new_job_total_salary = previous_job_salary) :=
by
  -- This is where you would include the proof steps, but it's sufficient to put 'sorry' for now.
  sorry

end previous_job_salary_is_correct_l71_71807


namespace fewest_printers_l71_71684

theorem fewest_printers (x y : ℕ) (h : 8 * x = 7 * y) : x + y = 15 :=
sorry

end fewest_printers_l71_71684


namespace rowing_speed_downstream_l71_71573

/--
A man can row upstream at 25 kmph and downstream at a certain speed. 
The speed of the man in still water is 30 kmph. 
Prove that the speed of the man rowing downstream is 35 kmph.
-/
theorem rowing_speed_downstream (V_u V_sw V_s V_d : ℝ)
  (h1 : V_u = 25) 
  (h2 : V_sw = 30) 
  (h3 : V_u = V_sw - V_s) 
  (h4 : V_d = V_sw + V_s) :
  V_d = 35 :=
by
  sorry

end rowing_speed_downstream_l71_71573


namespace find_radius_of_circle_B_l71_71220

noncomputable def radius_of_circle_B : Real :=
  sorry

theorem find_radius_of_circle_B :
  let A := 2
  let R := 4
  -- Define x as the horizontal distance (FG) and y as the vertical distance (GH)
  ∃ (x y : Real), 
  (y = x + (x^2 / 2)) ∧
  (y = 2 - (x^2 / 4)) ∧
  (5 * x^2 + 4 * x - 8 = 0) ∧
  -- Contains only the positive solution among possible valid radii
  (radius_of_circle_B = (22 / 25) + (2 * Real.sqrt 11 / 25))
:= 
sorry

end find_radius_of_circle_B_l71_71220


namespace purchasing_methods_count_l71_71368

def material_cost : ℕ := 40
def instrument_cost : ℕ := 60
def budget : ℕ := 400
def min_materials : ℕ := 4
def min_instruments : ℕ := 2

theorem purchasing_methods_count : 
  (∃ (n_m m : ℕ), 
    n_m ≥ min_materials ∧ m ≥ min_instruments ∧ 
    n_m * material_cost + m * instrument_cost ≤ budget) → 
  (∃ (count : ℕ), count = 7) :=
by 
  sorry

end purchasing_methods_count_l71_71368


namespace smallest_solution_l71_71094

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l71_71094


namespace smallest_number_exists_l71_71638

theorem smallest_number_exists (x : ℤ) :
  (x + 3) % 18 = 0 ∧ 
  (x + 3) % 70 = 0 ∧ 
  (x + 3) % 100 = 0 ∧ 
  (x + 3) % 84 = 0 → 
  x = 6297 :=
by
  sorry

end smallest_number_exists_l71_71638


namespace find_volume_of_pyramid_l71_71725

noncomputable def volume_of_pyramid
  (a : ℝ) (α : ℝ)
  (h1 : 0 < a) 
  (h2 : 0 < α ∧ α < π) 
  (h3 : ∀ θ, θ = α ∨ θ = π - α ∨ θ = 2 * π - α) : ℝ :=
  (a ^ 3 * abs (Real.cos α)) / 3

--and the theorem to prove the statement
theorem find_volume_of_pyramid
  (a α : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < α ∧ α < π) 
  (h3 : ∀ θ, θ = α ∨ θ = π - α ∨ θ = 2 * π - α) :
  volume_of_pyramid a α h1 h2 h3 = (a ^ 3 * abs (Real.cos α)) / 3 :=
sorry

end find_volume_of_pyramid_l71_71725


namespace intersection_of_intervals_l71_71320

theorem intersection_of_intervals (m n x : ℝ) (h1 : -1 < m) (h2 : m < 0) (h3 : 0 < n) :
  (m < x ∧ x < n) ∧ (-1 < x ∧ x < 0) ↔ -1 < x ∧ x < 0 :=
by sorry

end intersection_of_intervals_l71_71320


namespace total_interest_correct_l71_71595

-- Initial conditions
def initial_investment : ℝ := 1200
def annual_interest_rate : ℝ := 0.08
def additional_deposit : ℝ := 500
def first_period : ℕ := 2
def second_period : ℕ := 2

-- Calculate the accumulated value after the first period
def first_accumulated_value : ℝ := initial_investment * (1 + annual_interest_rate)^first_period

-- Calculate the new principal after additional deposit
def new_principal := first_accumulated_value + additional_deposit

-- Calculate the accumulated value after the second period
def final_value := new_principal * (1 + annual_interest_rate)^second_period

-- Calculate the total interest earned after 4 years
def total_interest_earned := final_value - initial_investment - additional_deposit

-- Final theorem statement to be proven
theorem total_interest_correct : total_interest_earned = 515.26 :=
by sorry

end total_interest_correct_l71_71595


namespace negative_linear_correlation_l71_71533

theorem negative_linear_correlation (x y : ℝ) (h : y = 3 - 2 * x) : 
  ∃ c : ℝ, c < 0 ∧ y = 3 + c * x := 
by  
  sorry

end negative_linear_correlation_l71_71533


namespace p_and_not_q_l71_71301

def p : Prop :=
  ∀ x : ℕ, x > 0 → (1 / 2) ^ x ≥ (1 / 3) ^ x

def q : Prop :=
  ∃ x : ℕ, x > 0 ∧ 2^x + 2^(1-x) = 2 * Real.sqrt 2

theorem p_and_not_q : p ∧ ¬q :=
by
  have h_p : p := sorry
  have h_not_q : ¬q := sorry
  exact ⟨h_p, h_not_q⟩

end p_and_not_q_l71_71301


namespace evaluate_expression_l71_71502

theorem evaluate_expression : (2^3002 * 3^3004) / (6^3003) = (3 / 2) := by
  sorry

end evaluate_expression_l71_71502


namespace divisible_by_65_l71_71827

theorem divisible_by_65 (n : ℕ) : 65 ∣ (5^n * (2^(2*n) - 3^n) + 2^n - 7^n) :=
sorry

end divisible_by_65_l71_71827


namespace smallest_positive_e_for_polynomial_l71_71102

theorem smallest_positive_e_for_polynomial :
  ∃ a b c d e : ℤ, e = 168 ∧
  (a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e = 0) ∧
  (a * (x + 3) * (x - 7) * (x - 8) * (4 * x + 1) = a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e) := sorry

end smallest_positive_e_for_polynomial_l71_71102


namespace square_possible_length_l71_71550

theorem square_possible_length (sticks : Finset ℕ) (H : sticks = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ∃ s, s = 9 ∧
  ∃ (a b c : ℕ), a ∈ sticks ∧ b ∈ sticks ∧ c ∈ sticks ∧ a + b + c = 9 :=
by
  sorry

end square_possible_length_l71_71550


namespace part1_part2_l71_71041

variable (a b : ℝ)
def A : ℝ := 2 * a * b - a
def B : ℝ := -a * b + 2 * a + b

theorem part1 : 5 * A a b - 2 * B a b = 12 * a * b - 9 * a - 2 * b := by
  sorry

theorem part2 : (∀ b : ℝ, 5 * A a b - 2 * B a b = 12 * a * b - 9 * a - 2 * b) -> a = 1 / 6 := by
  sorry

end part1_part2_l71_71041


namespace katherine_age_l71_71733

-- Define a Lean statement equivalent to the given problem
theorem katherine_age (K M : ℕ) (h1 : M = K - 3) (h2 : M = 21) : K = 24 := sorry

end katherine_age_l71_71733


namespace find_circle_center_l71_71185

theorem find_circle_center : ∃ (h k : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 2*x + 12*y + 1 = 0 ↔ (x - h)^2 + (y - k)^2 = 36) ∧ h = 1 ∧ k = -6 := 
sorry

end find_circle_center_l71_71185


namespace A_remaining_time_equals_B_remaining_time_l71_71823

variable (d_A d_B remaining_Distance_A remaining_Time_A remaining_Distance_B remaining_Time_B total_Distance : ℝ)

-- Given conditions as definitions
def A_traveled_more : d_A = d_B + 180 := sorry
def total_distance_between_X_Y : total_Distance = 900 := sorry
def sum_distance_traveled : d_A + d_B = total_Distance := sorry
def B_remaining_time : remaining_Time_B = 4.5 := sorry
def B_remaining_distance : remaining_Distance_B = total_Distance - d_B := sorry

-- Prove that: A travels the same remaining distance in the same time as B
theorem A_remaining_time_equals_B_remaining_time :
  remaining_Distance_A = remaining_Distance_B ∧ remaining_Time_A = remaining_Time_B := sorry

end A_remaining_time_equals_B_remaining_time_l71_71823


namespace eight_diamond_three_l71_71446

def diamond (x y : ℤ) : ℤ := sorry

axiom diamond_zero (x : ℤ) : diamond x 0 = x
axiom diamond_comm (x y : ℤ) : diamond x y = diamond y x
axiom diamond_recursive (x y : ℤ) : diamond (x + 2) y = diamond x y + 2 * y + 3

theorem eight_diamond_three : diamond 8 3 = 39 :=
sorry

end eight_diamond_three_l71_71446


namespace finite_set_elements_at_least_half_m_l71_71813

theorem finite_set_elements_at_least_half_m (m : ℕ) (A : Finset ℤ) (B : ℕ → Finset ℤ) 
  (hm : 2 ≤ m) 
  (hB : ∀ k : ℕ, 1 ≤ k → k ≤ m → (B k).sum id = (m : ℤ) ^ k) : 
  ∃ n : ℕ, (A.card ≥ n) ∧ (n ≥ m / 2) :=
by
  sorry

end finite_set_elements_at_least_half_m_l71_71813


namespace wire_length_l71_71044

variable (L M l a : ℝ) -- Assume these variables are real numbers.

theorem wire_length (h1 : a ≠ 0) : L = (M / a) * l :=
sorry

end wire_length_l71_71044


namespace sum_of_inserted_numbers_in_progressions_l71_71521

theorem sum_of_inserted_numbers_in_progressions (x y : ℝ) (hx : 4 * (y / x) = x) (hy : 2 * y = x + 64) :
  x + y = 131 + 3 * Real.sqrt 129 :=
by
  sorry

end sum_of_inserted_numbers_in_progressions_l71_71521


namespace coordinates_of_point_l71_71840

theorem coordinates_of_point (a : ℝ) (h : a - 3 = 0) : (a + 2, a - 3) = (5, 0) :=
by
  sorry

end coordinates_of_point_l71_71840


namespace distribution_count_l71_71841

def num_distributions (novels poetry students : ℕ) : ℕ :=
  -- This is where the formula for counting would go, but we'll just define it as sorry for now
  sorry

theorem distribution_count : num_distributions 3 2 4 = 28 :=
by
  sorry

end distribution_count_l71_71841


namespace equivalent_operation_l71_71394

theorem equivalent_operation (x : ℚ) :
  (x / (5 / 6) * (4 / 7)) = x * (24 / 35) :=
by
  sorry

end equivalent_operation_l71_71394


namespace bridget_bakery_profit_l71_71656

theorem bridget_bakery_profit :
  let loaves := 36
  let cost_per_loaf := 1
  let morning_sale_price := 3
  let afternoon_sale_price := 1.5
  let late_afternoon_sale_price := 1
  
  let morning_loaves := (2/3 : ℝ) * loaves
  let morning_revenue := morning_loaves * morning_sale_price
  
  let remaining_after_morning := loaves - morning_loaves
  let afternoon_loaves := (1/2 : ℝ) * remaining_after_morning
  let afternoon_revenue := afternoon_loaves * afternoon_sale_price
  
  let late_afternoon_loaves := remaining_after_morning - afternoon_loaves
  let late_afternoon_revenue := late_afternoon_loaves * late_afternoon_sale_price
  
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue
  let total_cost := loaves * cost_per_loaf
  
  total_revenue - total_cost = 51 := by sorry

end bridget_bakery_profit_l71_71656


namespace gcd_m_n_l71_71278

namespace GCDProof

def m : ℕ := 33333333
def n : ℕ := 666666666

theorem gcd_m_n : gcd m n = 2 := 
  sorry

end GCDProof

end gcd_m_n_l71_71278


namespace minutes_spent_calling_clients_l71_71246

theorem minutes_spent_calling_clients
    (C : ℕ)
    (H1 : 7 * C + C = 560) :
    C = 70 :=
sorry

end minutes_spent_calling_clients_l71_71246


namespace rectangle_area_error_percentage_l71_71894

theorem rectangle_area_error_percentage (L W : ℝ) : 
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let actual_area := L * W
  let measured_area := measured_length * measured_width
  let error := measured_area - actual_area
  let error_percentage := (error / actual_area) * 100
  error_percentage = 0.7 := 
by
  sorry

end rectangle_area_error_percentage_l71_71894


namespace perfect_square_condition_l71_71379

theorem perfect_square_condition (n : ℤ) : 
    ∃ k : ℤ, n^2 + 6*n + 1 = k^2 ↔ n = 0 ∨ n = -6 := by
  sorry

end perfect_square_condition_l71_71379


namespace marked_price_l71_71729

theorem marked_price (P : ℝ)
  (h₁ : 20 / 100 = 0.20)
  (h₂ : 15 / 100 = 0.15)
  (h₃ : 5 / 100 = 0.05)
  (h₄ : 7752 = 0.80 * 0.85 * 0.95 * P)
  : P = 11998.76 := by
  sorry

end marked_price_l71_71729


namespace inequality_proof_l71_71273

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^2 / (a + b)) + (b^2 / (b + c)) + (c^2 / (c + a)) ≥ (a + b + c) / 2 := 
by
  sorry

end inequality_proof_l71_71273


namespace xy_divides_x2_plus_y2_plus_one_l71_71848

theorem xy_divides_x2_plus_y2_plus_one 
    (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : (x * y) ∣ (x^2 + y^2 + 1)) :
  (x^2 + y^2 + 1) / (x * y) = 3 := by
  sorry

end xy_divides_x2_plus_y2_plus_one_l71_71848


namespace disk_difference_l71_71582

/-- Given the following conditions:
    1. Every disk is either blue, yellow, green, or red.
    2. The ratio of blue disks to yellow disks to green disks to red disks is 3 : 7 : 8 : 4.
    3. The total number of disks in the bag is 176.
    Prove that the number of green disks minus the number of blue disks is 40.
-/
theorem disk_difference (b y g r : ℕ) (h_ratio : b * 7 = y * 3 ∧ b * 8 = g * 3 ∧ b * 4 = r * 3) (h_total : b + y + g + r = 176) : g - b = 40 :=
by
  sorry

end disk_difference_l71_71582


namespace add_base3_numbers_l71_71605

theorem add_base3_numbers : 
  (2 + 1 * 3) + (0 + 2 * 3 + 1 * 3^2) + 
  (1 + 2 * 3 + 0 * 3^2 + 2 * 3^3) + (2 + 0 * 3 + 1 * 3^2 + 2 * 3^3)
  = 2 + 2 * 3 + 2 * 3^2 + 2 * 3^3 := 
by sorry

end add_base3_numbers_l71_71605


namespace first_term_of_geometric_sequence_l71_71215

theorem first_term_of_geometric_sequence (a r : ℕ) :
  (a * r ^ 3 = 54) ∧ (a * r ^ 4 = 162) → a = 2 :=
by
  -- Provided conditions and the goal
  sorry

end first_term_of_geometric_sequence_l71_71215


namespace nancy_balloons_l71_71724

variable (MaryBalloons : ℝ) (NancyBalloons : ℝ)

theorem nancy_balloons (h1 : NancyBalloons = 4 * MaryBalloons) (h2 : MaryBalloons = 1.75) : 
  NancyBalloons = 7 := 
by 
  sorry

end nancy_balloons_l71_71724


namespace range_of_alpha_l71_71575

theorem range_of_alpha :
  ∀ P : ℝ, 
  (∃ y : ℝ, y = 4 / (Real.exp P + 1)) →
  (∃ α : ℝ, α = Real.arctan (4 / (Real.exp P + 2 + 1 / Real.exp P)) ∧ (Real.tan α) ∈ Set.Ico (-1) 0) → 
  Set.Ico (3 * Real.pi / 4) Real.pi :=
by
  sorry

end range_of_alpha_l71_71575


namespace area_of_billboard_l71_71331

variable (L W : ℕ) (P : ℕ)
variable (hW : W = 8) (hP : P = 46)

theorem area_of_billboard (h1 : P = 2 * L + 2 * W) : L * W = 120 :=
by
  sorry

end area_of_billboard_l71_71331


namespace area_of_lune_l71_71205

theorem area_of_lune :
  let d1 := 2
  let d2 := 4
  let r1 := d1 / 2
  let r2 := d2 / 2
  let height := r2 - r1
  let area_triangle := (1 / 2) * d1 * height
  let area_semicircle_small := (1 / 2) * π * r1^2
  let area_combined := area_triangle + area_semicircle_small
  let area_sector_large := (1 / 4) * π * r2^2
  let area_lune := area_combined - area_sector_large
  area_lune = 1 - (1 / 2) * π := 
by
  sorry

end area_of_lune_l71_71205


namespace average_salary_for_company_l71_71324

theorem average_salary_for_company
    (number_of_managers : Nat)
    (number_of_associates : Nat)
    (average_salary_managers : Nat)
    (average_salary_associates : Nat)
    (hnum_managers : number_of_managers = 15)
    (hnum_associates : number_of_associates = 75)
    (has_managers : average_salary_managers = 90000)
    (has_associates : average_salary_associates = 30000) : 
    (number_of_managers * average_salary_managers + number_of_associates * average_salary_associates) / 
    (number_of_managers + number_of_associates) = 40000 := 
    by
    sorry

end average_salary_for_company_l71_71324


namespace linemen_count_l71_71291

-- Define the initial conditions
def linemen_drink := 8
def skill_position_players_drink := 6
def total_skill_position_players := 10
def cooler_capacity := 126
def skill_position_players_drink_first := 5

-- Define the number of ounces drunk by skill position players during the first break
def skill_position_players_first_break := skill_position_players_drink_first * skill_position_players_drink

-- Define the theorem stating that the number of linemen (L) is 12 given the conditions
theorem linemen_count :
  ∃ L : ℕ, linemen_drink * L + skill_position_players_first_break = cooler_capacity ∧ L = 12 :=
by {
  sorry -- Proof to be provided.
}

end linemen_count_l71_71291


namespace num_of_int_solutions_l71_71221

/-- 
  The number of integer solutions to the equation 
  \((x^3 - x - 1)^{2015} = 1\) is 3.
-/
theorem num_of_int_solutions :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℤ, (x ^ 3 - x - 1) ^ 2015 = 1 ↔ x = 0 ∨ x = 1 ∨ x = -1 := 
sorry

end num_of_int_solutions_l71_71221


namespace stamps_count_l71_71621

theorem stamps_count {x : ℕ} (h1 : x % 3 = 1) (h2 : x % 5 = 3) (h3 : x % 7 = 5) (h4 : 150 < x ∧ x ≤ 300) :
  x = 208 :=
sorry

end stamps_count_l71_71621


namespace prob_next_black_ball_l71_71692

theorem prob_next_black_ball
  (total_balls : ℕ := 100) 
  (black_balls : Fin 101) 
  (next_black_ball_probability : ℚ := 2 / 3) :
  black_balls.val ≤ total_balls →
  ∃ p q : ℕ, Nat.gcd p q = 1 ∧ (p : ℚ) / q = next_black_ball_probability ∧ p + q = 5 :=
by
  intros h
  use 2, 3
  repeat { sorry }

end prob_next_black_ball_l71_71692


namespace roger_collected_nickels_l71_71011

theorem roger_collected_nickels 
  (N : ℕ)
  (initial_pennies : ℕ := 42) 
  (initial_dimes : ℕ := 15)
  (donated_coins : ℕ := 66)
  (left_coins : ℕ := 27)
  (h_total_coins_initial : initial_pennies + N + initial_dimes - donated_coins = left_coins) :
  N = 36 := 
sorry

end roger_collected_nickels_l71_71011


namespace trapezoid_shorter_base_length_l71_71033

theorem trapezoid_shorter_base_length (longer_base : ℕ) (segment_length : ℕ) (shorter_base : ℕ) 
  (h1 : longer_base = 120) (h2 : segment_length = 7)
  (h3 : segment_length = (longer_base - shorter_base) / 2) : 
  shorter_base = 106 := by
  sorry

end trapezoid_shorter_base_length_l71_71033


namespace negation_of_forall_pos_l71_71047

open Real

theorem negation_of_forall_pos (h : ∀ x : ℝ, x^2 - x + 1 > 0) : 
  ¬(∀ x : ℝ, x^2 - x + 1 > 0) ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry

end negation_of_forall_pos_l71_71047


namespace book_length_ratio_is_4_l71_71406

-- Define the initial conditions
def pages_when_6 : ℕ := 8
def age_when_start := 6
def multiple_at_twice_age := 5
def multiple_eight_years_after := 3
def current_pages : ℕ := 480

def pages_when_12 := pages_when_6 * multiple_at_twice_age
def pages_when_20 := pages_when_12 * multiple_eight_years_after

theorem book_length_ratio_is_4 :
  (current_pages : ℚ) / pages_when_20 = 4 := by
  -- We need to show the proof for the equality
  sorry

end book_length_ratio_is_4_l71_71406


namespace rotate_image_eq_A_l71_71997

def image_A : Type := sorry -- Image data for option (A)
def original_image : Type := sorry -- Original image data

def rotate_90_clockwise (img : Type) : Type := sorry -- Function to rotate image 90 degrees clockwise

theorem rotate_image_eq_A :
  rotate_90_clockwise original_image = image_A :=
sorry

end rotate_image_eq_A_l71_71997


namespace greatest_distance_is_correct_l71_71049

-- Define the coordinates of the post.
def post_coordinate : ℝ × ℝ := (6, -2)

-- Define the length of the rope.
def rope_length : ℝ := 12

-- Define the origin.
def origin : ℝ × ℝ := (0, 0)

-- Define the formula to calculate the Euclidean distance between two points in ℝ².
noncomputable def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ := by
  sorry

-- Define the distance from the origin to the post.
noncomputable def distance_origin_to_post : ℝ := euclidean_distance origin post_coordinate

-- Define the greatest distance the dog can be from the origin.
noncomputable def greatest_distance_from_origin : ℝ := distance_origin_to_post + rope_length

-- Prove that the greatest distance the dog can be from the origin is 12 + 2 * sqrt 10.
theorem greatest_distance_is_correct : greatest_distance_from_origin = 12 + 2 * Real.sqrt 10 := by
  sorry

end greatest_distance_is_correct_l71_71049


namespace randy_initial_money_l71_71759

theorem randy_initial_money (X : ℕ) (h : X + 200 - 1200 = 2000) : X = 3000 :=
by {
  sorry
}

end randy_initial_money_l71_71759


namespace Queen_High_School_teachers_needed_l71_71062

def students : ℕ := 1500
def classes_per_student : ℕ := 6
def students_per_class : ℕ := 25
def classes_per_teacher : ℕ := 5

theorem Queen_High_School_teachers_needed : 
  (students * classes_per_student) / students_per_class / classes_per_teacher = 72 :=
by 
  sorry

end Queen_High_School_teachers_needed_l71_71062


namespace factorize_xy_squared_minus_x_l71_71527

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l71_71527


namespace sum_a_b_eq_neg2_l71_71718

def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

theorem sum_a_b_eq_neg2 (a b : ℝ) (h : f a + f b = 20) : a + b = -2 :=
by
  sorry

end sum_a_b_eq_neg2_l71_71718


namespace lychee_harvest_l71_71128

theorem lychee_harvest : 
  let last_year_red := 350
  let last_year_yellow := 490
  let this_year_red := 500
  let this_year_yellow := 700
  let sold_red := 2/3 * this_year_red
  let sold_yellow := 3/7 * this_year_yellow
  let remaining_red_after_sale := this_year_red - sold_red
  let remaining_yellow_after_sale := this_year_yellow - sold_yellow
  let family_ate_red := 3/5 * remaining_red_after_sale
  let family_ate_yellow := 4/9 * remaining_yellow_after_sale
  let remaining_red := remaining_red_after_sale - family_ate_red
  let remaining_yellow := remaining_yellow_after_sale - family_ate_yellow
  (this_year_red - last_year_red) / last_year_red * 100 = 42.86
  ∧ (this_year_yellow - last_year_yellow) / last_year_yellow * 100 = 42.86
  ∧ remaining_red = 67
  ∧ remaining_yellow = 223 :=
by
    intros
    sorry

end lychee_harvest_l71_71128


namespace at_least_12_lyamziks_rowed_l71_71583

-- Define the lyamziks, their weights, and constraints
def LyamzikWeight1 : ℕ := 7
def LyamzikWeight2 : ℕ := 14
def LyamzikWeight3 : ℕ := 21
def LyamzikWeight4 : ℕ := 28
def totalLyamziks : ℕ := LyamzikWeight1 + LyamzikWeight2 + LyamzikWeight3 + LyamzikWeight4
def boatCapacity : ℕ := 10
def maxRowsPerLyamzik : ℕ := 2

-- Question to prove
theorem at_least_12_lyamziks_rowed : totalLyamziks ≥ 12 :=
  by sorry


end at_least_12_lyamziks_rowed_l71_71583


namespace residue_mod_17_l71_71471

theorem residue_mod_17 : (230 * 15 - 20 * 9 + 5) % 17 = 0 :=
  by
  sorry

end residue_mod_17_l71_71471


namespace dogs_eat_each_day_l71_71636

theorem dogs_eat_each_day (h1 : 0.125 + 0.125 = 0.25) : true := by
  sorry

end dogs_eat_each_day_l71_71636


namespace max_sum_value_l71_71286

noncomputable def maxSum (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : ℤ :=
  i + j + k

theorem max_sum_value (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : 
  maxSum i j k h ≤ 77 :=
  sorry

end max_sum_value_l71_71286


namespace proof_problem_l71_71615

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x - 1)^2

theorem proof_problem : f (g (-3)) = 67 := 
by 
  sorry

end proof_problem_l71_71615


namespace intersection_complement_eq_l71_71509

def U : Set Int := { -2, -1, 0, 1, 2, 3 }
def M : Set Int := { 0, 1, 2 }
def N : Set Int := { 0, 1, 2, 3 }

noncomputable def C_U (A : Set Int) := U \ A

theorem intersection_complement_eq :
  (C_U M ∩ N) = {3} :=
by
  sorry

end intersection_complement_eq_l71_71509


namespace downstream_speed_l71_71740

noncomputable def upstream_speed : ℝ := 5
noncomputable def still_water_speed : ℝ := 15

theorem downstream_speed:
  ∃ (Vd : ℝ), Vd = 25 ∧ (still_water_speed = (upstream_speed + Vd) / 2) := 
sorry

end downstream_speed_l71_71740


namespace cyclist_distance_l71_71553

theorem cyclist_distance
  (x t d : ℝ)
  (h1 : d = x * t)
  (h2 : d = (x + 1) * (3 * t / 4))
  (h3 : d = (x - 1) * (t + 3)) :
  d = 18 :=
by {
  sorry
}

end cyclist_distance_l71_71553


namespace compare_trig_functions_l71_71883

theorem compare_trig_functions :
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c :=
by
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  sorry

end compare_trig_functions_l71_71883


namespace find_root_interval_l71_71343

noncomputable def f : ℝ → ℝ := sorry

theorem find_root_interval :
  f 2 < 0 ∧ f 3 > 0 ∧ f 2.5 < 0 ∧ f 2.75 > 0 ∧ f 2.625 > 0 ∧ f 2.5625 > 0 →
  ∃ x, 2.5 < x ∧ x < 2.5625 ∧ f x = 0 := sorry

end find_root_interval_l71_71343


namespace slices_with_both_pepperoni_and_mushrooms_l71_71035

theorem slices_with_both_pepperoni_and_mushrooms (n : ℕ)
  (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ)
  (all_have_topping : ∀ (s : ℕ), s < total_slices → s < pepperoni_slices ∨ s < mushroom_slices ∨ s < (total_slices - pepperoni_slices - mushroom_slices) )
  (total_condition : total_slices = 16)
  (pepperoni_condition : pepperoni_slices = 8)
  (mushroom_condition : mushroom_slices = 12) :
  (8 - n) + (12 - n) + n = 16 → n = 4 :=
sorry

end slices_with_both_pepperoni_and_mushrooms_l71_71035


namespace rabbits_in_cage_l71_71905

theorem rabbits_in_cage (rabbits_in_cage : ℕ) (rabbits_park : ℕ) : 
  rabbits_in_cage = 13 ∧ rabbits_park = 60 → (1/3 * rabbits_park - rabbits_in_cage) = 7 :=
by
  sorry

end rabbits_in_cage_l71_71905


namespace geometric_sequence_condition_l71_71880

-- Definition of a geometric sequence
def is_geometric_sequence (x y z : ℤ) : Prop :=
  y ^ 2 = x * z

-- Lean 4 statement based on the condition and correct answer tuple
theorem geometric_sequence_condition (a : ℤ) :
  is_geometric_sequence 4 a 9 ↔ (a = 6 ∨ a = -6) :=
by 
  sorry

end geometric_sequence_condition_l71_71880


namespace regular_octagon_side_length_l71_71876

theorem regular_octagon_side_length
  (side_length_pentagon : ℕ)
  (total_wire_length : ℕ)
  (side_length_octagon : ℕ) :
  side_length_pentagon = 16 →
  total_wire_length = 5 * side_length_pentagon →
  side_length_octagon = total_wire_length / 8 →
  side_length_octagon = 10 := 
sorry

end regular_octagon_side_length_l71_71876


namespace halfway_between_one_sixth_and_one_twelfth_is_one_eighth_l71_71241

theorem halfway_between_one_sixth_and_one_twelfth_is_one_eighth : 
  (1 / 6 + 1 / 12) / 2 = 1 / 8 := 
by
  sorry

end halfway_between_one_sixth_and_one_twelfth_is_one_eighth_l71_71241


namespace Eric_eggs_collected_l71_71150

theorem Eric_eggs_collected : 
  (∀ (chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (days : ℕ),
    chickens = 4 ∧ eggs_per_chicken_per_day = 3 ∧ days = 3 → 
    chickens * eggs_per_chicken_per_day * days = 36) :=
by
  sorry

end Eric_eggs_collected_l71_71150


namespace first_class_rate_l71_71999

def pass_rate : ℝ := 0.95
def cond_first_class_rate : ℝ := 0.20

theorem first_class_rate :
  (pass_rate * cond_first_class_rate) = 0.19 :=
by
  -- The proof is omitted as we're not required to provide it.
  sorry

end first_class_rate_l71_71999


namespace actual_road_length_l71_71409

theorem actual_road_length
  (scale_factor : ℕ → ℕ → Prop)
  (map_length_cm : ℕ)
  (actual_length_km : ℝ) : 
  (scale_factor 1 50000) →
  (map_length_cm = 15) →
  (actual_length_km = 7.5) :=
by
  sorry

end actual_road_length_l71_71409


namespace parallel_vectors_l71_71407

variables (x : ℝ)

theorem parallel_vectors (h : (1 + x) / 2 = (1 - 3 * x) / -1) : x = 3 / 5 :=
by {
  sorry
}

end parallel_vectors_l71_71407


namespace odd_function_condition_l71_71794

-- Definitions for real numbers and absolute value function
def f (x a b : ℝ) : ℝ := (x + a) * |x + b|

-- Theorem statement
theorem odd_function_condition (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = (x + a) * |x + b|) :
  (∀ x : ℝ, f x a b = -f (-x) a b) ↔ (a = 0 ∨ b = 0) :=
by
  sorry

end odd_function_condition_l71_71794


namespace coeff_x3_l71_71934

noncomputable def M (n : ℕ) : ℝ := (5 * (1:ℝ) - (1:ℝ)^(1/2)) ^ n
noncomputable def N (n : ℕ) : ℝ := 2 ^ n

theorem coeff_x3 (n : ℕ) (h : M n - N n = 240) : 
  (M 3) = 150 := sorry

end coeff_x3_l71_71934


namespace factor_polynomial_l71_71353

theorem factor_polynomial (x : ℝ) : 75 * x^5 - 300 * x^10 = 75 * x^5 * (1 - 4 * x^5) :=
by
  sorry

end factor_polynomial_l71_71353


namespace percentage_voting_for_biff_equals_45_l71_71837

variable (total : ℕ) (votingForMarty : ℕ) (undecidedPercent : ℝ)

theorem percentage_voting_for_biff_equals_45 :
  total = 200 →
  votingForMarty = 94 →
  undecidedPercent = 0.08 →
  let totalDecided := (1 - undecidedPercent) * total
  let votingForBiff := totalDecided - votingForMarty
  let votingForBiffPercent := (votingForBiff / total) * 100
  votingForBiffPercent = 45 :=
by
  intros h1 h2 h3
  let totalDecided := (1 - 0.08 : ℝ) * 200
  let votingForBiff := totalDecided - 94
  let votingForBiffPercent := (votingForBiff / 200) * 100
  sorry

end percentage_voting_for_biff_equals_45_l71_71837


namespace triangle_structure_twelve_rows_l71_71958

theorem triangle_structure_twelve_rows :
  let rods n := 3 * n * (n + 1) / 2
  let connectors n := n * (n + 1) / 2
  rods 12 + connectors 13 = 325 :=
by
  let rods n := 3 * n * (n + 1) / 2
  let connectors n := n * (n + 1) / 2
  sorry

end triangle_structure_twelve_rows_l71_71958


namespace exists_not_in_range_f_l71_71132

noncomputable def f : ℝ → ℕ :=
sorry

axiom functional_equation : ∀ (x y : ℝ), f (x + (1 / f y)) = f (y + (1 / f x))

theorem exists_not_in_range_f :
  ∃ n : ℕ, ∀ x : ℝ, f x ≠ n :=
sorry

end exists_not_in_range_f_l71_71132


namespace rectangle_width_l71_71809

-- Definitions and Conditions
variables (L W : ℕ)

-- Condition 1: The perimeter of the rectangle is 16 cm
def perimeter_eq : Prop := 2 * (L + W) = 16

-- Condition 2: The width is 2 cm longer than the length
def width_eq : Prop := W = L + 2

-- Proof Statement: Given the above conditions, the width of the rectangle is 5 cm
theorem rectangle_width (h1 : perimeter_eq L W) (h2 : width_eq L W) : W = 5 := 
by
  sorry

end rectangle_width_l71_71809


namespace change_received_correct_l71_71960

-- Define the conditions
def apples := 5
def cost_per_apple_cents := 80
def paid_dollars := 10

-- Convert the cost per apple to dollars
def cost_per_apple_dollars := (cost_per_apple_cents : ℚ) / 100

-- Calculate the total cost for 5 apples
def total_cost_dollars := apples * cost_per_apple_dollars

-- Calculate the change received
def change_received := paid_dollars - total_cost_dollars

-- Prove that the change received by Margie
theorem change_received_correct : change_received = 6 := by
  sorry

end change_received_correct_l71_71960


namespace clock_hands_angle_120_l71_71585

-- We are only defining the problem statement and conditions. No need for proof steps or calculations.

def angle_between_clock_hands (hour minute : ℚ) : ℚ :=
  abs ((30 * hour + minute / 2) - (6 * minute))

-- Given conditions
def time_in_range (hour : ℚ) (minute : ℚ) := 7 ≤ hour ∧ hour < 8

-- Problem statement to be proved
theorem clock_hands_angle_120 (hour minute : ℚ) :
  time_in_range hour minute → angle_between_clock_hands hour minute = 120 :=
sorry

end clock_hands_angle_120_l71_71585


namespace part_1_part_2_max_min_part_3_length_AC_part_4_range_a_l71_71515

-- Conditions
def quadratic (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2 * a * x + 2 * a
def point_A (a : ℝ) : ℝ × ℝ := (-1, quadratic a (-1))
def point_B (a : ℝ) : ℝ × ℝ := (3, quadratic a 3)
def line_EF (a : ℝ) : ℝ × ℝ × ℝ × ℝ := ((a - 1), -1, (2 * a + 3), -1)

-- Statements based on solution
theorem part_1 (a : ℝ) :
  (quadratic a (-1)) = -1 := sorry

theorem part_2_max_min (a : ℝ) : 
  a = 1 → 
  (∀ x, -2 ≤ x ∧ x ≤ 3 → 
    (quadratic 1 1 = 3 ∧ 
     quadratic 1 (-2) = -6 ∧ 
     quadratic 1 3 = -1)) := sorry

theorem part_3_length_AC (a : ℝ) (h : a > -1) :
  abs ((2 * a + 1) - (-1)) = abs ((2 * a + 2)) := sorry

theorem part_4_range_a (a : ℝ) : 
  quadratic a (a-1) = -1 ∧ quadratic a (2 * a + 3) = -1 → 
  a ∈ ({-2, -1} ∪ {b : ℝ | b ≥ 0}) := sorry

end part_1_part_2_max_min_part_3_length_AC_part_4_range_a_l71_71515


namespace chessboard_grains_difference_l71_71571

open BigOperators

def grains_on_square (k : ℕ) : ℕ := 2^k

def sum_of_first_n_squares (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), grains_on_square k

theorem chessboard_grains_difference : 
  grains_on_square 12 - sum_of_first_n_squares 10 = 2050 := 
by 
  -- Proof of the statement goes here.
  sorry

end chessboard_grains_difference_l71_71571


namespace proof_a_square_plus_a_plus_one_l71_71397

theorem proof_a_square_plus_a_plus_one (a : ℝ) (h : 2 * (5 - a) * (6 + a) = 100) : a^2 + a + 1 = -19 := 
by 
  sorry

end proof_a_square_plus_a_plus_one_l71_71397


namespace symmetric_line_equation_l71_71078

-- Define the given lines
def original_line (x y : ℝ) : Prop := y = 2 * x + 1
def line_of_symmetry (x y : ℝ) : Prop := y + 2 = 0

-- Define the problem statement as a theorem
theorem symmetric_line_equation :
  ∀ (x y : ℝ), line_of_symmetry x y → (original_line x (2 * (-2 - y) + 1)) ↔ (2 * x + y + 5 = 0) := 
sorry

end symmetric_line_equation_l71_71078


namespace square_area_l71_71154

theorem square_area 
  (s r l : ℝ)
  (h_r_s : r = s)
  (h_l_r : l = (2/5) * r)
  (h_area_rect : l * 10 = 120) : 
  s^2 = 900 := by
  -- Proof will go here
  sorry

end square_area_l71_71154


namespace average_chemistry_mathematics_l71_71345

noncomputable def marks (P C M B : ℝ) : Prop := 
  P + C + M + B = (P + B) + 180 ∧ P = 1.20 * B

theorem average_chemistry_mathematics 
  (P C M B : ℝ) (h : marks P C M B) : (C + M) / 2 = 90 :=
by
  sorry

end average_chemistry_mathematics_l71_71345


namespace cone_volume_difference_l71_71191

theorem cone_volume_difference (H R : ℝ) : ΔV = (1/12) * Real.pi * R^2 * H := 
sorry

end cone_volume_difference_l71_71191


namespace solve_eq1_solve_eq2_l71_71348

theorem solve_eq1 (x : ℝ) : (x+1)^2 = 4 ↔ x = 1 ∨ x = -3 := 
by sorry

theorem solve_eq2 (x : ℝ) : 3*x^2 - 2*x - 1 = 0 ↔ x = 1 ∨ x = -1/3 := 
by sorry

end solve_eq1_solve_eq2_l71_71348


namespace fermats_little_theorem_l71_71829

theorem fermats_little_theorem 
  (a n : ℕ) 
  (h₁ : 0 < a) 
  (h₂ : 0 < n) 
  (h₃ : Nat.gcd a n = 1) 
  (phi : ℕ := (Nat.totient n)) 
  : n ∣ (a ^ phi - 1) := sorry

end fermats_little_theorem_l71_71829


namespace interest_calculation_years_l71_71319

theorem interest_calculation_years
  (principal : ℤ) (rate : ℝ) (difference : ℤ) (n : ℤ)
  (h_principal : principal = 2400)
  (h_rate : rate = 0.04)
  (h_difference : difference = 1920)
  (h_equation : (principal : ℝ) * rate * n = principal - difference) :
  n = 5 := 
sorry

end interest_calculation_years_l71_71319


namespace probability_X_between_neg1_and_5_probability_X_le_8_probability_X_ge_5_probability_X_between_neg3_and_9_l71_71420

noncomputable def normalCDF (z : ℝ) : ℝ :=
  sorry -- Assuming some CDF function for the sake of the example.

variable (X : ℝ → ℝ)
variable (μ : ℝ := 3)
variable (σ : ℝ := sqrt 4)

-- 1. Proof that P(-1 < X < 5) = 0.8185
theorem probability_X_between_neg1_and_5 : 
  ((-1 < X) ∧ (X < 5) → (normalCDF 1 - normalCDF (-2)) = 0.8185) :=
  sorry

-- 2. Proof that P(X ≤ 8) = 0.9938
theorem probability_X_le_8 : 
  (X ≤ 8 → normalCDF 2.5 = 0.9938) :=
  sorry

-- 3. Proof that P(X ≥ 5) = 0.1587
theorem probability_X_ge_5 : 
  (X ≥ 5 → (1 - normalCDF 1) = 0.1587) :=
  sorry

-- 4. Proof that P(-3 < X < 9) = 0.9972
theorem probability_X_between_neg3_and_9 : 
  ((-3 < X) ∧ (X < 9) → (2 * normalCDF 3 - 1) = 0.9972) :=
  sorry

end probability_X_between_neg1_and_5_probability_X_le_8_probability_X_ge_5_probability_X_between_neg3_and_9_l71_71420


namespace inverse_proportionality_l71_71354

theorem inverse_proportionality:
  (∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → y = k / x) ∧ y = 1 ∧ x = 2 →
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → y = 2 / x :=
by
  sorry

end inverse_proportionality_l71_71354


namespace P_projection_matrix_P_not_invertible_l71_71622

noncomputable def v : ℝ × ℝ := (4, -1)

noncomputable def norm_v : ℝ := Real.sqrt (4^2 + (-1)^2)

noncomputable def u : ℝ × ℝ := (4 / norm_v, -1 / norm_v)

noncomputable def P : ℝ × ℝ × ℝ × ℝ :=
((4 * 4) / norm_v^2, (4 * -1) / norm_v^2, 
 (-1 * 4) / norm_v^2, (-1 * -1) / norm_v^2)

theorem P_projection_matrix :
  P = (16 / 17, -4 / 17, -4 / 17, 1 / 17) := by
  sorry

theorem P_not_invertible :
  ¬(∃ Q : ℝ × ℝ × ℝ × ℝ, P = Q) := by
  sorry

end P_projection_matrix_P_not_invertible_l71_71622


namespace incorrect_step_l71_71481

-- Given conditions
variables {a b : ℝ} (hab : a < b)

-- Proof statement of the incorrect step ③
theorem incorrect_step : ¬ (2 * (a - b) ^ 2 < (a - b) ^ 2) :=
by sorry

end incorrect_step_l71_71481


namespace correct_sunset_time_l71_71821

-- Definitions corresponding to the conditions
def length_of_daylight : ℕ × ℕ := (10, 30) -- (hours, minutes)
def sunrise_time : ℕ × ℕ := (6, 50) -- (hours, minutes)

-- The reaching goal is to prove the sunset time
def sunset_time (sunrise : ℕ × ℕ) (daylight : ℕ × ℕ) : ℕ × ℕ :=
  let (sh, sm) := sunrise
  let (dh, dm) := daylight
  let total_minutes := sm + dm
  let extra_hour := total_minutes / 60
  let final_minutes := total_minutes % 60
  (sh + dh + extra_hour, final_minutes)

-- The theorem to prove
theorem correct_sunset_time :
  sunset_time sunrise_time length_of_daylight = (17, 20) := sorry

end correct_sunset_time_l71_71821


namespace intersection_complement_l71_71616

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- The complement of B in U
def complement_U (U B : Set ℕ) : Set ℕ := U \ B

-- Statement to prove
theorem intersection_complement : A ∩ (complement_U U B) = {1} := 
by 
  sorry

end intersection_complement_l71_71616


namespace cos_100_eq_neg_sqrt_l71_71120

theorem cos_100_eq_neg_sqrt (a : ℝ) (h : Real.sin (80 * Real.pi / 180) = a) : 
  Real.cos (100 * Real.pi / 180) = -Real.sqrt (1 - a^2) := 
sorry

end cos_100_eq_neg_sqrt_l71_71120


namespace jerry_feathers_left_l71_71499

def hawk_feathers : ℕ := 37
def eagle_feathers : ℝ := 17.5 * hawk_feathers
def total_feathers : ℝ := hawk_feathers + eagle_feathers
def feathers_to_sister : ℝ := 0.45 * total_feathers
def remaining_feathers_after_sister : ℝ := total_feathers - feathers_to_sister
def feathers_sold : ℝ := 0.85 * remaining_feathers_after_sister
def final_remaining_feathers : ℝ := remaining_feathers_after_sister - feathers_sold

theorem jerry_feathers_left : ⌊final_remaining_feathers⌋₊ = 56 := by
  sorry

end jerry_feathers_left_l71_71499


namespace train_length_calculation_l71_71076

theorem train_length_calculation (speed_kmph : ℝ) (time_seconds : ℝ) (platform_length_m : ℝ) (train_length_m: ℝ) : speed_kmph = 45 → time_seconds = 51.99999999999999 → platform_length_m = 290 → train_length_m = 360 :=
by
  sorry

end train_length_calculation_l71_71076


namespace one_odd_one_even_l71_71477

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem one_odd_one_even (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prime : is_prime a) (h_eq : a^2 + b^2 = c^2) : 
(is_odd b ∧ is_even c) ∨ (is_even b ∧ is_odd c) :=
sorry

end one_odd_one_even_l71_71477


namespace odd_factors_count_l71_71352

-- Definition of the number to factorize
def n : ℕ := 252

-- The prime factorization of 252 (ignoring the even factor 2)
def p1 : ℕ := 3
def p2 : ℕ := 7
def e1 : ℕ := 2  -- exponent of 3 in the factorization
def e2 : ℕ := 1  -- exponent of 7 in the factorization

-- The statement to prove
theorem odd_factors_count : 
  let odd_factor_count := (e1 + 1) * (e2 + 1)
  odd_factor_count = 6 :=
by
  sorry

end odd_factors_count_l71_71352


namespace weekly_goal_cans_l71_71661

theorem weekly_goal_cans : (20 +  (20 * 1.5) + (20 * 2) + (20 * 2.5) + (20 * 3)) = 200 := by
  sorry

end weekly_goal_cans_l71_71661


namespace quadratic_transformation_l71_71908

theorem quadratic_transformation
    (a b c : ℝ)
    (h : ℝ)
    (cond : ∀ x, a * x^2 + b * x + c = 4 * (x - 5)^2 + 16) :
    (∀ x, 5 * a * x^2 + 5 * b * x + 5 * c = 20 * (x - h)^2 + 80) → h = 5 :=
by
  sorry

end quadratic_transformation_l71_71908


namespace simplify_expr_l71_71737

theorem simplify_expr (a : ℝ) : 2 * a * (3 * a ^ 2 - 4 * a + 3) - 3 * a ^ 2 * (2 * a - 4) = 4 * a ^ 2 + 6 * a :=
by
  sorry

end simplify_expr_l71_71737


namespace exists_indices_divisible_2019_l71_71435

theorem exists_indices_divisible_2019 (x : Fin 2020 → ℤ) : 
  ∃ (i j : Fin 2020), i ≠ j ∧ (x j - x i) % 2019 = 0 := 
  sorry

end exists_indices_divisible_2019_l71_71435


namespace minimum_value_l71_71751

noncomputable def min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : ℝ :=
  (x + y + z) * (1 / (x + y) + 1 / (x + z) + 1 / (y + z))

theorem minimum_value : ∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
  (x + y + z) * (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ≥ 9 / 2 :=
by
  intro x y z hx hy hz
  sorry

end minimum_value_l71_71751


namespace vertex_in_fourth_quadrant_l71_71890

theorem vertex_in_fourth_quadrant (a : ℝ) (ha : a < 0) :  
  let x_vertex := -a / 4
  let y_vertex := (-40 - a^2) / 8
  x_vertex > 0 ∧ y_vertex < 0 := by
  let x_vertex := -a / 4
  let y_vertex := (-40 - a^2) / 8
  have hx : x_vertex > 0 := by sorry
  have hy : y_vertex < 0 := by sorry
  exact And.intro hx hy

end vertex_in_fourth_quadrant_l71_71890


namespace area_of_triangle_MEF_correct_l71_71087

noncomputable def area_of_triangle_MEF : ℝ :=
  let r := 10
  let chord_length := 12
  let parallel_segment_length := 15
  let angle_MOA := 30.0
  (1 / 2) * chord_length * (2 * Real.sqrt 21)

theorem area_of_triangle_MEF_correct :
  area_of_triangle_MEF = 12 * Real.sqrt 21 :=
by
  -- proof will go here
  sorry

end area_of_triangle_MEF_correct_l71_71087


namespace second_hand_distance_l71_71711

theorem second_hand_distance (r : ℝ) (t : ℝ) (π : ℝ) (hand_length_6cm : r = 6) (time_15_min : t = 15) : 
  ∃ d : ℝ, d = 180 * π :=
by
  sorry

end second_hand_distance_l71_71711


namespace john_paid_8000_l71_71722

-- Define the variables according to the conditions
def upfront_fee : ℕ := 1000
def hourly_rate : ℕ := 100
def court_hours : ℕ := 50
def prep_hours : ℕ := 2 * court_hours
def total_hours : ℕ := court_hours + prep_hours
def total_fee : ℕ := upfront_fee + total_hours * hourly_rate
def john_share : ℕ := total_fee / 2

-- Prove that John's share is $8,000
theorem john_paid_8000 : john_share = 8000 :=
by sorry

end john_paid_8000_l71_71722


namespace marble_distribution_correct_l71_71136

def num_ways_to_distribute_marbles : ℕ :=
  -- Given:
  -- Evan divides 100 marbles among three volunteers with each getting at least one marble
  -- Lewis selects a positive integer n > 1 and for each volunteer, steals exactly 1/n of marbles if possible.
  -- Prove that the number of ways to distribute the marbles such that Lewis cannot steal from all volunteers
  3540

theorem marble_distribution_correct :
  num_ways_to_distribute_marbles = 3540 :=
sorry

end marble_distribution_correct_l71_71136


namespace angle_AXC_angle_ACB_l71_71369

-- Definitions of the problem conditions
variables (A B C D X : Type) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty X]
variables (AD DC: Type) [Nonempty AD] [Nonempty DC]
variables (angleB angleXDC angleAXC angleACB : ℝ)
variables (AB BX: ℝ)

-- Given conditions
axiom equal_sides: AD = DC
axiom pointX: BX = AB
axiom given_angleB: angleB = 34
axiom given_angleXDC: angleXDC = 52

-- Proof goals (no proof included, only the statements)
theorem angle_AXC: angleAXC = 107 :=
sorry

theorem angle_ACB: angleACB = 47 :=
sorry

end angle_AXC_angle_ACB_l71_71369


namespace percentage_problem_l71_71057

-- Define the main proposition
theorem percentage_problem (n : ℕ) (a : ℕ) (b : ℕ) (P : ℕ) :
  n = 6000 →
  a = (50 * n) / 100 →
  b = (30 * a) / 100 →
  (P * b) / 100 = 90 →
  P = 10 :=
by
  intros h_n h_a h_b h_Pb
  sorry

end percentage_problem_l71_71057


namespace postman_speeds_l71_71236

-- Define constants for the problem
def d1 : ℝ := 2 -- distance uphill in km
def d2 : ℝ := 4 -- distance on flat ground in km
def d3 : ℝ := 3 -- distance downhill in km
def time1 : ℝ := 2.267 -- time from A to B in hours
def time2 : ℝ := 2.4 -- time from B to A in hours
def half_time_round_trip : ℝ := 2.317 -- round trip to halfway point in hours

-- Define the speeds
noncomputable def V1 : ℝ := 3 -- speed uphill in km/h
noncomputable def V2 : ℝ := 4 -- speed on flat ground in km/h
noncomputable def V3 : ℝ := 5 -- speed downhill in km/h

-- The mathematically equivalent proof statement
theorem postman_speeds :
  (d1 / V1 + d2 / V2 + d3 / V3 = time1) ∧
  (d3 / V1 + d2 / V2 + d1 / V3 = time2) ∧
  (1 / V1 + 2 / V2 + 1.5 / V3 = half_time_round_trip / 2) :=
by 
  -- Equivalence holds because the speeds satisfy the given conditions
  sorry

end postman_speeds_l71_71236


namespace train_speed_l71_71925

/-- 
Train A leaves the station traveling at a certain speed v. 
Two hours later, Train B leaves the same station traveling in the same direction at 36 miles per hour. 
Train A was overtaken by Train B 360 miles from the station.
We need to prove that the speed of Train A was 30 miles per hour.
-/
theorem train_speed (v : ℕ) (t : ℕ) (h1 : 36 * (t - 2) = 360) (h2 : v * t = 360) : v = 30 :=
by 
  sorry

end train_speed_l71_71925


namespace inequality_am_gm_l71_71218

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a^3 / (b * c) + b^3 / (c * a) + c^3 / (a * b) ≥ a + b + c :=
by {
    sorry
}

end inequality_am_gm_l71_71218


namespace condition_a_condition_b_condition_c_l71_71456

-- Definitions for conditions
variable {ι : Type*} (f₁ f₂ f₃ f₄ : ι → ℝ) (x : ι)

-- First part: Condition to prove second equation is a consequence of first
theorem condition_a :
  (∀ x, f₁ x * f₄ x = f₂ x * f₃ x) →
  ((f₂ x ≠ 0) ∧ (f₂ x + f₄ x ≠ 0)) →
  (f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) :=
sorry

-- Second part: Condition to prove first equation is a consequence of second
theorem condition_b :
  (∀ x, f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) →
  ((f₄ x ≠ 0) ∧ (f₂ x ≠ 0)) →
  (f₁ x * f₄ x = f₂ x * f₃ x) :=
sorry

-- Third part: Condition for equivalence of the equations
theorem condition_c :
  (∀ x, (f₁ x * f₄ x = f₂ x * f₃ x) ∧ (x ∉ {x | f₂ x + f₄ x = 0})) ↔
  (∀ x, (f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) ∧ (x ∉ {x | f₄ x = 0})) :=
sorry

end condition_a_condition_b_condition_c_l71_71456


namespace sum_of_first_60_digits_l71_71611

noncomputable def decimal_expansion_period : List ℕ := [0, 0, 0, 8, 1, 0, 3, 7, 2, 7, 7, 1, 4, 7, 4, 8, 7, 8, 4, 4, 4, 0, 8, 4, 2, 7, 8, 7, 6, 8]

def sum_of_list (l : List ℕ) : ℕ := l.foldl (· + ·) 0

theorem sum_of_first_60_digits : sum_of_list (decimal_expansion_period ++ decimal_expansion_period) = 282 := 
by
  simp [decimal_expansion_period, sum_of_list]
  sorry

end sum_of_first_60_digits_l71_71611


namespace integral_value_l71_71141

theorem integral_value (a : ℝ) (h : -35 * a^3 = -280) : ∫ x in a..2 * Real.exp 1, 1 / x = 1 := by
  sorry

end integral_value_l71_71141


namespace polynomial_diff_l71_71080

theorem polynomial_diff (m n : ℤ) (h1 : 2 * m + 2 = 0) (h2 : n - 4 = 0) :
  (4 * m^2 * n - 3 * m * n^2) - 2 * (m^2 * n + m * n^2) = -72 := 
by {
  -- This is where the proof would go, so we put sorry for now
  sorry
}

end polynomial_diff_l71_71080


namespace bicycle_cost_price_l71_71728

variable (CP_A SP_B SP_C : ℝ)

theorem bicycle_cost_price 
  (h1 : SP_B = CP_A * 1.20) 
  (h2 : SP_C = SP_B * 1.25) 
  (h3 : SP_C = 225) :
  CP_A = 150 := 
by
  sorry

end bicycle_cost_price_l71_71728


namespace remainder_when_150_divided_by_k_is_2_l71_71856

theorem remainder_when_150_divided_by_k_is_2
  (k : ℕ) (q : ℤ)
  (hk_pos : k > 0)
  (hk_condition : 120 = q * k^2 + 8) :
  150 % k = 2 :=
sorry

end remainder_when_150_divided_by_k_is_2_l71_71856


namespace inequality_proof_l71_71703

theorem inequality_proof (a b c : ℝ) (k : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : k ≥ 1) : 
  (a^(k + 1) / b^k + b^(k + 1) / c^k + c^(k + 1) / a^k) ≥ (a^k / b^(k - 1) + b^k / c^(k - 1) + c^k / a^(k - 1)) :=
by
  sorry

end inequality_proof_l71_71703


namespace gift_cost_l71_71812

theorem gift_cost (half_cost : ℝ) (h : half_cost = 14) : 2 * half_cost = 28 :=
by
  sorry

end gift_cost_l71_71812


namespace radio_price_rank_l71_71382

theorem radio_price_rank (total_items : ℕ) (radio_position_highest : ℕ) (radio_position_lowest : ℕ) 
  (h1 : total_items = 40) (h2 : radio_position_highest = 17) : 
  radio_position_lowest = total_items - radio_position_highest + 1 :=
by
  sorry

end radio_price_rank_l71_71382


namespace mul_point_five_point_three_l71_71366

theorem mul_point_five_point_three : 0.5 * 0.3 = 0.15 := 
by  sorry

end mul_point_five_point_three_l71_71366


namespace largest_square_area_correct_l71_71346

noncomputable def area_of_largest_square (x y z : ℝ) : Prop := 
  ∃ (area : ℝ), (z^2 = area) ∧ 
                 (x^2 + y^2 = z^2) ∧ 
                 (x^2 + y^2 + 2*z^2 = 722) ∧ 
                 (area = 722 / 3)

theorem largest_square_area_correct (x y z : ℝ) :
  area_of_largest_square x y z :=
  sorry

end largest_square_area_correct_l71_71346


namespace fill_sacks_times_l71_71922

-- Define the capacities of the sacks
def father_sack_capacity : ℕ := 20
def senior_ranger_sack_capacity : ℕ := 30
def volunteer_sack_capacity : ℕ := 25
def number_of_volunteers : ℕ := 2

-- Total wood gathered
def total_wood_gathered : ℕ := 200

-- Statement of the proof problem
theorem fill_sacks_times : (total_wood_gathered / (father_sack_capacity + senior_ranger_sack_capacity + (number_of_volunteers * volunteer_sack_capacity))) = 2 := by
  sorry

end fill_sacks_times_l71_71922


namespace lcm_of_36_and_105_l71_71541

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_of_36_and_105_l71_71541


namespace find_m_value_l71_71778

theorem find_m_value (m : ℝ) (x : ℝ) (y : ℝ) :
  (∀ (x y : ℝ), x + m * y + 3 - 2 * m = 0) →
  (∃ (y : ℝ), x = 0 ∧ y = -1) →
  m = 1 :=
by
  sorry

end find_m_value_l71_71778


namespace area_ratio_l71_71475

-- Definitions for the conditions in the problem
variables (PQ QR RP : ℝ) (p q r : ℝ)

-- Conditions
def pq_condition := PQ = 18
def qr_condition := QR = 24
def rp_condition := RP = 30
def pqr_sum := p + q + r = 3 / 4
def pqr_squaresum := p^2 + q^2 + r^2 = 1 / 2

-- Goal statement that the area ratio of triangles XYZ to PQR is 23/32
theorem area_ratio (h1 : PQ = 18) (h2 : QR = 24) (h3 : RP = 30) 
  (h4 : p + q + r = 3 / 4) (h5 : p^2 + q^2 + r^2 = 1 / 2) : 
  ∃ (m n : ℕ), (m + n = 55) ∧ (m / n = 23 / 32) := 
sorry

end area_ratio_l71_71475


namespace weekly_car_mileage_l71_71540

-- Definitions of the conditions
def dist_school := 2.5 
def dist_market := 2 
def school_days := 4
def school_trips_per_day := 2
def market_trips_per_week := 1

-- Proof statement
theorem weekly_car_mileage : 
  4 * 2 * (2.5 * 2) + (1 * (2 * 2)) = 44 :=
by
  -- The goal is to prove that 4 days of 2 round trips to school plus 1 round trip to market equals 44 miles
  sorry

end weekly_car_mileage_l71_71540


namespace find_m_l71_71055

theorem find_m (m : ℤ) :
  (2 * m + 7) * (m - 2) = 51 → m = 5 := by
  sorry

end find_m_l71_71055


namespace find_a_l71_71882

theorem find_a (x y a : ℤ) (h₁ : x = 1) (h₂ : y = -1) (h₃ : 2 * x - a * y = 3) : a = 1 :=
sorry

end find_a_l71_71882


namespace price_reduction_for_target_profit_l71_71969
-- Import the necessary libraries

-- Define the conditions
def average_sales_per_day := 70
def initial_profit_per_item := 50
def sales_increase_per_dollar_decrease := 2

-- Define the functions for sales volume increase and profit per item
def sales_volume_increase (x : ℝ) : ℝ := 2 * x
def profit_per_item (x : ℝ) : ℝ := initial_profit_per_item - x

-- Define the function for daily profit
def daily_profit (x : ℝ) : ℝ := (profit_per_item x) * (average_sales_per_day + sales_volume_increase x)

-- State the main theorem
theorem price_reduction_for_target_profit : 
  ∃ x : ℝ, daily_profit x = 3572 ∧ x = 12 :=
sorry

end price_reduction_for_target_profit_l71_71969


namespace fraction_of_married_men_l71_71328

theorem fraction_of_married_men (prob_single_woman : ℚ) (H : prob_single_woman = 3 / 7) :
  ∃ (fraction_married_men : ℚ), fraction_married_men = 4 / 11 :=
by
  -- Further proof steps would go here if required
  sorry

end fraction_of_married_men_l71_71328


namespace surface_area_ratio_l71_71058

-- Definitions for side lengths in terms of common multiplier x
def side_length_a (x : ℝ) := 2 * x
def side_length_b (x : ℝ) := 1 * x
def side_length_c (x : ℝ) := 3 * x
def side_length_d (x : ℝ) := 4 * x
def side_length_e (x : ℝ) := 6 * x

-- Definitions for surface areas using the given formula
def surface_area (side_length : ℝ) := 6 * side_length^2

def surface_area_a (x : ℝ) := surface_area (side_length_a x)
def surface_area_b (x : ℝ) := surface_area (side_length_b x)
def surface_area_c (x : ℝ) := surface_area (side_length_c x)
def surface_area_d (x : ℝ) := surface_area (side_length_d x)
def surface_area_e (x : ℝ) := surface_area (side_length_e x)

-- Proof statement for the ratio of total surface areas
theorem surface_area_ratio (x : ℝ) (hx : x ≠ 0) :
  (surface_area_a x) / (surface_area_b x) = 4 ∧
  (surface_area_c x) / (surface_area_b x) = 9 ∧
  (surface_area_d x) / (surface_area_b x) = 16 ∧
  (surface_area_e x) / (surface_area_b x) = 36 :=
by {
  sorry
}

end surface_area_ratio_l71_71058


namespace quadratic_condition_l71_71903

theorem quadratic_condition (a b c : ℝ) : (a ≠ 0) ↔ ∃ (x : ℝ), ax^2 + bx + c = 0 :=
by sorry

end quadratic_condition_l71_71903


namespace not_sufficient_nor_necessary_l71_71360

theorem not_sufficient_nor_necessary (a b : ℝ) : ¬((a > b) → (a^2 > b^2)) ∧ ¬((a^2 > b^2) → (a > b)) := 
by 
  sorry

end not_sufficient_nor_necessary_l71_71360


namespace at_least_one_not_greater_than_minus_four_l71_71460

theorem at_least_one_not_greater_than_minus_four {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4 :=
sorry

end at_least_one_not_greater_than_minus_four_l71_71460


namespace medium_as_decoy_and_rational_choice_l71_71904

/-- 
  Define the prices and sizes of the popcorn containers:
  Small: 50g for 200 rubles.
  Medium: 70g for 400 rubles.
  Large: 130g for 500 rubles.
-/
structure PopcornContainer where
  size : ℕ -- in grams
  price : ℕ -- in rubles

def small := PopcornContainer.mk 50 200
def medium := PopcornContainer.mk 70 400
def large := PopcornContainer.mk 130 500

/-- 
  The medium-sized popcorn container can be considered a decoy
  in the context of asymmetric dominance.
  Additionally, under certain budget constraints and preferences, 
  rational economic agents may find the medium-sized container optimal.
-/
theorem medium_as_decoy_and_rational_choice :
  (medium.price = 400 ∧ medium.size = 70) ∧ 
  (∃ (budget : ℕ) (pref : ℕ → ℕ → Prop), (budget ≥ medium.price ∧ 
    pref medium.size (budget - medium.price))) :=
by
  sorry

end medium_as_decoy_and_rational_choice_l71_71904


namespace remaining_slices_correct_l71_71365

-- Define initial slices of pie and cake
def initial_pie_slices : Nat := 2 * 8
def initial_cake_slices : Nat := 12

-- Define slices eaten on Friday
def friday_pie_slices_eaten : Nat := 2
def friday_cake_slices_eaten : Nat := 2

-- Define slices eaten on Saturday
def saturday_pie_slices_eaten (remaining: Nat) : Nat := remaining / 2 -- 50%
def saturday_cake_slices_eaten (remaining: Nat) : Nat := remaining / 4 -- 25%

-- Define slices eaten on Sunday morning
def sunday_morning_pie_slices_eaten : Nat := 2
def sunday_morning_cake_slices_eaten : Nat := 3

-- Define slices eaten on Sunday evening
def sunday_evening_pie_slices_eaten : Nat := 4
def sunday_evening_cake_slices_eaten : Nat := 1

-- Function to calculate remaining slices
def remaining_slices : Nat × Nat :=
  let after_friday_pies := initial_pie_slices - friday_pie_slices_eaten
  let after_friday_cake := initial_cake_slices - friday_cake_slices_eaten
  let after_saturday_pies := after_friday_pies - saturday_pie_slices_eaten after_friday_pies
  let after_saturday_cake := after_friday_cake - saturday_cake_slices_eaten after_friday_cake
  let after_sunday_morning_pies := after_saturday_pies - sunday_morning_pie_slices_eaten
  let after_sunday_morning_cake := after_saturday_cake - sunday_morning_cake_slices_eaten
  let final_pies := after_sunday_morning_pies - sunday_evening_pie_slices_eaten
  let final_cake := after_sunday_morning_cake - sunday_evening_cake_slices_eaten
  (final_pies, final_cake)

theorem remaining_slices_correct :
  remaining_slices = (1, 4) :=
  by {
    sorry -- Proof is omitted
  }

end remaining_slices_correct_l71_71365


namespace distance_to_larger_cross_section_l71_71606

theorem distance_to_larger_cross_section
    (A B : ℝ)
    (a b : ℝ)
    (d : ℝ)
    (h : ℝ)
    (h_eq : h = 30):
  A = 300 * Real.sqrt 2 → 
  B = 675 * Real.sqrt 2 → 
  a = Real.sqrt (A / B) → 
  b = d / (1 - a) → 
  d = 10 → 
  b = h :=
by
  sorry

end distance_to_larger_cross_section_l71_71606


namespace batsman_average_increase_l71_71726

-- Definitions to capture the initial conditions
def runs_scored_in_17th_inning : ℕ := 74
def average_after_17_innings : ℕ := 26

-- Statement to prove the increment in average is 3 runs per inning
theorem batsman_average_increase (A : ℕ) (initial_avg : ℕ)
  (h_initial_runs : 16 * initial_avg + 74 = 17 * 26) :
  26 - initial_avg = 3 :=
by
  sorry

end batsman_average_increase_l71_71726


namespace cubic_binomial_expansion_l71_71046

theorem cubic_binomial_expansion :
  49^3 + 3 * 49^2 + 3 * 49 + 1 = 125000 :=
by
  sorry

end cubic_binomial_expansion_l71_71046


namespace sum_and_num_of_factors_eq_1767_l71_71158

theorem sum_and_num_of_factors_eq_1767 (n : ℕ) (σ d : ℕ → ℕ) :
  (σ n + d n = 1767) → 
  ∃ m : ℕ, σ m + d m = 1767 :=
by 
  sorry

end sum_and_num_of_factors_eq_1767_l71_71158


namespace point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence_l71_71106

def is_on_line (n : ℕ) (a_n : ℕ) : Prop := a_n = 2 * n + 1

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n m, a n - a m = d * (n - m)

theorem point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence (a : ℕ → ℕ) :
  (∀ n, is_on_line n (a n)) → is_arithmetic_sequence a ∧ 
  ¬ (is_arithmetic_sequence a → ∀ n, is_on_line n (a n)) :=
sorry

end point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence_l71_71106


namespace sugar_spilled_l71_71523

-- Define the initial amount of sugar and the amount left
def initial_sugar : ℝ := 9.8
def remaining_sugar : ℝ := 4.6

-- State the problem as a theorem
theorem sugar_spilled :
  initial_sugar - remaining_sugar = 5.2 := 
sorry

end sugar_spilled_l71_71523


namespace snail_total_distance_l71_71335

-- Conditions
def initial_pos : ℤ := 0
def pos1 : ℤ := 4
def pos2 : ℤ := -3
def pos3 : ℤ := 6

-- Total distance traveled by the snail
def distance_traveled : ℤ :=
  abs (pos1 - initial_pos) +
  abs (pos2 - pos1) +
  abs (pos3 - pos2)

-- Theorem statement
theorem snail_total_distance : distance_traveled = 20 :=
by
  -- Proof is omitted, as per request
  sorry

end snail_total_distance_l71_71335


namespace Leah_lost_11_dollars_l71_71468

-- Define the conditions
def LeahEarned : ℕ := 28
def MilkshakeCost : ℕ := LeahEarned / 7
def RemainingAfterMilkshake : ℕ := LeahEarned - MilkshakeCost
def Savings : ℕ := RemainingAfterMilkshake / 2
def WalletAfterSavings : ℕ := RemainingAfterMilkshake - Savings
def WalletAfterDog : ℕ := 1

-- Define the theorem to prove Leah's loss
theorem Leah_lost_11_dollars : WalletAfterSavings - WalletAfterDog = 11 := 
by 
  sorry

end Leah_lost_11_dollars_l71_71468


namespace spending_together_l71_71719

def sandwich_cost := 2
def hamburger_cost := 2
def hotdog_cost := 1
def juice_cost := 2
def selene_sandwiches := 3
def selene_juices := 1
def tanya_hamburgers := 2
def tanya_juices := 2

def selene_spending : ℕ := (selene_sandwiches * sandwich_cost) + (selene_juices * juice_cost)
def tanya_spending : ℕ := (tanya_hamburgers * hamburger_cost) + (tanya_juices * juice_cost)
def total_spending : ℕ := selene_spending + tanya_spending

theorem spending_together : total_spending = 16 :=
by
  sorry

end spending_together_l71_71719


namespace island_knights_liars_two_people_l71_71983

def islanders_knights_and_liars (n : ℕ) : Prop :=
  ∃ (knight liar : ℕ),
    knight + liar = n ∧
    (∀ i : ℕ, 1 ≤ i → i ≤ n → 
      ((i % i = 0 → liar > 0 ∧ knight > 0) ∧ (i % i ≠ 0 → liar > 0)))

theorem island_knights_liars_two_people :
  islanders_knights_and_liars 2 :=
sorry

end island_knights_liars_two_people_l71_71983


namespace lcm_pair_eq_sum_l71_71072

theorem lcm_pair_eq_sum (x y : ℕ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : Nat.lcm x y = 1 + 2 * x + 3 * y) :
  (x = 4 ∧ y = 9) ∨ (x = 9 ∧ y = 4) :=
by {
  sorry
}

end lcm_pair_eq_sum_l71_71072


namespace tan_of_angle_l71_71791

theorem tan_of_angle (α : ℝ) (h₁ : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (h₂ : Real.sin α = 3 / 5) : 
  Real.tan α = -3 / 4 := 
sorry

end tan_of_angle_l71_71791


namespace bacteria_growth_rate_l71_71310

theorem bacteria_growth_rate (r : ℝ) :
  (1 + r)^6 = 64 → r = 1 :=
by
  intro h
  sorry

end bacteria_growth_rate_l71_71310


namespace sandwich_cost_is_five_l71_71651

-- Define the cost of each sandwich
variables (x : ℝ)

-- Conditions
def jack_orders_sandwiches (cost_per_sandwich : ℝ) : Prop :=
  3 * cost_per_sandwich = 15

-- Proof problem statement (no proof provided)
theorem sandwich_cost_is_five (h : jack_orders_sandwiches x) : x = 5 :=
sorry

end sandwich_cost_is_five_l71_71651


namespace determine_h_l71_71780

noncomputable def h (x : ℝ) : ℝ :=
  -12*x^4 + 2*x^3 + 8*x^2 - 8*x + 3

theorem determine_h :
  (12*x^4 + 4*x^3 - 2*x + 3 + h x = 6*x^3 + 8*x^2 - 10*x + 6) ↔
  (h x = -12*x^4 + 2*x^3 + 8*x^2 - 8*x + 3) :=
by 
  sorry

end determine_h_l71_71780


namespace job_completion_time_l71_71134

theorem job_completion_time (A_rate D_rate Combined_rate : ℝ) (hA : A_rate = 1 / 3) (hD : D_rate = 1 / 6) (hCombined : Combined_rate = A_rate + D_rate) :
  (1 / Combined_rate) = 2 :=
by sorry

end job_completion_time_l71_71134


namespace midpoint_of_segment_l71_71238

theorem midpoint_of_segment :
  let p1 := (12, -8)
  let p2 := (-4, 10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint = (4, 1) :=
by
  let p1 := (12, -8)
  let p2 := (-4, 10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  show midpoint = (4, 1)
  sorry

end midpoint_of_segment_l71_71238


namespace value_of_a_plus_d_l71_71720

theorem value_of_a_plus_d 
  (a b c d : ℤ)
  (h1 : a + b = 12) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) 
  : a + d = 9 := 
  sorry

end value_of_a_plus_d_l71_71720


namespace tangent_line_at_point_l71_71467

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = (2 * x - 1)^3) (h_point : (x, y) = (1, 1)) :
  ∃ m b : ℝ, y = m * x + b ∧ m = 6 ∧ b = -5 :=
by
  sorry

end tangent_line_at_point_l71_71467


namespace transform_polynomial_l71_71510

theorem transform_polynomial (x y : ℝ) (h1 : y = x + 1 / x) (h2 : x^4 - x^3 - 6 * x^2 - x + 1 = 0) :
  x^2 * (y^2 - y - 6) = 0 := 
  sorry

end transform_polynomial_l71_71510


namespace range_of_a_l71_71769

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2 * a + 1)^x > (2 * a + 1)^y) → (-1/2 < a ∧ a < 0) :=
by
  sorry

end range_of_a_l71_71769


namespace problem_l71_71340

theorem problem (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h1 : p + 5 < q)
  (h2 : (p + (p + 2) + (p + 5) + q + (q + 1) + (2 * q - 1)) / 6 = q)
  (h3 : (p + 5 + q) / 2 = q) : p + q = 11 :=
by sorry

end problem_l71_71340


namespace solve_system_of_inequalities_l71_71213

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x + 3 ≤ x + 2) ∧ ((x + 1) / 3 > x - 1) → x ≤ -1 := by
  sorry

end solve_system_of_inequalities_l71_71213


namespace simplify_expression_l71_71163

theorem simplify_expression :
  ((1 + 2 + 3 + 6) / 3) + ((3 * 6 + 9) / 4) = 43 / 4 := 
sorry

end simplify_expression_l71_71163


namespace direct_proportion_function_l71_71748

theorem direct_proportion_function (m : ℝ) 
  (h1 : m + 1 ≠ 0) 
  (h2 : m^2 - 1 = 0) : 
  m = 1 :=
sorry

end direct_proportion_function_l71_71748


namespace time_for_A_to_finish_race_l71_71909

-- Definitions based on the conditions
def race_distance : ℝ := 120
def B_time : ℝ := 45
def B_beaten_distance : ℝ := 24

-- Proof statement: We need to show that A's time is 56.25 seconds
theorem time_for_A_to_finish_race : ∃ (t : ℝ), t = 56.25 ∧ (120 / t = 96 / 45)
  := sorry

end time_for_A_to_finish_race_l71_71909


namespace solution_set_inequality_l71_71450

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ ⦃a b⦄, a ∈ s → b ∈ s → a ≤ b → f a ≤ f b

def f_increasing_on_pos : Prop := is_increasing_on f (Set.Ioi 0)

def f_at_one_zero : Prop := f 1 = 0

theorem solution_set_inequality : 
    is_odd f →
    f_increasing_on_pos →
    f_at_one_zero →
    {x : ℝ | x * (f x - f (-x)) < 0} = {x : ℝ | -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1} :=
sorry

end solution_set_inequality_l71_71450


namespace pages_revised_only_once_l71_71230

theorem pages_revised_only_once 
  (total_pages : ℕ)
  (cost_per_page_first_time : ℝ)
  (cost_per_page_revised : ℝ)
  (revised_twice_pages : ℕ)
  (total_cost : ℝ)
  (pages_revised_only_once : ℕ) :
  total_pages = 100 →
  cost_per_page_first_time = 10 →
  cost_per_page_revised = 5 →
  revised_twice_pages = 30 →
  total_cost = 1400 →
  10 * (total_pages - pages_revised_only_once - revised_twice_pages) + 
  15 * pages_revised_only_once + 
  20 * revised_twice_pages = total_cost →
  pages_revised_only_once = 20 :=
by
  intros 
  sorry

end pages_revised_only_once_l71_71230


namespace divide_gray_area_l71_71370

-- The conditions
variables {A_rectangle A_square : ℝ} (h : 0 ≤ A_square ∧ A_square ≤ A_rectangle)

-- The main statement
theorem divide_gray_area : ∃ l : ℝ → ℝ → Prop, (∀ (x : ℝ), l x (A_rectangle / 2)) ∧ (∀ (y : ℝ), l (A_square / 2) y) ∧ (A_rectangle - A_square) / 2 = (A_rectangle - A_square) / 2 := by sorry

end divide_gray_area_l71_71370


namespace cos_4theta_l71_71929

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (4 * θ) = 17 / 81 := 
by 
  sorry

end cos_4theta_l71_71929


namespace geo_sequence_sum_l71_71325

theorem geo_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = 2)
  (h2 : a 4 + a 5 = 4)
  (h_geo : ∀ n, a (n + 1) = q * a n) :
  a 10 + a 11 = 16 := by
  -- Insert proof here
  sorry  -- skipping the proof

end geo_sequence_sum_l71_71325


namespace vasya_can_guess_number_in_10_questions_l71_71318

noncomputable def log2 (n : ℕ) : ℝ := 
  Real.log n / Real.log 2

theorem vasya_can_guess_number_in_10_questions (n q : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 1000) (h3 : q = 10) :
  q ≥ log2 n := 
by
  sorry

end vasya_can_guess_number_in_10_questions_l71_71318


namespace fourth_friend_payment_l71_71451

theorem fourth_friend_payment (a b c d : ℕ) 
  (h1 : a = (1 / 3) * (b + c + d)) 
  (h2 : b = (1 / 4) * (a + c + d)) 
  (h3 : c = (1 / 5) * (a + b + d))
  (h4 : a + b + c + d = 84) : 
  d = 40 := by
sorry

end fourth_friend_payment_l71_71451


namespace length_major_axis_eq_six_l71_71327

-- Define the given equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 9) = 1

-- The theorem stating the length of the major axis
theorem length_major_axis_eq_six (x y : ℝ) (h : ellipse_equation x y) : 
  2 * (Real.sqrt 9) = 6 :=
by
  sorry

end length_major_axis_eq_six_l71_71327


namespace x_k_expr_a_x_k_expr_b_x_k_expr_c_y_k_expr_a_y_k_expr_b_y_k_expr_c_l71_71408

theorem x_k_expr_a (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 6 * x (k - 1) - x (k - 2) := 
by sorry

theorem x_k_expr_b (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 34 * x (k - 2) - x (k - 4) := 
by sorry

theorem x_k_expr_c (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 198 * x (k - 3) - x (k - 6) := 
by sorry

theorem y_k_expr_a (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 6 * y (k - 1) - y (k - 2) := 
by sorry

theorem y_k_expr_b (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 34 * y (k - 2) - y (k - 4) := 
by sorry

theorem y_k_expr_c (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 198 * y (k - 3) - y (k - 6) := 
by sorry

end x_k_expr_a_x_k_expr_b_x_k_expr_c_y_k_expr_a_y_k_expr_b_y_k_expr_c_l71_71408


namespace function_property_l71_71961

def y (x : ℝ) : ℝ := x - 2

theorem function_property : y 1 = -1 :=
by
  -- place for proof
  sorry

end function_property_l71_71961


namespace find_xyz_l71_71316

theorem find_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 14 / 3 := 
sorry

end find_xyz_l71_71316


namespace cost_plane_l71_71839

def cost_boat : ℝ := 254.00
def savings_boat : ℝ := 346.00

theorem cost_plane : cost_boat + savings_boat = 600 := 
by 
  sorry

end cost_plane_l71_71839


namespace no_non_negative_solutions_l71_71427

theorem no_non_negative_solutions (a b : ℕ) (h_diff : a ≠ b) (d := Nat.gcd a b) 
                                 (a' := a / d) (b' := b / d) (n := d * (a' * b' - a' - b')) :
  ¬ ∃ x y : ℕ, a * x + b * y = n := 
by
  sorry

end no_non_negative_solutions_l71_71427


namespace original_triangle_area_l71_71863

-- Define the variables
variable (A_new : ℝ) (r : ℝ)

-- The conditions from the problem
def conditions := r = 5 ∧ A_new = 100

-- Goal: Prove that the original area is 4
theorem original_triangle_area (A_orig : ℝ) (h : conditions r A_new) : A_orig = 4 := by
  sorry

end original_triangle_area_l71_71863


namespace choose_4_from_15_l71_71576

theorem choose_4_from_15 : (Nat.choose 15 4) = 1365 :=
by
  sorry

end choose_4_from_15_l71_71576


namespace rock_height_at_30_l71_71613

theorem rock_height_at_30 (t : ℝ) (h : ℝ) 
  (h_eq : h = 80 - 9 * t - 5 * t^2) 
  (h_30 : h = 30) : 
  t = 2.3874 :=
by
  -- Proof omitted
  sorry

end rock_height_at_30_l71_71613


namespace purely_imaginary_iff_m_eq_1_l71_71131

theorem purely_imaginary_iff_m_eq_1 (m : ℝ) :
  (m^2 - 1 = 0 ∧ m + 1 ≠ 0) → m = 1 :=
by
  sorry

end purely_imaginary_iff_m_eq_1_l71_71131


namespace domain_h_l71_71372

noncomputable def h (x : ℝ) : ℝ := (x^2 + 5 * x + 6) / (|x - 2| + |x + 2|)

theorem domain_h : ∀ x : ℝ, ∃ y : ℝ, y = h x :=
by
  sorry

end domain_h_l71_71372


namespace roses_picked_second_time_l71_71561

-- Define the initial conditions
def initial_roses : ℝ := 37.0
def first_pick : ℝ := 16.0
def total_roses_after_second_picking : ℝ := 72.0

-- Define the calculation after the first picking
def roses_after_first_picking : ℝ := initial_roses + first_pick

-- The Lean statement to prove the number of roses picked the second time
theorem roses_picked_second_time : total_roses_after_second_picking - roses_after_first_picking = 19.0 := 
by
  -- Use the facts stated in the conditions
  sorry

end roses_picked_second_time_l71_71561


namespace sqrt_D_irrational_l71_71688

open Real

theorem sqrt_D_irrational (a : ℤ) (D : ℝ) (hD : D = a^2 + (a + 2)^2 + (a^2 + (a + 2))^2) : ¬ ∃ m : ℤ, D = m^2 :=
by
  sorry

end sqrt_D_irrational_l71_71688


namespace factorize_expression_l71_71752

theorem factorize_expression (a x : ℝ) : a * x^3 - 16 * a * x = a * x * (x + 4) * (x - 4) := by
  sorry

end factorize_expression_l71_71752


namespace proof_P_and_Q_l71_71866

/-!
Proposition P: The line y=2x is perpendicular to the line x+2y=0.
Proposition Q: The projections of skew lines in the same plane could be parallel lines.
Prove: P ∧ Q is true.
-/

def proposition_P : Prop := 
  let slope1 := 2
  let slope2 := -1 / 2
  slope1 * slope2 = -1

def proposition_Q : Prop :=
  ∃ (a b : ℝ), (∃ (p q r s : ℝ),
    (a * r + b * p = 0) ∧ (a * s + b * q = 0)) ∧
    (a ≠ 0 ∨ b ≠ 0)

theorem proof_P_and_Q : proposition_P ∧ proposition_Q :=
  by
  -- We need to prove the conjunction of both propositions is true.
  sorry

end proof_P_and_Q_l71_71866


namespace ratio_of_larger_to_smaller_l71_71564

theorem ratio_of_larger_to_smaller 
    (x y : ℝ) 
    (hx : x > 0) 
    (hy : y > 0) 
    (h : x + y = 7 * (x - y)) : 
    x / y = 4 / 3 := 
by 
    sorry

end ratio_of_larger_to_smaller_l71_71564


namespace infinitely_many_divisible_by_100_l71_71717

open Nat

theorem infinitely_many_divisible_by_100 : ∀ p : ℕ, ∃ n : ℕ, n = 100 * p + 6 ∧ 100 ∣ (2^n + n^2) := by
  sorry

end infinitely_many_divisible_by_100_l71_71717


namespace Benny_total_hours_l71_71842

def hours_per_day : ℕ := 7
def days_worked : ℕ := 14

theorem Benny_total_hours : hours_per_day * days_worked = 98 := by
  sorry

end Benny_total_hours_l71_71842


namespace max_value_l71_71321

def a_n (n : ℕ) : ℤ := -2 * (n : ℤ)^2 + 29 * (n : ℤ) + 3

theorem max_value : ∃ n : ℕ, a_n n = 108 ∧ ∀ m : ℕ, a_n m ≤ 108 := by
  sorry

end max_value_l71_71321


namespace ratio_3_2_l71_71026

theorem ratio_3_2 (m n : ℕ) (h1 : m + n = 300) (h2 : m > 100) (h3 : n > 100) : m / n = 3 / 2 := by
  sorry

end ratio_3_2_l71_71026


namespace problem_statement_l71_71944

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (-2^x + b) / (2^(x+1) + a)

theorem problem_statement :
  (∀ (x : ℝ), f (x) 2 1 = -f (-x) 2 1) ∧
  (∀ (t : ℝ), f (t^2 - 2*t) 2 1 + f (2*t^2 - k) 2 1 < 0 → k < -1/3) :=
by
  sorry

end problem_statement_l71_71944


namespace part_a_part_b_l71_71137

-- Part A: Proving the specific values of p and q
theorem part_a (p q : ℝ) : 
  (∀ x : ℝ, (x + 3) ^ 2 + (7 * x + p) ^ 2 = (kx + m) ^ 2) ∧
  (∀ x : ℝ, (3 * x + 5) ^ 2 + (p * x + q) ^ 2 = (cx + d) ^ 2) → 
  p = 21 ∧ q = 35 :=
sorry

-- Part B: Proving the new polynomial is a square of a linear polynomial
theorem part_b (a b c A B C : ℝ) (hab : a ≠ 0) (hA : A ≠ 0) (hb : b ≠ 0) (hB : B ≠ 0)
  (habc : (∀ x : ℝ, (a * x + b) ^ 2 + (A * x + B) ^ 2 = (kx + m) ^ 2) ∧
         (∀ x : ℝ, (b * x + c) ^ 2 + (B * x + C) ^ 2 = (cx + d) ^ 2)) :
  ∀ x : ℝ, (c * x + a) ^ 2 + (C * x + A) ^ 2 = (lx + n) ^ 2 :=
sorry

end part_a_part_b_l71_71137


namespace inequality_system_solution_l71_71834

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 1 > 2 * (x + 1) ∧ (x + 2) / 3 > x - 2) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end inequality_system_solution_l71_71834


namespace max_alligators_in_days_l71_71235

noncomputable def days := 616
noncomputable def weeks := 88  -- derived from 616 / 7
noncomputable def alligators_per_week := 1

theorem max_alligators_in_days
  (h1 : weeks = days / 7)
  (h2 : ∀ (w : ℕ), alligators_per_week = 1) :
  weeks * alligators_per_week = 88 := by
  sorry

end max_alligators_in_days_l71_71235


namespace tethered_dog_area_comparison_l71_71496

theorem tethered_dog_area_comparison :
  let fence_radius := 20
  let rope_length := 30
  let arrangement1_area := π * (rope_length ^ 2)
  let tether_distance := 12
  let arrangement2_effective_radius := rope_length - tether_distance
  let arrangement2_full_circle_area := π * (arrangement2_effective_radius ^ 2)
  let arrangement2_additional_area := (1 / 4) * π * (tether_distance ^ 2)
  let arrangement2_total_area := arrangement2_full_circle_area + arrangement2_additional_area
  (arrangement1_area - arrangement2_total_area) = 540 * π := 
by
  sorry

end tethered_dog_area_comparison_l71_71496


namespace savings_equal_after_25_weeks_l71_71434

theorem savings_equal_after_25_weeks (x : ℝ) :
  (160 + 25 * x = 210 + 125) → x = 7 :=
by 
  apply sorry

end savings_equal_after_25_weeks_l71_71434


namespace domain_of_tan_sub_pi_over_4_l71_71262

theorem domain_of_tan_sub_pi_over_4 :
  ∀ x : ℝ, (∃ k : ℤ, x = k * π + 3 * π / 4) ↔ ∃ y : ℝ, y = (x - π / 4) ∧ (∃ k : ℤ, y = (2 * k + 1) * π / 2) := 
sorry

end domain_of_tan_sub_pi_over_4_l71_71262


namespace symmetric_point_l71_71178

theorem symmetric_point (x0 y0 : ℝ) (P : ℝ × ℝ) (line : ℝ → ℝ) 
  (hP : P = (-1, 3)) (hline : ∀ x, line x = x) :
  ((x0, y0) = (3, -1)) ↔
    ( ∃ M : ℝ × ℝ, M = ((x0 - -1) / 2, (y0 + 3) / 2) ∧ M.1 = M.2 ) ∧ 
    ( ∃ l : ℝ, l = (y0 - 3) / (x0 + 1) ∧ l = -1 ) :=
by
  sorry

end symmetric_point_l71_71178


namespace tan_half_sum_eq_third_l71_71252

theorem tan_half_sum_eq_third
  (x y : ℝ)
  (h1 : Real.cos x + Real.cos y = 3/5)
  (h2 : Real.sin x + Real.sin y = 1/5) :
  Real.tan ((x + y) / 2) = 1/3 :=
by sorry

end tan_half_sum_eq_third_l71_71252


namespace f_neg2_eq_neg4_l71_71135

noncomputable def f (x : ℝ) : ℝ :=
  if hx : x >= 0 then 3^x - 2*x - 1
  else - (3^(-x) - 2*(-x) - 1)

theorem f_neg2_eq_neg4
: f (-2) = -4 :=
by
  sorry

end f_neg2_eq_neg4_l71_71135


namespace employee_pays_correct_amount_l71_71655

def wholesale_cost : ℝ := 200
def markup_percentage : ℝ := 0.20
def discount_percentage : ℝ := 0.10

def retail_price (wholesale: ℝ) (markup_percentage: ℝ) : ℝ :=
  wholesale * (1 + markup_percentage)

def discount_amount (price: ℝ) (discount_percentage: ℝ) : ℝ :=
  price * discount_percentage

def final_price (retail: ℝ) (discount: ℝ) : ℝ :=
  retail - discount

theorem employee_pays_correct_amount : final_price (retail_price wholesale_cost markup_percentage) 
                                                     (discount_amount (retail_price wholesale_cost markup_percentage) discount_percentage) = 216 := 
by
  sorry

end employee_pays_correct_amount_l71_71655


namespace infinite_six_consecutive_epsilon_squarish_l71_71520

def is_epsilon_squarish (ε : ℝ) (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 < a ∧ a < b ∧ b < (1 + ε) * a ∧ n = a * b

theorem infinite_six_consecutive_epsilon_squarish (ε : ℝ) (hε : 0 < ε) : 
  ∃ (N : ℕ), ∃ (n : ℕ), N ≤ n ∧
  (is_epsilon_squarish ε n) ∧ 
  (is_epsilon_squarish ε (n + 1)) ∧ 
  (is_epsilon_squarish ε (n + 2)) ∧ 
  (is_epsilon_squarish ε (n + 3)) ∧ 
  (is_epsilon_squarish ε (n + 4)) ∧ 
  (is_epsilon_squarish ε (n + 5)) :=
  sorry

end infinite_six_consecutive_epsilon_squarish_l71_71520


namespace total_students_in_class_l71_71270

def number_of_girls := 9
def number_of_boys := 16
def total_students := number_of_girls + number_of_boys

theorem total_students_in_class : total_students = 25 :=
by
  -- The proof will go here
  sorry

end total_students_in_class_l71_71270


namespace rectangle_area_90_l71_71454

theorem rectangle_area_90 {x y : ℝ} (h1 : (x + 3) * (y - 1) = x * y) (h2 : (x - 3) * (y + 1.5) = x * y) : x * y = 90 := 
  sorry

end rectangle_area_90_l71_71454


namespace radius_of_triangle_DEF_l71_71822

noncomputable def radius_of_inscribed_circle (DE DF EF : ℝ) : ℝ :=
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem radius_of_triangle_DEF :
  radius_of_inscribed_circle 26 15 17 = 121 / 29 := by
sorry

end radius_of_triangle_DEF_l71_71822


namespace simplify_expression_l71_71412

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1/3 : ℝ)

theorem simplify_expression :
  (cube_root 512) * (cube_root 343) = 56 := by
  -- conditions
  let h1 : 512 = 2^9 := by rfl
  let h2 : 343 = 7^3 := by rfl
  -- goal
  sorry

end simplify_expression_l71_71412


namespace total_cost_of_tires_and_battery_l71_71653

theorem total_cost_of_tires_and_battery :
  (4 * 42 + 56 = 224) := 
  by
    sorry

end total_cost_of_tires_and_battery_l71_71653


namespace angles_satisfy_system_l71_71516

theorem angles_satisfy_system (k : ℤ) : 
  let x := Real.pi / 3 + k * Real.pi
  let y := k * Real.pi
  x - y = Real.pi / 3 ∧ Real.tan x - Real.tan y = Real.sqrt 3 := 
by 
  sorry

end angles_satisfy_system_l71_71516


namespace find_m_l71_71833

theorem find_m (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) ∧ (m - 1 ≠ 0) → m = -1 :=
by
  sorry

end find_m_l71_71833


namespace range_of_a_l71_71008

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x - 1| + |x - 2| ≤ a^2 + a + 1)) → -1 < a ∧ a < 0 :=
by
  sorry

end range_of_a_l71_71008


namespace tan_subtraction_inequality_l71_71490

theorem tan_subtraction_inequality (x y : ℝ) 
  (hx : 0 < x ∧ x < (π / 2)) 
  (hy : 0 < y ∧ y < (π / 2)) 
  (h : Real.tan x = 3 * Real.tan y) : 
  x - y ≤ π / 6 ∧ (x - y = π / 6 ↔ (x = π / 3 ∧ y = π / 6)) := 
sorry

end tan_subtraction_inequality_l71_71490


namespace cos_alpha_values_l71_71860

theorem cos_alpha_values (α : ℝ) (h : Real.sin (π + α) = -3 / 5) :
  Real.cos α = 4 / 5 ∨ Real.cos α = -4 / 5 := 
sorry

end cos_alpha_values_l71_71860


namespace sin_inequality_of_triangle_l71_71869

theorem sin_inequality_of_triangle (B C : ℝ) (hB : 0 < B) (hB_lt_pi : B < π) 
(hC : 0 < C) (hC_lt_pi : C < π) :
  (B > C) ↔ (Real.sin B > Real.sin C) := 
  sorry

end sin_inequality_of_triangle_l71_71869


namespace cos_arcsin_l71_71488

theorem cos_arcsin (h : 8^2 + 15^2 = 17^2) : Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
by
  have opp_hyp_pos : 0 ≤ 8 / 17 := by norm_num
  have hyp_pos : 0 < 17 := by norm_num
  have opp_le_hyp : 8 / 17 ≤ 1 := by norm_num
  sorry

end cos_arcsin_l71_71488


namespace negation_of_universal_proposition_l71_71873

theorem negation_of_universal_proposition :
  (∀ x : ℝ, x^2 + 1 > 0) → ¬(∃ x : ℝ, x^2 + 1 ≤ 0) := sorry

end negation_of_universal_proposition_l71_71873


namespace football_team_total_progress_l71_71462

theorem football_team_total_progress :
  let play1 := -5
  let play2 := 13
  let play3 := -2 * play1
  let play4 := play3 / 2
  play1 + play2 + play3 + play4 = 3 :=
by
  sorry

end football_team_total_progress_l71_71462


namespace distances_perimeter_inequality_l71_71627

variable {Point Polygon : Type}

-- Definitions for the conditions
variables (O : Point) (M : Polygon)
variable (ρ : ℝ) -- perimeter of M
variable (d : ℝ) -- sum of distances to each vertex of M from O
variable (h : ℝ) -- sum of distances to each side of M from O

-- The theorem statement
theorem distances_perimeter_inequality :
  d^2 - h^2 ≥ ρ^2 / 4 :=
by
  sorry

end distances_perimeter_inequality_l71_71627


namespace electrical_bill_undetermined_l71_71690

theorem electrical_bill_undetermined
    (gas_bill : ℝ)
    (gas_paid_fraction : ℝ)
    (additional_gas_payment : ℝ)
    (water_bill : ℝ)
    (water_paid_fraction : ℝ)
    (internet_bill : ℝ)
    (internet_payments : ℝ)
    (payment_amounts: ℝ)
    (total_remaining : ℝ) :
    gas_bill = 40 →
    gas_paid_fraction = 3 / 4 →
    additional_gas_payment = 5 →
    water_bill = 40 →
    water_paid_fraction = 1 / 2 →
    internet_bill = 25 →
    internet_payments = 4 * 5 →
    total_remaining = 30 →
    (∃ electricity_bill : ℝ, true) -> 
    false := by
  intro gas_bill_eq gas_paid_fraction_eq additional_gas_payment_eq
  intro water_bill_eq water_paid_fraction_eq
  intro internet_bill_eq internet_payments_eq 
  intro total_remaining_eq 
  intro exists_electricity_bill 
  sorry -- Proof that the electricity bill cannot be determined

end electrical_bill_undetermined_l71_71690


namespace cost_of_paving_l71_71889

-- declaring the definitions and the problem statement
def length_of_room := 5.5
def width_of_room := 4
def rate_per_sq_meter := 700

theorem cost_of_paving (length : ℝ) (width : ℝ) (rate : ℝ) : length = 5.5 → width = 4 → rate = 700 → (length * width * rate) = 15400 :=
by
  intros h_length h_width h_rate
  rw [h_length, h_width, h_rate]
  sorry

end cost_of_paving_l71_71889


namespace calc_f_at_3_l71_71602

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem calc_f_at_3 : f 3 = 328 := 
sorry

end calc_f_at_3_l71_71602


namespace blake_spending_on_oranges_l71_71090

theorem blake_spending_on_oranges (spending_on_oranges spending_on_apples spending_on_mangoes initial_amount change_amount: ℝ)
  (h1 : spending_on_apples = 50)
  (h2 : spending_on_mangoes = 60)
  (h3 : initial_amount = 300)
  (h4 : change_amount = 150)
  (h5 : initial_amount - change_amount = spending_on_oranges + spending_on_apples + spending_on_mangoes) :
  spending_on_oranges = 40 := by
  sorry

end blake_spending_on_oranges_l71_71090


namespace min_containers_needed_l71_71091

def container_capacity : ℕ := 500
def required_tea : ℕ := 5000

theorem min_containers_needed (n : ℕ) : n * container_capacity ≥ required_tea → n = 10 :=
sorry

end min_containers_needed_l71_71091


namespace virginia_eggs_l71_71225

theorem virginia_eggs (initial_eggs : ℕ) (taken_eggs : ℕ) (result_eggs : ℕ) 
  (h_initial : initial_eggs = 200) 
  (h_taken : taken_eggs = 37) 
  (h_calculation: result_eggs = initial_eggs - taken_eggs) :
result_eggs = 163 :=
by {
  sorry
}

end virginia_eggs_l71_71225


namespace Beast_of_War_running_time_l71_71589

theorem Beast_of_War_running_time 
  (M : ℕ) 
  (AE : ℕ) 
  (BoWAC : ℕ)
  (h1 : M = 120)
  (h2 : AE = M - 30)
  (h3 : BoWAC = AE + 10) : 
  BoWAC = 100 
  := 
sorry

end Beast_of_War_running_time_l71_71589


namespace gcd_119_34_l71_71774

theorem gcd_119_34 : Nat.gcd 119 34 = 17 := by
  sorry

end gcd_119_34_l71_71774


namespace stratified_sampling_is_reasonable_l71_71085

-- Defining our conditions and stating our theorem
def flat_land := 150
def ditch_land := 30
def sloped_land := 90
def total_acres := 270
def sampled_acres := 18
def sampling_ratio := sampled_acres / total_acres

def flat_land_sampled := flat_land * sampling_ratio
def ditch_land_sampled := ditch_land * sampling_ratio
def sloped_land_sampled := sloped_land * sampling_ratio

theorem stratified_sampling_is_reasonable :
  flat_land_sampled = 10 ∧
  ditch_land_sampled = 2 ∧
  sloped_land_sampled = 6 := 
by
  sorry

end stratified_sampling_is_reasonable_l71_71085


namespace students_6_to_8_hours_study_l71_71544

-- Condition: 100 students were surveyed
def total_students : ℕ := 100

-- Hypothetical function representing the number of students studying for a specific range of hours based on the histogram
def histogram_students (lower_bound upper_bound : ℕ) : ℕ :=
  sorry  -- this would be defined based on actual histogram data

-- Question: Prove the number of students who studied for 6 to 8 hours
theorem students_6_to_8_hours_study : histogram_students 6 8 = 30 :=
  sorry -- the expected answer based on the histogram data

end students_6_to_8_hours_study_l71_71544


namespace sector_radius_l71_71074

theorem sector_radius (θ : ℝ) (s : ℝ) (R : ℝ) 
  (hθ : θ = 150)
  (hs : s = (5 / 2) * Real.pi)
  : (θ / 360) * (2 * Real.pi * R) = (5 / 2) * Real.pi → 
  R = 3 := 
sorry

end sector_radius_l71_71074


namespace tea_leaves_costs_l71_71053

theorem tea_leaves_costs (a_1 b_1 a_2 b_2 : ℕ) (c_A c_B : ℝ) :
  a_1 * c_A = 4000 ∧ 
  b_1 * c_B = 8400 ∧ 
  b_1 = a_1 + 10 ∧ 
  c_B = 1.4 * c_A ∧ 
  a_2 + b_2 = 100 ∧ 
  (300 - c_A) * (a_2 / 2) + (300 * 0.7 - c_A) * (a_2 / 2) + 
  (400 - c_B) * (b_2 / 2) + (400 * 0.7 - c_B) * (b_2 / 2) = 5800 
  → c_A = 200 ∧ c_B = 280 ∧ a_2 = 40 ∧ b_2 = 60 := 
sorry

end tea_leaves_costs_l71_71053


namespace cost_of_each_pack_l71_71384

theorem cost_of_each_pack (num_packs : ℕ) (total_paid : ℝ) (change_received : ℝ) 
(h1 : num_packs = 3) (h2 : total_paid = 20) (h3 : change_received = 11) : 
(total_paid - change_received) / num_packs = 3 := by
  sorry

end cost_of_each_pack_l71_71384


namespace gcd_ab_l71_71716

def a := 59^7 + 1
def b := 59^7 + 59^3 + 1

theorem gcd_ab : Nat.gcd a b = 1 := by
  sorry

end gcd_ab_l71_71716


namespace correct_option_l71_71007

-- Definitions based on the problem's conditions
def option_A (x : ℝ) : Prop := x^2 * x^4 = x^8
def option_B (x : ℝ) : Prop := (x^2)^3 = x^5
def option_C (x : ℝ) : Prop := x^2 + x^2 = 2 * x^2
def option_D (x : ℝ) : Prop := (3 * x)^2 = 3 * x^2

-- Theorem stating that out of the given options, option C is correct
theorem correct_option (x : ℝ) : option_C x :=
by {
  sorry
}

end correct_option_l71_71007


namespace average_height_l71_71130

def heights : List ℕ := [145, 142, 138, 136, 143, 146, 138, 144, 137, 141]

theorem average_height :
  (heights.sum : ℕ) / heights.length = 141 := by
  sorry

end average_height_l71_71130


namespace problem_D_l71_71734

theorem problem_D (a b c : ℝ) (h : |a^2 + b + c| + |a + b^2 - c| ≤ 1) : a^2 + b^2 + c^2 < 100 := 
sorry

end problem_D_l71_71734


namespace cone_volume_l71_71598

theorem cone_volume (r h : ℝ) (h_cylinder_vol : π * r^2 * h = 72 * π) : 
  (1 / 3) * π * r^2 * (h / 2) = 12 * π := by
  sorry

end cone_volume_l71_71598


namespace sum_of_all_three_digit_positive_even_integers_l71_71287

def sum_of_three_digit_even_integers : ℕ :=
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_of_all_three_digit_positive_even_integers :
  sum_of_three_digit_even_integers = 247050 :=
by
  -- proof to be completed
  sorry

end sum_of_all_three_digit_positive_even_integers_l71_71287


namespace smallest_possible_value_of_c_l71_71988

theorem smallest_possible_value_of_c
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (H : ∀ x : ℝ, (a * Real.sin (b * x + c)) ≤ (a * Real.sin (b * 0 + c))) :
  c = Real.pi / 2 :=
by
  sorry

end smallest_possible_value_of_c_l71_71988


namespace symmetric_circle_eq_l71_71590

-- Define the original circle equation
def originalCircle (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 4

-- Define the equation of the circle symmetric to the original with respect to the y-axis
def symmetricCircle (x y : ℝ) : Prop :=
  (x + 1) ^ 2 + (y - 2) ^ 2 = 4

-- Theorem to prove that the symmetric circle equation is correct
theorem symmetric_circle_eq :
  ∀ x y : ℝ, originalCircle x y → symmetricCircle (-x) y := 
by
  sorry

end symmetric_circle_eq_l71_71590


namespace value_of_sum_ratio_l71_71431

theorem value_of_sum_ratio (w x y: ℝ) (hx: w / x = 1 / 3) (hy: w / y = 2 / 3) : (x + y) / y = 3 :=
sorry

end value_of_sum_ratio_l71_71431


namespace complement_of_M_with_respect_to_U_l71_71248

noncomputable def U : Set ℕ := {1, 2, 3, 4}
noncomputable def M : Set ℕ := {1, 2, 3}

theorem complement_of_M_with_respect_to_U :
  (U \ M) = {4} :=
by
  sorry

end complement_of_M_with_respect_to_U_l71_71248


namespace greatest_value_of_x_l71_71990

theorem greatest_value_of_x : ∀ x : ℝ, 4*x^2 + 6*x + 3 = 5 → x ≤ 1/2 :=
by
  intro x
  intro h
  sorry

end greatest_value_of_x_l71_71990


namespace calculation_l71_71897

variable (x y z : ℕ)

theorem calculation (h1 : x + y + z = 20) (h2 : x + y - z = 8) :
  x + y = 14 :=
  sorry

end calculation_l71_71897


namespace min_value_of_u_l71_71029

theorem min_value_of_u (x y : ℝ) (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) (hxy : x * y = -1) :
  (∀ u, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) → u ≥ (12 / 5)) :=
by
  sorry

end min_value_of_u_l71_71029


namespace quadratic_inequality_solution_l71_71705

theorem quadratic_inequality_solution (a : ℝ) :
  ((0 ≤ a ∧ a < 3) → ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) :=
  sorry

end quadratic_inequality_solution_l71_71705


namespace power_of_sqrt2_minus_1_l71_71024

noncomputable def a (n : ℕ) : ℝ := (Real.sqrt 2 - 1) ^ n
noncomputable def b (n : ℕ) : ℝ := (Real.sqrt 2 + 1) ^ n
noncomputable def c (n : ℕ) : ℝ := (b n + a n) / 2
noncomputable def d (n : ℕ) : ℝ := (b n - a n) / 2

theorem power_of_sqrt2_minus_1 (n : ℕ) : a n = Real.sqrt (d n ^ 2 + 1) - Real.sqrt (d n ^ 2) :=
by
  sorry

end power_of_sqrt2_minus_1_l71_71024


namespace calculate_expression_l71_71010

theorem calculate_expression : ((18^18 / 18^17)^3 * 8^3) / 2^9 = 5832 := by
  sorry

end calculate_expression_l71_71010


namespace fraction_identity_l71_71819

theorem fraction_identity (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end fraction_identity_l71_71819


namespace a_share_is_6300_l71_71801

noncomputable def investment_split (x : ℝ) :  ℝ × ℝ × ℝ :=
  let a_share := x * 12
  let b_share := 2 * x * 6
  let c_share := 3 * x * 4
  (a_share, b_share, c_share)

noncomputable def total_gain : ℝ := 18900

noncomputable def a_share_calculation : ℝ :=
  let (a_share, b_share, c_share) := investment_split 1
  total_gain / (a_share + b_share + c_share) * a_share

theorem a_share_is_6300 : a_share_calculation = 6300 := by
  -- Here, you would provide the proof, but for now we skip it.
  sorry

end a_share_is_6300_l71_71801


namespace jack_time_to_school_l71_71802

noncomputable def dave_speed : ℚ := 8000 -- cm/min
noncomputable def distance_to_school : ℚ := 160000 -- cm
noncomputable def jack_speed : ℚ := 7650 -- cm/min
noncomputable def jack_start_delay : ℚ := 10 -- min

theorem jack_time_to_school : (distance_to_school / jack_speed) - jack_start_delay = 10.92 :=
by
  sorry

end jack_time_to_school_l71_71802


namespace cos_angle_identity_l71_71374

theorem cos_angle_identity (a : ℝ) (h : Real.sin (π / 6 - a) - Real.cos a = 1 / 3) :
  Real.cos (2 * a + π / 3) = 7 / 9 :=
by
  sorry

end cos_angle_identity_l71_71374


namespace range_of_m_l71_71065

variable (m t : ℝ)

namespace proof_problem

def proposition_p : Prop :=
  ∀ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) = 1) → (t + 2) * (t - 10) < 0

def proposition_q (m : ℝ) : Prop :=
  -m < t ∧ t < m + 1 ∧ m > 0

theorem range_of_m :
  (∃ t, proposition_q m t) → proposition_p t → 0 < m ∧ m ≤ 2 := by
  sorry

end proof_problem

end range_of_m_l71_71065


namespace smallest_positive_integer_n_l71_71240

theorem smallest_positive_integer_n (n : ℕ) (h : n > 0) : 3^n ≡ n^3 [MOD 5] ↔ n = 3 :=
sorry

end smallest_positive_integer_n_l71_71240


namespace heptagon_diagonals_l71_71308

theorem heptagon_diagonals : (7 * (7 - 3)) / 2 = 14 := 
by
  rfl

end heptagon_diagonals_l71_71308


namespace Cheryl_more_eggs_than_others_l71_71042

-- Definitions based on conditions
def KevinEggs : Nat := 5
def BonnieEggs : Nat := 13
def GeorgeEggs : Nat := 9
def CherylEggs : Nat := 56

-- Main theorem statement
theorem Cheryl_more_eggs_than_others : (CherylEggs - (KevinEggs + BonnieEggs + GeorgeEggs) = 29) :=
by
  -- Proof would go here
  sorry

end Cheryl_more_eggs_than_others_l71_71042


namespace interest_rate_per_annum_l71_71432

-- Given conditions
variables (BG TD t : ℝ) (FV r : ℝ)
axiom bg_eq : BG = 6
axiom td_eq : TD = 50
axiom t_eq : t = 1
axiom bankers_gain_eq : BG = FV * r * t - (FV - TD) * r * t

-- Proof problem
theorem interest_rate_per_annum : r = 0.12 :=
by sorry

end interest_rate_per_annum_l71_71432


namespace f_constant_1_l71_71202

theorem f_constant_1 (f : ℕ → ℕ) (h1 : ∀ n : ℕ, 0 < n → f (n + f n) = f n)
  (h2 : ∃ n0 : ℕ, 0 < n0 ∧ f n0 = 1) : ∀ n : ℕ, f n = 1 := 
by
  sorry

end f_constant_1_l71_71202


namespace isosceles_right_triangle_area_l71_71101

-- Define the isosceles right triangle and its properties

theorem isosceles_right_triangle_area 
  (h : ℝ)
  (hyp : h = 6) :
  let l : ℝ := h / Real.sqrt 2
  let A : ℝ := (l^2) / 2
  A = 9 :=
by
  -- The proof steps are skipped with sorry
  sorry

end isosceles_right_triangle_area_l71_71101


namespace happy_valley_zoo_animal_arrangement_l71_71871

theorem happy_valley_zoo_animal_arrangement :
  let parrots := 5
  let dogs := 3
  let cats := 4
  let total_animals := parrots + dogs + cats
  (total_animals = 12) →
    (∃ no_of_ways_to_arrange,
      no_of_ways_to_arrange = 2 * (parrots.factorial) * (dogs.factorial) * (cats.factorial) ∧
      no_of_ways_to_arrange = 34560) :=
by
  sorry

end happy_valley_zoo_animal_arrangement_l71_71871


namespace basketball_games_played_l71_71984

theorem basketball_games_played (G : ℕ) (H1 : 35 ≤ G) (H2 : 25 ≥ 0) (H3 : 64 = 100 * (48 / (G + 25))):
  G = 50 :=
sorry

end basketball_games_played_l71_71984


namespace quotient_of_2213_div_13_in_base4_is_53_l71_71625

-- Definitions of the numbers in base 4
def n₁ : ℕ := 2 * 4^3 + 2 * 4^2 + 1 * 4^1 + 3 * 4^0  -- 2213_4 in base 10
def n₂ : ℕ := 1 * 4^1 + 3 * 4^0  -- 13_4 in base 10

-- The correct quotient in base 4 (converted from quotient in base 10)
def expected_quotient : ℕ := 5 * 4^1 + 3 * 4^0  -- 53_4 in base 10

-- The proposition we want to prove
theorem quotient_of_2213_div_13_in_base4_is_53 : n₁ / n₂ = expected_quotient := by
  sorry

end quotient_of_2213_div_13_in_base4_is_53_l71_71625


namespace radius_of_smaller_molds_l71_71815

noncomputable def volumeOfHemisphere (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3

theorem radius_of_smaller_molds (r : ℝ) :
  volumeOfHemisphere 2 = 64 * volumeOfHemisphere r → r = 1 / 2 :=
by
  intro h
  sorry

end radius_of_smaller_molds_l71_71815


namespace arithmetic_sqrt_9_l71_71695

def arithmetic_sqrt (x : ℕ) : ℕ :=
  if h : 0 ≤ x then Nat.sqrt x else 0

theorem arithmetic_sqrt_9 : arithmetic_sqrt 9 = 3 :=
by {
  sorry
}

end arithmetic_sqrt_9_l71_71695


namespace picture_size_l71_71681

theorem picture_size (total_pics_A : ℕ) (size_A : ℕ) (total_pics_B : ℕ) (C : ℕ)
  (hA : total_pics_A * size_A = C) (hB : total_pics_B = 3000) : 
  (C / total_pics_B = 8) :=
by
  sorry

end picture_size_l71_71681


namespace cubic_sum_l71_71206

theorem cubic_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 14) : x ^ 3 + y ^ 3 = 580 :=
by 
  sorry

end cubic_sum_l71_71206


namespace darkCubeValidPositions_l71_71948

-- Conditions:
-- 1. The structure is made up of twelve identical cubes.
-- 2. The dark cube must be relocated to a position where the surface area remains unchanged.
-- 3. The cubes must touch each other with their entire faces.
-- 4. The positions of the light cubes cannot be changed.

-- Let's define the structure and the conditions in Lean.

structure Cube :=
  (id : ℕ) -- unique identifier for each cube

structure Position :=
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)

structure Configuration :=
  (cubes : List Cube)
  (positions : Cube → Position)

def initialCondition (config : Configuration) : Prop :=
  config.cubes.length = 12

def surfaceAreaUnchanged (config : Configuration) (darkCube : Cube) (newPos : Position) : Prop :=
  sorry -- This predicate should capture the logic that the surface area remains unchanged

def validPositions (config : Configuration) (darkCube : Cube) : List Position :=
  sorry -- This function should return the list of valid positions for the dark cube

-- Main theorem: The number of valid positions for the dark cube to maintain the surface area.
theorem darkCubeValidPositions (config : Configuration) (darkCube : Cube) :
    initialCondition config →
    (validPositions config darkCube).length = 3 :=
  by
  sorry

end darkCubeValidPositions_l71_71948


namespace problem_solution_l71_71245

-- Definitions and assumptions
variables (priceA priceB : ℕ)
variables (numBooksA numBooksB totalBooks : ℕ)
variables (costPriceA : priceA = 45)
variables (costPriceB : priceB = 65)
variables (totalCost : priceA * numBooksA + priceB * numBooksB ≤ 3550)
variables (totalBooksEq : numBooksA + numBooksB = 70)

-- Proof problem
theorem problem_solution :
  priceA = 45 ∧ priceB = 65 ∧ ∃ (numBooksA : ℕ), numBooksA ≥ 50 :=
by
  sorry

end problem_solution_l71_71245


namespace compute_expression_l71_71923

theorem compute_expression : 7^2 - 2 * 5 + 4^2 / 2 = 47 := by
  sorry

end compute_expression_l71_71923


namespace michael_birth_year_l71_71668

theorem michael_birth_year (first_imo_year : ℕ) (annual_event : ∀ n : ℕ, n > 0 → (first_imo_year + n) ≥ first_imo_year) 
  (michael_age_at_10th_imo : ℕ) (imo_count : ℕ) 
  (H1 : first_imo_year = 1959) (H2 : imo_count = 10) (H3 : michael_age_at_10th_imo = 15) : 
  (first_imo_year + imo_count - 1 - michael_age_at_10th_imo = 1953) := 
by 
  sorry

end michael_birth_year_l71_71668


namespace find_a_value_l71_71491

-- Problem statement
theorem find_a_value (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) :
  (∀ x : ℝ, x^2 + 2 * x^10 = a + a1 * (x+1) + a2 * (x+1)^2 + a3 * (x+1)^3 + a4 * (x+1)^4 + a5 * (x+1)^5 + a6 * (x+1)^6 + a7 * (x+1)^7 + a8 * (x+1)^8 + a9 * (x+1)^9 + a10 * (x+1)^(10)) → a = 3 :=
by sorry

end find_a_value_l71_71491


namespace sqrt_205_between_14_and_15_l71_71587

theorem sqrt_205_between_14_and_15 : 14 < Real.sqrt 205 ∧ Real.sqrt 205 < 15 := 
by
  sorry

end sqrt_205_between_14_and_15_l71_71587


namespace eight_b_plus_one_composite_l71_71758

theorem eight_b_plus_one_composite (a b : ℕ) (h₀ : a > b)
  (h₁ : a - b = 5 * b^2 - 4 * a^2) : ∃ (n m : ℕ), 1 < n ∧ 1 < m ∧ (8 * b + 1) = n * m :=
by
  sorry

end eight_b_plus_one_composite_l71_71758


namespace solution_set_eq_l71_71591

theorem solution_set_eq : { x : ℝ | |x| * (x - 2) ≥ 0 } = { x : ℝ | x ≥ 2 ∨ x = 0 } := by
  sorry

end solution_set_eq_l71_71591


namespace moles_of_KHSO4_formed_l71_71170

-- Chemical reaction definition
def reaction (n_KOH n_H2SO4 : ℕ) : ℕ :=
  if n_KOH = n_H2SO4 then n_KOH else 0

-- Given conditions
def moles_KOH : ℕ := 2
def moles_H2SO4 : ℕ := 2

-- Proof statement to be proved
theorem moles_of_KHSO4_formed : reaction moles_KOH moles_H2SO4 = 2 :=
by sorry

end moles_of_KHSO4_formed_l71_71170


namespace find_m_n_l71_71272

theorem find_m_n (m n : ℕ) (positive_m : 0 < m) (positive_n : 0 < n)
  (h1 : m = 3) (h2 : n = 4) :
    Real.arctan (1 / 3) + Real.arctan (1 / 4) + Real.arctan (1 / m) + Real.arctan (1 / n) = π / 2 :=
  by 
    -- Placeholder for the proof
    sorry

end find_m_n_l71_71272


namespace sum_of_digits_0_to_2012_l71_71779

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def sum_of_digits_in_range (a b : Nat) : Nat :=
  ((List.range (b + 1)).drop a).map sum_of_digits |>.sum

theorem sum_of_digits_0_to_2012 : 
  sum_of_digits_in_range 0 2012 = 28077 := 
by
  sorry

end sum_of_digits_0_to_2012_l71_71779


namespace percent_fewer_than_50000_is_75_l71_71458

-- Define the given conditions as hypotheses
variables {P_1 P_2 P_3 P_4 : ℝ}
variable (h1 : P_1 = 0.35)
variable (h2 : P_2 = 0.40)
variable (h3 : P_3 = 0.15)
variable (h4 : P_4 = 0.10)

-- Define the percentage of counties with fewer than 50,000 residents
def percent_fewer_than_50000 (P_1 P_2 : ℝ) : ℝ :=
  P_1 + P_2

-- The theorem statement we need to prove
theorem percent_fewer_than_50000_is_75 (h1 : P_1 = 0.35) (h2 : P_2 = 0.40) :
  percent_fewer_than_50000 P_1 P_2 = 0.75 :=
by
  sorry

end percent_fewer_than_50000_is_75_l71_71458


namespace tutors_meet_after_84_days_l71_71061

theorem tutors_meet_after_84_days :
  let jaclyn := 3
  let marcelle := 4
  let susanna := 6
  let wanda := 7
  Nat.lcm (Nat.lcm (Nat.lcm jaclyn marcelle) susanna) wanda = 84 := by
  sorry

end tutors_meet_after_84_days_l71_71061


namespace max_value_of_3x_plus_4y_l71_71558

theorem max_value_of_3x_plus_4y (x y : ℝ) 
(h : x^2 + y^2 = 14 * x + 6 * y + 6) : 
3 * x + 4 * y ≤ 73 := 
sorry

end max_value_of_3x_plus_4y_l71_71558


namespace maximum_distance_correct_l71_71341

noncomputable def maximum_distance 
  (m : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (intersection : (x y : ℝ) → (x + m * y = 0) ∧ (m * x - y - 2 * m + 4 = 0) → P = (x, y)) 
  (distance : (x y : ℝ) → (x - 1) * Real.cos θ + (y - 2) * Real.sin θ = 3) : 
  ℝ :=
3 + Real.sqrt 5

theorem maximum_distance_correct 
  (m : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (intersection : (x y : ℝ) → (x + m * y = 0) ∧ (m * x - y - 2 * m + 4 = 0) → P = (x, y)) 
  (distance : (x y : ℝ) → (x - 1) * Real.cos θ + (y - 2) * Real.sin θ = 3) : 
  maximum_distance m θ P intersection distance = 3 + Real.sqrt 5 := 
sorry

end maximum_distance_correct_l71_71341


namespace total_interest_paid_l71_71641

-- Define the problem as a theorem in Lean 4
theorem total_interest_paid
  (initial_investment : ℝ)
  (interest_6_months : ℝ)
  (interest_10_months : ℝ)
  (interest_18_months : ℝ)
  (total_interest : ℝ) :
  initial_investment = 10000 ∧ 
  interest_6_months = 0.02 * initial_investment ∧
  interest_10_months = 0.03 * (initial_investment + interest_6_months) ∧
  interest_18_months = 0.04 * (initial_investment + interest_6_months + interest_10_months) ∧
  total_interest = interest_6_months + interest_10_months + interest_18_months →
  total_interest = 926.24 :=
by
  sorry

end total_interest_paid_l71_71641


namespace cube_volume_l71_71040

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 294) : s^3 = 343 := 
by 
  sorry

end cube_volume_l71_71040


namespace garden_area_eq_450_l71_71121

theorem garden_area_eq_450
  (width length : ℝ)
  (fencing : ℝ := 60) 
  (length_eq_twice_width : length = 2 * width)
  (fencing_eq : 2 * width + length = fencing) :
  width * length = 450 := by
  sorry

end garden_area_eq_450_l71_71121


namespace different_pronunciation_in_group_C_l71_71526

theorem different_pronunciation_in_group_C :
  let groupC := [("戏谑", "xuè"), ("虐待", "nüè"), ("瘠薄", "jí"), ("脊梁", "jǐ"), ("赝品", "yàn"), ("义愤填膺", "yīng")]
  ∀ {a : String} {b : String}, (a, b) ∈ groupC → a ≠ b :=
by
  intro groupC h
  sorry

end different_pronunciation_in_group_C_l71_71526


namespace polynomial_roots_condition_l71_71916

open Real

def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem polynomial_roots_condition (a b : ℤ) (h1 : ∀ x ≠ 0, f (x + x⁻¹) a b = f x a b + f x⁻¹ a b) (h2 : ∃ p q : ℤ, f p a b = 0 ∧ f q a b = 0) : a^2 + b^2 = 13 := by
  sorry

end polynomial_roots_condition_l71_71916


namespace arithmetic_sequence_first_term_l71_71862

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 5 = 9) (h2 : 2 * a 3 = a 2 + 6) : a 1 = -3 :=
by
  -- a_5 = a_1 + 4d
  have h3 : a 5 = a 1 + 4 * d := sorry
  
  -- 2a_3 = a_2 + 6, which means 2 * (a_1 + 2d) = (a_1 + d) + 6
  have h4 : 2 * (a 1 + 2 * d) = (a 1 + d) + 6 := sorry
  
  -- solve the system of linear equations to find a_1 = -3
  sorry

end arithmetic_sequence_first_term_l71_71862


namespace greatest_possible_difference_l71_71293

theorem greatest_possible_difference (x y : ℝ) (hx1 : 6 < x) (hx2 : x < 10) (hy1 : 10 < y) (hy2 : y < 17) :
  ∃ n : ℤ, n = 9 ∧ ∀ x' y' : ℤ, (6 < x' ∧ x' < 10 ∧ 10 < y' ∧ y' < 17) → (y' - x' ≤ n) :=
by {
  -- here goes the actual proof
  sorry
}

end greatest_possible_difference_l71_71293


namespace smallest_possible_value_of_b_l71_71941

theorem smallest_possible_value_of_b (a b x : ℕ) (h_pos_x : 0 < x)
  (h_gcd : Nat.gcd a b = x + 7)
  (h_lcm : Nat.lcm a b = x * (x + 7))
  (h_a : a = 56)
  (h_x : x = 21) :
  b = 294 := by
  sorry

end smallest_possible_value_of_b_l71_71941


namespace plains_routes_count_l71_71165

-- Defining the total number of cities and the number of cities in each region
def total_cities : Nat := 100
def mountainous_cities : Nat := 30
def plains_cities : Nat := total_cities - mountainous_cities

-- Defining the number of routes established each year and over three years
def routes_per_year : Nat := 50
def total_routes : Nat := routes_per_year * 3

-- Defining the number of routes connecting pairs of mountainous cities
def mountainous_routes : Nat := 21

-- The statement to prove the number of routes connecting pairs of plains cities
theorem plains_routes_count :
  plains_cities = 70 →
  total_routes = 150 →
  mountainous_routes = 21 →
  3 * mountainous_cities - 2 * mountainous_routes = 48 →
  3 * plains_cities - 48 = 162 →
  81 = 81 := sorry

end plains_routes_count_l71_71165


namespace find_coordinates_of_symmetric_point_l71_71466

def point_on_parabola (A : ℝ × ℝ) : Prop :=
  A.2 = (A.1 - 1)^2 + 2

def symmetric_with_respect_to_axis (A A' : ℝ × ℝ) : Prop :=
  A'.1 = 2 * 1 - A.1 ∧ A'.2 = A.2

def correct_coordinates_of_A' (A' : ℝ × ℝ) : Prop :=
  A' = (3, 6)

theorem find_coordinates_of_symmetric_point (A A' : ℝ × ℝ)
  (hA : A = (-1, 6))
  (h_parabola : point_on_parabola A)
  (h_symmetric : symmetric_with_respect_to_axis A A') :
  correct_coordinates_of_A' A' :=
sorry

end find_coordinates_of_symmetric_point_l71_71466


namespace cuboid_third_edge_length_l71_71392

theorem cuboid_third_edge_length
  (l w : ℝ)
  (A : ℝ)
  (h : ℝ)
  (hl : l = 4)
  (hw : w = 5)
  (hA : A = 148)
  (surface_area_formula : A = 2 * (l * w + l * h + w * h)) :
  h = 6 :=
by
  sorry

end cuboid_third_edge_length_l71_71392


namespace domain_of_sqrt_log_function_l71_71919

def domain_of_function (x : ℝ) : Prop :=
  (1 ≤ x ∧ x < 2) ∨ (2 < x ∧ x < 3)

theorem domain_of_sqrt_log_function :
  ∀ x : ℝ, (x - 1 ≥ 0) → (x - 2 ≠ 0) → (-x^2 + 2 * x + 3 > 0) →
    domain_of_function x :=
by
  intros x h1 h2 h3
  unfold domain_of_function
  sorry

end domain_of_sqrt_log_function_l71_71919


namespace ab_sum_not_one_l71_71133

theorem ab_sum_not_one (a b : ℝ) : a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0 → a + b ≠ 1 :=
by
  intros h
  sorry

end ab_sum_not_one_l71_71133


namespace a_plus_b_eq_2007_l71_71142

theorem a_plus_b_eq_2007 (a b : ℕ) (ha : Prime a) (hb : Odd b)
  (h : a^2 + b = 2009) : a + b = 2007 :=
by
  sorry

end a_plus_b_eq_2007_l71_71142


namespace quadratic_factorization_l71_71892

theorem quadratic_factorization (p q x_1 x_2 : ℝ) (h1 : x_1 = 2) (h2 : x_2 = -3) 
    (h3 : x_1 + x_2 = -p) (h4 : x_1 * x_2 = q) : 
    (x - 2) * (x + 3) = x^2 + p * x + q :=
by
  sorry

end quadratic_factorization_l71_71892


namespace absolute_value_of_slope_l71_71259

noncomputable def circle_center1 : ℝ × ℝ := (14, 92)
noncomputable def circle_center2 : ℝ × ℝ := (17, 76)
noncomputable def circle_center3 : ℝ × ℝ := (19, 84)
noncomputable def radius : ℝ := 3
noncomputable def point_on_line : ℝ × ℝ := (17, 76)

theorem absolute_value_of_slope :
  ∃ m : ℝ, ∀ line : ℝ × ℝ → Prop,
    (line point_on_line) ∧ 
    (∀ p, (line p) → true) → 
    abs m = 24 := 
  sorry

end absolute_value_of_slope_l71_71259


namespace minimum_height_for_surface_area_geq_120_l71_71463

noncomputable def box_surface_area (x : ℝ) : ℝ :=
  6 * x^2 + 20 * x

theorem minimum_height_for_surface_area_geq_120 :
  ∃ (x : ℝ), (x ≥ 0) ∧ (box_surface_area x ≥ 120) ∧ (x + 5 = 9) := by
  sorry

end minimum_height_for_surface_area_geq_120_l71_71463


namespace mario_time_on_moving_sidewalk_l71_71056

theorem mario_time_on_moving_sidewalk (d w v : ℝ) (h_walk : d = 90 * w) (h_sidewalk : d = 45 * v) : 
  d / (w + v) = 30 :=
by
  sorry

end mario_time_on_moving_sidewalk_l71_71056


namespace possible_double_roots_l71_71457

theorem possible_double_roots (b₃ b₂ b₁ : ℤ) (s : ℤ) :
  s^2 ∣ 50 →
  (Polynomial.eval s (Polynomial.C 50 + Polynomial.C b₁ * Polynomial.X + Polynomial.C b₂ * Polynomial.X^2 + Polynomial.C b₃ * Polynomial.X^3 + Polynomial.X^4) = 0) →
  (Polynomial.eval s (Polynomial.derivative (Polynomial.C 50 + Polynomial.C b₁ * Polynomial.X + Polynomial.C b₂ * Polynomial.X^2 + Polynomial.C b₃ * Polynomial.X^3 + Polynomial.X^4)) = 0) →
  s = 1 ∨ s = -1 ∨ s = 5 ∨ s = -5 :=
by
  sorry

end possible_double_roots_l71_71457


namespace perpendicular_line_l71_71632

theorem perpendicular_line 
  (a b c : ℝ) 
  (p : ℝ × ℝ) 
  (h₁ : p = (-1, 3)) 
  (h₂ : a * (-1) + b * 3 + c = 0) 
  (h₃ : a * p.fst + b * p.snd + c = 0) 
  (hp : a = 1 ∧ b = -2 ∧ c = 3) : 
  ∃ a₁ b₁ c₁ : ℝ, 
  a₁ * (-1) + b₁ * 3 + c₁ = 0 ∧ a₁ = 2 ∧ b₁ = 1 ∧ c₁ = -1 := 
by 
  sorry

end perpendicular_line_l71_71632


namespace product_of_solutions_of_abs_eq_l71_71634

theorem product_of_solutions_of_abs_eq (x : ℝ) (h : |x - 5| - 4 = 3) : x * (if x = 12 then -2 else if x = -2 then 12 else 1) = -24 :=
by
  sorry

end product_of_solutions_of_abs_eq_l71_71634


namespace wrongly_noted_mark_l71_71739

theorem wrongly_noted_mark (n : ℕ) (avg_wrong avg_correct correct_mark : ℝ) (x : ℝ)
  (h1 : n = 30)
  (h2 : avg_wrong = 60)
  (h3 : avg_correct = 57.5)
  (h4 : correct_mark = 15)
  (h5 : n * avg_wrong - n * avg_correct = x - correct_mark)
  : x = 90 :=
sorry

end wrongly_noted_mark_l71_71739


namespace problem_statement_l71_71209

theorem problem_statement
  (f : ℝ → ℝ)
  (h0 : ∀ x, 0 <= x → x <= 1 → 0 <= f x)
  (h1 : ∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 
        (f x + f y) / 2 ≤ f ((x + y) / 2) + 1) :
  ∀ (u v w : ℝ), 
    0 ≤ u ∧ u < v ∧ v < w ∧ w ≤ 1 → 
    (w - v) / (w - u) * f u + (v - u) / (w - u) * f w ≤ f v + 2 :=
by
  intros u v w h
  sorry

end problem_statement_l71_71209


namespace esther_evening_speed_l71_71654

/-- Esther's average speed in the evening was 30 miles per hour -/
theorem esther_evening_speed : 
  let morning_speed := 45   -- miles per hour
  let total_commuting_time := 1 -- hour
  let morning_distance := 18  -- miles
  let evening_distance := 18  -- miles (same route)
  let time_morning := morning_distance / morning_speed
  let time_evening := total_commuting_time - time_morning
  let evening_speed := evening_distance / time_evening
  evening_speed = 30 := 
by sorry

end esther_evening_speed_l71_71654


namespace factorize_x4_minus_16y4_l71_71086

theorem factorize_x4_minus_16y4 (x y : ℚ) : 
  x^4 - 16 * y^4 = (x^2 + 4 * y^2) * (x + 2 * y) * (x - 2 * y) := 
by 
  sorry

end factorize_x4_minus_16y4_l71_71086


namespace car_circuit_velocity_solution_l71_71193

theorem car_circuit_velocity_solution
    (v_s v_p v_d : ℕ)
    (h1 : v_s < v_p)
    (h2 : v_p < v_d)
    (h3 : s = d)
    (h4 : s + p + d = 600)
    (h5 : (d : ℚ) / v_s + (p : ℚ) / v_p + (d : ℚ) / v_d = 50) :
    (v_s = 7 ∧ v_p = 12 ∧ v_d = 42) ∨
    (v_s = 8 ∧ v_p = 12 ∧ v_d = 24) ∨
    (v_s = 9 ∧ v_p = 12 ∧ v_d = 18) ∨
    (v_s = 10 ∧ v_p = 12 ∧ v_d = 15) :=
by
  sorry

end car_circuit_velocity_solution_l71_71193


namespace total_laundry_time_correct_l71_71338

-- Define the washing and drying times for each load
def whites_washing_time : Nat := 72
def whites_drying_time : Nat := 50
def darks_washing_time : Nat := 58
def darks_drying_time : Nat := 65
def colors_washing_time : Nat := 45
def colors_drying_time : Nat := 54

-- Define total times for each load
def whites_total_time : Nat := whites_washing_time + whites_drying_time
def darks_total_time : Nat := darks_washing_time + darks_drying_time
def colors_total_time : Nat := colors_washing_time + colors_drying_time

-- Define the total time for all three loads
def total_laundry_time : Nat := whites_total_time + darks_total_time + colors_total_time

-- The proof statement
theorem total_laundry_time_correct : total_laundry_time = 344 := by
  unfold total_laundry_time
  unfold whites_total_time darks_total_time colors_total_time
  unfold whites_washing_time whites_drying_time
  unfold darks_washing_time darks_drying_time
  unfold colors_washing_time colors_drying_time
  sorry

end total_laundry_time_correct_l71_71338


namespace recurring_decimal_difference_fraction_l71_71279

noncomputable def recurring_decimal_seventy_three := 73 / 99
noncomputable def decimal_seventy_three := 73 / 100

theorem recurring_decimal_difference_fraction :
  recurring_decimal_seventy_three - decimal_seventy_three = 73 / 9900 := sorry

end recurring_decimal_difference_fraction_l71_71279


namespace unique_solution_to_exponential_poly_equation_l71_71071

noncomputable def polynomial_has_unique_real_solution : Prop :=
  ∃! x : ℝ, (2 : ℝ)^(3 * x + 3) - 3 * (2 : ℝ)^(2 * x + 1) - (2 : ℝ)^x + 1 = 0

theorem unique_solution_to_exponential_poly_equation :
  polynomial_has_unique_real_solution :=
sorry

end unique_solution_to_exponential_poly_equation_l71_71071


namespace mother_nickels_eq_two_l71_71686

def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 9
def total_nickels : ℕ := 18

theorem mother_nickels_eq_two : (total_nickels = initial_nickels + dad_nickels + 2) :=
by
  sorry

end mother_nickels_eq_two_l71_71686


namespace part_one_part_two_l71_71430

theorem part_one (a b : ℝ) (h : a ≠ 0) : |a + b| + |a - b| ≥ 2 * |a| :=
by sorry

theorem part_two (x : ℝ) : |x - 1| + |x - 2| ≤ 2 ↔ (1 / 2 : ℝ) ≤ x ∧ x ≤ (5 / 2 : ℝ) :=
by sorry

end part_one_part_two_l71_71430


namespace find_a_l71_71038

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = Real.log (-a * x)) (h2 : ∀ x : ℝ, f (-x) = -f x) :
  a = 1 :=
by
  sorry

end find_a_l71_71038


namespace average_birth_rate_l71_71152

theorem average_birth_rate (B : ℕ) 
  (death_rate : ℕ := 3)
  (daily_net_increase : ℕ := 86400) 
  (intervals_per_day : ℕ := 86400 / 2) 
  (net_increase : ℕ := (B - death_rate) * intervals_per_day) : 
  net_increase = daily_net_increase → 
  B = 5 := 
sorry

end average_birth_rate_l71_71152


namespace lock_settings_are_5040_l71_71683

def num_unique_settings_for_lock : ℕ := 10 * 9 * 8 * 7

theorem lock_settings_are_5040 : num_unique_settings_for_lock = 5040 :=
by
  sorry

end lock_settings_are_5040_l71_71683


namespace sea_horses_count_l71_71417

theorem sea_horses_count (S P : ℕ) 
  (h1 : S / P = 5 / 11) 
  (h2 : P = S + 85) 
  : S = 70 := sorry

end sea_horses_count_l71_71417


namespace fraction_of_oil_sent_to_production_l71_71196

-- Definitions based on the problem's conditions
def initial_concentration : ℝ := 0.02
def replacement_concentration1 : ℝ := 0.03
def replacement_concentration2 : ℝ := 0.015
def final_concentration : ℝ := 0.02

-- Main theorem stating the fraction x is 1/2
theorem fraction_of_oil_sent_to_production (x : ℝ) (hx : x > 0) :
  (initial_concentration + (replacement_concentration1 - initial_concentration) * x) * (1 - x) +
  replacement_concentration2 * x = final_concentration →
  x = 0.5 :=
  sorry

end fraction_of_oil_sent_to_production_l71_71196


namespace value_of_a_l71_71645

theorem value_of_a (a : ℝ) : 
  (∀ (x : ℝ), (x < -4 ∨ x > 5) → x^2 + a * x + 20 > 0) → a = -1 :=
by
  sorry

end value_of_a_l71_71645


namespace sin_alpha_given_cos_alpha_plus_pi_over_3_l71_71362

theorem sin_alpha_given_cos_alpha_plus_pi_over_3 
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 3) = 1 / 5) : 
  Real.sin α = (2 * Real.sqrt 6 - Real.sqrt 3) / 10 := 
by 
  sorry

end sin_alpha_given_cos_alpha_plus_pi_over_3_l71_71362


namespace all_propositions_imply_l71_71211

variables (p q r : Prop)

theorem all_propositions_imply (hpqr : p ∧ q ∧ r)
                               (hnpqr : ¬p ∧ q ∧ ¬r)
                               (hpnqr : p ∧ ¬q ∧ r)
                               (hnpnqr : ¬p ∧ ¬q ∧ ¬r) :
  (p → q) ∨ r :=
by { sorry }

end all_propositions_imply_l71_71211


namespace intersection_solution_l71_71628

theorem intersection_solution (x : ℝ) (y : ℝ) (h₁ : y = 12 / (x^2 + 6)) (h₂ : x + y = 4) : x = 2 :=
by
  sorry

end intersection_solution_l71_71628


namespace PQ_is_10_5_l71_71445

noncomputable def PQ_length_proof_problem : Prop := 
  ∃ (PQ : ℝ),
    PQ = 10.5 ∧ 
    ∃ (ST : ℝ) (SU : ℝ),
      ST = 4.5 ∧ SU = 7.5 ∧ 
      ∃ (QR : ℝ) (PR : ℝ),
        QR = 21 ∧ PR = 15 ∧ 
        ∃ (angle_PQR angle_STU : ℝ),
          angle_PQR = 120 ∧ angle_STU = 120 ∧ 
          PQ / ST = PR / SU

theorem PQ_is_10_5 :
  PQ_length_proof_problem := sorry

end PQ_is_10_5_l71_71445


namespace total_savings_eighteen_l71_71569

theorem total_savings_eighteen :
  let fox_price := 15
  let pony_price := 18
  let discount_rate_sum := 50
  let fox_quantity := 3
  let pony_quantity := 2
  let pony_discount_rate := 50
  let total_price_without_discount := (fox_quantity * fox_price) + (pony_quantity * pony_price)
  let discounted_pony_price := (pony_price * (1 - (pony_discount_rate / 100)))
  let total_price_with_discount := (fox_quantity * fox_price) + (pony_quantity * discounted_pony_price)
  let total_savings := total_price_without_discount - total_price_with_discount
  total_savings = 18 :=
by sorry

end total_savings_eighteen_l71_71569


namespace other_root_of_quadratic_l71_71757

theorem other_root_of_quadratic 
  (a b c: ℝ) 
  (h : a * (b - c - d) * (1:ℝ)^2 + b * (c - a + d) * (1:ℝ) + c * (a - b - d) = 0) : 
  ∃ k: ℝ, k = c * (a - b - d) / (a * (b - c - d)) :=
sorry

end other_root_of_quadratic_l71_71757


namespace ratio_of_numbers_l71_71864

-- Definitions for the conditions
variable (S L : ℕ)

-- Given conditions
def condition1 : Prop := S + L = 44
def condition2 : Prop := S = 20
def condition3 : Prop := L = 6 * S

-- The theorem to be proven
theorem ratio_of_numbers (h1 : condition1 S L) (h2 : condition2 S) (h3 : condition3 S L) : L / S = 6 := 
  sorry

end ratio_of_numbers_l71_71864


namespace avg_weight_difference_l71_71363

-- Define the weights of the boxes following the given conditions.
def box1_weight : ℕ := 200
def box3_weight : ℕ := box1_weight + (25 * box1_weight / 100)
def box2_weight : ℕ := box3_weight + (20 * box3_weight / 100)
def box4_weight : ℕ := 350
def box5_weight : ℕ := box4_weight * 100 / 70

-- Define the average weight of the four heaviest boxes.
def avg_heaviest : ℕ := (box2_weight + box3_weight + box4_weight + box5_weight) / 4

-- Define the average weight of the four lightest boxes.
def avg_lightest : ℕ := (box1_weight + box2_weight + box3_weight + box4_weight) / 4

-- Define the difference between the average weights of the heaviest and lightest boxes.
def avg_difference : ℕ := avg_heaviest - avg_lightest

-- State the theorem with the expected result.
theorem avg_weight_difference : avg_difference = 75 :=
by
  -- Proof is not provided.
  sorry

end avg_weight_difference_l71_71363


namespace inscribed_circle_radius_l71_71253

noncomputable def semiPerimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def areaUsingHeron (a b c : ℝ) : ℝ :=
  let s := semiPerimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := semiPerimeter a b c
  let K := areaUsingHeron a b c
  K / s

theorem inscribed_circle_radius : inscribedCircleRadius 26 18 20 = Real.sqrt 31 :=
  sorry

end inscribed_circle_radius_l71_71253
