import Mathlib

namespace smallest_row_sum_greater_than_50_l131_131557

noncomputable def sum_interior_pascal (n : ℕ) : ℕ :=
  2^(n-1) - 2

theorem smallest_row_sum_greater_than_50 : ∃ n, sum_interior_pascal n > 50 ∧ (∀ m, m < n → sum_interior_pascal m ≤ 50) ∧ sum_interior_pascal 7 = 62 ∧ (sum_interior_pascal 7) % 2 = 0 :=
by
  sorry

end smallest_row_sum_greater_than_50_l131_131557


namespace big_bottles_sold_percentage_l131_131169

-- Definitions based on conditions
def small_bottles_initial : ℕ := 5000
def big_bottles_initial : ℕ := 12000
def small_bottles_sold_percentage : ℝ := 0.15
def total_bottles_remaining : ℕ := 14090

-- Question in Lean 4
theorem big_bottles_sold_percentage : 
  (12000 - (12000 * x / 100) + 5000 - (5000 * 15 / 100)) = 14090 → x = 18 :=
by
  intros h
  sorry

end big_bottles_sold_percentage_l131_131169


namespace amount_paid_is_200_l131_131645

-- Definitions of the costs and change received
def cost_of_pants := 140
def cost_of_shirt := 43
def cost_of_tie := 15
def change_received := 2

-- Total cost calculation
def total_cost := cost_of_pants + cost_of_shirt + cost_of_tie

-- Lean proof statement
theorem amount_paid_is_200 : total_cost + change_received = 200 := by
  -- Definitions ensure the total cost and change received are used directly from conditions
  sorry

end amount_paid_is_200_l131_131645


namespace sequence_inequality_l131_131132

theorem sequence_inequality (a : ℕ → ℕ) 
  (h_nonneg : ∀ n, 0 ≤ a n)
  (h_additive : ∀ m n, a (n + m) ≤ a n + a m) 
  (N n : ℕ) 
  (h_N_ge_n : N ≥ n) : 
  a n + a N ≤ n * a 1 + N / n * a n :=
sorry

end sequence_inequality_l131_131132


namespace triangle_is_isosceles_l131_131430

theorem triangle_is_isosceles (A B C a b c : ℝ) (h1 : c = 2 * a * Real.cos B) : 
  A = B → a = b := 
sorry

end triangle_is_isosceles_l131_131430


namespace race_participants_minimum_l131_131593

theorem race_participants_minimum : ∃ n : ℕ, 
  ((n - 1) % 3 = 0) ∧ 
  ((n - 1) % 4 = 0) ∧ 
  ((n - 1) % 5 = 0) ∧ 
  (∀ m : ℕ, 
    ((m - 1) % 3 = 0) ∧ 
    ((m - 1) % 4 = 0) ∧ 
    ((m - 1) % 5 = 0) → 
    n ≤ m) := 
sorry

end race_participants_minimum_l131_131593


namespace remainder_of_polynomial_l131_131907

def p (x : ℝ) : ℝ := x^4 + 2*x^2 + 5

theorem remainder_of_polynomial (x : ℝ) : p 2 = 29 :=
by
  sorry

end remainder_of_polynomial_l131_131907


namespace present_age_of_B_l131_131881

theorem present_age_of_B 
    (a b : ℕ) 
    (h1 : a + 10 = 2 * (b - 10)) 
    (h2 : a = b + 12) : 
    b = 42 := by 
  sorry

end present_age_of_B_l131_131881


namespace typeB_lines_l131_131591

noncomputable def isTypeBLine (line : Real → Real) : Prop :=
  ∃ P : ℝ × ℝ, line P.1 = P.2 ∧ (Real.sqrt ((P.1 + 5)^2 + P.2^2) - Real.sqrt ((P.1 - 5)^2 + P.2^2) = 6)

theorem typeB_lines :
  isTypeBLine (fun x => x + 1) ∧ isTypeBLine (fun x => 2) :=
by sorry

end typeB_lines_l131_131591


namespace hoseok_more_paper_than_minyoung_l131_131940

theorem hoseok_more_paper_than_minyoung : 
  ∀ (initial : ℕ) (minyoung_bought : ℕ) (hoseok_bought : ℕ), 
  initial = 150 →
  minyoung_bought = 32 →
  hoseok_bought = 49 →
  (initial + hoseok_bought) - (initial + minyoung_bought) = 17 :=
by
  intros initial minyoung_bought hoseok_bought h_initial h_min h_hos
  sorry

end hoseok_more_paper_than_minyoung_l131_131940


namespace g_neg_one_l131_131156

def g (d e f x : ℝ) : ℝ := d * x^9 - e * x^5 + f * x + 1

theorem g_neg_one {d e f : ℝ} (h : g d e f 1 = -1) : g d e f (-1) = 3 := by
  sorry

end g_neg_one_l131_131156


namespace silver_dollars_l131_131084

variable (C : ℕ)
variable (H : ℕ)
variable (P : ℕ)

theorem silver_dollars (h1 : H = P + 5) (h2 : P = C + 16) (h3 : C + P + H = 205) : C = 56 :=
by
  sorry

end silver_dollars_l131_131084


namespace rectangle_diagonal_length_l131_131772

theorem rectangle_diagonal_length (P L W k d : ℝ) 
  (h1 : P = 72) 
  (h2 : L / W = 3 / 2) 
  (h3 : L = 3 * k) 
  (h4 : W = 2 * k) 
  (h5 : P = 2 * (L + W))
  (h6 : d = Real.sqrt ((L^2) + (W^2))) :
  d = 25.96 :=
by
  sorry

end rectangle_diagonal_length_l131_131772


namespace train_crosses_signal_pole_l131_131152

theorem train_crosses_signal_pole 
  (length_train : ℝ) 
  (length_platform : ℝ) 
  (time_cross_platform : ℝ) 
  (speed : ℝ) 
  (time_cross_signal_pole : ℝ) : 
  length_train = 400 → 
  length_platform = 200 → 
  time_cross_platform = 45 → 
  speed = (length_train + length_platform) / time_cross_platform → 
  time_cross_signal_pole = length_train / speed -> 
  time_cross_signal_pole = 30 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [h1] at h5
  -- Add the necessary calculations here
  sorry

end train_crosses_signal_pole_l131_131152


namespace locus_equation_rectangle_perimeter_greater_l131_131188

open Real

theorem locus_equation (P : ℝ × ℝ) : 
  (abs P.2 = sqrt (P.1 ^ 2 + (P.2 - 1 / 2) ^ 2)) → (P.2 = P.1 ^ 2 + 1 / 4) :=
by
  intro h
  sorry

theorem rectangle_perimeter_greater (A B C D : ℝ × ℝ) :
  (A.2 = A.1 ^ 2 + 1 / 4) ∧ 
  (B.2 = B.1 ^ 2 + 1 / 4) ∧ 
  (C.2 = C.1 ^ 2 + 1 / 4) ∧ 
  (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) → 
  (2 * (dist A B + dist B C) > 3 * sqrt 3) :=
by
  intro h
  sorry

end locus_equation_rectangle_perimeter_greater_l131_131188


namespace derivative_f_at_zero_l131_131655

noncomputable def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then 4 * x * (1 - |x|) else 0

theorem derivative_f_at_zero : HasDerivAt f 4 0 :=
by
  -- Proof omitted
  sorry

end derivative_f_at_zero_l131_131655


namespace cyclic_sum_non_negative_equality_condition_l131_131367

theorem cyclic_sum_non_negative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b) / (a + b) + b^2 * (b - c) / (b + c) + c^2 * (c - a) / (c + a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b) / (a + b) + b^2 * (b - c) / (b + c) + c^2 * (c - a) / (c + a) = 0 ↔ a = b ∧ b = c :=
sorry

end cyclic_sum_non_negative_equality_condition_l131_131367


namespace aviana_brought_pieces_l131_131561

variable (total_people : ℕ) (fraction_eat_pizza : ℚ) (pieces_per_person : ℕ) (remaining_pieces : ℕ)

theorem aviana_brought_pieces (h1 : total_people = 15) 
                             (h2 : fraction_eat_pizza = 3 / 5) 
                             (h3 : pieces_per_person = 4) 
                             (h4 : remaining_pieces = 14) :
                             ∃ (brought_pieces : ℕ), brought_pieces = 50 :=
by sorry

end aviana_brought_pieces_l131_131561


namespace charlotte_total_dog_walking_time_l131_131146

def poodles_monday : ℕ := 4
def chihuahuas_monday : ℕ := 2
def poodles_tuesday : ℕ := 4
def chihuahuas_tuesday : ℕ := 2
def labradors_wednesday : ℕ := 4

def time_poodle : ℕ := 2
def time_chihuahua : ℕ := 1
def time_labrador : ℕ := 3

def total_time_monday : ℕ := poodles_monday * time_poodle + chihuahuas_monday * time_chihuahua
def total_time_tuesday : ℕ := poodles_tuesday * time_poodle + chihuahuas_tuesday * time_chihuahua
def total_time_wednesday : ℕ := labradors_wednesday * time_labrador

def total_time_week : ℕ := total_time_monday + total_time_tuesday + total_time_wednesday

theorem charlotte_total_dog_walking_time : total_time_week = 32 := by
  -- Lean allows us to state the theorem without proving it.
  sorry

end charlotte_total_dog_walking_time_l131_131146


namespace two_numbers_sum_l131_131339

theorem two_numbers_sum (N1 N2 : ℕ) (h1 : N1 % 10^5 = 0) (h2 : N2 % 10^5 = 0) 
  (h3 : N1 ≠ N2) (h4 : (Nat.divisors N1).card = 42) (h5 : (Nat.divisors N2).card = 42) : 
  N1 + N2 = 700000 := 
by
  sorry

end two_numbers_sum_l131_131339


namespace james_spends_252_per_week_l131_131458

noncomputable def cost_pistachios_per_ounce := 10 / 5
noncomputable def cost_almonds_per_ounce := 8 / 4
noncomputable def cost_walnuts_per_ounce := 12 / 6

noncomputable def daily_consumption_pistachios := 30 / 5
noncomputable def daily_consumption_almonds := 24 / 4
noncomputable def daily_consumption_walnuts := 18 / 3

noncomputable def weekly_consumption_pistachios := daily_consumption_pistachios * 7
noncomputable def weekly_consumption_almonds := daily_consumption_almonds * 7
noncomputable def weekly_consumption_walnuts := daily_consumption_walnuts * 7

noncomputable def weekly_cost_pistachios := weekly_consumption_pistachios * cost_pistachios_per_ounce
noncomputable def weekly_cost_almonds := weekly_consumption_almonds * cost_almonds_per_ounce
noncomputable def weekly_cost_walnuts := weekly_consumption_walnuts * cost_walnuts_per_ounce

noncomputable def total_weekly_cost := weekly_cost_pistachios + weekly_cost_almonds + weekly_cost_walnuts

theorem james_spends_252_per_week :
  total_weekly_cost = 252 := by
  sorry

end james_spends_252_per_week_l131_131458


namespace expr_1989_eval_expr_1990_eval_l131_131226

def nestedExpr : ℕ → ℤ
| 0     => 0
| (n+1) => -1 - (nestedExpr n)

-- Conditions translated into Lean definitions:
def expr_1989 := nestedExpr 1989
def expr_1990 := nestedExpr 1990

-- The proof statements:
theorem expr_1989_eval : expr_1989 = -1 := sorry
theorem expr_1990_eval : expr_1990 = 0 := sorry

end expr_1989_eval_expr_1990_eval_l131_131226


namespace relationship_between_sums_l131_131175

-- Conditions: four distinct positive integers
variables {a b c d : ℕ}
-- additional conditions: positive integers
variables (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)

-- Condition: a is the largest and d is the smallest
variables (a_largest : a > b ∧ a > c ∧ a > d)
variables (d_smallest : d < b ∧ d < c ∧ d < a)

-- Condition: a / b = c / d
variables (ratio_condition : a * d = b * c)

theorem relationship_between_sums :
  a + d > b + c :=
sorry

end relationship_between_sums_l131_131175


namespace molly_age_l131_131112

variable (S M : ℕ)

theorem molly_age (h1 : S / M = 4 / 3) (h2 : S + 6 = 38) : M = 24 :=
by
  sorry

end molly_age_l131_131112


namespace range_a_condition_l131_131946

theorem range_a_condition (a : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ a → x^2 ≤ 2 * x + 3) ↔ (1 / 2 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_a_condition_l131_131946


namespace area_of_circle_l131_131301

def circle_area (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 18 * y = -45

theorem area_of_circle :
  (∃ x y : ℝ, circle_area x y) → ∃ A : ℝ, A = 52 * Real.pi :=
by
  sorry

end area_of_circle_l131_131301


namespace simplify_and_evaluate_equals_l131_131085

noncomputable def simplify_and_evaluate (a : ℝ) : ℝ :=
  (a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2) / (a^2 + 2 * a) / (a - 2)

theorem simplify_and_evaluate_equals (a : ℝ) (h : a^2 + 2 * a - 8 = 0) : 
  simplify_and_evaluate a = 1 / 4 :=
sorry

end simplify_and_evaluate_equals_l131_131085


namespace grocery_store_spending_l131_131931

/-- Lenny has $84 initially. He spent $24 on video games and has $39 left.
We need to prove that he spent $21 at the grocery store. --/
theorem grocery_store_spending (initial_amount spent_on_video_games amount_left after_games_left : ℕ) 
    (h1 : initial_amount = 84)
    (h2 : spent_on_video_games = 24)
    (h3 : amount_left = 39)
    (h4 : after_games_left = initial_amount - spent_on_video_games) 
    : after_games_left - amount_left = 21 := 
sorry

end grocery_store_spending_l131_131931


namespace equilateral_triangle_side_length_l131_131436

theorem equilateral_triangle_side_length 
    (D A B C : ℝ × ℝ)
    (h_distances : dist D A = 2 ∧ dist D B = 3 ∧ dist D C = 5)
    (h_equilateral : dist A B = dist B C ∧ dist B C = dist C A) :
    dist A B = Real.sqrt 19 :=
by
    sorry -- Proof to be filled

end equilateral_triangle_side_length_l131_131436


namespace range_of_c_l131_131402

theorem range_of_c (a b c : ℝ) (h1 : 6 < a) (h2 : a < 10) (h3 : a / 2 ≤ b) (h4 : b ≤ 2 * a) (h5 : c = a + b) : 
  9 < c ∧ c < 30 :=
sorry

end range_of_c_l131_131402


namespace total_packs_sold_l131_131594

def packs_sold_village_1 : ℕ := 23
def packs_sold_village_2 : ℕ := 28

theorem total_packs_sold : packs_sold_village_1 + packs_sold_village_2 = 51 :=
by
  -- We acknowledge the correctness of the calculation.
  sorry

end total_packs_sold_l131_131594


namespace symmetric_point_of_M_origin_l131_131207

-- Define the point M with given coordinates
def M : (ℤ × ℤ) := (-3, -5)

-- The theorem stating that the symmetric point of M about the origin is (3, 5)
theorem symmetric_point_of_M_origin :
  let symmetric_point : (ℤ × ℤ) := (-M.1, -M.2)
  symmetric_point = (3, 5) :=
by
  -- (Proof should be filled)
  sorry

end symmetric_point_of_M_origin_l131_131207


namespace license_plate_combinations_l131_131186

theorem license_plate_combinations :
  let letters := 26
  let two_other_letters := Nat.choose 25 2
  let repeated_positions := Nat.choose 4 2
  let arrange_two_letters := 2
  let first_digit_choices := 10
  let second_digit_choices := 9
  letters * two_other_letters * repeated_positions * arrange_two_letters * first_digit_choices * second_digit_choices = 8424000 :=
  sorry

end license_plate_combinations_l131_131186


namespace problem1_problem2_l131_131246

namespace MathProofs

theorem problem1 : (0.25 * 4 - ((5 / 6) + (1 / 12)) * (6 / 5)) = (1 / 10) := by
  sorry

theorem problem2 : ((5 / 12) - (5 / 16)) * (4 / 5) + (2 / 3) - (3 / 4) = 0 := by
  sorry

end MathProofs

end problem1_problem2_l131_131246


namespace students_in_college_l131_131141

variable (P S : ℕ)

def condition1 : Prop := S = 15 * P
def condition2 : Prop := S + P = 40000

theorem students_in_college (h1 : condition1 S P) (h2 : condition2 S P) : S = 37500 := by
  sorry

end students_in_college_l131_131141


namespace gcd_exponent_min_speed_for_meeting_game_probability_difference_l131_131383

-- Problem p4
theorem gcd_exponent (a b : ℕ) (h1 : a = 6) (h2 : b = 9) (h3 : gcd a b = 3) : gcd (2^a - 1) (2^b - 1) = 7 := by
  sorry

-- Problem p5
theorem min_speed_for_meeting (v_S s : ℚ) (h : v_S = 1/2) : ∀ (s : ℚ), (s - v_S) ≥ 1 → s = 3/2 := by
  sorry

-- Problem p6
theorem game_probability_difference (N : ℕ) (p : ℚ) (h1 : N = 1) (h2 : p = 5/16) : N + p = 21/16 := by
  sorry

end gcd_exponent_min_speed_for_meeting_game_probability_difference_l131_131383


namespace operation_value_l131_131592

def operation1 (y : ℤ) : ℤ := 8 - y
def operation2 (y : ℤ) : ℤ := y - 8

theorem operation_value : operation2 (operation1 15) = -15 := by
  sorry

end operation_value_l131_131592


namespace rostov_survey_min_players_l131_131385

theorem rostov_survey_min_players :
  ∃ m : ℕ, (∀ n : ℕ, n < m → (95 + n * 1) % 100 ≠ 0) ∧ m = 11 :=
sorry

end rostov_survey_min_players_l131_131385


namespace center_circle_sum_l131_131476

theorem center_circle_sum (x y : ℝ) (h : x^2 + y^2 = 4 * x + 10 * y - 12) : x + y = 7 := 
sorry

end center_circle_sum_l131_131476


namespace optimal_play_winner_l131_131263

theorem optimal_play_winner (n : ℕ) (h : n > 1) : (n % 2 = 0) ↔ (first_player_wins: Bool) :=
  sorry

end optimal_play_winner_l131_131263


namespace find_a_l131_131563

theorem find_a (a : ℝ) (y : ℝ → ℝ) (y' : ℝ → ℝ) 
    (h_curve : ∀ x, y x = x^4 + a * x^2 + 1)
    (h_derivative : ∀ x, y' x = (4 * x^3 + 2 * a * x))
    (h_tangent_slope : y' (-1) = 8) :
    a = -6 :=
by
  -- To be proven
  sorry

end find_a_l131_131563


namespace approx_equal_e_l131_131890
noncomputable def a : ℝ := 69.28
noncomputable def b : ℝ := 0.004
noncomputable def c : ℝ := 0.03
noncomputable def d : ℝ := a * b
noncomputable def e : ℝ := d / c

theorem approx_equal_e : abs (e - 9.24) < 0.01 :=
by
  sorry

end approx_equal_e_l131_131890


namespace inspection_probability_l131_131852

noncomputable def defective_items : ℕ := 2
noncomputable def good_items : ℕ := 3
noncomputable def total_items : ℕ := defective_items + good_items

/-- Given 2 defective items and 3 good items mixed together,
the probability that the inspection stops exactly after
four inspections is 3/5 --/
theorem inspection_probability :
  (2 * (total_items - 1) * total_items / (total_items * (total_items - 1) * (total_items - 2) * (total_items - 3))) = (3 / 5) :=
by
  sorry

end inspection_probability_l131_131852


namespace twelve_women_reseated_l131_131131

def S (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 3
  else S (n - 1) + S (n - 2) + S (n - 3)

theorem twelve_women_reseated : S 12 = 1201 :=
by
  sorry

end twelve_women_reseated_l131_131131


namespace no_integer_roots_p_eq_2016_l131_131863

noncomputable def p (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots_p_eq_2016 
  (a b c d : ℤ)
  (h₁ : p a b c d 1 = 2015)
  (h₂ : p a b c d 2 = 2017) :
  ¬ ∃ x : ℤ, p a b c d x = 2016 :=
sorry

end no_integer_roots_p_eq_2016_l131_131863


namespace find_function_expression_l131_131541

theorem find_function_expression (f : ℝ → ℝ) : 
  (∀ x : ℝ, f (x - 1) = x^2 - 3 * x) → 
  (∀ x : ℝ, f x = x^2 - x - 2) :=
by
  sorry

end find_function_expression_l131_131541


namespace calculate_total_amount_l131_131531

theorem calculate_total_amount
  (price1 discount1 price2 discount2 additional_discount : ℝ)
  (h1 : price1 = 76) (h2 : discount1 = 25)
  (h3 : price2 = 85) (h4 : discount2 = 15)
  (h5 : additional_discount = 10) :
  price1 - discount1 + price2 - discount2 - additional_discount = 111 :=
by {
  sorry
}

end calculate_total_amount_l131_131531


namespace weekly_earnings_correct_l131_131381

-- Definitions based on the conditions
def hours_weekdays : Nat := 5 * 5
def hours_weekends : Nat := 3 * 2
def hourly_rate_weekday : Nat := 3
def hourly_rate_weekend : Nat := 3 * 2
def earnings_weekdays : Nat := hours_weekdays * hourly_rate_weekday
def earnings_weekends : Nat := hours_weekends * hourly_rate_weekend

-- The total weekly earnings Mitch gets
def weekly_earnings : Nat := earnings_weekdays + earnings_weekends

-- The theorem we need to prove:
theorem weekly_earnings_correct : weekly_earnings = 111 :=
by
  sorry

end weekly_earnings_correct_l131_131381


namespace yasna_finish_books_in_two_weeks_l131_131793

theorem yasna_finish_books_in_two_weeks (pages_book1 : ℕ) (pages_book2 : ℕ) (pages_per_day : ℕ) (days_per_week : ℕ) 
  (h1 : pages_book1 = 180) (h2 : pages_book2 = 100) (h3 : pages_per_day = 20) (h4 : days_per_week = 7) : 
  ((pages_book1 + pages_book2) / pages_per_day) / days_per_week = 2 := 
by
  sorry

end yasna_finish_books_in_two_weeks_l131_131793


namespace chord_length_l131_131619

theorem chord_length {r : ℝ} (h : r = 15) : 
  ∃ (CD : ℝ), CD = 26 * Real.sqrt 3 :=
by
  sorry

end chord_length_l131_131619


namespace evaluate_expression_l131_131646

theorem evaluate_expression :
  (3^2016 + 3^2014 + 3^2012) / (3^2016 - 3^2014 + 3^2012) = 91 / 73 := 
  sorry

end evaluate_expression_l131_131646


namespace max_value_f_on_interval_l131_131696

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_f_on_interval : 
  ∃ x ∈ Set.Icc (0 : ℝ) 1, ∀ y ∈ Set.Icc (0 : ℝ) 1, f y ≤ f x ∧ f x = Real.exp 1 - 1 := sorry

end max_value_f_on_interval_l131_131696


namespace evaluate_expression_l131_131990

def star (A B : ℚ) : ℚ := (A + B) / 3

theorem evaluate_expression : star (star 7 15) 10 = 52 / 9 := by
  sorry

end evaluate_expression_l131_131990


namespace sector_perimeter_l131_131352

theorem sector_perimeter (A θ r: ℝ) (hA : A = 2) (hθ : θ = 4) (hArea : A = (1/2) * r^2 * θ) : (2 * r + r * θ) = 6 :=
by 
  sorry

end sector_perimeter_l131_131352


namespace find_sides_of_triangle_l131_131945

theorem find_sides_of_triangle (c : ℝ) (θ : ℝ) (h_ratio : ℝ) 
  (h_c : c = 2 * Real.sqrt 7)
  (h_theta : θ = Real.pi / 6) -- 30 degrees in radians
  (h_ratio_eq : ∃ k : ℝ, ∀ a b : ℝ, a = k ∧ b = h_ratio * k) :
  ∃ (a b : ℝ), a = 2 ∧ b = 4 * Real.sqrt 3 := by
  sorry

end find_sides_of_triangle_l131_131945


namespace six_letter_vowel_words_count_l131_131233

noncomputable def vowel_count_six_letter_words : Nat := 27^6

theorem six_letter_vowel_words_count :
  vowel_count_six_letter_words = 531441 :=
  by
    sorry

end six_letter_vowel_words_count_l131_131233


namespace solve_for_y_l131_131883

theorem solve_for_y {y : ℝ} : (y - 5)^4 = 16 → y = 7 :=
by
  sorry

end solve_for_y_l131_131883


namespace max_total_length_of_cuts_l131_131850

theorem max_total_length_of_cuts (A : ℕ) (n : ℕ) (m : ℕ) (P : ℕ) (Q : ℕ)
  (h1 : A = 30 * 30)
  (h2 : n = 225)
  (h3 : m = A / n)
  (h4 : m = 4)
  (h5 : Q = 4 * 30)
  (h6 : P = 225 * 10 - Q)
  (h7 : P / 2 = 1065) :
  P / 2 = 1065 :=
by 
  exact h7

end max_total_length_of_cuts_l131_131850


namespace floor_plus_ceil_eq_seven_l131_131311

theorem floor_plus_ceil_eq_seven (x : ℝ) :
  (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
sorry

end floor_plus_ceil_eq_seven_l131_131311


namespace interval_x_2x_3x_l131_131229

theorem interval_x_2x_3x (x : ℝ) :
  (2 * x > 1) ∧ (2 * x < 2) ∧ (3 * x > 1) ∧ (3 * x < 2) ↔ (x > 1 / 2) ∧ (x < 2 / 3) :=
by
  sorry

end interval_x_2x_3x_l131_131229


namespace time_after_2004_hours_l131_131230

variable (h : ℕ) 

-- Current time is represented as an integer from 0 to 11 (9 o'clock).
def current_time : ℕ := 9

-- 12-hour clock cycles every 12 hours.
def cycle : ℕ := 12

-- Time after 2004 hours.
def hours_after : ℕ := 2004

-- Proof statement
theorem time_after_2004_hours (h : ℕ) :
  (current_time + hours_after) % cycle = current_time := 
sorry

end time_after_2004_hours_l131_131230


namespace red_peppers_weight_l131_131034

theorem red_peppers_weight (total_weight green_weight : ℝ) (h1 : total_weight = 5.666666667) (h2 : green_weight = 2.8333333333333335) : 
  total_weight - green_weight = 2.8333333336666665 :=
by
  sorry

end red_peppers_weight_l131_131034


namespace fraction_value_l131_131860

theorem fraction_value (x : ℝ) (h : 1 - 6 / x + 9 / (x^2) = 0) : 2 / x = 2 / 3 :=
  sorry

end fraction_value_l131_131860


namespace one_fifty_percent_of_eighty_l131_131031

theorem one_fifty_percent_of_eighty : (150 / 100) * 80 = 120 :=
  by sorry

end one_fifty_percent_of_eighty_l131_131031


namespace find_D_E_l131_131854

/--
Consider the circle given by \( x^2 + y^2 + D \cdot x + E \cdot y + F = 0 \) that is symmetrical with
respect to the line \( l_1: x - y + 4 = 0 \) and the line \( l_2: x + 3y = 0 \). Prove that the values 
of \( D \) and \( E \) are \( 12 \) and \( -4 \), respectively.
-/
theorem find_D_E (D E F : ℝ) (h1 : -D/2 + E/2 + 4 = 0) (h2 : -D/2 - 3*E/2 = 0) : D = 12 ∧ E = -4 :=
by
  sorry

end find_D_E_l131_131854


namespace kim_morning_routine_time_l131_131911

-- Definitions based on conditions
def minutes_coffee : ℕ := 5
def minutes_status_update_per_employee : ℕ := 2
def minutes_payroll_update_per_employee : ℕ := 3
def num_employees : ℕ := 9

-- Problem statement: Verifying the total morning routine time for Kim
theorem kim_morning_routine_time:
  minutes_coffee + (minutes_status_update_per_employee * num_employees) + 
  (minutes_payroll_update_per_employee * num_employees) = 50 :=
by
  -- Proof can follow here, but is currently skipped
  sorry

end kim_morning_routine_time_l131_131911


namespace water_bill_payment_ratio_l131_131894

variables (electricity_bill gas_bill water_bill internet_bill amount_remaining : ℤ)
variables (paid_gas_bill_payments paid_internet_bill_payments additional_gas_payment : ℤ)

-- Define the given conditions
def stephanie_budget := 
  electricity_bill = 60 ∧
  gas_bill = 40 ∧
  water_bill = 40 ∧
  internet_bill = 25 ∧
  amount_remaining = 30 ∧
  paid_gas_bill_payments = 3 ∧ -- three-quarters
  paid_internet_bill_payments = 4 ∧ -- four payments of $5
  additional_gas_payment = 5

-- Define the given problem as a theorem
theorem water_bill_payment_ratio 
  (h : stephanie_budget electricity_bill gas_bill water_bill internet_bill amount_remaining paid_gas_bill_payments paid_internet_bill_payments additional_gas_payment) :
  ∃ (paid_water_bill : ℤ), paid_water_bill / water_bill = 1 / 2 :=
sorry

end water_bill_payment_ratio_l131_131894


namespace equal_areas_of_ngons_l131_131878

noncomputable def area_of_ngon (n : ℕ) (sides : Fin n → ℝ) (radius : ℝ) (circumference : ℝ) : ℝ := sorry

theorem equal_areas_of_ngons 
  (n : ℕ) 
  (sides1 sides2 : Fin n → ℝ) 
  (radius : ℝ) 
  (circumference : ℝ)
  (h_sides : ∀ i : Fin n, ∃ j : Fin n, sides1 i = sides2 j)
  (h_inscribed1 : area_of_ngon n sides1 radius circumference = area_of_ngon n sides1 radius circumference)
  (h_inscribed2 : area_of_ngon n sides2 radius circumference = area_of_ngon n sides2 radius circumference) :
  area_of_ngon n sides1 radius circumference = area_of_ngon n sides2 radius circumference :=
sorry

end equal_areas_of_ngons_l131_131878


namespace find_other_number_l131_131240

theorem find_other_number (b : ℕ) (lcm_val gcd_val : ℕ)
  (h_lcm : Nat.lcm 240 b = 2520)
  (h_gcd : Nat.gcd 240 b = 24) :
  b = 252 :=
sorry

end find_other_number_l131_131240


namespace oil_leakage_calculation_l131_131397

def total_oil_leaked : ℕ := 11687
def oil_leaked_while_worked : ℕ := 5165
def oil_leaked_before_work : ℕ := 6522

theorem oil_leakage_calculation :
  oil_leaked_before_work = total_oil_leaked - oil_leaked_while_work :=
sorry

end oil_leakage_calculation_l131_131397


namespace coin_heads_probability_l131_131799

theorem coin_heads_probability
    (prob_tails : ℚ := 1/2)
    (prob_specific_sequence : ℚ := 0.0625)
    (flips : ℕ := 4)
    (ht : prob_tails = 1 / 2)
    (hs : prob_specific_sequence = (1 / 2) * (1 / 2) * (1 / 2) * (1 / 2)) 
    : ∀ (p_heads : ℚ), p_heads = 1 - prob_tails := by
  sorry

end coin_heads_probability_l131_131799


namespace geom_seq_thm_l131_131602

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
a 1 ≠ 0 ∧ ∀ n, a (n + 1) = (a n ^ 2) / (a (n - 1))

theorem geom_seq_thm (a : ℕ → ℝ) (h : geom_seq a) (h_neg : ∀ n, a n < 0) 
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) : a 3 + a 5 = -6 :=
by
  sorry

end geom_seq_thm_l131_131602


namespace S_n_min_at_5_min_nS_n_is_neg_49_l131_131373

variable {S_n : ℕ → ℝ}
variable {a_1 d : ℝ}

-- Conditions
axiom sum_first_n_terms (n : ℕ) : S_n n = n / 2 * (2 * a_1 + (n - 1) * d)

axiom S_10 : S_n 10 = 0
axiom S_15 : S_n 15 = 25

-- Proving the following statements
theorem S_n_min_at_5 :
  (∀ n, S_n n ≥ S_n 5) :=
sorry

theorem min_nS_n_is_neg_49 :
  (∀ n, n * S_n n ≥ -49) :=
sorry

end S_n_min_at_5_min_nS_n_is_neg_49_l131_131373


namespace sin_330_eq_neg_half_l131_131950

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end sin_330_eq_neg_half_l131_131950


namespace siblings_age_problem_l131_131371

variable {x y z : ℕ}

theorem siblings_age_problem
  (h1 : x - y = 3)
  (h2 : z - 1 = 2 * (x + y))
  (h3 : z + 20 = x + y + 40) :
  x = 11 ∧ y = 8 ∧ z = 39 :=
by
  sorry

end siblings_age_problem_l131_131371


namespace train_pass_time_correct_l131_131462

noncomputable def train_time_to_pass_post (length_of_train : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * (5 / 18)
  length_of_train / speed_mps

theorem train_pass_time_correct :
  train_time_to_pass_post 60 36 = 6 := by
  sorry

end train_pass_time_correct_l131_131462


namespace distinct_domino_paths_l131_131450

/-- Matt will arrange five identical, dotless dominoes (1 by 2 rectangles) 
on a 6 by 4 grid so that a path is formed from the upper left-hand corner 
(0, 0) to the lower right-hand corner (4, 5). Prove that the number of 
distinct arrangements is 126. -/
theorem distinct_domino_paths : 
  let m := 4
  let n := 5
  let total_moves := m + n
  let right_moves := m
  let down_moves := n
  (total_moves.choose right_moves) = 126 := by
{ 
  sorry 
}

end distinct_domino_paths_l131_131450


namespace combined_marble_remainder_l131_131797

theorem combined_marble_remainder (l j : ℕ) (h_l : l % 8 = 5) (h_j : j % 8 = 6) : (l + j) % 8 = 3 := by
  sorry

end combined_marble_remainder_l131_131797


namespace arcsin_cos_arcsin_arccos_sin_arccos_l131_131927

-- Define the statement
theorem arcsin_cos_arcsin_arccos_sin_arccos (x : ℝ) 
  (h1 : -1 ≤ x) 
  (h2 : x ≤ 1) :
  (Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = Real.pi / 2) := 
sorry

end arcsin_cos_arcsin_arccos_sin_arccos_l131_131927


namespace quadratic_roots_new_equation_l131_131261

theorem quadratic_roots_new_equation (a b c x1 x2 : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : x1 + x2 = -b / a) 
  (h3 : x1 * x2 = c / a) : 
  ∃ (a' b' c' : ℝ), a' * x^2 + b' * x + c' = 0 ∧ a' = a^2 ∧ b' = 3 * a * b ∧ c' = 2 * b^2 + a * c :=
sorry

end quadratic_roots_new_equation_l131_131261


namespace elle_practices_hours_l131_131006

variable (practice_time_weekday : ℕ) (days_weekday : ℕ) (multiplier_saturday : ℕ) (minutes_in_an_hour : ℕ) 
          (total_minutes_weekdays : ℕ) (total_minutes_saturday : ℕ) (total_minutes_week : ℕ) (total_hours : ℕ)

theorem elle_practices_hours :
  practice_time_weekday = 30 ∧
  days_weekday = 5 ∧
  multiplier_saturday = 3 ∧
  minutes_in_an_hour = 60 →
  total_minutes_weekdays = practice_time_weekday * days_weekday →
  total_minutes_saturday = practice_time_weekday * multiplier_saturday →
  total_minutes_week = total_minutes_weekdays + total_minutes_saturday →
  total_hours = total_minutes_week / minutes_in_an_hour →
  total_hours = 4 :=
by
  intros
  sorry

end elle_practices_hours_l131_131006


namespace nth_equation_identity_l131_131120

theorem nth_equation_identity (n : ℕ) (h : n ≥ 1) : 
  (n / (n + 2 : ℚ)) * (1 - 1 / (n + 1 : ℚ)) = (n^2 / ((n + 1) * (n + 2) : ℚ)) := 
by 
  sorry

end nth_equation_identity_l131_131120


namespace wednesday_more_than_tuesday_l131_131404

noncomputable def monday_minutes : ℕ := 450

noncomputable def tuesday_minutes : ℕ := monday_minutes / 2

noncomputable def wednesday_minutes : ℕ := 300

theorem wednesday_more_than_tuesday : wednesday_minutes - tuesday_minutes = 75 :=
by
  sorry

end wednesday_more_than_tuesday_l131_131404


namespace parallel_lines_k_value_l131_131393

theorem parallel_lines_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = 5 * x + 3 → y = (3 * k) * x + 1 → true) → k = 5 / 3 :=
by
  intros
  sorry

end parallel_lines_k_value_l131_131393


namespace calculation_correct_l131_131203

def f (x : ℚ) := (2 * x^2 + 6 * x + 9) / (x^2 + 3 * x + 5)
def g (x : ℚ) := 2 * x + 1

theorem calculation_correct : f (g 2) + g (f 2) = 308 / 45 := by
  sorry

end calculation_correct_l131_131203


namespace box_width_l131_131521

variable (l h vc : ℕ)
variable (nc : ℕ)
variable (v : ℕ)

-- Given
def length_box := 8
def height_box := 5
def volume_cube := 10
def num_cubes := 60
def volume_box := num_cubes * volume_cube

-- To Prove
theorem box_width : (volume_box = l * h * w) → w = 15 :=
by
  intro h1
  sorry

end box_width_l131_131521


namespace root_exists_between_0_and_1_l131_131362

theorem root_exists_between_0_and_1 (a b c : ℝ) (m : ℝ) (hm : 0 < m)
  (h : a / (m + 2) + b / (m + 1) + c / m = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x ^ 2 + b * x + c = 0 :=
by
  sorry

end root_exists_between_0_and_1_l131_131362


namespace unique_cube_coloring_l131_131631

-- Definition of vertices at the bottom of the cube with specific colors
inductive Color 
| Red | Green | Blue | Purple

open Color

def bottom_colors : Fin 4 → Color
| 0 => Red
| 1 => Green
| 2 => Blue
| 3 => Purple

-- Definition of the property that ensures each face of the cube has different colored corners
def all_faces_different_colors (top_colors : Fin 4 → Color) : Prop :=
  (top_colors 0 ≠ Red) ∧ (top_colors 0 ≠ Green) ∧ (top_colors 0 ≠ Blue) ∧
  (top_colors 1 ≠ Green) ∧ (top_colors 1 ≠ Blue) ∧ (top_colors 1 ≠ Purple) ∧
  (top_colors 2 ≠ Red) ∧ (top_colors 2 ≠ Blue) ∧ (top_colors 2 ≠ Purple) ∧
  (top_colors 3 ≠ Red) ∧ (top_colors 3 ≠ Green) ∧ (top_colors 3 ≠ Purple)

-- Prove there is exactly one way to achieve this coloring of the top corners
theorem unique_cube_coloring : ∃! (top_colors : Fin 4 → Color), all_faces_different_colors top_colors :=
sorry

end unique_cube_coloring_l131_131631


namespace total_coins_l131_131359

-- Definitions for the conditions
def number_of_nickels := 13
def number_of_quarters := 8

-- Statement of the proof problem
theorem total_coins : number_of_nickels + number_of_quarters = 21 :=
by
  sorry

end total_coins_l131_131359


namespace victoria_worked_weeks_l131_131510

-- Definitions for given conditions
def hours_worked_per_day : ℕ := 9
def total_hours_worked : ℕ := 315
def days_in_week : ℕ := 7

-- Main theorem to prove
theorem victoria_worked_weeks : total_hours_worked / hours_worked_per_day / days_in_week = 5 :=
by
  sorry

end victoria_worked_weeks_l131_131510


namespace value_of_nested_fraction_l131_131648

theorem value_of_nested_fraction :
  10 + 5 + (1 / 2) * (9 + 5 + (1 / 2) * (8 + 5 + (1 / 2) * (7 + 5 + (1 / 2) * (6 + 5 + (1 / 2) * (5 + 5 + (1 / 2) * (4 + 5 + (1 / 2) * (3 + 5 ))))))) = 28 + (1 / 128) :=
sorry

end value_of_nested_fraction_l131_131648


namespace x_equals_neg_x_is_zero_abs_x_equals_2_is_pm_2_l131_131124

theorem x_equals_neg_x_is_zero (x : ℝ) (h : x = -x) : x = 0 := sorry

theorem abs_x_equals_2_is_pm_2 (x : ℝ) (h : |x| = 2) : x = 2 ∨ x = -2 := sorry

end x_equals_neg_x_is_zero_abs_x_equals_2_is_pm_2_l131_131124


namespace tires_sale_price_l131_131611

variable (n : ℕ)
variable (t p_original p_sale : ℝ)

theorem tires_sale_price
  (h₁ : n = 4)
  (h₂ : t = 36)
  (h₃ : p_original = 84)
  (h₄ : p_sale = p_original - t / n) :
  p_sale = 75 := by
  sorry

end tires_sale_price_l131_131611


namespace compute_f_sum_l131_131255

noncomputable def f : ℝ → ℝ := sorry -- placeholder for f(x)

variables (x : ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom f_definition : ∀ x, 0 < x ∧ x < 1 → f x = x^2

-- Prove the main statement
theorem compute_f_sum : f (-3 / 2) + f 1 = 3 / 4 :=
by
  sorry

end compute_f_sum_l131_131255


namespace probability_fourth_roll_six_l131_131644

noncomputable def fair_die_prob : ℚ := 1 / 6
noncomputable def biased_die_prob : ℚ := 3 / 4
noncomputable def biased_die_other_face_prob : ℚ := 1 / 20
noncomputable def prior_prob : ℚ := 1 / 2

def p := 41
def q := 67

theorem probability_fourth_roll_six (p q : ℕ) (h1 : fair_die_prob = 1 / 6) (h2 : biased_die_prob = 3 / 4) (h3 : prior_prob = 1 / 2) :
  p + q = 108 :=
sorry

end probability_fourth_roll_six_l131_131644


namespace missing_digit_B_l131_131483

theorem missing_digit_B :
  ∃ B : ℕ, 0 ≤ B ∧ B ≤ 9 ∧ (200 + 10 * B + 5) % 13 = 0 := 
sorry

end missing_digit_B_l131_131483


namespace quadratic_general_form_l131_131601

theorem quadratic_general_form :
  ∀ x : ℝ, (x - 2) * (x + 3) = 1 → x^2 + x - 7 = 0 :=
by
  intros x h
  sorry

end quadratic_general_form_l131_131601


namespace contracting_schemes_l131_131168

theorem contracting_schemes :
  let total_projects := 6
  let a_contracts := 3
  let b_contracts := 2
  let c_contracts := 1
  (Nat.choose total_projects a_contracts) *
  (Nat.choose (total_projects - a_contracts) b_contracts) *
  (Nat.choose ((total_projects - a_contracts) - b_contracts) c_contracts) = 60 :=
by
  let total_projects := 6
  let a_contracts := 3
  let b_contracts := 2
  let c_contracts := 1
  sorry

end contracting_schemes_l131_131168


namespace grains_in_one_tsp_l131_131825

-- Definitions based on conditions
def grains_in_one_cup : Nat := 480
def half_cup_is_8_tbsp : Nat := 8
def one_tbsp_is_3_tsp : Nat := 3

-- Theorem statement
theorem grains_in_one_tsp :
  let grains_in_half_cup := grains_in_one_cup / 2
  let grains_in_one_tbsp := grains_in_half_cup / half_cup_is_8_tbsp
  grains_in_one_tbsp / one_tbsp_is_3_tsp = 10 :=
by
  sorry

end grains_in_one_tsp_l131_131825


namespace maximize_sum_of_sides_l131_131674

theorem maximize_sum_of_sides (a b c : ℝ) (A B C : ℝ) 
  (h_b : b = 2) (h_B : B = (Real.pi / 3)) (h_law_of_cosines : b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)) :
  a + c ≤ 4 :=
by
  sorry

end maximize_sum_of_sides_l131_131674


namespace solve_system_of_equations_l131_131732

theorem solve_system_of_equations (x y : ℝ) :
  (x^4 + (7/2) * x^2 * y + 2 * y^3 = 0) ∧
  (4 * x^2 + 7 * x * y + 2 * y^3 = 0) →
  (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = -1) ∨ (x = -11 / 2 ∧ y = -11 / 2) :=
sorry

end solve_system_of_equations_l131_131732


namespace ab_bc_ca_lt_quarter_l131_131630

theorem ab_bc_ca_lt_quarter (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 1) :
  (a * b)^(5/4) + (b * c)^(5/4) + (c * a)^(5/4) < 1/4 :=
sorry

end ab_bc_ca_lt_quarter_l131_131630


namespace translation_correctness_l131_131600

theorem translation_correctness :
  ( ∀ (x : ℝ), ((x + 4)^2 - 5) = ((x + 4)^2 - 5) ) :=
by
  sorry

end translation_correctness_l131_131600


namespace two_pow_start_digits_l131_131949

theorem two_pow_start_digits (A : ℕ) : 
  ∃ (m n : ℕ), 10^m * A < 2^n ∧ 2^n < 10^m * (A + 1) :=
  sorry

end two_pow_start_digits_l131_131949


namespace point_A_2019_pos_l131_131212

noncomputable def A : ℕ → ℤ
| 0       => 2
| (n + 1) =>
    if (n + 1) % 2 = 1 then A n - (n + 1)
    else A n + (n + 1)

theorem point_A_2019_pos : A 2019 = -1008 := by
  sorry

end point_A_2019_pos_l131_131212


namespace inequality_satisfied_for_a_l131_131300

theorem inequality_satisfied_for_a (a : ℝ) :
  (∀ x : ℝ, |2 * x - a| + |3 * x - 2 * a| ≥ a^2) ↔ -1/3 ≤ a ∧ a ≤ 1/3 :=
by
  sorry

end inequality_satisfied_for_a_l131_131300


namespace grace_hours_pulling_weeds_l131_131955

variable (Charge_mowing : ℕ) (Charge_weeding : ℕ) (Charge_mulching : ℕ)
variable (H_m : ℕ) (H_u : ℕ) (E_s : ℕ)

theorem grace_hours_pulling_weeds 
  (Charge_mowing_eq : Charge_mowing = 6)
  (Charge_weeding_eq : Charge_weeding = 11)
  (Charge_mulching_eq : Charge_mulching = 9)
  (H_m_eq : H_m = 63)
  (H_u_eq : H_u = 10)
  (E_s_eq : E_s = 567) :
  ∃ W : ℕ, 6 * 63 + 11 * W + 9 * 10 = 567 ∧ W = 9 := by
  sorry

end grace_hours_pulling_weeds_l131_131955


namespace determine_a_l131_131604

theorem determine_a (a b c : ℤ) (h_eq : ∀ x : ℤ, (x - a) * (x - 15) + 4 = (x + b) * (x + c)) :
  a = 16 ∨ a = 21 :=
  sorry

end determine_a_l131_131604


namespace angle_terminal_side_eq_l131_131193

theorem angle_terminal_side_eq (α : ℝ) : 
  (α = -4 * Real.pi / 3 + 2 * Real.pi) → (0 ≤ α ∧ α < 2 * Real.pi) → α = 2 * Real.pi / 3 := 
by 
  sorry

end angle_terminal_side_eq_l131_131193


namespace petya_result_less_than_one_tenth_l131_131734

theorem petya_result_less_than_one_tenth 
  (a b c d e f : ℕ) 
  (ha: a.gcd b = 1) (hb: c.gcd d = 1)
  (hc: e.gcd f = 1) 
  (vasya_correct: (a / b) + (c / d) + (e / f) = 1) :
  (a + c + e) / (b + d + f) < 1 / 10 :=
by
  -- proof goes here
  sorry

end petya_result_less_than_one_tenth_l131_131734


namespace maximum_profit_l131_131238

noncomputable def R (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then
  10.8 - (1/30) * x^2
else
  108 / x - 1000 / (3 * x^2)

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then
  x * R x - (10 + 2.7 * x)
else
  x * R x - (10 + 2.7 * x)

theorem maximum_profit : 
  ∃ x : ℝ, (0 < x ∧ x ≤ 10 → W x = 8.1 * x - (x^3 / 30) - 10) ∧ 
           (x > 10 → W x = 98 - 1000 / (3 * x) - 2.7 * x) ∧ 
           (∃ xmax : ℝ, xmax = 9 ∧ W 9 = 38.6) := 
sorry

end maximum_profit_l131_131238


namespace factory_dolls_per_day_l131_131347

-- Define the number of normal dolls made per day
def N : ℝ := 4800

-- Define the total number of dolls made per day as 1.33 times the number of normal dolls
def T : ℝ := 1.33 * N

-- The theorem statement to prove the factory makes 6384 dolls per day
theorem factory_dolls_per_day : T = 6384 :=
by
  -- Proof here
  sorry

end factory_dolls_per_day_l131_131347


namespace number_of_pencils_selling_price_equals_loss_l131_131118

theorem number_of_pencils_selling_price_equals_loss :
  ∀ (S C L : ℝ) (N : ℕ),
  C = 1.3333333333333333 * S →
  L = C - S →
  (S / 60) * N = L →
  N = 20 :=
by
  intros S C L N hC hL hN
  sorry

end number_of_pencils_selling_price_equals_loss_l131_131118


namespace find_speed_of_boat_l131_131867

noncomputable def speed_of_boat_in_still_water 
  (v : ℝ) 
  (current_speed : ℝ := 8) 
  (distance : ℝ := 36.67) 
  (time_in_minutes : ℝ := 44) : Prop :=
  v = 42

theorem find_speed_of_boat 
  (v : ℝ)
  (current_speed : ℝ := 8) 
  (distance : ℝ := 36.67) 
  (time_in_minutes : ℝ := 44) 
  (h1 : v + current_speed = distance / (time_in_minutes / 60)) : 
  speed_of_boat_in_still_water v :=
by
  sorry

end find_speed_of_boat_l131_131867


namespace relationship_abc_l131_131456

theorem relationship_abc (a b c : ℕ) (ha : a = 2^555) (hb : b = 3^444) (hc : c = 6^222) : a < c ∧ c < b := by
  sorry

end relationship_abc_l131_131456


namespace perpendicular_lines_values_of_a_l131_131809

theorem perpendicular_lines_values_of_a (a : ℝ) :
  (∃ (a : ℝ), (∀ x y : ℝ, a * x - y + 2 * a = 0 ∧ (2 * a - 1) * x + a * y = 0) 
    ↔ (a = 0 ∨ a = 1))
  := sorry

end perpendicular_lines_values_of_a_l131_131809


namespace polar_line_equation_l131_131897

theorem polar_line_equation
  (rho theta : ℝ)
  (h1 : rho = 4 * Real.cos theta)
  (h2 : ∀ (x y : ℝ), (x - 2) ^ 2 + y ^ 2 = 4 → x = 2)
  : rho * Real.cos theta = 2 :=
sorry

end polar_line_equation_l131_131897


namespace total_roses_planted_l131_131544

def roses_planted_two_days_ago := 50
def roses_planted_yesterday := roses_planted_two_days_ago + 20
def roses_planted_today := 2 * roses_planted_two_days_ago

theorem total_roses_planted :
  roses_planted_two_days_ago + roses_planted_yesterday + roses_planted_today = 220 := by
  sorry

end total_roses_planted_l131_131544


namespace james_browsers_l131_131944

def num_tabs_per_window := 10
def num_windows_per_browser := 3
def total_tabs := 60

theorem james_browsers : ∃ B : ℕ, (B * (num_windows_per_browser * num_tabs_per_window) = total_tabs) ∧ (B = 2) := sorry

end james_browsers_l131_131944


namespace meetings_percentage_l131_131415

def total_minutes_in_day (hours: ℕ): ℕ := hours * 60
def first_meeting_duration: ℕ := 60
def second_meeting_duration (first_meeting_duration: ℕ): ℕ := 3 * first_meeting_duration
def total_meeting_duration (first_meeting_duration: ℕ) (second_meeting_duration: ℕ): ℕ := first_meeting_duration + second_meeting_duration
def percentage_of_workday_spent_in_meetings (total_meeting_duration: ℕ) (total_minutes_in_day: ℕ): ℚ := (total_meeting_duration / total_minutes_in_day) * 100

theorem meetings_percentage (hours: ℕ) (first_meeting_duration: ℕ) (second_meeting_duration: ℕ) (total_meeting_duration: ℕ) (total_minutes_in_day: ℕ) 
(h1: total_minutes_in_day = 600) 
(h2: first_meeting_duration = 60) 
(h3: second_meeting_duration = 180) 
(h4: total_meeting_duration = 240):
percentage_of_workday_spent_in_meetings total_meeting_duration total_minutes_in_day = 40 := by
  sorry

end meetings_percentage_l131_131415


namespace total_cost_l131_131412

def copper_pipe_length := 10
def plastic_pipe_length := 15
def copper_pipe_cost_per_meter := 5
def plastic_pipe_cost_per_meter := 3

theorem total_cost (h₁ : copper_pipe_length = 10)
                   (h₂ : plastic_pipe_length = 15)
                   (h₃ : copper_pipe_cost_per_meter = 5)
                   (h₄ : plastic_pipe_cost_per_meter = 3) :
  10 * 5 + 15 * 3 = 95 :=
by sorry

end total_cost_l131_131412


namespace probability_Z_l131_131337

variable (p_X p_Y p_Z p_W : ℚ)

def conditions :=
  (p_X = 1/4) ∧ (p_Y = 1/3) ∧ (p_W = 1/6) ∧ (p_X + p_Y + p_Z + p_W = 1)

theorem probability_Z (h : conditions p_X p_Y p_Z p_W) : p_Z = 1/4 :=
by
  obtain ⟨hX, hY, hW, hSum⟩ := h
  sorry

end probability_Z_l131_131337


namespace min_value_ineq_l131_131993

theorem min_value_ineq (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 3 * y = 1) :
  (1 / x) + (1 / (3 * y)) ≥ 4 :=
  sorry

end min_value_ineq_l131_131993


namespace interior_angle_solution_l131_131884

noncomputable def interior_angle_of_inscribed_triangle (x : ℝ) (h : (2 * x + 40) + (x + 80) + (3 * x - 50) = 360) : ℝ :=
  (1 / 2) * (x + 80)

theorem interior_angle_solution (x : ℝ) (h : (2 * x + 40) + (x + 80) + (3 * x - 50) = 360) :
  interior_angle_of_inscribed_triangle x h = 64 :=
sorry

end interior_angle_solution_l131_131884


namespace number_of_cows_consume_in_96_days_l131_131573

-- Given conditions
def grass_growth_rate := 10 / 3
def consumption_by_70_cows_in_24_days := 70 * 24
def consumption_by_30_cows_in_60_days := 30 * 60
def total_grass_in_96_days := consumption_by_30_cows_in_60_days + 120

-- Problem statement
theorem number_of_cows_consume_in_96_days : 
  (x : ℕ) -> 96 * x = total_grass_in_96_days -> x = 20 :=
by
  intros x h
  sorry

end number_of_cows_consume_in_96_days_l131_131573


namespace A_can_complete_work_in_28_days_l131_131078
noncomputable def work_days_for_A (x : ℕ) (h : 4 / x = 1 / 21) : ℕ :=
  x / 3

theorem A_can_complete_work_in_28_days (x : ℕ) (h : 4 / x = 1 / 21) :
  work_days_for_A x h = 28 :=
  sorry

end A_can_complete_work_in_28_days_l131_131078


namespace combined_towel_weight_l131_131081

/-
Given:
1. Mary has 5 times as many towels as Frances.
2. Mary has 3 times as many towels as John.
3. The total weight of their towels is 145 pounds.
4. Mary has 60 towels.

To prove: 
The combined weight of Frances's and John's towels is 22.863 kilograms.
-/

theorem combined_towel_weight (total_weight_pounds : ℝ) (mary_towels frances_towels john_towels : ℕ) 
  (conversion_factor : ℝ) (combined_weight_kilograms : ℝ) :
  mary_towels = 60 →
  mary_towels = 5 * frances_towels →
  mary_towels = 3 * john_towels →
  total_weight_pounds = 145 →
  conversion_factor = 0.453592 →
  combined_weight_kilograms = 22.863 :=
by
  sorry

end combined_towel_weight_l131_131081


namespace simplify_expression_l131_131752

theorem simplify_expression (x : ℝ) (hx2 : x ≠ 2) (hx_2 : x ≠ -2) (hx1 : x ≠ 1) : 
  (1 + 1 / (x - 2)) / ((x^2 - 2 * x + 1) / (x^2 - 4)) = (x + 2) / (x - 1) :=
by
  sorry

end simplify_expression_l131_131752


namespace last_five_digits_of_sequence_l131_131585

theorem last_five_digits_of_sequence (seq : Fin 36 → Fin 2) 
  (h0 : seq 0 = 0) (h1 : seq 1 = 0) (h2 : seq 2 = 0) (h3 : seq 3 = 0) (h4 : seq 4 = 0)
  (unique_combos : ∀ (combo: Fin 32 → Fin 2), 
    ∃ (start_index : Fin 32), ∀ (i : Fin 5),
      combo i = seq ((start_index + i) % 36)) :
  seq 31 = 1 ∧ seq 32 = 1 ∧ seq 33 = 1 ∧ seq 34 = 0 ∧ seq 35 = 1 :=
by
  sorry

end last_five_digits_of_sequence_l131_131585


namespace tan_subtraction_modified_l131_131768

theorem tan_subtraction_modified (α β : ℝ) (h1 : Real.tan α = 9) (h2 : Real.tan β = 6) :
  Real.tan (α - β) = (3 : ℝ) / (157465 : ℝ) := by
  have h3 : Real.tan (α - β) = (Real.tan α - Real.tan β) / (1 + (Real.tan α * Real.tan β)^3) :=
    sorry -- this is assumed as given in the conditions
  sorry -- rest of the proof

end tan_subtraction_modified_l131_131768


namespace product_of_roots_l131_131106

theorem product_of_roots (x1 x2 : ℝ) (h1 : x1 ^ 2 - 2 * x1 = 2) (h2 : x2 ^ 2 - 2 * x2 = 2) (hne : x1 ≠ x2) :
  x1 * x2 = -2 := 
sorry

end product_of_roots_l131_131106


namespace outstanding_student_awards_l131_131465

theorem outstanding_student_awards :
  ∃ n : ℕ, 
  (n = Nat.choose 9 7) ∧ 
  (∀ (awards : ℕ) (classes : ℕ), awards = 10 → classes = 8 → n = 36) := 
by
  sorry

end outstanding_student_awards_l131_131465


namespace arithmetic_mean_median_l131_131101

theorem arithmetic_mean_median (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : a = 0) (h4 : (a + b + c) / 3 = 4 * b) : c / b = 11 :=
by
  sorry

end arithmetic_mean_median_l131_131101


namespace fourth_rectangle_area_l131_131391

-- Define the conditions and prove the area of the fourth rectangle
theorem fourth_rectangle_area (x y z w : ℝ)
  (h_xy : x * y = 24)
  (h_xw : x * w = 35)
  (h_zw : z * w = 42)
  (h_sum : x + z = 21) :
  y * w = 33.777 := 
sorry

end fourth_rectangle_area_l131_131391


namespace usual_time_is_24_l131_131249

variable (R T : ℝ)
variable (usual_rate fraction_of_rate early_min : ℝ)
variable (h1 : fraction_of_rate = 6 / 7)
variable (h2 : early_min = 4)
variable (h3 : (R / (fraction_of_rate * R)) = 7 / 6)
variable (h4 : ((T - early_min) / T) = fraction_of_rate)

theorem usual_time_is_24 {R T : ℝ} (fraction_of_rate := 6/7) (early_min := 4) :
  fraction_of_rate = 6 / 7 ∧ early_min = 4 → 
  (T - early_min) / T = fraction_of_rate → 
  T = 24 :=
by
  intros hfraction_hearly htime_eq_fraction
  sorry

end usual_time_is_24_l131_131249


namespace mr_roper_lawn_cuts_l131_131071

theorem mr_roper_lawn_cuts (x : ℕ) (h_apr_sep : ℕ → ℕ) (h_total_cuts : 12 * 9 = 108) :
  (6 * x + 6 * 3 = 108) → x = 15 :=
by
  -- The proof is not needed as per the instructions, hence we use sorry.
  sorry

end mr_roper_lawn_cuts_l131_131071


namespace range_of_n_l131_131712

def hyperbola_equation (m n : ℝ) : Prop :=
  (m^2 + n) * (3 * m^2 - n) > 0

def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

theorem range_of_n (m n : ℝ) :
  hyperbola_equation m n ∧ foci_distance m n →
  -1 < n ∧ n < 3 :=
by
  intro h
  have hyperbola_condition := h.1
  have distance_condition := h.2
  sorry

end range_of_n_l131_131712


namespace abs_m_minus_n_eq_five_l131_131788

theorem abs_m_minus_n_eq_five (m n : ℝ) (h₁ : m * n = 6) (h₂ : m + n = 7) : |m - n| = 5 :=
sorry

end abs_m_minus_n_eq_five_l131_131788


namespace annual_income_is_correct_l131_131727

noncomputable def total_investment : ℝ := 4455
noncomputable def price_per_share : ℝ := 8.25
noncomputable def dividend_rate : ℝ := 12 / 100
noncomputable def face_value : ℝ := 10

noncomputable def number_of_shares : ℝ := total_investment / price_per_share
noncomputable def dividend_per_share : ℝ := dividend_rate * face_value
noncomputable def annual_income : ℝ := dividend_per_share * number_of_shares

theorem annual_income_is_correct : annual_income = 648 := by
  sorry

end annual_income_is_correct_l131_131727


namespace integer_roots_count_l131_131298

theorem integer_roots_count (b c d e f : ℚ) :
  ∃ (n : ℕ), (n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 5) ∧
  (∃ (r : ℕ → ℤ), ∀ i, i < n → (∀ z : ℤ, (∃ m, z = r m) → (z^5 + b * z^4 + c * z^3 + d * z^2 + e * z + f = 0))) :=
sorry

end integer_roots_count_l131_131298


namespace false_proposition_p_and_q_l131_131607

open Classical

-- Define the propositions
def p (a b c : ℝ) : Prop := b * b = a * c
def q (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- We provide the conditions specified in the problem
variable (a b c : ℝ)
variable (f : ℝ → ℝ)
axiom hq : ∀ x, f x = f (-x)
axiom hp : ¬ (∀ a b c, p a b c ↔ (b ≠ 0 ∧ c ≠ 0 ∧ b * b = a * c))

-- The false proposition among the given options is "p and q"
theorem false_proposition_p_and_q : ¬ (∀ a b c (f : ℝ → ℝ), p a b c ∧ q f) :=
by
  -- This is where the proof would go, but is marked as a placeholder
  sorry

end false_proposition_p_and_q_l131_131607


namespace range_of_k_for_real_roots_l131_131939

theorem range_of_k_for_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - x + 3 = 0) ↔ (k <= 1 / 12 ∧ k ≠ 0) :=
sorry

end range_of_k_for_real_roots_l131_131939


namespace jenny_chocolate_squares_l131_131742

theorem jenny_chocolate_squares (mike_chocolates : ℕ) (jenny_chocolates : ℕ) 
  (h_mike : mike_chocolates = 20) 
  (h_jenny : jenny_chocolates = 3 * mike_chocolates + 5) :
  jenny_chocolates = 65 :=
by
  sorry

end jenny_chocolate_squares_l131_131742


namespace distinct_positive_integers_exists_l131_131008

theorem distinct_positive_integers_exists 
(n : ℕ)
(a b : ℕ)
(h1 : a ≠ b)
(h2 : b % a = 0)
(h3 : a > 10^(2 * n - 1) ∧ a < 10^(2 * n))
(h4 : b > 10^(2 * n - 1) ∧ b < 10^(2 * n))
(h5 : ∀ x y : ℕ, a = 10^n * x + y ∧ b = 10^n * y + x ∧ x < y ∧ x / 10^(n - 1) ≠ 0 ∧ y / 10^(n - 1) ≠ 0) :
a = (10^(2 * n) - 1) / 7 ∧ b = 6 * (10^(2 * n) - 1) / 7 := 
by
  sorry

end distinct_positive_integers_exists_l131_131008


namespace prove_inequalities_l131_131401

noncomputable def x := Real.log Real.pi
noncomputable def y := Real.logb 5 2
noncomputable def z := Real.exp (-1 / 2)

theorem prove_inequalities : y < z ∧ z < x := by
  unfold x y z
  sorry

end prove_inequalities_l131_131401


namespace gain_percentage_second_book_l131_131014

theorem gain_percentage_second_book (C1 C2 SP1 SP2 : ℝ) (H1 : C1 + C2 = 360) (H2 : C1 = 210) (H3 : SP1 = C1 - (15 / 100) * C1) (H4 : SP1 = SP2) (H5 : SP2 = C2 + (19 / 100) * C2) : 
  (19 : ℝ) = 19 := 
by
  sorry

end gain_percentage_second_book_l131_131014


namespace polynomial_perfect_square_trinomial_l131_131834

theorem polynomial_perfect_square_trinomial (k : ℝ) :
  (∀ x : ℝ, 4 * x^2 + 2 * k * x + 25 = (2 * x + 5) * (2 * x + 5)) → (k = 10 ∨ k = -10) :=
by
  sorry

end polynomial_perfect_square_trinomial_l131_131834


namespace quadratic_conversion_l131_131163

def quadratic_to_vertex_form (x : ℝ) : ℝ := 2 * x^2 - 8 * x - 1

theorem quadratic_conversion :
  (∀ x : ℝ, quadratic_to_vertex_form x = 2 * (x - 2)^2 - 9) :=
by
  sorry

end quadratic_conversion_l131_131163


namespace rectangle_area_is_12_l131_131302

noncomputable def rectangle_area_proof (w l y : ℝ) : Prop :=
  l = 3 * w ∧ 2 * (l + w) = 16 ∧ (l^2 + w^2 = y^2) → l * w = 12

theorem rectangle_area_is_12 (y : ℝ) : ∃ (w l : ℝ), rectangle_area_proof w l y :=
by
  -- Introducing variables
  exists 2
  exists 6
  -- Constructing proof steps (skipped here with sorry)
  sorry

end rectangle_area_is_12_l131_131302


namespace passing_percentage_correct_l131_131904

-- Define the conditions
def max_marks : ℕ := 500
def candidate_marks : ℕ := 180
def fail_by : ℕ := 45

-- Define the passing_marks based on given conditions
def passing_marks : ℕ := candidate_marks + fail_by

-- Theorem to prove: the passing percentage is 45%
theorem passing_percentage_correct : 
  (passing_marks / max_marks) * 100 = 45 := 
sorry

end passing_percentage_correct_l131_131904


namespace one_neither_prime_nor_composite_l131_131869

/-- Definition of a prime number in the natural numbers -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Definition of a composite number in the natural numbers -/
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n 

/-- Theorem stating that the number 1 is neither prime nor composite -/
theorem one_neither_prime_nor_composite : ¬is_prime 1 ∧ ¬is_composite 1 :=
sorry

end one_neither_prime_nor_composite_l131_131869


namespace number_of_sheep_l131_131495

variable (S H C : ℕ)

def ratio_constraint : Prop := 4 * H = 7 * S ∧ 5 * S = 4 * C

def horse_food_per_day (H : ℕ) : ℕ := 230 * H
def sheep_food_per_day (S : ℕ) : ℕ := 150 * S
def cow_food_per_day (C : ℕ) : ℕ := 300 * C

def total_horse_food : Prop := horse_food_per_day H = 12880
def total_sheep_food : Prop := sheep_food_per_day S = 9750
def total_cow_food : Prop := cow_food_per_day C = 15000

theorem number_of_sheep (h1 : ratio_constraint S H C)
                        (h2 : total_horse_food H)
                        (h3 : total_sheep_food S)
                        (h4 : total_cow_food C) :
  S = 98 :=
sorry

end number_of_sheep_l131_131495


namespace percent_games_lost_l131_131844

theorem percent_games_lost
  (w l t : ℕ)
  (h_ratio : 7 * l = 3 * w)
  (h_tied : t = 5) :
  (l : ℝ) / (w + l + t) * 100 = 20 :=
by
  sorry

end percent_games_lost_l131_131844


namespace one_eighth_of_2_pow_44_eq_2_pow_x_l131_131017

theorem one_eighth_of_2_pow_44_eq_2_pow_x (x : ℕ) :
  (2^44 / 8 = 2^x) → x = 41 :=
by
  sorry

end one_eighth_of_2_pow_44_eq_2_pow_x_l131_131017


namespace part1_part2_l131_131220

open Set Real

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem part1 (m : ℝ) (h : Disjoint A (B m)) : m ∈ Iio 2 ∪ Ioi 4 := 
sorry

theorem part2 (m : ℝ) (h : A ∪ (univ \ (B m)) = univ) : m ∈ Iic 3 := 
sorry

end part1_part2_l131_131220


namespace problem_l131_131543

def f (n : ℕ) : ℤ := 3 ^ (2 * n) - 32 * n ^ 2 + 24 * n - 1

theorem problem (n : ℕ) (h : 0 < n) : 512 ∣ f n := sorry

end problem_l131_131543


namespace purely_imaginary_iff_x_equals_one_l131_131336

theorem purely_imaginary_iff_x_equals_one (x : ℝ) :
  ((x^2 - 1) + (x + 1) * Complex.I).re = 0 → x = 1 :=
by
  sorry

end purely_imaginary_iff_x_equals_one_l131_131336


namespace correct_expression_l131_131502

variables {a b c : ℝ}

theorem correct_expression :
  -2 * (3 * a - b) + 3 * (2 * a + b) = 5 * b :=
by
  sorry

end correct_expression_l131_131502


namespace unique_positive_real_b_l131_131426

noncomputable def is_am_gm_satisfied (r s t : ℝ) (a : ℝ) : Prop :=
  r + s + t = 2 * a ∧ r * s * t = 2 * a ∧ (r+s+t)/3 = ((r * s * t) ^ (1/3))

noncomputable def poly_roots_real (r s t : ℝ) : Prop :=
  ∀ x : ℝ, (x = r ∨ x = s ∨ x = t)

theorem unique_positive_real_b :
  ∃ b a : ℝ, 0 < a ∧ 0 < b ∧
  (∃ r s t : ℝ, (r ≥ 0 ∧ s ≥ 0 ∧ t ≥ 0 ∧ poly_roots_real r s t) ∧
   is_am_gm_satisfied r s t a ∧
   (x^3 - 2*a*x^2 + b*x - 2*a = (x - r) * (x - s) * (x - t)) ∧
   b = 9) := sorry

end unique_positive_real_b_l131_131426


namespace parallel_lines_slope_l131_131780

theorem parallel_lines_slope (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 = (3 * k) * x + 7) → k = 5 / 3 := 
sorry

end parallel_lines_slope_l131_131780


namespace ratio_is_two_l131_131090

noncomputable def ratio_of_altitude_to_base (area base : ℕ) : ℕ :=
  have h : ℕ := area / base
  h / base

theorem ratio_is_two (area base : ℕ) (h : ℕ)  (h_area : area = 288) (h_base : base = 12) (h_altitude : h = area / base) : ratio_of_altitude_to_base area base = 2 :=
  by
    sorry 

end ratio_is_two_l131_131090


namespace axis_of_symmetry_circle_l131_131908

theorem axis_of_symmetry_circle (a : ℝ) : 
  (2 * a + 0 - 1 = 0) ↔ (a = 1 / 2) :=
by
  sorry

end axis_of_symmetry_circle_l131_131908


namespace hyperbola_correct_l131_131737

noncomputable def hyperbola_properties : Prop :=
  let h := 2
  let k := 0
  let a := 4
  let c := 8
  let b := Real.sqrt ((c^2) - (a^2))
  (h + k + a + b = 4 * Real.sqrt 3 + 6)

theorem hyperbola_correct : hyperbola_properties :=
by
  unfold hyperbola_properties
  let h := 2
  let k := 0
  let a := 4
  let c := 8
  have b : ℝ := Real.sqrt ((c^2) - (a^2))
  sorry

end hyperbola_correct_l131_131737


namespace train_length_is_120_l131_131670

-- Definitions based on conditions
def bridge_length : ℕ := 600
def total_time : ℕ := 30
def on_bridge_time : ℕ := 20

-- Proof statement
theorem train_length_is_120 (x : ℕ) (speed1 speed2 : ℕ) :
  (speed1 = (bridge_length + x) / total_time) ∧
  (speed2 = bridge_length / on_bridge_time) ∧
  (speed1 = speed2) →
  x = 120 :=
by
  sorry

end train_length_is_120_l131_131670


namespace black_car_overtakes_red_car_in_one_hour_l131_131448

-- Define the speeds of the cars
def red_car_speed := 30 -- in miles per hour
def black_car_speed := 50 -- in miles per hour

-- Define the initial distance between the cars
def initial_distance := 20 -- in miles

-- Calculate the time required for the black car to overtake the red car
theorem black_car_overtakes_red_car_in_one_hour : initial_distance / (black_car_speed - red_car_speed) = 1 := by
  sorry

end black_car_overtakes_red_car_in_one_hour_l131_131448


namespace at_least_one_half_l131_131039

theorem at_least_one_half (x y z : ℝ) (h : x + y + z - 2 * (x * y + y * z + x * z) + 4 * x * y * z = 1 / 2) :
  x = 1 / 2 ∨ y = 1 / 2 ∨ z = 1 / 2 :=
by
  sorry

end at_least_one_half_l131_131039


namespace avg_visitors_on_sunday_l131_131369

theorem avg_visitors_on_sunday (S : ℕ) :
  (30 * 285) = (5 * S + 25 * 240) -> S = 510 :=
by
  intros h
  sorry

end avg_visitors_on_sunday_l131_131369


namespace sin_300_eq_neg_sqrt3_div_2_l131_131974

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l131_131974


namespace number_of_storks_joined_l131_131318

theorem number_of_storks_joined (initial_birds : ℕ) (initial_storks : ℕ) (total_birds_and_storks : ℕ) 
    (h1 : initial_birds = 3) (h2 : initial_storks = 4) (h3 : total_birds_and_storks = 13) : 
    (total_birds_and_storks - (initial_birds + initial_storks)) = 6 := 
by
  sorry

end number_of_storks_joined_l131_131318


namespace minimum_value_of_f_l131_131121

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x - 15| + |x - a - 15|

theorem minimum_value_of_f {a : ℝ} (h0 : 0 < a) (h1 : a < 15) : ∃ Q, (∀ x, a ≤ x ∧ x ≤ 15 → f x a ≥ Q) ∧ Q = 15 := by
  sorry

end minimum_value_of_f_l131_131121


namespace arithmetic_mean_of_14_22_36_l131_131153

theorem arithmetic_mean_of_14_22_36 : (14 + 22 + 36) / 3 = 24 := by
  sorry

end arithmetic_mean_of_14_22_36_l131_131153


namespace number_of_possible_n_l131_131219

theorem number_of_possible_n :
  ∃ (a : ℕ), (∀ n, (n = a^3) ∧ 
  ((∃ b c : ℕ, b ≠ c ∧ b ≠ a ∧ c ≠ a ∧ a = b * c)) ∧ 
  (a + b + c = 2010) ∧ 
  (a > 0) ∧
  (b > 0) ∧
  (c > 0)) → 
  ∃ (num_n : ℕ), num_n = 2009 :=
  sorry

end number_of_possible_n_l131_131219


namespace rachel_pool_fill_time_l131_131725

theorem rachel_pool_fill_time :
  ∀ (pool_volume : ℕ) (num_hoses : ℕ) (hose_rate : ℕ),
  pool_volume = 30000 →
  num_hoses = 5 →
  hose_rate = 3 →
  (pool_volume / (num_hoses * hose_rate * 60) : ℤ) = 33 :=
by
  intros pool_volume num_hoses hose_rate h1 h2 h3
  sorry

end rachel_pool_fill_time_l131_131725


namespace convert_base_7_to_base_10_l131_131487

theorem convert_base_7_to_base_10 : 
  ∀ n : ℕ, (n = 3 * 7^2 + 2 * 7^1 + 1 * 7^0) → n = 162 :=
by
  intros n h
  rw [pow_zero, pow_one, pow_two] at h
  norm_num at h
  exact h

end convert_base_7_to_base_10_l131_131487


namespace circle_center_radius_l131_131155

theorem circle_center_radius (x y : ℝ) :
  (x^2 + y^2 + 4 * x - 6 * y = 11) →
  ∃ (h k r : ℝ), h = -2 ∧ k = 3 ∧ r = 2 * Real.sqrt 6 ∧
  (x+h)^2 + (y+k)^2 = r^2 :=
by
  sorry

end circle_center_radius_l131_131155


namespace number_of_standing_demons_l131_131702

variable (N : ℕ)
variable (initial_knocked_down : ℕ)
variable (initial_standing : ℕ)
variable (current_knocked_down : ℕ)
variable (current_standing : ℕ)

axiom initial_condition : initial_knocked_down = (3 * initial_standing) / 2
axiom condition_after_changes : current_knocked_down = initial_knocked_down + 2
axiom condition_after_changes_2 : current_standing = initial_standing - 10
axiom final_condition : current_standing = (5 * current_knocked_down) / 4

theorem number_of_standing_demons : current_standing = 35 :=
sorry

end number_of_standing_demons_l131_131702


namespace suki_bag_weight_is_22_l131_131800

noncomputable def weight_of_suki_bag : ℝ :=
  let bags_suki := 6.5
  let bags_jimmy := 4.5
  let weight_jimmy_per_bag := 18.0
  let total_containers := 28
  let weight_per_container := 8.0
  let total_weight_jimmy := bags_jimmy * weight_jimmy_per_bag
  let total_weight_combined := total_containers * weight_per_container
  let total_weight_suki := total_weight_combined - total_weight_jimmy
  total_weight_suki / bags_suki

theorem suki_bag_weight_is_22 : weight_of_suki_bag = 22 :=
by
  sorry

end suki_bag_weight_is_22_l131_131800


namespace trader_sold_45_meters_l131_131862

-- Definitions based on conditions
def selling_price_total : ℕ := 4500
def profit_per_meter : ℕ := 12
def cost_price_per_meter : ℕ := 88
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- The proof goal to show that the trader sold 45 meters of cloth
theorem trader_sold_45_meters : ∃ x : ℕ, selling_price_per_meter * x = selling_price_total ∧ x = 45 := 
by
  sorry

end trader_sold_45_meters_l131_131862


namespace player1_coins_l131_131806

theorem player1_coins (coin_distribution : Fin 9 → ℕ) :
  let rotations := 11
  let player_4_coins := 90
  let player_8_coins := 35
  ∀ player : Fin 9, player = 0 → 
    let player_1_coins := coin_distribution player
    (coin_distribution 3 = player_4_coins) →
    (coin_distribution 7 = player_8_coins) →
    player_1_coins = 57 := 
sorry

end player1_coins_l131_131806


namespace union_of_A_and_B_is_R_l131_131848

open Set Real

def A := {x : ℝ | log x > 0}
def B := {x : ℝ | x ≤ 1}

theorem union_of_A_and_B_is_R : A ∪ B = univ := by
  sorry

end union_of_A_and_B_is_R_l131_131848


namespace remainder_of_m_div_5_l131_131375

theorem remainder_of_m_div_5 (m n : ℕ) (hpos : 0 < m) (hdef : m = 15 * n - 1) : m % 5 = 4 :=
sorry

end remainder_of_m_div_5_l131_131375


namespace percent_increase_from_first_to_second_quarter_l131_131763

theorem percent_increase_from_first_to_second_quarter 
  (P : ℝ) :
  ((1.60 * P - 1.20 * P) / (1.20 * P)) * 100 = 33.33 := by
  sorry

end percent_increase_from_first_to_second_quarter_l131_131763


namespace martian_calendar_months_l131_131479

theorem martian_calendar_months (x y : ℕ) 
  (h1 : 100 * x + 77 * y = 5882) : x + y = 74 :=
sorry

end martian_calendar_months_l131_131479


namespace total_integers_at_least_eleven_l131_131582

theorem total_integers_at_least_eleven (n neg_count : ℕ) 
  (h1 : neg_count % 2 = 1)
  (h2 : neg_count ≤ 11) :
  n ≥ 11 := 
sorry

end total_integers_at_least_eleven_l131_131582


namespace scaling_factor_is_2_l131_131242

-- Define the volumes of the original and scaled cubes
def V1 : ℕ := 343
def V2 : ℕ := 2744

-- Assume s1 cubed equals V1 and s2 cubed equals V2
def s1 : ℕ := 7  -- because 7^3 = 343
def s2 : ℕ := 14 -- because 14^3 = 2744

-- Scaling factor between the cubes
def scaling_factor : ℕ := s2 / s1 

-- The theorem stating the scaling factor is 2 given the volumes
theorem scaling_factor_is_2 (h1 : s1 ^ 3 = V1) (h2 : s2 ^ 3 = V2) : scaling_factor = 2 := by
  sorry

end scaling_factor_is_2_l131_131242


namespace min_generic_tees_per_package_l131_131651

def total_golf_tees_needed (n : ℕ) : ℕ := 80
def max_generic_packages_used : ℕ := 2
def tees_per_aero_flight_package : ℕ := 2
def aero_flight_packages_needed : ℕ := 28
def total_tees_from_aero_flight_packages (n : ℕ) : ℕ := aero_flight_packages_needed * tees_per_aero_flight_package

theorem min_generic_tees_per_package (G : ℕ) :
  (total_golf_tees_needed 4) - (total_tees_from_aero_flight_packages aero_flight_packages_needed) ≤ max_generic_packages_used * G → G ≥ 12 :=
by
  sorry

end min_generic_tees_per_package_l131_131651


namespace largest_fraction_is_36_l131_131718

theorem largest_fraction_is_36 : 
  let A := (1 : ℚ) / 5
  let B := (2 : ℚ) / 10
  let C := (7 : ℚ) / 15
  let D := (9 : ℚ) / 20
  let E := (3 : ℚ) / 6
  A < E ∧ B < E ∧ C < E ∧ D < E :=
by
  let A := (1 : ℚ) / 5
  let B := (2 : ℚ) / 10
  let C := (7 : ℚ) / 15
  let D := (9 : ℚ) / 20
  let E := (3 : ℚ) / 6
  sorry

end largest_fraction_is_36_l131_131718


namespace circle_radius_tangent_to_circumcircles_l131_131196

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  (a * b * c) / (4 * (Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))))

theorem circle_radius_tangent_to_circumcircles (AB BC CA : ℝ) (H : Point) 
  (h_AB : AB = 13) (h_BC : BC = 14) (h_CA : CA = 15) : 
  (radius : ℝ) = 65 / 16 :=
by
  sorry

end circle_radius_tangent_to_circumcircles_l131_131196


namespace compare_charges_l131_131166

/-
Travel agencies A and B have group discount methods with the original price being $200 per person.
- Agency A: Buy 4 full-price tickets, the rest are half price.
- Agency B: All customers get a 30% discount.
Prove the given relationships based on the number of travelers.
-/

def agency_a_cost (x : ℕ) : ℕ :=
  if 0 < x ∧ x < 4 then 200 * x
  else if x ≥ 4 then 100 * x + 400
  else 0

def agency_b_cost (x : ℕ) : ℕ :=
  140 * x

theorem compare_charges (x : ℕ) :
  (agency_a_cost x < agency_b_cost x -> x > 10) ∧
  (agency_a_cost x = agency_b_cost x -> x = 10) ∧
  (agency_a_cost x > agency_b_cost x -> x < 10) :=
by
  sorry

end compare_charges_l131_131166


namespace pavan_travel_time_l131_131984

theorem pavan_travel_time (D : ℝ) (V1 V2 : ℝ) (distance : D = 300) (speed1 : V1 = 30) (speed2 : V2 = 25) : 
  ∃ t : ℝ, t = 11 := 
  by
    sorry

end pavan_travel_time_l131_131984


namespace simplify_expression_l131_131808

variable {a : ℝ}

theorem simplify_expression (h1 : a ≠ 2) (h2 : a ≠ -2) :
  ((a^2 + 4*a + 4) / (a^2 - 4) - (a + 3) / (a - 2)) / ((a + 2) / (a - 2)) = -1 / (a + 2) :=
by
  sorry

end simplify_expression_l131_131808


namespace angle_BAC_l131_131650

theorem angle_BAC
  (elevation_angle_B_from_A : ℝ)
  (depression_angle_C_from_A : ℝ)
  (h₁ : elevation_angle_B_from_A = 60)
  (h₂ : depression_angle_C_from_A = 70) :
  elevation_angle_B_from_A + depression_angle_C_from_A = 130 :=
by
  sorry

end angle_BAC_l131_131650


namespace student_game_incorrect_statement_l131_131765

theorem student_game_incorrect_statement (a : ℚ) : ¬ (∀ a : ℚ, -a - 2 < 0) :=
by
  -- skip the proof for now
  sorry

end student_game_incorrect_statement_l131_131765


namespace proof_problem_l131_131215

theorem proof_problem (a b c : ℝ) (h : a > b) (h1 : b > c) :
  (1 / (a - b) + 1 / (b - c) + 4 / (c - a) ≥ 0) :=
sorry

end proof_problem_l131_131215


namespace clay_weight_in_second_box_l131_131065

/-- Define the properties of the first and second boxes -/
structure Box where
  height : ℕ
  width : ℕ
  length : ℕ
  weight : ℕ

noncomputable def box1 : Box :=
  { height := 2, width := 3, length := 5, weight := 40 }

noncomputable def box2 : Box :=
  { height := 2 * 2, width := 3 * 3, length := 5, weight := 240 }

theorem clay_weight_in_second_box : 
  box2.weight = (box2.height * box2.width * box2.length) / 
                (box1.height * box1.width * box1.length) * box1.weight :=
by
  sorry

end clay_weight_in_second_box_l131_131065


namespace distance_between_points_l131_131055

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 2 5 5 1 = 5 :=
by
  sorry

end distance_between_points_l131_131055


namespace intersection_M_N_l131_131586

def M : Set ℝ := { x | x^2 - x - 6 ≤ 0 }
def N : Set ℝ := { x | -2 < x ∧ x ≤ 4 }

theorem intersection_M_N : (M ∩ N) = { x | -2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_M_N_l131_131586


namespace number_of_cars_l131_131746

theorem number_of_cars (b c : ℕ) (h1 : b = c / 10) (h2 : c - b = 90) : c = 100 :=
by
  sorry

end number_of_cars_l131_131746


namespace faye_age_l131_131210

def ages (C D E F : ℕ) :=
  D = E - 2 ∧
  E = C + 3 ∧
  F = C + 4 ∧
  D = 15

theorem faye_age (C D E F : ℕ) (h : ages C D E F) : F = 18 :=
by
  unfold ages at h
  sorry

end faye_age_l131_131210


namespace union_of_A_and_B_l131_131997

def A : Set ℝ := { x | 1 < x ∧ x < 3 }
def B : Set ℝ := { x | 2 < x ∧ x < 4 }

theorem union_of_A_and_B : A ∪ B = { x | 1 < x ∧ x < 4 } := by
  sorry

end union_of_A_and_B_l131_131997


namespace order_of_x_y_z_l131_131840

variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- Conditions
axiom h1 : 0.9 < x
axiom h2 : x < 1.0
axiom h3 : y = x^x
axiom h4 : z = x^(x^x)

-- Theorem to be proved
theorem order_of_x_y_z (h1 : 0.9 < x) (h2 : x < 1.0) (h3 : y = x^x) (h4 : z = x^(x^x)) : x < z ∧ z < y :=
by
  sorry

end order_of_x_y_z_l131_131840


namespace equivalent_xy_xxyy_not_equivalent_xyty_txy_not_equivalent_xy_xt_l131_131699

-- Define a transformation predicate for words
inductive transform : List Char -> List Char -> Prop
| xy_to_yyx : ∀ (l1 l2 : List Char), transform (l1 ++ ['x', 'y'] ++ l2) (l1 ++ ['y', 'y', 'x'] ++ l2)
| yyx_to_xy : ∀ (l1 l2 : List Char), transform (l1 ++ ['y', 'y', 'x'] ++ l2) (l1 ++ ['x', 'y'] ++ l2)
| xt_to_ttx : ∀ (l1 l2 : List Char), transform (l1 ++ ['x', 't'] ++ l2) (l1 ++ ['t', 't', 'x'] ++ l2)
| ttx_to_xt : ∀ (l1 l2 : List Char), transform (l1 ++ ['t', 't', 'x'] ++ l2) (l1 ++ ['x', 't'] ++ l2)
| yt_to_ty : ∀ (l1 l2 : List Char), transform (l1 ++ ['y', 't'] ++ l2) (l1 ++ ['t', 'y'] ++ l2)
| ty_to_yt : ∀ (l1 l2 : List Char), transform (l1 ++ ['t', 'y'] ++ l2) (l1 ++ ['y', 't'] ++ l2)

-- Reflexive and transitive closure of transform
inductive transforms : List Char -> List Char -> Prop
| base : ∀ l, transforms l l
| step : ∀ l m n, transform l m → transforms m n → transforms l n

-- Definitions for the words and their information
def word1 := ['x', 'x', 'y', 'y']
def word2 := ['x', 'y', 'y', 'y', 'y', 'x']
def word3 := ['x', 'y', 't', 'x']
def word4 := ['t', 'x', 'y', 't']
def word5 := ['x', 'y']
def word6 := ['x', 't']

-- Proof statements
theorem equivalent_xy_xxyy : transforms word1 word2 :=
by sorry

theorem not_equivalent_xyty_txy : ¬ transforms word3 word4 :=
by sorry

theorem not_equivalent_xy_xt : ¬ transforms word5 word6 :=
by sorry

end equivalent_xy_xxyy_not_equivalent_xyty_txy_not_equivalent_xy_xt_l131_131699


namespace how_many_pints_did_Annie_pick_l131_131139

theorem how_many_pints_did_Annie_pick (x : ℕ) (h1 : Kathryn = x + 2)
                                      (h2 : Ben = Kathryn - 3)
                                      (h3 : x + Kathryn + Ben = 25) : x = 8 :=
  sorry

end how_many_pints_did_Annie_pick_l131_131139


namespace trigonometric_identity_l131_131767

theorem trigonometric_identity (φ : ℝ) 
  (h : Real.cos (π / 2 + φ) = (Real.sqrt 3) / 2) : 
  Real.cos (3 * π / 2 - φ) + Real.sin (φ - π) = Real.sqrt 3 := 
by
  sorry

end trigonometric_identity_l131_131767


namespace emmanuel_jelly_beans_l131_131778

theorem emmanuel_jelly_beans (total_jelly_beans : ℕ)
      (thomas_percentage : ℕ)
      (barry_ratio : ℕ)
      (emmanuel_ratio : ℕ)
      (h1 : total_jelly_beans = 200)
      (h2 : thomas_percentage = 10)
      (h3 : barry_ratio = 4)
      (h4 : emmanuel_ratio = 5) :
  let thomas_jelly_beans := (thomas_percentage * total_jelly_beans) / 100
  let remaining_jelly_beans := total_jelly_beans - thomas_jelly_beans
  let total_ratio := barry_ratio + emmanuel_ratio
  let per_part_jelly_beans := remaining_jelly_beans / total_ratio
  let emmanuel_jelly_beans := emmanuel_ratio * per_part_jelly_beans
  emmanuel_jelly_beans = 100 :=
by
  sorry

end emmanuel_jelly_beans_l131_131778


namespace cans_per_bag_l131_131879

def total_cans : ℕ := 42
def bags_saturday : ℕ := 4
def bags_sunday : ℕ := 3
def total_bags : ℕ := bags_saturday + bags_sunday

theorem cans_per_bag (h1 : total_cans = 42) (h2 : total_bags = 7) : total_cans / total_bags = 6 :=
by {
    -- proof body to be filled
    sorry
}

end cans_per_bag_l131_131879


namespace inequality_holds_l131_131836

theorem inequality_holds (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  (a^3 / (a^3 + 15 * b * c * d))^(1/2) ≥ a^(15/8) / (a^(15/8) + b^(15/8) + c^(15/8) + d^(15/8)) :=
sorry

end inequality_holds_l131_131836


namespace domain_implies_range_a_range_implies_range_a_l131_131400

theorem domain_implies_range_a {a : ℝ} :
  (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) → 0 ≤ a ∧ a < 1 :=
sorry

theorem range_implies_range_a {a : ℝ} :
  (∀ y : ℝ, ∃ x : ℝ, ax^2 + 2 * a * x + 1 = y) → 1 ≤ a :=
sorry

end domain_implies_range_a_range_implies_range_a_l131_131400


namespace cherry_pie_probability_l131_131906

noncomputable def probability_of_cherry_pie : Real :=
  let packets := ["KK", "KV", "VV"]
  let prob :=
    (1/3 * 1/4) + -- Case KK broken, then picking from KV or VV
    (1/6 * 1/2) + -- Case KV broken (cabbage found), picking cherry from KV
    (1/3 * 1) + -- Case VV broken (cherry found), remaining cherry picked
    (1/6 * 0) -- Case KV broken (cherry found), remaining cabbage
  prob

theorem cherry_pie_probability : probability_of_cherry_pie = 2 / 3 :=
  sorry

end cherry_pie_probability_l131_131906


namespace minimum_zeros_l131_131828

theorem minimum_zeros (n : ℕ) (a : Fin n → ℤ) (h : n = 2011)
  (H : ∀ i j k : Fin n, a i + a j + a k ∈ Set.range a) : 
  ∃ (num_zeros : ℕ), num_zeros ≥ 2009 ∧ (∃ f : Fin (num_zeros) → Fin n, ∀ i : Fin (num_zeros), a (f i) = 0) :=
sorry

end minimum_zeros_l131_131828


namespace calculate_fraction_l131_131678

theorem calculate_fraction : 
  ∃ f : ℝ, (14.500000000000002 ^ 2) * f = 126.15 ∧ f = 0.6 :=
by
  sorry

end calculate_fraction_l131_131678


namespace find_solutions_l131_131154

-- Definitions
def is_solution (x y z n : ℕ) : Prop :=
  x^3 + y^3 + z^3 = n * (x^2) * (y^2) * (z^2)

-- Theorem statement
theorem find_solutions :
  {sol : ℕ × ℕ × ℕ × ℕ | is_solution sol.1 sol.2.1 sol.2.2.1 sol.2.2.2} =
  {(1, 1, 1, 3), (1, 2, 3, 1), (2, 1, 3, 1)} :=
by sorry

end find_solutions_l131_131154


namespace planting_flowers_cost_l131_131627

theorem planting_flowers_cost 
  (flower_cost : ℕ) (clay_cost : ℕ) (soil_cost : ℕ)
  (h₁ : flower_cost = 9)
  (h₂ : clay_cost = flower_cost + 20)
  (h₃ : soil_cost = flower_cost - 2) :
  flower_cost + clay_cost + soil_cost = 45 :=
sorry

end planting_flowers_cost_l131_131627


namespace total_earnings_proof_l131_131714

noncomputable def total_earnings (x y : ℝ) : ℝ :=
  let earnings_a := (18 * x * y) / 100
  let earnings_b := (20 * x * y) / 100
  let earnings_c := (20 * x * y) / 100
  earnings_a + earnings_b + earnings_c

theorem total_earnings_proof (x y : ℝ) (h : 2 * x * y = 15000) :
  total_earnings x y = 4350 := by
  sorry

end total_earnings_proof_l131_131714


namespace minimum_value_expression_l131_131306

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x * y * z) ≥ 343 :=
sorry

end minimum_value_expression_l131_131306


namespace other_candidate_votes_l131_131551

theorem other_candidate_votes (h1 : one_candidate_votes / valid_votes = 0.6)
    (h2 : 0.3 * total_votes = invalid_votes)
    (h3 : total_votes = 9000)
    (h4 : valid_votes + invalid_votes = total_votes) :
    valid_votes - one_candidate_votes = 2520 :=
by
  sorry

end other_candidate_votes_l131_131551


namespace domain_of_function_l131_131076

-- Define the setting and the constants involved
variables {f : ℝ → ℝ}
variable {c : ℝ}

-- The statement about the function's domain
theorem domain_of_function :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ (x ≤ 0 ∧ x ≠ -c) :=
sorry

end domain_of_function_l131_131076


namespace circles_intersect_l131_131509

-- Definition of the first circle
def circle1 (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Definition of the second circle
def circle2 (x y : ℝ) (r : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 49

-- Statement proving the range of r for which the circles intersect
theorem circles_intersect (r : ℝ) (h : r > 0) : (∃ x y : ℝ, circle1 x y r ∧ circle2 x y r) → (2 ≤ r ∧ r ≤ 12) :=
by
  -- Definition of the distance between centers and conditions for intersection
  sorry

end circles_intersect_l131_131509


namespace cat_and_dog_positions_l131_131113

def cat_position_after_365_moves : Nat :=
  let cycle_length := 9
  365 % cycle_length

def dog_position_after_365_moves : Nat :=
  let cycle_length := 16
  365 % cycle_length

theorem cat_and_dog_positions :
  cat_position_after_365_moves = 5 ∧ dog_position_after_365_moves = 13 :=
by
  sorry

end cat_and_dog_positions_l131_131113


namespace first_player_wins_l131_131935

def initial_piles (p1 p2 : Nat) : Prop :=
  p1 = 33 ∧ p2 = 35

def winning_strategy (p1 p2 : Nat) : Prop :=
  ∃ moves : List (Nat × Nat), 
  (initial_piles p1 p2) →
  (∀ (p1' p2' : Nat), 
    (p1', p2') ∈ moves →
    p1' = 1 ∧ p2' = 1 ∨ p1' = 2 ∧ p2' = 1)

theorem first_player_wins : winning_strategy 33 35 :=
sorry

end first_player_wins_l131_131935


namespace train_crossing_time_l131_131392

theorem train_crossing_time:
  ∀ (length_train : ℝ) (speed_man_kmph : ℝ) (speed_train_kmph : ℝ),
    length_train = 125 →
    speed_man_kmph = 5 →
    speed_train_kmph = 69.994 →
    (125 / ((69.994 + 5) * (1000 / 3600))) = 6.002 :=
by
  intros length_train speed_man_kmph speed_train_kmph h1 h2 h3
  sorry

end train_crossing_time_l131_131392


namespace vectors_coplanar_l131_131195

/-- Vectors defined as 3-dimensional Euclidean space vectors. --/
def vector3 := (ℝ × ℝ × ℝ)

/-- Definitions for vectors a, b, c as given in the problem conditions. --/
def a : vector3 := (3, 1, -1)
def b : vector3 := (1, 0, -1)
def c : vector3 := (8, 3, -2)

/-- The scalar triple product of vectors a, b, c is the determinant of the matrix formed. --/
noncomputable def scalarTripleProduct (u v w : vector3) : ℝ :=
  let (u1, u2, u3) := u
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  u1 * (v2 * w3 - v3 * w2) - u2 * (v1 * w3 - v3 * w1) + u3 * (v1 * w2 - v2 * w1)

/-- Statement to prove that vectors a, b, c are coplanar (i.e., their scalar triple product is zero). --/
theorem vectors_coplanar : scalarTripleProduct a b c = 0 :=
  by sorry

end vectors_coplanar_l131_131195


namespace cost_of_article_l131_131149

-- Conditions as Lean definitions
def price_1 : ℝ := 340
def price_2 : ℝ := 350
def price_diff : ℝ := price_2 - price_1 -- Rs. 10
def gain_percent_increase : ℝ := 0.04

-- Question: What is the cost of the article?
-- Answer: Rs. 90

theorem cost_of_article : ∃ C : ℝ, 
  price_diff = gain_percent_increase * (price_1 - C) ∧ C = 90 := 
sorry

end cost_of_article_l131_131149


namespace acute_angle_is_three_pi_over_eight_l131_131985

noncomputable def acute_angle_concentric_circles : Real :=
  let r₁ := 4
  let r₂ := 3
  let r₃ := 2
  let total_area := (r₁ * r₁ * Real.pi) + (r₂ * r₂ * Real.pi) + (r₃ * r₃ * Real.pi)
  let unshaded_area := 5 * (total_area / 8)
  let shaded_area := (3 / 5) * unshaded_area
  let theta := shaded_area / total_area * 2 * Real.pi
  theta

theorem acute_angle_is_three_pi_over_eight :
  acute_angle_concentric_circles = (3 * Real.pi / 8) :=
by
  sorry

end acute_angle_is_three_pi_over_eight_l131_131985


namespace derivative_f_at_pi_l131_131275

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem derivative_f_at_pi : (deriv f π) = -1 := 
by
  sorry

end derivative_f_at_pi_l131_131275


namespace correct_operation_l131_131126

theorem correct_operation (a : ℝ) : a^4 / a^2 = a^2 :=
by sorry

end correct_operation_l131_131126


namespace motorcyclist_initial_speed_l131_131082

theorem motorcyclist_initial_speed (x : ℝ) : 
  (120 = x * (120 / x)) ∧
  (120 = x + 6) → 
  (120 / x = 1 + 1/6 + (120 - x) / (x + 6)) →
  (x = 48) :=
by
  sorry

end motorcyclist_initial_speed_l131_131082


namespace find_a_and_vertices_find_y_range_find_a_range_l131_131083

noncomputable def quadratic_function (x a : ℝ) : ℝ :=
  x^2 - 6 * a * x + 9

theorem find_a_and_vertices (a : ℝ) :
  quadratic_function 2 a = 7 →
  a = 1 / 2 ∧
  (3 * a, quadratic_function (3 * a) a) = (3 / 2, 27 / 4) :=
sorry

theorem find_y_range (x a : ℝ) :
  a = 1 / 2 →
  -1 ≤ x ∧ x < 3 →
  27 / 4 ≤ quadratic_function x a ∧ quadratic_function x a ≤ 13 :=
sorry

theorem find_a_range (a : ℝ) (x1 x2 : ℝ) :
  (3 * a - 2 ≤ x1 ∧ x1 ≤ 5 ∧ 3 * a - 2 ≤ x2 ∧ x2 ≤ 5) →
  (x1 ≥ 3 ∧ x2 ≥ 3 → quadratic_function x1 a - quadratic_function x2 a ≤ 9 * a^2 + 20) →
  1 / 6 ≤ a ∧ a ≤ 1 :=
sorry

end find_a_and_vertices_find_y_range_find_a_range_l131_131083


namespace minimum_value_x_squared_plus_12x_plus_5_l131_131552

theorem minimum_value_x_squared_plus_12x_plus_5 : ∃ x : ℝ, x^2 + 12 * x + 5 = -31 :=
by sorry

end minimum_value_x_squared_plus_12x_plus_5_l131_131552


namespace tan_sum_identity_l131_131794

open Real

theorem tan_sum_identity : 
  tan (80 * π / 180) + tan (40 * π / 180) - sqrt 3 * tan (80 * π / 180) * tan (40 * π / 180) = -sqrt 3 :=
by
  sorry

end tan_sum_identity_l131_131794


namespace factorize_cubic_expression_l131_131380

theorem factorize_cubic_expression (m : ℝ) : 
  m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end factorize_cubic_expression_l131_131380


namespace grid_possible_configuration_l131_131902

theorem grid_possible_configuration (m n : ℕ) (hm : m > 100) (hn : n > 100) : 
  ∃ grid : ℕ → ℕ → ℕ,
  (∀ i j, grid i j = (if i > 0 then grid (i - 1) j else 0) + 
                       (if i < m - 1 then grid (i + 1) j else 0) + 
                       (if j > 0 then grid i (j - 1) else 0) + 
                       (if j < n - 1 then grid i (j + 1) else 0)) 
  ∧ (∃ i j, grid i j ≠ 0) 
  ∧ m > 14 
  ∧ n > 14 := 
sorry

end grid_possible_configuration_l131_131902


namespace profit_achieved_at_50_yuan_l131_131830

theorem profit_achieved_at_50_yuan :
  ∀ (x : ℝ), (30 ≤ x ∧ x ≤ 54) → 
  ((x - 30) * (80 - 2 * (x - 40)) = 1200) →
  x = 50 :=
by
  intros x h_range h_profit
  sorry

end profit_achieved_at_50_yuan_l131_131830


namespace find_n_plus_c_l131_131566

variables (n c : ℝ)

-- Conditions from the problem
def line1 := ∀ (x y : ℝ), (x = 4) → (y = 11) → (y = n * x + 3)
def line2 := ∀ (x y : ℝ), (x = 4) → (y = 11) → (y = 5 * x + c)

theorem find_n_plus_c (h1 : line1 n)
                      (h2 : line2 c) :
  n + c = -7 := by
  sorry

end find_n_plus_c_l131_131566


namespace num_terminating_decimals_l131_131975

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 518) :
  (∃ k, (1 ≤ k ∧ k ≤ 518) ∧ n = k * 21) ↔ n = 24 :=
sorry

end num_terminating_decimals_l131_131975


namespace B_plus_C_is_330_l131_131241

-- Definitions
def A : ℕ := 170
def B : ℕ := 300
def C : ℕ := 30

axiom h1 : A + B + C = 500
axiom h2 : A + C = 200
axiom h3 : C = 30

-- Theorem statement
theorem B_plus_C_is_330 : B + C = 330 :=
by
  sorry

end B_plus_C_is_330_l131_131241


namespace solve_for_x_l131_131424

theorem solve_for_x : ∃ x : ℚ, (1/4 : ℚ) + (1/x) = 7/8 ∧ x = 8/5 :=
by {
  sorry
}

end solve_for_x_l131_131424


namespace prob1_prob2_max_area_prob3_circle_diameter_l131_131027

-- Definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def line_through_center (x y : ℝ) : Prop := x - y - 3 = 0
def line_eq (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Problem 1: Line passes through the center of the circle
theorem prob1 (x y : ℝ) : line_through_center x y ↔ circle_eq x y :=
sorry

-- Problem 2: Maximum area of triangle CAB
theorem prob2_max_area (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 0 ∨ m = -6) :=
sorry

-- Problem 3: Circle with diameter AB passes through origin
theorem prob3_circle_diameter (x y : ℝ) (m : ℝ) : line_eq m x y → (m = 1 ∨ m = -4) :=
sorry

end prob1_prob2_max_area_prob3_circle_diameter_l131_131027


namespace area_ratio_of_similar_triangles_l131_131026

noncomputable def similarity_ratio := 3 / 5

theorem area_ratio_of_similar_triangles (k : ℝ) (h_sim : similarity_ratio = k) : (k^2 = 9 / 25) :=
by
  sorry

end area_ratio_of_similar_triangles_l131_131026


namespace arithmetic_seq_problem_l131_131668

variable (a : ℕ → ℕ)

def arithmetic_seq (a₁ d : ℕ) : ℕ → ℕ :=
  λ n => a₁ + n * d

theorem arithmetic_seq_problem (a₁ d : ℕ)
  (h_cond : (arithmetic_seq a₁ d 1) + 2 * (arithmetic_seq a₁ d 5) + (arithmetic_seq a₁ d 9) = 120)
  : (arithmetic_seq a₁ d 2) + (arithmetic_seq a₁ d 8) = 60 := 
sorry

end arithmetic_seq_problem_l131_131668


namespace driver_travel_distance_per_week_l131_131040

open Nat

-- Defining the parameters
def speed1 : ℕ := 30
def time1 : ℕ := 3
def speed2 : ℕ := 25
def time2 : ℕ := 4
def days : ℕ := 6

-- Lean statement to prove
theorem driver_travel_distance_per_week : 
  (speed1 * time1 + speed2 * time2) * days = 1140 := 
by 
  sorry

end driver_travel_distance_per_week_l131_131040


namespace percentage_employees_six_years_or_more_l131_131486

theorem percentage_employees_six_years_or_more:
  let marks : List ℕ := [6, 6, 7, 4, 3, 3, 3, 1, 1, 1]
  let total_employees (marks : List ℕ) (y : ℕ) := marks.foldl (λ acc m => acc + m * y) 0
  let employees_six_years_or_more (marks : List ℕ) (y : ℕ) := (marks.drop 6).foldl (λ acc m => acc + m * y) 0
  (employees_six_years_or_more marks 1 / total_employees marks 1 : ℚ) * 100 = 17.14 := by
  sorry

end percentage_employees_six_years_or_more_l131_131486


namespace num_unique_m_values_l131_131044

theorem num_unique_m_values : 
  ∃ (s : Finset Int), 
  (∀ (x1 x2 : Int), x1 * x2 = 36 → x1 + x2 ∈ s) ∧ 
  s.card = 10 := 
sorry

end num_unique_m_values_l131_131044


namespace distance_to_post_office_l131_131824

theorem distance_to_post_office
  (D : ℝ)
  (travel_rate : ℝ) (walk_rate : ℝ)
  (total_time_hours : ℝ)
  (h1 : travel_rate = 25)
  (h2 : walk_rate = 4)
  (h3 : total_time_hours = 5 + 48 / 60) :
  D = 20 :=
by
  sorry

end distance_to_post_office_l131_131824


namespace triangle_sin_double_angle_l131_131728

open Real

theorem triangle_sin_double_angle (A : ℝ) (h : cos (π / 4 + A) = 5 / 13) : sin (2 * A) = 119 / 169 :=
by
  sorry

end triangle_sin_double_angle_l131_131728


namespace sector_area_l131_131534

theorem sector_area (r θ : ℝ) (h₁ : θ = 2) (h₂ : r * θ = 4) : (1 / 2) * r^2 * θ = 4 :=
by
  sorry

end sector_area_l131_131534


namespace unicorn_rope_problem_l131_131271

theorem unicorn_rope_problem
  (d e f : ℕ)
  (h_prime_f : Prime f)
  (h_d : d = 75)
  (h_e : e = 450)
  (h_f : f = 3)
  : d + e + f = 528 := by
  sorry

end unicorn_rope_problem_l131_131271


namespace algebraic_expression_value_l131_131494

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 = 3 * y) :
  x^2 - 6 * x * y + 9 * y^2 = 4 :=
sorry

end algebraic_expression_value_l131_131494


namespace page_sum_incorrect_l131_131452

theorem page_sum_incorrect (sheets : List (Nat × Nat)) (h_sheets_len : sheets.length = 25)
  (h_consecutive : ∀ (a b : Nat), (a, b) ∈ sheets → (b = a + 1 ∨ a = b + 1))
  (h_sum_eq_2020 : (sheets.map (λ p => p.1 + p.2)).sum = 2020) : False :=
by
  sorry

end page_sum_incorrect_l131_131452


namespace cube_root_of_64_is_4_l131_131440

theorem cube_root_of_64_is_4 (x : ℝ) (h1 : 0 < x) (h2 : x^3 = 64) : x = 4 :=
by
  sorry

end cube_root_of_64_is_4_l131_131440


namespace grogg_expected_value_l131_131003

theorem grogg_expected_value (n : ℕ) (p : ℝ) (h_n : 2 ≤ n) (h_p : 0 < p ∧ p < 1) :
  (p + n * p^n * (1 - p) = 1) ↔ (p = 1 / n^(1/n:ℝ)) :=
sorry

end grogg_expected_value_l131_131003


namespace system_solution_y_greater_than_five_l131_131522

theorem system_solution_y_greater_than_five (m x y : ℝ) :
  (y = (m + 1) * x + 2) → 
  (y = (3 * m - 2) * x + 5) → 
  y > 5 ↔ 
  m ≠ 3 / 2 := 
sorry

end system_solution_y_greater_than_five_l131_131522


namespace lucas_siblings_product_is_35_l131_131774

-- Definitions based on the given conditions
def total_girls (lauren_sisters : ℕ) : ℕ := lauren_sisters + 1
def total_boys (lauren_brothers : ℕ) : ℕ := lauren_brothers + 1

-- Given conditions
def lauren_sisters : ℕ := 4
def lauren_brothers : ℕ := 7

-- Compute number of sisters (S) and brothers (B) Lucas has
def lucas_sisters : ℕ := total_girls lauren_sisters
def lucas_brothers : ℕ := lauren_brothers

theorem lucas_siblings_product_is_35 : 
  (lucas_sisters * lucas_brothers = 35) := by
  -- Asserting the correctness based on given family structure conditions
  sorry

end lucas_siblings_product_is_35_l131_131774


namespace cars_on_happy_street_l131_131829

theorem cars_on_happy_street :
  let cars_tuesday := 25
  let cars_monday := cars_tuesday - cars_tuesday * 20 / 100
  let cars_wednesday := cars_monday + 2
  let cars_thursday : ℕ := 10
  let cars_friday : ℕ := 10
  let cars_saturday : ℕ := 5
  let cars_sunday : ℕ := 5
  let total_cars := cars_monday + cars_tuesday + cars_wednesday + cars_thursday + cars_friday + cars_saturday + cars_sunday
  total_cars = 97 :=
by
  sorry

end cars_on_happy_street_l131_131829


namespace inequality_always_holds_l131_131858

theorem inequality_always_holds (a b : ℝ) (h₀ : a < b) (h₁ : b < 0) : a^2 > ab ∧ ab > b^2 :=
by
  sorry

end inequality_always_holds_l131_131858


namespace digit_distribution_l131_131449

theorem digit_distribution (n: ℕ) : 
(1 / 2) * n + (1 / 5) * n + (1 / 5) * n + (1 / 10) * n = n → 
n = 10 :=
by
  sorry

end digit_distribution_l131_131449


namespace prod_sum_leq_four_l131_131574

theorem prod_sum_leq_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 4) :
  ab + bc + cd + da ≤ 4 :=
sorry

end prod_sum_leq_four_l131_131574


namespace sum_of_eight_numbers_l131_131438

theorem sum_of_eight_numbers (avg : ℚ) (n : ℕ) (sum : ℚ) 
  (h_avg : avg = 5.3) (h_n : n = 8) : sum = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l131_131438


namespace find_K_3_15_10_l131_131856

def K (x y z : ℚ) : ℚ := 
  x / y + y / z + z / x + (x + y) / z

theorem find_K_3_15_10 : K 3 15 10 = 41 / 6 := 
  by
  sorry

end find_K_3_15_10_l131_131856


namespace intersection_of_A_and_B_l131_131581

def A : Set ℤ := {-2, -1}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B :
  A ∩ B = {-1} :=
sorry

end intersection_of_A_and_B_l131_131581


namespace two_digit_number_eq_27_l131_131058

theorem two_digit_number_eq_27 (A : ℕ) (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
    (h : A = 10 * x + y) (hcond : A = 3 * (x + y)) : A = 27 :=
by
  sorry

end two_digit_number_eq_27_l131_131058


namespace nickys_pace_l131_131698

theorem nickys_pace (distance : ℝ) (head_start_time : ℝ) (cristina_pace : ℝ) 
    (time_before_catchup : ℝ) (nicky_distance : ℝ) :
    distance = 100 ∧ head_start_time = 12 ∧ cristina_pace = 5 
    ∧ time_before_catchup = 30 ∧ nicky_distance = 90 →
    nicky_distance / time_before_catchup = 3 :=
by
  sorry

end nickys_pace_l131_131698


namespace min_sum_sequence_n_l131_131289

theorem min_sum_sequence_n (S : ℕ → ℤ) (h : ∀ n, S n = n * n - 48 * n) : 
  ∃ n, n = 24 ∧ ∀ m, S n ≤ S m :=
by
  sorry

end min_sum_sequence_n_l131_131289


namespace revenue_growth_20_percent_l131_131822

noncomputable def revenue_increase (R2000 R2003 R2005 : ℝ) : ℝ :=
  ((R2005 - R2003) / R2003) * 100

theorem revenue_growth_20_percent (R2000 : ℝ) (h1 : R2003 = 1.5 * R2000) (h2 : R2005 = 1.8 * R2000) :
  revenue_increase R2000 R2003 R2005 = 20 :=
by
  sorry

end revenue_growth_20_percent_l131_131822


namespace find_coefficient_m_l131_131107

theorem find_coefficient_m :
  ∃ m : ℝ, (1 + 2 * x)^3 = 1 + 6 * x + m * x^2 + 8 * x^3 ∧ m = 12 := by
  sorry

end find_coefficient_m_l131_131107


namespace topless_cubical_box_l131_131122

def squares : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

def valid_placement (s : Char) : Bool :=
  match s with
  | 'A' => true
  | 'B' => true
  | 'C' => true
  | 'D' => false
  | 'E' => false
  | 'F' => true
  | 'G' => true
  | 'H' => false
  | _ => false

def valid_configurations : List Char := squares.filter valid_placement

theorem topless_cubical_box:
  valid_configurations.length = 5 := by
  sorry

end topless_cubical_box_l131_131122


namespace middle_managers_to_be_selected_l131_131480

def total_employees : ℕ := 160
def senior_managers : ℕ := 10
def middle_managers : ℕ := 30
def staff_members : ℕ := 120
def total_to_be_selected : ℕ := 32

theorem middle_managers_to_be_selected : 
  (middle_managers * total_to_be_selected / total_employees) = 6 := by
  sorry

end middle_managers_to_be_selected_l131_131480


namespace polygon_E_has_largest_area_l131_131790

-- Define the areas of square and right triangle
def area_square (side : ℕ): ℕ := side * side
def area_right_triangle (leg : ℕ): ℕ := (leg * leg) / 2

-- Define the areas of each polygon
def area_polygon_A : ℕ := 2 * (area_square 2) + (area_right_triangle 2)
def area_polygon_B : ℕ := 3 * (area_square 2)
def area_polygon_C : ℕ := (area_square 2) + 4 * (area_right_triangle 2)
def area_polygon_D : ℕ := 3 * (area_right_triangle 2)
def area_polygon_E : ℕ := 4 * (area_square 2)

-- The theorem assertion
theorem polygon_E_has_largest_area : 
  area_polygon_E = 16 ∧ 
  16 > area_polygon_A ∧
  16 > area_polygon_B ∧
  16 > area_polygon_C ∧
  16 > area_polygon_D := 
sorry

end polygon_E_has_largest_area_l131_131790


namespace probability_diff_color_correct_l131_131903

noncomputable def probability_diff_color (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) : ℚ :=
  (red_balls * yellow_balls) / ((total_balls * (total_balls - 1)) / 2)

theorem probability_diff_color_correct :
  probability_diff_color 5 3 2 = 3 / 5 :=
by
  sorry

end probability_diff_color_correct_l131_131903


namespace total_assembly_time_l131_131748

-- Define the conditions
def chairs : ℕ := 2
def tables : ℕ := 2
def time_per_piece : ℕ := 8
def total_pieces : ℕ := chairs + tables

-- State the theorem
theorem total_assembly_time :
  total_pieces * time_per_piece = 32 :=
sorry

end total_assembly_time_l131_131748


namespace slope_of_asymptotes_l131_131615

theorem slope_of_asymptotes (a b : ℝ) (h : a^2 = 144) (k : b^2 = 81) : (b / a = 3 / 4) :=
by
  sorry

end slope_of_asymptotes_l131_131615


namespace original_number_is_600_l131_131356

theorem original_number_is_600 (x : Real) (h : x * 1.10 = 660) : x = 600 := by
  sorry

end original_number_is_600_l131_131356


namespace exists_odd_digit_div_by_five_power_l131_131303

theorem exists_odd_digit_div_by_five_power (n : ℕ) (h : 0 < n) : ∃ (k : ℕ), 
  (∃ (m : ℕ), k = m * 5^n) ∧ 
  (∀ (d : ℕ), (d = (k / (10^(n-1))) % 10) → d % 2 = 1) :=
sorry

end exists_odd_digit_div_by_five_power_l131_131303


namespace problem1_problem2_l131_131437

-- Problem 1: Prove \( \sqrt{10} \times \sqrt{2} + \sqrt{15} \div \sqrt{3} = 3\sqrt{5} \)
theorem problem1 : Real.sqrt 10 * Real.sqrt 2 + Real.sqrt 15 / Real.sqrt 3 = 3 * Real.sqrt 5 := 
by sorry

-- Problem 2: Prove \( \sqrt{27} - (\sqrt{12} - \sqrt{\frac{1}{3}}) = \frac{4\sqrt{3}}{3} \)
theorem problem2 : Real.sqrt 27 - (Real.sqrt 12 - Real.sqrt (1 / 3)) = (4 * Real.sqrt 3) / 3 :=
by sorry

end problem1_problem2_l131_131437


namespace g_at_5_l131_131394

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 35*x^2 - 40*x + 24

theorem g_at_5 : g 5 = 74 := 
by {
  sorry
}

end g_at_5_l131_131394


namespace original_board_length_before_final_cut_l131_131532

-- Given conditions
def initial_length : ℕ := 143
def first_cut_length : ℕ := 25
def final_cut_length : ℕ := 7

def board_length_after_first_cut : ℕ := initial_length - first_cut_length
def board_length_after_final_cut : ℕ := board_length_after_first_cut - final_cut_length

-- The theorem to be proved
theorem original_board_length_before_final_cut : board_length_after_first_cut + final_cut_length = 125 :=
by
  sorry

end original_board_length_before_final_cut_l131_131532


namespace students_attended_game_l131_131223

variable (s n : ℕ)

theorem students_attended_game (h1 : s + n = 3000) (h2 : 10 * s + 15 * n = 36250) : s = 1750 := by
  sorry

end students_attended_game_l131_131223


namespace solve_abs_quadratic_l131_131625

theorem solve_abs_quadratic :
  ∃ x : ℝ, (|x - 3| + x^2 = 10) ∧ 
  (x = (-1 + Real.sqrt 53) / 2 ∨ x = (1 + Real.sqrt 29) / 2 ∨ x = (1 - Real.sqrt 29) / 2) :=
by sorry

end solve_abs_quadratic_l131_131625


namespace binom_solution_l131_131211

theorem binom_solution (x y : ℕ) (hxy : x > 0 ∧ y > 0) (bin_eq : Nat.choose x y = 1999000) : x = 1999000 ∨ x = 2000 := 
by
  sorry

end binom_solution_l131_131211


namespace circle_values_of_a_l131_131916

theorem circle_values_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + 2*a*x + 2*a*y + 2*a^2 + a - 1 = 0) ↔ (a = -1 ∨ a = 0) :=
by
  sorry

end circle_values_of_a_l131_131916


namespace B_squared_ge_AC_l131_131823

variable {a b c A B C : ℝ}

theorem B_squared_ge_AC
  (h1 : b^2 < a * c)
  (h2 : a * C - 2 * b * B + c * A = 0) :
  B^2 ≥ A * C := 
sorry

end B_squared_ge_AC_l131_131823


namespace valid_fraction_l131_131649

theorem valid_fraction (x: ℝ) : x^2 + 1 ≠ 0 :=
by
  sorry

end valid_fraction_l131_131649


namespace parabola_vertex_l131_131519

theorem parabola_vertex :
  ∃ h k : ℝ, (∀ x : ℝ, y = 1 / 2 * (x + 1) ^ 2 - 1 / 2) →
    (h = -1 ∧ k = -1 / 2) :=
by
  sorry

end parabola_vertex_l131_131519


namespace union_of_A_and_B_l131_131443

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} := by
  sorry

end union_of_A_and_B_l131_131443


namespace solution_l131_131358

theorem solution (x : ℝ) (h : ¬ (x ^ 2 - 5 * x + 4 > 0)) : 1 ≤ x ∧ x ≤ 4 :=
by
  sorry

end solution_l131_131358


namespace max_value_of_expression_l131_131925

theorem max_value_of_expression (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + 2 * b + 3 * c = 1) :
    (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a) ≤ 7) :=
sorry

end max_value_of_expression_l131_131925


namespace divides_by_3_l131_131228

theorem divides_by_3 (a b c : ℕ) (h : 9 ∣ a ^ 3 + b ^ 3 + c ^ 3) : 3 ∣ a ∨ 3 ∣ b ∨ 3 ∣ c :=
sorry

end divides_by_3_l131_131228


namespace find_b_l131_131642

theorem find_b (b : ℝ) : (∃ x y : ℝ, x = 1 ∧ y = 2 ∧ y = 2 * x + b) → b = 0 := by
  sorry

end find_b_l131_131642


namespace fruit_difference_l131_131688

/-- Mr. Connell harvested 60 apples and 3 times as many peaches. The difference 
    between the number of peaches and apples is 120. -/
theorem fruit_difference (apples peaches : ℕ) (h1 : apples = 60) (h2 : peaches = 3 * apples) :
  peaches - apples = 120 :=
sorry

end fruit_difference_l131_131688


namespace min_value_of_f_range_of_a_l131_131464

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x - 1

theorem min_value_of_f : ∃ x ∈ Set.Ioi 0, ∀ y ∈ Set.Ioi 0, f y ≥ f x ∧ f x = -2 * Real.exp (-1) - 1 := 
  sorry

theorem range_of_a {a : ℝ} : (∀ x > 0, f x ≤ 3 * x^2 + 2 * a * x) ↔ a ∈ Set.Ici (-2) := 
  sorry

end min_value_of_f_range_of_a_l131_131464


namespace wholesale_price_is_90_l131_131770

theorem wholesale_price_is_90 
  (R S W: ℝ)
  (h1 : R = 120)
  (h2 : S = R - 0.1 * R)
  (h3 : S = W + 0.2 * W)
  : W = 90 := 
by
  sorry

end wholesale_price_is_90_l131_131770


namespace compare_m_n_l131_131111

theorem compare_m_n (b m n : ℝ) :
  m = -3 * (-2) + b ∧ n = -3 * (3) + b → m > n :=
by
  sorry

end compare_m_n_l131_131111


namespace cost_of_each_toy_l131_131781

theorem cost_of_each_toy (initial_money spent_money remaining_money toys_count toy_cost : ℕ) 
  (h1 : initial_money = 57)
  (h2 : spent_money = 27)
  (h3 : remaining_money = initial_money - spent_money)
  (h4 : toys_count = 5)
  (h5 : remaining_money / toys_count = toy_cost) :
  toy_cost = 6 :=
by
  sorry

end cost_of_each_toy_l131_131781


namespace abc_sum_is_twelve_l131_131466

theorem abc_sum_is_twelve
  (f : ℤ → ℤ)
  (a b c : ℕ)
  (h1 : f 1 = 10)
  (h2 : f 0 = 8)
  (h3 : f (-3) = -28)
  (h4 : ∀ x, x > 0 → f x = 2 * a * x + 6)
  (h5 : f 0 = a^2 * b)
  (h6 : ∀ x, x < 0 → f x = 2 * b * x + 2 * c)
  : a + b + c = 12 := sorry

end abc_sum_is_twelve_l131_131466


namespace find_unit_vector_l131_131960

theorem find_unit_vector (a b : ℝ) : 
  a^2 + b^2 = 1 ∧ 3 * a + 4 * b = 0 →
  (a = 4 / 5 ∧ b = -3 / 5) ∨ (a = -4 / 5 ∧ b = 3 / 5) :=
by sorry

end find_unit_vector_l131_131960


namespace b_2_pow_100_value_l131_131512

def seq (b : ℕ → ℕ) : Prop :=
  b 1 = 3 ∧ ∀ n > 0, b (2 * n) = 2 * n * b n

theorem b_2_pow_100_value
  (b : ℕ → ℕ)
  (h_seq : seq b) :
  b (2^100) = 2^5050 * 3 :=
by
  sorry

end b_2_pow_100_value_l131_131512


namespace sum_of_areas_of_circles_l131_131096

theorem sum_of_areas_of_circles :
  (∑' n : ℕ, π * (9 / 16) ^ n) = π * (16 / 7) :=
by
  sorry

end sum_of_areas_of_circles_l131_131096


namespace cistern_problem_l131_131556

theorem cistern_problem (fill_rate empty_rate net_rate : ℝ) (T : ℝ) : 
  fill_rate = 1 / 3 →
  net_rate = 7 / 30 →
  empty_rate = 1 / T →
  net_rate = fill_rate - empty_rate →
  T = 10 :=
by
  intros
  sorry

end cistern_problem_l131_131556


namespace profit_percentage_l131_131682

theorem profit_percentage (C S : ℝ) (hC : C = 800) (hS : S = 1080) :
  ((S - C) / C) * 100 = 35 := 
by
  sorry

end profit_percentage_l131_131682


namespace quarters_difference_nickels_eq_l131_131987

variable (q : ℕ)

def charles_quarters := 7 * q + 2
def richard_quarters := 3 * q + 7
def quarters_difference := charles_quarters q - richard_quarters q
def money_difference_in_nickels := 5 * quarters_difference q

theorem quarters_difference_nickels_eq :
  money_difference_in_nickels q = 20 * (q - 5/4) :=
by
  sorry

end quarters_difference_nickels_eq_l131_131987


namespace ratio_AD_DC_in_ABC_l131_131545

theorem ratio_AD_DC_in_ABC 
  (A B C D : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB BC AC : Real) 
  (hAB : AB = 6) (hBC : BC = 8) (hAC : AC = 10) 
  (BD : Real) 
  (hBD : BD = 8) 
  (AD DC : Real)
  (hAD : AD = 2 * Real.sqrt 7)
  (hDC : DC = 10 - 2 * Real.sqrt 7) :
  AD / DC = (10 * Real.sqrt 7 + 14) / 36 :=
sorry

end ratio_AD_DC_in_ABC_l131_131545


namespace original_price_of_coffee_l131_131000

variable (P : ℝ)

theorem original_price_of_coffee :
  (4 * P - 2 * (1.5 * P) = 2) → P = 2 :=
by
  sorry

end original_price_of_coffee_l131_131000


namespace solve_system_of_equations_l131_131272

theorem solve_system_of_equations (x y z : ℝ) :
  (x * y + 1 = 2 * z) →
  (y * z + 1 = 2 * x) →
  (z * x + 1 = 2 * y) →
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  ((x = -2 ∧ y = -2 ∧ z = 5/2) ∨
   (x = 5/2 ∧ y = -2 ∧ z = -2) ∨ 
   (x = -2 ∧ y = 5/2 ∧ z = -2)) :=
sorry

end solve_system_of_equations_l131_131272


namespace repeating_decimal_fraction_eq_l131_131839

-- Define repeating decimal and its equivalent fraction
def repeating_decimal_value : ℚ := 7 + 123 / 999

theorem repeating_decimal_fraction_eq :
  repeating_decimal_value = 2372 / 333 :=
by
  sorry

end repeating_decimal_fraction_eq_l131_131839


namespace right_triangle_roots_l131_131317

theorem right_triangle_roots (m a b c : ℝ) 
  (h_eq : ∀ x, x^2 - (2 * m + 1) * x + m^2 + m = 0)
  (h_roots : a^2 - (2 * m + 1) * a + m^2 + m = 0 ∧ b^2 - (2 * m + 1) * b + m^2 + m = 0)
  (h_triangle : a^2 + b^2 = c^2)
  (h_c : c = 5) : 
  m = 3 :=
by sorry

end right_triangle_roots_l131_131317


namespace range_of_m_l131_131851

theorem range_of_m (m: ℝ) : (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → x^2 - x + 1 > 2*x + m) → m < -1 :=
by
  intro h
  sorry

end range_of_m_l131_131851


namespace combined_area_difference_l131_131145

theorem combined_area_difference :
  let rect1_len := 11
  let rect1_wid := 11
  let rect2_len := 5.5
  let rect2_wid := 11
  2 * (rect1_len * rect1_wid) - 2 * (rect2_len * rect2_wid) = 121 := by
  sorry

end combined_area_difference_l131_131145


namespace min_segments_for_7_points_l131_131182

theorem min_segments_for_7_points (points : Fin 7 → ℝ × ℝ) : 
  ∃ (segments : Finset (Fin 7 × Fin 7)), 
    (∀ (a b c : Fin 7), a ≠ b ∧ b ≠ c ∧ c ≠ a → (a, b) ∈ segments ∨ (b, c) ∈ segments ∨ (c, a) ∈ segments) ∧
    segments.card = 9 :=
sorry

end min_segments_for_7_points_l131_131182


namespace farmer_purchase_l131_131769

theorem farmer_purchase : ∃ r c : ℕ, 30 * r + 45 * c = 1125 ∧ r > 0 ∧ c > 0 ∧ r = 3 ∧ c = 23 := 
by 
  sorry

end farmer_purchase_l131_131769


namespace negation_of_proposition_l131_131735

-- Definitions using the conditions stated
def p (x : ℝ) : Prop := x^2 - x + 1/4 ≥ 0

-- The statement to prove
theorem negation_of_proposition :
  (¬ (∀ x : ℝ, p x)) = (∃ x : ℝ, ¬ p x) :=
by
  -- Proof will go here; replaced by sorry as per instruction
  sorry

end negation_of_proposition_l131_131735


namespace jennys_wedding_guests_l131_131345

noncomputable def total_guests (C S : ℕ) : ℕ := C + S

theorem jennys_wedding_guests :
  ∃ (C S : ℕ), (S = 3 * C) ∧
               (18 * C + 25 * S = 1860) ∧
               (total_guests C S = 80) :=
sorry

end jennys_wedding_guests_l131_131345


namespace height_of_triangle_l131_131807

theorem height_of_triangle
    (A : ℝ) (b : ℝ) (h : ℝ)
    (h1 : A = 30)
    (h2 : b = 12)
    (h3 : A = (b * h) / 2) :
    h = 5 :=
by
  sorry

end height_of_triangle_l131_131807


namespace annual_income_calculation_l131_131133

noncomputable def annual_income (investment : ℝ) (price_per_share : ℝ) (dividend_rate : ℝ) (face_value : ℝ) : ℝ :=
  let number_of_shares := investment / price_per_share
  number_of_shares * face_value * dividend_rate

theorem annual_income_calculation :
  annual_income 4455 8.25 0.12 10 = 648 :=
by
  sorry

end annual_income_calculation_l131_131133


namespace solve_for_x_l131_131635

theorem solve_for_x (x : ℝ) (h₁ : x ≠ -3) :
  (7 * x^2 - 3) / (x + 3) - 3 / (x + 3) = 1 / (x + 3) ↔ x = 1 ∨ x = -1 := 
sorry

end solve_for_x_l131_131635


namespace periodic_odd_function_value_l131_131937

theorem periodic_odd_function_value (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
    (h_periodic : ∀ x : ℝ, f (x + 2) = f x) (h_value : f 0.5 = -1) : f 7.5 = 1 :=
by
  -- Proof would go here.
  sorry

end periodic_odd_function_value_l131_131937


namespace inequality_proof_l131_131256

open Real

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_product : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by {
  sorry
}

end inequality_proof_l131_131256


namespace parabola_circle_intersection_l131_131613

theorem parabola_circle_intersection (a : ℝ) : 
  a ≤ Real.sqrt 2 + 1 / 4 → 
  ∃ (b x y : ℝ), y = x^2 + a ∧ x^2 + y^2 + 2 * b^2 = 2 * b * (x - y) + 1 :=
by
  sorry

end parabola_circle_intersection_l131_131613


namespace average_speed_is_6_point_5_l131_131144

-- Define the given values
def total_distance : ℝ := 42
def riding_time : ℝ := 6
def break_time : ℝ := 0.5

-- Prove the average speed given the conditions
theorem average_speed_is_6_point_5 :
  (total_distance / (riding_time + break_time)) = 6.5 :=
by
  sorry

end average_speed_is_6_point_5_l131_131144


namespace jerry_age_l131_131871

theorem jerry_age (M J : ℕ) (hM : M = 24) (hCond : M = 4 * J - 20) : J = 11 := by
  sorry

end jerry_age_l131_131871


namespace symmetric_curve_eq_l131_131841

theorem symmetric_curve_eq : 
  (∃ x' y', (x' - 3)^2 + 4*(y' - 5)^2 = 4 ∧ (x' - 6 = x' + x) ∧ (y' - 10 = y' + y)) ->
  (∃ x y, (x - 6) ^ 2 + 4 * (y - 10) ^ 2 = 4) :=
by
  sorry

end symmetric_curve_eq_l131_131841


namespace inequality_solution_set_l131_131647

theorem inequality_solution_set (x : ℝ) : (x - 3) * (x + 2) < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end inequality_solution_set_l131_131647


namespace candy_distribution_l131_131889

theorem candy_distribution :
  let bags := 4
  let candies := 9
  (Nat.choose candies (candies - bags) * Nat.choose (candies - 1) (candies - bags - 1)) = 7056 :=
by
  -- define variables for bags and candies
  let bags := 4
  let candies := 9
  have h : (Nat.choose candies (candies - bags) * Nat.choose (candies - 1) (candies - bags - 1)) = 7056 := sorry
  exact h

end candy_distribution_l131_131889


namespace systematic_sampling_third_group_number_l131_131091

theorem systematic_sampling_third_group_number :
  ∀ (total_members groups sample_number group_5_number group_gap : ℕ),
  total_members = 200 →
  groups = 40 →
  sample_number = total_members / groups →
  group_5_number = 22 →
  group_gap = 5 →
  (group_this_number : ℕ) = group_5_number - (5 - 3) * group_gap →
  group_this_number = 12 :=
by
  intros total_members groups sample_number group_5_number group_gap Htotal Hgroups Hsample Hgroup5 Hgap Hthis_group
  sorry

end systematic_sampling_third_group_number_l131_131091


namespace value_of_a_8_l131_131523

noncomputable def S (n : ℕ) : ℕ := n^2
noncomputable def a (n : ℕ) : ℕ := if n = 1 then S n else S n - S (n - 1)

theorem value_of_a_8 : a 8 = 15 := 
by
  sorry

end value_of_a_8_l131_131523


namespace new_year_markup_l131_131127

variable (C : ℝ) -- original cost of the turtleneck sweater
variable (N : ℝ) -- New Year season markup in decimal form
variable (final_price : ℝ) -- final price in February

-- Conditions
def initial_markup (C : ℝ) := 1.20 * C
def after_new_year_markup (C : ℝ) (N : ℝ) := (1 + N) * initial_markup C
def discount_in_february (C : ℝ) (N : ℝ) := 0.94 * after_new_year_markup C N
def profit_in_february (C : ℝ) := 1.41 * C

-- Mathematically equivalent proof problem (statement only)
theorem new_year_markup :
  ∀ C : ℝ, ∀ N : ℝ,
    discount_in_february C N = profit_in_february C →
    N = 0.5 :=
by
  sorry

end new_year_markup_l131_131127


namespace average_length_of_strings_l131_131250

theorem average_length_of_strings : 
  let length1 := 2
  let length2 := 5
  let length3 := 3
  let total_length := length1 + length2 + length3 
  let average_length := total_length / 3
  average_length = 10 / 3 :=
by
  let length1 := 2
  let length2 := 5
  let length3 := 3
  let total_length := length1 + length2 + length3
  let average_length := total_length / 3
  have h1 : total_length = 10 := by rfl
  have h2 : average_length = 10 / 3 := by rfl
  exact h2

end average_length_of_strings_l131_131250


namespace cranberry_parts_l131_131423

theorem cranberry_parts (L C : ℕ) :
  L = 3 →
  L + C = 72 →
  C = L + 18 →
  C = 21 :=
by
  intros hL hSum hDiff
  sorry

end cranberry_parts_l131_131423


namespace smallest_root_of_polynomial_l131_131379

theorem smallest_root_of_polynomial :
  ∃ x : ℝ, (24 * x^3 - 106 * x^2 + 116 * x - 70 = 0) ∧ x = 0.67 :=
by
  sorry

end smallest_root_of_polynomial_l131_131379


namespace solve_system_of_equations_l131_131290

theorem solve_system_of_equations (x y : ℝ) :
  16 * x^3 + 4 * x = 16 * y + 5 ∧ 16 * y^3 + 4 * y = 16 * x + 5 → x = y ∧ 16 * x^3 - 12 * x - 5 = 0 :=
by
  sorry

end solve_system_of_equations_l131_131290


namespace product_of_three_numbers_l131_131596

theorem product_of_three_numbers (a b c : ℝ) 
  (h₁ : a + b + c = 45)
  (h₂ : a = 2 * (b + c))
  (h₃ : c = 4 * b) : 
  a * b * c = 1080 := 
sorry

end product_of_three_numbers_l131_131596


namespace weather_condition_l131_131721

theorem weather_condition (T : ℝ) (windy : Prop) (kites_will_fly : Prop) 
  (h1 : (T > 25 ∧ windy) → kites_will_fly) 
  (h2 : ¬ kites_will_fly) : T ≤ 25 ∨ ¬ windy :=
by 
  sorry

end weather_condition_l131_131721


namespace initial_gasoline_percentage_calculation_l131_131088

variable (initial_volume : ℝ)
variable (initial_ethanol_percentage : ℝ)
variable (additional_ethanol : ℝ)
variable (final_ethanol_percentage : ℝ)

theorem initial_gasoline_percentage_calculation
  (h1: initial_ethanol_percentage = 5)
  (h2: initial_volume = 45)
  (h3: additional_ethanol = 2.5)
  (h4: final_ethanol_percentage = 10) :
  100 - initial_ethanol_percentage = 95 :=
by
  sorry

end initial_gasoline_percentage_calculation_l131_131088


namespace find_y_l131_131811

theorem find_y (y : ℕ) : (1 / 8) * 2^36 = 8^y → y = 11 := by
  sorry

end find_y_l131_131811


namespace find_sum_of_squares_l131_131618

variable (x y : ℝ)

theorem find_sum_of_squares (h₁ : x * y = 8) (h₂ : x^2 * y + x * y^2 + x + y = 94) : 
  x^2 + y^2 = 7540 / 81 :=
by
  sorry

end find_sum_of_squares_l131_131618


namespace number_of_trees_l131_131628

theorem number_of_trees (length_of_yard : ℕ) (distance_between_trees : ℕ) 
(h1 : length_of_yard = 273) 
(h2 : distance_between_trees = 21) : 
(length_of_yard / distance_between_trees) + 1 = 14 := by
  sorry

end number_of_trees_l131_131628


namespace inequality_proof_l131_131873

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy: 0 < y) (hz : 0 < z):
  ( ( (x + y + z) / 3 ) ^ (x + y + z) ) ≤ x^x * y^y * z^z ∧ x^x * y^y * z^z ≤ ( (x^2 + y^2 + z^2) / (x + y + z) ) ^ (x + y + z) :=
by
  sorry

end inequality_proof_l131_131873


namespace bobby_final_paycheck_correct_l131_131332

def bobby_salary : ℕ := 450
def federal_tax_rate : ℚ := 1/3
def state_tax_rate : ℚ := 0.08
def health_insurance_deduction : ℕ := 50
def life_insurance_deduction : ℕ := 20
def city_parking_fee : ℕ := 10

def final_paycheck_amount : ℚ :=
  let federal_taxes := federal_tax_rate * bobby_salary
  let state_taxes := state_tax_rate * bobby_salary
  let total_deductions := federal_taxes + state_taxes + health_insurance_deduction + life_insurance_deduction + city_parking_fee
  bobby_salary - total_deductions

theorem bobby_final_paycheck_correct : final_paycheck_amount = 184 := by
  sorry

end bobby_final_paycheck_correct_l131_131332


namespace pins_after_one_month_l131_131234

def avg_pins_per_day : ℕ := 10
def delete_pins_per_week_per_person : ℕ := 5
def group_size : ℕ := 20
def initial_pins : ℕ := 1000

theorem pins_after_one_month
  (avg_pins_per_day_pos : avg_pins_per_day = 10)
  (delete_pins_per_week_per_person_pos : delete_pins_per_week_per_person = 5)
  (group_size_pos : group_size = 20)
  (initial_pins_pos : initial_pins = 1000) : 
  1000 + (avg_pins_per_day * group_size * 30) - (delete_pins_per_week_per_person * group_size * 4) = 6600 :=
by
  sorry

end pins_after_one_month_l131_131234


namespace monomial_sum_mn_l131_131506

theorem monomial_sum_mn (m n : ℤ) 
  (h1 : m + 6 = 1) 
  (h2 : 2 * n + 1 = 7) : 
  m * n = -15 := by
  sorry

end monomial_sum_mn_l131_131506


namespace domain_of_log_sqrt_l131_131805

theorem domain_of_log_sqrt (x : ℝ) : (-1 < x ∧ x ≤ 3) ↔ (0 < x + 1 ∧ 3 - x ≥ 0) :=
by
  sorry

end domain_of_log_sqrt_l131_131805


namespace f_odd_and_periodic_l131_131687

open Function

-- Define the function f : ℝ → ℝ satisfying the given conditions
variables (f : ℝ → ℝ)

-- Conditions
axiom f_condition1 : ∀ x : ℝ, f (10 + x) = f (10 - x)
axiom f_condition2 : ∀ x : ℝ, f (20 - x) = -f (20 + x)

-- Theorem statement
theorem f_odd_and_periodic : Odd f ∧ Periodic f 40 :=
by
  -- Proof will be filled here
  sorry

end f_odd_and_periodic_l131_131687


namespace weight_jordan_after_exercise_l131_131861

def initial_weight : ℕ := 250
def first_4_weeks_loss : ℕ := 3 * 4
def next_8_weeks_loss : ℕ := 2 * 8
def total_weight_loss : ℕ := first_4_weeks_loss + next_8_weeks_loss
def final_weight : ℕ := initial_weight - total_weight_loss

theorem weight_jordan_after_exercise : final_weight = 222 :=
by 
  sorry

end weight_jordan_after_exercise_l131_131861


namespace min_value_l131_131469

-- Conditions
variables {x y : ℝ}
variable (hx : x > 0)
variable (hy : y > 0)
variable (hxy : x + y = 2)

-- Theorem
theorem min_value (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  ∃ x y, (x > 0) ∧ (y > 0) ∧ (x + y = 2) ∧ (1/x + 4/y = 9/2) := 
by
  sorry

end min_value_l131_131469


namespace problem_a_b_n_l131_131472

theorem problem_a_b_n (a b n : ℕ) (h : ∀ k : ℕ, k ≠ 0 → (b - k) ∣ (a - k^n)) : a = b^n := 
sorry

end problem_a_b_n_l131_131472


namespace real_imag_equal_complex_l131_131733

/-- Given i is the imaginary unit, and a is a real number,
if the real part and the imaginary part of the complex number -3i(a+i) are equal,
then a = -1. -/
theorem real_imag_equal_complex (a : ℝ) (i : ℂ) (h_i : i * i = -1) 
    (h_eq : (3 : ℂ) = -(3 : ℂ) * a * i) : a = -1 :=
sorry

end real_imag_equal_complex_l131_131733


namespace income_of_deceased_l131_131900

def average_income (total_income : ℕ) (members : ℕ) : ℕ :=
  total_income / members

theorem income_of_deceased
  (total_income_before : ℕ) (members_before : ℕ) (avg_income_before : ℕ)
  (total_income_after : ℕ) (members_after : ℕ) (avg_income_after : ℕ) :
  total_income_before = members_before * avg_income_before →
  total_income_after = members_after * avg_income_after →
  members_before = 4 →
  members_after = 3 →
  avg_income_before = 735 →
  avg_income_after = 650 →
  total_income_before - total_income_after = 990 :=
by
  sorry

end income_of_deceased_l131_131900


namespace find_a10_l131_131116

theorem find_a10 (a : ℕ → ℚ) 
  (h1 : a 1 = 1/2)
  (h2 : ∀ n : ℕ, 0 < n → (1 / (a (n + 1) - 1)) = (1 / (a n - 1)) - 1) : 
  a 10 = 10/11 := 
sorry

end find_a10_l131_131116


namespace simplify_expression_l131_131609

theorem simplify_expression : 
    1 - 1 / (1 + Real.sqrt (2 + Real.sqrt 3)) + 1 / (1 - Real.sqrt (2 - Real.sqrt 3)) 
    = 1 + (Real.sqrt (2 - Real.sqrt 3) + Real.sqrt (2 + Real.sqrt 3)) / (-1 - Real.sqrt 3) := 
by
  sorry

end simplify_expression_l131_131609


namespace sum_of_youngest_and_oldest_friend_l131_131087

-- Given definitions
def mean_age_5 := 12
def median_age_5 := 11
def one_friend_age := 10

-- The total sum of ages is given by mean * number of friends
def total_sum_ages : ℕ := 5 * mean_age_5

-- Third friend's age as defined by median
def third_friend_age := 11

-- Proving the sum of the youngest and oldest friend's ages
theorem sum_of_youngest_and_oldest_friend:
  (∃ youngest oldest : ℕ, youngest + oldest = 38) :=
by
  sorry

end sum_of_youngest_and_oldest_friend_l131_131087


namespace stratified_sampling_category_A_l131_131171

def total_students_A : ℕ := 2000
def total_students_B : ℕ := 3000
def total_students_C : ℕ := 4000
def total_students : ℕ := total_students_A + total_students_B + total_students_C
def total_selected : ℕ := 900

theorem stratified_sampling_category_A :
  (total_students_A * total_selected) / total_students = 200 :=
by
  sorry

end stratified_sampling_category_A_l131_131171


namespace banana_cost_l131_131558

theorem banana_cost (pounds: ℕ) (rate: ℕ) (per_pounds: ℕ) : 
 (pounds = 18) → (rate = 3) → (per_pounds = 3) → 
  (pounds / per_pounds * rate = 18) := by
  intros
  sorry

end banana_cost_l131_131558


namespace woman_worked_days_l131_131913

theorem woman_worked_days :
  ∃ (W I : ℕ), (W + I = 25) ∧ (20 * W - 5 * I = 450) ∧ W = 23 := by
  sorry

end woman_worked_days_l131_131913


namespace perfect_square_polynomial_l131_131214

-- Define the polynomial and the conditions
def polynomial (a b : ℚ) := fun x : ℚ => x^4 + x^3 + 2 * x^2 + a * x + b

-- The expanded form of a quadratic trinomial squared
def quadratic_square (p q : ℚ) := fun x : ℚ =>
  x^4 + 2 * p * x^3 + (p^2 + 2 * q) * x^2 + 2 * p * q * x + q^2

-- Main theorem statement
theorem perfect_square_polynomial :
  ∃ (a b : ℚ), 
  (∀ x : ℚ, polynomial a b x = (quadratic_square (1/2 : ℚ) (7/8 : ℚ) x)) ↔ 
  a = 7/8 ∧ b = 49/64 :=
by
  sorry

end perfect_square_polynomial_l131_131214


namespace arithmetic_mean_of_numbers_l131_131333

theorem arithmetic_mean_of_numbers (n : ℕ) (h : n > 1) :
  let one_special_number := (1 / n) + (2 / n ^ 2)
  let other_numbers := (n - 1) * 1
  (other_numbers + one_special_number) / n = 1 + 2 / n ^ 2 :=
by
  sorry

end arithmetic_mean_of_numbers_l131_131333


namespace alpha_pi_over_four_sufficient_not_necessary_l131_131921

theorem alpha_pi_over_four_sufficient_not_necessary :
  (∀ α : ℝ, (α = (Real.pi / 4) → Real.cos α = Real.sqrt 2 / 2)) ∧
  (∃ α : ℝ, (Real.cos α = Real.sqrt 2 / 2) ∧ α ≠ (Real.pi / 4)) :=
by
  sorry

end alpha_pi_over_four_sufficient_not_necessary_l131_131921


namespace total_players_l131_131605

-- Definitions for conditions
def K : Nat := 10
def KK : Nat := 30
def B : Nat := 5

-- Statement of the proof problem
theorem total_players : K + KK - B = 35 :=
by
  -- Proof not required, just providing the statement
  sorry

end total_players_l131_131605


namespace simplify_fraction_l131_131433

variable {F : Type*} [Field F]

theorem simplify_fraction (a b : F) (h: a ≠ -1) :
  b / (a * b + b) = 1 / (a + 1) :=
by
  sorry

end simplify_fraction_l131_131433


namespace cakes_left_correct_l131_131777

def number_of_cakes_left (total_cakes sold_cakes : ℕ) : ℕ :=
  total_cakes - sold_cakes

theorem cakes_left_correct :
  number_of_cakes_left 54 41 = 13 :=
by
  sorry

end cakes_left_correct_l131_131777


namespace tiling_tromino_l131_131209

theorem tiling_tromino (m n : ℕ) : (∀ t : ℕ, (t = 3) → (3 ∣ m * n)) → (m * n % 6 = 0) → (m * n % 6 = 0) :=
by
  sorry

end tiling_tromino_l131_131209


namespace fish_weight_l131_131296

variables (H T X : ℝ)
-- Given conditions
def tail_weight : Prop := X = 1
def head_weight : Prop := H = X + 0.5 * T
def torso_weight : Prop := T = H + X

theorem fish_weight (H T X : ℝ) 
  (h_tail : tail_weight X)
  (h_head : head_weight H T X)
  (h_torso : torso_weight H T X) : 
  H + T + X = 8 :=
sorry

end fish_weight_l131_131296


namespace find_other_solution_l131_131550

theorem find_other_solution (x : ℚ) :
  (72 * x ^ 2 + 43 = 113 * x - 12) → (x = 3 / 8) → (x = 43 / 36 ∨ x = 3 / 8) :=
by
  sorry

end find_other_solution_l131_131550


namespace D_won_zero_matches_l131_131165

-- Define the players
inductive Player
| A | B | C | D deriving DecidableEq

-- Function to determine the winner of a match
def match_winner (p1 p2 : Player) : Option Player :=
  if p1 = Player.A ∧ p2 = Player.D then 
    some Player.A
  else if p2 = Player.A ∧ p1 = Player.D then 
    some Player.A
  else 
    none -- This represents that we do not know the outcome for matches not given

-- Assuming A, B, and C have won the same number of matches
def same_wins (w_A w_B w_C : Nat) : Prop := 
  w_A = w_B ∧ w_B = w_C

-- Define the problem statement
theorem D_won_zero_matches (w_D : Nat) (h_winner_AD: match_winner Player.A Player.D = some Player.A)
  (h_same_wins : ∃ w_A w_B w_C : Nat, same_wins w_A w_B w_C) : w_D = 0 :=
sorry

end D_won_zero_matches_l131_131165


namespace find_other_number_l131_131093

theorem find_other_number (lcm_ab : Nat) (gcd_ab : Nat) (a b : Nat) 
  (hlcm : Nat.lcm a b = lcm_ab) 
  (hgcd : Nat.gcd a b = gcd_ab) 
  (ha : a = 210) 
  (hlcm_ab : lcm_ab = 2310) 
  (hgcd_ab : gcd_ab = 55) 
  : b = 605 := 
by 
  sorry

end find_other_number_l131_131093


namespace factorize_expression_l131_131137

theorem factorize_expression (x : ℝ) : x^3 - 4 * x^2 + 4 * x = x * (x - 2)^2 := 
  sorry

end factorize_expression_l131_131137


namespace cos_B_in_third_quadrant_l131_131216

theorem cos_B_in_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB: Real.sin B = 5 / 13) : Real.cos B = - 12 / 13 := by
  sorry

end cos_B_in_third_quadrant_l131_131216


namespace a4_eq_12_l131_131885

-- Definitions of the sequences and conditions
def S (n : ℕ) : ℕ := 
  -- sum of the first n terms, initially undefined
  sorry  

def a (n : ℕ) : ℕ := 
  -- terms of the sequence, initially undefined
  sorry  

-- Given conditions
axiom a2_eq_3 : a 2 = 3
axiom Sn_recurrence : ∀ n ≥ 2, S (n + 1) = 2 * S n

-- Statement to prove
theorem a4_eq_12 : a 4 = 12 :=
  sorry

end a4_eq_12_l131_131885


namespace books_per_shelf_l131_131243

theorem books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves : ℕ) (books_left : ℕ) (books_per_shelf : ℕ) :
  total_books = 46 →
  books_taken = 10 →
  shelves = 9 →
  books_left = total_books - books_taken →
  books_per_shelf = books_left / shelves →
  books_per_shelf = 4 :=
by
  sorry

end books_per_shelf_l131_131243


namespace construct_origin_from_A_and_B_l131_131597

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def B : Point := ⟨3, 1⟩
def isAboveAndToLeft (p₁ p₂ : Point) : Prop := p₁.x < p₂.x ∧ p₁.y > p₂.y
def isOriginConstructed (A B : Point) : Prop := ∃ O : Point, O = ⟨0, 0⟩

theorem construct_origin_from_A_and_B : 
  isAboveAndToLeft A B → isOriginConstructed A B :=
by
  sorry

end construct_origin_from_A_and_B_l131_131597


namespace unique_elements_condition_l131_131366

theorem unique_elements_condition (x : ℝ) : 
  (1 ≠ x ∧ x ≠ x^2 ∧ 1 ≠ x^2) ↔ (x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :=
by 
  sorry

end unique_elements_condition_l131_131366


namespace age_of_son_l131_131999

theorem age_of_son (S M : ℕ) (h1 : M = S + 28) (h2 : M + 2 = 2 * (S + 2)) : S = 26 := by
  sorry

end age_of_son_l131_131999


namespace find_min_value_l131_131529

noncomputable def min_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_slope : 2 * a + b = 1) :=
  (8 * a + b) / (a * b)

theorem find_min_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_slope : 2 * a + b = 1) :
  min_value a b h_a h_b h_slope = 18 :=
sorry

end find_min_value_l131_131529


namespace james_total_cost_l131_131948

def suit1 := 300
def suit2_pretail := 3 * suit1
def suit2 := suit2_pretail + 200
def total_cost := suit1 + suit2

theorem james_total_cost : total_cost = 1400 := by
  sorry

end james_total_cost_l131_131948


namespace no_positive_integer_solutions_l131_131484

theorem no_positive_integer_solutions :
  ∀ (a b : ℕ), (a > 0) ∧ (b > 0) → 3 * a^2 ≠ b^2 + 1 :=
by
  sorry

end no_positive_integer_solutions_l131_131484


namespace a_n_divisible_by_2013_a_n_minus_207_is_cube_l131_131363

theorem a_n_divisible_by_2013 (n : ℕ) (h : n ≥ 1) : 2013 ∣ (4 ^ (6 ^ n) + 1943) :=
by sorry

theorem a_n_minus_207_is_cube (n : ℕ) : (∃ k : ℕ, 4 ^ (6 ^ n) + 1736 = k^3) ↔ (n = 1) :=
by sorry

end a_n_divisible_by_2013_a_n_minus_207_is_cube_l131_131363


namespace triangle_angle_measure_l131_131912

theorem triangle_angle_measure {D E F : ℝ} (hD : D = 90) (hE : E = 2 * F + 15) : 
  D + E + F = 180 → F = 25 :=
by
  intro h_sum
  sorry

end triangle_angle_measure_l131_131912


namespace combined_alloy_force_l131_131273

-- Define the masses and forces exerted by Alloy A and Alloy B
def mass_A : ℝ := 6
def force_A : ℝ := 30
def mass_B : ℝ := 3
def force_B : ℝ := 10

-- Define the combined mass and force
def combined_mass : ℝ := mass_A + mass_B
def combined_force : ℝ := force_A + force_B

-- Theorem statement
theorem combined_alloy_force :
  combined_force = 40 :=
by
  -- The proof is omitted.
  sorry

end combined_alloy_force_l131_131273


namespace remainder_of_98_times_102_divided_by_9_l131_131508

theorem remainder_of_98_times_102_divided_by_9 : (98 * 102) % 9 = 6 :=
by
  sorry

end remainder_of_98_times_102_divided_by_9_l131_131508


namespace Hadley_walked_to_grocery_store_in_2_miles_l131_131713

-- Define the variables and conditions
def distance_to_grocery_store (x : ℕ) : Prop :=
  x + (x - 1) + 3 = 6

-- Stating the main proposition to prove
theorem Hadley_walked_to_grocery_store_in_2_miles : ∃ x : ℕ, distance_to_grocery_store x ∧ x = 2 := 
by sorry

end Hadley_walked_to_grocery_store_in_2_miles_l131_131713


namespace evaluate_expression_l131_131691

theorem evaluate_expression (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (5 - x) + (5 - x) ^ 2 = 49 := 
sorry

end evaluate_expression_l131_131691


namespace vector_linear_combination_l131_131680

open Matrix

theorem vector_linear_combination :
  let v1 := ![3, -9]
  let v2 := ![2, -8]
  let v3 := ![1, -6]
  4 • v1 - 3 • v2 + 2 • v3 = ![8, -24] :=
by sorry

end vector_linear_combination_l131_131680


namespace solve_x_for_fraction_l131_131164

theorem solve_x_for_fraction :
  ∃ x : ℝ, (3 * x - 15) / 4 = (x + 7) / 3 ∧ x = 14.6 :=
by
  sorry

end solve_x_for_fraction_l131_131164


namespace max_n_for_factorable_quadratic_l131_131425

theorem max_n_for_factorable_quadratic :
  ∃ n : ℤ, (∀ x : ℤ, ∃ A B : ℤ, (3*x^2 + n*x + 108) = (3*x + A)*( x + B) ∧ A*B = 108 ∧ n = A + 3*B) ∧ n = 325 :=
by
  sorry

end max_n_for_factorable_quadratic_l131_131425


namespace three_distinct_real_roots_l131_131656

theorem three_distinct_real_roots 
  (c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (x1*x1 + 6*x1 + c)*(x1*x1 + 6*x1 + c) = 0 ∧ 
    (x2*x2 + 6*x2 + c)*(x2*x2 + 6*x2 + c) = 0 ∧ 
    (x3*x3 + 6*x3 + c)*(x3*x3 + 6*x3 + c) = 0) 
  ↔ c = (11 - Real.sqrt 13) / 2 :=
by sorry

end three_distinct_real_roots_l131_131656


namespace cubic_roots_proof_l131_131236

noncomputable def cubic_roots_reciprocal (a b c : ℝ) (h1 : a + b + c = 7) 
  (h2 : a * b + b * c + c * a = 3) (h3 : a * b * c = -4) : ℝ :=
  (1 / a^2) + (1 / b^2) + (1 / c^2)

theorem cubic_roots_proof (a b c : ℝ) (h1 : a + b + c = 7) 
  (h2 : a * b + b * c + c * a = 3) (h3 : a * b * c = -4) : 
  cubic_roots_reciprocal a b c h1 h2 h3 = 65 / 16 :=
sorry

end cubic_roots_proof_l131_131236


namespace max_viewers_per_week_l131_131842

theorem max_viewers_per_week :
  ∃ (x y : ℕ), 80 * x + 40 * y ≤ 320 ∧ x + y ≥ 6 ∧ 600000 * x + 200000 * y = 2000000 :=
by
  sorry

end max_viewers_per_week_l131_131842


namespace unique_real_solution_between_consecutive_integers_l131_131208

theorem unique_real_solution_between_consecutive_integers (k : ℕ) (h : k > 0) :
  ∃! x : ℝ, k < x ∧ x < k + 1 ∧ (⌊x⌋ : ℝ) * (x^2 + 1) = x^3 := sorry

end unique_real_solution_between_consecutive_integers_l131_131208


namespace solve_eq_1_solve_eq_2_l131_131539

open Real

theorem solve_eq_1 :
  ∃ x : ℝ, x - 2 * (x - 4) = 3 * (1 - x) ∧ x = -2.5 :=
by
  sorry

theorem solve_eq_2 :
  ∃ x : ℝ, (2 * x + 1) / 3 - (5 * x - 1) / 60 = 1 ∧ x = 39 / 35 :=
by
  sorry

end solve_eq_1_solve_eq_2_l131_131539


namespace repeating_decimal_sum_l131_131513

theorem repeating_decimal_sum :
  (0.12121212 + 0.003003003 + 0.0000500005 : ℚ) = 124215 / 999999 :=
by 
  have h1 : (0.12121212 : ℚ) = (0.12 + 0.0012) := sorry
  have h2 : (0.003003003 : ℚ) = (0.003 + 0.000003) := sorry
  have h3 : (0.0000500005 : ℚ) = (0.00005 + 0.0000000005) := sorry
  sorry


end repeating_decimal_sum_l131_131513


namespace range_of_a_l131_131833

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

def no_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c < 0

theorem range_of_a (a : ℝ) :
  no_real_roots 1 (2 * a - 1) 1 ↔ -1 / 2 < a ∧ a < 3 / 2 := 
by sorry

end range_of_a_l131_131833


namespace sqrt_10_bounds_l131_131492

theorem sqrt_10_bounds : 10 > 9 ∧ 10 < 16 → 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := 
by 
  sorry

end sqrt_10_bounds_l131_131492


namespace find_speed_ratio_l131_131991

noncomputable def circular_track_speed_ratio (v_V v_P C : ℝ) (H1 : v_V > 0) (H2 : v_P > 0) : Prop :=
  let t_1 := C / (v_V + v_P)
  let t_2 := (C * (2 * v_V + v_P)) / (v_V * (v_V + v_P))
  ∃ (r : ℝ), r = (1 + Real.sqrt 5) / 2 ∧ v_V / v_P = r

theorem find_speed_ratio
  (v_V v_P C : ℝ) (H1 : v_V > 0) (H2 : v_P > 0)
  (meeting1 : v_V * (C / (v_V + v_P)) + v_P * (C / (v_V + v_P)) = C)
  (lap_vasya : v_V * ((C * (2 * v_V + v_P)) / (v_V * (v_V + v_P))) = C + v_V * (C / (v_V + v_P)))
  (lap_petya : v_P * ((C * (2 * v_P + v_V)) / (v_P * (v_V + v_P))) = C + v_P * (C / (v_V + v_P))) :
  ∃ (r : ℝ), r = (1 + Real.sqrt 5) / 2 ∧ v_V / v_P = r :=
  sorry

end find_speed_ratio_l131_131991


namespace sum_of_coordinates_of_D_l131_131269

/--
Given points A = (4,8), B = (2,4), C = (6,6), and D = (a,b) in the first quadrant, if the quadrilateral formed by joining the midpoints of the segments AB, BC, CD, and DA is a square with sides inclined at 45 degrees to the x-axis, then the sum of the coordinates of point D is 6.
-/
theorem sum_of_coordinates_of_D 
  (a b : ℝ)
  (h_quadrilateral : ∃ A B C D : Prod ℝ ℝ, 
    A = (4, 8) ∧ B = (2, 4) ∧ C = (6, 6) ∧ D = (a, b) ∧ 
    ∃ M1 M2 M3 M4 : Prod ℝ ℝ,
    M1 = ((4 + 2) / 2, (8 + 4) / 2) ∧ M2 = ((2 + 6) / 2, (4 + 6) / 2) ∧ 
    M3 = (M2.1 + 1, M2.2 - 1) ∧ M4 = (M3.1 + 1, M3.2 + 1) ∧ 
    M3 = ((a + 6) / 2, (b + 6) / 2) ∧ M4 = ((a + 4) / 2, (b + 8) / 2)
  ) : 
  a + b = 6 := sorry

end sum_of_coordinates_of_D_l131_131269


namespace complement_of_M_in_U_l131_131956

namespace ProofProblem

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by
  sorry

end ProofProblem

end complement_of_M_in_U_l131_131956


namespace find_a_l131_131372

theorem find_a (a : ℝ) (h : ∀ B: ℝ × ℝ, (B = (a, 0)) → (2 - 0) * (0 - 2) = (4 - 2) * (2 - a)) : a = 4 :=
by
  sorry

end find_a_l131_131372


namespace isabel_homework_problems_l131_131979

theorem isabel_homework_problems (initial_problems finished_problems remaining_pages problems_per_page : ℕ) 
  (h1 : initial_problems = 72)
  (h2 : finished_problems = 32)
  (h3 : remaining_pages = 5)
  (h4 : initial_problems - finished_problems = 40)
  (h5 : 40 = remaining_pages * problems_per_page) : 
  problems_per_page = 8 := 
by sorry

end isabel_homework_problems_l131_131979


namespace tan_problem_l131_131222

theorem tan_problem (m : ℝ) (α : ℝ) (h1 : Real.tan α = m / 3) (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
sorry

end tan_problem_l131_131222


namespace find_a_l131_131409

variable (a b : ℝ × ℝ) 

axiom b_eq : b = (2, -1)
axiom length_eq_one : ‖a + b‖ = 1
axiom parallel_x_axis : (a + b).snd = 0

theorem find_a : a = (-1, 1) ∨ a = (-3, 1) := by
  sorry

end find_a_l131_131409


namespace solve_quadratic_eq_l131_131653

theorem solve_quadratic_eq (x : ℝ) : x ^ 2 + 2 * x - 5 = 0 → (x = -1 + Real.sqrt 6 ∨ x = -1 - Real.sqrt 6) :=
by 
  intro h
  sorry

end solve_quadratic_eq_l131_131653


namespace solution_set_inequality_l131_131007

theorem solution_set_inequality (m : ℤ) (h₁ : (∃! x : ℤ, |2 * x - m| ≤ 1 ∧ x = 2)) :
  {x : ℝ | |x - 1| + |x - 3| ≥ m} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 4} :=
by
  -- The detailed proof would be added here.
  sorry

end solution_set_inequality_l131_131007


namespace correctPropositions_l131_131399

-- Define the conditions and statement as Lean structures.
structure Geometry :=
  (Line : Type)
  (Plane : Type)
  (parallel : Plane → Plane → Prop)
  (parallelLine : Line → Plane → Prop)
  (perpendicular : Plane → Plane → Prop)
  (perpendicularLine : Line → Plane → Prop)
  (subsetLine : Line → Plane → Prop)

-- Main theorem to be proved in Lean 4
theorem correctPropositions (G : Geometry) :
  (∀ (α β : G.Plane) (a : G.Line), (G.parallel α β) → (G.subsetLine a α) → (G.parallelLine a β)) ∧ 
  (∀ (α β : G.Plane) (a : G.Line), (G.perpendicularLine a α) → (G.perpendicularLine a β) → (G.parallel α β)) :=
sorry -- The proof is omitted, as per instructions

end correctPropositions_l131_131399


namespace natural_number_squares_l131_131969

theorem natural_number_squares (n : ℕ) (h : ∃ k : ℕ, n^2 + 492 = k^2) :
    n = 122 ∨ n = 38 :=
by
  sorry

end natural_number_squares_l131_131969


namespace leap_years_among_given_years_l131_131826

-- Definitions for conditions
def is_divisible (a b : Nat) : Prop := b ≠ 0 ∧ a % b = 0

def is_leap_year (y : Nat) : Prop :=
  is_divisible y 4 ∧ (¬ is_divisible y 100 ∨ is_divisible y 400)

-- Statement of the problem
theorem leap_years_among_given_years :
  is_leap_year 1996 ∧ is_leap_year 2036 ∧ (¬ is_leap_year 1700) ∧ (¬ is_leap_year 1998) :=
by
  -- Proof would go here
  sorry

end leap_years_among_given_years_l131_131826


namespace color_coat_drying_time_l131_131190

theorem color_coat_drying_time : ∀ (x : ℕ), 2 + 2 * x + 5 = 13 → x = 3 :=
by
  intro x
  intro h
  sorry

end color_coat_drying_time_l131_131190


namespace simplify_polynomial_l131_131821

theorem simplify_polynomial :
  (3 * x^3 + 4 * x^2 + 8 * x - 5) - (2 * x^3 + x^2 + 6 * x - 7) = x^3 + 3 * x^2 + 2 * x + 2 := 
by
  sorry

end simplify_polynomial_l131_131821


namespace min_ab_value_l131_131982

theorem min_ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 / a + 20 / b = 4) : ab = 25 :=
sorry

end min_ab_value_l131_131982


namespace pinning_7_nails_l131_131098

theorem pinning_7_nails {n : ℕ} (circles : Fin n → Set (ℝ × ℝ)) :
  (∀ i j : Fin n, i ≠ j → ∃ p : ℝ × ℝ, p ∈ circles i ∧ p ∈ circles j) →
  ∃ s : Finset (ℝ × ℝ), s.card ≤ 7 ∧ ∀ i : Fin n, ∃ p : ℝ × ℝ, p ∈ s ∧ p ∈ circles i :=
by sorry

end pinning_7_nails_l131_131098


namespace minimum_a_inequality_l131_131660

variable {x y : ℝ}

/-- The inequality (x + y) * (1/x + a/y) ≥ 9 holds for any positive real numbers x and y 
     if and only if a ≥ 4.  -/
theorem minimum_a_inequality (a : ℝ) (h : ∀ (x y : ℝ), 0 < x → 0 < y → (x + y) * (1 / x + a / y) ≥ 9) :
  a ≥ 4 :=
by
  sorry

end minimum_a_inequality_l131_131660


namespace teacher_students_and_ticket_cost_l131_131328

theorem teacher_students_and_ticket_cost 
    (C_s C_a : ℝ) 
    (n_k n_h : ℕ)
    (hk_total ht_total : ℝ) 
    (h_students : n_h = n_k + 3)
    (hk  : n_k * C_s + C_a = hk_total)
    (ht : n_h * C_s + C_a = ht_total)
    (hk_total_val : hk_total = 994)
    (ht_total_val : ht_total = 1120)
    (C_s_val : C_s = 42) : 
    (n_h = 25) ∧ (C_a = 70) := 
by
  -- Proof steps would be provided here
  sorry

end teacher_students_and_ticket_cost_l131_131328


namespace ticket_cost_correct_l131_131420

def metro_sells (tickets_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  tickets_per_minute * minutes

def total_earnings (tickets_sold : ℕ) (ticket_cost : ℕ) : ℕ :=
  tickets_sold * ticket_cost

theorem ticket_cost_correct (ticket_cost : ℕ) : 
  (metro_sells 5 6 = 30) ∧ (total_earnings 30 ticket_cost = 90) → ticket_cost = 3 :=
by
  intro h
  sorry

end ticket_cost_correct_l131_131420


namespace cindy_correct_answer_l131_131853

theorem cindy_correct_answer (x : ℤ) (h : (x - 7) / 5 = 37) : (x - 5) / 7 = 26 :=
sorry

end cindy_correct_answer_l131_131853


namespace sin_cos_identity_l131_131041

theorem sin_cos_identity (α : ℝ) (h1 : Real.sin (α - Real.pi / 6) = 1 / 3) :
    Real.sin (2 * α - Real.pi / 6) + Real.cos (2 * α) = 7 / 9 :=
sorry

end sin_cos_identity_l131_131041


namespace probability_of_specific_sequence_l131_131951

variable (p : ℝ) (h : 0 < p ∧ p < 1)

theorem probability_of_specific_sequence :
  (1 - p)^7 * p^3 = sorry :=
by sorry

end probability_of_specific_sequence_l131_131951


namespace total_legs_correct_l131_131342

-- Number of animals
def horses : ℕ := 2
def dogs : ℕ := 5
def cats : ℕ := 7
def turtles : ℕ := 3
def goat : ℕ := 1

-- Total number of animals
def total_animals : ℕ := horses + dogs + cats + turtles + goat

-- Total number of legs
def total_legs : ℕ := total_animals * 4

theorem total_legs_correct : total_legs = 72 := by
  -- proof skipped
  sorry

end total_legs_correct_l131_131342


namespace find_pairs_1984_l131_131310

theorem find_pairs_1984 (m n : ℕ) :
  19 * m + 84 * n = 1984 ↔ (m = 100 ∧ n = 1) ∨ (m = 16 ∧ n = 20) :=
by
  sorry

end find_pairs_1984_l131_131310


namespace number_of_kids_l131_131013

theorem number_of_kids (A K : ℕ) (h1 : A + K = 13) (h2 : 7 * A = 28) : K = 9 :=
by
  sorry

end number_of_kids_l131_131013


namespace max_k_guarded_l131_131194

-- Define the size of the board
def board_size : ℕ := 8

-- Define the directions a guard can look
inductive Direction
| up | down | left | right

-- Define a guard's position on the board as a pair of Fin 8
def Position := Fin board_size × Fin board_size

-- Guard record that contains its position and direction
structure Guard where
  pos : Position
  dir : Direction

-- Function to determine if guard A is guarding guard B
def is_guarding (a b : Guard) : Bool :=
  match a.dir with
  | Direction.up    => a.pos.1 < b.pos.1 ∧ a.pos.2 = b.pos.2
  | Direction.down  => a.pos.1 > b.pos.1 ∧ a.pos.2 = b.pos.2
  | Direction.left  => a.pos.1 = b.pos.1 ∧ a.pos.2 > b.pos.2
  | Direction.right => a.pos.1 = b.pos.1 ∧ a.pos.2 < b.pos.2

-- The main theorem states that the maximum k is 5
theorem max_k_guarded : ∃ k : ℕ, (∀ g : Guard, ∃ S : Finset Guard, (S.card ≥ k) ∧ (∀ s ∈ S, is_guarding s g)) ∧ k = 5 :=
by
  sorry

end max_k_guarded_l131_131194


namespace largest_inscribed_square_l131_131693

-- Define the problem data
noncomputable def s : ℝ := 15
noncomputable def h : ℝ := s * (Real.sqrt 3) / 2
noncomputable def y : ℝ := s - h

-- Statement to prove
theorem largest_inscribed_square :
  y = (30 - 15 * Real.sqrt 3) / 2 := by
  sorry

end largest_inscribed_square_l131_131693


namespace new_average_production_l131_131258

theorem new_average_production (n : ℕ) (daily_avg : ℕ) (today_prod : ℕ) (new_avg : ℕ) 
  (h1 : daily_avg = 50) 
  (h2 : today_prod = 95) 
  (h3 : n = 8) 
  (h4 : new_avg = (daily_avg * n + today_prod) / (n + 1)) : 
  new_avg = 55 := 
sorry

end new_average_production_l131_131258


namespace arithmetic_sequence_9th_term_l131_131555

theorem arithmetic_sequence_9th_term (S : ℕ → ℕ) (d : ℕ) (Sn : ℕ) (a9 : ℕ) :
  (∀ n, S n = (n * (2 * S 1 + (n - 1) * d)) / 2) →
  d = 2 →
  Sn = 81 →
  S 9 = Sn →
  a9 = S 1 + 8 * d →
  a9 = 17 :=
by
  sorry

end arithmetic_sequence_9th_term_l131_131555


namespace number_of_squares_l131_131934

def draws_88_lines (lines: ℕ) : Prop := lines = 88
def draws_triangles (triangles: ℕ) : Prop := triangles = 12
def draws_pentagons (pentagons: ℕ) : Prop := pentagons = 4

theorem number_of_squares (triangles pentagons sq_sides: ℕ) (h1: draws_88_lines (triangles * 3 + pentagons * 5 + sq_sides * 4))
    (h2: draws_triangles triangles) (h3: draws_pentagons pentagons) : sq_sides = 8 := by
  sorry

end number_of_squares_l131_131934


namespace find_lawn_width_l131_131102

/-- Given a rectangular lawn with a length of 80 m and roads each 10 m wide,
    one running parallel to the length and the other running parallel to the width,
    with a total travel cost of Rs. 3300 at Rs. 3 per sq m, prove that the width of the lawn is 30 m. -/
theorem find_lawn_width (w : ℕ) (h_area_road : 10 * w + 10 * 80 = 1100) : w = 30 :=
by {
  sorry
}

end find_lawn_width_l131_131102


namespace square_root_combination_l131_131526

theorem square_root_combination (a : ℝ) (h : 1 + a = 4 - 2 * a) : a = 1 :=
by
  -- proof goes here
  sorry

end square_root_combination_l131_131526


namespace find_angle_A_l131_131159

variable {A B C a b c : ℝ}
variable {triangle_ABC : Prop}

theorem find_angle_A
  (h1 : a^2 + c^2 = b^2 + 2 * a * c * Real.cos C)
  (h2 : a = 2 * b * Real.sin A)
  (h3 : Real.cos B = Real.cos C)
  (h_triangle_angles : triangle_ABC) : A = 2 * Real.pi / 3 := 
by
  sorry

end find_angle_A_l131_131159


namespace time_to_drain_l131_131239

theorem time_to_drain (V R C : ℝ) (hV : V = 75000) (hR : R = 60) (hC : C = 0.80) : 
  (V * C) / R = 1000 := by
  sorry

end time_to_drain_l131_131239


namespace zoe_correct_percentage_l131_131888

noncomputable def t : ℝ := sorry  -- total number of problems
noncomputable def chloe_alone_correct : ℝ := 0.70 * (1/3 * t)  -- Chloe's correct answers alone
noncomputable def chloe_total_correct : ℝ := 0.85 * t  -- Chloe's overall correct answers
noncomputable def together_correct : ℝ := chloe_total_correct - chloe_alone_correct  -- Problems solved correctly together
noncomputable def zoe_alone_correct : ℝ := 0.85 * (1/3 * t)  -- Zoe's correct answers alone
noncomputable def zoe_total_correct : ℝ := zoe_alone_correct + together_correct  -- Zoe's total correct answers
noncomputable def zoe_percentage_correct : ℝ := (zoe_total_correct / t) * 100  -- Convert to percentage

theorem zoe_correct_percentage : zoe_percentage_correct = 90 := 
by
  sorry

end zoe_correct_percentage_l131_131888


namespace euclidean_division_mod_l131_131119

theorem euclidean_division_mod (h1 : 2022 % 19 = 8)
                               (h2 : 8^6 % 19 = 1)
                               (h3 : 2023 % 6 = 1)
                               (h4 : 2023^2024 % 6 = 1) 
: 2022^(2023^2024) % 19 = 8 := 
by
  sorry

end euclidean_division_mod_l131_131119


namespace factorization_eq_l131_131489

variable (x y : ℝ)

theorem factorization_eq : 9 * y - 25 * x^2 * y = y * (3 + 5 * x) * (3 - 5 * x) :=
by sorry 

end factorization_eq_l131_131489


namespace allen_reading_days_l131_131143

theorem allen_reading_days (pages_per_day : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 10) (h2 : total_pages = 120) : 
  (total_pages / pages_per_day) = 12 := by
  sorry

end allen_reading_days_l131_131143


namespace range_of_a_l131_131896

-- Define sets A and B and the condition A ∩ B = ∅
def set_A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def set_B (a : ℝ) : Set ℝ := { x | x > a }

-- State the condition: A ∩ B = ∅ implies a ≥ 1
theorem range_of_a (a : ℝ) : (set_A ∩ set_B a = ∅) → a ≥ 1 :=
  by
  sorry

end range_of_a_l131_131896


namespace compound_interest_amount_l131_131329

theorem compound_interest_amount (P r t SI : ℝ) (h1 : t = 3) (h2 : r = 0.10) (h3 : SI = 900) :
  SI = P * r * t → P = 900 / (0.10 * 3) → (P * (1 + r)^t - P = 993) :=
by
  intros hSI hP
  sorry

end compound_interest_amount_l131_131329


namespace power_of_2_not_sum_of_consecutive_not_power_of_2_is_sum_of_consecutive_l131_131632

-- Definitions and conditions
def is_power_of_2 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2^k

def is_sum_of_two_or_more_consecutive_naturals (n : ℕ) : Prop :=
  ∃ (a k : ℕ), k ≥ 2 ∧ n = (k * a) + (k * (k - 1)) / 2

-- Proofs to be stated
theorem power_of_2_not_sum_of_consecutive (n : ℕ) (h : is_power_of_2 n) : ¬ is_sum_of_two_or_more_consecutive_naturals n :=
by
    sorry

theorem not_power_of_2_is_sum_of_consecutive (M : ℕ) (h : ¬ is_power_of_2 M) : is_sum_of_two_or_more_consecutive_naturals M :=
by
    sorry

end power_of_2_not_sum_of_consecutive_not_power_of_2_is_sum_of_consecutive_l131_131632


namespace inverse_of_B_cubed_l131_131035

theorem inverse_of_B_cubed
  (B_inv : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![3, -1],
    ![0, 5]
  ]) :
  (B_inv ^ 3) = ![
    ![27, -49],
    ![0, 125]
  ] := 
by
  sorry

end inverse_of_B_cubed_l131_131035


namespace find_m_value_l131_131608

noncomputable def is_direct_proportion_function (m : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 1) * x ^ (2 - m^2) = k * x

theorem find_m_value (m : ℝ) (hk : ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 1) * x ^ (2 - m^2) = k * x) : m = -1 :=
by
  sorry

end find_m_value_l131_131608


namespace no_common_points_l131_131313

theorem no_common_points : 
  ∀ (x y : ℝ), ¬(x^2 + y^2 = 9 ∧ x^2 + y^2 = 4) := 
by
  sorry

end no_common_points_l131_131313


namespace toys_per_day_l131_131089

theorem toys_per_day (total_toys_per_week : ℕ) (days_worked_per_week : ℕ)
  (production_rate_constant : Prop) (h1 : total_toys_per_week = 8000)
  (h2 : days_worked_per_week = 4)
  (h3 : production_rate_constant)
  : (total_toys_per_week / days_worked_per_week) = 2000 :=
by
  sorry

end toys_per_day_l131_131089


namespace original_integer_is_21_l131_131882

theorem original_integer_is_21 (a b c d : ℕ) 
  (h1 : (a + b + c) / 3 + d = 29) 
  (h2 : (a + b + d) / 3 + c = 23) 
  (h3 : (a + c + d) / 3 + b = 21) 
  (h4 : (b + c + d) / 3 + a = 17) : 
  d = 21 :=
sorry

end original_integer_is_21_l131_131882


namespace bruce_age_multiple_of_son_l131_131053

structure Person :=
  (age : ℕ)

def bruce := Person.mk 36
def son := Person.mk 8
def multiple := 3

theorem bruce_age_multiple_of_son :
  ∃ (x : ℕ), bruce.age + x = multiple * (son.age + x) ∧ x = 6 :=
by
  use 6
  sorry

end bruce_age_multiple_of_son_l131_131053


namespace parabola_point_focus_distance_l131_131988

/-- 
  Given a point P on the parabola y^2 = 4x, and the distance from P to the line x = -2
  is 5 units, prove that the distance from P to the focus of the parabola is 4 units.
-/
theorem parabola_point_focus_distance {P : ℝ × ℝ} 
  (hP : P.2^2 = 4 * P.1) 
  (h_dist : (P.1 + 2)^2 + P.2^2 = 25) : 
  dist P (1, 0) = 4 :=
sorry

end parabola_point_focus_distance_l131_131988


namespace cos_product_triangle_l131_131620

theorem cos_product_triangle (A B C : ℝ) (h : A + B + C = π) (hA : A > 0) (hB : B > 0) (hC : C > 0) : 
  Real.cos A * Real.cos B * Real.cos C ≤ 1 / 8 := 
sorry

end cos_product_triangle_l131_131620


namespace quad_condition_l131_131801

noncomputable def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x - 4 * a

theorem quad_condition (a : ℝ) : (-16 ≤ a ∧ a ≤ 0) → (∀ x : ℝ, quadratic a x > 0) ↔ (¬ ∃ x : ℝ, quadratic a x ≤ 0) := by
  sorry

end quad_condition_l131_131801


namespace factor_expression_l131_131470

variable {a : ℝ}

theorem factor_expression :
  ((10 * a^4 - 160 * a^3 - 32) - (-2 * a^4 - 16 * a^3 + 32)) = 4 * (3 * a^3 * (a - 12) - 16) :=
by
  sorry

end factor_expression_l131_131470


namespace rhombus_area_l131_131744

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 150 :=
by
  rw [h1, h2]
  norm_num

end rhombus_area_l131_131744


namespace euler_polyhedron_problem_l131_131147

theorem euler_polyhedron_problem
  (V E F : ℕ)
  (t h T H : ℕ)
  (euler_formula : V - E + F = 2)
  (faces_count : F = 30)
  (tri_hex_faces : t + h = 30)
  (edges_equation : E = (3 * t + 6 * h) / 2)
  (vertices_equation1 : V = (3 * t) / T)
  (vertices_equation2 : V = (6 * h) / H)
  (T_val : T = 1)
  (H_val : H = 2)
  (t_val : t = 10)
  (h_val : h = 20)
  (edges_val : E = 75)
  (vertices_val : V = 60) :
  100 * H + 10 * T + V = 270 :=
by
  sorry

end euler_polyhedron_problem_l131_131147


namespace distance_AC_l131_131395

theorem distance_AC (A B C : ℤ) (h₁ : abs (B - A) = 5) (h₂ : abs (C - B) = 3) : abs (C - A) = 2 ∨ abs (C - A) = 8 :=
sorry

end distance_AC_l131_131395


namespace train_length_is_350_meters_l131_131237

noncomputable def length_of_train (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let time_hr := time_sec / 3600
  speed_kmh * time_hr * 1000

theorem train_length_is_350_meters :
  length_of_train 60 21 = 350 :=
by
  sorry

end train_length_is_350_meters_l131_131237


namespace first_tier_price_level_is_10000_l131_131260

noncomputable def first_tier_price_level (P : ℝ) : Prop :=
  ∀ (car_price : ℝ), car_price = 30000 → (P ≤ car_price ∧ 
    (0.25 * P + 0.15 * (car_price - P)) = 5500)

theorem first_tier_price_level_is_10000 :
  first_tier_price_level 10000 :=
by
  sorry

end first_tier_price_level_is_10000_l131_131260


namespace sufficient_and_necessary_condition_l131_131447

theorem sufficient_and_necessary_condition (x : ℝ) : 
  2 * x - 4 ≥ 0 ↔ x ≥ 2 :=
sorry

end sufficient_and_necessary_condition_l131_131447


namespace solve_for_q_l131_131932

theorem solve_for_q (t h q : ℝ) (h_eq : h = -14 * (t - 3)^2 + q) (h_5_eq : h = 94) (t_5_eq : t = 3 + 2) : q = 150 :=
by
  sorry

end solve_for_q_l131_131932


namespace jail_time_calculation_l131_131360

def total_arrests (arrests_per_day : ℕ) (cities : ℕ) (days : ℕ) : ℕ := 
  arrests_per_day * cities * days

def jail_time_before_trial (arrests : ℕ) (days_before_trial : ℕ) : ℕ := 
  days_before_trial * arrests

def jail_time_after_trial (arrests : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_after_trial * arrests

def combined_jail_time (weeks_before_trial : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_before_trial + weeks_after_trial

noncomputable def total_jail_time_in_weeks : ℕ := 
  let arrests := total_arrests 10 21 30
  let weeks_before_trial := jail_time_before_trial arrests 4 / 7
  let weeks_after_trial := jail_time_after_trial arrests 1
  combined_jail_time weeks_before_trial weeks_after_trial

theorem jail_time_calculation : 
  total_jail_time_in_weeks = 9900 :=
sorry

end jail_time_calculation_l131_131360


namespace problem_l131_131959

theorem problem (a b a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℕ)
  (b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ b₁₀ b₁₁ : ℕ) 
  (h1 : a₁ < a₂) (h2 : a₂ < a₃) (h3 : a₃ < a₄) 
  (h4 : a₄ < a₅) (h5 : a₅ < a₆) (h6 : a₆ < a₇)
  (h7 : a₇ < a₈) (h8 : a₈ < a₉) (h9 : a₉ < a₁₀)
  (h10 : a₁₀ < a₁₁) (h11 : b₁ < b₂) (h12 : b₂ < b₃)
  (h13 : b₃ < b₄) (h14 : b₄ < b₅) (h15 : b₅ < b₆)
  (h16 : b₆ < b₇) (h17 : b₇ < b₈) (h18 : b₈ < b₉)
  (h19 : b₉ < b₁₀) (h20 : b₁₀ < b₁₁) 
  (h21 : a₁₀ + b₁₀ = a) (h22 : a₁₁ + b₁₁ = b) : 
  a = 1024 ∧ b = 2048 :=
sorry

end problem_l131_131959


namespace range_of_a_l131_131831

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ x^2 + 2 * a * x + 2 * a + 3 < 0) ↔ a < -1 :=
sorry

end range_of_a_l131_131831


namespace min_value_function_l131_131810

open Real

theorem min_value_function (x y : ℝ) 
  (hx : x > -2 ∧ x < 2) 
  (hy : y > -2 ∧ y < 2) 
  (hxy : x * y = -1) : 
  (∃ u : ℝ, u = (4 / (4 - x^2) + 9 / (9 - y^2)) ∧ u = 12 / 5) :=
sorry

end min_value_function_l131_131810


namespace largest_n_for_crates_l131_131028

theorem largest_n_for_crates (total_crates : ℕ) (min_oranges max_oranges : ℕ)
  (h1 : total_crates = 145)
  (h2 : min_oranges = 110)
  (h3 : max_oranges = 140) : 
  ∃ n : ℕ, n = 5 ∧ ∀ k : ℕ, k ≤ max_oranges - min_oranges + 1 → total_crates / k ≤ n :=
  by {
    sorry
  }

end largest_n_for_crates_l131_131028


namespace find_k_unique_solution_l131_131930

theorem find_k_unique_solution (k : ℝ) (h: k ≠ 0) : (∀ x : ℝ, (x + 3) / (k * x - 2) = x → k = -3/4) :=
sorry

end find_k_unique_solution_l131_131930


namespace select_10_teams_l131_131599

def football_problem (teams : Finset ℕ) (played_on_day1 : Finset (ℕ × ℕ)) (played_on_day2 : Finset (ℕ × ℕ)) : Prop :=
  ∀ (v : ℕ), v ∈ teams → (∃ u w : ℕ, (u, v) ∈ played_on_day1 ∧ (v, w) ∈ played_on_day2)

theorem select_10_teams {teams : Finset ℕ}
  (h : teams.card = 20)
  {played_on_day1 played_on_day2 : Finset (ℕ × ℕ)}
  (h1 : ∀ ⦃u v : ℕ⦄, (u, v) ∈ played_on_day1 → u ∈ teams ∧ v ∈ teams)
  (h2 : ∀ ⦃u v : ℕ⦄, (u, v) ∈ played_on_day2 → u ∈ teams ∧ v ∈ teams)
  (h3 : ∀ x ∈ teams, ∃ u w, (u, x) ∈ played_on_day1 ∧ (x, w) ∈ played_on_day2) :
  ∃ S : Finset ℕ, S.card = 10 ∧ (∀ ⦃x y⦄, x ∈ S → y ∈ S → x ≠ y → (¬((x, y) ∈ played_on_day1) ∧ ¬((x, y) ∈ played_on_day2))) :=
by
  sorry

end select_10_teams_l131_131599


namespace original_volume_of_ice_l131_131378

variable (V : ℝ) 

theorem original_volume_of_ice (h1 : V * (1 / 4) * (1 / 4) = 0.25) : V = 4 :=
  sorry

end original_volume_of_ice_l131_131378


namespace smallest_n_sum_gt_10_pow_5_l131_131847

theorem smallest_n_sum_gt_10_pow_5 :
  ∃ (n : ℕ), (n ≥ 142) ∧ (5 * n^2 + 4 * n ≥ 100000) :=
by
  use 142
  sorry

end smallest_n_sum_gt_10_pow_5_l131_131847


namespace blue_eyed_kitten_percentage_is_correct_l131_131057

def total_blue_eyed_kittens : ℕ := 5 + 6 + 4 + 7 + 3

def total_kittens : ℕ := 12 + 16 + 11 + 19 + 12

def percentage_blue_eyed_kittens (blue : ℕ) (total : ℕ) : ℚ := (blue : ℚ) / (total : ℚ) * 100

theorem blue_eyed_kitten_percentage_is_correct :
  percentage_blue_eyed_kittens total_blue_eyed_kittens total_kittens = 35.71 := sorry

end blue_eyed_kitten_percentage_is_correct_l131_131057


namespace chocolate_ticket_fraction_l131_131629

theorem chocolate_ticket_fraction (box_cost : ℝ) (ticket_count_per_free_box : ℕ) (ticket_count_included : ℕ) :
  ticket_count_per_free_box = 10 →
  ticket_count_included = 1 →
  (1 / 9 : ℝ) * box_cost =
  box_cost / ticket_count_per_free_box + box_cost / (ticket_count_per_free_box - ticket_count_included + 1) :=
by 
  intros h1 h2 
  have h : ticket_count_per_free_box = 10 := h1 
  have h' : ticket_count_included = 1 := h2 
  sorry

end chocolate_ticket_fraction_l131_131629


namespace retailer_selling_price_l131_131405

theorem retailer_selling_price
  (cost_price_manufacturer : ℝ)
  (manufacturer_profit_rate : ℝ)
  (wholesaler_profit_rate : ℝ)
  (retailer_profit_rate : ℝ)
  (manufacturer_selling_price : ℝ)
  (wholesaler_selling_price : ℝ)
  (retailer_selling_price : ℝ)
  (h1 : cost_price_manufacturer = 17)
  (h2 : manufacturer_profit_rate = 0.18)
  (h3 : wholesaler_profit_rate = 0.20)
  (h4 : retailer_profit_rate = 0.25)
  (h5 : manufacturer_selling_price = cost_price_manufacturer + (manufacturer_profit_rate * cost_price_manufacturer))
  (h6 : wholesaler_selling_price = manufacturer_selling_price + (wholesaler_profit_rate * manufacturer_selling_price))
  (h7 : retailer_selling_price = wholesaler_selling_price + (retailer_profit_rate * wholesaler_selling_price)) :
  retailer_selling_price = 30.09 :=
by {
  sorry
}

end retailer_selling_price_l131_131405


namespace students_drawn_from_grade10_l131_131641

-- Define the initial conditions
def total_students_grade12 : ℕ := 750
def total_students_grade11 : ℕ := 850
def total_students_grade10 : ℕ := 900
def sample_size : ℕ := 50

-- Prove the number of students drawn from grade 10 is 18
theorem students_drawn_from_grade10 : 
  total_students_grade12 = 750 ∧
  total_students_grade11 = 850 ∧
  total_students_grade10 = 900 ∧
  sample_size = 50 →
  (sample_size * total_students_grade10 / 
  (total_students_grade12 + total_students_grade11 + total_students_grade10) = 18) :=
by
  sorry

end students_drawn_from_grade10_l131_131641


namespace min_quadratic_expression_l131_131899

theorem min_quadratic_expression:
  ∀ x y : ℝ, 2 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ 3 :=
by
  sorry

end min_quadratic_expression_l131_131899


namespace work_together_10_days_l131_131621

noncomputable def rate_A (W : ℝ) : ℝ := W / 20
noncomputable def rate_B (W : ℝ) : ℝ := W / 20

theorem work_together_10_days (W : ℝ) (hW : W > 0) :
  let A := rate_A W
  let B := rate_B W
  let combined_rate := A + B
  W / combined_rate = 10 :=
by
  sorry

end work_together_10_days_l131_131621


namespace female_computer_literacy_l131_131909

variable (E F C M CM CF : ℕ)

theorem female_computer_literacy (hE : E = 1200) 
                                (hF : F = 720) 
                                (hC : C = 744) 
                                (hM : M = 480) 
                                (hCM : CM = 240) 
                                (hCF : CF = C - CM) : 
                                CF = 504 :=
by {
  sorry
}

end female_computer_literacy_l131_131909


namespace infinitely_many_n_gt_sqrt_two_l131_131365

/-- A sequence of positive integers indexed by natural numbers. -/
def a (n : ℕ) : ℕ := sorry

/-- Main theorem stating there are infinitely many n such that 1 + a_n > a_{n-1} * root n of 2. -/
theorem infinitely_many_n_gt_sqrt_two :
  ∀ (a : ℕ → ℕ), (∀ n, a n > 0) → ∃ᶠ n in at_top, 1 + a n > a (n - 1) * (2 : ℝ) ^ (1 / n : ℝ) :=
by {
  sorry
}

end infinitely_many_n_gt_sqrt_two_l131_131365


namespace new_price_after_increase_l131_131161

def original_price (y : ℝ) : Prop := 2 * y = 540

theorem new_price_after_increase (y : ℝ) (h : original_price y) : 1.3 * y = 351 :=
by sorry

end new_price_after_increase_l131_131161


namespace inequality_proof_l131_131201

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a / b + b / c + c / a) ^ 2 ≥ 3 * (a / c + c / b + b / a) :=
  sorry

end inequality_proof_l131_131201


namespace negation_of_prop_p_l131_131849

theorem negation_of_prop_p (p : Prop) (h : ∀ x: ℝ, 0 < x → x > Real.log x) :
  (¬ (∀ x: ℝ, 0 < x → x > Real.log x)) ↔ (∃ x_0: ℝ, 0 < x_0 ∧ x_0 ≤ Real.log x_0) :=
by sorry

end negation_of_prop_p_l131_131849


namespace product_of_roots_l131_131052

theorem product_of_roots : 
  ∀ (r1 r2 r3 : ℝ), (2 * r1 * r2 * r3 - 3 * (r1 * r2 + r2 * r3 + r3 * r1) - 15 * (r1 + r2 + r3) + 35 = 0) → 
  (r1 * r2 * r3 = -35 / 2) :=
by
  sorry

end product_of_roots_l131_131052


namespace students_catching_up_on_homework_l131_131314

def total_students : ℕ := 24
def silent_reading_students : ℕ := total_students / 2
def board_games_students : ℕ := total_students / 3

theorem students_catching_up_on_homework : 
  total_students - (silent_reading_students + board_games_students) = 4 := by
  sorry

end students_catching_up_on_homework_l131_131314


namespace faye_money_left_is_30_l131_131282

-- Definitions and conditions
def initial_money : ℝ := 20
def mother_gave (initial : ℝ) : ℝ := 2 * initial
def cost_of_cupcakes : ℝ := 10 * 1.5
def cost_of_cookies : ℝ := 5 * 3

-- Calculate the total money Faye has left
def total_money_left (initial : ℝ) (mother_gave_ : ℝ) (cost_cupcakes : ℝ) (cost_cookies : ℝ) : ℝ :=
  initial + mother_gave_ - (cost_cupcakes + cost_cookies)

-- Theorem stating the money left
theorem faye_money_left_is_30 :
  total_money_left initial_money (mother_gave initial_money) cost_of_cupcakes cost_of_cookies = 30 :=
by sorry

end faye_money_left_is_30_l131_131282


namespace basketball_match_scores_l131_131880

theorem basketball_match_scores :
  ∃ (a r b d : ℝ), (a = b) ∧ (a * (1 + r + r^2 + r^3) < 120) ∧
  (4 * b + 6 * d < 120) ∧ ((a * (1 + r + r^2 + r^3) - (4 * b + 6 * d)) = 3) ∧
  a + b + (a * r + (b + d)) = 35.5 :=
sorry

end basketball_match_scores_l131_131880


namespace sum_of_two_greatest_values_of_b_sum_of_two_greatest_values_l131_131922

theorem sum_of_two_greatest_values_of_b (b : Real) 
  (h : 4 * b ^ 4 - 41 * b ^ 2 + 100 = 0) :
  b = 2.5 ∨ b = 2 ∨ b = -2.5 ∨ b = -2 :=
sorry

theorem sum_of_two_greatest_values (b1 b2 : Real)
  (hb1 : 4 * b1 ^ 4 - 41 * b1 ^ 2 + 100 = 0)
  (hb2 : 4 * b2 ^ 4 - 41 * b2 ^ 2 + 100 = 0) :
  b1 = 2.5 → b2 = 2 → b1 + b2 = 4.5 :=
sorry

end sum_of_two_greatest_values_of_b_sum_of_two_greatest_values_l131_131922


namespace robot_cost_l131_131307

theorem robot_cost (num_friends : ℕ) (total_tax change start_money : ℝ) (h_friends : num_friends = 7) (h_tax : total_tax = 7.22) (h_change : change = 11.53) (h_start : start_money = 80) :
  let spent_money := start_money - change
  let cost_robots := spent_money - total_tax
  let cost_per_robot := cost_robots / num_friends
  cost_per_robot = 8.75 :=
by
  sorry

end robot_cost_l131_131307


namespace next_shared_meeting_day_l131_131471

-- Definitions based on the conditions:
def dramaClubMeetingInterval : ℕ := 3
def choirMeetingInterval : ℕ := 5
def debateTeamMeetingInterval : ℕ := 7

-- Statement to prove:
theorem next_shared_meeting_day : Nat.lcm (Nat.lcm dramaClubMeetingInterval choirMeetingInterval) debateTeamMeetingInterval = 105 := by
  sorry

end next_shared_meeting_day_l131_131471


namespace jess_father_first_round_l131_131445

theorem jess_father_first_round (initial_blocks : ℕ)
  (players : ℕ)
  (blocks_before_jess_turn : ℕ)
  (jess_falls_tower_round : ℕ)
  (h1 : initial_blocks = 54)
  (h2 : players = 5)
  (h3 : blocks_before_jess_turn = 28)
  (h4 : ∀ rounds : ℕ, rounds * players ≥ 26 → jess_falls_tower_round = rounds + 1) :
  jess_falls_tower_round = 6 := 
by
  sorry

end jess_father_first_round_l131_131445


namespace zero_in_interval_l131_131297

def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem zero_in_interval :
  (f 0 < 0) → (f 0.5 > 0) → (f 0.25 < 0) → ∃ x, 0.25 < x ∧ x < 0.5 ∧ f x = 0 :=
by
  intro h0 h05 h025
  -- This is just the statement; the proof is not required as per instructions
  sorry

end zero_in_interval_l131_131297


namespace probability_f_leq_zero_l131_131577

noncomputable def f (k x : ℝ) : ℝ := k * x - 1

theorem probability_f_leq_zero : 
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1) →
  (∀ k ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f k x ≤ 0) →
  (∃ k ∈ Set.Icc (-2 : ℝ) (1 : ℝ), f k x ≤ 0) →
  ((1 - (-2)) / (2 - (-2)) = 3 / 4) :=
by sorry

end probability_f_leq_zero_l131_131577


namespace parabola_line_dot_product_l131_131958

theorem parabola_line_dot_product (k x1 x2 y1 y2 : ℝ) 
  (h_line: ∀ x, y = k * x + 2)
  (h_parabola: ∀ x, y = (1 / 4) * x ^ 2) 
  (h_A: y1 = k * x1 + 2 ∧ y1 = (1 / 4) * x1 ^ 2)
  (h_B: y2 = k * x2 + 2 ∧ y2 = (1 / 4) * x2 ^ 2) :
  x1 * x2 + y1 * y2 = -4 := 
sorry

end parabola_line_dot_product_l131_131958


namespace expression_S_max_value_S_l131_131092

section
variable (x t : ℝ)
def f (x : ℝ) := -3 * x^2 + 6 * x

-- Define the integral expression for S(t)
noncomputable def S (t : ℝ) := ∫ x in t..(t + 1), f x

-- Assert the expression for S(t)
theorem expression_S (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 2) :
  S t = -3 * t^2 + 3 * t + 2 :=
by
  sorry

-- Assert the maximum value of S(t)
theorem max_value_S :
  ∀ t, (0 ≤ t ∧ t ≤ 2) → S t ≤ 5 / 4 :=
by
  sorry

end

end expression_S_max_value_S_l131_131092


namespace increasing_sequence_a1_range_l131_131227

theorem increasing_sequence_a1_range
  (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = (4 * a n - 2) / (a n + 1))
  (strictly_increasing : ∀ n, a (n + 1) > a n) :
  1 < a 1 ∧ a 1 < 2 :=
sorry

end increasing_sequence_a1_range_l131_131227


namespace common_chord_length_l131_131343

/-- Two circles intersect such that each passes through the other's center.
Prove that the length of their common chord is 8√3 cm. -/
theorem common_chord_length (r : ℝ) (h : r = 8) :
  let chord_length := 2 * (r * (Real.sqrt 3 / 2))
  chord_length = 8 * Real.sqrt 3 := by
  sorry

end common_chord_length_l131_131343


namespace find_a_5_in_arithmetic_sequence_l131_131845

variable {a : ℕ → ℕ}

def arithmetic_sequence (a : ℕ → ℕ) (a1 d : ℕ) : Prop :=
  a 1 = a1 ∧ ∀ n, a (n + 1) = a n + d

theorem find_a_5_in_arithmetic_sequence (h : arithmetic_sequence a 1 2) : a 5 = 9 :=
sorry

end find_a_5_in_arithmetic_sequence_l131_131845


namespace expected_value_of_12_sided_die_is_6_5_l131_131467

noncomputable def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  n * (a + l) / 2

noncomputable def expected_value_12_sided_die : ℚ :=
  (sum_arithmetic_series 12 1 12 : ℚ) / 12

theorem expected_value_of_12_sided_die_is_6_5 :
  expected_value_12_sided_die = 6.5 :=
by
  sorry

end expected_value_of_12_sided_die_is_6_5_l131_131467


namespace determine_x_l131_131832

variables {m n x : ℝ}
variable (k : ℝ)
variable (Hmn : m ≠ 0 ∧ n ≠ 0)
variable (Hk : k = 5 * (m^2 - n^2))

theorem determine_x (H : (x + 2 * m)^2 - (x - 3 * n)^2 = k) : 
  x = (5 * m^2 - 9 * n^2) / (4 * m + 6 * n) := by
  sorry

end determine_x_l131_131832


namespace max_empty_squares_l131_131368

theorem max_empty_squares (board_size : ℕ) (total_cells : ℕ) 
  (initial_cockroaches : ℕ) (adjacent : ℕ → ℕ → Prop) 
  (different : ℕ → ℕ → Prop) :
  board_size = 8 → total_cells = 64 → initial_cockroaches = 2 →
  (∀ s : ℕ, s < total_cells → ∃ s1 s2 : ℕ, adjacent s s1 ∧ 
              adjacent s s2 ∧ 
              different s1 s2) →
  ∃ max_empty_cells : ℕ, max_empty_cells = 24 :=
by
  intros h_board_size h_total_cells h_initial_cockroaches h_moves
  sorry

end max_empty_squares_l131_131368


namespace unique_line_equal_intercepts_l131_131804

-- Definitions of the point and line
structure Point where
  x : ℝ
  y : ℝ

def passesThrough (L : ℝ → ℝ) (P : Point) : Prop :=
  L P.x = P.y

noncomputable def hasEqualIntercepts (L : ℝ → ℝ) : Prop :=
  ∃ a, L 0 = a ∧ L a = 0

-- The main theorem statement
theorem unique_line_equal_intercepts (L : ℝ → ℝ) (P : Point) (hP : P.x = 2 ∧ P.y = 1) (h_equal_intercepts : hasEqualIntercepts L) :
  ∃! (L : ℝ → ℝ), passesThrough L P ∧ hasEqualIntercepts L :=
sorry

end unique_line_equal_intercepts_l131_131804


namespace jordan_travel_distance_heavy_traffic_l131_131340

theorem jordan_travel_distance_heavy_traffic (x : ℝ) (h1 : x / 20 + x / 10 + x / 6 = 7 / 6) : 
  x = 3.7 :=
by
  sorry

end jordan_travel_distance_heavy_traffic_l131_131340


namespace total_sum_lent_l131_131350

-- Conditions
def interest_equal (x y : ℕ) : Prop :=
  (x * 3 * 8) / 100 = (y * 5 * 3) / 100

def second_sum : ℕ := 1704

-- Assertion
theorem total_sum_lent : ∃ x : ℕ, interest_equal x second_sum ∧ (x + second_sum = 2769) :=
  by
  -- Placeholder proof
  sorry

end total_sum_lent_l131_131350


namespace gcd_of_three_numbers_l131_131308

-- Define the given numbers
def a := 72
def b := 120
def c := 168

-- Define the GCD function and prove the required statement
theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd a b) c = 24 := by
  -- Intermediate steps and their justifications would go here in the proof, but we are putting sorry
  sorry

end gcd_of_three_numbers_l131_131308


namespace crows_cannot_be_on_same_tree_l131_131292

theorem crows_cannot_be_on_same_tree :
  (∀ (trees : ℕ) (crows : ℕ),
   trees = 22 ∧ crows = 22 →
   (∀ (positions : ℕ → ℕ),
    (∀ i, 1 ≤ positions i ∧ positions i ≤ 2) →
    ∀ (move : (ℕ → ℕ) → (ℕ → ℕ)),
    (∀ (pos : ℕ → ℕ) (i : ℕ),
     move pos i = pos i + positions (i + 1) ∨ move pos i = pos i - positions (i + 1)) →
    (∀ (pos : ℕ → ℕ) (i : ℕ),
     pos i % trees = (move pos i) % trees) →
    ¬ (∃ (final_pos : ℕ → ℕ),
      (∀ i, final_pos i = 0 ∨ final_pos i = 22) ∧
      (∀ i j, final_pos i = final_pos j)
    )
  )
) :=
sorry

end crows_cannot_be_on_same_tree_l131_131292


namespace smallest_number_is_C_l131_131901

-- Define the conditions
def A := 18 + 38
def B := A - 26
def C := B / 3

-- Proof statement: C is the smallest number among A, B, and C
theorem smallest_number_is_C : C = min A (min B C) :=
by
  sorry

end smallest_number_is_C_l131_131901


namespace distance_they_both_run_l131_131338

theorem distance_they_both_run
  (D : ℝ)
  (A_time : D / 28 = A_speed)
  (B_time : D / 32 = B_speed)
  (A_beats_B : A_speed * 28 = B_speed * 28 + 16) :
  D = 128 := 
sorry

end distance_they_both_run_l131_131338


namespace euclidean_division_l131_131771

theorem euclidean_division (a b : ℕ) (hb : b ≠ 0) : ∃ q r : ℤ, 0 ≤ r ∧ r < b ∧ a = b * q + r :=
by sorry

end euclidean_division_l131_131771


namespace tent_ratio_l131_131062

-- Define the relevant variables
variables (N E S C T : ℕ)

-- State the conditions
def conditions : Prop :=
  N = 100 ∧
  E = 2 * N ∧
  S = 200 ∧
  T = 900 ∧
  N + E + S + C = T

-- State the theorem to prove the ratio
theorem tent_ratio (h : conditions N E S C T) : C = 4 * N :=
by sorry

end tent_ratio_l131_131062


namespace sum_of_powers_of_i_l131_131964

-- Let i be the imaginary unit
def i : ℂ := Complex.I

theorem sum_of_powers_of_i : (1 + i + i^2 + i^3 + i^4 + i^5 + i^6 + i^7 + i^8 + i^9 + i^10) = i := by
  sorry

end sum_of_powers_of_i_l131_131964


namespace computer_sale_price_percent_l131_131724

theorem computer_sale_price_percent (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (discount3 : ℝ) :
  original_price = 500 ∧ discount1 = 0.25 ∧ discount2 = 0.10 ∧ discount3 = 0.05 →
  (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)) / original_price * 100 = 64.13 :=
by
  intro h
  sorry

end computer_sale_price_percent_l131_131724


namespace sum_abs_values_l131_131370

theorem sum_abs_values (a b : ℝ) (h₁ : abs a = 4) (h₂ : abs b = 7) (h₃ : a < b) : a + b = 3 ∨ a + b = 11 :=
by
  sorry

end sum_abs_values_l131_131370


namespace intersection_of_sets_l131_131623

def setA : Set ℝ := {x | (x - 2) / x ≤ 0}
def setB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def setC : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets : setA ∩ setB = setC :=
by
  sorry

end intersection_of_sets_l131_131623


namespace meal_cost_is_correct_l131_131410

def samosa_quantity : ℕ := 3
def samosa_price : ℝ := 2
def pakora_quantity : ℕ := 4
def pakora_price : ℝ := 3
def mango_lassi_quantity : ℕ := 1
def mango_lassi_price : ℝ := 2
def biryani_quantity : ℕ := 2
def biryani_price : ℝ := 5.5
def naan_quantity : ℕ := 1
def naan_price : ℝ := 1.5

def tip_rate : ℝ := 0.18
def sales_tax_rate : ℝ := 0.07

noncomputable def total_meal_cost : ℝ :=
  let subtotal := (samosa_quantity * samosa_price) + (pakora_quantity * pakora_price) +
                  (mango_lassi_quantity * mango_lassi_price) + (biryani_quantity * biryani_price) +
                  (naan_quantity * naan_price)
  let sales_tax := subtotal * sales_tax_rate
  let total_before_tip := subtotal + sales_tax
  let tip := total_before_tip * tip_rate
  total_before_tip + tip

theorem meal_cost_is_correct : total_meal_cost = 41.04 := by
  sorry

end meal_cost_is_correct_l131_131410


namespace mcq_options_l131_131820

theorem mcq_options :
  ∃ n : ℕ, (1/n : ℝ) * (1/2) * (1/2) = (1/12) ∧ n = 3 :=
by
  sorry

end mcq_options_l131_131820


namespace calculation_division_l131_131537

theorem calculation_division :
  ((27 * 0.92 * 0.85) / (23 * 1.7 * 1.8)) = 0.3 :=
by
  sorry

end calculation_division_l131_131537


namespace current_short_trees_l131_131683

theorem current_short_trees (S : ℕ) (S_planted : ℕ) (S_total : ℕ) 
  (H1 : S_planted = 105) 
  (H2 : S_total = 217) 
  (H3 : S + S_planted = S_total) :
  S = 112 :=
by
  sorry

end current_short_trees_l131_131683


namespace sum_of_distinct_integers_eq_36_l131_131294

theorem sum_of_distinct_integers_eq_36
  (p q r s t : ℤ)
  (hpq : p ≠ q) (hpr : p ≠ r) (hps : p ≠ s) (hpt : p ≠ t)
  (hqr : q ≠ r) (hqs : q ≠ s) (hqt : q ≠ t)
  (hrs : r ≠ s) (hrt : r ≠ t)
  (hst : s ≠ t)
  (h : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 80) :
  p + q + r + s + t = 36 :=
by
  sorry

end sum_of_distinct_integers_eq_36_l131_131294


namespace intersecting_rectangles_area_l131_131920

-- Define the dimensions of the rectangles
def rect1_length : ℝ := 12
def rect1_width : ℝ := 4
def rect2_length : ℝ := 7
def rect2_width : ℝ := 5

-- Define the areas of the individual rectangles
def area_rect1 : ℝ := rect1_length * rect1_width
def area_rect2 : ℝ := rect2_length * rect2_width

-- Assume overlapping region area
def area_overlap : ℝ := rect1_width * rect2_width

-- Define the total shaded area
def shaded_area : ℝ := area_rect1 + area_rect2 - area_overlap

-- Prove the shaded area is 63 square units
theorem intersecting_rectangles_area : shaded_area = 63 :=
by 
  -- Insert proof steps here, we only provide the theorem statement and leave the proof unfinished
  sorry

end intersecting_rectangles_area_l131_131920


namespace compare_logs_and_exp_l131_131690

theorem compare_logs_and_exp :
  let a := Real.log 3 / Real.log 5
  let b := Real.log 8 / Real.log 13
  let c := Real.exp (-1 / 2)
  c < a ∧ a < b := 
sorry

end compare_logs_and_exp_l131_131690


namespace candies_count_l131_131454

variable (m_and_m : Nat) (starbursts : Nat)
variable (ratio_m_and_m_to_starbursts : Nat → Nat → Prop)

-- Definition of the ratio condition
def ratio_condition : Prop :=
  ∃ (k : Nat), (m_and_m = 7 * k) ∧ (starbursts = 4 * k)

-- The main theorem to prove
theorem candies_count (h : m_and_m = 56) (r : ratio_condition m_and_m starbursts) : starbursts = 32 :=
  by
  sorry

end candies_count_l131_131454


namespace product_of_mixed_numbers_l131_131170

theorem product_of_mixed_numbers :
  let fraction1 := (13 : ℚ) / 6
  let fraction2 := (29 : ℚ) / 9
  (fraction1 * fraction2) = 377 / 54 := 
by
  sorry

end product_of_mixed_numbers_l131_131170


namespace find_prime_triplet_l131_131711

def is_geometric_sequence (x y z : ℕ) : Prop :=
  (y^2 = x * z)

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_prime_triplet :
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧
  a < b ∧ b < c ∧ c < 100 ∧
  is_geometric_sequence (a + 1) (b + 1) (c + 1) ∧
  (a = 17 ∧ b = 23 ∧ c = 31) :=
by
  sorry

end find_prime_triplet_l131_131711


namespace original_denominator_l131_131978

theorem original_denominator (d : ℤ) : 
  (∀ n : ℤ, n = 3 → (n + 8) / (d + 8) = 1 / 3) → d = 25 :=
by
  intro h
  specialize h 3 rfl
  sorry

end original_denominator_l131_131978


namespace minimum_greeting_pairs_l131_131134

def minimum_mutual_greetings (n: ℕ) (g: ℕ) : ℕ :=
  (n * g - (n * (n - 1)) / 2)

theorem minimum_greeting_pairs :
  minimum_mutual_greetings 400 200 = 200 :=
by 
  sorry

end minimum_greeting_pairs_l131_131134


namespace value_of_a_ab_b_l131_131924

-- Define conditions
variables {a b : ℝ} (h1 : a * b = 1) (h2 : b = a + 2)

-- The proof problem
theorem value_of_a_ab_b : a - a * b - b = -3 :=
by
  sorry

end value_of_a_ab_b_l131_131924


namespace no_same_last_four_digits_of_powers_of_five_and_six_l131_131281

theorem no_same_last_four_digits_of_powers_of_five_and_six : 
  ¬ ∃ (n m : ℕ), 0 < n ∧ 0 < m ∧ (5 ^ n % 10000 = 6 ^ m % 10000) := 
by 
  sorry

end no_same_last_four_digits_of_powers_of_five_and_six_l131_131281


namespace garden_breadth_l131_131501

-- Problem statement conditions
def perimeter : ℝ := 600
def length : ℝ := 205

-- Translate the problem into Lean:
theorem garden_breadth (breadth : ℝ) (h1 : 2 * (length + breadth) = perimeter) : breadth = 95 := 
by sorry

end garden_breadth_l131_131501


namespace least_value_m_n_l131_131125

theorem least_value_m_n :
  ∃ m n : ℕ, (m > 0 ∧ n > 0) ∧
            (Nat.gcd (m + n) 231 = 1) ∧
            (n^n ∣ m^m) ∧
            ¬ (m % n = 0) ∧
            m + n = 377 :=
by 
  sorry

end least_value_m_n_l131_131125


namespace boat_speed_in_still_water_l131_131740

def speed_of_stream : ℝ := 8
def downstream_distance : ℝ := 64
def upstream_distance : ℝ := 32

theorem boat_speed_in_still_water (x : ℝ) (t : ℝ) 
  (HS_downstream : t = downstream_distance / (x + speed_of_stream)) 
  (HS_upstream : t = upstream_distance / (x - speed_of_stream)) :
  x = 24 := by
  sorry

end boat_speed_in_still_water_l131_131740


namespace cost_B_solution_l131_131129

variable (cost_B : ℝ)

/-- The number of items of type A that can be purchased with 1000 yuan 
is equal to the number of items of type B that can be purchased with 800 yuan. -/
def items_purchased_equality (cost_B : ℝ) : Prop :=
  1000 / (cost_B + 10) = 800 / cost_B

/-- The cost of each item of type A is 10 yuan more than the cost of each item of type B. -/
def cost_difference (cost_B : ℝ) : Prop :=
  cost_B + 10 - cost_B = 10

/-- The cost of each item of type B is 40 yuan. -/
theorem cost_B_solution (h1: items_purchased_equality cost_B) (h2: cost_difference cost_B) :
  cost_B = 40 := by
sorry

end cost_B_solution_l131_131129


namespace division_correct_result_l131_131162

theorem division_correct_result (x : ℝ) (h : 8 * x = 56) : 42 / x = 6 := by
  sorry

end division_correct_result_l131_131162


namespace silas_payment_ratio_l131_131173

theorem silas_payment_ratio (total_bill : ℕ) (tip_rate : ℝ) (friend_payment : ℕ) (S : ℕ) :
  total_bill = 150 →
  tip_rate = 0.10 →
  friend_payment = 18 →
  (S + 5 * friend_payment = total_bill + total_bill * tip_rate) →
  (S : ℝ) / total_bill = 1 / 2 :=
by
  intros h_total_bill h_tip_rate h_friend_payment h_budget_eq
  sorry

end silas_payment_ratio_l131_131173


namespace mean_proportional_of_segments_l131_131280

theorem mean_proportional_of_segments (a b c : ℝ) (a_val : a = 2) (b_val : b = 6) :
  c = 2 * Real.sqrt 3 ↔ c*c = a * b := by
  sorry

end mean_proportional_of_segments_l131_131280


namespace find_k_value_l131_131994

theorem find_k_value (k : ℝ) :
  (∃ (x y : ℝ), x + k * y = 0 ∧ 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0) ↔ k = -1/2 := 
by
  sorry

end find_k_value_l131_131994


namespace second_race_distance_remaining_l131_131377

theorem second_race_distance_remaining
  (race_distance : ℕ)
  (A_finish_time : ℕ)
  (B_remaining_distance : ℕ)
  (A_start_behind : ℕ)
  (A_speed : ℝ)
  (B_speed : ℝ)
  (A_distance_second_race : ℕ)
  (B_distance_second_race : ℝ)
  (v_ratio : ℝ)
  (B_remaining_second_race : ℝ) :
  race_distance = 10000 →
  A_finish_time = 50 →
  B_remaining_distance = 500 →
  A_start_behind = 500 →
  A_speed = race_distance / A_finish_time →
  B_speed = (race_distance - B_remaining_distance) / A_finish_time →
  v_ratio = A_speed / B_speed →
  v_ratio = 20 / 19 →
  A_distance_second_race = race_distance + A_start_behind →
  B_distance_second_race = B_speed * (A_distance_second_race / A_speed) →
  B_remaining_second_race = race_distance - B_distance_second_race →
  B_remaining_second_race = 25 := 
by
  sorry

end second_race_distance_remaining_l131_131377


namespace motorcyclist_average_speed_l131_131388

theorem motorcyclist_average_speed :
  let distance_ab := 120
  let speed_ab := 45
  let distance_bc := 130
  let speed_bc := 60
  let distance_cd := 150
  let speed_cd := 50
  let time_ab := distance_ab / speed_ab
  let time_bc := distance_bc / speed_bc
  let time_cd := distance_cd / speed_cd
  (time_ab = time_bc + 2)
  → (time_cd = time_ab / 2)
  → avg_speed = (distance_ab + distance_bc + distance_cd) / (time_ab + time_bc + time_cd)
  → avg_speed = 2400 / 47 := sorry

end motorcyclist_average_speed_l131_131388


namespace xiaoning_pe_comprehensive_score_l131_131221

def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.7
def midterm_score : ℝ := 80
def final_score : ℝ := 90

theorem xiaoning_pe_comprehensive_score : midterm_score * midterm_weight + final_score * final_weight = 87 :=
by
  sorry

end xiaoning_pe_comprehensive_score_l131_131221


namespace range_of_y_l131_131305

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : Int.ceil y * Int.floor y = 72) : 
  -9 < y ∧ y < -8 :=
sorry

end range_of_y_l131_131305


namespace value_of_k_l131_131640

theorem value_of_k (k : ℤ) : 
  (∀ x : ℤ, (x + k) * (x - 4) = x^2 - 4 * x + k * x - 4 * k ∧ 
  (k - 4) * x = 0) → k = 4 := 
by 
  sorry

end value_of_k_l131_131640


namespace find_n_l131_131915

-- Declaring the necessary context and parameters.
variable (n : ℕ)

-- Defining the condition described in the problem.
def reposting_equation (n : ℕ) : Prop := 1 + n + n^2 = 111

-- Stating the theorem to prove that for n = 10, the reposting equation holds.
theorem find_n : ∃ (n : ℕ), reposting_equation n ∧ n = 10 :=
by
  use 10
  unfold reposting_equation
  sorry

end find_n_l131_131915


namespace min_val_x_2y_l131_131295

noncomputable def min_x_2y (x y : ℝ) : ℝ :=
  x + 2 * y

theorem min_val_x_2y : 
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (1 / (x + 2) + 1 / (y + 2) = 1 / 3) → 
  min_x_2y x y ≥ 3 + 6 * Real.sqrt 2 :=
by
  intros x y x_pos y_pos eqn
  sorry

end min_val_x_2y_l131_131295


namespace right_regular_prism_impossible_sets_l131_131812

-- Define a function to check if a given set of numbers {x, y, z} forms an invalid right regular prism
def not_possible (x y z : ℕ) : Prop := (x^2 + y^2 ≤ z^2)

-- Define individual propositions for the given sets of numbers
def set_a : Prop := not_possible 3 4 6
def set_b : Prop := not_possible 5 5 8
def set_e : Prop := not_possible 7 8 12

-- Define our overall proposition that these sets cannot be the lengths of the external diagonals of a right regular prism
theorem right_regular_prism_impossible_sets : 
  set_a ∧ set_b ∧ set_e :=
by
  -- Proof is omitted
  sorry

end right_regular_prism_impossible_sets_l131_131812


namespace sum_of_squares_of_roots_of_quadratic_l131_131703

theorem sum_of_squares_of_roots_of_quadratic :
  ( ∃ x1 x2 : ℝ, x1^2 - 3 * x1 - 1 = 0 ∧ x2^2 - 3 * x2 - 1 = 0 ∧ x1 ≠ x2) →
  x1^2 + x2^2 = 11 :=
by
  /- Proof goes here -/
  sorry

end sum_of_squares_of_roots_of_quadratic_l131_131703


namespace number_of_real_roots_l131_131150

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then Real.exp x else -x^2 + 2.5 * x

theorem number_of_real_roots : ∃! x, f x = 0.5 * x + 1 :=
sorry

end number_of_real_roots_l131_131150


namespace concentric_spheres_volume_l131_131533

theorem concentric_spheres_volume :
  let r1 := 4
  let r2 := 7
  let r3 := 10
  let volume (r : ℝ) := (4/3) * Real.pi * r^3
  volume r3 - volume r2 = 876 * Real.pi := 
by
  let r1 := 4
  let r2 := 7
  let r3 := 10
  let volume (r : ℝ) := (4/3) * Real.pi * r^3
  show volume r3 - volume r2 = 876 * Real.pi
  sorry

end concentric_spheres_volume_l131_131533


namespace no_real_solution_abs_eq_quadratic_l131_131109

theorem no_real_solution_abs_eq_quadratic (x : ℝ) : abs (2 * x - 6) ≠ x^2 - x + 2 := by
  sorry

end no_real_solution_abs_eq_quadratic_l131_131109


namespace m_range_l131_131267

/-- Given a point (x, y) on the circle x^2 + (y - 1)^2 = 2, show that the real number m,
such that x + y + m ≥ 0, must satisfy m ≥ 1. -/
theorem m_range (x y m : ℝ) (h₁ : x^2 + (y - 1)^2 = 2) (h₂ : x + y + m ≥ 0) : m ≥ 1 :=
sorry

end m_range_l131_131267


namespace hari_joins_l131_131067

theorem hari_joins {x : ℕ} :
  let praveen_start := 3500
  let hari_start := 9000
  let total_months := 12
  (praveen_start * total_months) * 3 = (hari_start * (total_months - x)) * 2
  → x = 5 :=
by
  intros
  sorry

end hari_joins_l131_131067


namespace raj_is_older_than_ravi_l131_131353

theorem raj_is_older_than_ravi
  (R V H L x : ℕ)
  (h1 : R = V + x)
  (h2 : H = V - 2)
  (h3 : R = 3 * L)
  (h4 : H * 2 = 3 * L)
  (h5 : 20 = (4 * H) / 3) :
  x = 13 :=
by
  sorry

end raj_is_older_than_ravi_l131_131353


namespace total_calories_burned_l131_131284

def base_distance : ℝ := 15
def records : List ℝ := [0.1, -0.8, 0.9, 16.5 - base_distance, 2.0, -1.5, 14.1 - base_distance, 1.0, 0.8, -1.1]
def calorie_burn_rate : ℝ := 20

theorem total_calories_burned :
  (base_distance * 10 + (List.sum records)) * calorie_burn_rate = 3040 :=
by
  sorry

end total_calories_burned_l131_131284


namespace G_greater_F_l131_131151

theorem G_greater_F (x : ℝ) : 
  let F := 2*x^2 - 3*x - 2
  let G := 3*x^2 - 7*x + 5
  G > F := 
sorry

end G_greater_F_l131_131151


namespace find_f_2017_l131_131421

noncomputable def f : ℝ → ℝ :=
sorry

axiom cond1 : ∀ x : ℝ, f (1 + x) + f (1 - x) = 0
axiom cond2 : ∀ x : ℝ, f (-x) = f x
axiom cond3 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = 2^x - 1

theorem find_f_2017 : f 2017 = 1 :=
by
  sorry

end find_f_2017_l131_131421


namespace cos_equivalent_l131_131567

open Real

theorem cos_equivalent (alpha : ℝ) (h : sin (π / 3 + alpha) = 1 / 3) : 
  cos (5 * π / 6 + alpha) = -1 / 3 :=
sorry

end cos_equivalent_l131_131567


namespace probability_intersecting_diagonals_l131_131199

def number_of_vertices := 10

def number_of_diagonals : ℕ := Nat.choose number_of_vertices 2 - number_of_vertices

def number_of_ways_choose_two_diagonals := Nat.choose number_of_diagonals 2

def number_of_sets_of_intersecting_diagonals : ℕ := Nat.choose number_of_vertices 4

def intersection_probability : ℚ :=
  (number_of_sets_of_intersecting_diagonals : ℚ) / (number_of_ways_choose_two_diagonals : ℚ)

theorem probability_intersecting_diagonals :
  intersection_probability = 42 / 119 :=
by
  sorry

end probability_intersecting_diagonals_l131_131199


namespace positive_integer_condition_l131_131886

theorem positive_integer_condition (p : ℕ) (hp : 0 < p) : 
  (∃ k : ℤ, k > 0 ∧ 4 * p + 17 = k * (3 * p - 8)) ↔ p = 3 :=
by {
  sorry
}

end positive_integer_condition_l131_131886


namespace f_of_g_of_2_l131_131176

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem f_of_g_of_2 : f (g 2) = 14 :=
by 
  sorry

end f_of_g_of_2_l131_131176


namespace walmart_total_sales_l131_131540

-- Define the constants for the prices
def thermometer_price : ℕ := 2
def hot_water_bottle_price : ℕ := 6

-- Define the quantities and relationships
def hot_water_bottles_sold : ℕ := 60
def thermometer_ratio : ℕ := 7
def thermometers_sold : ℕ := thermometer_ratio * hot_water_bottles_sold

-- Define the total sales for thermometers and hot-water bottles
def thermometer_sales : ℕ := thermometers_sold * thermometer_price
def hot_water_bottle_sales : ℕ := hot_water_bottles_sold * hot_water_bottle_price

-- Define the total sales amount
def total_sales : ℕ := thermometer_sales + hot_water_bottle_sales

-- Theorem statement
theorem walmart_total_sales : total_sales = 1200 := by
  sorry

end walmart_total_sales_l131_131540


namespace min_attendees_l131_131868

-- Define the constants and conditions
def writers : ℕ := 35
def min_editors : ℕ := 39
def x_max : ℕ := 26

-- Define the total number of people formula based on inclusion-exclusion principle
-- and conditions provided
def total_people (x : ℕ) : ℕ := writers + min_editors - x + 2 * x

-- Theorem to prove that the minimum number of attendees is 126
theorem min_attendees : ∃ x, x ≤ x_max ∧ total_people x = 126 :=
by
  use x_max
  sorry

end min_attendees_l131_131868


namespace peregrines_eat_30_percent_l131_131021

theorem peregrines_eat_30_percent (initial_pigeons : ℕ) (chicks_per_pigeon : ℕ) (pigeons_left : ℕ) :
  initial_pigeons = 40 →
  chicks_per_pigeon = 6 →
  pigeons_left = 196 →
  (100 * (initial_pigeons * chicks_per_pigeon + initial_pigeons - pigeons_left)) / 
  (initial_pigeons * chicks_per_pigeon + initial_pigeons) = 30 :=
by
  intros
  sorry

end peregrines_eat_30_percent_l131_131021


namespace vasya_correct_l131_131043

-- Define the condition of a convex quadrilateral
def convex_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a < 180 ∧ b < 180 ∧ c < 180 ∧ d < 180

-- Define the properties of forming two types of triangles from a quadrilateral
def can_form_two_acute_triangles (a b c d : ℝ) : Prop :=
  a < 90 ∧ b < 90 ∧ c < 90 ∧ d < 90

def can_form_two_right_triangles (a b c d : ℝ) : Prop :=
  (a = 90 ∧ b = 90) ∨ (b = 90 ∧ c = 90) ∨ (c = 90 ∧ d = 90) ∨ (d = 90 ∧ a = 90)

def can_form_two_obtuse_triangles (a b c d : ℝ) : Prop :=
  ∃ x y z w, (x > 90 ∧ y < 90 ∧ z < 90 ∧ w < 90 ∧ (x + y + z + w = 360)) ∧
             (x > 90 ∨ y > 90 ∨ z > 90 ∨ w > 90)

-- Prove that Vasya's claim is definitively correct
theorem vasya_correct (a b c d : ℝ) (h : convex_quadrilateral a b c d) :
  can_form_two_obtuse_triangles a b c d ∧
  ¬(can_form_two_acute_triangles a b c d) ∧
  ¬(can_form_two_right_triangles a b c d) ∨
  can_form_two_right_triangles a b c d ∧
  can_form_two_obtuse_triangles a b c d := sorry

end vasya_correct_l131_131043


namespace geometric_seq_sum_l131_131996

theorem geometric_seq_sum :
  ∀ (a : ℕ → ℤ) (q : ℤ), 
    (∀ n, a (n + 1) = a n * q) ∧ 
    (a 4 + a 7 = 2) ∧ 
    (a 5 * a 6 = -8) → 
    a 1 + a 10 = -7 := 
by sorry

end geometric_seq_sum_l131_131996


namespace negation_of_every_student_is_punctual_l131_131684

variable (Student : Type) (student punctual : Student → Prop)

theorem negation_of_every_student_is_punctual :
  ¬ (∀ x, student x → punctual x) ↔ ∃ x, student x ∧ ¬ punctual x := by
sorry

end negation_of_every_student_is_punctual_l131_131684


namespace range_x2y2z_range_a_inequality_l131_131321

theorem range_x2y2z {x y z : ℝ} (h : x^2 + y^2 + z^2 = 1) : 
  -3 ≤ x + 2*y + 2*z ∧ x + 2*y + 2*z ≤ 3 :=
by sorry

theorem range_a_inequality (a : ℝ) (h : ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → |a - 3| + a / 2 ≥ x + 2*y + 2*z) :
  (4 ≤ a) ∨ (a ≤ 0) :=
by sorry

end range_x2y2z_range_a_inequality_l131_131321


namespace sally_pens_initial_count_l131_131952

theorem sally_pens_initial_count :
  ∃ P : ℕ, (P - (7 * 44)) / 2 = 17 ∧ P = 342 :=
by 
  sorry

end sally_pens_initial_count_l131_131952


namespace parabola_axis_l131_131045

section
variable (x y : ℝ)

-- Condition: Defines the given parabola equation.
def parabola_eq (x y : ℝ) : Prop := x = (1 / 4) * y^2

-- The Proof Problem: Prove that the axis of this parabola is x = -1/2.
theorem parabola_axis (h : parabola_eq x y) : x = - (1 / 2) := 
sorry
end

end parabola_axis_l131_131045


namespace mechanic_hours_l131_131075

theorem mechanic_hours (h : ℕ) (labor_cost_per_hour parts_cost total_bill : ℕ) 
  (H1 : labor_cost_per_hour = 45) 
  (H2 : parts_cost = 225) 
  (H3 : total_bill = 450) 
  (H4 : labor_cost_per_hour * h + parts_cost = total_bill) : 
  h = 5 := 
by
  sorry

end mechanic_hours_l131_131075


namespace suzannes_book_pages_l131_131002

-- Conditions
def pages_read_on_monday : ℕ := 15
def pages_read_on_tuesday : ℕ := 31
def pages_left : ℕ := 18

-- Total number of pages in the book
def total_pages : ℕ := pages_read_on_monday + pages_read_on_tuesday + pages_left

-- Problem statement
theorem suzannes_book_pages : total_pages = 64 :=
by
  -- Proof is not required, only the statement
  sorry

end suzannes_book_pages_l131_131002


namespace ratio_PeteHand_to_TracyCartwheel_l131_131659

noncomputable def SusanWalkingSpeed (PeteBackwardSpeed : ℕ) : ℕ :=
  PeteBackwardSpeed / 3

noncomputable def TracyCartwheelSpeed (SusanSpeed : ℕ) : ℕ :=
  SusanSpeed * 2

def PeteHandsWalkingSpeed : ℕ := 2

def PeteBackwardWalkingSpeed : ℕ := 12

theorem ratio_PeteHand_to_TracyCartwheel :
  let SusanSpeed := SusanWalkingSpeed PeteBackwardWalkingSpeed
  let TracySpeed := TracyCartwheelSpeed SusanSpeed
  (PeteHandsWalkingSpeed : ℕ) / (TracySpeed : ℕ) = 1 / 4 :=
by
  sorry

end ratio_PeteHand_to_TracyCartwheel_l131_131659


namespace highest_price_per_shirt_l131_131070

theorem highest_price_per_shirt (x : ℝ) 
  (num_shirts : ℕ := 20)
  (total_money : ℝ := 180)
  (entrance_fee : ℝ := 5)
  (sales_tax : ℝ := 0.08)
  (whole_number: ∀ p : ℝ, ∃ n : ℕ, p = n) :
  (∀ (price_per_shirt : ℕ), price_per_shirt ≤ 8) :=
by
  sorry

end highest_price_per_shirt_l131_131070


namespace ab_value_l131_131876

theorem ab_value (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 :=
by 
  sorry

end ab_value_l131_131876


namespace quadratic_real_roots_l131_131105

theorem quadratic_real_roots (k : ℝ) (h1 : k ≠ 0) : (4 + 4 * k) ≥ 0 ↔ k ≥ -1 := 
by 
  sorry

end quadratic_real_roots_l131_131105


namespace cost_of_replaced_tomatoes_l131_131753

def original_order : ℝ := 25
def delivery_tip : ℝ := 8
def new_total : ℝ := 35
def original_tomatoes : ℝ := 0.99
def original_lettuce : ℝ := 1.00
def new_lettuce : ℝ := 1.75
def original_celery : ℝ := 1.96
def new_celery : ℝ := 2.00

def increase_in_lettuce := new_lettuce - original_lettuce
def increase_in_celery := new_celery - original_celery
def total_increase_except_tomatoes := increase_in_lettuce + increase_in_celery
def original_total_with_delivery := original_order + delivery_tip
def total_increase := new_total - original_total_with_delivery
def increase_due_to_tomatoes := total_increase - total_increase_except_tomatoes
def replaced_tomatoes := original_tomatoes + increase_due_to_tomatoes

theorem cost_of_replaced_tomatoes : replaced_tomatoes = 2.20 := by
  sorry

end cost_of_replaced_tomatoes_l131_131753


namespace inverse_function_evaluation_l131_131560

theorem inverse_function_evaluation :
  ∀ (f : ℕ → ℕ) (f_inv : ℕ → ℕ),
    (∀ y, f_inv (f y) = y) ∧ (∀ x, f (f_inv x) = x) →
    f 4 = 7 →
    f 6 = 3 →
    f 3 = 6 →
    f_inv (f_inv 6 + f_inv 7) = 4 :=
by
  intros f f_inv hf hf1 hf2 hf3
  sorry

end inverse_function_evaluation_l131_131560


namespace value_of_k_l131_131511

theorem value_of_k (k : ℤ) (h : (∀ x : ℤ, (x^2 - k * x - 6) = (x - 2) * (x + 3))) : k = -1 := by
  sorry

end value_of_k_l131_131511


namespace bucket_weight_l131_131942

theorem bucket_weight (c d : ℝ) (x y : ℝ) 
  (h1 : x + 3/4 * y = c) 
  (h2 : x + 1/3 * y = d) :
  x + 1/4 * y = (6 * d - c) / 5 := 
sorry

end bucket_weight_l131_131942


namespace cheaper_rock_cost_per_ton_l131_131719

theorem cheaper_rock_cost_per_ton (x : ℝ) 
    (h1 : 24 * 1 = 24) 
    (h2 : 800 = 16 * x + 8 * 40) : 
    x = 30 :=
sorry

end cheaper_rock_cost_per_ton_l131_131719


namespace intersection_of_A_and_B_l131_131722

def setA : Set ℝ := { x : ℝ | x > -1 }
def setB : Set ℝ := { y : ℝ | 0 ≤ y ∧ y < 1 }

theorem intersection_of_A_and_B :
  (setA ∩ setB) = { z : ℝ | 0 ≤ z ∧ z < 1 } :=
by
  sorry

end intersection_of_A_and_B_l131_131722


namespace floor_equation_solution_l131_131813

open Int

theorem floor_equation_solution (x : ℝ) :
  (⌊ ⌊ 3 * x ⌋ - 1/2 ⌋ = ⌊ x + 4 ⌋) ↔ (7/3 ≤ x ∧ x < 3) := sorry

end floor_equation_solution_l131_131813


namespace daily_rental_cost_l131_131327

def daily_cost (x : ℝ) (miles : ℝ) (cost_per_mile : ℝ) : ℝ :=
  x + miles * cost_per_mile

theorem daily_rental_cost (x : ℝ) (miles : ℝ) (cost_per_mile : ℝ) (total_budget : ℝ) 
  (h : daily_cost x miles cost_per_mile = total_budget) : x = 30 :=
by
  let constant_miles := 200
  let constant_cost_per_mile := 0.23
  let constant_budget := 76
  sorry

end daily_rental_cost_l131_131327


namespace cylinder_ellipse_major_axis_l131_131587

-- Given a right circular cylinder of radius 2
-- and a plane intersecting it forming an ellipse
-- with the major axis being 50% longer than the minor axis,
-- prove that the length of the major axis is 6.

theorem cylinder_ellipse_major_axis :
  ∀ (r : ℝ) (major minor : ℝ),
    r = 2 → major = 1.5 * minor → minor = 2 * r → major = 6 :=
by
  -- Proof step to be filled by the prover.
  sorry

end cylinder_ellipse_major_axis_l131_131587


namespace only_prime_satisfying_condition_l131_131917

theorem only_prime_satisfying_condition (p : ℕ) (h_prime : Prime p) : (Prime (p^2 + 14) ↔ p = 3) := 
by
  sorry

end only_prime_satisfying_condition_l131_131917


namespace integer_roots_of_polynomial_l131_131681

theorem integer_roots_of_polynomial : 
  {x : ℤ | x^3 - 4 * x^2 - 7 * x + 10 = 0} = {1, -2, 5} :=
by
  sorry

end integer_roots_of_polynomial_l131_131681


namespace price_per_pound_second_coffee_l131_131036

theorem price_per_pound_second_coffee
  (price_first : ℝ) (total_mix_weight : ℝ) (sell_price_per_pound : ℝ) (each_kind_weight : ℝ) 
  (total_sell_price : ℝ) (total_first_cost : ℝ) (total_second_cost : ℝ) (price_second : ℝ) :
  price_first = 2.15 →
  total_mix_weight = 18 →
  sell_price_per_pound = 2.30 →
  each_kind_weight = 9 →
  total_sell_price = total_mix_weight * sell_price_per_pound →
  total_first_cost = each_kind_weight * price_first →
  total_second_cost = total_sell_price - total_first_cost →
  price_second = total_second_cost / each_kind_weight →
  price_second = 2.45 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end price_per_pound_second_coffee_l131_131036


namespace range_subset_pos_iff_l131_131947

theorem range_subset_pos_iff (a : ℝ) : (∀ x : ℝ, ax^2 + ax + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end range_subset_pos_iff_l131_131947


namespace minimum_max_abs_x2_sub_2xy_l131_131254

theorem minimum_max_abs_x2_sub_2xy {y : ℝ} :
  ∃ y : ℝ, (∀ x ∈ (Set.Icc 0 1), abs (x^2 - 2*x*y) ≥ 0) ∧
           (∀ y' ∈ Set.univ, (∀ x ∈ (Set.Icc 0 1), abs (x^2 - 2*x*y') ≥ abs (x^2 - 2*x*y))) :=
sorry

end minimum_max_abs_x2_sub_2xy_l131_131254


namespace cats_left_correct_l131_131515

-- Define initial conditions
def siamese_cats : ℕ := 13
def house_cats : ℕ := 5
def sold_cats : ℕ := 10

-- Define the total number of cats initially
def total_cats_initial : ℕ := siamese_cats + house_cats

-- Define the number of cats left after the sale
def cats_left : ℕ := total_cats_initial - sold_cats

-- Prove the number of cats left is 8
theorem cats_left_correct : cats_left = 8 :=
by 
  sorry

end cats_left_correct_l131_131515


namespace find_integers_with_conditions_l131_131422

theorem find_integers_with_conditions :
  ∃ a b c d : ℕ, (1 ≤ a) ∧ (1 ≤ b) ∧ (1 ≤ c) ∧ (1 ≤ d) ∧ a * b * c * d = 2002 ∧ a + b + c + d < 40 := sorry

end find_integers_with_conditions_l131_131422


namespace original_number_is_144_l131_131488

theorem original_number_is_144 (x : ℕ) (h : x - x / 3 = x - 48) : x = 144 :=
by
  sorry

end original_number_is_144_l131_131488


namespace larger_triangle_perimeter_is_65_l131_131453

theorem larger_triangle_perimeter_is_65 (s1 s2 s3 t1 t2 t3 : ℝ)
  (h1 : s1 = 7) (h2 : s2 = 7) (h3 : s3 = 12)
  (h4 : t3 = 30)
  (similar : t1 / s1 = t2 / s2 ∧ t2 / s2 = t3 / s3) :
  t1 + t2 + t3 = 65 := by
  sorry

end larger_triangle_perimeter_is_65_l131_131453


namespace douglas_won_percentage_l131_131114

theorem douglas_won_percentage (p_X p_Y : ℝ) (r : ℝ) (V : ℝ) (h1 : p_X = 0.76) (h2 : p_Y = 0.4000000000000002) (h3 : r = 2) :
  (1.52 * V + 0.4000000000000002 * V) / (2 * V + V) * 100 = 64 := by
  sorry

end douglas_won_percentage_l131_131114


namespace digit_to_make_52B6_divisible_by_3_l131_131751

theorem digit_to_make_52B6_divisible_by_3 (B : ℕ) (hB : 0 ≤ B ∧ B ≤ 9) : 
  (5 + 2 + B + 6) % 3 = 0 ↔ (B = 2 ∨ B = 5 ∨ B = 8) := 
by
  sorry

end digit_to_make_52B6_divisible_by_3_l131_131751


namespace altered_solution_contains_60_liters_of_detergent_l131_131817

-- Definitions corresponding to the conditions
def initial_ratio_bleach_to_detergent_to_water : ℚ := 2 / 40 / 100
def initial_ratio_bleach_to_detergent : ℚ := 1 / 20
def initial_ratio_detergent_to_water : ℚ := 1 / 5

def altered_ratio_bleach_to_detergent : ℚ := 3 / 20
def altered_ratio_detergent_to_water : ℚ := 1 / 5

def water_in_altered_solution : ℚ := 300

-- We need to find the amount of detergent in the altered solution
def amount_of_detergent_in_altered_solution : ℚ := 20

-- The proportion and the final amount calculation
theorem altered_solution_contains_60_liters_of_detergent :
  (300 / 100) * (20) = 60 :=
by
  sorry

end altered_solution_contains_60_liters_of_detergent_l131_131817


namespace activity_popularity_order_l131_131252

theorem activity_popularity_order
  (dodgeball : ℚ := 13 / 40)
  (picnic : ℚ := 9 / 30)
  (swimming : ℚ := 7 / 20)
  (crafts : ℚ := 3 / 15) :
  (swimming > dodgeball ∧ dodgeball > picnic ∧ picnic > crafts) :=
by 
  sorry

end activity_popularity_order_l131_131252


namespace sum_sequence_six_l131_131933

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry

theorem sum_sequence_six :
  (∀ n, S n = 2 * a n + 1) → S 6 = 63 :=
by
  sorry

end sum_sequence_six_l131_131933


namespace ones_digit_of_prime_in_sequence_l131_131971

open Nat

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def valid_arithmetic_sequence (p1 p2 p3 p4: Nat) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  (p2 = p1 + 4) ∧ (p3 = p2 + 4) ∧ (p4 = p3 + 4)

theorem ones_digit_of_prime_in_sequence (p1 p2 p3 p4 : Nat) (hp_seq : valid_arithmetic_sequence p1 p2 p3 p4) (hp1_gt_3 : p1 > 3) : 
  (p1 % 10) = 9 :=
sorry

end ones_digit_of_prime_in_sequence_l131_131971


namespace fourth_root_expression_l131_131518

-- Define a positive real number y
variable (y : ℝ) (hy : 0 < y)

-- State the problem in Lean
theorem fourth_root_expression : 
  Real.sqrt (Real.sqrt (y^2 * Real.sqrt y)) = y^(5/8) := sorry

end fourth_root_expression_l131_131518


namespace total_red_yellow_black_l131_131976

/-- Calculate the total number of red, yellow, and black shirts Gavin has,
given that he has 420 shirts in total, 85 of them are blue, and 157 are
green. -/
theorem total_red_yellow_black (total_shirts : ℕ) (blue_shirts : ℕ) (green_shirts : ℕ) :
  total_shirts = 420 → blue_shirts = 85 → green_shirts = 157 → 
  (total_shirts - (blue_shirts + green_shirts) = 178) :=
by
  intros h1 h2 h3
  sorry

end total_red_yellow_black_l131_131976


namespace siblings_total_weight_l131_131351

/-- Given conditions:
Antonio's weight: 50 kilograms.
Antonio's sister weighs 12 kilograms less than Antonio.
Antonio's backpack weight: 5 kilograms.
Antonio's sister's backpack weight: 3 kilograms.
Marco's weight: 30 kilograms.
Marco's stuffed animal weight: 2 kilograms.
Prove that the total weight of the three siblings including additional weights is 128 kilograms.
-/
theorem siblings_total_weight :
  let antonio_weight := 50
  let antonio_sister_weight := antonio_weight - 12
  let antonio_backpack_weight := 5
  let antonio_sister_backpack_weight := 3
  let marco_weight := 30
  let marco_stuffed_animal_weight := 2
  antonio_weight + antonio_backpack_weight +
  antonio_sister_weight + antonio_sister_backpack_weight +
  marco_weight + marco_stuffed_animal_weight = 128 :=
by
  sorry

end siblings_total_weight_l131_131351


namespace compute_n_l131_131961

theorem compute_n (avg1 avg2 avg3 avg4 avg5 : ℚ) (h1 : avg1 = 1234 ∨ avg2 = 1234 ∨ avg3 = 1234 ∨ avg4 = 1234 ∨ avg5 = 1234)
  (h2 : avg1 = 345 ∨ avg2 = 345 ∨ avg3 = 345 ∨ avg4 = 345 ∨ avg5 = 345)
  (h3 : avg1 = 128 ∨ avg2 = 128 ∨ avg3 = 128 ∨ avg4 = 128 ∨ avg5 = 128)
  (h4 : avg1 = 19 ∨ avg2 = 19 ∨ avg3 = 19 ∨ avg4 = 19 ∨ avg5 = 19)
  (h5 : avg1 = 9.5 ∨ avg2 = 9.5 ∨ avg3 = 9.5 ∨ avg4 = 9.5 ∨ avg5 = 9.5) :
  ∃ n : ℕ, n = 2014 :=
by
  sorry

end compute_n_l131_131961


namespace David_min_max_rides_l131_131736

-- Definitions based on the conditions
variable (Alena_rides : ℕ := 11)
variable (Bara_rides : ℕ := 20)
variable (Cenek_rides : ℕ := 4)
variable (every_pair_rides_at_least_once : Prop := true)

-- Hypotheses for the problem
axiom Alena_has_ridden : Alena_rides = 11
axiom Bara_has_ridden : Bara_rides = 20
axiom Cenek_has_ridden : Cenek_rides = 4
axiom Pairs_have_ridden : every_pair_rides_at_least_once

-- Statement for the minimum and maximum rides of David
theorem David_min_max_rides (David_rides : ℕ) :
  (David_rides = 11) ∨ (David_rides = 29) :=
sorry

end David_min_max_rides_l131_131736


namespace gumball_problem_l131_131707

/--
A gumball machine contains 10 red, 6 white, 8 blue, and 9 green gumballs.
The least number of gumballs a person must buy to be sure of getting four gumballs of the same color is 13.
-/
theorem gumball_problem
  (red white blue green : ℕ)
  (h_red : red = 10)
  (h_white : white = 6)
  (h_blue : blue = 8)
  (h_green : green = 9) :
  ∃ n, n = 13 ∧ (∀ gumballs : ℕ, gumballs ≥ 13 → (∃ color_count : ℕ, color_count ≥ 4 ∧ (color_count = red ∨ color_count = white ∨ color_count = blue ∨ color_count = green))) :=
sorry

end gumball_problem_l131_131707


namespace samantha_spends_on_dog_toys_l131_131612

theorem samantha_spends_on_dog_toys:
  let toy_price := 12.00
  let discount := 0.5
  let num_toys := 4
  let tax_rate := 0.08
  let full_price_toys := num_toys / 2
  let half_price_toys := num_toys / 2
  let total_cost_before_tax := full_price_toys * toy_price + half_price_toys * (toy_price * discount)
  let sales_tax := total_cost_before_tax * tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  total_cost_after_tax = 38.88 :=
by {
  sorry
}

end samantha_spends_on_dog_toys_l131_131612


namespace total_people_on_boats_l131_131547

theorem total_people_on_boats (boats : ℕ) (people_per_boat : ℕ) (h_boats : boats = 5) (h_people : people_per_boat = 3) : boats * people_per_boat = 15 :=
by
  sorry

end total_people_on_boats_l131_131547


namespace solve_for_z_l131_131926

theorem solve_for_z (z : ℂ) (h : z * (1 - I) = 2) : z = 1 + I :=
  sorry

end solve_for_z_l131_131926


namespace arithmetic_sequence_common_difference_l131_131837

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) (h_arith_seq: ∀ n, a n = a 1 + (n - 1) * d) 
  (h_cond1 : a 3 + a 9 = 4 * a 5) (h_cond2 : a 2 = -8) : 
  d = 4 :=
sorry

end arithmetic_sequence_common_difference_l131_131837


namespace norm_photos_l131_131322

-- Define variables for the number of photos taken by Lisa, Mike, and Norm.
variables {L M N : ℕ}

-- Define the given conditions as hypotheses.
def condition1 (L M N : ℕ) : Prop := L + M = M + N - 60
def condition2 (L N : ℕ) : Prop := N = 2 * L + 10

-- State the problem in Lean: we want to prove that the number of photos Norm took is 110.
theorem norm_photos (L M N : ℕ) (h1 : condition1 L M N) (h2 : condition2 L N) : N = 110 :=
by
  sorry

end norm_photos_l131_131322


namespace greatest_price_book_l131_131784

theorem greatest_price_book (p : ℕ) (B : ℕ) (D : ℕ) (F : ℕ) (T : ℚ) 
  (h1 : B = 20) 
  (h2 : D = 200) 
  (h3 : F = 5)
  (h4 : T = 0.07) 
  (h5 : ∀ p, 20 * p * (1 + T) ≤ (D - F)) : 
  p ≤ 9 :=
by
  sorry

end greatest_price_book_l131_131784


namespace mean_problem_l131_131603

theorem mean_problem (x : ℝ) (h : (12 + x + 42 + 78 + 104) / 5 = 62) :
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 :=
by
  sorry

end mean_problem_l131_131603


namespace closest_integer_to_99_times_9_l131_131136

theorem closest_integer_to_99_times_9 :
  let target := 99 * 9
  let dist (n : ℤ) := |target - n|
  let choices := [10000, 100, 100000, 1000, 10]
  1000 ∈ choices ∧ ∀ (n : ℤ), n ∈ choices → dist 1000 ≤ dist n :=
by
  let target := 99 * 9
  let dist (n : ℤ) := |target - n|
  let choices := [10000, 100, 100000, 1000, 10]
  sorry

end closest_integer_to_99_times_9_l131_131136


namespace problem1_problem2_l131_131192

-- Definition and proof statement for Problem 1
theorem problem1 (y : ℝ) : 
  (y + 2) * (y - 2) + (y - 1) * (y + 3) = 2 * y^2 + 2 * y - 7 := 
by sorry

-- Definition and proof statement for Problem 2
theorem problem2 (x : ℝ) (h : x ≠ -1) :
  (1 + 2 / (x + 1)) / ((x^2 + 6 * x + 9) / (x + 1)) = 1 / (x + 3) :=
by sorry

end problem1_problem2_l131_131192


namespace count_ordered_pairs_no_distinct_real_solutions_l131_131923

theorem count_ordered_pairs_no_distinct_real_solutions :
  {n : Nat // ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (4 * b^2 - 4 * c ≤ 0) ∧ (4 * c^2 - 4 * b ≤ 0) ∧ n = 1} :=
sorry

end count_ordered_pairs_no_distinct_real_solutions_l131_131923


namespace property_depreciation_rate_l131_131983

noncomputable def initial_value : ℝ := 25599.08977777778
noncomputable def final_value : ℝ := 21093
noncomputable def annual_depreciation_rate : ℝ := 0.063

theorem property_depreciation_rate :
  final_value = initial_value * (1 - annual_depreciation_rate)^3 :=
sorry

end property_depreciation_rate_l131_131983


namespace tenth_term_arithmetic_sequence_l131_131288

theorem tenth_term_arithmetic_sequence :
  let a_1 := (1 : ℝ) / 2
  let a_2 := (5 : ℝ) / 6
  let d := a_2 - a_1
  (a_1 + 9 * d) = 7 / 2 := 
by
  sorry

end tenth_term_arithmetic_sequence_l131_131288


namespace possible_values_of_m_l131_131198

def F1 := (-3, 0)
def F2 := (3, 0)
def possible_vals := [2, -1, 4, -3, 1/2]

noncomputable def is_valid_m (m : ℝ) : Prop :=
  abs (2 * m - 1) < 6 ∧ m ≠ 1/2

theorem possible_values_of_m : {m ∈ possible_vals | is_valid_m m} = {2, -1} := by
  sorry

end possible_values_of_m_l131_131198


namespace product_of_x_y_l131_131396

theorem product_of_x_y (x y : ℝ) :
  (54 = 5 * y^2 + 20) →
  (8 * x^2 + 2 = 38) →
  x * y = Real.sqrt (30.6) :=
by
  intros h1 h2
  -- these would be the proof steps
  sorry

end product_of_x_y_l131_131396


namespace compressor_stations_distances_l131_131038

theorem compressor_stations_distances 
    (x y z a : ℝ) 
    (h1 : x + y = 2 * z)
    (h2 : z + y = x + a)
    (h3 : x + z = 75)
    (h4 : 0 ≤ x)
    (h5 : 0 ≤ y)
    (h6 : 0 ≤ z)
    (h7 : 0 < a)
    (h8 : a < 100) :
  (a = 15 → x = 42 ∧ y = 24 ∧ z = 33) :=
by 
  intro ha_eq_15
  sorry

end compressor_stations_distances_l131_131038


namespace line_circle_intersection_l131_131004

-- Define the line and circle in Lean
def line_eq (x y : ℝ) : Prop := x + y - 6 = 0
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 2

-- Define the proof about the intersection
theorem line_circle_intersection :
  (∃ x y : ℝ, line_eq x y ∧ circle_eq x y) ∧
  ∀ (x1 y1 x2 y2 : ℝ), (line_eq x1 y1 ∧ circle_eq x1 y1) → (line_eq x2 y2 ∧ circle_eq x2 y2) → (x1 = x2 ∧ y1 = y2) :=
by {
  sorry
}

end line_circle_intersection_l131_131004


namespace main_theorem_l131_131432

variable {a b c : ℝ}

noncomputable def inequality_1 (a b c : ℝ) : Prop :=
  a + b + c ≤ (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b)

noncomputable def inequality_2 (a b c : ℝ) : Prop :=
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≤ a^2 / (b * c) + b^2 / (c * a) + c^2 / (a * b)

theorem main_theorem (h : a < 0 ∧ b < 0 ∧ c < 0) :
  inequality_1 a b c ∧ inequality_2 a b c := by sorry

end main_theorem_l131_131432


namespace beam_reflection_problem_l131_131009

theorem beam_reflection_problem
  (A B D C : Point)
  (angle_CDA : ℝ)
  (total_path_length_max : ℝ)
  (equal_angle_reflections : ∀ (k : ℕ), angle_CDA * k ≤ 90)
  (path_length_constraint : ∀ (n : ℕ) (d : ℝ), 2 * n * d ≤ total_path_length_max)
  : angle_CDA = 5 ∧ total_path_length_max = 100 → ∃ (n : ℕ), n = 10 :=
sorry

end beam_reflection_problem_l131_131009


namespace arg_cubed_eq_pi_l131_131819

open Complex

theorem arg_cubed_eq_pi (z1 z2 : ℂ) (h1 : abs z1 = 3) (h2 : abs z2 = 5) (h3 : abs (z1 + z2) = 7) : 
  arg (z2 / z1) ^ 3 = π :=
by
  sorry

end arg_cubed_eq_pi_l131_131819


namespace number_of_children_l131_131493

-- Definitions based on conditions from the problem
def total_spectators := 10000
def men_spectators := 7000
def spectators_other_than_men := total_spectators - men_spectators
def women_and_children_ratio := 5

-- Prove there are 2500 children
theorem number_of_children : 
  ∃ (women children : ℕ), 
    spectators_other_than_men = women + women_and_children_ratio * women ∧ 
    children = women_and_children_ratio * women ∧
    children = 2500 :=
by
  sorry

end number_of_children_l131_131493


namespace car_travel_distance_l131_131047

-- Define the conditions: speed and time
def speed : ℝ := 160 -- in km/h
def time : ℝ := 5 -- in hours

-- Define the calculation for distance
def distance (s t : ℝ) : ℝ := s * t

-- Prove that given the conditions, the distance is 800 km
theorem car_travel_distance : distance speed time = 800 := by
  sorry

end car_travel_distance_l131_131047


namespace son_age_is_eight_l131_131439

theorem son_age_is_eight (F S : ℕ) (h1 : F + 6 + S + 6 = 68) (h2 : F = 6 * S) : S = 8 :=
by
  sorry

end son_age_is_eight_l131_131439


namespace daily_wage_of_c_l131_131244

theorem daily_wage_of_c 
  (a_days : ℕ) (b_days : ℕ) (c_days : ℕ) 
  (wage_ratio_a_b : ℚ) (wage_ratio_b_c : ℚ) 
  (total_earnings : ℚ) 
  (A : ℚ) (C : ℚ) :
  a_days = 6 →
  b_days = 9 →
  c_days = 4 →
  wage_ratio_a_b = 3 / 4 →
  wage_ratio_b_c = 4 / 5 →
  total_earnings = 1850 →
  A = 75 →
  C = 208.33 := 
sorry

end daily_wage_of_c_l131_131244


namespace julie_monthly_salary_l131_131218

def hourly_rate : ℕ := 5
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 6
def missed_days : ℕ := 1
def weeks_per_month : ℕ := 4

theorem julie_monthly_salary :
  (hourly_rate * hours_per_day * (days_per_week - missed_days) * weeks_per_month) = 920 :=
by
  sorry

end julie_monthly_salary_l131_131218


namespace problem_solution_l131_131205

noncomputable def f1 (x : ℝ) : ℝ := -2 * x + 2 * Real.sqrt 2
noncomputable def f2 (x : ℝ) : ℝ := Real.sin x
noncomputable def f3 (x : ℝ) : ℝ := x + (1 / x)
noncomputable def f4 (x : ℝ) : ℝ := Real.exp x
noncomputable def f5 (x : ℝ) : ℝ := -2 * Real.log x

def has_inverse_proportion_point (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∃ x ∈ domain, x * f x = 1

theorem problem_solution :
  (has_inverse_proportion_point f1 univ) ∧
  (has_inverse_proportion_point f2 (Set.Icc 0 (2 * Real.pi))) ∧
  ¬ (has_inverse_proportion_point f3 (Set.Ioi 0)) ∧
  (has_inverse_proportion_point f4 univ) ∧
  ¬ (has_inverse_proportion_point f5 (Set.Ioi 0)) :=
by
  sorry

end problem_solution_l131_131205


namespace evaluate_expression_l131_131595

theorem evaluate_expression (a b : ℕ) (ha : a = 7) (hb : b = 5) : 3 * (a^3 + b^3) / (a^2 - a * b + b^2) = 36 :=
by
  rw [ha, hb]
  sorry

end evaluate_expression_l131_131595


namespace total_time_six_laps_l131_131859

-- Defining the constants and conditions
def total_distance : Nat := 500
def speed_part1 : Nat := 3
def distance_part1 : Nat := 150
def speed_part2 : Nat := 6
def distance_part2 : Nat := total_distance - distance_part1
def laps : Nat := 6

-- Calculating the times based on conditions
def time_part1 := distance_part1 / speed_part1
def time_part2 := distance_part2 / speed_part2
def time_per_lap := time_part1 + time_part2
def total_time := laps * time_per_lap

-- The goal is to prove the total time is 10 minutes and 48 seconds (648 seconds)
theorem total_time_six_laps : total_time = 648 :=
-- proof would go here
sorry

end total_time_six_laps_l131_131859


namespace number_of_integers_between_sqrt10_and_sqrt100_l131_131729

theorem number_of_integers_between_sqrt10_and_sqrt100 :
  (∃ n : ℕ, n = 7) :=
sorry

end number_of_integers_between_sqrt10_and_sqrt100_l131_131729


namespace part_I_part_II_l131_131197

noncomputable def f (x : ℝ) := x * (Real.log x - 1) + Real.log x + 1

theorem part_I :
  let f_tangent (x y : ℝ) := x - y - 1
  (∀ x y, f_tangent x y = 0 ↔ y = x - 1) ∧ f_tangent 1 (f 1) = 0 :=
by
  sorry

theorem part_II (m : ℝ) :
  (∀ x : ℝ, x > 0 → x^2 + x * (m - (Real.log x + 1 / x)) + 1 ≥ 0) → m ≥ -1 :=
by
  sorry

end part_I_part_II_l131_131197


namespace complete_collection_probability_l131_131775

namespace Stickers

open scoped Classical

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.Ico 1 (n + 1))

def combinations (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_complete_collection :
  ℚ := (combinations 6 6 * combinations 12 4) / combinations 18 10

theorem complete_collection_probability :
  probability_complete_collection = (5 / 442 : ℚ) := by
  sorry

end Stickers

end complete_collection_probability_l131_131775


namespace range_of_x_l131_131864

variables {x : Real}

def P (x : Real) : Prop := (x + 1) / (x - 3) ≥ 0
def Q (x : Real) : Prop := abs (1 - x/2) < 1

theorem range_of_x (hP : P x) (hQ : ¬ Q x) : x ≤ -1 ∨ x ≥ 4 :=
  sorry

end range_of_x_l131_131864


namespace length_increase_percentage_l131_131266

theorem length_increase_percentage
  (L W : ℝ)
  (A : ℝ := L * W)
  (A' : ℝ := 1.30000000000000004 * A)
  (new_length : ℝ := L * (1 + x / 100))
  (new_width : ℝ := W / 2)
  (area_equiv : new_length * new_width = A')
  (x : ℝ) :
  1 + x / 100 = 2.60000000000000008 :=
by
  -- Proof goes here
  sorry

end length_increase_percentage_l131_131266


namespace songs_distribution_l131_131578

-- Define the sets involved
structure Girl := (Amy Beth Jo : Prop)
axiom no_song_liked_by_all : ∀ song : Girl, ¬(song.Amy ∧ song.Beth ∧ song.Jo)
axiom no_song_disliked_by_all : ∀ song : Girl, song.Amy ∨ song.Beth ∨ song.Jo
axiom pairwise_liked : ∀ song : Girl,
  (song.Amy ∧ song.Beth ∧ ¬song.Jo) ∨
  (song.Beth ∧ song.Jo ∧ ¬song.Amy) ∨
  (song.Jo ∧ song.Amy ∧ ¬song.Beth)

-- Define the theorem to prove that there are exactly 90 ways to distribute the songs
theorem songs_distribution : ∃ ways : ℕ, ways = 90 := sorry

end songs_distribution_l131_131578


namespace total_pay_l131_131610

-- Definitions based on the conditions
def y_pay : ℕ := 290
def x_pay : ℕ := (120 * y_pay) / 100

-- The statement to prove that the total pay is Rs. 638
theorem total_pay : x_pay + y_pay = 638 := 
by
  -- skipping the proof for now
  sorry

end total_pay_l131_131610


namespace triangle_inequality_l131_131705

noncomputable def p (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def r (a b c : ℝ) : ℝ := 
  let p := p a b c
  let x := p - a
  let y := p - b
  let z := p - c
  Real.sqrt ((x * y * z) / (x + y + z))

noncomputable def x (a b c : ℝ) : ℝ := p a b c - a
noncomputable def y (a b c : ℝ) : ℝ := p a b c - b
noncomputable def z (a b c : ℝ) : ℝ := p a b c - c

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : a + b > c ∧ a + c > b ∧ b + c > a) :
  1 / (x a b c)^2 + 1 / (y a b c)^2 + 1 / (z a b c)^2 ≥ (x a b c + y a b c + z a b c) / ((x a b c) * (y a b c) * (z a b c)) := by
    sorry

end triangle_inequality_l131_131705


namespace length_of_GH_l131_131677

-- Define the lengths of the segments as given in the conditions
def AB : ℕ := 11
def FE : ℕ := 13
def CD : ℕ := 5

-- Define what we need to prove: the length of GH is 29
theorem length_of_GH (AB FE CD : ℕ) : AB = 11 → FE = 13 → CD = 5 → (AB + CD + FE = 29) :=
by
  sorry

end length_of_GH_l131_131677


namespace train_length_l131_131030

noncomputable def convert_speed (v_kmh : ℝ) : ℝ :=
  v_kmh * (5 / 18)

def length_of_train (speed_mps : ℝ) (time_sec : ℝ) : ℝ :=
  speed_mps * time_sec

theorem train_length (v_kmh : ℝ) (t_sec : ℝ) (length_m : ℝ) :
  v_kmh = 60 →
  t_sec = 45 →
  length_m = 750 →
  length_of_train (convert_speed v_kmh) t_sec = length_m :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end train_length_l131_131030


namespace angle_perpendicular_vectors_l131_131814

theorem angle_perpendicular_vectors (α : ℝ) (h1 : 0 < α) (h2 : α < π)
  (h3 : (1 : ℝ) * Real.sin α + Real.cos α * (1 : ℝ) = 0) : α = 3 * Real.pi / 4 :=
sorry

end angle_perpendicular_vectors_l131_131814


namespace polygon_interior_sum_polygon_angle_ratio_l131_131024

-- Part 1: Number of sides based on the sum of interior angles
theorem polygon_interior_sum (n: ℕ) (h: (n - 2) * 180 = 2340) : n = 15 :=
  sorry

-- Part 2: Number of sides based on the ratio of interior to exterior angles
theorem polygon_angle_ratio (n: ℕ) (exterior_angle: ℕ) (ratio: 13 * exterior_angle + 2 * exterior_angle = 180) : n = 15 :=
  sorry

end polygon_interior_sum_polygon_angle_ratio_l131_131024


namespace car_r_speed_l131_131029

theorem car_r_speed (v : ℝ) (h : 150 / v - 2 = 150 / (v + 10)) : v = 25 :=
sorry

end car_r_speed_l131_131029


namespace train_speed_l131_131059

theorem train_speed (v : ℕ) :
  (∀ (d : ℕ), d = 480 → ∀ (ship_speed : ℕ), ship_speed = 60 → 
  (∀ (ship_time : ℕ), ship_time = d / ship_speed →
  (∀ (train_time : ℕ), train_time = ship_time + 2 →
  v = d / train_time))) → v = 48 :=
by
  sorry

end train_speed_l131_131059


namespace polygon_area_l131_131741

theorem polygon_area (sides : ℕ) (perpendicular_adjacent : Bool) (congruent_sides : Bool) (perimeter : ℝ) (area : ℝ) :
  sides = 32 → 
  perpendicular_adjacent = true → 
  congruent_sides = true →
  perimeter = 64 →
  area = 64 :=
by
  intros h1 h2 h3 h4
  sorry

end polygon_area_l131_131741


namespace arithmetic_sequence_sum_of_bn_l131_131046

variable (a : ℕ → ℕ) (b : ℕ → ℚ) (S : ℕ → ℚ)

theorem arithmetic_sequence (h1 : a 1 + a 4 = 10) (h2 : a 3 = 6) :
  (∀ n, a n = 2 * n) :=
by sorry

theorem sum_of_bn (h1 : a 1 + a 4 = 10) (h2 : a 3 = 6)
                  (h3 : ∀ n, a n = 2 * n)
                  (h4 : ∀ n, b n = 4 / (a n * a (n + 1))) :
  (∀ n, S n = n / (n + 1)) :=
by sorry

end arithmetic_sequence_sum_of_bn_l131_131046


namespace population_net_increase_in_one_day_l131_131183

-- Definitions based on the conditions
def birth_rate_per_two_seconds : ℝ := 4
def death_rate_per_two_seconds : ℝ := 3
def seconds_in_a_day : ℝ := 86400

-- The main theorem to prove
theorem population_net_increase_in_one_day : 
  (birth_rate_per_two_seconds / 2 - death_rate_per_two_seconds / 2) * seconds_in_a_day = 43200 :=
by
  sorry

end population_net_increase_in_one_day_l131_131183


namespace part1_part2_l131_131704

variables (a b c : ℝ)

theorem part1 (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  ab + bc + ac ≤ 1 / 3 := sorry

theorem part2 (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  1 / a + 1 / b + 1 / c ≥ 9 := sorry

end part1_part2_l131_131704


namespace ones_digit_of_8_pow_47_l131_131341

theorem ones_digit_of_8_pow_47 :
  (8^47) % 10 = 2 :=
by
  sorry

end ones_digit_of_8_pow_47_l131_131341


namespace sqrt_neg4_sq_eq_4_l131_131386

theorem sqrt_neg4_sq_eq_4 : Real.sqrt ((-4 : ℝ) ^ 2) = 4 := by
  sorry

end sqrt_neg4_sq_eq_4_l131_131386


namespace convex_polygon_triangle_count_l131_131504

theorem convex_polygon_triangle_count {n : ℕ} (h : n ≥ 5) :
  ∃ T : ℕ, T ≤ n * (2 * n - 5) / 3 :=
by
  sorry

end convex_polygon_triangle_count_l131_131504


namespace speed_of_first_train_l131_131954

-- Definitions of the conditions
def ratio_speed (speed1 speed2 : ℝ) := speed1 / speed2 = 7 / 8
def speed_of_second_train := 400 / 4

-- The theorem we want to prove
theorem speed_of_first_train (speed2 := speed_of_second_train) (h : ratio_speed S1 speed2) :
  S1 = 87.5 :=
by 
  sorry

end speed_of_first_train_l131_131954


namespace relationship_among_a_b_c_l131_131910

noncomputable def a : ℝ := Real.log 2 / Real.log 0.3
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.4 * Real.log 0.3)

theorem relationship_among_a_b_c : a < c ∧ c < b := by
  sorry

end relationship_among_a_b_c_l131_131910


namespace fixed_point_of_log_function_l131_131005

theorem fixed_point_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (-1, 2) ∧ ∀ x y : ℝ, y = 2 + Real.logb a (x + 2) → y = 2 → x = -1 :=
by
  sorry

end fixed_point_of_log_function_l131_131005


namespace distinct_exponentiation_values_l131_131564

theorem distinct_exponentiation_values : 
  (∃ v1 v2 v3 v4 v5 : ℕ, 
    v1 = (3 : ℕ)^(3 : ℕ)^(3 : ℕ)^(3 : ℕ) ∧
    v2 = (3 : ℕ)^((3 : ℕ)^(3 : ℕ)^(3 : ℕ)) ∧
    v3 = (3 : ℕ)^(((3 : ℕ)^(3 : ℕ))^(3 : ℕ)) ∧
    v4 = ((3 : ℕ)^(3 : ℕ)^3) ∧
    v5 = ((3 : ℕ)^((3 : ℕ)^(3 : ℕ)^3)) ∧
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ 
    v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ 
    v3 ≠ v4 ∧ v3 ≠ v5 ∧ 
    v4 ≠ v5) := 
sorry

end distinct_exponentiation_values_l131_131564


namespace renovation_project_cement_loads_l131_131757

theorem renovation_project_cement_loads
  (s : ℚ) (d : ℚ) (t : ℚ)
  (hs : s = 0.16666666666666666) 
  (hd : d = 0.3333333333333333)
  (ht : t = 0.6666666666666666) :
  t - (s + d) = 0.1666666666666666 := by
  sorry

end renovation_project_cement_loads_l131_131757


namespace compare_neg_numbers_l131_131606

theorem compare_neg_numbers : - 0.6 > - (2 / 3) := 
by sorry

end compare_neg_numbers_l131_131606


namespace time_to_meet_l131_131094

variables {v_g : ℝ} -- speed of Petya and Vasya on the dirt road
variable {v_a : ℝ} -- speed of Petya on the paved road
variable {t : ℝ} -- time from start until Petya and Vasya meet
variable {x : ℝ} -- distance from the starting point to the bridge

-- Conditions
axiom Petya_speed : v_a = 3 * v_g
axiom Petya_bridge_time : x / v_a = 1
axiom Vasya_speed : v_a ≠ 0 ∧ v_g ≠ 0

-- Statement
theorem time_to_meet (h1 : v_a = 3 * v_g) (h2 : x / v_a = 1) (h3 : v_a ≠ 0 ∧ v_g ≠ 0) : t = 2 :=
by
  have h4 : x = 3 * v_g := sorry
  have h5 : (2 * x - 2 * v_g) / (2 * v_g) = 1 := sorry
  exact sorry

end time_to_meet_l131_131094


namespace total_value_of_coins_l131_131516

variable (numCoins : ℕ) (coinsValue : ℕ) 

theorem total_value_of_coins : 
  numCoins = 15 → 
  (∀ n: ℕ, n = 5 → coinsValue = 12) → 
  ∃ totalValue : ℕ, totalValue = 36 :=
  by
    sorry

end total_value_of_coins_l131_131516


namespace find_y_when_x_is_6_l131_131692

variable (x y : ℝ)
variable (h₁ : x > 0)
variable (h₂ : y > 0)
variable (k : ℝ)

axiom inverse_proportional : 3 * x^2 * y = k
axiom initial_condition : 3 * 3^2 * 30 = k

theorem find_y_when_x_is_6 (h : x = 6) : y = 7.5 :=
by
  sorry

end find_y_when_x_is_6_l131_131692


namespace count_integers_in_range_l131_131891

theorem count_integers_in_range : 
  ∃ n : ℕ, n = 6 ∧ ∀ x : ℤ, (-7 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 8) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by
  sorry

end count_integers_in_range_l131_131891


namespace complement_union_A_B_eq_neg2_0_l131_131020

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_eq_neg2_0 :
  U \ (A ∪ B) = {-2, 0} := by
  sorry

end complement_union_A_B_eq_neg2_0_l131_131020


namespace age_difference_l131_131074

noncomputable def years_older (A B : ℕ) : ℕ :=
A - B

theorem age_difference (A B : ℕ) (h1 : B = 39) (h2 : A + 10 = 2 * (B - 10)) :
  years_older A B = 9 :=
by
  rw [years_older]
  rw [h1] at h2
  sorry

end age_difference_l131_131074


namespace mary_finds_eggs_l131_131782

theorem mary_finds_eggs (initial final found : ℕ) (h_initial : initial = 27) (h_final : final = 31) :
  found = final - initial → found = 4 :=
by
  intro h
  rw [h_initial, h_final] at h
  exact h

end mary_finds_eggs_l131_131782


namespace change_is_correct_l131_131023

-- Define the cost of the pencil in cents
def cost_of_pencil : ℕ := 35

-- Define the amount paid in cents
def amount_paid : ℕ := 100

-- State the theorem for the change
theorem change_is_correct : amount_paid - cost_of_pencil = 65 :=
by sorry

end change_is_correct_l131_131023


namespace find_c_l131_131785

theorem find_c (a b c d y1 y2 : ℝ) (h1 : y1 = a * 2^3 + b * 2^2 + c * 2 + d)
  (h2 : y2 = a * (-2)^3 + b * (-2)^2 + c * (-2) + d)
  (h3 : y1 - y2 = 12) : c = 3 - 4 * a := by
  sorry

end find_c_l131_131785


namespace probability_select_cooking_l131_131779

theorem probability_select_cooking : 
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := 4
  let cooking_selected := 1
  cooking_selected / total_courses = 1 / 4 :=
by
  sorry

end probability_select_cooking_l131_131779


namespace least_positive_multiple_of_45_with_product_of_digits_multiple_of_9_l131_131324

noncomputable def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else n.digits 10

noncomputable def product_of_digits (n : ℕ) : ℕ :=
  (digits n).foldl (λ x y => x * y) 1

def is_multiple_of (m n : ℕ) : Prop :=
  ∃ k, n = m * k

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of 45 n ∧ is_multiple_of 9 (product_of_digits n)

theorem least_positive_multiple_of_45_with_product_of_digits_multiple_of_9 : 
  ∀ n, satisfies_conditions n → 495 ≤ n :=
by
  sorry

end least_positive_multiple_of_45_with_product_of_digits_multiple_of_9_l131_131324


namespace hiking_rate_l131_131473

theorem hiking_rate (rate_uphill: ℝ) (time_total: ℝ) (time_uphill: ℝ) (rate_downhill: ℝ) 
  (h1: rate_uphill = 4) (h2: time_total = 3) (h3: time_uphill = 1.2) : rate_downhill = 4.8 / (time_total - time_uphill) :=
by
  sorry

end hiking_rate_l131_131473


namespace concert_total_revenue_l131_131348

def adult_ticket_price : ℕ := 26
def child_ticket_price : ℕ := adult_ticket_price / 2
def num_adults : ℕ := 183
def num_children : ℕ := 28

def revenue_from_adults : ℕ := num_adults * adult_ticket_price
def revenue_from_children : ℕ := num_children * child_ticket_price
def total_revenue : ℕ := revenue_from_adults + revenue_from_children

theorem concert_total_revenue :
  total_revenue = 5122 :=
by
  -- proof can be filled in here
  sorry

end concert_total_revenue_l131_131348


namespace decimal_equivalent_of_quarter_cubed_l131_131803

theorem decimal_equivalent_of_quarter_cubed :
    (1 / 4 : ℝ) ^ 3 = 0.015625 := 
by
    sorry

end decimal_equivalent_of_quarter_cubed_l131_131803


namespace arithmetic_proof_l131_131943

def arithmetic_expression := 3889 + 12.952 - 47.95000000000027
def expected_result := 3854.002

theorem arithmetic_proof : arithmetic_expression = expected_result := by
  -- The proof goes here
  sorry

end arithmetic_proof_l131_131943


namespace average_difference_l131_131622

theorem average_difference :
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 70 + 28) / 3
  avg1 - avg2 = 4 :=
by
  -- Define the averages
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 70 + 28) / 3
  sorry

end average_difference_l131_131622


namespace sanAntonioToAustin_passes_austinToSanAntonio_l131_131816

noncomputable def buses_passed : ℕ :=
  let austinToSanAntonio (n : ℕ) : ℕ := n * 2
  let sanAntonioToAustin (n : ℕ) : ℕ := n * 2 + 1
  let tripDuration : ℕ := 3
  if (austinToSanAntonio 3 - 0) <= tripDuration then 2 else 0

-- Proof statement
theorem sanAntonioToAustin_passes_austinToSanAntonio :
  buses_passed = 2 :=
  sorry

end sanAntonioToAustin_passes_austinToSanAntonio_l131_131816


namespace min_keychains_to_reach_profit_l131_131442

theorem min_keychains_to_reach_profit :
  let cost_per_keychain := 0.15
  let sell_price_per_keychain := 0.45
  let total_keychains := 1200
  let target_profit := 180
  let total_cost := total_keychains * cost_per_keychain
  let total_revenue := total_cost + target_profit
  let min_keychains_to_sell := total_revenue / sell_price_per_keychain
  min_keychains_to_sell = 800 := 
by
  sorry

end min_keychains_to_reach_profit_l131_131442


namespace total_distance_walked_l131_131758

-- Define the given conditions
def walks_to_work_days := 5
def walks_dog_days := 7
def walks_to_friend_days := 1
def walks_to_store_days := 2

def distance_to_work := 6
def distance_dog_walk := 2
def distance_to_friend := 1
def distance_to_store := 3

-- The proof statement
theorem total_distance_walked :
  (walks_to_work_days * (distance_to_work * 2)) +
  (walks_dog_days * (distance_dog_walk * 2)) +
  (walks_to_friend_days * distance_to_friend) +
  (walks_to_store_days * distance_to_store) = 95 := 
sorry

end total_distance_walked_l131_131758


namespace boy_usual_time_l131_131639

theorem boy_usual_time (R T : ℝ) (h : R * T = (7 / 6) * R * (T - 2)) : T = 14 :=
by
  sorry

end boy_usual_time_l131_131639


namespace square_side_percentage_increase_l131_131527

theorem square_side_percentage_increase (s : ℝ) (p : ℝ) :
  (s * (1 + p / 100)) ^ 2 = 1.44 * s ^ 2 → p = 20 :=
by
  sorry

end square_side_percentage_increase_l131_131527


namespace average_weight_of_girls_l131_131843

theorem average_weight_of_girls :
  ∀ (total_students boys girls total_weight class_average_weight boys_average_weight girls_average_weight : ℝ),
  total_students = 25 →
  boys = 15 →
  girls = 10 →
  boys + girls = total_students →
  class_average_weight = 45 →
  boys_average_weight = 48 →
  total_weight = 1125 →
  girls_average_weight = (total_weight - (boys * boys_average_weight)) / girls →
  total_weight = class_average_weight * total_students →
  girls_average_weight = 40.5 :=
by
  intros total_students boys girls total_weight class_average_weight boys_average_weight girls_average_weight
  sorry

end average_weight_of_girls_l131_131843


namespace find_sum_l131_131431

theorem find_sum (a b : ℝ) (ha : a^3 - 3 * a^2 + 5 * a - 17 = 0) (hb : b^3 - 3 * b^2 + 5 * b + 11 = 0) :
  a + b = 2 :=
sorry

end find_sum_l131_131431


namespace grayson_fraction_l131_131416

variable (A G O : ℕ) -- The number of boxes collected by Abigail, Grayson, and Olivia, respectively
variable (C_per_box : ℕ) -- The number of cookies per box
variable (TotalCookies : ℕ) -- The total number of cookies collected by Abigail, Grayson, and Olivia

-- Given conditions
def abigail_boxes : ℕ := 2
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48
def total_cookies : ℕ := 276

-- Prove the fraction of the box that Grayson collected
theorem grayson_fraction :
  G * C_per_box = TotalCookies - (abigail_boxes + olivia_boxes) * cookies_per_box → 
  G / C_per_box = 3 / 4 := 
by
  sorry

-- Assume the variables from conditions
variable (G : ℕ := 36 / 48)
variable (TotalCookies := 276)
variable (C_per_box := 48)
variable (A := 2)
variable (O := 3)


end grayson_fraction_l131_131416


namespace max_marks_l131_131178

variable (M : ℝ)

theorem max_marks (h1 : 0.35 * M = 175) : M = 500 := by
  -- Proof goes here
  sorry

end max_marks_l131_131178


namespace prove_m_value_l131_131468

theorem prove_m_value (m : ℕ) : 8^4 = 4^m → m = 6 := by
  sorry

end prove_m_value_l131_131468


namespace miles_driven_l131_131675

def rental_fee : ℝ := 20.99
def charge_per_mile : ℝ := 0.25
def total_amount_paid : ℝ := 95.74

theorem miles_driven (miles_driven: ℝ) : 
  (total_amount_paid - rental_fee) / charge_per_mile = miles_driven → miles_driven = 299 := by
  intros
  sorry

end miles_driven_l131_131675


namespace angles_on_line_y_eq_x_l131_131514

-- Define a predicate representing that an angle has its terminal side on the line y = x
def angle_on_line_y_eq_x (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 4

-- The goal is to prove that the set of all such angles is as stated
theorem angles_on_line_y_eq_x :
  { α : ℝ | ∃ k : ℤ, α = k * Real.pi + Real.pi / 4 } = { α : ℝ | angle_on_line_y_eq_x α } :=
sorry

end angles_on_line_y_eq_x_l131_131514


namespace customers_added_l131_131349

theorem customers_added (x : ℕ) (h : 29 + x = 49) : x = 20 := by
  sorry

end customers_added_l131_131349


namespace total_population_l131_131776

variable (b g t s : ℕ)

theorem total_population (hb : b = 4 * g) (hg : g = 8 * t) (ht : t = 2 * s) :
  b + g + t + s = (83 * g) / 16 :=
by sorry

end total_population_l131_131776


namespace total_fireworks_l131_131491

-- Define the conditions
def fireworks_per_number := 6
def fireworks_per_letter := 5
def numbers_in_year := 4
def letters_in_phrase := 12
def number_of_boxes := 50
def fireworks_per_box := 8

-- Main statement: Prove the total number of fireworks lit during the display
theorem total_fireworks : fireworks_per_number * numbers_in_year + fireworks_per_letter * letters_in_phrase + number_of_boxes * fireworks_per_box = 484 :=
by
  sorry

end total_fireworks_l131_131491


namespace solveExpression_l131_131257

noncomputable def evaluateExpression : ℝ := (Real.sqrt 3) / Real.sin (Real.pi / 9) - 1 / Real.sin (7 * Real.pi / 18)

theorem solveExpression : evaluateExpression = 4 :=
by sorry

end solveExpression_l131_131257


namespace coffee_blend_price_l131_131128

theorem coffee_blend_price (x : ℝ) : 
  (9 * 8 + x * 12) / 20 = 8.4 → x = 8 :=
by
  intro h
  sorry

end coffee_blend_price_l131_131128


namespace augmented_matrix_correct_l131_131187

-- Define the system of linear equations as a pair of equations
def system_of_equations (x y : ℝ) : Prop :=
  (2 * x + y = 1) ∧ (3 * x - 2 * y = 0)

-- Define what it means to be the correct augmented matrix for the system
def is_augmented_matrix (A : Matrix (Fin 2) (Fin 3) ℝ) : Prop :=
  A = ![
    ![2, 1, 1],
    ![3, -2, 0]
  ]

-- The theorem states that the augmented matrix of the given system of equations is the specified matrix
theorem augmented_matrix_correct :
  ∃ x y : ℝ, system_of_equations x y ∧ is_augmented_matrix ![
    ![2, 1, 1],
    ![3, -2, 0]
  ] :=
sorry

end augmented_matrix_correct_l131_131187


namespace angle_PQRS_l131_131344

theorem angle_PQRS (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : 
  P = 206 := 
by
  sorry

end angle_PQRS_l131_131344


namespace compare_fractions_neg_l131_131726

theorem compare_fractions_neg : (- (2 / 3) > - (3 / 4)) :=
  sorry

end compare_fractions_neg_l131_131726


namespace zengshan_suanfa_tongzong_l131_131525

-- Definitions
variables (x y : ℝ)
variables (h1 : x = y + 5) (h2 : (1 / 2) * x = y - 5)

-- Theorem
theorem zengshan_suanfa_tongzong :
  x = y + 5 ∧ (1 / 2) * x = y - 5 :=
by
  -- Starting with the given hypotheses
  exact ⟨h1, h2⟩

end zengshan_suanfa_tongzong_l131_131525


namespace satisfies_natural_solution_l131_131870

theorem satisfies_natural_solution (m : ℤ) :
  (∃ x : ℕ, x = 6 / (m - 1)) → (m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 7) :=
by
  sorry

end satisfies_natural_solution_l131_131870


namespace solution_inequality_l131_131887

open Real

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 1)

-- State the theorem for the given proof problem
theorem solution_inequality :
  {x : ℝ | f x > 2} = {x : ℝ | x > 3 ∨ x < -1} :=
by
  sorry

end solution_inequality_l131_131887


namespace eval_expr_l131_131584

theorem eval_expr : (1 / (5^2)^4 * 5^11 * 2) = 250 := by
  sorry

end eval_expr_l131_131584


namespace group_C_questions_l131_131012

theorem group_C_questions (a b c : ℕ) (total_questions : ℕ) (h1 : a + b + c = 100)
  (h2 : b = 23)
  (h3 : a ≥ (6 * (a + 2 * b + 3 * c)) / 10)
  (h4 : 2 * b ≤ (25 * (a + 2 * b + 3 * c)) / 100)
  (h5 : 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c) :
  c = 1 :=
sorry

end group_C_questions_l131_131012


namespace combined_height_is_9_l131_131634

def barrys_reach : ℝ := 5 -- Barry can reach apples that are 5 feet high

def larrys_full_height : ℝ := 5 -- Larry's full height is 5 feet

def larrys_shoulder_height : ℝ := larrys_full_height * 0.8 -- Larry's shoulder height is 20% less than his full height

def combined_reach (b_reach : ℝ) (l_shoulder : ℝ) : ℝ := b_reach + l_shoulder

theorem combined_height_is_9 : combined_reach barrys_reach larrys_shoulder_height = 9 := by
  sorry

end combined_height_is_9_l131_131634


namespace base_four_product_l131_131981

def base_four_to_decimal (n : ℕ) : ℕ :=
  -- definition to convert base 4 to decimal, skipping details for now
  sorry

def decimal_to_base_four (n : ℕ) : ℕ :=
  -- definition to convert decimal to base 4, skipping details for now
  sorry

theorem base_four_product : 
  base_four_to_decimal 212 * base_four_to_decimal 13 = base_four_to_decimal 10322 :=
sorry

end base_four_product_l131_131981


namespace max_popsicles_l131_131562

theorem max_popsicles (budget : ℕ) (cost_single : ℕ) (popsicles_single : ℕ) (cost_box3 : ℕ) (popsicles_box3 : ℕ) (cost_box7 : ℕ) (popsicles_box7 : ℕ)
  (h_budget : budget = 10) (h_cost_single : cost_single = 1) (h_popsicles_single : popsicles_single = 1)
  (h_cost_box3 : cost_box3 = 3) (h_popsicles_box3 : popsicles_box3 = 3)
  (h_cost_box7 : cost_box7 = 4) (h_popsicles_box7 : popsicles_box7 = 7) :
  ∃ n, n = 16 :=
by
  sorry

end max_popsicles_l131_131562


namespace induction_proof_l131_131616

open Nat

noncomputable def S (n : ℕ) : ℚ :=
  match n with
  | 0     => 0
  | (n+1) => S n + 1 / ((n+1) * (n+2))

theorem induction_proof : ∀ n : ℕ, S n = n / (n + 1) := by
  intro n
  induction n with
  | zero => 
    -- Base case: S(1) = 1/2
    sorry
  | succ n ih =>
    -- Induction step: Assume S(n) = n / (n + 1), prove S(n+1) = (n+1) / (n+2)
    sorry

end induction_proof_l131_131616


namespace exterior_angle_regular_octagon_l131_131546

theorem exterior_angle_regular_octagon : 
  (∃ n : ℕ, n = 8 ∧ ∀ (i : ℕ), i < n → true) → 
  ∃ θ : ℝ, θ = 45 := by
  sorry

end exterior_angle_regular_octagon_l131_131546


namespace alcohol_percentage_new_mixture_l131_131542

/--
Given:
1. The initial mixture has 15 liters.
2. The mixture contains 20% alcohol.
3. 5 liters of water is added to the mixture.

Prove:
The percentage of alcohol in the new mixture is 15%.
-/
theorem alcohol_percentage_new_mixture :
  let initial_mixture_volume := 15 -- in liters
  let initial_alcohol_percentage := 20 / 100
  let initial_alcohol_volume := initial_alcohol_percentage * initial_mixture_volume
  let added_water_volume := 5 -- in liters
  let new_total_volume := initial_mixture_volume + added_water_volume
  let new_alcohol_percentage := (initial_alcohol_volume / new_total_volume) * 100
  new_alcohol_percentage = 15 := 
by
  -- Proof steps go here
  sorry

end alcohol_percentage_new_mixture_l131_131542


namespace john_has_18_blue_pens_l131_131755

variables (R B Bl : ℕ)

-- Conditions from the problem
def john_has_31_pens : Prop := R + B + Bl = 31
def black_pens_5_more_than_red : Prop := B = R + 5
def blue_pens_twice_black : Prop := Bl = 2 * B

theorem john_has_18_blue_pens :
  john_has_31_pens R B Bl ∧ black_pens_5_more_than_red R B ∧ blue_pens_twice_black B Bl →
  Bl = 18 :=
by
  sorry

end john_has_18_blue_pens_l131_131755


namespace min_students_same_place_l131_131789

-- Define the context of the problem
def classSize := 45
def numberOfChoices := 6

-- The proof statement
theorem min_students_same_place : 
  ∃ (n : ℕ), 8 ≤ n ∧ n = Nat.ceil (classSize / numberOfChoices) :=
by
  sorry

end min_students_same_place_l131_131789


namespace geometric_sequence_sum_l131_131787

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h_geo : ∀ n, a (n + 1) = (3 : ℝ) * ((-2 : ℝ) ^ n))
  (h_first : a 1 = 3)
  (h_ratio_ne_1 : -2 ≠ 1)
  (h_arith : 2 * a 3 = a 4 + a 5) :
  a 1 + a 2 + a 3 + a 4 + a 5 = 33 := 
sorry

end geometric_sequence_sum_l131_131787


namespace find_large_number_l131_131048

theorem find_large_number (L S : ℕ) 
  (h1 : L - S = 50000) 
  (h2 : L = 13 * S + 317) : 
  L = 54140 := 
sorry

end find_large_number_l131_131048


namespace sculpture_and_base_height_l131_131802

def height_sculpture_ft : ℕ := 2
def height_sculpture_in : ℕ := 10
def height_base_in : ℕ := 2

def total_height_in (ft : ℕ) (inch1 inch2 : ℕ) : ℕ :=
  (ft * 12) + inch1 + inch2

def total_height_ft (total_in : ℕ) : ℕ :=
  total_in / 12

theorem sculpture_and_base_height :
  total_height_ft (total_height_in height_sculpture_ft height_sculpture_in height_base_in) = 3 :=
by
  sorry

end sculpture_and_base_height_l131_131802


namespace solve_equation_l131_131291

noncomputable def equation (x : ℝ) : Prop :=
  -2 * x ^ 3 = (5 * x ^ 2 + 2) / (2 * x - 1)

theorem solve_equation (x : ℝ) :
  equation x ↔ (x = (1 + Real.sqrt 17) / 4 ∨ x = (1 - Real.sqrt 17) / 4) :=
by
  sorry

end solve_equation_l131_131291


namespace units_digits_no_match_l131_131783

theorem units_digits_no_match : ∀ x : ℕ, 1 ≤ x ∧ x ≤ 100 → (x % 10 ≠ (101 - x) % 10) :=
by
  intro x hx
  sorry

end units_digits_no_match_l131_131783


namespace rope_length_comparison_l131_131754

theorem rope_length_comparison
  (L : ℝ)
  (hL1 : L > 0) 
  (cut1 cut2 : ℝ)
  (hcut1 : cut1 = 0.3)
  (hcut2 : cut2 = 3) :
  L - cut1 > L - cut2 :=
by
  sorry

end rope_length_comparison_l131_131754


namespace interest_rate_is_5_percent_l131_131549

noncomputable def interest_rate_1200_loan (R : ℝ) : Prop :=
  let time := 3.888888888888889
  let principal_1000 := 1000
  let principal_1200 := 1200
  let rate_1000 := 0.03
  let total_interest := 350
  principal_1000 * rate_1000 * time + principal_1200 * (R / 100) * time = total_interest

theorem interest_rate_is_5_percent :
  interest_rate_1200_loan 5 :=
by
  sorry

end interest_rate_is_5_percent_l131_131549


namespace avg_choc_pieces_per_cookie_l131_131335

theorem avg_choc_pieces_per_cookie {cookies chips mms pieces : ℕ} 
  (h1 : cookies = 48) 
  (h2 : chips = 108) 
  (h3 : mms = chips / 3) 
  (h4 : pieces = chips + mms) : 
  pieces / cookies = 3 := 
by sorry

end avg_choc_pieces_per_cookie_l131_131335


namespace canteen_distance_l131_131444

theorem canteen_distance (r G B : ℝ) (d_g d_b : ℝ) (h_g : G = 600) (h_b : B = 800) (h_dg_db : d_g = d_b) : 
  d_g = 781 :=
by
  -- Proof to be completed
  sorry

end canteen_distance_l131_131444


namespace parabola_eqn_min_distance_l131_131496

theorem parabola_eqn (a b : ℝ) (h_a : 0 < a)
  (h_A : -a + b = 1)
  (h_B : 4 * a + 2 * b = 0) :
  (∀ x : ℝ,  y = a * x^2 + b * x) ↔ (∀ x : ℝ, y = (1/3) * x^2 - (2/3) * x) :=
by
  sorry

theorem min_distance (a b : ℝ) (h_a : 0 < a)
  (h_A : -a + b = 1)
  (h_B : 4 * a + 2 * b = 0)
  (line_eq : ∀ x, (y : ℝ) = x - 25/4) :
  (∀ P : ℝ × ℝ, ∃ P_min : ℝ × ℝ, P_min = (5/2, 5/12)) :=
by
  sorry

end parabola_eqn_min_distance_l131_131496


namespace a_plus_b_l131_131331

-- Definitions and conditions
def f (x : ℝ) (a b : ℝ) := a * x + b
def g (x : ℝ) := 3 * x - 7

theorem a_plus_b (a b : ℝ) (h : ∀ x : ℝ, g (f x a b) = 4 * x + 5) : a + b = 16 / 3 :=
by
  sorry

end a_plus_b_l131_131331


namespace pie_split_l131_131068

theorem pie_split (initial_pie : ℚ) (number_of_people : ℕ) (amount_taken_by_each : ℚ) 
  (h1 : initial_pie = 5/6) (h2 : number_of_people = 4) : amount_taken_by_each = 5/24 :=
by
  sorry

end pie_split_l131_131068


namespace find_first_number_l131_131671

theorem find_first_number (x : ℕ) : 
    (x + 32 + 53) / 3 = (21 + 47 + 22) / 3 + 3 ↔ x = 14 := by
  sorry

end find_first_number_l131_131671


namespace value_of_y_minus_x_l131_131180

theorem value_of_y_minus_x (x y : ℝ) (h1 : x + y = 520) (h2 : x / y = 0.75) : y - x = 74 :=
sorry

end value_of_y_minus_x_l131_131180


namespace gcd_subtraction_method_gcd_euclidean_algorithm_l131_131225

theorem gcd_subtraction_method (a b : ℕ) (h₁ : a = 72) (h₂ : b = 168) : Int.gcd a b = 24 := by
  sorry

theorem gcd_euclidean_algorithm (a b : ℕ) (h₁ : a = 98) (h₂ : b = 280) : Int.gcd a b = 14 := by
  sorry

end gcd_subtraction_method_gcd_euclidean_algorithm_l131_131225


namespace milk_cost_l131_131652

theorem milk_cost (x : ℝ) (h1 : 4 * 2.50 + 2 * x = 17) : x = 3.50 :=
by
  sorry

end milk_cost_l131_131652


namespace air_conditioner_sales_l131_131535

-- Definitions based on conditions
def ratio_air_conditioners_refrigerators : ℕ := 5
def ratio_refrigerators_air_conditioners : ℕ := 3
def difference_in_sales : ℕ := 54

-- The property to be proven: 
def number_of_air_conditioners : ℕ := 135

theorem air_conditioner_sales
  (r_ac : ℕ := ratio_air_conditioners_refrigerators) 
  (r_ref : ℕ := ratio_refrigerators_air_conditioners) 
  (diff : ℕ := difference_in_sales) 
  : number_of_air_conditioners = 135 := sorry

end air_conditioner_sales_l131_131535


namespace ratio_jordana_jennifer_10_years_l131_131617

-- Let's define the necessary terms and conditions:
def Jennifer_future_age := 30
def Jordana_current_age := 80
def years := 10

-- Define the ratio of ages function:
noncomputable def ratio_of_ages (future_age_jen : ℕ) (current_age_jord : ℕ) (yrs : ℕ) : ℚ :=
  (current_age_jord + yrs) / future_age_jen

-- The statement we need to prove:
theorem ratio_jordana_jennifer_10_years :
  ratio_of_ages Jennifer_future_age Jordana_current_age years = 3 := by
  sorry

end ratio_jordana_jennifer_10_years_l131_131617


namespace february_first_is_friday_l131_131398

-- Definition of conditions
def february_has_n_mondays (n : ℕ) : Prop := n = 3
def february_has_n_fridays (n : ℕ) : Prop := n = 5

-- The statement to prove
theorem february_first_is_friday (n_mondays n_fridays : ℕ) (h_mondays : february_has_n_mondays n_mondays) (h_fridays : february_has_n_fridays n_fridays) : 
  (1 : ℕ) % 7 = 5 :=
by
  sorry

end february_first_is_friday_l131_131398


namespace sum_has_minimum_term_then_d_positive_Sn_positive_then_increasing_sequence_l131_131559

def is_sum_of_arithmetic_sequence (S : ℕ → ℚ) (a₁ d : ℚ) :=
  ∀ n : ℕ, S n = n * a₁ + (n * (n - 1) / 2) * d

theorem sum_has_minimum_term_then_d_positive
  {S : ℕ → ℚ} {d a₁ : ℚ} (h : d ≠ 0)
  (hS : is_sum_of_arithmetic_sequence S a₁ d)
  (h_min : ∃ n : ℕ, ∀ m : ℕ, S n ≤ S m) :
  d > 0 :=
sorry

theorem Sn_positive_then_increasing_sequence
  {S : ℕ → ℚ} {d a₁ : ℚ} (h : d ≠ 0)
  (hS : is_sum_of_arithmetic_sequence S a₁ d)
  (h_pos : ∀ n : ℕ, S n > 0) :
  (∀ n : ℕ, S n < S (n + 1)) :=
sorry

end sum_has_minimum_term_then_d_positive_Sn_positive_then_increasing_sequence_l131_131559


namespace contrapositive_even_contrapositive_not_even_l131_131279

theorem contrapositive_even (x y : ℤ) : 
  (∃ a b : ℤ, x = 2*a ∧ y = 2*b)  → (∃ c : ℤ, x + y = 2*c) :=
sorry

theorem contrapositive_not_even (x y : ℤ) :
  (¬ ∃ c : ℤ, x + y = 2*c) → (¬ ∃ a b : ℤ, x = 2*a ∧ y = 2*b) :=
sorry

end contrapositive_even_contrapositive_not_even_l131_131279


namespace circle_diameter_percentage_l131_131676

theorem circle_diameter_percentage (d_R d_S : ℝ) 
    (h : π * (d_R / 2)^2 = 0.04 * π * (d_S / 2)^2) : 
    d_R = 0.4 * d_S :=
by
    sorry

end circle_diameter_percentage_l131_131676


namespace screen_to_body_ratio_increases_l131_131073

theorem screen_to_body_ratio_increases
  (a b m : ℝ)
  (h1 : a > b)
  (h2 : 0 < m)
  (h3 : m < 1) :
  (b + m) / (a + m) > b / a :=
by
  sorry

end screen_to_body_ratio_increases_l131_131073


namespace crowdfunding_successful_l131_131572

variable (highest_level second_level lowest_level total_amount : ℕ)
variable (x y z : ℕ)

noncomputable def crowdfunding_conditions (highest_level second_level lowest_level : ℕ) := 
  second_level = highest_level / 10 ∧ lowest_level = second_level / 10

noncomputable def total_raised (highest_level second_level lowest_level x y z : ℕ) :=
  highest_level * x + second_level * y + lowest_level * z

theorem crowdfunding_successful (h1 : highest_level = 5000) 
                                (h2 : crowdfunding_conditions highest_level second_level lowest_level) 
                                (h3 : total_amount = 12000) 
                                (h4 : y = 3) 
                                (h5 : z = 10) :
  total_raised highest_level second_level lowest_level x y z = total_amount → x = 2 := by
  sorry

end crowdfunding_successful_l131_131572


namespace cats_left_l131_131033

theorem cats_left (siamese house persian sold_first sold_second : ℕ) (h1 : siamese = 23) (h2 : house = 17) (h3 : persian = 29) (h4 : sold_first = 40) (h5 : sold_second = 12) :
  siamese + house + persian - sold_first - sold_second = 17 :=
by sorry

end cats_left_l131_131033


namespace annual_average_growth_rate_estimated_output_value_2006_l131_131103

-- First problem: Prove the annual average growth rate from 2003 to 2005
theorem annual_average_growth_rate (x : ℝ) (h : 6.4 * (1 + x)^2 = 10) : 
  x = 1/4 :=
by
  sorry

-- Second problem: Prove the estimated output value for 2006 given the annual growth rate
theorem estimated_output_value_2006 (x : ℝ) (output_2005 : ℝ) (h_growth : x = 1/4) (h_2005 : output_2005 = 10) : 
  output_2005 * (1 + x) = 12.5 :=
by 
  sorry

end annual_average_growth_rate_estimated_output_value_2006_l131_131103


namespace tim_minus_tom_l131_131259

def sales_tax_rate : ℝ := 0.07
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def city_tax_rate : ℝ := 0.05

noncomputable def tim_total : ℝ :=
  let price_with_tax := original_price * (1 + sales_tax_rate)
  price_with_tax * (1 - discount_rate)

noncomputable def tom_total : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let price_with_sales_tax := discounted_price * (1 + sales_tax_rate)
  price_with_sales_tax * (1 + city_tax_rate)

theorem tim_minus_tom : tim_total - tom_total = -4.82 := 
by sorry

end tim_minus_tom_l131_131259


namespace principal_amount_correct_l131_131277

-- Define the given conditions and quantities
def P : ℝ := 1054.76
def final_amount : ℝ := 1232.0
def rate1 : ℝ := 0.05
def rate2 : ℝ := 0.07
def rate3 : ℝ := 0.04

-- Define the statement we want to prove
theorem principal_amount_correct :
  final_amount = P * (1 + rate1) * (1 + rate2) * (1 + rate3) :=
sorry

end principal_amount_correct_l131_131277


namespace point_in_second_quadrant_l131_131206

-- Define the point coordinates in the Cartesian plane
def x_coord : ℤ := -8
def y_coord : ℤ := 2

-- Define the quadrants based on coordinate conditions
def first_quadrant : Prop := x_coord > 0 ∧ y_coord > 0
def second_quadrant : Prop := x_coord < 0 ∧ y_coord > 0
def third_quadrant : Prop := x_coord < 0 ∧ y_coord < 0
def fourth_quadrant : Prop := x_coord > 0 ∧ y_coord < 0

-- Proof statement: The point (-8, 2) lies in the second quadrant
theorem point_in_second_quadrant : second_quadrant :=
by
  sorry

end point_in_second_quadrant_l131_131206


namespace tank_length_l131_131536

theorem tank_length (W D : ℝ) (cost_per_sq_m total_cost : ℝ) (L : ℝ):
  W = 12 →
  D = 6 →
  cost_per_sq_m = 0.70 →
  total_cost = 520.8 →
  total_cost = cost_per_sq_m * ((2 * (W * D)) + (2 * (L * D)) + (L * W)) →
  L = 25 :=
by
  intros hW hD hCostPerSqM hTotalCost hEquation
  sorry

end tank_length_l131_131536


namespace class_weighted_average_l131_131177

theorem class_weighted_average
    (num_students : ℕ)
    (sect1_avg sect2_avg sect3_avg remainder_avg : ℝ)
    (sect1_pct sect2_pct sect3_pct remainder_pct : ℝ)
    (weight1 weight2 weight3 weight4 : ℝ)
    (h_total_students : num_students = 120)
    (h_sect1_avg : sect1_avg = 96.5)
    (h_sect2_avg : sect2_avg = 78.4)
    (h_sect3_avg : sect3_avg = 88.2)
    (h_remainder_avg : remainder_avg = 64.7)
    (h_sect1_pct : sect1_pct = 0.187)
    (h_sect2_pct : sect2_pct = 0.355)
    (h_sect3_pct : sect3_pct = 0.258)
    (h_remainder_pct : remainder_pct = 1 - (sect1_pct + sect2_pct + sect3_pct))
    (h_weight1 : weight1 = 0.35)
    (h_weight2 : weight2 = 0.25)
    (h_weight3 : weight3 = 0.30)
    (h_weight4 : weight4 = 0.10) :
    (sect1_avg * weight1 + sect2_avg * weight2 + sect3_avg * weight3 + remainder_avg * weight4) * 100 = 86 := 
sorry

end class_weighted_average_l131_131177


namespace min_total_weight_l131_131568

theorem min_total_weight (crates: Nat) (weight_per_crate: Nat) (h1: crates = 6) (h2: weight_per_crate ≥ 120): 
  crates * weight_per_crate ≥ 720 :=
by
  sorry

end min_total_weight_l131_131568


namespace green_more_than_blue_l131_131664

variable (B Y G : ℕ)

theorem green_more_than_blue
  (h_sum : B + Y + G = 126)
  (h_ratio : ∃ k : ℕ, B = 3 * k ∧ Y = 7 * k ∧ G = 8 * k) :
  G - B = 35 := by
  sorry

end green_more_than_blue_l131_131664


namespace total_amount_shared_l131_131731

-- Define the amounts for Ken and Tony based on the conditions
def ken_amt : ℤ := 1750
def tony_amt : ℤ := 2 * ken_amt

-- The proof statement that the total amount shared is $5250
theorem total_amount_shared : ken_amt + tony_amt = 5250 :=
by 
  sorry

end total_amount_shared_l131_131731


namespace ab_value_l131_131160

theorem ab_value (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 18) : a * b = -1 :=
by {
  sorry
}

end ab_value_l131_131160


namespace smallest_five_digit_congruent_to_three_mod_seventeen_l131_131662

theorem smallest_five_digit_congruent_to_three_mod_seventeen :
  ∃ (n : ℤ), 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 3 ∧ n = 10012 :=
by
  sorry

end smallest_five_digit_congruent_to_three_mod_seventeen_l131_131662


namespace tv_show_years_l131_131738

theorem tv_show_years (s1 s2 s3 : ℕ) (e1 e2 e3 : ℕ) (avg : ℕ) :
  s1 = 8 → e1 = 15 →
  s2 = 4 → e2 = 20 →
  s3 = 2 → e3 = 12 →
  avg = 16 →
  (s1 * e1 + s2 * e2 + s3 * e3) / avg = 14 := by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end tv_show_years_l131_131738


namespace helicopter_rental_cost_l131_131320

noncomputable def rentCost (hours_per_day : ℕ) (days : ℕ) (cost_per_hour : ℕ) : ℕ :=
  hours_per_day * days * cost_per_hour

theorem helicopter_rental_cost :
  rentCost 2 3 75 = 450 := 
by
  sorry

end helicopter_rental_cost_l131_131320


namespace statement_not_always_true_l131_131919

theorem statement_not_always_true 
  (a b c d : ℝ)
  (h1 : (a + b) / (3 * a - b) = (b + c) / (3 * b - c))
  (h2 : (b + c) / (3 * b - c) = (c + d) / (3 * c - d))
  (h3 : (c + d) / (3 * c - d) = (d + a) / (3 * d - a))
  (h4 : (d + a) / (3 * d - a) = (a + b) / (3 * a - b)) :
  a^2 + b^2 + c^2 + d^2 ≠ ab + bc + cd + da :=
by {
  sorry
}

end statement_not_always_true_l131_131919


namespace ratio_of_average_speed_to_still_water_speed_l131_131482

noncomputable def speed_of_current := 6
noncomputable def speed_in_still_water := 18
noncomputable def downstream_speed := speed_in_still_water + speed_of_current
noncomputable def upstream_speed := speed_in_still_water - speed_of_current
noncomputable def distance_each_way := 1
noncomputable def total_distance := 2 * distance_each_way
noncomputable def time_downstream := (distance_each_way : ℝ) / (downstream_speed : ℝ)
noncomputable def time_upstream := (distance_each_way : ℝ) / (upstream_speed : ℝ)
noncomputable def total_time := time_downstream + time_upstream
noncomputable def average_speed := (total_distance : ℝ) / (total_time : ℝ)
noncomputable def ratio_average_speed := (average_speed : ℝ) / (speed_in_still_water : ℝ)

theorem ratio_of_average_speed_to_still_water_speed :
  ratio_average_speed = (8 : ℝ) / (9 : ℝ) :=
sorry

end ratio_of_average_speed_to_still_water_speed_l131_131482


namespace ab_square_value_l131_131217

noncomputable def cyclic_quadrilateral (AX AY BX BY CX CY AB2 : ℝ) : Prop :=
  AX * AY = 6 ∧
  BX * BY = 5 ∧
  CX * CY = 4 ∧
  AB2 = 122 / 15

theorem ab_square_value :
  ∃ (AX AY BX BY CX CY : ℝ), cyclic_quadrilateral AX AY BX BY CX CY (122 / 15) :=
by
  sorry

end ab_square_value_l131_131217


namespace jackie_apples_l131_131245

variable (A J : ℕ)

-- Condition: Adam has 3 more apples than Jackie.
axiom h1 : A = J + 3

-- Condition: Adam has 9 apples.
axiom h2 : A = 9

-- Question: How many apples does Jackie have?
theorem jackie_apples : J = 6 :=
by
  -- We would normally the proof steps here, but we'll skip to the answer
  sorry

end jackie_apples_l131_131245


namespace ratio_of_areas_of_concentric_circles_l131_131633

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ)
  (h : (30 / 360) * C1 = (24 / 360) * C2) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 16 / 25 := by
  sorry

end ratio_of_areas_of_concentric_circles_l131_131633


namespace percentage_of_brand_z_l131_131905

/-- Define the initial and subsequent conditions for the fuel tank -/
def initial_fuel_tank : ℕ := 1
def first_stage_z_gasoline : ℚ := 1 / 4
def first_stage_y_gasoline : ℚ := 3 / 4
def second_stage_z_gasoline : ℚ := first_stage_z_gasoline / 2 + 1 / 2
def second_stage_y_gasoline : ℚ := first_stage_y_gasoline / 2
def final_stage_z_gasoline : ℚ := second_stage_z_gasoline / 2
def final_stage_y_gasoline : ℚ := second_stage_y_gasoline / 2 + 1 / 2

/-- Formal statement of the problem: Prove the percentage of Brand Z gasoline -/
theorem percentage_of_brand_z :
  ∃ (percentage : ℚ), percentage = (final_stage_z_gasoline / (final_stage_z_gasoline + final_stage_y_gasoline)) * 100 ∧ percentage = 31.25 :=
by {
  sorry
}

end percentage_of_brand_z_l131_131905


namespace bread_per_day_baguettes_per_day_croissants_per_day_l131_131815

-- Define the conditions
def loaves_per_hour : ℕ := 10
def hours_per_day : ℕ := 6
def baguettes_per_2hours : ℕ := 30
def croissants_per_75minutes : ℕ := 20

-- Conversion factors
def minutes_per_hour : ℕ := 60
def minutes_per_block : ℕ := 75
def blocks_per_75minutes : ℕ := 360 / 75

-- Proof statements
theorem bread_per_day :
  loaves_per_hour * hours_per_day = 60 := by sorry

theorem baguettes_per_day :
  (hours_per_day / 2) * baguettes_per_2hours = 90 := by sorry

theorem croissants_per_day :
  (blocks_per_75minutes * croissants_per_75minutes) = 80 := by sorry

end bread_per_day_baguettes_per_day_croissants_per_day_l131_131815


namespace ellipse_equation_correct_l131_131481

theorem ellipse_equation_correct :
  ∃ (a b h k : ℝ), 
    h = 4 ∧ 
    k = 0 ∧ 
    a = 10 + 2 * Real.sqrt 10 ∧ 
    b = Real.sqrt (101 + 20 * Real.sqrt 10) ∧ 
    (∀ x y : ℝ, (x, y) = (9, 6) → 
    ((x - h)^2 / a^2 + y^2 / b^2 = 1)) ∧
    (dist (4 - 3, 0) (4 + 3, 0) = 6) := 
sorry

end ellipse_equation_correct_l131_131481


namespace start_A_to_B_l131_131553

theorem start_A_to_B (x : ℝ)
  (A_to_C : x = 1000 * (1000 / 571.43) - 1000)
  (h1 : 1000 / (1000 - 600) = 1000 / (1000 - 428.57))
  (h2 : x = 1750 - 1000) :
  x = 750 :=
by
  rw [h2]
  sorry   -- Proof to be filled in.

end start_A_to_B_l131_131553


namespace alex_average_speed_l131_131148

def total_distance : ℕ := 48
def biking_time : ℕ := 6

theorem alex_average_speed : (total_distance / biking_time) = 8 := 
by
  sorry

end alex_average_speed_l131_131148


namespace find_initial_children_l131_131174

-- Definition of conditions
def initial_children_on_bus (X : ℕ) := 
  let final_children := (X + 40) - 60 
  final_children = 2

-- Theorem statement
theorem find_initial_children : 
  ∃ X : ℕ, initial_children_on_bus X ∧ X = 22 :=
by
  sorry

end find_initial_children_l131_131174


namespace value_of_x_add_y_not_integer_l131_131010

theorem value_of_x_add_y_not_integer (x y: ℝ) (h1: y = 3 * ⌊x⌋ + 4) (h2: y = 2 * ⌊x - 3⌋ + 7) (h3: ¬ ∃ n: ℤ, x = n): -8 < x + y ∧ x + y < -7 := 
sorry

end value_of_x_add_y_not_integer_l131_131010


namespace range_of_a_l131_131866

open Set

def real_intervals (a : ℝ) : Prop :=
  let S := {x : ℝ | (x - 2)^2 > 9}
  let T := Ioo a (a + 8)
  S ∪ T = univ → -3 < a ∧ a < -1

theorem range_of_a (a : ℝ) : real_intervals a :=
sorry

end range_of_a_l131_131866


namespace probability_all_different_digits_l131_131376

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l131_131376


namespace melanie_total_value_l131_131253

-- Define the initial number of dimes Melanie had
def initial_dimes : ℕ := 7

-- Define the number of dimes given by her dad
def dimes_from_dad : ℕ := 8

-- Define the number of dimes given by her mom
def dimes_from_mom : ℕ := 4

-- Calculate the total number of dimes Melanie has now
def total_dimes : ℕ := initial_dimes + dimes_from_dad + dimes_from_mom

-- Define the value of each dime in dollars
def value_per_dime : ℝ := 0.10

-- Calculate the total value of dimes in dollars
def total_value_in_dollars : ℝ := total_dimes * value_per_dime

-- The theorem states that the total value in dollars is 1.90
theorem melanie_total_value : total_value_in_dollars = 1.90 := 
by
  -- Using the established definitions, the goal follows directly.
  sorry

end melanie_total_value_l131_131253


namespace number_of_balls_in_last_box_l131_131060

noncomputable def box_question (b : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2010 → b i + b (i + 1) = 14 + i) ∧
  (b 1 + b 2011 = 1023)

theorem number_of_balls_in_last_box (b : ℕ → ℕ) (h : box_question b) : b 2011 = 1014 :=
by
  sorry

end number_of_balls_in_last_box_l131_131060


namespace total_nails_needed_l131_131475

-- Definitions based on problem conditions
def nails_per_plank : ℕ := 2
def planks_needed : ℕ := 2

-- Theorem statement: Prove that the total number of nails John needs is 4.
theorem total_nails_needed : nails_per_plank * planks_needed = 4 := by
  sorry

end total_nails_needed_l131_131475


namespace find_a_l131_131918

def set_A : Set ℝ := {x | x^2 + x - 6 = 0}

def set_B (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem find_a (a : ℝ) : set_A ∪ set_B a = set_A ↔ a ∈ ({0, 1/3, -1/2} : Set ℝ) := 
by
  sorry

end find_a_l131_131918


namespace sum_tens_ones_digit_l131_131095

theorem sum_tens_ones_digit (a : ℕ) (b : ℕ) (n : ℕ) (h : a - b = 3) :
  let d := (3^n)
  let ones_digit := d % 10
  let tens_digit := (d / 10) % 10
  ones_digit + tens_digit = 9 :=
by 
  let d := 3^17
  let ones_digit := d % 10
  let tens_digit := (d / 10) % 10
  sorry

end sum_tens_ones_digit_l131_131095


namespace fuel_tank_capacity_l131_131507

theorem fuel_tank_capacity
  (ethanol_A_fraction : ℝ)
  (ethanol_B_fraction : ℝ)
  (ethanol_total : ℝ)
  (fuel_A_volume : ℝ)
  (C : ℝ)
  (h1 : ethanol_A_fraction = 0.12)
  (h2 : ethanol_B_fraction = 0.16)
  (h3 : ethanol_total = 28)
  (h4 : fuel_A_volume = 99.99999999999999)
  (h5 : 0.12 * 99.99999999999999 + 0.16 * (C - 99.99999999999999) = 28) :
  C = 200 := 
sorry

end fuel_tank_capacity_l131_131507


namespace books_left_over_after_repacking_l131_131167

def initial_boxes : ℕ := 1430
def books_per_initial_box : ℕ := 42
def weight_per_book : ℕ := 200 -- in grams
def books_per_new_box : ℕ := 45
def max_weight_per_new_box : ℕ := 9000 -- in grams (9 kg)

def total_books : ℕ := initial_boxes * books_per_initial_box

theorem books_left_over_after_repacking :
  total_books % books_per_new_box = 30 :=
by
  -- Proof goes here
  sorry

end books_left_over_after_repacking_l131_131167


namespace average_speed_remaining_l131_131417

theorem average_speed_remaining (D : ℝ) : 
    (0.4 * D / 40 + 0.6 * D / S) = D / 50 → S = 60 :=
by 
  sorry

end average_speed_remaining_l131_131417


namespace find_angles_l131_131590

theorem find_angles (A B : ℝ) (h1 : A + B = 90) (h2 : A = 4 * B) : A = 72 ∧ B = 18 :=
by {
  sorry
}

end find_angles_l131_131590


namespace sequence_distinct_l131_131474

theorem sequence_distinct (f : ℕ → ℕ) (h : ∀ n : ℕ, f (f n) = f (n + 1) + f n) :
  ∀ i j : ℕ, i ≠ j → f i ≠ f j :=
by
  sorry

end sequence_distinct_l131_131474


namespace barry_shirt_discount_l131_131100

theorem barry_shirt_discount 
  (original_price : ℤ) 
  (discount_percent : ℤ) 
  (discounted_price : ℤ) 
  (h1 : original_price = 80) 
  (h2 : discount_percent = 15)
  (h3 : discounted_price = original_price - (discount_percent * original_price / 100)) : 
  discounted_price = 68 :=
sorry

end barry_shirt_discount_l131_131100


namespace largest_divisor_is_one_l131_131530

theorem largest_divisor_is_one (p q : ℤ) (hpq : p > q) (hp : p % 2 = 1) (hq : q % 2 = 0) :
  ∀ d : ℤ, (∀ p q : ℤ, p > q → p % 2 = 1 → q % 2 = 0 → d ∣ (p^2 - q^2)) → d = 1 :=
sorry

end largest_divisor_is_one_l131_131530


namespace groupA_forms_triangle_l131_131361

theorem groupA_forms_triangle (a b c : ℝ) (h1 : a = 13) (h2 : b = 12) (h3 : c = 20) : 
  a + b > c ∧ a + c > b ∧ b + c > a :=
by {
  sorry
}

end groupA_forms_triangle_l131_131361


namespace nails_needed_l131_131626

theorem nails_needed (nails_own nails_found nails_total_needed : ℕ) 
  (h1 : nails_own = 247) 
  (h2 : nails_found = 144) 
  (h3 : nails_total_needed = 500) : 
  nails_total_needed - (nails_own + nails_found) = 109 := 
by
  sorry

end nails_needed_l131_131626


namespace smallest_number_l131_131011

theorem smallest_number (a b c d : ℤ) (h1 : a = -2) (h2 : b = 0) (h3 : c = -3) (h4 : d = 1) : 
  min (min a b) (min c d) = c :=
by
  -- Proof goes here
  sorry

end smallest_number_l131_131011


namespace problem_statement_l131_131025

open Set

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}

theorem problem_statement :
  {y : ℤ | ∃ x ∈ A, y = |x + 1|} = {0, 1, 2, 3} :=
by
  sorry

end problem_statement_l131_131025


namespace age_of_25th_student_l131_131505

-- Definitions derived from problem conditions
def averageAgeClass (totalAge : ℕ) (totalStudents : ℕ) : ℕ := totalAge / totalStudents
def totalAgeGivenAverage (numStudents : ℕ) (averageAge : ℕ) : ℕ := numStudents * averageAge

-- Given conditions
def totalAgeOfAllStudents := 25 * 24
def totalAgeOf8Students := totalAgeGivenAverage 8 22
def totalAgeOf10Students := totalAgeGivenAverage 10 20
def totalAgeOf6Students := totalAgeGivenAverage 6 28
def totalAgeOf24Students := totalAgeOf8Students + totalAgeOf10Students + totalAgeOf6Students

-- The proof that the age of the 25th student is 56 years
theorem age_of_25th_student : totalAgeOfAllStudents - totalAgeOf24Students = 56 := by
  sorry

end age_of_25th_student_l131_131505


namespace expectation_fair_coin_5_tosses_l131_131929

noncomputable def fairCoinExpectation (n : ℕ) : ℚ :=
  n * (1/2)

theorem expectation_fair_coin_5_tosses :
  fairCoinExpectation 5 = 5 / 2 :=
by
  sorry

end expectation_fair_coin_5_tosses_l131_131929


namespace change_sum_equals_108_l131_131743

theorem change_sum_equals_108 :
  ∃ (amounts : List ℕ), (∀ a ∈ amounts, a < 100 ∧ ((a % 25 = 4) ∨ (a % 5 = 4))) ∧
    amounts.sum = 108 := 
by
  sorry

end change_sum_equals_108_l131_131743


namespace mirror_area_correct_l131_131709

noncomputable def width_of_mirror (frame_width : ℕ) (side_width : ℕ) : ℕ :=
  frame_width - 2 * side_width

noncomputable def height_of_mirror (frame_height : ℕ) (side_width : ℕ) : ℕ :=
  frame_height - 2 * side_width

noncomputable def area_of_mirror (frame_width : ℕ) (frame_height : ℕ) (side_width : ℕ) : ℕ :=
  width_of_mirror frame_width side_width * height_of_mirror frame_height side_width

theorem mirror_area_correct :
  area_of_mirror 50 70 7 = 2016 :=
by
  sorry

end mirror_area_correct_l131_131709


namespace depth_notation_l131_131998

theorem depth_notation (x y : ℤ) (hx : x = 9050) (hy : y = -10907) : -y = x :=
by
  sorry

end depth_notation_l131_131998


namespace prob_at_least_one_heart_spade_or_king_l131_131583

theorem prob_at_least_one_heart_spade_or_king :
  let total_cards := 52
  let hearts := 13
  let spades := 13
  let kings := 4
  let unique_hsk := hearts + spades + 2  -- Two unique kings from other suits
  let prob_not_hsk := (total_cards - unique_hsk) / total_cards
  let prob_not_hsk_two_draws := prob_not_hsk * prob_not_hsk
  let prob_at_least_one_hsk := 1 - prob_not_hsk_two_draws
  prob_at_least_one_hsk = 133 / 169 :=
by sorry

end prob_at_least_one_heart_spade_or_king_l131_131583


namespace total_games_across_leagues_l131_131264

-- Defining the conditions for the leagues
def leagueA_teams := 20
def leagueB_teams := 25
def leagueC_teams := 30

-- Function to calculate the number of games in a round-robin tournament
def number_of_games (n : ℕ) := n * (n - 1) / 2

-- Proposition to prove total games across all leagues
theorem total_games_across_leagues :
  number_of_games leagueA_teams + number_of_games leagueB_teams + number_of_games leagueC_teams = 925 := by
  sorry

end total_games_across_leagues_l131_131264


namespace total_students_eq_seventeen_l131_131654

theorem total_students_eq_seventeen 
    (N : ℕ)
    (initial_students : N - 1 = 16)
    (avg_first_day : 77 * (N - 1) = 77 * 16)
    (avg_second_day : 78 * N = 78 * N)
    : N = 17 :=
sorry

end total_students_eq_seventeen_l131_131654


namespace range_of_a_l131_131077

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (1/2) 2 → x₂ ∈ Set.Icc (1/2) 2 → (a / x₁ + x₁ * Real.log x₁ ≥ x₂^3 - x₂^2 - 3)) →
  a ∈ Set.Ici 1 :=
by
  sorry

end range_of_a_l131_131077


namespace price_of_each_orange_l131_131200

theorem price_of_each_orange 
  (x : ℕ)
  (a o : ℕ)
  (h1 : a + o = 20)
  (h2 : 40 * a + x * o = 1120)
  (h3 : (a + o - 10) * 52 = 1120 - 10 * x) :
  x = 60 :=
sorry

end price_of_each_orange_l131_131200


namespace solve_inequality_l131_131624

theorem solve_inequality (x : ℝ) :
  x * Real.log (x^2 + x + 1) / Real.log 10 < 0 ↔ x < -1 :=
sorry

end solve_inequality_l131_131624


namespace number_of_blue_lights_l131_131293

-- Conditions
def total_colored_lights : Nat := 95
def red_lights : Nat := 26
def yellow_lights : Nat := 37
def blue_lights : Nat := total_colored_lights - (red_lights + yellow_lights)

-- Statement we need to prove
theorem number_of_blue_lights : blue_lights = 32 := by
  sorry

end number_of_blue_lights_l131_131293


namespace tim_will_attend_game_probability_l131_131936

theorem tim_will_attend_game_probability :
  let P_rain := 0.60
  let P_sunny := 1 - P_rain
  let P_attends_given_rain := 0.25
  let P_attends_given_sunny := 0.70
  let P_rain_and_attends := P_rain * P_attends_given_rain
  let P_sunny_and_attends := P_sunny * P_attends_given_sunny
  (P_rain_and_attends + P_sunny_and_attends) = 0.43 :=
by
  sorry

end tim_will_attend_game_probability_l131_131936


namespace mn_equals_neg3_l131_131679

noncomputable def function_with_extreme_value (m n : ℝ) : Prop :=
  let f := λ x : ℝ => m * x^3 + n * x
  let f' := λ x : ℝ => 3 * m * x^2 + n
  f' (1 / m) = 0

theorem mn_equals_neg3 (m n : ℝ) (h : function_with_extreme_value m n) : m * n = -3 :=
sorry

end mn_equals_neg3_l131_131679


namespace polynomial_roots_arithmetic_progression_complex_root_l131_131792

theorem polynomial_roots_arithmetic_progression_complex_root :
  ∃ a : ℝ, (∀ (r d : ℂ), (r - d) + r + (r + d) = 9 → (r - d) * r + (r - d) * (r + d) + r * (r + d) = 30 → d^2 = -3 → 
  (r - d) * r * (r + d) = -a) → a = -12 :=
by sorry

end polynomial_roots_arithmetic_progression_complex_root_l131_131792


namespace total_cakes_served_l131_131446

def L : Nat := 5
def D : Nat := 6
def Y : Nat := 3
def T : Nat := L + D + Y

theorem total_cakes_served : T = 14 := by
  sorry

end total_cakes_served_l131_131446


namespace santino_total_fruits_l131_131435

theorem santino_total_fruits :
  let p := 2
  let m := 3
  let a := 4
  let o := 5
  let fp := 10
  let fm := 20
  let fa := 15
  let fo := 25
  p * fp + m * fm + a * fa + o * fo = 265 := by
  sorry

end santino_total_fruits_l131_131435


namespace intersection_M_N_l131_131374

noncomputable def M := {x : ℕ | x < 6}
noncomputable def N := {x : ℕ | x^2 - 11 * x + 18 < 0}
noncomputable def intersection := {x : ℕ | x ∈ M ∧ x ∈ N}

theorem intersection_M_N : intersection = {3, 4, 5} := by
  sorry

end intersection_M_N_l131_131374


namespace least_multiple_of_36_with_digit_product_multiple_of_9_l131_131686

def is_multiple_of_36 (n : ℕ) : Prop :=
  n % 36 = 0

def product_of_digits_multiple_of_9 (n : ℕ) : Prop :=
  ∃ d : List ℕ, (n = List.foldl (λ x y => x * 10 + y) 0 d) ∧ (List.foldl (λ x y => x * y) 1 d) % 9 = 0

theorem least_multiple_of_36_with_digit_product_multiple_of_9 : ∃ n : ℕ, is_multiple_of_36 n ∧ product_of_digits_multiple_of_9 n ∧ n = 36 :=
by
  sorry

end least_multiple_of_36_with_digit_product_multiple_of_9_l131_131686


namespace Andrena_more_than_Debelyn_l131_131157

-- Definitions based on the problem conditions
def Debelyn_initial := 20
def Debelyn_gift_to_Andrena := 2
def Christel_initial := 24
def Christel_gift_to_Andrena := 5
def Andrena_more_than_Christel := 2

-- Calculating the number of dolls each person has after the gifts
def Debelyn_final := Debelyn_initial - Debelyn_gift_to_Andrena
def Christel_final := Christel_initial - Christel_gift_to_Andrena
def Andrena_final := Christel_final + Andrena_more_than_Christel

-- The proof problem statement
theorem Andrena_more_than_Debelyn : Andrena_final - Debelyn_final = 3 := by
  sorry

end Andrena_more_than_Debelyn_l131_131157


namespace coordinates_A_B_l131_131571

theorem coordinates_A_B : 
  (∃ x, 7 * x + 2 * 3 = 41) ∧ (∃ y, 7 * (-5) + 2 * y = 41) → 
  ((∃ x, x = 5) ∧ (∃ y, y = 38)) :=
by
  sorry

end coordinates_A_B_l131_131571


namespace coloring_probability_l131_131384

-- Definition of the problem and its conditions
def num_cells := 16
def num_diags := 2
def chosen_diags := 7

-- Define the probability
noncomputable def prob_coloring_correct : ℚ :=
  (num_diags ^ chosen_diags : ℚ) / (num_diags ^ num_cells)

-- The Lean theorem statement
theorem coloring_probability : prob_coloring_correct = 1 / 512 := 
by 
  unfold prob_coloring_correct
  -- The proof steps would follow here (omitted)
  sorry

end coloring_probability_l131_131384


namespace train_ride_duration_is_360_minutes_l131_131796

-- Define the conditions given in the problem
def arrived_at_station_at_8 (t : ℕ) : Prop := t = 8 * 60
def train_departed_at_835 (t_depart : ℕ) : Prop := t_depart = 8 * 60 + 35
def train_arrived_at_215 (t_arrive : ℕ) : Prop := t_arrive = 14 * 60 + 15
def exited_station_at_3 (t_exit : ℕ) : Prop := t_exit = 15 * 60

-- Define the problem statement
theorem train_ride_duration_is_360_minutes (boarding alighting : ℕ) :
  arrived_at_station_at_8 boarding ∧ 
  train_departed_at_835 boarding ∧ 
  train_arrived_at_215 alighting ∧ 
  exited_station_at_3 alighting → 
  alighting - boarding = 360 := 
by
  sorry

end train_ride_duration_is_360_minutes_l131_131796


namespace smallest_angle_of_cyclic_quadrilateral_l131_131953

theorem smallest_angle_of_cyclic_quadrilateral (angles : ℝ → ℝ) (a d : ℝ) :
  -- Conditions
  (∀ n : ℕ, angles n = a + n * d) ∧ 
  (angles 3 = 140) ∧
  (a + d + (a + 3 * d) = 180) →
  -- Conclusion
  (a = 40) :=
by sorry

end smallest_angle_of_cyclic_quadrilateral_l131_131953


namespace problem_1_problem_2_l131_131110

def M : Set ℕ := {0, 1}

def A := { p : ℕ × ℕ | p.fst ∈ M ∧ p.snd ∈ M }

def B := { p : ℕ × ℕ | p.snd = 1 - p.fst }

theorem problem_1 : A = {(0,0), (0,1), (1,0), (1,1)} :=
by
  sorry

theorem problem_2 : 
  let AB := { p ∈ A | p ∈ B }
  AB = {(1,0), (0,1)} ∧
  {S : Set (ℕ × ℕ) | S ⊆ AB} = {∅, {(1,0)}, {(0,1)}, {(1,0), (0,1)}} :=
by
  sorry

end problem_1_problem_2_l131_131110


namespace tables_capacity_l131_131747

theorem tables_capacity (invited attended : ℕ) (didn't_show_up : ℕ) (tables : ℕ) (capacity : ℕ) 
    (h1 : invited = 24) (h2 : didn't_show_up = 10) (h3 : attended = invited - didn't_show_up) 
    (h4 : attended = 14) (h5 : tables = 2) : capacity = attended / tables :=
by {
  -- Proof goes here
  sorry
}

end tables_capacity_l131_131747


namespace geo_sequence_arithmetic_l131_131326

variable {d : ℝ} (hd : d ≠ 0)
variable {a : ℕ → ℝ} (ha : ∀ n, a (n+1) = a n + d)

-- Hypothesis that a_5, a_9, a_15 form a geometric sequence
variable (hgeo : a 9 ^ 2 = (a 9 - 4 * d) * (a 9 + 6 * d))

theorem geo_sequence_arithmetic (hd : d ≠ 0) (ha : ∀ n, a (n + 1) = a n + d) (hgeo : a 9 ^ 2 = (a 9 - 4 * d) * (a 9 + 6 * d)) :
  a 15 / a 9 = 3 / 2 :=
by
  sorry

end geo_sequence_arithmetic_l131_131326


namespace intersection_P_Q_eq_Q_l131_131835

-- Definitions of P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Statement to prove P ∩ Q = Q
theorem intersection_P_Q_eq_Q : P ∩ Q = Q := 
by 
  sorry

end intersection_P_Q_eq_Q_l131_131835


namespace range_of_m_l131_131099

-- Defining the conditions
variable (x m : ℝ)

-- The theorem statement
theorem range_of_m (h : ∀ x : ℝ, x < m → 2*x + 1 < 5) : m ≤ 2 := by
  sorry

end range_of_m_l131_131099


namespace solve_eqn_l131_131158

theorem solve_eqn (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  3 ^ x = 2 ^ x * y + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 5) := by
  sorry

end solve_eqn_l131_131158


namespace second_rice_price_l131_131708

theorem second_rice_price (P : ℝ) 
  (price_first : ℝ := 3.10) 
  (price_mixture : ℝ := 3.25) 
  (ratio_first_to_second : ℝ := 3 / 7) :
  (3 * price_first + 7 * P) / 10 = price_mixture → 
  P = 3.3142857142857145 :=
by
  sorry

end second_rice_price_l131_131708


namespace sufficient_not_necessary_l131_131548

theorem sufficient_not_necessary (a : ℝ) (h1 : a > 0) : (a^2 + a ≥ 0) ∧ ¬(a^2 + a ≥ 0 → a > 0) :=
by
  sorry

end sufficient_not_necessary_l131_131548


namespace heartsuit_properties_l131_131669

def heartsuit (x y : ℝ) : ℝ := abs (x - y)

theorem heartsuit_properties (x y : ℝ) :
  (heartsuit x y ≥ 0) ∧ (heartsuit x y > 0 ↔ x ≠ y) := by
  -- Proof will go here 
  sorry

end heartsuit_properties_l131_131669


namespace factorization_example_l131_131015

theorem factorization_example (x: ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
sorry

end factorization_example_l131_131015


namespace simplify_expression_l131_131429

theorem simplify_expression (h : (Real.pi / 2) < 2 ∧ 2 < Real.pi) : 
  Real.sqrt (1 - 2 * Real.sin 2 * Real.cos 2) = Real.sin 2 - Real.cos 2 :=
sorry

end simplify_expression_l131_131429


namespace ratio_of_liquid_rise_l131_131636

theorem ratio_of_liquid_rise
  (h1 h2 : ℝ) (r1 r2 rm : ℝ)
  (V1 V2 Vm : ℝ)
  (H1 : r1 = 4)
  (H2 : r2 = 9)
  (H3 : V1 = (1 / 3) * π * r1^2 * h1)
  (H4 : V2 = (1 / 3) * π * r2^2 * h2)
  (H5 : V1 = V2)
  (H6 : rm = 2)
  (H7 : Vm = (4 / 3) * π * rm^3)
  (H8 : h2 = h1 * (81 / 16))
  (h1' h2' : ℝ)
  (H9 : h1' = h1 + Vm / ((1 / 3) * π * r1^2))
  (H10 : h2' = h2 + Vm / ((1 / 3) * π * r2^2)) :
  (h1' - h1) / (h2' - h2) = 81 / 16 :=
sorry

end ratio_of_liquid_rise_l131_131636


namespace number_of_red_balls_l131_131319

def total_balls : ℕ := 50
def frequency_red_ball : ℝ := 0.7

theorem number_of_red_balls :
  ∃ n : ℕ, n = (total_balls : ℝ) * frequency_red_ball ∧ n = 35 :=
by
  sorry

end number_of_red_balls_l131_131319


namespace find_x_ineq_solution_l131_131657

open Set

theorem find_x_ineq_solution :
  {x : ℝ | (x - 2) / (x - 4) ≥ 3} = Ioc 4 5 := 
sorry

end find_x_ineq_solution_l131_131657


namespace managers_non_managers_ratio_l131_131309

theorem managers_non_managers_ratio
  (M N : ℕ)
  (h_ratio : M / N > 7 / 24)
  (h_max_non_managers : N = 27) :
  ∃ M, 8 ≤ M ∧ M / 27 > 7 / 24 :=
by
  sorry

end managers_non_managers_ratio_l131_131309


namespace annalise_spending_l131_131172

theorem annalise_spending
  (n_boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (cost_per_tissue : ℝ)
  (h1 : n_boxes = 10)
  (h2 : packs_per_box = 20)
  (h3 : tissues_per_pack = 100)
  (h4 : cost_per_tissue = 0.05) :
  n_boxes * packs_per_box * tissues_per_pack * cost_per_tissue = 1000 := 
  by
  sorry

end annalise_spending_l131_131172


namespace base8_to_base10_4513_l131_131357

theorem base8_to_base10_4513 : (4 * 8^3 + 5 * 8^2 + 1 * 8^1 + 3 * 8^0 = 2379) :=
by
  sorry

end base8_to_base10_4513_l131_131357


namespace slope_range_l131_131614

theorem slope_range {A : ℝ × ℝ} (k : ℝ) : 
  A = (1, 1) → (0 < 1 - k ∧ 1 - k < 2) → -1 < k ∧ k < 1 :=
by
  sorry

end slope_range_l131_131614


namespace sum_of_squares_l131_131387

theorem sum_of_squares (x : ℤ) (h : (x + 1) ^ 2 - x ^ 2 = 199) : x ^ 2 + (x + 1) ^ 2 = 19801 :=
sorry

end sum_of_squares_l131_131387


namespace mark_owe_triple_amount_l131_131323

theorem mark_owe_triple_amount (P : ℝ) (r : ℝ) (t : ℕ) (hP : P = 2000) (hr : r = 0.04) :
  (1 + r)^t > 3 → t = 30 :=
by
  intro h
  norm_cast at h
  sorry

end mark_owe_triple_amount_l131_131323


namespace compare_neg_two_powers_l131_131066

theorem compare_neg_two_powers : (-2)^3 = -2^3 := by sorry

end compare_neg_two_powers_l131_131066


namespace coefficient_of_x9_in_expansion_l131_131022

-- Definitions as given in the problem
def binomial_expansion_coeff (n k : ℕ) (a b : ℤ) : ℤ :=
  (Nat.choose n k) * a^(n - k) * b^k

-- Mathematically equivalent statement in Lean 4
theorem coefficient_of_x9_in_expansion : binomial_expansion_coeff 10 9 (-2) 1 = -20 :=
by
  sorry

end coefficient_of_x9_in_expansion_l131_131022


namespace number_of_females_l131_131485

theorem number_of_females (total_people : ℕ) (avg_age_total : ℕ) 
  (avg_age_males : ℕ) (avg_age_females : ℕ) (females : ℕ) :
  total_people = 140 → avg_age_total = 24 →
  avg_age_males = 21 → avg_age_females = 28 → 
  females = 60 :=
by
  intros h1 h2 h3 h4
  -- Using the given conditions
  sorry

end number_of_females_l131_131485


namespace contrapositive_of_square_inequality_l131_131287

theorem contrapositive_of_square_inequality (x y : ℝ) :
  (x^2 > y^2 → x > y) ↔ (x ≤ y → x^2 ≤ y^2) :=
by
  sorry

end contrapositive_of_square_inequality_l131_131287


namespace samantha_erased_length_l131_131710

/--
Samantha drew a line that was originally 1 meter (100 cm) long, and then it was erased until the length was 90 cm.
This theorem proves that the amount erased was 10 cm.
-/
theorem samantha_erased_length : 
  let original_length := 100 -- original length in cm
  let final_length := 90 -- final length in cm
  original_length - final_length = 10 := 
by
  sorry

end samantha_erased_length_l131_131710


namespace find_b_l131_131204

theorem find_b (a b c : ℝ) (h₁ : c = 3)
  (h₂ : -a / 3 = c)
  (h₃ : -a / 3 = 1 + a + b + c) :
  b = -16 :=
by
  -- The solution steps are not necessary to include here.
  sorry

end find_b_l131_131204


namespace monkeys_bananas_l131_131972

theorem monkeys_bananas (c₁ c₂ c₃ : ℕ) (h1 : ∀ (k₁ k₂ k₃ : ℕ), k₁ = c₁ → k₂ = c₂ → k₃ = c₃ → 4 * (k₁ / 3 + k₂ / 6 + k₃ / 18) = 2 * (k₁ / 6 + k₂ / 3 + k₃ / 18) ∧ 2 * (k₁ / 6 + k₂ / 3 + k₃ / 18) = k₁ / 6 + k₂ / 6 + k₃ / 6)
  (h2 : c₃ % 6 = 0) (h3 : 4 * (c₁ / 3 + c₂ / 6 + c₃ / 18) < 2 * (c₁ / 6 + c₂ / 3 + c₃ / 18 + 1)) :
  c₁ + c₂ + c₃ = 2352 :=
sorry

end monkeys_bananas_l131_131972


namespace find_f_neg_two_l131_131786

def is_even_function (f : ℝ → ℝ) (h : ℝ → ℝ) := ∀ x, h (-x) = h x

theorem find_f_neg_two (f : ℝ → ℝ) (h : ℝ → ℝ) (hx : ∀ x, h x = f (2*x) + x)
  (h_even : is_even_function f h) 
  (h_f_two : f 2 = 1) : 
  f (-2) = 3 :=
  by
    sorry

end find_f_neg_two_l131_131786


namespace prob_all_pass_prob_at_least_one_pass_most_likely_event_l131_131213

noncomputable def probability_A := 2 / 5
noncomputable def probability_B := 3 / 4
noncomputable def probability_C := 1 / 3
noncomputable def prob_none_pass := (1 - probability_A) * (1 - probability_B) * (1 - probability_C)
noncomputable def prob_one_pass := 
  (probability_A * (1 - probability_B) * (1 - probability_C)) +
  ((1 - probability_A) * probability_B * (1 - probability_C)) +
  ((1 - probability_A) * (1 - probability_B) * probability_C)
noncomputable def prob_two_pass := 
  (probability_A * probability_B * (1 - probability_C)) +
  (probability_A * (1 - probability_B) * probability_C) +
  ((1 - probability_A) * probability_B * probability_C)

-- Prove that the probability that all three candidates pass is 1/10
theorem prob_all_pass : probability_A * probability_B * probability_C = 1 / 10 := by
  sorry

-- Prove that the probability that at least one candidate passes is 9/10
theorem prob_at_least_one_pass : 1 - prob_none_pass = 9 / 10 := by
  sorry

-- Prove that the most likely event of passing is exactly one candidate passing with probability 5/12
theorem most_likely_event : prob_one_pass > prob_two_pass ∧ prob_one_pass > probability_A * probability_B * probability_C ∧ prob_one_pass > prob_none_pass ∧ prob_one_pass = 5 / 12 := by
  sorry

end prob_all_pass_prob_at_least_one_pass_most_likely_event_l131_131213


namespace ribbon_length_ratio_l131_131037

theorem ribbon_length_ratio (original_length reduced_length : ℕ) (h1 : original_length = 55) (h2 : reduced_length = 35) : 
  (original_length / Nat.gcd original_length reduced_length) = 11 ∧
  (reduced_length / Nat.gcd original_length reduced_length) = 7 := 
  by
    sorry

end ribbon_length_ratio_l131_131037


namespace simplified_sum_l131_131286

def exp1 := -( -1 ^ 2006 )
def exp2 := -( -1 ^ 2007 )
def exp3 := -( 1 ^ 2008 )
def exp4 := -( -1 ^ 2009 )

theorem simplified_sum : 
  exp1 + exp2 + exp3 + exp4 = 0 := 
by 
  sorry

end simplified_sum_l131_131286


namespace pair_basis_of_plane_l131_131051

def vector_space := Type
variable (V : Type) [AddCommGroup V] [Module ℝ V]

variables (e1 e2 : V)
variable (h_basis : LinearIndependent ℝ ![e1, e2])
variable (hne : e1 ≠ 0 ∧ e2 ≠ 0)

theorem pair_basis_of_plane
  (v1 v2 : V)
  (hv1 : v1 = e1 + e2)
  (hv2 : v2 = e1 - e2) :
  LinearIndependent ℝ ![v1, v2] :=
sorry

end pair_basis_of_plane_l131_131051


namespace find_expression_l131_131455

theorem find_expression (E a : ℝ) (h1 : (E + (3 * a - 8)) / 2 = 84) (h2 : a = 32) : E = 80 :=
by
  -- Proof to be filled in here
  sorry

end find_expression_l131_131455


namespace water_leakage_l131_131865

theorem water_leakage (initial_quarts : ℚ) (remaining_gallons : ℚ)
  (conversion_rate : ℚ) (expected_leakage : ℚ) :
  initial_quarts = 4 ∧ remaining_gallons = 0.33 ∧ conversion_rate = 4 ∧ 
  expected_leakage = 2.68 →
  initial_quarts - remaining_gallons * conversion_rate = expected_leakage :=
by 
  sorry

end water_leakage_l131_131865


namespace expression_result_zero_l131_131419

theorem expression_result_zero (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = y + 1) : 
  (x + 1 / x) * (y - 1 / y) = 0 := 
by sorry

end expression_result_zero_l131_131419


namespace gcd_g102_g103_l131_131500

def g (x : ℕ) : ℕ := x^2 - x + 2007

theorem gcd_g102_g103 : 
  Nat.gcd (g 102) (g 103) = 3 :=
by
  sorry

end gcd_g102_g103_l131_131500


namespace negation_of_all_squares_positive_l131_131080

theorem negation_of_all_squares_positive :
  ¬ (∀ x : ℝ, x * x > 0) ↔ ∃ x : ℝ, x * x ≤ 0 :=
by sorry

end negation_of_all_squares_positive_l131_131080


namespace D_72_eq_22_l131_131104

def D(n : ℕ) : ℕ :=
  if n = 72 then 22 else 0 -- the actual function logic should define D properly

theorem D_72_eq_22 : D 72 = 22 :=
  by sorry

end D_72_eq_22_l131_131104


namespace initial_students_count_l131_131973

theorem initial_students_count (n : ℕ) (W : ℝ) :
  (W = n * 28) →
  (W + 4 = (n + 1) * 27.2) →
  n = 29 :=
by
  intros hW hw_avg
  -- Proof goes here
  sorry

end initial_students_count_l131_131973


namespace scientists_from_usa_l131_131054

theorem scientists_from_usa (total_scientists : ℕ)
  (from_europe : ℕ)
  (from_canada : ℕ)
  (h1 : total_scientists = 70)
  (h2 : from_europe = total_scientists / 2)
  (h3 : from_canada = total_scientists / 5) :
  (total_scientists - from_europe - from_canada) = 21 :=
by
  sorry

end scientists_from_usa_l131_131054


namespace intersection_of_sets_union_of_complement_and_set_l131_131278

def set1 := { x : ℝ | -1 < x ∧ x < 2 }
def set2 := { x : ℝ | x > 0 }
def complement_set2 := { x : ℝ | x ≤ 0 }
def intersection_set := { x : ℝ | 0 < x ∧ x < 2 }
def union_set := { x : ℝ | x < 2 }

theorem intersection_of_sets : 
  { x : ℝ | x ∈ set1 ∧ x ∈ set2 } = intersection_set := 
by 
  sorry

theorem union_of_complement_and_set : 
  { x : ℝ | x ∈ complement_set2 ∨ x ∈ set1 } = union_set := 
by 
  sorry

end intersection_of_sets_union_of_complement_and_set_l131_131278


namespace exist_alpha_beta_l131_131980

variables {a b : ℝ} {f : ℝ → ℝ}

-- Assume that f has the Intermediate Value Property (for simplicity, define it as a predicate)
def intermediate_value_property (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ k ∈ Set.Icc (min (f a) (f b)) (max (f a) (f b)),
    ∃ c ∈ Set.Ioo a b, f c = k

-- Assume the conditions from the problem
variables (h_ivp : intermediate_value_property f a b) (h_sign_change : f a * f b < 0)

-- The theorem we need to prove
theorem exist_alpha_beta (hivp : intermediate_value_property f a b) (hsign : f a * f b < 0) :
  ∃ α β, a < α ∧ α < β ∧ β < b ∧ f α + f β = f α * f β :=
sorry

end exist_alpha_beta_l131_131980


namespace robert_arrival_time_l131_131179

def arrival_time (T : ℕ) : Prop :=
  ∃ D : ℕ, D = 10 * (12 - T) ∧ D = 15 * (13 - T)

theorem robert_arrival_time : arrival_time 15 :=
by
  sorry

end robert_arrival_time_l131_131179


namespace hearing_news_probability_l131_131191

noncomputable def probability_of_hearing_news : ℚ :=
  let broadcast_cycle := 30 -- total time in minutes for each broadcast cycle
  let news_duration := 5  -- duration of each news broadcast in minutes
  news_duration / broadcast_cycle

theorem hearing_news_probability : probability_of_hearing_news = 1 / 6 := by
  sorry

end hearing_news_probability_l131_131191


namespace smallest_four_digit_divisible_by_six_l131_131986

theorem smallest_four_digit_divisible_by_six : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m, m ≥ 1000 ∧ m < n → ¬ (m % 6 = 0) :=
by
  sorry

end smallest_four_digit_divisible_by_six_l131_131986


namespace tony_fish_after_ten_years_l131_131276

theorem tony_fish_after_ten_years :
  let initial_fish := 6
  let x := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  let y := [4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
  (List.foldl (fun acc ⟨add, die⟩ => acc + add - die) initial_fish (List.zip x y)) = 34 := 
by
  sorry

end tony_fish_after_ten_years_l131_131276


namespace share_of_e_l131_131418

variable (E F : ℝ)
variable (D : ℝ := (5/3) * E)
variable (D_alt : ℝ := (1/2) * F)
variable (E_alt : ℝ := (3/2) * F)
variable (profit : ℝ := 25000)

theorem share_of_e (h1 : D = (5/3) * E) (h2 : D = (1/2) * F) (h3 : E = (3/2) * F) :
  (E / ((5/2) * F + (3/2) * F + F)) * profit = 7500 :=
by
  sorry

end share_of_e_l131_131418


namespace seating_solution_l131_131247

/-- 
Imagine Abby, Bret, Carl, and Dana are seated in a row of four seats numbered from 1 to 4.
Joe observes them and declares:

- "Bret is sitting next to Dana" (False)
- "Carl is between Abby and Dana" (False)

Further, it is known that Abby is in seat #2.

Who is seated in seat #3? 
-/

def seating_problem : Prop :=
  ∃ (seats : ℕ → ℕ),
  (¬ (seats 1 = 1 ∧ seats 1 = 4 ∨ seats 4 = 1 ∧ seats 4 = 4)) ∧
  (¬ (seats 3 > seats 1 ∧ seats 3 < seats 2 ∨ seats 3 > seats 2 ∧ seats 3 < seats 1)) ∧
  (seats 2 = 2) →
  (seats 3 = 3)

theorem seating_solution : seating_problem :=
sorry

end seating_solution_l131_131247


namespace proof1_proof2_l131_131791

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  |a * x - 2| - |x + 2|

-- Statement for proof 1
theorem proof1 (x : ℝ)
  (a : ℝ) (h : a = 2) (hx : f 2 x ≤ 1) : -1/3 ≤ x ∧ x ≤ 5 :=
sorry

-- Statement for proof 2
theorem proof2 (a : ℝ)
  (h : ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4) : a = 1 ∨ a = -1 :=
sorry

end proof1_proof2_l131_131791


namespace earphone_cost_correct_l131_131414

-- Given conditions
def mean_expenditure : ℕ := 500

def expenditure_mon : ℕ := 450
def expenditure_tue : ℕ := 600
def expenditure_wed : ℕ := 400
def expenditure_thu : ℕ := 500
def expenditure_sat : ℕ := 550
def expenditure_sun : ℕ := 300

def pen_cost : ℕ := 30
def notebook_cost : ℕ := 50

-- Goal: cost of the earphone
def total_expenditure_week : ℕ := 7 * mean_expenditure
def expenditure_6days : ℕ := expenditure_mon + expenditure_tue + expenditure_wed + expenditure_thu + expenditure_sat + expenditure_sun
def expenditure_fri : ℕ := total_expenditure_week - expenditure_6days
def expenditure_fri_items : ℕ := pen_cost + notebook_cost
def earphone_cost : ℕ := expenditure_fri - expenditure_fri_items

theorem earphone_cost_correct :
  earphone_cost = 620 :=
by
  sorry

end earphone_cost_correct_l131_131414


namespace A_inter_B_eq_A_A_union_B_l131_131069

-- Definitions for sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + 3 * a = (a + 3) * x}
def B : Set ℝ := {x | x^2 + 3 = 4 * x}

-- Proof problem for part (1)
theorem A_inter_B_eq_A (a : ℝ) : (A a ∩ B = A a) ↔ (a = 1 ∨ a = 3) :=
by
  sorry

-- Proof problem for part (2)
theorem A_union_B (a : ℝ) : A a ∪ B = if a = 1 then {1, 3} else if a = 3 then {1, 3} else {a, 1, 3} :=
by
  sorry

end A_inter_B_eq_A_A_union_B_l131_131069


namespace sum_even_if_product_odd_l131_131459

theorem sum_even_if_product_odd (a b : ℤ) (h : (a * b) % 2 = 1) : (a + b) % 2 = 0 := 
by
  sorry

end sum_even_if_product_odd_l131_131459


namespace prob_neither_alive_l131_131463

/-- Define the probability that a man will be alive for 10 more years -/
def prob_man_alive : ℚ := 1 / 4

/-- Define the probability that a wife will be alive for 10 more years -/
def prob_wife_alive : ℚ := 1 / 3

/-- Prove that the probability that neither the man nor his wife will be alive for 10 more years is 1/2 -/
theorem prob_neither_alive (p_man_alive p_wife_alive : ℚ)
    (h1 : p_man_alive = prob_man_alive) (h2 : p_wife_alive = prob_wife_alive) :
    (1 - p_man_alive) * (1 - p_wife_alive) = 1 / 2 :=
by
  sorry

end prob_neither_alive_l131_131463


namespace polygon_interior_angles_sum_l131_131138

theorem polygon_interior_angles_sum (n : ℕ) (hn : 180 * (n - 2) = 1980) : 180 * (n + 4 - 2) = 2700 :=
by
  sorry

end polygon_interior_angles_sum_l131_131138


namespace total_cost_proof_l131_131061

noncomputable def cost_of_4kg_mangos_3kg_rice_5kg_flour (M R F : ℝ) : ℝ :=
  4 * M + 3 * R + 5 * F

theorem total_cost_proof
  (M R F : ℝ)
  (h1 : 10 * M = 24 * R)
  (h2 : 6 * F = 2 * R)
  (h3 : F = 22) :
  cost_of_4kg_mangos_3kg_rice_5kg_flour M R F = 941.6 :=
  sorry

end total_cost_proof_l131_131061


namespace parallelogram_angle_sum_l131_131265

theorem parallelogram_angle_sum (ABCD : Type) (A B C D : ABCD) 
  (angle : ABCD → ℝ) (h_parallelogram : true) (h_B : angle B = 60) :
  ¬ (angle C + angle A = 180) :=
sorry

end parallelogram_angle_sum_l131_131265


namespace daniel_paid_more_l131_131665

noncomputable def num_slices : ℕ := 10
noncomputable def plain_cost : ℕ := 10
noncomputable def truffle_extra_cost : ℕ := 5
noncomputable def total_cost : ℕ := plain_cost + truffle_extra_cost
noncomputable def cost_per_slice : ℝ := total_cost / num_slices

noncomputable def truffle_slices_cost : ℝ := 5 * cost_per_slice
noncomputable def plain_slices_cost : ℝ := 5 * cost_per_slice

noncomputable def daniel_cost : ℝ := 5 * cost_per_slice + 2 * cost_per_slice
noncomputable def carl_cost : ℝ := 3 * cost_per_slice

noncomputable def payment_difference : ℝ := daniel_cost - carl_cost

theorem daniel_paid_more : payment_difference = 6 :=
by 
  sorry

end daniel_paid_more_l131_131665


namespace solve_for_y_l131_131185

theorem solve_for_y (x y : ℝ) (h : 5 * x - y = 6) : y = 5 * x - 6 :=
sorry

end solve_for_y_l131_131185


namespace solution_set_of_inequality_l131_131875

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x + 1) * (1 - 2 * x) > 0} = {x : ℝ | -1 / 3 < x ∧ x < 1 / 2} := 
by 
  sorry

end solution_set_of_inequality_l131_131875


namespace cauchy_schwarz_inequality_l131_131968

theorem cauchy_schwarz_inequality 
  (a b a1 b1 : ℝ) : ((a * a1 + b * b1) ^ 2 ≤ (a^2 + b^2) * (a1^2 + b1^2)) :=
 by sorry

end cauchy_schwarz_inequality_l131_131968


namespace max_fans_theorem_l131_131050

noncomputable def max_distinct_fans : ℕ :=
  let num_sectors := 6
  let total_configurations := 2 ^ num_sectors
  -- Configurations unchanged by flipping
  let unchanged_configurations := 8
  -- Subtracting unchanged from total and then divide by 2 to account for symmetric duplicates
  -- then add back the unchanged configurations
  (total_configurations - unchanged_configurations) / 2 + unchanged_configurations

theorem max_fans_theorem : max_distinct_fans = 36 := by
  sorry

end max_fans_theorem_l131_131050


namespace product_of_first_three_terms_l131_131490

/--
  The eighth term of an arithmetic sequence is 20.
  If the difference between two consecutive terms is 2,
  prove that the product of the first three terms of the sequence is 480.
-/
theorem product_of_first_three_terms (a d : ℕ) (h_d : d = 2) (h_eighth_term : a + 7 * d = 20) :
  (a * (a + d) * (a + 2 * d) = 480) :=
by
  sorry

end product_of_first_three_terms_l131_131490


namespace find_y_for_slope_l131_131248

theorem find_y_for_slope (y : ℝ) :
  let R := (-3, 9)
  let S := (3, y)
  let slope := (S.2 - R.2) / (S.1 - R.1)
  slope = -2 ↔ y = -3 :=
by
  simp [slope]
  sorry

end find_y_for_slope_l131_131248


namespace problem_statement_l131_131957

theorem problem_statement (x y : ℚ) (h1 : x + y = 500) (h2 : x / y = 4 / 5) : y - x = 500 / 9 := 
by
  sorry

end problem_statement_l131_131957


namespace correct_value_two_decimal_places_l131_131503

theorem correct_value_two_decimal_places (x : ℝ) 
  (h1 : 8 * x + 8 = 56) : 
  (x / 8) + 7 = 7.75 :=
sorry

end correct_value_two_decimal_places_l131_131503


namespace aaronTotalOwed_l131_131538

def monthlyPayment : ℝ := 100
def numberOfMonths : ℕ := 12
def interestRate : ℝ := 0.1

def totalCostWithoutInterest : ℝ := monthlyPayment * (numberOfMonths : ℝ)
def interestAmount : ℝ := totalCostWithoutInterest * interestRate
def totalAmountOwed : ℝ := totalCostWithoutInterest + interestAmount

theorem aaronTotalOwed : totalAmountOwed = 1320 := by
  sorry

end aaronTotalOwed_l131_131538


namespace handed_out_apples_l131_131184

def total_apples : ℤ := 96
def pies : ℤ := 9
def apples_per_pie : ℤ := 6
def apples_for_pies : ℤ := pies * apples_per_pie
def apples_handed_out : ℤ := total_apples - apples_for_pies

theorem handed_out_apples : apples_handed_out = 42 := by
  sorry

end handed_out_apples_l131_131184


namespace min_value_frac_inv_l131_131108

noncomputable def min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) : ℝ :=
  1 / a + 1 / b

theorem min_value_frac_inv (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 12) :
  min_value a b h1 h2 h3 = 1 / 3 :=
sorry

end min_value_frac_inv_l131_131108


namespace min_xy_min_x_plus_y_l131_131354

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : xy ≥ 4 := sorry

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : x + y ≥ 9 / 2 := sorry

end min_xy_min_x_plus_y_l131_131354


namespace inequality_holds_l131_131818

theorem inequality_holds (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 12 → x^2 + 25 + |x^3 - 5 * x^2| ≥ a * x) ↔ a ≤ 2.5 := 
by
  sorry

end inequality_holds_l131_131818


namespace fraction_eq_zero_iff_x_eq_2_l131_131499

theorem fraction_eq_zero_iff_x_eq_2 (x : ℝ) : (x - 2) / (x + 2) = 0 ↔ x = 2 := by sorry

end fraction_eq_zero_iff_x_eq_2_l131_131499


namespace angle_sum_acutes_l131_131966

theorem angle_sum_acutes (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
  (h_condition : |Real.sin α - 1/2| + Real.sqrt ((Real.tan β - 1)^2) = 0) : 
  α + β = π * 5/12 :=
by sorry

end angle_sum_acutes_l131_131966


namespace larger_number_is_sixty_three_l131_131579

theorem larger_number_is_sixty_three (x y : ℕ) (h1 : x + y = 84) (h2 : y = 3 * x) : y = 63 :=
  sorry

end larger_number_is_sixty_three_l131_131579


namespace salary_increase_l131_131231

variable (S : ℝ) -- Robert's original salary
variable (P : ℝ) -- Percentage increase after decrease in decimal form

theorem salary_increase (h1 : 0.5 * S * (1 + P) = 0.75 * S) : P = 0.5 := 
by 
  sorry

end salary_increase_l131_131231


namespace range_of_a_l131_131895

noncomputable def operation (x y : ℝ) := x * (1 - y)

theorem range_of_a
  (a : ℝ)
  (hx : ∀ x : ℝ, operation (x - a) (x + a) < 1) :
  -1/2 < a ∧ a < 3/2 := by
  sorry

end range_of_a_l131_131895


namespace volume_is_correct_l131_131941

noncomputable def volume_of_target_cube (V₁ : ℝ) (A₂ : ℝ) : ℝ :=
  if h₁ : V₁ = 8 then
    let s₁ := (8 : ℝ)^(1/3)
    let A₁ := 6 * s₁^2
    if h₂ : A₂ = 2 * A₁ then
      let s₂ := (A₂ / 6)^(1/2)
      let V₂ := s₂^3
      V₂
    else 0
  else 0

theorem volume_is_correct : volume_of_target_cube 8 48 = 16 * Real.sqrt 2 :=
by
  sorry

end volume_is_correct_l131_131941


namespace divide_P_Q_l131_131874

noncomputable def sequence_of_ones (n : ℕ) : ℕ := (10 ^ n - 1) / 9

theorem divide_P_Q (n : ℕ) (h : 1997 ∣ sequence_of_ones n) :
  1997 ∣ (sequence_of_ones (n + 1) * (10^(3*n) + 9 * 10^(2*n) + 9 * 10^n + 7)) ∧
  1997 ∣ (sequence_of_ones (n + 1) * (10^(3*(n + 1)) + 9 * 10^(2*(n + 1)) + 9 * 10^(n + 1) + 7)) := 
by
  sorry

end divide_P_Q_l131_131874


namespace min_value_a_plus_8b_min_value_a_plus_8b_min_l131_131798

theorem min_value_a_plus_8b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  a + 8 * b ≥ 9 :=
by sorry

-- The minimum value is 9 (achievable at specific values of a and b)
theorem min_value_a_plus_8b_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  ∃ a b, a > 0 ∧ b > 0 ∧ 2 * a * b = a + 2 * b ∧ a + 8 * b = 9 :=
by sorry

end min_value_a_plus_8b_min_value_a_plus_8b_min_l131_131798


namespace Malou_first_quiz_score_l131_131695

variable (score1 score2 score3 : ℝ)

theorem Malou_first_quiz_score (h1 : score1 = 90) (h2 : score2 = 92) (h_avg : (score1 + score2 + score3) / 3 = 91) : score3 = 91 := by
  sorry

end Malou_first_quiz_score_l131_131695


namespace each_boy_makes_14_dollars_l131_131408

noncomputable def victor_shrimp_caught := 26
noncomputable def austin_shrimp_caught := victor_shrimp_caught - 8
noncomputable def brian_shrimp_caught := (victor_shrimp_caught + austin_shrimp_caught) / 2
noncomputable def total_shrimp_caught := victor_shrimp_caught + austin_shrimp_caught + brian_shrimp_caught
noncomputable def money_made := (total_shrimp_caught / 11) * 7
noncomputable def each_boys_earnings := money_made / 3

theorem each_boy_makes_14_dollars : each_boys_earnings = 14 := by
  sorry

end each_boy_makes_14_dollars_l131_131408


namespace repeated_three_digit_divisible_101_l131_131079

theorem repeated_three_digit_divisible_101 (abc : ℕ) (h1 : 100 ≤ abc) (h2 : abc < 1000) :
  (1000000 * abc + 1000 * abc + abc) % 101 = 0 :=
by
  sorry

end repeated_three_digit_divisible_101_l131_131079


namespace yonderland_license_plates_l131_131739

/-!
# Valid License Plates in Yonderland

A valid license plate in Yonderland consists of three letters followed by four digits. 

We are tasked with determining the number of valid license plates possible under this format.
-/

def num_letters : ℕ := 26
def num_digits : ℕ := 10
def letter_combinations : ℕ := num_letters ^ 3
def digit_combinations : ℕ := num_digits ^ 4
def total_combinations : ℕ := letter_combinations * digit_combinations

theorem yonderland_license_plates : total_combinations = 175760000 := by
  sorry

end yonderland_license_plates_l131_131739


namespace sacks_filled_l131_131638

theorem sacks_filled (pieces_per_sack : ℕ) (total_pieces : ℕ) (h1 : pieces_per_sack = 20) (h2 : total_pieces = 80) : (total_pieces / pieces_per_sack) = 4 :=
by {
  sorry
}

end sacks_filled_l131_131638


namespace groom_age_proof_l131_131995

theorem groom_age_proof (G B : ℕ) (h1 : B = G + 19) (h2 : G + B = 185) : G = 83 :=
by
  sorry

end groom_age_proof_l131_131995


namespace number_of_n_with_odd_tens_digit_in_square_l131_131554

def ends_in_3_or_7 (n : ℕ) : Prop :=
  n % 10 = 3 ∨ n % 10 = 7

def tens_digit_odd (n : ℕ) : Prop :=
  ((n * n / 10) % 10) % 2 = 1

theorem number_of_n_with_odd_tens_digit_in_square :
  ∀ n ∈ {n : ℕ | n ≤ 50 ∧ ends_in_3_or_7 n}, ¬tens_digit_odd n :=
by 
  sorry

end number_of_n_with_odd_tens_digit_in_square_l131_131554


namespace linear_price_item_func_l131_131224

noncomputable def price_item_func (x : ℝ) : Prop :=
  ∃ (y : ℝ), y = - (1/4) * x + 50 ∧ 0 < x ∧ x < 200

theorem linear_price_item_func : ∀ x, price_item_func x ↔ (∃ y, y = - (1/4) * x + 50 ∧ 0 < x ∧ x < 200) :=
by
  sorry

end linear_price_item_func_l131_131224


namespace definitely_incorrect_conclusions_l131_131773

theorem definitely_incorrect_conclusions (a b c : ℝ) (x1 x2 : ℝ) 
  (h1 : a * x1^2 + b * x1 + c = 0) 
  (h2 : a * x2^2 + b * x2 + c = 0)
  (h3 : x1 > 0) 
  (h4 : x2 > 0) 
  (h5 : x1 + x2 = -b / a) 
  (h6 : x1 * x2 = c / a) : 
  (a > 0 ∧ b > 0 ∧ c > 0) = false ∧ 
  (a < 0 ∧ b < 0 ∧ c < 0) = false ∧ 
  (a > 0 ∧ b < 0 ∧ c < 0) = true ∧ 
  (a < 0 ∧ b > 0 ∧ c > 0) = true :=
sorry

end definitely_incorrect_conclusions_l131_131773


namespace find_n_l131_131667

theorem find_n (n : ℕ) (hn_pos : 0 < n) (hn_greater_30 : 30 < n) 
  (divides : (4 * n - 1) ∣ 2002 * n) : n = 36 := 
by
  sorry

end find_n_l131_131667


namespace family_boys_girls_l131_131528

theorem family_boys_girls (B G : ℕ) 
  (h1 : B - 1 = G) 
  (h2 : B = 2 * (G - 1)) : 
  B = 4 ∧ G = 3 := 
by {
  sorry
}

end family_boys_girls_l131_131528


namespace sin_cos_identity_l131_131117

theorem sin_cos_identity (α : ℝ) (hα_cos : Real.cos α = 3/5) (hα_sin : Real.sin α = 4/5) : Real.sin α + 2 * Real.cos α = 2 :=
by
  -- Proof omitted
  sorry

end sin_cos_identity_l131_131117


namespace range_of_m_l131_131413

def quadratic_function (m : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 + m * x + 1 = 0 → false)

def ellipse_condition (m : ℝ) : Prop :=
  0 < m

theorem range_of_m (m : ℝ) :
  (quadratic_function m ∨ ellipse_condition m) ∧ ¬ (quadratic_function m ∧ ellipse_condition m) →
  m ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi 2 :=
by
  sorry

end range_of_m_l131_131413


namespace maximum_partial_sum_l131_131745

theorem maximum_partial_sum (a : ℕ → ℕ) (d : ℕ) (S : ℕ → ℕ)
    (h_arith_seq : ∀ n, a n = a 0 + n * d)
    (h8_13 : 3 * a 8 = 5 * a 13)
    (h_pos : a 0 > 0)
    (h_sn_def : ∀ n, S n = n * (2 * a 0 + (n - 1) * d) / 2) :
  S 20 = max (max (S 10) (S 11)) (max (S 20) (S 21)) := 
sorry

end maximum_partial_sum_l131_131745


namespace train_seat_count_l131_131019

theorem train_seat_count (t : ℝ) (h1 : 0.20 * t = 0.2 * t)
  (h2 : 0.60 * t = 0.6 * t) (h3 : 30 + 0.20 * t + 0.60 * t = t) : t = 150 :=
by
  sorry

end train_seat_count_l131_131019


namespace candies_remaining_after_yellow_eaten_l131_131565

theorem candies_remaining_after_yellow_eaten :
  let red_candies := 40
  let yellow_candies := 3 * red_candies - 20
  let blue_candies := yellow_candies / 2
  red_candies + blue_candies = 90 :=
by
  sorry

end candies_remaining_after_yellow_eaten_l131_131565


namespace minimum_value_of_x_plus_y_l131_131181

theorem minimum_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (x - 1) * (y - 1) = 1) : x + y = 4 :=
sorry

end minimum_value_of_x_plus_y_l131_131181


namespace sum_of_acute_angles_l131_131056

theorem sum_of_acute_angles (α β : ℝ) (t : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_tanα : Real.tan α = 2 / t) (h_tanβ : Real.tan β = t / 15)
  (h_min : 10 * Real.tan α + 3 * Real.tan β = 4) :
  α + β = π / 4 :=
sorry

end sum_of_acute_angles_l131_131056


namespace symmetrical_point_l131_131576

-- Definition of symmetry with respect to the x-axis
def symmetrical (x y: ℝ) : ℝ × ℝ := (x, -y)

-- Coordinates of the original point A
def A : ℝ × ℝ := (-2, 3)

-- Coordinates of the symmetrical point
def symmetrical_A : ℝ × ℝ := symmetrical (-2) 3

-- The theorem we want to prove
theorem symmetrical_point :
  symmetrical_A = (-2, -3) :=
by
  -- Provide the proof here
  sorry

end symmetrical_point_l131_131576


namespace find_k_l131_131016

noncomputable def vec (a b : ℝ) : ℝ × ℝ := (a, b)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem find_k
  (k : ℝ)
  (a b c : ℝ × ℝ)
  (ha : a = vec 3 1)
  (hb : b = vec 1 3)
  (hc : c = vec k (-2))
  (h_perp : dot_product (vec (a.1 - c.1) (a.2 - c.2)) (vec (a.1 - b.1) (a.2 - b.2)) = 0) :
  k = 0 :=
sorry

end find_k_l131_131016


namespace painting_prices_l131_131403

theorem painting_prices (P : ℝ) (h₀ : 55000 = 3.5 * P - 500) : 
  P = 15857.14 :=
by
  -- P represents the average price of the previous three paintings.
  -- Given the condition: 55000 = 3.5 * P - 500
  -- We need to prove: P = 15857.14
  sorry

end painting_prices_l131_131403


namespace quadrant_of_angle_l131_131232

-- Definitions for conditions
def sin_pos_cos_pos (α : ℝ) : Prop := (Real.sin α) * (Real.cos α) > 0

-- The theorem to prove
theorem quadrant_of_angle (α : ℝ) (h : sin_pos_cos_pos α) : 
  (0 < α ∧ α < π / 2) ∨ (π < α ∧ α < 3 * π / 2) :=
sorry

end quadrant_of_angle_l131_131232


namespace maximum_watchman_demand_l131_131441

theorem maximum_watchman_demand (bet_loss : ℕ) (bet_win : ℕ) (x : ℕ) 
  (cond_bet_loss : bet_loss = 100)
  (cond_bet_win : bet_win = 100) :
  x < 200 :=
by
  have h₁ : bet_loss = 100 := cond_bet_loss
  have h₂ : bet_win = 100 := cond_bet_win
  sorry

end maximum_watchman_demand_l131_131441


namespace find_divisor_l131_131325

theorem find_divisor 
  (dividend : ℤ)
  (quotient : ℤ)
  (remainder : ℤ)
  (divisor : ℤ)
  (h : dividend = (divisor * quotient) + remainder)
  (h_dividend : dividend = 474232)
  (h_quotient : quotient = 594)
  (h_remainder : remainder = -968) :
  divisor = 800 :=
sorry

end find_divisor_l131_131325


namespace new_plan_cost_correct_l131_131477

def oldPlanCost : ℝ := 150
def rateIncrease : ℝ := 0.3
def newPlanCost : ℝ := oldPlanCost * (1 + rateIncrease) 

theorem new_plan_cost_correct : newPlanCost = 195 := by
  sorry

end new_plan_cost_correct_l131_131477


namespace hash_triple_l131_131270

def hash (N : ℝ) : ℝ := 0.5 * (N^2) + 1

theorem hash_triple  : hash (hash (hash 4)) = 862.125 :=
by {
  sorry
}

end hash_triple_l131_131270


namespace length_breadth_difference_l131_131032

theorem length_breadth_difference (b l : ℕ) (h1 : b = 5) (h2 : l * b = 15 * b) : l - b = 10 :=
by
  sorry

end length_breadth_difference_l131_131032


namespace floor_e_eq_2_l131_131064

theorem floor_e_eq_2 : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_eq_2_l131_131064


namespace jason_pokemon_cards_l131_131588

theorem jason_pokemon_cards :
  ∀ (initial_cards trade_benny_lost trade_benny_gain trade_sean_lost trade_sean_gain give_to_brother : ℕ),
  initial_cards = 5 →
  trade_benny_lost = 2 →
  trade_benny_gain = 3 →
  trade_sean_lost = 3 →
  trade_sean_gain = 4 →
  give_to_brother = 2 →
  initial_cards - trade_benny_lost + trade_benny_gain - trade_sean_lost + trade_sean_gain - give_to_brother = 5 :=
by
  intros
  sorry

end jason_pokemon_cards_l131_131588


namespace total_pawns_left_l131_131364

  -- Definitions of initial conditions
  def initial_pawns_in_chess : Nat := 8
  def kennedy_pawns_lost : Nat := 4
  def riley_pawns_lost : Nat := 1

  -- Theorem statement to prove the total number of pawns left
  theorem total_pawns_left : (initial_pawns_in_chess - kennedy_pawns_lost) + (initial_pawns_in_chess - riley_pawns_lost) = 11 := by
    sorry
  
end total_pawns_left_l131_131364


namespace integer_solution_unique_l131_131434

theorem integer_solution_unique
  (a b c d : ℤ)
  (h : a^2 + 5 * b^2 - 2 * c^2 - 2 * c * d - 3 * d^2 = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end integer_solution_unique_l131_131434


namespace triangle_area_l131_131877

/-- Given a triangle with a perimeter of 20 cm and an inradius of 2.5 cm,
prove that its area is 25 cm². -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ)
  (h1 : perimeter = 20) (h2 : inradius = 2.5) :
  area = 25 :=
by
  sorry

end triangle_area_l131_131877


namespace quadratic_single_root_pos_value_l131_131316

theorem quadratic_single_root_pos_value (m : ℝ) (h1 : (6 * m)^2 - 4 * 1 * 2 * m = 0) : m = 2 / 9 :=
sorry

end quadratic_single_root_pos_value_l131_131316


namespace chris_earnings_total_l131_131855

-- Define the conditions
variable (hours_week1 hours_week2 : ℕ) (wage_per_hour earnings_diff : ℝ)
variable (hours_week1_val : hours_week1 = 18)
variable (hours_week2_val : hours_week2 = 30)
variable (earnings_diff_val : earnings_diff = 65.40)
variable (constant_wage : wage_per_hour > 0)

-- Theorem statement
theorem chris_earnings_total (total_earnings : ℝ) :
  hours_week2 - hours_week1 = 12 →
  wage_per_hour = earnings_diff / 12 →
  total_earnings = (hours_week1 + hours_week2) * wage_per_hour →
  total_earnings = 261.60 :=
by
  intros h1 h2 h3
  sorry

end chris_earnings_total_l131_131855


namespace cos_300_eq_half_l131_131928

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l131_131928


namespace Maria_bought_7_roses_l131_131962

theorem Maria_bought_7_roses
  (R : ℕ)
  (h1 : ∀ f : ℕ, 6 * f = 6 * f)
  (h2 : ∀ r : ℕ, ∃ d : ℕ, r = R ∧ d = 3)
  (h3 : 6 * R + 18 = 60) : R = 7 := by
  sorry

end Maria_bought_7_roses_l131_131962


namespace quadratic_properties_l131_131389

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties 
  (a b c : ℝ) (ha : a ≠ 0) (h_passes_through : quadratic_function a b c 0 = 1) (h_unique_zero : quadratic_function a b c (-1) = 0) :
  quadratic_function a b c = quadratic_function 1 2 1 ∧ 
  (∀ k, ∃ g,
    (k ≤ -2 → g = k + 3) ∧ 
    (-2 < k ∧ k ≤ 6 → g = -((k^2 - 4*k) / 4)) ∧ 
    (6 < k → g = 9 - 2*k)) :=
sorry

end quadratic_properties_l131_131389


namespace distance_between_parallel_lines_l131_131706

theorem distance_between_parallel_lines :
  ∀ {x y : ℝ}, 
  (3 * x - 4 * y + 1 = 0) → (3 * x - 4 * y + 7 = 0) → 
  ∃ d, d = (6 : ℝ) / 5 :=
by 
  sorry

end distance_between_parallel_lines_l131_131706


namespace two_digit_number_l131_131346

theorem two_digit_number (x y : ℕ) (h1 : y = x + 4) (h2 : (10 * x + y) * (x + y) = 208) :
  10 * x + y = 26 :=
sorry

end two_digit_number_l131_131346


namespace rationalize_denominator_simplify_l131_131570

theorem rationalize_denominator_simplify :
  let a : ℝ := 3
  let b : ℝ := 2
  let c : ℝ := 1
  let d : ℝ := 2
  ∀ (x y z : ℝ), 
  (x = 3 * Real.sqrt 2) → 
  (y = 3) → 
  (z = Real.sqrt 3) → 
  (x / (y - z) = (3 * Real.sqrt 2 + Real.sqrt 6) / 2) :=
by
  sorry

end rationalize_denominator_simplify_l131_131570


namespace smallest_square_area_l131_131018

theorem smallest_square_area (a b c d : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 4) (h4 : d = 5) :
  ∃ s, s^2 = 81 ∧ (a ≤ s ∧ b ≤ s ∧ c ≤ s ∧ d ≤ s ∧ (a + c) ≤ s ∧ (b + d) ≤ s) :=
sorry

end smallest_square_area_l131_131018


namespace heptagon_angle_sum_l131_131304

theorem heptagon_angle_sum 
  (angle_A angle_B angle_C angle_D angle_E angle_F angle_G : ℝ) 
  (h : angle_A + angle_B + angle_C + angle_D + angle_E + angle_F + angle_G = 540) :
  angle_A + angle_B + angle_C + angle_D + angle_E + angle_F + angle_G = 540 :=
by
  sorry

end heptagon_angle_sum_l131_131304


namespace base_k_number_to_decimal_l131_131715

theorem base_k_number_to_decimal (k : ℕ) (h : 4 ≤ k) : 1 * k^2 + 3 * k + 2 = 30 ↔ k = 4 := by
  sorry

end base_k_number_to_decimal_l131_131715


namespace min_next_score_to_increase_avg_l131_131766

def Liam_initial_scores : List ℕ := [72, 85, 78, 66, 90, 82]

def current_average (scores: List ℕ) : ℚ :=
  (scores.sum / scores.length : ℚ)

def next_score_requirement (initial_scores: List ℕ) (desired_increase: ℚ) : ℚ :=
  let current_avg := current_average initial_scores
  let desired_avg := current_avg + desired_increase
  let total_tests := initial_scores.length + 1
  let total_required := desired_avg * total_tests
  total_required - initial_scores.sum

theorem min_next_score_to_increase_avg :
  next_score_requirement Liam_initial_scores 5 = 115 := by
  sorry

end min_next_score_to_increase_avg_l131_131766


namespace project_completion_advance_l131_131072

variables (a : ℝ) -- efficiency of each worker (units of work per day)
variables (total_days : ℕ) (initial_workers added_workers : ℕ) (fraction_completed : ℝ)
variables (initial_days remaining_days total_initial_work total_remaining_work total_workers_efficiency : ℝ)

-- Conditions
def conditions : Prop :=
  total_days = 100 ∧
  initial_workers = 10 ∧
  initial_days = 30 ∧
  fraction_completed = 1 / 5 ∧
  added_workers = 10 ∧
  total_initial_work = initial_workers * initial_days * a * 5 ∧ 
  total_remaining_work = total_initial_work - (initial_workers * initial_days * a) ∧
  total_workers_efficiency = (initial_workers + added_workers) * a ∧
  remaining_days = total_remaining_work / total_workers_efficiency

-- Proof statement
theorem project_completion_advance (h : conditions a total_days initial_workers added_workers fraction_completed initial_days remaining_days total_initial_work total_remaining_work total_workers_efficiency) :
  total_days - (initial_days + remaining_days) = 10 :=
  sorry

end project_completion_advance_l131_131072


namespace base4_sum_conversion_to_base10_l131_131989

theorem base4_sum_conversion_to_base10 :
  let n1 := 2213
  let n2 := 2703
  let n3 := 1531
  let base := 4
  let sum_base4 := n1 + n2 + n3 
  let sum_base10 :=
    (1 * base^4) + (0 * base^3) + (2 * base^2) + (5 * base^1) + (1 * base^0)
  sum_base10 = 309 :=
by
  sorry

end base4_sum_conversion_to_base10_l131_131989


namespace probability_not_blue_marble_l131_131478

-- Define the conditions
def odds_for_blue_marble : ℕ := 5
def odds_for_not_blue_marble : ℕ := 6
def total_outcomes := odds_for_blue_marble + odds_for_not_blue_marble

-- Define the question and statement to be proven
theorem probability_not_blue_marble :
  (odds_for_not_blue_marble : ℚ) / total_outcomes = 6 / 11 :=
by
  -- skipping the proof step as per instruction
  sorry

end probability_not_blue_marble_l131_131478


namespace number_is_48_l131_131299

theorem number_is_48 (x : ℝ) (h : (1/4) * x + 15 = 27) : x = 48 :=
by sorry

end number_is_48_l131_131299


namespace problem_xyz_l131_131451

noncomputable def distance_from_intersection_to_side_CD (s : ℝ) : ℝ :=
  s * ((8 - Real.sqrt 15) / 8)

theorem problem_xyz
  (s : ℝ)
  (ABCD_is_square : (0 ≤ s))
  (X_is_intersection: ∃ (X : ℝ × ℝ), (X.1^2 + X.2^2 = s^2) ∧ ((X.1 - s)^2 + X.2^2 = (s / 2)^2))
  : distance_from_intersection_to_side_CD s = (s * (8 - Real.sqrt 15) / 8) :=
sorry

end problem_xyz_l131_131451


namespace line_parameterization_l131_131189

theorem line_parameterization (s m : ℝ) :
  (∃ t : ℝ, ∀ x y : ℝ, (x = s + 2 * t ∧ y = 3 + m * t) ↔ y = 5 * x - 7) →
  s = 2 ∧ m = 10 :=
by
  intro h_conditions
  sorry

end line_parameterization_l131_131189


namespace seungho_more_marbles_l131_131857

variable (S H : ℕ)

-- Seungho gave 273 marbles to Hyukjin
def given_marbles : ℕ := 273

-- After giving 273 marbles, Seungho has 477 more marbles than Hyukjin
axiom marbles_condition : S - given_marbles = (H + given_marbles) + 477

theorem seungho_more_marbles (S H : ℕ) (marbles_condition : S - 273 = (H + 273) + 477) : S = H + 1023 :=
by
  sorry

end seungho_more_marbles_l131_131857


namespace trigonometric_expression_value_l131_131086

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 5 := by
  sorry

end trigonometric_expression_value_l131_131086


namespace rate_of_stream_l131_131661

def effectiveSpeedDownstream (v : ℝ) : ℝ := 36 + v
def effectiveSpeedUpstream (v : ℝ) : ℝ := 36 - v

theorem rate_of_stream (v : ℝ) (hf1 : effectiveSpeedUpstream v = 3 * effectiveSpeedDownstream v) : v = 18 := by
  sorry

end rate_of_stream_l131_131661


namespace not_equiv_2_pi_six_and_11_pi_six_l131_131202

def polar_equiv (r θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * ↑k * Real.pi

theorem not_equiv_2_pi_six_and_11_pi_six :
  ¬ polar_equiv 2 (Real.pi / 6) (11 * Real.pi / 6) := 
sorry

end not_equiv_2_pi_six_and_11_pi_six_l131_131202


namespace least_sum_of_exponents_520_l131_131049

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def sum_of_distinct_powers_of_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 2^a + 2^b = n

theorem least_sum_of_exponents_520 :
  ∀ (a b : ℕ), sum_of_distinct_powers_of_two 520 → a ≠ b → 2^a + 2^b = 520 → a + b = 12 :=
by
  sorry

end least_sum_of_exponents_520_l131_131049


namespace earnings_difference_is_200_l131_131701

noncomputable def difference_in_earnings : ℕ :=
  let asking_price := 5200
  let maintenance_cost := asking_price / 10
  let first_offer_earnings := asking_price - maintenance_cost
  let headlight_cost := 80
  let tire_cost := 3 * headlight_cost
  let total_repair_cost := headlight_cost + tire_cost
  let second_offer_earnings := asking_price - total_repair_cost
  second_offer_earnings - first_offer_earnings

theorem earnings_difference_is_200 : difference_in_earnings = 200 := by
  sorry

end earnings_difference_is_200_l131_131701


namespace ellipse_focus_and_axes_l131_131697

theorem ellipse_focus_and_axes (m : ℝ) :
  (∃ a b : ℝ, (a > b) ∧ (mx^2 + y^2 = 1) ∧ (a^2 = 1) ∧ (b^2 = 1/m) ∧ (2 * a = 3 * 2 * b)) → 
  m = 4 / 9 :=
by
  intro h
  rcases h with ⟨a, b, hab, h_eq, ha, hb, ha_b_eq⟩
  sorry

end ellipse_focus_and_axes_l131_131697


namespace total_liters_needed_to_fill_two_tanks_l131_131992

theorem total_liters_needed_to_fill_two_tanks (capacity : ℕ) (liters_first_tank : ℕ) (liters_second_tank : ℕ) (percent_filled : ℕ) :
  liters_first_tank = 300 → 
  liters_second_tank = 450 → 
  percent_filled = 45 → 
  capacity = (liters_second_tank * 100) / percent_filled → 
  1000 - 300 = 700 → 
  1000 - 450 = 550 → 
  700 + 550 = 1250 :=
by sorry

end total_liters_needed_to_fill_two_tanks_l131_131992


namespace inverse_proportion_expression_and_calculation_l131_131759

theorem inverse_proportion_expression_and_calculation :
  (∃ k : ℝ, (∀ (x y : ℝ), y = k / x) ∧
   (∀ x y : ℝ, y = 400 ∧ x = 0.25 → k = 100) ∧
   (∀ x : ℝ, 200 = 100 / x → x = 0.5)) :=
by
  sorry

end inverse_proportion_expression_and_calculation_l131_131759


namespace lever_equilibrium_min_force_l131_131716

noncomputable def lever_minimum_force (F L : ℝ) : Prop :=
  (F * L = 49 + 2 * (L^2))

theorem lever_equilibrium_min_force : ∃ F : ℝ, ∃ L : ℝ, L = 7 → lever_minimum_force F L :=
by
  sorry

end lever_equilibrium_min_force_l131_131716


namespace correct_age_equation_l131_131283

variable (x : ℕ)

def age_older_brother (x : ℕ) : ℕ := 2 * x
def age_younger_brother_six_years_ago (x : ℕ) : ℕ := x - 6
def age_older_brother_six_years_ago (x : ℕ) : ℕ := 2 * x - 6

theorem correct_age_equation (h1 : age_younger_brother_six_years_ago x + age_older_brother_six_years_ago x = 15) :
  (x - 6) + (2 * x - 6) = 15 :=
by
  sorry

end correct_age_equation_l131_131283


namespace grid_points_circumference_l131_131285

def numGridPointsOnCircumference (R : ℝ) : ℕ := sorry

def isInteger (x : ℝ) : Prop := ∃ (n : ℤ), x = n

theorem grid_points_circumference (R : ℝ) (h : numGridPointsOnCircumference R = 1988) : 
  isInteger R ∨ isInteger (Real.sqrt 2 * R) :=
by
  sorry

end grid_points_circumference_l131_131285


namespace minimum_value_inequality_maximum_value_inequality_l131_131135

noncomputable def minimum_value (x1 x2 x3 : ℝ) : ℝ :=
  (x1 + 3 * x2 + 5 * x3) * (x1 + x2 / 3 + x3 / 5)

theorem minimum_value_inequality (x1 x2 x3 : ℝ) (h : 0 ≤ x1) (h : 0 ≤ x2) (h : 0 ≤ x3) (sum_eq : x1 + x2 + x3 = 1) :
  1 ≤ minimum_value x1 x2 x3 :=
sorry

theorem maximum_value_inequality (x1 x2 x3 : ℝ) (h : 0 ≤ x1) (h : 0 ≤ x2) (h : 0 ≤ x3) (sum_eq : x1 + x2 + x3 = 1) :
  minimum_value x1 x2 x3 ≤ 9/5 :=
sorry

end minimum_value_inequality_maximum_value_inequality_l131_131135


namespace parallel_segments_l131_131598

structure Point2D where
  x : Int
  y : Int

def vector (P Q : Point2D) : Point2D :=
  { x := Q.x - P.x, y := Q.y - P.y }

def is_parallel (v1 v2 : Point2D) : Prop :=
  ∃ k : Int, v2.x = k * v1.x ∧ v2.y = k * v1.y 

theorem parallel_segments :
  let A := { x := 1, y := 3 }
  let B := { x := 2, y := -1 }
  let C := { x := 0, y := 4 }
  let D := { x := 2, y := -4 }
  is_parallel (vector A B) (vector C D) := 
  sorry

end parallel_segments_l131_131598


namespace area_of_rectangular_field_l131_131580

-- Definitions from conditions
def L : ℕ := 20
def total_fencing : ℕ := 32

-- Additional variables inferred from the conditions
def W : ℕ := (total_fencing - L) / 2

-- The theorem statement
theorem area_of_rectangular_field : L * W = 120 :=
by
  -- Definitions and substitutions are included in the theorem proof
  sorry

end area_of_rectangular_field_l131_131580


namespace pick_theorem_l131_131274

def lattice_polygon (vertices : List (ℤ × ℤ)) : Prop :=
  ∀ v ∈ vertices, ∃ i j : ℤ, v = (i, j)

variables {n m : ℕ}
variables {A : ℤ}
variables {vertices : List (ℤ × ℤ)}

def lattice_point_count_inside (vertices : List (ℤ × ℤ)) : ℕ :=
  -- Placeholder for the actual logic to count inside points
  sorry

def lattice_point_count_boundary (vertices : List (ℤ × ℤ)) : ℕ :=
  -- Placeholder for the actual logic to count boundary points
  sorry

theorem pick_theorem (h : lattice_polygon vertices) :
  lattice_point_count_inside vertices = n → 
  lattice_point_count_boundary vertices = m → 
  A = n + m / 2 - 1 :=
sorry

end pick_theorem_l131_131274


namespace find_a_b_l131_131140

-- Define that the roots of the corresponding equality yield the specific conditions.
theorem find_a_b (a b : ℝ) :
    (∀ x : ℝ, x^2 + (a + 1) * x + ab > 0 ↔ (x < -1 ∨ x > 4)) →
    a = -4 ∧ b = 1 := 
by
    sorry

end find_a_b_l131_131140


namespace area_of_triangle_l131_131694

theorem area_of_triangle (S_x S_y S_z S : ℝ)
  (hx : S_x = Real.sqrt 7) (hy : S_y = Real.sqrt 6)
  (hz : ∃ k : ℕ, S_z = k) (hs : ∃ n : ℕ, S = n)
  : S = 7 := by
  sorry

end area_of_triangle_l131_131694


namespace shaded_area_isosceles_right_triangle_l131_131251

theorem shaded_area_isosceles_right_triangle (y : ℝ) :
  (∃ (x : ℝ), 2 * x^2 = y^2) ∧
  (∃ (A : ℝ), A = (1 / 2) * (y^2 / 2)) ∧
  (∃ (shaded_area : ℝ), shaded_area = (1 / 2) * (y^2 / 4)) →
  (shaded_area = y^2 / 8) :=
sorry

end shaded_area_isosceles_right_triangle_l131_131251


namespace fraction_area_below_diagonal_is_one_l131_131142

noncomputable def fraction_below_diagonal (s : ℝ) : ℝ := 1

theorem fraction_area_below_diagonal_is_one (s : ℝ) :
  let long_side := 2 * s
  let P := (2 * s / 3, 0)
  let Q := (s, s / 2)
  -- Total area of the rectangle
  let total_area := s * 2 * s -- 2s^2
  -- Total area below the diagonal
  let area_below_diagonal := 2 * s * s  -- 2s^2
  -- Fraction of the area below diagonal
  fraction_below_diagonal s = area_below_diagonal / total_area := 
by 
  sorry

end fraction_area_below_diagonal_is_one_l131_131142


namespace pears_sold_in_a_day_l131_131520

-- Define the conditions
variable (morning_pears afternoon_pears : ℕ)
variable (h1 : afternoon_pears = 2 * morning_pears)
variable (h2 : afternoon_pears = 320)

-- Lean theorem statement to prove the question answer
theorem pears_sold_in_a_day :
  (morning_pears + afternoon_pears = 480) :=
by
  -- Insert proof here
  sorry

end pears_sold_in_a_day_l131_131520


namespace min_height_required_kingda_ka_l131_131457

-- Definitions of the given conditions
def brother_height : ℕ := 180
def mary_relative_height : ℚ := 2 / 3
def growth_needed : ℕ := 20

-- Definition and statement of the problem
def marys_height : ℚ := mary_relative_height * brother_height
def minimum_height_required : ℚ := marys_height + growth_needed

theorem min_height_required_kingda_ka :
  minimum_height_required = 140 := by
  sorry

end min_height_required_kingda_ka_l131_131457


namespace alberto_vs_bjorn_distance_difference_l131_131938

noncomputable def alberto_distance (t : ℝ) : ℝ := (3.75 / 5) * t
noncomputable def bjorn_distance (t : ℝ) : ℝ := (3.4375 / 5) * t

theorem alberto_vs_bjorn_distance_difference :
  alberto_distance 5 - bjorn_distance 5 = 0.3125 :=
by
  -- proof goes here
  sorry

end alberto_vs_bjorn_distance_difference_l131_131938


namespace budget_equality_year_l131_131460

theorem budget_equality_year :
  ∀ Q R V W : ℕ → ℝ,
  Q 0 = 540000 ∧ R 0 = 660000 ∧ V 0 = 780000 ∧ W 0 = 900000 ∧
  (∀ n, Q (n+1) = Q n + 40000 ∧ 
         R (n+1) = R n + 30000 ∧ 
         V (n+1) = V n - 10000 ∧ 
         W (n+1) = W n - 20000) →
  ∃ n : ℕ, 1990 + n = 1995 ∧ 
  Q n + R n = V n + W n := 
by 
  sorry

end budget_equality_year_l131_131460


namespace sum_youngest_oldest_l131_131115

-- Define the ages of the cousins
variables (a1 a2 a3 a4 : ℕ)

-- Conditions given in the problem
def mean_age (a1 a2 a3 a4 : ℕ) : Prop := (a1 + a2 + a3 + a4) / 4 = 8
def median_age (a2 a3 : ℕ) : Prop := (a2 + a3) / 2 = 5

-- Main theorem statement to be proved
theorem sum_youngest_oldest (h_mean : mean_age a1 a2 a3 a4) (h_median : median_age a2 a3) :
  a1 + a4 = 22 :=
sorry

end sum_youngest_oldest_l131_131115


namespace m_squared_divisible_by_64_l131_131235

theorem m_squared_divisible_by_64 (m : ℕ) (h : 8 ∣ m) : 64 ∣ m * m :=
sorry

end m_squared_divisible_by_64_l131_131235


namespace intercepts_of_line_l131_131756

theorem intercepts_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) :
  (∃ x_intercept : ℝ, x_intercept = 7 ∧ (4 * x_intercept + 7 * 0 = 28)) ∧
  (∃ y_intercept : ℝ, y_intercept = 4 ∧ (4 * 0 + 7 * y_intercept = 28)) :=
by
  sorry

end intercepts_of_line_l131_131756


namespace find_dividend_l131_131795

-- Define the given constants
def quotient : ℕ := 909899
def divisor : ℕ := 12

-- Define the dividend as the product of divisor and quotient
def dividend : ℕ := divisor * quotient

-- The theorem stating the equality we need to prove
theorem find_dividend : dividend = 10918788 := by
  sorry

end find_dividend_l131_131795


namespace farmer_pigs_chickens_l131_131461

-- Defining the problem in Lean 4

theorem farmer_pigs_chickens (p ch : ℕ) (h₁ : 30 * p + 24 * ch = 1200) (h₂ : p > 0) (h₃ : ch > 0) : 
  (p = 4) ∧ (ch = 45) :=
by sorry

end farmer_pigs_chickens_l131_131461


namespace work_days_l131_131720

theorem work_days (x : ℕ) (hx : 0 < x) :
  (1 / (x : ℚ) + 1 / 20) = 1 / 15 → x = 60 := by
sorry

end work_days_l131_131720


namespace range_m_l131_131762

theorem range_m (m : ℝ) : 
  (∀ x : ℝ, ((m * x - 1) * (x - 2) > 0) ↔ (1/m < x ∧ x < 2)) → m < 0 :=
by
  sorry

end range_m_l131_131762


namespace smallest_solution_x_squared_abs_x_eq_3x_plus_4_l131_131427

theorem smallest_solution_x_squared_abs_x_eq_3x_plus_4 :
  ∃ x : ℝ, x^2 * |x| = 3 * x + 4 ∧ ∀ y : ℝ, (y^2 * |y| = 3 * y + 4 → y ≥ x) := 
sorry

end smallest_solution_x_squared_abs_x_eq_3x_plus_4_l131_131427


namespace oil_leakage_problem_l131_131730

theorem oil_leakage_problem :
    let l_A := 25  -- Leakage rate of Pipe A (gallons/hour)
    let l_B := 37  -- Leakage rate of Pipe B (gallons/hour)
    let l_C := 55  -- Leakage rate of Pipe C (gallons/hour)
    let l_D := 41  -- Leakage rate of Pipe D (gallons/hour)
    let l_E := 30  -- Leakage rate of Pipe E (gallons/hour)

    let t_A := 10  -- Time taken to fix Pipe A (hours)
    let t_B := 7   -- Time taken to fix Pipe B (hours)
    let t_C := 12  -- Time taken to fix Pipe C (hours)
    let t_D := 9   -- Time taken to fix Pipe D (hours)
    let t_E := 14  -- Time taken to fix Pipe E (hours)

    let leak_A := l_A * t_A  -- Total leaked from Pipe A (gallons)
    let leak_B := l_B * t_B  -- Total leaked from Pipe B (gallons)
    let leak_C := l_C * t_C  -- Total leaked from Pipe C (gallons)
    let leak_D := l_D * t_D  -- Total leaked from Pipe D (gallons)
    let leak_E := l_E * t_E  -- Total leaked from Pipe E (gallons)
  
    let overall_total := leak_A + leak_B + leak_C + leak_D + leak_E
  
    leak_A = 250 ∧
    leak_B = 259 ∧
    leak_C = 660 ∧
    leak_D = 369 ∧
    leak_E = 420 ∧
    overall_total = 1958 :=
by
    sorry

end oil_leakage_problem_l131_131730


namespace rectangles_with_perimeter_equals_area_l131_131589

theorem rectangles_with_perimeter_equals_area (a b : ℕ) (h : 2 * (a + b) = a * b) : (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (a = 4 ∧ b = 4) :=
  sorry

end rectangles_with_perimeter_equals_area_l131_131589


namespace initial_percentage_reduction_l131_131407

theorem initial_percentage_reduction (x : ℝ) :
  (1 - x / 100) * 1.17649 = 1 → x = 15 :=
by
  sorry

end initial_percentage_reduction_l131_131407


namespace find_two_digit_number_l131_131872

noncomputable def original_number (a b : ℕ) : ℕ := 10 * a + b

theorem find_two_digit_number (a b : ℕ) (h1 : a = 2 * b) (h2 : original_number b a = original_number a b - 36) : original_number a b = 84 :=
by
  sorry

end find_two_digit_number_l131_131872


namespace siblings_total_weekly_water_l131_131063

noncomputable def Theo_daily : ℕ := 8
noncomputable def Mason_daily : ℕ := 7
noncomputable def Roxy_daily : ℕ := 9

noncomputable def daily_to_weekly (daily : ℕ) : ℕ := daily * 7

theorem siblings_total_weekly_water :
  daily_to_weekly Theo_daily + daily_to_weekly Mason_daily + daily_to_weekly Roxy_daily = 168 := by
  sorry

end siblings_total_weekly_water_l131_131063


namespace calculate_X_value_l131_131685

theorem calculate_X_value : 
  let M := (2025 : ℝ) / 3
  let N := M / 4
  let X := M - N
  X = 506.25 :=
by 
  sorry

end calculate_X_value_l131_131685


namespace remainder_of_3056_mod_32_l131_131262

theorem remainder_of_3056_mod_32 : 3056 % 32 = 16 := by
  sorry

end remainder_of_3056_mod_32_l131_131262


namespace find_number_l131_131382

theorem find_number (x n : ℝ) (h1 : 0.12 / x * n = 12) (h2 : x = 0.1) : n = 10 := by
  sorry

end find_number_l131_131382


namespace derivative_at_pi_over_3_l131_131406

noncomputable def f (x : Real) : Real := 2 * Real.sin x + Real.sqrt 3 * Real.cos x

theorem derivative_at_pi_over_3 : (deriv f) (π / 3) = -1 / 2 := 
by
  sorry

end derivative_at_pi_over_3_l131_131406


namespace simplify_radical_l131_131524

theorem simplify_radical (x : ℝ) :
  2 * Real.sqrt (50 * x^3) * Real.sqrt (45 * x^5) * Real.sqrt (98 * x^7) = 420 * x^7 * Real.sqrt (5 * x) :=
by
  sorry

end simplify_radical_l131_131524


namespace fg_of_2_l131_131658

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^2 + 2

-- Prove the specific property
theorem fg_of_2 : f (g 2) = -19 := by
  -- Placeholder for the proof
  sorry

end fg_of_2_l131_131658


namespace larger_number_is_34_l131_131914

theorem larger_number_is_34 (a b : ℕ) (h1 : a > b) (h2 : (a + b) + (a - b) = 68) : a = 34 := 
by
  sorry

end larger_number_is_34_l131_131914


namespace arithmetic_geometric_l131_131569

theorem arithmetic_geometric (a_n : ℕ → ℤ) (h1 : ∀ n, a_n n = a_n 0 + n * 2)
  (h2 : ∃ a, a = a_n 0 ∧ (a_n 0 + 4)^2 = a_n 0 * (a_n 0 + 6)) : a_n 0 = -8 := by
  sorry

end arithmetic_geometric_l131_131569


namespace find_x_l131_131411

def cube_volume (s : ℝ) := s^3
def cube_surface_area (s : ℝ) := 6 * s^2

theorem find_x (x : ℝ) (s : ℝ) 
  (hv : cube_volume s = 7 * x)
  (hs : cube_surface_area s = x) : 
  x = 42 := 
by
  sorry

end find_x_l131_131411


namespace pants_price_l131_131497

theorem pants_price (P B : ℝ) 
  (condition1 : P + B = 70.93)
  (condition2 : P = B - 2.93) : 
  P = 34.00 :=
by
  sorry

end pants_price_l131_131497


namespace find_n_l131_131330

theorem find_n (n : ℕ) (h : (17 + 98 + 39 + 54 + n) / 5 = n) : n = 52 :=
by
  sorry

end find_n_l131_131330


namespace cake_shop_problem_l131_131312

theorem cake_shop_problem :
  ∃ (N n K : ℕ), (N - n * K = 6) ∧ (N = (n - 1) * 8 + 1) ∧ (N = 97) :=
by
  sorry

end cake_shop_problem_l131_131312


namespace profit_percentage_correct_l131_131689

def SP : ℝ := 900
def P : ℝ := 100

theorem profit_percentage_correct : (P / (SP - P)) * 100 = 12.5 := sorry

end profit_percentage_correct_l131_131689


namespace roger_remaining_debt_is_correct_l131_131893

def house_price : ℝ := 100000
def down_payment_rate : ℝ := 0.20
def parents_payment_rate : ℝ := 0.30

def remaining_debt (house_price down_payment_rate parents_payment_rate : ℝ) : ℝ :=
  let down_payment := house_price * down_payment_rate
  let remaining_balance_after_down_payment := house_price - down_payment
  let parents_payment := remaining_balance_after_down_payment * parents_payment_rate
  remaining_balance_after_down_payment - parents_payment

theorem roger_remaining_debt_is_correct :
  remaining_debt house_price down_payment_rate parents_payment_rate = 56000 :=
by sorry

end roger_remaining_debt_is_correct_l131_131893


namespace solution_set_of_inequalities_l131_131637

-- Definitions
def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

def is_strictly_decreasing (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Main Statement
theorem solution_set_of_inequalities
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_periodic : is_periodic f 2)
  (h_decreasing : is_strictly_decreasing f 0 1)
  (h_f_pi : f π = 1)
  (h_f_2pi : f (2 * π) = 2) :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ f x ∧ f x ≤ 2} = {x | π - 2 ≤ x ∧ x ≤ 8 - 2 * π} :=
  sorry

end solution_set_of_inequalities_l131_131637


namespace customer_wants_score_of_eggs_l131_131428

def Score := 20
def Dozen := 12

def options (n : Nat) : Prop :=
  n = Score ∨ n = 2 * Score ∨ n = 2 * Dozen ∨ n = 3 * Score

theorem customer_wants_score_of_eggs : 
  ∃ n, options n ∧ n = Score := 
by
  exists Score
  constructor
  apply Or.inl
  rfl
  rfl

end customer_wants_score_of_eggs_l131_131428


namespace sqrt5_lt_sqrt2_plus_1_l131_131838

theorem sqrt5_lt_sqrt2_plus_1 : Real.sqrt 5 < Real.sqrt 2 + 1 :=
sorry

end sqrt5_lt_sqrt2_plus_1_l131_131838


namespace circumcircle_radius_proof_l131_131827

noncomputable def circumcircle_radius (AB A S : ℝ) : ℝ :=
  if AB = 3 ∧ A = 120 ∧ S = 9 * Real.sqrt 3 / 4 then 3 else 0

theorem circumcircle_radius_proof :
  circumcircle_radius 3 120 (9 * Real.sqrt 3 / 4) = 3 := by
  sorry

end circumcircle_radius_proof_l131_131827


namespace gcd_2197_2209_l131_131001

theorem gcd_2197_2209 : Nat.gcd 2197 2209 = 1 := 
by
  sorry

end gcd_2197_2209_l131_131001


namespace trigonometric_identity_proof_l131_131315

noncomputable def trigonometric_expression : ℝ := 
  (Real.sin (15 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) 
  + Real.cos (165 * Real.pi / 180) * Real.cos (115 * Real.pi / 180)) /
  (Real.sin (35 * Real.pi / 180) * Real.cos (5 * Real.pi / 180) 
  + Real.cos (145 * Real.pi / 180) * Real.cos (85 * Real.pi / 180))

theorem trigonometric_identity_proof : trigonometric_expression = 1 :=
by
  sorry

end trigonometric_identity_proof_l131_131315


namespace bugs_ate_each_l131_131963

theorem bugs_ate_each : 
  ∀ (total_bugs total_flowers each_bug_flowers : ℕ), 
    total_bugs = 3 ∧ total_flowers = 6 ∧ each_bug_flowers = total_flowers / total_bugs -> each_bug_flowers = 2 := by
  sorry

end bugs_ate_each_l131_131963


namespace rectangle_perimeter_l131_131764

-- Define the conditions
variables (z w : ℕ)
-- Define the side lengths of the rectangles
def rectangle_long_side := z - w
def rectangle_short_side := w

-- Theorem: The perimeter of one of the four rectangles
theorem rectangle_perimeter : 2 * (rectangle_long_side z w) + 2 * (rectangle_short_side w) = 2 * z :=
by sorry

end rectangle_perimeter_l131_131764


namespace min_value_expr_l131_131967

theorem min_value_expr (a b : ℝ) (h : a - 2 * b + 8 = 0) : ∃ x : ℝ, x = 2^a + 1 / 4^b ∧ x = 1 / 8 :=
by
  sorry

end min_value_expr_l131_131967


namespace vanya_four_times_faster_l131_131700

-- We let d be the total distance, and define the respective speeds
variables (d : ℝ) (v_m v_v : ℝ)

-- Conditions from the problem
-- 1. Vanya starts after Masha
axiom start_after_masha : ∀ t : ℝ, t > 0

-- 2. Vanya overtakes Masha at one-third of the distance
axiom vanya_overtakes_masha : ∀ t : ℝ, (v_v * t) = d / 3

-- 3. When Vanya reaches the school, Masha still has half of the way to go
axiom masha_halfway : ∀ t : ℝ, (v_m * t) = d / 2

-- Goal to prove
theorem vanya_four_times_faster : v_v = 4 * v_m :=
sorry

end vanya_four_times_faster_l131_131700


namespace total_amount_paid_l131_131643

-- Definitions of the conditions
def cost_earbuds : ℝ := 200
def tax_rate : ℝ := 0.15

-- Statement to prove
theorem total_amount_paid : (cost_earbuds + (cost_earbuds * tax_rate)) = 230 := sorry

end total_amount_paid_l131_131643


namespace factorize_x2_add_2x_sub_3_l131_131355

theorem factorize_x2_add_2x_sub_3 :
  (x^2 + 2 * x - 3) = (x + 3) * (x - 1) :=
by
  sorry

end factorize_x2_add_2x_sub_3_l131_131355


namespace number_of_integers_in_sequence_l131_131673

theorem number_of_integers_in_sequence 
  (a_0 : ℕ) 
  (h_0 : a_0 = 8820) 
  (seq : ℕ → ℕ) 
  (h_seq : ∀ n : ℕ, seq (n + 1) = seq n / 3) :
  ∃ n : ℕ, seq n = 980 ∧ n + 1 = 3 :=
by
  sorry

end number_of_integers_in_sequence_l131_131673


namespace janet_counts_total_birds_l131_131749

theorem janet_counts_total_birds :
  let crows := 30
  let hawks := crows + (60 / 100) * crows
  hawks + crows = 78 :=
by
  sorry

end janet_counts_total_birds_l131_131749


namespace total_koalas_l131_131898

namespace KangarooKoalaProof

variables {P Q R S T U V p q r s t u v : ℕ}
variables (h₁ : P = q + r + s + t + u + v)
variables (h₂ : Q = p + r + s + t + u + v)
variables (h₃ : R = p + q + s + t + u + v)
variables (h₄ : S = p + q + r + t + u + v)
variables (h₅ : T = p + q + r + s + u + v)
variables (h₆ : U = p + q + r + s + t + v)
variables (h₇ : V = p + q + r + s + t + u)
variables (h_total : P + Q + R + S + T + U + V = 2022)

theorem total_koalas : p + q + r + s + t + u + v = 337 :=
by
  sorry

end KangarooKoalaProof

end total_koalas_l131_131898


namespace factorization_proof_l131_131892

theorem factorization_proof (a : ℝ) : 2 * a^2 + 4 * a + 2 = 2 * (a + 1)^2 :=
by { sorry }

end factorization_proof_l131_131892


namespace count_ways_to_choose_4_cards_l131_131498

-- A standard deck has 4 suits
def suits : Finset ℕ := {1, 2, 3, 4}

-- Each suit has 6 even cards: 2, 4, 6, 8, 10, and Queen (12)
def even_cards_per_suit : Finset ℕ := {2, 4, 6, 8, 10, 12}

-- Define the problem in Lean: 
-- Total number of ways to choose 4 cards such that all cards are of different suits and each is an even card.
theorem count_ways_to_choose_4_cards : (suits.card = 4 ∧ even_cards_per_suit.card = 6) → (1 * 6^4 = 1296) :=
by
  intros h
  have suits_distinct : suits.card = 4 := h.1
  have even_cards_count : even_cards_per_suit.card = 6 := h.2
  sorry

end count_ways_to_choose_4_cards_l131_131498


namespace part1_solution_set_part2_range_of_a_l131_131390

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l131_131390


namespace find_last_number_l131_131965

theorem find_last_number (A B C D : ℕ) 
  (h1 : A + B + C = 18) 
  (h2 : B + C + D = 9) 
  (h3 : A + D = 13) 
  : D = 2 := by 
  sorry

end find_last_number_l131_131965


namespace time_to_empty_l131_131575

-- Definitions for the conditions
def rate_fill_no_leak (R : ℝ) := R = 1 / 2 -- Cistern fills in 2 hours without leak
def effective_fill_rate (R L : ℝ) := R - L = 1 / 4 -- Effective fill rate when leaking
def remember_fill_time_leak (R L : ℝ) := (R - L) * 4 = 1 -- 4 hours to fill with leak

-- Main theorem statement
theorem time_to_empty (R L : ℝ) (h1 : rate_fill_no_leak R) (h2 : effective_fill_rate R L)
  (h3 : remember_fill_time_leak R L) : (1 / L = 4) :=
by
  sorry

end time_to_empty_l131_131575


namespace find_fx2_l131_131334

theorem find_fx2 (f : ℝ → ℝ) (x : ℝ) (h : f (x - 1) = x ^ 2) : f (x ^ 2) = (x ^ 2 + 1) ^ 2 := by
  sorry

end find_fx2_l131_131334


namespace cone_height_is_2_sqrt_15_l131_131042

noncomputable def height_of_cone (radius : ℝ) (num_sectors : ℕ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let sector_arc_length := circumference / num_sectors
  let base_radius := sector_arc_length / (2 * Real.pi)
  let slant_height := radius
  Real.sqrt (slant_height ^ 2 - base_radius ^ 2)

theorem cone_height_is_2_sqrt_15 :
  height_of_cone 8 4 = 2 * Real.sqrt 15 :=
by
  sorry

end cone_height_is_2_sqrt_15_l131_131042


namespace function_relationship_l131_131672

-- Definitions of the conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ {x y}, x ∈ s → y ∈ s → x < y → f y ≤ f x

-- The main statement we want to prove
theorem function_relationship (f : ℝ → ℝ) 
  (hf_even : even_function f)
  (hf_decreasing : decreasing_on f (Set.Ici 0)) :
  f 1 > f (-10) :=
by sorry

end function_relationship_l131_131672


namespace distinct_value_expression_l131_131760

def tri (a b : ℕ) : ℕ := min a b
def nabla (a b : ℕ) : ℕ := max a b

theorem distinct_value_expression (x : ℕ) : (nabla 5 (nabla 4 (tri x 4))) = 5 := 
by
  sorry

end distinct_value_expression_l131_131760


namespace keira_guarantees_capture_l131_131663

theorem keira_guarantees_capture (k : ℕ) (n : ℕ) (h_k_pos : 0 < k) (h_n_cond : n > k / 2023) :
    k ≥ 1012 :=
sorry

end keira_guarantees_capture_l131_131663


namespace probability_solved_l131_131666

theorem probability_solved (pA pB pA_and_B : ℚ) :
  pA = 2 / 3 → pB = 3 / 4 → pA_and_B = (2 / 3) * (3 / 4) →
  pA + pB - pA_and_B = 11 / 12 :=
by
  intros hA hB hA_and_B
  rw [hA, hB, hA_and_B]
  sorry

end probability_solved_l131_131666


namespace complete_the_square_l131_131977

theorem complete_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) :=
by
  intro h
  sorry

end complete_the_square_l131_131977


namespace velocity_of_current_l131_131846

theorem velocity_of_current (v : ℝ) 
  (row_speed : ℝ) (distance : ℝ) (total_time : ℝ) 
  (h_row_speed : row_speed = 5)
  (h_distance : distance = 2.4)
  (h_total_time : total_time = 1)
  (h_equation : distance / (row_speed + v) + distance / (row_speed - v) = total_time) :
  v = 1 :=
sorry

end velocity_of_current_l131_131846


namespace complex_multiplication_l131_131750

-- Definition of the imaginary unit i
def i : ℂ := Complex.I

-- The theorem stating the equality
theorem complex_multiplication : (2 + i) * (3 + i) = 5 + 5 * i := 
sorry

end complex_multiplication_l131_131750


namespace fourth_triangle_exists_l131_131097

theorem fourth_triangle_exists (a b c d : ℝ)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)
  (h4 : a + b > d) (h5 : a + d > b) (h6 : b + d > a)
  (h7 : a + c > d) (h8 : a + d > c) (h9 : c + d > a) :
  b + c > d ∧ b + d > c ∧ c + d > b :=
by
  -- I skip the proof with "sorry"
  sorry

end fourth_triangle_exists_l131_131097


namespace larger_number_of_two_l131_131761

theorem larger_number_of_two (x y : ℝ) (h1 : x - y = 3) (h2 : x + y = 29) (h3 : x * y > 200) : x = 16 :=
by sorry

end larger_number_of_two_l131_131761


namespace problem_solution_l131_131717

noncomputable def S : ℝ :=
  1 / (5 - Real.sqrt 19) - 1 / (Real.sqrt 19 - Real.sqrt 18) + 
  1 / (Real.sqrt 18 - Real.sqrt 17) - 1 / (Real.sqrt 17 - 3) + 
  1 / (3 - Real.sqrt 2)

theorem problem_solution : S = 5 + Real.sqrt 2 := by
  sorry

end problem_solution_l131_131717


namespace number_of_chickens_l131_131123

theorem number_of_chickens (c b : ℕ) (h1 : c + b = 9) (h2 : 2 * c + 4 * b = 26) : c = 5 :=
by
  sorry

end number_of_chickens_l131_131123


namespace geometric_sequence_general_term_l131_131517

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 3) (h4 : a 4 = 81) :
  ∃ q : ℝ, (a n = 3 * q ^ (n - 1)) := by
  sorry

end geometric_sequence_general_term_l131_131517


namespace arithmetic_expression_count_l131_131723

theorem arithmetic_expression_count (f : ℕ → ℤ) 
  (h1 : f 1 = 9)
  (h2 : f 2 = 99)
  (h_recur : ∀ n ≥ 2, f n = 9 * (f (n - 1)) + 36 * (f (n - 2))) :
  ∀ n, f n = (7 / 10 : ℚ) * 12^n - (1 / 5 : ℚ) * (-3)^n := sorry

end arithmetic_expression_count_l131_131723


namespace least_3_digit_number_l131_131970

variables (k S h t u : ℕ)

def is_3_digit_number (k : ℕ) : Prop := k ≥ 100 ∧ k < 1000

def digits_sum_eq (k h t u S : ℕ) : Prop :=
  k = 100 * h + 10 * t + u ∧ S = h + t + u

def difference_condition (h t : ℕ) : Prop :=
  t - h = 8

theorem least_3_digit_number (k S h t u : ℕ) :
  is_3_digit_number k →
  digits_sum_eq k h t u S →
  difference_condition h t →
  k * 3 < 200 →
  k = 19 * S :=
sorry

end least_3_digit_number_l131_131970


namespace number_of_students_l131_131268

theorem number_of_students
    (average_marks : ℕ)
    (wrong_mark : ℕ)
    (correct_mark : ℕ)
    (correct_average_marks : ℕ)
    (h1 : average_marks = 100)
    (h2 : wrong_mark = 50)
    (h3 : correct_mark = 10)
    (h4 : correct_average_marks = 96)
  : ∃ n : ℕ, (100 * n - 40) / n = 96 ∧ n = 10 :=
by
  sorry

end number_of_students_l131_131268


namespace switches_connections_l131_131130

theorem switches_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 :=
by
  sorry

end switches_connections_l131_131130
