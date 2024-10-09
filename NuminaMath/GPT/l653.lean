import Mathlib

namespace expand_polynomial_l653_65313

theorem expand_polynomial :
  (x^2 - 3 * x + 3) * (x^2 + 3 * x + 1) = x^4 - 5 * x^2 + 6 * x + 3 :=
by
  sorry

end expand_polynomial_l653_65313


namespace campers_in_two_classes_l653_65372

-- Definitions of the sets and conditions
variable (S A R : Finset ℕ)
variable (n : ℕ)
variable (x : ℕ)

-- Given conditions
axiom hyp1 : S.card = 20
axiom hyp2 : A.card = 20
axiom hyp3 : R.card = 20
axiom hyp4 : (S ∩ A ∩ R).card = 4
axiom hyp5 : (S \ (A ∪ R)).card + (A \ (S ∪ R)).card + (R \ (S ∪ A)).card = 24

-- The hypothesis that n = |S ∪ A ∪ R|
axiom hyp6 : n = (S ∪ A ∪ R).card

-- Statement to be proven in Lean
theorem campers_in_two_classes : x = 12 :=
by
  sorry

end campers_in_two_classes_l653_65372


namespace minimum_positive_announcements_l653_65307

theorem minimum_positive_announcements (x y : ℕ) (h : x * (x - 1) = 132) (positive_products negative_products : ℕ)
  (hp : positive_products = y * (y - 1)) (hn : negative_products = (x - y) * (x - y - 1)) 
  (h_sum : positive_products + negative_products = 132) : 
  y = 2 :=
by sorry

end minimum_positive_announcements_l653_65307


namespace detail_understanding_word_meaning_guessing_logical_reasoning_l653_65387

-- Detail Understanding Question
theorem detail_understanding (sentence: String) (s: ∀ x : String, x ∈ ["He hardly watered his new trees,..."] → x = sentence) :
  sentence = "He hardly watered his new trees,..." :=
sorry

-- Word Meaning Guessing Question
theorem word_meaning_guessing (adversity_meaning: String) (meanings: ∀ y : String, y ∈ ["adversity means misfortune or disaster", "lack of water", "sufficient care/attention", "bad weather"] → y = adversity_meaning) :
  adversity_meaning = "adversity means misfortune or disaster" :=
sorry

-- Logical Reasoning Question
theorem logical_reasoning (hope: String) (sentences: ∀ z : String, z ∈ ["The author hopes his sons can withstand the tests of wind and rain in their life journey"] → z = hope) :
  hope = "The author hopes his sons can withstand the tests of wind and rain in their life journey" :=
sorry

end detail_understanding_word_meaning_guessing_logical_reasoning_l653_65387


namespace blue_bird_high_school_team_arrangement_l653_65353

theorem blue_bird_high_school_team_arrangement : 
  let girls := 2
  let boys := 3
  let girls_permutations := Nat.factorial girls
  let boys_permutations := Nat.factorial boys
  girls_permutations * boys_permutations = 12 := by
  sorry

end blue_bird_high_school_team_arrangement_l653_65353


namespace initial_production_rate_l653_65317

variable (x : ℕ) (t : ℝ)

-- Conditions
def produces_initial (x : ℕ) (t : ℝ) : Prop := x * t = 60
def produces_subsequent : Prop := 60 * 1 = 60
def overall_average (t : ℝ) : Prop := 72 = 120 / (t + 1)

-- Goal: Prove the initial production rate
theorem initial_production_rate : 
  (∃ t : ℝ, produces_initial x t ∧ produces_subsequent ∧ overall_average t) → x = 90 := 
  by
    sorry

end initial_production_rate_l653_65317


namespace diane_money_l653_65368

-- Define the conditions
def total_cost : ℤ := 65
def additional_needed : ℤ := 38
def initial_amount : ℤ := total_cost - additional_needed

-- Theorem statement
theorem diane_money : initial_amount = 27 := by
  sorry

end diane_money_l653_65368


namespace prove_inequality_l653_65344

variables {a b c A B C k : ℝ}

-- Define the conditions
def conditions (a b c A B C k : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ k > 0 ∧
  a + A = k ∧ b + B = k ∧ c + C = k

-- Define the theorem to be proven
theorem prove_inequality (a b c A B C k : ℝ) (h : conditions a b c A B C k) :
  a * B + b * C + c * A ≤ k^2 :=
sorry

end prove_inequality_l653_65344


namespace maximize_profit_l653_65399

noncomputable def production_problem : Prop :=
  ∃ (x y : ℕ), (3 * x + 2 * y ≤ 1200) ∧ (x + 2 * y ≤ 800) ∧ 
               (30 * x + 40 * y) = 18000 ∧ 
               x = 200 ∧ 
               y = 300

theorem maximize_profit : production_problem :=
sorry

end maximize_profit_l653_65399


namespace integer_values_satisfying_sqrt_condition_l653_65382

theorem integer_values_satisfying_sqrt_condition : ∃! n : Nat, 2.5 < Real.sqrt n ∧ Real.sqrt n < 3.5 :=
by {
  sorry -- Proof to be filled in
}

end integer_values_satisfying_sqrt_condition_l653_65382


namespace student_selection_l653_65379

theorem student_selection : 
  let first_year := 4
  let second_year := 5
  let third_year := 4
  (first_year * second_year) + (first_year * third_year) + (second_year * third_year) = 56 := by
  let first_year := 4
  let second_year := 5
  let third_year := 4
  sorry

end student_selection_l653_65379


namespace evaluate_expression_l653_65315

variables {a b c d e : ℝ}

theorem evaluate_expression (a b c d e : ℝ) : a * b^c - d + e = a * (b^c - (d + e)) :=
by
  sorry

end evaluate_expression_l653_65315


namespace pairwise_coprime_triples_l653_65310

open Nat

theorem pairwise_coprime_triples (a b c : ℕ) 
  (h1 : a.gcd b = 1) (h2 : a.gcd c = 1) (h3 : b.gcd c = 1)
  (h4 : (a + b) ∣ c) (h5 : (a + c) ∣ b) (h6 : (b + c) ∣ a) :
  { (a, b, c) | (a = 1 ∧ b = 1 ∧ (c = 1 ∨ c = 2)) ∨ (a = 1 ∧ b = 2 ∧ c = 3) } :=
by
  -- Proof omitted for conciseness
  sorry

end pairwise_coprime_triples_l653_65310


namespace problem_a_l653_65304

def part_a : Prop :=
  ∃ (tokens : Finset (Fin 4 × Fin 4)), 
    tokens.card = 7 ∧ 
    (∀ (rows : Finset (Fin 4)) (cols : Finset (Fin 4)), rows.card = 2 → cols.card = 2 → 
      ∃ (token : (Fin 4 × Fin 4)), token ∈ tokens ∧ token.1 ∉ rows ∧ token.2 ∉ cols)

theorem problem_a : part_a :=
  sorry

end problem_a_l653_65304


namespace total_price_before_increase_l653_65311

-- Conditions
def original_price_candy_box (c_or: ℝ) := 10 = c_or * 1.25
def original_price_soda_can (s_or: ℝ) := 15 = s_or * 1.50

-- Goal
theorem total_price_before_increase :
  ∃ (c_or s_or : ℝ), original_price_candy_box c_or ∧ original_price_soda_can s_or ∧ c_or + s_or = 25 :=
by
  sorry

end total_price_before_increase_l653_65311


namespace roll_seven_dice_at_least_one_pair_no_three_l653_65309

noncomputable def roll_seven_dice_probability : ℚ :=
  let total_outcomes := (6^7 : ℚ)
  let one_pair_case := (6 * 21 * 120 : ℚ)
  let two_pairs_case := (15 * 21 * 10 * 24 : ℚ)
  let successful_outcomes := one_pair_case + two_pairs_case
  successful_outcomes / total_outcomes

theorem roll_seven_dice_at_least_one_pair_no_three :
  roll_seven_dice_probability = 315 / 972 :=
by
  unfold roll_seven_dice_probability
  -- detailed steps to show the proof would go here
  sorry

end roll_seven_dice_at_least_one_pair_no_three_l653_65309


namespace billy_reads_60_pages_per_hour_l653_65300

theorem billy_reads_60_pages_per_hour
  (free_time_per_day : ℕ)
  (days : ℕ)
  (video_games_time_percentage : ℝ)
  (books : ℕ)
  (pages_per_book : ℕ)
  (remaining_time_percentage : ℝ)
  (total_free_time := free_time_per_day * days)
  (time_playing_video_games := video_games_time_percentage * total_free_time)
  (time_reading := remaining_time_percentage * total_free_time)
  (total_pages := books * pages_per_book)
  (pages_per_hour := total_pages / time_reading) :
  free_time_per_day = 8 →
  days = 2 →
  video_games_time_percentage = 0.75 →
  remaining_time_percentage = 0.25 →
  books = 3 →
  pages_per_book = 80 →
  pages_per_hour = 60 :=
by
  intros
  sorry

end billy_reads_60_pages_per_hour_l653_65300


namespace solve_cubic_eq_l653_65308

theorem solve_cubic_eq (x : ℝ) : x^3 + (2 - x)^3 = 8 ↔ x = 0 ∨ x = 2 := 
by 
  { sorry }

end solve_cubic_eq_l653_65308


namespace true_propositions_in_reverse_neg_neg_reverse_l653_65362

theorem true_propositions_in_reverse_neg_neg_reverse (a b : ℕ) : 
  (¬ (a ≠ 0 → a * b ≠ 0) ∧ ∃ (a : ℕ), (a = 0 ∧ a * b ≠ 0) ∨ (a ≠ 0 ∧ a * b = 0) ∧ ¬ (¬ ∃ (a : ℕ), a ≠ 0 ∧ a * b ≠ 0 ∧ ¬ ∃ (a : ℕ), a = 0 ∧ a * b = 0)) ∧ (0 = 1) :=
by {
  sorry
}

end true_propositions_in_reverse_neg_neg_reverse_l653_65362


namespace prob_A_wins_4_consecutive_prob_fifth_game_needed_prob_C_ultimate_winner_l653_65340

-- Conditions for the game settings
def initial_conditions (a b c : ℕ) : Prop :=
  a = 0 ∧ b = 0 ∧ c = 0

-- Probability of a player winning any game
def win_probability : ℚ := 1 / 2 

-- Probability calculation for A winning four consecutive games
theorem prob_A_wins_4_consecutive :
  win_probability ^ 4 = 1 / 16 :=
by
  sorry

-- Probability calculation for needing a fifth game to be played
theorem prob_fifth_game_needed :
  1 - 4 * (win_probability ^ 4) = 3 / 4 :=
by
  sorry

-- Probability calculation for C being the ultimate winner
theorem prob_C_ultimate_winner :
  1 - 2 * (9 / 32) = 7 / 16 :=
by
  sorry

end prob_A_wins_4_consecutive_prob_fifth_game_needed_prob_C_ultimate_winner_l653_65340


namespace total_houses_in_lincoln_county_l653_65397

theorem total_houses_in_lincoln_county 
  (original_houses : ℕ) 
  (houses_built : ℕ) 
  (h_original : original_houses = 20817) 
  (h_built : houses_built = 97741) : 
  original_houses + houses_built = 118558 := 
by 
  -- Proof steps or tactics would go here
  sorry

end total_houses_in_lincoln_county_l653_65397


namespace girls_more_than_boys_l653_65346

theorem girls_more_than_boys (total_students boys : ℕ) (h : total_students = 466) (b : boys = 127) (gt : total_students - boys > boys) :
  total_students - 2 * boys = 212 := by
  sorry

end girls_more_than_boys_l653_65346


namespace equivalent_expression_l653_65350

variable (x y : ℝ)

def is_positive_real (r : ℝ) : Prop := r > 0

theorem equivalent_expression 
  (hx : is_positive_real x) 
  (hy : is_positive_real y) : 
  (Real.sqrt (Real.sqrt (x ^ 2 * Real.sqrt (y ^ 3)))) = x ^ (1 / 2) * y ^ (1 / 12) :=
by
  sorry

end equivalent_expression_l653_65350


namespace total_money_correct_l653_65306

-- Define the number of pennies and quarters Sam has
def pennies : ℕ := 9
def quarters : ℕ := 7

-- Define the value of one penny and one quarter
def penny_value : ℝ := 0.01
def quarter_value : ℝ := 0.25

-- Calculate the total value of pennies and quarters Sam has
def total_value : ℝ := pennies * penny_value + quarters * quarter_value

-- Proof problem: Prove that the total value of money Sam has is $1.84
theorem total_money_correct : total_value = 1.84 :=
sorry

end total_money_correct_l653_65306


namespace prob_same_color_is_correct_l653_65394

-- Define the sides of one die
def blue_sides := 6
def yellow_sides := 8
def green_sides := 10
def purple_sides := 6
def total_sides := 30

-- Define the probability each die shows a specific color
def prob_blue := blue_sides / total_sides
def prob_yellow := yellow_sides / total_sides
def prob_green := green_sides / total_sides
def prob_purple := purple_sides / total_sides

-- The probability that both dice show the same color
def prob_same_color :=
  (prob_blue * prob_blue) + 
  (prob_yellow * prob_yellow) + 
  (prob_green * prob_green) + 
  (prob_purple * prob_purple)

-- We should prove that the computed probability is equal to the given answer
theorem prob_same_color_is_correct :
  prob_same_color = 59 / 225 := 
sorry

end prob_same_color_is_correct_l653_65394


namespace reduced_less_than_scaled_l653_65364

-- Define the conditions
def original_flow_rate : ℝ := 5.0
def reduced_flow_rate : ℝ := 2.0
def scaled_flow_rate : ℝ := 0.6 * original_flow_rate

-- State the theorem we need to prove
theorem reduced_less_than_scaled : scaled_flow_rate - reduced_flow_rate = 1.0 := 
by
  -- insert the detailed proof steps here
  sorry

end reduced_less_than_scaled_l653_65364


namespace equal_abc_l653_65329

theorem equal_abc {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ 
       b^2 * (c + a - b) = c^2 * (a + b - c)) : a = b ∧ b = c :=
by
  sorry

end equal_abc_l653_65329


namespace max_mixed_gender_groups_l653_65357

theorem max_mixed_gender_groups (b g : ℕ) (h_b : b = 31) (h_g : g = 32) : 
  ∃ max_groups, max_groups = min (b / 2) (g / 3) :=
by
  use 10
  sorry

end max_mixed_gender_groups_l653_65357


namespace speed_ratio_thirteen_l653_65305

noncomputable section

def speed_ratio (vNikita vCar : ℝ) : ℝ := vCar / vNikita

theorem speed_ratio_thirteen :
  ∀ (vNikita vCar : ℝ),
  (65 * vNikita = 5 * vCar) →
  speed_ratio vNikita vCar = 13 :=
by
  intros vNikita vCar h
  unfold speed_ratio
  sorry

end speed_ratio_thirteen_l653_65305


namespace geometric_sequence_a_l653_65345

open Real

theorem geometric_sequence_a (a : ℝ) (r : ℝ) (h1 : 20 * r = a) (h2 : a * r = 5/4) (h3 : 0 < a) : a = 5 :=
by
  -- The proof would go here
  sorry

end geometric_sequence_a_l653_65345


namespace no_appearance_1234_or_3269_no_reappearance_1975_from_2nd_time_l653_65347

def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 9
  | 2 => 7
  | 3 => 5
  | n + 4 => (seq n + seq (n + 1) + seq (n + 2) + seq (n + 3)) % 10

theorem no_appearance_1234_or_3269 : 
  ¬∃ n, seq n = 1 ∧ seq (n + 1) = 2 ∧ seq (n + 2) = 3 ∧ seq (n + 3) = 4 ∨
  seq n = 3 ∧ seq (n + 1) = 2 ∧ seq (n + 2) = 6 ∧ seq (n + 3) = 9 := 
sorry

theorem no_reappearance_1975_from_2nd_time : 
  ¬∃ n > 0, seq n = 1 ∧ seq (n + 1) = 9 ∧ seq (n + 2) = 7 ∧ seq (n + 3) = 5 :=
sorry

end no_appearance_1234_or_3269_no_reappearance_1975_from_2nd_time_l653_65347


namespace jose_investment_proof_l653_65392

noncomputable def jose_investment (total_profit jose_share : ℕ) (tom_investment : ℕ) (months_tom months_jose : ℕ) : ℕ :=
  let tom_share := total_profit - jose_share
  let tom_investment_mr := tom_investment * months_tom
  let ratio := tom_share * months_jose
  tom_investment_mr * jose_share / ratio

theorem jose_investment_proof : 
  ∃ (jose_invested : ℕ), 
    let total_profit := 5400
    let jose_share := 3000
    let tom_invested := 3000
    let months_tom := 12
    let months_jose := 10
    jose_investment total_profit jose_share tom_invested months_tom months_jose = 4500 :=
by
  use 4500
  sorry

end jose_investment_proof_l653_65392


namespace power_product_is_100_l653_65338

theorem power_product_is_100 :
  (10^0.6) * (10^0.4) * (10^0.3) * (10^0.2) * (10^0.5) = 100 :=
by
  sorry

end power_product_is_100_l653_65338


namespace isosceles_triangle_perimeter_l653_65328

theorem isosceles_triangle_perimeter (perimeter_eq_tri : ℕ) (side_eq_tri : ℕ) (base_iso_tri : ℕ) (perimeter_iso_tri : ℕ) 
  (h1 : perimeter_eq_tri = 60) 
  (h2 : side_eq_tri = perimeter_eq_tri / 3) 
  (h3 : base_iso_tri = 5)
  (h4 : perimeter_iso_tri = 2 * side_eq_tri + base_iso_tri) : 
  perimeter_iso_tri = 45 := by
  sorry

end isosceles_triangle_perimeter_l653_65328


namespace picture_area_l653_65370

-- Given dimensions of the paper
def paper_width : ℝ := 8.5
def paper_length : ℝ := 10

-- Given margins
def margin : ℝ := 1.5

-- Calculated dimensions of the picture
def picture_width := paper_width - 2 * margin
def picture_length := paper_length - 2 * margin

-- Statement to prove
theorem picture_area : picture_width * picture_length = 38.5 := by
  -- skipped the proof
  sorry

end picture_area_l653_65370


namespace least_positive_integer_property_l653_65377

theorem least_positive_integer_property : 
  ∃ (n d : ℕ) (p : ℕ) (h₁ : 1 ≤ d) (h₂ : d ≤ 9) (h₃ : p ≥ 2), 
  (10^p * d = 24 * n) ∧ (∃ k : ℕ, (n = 100 * 10^(p-2) / 3) ∧ (900 = 8 * 10^p + 100 / 3 * 10^(p-2))) := sorry

end least_positive_integer_property_l653_65377


namespace no_real_solutions_l653_65366

theorem no_real_solutions : ∀ x : ℝ, ¬(3 * x - 2 * x + 8) ^ 2 = -|x| - 4 :=
by
  intro x
  sorry

end no_real_solutions_l653_65366


namespace cell_value_l653_65341

variable (P Q R S : ℕ)

-- Condition definitions
def topLeftCell (P : ℕ) : ℕ := P
def topMiddleCell (P Q : ℕ) : ℕ := P + Q
def centerCell (P Q R S : ℕ) : ℕ := P + Q + R + S
def bottomLeftCell (S : ℕ) : ℕ := S

-- Given Conditions
axiom bottomLeftCell_value : bottomLeftCell S = 13
axiom topMiddleCell_value : topMiddleCell P Q = 18
axiom centerCell_value : centerCell P Q R S = 47

-- To prove: R = 16
theorem cell_value : R = 16 :=
by
  sorry

end cell_value_l653_65341


namespace mike_gave_pens_l653_65322

theorem mike_gave_pens (M : ℕ) 
  (initial_pens : ℕ := 5) 
  (pens_after_mike : ℕ := initial_pens + M)
  (pens_after_cindy : ℕ := 2 * pens_after_mike)
  (pens_after_sharon : ℕ := pens_after_cindy - 10)
  (final_pens : ℕ := 40) : 
  pens_after_sharon = final_pens → M = 20 := 
by 
  sorry

end mike_gave_pens_l653_65322


namespace value_of_f_5_l653_65301

-- Define the function f
def f (x y : ℕ) : ℕ := 2 * x ^ 2 + y

-- Given conditions
variable (some_value : ℕ)
axiom h1 : f some_value 52 = 60
axiom h2 : f 5 52 = 102

-- Proof statement
theorem value_of_f_5 : f 5 52 = 102 := by
  sorry

end value_of_f_5_l653_65301


namespace water_depth_when_upright_l653_65314
-- Import the entire Mathlib library

-- Define the conditions and question as a theorem
theorem water_depth_when_upright (height : ℝ) (diameter : ℝ) (horizontal_depth : ℝ) :
  height = 20 → diameter = 6 → horizontal_depth = 4 → water_depth = 5.3 :=
by
  intro h1 h2 h3
  -- The proof would go here, but we insert sorry to skip it
  sorry

end water_depth_when_upright_l653_65314


namespace find_alpha_polar_equation_l653_65303

noncomputable def alpha := (3 * Real.pi) / 4

theorem find_alpha (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) (A B : ℝ × ℝ)
  (hP : P = (2, 1))
  (hl_def : ∀ t, l t = (2 + t * Real.cos alpha, 1 + t * Real.sin alpha))
  (hA : ∃ t, l t = (A.1, 0) ∧ A.1 > 0)
  (hB : ∃ t, l t = (0, B.2) ∧ B.2 > 0)
  (h_cond : dist P A * dist P B = 4) : alpha = (3 * Real.pi) / 4 :=
sorry

theorem polar_equation (l : ℝ → ℝ × ℝ)
  (hl_def : ∀ t, l t = (2 + t * Real.cos alpha, 1 + t * Real.sin alpha))
  (h_alpha : alpha = (3 * Real.pi) / 4)
  (h_polar : ∀ ρ θ, l ρ = (ρ * Real.cos θ, ρ * Real.sin θ))
  : ∀ ρ θ, ρ * (Real.cos θ + Real.sin θ) = 3 :=
sorry

end find_alpha_polar_equation_l653_65303


namespace total_money_is_2800_l653_65327

-- Define variables for money
def Cecil_money : ℕ := 600
def Catherine_money : ℕ := 2 * Cecil_money - 250
def Carmela_money : ℕ := 2 * Cecil_money + 50

-- Assertion to prove the total money 
theorem total_money_is_2800 : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- placeholder proof
  sorry

end total_money_is_2800_l653_65327


namespace clock_angle_at_3_30_l653_65361

theorem clock_angle_at_3_30 
    (deg_per_hour: Real := 30)
    (full_circle_deg: Real := 360)
    (hours_on_clock: Real := 12)
    (hour_hand_extra_deg: Real := 30 / 2)
    (hour_hand_deg: Real := 3 * deg_per_hour + hour_hand_extra_deg)
    (minute_hand_deg: Real := 6 * deg_per_hour) : 
    hour_hand_deg = 105 ∧ minute_hand_deg = 180 ∧ (minute_hand_deg - hour_hand_deg) = 75 := 
sorry

-- The problem specifies to write the theorem statement only, without the proof steps.

end clock_angle_at_3_30_l653_65361


namespace point_P_on_number_line_l653_65386

variable (A : ℝ) (B : ℝ) (P : ℝ)

theorem point_P_on_number_line (hA : A = -1) (hB : B = 5) (hDist : abs (P - A) = abs (B - P)) : P = 2 := 
sorry

end point_P_on_number_line_l653_65386


namespace slower_pipe_filling_time_l653_65365

-- Definitions based on conditions
def faster_pipe_rate (S : ℝ) : ℝ := 3 * S
def combined_rate (S : ℝ) : ℝ := (faster_pipe_rate S) + S

-- Statement of what needs to be proved 
theorem slower_pipe_filling_time :
  (∀ S : ℝ, combined_rate S * 40 = 1) →
  ∃ t : ℝ, t = 160 :=
by
  intro h
  sorry

end slower_pipe_filling_time_l653_65365


namespace common_ratio_l653_65398

def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def arith_seq (a : ℕ → ℝ) (x y z : ℕ) := 2 * a z = a x + a y

theorem common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geom : geom_seq a q) (h_arith : arith_seq a 0 1 2) (h_nonzero : a 0 ≠ 0) : q = 1 ∨ q = -1/2 :=
by
  sorry

end common_ratio_l653_65398


namespace range_of_m_l653_65320

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^2 + (m - 1) * x + 1 = 0) → m ≤ -1 :=
by
  sorry

end range_of_m_l653_65320


namespace rational_sqrt_of_rational_xy_l653_65343

theorem rational_sqrt_of_rational_xy (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) :
  ∃ k : ℚ, k^2 = 1 - x * y := 
sorry

end rational_sqrt_of_rational_xy_l653_65343


namespace continuity_at_2_l653_65330

theorem continuity_at_2 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |(-3 * x^2 - 5) + 17| < ε :=
by
  sorry

end continuity_at_2_l653_65330


namespace solve_system_of_equations_l653_65354

theorem solve_system_of_equations :
  ∃ x : ℕ → ℝ,
  (∀ i : ℕ, i < 100 → x i > 0) ∧
  (x 0 + 1 / x 1 = 4) ∧
  (x 1 + 1 / x 2 = 1) ∧
  (x 2 + 1 / x 0 = 4) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i < 99 → x (2 * i + 1) + 1 / x (2 * i + 2) = 1) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i < 99 → x (2 * i + 2) + 1 / x (2 * i + 3) = 4) ∧
  (x 99 + 1 / x 0 = 1) ∧
  (∀ i : ℕ, i < 50 → x (2 * i) = 2) ∧
  (∀ i : ℕ, i < 50 → x (2 * i + 1) = 1 / 2) :=
sorry

end solve_system_of_equations_l653_65354


namespace jacob_current_age_l653_65384

theorem jacob_current_age 
  (M : ℕ) 
  (Drew_age : ℕ := M + 5) 
  (Peter_age : ℕ := Drew_age + 4) 
  (John_age : ℕ := 30) 
  (maya_age_eq : 2 * M = John_age) 
  (jacob_future_age : ℕ := Peter_age / 2) 
  (jacob_current_age_eq : ℕ := jacob_future_age - 2) : 
  jacob_current_age_eq = 11 := 
sorry

end jacob_current_age_l653_65384


namespace particle_speed_at_time_t_l653_65378

noncomputable def position (t : ℝ) : ℝ × ℝ :=
  (3 * t^2 + t + 1, 6 * t + 2)

theorem particle_speed_at_time_t (t : ℝ) :
  let dx := (position t).1
  let dy := (position t).2
  let vx := 6 * t + 1
  let vy := 6
  let speed := Real.sqrt (vx^2 + vy^2)
  speed = Real.sqrt (36 * t^2 + 12 * t + 37) :=
by
  sorry

end particle_speed_at_time_t_l653_65378


namespace sum_of_squares_not_perfect_square_l653_65335

theorem sum_of_squares_not_perfect_square (n : ℕ) (h : n > 4) :
  ¬ (∃ k : ℕ, 10 * n^2 + 10 * n + 85 = k^2) :=
sorry

end sum_of_squares_not_perfect_square_l653_65335


namespace initial_pens_l653_65359

theorem initial_pens (P : ℤ) (INIT : 2 * (P + 22) - 19 = 39) : P = 7 :=
by
  sorry

end initial_pens_l653_65359


namespace cosine_of_difference_l653_65334

theorem cosine_of_difference (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.cos (α - π / 3) = 1 / 3 :=
by
  sorry

end cosine_of_difference_l653_65334


namespace sum_of_series_equals_one_half_l653_65324

theorem sum_of_series_equals_one_half : 
  (∑' k : ℕ, (1 / ((2 * k + 1) * (2 * k + 3)))) = 1 / 2 :=
sorry

end sum_of_series_equals_one_half_l653_65324


namespace soja_finished_fraction_l653_65383

def pages_finished (x pages_left total_pages : ℕ) : Prop :=
  x - pages_left = 100 ∧ x + pages_left = total_pages

noncomputable def fraction_finished (x total_pages : ℕ) : ℚ :=
  x / total_pages

theorem soja_finished_fraction (x : ℕ) (h1 : pages_finished x (x - 100) 300) :
  fraction_finished x 300 = 2 / 3 :=
by
  sorry

end soja_finished_fraction_l653_65383


namespace max_path_length_is_32_l653_65319
-- Import the entire Mathlib library to use its definitions and lemmas

-- Definition of the problem setup
def number_of_edges_4x4_grid : Nat := 
  let total_squares := 4 * 4
  let total_edges_per_square := 4
  total_squares * total_edges_per_square

-- Definitions of internal edges shared by adjacent squares
def distinct_edges_4x4_grid : Nat := 
  let horizontal_lines := 5 * 4
  let vertical_lines := 5 * 4
  horizontal_lines + vertical_lines

-- Calculate the maximum length of the path
def max_length_of_path_4x4_grid : Nat := 
  let degree_3_nodes := 8
  distinct_edges_4x4_grid - degree_3_nodes

-- Main statement: Prove that the maximum length of the path is 32
theorem max_path_length_is_32 : max_length_of_path_4x4_grid = 32 := by
  -- Definitions for clarity and correctness
  have h1 : number_of_edges_4x4_grid = 64 := rfl
  have h2 : distinct_edges_4x4_grid = 40 := rfl
  have h3 : max_length_of_path_4x4_grid = 32 := rfl
  exact h3

end max_path_length_is_32_l653_65319


namespace undefined_expression_l653_65302

theorem undefined_expression (a : ℝ) : (a = 3 ∨ a = -3) ↔ (a^2 - 9 = 0) := 
by
  sorry

end undefined_expression_l653_65302


namespace bob_wins_game_l653_65396

theorem bob_wins_game : 
  ∀ n : ℕ, 0 < n → 
  (∃ k ≥ 1, ∀ m : ℕ, 0 < m → (∃ a : ℕ, a ≥ 1 ∧ m - a*a = 0) ∨ 
    (∃ k : ℕ, k ≥ 1 ∧ (m = m^k → ¬ (∃ a : ℕ, a ≥ 1 ∧ m - a*a = 0)))
  ) :=
sorry

end bob_wins_game_l653_65396


namespace find_value_of_expression_l653_65331

theorem find_value_of_expression (x y : ℝ)
  (h1 : 5 * x + y = 19)
  (h2 : x + 3 * y = 1) :
  3 * x + 2 * y = 10 :=
sorry

end find_value_of_expression_l653_65331


namespace minimum_value_of_3a_plus_b_l653_65337

theorem minimum_value_of_3a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 2) : 
  3 * a + b ≥ (7 + 2 * Real.sqrt 6) / 2 :=
sorry

end minimum_value_of_3a_plus_b_l653_65337


namespace telephone_number_fraction_calculation_l653_65351

theorem telephone_number_fraction_calculation :
  let valid_phone_numbers := 7 * 10^6
  let special_phone_numbers := 10^5
  (special_phone_numbers / valid_phone_numbers : ℚ) = 1 / 70 :=
by
  sorry

end telephone_number_fraction_calculation_l653_65351


namespace car_trip_time_l653_65367

theorem car_trip_time (walking_mixed: 1.5 = 1.25 + x) 
                      (walking_both: 2.5 = 2 * 1.25) : 
  2 * x * 60 = 30 :=
by sorry

end car_trip_time_l653_65367


namespace verify_cube_modifications_l653_65318

-- Definitions and conditions from the problem
def side_length : ℝ := 9
def initial_volume : ℝ := side_length^3
def initial_surface_area : ℝ := 6 * side_length^2

def volume_remaining : ℝ := 639
def surface_area_remaining : ℝ := 510

-- The theorem proving the volume and surface area of the remaining part after carving the cross-shaped groove
theorem verify_cube_modifications :
  initial_volume - (initial_volume - volume_remaining) = 639 ∧
  510 = surface_area_remaining :=
by
  sorry

end verify_cube_modifications_l653_65318


namespace optionB_is_difference_of_squares_l653_65371

-- Definitions from conditions
def A_expr (x : ℝ) : ℝ := (x - 2) * (x + 1)
def B_expr (x y : ℝ) : ℝ := (x + 2 * y) * (x - 2 * y)
def C_expr (x y : ℝ) : ℝ := (x + y) * (-x - y)
def D_expr (x : ℝ) : ℝ := (-x + 1) * (x - 1)

theorem optionB_is_difference_of_squares (x y : ℝ) : B_expr x y = x^2 - 4 * y^2 :=
by
  -- Proof is intentionally left out as per instructions
  sorry

end optionB_is_difference_of_squares_l653_65371


namespace women_in_department_l653_65352

theorem women_in_department : 
  ∀ (total_students men women : ℕ) (men_percentage women_percentage : ℝ),
  men_percentage = 0.70 →
  women_percentage = 0.30 →
  men = 420 →
  total_students = men / men_percentage →
  women = total_students * women_percentage →
  women = 180 :=
by
  intros total_students men women men_percentage women_percentage
  intros h1 h2 h3 h4 h5
  sorry

end women_in_department_l653_65352


namespace max_two_alphas_l653_65332

theorem max_two_alphas (k : ℕ) (α : ℕ → ℝ) (hα : ∀ n, ∃! i p : ℕ, n = ⌊p * α i⌋ + 1) : k ≤ 2 := 
sorry

end max_two_alphas_l653_65332


namespace tire_price_l653_65349

theorem tire_price {p : ℤ} (h : 4 * p + 1 = 421) : p = 105 :=
sorry

end tire_price_l653_65349


namespace red_ball_probability_l653_65342

-- Define the conditions
def total_balls : ℕ := 10
def yellow_balls : ℕ := 1
def green_balls : ℕ := 3
def red_balls : ℕ := total_balls - yellow_balls - green_balls

-- Define the probability function
def probability_of_red_ball (total red : ℕ) : ℚ := red / total

-- The main theorem statement to prove
theorem red_ball_probability :
  probability_of_red_ball total_balls red_balls = 3 / 5 :=
by
  sorry

end red_ball_probability_l653_65342


namespace planes_parallel_l653_65395

theorem planes_parallel (n1 n2 : ℝ × ℝ × ℝ)
  (h1 : n1 = (2, -1, 0)) 
  (h2 : n2 = (-4, 2, 0)) :
  ∃ k : ℝ, n2 = k • n1 := by
  -- Proof is beyond the scope of this exercise.
  sorry

end planes_parallel_l653_65395


namespace Nishita_preferred_shares_l653_65388

variable (P : ℕ)

def preferred_share_dividend : ℕ := 5 * P
def common_share_dividend : ℕ := 3500 * 3  -- 3.5 * 1000

theorem Nishita_preferred_shares :
  preferred_share_dividend P + common_share_dividend = 16500 → P = 1200 :=
by
  unfold preferred_share_dividend common_share_dividend
  intro h
  sorry

end Nishita_preferred_shares_l653_65388


namespace cleaning_time_ratio_l653_65323

/-- 
Given that Lilly and Fiona together take a total of 480 minutes to clean a room and Fiona
was cleaning for 360 minutes, prove that the ratio of the time Lilly spent cleaning 
to the total time spent cleaning the room is 1:4.
-/
theorem cleaning_time_ratio (total_time minutes Fiona_time : ℕ) 
  (h1 : total_time = 480)
  (h2 : Fiona_time = 360) : 
  (total_time - Fiona_time) / total_time = 1 / 4 :=
by
  sorry

end cleaning_time_ratio_l653_65323


namespace number_of_rooms_l653_65333

theorem number_of_rooms (x : ℕ) (h1 : ∀ n, 6 * (n - 1) = 5 * n + 4) : x = 10 :=
sorry

end number_of_rooms_l653_65333


namespace correct_operation_l653_65325

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := 
by 
  sorry

end correct_operation_l653_65325


namespace cubics_inequality_l653_65389

theorem cubics_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 :=
sorry

end cubics_inequality_l653_65389


namespace crossing_time_correct_l653_65326

def length_of_train : ℝ := 150 -- Length of the train in meters
def speed_of_train_km_per_hr : ℝ := 72 -- Speed of the train in km/hr
def length_of_bridge : ℝ := 132 -- Length of the bridge in meters

noncomputable def speed_of_train_m_per_s : ℝ := (speed_of_train_km_per_hr * 1000) / 3600 -- Speed of the train in m/s

noncomputable def time_to_cross_bridge : ℝ := (length_of_train + length_of_bridge) / speed_of_train_m_per_s -- Time in seconds

theorem crossing_time_correct : time_to_cross_bridge = 14.1 := by
  sorry

end crossing_time_correct_l653_65326


namespace carlos_marbles_l653_65336

theorem carlos_marbles :
  ∃ N : ℕ, N > 2 ∧
  (N % 6 = 2) ∧
  (N % 7 = 2) ∧
  (N % 8 = 2) ∧
  (N % 11 = 2) ∧
  N = 3698 :=
by
  sorry

end carlos_marbles_l653_65336


namespace natural_number_triplets_l653_65312

theorem natural_number_triplets :
  ∀ (a b c : ℕ), a^3 + b^3 + c^3 = (a * b * c)^2 → 
    (a = 3 ∧ b = 2 ∧ c = 1) ∨ (a = 3 ∧ b = 1 ∧ c = 2) ∨ 
    (a = 2 ∧ b = 3 ∧ c = 1) ∨ (a = 2 ∧ b = 1 ∧ c = 3) ∨ 
    (a = 1 ∧ b = 3 ∧ c = 2) ∨ (a = 1 ∧ b = 2 ∧ c = 3) := 
by
  sorry

end natural_number_triplets_l653_65312


namespace expenses_recorded_as_negative_l653_65391

/-*
  Given:
  1. The income of 5 yuan is recorded as +5 yuan.
  Prove:
  2. The expenses of 5 yuan are recorded as -5 yuan.
*-/

theorem expenses_recorded_as_negative (income_expenses_opposite_sign : ∀ (a : ℤ), -a = -a)
    (income_five_recorded_as_positive : (5 : ℤ) = 5) :
    (-5 : ℤ) = -5 :=
by sorry

end expenses_recorded_as_negative_l653_65391


namespace smallest_part_of_80_divided_by_proportion_l653_65390

theorem smallest_part_of_80_divided_by_proportion (x : ℕ) (h1 : 1 * x + 3 * x + 5 * x + 7 * x = 80) : x = 5 :=
sorry

end smallest_part_of_80_divided_by_proportion_l653_65390


namespace range_of_a_plus_b_l653_65380

theorem range_of_a_plus_b (a b : ℝ) (h : |a| + |b| + |a - 1| + |b - 1| ≤ 2) : 
  0 ≤ a + b ∧ a + b ≤ 2 :=
sorry

end range_of_a_plus_b_l653_65380


namespace count_valid_three_digit_numbers_l653_65375

theorem count_valid_three_digit_numbers : 
  let total_three_digit_numbers := 900 
  let invalid_AAB_or_ABA := 81 + 81
  total_three_digit_numbers - invalid_AAB_or_ABA = 738 := 
by 
  let total_three_digit_numbers := 900
  let invalid_AAB_or_ABA := 81 + 81
  show total_three_digit_numbers - invalid_AAB_or_ABA = 738 
  sorry

end count_valid_three_digit_numbers_l653_65375


namespace shifts_needed_l653_65356

-- Given definitions
def total_workers : ℕ := 12
def workers_per_shift : ℕ := 2
def total_ways_to_assign : ℕ := 23760

-- Prove the number of shifts needed
theorem shifts_needed : total_workers / workers_per_shift = 6 := by
  sorry

end shifts_needed_l653_65356


namespace distance_between_A_and_B_l653_65321

theorem distance_between_A_and_B 
    (Time_E : ℝ) (Time_F : ℝ) (D_AC : ℝ) (V_ratio : ℝ)
    (E_time : Time_E = 3) (F_time : Time_F = 4) 
    (AC_distance : D_AC = 300) (speed_ratio : V_ratio = 4) : 
    ∃ D_AB : ℝ, D_AB = 900 :=
by
  sorry

end distance_between_A_and_B_l653_65321


namespace reeya_fourth_subject_score_l653_65373

theorem reeya_fourth_subject_score (s1 s2 s3 s4 : ℕ) (avg : ℕ) (n : ℕ)
  (h_avg : avg = 75) (h_n : n = 4) (h_s1 : s1 = 65) (h_s2 : s2 = 67) (h_s3 : s3 = 76)
  (h_total_sum : avg * n = s1 + s2 + s3 + s4) : s4 = 92 := by
  sorry

end reeya_fourth_subject_score_l653_65373


namespace transylvanian_is_sane_human_l653_65385

def Transylvanian : Type := sorry -- Placeholder type for Transylvanian
def Human : Transylvanian → Prop := sorry
def Sane : Transylvanian → Prop := sorry
def InsaneVampire : Transylvanian → Prop := sorry

/-- The Transylvanian stated: "Either I am a human, or I am sane." -/
axiom statement (T : Transylvanian) : Human T ∨ Sane T

/-- Insane vampires only make true statements. -/
axiom insane_vampire_truth (T : Transylvanian) : InsaneVampire T → (Human T ∨ Sane T)

/-- Insane vampires cannot be sane or human. -/
axiom insane_vampire_condition (T : Transylvanian) : InsaneVampire T → ¬ Human T ∧ ¬ Sane T

theorem transylvanian_is_sane_human (T : Transylvanian) :
  ¬ (InsaneVampire T) → (Human T ∧ Sane T) := sorry

end transylvanian_is_sane_human_l653_65385


namespace triangle_side_b_l653_65374

theorem triangle_side_b (A B C a b c : ℝ)
  (hA : A = 135)
  (hc : c = 1)
  (hSinB_SinC : Real.sin B * Real.sin C = Real.sqrt 2 / 10) :
  b = Real.sqrt 2 ∨ b = Real.sqrt 2 / 2 :=
by
  sorry

end triangle_side_b_l653_65374


namespace probability_of_purple_marble_l653_65363

theorem probability_of_purple_marble 
  (P_blue : ℝ) 
  (P_green : ℝ) 
  (P_purple : ℝ) 
  (h1 : P_blue = 0.25) 
  (h2 : P_green = 0.55) 
  (h3 : P_blue + P_green + P_purple = 1) 
  : P_purple = 0.20 := 
by 
  sorry

end probability_of_purple_marble_l653_65363


namespace sequence_add_l653_65348

theorem sequence_add (x y : ℝ) (h1 : x = 81 * (1 / 3)) (h2 : y = x * (1 / 3)) : x + y = 36 :=
sorry

end sequence_add_l653_65348


namespace time_to_cross_signal_post_l653_65355

def train_length := 600 -- in meters
def bridge_length := 5400 -- in meters (5.4 kilometers)
def crossing_time_bridge := 6 * 60 -- in seconds (6 minutes)
def speed := bridge_length / crossing_time_bridge -- in meters per second

theorem time_to_cross_signal_post : 
  (600 / speed) = 40 :=
by
  sorry

end time_to_cross_signal_post_l653_65355


namespace fiona_pairs_l653_65369

-- Define the combinatorial calculation using the combination formula
def combination (n k : ℕ) := n.choose k

-- The main theorem stating that the number of pairs from 6 people is 15
theorem fiona_pairs : combination 6 2 = 15 :=
by
  sorry

end fiona_pairs_l653_65369


namespace foma_wait_time_probability_l653_65376

noncomputable def probability_no_more_than_four_minutes_wait (x y : ℝ) : ℝ :=
if h : 2 < x ∧ x < y ∧ y < 10 ∧ y - x ≤ 4 then
  (1 / 2)
else 0

theorem foma_wait_time_probability :
  ∀ (x y : ℝ), 2 < x → x < y → y < 10 → 
  (probability_no_more_than_four_minutes_wait x y) = 1 / 2 :=
sorry

end foma_wait_time_probability_l653_65376


namespace pencil_pen_costs_l653_65381

noncomputable def cost_of_items (p q : ℝ) : ℝ := 4 * p + 4 * q

theorem pencil_pen_costs (p q : ℝ) (h1 : 6 * p + 3 * q = 5.40) (h2 : 3 * p + 5 * q = 4.80) : cost_of_items p q = 4.80 :=
by
  sorry

end pencil_pen_costs_l653_65381


namespace g_at_10_is_neg48_l653_65360

variable (g : ℝ → ℝ)

-- Given condition
axiom functional_eqn : ∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2

-- Mathematical proof statement
theorem g_at_10_is_neg48 : g 10 = -48 :=
  sorry

end g_at_10_is_neg48_l653_65360


namespace alpha_in_fourth_quadrant_l653_65339

def point_in_third_quadrant (α : ℝ) : Prop :=
  (Real.tan α < 0) ∧ (Real.sin α < 0)

theorem alpha_in_fourth_quadrant (α : ℝ) (h : point_in_third_quadrant α) : 
  α ∈ Set.Ioc (3 * Real.pi / 2) (2 * Real.pi) :=
by sorry

end alpha_in_fourth_quadrant_l653_65339


namespace ax0_eq_b_condition_l653_65358

theorem ax0_eq_b_condition (a b x0 : ℝ) (h : a < 0) : (ax0 = b) ↔ (∀ x : ℝ, (1/2 * a * x^2 - b * x) ≤ (1/2 * a * x0^2 - b * x0)) :=
sorry

end ax0_eq_b_condition_l653_65358


namespace B_work_rate_l653_65393

theorem B_work_rate (B : ℕ) (A_rate C_rate : ℚ) 
  (A_work : A_rate = 1 / 6)
  (C_work : C_rate = 1 / 8 * (1 / 6 + 1 / B))
  (combined_work : 1 / 6 + 1 / B + C_rate = 1 / 3) : 
  B = 28 :=
by 
  sorry

end B_work_rate_l653_65393


namespace smallest_n_congruent_5n_eq_n5_mod_7_l653_65316

theorem smallest_n_congruent_5n_eq_n5_mod_7 : ∃ (n : ℕ), n > 0 ∧ (∀ m > 0, 5^m % 7 ≠ m^5 % 7 → m ≥ n) :=
by
  use 6
  -- Proof steps here which are skipped
  sorry

end smallest_n_congruent_5n_eq_n5_mod_7_l653_65316
