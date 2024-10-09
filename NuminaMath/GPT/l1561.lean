import Mathlib

namespace sum_of_integers_is_eleven_l1561_156169

theorem sum_of_integers_is_eleven (p q r s : ℤ) 
  (h1 : p - q + r = 7) 
  (h2 : q - r + s = 8) 
  (h3 : r - s + p = 4) 
  (h4 : s - p + q = 3) : 
  p + q + r + s = 11 := 
by
  sorry

end sum_of_integers_is_eleven_l1561_156169


namespace Nell_initial_cards_l1561_156187

theorem Nell_initial_cards (n : ℕ) (h1 : n - 136 = 106) : n = 242 := 
by
  sorry

end Nell_initial_cards_l1561_156187


namespace max_integer_value_fraction_l1561_156167

theorem max_integer_value_fraction (x : ℝ) : 
  (∃ t : ℤ, t = 2 ∧ (∀ y : ℝ, y = (4*x^2 + 8*x + 21) / (4*x^2 + 8*x + 9) → y <= t)) :=
sorry

end max_integer_value_fraction_l1561_156167


namespace calculate_value_of_expression_l1561_156197

theorem calculate_value_of_expression :
  3.5 * 7.2 * (6.3 - 1.4) = 122.5 :=
  by
  sorry

end calculate_value_of_expression_l1561_156197


namespace all_points_lie_on_line_l1561_156168

theorem all_points_lie_on_line:
  ∀ (s : ℝ), s ≠ 0 → ∀ (x y : ℝ),
  x = (2 * s + 3) / s → y = (2 * s - 3) / s → x + y = 4 :=
by
  intros s hs x y hx hy
  sorry

end all_points_lie_on_line_l1561_156168


namespace unit_vector_parallel_to_a_l1561_156118

theorem unit_vector_parallel_to_a (x y : ℝ) (h1 : x^2 + y^2 = 1) (h2 : 12 * y = 5 * x) :
  (x = 12 / 13 ∧ y = 5 / 13) ∨ (x = -12 / 13 ∧ y = -5 / 13) := by
  sorry

end unit_vector_parallel_to_a_l1561_156118


namespace marys_total_cards_l1561_156103

def initial_cards : ℕ := 18
def torn_cards : ℕ := 8
def cards_from_fred : ℕ := 26
def cards_bought_by_mary : ℕ := 40

theorem marys_total_cards :
  initial_cards - torn_cards + cards_from_fred + cards_bought_by_mary = 76 :=
by
  sorry

end marys_total_cards_l1561_156103


namespace probability_no_defective_pencils_l1561_156170

theorem probability_no_defective_pencils :
  let total_pencils := 9
  let defective_pencils := 2
  let total_ways_choose_3 := Nat.choose total_pencils 3
  let non_defective_pencils := total_pencils - defective_pencils
  let ways_choose_3_non_defective := Nat.choose non_defective_pencils 3
  (ways_choose_3_non_defective : ℚ) / total_ways_choose_3 = 5 / 12 :=
by
  sorry

end probability_no_defective_pencils_l1561_156170


namespace find_number_l1561_156175

theorem find_number (x : ℕ) (h1 : x > 7) (h2 : x ≠ 8) : x = 9 := by
  sorry

end find_number_l1561_156175


namespace find_students_with_equal_homework_hours_l1561_156198

theorem find_students_with_equal_homework_hours :
  let Dan := 6
  let Joe := 3
  let Bob := 5
  let Susie := 4
  let Grace := 1
  (Joe + Grace = Dan ∨ Joe + Bob = Dan ∨ Bob + Grace = Dan ∨ Dan + Bob = Dan ∨ Susie + Grace = Dan) → 
  (Bob + Grace = Dan) := 
by 
  intros
  sorry

end find_students_with_equal_homework_hours_l1561_156198


namespace functions_satisfying_equation_are_constants_l1561_156179

theorem functions_satisfying_equation_are_constants (f g : ℝ → ℝ) :
  (∀ x y : ℝ, f (f (x + y)) = x * f y + g x) → ∃ k : ℝ, (∀ x : ℝ, f x = k) ∧ (∀ x : ℝ, g x = k * (1 - x)) :=
by
  sorry

end functions_satisfying_equation_are_constants_l1561_156179


namespace clock_angle_at_3_40_l1561_156144

theorem clock_angle_at_3_40
  (hour_position : ℕ → ℝ)
  (minute_position : ℕ → ℝ)
  (h_hour : hour_position 3 = 3 * 30)
  (h_minute : minute_position 40 = 40 * 6)
  : abs (minute_position 40 - (hour_position 3 + 20 * 30 / 60)) = 130 :=
by
  -- Insert proof here
  sorry

end clock_angle_at_3_40_l1561_156144


namespace incorrect_statements_are_1_2_4_l1561_156155

theorem incorrect_statements_are_1_2_4:
    let statements := ["Inductive reasoning and analogical reasoning both involve reasoning from specific to general.",
                       "When making an analogy, it is more appropriate to use triangles in a plane and parallelepipeds in space as the objects of analogy.",
                       "'All multiples of 9 are multiples of 3, if a number m is a multiple of 9, then m must be a multiple of 3' is an example of syllogistic reasoning.",
                       "In deductive reasoning, as long as it follows the form of deductive reasoning, the conclusion is always correct."]
    let incorrect_statements := {1, 2, 4}
    incorrect_statements = {i | i ∈ [1, 2, 3, 4] ∧
                             ((i = 1 → ¬(∃ s, s ∈ statements ∧ s = statements[0])) ∧ 
                              (i = 2 → ¬(∃ s, s ∈ statements ∧ s = statements[1])) ∧ 
                              (i = 3 → ∃ s, s ∈ statements ∧ s = statements[2]) ∧ 
                              (i = 4 → ¬(∃ s, s ∈ statements ∧ s = statements[3])))} :=
by
  sorry

end incorrect_statements_are_1_2_4_l1561_156155


namespace probability_of_different_suits_l1561_156127

-- Let’s define the parameters of the problem
def total_cards : ℕ := 104
def first_card_remaining : ℕ := 103
def same_suit_cards : ℕ := 26
def different_suit_cards : ℕ := first_card_remaining - same_suit_cards

-- The probability that the two cards drawn are of different suits
def probability_different_suits : ℚ := different_suit_cards / first_card_remaining

-- The main statement to prove
theorem probability_of_different_suits :
  probability_different_suits = 78 / 103 :=
by {
  -- The proof would go here
  sorry
}

end probability_of_different_suits_l1561_156127


namespace carlos_goals_product_l1561_156182

theorem carlos_goals_product :
  ∃ (g11 g12 : ℕ), g11 < 8 ∧ g12 < 8 ∧ 
  (33 + g11) % 11 = 0 ∧ 
  (33 + g11 + g12) % 12 = 0 ∧ 
  g11 * g12 = 49 := 
by
  sorry

end carlos_goals_product_l1561_156182


namespace total_letters_correct_l1561_156156

-- Define the conditions
def letters_January := 6
def letters_February := 9
def letters_March := 3 * letters_January

-- Definition of the total number of letters sent
def total_letters := letters_January + letters_February + letters_March

-- The statement we need to prove in Lean
theorem total_letters_correct : total_letters = 33 := 
by
  sorry

end total_letters_correct_l1561_156156


namespace a4_equals_zero_l1561_156106

-- Define the general term of the sequence
def a (n : ℕ) (h : n > 0) : ℤ := n^2 - 3 * n - 4

-- The theorem statement to prove a_4 = 0
theorem a4_equals_zero : a 4 (by norm_num) = 0 :=
sorry

end a4_equals_zero_l1561_156106


namespace train_passing_time_l1561_156128

theorem train_passing_time
  (length_of_train : ℝ)
  (speed_in_kmph : ℝ)
  (conversion_factor : ℝ)
  (speed_in_mps : ℝ)
  (time : ℝ)
  (H1 : length_of_train = 65)
  (H2 : speed_in_kmph = 36)
  (H3 : conversion_factor = 5 / 18)
  (H4 : speed_in_mps = speed_in_kmph * conversion_factor)
  (H5 : time = length_of_train / speed_in_mps) :
  time = 6.5 :=
by
  sorry

end train_passing_time_l1561_156128


namespace proportion_equal_l1561_156183

theorem proportion_equal (x : ℝ) : (0.25 / x = 2 / 6) → x = 0.75 :=
by
  sorry

end proportion_equal_l1561_156183


namespace elena_snow_removal_l1561_156195

theorem elena_snow_removal :
  ∀ (length width depth : ℝ) (compaction_factor : ℝ), 
  length = 30 ∧ width = 3 ∧ depth = 0.75 ∧ compaction_factor = 0.90 → 
  (length * width * depth * compaction_factor = 60.75) :=
by
  intros length width depth compaction_factor h
  obtain ⟨length_eq, width_eq, depth_eq, compaction_factor_eq⟩ := h
  -- Proof steps go here
  sorry

end elena_snow_removal_l1561_156195


namespace snail_crawl_distance_l1561_156147

theorem snail_crawl_distance
  (α : ℕ → ℝ)  -- α represents the snail's position at each minute
  (crawls_forward : ∀ n m : ℕ, n < m → α n ≤ α m)  -- The snail moves forward (without going backward)
  (observer_finds : ∀ n : ℕ, α (n + 1) - α n = 1) -- Every observer finds that the snail crawled exactly 1 meter per minute
  (time_span : ℕ := 6)  -- Total observation period is 6 minutes
  : α time_span - α 0 ≤ 10 :=  -- The distance crawled in 6 minutes does not exceed 10 meters
by
  -- Proof goes here
  sorry

end snail_crawl_distance_l1561_156147


namespace math_problem_l1561_156125

theorem math_problem (x : ℤ) (h : x = 9) :
  (x^6 - 27*x^3 + 729) / (x^3 - 27) = 702 :=
by
  sorry

end math_problem_l1561_156125


namespace expression_equals_neg_one_l1561_156102

theorem expression_equals_neg_one (a b c : ℝ) (h : a + b + c = 0) :
  (|a| / a) + (|b| / b) + (|c| / c) + (|a * b| / (a * b)) + (|a * c| / (a * c)) + (|b * c| / (b * c)) + (|a * b * c| / (a * b * c)) = -1 :=
  sorry

end expression_equals_neg_one_l1561_156102


namespace cody_games_still_has_l1561_156138

def initial_games : ℕ := 9
def games_given_away_to_jake : ℕ := 4
def games_given_away_to_sarah : ℕ := 2
def games_bought_over_weekend : ℕ := 3

theorem cody_games_still_has : 
  initial_games - (games_given_away_to_jake + games_given_away_to_sarah) + games_bought_over_weekend = 6 := 
by
  sorry

end cody_games_still_has_l1561_156138


namespace smallest_four_digit_divisible_by_4_and_5_l1561_156100

theorem smallest_four_digit_divisible_by_4_and_5 : 
  ∃ n, (n % 4 = 0) ∧ (n % 5 = 0) ∧ 1000 ≤ n ∧ n < 10000 ∧ 
  ∀ m, (m % 4 = 0) ∧ (m % 5 = 0) ∧ 1000 ≤ m ∧ m < 10000 → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_4_and_5_l1561_156100


namespace garden_area_remaining_l1561_156109

variable (d : ℕ) (w : ℕ) (t : ℕ)

theorem garden_area_remaining (r : Real) (A_circle : Real) 
                              (A_path : Real) (A_remaining : Real) :
  r = 10 →
  A_circle = 100 * Real.pi →
  A_path = 66.66 * Real.pi - 50 * Real.sqrt 3 →
  A_remaining = 33.34 * Real.pi + 50 * Real.sqrt 3 :=
by
  -- Given the radius of the garden
  let r := (d : Real) / 2
  -- Calculate the total area of the garden
  let A_circle := Real.pi * r^2
  -- Area covered by the path computed using circular segments
  let A_path := 66.66 * Real.pi - 50 * Real.sqrt 3
  -- Remaining garden area
  let A_remaining := A_circle - A_path
  -- Statement to prove correct
  sorry 

end garden_area_remaining_l1561_156109


namespace correct_answer_l1561_156176

def mary_initial_cards : ℝ := 18.0
def mary_bought_cards : ℝ := 40.0
def mary_left_cards : ℝ := 32.0
def mary_promised_cards (initial_cards : ℝ) (bought_cards : ℝ) (left_cards : ℝ) : ℝ :=
  initial_cards + bought_cards - left_cards

theorem correct_answer :
  mary_promised_cards mary_initial_cards mary_bought_cards mary_left_cards = 26.0 := by
  sorry

end correct_answer_l1561_156176


namespace prob_none_three_win_prob_at_least_two_not_win_l1561_156193

-- Definitions for probabilities
def prob_win : ℚ := 1 / 6
def prob_not_win : ℚ := 1 - prob_win

-- Problem 1: Prove probability that none of the three students win
theorem prob_none_three_win : (prob_not_win ^ 3) = 125 / 216 := by
  sorry

-- Problem 2: Prove probability that at least two of the three students do not win
theorem prob_at_least_two_not_win : 1 - (3 * (prob_win ^ 2) * prob_not_win + prob_win ^ 3) = 25 / 27 := by
  sorry

end prob_none_three_win_prob_at_least_two_not_win_l1561_156193


namespace cookie_radius_l1561_156163

theorem cookie_radius (x y : ℝ) (h : x^2 + y^2 + 2 * x - 4 * y = 4) : 
  ∃ r : ℝ, (x + 1)^2 + (y - 2)^2 = r^2 ∧ r = 3 := by
  sorry

end cookie_radius_l1561_156163


namespace least_possible_b_l1561_156110

theorem least_possible_b (a b : ℕ) (h1 : a + b = 120) (h2 : (Prime a ∨ ∃ p : ℕ, Prime p ∧ a = 2 * p)) (h3 : Prime b) (h4 : a > b) : b = 7 :=
sorry

end least_possible_b_l1561_156110


namespace area_of_rhombus_l1561_156117

-- Defining the lengths of the diagonals
variable (d1 d2 : ℝ)
variable (d1_eq : d1 = 15)
variable (d2_eq : d2 = 20)

-- Goal is to prove the area given the diagonal lengths
theorem area_of_rhombus (d1 d2 : ℝ) (d1_eq : d1 = 15) (d2_eq : d2 = 20) : 
  (d1 * d2) / 2 = 150 := 
by
  -- Using the given conditions for the proof
  sorry

end area_of_rhombus_l1561_156117


namespace movie_final_length_l1561_156111

theorem movie_final_length (original_length : ℕ) (cut_length : ℕ) (final_length : ℕ) 
  (h1 : original_length = 60) (h2 : cut_length = 8) : 
  final_length = 52 :=
by
  sorry

end movie_final_length_l1561_156111


namespace range_of_a_l1561_156189

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, (x^2 + (a^2 + 1) * x + a - 2 = 0 ∧ y^2 + (a^2 + 1) * y + a - 2 = 0)
    ∧ x > 1 ∧ y < -1) ↔ (-1 < a ∧ a < 0) := sorry

end range_of_a_l1561_156189


namespace horner_method_value_v2_at_minus_one_l1561_156180

noncomputable def f (x : ℝ) : ℝ :=
  x^6 - 5*x^5 + 6*x^4 - 3*x^3 + 1.8*x^2 + 0.35*x + 2

theorem horner_method_value_v2_at_minus_one :
  let a : ℝ := -1
  let v_0 := 1
  let v_1 := v_0 * a - 5
  let v_2 := v_1 * a + 6
  v_2 = 12 :=
by
  intros
  sorry

end horner_method_value_v2_at_minus_one_l1561_156180


namespace decreasing_f_range_l1561_156185

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

theorem decreasing_f_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) → (0 < a ∧ a ≤ 1/4) :=
by
  sorry

end decreasing_f_range_l1561_156185


namespace ram_krish_task_completion_l1561_156154

/-!
  Given:
  1. Ram's efficiency (R) is half of Krish's efficiency (K).
  2. Ram can complete the task alone in 24 days.

  To Prove:
  Ram and Krish will complete the task together in 8 days.
-/

theorem ram_krish_task_completion {R K : ℝ} (hR : R = 1 / 2 * K)
  (hRAMalone : R ≠ 0) (hRAMtime : 24 * R = 1) :
  1 / (R + K) = 8 := by
  sorry

end ram_krish_task_completion_l1561_156154


namespace trigonometric_identity_l1561_156112

theorem trigonometric_identity (α : Real) (h : (1 + Real.sin α) / Real.cos α = -1 / 2) :
  (Real.cos α) / (Real.sin α - 1) = 1 / 2 :=
sorry

end trigonometric_identity_l1561_156112


namespace express_recurring_decimal_as_fraction_l1561_156142

theorem express_recurring_decimal_as_fraction (h : 0.01 = (1 : ℚ) / 99) : 2.02 = (200 : ℚ) / 99 :=
by 
  sorry

end express_recurring_decimal_as_fraction_l1561_156142


namespace number_of_lemons_l1561_156157

theorem number_of_lemons
  (total_fruits : ℕ)
  (mangoes : ℕ)
  (pears : ℕ)
  (pawpaws : ℕ)
  (kiwis : ℕ)
  (lemons : ℕ)
  (h_total : total_fruits = 58)
  (h_mangoes : mangoes = 18)
  (h_pears : pears = 10)
  (h_pawpaws : pawpaws = 12)
  (h_kiwis_lemons_equal : kiwis = lemons) :
  lemons = 9 :=
by
  sorry

end number_of_lemons_l1561_156157


namespace radius_of_circle_eqn_zero_l1561_156152

def circle_eqn (x y : ℝ) := x^2 + 8*x + y^2 - 4*y + 20 = 0

theorem radius_of_circle_eqn_zero :
  ∀ x y : ℝ, circle_eqn x y → ∃ r : ℝ, r = 0 :=
by
  intros x y h
  -- Sorry to skip the proof as per instructions
  sorry

end radius_of_circle_eqn_zero_l1561_156152


namespace board_division_condition_l1561_156174

open Nat

theorem board_division_condition (n : ℕ) : 
  (∃ k : ℕ, n = 4 * k) ↔ 
  (∃ v h : ℕ, v = h ∧ (2 * v + 2 * h = n * n ∧ n % 2 = 0)) := 
sorry

end board_division_condition_l1561_156174


namespace find_m_l1561_156148

theorem find_m (x y m : ℤ) (h1 : x = 2) (h2 : y = -3) (h3 : 3 * x - 4 * (m - 1) * y + 30 = 0) : m = -2 :=
by
  sorry

end find_m_l1561_156148


namespace darwin_spending_fraction_l1561_156136

theorem darwin_spending_fraction {x : ℝ} (h1 : 600 - 600 * x - (1 / 4) * (600 - 600 * x) = 300) :
  x = 1 / 3 :=
sorry

end darwin_spending_fraction_l1561_156136


namespace pen_distribution_l1561_156171

theorem pen_distribution (x : ℕ) :
  8 * x + 3 = 12 * (x - 2) - 1 :=
sorry

end pen_distribution_l1561_156171


namespace exists_square_with_only_invisible_points_l1561_156146

def is_invisible (p q : ℤ) : Prop := Int.gcd p q > 1

def all_points_in_square_invisible (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≥ 2 ∧ ∀ x y : ℕ, (x < n ∧ y < n) → is_invisible (k*x) (k*y)

theorem exists_square_with_only_invisible_points (n : ℕ) :
  all_points_in_square_invisible n := sorry

end exists_square_with_only_invisible_points_l1561_156146


namespace find_m_l1561_156151

theorem find_m (m : ℝ) (x1 x2 : ℝ) 
  (h_eq : x1 ^ 2 - 4 * x1 - 2 * m + 5 = 0)
  (h_distinct : x1 ≠ x2)
  (h_product_sum_eq : x1 * x2 + x1 + x2 = m ^ 2 + 6) : 
  m = 1 ∧ m > 1/2 :=
sorry

end find_m_l1561_156151


namespace find_base_b_l1561_156186

theorem find_base_b (b : ℕ) : ( (2 * b + 5) ^ 2 = 6 * b ^ 2 + 5 * b + 5 ) → b = 9 := 
by 
  sorry  -- Proof is not required as per instruction

end find_base_b_l1561_156186


namespace part1_part2_l1561_156164

def is_regressive_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, x n + x (n + 2) - x (n + 1) = x m

theorem part1 (a : ℕ → ℝ) (h : ∀ n : ℕ, a n = 3 ^ n) :
  ¬ is_regressive_sequence a := by
  sorry

theorem part2 (b : ℕ → ℝ) (h_reg : is_regressive_sequence b) (h_inc : ∀ n : ℕ, b n < b (n + 1)) :
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d := by
  sorry

end part1_part2_l1561_156164


namespace proof_equivalent_l1561_156123

variables {α : Type*} [Field α]

theorem proof_equivalent (a b c d e f : α)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1 :=
by sorry

end proof_equivalent_l1561_156123


namespace sum_of_squares_of_four_integers_equals_175_l1561_156105

theorem sum_of_squares_of_four_integers_equals_175 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a^2 + b^2 + c^2 + d^2 = 175 ∧ a + b + c + d = 23 :=
sorry

end sum_of_squares_of_four_integers_equals_175_l1561_156105


namespace mans_rate_in_still_water_l1561_156141

theorem mans_rate_in_still_water
  (V_m V_s : ℝ)
  (h1 : V_m + V_s = 20)
  (h2 : V_m - V_s = 4) :
  V_m = 12 :=
by
  sorry

end mans_rate_in_still_water_l1561_156141


namespace sphere_cone_radius_ratio_l1561_156158

-- Define the problem using given conditions and expected outcome.
theorem sphere_cone_radius_ratio (r R h : ℝ)
  (h1 : h = 2 * r)
  (h2 : (1/3) * π * R^2 * h = 3 * (4/3) * π * r^3) :
  r / R = 1 / Real.sqrt 6 :=
by
  sorry

end sphere_cone_radius_ratio_l1561_156158


namespace discount_is_5_percent_l1561_156143

-- Defining the conditions
def cost_per_iphone : ℕ := 600
def total_cost_3_iphones : ℕ := 3 * cost_per_iphone
def savings : ℕ := 90

-- Calculating the discount percentage
def discount_percentage : ℕ := (savings * 100) / total_cost_3_iphones

-- Stating the theorem
theorem discount_is_5_percent : discount_percentage = 5 :=
  sorry

end discount_is_5_percent_l1561_156143


namespace maximize_distance_l1561_156149

theorem maximize_distance (front_tires_lifetime: ℕ) (rear_tires_lifetime: ℕ):
  front_tires_lifetime = 20000 → rear_tires_lifetime = 30000 → 
  ∃ D, D = 30000 :=
by
  sorry

end maximize_distance_l1561_156149


namespace heartsuit_4_6_l1561_156161

-- Define the operation \heartsuit
def heartsuit (x y : ℤ) : ℤ := 5 * x + 3 * y

-- Prove that 4 \heartsuit 6 = 38 under the given operation definition
theorem heartsuit_4_6 : heartsuit 4 6 = 38 := by
  -- Using the definition of \heartsuit
  -- Calculation is straightforward and skipped by sorry
  sorry

end heartsuit_4_6_l1561_156161


namespace factor_is_2_l1561_156192

variable (x : ℕ) (f : ℕ)

theorem factor_is_2 (h₁ : x = 36)
                    (h₂ : ((f * (x + 10)) / 2) - 2 = 44) : f = 2 :=
by {
  sorry
}

end factor_is_2_l1561_156192


namespace baskets_weight_l1561_156135

theorem baskets_weight 
  (weight_per_basket : ℕ)
  (num_baskets : ℕ)
  (total_weight : ℕ) 
  (h1 : weight_per_basket = 30)
  (h2 : num_baskets = 8)
  (h3 : total_weight = weight_per_basket * num_baskets) :
  total_weight = 240 := 
by
  sorry

end baskets_weight_l1561_156135


namespace unique_solution_f_geq_0_inequality_hold_for_a_leq_1_l1561_156145

noncomputable def f (x k : ℝ) : ℝ := (Real.log x) - k * x + k

theorem unique_solution_f_geq_0 {k : ℝ} :
  (∃! x : ℝ, 0 < x ∧ f x k ≥ 0) ↔ k = 1 :=
sorry

theorem inequality_hold_for_a_leq_1 {a x : ℝ} (h₀ : a ≤ 1) :
  x * (f x 1 + x - 1) < Real.exp x - a * x^2 - 1 :=
sorry

end unique_solution_f_geq_0_inequality_hold_for_a_leq_1_l1561_156145


namespace find_percentage_l1561_156194

theorem find_percentage (P N : ℕ) (h1 : N = 100) (h2 : (P : ℝ) / 100 * N = 50 / 100 * 40 + 10) :
  P = 30 :=
by
  sorry

end find_percentage_l1561_156194


namespace polynomial_expansion_sum_eq_l1561_156126

theorem polynomial_expansion_sum_eq :
  (∀ (x : ℝ), (2 * x - 1)^5 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) →
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 243) :=
by
  sorry

end polynomial_expansion_sum_eq_l1561_156126


namespace equation_solutions_l1561_156153

theorem equation_solutions (a b : ℝ) (h : a + b = 0) :
  (∃ x : ℝ, ax + b = 0) ∨ (∃ x : ℝ, ∀ y : ℝ, ax + b = 0 → x = y) :=
sorry

end equation_solutions_l1561_156153


namespace donut_distribution_l1561_156120

theorem donut_distribution :
  ∃ (Alpha Beta Gamma Delta Epsilon : ℕ), 
    Delta = 8 ∧ 
    Beta = 3 * Gamma ∧ 
    Alpha = 2 * Delta ∧ 
    Epsilon = Gamma - 4 ∧ 
    Alpha + Beta + Gamma + Delta + Epsilon = 60 ∧ 
    Alpha = 16 ∧ 
    Beta = 24 ∧ 
    Gamma = 8 ∧ 
    Delta = 8 ∧ 
    Epsilon = 4 :=
by
  sorry

end donut_distribution_l1561_156120


namespace gcd_three_numbers_l1561_156188

def a : ℕ := 8650
def b : ℕ := 11570
def c : ℕ := 28980

theorem gcd_three_numbers : Nat.gcd (Nat.gcd a b) c = 10 :=
by 
  sorry

end gcd_three_numbers_l1561_156188


namespace Lisa_total_spoons_l1561_156139

def total_spoons (children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) (large_spoons : ℕ) (teaspoons : ℕ) : ℕ := 
  (children * spoons_per_child) + decorative_spoons + (large_spoons + teaspoons)

theorem Lisa_total_spoons :
  (total_spoons 4 3 2 10 15) = 39 :=
by
  sorry

end Lisa_total_spoons_l1561_156139


namespace chocolate_bar_cost_l1561_156173

theorem chocolate_bar_cost (x : ℝ) (total_bars : ℕ) (bars_sold : ℕ) (total_amount_made : ℝ)
    (h1 : total_bars = 7)
    (h2 : bars_sold = total_bars - 4)
    (h3 : total_amount_made = 9)
    (h4 : total_amount_made = bars_sold * x) : x = 3 :=
sorry

end chocolate_bar_cost_l1561_156173


namespace product_remainder_l1561_156101

theorem product_remainder
    (a b c : ℕ)
    (h₁ : a % 36 = 16)
    (h₂ : b % 36 = 8)
    (h₃ : c % 36 = 24) :
    (a * b * c) % 36 = 12 := 
    by
    sorry

end product_remainder_l1561_156101


namespace find_f_ln2_l1561_156122

noncomputable def f : ℝ → ℝ := sorry

axiom fx_monotonic : Monotone f
axiom fx_condition : ∀ x : ℝ, f (f x + Real.exp x) = 1 - Real.exp 1

theorem find_f_ln2 : f (Real.log 2) = -1 := 
sorry

end find_f_ln2_l1561_156122


namespace school_choir_robe_cost_l1561_156116

theorem school_choir_robe_cost :
  ∀ (total_robes_needed current_robes cost_per_robe : ℕ), 
  total_robes_needed = 30 → 
  current_robes = 12 → 
  cost_per_robe = 2 → 
  (total_robes_needed - current_robes) * cost_per_robe = 36 :=
by
  intros total_robes_needed current_robes cost_per_robe h1 h2 h3
  sorry

end school_choir_robe_cost_l1561_156116


namespace problem_area_triangle_PNT_l1561_156131

noncomputable def area_triangle_PNT (PQ QR x : ℝ) : ℝ :=
  let PS := Real.sqrt (PQ^2 + QR^2)
  let PN := PS / 2
  let area := (PN * Real.sqrt (61 - x^2)) / 4
  area

theorem problem_area_triangle_PNT :
  ∀ (PQ QR : ℝ) (x : ℝ), PQ = 10 → QR = 12 → 0 ≤ x ∧ x ≤ 10 → area_triangle_PNT PQ QR x = 
  (Real.sqrt (244) * Real.sqrt (61 - x^2)) / 4 :=
by
  intros PQ QR x hPQ hQR hx
  sorry

end problem_area_triangle_PNT_l1561_156131


namespace checkered_rectangles_unique_gray_cells_l1561_156165

noncomputable def num_checkered_rectangles (num_gray_cells : ℕ) (num_blue_cells : ℕ) (rects_per_blue_cell : ℕ)
    (num_red_cells : ℕ) (rects_per_red_cell : ℕ) : ℕ :=
    (num_blue_cells * rects_per_blue_cell) + (num_red_cells * rects_per_red_cell)

theorem checkered_rectangles_unique_gray_cells : num_checkered_rectangles 40 36 4 4 8 = 176 := 
sorry

end checkered_rectangles_unique_gray_cells_l1561_156165


namespace geometric_sequence_a5_l1561_156133

theorem geometric_sequence_a5 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_a3 : a 3 = -1)
  (h_a7 : a 7 = -9) : a 5 = -3 := 
sorry

end geometric_sequence_a5_l1561_156133


namespace inequality_holds_if_and_only_if_l1561_156134

noncomputable def absolute_inequality (x a : ℝ) : Prop :=
  |x - 3| + |x - 4| + |x - 5| < a

theorem inequality_holds_if_and_only_if (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, absolute_inequality x a) ↔ a > 4 := 
sorry

end inequality_holds_if_and_only_if_l1561_156134


namespace sum_of_geometric_sequence_l1561_156196

-- Consider a geometric sequence {a_n} with the first term a_1 = 1 and a common ratio of 1/3.
-- Let S_n denote the sum of the first n terms.
-- We need to prove that S_n = (3 - a_n) / 2, given the above conditions.
noncomputable def geometric_sequence_sum (n : ℕ) : ℝ :=
  let a_1 := 1
  let r := (1 : ℝ) / 3
  let a_n := r ^ (n - 1)
  (3 - a_n) / 2

theorem sum_of_geometric_sequence (n : ℕ) : geometric_sequence_sum n = 
  let a_1 := 1
  let r := (1 : ℝ) / 3
  let a_n := r ^ (n - 1)
  (3 - a_n) / 2 := sorry

end sum_of_geometric_sequence_l1561_156196


namespace max_smoothie_servings_l1561_156150

def servings (bananas yogurt strawberries : ℕ) : ℕ :=
  min (bananas * 4 / 3) (min (yogurt * 4 / 2) (strawberries * 4 / 1))

theorem max_smoothie_servings :
  servings 9 10 3 = 12 :=
by
  -- Proof steps would be inserted here
  sorry

end max_smoothie_servings_l1561_156150


namespace sum_geometric_sequence_l1561_156129

theorem sum_geometric_sequence {n : ℕ} (S : ℕ → ℝ) (h1 : S n = 10) (h2 : S (2 * n) = 30) : 
  S (3 * n) = 70 := 
by 
  sorry

end sum_geometric_sequence_l1561_156129


namespace number_of_solutions_l1561_156124

theorem number_of_solutions (h₁ : ∀ x, 50 * x % 100 = 0 → (x % 2 = 0)) 
                            (h₂ : ∀ x, (x % 2 = 0) → (∀ k, 1 ≤ k ∧ k ≤ 49 → (k * x % 100 ≠ 0)))
                            (h₃ : ∀ x, 1 ≤ x ∧ x ≤ 100) : 
  ∃ count, count = 20 := 
by {
  -- Here, we usually would provide a method to count all valid x values meeting the conditions,
  -- but we skip the proof as instructed.
  sorry
}

end number_of_solutions_l1561_156124


namespace blue_balls_prob_l1561_156113

def prob_same_color (n : ℕ) : ℚ :=
  (1 / (n + 5))^2 + (4 / (n + 5))^2 + (n / (n + 5))^2

theorem blue_balls_prob {n : ℕ} (h : prob_same_color n = 1 / 2) : n = 1 ∨ n = 9 :=
by
  sorry

end blue_balls_prob_l1561_156113


namespace value_of_question_l1561_156121

noncomputable def value_of_approx : ℝ := 0.2127541038062284

theorem value_of_question :
  ((0.76^3 - 0.1^3) / (0.76^2) + value_of_approx + 0.1^2) = 0.66 :=
by
  sorry

end value_of_question_l1561_156121


namespace distance_between_circle_centers_l1561_156199

theorem distance_between_circle_centers
  (R r d : ℝ)
  (h1 : R = 7)
  (h2 : r = 4)
  (h3 : d = 5 + 1)
  (h_total_diameter : 5 + 8 + 1 = 14)
  (h_radius_R : R = 14 / 2)
  (h_radius_r : r = 8 / 2) : d = 6 := 
by sorry

end distance_between_circle_centers_l1561_156199


namespace phil_quarters_collection_l1561_156140

theorem phil_quarters_collection
    (initial_quarters : ℕ)
    (doubled_quarters : ℕ)
    (additional_quarters_per_month : ℕ)
    (total_quarters_end_of_second_year : ℕ)
    (quarters_collected_every_third_month : ℕ)
    (total_quarters_end_of_third_year : ℕ)
    (remaining_quarters_after_loss : ℕ)
    (quarters_left : ℕ) :
    initial_quarters = 50 →
    doubled_quarters = 2 * initial_quarters →
    additional_quarters_per_month = 3 →
    total_quarters_end_of_second_year = doubled_quarters + 12 * additional_quarters_per_month →
    total_quarters_end_of_third_year = total_quarters_end_of_second_year + 4 * quarters_collected_every_third_month →
    remaining_quarters_after_loss = (3 / 4 : ℚ) * total_quarters_end_of_third_year → 
    quarters_left = 105 →
    quarters_collected_every_third_month = 1 := 
by
  sorry

end phil_quarters_collection_l1561_156140


namespace pipe_tank_fill_time_l1561_156104

/-- 
Given:
1. Pipe A fills the tank in 2 hours.
2. The leak empties the tank in 4 hours.
Prove: 
The tank is filled in 4 hours when both Pipe A and the leak are working together.
 -/
theorem pipe_tank_fill_time :
  let A := 1 / 2 -- rate at which Pipe A fills the tank (tank per hour)
  let L := 1 / 4 -- rate at which the leak empties the tank (tank per hour)
  let net_rate := A - L -- net rate of filling the tank
  net_rate > 0 → (1 / net_rate) = 4 := 
by
  intros
  sorry

end pipe_tank_fill_time_l1561_156104


namespace equation_of_line_passing_through_and_parallel_l1561_156159

theorem equation_of_line_passing_through_and_parallel :
  ∀ (x y : ℝ), (x = -3 ∧ y = -1) → (∃ (C : ℝ), x - 2 * y + C = 0) → C = 1 :=
by
  intros x y h₁ h₂
  sorry

end equation_of_line_passing_through_and_parallel_l1561_156159


namespace remaining_days_to_finish_l1561_156166

-- Define initial conditions and constants
def initial_play_hours_per_day : ℕ := 4
def initial_days : ℕ := 14
def completion_fraction : ℚ := 0.40
def increased_play_hours_per_day : ℕ := 7

-- Define the calculation for total initial hours played
def total_initial_hours_played : ℕ := initial_play_hours_per_day * initial_days

-- Define the total hours needed to complete the game
def total_hours_to_finish := total_initial_hours_played / completion_fraction

-- Define the remaining hours needed to finish the game
def remaining_hours := total_hours_to_finish - total_initial_hours_played

-- Prove that the remaining days to finish the game is 12
theorem remaining_days_to_finish : (remaining_hours / increased_play_hours_per_day) = 12 := by
  sorry -- Proof steps go here

end remaining_days_to_finish_l1561_156166


namespace gecko_sales_ratio_l1561_156115

theorem gecko_sales_ratio (x : ℕ) (h1 : 86 + x = 258) : 86 / Nat.gcd 172 86 = 1 ∧ 172 / Nat.gcd 172 86 = 2 := by
  sorry

end gecko_sales_ratio_l1561_156115


namespace Chris_age_l1561_156137

theorem Chris_age 
  (a b c : ℝ)
  (h1 : a + b + c = 36)
  (h2 : c - 5 = a)
  (h3 : b + 4 = (3 / 4) * (a + 4)) :
  c = 15.5454545454545 :=
by
  sorry

end Chris_age_l1561_156137


namespace kona_additional_miles_l1561_156181

theorem kona_additional_miles 
  (d_apartment_to_bakery : ℕ := 9) 
  (d_bakery_to_grandmother : ℕ := 24) 
  (d_grandmother_to_apartment : ℕ := 27) : 
  (d_apartment_to_bakery + d_bakery_to_grandmother + d_grandmother_to_apartment) - (2 * d_grandmother_to_apartment) = 6 := 
by 
  sorry

end kona_additional_miles_l1561_156181


namespace least_number_leaving_remainder_4_l1561_156132

theorem least_number_leaving_remainder_4 (x : ℤ) : 
  (x % 6 = 4) ∧ (x % 9 = 4) ∧ (x % 12 = 4) ∧ (x % 18 = 4) → x = 40 :=
by
  sorry

end least_number_leaving_remainder_4_l1561_156132


namespace largest_circle_area_l1561_156172

theorem largest_circle_area (PQ QR PR : ℝ)
  (h_right_triangle: PR^2 = PQ^2 + QR^2)
  (h_circle_areas_sum: π * (PQ/2)^2 + π * (QR/2)^2 + π * (PR/2)^2 = 338 * π) :
  π * (PR/2)^2 = 169 * π :=
by
  sorry

end largest_circle_area_l1561_156172


namespace x_intercept_l1561_156162

theorem x_intercept (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -8) (hy2 : y2 = -6) : 
  ∃ x : ℝ, (y = 0) ∧ (∃ m : ℝ, m = (y2 - y1) / (x2 - x1) ∧ y1 - y = m * (x1 - x)) ∧ x = 4 :=
sorry

end x_intercept_l1561_156162


namespace two_digit_number_digits_34_l1561_156160

theorem two_digit_number_digits_34 :
  let x := (34 / 99.0)
  ∃ n : ℕ, n = 34 ∧ (48 * x - 48 * 0.34 = 0.2) := 
by
  let x := (34.0 / 99.0)
  use 34
  sorry

end two_digit_number_digits_34_l1561_156160


namespace triangle_sine_ratio_l1561_156107

-- Define points A and C
def A : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (1, 0)

-- Define the condition of point B being on the ellipse
def isOnEllipse (B : ℝ × ℝ) : Prop :=
  (B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1

-- Define the sin law ratio we need to prove
noncomputable def sin_ratio (sin_A sin_C sin_B : ℝ) : ℝ := 
  (sin_A + sin_C) / sin_B

-- Prove the required sine ratio condition
theorem triangle_sine_ratio (B : ℝ × ℝ) (sin_A sin_C sin_B : ℝ)
  (hB : isOnEllipse B) (hA : sin_A = 0) (hC : sin_C = 0) (hB_nonzero : sin_B ≠ 0) :
  sin_ratio sin_A sin_C sin_B = 2 :=
by
  -- Skipping proof
  sorry

end triangle_sine_ratio_l1561_156107


namespace area_of_circle_eq_sixteen_pi_l1561_156114

theorem area_of_circle_eq_sixteen_pi :
  ∃ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) ↔ (π * 4^2 = 16 * π) :=
by
  sorry

end area_of_circle_eq_sixteen_pi_l1561_156114


namespace x_fourth_minus_inv_fourth_l1561_156178

theorem x_fourth_minus_inv_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/(x^4) = 727 :=
by
  sorry

end x_fourth_minus_inv_fourth_l1561_156178


namespace amazing_squares_exist_l1561_156108

structure Quadrilateral :=
(A B C D : Point)

def diagonals_not_perpendicular (quad : Quadrilateral) : Prop := sorry -- The precise definition will abstractly represent the non-perpendicularity of diagonals.

def amazing_square (quad : Quadrilateral) (square : Square) : Prop :=
  -- Definition stating that the sides of the square (extended if necessary) pass through distinct vertices of the quadrilateral
  sorry

theorem amazing_squares_exist (quad : Quadrilateral) (h : diagonals_not_perpendicular quad) :
  ∃ squares : Finset Square, squares.card ≥ 6 ∧ ∀ square ∈ squares, amazing_square quad square :=
by sorry

end amazing_squares_exist_l1561_156108


namespace find_chocolate_cakes_l1561_156191

variable (C : ℕ)
variable (h1 : 12 * C + 6 * 22 = 168)

theorem find_chocolate_cakes : C = 3 :=
by
  -- this is the proof placeholder
  sorry

end find_chocolate_cakes_l1561_156191


namespace sum_of_two_numbers_l1561_156190

theorem sum_of_two_numbers (a b : ℕ) (h1 : (a + b) * (a - b) = 1996) (h2 : (a + b) % 2 = (a - b) % 2) (h3 : a + b > a - b) : a + b = 998 := 
sorry

end sum_of_two_numbers_l1561_156190


namespace simplify_expression_l1561_156130

variable (x : ℝ)

theorem simplify_expression :
  (3 * x - 6) * (2 * x + 8) - (x + 6) * (3 * x + 1) = 3 * x^2 - 7 * x - 54 :=
by
  sorry

end simplify_expression_l1561_156130


namespace find_z_l1561_156184

-- Definitions based on the conditions from the problem
def x : ℤ := sorry
def y : ℤ := x - 1
def z : ℤ := x - 2
def condition1 : x > y ∧ y > z := by
  sorry

def condition2 : 2 * x + 3 * y + 3 * z = 5 * y + 11 := by
  sorry

-- Statement to prove
theorem find_z : z = 3 :=
by
  -- Use the conditions to prove the statement
  have h1 : x > y ∧ y > z := condition1
  have h2 : 2 * x + 3 * y + 3 * z = 5 * y + 11 := condition2
  sorry

end find_z_l1561_156184


namespace solve_for_x_l1561_156119

theorem solve_for_x (x : ℝ) : 3^(4 * x) = (81 : ℝ)^(1 / 4) → x = 1 / 4 :=
by
  intros
  sorry

end solve_for_x_l1561_156119


namespace range_of_x_l1561_156177

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x + 1

-- State the theorem to prove the condition
theorem range_of_x (x : ℝ) : f (1 - x) + f (2 * x) > 2 ↔ x > -1 :=
by {
  sorry -- Proof placeholder
}

end range_of_x_l1561_156177
