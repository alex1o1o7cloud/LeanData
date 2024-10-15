import Mathlib

namespace NUMINAMATH_GPT_sum_of_cubes_l1472_147274

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 17) : a^3 + b^3 = 490 := 
sorry

end NUMINAMATH_GPT_sum_of_cubes_l1472_147274


namespace NUMINAMATH_GPT_min_bills_required_l1472_147271

-- Conditions
def ten_dollar_bills := 13
def five_dollar_bills := 11
def one_dollar_bills := 17
def total_amount := 128

-- Prove that Tim can pay exactly $128 with the minimum number of bills being 16
theorem min_bills_required : (∃ ten five one : ℕ, 
    ten ≤ ten_dollar_bills ∧
    five ≤ five_dollar_bills ∧
    one ≤ one_dollar_bills ∧
    ten * 10 + five * 5 + one = total_amount ∧
    ten + five + one = 16) :=
by
  -- We will skip the proof for now
  sorry

end NUMINAMATH_GPT_min_bills_required_l1472_147271


namespace NUMINAMATH_GPT_proposition_q_false_for_a_lt_2_l1472_147227

theorem proposition_q_false_for_a_lt_2 (a : ℝ) (h : a < 2) : 
  ¬ ∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1 :=
sorry

end NUMINAMATH_GPT_proposition_q_false_for_a_lt_2_l1472_147227


namespace NUMINAMATH_GPT_at_least_two_foxes_met_same_number_of_koloboks_l1472_147264

-- Define the conditions
def number_of_foxes : ℕ := 14
def number_of_koloboks : ℕ := 92

-- The theorem statement to be proven
theorem at_least_two_foxes_met_same_number_of_koloboks :
  ∃ (f : Fin number_of_foxes.succ → ℕ), 
    (∀ i, f i ≤ number_of_koloboks) ∧ 
    ∃ i j, i ≠ j ∧ f i = f j :=
by
  sorry

end NUMINAMATH_GPT_at_least_two_foxes_met_same_number_of_koloboks_l1472_147264


namespace NUMINAMATH_GPT_sqrt_sum_eval_l1472_147267

theorem sqrt_sum_eval : 
  (Real.sqrt 50 + Real.sqrt 72) = 11 * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_sum_eval_l1472_147267


namespace NUMINAMATH_GPT_unique_solution_implies_relation_l1472_147242

open Nat

noncomputable def unique_solution (a b : ℤ) :=
  ∃! (x y z : ℤ), x + y = a - 1 ∧ x * (y + 1) - z^2 = b

theorem unique_solution_implies_relation (a b : ℤ) :
  unique_solution a b → b = (a * a) / 4 := sorry

end NUMINAMATH_GPT_unique_solution_implies_relation_l1472_147242


namespace NUMINAMATH_GPT_sequence_a4_value_l1472_147238

theorem sequence_a4_value :
  ∀ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n + 3) → a 4 = 29 :=
by sorry

end NUMINAMATH_GPT_sequence_a4_value_l1472_147238


namespace NUMINAMATH_GPT_hyperbola_condition_l1472_147283

theorem hyperbola_condition (k : ℝ) : 
  (-1 < k ∧ k < 1) ↔ (∃ x y : ℝ, (x^2 / (k-1) + y^2 / (k+1)) = 1) := 
sorry

end NUMINAMATH_GPT_hyperbola_condition_l1472_147283


namespace NUMINAMATH_GPT_inverse_of_k_l1472_147250

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def k (x : ℝ) : ℝ := f (g x)

noncomputable def k_inv (y : ℝ) : ℝ := (y + 11) / 12

theorem inverse_of_k :
  ∀ y : ℝ, k_inv (k y) = y :=
by
  intros x
  simp [k, k_inv, f, g]
  sorry

end NUMINAMATH_GPT_inverse_of_k_l1472_147250


namespace NUMINAMATH_GPT_polynomial_integer_roots_l1472_147245

theorem polynomial_integer_roots :
  ∀ x : ℤ, (x^3 - 3*x^2 - 10*x + 20 = 0) ↔ (x = -2 ∨ x = 5) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_integer_roots_l1472_147245


namespace NUMINAMATH_GPT_cos_neg_pi_div_3_l1472_147219

theorem cos_neg_pi_div_3 : Real.cos (-π / 3) = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_cos_neg_pi_div_3_l1472_147219


namespace NUMINAMATH_GPT_average_student_headcount_proof_l1472_147225

def average_student_headcount : ℕ := (11600 + 11800 + 12000 + 11400) / 4

theorem average_student_headcount_proof :
  average_student_headcount = 11700 :=
by
  -- calculation here
  sorry

end NUMINAMATH_GPT_average_student_headcount_proof_l1472_147225


namespace NUMINAMATH_GPT_purely_imaginary_condition_l1472_147289

-- Define the necessary conditions
def real_part_eq_zero (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 = 0
def imaginary_part_neq_zero (m : ℝ) : Prop := m^2 - 3 * m + 2 ≠ 0

-- State the theorem to be proved
theorem purely_imaginary_condition (m : ℝ) :
  real_part_eq_zero m ∧ imaginary_part_neq_zero m ↔ m = -1/2 :=
sorry

end NUMINAMATH_GPT_purely_imaginary_condition_l1472_147289


namespace NUMINAMATH_GPT_tickets_difference_l1472_147220

-- Define the conditions
def tickets_friday : ℕ := 181
def tickets_sunday : ℕ := 78
def tickets_saturday : ℕ := 2 * tickets_friday

-- The theorem to prove
theorem tickets_difference : tickets_saturday - tickets_sunday = 284 := by
  sorry

end NUMINAMATH_GPT_tickets_difference_l1472_147220


namespace NUMINAMATH_GPT_third_place_amount_l1472_147268

noncomputable def total_people : ℕ := 13
noncomputable def money_per_person : ℝ := 5
noncomputable def total_money : ℝ := total_people * money_per_person

noncomputable def first_place_percentage : ℝ := 0.65
noncomputable def second_third_place_percentage : ℝ := 0.35
noncomputable def split_factor : ℝ := 0.5

noncomputable def first_place_money : ℝ := first_place_percentage * total_money
noncomputable def second_third_place_money : ℝ := second_third_place_percentage * total_money
noncomputable def third_place_money : ℝ := split_factor * second_third_place_money

theorem third_place_amount : third_place_money = 11.38 := by
  sorry

end NUMINAMATH_GPT_third_place_amount_l1472_147268


namespace NUMINAMATH_GPT_floss_per_student_l1472_147259

theorem floss_per_student
  (students : ℕ)
  (yards_per_packet : ℕ)
  (floss_left_over : ℕ)
  (total_packets : ℕ)
  (total_floss : ℕ)
  (total_floss_bought : ℕ)
  (smallest_multiple_of_35 : ℕ)
  (each_student_needs : ℕ)
  (hs1 : students = 20)
  (hs2 : yards_per_packet = 35)
  (hs3 : floss_left_over = 5)
  (hs4 : total_floss = total_packets * yards_per_packet)
  (hs5 : total_floss_bought = total_floss + floss_left_over)
  (hs6 : total_floss_bought % 35 = 0)
  (hs7 : smallest_multiple_of_35 > total_packets * yards_per_packet - floss_left_over)
  (hs8 : 20 * each_student_needs + 5 = smallest_multiple_of_35)
  : each_student_needs = 5 :=
by
  sorry

end NUMINAMATH_GPT_floss_per_student_l1472_147259


namespace NUMINAMATH_GPT_edward_dunk_a_clown_tickets_l1472_147255

-- Definitions for conditions
def total_tickets : ℕ := 79
def rides : ℕ := 8
def tickets_per_ride : ℕ := 7

-- Theorem statement
theorem edward_dunk_a_clown_tickets :
  let tickets_spent_on_rides := rides * tickets_per_ride
  let tickets_remaining := total_tickets - tickets_spent_on_rides
  tickets_remaining = 23 :=
by
  sorry

end NUMINAMATH_GPT_edward_dunk_a_clown_tickets_l1472_147255


namespace NUMINAMATH_GPT_product_scaled_areas_l1472_147213

variable (a b c k V : ℝ)

def volume (a b c : ℝ) : ℝ := a * b * c

theorem product_scaled_areas (a b c k : ℝ) (V : ℝ) (hV : V = volume a b c) :
  (k * a * b) * (k * b * c) * (k * c * a) = k^3 * (V^2) := 
by
  -- Proof steps would go here, but we use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_product_scaled_areas_l1472_147213


namespace NUMINAMATH_GPT_find_mn_l1472_147277

theorem find_mn (m n : ℕ) (h : m > 0 ∧ n > 0) (eq1 : m^2 + n^2 + 4 * m - 46 = 0) :
  mn = 5 ∨ mn = 15 := by
  sorry

end NUMINAMATH_GPT_find_mn_l1472_147277


namespace NUMINAMATH_GPT_ordering_of_abc_l1472_147266

def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

theorem ordering_of_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_ordering_of_abc_l1472_147266


namespace NUMINAMATH_GPT_points_collinear_l1472_147248

theorem points_collinear 
  {a b c : ℝ} (h1 : 0 < b) (h2 : b < a) (h3 : c = Real.sqrt (a^2 - b^2))
  (α β : ℝ)
  (P : ℝ × ℝ) (hP : P = (a^2 / c, 0)) 
  (A : ℝ × ℝ) (hA : A = (a * Real.cos α, b * Real.sin α)) 
  (B : ℝ × ℝ) (hB : B = (a * Real.cos β, b * Real.sin β)) 
  (Q : ℝ × ℝ) (hQ : Q = (a * Real.cos α, -b * Real.sin α)) 
  (F : ℝ × ℝ) (hF : F = (c, 0))
  (line_through_F : (A.1 - F.1) * (B.2 - F.2) = (A.2 - F.2) * (B.1 - F.1)) :
  ∃ (k : ℝ), k * (Q.1 - P.1) = Q.2 - P.2 ∧ k * (B.1 - P.1) = B.2 - P.2 :=
by {
  sorry
}

end NUMINAMATH_GPT_points_collinear_l1472_147248


namespace NUMINAMATH_GPT_sum_of_numbers_greater_than_or_equal_to_0_1_l1472_147244

def num1 : ℝ := 0.8
def num2 : ℝ := 0.5  -- converting 1/2 to 0.5
def num3 : ℝ := 0.6

def is_greater_than_or_equal_to_0_1 (n : ℝ) : Prop :=
  n ≥ 0.1

theorem sum_of_numbers_greater_than_or_equal_to_0_1 :
  is_greater_than_or_equal_to_0_1 num1 ∧ 
  is_greater_than_or_equal_to_0_1 num2 ∧ 
  is_greater_than_or_equal_to_0_1 num3 →
  num1 + num2 + num3 = 1.9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_greater_than_or_equal_to_0_1_l1472_147244


namespace NUMINAMATH_GPT_complement_of_M_l1472_147254

open Set

def M : Set ℝ := { x | (2 - x) / (x + 3) < 0 }

theorem complement_of_M : (Mᶜ = { x : ℝ | -3 ≤ x ∧ x ≤ 2 }) :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_l1472_147254


namespace NUMINAMATH_GPT_no_integer_roots_of_polynomial_l1472_147297

theorem no_integer_roots_of_polynomial :
  ¬ ∃ x : ℤ, x^3 - 4 * x^2 - 14 * x + 28 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_roots_of_polynomial_l1472_147297


namespace NUMINAMATH_GPT_sum_of_fractions_l1472_147246

theorem sum_of_fractions :
  (1 / (2 * 3 * 4) + 1 / (3 * 4 * 5) + 1 / (4 * 5 * 6) + 1 / (5 * 6 * 7) + 1 / (6 * 7 * 8)) = 3 / 16 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l1472_147246


namespace NUMINAMATH_GPT_unique_prime_satisfying_condition_l1472_147240

theorem unique_prime_satisfying_condition :
  ∃! p : ℕ, Prime p ∧ (∀ q : ℕ, Prime q ∧ q < p → ∀ k r : ℕ, p = k * q + r ∧ 0 ≤ r ∧ r < q → ∀ a : ℕ, a > 1 → ¬ a^2 ∣ r) ∧ p = 13 :=
sorry

end NUMINAMATH_GPT_unique_prime_satisfying_condition_l1472_147240


namespace NUMINAMATH_GPT_problem_statement_l1472_147251

noncomputable def a : ℝ := (Real.tan 23) / (1 - (Real.tan 23) ^ 2)
noncomputable def b : ℝ := 2 * Real.sin 13 * Real.cos 13
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos 50) / 2)

theorem problem_statement : c < b ∧ b < a :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_problem_statement_l1472_147251


namespace NUMINAMATH_GPT_drawing_specific_cards_from_two_decks_l1472_147223

def prob_of_drawing_specific_cards (total_cards_deck1 total_cards_deck2 : ℕ) 
  (specific_card1 specific_card2 : ℕ) : ℚ :=
(specific_card1 / total_cards_deck1) * (specific_card2 / total_cards_deck2)

theorem drawing_specific_cards_from_two_decks :
  prob_of_drawing_specific_cards 52 52 1 1 = 1 / 2704 :=
by
  -- The proof can be filled in here
  sorry

end NUMINAMATH_GPT_drawing_specific_cards_from_two_decks_l1472_147223


namespace NUMINAMATH_GPT_train_crosses_bridge_in_30_seconds_l1472_147256

/--
A train 155 metres long, travelling at 45 km/hr, can cross a bridge with length 220 metres in 30 seconds.
-/
theorem train_crosses_bridge_in_30_seconds
  (length_train : ℕ)
  (length_bridge : ℕ)
  (speed_km_per_hr : ℕ)
  (total_distance : ℕ)
  (speed_m_per_s : ℚ)
  (time_seconds : ℚ) 
  (h1 : length_train = 155)
  (h2 : length_bridge = 220)
  (h3 : speed_km_per_hr = 45)
  (h4 : total_distance = length_train + length_bridge)
  (h5 : speed_m_per_s = (speed_km_per_hr * 1000) / 3600)
  (h6 : time_seconds = total_distance / speed_m_per_s) :
  time_seconds = 30 :=
sorry

end NUMINAMATH_GPT_train_crosses_bridge_in_30_seconds_l1472_147256


namespace NUMINAMATH_GPT_lucy_money_left_l1472_147209

theorem lucy_money_left : 
  ∀ (initial_money : ℕ) 
    (one_third_loss : ℕ → ℕ) 
    (one_fourth_spend : ℕ → ℕ), 
    initial_money = 30 → 
    one_third_loss initial_money = initial_money / 3 → 
    one_fourth_spend (initial_money - one_third_loss initial_money) = (initial_money - one_third_loss initial_money) / 4 → 
  initial_money - one_third_loss initial_money - one_fourth_spend (initial_money - one_third_loss initial_money) = 15 :=
by
  intros initial_money one_third_loss one_fourth_spend
  intro h_initial_money
  intro h_one_third_loss
  intro h_one_fourth_spend
  sorry

end NUMINAMATH_GPT_lucy_money_left_l1472_147209


namespace NUMINAMATH_GPT_math_problem_l1472_147216

noncomputable def a : ℝ := 0.137
noncomputable def b : ℝ := 0.098
noncomputable def c : ℝ := 0.123
noncomputable def d : ℝ := 0.086

theorem math_problem : 
  ( ((a + b)^2 - (a - b)^2) / (c * d) + (d^3 - c^3) / (a * b * (a + b)) ) = 4.6886 := 
  sorry

end NUMINAMATH_GPT_math_problem_l1472_147216


namespace NUMINAMATH_GPT_MargaretsMeanScore_l1472_147218

theorem MargaretsMeanScore :
  ∀ (scores : List ℕ)
    (cyprian_mean : ℝ)
    (highest_lowest_different : Prop),
    scores = [82, 85, 88, 90, 92, 95, 97, 99] →
    cyprian_mean = 88.5 →
    highest_lowest_different →
    ∃ (margaret_mean : ℝ), margaret_mean = 93.5 := by
  sorry

end NUMINAMATH_GPT_MargaretsMeanScore_l1472_147218


namespace NUMINAMATH_GPT_remainder_of_B_is_4_l1472_147201

theorem remainder_of_B_is_4 (A B : ℕ) (h : B = A * 9 + 13) : B % 9 = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_of_B_is_4_l1472_147201


namespace NUMINAMATH_GPT_find_geometric_arithmetic_progressions_l1472_147282

theorem find_geometric_arithmetic_progressions
    (b1 b2 b3 : ℚ)
    (h1 : b2^2 = b1 * b3)
    (h2 : b2 + 2 = (b1 + b3) / 2)
    (h3 : (b2 + 2)^2 = b1 * (b3 + 16)) :
    (b1 = 1 ∧ b2 = 3 ∧ b3 = 9) ∨ (b1 = 1/9 ∧ b2 = -5/9 ∧ b3 = 25/9) :=
  sorry

end NUMINAMATH_GPT_find_geometric_arithmetic_progressions_l1472_147282


namespace NUMINAMATH_GPT_walt_total_interest_l1472_147294

noncomputable def total_investment : ℝ := 12000
noncomputable def investment_at_7_percent : ℝ := 5500
noncomputable def investment_at_9_percent : ℝ := total_investment - investment_at_7_percent
noncomputable def rate_7_percent : ℝ := 0.07
noncomputable def rate_9_percent : ℝ := 0.09

theorem walt_total_interest :
  let interest_7 : ℝ := investment_at_7_percent * rate_7_percent
  let interest_9 : ℝ := investment_at_9_percent * rate_9_percent
  interest_7 + interest_9 = 970 := by
  sorry

end NUMINAMATH_GPT_walt_total_interest_l1472_147294


namespace NUMINAMATH_GPT_prove_expression_l1472_147233

noncomputable def omega := Complex.exp (2 * Real.pi * Complex.I / 5)

lemma root_of_unity : omega^5 = 1 := sorry
lemma sum_of_roots : omega^0 + omega + omega^2 + omega^3 + omega^4 = 0 := sorry

noncomputable def z := omega + omega^2 + omega^3 + omega^4

theorem prove_expression : z^2 + z + 1 = 1 :=
by 
  have h1 : omega^5 = 1 := root_of_unity
  have h2 : omega^0 + omega + omega^2 + omega^3 + omega^4 = 0 := sum_of_roots
  show z^2 + z + 1 = 1
  {
    -- Proof omitted
    sorry
  }

end NUMINAMATH_GPT_prove_expression_l1472_147233


namespace NUMINAMATH_GPT_not_perfect_square_n_l1472_147224

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ m : ℕ, m * m = x

theorem not_perfect_square_n (n : ℕ) : ¬ isPerfectSquare (4 * n^2 + 4 * n + 4) :=
sorry

end NUMINAMATH_GPT_not_perfect_square_n_l1472_147224


namespace NUMINAMATH_GPT_cover_black_squares_with_L_shape_l1472_147276

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the main theorem
theorem cover_black_squares_with_L_shape (n : ℕ) (h_odd : is_odd n) (h_corner_black : ∀i j, (i = 0 ∨ i = n - 1) ∧ (j = 0 ∨ j = n - 1) → (i + j) % 2 = 1) : n ≥ 7 :=
sorry

end NUMINAMATH_GPT_cover_black_squares_with_L_shape_l1472_147276


namespace NUMINAMATH_GPT_correct_statements_l1472_147214

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 4)

theorem correct_statements : 
  (∀ x, f (-x) = -f (x)) ∧  -- Statement A
  (∀ x₁ x₂, x₁ + x₂ = Real.pi / 2 → g x₁ = g x₂)  -- Statement C
:= by
  sorry

end NUMINAMATH_GPT_correct_statements_l1472_147214


namespace NUMINAMATH_GPT_number_of_small_pipes_needed_l1472_147247

theorem number_of_small_pipes_needed :
  let diameter_large := 8
  let diameter_small := 1
  let radius_large := diameter_large / 2
  let radius_small := diameter_small / 2
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let num_small_pipes := area_large / area_small
  num_small_pipes = 64 :=
by
  sorry

end NUMINAMATH_GPT_number_of_small_pipes_needed_l1472_147247


namespace NUMINAMATH_GPT_minimal_board_size_for_dominoes_l1472_147262

def board_size_is_minimal (n: ℕ) (total_area: ℕ) (domino_size: ℕ) (num_dominoes: ℕ) : Prop :=
  ∀ m: ℕ, m < n → ¬ (total_area ≥ m * m ∧ m * m = num_dominoes * domino_size)

theorem minimal_board_size_for_dominoes (n: ℕ) :
  board_size_is_minimal 77 2008 2 1004 :=
by
  sorry

end NUMINAMATH_GPT_minimal_board_size_for_dominoes_l1472_147262


namespace NUMINAMATH_GPT_sqrt_five_squared_minus_four_squared_eq_three_l1472_147237

theorem sqrt_five_squared_minus_four_squared_eq_three : Real.sqrt (5 ^ 2 - 4 ^ 2) = 3 := by
  sorry

end NUMINAMATH_GPT_sqrt_five_squared_minus_four_squared_eq_three_l1472_147237


namespace NUMINAMATH_GPT_positive_integers_satisfy_eq_l1472_147263

theorem positive_integers_satisfy_eq (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + 1 = c! → (a = 2 ∧ b = 1 ∧ c = 3) ∨ (a = 1 ∧ b = 2 ∧ c = 3) :=
by sorry

end NUMINAMATH_GPT_positive_integers_satisfy_eq_l1472_147263


namespace NUMINAMATH_GPT_tile_difference_is_42_l1472_147212

def original_blue_tiles : ℕ := 14
def original_green_tiles : ℕ := 8
def green_tiles_first_border : ℕ := 18
def green_tiles_second_border : ℕ := 30

theorem tile_difference_is_42 :
  (original_green_tiles + green_tiles_first_border + green_tiles_second_border) - original_blue_tiles = 42 :=
by
  sorry

end NUMINAMATH_GPT_tile_difference_is_42_l1472_147212


namespace NUMINAMATH_GPT_exists_k_square_congruent_neg_one_iff_l1472_147211

theorem exists_k_square_congruent_neg_one_iff (p : ℕ) [Fact p.Prime] :
  (∃ k : ℤ, (k^2 ≡ -1 [ZMOD p])) ↔ (p = 2 ∨ p % 4 = 1) :=
sorry

end NUMINAMATH_GPT_exists_k_square_congruent_neg_one_iff_l1472_147211


namespace NUMINAMATH_GPT_P_inequality_l1472_147249

variable {α : Type*} [LinearOrderedField α]

def P (a b c : α) (x : α) : α := a * x^2 + b * x + c

theorem P_inequality (a b c x y : α) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (P a b c (x * y))^2 ≤ (P a b c (x^2)) * (P a b c (y^2)) :=
sorry

end NUMINAMATH_GPT_P_inequality_l1472_147249


namespace NUMINAMATH_GPT_maximum_value_is_one_div_sqrt_two_l1472_147288

noncomputable def maximum_value_2ab_root2_plus_2ac_plus_2bc (a b c : ℝ) : ℝ :=
  2 * a * b * Real.sqrt 2 + 2 * a * c + 2 * b * c

theorem maximum_value_is_one_div_sqrt_two (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h : a^2 + b^2 + c^2 = 1) :
  maximum_value_2ab_root2_plus_2ac_plus_2bc a b c ≤ 1 / Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_is_one_div_sqrt_two_l1472_147288


namespace NUMINAMATH_GPT_new_computer_price_l1472_147221

-- Define the initial conditions
def initial_price_condition (x : ℝ) : Prop := 2 * x = 540

-- Define the calculation for the new price after a 30% increase
def new_price (x : ℝ) : ℝ := x * 1.30

-- Define the final proof problem statement
theorem new_computer_price : ∃ x : ℝ, initial_price_condition x ∧ new_price x = 351 :=
by sorry

end NUMINAMATH_GPT_new_computer_price_l1472_147221


namespace NUMINAMATH_GPT_slope_of_line_l1472_147273

variable (s : ℝ) -- real number s

def line1 (x y : ℝ) := x + 3 * y = 9 * s + 4
def line2 (x y : ℝ) := x - 2 * y = 3 * s - 3

theorem slope_of_line (s : ℝ) :
  ∀ (x y : ℝ), (line1 s x y ∧ line2 s x y) → y = (2 / 9) * x + (13 / 9) :=
sorry

end NUMINAMATH_GPT_slope_of_line_l1472_147273


namespace NUMINAMATH_GPT_p_necessary_not_sufficient_q_l1472_147290

def condition_p (x : ℝ) : Prop := abs x ≤ 2
def condition_q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

theorem p_necessary_not_sufficient_q (x : ℝ) :
  (condition_p x → condition_q x) = false ∧ (condition_q x → condition_p x) = true :=
by
  sorry

end NUMINAMATH_GPT_p_necessary_not_sufficient_q_l1472_147290


namespace NUMINAMATH_GPT_smallest_positive_integer_l1472_147292

-- Given integers m and n, prove the smallest positive integer of the form 2017m + 48576n
theorem smallest_positive_integer (m n : ℤ) : 
  ∃ m n : ℤ, 2017 * m + 48576 * n = 1 := by
sorry

end NUMINAMATH_GPT_smallest_positive_integer_l1472_147292


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1472_147230

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem necessary_but_not_sufficient_condition (f : ℝ → ℝ) :
  (f 0 = 0) ↔ is_odd_function f := 
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1472_147230


namespace NUMINAMATH_GPT_alexa_weight_proof_l1472_147293

variable (totalWeight katerinaWeight alexaWeight : ℕ)

def weight_relation (totalWeight katerinaWeight alexaWeight : ℕ) : Prop :=
  totalWeight = katerinaWeight + alexaWeight

theorem alexa_weight_proof (h1 : totalWeight = 95) (h2 : katerinaWeight = 49) : alexaWeight = 46 :=
by
  have h : alexaWeight = totalWeight - katerinaWeight := by
    sorry
  rw [h1, h2] at h
  exact h

end NUMINAMATH_GPT_alexa_weight_proof_l1472_147293


namespace NUMINAMATH_GPT_profit_percentage_with_discount_is_26_l1472_147203

noncomputable def cost_price : ℝ := 100
noncomputable def profit_percentage_without_discount : ℝ := 31.25
noncomputable def discount_percentage : ℝ := 4

noncomputable def selling_price_without_discount : ℝ :=
  cost_price * (1 + profit_percentage_without_discount / 100)

noncomputable def discount : ℝ := 
  discount_percentage / 100 * selling_price_without_discount

noncomputable def selling_price_with_discount : ℝ :=
  selling_price_without_discount - discount

noncomputable def profit_with_discount : ℝ := 
  selling_price_with_discount - cost_price

noncomputable def profit_percentage_with_discount : ℝ := 
  (profit_with_discount / cost_price) * 100

theorem profit_percentage_with_discount_is_26 :
  profit_percentage_with_discount = 26 := by 
  sorry

end NUMINAMATH_GPT_profit_percentage_with_discount_is_26_l1472_147203


namespace NUMINAMATH_GPT_tangent_normal_at_t1_l1472_147232

noncomputable def curve_param_x (t: ℝ) : ℝ := Real.arcsin (t / Real.sqrt (1 + t^2))
noncomputable def curve_param_y (t: ℝ) : ℝ := Real.arccos (1 / Real.sqrt (1 + t^2))

theorem tangent_normal_at_t1 : 
  curve_param_x 1 = Real.pi / 4 ∧
  curve_param_y 1 = Real.pi / 4 ∧
  ∃ (x y : ℝ), (y = 2*x - Real.pi/4) ∧ (y = -x/2 + 3*Real.pi/8) :=
  sorry

end NUMINAMATH_GPT_tangent_normal_at_t1_l1472_147232


namespace NUMINAMATH_GPT_inequality_proof_equality_condition_l1472_147215

variables {a b c x y z : ℕ}

theorem inequality_proof (h1 : a ^ 2 + b ^ 2 = c ^ 2) (h2 : x ^ 2 + y ^ 2 = z ^ 2) : 
  (a + x) ^ 2 + (b + y) ^ 2 ≤ (c + z) ^ 2 :=
sorry

theorem equality_condition (h1 : a ^ 2 + b ^ 2 = c ^ 2) (h2 : x ^ 2 + y ^ 2 = z ^ 2) : 
  (a + x) ^ 2 + (b + y) ^ 2 = (c + z) ^ 2 ↔ a * z = c * x ∧ a * y = b * x :=
sorry

end NUMINAMATH_GPT_inequality_proof_equality_condition_l1472_147215


namespace NUMINAMATH_GPT_problem_solution_l1472_147275

noncomputable def length_segment_AB : ℝ :=
  let k : ℝ := 1 -- derived from 3k - 3 = 0
  let A : ℝ × ℝ := (0, k) -- point (0, k)
  let C : ℝ × ℝ := (3, -1) -- center of the circle
  let r : ℝ := 1 -- radius of the circle
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) -- distance formula
  Real.sqrt (AC^2 - r^2)

theorem problem_solution :
  length_segment_AB = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1472_147275


namespace NUMINAMATH_GPT_spherical_circle_radius_l1472_147210

theorem spherical_circle_radius:
  (∀ (θ : Real), ∃ (r : Real), r = 1 * Real.sin (Real.pi / 6)) → ∀ (θ : Real), r = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_spherical_circle_radius_l1472_147210


namespace NUMINAMATH_GPT_number_of_kids_per_day_l1472_147272

theorem number_of_kids_per_day (K : ℕ) 
    (kids_charge : ℕ := 3) 
    (adults_charge : ℕ := kids_charge * 2) 
    (daily_earnings_from_adults : ℕ := 10 * adults_charge) 
    (weekly_earnings : ℕ := 588) 
    (daily_earnings : ℕ := weekly_earnings / 7) :
    (daily_earnings - daily_earnings_from_adults) / kids_charge = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_kids_per_day_l1472_147272


namespace NUMINAMATH_GPT_largest_divisor_of_n_l1472_147278

theorem largest_divisor_of_n (n : ℕ) (h_pos: n > 0) (h_div: 72 ∣ n^2) : 12 ∣ n :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_n_l1472_147278


namespace NUMINAMATH_GPT_fraction_sum_5625_l1472_147229

theorem fraction_sum_5625 : 
  ∃ (a b : ℕ), 0.5625 = (9 : ℚ) / 16 ∧ (a + b = 25) := 
by 
  sorry

end NUMINAMATH_GPT_fraction_sum_5625_l1472_147229


namespace NUMINAMATH_GPT_lowest_number_of_students_l1472_147291

theorem lowest_number_of_students (n : ℕ) (h1 : n % 18 = 0) (h2 : n % 24 = 0) : n = 72 := by
  sorry

end NUMINAMATH_GPT_lowest_number_of_students_l1472_147291


namespace NUMINAMATH_GPT_min_value_expression_l1472_147284

theorem min_value_expression (x : ℝ) (hx : x > 4) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 14) ∧ (∀ z : ℝ, (z = (x + 10) / Real.sqrt (x - 4)) → y ≤ z) := sorry

end NUMINAMATH_GPT_min_value_expression_l1472_147284


namespace NUMINAMATH_GPT_range_of_m_l1472_147235

theorem range_of_m {x1 x2 y1 y2 m : ℝ} 
  (h1 : x1 > x2) 
  (h2 : y1 > y2) 
  (ha : y1 = (m - 3) * x1 - 4) 
  (hb : y2 = (m - 3) * x2 - 4) : 
  m > 3 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1472_147235


namespace NUMINAMATH_GPT_candy_from_sister_l1472_147217

variable (total_neighbors : Nat) (pieces_per_day : Nat) (days : Nat) (total_pieces : Nat)
variable (pieces_per_day_eq : pieces_per_day = 9)
variable (days_eq : days = 9)
variable (total_neighbors_eq : total_neighbors = 66)
variable (total_pieces_eq : total_pieces = 81)

theorem candy_from_sister : 
  total_pieces = total_neighbors + 15 :=
by
  sorry

end NUMINAMATH_GPT_candy_from_sister_l1472_147217


namespace NUMINAMATH_GPT_cricket_match_count_l1472_147228

theorem cricket_match_count (x : ℕ) (h_avg_1 : ℕ → ℕ) (h_avg_2 : ℕ) (h_avg_all : ℕ) (h_eq : 50 * x + 26 * 15 = 42 * (x + 15)) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_cricket_match_count_l1472_147228


namespace NUMINAMATH_GPT_percentage_of_sikhs_l1472_147285

theorem percentage_of_sikhs
  (total_boys : ℕ := 400)
  (percent_muslims : ℕ := 44)
  (percent_hindus : ℕ := 28)
  (other_boys : ℕ := 72) :
  ((total_boys - (percent_muslims * total_boys / 100 + percent_hindus * total_boys / 100 + other_boys)) * 100 / total_boys) = 10 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_percentage_of_sikhs_l1472_147285


namespace NUMINAMATH_GPT_simple_interest_years_l1472_147207

theorem simple_interest_years (P : ℝ) (R : ℝ) (N : ℝ) (higher_interest_amount : ℝ) (additional_rate : ℝ) (initial_sum : ℝ) :
  (initial_sum * (R + additional_rate) * N) / 100 - (initial_sum * R * N) / 100 = higher_interest_amount →
  initial_sum = 3000 →
  higher_interest_amount = 1350 →
  additional_rate = 5 →
  N = 9 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_years_l1472_147207


namespace NUMINAMATH_GPT_treasures_coins_count_l1472_147253

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end NUMINAMATH_GPT_treasures_coins_count_l1472_147253


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1472_147286

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = (Real.sqrt 2) + 1) : 
  (1 - (1 / a)) / ((a ^ 2 - 2 * a + 1) / a) = (Real.sqrt 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1472_147286


namespace NUMINAMATH_GPT_max_g_f_inequality_l1472_147296

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := f x - x / 4 - 1

theorem max_g : ∃ x : ℝ, g x = 2 * Real.log 2 - 7 / 4 :=
sorry

theorem f_inequality (x : ℝ) (hx : 0 < x) : f x < (Real.exp x - 1) / x^2 :=
sorry

end NUMINAMATH_GPT_max_g_f_inequality_l1472_147296


namespace NUMINAMATH_GPT_solution_set_inequality_l1472_147298

def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2

theorem solution_set_inequality (a x : ℝ) (h : Set.Ioo (-1 : ℝ) (2 : ℝ) = {x | |f a x| < 6}) : 
    {x | f a x ≤ 1} = {x | x ≥ 1 / 4} :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1472_147298


namespace NUMINAMATH_GPT_Farrah_total_match_sticks_l1472_147243

def boxes := 4
def matchboxes_per_box := 20
def sticks_per_matchbox := 300

def total_matchboxes : Nat :=
  boxes * matchboxes_per_box

def total_match_sticks : Nat :=
  total_matchboxes * sticks_per_matchbox

theorem Farrah_total_match_sticks : total_match_sticks = 24000 := sorry

end NUMINAMATH_GPT_Farrah_total_match_sticks_l1472_147243


namespace NUMINAMATH_GPT_optimal_strategy_for_father_l1472_147231

-- Define the individual players
inductive player
| Father 
| Mother 
| Son

open player

-- Define the probabilities of player defeating another
def prob_defeat (p1 p2 : player) : ℝ := sorry  -- These will be defined as per the problem's conditions.

-- Define the probability of father winning given the first matchups
def P_father_vs_mother : ℝ :=
  prob_defeat Father Mother * prob_defeat Father Son +
  prob_defeat Father Mother * prob_defeat Son Father * prob_defeat Mother Son * prob_defeat Father Mother +
  prob_defeat Mother Father * prob_defeat Son Mother * prob_defeat Father Son * prob_defeat Father Mother

def P_father_vs_son : ℝ :=
  prob_defeat Father Son * prob_defeat Father Mother +
  prob_defeat Father Son * prob_defeat Mother Father * prob_defeat Son Mother * prob_defeat Father Son +
  prob_defeat Son Father * prob_defeat Mother Son * prob_defeat Father Mother * prob_defeat Father Son

-- Define the optimality condition
theorem optimal_strategy_for_father :
  P_father_vs_mother > P_father_vs_son :=
sorry

end NUMINAMATH_GPT_optimal_strategy_for_father_l1472_147231


namespace NUMINAMATH_GPT_angle_C_measure_l1472_147204

theorem angle_C_measure 
  (p q : Prop) 
  (h1 : p) (h2 : q) 
  (A B C : ℝ) 
  (h_parallel : p = q) 
  (h_A_B : A = B / 10) 
  (h_straight_line : B + C = 180) 
  : C = 16.36 := 
sorry

end NUMINAMATH_GPT_angle_C_measure_l1472_147204


namespace NUMINAMATH_GPT_find_k_l1472_147265

-- Define the vector structures for i and j
def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

-- Define the vectors a and b based on i, j, and k
def a : ℝ × ℝ := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : ℝ × ℝ := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Statement of the theorem
theorem find_k (k : ℝ) : dot_product a (b k) = 0 → k = 6 :=
by sorry

end NUMINAMATH_GPT_find_k_l1472_147265


namespace NUMINAMATH_GPT_petes_original_number_l1472_147279

theorem petes_original_number (x : ℤ) (h : 4 * (2 * x + 20) = 200) : x = 15 :=
sorry

end NUMINAMATH_GPT_petes_original_number_l1472_147279


namespace NUMINAMATH_GPT_calculate_area_of_pentagon_l1472_147260

noncomputable def area_of_pentagon (a b c d e : ℕ) : ℝ :=
  let triangle_area := (1/2 : ℝ) * b * a
  let trapezoid_area := (1/2 : ℝ) * (c + e) * d
  triangle_area + trapezoid_area

theorem calculate_area_of_pentagon : area_of_pentagon 18 25 28 30 25 = 1020 :=
sorry

end NUMINAMATH_GPT_calculate_area_of_pentagon_l1472_147260


namespace NUMINAMATH_GPT_equation_has_three_real_roots_l1472_147281

noncomputable def f (x : ℝ) : ℝ := 2^x - x^2 - 1

theorem equation_has_three_real_roots : ∃! (x : ℝ), f x = 0 :=
by sorry

end NUMINAMATH_GPT_equation_has_three_real_roots_l1472_147281


namespace NUMINAMATH_GPT_circle_origin_range_l1472_147287

theorem circle_origin_range (m : ℝ) : 
  (0 - m)^2 + (0 + m)^2 < 4 → -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_circle_origin_range_l1472_147287


namespace NUMINAMATH_GPT_factorize_square_diff_factorize_common_factor_l1472_147295

-- Problem 1: Difference of squares
theorem factorize_square_diff (x : ℝ) : 4 * x^2 - 9 = (2 * x + 3) * (2 * x - 3) := 
by
  sorry

-- Problem 2: Factoring out common terms
theorem factorize_common_factor (a b x y : ℝ) (h : y - x = -(x - y)) : 
  2 * a * (x - y) - 3 * b * (y - x) = (x - y) * (2 * a + 3 * b) := 
by
  sorry

end NUMINAMATH_GPT_factorize_square_diff_factorize_common_factor_l1472_147295


namespace NUMINAMATH_GPT_abs_diff_of_two_numbers_l1472_147257

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 504) : |x - y| = 3 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_of_two_numbers_l1472_147257


namespace NUMINAMATH_GPT_exist_two_pies_differing_in_both_l1472_147206

-- Define enumeration types for fillings and preparation methods
inductive Filling
| apple
| cherry

inductive Preparation
| fried
| baked

-- Define pie type with filling and preparation
structure Pie where
  filling : Filling
  preparation : Preparation

-- Define the types of pies available
def pie1 : Pie := { filling := Filling.apple, preparation := Preparation.fried }
def pie2 : Pie := { filling := Filling.cherry, preparation := Preparation.fried }
def pie3 : Pie := { filling := Filling.apple, preparation := Preparation.baked }
def pie4 : Pie := { filling := Filling.cherry, preparation := Preparation.baked }

-- Define the list of available pies
def availablePies : List Pie := [pie1, pie2, pie3, pie4]

theorem exist_two_pies_differing_in_both (pies : List Pie) (h : pies.length ≥ 3) :
  ∃ p1 p2 : Pie, p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
by
  -- Proof content to be filled in
  sorry

end NUMINAMATH_GPT_exist_two_pies_differing_in_both_l1472_147206


namespace NUMINAMATH_GPT_correct_statement_A_l1472_147252

-- Definitions for conditions
def general_dilution_range : Set ℕ := {10^3, 10^4, 10^5, 10^6, 10^7}

def actinomycetes_dilution_range : Set ℕ := {10^3, 10^4, 10^5}

def fungi_dilution_range : Set ℕ := {10^2, 10^3, 10^4}

def first_experiment_dilution_range : Set ℕ := {10^3, 10^4, 10^5, 10^6, 10^7}

-- Statement to prove
theorem correct_statement_A : 
  (general_dilution_range = {10^3, 10^4, 10^5, 10^6, 10^7}) :=
sorry

end NUMINAMATH_GPT_correct_statement_A_l1472_147252


namespace NUMINAMATH_GPT_expected_sides_of_red_polygon_l1472_147202

-- Define the conditions
def isChosenWithinSquare (F : ℝ × ℝ) (side_length: ℝ) : Prop :=
  0 ≤ F.1 ∧ F.1 ≤ side_length ∧ 0 ≤ F.2 ∧ F.2 ≤ side_length

def pointF (side_length: ℝ) : ℝ × ℝ := sorry
def foldToF (vertex: ℝ × ℝ) (F: ℝ × ℝ) : ℝ := sorry

-- Define the expected number of sides of the resulting red polygon
noncomputable def expected_sides (side_length : ℝ) : ℝ :=
  let P_g := 2 - (Real.pi / 2)
  let P_o := (Real.pi / 2) - 1 
  (3 * P_o) + (4 * P_g)

-- Prove the expected number of sides equals 5 - π / 2
theorem expected_sides_of_red_polygon (side_length : ℝ) :
  expected_sides side_length = 5 - (Real.pi / 2) := 
  by sorry

end NUMINAMATH_GPT_expected_sides_of_red_polygon_l1472_147202


namespace NUMINAMATH_GPT_project_completion_rate_l1472_147261

variables {a b c d e : ℕ} {f g : ℚ}  -- Assuming efficiency ratings can be represented by rational numbers.

theorem project_completion_rate (h : (a * f / c) = b / c) 
: (d * g / e) = bdge / ca := 
sorry

end NUMINAMATH_GPT_project_completion_rate_l1472_147261


namespace NUMINAMATH_GPT_determine_a_l1472_147226

theorem determine_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x + 1 > 0) ↔ a = 2 := by
  sorry

end NUMINAMATH_GPT_determine_a_l1472_147226


namespace NUMINAMATH_GPT_faster_train_speed_l1472_147205

theorem faster_train_speed (length_train : ℝ) (time_cross : ℝ) (speed_ratio : ℝ) (total_distance : ℝ) (relative_speed : ℝ) :
  length_train = 100 → 
  time_cross = 8 → 
  speed_ratio = 2 → 
  total_distance = 2 * length_train → 
  relative_speed = (1 + speed_ratio) * (total_distance / time_cross) → 
  (1 + speed_ratio) * (total_distance / time_cross) / 3 * 2 = 8.33 := 
by
  intros
  sorry

end NUMINAMATH_GPT_faster_train_speed_l1472_147205


namespace NUMINAMATH_GPT_nathan_ate_100_gumballs_l1472_147239

/-- Define the number of gumballs per package. -/
def gumballs_per_package : ℝ := 5.0

/-- Define the number of packages Nathan ate. -/
def number_of_packages : ℝ := 20.0

/-- Define the total number of gumballs Nathan ate. -/
def total_gumballs : ℝ := number_of_packages * gumballs_per_package

/-- Prove that Nathan ate 100.0 gumballs. -/
theorem nathan_ate_100_gumballs : total_gumballs = 100.0 :=
sorry

end NUMINAMATH_GPT_nathan_ate_100_gumballs_l1472_147239


namespace NUMINAMATH_GPT_cos_identity_l1472_147236

theorem cos_identity (α : ℝ) (h : Real.cos (Real.pi / 8 - α) = 1 / 6) :
  Real.cos (3 * Real.pi / 4 + 2 * α) = 17 / 18 :=
by
  sorry

end NUMINAMATH_GPT_cos_identity_l1472_147236


namespace NUMINAMATH_GPT_find_x_intervals_l1472_147258

theorem find_x_intervals :
  {x : ℝ | x^3 - x^2 + 11*x - 42 < 0} = { x | -2 < x ∧ x < 3 ∨ 3 < x ∧ x < 7 } :=
by sorry

end NUMINAMATH_GPT_find_x_intervals_l1472_147258


namespace NUMINAMATH_GPT_solution_set_f_x_leq_m_solution_set_inequality_a_2_l1472_147299

-- Part (I)
theorem solution_set_f_x_leq_m (a m : ℝ) (h : ∀ x : ℝ, |x - a| ≤ m ↔ -1 ≤ x ∧ x ≤ 5) :
  a = 2 ∧ m = 3 :=
sorry

-- Part (II)
theorem solution_set_inequality_a_2 (t : ℝ) (h_t : t ≥ 0) :
  (∀ x : ℝ, |x - 2| + t ≥ |x + 2 * t - 2| ↔ t = 0 ∧ (∀ x : ℝ, True) ∨ t > 0 ∧ ∀ x : ℝ, x ≤ 2 - t / 2) :=
sorry

end NUMINAMATH_GPT_solution_set_f_x_leq_m_solution_set_inequality_a_2_l1472_147299


namespace NUMINAMATH_GPT_steven_has_72_shirts_l1472_147208

def brian_shirts : ℕ := 3
def andrew_shirts (brian : ℕ) : ℕ := 6 * brian
def steven_shirts (andrew : ℕ) : ℕ := 4 * andrew

theorem steven_has_72_shirts : steven_shirts (andrew_shirts brian_shirts) = 72 := 
by 
  -- We add "sorry" here to indicate that the proof is omitted
  sorry

end NUMINAMATH_GPT_steven_has_72_shirts_l1472_147208


namespace NUMINAMATH_GPT_sum_f_84_eq_1764_l1472_147222

theorem sum_f_84_eq_1764 (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, 0 < n → f n < f (n + 1))
  (h2 : ∀ m n : ℕ, 0 < m → 0 < n → f (m * n) = f m * f n)
  (h3 : ∀ m n : ℕ, 0 < m → 0 < n → m ≠ n → m^n = n^m → (f m = n ∨ f n = m)) :
  f 84 = 1764 :=
by
  sorry

end NUMINAMATH_GPT_sum_f_84_eq_1764_l1472_147222


namespace NUMINAMATH_GPT_number_of_books_before_purchase_l1472_147269

theorem number_of_books_before_purchase (x : ℕ) (h1 : x + 140 = (27 / 25) * x) : x = 1750 :=
by
  sorry

end NUMINAMATH_GPT_number_of_books_before_purchase_l1472_147269


namespace NUMINAMATH_GPT_sin_minus_cos_eq_sqrt3_div2_l1472_147234

theorem sin_minus_cos_eq_sqrt3_div2
  (α : ℝ) 
  (h_range : (Real.pi / 4) < α ∧ α < (Real.pi / 2))
  (h_sincos : Real.sin α * Real.cos α = 1 / 8) :
  Real.sin α - Real.cos α = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_minus_cos_eq_sqrt3_div2_l1472_147234


namespace NUMINAMATH_GPT_parabola_equation_l1472_147241

open Classical

noncomputable def circle_center : ℝ × ℝ := (2, 0)

theorem parabola_equation (vertex : ℝ × ℝ) (focus : ℝ × ℝ) :
  vertex = (0, 0) ∧ focus = circle_center → ∀ x y : ℝ, y^2 = 8 * x := by
  intro h
  sorry

end NUMINAMATH_GPT_parabola_equation_l1472_147241


namespace NUMINAMATH_GPT_math_problem_l1472_147200

theorem math_problem (f_star f_ast : ℕ → ℕ → ℕ) (h₁ : f_star 20 5 = 15) (h₂ : f_ast 15 5 = 75) :
  (f_star 8 4) / (f_ast 10 2) = (1:ℚ) / 5 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1472_147200


namespace NUMINAMATH_GPT_intersection_and_area_l1472_147280

theorem intersection_and_area (A B : ℝ × ℝ) (x y : ℝ):
  (x - 2 * y - 5 = 0) → (x ^ 2 + y ^ 2 = 50) →
  (A = (-5, -5) ∨ A = (7, 1)) → (B = (-5, -5) ∨ B = (7, 1)) →
  (A ≠ B) →
  ∃ (area : ℝ), area = 15 :=
by
  sorry

end NUMINAMATH_GPT_intersection_and_area_l1472_147280


namespace NUMINAMATH_GPT_second_year_undeclared_fraction_l1472_147270

def total_students := 12

def fraction_first_year : ℚ := 1 / 4
def fraction_second_year : ℚ := 1 / 2
def fraction_third_year : ℚ := 1 / 6
def fraction_fourth_year : ℚ := 1 / 12

def fraction_undeclared_first_year : ℚ := 4 / 5
def fraction_undeclared_second_year : ℚ := 3 / 4
def fraction_undeclared_third_year : ℚ := 1 / 3
def fraction_undeclared_fourth_year : ℚ := 1 / 6

def students_first_year : ℚ := total_students * fraction_first_year
def students_second_year : ℚ := total_students * fraction_second_year
def students_third_year : ℚ := total_students * fraction_third_year
def students_fourth_year : ℚ := total_students * fraction_fourth_year

def undeclared_first_year : ℚ := students_first_year * fraction_undeclared_first_year
def undeclared_second_year : ℚ := students_second_year * fraction_undeclared_second_year
def undeclared_third_year : ℚ := students_third_year * fraction_undeclared_third_year
def undeclared_fourth_year : ℚ := students_fourth_year * fraction_undeclared_fourth_year

theorem second_year_undeclared_fraction :
  (undeclared_second_year / total_students) = 1 / 3 :=
by
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_second_year_undeclared_fraction_l1472_147270
