import Mathlib

namespace NUMINAMATH_GPT_vacation_books_l511_51122

-- Define the number of mystery, fantasy, and biography novels.
def num_mystery : ℕ := 3
def num_fantasy : ℕ := 4
def num_biography : ℕ := 3

-- Define the condition that we want to choose three books with no more than one from each genre.
def num_books_to_choose : ℕ := 3
def max_books_per_genre : ℕ := 1

-- The number of ways to choose one book from each genre
def num_combinations (m f b : ℕ) : ℕ :=
  m * f * b

-- Prove that the number of possible sets of books is 36
theorem vacation_books : num_combinations num_mystery num_fantasy num_biography = 36 := by
  sorry

end NUMINAMATH_GPT_vacation_books_l511_51122


namespace NUMINAMATH_GPT_sqrt_prime_irrational_l511_51159

theorem sqrt_prime_irrational (p : ℕ) (hp : Nat.Prime p) : Irrational (Real.sqrt p) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_prime_irrational_l511_51159


namespace NUMINAMATH_GPT_segment_AB_length_l511_51188

-- Defining the conditions
def area_ratio (AB CD : ℝ) : Prop := AB / CD = 5 / 2
def length_sum (AB CD : ℝ) : Prop := AB + CD = 280

-- The theorem stating the problem
theorem segment_AB_length (AB CD : ℝ) (h₁ : area_ratio AB CD) (h₂ : length_sum AB CD) : AB = 200 :=
by {
  -- Proof step would be inserted here, but it is omitted as per instructions
  sorry
}

end NUMINAMATH_GPT_segment_AB_length_l511_51188


namespace NUMINAMATH_GPT_problem_1_problem_2_l511_51155

def simplify_calc : Prop :=
  125 * 3.2 * 25 = 10000

def solve_equation : Prop :=
  ∀ x: ℝ, 24 * (x - 12) = 16 * (x - 4) → x = 28

theorem problem_1 : simplify_calc :=
by
  sorry

theorem problem_2 : solve_equation :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l511_51155


namespace NUMINAMATH_GPT_factorize_expression_l511_51109

theorem factorize_expression (a x : ℝ) : -a*x^2 + 2*a*x - a = -a*(x - 1)^2 := 
sorry

end NUMINAMATH_GPT_factorize_expression_l511_51109


namespace NUMINAMATH_GPT_susan_remaining_spaces_l511_51172

def susan_first_turn_spaces : ℕ := 15
def susan_second_turn_spaces : ℕ := 7 - 5
def susan_third_turn_spaces : ℕ := 20
def susan_fourth_turn_spaces : ℕ := 0
def susan_fifth_turn_spaces : ℕ := 10 - 8
def susan_sixth_turn_spaces : ℕ := 0
def susan_seventh_turn_roll : ℕ := 6
def susan_seventh_turn_spaces : ℕ := susan_seventh_turn_roll * 2
def susan_total_moved_spaces : ℕ := susan_first_turn_spaces + susan_second_turn_spaces + susan_third_turn_spaces + susan_fourth_turn_spaces + susan_fifth_turn_spaces + susan_sixth_turn_spaces + susan_seventh_turn_spaces
def game_total_spaces : ℕ := 100

theorem susan_remaining_spaces : susan_total_moved_spaces = 51 ∧ (game_total_spaces - susan_total_moved_spaces) = 49 := by
  sorry

end NUMINAMATH_GPT_susan_remaining_spaces_l511_51172


namespace NUMINAMATH_GPT_num_possible_pairs_l511_51121

theorem num_possible_pairs (a b : ℕ) (h1 : b > a) (h2 : (a - 8) * (b - 8) = 32) : 
    (∃ n, n = 3) :=
by { sorry }

end NUMINAMATH_GPT_num_possible_pairs_l511_51121


namespace NUMINAMATH_GPT_five_times_remaining_is_400_l511_51135

-- Define the conditions
def original_marbles := 800
def marbles_per_friend := 120
def num_friends := 6

-- Calculate total marbles given away
def marbles_given_away := num_friends * marbles_per_friend

-- Calculate marbles remaining after giving away
def marbles_remaining := original_marbles - marbles_given_away

-- Question: what is five times the marbles remaining?
def five_times_remaining_marbles := 5 * marbles_remaining

-- The proof problem: prove that this equals 400
theorem five_times_remaining_is_400 : five_times_remaining_marbles = 400 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_five_times_remaining_is_400_l511_51135


namespace NUMINAMATH_GPT_num_pure_Gala_trees_l511_51139

-- Define the problem statement conditions
variables (T F G H : ℝ)
variables (c1 : 0.125 * F + 0.075 * F + F = 315)
variables (c2 : F = (2 / 3) * T)
variables (c3 : H = (1 / 6) * T)
variables (c4 : T = F + G + H)

-- Prove the number of pure Gala trees G is 66
theorem num_pure_Gala_trees : G = 66 :=
by
  -- Proof will be filled out here
  sorry

end NUMINAMATH_GPT_num_pure_Gala_trees_l511_51139


namespace NUMINAMATH_GPT_matrix_determinant_eq_16_l511_51197

theorem matrix_determinant_eq_16 (x : ℝ) :
  (3 * x) * (4 * x) - (2 * x) = 16 ↔ x = 4 / 3 ∨ x = -1 :=
by sorry

end NUMINAMATH_GPT_matrix_determinant_eq_16_l511_51197


namespace NUMINAMATH_GPT_vasya_max_earning_l511_51173

theorem vasya_max_earning (k : ℕ) (h₀: k ≤ 2013) (h₁: 2013 - 2*k % 11 = 0) : k % 11 = 0 → (k ≤ 5) := 
by
  sorry

end NUMINAMATH_GPT_vasya_max_earning_l511_51173


namespace NUMINAMATH_GPT_two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one_l511_51166

theorem two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one
  (p a n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hn : 0 < n) 
  (h : 2 ^ p + 3 ^ p = a ^ n) : n = 1 :=
sorry

end NUMINAMATH_GPT_two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one_l511_51166


namespace NUMINAMATH_GPT_total_masks_correct_l511_51104

-- Define the conditions
def boxes := 18
def capacity_per_box := 15
def deficiency_per_box := 3
def masks_per_box := capacity_per_box - deficiency_per_box
def total_masks := boxes * masks_per_box

-- The theorem statement we need to prove
theorem total_masks_correct : total_masks = 216 := by
  unfold total_masks boxes masks_per_box capacity_per_box deficiency_per_box
  sorry

end NUMINAMATH_GPT_total_masks_correct_l511_51104


namespace NUMINAMATH_GPT_probability_of_red_black_or_white_l511_51106

def numberOfBalls := 12
def redBalls := 5
def blackBalls := 4
def whiteBalls := 2
def greenBalls := 1

def favorableOutcomes : Nat := redBalls + blackBalls + whiteBalls
def totalOutcomes : Nat := numberOfBalls

theorem probability_of_red_black_or_white :
  (favorableOutcomes : ℚ) / (totalOutcomes : ℚ) = 11 / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_red_black_or_white_l511_51106


namespace NUMINAMATH_GPT_quotient_of_sum_l511_51137

theorem quotient_of_sum (a b c x y z : ℝ)
  (h1 : a^2 + b^2 + c^2 = 25)
  (h2 : x^2 + y^2 + z^2 = 36)
  (h3 : a * x + b * y + c * z = 30) :
  (a + b + c) / (x + y + z) = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_quotient_of_sum_l511_51137


namespace NUMINAMATH_GPT_shape_formed_is_line_segment_l511_51153

def point := (ℝ × ℝ)

noncomputable def A : point := (0, 0)
noncomputable def B : point := (0, 4)
noncomputable def C : point := (6, 4)
noncomputable def D : point := (6, 0)

noncomputable def line_eq (p1 p2 : point) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (x2 - x1, y2 - y1)

theorem shape_formed_is_line_segment :
  let l1 := line_eq A (1, 1)  -- Line from A at 45°
  let l2 := line_eq B (-1, -1) -- Line from B at -45°
  let l3 := line_eq D (1, -1) -- Line from D at 45°
  let l4 := line_eq C (-1, 5) -- Line from C at -45°
  let intersection1 := (5, 5)  -- Intersection of l1 and l4: solve x = 10 - x
  let intersection2 := (5, -1)  -- Intersection of l2 and l3: solve 4 - x = x - 6
  intersection1.1 = intersection2.1 := 
by
  sorry

end NUMINAMATH_GPT_shape_formed_is_line_segment_l511_51153


namespace NUMINAMATH_GPT_number_of_valid_groupings_l511_51171

-- Definitions based on conditions
def num_guides : ℕ := 2
def num_tourists : ℕ := 6
def total_groupings : ℕ := 2 ^ num_tourists
def invalid_groupings : ℕ := 2  -- All tourists go to one guide either a or b

-- The theorem to prove
theorem number_of_valid_groupings : total_groupings - invalid_groupings = 62 :=
by sorry

end NUMINAMATH_GPT_number_of_valid_groupings_l511_51171


namespace NUMINAMATH_GPT_max_parabola_ratio_l511_51120

noncomputable def parabola_max_ratio (x y : ℝ) : ℝ :=
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let M : ℝ × ℝ := (x, y)
  
  let MO : ℝ := Real.sqrt (x^2 + y^2)
  let MF : ℝ := Real.sqrt ((x - 1)^2 + y^2)
  
  MO / MF

theorem max_parabola_ratio :
  ∃ x y : ℝ, y^2 = 4 * x ∧ parabola_max_ratio x y = 2 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_max_parabola_ratio_l511_51120


namespace NUMINAMATH_GPT_qin_jiushao_value_l511_51183

def polynomial (x : ℤ) : ℤ :=
  2 * x^5 + 5 * x^4 + 8 * x^3 + 7 * x^2 - 6 * x + 11

def step1 (x : ℤ) : ℤ := 2 * x + 5
def step2 (x : ℤ) (v : ℤ) : ℤ := v * x + 8
def step3 (x : ℤ) (v : ℤ) : ℤ := v * x + 7
def step_v3 (x : ℤ) (v : ℤ) : ℤ := v * x - 6

theorem qin_jiushao_value (x : ℤ) (v3 : ℤ) (h1 : x = 3) (h2 : v3 = 130) :
  step_v3 3 (step3 3 (step2 3 (step1 3))) = v3 :=
by {
  sorry
}

end NUMINAMATH_GPT_qin_jiushao_value_l511_51183


namespace NUMINAMATH_GPT_find_a6_of_arithmetic_seq_l511_51193

noncomputable def arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def sum_of_arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem find_a6_of_arithmetic_seq 
  (a1 d : ℝ) 
  (S3 : ℝ) 
  (h_a1 : a1 = 2) 
  (h_S3 : S3 = 12) 
  (h_sum : S3 = sum_of_arithmetic_sequence 3 a1 d) :
  arithmetic_sequence 6 a1 d = 12 := 
sorry

end NUMINAMATH_GPT_find_a6_of_arithmetic_seq_l511_51193


namespace NUMINAMATH_GPT_suji_present_age_l511_51190

/-- Present ages of Abi and Suji are in the ratio of 5:4. --/
def abi_suji_ratio (abi_age suji_age : ℕ) : Prop := abi_age = 5 * (suji_age / 4)

/-- 3 years hence, the ratio of their ages will be 11:9. --/
def abi_suji_ratio_future (abi_age suji_age : ℕ) : Prop :=
  ((abi_age + 3).toFloat / (suji_age + 3).toFloat) = 11 / 9

theorem suji_present_age (suji_age : ℕ) (abi_age : ℕ) (x : ℕ) 
  (h1 : abi_age = 5 * x) (h2 : suji_age = 4 * x)
  (h3 : abi_suji_ratio_future abi_age suji_age) :
  suji_age = 24 := 
sorry

end NUMINAMATH_GPT_suji_present_age_l511_51190


namespace NUMINAMATH_GPT_r_needs_35_days_l511_51167

def work_rate (P Q R: ℚ) : Prop :=
  (P = Q + R) ∧ (P + Q = 1/10) ∧ (Q = 1/28)

theorem r_needs_35_days (P Q R: ℚ) (h: work_rate P Q R) : 1 / R = 35 :=
by 
  sorry

end NUMINAMATH_GPT_r_needs_35_days_l511_51167


namespace NUMINAMATH_GPT_at_least_one_solves_l511_51103

/--
Given probabilities p1, p2, p3 that individuals A, B, and C solve a problem respectively,
prove that the probability that at least one of them solves the problem is 
1 - (1 - p1) * (1 - p2) * (1 - p3).
-/
theorem at_least_one_solves (p1 p2 p3 : ℝ) (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) (h3 : 0 ≤ p3 ∧ p3 ≤ 1) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 1 - (1 - p1) * (1 - p2) * (1 - p3) :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_solves_l511_51103


namespace NUMINAMATH_GPT_multiply_469160_999999_l511_51177

theorem multiply_469160_999999 :
  469160 * 999999 = 469159530840 :=
by
  sorry

end NUMINAMATH_GPT_multiply_469160_999999_l511_51177


namespace NUMINAMATH_GPT_kathleen_remaining_money_l511_51184

-- Define the conditions
def saved_june := 21
def saved_july := 46
def saved_august := 45
def spent_school_supplies := 12
def spent_clothes := 54
def aunt_gift_threshold := 125
def aunt_gift := 25

-- Prove that Kathleen has the correct remaining amount of money
theorem kathleen_remaining_money : 
    (saved_june + saved_july + saved_august) - 
    (spent_school_supplies + spent_clothes) = 46 := 
by
  sorry

end NUMINAMATH_GPT_kathleen_remaining_money_l511_51184


namespace NUMINAMATH_GPT_rectangle_area_change_area_analysis_l511_51148

noncomputable def original_area (a b : ℝ) : ℝ := a * b

noncomputable def new_area (a b : ℝ) : ℝ := (a - 3) * (b + 3)

theorem rectangle_area_change (a b : ℝ) :
  let S := original_area a b
  let S₁ := new_area a b
  S₁ - S = 3 * (a - b - 3) :=
by
  sorry

theorem area_analysis (a b : ℝ) :
  if a - b - 3 = 0 then new_area a b = original_area a b
  else if a - b - 3 > 0 then new_area a b > original_area a b
  else new_area a b < original_area a b :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_change_area_analysis_l511_51148


namespace NUMINAMATH_GPT_largest_a_pow_b_l511_51151

theorem largest_a_pow_b (a b : ℕ) (h_pos_a : 1 < a) (h_pos_b : 1 < b) (h_eq : a^b * b^a + a^b + b^a = 5329) : 
  a^b = 64 :=
by
  sorry

end NUMINAMATH_GPT_largest_a_pow_b_l511_51151


namespace NUMINAMATH_GPT_points_with_tangent_length_six_l511_51174

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y + 4 = 0

-- Define the property of a point having a tangent of length 6 to the circle
def tangent_length_six (h k cx cy r : ℝ) : Prop :=
  (cx - h)^2 + (cy - k)^2 - r^2 = 36

-- Main theorem statement
theorem points_with_tangent_length_six : 
  (∀ x1 y1 : ℝ, (x1 = -4 ∧ y1 = 6) ∨ (x1 = 5 ∧ y1 = -3) → 
    (∃ r1 : ℝ, tangent_length_six x1 y1 (-1) 0 3) ∧ 
    (∃ r2 : ℝ, tangent_length_six x1 y1 2 3 3)) :=
  by 
  sorry

end NUMINAMATH_GPT_points_with_tangent_length_six_l511_51174


namespace NUMINAMATH_GPT_arithmetic_sequence_S2008_l511_51182

theorem arithmetic_sequence_S2008 (a1 : ℤ) (S : ℕ → ℤ) (d : ℤ)
  (h1 : a1 = -2008)
  (h2 : ∀ n, S n = n * a1 + n * (n - 1) / 2 * d)
  (h3 : (S 12 / 12) - (S 10 / 10) = 2) :
  S 2008 = -2008 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_S2008_l511_51182


namespace NUMINAMATH_GPT_parallel_vectors_m_eq_neg3_l511_51169

theorem parallel_vectors_m_eq_neg3 {m : ℝ} :
  let a := (1, -2)
  let b := (1 + m, 1 - m)
  (a.1 * b.2 =  a.2 * b.1) → m = -3 :=
by 
  let a := (1, -2)
  let b := (1 + m, 1 - m)
  intro h
  sorry

end NUMINAMATH_GPT_parallel_vectors_m_eq_neg3_l511_51169


namespace NUMINAMATH_GPT_compound_interest_years_l511_51132

-- Definitions for the given conditions
def principal : ℝ := 1200
def rate : ℝ := 0.20
def compound_interest : ℝ := 873.60
def compounded_yearly : ℝ := 1

-- Calculate the future value from principal and compound interest
def future_value : ℝ := principal + compound_interest

-- Statement of the problem: Prove that the number of years t was 3 given the conditions
theorem compound_interest_years :
  ∃ (t : ℝ), future_value = principal * (1 + rate / compounded_yearly)^(compounded_yearly * t) := sorry

end NUMINAMATH_GPT_compound_interest_years_l511_51132


namespace NUMINAMATH_GPT_cost_of_apples_and_oranges_correct_l511_51163

-- Define the initial money jasmine had
def initial_money : ℝ := 100.00

-- Define the remaining money after purchase
def remaining_money : ℝ := 85.00

-- Define the cost of apples and oranges
def cost_of_apples_and_oranges : ℝ := initial_money - remaining_money

-- This is our theorem statement that needs to be proven
theorem cost_of_apples_and_oranges_correct :
  cost_of_apples_and_oranges = 15.00 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_apples_and_oranges_correct_l511_51163


namespace NUMINAMATH_GPT_impossible_to_convince_logical_jury_of_innocence_if_guilty_l511_51118

theorem impossible_to_convince_logical_jury_of_innocence_if_guilty :
  (guilty : Prop) →
  (jury_is_logical : Prop) →
  guilty →
  (∀ statement : Prop, (logical_deduction : Prop) → (logical_deduction → ¬guilty)) →
  False :=
by
  intro guilty jury_is_logical guilty_premise logical_argument
  sorry

end NUMINAMATH_GPT_impossible_to_convince_logical_jury_of_innocence_if_guilty_l511_51118


namespace NUMINAMATH_GPT_determine_sixth_face_l511_51180

-- Define a cube configuration and corresponding functions
inductive Color
| black
| white

structure Cube where
  faces : Fin 6 → Fin 9 → Color

noncomputable def sixth_face_color (cube : Cube) : Fin 9 → Color := sorry

-- The statement of the theorem proving the coloring of the sixth face
theorem determine_sixth_face (cube : Cube) : 
  (exists f : (Fin 9 → Color), f = sixth_face_color cube) := 
sorry

end NUMINAMATH_GPT_determine_sixth_face_l511_51180


namespace NUMINAMATH_GPT_probability_at_least_6_heads_in_8_flips_l511_51161

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

theorem probability_at_least_6_heads_in_8_flips :
  let total_outcomes := 2^8
  let successful_outcomes := binomial 8 6 + binomial 8 7 + binomial 8 8
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 37 / 256 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_6_heads_in_8_flips_l511_51161


namespace NUMINAMATH_GPT_problem_statement_l511_51125

theorem problem_statement (a b c : ℤ) (h : c = b + 2) : 
  (a - (b + c)) - ((a + c) - b) = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l511_51125


namespace NUMINAMATH_GPT_largest_integer_remainder_l511_51110

theorem largest_integer_remainder :
  ∃ (a : ℤ), a < 61 ∧ a % 6 = 5 ∧ ∀ b : ℤ, b < 61 ∧ b % 6 = 5 → b ≤ a :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_remainder_l511_51110


namespace NUMINAMATH_GPT_positive_integer_is_48_l511_51189

theorem positive_integer_is_48 (n p : ℕ) (h_prime : Prime p) (h_eq : n = 24 * p) (h_min : n ≥ 48) : n = 48 :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_is_48_l511_51189


namespace NUMINAMATH_GPT_carpet_dimensions_l511_51150

-- Define the problem parameters
def width_a : ℕ := 50
def width_b : ℕ := 38

-- The dimensions x and y are integral numbers of feet
variables (x y : ℕ)

-- The same length L for both rooms that touches all four walls
noncomputable def length (x y : ℕ) : ℚ := (22 * (x^2 + y^2)) / (x * y)

-- The final theorem to be proven
theorem carpet_dimensions (x y : ℕ) (h : (x^2 + y^2) * 1056 = (x * y) * 48 * (length x y)) : (x = 50) ∧ (y = 25) :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_carpet_dimensions_l511_51150


namespace NUMINAMATH_GPT_number_of_factors_of_x_l511_51138

theorem number_of_factors_of_x (a b c : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Prime b) (h3 : Nat.Prime c) (h4 : a < b) (h5 : b < c) (h6 : ¬ a = b) (h7 : ¬ b = c) (h8 : ¬ a = c) :
  let x := 2^2 * a^3 * b^2 * c^4
  let num_factors := (2 + 1) * (3 + 1) * (2 + 1) * (4 + 1)
  num_factors = 180 := by
sorry

end NUMINAMATH_GPT_number_of_factors_of_x_l511_51138


namespace NUMINAMATH_GPT_find_d_l511_51176

-- Definitions based on conditions
def f (x : ℝ) (c : ℝ) := 5 * x + c
def g (x : ℝ) (c : ℝ) := c * x + 3

-- The theorem statement
theorem find_d (c d : ℝ) (h₁ : f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry -- Proof is omitted as per the instructions

end NUMINAMATH_GPT_find_d_l511_51176


namespace NUMINAMATH_GPT_least_integer_square_double_l511_51156

theorem least_integer_square_double (x : ℤ) : x^2 = 2 * x + 50 → x = -5 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_square_double_l511_51156


namespace NUMINAMATH_GPT_smallest_value_of_x_l511_51185

theorem smallest_value_of_x (x : ℝ) (h : 6 * x ^ 2 - 37 * x + 48 = 0) : x = 13 / 6 :=
sorry

end NUMINAMATH_GPT_smallest_value_of_x_l511_51185


namespace NUMINAMATH_GPT_horses_tiles_equation_l511_51131

-- Conditions from the problem
def total_horses (x y : ℕ) : Prop := x + y = 100
def total_tiles (x y : ℕ) : Prop := 3 * x + (1 / 3 : ℚ) * y = 100

-- The statement to prove
theorem horses_tiles_equation (x y : ℕ) :
  total_horses x y ∧ total_tiles x y ↔ 
  (x + y = 100 ∧ (3 * x + (1 / 3 : ℚ) * y = 100)) :=
by
  sorry

end NUMINAMATH_GPT_horses_tiles_equation_l511_51131


namespace NUMINAMATH_GPT_expression_simplified_l511_51186

noncomputable def expression : ℚ := 1 + 3 / (4 + 5 / 6)

theorem expression_simplified : expression = 47 / 29 :=
by
  sorry

end NUMINAMATH_GPT_expression_simplified_l511_51186


namespace NUMINAMATH_GPT_find_f_prime_at_one_l511_51179

theorem find_f_prime_at_one (a b : ℝ)
  (h1 : ∀ x, f x = a * Real.exp x + b * x) 
  (h2 : f 0 = 1)
  (h3 : ∀ x, deriv f x = a * Real.exp x + b)
  (h4 : deriv f 0 = 0) :
  deriv f 1 = Real.exp 1 - 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_f_prime_at_one_l511_51179


namespace NUMINAMATH_GPT_extreme_value_at_x_eq_one_l511_51144

noncomputable def f (x a b: ℝ) : ℝ := x^3 - a * x^2 + b * x + a^2
noncomputable def f_prime (x a b: ℝ) : ℝ := 3 * x^2 - 2 * a * x + b

theorem extreme_value_at_x_eq_one (a b : ℝ) (h_prime : f_prime 1 a b = 0) (h_value : f 1 a b = 10) : a = -4 :=
by 
  sorry -- proof goes here

end NUMINAMATH_GPT_extreme_value_at_x_eq_one_l511_51144


namespace NUMINAMATH_GPT_minimum_value_expression_l511_51192

theorem minimum_value_expression :
  ∃ x y : ℝ, (∀ a b : ℝ, (a^2 + 4*a*b + 5*b^2 - 8*a - 6*b) ≥ -41) ∧ (x^2 + 4*x*y + 5*y^2 - 8*x - 6*y) = -41 := 
sorry

end NUMINAMATH_GPT_minimum_value_expression_l511_51192


namespace NUMINAMATH_GPT_sin_double_angle_identity_l511_51157

noncomputable def given_tan_alpha (α : ℝ) : Prop := 
  Real.tan α = 1/2

theorem sin_double_angle_identity (α : ℝ) (h : given_tan_alpha α) : 
  Real.sin (2 * α) = 4 / 5 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_identity_l511_51157


namespace NUMINAMATH_GPT_solve_simultaneous_equations_l511_51123

theorem solve_simultaneous_equations :
  (∃ x y : ℝ, x^2 + 3 * y = 10 ∧ 3 + y = 10 / x) ↔ 
  (x = 3 ∧ y = 1 / 3) ∨ 
  (x = 2 ∧ y = 2) ∨ 
  (x = -5 ∧ y = -5) := by sorry

end NUMINAMATH_GPT_solve_simultaneous_equations_l511_51123


namespace NUMINAMATH_GPT_sin_double_angle_l511_51165

theorem sin_double_angle (θ : ℝ) (h : Real.sin (π / 4 + θ) = 1 / 3) : Real.sin (2 * θ) = -7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l511_51165


namespace NUMINAMATH_GPT_probability_all_girls_is_correct_l511_51116

noncomputable def probability_all_girls : ℚ :=
  let total_members := 15
  let boys := 7
  let girls := 8
  let choose_3_from_15 := Nat.choose total_members 3
  let choose_3_from_8 := Nat.choose girls 3
  choose_3_from_8 / choose_3_from_15

theorem probability_all_girls_is_correct : 
  probability_all_girls = 8 / 65 := by
sorry

end NUMINAMATH_GPT_probability_all_girls_is_correct_l511_51116


namespace NUMINAMATH_GPT_determine_Y_in_arithmetic_sequence_matrix_l511_51175

theorem determine_Y_in_arithmetic_sequence_matrix :
  (exists a₁ a₂ a₃ a₄ a₅ : ℕ, 
    -- Conditions for the first row (arithmetic sequence with first term 3 and fifth term 15)
    a₁ = 3 ∧ a₅ = 15 ∧ 
    (∃ d₁ : ℕ, a₂ = a₁ + d₁ ∧ a₃ = a₂ + d₁ ∧ a₄ = a₃ + d₁ ∧ a₅ = a₄ + d₁) ∧

    -- Conditions for the fifth row (arithmetic sequence with first term 25 and fifth term 65)
    a₁ = 25 ∧ a₅ = 65 ∧ 
    (∃ d₅ : ℕ, a₂ = a₁ + d₅ ∧ a₃ = a₂ + d₅ ∧ a₄ = a₃ + d₅ ∧ a₅ = a₄ + d₅) ∧

    -- Middle element Y
    a₃ = 27) :=
sorry

end NUMINAMATH_GPT_determine_Y_in_arithmetic_sequence_matrix_l511_51175


namespace NUMINAMATH_GPT_author_earnings_calculation_l511_51113

open Real

namespace AuthorEarnings

def paperCoverCopies  : ℕ := 32000
def paperCoverPrice   : ℝ := 0.20
def paperCoverPercent : ℝ := 0.06

def hardCoverCopies   : ℕ := 15000
def hardCoverPrice    : ℝ := 0.40
def hardCoverPercent  : ℝ := 0.12

def total_earnings_paper_cover : ℝ := paperCoverCopies * paperCoverPrice
def earnings_paper_cover : ℝ := total_earnings_paper_cover * paperCoverPercent

def total_earnings_hard_cover : ℝ := hardCoverCopies * hardCoverPrice
def earnings_hard_cover : ℝ := total_earnings_hard_cover * hardCoverPercent

def author_total_earnings : ℝ := earnings_paper_cover + earnings_hard_cover

theorem author_earnings_calculation : author_total_earnings = 1104 := by
  sorry

end AuthorEarnings

end NUMINAMATH_GPT_author_earnings_calculation_l511_51113


namespace NUMINAMATH_GPT_negation_of_forall_statement_l511_51130

theorem negation_of_forall_statement :
  ¬ (∀ x : ℝ, x^2 + 2 * x ≥ 0) ↔ ∃ x : ℝ, x^2 + 2 * x < 0 := 
by
  sorry

end NUMINAMATH_GPT_negation_of_forall_statement_l511_51130


namespace NUMINAMATH_GPT_F_double_reflection_l511_51154

structure Point where
  x : ℝ
  y : ℝ

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def F : Point := { x := -1, y := -1 }

theorem F_double_reflection :
  reflect_x (reflect_y F) = { x := 1, y := 1 } :=
  sorry

end NUMINAMATH_GPT_F_double_reflection_l511_51154


namespace NUMINAMATH_GPT_uncounted_angle_measure_l511_51143

-- Define the given miscalculated sum
def miscalculated_sum : ℝ := 2240

-- Define the correct sum expression for an n-sided convex polygon
def correct_sum (n : ℕ) : ℝ := (n - 2) * 180

-- State the theorem: 
theorem uncounted_angle_measure (n : ℕ) (h1 : correct_sum n = 2340) (h2 : 2240 < correct_sum n) :
  correct_sum n - miscalculated_sum = 100 := 
by sorry

end NUMINAMATH_GPT_uncounted_angle_measure_l511_51143


namespace NUMINAMATH_GPT_chicken_feathers_after_crossing_l511_51133

def cars_dodged : ℕ := 23
def initial_feathers : ℕ := 5263
def feathers_lost : ℕ := 2 * cars_dodged
def final_feathers : ℕ := initial_feathers - feathers_lost

theorem chicken_feathers_after_crossing :
  final_feathers = 5217 := by
sorry

end NUMINAMATH_GPT_chicken_feathers_after_crossing_l511_51133


namespace NUMINAMATH_GPT_greyson_spent_on_fuel_l511_51199

theorem greyson_spent_on_fuel : ∀ (cost_per_refill times_refilled total_cost : ℕ), 
  cost_per_refill = 10 → 
  times_refilled = 4 → 
  total_cost = cost_per_refill * times_refilled → 
  total_cost = 40 :=
by
  intro cost_per_refill times_refilled total_cost
  intro h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  exact h3

end NUMINAMATH_GPT_greyson_spent_on_fuel_l511_51199


namespace NUMINAMATH_GPT_stratified_sampling_third_year_students_l511_51198

/-- 
A university's mathematics department has a total of 5000 undergraduate students, 
with the first, second, third, and fourth years having a ratio of their numbers as 4:3:2:1. 
If stratified sampling is employed to select a sample of 200 students from all undergraduates,
prove that the number of third-year students to be sampled is 40.
-/
theorem stratified_sampling_third_year_students :
  let total_students := 5000
  let ratio_first_second_third_fourth := (4, 3, 2, 1)
  let sample_size := 200
  let third_year_ratio := 2
  let total_ratio_units := 4 + 3 + 2 + 1
  let proportion_third_year := third_year_ratio / total_ratio_units
  let expected_third_year_students := sample_size * proportion_third_year
  expected_third_year_students = 40 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_third_year_students_l511_51198


namespace NUMINAMATH_GPT_distinct_real_roots_range_of_m_l511_51136

theorem distinct_real_roots_range_of_m (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + x₁ - m = 0) ∧ (x₂^2 + x₂ - m = 0)) → m > -1/4 := 
sorry

end NUMINAMATH_GPT_distinct_real_roots_range_of_m_l511_51136


namespace NUMINAMATH_GPT_car_distance_travelled_l511_51127

theorem car_distance_travelled (time_hours : ℝ) (time_minutes : ℝ) (time_seconds : ℝ)
    (actual_speed : ℝ) (reduced_speed : ℝ) (distance : ℝ) :
    time_hours = 1 → 
    time_minutes = 40 →
    time_seconds = 48 →
    actual_speed = 34.99999999999999 → 
    reduced_speed = (5 / 7) * actual_speed → 
    distance = reduced_speed * ((time_hours + time_minutes / 60 + time_seconds / 3600) : ℝ) →
    distance = 42 := sorry

end NUMINAMATH_GPT_car_distance_travelled_l511_51127


namespace NUMINAMATH_GPT_calvin_weeks_buying_chips_l511_51128

variable (daily_spending : ℝ := 0.50)
variable (days_per_week : ℝ := 5)
variable (total_spending : ℝ := 10)
variable (spending_per_week := daily_spending * days_per_week)

theorem calvin_weeks_buying_chips :
  total_spending / spending_per_week = 4 := by
  sorry

end NUMINAMATH_GPT_calvin_weeks_buying_chips_l511_51128


namespace NUMINAMATH_GPT_like_terms_exponents_l511_51105

theorem like_terms_exponents (n m : ℕ) (h1 : n + 2 = 3) (h2 : 2 * m - 1 = 3) : n = 1 ∧ m = 2 :=
by sorry

end NUMINAMATH_GPT_like_terms_exponents_l511_51105


namespace NUMINAMATH_GPT_periodicity_f_l511_51111

noncomputable def vectorA (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos x)
noncomputable def vectorB (x : ℝ) : ℝ × ℝ := (Real.cos x, 2 * Real.sqrt 3 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ :=
  let a := vectorA x
  let b := vectorB x
  a.1 * b.1 + a.2 * b.2

theorem periodicity_f :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), (f x = 2 + Real.sqrt 3 ∨ f x = 0)) :=
by
  sorry

end NUMINAMATH_GPT_periodicity_f_l511_51111


namespace NUMINAMATH_GPT_total_feet_in_garden_l511_51107

theorem total_feet_in_garden (num_dogs num_ducks feet_per_dog feet_per_duck : ℕ)
  (h1 : num_dogs = 6) (h2 : num_ducks = 2)
  (h3 : feet_per_dog = 4) (h4 : feet_per_duck = 2) :
  num_dogs * feet_per_dog + num_ducks * feet_per_duck = 28 :=
by
  sorry

end NUMINAMATH_GPT_total_feet_in_garden_l511_51107


namespace NUMINAMATH_GPT_linear_dependent_iff_38_div_3_l511_51117

theorem linear_dependent_iff_38_div_3 (k : ℚ) :
  k = 38 / 3 ↔ ∃ (α β γ : ℚ), α ≠ 0 ∨ β ≠ 0 ∨ γ ≠ 0 ∧
    α * 1 + β * 4 + γ * 7 = 0 ∧
    α * 2 + β * 5 + γ * 8 = 0 ∧
    α * 3 + β * k + γ * 9 = 0 :=
by
  sorry

end NUMINAMATH_GPT_linear_dependent_iff_38_div_3_l511_51117


namespace NUMINAMATH_GPT_original_price_of_computer_l511_51102

theorem original_price_of_computer
  (P : ℝ)
  (h1 : 1.30 * P = 351)
  (h2 : 2 * P = 540) :
  P = 270 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_computer_l511_51102


namespace NUMINAMATH_GPT_N_is_composite_l511_51195

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ Prime N :=
by {
  sorry
}

end NUMINAMATH_GPT_N_is_composite_l511_51195


namespace NUMINAMATH_GPT_orange_profit_loss_l511_51119

variable (C : ℝ) -- Cost price of one orange in rupees

-- Conditions as hypotheses
theorem orange_profit_loss :
  (1 / 16 - C) / C * 100 = 4 :=
by
  have h1 : 1.28 * C = 1 / 12 := sorry
  have h2 : C = 1 / (12 * 1.28) := sorry
  have h3 : C = 1 / 15.36 := sorry
  have h4 : (1/16 - C) = 1 / 384 := sorry
  -- Proof of main statement here
  sorry

end NUMINAMATH_GPT_orange_profit_loss_l511_51119


namespace NUMINAMATH_GPT_range_of_a_l511_51160

-- Definitions
def domain_f : Set ℝ := {x : ℝ | x ≤ -4 ∨ x ≥ 4}
def range_g (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ y = x^2 - 2*x + a}

-- Theorem to prove the range of values for a
theorem range_of_a :
  (∀ x : ℝ, x ∈ domain_f ∨ (∃ y : ℝ, ∃ a : ℝ, y ∈ range_g a ∧ x = y)) ↔ (-4 ≤ a ∧ a ≤ -3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l511_51160


namespace NUMINAMATH_GPT_geometric_sum_S5_l511_51187

variable (a_n : ℕ → ℝ)
variable (S : ℕ → ℝ)

def geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a_n (n+1) = a_n n * q

theorem geometric_sum_S5 (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_geom : geometric_sequence a_n)
  (h_cond1 : a_n 2 * a_n 3 = 8 * a_n 1)
  (h_cond2 : (a_n 4 + 2 * a_n 5) / 2 = 20) :
  S 5 = 31 :=
sorry

end NUMINAMATH_GPT_geometric_sum_S5_l511_51187


namespace NUMINAMATH_GPT_cuboid_inequality_l511_51196

theorem cuboid_inequality 
  (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 = 1) : 
  4*a + 4*b + 4*c + 4*a*b + 4*a*c + 4*b*c + 4*a*b*c < 12 := by
  sorry

end NUMINAMATH_GPT_cuboid_inequality_l511_51196


namespace NUMINAMATH_GPT_power_of_two_minus_one_divisible_by_seven_power_of_two_plus_one_not_divisible_by_seven_l511_51191

theorem power_of_two_minus_one_divisible_by_seven (n : ℕ) (hn : 0 < n) : 
  (∃ k : ℕ, 0 < k ∧ n = k * 3) ↔ (7 ∣ 2^n - 1) :=
by sorry

theorem power_of_two_plus_one_not_divisible_by_seven (n : ℕ) (hn : 0 < n) :
  ¬(7 ∣ 2^n + 1) :=
by sorry

end NUMINAMATH_GPT_power_of_two_minus_one_divisible_by_seven_power_of_two_plus_one_not_divisible_by_seven_l511_51191


namespace NUMINAMATH_GPT_not_prime_41_squared_plus_41_plus_41_l511_51115

def is_prime (n : ℕ) : Prop := ∀ m k : ℕ, m * k = n → m = 1 ∨ k = 1

theorem not_prime_41_squared_plus_41_plus_41 :
  ¬ is_prime (41^2 + 41 + 41) :=
by {
  sorry
}

end NUMINAMATH_GPT_not_prime_41_squared_plus_41_plus_41_l511_51115


namespace NUMINAMATH_GPT_inequality_ac2_bc2_implies_a_b_l511_51108

theorem inequality_ac2_bc2_implies_a_b (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b :=
sorry

end NUMINAMATH_GPT_inequality_ac2_bc2_implies_a_b_l511_51108


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l511_51141

-- Define the point (2, -3)
structure Point where
  x : ℤ
  y : ℤ

def A : Point := { x := 2, y := -3 }

-- Define what it means for a point to be in a specific quadrant
def inFirstQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y > 0

def inSecondQuadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y > 0

def inThirdQuadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

def inFourthQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y < 0

-- Define the theorem to prove that the point A lies in the fourth quadrant
theorem point_in_fourth_quadrant : inFourthQuadrant A :=
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l511_51141


namespace NUMINAMATH_GPT_trigonometric_relationship_l511_51126

noncomputable def α : ℝ := Real.cos 4
noncomputable def b : ℝ := Real.cos (4 * Real.pi / 5)
noncomputable def c : ℝ := Real.sin (7 * Real.pi / 6)

theorem trigonometric_relationship : b < α ∧ α < c := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_relationship_l511_51126


namespace NUMINAMATH_GPT_smallest_side_of_triangle_l511_51170

theorem smallest_side_of_triangle (A B C : ℝ) (a b c : ℝ) 
  (hA : A = 60) (hC : C = 45) (hb : b = 4) (h_sum : A + B + C = 180) : 
  c = 4 * Real.sqrt 3 - 4 := 
sorry

end NUMINAMATH_GPT_smallest_side_of_triangle_l511_51170


namespace NUMINAMATH_GPT_tan_sub_eq_one_eight_tan_add_eq_neg_four_seven_l511_51168

variable (α β : ℝ)

theorem tan_sub_eq_one_eight (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α - β) = 1 / 8 := 
sorry

theorem tan_add_eq_neg_four_seven (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -4 / 7 := 
sorry

end NUMINAMATH_GPT_tan_sub_eq_one_eight_tan_add_eq_neg_four_seven_l511_51168


namespace NUMINAMATH_GPT_total_cards_l511_51158

def basketball_boxes : ℕ := 12
def cards_per_basketball_box : ℕ := 20
def football_boxes : ℕ := basketball_boxes - 5
def cards_per_football_box : ℕ := 25

theorem total_cards : basketball_boxes * cards_per_basketball_box + football_boxes * cards_per_football_box = 415 := by
  sorry

end NUMINAMATH_GPT_total_cards_l511_51158


namespace NUMINAMATH_GPT_trainB_reaches_in_3_hours_l511_51162

variable (trainA_speed trainB_speed : ℕ) (x t : ℝ)

-- Given conditions
axiom h1 : trainA_speed = 70
axiom h2 : trainB_speed = 105
axiom h3 : ∀ x t, 70 * x + 70 * 9 = 105 * x + 105 * t

-- Prove that train B takes 3 hours to reach destination after meeting
theorem trainB_reaches_in_3_hours : t = 3 :=
by
  sorry

end NUMINAMATH_GPT_trainB_reaches_in_3_hours_l511_51162


namespace NUMINAMATH_GPT_min_value_of_fraction_l511_51181

theorem min_value_of_fraction (a b : ℝ) (h_pos : a > 0 ∧ b > 0) (h_sum : a + 3 * b = 2) : 
  ∃ m, (∀ (a b : ℝ), a > 0 → b > 0 → a + 3 * b = 2 → 1 / a + 3 / b ≥ m) ∧ m = 8 := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_fraction_l511_51181


namespace NUMINAMATH_GPT_segment_parallel_to_x_axis_l511_51146

theorem segment_parallel_to_x_axis 
  (f : ℤ → ℤ) 
  (hf : ∀ n, ∃ m, f n = m) 
  (a b : ℤ) 
  (h_dist : ∃ d : ℤ, d * d = (b - a) * (b - a) + (f b - f a) * (f b - f a)) : 
  f a = f b :=
sorry

end NUMINAMATH_GPT_segment_parallel_to_x_axis_l511_51146


namespace NUMINAMATH_GPT_find_k_for_perfect_square_trinomial_l511_51164

noncomputable def perfect_square_trinomial (k : ℝ) : Prop :=
∀ x : ℝ, (x^2 - 8*x + k) = (x - 4)^2

theorem find_k_for_perfect_square_trinomial :
  ∃ k : ℝ, perfect_square_trinomial k ∧ k = 16 :=
by
  use 16
  sorry

end NUMINAMATH_GPT_find_k_for_perfect_square_trinomial_l511_51164


namespace NUMINAMATH_GPT_pyramid_inscribed_sphere_radius_l511_51178

noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ := 
a * Real.sqrt 2 / (2 * (2 + Real.sqrt 3))

theorem pyramid_inscribed_sphere_radius (a : ℝ) (h1 : a > 0) : 
  inscribed_sphere_radius a = a * Real.sqrt 2 / (2 * (2 + Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_GPT_pyramid_inscribed_sphere_radius_l511_51178


namespace NUMINAMATH_GPT_mary_initial_flour_l511_51124

theorem mary_initial_flour (F_total F_add F_initial : ℕ) 
  (h_total : F_total = 9)
  (h_add : F_add = 6)
  (h_initial : F_initial = F_total - F_add) :
  F_initial = 3 :=
sorry

end NUMINAMATH_GPT_mary_initial_flour_l511_51124


namespace NUMINAMATH_GPT_factorial_division_l511_51142

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division (n : ℕ) (h : n > 0) : factorial (n) / factorial (n - 1) = n :=
by
  sorry

example : factorial 12 / factorial 11 = 12 :=
by
  exact factorial_division 12 (by norm_num)

end NUMINAMATH_GPT_factorial_division_l511_51142


namespace NUMINAMATH_GPT_forgot_to_mow_l511_51101

-- Definitions
def earning_per_lawn : ℕ := 9
def lawns_to_mow : ℕ := 12
def actual_earning : ℕ := 36

-- Statement to prove
theorem forgot_to_mow : (lawns_to_mow - (actual_earning / earning_per_lawn)) = 8 := by
  sorry

end NUMINAMATH_GPT_forgot_to_mow_l511_51101


namespace NUMINAMATH_GPT_not_possible_cut_l511_51134

theorem not_possible_cut (n : ℕ) : 
  let chessboard_area := 8 * 8
  let rectangle_area := 3
  let rectangles_needed := chessboard_area / rectangle_area
  rectangles_needed ≠ n :=
by
  sorry

end NUMINAMATH_GPT_not_possible_cut_l511_51134


namespace NUMINAMATH_GPT_cost_of_materials_l511_51129

theorem cost_of_materials (initial_bracelets given_away : ℕ) (sell_price profit : ℝ)
  (h1 : initial_bracelets = 52) 
  (h2 : given_away = 8) 
  (h3 : sell_price = 0.25) 
  (h4 : profit = 8) :
  let remaining_bracelets := initial_bracelets - given_away
  let total_revenue := remaining_bracelets * sell_price
  let cost_of_materials := total_revenue - profit
  cost_of_materials = 3 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_materials_l511_51129


namespace NUMINAMATH_GPT_line_equation_passing_through_points_l511_51140

theorem line_equation_passing_through_points 
  (a₁ b₁ a₂ b₂ : ℝ)
  (h1 : 2 * a₁ + 3 * b₁ + 1 = 0)
  (h2 : 2 * a₂ + 3 * b₂ + 1 = 0)
  (h3 : ∀ (x y : ℝ), (x, y) = (2, 3) → a₁ * x + b₁ * y + 1 = 0 ∧ a₂ * x + b₂ * y + 1 = 0) :
  (∀ (x y : ℝ), (2 * x + 3 * y + 1 = 0) ↔ 
                (a₁ = x ∧ b₁ = y) ∨ (a₂ = x ∧ b₂ = y)) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_passing_through_points_l511_51140


namespace NUMINAMATH_GPT_gcd_of_ropes_l511_51100

theorem gcd_of_ropes : Nat.gcd (Nat.gcd 45 75) 90 = 15 := 
by
  sorry

end NUMINAMATH_GPT_gcd_of_ropes_l511_51100


namespace NUMINAMATH_GPT_number_of_girls_in_class_l511_51149

theorem number_of_girls_in_class (B S G : ℕ)
  (h1 : 3 * B = 4 * 18)  -- 3/4 * B = 18
  (h2 : 2 * S = 3 * B)  -- 2/3 * S = B
  (h3 : G = S - B) : G = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_in_class_l511_51149


namespace NUMINAMATH_GPT_ratio_of_P_Q_l511_51194

theorem ratio_of_P_Q (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -5 → x ≠ 0 → x ≠ 4 →
    P / (x + 5) + Q / (x^2 - 4 * x) = (x^2 + x + 15) / (x^3 + x^2 - 20 * x)) :
  Q / P = -45 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_P_Q_l511_51194


namespace NUMINAMATH_GPT_expand_product_l511_51147

noncomputable def expand_poly (x : ℝ) : ℝ := (x + 3) * (x^2 + 2 * x + 4)

theorem expand_product (x : ℝ) : expand_poly x = x^3 + 5 * x^2 + 10 * x + 12 := 
by 
  -- This will be filled with the proof steps, but for now we use sorry.
  sorry

end NUMINAMATH_GPT_expand_product_l511_51147


namespace NUMINAMATH_GPT_fish_population_estimation_l511_51152

def tagged_fish_day1 := (30, 25, 25) -- (Species A, Species B, Species C)
def tagged_fish_day2 := (40, 35, 25) -- (Species A, Species B, Species C)
def caught_fish_day3 := (60, 50, 30) -- (Species A, Species B, Species C)
def tagged_fish_day3 := (4, 6, 2)    -- (Species A, Species B, Species C)
def caught_fish_day4 := (70, 40, 50) -- (Species A, Species B, Species C)
def tagged_fish_day4 := (5, 7, 3)    -- (Species A, Species B, Species C)

def total_tagged_fish (day1 : (ℕ × ℕ × ℕ)) (day2 : (ℕ × ℕ × ℕ)) :=
  let (a1, b1, c1) := day1
  let (a2, b2, c2) := day2
  (a1 + a2, b1 + b2, c1 + c2)

def average_proportion_tagged (caught3 tagged3 caught4 tagged4 : (ℕ × ℕ × ℕ)) :=
  let (c3a, c3b, c3c) := caught3
  let (t3a, t3b, t3c) := tagged3
  let (c4a, c4b, c4c) := caught4
  let (t4a, t4b, t4c) := tagged4
  ((t3a / c3a + t4a / c4a) / 2,
   (t3b / c3b + t4b / c4b) / 2,
   (t3c / c3c + t4c / c4c) / 2)

def estimate_population (total_tagged average_proportion : (ℕ × ℕ × ℕ)) :=
  let (ta, tb, tc) := total_tagged
  let (pa, pb, pc) := average_proportion
  (ta / pa, tb / pb, tc / pc)

theorem fish_population_estimation :
  let total_tagged := total_tagged_fish tagged_fish_day1 tagged_fish_day2
  let avg_prop := average_proportion_tagged caught_fish_day3 tagged_fish_day3 caught_fish_day4 tagged_fish_day4
  estimate_population total_tagged avg_prop = (1014, 407, 790) :=
by
  sorry

end NUMINAMATH_GPT_fish_population_estimation_l511_51152


namespace NUMINAMATH_GPT_ella_distance_from_start_l511_51114

noncomputable def compute_distance (m1 : ℝ) (f1 f2 m_to_f : ℝ) : ℝ :=
  let f1' := m1 * m_to_f
  let total_west := f1' + f2
  let distance_in_feet := Real.sqrt (f1^2 + total_west^2)
  distance_in_feet / m_to_f

theorem ella_distance_from_start :
  let starting_west := 10
  let first_north := 30
  let second_west := 40
  let meter_to_feet := 3.28084 
  compute_distance starting_west first_north second_west meter_to_feet = 24.01 := sorry

end NUMINAMATH_GPT_ella_distance_from_start_l511_51114


namespace NUMINAMATH_GPT_how_many_kids_joined_l511_51145

theorem how_many_kids_joined (original_kids : ℕ) (new_kids : ℕ) (h : original_kids = 14) (h1 : new_kids = 36) :
  new_kids - original_kids = 22 :=
by
  sorry

end NUMINAMATH_GPT_how_many_kids_joined_l511_51145


namespace NUMINAMATH_GPT_range_of_m_l511_51112

theorem range_of_m (m x : ℝ) (h : (x + m) / 3 - (2 * x - 1) / 2 = m) (hx : x ≤ 0) : m ≥ 3 / 4 := 
sorry

end NUMINAMATH_GPT_range_of_m_l511_51112
