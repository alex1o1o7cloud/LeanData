import Mathlib

namespace joe_lowest_score_dropped_l30_3099

theorem joe_lowest_score_dropped (A B C D : ℕ) 
  (h1 : A + B + C + D = 160)
  (h2 : A + B + C = 135) 
  (h3 : D ≤ A ∧ D ≤ B ∧ D ≤ C) :
  D = 25 :=
sorry

end joe_lowest_score_dropped_l30_3099


namespace sequence_inequality_l30_3066

-- Define the problem
theorem sequence_inequality (a : ℕ → ℕ) (h0 : ∀ n, 0 < a n) (h1 : a 1 > a 0) (h2 : ∀ n ≥ 2, a n = 3 * a (n-1) - 2 * a (n-2)) : a 100 > 2^99 :=
by
  sorry

end sequence_inequality_l30_3066


namespace min_value_of_2a_plus_3b_l30_3027

theorem min_value_of_2a_plus_3b
  (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_perpendicular : (x - (2 * b - 3) * y + 6 = 0) ∧ (2 * b * x + a * y - 5 = 0)) :
  2 * a + 3 * b = 25 / 2 :=
sorry

end min_value_of_2a_plus_3b_l30_3027


namespace ratio_of_width_to_length_l30_3082

variable {w: ℕ}

theorem ratio_of_width_to_length (w: ℕ) (h1: 2*w + 2*10 = 30) (h2: w = 5) :
  ∃ (x y : ℕ), x = 1 ∧ y = 2 ∧ x.gcd y = 1 ∧ w / 10 = 1 / 2 := 
by
  sorry

end ratio_of_width_to_length_l30_3082


namespace number_of_yellow_marbles_l30_3078

theorem number_of_yellow_marbles 
  (total_marbles : ℕ) 
  (red_marbles : ℕ) 
  (blue_marbles : ℕ) 
  (yellow_marbles : ℕ)
  (h1 : total_marbles = 85) 
  (h2 : red_marbles = 14) 
  (h3 : blue_marbles = 3 * red_marbles) 
  (h4 : yellow_marbles = total_marbles - (red_marbles + blue_marbles)) :
  yellow_marbles = 29 :=
  sorry

end number_of_yellow_marbles_l30_3078


namespace legs_in_room_l30_3055

def total_legs_in_room (tables4 : Nat) (sofa : Nat) (chairs4 : Nat) (tables3 : Nat) (table1 : Nat) (rocking_chair2 : Nat) : Nat :=
  (tables4 * 4) + (sofa * 4) + (chairs4 * 4) + (tables3 * 3) + (table1 * 1) + (rocking_chair2 * 2)

theorem legs_in_room :
  total_legs_in_room 4 1 2 3 1 1 = 40 :=
by
  -- Skipping proof steps
  sorry

end legs_in_room_l30_3055


namespace evaluate_expression_l30_3077

theorem evaluate_expression : ∀ (a b c d : ℤ), 
  a = 3 →
  b = a + 3 →
  c = b - 8 →
  d = a + 5 →
  (a + 2 ≠ 0) →
  (b - 4 ≠ 0) →
  (c + 5 ≠ 0) →
  (d - 3 ≠ 0) →
  ((a + 3) * (b - 2) * (c + 9) * (d + 1) = 1512 * (a + 2) * (b - 4) * (c + 5) * (d - 3)) :=
by
  intros a b c d ha hb hc hd ha2 hb4 hc5 hd3
  sorry

end evaluate_expression_l30_3077


namespace alpha_minus_beta_l30_3042

theorem alpha_minus_beta {α β : ℝ} (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
    (h_cos_alpha : Real.cos α = 2 * Real.sqrt 5 / 5) 
    (h_cos_beta : Real.cos β = Real.sqrt 10 / 10) : 
    α - β = -π / 4 := 
sorry

end alpha_minus_beta_l30_3042


namespace minimize_expression_l30_3011

theorem minimize_expression (a b c d e f : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) (h_sum : a + b + c + d + e + f = 10) :
  (1 / a + 9 / b + 25 / c + 49 / d + 81 / e + 121 / f) ≥ 129.6 :=
by
  sorry

end minimize_expression_l30_3011


namespace sum_of_powers_of_i_l30_3095

open Complex

def i := Complex.I

theorem sum_of_powers_of_i : (i + i^2 + i^3 + i^4) = 0 := 
by
  sorry

end sum_of_powers_of_i_l30_3095


namespace total_meters_examined_l30_3074

theorem total_meters_examined (total_meters : ℝ) (h : 0.10 * total_meters = 12) :
  total_meters = 120 :=
sorry

end total_meters_examined_l30_3074


namespace andrena_has_more_dolls_than_debelyn_l30_3089

-- Define the initial number of dolls
def initial_dolls_Debelyn : ℕ := 20
def initial_dolls_Christel : ℕ := 24

-- Define the number of dolls given to Andrena
def dolls_given_by_Debelyn : ℕ := 2
def dolls_given_by_Christel : ℕ := 5

-- Define the condition that Andrena has 2 more dolls than Christel after receiving the dolls
def andrena_more_than_christel : ℕ := 2

-- Define the dolls count after gift exchange
def dolls_Debelyn_after : ℕ := initial_dolls_Debelyn - dolls_given_by_Debelyn
def dolls_Christel_after : ℕ := initial_dolls_Christel - dolls_given_by_Christel
def dolls_Andrena_after : ℕ := dolls_Christel_after + andrena_more_than_christel

-- Define the proof problem
theorem andrena_has_more_dolls_than_debelyn : dolls_Andrena_after - dolls_Debelyn_after = 3 := by
  sorry

end andrena_has_more_dolls_than_debelyn_l30_3089


namespace valid_words_count_l30_3019

noncomputable def count_valid_words : Nat :=
  let total_possible_words : Nat := ((25^1) + (25^2) + (25^3) + (25^4) + (25^5))
  let total_possible_words_without_B : Nat := ((24^1) + (24^2) + (24^3) + (24^4) + (24^5))
  total_possible_words - total_possible_words_without_B

theorem valid_words_count : count_valid_words = 1864701 :=
by
  let total_1_letter_words := 25^1
  let total_2_letter_words := 25^2
  let total_3_letter_words := 25^3
  let total_4_letter_words := 25^4
  let total_5_letter_words := 25^5

  let total_words_without_B_1_letter := 24^1
  let total_words_without_B_2_letter := 24^2
  let total_words_without_B_3_letter := 24^3
  let total_words_without_B_4_letter := 24^4
  let total_words_without_B_5_letter := 24^5

  let valid_1_letter_words := total_1_letter_words - total_words_without_B_1_letter
  let valid_2_letter_words := total_2_letter_words - total_words_without_B_2_letter
  let valid_3_letter_words := total_3_letter_words - total_words_without_B_3_letter
  let valid_4_letter_words := total_4_letter_words - total_words_without_B_4_letter
  let valid_5_letter_words := total_5_letter_words - total_words_without_B_5_letter

  let valid_words := valid_1_letter_words + valid_2_letter_words + valid_3_letter_words + valid_4_letter_words + valid_5_letter_words
  sorry

end valid_words_count_l30_3019


namespace sector_area_l30_3001

theorem sector_area (r : ℝ) (alpha : ℝ) (h : r = 2) (h2 : alpha = π / 3) : 
  1/2 * alpha * r^2 = (2 * π) / 3 := by
  sorry

end sector_area_l30_3001


namespace angle_conversion_l30_3054

theorem angle_conversion :
  (12 * (Real.pi / 180)) = (Real.pi / 15) := by
  sorry

end angle_conversion_l30_3054


namespace acme_vowel_soup_l30_3059

-- Define the set of vowels
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

-- Define the number of each vowel
def num_vowels (v : Char) : ℕ := 5

-- Define a function to count the number of five-letter words
def count_five_letter_words : ℕ :=
  (vowels.card) ^ 5

-- Theorem to be proven
theorem acme_vowel_soup :
  count_five_letter_words = 3125 :=
by
  -- Proof omitted
  sorry

end acme_vowel_soup_l30_3059


namespace extra_apples_l30_3030

-- Defining the given conditions
def redApples : Nat := 60
def greenApples : Nat := 34
def studentsWantFruit : Nat := 7

-- Defining the theorem to prove the number of extra apples
theorem extra_apples : redApples + greenApples - studentsWantFruit = 87 := by
  sorry

end extra_apples_l30_3030


namespace find_x_when_y4_l30_3067

theorem find_x_when_y4 
  (k : ℝ) 
  (h_var : ∀ y : ℝ, ∃ x : ℝ, x = k * y^2)
  (h_initial : ∃ x : ℝ, x = 6 ∧ 1 = k) :
  ∃ x : ℝ, x = 96 :=
by 
  sorry

end find_x_when_y4_l30_3067


namespace largest_divisor_power_of_ten_l30_3024

theorem largest_divisor_power_of_ten (N : ℕ) (m : ℕ) (k : ℕ) 
  (h1 : m ∣ N)
  (h2 : m < N)
  (h3 : N + m = 10^k) : N = 75 := sorry

end largest_divisor_power_of_ten_l30_3024


namespace luke_games_l30_3022

theorem luke_games (F G : ℕ) (H1 : G = 2) (H2 : F + G - 2 = 2) : F = 2 := by
  sorry

end luke_games_l30_3022


namespace geom_prog_min_third_term_l30_3029

theorem geom_prog_min_third_term :
  ∃ (d : ℝ), (-4 + 10 * Real.sqrt 6 = d ∨ -4 - 10 * Real.sqrt 6 = d) ∧
  (∀ x, x = 37 + 2 * d → x ≤ 29 - 20 * Real.sqrt 6) := 
sorry

end geom_prog_min_third_term_l30_3029


namespace largest_value_f12_l30_3010

theorem largest_value_f12 (f : ℝ → ℝ) (hf_poly : ∀ x, f x ≥ 0) 
  (hf_6 : f 6 = 24) (hf_24 : f 24 = 1536) :
  f 12 ≤ 192 :=
sorry

end largest_value_f12_l30_3010


namespace jason_quarters_l30_3043

def quarters_original := 49
def quarters_added := 25
def quarters_total := 74

theorem jason_quarters : quarters_original + quarters_added = quarters_total :=
by
  sorry

end jason_quarters_l30_3043


namespace original_price_is_125_l30_3017

noncomputable def original_price (sold_price : ℝ) (discount_percent : ℝ) : ℝ :=
  sold_price / ((100 - discount_percent) / 100)

theorem original_price_is_125 : original_price 120 4 = 125 :=
by
  sorry

end original_price_is_125_l30_3017


namespace flour_maximum_weight_l30_3069

/-- Given that the bag of flour is marked with 25kg + 50g, prove that the maximum weight of the flour is 25.05kg. -/
theorem flour_maximum_weight :
  let weight_kg := 25
  let weight_g := 50
  (weight_kg + (weight_g / 1000 : ℝ)) = 25.05 :=
by 
  -- provide definitions
  let weight_kg := 25
  let weight_g := 50
  have : (weight_kg + (weight_g / 1000 : ℝ)) = 25.05 := sorry
  exact this

end flour_maximum_weight_l30_3069


namespace c_rent_share_l30_3034

-- Definitions based on conditions
def a_oxen := 10
def a_months := 7
def b_oxen := 12
def b_months := 5
def c_oxen := 15
def c_months := 3
def total_rent := 105

-- Calculate the shares in ox-months
def share_a := a_oxen * a_months
def share_b := b_oxen * b_months
def share_c := c_oxen * c_months

-- Calculate the total ox-months
def total_ox_months := share_a + share_b + share_c

-- Calculate the rent per ox-month
def rent_per_ox_month := total_rent / total_ox_months

-- Calculate the amount C should pay
def amount_c_should_pay := share_c * rent_per_ox_month

-- Prove the statement
theorem c_rent_share : amount_c_should_pay = 27 := by
  sorry

end c_rent_share_l30_3034


namespace point_A_in_fourth_quadrant_l30_3080

-- Defining the coordinates of point A
def x_A : ℝ := 2
def y_A : ℝ := -3

-- Defining the property of the quadrant
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Proposition stating point A is in the fourth quadrant
theorem point_A_in_fourth_quadrant : in_fourth_quadrant x_A y_A :=
by
  sorry

end point_A_in_fourth_quadrant_l30_3080


namespace garden_ratio_2_l30_3057

theorem garden_ratio_2 :
  ∃ (P C k R : ℤ), 
      P = 237 ∧ 
      C = P - 60 ∧ 
      P + C + k = 768 ∧ 
      R = k / C ∧ 
      R = 2 := 
by
  sorry

end garden_ratio_2_l30_3057


namespace find_socks_cost_l30_3085

variable (S : ℝ)
variable (socks_cost : ℝ := 9.5)
variable (shoe_cost : ℝ := 92)
variable (jack_has : ℝ := 40)
variable (needs_more : ℝ := 71)
variable (total_funds : ℝ := jack_has + needs_more)

theorem find_socks_cost (h : 2 * S + shoe_cost = total_funds) : S = socks_cost :=
by 
  sorry

end find_socks_cost_l30_3085


namespace total_grapes_is_157_l30_3036

def number_of_grapes_in_robs_bowl : ℕ := 25

def number_of_grapes_in_allies_bowl : ℕ :=
  number_of_grapes_in_robs_bowl + 5

def number_of_grapes_in_allyns_bowl : ℕ :=
  2 * number_of_grapes_in_allies_bowl - 2

def number_of_grapes_in_sams_bowl : ℕ :=
  (number_of_grapes_in_allies_bowl + number_of_grapes_in_allyns_bowl) / 2

def total_number_of_grapes : ℕ :=
  number_of_grapes_in_robs_bowl +
  number_of_grapes_in_allies_bowl +
  number_of_grapes_in_allyns_bowl +
  number_of_grapes_in_sams_bowl

theorem total_grapes_is_157 : total_number_of_grapes = 157 :=
  sorry

end total_grapes_is_157_l30_3036


namespace find_least_positive_x_l30_3005

theorem find_least_positive_x :
  ∃ x : ℕ, x + 5419 ≡ 3789 [MOD 15] ∧ x = 5 :=
by
  use 5
  constructor
  · sorry
  · rfl

end find_least_positive_x_l30_3005


namespace vincent_books_cost_l30_3093

theorem vincent_books_cost :
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  cost_per_book = 16 :=
by
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  show cost_per_book = 16
  sorry

end vincent_books_cost_l30_3093


namespace range_of_x_l30_3007

variable (x : ℝ)

def p := x^2 - 4 * x + 3 < 0
def q := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

theorem range_of_x : ¬ (p x ∧ q x) ∧ (p x ∨ q x) → (1 < x ∧ x ≤ 2) ∨ x = 3 :=
by 
  sorry

end range_of_x_l30_3007


namespace number_with_at_least_two_zeros_l30_3046

-- A 6-digit number can have for its leftmost digit anything from 1 to 9 inclusive,
-- and for each of its next five digits anything from 0 through 9 inclusive.
def total_6_digit_numbers : ℕ := 9 * 10^5

-- A 6-digit number with no zeros consists solely of digits from 1 to 9
def no_zero : ℕ := 9^6

-- A 6-digit number with exactly one zero
def exactly_one_zero : ℕ := 5 * 9^5

-- The number of 6-digit numbers with less than two zeros is the sum of no_zero and exactly_one_zero
def less_than_two_zeros : ℕ := no_zero + exactly_one_zero

-- The number of 6-digit numbers with at least two zeros is the difference between total_6_digit_numbers and less_than_two_zeros
def at_least_two_zeros : ℕ := total_6_digit_numbers - less_than_two_zeros

-- The theorem that states the number of 6-digit numbers with at least two zeros is 73,314
theorem number_with_at_least_two_zeros : at_least_two_zeros = 73314 := 
by
  sorry

end number_with_at_least_two_zeros_l30_3046


namespace probability_of_one_each_color_is_two_fifths_l30_3081

/-- Definition for marbles bag containing 2 red, 2 blue, and 2 green marbles -/
structure MarblesBag where
  red : ℕ
  blue : ℕ
  green : ℕ
  total : ℕ := red + blue + green

/-- Initial setup for the problem -/
def initialBag : MarblesBag := { red := 2, blue := 2, green := 2 }

/-- Represents the outcome of selecting marbles without replacement -/
def selectMarbles (bag : MarblesBag) (count : ℕ) : ℕ :=
  Nat.choose bag.total count

/-- The number of ways to select one marble of each color -/
def selectOneOfEachColor (bag : MarblesBag) : ℕ :=
  bag.red * bag.blue * bag.green

/-- Calculate the probability of selecting one marble of each color -/
def probabilityOneOfEachColor (bag : MarblesBag) (selectCount : ℕ) : ℚ :=
  selectOneOfEachColor bag / selectMarbles bag selectCount

/-- Theorem stating the answer to the probability problem -/
theorem probability_of_one_each_color_is_two_fifths (bag : MarblesBag) :
  probabilityOneOfEachColor bag 3 = 2 / 5 := by
  sorry

end probability_of_one_each_color_is_two_fifths_l30_3081


namespace friend_selling_price_l30_3073

-- Define the conditions
def CP : ℝ := 51136.36
def loss_percent : ℝ := 0.12
def gain_percent : ℝ := 0.20

-- Define the selling prices SP1 and SP2
def SP1 := CP * (1 - loss_percent)
def SP2 := SP1 * (1 + gain_percent)

-- State the theorem
theorem friend_selling_price : SP2 = 54000 := 
by sorry

end friend_selling_price_l30_3073


namespace tangent_line_at_one_l30_3079

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + Real.log x

theorem tangent_line_at_one (a : ℝ)
  (h : ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → |(f a x - f a 1) / (x - 1) - 3| < ε) :
  ∃ m b, m = 3 ∧ b = -2 ∧ (∀ x y, y = f a x → m * x = y + b) := sorry

end tangent_line_at_one_l30_3079


namespace David_marks_in_Chemistry_l30_3098

theorem David_marks_in_Chemistry (e m p b avg c : ℕ) 
  (h1 : e = 91) 
  (h2 : m = 65) 
  (h3 : p = 82) 
  (h4 : b = 85) 
  (h5 : avg = 78) 
  (h6 : avg * 5 = e + m + p + b + c) :
  c = 67 := 
sorry

end David_marks_in_Chemistry_l30_3098


namespace min_spiders_sufficient_spiders_l30_3002

def grid_size : ℕ := 2019

noncomputable def min_k_catch (k : ℕ) : Prop :=
∀ (fly spider1 spider2 : ℕ × ℕ) (fly_move spider1_move spider2_move: ℕ × ℕ → ℕ × ℕ), 
  (fly_move fly = fly ∨ fly_move fly = (fly.1 + 1, fly.2) ∨ fly_move fly = (fly.1 - 1, fly.2)
  ∨ fly_move fly = (fly.1, fly.2 + 1) ∨ fly_move fly = (fly.1, fly.2 - 1))
  ∧ (spider1_move spider1 = spider1 ∨ spider1_move spider1 = (spider1.1 + 1, spider1.2) ∨ spider1_move spider1 = (spider1.1 - 1, spider1.2)
  ∨ spider1_move spider1 = (spider1.1, spider1.2 + 1) ∨ spider1_move spider1 = (spider1.1, spider1.2 - 1))
  ∧ (spider2_move spider2 = spider2 ∨ spider2_move spider2 = (spider2.1 + 1, spider2.2) ∨ spider2_move spider2 = (spider2.1 - 1, spider2.2)
  ∨ spider2_move spider2 = (spider2.1, spider2.2 + 1) ∨ spider2_move spider2 = (spider2.1, spider2.2 - 1))
  → (spider1 = fly ∨ spider2 = fly)

theorem min_spiders (k : ℕ) : min_k_catch k → k ≥ 2 :=
sorry

theorem sufficient_spiders : min_k_catch 2 :=
sorry

end min_spiders_sufficient_spiders_l30_3002


namespace sum_of_first_five_terms_is_31_l30_3075

variable (a : ℕ → ℝ) (q : ℝ)

-- The geometric sequence a_n
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Condition 1: a_2 * a_3 = 2 * a_1
def condition1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 3 = 2 * a 1

-- Condition 2: The arithmetic mean of a_4 and 2 * a_7 is 5/4
def condition2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 4 + 2 * a 7) / 2 = 5 / 4

-- Sum of the first 5 terms of the geometric sequence
def S_5 (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4

-- The theorem to prove
theorem sum_of_first_five_terms_is_31 (a : ℕ → ℝ) (q : ℝ) 
  (hg : is_geometric_sequence a q) 
  (hc1 : condition1 a q) 
  (hc2 : condition2 a q) : 
  S_5 a = 31 := by
  sorry

end sum_of_first_five_terms_is_31_l30_3075


namespace box_dimensions_l30_3037

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) : 
  a = 5 ∧ b = 8 ∧ c = 12 := 
by
  sorry

end box_dimensions_l30_3037


namespace tank_depth_is_six_l30_3062

-- Definitions derived from the conditions
def tank_length : ℝ := 25
def tank_width : ℝ := 12
def plastering_cost_per_sq_meter : ℝ := 0.45
def total_cost : ℝ := 334.8

-- Compute the surface area to be plastered
def surface_area (d : ℝ) : ℝ := (tank_length * tank_width) + 2 * (tank_length * d) + 2 * (tank_width * d)

-- Equation relating the plastering cost to the surface area
def cost_equation (d : ℝ) : ℝ := plastering_cost_per_sq_meter * (surface_area d)

-- The mathematical result we need to prove
theorem tank_depth_is_six : ∃ d : ℝ, cost_equation d = total_cost ∧ d = 6 := by
  sorry

end tank_depth_is_six_l30_3062


namespace weight_of_new_student_l30_3026

-- Define some constants for the problem
def avg_weight_29_students : ℝ := 28
def number_of_students_29 : ℕ := 29
def new_avg_weight_30_students : ℝ := 27.5
def number_of_students_30 : ℕ := 30

-- Calculate total weights
def total_weight_29_students : ℝ := avg_weight_29_students * number_of_students_29
def new_total_weight_30_students : ℝ := new_avg_weight_30_students * number_of_students_30

-- The proposition we need to prove
theorem weight_of_new_student :
  new_total_weight_30_students - total_weight_29_students = 13 := by
  -- Placeholder for the actual proof
  sorry

end weight_of_new_student_l30_3026


namespace butter_remaining_correct_l30_3087

-- Definitions of the conditions
def cupsOfBakingMix : ℕ := 6
def butterPerCup : ℕ := 2
def substituteRatio : ℕ := 1
def coconutOilUsed : ℕ := 8

-- Calculation based on the conditions
def butterNeeded : ℕ := butterPerCup * cupsOfBakingMix
def butterReplaced : ℕ := coconutOilUsed * substituteRatio
def butterRemaining : ℕ := butterNeeded - butterReplaced

-- The theorem to prove the chef has 4 ounces of butter remaining
theorem butter_remaining_correct : butterRemaining = 4 := 
by
  -- Note: We insert 'sorry' since the proof itself is not required.
  sorry

end butter_remaining_correct_l30_3087


namespace decreasing_f_range_l30_3023

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log x - 2 * k * x - 1

theorem decreasing_f_range (k : ℝ) (x₁ x₂ : ℝ) (h₁ : 2 ≤ x₁) (h₂ : x₁ < x₂) (h₃ : x₂ ≤ 4) :
  k ≥ 1 / 4 → (x₁ - x₂) * (f x₁ k - f x₂ k) < 0 :=
sorry

end decreasing_f_range_l30_3023


namespace total_banana_produce_correct_l30_3033

-- Defining the conditions as variables and constants
def B_nearby : ℕ := 9000
def B_Jakies : ℕ := 10 * B_nearby
def T : ℕ := B_nearby + B_Jakies

-- Theorem statement
theorem total_banana_produce_correct : T = 99000 := by
  sorry  -- Proof placeholder

end total_banana_produce_correct_l30_3033


namespace dvd_packs_l30_3063

theorem dvd_packs (cost_per_pack : ℕ) (discount_per_pack : ℕ) (money_available : ℕ) 
  (h_cost : cost_per_pack = 107) 
  (h_discount : discount_per_pack = 106) 
  (h_money : money_available = 93) : 
  (money_available / (cost_per_pack - discount_per_pack)) = 93 := 
by 
  -- Implementation of the proof goes here
  sorry

end dvd_packs_l30_3063


namespace max_combinations_for_n_20_l30_3015

def num_combinations (s n k : ℕ) : ℕ :=
if n = 0 then if s = 0 then 1 else 0
else if s < n then 0
else if k = 0 then 0
else num_combinations (s - k) (n - 1) (k - 1) + num_combinations s n (k - 1)

theorem max_combinations_for_n_20 : ∀ s k, s = 20 ∧ k = 9 → num_combinations s 4 k = 12 :=
by
  intros s k h
  cases h
  sorry

end max_combinations_for_n_20_l30_3015


namespace kite_perimeter_l30_3056

-- Given the kite's diagonals, shorter sides, and longer sides
def diagonals : ℕ × ℕ := (12, 30)
def shorter_sides : ℕ := 10
def longer_sides : ℕ := 15

-- Problem statement: Prove that the perimeter is 50 inches
theorem kite_perimeter (diag1 diag2 short_len long_len : ℕ) 
                       (h_diag : diag1 = 12 ∧ diag2 = 30)
                       (h_short : short_len = 10)
                       (h_long : long_len = 15) : 
                       2 * short_len + 2 * long_len = 50 :=
by
  -- We provide no proof, only the statement
  sorry

end kite_perimeter_l30_3056


namespace roots_square_difference_l30_3035

theorem roots_square_difference (a b : ℚ)
  (ha : 6 * a^2 + 13 * a - 28 = 0)
  (hb : 6 * b^2 + 13 * b - 28 = 0) : (a - b)^2 = 841 / 36 :=
sorry

end roots_square_difference_l30_3035


namespace dice_probability_sum_12_l30_3039

open Nat

/-- Probability that the sum of three six-faced dice rolls equals 12 is 10 / 216 --/
theorem dice_probability_sum_12 : 
  let outcomes := 6^3
  let favorable := 10
  (favorable : ℚ) / outcomes = 10 / 216 := 
by
  let outcomes := 6^3
  let favorable := 10
  sorry

end dice_probability_sum_12_l30_3039


namespace volunteers_meet_again_in_360_days_l30_3004

-- Definitions of the given values for the problem
def ella_days := 5
def fiona_days := 6
def george_days := 8
def harry_days := 9

-- Statement of the problem in Lean 4
theorem volunteers_meet_again_in_360_days :
  Nat.lcm (Nat.lcm (Nat.lcm ella_days fiona_days) george_days) harry_days = 360 :=
by
  sorry

end volunteers_meet_again_in_360_days_l30_3004


namespace quadratic_inequality_solution_l30_3020

theorem quadratic_inequality_solution (x : ℝ) :
  (x < -7 ∨ x > 3) → x^2 + 4 * x - 21 > 0 :=
by
  -- The proof will go here
  sorry

end quadratic_inequality_solution_l30_3020


namespace missing_angle_correct_l30_3097

theorem missing_angle_correct (n : ℕ) (h1 : n ≥ 3) (angles_sum : ℕ) (h2 : angles_sum = 2017) 
    (sum_interior_angles : ℕ) (h3 : sum_interior_angles = 180 * (n - 2)) :
    (sum_interior_angles - angles_sum) = 143 :=
by
  sorry

end missing_angle_correct_l30_3097


namespace compute_expression_l30_3071

theorem compute_expression : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end compute_expression_l30_3071


namespace infinite_solutions_b_value_l30_3050

-- Given condition for the equation to hold
def equation_condition (x b : ℤ) : Prop :=
  4 * (3 * x - b) = 3 * (4 * x + 16)

-- The statement we need to prove: b = -12
theorem infinite_solutions_b_value :
  (∀ x : ℤ, equation_condition x b) → b = -12 :=
sorry

end infinite_solutions_b_value_l30_3050


namespace probability_of_picking_letter_in_mathematics_l30_3000

-- Definitions and conditions
def total_letters : ℕ := 26
def unique_letters_in_mathematics : ℕ := 8
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Theorem to be proven
theorem probability_of_picking_letter_in_mathematics :
  probability unique_letters_in_mathematics total_letters = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l30_3000


namespace min_value_reciprocal_sum_l30_3051

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 1) (h2 : 0 < m) (h3 : 0 < n) : 
  (1/m + 1/n) = 4 :=
by
  sorry

end min_value_reciprocal_sum_l30_3051


namespace cliff_rock_collection_l30_3058

theorem cliff_rock_collection (S I : ℕ) 
  (h1 : I = S / 2) 
  (h2 : 2 * I / 3 = 40) : S + I = 180 := by
  sorry

end cliff_rock_collection_l30_3058


namespace dice_sum_probability_l30_3072

-- Define a noncomputable function to calculate the number of ways to get a sum of 15
noncomputable def dice_sum_ways (dices : ℕ) (sides : ℕ) (target_sum : ℕ) : ℕ :=
  sorry

-- Define the Lean 4 statement
theorem dice_sum_probability :
  dice_sum_ways 5 6 15 = 2002 :=
sorry

end dice_sum_probability_l30_3072


namespace amount_of_flour_per_new_bread_roll_l30_3032

theorem amount_of_flour_per_new_bread_roll :
  (24 * (1 / 8) = 3) → (16 * f = 3) → (f = 3 / 16) :=
by
  intro h1 h2
  sorry

end amount_of_flour_per_new_bread_roll_l30_3032


namespace ratio_is_five_to_three_l30_3053

variable (g b : ℕ)

def girls_more_than_boys : Prop := g - b = 6
def total_pupils : Prop := g + b = 24
def ratio_girls_to_boys : ℚ := g / b

theorem ratio_is_five_to_three (h1 : girls_more_than_boys g b) (h2 : total_pupils g b) : ratio_girls_to_boys g b = 5 / 3 := by
  sorry

end ratio_is_five_to_three_l30_3053


namespace express_w_l30_3049

theorem express_w (w a b c : ℝ) (x y z : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ w ∧ b ≠ w ∧ c ≠ w)
  (h1 : x + y + z = 1)
  (h2 : x * a^2 + y * b^2 + z * c^2 = w^2)
  (h3 : x * a^3 + y * b^3 + z * c^3 = w^3)
  (h4 : x * a^4 + y * b^4 + z * c^4 = w^4) :
  w = - (a * b * c) / (a * b + b * c + c * a) :=
sorry

end express_w_l30_3049


namespace sum_of_two_numbers_l30_3008

variable {x y : ℝ}

theorem sum_of_two_numbers (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end sum_of_two_numbers_l30_3008


namespace area_triangle_PTS_l30_3006

theorem area_triangle_PTS {PQ QR PS QT PT TS : ℝ} 
  (hPQ : PQ = 4) 
  (hQR : QR = 6) 
  (hPS : PS = 2 * Real.sqrt 13) 
  (hQT : QT = 12 * Real.sqrt 13 / 13) 
  (hPT : PT = 4) 
  (hTS : TS = (2 * Real.sqrt 13) - 4) : 
  (1 / 2) * PT * QT = 24 * Real.sqrt 13 / 13 := 
by 
  sorry

end area_triangle_PTS_l30_3006


namespace car_speed_decrease_l30_3031

theorem car_speed_decrease (d : ℝ) (speed_first : ℝ) (distance_fifth : ℝ) (time_interval : ℝ) :
  speed_first = 45 ∧ distance_fifth = 4.4 ∧ time_interval = 8 / 60 ∧ speed_first - 4 * d = distance_fifth / time_interval -> d = 3 :=
by
  intros h
  obtain ⟨_, _, _, h_eq⟩ := h
  sorry

end car_speed_decrease_l30_3031


namespace point_of_tangency_l30_3041

def parabola1 (x y : ℝ) : Prop := y = x^2 + 15*x + 32
def parabola2 (x y : ℝ) : Prop := x = y^2 + 49*y + 593

theorem point_of_tangency :
  parabola1 (-7) (-24) ∧ parabola2 (-7) (-24) := by
  sorry

end point_of_tangency_l30_3041


namespace tile_ratio_l30_3090

-- Definitions corresponding to the conditions in the problem
def orig_grid_size : ℕ := 6
def orig_black_tiles : ℕ := 12
def orig_white_tiles : ℕ := 24
def border_size : ℕ := 1

-- The combined problem statement
theorem tile_ratio (orig_grid_size orig_black_tiles orig_white_tiles border_size : ℕ) :
  let new_grid_size := orig_grid_size + 2 * border_size
  let new_tiles := new_grid_size^2
  let added_tiles := new_tiles - orig_grid_size^2
  let total_white_tiles := orig_white_tiles + added_tiles
  let black_to_white_ratio := orig_black_tiles / total_white_tiles
  black_to_white_ratio = (3 : ℕ) / 13 :=
by {
  sorry
}

end tile_ratio_l30_3090


namespace intersection_points_of_lines_l30_3076

theorem intersection_points_of_lines :
  (∃ (x y : ℚ), 2 * y - 3 * x = 4 ∧ x + 3 * y = 3 ∧ x = 10 / 11 ∧ y = 13 / 11) ∧
  (∃ (x y : ℚ), 2 * y - 3 * x = 4 ∧ 5 * x - 3 * y = 6 ∧ x = 24 ∧ y = 38) :=
by
  sorry

end intersection_points_of_lines_l30_3076


namespace solve_for_q_l30_3052

theorem solve_for_q : 
  let n : ℤ := 63
  let m : ℤ := 14
  ∀ (q : ℤ),
  (7 : ℤ) / 9 = n / 81 ∧
  (7 : ℤ) / 9 = (m + n) / 99 ∧
  (7 : ℤ) / 9 = (q - m) / 135 → 
  q = 119 :=
by
  sorry

end solve_for_q_l30_3052


namespace min_value_of_a_is_five_l30_3044

-- Given: a, b, c in table satisfying the conditions
-- We are to prove that the minimum value of a is 5.
theorem min_value_of_a_is_five
  {a b c: ℤ} (h_pos: 0 < a) (hx_distinct: 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧ 
                               a*x₁^2 + b*x₁ + c = 0 ∧ 
                               a*x₂^2 + b*x₂ + c = 0) (hb_neg: b < 0) 
                               (h_disc_pos: (b^2 - 4*a*c) > 0) : a = 5 :=
sorry

end min_value_of_a_is_five_l30_3044


namespace cistern_fill_time_l30_3045

/--
  A cistern can be filled by tap A in 4 hours,
  emptied by tap B in 6 hours,
  and filled by tap C in 3 hours.
  If all the taps are opened simultaneously,
  then the cistern will be filled in exactly 2.4 hours.
-/
theorem cistern_fill_time :
  let rate_A := 1 / 4
  let rate_B := -1 / 6
  let rate_C := 1 / 3
  let combined_rate := rate_A + rate_B + rate_C
  let fill_time := 1 / combined_rate
  fill_time = 2.4 := by
  sorry

end cistern_fill_time_l30_3045


namespace John_age_l30_3021

theorem John_age (Drew Maya Peter John Jacob : ℕ)
  (h1 : Drew = Maya + 5)
  (h2 : Peter = Drew + 4)
  (h3 : John = 2 * Maya)
  (h4 : (Jacob + 2) * 2 = Peter + 2)
  (h5 : Jacob = 11) : John = 30 :=
by 
  sorry

end John_age_l30_3021


namespace second_bill_late_fee_l30_3048

def first_bill_amount : ℕ := 200
def first_bill_interest_rate : ℝ := 0.10
def first_bill_months : ℕ := 2
def second_bill_amount : ℕ := 130
def second_bill_months : ℕ := 6
def third_bill_first_month_fee : ℕ := 40
def third_bill_second_month_fee : ℕ := 80
def total_amount_owed : ℕ := 1234

theorem second_bill_late_fee (x : ℕ) 
(h : first_bill_amount * (first_bill_interest_rate * first_bill_months) + first_bill_amount + third_bill_first_month_fee + third_bill_second_month_fee + second_bill_amount + second_bill_months * x = total_amount_owed) : x = 124 :=
sorry

end second_bill_late_fee_l30_3048


namespace positive_integer_satisfies_condition_l30_3028

theorem positive_integer_satisfies_condition : 
  ∃ n : ℕ, (12 * n = n^2 + 36) ∧ n = 6 :=
by
  sorry

end positive_integer_satisfies_condition_l30_3028


namespace min_value_l30_3091

theorem min_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + 3 * y + 3 * x * y = 6) : 2 * x + 3 * y ≥ 4 :=
sorry

end min_value_l30_3091


namespace expand_product_l30_3014

variable (x : ℝ)

theorem expand_product :
  (x + 3) * (x^2 + 4 * x + 6) = x^3 + 7 * x^2 + 18 * x + 18 := 
  sorry

end expand_product_l30_3014


namespace triangle_c_and_area_l30_3016

theorem triangle_c_and_area
  (a b : ℝ) (C : ℝ)
  (h_a : a = 1)
  (h_b : b = 2)
  (h_C : C = Real.pi / 3) :
  ∃ (c S : ℝ), c = Real.sqrt 3 ∧ S = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_c_and_area_l30_3016


namespace zero_interval_of_f_l30_3088

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem zero_interval_of_f :
    ∃ c, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  sorry

end zero_interval_of_f_l30_3088


namespace abs_neg_eq_iff_nonpos_l30_3060

theorem abs_neg_eq_iff_nonpos (a : ℝ) : |a| = -a ↔ a ≤ 0 :=
by sorry

end abs_neg_eq_iff_nonpos_l30_3060


namespace polynomial_value_l30_3064

theorem polynomial_value :
  let a := -4
  let b := 23
  let c := -17
  let d := 10
  5 * a + 3 * b + 2 * c + d = 25 :=
by
  let a := -4
  let b := 23
  let c := -17
  let d := 10
  sorry

end polynomial_value_l30_3064


namespace parts_of_a_number_l30_3013

theorem parts_of_a_number 
  (a p q : ℝ) 
  (x y z : ℝ)
  (h1 : y + z = p * x)
  (h2 : x + y = q * z)
  (h3 : x + y + z = a) :
  x = a / (1 + p) ∧ y = a * (p * q - 1) / ((p + 1) * (q + 1)) ∧ z = a / (1 + q) := 
by 
  sorry

end parts_of_a_number_l30_3013


namespace socks_selection_l30_3092

theorem socks_selection :
  ∀ (R Y G B O : ℕ), 
    R = 80 → Y = 70 → G = 50 → B = 60 → O = 40 →
    (∃ k, k = 38 ∧ ∀ (N : ℕ → ℕ), (N R + N Y + N G + N B + N O ≥ k)
          → (exists (pairs : ℕ), pairs ≥ 15 ∧ pairs = (N R / 2) + (N Y / 2) + (N G / 2) + (N B / 2) + (N O / 2) )) :=
by
  sorry

end socks_selection_l30_3092


namespace apple_and_cherry_pies_total_l30_3086

-- Given conditions state that:
def apple_pies : ℕ := 6
def cherry_pies : ℕ := 5

-- We aim to prove that the total number of apple and cherry pies is 11.
theorem apple_and_cherry_pies_total : apple_pies + cherry_pies = 11 := by
  sorry

end apple_and_cherry_pies_total_l30_3086


namespace exists_m_divisible_by_2k_l30_3084

theorem exists_m_divisible_by_2k {k : ℕ} (h_k : 0 < k) {a : ℤ} (h_a : a % 8 = 3) :
  ∃ m : ℕ, 0 < m ∧ 2^k ∣ (a^m + a + 2) :=
sorry

end exists_m_divisible_by_2k_l30_3084


namespace Gwen_avg_speed_trip_l30_3065

theorem Gwen_avg_speed_trip : 
  ∀ (d1 d2 s1 s2 t1 t2 : ℝ), 
  d1 = 40 → d2 = 40 → s1 = 15 → s2 = 30 →
  d1 / s1 = t1 → d2 / s2 = t2 →
  (d1 + d2) / (t1 + t2) = 20 :=
by 
  intros d1 d2 s1 s2 t1 t2 hd1 hd2 hs1 hs2 ht1 ht2
  sorry

end Gwen_avg_speed_trip_l30_3065


namespace total_selling_price_l30_3083

theorem total_selling_price (cost_per_meter profit_per_meter : ℕ) (total_meters : ℕ) :
  cost_per_meter = 90 → 
  profit_per_meter = 15 → 
  total_meters = 85 → 
  (cost_per_meter + profit_per_meter) * total_meters = 8925 :=
by
  intros
  sorry

end total_selling_price_l30_3083


namespace compute_operation_value_l30_3018

def operation (a b c : ℝ) : ℝ := b^3 - 3 * a * b * c - 4 * a * c^2

theorem compute_operation_value : operation 2 (-1) 4 = -105 :=
by
  sorry

end compute_operation_value_l30_3018


namespace unique_intersection_of_line_and_parabola_l30_3070

theorem unique_intersection_of_line_and_parabola :
  ∃! k : ℚ, ∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k → k = 25 / 3 :=
by
  sorry

end unique_intersection_of_line_and_parabola_l30_3070


namespace min_buses_needed_l30_3040

theorem min_buses_needed (students : ℕ) (cap1 cap2 : ℕ) (h_students : students = 530) (h_cap1 : cap1 = 40) (h_cap2 : cap2 = 45) :
  min (Nat.ceil (students / cap1)) (Nat.ceil (students / cap2)) = 12 :=
  sorry

end min_buses_needed_l30_3040


namespace marissas_sunflower_height_in_meters_l30_3025

-- Define the conversion factors
def inches_per_foot : ℝ := 12
def cm_per_inch : ℝ := 2.54
def cm_per_meter : ℝ := 100

-- Define the given data
def sister_height_feet : ℝ := 4.15
def additional_height_cm : ℝ := 37
def height_difference_inches : ℝ := 63

-- Calculate the height of Marissa's sunflower in meters
theorem marissas_sunflower_height_in_meters :
  let sister_height_inches := sister_height_feet * inches_per_foot
  let sister_height_cm := sister_height_inches * cm_per_inch
  let total_sister_height_cm := sister_height_cm + additional_height_cm
  let height_difference_cm := height_difference_inches * cm_per_inch
  let marissas_sunflower_height_cm := total_sister_height_cm + height_difference_cm
  let marissas_sunflower_height_m := marissas_sunflower_height_cm / cm_per_meter
  marissas_sunflower_height_m = 3.23512 :=
by
  sorry

end marissas_sunflower_height_in_meters_l30_3025


namespace adult_ticket_cost_given_conditions_l30_3096

variables (C A S : ℕ)

def cost_relationships : Prop :=
  A = C + 10 ∧ S = A - 5 ∧ (5 * C + 2 * A + 2 * S + (S - 3) = 212)

theorem adult_ticket_cost_given_conditions :
  cost_relationships C A S → A = 28 :=
by
  intros h
  have h1 : A = C + 10 := h.left
  have h2 : S = A - 5 := h.right.left
  have h3 : (5 * C + 2 * A + 2 * S + (S - 3) = 212) := h.right.right
  sorry

end adult_ticket_cost_given_conditions_l30_3096


namespace simplify_evaluate_l30_3047

noncomputable def a := (1 / 2) + Real.sqrt (1 / 2)

theorem simplify_evaluate (a : ℝ) (h : a = (1 / 2) + Real.sqrt (1 / 2)) :
  (a + Real.sqrt 3) * (a - Real.sqrt 3) - a * (a - 6) = 3 * Real.sqrt 2 :=
by sorry

end simplify_evaluate_l30_3047


namespace prisoners_can_be_freed_l30_3094

-- Condition: We have 100 prisoners and 100 drawers.
def prisoners : Nat := 100
def drawers : Nat := 100

-- Predicate to represent the strategy
def successful_strategy (strategy: (Fin prisoners) → (Fin drawers) → Bool) : Bool :=
  -- We use a hypothetical strategy function to model this
  (true) -- Placeholder for the actual strategy computation

-- Statement: Prove that there exists a strategy where all prisoners finding their names has a probability greater than 30%.
theorem prisoners_can_be_freed :
  ∃ strategy: (Fin prisoners) → (Fin drawers) → Bool, 
    (successful_strategy strategy) ∧ (0.3118 > 0.3) :=
sorry

end prisoners_can_be_freed_l30_3094


namespace sequence_general_term_l30_3061

theorem sequence_general_term {a : ℕ → ℚ} 
  (h₀ : a 1 = 1) 
  (h₁ : ∀ n ≥ 2, a n = 3 * a (n - 1) / (a (n - 1) + 3)) : 
  ∀ n, a n = 3 / (n + 2) :=
by
  sorry

end sequence_general_term_l30_3061


namespace find_a_l30_3038

noncomputable def f (t : ℝ) (a : ℝ) : ℝ := (1 / (Real.cos t)) + (a / (1 - (Real.cos t)))

theorem find_a (t : ℝ) (a : ℝ) (h1 : 0 < t) (h2 : t < (Real.pi / 2)) (h3 : 0 < a) (h4 : ∀ t, 0 < t ∧ t < (Real.pi / 2) → f t a = 16) :
  a = 9 :=
sorry

end find_a_l30_3038


namespace abc_product_l30_3068

theorem abc_product (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b = 13) (h2 : b * c = 52) (h3 : c * a = 4) : a * b * c = 52 := 
  sorry

end abc_product_l30_3068


namespace train_lengths_combined_l30_3009

noncomputable def speed_to_mps (kmph : ℤ) : ℚ := (kmph : ℚ) * 5 / 18

def length_of_train (speed : ℚ) (time : ℚ) : ℚ := speed * time

theorem train_lengths_combined :
  let speed1_kmph := 100
  let speed2_kmph := 120
  let time1_sec := 9
  let time2_sec := 8
  let speed1_mps := speed_to_mps speed1_kmph
  let speed2_mps := speed_to_mps speed2_kmph
  let length1 := length_of_train speed1_mps time1_sec
  let length2 := length_of_train speed2_mps time2_sec
  length1 + length2 = 516.66 :=
by
  sorry

end train_lengths_combined_l30_3009


namespace exists_nat_pair_l30_3003

theorem exists_nat_pair 
  (k : ℕ) : 
  let a := 2 * k
  let b := 2 * k * k + 2 * k + 1
  (b - 1) % (a + 1) = 0 ∧ (a * a + a + 2) % b = 0 := by
  sorry

end exists_nat_pair_l30_3003


namespace least_n_l30_3012

theorem least_n (n : ℕ) (h : (1 : ℝ) / n - (1 / (n + 1)) < (1 / 15)) : n = 4 :=
sorry

end least_n_l30_3012
