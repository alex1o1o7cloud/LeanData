import Mathlib

namespace value_of_s_l204_20487

-- Define the variables as integers (they represent non-zero digits)
variables {a p v e s r : ℕ}

-- Define the conditions as hypotheses
theorem value_of_s (h1 : a + p = v) (h2 : v + e = s) (h3 : s + a = r) (h4 : p + e + r = 14) :
  s = 7 :=
by
  sorry

end value_of_s_l204_20487


namespace hallie_reads_121_pages_on_fifth_day_l204_20414

-- Definitions for the given conditions.
def book_length : ℕ := 480
def pages_day_one : ℕ := 63
def pages_day_two : ℕ := 95 -- Rounded from 94.5
def pages_day_three : ℕ := 115
def pages_day_four : ℕ := 86 -- Rounded from 86.25

-- Total pages read from day one to day four
def pages_read_first_four_days : ℕ :=
  pages_day_one + pages_day_two + pages_day_three + pages_day_four

-- Conclusion: the number of pages read on the fifth day.
def pages_day_five : ℕ := book_length - pages_read_first_four_days

-- Proof statement: Hallie reads 121 pages on the fifth day.
theorem hallie_reads_121_pages_on_fifth_day :
  pages_day_five = 121 :=
by
  -- Proof omitted
  sorry

end hallie_reads_121_pages_on_fifth_day_l204_20414


namespace divide_milk_into_equal_parts_l204_20490

def initial_state : (ℕ × ℕ × ℕ) := (8, 0, 0)

def is_equal_split (state : ℕ × ℕ × ℕ) : Prop :=
  state.1 = 4 ∧ state.2 = 4

theorem divide_milk_into_equal_parts : 
  ∃ (state_steps : Fin 25 → ℕ × ℕ × ℕ),
  initial_state = state_steps 0 ∧
  is_equal_split (state_steps 24) :=
sorry

end divide_milk_into_equal_parts_l204_20490


namespace problem_statement_l204_20483

theorem problem_statement (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : 2 ≤ n) : 
  (1 + x)^n + (1 - x)^n < 2^n :=
sorry

end problem_statement_l204_20483


namespace ellipse_focus_value_l204_20441

theorem ellipse_focus_value (m : ℝ) (h1 : m > 0) :
  (∃ (x y : ℝ), (x, y) = (-4, 0) ∧ (25 - m^2 = 16)) → m = 3 :=
by
  sorry

end ellipse_focus_value_l204_20441


namespace total_cost_l204_20450

def cost_of_items (x y : ℝ) : Prop :=
  (6 * x + 5 * y = 6.10) ∧ (3 * x + 4 * y = 4.60)

theorem total_cost (x y : ℝ) (h : cost_of_items x y) : 12 * x + 8 * y = 10.16 :=
by
  sorry

end total_cost_l204_20450


namespace problem_statement_l204_20494

theorem problem_statement (x : ℝ) (h1 : x = 3 ∨ x = -3) : 6 * x^2 + 4 * x - 2 * (x^2 - 1) - 2 * (2 * x + x^2) = 20 := 
by {
  sorry
}

end problem_statement_l204_20494


namespace peaches_left_in_baskets_l204_20457

theorem peaches_left_in_baskets :
  let initial_baskets := 5
  let initial_peaches_per_basket := 20
  let new_baskets := 4
  let new_peaches_per_basket := 25
  let peaches_removed_per_basket := 10

  let total_initial_peaches := initial_baskets * initial_peaches_per_basket
  let total_new_peaches := new_baskets * new_peaches_per_basket
  let total_peaches_before_removal := total_initial_peaches + total_new_peaches

  let total_baskets := initial_baskets + new_baskets
  let total_peaches_removed := total_baskets * peaches_removed_per_basket
  let peaches_left := total_peaches_before_removal - total_peaches_removed

  peaches_left = 110 := by
  sorry

end peaches_left_in_baskets_l204_20457


namespace true_weight_of_C_l204_20427

theorem true_weight_of_C (A1 B1 C1 A2 B2 : ℝ) (l1 l2 m1 m2 A B C : ℝ)
  (hA1 : (A + m1) * l1 = (A1 + m2) * l2)
  (hB1 : (B + m1) * l1 = (B1 + m2) * l2)
  (hC1 : (C + m1) * l1 = (C1 + m2) * l2)
  (hA2 : (A2 + m1) * l1 = (A + m2) * l2)
  (hB2 : (B2 + m1) * l1 = (B + m2) * l2) :
  C = (C1 - A1) * Real.sqrt ((A2 - B2) / (A1 - B1)) + 
      (A1 * Real.sqrt (A2 - B2) + A2 * Real.sqrt (A1 - B1)) / 
      (Real.sqrt (A1 - B1) + Real.sqrt (A2 - B2)) :=
sorry

end true_weight_of_C_l204_20427


namespace calculate_crayons_lost_l204_20447

def initial_crayons := 440
def given_crayons := 111
def final_crayons := 223

def crayons_left_after_giving := initial_crayons - given_crayons
def crayons_lost := crayons_left_after_giving - final_crayons

theorem calculate_crayons_lost : crayons_lost = 106 :=
  by
    sorry

end calculate_crayons_lost_l204_20447


namespace probability_point_between_lines_l204_20478

theorem probability_point_between_lines :
  let l (x : ℝ) := -2 * x + 8
  let m (x : ℝ) := -3 * x + 9
  let area_l := 1 / 2 * 4 * 8
  let area_m := 1 / 2 * 3 * 9
  let area_between := area_l - area_m
  let probability := area_between / area_l
  probability = 0.16 :=
by
  sorry

end probability_point_between_lines_l204_20478


namespace determine_a_plus_b_l204_20425

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b
noncomputable def f_inv (a b x : ℝ) : ℝ := b * x^2 + a

theorem determine_a_plus_b (a b : ℝ) (h: ∀ x : ℝ, f a b (f_inv a b x) = x) : a + b = 1 :=
sorry

end determine_a_plus_b_l204_20425


namespace initial_boxes_l204_20428

theorem initial_boxes (x : ℕ) (h1 : 80 + 165 = 245) (h2 : 2000 * 245 = 490000) 
                      (h3 : 4 * 245 * x + 245 * x = 1225 * x) : x = 400 :=
by
  sorry

end initial_boxes_l204_20428


namespace t_shirt_price_increase_t_shirt_max_profit_l204_20495

theorem t_shirt_price_increase (x : ℝ) : (x + 10) * (300 - 10 * x) = 3360 → x = 2 := 
by 
  sorry

theorem t_shirt_max_profit (x : ℝ) : (-10 * x^2 + 200 * x + 3000) = 4000 ↔ x = 10 := 
by 
  sorry

end t_shirt_price_increase_t_shirt_max_profit_l204_20495


namespace sin_double_angle_l204_20472

theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 := by
  sorry

end sin_double_angle_l204_20472


namespace earnings_difference_l204_20418

-- Definitions:
def investments_ratio := (3, 4, 5)
def return_ratio := (6, 5, 4)
def total_earnings := 5800

-- Target statement:
theorem earnings_difference (x y : ℝ)
  (h_investment_ratio : investments_ratio = (3, 4, 5))
  (h_return_ratio : return_ratio = (6, 5, 4))
  (h_total_earnings : (3 * x * 6 * y) / 100 + (4 * x * 5 * y) / 100 + (5 * x * 4 * y) / 100 = total_earnings) :
  ((4 * x * 5 * y) / 100 - (3 * x * 6 * y) / 100) = 200 := 
by
  sorry

end earnings_difference_l204_20418


namespace max_dot_product_on_circle_l204_20480

theorem max_dot_product_on_circle :
  ∀ (P : ℝ × ℝ) (O : ℝ × ℝ) (A : ℝ × ℝ),
  O = (0, 0) →
  A = (-2, 0) →
  P.1 ^ 2 + P.2 ^ 2 = 1 →
  let AO := (2, 0)
  let AP := (P.1 + 2, P.2)
  ∃ α : ℝ, P = (Real.cos α, Real.sin α) ∧ 
  ∃ max_val : ℝ, max_val = 6 ∧ 
  (2 * (Real.cos α + 2) = max_val) :=
by
  intro P O A hO hA hP 
  let AO := (2, 0)
  let AP := (P.1 + 2, P.2)
  sorry

end max_dot_product_on_circle_l204_20480


namespace arithmetic_sequence_n_terms_l204_20466

theorem arithmetic_sequence_n_terms:
  ∀ (a₁ d aₙ n: ℕ), 
  a₁ = 6 → d = 3 → aₙ = 300 → aₙ = a₁ + (n - 1) * d → n = 99 :=
by
  intros a₁ d aₙ n h1 h2 h3 h4
  sorry

end arithmetic_sequence_n_terms_l204_20466


namespace geometric_seq_a5_a7_l204_20458

theorem geometric_seq_a5_a7 (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : a 3 + a 5 = 6)
  (q : ℝ) :
  (a 5 + a 7 = 12) :=
sorry

end geometric_seq_a5_a7_l204_20458


namespace number_of_tiles_l204_20444

noncomputable def tile_count (room_length : ℝ) (room_width : ℝ) (tile_length : ℝ) (tile_width : ℝ) :=
  let room_area := room_length * room_width
  let tile_area := tile_length * tile_width
  room_area / tile_area

theorem number_of_tiles :
  tile_count 10 15 (1 / 4) (5 / 12) = 1440 := by
  sorry

end number_of_tiles_l204_20444


namespace duration_of_period_l204_20485

/-- The duration of the period at which B gains Rs. 1125 by lending 
Rs. 25000 at rate of 11.5% per annum and borrowing the same 
amount at 10% per annum -/
theorem duration_of_period (principal : ℝ) (rate_borrow : ℝ) (rate_lend : ℝ) (gain : ℝ) : 
  ∃ (t : ℝ), principal = 25000 ∧ rate_borrow = 0.10 ∧ rate_lend = 0.115 ∧ gain = 1125 → 
  t = 3 :=
by
  sorry

end duration_of_period_l204_20485


namespace slices_per_pizza_l204_20409

-- Definitions based on the conditions
def num_pizzas : Nat := 3
def total_cost : Nat := 72
def cost_per_5_slices : Nat := 10

-- To find the number of slices per pizza
theorem slices_per_pizza (num_pizzas : Nat) (total_cost : Nat) (cost_per_5_slices : Nat): 
  (total_cost / num_pizzas) / (cost_per_5_slices / 5) = 12 :=
by
  sorry

end slices_per_pizza_l204_20409


namespace probability_three_white_two_black_eq_eight_seventeen_l204_20445
-- Import Mathlib library to access combinatorics functions.

-- Define the total number of white and black balls.
def total_white := 8
def total_black := 7

-- The key function to calculate combinations.
noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Define the problem conditions as constants.
def total_balls := total_white + total_black
def chosen_balls := 5
def white_balls_chosen := 3
def black_balls_chosen := 2

-- Calculate number of combinations.
noncomputable def total_combinations : ℕ := choose total_balls chosen_balls
noncomputable def white_combinations : ℕ := choose total_white white_balls_chosen
noncomputable def black_combinations : ℕ := choose total_black black_balls_chosen

-- Calculate the probability as a rational number.
noncomputable def probability_exact_three_white_two_black : ℚ :=
  (white_combinations * black_combinations : ℚ) / total_combinations

-- The theorem we want to prove
theorem probability_three_white_two_black_eq_eight_seventeen :
  probability_exact_three_white_two_black = 8 / 17 := by
  sorry

end probability_three_white_two_black_eq_eight_seventeen_l204_20445


namespace geometric_sequence_a_eq_2_l204_20400

theorem geometric_sequence_a_eq_2 (a : ℝ) (h1 : ¬ a = 0) (h2 : (2 * a) ^ 2 = 8 * a) : a = 2 :=
by {
  sorry -- Proof not required, only the statement.
}

end geometric_sequence_a_eq_2_l204_20400


namespace factorize_expression_l204_20421

theorem factorize_expression (m : ℝ) : 2 * m^2 - 8 = 2 * (m + 2) * (m - 2) :=
sorry

end factorize_expression_l204_20421


namespace find_value_of_expression_l204_20461

theorem find_value_of_expression 
  (x y z w : ℤ)
  (hx : x = 3)
  (hy : y = 2)
  (hz : z = 4)
  (hw : w = -1) :
  x^2 * y - 2 * x * y + 3 * x * z - (x + y) * (y + z) * (z + w) = -48 :=
by
  sorry

end find_value_of_expression_l204_20461


namespace smallest_perfect_cube_divisor_l204_20434

theorem smallest_perfect_cube_divisor 
  (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) 
  (hpr : p ≠ r) (hqr : q ≠ r) (s := 4) (hs : ¬ Nat.Prime s) 
  (hdiv : Nat.Prime 2) :
  ∃ n : ℕ, n = (p * q * r^2 * s)^3 ∧ ∀ m : ℕ, (∃ a b c d : ℕ, a = 3 ∧ b = 3 ∧ c = 6 ∧ d = 3 ∧ m = p^a * q^b * r^c * s^d) → m ≥ n :=
sorry

end smallest_perfect_cube_divisor_l204_20434


namespace tan_beta_minus_2alpha_l204_20493

theorem tan_beta_minus_2alpha (alpha beta : ℝ) (h1 : Real.tan alpha = 2) (h2 : Real.tan (beta - alpha) = 3) : 
  Real.tan (beta - 2 * alpha) = 1 / 7 := 
sorry

end tan_beta_minus_2alpha_l204_20493


namespace vertex_of_quadratic_function_l204_20448

theorem vertex_of_quadratic_function :
  ∀ x: ℝ, (2 - (x + 1)^2) = 2 - (x + 1)^2 → (∃ h k : ℝ, (h, k) = (-1, 2) ∧ ∀ x: ℝ, (2 - (x + 1)^2) = k - (x - h)^2) :=
by
  sorry

end vertex_of_quadratic_function_l204_20448


namespace base_conversion_sum_l204_20463

def digit_C : ℕ := 12
def base_14_value : ℕ := 3 * 14^2 + 5 * 14^1 + 6 * 14^0
def base_13_value : ℕ := 4 * 13^2 + digit_C * 13^1 + 9 * 13^0

theorem base_conversion_sum :
  (base_14_value + base_13_value = 1505) :=
by sorry

end base_conversion_sum_l204_20463


namespace imo1983_q6_l204_20433

theorem imo1983_q6 (a b c : ℝ) (h : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
by
  sorry

end imo1983_q6_l204_20433


namespace total_pens_l204_20467

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l204_20467


namespace max_value_x2_plus_y2_l204_20454

theorem max_value_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 + 4 * y^2 = 10 * x) : 
  x^2 + y^2 ≤ 4 :=
sorry

end max_value_x2_plus_y2_l204_20454


namespace find_minimum_n_l204_20460

variable {a_1 d : ℝ}
variable {S : ℕ → ℝ}

def is_arithmetic_sequence (a_1 d : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n / 2 : ℝ) * (2 * a_1 + (n - 1) * d)

def condition1 (a_1 : ℝ) : Prop := a_1 < 0

def condition2 (S : ℕ → ℝ) : Prop := S 7 = S 13

theorem find_minimum_n (a_1 d : ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a_1 d S)
  (h_a1_neg : condition1 a_1)
  (h_s7_eq_s13 : condition2 S) :
  ∃ n : ℕ, n = 10 ∧ ∀ m : ℕ, S n ≤ S m := 
sorry

end find_minimum_n_l204_20460


namespace recover_original_sequence_l204_20422

theorem recover_original_sequence :
  ∃ (a d : ℤ),
    [a, a + d, a + 2 * d, a + 3 * d, a + 4 * d, a + 5 * d] = [113, 125, 137, 149, 161, 173] :=
by
  sorry

end recover_original_sequence_l204_20422


namespace cookies_left_l204_20481

theorem cookies_left (total_cookies : ℕ) (fraction_given : ℚ) (given_cookies : ℕ) (remaining_cookies : ℕ) 
  (h1 : total_cookies = 20)
  (h2 : fraction_given = 2/5)
  (h3 : given_cookies = fraction_given * total_cookies)
  (h4 : remaining_cookies = total_cookies - given_cookies) :
  remaining_cookies = 12 :=
by
  sorry

end cookies_left_l204_20481


namespace total_profit_l204_20476

theorem total_profit (A B C : ℕ) (A_invest B_invest C_invest A_share : ℕ) (total_invest total_profit : ℕ)
  (h1 : A_invest = 6300)
  (h2 : B_invest = 4200)
  (h3 : C_invest = 10500)
  (h4 : A_share = 3630)
  (h5 : total_invest = A_invest + B_invest + C_invest)
  (h6 : total_profit * A_share = A_invest * total_invest) :
  total_profit = 12100 :=
by
  sorry

end total_profit_l204_20476


namespace problem_l204_20489

theorem problem (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 + 2007 = 2008 :=
by
  sorry

end problem_l204_20489


namespace more_cats_than_dogs_l204_20482

-- Define the number of cats and dogs
def c : ℕ := 23
def d : ℕ := 9

-- The theorem we need to prove
theorem more_cats_than_dogs : c - d = 14 := by
  sorry

end more_cats_than_dogs_l204_20482


namespace sum_of_reciprocal_of_roots_l204_20431

theorem sum_of_reciprocal_of_roots :
  ∀ x1 x2 : ℝ, (x1 * x2 = 2) → (x1 + x2 = 3) → (1 / x1 + 1 / x2 = 3 / 2) :=
by
  intros x1 x2 h_prod h_sum
  sorry

end sum_of_reciprocal_of_roots_l204_20431


namespace sum_prob_less_one_l204_20442

theorem sum_prob_less_one (x y z : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) :
  x * (1 - y) * (1 - z) + (1 - x) * y * (1 - z) + (1 - x) * (1 - y) * z < 1 :=
by
  sorry

end sum_prob_less_one_l204_20442


namespace S_sum_l204_20429

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(n / 2)
  else (n + 1) / 2

theorem S_sum :
  S 19 + S 37 + S 52 = 3 :=
by
  sorry

end S_sum_l204_20429


namespace house_cost_ratio_l204_20469

theorem house_cost_ratio {base_salary commission house_A_cost total_income : ℕ}
    (H_base_salary: base_salary = 3000)
    (H_commission: commission = 2)
    (H_house_A_cost: house_A_cost = 60000)
    (H_total_income: total_income = 8000)
    (H_total_sales_price: ℕ)
    (H_house_B_cost: ℕ)
    (H_house_C_cost: ℕ)
    (H_m: ℕ)
    (h1: total_income - base_salary = 5000)
    (h2: total_sales_price * commission / 100 = 5000)
    (h3: total_sales_price = 250000)
    (h4: house_B_cost = 3 * house_A_cost)
    (h5: total_sales_price = house_A_cost + house_B_cost + house_C_cost)
    (h6: house_C_cost = m * house_A_cost - 110000)
  : m = 2 :=
by
  sorry

end house_cost_ratio_l204_20469


namespace sale_price_of_trouser_l204_20405

theorem sale_price_of_trouser : (100 - 0.70 * 100) = 30 := by
  sorry

end sale_price_of_trouser_l204_20405


namespace total_people_seated_l204_20473

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l204_20473


namespace ratio_sheep_to_horses_is_correct_l204_20426

-- Definitions of given conditions
def ounces_per_horse := 230
def total_ounces_per_day := 12880
def number_of_sheep := 16

-- Express the number of horses and the ratio of sheep to horses
def number_of_horses : ℕ := total_ounces_per_day / ounces_per_horse
def ratio_sheep_to_horses := number_of_sheep / number_of_horses

-- The main statement to be proved
theorem ratio_sheep_to_horses_is_correct : ratio_sheep_to_horses = 2 / 7 :=
by
  sorry

end ratio_sheep_to_horses_is_correct_l204_20426


namespace percentage_increase_l204_20470

variable (T : ℕ) (total_time : ℕ)

theorem percentage_increase (h1 : T = 4) (h2 : total_time = 10) : 
  ∃ P : ℕ, (T + P / 100 * T = total_time - T) → P = 50 := 
by 
  sorry

end percentage_increase_l204_20470


namespace gray_areas_trees_count_l204_20432

noncomputable def totalTreesInGrayAreas (T : ℕ) (white1 white2 white3 : ℕ) : ℕ :=
  let gray2 := T - white2
  let gray3 := T - white3
  gray2 + gray3

theorem gray_areas_trees_count (T : ℕ) :
  T = 100 → totalTreesInGrayAreas T 100 82 90 = 26 :=
by sorry

end gray_areas_trees_count_l204_20432


namespace train_speed_l204_20416

noncomputable def train_length : ℝ := 65 -- length of the train in meters
noncomputable def time_to_pass : ℝ := 6.5 -- time to pass the telegraph post in seconds
noncomputable def speed_conversion_factor : ℝ := 18 / 5 -- conversion factor from m/s to km/h

theorem train_speed (h_length : train_length = 65) (h_time : time_to_pass = 6.5) :
  (train_length / time_to_pass) * speed_conversion_factor = 36 :=
by
  simp [h_length, h_time, train_length, time_to_pass, speed_conversion_factor]
  sorry

end train_speed_l204_20416


namespace exponential_inequality_l204_20465

variables (x a b : ℝ)

theorem exponential_inequality (h1 : x > 0) (h2 : 1 < b^x) (h3 : b^x < a^x) : 1 < b ∧ b < a :=
by
   sorry

end exponential_inequality_l204_20465


namespace expression_meaning_l204_20440

variable (m n : ℤ) -- Assuming m and n are integers for the context.

theorem expression_meaning : 2 * (m - n) = 2 * (m - n) := 
by
  -- It simply follows from the definition of the expression
  sorry

end expression_meaning_l204_20440


namespace avg_salary_of_Raj_and_Roshan_l204_20479

variable (R S : ℕ)

theorem avg_salary_of_Raj_and_Roshan (h1 : (R + S + 7000) / 3 = 5000) : (R + S) / 2 = 4000 := by
  sorry

end avg_salary_of_Raj_and_Roshan_l204_20479


namespace reflect_parallelogram_l204_20424

theorem reflect_parallelogram :
  let D : ℝ × ℝ := (4,1)
  let Dx : ℝ × ℝ := (D.1, -D.2) -- Reflect across x-axis
  let Dxy : ℝ × ℝ := (Dx.2 - 1, Dx.1 - 1) -- Translate point down by 1 unit and reflect across y=x
  let D'' : ℝ × ℝ := (Dxy.1 + 1, Dxy.2 + 1) -- Translate point back up by 1 unit
  D'' = (-2, 5) := by
  sorry

end reflect_parallelogram_l204_20424


namespace rodney_probability_correct_guess_l204_20498

noncomputable def two_digit_integer (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

noncomputable def tens_digit (n : ℕ) : Prop :=
  (n / 10 = 7 ∨ n / 10 = 8 ∨ n / 10 = 9)

noncomputable def units_digit_even (n : ℕ) : Prop :=
  (n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8)

noncomputable def greater_than_seventy_five (n : ℕ) : Prop := n > 75

theorem rodney_probability_correct_guess (n : ℕ) :
  two_digit_integer n →
  tens_digit n →
  units_digit_even n →
  greater_than_seventy_five n →
  (∃ m, m = 1 / 12) :=
sorry

end rodney_probability_correct_guess_l204_20498


namespace smallest_portion_is_two_l204_20407

theorem smallest_portion_is_two (a1 a2 a3 a4 a5 : ℕ) (d : ℕ) (h1 : a1 = a3 - 2 * d) (h2 : a2 = a3 - d) (h3 : a4 = a3 + d) (h4 : a5 = a3 + 2 * d) (h5 : a1 + a2 + a3 + a4 + a5 = 120) (h6 : a3 + a4 + a5 = 7 * (a1 + a2)) : a1 = 2 :=
by sorry

end smallest_portion_is_two_l204_20407


namespace positive_difference_balances_l204_20452

noncomputable def laura_balance (L_0 : ℝ) (L_r : ℝ) (L_n : ℕ) (t : ℕ) : ℝ :=
  L_0 * (1 + L_r / L_n) ^ (L_n * t)

noncomputable def mark_balance (M_0 : ℝ) (M_r : ℝ) (t : ℕ) : ℝ :=
  M_0 * (1 + M_r * t)

theorem positive_difference_balances :
  let L_0 := 10000
  let L_r := 0.04
  let L_n := 2
  let t := 20
  let M_0 := 10000
  let M_r := 0.06
  abs ((laura_balance L_0 L_r L_n t) - (mark_balance M_0 M_r t)) = 80.40 :=
by
  sorry

end positive_difference_balances_l204_20452


namespace unique_positive_integer_solution_l204_20412

theorem unique_positive_integer_solution (n p : ℕ) (x y : ℕ) :
  (x + p * y = n ∧ x + y = p^2 ∧ x > 0 ∧ y > 0) ↔ 
  (p > 1 ∧ (p - 1) ∣ (n - 1) ∧ ∀ k : ℕ, n ≠ p^k ∧ ∃! t : ℕ × ℕ, (t.1 + p * t.2 = n ∧ t.1 + t.2 = p^2 ∧ t.1 > 0 ∧ t.2 > 0)) :=
by
  sorry

end unique_positive_integer_solution_l204_20412


namespace rectangle_area_constant_l204_20456

theorem rectangle_area_constant (d : ℝ) (length width : ℝ) (h_ratio : length / width = 5 / 2) (h_diag : d = Real.sqrt (length^2 + width^2)) :
  ∃ k : ℝ, (length * width) = k * d^2 ∧ k = 10 / 29 :=
by
  use 10 / 29
  sorry

end rectangle_area_constant_l204_20456


namespace find_ac_and_area_l204_20497

variables {a b c : ℝ} {A B C : ℝ}
variables (triangle_abc : ∀ {a b c : ℝ} {A B C : ℝ}, a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
variables (h1 : (a ^ 2 + c ^ 2 - b ^ 2) / Real.cos B = 4)
variables (h2 : (2 * b * Real.cos C - 2 * c * Real.cos B) / (b * Real.cos C + c * Real.cos B) - c / a = 2)

noncomputable def ac_value := 2

noncomputable def area_of_triangle_abc := (Real.sqrt 15) / 4

theorem find_ac_and_area (triangle_abc : ∀ {a b c : ℝ} {A B C : ℝ}, a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) 
                         (h1 : (a ^ 2 + c ^ 2 - b ^ 2) / Real.cos B = 4) 
                         (h2 : (2 * b * Real.cos C - 2 * c * Real.cos B) / (b * Real.cos C + c * Real.cos B) - c / a = 2):
  ac_value = 2 ∧
  area_of_triangle_abc = (Real.sqrt 15) / 4 := 
sorry

end find_ac_and_area_l204_20497


namespace find_xyz_ratio_l204_20408

theorem find_xyz_ratio (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 2) 
  (h2 : a^2 / x^2 + b^2 / y^2 + c^2 / z^2 = 1) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 :=
by sorry

end find_xyz_ratio_l204_20408


namespace find_g_inverse_84_l204_20496

-- Definition of the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- Definition stating the goal
theorem find_g_inverse_84 : g⁻¹ 84 = 3 :=
sorry

end find_g_inverse_84_l204_20496


namespace field_area_restriction_l204_20437

theorem field_area_restriction (S : ℚ) (b : ℤ) (a : ℚ) (x y : ℚ) 
  (h1 : 10 * 300 * S ≤ 10000)
  (h2 : 2 * a = - b)
  (h3 : abs (6 * y) + 3 ≥ 3)
  (h4 : 2 * abs (2 * x) - abs b ≤ 9)
  (h5 : b ∈ [-4, -3, -2, -1, 0, 1, 2, 3, 4])
: S ≤ 10 / 3 := sorry

end field_area_restriction_l204_20437


namespace pie_difference_l204_20492

theorem pie_difference (p1 p2 : ℚ) (h1 : p1 = 5 / 6) (h2 : p2 = 2 / 3) : p1 - p2 = 1 / 6 := 
by 
  sorry

end pie_difference_l204_20492


namespace people_off_second_eq_8_l204_20420

-- Initial number of people on the bus
def initial_people := 50

-- People who got off at the first stop
def people_off_first := 15

-- People who got on at the second stop
def people_on_second := 2

-- People who got off at the second stop (unknown, let's call it x)
variable (x : ℕ)

-- People who got off at the third stop
def people_off_third := 4

-- People who got on at the third stop
def people_on_third := 3

-- Number of people on the bus after the third stop
def people_after_third := 28

-- Equation formed by given conditions
def equation := initial_people - people_off_first - x + people_on_second - people_off_third + people_on_third = people_after_third

-- Goal: Prove the equation with given conditions results in x = 8
theorem people_off_second_eq_8 : equation x → x = 8 := by
  sorry

end people_off_second_eq_8_l204_20420


namespace can_form_isosceles_triangle_with_given_sides_l204_20419

-- Define a structure for the sides of a triangle
structure Triangle (α : Type _) :=
  (a b c : α)

-- Define the predicate for the triangle inequality
def triangle_inequality {α : Type _} [LinearOrder α] [Add α] (t : Triangle α) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

-- Define the predicate for an isosceles triangle
def is_isosceles {α : Type _} [DecidableEq α] (t : Triangle α) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

-- Define the main theorem which checks if the given sides can form an isosceles triangle
theorem can_form_isosceles_triangle_with_given_sides
  (t : Triangle ℕ)
  (h_tri : triangle_inequality t)
  (h_iso : is_isosceles t) :
  t = ⟨2, 2, 1⟩ :=
  sorry

end can_form_isosceles_triangle_with_given_sides_l204_20419


namespace cubic_coeff_relationship_l204_20401

theorem cubic_coeff_relationship (a b c d u v w : ℝ) 
  (h_eq : a * (u^3) + b * (u^2) + c * u + d = 0)
  (h_vieta1 : u + v + w = -(b / a)) 
  (h_vieta2 : u * v + u * w + v * w = c / a) 
  (h_vieta3 : u * v * w = -d / a) 
  (h_condition : u + v = u * v) :
  (c + d) * (b + c + d) = a * d :=
by 
  sorry

end cubic_coeff_relationship_l204_20401


namespace number_of_valid_pairs_l204_20435

-- Definition of the conditions according to step (a)
def perimeter (l w : ℕ) : Prop := 2 * (l + w) = 80
def integer_lengths (l w : ℕ) : Prop := true
def length_greater_than_width (l w : ℕ) : Prop := l > w

-- The mathematical proof problem according to step (c)
theorem number_of_valid_pairs : ∃ n : ℕ, 
  (∀ l w : ℕ, perimeter l w → integer_lengths l w → length_greater_than_width l w → ∃! pair : (ℕ × ℕ), pair = (l, w)) ∧
  n = 19 :=
by 
  sorry

end number_of_valid_pairs_l204_20435


namespace calculation_of_nested_cuberoot_l204_20464

theorem calculation_of_nested_cuberoot (M : Real) (h : 1 < M) : (M^1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3) = M^(40 / 81) := 
by 
  sorry

end calculation_of_nested_cuberoot_l204_20464


namespace egg_hunt_ratio_l204_20477

theorem egg_hunt_ratio :
  ∃ T : ℕ, (3 * T + 30 = 400 ∧ T = 123) ∧ (60 : ℚ) / (T - 20 : ℚ) = 60 / 103 :=
by
  sorry

end egg_hunt_ratio_l204_20477


namespace symmetric_line_origin_l204_20491

theorem symmetric_line_origin (a b : ℝ) :
  (∀ (m n : ℝ), a * m + 3 * n = 9 → -m + 3 * -n + b = 0) ↔ a = -1 ∧ b = -9 :=
by
  sorry

end symmetric_line_origin_l204_20491


namespace true_false_questions_count_l204_20459

noncomputable def number_of_true_false_questions (T F M : ℕ) : Prop :=
  T + F + M = 45 ∧ M = 2 * F ∧ F = T + 7

theorem true_false_questions_count : ∃ T F M : ℕ, number_of_true_false_questions T F M ∧ T = 6 :=
by
  sorry

end true_false_questions_count_l204_20459


namespace symmetry_condition_l204_20423

theorem symmetry_condition 
  (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) : 
  (∀ a b : ℝ, b = 2 * a → (∃ y, y = (p * (b/2) + 2*q) / (r * (b/2) + 2*s) ∧  b = 2*(y/2) )) → 
  p + r = 0 :=
by
  sorry

end symmetry_condition_l204_20423


namespace math_problem_l204_20451

theorem math_problem : ((-7)^3 / 7^2 - 2^5 + 4^3 - 8) = 81 :=
by
  sorry

end math_problem_l204_20451


namespace find_a_l204_20471

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then (1/2)^x - 7 else x^2

theorem find_a (a : ℝ) (h : f a = 1) : a = -3 ∨ a = 1 := 
by
  sorry

end find_a_l204_20471


namespace probability_of_selecting_cooking_is_one_fourth_l204_20484

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end probability_of_selecting_cooking_is_one_fourth_l204_20484


namespace impossible_division_l204_20436

noncomputable def total_matches := 1230

theorem impossible_division :
  ∀ (x y z : ℕ), 
  (x + y + z = total_matches) → 
  (z = (1 / 2) * (x + y + z)) → 
  false :=
by
  sorry

end impossible_division_l204_20436


namespace distribute_pencils_l204_20417

theorem distribute_pencils (number_of_pencils : ℕ) (number_of_people : ℕ)
  (h_pencils : number_of_pencils = 2) (h_people : number_of_people = 5) :
  number_of_distributions = 15 := by
  sorry

end distribute_pencils_l204_20417


namespace product_of_numbers_l204_20430

theorem product_of_numbers :
  ∃ (x y z : ℚ), (x + y + z = 30) ∧ (x = 3 * (y + z)) ∧ (y = 5 * z) ∧ (x * y * z = 175.78125) :=
by
  sorry

end product_of_numbers_l204_20430


namespace pay_docked_per_lateness_l204_20488

variable (hourly_rate : ℤ) (work_hours : ℤ) (times_late : ℕ) (actual_pay : ℤ) 

theorem pay_docked_per_lateness (h_rate : hourly_rate = 30) 
                                (w_hours : work_hours = 18) 
                                (t_late : times_late = 3) 
                                (a_pay : actual_pay = 525) :
                                (hourly_rate * work_hours - actual_pay) / times_late = 5 :=
by
  sorry

end pay_docked_per_lateness_l204_20488


namespace distance_Bella_to_Galya_l204_20404

theorem distance_Bella_to_Galya (D_B D_V D_G : ℕ) (BV VG : ℕ)
  (hD_B : D_B = 700)
  (hD_V : D_V = 600)
  (hD_G : D_G = 650)
  (hBV : BV = 100)
  (hVG : VG = 50)
  : BV + VG = 150 := by
  sorry

end distance_Bella_to_Galya_l204_20404


namespace vanessa_score_l204_20402

theorem vanessa_score (total_points team_score other_players_avg_score: ℝ) : 
  total_points = 72 ∧ team_score = 7 ∧ other_players_avg_score = 4.5 → 
  ∃ vanessa_points: ℝ, vanessa_points = 40.5 :=
by
  sorry

end vanessa_score_l204_20402


namespace age_difference_l204_20499

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 14) : C = A - 14 :=
by sorry

end age_difference_l204_20499


namespace last_two_digits_A_pow_20_l204_20475

/-- 
Proof that for any even number A not divisible by 10, 
the last two digits of A^20 are 76.
--/
theorem last_two_digits_A_pow_20 (A : ℕ) (h_even : A % 2 = 0) (h_not_div_by_10 : A % 10 ≠ 0) : 
  (A ^ 20) % 100 = 76 :=
by
  sorry

end last_two_digits_A_pow_20_l204_20475


namespace max_value_a_l204_20468

theorem max_value_a (a : ℝ) : 
  (∀ x : ℝ, x > -1 → x + 1 > 0 → x + 1 + 1 / (x + 1) - 2 ≥ a) → a ≤ 0 :=
by
  -- Proof omitted
  sorry

end max_value_a_l204_20468


namespace cos_pi_zero_l204_20462

theorem cos_pi_zero : ∃ f : ℝ → ℝ, (∀ x, f x = (Real.cos x) ^ 2 + Real.cos x) ∧ f Real.pi = 0 := by
  sorry

end cos_pi_zero_l204_20462


namespace probability_red_nonjoker_then_black_or_joker_l204_20439

theorem probability_red_nonjoker_then_black_or_joker :
  let total_cards := 60
  let red_non_joker := 26
  let black_or_joker := 40
  let total_ways_to_draw_two_cards := total_cards * (total_cards - 1)
  let probability := (red_non_joker * black_or_joker : ℚ) / total_ways_to_draw_two_cards
  probability = 5 / 17 :=
by
  -- Definitions for the conditions
  let total_cards := 60
  let red_non_joker := 26
  let black_or_joker := 40
  let total_ways_to_draw_two_cards := total_cards * (total_cards - 1)
  let probability := (red_non_joker * black_or_joker : ℚ) / total_ways_to_draw_two_cards
  -- Add sorry placeholder for proof
  sorry

end probability_red_nonjoker_then_black_or_joker_l204_20439


namespace sum_of_digits_l204_20449

def original_sum := 943587 + 329430
def provided_sum := 1412017
def correct_sum_after_change (d e : ℕ) : ℕ := 
  let new_first := if d = 3 then 944587 else 943587
  let new_second := if d = 3 then 429430 else 329430
  new_first + new_second

theorem sum_of_digits (d e : ℕ) : d = 3 ∧ e = 4 → d + e = 7 :=
by
  intros
  exact sorry

end sum_of_digits_l204_20449


namespace inequality_must_hold_l204_20443

theorem inequality_must_hold (m n : ℝ) (h : m > n) : 2 + m > 2 + n :=
sorry

end inequality_must_hold_l204_20443


namespace min_value_of_squares_l204_20438

theorem min_value_of_squares (a b s t : ℝ) (h1 : a + b = t) (h2 : a - b = s) :
  a^2 + b^2 = (t^2 + s^2) / 2 :=
sorry

end min_value_of_squares_l204_20438


namespace real_roots_range_l204_20486

theorem real_roots_range (a : ℝ) : (∃ x : ℝ, a*x^2 + 2*x - 1 = 0) ↔ (a >= -1 ∧ a ≠ 0) :=
by 
  sorry

end real_roots_range_l204_20486


namespace beaver_stores_60_carrots_l204_20453

theorem beaver_stores_60_carrots (b r : ℕ) (h1 : 4 * b = 5 * r) (h2 : b = r + 3) : 4 * b = 60 :=
by
  sorry

end beaver_stores_60_carrots_l204_20453


namespace simplify_expression_l204_20415

theorem simplify_expression :
  ( ( (11 / 4) / (11 / 10 + 10 / 3) ) / ( 5 / 2 - ( 4 / 3 ) ) ) /
  ( ( 5 / 7 ) - ( ( (13 / 6 + 9 / 2) * 3 / 8 ) / (11 / 4 - 3 / 2) ) )
  = - (35 / 9) :=
by
  sorry

end simplify_expression_l204_20415


namespace solution_set_inequality_l204_20446

theorem solution_set_inequality (x : ℝ) :
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ (-2 ≤ x ∧ x ≤ 2) ∨ (x = 6) := by
  sorry

end solution_set_inequality_l204_20446


namespace price_of_toy_organizers_is_78_l204_20474

variable (P : ℝ) -- Price per set of toy organizers

-- Conditions
def total_cost_of_toy_organizers (P : ℝ) : ℝ := 3 * P
def total_cost_of_gaming_chairs : ℝ := 2 * 83
def total_sales (P : ℝ) : ℝ := total_cost_of_toy_organizers P + total_cost_of_gaming_chairs
def delivery_fee (P : ℝ) : ℝ := 0.05 * total_sales P
def total_amount_paid (P : ℝ) : ℝ := total_sales P + delivery_fee P

-- Proof statement
theorem price_of_toy_organizers_is_78 (h : total_amount_paid P = 420) : P = 78 :=
by
  sorry

end price_of_toy_organizers_is_78_l204_20474


namespace find_divisor_l204_20411

variable (x y : ℝ)
variable (h1 : (x - 5) / 7 = 7)
variable (h2 : (x - 2) / y = 4)

theorem find_divisor (x y : ℝ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 2) / y = 4) : y = 13 := by
  sorry

end find_divisor_l204_20411


namespace desk_length_l204_20406

theorem desk_length (width perimeter length : ℤ) (h1 : width = 9) (h2 : perimeter = 46) (h3 : perimeter = 2 * (length + width)) : length = 14 :=
by
  rw [h1, h2] at h3
  sorry

end desk_length_l204_20406


namespace func_above_x_axis_l204_20403

theorem func_above_x_axis (a : ℝ) :
  (∀ x : ℝ, (x^4 + 4*x^3 + a*x^2 - 4*x + 1) > 0) ↔ a > 2 :=
sorry

end func_above_x_axis_l204_20403


namespace line_b_parallel_or_in_plane_l204_20410

def Line : Type := sorry    -- Placeholder for the type of line
def Plane : Type := sorry   -- Placeholder for the type of plane

def is_parallel (a b : Line) : Prop := sorry             -- Predicate for parallel lines
def is_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry   -- Predicate for a line being parallel to a plane
def lies_in_plane (l : Line) (p : Plane) : Prop := sorry          -- Predicate for a line lying in a plane

theorem line_b_parallel_or_in_plane (a b : Line) (α : Plane) 
  (h1 : is_parallel a b) 
  (h2 : is_parallel_to_plane a α) : 
  is_parallel_to_plane b α ∨ lies_in_plane b α :=
sorry

end line_b_parallel_or_in_plane_l204_20410


namespace triangle_angle_sum_depends_on_parallel_postulate_l204_20455

-- Definitions of conditions
def triangle_angle_sum_theorem (A B C : ℝ) : Prop :=
  A + B + C = 180

def parallel_postulate : Prop :=
  ∀ (l : ℝ) (P : ℝ), ∃! (m : ℝ), m ≠ l ∧ ∀ (Q : ℝ), Q ≠ P → (Q = l ∧ Q = m)

-- Theorem statement: proving the dependence of the triangle_angle_sum_theorem on the parallel_postulate
theorem triangle_angle_sum_depends_on_parallel_postulate: 
  ∀ (A B C : ℝ), parallel_postulate → triangle_angle_sum_theorem A B C :=
sorry

end triangle_angle_sum_depends_on_parallel_postulate_l204_20455


namespace average_of_second_set_of_two_numbers_l204_20413

theorem average_of_second_set_of_two_numbers
  (S : ℝ)
  (avg1 avg2 avg3 : ℝ)
  (h1 : S = 6 * 3.95)
  (h2 : avg1 = 3.4)
  (h3 : avg3 = 4.6) :
  (S - (2 * avg1) - (2 * avg3)) / 2 = 3.85 :=
by
  sorry

end average_of_second_set_of_two_numbers_l204_20413
