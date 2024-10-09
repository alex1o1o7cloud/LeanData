import Mathlib

namespace age_difference_l13_1367

theorem age_difference (A B n : ℕ) (h1 : A = B + n) (h2 : A - 1 = 3 * (B - 1)) (h3 : A = B^2) : n = 2 :=
by
  sorry

end age_difference_l13_1367


namespace gcd_lcm_product_l13_1316

theorem gcd_lcm_product (a b : ℕ) (ha : a = 225) (hb : b = 252) :
  Nat.gcd a b * Nat.lcm a b = 56700 := by
  sorry

end gcd_lcm_product_l13_1316


namespace ellipse_properties_l13_1303

theorem ellipse_properties
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b ≥ 0)
  (e : ℝ)
  (hc : e = 4 / 5)
  (directrix : ℝ)
  (hd : directrix = 25 / 4)
  (x y : ℝ)
  (hx : (x - 6)^2 / 25 + (y - 6)^2 / 9 = 1) :
  x^2 / 25 + y^2 / 9 = 1 :=
sorry

end ellipse_properties_l13_1303


namespace solve_fraction_equation_l13_1359

theorem solve_fraction_equation (x : ℚ) (h : (x + 7) / (x - 4) = (x - 5) / (x + 3)) : x = -1 / 19 := 
sorry

end solve_fraction_equation_l13_1359


namespace range_of_a_l13_1330

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ x^2 + (a - 1) * x + 1 < 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l13_1330


namespace log_div_log_inv_of_16_l13_1398

theorem log_div_log_inv_of_16 : (Real.log 16) / (Real.log (1 / 16)) = -1 :=
by
  sorry

end log_div_log_inv_of_16_l13_1398


namespace max_a2_plus_b2_l13_1356

theorem max_a2_plus_b2 (a b : ℝ) (h1 : b = 1) (h2 : 1 ≤ -a + 7) (h3 : 1 ≥ a - 3) : a^2 + b^2 = 37 :=
by {
  sorry
}

end max_a2_plus_b2_l13_1356


namespace fraction_subtraction_l13_1315

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem fraction_subtraction : 
  (18 / 42 - 2 / 9) = (13 / 63) := 
by 
  sorry

end fraction_subtraction_l13_1315


namespace robert_ate_more_chocolates_l13_1362

-- Define the number of chocolates eaten by Robert and Nickel
def robert_chocolates : ℕ := 12
def nickel_chocolates : ℕ := 3

-- State the problem as a theorem to prove
theorem robert_ate_more_chocolates :
  robert_chocolates - nickel_chocolates = 9 :=
by
  sorry

end robert_ate_more_chocolates_l13_1362


namespace dormouse_is_thief_l13_1364

-- Definitions of the suspects
inductive Suspect
| MarchHare
| Hatter
| Dormouse

open Suspect

-- Definitions of the statement conditions
def statement (s : Suspect) : Suspect :=
match s with
| MarchHare => Hatter
| Hatter => sorry -- Sonya and Hatter's testimonies are not recorded
| Dormouse => sorry -- Sonya and Hatter's testimonies are not recorded

-- Condition that only the thief tells the truth
def tells_truth (thief : Suspect) (s : Suspect) : Prop :=
s = thief

-- Conditions of the problem
axiom condition1 : statement MarchHare = Hatter
axiom condition2 : ∃ t, tells_truth t MarchHare ∧ ¬ tells_truth t Hatter ∧ ¬ tells_truth t Dormouse

-- Proposition that Dormouse (Sonya) is the thief
theorem dormouse_is_thief : (∃ t, tells_truth t MarchHare ∧ ¬ tells_truth t Hatter ∧ ¬ tells_truth t Dormouse) → t = Dormouse :=
sorry

end dormouse_is_thief_l13_1364


namespace evaluate_expression_l13_1304

theorem evaluate_expression (x y z : ℕ) (hx : x = 5) (hy : y = 10) (hz : z = 3) : z * (y - 2 * x) = 0 := by
  sorry

end evaluate_expression_l13_1304


namespace sum_of_three_numbers_l13_1345

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1 : B = 30)
  (h2 : A * 3 = 2 * B)
  (h3 : C * 5 = 8 * B) : 
  A + B + C = 98 :=
by
  sorry

end sum_of_three_numbers_l13_1345


namespace find_a_l13_1357

-- Defining the curve y in terms of x and a
def curve (x : ℝ) (a : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Defining the derivative of the curve
def derivative (x : ℝ) (a : ℝ) : ℝ := 4*x^3 + 2*a*x

-- The proof statement asserting the value of a
theorem find_a (a : ℝ) (h1 : derivative (-1) a = 8): a = -6 :=
by
  -- we assume here the necessary calculations and logical steps to prove the theorem
  sorry

end find_a_l13_1357


namespace tangent_line_through_P_is_correct_l13_1333

-- Define the point P
def P : ℝ × ℝ := (2, 4)

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the tangent line equation to prove
def tangent_line (x y : ℝ) : Prop := x + 2 * y - 10 = 0

-- Problem statement in Lean 4
theorem tangent_line_through_P_is_correct :
  C P.1 P.2 → tangent_line P.1 P.2 :=
by
  intros hC
  sorry

end tangent_line_through_P_is_correct_l13_1333


namespace min_bench_sections_l13_1388

theorem min_bench_sections (N : ℕ) :
  ∀ x y : ℕ, (x = y) → (x = 8 * N) → (y = 12 * N) → (24 * N) % 20 = 0 → N = 5 :=
by
  intros
  sorry

end min_bench_sections_l13_1388


namespace part1_part2_range_of_a_l13_1387

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x - Real.log (x + 1)

theorem part1 (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) : f1 x ≥ 0 := sorry

noncomputable def f2 (x a : ℝ) : ℝ := Real.sin x - a * Real.log (x + 1)

theorem part2 {a : ℝ} (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ Real.pi) : f2 x a ≤ 2 * Real.exp x - 2 := sorry

theorem range_of_a : {a : ℝ | ∀ x : ℝ, 0 ≤ x → x ≤ Real.pi → f2 x a ≤ 2 * Real.exp x - 2} = {a : ℝ | a ≥ -1} := sorry

end part1_part2_range_of_a_l13_1387


namespace cristian_cookie_problem_l13_1318

theorem cristian_cookie_problem (white_cookies_init black_cookies_init eaten_black_cookies eaten_white_cookies remaining_black_cookies remaining_white_cookies total_remaining_cookies : ℕ) 
  (h_initial_white : white_cookies_init = 80)
  (h_black_more : black_cookies_init = white_cookies_init + 50)
  (h_eats_half_black : eaten_black_cookies = black_cookies_init / 2)
  (h_eats_three_fourth_white : eaten_white_cookies = (3 / 4) * white_cookies_init)
  (h_remaining_black : remaining_black_cookies = black_cookies_init - eaten_black_cookies)
  (h_remaining_white : remaining_white_cookies = white_cookies_init - eaten_white_cookies)
  (h_total_remaining : total_remaining_cookies = remaining_black_cookies + remaining_white_cookies) :
  total_remaining_cookies = 85 :=
by
  sorry

end cristian_cookie_problem_l13_1318


namespace purple_chips_selected_is_one_l13_1392

noncomputable def chips_selected (B G P R x : ℕ) : Prop :=
  (1^B) * (5^G) * (x^P) * (11^R) = 140800 ∧ 5 < x ∧ x < 11

theorem purple_chips_selected_is_one :
  ∃ B G P R x, chips_selected B G P R x ∧ P = 1 :=
by {
  sorry
}

end purple_chips_selected_is_one_l13_1392


namespace infinitely_many_k_numbers_unique_k_4_l13_1341

theorem infinitely_many_k_numbers_unique_k_4 :
  ∀ k : ℕ, (∃ n : ℕ, (∃ r : ℕ, n = r * (r + k)) ∧ (∃ m : ℕ, n = m^2 - k)
          ∧ ∀ N : ℕ, ∃ r : ℕ, ∃ m : ℕ, N < r ∧ (r * (r + k) = m^2 - k)) ↔ k = 4 :=
by
  sorry

end infinitely_many_k_numbers_unique_k_4_l13_1341


namespace gcd_98_63_l13_1326

-- The statement of the problem in Lean 4
theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l13_1326


namespace problem_statement_l13_1323

def f (x : ℝ) : ℝ := 3 * x^2 - 2
def k (x : ℝ) : ℝ := -2 * x^3 + 2

theorem problem_statement : f (k 2) = 586 := by
  sorry

end problem_statement_l13_1323


namespace meaning_of_sum_of_squares_l13_1349

theorem meaning_of_sum_of_squares (a b : ℝ) : a ^ 2 + b ^ 2 ≠ 0 ↔ (a ≠ 0 ∨ b ≠ 0) :=
by
  sorry

end meaning_of_sum_of_squares_l13_1349


namespace mr_castiel_sausages_l13_1344

theorem mr_castiel_sausages (S : ℕ) :
  S * (3 / 5) * (1 / 2) * (1 / 4) * (3 / 4) = 45 → S = 600 :=
by
  sorry

end mr_castiel_sausages_l13_1344


namespace highest_place_value_quotient_and_remainder_l13_1337

-- Conditions
def dividend := 438
def divisor := 4

-- Theorem stating that the highest place value of the quotient is the hundreds place, and the remainder is 2
theorem highest_place_value_quotient_and_remainder : 
  (dividend = divisor * (dividend / divisor) + (dividend % divisor)) ∧ 
  ((dividend / divisor) >= 100) ∧ 
  ((dividend % divisor) = 2) :=
by
  sorry

end highest_place_value_quotient_and_remainder_l13_1337


namespace triangle_internal_region_l13_1343

-- Define the three lines forming the triangle
def line1 (x y : ℝ) : Prop := x + 2 * y = 2
def line2 (x y : ℝ) : Prop := 2 * x + y = 2
def line3 (x y : ℝ) : Prop := x - y = 3

-- Define the inequalities representing the internal region of the triangle
def region (x y : ℝ) : Prop :=
  x - y < 3 ∧ x + 2 * y < 2 ∧ 2 * x + y > 2

-- State that the internal region excluding the boundary is given by the inequalities
theorem triangle_internal_region (x y : ℝ) :
  (∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 x y) → region x y :=
  sorry

end triangle_internal_region_l13_1343


namespace find_b_l13_1329

variable (x : ℝ)

theorem find_b (a b: ℝ) (h1 : x + 1/x = a) (h2 : x^3 + 1/x^3 = b) (ha : a = 3): b = 18 :=
by
  sorry

end find_b_l13_1329


namespace positive_multiples_of_4_with_units_digit_4_l13_1314

theorem positive_multiples_of_4_with_units_digit_4 (n : ℕ) : 
  ∃ n ≤ 15, ∀ m, m = 4 + 10 * (n - 1) → m < 150 ∧ m % 10 = 4 :=
by {
  sorry
}

end positive_multiples_of_4_with_units_digit_4_l13_1314


namespace amount_spent_on_milk_is_1500_l13_1335

def total_salary (saved : ℕ) (saving_percent : ℕ) : ℕ := 
  saved / (saving_percent / 100)

def total_spent_excluding_milk (rent groceries education petrol misc : ℕ) : ℕ := 
  rent + groceries + education + petrol + misc

def amount_spent_on_milk (total_salary total_spent savings : ℕ) : ℕ := 
  total_salary - total_spent - savings

theorem amount_spent_on_milk_is_1500 :
  let rent := 5000
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let misc := 2500
  let savings := 2000
  let saving_percent := 10
  let salary := total_salary savings saving_percent
  let spent_excluding_milk := total_spent_excluding_milk rent groceries education petrol misc
  amount_spent_on_milk salary spent_excluding_milk savings = 1500 :=
by {
  sorry
}

end amount_spent_on_milk_is_1500_l13_1335


namespace Avianna_red_candles_l13_1355

theorem Avianna_red_candles (R : ℕ) : 
  (R / 27 = 5 / 3) → R = 45 := 
by
  sorry

end Avianna_red_candles_l13_1355


namespace evaluate_expression_at_zero_l13_1360

theorem evaluate_expression_at_zero :
  (0^2 + 5 * 0 - 10) = -10 :=
by
  sorry

end evaluate_expression_at_zero_l13_1360


namespace batsman_average_runs_l13_1373

theorem batsman_average_runs
  (average_20_matches : ℕ → ℕ)
  (average_10_matches : ℕ → ℕ)
  (h1 : average_20_matches = 20 * 40)
  (h2 : average_10_matches = 10 * 13) :
  (average_20_matches + average_10_matches) / 30 = 31 := 
by 
  sorry

end batsman_average_runs_l13_1373


namespace Corey_found_golf_balls_on_Saturday_l13_1399

def goal : ℕ := 48
def golf_balls_found_on_sunday : ℕ := 18
def golf_balls_needed : ℕ := 14
def golf_balls_found_on_saturday : ℕ := 16

theorem Corey_found_golf_balls_on_Saturday :
  (goal - golf_balls_found_on_sunday - golf_balls_needed) = golf_balls_found_on_saturday := 
by
  sorry

end Corey_found_golf_balls_on_Saturday_l13_1399


namespace product_of_first_two_numbers_l13_1350

theorem product_of_first_two_numbers (A B C : ℕ) (h_coprime: Nat.gcd A B = 1 ∧ Nat.gcd B C = 1 ∧ Nat.gcd A C = 1)
  (h_product: B * C = 1073) (h_sum: A + B + C = 85) : A * B = 703 :=
sorry

end product_of_first_two_numbers_l13_1350


namespace probability_red_card_top_l13_1395

def num_red_cards : ℕ := 26
def total_cards : ℕ := 52
def prob_red_card_top : ℚ := num_red_cards / total_cards

theorem probability_red_card_top : prob_red_card_top = (1 / 2) := by
  sorry

end probability_red_card_top_l13_1395


namespace melanie_total_dimes_l13_1348

/-- Melanie had 7 dimes in her bank. Her dad gave her 8 dimes. Her mother gave her 4 dimes. -/
def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4

/-- How many dimes does Melanie have now? -/
theorem melanie_total_dimes : initial_dimes + dimes_from_dad + dimes_from_mom = 19 := by
  sorry

end melanie_total_dimes_l13_1348


namespace rachel_picture_books_shelves_l13_1378

theorem rachel_picture_books_shelves (mystery_shelves : ℕ) (books_per_shelf : ℕ) (total_books : ℕ) 
  (h1 : mystery_shelves = 6) 
  (h2 : books_per_shelf = 9) 
  (h3 : total_books = 72) : 
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 2 :=
by sorry

end rachel_picture_books_shelves_l13_1378


namespace quadratic_expression_value_l13_1352

theorem quadratic_expression_value (a : ℝ) (h : a^2 - 2 * a - 3 = 0) : a^2 - 2 * a + 1 = 4 :=
by 
  -- Proof omitted for clarity in this part
  sorry 

end quadratic_expression_value_l13_1352


namespace initial_markup_percentage_l13_1302

theorem initial_markup_percentage (C : ℝ) (M : ℝ) :
  (C > 0) →
  (1 + M) * 1.25 * 0.90 = 1.35 →
  M = 0.2 :=
by
  intros
  sorry

end initial_markup_percentage_l13_1302


namespace logarithm_function_decreasing_l13_1332

theorem logarithm_function_decreasing (a : ℝ) : 
  (∀ x ∈ Set.Ici (-1), (3 * x^2 - a * x + 5) ≤ (3 * x^2 - a * (x + 1) + 5)) ↔ (-8 < a ∧ a ≤ -6) :=
by
  sorry

end logarithm_function_decreasing_l13_1332


namespace average_math_test_score_l13_1365

theorem average_math_test_score :
    let june_score := 97
    let patty_score := 85
    let josh_score := 100
    let henry_score := 94
    let num_children := 4
    let total_score := june_score + patty_score + josh_score + henry_score
    total_score / num_children = 94 := by
  sorry

end average_math_test_score_l13_1365


namespace smallest_sector_angle_divided_circle_l13_1379

theorem smallest_sector_angle_divided_circle : ∃ a d : ℕ, 
  (2 * a + 7 * d = 90) ∧ 
  (8 * (a + (a + 7 * d)) / 2 = 360) ∧ 
  a = 38 := 
by
  sorry

end smallest_sector_angle_divided_circle_l13_1379


namespace find_y_l13_1347

theorem find_y (y : ℝ) : (∃ y : ℝ, (4, y) ≠ (2, -3) ∧ ((-3 - y) / (2 - 4) = 1)) → y = -1 :=
by
  sorry

end find_y_l13_1347


namespace fibonacci_contains_21_l13_1369

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ 
| 0 => 1
| 1 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

-- Theorem statement: Proving that 21 is in the Fibonacci sequence
theorem fibonacci_contains_21 : ∃ n, fibonacci n = 21 :=
by
  sorry

end fibonacci_contains_21_l13_1369


namespace a_plus_b_eq_11_l13_1391

noncomputable def f (a b x : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

theorem a_plus_b_eq_11 (a b : ℝ) 
  (h1 : ∀ x, f a b x ≤ f a b (-1))
  (h2 : f a b (-1) = 0) 
  : a + b = 11 :=
sorry

end a_plus_b_eq_11_l13_1391


namespace ticket_cost_difference_l13_1374

noncomputable def total_cost_adults (tickets : ℕ) (price : ℝ) : ℝ := tickets * price
noncomputable def total_cost_children (tickets : ℕ) (price : ℝ) : ℝ := tickets * price
noncomputable def total_tickets (adults : ℕ) (children : ℕ) : ℕ := adults + children
noncomputable def discount (threshold : ℕ) (discount_rate : ℝ) (cost : ℝ) (tickets : ℕ) : ℝ :=
  if tickets > threshold then cost * discount_rate else 0
noncomputable def final_cost (initial_cost : ℝ) (discount : ℝ) : ℝ := initial_cost - discount
noncomputable def proportional_discount (partial_cost : ℝ) (total_cost : ℝ) (total_discount : ℝ) : ℝ :=
  (partial_cost / total_cost) * total_discount
noncomputable def difference (cost1 : ℝ) (cost2 : ℝ) : ℝ := cost1 - cost2

theorem ticket_cost_difference :
  let adult_tickets := 9
  let children_tickets := 7
  let adult_price := 11
  let children_price := 7
  let discount_rate := 0.15
  let discount_threshold := 10
  let total_adult_cost := total_cost_adults adult_tickets adult_price
  let total_children_cost := total_cost_children children_tickets children_price
  let all_tickets := total_tickets adult_tickets children_tickets
  let initial_total_cost := total_adult_cost + total_children_cost
  let total_discount := discount discount_threshold discount_rate initial_total_cost all_tickets
  let final_total_cost := final_cost initial_total_cost total_discount
  let adult_discount := proportional_discount total_adult_cost initial_total_cost total_discount
  let children_discount := proportional_discount total_children_cost initial_total_cost total_discount
  let final_adult_cost := final_cost total_adult_cost adult_discount
  let final_children_cost := final_cost total_children_cost children_discount
  difference final_adult_cost final_children_cost = 42.52 := by
  sorry

end ticket_cost_difference_l13_1374


namespace simplify_expression_l13_1324

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end simplify_expression_l13_1324


namespace largest_three_digit_n_l13_1320

theorem largest_three_digit_n (n : ℕ) : 
  (70 * n ≡ 210 [MOD 350]) ∧ (n < 1000) → 
  n = 998 := by
  sorry

end largest_three_digit_n_l13_1320


namespace inequality_condition_l13_1396

theorem inequality_condition (x : ℝ) :
  ((x + 3) * (x - 2) < 0 ↔ -3 < x ∧ x < 2) →
  ((-3 < x ∧ x < 0) → (x + 3) * (x - 2) < 0) →
  ∃ p q : Prop, (p → q) ∧ ¬(q → p) ∧
  p = ((x + 3) * (x - 2) < 0) ∧ q = (-3 < x ∧ x < 0) := by
  sorry

end inequality_condition_l13_1396


namespace find_quotient_l13_1309

theorem find_quotient (A : ℕ) (h : 41 = (5 * A) + 1) : A = 8 :=
by
  sorry

end find_quotient_l13_1309


namespace find_number_l13_1381

def x : ℝ := 33.75

theorem find_number (x: ℝ) :
  (0.30 * x = 0.25 * 45) → x = 33.75 :=
by
  sorry

end find_number_l13_1381


namespace rich_walks_ratio_is_2_l13_1321

-- Define the conditions in the problem
def house_to_sidewalk : ℕ := 20
def sidewalk_to_end : ℕ := 200
def total_distance_walked : ℕ := 1980
def ratio_after_left_to_so_far (x : ℕ) : ℕ := (house_to_sidewalk + sidewalk_to_end) * x / (house_to_sidewalk + sidewalk_to_end)

-- Main theorem to prove the ratio is 2:1
theorem rich_walks_ratio_is_2 (x : ℕ) (h : 2 * ((house_to_sidewalk + sidewalk_to_end) * 2 + house_to_sidewalk + sidewalk_to_end / 2 * 3 ) = total_distance_walked) :
  ratio_after_left_to_so_far x = 2 :=
by
  sorry

end rich_walks_ratio_is_2_l13_1321


namespace belt_and_road_scientific_notation_l13_1306

theorem belt_and_road_scientific_notation : 
  4600000000 = 4.6 * 10^9 := 
by
  sorry

end belt_and_road_scientific_notation_l13_1306


namespace vertex_hyperbola_l13_1353

theorem vertex_hyperbola (a b : ℝ) (h_cond : 8 * a^2 + 4 * a * b = b^3) :
    let xv := -b / (2 * a)
    let yv := (4 * a - b^2) / (4 * a)
    (xv * yv = 1) :=
  by
  sorry

end vertex_hyperbola_l13_1353


namespace tax_per_pound_is_one_l13_1376

-- Define the conditions
def bulk_price_per_pound : ℝ := 5          -- Condition 1
def minimum_spend : ℝ := 40               -- Condition 2
def total_paid : ℝ := 240                 -- Condition 4
def excess_pounds : ℝ := 32               -- Condition 5

-- Define the proof problem statement
theorem tax_per_pound_is_one :
  ∃ (T : ℝ), total_paid = (minimum_spend / bulk_price_per_pound + excess_pounds) * bulk_price_per_pound + 
  (minimum_spend / bulk_price_per_pound + excess_pounds) * T ∧ 
  T = 1 :=
by 
  sorry

end tax_per_pound_is_one_l13_1376


namespace greatest_possible_employees_take_subway_l13_1328

variable (P F : ℕ)

def part_time_employees_take_subway : ℕ := P / 3
def full_time_employees_take_subway : ℕ := F / 4

theorem greatest_possible_employees_take_subway 
  (h1 : P + F = 48) : part_time_employees_take_subway P + full_time_employees_take_subway F ≤ 15 := 
sorry

end greatest_possible_employees_take_subway_l13_1328


namespace find_c_l13_1346

theorem find_c (c : ℝ) (h : ∃ (f : ℝ → ℝ), (f = λ x => c * x^3 + 23 * x^2 - 5 * c * x + 55) ∧ f (-5) = 0) : c = 6.3 := 
by {
  sorry
}

end find_c_l13_1346


namespace primes_sum_product_condition_l13_1317

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem primes_sum_product_condition (m n p : ℕ) (hm : is_prime m) (hn : is_prime n) (hp : is_prime p)  
  (h : m * n * p = 5 * (m + n + p)) : 
  m^2 + n^2 + p^2 = 78 :=
sorry

end primes_sum_product_condition_l13_1317


namespace cara_meets_don_distance_l13_1386

theorem cara_meets_don_distance (distance total_distance : ℝ) (cara_speed don_speed : ℝ) (delay : ℝ) 
  (h_total_distance : total_distance = 45)
  (h_cara_speed : cara_speed = 6)
  (h_don_speed : don_speed = 5)
  (h_delay : delay = 2) :
  distance = 30 :=
by
  have h := 1 / total_distance
  have : cara_speed * (distance / cara_speed) + don_speed * (distance / cara_speed - delay) = 45 := sorry
  exact sorry

end cara_meets_don_distance_l13_1386


namespace arjun_becca_3_different_colors_l13_1370

open Classical

noncomputable def arjun_becca_probability : ℚ := 
  let arjun_initial := [2, 1, 1, 1] -- 2 red, 1 green, 1 yellow, 1 violet
  let becca_initial := [2, 1] -- 2 black, 1 orange
  
  -- possible cases represented as a list of probabilities
  let cases := [
    (2/5) * (1/4) * (3/5),    -- Case 1: Arjun does move a red ball to Becca, and then processes accordingly
    (3/5) * (1/2) * (1/5),    -- Case 2a: Arjun moves a non-red ball, followed by Becca moving a black ball, concluding in the defined manner
    (3/5) * (1/2) * (3/5)     -- Case 2b: Arjun moves a non-red ball, followed by Becca moving a non-black ball, again concluding appropriately
  ]
  
  -- sum of cases representing the total probability
  let total_probability := List.sum cases
  
  total_probability

theorem arjun_becca_3_different_colors : arjun_becca_probability = 3/10 := 
  by
    simp [arjun_becca_probability]
    sorry

end arjun_becca_3_different_colors_l13_1370


namespace function_problem_l13_1382

theorem function_problem (f : ℕ → ℝ) (h1 : ∀ p q : ℕ, f (p + q) = f p * f q) (h2 : f 1 = 3) :
  (f (1) ^ 2 + f (2)) / f (1) + (f (2) ^ 2 + f (4)) / f (3) + (f (3) ^ 2 + f (6)) / f (5) + 
  (f (4) ^ 2 + f (8)) / f (7) + (f (5) ^ 2 + f (10)) / f (9) = 30 := by
  sorry

end function_problem_l13_1382


namespace rotation_transform_l13_1358

theorem rotation_transform (x y α : ℝ) :
    let x' := x * Real.cos α - y * Real.sin α
    let y' := x * Real.sin α + y * Real.cos α
    (x', y') = (x * Real.cos α - y * Real.sin α, x * Real.sin α + y * Real.cos α) := by
  sorry

end rotation_transform_l13_1358


namespace eve_ran_further_l13_1380

variable (ran_distance walked_distance difference_distance : ℝ)

theorem eve_ran_further (h1 : ran_distance = 0.7) (h2 : walked_distance = 0.6) : ran_distance - walked_distance = 0.1 := by
  sorry

end eve_ran_further_l13_1380


namespace complement_A_l13_1310

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := { x : ℝ | x < 2 }

theorem complement_A :
  (U \ A) = { x : ℝ | x >= 2 } :=
by
  sorry

end complement_A_l13_1310


namespace area_of_octagon_in_square_l13_1313

theorem area_of_octagon_in_square : 
  let A := (0, 0)
  let B := (6, 0)
  let C := (6, 6)
  let D := (0, 6)
  let E := (3, 0)
  let F := (6, 3)
  let G := (3, 6)
  let H := (0, 3)
  ∃ (octagon_area : ℚ),
    octagon_area = 6 :=
by
  sorry

end area_of_octagon_in_square_l13_1313


namespace total_profit_calculation_l13_1354

-- Define the parameters of the problem
def rajan_investment : ℕ := 20000
def rakesh_investment : ℕ := 25000
def mukesh_investment : ℕ := 15000
def rajan_investment_time : ℕ := 12 -- in months
def rakesh_investment_time : ℕ := 4 -- in months
def mukesh_investment_time : ℕ := 8 -- in months
def rajan_final_share : ℕ := 2400

-- Calculation for total profit
def total_profit (rajan_investment rakesh_investment mukesh_investment
                  rajan_investment_time rakesh_investment_time mukesh_investment_time
                  rajan_final_share : ℕ) : ℕ :=
  let rajan_share := rajan_investment * rajan_investment_time
  let rakesh_share := rakesh_investment * rakesh_investment_time
  let mukesh_share := mukesh_investment * mukesh_investment_time
  let total_investment := rajan_share + rakesh_share + mukesh_share
  (rajan_final_share * total_investment) / rajan_share

-- Proof problem statement
theorem total_profit_calculation :
  total_profit rajan_investment rakesh_investment mukesh_investment
               rajan_investment_time rakesh_investment_time mukesh_investment_time
               rajan_final_share = 4600 :=
by sorry

end total_profit_calculation_l13_1354


namespace min_value_of_x_l13_1393

theorem min_value_of_x (x : ℝ) (h : 2 * (x + 1) ≥ x + 1) : x ≥ -1 := sorry

end min_value_of_x_l13_1393


namespace girl_scouts_short_amount_l13_1342

-- Definitions based on conditions
def amount_earned : ℝ := 30
def pool_entry_cost_per_person : ℝ := 2.50
def num_people : ℕ := 10
def transportation_fee_per_person : ℝ := 1.25
def snack_cost_per_person : ℝ := 3.00

-- Calculate individual costs
def total_pool_entry_cost : ℝ := pool_entry_cost_per_person * num_people
def total_transportation_fee : ℝ := transportation_fee_per_person * num_people
def total_snack_cost : ℝ := snack_cost_per_person * num_people

-- Calculate total expenses
def total_expenses : ℝ := total_pool_entry_cost + total_transportation_fee + total_snack_cost

-- The amount left after expenses
def amount_left : ℝ := amount_earned - total_expenses

-- Proof problem statement
theorem girl_scouts_short_amount : amount_left = -37.50 := by
  sorry

end girl_scouts_short_amount_l13_1342


namespace abs_eq_self_iff_nonneg_l13_1334

variable (a : ℝ)

theorem abs_eq_self_iff_nonneg (h : |a| = a) : a ≥ 0 :=
by
  sorry

end abs_eq_self_iff_nonneg_l13_1334


namespace sum_of_series_l13_1312

theorem sum_of_series :
  ∑' n : ℕ, (if n = 0 then 0 else (3 * (n : ℤ) - 2) / ((n : ℤ) * ((n : ℤ) + 1) * ((n : ℤ) + 3))) = -19 / 30 :=
by
  sorry

end sum_of_series_l13_1312


namespace root_quad_eq_sum_l13_1301

theorem root_quad_eq_sum (a b : ℝ) (h1 : a^2 + a - 2022 = 0) (h2 : b^2 + b - 2022 = 0) (h3 : a + b = -1) : a^2 + 2 * a + b = 2021 :=
by sorry

end root_quad_eq_sum_l13_1301


namespace volume_le_one_fourth_of_original_volume_of_sub_tetrahedron_l13_1331

noncomputable def volume_tetrahedron (V A B C : Point) : ℝ := sorry

def is_interior_point (M V A B C : Point) : Prop := sorry -- Definition of an interior point

def is_barycenter (M V A B C : Point) : Prop := sorry -- Definition of a barycenter

def intersects_lines_planes (M V A B C A1 B1 C1 : Point) : Prop := sorry -- Definition of intersection points

def intersects_lines_sides (V A1 B1 C1 A B C A2 B2 C2 : Point) : Prop := sorry -- Definition of intersection points with sides

theorem volume_le_one_fourth_of_original (V A B C: Point) 
  (M : Point) (A1 B1 C1 A2 B2 C2 : Point) 
  (h_interior : is_interior_point M V A B C) 
  (h_intersects_planes : intersects_lines_planes M V A B C A1 B1 C1) 
  (h_intersects_sides : intersects_lines_sides V A1 B1 C1 A B C A2 B2 C2) :
  volume_tetrahedron V A2 B2 C2 ≤ (1/4) * volume_tetrahedron V A B C :=
sorry

theorem volume_of_sub_tetrahedron (V A B C: Point) 
  (M V1 : Point) (A1 B1 C1 : Point)
  (h_barycenter : is_barycenter M V A B C)
  (h_intersects_planes : intersects_lines_planes M V A B C A1 B1 C1)
  (h_point_V1 : intersects_something_to_find_V1) : 
  volume_tetrahedron V1 A1 B1 C1 = (1/4) * volume_tetrahedron V A B C :=
sorry

end volume_le_one_fourth_of_original_volume_of_sub_tetrahedron_l13_1331


namespace product_of_y_values_l13_1389

theorem product_of_y_values (y : ℝ) (h : abs (2 * y * 3) + 5 = 47) :
  ∃ y1 y2, (abs (2 * y1 * 3) + 5 = 47) ∧ (abs (2 * y2 * 3) + 5 = 47) ∧ y1 * y2 = -49 :=
by 
  sorry

end product_of_y_values_l13_1389


namespace calculate_AE_l13_1363

variable {k : ℝ} (A B C D E : Type*)

namespace Geometry

def shared_angle (A B C : Type*) : Prop := sorry -- assumes triangles share angle A

def prop_constant_proportion (AB AC AD AE : ℝ) (k : ℝ) : Prop :=
  AB * AC = k * AD * AE

theorem calculate_AE
  (A B C D E : Type*) 
  (AB AC AD AE : ℝ)
  (h_shared : shared_angle A B C)
  (h_AB : AB = 5)
  (h_AC : AC = 7)
  (h_AD : AD = 2)
  (h_proportion : prop_constant_proportion AB AC AD AE k)
  (h_k : k = 1) :
  AE = 17.5 := 
sorry

end Geometry

end calculate_AE_l13_1363


namespace yard_length_l13_1372

theorem yard_length
  (trees : ℕ) (gaps : ℕ) (distance_between_trees : ℕ) :
  trees = 26 → 
  gaps = trees - 1 → 
  distance_between_trees = 14 → 
  length_of_yard = gaps * distance_between_trees → 
  length_of_yard = 350 :=
by
  intros h_trees h_gaps h_distance h_length
  sorry

end yard_length_l13_1372


namespace neg_p_range_of_x_neg_q_sufficient_not_necessary_for_neg_p_l13_1305

def p (x : ℝ) : Prop := (x^2 - x - 2) ≤ 0
def q (x m : ℝ) : Prop := (x^2 - x - m^2 - m) ≤ 0

theorem neg_p_range_of_x (x : ℝ) : ¬ p x → x > 2 ∨ x < -1 :=
by
-- proof steps here
sorry

theorem neg_q_sufficient_not_necessary_for_neg_p (m : ℝ) : 
  (∀ x, ¬ q x m → ¬ p x) ∧ (∃ x, p x → ¬ q x m) → m > 1 ∨ m < -2 :=
by
-- proof steps here
sorry

end neg_p_range_of_x_neg_q_sufficient_not_necessary_for_neg_p_l13_1305


namespace most_marbles_l13_1371

def total_marbles := 24
def red_marble_fraction := 1 / 4
def red_marbles := red_marble_fraction * total_marbles
def blue_marbles := red_marbles + 6
def yellow_marbles := total_marbles - red_marbles - blue_marbles

theorem most_marbles : blue_marbles > red_marbles ∧ blue_marbles > yellow_marbles :=
by
  sorry

end most_marbles_l13_1371


namespace sum_of_powers_modulo_seven_l13_1300

theorem sum_of_powers_modulo_seven :
  ((1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7) % 7) = 1 := by
  sorry

end sum_of_powers_modulo_seven_l13_1300


namespace mean_identity_example_l13_1327

theorem mean_identity_example {x y z : ℝ} 
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 343)
  (h3 : x * y + y * z + z * x = 257.25) :
  x^2 + y^2 + z^2 = 385.5 :=
by
  sorry

end mean_identity_example_l13_1327


namespace tan_alpha_minus_beta_alpha_plus_beta_l13_1325

variable (α β : ℝ)

-- Conditions as hypotheses
axiom tan_alpha : Real.tan α = 2
axiom tan_beta : Real.tan β = -1 / 3
axiom alpha_range : 0 < α ∧ α < Real.pi / 2
axiom beta_range : Real.pi / 2 < β ∧ β < Real.pi

-- Proof statements
theorem tan_alpha_minus_beta : Real.tan (α - β) = 7 := by
  sorry

theorem alpha_plus_beta : α + β = 5 * Real.pi / 4 := by
  sorry

end tan_alpha_minus_beta_alpha_plus_beta_l13_1325


namespace inv_of_15_mod_1003_l13_1322

theorem inv_of_15_mod_1003 : ∃ x : ℕ, x ≤ 1002 ∧ 15 * x ≡ 1 [MOD 1003] ∧ x = 937 :=
by sorry

end inv_of_15_mod_1003_l13_1322


namespace pony_average_speed_l13_1375

theorem pony_average_speed
  (time_head_start : ℝ)
  (time_catch : ℝ)
  (horse_speed : ℝ)
  (distance_covered_by_horse : ℝ)
  (distance_covered_by_pony : ℝ)
  (pony's_head_start : ℝ)
  : (time_head_start = 3) → (time_catch = 4) → (horse_speed = 35) → 
    (distance_covered_by_horse = horse_speed * time_catch) → 
    (pony's_head_start = time_head_start * v) → 
    (distance_covered_by_pony = pony's_head_start + (v * time_catch)) → 
    (distance_covered_by_horse = distance_covered_by_pony) → v = 20 :=
  by 
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end pony_average_speed_l13_1375


namespace time_after_9876_seconds_l13_1338

noncomputable def currentTime : Nat := 2 * 3600 + 45 * 60 + 0
noncomputable def futureDuration : Nat := 9876
noncomputable def resultingTime : Nat := 5 * 3600 + 29 * 60 + 36

theorem time_after_9876_seconds : 
  (currentTime + futureDuration) % (24 * 3600) = resultingTime := 
by 
  sorry

end time_after_9876_seconds_l13_1338


namespace depth_of_channel_l13_1394

noncomputable def trapezium_area (a b h : ℝ) : ℝ :=
1/2 * (a + b) * h

theorem depth_of_channel :
  ∃ h : ℝ, trapezium_area 12 8 h = 700 ∧ h = 70 :=
by
  use 70
  unfold trapezium_area
  sorry

end depth_of_channel_l13_1394


namespace geric_bills_l13_1366

variable (G K J : ℕ)

theorem geric_bills (h1 : G = 2 * K) 
                    (h2 : K = J - 2) 
                    (h3 : J = 7 + 3) : 
    G = 16 := by
  sorry

end geric_bills_l13_1366


namespace Alice_min_speed_l13_1368

theorem Alice_min_speed
  (distance : Real := 120)
  (bob_speed : Real := 40)
  (alice_delay : Real := 0.5)
  (alice_min_speed : Real := distance / (distance / bob_speed - alice_delay)) :
  alice_min_speed = 48 := 
by
  sorry

end Alice_min_speed_l13_1368


namespace books_distribution_l13_1307

noncomputable def distribution_ways : ℕ :=
  let books := 5
  let people := 4
  let combination := Nat.choose books 2
  let arrangement := Nat.factorial people
  combination * arrangement ^ people

theorem books_distribution : distribution_ways = 240 := by
  sorry

end books_distribution_l13_1307


namespace find_number_l13_1336

theorem find_number (x : ℝ) (h : x + 33 + 333 + 33.3 = 399.6) : x = 0.3 :=
by
  sorry

end find_number_l13_1336


namespace debbys_sister_candy_l13_1308

-- Defining the conditions
def debby_candy : ℕ := 32
def eaten_candy : ℕ := 35
def remaining_candy : ℕ := 39

-- The proof problem
theorem debbys_sister_candy : ∃ S : ℕ, debby_candy + S - eaten_candy = remaining_candy → S = 42 :=
by
  sorry  -- The proof goes here

end debbys_sister_candy_l13_1308


namespace paving_stone_length_l13_1383

theorem paving_stone_length (courtyard_length courtyard_width paving_stone_width : ℝ)
  (num_paving_stones : ℕ)
  (courtyard_dims : courtyard_length = 40 ∧ courtyard_width = 20) 
  (paving_stone_dims : paving_stone_width = 2) 
  (num_stones : num_paving_stones = 100) 
  : (courtyard_length * courtyard_width) / (num_paving_stones * paving_stone_width) = 4 :=
by 
  sorry

end paving_stone_length_l13_1383


namespace subtraction_equality_l13_1385

theorem subtraction_equality : 3.56 - 2.15 = 1.41 :=
by
  sorry

end subtraction_equality_l13_1385


namespace megatek_manufacturing_percentage_l13_1351

theorem megatek_manufacturing_percentage 
  (total_degrees : ℝ := 360)
  (manufacturing_degrees : ℝ := 18)
  (is_proportional : (manufacturing_degrees / total_degrees) * 100 = 5) :
  (manufacturing_degrees / total_degrees) * 100 = 5 := 
  by
  exact is_proportional

end megatek_manufacturing_percentage_l13_1351


namespace no_such_number_exists_l13_1390

theorem no_such_number_exists :
  ¬ ∃ n : ℕ, 529 < n ∧ n < 538 ∧ 16 ∣ n :=
by sorry

end no_such_number_exists_l13_1390


namespace billy_initial_lemon_heads_l13_1361

theorem billy_initial_lemon_heads (n f : ℕ) (h_friends : f = 6) (h_eat : n = 12) :
  f * n = 72 := 
by
  -- Proceed by proving the statement using Lean
  sorry

end billy_initial_lemon_heads_l13_1361


namespace tan_70_sin_80_eq_neg1_l13_1319

theorem tan_70_sin_80_eq_neg1 :
  (Real.tan 70 * Real.sin 80 * (Real.sqrt 3 * Real.tan 20 - 1) = -1) :=
sorry

end tan_70_sin_80_eq_neg1_l13_1319


namespace cole_drive_time_correct_l13_1340

noncomputable def cole_drive_time : ℕ :=
  let distance_to_work := 45 -- derived from the given problem   
  let speed_to_work := 30
  let time_to_work := distance_to_work / speed_to_work -- in hours
  (time_to_work * 60 : ℕ) -- converting hours to minutes

theorem cole_drive_time_correct
  (speed_to_work speed_return: ℕ)
  (total_time: ℕ)
  (H1: speed_to_work = 30)
  (H2: speed_return = 90)
  (H3: total_time = 2):
  cole_drive_time = 90 := by
  -- Proof omitted
  sorry

end cole_drive_time_correct_l13_1340


namespace gcd_18_30_l13_1384

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l13_1384


namespace determine_pairs_l13_1377

open Int

-- Definitions corresponding to the conditions of the problem:
def is_prime (p : ℕ) : Prop := Nat.Prime p
def condition1 (p n : ℕ) : Prop := is_prime p
def condition2 (p n : ℕ) : Prop := n ≤ 2 * p
def condition3 (p n : ℕ) : Prop := (n^(p-1)) ∣ ((p-1)^n + 1)

-- Main theorem statement:
theorem determine_pairs (n p : ℕ) (h1 : condition1 p n) (h2 : condition2 p n) (h3 : condition3 p n) :
  (n = 1 ∧ is_prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
sorry

end determine_pairs_l13_1377


namespace find_c_for_circle_radius_five_l13_1397

theorem find_c_for_circle_radius_five
  (c : ℝ)
  (h : ∀ x y : ℝ, x^2 + 8 * x + y^2 + 2 * y + c = 0) :
  c = -8 :=
sorry

end find_c_for_circle_radius_five_l13_1397


namespace parallel_vectors_angle_l13_1311

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (1 / 2, Real.sin α)
noncomputable def vec_b (α : ℝ) : ℝ × ℝ := (Real.sin α, 1)

theorem parallel_vectors_angle (α : ℝ) (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ (vec_a α).1 = k * (vec_b α).1 ∧ (vec_a α).2 = k * (vec_b α).2) (h_acute : 0 < α ∧ α < π / 2) :
  α = π / 4 :=
sorry

end parallel_vectors_angle_l13_1311


namespace focal_length_of_hyperbola_l13_1339

theorem focal_length_of_hyperbola (a b p: ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (p_pos : 0 < p) :
  (∃ (F V : ℝ × ℝ), 4 = dist F V ∧ F = (2, 0) ∧ V = (-2, 0)) ∧
  (∃ (P : ℝ × ℝ), P = (-2, -1) ∧ (∃ (d : ℝ), d = d / 2 ∧ P = (d, 0))) →
  2 * (Real.sqrt (a^2 + b^2)) = 2 * Real.sqrt 5 := 
sorry

end focal_length_of_hyperbola_l13_1339
