import Mathlib

namespace NUMINAMATH_GPT_is_linear_equation_l14_1442

def quadratic_equation (x y : ℝ) : Prop := x * y + 2 * x = 7
def fractional_equation (x y : ℝ) : Prop := (1 / x) + y = 5
def quadratic_equation_2 (x y : ℝ) : Prop := x^2 + y = 2

def linear_equation (x y : ℝ) : Prop := 2 * x - y = 2

theorem is_linear_equation (x y : ℝ) (h1 : quadratic_equation x y) (h2 : fractional_equation x y) (h3 : quadratic_equation_2 x y) : linear_equation x y :=
  sorry

end NUMINAMATH_GPT_is_linear_equation_l14_1442


namespace NUMINAMATH_GPT_opposite_number_in_circle_l14_1439

theorem opposite_number_in_circle (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) (h3 : ∀ k, 1 ≤ k ∧ k ≤ 200 → ∃ m, (m = (k + 100) % 200) ∧ (m ≠ k) ∧ (n + 100) % 200 < k):
  ∃ m : ℕ, m = 114 ∧ (113 + 100) % 200 = m :=
by
  sorry

end NUMINAMATH_GPT_opposite_number_in_circle_l14_1439


namespace NUMINAMATH_GPT_ratio_of_areas_l14_1408

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ :=
1 / 2 * a * b

theorem ratio_of_areas (a b c x y z : ℝ)
  (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) 
  (h4 : x = 9) (h5 : y = 12) (h6 : z = 15)
  (h7 : a^2 + b^2 = c^2) (h8 : x^2 + y^2 = z^2) :
  (area_of_right_triangle a b) / (area_of_right_triangle x y) = 4 / 9 :=
sorry

end NUMINAMATH_GPT_ratio_of_areas_l14_1408


namespace NUMINAMATH_GPT_hyperbola_foci_condition_l14_1437

theorem hyperbola_foci_condition (m n : ℝ) (h : m * n > 0) :
    (m > 0 ∧ n > 0) ↔ ((∃ (x y : ℝ), m * x^2 - n * y^2 = 1) ∧ (∃ (x y : ℝ), m * x^2 - n * y^2 = 1)) :=
sorry

end NUMINAMATH_GPT_hyperbola_foci_condition_l14_1437


namespace NUMINAMATH_GPT_find_x_l14_1440

-- Defining the sum of integers from 30 to 40 inclusive
def sum_30_to_40 : ℕ := (30 + 31 + 32 + 33 + 34 + 35 + 36 + 37 + 38 + 39 + 40)

-- Defining the number of even integers from 30 to 40 inclusive
def count_even_30_to_40 : ℕ := 6

-- Given that x + y = 391, and y = count_even_30_to_40
-- Prove that x is equal to 385
theorem find_x (h : sum_30_to_40 + count_even_30_to_40 = 391) : sum_30_to_40 = 385 :=
by
  simp [sum_30_to_40, count_even_30_to_40] at h
  sorry

end NUMINAMATH_GPT_find_x_l14_1440


namespace NUMINAMATH_GPT_ratio_of_speeds_l14_1427

variable (b r : ℝ) (h1 : 1 / (b - r) = 2 * (1 / (b + r)))
variable (f1 f2 : ℝ) (h2 : b * (1/4) + b * (3/4) = b)

theorem ratio_of_speeds (b r : ℝ) (h1 : 1 / (b - r) = 2 * (1 / (b + r))) : b = 3 * r :=
by sorry

end NUMINAMATH_GPT_ratio_of_speeds_l14_1427


namespace NUMINAMATH_GPT_cost_difference_zero_l14_1453

theorem cost_difference_zero
  (A O X : ℝ)
  (h1 : 3 * A + 7 * O = 4.56)
  (h2 : A + O = 0.26)
  (h3 : O = A + X) :
  X = 0 := 
sorry

end NUMINAMATH_GPT_cost_difference_zero_l14_1453


namespace NUMINAMATH_GPT_triangle_inequality_l14_1418

variables (a b c S : ℝ) (S_def : S = (a + b + c) / 2)

theorem triangle_inequality 
  (h_triangle: a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 * S * (Real.sqrt (S - a) + Real.sqrt (S - b) + Real.sqrt (S - c)) 
  ≤ 3 * (Real.sqrt (b * c * (S - a)) + Real.sqrt (c * a * (S - b)) + Real.sqrt (a * b * (S - c))) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l14_1418


namespace NUMINAMATH_GPT_number_of_subsets_with_four_adj_chairs_l14_1481

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end NUMINAMATH_GPT_number_of_subsets_with_four_adj_chairs_l14_1481


namespace NUMINAMATH_GPT_choir_members_count_l14_1499

theorem choir_members_count : 
  ∃ n : ℕ, 120 ≤ n ∧ n ≤ 300 ∧
    n % 6 = 1 ∧
    n % 8 = 5 ∧
    n % 9 = 2 ∧
    n = 241 :=
by
  -- Proof will follow
  sorry

end NUMINAMATH_GPT_choir_members_count_l14_1499


namespace NUMINAMATH_GPT_best_years_to_scrap_l14_1487

-- Define the conditions from the problem
def purchase_cost : ℕ := 150000
def annual_cost : ℕ := 15000
def maintenance_initial : ℕ := 3000
def maintenance_difference : ℕ := 3000

-- Define the total_cost function
def total_cost (n : ℕ) : ℕ :=
  purchase_cost + annual_cost * n + (n * (2 * maintenance_initial + (n - 1) * maintenance_difference)) / 2

-- Define the average annual cost function
def average_annual_cost (n : ℕ) : ℕ :=
  total_cost n / n

-- Statement to be proven: the best number of years to minimize average annual cost is 10
theorem best_years_to_scrap : 
  (∀ n : ℕ, average_annual_cost 10 ≤ average_annual_cost n) :=
by
  sorry
  
end NUMINAMATH_GPT_best_years_to_scrap_l14_1487


namespace NUMINAMATH_GPT_total_weight_four_pets_l14_1488

-- Define the weights
def Evan_dog := 63
def Ivan_dog := Evan_dog / 7
def combined_weight_dogs := Evan_dog + Ivan_dog
def Kara_cat := combined_weight_dogs * 5
def combined_weight_dogs_and_cat := Evan_dog + Ivan_dog + Kara_cat
def Lisa_parrot := combined_weight_dogs_and_cat * 3
def total_weight := Evan_dog + Ivan_dog + Kara_cat + Lisa_parrot

-- Total weight of the four pets
theorem total_weight_four_pets : total_weight = 1728 := by
  sorry

end NUMINAMATH_GPT_total_weight_four_pets_l14_1488


namespace NUMINAMATH_GPT_part1_part2_l14_1475

def f (x a : ℝ) := abs (x - a)

theorem part1 (a : ℝ) :
  (∀ x : ℝ, (f x a) ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) → a = 3 :=
by
  intros h
  sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f (2 * x) 3 + f (x + 2) 3 ≥ m) → m ≤ 1 / 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_part1_part2_l14_1475


namespace NUMINAMATH_GPT_number_of_solutions_eq_two_l14_1452

theorem number_of_solutions_eq_two : 
  (∃ (x y : ℝ), x^2 - 3*x - 4 = 0 ∧ y^2 - 6*y + 9 = 0) ∧
  (∀ (x y : ℝ), (x^2 - 3*x - 4 = 0 ∧ y^2 - 6*y + 9 = 0) → ((x = 4 ∨ x = -1) ∧ y = 3)) :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_eq_two_l14_1452


namespace NUMINAMATH_GPT_combined_spots_l14_1458

-- Definitions of the conditions
def Rover_spots : ℕ := 46
def Cisco_spots : ℕ := Rover_spots / 2 - 5
def Granger_spots : ℕ := 5 * Cisco_spots

-- The proof statement
theorem combined_spots :
  Granger_spots + Cisco_spots = 108 := by
  sorry

end NUMINAMATH_GPT_combined_spots_l14_1458


namespace NUMINAMATH_GPT_geom_seq_product_a2_a3_l14_1429

theorem geom_seq_product_a2_a3 :
  ∃ (a_n : ℕ → ℝ), (a_n 1 * a_n 4 = -3) ∧ (∀ n, a_n n = a_n 1 * (a_n 2 / a_n 1) ^ (n - 1)) → a_n 2 * a_n 3 = -3 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_product_a2_a3_l14_1429


namespace NUMINAMATH_GPT_time_for_B_alone_l14_1448

theorem time_for_B_alone (W_A W_B : ℝ) (h1 : W_A = 2 * W_B) (h2 : W_A + W_B = 1/6) : 1 / W_B = 18 := by
  sorry

end NUMINAMATH_GPT_time_for_B_alone_l14_1448


namespace NUMINAMATH_GPT_remainder_8437_by_9_l14_1432

theorem remainder_8437_by_9 : 8437 % 9 = 4 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_remainder_8437_by_9_l14_1432


namespace NUMINAMATH_GPT_smallest_positive_x_l14_1464

theorem smallest_positive_x (x : ℝ) (h : x > 0) (h_eq : x / 4 + 3 / (4 * x) = 1) : x = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_x_l14_1464


namespace NUMINAMATH_GPT_not_lt_neg_version_l14_1406

theorem not_lt_neg_version (a b : ℝ) (h : a < b) : ¬ (-3 * a < -3 * b) :=
by 
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_not_lt_neg_version_l14_1406


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l14_1468

variable (A B C : ℝ × ℝ)
variable (x1 y1 x2 y2 x3 y3 : ℝ)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem area_of_triangle_ABC :
  let A := (1, 2)
  let B := (-2, 5)
  let C := (4, -2)
  area_of_triangle A B C = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l14_1468


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l14_1443

theorem sum_of_squares_of_roots (x_1 x_2 : ℚ) (h1 : 6 * x_1^2 - 13 * x_1 + 5 = 0)
                                (h2 : 6 * x_2^2 - 13 * x_2 + 5 = 0) 
                                (h3 : x_1 ≠ x_2) :
  x_1^2 + x_2^2 = 109 / 36 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l14_1443


namespace NUMINAMATH_GPT_brownie_pieces_count_l14_1423

def pan_width : ℕ := 24
def pan_height : ℕ := 15
def brownie_width : ℕ := 3
def brownie_height : ℕ := 2

theorem brownie_pieces_count : (pan_width * pan_height) / (brownie_width * brownie_height) = 60 := by
  sorry

end NUMINAMATH_GPT_brownie_pieces_count_l14_1423


namespace NUMINAMATH_GPT_probability_selecting_cooking_l14_1460

theorem probability_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let favorable_outcomes := 1
  let total_outcomes := courses.length
  let probability := favorable_outcomes / total_outcomes
  probability = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_selecting_cooking_l14_1460


namespace NUMINAMATH_GPT_problem_statement_l14_1496

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

-- Conditions of the problem
def cond1 : Prop := (1 / x) + (1 / y) = 2
def cond2 : Prop := (x * y) + x - y = 6

-- The corresponding theorem to prove: x² - y² = 2
theorem problem_statement (h1 : cond1) (h2 : cond2) : x^2 - y^2 = 2 :=
  sorry

end NUMINAMATH_GPT_problem_statement_l14_1496


namespace NUMINAMATH_GPT_green_square_area_percentage_l14_1459

variable (s a : ℝ)
variable (h : a^2 + 4 * a * (s - 2 * a) = 0.49 * s^2)

theorem green_square_area_percentage :
  (a^2 / s^2) = 0.1225 :=
sorry

end NUMINAMATH_GPT_green_square_area_percentage_l14_1459


namespace NUMINAMATH_GPT_sum_of_first_100_digits_of_1_div_2222_l14_1404

theorem sum_of_first_100_digits_of_1_div_2222 : 
  (let repeating_block := [0, 0, 0, 4, 5];
  let sum_of_digits (lst : List ℕ) := lst.sum;
  let block_sum := sum_of_digits repeating_block;
  let num_blocks := 100 / 5;
  num_blocks * block_sum = 180) :=
by 
  let repeating_block := [0, 0, 0, 4, 5]
  let sum_of_digits (lst : List ℕ) := lst.sum
  let block_sum := sum_of_digits repeating_block
  let num_blocks := 100 / 5
  have h : num_blocks * block_sum = 180 := sorry
  exact h

end NUMINAMATH_GPT_sum_of_first_100_digits_of_1_div_2222_l14_1404


namespace NUMINAMATH_GPT_division_of_power_l14_1444

theorem division_of_power (m : ℕ) 
  (h : m = 16^2018) : m / 8 = 2^8069 := by
  sorry

end NUMINAMATH_GPT_division_of_power_l14_1444


namespace NUMINAMATH_GPT_equivalent_fraction_power_multiplication_l14_1409

theorem equivalent_fraction_power_multiplication : 
  (8 / 9) ^ 2 * (1 / 3) ^ 2 * (2 / 5) = (128 / 3645) := 
by 
  sorry

end NUMINAMATH_GPT_equivalent_fraction_power_multiplication_l14_1409


namespace NUMINAMATH_GPT_net_percentage_gain_approx_l14_1420

noncomputable def netPercentageGain : ℝ :=
  let costGlassBowls := 250 * 18
  let costCeramicPlates := 150 * 25
  let totalCostBeforeDiscount := costGlassBowls + costCeramicPlates
  let discount := 0.05 * totalCostBeforeDiscount
  let totalCostAfterDiscount := totalCostBeforeDiscount - discount
  let revenueGlassBowls := 200 * 25
  let revenueCeramicPlates := 120 * 32
  let totalRevenue := revenueGlassBowls + revenueCeramicPlates
  let costBrokenGlassBowls := 30 * 18
  let costBrokenCeramicPlates := 10 * 25
  let totalCostBrokenItems := costBrokenGlassBowls + costBrokenCeramicPlates
  let netGain := totalRevenue - (totalCostAfterDiscount + totalCostBrokenItems)
  let netPercentageGain := (netGain / totalCostAfterDiscount) * 100
  netPercentageGain

theorem net_percentage_gain_approx :
  abs (netPercentageGain - 2.71) < 0.01 := sorry

end NUMINAMATH_GPT_net_percentage_gain_approx_l14_1420


namespace NUMINAMATH_GPT_golden_state_total_points_l14_1455

theorem golden_state_total_points :
  ∀ (Draymond Curry Kelly Durant Klay : ℕ),
  Draymond = 12 →
  Curry = 2 * Draymond →
  Kelly = 9 →
  Durant = 2 * Kelly →
  Klay = Draymond / 2 →
  Draymond + Curry + Kelly + Durant + Klay = 69 :=
by
  intros Draymond Curry Kelly Durant Klay
  intros hD hC hK hD2 hK2
  rw [hD, hC, hK, hD2, hK2]
  sorry

end NUMINAMATH_GPT_golden_state_total_points_l14_1455


namespace NUMINAMATH_GPT_equation_of_line_l_l14_1431

theorem equation_of_line_l :
  (∃ l : ℝ → ℝ → Prop, 
     (∀ x y, l x y ↔ (x - y + 3) = 0)
     ∧ (∀ x y, l x y → x^2 + (y - 3)^2 = 4)
     ∧ (∀ x y, l x y → x + y + 1 = 0)) :=
sorry

end NUMINAMATH_GPT_equation_of_line_l_l14_1431


namespace NUMINAMATH_GPT_combined_original_price_of_books_l14_1493

theorem combined_original_price_of_books (p1 p2 : ℝ) (h1 : p1 / 8 = 8) (h2 : p2 / 9 = 9) :
  p1 + p2 = 145 :=
sorry

end NUMINAMATH_GPT_combined_original_price_of_books_l14_1493


namespace NUMINAMATH_GPT_total_amount_spent_l14_1401

theorem total_amount_spent : 
  let value_half_dollar := 0.50
  let wednesday_spending := 4 * value_half_dollar
  let next_day_spending := 14 * value_half_dollar
  wednesday_spending + next_day_spending = 9.00 :=
by
  let value_half_dollar := 0.50
  let wednesday_spending := 4 * value_half_dollar
  let next_day_spending := 14 * value_half_dollar
  show _ 
  sorry

end NUMINAMATH_GPT_total_amount_spent_l14_1401


namespace NUMINAMATH_GPT_div_by_5_l14_1467

theorem div_by_5 (a b : ℕ) (h: 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by
  -- Proof by contradiction
  -- Assume the negation of the conclusion
  have h_nand : ¬ (5 ∣ a) ∧ ¬ (5 ∣ b) := sorry

  -- Derive a contradiction based on the assumptions
  sorry

end NUMINAMATH_GPT_div_by_5_l14_1467


namespace NUMINAMATH_GPT_difference_of_interchanged_digits_l14_1474

theorem difference_of_interchanged_digits (X Y : ℕ) (h : X - Y = 5) : (10 * X + Y) - (10 * Y + X) = 45 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_interchanged_digits_l14_1474


namespace NUMINAMATH_GPT_find_lambda_l14_1445

noncomputable def vec_length (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

theorem find_lambda {a b : ℝ × ℝ} (lambda : ℝ) 
  (ha : vec_length a = 1) (hb : vec_length b = 2)
  (hab_angle : dot_product a b = -1) 
  (h_perp : dot_product (lambda • a + b) (a - 2 • b) = 0) : 
  lambda = 3 := 
sorry

end NUMINAMATH_GPT_find_lambda_l14_1445


namespace NUMINAMATH_GPT_factorize_cubic_l14_1413

theorem factorize_cubic (a : ℝ) : a^3 - 16 * a = a * (a + 4) * (a - 4) :=
sorry

end NUMINAMATH_GPT_factorize_cubic_l14_1413


namespace NUMINAMATH_GPT_cost_of_adult_ticket_is_8_l14_1411

variables (A : ℕ) (num_people : ℕ := 22) (total_money : ℕ := 50) (num_children : ℕ := 18) (child_ticket_cost : ℕ := 1)

-- Definitions based on the given conditions
def child_tickets_cost := num_children * child_ticket_cost
def num_adults := num_people - num_children
def adult_tickets_cost := total_money - child_tickets_cost
def cost_per_adult_ticket := adult_tickets_cost / num_adults

-- The theorem stating that the cost of an adult ticket is 8 dollars
theorem cost_of_adult_ticket_is_8 : cost_per_adult_ticket = 8 :=
by sorry

end NUMINAMATH_GPT_cost_of_adult_ticket_is_8_l14_1411


namespace NUMINAMATH_GPT_problem1_problem2_l14_1462

-- First Problem Statement:
theorem problem1 :  12 - (-18) + (-7) - 20 = 3 := 
by 
  sorry

-- Second Problem Statement:
theorem problem2 : -4 / (1 / 2) * 8 = -64 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l14_1462


namespace NUMINAMATH_GPT_roots_polynomial_sum_pow_l14_1485

open Real

theorem roots_polynomial_sum_pow (a b : ℝ) (h : a^2 - 5 * a + 6 = 0) (h_b : b^2 - 5 * b + 6 = 0) :
  a^5 + a^4 * b + b^5 = -16674 := by
sorry

end NUMINAMATH_GPT_roots_polynomial_sum_pow_l14_1485


namespace NUMINAMATH_GPT_people_got_off_at_second_stop_l14_1484

theorem people_got_off_at_second_stop (x : ℕ) :
  (10 - x) + 20 - 18 + 2 = 12 → x = 2 :=
  by sorry

end NUMINAMATH_GPT_people_got_off_at_second_stop_l14_1484


namespace NUMINAMATH_GPT_fourth_root_sum_of_square_roots_eq_l14_1489

theorem fourth_root_sum_of_square_roots_eq :
  (1 + Real.sqrt 2 + Real.sqrt 3) = 
    Real.sqrt (Real.sqrt 6400 + Real.sqrt 6144 + Real.sqrt 4800 + Real.sqrt 4608) ^ 4 :=
by
  sorry

end NUMINAMATH_GPT_fourth_root_sum_of_square_roots_eq_l14_1489


namespace NUMINAMATH_GPT_range_of_a_l14_1436

theorem range_of_a (a : ℝ) (h : (2 - a)^3 > (a - 1)^3) : a < 3/2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l14_1436


namespace NUMINAMATH_GPT_original_profit_percentage_l14_1470

theorem original_profit_percentage (C : ℝ) (C' : ℝ) (S' : ℝ) (H1 : C = 40) (H2 : C' = 32) (H3 : S' = 41.60) 
  (H4 : S' = (1.30 * C')) : (S' + 8.40 - C) / C * 100 = 25 := 
by 
  sorry

end NUMINAMATH_GPT_original_profit_percentage_l14_1470


namespace NUMINAMATH_GPT_students_neither_cool_l14_1430

variable (total_students : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ)

def only_cool_dads := cool_dads - both_cool
def only_cool_moms := cool_moms - both_cool
def only_cool := only_cool_dads + only_cool_moms + both_cool
def neither_cool := total_students - only_cool

theorem students_neither_cool (h1 : total_students = 40) (h2 : cool_dads = 18) (h3 : cool_moms = 22) (h4 : both_cool = 10) 
: neither_cool total_students cool_dads cool_moms both_cool = 10 :=
by 
  sorry

end NUMINAMATH_GPT_students_neither_cool_l14_1430


namespace NUMINAMATH_GPT_circumference_ratio_l14_1407

theorem circumference_ratio (C D : ℝ) (hC : C = 94.2) (hD : D = 30) : C / D = 3.14 :=
by {
  sorry
}

end NUMINAMATH_GPT_circumference_ratio_l14_1407


namespace NUMINAMATH_GPT_total_pairs_purchased_l14_1434

-- Define the conditions as hypotheses
def foxPrice : ℝ := 15
def ponyPrice : ℝ := 18
def totalSaved : ℝ := 8.91
def foxPairs : ℕ := 3
def ponyPairs : ℕ := 2
def sumDiscountRates : ℝ := 0.22
def ponyDiscountRate : ℝ := 0.10999999999999996

-- Prove that the total number of pairs of jeans purchased is 5
theorem total_pairs_purchased : foxPairs + ponyPairs = 5 := by
  sorry

end NUMINAMATH_GPT_total_pairs_purchased_l14_1434


namespace NUMINAMATH_GPT_find_a_b_find_extreme_values_l14_1469

-- Definitions based on the conditions in the problem
def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 2 * b

-- The function f attains a maximum value of 2 at x = -1
def f_max_at_neg_1 (a b : ℝ) : Prop :=
  (∃ x : ℝ, x = -1 ∧ 
  (∀ y : ℝ, f x a b ≤ f y a b)) ∧ f (-1) a b = 2

-- Statement (1): Finding the values of a and b
theorem find_a_b : ∃ a b : ℝ, f_max_at_neg_1 a b ∧ a = 2 ∧ b = 1 :=
sorry

-- The function f with a=2 and b=1
def f_specific (x : ℝ) : ℝ := f x 2 1

-- Statement (2): Finding the extreme values of f(x) on the interval [-1, 1]
def extreme_values_on_interval : Prop :=
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f_specific x ≤ 6 ∧ f_specific x ≥ 50/27) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f_specific x = 6) ∧ 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f_specific x = 50/27)

theorem find_extreme_values : extreme_values_on_interval :=
sorry

end NUMINAMATH_GPT_find_a_b_find_extreme_values_l14_1469


namespace NUMINAMATH_GPT_angela_spent_78_l14_1477

-- Definitions
def angela_initial_money : ℕ := 90
def angela_left_money : ℕ := 12
def angela_spent_money : ℕ := angela_initial_money - angela_left_money

-- Theorem statement
theorem angela_spent_78 : angela_spent_money = 78 := by
  -- Proof would go here, but it is not required.
  sorry

end NUMINAMATH_GPT_angela_spent_78_l14_1477


namespace NUMINAMATH_GPT_maximum_withdraw_l14_1491

theorem maximum_withdraw (initial_amount withdraw deposit : ℕ) (h_initial : initial_amount = 500)
    (h_withdraw : withdraw = 300) (h_deposit : deposit = 198) :
    ∃ x y : ℕ, initial_amount - x * withdraw + y * deposit ≥ 0 ∧ initial_amount - x * withdraw + y * deposit = 194 ∧ initial_amount - x * withdraw = 300 := sorry

end NUMINAMATH_GPT_maximum_withdraw_l14_1491


namespace NUMINAMATH_GPT_gcd_sequence_inequality_l14_1416

-- Add your Lean 4 statement here
theorem gcd_sequence_inequality {n : ℕ} 
  (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 35 → Nat.gcd n k < Nat.gcd n (k+1)) : 
  Nat.gcd n 35 < Nat.gcd n 36 := 
sorry

end NUMINAMATH_GPT_gcd_sequence_inequality_l14_1416


namespace NUMINAMATH_GPT_DVDs_sold_is_168_l14_1451

variables (C D : ℕ)
variables (h1 : D = (16 * C) / 10)
variables (h2 : D + C = 273)

theorem DVDs_sold_is_168 : D = 168 := by
  sorry

end NUMINAMATH_GPT_DVDs_sold_is_168_l14_1451


namespace NUMINAMATH_GPT_tumblers_count_correct_l14_1497

section MrsPetersonsTumblers

-- Define the cost of one tumbler
def tumbler_cost : ℕ := 45

-- Define the amount paid in total by Mrs. Petersons
def total_paid : ℕ := 5 * 100

-- Define the change received by Mrs. Petersons
def change_received : ℕ := 50

-- Calculate the total amount spent
def total_spent : ℕ := total_paid - change_received

-- Calculate the number of tumblers bought
def tumblers_bought : ℕ := total_spent / tumbler_cost

-- Prove the number of tumblers bought is 10
theorem tumblers_count_correct : tumblers_bought = 10 :=
  by
    -- Proof steps will be filled here
    sorry

end MrsPetersonsTumblers

end NUMINAMATH_GPT_tumblers_count_correct_l14_1497


namespace NUMINAMATH_GPT_thirty_percent_less_than_ninety_eq_one_fourth_more_than_n_l14_1435

theorem thirty_percent_less_than_ninety_eq_one_fourth_more_than_n (n : ℝ) :
  0.7 * 90 = (5 / 4) * n → n = 50.4 :=
by sorry

end NUMINAMATH_GPT_thirty_percent_less_than_ninety_eq_one_fourth_more_than_n_l14_1435


namespace NUMINAMATH_GPT_part_1_part_2_l14_1422

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x + a) + 2 * a

theorem part_1 (h : ∀ x : ℝ, f x a = f (3 - x) a) : a = -3 :=
by
  sorry

theorem part_2 (h : ∃ x : ℝ, f x a ≤ -abs (2 * x - 1) + a) : a ≤ -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_part_1_part_2_l14_1422


namespace NUMINAMATH_GPT_max_x_minus_y_isosceles_l14_1466

theorem max_x_minus_y_isosceles (x y : ℝ) (hx : x ≠ 50) (hy : y ≠ 50) 
  (h_iso1 : x = y ∨ 50 = y) (h_iso2 : x = y ∨ 50 = x)
  (h_triangle : 50 + x + y = 180) : 
  max (x - y) (y - x) = 30 :=
sorry

end NUMINAMATH_GPT_max_x_minus_y_isosceles_l14_1466


namespace NUMINAMATH_GPT_simplify_expression_l14_1472

theorem simplify_expression (p q x : ℝ) (h₀ : p ≠ 0) (h₁ : q ≠ 0) (h₂ : x > 0) (h₃ : x ≠ 1) :
  (x^(3 / p) - x^(3 / q)) / ((x^(1 / p) + x^(1 / q))^2 - 2 * x^(1 / q) * (x^(1 / q) + x^(1 / p)))
  + x^(1 / p) / (x^((q - p) / (p * q)) + 1) = x^(1 / p) + x^(1 / q) := 
sorry

end NUMINAMATH_GPT_simplify_expression_l14_1472


namespace NUMINAMATH_GPT_solve_for_x_l14_1473

theorem solve_for_x (x : ℚ) (h : 3 / 4 - 1 / x = 1 / 2) : x = 4 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l14_1473


namespace NUMINAMATH_GPT_coffee_grinder_assembly_time_l14_1405

-- Variables for the assembly rates
variables (h r : ℝ)

-- Definitions of conditions
def condition1 : Prop := h / 4 = r
def condition2 : Prop := r / 4 = h
def condition3 : Prop := ∀ start_time end_time net_added, 
  start_time = 9 ∧ end_time = 12 ∧ net_added = 27 → 3 * 3/4 * h = net_added
def condition4 : Prop := ∀ start_time end_time net_added, 
  start_time = 13 ∧ end_time = 19 ∧ net_added = 120 → 6 * 3/4 * r = net_added

-- Theorem statement
theorem coffee_grinder_assembly_time
  (h r : ℝ)
  (c1 : condition1 h r)
  (c2 : condition2 h r)
  (c3 : condition3 h)
  (c4 : condition4 r) :
  h = 12 ∧ r = 80 / 3 :=
sorry

end NUMINAMATH_GPT_coffee_grinder_assembly_time_l14_1405


namespace NUMINAMATH_GPT_range_independent_variable_l14_1463

noncomputable def range_of_independent_variable (x : ℝ) : Prop :=
  x ≠ 3

theorem range_independent_variable (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) → x ≠ 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_independent_variable_l14_1463


namespace NUMINAMATH_GPT_average_rainfall_february_1964_l14_1471

theorem average_rainfall_february_1964 :
  let total_rainfall := 280
  let days_february := 29
  let hours_per_day := 24
  (total_rainfall / (days_february * hours_per_day)) = (280 / (29 * 24)) :=
by
  sorry

end NUMINAMATH_GPT_average_rainfall_february_1964_l14_1471


namespace NUMINAMATH_GPT_value_at_1971_l14_1433

def sequence_x (x : ℕ → ℝ) :=
  ∀ n > 1, 3 * x n - x (n - 1) = n

theorem value_at_1971 (x : ℕ → ℝ) (hx : sequence_x x) (h_initial : abs (x 1) < 1971) :
  abs (x 1971 - 985.25) < 0.000001 :=
by sorry

end NUMINAMATH_GPT_value_at_1971_l14_1433


namespace NUMINAMATH_GPT_pythagorean_triple_345_l14_1498

theorem pythagorean_triple_345 : (3^2 + 4^2 = 5^2) := 
by 
  -- Here, the proof will be filled in, but we use 'sorry' for now.
  sorry

end NUMINAMATH_GPT_pythagorean_triple_345_l14_1498


namespace NUMINAMATH_GPT_area_of_square_l14_1426

theorem area_of_square (side_length : ℝ) (h : side_length = 17) : side_length * side_length = 289 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_l14_1426


namespace NUMINAMATH_GPT_actual_selling_price_l14_1480

-- Define the original price m
variable (m : ℝ)

-- Define the discount rate
def discount_rate : ℝ := 0.2

-- Define the selling price
def selling_price := m * (1 - discount_rate)

-- The theorem states the relationship between the original price and the selling price after discount
theorem actual_selling_price : selling_price m = 0.8 * m :=
by
-- Proof step would go here
sorry

end NUMINAMATH_GPT_actual_selling_price_l14_1480


namespace NUMINAMATH_GPT_ara_current_height_l14_1461

variable (h : ℝ)  -- Original height of both Shea and Ara
variable (sheas_growth_rate : ℝ := 0.20)  -- Shea's growth rate (20%)
variable (sheas_current_height : ℝ := 60)  -- Shea's current height
variable (aras_growth_rate : ℝ := 0.5)  -- Ara's growth rate in terms of Shea's growth

theorem ara_current_height : 
  h * (1 + sheas_growth_rate) = sheas_current_height →
  (h + (sheas_current_height - h) * aras_growth_rate) = 55 :=
  by
    sorry

end NUMINAMATH_GPT_ara_current_height_l14_1461


namespace NUMINAMATH_GPT_junior_titles_in_sample_l14_1414

noncomputable def numberOfJuniorTitlesInSample (totalEmployees: ℕ) (juniorEmployees: ℕ) (sampleSize: ℕ) : ℕ :=
  (juniorEmployees * sampleSize) / totalEmployees

theorem junior_titles_in_sample (totalEmployees juniorEmployees intermediateEmployees seniorEmployees sampleSize : ℕ) 
  (h_total : totalEmployees = 150) 
  (h_junior : juniorEmployees = 90) 
  (h_intermediate : intermediateEmployees = 45) 
  (h_senior : seniorEmployees = 15) 
  (h_sampleSize : sampleSize = 30) : 
  numberOfJuniorTitlesInSample totalEmployees juniorEmployees sampleSize = 18 := by
  sorry

end NUMINAMATH_GPT_junior_titles_in_sample_l14_1414


namespace NUMINAMATH_GPT_books_per_shelf_l14_1492

def initial_coloring_books : ℕ := 86
def sold_coloring_books : ℕ := 37
def shelves : ℕ := 7

theorem books_per_shelf : (initial_coloring_books - sold_coloring_books) / shelves = 7 := by
  sorry

end NUMINAMATH_GPT_books_per_shelf_l14_1492


namespace NUMINAMATH_GPT_exp_neg_eq_l14_1450

theorem exp_neg_eq (θ φ : ℝ) (h : Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1 / 2 : ℂ) + (1 / 3 : ℂ) * Complex.I) :
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1 / 2 : ℂ) - (1 / 3 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_GPT_exp_neg_eq_l14_1450


namespace NUMINAMATH_GPT_ernaldo_friends_count_l14_1402

-- Define the members of the group
inductive Member
| Arnaldo
| Bernaldo
| Cernaldo
| Dernaldo
| Ernaldo

open Member

-- Define the number of friends for each member
def number_of_friends : Member → ℕ
| Arnaldo  => 1
| Bernaldo => 2
| Cernaldo => 3
| Dernaldo => 4
| Ernaldo  => 0  -- This will be our unknown to solve

-- The main theorem we need to prove
theorem ernaldo_friends_count : number_of_friends Ernaldo = 2 :=
sorry

end NUMINAMATH_GPT_ernaldo_friends_count_l14_1402


namespace NUMINAMATH_GPT_division_multiplication_calculation_l14_1438

theorem division_multiplication_calculation :
  (30 / (7 + 2 - 3)) * 4 = 20 :=
by
  sorry

end NUMINAMATH_GPT_division_multiplication_calculation_l14_1438


namespace NUMINAMATH_GPT_fraction_work_AC_l14_1449

theorem fraction_work_AC (total_payment Rs B_payment : ℝ)
  (payment_AC : ℝ)
  (h1 : total_payment = 529)
  (h2 : B_payment = 12)
  (h3 : payment_AC = total_payment - B_payment) : 
  payment_AC / total_payment = 517 / 529 :=
by
  rw [h1, h2] at h3
  rw [h3]
  norm_num
  sorry

end NUMINAMATH_GPT_fraction_work_AC_l14_1449


namespace NUMINAMATH_GPT_part1_part2_1_part2_2_l14_1482

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a - 2) * x + 4
noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x + b - 3) / (a * x^2 + 2)

theorem part1 (a : ℝ) (b : ℝ) :
  (∀ x, f x a = f (-x) a) → b = 3 :=
by sorry

theorem part2_1 (a : ℝ) (b : ℝ) :
  a = 2 → b = 3 →
  ∀ x₁ x₂, -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 ∧ x₁ < x₂ → g x₁ a b < g x₂ a b :=
by sorry

theorem part2_2 (a : ℝ) (b : ℝ) (t : ℝ) :
  a = 2 → b = 3 →
  g (t - 1) a b + g (2 * t) a b < 0 →
  0 < t ∧ t < 1 / 3 :=
by sorry

end NUMINAMATH_GPT_part1_part2_1_part2_2_l14_1482


namespace NUMINAMATH_GPT_translate_parabola_l14_1415

theorem translate_parabola (x : ℝ) :
  (∃ (h k : ℝ), h = 1 ∧ k = 3 ∧ ∀ x: ℝ, y = 2*x^2 → y = 2*(x - h)^2 + k) := 
by
  use 1, 3
  sorry

end NUMINAMATH_GPT_translate_parabola_l14_1415


namespace NUMINAMATH_GPT_iron_column_lifted_by_9_6_cm_l14_1424

namespace VolumeLift

def base_area_container : ℝ := 200
def base_area_column : ℝ := 40
def height_water : ℝ := 16
def distance_water_surface : ℝ := 4

theorem iron_column_lifted_by_9_6_cm :
  ∃ (h_lift : ℝ),
    h_lift = 9.6 ∧ height_water - distance_water_surface = 16 - h_lift :=
by
sorry

end VolumeLift

end NUMINAMATH_GPT_iron_column_lifted_by_9_6_cm_l14_1424


namespace NUMINAMATH_GPT_bike_price_l14_1465

theorem bike_price (x : ℝ) (h : 0.20 * x = 240) : x = 1200 :=
by
  sorry

end NUMINAMATH_GPT_bike_price_l14_1465


namespace NUMINAMATH_GPT_odd_function_five_value_l14_1457

variable (f : ℝ → ℝ)

theorem odd_function_five_value (h_odd : ∀ x : ℝ, f (-x) = -f x)
                               (h_f1 : f 1 = 1 / 2)
                               (h_f_recurrence : ∀ x : ℝ, f (x + 2) = f x + f 2) :
  f 5 = 5 / 2 :=
sorry

end NUMINAMATH_GPT_odd_function_five_value_l14_1457


namespace NUMINAMATH_GPT_person_half_Jordyn_age_is_6_l14_1410

variables (Mehki_age Jordyn_age certain_age : ℕ)
axiom h1 : Mehki_age = Jordyn_age + 10
axiom h2 : Jordyn_age = 2 * certain_age
axiom h3 : Mehki_age = 22

theorem person_half_Jordyn_age_is_6 : certain_age = 6 :=
by sorry

end NUMINAMATH_GPT_person_half_Jordyn_age_is_6_l14_1410


namespace NUMINAMATH_GPT_trains_total_distance_l14_1428

theorem trains_total_distance (speed_A speed_B : ℝ) (time_A time_B : ℝ) (dist_A dist_B : ℝ):
  speed_A = 90 ∧ 
  speed_B = 120 ∧ 
  time_A = 1 ∧ 
  time_B = 5/6 ∧ 
  dist_A = speed_A * time_A ∧ 
  dist_B = speed_B * time_B ->
  (dist_A + dist_B) = 190 :=
by 
  intros h
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  sorry

end NUMINAMATH_GPT_trains_total_distance_l14_1428


namespace NUMINAMATH_GPT_y_in_terms_of_x_l14_1494

theorem y_in_terms_of_x (p x y : ℝ) (h1 : x = 2 + 2^p) (h2 : y = 1 + 2^(-p)) : 
  y = (x-1)/(x-2) :=
by
  sorry

end NUMINAMATH_GPT_y_in_terms_of_x_l14_1494


namespace NUMINAMATH_GPT_find_LCM_of_three_numbers_l14_1403

noncomputable def LCM_of_three_numbers (a b c : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) c

theorem find_LCM_of_three_numbers
  (a b c : ℕ)
  (h_prod : a * b * c = 1354808)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 11) :
  LCM_of_three_numbers a b c = 123164 := by
  sorry

end NUMINAMATH_GPT_find_LCM_of_three_numbers_l14_1403


namespace NUMINAMATH_GPT_subset_iff_a_values_l14_1446

theorem subset_iff_a_values (a : ℝ) :
  let P := { x : ℝ | x^2 = 1 }
  let Q := { x : ℝ | a * x = 1 }
  Q ⊆ P ↔ a = 0 ∨ a = 1 ∨ a = -1 :=
by sorry

end NUMINAMATH_GPT_subset_iff_a_values_l14_1446


namespace NUMINAMATH_GPT_find_quotient_l14_1419

theorem find_quotient (D d R Q : ℤ) (hD : D = 729) (hd : d = 38) (hR : R = 7)
  (h : D = d * Q + R) : Q = 19 := by
  sorry

end NUMINAMATH_GPT_find_quotient_l14_1419


namespace NUMINAMATH_GPT_smallest_natural_with_50_perfect_squares_l14_1476

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_natural_with_50_perfect_squares_l14_1476


namespace NUMINAMATH_GPT_smallest_number_divisible_l14_1495

theorem smallest_number_divisible (x : ℕ) : 
  (∃ x, x + 7 % 8 = 0 ∧ x + 7 % 11 = 0 ∧ x + 7 % 24 = 0) ∧
  (∀ y, (y + 7 % 8 = 0 ∧ y + 7 % 11 = 0 ∧ y + 7 % 24 = 0) → 257 ≤ y) :=
by { sorry }

end NUMINAMATH_GPT_smallest_number_divisible_l14_1495


namespace NUMINAMATH_GPT_percentage_hindus_l14_1454

-- Conditions 
def total_boys : ℕ := 850
def percentage_muslims : ℝ := 0.44
def percentage_sikhs : ℝ := 0.10
def boys_other_communities : ℕ := 272

-- Question and proof statement
theorem percentage_hindus (total_boys : ℕ) (percentage_muslims percentage_sikhs : ℝ) (boys_other_communities : ℕ) : 
  (total_boys = 850) →
  (percentage_muslims = 0.44) →
  (percentage_sikhs = 0.10) →
  (boys_other_communities = 272) →
  ((850 - (374 + 85 + 272)) / 850) * 100 = 14 := 
by
  intros
  sorry

end NUMINAMATH_GPT_percentage_hindus_l14_1454


namespace NUMINAMATH_GPT_article_initial_cost_l14_1447

theorem article_initial_cost (x : ℝ) (h : 0.44 * x = 4400) : x = 10000 :=
by
  sorry

end NUMINAMATH_GPT_article_initial_cost_l14_1447


namespace NUMINAMATH_GPT_degree_of_divisor_l14_1425

theorem degree_of_divisor (f d q r : Polynomial ℝ) 
  (hf : f.degree = 15) 
  (hq : q.degree = 9) 
  (hr : r.degree = 4) 
  (hr_poly : r = (Polynomial.C 5) * (Polynomial.X^4) + (Polynomial.C 6) * (Polynomial.X^3) - (Polynomial.C 2) * (Polynomial.X) + (Polynomial.C 7)) 
  (hdiv : f = d * q + r) : 
  d.degree = 6 := 
sorry

end NUMINAMATH_GPT_degree_of_divisor_l14_1425


namespace NUMINAMATH_GPT_powderman_distance_when_blast_heard_l14_1456

-- Define constants
def fuse_time : ℝ := 30  -- seconds
def run_rate : ℝ := 8    -- yards per second
def sound_rate : ℝ := 1080  -- feet per second
def yards_to_feet : ℝ := 3  -- conversion factor

-- Define the time at which the blast was heard
noncomputable def blast_heard_time : ℝ := 675 / 22

-- Define distance functions
def p (t : ℝ) : ℝ := run_rate * yards_to_feet * t  -- distance run by powderman in feet
def q (t : ℝ) : ℝ := sound_rate * (t - fuse_time)  -- distance sound has traveled in feet

-- Proof statement: given the conditions, the distance run by the powderman equals 245 yards
theorem powderman_distance_when_blast_heard :
  p (blast_heard_time) / yards_to_feet = 245 := by
  sorry

end NUMINAMATH_GPT_powderman_distance_when_blast_heard_l14_1456


namespace NUMINAMATH_GPT_min_moves_to_visit_all_non_forbidden_squares_l14_1412

def min_diagonal_moves (n : ℕ) : ℕ :=
  2 * (n / 2) - 1

theorem min_moves_to_visit_all_non_forbidden_squares (n : ℕ) :
  min_diagonal_moves n = 2 * (n / 2) - 1 := by
  sorry

end NUMINAMATH_GPT_min_moves_to_visit_all_non_forbidden_squares_l14_1412


namespace NUMINAMATH_GPT_matrix_addition_l14_1478

def M1 : Matrix (Fin 3) (Fin 3) ℤ :=
![![4, 1, -3],
  ![0, -2, 5],
  ![7, 0, 1]]

def M2 : Matrix (Fin 3) (Fin 3) ℤ :=
![![ -6,  9, 2],
  ![  3, -4, -8],
  ![  0,  5, -3]]

def M3 : Matrix (Fin 3) (Fin 3) ℤ :=
![![ -2, 10, -1],
  ![  3, -6, -3],
  ![  7,  5, -2]]

theorem matrix_addition : M1 + M2 = M3 := by
  sorry

end NUMINAMATH_GPT_matrix_addition_l14_1478


namespace NUMINAMATH_GPT_ellipse_focal_distance_l14_1486

theorem ellipse_focal_distance (m : ℝ) :
  (∀ x y : ℝ, (x^2 / 16 + y^2 / m = 1) ∧ (2 * Real.sqrt (16 - m) = 2 * Real.sqrt 7)) → m = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ellipse_focal_distance_l14_1486


namespace NUMINAMATH_GPT_probability_all_and_at_least_one_pass_l14_1417

-- Define conditions
def pA : ℝ := 0.8
def pB : ℝ := 0.6
def pC : ℝ := 0.5

-- Define the main theorem we aim to prove
theorem probability_all_and_at_least_one_pass :
  (pA * pB * pC = 0.24) ∧ ((1 - (1 - pA) * (1 - pB) * (1 - pC)) = 0.96) := by
  sorry

end NUMINAMATH_GPT_probability_all_and_at_least_one_pass_l14_1417


namespace NUMINAMATH_GPT_unique_solution_l14_1490

theorem unique_solution (a b c : ℝ) (hb : b ≠ 2) (hc : c ≠ 0) : 
  ∃! x : ℝ, 4 * x - 7 + a = 2 * b * x + c ∧ x = (c + 7 - a) / (4 - 2 * b) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l14_1490


namespace NUMINAMATH_GPT_systemOfEquationsUniqueSolution_l14_1479

def largeBarrelHolds (x : ℝ) (y : ℝ) : Prop :=
  5 * x + y = 3

def smallBarrelHolds (x : ℝ) (y : ℝ) : Prop :=
  x + 5 * y = 2

theorem systemOfEquationsUniqueSolution (x y : ℝ) :
  (largeBarrelHolds x y) ∧ (smallBarrelHolds x y) ↔ 
  (5 * x + y = 3 ∧ x + 5 * y = 2) :=
by
  sorry

end NUMINAMATH_GPT_systemOfEquationsUniqueSolution_l14_1479


namespace NUMINAMATH_GPT_crates_needed_l14_1400

def ceil_div (a b : ℕ) : ℕ := (a + b - 1) / b

theorem crates_needed :
  ceil_div 145 12 + ceil_div 271 8 + ceil_div 419 10 + ceil_div 209 14 = 104 :=
by
  sorry

end NUMINAMATH_GPT_crates_needed_l14_1400


namespace NUMINAMATH_GPT_values_of_a_plus_b_l14_1421

theorem values_of_a_plus_b (a b : ℝ) (h1 : abs (-a) = abs (-1)) (h2 : b^2 = 9) (h3 : abs (a - b) = b - a) : a + b = 2 ∨ a + b = 4 := 
by 
  sorry

end NUMINAMATH_GPT_values_of_a_plus_b_l14_1421


namespace NUMINAMATH_GPT_negation_of_existential_l14_1483

theorem negation_of_existential :
  (¬ (∃ x : ℝ, x^2 - x - 1 > 0)) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_existential_l14_1483


namespace NUMINAMATH_GPT_gcd_lcm_product_24_36_proof_l14_1441

def gcd_lcm_product_24_36 : Prop :=
  let a := 24
  let b := 36
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  gcd_ab * lcm_ab = 864

theorem gcd_lcm_product_24_36_proof : gcd_lcm_product_24_36 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_24_36_proof_l14_1441
