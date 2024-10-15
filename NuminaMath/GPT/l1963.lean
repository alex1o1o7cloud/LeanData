import Mathlib

namespace NUMINAMATH_GPT_desiree_age_l1963_196391

-- Definitions of the given variables and conditions
variables (D C : ℝ)

-- Given conditions
def condition1 : Prop := D = 2 * C
def condition2 : Prop := D + 30 = 0.6666666 * (C + 30) + 14
def condition3 : Prop := D = 2.99999835

-- Main theorem to prove
theorem desiree_age : D = 2.99999835 :=
by
  { sorry }

end NUMINAMATH_GPT_desiree_age_l1963_196391


namespace NUMINAMATH_GPT_limit_of_p_n_is_tenth_l1963_196375

noncomputable def p_n (n : ℕ) : ℝ := sorry -- Definition of p_n needs precise formulation.

def tends_to_tenth_as_n_infty (p : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (p n - 1/10) < ε

theorem limit_of_p_n_is_tenth : tends_to_tenth_as_n_infty p_n := sorry

end NUMINAMATH_GPT_limit_of_p_n_is_tenth_l1963_196375


namespace NUMINAMATH_GPT_weekly_milk_consumption_l1963_196313

def milk_weekday : Nat := 3
def milk_saturday := 2 * milk_weekday
def milk_sunday := 3 * milk_weekday

theorem weekly_milk_consumption : (5 * milk_weekday) + milk_saturday + milk_sunday = 30 := by
  sorry

end NUMINAMATH_GPT_weekly_milk_consumption_l1963_196313


namespace NUMINAMATH_GPT_unique_line_through_point_odd_x_prime_y_intercepts_l1963_196377

theorem unique_line_through_point_odd_x_prime_y_intercepts :
  ∃! (a b : ℕ), 0 < b ∧ Nat.Prime b ∧ a % 2 = 1 ∧
  (4 * b + 3 * a = a * b) :=
sorry

end NUMINAMATH_GPT_unique_line_through_point_odd_x_prime_y_intercepts_l1963_196377


namespace NUMINAMATH_GPT_sum_cubes_l1963_196316

variables (a b : ℝ)
noncomputable def calculate_sum_cubes (a b : ℝ) : ℝ :=
a^3 + b^3

theorem sum_cubes (h1 : a + b = 11) (h2 : a * b = 21) : calculate_sum_cubes a b = 638 :=
by
  sorry

end NUMINAMATH_GPT_sum_cubes_l1963_196316


namespace NUMINAMATH_GPT_find_N_l1963_196387

theorem find_N (N : ℕ) (h : (Real.sqrt 3 - 1)^N = 4817152 - 2781184 * Real.sqrt 3) : N = 16 :=
sorry

end NUMINAMATH_GPT_find_N_l1963_196387


namespace NUMINAMATH_GPT_carrots_picked_next_day_l1963_196355

-- Definitions based on conditions
def initial_carrots : Nat := 48
def carrots_thrown_away : Nat := 45
def total_carrots_next_day : Nat := 45

-- The proof problem statement
theorem carrots_picked_next_day : 
  (initial_carrots - carrots_thrown_away + x = total_carrots_next_day) → (x = 42) :=
by 
  sorry

end NUMINAMATH_GPT_carrots_picked_next_day_l1963_196355


namespace NUMINAMATH_GPT_stapler_machines_l1963_196302

theorem stapler_machines (x : ℝ) :
  (∃ (x : ℝ), x > 0) ∧
  ((∀ r1 r2 : ℝ, (r1 = 800 / 6) → (r2 = 800 / x) → (r1 + r2 = 800 / 3)) ↔
    (1 / 6 + 1 / x = 1 / 3)) :=
by sorry

end NUMINAMATH_GPT_stapler_machines_l1963_196302


namespace NUMINAMATH_GPT_opposite_event_of_hitting_at_least_once_is_missing_both_times_l1963_196317

theorem opposite_event_of_hitting_at_least_once_is_missing_both_times
  (A B : Prop) :
  ¬(A ∨ B) ↔ (¬A ∧ ¬B) :=
by
  sorry

end NUMINAMATH_GPT_opposite_event_of_hitting_at_least_once_is_missing_both_times_l1963_196317


namespace NUMINAMATH_GPT_total_birds_in_tree_l1963_196361

def initial_birds := 14
def additional_birds := 21

theorem total_birds_in_tree : initial_birds + additional_birds = 35 := by
  sorry

end NUMINAMATH_GPT_total_birds_in_tree_l1963_196361


namespace NUMINAMATH_GPT_roots_are_distinct_and_negative_l1963_196329

theorem roots_are_distinct_and_negative : 
  (∀ x : ℝ, x^2 + m * x + 1 = 0 → ∃! (x1 x2 : ℝ), x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2) ↔ m > 2 :=
by
  sorry

end NUMINAMATH_GPT_roots_are_distinct_and_negative_l1963_196329


namespace NUMINAMATH_GPT_sandbox_width_l1963_196396

theorem sandbox_width :
  ∀ (length area width : ℕ), length = 312 → area = 45552 →
  area = length * width → width = 146 :=
by
  intros length area width h_length h_area h_eq
  sorry

end NUMINAMATH_GPT_sandbox_width_l1963_196396


namespace NUMINAMATH_GPT_fermat_little_theorem_l1963_196379

theorem fermat_little_theorem (p : ℕ) (a : ℤ) (hp : Nat.Prime p) (hcoprime : Int.gcd a p = 1) : 
  (a ^ (p - 1)) % p = 1 % p := 
sorry

end NUMINAMATH_GPT_fermat_little_theorem_l1963_196379


namespace NUMINAMATH_GPT_find_acute_angle_as_pi_over_4_l1963_196347
open Real

-- Definitions from the problem's conditions
variables (x : ℝ)
def is_acute (x : ℝ) : Prop := 0 < x ∧ x < π / 2
def trig_eq (x : ℝ) : Prop := (sin x) ^ 3 + (cos x) ^ 3 = sqrt 2 / 2

-- The math proof problem statement
theorem find_acute_angle_as_pi_over_4 (h_acute : is_acute x) (h_trig_eq : trig_eq x) : x = π / 4 := 
sorry

end NUMINAMATH_GPT_find_acute_angle_as_pi_over_4_l1963_196347


namespace NUMINAMATH_GPT_square_diff_correctness_l1963_196378

theorem square_diff_correctness (x y : ℝ) :
  let A := (x + y) * (x - 2*y)
  let B := (x + y) * (-x + y)
  let C := (x + y) * (-x - y)
  let D := (-x + y) * (x - y)
  (∃ (a b : ℝ), B = (a + b) * (a - b)) ∧ (∀ (p q : ℝ), A ≠ (p + q) * (p - q)) ∧ (∀ (r s : ℝ), C ≠ (r + s) * (r - s)) ∧ (∀ (t u : ℝ), D ≠ (t + u) * (t - u)) :=
by
  sorry

end NUMINAMATH_GPT_square_diff_correctness_l1963_196378


namespace NUMINAMATH_GPT_cube_probability_l1963_196335

def prob_same_color_vertical_faces : ℕ := sorry

theorem cube_probability :
  prob_same_color_vertical_faces = 1 / 27 := 
sorry

end NUMINAMATH_GPT_cube_probability_l1963_196335


namespace NUMINAMATH_GPT_breadth_of_water_tank_l1963_196397

theorem breadth_of_water_tank (L H V : ℝ) (n : ℕ) (avg_displacement : ℝ) (total_displacement : ℝ)
  (h_len : L = 40)
  (h_height : H = 0.25)
  (h_avg_disp : avg_displacement = 4)
  (h_number : n = 50)
  (h_total_disp : total_displacement = avg_displacement * n)
  (h_displacement_value : total_displacement = 200) :
  (40 * B * 0.25 = 200) → B = 20 :=
by
  intro h_eq
  sorry

end NUMINAMATH_GPT_breadth_of_water_tank_l1963_196397


namespace NUMINAMATH_GPT_martin_rings_big_bell_l1963_196367

/-
Problem Statement:
Martin rings the small bell 4 times more than 1/3 as often as the big bell.
If he rings both of them a combined total of 52 times, prove that he rings the big bell 36 times.
-/

theorem martin_rings_big_bell (s b : ℕ) 
  (h1 : s + b = 52) 
  (h2 : s = 4 + (1 / 3 : ℚ) * b) : 
  b = 36 := 
by
  sorry

end NUMINAMATH_GPT_martin_rings_big_bell_l1963_196367


namespace NUMINAMATH_GPT_evaluate_expression_l1963_196350

theorem evaluate_expression (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 2018) 
  (h2 : 3 * a + 8 * b + 24 * c + 37 * d = 2018) : 
  3 * b + 8 * c + 24 * d + 37 * a = 1215 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1963_196350


namespace NUMINAMATH_GPT_percentage_decrease_of_b_l1963_196389

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b)
  (h1 : a / b = 4 / 5)
  (h2 : x = a + 0.25 * a)
  (h3 : m = b * (1 - p / 100))
  (h4 : m / x = 0.4) :
  p = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_of_b_l1963_196389


namespace NUMINAMATH_GPT_problem1_problem2_l1963_196356

-- Define propositions P and Q under the given conditions
def P (a x : ℝ) : Prop := 2 * x^2 - 5 * a * x - 3 * a^2 < 0

def Q (x : ℝ) : Prop := (2 * Real.sin x > 1) ∧ (x^2 - x - 2 < 0)

-- Problem 1: Prove that if a = 2 and p ∧ q holds true, then the range of x is (π/6, 2)
theorem problem1 (x : ℝ) (hx1 : P 2 x ∧ Q x) : (Real.pi / 6 < x ∧ x < 2) :=
sorry

-- Problem 2: Prove that if ¬P is a sufficient but not necessary condition for ¬Q, then the range of a is [2/3, ∞)
theorem problem2 (a : ℝ) (h₁ : ∀ x, Q x → P a x) (h₂ : ∃ x, Q x → ¬P a x) : a ≥ 2 / 3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1963_196356


namespace NUMINAMATH_GPT_arithmetic_sequence_a15_l1963_196364

theorem arithmetic_sequence_a15 {a : ℕ → ℝ} (d : ℝ) (a7 a23 : ℝ) 
    (h1 : a 7 = 8) (h2 : a 23 = 22) : 
    a 15 = 15 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a15_l1963_196364


namespace NUMINAMATH_GPT_factorable_polynomial_l1963_196342

theorem factorable_polynomial (a b : ℝ) :
  (∀ x y : ℝ, ∃ u v p q : ℝ, (x + uy + v) * (x + py + q) = x * (x + 4) + a * (y^2 - 1) + 2 * b * y) ↔
  (a + 2)^2 + b^2 = 4 :=
  sorry

end NUMINAMATH_GPT_factorable_polynomial_l1963_196342


namespace NUMINAMATH_GPT_product_of_B_coordinates_l1963_196308

theorem product_of_B_coordinates :
  (∃ (x y : ℝ), (1 / 3 * x + 2 / 3 * 4 = 1 ∧ 1 / 3 * y + 2 / 3 * 2 = 7) ∧ x * y = -85) :=
by
  sorry

end NUMINAMATH_GPT_product_of_B_coordinates_l1963_196308


namespace NUMINAMATH_GPT_max_rectangle_area_l1963_196369

theorem max_rectangle_area (P : ℕ) (hP : P = 40) (l w : ℕ) (h : 2 * l + 2 * w = P) : ∃ A, A = l * w ∧ ∀ l' w', 2 * l' + 2 * w' = P → l' * w' ≤ 100 :=
by 
  sorry

end NUMINAMATH_GPT_max_rectangle_area_l1963_196369


namespace NUMINAMATH_GPT_part_one_part_two_part_three_l1963_196383

-- Define the sequence and the sum of its first n terms
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - 2 ^ n

-- Prove that a_1 = 2 and a_4 = 40
theorem part_one (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  a 1 = 2 ∧ a 4 = 40 := by
  sorry
  
-- Prove that the sequence {a_{n+1} - 2a_n} is a geometric sequence
theorem part_two (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  ∃ r : ℕ, (r = 2) ∧ (∀ n, (a (n + 1) - 2 * a n) = r ^ n) := by
  sorry

-- Prove the general term formula for the sequence {a_n}
theorem part_three (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  ∀ n, a n = 2 ^ (n + 1) - 2 := by
  sorry

end NUMINAMATH_GPT_part_one_part_two_part_three_l1963_196383


namespace NUMINAMATH_GPT_problem_2003_divisibility_l1963_196348

theorem problem_2003_divisibility :
  let N := (List.range' 1 1001).prod + (List.range' 1002 1001).prod
  N % 2003 = 0 := by
  sorry

end NUMINAMATH_GPT_problem_2003_divisibility_l1963_196348


namespace NUMINAMATH_GPT_sum_of_solutions_l1963_196325

theorem sum_of_solutions : ∀ x : ℚ, (4 * x + 6) * (3 * x - 8) = 0 → 
  (x = -3 / 2 ∨ x = 8 / 3) → 
  (-3 / 2 + 8 / 3) = 7 / 6 :=
by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l1963_196325


namespace NUMINAMATH_GPT_average_weight_of_16_boys_l1963_196380

theorem average_weight_of_16_boys :
  ∃ A : ℝ,
    (16 * A + 8 * 45.15 = 24 * 48.55) ∧
    A = 50.25 :=
by {
  -- Proof skipped, using sorry to denote the proof is required.
  sorry
}

end NUMINAMATH_GPT_average_weight_of_16_boys_l1963_196380


namespace NUMINAMATH_GPT_time_bob_cleans_room_l1963_196312

variable (timeAlice : ℕ) (fractionBob : ℚ)

-- Definitions based on conditions from the problem
def timeAliceCleaningRoom : ℕ := 40
def fractionOfTimeBob : ℚ := 3 / 8

-- Prove the time it takes Bob to clean his room
theorem time_bob_cleans_room : (timeAliceCleaningRoom * fractionOfTimeBob : ℚ) = 15 := 
by
  sorry

end NUMINAMATH_GPT_time_bob_cleans_room_l1963_196312


namespace NUMINAMATH_GPT_ratio_of_part_diminished_by_4_l1963_196357

theorem ratio_of_part_diminished_by_4 (N P : ℕ) (h1 : N = 160)
    (h2 : (1/5 : ℝ) * N + 4 = P - 4) : (P - 4) / N = 9 / 40 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_part_diminished_by_4_l1963_196357


namespace NUMINAMATH_GPT_smallest_k_sum_of_squares_multiple_of_200_l1963_196374

-- Define the sum of squares for positive integer k
def sum_of_squares (k : ℕ) : ℕ := (k * (k + 1) * (2 * k + 1)) / 6

-- Prove that the sum of squares for k = 112 is a multiple of 200
theorem smallest_k_sum_of_squares_multiple_of_200 :
  ∃ k : ℕ, sum_of_squares k = sum_of_squares 112 ∧ 200 ∣ sum_of_squares 112 :=
sorry

end NUMINAMATH_GPT_smallest_k_sum_of_squares_multiple_of_200_l1963_196374


namespace NUMINAMATH_GPT_ninth_term_arithmetic_sequence_l1963_196331

-- Definitions based on conditions:
def first_term : ℚ := 5 / 6
def seventeenth_term : ℚ := 5 / 8

-- Here is the main statement we need to prove:
theorem ninth_term_arithmetic_sequence : (first_term + 8 * ((seventeenth_term - first_term) / 16) = 15 / 16) :=
by
  sorry

end NUMINAMATH_GPT_ninth_term_arithmetic_sequence_l1963_196331


namespace NUMINAMATH_GPT_distinct_solutions_subtraction_eq_two_l1963_196304

theorem distinct_solutions_subtraction_eq_two :
  ∃ p q : ℝ, (p ≠ q) ∧ (p > q) ∧ ((6 * p - 18) / (p^2 + 4 * p - 21) = p + 3) ∧ ((6 * q - 18) / (q^2 + 4 * q - 21) = q + 3) ∧ (p - q = 2) :=
by
  have p := -3
  have q := -5
  exists p, q
  sorry

end NUMINAMATH_GPT_distinct_solutions_subtraction_eq_two_l1963_196304


namespace NUMINAMATH_GPT_find_m_of_lcm_conditions_l1963_196322

theorem find_m_of_lcm_conditions (m : ℕ) (h_pos : 0 < m)
  (h1 : Int.lcm 18 m = 54)
  (h2 : Int.lcm m 45 = 180) : m = 36 :=
sorry

end NUMINAMATH_GPT_find_m_of_lcm_conditions_l1963_196322


namespace NUMINAMATH_GPT_sum_even_and_odd_numbers_up_to_50_l1963_196338

def sum_even_numbers (n : ℕ) : ℕ :=
  (2 + 50) * n / 2

def sum_odd_numbers (n : ℕ) : ℕ :=
  (1 + 49) * n / 2

theorem sum_even_and_odd_numbers_up_to_50 : 
  sum_even_numbers 25 + sum_odd_numbers 25 = 1275 :=
by
  sorry

end NUMINAMATH_GPT_sum_even_and_odd_numbers_up_to_50_l1963_196338


namespace NUMINAMATH_GPT_sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l1963_196314

theorem sqrt_8_plus_sqrt_2_minus_sqrt_18 :
  (Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 18 = 0) :=
sorry

theorem sqrt_3_minus_2_squared :
  ((Real.sqrt 3 - 2) ^ 2 = 7 - 4 * Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l1963_196314


namespace NUMINAMATH_GPT_todd_savings_l1963_196371

def original_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon : ℝ := 10
def credit_card_discount : ℝ := 0.10
def rebate : ℝ := 0.05
def sales_tax : ℝ := 0.08

def calculate_savings (original_price sale_discount coupon credit_card_discount rebate sales_tax : ℝ) : ℝ :=
  let after_sale := original_price * (1 - sale_discount)
  let after_coupon := after_sale - coupon
  let after_credit_card := after_coupon * (1 - credit_card_discount)
  let after_rebate := after_credit_card * (1 - rebate)
  let tax := after_credit_card * sales_tax
  let final_price := after_rebate + tax
  original_price - final_price

theorem todd_savings : calculate_savings 125 0.20 10 0.10 0.05 0.08 = 41.57 :=
by
  sorry

end NUMINAMATH_GPT_todd_savings_l1963_196371


namespace NUMINAMATH_GPT_value_of_3_W_4_l1963_196381

def W (a b : ℤ) : ℤ := b + 5 * a - 3 * a ^ 2

theorem value_of_3_W_4 : W 3 4 = -8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_3_W_4_l1963_196381


namespace NUMINAMATH_GPT_taxi_range_l1963_196323

noncomputable def fare (x : ℝ) : ℝ :=
  if x <= 3 then 
    5
  else if x <= 10 then
    5 + (x - 3) * 2
  else
    5 + 7 * 2 + (x - 10) * 3

theorem taxi_range (x : ℝ) (h : fare x + 1 = 38) : 15 < x ∧ x ≤ 16 := 
  sorry

end NUMINAMATH_GPT_taxi_range_l1963_196323


namespace NUMINAMATH_GPT_kendall_total_change_l1963_196395

-- Definition of values of coins
def value_of_quarters (q : ℕ) : ℝ := q * 0.25
def value_of_dimes (d : ℕ) : ℝ := d * 0.10
def value_of_nickels (n : ℕ) : ℝ := n * 0.05

-- Conditions
def quarters := 10
def dimes := 12
def nickels := 6

-- Theorem statement
theorem kendall_total_change : 
  value_of_quarters quarters + value_of_dimes dimes + value_of_nickels nickels = 4.00 :=
by
  sorry

end NUMINAMATH_GPT_kendall_total_change_l1963_196395


namespace NUMINAMATH_GPT_games_in_each_box_l1963_196386

theorem games_in_each_box (start_games sold_games total_boxes remaining_games games_per_box : ℕ) 
  (h_start: start_games = 35) (h_sold: sold_games = 19) (h_boxes: total_boxes = 2) 
  (h_remaining: remaining_games = start_games - sold_games) 
  (h_per_box: games_per_box = remaining_games / total_boxes) : games_per_box = 8 :=
by
  sorry

end NUMINAMATH_GPT_games_in_each_box_l1963_196386


namespace NUMINAMATH_GPT_smallest_positive_period_of_sin_2x_l1963_196326

noncomputable def period_of_sine (B : ℝ) : ℝ := (2 * Real.pi) / B

theorem smallest_positive_period_of_sin_2x :
  period_of_sine 2 = Real.pi := sorry

end NUMINAMATH_GPT_smallest_positive_period_of_sin_2x_l1963_196326


namespace NUMINAMATH_GPT_cells_count_after_9_days_l1963_196352

theorem cells_count_after_9_days :
  let a := 5
  let r := 3
  let n := 3
  a * r^(n-1) = 45 :=
by
  let a := 5
  let r := 3
  let n := 3
  sorry

end NUMINAMATH_GPT_cells_count_after_9_days_l1963_196352


namespace NUMINAMATH_GPT_find_base_l1963_196373

theorem find_base (r : ℕ) : 
  (2 * r^2 + 1 * r + 0) + (2 * r^2 + 6 * r + 0) = 5 * r^2 + 0 * r + 0 → r = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_base_l1963_196373


namespace NUMINAMATH_GPT_gas_cost_per_gallon_l1963_196353

-- Define the conditions as Lean definitions
def miles_per_gallon : ℕ := 32
def total_miles : ℕ := 336
def total_cost : ℕ := 42

-- Prove the cost of gas per gallon, which is $4 per gallon
theorem gas_cost_per_gallon : total_cost / (total_miles / miles_per_gallon) = 4 :=
by
  sorry

end NUMINAMATH_GPT_gas_cost_per_gallon_l1963_196353


namespace NUMINAMATH_GPT_loss_percentage_is_11_l1963_196393

-- Constants for the given problem conditions
def cost_price : ℝ := 1500
def selling_price : ℝ := 1335

-- Formulation of the proof problem
theorem loss_percentage_is_11 :
  ((cost_price - selling_price) / cost_price) * 100 = 11 := by
  sorry

end NUMINAMATH_GPT_loss_percentage_is_11_l1963_196393


namespace NUMINAMATH_GPT_total_students_in_high_school_l1963_196321

theorem total_students_in_high_school 
  (num_freshmen : ℕ)
  (num_sample : ℕ) 
  (num_sophomores : ℕ)
  (num_seniors : ℕ)
  (freshmen_drawn : ℕ)
  (sampling_ratio : ℕ)
  (total_students : ℕ)
  (h1 : num_freshmen = 600)
  (h2 : num_sample = 45)
  (h3 : num_sophomores = 20)
  (h4 : num_seniors = 10)
  (h5 : freshmen_drawn = 15)
  (h6 : sampling_ratio = 40)
  (h7 : freshmen_drawn * sampling_ratio = num_freshmen)
  : total_students = 1800 :=
sorry

end NUMINAMATH_GPT_total_students_in_high_school_l1963_196321


namespace NUMINAMATH_GPT_mario_hibiscus_l1963_196318

def hibiscus_flowers (F : ℕ) : Prop :=
  let F2 := 2 * F
  let F3 := 4 * F2
  F + F2 + F3 = 22 → F = 2

theorem mario_hibiscus (F : ℕ) : hibiscus_flowers F :=
  sorry

end NUMINAMATH_GPT_mario_hibiscus_l1963_196318


namespace NUMINAMATH_GPT_input_value_for_output_16_l1963_196301

theorem input_value_for_output_16 (x : ℝ) (y : ℝ) (h1 : x < 0 → y = (x + 1)^2) (h2 : x ≥ 0 → y = (x - 1)^2) (h3 : y = 16) : x = 5 ∨ x = -5 := by
  sorry

end NUMINAMATH_GPT_input_value_for_output_16_l1963_196301


namespace NUMINAMATH_GPT_quadratic_roots_square_l1963_196360

theorem quadratic_roots_square (q : ℝ) :
  (∃ a : ℝ, a + a^2 = 12 ∧ q = a * a^2) → (q = 27 ∨ q = -64) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_square_l1963_196360


namespace NUMINAMATH_GPT_difference_of_decimal_and_fraction_l1963_196305

theorem difference_of_decimal_and_fraction :
  0.127 - (1 / 8) = 0.002 := 
by
  sorry

end NUMINAMATH_GPT_difference_of_decimal_and_fraction_l1963_196305


namespace NUMINAMATH_GPT_Alton_profit_l1963_196343

variable (earnings_per_day : ℕ)
variable (days_per_week : ℕ)
variable (rent_per_week : ℕ)

theorem Alton_profit (h1 : earnings_per_day = 8) (h2 : days_per_week = 7) (h3 : rent_per_week = 20) :
  earnings_per_day * days_per_week - rent_per_week = 36 := 
by sorry

end NUMINAMATH_GPT_Alton_profit_l1963_196343


namespace NUMINAMATH_GPT_problem_solution_l1963_196310

def is_desirable_n (n : ℕ) : Prop :=
  ∃ (r b : ℕ), n = r + b ∧ r^2 - r*b + b^2 = 2007 ∧ 3 ∣ r ∧ 3 ∣ b

theorem problem_solution :
  ∀ n : ℕ, (is_desirable_n n → n = 69 ∨ n = 84) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1963_196310


namespace NUMINAMATH_GPT_distance_difference_l1963_196354

-- Definition of speeds and time
def speed_alberto : ℕ := 16
def speed_clara : ℕ := 12
def time_hours : ℕ := 5

-- Distance calculation functions
def distance (speed time : ℕ) : ℕ := speed * time

-- Main theorem statement
theorem distance_difference : 
  distance speed_alberto time_hours - distance speed_clara time_hours = 20 :=
by
  sorry

end NUMINAMATH_GPT_distance_difference_l1963_196354


namespace NUMINAMATH_GPT_roxy_garden_problem_l1963_196300

variable (initial_flowering : ℕ)
variable (multiplier : ℕ)
variable (bought_flowering : ℕ)
variable (bought_fruiting : ℕ)
variable (given_flowering : ℕ)
variable (given_fruiting : ℕ)

def initial_fruiting (initial_flowering : ℕ) (multiplier : ℕ) : ℕ :=
  initial_flowering * multiplier

def saturday_flowering (initial_flowering : ℕ) (bought_flowering : ℕ) : ℕ :=
  initial_flowering + bought_flowering

def saturday_fruiting (initial_fruiting : ℕ) (bought_fruiting : ℕ) : ℕ :=
  initial_fruiting + bought_fruiting

def sunday_flowering (saturday_flowering : ℕ) (given_flowering : ℕ) : ℕ :=
  saturday_flowering - given_flowering

def sunday_fruiting (saturday_fruiting : ℕ) (given_fruiting : ℕ) : ℕ :=
  saturday_fruiting - given_fruiting

def total_plants_remaining (sunday_flowering : ℕ) (sunday_fruiting : ℕ) : ℕ :=
  sunday_flowering + sunday_fruiting

theorem roxy_garden_problem 
  (h1 : initial_flowering = 7)
  (h2 : multiplier = 2)
  (h3 : bought_flowering = 3)
  (h4 : bought_fruiting = 2)
  (h5 : given_flowering = 1)
  (h6 : given_fruiting = 4) :
  total_plants_remaining 
    (sunday_flowering 
      (saturday_flowering initial_flowering bought_flowering) 
      given_flowering) 
    (sunday_fruiting 
      (saturday_fruiting 
        (initial_fruiting initial_flowering multiplier) 
        bought_fruiting) 
      given_fruiting) = 21 := 
  sorry

end NUMINAMATH_GPT_roxy_garden_problem_l1963_196300


namespace NUMINAMATH_GPT_sum_of_reciprocals_squares_l1963_196307

theorem sum_of_reciprocals_squares (a b : ℕ) (h : a * b = 17) :
  (1 : ℚ) / (a * a) + 1 / (b * b) = 290 / 289 :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_squares_l1963_196307


namespace NUMINAMATH_GPT_molly_total_cost_l1963_196385

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_children_per_brother : ℕ := 2
def num_spouse_per_brother : ℕ := 1

def total_num_relatives : ℕ := 
  let parents_and_siblings := num_parents + num_brothers
  let additional_relatives := num_brothers * (1 + num_spouse_per_brother + num_children_per_brother)
  parents_and_siblings + additional_relatives

def total_cost : ℕ :=
  total_num_relatives * cost_per_package

theorem molly_total_cost : total_cost = 85 := sorry

end NUMINAMATH_GPT_molly_total_cost_l1963_196385


namespace NUMINAMATH_GPT_max_possible_salary_l1963_196334

-- Definition of the conditions
def num_players : ℕ := 25
def min_salary : ℕ := 20000
def total_salary_cap : ℕ := 800000

-- The theorem we want to prove: the maximum possible salary for a single player is $320,000
theorem max_possible_salary (total_salary_cap : ℕ) (num_players : ℕ) (min_salary : ℕ) :
  total_salary_cap - (num_players - 1) * min_salary = 320000 :=
by sorry

end NUMINAMATH_GPT_max_possible_salary_l1963_196334


namespace NUMINAMATH_GPT_range_of_a_l1963_196328

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 1 → x > a) ∧ (∃ x : ℝ, x > a ∧ x ≤ 1) → a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1963_196328


namespace NUMINAMATH_GPT_profit_percent_l1963_196320

theorem profit_percent (cost_price : ℝ) (selling_price : ℝ) (marked_price : ℝ) (n_pens : ℕ) 
  (h1 : n_pens = 60) (h2 : marked_price = 1) (h3 : cost_price = (46 : ℝ) / (60 : ℝ)) 
  (h4 : selling_price = 0.99 * marked_price) : 
  (selling_price - cost_price) / cost_price * 100 = 29.11 :=
by
  sorry

end NUMINAMATH_GPT_profit_percent_l1963_196320


namespace NUMINAMATH_GPT_gravel_amount_l1963_196376

theorem gravel_amount (total_material sand gravel : ℝ) 
  (h1 : total_material = 14.02) 
  (h2 : sand = 8.11) 
  (h3 : gravel = total_material - sand) : 
  gravel = 5.91 :=
  sorry

end NUMINAMATH_GPT_gravel_amount_l1963_196376


namespace NUMINAMATH_GPT_intersection_point_finv_l1963_196303

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 4 * x + b

theorem intersection_point_finv (a b : ℤ) : 
  (∀ x : ℝ, f (f x b) b = x) → 
  (∀ y : ℝ, f (f y b) b = y) → 
  (f (-4) b = a) → 
  (f a b = -4) → 
  a = -4 := 
by
  intros
  sorry

end NUMINAMATH_GPT_intersection_point_finv_l1963_196303


namespace NUMINAMATH_GPT_project_completion_time_l1963_196311

theorem project_completion_time
  (x y z : ℝ)
  (h1 : x + y = 1 / 2)
  (h2 : y + z = 1 / 4)
  (h3 : z + x = 1 / 2.4) :
  (1 / x) = 3 :=
by
  sorry

end NUMINAMATH_GPT_project_completion_time_l1963_196311


namespace NUMINAMATH_GPT_find_c_in_parabola_l1963_196327

theorem find_c_in_parabola (b c : ℝ) (h₁ : 2 = (-1) ^ 2 + b * (-1) + c) (h₂ : 2 = 3 ^ 2 + b * 3 + c) : c = -1 :=
sorry

end NUMINAMATH_GPT_find_c_in_parabola_l1963_196327


namespace NUMINAMATH_GPT_Linda_original_savings_l1963_196363

variable (TV_cost : ℝ := 200) -- TV cost
variable (savings : ℝ) -- Linda's original savings

-- Prices, Discounts, Taxes
variable (sofa_price : ℝ := 600)
variable (sofa_discount : ℝ := 0.20)
variable (sofa_tax : ℝ := 0.05)

variable (dining_table_price : ℝ := 400)
variable (dining_table_discount : ℝ := 0.15)
variable (dining_table_tax : ℝ := 0.06)

variable (chair_set_price : ℝ := 300)
variable (chair_set_discount : ℝ := 0.25)
variable (chair_set_tax : ℝ := 0.04)

variable (coffee_table_price : ℝ := 100)
variable (coffee_table_discount : ℝ := 0.10)
variable (coffee_table_tax : ℝ := 0.03)

variable (service_charge_rate : ℝ := 0.02) -- Service charge rate

noncomputable def discounted_price_with_tax (price discount tax : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  taxed_price

noncomputable def total_furniture_cost : ℝ :=
  let sofa_cost := discounted_price_with_tax sofa_price sofa_discount sofa_tax
  let dining_table_cost := discounted_price_with_tax dining_table_price dining_table_discount dining_table_tax
  let chair_set_cost := discounted_price_with_tax chair_set_price chair_set_discount chair_set_tax
  let coffee_table_cost := discounted_price_with_tax coffee_table_price coffee_table_discount coffee_table_tax
  let combined_cost := sofa_cost + dining_table_cost + chair_set_cost + coffee_table_cost
  combined_cost * (1 + service_charge_rate)

theorem Linda_original_savings : savings = 4 * TV_cost ∧ savings / 4 * 3 = total_furniture_cost :=
by
  sorry -- Proof skipped

end NUMINAMATH_GPT_Linda_original_savings_l1963_196363


namespace NUMINAMATH_GPT_together_time_l1963_196372

theorem together_time (P_time Q_time : ℝ) (hP : P_time = 4) (hQ : Q_time = 6) : (1 / ((1 / P_time) + (1 / Q_time))) = 2.4 :=
by
  sorry

end NUMINAMATH_GPT_together_time_l1963_196372


namespace NUMINAMATH_GPT_find_limit_of_hours_l1963_196349

def regular_rate : ℝ := 16
def overtime_rate (r : ℝ) : ℝ := r * 1.75
def total_compensation : ℝ := 920
def total_hours : ℝ := 50

theorem find_limit_of_hours : 
  ∃ (L : ℝ), 
    total_compensation = (regular_rate * L) + ((overtime_rate regular_rate) * (total_hours - L)) →
    L = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_limit_of_hours_l1963_196349


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l1963_196332

theorem rhombus_diagonal_length (side : ℝ) (shorter_diagonal : ℝ) 
  (h1 : side = 51) (h2 : shorter_diagonal = 48) : 
  ∃ longer_diagonal : ℝ, longer_diagonal = 90 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l1963_196332


namespace NUMINAMATH_GPT_solve_for_x_l1963_196344

theorem solve_for_x (x : ℕ) : (3 : ℝ)^(27^x) = (27 : ℝ)^(3^x) → x = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1963_196344


namespace NUMINAMATH_GPT_total_results_count_l1963_196306

theorem total_results_count (N : ℕ) (S : ℕ) 
  (h1 : S = 50 * N) 
  (h2 : (12 * 14) + (12 * 17) = 372)
  (h3 : S = 372 + 878) : N = 25 := 
by 
  sorry

end NUMINAMATH_GPT_total_results_count_l1963_196306


namespace NUMINAMATH_GPT_monthly_growth_rate_l1963_196358

-- Definitions based on the conditions given in the original problem.
def final_height : ℝ := 80
def current_height : ℝ := 20
def months_in_year : ℕ := 12

-- Prove the monthly growth rate.
theorem monthly_growth_rate : (final_height - current_height) / months_in_year = 5 := by
  sorry

end NUMINAMATH_GPT_monthly_growth_rate_l1963_196358


namespace NUMINAMATH_GPT_polar_to_rectangular_l1963_196351

open Real

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 8) (h_θ : θ = π / 4) :
    (r * cos θ, r * sin θ) = (4 * sqrt 2, 4 * sqrt 2) :=
by
  rw [h_r, h_θ]
  rw [cos_pi_div_four, sin_pi_div_four]
  norm_num
  field_simp [sqrt_eq_rpow]
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_l1963_196351


namespace NUMINAMATH_GPT_amaya_total_marks_l1963_196370

theorem amaya_total_marks 
  (m_a s_a a m m_s : ℕ) 
  (h_music : m_a = 70)
  (h_social_studies : s_a = m_a + 10)
  (h_maths_art_diff : m = a - 20)
  (h_maths_fraction : m = a - 1/10 * a)
  (h_maths_eq_fraction : m = 9/10 * a)
  (h_arts : 9/10 * a = a - 20)
  (h_total : m_a + s_a + a + m = 530) :
  m_a + s_a + a + m = 530 :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_amaya_total_marks_l1963_196370


namespace NUMINAMATH_GPT_tax_rate_equals_65_l1963_196390

def tax_rate_percentage := 65
def tax_rate_per_dollars (rate_percentage : ℕ) : ℕ :=
  (rate_percentage / 100) * 100

theorem tax_rate_equals_65 :
  tax_rate_per_dollars tax_rate_percentage = 65 := by
  sorry

end NUMINAMATH_GPT_tax_rate_equals_65_l1963_196390


namespace NUMINAMATH_GPT_students_prefer_dogs_l1963_196399

theorem students_prefer_dogs (total_students : ℕ) (perc_dogs_vg perc_dogs_mv : ℕ) (h_total: total_students = 30)
  (h_perc_dogs_vg: perc_dogs_vg = 50) (h_perc_dogs_mv: perc_dogs_mv = 10) :
  total_students * perc_dogs_vg / 100 + total_students * perc_dogs_mv / 100 = 18 := by
  sorry

end NUMINAMATH_GPT_students_prefer_dogs_l1963_196399


namespace NUMINAMATH_GPT_fermat_prime_solution_unique_l1963_196392

def is_fermat_prime (p : ℕ) : Prop :=
  ∃ r : ℕ, p = 2^(2^r) + 1

def problem_statement (p n k : ℕ) : Prop :=
  is_fermat_prime p ∧ p^n + n = (n + 1)^k

theorem fermat_prime_solution_unique (p n k : ℕ) :
  problem_statement p n k → (p, n, k) = (3, 1, 2) ∨ (p, n, k) = (5, 2, 3) :=
by
  sorry

end NUMINAMATH_GPT_fermat_prime_solution_unique_l1963_196392


namespace NUMINAMATH_GPT_inequality_proof_l1963_196324

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2 * (a - 1) * (b - 1) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1963_196324


namespace NUMINAMATH_GPT_minimum_value_of_expression_l1963_196336

noncomputable def f (x : ℝ) : ℝ := 16^x - 2^x + x^2 + 1

theorem minimum_value_of_expression : ∃ (x : ℝ), f x = 1 ∧ ∀ y : ℝ, f y ≥ 1 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l1963_196336


namespace NUMINAMATH_GPT_police_coverage_l1963_196319

-- Define the intersections and streets
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

open Intersection

-- Define the streets
def Streets : List (List Intersection) :=
  [ [A, B, C, D],    -- Horizontal street 1
    [E, F, G],       -- Horizontal street 2
    [H, I, J, K],    -- Horizontal street 3
    [A, E, H],       -- Vertical street 1
    [B, F, I],       -- Vertical street 2
    [D, G, J],       -- Vertical street 3
    [H, F, C],       -- Diagonal street 1
    [C, G, K]        -- Diagonal street 2
  ]

-- Define the set of intersections where police officers are 
def policeIntersections : List Intersection := [B, G, H]

-- State the theorem to be proved
theorem police_coverage : 
  ∀ (street : List Intersection), street ∈ Streets → 
  ∃ (i : Intersection), i ∈ policeIntersections ∧ i ∈ street := 
sorry

end NUMINAMATH_GPT_police_coverage_l1963_196319


namespace NUMINAMATH_GPT_fraction_subtraction_l1963_196341

theorem fraction_subtraction : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) 
  = 9 / 20 := by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l1963_196341


namespace NUMINAMATH_GPT_cos_diff_angle_l1963_196340

theorem cos_diff_angle
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) 
  (h : 3 * Real.sin α = Real.tan α) :
  Real.cos (α - π / 4) = (4 + Real.sqrt 2) / 6 :=
sorry

end NUMINAMATH_GPT_cos_diff_angle_l1963_196340


namespace NUMINAMATH_GPT_white_stones_count_l1963_196394

/-- We define the total number of stones as a constant. -/
def total_stones : ℕ := 120

/-- We define the difference between white and black stones as a constant. -/
def white_minus_black : ℕ := 36

/-- The theorem states that if there are 120 go stones in total and 
    36 more white go stones than black go stones, then there are 78 white go stones. -/
theorem white_stones_count (W B : ℕ) (h1 : W = B + white_minus_black) (h2 : B + W = total_stones) : W = 78 := 
sorry

end NUMINAMATH_GPT_white_stones_count_l1963_196394


namespace NUMINAMATH_GPT_which_two_students_donated_l1963_196333

theorem which_two_students_donated (A B C D : Prop) 
  (h1 : A ∨ D) 
  (h2 : ¬(A ∧ D)) 
  (h3 : (A ∧ B) ∨ (A ∧ D) ∨ (B ∧ D))
  (h4 : ¬(A ∧ B ∧ D)) 
  : B ∧ D :=
sorry

end NUMINAMATH_GPT_which_two_students_donated_l1963_196333


namespace NUMINAMATH_GPT_sum_of_cubes_l1963_196398

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
sorry

end NUMINAMATH_GPT_sum_of_cubes_l1963_196398


namespace NUMINAMATH_GPT_find_numbers_l1963_196345

theorem find_numbers (a b c : ℝ) (x y z: ℝ) (h1 : x + y = z + a) (h2 : x + z = y + b) (h3 : y + z = x + c) :
    x = (a + b - c) / 2 ∧ y = (a - b + c) / 2 ∧ z = (-a + b + c) / 2 := by
  sorry

end NUMINAMATH_GPT_find_numbers_l1963_196345


namespace NUMINAMATH_GPT_graph_passes_through_point_l1963_196346

noncomputable def exponential_shift (a : ℝ) (x : ℝ) := a^(x - 2)

theorem graph_passes_through_point (a : ℝ) (h : a > 0) (h1 : a ≠ 1) : exponential_shift a 2 = 1 :=
by
  unfold exponential_shift
  sorry

end NUMINAMATH_GPT_graph_passes_through_point_l1963_196346


namespace NUMINAMATH_GPT_calculate_difference_of_squares_l1963_196365

theorem calculate_difference_of_squares :
  (153^2 - 147^2) = 1800 :=
by
  sorry

end NUMINAMATH_GPT_calculate_difference_of_squares_l1963_196365


namespace NUMINAMATH_GPT_find_angle3_l1963_196388

theorem find_angle3 (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 + angle2 = 90)
  (h2 : angle2 + angle3 = 180)
  (h3 : angle1 = 20) :
  angle3 = 110 :=
sorry

end NUMINAMATH_GPT_find_angle3_l1963_196388


namespace NUMINAMATH_GPT_gigi_ate_33_bananas_l1963_196368

def gigi_bananas (total_bananas : ℕ) (days : ℕ) (diff : ℕ) (bananas_day_7 : ℕ) : Prop :=
  ∃ b, (days * b + diff * ((days * (days - 1)) / 2)) = total_bananas ∧ 
       (b + 6 * diff) = bananas_day_7

theorem gigi_ate_33_bananas :
  gigi_bananas 150 7 4 33 :=
by {
  sorry
}

end NUMINAMATH_GPT_gigi_ate_33_bananas_l1963_196368


namespace NUMINAMATH_GPT_number_of_ninth_graders_l1963_196339

def num_students_total := 50
def num_students_7th (x : Int) := 2 * x - 1
def num_students_8th (x : Int) := x

theorem number_of_ninth_graders (x : Int) :
  num_students_7th x + num_students_8th x + (51 - 3 * x) = num_students_total := by
  sorry

end NUMINAMATH_GPT_number_of_ninth_graders_l1963_196339


namespace NUMINAMATH_GPT_tan_diff_l1963_196309

theorem tan_diff (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) : Real.tan (α - β) = 1 / 7 := by
  sorry

end NUMINAMATH_GPT_tan_diff_l1963_196309


namespace NUMINAMATH_GPT_logical_equivalence_l1963_196337

theorem logical_equivalence (P Q R : Prop) :
  ((¬ P ∧ ¬ Q) → ¬ R) ↔ (R → (P ∨ Q)) :=
by sorry

end NUMINAMATH_GPT_logical_equivalence_l1963_196337


namespace NUMINAMATH_GPT_geometric_series_sum_l1963_196315

theorem geometric_series_sum :
  (1 / 5 - 1 / 25 + 1 / 125 - 1 / 625 + 1 / 3125) = 521 / 3125 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1963_196315


namespace NUMINAMATH_GPT_math_competition_question_1_math_competition_question_2_l1963_196382

noncomputable def participant_score_probabilities : Prop :=
  let P1 := (3 / 5)^2 * (2 / 5)^2
  let P2 := 2 * (3 / 5) * (2 / 5)
  let P3 := 2 * (3 / 5) * (2 / 5)^2
  let P4 := (3 / 5)^2
  P1 + P2 + P3 + P4 = 208 / 625

noncomputable def winning_probabilities : Prop :=
  let P_100_or_more := (4 / 5)^8 * (3 / 5)^3 + 3 * (4 / 5)^8 * (3 / 5)^2 * (2 / 5) + 
                      (8 * (4 / 5)^7 * (1/5) * (3 / 5)^3 + 
                      28 * (4 / 5)^6 * (1/5)^2 * (3 / 5)^3)
  let winning_if_100_or_more := P_100_or_more * (9 / 10)
  let winning_if_less_100 := (1 - P_100_or_more) * (2 / 5)
  winning_if_100_or_more + winning_if_less_100 ≥ 1 / 2

theorem math_competition_question_1 : participant_score_probabilities :=
by sorry

theorem math_competition_question_2 : winning_probabilities :=
by sorry

end NUMINAMATH_GPT_math_competition_question_1_math_competition_question_2_l1963_196382


namespace NUMINAMATH_GPT_all_lucky_years_l1963_196330

def is_lucky_year (y : ℕ) : Prop :=
  ∃ m d : ℕ, 1 ≤ m ∧ m ≤ 12 ∧ 1 ≤ d ∧ d ≤ 31 ∧ (m * d = y % 100)

theorem all_lucky_years :
  (is_lucky_year 2024) ∧ (is_lucky_year 2025) ∧ (is_lucky_year 2026) ∧ (is_lucky_year 2027) ∧ (is_lucky_year 2028) :=
sorry

end NUMINAMATH_GPT_all_lucky_years_l1963_196330


namespace NUMINAMATH_GPT_simplify_expression_l1963_196359

theorem simplify_expression :
  1 + 1 / (1 + 1 / (2 + 2)) = 9 / 5 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1963_196359


namespace NUMINAMATH_GPT_initial_volume_of_mixture_l1963_196362

theorem initial_volume_of_mixture (p q : ℕ) (x : ℕ) (h_ratio1 : p = 5 * x) (h_ratio2 : q = 3 * x) (h_added : q + 15 = 6 * x) (h_new_ratio : 5 * (3 * x + 15) = 6 * 5 * x) : 
  p + q = 40 :=
by
  sorry

end NUMINAMATH_GPT_initial_volume_of_mixture_l1963_196362


namespace NUMINAMATH_GPT_largest_x_satisfying_inequality_l1963_196366

theorem largest_x_satisfying_inequality :
  (∃ x : ℝ, 
    (∀ y : ℝ, |(y^2 - 4 * y - 39601)| ≥ |(y^2 + 4 * y - 39601)| → y ≤ x) ∧ 
    |(x^2 - 4 * x - 39601)| ≥ |(x^2 + 4 * x - 39601)|
  ) → x = 199 := 
sorry

end NUMINAMATH_GPT_largest_x_satisfying_inequality_l1963_196366


namespace NUMINAMATH_GPT_find_difference_l1963_196384

noncomputable def g : ℝ → ℝ := sorry    -- Definition of the function g (since it's graph-based and specific)

-- Given conditions
variables (c d : ℝ)
axiom h1 : Function.Injective g          -- g is an invertible function (injective functions have inverses)
axiom h2 : g c = d
axiom h3 : g d = 6

-- Theorem to prove
theorem find_difference : c - d = -2 :=
by {
  -- sorry is needed since the exact proof steps are not provided
  sorry
}

end NUMINAMATH_GPT_find_difference_l1963_196384
