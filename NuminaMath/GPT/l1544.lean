import Mathlib

namespace candy_group_size_l1544_154403

-- Define the given conditions
def num_candies : ℕ := 30
def num_groups : ℕ := 10

-- Define the statement that needs to be proven
theorem candy_group_size : num_candies / num_groups = 3 := 
by 
  sorry

end candy_group_size_l1544_154403


namespace two_pow_div_factorial_iff_l1544_154402

theorem two_pow_div_factorial_iff (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ n = 2^(k - 1)) ↔ (∃ m : ℕ, m > 0 ∧ 2^(n - 1) ∣ n!) :=
by
  sorry

end two_pow_div_factorial_iff_l1544_154402


namespace inequality_solution_l1544_154415

theorem inequality_solution :
  { x : ℝ | (x-1)/(x+4) ≤ 0 } = { x : ℝ | (-4 < x ∧ x ≤ 0) ∨ (x = 1) } :=
by 
  sorry

end inequality_solution_l1544_154415


namespace log2_3_value_l1544_154495

variables (a b log2 log3 : ℝ)

-- Define the conditions
axiom h1 : a = log2 + log3
axiom h2 : b = 1 + log2

-- Define the logarithmic requirement to be proved
theorem log2_3_value : log2 * log3 = (a - b + 1) / (b - 1) :=
sorry

end log2_3_value_l1544_154495


namespace baby_grasshoppers_l1544_154430

-- Definition for the number of grasshoppers on the plant
def grasshoppers_on_plant : ℕ := 7

-- Definition for the total number of grasshoppers found
def total_grasshoppers : ℕ := 31

-- The theorem to prove the number of baby grasshoppers under the plant
theorem baby_grasshoppers : 
  (total_grasshoppers - grasshoppers_on_plant) = 24 := 
by
  sorry

end baby_grasshoppers_l1544_154430


namespace geometric_sequence_fifth_term_l1544_154438

theorem geometric_sequence_fifth_term (α : ℕ → ℝ) (h : α 4 * α 5 * α 6 = 27) : α 5 = 3 :=
sorry

end geometric_sequence_fifth_term_l1544_154438


namespace pq_sum_l1544_154493

def single_digit (n : ℕ) : Prop := n < 10

theorem pq_sum (P Q : ℕ) (hP : single_digit P) (hQ : single_digit Q)
  (hSum : P * 100 + Q * 10 + Q + P * 110 + Q + Q * 111 = 876) : P + Q = 5 :=
by 
  -- Here we assume the expected outcome based on the problem solution
  sorry

end pq_sum_l1544_154493


namespace Melanie_gumballs_sale_l1544_154420

theorem Melanie_gumballs_sale (gumballs : ℕ) (price_per_gumball : ℕ) (total_price : ℕ) :
  gumballs = 4 →
  price_per_gumball = 8 →
  total_price = gumballs * price_per_gumball →
  total_price = 32 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end Melanie_gumballs_sale_l1544_154420


namespace half_angle_quadrants_l1544_154429

theorem half_angle_quadrants (α : ℝ) (k : ℤ) (hα : ∃ k : ℤ, (π/2 + k * 2 * π < α ∧ α < π + k * 2 * π)) : 
  ∃ k : ℤ, (π/4 + k * π < α/2 ∧ α/2 < π/2 + k * π) := 
sorry

end half_angle_quadrants_l1544_154429


namespace largest_integer_less_than_80_with_remainder_3_when_divided_by_5_l1544_154488

theorem largest_integer_less_than_80_with_remainder_3_when_divided_by_5 : 
  ∃ x : ℤ, x < 80 ∧ x % 5 = 3 ∧ (∀ y : ℤ, y < 80 ∧ y % 5 = 3 → y ≤ x) :=
sorry

end largest_integer_less_than_80_with_remainder_3_when_divided_by_5_l1544_154488


namespace ratio_of_width_to_length_is_correct_l1544_154405

-- Define the given conditions
def length := 10
def perimeter := 36

-- Define the width and the expected ratio
def width (l P : Nat) : Nat := (P - 2 * l) / 2
def ratio (w l : Nat) := w / l

-- Statement to prove that given the conditions, the ratio of width to length is 4/5
theorem ratio_of_width_to_length_is_correct :
  ratio (width length perimeter) length = 4 / 5 :=
by
  sorry

end ratio_of_width_to_length_is_correct_l1544_154405


namespace fraction_sum_l1544_154407

-- Define the fractions
def frac1: ℚ := 3/9
def frac2: ℚ := 5/12

-- The theorem statement
theorem fraction_sum : frac1 + frac2 = 3/4 := 
sorry

end fraction_sum_l1544_154407


namespace correct_equation_l1544_154404

variables (x : ℝ) (production_planned total_clothings : ℝ)
variables (increase_rate days_ahead : ℝ)

noncomputable def daily_production (x : ℝ) := x
noncomputable def total_production := 1000
noncomputable def production_per_day_due_to_overtime (x : ℝ) := x * (1 + 0.2 : ℝ)
noncomputable def original_completion_days (x : ℝ) := total_production / daily_production x
noncomputable def increased_production_completion_days (x : ℝ) := total_production / production_per_day_due_to_overtime x
noncomputable def days_difference := original_completion_days x - increased_production_completion_days x

theorem correct_equation : days_difference x = 2 := by
  sorry

end correct_equation_l1544_154404


namespace max_height_l1544_154497

-- Define the parabolic function h(t) representing the height of the soccer ball.
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 11

-- State that the maximum height of the soccer ball is 136 feet.
theorem max_height : ∃ t : ℝ, h t = 136 :=
by
  sorry

end max_height_l1544_154497


namespace at_least_one_variety_has_27_apples_l1544_154492

theorem at_least_one_variety_has_27_apples (total_apples : ℕ) (varieties : ℕ) 
  (h_total : total_apples = 105) (h_varieties : varieties = 4) : 
  ∃ v : ℕ, v ≥ 27 := 
sorry

end at_least_one_variety_has_27_apples_l1544_154492


namespace profit_correct_A_B_l1544_154447

noncomputable def profit_per_tire_A (batch_cost_A1 batch_cost_A2 cost_per_tire_A1 cost_per_tire_A2 sell_price_tire_A1 sell_price_tire_A2 produced_A : ℕ) : ℚ :=
  let cost_first_5000 := batch_cost_A1 + (cost_per_tire_A1 * 5000)
  let revenue_first_5000 := sell_price_tire_A1 * 5000
  let profit_first_5000 := revenue_first_5000 - cost_first_5000
  let cost_remaining := batch_cost_A2 + (cost_per_tire_A2 * (produced_A - 5000))
  let revenue_remaining := sell_price_tire_A2 * (produced_A - 5000)
  let profit_remaining := revenue_remaining - cost_remaining
  let total_profit := profit_first_5000 + profit_remaining
  total_profit / produced_A

noncomputable def profit_per_tire_B (batch_cost_B cost_per_tire_B sell_price_tire_B produced_B : ℕ) : ℚ :=
  let cost := batch_cost_B + (cost_per_tire_B * produced_B)
  let revenue := sell_price_tire_B * produced_B
  let profit := revenue - cost
  profit / produced_B

theorem profit_correct_A_B
  (batch_cost_A1 : ℕ := 22500) 
  (batch_cost_A2 : ℕ := 20000) 
  (cost_per_tire_A1 : ℕ := 8) 
  (cost_per_tire_A2 : ℕ := 6) 
  (sell_price_tire_A1 : ℕ := 20) 
  (sell_price_tire_A2 : ℕ := 18) 
  (produced_A : ℕ := 15000)
  (batch_cost_B : ℕ := 24000) 
  (cost_per_tire_B : ℕ := 7) 
  (sell_price_tire_B : ℕ := 19) 
  (produced_B : ℕ := 10000) :
  profit_per_tire_A batch_cost_A1 batch_cost_A2 cost_per_tire_A1 cost_per_tire_A2 sell_price_tire_A1 sell_price_tire_A2 produced_A = 9.17 ∧
  profit_per_tire_B batch_cost_B cost_per_tire_B sell_price_tire_B produced_B = 9.60 :=
by
  sorry

end profit_correct_A_B_l1544_154447


namespace determine_a_l1544_154480

theorem determine_a
  (h : ∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - a * x - 2) ≥ 0) : 
  a = 1 :=
sorry

end determine_a_l1544_154480


namespace smallest_y_divisible_l1544_154427

theorem smallest_y_divisible (y : ℕ) : 
  (y % 3 = 2) ∧ (y % 5 = 4) ∧ (y % 7 = 6) → y = 104 :=
by
  sorry

end smallest_y_divisible_l1544_154427


namespace min_value_expression_l1544_154440

noncomputable def min_expression (a b c : ℝ) : ℝ :=
  (a + b) / c + (b + c) / a + (c + a) / b + 3

theorem min_value_expression {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
  ∃ x, min_expression a b c = x ∧ x ≥ 9 := 
sorry

end min_value_expression_l1544_154440


namespace side_length_of_square_l1544_154450

theorem side_length_of_square (s : ℝ) (h : s^2 = 6 * (4 * s)) : s = 24 :=
by sorry

end side_length_of_square_l1544_154450


namespace light_match_first_l1544_154476

-- Define the conditions
def dark_room : Prop := true
def has_candle : Prop := true
def has_kerosene_lamp : Prop := true
def has_ready_to_use_stove : Prop := true
def has_single_match : Prop := true

-- Define the main question as a theorem
theorem light_match_first (h1 : dark_room) (h2 : has_candle) (h3 : has_kerosene_lamp) (h4 : has_ready_to_use_stove) (h5 : has_single_match) : true :=
by
  sorry

end light_match_first_l1544_154476


namespace flashlight_price_percentage_l1544_154482

theorem flashlight_price_percentage 
  (hoodie_price boots_price total_spent flashlight_price : ℝ)
  (discount_rate : ℝ)
  (h1 : hoodie_price = 80)
  (h2 : boots_price = 110)
  (h3 : discount_rate = 0.10)
  (h4 : total_spent = 195) 
  (h5 : total_spent = hoodie_price + ((1 - discount_rate) * boots_price) + flashlight_price) : 
  (flashlight_price / hoodie_price) * 100 = 20 :=
by
  sorry

end flashlight_price_percentage_l1544_154482


namespace paco_cookies_l1544_154437

theorem paco_cookies (initial_cookies: ℕ) (eaten_cookies: ℕ) (final_cookies: ℕ) (bought_cookies: ℕ) 
  (h1 : initial_cookies = 40)
  (h2 : eaten_cookies = 2)
  (h3 : final_cookies = 75)
  (h4 : initial_cookies - eaten_cookies + bought_cookies = final_cookies) :
  bought_cookies = 37 :=
by
  rw [h1, h2, h3] at h4
  sorry

end paco_cookies_l1544_154437


namespace h_of_neg_one_l1544_154416

def f (x : ℝ) : ℝ := 3 * x + 7
def g (x : ℝ) : ℝ := (f x) ^ 2 - 3
def h (x : ℝ) : ℝ := f (g x)

theorem h_of_neg_one :
  h (-1) = 298 :=
by
  sorry

end h_of_neg_one_l1544_154416


namespace man_older_than_son_l1544_154428

theorem man_older_than_son (S M : ℕ) (h1 : S = 18) (h2 : M + 2 = 2 * (S + 2)) : M - S = 20 :=
by
  sorry

end man_older_than_son_l1544_154428


namespace abs_neg_2023_l1544_154424

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l1544_154424


namespace parabola_equation_l1544_154448

def is_parabola (a b c x y : ℝ) : Prop :=
  y = a*x^2 + b*x + c

def has_vertex (h k a b c : ℝ) : Prop :=
  b = -2 * a * h ∧ c = k + a * h^2 

def contains_point (a b c x y : ℝ) : Prop :=
  y = a*x^2 + b*x + c

theorem parabola_equation (a b c : ℝ) :
  has_vertex 3 (-2) a b c ∧ contains_point a b c 5 6 → 
  a = 2 ∧ b = -12 ∧ c = 16 := by
  sorry

end parabola_equation_l1544_154448


namespace evaluate_expression_l1544_154431

theorem evaluate_expression : (4 + 6 + 7) / 3 - 2 / 3 = 5 := by
  sorry

end evaluate_expression_l1544_154431


namespace quadratic_sum_solutions_l1544_154461

noncomputable def sum_of_solutions (a b c : ℝ) : ℝ := 
  (-b/a)

theorem quadratic_sum_solutions : 
  ∀ x : ℝ, sum_of_solutions 1 (-9) (-45) = 9 := 
by
  intro x
  sorry

end quadratic_sum_solutions_l1544_154461


namespace simon_spending_l1544_154400

-- Assume entities and their properties based on the problem
def kabobStickCubes : Nat := 4
def slabCost : Nat := 25
def slabCubes : Nat := 80
def kabobSticksNeeded : Nat := 40

-- Theorem statement based on the problem analysis
theorem simon_spending : 
  (kabobSticksNeeded / (slabCubes / kabobStickCubes)) * slabCost = 50 := by
  sorry

end simon_spending_l1544_154400


namespace inverse_modulo_1000000_l1544_154449

def A : ℕ := 123456
def B : ℕ := 769230
def N : ℕ := 1053

theorem inverse_modulo_1000000 : (A * B * N) % 1000000 = 1 := 
  by 
  sorry

end inverse_modulo_1000000_l1544_154449


namespace weight_of_piece_l1544_154490

theorem weight_of_piece (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := 
by
  sorry

end weight_of_piece_l1544_154490


namespace symmetric_points_sum_l1544_154457

theorem symmetric_points_sum
  (a b : ℝ)
  (h1 : a = -3)
  (h2 : b = 2) :
  a + b = -1 := by
  sorry

end symmetric_points_sum_l1544_154457


namespace math_problem_l1544_154421

-- Definitions of conditions
def cond1 (x a y b z c : ℝ) : Prop := x / a + y / b + z / c = 1
def cond2 (x a y b z c : ℝ) : Prop := a / x + b / y + c / z = 0

-- Theorem statement
theorem math_problem (x a y b z c : ℝ)
  (h1 : cond1 x a y b z c) (h2 : cond2 x a y b z c) :
  (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 1 :=
by
  sorry

end math_problem_l1544_154421


namespace probability_event_B_l1544_154419

-- Define the type of trial outcomes, we're considering binary outcomes for simplicity
inductive Outcome
| win : Outcome
| lose : Outcome

open Outcome

def all_possible_outcomes := [
  [win, win, win],
  [win, win, win, lose],
  [win],
  [win],
  [lose],
  [win, win, lose, lose],
  [win, lose],
  [win, lose, win, lose, win],
  [win],
  [lose],
  [lose],
  [lose],
  [lose, win, win],
  [win, lose, lose, win],
  [lose, win, lose, lose],
  [win],
  [win],
  [lose],
  [lose],
  [lose, lose],
  [lose],
  [lose],
  [],
  [lose, lose, lose, lose]
]

-- Event A is winning a prize
def event_A := [
  [win, win, win],
  [win, win, win, lose],
  [win, win, lose, lose],
  [win, lose, win, lose, win],
  [win, lose, lose, win]
]

-- Event B is satisfying the condition \(a + b + c + d \leq 2\)
def event_B := [
  [lose],
  [win, lose],
  [lose, win],
  [win],
  [lose, lose],
  [lose, win, lose],
  [lose, lose, win],
  [lose, win, win],
  [win, lose, lose],
  [lose, lose, lose],
  []
]

-- Proof that the probability of event B equals 11/16
theorem probability_event_B : (event_B.length / all_possible_outcomes.length) = 11 / 16 := by
  sorry

end probability_event_B_l1544_154419


namespace height_relationship_height_at_90_l1544_154408

noncomputable def f (x : ℝ) : ℝ := (1/2) * x

theorem height_relationship :
  (∀ x : ℝ, (x = 10 -> f x = 5) ∧ (x = 30 -> f x = 15) ∧ (x = 50 -> f x = 25) ∧ (x = 70 -> f x = 35)) → (∀ x : ℝ, f x = (1/2) * x) :=
by
  sorry

theorem height_at_90 :
  f 90 = 45 :=
by
  sorry

end height_relationship_height_at_90_l1544_154408


namespace max_marks_l1544_154491

theorem max_marks {M : ℝ} (h : 0.90 * M = 550) : M = 612 :=
sorry

end max_marks_l1544_154491


namespace x_intercept_of_perpendicular_line_l1544_154484

theorem x_intercept_of_perpendicular_line (x y : ℝ) (h1 : 5 * x - 3 * y = 9) (y_intercept : ℝ) 
  (h2 : y_intercept = 4) : x = 20 / 3 :=
sorry

end x_intercept_of_perpendicular_line_l1544_154484


namespace sum_of_207_instances_of_33_difference_when_25_instances_of_112_are_subtracted_from_3000_difference_between_product_and_sum_of_12_and_13_l1544_154445

-- 1. Prove that 33 * 207 = 6831
theorem sum_of_207_instances_of_33 : 33 * 207 = 6831 := by
    sorry

-- 2. Prove that 3000 - 112 * 25 = 200
theorem difference_when_25_instances_of_112_are_subtracted_from_3000 : 3000 - 112 * 25 = 200 := by
    sorry

-- 3. Prove that 12 * 13 - (12 + 13) = 131
theorem difference_between_product_and_sum_of_12_and_13 : 12 * 13 - (12 + 13) = 131 := by
    sorry

end sum_of_207_instances_of_33_difference_when_25_instances_of_112_are_subtracted_from_3000_difference_between_product_and_sum_of_12_and_13_l1544_154445


namespace joe_initial_paint_l1544_154471
-- Use necessary imports

-- Define the hypothesis
def initial_paint_gallons (g : ℝ) :=
  (1 / 4) * g + (1 / 7) * (3 / 4) * g = 128.57

-- Define the theorem
theorem joe_initial_paint (P : ℝ) (h : initial_paint_gallons P) : P = 360 :=
  sorry

end joe_initial_paint_l1544_154471


namespace evaluate_expression_l1544_154479

theorem evaluate_expression : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 28 :=
by 
  sorry

end evaluate_expression_l1544_154479


namespace dave_final_tickets_l1544_154446

-- Define the initial number of tickets and operations
def initial_tickets : ℕ := 25
def tickets_spent_on_beanie : ℕ := 22
def tickets_won_after : ℕ := 15

-- Define the final number of tickets function
def final_tickets (initial : ℕ) (spent : ℕ) (won : ℕ) : ℕ :=
  initial - spent + won

-- Theorem stating that Dave would end up with 18 tickets given the conditions
theorem dave_final_tickets : final_tickets initial_tickets tickets_spent_on_beanie tickets_won_after = 18 :=
by
  -- Proof to be filled in
  sorry

end dave_final_tickets_l1544_154446


namespace largest_value_B_l1544_154409

theorem largest_value_B :
  let A := ((1 / 2) / (3 / 4))
  let B := (1 / ((2 / 3) / 4))
  let C := (((1 / 2) / 3) / 4)
  let E := ((1 / (2 / 3)) / 4)
  B > A ∧ B > C ∧ B > E :=
by
  sorry

end largest_value_B_l1544_154409


namespace computation_result_l1544_154466

def a : ℕ := 3
def b : ℕ := 5
def c : ℕ := 7

theorem computation_result :
  (a + b + c) ^ 2 + (a ^ 2 + b ^ 2 + c ^ 2) = 308 := by
  sorry

end computation_result_l1544_154466


namespace no_positive_integer_solution_l1544_154472

def is_solution (x y z t : ℕ) : Prop :=
  x^2 + 5 * y^2 = z^2 ∧ 5 * x^2 + y^2 = t^2

theorem no_positive_integer_solution :
  ¬ ∃ (x y z t : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < t ∧ is_solution x y z t :=
by
  sorry

end no_positive_integer_solution_l1544_154472


namespace price_per_vanilla_cookie_l1544_154483

theorem price_per_vanilla_cookie (P : ℝ) (h1 : 220 + 70 * P = 360) : P = 2 := 
by 
  sorry

end price_per_vanilla_cookie_l1544_154483


namespace symmetric_point_l1544_154465

theorem symmetric_point (x y : ℝ) (hx : x = -2) (hy : y = 3) (a b : ℝ) (hne : y = x + 1)
  (halfway : (a = (x + (-2)) / 2) ∧ (b = (y + 3) / 2) ∧ (2 * b = 2 * a + 2) ∧ (2 * b = 1)):
  (a, b) = (0, 1) :=
by
  sorry

end symmetric_point_l1544_154465


namespace solve_for_y_l1544_154475

-- Define the main theorem to be proven
theorem solve_for_y (y : ℤ) (h : 7 * (4 * y + 3) - 5 = -3 * (2 - 8 * y) + 5 * y) : y = 22 :=
by
  sorry

end solve_for_y_l1544_154475


namespace price_of_pants_l1544_154426

theorem price_of_pants (P : ℝ) 
  (h1 : (3 / 4) * P + P + (P + 10) = 340)
  (h2 : ∃ P, (3 / 4) * P + P + (P + 10) = 340) : 
  P = 120 :=
sorry

end price_of_pants_l1544_154426


namespace arithmetic_sequence_tenth_term_l1544_154478

/- 
  Define the arithmetic sequence in terms of its properties 
  and prove that the 10th term is 18.
-/

theorem arithmetic_sequence_tenth_term (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 2) (h_a5 : a 5 = 8) : a 10 = 18 := 
by 
  sorry

end arithmetic_sequence_tenth_term_l1544_154478


namespace mrs_hilt_rocks_l1544_154414

def garden_length := 10
def garden_width := 15
def rock_coverage := 1
def available_rocks := 64

theorem mrs_hilt_rocks :
  ∃ extra_rocks : ℕ, 2 * (garden_length + garden_width) <= available_rocks ∧ extra_rocks = available_rocks - 2 * (garden_length + garden_width) ∧ extra_rocks = 14 :=
by
  sorry

end mrs_hilt_rocks_l1544_154414


namespace problem_180_l1544_154418

variables (P Q : Prop)

theorem problem_180 (h : P → Q) : ¬ (P ∨ ¬Q) :=
sorry

end problem_180_l1544_154418


namespace smallest_denominator_is_168_l1544_154469

theorem smallest_denominator_is_168 (a b : ℕ) (h1: Nat.gcd a 600 = 1) (h2: Nat.gcd b 700 = 1) :
  ∃ k, Nat.gcd (7 * a + 6 * b) 4200 = k ∧ k = 25 ∧ (4200 / k) = 168 :=
sorry

end smallest_denominator_is_168_l1544_154469


namespace volume_of_prism_l1544_154462

-- Define the variables a, b, c and the conditions
variables (a b c : ℝ)

-- Given conditions
theorem volume_of_prism (h1 : a * b = 48) (h2 : b * c = 49) (h3 : a * c = 50) :
  a * b * c = 343 :=
by {
  sorry
}

end volume_of_prism_l1544_154462


namespace total_cost_of_pencils_l1544_154417

def pencil_price : ℝ := 0.20
def pencils_Tolu : ℕ := 3
def pencils_Robert : ℕ := 5
def pencils_Melissa : ℕ := 2

theorem total_cost_of_pencils :
  (pencil_price * pencils_Tolu + pencil_price * pencils_Robert + pencil_price * pencils_Melissa) = 2.00 := 
sorry

end total_cost_of_pencils_l1544_154417


namespace part1_part2_l1544_154498

-- Part (1) Lean 4 statement
theorem part1 {x : ℕ} (h : 0 < x ∧ 4 * (x + 2) < 18 + 2 * x) : x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 :=
sorry

-- Part (2) Lean 4 statement
theorem part2 (x : ℝ) (h1 : 5 * x + 2 ≥ 4 * x + 1) (h2 : (x + 1) / 4 > (x - 3) / 2 + 1) : -1 ≤ x ∧ x < 3 :=
sorry

end part1_part2_l1544_154498


namespace n_squared_plus_n_divisible_by_2_l1544_154487

theorem n_squared_plus_n_divisible_by_2 (n : ℤ) : 2 ∣ (n^2 + n) :=
sorry

end n_squared_plus_n_divisible_by_2_l1544_154487


namespace time_to_fill_box_correct_l1544_154468

def total_toys := 50
def mom_rate := 5
def mia_rate := 3

def time_to_fill_box (total_toys mom_rate mia_rate : ℕ) : ℚ :=
  let net_rate_per_cycle := mom_rate - mia_rate
  let cycles := ((total_toys - 1) / net_rate_per_cycle) + 1
  let total_seconds := cycles * 30
  total_seconds / 60

theorem time_to_fill_box_correct : time_to_fill_box total_toys mom_rate mia_rate = 12.5 :=
by
  sorry

end time_to_fill_box_correct_l1544_154468


namespace total_days_2010_to_2013_l1544_154489

theorem total_days_2010_to_2013 :
  let year2010_days := 365
  let year2011_days := 365
  let year2012_days := 366
  let year2013_days := 365
  year2010_days + year2011_days + year2012_days + year2013_days = 1461 := by
  sorry

end total_days_2010_to_2013_l1544_154489


namespace problem_simplify_and_evaluate_l1544_154406

theorem problem_simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 3 + 3) :
  (1 - (m / (m + 3))) / ((m^2 - 9) / (m^2 + 6 * m + 9)) = Real.sqrt 3 :=
by
  sorry

end problem_simplify_and_evaluate_l1544_154406


namespace graph_transformation_matches_B_l1544_154470

noncomputable def f (x : ℝ) : ℝ :=
  if (-3 : ℝ) ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0 -- Define this part to handle cases outside the given range.

noncomputable def g (x : ℝ) : ℝ :=
  f ((1 - x) / 2)

theorem graph_transformation_matches_B :
  g = some_graph_function_B := 
sorry

end graph_transformation_matches_B_l1544_154470


namespace solve_for_a_l1544_154467

theorem solve_for_a (x a : ℝ) (h : x = -2) (hx : 2 * x + 3 * a = 0) : a = 4 / 3 :=
by
  sorry

end solve_for_a_l1544_154467


namespace arithmetic_mean_frac_l1544_154443

theorem arithmetic_mean_frac (y b : ℝ) (h : y ≠ 0) : 
  (1 / 2 : ℝ) * ((y + b) / y + (2 * y - b) / y) = 1.5 := 
by 
  sorry

end arithmetic_mean_frac_l1544_154443


namespace inequality_proof_l1544_154473

theorem inequality_proof (x y z : ℝ) (hx : -1 < x) (hy : -1 < y) (hz : -1 < z) :
    (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
sorry

end inequality_proof_l1544_154473


namespace find_k_l1544_154474

theorem find_k 
  (h : ∀ x k : ℝ, x^2 + (k^2 - 4)*x + k - 1 = 0 → ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0):
  ∃ (k : ℝ), k = -2 :=
sorry

end find_k_l1544_154474


namespace train_trip_length_l1544_154460

theorem train_trip_length (x D : ℝ) (h1 : D > 0) (h2 : x > 0) 
(h3 : 2 + 3 * (D - 2 * x) / (2 * x) + 1 = (x + 240) / x + 1 + 3 * (D - 2 * x - 120) / (2 * x) - 0.5) 
(h4 : 3 + 3 * (D - 2 * x) / (2 * x) = 7) :
  D = 640 :=
by
  sorry

end train_trip_length_l1544_154460


namespace agent_commission_calculation_l1544_154432

-- Define the conditions
def total_sales : ℝ := 250
def commission_rate : ℝ := 0.05

-- Define the commission calculation function
def calculate_commission (sales : ℝ) (rate : ℝ) : ℝ :=
  sales * rate

-- Proposition stating the desired commission
def agent_commission_is_correct : Prop :=
  calculate_commission total_sales commission_rate = 12.5

-- State the proof problem
theorem agent_commission_calculation : agent_commission_is_correct :=
by sorry

end agent_commission_calculation_l1544_154432


namespace father_age_three_times_xiaojun_after_years_l1544_154442

theorem father_age_three_times_xiaojun_after_years (years_passed : ℕ) (xiaojun_current_age : ℕ) (father_current_age : ℕ) 
  (h1 : xiaojun_current_age = 5) (h2 : father_current_age = 31) (h3 : years_passed = 8) :
  father_current_age + years_passed = 3 * (xiaojun_current_age + years_passed) := by
  sorry

end father_age_three_times_xiaojun_after_years_l1544_154442


namespace inscribed_circle_radius_third_of_circle_l1544_154412

noncomputable def inscribed_circle_radius (R : ℝ) : ℝ := 
  R * (Real.sqrt 3 - 1) / 2

theorem inscribed_circle_radius_third_of_circle (R : ℝ) (hR : R = 5) :
  inscribed_circle_radius R = 5 * (Real.sqrt 3 - 1) / 2 := by
  sorry

end inscribed_circle_radius_third_of_circle_l1544_154412


namespace union_sets_l1544_154481

def setA : Set ℝ := { x | -1 ≤ x ∧ x < 3 }

def setB : Set ℝ := { x | x^2 - 7 * x + 10 ≤ 0 }

theorem union_sets : setA ∪ setB = { x | -1 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end union_sets_l1544_154481


namespace triangles_exist_l1544_154413

def exists_triangles : Prop :=
  ∃ (T : Fin 100 → Type) 
    (h : (i : Fin 100) → ℝ) 
    (A : (i : Fin 100) → ℝ)
    (is_isosceles : (i : Fin 100) → Prop),
    (∀ i : Fin 100, is_isosceles i) ∧
    (∀ i : Fin 99, h (i + 1) = 200 * h i) ∧
    (∀ i : Fin 99, A (i + 1) = A i / 20000) ∧
    (∀ i : Fin 100, 
      ¬(∃ (cover : (Fin 99) → Type),
        (∀ j : Fin 99, cover j = T j) ∧
        (∀ j : Fin 99, ∀ k : Fin 100, k ≠ i → ¬(cover j = T k))))

theorem triangles_exist : exists_triangles :=
sorry

end triangles_exist_l1544_154413


namespace chastity_leftover_money_l1544_154455

theorem chastity_leftover_money (n_lollipops : ℕ) (price_lollipop : ℝ) (n_gummies : ℕ) (price_gummy : ℝ) (initial_money : ℝ) :
  n_lollipops = 4 →
  price_lollipop = 1.50 →
  n_gummies = 2 →
  price_gummy = 2 →
  initial_money = 15 →
  initial_money - ((n_lollipops * price_lollipop) + (n_gummies * price_gummy)) = 5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end chastity_leftover_money_l1544_154455


namespace seashells_second_day_l1544_154485

theorem seashells_second_day (x : ℕ) (h1 : 5 + x + 2 * (5 + x) = 36) : x = 7 :=
by
  sorry

end seashells_second_day_l1544_154485


namespace construction_paper_initial_count_l1544_154411

theorem construction_paper_initial_count 
    (b r d : ℕ)
    (ratio_cond : b = 2 * r)
    (daily_usage : ∀ n : ℕ, n ≤ d → n * 1 = b ∧ n * 3 = r)
    (last_day_cond : 0 = b ∧ 15 = r):
    b + r = 135 :=
sorry

end construction_paper_initial_count_l1544_154411


namespace some_number_is_ten_l1544_154444

theorem some_number_is_ten (x : ℕ) (h : 5 ^ 29 * 4 ^ 15 = 2 * x ^ 29) : x = 10 :=
by
  sorry

end some_number_is_ten_l1544_154444


namespace number_of_pears_in_fruit_gift_set_l1544_154410

theorem number_of_pears_in_fruit_gift_set 
  (F : ℕ) 
  (h1 : (2 / 9) * F = 10) 
  (h2 : 2 / 5 * F = 18) : 
  (2 / 5) * F = 18 :=
by 
  -- Sorry is used to skip the actual proof for now
  sorry

end number_of_pears_in_fruit_gift_set_l1544_154410


namespace find_fixed_point_on_ellipse_l1544_154459

theorem find_fixed_point_on_ellipse (a b c : ℝ) (h_gt_zero : a > b ∧ b > 0)
    (h_ellipse : ∀ (P : ℝ × ℝ), (P.1 ^ 2 / a ^ 2) + (P.2 ^ 2 / b ^ 2) = 1)
    (A1 A2 : ℝ × ℝ)
    (h_A1 : A1 = (-a, 0))
    (h_A2 : A2 = (a, 0))
    (MC : ℝ) (h_MC : MC = (a^2 + b^2) / c) :
  ∃ (M : ℝ × ℝ), M = (MC, 0) := 
sorry

end find_fixed_point_on_ellipse_l1544_154459


namespace markdown_calculation_l1544_154436

noncomputable def markdown_percentage (P S : ℝ) (h_inc : P = S * 1.1494) : ℝ :=
  1 - (1 / 1.1494)

theorem markdown_calculation (P S : ℝ) (h_sale : S = P * (1 - markdown_percentage P S sorry / 100)) (h_inc : P = S * 1.1494) :
  markdown_percentage P S h_inc = 12.99 := 
sorry

end markdown_calculation_l1544_154436


namespace percent_increase_l1544_154452

theorem percent_increase (N : ℝ) (h : (1 / 7) * N = 1) : 
  N = 7 ∧ (N - (4 / 7)) / (4 / 7) * 100 = 1125.0000000000002 := 
by 
  sorry

end percent_increase_l1544_154452


namespace tan_alpha_plus_pi_cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two_l1544_154464

open Real

theorem tan_alpha_plus_pi (α : ℝ)
  (h1 : sin α = 3 / 5)
  (h2 : π / 2 < α ∧ α < π) :
  tan (α + π) = -3 / 4 :=
sorry

theorem cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two (α : ℝ)
  (h1 : sin α = 3 / 5)
  (h2 : π / 2 < α ∧ α < π) :
  cos (α - π / 2) * sin (α + 3 * π / 2) = 12 / 25 :=
sorry

end tan_alpha_plus_pi_cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two_l1544_154464


namespace actual_area_of_region_l1544_154477

-- Problem Definitions
def map_scale : ℕ := 300000
def map_area_cm_squared : ℕ := 24

-- The actual area calculation should be 216 km²
theorem actual_area_of_region :
  let scale_factor_distance := map_scale
  let scale_factor_area := scale_factor_distance ^ 2
  let actual_area_cm_squared := map_area_cm_squared * scale_factor_area
  let actual_area_km_squared := actual_area_cm_squared / 10^10
  actual_area_km_squared = 216 := 
by
  sorry

end actual_area_of_region_l1544_154477


namespace find_number_of_spiders_l1544_154456

theorem find_number_of_spiders (S : ℕ) (h1 : (1 / 2) * S = 5) : S = 10 := sorry

end find_number_of_spiders_l1544_154456


namespace arithmetic_expression_evaluation_l1544_154433

theorem arithmetic_expression_evaluation : 
  2000 - 80 + 200 - 120 = 2000 := by
  sorry

end arithmetic_expression_evaluation_l1544_154433


namespace propP_necessary_but_not_sufficient_l1544_154422

open Function Real

variable (f : ℝ → ℝ)

-- Conditions: differentiable function f and the proposition Q
def diff_and_propQ (h_deriv : Differentiable ℝ f) : Prop :=
∀ x : ℝ, abs (deriv f x) < 2018

-- Proposition P
def propP : Prop :=
∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018

-- Final statement
theorem propP_necessary_but_not_sufficient (h_deriv : Differentiable ℝ f) (hQ : diff_and_propQ f h_deriv) : 
  (∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018) ∧ 
  ¬(∀ x : ℝ, abs (deriv f x) < 2018 ↔ ∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018) :=
by
  sorry

end propP_necessary_but_not_sufficient_l1544_154422


namespace functional_eq_solutions_l1544_154486

theorem functional_eq_solutions
  (f : ℚ → ℚ)
  (h0 : f 0 = 0)
  (h1 : ∀ x y : ℚ, f (f x + f y) = x + y) :
  ∀ x : ℚ, f x = x ∨ f x = -x := 
sorry

end functional_eq_solutions_l1544_154486


namespace total_first_year_students_l1544_154453

theorem total_first_year_students (males : ℕ) (sample_size : ℕ) (female_in_sample : ℕ) (N : ℕ)
  (h1 : males = 570)
  (h2 : sample_size = 110)
  (h3 : female_in_sample = 53)
  (h4 : N = ((sample_size - female_in_sample) * males) / (sample_size - (sample_size - female_in_sample)))
  : N = 1100 := 
by
  sorry

end total_first_year_students_l1544_154453


namespace base_number_exponent_l1544_154463

theorem base_number_exponent (x : ℝ) (h : ((x^4) * 3.456789) ^ 12 = y) (has_24_digits : true) : x = 10^12 :=
  sorry

end base_number_exponent_l1544_154463


namespace pos_sol_eq_one_l1544_154425

theorem pos_sol_eq_one (n : ℕ) (hn : 1 < n) :
  ∀ x : ℝ, 0 < x → (x ^ n - n * x + n - 1 = 0) → x = 1 := by
  -- The proof goes here
  sorry

end pos_sol_eq_one_l1544_154425


namespace evaluate_expression_l1544_154439

theorem evaluate_expression (x : ℝ) (h : |7 - 8 * (x - 12)| - |5 - 11| = 73) : x = 3 :=
  sorry

end evaluate_expression_l1544_154439


namespace sequence_integral_terms_l1544_154441

theorem sequence_integral_terms (x : ℕ → ℝ) (h1 : ∀ n, x n ≠ 0)
  (h2 : ∀ n > 2, x n = (x (n - 2) * x (n - 1)) / (2 * x (n - 2) - x (n - 1))) :
  (∀ n, ∃ k : ℤ, x n = k) → x 1 = x 2 :=
by
  sorry

end sequence_integral_terms_l1544_154441


namespace power_inequality_l1544_154423

variable (a b c : ℝ)

theorem power_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a^3 + b^3 + c^3) ≥ a * b^2 + a^2 * b + b * c^2 + b^2 * c + a * c^2 + a^2 * c :=
by sorry

end power_inequality_l1544_154423


namespace lines_parallel_a_eq_sqrt2_l1544_154435

theorem lines_parallel_a_eq_sqrt2 (a : ℝ) (h1 : 1 ≠ 0) :
  (∀ a ≠ 0, ((- (1 / (2 * a))) = (- a / 2)) → a = Real.sqrt 2) :=
by
  sorry

end lines_parallel_a_eq_sqrt2_l1544_154435


namespace ellipse_product_axes_l1544_154401

/-- Prove that the product of the lengths of the major and minor axes (AB)(CD) of an ellipse
is 240, given the following conditions:
- Point O is the center of the ellipse.
- Point F is one focus of the ellipse.
- OF = 8
- The diameter of the inscribed circle of triangle OCF is 4.
- OA = OB = a
- OC = OD = b
- a² - b² = 64
- a - b = 4
-/
theorem ellipse_product_axes (a b : ℝ) (OF : ℝ) (d_inscribed_circle : ℝ) 
  (h1 : OF = 8) (h2 : d_inscribed_circle = 4) (h3 : a^2 - b^2 = 64) 
  (h4 : a - b = 4) : (2 * a) * (2 * b) = 240 :=
sorry

end ellipse_product_axes_l1544_154401


namespace polygon_interior_exterior_angles_l1544_154496

theorem polygon_interior_exterior_angles (n : ℕ) :
  (n - 2) * 180 = 360 + 720 → n = 8 := 
by {
  sorry
}

end polygon_interior_exterior_angles_l1544_154496


namespace expand_binomial_trinomial_l1544_154451

theorem expand_binomial_trinomial (x y z : ℝ) :
  (x + 12) * (3 * y + 4 * z + 15) = 3 * x * y + 4 * x * z + 15 * x + 36 * y + 48 * z + 180 :=
by sorry

end expand_binomial_trinomial_l1544_154451


namespace correct_points_per_answer_l1544_154434

noncomputable def points_per_correct_answer (total_questions : ℕ) 
  (answered_correctly : ℕ) (final_score : ℝ) (penalty_per_incorrect : ℝ)
  (total_incorrect : ℕ := total_questions - answered_correctly) 
  (points_subtracted : ℝ := total_incorrect * penalty_per_incorrect) 
  (earned_points : ℝ := final_score + points_subtracted) : ℝ := 
    earned_points / answered_correctly

theorem correct_points_per_answer :
  points_per_correct_answer 120 104 100 0.25 = 1 := 
by 
  sorry

end correct_points_per_answer_l1544_154434


namespace shaded_area_l1544_154454

theorem shaded_area 
  (side_of_square : ℝ)
  (arc_radius : ℝ)
  (side_length_eq_sqrt_two : side_of_square = Real.sqrt 2)
  (radius_eq_one : arc_radius = 1) :
  let square_area := 4
  let sector_area := 3 * Real.pi
  let shaded_area := square_area + sector_area
  shaded_area = 4 + 3 * Real.pi :=
by
  sorry

end shaded_area_l1544_154454


namespace probability_of_two_digit_number_l1544_154458

def total_elements_in_set : ℕ := 961
def two_digit_elements_in_set : ℕ := 60

theorem probability_of_two_digit_number :
  (two_digit_elements_in_set : ℚ) / total_elements_in_set = 60 / 961 := by
  sorry

end probability_of_two_digit_number_l1544_154458


namespace no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10_l1544_154494

theorem no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10 :
  ¬ ∃ x : ℝ, x^4 + (x + 1)^4 + (x + 2)^4 = (x + 3)^4 + 10 :=
by {
  sorry
}

end no_real_solution_x_4_plus_x_plus1_4_plus_x_plus2_4_eq_x_plus3_4_plus_10_l1544_154494


namespace election_1002nd_k_election_1001st_k_l1544_154499

variable (k : ℕ)

noncomputable def election_in_1002nd_round_max_k : Prop :=
  ∀ (n : ℕ), (n = 2002) → (k ≤ n - 1) → k = 2001 → -- The conditions include the number of candidates 'n', and specifying that 'k' being the maximum initially means k ≤ 2001.
  true

noncomputable def election_in_1001st_round_max_k : Prop :=
  ∀ (n : ℕ), (n = 2002) → (k ≤ n - 1) → k = 1 → -- Similarly, these conditions specify the initial maximum placement as 1 when elected in 1001st round.
  true

-- Definitions specifying the problem to identify max k for given rounds
theorem election_1002nd_k : election_in_1002nd_round_max_k k := sorry

theorem election_1001st_k : election_in_1001st_round_max_k k := sorry

end election_1002nd_k_election_1001st_k_l1544_154499
