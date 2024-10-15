import Mathlib

namespace NUMINAMATH_GPT_volume_of_adjacent_cubes_l1196_119608

theorem volume_of_adjacent_cubes 
(side_length count : ℝ) 
(h_side : side_length = 5) 
(h_count : count = 5) : 
  (count * side_length ^ 3) = 625 :=
by
  -- Proof steps (skipped)
  sorry

end NUMINAMATH_GPT_volume_of_adjacent_cubes_l1196_119608


namespace NUMINAMATH_GPT_general_formula_sequence_l1196_119698

theorem general_formula_sequence (a : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h_rec : ∀ n : ℕ, n > 0 → a (n + 1) = 4 * a n + 3) :
  ∀ n : ℕ, n > 0 → a n = 4^n - 1 :=
by 
  sorry

end NUMINAMATH_GPT_general_formula_sequence_l1196_119698


namespace NUMINAMATH_GPT_remainder_when_divided_by_17_l1196_119651

theorem remainder_when_divided_by_17
  (N k : ℤ)
  (h : N = 357 * k + 36) :
  N % 17 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_17_l1196_119651


namespace NUMINAMATH_GPT_total_price_eq_2500_l1196_119618

theorem total_price_eq_2500 (C P : ℕ)
  (hC : C = 2000)
  (hE : C + 500 + P = 6 * P)
  : C + P = 2500 := 
by
  sorry

end NUMINAMATH_GPT_total_price_eq_2500_l1196_119618


namespace NUMINAMATH_GPT_find_unknown_rate_l1196_119655

theorem find_unknown_rate :
  ∃ x : ℝ, (300 + 750 + 2 * x) / 10 = 170 ↔ x = 325 :=
by
    sorry

end NUMINAMATH_GPT_find_unknown_rate_l1196_119655


namespace NUMINAMATH_GPT_John_total_weekly_consumption_l1196_119668

/-
  Prove that John's total weekly consumption of water, milk, and juice in quarts is 49.25 quarts, 
  given the specified conditions on his daily and periodic consumption.
-/

def John_consumption_problem (gallons_per_day : ℝ) (pints_every_other_day : ℝ) (ounces_every_third_day : ℝ) 
  (quarts_per_gallon : ℝ) (quarts_per_pint : ℝ) (quarts_per_ounce : ℝ) : ℝ :=
  let water_per_day := gallons_per_day * quarts_per_gallon
  let water_per_week := water_per_day * 7
  let milk_per_other_day := pints_every_other_day * quarts_per_pint
  let milk_per_week := milk_per_other_day * 4 -- assuming he drinks milk 4 times a week
  let juice_per_third_day := ounces_every_third_day * quarts_per_ounce
  let juice_per_week := juice_per_third_day * 2 -- assuming he drinks juice 2 times a week
  water_per_week + milk_per_week + juice_per_week

theorem John_total_weekly_consumption :
  John_consumption_problem 1.5 3 20 4 (1/2) (1/32) = 49.25 :=
by
  sorry

end NUMINAMATH_GPT_John_total_weekly_consumption_l1196_119668


namespace NUMINAMATH_GPT_numbers_sum_and_difference_l1196_119691

variables (a b : ℝ)

theorem numbers_sum_and_difference (h : a / b = -1) : a + b = 0 ∧ (a - b = 2 * b ∨ a - b = -2 * b) :=
by {
  sorry
}

end NUMINAMATH_GPT_numbers_sum_and_difference_l1196_119691


namespace NUMINAMATH_GPT_optimalBananaBuys_l1196_119605

noncomputable def bananaPrices : List ℕ := [1, 5, 1, 6, 7, 8, 1, 8, 7, 2, 7, 8, 1, 9, 2, 8, 7, 1]

def days := List.range 18

def computeOptimalBuys : List ℕ :=
  sorry -- Implement the logic to compute the optimal number of bananas to buy each day.

theorem optimalBananaBuys :
  computeOptimalBuys = [4, 0, 0, 3, 0, 0, 7, 0, 0, 1, 0, 0, 4, 0, 0, 3, 0, 1] :=
sorry

end NUMINAMATH_GPT_optimalBananaBuys_l1196_119605


namespace NUMINAMATH_GPT_older_brother_stamps_l1196_119602

variable (y o : ℕ)

def condition1 : Prop := o = 2 * y + 1
def condition2 : Prop := o + y = 25

theorem older_brother_stamps (h1 : condition1 y o) (h2 : condition2 y o) : o = 17 :=
by
  sorry

end NUMINAMATH_GPT_older_brother_stamps_l1196_119602


namespace NUMINAMATH_GPT_problem_statement_l1196_119685

theorem problem_statement : 
  (∀ (base : ℤ) (exp : ℕ), (-3) = base ∧ 2 = exp → (base ^ exp ≠ -9)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1196_119685


namespace NUMINAMATH_GPT_bob_stickers_l1196_119601

variables {B T D : ℕ}

theorem bob_stickers (h1 : D = 72) (h2 : T = 3 * B) (h3 : D = 2 * T) : B = 12 :=
by
  sorry

end NUMINAMATH_GPT_bob_stickers_l1196_119601


namespace NUMINAMATH_GPT_robert_elizabeth_age_difference_l1196_119692

theorem robert_elizabeth_age_difference 
  (patrick_age_1_5_times_robert : ∀ (robert_age : ℝ), ∃ (patrick_age : ℝ), patrick_age = 1.5 * robert_age)
  (elizabeth_born_after_richard : ∀ (richard_age : ℝ), ∃ (elizabeth_age : ℝ), elizabeth_age = richard_age - 7 / 12)
  (elizabeth_younger_by_4_5_years : ∀ (patrick_age : ℝ), ∃ (elizabeth_age : ℝ), elizabeth_age = patrick_age - 4.5)
  (robert_will_be_30_3_after_2_5_years : ∃ (robert_age_current : ℝ), robert_age_current = 30.3 - 2.5) :
  ∃ (years : ℤ) (months : ℤ), years = 9 ∧ months = 4 := by
  sorry

end NUMINAMATH_GPT_robert_elizabeth_age_difference_l1196_119692


namespace NUMINAMATH_GPT_fraction_problem_l1196_119627

noncomputable def zero_point_one_five : ℚ := 5 / 33
noncomputable def two_point_four_zero_three : ℚ := 2401 / 999

theorem fraction_problem :
  (zero_point_one_five / two_point_four_zero_three) = (4995 / 79233) :=
by
  sorry

end NUMINAMATH_GPT_fraction_problem_l1196_119627


namespace NUMINAMATH_GPT_adam_change_l1196_119687

theorem adam_change : 
  let amount : ℝ := 5.00
  let cost : ℝ := 4.28
  amount - cost = 0.72 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_adam_change_l1196_119687


namespace NUMINAMATH_GPT_x_cubed_inverse_cubed_l1196_119679

theorem x_cubed_inverse_cubed (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 :=
by sorry

end NUMINAMATH_GPT_x_cubed_inverse_cubed_l1196_119679


namespace NUMINAMATH_GPT_weight_loss_percentage_l1196_119652

theorem weight_loss_percentage {W : ℝ} (hW : 0 < W) :
  (((W - ((1 - 0.13 + 0.02 * (1 - 0.13)) * W)) / W) * 100) = 11.26 :=
by
  sorry

end NUMINAMATH_GPT_weight_loss_percentage_l1196_119652


namespace NUMINAMATH_GPT_largest_smallest_difference_l1196_119676

theorem largest_smallest_difference (a b c d : ℚ) (h₁ : a = 2.5) (h₂ : b = 22/13) (h₃ : c = 0.7) (h₄ : d = 32/33) :
  max (max a b) (max c d) - min (min a b) (min c d) = 1.8 := by
  sorry

end NUMINAMATH_GPT_largest_smallest_difference_l1196_119676


namespace NUMINAMATH_GPT_shortest_distance_l1196_119686

-- The initial position of the cowboy.
def initial_position : ℝ × ℝ := (-2, -6)

-- The position of the cabin relative to the cowboy's initial position.
def cabin_position : ℝ × ℝ := (10, -15)

-- The equation of the stream flowing due northeast.
def stream_equation : ℝ → ℝ := id  -- y = x

-- Function to calculate the distance between two points (x1, y1) and (x2, y2).
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Calculate the reflection point of C over y = x.
def reflection_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

-- Main proof statement: shortest distance the cowboy can travel.
theorem shortest_distance : distance initial_position (reflection_point initial_position) +
                            distance (reflection_point initial_position) cabin_position = 8 +
                            Real.sqrt 545 :=
by
  sorry

end NUMINAMATH_GPT_shortest_distance_l1196_119686


namespace NUMINAMATH_GPT_chess_tournament_points_distribution_l1196_119637

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end NUMINAMATH_GPT_chess_tournament_points_distribution_l1196_119637


namespace NUMINAMATH_GPT_pages_left_after_all_projects_l1196_119600

-- Definitions based on conditions
def initial_pages : ℕ := 120
def pages_for_science : ℕ := (initial_pages * 25) / 100
def pages_for_math : ℕ := 10
def pages_after_science_and_math : ℕ := initial_pages - pages_for_science - pages_for_math
def pages_for_history : ℕ := (initial_pages * 15) / 100
def pages_after_history : ℕ := pages_after_science_and_math - pages_for_history
def remaining_pages : ℕ := pages_after_history / 2

theorem pages_left_after_all_projects :
  remaining_pages = 31 :=
  by
  sorry

end NUMINAMATH_GPT_pages_left_after_all_projects_l1196_119600


namespace NUMINAMATH_GPT_problem_l1196_119624

noncomputable def f (x φ : ℝ) : ℝ := 4 * Real.cos (3 * x + φ)

theorem problem 
  (φ : ℝ) (x1 x2 : ℝ)
  (hφ : |φ| < Real.pi / 2)
  (h_symm : ∀ x, f x φ = f (2 * (11 * Real.pi / 12) - x) φ)
  (hx1x2 : x1 ≠ x2)
  (hx1_range : -7 * Real.pi / 12 < x1 ∧ x1 < -Real.pi / 12)
  (hx2_range : -7 * Real.pi / 12 < x2 ∧ x2 < -Real.pi / 12)
  (h_eq : f x1 φ = f x2 φ) : 
  f (x1 + x2) (-Real.pi / 4) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_problem_l1196_119624


namespace NUMINAMATH_GPT_inequality_proof_l1196_119674

theorem inequality_proof (a b c d : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) (h5 : a * d = b * c) :
  (a - d) ^ 2 ≥ 4 * d + 8 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1196_119674


namespace NUMINAMATH_GPT_smaller_number_l1196_119617

theorem smaller_number {a b : ℕ} (h_ratio : b = 5 * a / 2) (h_lcm : Nat.lcm a b = 160) : a = 64 := 
by
  sorry

end NUMINAMATH_GPT_smaller_number_l1196_119617


namespace NUMINAMATH_GPT_least_multiple_of_25_gt_390_l1196_119630

theorem least_multiple_of_25_gt_390 : ∃ n : ℕ, n * 25 > 390 ∧ (∀ m : ℕ, m * 25 > 390 → m * 25 ≥ n * 25) ∧ n * 25 = 400 :=
by
  sorry

end NUMINAMATH_GPT_least_multiple_of_25_gt_390_l1196_119630


namespace NUMINAMATH_GPT_total_payment_is_correct_l1196_119629

def daily_rental_cost : ℝ := 30
def per_mile_cost : ℝ := 0.25
def one_time_service_charge : ℝ := 15
def rent_duration : ℝ := 4
def distance_driven : ℝ := 500

theorem total_payment_is_correct :
  (daily_rental_cost * rent_duration + per_mile_cost * distance_driven + one_time_service_charge) = 260 := 
by
  sorry

end NUMINAMATH_GPT_total_payment_is_correct_l1196_119629


namespace NUMINAMATH_GPT_find_f_neg_2_l1196_119607

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 3 * x + 4 else 7 - 3 * x

theorem find_f_neg_2 : f (-2) = 13 := by
  sorry

end NUMINAMATH_GPT_find_f_neg_2_l1196_119607


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l1196_119610

-- Let x be a real number
variable {x : ℝ}

-- The first equation and its solutions
def equation1 (x : ℝ) : Prop := (x - 1) ^ 2 - 25 = 0

-- Asserting that the solutions to the first equation are x = 6 or x = -4
theorem solve_equation1 (x : ℝ) : equation1 x ↔ x = 6 ∨ x = -4 :=
by
  sorry

-- The second equation and its solution
def equation2 (x : ℝ) : Prop := (1 / 4) * (2 * x + 3) ^ 3 = 16

-- Asserting that the solution to the second equation is x = 1/2
theorem solve_equation2 (x : ℝ) : equation2 x ↔ x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l1196_119610


namespace NUMINAMATH_GPT_remainder_calculation_l1196_119639

theorem remainder_calculation :
  (7 * 10^23 + 3^25) % 11 = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_calculation_l1196_119639


namespace NUMINAMATH_GPT_probability_red_blue_l1196_119696

-- Declare the conditions (probabilities for white, green and yellow marbles).
variables (total_marbles : ℕ) (P_white P_green P_yellow P_red_blue : ℚ)
-- implicitly P_white, P_green, P_yellow, P_red_blue are probabilities, therefore between 0 and 1

-- Assume the conditions given in the problem
axiom total_marbles_condition : total_marbles = 250
axiom P_white_condition : P_white = 2 / 5
axiom P_green_condition : P_green = 1 / 4
axiom P_yellow_condition : P_yellow = 1 / 10

-- Proving the required probability of red or blue marbles
theorem probability_red_blue :
  P_red_blue = 1 - (P_white + P_green + P_yellow) :=
sorry

end NUMINAMATH_GPT_probability_red_blue_l1196_119696


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1196_119654

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 2 → 2 / a < 1) ∧ (2 / a < 1 → a > 2 ∨ a < 0) :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1196_119654


namespace NUMINAMATH_GPT_last_score_is_80_l1196_119650

-- Define the list of scores
def scores : List ℕ := [71, 76, 80, 82, 91]

-- Define the total sum of the scores
def total_sum : ℕ := 400

-- Define the condition that the average after each score is an integer
def average_integer_condition (scores : List ℕ) (total_sum : ℕ) : Prop :=
  ∀ (sublist : List ℕ), sublist ≠ [] → sublist ⊆ scores → 
  (sublist.sum / sublist.length : ℕ) * sublist.length = sublist.sum

-- Define the proposition to prove that the last score entered must be 80
theorem last_score_is_80 : ∃ (last_score : ℕ), (last_score = 80) ∧
  average_integer_condition scores total_sum :=
sorry

end NUMINAMATH_GPT_last_score_is_80_l1196_119650


namespace NUMINAMATH_GPT_merchant_gross_profit_l1196_119666

noncomputable def purchase_price : ℝ := 48
noncomputable def markup_rate : ℝ := 0.40
noncomputable def discount_rate : ℝ := 0.20

theorem merchant_gross_profit :
  ∃ S : ℝ, S = purchase_price + markup_rate * S ∧ 
  ((S - discount_rate * S) - purchase_price = 16) :=
by
  sorry

end NUMINAMATH_GPT_merchant_gross_profit_l1196_119666


namespace NUMINAMATH_GPT_roy_missed_days_l1196_119612

theorem roy_missed_days {hours_per_day days_per_week actual_hours_week missed_days : ℕ}
    (h1 : hours_per_day = 2)
    (h2 : days_per_week = 5)
    (h3 : actual_hours_week = 6)
    (expected_hours_week : ℕ := hours_per_day * days_per_week)
    (missed_hours : ℕ := expected_hours_week - actual_hours_week)
    (missed_days := missed_hours / hours_per_day) :
  missed_days = 2 := by
  sorry

end NUMINAMATH_GPT_roy_missed_days_l1196_119612


namespace NUMINAMATH_GPT_pizza_slices_count_l1196_119631

/-
  We ordered 21 pizzas. Each pizza has 8 slices. 
  Prove that the total number of slices of pizza is 168.
-/

theorem pizza_slices_count :
  (21 * 8) = 168 :=
by
  sorry

end NUMINAMATH_GPT_pizza_slices_count_l1196_119631


namespace NUMINAMATH_GPT_gecko_cricket_eating_l1196_119646

theorem gecko_cricket_eating :
  ∀ (total_crickets : ℕ) (first_day_percent : ℚ) (second_day_less : ℕ),
    total_crickets = 70 →
    first_day_percent = 0.3 →
    second_day_less = 6 →
    let first_day_crickets := total_crickets * first_day_percent
    let second_day_crickets := first_day_crickets - second_day_less
    total_crickets - first_day_crickets - second_day_crickets = 34 :=
by
  intros total_crickets first_day_percent second_day_less h_total h_percent h_less
  let first_day_crickets := total_crickets * first_day_percent
  let second_day_crickets := first_day_crickets - second_day_less
  have : total_crickets - first_day_crickets - second_day_crickets = 34 := sorry
  exact this

end NUMINAMATH_GPT_gecko_cricket_eating_l1196_119646


namespace NUMINAMATH_GPT_salary_for_may_l1196_119684

theorem salary_for_may
  (J F M A May : ℝ)
  (h1 : J + F + M + A = 32000)
  (h2 : F + M + A + May = 34400)
  (h3 : J = 4100) :
  May = 6500 := 
by 
  sorry

end NUMINAMATH_GPT_salary_for_may_l1196_119684


namespace NUMINAMATH_GPT_sum_of_decimals_l1196_119697

theorem sum_of_decimals : 5.47 + 2.58 + 1.95 = 10.00 := by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_l1196_119697


namespace NUMINAMATH_GPT_triangle_cosine_l1196_119604

theorem triangle_cosine {A : ℝ} (h : 0 < A ∧ A < π / 2) (tan_A : Real.tan A = -2) :
  Real.cos A = - (Real.sqrt 5) / 5 :=
sorry

end NUMINAMATH_GPT_triangle_cosine_l1196_119604


namespace NUMINAMATH_GPT_total_animals_after_addition_l1196_119644

def current_cows := 2
def current_pigs := 3
def current_goats := 6

def added_cows := 3
def added_pigs := 5
def added_goats := 2

def total_current_animals := current_cows + current_pigs + current_goats
def total_added_animals := added_cows + added_pigs + added_goats
def total_animals := total_current_animals + total_added_animals

theorem total_animals_after_addition : total_animals = 21 := by
  sorry

end NUMINAMATH_GPT_total_animals_after_addition_l1196_119644


namespace NUMINAMATH_GPT_solve_quadratic_equation_l1196_119614

theorem solve_quadratic_equation : 
  ∃ (a b c : ℤ), (0 < a) ∧ (64 * x^2 + 48 * x - 36 = 0) ∧ ((a * x + b)^2 = c) ∧ (a + b + c = 56) := 
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l1196_119614


namespace NUMINAMATH_GPT_find_a_b_find_k_range_l1196_119623

-- Define the conditions for part 1
def quad_inequality (a x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def solution_set (x b : ℝ) : Prop :=
  x < 1 ∨ x > b

theorem find_a_b (a b : ℝ) :
  (∀ x, quad_inequality a x ↔ solution_set x b) → (a = 1 ∧ b = 2) :=
sorry

-- Define the conditions for part 2
def valid_x_y (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def equation1 (a b x y : ℝ) : Prop :=
  a / x + b / y = 1

def inequality1 (x y k : ℝ) : Prop :=
  2 * x + y ≥ k^2 + k + 2

theorem find_k_range (a b : ℝ) (x y k : ℝ) :
  a = 1 → b = 2 → valid_x_y x y → equation1 a b x y → inequality1 x y k →
  (-3 ≤ k ∧ k ≤ 2) :=
sorry

end NUMINAMATH_GPT_find_a_b_find_k_range_l1196_119623


namespace NUMINAMATH_GPT_difference_of_squares_example_l1196_119636

theorem difference_of_squares_example :
  262^2 - 258^2 = 2080 := by
sorry

end NUMINAMATH_GPT_difference_of_squares_example_l1196_119636


namespace NUMINAMATH_GPT_correct_operation_l1196_119664

variable (a b : ℝ)

theorem correct_operation : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end NUMINAMATH_GPT_correct_operation_l1196_119664


namespace NUMINAMATH_GPT_sally_cards_l1196_119659

theorem sally_cards (initial_cards dan_cards bought_cards : ℕ) (h1 : initial_cards = 27) (h2 : dan_cards = 41) (h3 : bought_cards = 20) :
  initial_cards + dan_cards + bought_cards = 88 := by
  sorry

end NUMINAMATH_GPT_sally_cards_l1196_119659


namespace NUMINAMATH_GPT_max_xy_max_xy_value_l1196_119683

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + 3 * y = 12) : x * y ≤ 3 :=
sorry

theorem max_xy_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + 3 * y = 12) : x * y = 3 → x = 3 / 2 ∧ y = 2 :=
sorry

end NUMINAMATH_GPT_max_xy_max_xy_value_l1196_119683


namespace NUMINAMATH_GPT_value_of_a_l1196_119694

-- Definitions based on conditions
def A (a : ℝ) : Set ℝ := {1, 2, a}
def B : Set ℝ := {1, 7}

-- Theorem statement
theorem value_of_a (a : ℝ) (h : B ⊆ A a) : a = 7 :=
sorry

end NUMINAMATH_GPT_value_of_a_l1196_119694


namespace NUMINAMATH_GPT_rate_of_second_batch_l1196_119678

-- Define the problem statement
theorem rate_of_second_batch
  (rate_first : ℝ)
  (weight_first weight_second weight_total : ℝ)
  (rate_mixture : ℝ)
  (profit_multiplier : ℝ) 
  (total_selling_price : ℝ) :
  rate_first = 11.5 →
  weight_first = 30 →
  weight_second = 20 →
  weight_total = weight_first + weight_second →
  rate_mixture = 15.12 →
  profit_multiplier = 1.20 →
  total_selling_price = weight_total * rate_mixture →
  (rate_first * weight_first + (weight_second * x) * profit_multiplier = total_selling_price) →
  x = 14.25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rate_of_second_batch_l1196_119678


namespace NUMINAMATH_GPT_new_mixture_alcohol_percentage_l1196_119673

/-- 
Given: 
  - a solution with 15 liters containing 26% alcohol
  - 5 liters of water added to the solution
Prove:
  The percentage of alcohol in the new mixture is 19.5%
-/
theorem new_mixture_alcohol_percentage 
  (original_volume : ℝ) (original_percent_alcohol : ℝ) (added_water_volume : ℝ) :
  original_volume = 15 → 
  original_percent_alcohol = 26 →
  added_water_volume = 5 →
  (original_volume * (original_percent_alcohol / 100) / (original_volume + added_water_volume)) * 100 = 19.5 :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_new_mixture_alcohol_percentage_l1196_119673


namespace NUMINAMATH_GPT_min_a_plus_b_l1196_119695

theorem min_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : a + b >= 4 :=
sorry

end NUMINAMATH_GPT_min_a_plus_b_l1196_119695


namespace NUMINAMATH_GPT_octal_to_decimal_l1196_119681

theorem octal_to_decimal : (1 * 8^3 + 7 * 8^2 + 4 * 8^1 + 3 * 8^0) = 995 :=
by
  sorry

end NUMINAMATH_GPT_octal_to_decimal_l1196_119681


namespace NUMINAMATH_GPT_john_works_30_hours_per_week_l1196_119619

/-- Conditions --/
def hours_per_week_fiona : ℕ := 40
def hours_per_week_jeremy : ℕ := 25
def hourly_wage : ℕ := 20
def monthly_total_payment : ℕ := 7600
def weeks_in_month : ℕ := 4

/-- Derived Definitions --/
def monthly_hours_fiona_jeremy : ℕ :=
  (hours_per_week_fiona + hours_per_week_jeremy) * weeks_in_month

def monthly_payment_fiona_jeremy : ℕ :=
  hourly_wage * monthly_hours_fiona_jeremy

def monthly_payment_john : ℕ :=
  monthly_total_payment - monthly_payment_fiona_jeremy

def hours_per_month_john : ℕ :=
  monthly_payment_john / hourly_wage

def hours_per_week_john : ℕ :=
  hours_per_month_john / weeks_in_month

/-- Theorem stating that John works 30 hours per week --/
theorem john_works_30_hours_per_week :
  hours_per_week_john = 30 := by
  sorry

end NUMINAMATH_GPT_john_works_30_hours_per_week_l1196_119619


namespace NUMINAMATH_GPT_probability_of_neither_event_l1196_119642

theorem probability_of_neither_event (P_A P_B P_A_and_B : ℝ) (h1 : P_A = 0.25) (h2 : P_B = 0.40) (h3 : P_A_and_B = 0.15) : 
  1 - (P_A + P_B - P_A_and_B) = 0.50 :=
by
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_probability_of_neither_event_l1196_119642


namespace NUMINAMATH_GPT_find_functions_l1196_119640

variable (f : ℝ → ℝ)

def isFunctionPositiveReal := ∀ x : ℝ, x > 0 → f x > 0

axiom functional_eq (x y : ℝ) (hx : x > 0) (hy : y > 0) : f (x ^ y) = f x ^ f y

theorem find_functions (hf : isFunctionPositiveReal f) :
  (∀ x : ℝ, x > 0 → f x = 1) ∨ (∀ x : ℝ, x > 0 → f x = x) := sorry

end NUMINAMATH_GPT_find_functions_l1196_119640


namespace NUMINAMATH_GPT_power_sum_result_l1196_119606

theorem power_sum_result : (64 ^ (-1/3 : ℝ)) + (81 ^ (-1/4 : ℝ)) = (7 / 12 : ℝ) :=
by
  have h64 : (64 : ℝ) = 2 ^ 6 := by norm_num
  have h81 : (81 : ℝ) = 3 ^ 4 := by norm_num
  sorry

end NUMINAMATH_GPT_power_sum_result_l1196_119606


namespace NUMINAMATH_GPT_trigonometric_identity_l1196_119661

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -3) : 
  (Real.cos α - Real.sin α) / (Real.cos α + Real.sin α) = -2 :=
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1196_119661


namespace NUMINAMATH_GPT_pressure_relation_l1196_119699

-- Definitions from the problem statement
variables (Q Δu A k x P S ΔV V R T T₀ c_v n P₀ V₀ : ℝ)
noncomputable def first_law := Q = Δu + A
noncomputable def Δu_def := Δu = c_v * (T - T₀)
noncomputable def A_def := A = (k * x^2) / 2
noncomputable def spring_relation := k * x = P * S
noncomputable def volume_change := ΔV = S * x
noncomputable def volume_after_expansion := V = (n / (n - 1)) * (S * x)
noncomputable def ideal_gas_law := P * V = R * T
noncomputable def initial_state := P₀ * V₀ = R * T₀
noncomputable def expanded_state := P * (n * V₀) = R * T

-- Theorem to prove the final relation
theorem pressure_relation
  (h1: first_law Q Δu A)
  (h2: Δu_def Δu c_v T T₀)
  (h3: A_def A k x)
  (h4: spring_relation k x P S)
  (h5: volume_change ΔV S x)
  (h6: volume_after_expansion V S x n)
  (h7: ideal_gas_law P V R T)
  (h8: initial_state P₀ V₀ R T₀)
  (h9: expanded_state P R T n V₀)
  : P / P₀ = 1 / (n * (1 + ((n - 1) * R) / (2 * n * c_v))) :=
  sorry

end NUMINAMATH_GPT_pressure_relation_l1196_119699


namespace NUMINAMATH_GPT_negation_exists_implies_forall_l1196_119669

theorem negation_exists_implies_forall : 
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_exists_implies_forall_l1196_119669


namespace NUMINAMATH_GPT_common_chord_length_l1196_119663

theorem common_chord_length (r : ℝ) (h : r = 12) 
  (condition : ∀ (C₁ C₂ : Set (ℝ × ℝ)), 
      ((C₁ = {p : ℝ × ℝ | dist p (0, 0) = r}) ∧ 
       (C₂ = {p : ℝ × ℝ | dist p (12, 0) = r}) ∧
       (C₂ ∩ C₁ ≠ ∅))) : 
  ∃ chord_len : ℝ, chord_len = 12 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_common_chord_length_l1196_119663


namespace NUMINAMATH_GPT_no_unique_solution_for_c_l1196_119632

theorem no_unique_solution_for_c (k : ℕ) (hk : k = 9) (c : ℕ) :
  (∀ x y : ℕ, 9 * x + c * y = 30 → 3 * x + 4 * y = 12) → c = 12 :=
by
  sorry

end NUMINAMATH_GPT_no_unique_solution_for_c_l1196_119632


namespace NUMINAMATH_GPT_other_solution_of_quadratic_l1196_119633

theorem other_solution_of_quadratic (x : ℚ) (h1 : x = 3 / 8) 
  (h2 : 72 * x^2 + 37 = -95 * x + 12) : ∃ y : ℚ, y ≠ 3 / 8 ∧ 72 * y^2 + 95 * y + 25 = 0 ∧ y = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_other_solution_of_quadratic_l1196_119633


namespace NUMINAMATH_GPT_sixteen_a_four_plus_one_div_a_four_l1196_119665

theorem sixteen_a_four_plus_one_div_a_four (a : ℝ) (h : 2 * a - 1 / a = 3) :
  16 * a^4 + (1 / a^4) = 161 :=
sorry

end NUMINAMATH_GPT_sixteen_a_four_plus_one_div_a_four_l1196_119665


namespace NUMINAMATH_GPT_minimum_value_x_plus_four_over_x_minimum_value_occurs_at_x_eq_2_l1196_119672

theorem minimum_value_x_plus_four_over_x (x : ℝ) (h : x ≥ 2) : 
  x + 4 / x ≥ 4 :=
by sorry

theorem minimum_value_occurs_at_x_eq_2 : ∀ (x : ℝ), x ≥ 2 → (x + 4 / x = 4 ↔ x = 2) :=
by sorry

end NUMINAMATH_GPT_minimum_value_x_plus_four_over_x_minimum_value_occurs_at_x_eq_2_l1196_119672


namespace NUMINAMATH_GPT_ratio_of_boys_l1196_119677

theorem ratio_of_boys (p : ℝ) (h : p = (3 / 5) * (1 - p)) 
  : p = 3 / 8 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_l1196_119677


namespace NUMINAMATH_GPT_total_distance_is_correct_l1196_119682

def Jonathan_d : Real := 7.5

def Mercedes_d (J : Real) : Real := 2 * J

def Davonte_d (M : Real) : Real := M + 2

theorem total_distance_is_correct : 
  let J := Jonathan_d
  let M := Mercedes_d J
  let D := Davonte_d M
  M + D = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_is_correct_l1196_119682


namespace NUMINAMATH_GPT_no_such_set_exists_l1196_119603

theorem no_such_set_exists :
  ¬ ∃ (A : Finset ℕ), A.card = 11 ∧
  (∀ (s : Finset ℕ), s ⊆ A → s.card = 6 → ¬ 6 ∣ s.sum id) :=
sorry

end NUMINAMATH_GPT_no_such_set_exists_l1196_119603


namespace NUMINAMATH_GPT_pyramid_total_surface_area_l1196_119667

theorem pyramid_total_surface_area :
  ∀ (s h : ℝ), s = 8 → h = 10 →
  6 * (1/2 * s * (Real.sqrt (h^2 - (s/2)^2))) = 48 * Real.sqrt 21 :=
by
  intros s h s_eq h_eq
  rw [s_eq, h_eq]
  sorry

end NUMINAMATH_GPT_pyramid_total_surface_area_l1196_119667


namespace NUMINAMATH_GPT_expansion_coefficients_sum_l1196_119641

theorem expansion_coefficients_sum : 
  ∀ (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), 
    (x - 2)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 → 
    a_0 + a_2 + a_4 = -122 := 
by 
  intros x a_0 a_1 a_2 a_3 a_4 a_5 h_eq
  sorry

end NUMINAMATH_GPT_expansion_coefficients_sum_l1196_119641


namespace NUMINAMATH_GPT_jasmine_stops_at_S_l1196_119670

-- Definitions of the given conditions
def circumference : ℕ := 60
def total_distance : ℕ := 5400
def quadrants : ℕ := 4
def laps (distance circumference : ℕ) := distance / circumference
def isMultiple (a b : ℕ) := b ∣ a
def onSamePoint (distance circumference : ℕ) := (distance % circumference) = 0

-- The theorem to be proved: Jasmine stops at point S after running the total distance
theorem jasmine_stops_at_S 
  (circumference : ℕ) (total_distance : ℕ) (quadrants : ℕ)
  (h1 : circumference = 60) 
  (h2 : total_distance = 5400)
  (h3 : quadrants = 4)
  (h4 : laps total_distance circumference = 90)
  (h5 : isMultiple total_distance circumference)
  : onSamePoint total_distance circumference := 
  sorry

end NUMINAMATH_GPT_jasmine_stops_at_S_l1196_119670


namespace NUMINAMATH_GPT_rectangle_area_and_perimeter_l1196_119616

-- Given conditions as definitions
def length : ℕ := 5
def width : ℕ := 3

-- Proof problems
theorem rectangle_area_and_perimeter :
  (length * width = 15) ∧ (2 * (length + width) = 16) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_and_perimeter_l1196_119616


namespace NUMINAMATH_GPT_parabola_cubic_intersection_points_l1196_119680

def parabola (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 15

def cubic (x : ℝ) : ℝ := x^3 - 6 * x^2 + 11 * x - 6

theorem parabola_cubic_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    p1 = (-1, 0) ∧ p2 = (1, -24) ∧ p3 = (9, 162) ∧
    parabola p1.1 = p1.2 ∧ cubic p1.1 = p1.2 ∧
    parabola p2.1 = p2.2 ∧ cubic p2.1 = p2.2 ∧
    parabola p3.1 = p3.2 ∧ cubic p3.1 = p3.2 :=
by {
  -- This is the statement
  sorry
}

end NUMINAMATH_GPT_parabola_cubic_intersection_points_l1196_119680


namespace NUMINAMATH_GPT_circle_equation_l1196_119649

open Real

variable {x y : ℝ}

theorem circle_equation (a : ℝ) (h_a_positive : a > 0) 
    (h_tangent : abs (3 * a + 4) / sqrt (3^2 + 4^2) = 2) :
    (∀ x y : ℝ, (x - a)^2 + y^2 = 4) := sorry

end NUMINAMATH_GPT_circle_equation_l1196_119649


namespace NUMINAMATH_GPT_product_zero_probability_l1196_119643

noncomputable def probability_product_is_zero : ℚ :=
  let S := [-3, -1, 0, 0, 2, 5]
  let total_ways := 15 -- Calculated as 6 choose 2 taking into account repetition
  let favorable_ways := 8 -- Calculated as (2 choose 1) * (4 choose 1)
  favorable_ways / total_ways

theorem product_zero_probability : probability_product_is_zero = 8 / 15 := by
  sorry

end NUMINAMATH_GPT_product_zero_probability_l1196_119643


namespace NUMINAMATH_GPT_cyclist_speed_l1196_119653

variable (circumference : ℝ) (v₂ : ℝ) (t : ℝ)

theorem cyclist_speed (h₀ : circumference = 180) (h₁ : v₂ = 8) (h₂ : t = 12)
  (h₃ : (7 * t + v₂ * t) = circumference) : 7 = 7 :=
by
  -- From given conditions, we derived that v₁ should be 7
  sorry

end NUMINAMATH_GPT_cyclist_speed_l1196_119653


namespace NUMINAMATH_GPT_positive_root_gt_1008_l1196_119609

noncomputable def P (x : ℝ) : ℝ := sorry
-- where P is a non-constant polynomial with integer coefficients bounded by 2015 in absolute value
-- Assume it has been properly defined according to the conditions in the problem statement

theorem positive_root_gt_1008 (x : ℝ) (hx : 0 < x) (hroot : P x = 0) : x > 1008 := 
sorry

end NUMINAMATH_GPT_positive_root_gt_1008_l1196_119609


namespace NUMINAMATH_GPT_cost_formula_correct_l1196_119638

def total_cost (P : ℕ) : ℕ :=
  if P ≤ 2 then 15 else 15 + 5 * (P - 2)

theorem cost_formula_correct (P : ℕ) : 
  total_cost P = (if P ≤ 2 then 15 else 15 + 5 * (P - 2)) :=
by 
  exact rfl

end NUMINAMATH_GPT_cost_formula_correct_l1196_119638


namespace NUMINAMATH_GPT_score_entered_twice_l1196_119675

theorem score_entered_twice (scores : List ℕ) (h : scores = [68, 74, 77, 82, 85, 90]) :
  ∃ (s : ℕ), s = 82 ∧ ∀ (entered : List ℕ), entered.length = 7 ∧ (∀ i, (List.take (i + 1) entered).sum % (i + 1) = 0) →
  (List.count (List.insertNth i 82 scores)) = 2 ∧ (∀ x, x ∈ scores.remove 82 → x ≠ s) :=
by
  sorry

end NUMINAMATH_GPT_score_entered_twice_l1196_119675


namespace NUMINAMATH_GPT_sum_of_fractions_irreducible_l1196_119628

noncomputable def is_irreducible (num denom : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ num ∧ d ∣ denom → d = 1

theorem sum_of_fractions_irreducible (a b : ℕ) (h_coprime : Nat.gcd a b = 1) :
  is_irreducible (2 * a + b) (a * (a + b)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_irreducible_l1196_119628


namespace NUMINAMATH_GPT_silk_original_amount_l1196_119626

theorem silk_original_amount (s r : ℕ) (l d x : ℚ)
  (h1 : s = 30)
  (h2 : r = 3)
  (h3 : d = 12)
  (h4 : 30 - 3 = 27)
  (h5 : x / 12 = 30 / 27):
  x = 40 / 3 :=
by
  sorry

end NUMINAMATH_GPT_silk_original_amount_l1196_119626


namespace NUMINAMATH_GPT_smallest_x_absolute_value_l1196_119657

theorem smallest_x_absolute_value :
  ∃ x : ℝ, (|5 * x + 15| = 40) ∧ (∀ y : ℝ, |5 * y + 15| = 40 → x ≤ y) ∧ x = -11 :=
sorry

end NUMINAMATH_GPT_smallest_x_absolute_value_l1196_119657


namespace NUMINAMATH_GPT_sum_of_angles_l1196_119621

theorem sum_of_angles (α β : ℝ) (hα: 0 < α ∧ α < π) (hβ: 0 < β ∧ β < π) (h_tan_α: Real.tan α = 1 / 2) (h_tan_β: Real.tan β = 1 / 3) : α + β = π / 4 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_angles_l1196_119621


namespace NUMINAMATH_GPT_evaluate_fraction_l1196_119690

theorem evaluate_fraction : (8 / 29) - (5 / 87) = (19 / 87) := sorry

end NUMINAMATH_GPT_evaluate_fraction_l1196_119690


namespace NUMINAMATH_GPT_find_principal_amount_l1196_119645

theorem find_principal_amount 
  (total_interest : ℝ)
  (rate1 rate2 : ℝ)
  (years1 years2 : ℕ)
  (P : ℝ)
  (A1 A2 : ℝ) 
  (hA1 : A1 = P * (1 + rate1/100)^years1)
  (hA2 : A2 = A1 * (1 + rate2/100)^years2)
  (hInterest : A2 = P + total_interest) : 
  P = 25252.57 :=
by
  -- Given the conditions above, we prove the main statement.
  sorry

end NUMINAMATH_GPT_find_principal_amount_l1196_119645


namespace NUMINAMATH_GPT_solve_for_x_l1196_119625

theorem solve_for_x :
  (∀ x : ℝ, (1 / Real.log x / Real.log 3 + 1 / Real.log x / Real.log 4 + 1 / Real.log x / Real.log 5 = 2))
  → x = 2 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1196_119625


namespace NUMINAMATH_GPT_linda_total_distance_l1196_119671

theorem linda_total_distance :
  ∃ x: ℕ, 
    (x > 0) ∧ (60 % x = 0) ∧
    ((x + 5) > 0) ∧ (60 % (x + 5) = 0) ∧
    ((x + 10) > 0) ∧ (60 % (x + 10) = 0) ∧
    ((x + 15) > 0) ∧ (60 % (x + 15) = 0) ∧
    (60 / x + 60 / (x + 5) + 60 / (x + 10) + 60 / (x + 15) = 25) :=
by
  sorry

end NUMINAMATH_GPT_linda_total_distance_l1196_119671


namespace NUMINAMATH_GPT_train_speed_proof_l1196_119689

variables (distance_to_syracuse total_time_hours return_trip_speed average_speed_to_syracuse : ℝ)

def question_statement : Prop :=
  distance_to_syracuse = 120 ∧
  total_time_hours = 5.5 ∧
  return_trip_speed = 38.71 →
  average_speed_to_syracuse = 50

theorem train_speed_proof :
  question_statement distance_to_syracuse total_time_hours return_trip_speed average_speed_to_syracuse :=
by
  -- sorry is used to indicate that the proof is omitted
  sorry

end NUMINAMATH_GPT_train_speed_proof_l1196_119689


namespace NUMINAMATH_GPT_quotient_in_first_division_l1196_119647

theorem quotient_in_first_division (N Q Q' : ℕ) (h₁ : N = 68 * Q) (h₂ : N % 67 = 1) : Q = 1 :=
by
  -- rest of the proof goes here
  sorry

end NUMINAMATH_GPT_quotient_in_first_division_l1196_119647


namespace NUMINAMATH_GPT_hat_value_in_rice_l1196_119693

variables (f l r h : ℚ)

theorem hat_value_in_rice :
  (4 * f = 3 * l) →
  (l = 5 * r) →
  (5 * f = 7 * h) →
  h = (75 / 28) * r :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end NUMINAMATH_GPT_hat_value_in_rice_l1196_119693


namespace NUMINAMATH_GPT_gcd_gx_x_l1196_119620

theorem gcd_gx_x (x : ℕ) (h : 2520 ∣ x) : 
  Nat.gcd ((4*x + 5) * (5*x + 2) * (11*x + 8) * (3*x + 7)) x = 280 := 
sorry

end NUMINAMATH_GPT_gcd_gx_x_l1196_119620


namespace NUMINAMATH_GPT_find_function_satisfying_property_l1196_119688

noncomputable def example_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + y^2) = f (x^2 - y^2) + f (2 * x * y)

theorem find_function_satisfying_property (f : ℝ → ℝ) (h : ∀ x, 0 ≤ f x) (hf : example_function f) :
  ∃ a : ℝ, 0 ≤ a ∧ ∀ x : ℝ, f x = a * x^2 :=
sorry

end NUMINAMATH_GPT_find_function_satisfying_property_l1196_119688


namespace NUMINAMATH_GPT_rectangle_area_l1196_119656

theorem rectangle_area
  (s : ℝ)
  (h_square_area : s^2 = 49)
  (rect_width : ℝ := s)
  (rect_length : ℝ := 3 * rect_width)
  (h_rect_width_eq_s : rect_width = s)
  (h_rect_length_eq_3w : rect_length = 3 * rect_width) :
  rect_width * rect_length = 147 :=
by 
  skip
  sorry

end NUMINAMATH_GPT_rectangle_area_l1196_119656


namespace NUMINAMATH_GPT_cost_of_fencing_per_meter_l1196_119615

theorem cost_of_fencing_per_meter
  (breadth : ℝ)
  (length : ℝ)
  (cost : ℝ)
  (length_eq : length = breadth + 40)
  (total_cost : cost = 5300)
  (length_given : length = 70) :
  cost / (2 * length + 2 * breadth) = 26.5 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_per_meter_l1196_119615


namespace NUMINAMATH_GPT_stop_signs_per_mile_l1196_119662

-- Define the conditions
def miles_traveled := 5 + 2
def stop_signs_encountered := 17 - 3

-- Define the proof statement
theorem stop_signs_per_mile : (stop_signs_encountered / miles_traveled) = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_stop_signs_per_mile_l1196_119662


namespace NUMINAMATH_GPT_loan_period_l1196_119648

theorem loan_period (principal : ℝ) (rate_A rate_C gain_B : ℝ) (n : ℕ) 
  (h1 : principal = 3150)
  (h2 : rate_A = 0.08)
  (h3 : rate_C = 0.125)
  (h4 : gain_B = 283.5) :
  (gain_B = (rate_C * principal - rate_A * principal) * n) → n = 2 := by
  sorry

end NUMINAMATH_GPT_loan_period_l1196_119648


namespace NUMINAMATH_GPT_xiaohui_pe_score_l1196_119611

-- Define the conditions
def morning_score : ℝ := 95
def midterm_score : ℝ := 90
def final_score : ℝ := 85

def morning_weight : ℝ := 0.2
def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.5

-- The problem is to prove that Xiaohui's physical education score for the semester is 88.5 points.
theorem xiaohui_pe_score :
  morning_score * morning_weight +
  midterm_score * midterm_weight +
  final_score * final_weight = 88.5 :=
by
  sorry

end NUMINAMATH_GPT_xiaohui_pe_score_l1196_119611


namespace NUMINAMATH_GPT_gas_mixture_pressure_l1196_119634

theorem gas_mixture_pressure
  (m : ℝ) -- mass of each gas
  (p : ℝ) -- initial pressure
  (T : ℝ) -- initial temperature
  (V : ℝ) -- volume of the container
  (R : ℝ) -- ideal gas constant
  (mu_He : ℝ := 4) -- molar mass of helium
  (mu_N2 : ℝ := 28) -- molar mass of nitrogen
  (is_ideal : True) -- assumption that the gases are ideal
  (temp_doubled : True) -- assumption that absolute temperature is doubled
  (N2_dissociates : True) -- assumption that nitrogen dissociates into atoms
  : (9 / 4) * p = p' :=
by
  sorry

end NUMINAMATH_GPT_gas_mixture_pressure_l1196_119634


namespace NUMINAMATH_GPT_ratio_of_new_r_to_original_r_l1196_119635

theorem ratio_of_new_r_to_original_r
  (r₁ r₂ : ℝ)
  (a₁ a₂ : ℝ)
  (h₁ : a₁ = (2 * r₁)^3)
  (h₂ : a₂ = (2 * r₂)^3)
  (h : a₂ = 0.125 * a₁) :
  r₂ / r₁ = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_new_r_to_original_r_l1196_119635


namespace NUMINAMATH_GPT_arc_length_of_circle_l1196_119660

theorem arc_length_of_circle (r : ℝ) (θ_peripheral : ℝ) (h_r : r = 5) (h_θ : θ_peripheral = 2/3 * π) :
  r * (2/3 * θ_peripheral) = 20 * π / 3 := 
by sorry

end NUMINAMATH_GPT_arc_length_of_circle_l1196_119660


namespace NUMINAMATH_GPT_luncheon_cost_l1196_119613

theorem luncheon_cost (s c p : ℝ)
  (h1 : 2 * s + 5 * c + 2 * p = 3.50)
  (h2 : 3 * s + 7 * c + 2 * p = 4.90) :
  s + c + p = 1.00 :=
  sorry

end NUMINAMATH_GPT_luncheon_cost_l1196_119613


namespace NUMINAMATH_GPT_grandmother_ratio_l1196_119622

noncomputable def Grace_Age := 60
noncomputable def Mother_Age := 80

theorem grandmother_ratio :
  ∃ GM, Grace_Age = (3 / 8 : Rat) * GM ∧ GM / Mother_Age = 2 :=
by
  sorry

end NUMINAMATH_GPT_grandmother_ratio_l1196_119622


namespace NUMINAMATH_GPT_min_value_inequality_l1196_119658

theorem min_value_inequality (y1 y2 y3 : ℝ) (h_pos : 0 < y1 ∧ 0 < y2 ∧ 0 < y3) (h_sum : 2 * y1 + 3 * y2 + 4 * y3 = 120) :
  y1^2 + 4 * y2^2 + 9 * y3^2 ≥ 14400 / 29 :=
sorry

end NUMINAMATH_GPT_min_value_inequality_l1196_119658
