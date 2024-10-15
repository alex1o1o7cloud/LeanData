import Mathlib

namespace NUMINAMATH_GPT_craig_age_l2252_225217

theorem craig_age (C M : ℕ) (h1 : C = M - 24) (h2 : C + M = 56) : C = 16 := 
by
  sorry

end NUMINAMATH_GPT_craig_age_l2252_225217


namespace NUMINAMATH_GPT_freeRangingChickens_l2252_225265

-- Define the number of chickens in the coop
def chickensInCoop : Nat := 14

-- Define the number of chickens in the run
def chickensInRun : Nat := 2 * chickensInCoop

-- Define the number of chickens free ranging
def chickensFreeRanging : Nat := 2 * chickensInRun - 4

-- State the theorem
theorem freeRangingChickens : chickensFreeRanging = 52 := by
  -- We cannot provide the proof, so we use sorry
  sorry

end NUMINAMATH_GPT_freeRangingChickens_l2252_225265


namespace NUMINAMATH_GPT_terminating_decimal_l2252_225252

-- Define the given fraction
def frac : ℚ := 21 / 160

-- Define the decimal representation
def dec : ℚ := 13125 / 100000

-- State the theorem to be proved
theorem terminating_decimal : frac = dec := by
  sorry

end NUMINAMATH_GPT_terminating_decimal_l2252_225252


namespace NUMINAMATH_GPT_find_b_l2252_225290

theorem find_b 
  (b : ℝ)
  (h_pos : 0 < b)
  (h_geom_sequence : ∃ r : ℝ, 10 * r = b ∧ b * r = 2 / 3) :
  b = 2 * Real.sqrt 15 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l2252_225290


namespace NUMINAMATH_GPT_determine_a_l2252_225251

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.exp x - a)^2

theorem determine_a (a x₀ : ℝ)
  (h₀ : f x₀ a ≤ 1/2) : a = 1/2 :=
sorry

end NUMINAMATH_GPT_determine_a_l2252_225251


namespace NUMINAMATH_GPT_Linda_total_distance_is_25_l2252_225223

theorem Linda_total_distance_is_25 : 
  ∃ (x : ℤ), x > 0 ∧ 
  (60/x + 60/(x+5) + 60/(x+10) + 60/(x+15) = 25) :=
by 
  sorry

end NUMINAMATH_GPT_Linda_total_distance_is_25_l2252_225223


namespace NUMINAMATH_GPT_pies_baked_l2252_225294

/-- Mrs. Hilt baked 16.0 pecan pies and 14.0 apple pies. She needs 5.0 times this amount.
    Prove that the total number of pies she has to bake is 150.0. -/
theorem pies_baked (pecan_pies : ℝ) (apple_pies : ℝ) (times : ℝ)
  (h1 : pecan_pies = 16.0) (h2 : apple_pies = 14.0) (h3 : times = 5.0) :
  times * (pecan_pies + apple_pies) = 150.0 := by
  sorry

end NUMINAMATH_GPT_pies_baked_l2252_225294


namespace NUMINAMATH_GPT_average_age_increase_l2252_225200

theorem average_age_increase
  (n : ℕ)
  (A : ℝ)
  (w : ℝ)
  (h1 : (n + 1) * (A + w) = n * A + 39)
  (h2 : (n + 1) * (A - 1) = n * A + 15)
  (hw : w = 7) :
  w = 7 := 
by
  sorry

end NUMINAMATH_GPT_average_age_increase_l2252_225200


namespace NUMINAMATH_GPT_alice_min_speed_l2252_225213

open Real

theorem alice_min_speed (d : ℝ) (bob_speed : ℝ) (alice_delay : ℝ) (alice_time : ℝ) :
  d = 180 → bob_speed = 40 → alice_delay = 0.5 → alice_time = 4 → d / alice_time > (d / bob_speed) - alice_delay →
  d / alice_time > 45 := by
  sorry


end NUMINAMATH_GPT_alice_min_speed_l2252_225213


namespace NUMINAMATH_GPT_negation_of_proposition_l2252_225275

theorem negation_of_proposition (p : ∀ x : ℝ, -x^2 + 4 * x + 3 > 0) :
  (∃ x : ℝ, -x^2 + 4 * x + 3 ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l2252_225275


namespace NUMINAMATH_GPT_parity_of_solutions_l2252_225238

theorem parity_of_solutions
  (n m x y : ℤ)
  (hn : Odd n) 
  (hm : Odd m) 
  (h1 : x + 2 * y = n) 
  (h2 : 3 * x - y = m) :
  Odd x ∧ Even y :=
by
  sorry

end NUMINAMATH_GPT_parity_of_solutions_l2252_225238


namespace NUMINAMATH_GPT_sebastian_age_correct_l2252_225219

-- Define the ages involved
def sebastian_age_now := 40
def sister_age_now (S : ℕ) := S - 10
def father_age_now := 85

-- Define the conditions
def age_difference_condition (S : ℕ) := (sister_age_now S) = S - 10
def father_age_condition := father_age_now = 85
def past_age_sum_condition (S : ℕ) := (S - 5) + (sister_age_now S - 5) = 3 / 4 * (father_age_now - 5)

theorem sebastian_age_correct (S : ℕ) 
  (h1 : age_difference_condition S) 
  (h2 : father_age_condition) 
  (h3 : past_age_sum_condition S) : 
  S = sebastian_age_now := 
  by sorry

end NUMINAMATH_GPT_sebastian_age_correct_l2252_225219


namespace NUMINAMATH_GPT_Vanya_original_number_l2252_225287

theorem Vanya_original_number (m n : ℕ) (hm : m ≤ 9) (hn : n ≤ 9) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 10 * m + n = 81 := by
  sorry

end NUMINAMATH_GPT_Vanya_original_number_l2252_225287


namespace NUMINAMATH_GPT_least_non_lucky_multiple_of_11_l2252_225204

/--
A lucky integer is a positive integer which is divisible by the sum of its digits.
Example:
- 18 is a lucky integer because 1 + 8 = 9 and 18 is divisible by 9.
- 20 is not a lucky integer because 2 + 0 = 2 and 20 is not divisible by 2.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_non_lucky_multiple_of_11 : ∃ n, n > 0 ∧ n % 11 = 0 ∧ ¬ is_lucky n ∧ ∀ m, m > 0 → m % 11 = 0 → ¬ is_lucky m → n ≤ m := 
by
  sorry

end NUMINAMATH_GPT_least_non_lucky_multiple_of_11_l2252_225204


namespace NUMINAMATH_GPT_mark_charged_more_hours_than_kate_l2252_225299

variables (K P M : ℝ)
variables (h1 : K + P + M = 198) (h2 : P = 2 * K) (h3 : M = 3 * P)

theorem mark_charged_more_hours_than_kate : M - K = 110 :=
by
  sorry

end NUMINAMATH_GPT_mark_charged_more_hours_than_kate_l2252_225299


namespace NUMINAMATH_GPT_cup_of_coffee_price_l2252_225242

def price_cheesecake : ℝ := 10
def price_set : ℝ := 12
def discount : ℝ := 0.75

theorem cup_of_coffee_price (C : ℝ) (h : price_set = discount * (C + price_cheesecake)) : C = 6 :=
by
  sorry

end NUMINAMATH_GPT_cup_of_coffee_price_l2252_225242


namespace NUMINAMATH_GPT_total_employees_l2252_225214

variable (E : ℕ)
variable (employees_prefer_X employees_prefer_Y number_of_prefers : ℕ)
variable (X_percentage Y_percentage : ℝ)

-- Conditions based on the problem
axiom prefer_X : X_percentage = 0.60
axiom prefer_Y : Y_percentage = 0.40
axiom max_preference_relocation : number_of_prefers = 140

-- Defining the total number of employees who prefer city X or Y and get relocated accordingly:
axiom equation : X_percentage * E + Y_percentage * E = number_of_prefers

-- The theorem we are proving
theorem total_employees : E = 140 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_total_employees_l2252_225214


namespace NUMINAMATH_GPT_new_length_maintains_area_l2252_225295

noncomputable def new_length_for_doubled_width (A W : ℝ) : ℝ := A / (2 * W)

theorem new_length_maintains_area (A W : ℝ) (hA : A = 35.7) (hW : W = 3.8) :
  new_length_for_doubled_width A W = 4.69736842 :=
by
  rw [new_length_for_doubled_width, hA, hW]
  norm_num
  sorry

end NUMINAMATH_GPT_new_length_maintains_area_l2252_225295


namespace NUMINAMATH_GPT_sales_in_fourth_month_l2252_225234

theorem sales_in_fourth_month (sale_m1 sale_m2 sale_m3 sale_m5 sale_m6 avg_sales total_months : ℕ)
    (H1 : sale_m1 = 7435) (H2 : sale_m2 = 7927) (H3 : sale_m3 = 7855) 
    (H4 : sale_m5 = 7562) (H5 : sale_m6 = 5991) (H6 : avg_sales = 7500) (H7 : total_months = 6) :
    ∃ sale_m4 : ℕ, sale_m4 = 8230 := by
  sorry

end NUMINAMATH_GPT_sales_in_fourth_month_l2252_225234


namespace NUMINAMATH_GPT_jane_reading_period_l2252_225257

theorem jane_reading_period (total_pages pages_per_day : ℕ) (H1 : pages_per_day = 5 + 10) (H2 : total_pages = 105) : 
  total_pages / pages_per_day = 7 :=
by
  sorry

end NUMINAMATH_GPT_jane_reading_period_l2252_225257


namespace NUMINAMATH_GPT_valid_numbers_count_l2252_225249

def count_valid_numbers : ℕ :=
  sorry

theorem valid_numbers_count :
  count_valid_numbers = 7 :=
sorry

end NUMINAMATH_GPT_valid_numbers_count_l2252_225249


namespace NUMINAMATH_GPT_ratio_of_numbers_l2252_225263

theorem ratio_of_numbers (a b : ℕ) (h1 : a.gcd b = 5) (h2 : a.lcm b = 60) (h3 : a = 3 * 5) (h4 : b = 4 * 5) : (a / a.gcd b) / (b / a.gcd b) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l2252_225263


namespace NUMINAMATH_GPT_penguins_count_l2252_225298

theorem penguins_count (fish_total penguins_fed penguins_require : ℕ) (h1 : fish_total = 68) (h2 : penguins_fed = 19) (h3 : penguins_require = 17) : penguins_fed + penguins_require = 36 :=
by
  sorry

end NUMINAMATH_GPT_penguins_count_l2252_225298


namespace NUMINAMATH_GPT_other_function_value_at_20_l2252_225289

def linear_function (k b : ℝ) (x : ℝ) : ℝ :=
  k * x + b

theorem other_function_value_at_20
    (k1 k2 b1 b2 : ℝ)
    (h_intersect : linear_function k1 b1 2 = linear_function k2 b2 2)
    (h_diff_at_8 : abs (linear_function k1 b1 8 - linear_function k2 b2 8) = 8)
    (h_y1_at_20 : linear_function k1 b1 20 = 100) :
  linear_function k2 b2 20 = 76 ∨ linear_function k2 b2 20 = 124 :=
sorry

end NUMINAMATH_GPT_other_function_value_at_20_l2252_225289


namespace NUMINAMATH_GPT_chord_length_l2252_225270

theorem chord_length (r d: ℝ) (h1: r = 5) (h2: d = 4) : ∃ EF, EF = 6 := by
  sorry

end NUMINAMATH_GPT_chord_length_l2252_225270


namespace NUMINAMATH_GPT_find_equation_of_line_l2252_225202

theorem find_equation_of_line
  (midpoint : ℝ × ℝ)
  (ellipse : ℝ → ℝ → Prop)
  (l_eq : ℝ → ℝ → Prop)
  (H_mid : midpoint = (1, 2))
  (H_ellipse : ∀ (x y : ℝ), ellipse x y ↔ x^2 / 64 + y^2 / 16 = 1)
  (H_line : ∀ (x y : ℝ), l_eq x y ↔ y - 2 = - (1/8) * (x - 1))
  : ∃ (a b c : ℝ), (a, b, c) = (1, 8, -17) ∧ (∀ (x y : ℝ), l_eq x y ↔ a * x + b * y + c = 0) :=
by 
  sorry

end NUMINAMATH_GPT_find_equation_of_line_l2252_225202


namespace NUMINAMATH_GPT_how_many_peaches_l2252_225264

-- Define the variables
variables (Jake Steven : ℕ)

-- Conditions
def has_fewer_peaches : Prop := Jake = Steven - 7
def jake_has_9_peaches : Prop := Jake = 9

-- The theorem that proves Steven's number of peaches
theorem how_many_peaches (Jake Steven : ℕ) (h1 : has_fewer_peaches Jake Steven) (h2 : jake_has_9_peaches Jake) : Steven = 16 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_how_many_peaches_l2252_225264


namespace NUMINAMATH_GPT_transformed_eq_l2252_225237

theorem transformed_eq (a b c : ℤ) (h : a > 0) :
  (∀ x : ℝ, 16 * x^2 + 32 * x - 40 = 0 → (a * x + b)^2 = c) →
  a + b + c = 64 :=
by
  sorry

end NUMINAMATH_GPT_transformed_eq_l2252_225237


namespace NUMINAMATH_GPT_fraction_is_one_fourth_l2252_225284

theorem fraction_is_one_fourth (f N : ℝ) 
  (h1 : (1/3) * f * N = 15) 
  (h2 : (3/10) * N = 54) : 
  f = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_one_fourth_l2252_225284


namespace NUMINAMATH_GPT_rationalize_and_subtract_l2252_225224

theorem rationalize_and_subtract :
  (7 / (3 + Real.sqrt 15)) * (3 - Real.sqrt 15) / (3^2 - (Real.sqrt 15)^2) 
  - (1 / 2) = -4 + (7 * Real.sqrt 15) / 6 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_and_subtract_l2252_225224


namespace NUMINAMATH_GPT_part1_part2_l2252_225231

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a

theorem part1 (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by
  sorry

theorem part2 (a x : ℝ) (h : a ≠ -3) :
  (f x a > 4 * a - (a + 3) * x) ↔ 
  ((a > -3 ∧ (x < -3 ∨ x > a)) ∨ (a < -3 ∧ (x < a ∨ x > -3))) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2252_225231


namespace NUMINAMATH_GPT_age_difference_l2252_225277

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 12) : A - C = 12 :=
sorry

end NUMINAMATH_GPT_age_difference_l2252_225277


namespace NUMINAMATH_GPT_number_of_5_dollar_bills_l2252_225285

def total_money : ℤ := 45
def value_of_each_bill : ℤ := 5

theorem number_of_5_dollar_bills : total_money / value_of_each_bill = 9 := by
  sorry

end NUMINAMATH_GPT_number_of_5_dollar_bills_l2252_225285


namespace NUMINAMATH_GPT_evaluate_composite_function_l2252_225226

def f (x : ℝ) : ℝ := x^2 - 2 * x + 2
def g (x : ℝ) : ℝ := 3 * x + 2

theorem evaluate_composite_function :
  f (g (-2)) = 26 := by
  sorry

end NUMINAMATH_GPT_evaluate_composite_function_l2252_225226


namespace NUMINAMATH_GPT_proportion_of_adopted_kittens_l2252_225291

-- Define the relevant objects and conditions in Lean
def breeding_rabbits : ℕ := 10
def kittens_first_spring := 10 * breeding_rabbits -- 100 kittens
def kittens_second_spring : ℕ := 60
def adopted_first_spring (P : ℝ) := 100 * P
def returned_first_spring : ℕ := 5
def adopted_second_spring : ℕ := 4
def total_rabbits_in_house (P : ℝ) :=
  breeding_rabbits + (kittens_first_spring - adopted_first_spring P + returned_first_spring) +
  (kittens_second_spring - adopted_second_spring)

theorem proportion_of_adopted_kittens : ∃ (P : ℝ), total_rabbits_in_house P = 121 ∧ P = 0.5 :=
by
  use 0.5
  -- Proof part (with "sorry" to skip the detailed proof)
  sorry

end NUMINAMATH_GPT_proportion_of_adopted_kittens_l2252_225291


namespace NUMINAMATH_GPT_avg_annual_growth_rate_optimal_room_price_l2252_225280

-- Problem 1: Average Annual Growth Rate
theorem avg_annual_growth_rate (visitors_2021 visitors_2023 : ℝ) (years : ℕ) (visitors_2021_pos : 0 < visitors_2021) :
  visitors_2023 > visitors_2021 → visitors_2023 / visitors_2021 = 2.25 → 
  ∃ x : ℝ, (1 + x)^2 = 2.25 ∧ x = 0.5 :=
by sorry

-- Problem 2: Optimal Room Price for Desired Profit
theorem optimal_room_price (rooms : ℕ) (base_price cost_per_room desired_profit : ℝ)
  (rooms_pos : 0 < rooms) :
  base_price = 180 → cost_per_room = 20 → desired_profit = 9450 → 
  ∃ y : ℝ, (y - cost_per_room) * (rooms - (y - base_price) / 10) = desired_profit ∧ y = 230 :=
by sorry

end NUMINAMATH_GPT_avg_annual_growth_rate_optimal_room_price_l2252_225280


namespace NUMINAMATH_GPT_total_pictures_on_wall_l2252_225210

theorem total_pictures_on_wall (oil_paintings watercolor_paintings : ℕ) (h1 : oil_paintings = 9) (h2 : watercolor_paintings = 7) :
  oil_paintings + watercolor_paintings = 16 := 
by
  sorry

end NUMINAMATH_GPT_total_pictures_on_wall_l2252_225210


namespace NUMINAMATH_GPT_system_solutions_are_equivalent_l2252_225221

theorem system_solutions_are_equivalent :
  ∀ (a b x y : ℝ),
  (2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9) ∧
  (a = 8.3 ∧ b = 1.2) ∧
  (x + 2 = a ∧ y - 1 = b) →
  x = 6.3 ∧ y = 2.2 :=
by
  -- Sorry is added intentionally to skip the proof
  sorry

end NUMINAMATH_GPT_system_solutions_are_equivalent_l2252_225221


namespace NUMINAMATH_GPT_M_inter_N_l2252_225293

def M : Set ℝ := { y | y > 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem M_inter_N : M ∩ N = { z | 1 < z ∧ z < 2 } :=
by 
  sorry

end NUMINAMATH_GPT_M_inter_N_l2252_225293


namespace NUMINAMATH_GPT_total_bees_including_queen_at_end_of_14_days_l2252_225239

-- Conditions definitions
def bees_hatched_per_day : ℕ := 5000
def bees_lost_per_day : ℕ := 1800
def duration_days : ℕ := 14
def initial_bees : ℕ := 20000
def queen_bees : ℕ := 1

-- Question statement as Lean theorem
theorem total_bees_including_queen_at_end_of_14_days :
  (initial_bees + (bees_hatched_per_day - bees_lost_per_day) * duration_days + queen_bees) = 64801 := 
by
  sorry

end NUMINAMATH_GPT_total_bees_including_queen_at_end_of_14_days_l2252_225239


namespace NUMINAMATH_GPT_triangle_acute_l2252_225253

theorem triangle_acute
  (A B C : ℝ)
  (h_sum : A + B + C = 180)
  (h_ratio : A / B = 2 / 3 ∧ B / C = 3 / 4) :
  A < 90 ∧ B < 90 ∧ C < 90 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_triangle_acute_l2252_225253


namespace NUMINAMATH_GPT_black_squares_in_20th_row_l2252_225233

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def squares_in_row (n : ℕ) : ℕ := 1 + sum_natural (n - 2)

noncomputable def black_squares_in_row (n : ℕ) : ℕ := 
  if squares_in_row n % 2 = 1 then (squares_in_row n - 1) / 2 else squares_in_row n / 2

theorem black_squares_in_20th_row : black_squares_in_row 20 = 85 := 
by
  sorry

end NUMINAMATH_GPT_black_squares_in_20th_row_l2252_225233


namespace NUMINAMATH_GPT_ratio_of_inscribed_squares_l2252_225218

open Real

-- Condition: A square inscribed in a right triangle with sides 3, 4, and 5
def inscribedSquareInRightTriangle1 (x : ℝ) (a b c : ℝ) : Prop :=
  a = 3 ∧ b = 4 ∧ c = 5 ∧ x = 12 / 7

-- Condition: A square inscribed in a different right triangle with sides 5, 12, and 13
def inscribedSquareInRightTriangle2 (y : ℝ) (d e f : ℝ) : Prop :=
  d = 5 ∧ e = 12 ∧ f = 13 ∧ y = 169 / 37

-- The ratio x / y is 444 / 1183
theorem ratio_of_inscribed_squares (x y : ℝ) (a b c d e f : ℝ) :
  inscribedSquareInRightTriangle1 x a b c →
  inscribedSquareInRightTriangle2 y d e f →
  x / y = 444 / 1183 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_ratio_of_inscribed_squares_l2252_225218


namespace NUMINAMATH_GPT_solve_consecutive_integers_solve_consecutive_even_integers_l2252_225227

-- Conditions: x, y, z, w are positive integers and x + y + z + w = 46.
def consecutive_integers_solution (x y z w : ℕ) : Prop :=
  x < y ∧ y < z ∧ z < w ∧ (x + 1 = y) ∧ (y + 1 = z) ∧ (z + 1 = w) ∧ (x + y + z + w = 46)

def consecutive_even_integers_solution (x y z w : ℕ) : Prop :=
  x < y ∧ y < z ∧ z < w ∧ (x + 2 = y) ∧ (y + 2 = z) ∧ (z + 2 = w) ∧ (x + y + z + w = 46)

-- Proof that consecutive integers can solve the equation II (x + y + z + w = 46)
theorem solve_consecutive_integers : ∃ x y z w : ℕ, consecutive_integers_solution x y z w :=
sorry

-- Proof that consecutive even integers can solve the equation II (x + y + z + w = 46)
theorem solve_consecutive_even_integers : ∃ x y z w : ℕ, consecutive_even_integers_solution x y z w :=
sorry

end NUMINAMATH_GPT_solve_consecutive_integers_solve_consecutive_even_integers_l2252_225227


namespace NUMINAMATH_GPT_zhuzhuxia_defeats_monsters_l2252_225281

theorem zhuzhuxia_defeats_monsters {a : ℕ} (H1 : zhuzhuxia_total_defeated_monsters = 20) :
  zhuzhuxia_total_defeated_by_monsters = 8 :=
sorry

end NUMINAMATH_GPT_zhuzhuxia_defeats_monsters_l2252_225281


namespace NUMINAMATH_GPT_probability_of_choosing_A_on_second_day_l2252_225273

-- Definitions of the probabilities given in the problem conditions.
def p_first_day_A := 0.5
def p_first_day_B := 0.5
def p_second_day_A_given_first_day_A := 0.6
def p_second_day_A_given_first_day_B := 0.5

-- Define the problem to be proved in Lean 4
theorem probability_of_choosing_A_on_second_day :
  (p_first_day_A * p_second_day_A_given_first_day_A) +
  (p_first_day_B * p_second_day_A_given_first_day_B) = 0.55 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_choosing_A_on_second_day_l2252_225273


namespace NUMINAMATH_GPT_factorize_problem1_factorize_problem2_l2252_225207

theorem factorize_problem1 (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 :=
by sorry

theorem factorize_problem2 (x y : ℝ) : 
  (x - y)^3 - 16 * (x - y) = (x - y) * (x - y + 4) * (x - y - 4) :=
by sorry

end NUMINAMATH_GPT_factorize_problem1_factorize_problem2_l2252_225207


namespace NUMINAMATH_GPT_add_and_multiply_l2252_225292

def num1 : ℝ := 0.0034
def num2 : ℝ := 0.125
def num3 : ℝ := 0.00678
def sum := num1 + num2 + num3

theorem add_and_multiply :
  (sum * 2) = 0.27036 := by
  sorry

end NUMINAMATH_GPT_add_and_multiply_l2252_225292


namespace NUMINAMATH_GPT_line_through_intersection_points_l2252_225267

noncomputable def circle1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 10 }
noncomputable def circle2 := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 3)^2 = 10 }

theorem line_through_intersection_points (p : ℝ × ℝ) (hp1 : p ∈ circle1) (hp2 : p ∈ circle2) :
  p.1 + 3 * p.2 - 5 = 0 :=
sorry

end NUMINAMATH_GPT_line_through_intersection_points_l2252_225267


namespace NUMINAMATH_GPT_max_ratio_three_digit_l2252_225283

theorem max_ratio_three_digit (x a b c : ℕ) (h1 : 100 * a + 10 * b + c = x) (h2 : 1 ≤ a ∧ a ≤ 9)
  (h3 : 0 ≤ b ∧ b ≤ 9) (h4 : 0 ≤ c ∧ c ≤ 9) : 
  (x : ℚ) / (a + b + c) ≤ 100 := sorry

end NUMINAMATH_GPT_max_ratio_three_digit_l2252_225283


namespace NUMINAMATH_GPT_riding_time_fraction_l2252_225245

-- Definitions for conditions
def M : ℕ := 6
def total_days : ℕ := 6
def max_time_days : ℕ := 2
def part_time_days : ℕ := 2
def fixed_time : ℝ := 1.5
def total_riding_time : ℝ := 21

-- Prove the statement
theorem riding_time_fraction :
  ∃ F : ℝ, 2 * M + 2 * fixed_time + 2 * F * M = total_riding_time ∧ F = 0.5 :=
by
  exists 0.5
  sorry

end NUMINAMATH_GPT_riding_time_fraction_l2252_225245


namespace NUMINAMATH_GPT_players_scores_l2252_225208

/-- Lean code to verify the scores of three players in a guessing game -/
theorem players_scores (H F S : ℕ) (h1 : H = 42) (h2 : F - H = 24) (h3 : S - F = 18) (h4 : H < F) (h5 : H < S) : 
  F = 66 ∧ S = 84 :=
by
  sorry

end NUMINAMATH_GPT_players_scores_l2252_225208


namespace NUMINAMATH_GPT_expression_increase_fraction_l2252_225211

theorem expression_increase_fraction (x y : ℝ) :
  let x' := 1.4 * x
  let y' := 1.4 * y
  let original := x * y^2
  let increased := x' * y'^2
  increased - original = (1744/1000) * original := by
sorry

end NUMINAMATH_GPT_expression_increase_fraction_l2252_225211


namespace NUMINAMATH_GPT_min_value_x_y_l2252_225215

open Real

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1/y) * (x + 1/y - 1024) + (y + 1/x) * (y + 1/x - 1024) ≥ -524288 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_x_y_l2252_225215


namespace NUMINAMATH_GPT_remainder_of_sum_mod_eight_l2252_225230

theorem remainder_of_sum_mod_eight (m : ℤ) : 
  ((10 - 3 * m) + (5 * m + 6)) % 8 = (2 * m) % 8 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_mod_eight_l2252_225230


namespace NUMINAMATH_GPT_factor_expression_l2252_225261

variable (a b : ℤ)

theorem factor_expression : 2 * a^2 * b - 4 * a * b^2 + 2 * b^3 = 2 * b * (a - b)^2 := 
sorry

end NUMINAMATH_GPT_factor_expression_l2252_225261


namespace NUMINAMATH_GPT_find_floors_l2252_225240

theorem find_floors
  (a b : ℕ)
  (alexie_bathrooms_per_floor : ℕ := 3)
  (alexie_bedrooms_per_floor : ℕ := 2)
  (baptiste_bathrooms_per_floor : ℕ := 4)
  (baptiste_bedrooms_per_floor : ℕ := 3)
  (total_bathrooms : ℕ := 25)
  (total_bedrooms : ℕ := 18)
  (h1 : alexie_bathrooms_per_floor * a + baptiste_bathrooms_per_floor * b = total_bathrooms)
  (h2 : alexie_bedrooms_per_floor * a + baptiste_bedrooms_per_floor * b = total_bedrooms) :
  a = 3 ∧ b = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_floors_l2252_225240


namespace NUMINAMATH_GPT_hcf_of_two_numbers_l2252_225276

noncomputable def find_hcf (x y : ℕ) (lcm_xy : ℕ) (prod_xy : ℕ) : ℕ :=
  prod_xy / lcm_xy

theorem hcf_of_two_numbers (x y : ℕ) (lcm_xy: ℕ) (prod_xy: ℕ) 
  (h_lcm: lcm x y = lcm_xy) (h_prod: x * y = prod_xy) :
  find_hcf x y lcm_xy prod_xy = 75 :=
by
  sorry

end NUMINAMATH_GPT_hcf_of_two_numbers_l2252_225276


namespace NUMINAMATH_GPT_initial_percentage_of_milk_l2252_225229

theorem initial_percentage_of_milk 
  (initial_solution_volume : ℝ)
  (extra_water_volume : ℝ)
  (desired_percentage : ℝ)
  (new_total_volume : ℝ)
  (initial_percentage : ℝ) :
  initial_solution_volume = 60 →
  extra_water_volume = 33.33333333333333 →
  desired_percentage = 54 →
  new_total_volume = initial_solution_volume + extra_water_volume →
  (initial_percentage / 100 * initial_solution_volume = desired_percentage / 100 * new_total_volume) →
  initial_percentage = 84 := 
by 
  intros initial_volume_eq extra_water_eq desired_perc_eq new_volume_eq equation
  -- proof steps here
  sorry

end NUMINAMATH_GPT_initial_percentage_of_milk_l2252_225229


namespace NUMINAMATH_GPT_money_left_is_correct_l2252_225241

-- Define initial amount of money Dan has
def initial_amount : ℕ := 3

-- Define the cost of the candy bar
def candy_cost : ℕ := 1

-- Define the money left after the purchase
def money_left : ℕ := initial_amount - candy_cost

-- The theorem stating that the money left is 2
theorem money_left_is_correct : money_left = 2 := by
  sorry

end NUMINAMATH_GPT_money_left_is_correct_l2252_225241


namespace NUMINAMATH_GPT_side_length_of_square_l2252_225268

theorem side_length_of_square (P : ℕ) (h1 : P = 28) (h2 : P = 4 * s) : s = 7 :=
  by sorry

end NUMINAMATH_GPT_side_length_of_square_l2252_225268


namespace NUMINAMATH_GPT_average_height_corrected_l2252_225209

theorem average_height_corrected (students : ℕ) (incorrect_avg_height : ℝ) (incorrect_height : ℝ) (actual_height : ℝ)
  (h1 : students = 20)
  (h2 : incorrect_avg_height = 175)
  (h3 : incorrect_height = 151)
  (h4 : actual_height = 111) :
  (incorrect_avg_height * students - incorrect_height + actual_height) / students = 173 :=
by
  sorry

end NUMINAMATH_GPT_average_height_corrected_l2252_225209


namespace NUMINAMATH_GPT_most_cost_effective_payment_l2252_225288

theorem most_cost_effective_payment :
  let worker_days := 5 * 10
  let hourly_rate_per_worker := 8 * 10 * 4
  let paint_cost := 4800
  let area_painted := 150
  let cost_option_1 := worker_days * 30
  let cost_option_2 := paint_cost * 0.30
  let cost_option_3 := area_painted * 12
  let cost_option_4 := 5 * hourly_rate_per_worker
  (cost_option_2 < cost_option_1) ∧ (cost_option_2 < cost_option_3) ∧ (cost_option_2 < cost_option_4) :=
by
  sorry

end NUMINAMATH_GPT_most_cost_effective_payment_l2252_225288


namespace NUMINAMATH_GPT_temperature_fraction_l2252_225216

def current_temperature : ℤ := 84
def temperature_decrease : ℤ := 21

theorem temperature_fraction :
  (current_temperature - temperature_decrease) = (3 * current_temperature / 4) := 
by
  sorry

end NUMINAMATH_GPT_temperature_fraction_l2252_225216


namespace NUMINAMATH_GPT_sixth_oak_placement_l2252_225272

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_aligned (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

noncomputable def intersection_point (p1 p2 p3 p4 : Point) : Point := 
  let m1 := (p2.y - p1.y) / (p2.x - p1.x)
  let m2 := (p4.y - p3.y) / (p4.x - p3.x)
  let c1 := p1.y - (m1 * p1.x)
  let c2 := p3.y - (m2 * p3.x)
  let x := (c2 - c1) / (m1 - m2)
  let y := m1 * x + c1
  ⟨x, y⟩

theorem sixth_oak_placement 
  (A1 A2 A3 B1 B2 B3 : Point) 
  (hA : ¬ is_aligned A1 A2 A3)
  (hB : ¬ is_aligned B1 B2 B3) :
  ∃ P : Point, (∃ (C1 C2 : Point), C1 = A1 ∧ C2 = B1 ∧ is_aligned C1 C2 P) ∧ 
               (∃ (C3 C4 : Point), C3 = A2 ∧ C4 = B2 ∧ is_aligned C3 C4 P) := by
  sorry

end NUMINAMATH_GPT_sixth_oak_placement_l2252_225272


namespace NUMINAMATH_GPT_saved_percent_l2252_225258

-- Definitions for conditions:
def last_year_saved (S : ℝ) : ℝ := 0.10 * S
def this_year_salary (S : ℝ) : ℝ := 1.10 * S
def this_year_saved (S : ℝ) : ℝ := 0.06 * (1.10 * S)

-- Given conditions and proof goal:
theorem saved_percent (S : ℝ) (hl_last_year_saved : last_year_saved S = 0.10 * S)
  (hl_this_year_salary : this_year_salary S = 1.10 * S)
  (hl_this_year_saved : this_year_saved S = 0.066 * S) :
  (this_year_saved S / last_year_saved S) * 100 = 66 :=
by
  sorry

end NUMINAMATH_GPT_saved_percent_l2252_225258


namespace NUMINAMATH_GPT_value_large_cube_l2252_225244

-- Definitions based on conditions
def volume_small := 1 -- volume of one-inch cube in cubic inches
def volume_large := 64 -- volume of four-inch cube in cubic inches
def value_small : ℝ := 1000 -- value of one-inch cube of gold in dollars
def proportion (x y : ℝ) : Prop := y = 64 * x -- proportionality condition

-- Prove that the value of the four-inch cube of gold is $64000
theorem value_large_cube : proportion value_small 64000 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_value_large_cube_l2252_225244


namespace NUMINAMATH_GPT_problem_1_problem_2_l2252_225297

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- 1. Prove that A ∩ B = {x | -2 < x ≤ 2}
theorem problem_1 : A ∩ B = {x | -2 < x ∧ x ≤ 2} :=
by
  sorry

-- 2. Prove that (complement U A) ∪ B = {x | x ≤ 2 ∨ x ≥ 3}
theorem problem_2 : (U \ A) ∪ B = {x | x ≤ 2 ∨ x ≥ 3} :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2252_225297


namespace NUMINAMATH_GPT_cos_alpha_in_second_quadrant_l2252_225296

theorem cos_alpha_in_second_quadrant 
  (alpha : ℝ) 
  (h1 : π / 2 < alpha ∧ alpha < π)
  (h2 : ∀ x y : ℝ, 2 * x + (Real.tan alpha) * y + 1 = 0 → 8 / 3 = -(2 / (Real.tan alpha))) :
  Real.cos alpha = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_in_second_quadrant_l2252_225296


namespace NUMINAMATH_GPT_kylie_coins_left_l2252_225278

-- Definitions for each condition
def coins_from_piggy_bank : ℕ := 15
def coins_from_brother : ℕ := 13
def coins_from_father : ℕ := 8
def coins_given_to_friend : ℕ := 21

-- The total coins Kylie has initially
def initial_coins : ℕ := coins_from_piggy_bank + coins_from_brother
def total_coins_after_father : ℕ := initial_coins + coins_from_father
def coins_left : ℕ := total_coins_after_father - coins_given_to_friend

-- The theorem to prove the final number of coins left is 15
theorem kylie_coins_left : coins_left = 15 :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_kylie_coins_left_l2252_225278


namespace NUMINAMATH_GPT_number_added_is_10_l2252_225246

-- Define the conditions.
def number_thought_of : ℕ := 55
def result : ℕ := 21

-- Define the statement of the problem.
theorem number_added_is_10 : ∃ (y : ℕ), (number_thought_of / 5 + y = result) ∧ (y = 10) := by
  sorry

end NUMINAMATH_GPT_number_added_is_10_l2252_225246


namespace NUMINAMATH_GPT_min_value_of_number_l2252_225271

theorem min_value_of_number (a b c d : ℕ) (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 9) (h6 : 1 ≤ d) : 
  a + b * 10 + c * 100 + d * 1000 = 1119 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_number_l2252_225271


namespace NUMINAMATH_GPT_area_of_quadrilateral_is_correct_l2252_225262

noncomputable def area_of_quadrilateral_BGFAC : ℝ :=
  let a := 3 -- side of the equilateral triangle
  let triangle_area := (a^2 * Real.sqrt 3) / 4 -- area of ABC
  let ratio_AG_GC := 2 -- ratio AG:GC = 2:1
  let area_AGC := triangle_area / 3 -- area of triangle AGC
  let area_BGC := triangle_area / 3 -- area of triangle BGC
  let area_BFC := (2 : ℝ) * triangle_area / 3 -- area of triangle BFC
  let area_BGFC := area_BGC + area_BFC -- area of quadrilateral BGFC
  area_BGFC

theorem area_of_quadrilateral_is_correct :
  area_of_quadrilateral_BGFAC = (3 * Real.sqrt 3) / 2 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_is_correct_l2252_225262


namespace NUMINAMATH_GPT_percentage_taken_l2252_225203

theorem percentage_taken (P : ℝ) (h : (P / 100) * 150 - 40 = 50) : P = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_taken_l2252_225203


namespace NUMINAMATH_GPT_prove_f_x1_minus_f_x2_lt_zero_l2252_225232

variable {f : ℝ → ℝ}

-- Define even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Specify that f is decreasing for x < 0
def decreasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < 0 → y < 0 → x < y → f x > f y

theorem prove_f_x1_minus_f_x2_lt_zero (hx1x2 : |x1| < |x2|)
  (h_even : even_function f)
  (h_decreasing : decreasing_on_negative f) :
  f x1 - f x2 < 0 :=
sorry

end NUMINAMATH_GPT_prove_f_x1_minus_f_x2_lt_zero_l2252_225232


namespace NUMINAMATH_GPT_find_area_triangle_boc_l2252_225255

noncomputable def area_ΔBOC := 21

theorem find_area_triangle_boc (A B C K O : Type) 
  [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C] [NormedAddCommGroup K] [NormedAddCommGroup O]
  (AC : ℝ) (AB : ℝ) (h1 : AC = 14) (h2 : AB = 6)
  (circle_centered_on_AC : Prop)
  (K_on_BC : Prop)
  (angle_BAK_eq_angle_ACB : Prop)
  (midpoint_O_AC : Prop)
  (angle_AKC_eq_90 : Prop)
  (area_ABC : Prop) : 
  area_ΔBOC = 21 := 
sorry

end NUMINAMATH_GPT_find_area_triangle_boc_l2252_225255


namespace NUMINAMATH_GPT_tangent_parallel_l2252_225256

noncomputable def f (x : ℝ) : ℝ := x^4 - x

theorem tangent_parallel (P : ℝ × ℝ) (hP : P.1 = 1) (hP_cond : P.2 = f P.1) 
  (tangent_parallel : ∀ x, deriv f x = 3) : P = (1, 0) := 
by 
  have h_deriv : deriv f 1 = 4 * 1^3 - 1 := by sorry
  have slope_eq : deriv f (P.1) = 3 := by sorry
  have solve_a : P.1 = 1 := by sorry
  have solve_b : f 1 = 0 := by sorry
  exact sorry

end NUMINAMATH_GPT_tangent_parallel_l2252_225256


namespace NUMINAMATH_GPT_find_number_l2252_225259

theorem find_number 
  (x : ℝ)
  (h : (1 / 10) * x - (1 / 1000) * x = 700) :
  x = 700000 / 99 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l2252_225259


namespace NUMINAMATH_GPT_Derek_is_42_l2252_225222

def Aunt_Anne_age : ℕ := 36

def Brianna_age : ℕ := (2 * Aunt_Anne_age) / 3

def Caitlin_age : ℕ := Brianna_age - 3

def Derek_age : ℕ := 2 * Caitlin_age

theorem Derek_is_42 : Derek_age = 42 := by
  sorry

end NUMINAMATH_GPT_Derek_is_42_l2252_225222


namespace NUMINAMATH_GPT_track_and_field_unit_incorrect_l2252_225250

theorem track_and_field_unit_incorrect :
  ∀ (L : ℝ), L = 200 → "mm" ≠ "m" → false :=
by
  intros L hL hUnit
  sorry

end NUMINAMATH_GPT_track_and_field_unit_incorrect_l2252_225250


namespace NUMINAMATH_GPT_inequality_proof_l2252_225266

theorem inequality_proof (x y : ℝ) : 5 * x^2 + y^2 + 4 ≥ 4 * x + 4 * x * y :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2252_225266


namespace NUMINAMATH_GPT_model_tower_height_l2252_225274

-- Definitions based on conditions
def height_actual_tower : ℝ := 60
def volume_actual_tower : ℝ := 80000
def volume_model_tower : ℝ := 0.5

-- Theorem statement
theorem model_tower_height (h: ℝ) : h = 0.15 :=
by
  sorry

end NUMINAMATH_GPT_model_tower_height_l2252_225274


namespace NUMINAMATH_GPT_cost_price_percentage_l2252_225286

theorem cost_price_percentage (MP CP : ℝ) (h_discount : 0.75 * MP = CP * 1.171875) :
  ((CP / MP) * 100) = 64 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_percentage_l2252_225286


namespace NUMINAMATH_GPT_hyunwoo_cookies_l2252_225254

theorem hyunwoo_cookies (packs_initial : Nat) (pieces_per_pack : Nat) (packs_given_away : Nat)
  (h1 : packs_initial = 226) (h2 : pieces_per_pack = 3) (h3 : packs_given_away = 3) :
  (packs_initial - packs_given_away) * pieces_per_pack = 669 := 
by
  sorry

end NUMINAMATH_GPT_hyunwoo_cookies_l2252_225254


namespace NUMINAMATH_GPT_annual_return_percentage_l2252_225236

theorem annual_return_percentage (initial_value final_value gain : ℕ)
    (h1 : initial_value = 8000)
    (h2 : final_value = initial_value + 400)
    (h3 : gain = final_value - initial_value) :
    (gain * 100 / initial_value) = 5 := by
  sorry

end NUMINAMATH_GPT_annual_return_percentage_l2252_225236


namespace NUMINAMATH_GPT_iced_tea_cost_is_correct_l2252_225205

noncomputable def iced_tea_cost (cost_cappuccino cost_latte cost_espresso : ℝ) (num_cappuccino num_iced_tea num_latte num_espresso : ℕ) (bill_amount change_amount : ℝ) : ℝ :=
  let total_cappuccino_cost := cost_cappuccino * num_cappuccino
  let total_latte_cost := cost_latte * num_latte
  let total_espresso_cost := cost_espresso * num_espresso
  let total_spent := bill_amount - change_amount
  let total_other_cost := total_cappuccino_cost + total_latte_cost + total_espresso_cost
  let total_iced_tea_cost := total_spent - total_other_cost
  total_iced_tea_cost / num_iced_tea

theorem iced_tea_cost_is_correct:
  iced_tea_cost 2 1.5 1 3 2 2 2 20 3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_iced_tea_cost_is_correct_l2252_225205


namespace NUMINAMATH_GPT_correct_equations_l2252_225206

variable (x y : ℝ)

theorem correct_equations :
  (18 * x = y + 3) ∧ (17 * x = y - 4) ↔ (18 * x = y + 3) ∧ (17 * x = y - 4) :=
by
  sorry

end NUMINAMATH_GPT_correct_equations_l2252_225206


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2252_225225

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x > -1 ∧ x < 1) ↔ (x^2 - 1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2252_225225


namespace NUMINAMATH_GPT_evaluate_expression_l2252_225220

theorem evaluate_expression : (2^2010 * 3^2012 * 25) / 6^2011 = 37.5 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2252_225220


namespace NUMINAMATH_GPT_find_y_l2252_225212

theorem find_y (y : ℝ) (hy_pos : y > 0) (hy_prop : y^2 / 100 = 9) : y = 30 := by
  sorry

end NUMINAMATH_GPT_find_y_l2252_225212


namespace NUMINAMATH_GPT_john_total_distance_l2252_225201

theorem john_total_distance :
  let speed1 := 35
  let time1 := 2
  let distance1 := speed1 * time1

  let speed2 := 55
  let time2 := 3
  let distance2 := speed2 * time2

  let total_distance := distance1 + distance2

  total_distance = 235 := by
    sorry

end NUMINAMATH_GPT_john_total_distance_l2252_225201


namespace NUMINAMATH_GPT_actual_distance_traveled_l2252_225282

theorem actual_distance_traveled
  (t : ℕ)
  (H1 : 6 * t = 3 * t + 15) :
  3 * t = 15 :=
by
  exact sorry

end NUMINAMATH_GPT_actual_distance_traveled_l2252_225282


namespace NUMINAMATH_GPT_sqrt_nested_eq_x_pow_eleven_eighths_l2252_225243

theorem sqrt_nested_eq_x_pow_eleven_eighths (x : ℝ) (hx : 0 ≤ x) : 
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (11 / 8) :=
  sorry

end NUMINAMATH_GPT_sqrt_nested_eq_x_pow_eleven_eighths_l2252_225243


namespace NUMINAMATH_GPT_number_of_carbon_atoms_l2252_225260

-- Definitions and Conditions
def hydrogen_atoms : ℕ := 6
def molecular_weight : ℕ := 78
def hydrogen_atomic_weight : ℕ := 1
def carbon_atomic_weight : ℕ := 12

-- Theorem Statement: Number of Carbon Atoms
theorem number_of_carbon_atoms 
  (H_atoms : ℕ := hydrogen_atoms)
  (M_weight : ℕ := molecular_weight)
  (H_weight : ℕ := hydrogen_atomic_weight)
  (C_weight : ℕ := carbon_atomic_weight) : 
  (M_weight - H_atoms * H_weight) / C_weight = 6 :=
sorry

end NUMINAMATH_GPT_number_of_carbon_atoms_l2252_225260


namespace NUMINAMATH_GPT_M_positive_l2252_225248

theorem M_positive (x y : ℝ) : (3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13) > 0 :=
by
  sorry

end NUMINAMATH_GPT_M_positive_l2252_225248


namespace NUMINAMATH_GPT_counter_example_exists_l2252_225279

theorem counter_example_exists : 
  ∃ n : ℕ, n ≥ 2 ∧ ¬(∃ k : ℕ, (2 ^ 2 ^ n) % (2 ^ n - 1) = 4 ^ k) :=
  sorry

end NUMINAMATH_GPT_counter_example_exists_l2252_225279


namespace NUMINAMATH_GPT_arithmetic_square_root_of_9_is_3_l2252_225228

-- Define the arithmetic square root property
def is_arithmetic_square_root (x : ℝ) (n : ℝ) : Prop :=
  x * x = n ∧ x ≥ 0

-- The main theorem: The arithmetic square root of 9 is 3
theorem arithmetic_square_root_of_9_is_3 : 
  is_arithmetic_square_root 3 9 :=
by
  -- This is where the proof would go, but since only the statement is required:
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_9_is_3_l2252_225228


namespace NUMINAMATH_GPT_pencils_per_student_l2252_225269

theorem pencils_per_student (total_pencils : ℕ) (students : ℕ) (pencils_per_student : ℕ) 
  (h_total : total_pencils = 125) 
  (h_students : students = 25) 
  (h_div : pencils_per_student = total_pencils / students) : 
  pencils_per_student = 5 :=
by
  sorry

end NUMINAMATH_GPT_pencils_per_student_l2252_225269


namespace NUMINAMATH_GPT_find_f_of_4_l2252_225247

noncomputable def power_function (x : ℝ) (α : ℝ) : ℝ := x^α

theorem find_f_of_4 :
  (∃ α : ℝ, power_function 3 α = Real.sqrt 3) →
  power_function 4 (1/2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_4_l2252_225247


namespace NUMINAMATH_GPT_faster_ship_speed_l2252_225235

theorem faster_ship_speed :
  ∀ (x y : ℕ),
    (200 + 100 = 300) → -- Total distance covered for both directions
    (x + y) * 10 = 300 → -- Opposite direction equation
    (x - y) * 25 = 300 → -- Same direction equation
    x = 21 := 
by
  intros x y _ eq1 eq2
  sorry

end NUMINAMATH_GPT_faster_ship_speed_l2252_225235
