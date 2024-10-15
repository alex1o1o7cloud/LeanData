import Mathlib

namespace NUMINAMATH_GPT_lcm_factor_of_hcf_and_larger_number_l1592_159235

theorem lcm_factor_of_hcf_and_larger_number (A B : ℕ) (hcf : ℕ) (hlarger : A = 450) (hhcf : hcf = 30) (hwrel : A % hcf = 0) : ∃ x y, x = 15 ∧ (A * B = hcf * x * y) :=
by
  sorry

end NUMINAMATH_GPT_lcm_factor_of_hcf_and_larger_number_l1592_159235


namespace NUMINAMATH_GPT_number_of_convex_quadrilaterals_with_parallel_sides_l1592_159244

-- Define a regular 20-sided polygon
def regular_20_sided_polygon : Type := 
  { p : ℕ // 0 < p ∧ p ≤ 20 }

-- The main theorem statement
theorem number_of_convex_quadrilaterals_with_parallel_sides : 
  ∃ (n : ℕ), n = 765 :=
sorry

end NUMINAMATH_GPT_number_of_convex_quadrilaterals_with_parallel_sides_l1592_159244


namespace NUMINAMATH_GPT_initial_amount_l1592_159228

theorem initial_amount (x : ℝ) (h1 : x = (2*x - 10) / 2) (h2 : x = (4*x - 30) / 2) (h3 : 8*x - 70 = 0) : x = 8.75 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_l1592_159228


namespace NUMINAMATH_GPT_solve_price_of_meat_l1592_159230

def price_of_meat_per_ounce (x : ℕ) : Prop :=
  16 * x - 30 = 8 * x + 18

theorem solve_price_of_meat : ∃ x, price_of_meat_per_ounce x ∧ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_price_of_meat_l1592_159230


namespace NUMINAMATH_GPT_each_person_gets_4_roses_l1592_159250

def ricky_roses_total : Nat := 40
def roses_stolen : Nat := 4
def people : Nat := 9
def remaining_roses : Nat := ricky_roses_total - roses_stolen
def roses_per_person : Nat := remaining_roses / people

theorem each_person_gets_4_roses : roses_per_person = 4 := by
  sorry

end NUMINAMATH_GPT_each_person_gets_4_roses_l1592_159250


namespace NUMINAMATH_GPT_gumball_difference_l1592_159289

theorem gumball_difference :
  let c := 17
  let l := 12
  let a := 24
  let t := 8
  let n := c + l + a + t
  let low := 14
  let high := 32
  ∃ x : ℕ, (low ≤ (n + x) / 7 ∧ (n + x) / 7 ≤ high) →
  (∃ x_min x_max, x_min ≤ x ∧ x ≤ x_max ∧ x_max - x_min = 126) :=
by
  sorry

end NUMINAMATH_GPT_gumball_difference_l1592_159289


namespace NUMINAMATH_GPT_coin_game_goal_l1592_159231

theorem coin_game_goal (a b : ℕ) (h_diff : a ≤ 3 * b ∧ b ≤ 3 * a) (h_sum : (a + b) % 4 = 0) :
  ∃ x y p q : ℕ, (a + 2 * x - 2 * y = 3 * (b + 2 * p - 2 * q)) ∨ (a + 2 * y - 2 * x = 3 * (b + 2 * q - 2 * p)) :=
sorry

end NUMINAMATH_GPT_coin_game_goal_l1592_159231


namespace NUMINAMATH_GPT_find_f_k_l_l1592_159205

noncomputable
def f : ℕ → ℕ := sorry

axiom f_condition_1 : f 1 = 1
axiom f_condition_2 : ∀ n : ℕ, 3 * f n * f (2 * n + 1) = f (2 * n) * (1 + 3 * f n)
axiom f_condition_3 : ∀ n : ℕ, f (2 * n) < 6 * f n

theorem find_f_k_l (k l : ℕ) (h : k < l) : 
  (f k + f l = 293) ↔ 
  ((k = 121 ∧ l = 4) ∨ (k = 118 ∧ l = 4) ∨ 
   (k = 109 ∧ l = 16) ∨ (k = 16 ∧ l = 109)) := 
by 
  sorry

end NUMINAMATH_GPT_find_f_k_l_l1592_159205


namespace NUMINAMATH_GPT_binary_to_decimal_l1592_159219

theorem binary_to_decimal : 
  (0 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4) = 18 := 
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_binary_to_decimal_l1592_159219


namespace NUMINAMATH_GPT_tree_height_increase_l1592_159214

theorem tree_height_increase
  (initial_height : ℝ)
  (height_increase : ℝ)
  (h6 : ℝ) :
  initial_height = 4 →
  (0 ≤ height_increase) →
  height_increase * 6 + initial_height = (height_increase * 4 + initial_height) + 1 / 7 * (height_increase * 4 + initial_height) →
  height_increase = 2 / 5 :=
by
  intro h_initial h_nonneg h_eq
  sorry

end NUMINAMATH_GPT_tree_height_increase_l1592_159214


namespace NUMINAMATH_GPT_valid_grid_iff_divisible_by_9_l1592_159252

-- Definitions for the letters used in the grid
inductive Letter
| I
| M
| O

-- Function that captures the condition that each row and column must contain exactly one-third of each letter
def valid_row_col (n : ℕ) (grid : ℕ -> ℕ -> Letter) : Prop :=
  ∀ row, (∃ count_I, ∃ count_M, ∃ count_O,
    count_I = n / 3 ∧ count_M = n / 3 ∧ count_O = n / 3 ∧
    (∀ col, grid row col ∈ [Letter.I, Letter.M, Letter.O])) ∧
  ∀ col, (∃ count_I, ∃ count_M, ∃ count_O,
    count_I = n / 3 ∧ count_M = n / 3 ∧ count_O = n / 3 ∧
    (∀ row, grid row col ∈ [Letter.I, Letter.M, Letter.O]))

-- Function that captures the condition that each diagonal must contain exactly one-third of each letter when the length is a multiple of 3
def valid_diagonals (n : ℕ) (grid : ℕ -> ℕ -> Letter) : Prop :=
  ∀ k, (3 ∣ k → (∃ count_I, ∃ count_M, ∃ count_O,
    count_I = k / 3 ∧ count_M = k / 3 ∧ count_O = k / 3 ∧
    ((∀ (i j : ℕ), (i + j = k) → grid i j ∈ [Letter.I, Letter.M, Letter.O]) ∨
     (∀ (i j : ℕ), (i - j = k) → grid i j ∈ [Letter.I, Letter.M, Letter.O]))))

-- The main theorem stating that if we can fill the grid according to the rules, then n must be a multiple of 9
theorem valid_grid_iff_divisible_by_9 (n : ℕ) :
  (∃ grid : ℕ → ℕ → Letter, valid_row_col n grid ∧ valid_diagonals n grid) ↔ 9 ∣ n :=
by
  sorry

end NUMINAMATH_GPT_valid_grid_iff_divisible_by_9_l1592_159252


namespace NUMINAMATH_GPT_mark_money_l1592_159293

theorem mark_money (M : ℝ) 
  (h1 : (1 / 2) * M + 14 + (1 / 3) * M + 16 + (1 / 4) * M + 18 = M) : 
  M = 576 := 
sorry

end NUMINAMATH_GPT_mark_money_l1592_159293


namespace NUMINAMATH_GPT_monomial_sum_l1592_159202

variable {x y : ℝ}

theorem monomial_sum (a : ℝ) (h : -2 * x^2 * y^3 + 5 * x^(a-1) * y^3 = c * x^k * y^3) : a = 3 :=
  by
  sorry

end NUMINAMATH_GPT_monomial_sum_l1592_159202


namespace NUMINAMATH_GPT_total_number_of_outfits_l1592_159251

-- Definitions of the conditions as functions/values
def num_shirts : Nat := 8
def num_pants : Nat := 5
def num_ties_options : Nat := 4 + 1  -- 4 ties + 1 option for no tie
def num_belts_options : Nat := 2 + 1  -- 2 belts + 1 option for no belt

-- Lean statement to formulate the proof problem
theorem total_number_of_outfits : 
  num_shirts * num_pants * num_ties_options * num_belts_options = 600 := by
  sorry

end NUMINAMATH_GPT_total_number_of_outfits_l1592_159251


namespace NUMINAMATH_GPT_parabola_focus_l1592_159229

theorem parabola_focus : ∃ f : ℝ, 
  (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (1/4 + f))^2)) ∧
  f = 1/8 := 
by
  sorry

end NUMINAMATH_GPT_parabola_focus_l1592_159229


namespace NUMINAMATH_GPT_sum_of_numbers_in_row_l1592_159233

theorem sum_of_numbers_in_row 
  (n : ℕ)
  (sum_eq : (n * (3 * n - 1)) / 2 = 20112) : 
  n = 1006 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_in_row_l1592_159233


namespace NUMINAMATH_GPT_vacation_cost_proof_l1592_159218

noncomputable def vacation_cost (C : ℝ) :=
  C / 5 - C / 8 = 120

theorem vacation_cost_proof {C : ℝ} (h : vacation_cost C) : C = 1600 :=
by
  sorry

end NUMINAMATH_GPT_vacation_cost_proof_l1592_159218


namespace NUMINAMATH_GPT_num_adults_attended_l1592_159296

-- Definitions for the conditions
def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := 13
def num_children : ℕ := 28
def total_revenue : ℕ := 5122

-- The goal is to prove the number of adults who attended the show
theorem num_adults_attended :
  ∃ (A : ℕ), A * ticket_price_adult + num_children * ticket_price_child = total_revenue ∧ A = 183 :=
by
  sorry

end NUMINAMATH_GPT_num_adults_attended_l1592_159296


namespace NUMINAMATH_GPT_calculate_value_l1592_159256

theorem calculate_value :
  12 * ( (1 / 3 : ℝ) + (1 / 4) + (1 / 6) )⁻¹ = 16 :=
sorry

end NUMINAMATH_GPT_calculate_value_l1592_159256


namespace NUMINAMATH_GPT_find_minimal_positive_n_l1592_159269

-- Define the arithmetic sequence
def arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the conditions
variables (a1 d : ℤ)
axiom condition_1 : arithmetic_seq a1 d 11 / arithmetic_seq a1 d 10 < -1
axiom condition_2 : ∃ n : ℕ, ∀ k : ℕ, k ≤ n → sum_arithmetic_seq a1 d k ≤ sum_arithmetic_seq a1 d n

-- Prove the statement
theorem find_minimal_positive_n : ∃ n : ℕ, n = 19 ∧ sum_arithmetic_seq a1 d n = 0 ∧
  (∀ m : ℕ, 0 < sum_arithmetic_seq a1 d m ∧ sum_arithmetic_seq a1 d m < sum_arithmetic_seq a1 d n) :=
sorry

end NUMINAMATH_GPT_find_minimal_positive_n_l1592_159269


namespace NUMINAMATH_GPT_squares_total_l1592_159264

def number_of_squares (figure : Type) : ℕ := sorry

theorem squares_total (figure : Type) : number_of_squares figure = 38 := sorry

end NUMINAMATH_GPT_squares_total_l1592_159264


namespace NUMINAMATH_GPT_age_ordered_youngest_to_oldest_l1592_159290

variable (M Q S : Nat)

theorem age_ordered_youngest_to_oldest 
  (h1 : M = Q ∨ S = Q)
  (h2 : M ≥ Q)
  (h3 : S ≤ Q) : S = Q ∧ M > Q :=
by 
  sorry

end NUMINAMATH_GPT_age_ordered_youngest_to_oldest_l1592_159290


namespace NUMINAMATH_GPT_B_finishes_remaining_work_in_3_days_l1592_159271

theorem B_finishes_remaining_work_in_3_days
  (A_works_in : ℕ)
  (B_works_in : ℕ)
  (work_days_together : ℕ)
  (A_leaves : A_works_in = 4)
  (B_leaves : B_works_in = 10)
  (work_days : work_days_together = 2) :
  ∃ days_remaining : ℕ, days_remaining = 3 :=
by
  sorry

end NUMINAMATH_GPT_B_finishes_remaining_work_in_3_days_l1592_159271


namespace NUMINAMATH_GPT_ex_ineq_l1592_159220

theorem ex_ineq (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 40) :
  x + y = 2 + 2 * Real.sqrt 3 ∨ x + y = 2 - 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_ex_ineq_l1592_159220


namespace NUMINAMATH_GPT_continuous_at_1_l1592_159299

theorem continuous_at_1 (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ x, |x - 1| < δ → |(-4 * x^2 - 6) - (-10)| < ε :=
by
  sorry

end NUMINAMATH_GPT_continuous_at_1_l1592_159299


namespace NUMINAMATH_GPT_cone_radius_height_ratio_l1592_159224

theorem cone_radius_height_ratio 
  (V : ℝ) (π : ℝ) (r h : ℝ)
  (circumference : ℝ) 
  (original_height : ℝ)
  (new_volume : ℝ)
  (volume_formula : V = (1/3) * π * r^2 * h)
  (radius_from_circumference : 2 * π * r = circumference)
  (base_circumference : circumference = 28 * π)
  (original_height_eq : original_height = 45)
  (new_volume_eq : new_volume = 441 * π) :
  (r / h) = 14 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cone_radius_height_ratio_l1592_159224


namespace NUMINAMATH_GPT_combined_capacity_is_40_l1592_159275

/-- Define the bus capacity as 1/6 the train capacity -/
def bus_capacity (train_capacity : ℕ) := train_capacity / 6

/-- There are two buses in the problem -/
def number_of_buses := 2

/-- The train capacity given in the problem is 120 people -/
def train_capacity := 120

/-- The combined capacity of the two buses is -/
def combined_bus_capacity := number_of_buses * bus_capacity train_capacity

/-- Proof that the combined capacity of the two buses is 40 people -/
theorem combined_capacity_is_40 : combined_bus_capacity = 40 := by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_combined_capacity_is_40_l1592_159275


namespace NUMINAMATH_GPT_tan_domain_l1592_159204

theorem tan_domain (x : ℝ) : 
  (∃ (k : ℤ), x = k * Real.pi - Real.pi / 4) ↔ 
  ¬(∃ (k : ℤ), x = k * Real.pi - Real.pi / 4) :=
sorry

end NUMINAMATH_GPT_tan_domain_l1592_159204


namespace NUMINAMATH_GPT_second_rate_of_return_l1592_159270

namespace Investment

def total_investment : ℝ := 33000
def interest_total : ℝ := 970
def investment_4_percent : ℝ := 13000
def interest_rate_4_percent : ℝ := 0.04

def amount_second_investment : ℝ := total_investment - investment_4_percent
def interest_from_first_part : ℝ := interest_rate_4_percent * investment_4_percent
def interest_from_second_part (R : ℝ) : ℝ := R * amount_second_investment

theorem second_rate_of_return : (∃ R : ℝ, interest_from_first_part + interest_from_second_part R = interest_total) → 
  R = 0.0225 :=
by
  intro h
  sorry

end Investment

end NUMINAMATH_GPT_second_rate_of_return_l1592_159270


namespace NUMINAMATH_GPT_simplify_fraction_sum_eq_zero_l1592_159283

variable (a b c : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hc : c ≠ 0)
variable (h : a + b + 2 * c = 0)

theorem simplify_fraction_sum_eq_zero :
  (1 / (b^2 + 4*c^2 - a^2) + 1 / (a^2 + 4*c^2 - b^2) + 1 / (a^2 + b^2 - 4*c^2)) = 0 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_sum_eq_zero_l1592_159283


namespace NUMINAMATH_GPT_prime_factor_count_l1592_159259

theorem prime_factor_count (n : ℕ) (H : 22 + n + 2 = 29) : n = 5 := 
  sorry

end NUMINAMATH_GPT_prime_factor_count_l1592_159259


namespace NUMINAMATH_GPT_prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l1592_159207

-- Definitions for the conditions
def white_balls : ℕ := 4
def black_balls : ℕ := 2

-- Total number of balls
def total_balls : ℕ := white_balls + black_balls

-- Part (I): Without Replacement
theorem prob_at_least_one_black_without_replacement : 
  (20 - 4) / 20 = 4 / 5 :=
by sorry

-- Part (II): With Replacement
theorem prob_exactly_one_black_with_replacement : 
  (3 * 2 * 4 * 4) / (6 * 6 * 6) = 4 / 9 :=
by sorry

end NUMINAMATH_GPT_prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l1592_159207


namespace NUMINAMATH_GPT_number_of_houses_l1592_159236

theorem number_of_houses (total_mail_per_block : ℕ) (mail_per_house : ℕ) (h1 : total_mail_per_block = 24) (h2 : mail_per_house = 4) : total_mail_per_block / mail_per_house = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_houses_l1592_159236


namespace NUMINAMATH_GPT_greatest_points_for_top_teams_l1592_159221

-- Definitions as per the conditions
def teams := 9 -- Number of teams
def games_per_pair := 2 -- Each team plays every other team twice
def points_win := 3 -- Points for a win
def points_draw := 1 -- Points for a draw
def points_loss := 0 -- Points for a loss

-- Total number of games played
def total_games := (teams * (teams - 1) / 2) * games_per_pair

-- Total points available in the tournament
def total_points := total_games * points_win

-- Given the conditions, prove that the greatest possible number of total points each of the top three teams can accumulate is 42.
theorem greatest_points_for_top_teams :
  ∃ k, (∀ A B C : ℕ, A = B ∧ B = C → A ≤ k) ∧ k = 42 :=
sorry

end NUMINAMATH_GPT_greatest_points_for_top_teams_l1592_159221


namespace NUMINAMATH_GPT_fifth_boat_more_than_average_l1592_159272

theorem fifth_boat_more_than_average :
  let total_people := 2 + 4 + 3 + 5 + 6
  let num_boats := 5
  let average_people := total_people / num_boats
  let fifth_boat := 6
  (fifth_boat - average_people) = 2 :=
by
  sorry

end NUMINAMATH_GPT_fifth_boat_more_than_average_l1592_159272


namespace NUMINAMATH_GPT_probability_of_nonzero_product_probability_of_valid_dice_values_l1592_159281

def dice_values := {x : ℕ | 1 ≤ x ∧ x ≤ 6}

def valid_dice_values := {x : ℕ | 2 ≤ x ∧ x ≤ 6}

noncomputable def probability_no_one : ℚ := 625 / 1296

theorem probability_of_nonzero_product (a b c d : ℕ) 
  (ha : a ∈ dice_values) (hb : b ∈ dice_values) 
  (hc : c ∈ dice_values) (hd : d ∈ dice_values) : 
  (a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0 ↔ 
  (a ∈ valid_dice_values ∧ b ∈ valid_dice_values ∧ 
   c ∈ valid_dice_values ∧ d ∈ valid_dice_values) :=
sorry

theorem probability_of_valid_dice_values : 
  probability_no_one = (5 / 6) ^ 4 :=
sorry

end NUMINAMATH_GPT_probability_of_nonzero_product_probability_of_valid_dice_values_l1592_159281


namespace NUMINAMATH_GPT_jelly_beans_correct_l1592_159232

-- Define the constants and conditions
def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def total_amount : ℕ := sandra_savings + mother_gift + father_gift

def candy_cost : ℕ := 5 / 10 -- == 0.5
def jelly_bean_cost : ℕ := 2 / 10 -- == 0.2

def candies_bought : ℕ := 14
def money_spent_on_candies : ℕ := candies_bought * candy_cost

def remaining_money : ℕ := total_amount - money_spent_on_candies
def money_left : ℕ := 11

-- Prove the number of jelly beans bought is 20
def number_of_jelly_beans : ℕ :=
  (remaining_money - money_left) / jelly_bean_cost

theorem jelly_beans_correct : number_of_jelly_beans = 20 :=
sorry

end NUMINAMATH_GPT_jelly_beans_correct_l1592_159232


namespace NUMINAMATH_GPT_caterer_preparations_l1592_159222

theorem caterer_preparations :
  let b_guests := 84
  let a_guests := (2/3) * b_guests
  let total_guests := b_guests + a_guests
  let extra_plates := 10
  let total_plates := total_guests + extra_plates

  let cherry_tomatoes_per_plate := 5
  let regular_asparagus_per_plate := 8
  let vegetarian_asparagus_per_plate := 6
  let larger_asparagus_per_plate := 12
  let larger_asparagus_portion_guests := 0.1 * total_plates

  let blueberries_per_plate := 15
  let raspberries_per_plate := 8
  let blackberries_per_plate := 10

  let cherry_tomatoes_needed := cherry_tomatoes_per_plate * total_plates

  let regular_portion_guests := 0.9 * total_plates
  let regular_asparagus_needed := regular_asparagus_per_plate * regular_portion_guests
  let larger_asparagus_needed := larger_asparagus_per_plate * larger_asparagus_portion_guests
  let asparagus_needed := regular_asparagus_needed + larger_asparagus_needed

  let blueberries_needed := blueberries_per_plate * total_plates
  let raspberries_needed := raspberries_per_plate * total_plates
  let blackberries_needed := blackberries_per_plate * total_plates

  cherry_tomatoes_needed = 750 ∧
  asparagus_needed = 1260 ∧
  blueberries_needed = 2250 ∧
  raspberries_needed = 1200 ∧
  blackberries_needed = 1500 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_caterer_preparations_l1592_159222


namespace NUMINAMATH_GPT_john_took_away_oranges_l1592_159285

-- Define the initial number of oranges Melissa had.
def initial_oranges : ℕ := 70

-- Define the number of oranges Melissa has left.
def oranges_left : ℕ := 51

-- Define the expected number of oranges John took away.
def oranges_taken : ℕ := 19

-- The theorem that needs to be proven.
theorem john_took_away_oranges :
  initial_oranges - oranges_left = oranges_taken :=
by
  sorry

end NUMINAMATH_GPT_john_took_away_oranges_l1592_159285


namespace NUMINAMATH_GPT_no_solution_for_floor_eq_l1592_159257

theorem no_solution_for_floor_eq :
  ∀ s : ℝ, ¬ (⌊s⌋ + s = 15.6) :=
by sorry

end NUMINAMATH_GPT_no_solution_for_floor_eq_l1592_159257


namespace NUMINAMATH_GPT_seating_arrangements_equal_600_l1592_159284

-- Definitions based on the problem conditions
def number_of_people : Nat := 4
def number_of_chairs : Nat := 8
def consecutive_empty_seats : Nat := 3

-- Theorem statement
theorem seating_arrangements_equal_600
  (h_people : number_of_people = 4)
  (h_chairs : number_of_chairs = 8)
  (h_consecutive_empty_seats : consecutive_empty_seats = 3) :
  (∃ (arrangements : Nat), arrangements = 600) :=
sorry

end NUMINAMATH_GPT_seating_arrangements_equal_600_l1592_159284


namespace NUMINAMATH_GPT_additional_charge_per_segment_l1592_159262

variable (initial_fee : ℝ := 2.35)
variable (total_charge : ℝ := 5.5)
variable (distance : ℝ := 3.6)
variable (segment_length : ℝ := (2/5 : ℝ))

theorem additional_charge_per_segment :
  let number_of_segments := distance / segment_length
  let charge_for_distance := total_charge - initial_fee
  let additional_charge_per_segment := charge_for_distance / number_of_segments
  additional_charge_per_segment = 0.35 :=
by
  sorry

end NUMINAMATH_GPT_additional_charge_per_segment_l1592_159262


namespace NUMINAMATH_GPT_solution_set_l1592_159266

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn (a b : ℝ) : f (a + b) = f a + f b - 1
axiom monotonic (x y : ℝ) : x ≤ y → f x ≤ f y
axiom initial_condition : f 4 = 5

theorem solution_set : {m : ℝ | f (3 * m^2 - m - 2) < 3} = {m : ℝ | -4/3 < m ∧ m < 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1592_159266


namespace NUMINAMATH_GPT_inequality_abc_d_l1592_159217

theorem inequality_abc_d (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (H1 : d ≥ a) (H2 : d ≥ b) (H3 : d ≥ c) : a * (d - b) + b * (d - c) + c * (d - a) ≤ d^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_d_l1592_159217


namespace NUMINAMATH_GPT_min_value_expr_l1592_159234

theorem min_value_expr : 
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 10 * y ≥ -3 := 
sorry

end NUMINAMATH_GPT_min_value_expr_l1592_159234


namespace NUMINAMATH_GPT_gain_percent_of_50C_eq_25S_l1592_159277

variable {C S : ℝ}

theorem gain_percent_of_50C_eq_25S (h : 50 * C = 25 * S) : 
  ((S - C) / C) * 100 = 100 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_of_50C_eq_25S_l1592_159277


namespace NUMINAMATH_GPT_exist_matrices_with_dets_l1592_159241

noncomputable section

open Matrix BigOperators

variables {α : Type} [Field α] [DecidableEq α]

theorem exist_matrices_with_dets (m n : ℕ) (h₁ : 1 < m) (h₂ : 1 < n)
  (αs : Fin m → α) (β : α) :
  ∃ (A : Fin m → Matrix (Fin n) (Fin n) α), (∀ i, det (A i) = αs i) ∧ det (∑ i, A i) = β :=
sorry

end NUMINAMATH_GPT_exist_matrices_with_dets_l1592_159241


namespace NUMINAMATH_GPT_coeff_x5_term_l1592_159242

-- We define the binomial coefficient function C(n, k)
def C (n k : ℕ) : ℕ := Nat.choose n k

-- We define the expression in question
noncomputable def expr (x : ℝ) : ℝ := (1/x + 2*x)^7

-- The coefficient of x^5 term in the expansion
theorem coeff_x5_term : 
  let general_term (r : ℕ) (x : ℝ) := (2:ℝ)^r * C 7 r * x^(2 * r - 7)
  -- r is chosen such that the power of x is 5
  let r := 6
  -- The coefficient for r=6
  general_term r 1 = 448 := 
by sorry

end NUMINAMATH_GPT_coeff_x5_term_l1592_159242


namespace NUMINAMATH_GPT_part1_part2_l1592_159274

open Complex

noncomputable def z0 : ℂ := 3 + 4 * Complex.I

theorem part1 (z1 : ℂ) (h : z1 * z0 = 3 * z1 + z0) : z1.im = -3/4 := by
  sorry

theorem part2 (x : ℝ) 
    (z : ℂ := (x^2 - 4 * x) + (x + 2) * Complex.I) 
    (z0_conj : ℂ := 3 - 4 * Complex.I) 
    (h : (z + z0_conj).re < 0 ∧ (z + z0_conj).im > 0) : 
    2 < x ∧ x < 3 :=
  by 
  sorry

end NUMINAMATH_GPT_part1_part2_l1592_159274


namespace NUMINAMATH_GPT_measure_8_liters_with_buckets_l1592_159258

theorem measure_8_liters_with_buckets (capacity_B10 capacity_B6 : ℕ) (B10_target : ℕ) (B10_initial B6_initial : ℕ) : 
  capacity_B10 = 10 ∧ capacity_B6 = 6 ∧ B10_target = 8 ∧ B10_initial = 0 ∧ B6_initial = 0 →
  ∃ (B10 B6 : ℕ), B10 = 8 ∧ (B10 ≥ 0 ∧ B10 ≤ capacity_B10) ∧ (B6 ≥ 0 ∧ B6 ≤ capacity_B6) :=
by
  sorry

end NUMINAMATH_GPT_measure_8_liters_with_buckets_l1592_159258


namespace NUMINAMATH_GPT_area_enclosed_l1592_159261

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (x - Real.pi / 3)
noncomputable def area_between (a b : ℝ) (f g : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, (g x - f x)

theorem area_enclosed (h₀ : 0 ≤ 2 * Real.pi) (h₁ : 2 * Real.pi ≤ 2 * Real.pi) :
  area_between (2 * Real.pi / 3) (5 * Real.pi / 3) f g = 2 :=
by 
  sorry

end NUMINAMATH_GPT_area_enclosed_l1592_159261


namespace NUMINAMATH_GPT_remainder_when_M_divided_by_32_l1592_159294

-- Define M as the product of all odd primes less than 32.
def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_M_divided_by_32 :
    M % 32 = 1 := by
  -- We must prove that M % 32 = 1.
  sorry

end NUMINAMATH_GPT_remainder_when_M_divided_by_32_l1592_159294


namespace NUMINAMATH_GPT_semifinalists_count_l1592_159216

theorem semifinalists_count (n : ℕ) (h : (n - 2) * (n - 3) * (n - 4) = 336) : n = 10 := 
by {
  sorry
}

end NUMINAMATH_GPT_semifinalists_count_l1592_159216


namespace NUMINAMATH_GPT_price_increase_ratio_l1592_159278

theorem price_increase_ratio 
  (c : ℝ)
  (h1 : 351 = c * 1.30) :
  (c + 351) / c = 2.3 :=
sorry

end NUMINAMATH_GPT_price_increase_ratio_l1592_159278


namespace NUMINAMATH_GPT_parabola_focus_directrix_distance_l1592_159226

theorem parabola_focus_directrix_distance (a : ℝ) (h_pos : a > 0) (h_dist : 1 / (2 * 2 * a) = 1) : a = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_directrix_distance_l1592_159226


namespace NUMINAMATH_GPT_initial_investment_l1592_159201

theorem initial_investment (b : ℝ) (t_b : ℝ) (t_a : ℝ) (ratio_profit : ℝ) (x : ℝ) :
  b = 36000 → t_b = 4.5 → t_a = 12 → ratio_profit = 2 →
  (x * t_a) / (b * t_b) = ratio_profit → x = 27000 := 
by
  intros hb ht_b ht_a hr hp
  rw [hb, ht_b, ht_a, hr] at hp
  sorry

end NUMINAMATH_GPT_initial_investment_l1592_159201


namespace NUMINAMATH_GPT_Rohan_earning_after_6_months_l1592_159246

def farm_area : ℕ := 20
def trees_per_sqm : ℕ := 2
def coconuts_per_tree : ℕ := 6
def harvest_interval : ℕ := 3
def sale_price : ℝ := 0.50
def total_months : ℕ := 6

theorem Rohan_earning_after_6_months :
  farm_area * trees_per_sqm * coconuts_per_tree * (total_months / harvest_interval) * sale_price 
    = 240 := by
  sorry

end NUMINAMATH_GPT_Rohan_earning_after_6_months_l1592_159246


namespace NUMINAMATH_GPT_maximize_k_l1592_159249

open Real

theorem maximize_k (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : log x + log y = 0)
  (h₄ : ∀ x y : ℝ, 0 < x → 0 < y → k * (x + 2 * y) ≤ x^2 + 4 * y^2) : k ≤ sqrt 2 :=
sorry

end NUMINAMATH_GPT_maximize_k_l1592_159249


namespace NUMINAMATH_GPT_circumscribed_quadrilateral_identity_l1592_159263

variables 
  (α β γ θ : ℝ)
  (h_angle_sum : α + β + γ + θ = 180)
  (OA OB OC OD AB BC CD DA : ℝ)
  (h_OA : OA = 1 / Real.sin α)
  (h_OB : OB = 1 / Real.sin β)
  (h_OC : OC = 1 / Real.sin γ)
  (h_OD : OD = 1 / Real.sin θ)
  (h_AB : AB = Real.sin (α + β) / (Real.sin α * Real.sin β))
  (h_BC : BC = Real.sin (β + γ) / (Real.sin β * Real.sin γ))
  (h_CD : CD = Real.sin (γ + θ) / (Real.sin γ * Real.sin θ))
  (h_DA : DA = Real.sin (θ + α) / (Real.sin θ * Real.sin α))

theorem circumscribed_quadrilateral_identity :
  OA * OC + OB * OD = Real.sqrt (AB * BC * CD * DA) := 
sorry

end NUMINAMATH_GPT_circumscribed_quadrilateral_identity_l1592_159263


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1592_159298

theorem simplify_and_evaluate_expression (a : ℤ) (ha : a = -2) : 
  (1 + 1 / (a - 1)) / ((2 * a) / (a ^ 2 - 1)) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1592_159298


namespace NUMINAMATH_GPT_apple_price_36_kgs_l1592_159260

theorem apple_price_36_kgs (l q : ℕ) 
  (H1 : ∀ n, n ≤ 30 → ∀ n', n' ≤ 30 → l * n' = 100)
  (H2 : 30 * l + 3 * q = 168) : 
  30 * l + 6 * q = 186 :=
by {
  sorry
}

end NUMINAMATH_GPT_apple_price_36_kgs_l1592_159260


namespace NUMINAMATH_GPT_solve_inequality_zero_solve_inequality_neg_solve_inequality_pos_l1592_159273

variable (a x : ℝ)

def inequality (a x : ℝ) : Prop := (1 - a * x) ^ 2 < 1

theorem solve_inequality_zero : a = 0 → ¬∃ x, inequality a x := by
  sorry

theorem solve_inequality_neg (h : a < 0) : (∃ x, inequality a x) →
  ∀ x, inequality a x ↔ (a ≠ 0 ∧ (2 / a < x ∧ x < 0)) := by
  sorry

theorem solve_inequality_pos (h : a > 0) : (∃ x, inequality a x) →
  ∀ x, inequality a x ↔ (a ≠ 0 ∧ (0 < x ∧ x < 2 / a)) := by
  sorry

end NUMINAMATH_GPT_solve_inequality_zero_solve_inequality_neg_solve_inequality_pos_l1592_159273


namespace NUMINAMATH_GPT_no_cubic_term_l1592_159288

noncomputable def p1 (a b k : ℝ) : ℝ := -2 * a * b + (1 / 3) * k * a^2 * b + 5 * b^2
noncomputable def p2 (a b : ℝ) : ℝ := b^2 + 3 * a^2 * b - 5 * a * b + 1
noncomputable def diff (a b k : ℝ) : ℝ := p1 a b k - p2 a b
noncomputable def cubic_term_coeff (a b k : ℝ) : ℝ := (1 / 3) * k - 3

theorem no_cubic_term (a b : ℝ) : ∀ k, (cubic_term_coeff a b k = 0) → k = 9 :=
by
  intro k h
  sorry

end NUMINAMATH_GPT_no_cubic_term_l1592_159288


namespace NUMINAMATH_GPT_b_completes_work_in_48_days_l1592_159267

noncomputable def work_rate (days : ℕ) : ℚ := 1 / days

theorem b_completes_work_in_48_days (a b c : ℕ) 
  (h1 : work_rate (a + b) = work_rate 16)
  (h2 : work_rate a = work_rate 24)
  (h3 : work_rate c = work_rate 48) :
  work_rate b = work_rate 48 :=
by
  sorry

end NUMINAMATH_GPT_b_completes_work_in_48_days_l1592_159267


namespace NUMINAMATH_GPT_find_a9_l1592_159291

variable (a : ℕ → ℝ)

theorem find_a9 (h1 : a 4 - a 2 = -2) (h2 : a 7 = -3) : a 9 = -5 :=
sorry

end NUMINAMATH_GPT_find_a9_l1592_159291


namespace NUMINAMATH_GPT_combined_height_l1592_159253

/-- Given that Mr. Martinez is two feet taller than Chiquita and Chiquita is 5 feet tall, prove that their combined height is 12 feet. -/
theorem combined_height (h_chiquita : ℕ) (h_martinez : ℕ) 
  (h1 : h_chiquita = 5) (h2 : h_martinez = h_chiquita + 2) : 
  h_chiquita + h_martinez = 12 :=
by sorry

end NUMINAMATH_GPT_combined_height_l1592_159253


namespace NUMINAMATH_GPT_solve_first_system_solve_second_system_solve_third_system_l1592_159279

-- First system of equations
theorem solve_first_system (x y : ℝ) 
  (h1 : 2*x + 3*y = 16)
  (h2 : x + 4*y = 13) : 
  x = 5 ∧ y = 2 := 
sorry

-- Second system of equations
theorem solve_second_system (x y : ℝ) 
  (h1 : 0.3*x - y = 1)
  (h2 : 0.2*x - 0.5*y = 19) : 
  x = 370 ∧ y = 110 := 
sorry

-- Third system of equations
theorem solve_third_system (x y : ℝ) 
  (h1 : 3 * (x - 1) = y + 5)
  (h2 : (x + 2) / 2 = ((y - 1) / 3) + 1) : 
  x = 6 ∧ y = 10 := 
sorry

end NUMINAMATH_GPT_solve_first_system_solve_second_system_solve_third_system_l1592_159279


namespace NUMINAMATH_GPT_greatest_b_for_no_minus_nine_in_range_l1592_159247

theorem greatest_b_for_no_minus_nine_in_range :
  ∃ b_max : ℤ, (b_max = 16) ∧ (∀ b : ℤ, (b^2 < 288) ↔ (b ≤ 16)) :=
by
  sorry

end NUMINAMATH_GPT_greatest_b_for_no_minus_nine_in_range_l1592_159247


namespace NUMINAMATH_GPT_proof_problem_l1592_159225

noncomputable def red_balls : ℕ := 5
noncomputable def black_balls : ℕ := 2
noncomputable def total_balls : ℕ := red_balls + black_balls
noncomputable def draws : ℕ := 3

noncomputable def prob_red_ball := red_balls / total_balls
noncomputable def prob_black_ball := black_balls / total_balls

noncomputable def E_X : ℚ := (1/7) + 2*(4/7) + 3*(2/7)
noncomputable def E_Y : ℚ := 2*(1/7) + 1*(4/7) + 0*(2/7)
noncomputable def E_xi : ℚ := 3 * (5/7)

noncomputable def D_X : ℚ := (1 - 15/7) ^ 2 * (1/7) + (2 - 15/7) ^ 2 * (4/7) + (3 - 15/7) ^ 2 * (2/7)
noncomputable def D_Y : ℚ := (2 - 6/7) ^ 2 * (1/7) + (1 - 6/7) ^ 2 * (4/7) + (0 - 6/7) ^ 2 * (2/7)
noncomputable def D_xi : ℚ := 3 * (5/7) * (1 - 5/7)

theorem proof_problem :
  (E_X / E_Y = 5 / 2) ∧ 
  (D_X ≤ D_Y) ∧ 
  (E_X = E_xi) ∧ 
  (D_X < D_xi) :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_problem_l1592_159225


namespace NUMINAMATH_GPT_seashells_total_correct_l1592_159243

def total_seashells (red_shells green_shells other_shells : ℕ) : ℕ :=
  red_shells + green_shells + other_shells

theorem seashells_total_correct :
  total_seashells 76 49 166 = 291 :=
by
  sorry

end NUMINAMATH_GPT_seashells_total_correct_l1592_159243


namespace NUMINAMATH_GPT_selling_price_correct_l1592_159200

/-- Define the initial cost of the gaming PC. -/
def initial_pc_cost : ℝ := 1200

/-- Define the cost of the new video card. -/
def new_video_card_cost : ℝ := 500

/-- Define the total spending after selling the old card. -/
def total_spending : ℝ := 1400

/-- Define the selling price of the old card -/
def selling_price_of_old_card : ℝ := (initial_pc_cost + new_video_card_cost) - total_spending

/-- Prove that John sold the old card for $300. -/
theorem selling_price_correct : selling_price_of_old_card = 300 := by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l1592_159200


namespace NUMINAMATH_GPT_tan_of_angle_in_third_quadrant_l1592_159213

theorem tan_of_angle_in_third_quadrant 
  (α : ℝ) 
  (h1 : α < -π / 2 ∧ α > -π) 
  (h2 : Real.sin α = -Real.sqrt 5 / 5) :
  Real.tan α = 1 / 2 := 
sorry

end NUMINAMATH_GPT_tan_of_angle_in_third_quadrant_l1592_159213


namespace NUMINAMATH_GPT_find_x_l1592_159212

theorem find_x (x : ℝ) (h : (1 + x) / (5 + x) = 1 / 3) : x = 1 :=
sorry

end NUMINAMATH_GPT_find_x_l1592_159212


namespace NUMINAMATH_GPT_linear_function_points_relation_l1592_159206

theorem linear_function_points_relation :
  ∀ (y1 y2 : ℝ), 
  (y1 = -3 * 2 + 1) ∧ (y2 = -3 * 3 + 1) → y1 > y2 :=
by
  intro y1 y2
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_linear_function_points_relation_l1592_159206


namespace NUMINAMATH_GPT_largest_three_digit_multiple_of_8_and_sum_24_is_888_l1592_159208

noncomputable def largest_three_digit_multiple_of_8_with_digit_sum_24 : ℕ :=
  888

theorem largest_three_digit_multiple_of_8_and_sum_24_is_888 :
  ∃ n : ℕ, (300 ≤ n ∧ n ≤ 999) ∧ (n % 8 = 0) ∧ ((n.digits 10).sum = 24) ∧ n = largest_three_digit_multiple_of_8_with_digit_sum_24 :=
by
  existsi 888
  sorry

end NUMINAMATH_GPT_largest_three_digit_multiple_of_8_and_sum_24_is_888_l1592_159208


namespace NUMINAMATH_GPT_complementary_angle_beta_l1592_159255

theorem complementary_angle_beta (α β : ℝ) (h_compl : α + β = 90) (h_alpha : α = 40) : β = 50 :=
by
  -- Skipping the proof, which initial assumption should be defined.
  sorry

end NUMINAMATH_GPT_complementary_angle_beta_l1592_159255


namespace NUMINAMATH_GPT_max_truck_speed_l1592_159292

theorem max_truck_speed (D : ℝ) (C : ℝ) (F : ℝ) (L : ℝ → ℝ) (T : ℝ) (x : ℝ) : 
  D = 125 ∧ C = 30 ∧ F = 1000 ∧ (∀ s, L s = 2 * s) ∧ (∃ s, D / s * C + F + L s ≤ T) → x ≤ 75 :=
by
  sorry

end NUMINAMATH_GPT_max_truck_speed_l1592_159292


namespace NUMINAMATH_GPT_range_of_a_l1592_159268

open Real

theorem range_of_a (x y z a : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hsum : x + y + z = 1)
  (heq : a / (x * y * z) = (1 / x) + (1 / y) + (1 / z) - 2) :
  0 < a ∧ a ≤ 7 / 27 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1592_159268


namespace NUMINAMATH_GPT_find_angle_C_find_triangle_area_l1592_159239

theorem find_angle_C (A B C : ℝ) (a b c : ℝ) 
  (h1 : B = Real.pi / 4) 
  (h2 : Real.cos A - Real.cos (2 * A) = 0) 
  (h3 : B + C + A = Real.pi) :
  C = Real.pi / 12 :=
by
  sorry

theorem find_triangle_area (A B C : ℝ) (a b c : ℝ)
  (h1 : B = Real.pi / 4) 
  (h2 : Real.cos A - Real.cos (2 * A) = 0) 
  (h3 : b^2 + c^2 = a - b * c + 2) 
  (h4 : B + C + A = Real.pi) 
  (h5 : a^2 = b^2 + c^2 + b * c) :
  (1/2) * a * b * Real.sin C = 1 - Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_C_find_triangle_area_l1592_159239


namespace NUMINAMATH_GPT_choosing_top_cases_l1592_159248

def original_tops : Nat := 2
def bought_tops : Nat := 4
def total_tops : Nat := original_tops + bought_tops

theorem choosing_top_cases : total_tops = 6 := by
  sorry

end NUMINAMATH_GPT_choosing_top_cases_l1592_159248


namespace NUMINAMATH_GPT_cos_diff_l1592_159245

theorem cos_diff (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.tan α = 2) : 
  Real.cos (α - π / 4) = 3 * Real.sqrt 10 / 10 :=
sorry

end NUMINAMATH_GPT_cos_diff_l1592_159245


namespace NUMINAMATH_GPT_red_not_equal_blue_l1592_159215

theorem red_not_equal_blue (total_cubes : ℕ) (red_cubes : ℕ) (blue_cubes : ℕ) (edge_length : ℕ)
  (total_surface_squares : ℕ) (max_red_squares : ℕ) :
  total_cubes = 27 →
  red_cubes = 9 →
  blue_cubes = 18 →
  edge_length = 3 →
  total_surface_squares = 6 * edge_length^2 →
  max_red_squares = 26 →
  ¬ (total_surface_squares = 2 * max_red_squares) :=
by
  intros
  sorry

end NUMINAMATH_GPT_red_not_equal_blue_l1592_159215


namespace NUMINAMATH_GPT_replace_question_with_division_l1592_159240

theorem replace_question_with_division :
  ∃ op: (ℤ → ℤ → ℤ), (op 8 2) + 5 - (3 - 2) = 8 ∧ 
  (∀ a b, op = Int.div ∧ ((op a b) = a / b)) :=
by
  sorry

end NUMINAMATH_GPT_replace_question_with_division_l1592_159240


namespace NUMINAMATH_GPT_infinite_bad_integers_l1592_159237

theorem infinite_bad_integers (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ᶠ n in at_top, (¬(n^b + 1) ∣ (a^n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_infinite_bad_integers_l1592_159237


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1592_159286

theorem part_a (θ : ℝ) (m : ℕ) : |Real.sin (m * θ)| ≤ m * |Real.sin θ| :=
sorry

theorem part_b (θ₁ θ₂ : ℝ) (m : ℕ) (hm_even : Even m) : 
  |Real.sin (m * θ₂) - Real.sin (m * θ₁)| ≤ m * |Real.sin (θ₂ - θ₁)| :=
sorry

theorem part_c (m : ℕ) (hm_odd : Odd m) : 
  ∃ θ₁ θ₂ : ℝ, |Real.sin (m * θ₂) - Real.sin (m * θ₁)| > m * |Real.sin (θ₂ - θ₁)| :=
sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1592_159286


namespace NUMINAMATH_GPT_min_radius_for_area_l1592_159295

theorem min_radius_for_area (A : ℝ) (hA : A = 500) : ∃ r : ℝ, r = 13 ∧ π * r^2 ≥ A :=
by
  sorry

end NUMINAMATH_GPT_min_radius_for_area_l1592_159295


namespace NUMINAMATH_GPT_population_change_over_3_years_l1592_159265

-- Define the initial conditions
def annual_growth_rate := 0.09
def migration_rate_year1 := -0.01
def migration_rate_year2 := -0.015
def migration_rate_year3 := -0.02
def natural_disaster_rate := -0.03

-- Lemma stating the overall percentage increase in population over three years
theorem population_change_over_3_years :
  (1 + annual_growth_rate) * (1 + migration_rate_year1) * 
  (1 + annual_growth_rate) * (1 + migration_rate_year2) * 
  (1 + annual_growth_rate) * (1 + migration_rate_year3) * 
  (1 + natural_disaster_rate) = 1.195795 := 
sorry

end NUMINAMATH_GPT_population_change_over_3_years_l1592_159265


namespace NUMINAMATH_GPT_extremum_points_of_f_l1592_159280

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x + 1)^3 * Real.exp (x + 1) - Real.exp 1
  else -((if -x < 0 then (-x + 1)^3 * Real.exp (-x + 1) - Real.exp 1 else 0))

theorem extremum_points_of_f : ∃! (a b : ℝ), 
  (∀ x < 0, f x = (x + 1)^3 * Real.exp (x + 1) - Real.exp 1) ∧ (f a = f b) ∧ a ≠ b :=
sorry

end NUMINAMATH_GPT_extremum_points_of_f_l1592_159280


namespace NUMINAMATH_GPT_max_fourth_term_l1592_159297

open Nat

/-- Constants representing the properties of the arithmetic sequence -/
axiom a : ℕ
axiom d : ℕ
axiom pos1 : a > 0
axiom pos2 : a + d > 0
axiom pos3 : a + 2 * d > 0
axiom pos4 : a + 3 * d > 0
axiom pos5 : a + 4 * d > 0
axiom sum_condition : 5 * a + 10 * d = 75

/-- Theorem stating the maximum fourth term of the arithmetic sequence -/
theorem max_fourth_term : a + 3 * d = 22 := sorry

end NUMINAMATH_GPT_max_fourth_term_l1592_159297


namespace NUMINAMATH_GPT_max_k_constant_l1592_159203

theorem max_k_constant : 
  (∃ k, (∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → 
  (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) <= k * Real.sqrt (x + y + z))) 
  ∧ k = Real.sqrt 6 / 2) :=
sorry

end NUMINAMATH_GPT_max_k_constant_l1592_159203


namespace NUMINAMATH_GPT_tip_count_proof_l1592_159282

def initial_customers : ℕ := 29
def additional_customers : ℕ := 20
def customers_who_tipped : ℕ := 15
def total_customers : ℕ := initial_customers + additional_customers
def customers_didn't_tip : ℕ := total_customers - customers_who_tipped

theorem tip_count_proof : customers_didn't_tip = 34 :=
by
  -- This is a proof outline, not the actual proof.
  sorry

end NUMINAMATH_GPT_tip_count_proof_l1592_159282


namespace NUMINAMATH_GPT_probability_of_Q_section_l1592_159210

theorem probability_of_Q_section (sections : ℕ) (Q_sections : ℕ) (h1 : sections = 6) (h2 : Q_sections = 2) :
  Q_sections / sections = 2 / 6 :=
by
  -- solution proof is skipped
  sorry

end NUMINAMATH_GPT_probability_of_Q_section_l1592_159210


namespace NUMINAMATH_GPT_ellipse_eccentricity_l1592_159223

noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

def ellipse_conditions (F1 B : ℝ × ℝ) (c b : ℝ) : Prop :=
  F1 = (-2, 0) ∧ B = (0, 1) ∧ c = 2 ∧ b = 1

theorem ellipse_eccentricity (F1 B : ℝ × ℝ) (c b a : ℝ)
  (h : ellipse_conditions F1 B c b) :
  eccentricity c a = 2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l1592_159223


namespace NUMINAMATH_GPT_switch_pairs_bound_l1592_159276

theorem switch_pairs_bound (odd_blocks_n odd_blocks_prev : ℕ) 
  (switch_pairs_n switch_pairs_prev : ℕ)
  (H1 : switch_pairs_n = 2 * odd_blocks_n)
  (H2 : odd_blocks_n ≤ switch_pairs_prev) : 
  switch_pairs_n ≤ 2 * switch_pairs_prev :=
by
  sorry

end NUMINAMATH_GPT_switch_pairs_bound_l1592_159276


namespace NUMINAMATH_GPT_gary_profit_l1592_159211

theorem gary_profit :
  let total_flour := 8 -- pounds
  let cost_flour := 4 -- dollars
  let large_cakes_flour := 5 -- pounds
  let small_cakes_flour := 3 -- pounds
  let flour_per_large_cake := 0.75 -- pounds per large cake
  let flour_per_small_cake := 0.25 -- pounds per small cake
  let cost_additional_large := 1.5 -- dollars per large cake
  let cost_additional_small := 0.75 -- dollars per small cake
  let cost_baking_equipment := 10 -- dollars
  let revenue_per_large := 6.5 -- dollars per large cake
  let revenue_per_small := 2.5 -- dollars per small cake
  let num_large_cakes := 6 -- (from calculation: ⌊5 / 0.75⌋)
  let num_small_cakes := 12 -- (from calculation: 3 / 0.25)
  let cost_additional_ingredients := num_large_cakes * cost_additional_large + num_small_cakes * cost_additional_small
  let total_revenue := num_large_cakes * revenue_per_large + num_small_cakes * revenue_per_small
  let total_cost := cost_flour + cost_baking_equipment + cost_additional_ingredients
  let profit := total_revenue - total_cost
  profit = 37 := by
  sorry

end NUMINAMATH_GPT_gary_profit_l1592_159211


namespace NUMINAMATH_GPT_calcium_carbonate_required_l1592_159209

theorem calcium_carbonate_required (HCl_moles CaCO3_moles CaCl2_moles CO2_moles H2O_moles : ℕ) 
  (reaction_balanced : CaCO3_moles + 2 * HCl_moles = CaCl2_moles + CO2_moles + H2O_moles) 
  (HCl_moles_value : HCl_moles = 2) : CaCO3_moles = 1 :=
by sorry

end NUMINAMATH_GPT_calcium_carbonate_required_l1592_159209


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l1592_159227

theorem cone_lateral_surface_area (r l : ℝ) (h1 : r = 2) (h2 : l = 5) : 
    0.5 * (2 * Real.pi * r * l) = 10 * Real.pi := by
    sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l1592_159227


namespace NUMINAMATH_GPT_number_of_green_hats_l1592_159254

theorem number_of_green_hats (B G : ℕ) 
  (h1 : B + G = 85) 
  (h2 : 6 * B + 7 * G = 550) : 
  G = 40 := by
  sorry

end NUMINAMATH_GPT_number_of_green_hats_l1592_159254


namespace NUMINAMATH_GPT_find_m_l1592_159238

theorem find_m (x y m : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : x + m * y = 5) : m = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l1592_159238


namespace NUMINAMATH_GPT_number_solution_l1592_159287

theorem number_solution (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
by
  -- The proof is omitted according to the instructions
  sorry

end NUMINAMATH_GPT_number_solution_l1592_159287
