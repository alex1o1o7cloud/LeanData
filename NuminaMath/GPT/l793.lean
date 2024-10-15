import Mathlib

namespace NUMINAMATH_GPT_calculate_expression_l793_79330

variable (x y : ℝ)

theorem calculate_expression (h1 : x + y = 5) (h2 : x * y = 3) : 
   x + (x^4 / y^3) + (y^4 / x^3) + y = 27665 / 27 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l793_79330


namespace NUMINAMATH_GPT_Cora_book_reading_problem_l793_79325

theorem Cora_book_reading_problem
  (total_pages: ℕ)
  (read_monday: ℕ)
  (read_tuesday: ℕ)
  (read_wednesday: ℕ)
  (H: total_pages = 158 ∧ read_monday = 23 ∧ read_tuesday = 38 ∧ read_wednesday = 61) :
  ∃ P: ℕ, 23 + 38 + 61 + P + 2 * P = total_pages ∧ P = 12 :=
  sorry

end NUMINAMATH_GPT_Cora_book_reading_problem_l793_79325


namespace NUMINAMATH_GPT_closest_correct_option_l793_79361

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f x = f (-x + 16)) -- y = f(x + 8) is an even function
variable (h2 : ∀ a b, 8 < a → 8 < b → a < b → f b < f a) -- f is decreasing on (8, +∞)

theorem closest_correct_option :
  f 7 > f 10 := by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_closest_correct_option_l793_79361


namespace NUMINAMATH_GPT_product_of_roots_of_quadratics_l793_79300

noncomputable def product_of_roots : ℝ :=
  let r1 := 2021 / 2020
  let r2 := 2020 / 2019
  let r3 := 2019
  r1 * r2 * r3

theorem product_of_roots_of_quadratics (b : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, 2020 * x1 * x1 + b * x1 + 2021 = 0 ∧ 2020 * x2 * x2 + b * x2 + 2021 = 0) 
  (h2 : ∃ y1 y2 : ℝ, 2019 * y1 * y1 + b * y1 + 2020 = 0 ∧ 2019 * y2 * y2 + b * y2 + 2020 = 0) 
  (h3 : ∃ z1 z2 : ℝ, z1 * z1 + b * z1 + 2019 = 0 ∧ z1 * z1 + b * z2 + 2019 = 0) :
  product_of_roots = 2021 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_of_quadratics_l793_79300


namespace NUMINAMATH_GPT_luke_total_coins_l793_79322

def piles_coins_total (piles_quarters : ℕ) (coins_per_pile_quarters : ℕ) 
                      (piles_dimes : ℕ) (coins_per_pile_dimes : ℕ) 
                      (piles_nickels : ℕ) (coins_per_pile_nickels : ℕ) 
                      (piles_pennies : ℕ) (coins_per_pile_pennies : ℕ) : ℕ :=
  (piles_quarters * coins_per_pile_quarters) +
  (piles_dimes * coins_per_pile_dimes) +
  (piles_nickels * coins_per_pile_nickels) +
  (piles_pennies * coins_per_pile_pennies)

theorem luke_total_coins : 
  piles_coins_total 8 5 6 7 4 4 3 6 = 116 :=
by
  sorry

end NUMINAMATH_GPT_luke_total_coins_l793_79322


namespace NUMINAMATH_GPT_total_elephants_in_two_parks_l793_79395

theorem total_elephants_in_two_parks (n1 n2 : ℕ) (h1 : n1 = 70) (h2 : n2 = 3 * n1) : n1 + n2 = 280 := by
  sorry

end NUMINAMATH_GPT_total_elephants_in_two_parks_l793_79395


namespace NUMINAMATH_GPT_debby_bottles_per_day_l793_79343

theorem debby_bottles_per_day :
  let total_bottles := 153
  let days := 17
  total_bottles / days = 9 :=
by
  sorry

end NUMINAMATH_GPT_debby_bottles_per_day_l793_79343


namespace NUMINAMATH_GPT_find_x_to_print_800_leaflets_in_3_minutes_l793_79302

theorem find_x_to_print_800_leaflets_in_3_minutes (x : ℝ) :
  (800 / 12 + 800 / x = 800 / 3) → (1 / 12 + 1 / x = 1 / 3) :=
by
  intro h
  have h1 : 800 / 12 = 200 / 3 := by norm_num
  have h2 : 800 / 3 = 800 / 3 := by norm_num
  sorry

end NUMINAMATH_GPT_find_x_to_print_800_leaflets_in_3_minutes_l793_79302


namespace NUMINAMATH_GPT_monotonic_intervals_l793_79378

open Set

noncomputable def f (a x : ℝ) : ℝ := - (1 / 3) * a * x^3 + x^2 + 1

theorem monotonic_intervals (a : ℝ) (h : a ≤ 0) :
  (a = 0 → (∀ x : ℝ, (x < 0 → deriv (f a) x < 0) ∧ (0 < x → deriv (f a) x > 0))) ∧
  (a < 0 → (∀ x : ℝ, (x < 2 / a → deriv (f a) x > 0 ∨ deriv (f a) x = 0) ∧ 
                     (2 / a < x → deriv (f a) x < 0 ∨ deriv (f a) x = 0))) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_intervals_l793_79378


namespace NUMINAMATH_GPT_average_speed_l793_79398

section
def flat_sand_speed : ℕ := 60
def downhill_slope_speed : ℕ := flat_sand_speed + 12
def uphill_slope_speed : ℕ := flat_sand_speed - 18

/-- Conner's average speed on flat, downhill, and uphill slopes, each of which he spends one-third of his time traveling on, is 58 miles per hour -/
theorem average_speed : (flat_sand_speed + downhill_slope_speed + uphill_slope_speed) / 3 = 58 := by
  sorry

end

end NUMINAMATH_GPT_average_speed_l793_79398


namespace NUMINAMATH_GPT_color_of_85th_bead_l793_79369

/-- Definition for the repeating pattern of beads -/
def pattern : List String := ["red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

/-- Definition for finding the color of the n-th bead -/
def bead_color (n : Nat) : Option String :=
  let index := (n - 1) % pattern.length
  pattern.get? index

theorem color_of_85th_bead : bead_color 85 = some "yellow" := by
  sorry

end NUMINAMATH_GPT_color_of_85th_bead_l793_79369


namespace NUMINAMATH_GPT_average_monthly_growth_rate_equation_l793_79351

-- Definitions directly from the conditions
def JanuaryOutput : ℝ := 50
def QuarterTotalOutput : ℝ := 175
def averageMonthlyGrowthRate (x : ℝ) : ℝ :=
  JanuaryOutput + JanuaryOutput * (1 + x) + JanuaryOutput * (1 + x) ^ 2

-- The statement to prove that the derived equation is correct
theorem average_monthly_growth_rate_equation (x : ℝ) :
  averageMonthlyGrowthRate x = QuarterTotalOutput :=
sorry

end NUMINAMATH_GPT_average_monthly_growth_rate_equation_l793_79351


namespace NUMINAMATH_GPT_middle_digit_zero_l793_79314

theorem middle_digit_zero (a b c M : ℕ) (h1 : M = 36 * a + 6 * b + c) (h2 : M = 64 * a + 8 * b + c) (ha : 0 ≤ a ∧ a < 6) (hb : 0 ≤ b ∧ b < 6) (hc : 0 ≤ c ∧ c < 6) : 
  b = 0 := 
  by sorry

end NUMINAMATH_GPT_middle_digit_zero_l793_79314


namespace NUMINAMATH_GPT_price_of_shares_l793_79371

variable (share_value : ℝ) (dividend_rate : ℝ) (tax_rate : ℝ) (effective_return : ℝ) (price : ℝ)

-- Given conditions
axiom H1 : share_value = 50
axiom H2 : dividend_rate = 0.185
axiom H3 : tax_rate = 0.05
axiom H4 : effective_return = 0.25
axiom H5 : 0.25 * price = 0.185 * 50 - (0.05 * (0.185 * 50))

-- Prove that the price at which the investor bought the shares is Rs. 35.15
theorem price_of_shares : price = 35.15 :=
by
  sorry

end NUMINAMATH_GPT_price_of_shares_l793_79371


namespace NUMINAMATH_GPT_rain_at_least_once_l793_79319

noncomputable def rain_probability (day_prob : ℚ) (days : ℕ) : ℚ :=
  1 - (1 - day_prob)^days

theorem rain_at_least_once :
  ∀ (day_prob : ℚ) (days : ℕ),
    day_prob = 3/4 → days = 4 →
    rain_probability day_prob days = 255/256 :=
by
  intros day_prob days h1 h2
  sorry

end NUMINAMATH_GPT_rain_at_least_once_l793_79319


namespace NUMINAMATH_GPT_John_l793_79380

theorem John's_score_in_blackjack
  (Theodore_score : ℕ)
  (Zoey_cards : List ℕ)
  (winning_score : ℕ)
  (John_score : ℕ)
  (h1 : Theodore_score = 13)
  (h2 : Zoey_cards = [11, 3, 5])
  (h3 : winning_score = 19)
  (h4 : Zoey_cards.sum = winning_score)
  (h5 : winning_score ≠ Theodore_score) :
  John_score < 19 :=
by
  -- Here we would provide the proof if required
  sorry

end NUMINAMATH_GPT_John_l793_79380


namespace NUMINAMATH_GPT_parallelogram_base_l793_79388

theorem parallelogram_base (height area : ℕ) (h_height : height = 18) (h_area : area = 612) : ∃ base, base = 34 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_parallelogram_base_l793_79388


namespace NUMINAMATH_GPT_solve_z_solutions_l793_79321

noncomputable def z_solutions (z : ℂ) : Prop :=
  z ^ 6 = -16

theorem solve_z_solutions :
  {z : ℂ | z_solutions z} = {2 * Complex.I, -2 * Complex.I} :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_z_solutions_l793_79321


namespace NUMINAMATH_GPT_find_difference_between_larger_and_fraction_smaller_l793_79356

theorem find_difference_between_larger_and_fraction_smaller
  (x y : ℝ) 
  (h1 : x + y = 147)
  (h2 : x - 0.375 * y = 4) : x - 0.375 * y = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_difference_between_larger_and_fraction_smaller_l793_79356


namespace NUMINAMATH_GPT_find_a12_a12_value_l793_79393

variable (a : ℕ → ℝ)

-- Given conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

axiom h1 : a 6 + a 10 = 16
axiom h2 : a 4 = 1

-- Theorem to prove
theorem find_a12 : a 6 + a 10 = a 4 + a 12 := by
  -- Place for the proof
  sorry

theorem a12_value : (∃ a12, a 6 + a 10 = 16 ∧ a 4 = 1 ∧ a 6 + a 10 = a 4 + a12) → a 12 = 15 :=
by
  -- Place for the proof
  sorry

end NUMINAMATH_GPT_find_a12_a12_value_l793_79393


namespace NUMINAMATH_GPT_total_cost_l793_79382

theorem total_cost
  (cost_berries   : ℝ := 11.08)
  (cost_apples    : ℝ := 14.33)
  (cost_peaches   : ℝ := 9.31)
  (cost_grapes    : ℝ := 7.50)
  (cost_bananas   : ℝ := 5.25)
  (cost_pineapples: ℝ := 4.62)
  (total_cost     : ℝ := cost_berries + cost_apples + cost_peaches + cost_grapes + cost_bananas + cost_pineapples) :
  total_cost = 52.09 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l793_79382


namespace NUMINAMATH_GPT_solve_monomial_equation_l793_79333

theorem solve_monomial_equation (x : ℝ) (m n : ℝ) (a b : ℝ) 
  (h1 : m = 2) (h2 : n = 3) 
  (h3 : (1/3) * a^m * b^3 + (-2) * a^2 * b^n = (1/3) * a^2 * b^3 + (-2) * a^2 * b^3) :
  (x - 7) / n - (1 + x) / m = 1 → x = -23 := 
by
  sorry

end NUMINAMATH_GPT_solve_monomial_equation_l793_79333


namespace NUMINAMATH_GPT_wax_initial_amount_l793_79338

def needed : ℕ := 17
def total : ℕ := 574
def initial : ℕ := total - needed

theorem wax_initial_amount :
  initial = 557 :=
by
  sorry

end NUMINAMATH_GPT_wax_initial_amount_l793_79338


namespace NUMINAMATH_GPT_other_x_intercept_l793_79336

theorem other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, y = a * x ^ 2 + b * x + c → (x, y) = (4, -3)) (h_x_intercept : ∀ y, y = a * 1 ^ 2 + b * 1 + c → (1, y) = (1, 0)) : 
  ∃ x, x = 7 := by
sorry

end NUMINAMATH_GPT_other_x_intercept_l793_79336


namespace NUMINAMATH_GPT_actual_speed_of_car_l793_79348

noncomputable def actual_speed (t : ℝ) (d : ℝ) (reduced_speed_factor : ℝ) : ℝ := 
  (d / t) * (1 / reduced_speed_factor)

noncomputable def time_in_hours : ℝ := 1 + (40 / 60) + (48 / 3600)

theorem actual_speed_of_car : 
  actual_speed time_in_hours 42 (5 / 7) = 35 :=
by
  sorry

end NUMINAMATH_GPT_actual_speed_of_car_l793_79348


namespace NUMINAMATH_GPT_coupon_redeem_day_l793_79389

theorem coupon_redeem_day (first_day : ℕ) (redeem_every : ℕ) : 
  (∀ n : ℕ, n < 8 → (first_day + n * redeem_every) % 7 ≠ 6) ↔ (first_day % 7 = 2 ∨ first_day % 7 = 5) :=
by
  sorry

end NUMINAMATH_GPT_coupon_redeem_day_l793_79389


namespace NUMINAMATH_GPT_trigonometric_identity_l793_79332

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / (Real.cos (3 * π / 2 - θ) - Real.sin (π - θ)) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l793_79332


namespace NUMINAMATH_GPT_joan_sandwiches_l793_79320

theorem joan_sandwiches :
  ∀ (H : ℕ), (∀ (h_slice g_slice total_cheese num_grilled_cheese : ℕ),
  h_slice = 2 →
  g_slice = 3 →
  num_grilled_cheese = 10 →
  total_cheese = 50 →
  total_cheese - num_grilled_cheese * g_slice = H * h_slice →
  H = 10) :=
by
  intros H h_slice g_slice total_cheese num_grilled_cheese h_slice_eq g_slice_eq num_grilled_cheese_eq total_cheese_eq cheese_eq
  sorry

end NUMINAMATH_GPT_joan_sandwiches_l793_79320


namespace NUMINAMATH_GPT_total_value_of_button_collection_l793_79323

theorem total_value_of_button_collection:
  (∀ (n : ℕ) (v : ℕ), n = 2 → v = 8 → has_same_value → total_value = 10 * (v / n)) →
  has_same_value :=
  sorry

end NUMINAMATH_GPT_total_value_of_button_collection_l793_79323


namespace NUMINAMATH_GPT_ratio_c_d_l793_79318

theorem ratio_c_d (x y c d : ℝ) (h1 : 4 * x + 5 * y = c) (h2 : 8 * y - 10 * x = d) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) : c / d = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_c_d_l793_79318


namespace NUMINAMATH_GPT_find_number_l793_79364

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 15) : x = 7.5 :=
sorry

end NUMINAMATH_GPT_find_number_l793_79364


namespace NUMINAMATH_GPT_time_difference_in_minutes_l793_79372

def speed := 60 -- speed of the car in miles per hour
def distance1 := 360 -- distance of the first trip in miles
def distance2 := 420 -- distance of the second trip in miles
def hours_to_minutes := 60 -- conversion factor from hours to minutes

theorem time_difference_in_minutes :
  ((distance2 / speed) - (distance1 / speed)) * hours_to_minutes = 60 :=
by
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_time_difference_in_minutes_l793_79372


namespace NUMINAMATH_GPT_total_sign_up_methods_l793_79312

theorem total_sign_up_methods (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 2) :
  k ^ n = 16 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_total_sign_up_methods_l793_79312


namespace NUMINAMATH_GPT_probability_odd_product_lt_one_eighth_l793_79349

theorem probability_odd_product_lt_one_eighth :
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  p < 1 / 8 :=
by
  let N := 2020
  let num_odds := N / 2
  let p := (num_odds / N) * ((num_odds - 1) / (N - 1)) * ((num_odds - 2) / (N - 2))
  sorry

end NUMINAMATH_GPT_probability_odd_product_lt_one_eighth_l793_79349


namespace NUMINAMATH_GPT_winning_percentage_l793_79346

-- Defining the conditions
def election_conditions (winner_votes : ℕ) (win_margin : ℕ) (total_candidates : ℕ) : Prop :=
  total_candidates = 2 ∧ winner_votes = 864 ∧ win_margin = 288

-- Stating the question: What percentage of votes did the winner candidate receive?
theorem winning_percentage (V : ℕ) (winner_votes : ℕ) (win_margin : ℕ) (total_candidates : ℕ) :
  election_conditions winner_votes win_margin total_candidates → (winner_votes * 100 / V) = 60 :=
by
  sorry

end NUMINAMATH_GPT_winning_percentage_l793_79346


namespace NUMINAMATH_GPT_totalExerciseTime_l793_79342

-- Define the conditions
def caloriesBurnedRunningPerMinute := 10
def caloriesBurnedWalkingPerMinute := 4
def totalCaloriesBurned := 450
def runningTime := 35

-- Define the problem as a theorem to be proven
theorem totalExerciseTime :
  ((runningTime * caloriesBurnedRunningPerMinute) + 
  ((totalCaloriesBurned - runningTime * caloriesBurnedRunningPerMinute) / caloriesBurnedWalkingPerMinute)) = 60 := 
sorry

end NUMINAMATH_GPT_totalExerciseTime_l793_79342


namespace NUMINAMATH_GPT_hexagonalPrismCannotIntersectAsCircle_l793_79352

-- Define each geometric shape as a type
inductive GeometricShape
| Sphere
| Cone
| Cylinder
| HexagonalPrism

-- Define a function that checks if a shape can be intersected by a plane to form a circular cross-section
def canIntersectAsCircle (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => True -- Sphere can always form a circular cross-section
  | GeometricShape.Cone => True -- Cone can form a circular cross-section if the plane is parallel to the base
  | GeometricShape.Cylinder => True -- Cylinder can form a circular cross-section if the plane is parallel to the base
  | GeometricShape.HexagonalPrism => False -- Hexagonal Prism cannot form a circular cross-section

-- The theorem to prove
theorem hexagonalPrismCannotIntersectAsCircle :
  ∀ shape : GeometricShape,
  (shape = GeometricShape.HexagonalPrism) ↔ ¬ canIntersectAsCircle shape := by
  sorry

end NUMINAMATH_GPT_hexagonalPrismCannotIntersectAsCircle_l793_79352


namespace NUMINAMATH_GPT_sqrt_180_simplified_l793_79326

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 :=
   sorry

end NUMINAMATH_GPT_sqrt_180_simplified_l793_79326


namespace NUMINAMATH_GPT_age_of_older_teenager_l793_79324

theorem age_of_older_teenager
  (a b : ℕ) 
  (h1 : a^2 - b^2 = 4 * (a + b)) 
  (h2 : a + b = 8 * (a - b)) 
  (h3 : a > b) : 
  a = 18 :=
sorry

end NUMINAMATH_GPT_age_of_older_teenager_l793_79324


namespace NUMINAMATH_GPT_sum_of_first_and_fourth_l793_79310

theorem sum_of_first_and_fourth (x : ℤ) (h : x + (x + 6) = 156) : (x + 2) = 77 :=
by {
  -- This block represents the assumptions and goal as expressed above,
  -- but the proof steps are omitted.
  sorry
}

end NUMINAMATH_GPT_sum_of_first_and_fourth_l793_79310


namespace NUMINAMATH_GPT_find_a_l793_79309

variable (a : ℝ)

def average_condition (a : ℝ) : Prop :=
  ((2 * a + 16) + (3 * a - 8)) / 2 = 74

theorem find_a (h: average_condition a) : a = 28 :=
  sorry

end NUMINAMATH_GPT_find_a_l793_79309


namespace NUMINAMATH_GPT_last_digit_of_large_exponentiation_l793_79392

theorem last_digit_of_large_exponentiation
  (a : ℕ) (b : ℕ)
  (h1 : a = 954950230952380948328708) 
  (h2 : b = 470128749397540235934750230) :
  (a ^ b) % 10 = 4 :=
sorry

end NUMINAMATH_GPT_last_digit_of_large_exponentiation_l793_79392


namespace NUMINAMATH_GPT_handshakes_in_octagonal_shape_l793_79316

-- Definitions
def number_of_students : ℕ := 8

def non_adjacent_handshakes_per_student : ℕ := number_of_students - 1 - 2

def total_handshakes : ℕ := (number_of_students * non_adjacent_handshakes_per_student) / 2

-- Theorem to prove
theorem handshakes_in_octagonal_shape : total_handshakes = 20 := 
by
  -- Provide the proof here.
  sorry

end NUMINAMATH_GPT_handshakes_in_octagonal_shape_l793_79316


namespace NUMINAMATH_GPT_difference_before_exchange_l793_79387

--Definitions
variables {S B : ℤ}

-- Conditions
axiom h1 : S - 2 = B + 2
axiom h2 : B > S

theorem difference_before_exchange : B - S = 2 :=
by
-- Proof will go here
sorry

end NUMINAMATH_GPT_difference_before_exchange_l793_79387


namespace NUMINAMATH_GPT_sales_on_second_day_l793_79306

variable (m : ℕ)

-- Define the condition for sales on the first day
def first_day_sales : ℕ := m

-- Define the condition for sales on the second day
def second_day_sales : ℕ := 2 * first_day_sales m - 3

-- The proof statement
theorem sales_on_second_day (m : ℕ) : second_day_sales m = 2 * m - 3 := by
  -- provide the actual proof here
  sorry

end NUMINAMATH_GPT_sales_on_second_day_l793_79306


namespace NUMINAMATH_GPT_impossible_to_repaint_white_l793_79384

-- Define the board as a 7x7 grid 
def boardSize : ℕ := 7

-- Define the initial coloring function (checkerboard with corners black)
def initialColor (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

-- Define the repainting operation allowed
def repaint (cell1 cell2 : (ℕ × ℕ)) (color1 color2 : Prop) : Prop :=
  ¬color1 = color2 

-- Define the main theorem to prove
theorem impossible_to_repaint_white :
  ¬(∃ f : ℕ × ℕ -> Prop, 
    (∀ i j, (i < boardSize) → (j < boardSize) → (f (i, j) = true)) ∧ 
    (∀ i j, (i < boardSize - 1) → (repaint (i, j) (i, j+1) (f (i, j)) (f (i, j+1))) ∧
             (i < boardSize - 1) → (repaint (i, j) (i+1, j) (f (i, j)) (f (i+1, j)))))
  :=
  sorry

end NUMINAMATH_GPT_impossible_to_repaint_white_l793_79384


namespace NUMINAMATH_GPT_digits_solution_exists_l793_79354

theorem digits_solution_exists (a b : ℕ) (ha : a < 10) (hb : b < 10) 
  (h : a = (b * (10 * b)) / (10 - b)) : a = 5 ∧ b = 2 :=
by
  sorry

end NUMINAMATH_GPT_digits_solution_exists_l793_79354


namespace NUMINAMATH_GPT_guesthouse_rolls_probability_l793_79304

theorem guesthouse_rolls_probability :
  let rolls := 12
  let guests := 3
  let types := 4
  let rolls_per_guest := 3
  let total_probability : ℚ := (12 / 12) * (9 / 11) * (6 / 10) * (3 / 9) *
                               (8 / 8) * (6 / 7) * (4 / 6) * (2 / 5) *
                               1
  let simplified_probability : ℚ := 24 / 1925
  total_probability = simplified_probability := sorry

end NUMINAMATH_GPT_guesthouse_rolls_probability_l793_79304


namespace NUMINAMATH_GPT_overtime_pay_rate_increase_l793_79386

theorem overtime_pay_rate_increase
  (regular_rate : ℝ)
  (total_compensation : ℝ)
  (total_hours : ℝ)
  (overtime_hours : ℝ)
  (expected_percentage_increase : ℝ)
  (h1 : regular_rate = 16)
  (h2 : total_hours = 48)
  (h3 : total_compensation = 864)
  (h4 : overtime_hours = total_hours - 40)
  (h5 : 40 * regular_rate + overtime_hours * (regular_rate + regular_rate * expected_percentage_increase / 100) = total_compensation) :
  expected_percentage_increase = 75 := 
by
  sorry

end NUMINAMATH_GPT_overtime_pay_rate_increase_l793_79386


namespace NUMINAMATH_GPT_range_of_values_l793_79385

theorem range_of_values (x y : ℝ) (h : (x + 2)^2 + y^2 / 4 = 1) :
  ∃ (a b : ℝ), a = 1 ∧ b = 28 / 3 ∧ a ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ b := by
  sorry

end NUMINAMATH_GPT_range_of_values_l793_79385


namespace NUMINAMATH_GPT_groupB_avg_weight_eq_141_l793_79391

def initial_group_weight (avg_weight : ℝ) : ℝ := 50 * avg_weight
def groupA_weight_gain : ℝ := 20 * 15
def groupB_weight_gain (x : ℝ) : ℝ := 20 * x

def total_weight (avg_weight : ℝ) (x : ℝ) : ℝ :=
  initial_group_weight avg_weight + groupA_weight_gain + groupB_weight_gain x

def total_avg_weight : ℝ := 46
def num_friends : ℝ := 90

def original_avg_weight : ℝ := total_avg_weight - 12
def final_total_weight : ℝ := num_friends * total_avg_weight

theorem groupB_avg_weight_eq_141 : 
  ∀ (avg_weight : ℝ) (x : ℝ),
    avg_weight = original_avg_weight →
    initial_group_weight avg_weight + groupA_weight_gain + groupB_weight_gain x = final_total_weight →
    avg_weight + x = 141 :=
by 
  intros avg_weight x h₁ h₂
  sorry

end NUMINAMATH_GPT_groupB_avg_weight_eq_141_l793_79391


namespace NUMINAMATH_GPT_part1_part2_l793_79373

-- Definitions of y1 and y2 based on given conditions
def y1 (x : ℝ) : ℝ := -x + 3
def y2 (x : ℝ) : ℝ := 2 + x

-- Prove for x such that y1 = y2
theorem part1 (x : ℝ) : y1 x = y2 x ↔ x = 1 / 2 := by
  sorry

-- Prove for x such that y1 = 2y2 + 5
theorem part2 (x : ℝ) : y1 x = 2 * y2 x + 5 ↔ x = -2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l793_79373


namespace NUMINAMATH_GPT_multiplicative_inverse_CD_mod_1000000_l793_79311

theorem multiplicative_inverse_CD_mod_1000000 :
  let C := 123456
  let D := 166666
  let M := 48
  M * (C * D) % 1000000 = 1 := by
  sorry

end NUMINAMATH_GPT_multiplicative_inverse_CD_mod_1000000_l793_79311


namespace NUMINAMATH_GPT_birds_nest_building_area_scientific_notation_l793_79355

theorem birds_nest_building_area_scientific_notation :
  (258000 : ℝ) = 2.58 * 10^5 :=
by sorry

end NUMINAMATH_GPT_birds_nest_building_area_scientific_notation_l793_79355


namespace NUMINAMATH_GPT_father_l793_79317

-- Let s be the circumference of the circular rink.
-- Let x be the son's speed.
-- Let k be the factor by which the father's speed is greater than the son's speed.

-- Define a theorem to state that k = 3/2.
theorem father's_speed_is_3_over_2_times_son's_speed
  (s x : ℝ) (k : ℝ) (h : s / (k * x - x) = (s / (k * x + x)) * 5) :
  k = 3 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_father_l793_79317


namespace NUMINAMATH_GPT_lollipops_Lou_received_l793_79383

def initial_lollipops : ℕ := 42
def given_to_Emily : ℕ := 2 * initial_lollipops / 3
def kept_by_Marlon : ℕ := 4
def lollipops_left_after_Emily : ℕ := initial_lollipops - given_to_Emily
def lollipops_given_to_Lou : ℕ := lollipops_left_after_Emily - kept_by_Marlon

theorem lollipops_Lou_received : lollipops_given_to_Lou = 10 := by
  sorry

end NUMINAMATH_GPT_lollipops_Lou_received_l793_79383


namespace NUMINAMATH_GPT_combined_loading_time_l793_79368

theorem combined_loading_time (rA rB rC : ℝ) (hA : rA = 1 / 6) (hB : rB = 1 / 8) (hC : rC = 1 / 10) :
  1 / (rA + rB + rC) = 120 / 47 := by
  sorry

end NUMINAMATH_GPT_combined_loading_time_l793_79368


namespace NUMINAMATH_GPT_maximum_temperature_difference_l793_79360

theorem maximum_temperature_difference
  (highest_temp : ℝ) (lowest_temp : ℝ)
  (h_highest : highest_temp = 58)
  (h_lowest : lowest_temp = -34) :
  highest_temp - lowest_temp = 92 :=
by sorry

end NUMINAMATH_GPT_maximum_temperature_difference_l793_79360


namespace NUMINAMATH_GPT_convert_angle_l793_79397

theorem convert_angle (α : ℝ) (k : ℤ) :
  -1485 * (π / 180) = α + 2 * k * π ∧ 0 ≤ α ∧ α < 2 * π ∧ k = -10 ∧ α = 7 * π / 4 :=
by
  sorry

end NUMINAMATH_GPT_convert_angle_l793_79397


namespace NUMINAMATH_GPT_stamps_problem_l793_79301

def largest_common_divisor (a b c : ℕ) : ℕ :=
  gcd (gcd a b) c

theorem stamps_problem :
  largest_common_divisor 1020 1275 1350 = 15 :=
by
  sorry

end NUMINAMATH_GPT_stamps_problem_l793_79301


namespace NUMINAMATH_GPT_arithmetic_square_root_l793_79353

noncomputable def cube_root (x : ℝ) : ℝ :=
  x^(1/3)

noncomputable def sqrt_int_part (x : ℝ) : ℤ :=
  ⌊Real.sqrt x⌋

theorem arithmetic_square_root 
  (a : ℝ) (b : ℤ) (c : ℝ) 
  (h1 : cube_root a = 2) 
  (h2 : b = sqrt_int_part 5) 
  (h3 : c = 4 ∨ c = -4) : 
  Real.sqrt (a + ↑b + c) = Real.sqrt 14 ∨ Real.sqrt (a + ↑b + c) = Real.sqrt 6 := 
sorry

end NUMINAMATH_GPT_arithmetic_square_root_l793_79353


namespace NUMINAMATH_GPT_max_sides_of_convex_polygon_with_4_obtuse_l793_79307

theorem max_sides_of_convex_polygon_with_4_obtuse (n : ℕ) (hn : n ≥ 3) :
  (∃ k : ℕ, k = 4 ∧
    ∀ θ : Fin n → ℝ, 
      (∀ p, θ p > 90 ∧ ∃ t, θ t = 180 ∨ θ t < 90 ∨ θ t = 90) →
      4 = k →
      n ≤ 7
  ) :=
sorry

end NUMINAMATH_GPT_max_sides_of_convex_polygon_with_4_obtuse_l793_79307


namespace NUMINAMATH_GPT_inequality_abc_l793_79357

theorem inequality_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
sorry

end NUMINAMATH_GPT_inequality_abc_l793_79357


namespace NUMINAMATH_GPT_algebra_expression_value_l793_79303

theorem algebra_expression_value
  (x y : ℝ)
  (h : x - 2 * y + 2 = 5) : 4 * y - 2 * x + 1 = -5 :=
by sorry

end NUMINAMATH_GPT_algebra_expression_value_l793_79303


namespace NUMINAMATH_GPT_solve_for_x_l793_79362

theorem solve_for_x : ∀ x : ℝ, (x - 3) ≠ 0 → (x + 6) / (x - 3) = 4 → x = 6 :=
by 
  intros x hx h
  sorry

end NUMINAMATH_GPT_solve_for_x_l793_79362


namespace NUMINAMATH_GPT_reassemble_into_square_conditions_l793_79313

noncomputable def graph_paper_figure : Type := sorry
noncomputable def is_cuttable_into_parts (figure : graph_paper_figure) (parts : ℕ) : Prop := sorry
noncomputable def all_parts_are_triangles (figure : graph_paper_figure) (parts : ℕ) : Prop := sorry
noncomputable def can_reassemble_to_square (figure : graph_paper_figure) : Prop := sorry

theorem reassemble_into_square_conditions :
  ∀ (figure : graph_paper_figure), 
  (is_cuttable_into_parts figure 4 ∧ can_reassemble_to_square figure) ∧ 
  (is_cuttable_into_parts figure 5 ∧ all_parts_are_triangles figure 5 ∧ can_reassemble_to_square figure) :=
sorry

end NUMINAMATH_GPT_reassemble_into_square_conditions_l793_79313


namespace NUMINAMATH_GPT_baker_work_alone_time_l793_79327

theorem baker_work_alone_time 
  (rate_baker_alone : ℕ) 
  (rate_baker_with_helper : ℕ) 
  (total_time : ℕ) 
  (total_flour : ℕ)
  (time_with_helper : ℕ)
  (flour_used_baker_alone_time : ℕ)
  (flour_used_with_helper_time : ℕ)
  (total_flour_used : ℕ) 
  (h1 : rate_baker_alone = total_flour / 6) 
  (h2 : rate_baker_with_helper = total_flour / 2) 
  (h3 : total_time = 150)
  (h4 : flour_used_baker_alone_time = total_flour * flour_used_baker_alone_time / 6)
  (h5 : flour_used_with_helper_time = total_flour * (total_time - flour_used_baker_alone_time) / 2)
  (h6 : total_flour_used = total_flour) :
  flour_used_baker_alone_time = 45 :=
by
  sorry

end NUMINAMATH_GPT_baker_work_alone_time_l793_79327


namespace NUMINAMATH_GPT_t_shirts_left_yesterday_correct_l793_79367

-- Define the conditions
def t_shirts_left_yesterday (x : ℕ) : Prop :=
  let t_shirts_sold_morning := (3 / 5) * x
  let t_shirts_sold_afternoon := 180
  t_shirts_sold_morning = t_shirts_sold_afternoon

-- Prove that x = 300 given the above conditions
theorem t_shirts_left_yesterday_correct (x : ℕ) (h : t_shirts_left_yesterday x) : x = 300 :=
by
  sorry

end NUMINAMATH_GPT_t_shirts_left_yesterday_correct_l793_79367


namespace NUMINAMATH_GPT_largest_integer_satisfying_inequality_l793_79345

theorem largest_integer_satisfying_inequality :
  ∃ n : ℤ, n = 4 ∧ (1 / 4 + n / 8 < 7 / 8) ∧ ∀ m : ℤ, m > 4 → ¬(1 / 4 + m / 8 < 7 / 8) :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_satisfying_inequality_l793_79345


namespace NUMINAMATH_GPT_jane_stopped_babysitting_l793_79344

noncomputable def stopped_babysitting_years_ago := 12

-- Definitions for the problem conditions
def jane_age_started_babysitting := 20
def jane_current_age := 32
def oldest_child_current_age := 22

-- Final statement to prove the equivalence
theorem jane_stopped_babysitting : 
    ∃ (x : ℕ), 
    (jane_current_age - x = stopped_babysitting_years_ago) ∧
    (oldest_child_current_age - x ≤ 1/2 * (jane_current_age - x)) := 
sorry

end NUMINAMATH_GPT_jane_stopped_babysitting_l793_79344


namespace NUMINAMATH_GPT_clean_room_time_l793_79350

theorem clean_room_time :
  let lisa_time := 8
  let kay_time := 12
  let ben_time := 16
  let combined_work_rate := (1 / lisa_time) + (1 / kay_time) + (1 / ben_time)
  let total_time := 1 / combined_work_rate
  total_time = 48 / 13 :=
by
  sorry

end NUMINAMATH_GPT_clean_room_time_l793_79350


namespace NUMINAMATH_GPT_polynomial_division_l793_79363

open Polynomial

theorem polynomial_division (a b : ℤ) (h : a^2 ≥ 4*b) :
  ∀ n : ℕ, ∃ (k l : ℤ), (x^2 + (C a) * x + (C b)) ∣ (x^2) * (x^2) ^ n + (C a) * x ^ n + (C b) ↔ 
    ((a = -2 ∧ b = 1) ∨ (a = 2 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) :=
sorry

end NUMINAMATH_GPT_polynomial_division_l793_79363


namespace NUMINAMATH_GPT_print_time_including_warmup_l793_79334

def warmUpTime : ℕ := 2
def pagesPerMinute : ℕ := 15
def totalPages : ℕ := 225

theorem print_time_including_warmup :
  (totalPages / pagesPerMinute) + warmUpTime = 17 := by
  sorry

end NUMINAMATH_GPT_print_time_including_warmup_l793_79334


namespace NUMINAMATH_GPT_almonds_addition_l793_79399

theorem almonds_addition (walnuts almonds total_nuts : ℝ) 
  (h_walnuts : walnuts = 0.25) 
  (h_total_nuts : total_nuts = 0.5)
  (h_sum : total_nuts = walnuts + almonds) : 
  almonds = 0.25 := by
  sorry

end NUMINAMATH_GPT_almonds_addition_l793_79399


namespace NUMINAMATH_GPT_calculate_f_f_f_l793_79376

def f (x : ℤ) : ℤ := 3 * x + 2

theorem calculate_f_f_f :
  f (f (f 3)) = 107 :=
by
  sorry

end NUMINAMATH_GPT_calculate_f_f_f_l793_79376


namespace NUMINAMATH_GPT_find_triplet_l793_79390

def ordered_triplet : Prop :=
  ∃ (x y z : ℚ), 
  7 * x + 3 * y = z - 10 ∧ 
  2 * x - 4 * y = 3 * z + 20 ∧ 
  x = 0 ∧ 
  y = -50 / 13 ∧ 
  z = -20 / 13

theorem find_triplet : ordered_triplet := 
  sorry

end NUMINAMATH_GPT_find_triplet_l793_79390


namespace NUMINAMATH_GPT_cost_of_one_bag_l793_79315

theorem cost_of_one_bag (x : ℝ) (h1 : ∀ p : ℝ, 60 * x = p -> 60 * p = 120 * x ) 
  (h2 : ∀ p1 p2: ℝ, 60 * x = p1 ∧ 15 * 1.6 * x = p2 ∧ 45 * 2.24 * x = 100.8 * x -> 124.8 * x - 120 * x = 1200) :
  x = 250 := 
sorry

end NUMINAMATH_GPT_cost_of_one_bag_l793_79315


namespace NUMINAMATH_GPT_solution_set_of_inequality_l793_79331

theorem solution_set_of_inequality :
  {x : ℝ | 4*x^2 - 9*x > 5} = {x : ℝ | x < -1/4} ∪ {x : ℝ | x > 5} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l793_79331


namespace NUMINAMATH_GPT_sum_weights_second_fourth_l793_79381

-- Definitions based on given conditions
noncomputable section

def weight (n : ℕ) : ℕ := 4 - (n - 1)

-- Assumption that weights form an arithmetic sequence.
-- 1st foot weighs 4 jin, 5th foot weighs 2 jin, and weights are linearly decreasing.
axiom weight_arith_seq (n : ℕ) : weight n = 4 - (n - 1)

-- Prove the sum of the weights of the second and fourth feet
theorem sum_weights_second_fourth :
  weight 2 + weight 4 = 6 :=
by
  simp [weight_arith_seq]
  sorry

end NUMINAMATH_GPT_sum_weights_second_fourth_l793_79381


namespace NUMINAMATH_GPT_son_age_l793_79308

theorem son_age {S M : ℕ} 
  (h1 : M = S + 37)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 35 :=
by sorry

end NUMINAMATH_GPT_son_age_l793_79308


namespace NUMINAMATH_GPT_train_seats_count_l793_79365

theorem train_seats_count 
  (Standard Comfort Premium : ℝ)
  (Total_SEATS : ℝ)
  (hs : Standard = 36)
  (hc : Comfort = 0.20 * Total_SEATS)
  (hp : Premium = (3/5) * Total_SEATS)
  (ht : Standard + Comfort + Premium = Total_SEATS) :
  Total_SEATS = 180 := sorry

end NUMINAMATH_GPT_train_seats_count_l793_79365


namespace NUMINAMATH_GPT_eliminate_denominators_eq_l793_79379

theorem eliminate_denominators_eq :
  ∀ (x : ℝ), 1 - (x + 3) / 6 = x / 2 → 6 - x - 3 = 3 * x :=
by
  intro x
  intro h
  -- Place proof steps here.
  sorry

end NUMINAMATH_GPT_eliminate_denominators_eq_l793_79379


namespace NUMINAMATH_GPT_incorrect_option_c_l793_79394

theorem incorrect_option_c (R : ℝ) : 
  let cylinder_lateral_area := 4 * π * R^2
  let sphere_surface_area := 4 * π * R^2
  cylinder_lateral_area = sphere_surface_area :=
  sorry

end NUMINAMATH_GPT_incorrect_option_c_l793_79394


namespace NUMINAMATH_GPT_special_operation_value_l793_79340

def special_operation (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : ℚ := (1 / a : ℚ) + (1 / b : ℚ)

theorem special_operation_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : a + b = 15) (h₃ : a * b = 36) : special_operation a b h₀ h₁ = 5 / 12 :=
by sorry

end NUMINAMATH_GPT_special_operation_value_l793_79340


namespace NUMINAMATH_GPT_thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one_l793_79366

theorem thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one : 37 * 23 = 851 := by
  sorry

end NUMINAMATH_GPT_thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one_l793_79366


namespace NUMINAMATH_GPT_divides_expression_l793_79329

theorem divides_expression (x : ℕ) (hx : Even x) : 90 ∣ (15 * x + 3) * (15 * x + 9) * (5 * x + 10) :=
sorry

end NUMINAMATH_GPT_divides_expression_l793_79329


namespace NUMINAMATH_GPT_distinct_digits_unique_D_l793_79377

theorem distinct_digits_unique_D 
  (A B C D : ℕ)
  (hA : A ≠ B)
  (hB : B ≠ C)
  (hC : C ≠ D)
  (hD : D ≠ A)
  (h1 : D < 10)
  (h2 : B < 10)
  (h3 : C < 10)
  (h4 : A < 10)
  (h_add : A * 1000 + A * 100 + C * 10 + B + B * 1000 + C * 100 + B * 10 + D = B * 1000 + D * 100 + A * 10 + B) :
  D = 0 :=
by sorry

end NUMINAMATH_GPT_distinct_digits_unique_D_l793_79377


namespace NUMINAMATH_GPT_min_value_at_1_l793_79370

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2 * a * x + 8 else x + 4 / x + 2 * a

theorem min_value_at_1 (a : ℝ) :
  (∀ x, f x a ≥ f 1 a) ↔ (a = 5/4 ∨ a = 2 ∨ a = 4) :=
by
  sorry

end NUMINAMATH_GPT_min_value_at_1_l793_79370


namespace NUMINAMATH_GPT_range_of_a_l793_79337

variable (a : ℝ)

theorem range_of_a (h : ∀ x : ℤ, 2 * (x:ℝ)^2 - 17 * x + a ≤ 0 →  (x = 3 ∨ x = 4 ∨ x = 5)) : 
  30 < a ∧ a ≤ 33 :=
sorry

end NUMINAMATH_GPT_range_of_a_l793_79337


namespace NUMINAMATH_GPT_cost_of_bricks_l793_79347

theorem cost_of_bricks
  (N: ℕ)
  (half_bricks:ℕ)
  (full_price: ℝ)
  (discount_percentage: ℝ)
  (n_half: half_bricks = N / 2)
  (P1: full_price = 0.5)
  (P2: discount_percentage = 0.5):
  (half_bricks * (full_price * discount_percentage) + 
  half_bricks * full_price = 375) := 
by sorry

end NUMINAMATH_GPT_cost_of_bricks_l793_79347


namespace NUMINAMATH_GPT_cows_problem_l793_79328

theorem cows_problem :
  ∃ (M X : ℕ), 
  (5 * M = X + 30) ∧ 
  (5 * M + X = 570) ∧ 
  M = 60 :=
by
  sorry

end NUMINAMATH_GPT_cows_problem_l793_79328


namespace NUMINAMATH_GPT_fraction_of_robs_doubles_is_one_third_l793_79358

theorem fraction_of_robs_doubles_is_one_third 
  (total_robs_cards : ℕ) (total_jess_doubles : ℕ) 
  (times_jess_doubles_robs : ℕ)
  (robs_doubles : ℕ) :
  total_robs_cards = 24 →
  total_jess_doubles = 40 →
  times_jess_doubles_robs = 5 →
  total_jess_doubles = times_jess_doubles_robs * robs_doubles →
  (robs_doubles : ℚ) / total_robs_cards = 1 / 3 := 
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_fraction_of_robs_doubles_is_one_third_l793_79358


namespace NUMINAMATH_GPT_find_added_number_l793_79359

variable (x : ℝ) -- We define the variable x as a real number
-- We define the given conditions

def added_number (y : ℝ) : Prop :=
  (2 * (62.5 + y) / 5) - 5 = 22

theorem find_added_number : added_number x → x = 5 := by
  sorry

end NUMINAMATH_GPT_find_added_number_l793_79359


namespace NUMINAMATH_GPT_painters_work_days_l793_79375

noncomputable def work_product (n : ℕ) (d : ℚ) : ℚ := n * d

theorem painters_work_days :
  (work_product 5 2 = work_product 4 (2 + 1/2)) :=
by
  sorry

end NUMINAMATH_GPT_painters_work_days_l793_79375


namespace NUMINAMATH_GPT_solve_trigonometric_inequality_l793_79374

noncomputable def trigonometric_inequality (x : ℝ) : Prop :=
  x ∈ Set.Ioo 0 (2 * Real.pi) ∧ 2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0

theorem solve_trigonometric_inequality :
  ∀ x, x ∈ Set.Ioo 0 (2 * Real.pi) → (2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0 ↔ x ∈ Set.Icc (Real.pi / 3) (2 * Real.pi / 3)) :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_solve_trigonometric_inequality_l793_79374


namespace NUMINAMATH_GPT_solution_set_empty_for_k_l793_79335

theorem solution_set_empty_for_k (k : ℝ) :
  (∀ x : ℝ, ¬ (kx^2 - 2 * |x - 1| + 3 * k < 0)) ↔ (1 ≤ k) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_empty_for_k_l793_79335


namespace NUMINAMATH_GPT_cans_ounces_per_day_l793_79339

-- Definitions of the conditions
def daily_soda_cans : ℕ := 5
def daily_water_ounces : ℕ := 64
def weekly_fluid_ounces : ℕ := 868

-- Theorem statement proving the number of ounces per can of soda
theorem cans_ounces_per_day (h_soda_daily : daily_soda_cans * 7 = 35)
    (h_weekly_soda : weekly_fluid_ounces - daily_water_ounces * 7 = 420) 
    (h_total_weekly : 35 = ((daily_soda_cans * 7))):
  420 / 35 = 12 := by
  sorry

end NUMINAMATH_GPT_cans_ounces_per_day_l793_79339


namespace NUMINAMATH_GPT_noah_has_largest_final_answer_l793_79396

def liam_initial := 15
def liam_final := (liam_initial - 2) * 3 + 3

def mia_initial := 15
def mia_final := (mia_initial * 3 - 4) + 3

def noah_initial := 15
def noah_final := ((noah_initial - 3) + 4) * 3

theorem noah_has_largest_final_answer : noah_final > liam_final ∧ noah_final > mia_final := by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_noah_has_largest_final_answer_l793_79396


namespace NUMINAMATH_GPT_problem_ineq_l793_79305

theorem problem_ineq (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
(h4 : x * y * z = 1) :
    (x^3 / ((1 + y)*(1 + z)) + y^3 / ((1 + z)*(1 + x)) + z^3 / ((1 + x)*(1 + y))) ≥ 3 / 4 := 
sorry

end NUMINAMATH_GPT_problem_ineq_l793_79305


namespace NUMINAMATH_GPT_clothing_percentage_l793_79341

variable (T : ℝ) -- Total amount excluding taxes.
variable (C : ℝ) -- Percentage of total amount spent on clothing.

-- Conditions
def spent_on_food := 0.2 * T
def spent_on_other_items := 0.3 * T

-- Taxes
def tax_on_clothing := 0.04 * (C * T)
def tax_on_food := 0.0
def tax_on_other_items := 0.08 * (0.3 * T)
def total_tax_paid := 0.044 * T

-- Statement to prove
theorem clothing_percentage : 
  0.04 * (C * T) + 0.08 * (0.3 * T) = 0.044 * T ↔ C = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_clothing_percentage_l793_79341
