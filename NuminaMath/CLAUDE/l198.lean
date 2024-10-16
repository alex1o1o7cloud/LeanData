import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_lines_sum_l198_19898

/-- Two lines are symmetric with respect to the x-axis if their slopes are opposite
    and their y-intercepts are negatives of each other -/
def symmetric_lines (l₁ l₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ (m₁ m₂ b₁ b₂ : ℝ), 
    (∀ x y, l₁ x y ↔ y = m₁ * x + b₁) ∧
    (∀ x y, l₂ x y ↔ y = m₂ * x + b₂) ∧
    m₂ = -m₁ ∧ b₂ = -b₁

/-- The first line l₁ -/
def l₁ (x y : ℝ) : Prop := x - 3 * y + 2 = 0

/-- The second line l₂ -/
def l₂ (m b : ℝ) (x y : ℝ) : Prop := m * x - y + b = 0

theorem symmetric_lines_sum (m b : ℝ) :
  symmetric_lines l₁ (l₂ m b) → m + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_lines_sum_l198_19898


namespace NUMINAMATH_CALUDE_handbag_profit_optimization_handbag_profit_constraint_l198_19893

/-- Represents the daily sales quantity as a function of price -/
def daily_sales (x : ℝ) : ℝ := -x + 80

/-- Represents the daily profit as a function of price -/
def daily_profit (x : ℝ) : ℝ := (x - 50) * (daily_sales x)

/-- The cost price of the handbag -/
def cost_price : ℝ := 50

/-- The lower bound of the selling price -/
def price_lower_bound : ℝ := 50

/-- The upper bound of the selling price -/
def price_upper_bound : ℝ := 80

theorem handbag_profit_optimization :
  ∃ (max_price max_profit : ℝ),
    (∀ x, price_lower_bound < x ∧ x < price_upper_bound → daily_profit x ≤ max_profit) ∧
    daily_profit max_price = max_profit ∧
    max_price = 65 ∧
    max_profit = 225 :=
sorry

theorem handbag_profit_constraint (target_profit : ℝ) (price_limit : ℝ) :
  target_profit = 200 →
  price_limit = 68 →
  ∃ (optimal_price : ℝ),
    optimal_price ≤ price_limit ∧
    daily_profit optimal_price = target_profit ∧
    optimal_price = 60 :=
sorry

end NUMINAMATH_CALUDE_handbag_profit_optimization_handbag_profit_constraint_l198_19893


namespace NUMINAMATH_CALUDE_seminar_room_chairs_l198_19815

/-- Converts a number from base 6 to decimal --/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Calculates the number of chairs needed, rounding up --/
def chairsNeeded (participants : Nat) (participantsPerChair : Nat) : Nat :=
  (participants + participantsPerChair - 1) / participantsPerChair

theorem seminar_room_chairs :
  let base6Capacity := [5, 1, 3]  -- 315 in base 6, least significant digit first
  let participantsPerChair := 3
  chairsNeeded (base6ToDecimal base6Capacity) participantsPerChair = 40 := by
  sorry

#eval chairsNeeded (base6ToDecimal [5, 1, 3]) 3  -- Should output 40

end NUMINAMATH_CALUDE_seminar_room_chairs_l198_19815


namespace NUMINAMATH_CALUDE_white_bellied_minnows_count_l198_19830

/-- Proves the number of white-bellied minnows in a pond given the percentages of red, green, and white-bellied minnows and the number of red-bellied minnows. -/
theorem white_bellied_minnows_count 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (red_count : ℕ) 
  (h_red_percent : red_percent = 40 / 100)
  (h_green_percent : green_percent = 30 / 100)
  (h_red_count : red_count = 20)
  : ∃ (total : ℕ) (white_count : ℕ),
    red_percent * total = red_count ∧
    (1 - red_percent - green_percent) * total = white_count ∧
    white_count = 15 := by
  sorry

#check white_bellied_minnows_count

end NUMINAMATH_CALUDE_white_bellied_minnows_count_l198_19830


namespace NUMINAMATH_CALUDE_custard_pies_sold_is_five_l198_19825

/-- Represents the bakery sales problem --/
structure BakerySales where
  pumpkin_slices_per_pie : ℕ
  custard_slices_per_pie : ℕ
  pumpkin_price_per_slice : ℕ
  custard_price_per_slice : ℕ
  pumpkin_pies_sold : ℕ
  total_revenue : ℕ

/-- Calculates the number of custard pies sold --/
def custard_pies_sold (bs : BakerySales) : ℕ :=
  sorry

/-- Theorem stating that the number of custard pies sold is 5 --/
theorem custard_pies_sold_is_five (bs : BakerySales)
  (h1 : bs.pumpkin_slices_per_pie = 8)
  (h2 : bs.custard_slices_per_pie = 6)
  (h3 : bs.pumpkin_price_per_slice = 5)
  (h4 : bs.custard_price_per_slice = 6)
  (h5 : bs.pumpkin_pies_sold = 4)
  (h6 : bs.total_revenue = 340) :
  custard_pies_sold bs = 5 :=
sorry

end NUMINAMATH_CALUDE_custard_pies_sold_is_five_l198_19825


namespace NUMINAMATH_CALUDE_bank_interest_equation_l198_19867

theorem bank_interest_equation (initial_deposit : ℝ) (interest_tax_rate : ℝ) 
  (total_amount : ℝ) (annual_interest_rate : ℝ) 
  (h1 : initial_deposit = 2500)
  (h2 : interest_tax_rate = 0.2)
  (h3 : total_amount = 2650) :
  initial_deposit * (1 + annual_interest_rate * (1 - interest_tax_rate)) = total_amount :=
by sorry

end NUMINAMATH_CALUDE_bank_interest_equation_l198_19867


namespace NUMINAMATH_CALUDE_problem_solution_l198_19883

theorem problem_solution (a b c d e : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |e| = 4) : 
  e^2 - (a + b)^2022 + (-c * d)^2021 = 15 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l198_19883


namespace NUMINAMATH_CALUDE_modulus_of_one_minus_i_times_one_plus_i_l198_19891

theorem modulus_of_one_minus_i_times_one_plus_i : 
  Complex.abs ((1 - Complex.I) * (1 + Complex.I)) = 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_one_minus_i_times_one_plus_i_l198_19891


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l198_19840

-- Problem 1
theorem problem_1 : (-10) - (-4) + 5 = -1 := by sorry

-- Problem 2
theorem problem_2 : (-72) * (2/3 - 1/4 - 5/6) = 30 := by sorry

-- Problem 3
theorem problem_3 : -3^2 - (-2)^3 * (-1)^4 + Real.rpow 27 (1/3) = 2 := by sorry

-- Problem 4
theorem problem_4 : 5 + 4 * (Real.sqrt 6 - 2) - 4 * (Real.sqrt 6 - 1) = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l198_19840


namespace NUMINAMATH_CALUDE_amount_to_hand_in_l198_19874

/-- Represents the contents of Jack's till -/
structure TillContents where
  usd_100: Nat
  usd_50: Nat
  usd_20: Nat
  usd_10: Nat
  usd_5: Nat
  usd_1: Nat
  quarters: Nat
  dimes: Nat
  nickels: Nat
  pennies: Nat
  euro_5: Nat
  gbp_10: Nat

/-- Exchange rates -/
def euro_to_usd : Rat := 118/100
def gbp_to_usd : Rat := 139/100

/-- The amount to be left in the till -/
def amount_to_leave : Rat := 300

/-- Calculate the total amount in USD -/
def total_amount (contents : TillContents) : Rat :=
  contents.usd_100 * 100 +
  contents.usd_50 * 50 +
  contents.usd_20 * 20 +
  contents.usd_10 * 10 +
  contents.usd_5 * 5 +
  contents.usd_1 +
  contents.quarters * (1/4) +
  contents.dimes * (1/10) +
  contents.nickels * (1/20) +
  contents.pennies * (1/100) +
  contents.euro_5 * 5 * euro_to_usd +
  contents.gbp_10 * 10 * gbp_to_usd

/-- Calculate the total amount of coins -/
def total_coins (contents : TillContents) : Rat :=
  contents.quarters * (1/4) +
  contents.dimes * (1/10) +
  contents.nickels * (1/20) +
  contents.pennies * (1/100)

/-- Jack's till contents -/
def jacks_till : TillContents := {
  usd_100 := 2,
  usd_50 := 1,
  usd_20 := 5,
  usd_10 := 3,
  usd_5 := 7,
  usd_1 := 27,
  quarters := 42,
  dimes := 19,
  nickels := 36,
  pennies := 47,
  euro_5 := 20,
  gbp_10 := 25
}

theorem amount_to_hand_in :
  total_amount jacks_till - (amount_to_leave + total_coins jacks_till) = 607.5 := by
  sorry

end NUMINAMATH_CALUDE_amount_to_hand_in_l198_19874


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l198_19856

theorem root_shift_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 - 9*x^2 + 22*x - 5 = 0 ↔ x = a + 3 ∨ x = b + 3 ∨ x = c + 3) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l198_19856


namespace NUMINAMATH_CALUDE_coin_flip_expected_earnings_l198_19838

/-- Represents the possible outcomes of the coin flip -/
inductive CoinOutcome
| A
| B
| C
| Disappear

/-- The probability of each outcome -/
def probability (outcome : CoinOutcome) : ℚ :=
  match outcome with
  | CoinOutcome.A => 1/4
  | CoinOutcome.B => 1/4
  | CoinOutcome.C => 1/3
  | CoinOutcome.Disappear => 1/6

/-- The payout for each outcome -/
def payout (outcome : CoinOutcome) : ℤ :=
  match outcome with
  | CoinOutcome.A => 2
  | CoinOutcome.B => -1
  | CoinOutcome.C => 4
  | CoinOutcome.Disappear => -3

/-- The expected earnings from flipping the coin -/
def expected_earnings : ℚ :=
  (probability CoinOutcome.A * payout CoinOutcome.A) +
  (probability CoinOutcome.B * payout CoinOutcome.B) +
  (probability CoinOutcome.C * payout CoinOutcome.C) +
  (probability CoinOutcome.Disappear * payout CoinOutcome.Disappear)

theorem coin_flip_expected_earnings :
  expected_earnings = 13/12 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_expected_earnings_l198_19838


namespace NUMINAMATH_CALUDE_problem_solution_l198_19866

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * 2^(x+1) + (k-3) * 2^(-x)

theorem problem_solution (k : ℝ) (t : ℝ) :
  (∀ x, f k (-x) = -(f k x)) →
  (∀ x ∈ Set.Icc 1 3, f k (x^2 - x) + f k (t*x + 4) > 0) →
  (k = 1 ∧
   (∀ x₁ x₂, x₁ < x₂ → f k x₁ < f k x₂) ∧
   t > -3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l198_19866


namespace NUMINAMATH_CALUDE_john_bought_three_reels_l198_19882

/-- The number of reels John bought -/
def num_reels : ℕ := sorry

/-- The length of fishing line in each reel (in meters) -/
def reel_length : ℕ := 100

/-- The length of each section after cutting (in meters) -/
def section_length : ℕ := 10

/-- The number of sections John got after cutting -/
def num_sections : ℕ := 30

/-- Theorem: John bought 3 reels of fishing line -/
theorem john_bought_three_reels :
  num_reels = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_john_bought_three_reels_l198_19882


namespace NUMINAMATH_CALUDE_min_value_T_l198_19805

theorem min_value_T (p : ℝ) (h1 : 0 < p) (h2 : p < 15) :
  ∃ (min_T : ℝ), min_T = 15 ∧
  ∀ x : ℝ, p ≤ x → x ≤ 15 →
    |x - p| + |x - 15| + |x - (15 + p)| ≥ min_T :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_T_l198_19805


namespace NUMINAMATH_CALUDE_correct_equation_after_digit_move_l198_19803

theorem correct_equation_after_digit_move : 101 - 10^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_after_digit_move_l198_19803


namespace NUMINAMATH_CALUDE_specific_pentagon_area_l198_19801

/-- Represents a pentagon that can be decomposed into two triangles and a trapezoid -/
structure DecomposablePentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  triangle1_area : ℝ
  triangle2_area : ℝ
  trapezoid_area : ℝ
  decomposable : side1 > 0 ∧ side2 > 0 ∧ side3 > 0 ∧ side4 > 0 ∧ side5 > 0

/-- Calculate the area of a decomposable pentagon -/
def area (p : DecomposablePentagon) : ℝ :=
  p.triangle1_area + p.triangle2_area + p.trapezoid_area

/-- Theorem stating that a specific pentagon has an area of 848 square units -/
theorem specific_pentagon_area :
  ∃ (p : DecomposablePentagon),
    p.side1 = 18 ∧ p.side2 = 22 ∧ p.side3 = 30 ∧ p.side4 = 26 ∧ p.side5 = 22 ∧
    area p = 848 := by
  sorry


end NUMINAMATH_CALUDE_specific_pentagon_area_l198_19801


namespace NUMINAMATH_CALUDE_pentagon_angle_sum_l198_19858

theorem pentagon_angle_sum (a b c d : ℝ) (h1 : a = 130) (h2 : b = 95) (h3 : c = 110) (h4 : d = 104) :
  ∃ q : ℝ, a + b + c + d + q = 540 ∧ q = 101 := by sorry

end NUMINAMATH_CALUDE_pentagon_angle_sum_l198_19858


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l198_19848

theorem regular_polygon_sides (central_angle : ℝ) (h : central_angle = 36) :
  (360 : ℝ) / central_angle = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l198_19848


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_two_l198_19807

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (m-1)x^2 + (m-2)x + (m^2 - 7m + 12) -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 + (m - 2) * x + (m^2 - 7*m + 12)

theorem even_function_implies_m_equals_two :
  ∀ m : ℝ, IsEven (f m) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_two_l198_19807


namespace NUMINAMATH_CALUDE_dana_weekend_earnings_l198_19826

def dana_earnings (hourly_rate : ℝ) (commission_rate : ℝ) 
  (friday_hours : ℝ) (friday_sales : ℝ)
  (saturday_hours : ℝ) (saturday_sales : ℝ)
  (sunday_hours : ℝ) (sunday_sales : ℝ) : ℝ :=
  let total_hours := friday_hours + saturday_hours + sunday_hours
  let total_sales := friday_sales + saturday_sales + sunday_sales
  let hourly_earnings := hourly_rate * total_hours
  let commission_earnings := commission_rate * total_sales
  hourly_earnings + commission_earnings

theorem dana_weekend_earnings :
  dana_earnings 13 0.05 9 800 10 1000 3 300 = 391 := by
  sorry

end NUMINAMATH_CALUDE_dana_weekend_earnings_l198_19826


namespace NUMINAMATH_CALUDE_bed_weight_difference_bed_weight_difference_proof_l198_19802

theorem bed_weight_difference : ℝ → ℝ → Prop :=
  fun single_bed_weight double_bed_weight =>
    (5 * single_bed_weight = 50) →
    (2 * single_bed_weight + 4 * double_bed_weight = 100) →
    (double_bed_weight - single_bed_weight = 10)

-- The proof is omitted
theorem bed_weight_difference_proof : ∃ (s d : ℝ), bed_weight_difference s d :=
  sorry

end NUMINAMATH_CALUDE_bed_weight_difference_bed_weight_difference_proof_l198_19802


namespace NUMINAMATH_CALUDE_solve_for_x_l198_19875

theorem solve_for_x (x y : ℝ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l198_19875


namespace NUMINAMATH_CALUDE_shoes_sold_shoes_sold_is_six_l198_19859

theorem shoes_sold (shoe_price : ℕ) (shirt_price : ℕ) (num_shirts : ℕ) (individual_earnings : ℕ) : ℕ :=
  let total_earnings := 2 * individual_earnings
  let shirt_earnings := shirt_price * num_shirts
  let shoe_earnings := total_earnings - shirt_earnings
  shoe_earnings / shoe_price

theorem shoes_sold_is_six : shoes_sold 3 2 18 27 = 6 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sold_shoes_sold_is_six_l198_19859


namespace NUMINAMATH_CALUDE_f_neg_two_eq_three_l198_19873

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

-- State the theorem
theorem f_neg_two_eq_three 
  (a b c : ℝ) 
  (h : f a b c 2 = -1) : 
  f a b c (-2) = 3 := by
sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_three_l198_19873


namespace NUMINAMATH_CALUDE_ham_bread_percentage_l198_19894

def bread_cost : ℕ := 50
def ham_cost : ℕ := 150
def cake_cost : ℕ := 200

def total_cost : ℕ := bread_cost + ham_cost + cake_cost
def ham_bread_cost : ℕ := bread_cost + ham_cost

theorem ham_bread_percentage :
  (ham_bread_cost : ℚ) / (total_cost : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ham_bread_percentage_l198_19894


namespace NUMINAMATH_CALUDE_a_plus_b_eighth_power_l198_19811

theorem a_plus_b_eighth_power (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7) :
  a^8 + b^8 = 47 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_eighth_power_l198_19811


namespace NUMINAMATH_CALUDE_x_squared_is_quadratic_l198_19819

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: x^2 = 0 is a quadratic equation -/
theorem x_squared_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_is_quadratic_l198_19819


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l198_19896

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + 2*x + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l198_19896


namespace NUMINAMATH_CALUDE_smallest_n_with_eight_and_terminating_l198_19836

/-- A function that checks if a positive integer contains the digit 8 -/
def containsEight (n : ℕ+) : Prop := sorry

/-- A function that checks if the reciprocal of a positive integer is a terminating decimal -/
def isTerminatingDecimal (n : ℕ+) : Prop := ∃ (a b : ℕ), n = 2^a * 5^b

theorem smallest_n_with_eight_and_terminating : 
  (∀ m : ℕ+, m < 8 → ¬(containsEight m ∧ isTerminatingDecimal m)) ∧ 
  (containsEight 8 ∧ isTerminatingDecimal 8) := by sorry

end NUMINAMATH_CALUDE_smallest_n_with_eight_and_terminating_l198_19836


namespace NUMINAMATH_CALUDE_candy_bar_cost_l198_19887

/-- The cost of each candy bar given Benny's purchase -/
theorem candy_bar_cost (soft_drink_cost : ℝ) (num_candy_bars : ℕ) (total_spent : ℝ)
  (h1 : soft_drink_cost = 2)
  (h2 : num_candy_bars = 5)
  (h3 : total_spent = 27)
  : (total_spent - soft_drink_cost) / num_candy_bars = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l198_19887


namespace NUMINAMATH_CALUDE_girls_in_class_l198_19857

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (num_girls : ℕ) :
  total = 35 →
  ratio_girls = 3 →
  ratio_boys = 4 →
  ratio_girls + ratio_boys = num_girls + (total - num_girls) →
  num_girls * ratio_boys = (total - num_girls) * ratio_girls →
  num_girls = 15 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l198_19857


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l198_19831

theorem simplify_and_evaluate : 
  let x : ℝ := 1
  let y : ℝ := -2
  7 * x * y - 2 * (5 * x * y - 2 * x^2 * y) + 3 * x * y = -8 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l198_19831


namespace NUMINAMATH_CALUDE_maciek_purchase_cost_l198_19849

-- Define the pricing structure
def pretzel_price (quantity : ℕ) : ℚ :=
  if quantity > 4 then 3.5 else 4

def chip_price (quantity : ℕ) : ℚ :=
  if quantity > 3 then 6.5 else 7

def soda_price (quantity : ℕ) : ℚ :=
  if quantity > 5 then 1.5 else 2

-- Define Maciek's purchase quantities
def pretzel_quantity : ℕ := 5
def chip_quantity : ℕ := 4
def soda_quantity : ℕ := 6

-- Calculate the total cost
def total_cost : ℚ :=
  pretzel_price pretzel_quantity * pretzel_quantity +
  chip_price chip_quantity * chip_quantity +
  soda_price soda_quantity * soda_quantity

-- Theorem statement
theorem maciek_purchase_cost :
  total_cost = 52.5 := by sorry

end NUMINAMATH_CALUDE_maciek_purchase_cost_l198_19849


namespace NUMINAMATH_CALUDE_batsman_average_increase_l198_19833

/-- Represents a batsman's statistics -/
structure Batsman where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the increase in average for a batsman -/
def averageIncrease (prevInnings : ℕ) (prevTotalScore : ℕ) (newScore : ℕ) : ℚ :=
  let newAverage := (prevTotalScore + newScore) / (prevInnings + 1)
  let oldAverage := prevTotalScore / prevInnings
  newAverage - oldAverage

/-- Theorem stating the increase in average for the given batsman -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 19 →
    b.average = 64 →
    averageIncrease 18 (18 * (b.totalScore / 19)) 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l198_19833


namespace NUMINAMATH_CALUDE_inequality_solution_set_l198_19892

/-- The solution set of the inequality (x-2)(ax-2) > 0 -/
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then Set.Iio 2
  else if a < 0 then Set.Ioo (2/a) 2
  else if 0 < a ∧ a < 1 then Set.Iio 2 ∪ Set.Ioi (2/a)
  else if a > 1 then Set.Iio (2/a) ∪ Set.Ioi 2
  else Set.Iio 2 ∪ Set.Ioi 2

theorem inequality_solution_set (a : ℝ) (x : ℝ) :
  (x - 2) * (a * x - 2) > 0 ↔ x ∈ solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l198_19892


namespace NUMINAMATH_CALUDE_typist_salary_problem_l198_19814

theorem typist_salary_problem (x : ℝ) : 
  (x * 1.1 * 0.95 = 6270) → x = 6000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l198_19814


namespace NUMINAMATH_CALUDE_room_occupancy_l198_19823

theorem room_occupancy (x : ℕ) : 
  (3 * x / 8 : ℚ) - 6 = 18 → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_room_occupancy_l198_19823


namespace NUMINAMATH_CALUDE_consecutive_cubes_divisibility_l198_19886

theorem consecutive_cubes_divisibility (n : ℕ) (h : ¬ 3 ∣ n) :
  9 * n ∣ ((n - 1)^3 + n^3 + (n + 1)^3) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_cubes_divisibility_l198_19886


namespace NUMINAMATH_CALUDE_cyclic_pentagon_area_diagonal_ratio_l198_19832

/-- A cyclic pentagon is a pentagon inscribed in a circle -/
structure CyclicPentagon where
  vertices : Fin 5 → ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ
  is_cyclic : ∀ i : Fin 5, dist (vertices i) center = radius

/-- The area of a cyclic pentagon -/
def area (p : CyclicPentagon) : ℝ := sorry

/-- The sum of the diagonals of a cyclic pentagon -/
def sum_diagonals (p : CyclicPentagon) : ℝ := sorry

/-- The theorem stating that the ratio of a cyclic pentagon's area to the sum of its diagonals
    is not greater than a quarter of its circumradius -/
theorem cyclic_pentagon_area_diagonal_ratio (p : CyclicPentagon) :
  area p / sum_diagonals p ≤ p.radius / 4 := by sorry

end NUMINAMATH_CALUDE_cyclic_pentagon_area_diagonal_ratio_l198_19832


namespace NUMINAMATH_CALUDE_sum_of_specific_T_values_l198_19851

def T (n : ℕ) : ℤ :=
  (-1 : ℤ) + 4 - 3 + 8 - 5 + ((-1)^n * (2*n : ℤ)) + ((-1)^(n+1) * (n : ℤ))

theorem sum_of_specific_T_values :
  T 27 + T 43 + T 60 = -84 ∨
  T 27 + T 43 + T 60 = -42 ∨
  T 27 + T 43 + T 60 = 0 ∨
  T 27 + T 43 + T 60 = 42 ∨
  T 27 + T 43 + T 60 = 84 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_T_values_l198_19851


namespace NUMINAMATH_CALUDE_average_books_borrowed_l198_19847

theorem average_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ)
  (h1 : total_students = 40)
  (h2 : zero_books = 2)
  (h3 : one_book = 12)
  (h4 : two_books = 12)
  (h5 : zero_books + one_book + two_books < total_students) :
  let remaining_students := total_students - (zero_books + one_book + two_books)
  let total_books := one_book * 1 + two_books * 2 + remaining_students * 3
  (total_books : ℚ) / total_students = 39/20 := by
sorry

end NUMINAMATH_CALUDE_average_books_borrowed_l198_19847


namespace NUMINAMATH_CALUDE_solve_for_a_l198_19806

theorem solve_for_a (x a : ℝ) (h : 2 * x - a = -5) (hx : x = 5) : a = 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l198_19806


namespace NUMINAMATH_CALUDE_smaller_number_problem_l198_19829

theorem smaller_number_problem (a b : ℝ) (h1 : a + b = 15) (h2 : 3 * (a - b) = 21) : 
  min a b = 4 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l198_19829


namespace NUMINAMATH_CALUDE_fixed_fee_is_ten_l198_19877

/-- Represents the billing structure for an online service provider -/
structure BillingStructure where
  fixed_fee : ℝ
  hourly_charge : ℝ

/-- Represents the monthly usage and bill -/
structure MonthlyBill where
  connect_time : ℝ
  total_bill : ℝ

/-- The billing problem with given conditions -/
def billing_problem (b : BillingStructure) : Prop :=
  ∃ (feb_time : ℝ),
    let feb : MonthlyBill := ⟨feb_time, 20⟩
    let mar : MonthlyBill := ⟨2 * feb_time, 30⟩
    let apr : MonthlyBill := ⟨3 * feb_time, 40⟩
    (b.fixed_fee + b.hourly_charge * feb.connect_time = feb.total_bill) ∧
    (b.fixed_fee + b.hourly_charge * mar.connect_time = mar.total_bill) ∧
    (b.fixed_fee + b.hourly_charge * apr.connect_time = apr.total_bill)

/-- The theorem stating that the fixed monthly fee is $10.00 -/
theorem fixed_fee_is_ten :
  ∀ b : BillingStructure, billing_problem b → b.fixed_fee = 10 := by
  sorry


end NUMINAMATH_CALUDE_fixed_fee_is_ten_l198_19877


namespace NUMINAMATH_CALUDE_find_number_l198_19880

theorem find_number : ∃ x : ℚ, x - (3/5) * x = 62 ∧ x = 155 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l198_19880


namespace NUMINAMATH_CALUDE_harolds_marbles_l198_19864

/-- Given that Harold has 100 marbles, keeps 20 for himself, and shares the rest evenly among 5 friends,
    prove that each friend receives 16 marbles. -/
theorem harolds_marbles (total : ℕ) (kept : ℕ) (friends : ℕ) 
    (h1 : total = 100) 
    (h2 : kept = 20)
    (h3 : friends = 5) :
    (total - kept) / friends = 16 := by
  sorry

end NUMINAMATH_CALUDE_harolds_marbles_l198_19864


namespace NUMINAMATH_CALUDE_horner_method_proof_polynomial_value_at_2_l198_19869

def horner_polynomial (x : ℝ) : ℝ :=
  ((((2 * x + 4) * x - 2) * x - 3) * x + 1) * x

theorem horner_method_proof :
  horner_polynomial 2 = 2 * 2^5 - 3 * 2^2 + 4 * 2^4 - 2 * 2^3 + 2 :=
by sorry

theorem polynomial_value_at_2 :
  horner_polynomial 2 = 102 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_proof_polynomial_value_at_2_l198_19869


namespace NUMINAMATH_CALUDE_equation_proof_l198_19820

theorem equation_proof : ∃ (op1 op2 op3 op4 : ℕ → ℕ → ℕ), 
  (op1 = (·-·) ∧ op2 = (·*·) ∧ op3 = (·/·) ∧ op4 = (·+·)) ∨
  (op1 = (·-·) ∧ op2 = (·*·) ∧ op3 = (·+·) ∧ op4 = (·/·)) ∨
  (op1 = (·-·) ∧ op2 = (·+·) ∧ op3 = (·*·) ∧ op4 = (·/·)) ∨
  (op1 = (·-·) ∧ op2 = (·+·) ∧ op3 = (·/·) ∧ op4 = (·*·)) ∨
  (op1 = (·-·) ∧ op2 = (·/·) ∧ op3 = (·*·) ∧ op4 = (·+·)) ∨
  (op1 = (·-·) ∧ op2 = (·/·) ∧ op3 = (·+·) ∧ op4 = (·*·)) ∨
  (op1 = (·+·) ∧ op2 = (·-·) ∧ op3 = (·*·) ∧ op4 = (·/·)) ∨
  (op1 = (·+·) ∧ op2 = (·-·) ∧ op3 = (·/·) ∧ op4 = (·*·)) ∨
  (op1 = (·+·) ∧ op2 = (·*·) ∧ op3 = (·-·) ∧ op4 = (·/·)) ∨
  (op1 = (·+·) ∧ op2 = (·*·) ∧ op3 = (·/·) ∧ op4 = (·-·)) ∨
  (op1 = (·+·) ∧ op2 = (·/·) ∧ op3 = (·-·) ∧ op4 = (·*·)) ∨
  (op1 = (·+·) ∧ op2 = (·/·) ∧ op3 = (·*·) ∧ op4 = (·-·)) ∨
  (op1 = (·*·) ∧ op2 = (·-·) ∧ op3 = (·+·) ∧ op4 = (·/·)) ∨
  (op1 = (·*·) ∧ op2 = (·-·) ∧ op3 = (·/·) ∧ op4 = (·+·)) ∨
  (op1 = (·*·) ∧ op2 = (·+·) ∧ op3 = (·-·) ∧ op4 = (·/·)) ∨
  (op1 = (·*·) ∧ op2 = (·+·) ∧ op3 = (·/·) ∧ op4 = (·-·)) ∨
  (op1 = (·*·) ∧ op2 = (·/·) ∧ op3 = (·-·) ∧ op4 = (·+·)) ∨
  (op1 = (·*·) ∧ op2 = (·/·) ∧ op3 = (·+·) ∧ op4 = (·-·)) ∨
  (op1 = (·/·) ∧ op2 = (·-·) ∧ op3 = (·*·) ∧ op4 = (·+·)) ∨
  (op1 = (·/·) ∧ op2 = (·-·) ∧ op3 = (·+·) ∧ op4 = (·*·)) ∨
  (op1 = (·/·) ∧ op2 = (·+·) ∧ op3 = (·-·) ∧ op4 = (·*·)) ∨
  (op1 = (·/·) ∧ op2 = (·+·) ∧ op3 = (·*·) ∧ op4 = (·-·)) ∨
  (op1 = (·/·) ∧ op2 = (·*·) ∧ op3 = (·-·) ∧ op4 = (·+·)) ∨
  (op1 = (·/·) ∧ op2 = (·*·) ∧ op3 = (·+·) ∧ op4 = (·-·)) →
  (op3 (op1 132 (op2 7 6)) (op4 12 3)) = 6 := by
sorry

end NUMINAMATH_CALUDE_equation_proof_l198_19820


namespace NUMINAMATH_CALUDE_circle_tripled_radius_l198_19889

theorem circle_tripled_radius (r : ℝ) (h : r > 0) :
  let new_r := 3 * r
  let original_area := π * r^2
  let new_area := π * new_r^2
  let original_circumference := 2 * π * r
  let new_circumference := 2 * π * new_r
  (new_area = 9 * original_area) ∧ (new_circumference = 3 * original_circumference) := by
  sorry

end NUMINAMATH_CALUDE_circle_tripled_radius_l198_19889


namespace NUMINAMATH_CALUDE_division_theorem_l198_19837

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = divisor * quotient + remainder →
  dividend = 162 →
  divisor = 17 →
  remainder = 9 →
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l198_19837


namespace NUMINAMATH_CALUDE_arithmetic_computation_l198_19899

theorem arithmetic_computation : -5 * 3 - (-8 * -2) + (-7 * -4) / 2 = -17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l198_19899


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_attainable_l198_19888

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  ((2*a + b)^2 + (b - 2*c)^2 + (c - a)^2) / b^2 ≥ 4/3 :=
by sorry

theorem lower_bound_attainable (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  ∃ (a' b' c' : ℝ), b' > c' ∧ c' > a' ∧ b' ≠ 0 ∧
    ((2*a' + b')^2 + (b' - 2*c')^2 + (c' - a')^2) / b'^2 = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_attainable_l198_19888


namespace NUMINAMATH_CALUDE_potion_price_l198_19872

theorem potion_price (current_price : ℚ) (original_price : ℚ) : 
  current_price = 9 → current_price = (1 / 15) * original_price → original_price = 135 := by
  sorry

end NUMINAMATH_CALUDE_potion_price_l198_19872


namespace NUMINAMATH_CALUDE_cafeteria_pies_l198_19862

/-- Given a cafeteria with initial apples, apples handed out, and apples required per pie,
    calculate the number of pies that can be made. -/
def calculate_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

/-- Theorem stating that with 47 initial apples, 27 handed out, and 4 apples per pie,
    the number of pies that can be made is 5. -/
theorem cafeteria_pies :
  calculate_pies 47 27 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l198_19862


namespace NUMINAMATH_CALUDE_cylinder_volume_doubling_l198_19895

/-- Given a cylinder with original volume 10 cubic feet, prove that doubling its height
    while keeping the radius constant results in a new volume of 20 cubic feet. -/
theorem cylinder_volume_doubling (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  π * r^2 * h = 10 → π * r^2 * (2 * h) = 20 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_doubling_l198_19895


namespace NUMINAMATH_CALUDE_set_equality_implies_m_zero_l198_19812

theorem set_equality_implies_m_zero (m : ℝ) : 
  ({3, m} : Set ℝ) = ({3*m, 3} : Set ℝ) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_m_zero_l198_19812


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_two_l198_19881

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of 2 is -2 -/
theorem opposite_of_two : opposite 2 = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_two_l198_19881


namespace NUMINAMATH_CALUDE_ghost_castle_windows_l198_19853

theorem ghost_castle_windows (n : ℕ) (h : n = 8) : n * (n - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ghost_castle_windows_l198_19853


namespace NUMINAMATH_CALUDE_sum_of_remainders_mod_500_l198_19852

def remainders : Finset ℕ := Finset.image (fun n => (3^n) % 500) (Finset.range 101)

def T : ℕ := Finset.sum remainders id

theorem sum_of_remainders_mod_500 : T % 500 = (Finset.sum (Finset.range 101) (fun n => (3^n) % 500)) % 500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_remainders_mod_500_l198_19852


namespace NUMINAMATH_CALUDE_min_sum_squares_given_cubic_constraint_l198_19843

/-- Given real numbers x, y, and z satisfying x^3 + y^3 + z^3 - 3xyz = 1,
    the sum of their squares x^2 + y^2 + z^2 is always greater than or equal to 1 -/
theorem min_sum_squares_given_cubic_constraint (x y z : ℝ) 
    (h : x^3 + y^3 + z^3 - 3*x*y*z = 1) : 
    x^2 + y^2 + z^2 ≥ 1 := by
  sorry

#check min_sum_squares_given_cubic_constraint

end NUMINAMATH_CALUDE_min_sum_squares_given_cubic_constraint_l198_19843


namespace NUMINAMATH_CALUDE_simplify_expression_l198_19846

theorem simplify_expression (a b : ℝ) : a - 4*(2*a - b) - 2*(a + 2*b) = -9*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l198_19846


namespace NUMINAMATH_CALUDE_thomas_books_proof_l198_19897

/-- The number of books Thomas owns -/
def num_books : ℕ := 200

/-- The selling price of each book in dollars -/
def book_price : ℚ := 3/2

/-- The cost of each record in dollars -/
def record_price : ℕ := 3

/-- The number of records Thomas buys -/
def num_records : ℕ := 75

/-- The amount of money Thomas has left over in dollars -/
def money_left : ℕ := 75

theorem thomas_books_proof :
  num_books * book_price = record_price * num_records + money_left :=
by sorry

end NUMINAMATH_CALUDE_thomas_books_proof_l198_19897


namespace NUMINAMATH_CALUDE_sum_first_six_primes_l198_19841

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the first n prime numbers -/
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the first 6 prime numbers is 41 -/
theorem sum_first_six_primes : sumFirstNPrimes 6 = 41 := by sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_l198_19841


namespace NUMINAMATH_CALUDE_range_of_a_l198_19890

/-- A function that is decreasing on R and defined piecewise --/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (-x) else -x^2 - 2*x + 1

/-- f is decreasing on R --/
axiom f_decreasing : ∀ x y : ℝ, x < y → f y < f x

/-- The theorem to prove --/
theorem range_of_a (a : ℝ) : f (a - 1) ≥ f (-a^2 + 1) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l198_19890


namespace NUMINAMATH_CALUDE_parallel_lines_length_l198_19828

/-- Represents a line segment with a length -/
structure LineSegment where
  length : ℝ

/-- Represents a set of parallel line segments -/
structure ParallelLines where
  ab : LineSegment
  cd : LineSegment
  ef : LineSegment
  gh : LineSegment

/-- 
Given parallel lines AB, CD, EF, and GH, where DC = 120 cm and AB = 180 cm, 
the length of GH is 72 cm.
-/
theorem parallel_lines_length (lines : ParallelLines) 
  (h1 : lines.cd.length = 120)
  (h2 : lines.ab.length = 180) :
  lines.gh.length = 72 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_length_l198_19828


namespace NUMINAMATH_CALUDE_simplify_expression_l198_19804

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 81) - Real.sqrt (8 + 1/2))^2 = (35 - 6 * Real.sqrt 34) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l198_19804


namespace NUMINAMATH_CALUDE_locus_of_P_l198_19821

-- Define the circle
def Circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define the property for point P
def P_property (P : ℝ × ℝ) : Prop :=
  ∃ (B : ℝ × ℝ),
    B ∈ Circle ∧
    (P.1 - A.1) * B.1 = P.2 * B.2 ∧  -- AP || OB
    (P.1 - A.1) * (B.1 - A.1) + P.2 * B.2 = 1  -- AP · AB = 1

-- The theorem to prove
theorem locus_of_P :
  ∀ (P : ℝ × ℝ), P_property P ↔ P.2^2 = 2 * P.1 - 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_P_l198_19821


namespace NUMINAMATH_CALUDE_number_solution_l198_19865

theorem number_solution : 
  ∀ (number : ℝ), (number * (-8) = 1600) → number = -200 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l198_19865


namespace NUMINAMATH_CALUDE_taut_if_pred_prime_l198_19850

def is_taut (n : ℕ) : Prop :=
  ∃ (S : Finset (Fin (n^2 - n + 1))),
    S.card = n ∧
    ∀ (a b c d : Fin (n^2 - n + 1)),
      a ∈ S → b ∈ S → c ∈ S → d ∈ S →
      a ≠ b → c ≠ d →
      (a : ℕ) * (d : ℕ) ≠ (b : ℕ) * (c : ℕ)

theorem taut_if_pred_prime (n : ℕ) (h : n ≥ 2) (h_prime : Nat.Prime (n - 1)) :
  is_taut n :=
sorry

end NUMINAMATH_CALUDE_taut_if_pred_prime_l198_19850


namespace NUMINAMATH_CALUDE_hand_count_theorem_l198_19845

def special_deck_size : ℕ := 60
def hand_size : ℕ := 12

def number_of_hands : ℕ := Nat.choose special_deck_size hand_size

theorem hand_count_theorem (C : ℕ) (h : C < 10) :
  ∃ (B : ℕ), number_of_hands = 192 * (10^6) + B * (10^5) + C * (10^4) + 3210 :=
by sorry

end NUMINAMATH_CALUDE_hand_count_theorem_l198_19845


namespace NUMINAMATH_CALUDE_basis_linear_independence_l198_19824

-- Define a 2D vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the property of being a basis for a plane
def IsBasisForPlane (e₁ e₂ : V) : Prop :=
  ∀ v : V, ∃ (m n : ℝ), v = m • e₁ + n • e₂

-- Define the property of vectors being not collinear
def NotCollinear (e₁ e₂ : V) : Prop :=
  ∀ (k : ℝ), k • e₁ ≠ e₂

-- The main theorem
theorem basis_linear_independence
  (e₁ e₂ : V)
  (h_basis : IsBasisForPlane e₁ e₂)
  (h_not_collinear : NotCollinear e₁ e₂) :
  ∀ (m n : ℝ), m • e₁ + n • e₂ = 0 → m = 0 ∧ n = 0 :=
by sorry

end NUMINAMATH_CALUDE_basis_linear_independence_l198_19824


namespace NUMINAMATH_CALUDE_vaccine_cost_l198_19863

theorem vaccine_cost (num_vaccines : ℕ) (doctor_visit_cost : ℝ) 
  (insurance_coverage : ℝ) (trip_cost : ℝ) (total_payment : ℝ) :
  num_vaccines = 10 ∧ 
  doctor_visit_cost = 250 ∧ 
  insurance_coverage = 0.8 ∧ 
  trip_cost = 1200 ∧ 
  total_payment = 1340 →
  (total_payment - trip_cost - (1 - insurance_coverage) * doctor_visit_cost) / 
  ((1 - insurance_coverage) * num_vaccines) = 45 := by
  sorry

end NUMINAMATH_CALUDE_vaccine_cost_l198_19863


namespace NUMINAMATH_CALUDE_unique_four_digit_square_divisible_by_11_ending_in_1_l198_19822

theorem unique_four_digit_square_divisible_by_11_ending_in_1 :
  ∃! n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ ∃ k : ℕ, n = k^2 ∧ n % 11 = 0 ∧ n % 10 = 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_divisible_by_11_ending_in_1_l198_19822


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l198_19885

-- Define a function to check if an angle is in the first quadrant
def is_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90

-- Define a function to check if an angle is in the first or third quadrant
def is_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ n : ℤ, (n * 360 < α ∧ α < n * 360 + 45) ∨ 
            (n * 360 + 180 < α ∧ α < n * 360 + 225)

-- Theorem statement
theorem half_angle_quadrant (α : ℝ) :
  is_first_quadrant α → is_first_or_third_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l198_19885


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l198_19810

theorem sum_of_a_and_b (a b : ℝ) (h : Real.sqrt (a + 2) + (b - 3)^2 = 0) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l198_19810


namespace NUMINAMATH_CALUDE_binomial_coefficient_16_11_l198_19854

theorem binomial_coefficient_16_11 
  (h1 : Nat.choose 15 9 = 5005)
  (h2 : Nat.choose 15 10 = 3003)
  (h3 : Nat.choose 17 11 = 12376) :
  Nat.choose 16 11 = 4368 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_16_11_l198_19854


namespace NUMINAMATH_CALUDE_kates_retirement_fund_l198_19870

/-- Given a retirement fund with an initial value and a decrease amount, 
    calculate the current value of the fund. -/
def current_fund_value (initial_value decrease : ℕ) : ℕ :=
  initial_value - decrease

/-- Theorem: Kate's retirement fund's current value -/
theorem kates_retirement_fund : 
  current_fund_value 1472 12 = 1460 := by
  sorry

end NUMINAMATH_CALUDE_kates_retirement_fund_l198_19870


namespace NUMINAMATH_CALUDE_conference_handshakes_l198_19876

/-- The number of handshakes in a conference of n people where each person
    shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := Nat.choose n 2

/-- Theorem stating that in a conference of 10 people, where each person
    shakes hands with every other person exactly once, there are 45 handshakes. -/
theorem conference_handshakes :
  handshakes 10 = 45 := by sorry

end NUMINAMATH_CALUDE_conference_handshakes_l198_19876


namespace NUMINAMATH_CALUDE_locus_of_touching_parabolas_l198_19855

/-- Given a parabola y = x^2 with directrix y = -1/4, this theorem describes
    the locus of points P(u, v) for which there exists a line v parallel to
    the directrix and at a distance s from it, such that the parabola with
    directrix v and focus P touches the given parabola. -/
theorem locus_of_touching_parabolas (s : ℝ) (u v : ℝ) :
  (2 * s ≠ 1 → (v = (1 / (1 - 2 * s)) * u^2 + s / 2 ∨
                v = (1 / (1 + 2 * s)) * u^2 - s / 2)) ∧
  (2 * s = 1 → v = u^2 / 2 - 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_touching_parabolas_l198_19855


namespace NUMINAMATH_CALUDE_tangent_line_equation_max_integer_k_l198_19808

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := (Real.log x + k) / Real.exp x

-- Define the derivative of f
def f_derivative (k : ℝ) (x : ℝ) : ℝ := (1 - k*x - x * Real.log x) / (x * Real.exp x)

-- Theorem for the tangent line equation
theorem tangent_line_equation (x y : ℝ) :
  k = 2 →
  y = f 2 x →
  x = 1 →
  (x + Real.exp y - 3 = 0) :=
sorry

-- Theorem for the maximum integer value of k
theorem max_integer_k (k : ℤ) :
  (∀ x > 1, x * Real.exp x * f_derivative k x + (2 * ↑k - 1) * x < 1 + ↑k) →
  k ≤ 3 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_equation_max_integer_k_l198_19808


namespace NUMINAMATH_CALUDE_koshchey_chest_count_l198_19800

/-- Represents the number of chests Koshchey has -/
structure KoshcheyChests where
  large : ℕ
  medium : ℕ
  small : ℕ
  empty : ℕ

/-- The total number of chests Koshchey has -/
def total_chests (k : KoshcheyChests) : ℕ :=
  k.large + k.medium + k.small

/-- Koshchey's chest configuration satisfies the problem conditions -/
def is_valid_configuration (k : KoshcheyChests) : Prop :=
  k.large = 11 ∧
  k.empty = 102 ∧
  ∃ (x : ℕ), x ≤ k.large ∧ k.medium = 8 * x

theorem koshchey_chest_count (k : KoshcheyChests) 
  (h : is_valid_configuration k) : total_chests k = 115 :=
by sorry

end NUMINAMATH_CALUDE_koshchey_chest_count_l198_19800


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocal_implies_fraction_l198_19842

theorem sqrt_sum_reciprocal_implies_fraction (x : ℝ) (h : Real.sqrt x + 1 / Real.sqrt x = 3) :
  x / (x^2 + 2018*x + 1) = 1 / 2025 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocal_implies_fraction_l198_19842


namespace NUMINAMATH_CALUDE_quadratic_part_of_equation_l198_19839

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - 8*x + 21 = |x - 5| + 4

-- Define the sum of solutions
def sum_of_solutions : ℝ := 20

-- Theorem to prove
theorem quadratic_part_of_equation :
  ∃ (a b c : ℝ), 
    (∀ x, quadratic_equation x → a*x^2 + b*x + c = |x - 5| + 4) ∧
    (a = 1 ∧ b = -8 ∧ c = 21) :=
sorry

end NUMINAMATH_CALUDE_quadratic_part_of_equation_l198_19839


namespace NUMINAMATH_CALUDE_angle_AEC_measure_l198_19817

-- Define the angles in the triangle
def angle_ABE' : ℝ := 150
def angle_BAC : ℝ := 108

-- Define the property of supplementary angles
def supplementary (a b : ℝ) : Prop := a + b = 180

-- Theorem statement
theorem angle_AEC_measure :
  ∀ angle_ABE angle_AEC,
  supplementary angle_ABE angle_ABE' →
  angle_ABE + angle_BAC + angle_AEC = 180 →
  angle_AEC = 42 := by
    sorry

end NUMINAMATH_CALUDE_angle_AEC_measure_l198_19817


namespace NUMINAMATH_CALUDE_largest_common_term_l198_19827

def sequence1 (n : ℕ) : ℕ := 2 + 4 * n
def sequence2 (m : ℕ) : ℕ := 3 + 5 * m

theorem largest_common_term : 
  (∃ (n m : ℕ), sequence1 n = sequence2 m ∧ sequence1 n = 138) ∧ 
  (∀ (n m : ℕ), sequence1 n = sequence2 m → sequence1 n ≤ 150 → sequence1 n ≤ 138) :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l198_19827


namespace NUMINAMATH_CALUDE_street_lamp_combinations_l198_19813

/-- The number of lamps in the row -/
def total_lamps : ℕ := 12

/-- The number of lamps that can be turned off -/
def lamps_to_turn_off : ℕ := 3

/-- The number of valid positions to insert turned-off lamps -/
def valid_positions : ℕ := total_lamps - lamps_to_turn_off - 1

theorem street_lamp_combinations : 
  (valid_positions.choose lamps_to_turn_off) = 56 := by
  sorry

#eval valid_positions.choose lamps_to_turn_off

end NUMINAMATH_CALUDE_street_lamp_combinations_l198_19813


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l198_19868

/-- The perpendicular bisector of a line segment AB is the line that passes through 
    the midpoint of AB and is perpendicular to AB. This theorem proves that 
    y = -2x + 3 is the equation of the perpendicular bisector of the line segment 
    connecting points A(-1, 0) and B(3, 2). -/
theorem perpendicular_bisector_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (3, 2)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let slope_AB : ℝ := (B.2 - A.2) / (B.1 - A.1)
  let slope_perp : ℝ := -1 / slope_AB
  y = -2 * x + 3 ↔ 
    (x, y) ∈ {p : ℝ × ℝ | (p.1 - midpoint.1) * slope_AB = (midpoint.2 - p.2)} ∧
    (y - midpoint.2) = slope_perp * (x - midpoint.1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l198_19868


namespace NUMINAMATH_CALUDE_investment_calculation_l198_19879

/-- Represents the investment of two business partners -/
structure BusinessInvestment where
  p_investment : ℕ  -- P's investment
  q_investment : ℕ  -- Q's investment
  profit_ratio : Rat  -- Ratio of P's profit share to Q's profit share

/-- 
Theorem: Given two investors P and Q who divide their profit in the ratio 4:6,
if P invested 60000, then Q invested 90000.
-/
theorem investment_calculation (investment : BusinessInvestment) :
  investment.p_investment = 60000 ∧ 
  investment.profit_ratio = 4 / 6 →
  investment.q_investment = 90000 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l198_19879


namespace NUMINAMATH_CALUDE_solve_equation_XY_l198_19884

/-- Given the equation Yx + 8 / (x^2 - 11x + 30) = X / (x - 5) + 7 / (x - 6),
    prove that X + Y = -22/3 -/
theorem solve_equation_XY (X Y : ℚ) :
  (∀ x : ℚ, x ≠ 5 ∧ x ≠ 6 → (Y * x + 8) / (x^2 - 11*x + 30) = X / (x - 5) + 7 / (x - 6)) →
  X + Y = -22/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_XY_l198_19884


namespace NUMINAMATH_CALUDE_sisters_ages_solution_l198_19816

/-- Represents the ages of three sisters -/
structure SistersAges where
  oldest : ℕ
  middle : ℕ
  youngest : ℕ

/-- The condition given in the problem -/
def ageCondition (ages : SistersAges) : Prop :=
  ages.oldest * 10 - 9 * ages.youngest = 89

/-- The theorem stating the correct ages of the sisters -/
theorem sisters_ages_solution :
  ∃ (ages : SistersAges),
    ages.middle = 10 ∧
    ageCondition ages ∧
    ages.oldest = 17 ∧
    ages.youngest = 9 := by
  sorry

end NUMINAMATH_CALUDE_sisters_ages_solution_l198_19816


namespace NUMINAMATH_CALUDE_sqrt_sum_div_sqrt_eq_rational_l198_19835

theorem sqrt_sum_div_sqrt_eq_rational : (Real.sqrt 112 + Real.sqrt 567) / Real.sqrt 175 = 13 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_div_sqrt_eq_rational_l198_19835


namespace NUMINAMATH_CALUDE_dodecahedral_die_expected_value_l198_19844

/-- A fair dodecahedral die with faces numbered from 1 to 12 -/
def DodecahedralDie : Finset ℕ := Finset.range 12

/-- The probability of each outcome for a fair die -/
def prob (n : ℕ) : ℚ := 1 / 12

/-- The expected value of rolling the die -/
def expected_value : ℚ := (DodecahedralDie.sum (fun i => prob i * (i + 1)))

/-- Theorem: The expected value of rolling a fair dodecahedral die is 6.5 -/
theorem dodecahedral_die_expected_value :
  expected_value = 13 / 2 := by sorry

end NUMINAMATH_CALUDE_dodecahedral_die_expected_value_l198_19844


namespace NUMINAMATH_CALUDE_distributive_property_l198_19834

theorem distributive_property (x : ℝ) : -2 * (x + 1) = -2 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_l198_19834


namespace NUMINAMATH_CALUDE_apple_weight_l198_19860

/-- Given a bag containing apples, prove the weight of one apple. -/
theorem apple_weight (total_weight : ℝ) (empty_bag_weight : ℝ) (apple_count : ℕ) :
  total_weight = 1.82 →
  empty_bag_weight = 0.5 →
  apple_count = 6 →
  (total_weight - empty_bag_weight) / apple_count = 0.22 := by
  sorry

end NUMINAMATH_CALUDE_apple_weight_l198_19860


namespace NUMINAMATH_CALUDE_chicken_count_l198_19861

theorem chicken_count (coop run free_range : ℕ) : 
  coop = 14 →
  run = 2 * coop →
  free_range = 2 * run - 4 →
  free_range = 52 := by
sorry

end NUMINAMATH_CALUDE_chicken_count_l198_19861


namespace NUMINAMATH_CALUDE_expression_simplification_l198_19871

theorem expression_simplification (a b : ℝ) : 
  3 * a - 4 * b + 2 * a^2 - (7 * a - 2 * a^2 + 3 * b - 5) = -4 * a - 7 * b + 4 * a^2 + 5 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l198_19871


namespace NUMINAMATH_CALUDE_u_2023_equals_4_l198_19818

-- Define the function f
def f : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 2
| 4 => 1
| 5 => 4
| _ => 0  -- Default case for inputs not in the table

-- Define the sequence u
def u : ℕ → ℕ
| 0 => 5  -- Initial value u₀ = 5
| (n + 1) => f (u n)  -- Recursive definition: u_{n+1} = f(u_n)

-- Theorem to prove
theorem u_2023_equals_4 : u 2023 = 4 := by
  sorry

end NUMINAMATH_CALUDE_u_2023_equals_4_l198_19818


namespace NUMINAMATH_CALUDE_probability_one_red_one_green_l198_19809

def total_marbles : ℕ := 4 + 6 + 11

def prob_red_then_green : ℚ :=
  (4 : ℚ) / total_marbles * 6 / (total_marbles - 1)

def prob_green_then_red : ℚ :=
  (6 : ℚ) / total_marbles * 4 / (total_marbles - 1)

theorem probability_one_red_one_green :
  prob_red_then_green + prob_green_then_red = 4 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_red_one_green_l198_19809


namespace NUMINAMATH_CALUDE_largest_common_value_l198_19878

def first_progression (n : ℕ) : ℕ := 4 + 5 * n

def second_progression (m : ℕ) : ℕ := 5 + 10 * m

theorem largest_common_value :
  ∃ (n m : ℕ),
    first_progression n = second_progression m ∧
    first_progression n < 1000 ∧
    first_progression n ≡ 1 [MOD 4] ∧
    ∀ (k l : ℕ),
      first_progression k = second_progression l →
      first_progression k < 1000 →
      first_progression k ≡ 1 [MOD 4] →
      first_progression k ≤ first_progression n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_common_value_l198_19878
