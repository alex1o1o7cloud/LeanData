import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_at_point_l2964_296466

/-- The equation of the tangent line to y = 2x² at (1, 2) is y = 4x - 2 -/
theorem tangent_line_at_point (x y : ℝ) :
  (y = 2 * x^2) →  -- Given curve
  (∃ P : ℝ × ℝ, P = (1, 2)) →  -- Given point
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ∧ y = 4 * x - 2) -- Tangent line equation
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l2964_296466


namespace NUMINAMATH_CALUDE_janes_change_calculation_l2964_296488

-- Define the prices and quantities
def skirt_price : ℝ := 65
def skirt_quantity : ℕ := 2
def blouse_price : ℝ := 30
def blouse_quantity : ℕ := 3
def shoes_price : ℝ := 125
def handbag_price : ℝ := 175

-- Define the discounts and taxes
def handbag_discount : ℝ := 0.10
def total_discount : ℝ := 0.05
def coupon_discount : ℝ := 20
def sales_tax : ℝ := 0.08

-- Define the exchange rate and amount paid
def exchange_rate : ℝ := 0.8
def amount_paid : ℝ := 600

-- Theorem to prove
theorem janes_change_calculation :
  let initial_total := skirt_price * skirt_quantity + blouse_price * blouse_quantity + shoes_price + handbag_price
  let handbag_discounted := initial_total - handbag_discount * handbag_price
  let total_discounted := handbag_discounted * (1 - total_discount)
  let coupon_applied := total_discounted - coupon_discount
  let taxed_total := coupon_applied * (1 + sales_tax)
  let home_currency_total := taxed_total * exchange_rate
  amount_paid - home_currency_total = 204.828 := by sorry

end NUMINAMATH_CALUDE_janes_change_calculation_l2964_296488


namespace NUMINAMATH_CALUDE_circular_sequence_three_elements_l2964_296491

/-- A circular sequence of distinct elements -/
structure CircularSequence (α : Type*) where
  elements : List α
  distinct : elements.Nodup
  circular : elements ≠ []

/-- Predicate to check if a CircularSequence contains zero -/
def containsZero (s : CircularSequence ℤ) : Prop :=
  0 ∈ s.elements

/-- Predicate to check if a CircularSequence has an odd number of elements -/
def hasOddElements (s : CircularSequence ℤ) : Prop :=
  s.elements.length % 2 = 1

/-- The main theorem -/
theorem circular_sequence_three_elements
  (s : CircularSequence ℤ)
  (zero_in_s : containsZero s)
  (odd_elements : hasOddElements s) :
  s.elements.length = 3 :=
sorry

end NUMINAMATH_CALUDE_circular_sequence_three_elements_l2964_296491


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l2964_296452

def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {1, 3, 9}

theorem union_of_P_and_Q : P ∪ Q = {1, 2, 3, 9} := by
  sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l2964_296452


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l2964_296439

theorem arithmetic_mean_reciprocals_first_four_primes : 
  let first_four_primes := [2, 3, 5, 7]
  ((first_four_primes.map (λ x => 1 / x)).sum) / first_four_primes.length = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_four_primes_l2964_296439


namespace NUMINAMATH_CALUDE_divisible_by_three_exists_l2964_296487

/-- A type representing the arrangement of natural numbers in a circle. -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- Predicate to check if two numbers differ by 1, 2, or by a factor of two. -/
def ValidDifference (a b : ℕ) : Prop :=
  (a = b + 1) ∨ (b = a + 1) ∨ (a = b + 2) ∨ (b = a + 2) ∨ (a = 2 * b) ∨ (b = 2 * a)

/-- Theorem stating that in any arrangement of 99 natural numbers in a circle
    where any two neighboring numbers differ either by 1, or by 2, or by a factor of two,
    at least one of these numbers is divisible by 3. -/
theorem divisible_by_three_exists (arr : CircularArrangement 99)
  (h : ∀ i : Fin 99, ValidDifference (arr i) (arr (i + 1))) :
  ∃ i : Fin 99, 3 ∣ arr i := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_exists_l2964_296487


namespace NUMINAMATH_CALUDE_intersection_equality_condition_l2964_296410

theorem intersection_equality_condition (M N P : Set α) :
  (∀ (M N P : Set α), M = N → M ∩ P = N ∩ P) ∧
  (∃ (M N P : Set α), M ∩ P = N ∩ P ∧ M ≠ N) :=
sorry

end NUMINAMATH_CALUDE_intersection_equality_condition_l2964_296410


namespace NUMINAMATH_CALUDE_sufficient_condition_l2964_296400

theorem sufficient_condition (x y : ℝ) : x^2 + y^2 < 4 → x*y + 4 > 2*x + 2*y := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_l2964_296400


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l2964_296495

theorem least_five_digit_square_cube : 
  (∀ n : ℕ, n < 15625 → (n < 10000 ∨ ¬∃ a b : ℕ, n = a^2 ∧ n = b^3)) ∧ 
  15625 ≥ 10000 ∧ 
  ∃ a b : ℕ, 15625 = a^2 ∧ 15625 = b^3 :=
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l2964_296495


namespace NUMINAMATH_CALUDE_smallest_value_l2964_296408

theorem smallest_value : 
  let a := -((-3 - 2)^2)
  let b := (-3) * (-2)
  let c := (-3)^2 / (-2)^2
  let d := (-3)^2 / (-2)
  (a ≤ b) ∧ (a ≤ c) ∧ (a ≤ d) := by sorry

end NUMINAMATH_CALUDE_smallest_value_l2964_296408


namespace NUMINAMATH_CALUDE_problem_statement_l2964_296476

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : Real.exp a + Real.log b = 1) :
  (a + Real.log b < 0) ∧ (Real.exp a + b > 2) ∧ (a + b > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2964_296476


namespace NUMINAMATH_CALUDE_pascal_row_10_sum_l2964_296401

/-- The sum of the numbers in a row of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

/-- Theorem: The sum of the numbers in Row 10 of Pascal's Triangle is 1024 -/
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row_10_sum_l2964_296401


namespace NUMINAMATH_CALUDE_parallelogram_area_l2964_296406

def v1 : Fin 3 → ℝ := ![4, -1, 3]
def v2 : Fin 3 → ℝ := ![-2, 5, -1]

theorem parallelogram_area : 
  Real.sqrt ((v1 1 * v2 2 - v1 2 * v2 1)^2 + 
             (v1 2 * v2 0 - v1 0 * v2 2)^2 + 
             (v1 0 * v2 1 - v1 1 * v2 0)^2) = Real.sqrt 684 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2964_296406


namespace NUMINAMATH_CALUDE_perfect_square_property_l2964_296443

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + a n

theorem perfect_square_property (n : ℕ) (h : n > 0) : 
  ∃ m : ℤ, 2 * ((a (2 * n))^2 - 1) = m^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_property_l2964_296443


namespace NUMINAMATH_CALUDE_simplify_expression_a_l2964_296413

theorem simplify_expression_a (x a b : ℝ) :
  (3 * x^2 * (a^2 + b^2) - 3 * a^2 * b^2 + 3 * (x^2 + (a + b) * x + a * b) * (x * (x - a) - b * (x - a))) / x^2 = 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_a_l2964_296413


namespace NUMINAMATH_CALUDE_coin_distribution_impossibility_l2964_296489

theorem coin_distribution_impossibility : ∀ n : ℕ,
  n = 44 →
  n < (10 * 9) / 2 :=
by
  sorry

#check coin_distribution_impossibility

end NUMINAMATH_CALUDE_coin_distribution_impossibility_l2964_296489


namespace NUMINAMATH_CALUDE_frosting_theorem_l2964_296446

/-- Jon's frosting rate in cupcakes per second -/
def jon_rate : ℚ := 1 / 40

/-- Mary's frosting rate in cupcakes per second -/
def mary_rate : ℚ := 1 / 24

/-- Time frame in seconds -/
def time_frame : ℕ := 12 * 60

/-- The number of cupcakes Jon and Mary can frost together in the given time frame -/
def cupcakes_frosted : ℕ := 48

theorem frosting_theorem : 
  ⌊(jon_rate + mary_rate) * time_frame⌋ = cupcakes_frosted := by
  sorry

end NUMINAMATH_CALUDE_frosting_theorem_l2964_296446


namespace NUMINAMATH_CALUDE_pony_price_is_20_l2964_296436

/-- The regular price of fox jeans in dollars -/
def fox_price : ℝ := 15

/-- The regular price of pony jeans in dollars -/
def pony_price : ℝ := 20

/-- The number of fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings in dollars -/
def total_savings : ℝ := 9

/-- The sum of the two discount rates as a percentage -/
def total_discount_rate : ℝ := 22

/-- The discount rate on pony jeans as a percentage -/
def pony_discount_rate : ℝ := 18

/-- Theorem stating that the regular price of pony jeans is $20 given the conditions -/
theorem pony_price_is_20 : 
  fox_price * fox_quantity * (total_discount_rate - pony_discount_rate) / 100 +
  pony_price * pony_quantity * pony_discount_rate / 100 = total_savings :=
by sorry

end NUMINAMATH_CALUDE_pony_price_is_20_l2964_296436


namespace NUMINAMATH_CALUDE_rectangle_area_l2964_296428

/-- A rectangle with specific properties -/
structure Rectangle where
  width : ℝ
  length : ℝ
  length_exceed_twice_width : length = 2 * width + 25
  perimeter_650 : 2 * (length + width) = 650

/-- The area of a rectangle with the given properties is 22500 -/
theorem rectangle_area (r : Rectangle) : r.length * r.width = 22500 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2964_296428


namespace NUMINAMATH_CALUDE_initial_wallet_amount_l2964_296451

def initial_investment : ℝ := 2000
def stock_price_increase : ℝ := 0.3
def final_total : ℝ := 2900

theorem initial_wallet_amount :
  let investment_value := initial_investment * (1 + stock_price_increase)
  let initial_wallet := final_total - investment_value
  initial_wallet = 300 := by sorry

end NUMINAMATH_CALUDE_initial_wallet_amount_l2964_296451


namespace NUMINAMATH_CALUDE_houses_in_block_l2964_296469

theorem houses_in_block (junk_mail_per_house : ℕ) (total_junk_mail : ℕ) 
  (h1 : junk_mail_per_house = 2) 
  (h2 : total_junk_mail = 14) : 
  total_junk_mail / junk_mail_per_house = 7 := by
sorry

end NUMINAMATH_CALUDE_houses_in_block_l2964_296469


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_of_4500_l2964_296430

/-- The number of perfect square factors of 4500 -/
def perfectSquareFactorsOf4500 : ℕ :=
  -- Define the number of perfect square factors of 4500
  -- We don't implement the calculation here, just define it
  -- The actual value will be proven to be 8
  sorry

/-- Theorem: The number of perfect square factors of 4500 is 8 -/
theorem count_perfect_square_factors_of_4500 :
  perfectSquareFactorsOf4500 = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_of_4500_l2964_296430


namespace NUMINAMATH_CALUDE_arun_weight_average_l2964_296456

def weight_range (w : ℝ) : Prop :=
  66 < w ∧ w ≤ 69 ∧ 60 < w ∧ w < 70

theorem arun_weight_average : 
  ∃ (w₁ w₂ w₃ : ℝ), 
    weight_range w₁ ∧ 
    weight_range w₂ ∧ 
    weight_range w₃ ∧ 
    w₁ ≠ w₂ ∧ w₁ ≠ w₃ ∧ w₂ ≠ w₃ ∧
    (w₁ + w₂ + w₃) / 3 = 68 := by
  sorry

end NUMINAMATH_CALUDE_arun_weight_average_l2964_296456


namespace NUMINAMATH_CALUDE_rotated_A_coordinates_l2964_296447

-- Define the triangle OAB
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)

-- Define the properties of the triangle
structure Triangle where
  A : ℝ × ℝ
  first_quadrant : A.1 > 0 ∧ A.2 > 0
  right_angle : (A.1 - B.1) * (A.1 - O.1) + (A.2 - B.2) * (A.2 - O.2) = 0
  angle_AOB : Real.arctan ((A.2 - O.2) / (A.1 - O.1)) = π / 4

-- Function to rotate a point 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)

-- Theorem statement
theorem rotated_A_coordinates (t : Triangle) : 
  rotate90 t.A = (-8, 8) := by sorry

end NUMINAMATH_CALUDE_rotated_A_coordinates_l2964_296447


namespace NUMINAMATH_CALUDE_solution_difference_l2964_296497

theorem solution_difference (a b : ℝ) : 
  (∀ x, (x - 5) * (x + 5) = 26 * x - 130 ↔ x = a ∨ x = b) →
  a ≠ b →
  a > b →
  a - b = 16 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l2964_296497


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2964_296444

/-- Given that m + 2n - 1 = 0, prove that the line mx + 3y + n = 0 passes through the point (1/2, -1/6) -/
theorem line_passes_through_point (m n : ℝ) (h : m + 2 * n - 1 = 0) :
  m * (1/2 : ℝ) + 3 * (-1/6 : ℝ) + n = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2964_296444


namespace NUMINAMATH_CALUDE_binomial_coefficient_third_term_l2964_296404

theorem binomial_coefficient_third_term (x : ℝ) : 
  Nat.choose 4 2 = 6 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_third_term_l2964_296404


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l2964_296427

def a (n : ℕ) : ℕ := n.factorial + n^2

theorem max_gcd_consecutive_terms : 
  ∃ (k : ℕ), (∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ k) ∧ 
             (∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = k) ∧ 
             k = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l2964_296427


namespace NUMINAMATH_CALUDE_convenience_store_analysis_l2964_296474

-- Define the data types
structure YearData :=
  (year : Nat)
  (profit : Real)

-- Define the dataset
def dataset : List YearData := [
  ⟨2014, 27.6⟩, ⟨2015, 42.0⟩, ⟨2016, 38.4⟩, ⟨2017, 48.0⟩, ⟨2018, 63.6⟩,
  ⟨2019, 63.7⟩, ⟨2020, 72.8⟩, ⟨2021, 80.1⟩, ⟨2022, 60.5⟩, ⟨2023, 99.3⟩
]

-- Define the contingency table
def contingencyTable : Matrix (Fin 2) (Fin 2) Nat :=
  ![![2, 5],
    ![3, 0]]

-- Define the chi-square critical value
def chiSquareCritical : Real := 3.841

-- Define the prediction year
def predictionYear : Nat := 2024

-- Define the theorem
theorem convenience_store_analysis :
  -- Chi-square value is greater than the critical value
  ∃ (chiSquareValue : Real),
    chiSquareValue > chiSquareCritical ∧
    -- Predictions from two models are different
    ∃ (prediction1 prediction2 : Real),
      prediction1 ≠ prediction2 ∧
      -- Model 1: Using data from 2014 to 2023 (excluding 2022)
      (∃ (a1 b1 : Real),
        prediction1 = a1 * predictionYear + b1 ∧
        -- Model 2: Using data from 2019 to 2023
        ∃ (a2 b2 : Real),
          prediction2 = a2 * predictionYear + b2) :=
sorry

end NUMINAMATH_CALUDE_convenience_store_analysis_l2964_296474


namespace NUMINAMATH_CALUDE_investment_sum_l2964_296450

/-- Represents the investment scenario described in the problem -/
structure Investment where
  principal : ℝ  -- The initial sum invested
  rate : ℝ       -- The annual simple interest rate
  peter_years : ℕ := 3
  david_years : ℕ := 4
  peter_return : ℝ := 815
  david_return : ℝ := 854

/-- The amount returned after a given number of years with simple interest -/
def amount_after (i : Investment) (years : ℕ) : ℝ :=
  i.principal + (i.principal * i.rate * years)

/-- The theorem stating that the invested sum is 698 given the conditions -/
theorem investment_sum (i : Investment) : 
  (amount_after i i.peter_years = i.peter_return) → 
  (amount_after i i.david_years = i.david_return) → 
  i.principal = 698 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l2964_296450


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l2964_296499

theorem coefficient_of_x_squared (k : ℝ) : 
  k = 1.7777777777777777 → 2 * k = 3.5555555555555554 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l2964_296499


namespace NUMINAMATH_CALUDE_solve_pencil_problem_l2964_296421

def pencil_problem (anna_pencils : ℕ) (harry_multiplier : ℕ) (harry_lost : ℕ) : Prop :=
  let harry_initial := anna_pencils * harry_multiplier
  harry_initial - harry_lost = 81

theorem solve_pencil_problem :
  pencil_problem 50 2 19 := by sorry

end NUMINAMATH_CALUDE_solve_pencil_problem_l2964_296421


namespace NUMINAMATH_CALUDE_product_derivative_at_one_l2964_296493

theorem product_derivative_at_one
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (f1 : f 1 = -1)
  (f'1 : deriv f 1 = 2)
  (g1 : g 1 = -2)
  (g'1 : deriv g 1 = 1) :
  deriv (λ x => f x * g x) 1 = -5 := by
  sorry

end NUMINAMATH_CALUDE_product_derivative_at_one_l2964_296493


namespace NUMINAMATH_CALUDE_fruits_left_l2964_296475

theorem fruits_left (oranges apples : ℕ) 
  (h1 : oranges = 40)
  (h2 : apples = 70)
  (h3 : oranges / 4 + apples / 2 = oranges + apples - 65) : 
  oranges + apples - (oranges / 4 + apples / 2) = 65 := by
  sorry

#check fruits_left

end NUMINAMATH_CALUDE_fruits_left_l2964_296475


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_elements_l2964_296440

theorem pascal_triangle_row20_elements : 
  (Nat.choose 20 4 = 4845) ∧ (Nat.choose 20 5 = 15504) := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_elements_l2964_296440


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l2964_296434

theorem simplify_sqrt_difference : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l2964_296434


namespace NUMINAMATH_CALUDE_whipped_cream_theorem_l2964_296459

/-- Represents the number of each type of baked good produced on odd and even days -/
structure BakingSchedule where
  odd_pumpkin : ℕ
  odd_apple : ℕ
  odd_chocolate : ℕ
  even_pumpkin : ℕ
  even_apple : ℕ
  even_chocolate : ℕ
  even_lemon : ℕ

/-- Represents the amount of whipped cream needed for each type of baked good -/
structure WhippedCreamRequirement where
  pumpkin : ℚ
  apple : ℚ
  chocolate : ℚ
  lemon : ℚ

/-- Represents the number of each type of baked good Tiffany eats -/
structure TiffanyEats where
  pumpkin : ℕ
  apple : ℕ
  chocolate : ℕ
  lemon : ℕ

/-- Calculates the number of cans of whipped cream needed given the baking schedule,
    whipped cream requirements, and what Tiffany eats -/
def whippedCreamNeeded (schedule : BakingSchedule) (requirement : WhippedCreamRequirement) 
                       (tiffanyEats : TiffanyEats) : ℕ :=
  sorry

theorem whipped_cream_theorem (schedule : BakingSchedule) (requirement : WhippedCreamRequirement) 
                               (tiffanyEats : TiffanyEats) : 
  schedule = {
    odd_pumpkin := 3, odd_apple := 2, odd_chocolate := 1,
    even_pumpkin := 2, even_apple := 4, even_chocolate := 2, even_lemon := 1
  } →
  requirement = {
    pumpkin := 2, apple := 1, chocolate := 3, lemon := 3/2
  } →
  tiffanyEats = {
    pumpkin := 2, apple := 5, chocolate := 1, lemon := 1
  } →
  whippedCreamNeeded schedule requirement tiffanyEats = 252 :=
by
  sorry


end NUMINAMATH_CALUDE_whipped_cream_theorem_l2964_296459


namespace NUMINAMATH_CALUDE_problem_solution_l2964_296423

def A (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2*m + 3}

def B : Set ℝ := {x | -x^2 + 2*x + 8 > 0}

theorem problem_solution :
  (∀ m : ℝ, 
    (m = 2 → A m ∪ B = {x | -2 < x ∧ x ≤ 7}) ∧
    (m = 2 → (Set.univ \ A m) ∩ B = {x | -2 < x ∧ x < 1})) ∧
  (∀ m : ℝ, A m ∩ B = A m ↔ m < -4 ∨ (-1 < m ∧ m < 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2964_296423


namespace NUMINAMATH_CALUDE_systematic_sampling_distance_l2964_296448

/-- Calculates the sampling distance for systematic sampling -/
def sampling_distance (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

theorem systematic_sampling_distance :
  let population : ℕ := 1200
  let sample_size : ℕ := 30
  sampling_distance population sample_size = 40 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_distance_l2964_296448


namespace NUMINAMATH_CALUDE_shop_length_is_20_l2964_296494

/-- Calculates the length of a shop given its monthly rent, width, and annual rent per square foot. -/
def shop_length (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ) : ℕ :=
  let annual_rent := monthly_rent * 12
  let total_sqft := annual_rent / annual_rent_per_sqft
  total_sqft / width

/-- Theorem stating that for a shop with given parameters, its length is 20 feet. -/
theorem shop_length_is_20 :
  shop_length 3600 15 144 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shop_length_is_20_l2964_296494


namespace NUMINAMATH_CALUDE_alice_most_dogs_l2964_296498

-- Define the number of cats and dogs for each person
variable (Kc Ac Bc Kd Ad Bd : ℕ)

-- Define the conditions
variable (h1 : Kc > Ac)  -- Kathy owns more cats than Alice
variable (h2 : Kd > Bd)  -- Kathy owns more dogs than Bruce
variable (h3 : Ad > Kd)  -- Alice owns more dogs than Kathy
variable (h4 : Bc > Ac)  -- Bruce owns more cats than Alice

-- Theorem statement
theorem alice_most_dogs : Ad > Kd ∧ Ad > Bd := by
  sorry

end NUMINAMATH_CALUDE_alice_most_dogs_l2964_296498


namespace NUMINAMATH_CALUDE_even_increasing_function_inequality_l2964_296483

/-- An even function that is monotonically increasing on the positive reals -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y)

/-- Theorem: For an even function that is monotonically increasing on the positive reals,
    f(-3) > f(2) > f(-1) -/
theorem even_increasing_function_inequality (f : ℝ → ℝ) 
  (hf : EvenIncreasingFunction f) : f (-3) > f 2 ∧ f 2 > f (-1) := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_function_inequality_l2964_296483


namespace NUMINAMATH_CALUDE_fraction_equality_l2964_296445

theorem fraction_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let x := (1/2) * (Real.sqrt (a/b) - Real.sqrt (b/a))
  (2*a * Real.sqrt (1 + x^2)) / (x + Real.sqrt (1 + x^2)) = a + b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2964_296445


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l2964_296431

theorem isosceles_triangle_largest_angle (α β γ : ℝ) :
  -- The triangle is isosceles with two equal angles
  α = β ∧
  -- One of the equal angles is 30°
  α = 30 ∧
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 120°
  max α (max β γ) = 120 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l2964_296431


namespace NUMINAMATH_CALUDE_find_number_B_l2964_296411

/-- Given that A = 5 and A = 2.8B - 0.6, prove that B = 2 -/
theorem find_number_B (A B : ℝ) (h1 : A = 5) (h2 : A = 2.8 * B - 0.6) : B = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_number_B_l2964_296411


namespace NUMINAMATH_CALUDE_orange_cost_calculation_l2964_296470

theorem orange_cost_calculation (cost_three_dozen : ℝ) (dozen_count : ℕ) :
  cost_three_dozen = 22.5 →
  dozen_count = 4 →
  (cost_three_dozen / 3) * dozen_count = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_orange_cost_calculation_l2964_296470


namespace NUMINAMATH_CALUDE_wrong_number_correction_l2964_296403

theorem wrong_number_correction (n : ℕ) (initial_avg correct_avg : ℚ) 
  (first_error second_correct : ℚ) :
  n = 10 ∧ 
  initial_avg = 40.2 ∧ 
  correct_avg = 40.1 ∧ 
  first_error = 17 ∧ 
  second_correct = 31 →
  ∃ second_error : ℚ,
    n * initial_avg - first_error - second_error + second_correct = n * correct_avg ∧
    second_error = 15 :=
by sorry

end NUMINAMATH_CALUDE_wrong_number_correction_l2964_296403


namespace NUMINAMATH_CALUDE_min_surface_area_angle_l2964_296480

/-- The angle that minimizes the surface area of a rotated right triangle -/
theorem min_surface_area_angle (AC BC CD : ℝ) (h1 : AC = 3) (h2 : BC = 4) (h3 : CD = 10) :
  let α := Real.arctan (2 / 3)
  let surface_area (θ : ℝ) := π * (240 - 12 * (2 * Real.sin θ + 3 * Real.cos θ))
  ∀ θ, surface_area α ≤ surface_area θ := by
sorry


end NUMINAMATH_CALUDE_min_surface_area_angle_l2964_296480


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2964_296419

/-- A quadratic function with vertex (2, -1) passing through (-1, -16) has a = -5/3 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, (a * x^2 + b * x + c) = a * (x - 2)^2 - 1) →  -- vertex form
  (a * (-1)^2 + b * (-1) + c = -16) →                 -- passes through (-1, -16)
  a = -5/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2964_296419


namespace NUMINAMATH_CALUDE_adam_book_spending_l2964_296486

theorem adam_book_spending :
  ∀ (initial_amount spent_amount : ℝ),
    initial_amount = 91 →
    (initial_amount - spent_amount) / spent_amount = 10 / 3 →
    spent_amount = 21 := by
  sorry

end NUMINAMATH_CALUDE_adam_book_spending_l2964_296486


namespace NUMINAMATH_CALUDE_classroom_count_l2964_296438

/-- Given a classroom with a 1:2 ratio of girls to boys and 20 boys, prove the total number of students is 30. -/
theorem classroom_count (num_boys : ℕ) (ratio_girls_to_boys : ℚ) : 
  num_boys = 20 → ratio_girls_to_boys = 1/2 → num_boys + (ratio_girls_to_boys * num_boys) = 30 := by
  sorry

end NUMINAMATH_CALUDE_classroom_count_l2964_296438


namespace NUMINAMATH_CALUDE_no_two_digit_integer_satisfies_conditions_l2964_296467

theorem no_two_digit_integer_satisfies_conditions : 
  ∀ n : ℕ, 10 ≤ n → n < 100 → 
  ¬(∃ (a b : ℕ), n = 10 * a + b ∧ a < 10 ∧ b < 10 ∧ 
    (n % (a + b) = 0) ∧ (n % (a^2 * b) = 0)) := by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_integer_satisfies_conditions_l2964_296467


namespace NUMINAMATH_CALUDE_cheaper_to_buy_more_count_l2964_296433

-- Define the cost function C(n)
def C (n : ℕ) : ℕ :=
  if n ≤ 30 then 15 * n
  else if n ≤ 60 then 13 * n
  else 12 * n

-- Define a function that checks if buying n+1 books is cheaper than n books
def isCheaperToBuyMore (n : ℕ) : Prop :=
  C (n + 1) < C n

-- Theorem statement
theorem cheaper_to_buy_more_count :
  (∃ (S : Finset ℕ), S.card = 5 ∧ (∀ n, n ∈ S ↔ isCheaperToBuyMore n)) :=
by sorry

end NUMINAMATH_CALUDE_cheaper_to_buy_more_count_l2964_296433


namespace NUMINAMATH_CALUDE_unique_cube_root_property_l2964_296482

theorem unique_cube_root_property : ∃! (n : ℕ), n > 0 ∧ (∃ (a b : ℕ), 
  n = 1000 * a + b ∧ 
  b < 1000 ∧ 
  a^3 = n ∧ 
  a = n / 1000) :=
by sorry

end NUMINAMATH_CALUDE_unique_cube_root_property_l2964_296482


namespace NUMINAMATH_CALUDE_julia_watch_collection_l2964_296485

theorem julia_watch_collection (silver : ℕ) (bronze : ℕ) (gold : ℕ) : 
  silver = 20 →
  bronze = 3 * silver →
  gold = (silver + bronze) / 10 →
  silver + bronze + gold = 88 :=
by
  sorry

end NUMINAMATH_CALUDE_julia_watch_collection_l2964_296485


namespace NUMINAMATH_CALUDE_factor_expression_l2964_296478

theorem factor_expression :
  ∀ x : ℝ, 63 * x^2 + 42 = 21 * (3 * x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2964_296478


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_l2964_296402

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to seat 9 people in a row with the given constraint. -/
def seating_arrangements : ℕ :=
  factorial 9 - factorial 7 * factorial 3

/-- Theorem stating the number of valid seating arrangements. -/
theorem valid_seating_arrangements :
  seating_arrangements = 332640 := by sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_l2964_296402


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2964_296463

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- Given a_13 = S_13 = 13, prove a_1 = -11 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 13 = 13) (h2 : seq.S 13 = 13) : seq.a 1 = -11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2964_296463


namespace NUMINAMATH_CALUDE_linear_equation_solutions_l2964_296432

theorem linear_equation_solutions : 
  {(x, y) : ℕ × ℕ | 5 * x + 2 * y = 25 ∧ x > 0 ∧ y > 0} = {(1, 10), (3, 5)} := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solutions_l2964_296432


namespace NUMINAMATH_CALUDE_sin_plus_two_cos_equals_neg_two_fifths_l2964_296412

/-- Given a point P(-3,4) on the terminal side of angle θ, prove that sin θ + 2cos θ = -2/5 -/
theorem sin_plus_two_cos_equals_neg_two_fifths (θ : ℝ) (P : ℝ × ℝ) :
  P = (-3, 4) →
  (∃ r : ℝ, r > 0 ∧ r * (Real.cos θ) = -3 ∧ r * (Real.sin θ) = 4) →
  Real.sin θ + 2 * Real.cos θ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_two_cos_equals_neg_two_fifths_l2964_296412


namespace NUMINAMATH_CALUDE_equation_equivalence_and_product_l2964_296472

theorem equation_equivalence_and_product (a c x y : ℝ) :
  ∃ (r s t u : ℤ),
    ((a^r * x - a^s) * (a^t * y - a^u) = a^6 * c^3) ↔
    (a^10 * x * y - a^8 * y - a^7 * x = a^6 * (c^3 - 1)) ∧
    r * s * t * u = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_and_product_l2964_296472


namespace NUMINAMATH_CALUDE_simplify_expression_l2964_296415

theorem simplify_expression (m n : ℝ) : (8*m - 7*n) - 2*(m - 3*n) = 6*m - n := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2964_296415


namespace NUMINAMATH_CALUDE_range_theorem_fixed_point_theorem_l2964_296453

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y = 0

-- Define the range function
def in_range (z : ℝ) : Prop := (6 - 2*Real.sqrt 3) / 3 ≤ z ∧ z ≤ (6 + 2*Real.sqrt 3) / 3

-- Theorem 1: Range of (y+3)/x for points on circle C
theorem range_theorem (x y : ℝ) : 
  circle_C x y → in_range ((y + 3) / x) :=
sorry

-- Define a point on line l
def point_on_line_l (t : ℝ) : ℝ × ℝ := (t, 2*t)

-- Define the circle passing through P, A, C, and B
def circle_PACB (t x y : ℝ) : Prop :=
  (x - (t + 2) / 2)^2 + (y - t)^2 = (5*t^2 - 4*t + 4) / 4

-- Theorem 2: Circle PACB passes through (2/5, 4/5)
theorem fixed_point_theorem (t : ℝ) :
  circle_PACB t (2/5) (4/5) :=
sorry

end NUMINAMATH_CALUDE_range_theorem_fixed_point_theorem_l2964_296453


namespace NUMINAMATH_CALUDE_f_monotonicity_and_properties_l2964_296496

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / a + a / x

theorem f_monotonicity_and_properties :
  ∀ a : ℝ, ∀ x : ℝ, x > 0 →
  (a > 0 → (
    (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → f a x₁ > f a x₂)
  )) ∧
  (a < 0 → (
    (∀ x₁ x₂, -a < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < -a → f a x₁ < f a x₂)
  )) ∧
  (a = 1/2 → (
    ∀ x₀, x₀ > 0 →
    (2 - 1 / (2 * x₀^2) = 3/2 →
      ∀ x y, 3*x - 2*y + 2 = 0 ↔ y - f (1/2) x₀ = 3/2 * (x - x₀))
  )) ∧
  (a = 1/2 → ∀ x, x > 0 → f (1/2) x > Real.log x + x/2) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_properties_l2964_296496


namespace NUMINAMATH_CALUDE_clay_capacity_scaling_l2964_296422

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  depth : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.depth * d.width * d.length

/-- Theorem: Given a box with dimensions 3x4x6 cm holding 60g of clay,
    a box with dimensions 9x16x6 cm will hold 720g of clay -/
theorem clay_capacity_scaling (clayMass₁ : ℝ) :
  let box₁ : BoxDimensions := ⟨3, 4, 6⟩
  let box₂ : BoxDimensions := ⟨9, 16, 6⟩
  clayMass₁ = 60 →
  (boxVolume box₂ / boxVolume box₁) * clayMass₁ = 720 := by
  sorry

end NUMINAMATH_CALUDE_clay_capacity_scaling_l2964_296422


namespace NUMINAMATH_CALUDE_product_xyz_equals_one_l2964_296490

theorem product_xyz_equals_one (x y z : ℝ) 
  (eq1 : x + 1/y = 2) 
  (eq2 : y + 1/z = 2) 
  (eq3 : z + 1/x = 2) : 
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_one_l2964_296490


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_47_plus_one_l2964_296435

theorem gcd_of_powers_of_47_plus_one (h : Prime 47) :
  Nat.gcd (47^5 + 1) (47^5 + 47^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_47_plus_one_l2964_296435


namespace NUMINAMATH_CALUDE_circle_tangents_l2964_296484

-- Define the circles
def circle_C (m : ℝ) (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8 - m
def circle_D (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 1

-- Define the property of having three common tangents
def has_three_common_tangents (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (circle_C m x1 y1 ∧ circle_D x1 y1) ∧
    (circle_C m x2 y2 ∧ circle_D x2 y2) ∧
    (circle_C m x3 y3 ∧ circle_D x3 y3) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3)

-- Theorem statement
theorem circle_tangents (m : ℝ) :
  has_three_common_tangents m → m = -8 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangents_l2964_296484


namespace NUMINAMATH_CALUDE_calculation_result_l2964_296462

theorem calculation_result : 50 + 50 / 50 + 50 = 101 := by sorry

end NUMINAMATH_CALUDE_calculation_result_l2964_296462


namespace NUMINAMATH_CALUDE_matrix_sum_equals_result_l2964_296468

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 0; 1, 2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-5, -7; 4, -9]

theorem matrix_sum_equals_result : A + B = !![(-2), (-7); 5, (-7)] := by sorry

end NUMINAMATH_CALUDE_matrix_sum_equals_result_l2964_296468


namespace NUMINAMATH_CALUDE_divisor_sum_of_2_3_power_l2964_296481

/-- Sum of positive divisors of n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Sum of geometric series -/
def geometric_sum (a r : ℕ) (n : ℕ) : ℕ := sorry

theorem divisor_sum_of_2_3_power (i j : ℕ) :
  sum_of_divisors (2^i * 3^j) = 540 → i + j = 5 := by sorry

end NUMINAMATH_CALUDE_divisor_sum_of_2_3_power_l2964_296481


namespace NUMINAMATH_CALUDE_quadratic_equation_has_solution_l2964_296454

theorem quadratic_equation_has_solution (a b : ℝ) :
  ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_has_solution_l2964_296454


namespace NUMINAMATH_CALUDE_max_non_managers_l2964_296407

/-- Represents the number of managers in a department -/
def managers : ℕ := 11

/-- Represents the ratio of managers to non-managers -/
def ratio : ℚ := 7 / 37

/-- Theorem stating the maximum number of non-managers in a department -/
theorem max_non_managers :
  ∀ n : ℕ, (managers : ℚ) / n > ratio → n ≤ 58 :=
sorry

end NUMINAMATH_CALUDE_max_non_managers_l2964_296407


namespace NUMINAMATH_CALUDE_log_relation_l2964_296460

theorem log_relation (a b : ℝ) (h1 : a = Real.log 625 / Real.log 4) (h2 : b = Real.log 25 / Real.log 5) :
  a = 4 / b := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l2964_296460


namespace NUMINAMATH_CALUDE_petya_wins_l2964_296449

/-- Represents the state of the game -/
structure GameState :=
  (contacts : Nat)
  (wires : Nat)
  (player_turn : Bool)

/-- The initial game state -/
def initial_state : GameState :=
  { contacts := 2000
  , wires := 2000 * 1999 / 2
  , player_turn := true }

/-- Represents a move in the game -/
inductive Move
  | cut_one
  | cut_three

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.cut_one => 
      { state with 
        wires := state.wires - 1
        player_turn := ¬state.player_turn }
  | Move.cut_three => 
      { state with 
        wires := state.wires - 3
        player_turn := ¬state.player_turn }

/-- Checks if the game is over -/
def is_game_over (state : GameState) : Bool :=
  state.wires < state.contacts

/-- Theorem: Player 2 (Petya) has a winning strategy -/
theorem petya_wins : 
  ∃ (strategy : GameState → Move), 
    ∀ (game : List Move), 
      is_game_over (List.foldl apply_move initial_state game) → 
        (List.length game % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_petya_wins_l2964_296449


namespace NUMINAMATH_CALUDE_square_field_area_l2964_296455

/-- Given a square field where a horse takes 4 hours to run around it at 20 km/h, 
    prove that the area of the field is 400 km². -/
theorem square_field_area (s : ℝ) (h : s > 0) : 
  (4 * s = 20 * 4) → s^2 = 400 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l2964_296455


namespace NUMINAMATH_CALUDE_trigonometric_sum_trigonometric_fraction_l2964_296429

-- Part 1
theorem trigonometric_sum : 
  Real.cos (9 * Real.pi / 4) + Real.tan (-Real.pi / 4) + Real.sin (21 * Real.pi) = Real.sqrt 2 / 2 - 1 :=
by sorry

-- Part 2
theorem trigonometric_fraction (θ : Real) (h : Real.sin θ = 2 * Real.cos θ) : 
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_sum_trigonometric_fraction_l2964_296429


namespace NUMINAMATH_CALUDE_multiplication_factor_l2964_296473

theorem multiplication_factor (N : ℝ) (h : N ≠ 0) : 
  let X : ℝ := 5
  let incorrect_value := N / 10
  let correct_value := N * X
  let percentage_error := |correct_value - incorrect_value| / correct_value * 100
  percentage_error = 98 := by sorry

end NUMINAMATH_CALUDE_multiplication_factor_l2964_296473


namespace NUMINAMATH_CALUDE_sum_of_new_observations_l2964_296417

/-- Given 10 observations with an average of 21, prove that adding two new observations
    that increase the average by 2 results in the sum of the two new observations being 66. -/
theorem sum_of_new_observations (initial_count : Nat) (initial_avg : ℝ) (new_count : Nat) (avg_increase : ℝ) :
  initial_count = 10 →
  initial_avg = 21 →
  new_count = initial_count + 2 →
  avg_increase = 2 →
  (new_count : ℝ) * (initial_avg + avg_increase) - (initial_count : ℝ) * initial_avg = 66 := by
  sorry

#check sum_of_new_observations

end NUMINAMATH_CALUDE_sum_of_new_observations_l2964_296417


namespace NUMINAMATH_CALUDE_ratio_simplification_l2964_296458

theorem ratio_simplification (A B C : ℚ) : 
  (A / B = 5 / 3 / (29 / 6)) → 
  (C / A = (11 / 5) / (11 / 3)) → 
  ∃ (k : ℚ), k * A = 10 ∧ k * B = 29 ∧ k * C = 6 :=
by sorry

end NUMINAMATH_CALUDE_ratio_simplification_l2964_296458


namespace NUMINAMATH_CALUDE_fraction_equality_l2964_296461

theorem fraction_equality (x y p q : ℚ) : 
  (7 * x + 6 * y) / (x - 2 * y) = 27 → x / (2 * y) = p / q → p / q = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2964_296461


namespace NUMINAMATH_CALUDE_product_abcd_l2964_296424

theorem product_abcd : 
  ∀ (a b c d : ℚ),
  (3 * a + 2 * b + 4 * c + 6 * d = 36) →
  (4 * (d + c) = b) →
  (4 * b + 2 * c = a) →
  (c - 2 = d) →
  (a * b * c * d = -315/32) :=
by
  sorry

end NUMINAMATH_CALUDE_product_abcd_l2964_296424


namespace NUMINAMATH_CALUDE_swimming_speed_is_10_l2964_296441

/-- The swimming speed of a person in still water. -/
def swimming_speed : ℝ := 10

/-- The speed of the water current. -/
def water_speed : ℝ := 8

/-- The time taken to swim against the current. -/
def swim_time : ℝ := 8

/-- The distance swam against the current. -/
def swim_distance : ℝ := 16

/-- Theorem stating that the swimming speed in still water is 10 km/h given the conditions. -/
theorem swimming_speed_is_10 :
  swimming_speed = 10 ∧
  water_speed = 8 ∧
  swim_time = 8 ∧
  swim_distance = 16 ∧
  swim_distance = (swimming_speed - water_speed) * swim_time :=
by sorry

end NUMINAMATH_CALUDE_swimming_speed_is_10_l2964_296441


namespace NUMINAMATH_CALUDE_divisibility_implies_lower_bound_l2964_296442

theorem divisibility_implies_lower_bound (n a : ℕ) 
  (h1 : n > 1) 
  (h2 : a > n^2) 
  (h3 : ∀ i ∈ Finset.range n, ∃ x ∈ Finset.range n, (n^2 + i + 1) ∣ (a + x + 1)) : 
  a > n^4 - n^3 := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_lower_bound_l2964_296442


namespace NUMINAMATH_CALUDE_alcohol_fraction_in_mixture_l2964_296405

theorem alcohol_fraction_in_mixture (alcohol_water_ratio : ℚ) :
  alcohol_water_ratio = 2 / 3 →
  (alcohol_water_ratio / (1 + alcohol_water_ratio)) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_fraction_in_mixture_l2964_296405


namespace NUMINAMATH_CALUDE_original_number_proof_l2964_296479

theorem original_number_proof : ∃! N : ℕ, N > 0 ∧ (N - 5) % 13 = 0 ∧ ∀ M : ℕ, M > 0 → (M - 5) % 13 = 0 → M ≥ N :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l2964_296479


namespace NUMINAMATH_CALUDE_lady_bird_flour_theorem_l2964_296437

/-- The amount of flour needed for a given number of guests at Lady Bird's Junior League club meeting -/
def flour_needed (guests : ℕ) : ℚ :=
  let biscuits_per_guest : ℕ := 2
  let biscuits_per_batch : ℕ := 9
  let flour_per_batch : ℚ := 5 / 4
  let total_biscuits : ℕ := guests * biscuits_per_guest
  let batches : ℕ := (total_biscuits + biscuits_per_batch - 1) / biscuits_per_batch
  (batches : ℚ) * flour_per_batch

/-- Theorem stating that Lady Bird needs 5 cups of flour for 18 guests -/
theorem lady_bird_flour_theorem :
  flour_needed 18 = 5 := by
  sorry

end NUMINAMATH_CALUDE_lady_bird_flour_theorem_l2964_296437


namespace NUMINAMATH_CALUDE_margarets_mean_score_l2964_296426

def scores : List ℝ := [82, 85, 89, 91, 95, 97]

theorem margarets_mean_score (cyprians_mean : ℝ) (h1 : cyprians_mean = 88) :
  let total_sum := scores.sum
  let cyprians_sum := 3 * cyprians_mean
  let margarets_sum := total_sum - cyprians_sum
  margarets_sum / 3 = 91 + 2/3 := by sorry

end NUMINAMATH_CALUDE_margarets_mean_score_l2964_296426


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2964_296457

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_properties
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = -3)
  (h_S5 : S a 5 = 0) :
  (∀ n : ℕ, a n = (3 * (3 - n : ℚ)) / 2) ∧
  (∀ n : ℕ, a n * S a n < 0 ↔ n = 4) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2964_296457


namespace NUMINAMATH_CALUDE_p_false_and_q_true_l2964_296465

-- Define proposition p
def p : Prop := ∀ x > 0, 3^x > 1

-- Define proposition q
def q : Prop := ∀ a, (a < -2 → ∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) ∧
                    (∃ b, b ≥ -2 ∧ ∃ x ∈ Set.Icc (-1) 2, b * x + 3 = 0)

-- Theorem stating that p is false and q is true
theorem p_false_and_q_true : ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_p_false_and_q_true_l2964_296465


namespace NUMINAMATH_CALUDE_xy_value_l2964_296464

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x + y) = 32)
  (h2 : (16 : ℝ)^(x + y) / (4 : ℝ)^(3 * y) = 256) : 
  x * y = -2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2964_296464


namespace NUMINAMATH_CALUDE_division_sum_theorem_l2964_296425

theorem division_sum_theorem (quotient divisor remainder : ℕ) : 
  quotient = 120 → divisor = 456 → remainder = 333 → 
  (divisor * quotient + remainder = 55053) := by sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l2964_296425


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2964_296492

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | -1 < x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2964_296492


namespace NUMINAMATH_CALUDE_third_number_in_sum_l2964_296477

theorem third_number_in_sum (a b c : ℝ) (h1 : a = 3.15) (h2 : b = 0.014) (h3 : a + b + c = 3.622) : c = 0.458 := by
  sorry

end NUMINAMATH_CALUDE_third_number_in_sum_l2964_296477


namespace NUMINAMATH_CALUDE_max_non_multiples_of_three_l2964_296420

/-- Given a list of 6 positive integers whose product is a multiple of 3,
    the maximum number of integers in the list that are not multiples of 3 is 5. -/
theorem max_non_multiples_of_three (integers : List ℕ+) : 
  integers.length = 6 → 
  integers.prod.val % 3 = 0 → 
  (integers.filter (fun x => x.val % 3 ≠ 0)).length ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_non_multiples_of_three_l2964_296420


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l2964_296416

def f (x : ℝ) : ℝ := -3 * (x - 2)^2 + 12

theorem quadratic_function_proof :
  (∀ x, f x > 0 ↔ 0 < x ∧ x < 4) ∧
  (∀ x ∈ Set.Icc (-1) 5, f x ≤ 12) ∧
  (∃ x ∈ Set.Icc (-1) 5, f x = 12) →
  ∀ x, f x = -3 * (x - 2)^2 + 12 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l2964_296416


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l2964_296418

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/2)^(x-2)

theorem zero_point_in_interval :
  ∃ x₀ ∈ Set.Ioo 2 3, f x₀ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l2964_296418


namespace NUMINAMATH_CALUDE_soda_price_increase_l2964_296471

theorem soda_price_increase (candy_new : ℝ) (soda_new : ℝ) (candy_increase : ℝ) (total_old : ℝ)
  (h1 : candy_new = 20)
  (h2 : soda_new = 6)
  (h3 : candy_increase = 0.25)
  (h4 : total_old = 20) :
  (soda_new - (total_old - candy_new / (1 + candy_increase))) / (total_old - candy_new / (1 + candy_increase)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_increase_l2964_296471


namespace NUMINAMATH_CALUDE_true_propositions_l2964_296409

-- Define the four propositions
def proposition1 : Prop := sorry
def proposition2 : Prop := sorry
def proposition3 : Prop := sorry
def proposition4 : Prop := sorry

-- Theorem stating which propositions are true
theorem true_propositions : 
  (¬ proposition1) ∧ proposition2 ∧ proposition3 ∧ (¬ proposition4) := by
  sorry

end NUMINAMATH_CALUDE_true_propositions_l2964_296409


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2964_296414

/-- Given a train and bridge scenario, calculate the train's speed in km/h -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (time : ℝ) 
  (h1 : train_length = 360) 
  (h2 : bridge_length = 140) 
  (h3 : time = 40) : 
  (train_length + bridge_length) / time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2964_296414
