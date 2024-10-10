import Mathlib

namespace max_abc_value_l3567_356742

theorem max_abc_value (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a * b + b * c + a * c = 1) : 
  a * b * c ≤ Real.sqrt 3 / 9 := by
sorry

end max_abc_value_l3567_356742


namespace special_function_sum_negative_l3567_356765

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  odd : ∀ x, f (x + 2) = -f (-x + 2)
  mono : ∀ x y, x > 2 → y > 2 → x < y → f x < f y

/-- The main theorem -/
theorem special_function_sum_negative (F : SpecialFunction) 
  (x₁ x₂ : ℝ) (h1 : x₁ + x₂ < 4) (h2 : (x₁ - 2) * (x₂ - 2) < 0) :
  F.f x₁ + F.f x₂ < 0 := by
  sorry

end special_function_sum_negative_l3567_356765


namespace smallest_positive_difference_l3567_356718

/-- Vovochka's addition method for three-digit numbers -/
def vovochkaAdd (a b c d e f : ℕ) : ℕ :=
  1000 * (a + d) + 100 * (b + e) + (c + f)

/-- Regular addition for three-digit numbers -/
def regularAdd (a b c d e f : ℕ) : ℕ :=
  100 * (a + d) + 10 * (b + e) + (c + f)

/-- The difference between Vovochka's addition and regular addition -/
def addDifference (a b c d e f : ℕ) : ℤ :=
  (vovochkaAdd a b c d e f : ℤ) - (regularAdd a b c d e f : ℤ)

/-- Theorem: The smallest positive difference between Vovochka's addition and regular addition is 1800 -/
theorem smallest_positive_difference :
  ∃ (a b c d e f : ℕ),
    (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10) ∧
    (a + d > 0) ∧
    (addDifference a b c d e f > 0) ∧
    (∀ (x y z u v w : ℕ),
      (x < 10 ∧ y < 10 ∧ z < 10 ∧ u < 10 ∧ v < 10 ∧ w < 10) →
      (x + u > 0) →
      (addDifference x y z u v w > 0) →
      (addDifference a b c d e f ≤ addDifference x y z u v w)) ∧
    (addDifference a b c d e f = 1800) :=
  sorry

end smallest_positive_difference_l3567_356718


namespace gary_gold_amount_l3567_356741

/-- Proves that Gary has 30 grams of gold given the conditions of the problem -/
theorem gary_gold_amount (gary_cost_per_gram : ℝ) (anna_amount : ℝ) (anna_cost_per_gram : ℝ) (total_cost : ℝ)
  (h1 : gary_cost_per_gram = 15)
  (h2 : anna_amount = 50)
  (h3 : anna_cost_per_gram = 20)
  (h4 : total_cost = 1450)
  (h5 : gary_cost_per_gram * gary_amount + anna_amount * anna_cost_per_gram = total_cost) :
  gary_amount = 30 := by
  sorry

end gary_gold_amount_l3567_356741


namespace solution_x_l3567_356737

theorem solution_x (x : Real) 
  (h1 : x ∈ Set.Ioo 0 (π / 2))
  (h2 : 1 / Real.sin x = 1 / Real.sin (2 * x) + 1 / Real.sin (4 * x) + 1 / Real.sin (8 * x)) :
  x = π / 15 ∨ x = π / 5 ∨ x = π / 3 ∨ x = 7 * π / 15 := by
  sorry

end solution_x_l3567_356737


namespace adam_laundry_theorem_l3567_356760

/-- Given a total number of laundry loads and the number of loads already washed,
    calculate the number of loads still to be washed. -/
def loads_remaining (total : ℕ) (washed : ℕ) : ℕ :=
  total - washed

/-- Theorem: Given 25 total loads and 6 washed loads, 19 loads remain to be washed. -/
theorem adam_laundry_theorem :
  loads_remaining 25 6 = 19 := by
  sorry

end adam_laundry_theorem_l3567_356760


namespace lee_cookies_l3567_356794

/-- Given that Lee can make 24 cookies with 3 cups of flour,
    this function calculates how many cookies he can make with any amount of flour. -/
def cookies_from_flour (flour : ℚ) : ℚ :=
  (24 * flour) / 3

/-- Theorem stating that Lee can make 40 cookies with 5 cups of flour. -/
theorem lee_cookies : cookies_from_flour 5 = 40 := by
  sorry

end lee_cookies_l3567_356794


namespace simplify_and_sum_exponents_l3567_356746

def simplify_cube_root (x y z : ℝ) : ℝ := (40 * x^5 * y^9 * z^14) ^ (1/3)

theorem simplify_and_sum_exponents (x y z : ℝ) :
  ∃ (a e : ℝ) (b c d f g h : ℕ),
    simplify_cube_root x y z = a * x^b * y^c * z^d * (e * x^f * y^g * z^h)^(1/3) ∧
    b + c + d = 5 :=
by sorry

end simplify_and_sum_exponents_l3567_356746


namespace probNotAllSame_l3567_356725

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The probability that all dice show the same number -/
def probAllSame : ℚ := 1 / numSides^(numDice - 1)

/-- The probability that not all dice show the same number -/
theorem probNotAllSame : (1 : ℚ) - probAllSame = 215 / 216 := by sorry

end probNotAllSame_l3567_356725


namespace integer_roots_of_polynomial_l3567_356758

def polynomial (x : ℤ) : ℤ := x^4 + 4*x^3 - x^2 + 3*x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial :
  {x : ℤ | polynomial x = 0} = possible_roots := by sorry

end integer_roots_of_polynomial_l3567_356758


namespace train_crossing_bridge_time_l3567_356723

/-- Time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120) 
  (h2 : train_speed_kmh = 60) 
  (h3 : bridge_length = 170) : 
  ∃ (time : ℝ), abs (time - 17.40) < 0.01 :=
by
  sorry

end train_crossing_bridge_time_l3567_356723


namespace spinner_prime_sum_probability_l3567_356730

-- Define the spinners
def spinner1 : List ℕ := [1, 2, 3, 4]
def spinner2 : List ℕ := [3, 4, 5, 6]

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Bool := sorry

-- Define a function to calculate all possible sums
def allSums (s1 s2 : List ℕ) : List ℕ := sorry

-- Define a function to count prime sums
def countPrimeSums (sums : List ℕ) : ℕ := sorry

-- Theorem to prove
theorem spinner_prime_sum_probability :
  let sums := allSums spinner1 spinner2
  let primeCount := countPrimeSums sums
  let totalCount := spinner1.length * spinner2.length
  (primeCount : ℚ) / totalCount = 5 / 16 := by sorry

end spinner_prime_sum_probability_l3567_356730


namespace stamp_collection_value_l3567_356795

theorem stamp_collection_value 
  (total_stamps : ℕ) 
  (sample_stamps : ℕ) 
  (sample_value : ℝ) 
  (h1 : total_stamps = 20)
  (h2 : sample_stamps = 4)
  (h3 : sample_value = 16) :
  (total_stamps : ℝ) * (sample_value / sample_stamps) = 80 :=
by sorry

end stamp_collection_value_l3567_356795


namespace harry_pizza_order_cost_l3567_356756

/-- Calculates the total cost of Harry's pizza order -/
theorem harry_pizza_order_cost :
  let large_pizza_cost : ℚ := 14
  let topping_cost : ℚ := 2
  let num_pizzas : ℕ := 2
  let num_toppings : ℕ := 3
  let tip_percentage : ℚ := 25 / 100

  let pizza_with_toppings_cost := large_pizza_cost + num_toppings * topping_cost
  let total_pizza_cost := num_pizzas * pizza_with_toppings_cost
  let tip_amount := tip_percentage * total_pizza_cost
  let total_cost := total_pizza_cost + tip_amount

  total_cost = 50 := by sorry

end harry_pizza_order_cost_l3567_356756


namespace zoe_yogurt_consumption_l3567_356745

/-- Calculates the number of ounces of yogurt Zoe ate given the following conditions:
  * Zoe ate 12 strawberries and some ounces of yogurt
  * Strawberries have 4 calories each
  * Yogurt has 17 calories per ounce
  * Zoe ate a total of 150 calories
-/
theorem zoe_yogurt_consumption (
  strawberry_count : ℕ)
  (strawberry_calories : ℕ)
  (yogurt_calories_per_ounce : ℕ)
  (total_calories : ℕ)
  (h1 : strawberry_count = 12)
  (h2 : strawberry_calories = 4)
  (h3 : yogurt_calories_per_ounce = 17)
  (h4 : total_calories = 150)
  : (total_calories - strawberry_count * strawberry_calories) / yogurt_calories_per_ounce = 6 := by
  sorry

end zoe_yogurt_consumption_l3567_356745


namespace max_value_of_cosine_function_l3567_356766

theorem max_value_of_cosine_function :
  ∀ x : ℝ, 4 * (Real.cos x)^3 - 3 * (Real.cos x)^2 - 6 * (Real.cos x) + 5 ≤ 27/4 := by
  sorry

end max_value_of_cosine_function_l3567_356766


namespace purchase_price_is_31_l3567_356753

/-- Calculates the purchase price of a share given the dividend rate, face value, and return on investment. -/
def calculate_purchase_price (dividend_rate : ℚ) (face_value : ℚ) (roi : ℚ) : ℚ :=
  (dividend_rate * face_value) / roi

/-- Theorem stating that given the specific conditions, the purchase price is 31. -/
theorem purchase_price_is_31 :
  let dividend_rate : ℚ := 155 / 1000  -- 15.5%
  let face_value : ℚ := 50
  let roi : ℚ := 1 / 4  -- 25%
  calculate_purchase_price dividend_rate face_value roi = 31 := by
  sorry

#eval calculate_purchase_price (155 / 1000) 50 (1 / 4)

end purchase_price_is_31_l3567_356753


namespace function_comparison_l3567_356761

/-- A function f is strictly decreasing on the non-negative real numbers -/
def StrictlyDecreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₂ < f x₁

/-- An even function -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem function_comparison 
  (f : ℝ → ℝ) 
  (h_even : EvenFunction f)
  (h_decreasing : StrictlyDecreasingOnNonnegative f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 :=
sorry

end function_comparison_l3567_356761


namespace paint_usage_l3567_356754

def paint_problem (paint_large : ℕ) (paint_small : ℕ) (num_large : ℕ) (num_small : ℕ) : Prop :=
  paint_large * num_large + paint_small * num_small = 17

theorem paint_usage : paint_problem 3 2 3 4 := by
  sorry

end paint_usage_l3567_356754


namespace bobby_total_consumption_l3567_356706

def bobby_consumption (initial_candy : ℕ) (additional_candy : ℕ) (candy_fraction : ℚ) 
                      (chocolate : ℕ) (chocolate_fraction : ℚ) : ℚ :=
  initial_candy + candy_fraction * additional_candy + chocolate_fraction * chocolate

theorem bobby_total_consumption : 
  bobby_consumption 28 42 (3/4) 63 (1/2) = 91 := by
  sorry

end bobby_total_consumption_l3567_356706


namespace cistern_fill_time_l3567_356780

theorem cistern_fill_time (empty_time second_tap : ℝ) (fill_time_both : ℝ) 
  (h1 : empty_time = 9)
  (h2 : fill_time_both = 7.2) : 
  ∃ (fill_time_first : ℝ), 
    fill_time_first = 4 ∧ 
    (1 / fill_time_first - 1 / empty_time = 1 / fill_time_both) :=
by sorry

end cistern_fill_time_l3567_356780


namespace school_ratio_proof_l3567_356763

/-- Represents the ratio of two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Simplifies a ratio by dividing both numerator and denominator by their GCD -/
def simplifyRatio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd, denominator := r.denominator / gcd }

theorem school_ratio_proof (total_students : ℕ) (num_girls : ℕ) 
    (h1 : total_students = 300) (h2 : num_girls = 160) : 
    simplifyRatio { numerator := num_girls, denominator := total_students - num_girls } = 
    { numerator := 8, denominator := 7 } := by
  sorry

#check school_ratio_proof

end school_ratio_proof_l3567_356763


namespace max_third_side_length_l3567_356735

theorem max_third_side_length (a b : ℝ) (ha : a = 5) (hb : b = 11) :
  ∀ c : ℝ, (c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) → c ≤ 15 :=
by
  sorry

end max_third_side_length_l3567_356735


namespace critical_point_and_zeros_l3567_356799

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + (1 - 3 * Real.log x) / a

theorem critical_point_and_zeros (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → (deriv (f a)) x = 0 → x = 1 → a = 1) ∧
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ↔ 0 < a ∧ a < Real.exp (-1)) :=
sorry

end critical_point_and_zeros_l3567_356799


namespace fraction_addition_equivalence_l3567_356726

theorem fraction_addition_equivalence (a b : ℤ) (h : b > 0) :
  ∀ x y : ℤ, y > 0 →
    (a / b + x / y = (a + x) / (b + y)) ↔
    ∃ k : ℤ, x = -a * k^2 ∧ y = b * k :=
by sorry

end fraction_addition_equivalence_l3567_356726


namespace min_value_implications_l3567_356705

theorem min_value_implications (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, |x + a| + |x - b| ≥ 2) : 
  (1 / a + 1 / b + 1 / (a * b) ≥ 3) ∧ 
  (∀ t : ℝ, Real.sin t ^ 4 / a + Real.cos t ^ 4 / b ≥ 1 / 2) := by
  sorry

end min_value_implications_l3567_356705


namespace tangent_slope_angle_range_l3567_356779

theorem tangent_slope_angle_range (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n = Real.sqrt 3 / 2) :
  let f : ℝ → ℝ := λ x ↦ (1/3) * x^3 + n^2 * x
  let k := (m^2 + n^2)
  let θ := Real.arctan k
  θ ∈ Set.Ici (π/3) ∩ Set.Iio (π/2) :=
by sorry

end tangent_slope_angle_range_l3567_356779


namespace prescription_rebate_calculation_l3567_356708

/-- Calculates the mail-in rebate amount for a prescription purchase -/
def calculate_rebate (original_cost cashback_percent final_cost : ℚ) : ℚ :=
  let cashback := original_cost * (cashback_percent / 100)
  let cost_after_cashback := original_cost - cashback
  cost_after_cashback - final_cost

theorem prescription_rebate_calculation :
  let original_cost : ℚ := 150
  let cashback_percent : ℚ := 10
  let final_cost : ℚ := 110
  calculate_rebate original_cost cashback_percent final_cost = 25 := by
  sorry

#eval calculate_rebate 150 10 110

end prescription_rebate_calculation_l3567_356708


namespace page_lines_increase_l3567_356709

/-- 
Given an original number of lines L in a page, 
if increasing the number of lines by 80 results in a 50% increase, 
then the new total number of lines is 240.
-/
theorem page_lines_increase (L : ℕ) : 
  (L + 80 = L + L / 2) → (L + 80 = 240) := by
  sorry

end page_lines_increase_l3567_356709


namespace great_pyramid_sum_height_width_l3567_356738

/-- The Great Pyramid of Giza's dimensions -/
def great_pyramid (H W : ℕ) : Prop :=
  H = 500 + 20 ∧ W = H + 234

/-- Theorem: The sum of the height and width of the Great Pyramid of Giza is 1274 feet -/
theorem great_pyramid_sum_height_width :
  ∀ H W : ℕ, great_pyramid H W → H + W = 1274 :=
by
  sorry

#check great_pyramid_sum_height_width

end great_pyramid_sum_height_width_l3567_356738


namespace average_speed_barney_schwinn_l3567_356703

/-- Proves that the average speed is 31 miles per hour given the problem conditions --/
theorem average_speed_barney_schwinn : 
  let initial_reading : ℕ := 2552
  let final_reading : ℕ := 2992
  let total_time : ℕ := 14
  let distance := final_reading - initial_reading
  let exact_speed := (distance : ℚ) / total_time
  Int.floor (exact_speed + 1/2) = 31 := by sorry

end average_speed_barney_schwinn_l3567_356703


namespace math_books_together_probability_l3567_356781

def total_textbooks : ℕ := 15
def math_textbooks : ℕ := 4
def box_sizes : List ℕ := [4, 5, 6]

def is_valid_distribution (dist : List ℕ) : Prop :=
  dist.length = 3 ∧ 
  dist.sum = total_textbooks ∧
  ∀ b ∈ dist, b ≤ total_textbooks - math_textbooks + 1

def probability_math_books_together : ℚ :=
  4 / 273

theorem math_books_together_probability :
  probability_math_books_together = 
    (number_of_valid_distributions_with_math_books_together : ℚ) / 
    (total_number_of_valid_distributions : ℚ) :=
by sorry

#check math_books_together_probability

end math_books_together_probability_l3567_356781


namespace parabola_x_intercepts_l3567_356783

/-- The quadratic equation 3x^2 + 2x - 5 = 0 has exactly two distinct real solutions. -/
theorem parabola_x_intercepts :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  3 * x₁^2 + 2 * x₁ - 5 = 0 ∧
  3 * x₂^2 + 2 * x₂ - 5 = 0 ∧
  ∀ (x : ℝ), 3 * x^2 + 2 * x - 5 = 0 → (x = x₁ ∨ x = x₂) :=
by sorry

end parabola_x_intercepts_l3567_356783


namespace no_real_c_solution_l3567_356736

/-- Given a polynomial x^2 + bx + c with exactly one real root and b = c + 2,
    prove that there are no real values of c that satisfy these conditions. -/
theorem no_real_c_solution (b c : ℝ) 
    (h1 : ∃! x : ℝ, x^2 + b*x + c = 0)  -- exactly one real root
    (h2 : b = c + 2) :                  -- condition b = c + 2
    False :=                            -- no real c satisfies the conditions
  sorry

end no_real_c_solution_l3567_356736


namespace range_of_m_l3567_356797

def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m (m : ℝ) : 
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (m < -2 ∨ m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by sorry

end range_of_m_l3567_356797


namespace equal_sum_sequence_18th_term_l3567_356702

/-- An equal sum sequence is a sequence where the sum of each term and its next term is constant. -/
def EqualSumSequence (a : ℕ → ℝ) :=
  ∃ s : ℝ, ∀ n : ℕ, a n + a (n + 1) = s

theorem equal_sum_sequence_18th_term 
  (a : ℕ → ℝ) 
  (h_equal_sum : EqualSumSequence a)
  (h_first_term : a 1 = 2)
  (h_common_sum : ∃ s : ℝ, s = 5 ∧ ∀ n : ℕ, a n + a (n + 1) = s) :
  a 18 = 3 := by
  sorry

#check equal_sum_sequence_18th_term

end equal_sum_sequence_18th_term_l3567_356702


namespace absolute_value_equality_l3567_356731

theorem absolute_value_equality (x : ℝ) : |x + 2| = |x - 3| → x = 1/2 := by
  sorry

end absolute_value_equality_l3567_356731


namespace molar_mass_calculation_l3567_356707

/-- Given that 3 moles of a substance weigh 264 grams, prove that its molar mass is 88 grams/mole -/
theorem molar_mass_calculation (total_weight : ℝ) (num_moles : ℝ) (h1 : total_weight = 264) (h2 : num_moles = 3) :
  total_weight / num_moles = 88 := by
  sorry

end molar_mass_calculation_l3567_356707


namespace expression_evaluation_l3567_356750

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  (x^3 + 1) / x * (y^3 + 1) / y + (x^3 - 1) / y * (y^3 - 1) / x = 2 * x^2 * y^2 + 2 / (x * y) := by
  sorry

end expression_evaluation_l3567_356750


namespace negative_one_exponent_division_l3567_356789

theorem negative_one_exponent_division : ((-1 : ℤ) ^ 2003) / ((-1 : ℤ) ^ 2004) = -1 := by
  sorry

end negative_one_exponent_division_l3567_356789


namespace inequality_proof_l3567_356740

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : y * z + z * x + x * y = 1) : 
  x * (1 - y)^2 * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) ≤ 4 / (9 * Real.sqrt 3) := by
sorry

end inequality_proof_l3567_356740


namespace set_range_with_given_mean_median_l3567_356719

/-- Given a set of three real numbers with mean and median both equal to 5,
    and the smallest number being 2, the range of the set is 6. -/
theorem set_range_with_given_mean_median (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ordered set of three numbers
  a = 2 →  -- Smallest number is 2
  (a + b + c) / 3 = 5 →  -- Mean is 5
  b = 5 →  -- Median is 5 (for three numbers, the median is the middle number)
  c - a = 6 :=  -- Range is 6
by sorry

end set_range_with_given_mean_median_l3567_356719


namespace odd_operations_l3567_356788

theorem odd_operations (a b : ℤ) (ha : Odd a) (hb : Odd b) :
  Odd (a * b) ∧ Odd (a ^ 2) ∧ ¬(∀ x y : ℤ, Odd x → Odd y → Odd (x + y)) ∧ ¬(∀ x y : ℤ, Odd x → Odd y → Odd (x - y)) :=
by sorry

end odd_operations_l3567_356788


namespace coat_shirt_ratio_l3567_356762

theorem coat_shirt_ratio (pants shirt coat : ℕ) : 
  pants + shirt = 100 →
  pants + coat = 244 →
  coat = 180 →
  ∃ (k : ℕ), coat = k * shirt →
  coat / shirt = 5 := by
sorry

end coat_shirt_ratio_l3567_356762


namespace triangle_angle_bounds_l3567_356724

theorem triangle_angle_bounds (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = π) (h5 : A ≤ B) (h6 : B ≤ C) :
  (0 < A ∧ A ≤ π/3) ∧
  (0 < B ∧ B < π/2) ∧
  (π/3 ≤ C ∧ C < π) := by
  sorry

end triangle_angle_bounds_l3567_356724


namespace problem_statement_l3567_356792

theorem problem_statement (p q r : ℝ) 
  (h1 : p < q)
  (h2 : ∀ x, ((x - p) * (x - q)) / (x - r) ≥ 0 ↔ (x > 5 ∨ (7 ≤ x ∧ x ≤ 15))) :
  p + 2*q + 3*r = 52 := by
  sorry

end problem_statement_l3567_356792


namespace circle_ratio_theorem_l3567_356715

theorem circle_ratio_theorem (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) 
  (h : π * r₂^2 - π * r₁^2 = 4 * π * r₁^2) : 
  r₁ / r₂ = Real.sqrt 5 / 5 := by
  sorry

end circle_ratio_theorem_l3567_356715


namespace polynomial_simplification_l3567_356768

theorem polynomial_simplification (x : ℝ) :
  (3 * x^5 - 2 * x^3 + 5 * x^2 - 8 * x + 6) + (7 * x^4 + x^3 - 3 * x^2 + x - 9) =
  3 * x^5 + 7 * x^4 - x^3 + 2 * x^2 - 7 * x - 3 := by
  sorry

end polynomial_simplification_l3567_356768


namespace parametric_to_slope_intercept_l3567_356778

/-- A line parameterized by (x, y) = (3t + 6, 5t - 7) where t is a real number -/
def parametric_line (t : ℝ) : ℝ × ℝ := (3 * t + 6, 5 * t - 7)

/-- The slope-intercept form of a line -/
def slope_intercept_form (m b : ℝ) (x : ℝ) : ℝ := m * x + b

theorem parametric_to_slope_intercept :
  ∀ (x y : ℝ), (∃ t : ℝ, parametric_line t = (x, y)) →
  y = slope_intercept_form (5/3) (-17) x :=
by sorry

end parametric_to_slope_intercept_l3567_356778


namespace factorial_3_equals_6_l3567_356793

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_3_equals_6 : factorial 3 = 6 := by
  sorry

end factorial_3_equals_6_l3567_356793


namespace max_value_expression_l3567_356782

theorem max_value_expression (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  ∃ (m : ℝ), m = 2 ∧ ∀ x y z w, 
    (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) → (0 ≤ w ∧ w ≤ 1) →
    x + y + z + w - x*y - y*z - z*w - w*x ≤ m :=
by
  sorry

#check max_value_expression

end max_value_expression_l3567_356782


namespace total_broken_marbles_l3567_356771

def marble_set_1 : ℕ := 50
def marble_set_2 : ℕ := 60
def broken_percent_1 : ℚ := 10 / 100
def broken_percent_2 : ℚ := 20 / 100

theorem total_broken_marbles :
  ⌊marble_set_1 * broken_percent_1⌋ + ⌊marble_set_2 * broken_percent_2⌋ = 17 := by
  sorry

end total_broken_marbles_l3567_356771


namespace wall_length_is_800_l3567_356785

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  height : ℝ
  width : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.height * w.width

/-- Theorem: The length of the wall is 800 cm -/
theorem wall_length_is_800 (brick : BrickDimensions) (wall : WallDimensions) 
    (h1 : brick.length = 40)
    (h2 : brick.width = 11.25)
    (h3 : brick.height = 6)
    (h4 : wall.height = 600)
    (h5 : wall.width = 22.5)
    (h6 : wallVolume wall / brickVolume brick = 4000) :
    wall.length = 800 := by
  sorry

end wall_length_is_800_l3567_356785


namespace program_duration_l3567_356777

/-- Proves that the duration of each program is 30 minutes -/
theorem program_duration (num_programs : ℕ) (commercial_fraction : ℚ) (total_commercial_time : ℕ) :
  num_programs = 6 →
  commercial_fraction = 1/4 →
  total_commercial_time = 45 →
  ∃ (program_duration : ℕ),
    program_duration = 30 ∧
    (↑num_programs * commercial_fraction * ↑program_duration = ↑total_commercial_time) :=
by
  sorry

end program_duration_l3567_356777


namespace sets_equal_iff_m_eq_neg_two_sqrt_two_l3567_356759

def A (m : ℝ) : Set ℝ := {x | x^2 + m*x + 2 ≥ 0 ∧ x ≥ 0}

def B (m : ℝ) : Set ℝ := {y | ∃ x ∈ A m, y = Real.sqrt (x^2 + m*x + 2)}

theorem sets_equal_iff_m_eq_neg_two_sqrt_two (m : ℝ) :
  A m = B m ↔ m = -2 * Real.sqrt 2 := by sorry

end sets_equal_iff_m_eq_neg_two_sqrt_two_l3567_356759


namespace minimum_blocks_for_wall_l3567_356717

/-- Represents the dimensions of a wall -/
structure WallDimensions where
  length : ℝ
  height : ℝ

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  height : ℝ
  length1 : ℝ
  length2 : ℝ

/-- Calculates the minimum number of blocks needed for a wall -/
def minimumBlocksNeeded (wall : WallDimensions) (block : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the minimum number of blocks needed for the specific wall -/
theorem minimum_blocks_for_wall :
  let wall := WallDimensions.mk 150 8
  let block := BlockDimensions.mk 1 2 1.5
  minimumBlocksNeeded wall block = 604 :=
by sorry

end minimum_blocks_for_wall_l3567_356717


namespace four_row_arrangement_has_27_triangles_l3567_356733

/-- Represents a triangular arrangement of smaller triangles -/
structure TriangularArrangement where
  rows : ℕ

/-- Counts the number of small triangles in the arrangement -/
def count_small_triangles (arr : TriangularArrangement) : ℕ :=
  (arr.rows * (arr.rows + 1)) / 2

/-- Counts the number of medium triangles (made of 4 small triangles) -/
def count_medium_triangles (arr : TriangularArrangement) : ℕ :=
  if arr.rows ≥ 3 then
    ((arr.rows - 2) * (arr.rows - 1)) / 2
  else
    0

/-- Counts the number of large triangles (made of 9 small triangles) -/
def count_large_triangles (arr : TriangularArrangement) : ℕ :=
  if arr.rows ≥ 4 then
    (arr.rows - 3)
  else
    0

/-- Counts the total number of triangles in the arrangement -/
def total_triangles (arr : TriangularArrangement) : ℕ :=
  count_small_triangles arr + count_medium_triangles arr + count_large_triangles arr

/-- Theorem: In a triangular arrangement with 4 rows, there are 27 triangles in total -/
theorem four_row_arrangement_has_27_triangles :
  ∀ (arr : TriangularArrangement), arr.rows = 4 → total_triangles arr = 27 := by
  sorry

end four_row_arrangement_has_27_triangles_l3567_356733


namespace mod_product_equals_one_l3567_356732

theorem mod_product_equals_one (m : ℕ) : 
  187 * 973 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 1 := by
  sorry

end mod_product_equals_one_l3567_356732


namespace algebra_books_not_unique_l3567_356712

/-- Represents the number of books on a shelf -/
structure ShelfBooks where
  algebra : ℕ+
  geometry : ℕ+

/-- Represents the two shelves in the library -/
structure Library where
  longer_shelf : ShelfBooks
  shorter_shelf : ShelfBooks
  algebra_only : ℕ+

/-- The conditions of the library problem -/
def LibraryProblem (lib : Library) : Prop :=
  lib.longer_shelf.algebra > lib.shorter_shelf.algebra ∧
  lib.longer_shelf.geometry < lib.shorter_shelf.geometry ∧
  lib.longer_shelf.algebra ≠ lib.longer_shelf.geometry ∧
  lib.longer_shelf.algebra ≠ lib.shorter_shelf.algebra ∧
  lib.longer_shelf.algebra ≠ lib.shorter_shelf.geometry ∧
  lib.longer_shelf.geometry ≠ lib.shorter_shelf.algebra ∧
  lib.longer_shelf.geometry ≠ lib.shorter_shelf.geometry ∧
  lib.shorter_shelf.algebra ≠ lib.shorter_shelf.geometry ∧
  lib.longer_shelf.algebra ≠ lib.algebra_only ∧
  lib.longer_shelf.geometry ≠ lib.algebra_only ∧
  lib.shorter_shelf.algebra ≠ lib.algebra_only ∧
  lib.shorter_shelf.geometry ≠ lib.algebra_only

/-- The theorem stating that the number of algebra books to fill the longer shelf cannot be uniquely determined -/
theorem algebra_books_not_unique (lib : Library) (h : LibraryProblem lib) :
  ∃ (lib' : Library), LibraryProblem lib' ∧ lib'.algebra_only ≠ lib.algebra_only :=
sorry

end algebra_books_not_unique_l3567_356712


namespace experience_difference_l3567_356704

/-- Represents the years of experience for each coworker -/
structure Experience where
  roger : ℕ
  peter : ℕ
  tom : ℕ
  robert : ℕ
  mike : ℕ

/-- The conditions of the problem -/
def problemConditions (e : Experience) : Prop :=
  e.roger = e.peter + e.tom + e.robert + e.mike ∧
  e.roger = 50 - 8 ∧
  e.peter = 19 - 7 ∧
  e.tom = 2 * e.robert ∧
  e.robert = e.peter - 4 ∧
  e.robert > e.mike

/-- The theorem to prove -/
theorem experience_difference (e : Experience) :
  problemConditions e → e.robert - e.mike = 2 := by
  sorry

end experience_difference_l3567_356704


namespace car_distance_ratio_l3567_356767

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ  -- Speed in km/hr
  time : ℝ   -- Time in hours

/-- Calculates the distance traveled by a car -/
def distance (c : Car) : ℝ := c.speed * c.time

/-- The problem statement -/
theorem car_distance_ratio :
  let car_a : Car := ⟨50, 8⟩
  let car_b : Car := ⟨25, 4⟩
  (distance car_a) / (distance car_b) = 4
  := by sorry

end car_distance_ratio_l3567_356767


namespace oranges_left_l3567_356757

theorem oranges_left (initial_oranges : ℕ) (taken_oranges : ℕ) : 
  initial_oranges = 60 → taken_oranges = 35 → initial_oranges - taken_oranges = 25 := by
  sorry

end oranges_left_l3567_356757


namespace candy_distribution_l3567_356751

theorem candy_distribution (n : ℕ) : 
  (n > 0) → 
  (120 % n = 1) → 
  (n = 7 ∨ n = 17) :=
by sorry

end candy_distribution_l3567_356751


namespace box_weight_l3567_356721

/-- Given a pallet with boxes, calculate the weight of each box. -/
theorem box_weight (total_weight : ℕ) (num_boxes : ℕ) (h1 : total_weight = 267) (h2 : num_boxes = 3) :
  total_weight / num_boxes = 89 := by
  sorry

end box_weight_l3567_356721


namespace three_digit_reverse_divisible_by_11_l3567_356714

theorem three_digit_reverse_divisible_by_11 (a b c : Nat) (ha : a ≠ 0) (hb : b < 10) (hc : c < 10) :
  ∃ k : Nat, 100001 * a + 10010 * b + 1100 * c = 11 * k := by
  sorry

end three_digit_reverse_divisible_by_11_l3567_356714


namespace negation_of_existence_negation_of_greater_than_100_l3567_356798

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_greater_than_100 :
  (¬ ∃ n : ℕ, 2^n > 100) ↔ (∀ n : ℕ, 2^n ≤ 100) :=
by sorry

end negation_of_existence_negation_of_greater_than_100_l3567_356798


namespace octal_sum_equality_l3567_356772

/-- Converts a list of digits in base 8 to a natural number -/
def fromOctal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- The sum of 235₈, 647₈, and 54₈ is equal to 1160₈ -/
theorem octal_sum_equality :
  fromOctal [2, 3, 5] + fromOctal [6, 4, 7] + fromOctal [5, 4] = fromOctal [1, 1, 6, 0] := by
  sorry

#eval fromOctal [2, 3, 5] + fromOctal [6, 4, 7] + fromOctal [5, 4]
#eval fromOctal [1, 1, 6, 0]

end octal_sum_equality_l3567_356772


namespace two_circles_exist_l3567_356749

/-- The parabola y^2 = 4x with focus F(1,0) and directrix x = -1 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- The directrix of the parabola -/
def Directrix : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -1}

/-- Point M -/
def M : ℝ × ℝ := (4, 4)

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate for a circle passing through two points and tangent to a line -/
def CirclePassesThroughAndTangent (c : Circle) (p1 p2 : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  (c.center.1 - p1.1)^2 + (c.center.2 - p1.2)^2 = c.radius^2 ∧
  (c.center.1 - p2.1)^2 + (c.center.2 - p2.2)^2 = c.radius^2 ∧
  ∃ (q : ℝ × ℝ), q ∈ l ∧ (c.center.1 - q.1)^2 + (c.center.2 - q.2)^2 = c.radius^2

theorem two_circles_exist : ∃ (c1 c2 : Circle),
  CirclePassesThroughAndTangent c1 F M Directrix ∧
  CirclePassesThroughAndTangent c2 F M Directrix ∧
  c1 ≠ c2 ∧
  ∀ (c : Circle), CirclePassesThroughAndTangent c F M Directrix → c = c1 ∨ c = c2 :=
sorry

end two_circles_exist_l3567_356749


namespace smaller_circle_area_l3567_356787

/-- Two circles are externally tangent with common tangents. Given specific conditions, 
    prove that the area of the smaller circle is 5π/3. -/
theorem smaller_circle_area (r : ℝ) : 
  r > 0 → -- radius of smaller circle is positive
  (∃ (P A B : ℝ × ℝ), 
    -- PA and AB are tangent lines
    dist P A = dist A B ∧ 
    dist P A = 5 ∧
    -- Larger circle has radius 3r
    (∃ (C : ℝ × ℝ), dist C B = 3 * r)) →
  π * r^2 = 5 * π / 3 := by
  sorry

end smaller_circle_area_l3567_356787


namespace parabola_focus_centroid_l3567_356776

/-- Given three points A, B, C in a 2D plane, and a parabola y^2 = ax,
    if the focus of the parabola is exactly the centroid of triangle ABC,
    then a = 8. -/
theorem parabola_focus_centroid (A B C : ℝ × ℝ) (a : ℝ) : 
  A = (-1, 2) →
  B = (3, 4) →
  C = (4, -6) →
  let centroid := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  let focus := (a / 4, 0)
  centroid = focus →
  a = 8 := by sorry

end parabola_focus_centroid_l3567_356776


namespace fathers_age_l3567_356790

/-- Proves that given the conditions about the father's and Ming Ming's ages, the father's age this year is 35 -/
theorem fathers_age (ming_age ming_age_3_years_ago father_age father_age_3_years_ago : ℕ) :
  father_age_3_years_ago = 8 * ming_age_3_years_ago →
  father_age = 5 * ming_age →
  father_age = ming_age + 3 →
  father_age_3_years_ago = father_age - 3 →
  father_age = 35 := by
sorry


end fathers_age_l3567_356790


namespace distribute_5_3_l3567_356700

/-- The number of ways to distribute n distinct objects into k non-empty groups,
    where the order of the groups matters. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

theorem distribute_5_3 : distribute 5 3 = 150 := by
  sorry

end distribute_5_3_l3567_356700


namespace two_digit_number_representation_l3567_356743

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

theorem two_digit_number_representation (n : TwoDigitNumber) :
  n.value = 10 * n.tens + n.units := by
  sorry

end two_digit_number_representation_l3567_356743


namespace runners_capture_probability_l3567_356734

/-- Represents a runner on a circular track -/
structure Runner where
  direction : Bool -- true for counterclockwise, false for clockwise
  lap_time : ℕ -- time to complete one lap in seconds

/-- Represents the photographer's capture area -/
structure CaptureArea where
  fraction : ℚ -- fraction of the track captured
  center : ℚ -- position of the center of the capture area (0 ≤ center < 1)

/-- Calculates the probability of both runners being in the capture area -/
def probability_both_in_picture (runner1 runner2 : Runner) (capture : CaptureArea) 
  (start_time end_time : ℕ) : ℚ :=
sorry

theorem runners_capture_probability :
  let jenna : Runner := { direction := true, lap_time := 75 }
  let jonathan : Runner := { direction := false, lap_time := 60 }
  let capture : CaptureArea := { fraction := 1/3, center := 0 }
  probability_both_in_picture jenna jonathan capture (15 * 60) (16 * 60) = 2/3 :=
sorry

end runners_capture_probability_l3567_356734


namespace expression_evaluation_l3567_356739

theorem expression_evaluation : 
  Real.sqrt 3 + 3 + (1 / (Real.sqrt 3 + 3))^2 + 1 / (3 - Real.sqrt 3) = Real.sqrt 3 + 3 + 5/6 := by
  sorry

end expression_evaluation_l3567_356739


namespace salary_problem_l3567_356747

/-- Proves that A's salary is $3750 given the conditions of the problem -/
theorem salary_problem (a b : ℝ) : 
  a + b = 5000 →
  0.05 * a = 0.15 * b →
  a = 3750 := by
  sorry

end salary_problem_l3567_356747


namespace function_composition_l3567_356752

theorem function_composition (f : ℝ → ℝ) :
  (∀ x, f (x - 2) = 3 * x - 5) → (∀ x, f x = 3 * x + 1) := by
  sorry

end function_composition_l3567_356752


namespace triangle_area_fraction_l3567_356786

/-- The size of the grid -/
def gridSize : ℕ := 6

/-- The coordinates of the triangle vertices -/
def triangleVertices : List (ℕ × ℕ) := [(3, 3), (3, 5), (5, 5)]

/-- The area of the triangle -/
def triangleArea : ℚ := 2

/-- The area of the entire grid -/
def gridArea : ℕ := gridSize * gridSize

/-- The fraction of the grid area occupied by the triangle -/
def areaFraction : ℚ := triangleArea / gridArea

theorem triangle_area_fraction :
  areaFraction = 1 / 18 := by sorry

end triangle_area_fraction_l3567_356786


namespace number_of_female_students_l3567_356796

theorem number_of_female_students 
  (total_average : ℝ) 
  (num_male : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 90)
  (h2 : num_male = 8)
  (h3 : male_average = 82)
  (h4 : female_average = 92) :
  ∃ (num_female : ℕ), 
    (num_male : ℝ) * male_average + (num_female : ℝ) * female_average = 
    ((num_male : ℝ) + (num_female : ℝ)) * total_average ∧ 
    num_female = 32 := by
  sorry

end number_of_female_students_l3567_356796


namespace taxi_fare_100_miles_l3567_356711

/-- The cost of a taxi trip given the distance traveled. -/
noncomputable def taxi_cost (base_fare : ℝ) (rate : ℝ) (distance : ℝ) : ℝ :=
  base_fare + rate * distance

theorem taxi_fare_100_miles :
  let base_fare : ℝ := 40
  let rate : ℝ := (200 - base_fare) / 80
  taxi_cost base_fare rate 100 = 240 := by
  sorry

end taxi_fare_100_miles_l3567_356711


namespace min_weighings_to_find_fake_pearl_l3567_356774

/-- Represents the result of a weighing operation -/
inductive WeighResult
  | Equal : WeighResult
  | Left : WeighResult
  | Right : WeighResult

/-- Represents a strategy for finding the fake pearl -/
def Strategy := List WeighResult → Nat

/-- The number of pearls -/
def numPearls : Nat := 9

/-- The minimum number of weighings needed to find the fake pearl -/
def minWeighings : Nat := 2

/-- A theorem stating that the minimum number of weighings to find the fake pearl is 2 -/
theorem min_weighings_to_find_fake_pearl :
  ∃ (s : Strategy), ∀ (outcomes : List WeighResult),
    outcomes.length ≤ minWeighings →
    s outcomes < numPearls ∧
    (∀ (t : Strategy),
      (∀ (outcomes' : List WeighResult),
        outcomes'.length < outcomes.length →
        t outcomes' = numPearls) →
      s outcomes ≤ t outcomes) :=
sorry

end min_weighings_to_find_fake_pearl_l3567_356774


namespace hair_cut_length_l3567_356773

/-- The length of hair cut off is equal to the difference between the initial and final hair lengths. -/
theorem hair_cut_length (initial_length final_length cut_length : ℝ) 
  (h1 : initial_length = 11)
  (h2 : final_length = 7)
  (h3 : cut_length = initial_length - final_length) :
  cut_length = 4 := by
  sorry

end hair_cut_length_l3567_356773


namespace first_number_value_l3567_356728

theorem first_number_value (x y z : ℤ) 
  (sum_xy : x + y = 31)
  (sum_yz : y + z = 47)
  (sum_xz : x + z = 52)
  (condition : y + z = x + 16) :
  x = 31 := by
sorry

end first_number_value_l3567_356728


namespace ellipse_foci_condition_l3567_356770

theorem ellipse_foci_condition (α : Real) (h1 : 0 < α) (h2 : α < π / 2) :
  (∀ x y : Real, x^2 / Real.sin α + y^2 / Real.cos α = 1 →
    ∃ c : Real, c > 0 ∧ 
      ∀ x₀ y₀ : Real, (x₀ + c)^2 + y₀^2 + (x₀ - c)^2 + y₀^2 = 
        2 * ((x^2 / Real.sin α + y^2 / Real.cos α) * (1 / Real.sin α + 1 / Real.cos α))) →
  π / 4 < α ∧ α < π / 2 := by sorry

end ellipse_foci_condition_l3567_356770


namespace unique_satisfying_function_l3567_356791

/-- A function satisfying the given inequality for all real numbers -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x * z) - f x * f (y * z) ≥ 1

/-- The theorem stating that there is a unique function satisfying the inequality -/
theorem unique_satisfying_function :
  ∃! f : ℝ → ℝ, SatisfiesInequality f ∧ ∀ x : ℝ, f x = 1 := by
  sorry

end unique_satisfying_function_l3567_356791


namespace book_price_adjustment_l3567_356722

theorem book_price_adjustment (x : ℝ) :
  (1 + x / 100) * (1 - x / 100) = 0.75 → x = 50 := by
  sorry

end book_price_adjustment_l3567_356722


namespace jacob_painting_fraction_l3567_356716

/-- Jacob's painting rate in houses per minute -/
def painting_rate : ℚ := 1 / 60

/-- Time given to paint in minutes -/
def paint_time : ℚ := 15

/-- Theorem: If Jacob can paint a house in 60 minutes, then he can paint 1/4 of the house in 15 minutes -/
theorem jacob_painting_fraction :
  painting_rate * paint_time = 1 / 4 := by sorry

end jacob_painting_fraction_l3567_356716


namespace power_three_mod_eleven_l3567_356775

theorem power_three_mod_eleven : 3^87 + 5 ≡ 3 [ZMOD 11] := by sorry

end power_three_mod_eleven_l3567_356775


namespace first_nonzero_digit_after_decimal_1_271_l3567_356764

theorem first_nonzero_digit_after_decimal_1_271 :
  ∃ (n : ℕ) (r : ℚ), 1000 * (1 / 271) = n + r ∧ n = 3 ∧ 0 < r ∧ r < 1 := by
  sorry

end first_nonzero_digit_after_decimal_1_271_l3567_356764


namespace student_arrangements_eq_60_l3567_356727

/-- The number of ways to arrange 6 students among three venues A, B, and C,
    where venue A receives 1 student, venue B receives 2 students,
    and venue C receives 3 students. -/
def student_arrangements : ℕ :=
  Nat.choose 6 1 * Nat.choose 5 2

theorem student_arrangements_eq_60 : student_arrangements = 60 := by
  sorry

end student_arrangements_eq_60_l3567_356727


namespace exists_subset_with_unique_adjacency_l3567_356769

def adjacent (p q : ℤ × ℤ × ℤ) : Prop :=
  let (x, y, z) := p
  let (u, v, w) := q
  abs (x - u) + abs (y - v) + abs (z - w) = 1

theorem exists_subset_with_unique_adjacency :
  ∃ (S : Set (ℤ × ℤ × ℤ)), ∀ p : ℤ × ℤ × ℤ,
    (p ∈ S ∧ ∀ q, adjacent p q → q ∉ S) ∨
    (p ∉ S ∧ ∃! q, adjacent p q ∧ q ∈ S) :=
by sorry

end exists_subset_with_unique_adjacency_l3567_356769


namespace mary_regular_rate_l3567_356710

/-- Represents Mary's work schedule and pay structure --/
structure MaryWork where
  maxHours : ℕ
  regularHours : ℕ
  overtimeRate : ℚ
  weeklyEarnings : ℚ

/-- Calculates Mary's regular hourly rate --/
def regularRate (w : MaryWork) : ℚ :=
  let overtimeHours := w.maxHours - w.regularHours
  w.weeklyEarnings / (w.regularHours + w.overtimeRate * overtimeHours)

/-- Theorem: Mary's regular hourly rate is $8 per hour --/
theorem mary_regular_rate :
  let w : MaryWork := {
    maxHours := 45,
    regularHours := 20,
    overtimeRate := 1.25,
    weeklyEarnings := 410
  }
  regularRate w = 8 := by sorry

end mary_regular_rate_l3567_356710


namespace equation_rewrite_l3567_356720

theorem equation_rewrite (x y : ℝ) : (2 * x + y = 5) ↔ (y = 5 - 2 * x) := by
  sorry

end equation_rewrite_l3567_356720


namespace max_tournament_size_l3567_356713

/-- Represents a tournament with 2^n students --/
structure Tournament (n : ℕ) where
  students : Fin (2^n)
  day1_pairs : List (Fin (2^n) × Fin (2^n))
  day2_pairs : List (Fin (2^n) × Fin (2^n))

/-- The sets of pairs that played on both days are the same --/
def same_pairs (t : Tournament n) : Prop :=
  t.day1_pairs.toFinset = t.day2_pairs.toFinset

/-- The maximum value of n for which the tournament conditions hold --/
def max_n : ℕ := 3

/-- Theorem stating that 3 is the maximum value of n for which the tournament conditions hold --/
theorem max_tournament_size :
  ∀ n : ℕ, n > max_n → ¬∃ t : Tournament n, same_pairs t :=
sorry

end max_tournament_size_l3567_356713


namespace vector_equality_implies_equal_norm_vector_equality_transitivity_l3567_356755

variable {V : Type*} [NormedAddCommGroup V]

theorem vector_equality_implies_equal_norm (a b : V) :
  a = b → ‖a‖ = ‖b‖ := by sorry

theorem vector_equality_transitivity (a b c : V) :
  a = b → b = c → a = c := by sorry

end vector_equality_implies_equal_norm_vector_equality_transitivity_l3567_356755


namespace turtle_race_time_difference_l3567_356729

theorem turtle_race_time_difference (greta_time gloria_time : ℕ) 
  (h1 : greta_time = 6)
  (h2 : gloria_time = 8)
  (h3 : gloria_time = 2 * (gloria_time / 2)) :
  greta_time - (gloria_time / 2) = 2 := by
  sorry

end turtle_race_time_difference_l3567_356729


namespace complex_equation_solution_l3567_356784

theorem complex_equation_solution (z : ℂ) : (1 + z) * Complex.I = 1 - z → z = -Complex.I := by
  sorry

end complex_equation_solution_l3567_356784


namespace intersection_of_A_and_B_l3567_356748

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x > 2}
def B : Set ℝ := {x : ℝ | (x - 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l3567_356748


namespace distribute_4_3_l3567_356701

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    where each box must contain at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  if n < k then 0
  else if n = k then Nat.factorial k
  else sorry  -- Actual implementation would go here

/-- The theorem stating that distributing 4 distinct objects into 3 distinct boxes,
    where each box must contain at least one object, can be done in 60 ways. -/
theorem distribute_4_3 : distribute 4 3 = 60 := by
  sorry

end distribute_4_3_l3567_356701


namespace initial_peaches_l3567_356744

theorem initial_peaches (initial : ℕ) : initial + 52 = 86 → initial = 34 := by
  sorry

end initial_peaches_l3567_356744
