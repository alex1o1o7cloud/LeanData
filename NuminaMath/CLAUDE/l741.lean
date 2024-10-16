import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_problem_l741_74184

def i : ℂ := Complex.I

theorem complex_number_problem (z : ℂ) (h : (1 + 2*i)*z = 3 - 4*i) : 
  z.im = -2 ∧ Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l741_74184


namespace NUMINAMATH_CALUDE_odd_then_even_probability_l741_74120

/-- Represents a ball with a number -/
structure Ball :=
  (number : Nat)

/-- Represents the bag of balls -/
def Bag : Finset Ball := sorry

/-- The bag contains 5 balls numbered 1 to 5 -/
axiom bag_content : Bag.card = 5 ∧ ∀ n : Nat, 1 ≤ n ∧ n ≤ 5 → ∃! b : Ball, b ∈ Bag ∧ b.number = n

/-- A ball is odd-numbered if its number is odd -/
def is_odd (b : Ball) : Prop := b.number % 2 = 1

/-- A ball is even-numbered if its number is even -/
def is_even (b : Ball) : Prop := b.number % 2 = 0

/-- The probability of drawing an odd-numbered ball first and an even-numbered ball second -/
def prob_odd_then_even : ℚ := sorry

/-- The main theorem to prove -/
theorem odd_then_even_probability : prob_odd_then_even = 3 / 10 := sorry

end NUMINAMATH_CALUDE_odd_then_even_probability_l741_74120


namespace NUMINAMATH_CALUDE_garden_potato_yield_l741_74107

/-- Calculates the expected potato yield from a rectangular garden --/
theorem garden_potato_yield 
  (length_steps width_steps : ℕ) 
  (step_length : ℝ) 
  (planting_ratio : ℝ) 
  (yield_rate : ℝ) :
  length_steps = 10 →
  width_steps = 30 →
  step_length = 3 →
  planting_ratio = 0.9 →
  yield_rate = 3/4 →
  (length_steps : ℝ) * step_length * (width_steps : ℝ) * step_length * planting_ratio * yield_rate = 1822.5 := by
  sorry

end NUMINAMATH_CALUDE_garden_potato_yield_l741_74107


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l741_74172

theorem cubic_polynomial_satisfies_conditions :
  let q : ℝ → ℝ := λ x => -2/3 * x^3 + 2 * x^2 - 8/3 * x - 16/3
  (q 1 = -6) ∧ (q 2 = -8) ∧ (q 3 = -14) ∧ (q 4 = -28) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l741_74172


namespace NUMINAMATH_CALUDE_CD_parallel_BE_l741_74161

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define points C and D
def C : ℝ × ℝ := (1, 0)
def D : ℝ × ℝ := (2, 0)

-- Define a line passing through C and a point (x, y) on Γ
def line_through_C (x y : ℝ) : Set (ℝ × ℝ) :=
  {(t, u) | ∃ (k : ℝ), u - C.2 = k * (t - C.1) ∧ t ≠ C.1}

-- Define point A as an intersection of the line and Γ
def A (x y : ℝ) : ℝ × ℝ :=
  (x, y)

-- Define point B as the other intersection of the line and Γ
def B (x y : ℝ) : ℝ × ℝ :=
  sorry

-- Define point E as the intersection of AD and x=3
def E (x y : ℝ) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem CD_parallel_BE (x y : ℝ) :
  Γ x y →
  (x, y) ∈ line_through_C x y →
  (B x y).1 ≠ x →
  (let slope_CD := (D.2 - C.2) / (D.1 - C.1)
   let slope_BE := (E x y).2 / ((E x y).1 - (B x y).1)
   slope_CD = slope_BE) :=
sorry

end NUMINAMATH_CALUDE_CD_parallel_BE_l741_74161


namespace NUMINAMATH_CALUDE_secretary_discussions_l741_74147

/-- Represents the number of emails sent in a small discussion -/
def small_discussion_emails : ℕ := 7 * 6

/-- Represents the number of emails sent in a large discussion -/
def large_discussion_emails : ℕ := 15 * 14

/-- Represents the total number of emails sent excluding the secretary's -/
def total_emails : ℕ := 1994

/-- Represents the maximum number of discussions a jury member can participate in -/
def max_discussions : ℕ := 10

theorem secretary_discussions (m b : ℕ) :
  m + b ≤ max_discussions →
  small_discussion_emails * m + large_discussion_emails * b + 6 * m + 14 * b = total_emails →
  m = 6 ∧ b = 2 := by
  sorry

#check secretary_discussions

end NUMINAMATH_CALUDE_secretary_discussions_l741_74147


namespace NUMINAMATH_CALUDE_min_red_chips_l741_74104

theorem min_red_chips (r w b : ℕ) : 
  b ≥ (w : ℚ) / 3 →
  (b : ℚ) ≤ r / 4 →
  w + b ≥ 70 →
  r ≥ 72 ∧ ∀ (r' : ℕ), (∃ (w' b' : ℕ), 
    b' ≥ (w' : ℚ) / 3 ∧
    (b' : ℚ) ≤ r' / 4 ∧
    w' + b' ≥ 70) → 
  r' ≥ 72 := by
sorry

end NUMINAMATH_CALUDE_min_red_chips_l741_74104


namespace NUMINAMATH_CALUDE_twenty_percent_of_number_is_fifty_l741_74126

theorem twenty_percent_of_number_is_fifty (x : ℝ) : (20 / 100) * x = 50 → x = 250 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_of_number_is_fifty_l741_74126


namespace NUMINAMATH_CALUDE_equivalence_sqrt_and_fraction_l741_74186

theorem equivalence_sqrt_and_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt a + 1 > Real.sqrt b) ↔ 
  (∀ x > 1, a * x + x / (x - 1) > b) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_sqrt_and_fraction_l741_74186


namespace NUMINAMATH_CALUDE_cube_root_8000_l741_74190

theorem cube_root_8000 (c d : ℕ+) : 
  (c : ℝ) * (d : ℝ)^(1/3 : ℝ) = 20 → 
  (∀ (c' d' : ℕ+), (c' : ℝ) * (d' : ℝ)^(1/3 : ℝ) = 20 → d ≤ d') → 
  c + d = 21 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_8000_l741_74190


namespace NUMINAMATH_CALUDE_largest_two_digit_number_with_one_l741_74179

def digits : List Nat := [1, 5, 6, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  n % 10 = 1 ∧
  (n / 10) ∈ digits ∧
  1 ∈ digits

theorem largest_two_digit_number_with_one :
  ∀ n : Nat, is_valid_number n → n ≤ 91 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_number_with_one_l741_74179


namespace NUMINAMATH_CALUDE_coefficient_of_x_fourth_l741_74100

/-- The expression for which we need to find the coefficient of x^4 -/
def expression (x : ℝ) : ℝ := 5 * (x^4 - 2*x^3) + 3 * (2*x^2 - 3*x^4 + x^6) - (5*x^6 - 2*x^4)

/-- The coefficient of x^4 in the given expression is -2 -/
theorem coefficient_of_x_fourth : (deriv^[4] expression 0) / 24 = -2 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_fourth_l741_74100


namespace NUMINAMATH_CALUDE_prob_king_ace_ten_l741_74116

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of Tens in a standard deck -/
def NumTens : ℕ := 4

/-- Probability of drawing a King, then an Ace, then a 10 from a standard deck -/
theorem prob_king_ace_ten (deck : ℕ) (kings aces tens : ℕ) : 
  deck = StandardDeck → kings = NumKings → aces = NumAces → tens = NumTens →
  (kings : ℚ) / deck * aces / (deck - 1) * tens / (deck - 2) = 8 / 16575 := by
sorry

end NUMINAMATH_CALUDE_prob_king_ace_ten_l741_74116


namespace NUMINAMATH_CALUDE_toy_store_revenue_l741_74175

theorem toy_store_revenue (D : ℚ) (D_pos : D > 0) : 
  let nov := (2 / 5 : ℚ) * D
  let jan := (1 / 5 : ℚ) * nov
  let avg := (nov + jan) / 2
  D / avg = 25 / 6 := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_l741_74175


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l741_74157

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2 + x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + 1

theorem cubic_function_extrema (a b : ℝ) :
  (f' a b 1 = 0 ∧ f' a b 2 = 0) →
  (a = 1/6 ∧ b = -3/4) ∧
  (IsLocalMax (f a b) 1 ∧ IsLocalMin (f a b) 2) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l741_74157


namespace NUMINAMATH_CALUDE_bills_average_speed_day2_l741_74103

/-- Represents the driving scenario of Bill's two-day journey --/
structure DrivingScenario where
  speed_day2 : ℝ  -- Average speed on the second day
  time_day2 : ℝ   -- Time spent driving on the second day
  total_distance : ℝ  -- Total distance driven over two days
  total_time : ℝ      -- Total time spent driving over two days

/-- Defines the conditions of Bill's journey --/
def journey_conditions (s : DrivingScenario) : Prop :=
  s.total_distance = 680 ∧
  s.total_time = 18 ∧
  s.total_distance = (s.speed_day2 + 5) * (s.time_day2 + 2) + s.speed_day2 * s.time_day2 ∧
  s.total_time = (s.time_day2 + 2) + s.time_day2

/-- Theorem stating that given the journey conditions, Bill's average speed on the second day was 35 mph --/
theorem bills_average_speed_day2 (s : DrivingScenario) : 
  journey_conditions s → s.speed_day2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_bills_average_speed_day2_l741_74103


namespace NUMINAMATH_CALUDE_value_calculation_l741_74115

theorem value_calculation (N : ℝ) (h : 0.4 * N = 300) : (1/4) * (1/3) * (2/5) * N = 25 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l741_74115


namespace NUMINAMATH_CALUDE_p_and_q_properties_l741_74141

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.sqrt x = Real.sqrt (2 * x + 1)

-- Define proposition q
def q : Prop := ∀ x : ℝ, x > 0 → x^2 < x^3

-- Theorem stating the properties of p and q
theorem p_and_q_properties :
  (∃ x : ℝ, Real.sqrt x = Real.sqrt (2 * x + 1)) ∧  -- p is existential
  ¬p ∧                                             -- p is false
  (∀ x : ℝ, x > 0 → x^2 < x^3) ∧                   -- q is universal
  ¬q                                               -- q is false
  := by sorry

end NUMINAMATH_CALUDE_p_and_q_properties_l741_74141


namespace NUMINAMATH_CALUDE_coordinates_sum_of_X_l741_74152

-- Define the points as pairs of real numbers
def X : ℝ × ℝ := sorry
def Y : ℝ × ℝ := (3, 5)
def Z : ℝ × ℝ := (1, -3)

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem coordinates_sum_of_X :
  (distance X Z) / (distance X Y) = 1/2 ∧
  (distance Z Y) / (distance X Y) = 1/2 →
  X.1 + X.2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_sum_of_X_l741_74152


namespace NUMINAMATH_CALUDE_equation_solution_l741_74123

theorem equation_solution (x : ℝ) : 
  x ≠ 2 → ((4*x^2 + 3*x + 2)/(x - 2) = 4*x + 5 ↔ x = -4) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l741_74123


namespace NUMINAMATH_CALUDE_boys_age_l741_74142

theorem boys_age (current_age : ℕ) : 
  (current_age = 2 * (current_age - 5)) → current_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_boys_age_l741_74142


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l741_74171

/-- The imaginary part of the product of two complex numbers -/
theorem imaginary_part_of_product (ω₁ ω₂ : ℂ) : 
  let z := ω₁ * ω₂
  ω₁ = -1/2 + (Real.sqrt 3/2) * I →
  ω₂ = Complex.exp (I * (π/12)) →
  z.im = Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l741_74171


namespace NUMINAMATH_CALUDE_inverse_proportion_l741_74101

theorem inverse_proportion (a b : ℝ → ℝ) (k : ℝ) :
  (∀ x, a x * b x = k) →  -- a and b are inversely proportional
  (a 5 = 40) →            -- a = 40 when b = 5
  (b 5 = 5) →             -- explicitly stating b = 5
  (b 10 = 10) →           -- explicitly stating b = 10
  (a 10 = 20) :=          -- a = 20 when b = 10
by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_l741_74101


namespace NUMINAMATH_CALUDE_exam_failure_rate_l741_74199

/-- Examination results -/
structure ExamResults where
  total_candidates : ℕ
  num_girls : ℕ
  boys_math_pass_rate : ℚ
  boys_science_pass_rate : ℚ
  boys_lang_pass_rate : ℚ
  girls_math_pass_rate : ℚ
  girls_science_pass_rate : ℚ
  girls_lang_pass_rate : ℚ

/-- Calculate the failure rate given exam results -/
def calculate_failure_rate (results : ExamResults) : ℚ :=
  let num_boys := results.total_candidates - results.num_girls
  let boys_passing := min (results.boys_math_pass_rate * num_boys)
                          (min (results.boys_science_pass_rate * num_boys)
                               (results.boys_lang_pass_rate * num_boys))
  let girls_passing := min (results.girls_math_pass_rate * results.num_girls)
                           (min (results.girls_science_pass_rate * results.num_girls)
                                (results.girls_lang_pass_rate * results.num_girls))
  let total_passing := boys_passing + girls_passing
  let total_failing := results.total_candidates - total_passing
  total_failing / results.total_candidates

/-- The main theorem about the examination failure rate -/
theorem exam_failure_rate :
  let results := ExamResults.mk 2500 1100 (42/100) (39/100) (36/100) (35/100) (32/100) (40/100)
  calculate_failure_rate results = 6576/10000 := by
  sorry


end NUMINAMATH_CALUDE_exam_failure_rate_l741_74199


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l741_74164

theorem complex_fraction_simplification :
  (7 + 8 * Complex.I) / (3 - 4 * Complex.I) = -11/25 + 52/25 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l741_74164


namespace NUMINAMATH_CALUDE_lcm_inequality_l741_74198

theorem lcm_inequality (n : ℕ) (k : ℕ) (a : Fin k → ℕ) 
  (h1 : ∀ i : Fin k, n ≥ a i)
  (h2 : ∀ i j : Fin k, i < j → a i > a j)
  (h3 : ∀ i j : Fin k, Nat.lcm (a i) (a j) ≤ n) :
  ∀ i : Fin k, (i.val + 1) * a i ≤ n := by
  sorry

end NUMINAMATH_CALUDE_lcm_inequality_l741_74198


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_l741_74106

-- Define the parametric equations
def x_param (t : ℝ) : ℝ := t + 1
def y_param (t : ℝ) : ℝ := 3 - t^2

-- State the theorem
theorem parametric_to_cartesian :
  ∀ (x y : ℝ), (∃ t : ℝ, x = x_param t ∧ y = y_param t) ↔ y = -x^2 + 2*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_l741_74106


namespace NUMINAMATH_CALUDE_shaded_area_is_three_l741_74146

def grid_area : ℕ := 3 * 2 + 4 * 6 + 5 * 3

def unshaded_triangle_area : ℕ := (14 * 6) / 2

def shaded_area : ℕ := grid_area - unshaded_triangle_area

theorem shaded_area_is_three : shaded_area = 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_three_l741_74146


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_and_necessary_not_sufficient_l741_74160

theorem sufficient_not_necessary_and_necessary_not_sufficient :
  (∀ a : ℝ, a > 1 → 1 / a < 1) ∧
  (∃ a : ℝ, ¬(a > 1) ∧ 1 / a < 1) ∧
  (∀ a b : ℝ, a * b ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ a * b = 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_and_necessary_not_sufficient_l741_74160


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l741_74121

def tank_capacity (initial_loss_rate : ℕ) (initial_loss_hours : ℕ) 
  (secondary_loss_rate : ℕ) (secondary_loss_hours : ℕ)
  (fill_rate : ℕ) (fill_hours : ℕ) (remaining_to_fill : ℕ) : ℕ :=
  (initial_loss_rate * initial_loss_hours) + 
  (secondary_loss_rate * secondary_loss_hours) + 
  (fill_rate * fill_hours) + 
  remaining_to_fill

theorem tank_capacity_proof : 
  tank_capacity 32000 5 10000 10 40000 3 140000 = 520000 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l741_74121


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l741_74138

/-- An arithmetic sequence with 12 terms -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of odd-numbered terms is 10 -/
def SumOddTerms (a : ℕ → ℚ) : Prop :=
  (a 1) + (a 3) + (a 5) + (a 7) + (a 9) + (a 11) = 10

/-- The sum of even-numbered terms is 22 -/
def SumEvenTerms (a : ℕ → ℚ) : Prop :=
  (a 2) + (a 4) + (a 6) + (a 8) + (a 10) + (a 12) = 22

/-- The common difference of the arithmetic sequence is 2 -/
def CommonDifferenceIsTwo (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h1 : ArithmeticSequence a)
  (h2 : SumOddTerms a)
  (h3 : SumEvenTerms a) :
  CommonDifferenceIsTwo a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l741_74138


namespace NUMINAMATH_CALUDE_exists_multiple_without_zero_l741_74111

def has_no_zero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

theorem exists_multiple_without_zero (k : ℕ) : 
  k > 0 → ∃ n : ℕ, 5^k ∣ n ∧ has_no_zero n :=
sorry

end NUMINAMATH_CALUDE_exists_multiple_without_zero_l741_74111


namespace NUMINAMATH_CALUDE_apple_distribution_l741_74109

/-- The number of ways to distribute n items among k people with a minimum of m items each. -/
def distribution_ways (n m k : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The problem statement -/
theorem apple_distribution :
  distribution_ways 30 3 3 = 253 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l741_74109


namespace NUMINAMATH_CALUDE_convention_center_distance_l741_74127

/-- The distance from Elena's home to the convention center -/
def distance : ℝ := sorry

/-- Elena's initial speed in miles per hour -/
def initial_speed : ℝ := 45

/-- The increase in Elena's speed for the rest of the journey -/
def speed_increase : ℝ := 20

/-- The time Elena would be late if she continued at the initial speed -/
def late_time : ℝ := 0.75

/-- The time Elena arrives early after increasing her speed -/
def early_time : ℝ := 0.25

/-- The actual time needed to arrive on time -/
def actual_time : ℝ := sorry

theorem convention_center_distance :
  (distance = initial_speed * (actual_time + late_time)) ∧
  (distance - initial_speed = (initial_speed + speed_increase) * (actual_time - 1 - early_time)) ∧
  (distance = 191.25) := by sorry

end NUMINAMATH_CALUDE_convention_center_distance_l741_74127


namespace NUMINAMATH_CALUDE_maria_coffee_shop_visits_l741_74158

/-- 
Given that Maria orders 3 cups of coffee each time she goes to the coffee shop
and orders 6 cups of coffee per day, prove that she goes to the coffee shop 2 times per day.
-/
theorem maria_coffee_shop_visits 
  (cups_per_visit : ℕ) 
  (cups_per_day : ℕ) 
  (h1 : cups_per_visit = 3)
  (h2 : cups_per_day = 6) :
  cups_per_day / cups_per_visit = 2 := by
  sorry

end NUMINAMATH_CALUDE_maria_coffee_shop_visits_l741_74158


namespace NUMINAMATH_CALUDE_negation_of_existence_cubic_equation_negation_l741_74122

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ (∀ x, f x ≠ 0) := by sorry

theorem cubic_equation_negation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by
  apply negation_of_existence

end NUMINAMATH_CALUDE_negation_of_existence_cubic_equation_negation_l741_74122


namespace NUMINAMATH_CALUDE_angies_age_l741_74187

theorem angies_age : ∃ (age : ℕ), 2 * age + 4 = 20 ∧ age = 8 := by
  sorry

end NUMINAMATH_CALUDE_angies_age_l741_74187


namespace NUMINAMATH_CALUDE_inequality_solution_l741_74140

theorem inequality_solution (x : ℝ) : 
  (2 * Real.sqrt ((4 * x - 9)^2) + Real.sqrt (3 * Real.sqrt x - 5 + 2 * |x - 2|) ≤ 18 - 8 * x) ↔ 
  (x = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l741_74140


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_original_l741_74119

-- Define a line in 2D space
structure Line2D where
  f : ℝ → ℝ → ℝ
  is_line : ∀ x y, f x y = 0 ↔ (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ (a ≠ 0 ∨ b ≠ 0))

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be on a line
def on_line (p : Point2D) (l : Line2D) : Prop :=
  l.f p.x p.y = 0

-- Define what it means for a point to not be on a line
def not_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.f p.x p.y ≠ 0

-- Define parallel lines
def parallel (l1 l2 : Line2D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1.f x y = k * l2.f x y

-- Theorem statement
theorem line_through_point_parallel_to_original 
  (l : Line2D) (p1 p2 : Point2D) 
  (h1 : on_line p1 l) 
  (h2 : not_on_line p2 l) : 
  ∃ l2 : Line2D, 
    (∀ x y, l2.f x y = l.f x y - l.f p1.x p1.y - l.f p2.x p2.y) ∧ 
    on_line p2 l2 ∧ 
    parallel l l2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_original_l741_74119


namespace NUMINAMATH_CALUDE_quadratic_interval_l741_74118

theorem quadratic_interval (x : ℝ) : 
  (6 ≤ x^2 + 5*x + 6 ∧ x^2 + 5*x + 6 ≤ 12) ↔ ((-6 ≤ x ∧ x ≤ -5) ∨ (0 ≤ x ∧ x ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_interval_l741_74118


namespace NUMINAMATH_CALUDE_polar_midpoint_specific_l741_74169

/-- The midpoint of a line segment in polar coordinates -/
def polar_midpoint (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: The midpoint of the line segment with endpoints (10, π/4) and (10, 3π/4) in polar coordinates -/
theorem polar_midpoint_specific :
  let (r, θ) := polar_midpoint 10 (π/4) 10 (3*π/4)
  r = 5 * Real.sqrt 2 ∧ θ = π/2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π :=
sorry

end NUMINAMATH_CALUDE_polar_midpoint_specific_l741_74169


namespace NUMINAMATH_CALUDE_function_minimum_value_l741_74165

theorem function_minimum_value (x : ℝ) (h : x ≥ 5) :
  (x^2 - 4*x + 9) / (x - 4) ≥ 10 := by
  sorry

#check function_minimum_value

end NUMINAMATH_CALUDE_function_minimum_value_l741_74165


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l741_74139

-- Problem 1
theorem problem_1 : 2 * Real.sqrt 12 - 6 * Real.sqrt (1/3) + 3 * Real.sqrt 48 = 14 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx : x > 0) : 
  (2/3) * Real.sqrt (9*x) + 6 * Real.sqrt (x/4) - x * Real.sqrt (1/x) = 4 * Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l741_74139


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l741_74154

/-- Given a quadratic equation kx^2 - 2(k+1)x + k-1 = 0 with two distinct real roots, 
    this theorem proves properties about the range of k and the sum of reciprocals of roots. -/
theorem quadratic_equation_properties (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k*x₁^2 - 2*(k+1)*x₁ + (k-1) = 0 ∧ k*x₂^2 - 2*(k+1)*x₂ + (k-1) = 0) →
  (k > -1/3 ∧ k ≠ 0) ∧
  ¬(∃ k : ℝ, ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → k*x₁^2 - 2*(k+1)*x₁ + (k-1) = 0 → k*x₂^2 - 2*(k+1)*x₂ + (k-1) = 0 → 
    1/x₁ + 1/x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l741_74154


namespace NUMINAMATH_CALUDE_meeting_point_equation_correct_l741_74188

/-- Represents the time taken for two travelers to meet given their journey durations and a head start for one traveler. -/
def meeting_equation (x : ℚ) : Prop :=
  (x + 2) / 7 + x / 5 = 1

/-- The total journey time for the first traveler -/
def journey_time_A : ℚ := 5

/-- The total journey time for the second traveler -/
def journey_time_B : ℚ := 7

/-- The head start time for the second traveler -/
def head_start : ℚ := 2

/-- Theorem stating that the meeting equation correctly represents the meeting point of two travelers given the conditions -/
theorem meeting_point_equation_correct :
  ∃ x : ℚ, 
    x > 0 ∧ 
    x < journey_time_A ∧
    x + head_start < journey_time_B ∧
    meeting_equation x :=
sorry

end NUMINAMATH_CALUDE_meeting_point_equation_correct_l741_74188


namespace NUMINAMATH_CALUDE_parabola_rectangle_problem_l741_74192

/-- The parabola equation -/
def parabola_equation (k x y : ℝ) : Prop := y = k^2 - x^2

/-- Rectangle ABCD properties -/
structure Rectangle (k : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  parallel_to_axes : Prop
  A_on_x_axis : A.2 = 0
  D_on_x_axis : D.2 = 0
  V_midpoint_BC : (B.1 + C.1) / 2 = 0 ∧ (B.2 + C.2) / 2 = k^2

/-- Perimeter of the rectangle -/
def perimeter (rect : Rectangle k) : ℝ :=
  2 * (|rect.A.1 - rect.B.1| + |rect.A.2 - rect.B.2|)

/-- Main theorem -/
theorem parabola_rectangle_problem (k : ℝ) 
  (h_pos : k > 0)
  (rect : Rectangle k)
  (h_perimeter : perimeter rect = 48) :
  k = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_rectangle_problem_l741_74192


namespace NUMINAMATH_CALUDE_greater_number_problem_l741_74137

theorem greater_number_problem (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : 
  max x y = 35 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l741_74137


namespace NUMINAMATH_CALUDE_solution_set_inequality_l741_74180

/-- Given that the solution set of ax^2 + bx + 4 > 0 is (-1, 2),
    prove that the solution set of ax + b + 4 > 0 is (-∞, 3) -/
theorem solution_set_inequality (a b : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + 4 > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x : ℝ, a*x + b + 4 > 0 ↔ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l741_74180


namespace NUMINAMATH_CALUDE_cosine_sine_shift_l741_74112

theorem cosine_sine_shift :
  let f (x : ℝ) := Real.cos (2 * x + π / 3)
  let g (x : ℝ) := Real.sin (2 * x)
  ∃ (shift : ℝ), shift = 5 * π / 6 ∧
    ∀ (x : ℝ), f x = g (x + shift) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_shift_l741_74112


namespace NUMINAMATH_CALUDE_third_side_possible_length_l741_74133

/-- Given a triangle with two sides of lengths 3 and 7, 
    prove that 6 is a possible length for the third side. -/
theorem third_side_possible_length :
  ∃ (a b c : ℝ), a = 3 ∧ b = 7 ∧ c = 6 ∧ 
  a + b > c ∧ b + c > a ∧ a + c > b ∧
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_third_side_possible_length_l741_74133


namespace NUMINAMATH_CALUDE_elizabeth_granola_profit_l741_74194

/-- Elizabeth's granola business problem -/
theorem elizabeth_granola_profit : 
  let ingredient_cost_per_bag : ℚ := 3
  let total_bags : ℕ := 20
  let original_price : ℚ := 6
  let discounted_price : ℚ := 4
  let full_price_sales : ℕ := 15
  let discounted_sales : ℕ := 5

  let total_revenue : ℚ := 
    (full_price_sales : ℚ) * original_price + 
    (discounted_sales : ℚ) * discounted_price

  let total_cost : ℚ := (total_bags : ℚ) * ingredient_cost_per_bag

  let net_profit : ℚ := total_revenue - total_cost

  net_profit = 50 := by
sorry


end NUMINAMATH_CALUDE_elizabeth_granola_profit_l741_74194


namespace NUMINAMATH_CALUDE_peter_erasers_l741_74168

theorem peter_erasers (initial_erasers : ℕ) (received_erasers : ℕ) : 
  initial_erasers = 8 → received_erasers = 3 → initial_erasers + received_erasers = 11 := by
  sorry

end NUMINAMATH_CALUDE_peter_erasers_l741_74168


namespace NUMINAMATH_CALUDE_highlighter_spent_theorem_l741_74177

def total_money : ℝ := 150
def sharpener_price : ℝ := 3
def notebook_price : ℝ := 7
def eraser_price : ℝ := 2
def sharpener_count : ℕ := 5
def notebook_count : ℕ := 6
def eraser_count : ℕ := 15

def heaven_spent : ℝ := sharpener_price * sharpener_count + notebook_price * notebook_count

def money_left_after_heaven : ℝ := total_money - heaven_spent

def brother_eraser_spent : ℝ := eraser_price * eraser_count

theorem highlighter_spent_theorem :
  money_left_after_heaven - brother_eraser_spent = 63 := by sorry

end NUMINAMATH_CALUDE_highlighter_spent_theorem_l741_74177


namespace NUMINAMATH_CALUDE_unique_solution_is_zero_function_l741_74167

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 * y) = f (x * y) + y * f (f x + y)

/-- The theorem stating that the only function satisfying the equation is the zero function -/
theorem unique_solution_is_zero_function
  (f : ℝ → ℝ) (h : SatisfiesFunctionalEquation f) :
  ∀ y : ℝ, f y = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_zero_function_l741_74167


namespace NUMINAMATH_CALUDE_special_polynomial_exists_l741_74145

/-- A fifth-degree polynomial with specific root properties -/
def exists_special_polynomial : Prop :=
  ∃ (P : ℝ → ℝ),
    (∀ x : ℝ, ∃ (a b c d e f : ℝ), P x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) ∧
    (∀ r : ℝ, P r = 0 → r < 0) ∧
    (∀ s : ℝ, (deriv P) s = 0 → s > 0) ∧
    (∃ t : ℝ, P t = 0) ∧
    (∃ u : ℝ, (deriv P) u = 0)

/-- Theorem stating the existence of a special polynomial -/
theorem special_polynomial_exists : exists_special_polynomial :=
sorry

end NUMINAMATH_CALUDE_special_polynomial_exists_l741_74145


namespace NUMINAMATH_CALUDE_laptop_price_calculation_l741_74108

def original_price : ℝ := 1200
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.12

def discounted_price : ℝ := original_price * (1 - discount_rate)
def total_price : ℝ := discounted_price * (1 + tax_rate)

theorem laptop_price_calculation :
  total_price = 940.8 := by sorry

end NUMINAMATH_CALUDE_laptop_price_calculation_l741_74108


namespace NUMINAMATH_CALUDE_solve_for_a_l741_74166

theorem solve_for_a (x a : ℝ) : 2 * x + a - 8 = 0 → x = 2 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l741_74166


namespace NUMINAMATH_CALUDE_remainder_theorem_l741_74105

theorem remainder_theorem (r : ℤ) : (r^15 - 1) % (r + 2) = -32769 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l741_74105


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l741_74110

/- Define the pocket contents -/
def total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2

/- Define the number of balls drawn -/
def drawn_balls : ℕ := 2

/- Define the events -/
def at_least_one_white (x : ℕ) : Prop := x ≥ 1
def both_red (x : ℕ) : Prop := x = 2

/- Theorem statement -/
theorem mutually_exclusive_events :
  ¬(∃ (white_drawn : ℕ), 
    white_drawn ≤ drawn_balls ∧ 
    at_least_one_white white_drawn ∧ 
    both_red (drawn_balls - white_drawn)) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l741_74110


namespace NUMINAMATH_CALUDE_zoe_coloring_books_l741_74130

/-- Given two coloring books with the same number of pictures and the number of pictures left to color,
    calculate the number of pictures colored. -/
def pictures_colored (pictures_per_book : ℕ) (books : ℕ) (pictures_left : ℕ) : ℕ :=
  pictures_per_book * books - pictures_left

/-- Theorem stating that given two coloring books with 44 pictures each and 68 pictures left to color,
    the number of pictures colored is 20. -/
theorem zoe_coloring_books : pictures_colored 44 2 68 = 20 := by
  sorry

end NUMINAMATH_CALUDE_zoe_coloring_books_l741_74130


namespace NUMINAMATH_CALUDE_medical_team_selection_l741_74124

theorem medical_team_selection (orthopedic neurosurgeons internists : ℕ) 
  (h1 : orthopedic = 3) 
  (h2 : neurosurgeons = 4) 
  (h3 : internists = 5) 
  (team_size : ℕ) 
  (h4 : team_size = 5) : 
  (Nat.choose (orthopedic + neurosurgeons + internists) team_size) -
  ((Nat.choose (neurosurgeons + internists) team_size - 1) +
   (Nat.choose (orthopedic + internists) team_size - 1) +
   (Nat.choose (orthopedic + neurosurgeons) team_size) +
   1) = 590 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_selection_l741_74124


namespace NUMINAMATH_CALUDE_trajectory_equation_l741_74195

/-- The circle C -/
def C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

/-- The fixed point F -/
def F : ℝ × ℝ := (2, 0)

/-- Predicate for a point being on the trajectory of Q -/
def on_trajectory (x y : ℝ) : Prop :=
  ∃ (px py : ℝ),
    C px py ∧
    ∃ (qx qy : ℝ),
      -- Q is on the perpendicular bisector of PF
      (qx - px)^2 + (qy - py)^2 = (qx - F.1)^2 + (qy - F.2)^2 ∧
      -- Q is on the line CP
      (qx + 2) * py = (qy) * (px + 2) ∧
      -- Q is the point we're considering
      qx = x ∧ qy = y

/-- The main theorem -/
theorem trajectory_equation :
  ∀ x y : ℝ, on_trajectory x y ↔ x^2 - y^2/3 = 1 := by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l741_74195


namespace NUMINAMATH_CALUDE_min_abs_z_l741_74134

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z - Complex.I * 7) = 15) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 8) + Complex.abs (w - Complex.I * 7) = 15 ∧ Complex.abs w = 56 / 15 :=
by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_l741_74134


namespace NUMINAMATH_CALUDE_prime_relation_l741_74178

theorem prime_relation (P Q : ℕ) (hP : Nat.Prime P) (hQ : Nat.Prime Q)
  (h1 : P ∣ (Q^3 - 1)) (h2 : Q ∣ (P - 1)) : P = 1 + Q + Q^2 := by
  sorry

end NUMINAMATH_CALUDE_prime_relation_l741_74178


namespace NUMINAMATH_CALUDE_smartphone_price_decrease_l741_74125

/-- The average percentage decrease in price for a smartphone that underwent two price reductions -/
theorem smartphone_price_decrease (original_price final_price : ℝ) 
  (h1 : original_price = 2000)
  (h2 : final_price = 1280) : 
  (original_price - final_price) / original_price / 2 * 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_decrease_l741_74125


namespace NUMINAMATH_CALUDE_electric_bicycle_sales_l741_74148

theorem electric_bicycle_sales (model_a_first_quarter : Real) 
  (model_bc_first_quarter : Real) (a : Real) :
  model_a_first_quarter = 0.56 ∧ 
  model_bc_first_quarter = 1 - model_a_first_quarter ∧
  0.56 * (1 + 0.23) + (1 - 0.56) * (1 - a / 100) = 1 + 0.12 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_electric_bicycle_sales_l741_74148


namespace NUMINAMATH_CALUDE_differential_pricing_profitability_l741_74135

theorem differential_pricing_profitability 
  (n : ℝ) (t : ℝ) (h_n_pos : n > 0) (h_t_pos : t > 0) : 
  let shorts_ratio : ℝ := 0.75
  let suits_ratio : ℝ := 0.25
  let businessmen_ratio : ℝ := 0.8
  let tourists_ratio : ℝ := 0.2
  let uniform_revenue := n * t
  let differential_revenue (X : ℝ) := 
    (shorts_ratio * n * t) + 
    (suits_ratio * businessmen_ratio * n * (t + X))
  ∃ X : ℝ, X ≥ 0 ∧ 
    ∀ Y : ℝ, Y ≥ 0 → differential_revenue Y ≥ uniform_revenue → Y ≥ X ∧
    differential_revenue X = uniform_revenue ∧
    X = t / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_differential_pricing_profitability_l741_74135


namespace NUMINAMATH_CALUDE_angle_with_touching_circles_theorem_l741_74151

/-- Represents a circle in 2D space --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an angle in 2D space --/
structure Angle where
  vertex : ℝ × ℝ
  side1 : ℝ × ℝ → Prop
  side2 : ℝ × ℝ → Prop

/-- Predicate to check if a circle touches a line internally --/
def touches_internally (c : Circle) (l : ℝ × ℝ → Prop) : Prop := sorry

/-- Predicate to check if two circles are non-intersecting --/
def non_intersecting (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a point is on an angle --/
def on_angle (p : ℝ × ℝ) (a : Angle) : Prop := sorry

/-- Predicate to check if a point describes the arc of a circle --/
def describes_circle_arc (p : ℝ × ℝ) : Prop := sorry

theorem angle_with_touching_circles_theorem (a : Angle) (c1 c2 : Circle) 
  (h1 : touches_internally c1 a.side1)
  (h2 : touches_internally c2 a.side2)
  (h3 : non_intersecting c1 c2) :
  ∃ p : ℝ × ℝ, on_angle p a ∧ describes_circle_arc p := by
  sorry

end NUMINAMATH_CALUDE_angle_with_touching_circles_theorem_l741_74151


namespace NUMINAMATH_CALUDE_largest_number_l741_74117

theorem largest_number (a b c d : ℝ) (ha : a = 1) (hb : b = -2) (hc : c = 0) (hd : d = Real.sqrt 3) :
  d = max a (max b (max c d)) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l741_74117


namespace NUMINAMATH_CALUDE_adam_has_more_apple_difference_l741_74173

/-- The number of apples Adam has -/
def adam_apples : ℕ := 9

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 6

/-- Adam has more apples than Jackie -/
theorem adam_has_more : adam_apples > jackie_apples := by sorry

/-- The difference in apples between Adam and Jackie is 3 -/
theorem apple_difference : adam_apples - jackie_apples = 3 := by sorry

end NUMINAMATH_CALUDE_adam_has_more_apple_difference_l741_74173


namespace NUMINAMATH_CALUDE_mikes_stamp_collection_last_page_l741_74128

/-- Represents the stamp collection problem --/
structure StampCollection where
  initial_books : ℕ
  pages_per_book : ℕ
  initial_stamps_per_page : ℕ
  new_stamps_per_page : ℕ
  filled_books : ℕ
  filled_pages_in_last_book : ℕ

/-- Calculates the number of stamps on the last page after reorganization --/
def stamps_on_last_page (sc : StampCollection) : ℕ :=
  let total_stamps := sc.initial_books * sc.pages_per_book * sc.initial_stamps_per_page
  let filled_pages := sc.filled_books * sc.pages_per_book + sc.filled_pages_in_last_book
  let stamps_on_filled_pages := filled_pages * sc.new_stamps_per_page
  total_stamps - stamps_on_filled_pages

/-- Theorem stating that for Mike's stamp collection, the last page contains 9 stamps --/
theorem mikes_stamp_collection_last_page :
  let sc : StampCollection := {
    initial_books := 6,
    pages_per_book := 30,
    initial_stamps_per_page := 7,
    new_stamps_per_page := 9,
    filled_books := 3,
    filled_pages_in_last_book := 26
  }
  stamps_on_last_page sc = 9 := by
  sorry


end NUMINAMATH_CALUDE_mikes_stamp_collection_last_page_l741_74128


namespace NUMINAMATH_CALUDE_rectangle_dimensions_area_l741_74143

theorem rectangle_dimensions_area (x : ℝ) : 
  (x - 2) * (2 * x + 5) = 8 * x - 6 → x = 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_area_l741_74143


namespace NUMINAMATH_CALUDE_delta_value_l741_74182

theorem delta_value (Δ : ℤ) : 4 * (-3) = Δ + 5 → Δ = -17 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l741_74182


namespace NUMINAMATH_CALUDE_sum_of_six_odd_squares_not_1986_l741_74114

theorem sum_of_six_odd_squares_not_1986 : ¬ ∃ (a b c d e f : ℤ), 
  (Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f) ∧
  (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 1986) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_six_odd_squares_not_1986_l741_74114


namespace NUMINAMATH_CALUDE_carbonated_water_in_solution2_l741_74163

/-- Represents a solution mixture of lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated_water : ℝ
  sum_to_one : lemonade + carbonated_water = 1

/-- Represents the final mixture of two solutions -/
structure Mixture where
  solution1 : Solution
  solution2 : Solution
  proportion1 : ℝ
  proportion2 : ℝ
  sum_to_one : proportion1 + proportion2 = 1
  carbonated_water_percent : ℝ

/-- The main theorem to prove -/
theorem carbonated_water_in_solution2 
  (mix : Mixture)
  (h1 : mix.solution1.lemonade = 0.2)
  (h2 : mix.solution2.lemonade = 0.45)
  (h3 : mix.carbonated_water_percent = 0.72)
  (h4 : mix.proportion1 = 0.6799999999999997) :
  mix.solution2.carbonated_water = 0.55 := by
  sorry

#eval 1 - 0.45 -- Expected output: 0.55

end NUMINAMATH_CALUDE_carbonated_water_in_solution2_l741_74163


namespace NUMINAMATH_CALUDE_area_between_line_and_curve_l741_74132

theorem area_between_line_and_curve : 
  let f (x : ℝ) := 3 * x
  let g (x : ℝ) := x^2
  let lower_bound := (0 : ℝ)
  let upper_bound := (3 : ℝ)
  let area := ∫ x in lower_bound..upper_bound, (f x - g x)
  area = 9/2 := by sorry

end NUMINAMATH_CALUDE_area_between_line_and_curve_l741_74132


namespace NUMINAMATH_CALUDE_parallel_line_slope_l741_74131

/-- The slope of a line parallel to 3x - 6y = 12 is 1/2 -/
theorem parallel_line_slope (a b c : ℝ) : 
  (3 : ℝ) * a - (6 : ℝ) * b = (12 : ℝ) → 
  ∃ (m : ℝ), m = (1 : ℝ) / (2 : ℝ) ∧ 
  ∀ (x y : ℝ), (y = m * x + c) → 
  (∃ (k : ℝ), (3 : ℝ) * x - (6 : ℝ) * y = k) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l741_74131


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l741_74102

theorem triangle_angle_measure (a b c A B C : Real) : 
  a = 4 →
  b = 4 * Real.sqrt 3 →
  A = π / 6 →
  a * Real.sin B = b * Real.sin A →
  (B = π / 3 ∨ B = 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l741_74102


namespace NUMINAMATH_CALUDE_octal_sum_451_167_l741_74189

/-- Converts a base-8 number to base-10 --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- The sum of two octal numbers in base 8 --/
def octal_sum (a b : ℕ) : ℕ := decimal_to_octal (octal_to_decimal a + octal_to_decimal b)

theorem octal_sum_451_167 : octal_sum 451 167 = 640 := by sorry

end NUMINAMATH_CALUDE_octal_sum_451_167_l741_74189


namespace NUMINAMATH_CALUDE_mark_kangaroos_l741_74162

theorem mark_kangaroos (num_kangaroos num_goats : ℕ) : 
  num_goats = 3 * num_kangaroos →
  2 * num_kangaroos + 4 * num_goats = 322 →
  num_kangaroos = 23 := by
sorry

end NUMINAMATH_CALUDE_mark_kangaroos_l741_74162


namespace NUMINAMATH_CALUDE_elisa_current_amount_l741_74159

def current_amount (target : ℕ) (needed : ℕ) : ℕ :=
  target - needed

theorem elisa_current_amount :
  let target : ℕ := 53
  let needed : ℕ := 16
  current_amount target needed = 37 := by
  sorry

end NUMINAMATH_CALUDE_elisa_current_amount_l741_74159


namespace NUMINAMATH_CALUDE_min_sum_triangular_grid_l741_74196

/-- Represents a triangular grid with 16 cells --/
structure TriangularGrid :=
  (cells : Fin 16 → ℕ)

/-- Checks if a number is prime --/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Represents the layers of the triangular grid --/
def layers : List (List (Fin 16)) := sorry

/-- The sum of numbers in each layer is prime --/
def layerSumsPrime (grid : TriangularGrid) : Prop :=
  ∀ layer ∈ layers, isPrime (layer.map grid.cells).sum

/-- The theorem stating the minimum sum of all numbers in the grid --/
theorem min_sum_triangular_grid :
  ∀ grid : TriangularGrid, layerSumsPrime grid →
  (Finset.univ.sum (λ i => grid.cells i) ≥ 22) :=
sorry

end NUMINAMATH_CALUDE_min_sum_triangular_grid_l741_74196


namespace NUMINAMATH_CALUDE_jacob_dimes_l741_74156

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes : ℕ) : ℕ :=
  pennies * coin_value "penny" + nickels * coin_value "nickel" + dimes * coin_value "dime"

theorem jacob_dimes (mrs_hilt_pennies mrs_hilt_nickels mrs_hilt_dimes : ℕ)
                    (jacob_pennies jacob_nickels : ℕ)
                    (difference : ℕ) :
  mrs_hilt_pennies = 2 →
  mrs_hilt_nickels = 2 →
  mrs_hilt_dimes = 2 →
  jacob_pennies = 4 →
  jacob_nickels = 1 →
  difference = 13 →
  ∃ jacob_dimes : ℕ,
    total_value mrs_hilt_pennies mrs_hilt_nickels mrs_hilt_dimes -
    total_value jacob_pennies jacob_nickels jacob_dimes = difference ∧
    jacob_dimes = 1 :=
by sorry

end NUMINAMATH_CALUDE_jacob_dimes_l741_74156


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l741_74170

theorem perfect_square_polynomial (n : ℤ) : 
  ∃ (m : ℤ), n^4 + 6*n^3 + 11*n^2 + 3*n + 31 = m^2 ↔ n = 10 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l741_74170


namespace NUMINAMATH_CALUDE_prob_heart_diamond_standard_deck_l741_74153

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (num_red_suits : ℕ)
  (num_black_suits : ℕ)

/-- Standard deck properties -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    num_red_suits := 2,
    num_black_suits := 2 }

/-- Probability of drawing a heart first and a diamond second -/
def prob_heart_then_diamond (d : Deck) : ℚ :=
  (d.cards_per_suit : ℚ) / (d.total_cards : ℚ) *
  (d.cards_per_suit : ℚ) / ((d.total_cards - 1) : ℚ)

/-- Theorem stating the probability of drawing a heart then a diamond -/
theorem prob_heart_diamond_standard_deck :
  prob_heart_then_diamond standard_deck = 169 / 2652 :=
sorry

end NUMINAMATH_CALUDE_prob_heart_diamond_standard_deck_l741_74153


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l741_74150

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 * x * 2*x = 64) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l741_74150


namespace NUMINAMATH_CALUDE_no_fibonacci_right_triangle_l741_74155

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Theorem: No right-angled triangle has all sides as Fibonacci numbers -/
theorem no_fibonacci_right_triangle (n : ℕ) : 
  (fib n)^2 + (fib (n + 1))^2 ≠ (fib (n + 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_no_fibonacci_right_triangle_l741_74155


namespace NUMINAMATH_CALUDE_largest_band_size_l741_74181

/-- Represents a band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- Checks if a band formation is valid --/
def isValidFormation (f : BandFormation) (totalMembers : ℕ) : Prop :=
  f.rows * f.membersPerRow + 3 = totalMembers

/-- Checks if the new formation after rearrangement is valid --/
def isValidNewFormation (f : BandFormation) (totalMembers : ℕ) : Prop :=
  (f.rows - 3) * (f.membersPerRow + 2) = totalMembers

/-- Main theorem: The largest possible number of band members is 147 --/
theorem largest_band_size :
  ∃ (f : BandFormation) (m : ℕ),
    m < 150 ∧
    isValidFormation f m ∧
    isValidNewFormation f m ∧
    ∀ (f' : BandFormation) (m' : ℕ),
      m' < 150 →
      isValidFormation f' m' →
      isValidNewFormation f' m' →
      m' ≤ m ∧
    m = 147 := by
  sorry

end NUMINAMATH_CALUDE_largest_band_size_l741_74181


namespace NUMINAMATH_CALUDE_latestPossibleTime_is_latest_l741_74113

/-- Represents a time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60
  seconds_valid : seconds < 60

/-- Checks if a given time matches the visible digits pattern -/
def matchesVisibleDigits (t : Time) : Bool :=
  let h1 := t.hours / 10
  let h2 := t.hours % 10
  let m1 := t.minutes / 10
  let m2 := t.minutes % 10
  let s1 := t.seconds / 10
  let s2 := t.seconds % 10
  (h1 = 2 ∧ m1 = 0 ∧ s1 = 2 ∧ s2 = 2) ∨
  (h2 = 2 ∧ m1 = 0 ∧ s1 = 2 ∧ s2 = 2) ∨
  (h1 = 2 ∧ m2 = 0 ∧ s1 = 2 ∧ s2 = 2) ∨
  (h1 = 2 ∧ m1 = 0 ∧ s2 = 2 ∧ h2 = 2) ∨
  (h1 = 2 ∧ m1 = 0 ∧ s1 = 2 ∧ m2 = 2)

/-- The latest possible time satisfying the conditions -/
def latestPossibleTime : Time := {
  hours := 23
  minutes := 50
  seconds := 22
  hours_valid := by simp
  minutes_valid := by simp
  seconds_valid := by simp
}

/-- Theorem stating that the latestPossibleTime is indeed the latest time satisfying the conditions -/
theorem latestPossibleTime_is_latest :
  matchesVisibleDigits latestPossibleTime ∧
  ∀ t : Time, matchesVisibleDigits t → t.hours * 3600 + t.minutes * 60 + t.seconds ≤
    latestPossibleTime.hours * 3600 + latestPossibleTime.minutes * 60 + latestPossibleTime.seconds :=
by
  sorry


end NUMINAMATH_CALUDE_latestPossibleTime_is_latest_l741_74113


namespace NUMINAMATH_CALUDE_closest_to_quotient_l741_74174

def options : List ℝ := [500, 1500, 2500, 5000, 7500]

theorem closest_to_quotient (x : ℝ) (h : x ∈ options \ {2500}) :
  |503 / 0.198 - 2500| < |503 / 0.198 - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_quotient_l741_74174


namespace NUMINAMATH_CALUDE_gcd_50404_40303_l741_74149

theorem gcd_50404_40303 : Nat.gcd 50404 40303 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_50404_40303_l741_74149


namespace NUMINAMATH_CALUDE_f_of_2_equals_negative_2_l741_74144

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x

-- State the theorem
theorem f_of_2_equals_negative_2 : f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_negative_2_l741_74144


namespace NUMINAMATH_CALUDE_function_equality_l741_74136

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then -x^2 + 1 else x - 1

theorem function_equality (a : ℝ) : f (a + 1) = f a ↔ a = -1/2 ∨ a = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l741_74136


namespace NUMINAMATH_CALUDE_integer_cube_between_zero_and_nine_l741_74176

theorem integer_cube_between_zero_and_nine (a : ℤ) : 0 < a^3 ∧ a^3 < 9 → a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_cube_between_zero_and_nine_l741_74176


namespace NUMINAMATH_CALUDE_third_stick_shorter_by_one_cm_l741_74183

/-- The length difference between the second and third stick -/
def stick_length_difference (first_stick second_stick third_stick : ℝ) : ℝ :=
  second_stick - third_stick

/-- Proof that the third stick is 1 cm shorter than the second stick -/
theorem third_stick_shorter_by_one_cm 
  (first_stick : ℝ)
  (second_stick : ℝ)
  (third_stick : ℝ)
  (h1 : first_stick = 3)
  (h2 : second_stick = 2 * first_stick)
  (h3 : first_stick + second_stick + third_stick = 14) :
  stick_length_difference first_stick second_stick third_stick = 1 := by
sorry

end NUMINAMATH_CALUDE_third_stick_shorter_by_one_cm_l741_74183


namespace NUMINAMATH_CALUDE_line_equation_proof_l741_74191

-- Define the circle P
def circle_P (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the parabola S
def parabola_S (x y : ℝ) : Prop := y = x^2 / 8

-- Define a line passing through a point
def line_through_point (k m x y : ℝ) : Prop := y = k*x + m

-- Define the center of the circle
def circle_center : ℝ × ℝ := (0, 2)

-- Define the property of four points being in arithmetic sequence
def arithmetic_sequence (a b c d : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  let d2 := Real.sqrt ((b.1 - c.1)^2 + (b.2 - c.2)^2)
  let d3 := Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2)
  d2 - d1 = d3 - d2

theorem line_equation_proof :
  ∀ (k m : ℝ) (a b c d : ℝ × ℝ),
    (∀ x y, line_through_point k m x y → (circle_P x y ∨ parabola_S x y)) →
    line_through_point k m circle_center.1 circle_center.2 →
    arithmetic_sequence a b c d →
    (k = -Real.sqrt 2 / 2 ∨ k = Real.sqrt 2 / 2) ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l741_74191


namespace NUMINAMATH_CALUDE_product_sum_equality_l741_74197

theorem product_sum_equality : 1520 * 1997 * 0.152 * 100 + 152^2 = 46161472 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equality_l741_74197


namespace NUMINAMATH_CALUDE_prime_square_plus_2007p_minus_one_prime_l741_74129

theorem prime_square_plus_2007p_minus_one_prime (p : ℕ) : 
  Prime p ∧ Prime (p^2 + 2007*p - 1) ↔ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_plus_2007p_minus_one_prime_l741_74129


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l741_74193

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 3 - 3 * Complex.I) * (2 * Real.sqrt 2 + 2 * Complex.I)) = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l741_74193


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l741_74185

/-- Given a square with side length 2 and four congruent isosceles triangles constructed with 
    their bases on the sides of the square, if the sum of the areas of the four isosceles 
    triangles is equal to the area of the square, then the length of one of the two congruent 
    sides of one isosceles triangle is √17/2. -/
theorem isosceles_triangle_side_length (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 2 →
  triangle_base = square_side →
  (4 * (1/2 * triangle_base * triangle_height)) = square_side^2 →
  ∃ (triangle_side : ℝ), 
    triangle_side^2 = (triangle_base/2)^2 + triangle_height^2 ∧ 
    triangle_side = Real.sqrt 17 / 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l741_74185
