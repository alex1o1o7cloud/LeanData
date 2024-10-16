import Mathlib

namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l1892_189273

/-- Represents the number of years in a period -/
def period : ℕ := 200

/-- Represents the frequency of leap years -/
def leapYearFrequency : ℕ := 5

/-- Calculates the maximum number of leap years in the given period -/
def maxLeapYears : ℕ := period / leapYearFrequency

/-- Theorem: The maximum number of leap years in a 200-year period,
    with leap years occurring every 5 years, is 40 -/
theorem max_leap_years_in_period :
  maxLeapYears = 40 := by sorry

end NUMINAMATH_CALUDE_max_leap_years_in_period_l1892_189273


namespace NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_8_equals_sqrt_2_l1892_189234

theorem sqrt_18_minus_sqrt_8_equals_sqrt_2 : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_8_equals_sqrt_2_l1892_189234


namespace NUMINAMATH_CALUDE_a_less_than_one_l1892_189248

theorem a_less_than_one : 
  (0.99999 : ℝ)^(1.00001 : ℝ) * (1.00001 : ℝ)^(0.99999 : ℝ) < 1 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_one_l1892_189248


namespace NUMINAMATH_CALUDE_digit_sum_property_l1892_189252

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_number (A : ℕ) : Prop :=
  10 ≤ A ∧ A ≤ 99 ∧ (sum_of_digits A)^2 = sum_of_digits (A^2)

def solution_set : Finset ℕ := {11, 12, 13, 20, 21, 22, 30, 31, 50}

theorem digit_sum_property :
  ∀ A : ℕ, is_valid_number A ↔ A ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_digit_sum_property_l1892_189252


namespace NUMINAMATH_CALUDE_equilateral_triangle_point_distance_l1892_189219

/-- Given an equilateral triangle ABC with side length a and a point P inside the triangle
    such that PA = u, PB = v, PC = w, and u^2 + v^2 = w^2, prove that w^2 + √3uv = a^2. -/
theorem equilateral_triangle_point_distance (a u v w : ℝ) :
  a > 0 →  -- Ensure positive side length
  u > 0 ∧ v > 0 ∧ w > 0 →  -- Ensure positive distances
  u^2 + v^2 = w^2 →  -- Given condition
  w^2 + Real.sqrt 3 * u * v = a^2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_point_distance_l1892_189219


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l1892_189215

theorem sum_of_fractions_equals_seven : 
  let S := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
           1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 
           1 / (Real.sqrt 12 - 3)
  S = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l1892_189215


namespace NUMINAMATH_CALUDE_number_of_black_balls_is_random_variable_l1892_189218

-- Define the total number of balls
def total_balls : ℕ := 8

-- Define the number of black balls
def black_balls : ℕ := 5

-- Define the number of white balls
def white_balls : ℕ := 3

-- Define the number of balls drawn
def drawn_balls : ℕ := 2

-- Define the possible outcomes for the number of black balls drawn
def possible_outcomes : Set ℕ := {0, 1, 2}

-- Define a random variable as a function from the sample space to the set of real numbers
def is_random_variable (X : Set ℕ → ℝ) : Prop :=
  ∀ n ∈ possible_outcomes, X {n} ∈ Set.range X

-- State the theorem
theorem number_of_black_balls_is_random_variable :
  ∃ X : Set ℕ → ℝ, is_random_variable X ∧ 
  (∀ n, X {n} = n) ∧
  (∀ n ∉ possible_outcomes, X {n} = 0) :=
sorry

end NUMINAMATH_CALUDE_number_of_black_balls_is_random_variable_l1892_189218


namespace NUMINAMATH_CALUDE_max_lateral_area_cylinder_l1892_189269

/-- The maximum lateral area of a cylinder with a rectangular cross-section of perimeter 4 is π. -/
theorem max_lateral_area_cylinder (r h : ℝ) : 
  r > 0 → h > 0 → 2 * (2 * r + h) = 4 → 2 * π * r * h ≤ π := by
  sorry

end NUMINAMATH_CALUDE_max_lateral_area_cylinder_l1892_189269


namespace NUMINAMATH_CALUDE_quarter_value_percentage_l1892_189256

theorem quarter_value_percentage (num_dimes : ℕ) (num_quarters : ℕ) 
  (h1 : num_dimes = 75) (h2 : num_quarters = 30) : 
  (num_quarters * 25) / (num_dimes * 10 + num_quarters * 25) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quarter_value_percentage_l1892_189256


namespace NUMINAMATH_CALUDE_special_divisors_count_l1892_189226

/-- The base number -/
def base : ℕ := 540

/-- The exponent of the base number -/
def exponent : ℕ := 540

/-- The number of divisors we're looking for -/
def target_divisors : ℕ := 108

/-- A function that counts the number of positive integer divisors of base^exponent 
    that are divisible by exactly target_divisors positive integers -/
def count_special_divisors (base exponent target_divisors : ℕ) : ℕ := sorry

/-- The main theorem stating that the count of special divisors is 6 -/
theorem special_divisors_count : 
  count_special_divisors base exponent target_divisors = 6 := by sorry

end NUMINAMATH_CALUDE_special_divisors_count_l1892_189226


namespace NUMINAMATH_CALUDE_red_flesh_probability_l1892_189225

/-- Represents the probability of a tomato having yellow skin -/
def yellow_skin_prob : ℚ := 3/8

/-- Represents the probability of a tomato having red flesh given it has yellow skin -/
def red_flesh_given_yellow_skin_prob : ℚ := 8/15

/-- Represents the probability of a tomato having yellow skin given it doesn't have red flesh -/
def yellow_skin_given_not_red_flesh_prob : ℚ := 7/30

/-- Theorem stating that the probability of red flesh is 1/4 given the conditions -/
theorem red_flesh_probability :
  let yellow_and_not_red : ℚ := yellow_skin_prob * (1 - red_flesh_given_yellow_skin_prob)
  let not_red_flesh_prob : ℚ := yellow_and_not_red / yellow_skin_given_not_red_flesh_prob
  let red_flesh_prob : ℚ := 1 - not_red_flesh_prob
  red_flesh_prob = 1/4 := by sorry

end NUMINAMATH_CALUDE_red_flesh_probability_l1892_189225


namespace NUMINAMATH_CALUDE_round_and_convert_0_000359_l1892_189243

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_normalized : 1 ≤ coefficient ∧ coefficient < 10

/-- Rounds a real number to a given number of significant figures -/
def round_to_sig_figs (x : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem round_and_convert_0_000359 :
  let rounded := round_to_sig_figs 0.000359 2
  let scientific := to_scientific_notation rounded
  scientific.coefficient = 3.6 ∧ scientific.exponent = -4 := by
  sorry

end NUMINAMATH_CALUDE_round_and_convert_0_000359_l1892_189243


namespace NUMINAMATH_CALUDE_sqrt_two_minus_one_power_l1892_189299

theorem sqrt_two_minus_one_power (n : ℤ) :
  ∃ k : ℤ, (Real.sqrt 2 - 1) ^ n = Real.sqrt (k + 1) - Real.sqrt k := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_one_power_l1892_189299


namespace NUMINAMATH_CALUDE_transformed_sine_graph_sum_l1892_189210

theorem transformed_sine_graph_sum (ω A a φ : Real) : 
  ω > 0 → A > 0 → a > 0 → 0 < φ → φ < Real.pi →
  (∀ x, A * Real.sin (ω * x - φ) + a = 3 * Real.sin (2 * x - Real.pi / 6) + 1) →
  A + a + ω + φ = 16 / 3 + 11 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_transformed_sine_graph_sum_l1892_189210


namespace NUMINAMATH_CALUDE_farm_animals_l1892_189284

theorem farm_animals (total_animals : ℕ) (num_ducks : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 11)
  (h2 : num_ducks = 6)
  (h3 : total_legs = 32) :
  total_animals - num_ducks = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l1892_189284


namespace NUMINAMATH_CALUDE_father_son_age_relation_l1892_189201

theorem father_son_age_relation :
  ∃ (x : ℕ),
    (38 - x : ℕ) = 7 * (14 - x : ℕ) ∧
    x = 10
  := by sorry

end NUMINAMATH_CALUDE_father_son_age_relation_l1892_189201


namespace NUMINAMATH_CALUDE_largest_domain_of_g_l1892_189258

def is_valid_domain (S : Set ℝ) (g : ℝ → ℝ) : Prop :=
  ∀ x ∈ S, x^2 ∈ S ∧ 1/x^2 ∈ S ∧ g x + g (1/x^2) = x^2

theorem largest_domain_of_g :
  ∃! S : Set ℝ, is_valid_domain S g ∧
    ∀ T : Set ℝ, is_valid_domain T g → T ⊆ S :=
by
  sorry

#check largest_domain_of_g

end NUMINAMATH_CALUDE_largest_domain_of_g_l1892_189258


namespace NUMINAMATH_CALUDE_two_car_garage_count_l1892_189297

theorem two_car_garage_count (total_houses : ℕ) (pool_houses : ℕ) (garage_and_pool : ℕ) (neither : ℕ) :
  total_houses = 65 →
  pool_houses = 40 →
  garage_and_pool = 35 →
  neither = 10 →
  ∃ (garage_houses : ℕ), garage_houses = 50 ∧ 
    total_houses = garage_houses + pool_houses - garage_and_pool + neither :=
by sorry

end NUMINAMATH_CALUDE_two_car_garage_count_l1892_189297


namespace NUMINAMATH_CALUDE_vector_computation_l1892_189283

def a : Fin 2 → ℝ := ![2, 4]
def b : Fin 2 → ℝ := ![-1, 1]

theorem vector_computation : 
  (2 • a - b) = ![5, 7] := by sorry

end NUMINAMATH_CALUDE_vector_computation_l1892_189283


namespace NUMINAMATH_CALUDE_number_of_schnauzers_l1892_189298

/-- Given the number of Doberman puppies and an equation relating it to the number of Schnauzers,
    this theorem proves the number of Schnauzers. -/
theorem number_of_schnauzers (D S : ℤ) (h1 : 3*D - 5 + (D - S) = 90) (h2 : D = 20) : S = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_of_schnauzers_l1892_189298


namespace NUMINAMATH_CALUDE_storage_unit_solution_l1892_189228

/-- Represents the storage unit problem -/
def storage_unit_problem (total_units : ℕ) (small_units : ℕ) (small_length : ℕ) (small_width : ℕ) (large_area : ℕ) : Prop :=
  let small_area : ℕ := small_length * small_width
  let large_units : ℕ := total_units - small_units
  let total_area : ℕ := small_units * small_area + large_units * large_area
  total_area = 5040

/-- Theorem stating the solution to the storage unit problem -/
theorem storage_unit_solution : storage_unit_problem 42 20 8 4 200 := by
  sorry


end NUMINAMATH_CALUDE_storage_unit_solution_l1892_189228


namespace NUMINAMATH_CALUDE_difference_of_roots_absolute_value_l1892_189217

theorem difference_of_roots_absolute_value (a b c : ℝ) (h : a ≠ 0) :
  let r₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 1 ∧ b = -7 ∧ c = 10 → |r₁ - r₂| = 3 :=
by sorry

end NUMINAMATH_CALUDE_difference_of_roots_absolute_value_l1892_189217


namespace NUMINAMATH_CALUDE_sine_graph_shift_l1892_189274

theorem sine_graph_shift (x : ℝ) :
  3 * Real.sin (2 * x - π / 6) = 3 * Real.sin (2 * (x - π / 12)) :=
by sorry

#check sine_graph_shift

end NUMINAMATH_CALUDE_sine_graph_shift_l1892_189274


namespace NUMINAMATH_CALUDE_max_monthly_profit_l1892_189216

/-- Represents the monthly profit function for Xiao Ming's eye-protecting desk lamp business. -/
def monthly_profit (x : ℝ) : ℝ := -10 * x^2 + 700 * x - 10000

/-- Represents the monthly sales volume function. -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 500

/-- The cost price of each lamp. -/
def cost_price : ℝ := 20

/-- The maximum allowed profit percentage. -/
def max_profit_percentage : ℝ := 0.6

/-- Theorem stating the maximum monthly profit and the corresponding selling price. -/
theorem max_monthly_profit :
  ∃ (max_profit : ℝ) (optimal_price : ℝ),
    max_profit = 2160 ∧
    optimal_price = 32 ∧
    (∀ x : ℝ, cost_price ≤ x ∧ x ≤ cost_price * (1 + max_profit_percentage) →
      monthly_profit x ≤ max_profit) ∧
    monthly_profit optimal_price = max_profit :=
  sorry

/-- Lemma: The monthly profit function is correctly defined based on the given conditions. -/
lemma profit_function_correct :
  ∀ x : ℝ, monthly_profit x = (x - cost_price) * sales_volume x :=
  sorry

/-- Lemma: The selling price is within the specified range. -/
lemma selling_price_range :
  ∀ x : ℝ, monthly_profit x > 0 → cost_price ≤ x ∧ x ≤ cost_price * (1 + max_profit_percentage) :=
  sorry

end NUMINAMATH_CALUDE_max_monthly_profit_l1892_189216


namespace NUMINAMATH_CALUDE_min_sum_given_product_l1892_189238

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → a + b = a * b → (∀ x y : ℝ, x > 0 → y > 0 → x + y = x * y → a + b ≤ x + y) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l1892_189238


namespace NUMINAMATH_CALUDE_smallest_base_for_150_l1892_189280

theorem smallest_base_for_150 :
  ∃ b : ℕ, b = 6 ∧ b^2 ≤ 150 ∧ 150 < b^3 ∧ ∀ n : ℕ, n < b → (n^2 > 150 ∨ 150 ≥ n^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_150_l1892_189280


namespace NUMINAMATH_CALUDE_book_profit_rate_l1892_189266

/-- Calculates the rate of profit given cost price and selling price -/
def rate_of_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at 50 rupees and sold at 80 rupees is 60% -/
theorem book_profit_rate :
  let cost_price : ℚ := 50
  let selling_price : ℚ := 80
  rate_of_profit cost_price selling_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_profit_rate_l1892_189266


namespace NUMINAMATH_CALUDE_unique_divisor_function_exists_l1892_189262

open Nat

/-- The divisor function τ(n) counts the number of positive divisors of n. -/
noncomputable def tau (n : ℕ) : ℕ := (divisors n).card

/-- 
Given a finite set of natural numbers, there exists a number x such that 
the divisor function τ applied to the product of x and any element of the set 
yields a unique result for each element of the set.
-/
theorem unique_divisor_function_exists (S : Finset ℕ) : 
  ∃ x : ℕ, ∀ s₁ s₂ : ℕ, s₁ ∈ S → s₂ ∈ S → s₁ ≠ s₂ → tau (s₁ * x) ≠ tau (s₂ * x) := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_function_exists_l1892_189262


namespace NUMINAMATH_CALUDE_nursery_school_students_l1892_189213

theorem nursery_school_students (total : ℕ) 
  (h1 : (total : ℚ) / 10 = (total - (total - 50) : ℚ))
  (h2 : total - 50 ≥ 20) : total = 300 := by
  sorry

end NUMINAMATH_CALUDE_nursery_school_students_l1892_189213


namespace NUMINAMATH_CALUDE_circle_diameter_problem_l1892_189211

/-- Given two circles A and B where A is inside B, proves that the diameter of A
    satisfies the given conditions. -/
theorem circle_diameter_problem (center_distance : ℝ) (diameter_B : ℝ) :
  center_distance = 5 →
  diameter_B = 20 →
  let radius_B := diameter_B / 2
  let area_B := π * radius_B ^ 2
  ∃ (radius_A : ℝ),
    π * radius_A ^ 2 * 6 = area_B ∧
    (2 * radius_A : ℝ) = 2 * Real.sqrt (50 / 3) := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_problem_l1892_189211


namespace NUMINAMATH_CALUDE_inequality_relation_l1892_189241

theorem inequality_relation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b, a > b → 1/a < 1/b) ∧ ¬(∀ a b, 1/a < 1/b → a > b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relation_l1892_189241


namespace NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l1892_189250

/-- The smallest positive integer divisible by all integers from 1 to 10 -/
def smallestDivisibleByOneToTen : ℕ := 2520

/-- Proposition: smallestDivisibleByOneToTen is the smallest positive integer 
    divisible by all integers from 1 to 10 -/
theorem smallest_divisible_by_one_to_ten :
  ∀ n : ℕ, n > 0 → (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) → smallestDivisibleByOneToTen ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l1892_189250


namespace NUMINAMATH_CALUDE_password_probability_l1892_189206

/-- The number of digits in the password -/
def password_length : ℕ := 6

/-- The set of possible even digits for the last position -/
def even_digits : Set ℕ := {0, 2, 4, 6, 8}

/-- The probability of guessing the correct password in one attempt, given the last digit is even -/
def prob_correct_first_attempt : ℚ := 1 / 5

/-- The probability of guessing the correct password in exactly two attempts, given the last digit is even -/
def prob_correct_second_attempt : ℚ := 4 / 25

/-- The probability of guessing the correct password in no more than two attempts, given the last digit is even -/
def prob_correct_within_two_attempts : ℚ := prob_correct_first_attempt + prob_correct_second_attempt

theorem password_probability : prob_correct_within_two_attempts = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_password_probability_l1892_189206


namespace NUMINAMATH_CALUDE_divisor_problem_l1892_189224

theorem divisor_problem (n d k q : ℤ) : 
  n = 25 * k + 4 →
  n + 15 = d * q + 4 →
  d > 0 →
  d = 19 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1892_189224


namespace NUMINAMATH_CALUDE_incorrect_calculation_l1892_189293

theorem incorrect_calculation (a : ℝ) : a^3 + a^3 ≠ 2*a^6 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l1892_189293


namespace NUMINAMATH_CALUDE_f_inequality_l1892_189200

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_increasing : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y
axiom f_even_shift : ∀ x, f (x + 2) = f (2 - x)

-- State the theorem
theorem f_inequality : f 3.5 < f 1 ∧ f 1 < f 2.5 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1892_189200


namespace NUMINAMATH_CALUDE_five_people_arrangement_with_restriction_l1892_189245

/-- The number of ways to arrange n people in a line. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line where two specific people are always adjacent. -/
def adjacentArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of ways to arrange n people in a line where two specific people are not adjacent. -/
def nonAdjacentArrangements (n : ℕ) : ℕ := totalArrangements n - adjacentArrangements n

theorem five_people_arrangement_with_restriction :
  nonAdjacentArrangements 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_five_people_arrangement_with_restriction_l1892_189245


namespace NUMINAMATH_CALUDE_rahul_share_l1892_189272

/-- Calculates the share of payment for a worker in a joint task --/
def calculateShare (daysToComplete : ℚ) (totalDays : ℚ) (totalPayment : ℚ) : ℚ :=
  (totalDays / daysToComplete) * totalPayment / (totalDays / daysToComplete + totalDays / 2)

/-- Proves that Rahul's share of the payment is $42 --/
theorem rahul_share :
  let rahulDays : ℚ := 3
  let rajeshDays : ℚ := 2
  let totalPayment : ℚ := 105
  calculateShare rahulDays (rahulDays * rajeshDays / (rahulDays + rajeshDays)) totalPayment = 42 := by
  sorry

#eval calculateShare 3 (3 * 2 / (3 + 2)) 105

end NUMINAMATH_CALUDE_rahul_share_l1892_189272


namespace NUMINAMATH_CALUDE_least_positive_tangent_inverse_l1892_189233

theorem least_positive_tangent_inverse (y p q : ℝ) (h1 : Real.tan y = p / q) (h2 : Real.tan (3 * y) = q / (p + q)) :
  ∃ m : ℝ, m > 0 ∧ y = Real.arctan m ∧ ∀ m' : ℝ, m' > 0 → y = Real.arctan m' → m ≤ m' ∧ m = 1 :=
sorry

end NUMINAMATH_CALUDE_least_positive_tangent_inverse_l1892_189233


namespace NUMINAMATH_CALUDE_triangle_cosine_identity_l1892_189223

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  -- Ensure A + B + C = π (180 degrees)
  angle_sum : A + B + C = Real.pi
  -- Ensure all angles are positive
  A_pos : A > 0
  B_pos : B > 0
  C_pos : C > 0
  -- Ensure the given condition 2b = a + c
  side_condition : 2 * b = a + c

-- Theorem statement
theorem triangle_cosine_identity (t : Triangle) :
  5 * Real.cos t.A - 4 * Real.cos t.A * Real.cos t.C + 5 * Real.cos t.C = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_identity_l1892_189223


namespace NUMINAMATH_CALUDE_triangle_area_72_l1892_189237

/-- 
Given a right triangle with vertices (0, 0), (x, 3x), and (x, 0),
prove that if its area is 72 square units and x > 0, then x = 4√3.
-/
theorem triangle_area_72 (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 72 → x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_72_l1892_189237


namespace NUMINAMATH_CALUDE_reseating_problem_l1892_189265

/-- Number of ways n people can be reseated according to the rules -/
def S : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | n + 3 => S (n + 2) + S (n + 1)

/-- The reseating problem for 12 people -/
theorem reseating_problem : S 12 = 89 := by
  sorry

end NUMINAMATH_CALUDE_reseating_problem_l1892_189265


namespace NUMINAMATH_CALUDE_unique_base_twelve_l1892_189278

/-- Given a base b ≥ 10, this function checks if the equation 166 × 56 = 8590 is valid in base b -/
def is_valid_equation (b : ℕ) : Prop :=
  b ≥ 10 ∧ 
  (1 * b^2 + 6 * b + 6) * (5 * b + 6) = 8 * b^3 + 5 * b^2 + 9 * b + 0

/-- Theorem stating that 12 is the only base ≥ 10 satisfying the equation -/
theorem unique_base_twelve : 
  (∃ (b : ℕ), is_valid_equation b) ∧ 
  (∀ (b : ℕ), is_valid_equation b → b = 12) := by
  sorry

#check unique_base_twelve

end NUMINAMATH_CALUDE_unique_base_twelve_l1892_189278


namespace NUMINAMATH_CALUDE_chopstick_length_l1892_189294

/-- The length of a chopstick given specific wetness conditions -/
theorem chopstick_length (wetted_length : ℝ) (h1 : wetted_length = 8) 
  (h2 : wetted_length + wetted_length / 2 + wetted_length = 24) : ℝ :=
by
  sorry

#check chopstick_length

end NUMINAMATH_CALUDE_chopstick_length_l1892_189294


namespace NUMINAMATH_CALUDE_factorial_prime_factorization_l1892_189275

theorem factorial_prime_factorization (x a k m p : ℕ) : 
  x = Nat.factorial 8 →
  x = 2^a * 3^k * 5^m * 7^p →
  a > 0 ∧ k > 0 ∧ m > 0 ∧ p > 0 →
  a + k + m + p = 11 →
  a = 7 := by
sorry

end NUMINAMATH_CALUDE_factorial_prime_factorization_l1892_189275


namespace NUMINAMATH_CALUDE_gcd_of_136_and_1275_l1892_189205

theorem gcd_of_136_and_1275 : Nat.gcd 136 1275 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_136_and_1275_l1892_189205


namespace NUMINAMATH_CALUDE_first_patient_sessions_l1892_189232

/-- Given a group of patients and their session requirements, prove the number of sessions for the first patient. -/
theorem first_patient_sessions
  (total_patients : ℕ)
  (total_sessions : ℕ)
  (patient2_sessions : ℕ → ℕ)
  (remaining_patients_sessions : ℕ)
  (h1 : total_patients = 4)
  (h2 : total_sessions = 25)
  (h3 : patient2_sessions x = x + 5)
  (h4 : remaining_patients_sessions = 8 + 8)
  (h5 : x + patient2_sessions x + remaining_patients_sessions = total_sessions) :
  x = 2 :=
by sorry

end NUMINAMATH_CALUDE_first_patient_sessions_l1892_189232


namespace NUMINAMATH_CALUDE_median_of_four_numbers_l1892_189230

theorem median_of_four_numbers (x : ℝ) : 
  (0 < 4) ∧ (4 < x) ∧ (x < 10) ∧  -- ascending order condition
  ((4 + x) / 2 = 5)                -- median condition
  → x = 6 := by
sorry

end NUMINAMATH_CALUDE_median_of_four_numbers_l1892_189230


namespace NUMINAMATH_CALUDE_science_club_membership_l1892_189222

theorem science_club_membership (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ)
  (h1 : total = 120)
  (h2 : math = 75)
  (h3 : physics = 50)
  (h4 : both = 15) :
  total - (math + physics - both) = 10 := by
sorry

end NUMINAMATH_CALUDE_science_club_membership_l1892_189222


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1892_189246

/-- The standard equation of a hyperbola with the same asymptotes as x²/9 - y²/16 = 1
    and passing through the point (-√3, 2√3) -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ m : ℝ, x^2 / 9 - y^2 / 16 = m) ∧
  ((-Real.sqrt 3)^2 / 9 - (2 * Real.sqrt 3)^2 / 16 = -5/12) →
  y^2 / 5 - x^2 / (15/4) = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1892_189246


namespace NUMINAMATH_CALUDE_smallest_n_with_square_and_fifth_power_l1892_189286

theorem smallest_n_with_square_and_fifth_power : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (x : ℕ), 2 * n = x^2) ∧ 
    (∃ (y : ℕ), 3 * n = y^5)) →
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), 2 * m = x^2) → 
    (∃ (y : ℕ), 3 * m = y^5) → 
    m ≥ 2592) ∧
  (∃ (x : ℕ), 2 * 2592 = x^2) ∧ 
  (∃ (y : ℕ), 3 * 2592 = y^5) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_square_and_fifth_power_l1892_189286


namespace NUMINAMATH_CALUDE_unique_w_exists_l1892_189220

theorem unique_w_exists : ∃! w : ℝ, w > 0 ∧ 
  (Real.sqrt 1.5) / (Real.sqrt 0.81) + (Real.sqrt 1.44) / (Real.sqrt w) = 3.0751133491652576 := by
  sorry

end NUMINAMATH_CALUDE_unique_w_exists_l1892_189220


namespace NUMINAMATH_CALUDE_santos_salvadore_earnings_ratio_l1892_189204

/-- Proves that the ratio of Santo's earnings to Salvadore's earnings is 1:2 -/
theorem santos_salvadore_earnings_ratio :
  let salvadore_earnings : ℚ := 1956
  let total_earnings : ℚ := 2934
  let santo_earnings : ℚ := total_earnings - salvadore_earnings
  santo_earnings / salvadore_earnings = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_santos_salvadore_earnings_ratio_l1892_189204


namespace NUMINAMATH_CALUDE_number_puzzle_l1892_189207

theorem number_puzzle : ∃ x : ℝ, (20 * (x / 5) = 40) ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1892_189207


namespace NUMINAMATH_CALUDE_friendly_number_F_formula_max_friendly_N_l1892_189257

def is_friendly_number (M : ℕ) : Prop :=
  ∃ (a b c d : ℕ), M = 1000 * a + 100 * b + 10 * c + d ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ a - b = c - d

def F (M : ℕ) : ℤ :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  100 * a - 100 * b - 10 * b + c - d

theorem friendly_number_F_formula (M : ℕ) (h : is_friendly_number M) :
  F M = 100 * (M / 1000) - 110 * ((M / 100) % 10) + (M / 10) % 10 - M % 10 :=
sorry

def N (x y m n : ℕ) : ℕ := 1000 * x + 100 * y + 30 * m + n + 1001

theorem max_friendly_N (x y m n : ℕ) 
  (h1 : 0 ≤ y ∧ y < x ∧ x ≤ 8) 
  (h2 : 0 ≤ m ∧ m ≤ 3) 
  (h3 : 0 ≤ n ∧ n ≤ 8) 
  (h4 : is_friendly_number (N x y m n)) 
  (h5 : F (N x y m n) % 5 = 1) :
  N x y m n ≤ 9696 :=
sorry

end NUMINAMATH_CALUDE_friendly_number_F_formula_max_friendly_N_l1892_189257


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l1892_189239

theorem three_digit_number_problem : ∃ (a b c : ℕ), 
  (8 * a + 5 * b + c = 100) ∧ 
  (a + b + c = 20) ∧ 
  (a < 10) ∧ (b < 10) ∧ (c < 10) ∧
  (a * 100 + b * 10 + c = 866) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l1892_189239


namespace NUMINAMATH_CALUDE_triangle_altitude_reciprocal_sum_bounds_l1892_189289

/-- For any triangle, the sum of the reciprocals of two altitudes lies between the reciprocal of the radius of the inscribed circle and the reciprocal of its diameter. -/
theorem triangle_altitude_reciprocal_sum_bounds (a b c m_a m_b m_c ρ s t : ℝ) 
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_altitudes : m_a > 0 ∧ m_b > 0 ∧ m_c > 0)
  (h_perimeter : a + b + c = 2 * s)
  (h_area : t > 0)
  (h_inscribed_radius : ρ > 0)
  (h_altitude_a : a * m_a = 2 * t)
  (h_altitude_b : b * m_b = 2 * t)
  (h_altitude_c : c * m_c = 2 * t)
  (h_inscribed_radius_def : s * ρ = t) :
  1 / (2 * ρ) < 1 / m_a + 1 / m_b ∧ 1 / m_a + 1 / m_b < 1 / ρ :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_reciprocal_sum_bounds_l1892_189289


namespace NUMINAMATH_CALUDE_gecko_eats_15_bugs_l1892_189268

/-- The number of bugs eaten by various creatures in a garden --/
structure GardenBugs where
  gecko : ℕ
  lizard : ℕ
  frog : ℕ
  toad : ℕ

/-- The conditions of the bug-eating scenario in the garden --/
def validGardenBugs (bugs : GardenBugs) : Prop :=
  bugs.lizard = bugs.gecko / 2 ∧
  bugs.frog = 3 * bugs.lizard ∧
  bugs.toad = (3 * bugs.frog) / 2 ∧
  bugs.gecko + bugs.lizard + bugs.frog + bugs.toad = 63

/-- The theorem stating that the gecko eats 15 bugs --/
theorem gecko_eats_15_bugs :
  ∃ (bugs : GardenBugs), validGardenBugs bugs ∧ bugs.gecko = 15 := by
  sorry

end NUMINAMATH_CALUDE_gecko_eats_15_bugs_l1892_189268


namespace NUMINAMATH_CALUDE_largest_three_digit_integer_l1892_189202

theorem largest_three_digit_integer (n : ℕ) (a b c : ℕ) : 
  n = 100 * a + 10 * b + c →
  100 ≤ n → n < 1000 →
  2 ∣ a →
  3 ∣ (10 * a + b) →
  ¬(6 ∣ (10 * a + b)) →
  5 ∣ n →
  ¬(7 ∣ n) →
  n ≤ 870 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_integer_l1892_189202


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1892_189203

theorem polynomial_evaluation :
  let x : ℝ := -2
  x^4 + x^3 + x^2 + x + 2 = 12 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1892_189203


namespace NUMINAMATH_CALUDE_joan_payment_l1892_189264

/-- The amount Joan paid for her purchases, given the costs and change received -/
def amount_paid (cat_toy_cost cage_cost change : ℚ) : ℚ :=
  cat_toy_cost + cage_cost - change

/-- Theorem stating that Joan paid $19.48 for her purchases -/
theorem joan_payment : amount_paid 8.77 10.97 0.26 = 19.48 := by
  sorry

end NUMINAMATH_CALUDE_joan_payment_l1892_189264


namespace NUMINAMATH_CALUDE_factor_implies_a_value_l1892_189260

theorem factor_implies_a_value (a b : ℝ) :
  (∀ x : ℝ, (x^2 + x - 6) ∣ (2*x^4 + x^3 - a*x^2 + b*x + a + b - 1)) →
  a = 16 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_a_value_l1892_189260


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l1892_189231

/-- Given collinear points A, B, C, D, and E with specified distances between them,
    this function calculates the sum of squared distances from these points to a point P on AE. -/
def sum_of_squared_distances (x : ℝ) : ℝ :=
  x^2 + (x - 2)^2 + (x - 4)^2 + (x - 7)^2 + (x - 11)^2

/-- The theorem states that the minimum value of the sum of squared distances
    from points A, B, C, D, and E to any point P on line segment AE is 54.8,
    given the specified distances between the points. -/
theorem min_sum_squared_distances :
  ∃ (min_value : ℝ), min_value = 54.8 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 11 → sum_of_squared_distances x ≥ min_value :=
sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l1892_189231


namespace NUMINAMATH_CALUDE_ants_sugar_harvesting_l1892_189288

def sugar_harvesting (initial_sugar : ℝ) (removal_rate : ℝ) (time_passed : ℝ) : Prop :=
  let remaining_sugar := initial_sugar - removal_rate * time_passed
  let remaining_time := remaining_sugar / removal_rate
  remaining_time = 3

theorem ants_sugar_harvesting :
  sugar_harvesting 24 4 3 :=
sorry

end NUMINAMATH_CALUDE_ants_sugar_harvesting_l1892_189288


namespace NUMINAMATH_CALUDE_birthday_paradox_l1892_189292

theorem birthday_paradox (people : Finset ℕ) (birthdays : ℕ → Fin 366) :
  people.card = 367 → ∃ i j : ℕ, i ∈ people ∧ j ∈ people ∧ i ≠ j ∧ birthdays i = birthdays j :=
sorry

end NUMINAMATH_CALUDE_birthday_paradox_l1892_189292


namespace NUMINAMATH_CALUDE_smartphone_sale_price_l1892_189279

theorem smartphone_sale_price (initial_cost : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  initial_cost = 300 →
  loss_percentage = 15 →
  selling_price = initial_cost - (loss_percentage / 100) * initial_cost →
  selling_price = 255 := by
sorry

end NUMINAMATH_CALUDE_smartphone_sale_price_l1892_189279


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l1892_189227

/-- Given vectors a and b, where a is parallel to b, prove the minimum value of 1/m + 8/n is 9/2 -/
theorem min_value_parallel_vectors (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (4 - n, 2)
  (∃ (k : ℝ), a = k • b) → 
  (∀ (x y : ℝ), x > 0 → y > 0 → 2 * x + y = 4 → 1/x + 8/y ≥ 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l1892_189227


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1892_189254

/-- Given a line segment with midpoint (2, -3) and one endpoint (3, 1),
    prove that the other endpoint is (1, -7) -/
theorem line_segment_endpoint (x₁ y₁ x₂ y₂ : ℝ) :
  (x₁ = 3 ∧ y₁ = 1) →  -- One endpoint is (3, 1)
  ((x₁ + x₂) / 2 = 2 ∧ (y₁ + y₂) / 2 = -3) →  -- Midpoint is (2, -3)
  x₂ = 1 ∧ y₂ = -7  -- Other endpoint is (1, -7)
  := by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1892_189254


namespace NUMINAMATH_CALUDE_function_decomposition_l1892_189271

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ g h : ℝ → ℝ, 
    (∀ x, f x = g x + h x) ∧ 
    (∀ x, g (-x) = g x) ∧ 
    (∀ x, h (-x) = -h x) := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l1892_189271


namespace NUMINAMATH_CALUDE_shoes_discount_percentage_l1892_189291

/-- Given the original price and sale price of an item, calculate the discount percentage. -/
def discount_percentage (original_price sale_price : ℚ) : ℚ :=
  (original_price - sale_price) / original_price * 100

/-- Theorem: The discount percentage for shoes with original price $204 and sale price $51 is 75%. -/
theorem shoes_discount_percentage :
  discount_percentage 204 51 = 75 := by sorry

end NUMINAMATH_CALUDE_shoes_discount_percentage_l1892_189291


namespace NUMINAMATH_CALUDE_evaluate_expression_l1892_189281

theorem evaluate_expression : -(16 / 4 * 7 + 25 - 2 * 7) = -39 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1892_189281


namespace NUMINAMATH_CALUDE_february_to_january_ratio_l1892_189242

def january_bill : ℚ := 180

def february_bill : ℚ := 270

theorem february_to_january_ratio :
  (february_bill / january_bill) = 3 / 2 ∧
  ((february_bill + 30) / january_bill) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_february_to_january_ratio_l1892_189242


namespace NUMINAMATH_CALUDE_james_total_score_l1892_189214

theorem james_total_score (field_goals : ℕ) (two_point_shots : ℕ) : field_goals = 13 → two_point_shots = 20 → field_goals * 3 + two_point_shots * 2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_james_total_score_l1892_189214


namespace NUMINAMATH_CALUDE_not_perfect_square_l1892_189221

theorem not_perfect_square (n : ℕ) : ¬ ∃ m : ℤ, (3^n : ℤ) + 2 * (17^n : ℤ) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1892_189221


namespace NUMINAMATH_CALUDE_school_athletes_equation_l1892_189255

/-- 
Given a school with x athletes divided into y groups, prove that the following system of equations holds:
7y = x - 3
8y = x + 5
-/
theorem school_athletes_equation (x y : ℕ) 
  (h1 : 7 * y = x - 3)  -- If there are 7 people in each group, there will be 3 people left over
  (h2 : 8 * y = x + 5)  -- If there are 8 people in each group, there will be a shortage of 5 people
  : 7 * y = x - 3 ∧ 8 * y = x + 5 := by
  sorry

end NUMINAMATH_CALUDE_school_athletes_equation_l1892_189255


namespace NUMINAMATH_CALUDE_paula_shopping_theorem_l1892_189247

/-- Calculates the remaining money after Paula's shopping trip -/
def remaining_money (initial_amount : ℕ) (num_shirts : ℕ) (shirt_price : ℕ) 
  (num_pants : ℕ) (pants_price : ℕ) : ℕ :=
  initial_amount - (num_shirts * shirt_price + num_pants * pants_price)

/-- Proves that Paula has $100 left after her shopping trip -/
theorem paula_shopping_theorem :
  remaining_money 250 5 15 3 25 = 100 := by
  sorry

end NUMINAMATH_CALUDE_paula_shopping_theorem_l1892_189247


namespace NUMINAMATH_CALUDE_candy_bar_profit_l1892_189261

/-- Calculates the profit from selling candy bars -/
def candy_profit (
  num_bars : ℕ
  ) (purchase_price : ℚ)
    (selling_price : ℚ)
    (sales_fee : ℚ) : ℚ :=
  num_bars * selling_price - num_bars * purchase_price - num_bars * sales_fee

/-- Theorem stating the profit from the candy bar sale -/
theorem candy_bar_profit :
  candy_profit 800 (3/4) (2/3) (1/20) = -533/5 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l1892_189261


namespace NUMINAMATH_CALUDE_corn_amount_approx_l1892_189290

/-- The cost of corn per pound -/
def corn_cost : ℝ := 1.05

/-- The cost of beans per pound -/
def bean_cost : ℝ := 0.39

/-- The total pounds of corn and beans bought -/
def total_pounds : ℝ := 30

/-- The total cost of the purchase -/
def total_cost : ℝ := 23.10

/-- The amount of corn bought (in pounds) -/
noncomputable def corn_amount : ℝ := 
  (total_cost - bean_cost * total_pounds) / (corn_cost - bean_cost)

theorem corn_amount_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |corn_amount - 17.3| < ε :=
sorry

end NUMINAMATH_CALUDE_corn_amount_approx_l1892_189290


namespace NUMINAMATH_CALUDE_john_cycling_distance_l1892_189244

def base_eight_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8^1 + (n % 10) * 8^0

theorem john_cycling_distance : base_eight_to_decimal 6375 = 3325 := by
  sorry

end NUMINAMATH_CALUDE_john_cycling_distance_l1892_189244


namespace NUMINAMATH_CALUDE_initial_orchids_is_three_l1892_189295

/-- Represents the number of flowers in a vase -/
structure FlowerVase where
  initialRoses : ℕ
  finalRoses : ℕ
  finalOrchids : ℕ
  orchidsCut : ℕ

/-- Calculates the initial number of orchids in the vase -/
def initialOrchids (v : FlowerVase) : ℕ :=
  v.finalOrchids - v.orchidsCut

/-- Theorem stating that the initial number of orchids is 3 -/
theorem initial_orchids_is_three (v : FlowerVase) 
  (h1 : v.initialRoses = 16)
  (h2 : v.finalRoses = 13)
  (h3 : v.finalOrchids = 7)
  (h4 : v.orchidsCut = 4) : 
  initialOrchids v = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_orchids_is_three_l1892_189295


namespace NUMINAMATH_CALUDE_max_weekly_earnings_675_l1892_189209

/-- Represents the maximum weekly earnings for a restaurant worker with specified pay rates and overtime rules. -/
def max_weekly_earnings (max_hours : ℕ) (regular_rate : ℚ) (overtime_rate_1 : ℚ) (overtime_rate_2 : ℚ) : ℚ :=
  let regular_pay := regular_rate * 40
  let overtime_pay_1 := regular_rate * overtime_rate_1 * 10
  let overtime_pay_2 := regular_rate * overtime_rate_2 * 10
  regular_pay + overtime_pay_1 + overtime_pay_2

/-- Theorem stating that the maximum weekly earnings for a restaurant worker under specified conditions is $675. -/
theorem max_weekly_earnings_675 :
  max_weekly_earnings 60 10 1.25 1.5 = 675 := by
  sorry

end NUMINAMATH_CALUDE_max_weekly_earnings_675_l1892_189209


namespace NUMINAMATH_CALUDE_luke_game_points_per_round_l1892_189276

/-- Given a total score and number of rounds in a game where equal points are gained in each round,
    calculate the points gained per round. -/
def points_per_round (total_score : ℕ) (num_rounds : ℕ) : ℚ :=
  total_score / num_rounds

/-- Theorem stating that for Luke's game with 154 total points over 14 rounds,
    the points gained per round is 11. -/
theorem luke_game_points_per_round :
  points_per_round 154 14 = 11 := by
  sorry

end NUMINAMATH_CALUDE_luke_game_points_per_round_l1892_189276


namespace NUMINAMATH_CALUDE_congruence_solution_l1892_189287

theorem congruence_solution (n : ℤ) : 
  0 ≤ n ∧ n < 203 ∧ (150 * n) % 203 = 95 % 203 → n = 144 := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l1892_189287


namespace NUMINAMATH_CALUDE_perpendicular_planes_l1892_189270

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (m n : Line) (α β : Plane) 
  (h1 : perp_line_plane m α) 
  (h2 : perp_line_plane n β) 
  (h3 : perp_line_line m n) : 
  perp_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l1892_189270


namespace NUMINAMATH_CALUDE_total_commute_time_is_16_l1892_189229

/-- Time it takes Roque to walk to work (in hours) -/
def walk_time : ℕ := 2

/-- Time it takes Roque to bike to work (in hours) -/
def bike_time : ℕ := 1

/-- Number of times Roque walks to and from work per week -/
def walk_frequency : ℕ := 3

/-- Number of times Roque bikes to and from work per week -/
def bike_frequency : ℕ := 2

/-- Total time Roque spends commuting in a week -/
def total_commute_time : ℕ := (walk_time * walk_frequency * 2) + (bike_time * bike_frequency * 2)

theorem total_commute_time_is_16 : total_commute_time = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_commute_time_is_16_l1892_189229


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1892_189282

/-- A quadratic function that takes specific values at consecutive natural numbers -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 6 ∧ f (n + 1) = 14 ∧ f (n + 2) = 14

/-- The theorem stating the maximum value of the quadratic function -/
theorem quadratic_max_value (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ c : ℝ, c = 15 ∧ ∀ x : ℝ, f x ≤ c :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1892_189282


namespace NUMINAMATH_CALUDE_triangle_special_angle_l1892_189267

/-- In a triangle ABC, if 2b*cos(A) = 2c - sqrt(3)*a, then the measure of angle B is π/6 --/
theorem triangle_special_angle (a b c : ℝ) (A B : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A) (h5 : A < π) (h6 : 0 < B) (h7 : B < π)
  (h8 : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a) :
  B = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l1892_189267


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l1892_189263

def polynomial (x : ℂ) : ℂ := x^4 - 3*x^3 + 5*x^2 - 27*x - 36

theorem pure_imaginary_solutions :
  ∃ (k : ℝ), k > 0 ∧ 
  polynomial (k * Complex.I) = 0 ∧
  polynomial (-k * Complex.I) = 0 ∧
  ∀ (z : ℂ), polynomial z = 0 → z.re = 0 → z = k * Complex.I ∨ z = -k * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l1892_189263


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l1892_189277

/-- Regular hexagon with vertices A and C -/
structure RegularHexagon where
  A : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a regular hexagon -/
def hexagon_area (h : RegularHexagon) : ℝ := sorry

/-- Theorem: Area of the specific regular hexagon -/
theorem specific_hexagon_area :
  let h : RegularHexagon := { A := (0, 0), C := (8, 2) }
  hexagon_area h = 34 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l1892_189277


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1892_189285

theorem min_value_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1892_189285


namespace NUMINAMATH_CALUDE_max_triangle_area_l1892_189208

/-- The maximum area of a triangle with constrained side lengths -/
theorem max_triangle_area (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 1 ≤ b ∧ b ≤ 2) (hc : 2 ≤ c ∧ c ≤ 3)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (S : ℝ), S ≤ 1 ∧ ∀ (S' : ℝ), (∃ (a' b' c' : ℝ),
    0 ≤ a' ∧ a' ≤ 1 ∧
    1 ≤ b' ∧ b' ≤ 2 ∧
    2 ≤ c' ∧ c' ≤ 3 ∧
    a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧
    S' = (a' * b' * Real.sqrt (1 - (a'*a' + b'*b' - c'*c')^2 / (4*a'*a'*b'*b'))) / 2) →
    S' ≤ S :=
by
  sorry

end NUMINAMATH_CALUDE_max_triangle_area_l1892_189208


namespace NUMINAMATH_CALUDE_balls_in_boxes_l1892_189249

/-- The number of ways to place n different balls into m different boxes, with at most one ball per box -/
def place_balls (n m : ℕ) : ℕ :=
  Nat.descFactorial m n

theorem balls_in_boxes : place_balls 3 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l1892_189249


namespace NUMINAMATH_CALUDE_certain_number_value_l1892_189235

theorem certain_number_value (y : ℕ) :
  (2^14 : ℕ) - (2^y : ℕ) = 3 * (2^12 : ℕ) → y = 13 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l1892_189235


namespace NUMINAMATH_CALUDE_police_officers_on_duty_l1892_189296

theorem police_officers_on_duty 
  (total_female_officers : ℕ)
  (female_duty_percentage : ℚ)
  (female_duty_ratio : ℚ)
  (h1 : total_female_officers = 600)
  (h2 : female_duty_percentage = 17 / 100)
  (h3 : female_duty_ratio = 1 / 2) :
  ∃ (officers_on_duty : ℕ), 
    officers_on_duty = 204 ∧ 
    (officers_on_duty : ℚ) * female_duty_ratio = (total_female_officers : ℚ) * female_duty_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_police_officers_on_duty_l1892_189296


namespace NUMINAMATH_CALUDE_f_properties_l1892_189251

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.cos x ^ 2)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 2)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∃ M, ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ M) ∧
  (∃ m, ∀ x ∈ Set.Icc 0 (Real.pi / 2), m ≤ f x) ∧
  (∃ x₁ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ f x₁) ∧
  (∃ x₂ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x₂ ≤ f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1892_189251


namespace NUMINAMATH_CALUDE_digit_strike_out_theorem_l1892_189240

/-- Represents a positive integer as a list of its digits --/
def DigitList := List Nat

/-- Checks if a number represented as a list of digits is divisible by 9 --/
def isDivisibleBy9 (n : DigitList) : Prop :=
  (n.sum % 9 = 0)

/-- Checks if a number can be obtained by striking out one digit from another number --/
def canBeObtainedByStrikingOut (m n : DigitList) : Prop :=
  ∃ (i : Nat), i < n.length ∧ m = (n.take i ++ n.drop (i+1))

/-- The main theorem --/
theorem digit_strike_out_theorem (N : DigitList) :
  (∃ (M : DigitList), N.sum = 9 * M.sum ∧ 
    canBeObtainedByStrikingOut M N ∧ 
    isDivisibleBy9 M) →
  (∀ (K : DigitList), canBeObtainedByStrikingOut K M → isDivisibleBy9 K) ∧
  (N ∈ [[1,0,1,2,5], [2,0,2,5], [3,0,3,7,5], [4,0,5], [5,0,6,2,5], [6,7,5], [7,0,8,7,5]]) :=
by
  sorry


end NUMINAMATH_CALUDE_digit_strike_out_theorem_l1892_189240


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_right_triangle_perimeter_proof_l1892_189259

/-- The perimeter of a right triangle with legs 8 and 6 is 24. -/
theorem right_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun PQ QR PR =>
    QR = 8 ∧ PR = 6 ∧ PQ ^ 2 = QR ^ 2 + PR ^ 2 →
    PQ + QR + PR = 24

/-- Proof of the theorem -/
theorem right_triangle_perimeter_proof : right_triangle_perimeter 10 8 6 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_right_triangle_perimeter_proof_l1892_189259


namespace NUMINAMATH_CALUDE_annual_population_change_l1892_189253

def town_population (initial_pop : ℕ) (new_people : ℕ) (moved_out : ℕ) (years : ℕ) (final_pop : ℕ) : ℤ :=
  let pop_after_changes : ℤ := initial_pop + new_people - moved_out
  let total_change : ℤ := pop_after_changes - final_pop
  total_change / years

theorem annual_population_change :
  town_population 780 100 400 4 60 = -105 :=
sorry

end NUMINAMATH_CALUDE_annual_population_change_l1892_189253


namespace NUMINAMATH_CALUDE_electrolysis_mass_proportionality_l1892_189236

/-- Represents the mass of metal deposited during electrolysis -/
noncomputable def mass_deposited (current : ℝ) (time : ℝ) (ion_charge : ℝ) : ℝ :=
  sorry

/-- The mass deposited is directly proportional to the current -/
axiom mass_prop_current (time : ℝ) (ion_charge : ℝ) (k : ℝ) :
  ∀ current₁ current₂ : ℝ, mass_deposited (k * current₁) time ion_charge = k * mass_deposited current₂ time ion_charge

/-- The mass deposited is directly proportional to the time -/
axiom mass_prop_time (current : ℝ) (ion_charge : ℝ) (k : ℝ) :
  ∀ time₁ time₂ : ℝ, mass_deposited current (k * time₁) ion_charge = k * mass_deposited current time₂ ion_charge

/-- The mass deposited is inversely proportional to the ion charge -/
axiom mass_inv_prop_charge (current : ℝ) (time : ℝ) (k : ℝ) :
  ∀ charge₁ charge₂ : ℝ, charge₁ ≠ 0 → charge₂ ≠ 0 →
    mass_deposited current time (k * charge₁) = (1 / k) * mass_deposited current time charge₂

theorem electrolysis_mass_proportionality :
  (∀ k current time charge, mass_deposited (k * current) time charge = k * mass_deposited current time charge) ∧
  (∀ k current time charge, mass_deposited current (k * time) charge = k * mass_deposited current time charge) ∧
  ¬(∀ k current time charge, charge ≠ 0 → mass_deposited current time (k * charge) = k * mass_deposited current time charge) :=
by sorry

end NUMINAMATH_CALUDE_electrolysis_mass_proportionality_l1892_189236


namespace NUMINAMATH_CALUDE_a_less_than_b_less_than_one_l1892_189212

theorem a_less_than_b_less_than_one
  (x : ℝ) (a b : ℝ) 
  (hx : x > 0)
  (hab : a^x < b^x)
  (hb1 : b^x < 1)
  (ha_pos : a > 0)
  (hb_pos : b > 0) :
  a < b ∧ b < 1 := by
sorry

end NUMINAMATH_CALUDE_a_less_than_b_less_than_one_l1892_189212
