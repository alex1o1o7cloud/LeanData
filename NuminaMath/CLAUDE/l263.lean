import Mathlib

namespace NUMINAMATH_CALUDE_difference_of_squares_101_99_l263_26349

theorem difference_of_squares_101_99 : 101^2 - 99^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_101_99_l263_26349


namespace NUMINAMATH_CALUDE_total_cost_is_124_80_l263_26343

-- Define the number of lessons and prices for each studio
def total_lessons : Nat := 10
def price_A : ℚ := 15
def price_B : ℚ := 12
def price_C : ℚ := 18

-- Define the number of lessons Tom takes at each studio
def lessons_A : Nat := 4
def lessons_B : Nat := 3
def lessons_C : Nat := 3

-- Define the discount percentage for Studio B
def discount_B : ℚ := 20 / 100

-- Define the number of free lessons at Studio C
def free_lessons_C : Nat := 1

-- Theorem to prove
theorem total_cost_is_124_80 : 
  (lessons_A * price_A) + 
  (lessons_B * price_B * (1 - discount_B)) + 
  ((lessons_C - free_lessons_C) * price_C) = 124.80 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_124_80_l263_26343


namespace NUMINAMATH_CALUDE_calculate_expression_l263_26324

theorem calculate_expression : (1/3)⁻¹ + (2023 - Real.pi)^0 - Real.sqrt 12 * Real.sin (π/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l263_26324


namespace NUMINAMATH_CALUDE_sequence_existence_l263_26398

theorem sequence_existence : ∃ (a b : ℕ → ℕ), 
  (∀ n : ℕ, n ≥ 1 → (
    (0 < a n ∧ a n < a (n + 1)) ∧
    (a n < b n ∧ b n < a n ^ 2) ∧
    ((b n - 1) % (a n - 1) = 0) ∧
    ((b n ^ 2 - 1) % (a n ^ 2 - 1) = 0)
  )) := by
  sorry

end NUMINAMATH_CALUDE_sequence_existence_l263_26398


namespace NUMINAMATH_CALUDE_f_equals_g_l263_26399

-- Define the two functions
def f (x : ℝ) : ℝ := x - 1
def g (t : ℝ) : ℝ := t - 1

-- Theorem statement
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l263_26399


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l263_26386

-- Define the parabola equation
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

-- Define the x-intercept
def a : ℝ := parabola 0

-- Define the y-intercepts
def b_and_c : Set ℝ := {y | parabola y = 0}

-- Theorem statement
theorem parabola_intercepts_sum :
  ∃ (b c : ℝ), b ∈ b_and_c ∧ c ∈ b_and_c ∧ b ≠ c ∧ a + b + c = 7 :=
sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l263_26386


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_progression_l263_26320

theorem geometric_to_arithmetic_progression :
  ∀ (a q : ℝ),
    a > 0 → q > 0 →
    a + a * q + a * q^2 = 105 →
    ∃ d : ℝ, a * q - a = (a * q^2 - 15) - a * q →
    (a = 15 ∧ q = 2) ∨ (a = 60 ∧ q = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_progression_l263_26320


namespace NUMINAMATH_CALUDE_prime_divides_power_plus_one_l263_26384

theorem prime_divides_power_plus_one (n b p : ℕ) :
  n ≠ 0 →
  b ≠ 0 →
  Nat.Prime p →
  Odd p →
  p ∣ b^(2^n) + 1 →
  ∃ m : ℕ, p = 2^(n+1) * m + 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_power_plus_one_l263_26384


namespace NUMINAMATH_CALUDE_no_solution_triple_inequality_l263_26391

theorem no_solution_triple_inequality :
  ¬ ∃ (x y z : ℝ), (|x| < |y - z| ∧ |y| < |z - x| ∧ |z| < |x - y|) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_triple_inequality_l263_26391


namespace NUMINAMATH_CALUDE_apple_distribution_l263_26328

theorem apple_distribution (total_apples : ℕ) (total_bags : ℕ) (x : ℕ) :
  total_apples = 109 →
  total_bags = 20 →
  (∃ k : ℕ, k * x + (total_bags - k) * 3 = total_apples ∧ 0 < k ∧ k ≤ total_bags) →
  (x = 10 ∨ x = 52) :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l263_26328


namespace NUMINAMATH_CALUDE_negation_equivalence_l263_26373

-- Define the original proposition
def original_proposition : Prop := ∀ x : ℝ, x > Real.sin x

-- Define the negation of the original proposition
def negation_proposition : Prop := ∃ x : ℝ, x ≤ Real.sin x

-- Theorem stating that the negation of the original proposition is equivalent to the negation_proposition
theorem negation_equivalence : ¬original_proposition ↔ negation_proposition := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l263_26373


namespace NUMINAMATH_CALUDE_buses_passed_count_l263_26360

/-- Represents the frequency of Dallas to Houston buses in minutes -/
def dallas_to_houston_frequency : ℕ := 40

/-- Represents the frequency of Houston to Dallas buses in minutes -/
def houston_to_dallas_frequency : ℕ := 60

/-- Represents the trip duration in hours -/
def trip_duration : ℕ := 6

/-- Represents the minute offset for Houston to Dallas buses -/
def houston_to_dallas_offset : ℕ := 30

/-- Calculates the number of Dallas-bound buses a Houston-bound bus passes on the highway -/
def buses_passed : ℕ := 
  sorry

theorem buses_passed_count : buses_passed = 10 := by
  sorry

end NUMINAMATH_CALUDE_buses_passed_count_l263_26360


namespace NUMINAMATH_CALUDE_lines_are_parallel_l263_26370

/-- Two lines are parallel if they have the same slope but different y-intercepts -/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁ ∧ a₁ * c₂ ≠ a₂ * c₁

/-- The given line: x - 2y + 1 = 0 -/
def line1 : ℝ → ℝ → ℝ := fun x y ↦ x - 2 * y + 1

/-- The line to be proved parallel: 2x - 4y + 1 = 0 -/
def line2 : ℝ → ℝ → ℝ := fun x y ↦ 2 * x - 4 * y + 1

theorem lines_are_parallel : parallel 1 (-2) 1 2 (-4) 1 := by
  sorry

#check lines_are_parallel

end NUMINAMATH_CALUDE_lines_are_parallel_l263_26370


namespace NUMINAMATH_CALUDE_log_base_2_derivative_l263_26394

open Real

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => log x / log 2) x = 1 / (x * log 2) := by
sorry

end NUMINAMATH_CALUDE_log_base_2_derivative_l263_26394


namespace NUMINAMATH_CALUDE_digit_multiplication_puzzle_l263_26357

theorem digit_multiplication_puzzle :
  ∃! (A B C D E F : ℕ),
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧ F ≤ 9 ∧
    A * (10 * B + A) = 10 * C + D ∧
    F * (10 * B + E) = 10 * D + C ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F :=
by sorry

end NUMINAMATH_CALUDE_digit_multiplication_puzzle_l263_26357


namespace NUMINAMATH_CALUDE_unique_prime_in_range_l263_26385

def f (n : ℕ) : ℤ := n^3 - 7*n^2 + 15*n - 12

def is_prime (z : ℤ) : Prop := z > 1 ∧ ∀ m : ℕ, 1 < m → m < |z| → ¬(z % m = 0)

theorem unique_prime_in_range :
  ∃! (n : ℕ), 0 < n ∧ n ≤ 6 ∧ is_prime (f n) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_in_range_l263_26385


namespace NUMINAMATH_CALUDE_taxi_ride_distance_l263_26363

/-- Calculates the distance of a taxi ride given the fare structure and total fare -/
theorem taxi_ride_distance 
  (initial_fare : ℚ) 
  (initial_distance : ℚ) 
  (additional_fare : ℚ) 
  (additional_distance : ℚ) 
  (total_fare : ℚ) 
  (h1 : initial_fare = 2)
  (h2 : initial_distance = 1/5)
  (h3 : additional_fare = 3/5)
  (h4 : additional_distance = 1/5)
  (h5 : total_fare = 127/5) : 
  ∃ (distance : ℚ), distance = 8 ∧ 
    total_fare = initial_fare + (distance - initial_distance) / additional_distance * additional_fare :=
by sorry

end NUMINAMATH_CALUDE_taxi_ride_distance_l263_26363


namespace NUMINAMATH_CALUDE_simplify_expression_l263_26364

theorem simplify_expression : (5 * 10^10) / (2 * 10^4 * 10^2) = 25000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l263_26364


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l263_26327

theorem geometric_sequence_fourth_term :
  ∀ x : ℚ,
  let a₁ := x
  let a₂ := 3*x + 3
  let a₃ := 5*x + 5
  let r := a₂ / a₁
  (a₂ = r * a₁) ∧ (a₃ = r * a₂) →
  let a₄ := r * a₃
  a₄ = -125/12 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l263_26327


namespace NUMINAMATH_CALUDE_alyssa_car_wash_earnings_l263_26387

/-- The amount Alyssa earned from washing the family car -/
def car_wash_earnings (weekly_allowance : ℝ) (movie_spending_fraction : ℝ) (final_amount : ℝ) : ℝ :=
  final_amount - (weekly_allowance * (1 - movie_spending_fraction))

/-- Theorem: Alyssa earned 8 dollars from washing the family car -/
theorem alyssa_car_wash_earnings :
  car_wash_earnings 8 0.5 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_car_wash_earnings_l263_26387


namespace NUMINAMATH_CALUDE_subtraction_problem_l263_26303

theorem subtraction_problem (v : Nat) : v < 10 → 400 + 10 * v + 7 - 189 = 268 → v = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l263_26303


namespace NUMINAMATH_CALUDE_range_of_a_l263_26300

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l263_26300


namespace NUMINAMATH_CALUDE_isabellas_house_number_l263_26319

/-- A predicate that checks if a natural number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- A predicate that checks if a natural number has 9 as one of its digits -/
def has_digit_nine (n : ℕ) : Prop := ∃ d, d ∈ n.digits 10 ∧ d = 9

theorem isabellas_house_number :
  ∃! n : ℕ, is_two_digit n ∧
           ¬ Nat.Prime n ∧
           Even n ∧
           n % 7 = 0 ∧
           has_digit_nine n ∧
           n % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_isabellas_house_number_l263_26319


namespace NUMINAMATH_CALUDE_x_equation_value_l263_26304

theorem x_equation_value (x : ℝ) (h : x + 1/x = 3) :
  x^10 - 6*x^6 + x^2 = -328*x^2 := by
sorry

end NUMINAMATH_CALUDE_x_equation_value_l263_26304


namespace NUMINAMATH_CALUDE_coefficient_of_fifth_power_l263_26362

theorem coefficient_of_fifth_power (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (2*x - 1)^5 * (x + 2) = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 →
  a₅ = 176 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_fifth_power_l263_26362


namespace NUMINAMATH_CALUDE_pharmacy_tubs_in_storage_l263_26341

theorem pharmacy_tubs_in_storage (total_needed : ℕ) (bought_usual : ℕ) : ℕ :=
  let tubs_in_storage := total_needed - (bought_usual + bought_usual / 3)
  by
    sorry

#check pharmacy_tubs_in_storage 100 60

end NUMINAMATH_CALUDE_pharmacy_tubs_in_storage_l263_26341


namespace NUMINAMATH_CALUDE_company_earnings_difference_l263_26336

/-- Represents a company selling bottled milk -/
structure Company where
  price : ℝ  -- Price of a big bottle
  sold : ℕ   -- Number of big bottles sold

/-- Calculates the earnings of a company -/
def earnings (c : Company) : ℝ := c.price * c.sold

/-- The problem statement -/
theorem company_earnings_difference 
  (company_a company_b : Company)
  (ha : company_a.price = 4)
  (hb : company_b.price = 3.5)
  (sa : company_a.sold = 300)
  (sb : company_b.sold = 350) :
  earnings company_b - earnings company_a = 25 := by
  sorry

end NUMINAMATH_CALUDE_company_earnings_difference_l263_26336


namespace NUMINAMATH_CALUDE_no_solution_in_interval_l263_26340

theorem no_solution_in_interval : 
  ¬ ∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ (3 * x - 2) ≥ 3 * (12 - 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_in_interval_l263_26340


namespace NUMINAMATH_CALUDE_one_third_minus_0_3333_l263_26352

-- Define 0.3333 as a rational number
def decimal_0_3333 : ℚ := 3333 / 10000

-- State the theorem
theorem one_third_minus_0_3333 : (1 : ℚ) / 3 - decimal_0_3333 = 1 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_one_third_minus_0_3333_l263_26352


namespace NUMINAMATH_CALUDE_social_gathering_attendance_l263_26348

theorem social_gathering_attendance
  (num_men : ℕ)
  (women_per_man : ℕ)
  (men_per_woman : ℕ)
  (h_num_men : num_men = 15)
  (h_women_per_man : women_per_man = 4)
  (h_men_per_woman : men_per_woman = 3) :
  (num_men * women_per_man) / men_per_woman = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_social_gathering_attendance_l263_26348


namespace NUMINAMATH_CALUDE_supermarket_max_profit_l263_26397

/-- Represents the daily profit function for the supermarket -/
def daily_profit (x : ℝ) : ℝ := -10 * x^2 + 1100 * x - 28000

/-- The maximum daily profit achievable by the supermarket -/
def max_profit : ℝ := 2250

theorem supermarket_max_profit :
  ∃ (x : ℝ), daily_profit x = max_profit ∧
  ∀ (y : ℝ), daily_profit y ≤ max_profit := by
  sorry

#check supermarket_max_profit

end NUMINAMATH_CALUDE_supermarket_max_profit_l263_26397


namespace NUMINAMATH_CALUDE_simple_interest_rate_l263_26332

/-- 
Given a principal amount P and a time period of 10 years, 
prove that the rate percent per annum R is 12% when the simple interest 
is 6/5 of the principal amount.
-/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  let R := (6 / 5) * 100 / 10
  let simple_interest := (P * R * 10) / 100
  simple_interest = (6 / 5) * P → R = 12 := by
  sorry

#check simple_interest_rate

end NUMINAMATH_CALUDE_simple_interest_rate_l263_26332


namespace NUMINAMATH_CALUDE_additive_inverse_equation_l263_26351

theorem additive_inverse_equation (x : ℝ) : (6 * x - 12 = -(4 + 2 * x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverse_equation_l263_26351


namespace NUMINAMATH_CALUDE_number_with_percentage_increase_l263_26334

theorem number_with_percentage_increase : ∃ x : ℝ, x + 0.35 * x = x + 150 := by
  sorry

end NUMINAMATH_CALUDE_number_with_percentage_increase_l263_26334


namespace NUMINAMATH_CALUDE_common_difference_is_two_l263_26354

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_is_two (seq : ArithmeticSequence) 
  (h : seq.S 4 / 4 - seq.S 2 / 2 = 2) : 
  commonDifference seq = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l263_26354


namespace NUMINAMATH_CALUDE_abc_product_l263_26306

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 45 * Real.rpow 3 (1/3))
  (hac : a * c = 75 * Real.rpow 3 (1/3))
  (hbc : b * c = 30 * Real.rpow 3 (1/3)) :
  a * b * c = 75 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l263_26306


namespace NUMINAMATH_CALUDE_line_intersects_parabola_at_one_point_l263_26347

/-- The value of k for which the line x = k intersects the parabola x = -3y² - 4y + 7 at exactly one point -/
def k : ℚ := 25/3

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -3*y^2 - 4*y + 7

theorem line_intersects_parabola_at_one_point :
  ∃! y : ℝ, parabola y = k := by sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_at_one_point_l263_26347


namespace NUMINAMATH_CALUDE_edward_book_purchase_l263_26330

/-- Given that Edward spent $6 on books and each book cost $3, prove that he bought 2 books. -/
theorem edward_book_purchase (total_spent : ℕ) (cost_per_book : ℕ) (h1 : total_spent = 6) (h2 : cost_per_book = 3) :
  total_spent / cost_per_book = 2 := by
  sorry

end NUMINAMATH_CALUDE_edward_book_purchase_l263_26330


namespace NUMINAMATH_CALUDE_range_of_a_l263_26355

-- Define the set M
def M (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 4*x + 4*a < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : (2 ∉ M a) ↔ a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l263_26355


namespace NUMINAMATH_CALUDE_inscribed_circle_area_l263_26329

/-- A circle inscribed in a right triangle with specific properties -/
structure InscribedCircle (A B C X Y : ℝ × ℝ) :=
  (right_angle : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0)
  (ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6)
  (tangent_x : (X.1 - A.1) * (B.1 - A.1) + (X.2 - A.2) * (B.2 - A.2) = 0)
  (tangent_y : (Y.1 - A.1) * (C.1 - A.1) + (Y.2 - A.2) * (C.2 - A.2) = 0)
  (opposite_on_bc : ∃ (X' Y' : ℝ × ℝ), 
    (X'.1 - B.1) * (C.1 - B.1) + (X'.2 - B.2) * (C.2 - B.2) = 0 ∧
    (Y'.1 - B.1) * (C.1 - B.1) + (Y'.2 - B.2) * (C.2 - B.2) = 0 ∧
    (X'.1 - X.1)^2 + (X'.2 - X.2)^2 = (Y'.1 - Y.1)^2 + (Y'.2 - Y.2)^2)

/-- The area of the portion of the circle outside the triangle is π - 2 -/
theorem inscribed_circle_area (A B C X Y : ℝ × ℝ) 
  (h : InscribedCircle A B C X Y) : 
  ∃ (r : ℝ), r > 0 ∧ π * r^2 / 4 - r^2 / 2 = π - 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_l263_26329


namespace NUMINAMATH_CALUDE_andy_initial_candies_l263_26321

/-- The number of candies each person has initially and after distribution --/
structure CandyDistribution where
  billy_initial : ℕ
  caleb_initial : ℕ
  andy_initial : ℕ
  father_bought : ℕ
  billy_received : ℕ
  caleb_received : ℕ
  andy_final_diff : ℕ

/-- Theorem stating that Andy initially took 9 candies --/
theorem andy_initial_candies (d : CandyDistribution) 
  (h1 : d.billy_initial = 6)
  (h2 : d.caleb_initial = 11)
  (h3 : d.father_bought = 36)
  (h4 : d.billy_received = 8)
  (h5 : d.caleb_received = 11)
  (h6 : d.andy_final_diff = 4)
  : d.andy_initial = 9 := by
  sorry


end NUMINAMATH_CALUDE_andy_initial_candies_l263_26321


namespace NUMINAMATH_CALUDE_shaded_fraction_is_one_eighth_l263_26308

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

theorem shaded_fraction_is_one_eighth (r : Rectangle) 
  (h1 : r.width = 15)
  (h2 : r.height = 20)
  (h3 : ∃ (shaded_area : ℝ), shaded_area = (1/4) * ((1/2) * r.area)) :
  ∃ (shaded_area : ℝ), shaded_area / r.area = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_is_one_eighth_l263_26308


namespace NUMINAMATH_CALUDE_number_times_five_equals_hundred_l263_26368

theorem number_times_five_equals_hundred :
  ∃ x : ℝ, 5 * x = 100 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_times_five_equals_hundred_l263_26368


namespace NUMINAMATH_CALUDE_linear_function_increasing_l263_26337

/-- Given a linear function f(x) = 2x - 1, prove that for any two points
    (x₁, y₁) and (x₂, y₂) on its graph, if x₁ > x₂, then y₁ > y₂ -/
theorem linear_function_increasing (x₁ x₂ y₁ y₂ : ℝ) 
    (h1 : y₁ = 2 * x₁ - 1)
    (h2 : y₂ = 2 * x₂ - 1)
    (h3 : x₁ > x₂) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_linear_function_increasing_l263_26337


namespace NUMINAMATH_CALUDE_complex_equation_solution_l263_26350

variable (z : ℂ)

theorem complex_equation_solution :
  (1 - Complex.I) * z = 2 * Complex.I →
  z = -1 + Complex.I ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l263_26350


namespace NUMINAMATH_CALUDE_max_term_binomial_expansion_l263_26312

theorem max_term_binomial_expansion :
  let n : ℕ := 213
  let x : ℝ := Real.sqrt 5
  let term (k : ℕ) := (n.choose k) * x^k
  ∃ k_max : ℕ, k_max = 147 ∧ ∀ k : ℕ, k ≤ n → term k ≤ term k_max :=
by sorry

end NUMINAMATH_CALUDE_max_term_binomial_expansion_l263_26312


namespace NUMINAMATH_CALUDE_monotonicity_when_a_eq_1_extreme_value_two_zero_points_l263_26318

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2 + (2*a - 1) * x

-- State the theorems
theorem monotonicity_when_a_eq_1 :
  ∀ x y, 0 < x ∧ x < 1 ∧ 0 < y ∧ 1 < y → f 1 x < f 1 1 ∧ f 1 1 > f 1 y := by sorry

theorem extreme_value :
  ∀ a, a > 0 → ∃ x, x > 0 ∧ ∀ y, y > 0 → f a y ≤ f a x ∧ f a x = a * (Real.log a + a - 1) := by sorry

theorem two_zero_points :
  ∀ a, (∃ x y, 0 < x ∧ x < y ∧ f a x = 0 ∧ f a y = 0) ↔ a > 1 := by sorry

end

end NUMINAMATH_CALUDE_monotonicity_when_a_eq_1_extreme_value_two_zero_points_l263_26318


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l263_26378

def set_A : Set ℝ := {x | -1 ≤ 2*x+1 ∧ 2*x+1 ≤ 3}
def set_B : Set ℝ := {x | x ≠ 0 ∧ (x-2)/x ≤ 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l263_26378


namespace NUMINAMATH_CALUDE_total_bricks_used_l263_26326

/-- The number of walls being built -/
def number_of_walls : ℕ := 4

/-- The number of bricks in a single row of a wall -/
def bricks_per_row : ℕ := 60

/-- The number of rows in each wall -/
def rows_per_wall : ℕ := 100

/-- Theorem stating the total number of bricks used for all walls -/
theorem total_bricks_used :
  number_of_walls * bricks_per_row * rows_per_wall = 24000 := by
  sorry

end NUMINAMATH_CALUDE_total_bricks_used_l263_26326


namespace NUMINAMATH_CALUDE_four_bottles_left_l263_26315

/-- The number of bottles left after a given number of days for a person who drinks half a bottle per day -/
def bottles_left (initial_bottles : ℕ) (days : ℕ) : ℕ :=
  initial_bottles - (days / 2)

/-- Theorem stating that 4 bottles will be left after 28 days, starting with 18 bottles -/
theorem four_bottles_left : bottles_left 18 28 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_bottles_left_l263_26315


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l263_26365

/-- Given positive real numbers a and b, prove two inequalities based on given conditions -/
theorem positive_real_inequalities (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b = 1 → (1 + 1/a) * (1 + 1/b) ≥ 9) ∧
  (2*a + b = a*b → a + b ≥ 3 + 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l263_26365


namespace NUMINAMATH_CALUDE_meaningful_expression_l263_26314

/-- The expression x + 1/(x-2) is meaningful for all real x except 2 -/
theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = x + 1 / (x - 2)) ↔ x ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l263_26314


namespace NUMINAMATH_CALUDE_train_meeting_distance_l263_26301

/-- Proves that Train A travels 75 miles before meeting Train B -/
theorem train_meeting_distance (route_length : ℝ) (time_a : ℝ) (time_b : ℝ) 
  (h1 : route_length = 200)
  (h2 : time_a = 10)
  (h3 : time_b = 6)
  : (route_length / time_a) * (route_length / (route_length / time_a + route_length / time_b)) = 75 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_distance_l263_26301


namespace NUMINAMATH_CALUDE_erased_number_l263_26396

theorem erased_number (n : ℕ) (x : ℕ) : 
  x ≤ n →
  (n * (n + 1) / 2 - x) / (n - 1) = 45 / 4 →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_erased_number_l263_26396


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l263_26302

theorem chocolate_box_problem (total : ℕ) (p_peanut : ℚ) : 
  total = 50 → p_peanut = 64/100 → 
  ∃ (caramels nougats truffles peanuts : ℕ),
    nougats = 2 * caramels ∧
    truffles = caramels + 6 ∧
    caramels + nougats + truffles + peanuts = total ∧
    p_peanut = peanuts / total ∧
    caramels = 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l263_26302


namespace NUMINAMATH_CALUDE_simplify_expression_l263_26359

theorem simplify_expression :
  (6 * 10^7) * (2 * 10^3)^2 / (4 * 10^4) = 6 * 10^9 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l263_26359


namespace NUMINAMATH_CALUDE_constant_difference_of_equal_second_derivatives_l263_26383

theorem constant_difference_of_equal_second_derivatives 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g) 
  (h : ∀ x, (deriv^[2] f) x = (deriv^[2] g) x) : 
  ∃ c : ℝ, ∀ x, f x - g x = c :=
sorry

end NUMINAMATH_CALUDE_constant_difference_of_equal_second_derivatives_l263_26383


namespace NUMINAMATH_CALUDE_hit_first_third_fifth_probability_hit_exactly_three_probability_l263_26339

-- Define the probability of hitting the target
def hit_probability : ℚ := 3/5

-- Define the number of shots
def num_shots : ℕ := 5

-- Theorem for the first part of the problem
theorem hit_first_third_fifth_probability :
  (hit_probability * (1 - hit_probability) * hit_probability * (1 - hit_probability) * hit_probability : ℚ) = 108/3125 :=
sorry

-- Theorem for the second part of the problem
theorem hit_exactly_three_probability :
  (Nat.choose num_shots 3 : ℚ) * hit_probability^3 * (1 - hit_probability)^2 = 216/625 :=
sorry

end NUMINAMATH_CALUDE_hit_first_third_fifth_probability_hit_exactly_three_probability_l263_26339


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l263_26374

theorem book_sale_loss_percentage 
  (selling_price_loss : ℝ) 
  (selling_price_gain : ℝ) 
  (gain_percentage : ℝ) :
  selling_price_loss = 800 →
  selling_price_gain = 1100 →
  gain_percentage = 10 →
  (1 - selling_price_loss / (selling_price_gain / (1 + gain_percentage / 100))) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l263_26374


namespace NUMINAMATH_CALUDE_no_consecutive_integers_sum_75_l263_26393

theorem no_consecutive_integers_sum_75 : 
  ¬∃ (a n : ℕ), n ≥ 2 ∧ (n * (2 * a + n - 1) / 2 = 75) := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_integers_sum_75_l263_26393


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l263_26335

def U : Set ℤ := {-1, 0, 1, 2}

def A : Set ℤ := {x ∈ U | x^2 < 1}

theorem complement_of_A_in_U : Set.compl A = {-1, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l263_26335


namespace NUMINAMATH_CALUDE_slope_product_theorem_l263_26342

theorem slope_product_theorem (m n : ℝ) : 
  m ≠ 0 → n ≠ 0 →  -- non-horizontal lines
  (∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ m = Real.tan θ₁ ∧ n = Real.tan θ₂) →  -- angle relationship
  m = 6 * n →  -- slope relationship
  m * n = 9 / 17 := by
sorry

end NUMINAMATH_CALUDE_slope_product_theorem_l263_26342


namespace NUMINAMATH_CALUDE_milk_water_ratio_l263_26372

theorem milk_water_ratio 
  (initial_volume : ℝ) 
  (initial_milk_ratio : ℝ) 
  (initial_water_ratio : ℝ) 
  (added_water : ℝ) : 
  initial_volume = 45 ∧ 
  initial_milk_ratio = 4 ∧ 
  initial_water_ratio = 1 ∧ 
  added_water = 21 → 
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let new_water := initial_water + added_water
  let new_ratio_milk := initial_milk / (initial_milk + new_water) * 11
  let new_ratio_water := new_water / (initial_milk + new_water) * 11
  new_ratio_milk = 6 ∧ new_ratio_water = 5 :=
by sorry


end NUMINAMATH_CALUDE_milk_water_ratio_l263_26372


namespace NUMINAMATH_CALUDE_disease_cases_estimation_l263_26392

/-- A function representing the number of disease cases over time -/
def cases (t : ℝ) : ℝ := 800000 - 19995 * (t - 1970)

theorem disease_cases_estimation :
  cases 1995 = 300125 ∧ cases 2005 = 100175 :=
by
  sorry

end NUMINAMATH_CALUDE_disease_cases_estimation_l263_26392


namespace NUMINAMATH_CALUDE_washington_goat_count_l263_26307

/-- The number of goats Washington has -/
def washington_goats : ℕ := 140

/-- The number of goats Paddington has -/
def paddington_goats : ℕ := washington_goats + 40

/-- The total number of goats -/
def total_goats : ℕ := 320

theorem washington_goat_count : washington_goats = 140 :=
  by sorry

end NUMINAMATH_CALUDE_washington_goat_count_l263_26307


namespace NUMINAMATH_CALUDE_remainder_three_to_forty_plus_five_mod_five_l263_26361

theorem remainder_three_to_forty_plus_five_mod_five :
  (3^40 + 5) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_to_forty_plus_five_mod_five_l263_26361


namespace NUMINAMATH_CALUDE_luncheon_attendance_l263_26356

/-- A luncheon problem -/
theorem luncheon_attendance (total_invited : ℕ) (tables_needed : ℕ) (capacity_per_table : ℕ)
  (h1 : total_invited = 45)
  (h2 : tables_needed = 5)
  (h3 : capacity_per_table = 2) :
  total_invited - (tables_needed * capacity_per_table) = 35 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_attendance_l263_26356


namespace NUMINAMATH_CALUDE_cricket_player_average_increase_l263_26317

/-- 
Theorem: Cricket Player's Average Increase

Given:
- A cricket player has played 10 innings
- The current average is 32 runs per innings
- The player needs to make 76 runs in the next innings

Prove: The increase in average is 4 runs per innings
-/
theorem cricket_player_average_increase 
  (innings : ℕ) 
  (current_average : ℚ) 
  (next_innings_runs : ℕ) 
  (h1 : innings = 10)
  (h2 : current_average = 32)
  (h3 : next_innings_runs = 76) : 
  (((innings : ℚ) * current_average + next_innings_runs) / (innings + 1) - current_average) = 4 := by
  sorry


end NUMINAMATH_CALUDE_cricket_player_average_increase_l263_26317


namespace NUMINAMATH_CALUDE_special_triangle_exists_l263_26380

-- Define the color type
inductive Color
| Red
| Green
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
def colorFunction : Point → Color := sorry

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the circumradius of a triangle
def circumradius (t : Triangle) : ℝ := sorry

-- Define a predicate for monochromatic triangle
def isMonochromatic (t : Triangle) : Prop :=
  colorFunction t.A = colorFunction t.B ∧ colorFunction t.B = colorFunction t.C

-- Define a predicate for angle ratio condition
def satisfiesAngleRatio (t : Triangle) : Prop := sorry

-- The main theorem
theorem special_triangle_exists :
  ∃ (t : Triangle), isMonochromatic t ∧ circumradius t = 2008 ∧ satisfiesAngleRatio t := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_exists_l263_26380


namespace NUMINAMATH_CALUDE_constant_term_value_l263_26322

theorem constant_term_value (y : ℝ) (d : ℝ) :
  y = 2 → (5 * y^2 - 8 * y + 55 = d ↔ d = 59) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_value_l263_26322


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l263_26376

def f (x : ℝ) := x^2

theorem f_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ a b, 0 ≤ a → a < b → f a ≤ f b) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l263_26376


namespace NUMINAMATH_CALUDE_group_b_more_stable_l263_26305

-- Define the structure for a group's statistics
structure GroupStats where
  mean : ℝ
  variance : ℝ

-- Define the stability comparison function
def more_stable (a b : GroupStats) : Prop :=
  a.variance < b.variance

-- Theorem statement
theorem group_b_more_stable (group_a group_b : GroupStats) 
  (h1 : group_a.mean = group_b.mean)
  (h2 : group_a.variance = 36)
  (h3 : group_b.variance = 30) :
  more_stable group_b group_a :=
by
  sorry

end NUMINAMATH_CALUDE_group_b_more_stable_l263_26305


namespace NUMINAMATH_CALUDE_paper_boats_problem_l263_26323

theorem paper_boats_problem (initial_boats : ℕ) : 
  (initial_boats : ℝ) * 0.8 - 2 = 22 → initial_boats = 30 := by
  sorry

end NUMINAMATH_CALUDE_paper_boats_problem_l263_26323


namespace NUMINAMATH_CALUDE_concave_probability_is_one_third_l263_26389

/-- A digit is a natural number between 4 and 8 inclusive -/
def Digit : Type := { n : ℕ // 4 ≤ n ∧ n ≤ 8 }

/-- A three-digit number is a tuple of three digits -/
def ThreeDigitNumber : Type := Digit × Digit × Digit

/-- A concave number is a three-digit number where the first and third digits are greater than the second -/
def is_concave (n : ThreeDigitNumber) : Prop :=
  let (a, b, c) := n
  a.val > b.val ∧ c.val > b.val

/-- The set of all possible three-digit numbers with distinct digits from {4,5,6,7,8} -/
def all_numbers : Finset ThreeDigitNumber :=
  sorry

/-- The set of all concave numbers from all_numbers -/
def concave_numbers : Finset ThreeDigitNumber :=
  sorry

/-- The probability of a randomly chosen three-digit number being concave -/
def concave_probability : ℚ :=
  (Finset.card concave_numbers : ℚ) / (Finset.card all_numbers : ℚ)

theorem concave_probability_is_one_third :
  concave_probability = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_concave_probability_is_one_third_l263_26389


namespace NUMINAMATH_CALUDE_connor_date_cost_l263_26375

/-- The cost of a movie date for Connor and his date -/
def movie_date_cost (ticket_price : ℚ) (combo_meal_price : ℚ) (candy_price : ℚ) : ℚ :=
  2 * ticket_price + combo_meal_price + 2 * candy_price

/-- Theorem stating the total cost of Connor's movie date -/
theorem connor_date_cost :
  movie_date_cost 10 11 2.5 = 36 :=
by sorry

end NUMINAMATH_CALUDE_connor_date_cost_l263_26375


namespace NUMINAMATH_CALUDE_number_reciprocal_problem_l263_26381

theorem number_reciprocal_problem (x : ℝ) (h : 8 * x - 6 = 10) :
  50 * (1 / x) + 150 = 175 := by
  sorry

end NUMINAMATH_CALUDE_number_reciprocal_problem_l263_26381


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l263_26382

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ x * (x - 3) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l263_26382


namespace NUMINAMATH_CALUDE_blue_highlighters_count_l263_26331

/-- Given the number of highlighters in a teacher's desk, calculate the number of blue highlighters. -/
theorem blue_highlighters_count 
  (total : ℕ) 
  (pink : ℕ) 
  (yellow : ℕ) 
  (h1 : total = 11) 
  (h2 : pink = 4) 
  (h3 : yellow = 2) : 
  total - pink - yellow = 5 := by
  sorry

#check blue_highlighters_count

end NUMINAMATH_CALUDE_blue_highlighters_count_l263_26331


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l263_26366

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h : 50 * cost_price = 32 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l263_26366


namespace NUMINAMATH_CALUDE_pool_capacity_theorem_l263_26358

/-- Represents the dimensions of a pool -/
structure PoolDimensions where
  width : ℝ
  length : ℝ
  depth : ℝ

/-- Calculates the volume of a pool given its dimensions -/
def poolVolume (d : PoolDimensions) : ℝ := d.width * d.length * d.depth

/-- Represents the draining parameters of a pool -/
structure DrainParameters where
  rate : ℝ
  time : ℝ

/-- Calculates the amount of water drained given drain parameters -/
def waterDrained (p : DrainParameters) : ℝ := p.rate * p.time

/-- Theorem: The initial capacity of the pool was 80% of its total volume -/
theorem pool_capacity_theorem (d : PoolDimensions) (p : DrainParameters) :
  d.width = 60 ∧ d.length = 150 ∧ d.depth = 10 ∧
  p.rate = 60 ∧ p.time = 1200 →
  waterDrained p / poolVolume d = 0.8 := by
  sorry

#eval (80 : ℚ) / 100  -- Expected output: 4/5

end NUMINAMATH_CALUDE_pool_capacity_theorem_l263_26358


namespace NUMINAMATH_CALUDE_negation_of_positive_square_is_false_l263_26353

theorem negation_of_positive_square_is_false :
  ¬(∀ x : ℝ, x > 0 → x^2 > 0) = False :=
by sorry

end NUMINAMATH_CALUDE_negation_of_positive_square_is_false_l263_26353


namespace NUMINAMATH_CALUDE_count_decreasing_digit_numbers_l263_26345

/-- A function that checks if a natural number has strictly decreasing digits. -/
def hasDecreasingDigits (n : ℕ) : Bool :=
  sorry

/-- The count of natural numbers with at least two digits and strictly decreasing digits. -/
def countDecreasingDigitNumbers : ℕ :=
  sorry

/-- Theorem stating that the count of natural numbers with at least two digits 
    and strictly decreasing digits is 1013. -/
theorem count_decreasing_digit_numbers :
  countDecreasingDigitNumbers = 1013 := by
  sorry

end NUMINAMATH_CALUDE_count_decreasing_digit_numbers_l263_26345


namespace NUMINAMATH_CALUDE_currency_exchange_problem_l263_26313

def exchange_rate : ℚ := 8 / 6

theorem currency_exchange_problem (d : ℕ) :
  (d : ℚ) * exchange_rate - 96 = d →
  d = 288 := by sorry

end NUMINAMATH_CALUDE_currency_exchange_problem_l263_26313


namespace NUMINAMATH_CALUDE_factorize_quadratic_factorize_cubic_factorize_quartic_l263_26311

-- Problem 1
theorem factorize_quadratic (m : ℝ) : m^2 + 4*m + 4 = (m + 2)^2 := by sorry

-- Problem 2
theorem factorize_cubic (a b : ℝ) : a^2*b - 4*a*b^2 + 3*b^3 = b*(a-b)*(a-3*b) := by sorry

-- Problem 3
theorem factorize_quartic (x y : ℝ) : (x^2 + y^2)^2 - 4*x^2*y^2 = (x + y)^2 * (x - y)^2 := by sorry

end NUMINAMATH_CALUDE_factorize_quadratic_factorize_cubic_factorize_quartic_l263_26311


namespace NUMINAMATH_CALUDE_first_equation_is_double_root_second_equation_values_l263_26338

/-- Definition of a double root equation -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 2 * x

/-- The equation x^2 - 3x + 2 = 0 is a double root equation -/
theorem first_equation_is_double_root : is_double_root_equation 1 (-3) 2 :=
sorry

/-- For ax^2 + bx - 6 = 0, if it's a double root equation with one root as 2,
    then a and b have specific values -/
theorem second_equation_values (a b : ℝ) :
  is_double_root_equation a b (-6) ∧ (∃ x : ℝ, a * x^2 + b * x - 6 = 0 ∧ x = 2) →
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) :=
sorry

end NUMINAMATH_CALUDE_first_equation_is_double_root_second_equation_values_l263_26338


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_6_l263_26316

theorem largest_integer_less_than_100_with_remainder_5_mod_6 :
  ∀ n : ℕ, n < 100 ∧ n % 6 = 5 → n ≤ 99 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_6_l263_26316


namespace NUMINAMATH_CALUDE_fraction_division_result_l263_26325

theorem fraction_division_result : (3 / 8) / (5 / 9) = 27 / 40 := by sorry

end NUMINAMATH_CALUDE_fraction_division_result_l263_26325


namespace NUMINAMATH_CALUDE_largest_base_not_18_l263_26309

/-- Represents a number in a given base as a list of digits -/
def Digits := List Nat

/-- Calculates the sum of digits -/
def sum_of_digits (digits : Digits) : Nat :=
  digits.sum

/-- Converts a number to its representation in a given base -/
def to_base (n : Nat) (base : Nat) : Digits :=
  sorry

theorem largest_base_not_18 :
  ∃ (max_base : Nat),
    (sum_of_digits (to_base (12^3) 10) = 18) ∧
    (12^3 = 1728) ∧
    (∀ b > 10, to_base (12^3) b = to_base 1728 b) ∧
    (to_base (12^3) 9 = [1, 4, 6, 7]) ∧
    (to_base (12^3) 8 = [1, 3, 7, 6]) ∧
    (∀ b > max_base, sum_of_digits (to_base (12^3) b) = 18) ∧
    (sum_of_digits (to_base (12^3) max_base) ≠ 18) ∧
    max_base = 8 :=
  sorry

end NUMINAMATH_CALUDE_largest_base_not_18_l263_26309


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l263_26344

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- Theorem statement
theorem line_perp_parallel_planes 
  (a : Line) (α β : Plane) 
  (h1 : perpendicular a α) 
  (h2 : parallel α β) : 
  perpendicular a β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l263_26344


namespace NUMINAMATH_CALUDE_problem_solving_probability_l263_26395

theorem problem_solving_probability 
  (xavier_prob : ℚ) 
  (yvonne_prob : ℚ) 
  (zelda_prob : ℚ) 
  (hx : xavier_prob = 1/6)
  (hy : yvonne_prob = 1/2)
  (hz : zelda_prob = 5/8) :
  xavier_prob * yvonne_prob * (1 - zelda_prob) = 1/32 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l263_26395


namespace NUMINAMATH_CALUDE_total_teachers_l263_26379

theorem total_teachers (senior : ℕ) (intermediate : ℕ) (sampled_total : ℕ) (sampled_other : ℕ)
  (h1 : senior = 26)
  (h2 : intermediate = 104)
  (h3 : sampled_total = 56)
  (h4 : sampled_other = 16)
  (h5 : ∀ (category : ℕ) (sampled_category : ℕ) (total : ℕ),
    (category : ℚ) / total = (sampled_category : ℚ) / sampled_total) :
  ∃ (total : ℕ), total = 52 := by
sorry

end NUMINAMATH_CALUDE_total_teachers_l263_26379


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l263_26310

theorem framed_painting_ratio : 
  let painting_width : ℝ := 18
  let painting_height : ℝ := 24
  let frame_side_width : ℝ := 3  -- This is derived from solving the equation in the solution
  let frame_top_bottom_width : ℝ := 2 * frame_side_width
  let framed_width : ℝ := painting_width + 2 * frame_side_width
  let framed_height : ℝ := painting_height + 2 * frame_top_bottom_width
  let frame_area : ℝ := framed_width * framed_height - painting_width * painting_height
  frame_area = painting_width * painting_height →
  (min framed_width framed_height) / (max framed_width framed_height) = 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l263_26310


namespace NUMINAMATH_CALUDE_purple_ribbons_l263_26390

theorem purple_ribbons (total : ℕ) (yellow purple orange black : ℕ) : 
  yellow = total / 4 →
  purple = total / 3 →
  orange = total / 6 →
  black = 40 →
  yellow + purple + orange + black = total →
  purple = 53 := by
sorry

end NUMINAMATH_CALUDE_purple_ribbons_l263_26390


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_l263_26369

theorem least_sum_of_exponents (h : ℕ+) (a b c : ℕ+) 
  (h_div_225 : 225 ∣ h) 
  (h_div_216 : 216 ∣ h) 
  (h_eq : h = 2^(a:ℕ) * 3^(b:ℕ) * 5^(c:ℕ)) : 
  (∀ a' b' c' : ℕ+, 
    (225 ∣ (2^(a':ℕ) * 3^(b':ℕ) * 5^(c':ℕ))) → 
    (216 ∣ (2^(a':ℕ) * 3^(b':ℕ) * 5^(c':ℕ))) → 
    (a:ℕ) + (b:ℕ) + (c:ℕ) ≤ (a':ℕ) + (b':ℕ) + (c':ℕ)) ∧ 
  (a:ℕ) + (b:ℕ) + (c:ℕ) = 10 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_l263_26369


namespace NUMINAMATH_CALUDE_odot_inequality_iff_l263_26346

-- Define the ⊙ operation
def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

-- State the theorem
theorem odot_inequality_iff (x : ℝ) : odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_odot_inequality_iff_l263_26346


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l263_26377

theorem binomial_coefficient_equality (a : ℝ) (ha : a ≠ 0) :
  (Nat.choose 5 4 : ℝ) * a^4 = (Nat.choose 5 3 : ℝ) * a^3 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l263_26377


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_implies_perpendicular_to_contained_line_l263_26371

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_to_plane : Line → Plane → Prop)

-- Define the contained relation between a line and a plane
variable (contained_in_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_to_line : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_implies_perpendicular_to_contained_line 
  (l m : Line) (α : Plane) :
  perpendicular_to_plane l α → contained_in_plane m α → perpendicular_to_line l m :=
by sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_implies_perpendicular_to_contained_line_l263_26371


namespace NUMINAMATH_CALUDE_volume_ratio_l263_26388

theorem volume_ratio (A B C : ℝ) 
  (h1 : A = (B + C) / 4)
  (h2 : B = (C + A) / 6) :
  C / (A + B) = 23 / 12 := by
sorry

end NUMINAMATH_CALUDE_volume_ratio_l263_26388


namespace NUMINAMATH_CALUDE_complex_multiplication_l263_26367

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (i + 1) = -1 + i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l263_26367


namespace NUMINAMATH_CALUDE_congruence_problem_l263_26333

theorem congruence_problem (x y n : ℤ) : 
  x ≡ 45 [ZMOD 60] →
  y ≡ 98 [ZMOD 60] →
  n ∈ Finset.Icc 150 210 →
  (x - y ≡ n [ZMOD 60]) ↔ n = 187 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l263_26333
