import Mathlib

namespace peach_pies_count_l1024_102432

/-- Given a total of 30 pies distributed among apple, blueberry, and peach flavors
    in the ratio 3:2:5, prove that the number of peach pies is 15. -/
theorem peach_pies_count (total_pies : ℕ) (apple_ratio blueberry_ratio peach_ratio : ℕ) :
  total_pies = 30 →
  apple_ratio = 3 →
  blueberry_ratio = 2 →
  peach_ratio = 5 →
  peach_ratio * (total_pies / (apple_ratio + blueberry_ratio + peach_ratio)) = 15 := by
  sorry

end peach_pies_count_l1024_102432


namespace max_xy_value_l1024_102479

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 200) : x * y ≤ 28 := by
  sorry

end max_xy_value_l1024_102479


namespace subtracted_amount_l1024_102484

theorem subtracted_amount (chosen_number : ℕ) (subtracted_amount : ℕ) : 
  chosen_number = 208 → 
  (chosen_number / 2 : ℚ) - subtracted_amount = 4 → 
  subtracted_amount = 100 := by
sorry

end subtracted_amount_l1024_102484


namespace product_plus_one_is_square_l1024_102405

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) :
  ∃ n : ℕ, x * y + 1 = n^2 := by
  sorry

end product_plus_one_is_square_l1024_102405


namespace cube_equation_solution_l1024_102440

theorem cube_equation_solution (a d : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * d) : d = 49 := by
  sorry

end cube_equation_solution_l1024_102440


namespace quadratic_discriminant_l1024_102433

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 + 8x - 6 -/
def a : ℝ := 5
def b : ℝ := 8
def c : ℝ := -6

/-- Theorem: The discriminant of 5x^2 + 8x - 6 is 184 -/
theorem quadratic_discriminant : discriminant a b c = 184 := by
  sorry

end quadratic_discriminant_l1024_102433


namespace smallest_number_l1024_102404

theorem smallest_number : ∀ (a b c d : ℝ), a = 0 ∧ b = -1 ∧ c = -Real.sqrt 2 ∧ d = 2 → 
  c ≤ a ∧ c ≤ b ∧ c ≤ d := by
  sorry

end smallest_number_l1024_102404


namespace trumpington_band_size_l1024_102406

theorem trumpington_band_size (n : ℕ) : 
  (∃ k : ℕ, 20 * n = 26 * k + 4) → 
  20 * n < 1000 → 
  (∀ m : ℕ, (∃ j : ℕ, 20 * m = 26 * j + 4) → 20 * m < 1000 → 20 * m ≤ 20 * n) →
  20 * n = 940 :=
by sorry

end trumpington_band_size_l1024_102406


namespace correct_calculation_l1024_102489

theorem correct_calculation (a b : ℝ) : 4 * a^2 * b - 3 * b * a^2 = a^2 * b := by
  sorry

end correct_calculation_l1024_102489


namespace gcd_228_1995_base3_11102_to_decimal_l1024_102487

-- Problem 1: GCD of 228 and 1995
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by sorry

-- Problem 2: Base 3 to decimal conversion
def base3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base3_11102_to_decimal :
  base3_to_decimal [1, 1, 1, 0, 2] = 119 := by sorry

end gcd_228_1995_base3_11102_to_decimal_l1024_102487


namespace toy_store_revenue_l1024_102426

theorem toy_store_revenue (december : ℝ) (november january : ℝ) 
  (h1 : november = (3/5) * december) 
  (h2 : january = (1/3) * november) : 
  december = (5/2) * ((november + january) / 2) := by
  sorry

end toy_store_revenue_l1024_102426


namespace linear_dependence_condition_l1024_102461

/-- Two 2D vectors are linearly dependent -/
def linearlyDependent (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), (a, b) ≠ (0, 0) ∧ a • v1 + b • v2 = (0, 0)

/-- The main theorem: vectors (2, 4) and (5, p) are linearly dependent iff p = 10 -/
theorem linear_dependence_condition (p : ℝ) :
  linearlyDependent (2, 4) (5, p) ↔ p = 10 := by
  sorry


end linear_dependence_condition_l1024_102461


namespace subset_implies_a_equals_one_l1024_102421

theorem subset_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {0, -a}
  let B : Set ℝ := {1, a-2, 2*a-2}
  A ⊆ B → a = 1 := by
  sorry

end subset_implies_a_equals_one_l1024_102421


namespace complex_magnitude_l1024_102486

theorem complex_magnitude (z : ℂ) (h : (2 - I) * z = 6 + 2 * I) : Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end complex_magnitude_l1024_102486


namespace total_selling_price_proof_l1024_102459

/-- Calculates the total selling price of toys given the number of toys sold,
    the cost price per toy, and the number of toys whose cost price equals the total gain. -/
def totalSellingPrice (numToysSold : ℕ) (costPricePerToy : ℕ) (numToysGain : ℕ) : ℕ :=
  numToysSold * costPricePerToy + numToysGain * costPricePerToy

/-- Proves that the total selling price of 18 toys is 16800,
    given a cost price of 800 per toy and a gain equal to the cost of 3 toys. -/
theorem total_selling_price_proof :
  totalSellingPrice 18 800 3 = 16800 := by
  sorry

end total_selling_price_proof_l1024_102459


namespace quarter_circles_sum_approaches_diameter_l1024_102456

/-- The sum of the lengths of the arcs of quarter circles approaches the diameter as n approaches infinity -/
theorem quarter_circles_sum_approaches_diameter (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * D / (4 * n)) - D| < ε :=
sorry

end quarter_circles_sum_approaches_diameter_l1024_102456


namespace no_real_solutions_l1024_102443

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 3) ∧ (x * y - z^2 = 3) ∧ (x = 2) := by
  sorry

end no_real_solutions_l1024_102443


namespace complex_exp_13pi_div_2_l1024_102494

theorem complex_exp_13pi_div_2 : Complex.exp (13 * π * Complex.I / 2) = Complex.I := by
  sorry

end complex_exp_13pi_div_2_l1024_102494


namespace arithmetic_sequence_sum_l1024_102474

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 38 →
  a 4 + a 5 = 55 := by
sorry

end arithmetic_sequence_sum_l1024_102474


namespace geometric_sequence_problem_l1024_102468

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a with a₂ = 4 and a₆ = 6, prove that a₁₀ = 9 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_a2 : a 2 = 4) 
    (h_a6 : a 6 = 6) : 
  a 10 = 9 := by
sorry

end geometric_sequence_problem_l1024_102468


namespace journey_time_proof_l1024_102488

/-- The total distance of the journey in miles -/
def total_distance : ℝ := 120

/-- The speed of the car in miles per hour -/
def car_speed : ℝ := 30

/-- The walking speed in miles per hour -/
def walking_speed : ℝ := 5

/-- The distance Tom and Harry initially travel by car -/
def initial_car_distance : ℝ := 40

/-- Theorem stating that under the given conditions, the total journey time is 52/3 hours -/
theorem journey_time_proof :
  ∃ (T d : ℝ),
    -- Tom and Harry's initial car journey
    car_speed * (4/3) = initial_car_distance ∧
    -- Harry's walk back
    walking_speed * (T - 4/3) = d ∧
    -- Dick's walk
    walking_speed * (T - 4/3) = total_distance - d ∧
    -- Tom's return journey
    car_speed * T = 2 * initial_car_distance + d ∧
    -- Total journey time
    T = 52/3 := by
  sorry

end journey_time_proof_l1024_102488


namespace determinant_expansion_second_column_l1024_102417

theorem determinant_expansion_second_column 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) : 
  let M := ![![a₁, 3, b₁], ![a₂, 2, b₂], ![a₃, -2, b₃]]
  Matrix.det M = 3 * Matrix.det ![![a₂, b₂], ![a₃, b₃]] + 
                 2 * Matrix.det ![![a₁, b₁], ![a₃, b₃]] - 
                 2 * Matrix.det ![![a₁, b₁], ![a₂, b₂]] := by
  sorry

end determinant_expansion_second_column_l1024_102417


namespace inequality_solution_part1_inequality_solution_part2_l1024_102400

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the inequality function
def inequality (x a : ℝ) : Prop := lg (|x + 3| + |x - 7|) > a

theorem inequality_solution_part1 :
  ∀ x : ℝ, inequality x 1 ↔ (x < -3 ∨ x > 7) := by sorry

theorem inequality_solution_part2 :
  ∀ a : ℝ, (∀ x : ℝ, inequality x a) ↔ a < 1 := by sorry

end inequality_solution_part1_inequality_solution_part2_l1024_102400


namespace exists_expression_equal_100_l1024_102480

/-- An arithmetic expression using numbers 1 to 9 --/
inductive Expr
  | num : Fin 9 → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression --/
def eval : Expr → ℚ
  | Expr.num n => n.val + 1
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Check if an expression uses each number from 1 to 9 exactly once --/
def usesAllNumbers : Expr → Bool := sorry

/-- Theorem: There exists an arithmetic expression using numbers 1 to 9 that evaluates to 100 --/
theorem exists_expression_equal_100 : 
  ∃ e : Expr, usesAllNumbers e ∧ eval e = 100 := by sorry

end exists_expression_equal_100_l1024_102480


namespace exists_y_d_satisfying_equation_l1024_102482

theorem exists_y_d_satisfying_equation : ∃ (y d : ℕ), 3^y + 6*d = 735 := by sorry


end exists_y_d_satisfying_equation_l1024_102482


namespace haley_sunday_tv_hours_l1024_102413

/-- Represents the number of hours Haley watched TV -/
structure TVWatchingHours where
  saturday : ℕ
  total : ℕ

/-- Calculates the number of hours Haley watched TV on Sunday -/
def sunday_hours (h : TVWatchingHours) : ℕ :=
  h.total - h.saturday

/-- Theorem stating that Haley watched TV for 3 hours on Sunday -/
theorem haley_sunday_tv_hours :
  ∀ h : TVWatchingHours, h.saturday = 6 → h.total = 9 → sunday_hours h = 3 := by
  sorry

end haley_sunday_tv_hours_l1024_102413


namespace seating_arrangements_count_l1024_102497

/-- The number of ways five people can sit in a row of six chairs -/
def seating_arrangements : ℕ :=
  let total_chairs : ℕ := 6
  let total_people : ℕ := 5
  let odd_numbered_chairs : ℕ := 3  -- chairs 1, 3, and 5
  odd_numbered_chairs * (total_chairs - 1) * (total_chairs - 2) * (total_chairs - 3) * (total_chairs - 4)

/-- Theorem stating that the number of seating arrangements is 360 -/
theorem seating_arrangements_count : seating_arrangements = 360 := by
  sorry

end seating_arrangements_count_l1024_102497


namespace simplify_radicals_l1024_102429

theorem simplify_radicals : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end simplify_radicals_l1024_102429


namespace probability_of_prime_is_two_fifths_l1024_102411

/-- A function that determines if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The set of numbers from 1 to 10 -/
def numberSet : Finset ℕ := sorry

/-- The set of prime numbers in the numberSet -/
def primeSet : Finset ℕ := sorry

/-- The probability of selecting a prime number from the numberSet -/
def probabilityOfPrime : ℚ := sorry

theorem probability_of_prime_is_two_fifths : 
  probabilityOfPrime = 2 / 5 := sorry

end probability_of_prime_is_two_fifths_l1024_102411


namespace units_digit_of_m_squared_plus_two_to_m_l1024_102495

def m : ℕ := 2023^2 + 2^2023

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : 
  m = 2023^2 + 2^2023 → (m^2 + 2^m) % 10 = 7 := by
  sorry

end units_digit_of_m_squared_plus_two_to_m_l1024_102495


namespace binomial_distribution_params_l1024_102496

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectation (bd : BinomialDistribution) : ℝ := bd.n * bd.p

/-- The variance of a binomial distribution -/
def variance (bd : BinomialDistribution) : ℝ := bd.n * bd.p * (1 - bd.p)

/-- Theorem: For a binomial distribution with expectation 8 and variance 1.6,
    the parameters are n = 10 and p = 0.8 -/
theorem binomial_distribution_params :
  ∀ (bd : BinomialDistribution),
    expectation bd = 8 →
    variance bd = 1.6 →
    bd.n = 10 ∧ bd.p = 0.8 := by
  sorry

end binomial_distribution_params_l1024_102496


namespace arctan_arcsin_arccos_sum_l1024_102458

theorem arctan_arcsin_arccos_sum : Real.arctan (Real.sqrt 3 / 3) + Real.arcsin (-1/2) + Real.arccos 1 = 0 := by
  sorry

end arctan_arcsin_arccos_sum_l1024_102458


namespace find_multiple_l1024_102414

theorem find_multiple (x y m : ℝ) : 
  x + y = 8 → 
  y - m * x = 7 → 
  y - x = 7.5 → 
  m = 3 := by
sorry

end find_multiple_l1024_102414


namespace expression_value_l1024_102407

theorem expression_value (x y z w : ℝ) 
  (eq1 : 4 * x * z + y * w = 3) 
  (eq2 : x * w + y * z = 6) : 
  (2 * x + y) * (2 * z + w) = 15 := by
  sorry

end expression_value_l1024_102407


namespace pipe_A_fill_time_l1024_102441

/-- The time it takes for pipe A to fill the cistern -/
def fill_time_A : ℝ := 10

/-- The time it takes for pipe B to empty the cistern -/
def empty_time_B : ℝ := 12

/-- The time it takes to fill the cistern with both pipes open -/
def fill_time_both : ℝ := 60

/-- Theorem stating that the fill time for pipe A is correct -/
theorem pipe_A_fill_time :
  fill_time_A = 10 ∧
  (1 / fill_time_A - 1 / empty_time_B = 1 / fill_time_both) :=
sorry

end pipe_A_fill_time_l1024_102441


namespace derivative_at_point_is_constant_l1024_102447

/-- The derivative of a function at a point is a constant value. -/
theorem derivative_at_point_is_constant (f : ℝ → ℝ) (a : ℝ) : 
  ∃ (c : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |x - a| ≠ 0 → 
    |(f x - f a) / (x - a) - c| < ε :=
sorry

end derivative_at_point_is_constant_l1024_102447


namespace smallest_d_for_perfect_square_l1024_102476

theorem smallest_d_for_perfect_square : ∃ (n : ℕ), 
  14 * 3150 = n^2 ∧ 
  ∀ (d : ℕ), d > 0 ∧ d < 14 → ¬∃ (m : ℕ), d * 3150 = m^2 :=
by sorry

end smallest_d_for_perfect_square_l1024_102476


namespace cheryl_material_calculation_l1024_102422

/-- The amount of the second type of material Cheryl needed for her project -/
def second_material_amount : ℚ := 1 / 8

/-- The amount of the first type of material Cheryl bought -/
def first_material_amount : ℚ := 2 / 9

/-- The amount of material Cheryl had left after the project -/
def leftover_amount : ℚ := 4 / 18

/-- The total amount of material Cheryl used -/
def total_used : ℚ := 1 / 8

theorem cheryl_material_calculation :
  second_material_amount = 
    (first_material_amount + leftover_amount + total_used) - first_material_amount := by
  sorry

end cheryl_material_calculation_l1024_102422


namespace fraction_simplification_l1024_102478

theorem fraction_simplification :
  let x := 5 / (1 + (32 * (Real.cos (15 * π / 180))^4 - 10 - 8 * Real.sqrt 3)^(1/3))
  x = 1 - 4^(1/3) + 16^(1/3) := by
  sorry

end fraction_simplification_l1024_102478


namespace non_foreign_male_students_l1024_102475

theorem non_foreign_male_students 
  (total_students : ℕ) 
  (female_ratio : ℚ) 
  (foreign_male_ratio : ℚ) :
  total_students = 300 →
  female_ratio = 2/3 →
  foreign_male_ratio = 1/10 →
  (total_students : ℚ) * (1 - female_ratio) * (1 - foreign_male_ratio) = 90 := by
  sorry

end non_foreign_male_students_l1024_102475


namespace mans_age_twice_sons_l1024_102485

/-- 
Proves that the number of years until a man's age is twice his son's age is 2,
given the initial conditions of their ages.
-/
theorem mans_age_twice_sons (man_age son_age : ℕ) (h1 : man_age = son_age + 26) (h2 : son_age = 24) :
  ∃ y : ℕ, y = 2 ∧ man_age + y = 2 * (son_age + y) :=
by sorry

end mans_age_twice_sons_l1024_102485


namespace sqrt_200_equals_10_l1024_102469

theorem sqrt_200_equals_10 : Real.sqrt 200 = 10 := by
  sorry

end sqrt_200_equals_10_l1024_102469


namespace olivia_basketball_cards_l1024_102420

theorem olivia_basketball_cards 
  (basketball_price : ℕ)
  (baseball_decks : ℕ)
  (baseball_price : ℕ)
  (total_paid : ℕ)
  (change : ℕ)
  (h1 : basketball_price = 3)
  (h2 : baseball_decks = 5)
  (h3 : baseball_price = 4)
  (h4 : total_paid = 50)
  (h5 : change = 24) :
  ∃ (x : ℕ), x * basketball_price + baseball_decks * baseball_price = total_paid - change ∧ x = 2 :=
by sorry

end olivia_basketball_cards_l1024_102420


namespace jake_and_sister_weight_l1024_102437

theorem jake_and_sister_weight (jake_weight : ℕ) (sister_weight : ℕ) : 
  jake_weight = 113 →
  jake_weight - 33 = 2 * sister_weight →
  jake_weight + sister_weight = 153 := by
sorry

end jake_and_sister_weight_l1024_102437


namespace second_fraction_greater_l1024_102431

/-- Define the first fraction -/
def fraction1 : ℚ := (77 * 10^2009 + 7) / (77.77 * 10^2010)

/-- Define the second fraction -/
def fraction2 : ℚ := (33 * (10^2010 - 1) / 9) / (33 * (10^2011 - 1) / 99)

/-- Theorem stating that the second fraction is greater than the first -/
theorem second_fraction_greater : fraction2 > fraction1 := by
  sorry

end second_fraction_greater_l1024_102431


namespace proposition_conditions_l1024_102446

theorem proposition_conditions (p q : Prop) : 
  (p ∨ q) → ¬(p ∧ q) → ¬p → (¬p ∧ q) := by sorry

end proposition_conditions_l1024_102446


namespace fraction_equation_solution_l1024_102490

theorem fraction_equation_solution : 
  {x : ℝ | (1 / (x^2 + 17*x + 20) + 1 / (x^2 + 12*x + 20) + 1 / (x^2 - 15*x + 20) = 0) ∧ 
           x ≠ -20 ∧ x ≠ -5 ∧ x ≠ -4 ∧ x ≠ -1} = 
  {x : ℝ | x = -20 ∨ x = -5 ∨ x = -4 ∨ x = -1} := by
sorry

end fraction_equation_solution_l1024_102490


namespace alarm_clock_noon_time_l1024_102481

/-- Represents the time difference between a correct clock and a slow clock -/
def timeDifference (slowRate : ℚ) (elapsedTime : ℚ) : ℚ :=
  slowRate * elapsedTime

/-- Calculates the time until noon for the slow clock -/
def timeUntilNoon (slowRate : ℚ) (elapsedTime : ℚ) : ℚ :=
  60 - (60 - timeDifference slowRate elapsedTime) % 60

theorem alarm_clock_noon_time (slowRate elapsedTime : ℚ) 
  (h1 : slowRate = 4 / 60) 
  (h2 : elapsedTime = 7 / 2) : 
  timeUntilNoon slowRate elapsedTime = 14 := by
sorry

#eval timeUntilNoon (4/60) (7/2)

end alarm_clock_noon_time_l1024_102481


namespace lexi_run_distance_l1024_102439

/-- Proves that running 13 laps on a quarter-mile track equals 3.25 miles -/
theorem lexi_run_distance (lap_length : ℚ) (num_laps : ℕ) : 
  lap_length = 1/4 → num_laps = 13 → lap_length * num_laps = 3.25 := by
  sorry

end lexi_run_distance_l1024_102439


namespace function_characterization_l1024_102412

def is_valid_function (f : ℝ → ℝ) : Prop :=
  (∀ m n : ℝ, f (m + n) = f m + f n - 6) ∧
  (∃ k : ℕ, 0 < k ∧ k ≤ 5 ∧ f (-1) = k) ∧
  (∀ x : ℝ, x > -1 → f x > 0)

theorem function_characterization (f : ℝ → ℝ) (h : is_valid_function f) :
  ∃ k : ℕ, 0 < k ∧ k ≤ 5 ∧ ∀ x : ℝ, f x = k * x + 6 :=
sorry

end function_characterization_l1024_102412


namespace divide_by_recurring_decimal_l1024_102492

/-- The recurring decimal 0.363636... represented as a rational number -/
def recurring_decimal : ℚ := 4 / 11

/-- The result of dividing 12 by the recurring decimal 0.363636... -/
theorem divide_by_recurring_decimal : 12 / recurring_decimal = 33 := by
  sorry

end divide_by_recurring_decimal_l1024_102492


namespace matrix_product_result_l1024_102423

def odd_matrix (k : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  !![1, k; 0, 1]

def matrix_product : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range 50).foldl (λ acc i => acc * odd_matrix (2 * i + 1)) (odd_matrix 1)

theorem matrix_product_result :
  matrix_product = !![1, 2500; 0, 1] := by
  sorry

end matrix_product_result_l1024_102423


namespace room_length_perimeter_ratio_l1024_102467

/-- Given a rectangular room with length 19 feet and width 11 feet,
    prove that the ratio of its length to its perimeter is 19:60. -/
theorem room_length_perimeter_ratio :
  let length : ℕ := 19
  let width : ℕ := 11
  let perimeter : ℕ := 2 * (length + width)
  (length : ℚ) / perimeter = 19 / 60 := by sorry

end room_length_perimeter_ratio_l1024_102467


namespace sandbag_weight_increase_l1024_102477

/-- Proves that the percentage increase in weight of a heavier filling material compared to sand is 40% given specific conditions. -/
theorem sandbag_weight_increase (capacity : ℝ) (fill_level : ℝ) (actual_weight : ℝ) : 
  capacity = 250 →
  fill_level = 0.8 →
  actual_weight = 280 →
  (actual_weight - fill_level * capacity) / (fill_level * capacity) * 100 = 40 := by
sorry

end sandbag_weight_increase_l1024_102477


namespace sin_greater_cos_range_l1024_102425

theorem sin_greater_cos_range (x : ℝ) : 
  x ∈ Set.Ioo (0 : ℝ) (2 * Real.pi) → 
  (Real.sin x > Real.cos x ↔ x ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4)) :=
by sorry

end sin_greater_cos_range_l1024_102425


namespace num_bedrooms_is_three_l1024_102483

/-- The number of bedrooms in the house -/
def num_bedrooms : ℕ := 3

/-- Time to renovate one bedroom (in hours) -/
def bedroom_time : ℕ := 4

/-- Time to renovate the kitchen (in hours) -/
def kitchen_time : ℕ := 6

/-- Total renovation time (in hours) -/
def total_time : ℕ := 54

/-- Theorem: The number of bedrooms is 3 given the renovation times -/
theorem num_bedrooms_is_three :
  num_bedrooms = 3 ∧
  bedroom_time = 4 ∧
  kitchen_time = 6 ∧
  total_time = 54 ∧
  total_time = num_bedrooms * bedroom_time + kitchen_time + 2 * (num_bedrooms * bedroom_time + kitchen_time) :=
by sorry

end num_bedrooms_is_three_l1024_102483


namespace negation_of_existence_negation_of_specific_proposition_l1024_102418

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_specific_proposition :
  (¬ ∃ x : ℝ, x^2 - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2 > 0) := by sorry

end negation_of_existence_negation_of_specific_proposition_l1024_102418


namespace three_digit_palindrome_average_l1024_102419

/-- Reverses the digits of a three-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- Checks if a three-digit number is a palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  n % 10 = n / 100

theorem three_digit_palindrome_average (m n : ℕ) : 
  100 ≤ m ∧ m < 1000 ∧
  100 ≤ n ∧ n < 1000 ∧
  is_palindrome m ∧
  (m + n) / 2 = reverse_digits m ∧
  m = 161 ∧ n = 161 := by
  sorry

end three_digit_palindrome_average_l1024_102419


namespace difference_of_numbers_l1024_102472

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 45) (h2 : x * y = 504) : 
  |x - y| = 3 := by sorry

end difference_of_numbers_l1024_102472


namespace fruit_cost_percentage_increase_l1024_102463

theorem fruit_cost_percentage_increase (max_cost min_cost : ℝ) 
  (h_max : max_cost = 45)
  (h_min : min_cost = 30) :
  (max_cost - min_cost) / min_cost * 100 = 50 := by
sorry

end fruit_cost_percentage_increase_l1024_102463


namespace arithmetic_sequence_condition_l1024_102435

/-- Given an arithmetic sequence with first three terms 2x - 3, 3x + 1, and 5x + k,
    prove that k = 5 - x makes these terms form an arithmetic sequence. -/
theorem arithmetic_sequence_condition (x k : ℝ) : 
  let a₁ := 2*x - 3
  let a₂ := 3*x + 1
  let a₃ := 5*x + k
  (a₂ - a₁ = a₃ - a₂) → k = 5 - x := by
sorry

end arithmetic_sequence_condition_l1024_102435


namespace sequence_product_l1024_102410

/-- An arithmetic sequence where no term is zero -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m ∧ a n ≠ 0

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem sequence_product (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  a 4 - 2 * (a 7)^2 + 3 * a 8 = 0 →
  b 7 = a 7 →
  b 3 * b 7 * b 11 = 8 := by
  sorry

end sequence_product_l1024_102410


namespace inequality_range_l1024_102434

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (3 : ℝ)^(a*x - 1) < (1/3 : ℝ)^(a*x^2)) ↔ -4 < a ∧ a ≤ 0 :=
sorry

end inequality_range_l1024_102434


namespace expand_product_l1024_102427

theorem expand_product (x : ℝ) : 4 * (x - 5) * (x + 8) = 4 * x^2 + 12 * x - 160 := by
  sorry

end expand_product_l1024_102427


namespace cinema_ticket_cost_l1024_102498

/-- Given Samuel and Kevin's cinema outing expenses, prove their combined ticket cost --/
theorem cinema_ticket_cost (total_budget : ℕ) 
  (samuel_food_drink : ℕ) (kevin_drink : ℕ) (kevin_food : ℕ) 
  (h1 : total_budget = 20)
  (h2 : samuel_food_drink = 6)
  (h3 : kevin_drink = 2)
  (h4 : kevin_food = 4) :
  ∃ (samuel_ticket kevin_ticket : ℕ),
    samuel_ticket + kevin_ticket = total_budget - (samuel_food_drink + kevin_drink + kevin_food) :=
by sorry

end cinema_ticket_cost_l1024_102498


namespace ball_max_height_l1024_102462

/-- The height function of the ball's trajectory -/
def h (t : ℝ) : ℝ := -16 * t^2 + 32 * t + 15

/-- Theorem stating that the maximum height of the ball is 31 feet -/
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 31 :=
sorry

end ball_max_height_l1024_102462


namespace magnitude_of_z_l1024_102473

theorem magnitude_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end magnitude_of_z_l1024_102473


namespace literacy_test_probabilities_l1024_102470

/-- Scientific literacy test model -/
structure LiteracyTest where
  /-- Probability of answering a question correctly -/
  p_correct : ℝ
  /-- Number of questions in the test -/
  total_questions : ℕ
  /-- Number of correct answers in a row needed for A rating -/
  a_threshold : ℕ
  /-- Number of incorrect answers in a row needed for C rating -/
  c_threshold : ℕ

/-- Probabilities of different outcomes in the literacy test -/
def test_probabilities (test : LiteracyTest) :
  (ℝ × ℝ × ℝ × ℝ) :=
  sorry

/-- The main theorem about the scientific literacy test -/
theorem literacy_test_probabilities :
  let test := LiteracyTest.mk (2/3) 5 4 3
  let (p_a, p_b, p_four, p_five) := test_probabilities test
  p_a = 64/243 ∧ p_b = 158/243 ∧ p_four = 2/9 ∧ p_five = 20/27 :=
sorry

end literacy_test_probabilities_l1024_102470


namespace ball_bounce_distance_l1024_102460

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistanceTraveled (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  let descentDistances := List.range (bounces + 1) |>.map (fun i => initialHeight * reboundRatio ^ i)
  let ascentDistances := descentDistances.tail
  (descentDistances.sum + ascentDistances.sum)

/-- The theorem stating the total distance traveled by the ball -/
theorem ball_bounce_distance :
  totalDistanceTraveled 200 (1/3) 4 = 397 + 2/9 := by
  sorry

end ball_bounce_distance_l1024_102460


namespace g_of_one_eq_neg_three_l1024_102454

-- Define the function f
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

-- State the theorem
theorem g_of_one_eq_neg_three
  (g : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f g (-x) + f g x = 0)
  (h2 : g (-1) = 1) :
  g 1 = -3 :=
by sorry

end g_of_one_eq_neg_three_l1024_102454


namespace perpendicular_vectors_imply_a_equals_two_l1024_102409

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The dot product of two Vector2D -/
def dot_product (v w : Vector2D) : ℝ :=
  v.x * w.x + v.y * w.y

/-- Perpendicularity of two Vector2D -/
def perpendicular (v w : Vector2D) : Prop :=
  dot_product v w = 0

theorem perpendicular_vectors_imply_a_equals_two (a : ℝ) :
  let m : Vector2D := ⟨a, 2⟩
  let n : Vector2D := ⟨1, 1 - a⟩
  perpendicular m n → a = 2 := by
  sorry

end perpendicular_vectors_imply_a_equals_two_l1024_102409


namespace integer_fraction_sum_equals_three_l1024_102448

theorem integer_fraction_sum_equals_three (a b : ℕ+) :
  let A := (a + 1 : ℝ) / b + b / a
  (∃ k : ℤ, A = k) → A = 3 := by
  sorry

end integer_fraction_sum_equals_three_l1024_102448


namespace chess_game_duration_l1024_102438

theorem chess_game_duration (game_hours : ℕ) (game_minutes : ℕ) (analysis_minutes : ℕ) : 
  game_hours = 20 → game_minutes = 15 → analysis_minutes = 22 →
  game_hours * 60 + game_minutes + analysis_minutes = 1237 := by
  sorry

end chess_game_duration_l1024_102438


namespace equivalence_condition_l1024_102450

/-- Hyperbola C with equation x² - y²/3 = 1 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- Left vertex of the hyperbola -/
def A₁ : ℝ × ℝ := (-1, 0)

/-- Right vertex of the hyperbola -/
def A₂ : ℝ × ℝ := (1, 0)

/-- Moving line l with equation x = my + n -/
def line_l (m n y : ℝ) : ℝ := m * y + n

/-- Intersection point T of A₁M and A₂N -/
structure Point_T (m n : ℝ) where
  x₀ : ℝ
  y₀ : ℝ
  on_A₁M : ∃ (x₁ y₁ : ℝ), hyperbola_C x₁ y₁ ∧ y₀ = (y₁ / (x₁ + 1)) * (x₀ + 1)
  on_A₂N : ∃ (x₂ y₂ : ℝ), hyperbola_C x₂ y₂ ∧ y₀ = (y₂ / (x₂ - 1)) * (x₀ - 1)
  on_line_l : x₀ = line_l m n y₀

/-- The main theorem to prove -/
theorem equivalence_condition (m : ℝ) :
  ∀ (n : ℝ), (∃ (T : Point_T m n), n = 2 ↔ T.x₀ = 1/2) := by sorry

end equivalence_condition_l1024_102450


namespace jovanas_shells_l1024_102453

theorem jovanas_shells (initial_shells : ℕ) : 
  initial_shells + 23 = 28 → initial_shells = 5 := by
  sorry

end jovanas_shells_l1024_102453


namespace f_of_two_eq_two_fifths_l1024_102451

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 - Real.sin x * Real.cos x

theorem f_of_two_eq_two_fifths : f (Real.arctan 2) = 2/5 := by
  sorry

end f_of_two_eq_two_fifths_l1024_102451


namespace negation_of_existence_cubic_equation_negation_l1024_102436

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ ∀ x, f x ≠ 0 := by sorry

theorem cubic_equation_negation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by
  apply negation_of_existence

end negation_of_existence_cubic_equation_negation_l1024_102436


namespace no_real_solutions_l1024_102428

theorem no_real_solutions :
  ∀ x : ℝ, (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 12 * x^9 :=
by sorry

end no_real_solutions_l1024_102428


namespace inequality_preservation_l1024_102465

theorem inequality_preservation (x y z : ℝ) (k : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x * y * z = 1) (h5 : 1/x + 1/y + 1/z ≥ x + y + z) :
  1/(x^k) + 1/(y^k) + 1/(z^k) ≥ x^k + y^k + z^k := by
  sorry

end inequality_preservation_l1024_102465


namespace larger_integer_value_l1024_102493

theorem larger_integer_value (a b : ℕ+) (h1 : (a : ℚ) / (b : ℚ) = 7 / 3) (h2 : (a : ℕ) * b = 189) : a = 21 := by
  sorry

end larger_integer_value_l1024_102493


namespace quadratic_equation_roots_l1024_102452

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*x₁ - 6 = 0) ∧ (x₂^2 - 2*x₂ - 6 = 0) := by
  sorry

end quadratic_equation_roots_l1024_102452


namespace locus_of_G_l1024_102408

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point F
def F : ℝ × ℝ := (2, 0)

-- Define the locus W
def W (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Theorem statement
theorem locus_of_G (x y : ℝ) :
  (∃ (h : ℝ × ℝ), unit_circle h.1 h.2 ∧
    (∃ (c : ℝ × ℝ), (c.1 - F.1)^2 + (c.2 - F.2)^2 = (c.1 - x)^2 + (c.2 - y)^2 ∧
      (c.1 - h.1)^2 + (c.2 - h.2)^2 = ((c.1 - F.1)^2 + (c.2 - F.2)^2) / 4)) →
  W x y :=
sorry

end locus_of_G_l1024_102408


namespace sugar_needed_proof_l1024_102430

/-- Given a recipe requiring a total amount of flour, with some flour already added,
    and the remaining flour needed being 2 cups more than the sugar needed,
    prove that the amount of sugar needed is correct. -/
theorem sugar_needed_proof 
  (total_flour : ℕ)  -- Total flour needed
  (added_flour : ℕ)  -- Flour already added
  (h1 : total_flour = 11)  -- Total flour is 11 cups
  (h2 : added_flour = 2)   -- 2 cups of flour already added
  : 
  total_flour - added_flour - 2 = 7  -- Sugar needed is 7 cups
  := by sorry

end sugar_needed_proof_l1024_102430


namespace x_varies_as_three_sevenths_power_of_z_l1024_102457

/-- Given that x varies directly as the cube of y, and y varies directly as the seventh root of z,
    prove that x varies as the (3/7)th power of z. -/
theorem x_varies_as_three_sevenths_power_of_z 
  (x y z : ℝ) 
  (hxy : ∃ (k : ℝ), x = k * y^3) 
  (hyz : ∃ (j : ℝ), y = j * z^(1/7)) :
  ∃ (m : ℝ), x = m * z^(3/7) := by
sorry

end x_varies_as_three_sevenths_power_of_z_l1024_102457


namespace total_laces_is_6x_l1024_102424

/-- Given a number of shoe pairs, calculate the total number of laces needed -/
def total_laces (x : ℕ) : ℕ :=
  let lace_sets_per_pair := 2
  let color_options := 3
  x * lace_sets_per_pair * color_options

/-- Theorem stating that the total number of laces is 6x -/
theorem total_laces_is_6x (x : ℕ) : total_laces x = 6 * x := by
  sorry

end total_laces_is_6x_l1024_102424


namespace sufficient_not_necessary_condition_l1024_102445

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x ∈ Set.Ico 1 2 → (x^2 - a ≤ 0 → a > 4)) ∧
  ¬(∀ x : ℝ, x ∈ Set.Ico 1 2 → (a > 4 → x^2 - a ≤ 0)) :=
by sorry

end sufficient_not_necessary_condition_l1024_102445


namespace loose_coins_amount_l1024_102499

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def bills_given : ℕ := 40
def change_received : ℕ := 10

theorem loose_coins_amount : 
  (flour_cost + cake_stand_cost + change_received) - bills_given = 3 := by
  sorry

end loose_coins_amount_l1024_102499


namespace final_time_sum_is_82_l1024_102466

def initial_hour : Nat := 3
def initial_minute : Nat := 0
def initial_second : Nat := 0
def hours_elapsed : Nat := 314
def minutes_elapsed : Nat := 21
def seconds_elapsed : Nat := 56

def final_time (ih im is he me se : Nat) : Nat × Nat × Nat :=
  let total_seconds := (ih * 3600 + im * 60 + is + he * 3600 + me * 60 + se) % 86400
  let h := (total_seconds / 3600) % 12
  let m := (total_seconds % 3600) / 60
  let s := total_seconds % 60
  (h, m, s)

theorem final_time_sum_is_82 :
  let (h, m, s) := final_time initial_hour initial_minute initial_second hours_elapsed minutes_elapsed seconds_elapsed
  h + m + s = 82 := by sorry

end final_time_sum_is_82_l1024_102466


namespace work_day_ends_at_430pm_l1024_102402

-- Define the structure for time
structure Time where
  hours : Nat
  minutes : Nat

-- Define the work schedule
def workStartTime : Time := { hours := 8, minutes := 0 }
def lunchStartTime : Time := { hours := 13, minutes := 0 }
def lunchDuration : Nat := 30
def totalWorkHours : Nat := 8

-- Function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

-- Function to calculate time difference in hours
def timeDifferenceInHours (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes - (t1.hours * 60 + t1.minutes)) / 60

-- Theorem stating that Maria's work day ends at 4:30 P.M.
theorem work_day_ends_at_430pm :
  let lunchEndTime := addMinutes lunchStartTime lunchDuration
  let workBeforeLunch := timeDifferenceInHours workStartTime lunchStartTime
  let remainingWorkHours := totalWorkHours - workBeforeLunch
  let endTime := addMinutes lunchEndTime (remainingWorkHours * 60)
  endTime = { hours := 16, minutes := 30 } :=
by sorry

end work_day_ends_at_430pm_l1024_102402


namespace selection_theorem_l1024_102455

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of workers -/
def total_workers : ℕ := 11

/-- The number of workers who can do typesetting -/
def typesetting_workers : ℕ := 7

/-- The number of workers who can do printing -/
def printing_workers : ℕ := 6

/-- The number of workers to be selected for each task -/
def workers_per_task : ℕ := 4

/-- The number of ways to select workers for typesetting and printing -/
def selection_ways : ℕ := 
  choose typesetting_workers workers_per_task * 
  choose (total_workers - workers_per_task) workers_per_task +
  choose (printing_workers - workers_per_task + 1) (printing_workers - workers_per_task) * 
  choose 2 1 * 
  choose (typesetting_workers - 1) workers_per_task +
  choose (printing_workers - workers_per_task + 2) (printing_workers - workers_per_task) * 
  choose (typesetting_workers - 2) workers_per_task * 
  choose 2 2

theorem selection_theorem : selection_ways = 185 := by sorry

end selection_theorem_l1024_102455


namespace expression_evaluation_l1024_102415

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 5 + 1/2
  ((x^2 / (x - 1) - x + 1) / ((4*x^2 - 4*x + 1) / (1 - x))) = -Real.sqrt 5 / 10 :=
by sorry

end expression_evaluation_l1024_102415


namespace ali_money_problem_l1024_102444

theorem ali_money_problem (initial_money : ℝ) : 
  (initial_money / 2 - (initial_money / 2) / 3 = 160) → initial_money = 480 := by
  sorry

end ali_money_problem_l1024_102444


namespace triangle_sin_c_equals_one_l1024_102449

theorem triangle_sin_c_equals_one (a b c A B C : ℝ) : 
  a = 1 → 
  b = Real.sqrt 3 → 
  A + C = 2 * B → 
  0 < a ∧ 0 < b ∧ 0 < c → 
  0 < A ∧ 0 < B ∧ 0 < C → 
  A + B + C = π → 
  a / (Real.sin A) = b / (Real.sin B) → 
  a / (Real.sin A) = c / (Real.sin C) → 
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A → 
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B → 
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C → 
  Real.sin C = 1 := by
sorry

end triangle_sin_c_equals_one_l1024_102449


namespace max_value_constraint_l1024_102416

/-- Given a point (3,1) lying on the line mx + ny + 1 = 0 where mn > 0, 
    the maximum value of 3/m + 1/n is -16. -/
theorem max_value_constraint (m n : ℝ) : 
  m * n > 0 → 
  3 * m + n = -1 → 
  (3 / m + 1 / n) ≤ -16 ∧ 
  ∃ m₀ n₀ : ℝ, m₀ * n₀ > 0 ∧ 3 * m₀ + n₀ = -1 ∧ 3 / m₀ + 1 / n₀ = -16 := by
  sorry

end max_value_constraint_l1024_102416


namespace permutation_absolute_difference_equality_l1024_102401

theorem permutation_absolute_difference_equality :
  ∀ (a : Fin 2011 → Fin 2011), Function.Bijective a →
  ∃ j k : Fin 2011, j < k ∧ |a j - j| = |a k - k| :=
by
  sorry

end permutation_absolute_difference_equality_l1024_102401


namespace eight_six_four_combinations_l1024_102491

/-- The number of unique outfit combinations given the number of shirts, ties, and belts. -/
def outfitCombinations (shirts : ℕ) (ties : ℕ) (belts : ℕ) : ℕ :=
  shirts * ties * belts

/-- Theorem stating that 8 shirts, 6 ties, and 4 belts result in 192 unique combinations. -/
theorem eight_six_four_combinations :
  outfitCombinations 8 6 4 = 192 := by
  sorry

end eight_six_four_combinations_l1024_102491


namespace power_product_simplification_l1024_102464

theorem power_product_simplification : (-0.25)^11 * (-4)^12 = -4 := by
  sorry

end power_product_simplification_l1024_102464


namespace intersection_of_A_and_B_l1024_102442

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | 3*x - 2 ≥ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

end intersection_of_A_and_B_l1024_102442


namespace negation_of_forall_inequality_l1024_102471

theorem negation_of_forall_inequality (x : ℝ) :
  ¬(∀ x > 1, x - 1 > Real.log x) ↔ ∃ x > 1, x - 1 ≤ Real.log x :=
by sorry

end negation_of_forall_inequality_l1024_102471


namespace bread_slices_calculation_l1024_102403

/-- Represents the number of pieces a single slice of bread is torn into -/
def pieces_per_slice : ℕ := 4

/-- Represents the total number of bread pieces -/
def total_pieces : ℕ := 8

/-- Calculates the number of original bread slices -/
def original_slices : ℕ := total_pieces / pieces_per_slice

theorem bread_slices_calculation :
  original_slices = 2 := by sorry

end bread_slices_calculation_l1024_102403
