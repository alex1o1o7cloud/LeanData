import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3379_337930

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - a)^2 / (a^2 + (b + c)^2) +
  (c + a - b)^2 / (b^2 + (c + a)^2) +
  (a + b - c)^2 / (c^2 + (a + b)^2) ≥ 3/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3379_337930


namespace NUMINAMATH_CALUDE_coefficient_x5_expansion_l3379_337945

/-- The coefficient of x^5 in the expansion of (1+x)^2(1-x)^5 is -1 -/
theorem coefficient_x5_expansion : Int := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5_expansion_l3379_337945


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3379_337936

/-- Given a geometric sequence, prove that if the sum of the first n terms is 48
    and the sum of the first 2n terms is 60, then the sum of the first 3n terms is 63. -/
theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℝ) : 
  S n = 48 → S (2*n) = 60 → S (3*n) = 63 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3379_337936


namespace NUMINAMATH_CALUDE_complex_sum_and_reciprocal_l3379_337976

theorem complex_sum_and_reciprocal (z : ℂ) : z = 1 + I → z + 2 / z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_and_reciprocal_l3379_337976


namespace NUMINAMATH_CALUDE_linear_function_unique_solution_l3379_337984

/-- Given a linear function f(x) = ax + 19 where f(3) = 7, 
    prove that if f(t) = 15, then t = 1 -/
theorem linear_function_unique_solution 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = a * x + 19) 
  (h2 : f 3 = 7) 
  (t : ℝ) 
  (h3 : f t = 15) : 
  t = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_unique_solution_l3379_337984


namespace NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l3379_337902

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of positive integer divisors of n -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- The number of odd positive integer divisors of n -/
def numOddDivisors (n : ℕ) : ℕ := sorry

/-- The probability of randomly choosing an odd divisor from the positive integer divisors of n -/
def probOddDivisor (n : ℕ) : ℚ := (numOddDivisors n : ℚ) / (numDivisors n : ℚ)

theorem prob_odd_divisor_15_factorial :
  probOddDivisor (factorial 15) = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l3379_337902


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3379_337999

theorem fraction_subtraction (x : ℝ) : 
  x * 8000 - (1 / 20) * (1 / 100) * 8000 = 796 → x = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3379_337999


namespace NUMINAMATH_CALUDE_solve_complex_equation_l3379_337926

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (w : ℂ) : Prop :=
  2 + 3 * i * w = 4 - 2 * i * w

-- State the theorem
theorem solve_complex_equation :
  ∃ w : ℂ, equation w ∧ w = -2 * i / 5 :=
by sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l3379_337926


namespace NUMINAMATH_CALUDE_circle_slash_problem_l3379_337968

/-- Custom operation ⊘ defined as (a ⊘ b) = (√(k*a + b))^3 -/
noncomputable def circle_slash (k : ℝ) (a b : ℝ) : ℝ := (Real.sqrt (k * a + b)) ^ 3

/-- Theorem: If 9 ⊘ x = 64 and k = 3, then x = -11 -/
theorem circle_slash_problem (x : ℝ) (h1 : circle_slash 3 9 x = 64) : x = -11 := by
  sorry

end NUMINAMATH_CALUDE_circle_slash_problem_l3379_337968


namespace NUMINAMATH_CALUDE_optimus_prime_distance_l3379_337906

/-- Prove that the distance between points A and B is 750 km given the conditions in the problem --/
theorem optimus_prime_distance : ∀ (D S : ℝ),
  (D / S - D / (S * (1 + 1/4)) = 1) →
  (150 / S + (D - 150) / S - (150 / S + (D - 150) / (S * (1 + 1/5))) = 2/3) →
  D = 750 := by
  sorry

end NUMINAMATH_CALUDE_optimus_prime_distance_l3379_337906


namespace NUMINAMATH_CALUDE_prob_10_or_9_prob_less_than_7_l3379_337959

-- Define the probabilities
def p_10 : ℝ := 0.21
def p_9 : ℝ := 0.23
def p_8 : ℝ := 0.25
def p_7 : ℝ := 0.28

-- Theorem for the first question
theorem prob_10_or_9 : p_10 + p_9 = 0.44 := by sorry

-- Theorem for the second question
theorem prob_less_than_7 : 1 - (p_10 + p_9 + p_8 + p_7) = 0.03 := by sorry

end NUMINAMATH_CALUDE_prob_10_or_9_prob_less_than_7_l3379_337959


namespace NUMINAMATH_CALUDE_unique_solution_system_of_equations_l3379_337960

theorem unique_solution_system_of_equations :
  ∃! (x y : ℝ), x + 2 * y = 2 ∧ 3 * x - 4 * y = -24 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_of_equations_l3379_337960


namespace NUMINAMATH_CALUDE_smallest_square_partition_l3379_337971

theorem smallest_square_partition : ∃ (n : ℕ), 
  (40 ∣ n) ∧ 
  (49 ∣ n) ∧ 
  (∀ (m : ℕ), (40 ∣ m) ∧ (49 ∣ m) → m ≥ n) ∧
  n = 1960 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_partition_l3379_337971


namespace NUMINAMATH_CALUDE_x_value_for_y_4_l3379_337990

/-- The relationship between x, y, and z -/
def x_relation (x y z k : ℚ) : Prop :=
  x = k * (z / y^2)

/-- The function defining z in terms of y -/
def z_function (y : ℚ) : ℚ :=
  2 * y - 1

theorem x_value_for_y_4 (k : ℚ) :
  (∃ x₀ : ℚ, x_relation x₀ 3 (z_function 3) k ∧ x₀ = 1) →
  (∃ x : ℚ, x_relation x 4 (z_function 4) k ∧ x = 63/80) :=
by sorry

end NUMINAMATH_CALUDE_x_value_for_y_4_l3379_337990


namespace NUMINAMATH_CALUDE_inequality_proof_l3379_337931

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) : 
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3379_337931


namespace NUMINAMATH_CALUDE_max_x_value_l3379_337923

theorem max_x_value (x : ℤ) 
  (h : Real.log (2 * x + 1) / Real.log (1/4) < Real.log (x - 1) / Real.log (1/2)) : 
  x ≤ 3 ∧ ∃ y : ℤ, y ≤ 3 ∧ Real.log (2 * y + 1) / Real.log (1/4) < Real.log (y - 1) / Real.log (1/2) :=
sorry

end NUMINAMATH_CALUDE_max_x_value_l3379_337923


namespace NUMINAMATH_CALUDE_unique_perfect_square_grid_l3379_337979

/-- A type representing a 2x3 grid of natural numbers -/
def Grid := Fin 2 → Fin 3 → ℕ

/-- Check if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- Check if a Grid forms valid perfect squares horizontally and vertically -/
def is_valid_grid (g : Grid) : Prop :=
  (is_perfect_square (g 0 0 * 100 + g 0 1 * 10 + g 0 2)) ∧
  (is_perfect_square (g 1 0 * 100 + g 1 1 * 10 + g 1 2)) ∧
  (is_perfect_square (g 0 0 * 10 + g 1 0)) ∧
  (is_perfect_square (g 0 1 * 10 + g 1 1)) ∧
  (is_perfect_square (g 0 2 * 10 + g 1 2)) ∧
  (∀ i j, g i j < 10)

/-- The main theorem stating the existence and uniqueness of the solution -/
theorem unique_perfect_square_grid :
  ∃! g : Grid, is_valid_grid g ∧ g 0 0 = 8 ∧ g 0 1 = 4 ∧ g 0 2 = 1 ∧
                               g 1 0 = 1 ∧ g 1 1 = 9 ∧ g 1 2 = 6 :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_grid_l3379_337979


namespace NUMINAMATH_CALUDE_quadratic_inequality_minimum_l3379_337927

theorem quadratic_inequality_minimum (a b : ℝ) (h1 : Set.Icc 1 4 = {x : ℝ | x^2 - 5*a*x + b ≤ 0}) :
  let t (x y : ℝ) := a/x + b/y
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → t x y ≥ 9/2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_minimum_l3379_337927


namespace NUMINAMATH_CALUDE_number_equation_l3379_337993

theorem number_equation (x : ℝ) : (40 / 100) * x = (10 / 100) * 70 → x = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3379_337993


namespace NUMINAMATH_CALUDE_three_students_same_group_l3379_337956

/-- The number of students in the school -/
def total_students : ℕ := 900

/-- The number of lunch groups -/
def num_groups : ℕ := 4

/-- The size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- The probability of three specific students being in the same lunch group -/
def prob_same_group : ℚ := 1 / 16

theorem three_students_same_group :
  let n := total_students
  let k := num_groups
  let g := group_size
  prob_same_group = (g / n) * ((g - 1) / (n - 1)) * ((g - 2) / (n - 2)) :=
sorry

end NUMINAMATH_CALUDE_three_students_same_group_l3379_337956


namespace NUMINAMATH_CALUDE_max_fraction_of_three_numbers_l3379_337980

/-- Two-digit natural number -/
def TwoDigitNat : Type := {n : ℕ // 10 ≤ n ∧ n ≤ 99}

theorem max_fraction_of_three_numbers (x y z : TwoDigitNat) 
  (h : (x.val + y.val + z.val) / 3 = 60) :
  (x.val + y.val) / z.val ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_max_fraction_of_three_numbers_l3379_337980


namespace NUMINAMATH_CALUDE_probability_point_between_C_and_E_l3379_337987

/-- Given a line segment AB with points C, D, and E, where AB = 4AD = 8BC and E divides CD into two equal parts,
    the probability of a randomly selected point on AB falling between C and E is 5/16. -/
theorem probability_point_between_C_and_E (A B C D E : ℝ) : 
  A < C ∧ C < D ∧ D < E ∧ E < B →  -- Points are ordered on the line
  B - A = 4 * (D - A) →            -- AB = 4AD
  B - A = 8 * (C - B) →            -- AB = 8BC
  E - C = D - E →                  -- E divides CD into two equal parts
  (E - C) / (B - A) = 5 / 16 :=     -- Probability is 5/16
by sorry

end NUMINAMATH_CALUDE_probability_point_between_C_and_E_l3379_337987


namespace NUMINAMATH_CALUDE_difference_between_decimal_and_fraction_l3379_337932

theorem difference_between_decimal_and_fraction : 0.127 - (1 / 8 : ℚ) = 0.002 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_decimal_and_fraction_l3379_337932


namespace NUMINAMATH_CALUDE_insects_eaten_by_geckos_and_lizards_l3379_337920

/-- The number of insects eaten by geckos and lizards -/
def total_insects_eaten (num_geckos : ℕ) (insects_per_gecko : ℕ) (num_lizards : ℕ) : ℕ :=
  num_geckos * insects_per_gecko + num_lizards * (2 * insects_per_gecko)

/-- Theorem stating the total number of insects eaten in the given scenario -/
theorem insects_eaten_by_geckos_and_lizards :
  total_insects_eaten 5 6 3 = 66 := by
  sorry


end NUMINAMATH_CALUDE_insects_eaten_by_geckos_and_lizards_l3379_337920


namespace NUMINAMATH_CALUDE_remainder_of_sum_l3379_337954

theorem remainder_of_sum (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) :
  (x + 3 * u * y) % y = v :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_sum_l3379_337954


namespace NUMINAMATH_CALUDE_student_a_final_score_l3379_337970

/-- Calculate the final score for a test -/
def finalScore (totalQuestions : ℕ) (correctAnswers : ℕ) : ℕ :=
  let incorrectAnswers := totalQuestions - correctAnswers
  correctAnswers - 2 * incorrectAnswers

/-- Theorem: The final score for a test with 100 questions and 92 correct answers is 76 -/
theorem student_a_final_score :
  finalScore 100 92 = 76 := by
  sorry

end NUMINAMATH_CALUDE_student_a_final_score_l3379_337970


namespace NUMINAMATH_CALUDE_tv_monthly_payment_l3379_337913

/-- Calculates the monthly payment for a discounted television purchase with installments -/
theorem tv_monthly_payment 
  (original_price : ℝ) 
  (discount_rate : ℝ) 
  (first_installment : ℝ) 
  (num_installments : ℕ) 
  (h1 : original_price = 480) 
  (h2 : discount_rate = 0.05) 
  (h3 : first_installment = 150) 
  (h4 : num_installments = 3) : 
  ∃ (monthly_payment : ℝ), 
    monthly_payment = (original_price * (1 - discount_rate) - first_installment) / num_installments ∧ 
    monthly_payment = 102 := by
  sorry

end NUMINAMATH_CALUDE_tv_monthly_payment_l3379_337913


namespace NUMINAMATH_CALUDE_sequence_nature_l3379_337919

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem sequence_nature (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n^2) →
  (∀ n, a n = S n - S (n-1)) →
  arithmetic_sequence a 2 :=
sorry

end NUMINAMATH_CALUDE_sequence_nature_l3379_337919


namespace NUMINAMATH_CALUDE_days_missed_proof_l3379_337967

/-- The total number of days missed by Vanessa, Mike, and Sarah -/
def total_days_missed (v m s : ℕ) : ℕ := v + m + s

/-- Theorem: Given the conditions, the total number of days missed is 17 -/
theorem days_missed_proof (v m s : ℕ) 
  (h1 : v + m = 14)  -- Vanessa and Mike have missed 14 days total
  (h2 : m + s = 12)  -- Mike and Sarah have missed 12 days total
  (h3 : v = 5)       -- Vanessa missed 5 days of school alone
  : total_days_missed v m s = 17 := by
  sorry

#check days_missed_proof

end NUMINAMATH_CALUDE_days_missed_proof_l3379_337967


namespace NUMINAMATH_CALUDE_f_two_zeros_iff_a_in_range_l3379_337949

-- Define the function f(x) = 2x³ - ax² + 1
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - a * x^2 + 1

-- Define the interval [1/2, 2]
def interval : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}

-- Define the condition of having exactly two zeros in the interval
def has_two_zeros (a : ℝ) : Prop :=
  ∃ x y, x ∈ interval ∧ y ∈ interval ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
  ∀ z, z ∈ interval ∧ f a z = 0 → z = x ∨ z = y

-- State the theorem
theorem f_two_zeros_iff_a_in_range :
  ∀ a : ℝ, has_two_zeros a ↔ 3/2 < a ∧ a ≤ 17/4 :=
sorry

end NUMINAMATH_CALUDE_f_two_zeros_iff_a_in_range_l3379_337949


namespace NUMINAMATH_CALUDE_dans_remaining_money_l3379_337912

/-- Proves that if Dan has $3 and spends $1 on a candy bar, the amount of money left is $2. -/
theorem dans_remaining_money (initial_amount : ℕ) (candy_bar_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 3 →
  candy_bar_cost = 1 →
  remaining_amount = initial_amount - candy_bar_cost →
  remaining_amount = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l3379_337912


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3379_337995

theorem shaded_area_calculation (square_side : ℝ) (triangle1_base triangle1_height : ℝ) (triangle2_base triangle2_height : ℝ) :
  square_side = 40 →
  triangle1_base = 15 →
  triangle1_height = 20 →
  triangle2_base = 15 →
  triangle2_height = 10 →
  square_side * square_side - (0.5 * triangle1_base * triangle1_height + 0.5 * triangle2_base * triangle2_height) = 1375 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3379_337995


namespace NUMINAMATH_CALUDE_martin_purchase_cost_l3379_337921

/-- The cost of items at a store -/
structure StorePrices where
  pencil : ℕ
  notebook : ℕ
  eraser : ℕ

/-- The conditions of the problem -/
def store_conditions (prices : StorePrices) : Prop :=
  prices.notebook + prices.eraser = 85 ∧
  prices.pencil + prices.eraser = 45 ∧
  3 * prices.pencil + 3 * prices.notebook + 3 * prices.eraser = 315

/-- The theorem stating that Martin's purchase costs 80 cents -/
theorem martin_purchase_cost (prices : StorePrices) 
  (h : store_conditions prices) : 
  prices.pencil + prices.notebook = 80 := by
  sorry

end NUMINAMATH_CALUDE_martin_purchase_cost_l3379_337921


namespace NUMINAMATH_CALUDE_faster_train_speed_l3379_337962

/-- Proves the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed (train_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  crossing_time = 8 →
  let relative_speed := 2 * train_length / crossing_time
  let slower_speed := relative_speed / 3
  let faster_speed := 2 * slower_speed
  faster_speed = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l3379_337962


namespace NUMINAMATH_CALUDE_quadratic_constant_term_l3379_337940

theorem quadratic_constant_term (m : ℝ) : 
  (∀ x, (m - 3) * x^2 - 3 * x + m^2 = 9) → m = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_constant_term_l3379_337940


namespace NUMINAMATH_CALUDE_sport_water_amount_l3379_337983

/-- Represents the ratio of ingredients in a flavored drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- Standard formulation of the flavored drink -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

/-- Sport formulation of the flavored drink -/
def sport_ratio : DrinkRatio :=
  { flavoring := standard_ratio.flavoring,
    corn_syrup := standard_ratio.corn_syrup / 3,
    water := standard_ratio.water * 2 }

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 5

/-- Theorem: The amount of water in the sport formulation is 75 ounces -/
theorem sport_water_amount :
  (sport_ratio.water / sport_ratio.flavoring) * (sport_corn_syrup / sport_ratio.corn_syrup) * sport_ratio.flavoring = 75 := by
  sorry

end NUMINAMATH_CALUDE_sport_water_amount_l3379_337983


namespace NUMINAMATH_CALUDE_five_solutions_l3379_337933

/-- The number of positive integer solutions to 2x + y = 11 -/
def solution_count : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + p.2 = 11 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 11) (Finset.range 11))).card

/-- Theorem stating that there are exactly 5 positive integer solutions to 2x + y = 11 -/
theorem five_solutions : solution_count = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_solutions_l3379_337933


namespace NUMINAMATH_CALUDE_greatest_gcd_value_l3379_337903

def S (n : ℕ+) : ℕ := n^2

theorem greatest_gcd_value (n : ℕ+) :
  (∃ m : ℕ+, Nat.gcd (2 * S m + 10 * m) (m - 3) = 42) ∧
  (∀ k : ℕ+, Nat.gcd (2 * S k + 10 * k) (k - 3) ≤ 42) :=
sorry

end NUMINAMATH_CALUDE_greatest_gcd_value_l3379_337903


namespace NUMINAMATH_CALUDE_second_markdown_percentage_l3379_337924

theorem second_markdown_percentage
  (P : ℝ)  -- Original price
  (first_markdown_percent : ℝ := 50)  -- First markdown percentage
  (final_price_percent : ℝ := 45)  -- Final price as percentage of original
  (h_P_pos : P > 0)  -- Assumption that the original price is positive
  : ∃ (second_markdown_percent : ℝ),
    (1 - second_markdown_percent / 100) * ((100 - first_markdown_percent) / 100 * P) = 
    final_price_percent / 100 * P ∧
    second_markdown_percent = 10 := by
sorry

end NUMINAMATH_CALUDE_second_markdown_percentage_l3379_337924


namespace NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_increase_percentage_l3379_337939

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l) * (1.2 * w) = 1.56 * (l * w) := by
  sorry

theorem rectangle_area_increase_percentage (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  ((1.3 * l) * (1.2 * w) - l * w) / (l * w) = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_increase_percentage_l3379_337939


namespace NUMINAMATH_CALUDE_a_in_M_l3379_337910

def M : Set ℝ := { x | x ≤ 5 }

def a : ℝ := 2

theorem a_in_M : a ∈ M := by sorry

end NUMINAMATH_CALUDE_a_in_M_l3379_337910


namespace NUMINAMATH_CALUDE_paint_room_combinations_l3379_337994

theorem paint_room_combinations (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 2) :
  (Nat.choose n k) * k.factorial = 72 := by
  sorry

end NUMINAMATH_CALUDE_paint_room_combinations_l3379_337994


namespace NUMINAMATH_CALUDE_negative_abs_of_negative_one_l3379_337918

theorem negative_abs_of_negative_one : -|-1| = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_of_negative_one_l3379_337918


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_quadratic_l3379_337937

theorem negation_of_universal_positive_quadratic (p : Prop) :
  (p ↔ ∀ x : ℝ, x^2 - x + 1 > 0) →
  (¬p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_quadratic_l3379_337937


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_l3379_337938

/-- The final stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease)

/-- Theorem stating the final stock price after two years -/
theorem stock_price_after_two_years :
  final_stock_price 120 0.80 0.30 = 151.20 := by
  sorry


end NUMINAMATH_CALUDE_stock_price_after_two_years_l3379_337938


namespace NUMINAMATH_CALUDE_range_of_a_min_value_of_fraction_l3379_337917

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + 4 * x + b

-- Theorem 1
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a 2 x ≥ 0) → a ≥ -5/2 :=
sorry

-- Theorem 2
theorem min_value_of_fraction (a b : ℝ) :
  a > b →
  (∀ x : ℝ, f a b x ≥ 0) →
  (∃ x₀ : ℝ, f a b x₀ = 0) →
  (a^2 + b^2) / (a - b) ≥ 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_min_value_of_fraction_l3379_337917


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l3379_337977

/-- The area of a square wrapping paper used to wrap a rectangular box with a square base. -/
theorem wrapping_paper_area (w h x : ℝ) (hw : w > 0) (hh : h > 0) (hx : x ≥ 0) :
  let s := Real.sqrt ((h + x)^2 + (w/2)^2)
  s^2 = (h + x)^2 + w^2/4 :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l3379_337977


namespace NUMINAMATH_CALUDE_equal_numbers_from_different_sequences_l3379_337950

/-- Represents a sequence of consecutive natural numbers -/
def ConsecutiveSequence (start : ℕ) (length : ℕ) : List ℕ :=
  List.range length |>.map (· + start)

/-- Concatenates a list of natural numbers into a single number -/
def concatenateToNumber (list : List ℕ) : ℕ := sorry

theorem equal_numbers_from_different_sequences :
  ∃ (a b : ℕ) (orderA : List ℕ → List ℕ) (orderB : List ℕ → List ℕ),
    let seqA := ConsecutiveSequence a 20
    let seqB := ConsecutiveSequence b 21
    concatenateToNumber (orderA seqA) = concatenateToNumber (orderB seqB) := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_from_different_sequences_l3379_337950


namespace NUMINAMATH_CALUDE_distance_between_A_and_B_implies_a_eq_neg_one_l3379_337904

/-- Given two points A and B in 3D space, returns the square of the distance between them. -/
def distance_squared (A B : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, z₁) := A
  let (x₂, y₂, z₂) := B
  (x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2

theorem distance_between_A_and_B_implies_a_eq_neg_one :
  ∀ a : ℝ, 
  let A : ℝ × ℝ × ℝ := (-1, 1, -a)
  let B : ℝ × ℝ × ℝ := (-a, 3, -1)
  distance_squared A B = 4 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_A_and_B_implies_a_eq_neg_one_l3379_337904


namespace NUMINAMATH_CALUDE_jenn_bike_purchase_l3379_337988

/-- Calculates the amount left over after buying a bike, given the number of jars of quarters,
    quarters per jar, and the cost of the bike. -/
def money_left_over (num_jars : ℕ) (quarters_per_jar : ℕ) (bike_cost : ℚ) : ℚ :=
  (num_jars * quarters_per_jar * (1/4 : ℚ)) - bike_cost

/-- Proves that given 5 jars of quarters with 160 quarters per jar, and a bike costing $180,
    the amount left over after buying the bike is $20. -/
theorem jenn_bike_purchase : money_left_over 5 160 180 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jenn_bike_purchase_l3379_337988


namespace NUMINAMATH_CALUDE_final_result_l3379_337957

def loop_calculation (i : ℕ) (S : ℕ) : ℕ :=
  if i < 9 then S else loop_calculation (i - 1) (S * i)

theorem final_result :
  loop_calculation 11 1 = 990 := by
  sorry

end NUMINAMATH_CALUDE_final_result_l3379_337957


namespace NUMINAMATH_CALUDE_root_sum_fraction_l3379_337935

/-- Given a, b, c are roots of x^3 - 20x^2 + 22, prove bc/a^2 + ac/b^2 + ab/c^2 = -40 -/
theorem root_sum_fraction (a b c : ℝ) : 
  (a^3 - 20*a^2 + 22 = 0) → 
  (b^3 - 20*b^2 + 22 = 0) → 
  (c^3 - 20*c^2 + 22 = 0) → 
  (b*c/a^2 + a*c/b^2 + a*b/c^2 = -40) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l3379_337935


namespace NUMINAMATH_CALUDE_rosies_pies_l3379_337996

/-- Given that Rosie can make 3 pies out of 12 apples, 
    this theorem proves how many pies she can make out of 36 apples. -/
theorem rosies_pies (apples_per_three_pies : ℕ) (total_apples : ℕ) : 
  apples_per_three_pies = 12 → total_apples = 36 → (total_apples / apples_per_three_pies) * 3 = 27 := by
  sorry

#check rosies_pies

end NUMINAMATH_CALUDE_rosies_pies_l3379_337996


namespace NUMINAMATH_CALUDE_set_B_elements_l3379_337963

def A : Set Int := {-2, 0, 1, 3}

def B : Set Int := {x | -x ∈ A ∧ (1 - x) ∉ A}

theorem set_B_elements : B = {-3, -1, 2} := by sorry

end NUMINAMATH_CALUDE_set_B_elements_l3379_337963


namespace NUMINAMATH_CALUDE_remainder_zero_l3379_337982

theorem remainder_zero : (88144 * 88145 + 88146 + 88147 + 88148 + 88149 + 88150) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_zero_l3379_337982


namespace NUMINAMATH_CALUDE_linear_functions_intersection_l3379_337911

theorem linear_functions_intersection (a b c d : ℝ) (h : a ≠ b) :
  (∃ x y : ℝ, y = a * x + a ∧ y = b * x + b ∧ y = c * x + d) → c = d := by
  sorry

end NUMINAMATH_CALUDE_linear_functions_intersection_l3379_337911


namespace NUMINAMATH_CALUDE_point_on_axes_with_inclination_l3379_337966

/-- Given point A(-2, 1) and the angle of inclination of line PA is 30°,
    prove that point P on the coordinate axes is either (-4, 0) or (0, 2). -/
theorem point_on_axes_with_inclination (A : ℝ × ℝ) (P : ℝ × ℝ) :
  A = (-2, 1) →
  (P.1 = 0 ∨ P.2 = 0) →
  (P.2 - A.2) / (P.1 - A.1) = Real.tan (30 * π / 180) →
  (P = (-4, 0) ∨ P = (0, 2)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_axes_with_inclination_l3379_337966


namespace NUMINAMATH_CALUDE_anne_distance_l3379_337929

/-- Calculates the distance traveled given time and speed -/
def distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Proves that wandering for 5 hours at 4 miles per hour results in a distance of 20 miles -/
theorem anne_distance : distance 5 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_anne_distance_l3379_337929


namespace NUMINAMATH_CALUDE_fraction_and_sum_problem_l3379_337972

theorem fraction_and_sum_problem :
  (5 : ℚ) / 40 = 0.125 ∧ 0.125 + 0.375 = 0.500 := by
  sorry

end NUMINAMATH_CALUDE_fraction_and_sum_problem_l3379_337972


namespace NUMINAMATH_CALUDE_martha_black_butterflies_l3379_337941

/-- The number of black butterflies in Martha's collection --/
def num_black_butterflies (total : ℕ) (blue : ℕ) (red : ℕ) : ℕ :=
  total - (blue + red)

/-- Proof that Martha has 34 black butterflies --/
theorem martha_black_butterflies :
  num_black_butterflies 56 12 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_martha_black_butterflies_l3379_337941


namespace NUMINAMATH_CALUDE_complex_number_location_l3379_337947

theorem complex_number_location (z : ℂ) (h : (1 - Complex.I) * z = Complex.I ^ 2013) :
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3379_337947


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3379_337998

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3379_337998


namespace NUMINAMATH_CALUDE_stating_perfect_match_equation_l3379_337914

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 22

/-- Represents the number of screws a worker can produce per day -/
def screws_per_worker : ℕ := 1200

/-- Represents the number of nuts a worker can produce per day -/
def nuts_per_worker : ℕ := 2000

/-- Represents the number of nuts needed for each screw -/
def nuts_per_screw : ℕ := 2

/-- 
Theorem stating that for a perfect match of products, 
the equation 2 × 1200(22 - x) = 2000x must hold, 
where x is the number of workers assigned to produce nuts
-/
theorem perfect_match_equation (x : ℕ) (h : x ≤ total_workers) : 
  2 * screws_per_worker * (total_workers - x) = nuts_per_worker * x := by
  sorry

end NUMINAMATH_CALUDE_stating_perfect_match_equation_l3379_337914


namespace NUMINAMATH_CALUDE_shared_root_quadratic_equation_l3379_337905

theorem shared_root_quadratic_equation (a b p q : ℝ) (h : a ≠ p ∧ b ≠ q) :
  ∃ (α β γ : ℝ),
    (α^2 + a*α + b = 0 ∧ α^2 + p*α + q = 0) →
    (β^2 + a*β + b = 0 ∧ β ≠ α) →
    (γ^2 + p*γ + q = 0 ∧ γ ≠ α) →
    (x^2 - (-p - (b - q)/(p - a))*x + (b*q*(p - a)^2)/(b - q)^2 = (x - β)*(x - γ)) := by
  sorry

end NUMINAMATH_CALUDE_shared_root_quadratic_equation_l3379_337905


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l3379_337907

-- Define the number we're working with
def n : Nat := 32767

-- Define a function to get the greatest prime divisor
def greatestPrimeDivisor (m : Nat) : Nat :=
  sorry

-- Define a function to sum the digits of a number
def sumOfDigits (m : Nat) : Nat :=
  sorry

-- The theorem to prove
theorem sum_of_digits_of_greatest_prime_divisor :
  sumOfDigits (greatestPrimeDivisor n) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l3379_337907


namespace NUMINAMATH_CALUDE_hexagon_shell_arrangements_l3379_337925

/-- The number of rotational symmetries in a regular hexagon -/
def hexagon_rotations : ℕ := 6

/-- The number of distinct points on the hexagon (corners and midpoints) -/
def total_points : ℕ := 12

/-- The number of distinct sea shells -/
def total_shells : ℕ := 12

/-- The number of distinct arrangements of sea shells on a regular hexagon,
    considering only rotational equivalence -/
def distinct_arrangements : ℕ := (Nat.factorial total_shells) / hexagon_rotations

theorem hexagon_shell_arrangements :
  distinct_arrangements = 79833600 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_shell_arrangements_l3379_337925


namespace NUMINAMATH_CALUDE_f_inequality_l3379_337901

/-- The number of ways to represent a positive integer as a sum of non-decreasing positive integers -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating that f(n+1) is less than or equal to the average of f(n) and f(n+2) for any positive integer n -/
theorem f_inequality (n : ℕ) (h : n > 0) : f (n + 1) ≤ (f n + f (n + 2)) / 2 := by sorry

end NUMINAMATH_CALUDE_f_inequality_l3379_337901


namespace NUMINAMATH_CALUDE_solve_equation_l3379_337946

theorem solve_equation (y : ℝ) (h : (9 / y^2) = (y / 36)) : y = (324 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3379_337946


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l3379_337955

theorem quadratic_two_roots
  (a b c d e : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > d)
  (h4 : e ≠ -1) :
  ∃ (x y : ℝ), x ≠ y ∧
  (e + 1) * x^2 - (a + c + b*e + d*e) * x + a*c + e*b*d = 0 ∧
  (e + 1) * y^2 - (a + c + b*e + d*e) * y + a*c + e*b*d = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l3379_337955


namespace NUMINAMATH_CALUDE_non_equilateral_combinations_l3379_337964

/-- The number of dots evenly spaced on the circle's circumference -/
def n : ℕ := 6

/-- The number of dots to be selected in each combination -/
def k : ℕ := 3

/-- The total number of combinations of k dots from n dots -/
def total_combinations : ℕ := Nat.choose n k

/-- The number of equilateral triangles that can be formed -/
def equilateral_triangles : ℕ := 2

/-- Theorem: The number of combinations of 3 dots that do not form an equilateral triangle
    is equal to the total number of 3-dot combinations minus the number of equilateral triangles -/
theorem non_equilateral_combinations :
  total_combinations - equilateral_triangles = 18 := by sorry

end NUMINAMATH_CALUDE_non_equilateral_combinations_l3379_337964


namespace NUMINAMATH_CALUDE_max_candy_leftover_l3379_337908

theorem max_candy_leftover (x : ℕ) : ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l3379_337908


namespace NUMINAMATH_CALUDE_democrat_ratio_l3379_337944

theorem democrat_ratio (total_participants male_participants female_participants male_democrats female_democrats : ℕ) :
  total_participants = 720 ∧
  female_participants = 240 ∧
  male_participants = 480 ∧
  female_democrats = 120 ∧
  2 * female_democrats = female_participants ∧
  3 * (male_democrats + female_democrats) = total_participants ∧
  male_participants + female_participants = total_participants →
  4 * male_democrats = male_participants :=
by
  sorry

end NUMINAMATH_CALUDE_democrat_ratio_l3379_337944


namespace NUMINAMATH_CALUDE_first_page_drawings_count_l3379_337997

/-- The number of drawings on the first page of an art book. -/
def first_page_drawings : ℕ := 5

/-- The increase in the number of drawings on each subsequent page. -/
def drawing_increase : ℕ := 5

/-- The total number of pages considered. -/
def total_pages : ℕ := 5

/-- The total number of drawings on the first five pages. -/
def total_drawings : ℕ := 75

/-- Theorem stating that the number of drawings on the first page is 5,
    given the conditions of the problem. -/
theorem first_page_drawings_count :
  (first_page_drawings +
   (first_page_drawings + drawing_increase) +
   (first_page_drawings + 2 * drawing_increase) +
   (first_page_drawings + 3 * drawing_increase) +
   (first_page_drawings + 4 * drawing_increase)) = total_drawings :=
by sorry

end NUMINAMATH_CALUDE_first_page_drawings_count_l3379_337997


namespace NUMINAMATH_CALUDE_slope_is_two_l3379_337928

/-- A linear function y = kx + b where y increases by 6 when x increases by 3 -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  increase_property : ∀ (x : ℝ), (k * (x + 3) + b) - (k * x + b) = 6

/-- Theorem: The slope k of the linear function is equal to 2 -/
theorem slope_is_two (f : LinearFunction) : f.k = 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_is_two_l3379_337928


namespace NUMINAMATH_CALUDE_books_remaining_on_shelf_l3379_337965

theorem books_remaining_on_shelf (initial_books : Real) (books_taken : Real) 
  (h1 : initial_books = 38.0) (h2 : books_taken = 10.0) : 
  initial_books - books_taken = 28.0 := by
  sorry

end NUMINAMATH_CALUDE_books_remaining_on_shelf_l3379_337965


namespace NUMINAMATH_CALUDE_some_students_not_club_members_l3379_337934

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (Punctual : U → Prop)
variable (ClubMember : U → Prop)
variable (FraternityMember : U → Prop)

-- Define the conditions
variable (h1 : ∃ x, Student x ∧ ¬Punctual x)
variable (h2 : ∀ x, ClubMember x → Punctual x)
variable (h3 : ∀ x, FraternityMember x → ¬ClubMember x)

-- State the theorem
theorem some_students_not_club_members :
  ∃ x, Student x ∧ ¬ClubMember x :=
sorry

end NUMINAMATH_CALUDE_some_students_not_club_members_l3379_337934


namespace NUMINAMATH_CALUDE_final_amount_is_47500_l3379_337951

def income : ℝ := 200000

def children_share : ℝ := 0.15
def num_children : ℕ := 3
def wife_share : ℝ := 0.30
def donation_rate : ℝ := 0.05

def final_amount : ℝ :=
  let children_total := children_share * num_children * income
  let wife_total := wife_share * income
  let remaining_after_family := income - children_total - wife_total
  let donation := donation_rate * remaining_after_family
  remaining_after_family - donation

theorem final_amount_is_47500 :
  final_amount = 47500 := by sorry

end NUMINAMATH_CALUDE_final_amount_is_47500_l3379_337951


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l3379_337942

theorem smallest_sum_of_squares (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    a + b = x^2 ∧ b + c = y^2 ∧ c + a = z^2 →
  55 ≤ a + b + c :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l3379_337942


namespace NUMINAMATH_CALUDE_arccos_neg_one_eq_pi_l3379_337953

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_neg_one_eq_pi_l3379_337953


namespace NUMINAMATH_CALUDE_bc_length_fraction_l3379_337922

/-- Given a line segment AD with points B and C on it, prove that BC = 5/36 * AD -/
theorem bc_length_fraction (A B C D : ℝ) : 
  (B - A) = 3 * (D - B) →  -- AB = 3 * BD
  (C - A) = 8 * (D - C) →  -- AC = 8 * CD
  (C - B) = (5 / 36) * (D - A) := by sorry

end NUMINAMATH_CALUDE_bc_length_fraction_l3379_337922


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3379_337915

theorem sin_2alpha_value (α : Real) (h : Real.cos (α - Real.pi / 4) = Real.sqrt 3 / 3) :
  Real.sin (2 * α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3379_337915


namespace NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l3379_337969

/-- Represents the repeating decimal 0.333... -/
def repeating_third : ℚ := 1 / 3

/-- Proves that 8 divided by 0.333... equals 24 -/
theorem eight_divided_by_repeating_third : 8 / repeating_third = 24 := by sorry

end NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l3379_337969


namespace NUMINAMATH_CALUDE_continued_fraction_sum_l3379_337943

theorem continued_fraction_sum (x y z : ℕ+) :
  (30 : ℚ) / 7 = x + 1 / (y + 1 / z) →
  x + y + z = 9 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_sum_l3379_337943


namespace NUMINAMATH_CALUDE_gravel_weight_in_specific_mixture_l3379_337900

/-- A cement mixture with sand, water, and gravel -/
structure CementMixture where
  total_weight : ℝ
  sand_fraction : ℝ
  water_fraction : ℝ

/-- Calculate the weight of gravel in a cement mixture -/
def gravel_weight (m : CementMixture) : ℝ :=
  m.total_weight * (1 - m.sand_fraction - m.water_fraction)

/-- Theorem stating the weight of gravel in the specific mixture -/
theorem gravel_weight_in_specific_mixture :
  let m : CementMixture := {
    total_weight := 120,
    sand_fraction := 1/5,
    water_fraction := 3/4
  }
  gravel_weight m = 6 := by
  sorry

end NUMINAMATH_CALUDE_gravel_weight_in_specific_mixture_l3379_337900


namespace NUMINAMATH_CALUDE_distribution_methods_l3379_337991

/-- Represents the number of ways to distribute books to students -/
def distribute_books (novels : ℕ) (picture_books : ℕ) (students : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of distribution methods -/
theorem distribution_methods :
  distribute_books 2 2 3 = 12 :=
by sorry

end NUMINAMATH_CALUDE_distribution_methods_l3379_337991


namespace NUMINAMATH_CALUDE_mark_young_fish_count_l3379_337916

/-- Calculates the total number of young fish given the number of tanks, pregnant fish per tank, and young per fish. -/
def total_young_fish (num_tanks : ℕ) (fish_per_tank : ℕ) (young_per_fish : ℕ) : ℕ :=
  num_tanks * fish_per_tank * young_per_fish

/-- Proves that given 5 tanks, 6 pregnant fish per tank, and 25 young per fish, the total number of young fish is 750. -/
theorem mark_young_fish_count :
  total_young_fish 5 6 25 = 750 := by
  sorry

end NUMINAMATH_CALUDE_mark_young_fish_count_l3379_337916


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3379_337958

/-- The equation of a hyperbola sharing foci with a given ellipse and passing through a specific point -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (a b : ℝ), (x^2 / 9 + y^2 / 5 = 1) ∧ 
   (x^2 / a^2 - y^2 / b^2 = 1) ∧
   (3^2 / a^2 - 2 / b^2 = 1) ∧
   (a^2 + b^2 = 4)) →
  (x^2 / 3 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3379_337958


namespace NUMINAMATH_CALUDE_point_on_curve_l3379_337978

theorem point_on_curve : ∃ θ : ℝ, 1 + Real.sin θ = 3/2 ∧ Real.sin (2*θ) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_l3379_337978


namespace NUMINAMATH_CALUDE_brian_read_75_chapters_l3379_337986

/-- The total number of chapters Brian read -/
def total_chapters : ℕ :=
  let book1 : ℕ := 20
  let book2 : ℕ := 15
  let book3 : ℕ := 15
  let first_three : ℕ := book1 + book2 + book3
  let book4 : ℕ := first_three / 2
  book1 + book2 + book3 + book4

/-- Proof that Brian read 75 chapters in total -/
theorem brian_read_75_chapters : total_chapters = 75 := by
  sorry

end NUMINAMATH_CALUDE_brian_read_75_chapters_l3379_337986


namespace NUMINAMATH_CALUDE_kim_cherry_saplings_l3379_337975

/-- Given that Kim plants 80 cherry pits, 25% of them sprout, and she sells 6 saplings,
    prove that she has 14 cherry saplings left. -/
theorem kim_cherry_saplings (total_pits : ℕ) (sprout_rate : ℚ) (sold_saplings : ℕ) :
  total_pits = 80 →
  sprout_rate = 1/4 →
  sold_saplings = 6 →
  (total_pits : ℚ) * sprout_rate - sold_saplings = 14 := by
  sorry

end NUMINAMATH_CALUDE_kim_cherry_saplings_l3379_337975


namespace NUMINAMATH_CALUDE_johns_initial_contribution_l3379_337948

theorem johns_initial_contribution 
  (total_initial : ℝ) 
  (total_final : ℝ) 
  (john_initial : ℝ) 
  (kelly_initial : ℝ) 
  (luke_initial : ℝ) 
  (h1 : total_initial = 1200)
  (h2 : total_final = 1800)
  (h3 : total_initial = john_initial + kelly_initial + luke_initial)
  (h4 : total_final = (john_initial - 200) + 3 * kelly_initial + 3 * luke_initial) :
  john_initial = 800 := by
sorry

end NUMINAMATH_CALUDE_johns_initial_contribution_l3379_337948


namespace NUMINAMATH_CALUDE_large_marshmallows_count_l3379_337992

/-- Represents the number of Rice Krispie Treats made -/
def rice_krispie_treats : ℕ := 5

/-- Represents the total number of marshmallows used -/
def total_marshmallows : ℕ := 18

/-- Represents the number of mini marshmallows used -/
def mini_marshmallows : ℕ := 10

/-- Represents the number of large marshmallows used -/
def large_marshmallows : ℕ := total_marshmallows - mini_marshmallows

theorem large_marshmallows_count : large_marshmallows = 8 := by
  sorry

end NUMINAMATH_CALUDE_large_marshmallows_count_l3379_337992


namespace NUMINAMATH_CALUDE_no_real_solution_l3379_337989

theorem no_real_solution : ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x^2 = 1 + 1/y^2 ∧ y^2 = 1 + 1/x^2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l3379_337989


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3379_337974

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 * a 3 * a 7 = 8 →
  a 4 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3379_337974


namespace NUMINAMATH_CALUDE_square_land_side_length_l3379_337909

theorem square_land_side_length 
  (area : ℝ) 
  (h : area = Real.sqrt 100) : 
  ∃ (side : ℝ), side * side = area ∧ side = 10 :=
sorry

end NUMINAMATH_CALUDE_square_land_side_length_l3379_337909


namespace NUMINAMATH_CALUDE_hex_numeric_count_and_sum_l3379_337961

/-- Represents a hexadecimal digit --/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Converts a natural number to its hexadecimal representation --/
def toHex (n : ℕ) : List HexDigit :=
  sorry

/-- Checks if a hexadecimal representation contains only numeric digits --/
def isAllNumeric (hex : List HexDigit) : Bool :=
  sorry

/-- Counts numbers up to n (exclusive) with only numeric hexadecimal digits --/
def countNumericHex (n : ℕ) : ℕ :=
  sorry

/-- Sums the digits of a natural number --/
def sumDigits (n : ℕ) : ℕ :=
  sorry

theorem hex_numeric_count_and_sum :
  countNumericHex 512 = 200 ∧ sumDigits 200 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hex_numeric_count_and_sum_l3379_337961


namespace NUMINAMATH_CALUDE_root_expression_value_l3379_337985

theorem root_expression_value (a : ℝ) : 
  (2 * a^2 - 7 * a - 1 = 0) → (a * (2 * a - 7) + 5 = 6) := by
  sorry

end NUMINAMATH_CALUDE_root_expression_value_l3379_337985


namespace NUMINAMATH_CALUDE_percentage_relation_l3379_337981

theorem percentage_relation (A B C : ℝ) (h1 : A = 1.71 * C) (h2 : A = 0.05 * B) : B = 14.2 * C := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3379_337981


namespace NUMINAMATH_CALUDE_reciprocal_sum_diff_l3379_337973

theorem reciprocal_sum_diff : (1 / (1/4 + 1/6 - 1/12) : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_diff_l3379_337973


namespace NUMINAMATH_CALUDE_investment_problem_l3379_337952

theorem investment_problem (x : ℝ) : 
  x > 0 ∧ 
  0.07 * x + 0.27 * 1500 = 0.22 * (x + 1500) → 
  x = 500 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l3379_337952
