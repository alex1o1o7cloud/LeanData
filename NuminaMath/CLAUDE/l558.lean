import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_eq_17965_l558_55823

/-- A number consisting of n repetitions of a digit d in base 10 -/
def repeatedDigit (d : ℕ) (n : ℕ) : ℕ :=
  d * ((10^n - 1) / 9)

/-- The sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

theorem sum_of_digits_9ab_eq_17965 :
  let a := repeatedDigit 4 1995
  let b := repeatedDigit 7 1995
  sumOfDigits (9 * a * b) = 17965 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_eq_17965_l558_55823


namespace NUMINAMATH_CALUDE_fraction_equality_l558_55847

theorem fraction_equality (x y : ℝ) (h : x / (2 * y) = 3 / 2) :
  (7 * x + 8 * y) / (x - 2 * y) = 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l558_55847


namespace NUMINAMATH_CALUDE_three_stamps_cost_l558_55860

/-- The cost of a single stamp in dollars -/
def stamp_cost : ℚ := 34 / 100

/-- The cost of two stamps in dollars -/
def two_stamps_cost : ℚ := 68 / 100

/-- Theorem: The cost of three stamps is $1.02 -/
theorem three_stamps_cost : stamp_cost * 3 = 102 / 100 := by
  sorry

end NUMINAMATH_CALUDE_three_stamps_cost_l558_55860


namespace NUMINAMATH_CALUDE_positive_slope_implies_positive_correlation_l558_55846

/-- A linear regression model relating variables x and y. -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope
  equation : ∀ x y : ℝ, y = a + b * x

/-- Definition of positive linear correlation between two variables. -/
def positively_correlated (x y : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → y x₁ < y x₂

/-- Theorem stating that a linear regression with positive slope implies positive correlation. -/
theorem positive_slope_implies_positive_correlation
  (model : LinearRegression)
  (h_positive_slope : model.b > 0) :
  positively_correlated (λ x => x) (λ x => model.a + model.b * x) :=
by
  sorry


end NUMINAMATH_CALUDE_positive_slope_implies_positive_correlation_l558_55846


namespace NUMINAMATH_CALUDE_prime_cube_plus_one_l558_55885

theorem prime_cube_plus_one (p : ℕ) (x y : ℕ+) (h_prime : Nat.Prime p) 
  (h_eq : p ^ x.val = y.val ^ 3 + 1) :
  ((p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_plus_one_l558_55885


namespace NUMINAMATH_CALUDE_arithmetic_sequence_angles_eq_solution_angles_l558_55867

open Real

-- Define the set of angles that satisfy the condition
def ArithmeticSequenceAngles : Set ℝ :=
  {a | 0 < a ∧ a < 2 * π ∧ 2 * sin (2 * a) = sin a + sin (3 * a)}

-- Define the set of solution angles in radians
def SolutionAngles : Set ℝ :=
  {π/6, 5*π/6, 7*π/6, 11*π/6}

-- Theorem statement
theorem arithmetic_sequence_angles_eq_solution_angles :
  ArithmeticSequenceAngles = SolutionAngles := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_angles_eq_solution_angles_l558_55867


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l558_55824

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let t : ℕ := 3^n + n^2
  let r : ℕ := 4^t - t^2
  r = 2^72 - 1296 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l558_55824


namespace NUMINAMATH_CALUDE_min_distinct_prime_factors_l558_55854

theorem min_distinct_prime_factors (m n : ℕ) :
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  p ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) ∧
  q ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) :=
sorry

end NUMINAMATH_CALUDE_min_distinct_prime_factors_l558_55854


namespace NUMINAMATH_CALUDE_prob_product_divisible_l558_55842

/-- Represents a standard 6-sided die --/
def StandardDie : Type := Fin 6

/-- The probability of rolling a number on a standard die --/
def prob_roll (n : Nat) : ℚ := if n ≥ 1 ∧ n ≤ 6 then 1 / 6 else 0

/-- The probability that a single die roll is not divisible by 2, 3, and 5 --/
def prob_not_divisible : ℚ := 5 / 18

/-- The number of dice rolled --/
def num_dice : Nat := 6

/-- The probability that the product of 6 dice rolls is divisible by 2, 3, or 5 --/
theorem prob_product_divisible :
  1 - prob_not_divisible ^ num_dice = 33996599 / 34012224 := by sorry

end NUMINAMATH_CALUDE_prob_product_divisible_l558_55842


namespace NUMINAMATH_CALUDE_area_of_square_B_l558_55899

/-- Given a square A with diagonal x and a square B with diagonal 3x, 
    the area of square B is 9x^2/2 -/
theorem area_of_square_B (x : ℝ) :
  let diag_A := x
  let diag_B := 3 * diag_A
  let area_B := (diag_B^2) / 4
  area_B = 9 * x^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_square_B_l558_55899


namespace NUMINAMATH_CALUDE_number_of_nephews_l558_55877

-- Define the price of a candy as the base unit
def candy_price : ℚ := 1

-- Define the prices of other items in terms of candy price
def orange_price : ℚ := 2 * candy_price
def cake_price : ℚ := 4 * candy_price
def chocolate_price : ℚ := 7 * candy_price
def book_price : ℚ := 14 * candy_price

-- Define the cost of one gift
def gift_cost : ℚ := candy_price + orange_price + cake_price + chocolate_price + book_price

-- Define the total number of each item if all money was spent on that item
def total_candies : ℕ := 224
def total_oranges : ℕ := 112
def total_cakes : ℕ := 56
def total_chocolates : ℕ := 32
def total_books : ℕ := 16

-- Theorem: The number of nephews is 8
theorem number_of_nephews : ℕ := by
  sorry

end NUMINAMATH_CALUDE_number_of_nephews_l558_55877


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l558_55811

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 1}

theorem set_intersection_theorem : M ∩ N = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l558_55811


namespace NUMINAMATH_CALUDE_antecedent_value_l558_55802

/-- Given a ratio of 4:6 and a consequent of 30, prove the antecedent is 20 -/
theorem antecedent_value (ratio_antecedent ratio_consequent consequent : ℕ) 
  (h1 : ratio_antecedent = 4)
  (h2 : ratio_consequent = 6)
  (h3 : consequent = 30) :
  ratio_antecedent * consequent / ratio_consequent = 20 := by
  sorry

end NUMINAMATH_CALUDE_antecedent_value_l558_55802


namespace NUMINAMATH_CALUDE_hair_extension_ratio_l558_55822

theorem hair_extension_ratio : 
  let initial_length : ℕ := 18
  let final_length : ℕ := 36
  (final_length : ℚ) / (initial_length : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_hair_extension_ratio_l558_55822


namespace NUMINAMATH_CALUDE_unique_a_value_l558_55831

/-- A quadratic function of the form y = 3x^2 + 2(a-1)x + b -/
def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * (a - 1) * x + b

/-- The derivative of the quadratic function -/
def quadratic_derivative (a : ℝ) (x : ℝ) : ℝ := 6 * x + 2 * (a - 1)

theorem unique_a_value (a b : ℝ) :
  (∀ x < 1, quadratic_derivative a x < 0) →
  (∀ x ≥ 1, quadratic_derivative a x ≥ 0) →
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_unique_a_value_l558_55831


namespace NUMINAMATH_CALUDE_solution_difference_l558_55868

-- Define the equation
def equation (x : ℝ) : Prop := (4 - x^2 / 3)^(1/3) = -2

-- Define the set of solutions
def solutions : Set ℝ := {x : ℝ | equation x}

-- Theorem statement
theorem solution_difference : 
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 12 :=
sorry

end NUMINAMATH_CALUDE_solution_difference_l558_55868


namespace NUMINAMATH_CALUDE_abc_sum_l558_55845

theorem abc_sum (a b c : ℚ) : 
  (a : ℚ) / 3 = (b : ℚ) / 5 ∧ (b : ℚ) / 5 = (c : ℚ) / 7 ∧ 
  3 * a + 2 * b - 4 * c = -9 → 
  a + b - c = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_l558_55845


namespace NUMINAMATH_CALUDE_flower_bouquet_row_length_l558_55856

theorem flower_bouquet_row_length 
  (num_students : ℕ) 
  (student_space : ℝ) 
  (gap_space : ℝ) 
  (h1 : num_students = 50) 
  (h2 : student_space = 0.4) 
  (h3 : gap_space = 0.5) : 
  num_students * student_space + (num_students - 1) * gap_space = 44.5 := by
  sorry

end NUMINAMATH_CALUDE_flower_bouquet_row_length_l558_55856


namespace NUMINAMATH_CALUDE_sum_distances_constant_l558_55818

/-- A regular tetrahedron in 3D space -/
structure RegularTetrahedron where
  -- Add necessary fields for a regular tetrahedron

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance from a point to a plane in 3D space -/
def distanceToPlane (p : Point3D) (plane : Set Point3D) : ℝ :=
  sorry

/-- Predicate to check if a point is inside a regular tetrahedron -/
def isInside (p : Point3D) (t : RegularTetrahedron) : Prop :=
  sorry

/-- The faces of a regular tetrahedron -/
def faces (t : RegularTetrahedron) : Finset (Set Point3D) :=
  sorry

/-- Theorem: The sum of distances from any point inside a regular tetrahedron to all its faces is constant -/
theorem sum_distances_constant (t : RegularTetrahedron) :
  ∃ c : ℝ, ∀ p : Point3D, isInside p t →
    (faces t).sum (λ face => distanceToPlane p face) = c :=
  sorry

end NUMINAMATH_CALUDE_sum_distances_constant_l558_55818


namespace NUMINAMATH_CALUDE_unique_prime_pair_l558_55869

def isPrime (n : ℕ) : Prop := sorry

def nthPrime (n : ℕ) : ℕ := sorry

theorem unique_prime_pair :
  ∀ a b : ℕ, 
    a > 0 → b > 0 → 
    a - b ≥ 2 → 
    (nthPrime a - nthPrime b) ∣ (2 * (a - b)) → 
    a = 4 ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l558_55869


namespace NUMINAMATH_CALUDE_checkerboard_matching_sum_l558_55801

/-- Row-wise numbering function -/
def f (i j : ℕ) : ℕ := 19 * (i - 1) + j

/-- Column-wise numbering function -/
def g (i j : ℕ) : ℕ := 15 * (j - 1) + i

/-- The set of pairs (i, j) where the numbers match in both systems -/
def matching_squares : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => f p.1 p.2 = g p.1 p.2)
    (Finset.product (Finset.range 15) (Finset.range 19))

theorem checkerboard_matching_sum :
  (matching_squares.sum fun p => f p.1 p.2) = 668 := by
  sorry


end NUMINAMATH_CALUDE_checkerboard_matching_sum_l558_55801


namespace NUMINAMATH_CALUDE_smallest_two_digit_k_for_45k_perfect_square_l558_55883

/-- A number is a perfect square if it has an integer square root -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A number is two-digit if it's between 10 and 99 inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_two_digit_k_for_45k_perfect_square :
  ∃ k : ℕ, is_two_digit k ∧ 
           is_perfect_square (45 * k) ∧ 
           (∀ m : ℕ, is_two_digit m → is_perfect_square (45 * m) → k ≤ m) ∧
           k = 20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_k_for_45k_perfect_square_l558_55883


namespace NUMINAMATH_CALUDE_candy_problem_l558_55882

theorem candy_problem (x : ℕ) : 
  (x % 12 = 0) →
  (∃ c : ℕ, c ≥ 1 ∧ c ≤ 3 ∧ 
   ((3 * x / 4) * 2 / 3 - 20 - c = 5)) →
  (x = 52 ∨ x = 56) := by
sorry

end NUMINAMATH_CALUDE_candy_problem_l558_55882


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_equals_10_l558_55829

theorem mean_equality_implies_y_equals_10 : ∀ y : ℝ, 
  (6 + 9 + 18) / 3 = (12 + y) / 2 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_equals_10_l558_55829


namespace NUMINAMATH_CALUDE_remainder_is_x_squared_l558_55808

-- Define the polynomials
def f (x : ℝ) := x^1010
def g (x : ℝ) := (x^2 + 1) * (x + 1) * (x - 1)

-- Define the remainder function
noncomputable def remainder (x : ℝ) := f x % g x

-- Theorem statement
theorem remainder_is_x_squared :
  ∀ x : ℝ, remainder x = x^2 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_is_x_squared_l558_55808


namespace NUMINAMATH_CALUDE_linear_function_properties_l558_55859

/-- A linear function y = kx + b where k < 0 and b > 0 -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  h₁ : k < 0
  h₂ : b > 0

/-- Properties of the linear function -/
theorem linear_function_properties (f : LinearFunction) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f.k * x₁ + f.b > f.k * x₂ + f.b) ∧ 
  (f.k * (-1) + f.b ≠ -2) ∧
  (f.k * 0 + f.b = f.b) ∧
  (∀ x : ℝ, x > -f.b / f.k → f.k * x + f.b < 0) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l558_55859


namespace NUMINAMATH_CALUDE_average_difference_l558_55819

theorem average_difference (a c x : ℝ) 
  (h1 : (a + x) / 2 = 40)
  (h2 : (x + c) / 2 = 60) : 
  c - a = 40 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l558_55819


namespace NUMINAMATH_CALUDE_soda_price_calculation_l558_55851

/-- The cost of a burger in cents -/
def burger_cost : ℕ := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℕ := sorry

/-- The cost of a side dish in cents -/
def side_dish_cost : ℕ := 30

theorem soda_price_calculation :
  (3 * burger_cost + 2 * soda_cost + side_dish_cost = 510) →
  (2 * burger_cost + 3 * soda_cost = 540) →
  soda_cost = 132 := by sorry

end NUMINAMATH_CALUDE_soda_price_calculation_l558_55851


namespace NUMINAMATH_CALUDE_water_bottles_problem_l558_55800

theorem water_bottles_problem (initial_bottles : ℕ) : 
  (3 * (initial_bottles - 3) = 21) → initial_bottles = 10 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_problem_l558_55800


namespace NUMINAMATH_CALUDE_bundle_promotion_better_l558_55838

-- Define the prices and discounts
def cellphone_price : ℝ := 800
def earbud_price : ℝ := 150
def case_price : ℝ := 40
def cellphone_discount : ℝ := 0.05
def earbud_discount : ℝ := 0.10
def bundle_discount : ℝ := 0.07
def loyalty_discount : ℝ := 0.03
def sales_tax : ℝ := 0.08

-- Define the total cost before promotions
def total_before_promotions : ℝ :=
  (2 * cellphone_price * (1 - cellphone_discount)) +
  (2 * earbud_price * (1 - earbud_discount)) +
  case_price

-- Define the cost after each promotion
def bundle_promotion_cost : ℝ :=
  total_before_promotions * (1 - bundle_discount)

def loyalty_promotion_cost : ℝ :=
  total_before_promotions * (1 - loyalty_discount)

-- Define the final costs including tax
def bundle_final_cost : ℝ :=
  bundle_promotion_cost * (1 + sales_tax)

def loyalty_final_cost : ℝ :=
  loyalty_promotion_cost * (1 + sales_tax)

-- Theorem statement
theorem bundle_promotion_better :
  bundle_final_cost < loyalty_final_cost :=
sorry

end NUMINAMATH_CALUDE_bundle_promotion_better_l558_55838


namespace NUMINAMATH_CALUDE_cos_double_angle_special_l558_55843

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and a point (3,4) on its terminal side, prove that cos 2α = -7/25 -/
theorem cos_double_angle_special (α : Real) 
  (h1 : ∃ (x y : Real), x = 3 ∧ y = 4 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
                        y = Real.sin α * Real.sqrt (x^2 + y^2)) : 
  Real.cos (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_l558_55843


namespace NUMINAMATH_CALUDE_number_of_office_workers_l558_55813

/-- Proves the number of office workers in company J --/
theorem number_of_office_workers :
  let factory_workers : ℕ := 15
  let factory_payroll : ℕ := 30000
  let office_payroll : ℕ := 75000
  let salary_difference : ℕ := 500
  let factory_avg_salary : ℕ := factory_payroll / factory_workers
  let office_avg_salary : ℕ := factory_avg_salary + salary_difference
  office_payroll / office_avg_salary = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_of_office_workers_l558_55813


namespace NUMINAMATH_CALUDE_parabola_point_and_line_intersection_l558_55889

/-- Parabola C defined by y^2 = 4x -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point P on the parabola C -/
def P : ℝ × ℝ := (1, 2)

/-- Point Q symmetrical to P across the x-axis -/
def Q : ℝ × ℝ := (P.1, -P.2)

/-- Origin O -/
def O : ℝ × ℝ := (0, 0)

/-- Area of triangle POQ -/
def area_POQ : ℝ := 2

/-- Slopes of lines PA and PB -/
def k₁ : ℝ := sorry
def k₂ : ℝ := sorry

/-- Fixed point that AB passes through -/
def fixed_point : ℝ × ℝ := (0, -2)

theorem parabola_point_and_line_intersection :
  (P ∈ C) ∧
  (P.2 > 0) ∧
  (area_POQ = 2) ∧
  (k₁ * k₂ = 4) →
  (P = (1, 2)) ∧
  (∀ (A B : ℝ × ℝ), A ∈ C → B ∈ C →
    (A.2 - P.2) / (A.1 - P.1) = k₁ →
    (B.2 - P.2) / (B.1 - P.1) = k₂ →
    ∃ (m b : ℝ), (A.2 = m * A.1 + b) ∧ (B.2 = m * B.1 + b) ∧
    (fixed_point.2 = m * fixed_point.1 + b)) :=
sorry

end NUMINAMATH_CALUDE_parabola_point_and_line_intersection_l558_55889


namespace NUMINAMATH_CALUDE_problem_statement_l558_55893

theorem problem_statement (a b : ℝ) (h1 : a * b = -3) (h2 : a + b = 2) :
  a^2 * b + a * b^2 = -6 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l558_55893


namespace NUMINAMATH_CALUDE_cafeteria_extra_fruits_l558_55886

/-- The number of extra fruits ordered by the cafeteria -/
def extra_fruits (total_fruits students max_per_student : ℕ) : ℕ :=
  total_fruits - (students * max_per_student)

/-- Theorem stating that the cafeteria ordered 43 extra fruits -/
theorem cafeteria_extra_fruits :
  extra_fruits 85 21 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_extra_fruits_l558_55886


namespace NUMINAMATH_CALUDE_min_distance_to_line_min_distance_achievable_l558_55896

/-- The minimum distance from the origin to a point on the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : ∀ m n : ℝ, m + n = 4 → Real.sqrt (m^2 + n^2) ≥ 2 * Real.sqrt 2 := by
  sorry

/-- The minimum distance 2√2 is achievable -/
theorem min_distance_achievable : ∃ m n : ℝ, m + n = 4 ∧ Real.sqrt (m^2 + n^2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_min_distance_achievable_l558_55896


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_2021_l558_55848

theorem tens_digit_of_13_pow_2021 : ∃ n : ℕ, 13^2021 ≡ 10 * n + 1 [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_2021_l558_55848


namespace NUMINAMATH_CALUDE_food_drive_total_cans_l558_55888

theorem food_drive_total_cans 
  (mark_cans jaydon_cans rachel_cans : ℕ) 
  (h1 : mark_cans = 4 * jaydon_cans)
  (h2 : jaydon_cans = 2 * rachel_cans + 5)
  (h3 : mark_cans = 100) : 
  mark_cans + jaydon_cans + rachel_cans = 135 := by
sorry


end NUMINAMATH_CALUDE_food_drive_total_cans_l558_55888


namespace NUMINAMATH_CALUDE_triangle_angle_impossibility_l558_55810

theorem triangle_angle_impossibility : ¬ ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- all angles are positive
  a + b + c = 180 ∧        -- sum of angles is 180 degrees
  a = 60 ∧                 -- one angle is 60 degrees
  b = 2 * a ∧              -- another angle is twice the first
  c ≠ 0                    -- the third angle is non-zero
  := by sorry

end NUMINAMATH_CALUDE_triangle_angle_impossibility_l558_55810


namespace NUMINAMATH_CALUDE_sum_of_coefficients_after_shift_l558_55870

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_left (f : QuadraticFunction) (units : ℝ) : QuadraticFunction :=
  QuadraticFunction.mk
    f.a
    (f.b + 2 * f.a * units)
    (f.a * units^2 + f.b * units + f.c)

/-- The original quadratic function y = 3x^2 + 2x - 5 -/
def original : QuadraticFunction :=
  QuadraticFunction.mk 3 2 (-5)

/-- The shifted quadratic function -/
def shifted : QuadraticFunction :=
  shift_left original 6

/-- Theorem stating that the sum of coefficients of the shifted function is 156 -/
theorem sum_of_coefficients_after_shift :
  shifted.a + shifted.b + shifted.c = 156 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_after_shift_l558_55870


namespace NUMINAMATH_CALUDE_integers_between_cubes_l558_55862

theorem integers_between_cubes : ∃ n : ℕ, n = (⌊(9.5 : ℝ)^3⌋ - ⌈(9.4 : ℝ)^3⌉ + 1) ∧ n = 27 := by
  sorry

end NUMINAMATH_CALUDE_integers_between_cubes_l558_55862


namespace NUMINAMATH_CALUDE_group_average_age_l558_55864

/-- Given a group of people, prove that their current average age is as calculated -/
theorem group_average_age 
  (n : ℕ) -- number of people in the group
  (youngest_age : ℕ) -- age of the youngest person
  (past_average : ℚ) -- average age when the youngest was born
  (h1 : n = 7) -- there are 7 people
  (h2 : youngest_age = 4) -- the youngest is 4 years old
  (h3 : past_average = 26) -- average age when youngest was born was 26
  : (n : ℚ) * ((n - 1 : ℚ) * past_average + n * (youngest_age : ℚ)) / n = 184 / 7 := by
  sorry

end NUMINAMATH_CALUDE_group_average_age_l558_55864


namespace NUMINAMATH_CALUDE_fraction_numerator_l558_55837

theorem fraction_numerator (y : ℝ) (n : ℝ) (h1 : y > 0) 
  (h2 : (2 * y) / 10 + (n / y) * y = 0.5 * y) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_l558_55837


namespace NUMINAMATH_CALUDE_fraction_inequality_not_sufficient_nor_necessary_sufficient_condition_implies_subset_l558_55855

-- Statement B
theorem fraction_inequality_not_sufficient_nor_necessary :
  ¬(∀ a b : ℝ, (1 / a > 1 / b → a < b) ∧ (a < b → 1 / a > 1 / b)) := by sorry

-- Statement C
theorem sufficient_condition_implies_subset (A B : Set α) :
  (∀ x, x ∈ A → x ∈ B) → A ⊆ B := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_not_sufficient_nor_necessary_sufficient_condition_implies_subset_l558_55855


namespace NUMINAMATH_CALUDE_g_has_four_zeros_l558_55807

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 1/x else Real.log x

noncomputable def g (x : ℝ) : ℝ :=
  f (f x + 2) + 2

theorem g_has_four_zeros :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0 ∧
  (∀ x : ℝ, g x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d) :=
sorry

end NUMINAMATH_CALUDE_g_has_four_zeros_l558_55807


namespace NUMINAMATH_CALUDE_cookies_for_students_minimum_recipes_needed_l558_55891

/-- Calculates the minimum number of full recipes needed to provide cookies for students -/
theorem cookies_for_students (original_students : ℕ) (increase_percent : ℕ) 
  (cookies_per_student : ℕ) (cookies_per_recipe : ℕ) : ℕ :=
  let new_students := original_students * (100 + increase_percent) / 100
  let total_cookies := new_students * cookies_per_student
  let recipes_needed := (total_cookies + cookies_per_recipe - 1) / cookies_per_recipe
  recipes_needed

/-- The minimum number of full recipes needed for the given conditions is 33 -/
theorem minimum_recipes_needed : 
  cookies_for_students 108 50 3 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_cookies_for_students_minimum_recipes_needed_l558_55891


namespace NUMINAMATH_CALUDE_total_guesses_l558_55894

def digits : List ℕ := [1, 1, 1, 1, 2, 2, 2, 2]

def valid_partition (p : List ℕ) : Prop :=
  p.length = 4 ∧ p.sum = 8 ∧ ∀ x ∈ p, 1 ≤ x ∧ x ≤ 3

def num_arrangements : ℕ := Nat.choose 8 4

def num_partitions : ℕ := 35

theorem total_guesses :
  num_arrangements * num_partitions = 2450 :=
sorry

end NUMINAMATH_CALUDE_total_guesses_l558_55894


namespace NUMINAMATH_CALUDE_range_of_a_l558_55816

theorem range_of_a (a : ℝ) : 
  (a - 3*3 < 4*a*3 + 2) → 
  (a - 3*0 < 4*a*0 + 2) → 
  (-1 < a ∧ a < 2) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l558_55816


namespace NUMINAMATH_CALUDE_students_playing_neither_l558_55874

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) 
  (h1 : total = 35)
  (h2 : football = 26)
  (h3 : tennis = 20)
  (h4 : both = 17) :
  total - (football + tennis - both) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_l558_55874


namespace NUMINAMATH_CALUDE_systematic_sampling_seat_number_l558_55890

/-- Systematic sampling function that returns the seat numbers in the sample -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) : List ℕ :=
  sorry

theorem systematic_sampling_seat_number
  (totalStudents : ℕ) (sampleSize : ℕ) (knownSeats : List ℕ) :
  totalStudents = 52 →
  sampleSize = 4 →
  knownSeats = [3, 29, 42] →
  let sample := systematicSample totalStudents sampleSize
  (∀ s ∈ knownSeats, s ∈ sample) →
  ∃ s ∈ sample, s = 16 ∧ s ∉ knownSeats :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_seat_number_l558_55890


namespace NUMINAMATH_CALUDE_inequality_proof_l558_55835

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  (a^4 + b^4)/(a^2 + b^2) + (b^4 + c^4)/(b^2 + c^2) + (c^4 + d^4)/(c^2 + d^2) + (d^4 + a^4)/(d^2 + a^2) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l558_55835


namespace NUMINAMATH_CALUDE_solution_for_a_l558_55828

theorem solution_for_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (eq1 : a + 2 / b = 17) (eq2 : b + 2 / a = 1 / 3) :
  a = 6 ∨ a = 17 := by
  sorry

end NUMINAMATH_CALUDE_solution_for_a_l558_55828


namespace NUMINAMATH_CALUDE_garden_area_theorem_l558_55884

/-- The area of a rectangle with a square cut out from each of two different corners -/
def garden_area (length width cut1_side cut2_side : ℝ) : ℝ :=
  length * width - cut1_side^2 - cut2_side^2

/-- Theorem: The area of a 20x18 rectangle with 4x4 and 2x2 squares cut out is 340 sq ft -/
theorem garden_area_theorem :
  garden_area 20 18 4 2 = 340 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_theorem_l558_55884


namespace NUMINAMATH_CALUDE_almonds_in_trail_mix_l558_55857

/-- Given the amount of walnuts and the total amount of nuts in a trail mix,
    calculate the amount of almonds added. -/
theorem almonds_in_trail_mix (walnuts total : ℚ) (h1 : walnuts = 0.25) (h2 : total = 0.5) :
  total - walnuts = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_almonds_in_trail_mix_l558_55857


namespace NUMINAMATH_CALUDE_annie_initial_money_l558_55880

/-- Annie's hamburger and milkshake purchase problem -/
theorem annie_initial_money :
  let hamburger_price : ℕ := 4
  let milkshake_price : ℕ := 3
  let hamburgers_bought : ℕ := 8
  let milkshakes_bought : ℕ := 6
  let money_left : ℕ := 70
  let initial_money : ℕ := hamburger_price * hamburgers_bought + milkshake_price * milkshakes_bought + money_left
  initial_money = 120 := by sorry

end NUMINAMATH_CALUDE_annie_initial_money_l558_55880


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l558_55875

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  x_tangent : Point
  has_y_tangent : Bool

/-- Calculate the length of the major axis of an ellipse -/
def majorAxisLength (e : Ellipse) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem: The length of the major axis of the given ellipse is 4 -/
theorem ellipse_major_axis_length :
  let e : Ellipse := {
    focus1 := { x := 4, y := 2 + 2 * Real.sqrt 2 },
    focus2 := { x := 4, y := 2 - 2 * Real.sqrt 2 },
    x_tangent := { x := 4, y := 0 },
    has_y_tangent := true
  }
  majorAxisLength e = 4 := by sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l558_55875


namespace NUMINAMATH_CALUDE_computer_pricing_l558_55825

/-- Represents the prices of computer components -/
structure Prices where
  basic_computer : ℝ
  printer : ℝ
  regular_monitor : ℝ

/-- Proves the correct prices given the problem conditions -/
theorem computer_pricing (prices : Prices) 
  (total_basic : prices.basic_computer + prices.printer + prices.regular_monitor = 3000)
  (enhanced_printer_ratio : prices.printer = (1/4) * (prices.basic_computer + 500 + prices.printer + prices.regular_monitor + 300)) :
  prices.printer = 950 ∧ prices.basic_computer + prices.regular_monitor = 2050 := by
  sorry


end NUMINAMATH_CALUDE_computer_pricing_l558_55825


namespace NUMINAMATH_CALUDE_photo_arrangements_l558_55836

def team_size : ℕ := 6

theorem photo_arrangements (captain_positions : ℕ) (ab_arrangements : ℕ) (remaining_arrangements : ℕ) :
  captain_positions = 2 →
  ab_arrangements = 2 →
  remaining_arrangements = 24 →
  captain_positions * ab_arrangements * remaining_arrangements = 96 :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l558_55836


namespace NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l558_55832

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x ≤ x - a}
def B : Set ℝ := {x | 4*x - x^2 - 3 ≥ 0}

-- State the theorem
theorem union_equality_iff_a_in_range : 
  ∀ a : ℝ, (A a ∪ B = B) ↔ a ∈ Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l558_55832


namespace NUMINAMATH_CALUDE_yogurt_combinations_l558_55858

theorem yogurt_combinations (num_flavors : Nat) (num_toppings : Nat) : 
  num_flavors = 6 → num_toppings = 8 → num_flavors * (num_toppings.choose 3) = 336 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l558_55858


namespace NUMINAMATH_CALUDE_triangle_isosceles_from_side_condition_l558_55826

theorem triangle_isosceles_from_side_condition (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_condition : a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b) = 0) :
  a = b ∨ b = c ∨ c = a := by
sorry


end NUMINAMATH_CALUDE_triangle_isosceles_from_side_condition_l558_55826


namespace NUMINAMATH_CALUDE_y_range_l558_55844

theorem y_range (a b y : ℝ) (h1 : a + b = 2) (h2 : b ≤ 2) (h3 : y - a^2 - 2*a + 2 = 0) :
  y ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_y_range_l558_55844


namespace NUMINAMATH_CALUDE_randy_blocks_left_l558_55853

/-- The number of blocks Randy has left after constructions -/
def blocks_left (initial : ℕ) (tower : ℕ) (house : ℕ) : ℕ :=
  let remaining_after_tower := initial - tower
  let bridge := remaining_after_tower / 2
  let remaining_after_bridge := remaining_after_tower - bridge
  remaining_after_bridge - house

/-- Theorem stating that Randy has 19 blocks left after constructions -/
theorem randy_blocks_left :
  blocks_left 78 19 11 = 19 := by sorry

end NUMINAMATH_CALUDE_randy_blocks_left_l558_55853


namespace NUMINAMATH_CALUDE_pen_cost_l558_55865

theorem pen_cost (pen_price : ℝ) (briefcase_price : ℝ) : 
  briefcase_price = 5 * pen_price →
  pen_price + briefcase_price = 24 →
  pen_price = 4 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_l558_55865


namespace NUMINAMATH_CALUDE_comic_books_calculation_l558_55878

theorem comic_books_calculation (initial : ℕ) (bought : ℕ) : 
  initial = 14 → bought = 6 → initial / 2 + bought = 13 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_calculation_l558_55878


namespace NUMINAMATH_CALUDE_custom_op_properties_l558_55809

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 2 * a * b

-- State the theorem
theorem custom_op_properties :
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → custom_op a b = 2 * a * b) →
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → custom_op a b = custom_op b a) ∧
  (∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 → custom_op a (custom_op b c) = custom_op (custom_op a b) c) ∧
  (∀ a : ℝ, a ≠ 0 → custom_op a (1/2) = a) ∧
  (∀ a : ℝ, a ≠ 0 → custom_op a (1/(2*a)) ≠ 1/2) :=
by sorry


end NUMINAMATH_CALUDE_custom_op_properties_l558_55809


namespace NUMINAMATH_CALUDE_stratified_sample_size_l558_55840

/-- Represents a workshop with its production quantity -/
structure Workshop where
  quantity : ℕ

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  workshops : List Workshop
  sampleSize : ℕ
  sampledFromC : ℕ

/-- Calculates the total production quantity across all workshops -/
def totalQuantity (s : StratifiedSample) : ℕ :=
  s.workshops.foldl (fun acc w => acc + w.quantity) 0

/-- Theorem stating the relationship between sample size and workshop quantities -/
theorem stratified_sample_size 
  (s : StratifiedSample)
  (hWorkshops : s.workshops = [⟨600⟩, ⟨400⟩, ⟨300⟩])
  (hSampledC : s.sampledFromC = 6) :
  s.sampleSize = 26 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l558_55840


namespace NUMINAMATH_CALUDE_inequality_proof_l558_55805

theorem inequality_proof (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l558_55805


namespace NUMINAMATH_CALUDE_tangent_lines_range_l558_55871

/-- The range of k values for which two tangent lines exist from (1, 2) to the circle -/
theorem tangent_lines_range (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + k*x + 2*y + k^2 - 15 = 0) ∧ 
  ((1:ℝ)^2 + 2^2 + k*1 + 2*2 + k^2 - 15 > 0) ↔ 
  (k > 2 ∧ k < 8/3 * Real.sqrt 3) ∨ (k > -8/3 * Real.sqrt 3 ∧ k < -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_range_l558_55871


namespace NUMINAMATH_CALUDE_valid_attachments_count_l558_55833

/-- Represents a square in our figure -/
structure Square

/-- Represents the cross-shaped figure -/
structure CrossFigure where
  center : Square
  extensions : Fin 4 → Square

/-- Represents a position where an extra square can be attached -/
inductive AttachmentPosition
  | TopOfExtension (i : Fin 4)
  | Other

/-- Represents the resulting figure after attaching an extra square -/
structure ResultingFigure where
  base : CrossFigure
  extraSquare : Square
  position : AttachmentPosition

/-- Predicate to check if a resulting figure can be folded into a topless square pyramid -/
def canFoldIntoPyramid (fig : ResultingFigure) : Prop :=
  match fig.position with
  | AttachmentPosition.TopOfExtension _ => True
  | AttachmentPosition.Other => False

/-- The main theorem to prove -/
theorem valid_attachments_count :
  ∃ (validPositions : Finset AttachmentPosition),
    (∀ pos, pos ∈ validPositions ↔ ∃ fig : ResultingFigure, fig.position = pos ∧ canFoldIntoPyramid fig) ∧
    Finset.card validPositions = 4 := by
  sorry

end NUMINAMATH_CALUDE_valid_attachments_count_l558_55833


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l558_55812

theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) :
  a = (1, 2) →
  b = (-1, x) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l558_55812


namespace NUMINAMATH_CALUDE_cos_495_degrees_l558_55895

theorem cos_495_degrees : Real.cos (495 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_495_degrees_l558_55895


namespace NUMINAMATH_CALUDE_tan_identities_l558_55804

theorem tan_identities (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  ((Real.sin (2*α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_identities_l558_55804


namespace NUMINAMATH_CALUDE_solution_of_system_l558_55827

def augmented_matrix : Matrix (Fin 2) (Fin 3) ℝ := !![1, -1, 1; 1, 1, 3]

theorem solution_of_system (x y : ℝ) : 
  x = 2 ∧ y = 1 ↔ 
  (augmented_matrix 0 0 * x + augmented_matrix 0 1 * y = augmented_matrix 0 2) ∧
  (augmented_matrix 1 0 * x + augmented_matrix 1 1 * y = augmented_matrix 1 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l558_55827


namespace NUMINAMATH_CALUDE_milk_replacement_l558_55830

theorem milk_replacement (x : ℝ) : 
  x > 0 ∧ x < 30 →
  30 - x - (x - x^2/30) = 14.7 →
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_milk_replacement_l558_55830


namespace NUMINAMATH_CALUDE_volume_common_part_equal_cones_l558_55852

/-- Given two equal cones with common height and parallel bases, 
    the volume of their common part is 1/4 of the volume of each cone. -/
theorem volume_common_part_equal_cones (R h : ℝ) (hR : R > 0) (hh : h > 0) : 
  let V_cone := (1/3) * π * R^2 * h
  let V_common := (1/12) * π * R^2 * h
  V_common = (1/4) * V_cone := by
  sorry

end NUMINAMATH_CALUDE_volume_common_part_equal_cones_l558_55852


namespace NUMINAMATH_CALUDE_fraction_equality_l558_55861

theorem fraction_equality (m n : ℚ) (h : 2/3 * m = 5/6 * n) : (m - n) / n = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l558_55861


namespace NUMINAMATH_CALUDE_nowhere_negative_polynomial_is_sum_of_squares_l558_55872

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- A polynomial is nowhere negative if it's non-negative for all real inputs -/
def NowhereNegative (p : RealPolynomial) : Prop :=
  ∀ x : ℝ, p x ≥ 0

/-- Theorem: Any nowhere negative real polynomial can be expressed as a sum of squares -/
theorem nowhere_negative_polynomial_is_sum_of_squares :
  ∀ p : RealPolynomial, NowhereNegative p →
  ∃ q r s : RealPolynomial, ∀ x : ℝ, p x = (q x)^2 * ((r x)^2 + (s x)^2) :=
sorry

end NUMINAMATH_CALUDE_nowhere_negative_polynomial_is_sum_of_squares_l558_55872


namespace NUMINAMATH_CALUDE_product_and_remainder_problem_l558_55814

theorem product_and_remainder_problem :
  ∃ (a b c d : ℤ),
    d = a * b * c ∧
    1 < a ∧ a < b ∧ b < c ∧
    233 % d = 79 ∧
    a + c = 13 := by
  sorry

end NUMINAMATH_CALUDE_product_and_remainder_problem_l558_55814


namespace NUMINAMATH_CALUDE_min_value_problem_l558_55820

theorem min_value_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3*y = 5*x*y) :
  3*x + 4*y ≥ 5 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 3*y = 5*x*y ∧ 3*x + 4*y = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l558_55820


namespace NUMINAMATH_CALUDE_sum_of_cubes_l558_55834

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l558_55834


namespace NUMINAMATH_CALUDE_integer_fraction_problem_l558_55898

theorem integer_fraction_problem (a b : ℕ+) :
  (a.val > 0) →
  (b.val > 0) →
  (∃ k : ℤ, (a.val^3 * b.val - 1) = k * (a.val + 1)) →
  (∃ m : ℤ, (b.val^3 * a.val + 1) = m * (b.val - 1)) →
  ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) := by
sorry

end NUMINAMATH_CALUDE_integer_fraction_problem_l558_55898


namespace NUMINAMATH_CALUDE_meaningful_fraction_l558_55873

theorem meaningful_fraction (x : ℝ) : (x - 5)⁻¹ ≠ 0 ↔ x ≠ 5 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l558_55873


namespace NUMINAMATH_CALUDE_johns_age_l558_55803

theorem johns_age (j d : ℕ) 
  (h1 : j + 28 = d)
  (h2 : j + d = 76)
  (h3 : d = 2 * (j - 4)) :
  j = 24 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l558_55803


namespace NUMINAMATH_CALUDE_milk_water_ratio_after_addition_l558_55815

def initial_volume : ℚ := 45
def initial_milk_ratio : ℚ := 4
def initial_water_ratio : ℚ := 1
def added_water : ℚ := 23

theorem milk_water_ratio_after_addition :
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let final_water := initial_water + added_water
  (initial_milk : ℚ) / final_water = 9 / 8 := by sorry

end NUMINAMATH_CALUDE_milk_water_ratio_after_addition_l558_55815


namespace NUMINAMATH_CALUDE_allison_wins_prob_l558_55821

/-- Represents a die with a fixed number of faces -/
structure Die where
  faces : List Nat

/-- Allison's die always shows 5 -/
def allison_die : Die := ⟨[5, 5, 5, 5, 5, 5]⟩

/-- Brian's die has faces numbered 1, 2, 3, 4, 4, 5, 5, and 6 -/
def brian_die : Die := ⟨[1, 2, 3, 4, 4, 5, 5, 6]⟩

/-- Noah's die has faces numbered 2, 2, 6, 6, 3, 3, 7, and 7 -/
def noah_die : Die := ⟨[2, 2, 6, 6, 3, 3, 7, 7]⟩

/-- Calculate the probability of rolling less than a given number on a die -/
def prob_less_than (d : Die) (n : Nat) : Rat :=
  (d.faces.filter (· < n)).length / d.faces.length

/-- The probability that Allison's roll is greater than both Brian's and Noah's -/
theorem allison_wins_prob : 
  prob_less_than brian_die 5 * prob_less_than noah_die 5 = 5 / 16 := by
  sorry


end NUMINAMATH_CALUDE_allison_wins_prob_l558_55821


namespace NUMINAMATH_CALUDE_min_segments_for_perimeter_is_three_l558_55897

/-- Represents an octagon formed by cutting a smaller rectangle from a larger rectangle -/
structure CutOutOctagon where
  /-- The length of the larger rectangle -/
  outer_length : ℝ
  /-- The width of the larger rectangle -/
  outer_width : ℝ
  /-- The length of the smaller cut-out rectangle -/
  inner_length : ℝ
  /-- The width of the smaller cut-out rectangle -/
  inner_width : ℝ
  /-- Ensures the inner rectangle fits inside the outer rectangle -/
  h_inner_fits : inner_length < outer_length ∧ inner_width < outer_width

/-- The minimum number of line segment lengths required to calculate the perimeter of a CutOutOctagon -/
def min_segments_for_perimeter (oct : CutOutOctagon) : ℕ := 3

/-- Theorem stating that the minimum number of line segment lengths required to calculate
    the perimeter of a CutOutOctagon is always 3 -/
theorem min_segments_for_perimeter_is_three (oct : CutOutOctagon) :
  min_segments_for_perimeter oct = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_segments_for_perimeter_is_three_l558_55897


namespace NUMINAMATH_CALUDE_salary_calculation_l558_55839

def initial_salary : ℚ := 3000
def raise_percentage : ℚ := 10 / 100
def pay_cut_percentage : ℚ := 15 / 100
def bonus : ℚ := 500

def final_salary : ℚ := 
  (initial_salary * (1 + raise_percentage) * (1 - pay_cut_percentage)) + bonus

theorem salary_calculation : final_salary = 3305 := by sorry

end NUMINAMATH_CALUDE_salary_calculation_l558_55839


namespace NUMINAMATH_CALUDE_distance_to_origin_l558_55887

open Complex

theorem distance_to_origin : let z : ℂ := (1 - I) * (1 + I) / I
  abs z = 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_origin_l558_55887


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_one_range_of_m_for_solution_l558_55876

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - |x - 1|

-- Theorem for the solution set of f(x) > 1
theorem solution_set_f_greater_than_one :
  {x : ℝ | f x > 1} = {x : ℝ | x > 0} := by sorry

-- Theorem for the range of m
theorem range_of_m_for_solution (m : ℝ) :
  (∃ x : ℝ, f x + 4 ≥ |1 - 2*m|) ↔ m ∈ Set.Icc (-3) 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_one_range_of_m_for_solution_l558_55876


namespace NUMINAMATH_CALUDE_factor_implies_root_l558_55863

theorem factor_implies_root (a : ℝ) : 
  (∀ t : ℝ, (2*t + 1) ∣ (4*t^2 + 12*t + a)) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_root_l558_55863


namespace NUMINAMATH_CALUDE_real_roots_iff_k_leq_5_root_one_implies_k_values_l558_55881

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := x^2 - 2*(k-3)*x + k^2 - 4*k - 1

-- Theorem 1: The equation has real roots iff k ≤ 5
theorem real_roots_iff_k_leq_5 (k : ℝ) : 
  (∃ x : ℝ, quadratic k x = 0) ↔ k ≤ 5 := by sorry

-- Theorem 2: If 1 is a root, then k = 3 + √3 or k = 3 - √3
theorem root_one_implies_k_values (k : ℝ) : 
  quadratic k 1 = 0 → k = 3 + Real.sqrt 3 ∨ k = 3 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_real_roots_iff_k_leq_5_root_one_implies_k_values_l558_55881


namespace NUMINAMATH_CALUDE_division_of_fractions_l558_55879

theorem division_of_fractions : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l558_55879


namespace NUMINAMATH_CALUDE_trig_expression_equals_four_l558_55841

theorem trig_expression_equals_four :
  1 / Real.cos (80 * π / 180) - Real.sqrt 3 / Real.cos (10 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_four_l558_55841


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l558_55866

/-- The speed of the man in still water -/
def man_speed : ℝ := 7

/-- The speed of the stream -/
def stream_speed : ℝ := 1

/-- The distance traveled downstream -/
def downstream_distance : ℝ := 40

/-- The distance traveled upstream -/
def upstream_distance : ℝ := 30

/-- The time taken for each journey -/
def journey_time : ℝ := 5

theorem man_speed_in_still_water :
  (downstream_distance / journey_time = man_speed + stream_speed) ∧
  (upstream_distance / journey_time = man_speed - stream_speed) →
  man_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_in_still_water_l558_55866


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l558_55817

theorem contrapositive_equivalence (a b : ℤ) :
  ((Odd a ∧ Odd b) → Even (a + b)) ↔
  (¬Even (a + b) → ¬(Odd a ∧ Odd b)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l558_55817


namespace NUMINAMATH_CALUDE_tim_took_eleven_rulers_l558_55850

/-- The number of rulers initially in the drawer -/
def initial_rulers : ℕ := 14

/-- The number of rulers left in the drawer after Tim took some out -/
def remaining_rulers : ℕ := 3

/-- The number of rulers Tim took out -/
def rulers_taken : ℕ := initial_rulers - remaining_rulers

theorem tim_took_eleven_rulers : rulers_taken = 11 := by
  sorry

end NUMINAMATH_CALUDE_tim_took_eleven_rulers_l558_55850


namespace NUMINAMATH_CALUDE_sine_sum_simplification_l558_55892

theorem sine_sum_simplification (x y : ℝ) :
  Real.sin (x - y) * Real.cos y + Real.cos (x - y) * Real.sin y = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_sine_sum_simplification_l558_55892


namespace NUMINAMATH_CALUDE_tan_half_angle_l558_55849

theorem tan_half_angle (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : 3 * Real.sin α + 2 * Real.cos α = 2) : Real.tan (α / 2) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_angle_l558_55849


namespace NUMINAMATH_CALUDE_sphere_expansion_l558_55806

theorem sphere_expansion (r : ℝ) (h : r > 0) :
  let V₁ := (4 / 3) * Real.pi * r^3
  let S₁ := 4 * Real.pi * r^2
  let V₂ := 8 * V₁
  let S₂ := 4 * Real.pi * (2*r)^2
  S₂ = 4 * S₁ :=
by sorry

end NUMINAMATH_CALUDE_sphere_expansion_l558_55806
