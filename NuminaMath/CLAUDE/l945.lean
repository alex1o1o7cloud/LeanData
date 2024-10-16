import Mathlib

namespace NUMINAMATH_CALUDE_sphere_identical_views_other_bodies_different_views_l945_94556

-- Define the geometric bodies
inductive GeometricBody
  | Cylinder
  | Cone
  | Sphere
  | TriangularPyramid

-- Define a function to check if a geometric body has identical views
def hasIdenticalViews (body : GeometricBody) : Prop :=
  match body with
  | GeometricBody.Sphere => true
  | _ => false

-- Theorem stating that only a sphere has identical views
theorem sphere_identical_views :
  ∀ (body : GeometricBody),
    hasIdenticalViews body ↔ body = GeometricBody.Sphere :=
by sorry

-- Prove that other geometric bodies do not have identical views
theorem other_bodies_different_views :
  ¬(hasIdenticalViews GeometricBody.Cylinder) ∧
  ¬(hasIdenticalViews GeometricBody.Cone) ∧
  ¬(hasIdenticalViews GeometricBody.TriangularPyramid) :=
by sorry

end NUMINAMATH_CALUDE_sphere_identical_views_other_bodies_different_views_l945_94556


namespace NUMINAMATH_CALUDE_equation_solution_l945_94572

theorem equation_solution (p q : ℝ) (h : p^2*q = p*q + p^2) : 
  p = 0 ∨ (q ≠ 1 ∧ p = q / (q - 1)) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l945_94572


namespace NUMINAMATH_CALUDE_power_of_128_fourth_sevenths_l945_94549

theorem power_of_128_fourth_sevenths (h : 128 = 2^7) : (128 : ℝ)^(4/7) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_128_fourth_sevenths_l945_94549


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l945_94561

-- Define the quadratic inequality and its solution set
def quadratic_inequality (a b : ℝ) (x : ℝ) : Prop := a * x^2 + x + b > 0
def solution_set (a b : ℝ) : Set ℝ := {x | x < -2 ∨ x > 1}

-- Define the second inequality
def second_inequality (a b c : ℝ) (x : ℝ) : Prop := a * x^2 - (c + b) * x + b * c < 0

-- Theorem statement
theorem quadratic_inequality_theorem :
  ∀ a b : ℝ, (∀ x : ℝ, quadratic_inequality a b x ↔ x ∈ solution_set a b) →
  (a = 1 ∧ b = -2) ∧
  (∀ c : ℝ, 
    (c = -2 → ∀ x : ℝ, ¬(second_inequality a b c x)) ∧
    (c > -2 → ∀ x : ℝ, second_inequality a b c x ↔ -2 < x ∧ x < c) ∧
    (c < -2 → ∀ x : ℝ, second_inequality a b c x ↔ c < x ∧ x < -2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l945_94561


namespace NUMINAMATH_CALUDE_cars_already_parked_equals_62_l945_94590

/-- Represents a multi-story parking lot -/
structure ParkingLot where
  totalCapacity : ℕ
  levels : ℕ
  additionalCapacity : ℕ

/-- The number of cars already parked on one level -/
def carsAlreadyParked (p : ParkingLot) : ℕ :=
  p.totalCapacity / p.levels - p.additionalCapacity

/-- Theorem stating the number of cars already parked on one level -/
theorem cars_already_parked_equals_62 (p : ParkingLot) 
    (h1 : p.totalCapacity = 425)
    (h2 : p.levels = 5)
    (h3 : p.additionalCapacity = 62) :
    carsAlreadyParked p = 62 := by
  sorry

#eval carsAlreadyParked { totalCapacity := 425, levels := 5, additionalCapacity := 62 }

end NUMINAMATH_CALUDE_cars_already_parked_equals_62_l945_94590


namespace NUMINAMATH_CALUDE_consecutive_binomial_ratio_l945_94555

theorem consecutive_binomial_ratio (n k : ℕ) : 
  n > k → 
  (n.choose k : ℚ) / (n.choose (k + 1)) = 1 / 3 →
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2)) = 1 / 2 →
  n + k = 13 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_binomial_ratio_l945_94555


namespace NUMINAMATH_CALUDE_same_terminal_side_l945_94514

/-- Proves that given an angle of -3π/10 radians, 306° has the same terminal side when converted to degrees -/
theorem same_terminal_side : ∃ (β : ℝ), β = 306 ∧ ∃ (k : ℤ), β = (-3/10 * π) * (180/π) + 360 * k :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_l945_94514


namespace NUMINAMATH_CALUDE_cars_for_sale_l945_94563

theorem cars_for_sale 
  (num_salespeople : ℕ)
  (cars_per_salesperson_per_month : ℕ)
  (num_months : ℕ)
  (h1 : num_salespeople = 10)
  (h2 : cars_per_salesperson_per_month = 10)
  (h3 : num_months = 5) :
  num_salespeople * cars_per_salesperson_per_month * num_months = 500 := by
  sorry

end NUMINAMATH_CALUDE_cars_for_sale_l945_94563


namespace NUMINAMATH_CALUDE_sequence_periodicity_l945_94593

def isEventuallyPeriodic (a : ℕ → ℕ) : Prop :=
  ∃ (n k : ℕ), k > 0 ∧ ∀ m, m ≥ n → a (m + k) = a m

theorem sequence_periodicity (a : ℕ → ℕ) 
    (h : ∀ n : ℕ, a n * a (n + 1) = a (n + 2) * a (n + 3)) :
    isEventuallyPeriodic a := by
  sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l945_94593


namespace NUMINAMATH_CALUDE_vehicle_original_value_l945_94543

/-- The original value of a vehicle given its insurance details -/
def original_value (insured_fraction : ℚ) (premium : ℚ) (premium_rate : ℚ) : ℚ :=
  premium / (premium_rate / 100) / insured_fraction

/-- Theorem stating the original value of the vehicle -/
theorem vehicle_original_value :
  original_value (4/5) 910 1.3 = 87500 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_original_value_l945_94543


namespace NUMINAMATH_CALUDE_hex_addition_l945_94542

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C | D | E | F

/-- Converts a HexDigit to its decimal value --/
def hexToDecimal (h : HexDigit) : Nat :=
  match h with
  | HexDigit.D0 => 0
  | HexDigit.D1 => 1
  | HexDigit.D2 => 2
  | HexDigit.D3 => 3
  | HexDigit.D4 => 4
  | HexDigit.D5 => 5
  | HexDigit.D6 => 6
  | HexDigit.D7 => 7
  | HexDigit.D8 => 8
  | HexDigit.D9 => 9
  | HexDigit.A => 10
  | HexDigit.B => 11
  | HexDigit.C => 12
  | HexDigit.D => 13
  | HexDigit.E => 14
  | HexDigit.F => 15

/-- Converts a list of HexDigits to its decimal value --/
def hexListToDecimal (l : List HexDigit) : Nat :=
  l.foldr (fun d acc => 16 * acc + hexToDecimal d) 0

/-- Theorem: The sum of 7A3₁₆ and 1F4₁₆ is equal to 997₁₆ --/
theorem hex_addition : 
  let a := [HexDigit.D7, HexDigit.A, HexDigit.D3]
  let b := [HexDigit.D1, HexDigit.F, HexDigit.D4]
  let result := [HexDigit.D9, HexDigit.D9, HexDigit.D7]
  hexListToDecimal a + hexListToDecimal b = hexListToDecimal result := by
  sorry


end NUMINAMATH_CALUDE_hex_addition_l945_94542


namespace NUMINAMATH_CALUDE_object_with_22_opposite_directions_is_clock_l945_94560

/-- An object with hands that can show opposite directions -/
structure ObjectWithHands :=
  (oppositeDirectionsPerDay : ℕ)

/-- Definition of a clock based on its behavior -/
def isClock (obj : ObjectWithHands) : Prop :=
  obj.oppositeDirectionsPerDay = 22

/-- Theorem stating that an object with hands showing opposite directions 22 times a day is a clock -/
theorem object_with_22_opposite_directions_is_clock (obj : ObjectWithHands) :
  obj.oppositeDirectionsPerDay = 22 → isClock obj :=
by
  sorry

#check object_with_22_opposite_directions_is_clock

end NUMINAMATH_CALUDE_object_with_22_opposite_directions_is_clock_l945_94560


namespace NUMINAMATH_CALUDE_fourth_rectangle_perimeter_is_10_l945_94568

/-- The perimeter of the fourth rectangle in a large rectangle cut into four smaller ones --/
def fourth_rectangle_perimeter (p1 p2 p3 : ℕ) : ℕ :=
  p1 + p2 - p3

/-- Theorem stating that the perimeter of the fourth rectangle is 10 --/
theorem fourth_rectangle_perimeter_is_10 :
  fourth_rectangle_perimeter 16 18 24 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_rectangle_perimeter_is_10_l945_94568


namespace NUMINAMATH_CALUDE_square_plus_product_plus_square_nonnegative_l945_94596

theorem square_plus_product_plus_square_nonnegative (x y : ℝ) :
  x^2 + x*y + y^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_product_plus_square_nonnegative_l945_94596


namespace NUMINAMATH_CALUDE_sum_of_digits_eight_to_hundred_l945_94526

theorem sum_of_digits_eight_to_hundred (n : ℕ) (h : n = 8^100) : 
  (n % 100 / 10 + n % 10) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_eight_to_hundred_l945_94526


namespace NUMINAMATH_CALUDE_jordan_terry_income_difference_l945_94562

/-- Calculates the difference in weekly income between two people given their daily incomes and the number of days worked per week. -/
def weekly_income_difference (terry_daily_income jordan_daily_income days_per_week : ℕ) : ℕ :=
  (jordan_daily_income * days_per_week) - (terry_daily_income * days_per_week)

/-- Proves that the difference in weekly income between Jordan and Terry is $42. -/
theorem jordan_terry_income_difference :
  weekly_income_difference 24 30 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_jordan_terry_income_difference_l945_94562


namespace NUMINAMATH_CALUDE_quadratic_discriminant_zero_not_harmonic_l945_94505

/-- The discriminant of the quadratic equation 3ax^2 + bx + 2c = 0 is zero -/
def discriminant_zero (a b c : ℝ) : Prop :=
  b^2 = 24*a*c

/-- a, b, and c form a harmonic progression -/
def harmonic_progression (a b c : ℝ) : Prop :=
  2/b = 1/a + 1/c

theorem quadratic_discriminant_zero_not_harmonic
  (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  discriminant_zero a b c → ¬harmonic_progression a b c :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_zero_not_harmonic_l945_94505


namespace NUMINAMATH_CALUDE_find_A_value_l945_94569

theorem find_A_value (A : Nat) : A < 10 → (691 - (A * 100 + 87) = 4) → A = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_A_value_l945_94569


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l945_94516

/-- The solution set of the quadratic inequality ax^2 + 3x - 2 < 0 --/
def SolutionSet (a b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

/-- The quadratic function ax^2 + 3x - 2 --/
def QuadraticFunction (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 3*x - 2

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, QuadraticFunction a x < 0 ↔ x ∈ SolutionSet a b) →
  a = -1 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l945_94516


namespace NUMINAMATH_CALUDE_divide_by_eight_l945_94521

theorem divide_by_eight (x y z : ℕ) (h1 : x > 0) (h2 : x = 11 * y + 4) 
  (h3 : 2 * x = 3 * y * z + 3) (h4 : 13 * y - x = 1) : z = 8 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_eight_l945_94521


namespace NUMINAMATH_CALUDE_stellas_dolls_count_l945_94591

theorem stellas_dolls_count : 
  ∀ (num_dolls : ℕ),
  (num_dolls : ℝ) * 5 + 2 * 15 + 5 * 4 - 40 = 25 →
  num_dolls = 3 := by
sorry

end NUMINAMATH_CALUDE_stellas_dolls_count_l945_94591


namespace NUMINAMATH_CALUDE_smallest_integers_difference_smallest_integers_difference_exists_l945_94585

theorem smallest_integers_difference : ℕ → Prop :=
  fun n =>
    (∃ a b : ℕ,
      (a > 1 ∧ b > 1 ∧ a < b) ∧
      (∀ k : ℕ, 3 ≤ k → k ≤ 12 → a % k = 1 ∧ b % k = 1) ∧
      (∀ x : ℕ, x > 1 ∧ x < a → ∃ k : ℕ, 3 ≤ k ∧ k ≤ 12 ∧ x % k ≠ 1) ∧
      (b - a = n)) →
    n = 13860

theorem smallest_integers_difference_exists : ∃ n : ℕ, smallest_integers_difference n :=
  sorry

end NUMINAMATH_CALUDE_smallest_integers_difference_smallest_integers_difference_exists_l945_94585


namespace NUMINAMATH_CALUDE_expected_net_profit_l945_94584

/-- The expected value of net profit from selling one electronic product -/
theorem expected_net_profit (purchase_price : ℝ) (pass_rate : ℝ) (profit_qualified : ℝ) (loss_defective : ℝ)
  (h1 : purchase_price = 10)
  (h2 : pass_rate = 0.95)
  (h3 : profit_qualified = 2)
  (h4 : loss_defective = 10) :
  profit_qualified * pass_rate + (-loss_defective) * (1 - pass_rate) = 1.4 := by
sorry

end NUMINAMATH_CALUDE_expected_net_profit_l945_94584


namespace NUMINAMATH_CALUDE_inequality_proof_l945_94520

theorem inequality_proof (a b c d : ℝ) 
  (h1 : b + Real.sin a > d + Real.sin c) 
  (h2 : a + Real.sin b > c + Real.sin d) : 
  a + b > c + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l945_94520


namespace NUMINAMATH_CALUDE_solution_set_x_squared_minus_one_l945_94597

theorem solution_set_x_squared_minus_one (x : ℝ) : 
  {x : ℝ | x^2 - 1 = 0} = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_minus_one_l945_94597


namespace NUMINAMATH_CALUDE_triangle_perimeter_l945_94577

/-- Given a triangle with sides satisfying specific conditions, prove its perimeter. -/
theorem triangle_perimeter (a b : ℝ) : 
  let side1 := a + b
  let side2 := side1 + (a + 2)
  let side3 := side2 - 3
  side1 + side2 + side3 = 5*a + 3*b + 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l945_94577


namespace NUMINAMATH_CALUDE_cost_of_soft_drink_l945_94554

/-- The cost of a can of soft drink given the following conditions:
  * 5 boxes of pizza cost $50
  * 6 hamburgers cost $18
  * 20 cans of soft drinks were bought
  * Total spent is $106
-/
theorem cost_of_soft_drink :
  let pizza_cost : ℚ := 50
  let hamburger_cost : ℚ := 18
  let total_cans : ℕ := 20
  let total_spent : ℚ := 106
  let soft_drink_cost : ℚ := (total_spent - pizza_cost - hamburger_cost) / total_cans
  soft_drink_cost = 19/10 := by sorry

end NUMINAMATH_CALUDE_cost_of_soft_drink_l945_94554


namespace NUMINAMATH_CALUDE_ending_number_divisible_by_three_eleven_numbers_divisible_by_three_l945_94511

theorem ending_number_divisible_by_three (start : Nat) (count : Nat) (divisor : Nat) : Nat :=
  let first_divisible := start + (divisor - start % divisor) % divisor
  first_divisible + (count - 1) * divisor

theorem eleven_numbers_divisible_by_three : 
  ending_number_divisible_by_three 10 11 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ending_number_divisible_by_three_eleven_numbers_divisible_by_three_l945_94511


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l945_94550

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 15) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) ≤ Real.sqrt 48 ∧
  ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 15 ∧
    Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) = Real.sqrt 48 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l945_94550


namespace NUMINAMATH_CALUDE_divisibility_by_three_l945_94538

theorem divisibility_by_three (a b : ℤ) (h : 3 ∣ (a * b)) :
  ¬(¬(3 ∣ a) ∧ ¬(3 ∣ b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l945_94538


namespace NUMINAMATH_CALUDE_calculate_expression_l945_94578

theorem calculate_expression : 500 * 996 * 0.0996 * 20 + 5000 = 997016 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l945_94578


namespace NUMINAMATH_CALUDE_freds_marbles_l945_94535

/-- Given Fred's marble collection, prove the number of dark blue marbles. -/
theorem freds_marbles (total : ℕ) (red : ℕ) (green : ℕ) (blue : ℕ) : 
  total = 63 →
  red = 38 →
  green = red / 2 →
  total = red + green + blue →
  blue = 6 := by sorry

end NUMINAMATH_CALUDE_freds_marbles_l945_94535


namespace NUMINAMATH_CALUDE_certain_number_problem_l945_94507

theorem certain_number_problem : ∃ x : ℕ, 
  220020 = (x + 445) * (2 * (x - 445)) + 20 ∧ x = 555 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l945_94507


namespace NUMINAMATH_CALUDE_prime_solution_equation_l945_94571

theorem prime_solution_equation : 
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    p^2 - 6*p*q + q^2 + 3*q - 1 = 0 → 
    (p = 17 ∧ q = 3) := by
  sorry

end NUMINAMATH_CALUDE_prime_solution_equation_l945_94571


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l945_94523

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 4 = 5) : 
  2 * (a 1) - (a 5) + (a 11) = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l945_94523


namespace NUMINAMATH_CALUDE_tetrahedron_edge_length_is_2_sqrt_5_l945_94522

-- Define the radius of each ball
def ball_radius : ℝ := 2

-- Define the arrangement of balls
structure BallArrangement where
  bottom_balls : Fin 4 → ℝ × ℝ × ℝ  -- Centers of the four bottom balls
  top_ball : ℝ × ℝ × ℝ              -- Center of the top ball

-- Define the properties of the arrangement
def valid_arrangement (arr : BallArrangement) : Prop :=
  -- Four bottom balls are mutually tangent
  ∀ i j, i ≠ j → ‖arr.bottom_balls i - arr.bottom_balls j‖ = 2 * ball_radius
  -- Top ball is tangent to all bottom balls
  ∧ ∀ i, ‖arr.top_ball - arr.bottom_balls i‖ = 2 * ball_radius

-- Define the tetrahedron circumscribed around the arrangement
def tetrahedron_edge_length (arr : BallArrangement) : ℝ :=
  ‖arr.top_ball - arr.bottom_balls 0‖

-- Theorem statement
theorem tetrahedron_edge_length_is_2_sqrt_5 (arr : BallArrangement) 
  (h : valid_arrangement arr) : 
  tetrahedron_edge_length arr = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_length_is_2_sqrt_5_l945_94522


namespace NUMINAMATH_CALUDE_bacteria_growth_l945_94587

/-- Bacteria growth problem -/
theorem bacteria_growth (initial_count : ℕ) (growth_factor : ℕ) (interval_count : ℕ) : 
  initial_count = 50 → 
  growth_factor = 3 → 
  interval_count = 5 → 
  initial_count * growth_factor ^ interval_count = 12150 := by
sorry

#eval 50 * 3 ^ 5  -- Expected output: 12150

end NUMINAMATH_CALUDE_bacteria_growth_l945_94587


namespace NUMINAMATH_CALUDE_line_segment_proportions_l945_94501

theorem line_segment_proportions (a b x : ℝ) : 
  (a / b = 3 / 2) → 
  (a + 2 * b = 28) → 
  (x^2 = a * b) →
  (a = 12 ∧ b = 8 ∧ x = 4 * Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_line_segment_proportions_l945_94501


namespace NUMINAMATH_CALUDE_function_value_at_three_l945_94576

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + 3 * f (1 - x) = 4 * x^2

theorem function_value_at_three 
  (f : ℝ → ℝ) 
  (h : FunctionalEquation f) : 
  f 3 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_three_l945_94576


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l945_94575

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (1 - a / (a + 1)) / ((a^2 - 2*a + 1) / (a^2 - 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l945_94575


namespace NUMINAMATH_CALUDE_terrell_workout_equivalence_l945_94510

/-- Given Terrell's original workout and new weights, calculate the number of lifts needed to match the total weight. -/
theorem terrell_workout_equivalence (original_weight original_reps new_weight : ℕ) : 
  original_weight = 30 →
  original_reps = 10 →
  new_weight = 20 →
  (2 * new_weight * (600 / (2 * new_weight)) : ℕ) = 2 * original_weight * original_reps :=
by
  sorry

#check terrell_workout_equivalence

end NUMINAMATH_CALUDE_terrell_workout_equivalence_l945_94510


namespace NUMINAMATH_CALUDE_inverse_of_complex_expression_l945_94513

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem inverse_of_complex_expression :
  i ^ 2 = -1 → (3 * i - 3 * i⁻¹)⁻¹ = -i / 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_complex_expression_l945_94513


namespace NUMINAMATH_CALUDE_shift_sine_graph_l945_94553

theorem shift_sine_graph (x : ℝ) :
  let f (x : ℝ) := 2 * Real.sin (2 * x + π / 6)
  let period := 2 * π / 2
  let shift := period / 4
  let g (x : ℝ) := f (x - shift)
  g x = 2 * Real.sin (2 * x - π / 3) := by sorry

end NUMINAMATH_CALUDE_shift_sine_graph_l945_94553


namespace NUMINAMATH_CALUDE_set_operations_with_empty_l945_94528

theorem set_operations_with_empty (A : Set α) : 
  (A ∩ ∅ = ∅) ∧ 
  (A ∪ ∅ = A) ∧ 
  ((A ∩ ∅ = ∅) ∧ (A ∪ ∅ = A)) ∧ 
  ((A ∩ ∅ = ∅) ∨ (A ∪ ∅ = A)) ∧ 
  ¬¬(A ∩ ∅ = ∅) ∧ 
  ¬¬(A ∪ ∅ = A) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_with_empty_l945_94528


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l945_94539

theorem gold_coin_distribution (x y : ℕ) (h : x^2 - y^2 = 16 * (x - y)) : x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l945_94539


namespace NUMINAMATH_CALUDE_integral_sum_reciprocal_and_semicircle_l945_94557

open Real MeasureTheory

theorem integral_sum_reciprocal_and_semicircle :
  ∫ x in (1 : ℝ)..3, (1 / x + Real.sqrt (1 - (x - 2)^2)) = Real.log 3 + π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sum_reciprocal_and_semicircle_l945_94557


namespace NUMINAMATH_CALUDE_equation_implies_conditions_l945_94530

theorem equation_implies_conditions (a b c d : ℝ) 
  (h : (a^2 + b^2) / (b^2 + c^2) = (c^2 + d^2) / (d^2 + a^2)) :
  a = c ∨ a = -c ∨ a^2 - c^2 + d^2 = b^2 := by
sorry

end NUMINAMATH_CALUDE_equation_implies_conditions_l945_94530


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_t_1_l945_94581

-- Define the position function S(t)
def S (t : ℝ) : ℝ := 2 * t^2 + t

-- Define the velocity function as the derivative of S(t)
def v (t : ℝ) : ℝ := 4 * t + 1

-- Theorem statement
theorem instantaneous_velocity_at_t_1 : v 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_t_1_l945_94581


namespace NUMINAMATH_CALUDE_field_width_l945_94502

/-- Given a rectangular field with length 75 meters, where running around it 3 times
    covers a distance of 540 meters, prove that the width of the field is 15 meters. -/
theorem field_width (length : ℝ) (width : ℝ) (perimeter : ℝ) :
  length = 75 →
  3 * perimeter = 540 →
  perimeter = 2 * (length + width) →
  width = 15 := by
sorry

end NUMINAMATH_CALUDE_field_width_l945_94502


namespace NUMINAMATH_CALUDE_percent_difference_l945_94541

theorem percent_difference (w e y z : ℝ) 
  (hw : w = 0.6 * e) 
  (he : e = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  z = 1.5 * w := by
  sorry

end NUMINAMATH_CALUDE_percent_difference_l945_94541


namespace NUMINAMATH_CALUDE_glucose_solution_volume_l945_94579

/-- The volume of glucose solution containing 15 grams of glucose -/
def volume_15g : ℝ := 100

/-- The volume of glucose solution used in the given condition -/
def volume_given : ℝ := 65

/-- The mass of glucose in the given volume -/
def mass_given : ℝ := 9.75

/-- The target mass of glucose -/
def mass_target : ℝ := 15

theorem glucose_solution_volume :
  (mass_given / volume_given) * volume_15g = mass_target :=
by sorry

end NUMINAMATH_CALUDE_glucose_solution_volume_l945_94579


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l945_94517

theorem geometric_sequence_sum_inequality (n : ℕ) : 2^n - 1 < 2^n := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l945_94517


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l945_94566

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

def left_append_2 (n : ℕ) : ℕ := 2000 + n

def right_append_2 (n : ℕ) : ℕ := n * 10 + 2

theorem three_digit_number_proof :
  ∃! n : ℕ, is_three_digit n ∧ has_distinct_digits n ∧
  (left_append_2 n - right_append_2 n = 945 ∨ right_append_2 n - left_append_2 n = 945) ∧
  n = 327 :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l945_94566


namespace NUMINAMATH_CALUDE_graph_quadrants_l945_94583

def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - k

theorem graph_quadrants (k : ℝ) (h : k < 0) :
  ∃ (x₁ x₂ x₃ : ℝ), 
    (x₁ > 0 ∧ linear_function k x₁ > 0) ∧  -- Quadrant I
    (x₂ < 0 ∧ linear_function k x₂ > 0) ∧  -- Quadrant II
    (x₃ > 0 ∧ linear_function k x₃ < 0)    -- Quadrant IV
  := by sorry

end NUMINAMATH_CALUDE_graph_quadrants_l945_94583


namespace NUMINAMATH_CALUDE_raffle_probabilities_l945_94586

/-- Represents the raffle ticket distribution -/
structure RaffleTickets where
  total : ℕ
  first_prize : ℕ
  second_prize : ℕ
  third_prize : ℕ
  h_total : total = first_prize + second_prize + third_prize

/-- The probability of drawing exactly k tickets of a specific type from n tickets in m draws -/
def prob_draw (n k m : ℕ) : ℚ :=
  (n.choose k * (n - k).choose (m - k)) / n.choose m

theorem raffle_probabilities (r : RaffleTickets)
    (h1 : r.total = 10)
    (h2 : r.first_prize = 2)
    (h3 : r.second_prize = 3)
    (h4 : r.third_prize = 5) :
  /- (I) Probability of drawing 2 first prize tickets -/
  (prob_draw r.first_prize 2 2 = 1 / 45) ∧
  /- (II) Probability of drawing at most 1 first prize ticket in 3 draws -/
  (prob_draw r.first_prize 0 3 + prob_draw r.first_prize 1 3 = 14 / 15) ∧
  /- (III) Mathematical expectation of second prize tickets in 3 draws -/
  (0 * prob_draw r.second_prize 0 3 +
   1 * prob_draw r.second_prize 1 3 +
   2 * prob_draw r.second_prize 2 3 +
   3 * prob_draw r.second_prize 3 3 = 9 / 10) :=
by sorry

end NUMINAMATH_CALUDE_raffle_probabilities_l945_94586


namespace NUMINAMATH_CALUDE_number_of_dime_piles_l945_94574

/-- Represents the number of coins in each pile -/
def coins_per_pile : ℕ := 10

/-- Represents the number of piles of quarters -/
def quarter_piles : ℕ := 4

/-- Represents the number of piles of nickels -/
def nickel_piles : ℕ := 9

/-- Represents the number of piles of pennies -/
def penny_piles : ℕ := 5

/-- Represents the total value of all coins in dollars -/
def total_value : ℚ := 21

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- Represents the value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- Represents the value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- Theorem stating that the number of piles of dimes is 6 -/
theorem number_of_dime_piles : ℕ := by
  sorry

end NUMINAMATH_CALUDE_number_of_dime_piles_l945_94574


namespace NUMINAMATH_CALUDE_triangle_inequality_l945_94503

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b + b * c + c * a ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l945_94503


namespace NUMINAMATH_CALUDE_remaining_crayons_l945_94500

def initial_crayons : ℕ := 440
def crayons_given_away : ℕ := 111
def crayons_lost : ℕ := 106

theorem remaining_crayons :
  initial_crayons - crayons_given_away - crayons_lost = 223 := by
  sorry

end NUMINAMATH_CALUDE_remaining_crayons_l945_94500


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l945_94589

/-- Represents the number of students in each year -/
structure StudentPopulation where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the total number of students -/
def total_students (pop : StudentPopulation) : ℕ :=
  pop.first_year + pop.second_year + pop.third_year

/-- Represents a stratified sample -/
structure StratifiedSample where
  population : StudentPopulation
  sample_size : ℕ

/-- Calculates the number of third-year students in the sample -/
def third_year_sample_size (sample : StratifiedSample) : ℕ :=
  (sample.population.third_year * sample.sample_size) / (total_students sample.population)

theorem stratified_sampling_theorem (sample : StratifiedSample) 
  (h1 : sample.population.first_year = 1300)
  (h2 : sample.population.second_year = 1200)
  (h3 : sample.population.third_year = 1500)
  (h4 : sample.sample_size = 200) :
  third_year_sample_size sample = 75 := by
  sorry

#eval third_year_sample_size {
  population := { first_year := 1300, second_year := 1200, third_year := 1500 },
  sample_size := 200
}

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l945_94589


namespace NUMINAMATH_CALUDE_percentage_problem_l945_94534

theorem percentage_problem : ∃ P : ℝ, P = (0.25 * 16 + 2) ∧ P = 6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l945_94534


namespace NUMINAMATH_CALUDE_tv_show_payment_ratio_l945_94504

/-- The ratio of payments to major and minor characters in a TV show -/
theorem tv_show_payment_ratio :
  let num_main_characters : ℕ := 5
  let num_minor_characters : ℕ := 4
  let minor_character_payment : ℕ := 15000
  let total_payment : ℕ := 285000
  let minor_characters_total : ℕ := num_minor_characters * minor_character_payment
  let major_characters_total : ℕ := total_payment - minor_characters_total
  (major_characters_total : ℚ) / minor_characters_total = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_payment_ratio_l945_94504


namespace NUMINAMATH_CALUDE_area_increase_is_204_l945_94537

/-- Represents the increase in vegetables from last year to this year -/
structure VegetableIncrease where
  broccoli : ℕ
  cauliflower : ℕ
  cabbage : ℕ

/-- Calculates the total increase in area given the increase in vegetables -/
def totalAreaIncrease (v : VegetableIncrease) : ℝ :=
  v.broccoli * 1 + v.cauliflower * 2 + v.cabbage * 1.5

/-- The theorem stating that the total increase in area is 204 square feet -/
theorem area_increase_is_204 (v : VegetableIncrease) 
  (h1 : v.broccoli = 79)
  (h2 : v.cauliflower = 25)
  (h3 : v.cabbage = 50) : 
  totalAreaIncrease v = 204 := by
  sorry

#eval totalAreaIncrease { broccoli := 79, cauliflower := 25, cabbage := 50 }

end NUMINAMATH_CALUDE_area_increase_is_204_l945_94537


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l945_94506

/-- A regular polygon with side length 2 and interior angles measuring 135° has a perimeter of 16. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (interior_angle : ℝ) :
  n ≥ 3 ∧
  side_length = 2 ∧
  interior_angle = 135 ∧
  (n : ℝ) * (180 - interior_angle) = 360 →
  n * side_length = 16 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l945_94506


namespace NUMINAMATH_CALUDE_abs_eq_neg_implies_nonpositive_l945_94508

theorem abs_eq_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_neg_implies_nonpositive_l945_94508


namespace NUMINAMATH_CALUDE_octagon_perimeter_l945_94532

/-- Represents an eight-sided polygon that can be divided into a rectangle and a square --/
structure OctagonWithRectAndSquare where
  rectangle_area : ℕ
  square_area : ℕ
  sum_perimeter : ℕ
  h1 : square_area > rectangle_area
  h2 : square_area * rectangle_area = 98
  h3 : ∃ (a b : ℕ), rectangle_area = a * b ∧ a > 0 ∧ b > 0
  h4 : ∃ (s : ℕ), square_area = s * s ∧ s > 0

/-- The perimeter of the octagon is 32 --/
theorem octagon_perimeter (oct : OctagonWithRectAndSquare) : oct.sum_perimeter = 32 := by
  sorry

end NUMINAMATH_CALUDE_octagon_perimeter_l945_94532


namespace NUMINAMATH_CALUDE_water_in_bucket_l945_94595

theorem water_in_bucket (initial_amount : ℝ) (poured_out : ℝ) (remaining_amount : ℝ) :
  initial_amount = 0.8 →
  poured_out = 0.2 →
  remaining_amount = initial_amount - poured_out →
  remaining_amount = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_water_in_bucket_l945_94595


namespace NUMINAMATH_CALUDE_area_covered_is_56_l945_94524

/-- The total area covered by five rectangular strips arranged in a specific pattern. -/
def total_area_covered (strip_length : ℝ) (strip_width : ℝ) (center_overlap : ℝ) : ℝ :=
  let single_strip_area := strip_length * strip_width
  let total_area_without_overlap := 5 * single_strip_area
  let center_overlap_area := 4 * (center_overlap * center_overlap)
  let fifth_strip_overlap_area := 2 * (center_overlap * center_overlap)
  total_area_without_overlap - (center_overlap_area + fifth_strip_overlap_area)

/-- Theorem stating that the total area covered by the strips is 56. -/
theorem area_covered_is_56 :
  total_area_covered 8 2 2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_area_covered_is_56_l945_94524


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l945_94529

theorem units_digit_of_expression : ∃ n : ℕ, (9 * 19 * 1989 - 9^4) % 10 = 8 ∧ n * 10 + 8 = 9 * 19 * 1989 - 9^4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l945_94529


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l945_94598

theorem trig_expression_equals_one :
  (Real.sqrt 3 * Real.sin (20 * π / 180) + Real.sin (70 * π / 180)) /
  Real.sqrt (2 - 2 * Real.cos (100 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l945_94598


namespace NUMINAMATH_CALUDE_quadratic_single_intersection_l945_94558

theorem quadratic_single_intersection (m : ℝ) : 
  (∃! x, (m + 1) * x^2 - 2*(m + 1) * x - 1 = 0) ↔ m = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_single_intersection_l945_94558


namespace NUMINAMATH_CALUDE_pythagorean_triple_in_range_l945_94519

theorem pythagorean_triple_in_range : 
  ∀ a b c : ℕ, 
    a^2 + b^2 = c^2 → 
    Nat.gcd a (Nat.gcd b c) = 1 → 
    2000 ≤ a ∧ a ≤ 3000 → 
    2000 ≤ b ∧ b ≤ 3000 → 
    2000 ≤ c ∧ c ≤ 3000 → 
    (a, b, c) = (2100, 2059, 2941) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_in_range_l945_94519


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l945_94531

theorem right_triangle_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  a / b = 2 / 5 →    -- Given ratio of a to b
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  r + s = c →        -- r and s are segments of c
  r / s = 4 / 25 :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l945_94531


namespace NUMINAMATH_CALUDE_inequality_solution_l945_94599

theorem inequality_solution (x : ℝ) : 
  (3 * x - 2 ≥ 0) → 
  (|Real.sqrt (3 * x - 2) - 3| > 1 ↔ (x > 6 ∨ (2/3 ≤ x ∧ x < 2))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l945_94599


namespace NUMINAMATH_CALUDE_kayla_driving_years_l945_94518

/-- The minimum driving age in Kayla's state -/
def minimum_driving_age : ℕ := 18

/-- Kimiko's age -/
def kimiko_age : ℕ := 26

/-- Kayla's current age -/
def kayla_age : ℕ := kimiko_age / 2

/-- The number of years before Kayla can reach the minimum driving age -/
def years_until_driving : ℕ := minimum_driving_age - kayla_age

theorem kayla_driving_years :
  years_until_driving = 5 :=
by sorry

end NUMINAMATH_CALUDE_kayla_driving_years_l945_94518


namespace NUMINAMATH_CALUDE_paige_homework_pages_l945_94592

/-- Given the total number of homework problems, the number of finished problems,
    and the number of problems per page, calculate the number of remaining pages. -/
def remaining_pages (total_problems : ℕ) (finished_problems : ℕ) (problems_per_page : ℕ) : ℕ :=
  (total_problems - finished_problems) / problems_per_page

/-- Theorem stating that for Paige's homework scenario, the number of remaining pages is 7. -/
theorem paige_homework_pages : remaining_pages 110 47 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_paige_homework_pages_l945_94592


namespace NUMINAMATH_CALUDE_train_optimization_l945_94527

/-- Represents the relationship between carriages and round trips -/
def round_trips (x : ℝ) : ℝ := -2 * x + 24

/-- Represents the total number of passengers transported per day -/
def passengers (x : ℝ) : ℝ := 110 * x * round_trips x

/-- The optimal number of carriages -/
def optimal_carriages : ℝ := 6

/-- The optimal number of round trips -/
def optimal_trips : ℝ := round_trips optimal_carriages

/-- The maximum number of passengers per day -/
def max_passengers : ℝ := passengers optimal_carriages

theorem train_optimization :
  (round_trips 4 = 16) →
  (round_trips 7 = 10) →
  (∀ x, round_trips x = -2 * x + 24) →
  (optimal_carriages = 6) →
  (optimal_trips = 12) →
  (max_passengers = 7920) →
  (∀ x, passengers x ≤ max_passengers) :=
by sorry

end NUMINAMATH_CALUDE_train_optimization_l945_94527


namespace NUMINAMATH_CALUDE_three_numbers_average_l945_94545

theorem three_numbers_average (a b c : ℝ) 
  (h1 : a + (b + c)/2 = 65)
  (h2 : b + (a + c)/2 = 69)
  (h3 : c + (a + b)/2 = 76) :
  (a + b + c)/3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_average_l945_94545


namespace NUMINAMATH_CALUDE_quadratic_vertex_l945_94559

/-- A quadratic function f(x) = x^2 + bx + c -/
def QuadraticFunction (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

theorem quadratic_vertex (b c : ℝ) :
  (QuadraticFunction b c 1 = 0) →
  (∀ x, QuadraticFunction b c (2 + x) = QuadraticFunction b c (2 - x)) →
  (∃ y, QuadraticFunction b c 2 = y ∧ ∀ x, QuadraticFunction b c x ≥ y) →
  (2, -1) = (2, QuadraticFunction b c 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l945_94559


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l945_94582

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 10th term is 20 and the 11th term is 24,
    the 2nd term of the sequence is -12. -/
theorem arithmetic_sequence_second_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_10th : a 10 = 20)
  (h_11th : a 11 = 24) :
  a 2 = -12 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l945_94582


namespace NUMINAMATH_CALUDE_ball_probabilities_l945_94567

-- Define the sample space
def Ω : Type := Unit

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the events
def red : Set Ω := sorry
def black : Set Ω := sorry
def white : Set Ω := sorry
def green : Set Ω := sorry

-- State the theorem
theorem ball_probabilities 
  (h1 : P red = 5/12)
  (h2 : P black = 1/3)
  (h3 : P white = 1/6)
  (h4 : P green = 1/12)
  (h5 : Disjoint red black)
  (h6 : Disjoint red white)
  (h7 : Disjoint red green)
  (h8 : Disjoint black white)
  (h9 : Disjoint black green)
  (h10 : Disjoint white green) :
  (P (red ∪ black) = 3/4) ∧ 
  (P (red ∪ black ∪ white) = 11/12) := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l945_94567


namespace NUMINAMATH_CALUDE_compare_sqrt_expressions_l945_94588

theorem compare_sqrt_expressions : 2 * Real.sqrt 2 - Real.sqrt 7 < Real.sqrt 6 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_compare_sqrt_expressions_l945_94588


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l945_94552

/-- Given three points A, B, and C in 2D space, determines if they are collinear -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem stating that if A(1,2), B(3,m), and C(7,m+6) are collinear, then m = 5 -/
theorem collinear_points_m_value :
  ∀ m : ℝ, collinear (1, 2) (3, m) (7, m + 6) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l945_94552


namespace NUMINAMATH_CALUDE_count_negative_numbers_l945_94515

def numbers : List ℝ := [-2.5, 7, -3, 2, 0, 4, 5, -1]

theorem count_negative_numbers : 
  (numbers.filter (· < 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l945_94515


namespace NUMINAMATH_CALUDE_athlete_running_time_l945_94548

/-- Proof that an athlete spends 35 minutes running given the conditions -/
theorem athlete_running_time 
  (calories_per_minute_running : ℕ) 
  (calories_per_minute_walking : ℕ)
  (total_calories_burned : ℕ)
  (total_time : ℕ)
  (h1 : calories_per_minute_running = 10)
  (h2 : calories_per_minute_walking = 4)
  (h3 : total_calories_burned = 450)
  (h4 : total_time = 60) :
  ∃ (running_time : ℕ), 
    running_time = 35 ∧ 
    running_time + (total_time - running_time) = total_time ∧
    calories_per_minute_running * running_time + 
    calories_per_minute_walking * (total_time - running_time) = total_calories_burned :=
by
  sorry


end NUMINAMATH_CALUDE_athlete_running_time_l945_94548


namespace NUMINAMATH_CALUDE_opposite_and_reciprocal_expression_l945_94573

theorem opposite_and_reciprocal_expression (a b c d : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) : 
  (a + b) / 2 - c * d = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_and_reciprocal_expression_l945_94573


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l945_94570

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    n = 113 ∧ 
    100 ≤ n ∧ n < 1000 ∧ 
    (77 * n) % 385 = 231 % 385 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < n → (77 * m) % 385 ≠ 231 % 385 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l945_94570


namespace NUMINAMATH_CALUDE_stream_speed_l945_94547

/-- Proves that given a boat with a speed of 8 kmph in standing water,
    traveling a round trip of 420 km (210 km each way) in 56 hours,
    the speed of the stream is 2 kmph. -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) :
  boat_speed = 8 →
  distance = 210 →
  total_time = 56 →
  ∃ (stream_speed : ℝ),
    stream_speed = 2 ∧
    (distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed) = total_time) :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l945_94547


namespace NUMINAMATH_CALUDE_shortest_side_of_similar_triangle_l945_94540

/-- Given two similar right triangles, where the first triangle has a side of 15 and a hypotenuse of 17,
    and the second triangle has a hypotenuse of 102, the shortest side of the second triangle is 48. -/
theorem shortest_side_of_similar_triangle (a b c : ℝ) : 
  a ^ 2 + 15 ^ 2 = 17 ^ 2 → -- First triangle is right-angled with side 15 and hypotenuse 17
  a ≤ 15 → -- a is the shortest side of the first triangle
  ∃ (k : ℝ), k > 0 ∧ k * 17 = 102 ∧ k * a = 48 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_of_similar_triangle_l945_94540


namespace NUMINAMATH_CALUDE_existence_of_binomial_solution_l945_94546

theorem existence_of_binomial_solution (a b : ℕ+) :
  ∃ (x y : ℕ+), Nat.choose (x + y) 2 = a * x + b * y := by
  sorry

end NUMINAMATH_CALUDE_existence_of_binomial_solution_l945_94546


namespace NUMINAMATH_CALUDE_patricia_books_count_l945_94564

/-- Given the number of books read by Candice, calculate the number of books read by Patricia -/
def books_read_by_patricia (candice_books : ℕ) : ℕ :=
  let amanda_books := candice_books / 3
  let kara_books := amanda_books / 2
  7 * kara_books

/-- Theorem stating that if Candice read 18 books, Patricia read 21 books -/
theorem patricia_books_count (h : books_read_by_patricia 18 = 21) : 
  books_read_by_patricia 18 = 21 := by
  sorry

#eval books_read_by_patricia 18

end NUMINAMATH_CALUDE_patricia_books_count_l945_94564


namespace NUMINAMATH_CALUDE_arctan_sum_three_fourths_four_thirds_l945_94594

theorem arctan_sum_three_fourths_four_thirds : 
  Real.arctan (3/4) + Real.arctan (4/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_fourths_four_thirds_l945_94594


namespace NUMINAMATH_CALUDE_identify_real_coins_l945_94509

/-- Represents the result of weighing two coins -/
inductive WeighResult
| Equal : WeighResult
| LeftHeavier : WeighResult
| RightHeavier : WeighResult

/-- Represents a coin -/
structure Coin :=
  (id : Nat)
  (isReal : Bool)

/-- Represents the balance scale that always shows an incorrect result -/
def incorrectBalance (left right : Coin) : WeighResult :=
  sorry

/-- The main theorem to prove -/
theorem identify_real_coins 
  (coins : Finset Coin) 
  (h_count : coins.card = 100) 
  (h_real : ∃ (fake : Coin), fake ∈ coins ∧ 
    (∀ c ∈ coins, c ≠ fake → c.isReal) ∧ 
    (¬fake.isReal)) : 
  ∃ (realCoins : Finset Coin), realCoins ⊆ coins ∧ realCoins.card = 98 ∧ 
    (∀ c ∈ realCoins, c.isReal) :=
  sorry

end NUMINAMATH_CALUDE_identify_real_coins_l945_94509


namespace NUMINAMATH_CALUDE_proposition_logic_l945_94536

theorem proposition_logic : 
  let p := (3 : ℝ) ≥ 3
  let q := (3 : ℝ) > 4
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by sorry

end NUMINAMATH_CALUDE_proposition_logic_l945_94536


namespace NUMINAMATH_CALUDE_handshake_arrangements_mod_1000_l945_94512

/-- A handshake arrangement for a group of people -/
structure HandshakeArrangement (n : ℕ) where
  shakes : Fin n → Finset (Fin n)
  two_shakes : ∀ i, (shakes i).card = 2
  symmetry : ∀ i j, j ∈ shakes i ↔ i ∈ shakes j

/-- Two handshake arrangements are different if at least two people who shake hands
    in one arrangement do not shake hands in the other -/
def different_arrangements {n : ℕ} (a b : HandshakeArrangement n) : Prop :=
  ∃ i j, (j ∈ a.shakes i ∧ j ∉ b.shakes i) ∨ (j ∉ a.shakes i ∧ j ∈ b.shakes i)

/-- The number of distinct handshake arrangements for 10 people -/
def num_arrangements : ℕ := sorry

theorem handshake_arrangements_mod_1000 :
  num_arrangements ≡ 444 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_handshake_arrangements_mod_1000_l945_94512


namespace NUMINAMATH_CALUDE_boats_first_meeting_distance_l945_94565

/-- Two boats traveling across a river, meeting twice without stopping at shores -/
structure BoatMeeting where
  /-- Total distance between shore A and B in yards -/
  total_distance : ℝ
  /-- Distance from shore B to the second meeting point in yards -/
  second_meeting_distance : ℝ
  /-- Distance from shore A to the first meeting point in yards -/
  first_meeting_distance : ℝ

/-- Theorem stating that the boats first meet at 300 yards from shore A -/
theorem boats_first_meeting_distance (meeting : BoatMeeting)
    (h1 : meeting.total_distance = 1200)
    (h2 : meeting.second_meeting_distance = 300) :
    meeting.first_meeting_distance = 300 := by
  sorry


end NUMINAMATH_CALUDE_boats_first_meeting_distance_l945_94565


namespace NUMINAMATH_CALUDE_inequality_proof_l945_94580

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  Real.sqrt ((x^3 + y + 1) * (y^3 + x + 1)) ≥ x^2 + y^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l945_94580


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l945_94544

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2, x)
  let b : ℝ × ℝ := (1, 2)
  parallel a b → x = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l945_94544


namespace NUMINAMATH_CALUDE_find_divisor_l945_94533

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) (divisor : Nat) :
  dividend = divisor * quotient + remainder →
  dividend = 109 →
  quotient = 9 →
  remainder = 1 →
  divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l945_94533


namespace NUMINAMATH_CALUDE_enjoy_both_activities_l945_94551

theorem enjoy_both_activities (total : ℕ) (reading : ℕ) (movies : ℕ) (neither : ℕ)
  (h1 : total = 50)
  (h2 : reading = 22)
  (h3 : movies = 20)
  (h4 : neither = 15) :
  total - neither - (reading + movies - (total - neither)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_enjoy_both_activities_l945_94551


namespace NUMINAMATH_CALUDE_exists_unique_polynomial_l945_94525

/-- Definition of the polynomial p(x, y) -/
def p (x y : ℕ) : ℕ := (x + y)^2 + 3*x + y

/-- Statement of the theorem -/
theorem exists_unique_polynomial :
  ∀ n : ℕ, ∃! (k m : ℕ), p k m = n :=
by sorry

end NUMINAMATH_CALUDE_exists_unique_polynomial_l945_94525
