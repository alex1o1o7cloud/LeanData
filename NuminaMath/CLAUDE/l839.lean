import Mathlib

namespace NUMINAMATH_CALUDE_intersection_slope_l839_83982

/-- Given two lines p and q that intersect at (-3, -9), prove that the slope of line q is 0 -/
theorem intersection_slope (k : ℝ) : 
  (∀ x y : ℝ, y = 4*x + 3 → y = k*x - 9 → x = -3 ∧ y = -9) → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l839_83982


namespace NUMINAMATH_CALUDE_problem_statement_l839_83930

/-- An arithmetic sequence with a non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) - a n = d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) / b n = r

theorem problem_statement (a b : ℕ → ℝ) : 
  arithmetic_sequence a →
  geometric_sequence b →
  3 * a 2005 - (a 2007)^2 + 3 * a 2009 = 0 →
  b 2007 = a 2007 →
  b 2006 * b 2008 = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l839_83930


namespace NUMINAMATH_CALUDE_total_hours_is_fifty_l839_83926

/-- Represents the worker's pay structure and work week --/
structure WorkWeek where
  ordinary_rate : ℚ  -- Rate for ordinary time in dollars per hour
  overtime_rate : ℚ  -- Rate for overtime in dollars per hour
  total_pay : ℚ      -- Total pay for the week in dollars
  overtime_hours : ℕ  -- Number of overtime hours worked

/-- Calculates the total hours worked given a WorkWeek --/
def total_hours (w : WorkWeek) : ℚ :=
  let ordinary_hours := (w.total_pay - w.overtime_rate * w.overtime_hours) / w.ordinary_rate
  ordinary_hours + w.overtime_hours

/-- Theorem stating that given the specific conditions, the total hours worked is 50 --/
theorem total_hours_is_fifty : 
  ∀ (w : WorkWeek), 
    w.ordinary_rate = 0.60 ∧ 
    w.overtime_rate = 0.90 ∧ 
    w.total_pay = 32.40 ∧ 
    w.overtime_hours = 8 → 
    total_hours w = 50 :=
by
  sorry


end NUMINAMATH_CALUDE_total_hours_is_fifty_l839_83926


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_l839_83900

/-- Sum of digits in decimal representation -/
def sumOfDecimalDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDecimalDigits (n / 10)

/-- Sum of digits in binary representation -/
def sumOfBinaryDigits (n : Nat) : Nat :=
  if n = 0 then 0 else (n % 2) + sumOfBinaryDigits (n / 2)

/-- Cost calculation for Option 1 -/
def option1Cost (n : Nat) : Nat :=
  2 * sumOfDecimalDigits n

/-- Cost calculation for Option 2 -/
def option2Cost (n : Nat) : Nat :=
  sumOfBinaryDigits n

theorem largest_equal_cost_number :
  ∀ n : Nat, n < 2000 → n > 1023 →
    option1Cost n ≠ option2Cost n ∧
    option1Cost 1023 = option2Cost 1023 :=
by sorry

end NUMINAMATH_CALUDE_largest_equal_cost_number_l839_83900


namespace NUMINAMATH_CALUDE_claire_earnings_l839_83939

-- Define the given quantities
def total_flowers : ℕ := 400
def tulips : ℕ := 120
def white_roses : ℕ := 80
def small_red_roses : ℕ := 40
def medium_red_roses : ℕ := 60

-- Define the prices
def price_small : ℚ := 3/4
def price_medium : ℚ := 1
def price_large : ℚ := 5/4

-- Calculate the number of roses and red roses
def roses : ℕ := total_flowers - tulips
def red_roses : ℕ := roses - white_roses

-- Calculate the number of large red roses
def large_red_roses : ℕ := red_roses - small_red_roses - medium_red_roses

-- Define the function to calculate earnings
def earnings : ℚ :=
  (small_red_roses / 2 : ℚ) * price_small +
  (medium_red_roses / 2 : ℚ) * price_medium +
  (large_red_roses / 2 : ℚ) * price_large

-- Theorem statement
theorem claire_earnings : earnings = 215/2 := by sorry

end NUMINAMATH_CALUDE_claire_earnings_l839_83939


namespace NUMINAMATH_CALUDE_units_digit_of_power_l839_83927

theorem units_digit_of_power (n : ℕ) : n > 0 → (7^(7 * (13^13))) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_l839_83927


namespace NUMINAMATH_CALUDE_tangent_line_and_root_range_l839_83938

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x

theorem tangent_line_and_root_range :
  -- Part 1: Tangent line equation
  (∀ x y : ℝ, y = f x → (x = 2 → 12 * x - y - 17 = 0)) ∧
  -- Part 2: Range of m for three distinct real roots
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔ -3 < m ∧ m < -2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_root_range_l839_83938


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l839_83923

theorem lcm_gcd_product (a b : ℕ) (ha : a = 8) (hb : b = 6) :
  Nat.lcm a b * Nat.gcd a b = a * b := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l839_83923


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l839_83991

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I)^2 = 2 - Complex.I) :
  Complex.abs z = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l839_83991


namespace NUMINAMATH_CALUDE_inequality_proof_l839_83945

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1) / (y + 1) + (y + 1) / (z + 1) + (z + 1) / (x + 1) ≤ x / y + y / z + z / x :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l839_83945


namespace NUMINAMATH_CALUDE_max_sum_is_24_l839_83957

def numbers : Finset ℕ := {1, 4, 7, 10, 13}

def valid_arrangement (a b c d e : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ d ∈ numbers ∧ e ∈ numbers ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  a + b + e = a + c + e

def sum_of_arrangement (a b c d e : ℕ) : ℕ := a + b + e

theorem max_sum_is_24 :
  ∀ a b c d e : ℕ, valid_arrangement a b c d e →
    sum_of_arrangement a b c d e ≤ 24 :=
sorry

end NUMINAMATH_CALUDE_max_sum_is_24_l839_83957


namespace NUMINAMATH_CALUDE_cube_root_of_sqrt_64_l839_83989

theorem cube_root_of_sqrt_64 : (64 : ℝ) ^ (1/2 : ℝ) ^ (1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_sqrt_64_l839_83989


namespace NUMINAMATH_CALUDE_solve_equation_and_evaluate_l839_83975

theorem solve_equation_and_evaluate : ∃ x : ℝ, 
  (5 * x - 3 = 15 * x + 15) ∧ (6 * (x + 5) = 19.2) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_and_evaluate_l839_83975


namespace NUMINAMATH_CALUDE_polar_coordinates_not_bijective_l839_83918

-- Define the types for different coordinate systems
def CartesianPoint := ℝ × ℝ
def ComplexPoint := ℂ
def PolarPoint := ℝ × ℝ  -- (r, θ)
def Vector2D := ℝ × ℝ

-- Define the bijection property
def IsBijective (f : α → β) : Prop :=
  Function.Injective f ∧ Function.Surjective f

-- State the theorem
theorem polar_coordinates_not_bijective :
  ∃ (f : CartesianPoint → ℝ × ℝ), IsBijective f ∧
  ∃ (g : ComplexPoint → ℝ × ℝ), IsBijective g ∧
  ∃ (h : Vector2D → ℝ × ℝ), IsBijective h ∧
  ¬∃ (k : PolarPoint → ℝ × ℝ), IsBijective k :=
sorry

end NUMINAMATH_CALUDE_polar_coordinates_not_bijective_l839_83918


namespace NUMINAMATH_CALUDE_triangle_inequality_l839_83959

open Real

theorem triangle_inequality (A B C : ℝ) (R r : ℝ) :
  R > 0 ∧ r > 0 →
  (3 * Real.sqrt 3 * r^2) / (2 * R^2) ≤ Real.sin A * Real.sin B * Real.sin C ∧
  Real.sin A * Real.sin B * Real.sin C ≤ (3 * Real.sqrt 3 * r) / (4 * R) ∧
  (3 * Real.sqrt 3 * r) / (4 * R) ≤ 3 * Real.sqrt 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l839_83959


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l839_83976

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  is_geometric a → a 1 = 8 → a 2 * a 3 = -8 → a 4 = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l839_83976


namespace NUMINAMATH_CALUDE_max_toys_frank_can_buy_l839_83915

/-- Represents the types of toys available --/
inductive Toy
| SmallCar
| Puzzle
| LegoSet

/-- Returns the price of a toy --/
def toyPrice (t : Toy) : ℕ :=
  match t with
  | Toy.SmallCar => 8
  | Toy.Puzzle => 12
  | Toy.LegoSet => 20

/-- Represents a shopping cart with toys --/
structure Cart :=
  (smallCars : ℕ)
  (puzzles : ℕ)
  (legoSets : ℕ)

/-- Calculates the total cost of a cart, considering the promotion --/
def cartCost (c : Cart) : ℕ :=
  (c.smallCars / 3 * 2 + c.smallCars % 3) * toyPrice Toy.SmallCar +
  (c.puzzles / 3 * 2 + c.puzzles % 3) * toyPrice Toy.Puzzle +
  (c.legoSets / 3 * 2 + c.legoSets % 3) * toyPrice Toy.LegoSet

/-- Calculates the total number of toys in a cart --/
def cartSize (c : Cart) : ℕ :=
  c.smallCars + c.puzzles + c.legoSets

/-- Theorem: The maximum number of toys Frank can buy with $40 is 6 --/
theorem max_toys_frank_can_buy :
  ∀ c : Cart, cartCost c ≤ 40 → cartSize c ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_toys_frank_can_buy_l839_83915


namespace NUMINAMATH_CALUDE_circle_diameter_from_triangle_l839_83960

/-- Theorem: The diameter of a circle inscribing a right triangle with area 150 and one leg 30 is 10√10 -/
theorem circle_diameter_from_triangle (triangle_area : ℝ) (leg : ℝ) (diameter : ℝ) : 
  triangle_area = 150 →
  leg = 30 →
  diameter = 10 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_triangle_l839_83960


namespace NUMINAMATH_CALUDE_floor_neg_seven_halves_l839_83948

theorem floor_neg_seven_halves : ⌊(-7 : ℚ) / 2⌋ = -4 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_seven_halves_l839_83948


namespace NUMINAMATH_CALUDE_twin_prime_conjecture_equivalence_l839_83966

def is_twin_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (p + 2)

def cannot_be_written (k : ℤ) : Prop :=
  ∀ u v : ℕ, u > 0 ∧ v > 0 →
    k ≠ 6*u*v + u + v ∧
    k ≠ 6*u*v + u - v ∧
    k ≠ 6*u*v - u + v ∧
    k ≠ 6*u*v - u - v

theorem twin_prime_conjecture_equivalence :
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ p ∈ S, is_twin_prime p) ↔
  (∃ (T : Set ℤ), Set.Infinite T ∧ ∀ k ∈ T, cannot_be_written k) :=
sorry

end NUMINAMATH_CALUDE_twin_prime_conjecture_equivalence_l839_83966


namespace NUMINAMATH_CALUDE_function_domain_range_implies_b_equals_two_l839_83973

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the properties of the function
def has_domain_range (b : ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 1 b ↔ f x ∈ Set.Icc 1 b) ∧
  (∀ y ∈ Set.Icc 1 b, ∃ x ∈ Set.Icc 1 b, f x = y)

-- Theorem statement
theorem function_domain_range_implies_b_equals_two :
  ∃ b : ℝ, has_domain_range b → b = 2 := by sorry

end NUMINAMATH_CALUDE_function_domain_range_implies_b_equals_two_l839_83973


namespace NUMINAMATH_CALUDE_average_speed_is_27_point_5_l839_83956

-- Define the initial and final odometer readings
def initial_reading : ℕ := 1551
def final_reading : ℕ := 1881

-- Define the total riding time in hours
def total_time : ℕ := 12

-- Define the average speed
def average_speed : ℚ := (final_reading - initial_reading : ℚ) / total_time

-- Theorem statement
theorem average_speed_is_27_point_5 : average_speed = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_is_27_point_5_l839_83956


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l839_83978

theorem arithmetic_simplification : 
  (427 / 2.68) * 16 * 26.8 / 42.7 * 16 = 25600 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l839_83978


namespace NUMINAMATH_CALUDE_at_least_one_angle_leq_60_l839_83985

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : a + b + c = 180

-- Theorem statement
theorem at_least_one_angle_leq_60 (t : Triangle) : 
  t.a ≤ 60 ∨ t.b ≤ 60 ∨ t.c ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_angle_leq_60_l839_83985


namespace NUMINAMATH_CALUDE_possible_values_l839_83980

def Rectangle := Fin 3 → Fin 4 → ℕ

def valid_rectangle (r : Rectangle) : Prop :=
  (∀ i j, r i j ∈ Finset.range 13) ∧
  (∀ i j k, i ≠ j → r i k ≠ r j k) ∧
  (∀ k, r 0 k + r 1 k = 2 * r 2 k) ∧
  (r 0 0 = 6 ∧ r 1 0 = 4 ∧ r 2 1 = 8 ∧ r 2 2 = 11)

theorem possible_values (r : Rectangle) (h : valid_rectangle r) :
  r 2 3 = 2 ∨ r 2 3 = 11 :=
sorry

end NUMINAMATH_CALUDE_possible_values_l839_83980


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l839_83916

-- Define the solution set type
def SolutionSet := Set ℝ

-- Define the given inequality
def givenInequality (k a b c x : ℝ) : Prop :=
  (k / (x + a) + (x + b) / (x + c)) < 0

-- Define the target inequality
def targetInequality (k a b c x : ℝ) : Prop :=
  (k * x / (a * x + 1) + (b * x + 1) / (c * x + 1)) < 0

-- State the theorem
theorem solution_set_equivalence 
  (k a b c : ℝ) 
  (h : SolutionSet = {x | x ∈ (Set.Ioo (-1) (-1/3) ∪ Set.Ioo (1/2) 1) ∧ givenInequality k a b c x}) :
  SolutionSet = {x | x ∈ (Set.Ioo (-3) (-1) ∪ Set.Ioo 1 2) ∧ targetInequality k a b c x} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l839_83916


namespace NUMINAMATH_CALUDE_solve_proportion_l839_83902

theorem solve_proportion (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_proportion_l839_83902


namespace NUMINAMATH_CALUDE_problem_statement_l839_83967

theorem problem_statement (x y : ℝ) (hx : x = 7) (hy : y = 3) :
  (x - y)^2 * (x + y) = 160 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l839_83967


namespace NUMINAMATH_CALUDE_school_picnic_volunteers_l839_83995

theorem school_picnic_volunteers (total_parents : ℕ) (supervise : ℕ) (both : ℕ) (refresh_ratio : ℚ) : 
  total_parents = 84 →
  supervise = 25 →
  both = 11 →
  refresh_ratio = 3/2 →
  ∃ (refresh : ℕ) (neither : ℕ),
    refresh = refresh_ratio * neither ∧
    total_parents = (supervise - both) + (refresh - both) + both + neither ∧
    refresh = 42 := by
  sorry

end NUMINAMATH_CALUDE_school_picnic_volunteers_l839_83995


namespace NUMINAMATH_CALUDE_scholarship_sum_l839_83946

theorem scholarship_sum (wendy kelly nina : ℕ) : 
  wendy = 20000 →
  kelly = 2 * wendy →
  nina = kelly - 8000 →
  wendy + kelly + nina = 92000 := by
  sorry

end NUMINAMATH_CALUDE_scholarship_sum_l839_83946


namespace NUMINAMATH_CALUDE_exists_divisible_by_11_in_39_consecutive_integers_l839_83965

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_divisible_by_11_in_39_consecutive_integers :
  ∀ (start : ℕ), ∃ (k : ℕ), k ∈ Finset.range 39 ∧ (sumOfDigits (start + k) % 11 = 0) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_11_in_39_consecutive_integers_l839_83965


namespace NUMINAMATH_CALUDE_average_of_first_group_l839_83922

theorem average_of_first_group (n₁ : ℕ) (n₂ : ℕ) (avg₂ : ℝ) (avg_total : ℝ) :
  n₁ = 40 →
  n₂ = 30 →
  avg₂ = 40 →
  avg_total = 34.285714285714285 →
  (n₁ * (n₁ + n₂) * avg_total - n₂ * avg₂ * (n₁ + n₂)) / (n₁ * (n₁ + n₂)) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_average_of_first_group_l839_83922


namespace NUMINAMATH_CALUDE_homework_decrease_iff_thirty_percent_l839_83913

/-- Represents the decrease in homework duration over two reforms -/
def homework_decrease (a : ℝ) (x : ℝ) : Prop :=
  a * (1 - x)^2 = 0.3 * a

/-- Theorem stating that the homework decrease equation holds if and only if
    the final duration is 30% of the initial duration -/
theorem homework_decrease_iff_thirty_percent (a : ℝ) (x : ℝ) (h_a : a > 0) :
  homework_decrease a x ↔ a * (1 - x)^2 = 0.3 * a :=
sorry

end NUMINAMATH_CALUDE_homework_decrease_iff_thirty_percent_l839_83913


namespace NUMINAMATH_CALUDE_sum_of_ac_l839_83911

theorem sum_of_ac (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 48) 
  (h2 : b + d = 6) : 
  a + c = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ac_l839_83911


namespace NUMINAMATH_CALUDE_smallest_number_with_rearranged_double_l839_83963

def digits_to_num (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

def num_to_digits (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

def rearrange_digits (digits : List Nat) : List Nat :=
  (digits.take 2).reverse ++ (digits.drop 2).reverse

theorem smallest_number_with_rearranged_double :
  ∃ (n : Nat),
    n = 263157894736842105 ∧
    (∀ m : Nat, m < n →
      let digits_m := num_to_digits m
      let r_m := digits_to_num (rearrange_digits digits_m)
      r_m ≠ 2 * m) ∧
    let digits_n := num_to_digits n
    let r_n := digits_to_num (rearrange_digits digits_n)
    r_n = 2 * n :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_rearranged_double_l839_83963


namespace NUMINAMATH_CALUDE_sum_expression_value_l839_83903

theorem sum_expression_value (a b c : ℝ) 
  (h1 : a + b = 8) 
  (h2 : a * b = c^2 + 16) : 
  a + 2*b + 3*c = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_expression_value_l839_83903


namespace NUMINAMATH_CALUDE_eight_sided_die_expected_value_l839_83998

/-- The number of sides on the die -/
def num_sides : ℕ := 8

/-- The set of possible outcomes when rolling the die -/
def outcomes : Finset ℕ := Finset.range num_sides

/-- The expected value of rolling the die -/
def expected_value : ℚ := (Finset.sum outcomes (λ i => i + 1)) / num_sides

/-- Theorem: The expected value of rolling an eight-sided die with faces numbered from 1 to 8 is 4.5 -/
theorem eight_sided_die_expected_value :
  expected_value = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_eight_sided_die_expected_value_l839_83998


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l839_83935

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = Complex.abs (1 - Complex.I) + Complex.I) : 
  z.im = (1 - Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l839_83935


namespace NUMINAMATH_CALUDE_codemaster_secret_codes_l839_83968

/-- The number of slots in a CodeMaster secret code -/
def num_slots : ℕ := 5

/-- The total number of colors available -/
def total_colors : ℕ := 8

/-- The number of colors available for the last three slots (excluding black) -/
def colors_without_black : ℕ := 7

/-- The number of slots where black is allowed -/
def black_allowed_slots : ℕ := 2

/-- The number of different secret codes possible in the CodeMaster game -/
def num_secret_codes : ℕ := total_colors ^ black_allowed_slots * colors_without_black ^ (num_slots - black_allowed_slots)

theorem codemaster_secret_codes :
  num_secret_codes = 21952 := by
  sorry

end NUMINAMATH_CALUDE_codemaster_secret_codes_l839_83968


namespace NUMINAMATH_CALUDE_gcd_problem_l839_83929

theorem gcd_problem (a b : ℕ+) (h : Nat.gcd a.val b.val = 15) :
  Nat.gcd (12 * a.val) (18 * b.val) ≥ 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l839_83929


namespace NUMINAMATH_CALUDE_vector_sum_proof_l839_83951

/-- Given two vectors a and b in ℝ², prove that their sum is (2, 4) -/
theorem vector_sum_proof :
  let a : ℝ × ℝ := (-1, 6)
  let b : ℝ × ℝ := (3, -2)
  a + b = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l839_83951


namespace NUMINAMATH_CALUDE_f_is_even_l839_83986

-- Define the function
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem
theorem f_is_even : ∀ x : ℝ, f x = f (-x) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l839_83986


namespace NUMINAMATH_CALUDE_range_of_f_l839_83925

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 3

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.Icc (-5 : ℝ) 13, ∃ x ∈ Set.Icc 2 5, f x = y ∧
  ∀ x ∈ Set.Icc 2 5, f x ∈ Set.Icc (-5 : ℝ) 13 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l839_83925


namespace NUMINAMATH_CALUDE_school_ball_purchase_l839_83937

-- Define the unit prices
def soccer_price : ℝ := 40
def basketball_price : ℝ := 60

-- Define the total number of balls and max cost
def total_balls : ℕ := 200
def max_cost : ℝ := 9600

-- Theorem statement
theorem school_ball_purchase :
  -- Condition 1: Basketball price is 20 more than soccer price
  (basketball_price = soccer_price + 20) →
  -- Condition 2: Cost ratio of basketballs to soccer balls
  (6000 / basketball_price = 1.25 * (3200 / soccer_price)) →
  -- Condition 3 and 4 are implicitly used in the conclusion
  -- Conclusion: Correct prices and minimum number of soccer balls
  (soccer_price = 40 ∧ 
   basketball_price = 60 ∧ 
   ∀ m : ℕ, (m : ℝ) * soccer_price + (total_balls - m : ℝ) * basketball_price ≤ max_cost → m ≥ 120) :=
by sorry

end NUMINAMATH_CALUDE_school_ball_purchase_l839_83937


namespace NUMINAMATH_CALUDE_train_length_calculation_l839_83958

theorem train_length_calculation (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 300 ∧ crossing_time = 45 ∧ train_speed = 55.99999999999999 →
  2220 = train_speed * crossing_time - bridge_length :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l839_83958


namespace NUMINAMATH_CALUDE_ivan_walking_time_l839_83949

/-- Represents the journey of Ivan Ivanovich to work -/
structure Journey where
  /-- Walking speed of Ivan Ivanovich -/
  u : ℝ
  /-- Speed of the service car -/
  v : ℝ
  /-- Usual time it takes the service car to drive Ivan from home to work -/
  t : ℝ
  /-- Time Ivan Ivanovich walked -/
  T : ℝ
  /-- The service car speed is positive -/
  hv : v > 0
  /-- The walking speed is positive and less than the car speed -/
  hu : 0 < u ∧ u < v
  /-- The usual journey time is positive -/
  ht : t > 0
  /-- The walking time is positive and less than the usual journey time -/
  hT : 0 < T ∧ T < t
  /-- Ivan left 90 minutes earlier and arrived 20 minutes earlier -/
  h_time_diff : T + (t - T + 70) = t + 70
  /-- The distance walked equals the distance the car would travel in 10 minutes -/
  h_meeting_point : u * T = 10 * v

/-- Theorem stating that Ivan Ivanovich walked for 80 minutes -/
theorem ivan_walking_time (j : Journey) : j.T = 80 := by sorry

end NUMINAMATH_CALUDE_ivan_walking_time_l839_83949


namespace NUMINAMATH_CALUDE_logarithm_identity_l839_83964

theorem logarithm_identity (a : ℝ) (ha : a > 0) : 
  a^(Real.log (Real.log a)) - (Real.log a)^(Real.log a) = 0 :=
by sorry

end NUMINAMATH_CALUDE_logarithm_identity_l839_83964


namespace NUMINAMATH_CALUDE_congruence_solution_l839_83940

theorem congruence_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 14567 [MOD 16] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l839_83940


namespace NUMINAMATH_CALUDE_certain_percent_problem_l839_83924

theorem certain_percent_problem (P : ℝ) : 
  (P / 100) * 500 = (50 / 100) * 600 → P = 60 := by
  sorry

end NUMINAMATH_CALUDE_certain_percent_problem_l839_83924


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l839_83961

def A : Set ℝ := {x | 1 / x ≤ 0}
def B : Set ℝ := {x | x^2 - 1 < 0}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l839_83961


namespace NUMINAMATH_CALUDE_custom_op_example_l839_83919

-- Define the custom operation
def custom_op (a b : ℤ) : ℤ := a^2 - b

-- State the theorem
theorem custom_op_example : custom_op (custom_op 1 2) 4 = -3 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l839_83919


namespace NUMINAMATH_CALUDE_largest_sum_of_digits_l839_83914

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the decimal 0.abc -/
def DecimalABC (a b c : Digit) : ℚ :=
  (a.val * 100 + b.val * 10 + c.val : ℕ) / 1000

theorem largest_sum_of_digits (a b c : Digit) (y : ℕ) 
  (h1 : DecimalABC a b c = 1 / y)
  (h2 : 0 < y) (h3 : y ≤ 16) :
  a.val + b.val + c.val ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_largest_sum_of_digits_l839_83914


namespace NUMINAMATH_CALUDE_polygon_triangulation_l839_83904

theorem polygon_triangulation (n : ℕ) :
  (n ≥ 3) →  -- Ensure the polygon has at least 3 sides
  (n - 2 = 7) →  -- Number of triangles formed is n - 2, which equals 7
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_polygon_triangulation_l839_83904


namespace NUMINAMATH_CALUDE_no_valid_distribution_of_skittles_l839_83921

theorem no_valid_distribution_of_skittles : ¬ ∃ (F : ℕ+), 
  (14 - 3 * F.val ≥ 3) ∧ (14 - 3 * F.val) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_distribution_of_skittles_l839_83921


namespace NUMINAMATH_CALUDE_polynomial_roots_arithmetic_progression_l839_83952

theorem polynomial_roots_arithmetic_progression 
  (a b : ℝ) 
  (ha : a = 3 * Real.sqrt 3) 
  (hroots : ∀ (r s t : ℝ), 
    (r^3 - a*r^2 + b*r + a = 0 ∧ 
     s^3 - a*s^2 + b*s + a = 0 ∧ 
     t^3 - a*t^2 + b*t + a = 0) → 
    (r > 0 ∧ s > 0 ∧ t > 0) ∧ 
    ∃ (d : ℝ), (s = r + d ∧ t = r + 2*d) ∨ (s = r ∧ t = r)) : 
  b = 3 * (Real.sqrt 3 + 1) := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_arithmetic_progression_l839_83952


namespace NUMINAMATH_CALUDE_lunchroom_students_l839_83981

theorem lunchroom_students (tables : ℕ) (seated_per_table : ℕ) (standing : ℕ) : 
  tables = 34 → seated_per_table = 6 → standing = 15 →
  tables * seated_per_table + standing = 219 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_students_l839_83981


namespace NUMINAMATH_CALUDE_emily_sandra_orange_ratio_l839_83947

theorem emily_sandra_orange_ratio :
  ∀ (betty_oranges sandra_oranges emily_oranges : ℕ),
    betty_oranges = 12 →
    sandra_oranges = 3 * betty_oranges →
    emily_oranges = 252 →
    emily_oranges / sandra_oranges = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_emily_sandra_orange_ratio_l839_83947


namespace NUMINAMATH_CALUDE_semicircle_rotation_area_l839_83983

theorem semicircle_rotation_area (R : ℝ) (h : R > 0) :
  let α : ℝ := 20 * π / 180
  let semicircle_area : ℝ := π * R^2 / 2
  let rotated_area : ℝ := α * (2*R)^2 / 2
  rotated_area = 2 * π * R^2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_rotation_area_l839_83983


namespace NUMINAMATH_CALUDE_partitioned_rectangle_is_square_l839_83987

-- Define the structure for a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the structure for the partitioned rectangle
structure PartitionedRectangle where
  main : Rectangle
  p1 : Rectangle
  p2 : Rectangle
  p3 : Rectangle
  p4 : Rectangle
  p5 : Rectangle

-- Define the property of being a square
def isSquare (r : Rectangle) : Prop :=
  r.width = r.height

-- Define the property of having equal areas
def equalAreas (r1 r2 r3 r4 : Rectangle) : Prop :=
  r1.width * r1.height = r2.width * r2.height ∧
  r2.width * r2.height = r3.width * r3.height ∧
  r3.width * r3.height = r4.width * r4.height

-- Theorem statement
theorem partitioned_rectangle_is_square 
  (pr : PartitionedRectangle) 
  (h1 : isSquare pr.p5)
  (h2 : equalAreas pr.p1 pr.p2 pr.p3 pr.p4) :
  isSquare pr.main :=
sorry

end NUMINAMATH_CALUDE_partitioned_rectangle_is_square_l839_83987


namespace NUMINAMATH_CALUDE_sale_price_percentage_l839_83942

theorem sale_price_percentage (original_price : ℝ) (h : original_price > 0) :
  let first_sale_price := original_price * 0.5
  let final_price := first_sale_price * 0.9
  (final_price / original_price) * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_percentage_l839_83942


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_equivalence_l839_83912

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The vertex form of the quadratic function -/
def g (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem stating that f and g are equivalent -/
theorem quadratic_vertex_form_equivalence :
  ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_equivalence_l839_83912


namespace NUMINAMATH_CALUDE_prob_even_sum_three_dice_l839_83997

/-- The number of faces on each die -/
def num_faces : ℕ := 9

/-- The probability of rolling an even number on one die -/
def p_even : ℚ := 5/9

/-- The probability of rolling an odd number on one die -/
def p_odd : ℚ := 4/9

/-- The probability of getting an even sum when rolling three 9-sided dice -/
theorem prob_even_sum_three_dice : 
  (p_even^3) + 3 * (p_odd^2 * p_even) + 3 * (p_odd * p_even^2) = 665/729 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_three_dice_l839_83997


namespace NUMINAMATH_CALUDE_tens_digit_of_19_power_2021_l839_83920

theorem tens_digit_of_19_power_2021 : ∃ n : ℕ, 19^2021 ≡ 10*n + 1 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_19_power_2021_l839_83920


namespace NUMINAMATH_CALUDE_school_classes_count_l839_83928

theorem school_classes_count (sheets_per_class_per_day : ℕ) 
                              (total_sheets_per_week : ℕ) 
                              (school_days_per_week : ℕ) :
  sheets_per_class_per_day = 200 →
  total_sheets_per_week = 9000 →
  school_days_per_week = 5 →
  (total_sheets_per_week / (sheets_per_class_per_day * school_days_per_week) : ℕ) = 9 := by
sorry

end NUMINAMATH_CALUDE_school_classes_count_l839_83928


namespace NUMINAMATH_CALUDE_water_tank_capacity_l839_83996

theorem water_tank_capacity (initial_fraction : Rat) (final_fraction : Rat) (removed_liters : ℕ) 
  (h1 : initial_fraction = 2/3)
  (h2 : final_fraction = 1/3)
  (h3 : removed_liters = 20)
  (h4 : initial_fraction * tank_capacity - removed_liters = final_fraction * tank_capacity) :
  tank_capacity = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l839_83996


namespace NUMINAMATH_CALUDE_problem_statement_l839_83941

theorem problem_statement (p q : Prop) 
  (h1 : p ∨ q) 
  (h2 : ¬(p ∧ q)) 
  (h3 : ¬p) : 
  ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l839_83941


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l839_83936

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a) ≥ 3 ∧
  (1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a) = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l839_83936


namespace NUMINAMATH_CALUDE_problem_solution_l839_83974

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : a^b = b^a) (h2 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l839_83974


namespace NUMINAMATH_CALUDE_root_equation_problem_l839_83990

theorem root_equation_problem (a b c m : ℝ) : 
  (a^2 - 4*a + m = 0 ∧ b^2 - 4*b + m = 0) →
  (b^2 - 8*b + 5*m = 0 ∧ c^2 - 8*c + 5*m = 0) →
  m = 0 ∨ m = 3 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l839_83990


namespace NUMINAMATH_CALUDE_value_equals_scientific_notation_l839_83955

/-- Represents the value in billion yuan -/
def value : ℝ := 24953

/-- Represents the scientific notation coefficient -/
def coefficient : ℝ := 2.4953

/-- Represents the scientific notation exponent -/
def exponent : ℕ := 13

/-- Theorem stating that the given value in billion yuan is equal to its scientific notation representation -/
theorem value_equals_scientific_notation : value * 10^9 = coefficient * 10^exponent := by
  sorry

end NUMINAMATH_CALUDE_value_equals_scientific_notation_l839_83955


namespace NUMINAMATH_CALUDE_cube_frame_impossible_without_cuts_minimum_cuts_for_cube_frame_l839_83917

-- Define the wire length and cube edge length
def wire_length : ℝ := 120
def cube_edge_length : ℝ := 10

-- Define the number of edges in a cube
def cube_edges : ℕ := 12

-- Define the number of vertices in a cube
def cube_vertices : ℕ := 8

-- Define the number of edges meeting at each vertex of a cube
def edges_per_vertex : ℕ := 3

-- Theorem 1: It's impossible to create the cube frame without cuts
theorem cube_frame_impossible_without_cuts :
  ¬ ∃ (path : List ℝ), 
    (path.length = cube_edges) ∧ 
    (path.sum = wire_length) ∧
    (∀ edge ∈ path, edge = cube_edge_length) :=
sorry

-- Theorem 2: The minimum number of cuts required is 3
theorem minimum_cuts_for_cube_frame :
  (cube_vertices / 2 : ℕ) - 1 = 3 :=
sorry

end NUMINAMATH_CALUDE_cube_frame_impossible_without_cuts_minimum_cuts_for_cube_frame_l839_83917


namespace NUMINAMATH_CALUDE_shelter_blocks_count_l839_83905

/-- Calculates the number of blocks needed for a rectangular shelter --/
def shelter_blocks (length width height : ℕ) : ℕ :=
  let exterior_volume := length * width * height
  let interior_length := length - 2
  let interior_width := width - 2
  let interior_height := height - 2
  let interior_volume := interior_length * interior_width * interior_height
  exterior_volume - interior_volume

/-- Proves that the number of blocks for a shelter with given dimensions is 528 --/
theorem shelter_blocks_count :
  shelter_blocks 14 12 6 = 528 := by
  sorry

end NUMINAMATH_CALUDE_shelter_blocks_count_l839_83905


namespace NUMINAMATH_CALUDE_no_solution_for_inequalities_l839_83909

theorem no_solution_for_inequalities :
  ¬∃ x : ℝ, (4 * x^2 + 7 * x - 2 < 0) ∧ (3 * x - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_inequalities_l839_83909


namespace NUMINAMATH_CALUDE_average_salary_proof_l839_83931

def workshop_problem (total_workers : ℕ) (technicians : ℕ) (avg_salary_technicians : ℚ) (avg_salary_others : ℚ) : Prop :=
  let non_technicians : ℕ := total_workers - technicians
  let total_salary_technicians : ℚ := technicians * avg_salary_technicians
  let total_salary_others : ℚ := non_technicians * avg_salary_others
  let total_salary : ℚ := total_salary_technicians + total_salary_others
  let avg_salary_all : ℚ := total_salary / total_workers
  avg_salary_all = 8000

theorem average_salary_proof :
  workshop_problem 28 7 14000 6000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_proof_l839_83931


namespace NUMINAMATH_CALUDE_sum_of_smallest_solutions_l839_83943

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the equation
def equation (x : ℝ) : Prop := x - floor x = 2 / (floor x : ℝ)

-- Define the set of positive solutions
def positive_solutions : Set ℝ := {x : ℝ | x > 0 ∧ equation x}

-- State the theorem
theorem sum_of_smallest_solutions :
  ∃ (s₁ s₂ s₃ : ℝ),
    s₁ ∈ positive_solutions ∧
    s₂ ∈ positive_solutions ∧
    s₃ ∈ positive_solutions ∧
    (∀ x ∈ positive_solutions, x ≤ s₁ ∨ x ≤ s₂ ∨ x ≤ s₃) ∧
    s₁ + s₂ + s₃ = 13 + 17 / 30 :=
sorry

end NUMINAMATH_CALUDE_sum_of_smallest_solutions_l839_83943


namespace NUMINAMATH_CALUDE_amit_left_after_three_days_l839_83944

/-- The number of days Amit takes to complete the work alone -/
def amit_days : ℕ := 15

/-- The number of days Ananthu takes to complete the work alone -/
def ananthu_days : ℕ := 45

/-- The total number of days taken to complete the work -/
def total_days : ℕ := 39

/-- The number of days Amit worked before leaving -/
def amit_worked_days : ℕ := 3

theorem amit_left_after_three_days :
  ∃ (w : ℝ), w > 0 ∧
  amit_worked_days * (w / amit_days) + (total_days - amit_worked_days) * (w / ananthu_days) = w :=
sorry

end NUMINAMATH_CALUDE_amit_left_after_three_days_l839_83944


namespace NUMINAMATH_CALUDE_nonagon_side_length_l839_83969

/-- A regular nonagon with perimeter 171 cm has sides of length 19 cm -/
theorem nonagon_side_length : ∀ (perimeter side_length : ℝ),
  perimeter = 171 →
  side_length * 9 = perimeter →
  side_length = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_nonagon_side_length_l839_83969


namespace NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l839_83953

theorem sqrt_six_div_sqrt_two_eq_sqrt_three :
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l839_83953


namespace NUMINAMATH_CALUDE_A_closed_under_mult_l839_83970

/-- The set A of quadratic forms over integers -/
def A : Set ℤ := {n : ℤ | ∃ (a b k : ℤ), n = a^2 + k*a*b + b^2}

/-- A is closed under multiplication -/
theorem A_closed_under_mult :
  ∀ (x y : ℤ), x ∈ A → y ∈ A → (x * y) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_A_closed_under_mult_l839_83970


namespace NUMINAMATH_CALUDE_exponent_calculation_l839_83977

theorem exponent_calculation : (64 : ℝ)^(1/4) * (16 : ℝ)^(3/8) = 8 := by
  have h1 : (64 : ℝ) = 2^6 := by sorry
  have h2 : (16 : ℝ) = 2^4 := by sorry
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l839_83977


namespace NUMINAMATH_CALUDE_rectangle_count_num_rectangles_nat_l839_83979

/-- The number of rectangles formed in a rectangle ABCD with additional points and lines -/
theorem rectangle_count (m n : ℕ) : 
  (m + 2) * (m + 1) * (n + 2) * (n + 1) / 4 = 
  (Nat.choose (m + 2) 2) * (Nat.choose (n + 2) 2) :=
by sorry

/-- The formula for the number of rectangles formed -/
def num_rectangles (m n : ℕ) : ℕ := (m + 2) * (m + 1) * (n + 2) * (n + 1) / 4

/-- The number of rectangles is always a natural number -/
theorem num_rectangles_nat (m n : ℕ) : 
  ∃ k : ℕ, num_rectangles m n = k :=
by sorry

end NUMINAMATH_CALUDE_rectangle_count_num_rectangles_nat_l839_83979


namespace NUMINAMATH_CALUDE_expression_simplification_l839_83972

theorem expression_simplification (x y : ℝ) :
  (x^3 - 9*x*y^2) / (9*y^2 + x^2) * ((x + 3*y) / (x^2 - 3*x*y) + (x - 3*y) / (x^2 + 3*x*y)) = x - 3*y :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l839_83972


namespace NUMINAMATH_CALUDE_same_window_probability_l839_83999

theorem same_window_probability (n : ℕ) (h : n = 3) :
  (n : ℝ) / (n * n : ℝ) = 1 / 3 := by
  sorry

#check same_window_probability

end NUMINAMATH_CALUDE_same_window_probability_l839_83999


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l839_83910

theorem floor_negative_seven_fourths :
  ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l839_83910


namespace NUMINAMATH_CALUDE_worker_payment_schedule_l839_83901

/-- Proves that the amount to return for each day not worked is $25 --/
theorem worker_payment_schedule (total_days : Nat) (days_not_worked : Nat) (payment_per_day : Nat) (total_earnings : Nat) :
  total_days = 30 →
  days_not_worked = 24 →
  payment_per_day = 100 →
  total_earnings = 0 →
  (total_days - days_not_worked) * payment_per_day = days_not_worked * 25 := by
  sorry

end NUMINAMATH_CALUDE_worker_payment_schedule_l839_83901


namespace NUMINAMATH_CALUDE_unequal_grandchildren_probability_l839_83907

/-- The number of grandchildren -/
def n : ℕ := 12

/-- The probability of a child being male (or female) -/
def p : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters -/
def prob_unequal : ℚ := 793/1024

theorem unequal_grandchildren_probability :
  (1 : ℚ) - (n.choose (n/2) : ℚ) / (2^n : ℚ) = prob_unequal :=
sorry

end NUMINAMATH_CALUDE_unequal_grandchildren_probability_l839_83907


namespace NUMINAMATH_CALUDE_square_side_length_l839_83992

/-- Given a rectangle with width 36 cm and length 64 cm, and a square whose perimeter
    equals the rectangle's perimeter, prove that the side length of the square is 50 cm. -/
theorem square_side_length (rectangle_width rectangle_length : ℝ)
                            (square_side : ℝ)
                            (h1 : rectangle_width = 36)
                            (h2 : rectangle_length = 64)
                            (h3 : 4 * square_side = 2 * (rectangle_width + rectangle_length)) :
  square_side = 50 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l839_83992


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l839_83984

/-- The quadratic equation bx^2 - 12x + 9 = 0 has exactly one solution when b = 4 -/
theorem quadratic_one_solution (b : ℝ) : 
  (∃! x, b * x^2 - 12 * x + 9 = 0) ↔ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l839_83984


namespace NUMINAMATH_CALUDE_min_cost_for_nine_hamburgers_l839_83934

/-- Represents the cost of hamburgers under a "buy two, get one free" promotion -/
def hamburger_cost (unit_price : ℕ) (quantity : ℕ) : ℕ :=
  let sets := quantity / 3
  let remainder := quantity % 3
  sets * (2 * unit_price) + remainder * unit_price

/-- Theorem stating the minimum cost for 9 hamburgers under the given promotion -/
theorem min_cost_for_nine_hamburgers :
  hamburger_cost 10 9 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_nine_hamburgers_l839_83934


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l839_83950

/-- Theorem: For a parabola y = ax^2 where a > 0, if the distance from the focus to the directrix is 1, then a = 1/2 -/
theorem parabola_focus_directrix_distance (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, y = a * x^2) → -- Parabola equation
  (∃ p : ℝ, p = 1 ∧ p = 1 / (2 * a)) → -- Distance from focus to directrix is 1
  a = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l839_83950


namespace NUMINAMATH_CALUDE_B_power_2017_l839_83962

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_2017 : B^2017 = B := by sorry

end NUMINAMATH_CALUDE_B_power_2017_l839_83962


namespace NUMINAMATH_CALUDE_cube_sum_geq_triple_product_l839_83906

theorem cube_sum_geq_triple_product (x y z : ℝ) : x^3 + y^3 + z^3 ≥ 3*x*y*z := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_geq_triple_product_l839_83906


namespace NUMINAMATH_CALUDE_fort_block_count_l839_83933

/-- Calculates the number of one-foot cubical blocks required to construct a rectangular fort -/
def fort_blocks (length width height : ℕ) (wall_thickness : ℕ) : ℕ :=
  length * width * height - 
  (length - 2 * wall_thickness) * (width - 2 * wall_thickness) * (height - wall_thickness)

/-- Proves that a fort with given dimensions requires 430 blocks -/
theorem fort_block_count : fort_blocks 15 12 6 1 = 430 := by
  sorry

end NUMINAMATH_CALUDE_fort_block_count_l839_83933


namespace NUMINAMATH_CALUDE_square_expansion_l839_83971

theorem square_expansion (n : ℕ) (h : ∃ k : ℕ, k > 0 ∧ (n + k)^2 - n^2 = 47) : n = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_expansion_l839_83971


namespace NUMINAMATH_CALUDE_sum_of_digits_power_of_nine_l839_83988

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits of 9^n is greater than 9 for all n ≥ 3 -/
theorem sum_of_digits_power_of_nine (n : ℕ) (h : n ≥ 3) : sum_of_digits (9^n) > 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_of_nine_l839_83988


namespace NUMINAMATH_CALUDE_rotation_equivalence_l839_83994

/-- 
Given that:
1. A point A is rotated 550 degrees clockwise about a center point B to reach point C.
2. The same point A is rotated x degrees counterclockwise about the same center point B to reach point C.
3. x is less than 360 degrees.

Prove that x equals 170 degrees.
-/
theorem rotation_equivalence (x : ℝ) 
  (h1 : x < 360) 
  (h2 : (550 % 360 : ℝ) + x = 360) : x = 170 :=
by sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l839_83994


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l839_83993

theorem expression_simplification_and_evaluation (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (1 - x / (x + 1)) / ((x^2 - 2*x + 1) / (x^2 - 1)) = 1 / (x - 1) :=
sorry

theorem expression_evaluation_at_3 :
  (1 - 3 / (3 + 1)) / ((3^2 - 2*3 + 1) / (3^2 - 1)) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l839_83993


namespace NUMINAMATH_CALUDE_special_function_properties_l839_83932

/-- A function satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y - 3) ∧
  (∀ x : ℝ, x > 0 → f x < 3)

theorem special_function_properties (f : ℝ → ℝ) (hf : SpecialFunction f) :
  (f 0 = 3) ∧
  (∀ x y : ℝ, x < y → f y < f x) ∧
  (∀ x : ℝ, (∀ t : ℝ, t ∈ Set.Ioo 2 4 → 
    f ((t - 2) * |x - 4|) + 3 > f (t^2 + 8) + f (5 - 4*t)) →
    x ∈ Set.Icc (-5/2) (21/2)) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l839_83932


namespace NUMINAMATH_CALUDE_ten_passengers_five_stops_l839_83954

/-- The number of ways for passengers to get off a bus -/
def bus_stop_combinations (num_passengers : ℕ) (num_stops : ℕ) : ℕ :=
  num_stops ^ num_passengers

/-- Theorem: 10 passengers and 5 stops result in 5^10 combinations -/
theorem ten_passengers_five_stops :
  bus_stop_combinations 10 5 = 5^10 := by
  sorry

end NUMINAMATH_CALUDE_ten_passengers_five_stops_l839_83954


namespace NUMINAMATH_CALUDE_triangle_third_side_prime_l839_83908

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def valid_third_side (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_third_side_prime (a b : ℕ) (ha : a = 7) (hb : b = 11) :
  ∃ (c : ℕ), is_prime c ∧ valid_third_side a b c ↔ 
  c = 5 ∨ c = 7 ∨ c = 11 ∨ c = 13 ∨ c = 17 :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_prime_l839_83908
