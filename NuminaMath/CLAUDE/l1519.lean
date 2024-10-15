import Mathlib

namespace NUMINAMATH_CALUDE_sum_first_150_remainder_l1519_151966

theorem sum_first_150_remainder (n : Nat) (sum : Nat) : 
  n = 150 → 
  sum = n * (n + 1) / 2 → 
  sum % 11300 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_150_remainder_l1519_151966


namespace NUMINAMATH_CALUDE_equality_iff_inequality_holds_l1519_151929

theorem equality_iff_inequality_holds (x y : ℝ) : x = y ↔ x * y ≥ ((x + y) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_equality_iff_inequality_holds_l1519_151929


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_quadratic_l1519_151947

theorem smallest_prime_divisor_of_quadratic : 
  ∃ (p : ℕ), Prime p ∧ 
  (∃ (n : ℤ), (n^2 + n + 11).natAbs % p = 0) ∧
  (∀ (q : ℕ), Prime q → q < p → 
    ∀ (m : ℤ), (m^2 + m + 11).natAbs % q ≠ 0) ∧
  p = 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_quadratic_l1519_151947


namespace NUMINAMATH_CALUDE_charlotte_tuesday_poodles_l1519_151903

/-- Represents the schedule and constraints for Charlotte's dog walking --/
structure DogWalkingSchedule where
  monday_poodles : Nat
  monday_chihuahuas : Nat
  wednesday_labradors : Nat
  poodle_time : Nat
  chihuahua_time : Nat
  labrador_time : Nat
  total_time : Nat

/-- Calculates the number of poodles Charlotte can walk on Tuesday --/
def tuesday_poodles (schedule : DogWalkingSchedule) : Nat :=
  let monday_time := schedule.monday_poodles * schedule.poodle_time + 
                     schedule.monday_chihuahuas * schedule.chihuahua_time
  let tuesday_chihuahua_time := schedule.monday_chihuahuas * schedule.chihuahua_time
  let wednesday_time := schedule.wednesday_labradors * schedule.labrador_time
  let available_time := schedule.total_time - monday_time - tuesday_chihuahua_time - wednesday_time
  available_time / schedule.poodle_time

/-- Theorem stating that given the schedule constraints, Charlotte can walk 4 poodles on Tuesday --/
theorem charlotte_tuesday_poodles : 
  ∀ (schedule : DogWalkingSchedule), 
  schedule.monday_poodles = 4 ∧ 
  schedule.monday_chihuahuas = 2 ∧ 
  schedule.wednesday_labradors = 4 ∧ 
  schedule.poodle_time = 2 ∧ 
  schedule.chihuahua_time = 1 ∧ 
  schedule.labrador_time = 3 ∧ 
  schedule.total_time = 32 → 
  tuesday_poodles schedule = 4 := by
  sorry


end NUMINAMATH_CALUDE_charlotte_tuesday_poodles_l1519_151903


namespace NUMINAMATH_CALUDE_volunteers_selection_theorem_l1519_151970

theorem volunteers_selection_theorem :
  let n : ℕ := 5  -- Total number of volunteers
  let k : ℕ := 2  -- Number of people to be sent to each location
  let locations : ℕ := 2  -- Number of locations
  Nat.choose n k * Nat.choose (n - k) k = 30 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_selection_theorem_l1519_151970


namespace NUMINAMATH_CALUDE_three_lines_intersection_l1519_151920

/-- Three lines intersect at a single point if and only if k = -7 -/
theorem three_lines_intersection (x y k : ℝ) : 
  (∃! p : ℝ × ℝ, (y = 7*x + 5 ∧ y = -3*x - 35 ∧ y = 4*x + k) → p.1 = x ∧ p.2 = y) ↔ k = -7 :=
by sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l1519_151920


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1519_151986

def f (x : ℝ) : ℝ := x * abs x

theorem inequality_equivalence (m : ℝ) : 
  (∀ x ≥ 1, f (x + m) + m * f x < 0) ↔ m ∈ Set.Iic (-1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1519_151986


namespace NUMINAMATH_CALUDE_triangle_side_decomposition_l1519_151939

/-- Given a triangle with side lengths a, b, and c, there exist positive numbers x, y, and z
    such that a = y + z, b = x + z, and c = x + y -/
theorem triangle_side_decomposition (a b c : ℝ) (h_triangle : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ a = y + z ∧ b = x + z ∧ c = x + y :=
sorry

end NUMINAMATH_CALUDE_triangle_side_decomposition_l1519_151939


namespace NUMINAMATH_CALUDE_weighted_power_inequality_l1519_151946

theorem weighted_power_inequality (a b c : ℝ) (n : ℕ) (p q r : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^r * b^p * c^q + a^q * b^r * c^p := by
  sorry

end NUMINAMATH_CALUDE_weighted_power_inequality_l1519_151946


namespace NUMINAMATH_CALUDE_order_of_rational_numbers_l1519_151933

theorem order_of_rational_numbers
  (a b c d : ℚ)
  (sum_eq : a + b = c + d)
  (ineq_1 : a + d < b + c)
  (ineq_2 : c < d) :
  b > d ∧ d > c ∧ c > a :=
sorry

end NUMINAMATH_CALUDE_order_of_rational_numbers_l1519_151933


namespace NUMINAMATH_CALUDE_triangular_prism_theorem_l1519_151900

theorem triangular_prism_theorem (V k : ℝ) (S H : Fin 4 → ℝ) : 
  (∀ i : Fin 4, S i = (i.val + 1 : ℕ) * k) →
  (∀ i : Fin 4, V = (1/3) * S i * H i) →
  H 0 + 2 * H 1 + 3 * H 2 + 4 * H 3 = 3 * V / k :=
by sorry

end NUMINAMATH_CALUDE_triangular_prism_theorem_l1519_151900


namespace NUMINAMATH_CALUDE_dolls_in_small_box_l1519_151963

theorem dolls_in_small_box :
  let big_box_count : ℕ := 5
  let big_box_dolls : ℕ := 7
  let small_box_count : ℕ := 9
  let total_dolls : ℕ := 71
  let small_box_dolls : ℕ := (total_dolls - big_box_count * big_box_dolls) / small_box_count
  small_box_dolls = 4 := by sorry

end NUMINAMATH_CALUDE_dolls_in_small_box_l1519_151963


namespace NUMINAMATH_CALUDE_intersection_implies_sum_of_slopes_is_five_l1519_151953

/-- Given two sets A and B in R^2, defined by linear equations,
    prove that if their intersection is a single point (2, 5),
    then the sum of their slopes is 5. -/
theorem intersection_implies_sum_of_slopes_is_five 
  (a b : ℝ) 
  (A : Set (ℝ × ℝ)) 
  (B : Set (ℝ × ℝ)) 
  (h1 : A = {p : ℝ × ℝ | p.2 = a * p.1 + 1})
  (h2 : B = {p : ℝ × ℝ | p.2 = p.1 + b})
  (h3 : A ∩ B = {(2, 5)}) : 
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_of_slopes_is_five_l1519_151953


namespace NUMINAMATH_CALUDE_club_officer_selection_l1519_151911

theorem club_officer_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  n * (n - 1) * (n - 2) = 6840 :=
sorry

end NUMINAMATH_CALUDE_club_officer_selection_l1519_151911


namespace NUMINAMATH_CALUDE_unique_cuddly_number_l1519_151988

/-- A two-digit positive integer is cuddly if it equals the sum of its nonzero tens digit and the square of its units digit. -/
def IsCuddly (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n = (n / 10) + (n % 10)^2

/-- There exists exactly one cuddly two-digit positive integer. -/
theorem unique_cuddly_number : ∃! n : ℕ, IsCuddly n :=
  sorry

end NUMINAMATH_CALUDE_unique_cuddly_number_l1519_151988


namespace NUMINAMATH_CALUDE_symmetric_function_zero_l1519_151989

/-- A function f: ℝ → ℝ satisfying specific symmetry properties -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2*x + 2) = -f (-2*x - 2)) ∧ 
  (∀ x, f (x + 1) = f (-x + 1))

/-- Theorem stating that for a function with the given symmetry properties, f(4) = 0 -/
theorem symmetric_function_zero (f : ℝ → ℝ) (h : SymmetricFunction f) : f 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_zero_l1519_151989


namespace NUMINAMATH_CALUDE_complex_number_equal_parts_l1519_151979

theorem complex_number_equal_parts (b : ℝ) : 
  let z : ℂ := (3 - b * Complex.I) / (2 + Complex.I)
  z.re = z.im ↔ b = -9 := by
sorry

end NUMINAMATH_CALUDE_complex_number_equal_parts_l1519_151979


namespace NUMINAMATH_CALUDE_bowTie_equation_solution_l1519_151916

-- Define the bow tie operation
noncomputable def bowTie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem bowTie_equation_solution (h : ℝ) :
  bowTie 5 h = 7 → h = 2 := by sorry

end NUMINAMATH_CALUDE_bowTie_equation_solution_l1519_151916


namespace NUMINAMATH_CALUDE_A_single_element_A_at_most_one_element_l1519_151973

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 - x + a + 2 = 0}

-- Theorem 1: A contains only one element iff a ∈ {0, -2+√5, -2-√5}
theorem A_single_element (a : ℝ) :
  (∃! x, x ∈ A a) ↔ a = 0 ∨ a = -2 + Real.sqrt 5 ∨ a = -2 - Real.sqrt 5 :=
sorry

-- Theorem 2: A contains at most one element iff a ∈ (-∞, -2-√5] ∪ {0} ∪ [-2+√5, +∞)
theorem A_at_most_one_element (a : ℝ) :
  (∀ x y, x ∈ A a → y ∈ A a → x = y) ↔
  a ≤ -2 - Real.sqrt 5 ∨ a = 0 ∨ a ≥ -2 + Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_A_single_element_A_at_most_one_element_l1519_151973


namespace NUMINAMATH_CALUDE_profit_starts_third_year_option_one_more_profitable_l1519_151941

/-- Represents the financial model of the fishing boat -/
structure FishingBoat where
  initialCost : ℕ
  firstYearExpenses : ℕ
  annualExpenseIncrease : ℕ
  annualIncome : ℕ

/-- Calculates the cumulative profit after n years -/
def cumulativeProfit (boat : FishingBoat) (n : ℕ) : ℤ :=
  n * boat.annualIncome - boat.initialCost - boat.firstYearExpenses
    - (n - 1) * boat.annualExpenseIncrease * n / 2

/-- Calculates the average profit after n years -/
def averageProfit (boat : FishingBoat) (n : ℕ) : ℚ :=
  (cumulativeProfit boat n : ℚ) / n

/-- The boat configuration from the problem -/
def problemBoat : FishingBoat :=
  { initialCost := 980000
    firstYearExpenses := 120000
    annualExpenseIncrease := 40000
    annualIncome := 500000 }

theorem profit_starts_third_year :
  ∀ n : ℕ, n < 3 → cumulativeProfit problemBoat n ≤ 0
  ∧ cumulativeProfit problemBoat 3 > 0 := by sorry

theorem option_one_more_profitable :
  let optionOne := cumulativeProfit problemBoat 7 + 260000
  let optionTwo := cumulativeProfit problemBoat 10 + 80000
  optionOne = optionTwo ∧ 7 < 10 := by sorry

end NUMINAMATH_CALUDE_profit_starts_third_year_option_one_more_profitable_l1519_151941


namespace NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l1519_151942

theorem sqrt_a_div_sqrt_b (a b : ℝ) (h : (1/3)^2 + (1/4)^2 = (37*a/100/b) * ((1/5)^2 + (1/6)^2)) :
  Real.sqrt a / Real.sqrt b = 50/19 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l1519_151942


namespace NUMINAMATH_CALUDE_cuboid_volume_example_l1519_151927

/-- A cuboid with given base area and height -/
structure Cuboid where
  base_area : ℝ
  height : ℝ

/-- The volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.base_area * c.height

/-- Theorem: The volume of a cuboid with base area 14 cm² and height 13 cm is 182 cm³ -/
theorem cuboid_volume_example : 
  let c : Cuboid := { base_area := 14, height := 13 }
  volume c = 182 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_example_l1519_151927


namespace NUMINAMATH_CALUDE_container_water_percentage_l1519_151984

theorem container_water_percentage (capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) 
  (h1 : capacity = 120)
  (h2 : added_water = 54)
  (h3 : final_fraction = 3/4) :
  let initial_percentage := (final_fraction * capacity - added_water) / capacity * 100
  initial_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_container_water_percentage_l1519_151984


namespace NUMINAMATH_CALUDE_parabola_area_theorem_l1519_151938

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given a parabola y^2 = 2px (p > 0) with a point A(m,1) on it,
    and a point B on the directrix such that AB is perpendicular to the directrix,
    if the area of triangle AOB (where O is the origin) is 1/2, then p = 1. -/
theorem parabola_area_theorem (C : Parabola) (A : Point) (m : ℝ) :
  A.x = m →
  A.y = 1 →
  A.y^2 = 2 * C.p * A.x →
  (∃ B : Point, B.y = -C.p/2 ∧ (A.x - B.x) * (A.y - B.y) = 0) →
  (1/2 * m * 1 + 1/2 * (C.p/2) * 1 = 1/2) →
  C.p = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_area_theorem_l1519_151938


namespace NUMINAMATH_CALUDE_trig_simplification_l1519_151934

theorem trig_simplification :
  (Real.sin (30 * π / 180) + Real.sin (40 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (40 * π / 180)) =
  Real.tan (35 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trig_simplification_l1519_151934


namespace NUMINAMATH_CALUDE_minute_hand_angle_for_110_minutes_l1519_151980

/-- The angle turned by the minute hand when the hour hand moves for a given time -/
def minuteHandAngle (hours : ℝ) (minutes : ℝ) : ℝ :=
  -(hours * 360 + minutes * 6)

/-- Theorem: When the hour hand moves for 1 hour and 50 minutes, 
    the angle turned by the minute hand is -660° -/
theorem minute_hand_angle_for_110_minutes : 
  minuteHandAngle 1 50 = -660 := by sorry

end NUMINAMATH_CALUDE_minute_hand_angle_for_110_minutes_l1519_151980


namespace NUMINAMATH_CALUDE_power_function_not_in_second_quadrant_l1519_151969

def f (x : ℝ) : ℝ := x

theorem power_function_not_in_second_quadrant :
  (∀ x : ℝ, x < 0 → f x ≤ 0) ∧
  (∀ x : ℝ, f x = x) :=
sorry

end NUMINAMATH_CALUDE_power_function_not_in_second_quadrant_l1519_151969


namespace NUMINAMATH_CALUDE_instances_in_one_hour_l1519_151971

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The interval in seconds at which the device records data -/
def recording_interval : ℕ := 5

/-- Proves that the number of 5-second intervals in one hour is equal to 720 -/
theorem instances_in_one_hour :
  (seconds_per_minute * minutes_per_hour) / recording_interval = 720 := by
  sorry

end NUMINAMATH_CALUDE_instances_in_one_hour_l1519_151971


namespace NUMINAMATH_CALUDE_system_solvable_iff_a_in_range_l1519_151945

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*(a - x - y) = 64 ∧
  y = 8 * Real.sin (x - 2*b) - 6 * Real.cos (x - 2*b)

-- Theorem statement
theorem system_solvable_iff_a_in_range :
  ∀ a : ℝ, (∃ b x y : ℝ, system a b x y) ↔ -18 ≤ a ∧ a ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_system_solvable_iff_a_in_range_l1519_151945


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l1519_151959

def A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -5; -4, 3]
def A_inv : Matrix (Fin 2) (Fin 2) ℤ := !![3, 5; 4, 7]

theorem matrix_inverse_proof :
  IsUnit (Matrix.det A) ∧ A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l1519_151959


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1519_151982

theorem quadratic_transformation (x : ℝ) :
  x^2 - 6*x - 1 = 0 ↔ (x + 3)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1519_151982


namespace NUMINAMATH_CALUDE_function_maximum_value_l1519_151999

theorem function_maximum_value (x : ℝ) (h : x < 1/2) :
  ∃ M : ℝ, M = -1 ∧ ∀ y : ℝ, y = 2*x + 1/(2*x - 1) → y ≤ M := by
  sorry

end NUMINAMATH_CALUDE_function_maximum_value_l1519_151999


namespace NUMINAMATH_CALUDE_min_value_and_sum_l1519_151967

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

-- State the theorem
theorem min_value_and_sum (a b : ℝ) : 
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) ∧ m = 2) ∧
  (a^2 + b^2 = 2 → 1 / (a^2 + 1) + 4 / (b^2 + 1) ≥ 9/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_sum_l1519_151967


namespace NUMINAMATH_CALUDE_luke_spent_3_dollars_per_week_l1519_151912

def luke_problem (lawn_income weed_income total_weeks : ℕ) : Prop :=
  let total_income := lawn_income + weed_income
  total_income / total_weeks = 3

theorem luke_spent_3_dollars_per_week :
  luke_problem 9 18 9 := by
  sorry

end NUMINAMATH_CALUDE_luke_spent_3_dollars_per_week_l1519_151912


namespace NUMINAMATH_CALUDE_unique_solution_l1519_151974

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y)) = f x * f y + f x + f y + x * y

/-- The theorem stating that f(1) = 1 is the only solution -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1519_151974


namespace NUMINAMATH_CALUDE_product_cube_square_l1519_151965

theorem product_cube_square : ((-1 : ℤ)^3) * ((-2 : ℤ)^2) = -4 := by sorry

end NUMINAMATH_CALUDE_product_cube_square_l1519_151965


namespace NUMINAMATH_CALUDE_union_A_B_equals_open_interval_l1519_151972

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x}

-- Theorem statement
theorem union_A_B_equals_open_interval :
  A ∪ B = Set.Ioo (-2 : ℝ) 4 := by sorry

end NUMINAMATH_CALUDE_union_A_B_equals_open_interval_l1519_151972


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisor_l1519_151957

def is_valid_seven_digit_number (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧
  (n / 100 % 10 = 2 * (n / 1000000)) ∧
  (n / 10 % 10 = 2 * (n / 100000 % 10)) ∧
  (n % 10 = 2 * (n / 10000 % 10)) ∧
  (n / 1000 % 10 = 0)

theorem smallest_five_digit_divisor :
  ∃ (n : ℕ), is_valid_seven_digit_number n ∧ n % 10002 = 0 ∧
  ∀ (m : ℕ), 10000 ≤ m ∧ m < 10002 → ¬(∃ (k : ℕ), is_valid_seven_digit_number k ∧ k % m = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisor_l1519_151957


namespace NUMINAMATH_CALUDE_quadratic_inequality_sum_l1519_151922

/-- Given a quadratic inequality ax^2 - 3ax - 6 < 0 with solution set {x | x < 1 or x > b}, 
    prove that a + b = -1 -/
theorem quadratic_inequality_sum (a b : ℝ) : 
  (∀ x, ax^2 - 3*a*x - 6 < 0 ↔ x < 1 ∨ x > b) → 
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_sum_l1519_151922


namespace NUMINAMATH_CALUDE_walnut_trees_in_park_l1519_151996

theorem walnut_trees_in_park (current_trees : ℕ) : 
  (current_trees + 44 = 77) → current_trees = 33 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_in_park_l1519_151996


namespace NUMINAMATH_CALUDE_c_investment_is_2000_l1519_151948

/-- Represents the investment and profit distribution in a business partnership --/
structure BusinessPartnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  b_profit_share : ℕ
  a_c_profit_diff : ℕ

/-- Calculates the profit share for a given investment --/
def profit_share (investment total_investment total_profit : ℕ) : ℕ :=
  (investment * total_profit) / total_investment

/-- Theorem stating that C's investment is 2000 given the problem conditions --/
theorem c_investment_is_2000 (bp : BusinessPartnership)
  (h1 : bp.a_investment = 6000)
  (h2 : bp.b_investment = 8000)
  (h3 : bp.b_profit_share = 1000)
  (h4 : bp.a_c_profit_diff = 500)
  (h5 : bp.b_profit_share = profit_share bp.b_investment (bp.a_investment + bp.b_investment + bp.c_investment) bp.total_profit)
  (h6 : bp.a_c_profit_diff = profit_share bp.a_investment (bp.a_investment + bp.b_investment + bp.c_investment) bp.total_profit -
                             profit_share bp.c_investment (bp.a_investment + bp.b_investment + bp.c_investment) bp.total_profit) :
  bp.c_investment = 2000 := by
  sorry


end NUMINAMATH_CALUDE_c_investment_is_2000_l1519_151948


namespace NUMINAMATH_CALUDE_union_equals_universe_l1519_151917

def U : Finset ℕ := {2, 3, 4, 5, 6}
def M : Finset ℕ := {3, 4, 5}
def N : Finset ℕ := {2, 4, 5, 6}

theorem union_equals_universe : M ∪ N = U := by
  sorry

end NUMINAMATH_CALUDE_union_equals_universe_l1519_151917


namespace NUMINAMATH_CALUDE_two_digit_square_l1519_151968

/-- Given distinct digits a, b, c, prove that the two-digit number 'ab' is 21 -/
theorem two_digit_square (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c →
  b = 1 →
  10 * a + b < 100 →
  100 * c + 10 * c + b > 300 →
  (10 * a + b)^2 = 100 * c + 10 * c + b →
  10 * a + b = 21 := by
sorry

end NUMINAMATH_CALUDE_two_digit_square_l1519_151968


namespace NUMINAMATH_CALUDE_incircle_center_locus_is_mid_distance_strip_l1519_151958

/-- Represents a line in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents the incircle of a triangle -/
structure Incircle :=
  (center : Point)
  (radius : ℝ)

/-- Three parallel lines on a plane -/
def parallel_lines : List Line := sorry

/-- A triangle with vertices on the parallel lines -/
def triangle_on_lines (t : Triangle) : Prop := sorry

/-- The incircle of a triangle -/
def incircle_of_triangle (t : Triangle) : Incircle := sorry

/-- The strip between the mid-distances of outer and middle lines -/
def mid_distance_strip : Set Point := sorry

/-- The geometric locus of incircle centers -/
def incircle_center_locus : Set Point := sorry

/-- Theorem: The geometric locus of the centers of incircles of triangles with vertices on three parallel lines
    is the strip bound by lines parallel and in the mid-distance between the outer and mid lines -/
theorem incircle_center_locus_is_mid_distance_strip :
  incircle_center_locus = mid_distance_strip := by sorry

end NUMINAMATH_CALUDE_incircle_center_locus_is_mid_distance_strip_l1519_151958


namespace NUMINAMATH_CALUDE_store_profit_theorem_l1519_151928

/-- Represents the store's sales and profit model -/
structure Store where
  initial_profit_per_unit : ℝ
  initial_daily_sales : ℝ
  sales_increase_per_yuan : ℝ

/-- Calculates the new daily sales after a price reduction -/
def new_daily_sales (s : Store) (price_reduction : ℝ) : ℝ :=
  s.initial_daily_sales + s.sales_increase_per_yuan * price_reduction

/-- Calculates the daily profit after a price reduction -/
def daily_profit (s : Store) (price_reduction : ℝ) : ℝ :=
  (s.initial_profit_per_unit - price_reduction) * (new_daily_sales s price_reduction)

/-- The store model based on the given conditions -/
def my_store : Store := {
  initial_profit_per_unit := 60,
  initial_daily_sales := 40,
  sales_increase_per_yuan := 2
}

theorem store_profit_theorem (s : Store) :
  (new_daily_sales s 10 = 60) ∧
  (∃ x : ℝ, x = 30 ∧ daily_profit s x = 3000) ∧
  (¬ ∃ y : ℝ, daily_profit s y = 3300) := by
  sorry

#check store_profit_theorem my_store

end NUMINAMATH_CALUDE_store_profit_theorem_l1519_151928


namespace NUMINAMATH_CALUDE_aunt_angela_nieces_l1519_151914

theorem aunt_angela_nieces (total_jellybeans : ℕ) (num_nephews : ℕ) (jellybeans_per_child : ℕ) 
  (h1 : total_jellybeans = 70)
  (h2 : num_nephews = 3)
  (h3 : jellybeans_per_child = 14) :
  total_jellybeans / jellybeans_per_child - num_nephews = 2 :=
by sorry

end NUMINAMATH_CALUDE_aunt_angela_nieces_l1519_151914


namespace NUMINAMATH_CALUDE_ryan_study_time_l1519_151935

/-- Ryan's daily study hours for English -/
def english_hours : ℕ := 6

/-- Ryan's daily study hours for Chinese -/
def chinese_hours : ℕ := 7

/-- Number of days Ryan studies -/
def study_days : ℕ := 5

/-- Total study hours for both languages over the given period -/
def total_study_hours : ℕ := (english_hours + chinese_hours) * study_days

theorem ryan_study_time : total_study_hours = 65 := by
  sorry

end NUMINAMATH_CALUDE_ryan_study_time_l1519_151935


namespace NUMINAMATH_CALUDE_maintenance_cost_third_year_l1519_151978

/-- Represents the maintenance cost function for factory equipment -/
def maintenance_cost (x : ℝ) : ℝ := 0.8 * x + 1.5

/-- Proves that the maintenance cost for equipment in its third year is 3.9 ten thousand yuan -/
theorem maintenance_cost_third_year :
  maintenance_cost 3 = 3.9 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_cost_third_year_l1519_151978


namespace NUMINAMATH_CALUDE_last_three_average_l1519_151997

theorem last_three_average (numbers : List ℝ) : 
  numbers.length = 7 → 
  numbers.sum / 7 = 60 → 
  (numbers.take 4).sum / 4 = 55 → 
  (numbers.drop 4).sum / 3 = 200 / 3 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l1519_151997


namespace NUMINAMATH_CALUDE_kite_plot_area_l1519_151998

/-- The scale of the map in miles per inch -/
def scale : ℚ := 200 / 2

/-- The length of the first diagonal on the map in inches -/
def diagonal1_map : ℚ := 2

/-- The length of the second diagonal on the map in inches -/
def diagonal2_map : ℚ := 10

/-- The area of a kite given its diagonals -/
def kite_area (d1 d2 : ℚ) : ℚ := (1 / 2) * d1 * d2

/-- The theorem stating that the area of the kite-shaped plot is 100,000 square miles -/
theorem kite_plot_area : 
  kite_area (diagonal1_map * scale) (diagonal2_map * scale) = 100000 := by
  sorry

end NUMINAMATH_CALUDE_kite_plot_area_l1519_151998


namespace NUMINAMATH_CALUDE_rain_forest_animal_count_l1519_151943

/-- The number of animals in the Rain Forest exhibit -/
def rain_forest_animals : ℕ := 7

/-- The number of animals in the Reptile House -/
def reptile_house_animals : ℕ := 16

/-- Theorem stating the relationship between the number of animals in the Rain Forest exhibit and the Reptile House -/
theorem rain_forest_animal_count : 
  reptile_house_animals = 3 * rain_forest_animals - 5 ∧ 
  rain_forest_animals = 7 := by
  sorry

end NUMINAMATH_CALUDE_rain_forest_animal_count_l1519_151943


namespace NUMINAMATH_CALUDE_texas_maine_plate_difference_l1519_151923

/-- The number of possible choices for a letter in a license plate. -/
def letter_choices : ℕ := 26

/-- The number of possible choices for a number in a license plate. -/
def number_choices : ℕ := 10

/-- The number of possible license plates in Texas format (LLNNNNL). -/
def texas_plates : ℕ := letter_choices^3 * number_choices^4

/-- The number of possible license plates in Maine format (LLLNNN). -/
def maine_plates : ℕ := letter_choices^3 * number_choices^3

/-- The difference in the number of possible license plates between Texas and Maine. -/
def plate_difference : ℕ := texas_plates - maine_plates

theorem texas_maine_plate_difference :
  plate_difference = 158184000 := by
  sorry

end NUMINAMATH_CALUDE_texas_maine_plate_difference_l1519_151923


namespace NUMINAMATH_CALUDE_cos_to_sin_shift_l1519_151960

theorem cos_to_sin_shift (x : ℝ) : 
  3 * Real.cos (2 * x - π / 4) = 3 * Real.sin (2 * (x + π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_cos_to_sin_shift_l1519_151960


namespace NUMINAMATH_CALUDE_yellow_preference_l1519_151975

/-- Proves that 9 students like yellow best given the survey conditions --/
theorem yellow_preference (total_students : ℕ) (total_girls : ℕ) 
  (h_total : total_students = 30)
  (h_girls : total_girls = 18)
  (h_green : total_students / 2 = total_students - total_students / 2)
  (h_pink : total_girls / 3 = total_girls - 2 * (total_girls / 3)) :
  total_students - (total_students / 2 + total_girls / 3) = 9 := by
  sorry

#check yellow_preference

end NUMINAMATH_CALUDE_yellow_preference_l1519_151975


namespace NUMINAMATH_CALUDE_percentage_of_invalid_votes_l1519_151908

theorem percentage_of_invalid_votes
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 80 / 100)
  (h3 : candidate_a_votes = 380800) :
  (total_votes - (candidate_a_votes / candidate_a_percentage)) / total_votes = 15 / 100 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_invalid_votes_l1519_151908


namespace NUMINAMATH_CALUDE_rectangular_screen_area_l1519_151925

/-- Proves that a rectangular screen with width-to-height ratio of 3:2 and diagonal length of 65 cm has an area of 1950 cm². -/
theorem rectangular_screen_area (width height diagonal : ℝ) : 
  width / height = 3 / 2 →
  width^2 + height^2 = diagonal^2 →
  diagonal = 65 →
  width * height = 1950 := by
sorry

end NUMINAMATH_CALUDE_rectangular_screen_area_l1519_151925


namespace NUMINAMATH_CALUDE_xy_geq_ac_plus_bd_l1519_151992

theorem xy_geq_ac_plus_bd (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (hx : x = Real.sqrt (a^2 + b^2)) (hy : y = Real.sqrt (c^2 + d^2)) : x * y ≥ a * c + b * d :=
by sorry

end NUMINAMATH_CALUDE_xy_geq_ac_plus_bd_l1519_151992


namespace NUMINAMATH_CALUDE_min_box_value_l1519_151924

theorem min_box_value (a b Box : ℤ) : 
  a ≠ b ∧ a ≠ Box ∧ b ≠ Box →
  (∀ x, (a*x + b)*(b*x + a) = 30*x^2 + Box*x + 30) →
  a * b = 30 →
  Box = a^2 + b^2 →
  (∀ a' b' Box' : ℤ, 
    a' ≠ b' ∧ a' ≠ Box' ∧ b' ≠ Box' →
    (∀ x, (a'*x + b')*(b'*x + a') = 30*x^2 + Box'*x + 30) →
    a' * b' = 30 →
    Box' = a'^2 + b'^2 →
    Box ≤ Box') →
  Box = 61 := by
sorry

end NUMINAMATH_CALUDE_min_box_value_l1519_151924


namespace NUMINAMATH_CALUDE_odd_divides_power_factorial_minus_one_l1519_151906

theorem odd_divides_power_factorial_minus_one (n : ℕ) (h_pos : 0 < n) (h_odd : Odd n) :
  n ∣ 2^(n.factorial) - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_divides_power_factorial_minus_one_l1519_151906


namespace NUMINAMATH_CALUDE_unique_plane_through_parallel_lines_l1519_151937

-- Define a type for points in 3D space
variable (Point : Type)

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define a type for planes in 3D space
variable (Plane : Type)

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Define a relation for a line being contained in a plane
variable (contains : Plane → Line → Prop)

-- Theorem: Through two parallel lines, there is exactly one plane
theorem unique_plane_through_parallel_lines 
  (l1 l2 : Line) 
  (h : parallel l1 l2) : 
  ∃! p : Plane, contains p l1 ∧ contains p l2 :=
sorry

end NUMINAMATH_CALUDE_unique_plane_through_parallel_lines_l1519_151937


namespace NUMINAMATH_CALUDE_min_tiles_to_cover_min_tiles_for_given_dimensions_l1519_151904

/-- Represents the dimensions of a rectangle in inches -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle in square inches -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Theorem: The minimum number of 3x5 inch tiles needed to cover a 3x4 foot region is 116 -/
theorem min_tiles_to_cover (tile : Rectangle) (region : Rectangle) : ℕ :=
  let tileArea := area tile
  let regionArea := area { length := feetToInches region.length, width := feetToInches region.width }
  ((regionArea + tileArea - 1) / tileArea : ℕ)

/-- Main theorem statement -/
theorem min_tiles_for_given_dimensions : 
  min_tiles_to_cover { length := 3, width := 5 } { length := 3, width := 4 } = 116 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_to_cover_min_tiles_for_given_dimensions_l1519_151904


namespace NUMINAMATH_CALUDE_sunflower_majority_on_day_two_l1519_151940

/-- Represents the proportion of sunflower seeds in the feeder on a given day -/
def sunflower_proportion (day : ℕ) : ℝ :=
  1 - (0.7 : ℝ) ^ day

/-- The day when more than half of the seeds are sunflower seeds -/
def target_day : ℕ := 2

theorem sunflower_majority_on_day_two :
  sunflower_proportion target_day > (1/2 : ℝ) :=
by sorry

#check sunflower_majority_on_day_two

end NUMINAMATH_CALUDE_sunflower_majority_on_day_two_l1519_151940


namespace NUMINAMATH_CALUDE_f_30_value_l1519_151926

/-- A function from positive integers to positive integers satisfying certain properties -/
def special_function (f : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ+, f (n + 1) > f n) ∧ 
  (∀ m n : ℕ+, f (m * n) = f m * f n) ∧
  (∀ m n : ℕ+, m ≠ n → m ^ m.val = n ^ n.val → (f m = n ∨ f n = m))

theorem f_30_value (f : ℕ+ → ℕ+) (h : special_function f) : f 30 = 900 := by
  sorry

end NUMINAMATH_CALUDE_f_30_value_l1519_151926


namespace NUMINAMATH_CALUDE_function_zeros_imply_k_range_l1519_151964

open Real

theorem function_zeros_imply_k_range (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, f x = (log x) / x - k * x) →
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 1/ℯ ≤ x₁ ∧ x₁ ≤ ℯ^2 ∧ 1/ℯ ≤ x₂ ∧ x₂ ≤ ℯ^2 ∧ f x₁ = 0 ∧ f x₂ = 0) →
  2/ℯ^4 ≤ k ∧ k < 1/(2*ℯ) :=
by sorry

end NUMINAMATH_CALUDE_function_zeros_imply_k_range_l1519_151964


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l1519_151921

theorem at_least_one_greater_than_one (a b : ℝ) :
  (a + b > 2 → max a b > 1) ∧ (a * b > 1 → max a b > 1) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l1519_151921


namespace NUMINAMATH_CALUDE_log_product_equals_two_thirds_l1519_151919

theorem log_product_equals_two_thirds : 
  Real.log 2 / Real.log 3 * Real.log 9 / Real.log 8 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_two_thirds_l1519_151919


namespace NUMINAMATH_CALUDE_range_of_x_minus_2y_l1519_151936

theorem range_of_x_minus_2y (x y : ℝ) 
  (hx : -1 ≤ x ∧ x < 2) (hy : 0 < y ∧ y ≤ 1) : 
  -3 ≤ x - 2*y ∧ x - 2*y < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_minus_2y_l1519_151936


namespace NUMINAMATH_CALUDE_banana_permutations_l1519_151985

/-- The number of permutations of a multiset -/
def multiset_permutations (n : ℕ) (frequencies : List ℕ) : ℕ :=
  Nat.factorial n / (List.prod (List.map Nat.factorial frequencies))

/-- Theorem: The number of distinct permutations of BANANA is 60 -/
theorem banana_permutations :
  multiset_permutations 6 [3, 2, 1] = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_l1519_151985


namespace NUMINAMATH_CALUDE_horner_method_v2_l1519_151976

def horner_polynomial (x : ℝ) : ℝ := 2*x^6 + 3*x^5 + 5*x^3 + 6*x^2 + 7*x + 8

def horner_v0 : ℝ := 2
def horner_v1 (x : ℝ) : ℝ := horner_v0 * x + 3
def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x + 0

theorem horner_method_v2 :
  horner_v2 2 = 14 ∧ horner_polynomial 2 = horner_v2 2 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v2_l1519_151976


namespace NUMINAMATH_CALUDE_triangle_and_circle_symmetry_l1519_151913

-- Define the point A
def A : ℝ × ℝ := (4, -3)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the vector OA
def OA : ℝ × ℝ := A

-- Define the vector AB
def AB : ℝ × ℝ := (6, 8)

-- Define point B
def B : ℝ × ℝ := (A.1 + AB.1, A.2 + AB.2)

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 10

-- Theorem statement
theorem triangle_and_circle_symmetry :
  -- A is the right-angle vertex of triangle OAB
  (OA.1 * AB.1 + OA.2 * AB.2 = 0) →
  -- |AB| = 2|OA|
  (AB.1^2 + AB.2^2 = 4 * (OA.1^2 + OA.2^2)) →
  -- The ordinate of point B is greater than 0
  (B.2 > 0) →
  -- AB has coordinates (6, 8)
  (AB = (6, 8)) ∧
  -- The equation of the symmetric circle is correct
  (∀ x y, symmetric_circle x y ↔
    ∃ x' y', original_circle x' y' ∧
      -- x and y are symmetric to x' and y' with respect to line OB
      ((x + x') / 2 = B.1 * ((y + y') / 2) / B.2)) :=
sorry

end NUMINAMATH_CALUDE_triangle_and_circle_symmetry_l1519_151913


namespace NUMINAMATH_CALUDE_number_exceeds_fraction_by_40_l1519_151931

theorem number_exceeds_fraction_by_40 (x : ℝ) : x = (3 / 8) * x + 40 → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeds_fraction_by_40_l1519_151931


namespace NUMINAMATH_CALUDE_parabola_distance_theorem_l1519_151905

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem parabola_distance_theorem 
  (A : ℝ × ℝ) -- Point A
  (h1 : parabola A.1 A.2) -- A is on the parabola
  (h2 : ‖A - focus‖ = ‖B - focus‖) -- |AF| = |BF|
  : ‖A - B‖ = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_parabola_distance_theorem_l1519_151905


namespace NUMINAMATH_CALUDE_pen_count_l1519_151907

theorem pen_count (num_pencils : ℕ) (max_students : ℕ) (num_pens : ℕ) : 
  num_pencils = 520 →
  max_students = 40 →
  num_pencils % max_students = 0 →
  num_pens % max_students = 0 →
  (num_pencils / max_students = num_pens / max_students) →
  num_pens = 520 := by
sorry

end NUMINAMATH_CALUDE_pen_count_l1519_151907


namespace NUMINAMATH_CALUDE_theater_seats_l1519_151995

/-- The number of seats in the nth row of the theater -/
def seats_in_row (n : ℕ) : ℕ := 2 * n + 26

/-- The total number of seats in the theater -/
def total_seats (rows : ℕ) : ℕ :=
  (seats_in_row 1 + seats_in_row rows) * rows / 2

/-- Theorem stating the total number of seats in the theater -/
theorem theater_seats :
  total_seats 20 = 940 := by sorry

end NUMINAMATH_CALUDE_theater_seats_l1519_151995


namespace NUMINAMATH_CALUDE_min_sum_squares_l1519_151954

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1519_151954


namespace NUMINAMATH_CALUDE_largest_gold_coins_distribution_l1519_151951

theorem largest_gold_coins_distribution (n : ℕ) : 
  (n % 13 = 3) →  -- Condition: 3 people receive an extra coin after equal distribution
  (n < 150) →     -- Condition: Total coins less than 150
  (∀ m : ℕ, (m % 13 = 3) ∧ (m < 150) → m ≤ n) →  -- n is the largest number satisfying conditions
  n = 146 :=      -- Conclusion: The largest number of coins is 146
by sorry

end NUMINAMATH_CALUDE_largest_gold_coins_distribution_l1519_151951


namespace NUMINAMATH_CALUDE_max_value_a_l1519_151955

theorem max_value_a (a b c d : ℤ) 
  (h1 : a < 2 * b) 
  (h2 : b < 3 * c) 
  (h3 : c < 4 * d) 
  (h4 : d < 100) : 
  a ≤ 2367 ∧ ∃ (a' b' c' d' : ℤ), a' = 2367 ∧ a' < 2 * b' ∧ b' < 3 * c' ∧ c' < 4 * d' ∧ d' < 100 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_a_l1519_151955


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l1519_151949

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (is_arithmetic : arithmetic_sequence a)
  (first_term : a 1 = 2)
  (sum_of_three : a 1 + a 2 + a 3 = 12) :
  a 6 = 12 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l1519_151949


namespace NUMINAMATH_CALUDE_min_value_expression_l1519_151994

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / (2 * a)) + (a / (b + 1)) ≥ 5/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1519_151994


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l1519_151987

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The problem statement -/
theorem base_conversion_subtraction :
  let base7_number := [3, 0, 4, 2, 5]  -- 52403 in base 7 (least significant digit first)
  let base5_number := [5, 4, 3, 0, 2]  -- 20345 in base 5 (least significant digit first)
  toBase10 base7_number 7 - toBase10 base5_number 5 = 11540 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l1519_151987


namespace NUMINAMATH_CALUDE_tax_revenue_change_l1519_151950

theorem tax_revenue_change 
  (original_tax : ℝ) 
  (original_consumption : ℝ) 
  (tax_reduction_rate : ℝ) 
  (consumption_increase_rate : ℝ) 
  (h1 : tax_reduction_rate = 0.16) 
  (h2 : consumption_increase_rate = 0.15) : 
  let new_tax := original_tax * (1 - tax_reduction_rate)
  let new_consumption := original_consumption * (1 + consumption_increase_rate)
  let original_revenue := original_tax * original_consumption
  let new_revenue := new_tax * new_consumption
  (new_revenue - original_revenue) / original_revenue = -0.034 :=
by sorry

end NUMINAMATH_CALUDE_tax_revenue_change_l1519_151950


namespace NUMINAMATH_CALUDE_problem_1_l1519_151977

theorem problem_1 : (-1)^3 + |2 - Real.sqrt 5| + (Real.pi / 2 - 1.57)^0 + Real.sqrt 20 = 3 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1519_151977


namespace NUMINAMATH_CALUDE_probability_of_specific_selection_l1519_151915

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 8

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 7

/-- The number of jackets in the drawer -/
def num_jackets : ℕ := 3

/-- The total number of clothing items in the drawer -/
def total_items : ℕ := num_shirts + num_shorts + num_socks + num_jackets

/-- The number of items to be selected -/
def items_to_select : ℕ := 4

theorem probability_of_specific_selection :
  (num_shirts : ℚ) * num_shorts * num_socks * num_jackets /
  (total_items.choose items_to_select) = 144 / 1815 :=
sorry

end NUMINAMATH_CALUDE_probability_of_specific_selection_l1519_151915


namespace NUMINAMATH_CALUDE_sum_of_specific_repeating_decimals_l1519_151902

/-- Definition of a repeating decimal with a 3-digit repetend -/
def repeating_decimal (a b c : ℕ) : ℚ := (a * 100 + b * 10 + c) / 999

/-- The sum of two specific repeating decimals equals 161/999 -/
theorem sum_of_specific_repeating_decimals : 
  repeating_decimal 1 3 7 + repeating_decimal 0 2 4 = 161 / 999 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_repeating_decimals_l1519_151902


namespace NUMINAMATH_CALUDE_projection_of_two_vectors_l1519_151909

/-- Given two vectors that project to the same vector, find the projection --/
theorem projection_of_two_vectors (v₁ v₂ v : ℝ × ℝ) (h : v ≠ (0, 0)) :
  let p := (14/73, 214/73)
  (∃ (k₁ k₂ : ℝ), v₁ - k₁ • v = p ∧ v₂ - k₂ • v = p) →
  v₁ = (5, -2) →
  v₂ = (2, 6) →
  (∃ (k : ℝ), v₁ - k • v = p ∧ v₂ - k • v = p) :=
by sorry


end NUMINAMATH_CALUDE_projection_of_two_vectors_l1519_151909


namespace NUMINAMATH_CALUDE_students_remaining_l1519_151991

theorem students_remaining (groups : ℕ) (students_per_group : ℕ) (left_early : ℕ) : 
  groups = 5 → students_per_group = 12 → left_early = 7 → 
  groups * students_per_group - left_early = 53 := by
  sorry

end NUMINAMATH_CALUDE_students_remaining_l1519_151991


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l1519_151910

theorem maintenance_check_increase (original_time new_time : ℕ) 
  (h1 : original_time = 30) 
  (h2 : new_time = 60) : 
  (new_time - original_time) / original_time * 100 = 100 :=
by sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l1519_151910


namespace NUMINAMATH_CALUDE_probability_continuous_stripe_is_one_fourth_l1519_151990

/-- A regular tetrahedron with painted stripes on its faces -/
structure StripedTetrahedron where
  /-- The number of faces of the tetrahedron -/
  num_faces : Nat
  /-- The number of possible stripe configurations per face -/
  stripe_configs : Nat
  /-- The probability of a continuous stripe pattern given all faces have intersecting stripes -/
  prob_continuous_intersecting : ℚ

/-- The probability of at least one continuous stripe pattern encircling the tetrahedron -/
def probability_continuous_stripe (t : StripedTetrahedron) : ℚ :=
  2 * (1 / 2) ^ 3

theorem probability_continuous_stripe_is_one_fourth (t : StripedTetrahedron) 
    (h1 : t.num_faces = 4)
    (h2 : t.stripe_configs = 2)
    (h3 : t.prob_continuous_intersecting = 1 / 16) :
  probability_continuous_stripe t = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_continuous_stripe_is_one_fourth_l1519_151990


namespace NUMINAMATH_CALUDE_stacy_berries_l1519_151932

theorem stacy_berries (steve_initial : ℕ) (steve_takes : ℕ) (difference : ℕ) : 
  steve_initial = 21 → steve_takes = 4 → difference = 7 → 
  ∃ stacy_initial : ℕ, stacy_initial = 32 ∧ 
    steve_initial + steve_takes = stacy_initial - difference :=
by sorry

end NUMINAMATH_CALUDE_stacy_berries_l1519_151932


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l1519_151981

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n : ℕ,
  n < 1000 ∧ 
  5 ∣ n ∧ 
  6 ∣ n ∧ 
  ∀ m : ℕ, m < 1000 → 5 ∣ m → 6 ∣ m → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l1519_151981


namespace NUMINAMATH_CALUDE_production_time_calculation_l1519_151930

/-- The number of days it takes for a given number of machines to produce a certain amount of product P -/
def production_time (num_machines : ℕ) (units : ℝ) : ℝ := sorry

/-- The number of units produced by a given number of machines in a certain number of days -/
def units_produced (num_machines : ℕ) (days : ℝ) : ℝ := sorry

theorem production_time_calculation :
  let d := production_time 5 x
  let x : ℝ := units_produced 5 d
  units_produced 20 2 = 2 * x →
  d = 4 := by sorry

end NUMINAMATH_CALUDE_production_time_calculation_l1519_151930


namespace NUMINAMATH_CALUDE_wickets_before_match_l1519_151952

/-- Represents a cricketer's bowling statistics -/
structure BowlingStats where
  wickets : ℕ
  runs : ℕ
  average : ℚ

/-- Calculates the new average after a match -/
def newAverage (stats : BowlingStats) (newWickets : ℕ) (newRuns : ℕ) : ℚ :=
  (stats.runs + newRuns) / (stats.wickets + newWickets)

/-- Theorem: The cricketer had taken 85 wickets before the match -/
theorem wickets_before_match (stats : BowlingStats) : 
  stats.average = 12.4 →
  newAverage stats 5 26 = 12 →
  stats.wickets = 85 := by
  sorry

#check wickets_before_match

end NUMINAMATH_CALUDE_wickets_before_match_l1519_151952


namespace NUMINAMATH_CALUDE_cube_coloring_ways_octahedron_coloring_ways_l1519_151961

-- Define the number of colors for each shape
def cube_colors : ℕ := 6
def octahedron_colors : ℕ := 8

-- Define the number of faces for each shape
def cube_faces : ℕ := 6
def octahedron_faces : ℕ := 8

-- Theorem for coloring the cube
theorem cube_coloring_ways :
  (cube_colors.factorial / (cube_colors - cube_faces).factorial) = 30 := by
  sorry

-- Theorem for coloring the octahedron
theorem octahedron_coloring_ways :
  (octahedron_colors.factorial / (octahedron_colors - octahedron_faces).factorial) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_cube_coloring_ways_octahedron_coloring_ways_l1519_151961


namespace NUMINAMATH_CALUDE_project_completion_time_l1519_151956

/-- The number of days B takes to complete the project alone -/
def B_days : ℝ := 30

/-- The total number of days the project takes when A and B work together, with A quitting 15 days before completion -/
def total_days : ℝ := 21

/-- The number of days before project completion that A quits -/
def A_quit_days : ℝ := 15

/-- The number of days A takes to complete the project alone -/
def A_days : ℝ := 20

theorem project_completion_time :
  A_days = 20 :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l1519_151956


namespace NUMINAMATH_CALUDE_f_at_one_plus_sqrt_two_l1519_151944

-- Define the function f
def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1

-- State the theorem
theorem f_at_one_plus_sqrt_two : f (1 + Real.sqrt 2) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_at_one_plus_sqrt_two_l1519_151944


namespace NUMINAMATH_CALUDE_parking_lot_spaces_l1519_151983

theorem parking_lot_spaces (total_spaces : ℕ) (full_ratio compact_ratio : ℕ) 
  (h1 : total_spaces = 450)
  (h2 : full_ratio = 11)
  (h3 : compact_ratio = 4) :
  (total_spaces * full_ratio) / (full_ratio + compact_ratio) = 330 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_spaces_l1519_151983


namespace NUMINAMATH_CALUDE_same_number_on_four_dice_l1519_151962

theorem same_number_on_four_dice : 
  let n : ℕ := 6  -- number of sides on each die
  let k : ℕ := 4  -- number of dice
  (1 : ℚ) / n^(k-1) = (1 : ℚ) / 216 :=
by sorry

end NUMINAMATH_CALUDE_same_number_on_four_dice_l1519_151962


namespace NUMINAMATH_CALUDE_ellipse_equation_l1519_151918

/-- An ellipse centered at the origin -/
structure Ellipse where
  equation : ℝ → ℝ → Prop

/-- A hyperbola -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop

/-- The eccentricity of a conic section -/
def eccentricity (c : ℝ) : ℝ := c

/-- Theorem: Given an ellipse centered at the origin sharing a common focus with 
    the hyperbola 2x^2 - 2y^2 = 1, and their eccentricities being reciprocal to 
    each other, the equation of the ellipse is x^2/2 + y^2 = 1 -/
theorem ellipse_equation 
  (e : Ellipse) 
  (h : Hyperbola) 
  (h_eq : h.equation = fun x y => 2 * x^2 - 2 * y^2 = 1) 
  (common_focus : ∃ (f : ℝ × ℝ), f ∈ {p | ∃ (x y : ℝ), p = (x, y) ∧ h.equation x y} ∧ 
                                 f ∈ {p | ∃ (x y : ℝ), p = (x, y) ∧ e.equation x y})
  (reciprocal_eccentricity : ∃ (e_ecc h_ecc : ℝ), 
    eccentricity e_ecc * eccentricity h_ecc = 1) :
  e.equation = fun x y => x^2 / 2 + y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1519_151918


namespace NUMINAMATH_CALUDE_george_carries_two_buckets_l1519_151993

/-- The number of buckets George can carry each round -/
def george_buckets : ℕ := 2

/-- The number of buckets Harry can carry each round -/
def harry_buckets : ℕ := 3

/-- The total number of buckets needed to fill the pool -/
def total_buckets : ℕ := 110

/-- The number of rounds needed to fill the pool -/
def total_rounds : ℕ := 22

theorem george_carries_two_buckets :
  george_buckets = 2 ∧
  harry_buckets * total_rounds + george_buckets * total_rounds = total_buckets :=
sorry

end NUMINAMATH_CALUDE_george_carries_two_buckets_l1519_151993


namespace NUMINAMATH_CALUDE_tan_seven_pi_fourth_l1519_151901

theorem tan_seven_pi_fourth : Real.tan (7 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_fourth_l1519_151901
