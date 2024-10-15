import Mathlib

namespace NUMINAMATH_CALUDE_triangle_altitude_l2847_284787

theorem triangle_altitude (A b : ℝ) (h : A = 900 ∧ b = 45) :
  ∃ h : ℝ, A = (1/2) * b * h ∧ h = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_l2847_284787


namespace NUMINAMATH_CALUDE_total_items_eq_137256_l2847_284745

/-- The number of old women going to Rome -/
def num_women : ℕ := 7

/-- The number of mules each woman has -/
def mules_per_woman : ℕ := 7

/-- The number of bags each mule carries -/
def bags_per_mule : ℕ := 7

/-- The number of loaves each bag contains -/
def loaves_per_bag : ℕ := 7

/-- The number of knives each loaf contains -/
def knives_per_loaf : ℕ := 7

/-- The number of sheaths each knife is in -/
def sheaths_per_knife : ℕ := 7

/-- The total number of items -/
def total_items : ℕ := 
  num_women +
  (num_women * mules_per_woman) +
  (num_women * mules_per_woman * bags_per_mule) +
  (num_women * mules_per_woman * bags_per_mule * loaves_per_bag) +
  (num_women * mules_per_woman * bags_per_mule * loaves_per_bag * knives_per_loaf) +
  (num_women * mules_per_woman * bags_per_mule * loaves_per_bag * knives_per_loaf * sheaths_per_knife)

theorem total_items_eq_137256 : total_items = 137256 := by
  sorry

end NUMINAMATH_CALUDE_total_items_eq_137256_l2847_284745


namespace NUMINAMATH_CALUDE_box_volume_problem_l2847_284726

theorem box_volume_problem :
  ∃! (x : ℕ), x > 3 ∧ (x + 3) * (x - 3) * (x^2 + 9) < 500 := by sorry

end NUMINAMATH_CALUDE_box_volume_problem_l2847_284726


namespace NUMINAMATH_CALUDE_robbery_participants_l2847_284759

-- Define the suspects
variable (A B V G : Prop)

-- A: Alexey is guilty
-- B: Boris is guilty
-- V: Veniamin is guilty
-- G: Grigory is guilty

-- Define the conditions
variable (h1 : ¬G → (B ∧ ¬A))
variable (h2 : V → (¬A ∧ ¬B))
variable (h3 : G → B)
variable (h4 : B → (A ∨ V))

-- Theorem statement
theorem robbery_participants : A ∧ B ∧ G ∧ ¬V := by
  sorry

end NUMINAMATH_CALUDE_robbery_participants_l2847_284759


namespace NUMINAMATH_CALUDE_equation_solutions_l2847_284723

def is_solution (x y : ℤ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 1987

def solution_count : ℕ := 5

theorem equation_solutions :
  (∃! (s : Finset (ℤ × ℤ)), s.card = solution_count ∧
    ∀ (p : ℤ × ℤ), p ∈ s ↔ is_solution p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2847_284723


namespace NUMINAMATH_CALUDE_fitness_center_member_ratio_l2847_284746

theorem fitness_center_member_ratio 
  (f m : ℕ) -- number of female and male members
  (avg_female : ℕ := 45) -- average age of female members
  (avg_male : ℕ := 30) -- average age of male members
  (avg_all : ℕ := 35) -- average age of all members
  (h : (f * avg_female + m * avg_male) / (f + m) = avg_all) : 
  f / m = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_fitness_center_member_ratio_l2847_284746


namespace NUMINAMATH_CALUDE_measles_cases_1987_l2847_284791

/-- Calculates the number of measles cases in a given year assuming a linear decrease --/
def measlesCases (initialYear finalYear targetYear : ℕ) (initialCases finalCases : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let targetYears := targetYear - initialYear
  let totalDecrease := initialCases - finalCases
  let decrease := (targetYears * totalDecrease) / totalYears
  initialCases - decrease

/-- Theorem stating that the number of measles cases in 1987 would be 112,875 --/
theorem measles_cases_1987 :
  measlesCases 1960 1996 1987 450000 500 = 112875 := by
  sorry

#eval measlesCases 1960 1996 1987 450000 500

end NUMINAMATH_CALUDE_measles_cases_1987_l2847_284791


namespace NUMINAMATH_CALUDE_min_value_theorem_l2847_284763

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * (2^x) + Real.log 2 * (8^y) = Real.log 2) : 
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log 2 * (2^a) + Real.log 2 * (8^b) = Real.log 2 → 
    1/x + 1/(3*y) ≤ 1/a + 1/(3*b)) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ Real.log 2 * (2^x) + Real.log 2 * (8^y) = Real.log 2 ∧ 
    1/x + 1/(3*y) = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2847_284763


namespace NUMINAMATH_CALUDE_acute_triangle_theorem_l2847_284773

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : a^2 = b^2 + c^2 - 2*b*c*Real.cos A

/-- Main theorem about the acute triangle -/
theorem acute_triangle_theorem (t : AcuteTriangle) :
  (Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A) →
  (t.c = Real.sqrt 7 ∧ t.a * t.b = 6) →
  (t.C = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7) :=
by sorry


end NUMINAMATH_CALUDE_acute_triangle_theorem_l2847_284773


namespace NUMINAMATH_CALUDE_cans_collected_l2847_284749

/-- Proves that the number of cans collected is 144 given the recycling rates and total money received -/
theorem cans_collected (can_rate : ℚ) (newspaper_rate : ℚ) (newspaper_collected : ℚ) (total_money : ℚ) :
  can_rate = 1/24 →
  newspaper_rate = 3/10 →
  newspaper_collected = 20 →
  total_money = 12 →
  ∃ (cans : ℚ), cans * can_rate + newspaper_collected * newspaper_rate = total_money ∧ cans = 144 := by
  sorry

end NUMINAMATH_CALUDE_cans_collected_l2847_284749


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2847_284768

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -2
  let y : ℝ := 3
  second_quadrant x y :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2847_284768


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2023_l2847_284789

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def GeometricSequence (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem arithmetic_sequence_2023 (a : ℕ → ℝ) (d : ℝ) :
  ArithmeticSequence a d →
  a 1 = 2 →
  GeometricSequence (a 1) (a 3) (a 7) →
  a 2023 = 2024 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2023_l2847_284789


namespace NUMINAMATH_CALUDE_max_value_sum_l2847_284724

theorem max_value_sum (a b c d e : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e)
  (sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 504) : 
  ∃ (N a_N b_N c_N d_N e_N : ℝ),
    (∀ (x y z w v : ℝ), x * z + 3 * y * z + 4 * z * w + 8 * z * v ≤ N) ∧
    (N = a_N * c_N + 3 * b_N * c_N + 4 * c_N * d_N + 8 * c_N * e_N) ∧
    (a_N^2 + b_N^2 + c_N^2 + d_N^2 + e_N^2 = 504) ∧
    (N + a_N + b_N + c_N + d_N + e_N = 32 + 1512 * Real.sqrt 10 + 6 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_l2847_284724


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2847_284730

theorem opposite_of_2023 : 
  ∀ x : ℤ, (x + 2023 = 0) ↔ (x = -2023) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2847_284730


namespace NUMINAMATH_CALUDE_dans_remaining_marbles_l2847_284776

theorem dans_remaining_marbles (initial_green : ℝ) (taken : ℝ) (remaining : ℝ) : 
  initial_green = 32.0 → 
  taken = 23.0 → 
  remaining = initial_green - taken → 
  remaining = 9.0 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_marbles_l2847_284776


namespace NUMINAMATH_CALUDE_sqrt_x_minus_two_real_l2847_284711

theorem sqrt_x_minus_two_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_two_real_l2847_284711


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l2847_284712

/-- Three positive numbers form an arithmetic sequence -/
def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

/-- Three numbers form a geometric sequence -/
def is_geometric_sequence (a b c : ℝ) : Prop := b / a = c / b

theorem arithmetic_geometric_sequence_problem :
  ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  is_arithmetic_sequence a b c →
  a + b + c = 15 →
  is_geometric_sequence (a + 1) (b + 3) (c + 9) →
  a = 3 ∧ b = 5 ∧ c = 7 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l2847_284712


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l2847_284785

theorem maximum_marks_calculation (victor_percentage : ℝ) (victor_marks : ℝ) : 
  victor_percentage = 92 → 
  victor_marks = 460 → 
  (victor_marks / (victor_percentage / 100)) = 500 := by
sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l2847_284785


namespace NUMINAMATH_CALUDE_square_area_from_circles_l2847_284706

/-- Given two circles where one passes through the center of and is tangent to the other,
    and the other is inscribed in a square, this theorem proves the area of the square
    given the area of the first circle. -/
theorem square_area_from_circles (circle_I circle_II : Real → Prop) (square : Real → Prop) : 
  (∃ r R s : Real,
    -- Circle I has area 9π
    circle_I r ∧ π * r^2 = 9 * π ∧
    -- Circle I passes through center of and is tangent to Circle II
    circle_II R ∧ R = 2 * r ∧
    -- Circle II is inscribed in the square
    square s ∧ s = 2 * R) →
  (∃ area : Real, square area ∧ area = 36) :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_circles_l2847_284706


namespace NUMINAMATH_CALUDE_min_value_cube_sum_squared_l2847_284721

theorem min_value_cube_sum_squared (a b c : ℝ) :
  (∃ (α β γ : ℤ), α ∈ ({-1, 1} : Set ℤ) ∧ β ∈ ({-1, 1} : Set ℤ) ∧ γ ∈ ({-1, 1} : Set ℤ) ∧ a * α + b * β + c * γ = 0) →
  ((a^3 + b^3 + c^3) / (a * b * c))^2 ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_cube_sum_squared_l2847_284721


namespace NUMINAMATH_CALUDE_harry_fish_count_l2847_284774

/-- Given three friends with fish, prove Harry's fish count -/
theorem harry_fish_count (sam joe harry : ℕ) : 
  sam = 7 →
  joe = 8 * sam →
  harry = 4 * joe →
  harry = 224 := by
  sorry

end NUMINAMATH_CALUDE_harry_fish_count_l2847_284774


namespace NUMINAMATH_CALUDE_total_bathing_suits_l2847_284700

theorem total_bathing_suits (men_suits women_suits : ℕ) 
  (h1 : men_suits = 14797) 
  (h2 : women_suits = 4969) : 
  men_suits + women_suits = 19766 := by
  sorry

end NUMINAMATH_CALUDE_total_bathing_suits_l2847_284700


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2847_284752

theorem sum_of_x_and_y (x y : ℝ) 
  (h1 : x^2 * y^3 + y^2 * x^3 = 27) 
  (h2 : x * y = 3) : 
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2847_284752


namespace NUMINAMATH_CALUDE_tickets_to_buy_l2847_284772

def ferris_wheel_cost : ℝ := 2.0
def roller_coaster_cost : ℝ := 7.0
def multiple_ride_discount : ℝ := 1.0
def newspaper_coupon : ℝ := 1.0

theorem tickets_to_buy :
  ferris_wheel_cost + roller_coaster_cost - multiple_ride_discount - newspaper_coupon = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_tickets_to_buy_l2847_284772


namespace NUMINAMATH_CALUDE_predict_grain_demand_2012_l2847_284753

/-- Regression equation for grain demand -/
def grain_demand (x : ℝ) : ℝ := 6.5 * (x - 2006) + 261

/-- Theorem: The predicted grain demand for 2012 is 300 ten thousand tons -/
theorem predict_grain_demand_2012 : grain_demand 2012 = 300 := by
  sorry

end NUMINAMATH_CALUDE_predict_grain_demand_2012_l2847_284753


namespace NUMINAMATH_CALUDE_max_power_under_500_l2847_284720

theorem max_power_under_500 :
  ∃ (a b : ℕ), 
    b > 1 ∧
    a^b < 500 ∧
    (∀ (c d : ℕ), d > 1 → c^d < 500 → c^d ≤ a^b) ∧
    a = 22 ∧
    b = 2 ∧
    a + b = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_power_under_500_l2847_284720


namespace NUMINAMATH_CALUDE_array_transformation_theorem_l2847_284770

/-- Represents an 8x8 array of +1 and -1 -/
def Array8x8 := Fin 8 → Fin 8 → Int

/-- Represents a move in the array -/
structure Move where
  row : Fin 8
  col : Fin 8

/-- Applies a move to an array -/
def applyMove (arr : Array8x8) (m : Move) : Array8x8 :=
  fun i j => if i = m.row ∨ j = m.col then -arr i j else arr i j

/-- Checks if an array is all +1 -/
def isAllPlusOne (arr : Array8x8) : Prop :=
  ∀ i j, arr i j = 1

theorem array_transformation_theorem :
  ∀ (initial : Array8x8),
  (∀ i j, initial i j = 1 ∨ initial i j = -1) →
  ∃ (moves : List Move),
  isAllPlusOne (moves.foldl applyMove initial) :=
sorry

end NUMINAMATH_CALUDE_array_transformation_theorem_l2847_284770


namespace NUMINAMATH_CALUDE_complex_roots_of_unity_real_sixth_power_l2847_284704

theorem complex_roots_of_unity_real_sixth_power :
  ∃! (S : Finset ℂ), 
    (∀ z ∈ S, z^24 = 1 ∧ (∃ r : ℝ, z^6 = r)) ∧ 
    Finset.card S = 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_of_unity_real_sixth_power_l2847_284704


namespace NUMINAMATH_CALUDE_final_amount_calculation_l2847_284758

def monthly_salary : ℝ := 2000
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def utility_bill_rate : ℝ := 0.25

theorem final_amount_calculation : 
  let tax := monthly_salary * tax_rate
  let insurance := monthly_salary * insurance_rate
  let after_deductions := monthly_salary - (tax + insurance)
  let utility_bills := after_deductions * utility_bill_rate
  monthly_salary - (tax + insurance + utility_bills) = 1125 := by
sorry

end NUMINAMATH_CALUDE_final_amount_calculation_l2847_284758


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_shorter_base_l2847_284775

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  a : ℝ  -- Length of the longer base
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid
  is_isosceles : a > b -- Condition for isosceles trapezoid

/-- 
  Theorem: In an isosceles trapezoid, if the foot of the height from a vertex 
  of the shorter base divides the longer base into two segments with a 
  difference of 10 units, then the length of the shorter base is 10 units.
-/
theorem isosceles_trapezoid_shorter_base 
  (t : IsoscelesTrapezoid) 
  (h : (t.a + t.b) / 2 = (t.a - t.b) / 2 + 10) : 
  t.b = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_shorter_base_l2847_284775


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l2847_284705

theorem quadratic_solution_difference : 
  let f : ℝ → ℝ := λ x => x^2 + 5*x - 4 - (x + 66)
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 2 * Real.sqrt 74 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l2847_284705


namespace NUMINAMATH_CALUDE_point_distance_to_line_l2847_284793

theorem point_distance_to_line (m : ℝ) : 
  let M : ℝ × ℝ := (1, 4)
  let l := {(x, y) : ℝ × ℝ | m * x + y - 1 = 0}
  (abs (m * M.1 + M.2 - 1) / Real.sqrt (m^2 + 1) = 3) → (m = 0 ∨ m = 3/4) := by
sorry

end NUMINAMATH_CALUDE_point_distance_to_line_l2847_284793


namespace NUMINAMATH_CALUDE_solve_timmys_orange_problem_l2847_284779

/-- Represents the problem of calculating Timmy's remaining money after buying oranges --/
def timmys_orange_problem (calories_per_orange : ℕ) (oranges_per_pack : ℕ) 
  (price_per_orange : ℚ) (initial_money : ℚ) (calorie_goal : ℕ) (tax_rate : ℚ) : Prop :=
  let packs_needed : ℕ := ((calorie_goal + calories_per_orange - 1) / calories_per_orange + oranges_per_pack - 1) / oranges_per_pack
  let total_cost : ℚ := price_per_orange * (packs_needed * oranges_per_pack : ℚ)
  let tax_amount : ℚ := total_cost * tax_rate
  let final_cost : ℚ := total_cost + tax_amount
  let remaining_money : ℚ := initial_money - final_cost
  remaining_money = 244/100

/-- Theorem stating the solution to Timmy's orange problem --/
theorem solve_timmys_orange_problem : 
  timmys_orange_problem 80 3 (120/100) 10 400 (5/100) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_timmys_orange_problem_l2847_284779


namespace NUMINAMATH_CALUDE_certain_value_problem_l2847_284731

theorem certain_value_problem (n : ℤ) (v : ℤ) (h1 : n = -7) (h2 : 3 * n = 2 * n - v) : v = 7 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_problem_l2847_284731


namespace NUMINAMATH_CALUDE_power_function_through_point_l2847_284756

/-- A power function that passes through (2, 2√2) and evaluates to 27 at x = 9 -/
theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x ^ a) →  -- f is a power function
  f 2 = 2 * Real.sqrt 2 →  -- f passes through (2, 2√2)
  f 9 = 27 :=  -- prove that f(9) = 27
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2847_284756


namespace NUMINAMATH_CALUDE_a_can_be_any_real_l2847_284792

theorem a_can_be_any_real : ∀ (a b c d : ℝ), 
  b * (3 * d + 2) ≠ 0 → 
  a / b < -c / (3 * d + 2) → 
  ∃ (a₁ a₂ a₃ : ℝ), a₁ > 0 ∧ a₂ < 0 ∧ a₃ = 0 ∧ 
    (a₁ / b < -c / (3 * d + 2)) ∧ 
    (a₂ / b < -c / (3 * d + 2)) ∧ 
    (a₃ / b < -c / (3 * d + 2)) := by
  sorry

end NUMINAMATH_CALUDE_a_can_be_any_real_l2847_284792


namespace NUMINAMATH_CALUDE_multiply_72519_9999_l2847_284780

theorem multiply_72519_9999 : 72519 * 9999 = 725117481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72519_9999_l2847_284780


namespace NUMINAMATH_CALUDE_lyon_marseille_distance_l2847_284713

/-- Given a map distance and scale, calculates the real distance between two points. -/
def real_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Proves that the real distance between Lyon and Marseille is 1200 km. -/
theorem lyon_marseille_distance :
  let map_distance : ℝ := 120
  let scale : ℝ := 10
  real_distance map_distance scale = 1200 := by
  sorry

end NUMINAMATH_CALUDE_lyon_marseille_distance_l2847_284713


namespace NUMINAMATH_CALUDE_exponential_function_condition_l2847_284778

theorem exponential_function_condition (x₁ x₂ : ℝ) :
  (x₁ + x₂ > 0) ↔ ((1/2 : ℝ)^x₁ * (1/2 : ℝ)^x₂ < 1) := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_condition_l2847_284778


namespace NUMINAMATH_CALUDE_saucer_area_l2847_284786

/-- The area of a circular saucer with radius 3 centimeters is 9π square centimeters. -/
theorem saucer_area (π : ℝ) (h : π > 0) : 
  let r : ℝ := 3
  let area : ℝ := π * r^2
  area = 9 * π := by sorry

end NUMINAMATH_CALUDE_saucer_area_l2847_284786


namespace NUMINAMATH_CALUDE_max_lateral_surface_area_cylinder_in_sphere_l2847_284719

/-- The maximum lateral surface area of a cylinder inscribed in a sphere -/
theorem max_lateral_surface_area_cylinder_in_sphere :
  ∀ (R r l : ℝ),
  R > 0 →
  r > 0 →
  l > 0 →
  (4 / 3) * Real.pi * R^3 = (32 / 3) * Real.pi →
  r^2 + (l / 2)^2 = R^2 →
  2 * Real.pi * r * l ≤ 8 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_max_lateral_surface_area_cylinder_in_sphere_l2847_284719


namespace NUMINAMATH_CALUDE_circle_through_points_l2847_284701

/-- A circle passing through three points -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Check if a point lies on the circle -/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 + c.D * x + c.E * y + c.F = 0

/-- The specific circle we're interested in -/
def our_circle : Circle := { D := -4, E := -6, F := 0 }

theorem circle_through_points : 
  (our_circle.contains 0 0) ∧ 
  (our_circle.contains 4 0) ∧ 
  (our_circle.contains (-1) 1) :=
by sorry

#check circle_through_points

end NUMINAMATH_CALUDE_circle_through_points_l2847_284701


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2847_284714

theorem isosceles_triangle_perimeter (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Sides are positive
  (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2) →  -- Two sides measure 2 and 5
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  (a = b ∨ b = c ∨ c = a) →  -- Isosceles condition
  a + b + c = 12 :=  -- Perimeter is 12
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2847_284714


namespace NUMINAMATH_CALUDE_sphere_radius_from_shadows_l2847_284715

/-- The radius of a sphere given its shadow and a reference post's shadow. -/
theorem sphere_radius_from_shadows
  (sphere_shadow : ℝ)
  (post_height : ℝ)
  (post_shadow : ℝ)
  (h1 : sphere_shadow = 15)
  (h2 : post_height = 1.5)
  (h3 : post_shadow = 3)
  (h4 : post_shadow > 0) -- Ensure division is valid
  : ∃ (r : ℝ), r = sphere_shadow * (post_height / post_shadow) ∧ r = 7.5 :=
by
  sorry


end NUMINAMATH_CALUDE_sphere_radius_from_shadows_l2847_284715


namespace NUMINAMATH_CALUDE_at_least_one_quadratic_has_solution_l2847_284798

theorem at_least_one_quadratic_has_solution (a b c : ℝ) : 
  (∃ x : ℝ, x^2 + (a-b)*x + (b-c) = 0) ∨ 
  (∃ x : ℝ, x^2 + (b-c)*x + (c-a) = 0) ∨ 
  (∃ x : ℝ, x^2 + (c-a)*x + (a-b) = 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_quadratic_has_solution_l2847_284798


namespace NUMINAMATH_CALUDE_right_triangle_circle_intersection_l2847_284766

-- Define the triangle and circle
structure RightTriangle :=
  (A B C : ℝ × ℝ)
  (isRight : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)

structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the theorem
theorem right_triangle_circle_intersection
  (triangle : RightTriangle)
  (circle : Circle)
  (D : ℝ × ℝ)
  (h1 : circle.center = ((triangle.B.1 + triangle.C.1) / 2, (triangle.B.2 + triangle.C.2) / 2))
  (h2 : circle.radius = Real.sqrt ((triangle.B.1 - triangle.C.1)^2 + (triangle.B.2 - triangle.C.2)^2) / 2)
  (h3 : D.1 = triangle.A.1 + 2 * (triangle.C.1 - triangle.A.1) / (triangle.C.1 - triangle.A.1 + triangle.C.2 - triangle.A.2))
  (h4 : D.2 = triangle.A.2 + 2 * (triangle.C.2 - triangle.A.2) / (triangle.C.1 - triangle.A.1 + triangle.C.2 - triangle.A.2))
  (h5 : Real.sqrt ((D.1 - triangle.A.1)^2 + (D.2 - triangle.A.2)^2) = 2)
  (h6 : Real.sqrt ((D.1 - triangle.B.1)^2 + (D.2 - triangle.B.2)^2) = 3)
  : Real.sqrt ((D.1 - triangle.C.1)^2 + (D.2 - triangle.C.2)^2) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circle_intersection_l2847_284766


namespace NUMINAMATH_CALUDE_Q_four_roots_implies_d_value_l2847_284703

/-- The polynomial Q(x) -/
def Q (d x : ℂ) : ℂ := (x^2 - 3*x + 3) * (x^2 - d*x + 5) * (x^2 - 5*x + 15)

/-- The theorem stating that if Q(x) has exactly 4 distinct roots, then |d| = 13/2 -/
theorem Q_four_roots_implies_d_value (d : ℂ) :
  (∃ (s : Finset ℂ), s.card = 4 ∧ (∀ x ∈ s, Q d x = 0) ∧ (∀ x, Q d x = 0 → x ∈ s)) →
  Complex.abs d = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_Q_four_roots_implies_d_value_l2847_284703


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_zero_is_exists_lt_zero_l2847_284799

theorem negation_of_forall_geq_zero_is_exists_lt_zero :
  (¬ ∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 - x₀ + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_zero_is_exists_lt_zero_l2847_284799


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_l2847_284762

/-- The function f(x) = x^2 + x - 1 -/
def f (x : ℝ) : ℝ := x^2 + x - 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2*x + 1

theorem tangent_line_at_point_one :
  ∃ (m b : ℝ), 
    (f 1 = 1) ∧ 
    (f' 1 = m) ∧ 
    (∀ x y : ℝ, y = m * (x - 1) + 1 ↔ m * x - y + b = 0) ∧
    (3 * 1 - 1 + b = 0) ∧
    (∀ x y : ℝ, y = m * (x - 1) + 1 ↔ 3 * x - y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_l2847_284762


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2847_284709

theorem repeating_decimal_sum : 
  (0.12121212 : ℚ) + (0.003003003 : ℚ) + (0.0000500005 : ℚ) = 124215 / 999999 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2847_284709


namespace NUMINAMATH_CALUDE_round_trip_distance_prove_round_trip_distance_l2847_284748

def boat_speed : ℝ := 9
def stream_speed : ℝ := 6
def total_time : ℝ := 68

theorem round_trip_distance : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let distance := (total_time * downstream_speed * upstream_speed) / (downstream_speed + upstream_speed)
  170

theorem prove_round_trip_distance : round_trip_distance = 170 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_distance_prove_round_trip_distance_l2847_284748


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2847_284795

/-- An isosceles triangle with side lengths 1 and 2 has perimeter 5 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) →
  (a = b ∨ b = c ∨ a = c) →
  a + b + c = 5 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2847_284795


namespace NUMINAMATH_CALUDE_group_morphism_identity_or_inverse_l2847_284740

variable {G : Type*} [Group G]

theorem group_morphism_identity_or_inverse
  (no_order_4 : ∀ g : G, g^4 = 1 → g = 1)
  (f : G → G)
  (f_hom : ∀ x y : G, f (x * y) = f x * f y)
  (f_property : ∀ x : G, f x = x ∨ f x = x⁻¹) :
  (∀ x : G, f x = x) ∨ (∀ x : G, f x = x⁻¹) := by
  sorry

end NUMINAMATH_CALUDE_group_morphism_identity_or_inverse_l2847_284740


namespace NUMINAMATH_CALUDE_estimate_keyboard_warriors_opposition_l2847_284784

/-- Estimates the number of people with a certain characteristic in a population based on a sample. -/
def estimatePopulation (totalPopulation : ℕ) (sampleSize : ℕ) (sampleOpposed : ℕ) : ℕ :=
  (totalPopulation * sampleOpposed) / sampleSize

/-- Theorem stating that the estimated number of people opposed to "keyboard warriors" is 6912. -/
theorem estimate_keyboard_warriors_opposition :
  let totalPopulation : ℕ := 9600
  let sampleSize : ℕ := 50
  let sampleOpposed : ℕ := 36
  estimatePopulation totalPopulation sampleSize sampleOpposed = 6912 := by
  sorry

#eval estimatePopulation 9600 50 36

end NUMINAMATH_CALUDE_estimate_keyboard_warriors_opposition_l2847_284784


namespace NUMINAMATH_CALUDE_andy_profit_per_cake_l2847_284718

/-- Andy's cake business model -/
structure CakeBusiness where
  ingredient_cost_two_cakes : ℕ
  packaging_cost_per_cake : ℕ
  selling_price_per_cake : ℕ

/-- Calculate the profit per cake -/
def profit_per_cake (b : CakeBusiness) : ℕ :=
  b.selling_price_per_cake - (b.ingredient_cost_two_cakes / 2 + b.packaging_cost_per_cake)

/-- Theorem: Andy's profit per cake is $8 -/
theorem andy_profit_per_cake :
  ∃ (b : CakeBusiness),
    b.ingredient_cost_two_cakes = 12 ∧
    b.packaging_cost_per_cake = 1 ∧
    b.selling_price_per_cake = 15 ∧
    profit_per_cake b = 8 := by
  sorry

end NUMINAMATH_CALUDE_andy_profit_per_cake_l2847_284718


namespace NUMINAMATH_CALUDE_number_puzzle_l2847_284733

theorem number_puzzle (A B : ℝ) (h1 : A + B = 14.85) (h2 : B = 10 * A) : A = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2847_284733


namespace NUMINAMATH_CALUDE_dog_division_theorem_l2847_284790

def number_of_dogs : ℕ := 12
def group_sizes : List ℕ := [4, 5, 3]

def ways_to_divide_dogs (n : ℕ) (sizes : List ℕ) : ℕ :=
  sorry

theorem dog_division_theorem :
  ways_to_divide_dogs number_of_dogs group_sizes = 4200 :=
by sorry

end NUMINAMATH_CALUDE_dog_division_theorem_l2847_284790


namespace NUMINAMATH_CALUDE_darker_tile_fraction_is_three_fourths_l2847_284760

/-- Represents a floor with a repeating tile pattern -/
structure Floor :=
  (pattern_size : Nat)
  (corner_size : Nat)
  (dark_tiles_in_corner : Nat)

/-- The fraction of darker tiles in the floor -/
def darker_tile_fraction (f : Floor) : Rat :=
  let total_tiles := f.pattern_size * f.pattern_size
  let corner_tiles := f.corner_size * f.corner_size
  let num_corners := (f.pattern_size / f.corner_size) ^ 2
  let total_dark_tiles := f.dark_tiles_in_corner * num_corners
  total_dark_tiles / total_tiles

/-- Theorem stating that for a floor with a 4x4 repeating pattern and 3 darker tiles in each 2x2 corner,
    the fraction of darker tiles is 3/4 -/
theorem darker_tile_fraction_is_three_fourths (f : Floor)
  (h1 : f.pattern_size = 4)
  (h2 : f.corner_size = 2)
  (h3 : f.dark_tiles_in_corner = 3) :
  darker_tile_fraction f = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_darker_tile_fraction_is_three_fourths_l2847_284760


namespace NUMINAMATH_CALUDE_light_ray_equation_l2847_284771

/-- A light ray is emitted from point A(-3, 3), hits the x-axis, gets reflected, and is tangent to a circle. This theorem proves that the equation of the line on which the light ray lies is either 3x + 4y - 3 = 0 or 4x + 3y + 3 = 0. -/
theorem light_ray_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (-3, 3)
  let circle (x y : ℝ) := x^2 + y^2 - 4*x - 4*y + 7 = 0
  let ray_hits_x_axis : Prop := ∃ (t : ℝ), t * (A.1 + 3) = -3 ∧ t * (A.2 - 3) = 0
  let is_tangent_to_circle : Prop := ∃ (x₀ y₀ : ℝ), circle x₀ y₀ ∧ 
    ((x - x₀) * (2*x₀ - 4) + (y - y₀) * (2*y₀ - 4) = 0)
  ray_hits_x_axis → is_tangent_to_circle → 
    (3*x + 4*y - 3 = 0) ∨ (4*x + 3*y + 3 = 0) :=
by sorry


end NUMINAMATH_CALUDE_light_ray_equation_l2847_284771


namespace NUMINAMATH_CALUDE_banana_box_cost_l2847_284728

/-- Calculates the total cost of bananas after discount -/
def totalCostAfterDiscount (
  bunches8 : ℕ)  -- Number of bunches with 8 bananas
  (price8 : ℚ)   -- Price of each bunch with 8 bananas
  (bunches7 : ℕ)  -- Number of bunches with 7 bananas
  (price7 : ℚ)   -- Price of each bunch with 7 bananas
  (discount : ℚ)  -- Discount as a decimal
  : ℚ :=
  let totalCost := bunches8 * price8 + bunches7 * price7
  totalCost * (1 - discount)

/-- Proves that the total cost after discount for the given conditions is $23.40 -/
theorem banana_box_cost :
  totalCostAfterDiscount 6 2.5 5 2.2 0.1 = 23.4 := by
  sorry

end NUMINAMATH_CALUDE_banana_box_cost_l2847_284728


namespace NUMINAMATH_CALUDE_diamond_value_l2847_284734

def diamond (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

theorem diamond_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : a + b = 10) (h2 : a * b = 24) : 
  diamond a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_diamond_value_l2847_284734


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2847_284732

/-- A function satisfying the given conditions -/
def f_satisfies (f : ℝ → ℝ) : Prop :=
  f 0 = 2 ∧ 
  ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 1

/-- The solution set of the inequality -/
def solution_set (x : ℝ) : Prop :=
  Real.log 2 < x ∧ x < Real.log 3

/-- The main theorem -/
theorem inequality_solution_set (f : ℝ → ℝ) (hf : f_satisfies f) :
  ∀ x, f (Real.log (Real.exp x - 2)) < 2 + Real.log (Real.exp x - 2) ↔ solution_set x := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2847_284732


namespace NUMINAMATH_CALUDE_sequence_divisibility_l2847_284737

theorem sequence_divisibility (m n k : ℕ) (a : ℕ → ℕ) (hm : m > 1) (hn : n ≥ 0) :
  m^n ∣ a k → m^(n+1) ∣ (a (k+1))^m - (a (k-1))^m :=
by sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l2847_284737


namespace NUMINAMATH_CALUDE_f_difference_l2847_284796

/-- The function f(x) = 2x^3 - 3x^2 + 4x - 5 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 4 * x - 5

/-- Theorem stating that f(x + h) - f(x) equals the given expression -/
theorem f_difference (x h : ℝ) : 
  f (x + h) - f x = 6 * x^2 - 6 * x + 6 * x * h + 2 * h^2 - 3 * h + 4 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l2847_284796


namespace NUMINAMATH_CALUDE_rooster_earnings_l2847_284754

def price_per_kg : ℚ := 1/2

def rooster1_weight : ℚ := 30
def rooster2_weight : ℚ := 40

def total_earnings : ℚ := price_per_kg * (rooster1_weight + rooster2_weight)

theorem rooster_earnings : total_earnings = 35 := by
  sorry

end NUMINAMATH_CALUDE_rooster_earnings_l2847_284754


namespace NUMINAMATH_CALUDE_geometric_sum_first_five_terms_l2847_284736

/-- Sum of first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric sequence -/
def a : ℚ := 1/4

/-- The common ratio of the geometric sequence -/
def r : ℚ := 1/4

/-- The number of terms to sum -/
def n : ℕ := 5

theorem geometric_sum_first_five_terms :
  geometric_sum a r n = 341/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_five_terms_l2847_284736


namespace NUMINAMATH_CALUDE_line_parameterization_l2847_284764

/-- Given a line y = 5x - 7 parameterized by [x; y] = [s; 2] + t[3; h], 
    prove that s = 9/5 and h = 15 -/
theorem line_parameterization (x y s h t : ℝ) : 
  y = 5 * x - 7 ∧ 
  ∃ (v : ℝ × ℝ), v.1 = x ∧ v.2 = y ∧ v = (s, 2) + t • (3, h) →
  s = 9/5 ∧ h = 15 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l2847_284764


namespace NUMINAMATH_CALUDE_systematic_sampling_fourth_element_l2847_284788

/-- Systematic sampling function -/
def systematicSample (totalSize : Nat) (sampleSize : Nat) : Nat → Nat :=
  fun i => i * (totalSize / sampleSize) + 1

theorem systematic_sampling_fourth_element
  (totalSize : Nat)
  (sampleSize : Nat)
  (h1 : totalSize = 36)
  (h2 : sampleSize = 4)
  (h3 : systematicSample totalSize sampleSize 0 = 6)
  (h4 : systematicSample totalSize sampleSize 2 = 24)
  (h5 : systematicSample totalSize sampleSize 3 = 33) :
  systematicSample totalSize sampleSize 1 = 15 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_fourth_element_l2847_284788


namespace NUMINAMATH_CALUDE_no_two_common_tangents_l2847_284722

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Number of common tangents between two circles --/
def commonTangents (c1 c2 : Circle) : ℕ := sorry

theorem no_two_common_tangents (c1 c2 : Circle) (h : c1.radius ≠ c2.radius) :
  commonTangents c1 c2 ≠ 2 := by sorry

end NUMINAMATH_CALUDE_no_two_common_tangents_l2847_284722


namespace NUMINAMATH_CALUDE_square_root_problem_l2847_284769

theorem square_root_problem (a b : ℝ) 
  (h1 : Real.sqrt (a + 9) = -5)
  (h2 : (2 * b - a) ^ (1/3 : ℝ) = -2) :
  Real.sqrt (2 * a + b) = 6 := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l2847_284769


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2847_284777

-- Define set A
def A : Set ℝ := {x | |x + 1| < 2}

-- Define set B
def B : Set ℝ := {x | -x^2 + 2*x + 3 ≥ 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x | -3 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2847_284777


namespace NUMINAMATH_CALUDE_coprime_power_minus_one_divisible_l2847_284708

theorem coprime_power_minus_one_divisible
  (N₁ N₂ : ℕ+) (k : ℕ) 
  (h_coprime : Nat.Coprime N₁ N₂)
  (h_k : k = Nat.totient N₂) :
  N₂ ∣ (N₁^k - 1) :=
by sorry

end NUMINAMATH_CALUDE_coprime_power_minus_one_divisible_l2847_284708


namespace NUMINAMATH_CALUDE_box_plates_cups_weight_l2847_284747

/-- Given the weights of various combinations of a box, plates, and cups, 
    prove that the weight of the box with 10 plates and 20 cups is 3 kg. -/
theorem box_plates_cups_weight :
  ∀ (b p c : ℝ),
  (b + 20 * p + 30 * c = 4.8) →
  (b + 40 * p + 50 * c = 8.4) →
  (b + 10 * p + 20 * c = 3) :=
by sorry

end NUMINAMATH_CALUDE_box_plates_cups_weight_l2847_284747


namespace NUMINAMATH_CALUDE_shortest_segment_length_l2847_284744

/-- Represents the paper strip and folding operations -/
structure PaperStrip where
  length : Real
  red_dot_position : Real
  yellow_dot_position : Real

/-- Calculates the position of the yellow dot after the first fold -/
def calculate_yellow_dot_position (strip : PaperStrip) : Real :=
  strip.length - strip.red_dot_position

/-- Calculates the length of the segment between red and yellow dots -/
def calculate_middle_segment (strip : PaperStrip) : Real :=
  strip.length - 2 * strip.yellow_dot_position

/-- Calculates the length of the shortest segment after all folds and cuts -/
def calculate_shortest_segment (strip : PaperStrip) : Real :=
  strip.red_dot_position - 2 * (strip.red_dot_position - strip.yellow_dot_position)

/-- Theorem stating that the shortest segment is 0.146 meters long -/
theorem shortest_segment_length :
  let initial_strip : PaperStrip := {
    length := 1,
    red_dot_position := 0.618,
    yellow_dot_position := calculate_yellow_dot_position { length := 1, red_dot_position := 0.618, yellow_dot_position := 0 }
  }
  calculate_shortest_segment initial_strip = 0.146 := by
  sorry

end NUMINAMATH_CALUDE_shortest_segment_length_l2847_284744


namespace NUMINAMATH_CALUDE_speed_in_still_water_l2847_284739

/-- The speed of a man in still water given his upstream and downstream speeds -/
theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) :
  upstream_speed = 20 →
  downstream_speed = 80 →
  (upstream_speed + downstream_speed) / 2 = 50 := by
  sorry

#check speed_in_still_water

end NUMINAMATH_CALUDE_speed_in_still_water_l2847_284739


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_l2847_284781

/-- A quadratic function f(x) = ax² + bx + c has exactly two distinct real zeros when a·c < 0 -/
theorem quadratic_two_zeros (a b c : ℝ) (h : a * c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_l2847_284781


namespace NUMINAMATH_CALUDE_rectangle_area_l2847_284707

-- Define a rectangle type
structure Rectangle where
  width : ℝ
  length : ℝ
  diagonal : ℝ

-- Theorem statement
theorem rectangle_area (r : Rectangle) 
  (h1 : r.length = 2 * r.width) 
  (h2 : r.diagonal = 15 * Real.sqrt 2) : 
  r.width * r.length = 180 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2847_284707


namespace NUMINAMATH_CALUDE_degree_to_radian_300_l2847_284741

theorem degree_to_radian_300 : 
  (300 : ℝ) * (π / 180) = (5 * π) / 3 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_300_l2847_284741


namespace NUMINAMATH_CALUDE_athletes_leaving_rate_l2847_284742

/-- The rate at which athletes left the camp per hour -/
def leaving_rate : ℝ := 24.5

/-- The initial number of athletes at the camp -/
def initial_athletes : ℕ := 300

/-- The number of hours athletes left the camp -/
def leaving_hours : ℕ := 4

/-- The rate at which new athletes entered the camp per hour -/
def entering_rate : ℕ := 15

/-- The number of hours new athletes entered the camp -/
def entering_hours : ℕ := 7

/-- The difference in total number of athletes over the two nights -/
def athlete_difference : ℕ := 7

theorem athletes_leaving_rate : 
  initial_athletes - leaving_rate * leaving_hours + entering_rate * entering_hours 
  = initial_athletes + athlete_difference :=
sorry

end NUMINAMATH_CALUDE_athletes_leaving_rate_l2847_284742


namespace NUMINAMATH_CALUDE_train_length_l2847_284783

/-- Given a train that crosses a platform in 39 seconds, crosses a signal pole in 18 seconds,
    and the platform length is 175 meters, the length of the train is 150 meters. -/
theorem train_length (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
    (h1 : platform_crossing_time = 39)
    (h2 : pole_crossing_time = 18)
    (h3 : platform_length = 175) :
    ∃ train_length : ℝ, train_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2847_284783


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_find_b_find_c_find_d_l2847_284782

-- Problem 1
theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 10) :
  1 / x + 1 / y = 2 := by sorry

-- Problem 2
theorem find_b (b : ℝ) (h1 : b > 0) (h2 : b^2 - 1 = 135 * 137) :
  b = 136 := by sorry

-- Problem 3
theorem find_c (c : ℝ) 
  (h : (1 : ℝ) * (-1 / 2) * (-c / 3) = -1) :
  c = -6 := by sorry

-- Problem 4
theorem find_d (c d : ℝ) 
  (h : (d - 1) / c = -1) (h2 : c = 2) :
  d = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_find_b_find_c_find_d_l2847_284782


namespace NUMINAMATH_CALUDE_percent_of_x_is_y_l2847_284794

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y = 0.25 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_is_y_l2847_284794


namespace NUMINAMATH_CALUDE_solution_ratio_l2847_284743

/-- Given a system of equations with solution (2, 5), prove that a/c = 3 -/
theorem solution_ratio (a c : ℝ) : 
  (∃ x y : ℝ, x = 2 ∧ y = 5 ∧ a * x + 2 * y = 16 ∧ 3 * x - y = c) →
  a / c = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_ratio_l2847_284743


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2847_284755

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x - 1) / (x + 2) ≤ 3 ↔ x ∈ Set.Iic (-7) ∪ Set.Ioi (-2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2847_284755


namespace NUMINAMATH_CALUDE_second_integer_value_l2847_284751

theorem second_integer_value (n : ℤ) : (n - 2) + (n + 2) = 132 → n = 66 := by
  sorry

end NUMINAMATH_CALUDE_second_integer_value_l2847_284751


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2847_284717

/-- Given that M(-3, 2) is the midpoint of AB and A(-8, 5) is one endpoint,
    prove that the sum of coordinates of B is 1. -/
theorem midpoint_coordinate_sum :
  let M : ℝ × ℝ := (-3, 2)
  let A : ℝ × ℝ := (-8, 5)
  ∀ B : ℝ × ℝ,
  (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) →
  B.1 + B.2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2847_284717


namespace NUMINAMATH_CALUDE_downstream_distance_l2847_284710

/-- Calculates the distance traveled downstream by a boat -/
theorem downstream_distance
  (boat_speed : ℝ)  -- Speed of the boat in still water
  (stream_speed : ℝ) -- Speed of the stream
  (time : ℝ)  -- Time taken to travel downstream
  (h1 : boat_speed = 40)  -- Condition: Boat speed is 40 km/hr
  (h2 : stream_speed = 5)  -- Condition: Stream speed is 5 km/hr
  (h3 : time = 1)  -- Condition: Time taken is 1 hour
  : boat_speed + stream_speed * time = 45 := by
  sorry

#check downstream_distance

end NUMINAMATH_CALUDE_downstream_distance_l2847_284710


namespace NUMINAMATH_CALUDE_marias_sister_bottles_l2847_284761

/-- Given the initial number of water bottles, the number Maria drank, and the number left,
    calculate the number of bottles Maria's sister drank. -/
theorem marias_sister_bottles (initial : ℝ) (maria_drank : ℝ) (left : ℝ) :
  initial = 45.0 →
  maria_drank = 14.0 →
  left = 23.0 →
  initial - maria_drank - left = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_marias_sister_bottles_l2847_284761


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2847_284735

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ ¬(1/a < 1 → a > 1) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2847_284735


namespace NUMINAMATH_CALUDE_prob_at_least_six_heads_in_nine_flips_prob_at_least_six_heads_in_nine_flips_proof_l2847_284765

/-- The probability of getting at least 6 heads in 9 fair coin flips -/
theorem prob_at_least_six_heads_in_nine_flips : ℝ :=
  130 / 512

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The probability of getting exactly k heads in n fair coin flips -/
def prob_exactly_k_heads (n k : ℕ) : ℝ := sorry

/-- The probability of getting at least k heads in n fair coin flips -/
def prob_at_least_k_heads (n k : ℕ) : ℝ := sorry

theorem prob_at_least_six_heads_in_nine_flips_proof :
  prob_at_least_k_heads 9 6 = prob_at_least_six_heads_in_nine_flips :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_six_heads_in_nine_flips_prob_at_least_six_heads_in_nine_flips_proof_l2847_284765


namespace NUMINAMATH_CALUDE_probability_not_spade_first_draw_l2847_284716

theorem probability_not_spade_first_draw (total_cards : ℕ) (spade_cards : ℕ) 
  (h1 : total_cards = 52) (h2 : spade_cards = 13) :
  (total_cards - spade_cards : ℚ) / total_cards = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_spade_first_draw_l2847_284716


namespace NUMINAMATH_CALUDE_total_skittles_l2847_284738

theorem total_skittles (num_students : ℕ) (skittles_per_student : ℕ) 
  (h1 : num_students = 9)
  (h2 : skittles_per_student = 3) :
  num_students * skittles_per_student = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_skittles_l2847_284738


namespace NUMINAMATH_CALUDE_lawn_mowing_price_l2847_284750

def sneaker_cost : ℕ := 92
def lawns_to_mow : ℕ := 3
def figures_to_sell : ℕ := 2
def figure_price : ℕ := 9
def job_hours : ℕ := 10
def hourly_rate : ℕ := 5

theorem lawn_mowing_price : 
  (sneaker_cost - (figures_to_sell * figure_price + job_hours * hourly_rate)) / lawns_to_mow = 8 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_price_l2847_284750


namespace NUMINAMATH_CALUDE_consecutive_composite_numbers_l2847_284727

theorem consecutive_composite_numbers (k k' : ℕ) :
  (∀ i ∈ Finset.range 7, ¬ Nat.Prime (210 * k + 1 + i + 1)) ∧
  (∀ i ∈ Finset.range 15, ¬ Nat.Prime (30030 * k' + 1 + i + 1)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_composite_numbers_l2847_284727


namespace NUMINAMATH_CALUDE_stating_solutions_depend_on_angle_l2847_284757

/-- Represents a plane in 3D space --/
structure Plane where
  s₁ : ℝ
  s₂ : ℝ

/-- Represents an axis in 3D space --/
structure Axis where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the number of solutions --/
inductive NumSolutions
  | zero
  | one
  | two

/-- 
Given a plane S and an angle α₁ between S and the horizontal plane,
determines the number of possible x₁,₄ axes such that S is perpendicular
to the bisectors of the first and fourth quadrants.
--/
def num_solutions (S : Plane) (α₁ : ℝ) : NumSolutions :=
  sorry

/-- 
Theorem stating that the number of solutions depends on α₁
--/
theorem solutions_depend_on_angle (S : Plane) (α₁ : ℝ) :
  (α₁ > 45 → num_solutions S α₁ = NumSolutions.two) ∧
  (α₁ ≤ 45 → num_solutions S α₁ = NumSolutions.one ∨ num_solutions S α₁ = NumSolutions.zero) :=
  sorry

end NUMINAMATH_CALUDE_stating_solutions_depend_on_angle_l2847_284757


namespace NUMINAMATH_CALUDE_sandbag_weight_l2847_284702

/-- Calculates the weight of a partially filled sandbag with a heavier filling material -/
theorem sandbag_weight (bag_capacity : ℝ) (fill_percentage : ℝ) (material_weight_increase : ℝ) : 
  bag_capacity > 0 → 
  fill_percentage > 0 → 
  fill_percentage ≤ 1 → 
  material_weight_increase ≥ 0 →
  let sand_weight := bag_capacity * fill_percentage
  let material_weight := sand_weight * (1 + material_weight_increase)
  bag_capacity + material_weight = 530 :=
by
  sorry

#check sandbag_weight 250 0.8 0.4

end NUMINAMATH_CALUDE_sandbag_weight_l2847_284702


namespace NUMINAMATH_CALUDE_cubic_equation_solution_sum_l2847_284767

/-- Given r, s, and t are solutions of x^3 - 6x^2 + 11x - 16 = 0, prove that (r+s)/t + (s+t)/r + (t+r)/s = 11/8 -/
theorem cubic_equation_solution_sum (r s t : ℝ) : 
  r^3 - 6*r^2 + 11*r - 16 = 0 →
  s^3 - 6*s^2 + 11*s - 16 = 0 →
  t^3 - 6*t^2 + 11*t - 16 = 0 →
  (r+s)/t + (s+t)/r + (t+r)/s = 11/8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_sum_l2847_284767


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_condition_l2847_284725

theorem smallest_positive_integer_satisfying_condition : 
  ∃ (x : ℕ+), (x : ℝ) + 1000 > 1000 * x ∧ 
  ∀ (y : ℕ+), ((y : ℝ) + 1000 > 1000 * y → x ≤ y) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_condition_l2847_284725


namespace NUMINAMATH_CALUDE_regular_100gon_rectangle_two_colors_l2847_284729

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A coloring of the vertices of a polygon -/
def Coloring (n : ℕ) (k : ℕ) := Fin n → Fin k

/-- Four vertices form a rectangle in a regular polygon -/
def IsRectangle (p : RegularPolygon 100) (v1 v2 v3 v4 : Fin 100) : Prop :=
  sorry

/-- The number of distinct colors used for given vertices -/
def NumColors (c : Coloring 100 10) (vs : List (Fin 100)) : ℕ :=
  sorry

theorem regular_100gon_rectangle_two_colors :
  ∀ (p : RegularPolygon 100) (c : Coloring 100 10),
  ∃ (v1 v2 v3 v4 : Fin 100),
    IsRectangle p v1 v2 v3 v4 ∧ NumColors c [v1, v2, v3, v4] ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_regular_100gon_rectangle_two_colors_l2847_284729


namespace NUMINAMATH_CALUDE_vasya_numbers_l2847_284797

theorem vasya_numbers (x y : ℝ) : (x - 1) * (y - 1) = x * y → x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_vasya_numbers_l2847_284797
