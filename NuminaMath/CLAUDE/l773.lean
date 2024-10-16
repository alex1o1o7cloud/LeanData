import Mathlib

namespace NUMINAMATH_CALUDE_max_value_condition_l773_77354

/-- 
Given that x and y are real numbers, prove that when 2005 - (x + y)^2 takes its maximum value, x = -y.
-/
theorem max_value_condition (x y : ℝ) : 
  (∀ a b : ℝ, 2005 - (x + y)^2 ≥ 2005 - (a + b)^2) → x = -y := by
sorry

end NUMINAMATH_CALUDE_max_value_condition_l773_77354


namespace NUMINAMATH_CALUDE_function_value_proof_l773_77320

/-- Given a function f(x) = x^5 - ax^3 + bx - 6 where f(-2) = 10, prove that f(2) = -22 -/
theorem function_value_proof (a b : ℝ) : 
  let f := λ x : ℝ => x^5 - a*x^3 + b*x - 6
  f (-2) = 10 → f 2 = -22 := by
sorry

end NUMINAMATH_CALUDE_function_value_proof_l773_77320


namespace NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l773_77305

-- Problem 1
theorem calculate_expression : (-3)^2 - 3^0 + (-2) = 6 := by sorry

-- Problem 2
theorem solve_system_of_equations :
  ∃ x y : ℝ, 2*x - y = 3 ∧ x + y = 6 ∧ x = 3 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l773_77305


namespace NUMINAMATH_CALUDE_no_integer_solutions_l773_77347

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l773_77347


namespace NUMINAMATH_CALUDE_line_slope_l773_77325

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 6 * x + 7 * y - 3 = 0

-- State the theorem
theorem line_slope :
  ∃ m b : ℝ, (∀ x y : ℝ, line_equation x y ↔ y = m * x + b) ∧ m = -6/7 :=
sorry

end NUMINAMATH_CALUDE_line_slope_l773_77325


namespace NUMINAMATH_CALUDE_quadratic_y_order_l773_77330

/-- A quadratic function f(x) = -x^2 - 2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 - 2*x + m

/-- Theorem: For a quadratic function f(x) = -x^2 - 2x + m and three points 
    on its graph A(-1, y₁), B(√2-1, y₂), and C(5, y₃), y₃ < y₂ < y₁ -/
theorem quadratic_y_order (m : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : f m (-1) = y₁)
  (h₂ : f m (Real.sqrt 2 - 1) = y₂)
  (h₃ : f m 5 = y₃) :
  y₃ < y₂ ∧ y₂ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_y_order_l773_77330


namespace NUMINAMATH_CALUDE_whale_weight_precision_l773_77341

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : Float
  exponent : Int

/-- Represents the level of precision for a number -/
inductive Precision
  | Hundreds
  | Thousands
  | TenThousands
  | HundredThousands

/-- Determines the precision of a number in scientific notation -/
def getPrecision (n : ScientificNotation) : Precision :=
  sorry

/-- The approximate weight of the whale in scientific notation -/
def whaleWeight : ScientificNotation :=
  { coefficient := 1.36, exponent := 5 }

/-- Theorem stating that the whale weight is precise to the thousands place -/
theorem whale_weight_precision :
  getPrecision whaleWeight = Precision.Thousands :=
sorry

end NUMINAMATH_CALUDE_whale_weight_precision_l773_77341


namespace NUMINAMATH_CALUDE_second_draw_probability_l773_77367

/-- Represents the probability of drawing a red sweet in the second draw -/
def probability_second_red (x y : ℕ) : ℚ :=
  y / (x + y)

/-- Theorem stating that the probability of drawing a red sweet in the second draw
    is equal to the initial ratio of red sweets to total sweets -/
theorem second_draw_probability (x y : ℕ) (hxy : x + y > 0) :
  probability_second_red x y = y / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_second_draw_probability_l773_77367


namespace NUMINAMATH_CALUDE_harriet_miles_run_l773_77334

theorem harriet_miles_run (total_miles : ℝ) (katarina_miles : ℝ) (adriana_miles : ℝ) 
  (h1 : total_miles = 285)
  (h2 : katarina_miles = 51)
  (h3 : adriana_miles = 74)
  (h4 : ∃ (x : ℝ), x * 3 + katarina_miles + adriana_miles = total_miles) :
  ∃ (harriet_miles : ℝ), harriet_miles = 53.33 ∧ 
    harriet_miles * 3 + katarina_miles + adriana_miles = total_miles :=
by
  sorry

end NUMINAMATH_CALUDE_harriet_miles_run_l773_77334


namespace NUMINAMATH_CALUDE_min_sum_of_probabilities_l773_77340

theorem min_sum_of_probabilities (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let p_a := 4 / x
  let p_b := 1 / y
  (p_a + p_b = 1) → (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 / x' + 1 / y' = 1 → x' + y' ≥ x + y) →
  x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_probabilities_l773_77340


namespace NUMINAMATH_CALUDE_plank_length_l773_77378

/-- The length of a plank given specific movements of its ends -/
theorem plank_length (a b : ℝ) : 
  (∀ x y, x^2 + y^2 = a^2 + b^2 → (x - 8)^2 + (y + 4)^2 = a^2 + b^2) →
  (∀ x y, x^2 + y^2 = a^2 + b^2 → (x - 17)^2 + (y + 7)^2 = a^2 + b^2) →
  a^2 + b^2 = 65^2 := by
  sorry

end NUMINAMATH_CALUDE_plank_length_l773_77378


namespace NUMINAMATH_CALUDE_remainder_problem_l773_77377

theorem remainder_problem (k : ℕ+) (h : ∃ b : ℕ, 120 = b * k^2 + 12) : 
  ∃ q : ℕ, 200 = q * k + 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l773_77377


namespace NUMINAMATH_CALUDE_segment_division_l773_77372

theorem segment_division (AB : ℝ) (n : ℕ) (h : n > 1) :
  ∃ E : ℝ, (E = AB / (n^2 + 1) ∨ E = AB / (n^2 - 1)) ∧ 0 ≤ E ∧ E ≤ AB :=
sorry

end NUMINAMATH_CALUDE_segment_division_l773_77372


namespace NUMINAMATH_CALUDE_lower_variance_more_stable_l773_77311

/-- Represents a set of data -/
structure DataSet where
  variance : ℝ

/-- Defines the stability relation between two data sets -/
def more_stable (a b : DataSet) : Prop := a.variance < b.variance

/-- Theorem stating that a data set with lower variance is more stable -/
theorem lower_variance_more_stable (A B : DataSet) 
  (hA : A.variance = 0.01) (hB : B.variance = 0.1) : 
  more_stable A B := by
  sorry

#check lower_variance_more_stable

end NUMINAMATH_CALUDE_lower_variance_more_stable_l773_77311


namespace NUMINAMATH_CALUDE_complex_roots_cubic_l773_77366

theorem complex_roots_cubic (a b c : ℂ) 
  (h1 : a + b + c = 1)
  (h2 : a * b + a * c + b * c = 0)
  (h3 : a * b * c = -1) :
  (∀ x : ℂ, x^3 - x^2 + 1 = 0 ↔ x = a ∨ x = b ∨ x = c) :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_cubic_l773_77366


namespace NUMINAMATH_CALUDE_second_catch_up_l773_77351

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ

/-- Represents the state of the race -/
structure RaceState where
  runner1 : Runner
  runner2 : Runner
  laps_completed : ℝ

/-- Defines the initial state of the race -/
def initial_state : RaceState :=
  { runner1 := { speed := 3 },
    runner2 := { speed := 1 },
    laps_completed := 0 }

/-- Defines the state after the second runner doubles their speed -/
def intermediate_state : RaceState :=
  { runner1 := { speed := 3 },
    runner2 := { speed := 2 },
    laps_completed := 0.5 }

/-- Theorem stating that the first runner will catch up again when the second runner has completed 2.5 laps -/
theorem second_catch_up (state : RaceState) :
  state.runner1.speed > state.runner2.speed →
  ∃ t : ℝ, t > 0 ∧ 
    state.runner1.speed * t = (state.laps_completed + 2.5) ∧
    state.runner2.speed * t = 2 :=
  sorry

end NUMINAMATH_CALUDE_second_catch_up_l773_77351


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_two_l773_77307

/-- Given two nonconstant geometric sequences with different common ratios,
    if a specific condition is met, then the sum of their common ratios is 2. -/
theorem sum_of_common_ratios_is_two
  (k : ℝ)
  (a₂ a₃ b₂ b₃ : ℝ)
  (ha : a₂ ≠ k ∧ a₃ ≠ a₂)  -- First sequence is nonconstant
  (hb : b₂ ≠ k ∧ b₃ ≠ b₂)  -- Second sequence is nonconstant
  (hseq₁ : ∃ p : ℝ, p ≠ 1 ∧ a₂ = k * p ∧ a₃ = k * p^2)  -- First sequence is geometric
  (hseq₂ : ∃ r : ℝ, r ≠ 1 ∧ b₂ = k * r ∧ b₃ = k * r^2)  -- Second sequence is geometric
  (hdiff : ∀ p r, (a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2) → p ≠ r)  -- Different common ratios
  (hcond : a₃ - b₃ = 2 * (a₂ - b₂))  -- Given condition
  : ∃ p r : ℝ, (a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2) ∧ p + r = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_two_l773_77307


namespace NUMINAMATH_CALUDE_right_triangle_legs_l773_77332

/-- A right triangle with specific median and altitude properties -/
structure RightTriangle where
  -- The length of the median from the right angle vertex
  median : ℝ
  -- The length of the altitude from the right angle vertex
  altitude : ℝ
  -- Condition that the median is 5
  median_is_five : median = 5
  -- Condition that the altitude is 4
  altitude_is_four : altitude = 4

/-- The legs of a right triangle -/
structure TriangleLegs where
  -- The length of the first leg
  leg1 : ℝ
  -- The length of the second leg
  leg2 : ℝ

/-- Theorem stating the legs of the triangle given the median and altitude -/
theorem right_triangle_legs (t : RightTriangle) : 
  ∃ (legs : TriangleLegs), legs.leg1 = 2 * Real.sqrt 5 ∧ legs.leg2 = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l773_77332


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_fifty_l773_77375

theorem largest_multiple_of_seven_less_than_fifty :
  ∃ n : ℕ, n = 49 ∧ 7 ∣ n ∧ n < 50 ∧ ∀ m : ℕ, 7 ∣ m → m < 50 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_fifty_l773_77375


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l773_77394

/-- Coconut grove problem -/
theorem coconut_grove_problem (x : ℝ) 
  (h1 : 60 * (x + 1) + 120 * x + 180 * (x - 1) = 100 * (3 * x)) : 
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l773_77394


namespace NUMINAMATH_CALUDE_cafeteria_stacking_l773_77317

theorem cafeteria_stacking (initial_cartons : ℕ) (cartons_per_stack : ℕ) (teacher_cartons : ℕ) :
  initial_cartons = 799 →
  cartons_per_stack = 6 →
  teacher_cartons = 23 →
  let remaining_cartons := initial_cartons - teacher_cartons
  let full_stacks := remaining_cartons / cartons_per_stack
  let double_stacks := full_stacks / 2
  let leftover_cartons := remaining_cartons % cartons_per_stack + (full_stacks % 2) * cartons_per_stack
  double_stacks = 64 ∧ leftover_cartons = 8 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_stacking_l773_77317


namespace NUMINAMATH_CALUDE_function_non_negative_iff_a_leq_four_l773_77319

theorem function_non_negative_iff_a_leq_four (a : ℝ) :
  (∀ x : ℝ, 2^(2*x) - a * 2^x + 4 ≥ 0) ↔ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_function_non_negative_iff_a_leq_four_l773_77319


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l773_77326

theorem solution_set_of_inequality (x : ℝ) :
  x * |x - 1| > 0 ↔ x ∈ Set.Ioo 0 1 ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l773_77326


namespace NUMINAMATH_CALUDE_coefficient_of_negative_six_xy_l773_77388

/-- The coefficient of a monomial is the numeric factor that multiplies the variable parts. -/
def coefficient (m : ℤ) (x : String) (y : String) : ℤ := m

theorem coefficient_of_negative_six_xy :
  coefficient (-6) "x" "y" = -6 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_negative_six_xy_l773_77388


namespace NUMINAMATH_CALUDE_modified_number_wall_m_value_l773_77338

/-- Represents a modified Number Wall with given values -/
structure ModifiedNumberWall where
  m : ℕ
  row1 : Vector ℕ 4
  row2 : Vector ℕ 3
  row3 : Vector ℕ 2
  row4 : ℕ

/-- The modified Number Wall satisfies the sum property -/
def is_valid_wall (wall : ModifiedNumberWall) : Prop :=
  wall.row1.get 0 = wall.m ∧
  wall.row1.get 1 = 5 ∧
  wall.row1.get 2 = 10 ∧
  wall.row1.get 3 = 6 ∧
  wall.row2.get 1 = 18 ∧
  wall.row4 = 56 ∧
  wall.row2.get 0 = wall.row1.get 0 + wall.row1.get 1 ∧
  wall.row2.get 1 = wall.row1.get 1 + wall.row1.get 2 ∧
  wall.row2.get 2 = wall.row1.get 2 + wall.row1.get 3 ∧
  wall.row3.get 0 = wall.row2.get 0 + wall.row2.get 1 ∧
  wall.row3.get 1 = wall.row2.get 1 + wall.row2.get 2 ∧
  wall.row4 = wall.row3.get 0 + wall.row3.get 1

/-- The value of 'm' in a valid modified Number Wall is 17 -/
theorem modified_number_wall_m_value (wall : ModifiedNumberWall) :
  is_valid_wall wall → wall.m = 17 := by sorry

end NUMINAMATH_CALUDE_modified_number_wall_m_value_l773_77338


namespace NUMINAMATH_CALUDE_correct_arrangement_l773_77396

-- Define the squares
inductive Square
| A | B | C | D | E | F | G | One | Nine

-- Define the arrow directions
def points_to : Square → Square → Prop :=
  fun s1 s2 => match s1, s2 with
    | Square.One, Square.B => True
    | Square.B, Square.E => True
    | Square.E, Square.C => True
    | Square.C, Square.D => True
    | Square.D, Square.A => True
    | Square.A, Square.G => True
    | Square.G, Square.F => True
    | Square.F, Square.Nine => True
    | _, _ => False

-- Define the arrangement
def arrangement : Square → Nat
| Square.A => 6
| Square.B => 2
| Square.C => 4
| Square.D => 5
| Square.E => 3
| Square.F => 8
| Square.G => 7
| Square.One => 1
| Square.Nine => 9

-- Theorem statement
theorem correct_arrangement :
  ∀ s : Square, s ≠ Square.One ∧ s ≠ Square.Nine →
    ∃ s' : Square, points_to s s' ∧ arrangement s' = arrangement s + 1 :=
by sorry

end NUMINAMATH_CALUDE_correct_arrangement_l773_77396


namespace NUMINAMATH_CALUDE_belongs_to_32nd_group_l773_77346

/-- The last number in the n-th group of odd numbers -/
def last_number_in_group (n : ℕ) : ℕ := 2 * n^2 - 1

/-- The first number in the n-th group of odd numbers -/
def first_number_in_group (n : ℕ) : ℕ := 2 * (n-1)^2 + 1

/-- Theorem stating that 1991 belongs to the 32nd group -/
theorem belongs_to_32nd_group : 
  first_number_in_group 32 ≤ 1991 ∧ 1991 ≤ last_number_in_group 32 :=
sorry

end NUMINAMATH_CALUDE_belongs_to_32nd_group_l773_77346


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l773_77329

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_power_dividing_factorial :
  ∃ (n : ℕ), n = 7 ∧ 
  (∀ m : ℕ, m > n → ¬(factorial 30 % (18^m) = 0)) ∧
  (factorial 30 % (18^n) = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l773_77329


namespace NUMINAMATH_CALUDE_remainder_987543_div_12_l773_77380

theorem remainder_987543_div_12 : 987543 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_987543_div_12_l773_77380


namespace NUMINAMATH_CALUDE_speakers_cost_calculation_l773_77392

/-- The amount spent on speakers, given the total amount spent on car parts and the amount spent on new tires. -/
def amount_spent_on_speakers (total_spent : ℚ) (tires_cost : ℚ) : ℚ :=
  total_spent - tires_cost

/-- Theorem stating that the amount spent on speakers is $118.54, given the total spent and the cost of tires. -/
theorem speakers_cost_calculation (total_spent tires_cost : ℚ) 
  (h1 : total_spent = 224.87)
  (h2 : tires_cost = 106.33) : 
  amount_spent_on_speakers total_spent tires_cost = 118.54 := by
  sorry

#eval amount_spent_on_speakers 224.87 106.33

end NUMINAMATH_CALUDE_speakers_cost_calculation_l773_77392


namespace NUMINAMATH_CALUDE_frog_climb_time_l773_77384

/-- Represents the frog's climbing scenario -/
structure FrogClimb where
  wellDepth : ℝ
  climbUp : ℝ
  slipDown : ℝ
  slipTime : ℝ
  timeAt3mBelow : ℝ

/-- Calculates the time taken for the frog to reach the top of the well -/
def timeToReachTop (f : FrogClimb) : ℝ :=
  sorry

/-- Theorem stating that the frog takes 22 minutes to reach the top -/
theorem frog_climb_time (f : FrogClimb) 
  (h1 : f.wellDepth = 12)
  (h2 : f.climbUp = 3)
  (h3 : f.slipDown = 1)
  (h4 : f.slipTime = f.climbUp / 3)
  (h5 : f.timeAt3mBelow = 17) :
  timeToReachTop f = 22 :=
sorry

end NUMINAMATH_CALUDE_frog_climb_time_l773_77384


namespace NUMINAMATH_CALUDE_other_denomination_is_50_l773_77395

/-- Proves that the denomination of the other currency notes is 50 given the problem conditions --/
theorem other_denomination_is_50 
  (total_notes : ℕ) 
  (total_amount : ℕ) 
  (amount_other_denom : ℕ) 
  (h_total_notes : total_notes = 85)
  (h_total_amount : total_amount = 5000)
  (h_amount_other_denom : amount_other_denom = 3500) :
  ∃ (x y D : ℕ), 
    x + y = total_notes ∧ 
    100 * x + D * y = total_amount ∧
    D * y = amount_other_denom ∧
    D = 50 := by
  sorry

#check other_denomination_is_50

end NUMINAMATH_CALUDE_other_denomination_is_50_l773_77395


namespace NUMINAMATH_CALUDE_clock_cost_price_l773_77331

theorem clock_cost_price (total_clocks : ℕ) (clocks_profit1 : ℕ) (clocks_profit2 : ℕ)
  (profit1 : ℚ) (profit2 : ℚ) (uniform_profit : ℚ) (revenue_difference : ℚ) :
  total_clocks = 200 →
  clocks_profit1 = 80 →
  clocks_profit2 = 120 →
  profit1 = 5 / 25 →
  profit2 = 7 / 25 →
  uniform_profit = 6 / 25 →
  revenue_difference = 200 →
  ∃ (cost_price : ℚ),
    cost_price * (clocks_profit1 * (1 + profit1) + clocks_profit2 * (1 + profit2)) -
    cost_price * (total_clocks * (1 + uniform_profit)) = revenue_difference ∧
    cost_price = 125 :=
by sorry

end NUMINAMATH_CALUDE_clock_cost_price_l773_77331


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l773_77350

theorem perpendicular_lines_a_values (a : ℝ) : 
  ((3*a + 2) * (5*a - 2) + (1 - 4*a) * (a + 4) = 0) → (a = 0 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l773_77350


namespace NUMINAMATH_CALUDE_subtract_preserves_inequality_l773_77321

theorem subtract_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtract_preserves_inequality_l773_77321


namespace NUMINAMATH_CALUDE_calculate_expression_l773_77365

theorem calculate_expression : (-2022)^0 - 2 * Real.tan (π/4) + |-2| + Real.sqrt 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l773_77365


namespace NUMINAMATH_CALUDE_inequality_solution_l773_77315

def inequality (x : ℝ) : Prop :=
  1 / (x - 1) - 3 / (x - 2) + 5 / (x - 3) - 1 / (x - 4) < 1 / 24

theorem inequality_solution (x : ℝ) :
  inequality x → (x > -7 ∧ x < 1) ∨ (x > 3 ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l773_77315


namespace NUMINAMATH_CALUDE_darma_peanut_eating_l773_77399

/-- Darma's peanut eating rate -/
def peanuts_per_15_seconds : ℕ := 20

/-- Convert minutes to seconds -/
def minutes_to_seconds (minutes : ℕ) : ℕ := minutes * 60

/-- Calculate peanuts eaten in a given time -/
def peanuts_eaten (seconds : ℕ) : ℕ :=
  (seconds / 15) * peanuts_per_15_seconds

theorem darma_peanut_eating (minutes : ℕ) (h : minutes = 6) :
  peanuts_eaten (minutes_to_seconds minutes) = 480 := by
  sorry

end NUMINAMATH_CALUDE_darma_peanut_eating_l773_77399


namespace NUMINAMATH_CALUDE_line_slope_proof_l773_77362

theorem line_slope_proof (x y : ℝ) :
  x + Real.sqrt 3 * y - 2 = 0 →
  ∃ (α : ℝ), α ∈ Set.Icc 0 π ∧ Real.tan α = -Real.sqrt 3 / 3 ∧ α = 5 * π / 6 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_proof_l773_77362


namespace NUMINAMATH_CALUDE_square_difference_l773_77314

theorem square_difference (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 4) : x^2 - y^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l773_77314


namespace NUMINAMATH_CALUDE_december_sales_fraction_l773_77364

theorem december_sales_fraction (average_sales : ℝ) (h : average_sales > 0) :
  let other_months_total := 11 * average_sales
  let december_sales := 6 * average_sales
  let annual_sales := other_months_total + december_sales
  december_sales / annual_sales = 6 / 17 := by
sorry

end NUMINAMATH_CALUDE_december_sales_fraction_l773_77364


namespace NUMINAMATH_CALUDE_integer_solution_of_inequality_l773_77322

theorem integer_solution_of_inequality (x : ℤ) : 3 ≤ 3 * x + 3 ∧ 3 * x + 3 ≤ 5 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_of_inequality_l773_77322


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l773_77335

/-- Given a geometric sequence {a_n} where a_3 + a_7 = 5, 
    prove that a_2a_4 + 2a_4a_6 + a_6a_8 = 25 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
    (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_sum : a 3 + a 7 = 5) :
    a 2 * a 4 + 2 * a 4 * a 6 + a 6 * a 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l773_77335


namespace NUMINAMATH_CALUDE_taehun_shortest_hair_l773_77357

-- Define the hair lengths
def junseop_hair_cm : ℝ := 9
def junseop_hair_mm : ℝ := 8
def taehun_hair : ℝ := 8.9
def hayul_hair : ℝ := 9.3

-- Define the conversion factor from mm to cm
def mm_to_cm : ℝ := 0.1

-- Theorem statement
theorem taehun_shortest_hair :
  let junseop_total := junseop_hair_cm + junseop_hair_mm * mm_to_cm
  taehun_hair < junseop_total ∧ taehun_hair < hayul_hair := by sorry

end NUMINAMATH_CALUDE_taehun_shortest_hair_l773_77357


namespace NUMINAMATH_CALUDE_triangle_conditions_l773_77391

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a triangle is right-angled
def is_right_triangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

-- Condition A
def condition_A (t : Triangle) : Prop :=
  t.a = 1/3 ∧ t.b = 1/4 ∧ t.c = 1/5

-- Condition B (using angle ratios)
def condition_B (A B C : ℝ) : Prop :=
  A / B = 1/3 ∧ A / C = 1/2 ∧ B / C = 3/2

-- Condition C
def condition_C (t : Triangle) : Prop :=
  (t.b + t.c) * (t.b - t.c) = t.a^2

theorem triangle_conditions :
  (∃ t1 t2 : Triangle, condition_A t1 ∧ is_right_triangle t1 ∧
                       condition_A t2 ∧ ¬is_right_triangle t2) ∧
  (∀ A B C : ℝ, condition_B A B C → A + B + C = 180 → B = 90) ∧
  (∀ t : Triangle, condition_C t → is_right_triangle t) :=
sorry

end NUMINAMATH_CALUDE_triangle_conditions_l773_77391


namespace NUMINAMATH_CALUDE_max_fewer_cards_l773_77370

/-- The set of digits that remain valid when flipped upside down -/
def valid_flip_digits : Finset ℕ := {1, 6, 8, 9}

/-- The set of digits that can be used in the tens place for reversible numbers -/
def valid_tens_digits : Finset ℕ := {0, 1, 6, 8, 9}

/-- The set of digits that can be used in the tens place for symmetrical numbers -/
def symmetrical_tens_digits : Finset ℕ := {0, 1, 8}

/-- The total number of three-digit numbers -/
def total_numbers : ℕ := 900

/-- The number of reversible three-digit numbers -/
def reversible_numbers : ℕ := (valid_tens_digits.card) * (valid_flip_digits.card) * (valid_flip_digits.card)

/-- The number of symmetrical three-digit numbers -/
def symmetrical_numbers : ℕ := (symmetrical_tens_digits.card) * (valid_flip_digits.card)

/-- The maximum number of cards needed considering reversible and symmetrical numbers -/
def max_cards_needed : ℕ := symmetrical_numbers + ((reversible_numbers - symmetrical_numbers) / 2)

/-- The theorem stating the maximum number of fewer cards that need to be printed -/
theorem max_fewer_cards : total_numbers - max_cards_needed = 854 := by sorry

end NUMINAMATH_CALUDE_max_fewer_cards_l773_77370


namespace NUMINAMATH_CALUDE_e_sequence_property_l773_77393

/-- Definition of an E-sequence -/
def is_e_sequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k, k < n - 1 → |a (k + 1) - a k| = 1

/-- The sequence is increasing -/
def is_increasing (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k, k < n - 1 → a k < a (k + 1)

theorem e_sequence_property (a : ℕ → ℤ) :
  is_e_sequence a 2000 →
  a 1 = 13 →
  (is_increasing a 2000 ↔ a 2000 = 2012) :=
by sorry

end NUMINAMATH_CALUDE_e_sequence_property_l773_77393


namespace NUMINAMATH_CALUDE_cake_division_theorem_l773_77349

-- Define the cake and its properties
structure Cake where
  length : ℝ
  width : ℝ
  area : ℝ
  h_area_positive : area > 0
  h_area_calc : area = length * width

-- Define the cuts
structure Cuts where
  x : ℝ
  y : ℝ
  z : ℝ
  h_x_positive : x > 0
  h_y_positive : y > 0
  h_z_positive : z > 0
  h_sum : x + y + z = 1

-- Define the theorem
theorem cake_division_theorem (cake : Cake) (cuts : Cuts) :
  ∃ (piece1 piece2 : ℝ),
    piece1 + piece2 ≥ 0.25 * cake.area ∧
    piece1 = max (cake.length * cuts.x * cake.width) (cake.length * cuts.y * cake.width) ∧
    piece2 = min (cake.length * cuts.x * cake.width) (cake.length * cuts.y * cake.width) ∧
    cake.area - (piece1 + piece2) ≤ 0.75 * cake.area :=
by sorry

end NUMINAMATH_CALUDE_cake_division_theorem_l773_77349


namespace NUMINAMATH_CALUDE_evaluate_expression_l773_77353

theorem evaluate_expression : 
  (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l773_77353


namespace NUMINAMATH_CALUDE_round_trip_speed_calculation_l773_77368

/-- Proves that given specific conditions for a round trip, the return speed must be 45 mph -/
theorem round_trip_speed_calculation (distance : ℝ) (speed_there : ℝ) (avg_speed : ℝ) :
  distance = 180 →
  speed_there = 90 →
  avg_speed = 60 →
  (2 * distance) / (distance / speed_there + distance / (2 * avg_speed - speed_there)) = avg_speed →
  2 * avg_speed - speed_there = 45 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_speed_calculation_l773_77368


namespace NUMINAMATH_CALUDE_min_occupied_seats_l773_77324

/-- Represents a row of seats -/
structure SeatRow :=
  (total : ℕ)
  (occupied : Finset ℕ)
  (h_occupied : occupied.card ≤ total)

/-- Predicts if a new person must sit next to someone -/
def mustSitNext (row : SeatRow) : Prop :=
  ∀ n : ℕ, n ≤ row.total → n ∉ row.occupied →
    (n > 1 ∧ n - 1 ∈ row.occupied) ∨ (n < row.total ∧ n + 1 ∈ row.occupied)

/-- The theorem to be proved -/
theorem min_occupied_seats :
  ∃ (row : SeatRow),
    row.total = 120 ∧
    row.occupied.card = 40 ∧
    mustSitNext row ∧
    ∀ (row' : SeatRow),
      row'.total = 120 →
      row'.occupied.card < 40 →
      ¬mustSitNext row' :=
by sorry

end NUMINAMATH_CALUDE_min_occupied_seats_l773_77324


namespace NUMINAMATH_CALUDE_divisibility_product_l773_77343

theorem divisibility_product (n a b c d : ℤ) 
  (ha : n ∣ a) (hb : n ∣ b) (hc : n ∣ c) (hd : n ∣ d) :
  n ∣ ((a - d) * (b - c)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_product_l773_77343


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l773_77374

theorem quadratic_roots_range (a : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 + (3*a - 1)*x + a + 8 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  x₁ < 1 →
  x₂ > 1 →
  a < -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l773_77374


namespace NUMINAMATH_CALUDE_james_and_louise_ages_l773_77333

theorem james_and_louise_ages :
  ∀ (james louise : ℕ),
  james = louise + 7 →
  james + 10 = 3 * (louise - 3) →
  james + louise = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_james_and_louise_ages_l773_77333


namespace NUMINAMATH_CALUDE_amoeba_count_after_week_l773_77363

/-- The number of amoebas after a given number of days -/
def amoeba_count (days : ℕ) : ℕ :=
  3^days

/-- The theorem stating that after 7 days, there will be 2187 amoebas -/
theorem amoeba_count_after_week : amoeba_count 7 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_week_l773_77363


namespace NUMINAMATH_CALUDE_gummy_worms_problem_l773_77369

theorem gummy_worms_problem (initial_amount : ℕ) : 
  (((initial_amount / 2) / 2) / 2) / 2 = 4 → initial_amount = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_gummy_worms_problem_l773_77369


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l773_77385

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_a1 : a 1 = 2)
    (h_sum : a 3 + a 5 = 8) :
  a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l773_77385


namespace NUMINAMATH_CALUDE_coefficient_x4_is_80_l773_77382

/-- The coefficient of x^4 in the expansion of (4x^2-2x+1)(2x+1)^5 -/
def coefficient_x4 : ℕ :=
  -- Define the coefficient here
  sorry

/-- Theorem stating that the coefficient of x^4 is 80 -/
theorem coefficient_x4_is_80 : coefficient_x4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_is_80_l773_77382


namespace NUMINAMATH_CALUDE_triangle_perimeter_l773_77313

theorem triangle_perimeter (a b c A B C : ℝ) : 
  (c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A) →
  (a = 2) →
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →
  (a + b + c = 6) := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l773_77313


namespace NUMINAMATH_CALUDE_triangle_area_proof_l773_77302

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the area of the triangle is √7/4 under the given conditions. -/
theorem triangle_area_proof (A B C : Real) (a b c : Real) :
  sinA = 2 * sinB →
  c = Real.sqrt 2 →
  cosC = 3 / 4 →
  (1 / 2) * a * b * sinC = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l773_77302


namespace NUMINAMATH_CALUDE_power_of_power_l773_77376

theorem power_of_power (x : ℝ) : (x^3)^2 = x^6 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l773_77376


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l773_77381

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_existence_condition (x : ℕ) :
  x > 0 →
  (triangle_exists 8 11 (2 * x + 1) ↔ x ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l773_77381


namespace NUMINAMATH_CALUDE_sqrt_42_minus_1_range_l773_77390

theorem sqrt_42_minus_1_range : 5 < Real.sqrt 42 - 1 ∧ Real.sqrt 42 - 1 < 6 := by
  have h1 : 36 < 42 := by sorry
  have h2 : 42 < 49 := by sorry
  have h3 : Real.sqrt 36 = 6 := by sorry
  have h4 : Real.sqrt 49 = 7 := by sorry
  sorry

end NUMINAMATH_CALUDE_sqrt_42_minus_1_range_l773_77390


namespace NUMINAMATH_CALUDE_fraction_comparison_l773_77318

theorem fraction_comparison : (5 / 8 : ℚ) - (1 / 16 : ℚ) > (5 / 9 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l773_77318


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l773_77336

theorem simplify_nested_roots (x : ℝ) :
  (((x^16)^(1/8))^(1/4))^3 * (((x^16)^(1/4))^(1/8))^5 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l773_77336


namespace NUMINAMATH_CALUDE_martha_lasagna_cost_l773_77310

/-- The cost of ingredients for Martha's lasagna -/
def lasagna_cost (cheese_quantity : Real) (cheese_price : Real) 
                 (meat_quantity : Real) (meat_price : Real) : Real :=
  cheese_quantity * cheese_price + meat_quantity * meat_price

/-- Theorem: The cost of ingredients for Martha's lasagna is $13 -/
theorem martha_lasagna_cost : 
  lasagna_cost 1.5 6 0.5 8 = 13 := by
  sorry

end NUMINAMATH_CALUDE_martha_lasagna_cost_l773_77310


namespace NUMINAMATH_CALUDE_whiteboard_ink_cost_l773_77397

/-- Calculates the cost of whiteboard ink usage for one day -/
theorem whiteboard_ink_cost (num_classes : ℕ) (boards_per_class : ℕ) (ink_per_board : ℝ) (cost_per_ml : ℝ) : 
  num_classes = 5 → 
  boards_per_class = 2 → 
  ink_per_board = 20 → 
  cost_per_ml = 0.5 → 
  (num_classes * boards_per_class * ink_per_board * cost_per_ml : ℝ) = 100 := by
sorry

end NUMINAMATH_CALUDE_whiteboard_ink_cost_l773_77397


namespace NUMINAMATH_CALUDE_chess_piece_position_l773_77373

theorem chess_piece_position : ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + x*y - 2*y^2 = 13 ∧ x = 5 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_chess_piece_position_l773_77373


namespace NUMINAMATH_CALUDE_train_speed_l773_77304

/-- The speed of a train given its length and time to cross a point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 800) (h2 : time = 10) :
  length / time = 80 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l773_77304


namespace NUMINAMATH_CALUDE_trapezium_side_length_l773_77355

/-- Given a trapezium with the following properties:
  * One parallel side is 18 cm long
  * The distance between parallel sides is 11 cm
  * The area is 209 cm²
  Then the length of the other parallel side is 20 cm -/
theorem trapezium_side_length (a b h : ℝ) (hb : b = 18) (hh : h = 11) (harea : (a + b) * h / 2 = 209) :
  a = 20 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l773_77355


namespace NUMINAMATH_CALUDE_probability_of_pair_letter_l773_77360

def word : String := "PROBABILITY"
def target : String := "PAIR"

theorem probability_of_pair_letter : 
  (word.toList.filter (fun c => target.contains c)).length / word.length = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_pair_letter_l773_77360


namespace NUMINAMATH_CALUDE_fermats_little_theorem_distinct_colorings_l773_77386

theorem fermats_little_theorem (p : ℕ) (n : ℤ) (hp : Nat.Prime p) :
  (↑n ^ p - n : ℤ) % ↑p = 0 := by
  sorry

theorem distinct_colorings (p : ℕ) (n : ℕ) (hp : Nat.Prime p) :
  ∃ k : ℕ, (n ^ p - n : ℕ) / p + n = k := by
  sorry

end NUMINAMATH_CALUDE_fermats_little_theorem_distinct_colorings_l773_77386


namespace NUMINAMATH_CALUDE_gift_cost_per_teacher_l773_77327

/-- Proves that if a person buys gifts for 7 teachers and spends $70 in total, then each gift costs $10. -/
theorem gift_cost_per_teacher (num_teachers : ℕ) (total_spent : ℚ) : 
  num_teachers = 7 → total_spent = 70 → total_spent / num_teachers = 10 := by
  sorry

end NUMINAMATH_CALUDE_gift_cost_per_teacher_l773_77327


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_200_l773_77383

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  AD : ℝ
  BC : ℝ
  angle_BAD : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.AB + t.CD + t.AD + t.BC

/-- Theorem stating that the perimeter of the given trapezoid is 200 units -/
theorem trapezoid_perimeter_is_200 (t : Trapezoid) 
  (h1 : t.AB = 40)
  (h2 : t.CD = 35)
  (h3 : t.AD = 70)
  (h4 : t.BC = 55)
  (h5 : t.angle_BAD = 30 * π / 180)  -- Convert 30° to radians
  : perimeter t = 200 := by
  sorry

#check trapezoid_perimeter_is_200

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_200_l773_77383


namespace NUMINAMATH_CALUDE_chess_group_players_l773_77339

/-- The number of players in a chess group -/
def num_players : ℕ := 8

/-- The total number of games played -/
def total_games : ℕ := 28

/-- Calculates the number of games played given the number of players -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_group_players :
  (games_played num_players = total_games) ∧ 
  (∀ m : ℕ, m ≠ num_players → games_played m ≠ total_games) :=
sorry

end NUMINAMATH_CALUDE_chess_group_players_l773_77339


namespace NUMINAMATH_CALUDE_teaching_fee_sum_l773_77387

theorem teaching_fee_sum (k : ℚ) : 
  (5 * k) / (4 * k) = 5 / 4 →
  (5 * k + 20) / (4 * k + 20) = 6 / 5 →
  (5 * k + 20) + (4 * k + 20) = 220 := by
  sorry

end NUMINAMATH_CALUDE_teaching_fee_sum_l773_77387


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l773_77337

theorem least_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 27) ∧ k ∣ (7 * m + 4))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 27) ∧ k ∣ (7 * n + 4)) ∧
  n = 220 :=
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l773_77337


namespace NUMINAMATH_CALUDE_bananas_per_visit_l773_77348

theorem bananas_per_visit (store_visits : ℕ) (total_bananas : ℕ) (bananas_per_visit : ℕ) : 
  store_visits = 2 → total_bananas = 20 → bananas_per_visit * store_visits = total_bananas → bananas_per_visit = 10 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_visit_l773_77348


namespace NUMINAMATH_CALUDE_larger_cube_volume_l773_77344

/-- The volume of a cube composed of 125 smaller cubes is equal to 125 times the volume of one small cube. -/
theorem larger_cube_volume (small_cube_volume : ℝ) (larger_cube_volume : ℝ) 
  (h1 : small_cube_volume > 0)
  (h2 : larger_cube_volume > 0)
  (h3 : ∃ (n : ℕ), n ^ 3 = 125)
  (h4 : larger_cube_volume = (5 : ℝ) ^ 3 * small_cube_volume) :
  larger_cube_volume = 125 * small_cube_volume := by
  sorry

end NUMINAMATH_CALUDE_larger_cube_volume_l773_77344


namespace NUMINAMATH_CALUDE_cat_weight_l773_77371

theorem cat_weight (num_puppies num_cats : ℕ) (puppy_weight : ℝ) (weight_difference : ℝ) :
  num_puppies = 4 →
  num_cats = 14 →
  puppy_weight = 7.5 →
  weight_difference = 5 →
  puppy_weight + weight_difference = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_cat_weight_l773_77371


namespace NUMINAMATH_CALUDE_price_quantity_difference_l773_77316

/-- Given a price increase and quantity reduction, proves the difference in cost -/
theorem price_quantity_difference (P Q : ℝ) (h_pos_P : P > 0) (h_pos_Q : Q > 0) : 
  (P * 1.1 * (Q * 0.8)) - (P * Q) = -0.12 * (P * Q) := by
  sorry

#check price_quantity_difference

end NUMINAMATH_CALUDE_price_quantity_difference_l773_77316


namespace NUMINAMATH_CALUDE_product_to_standard_form_l773_77308

theorem product_to_standard_form (x : ℝ) : 
  (x - 1) * (x + 3) * (x + 5) = x^3 + 7*x^2 + 7*x - 15 := by
  sorry

end NUMINAMATH_CALUDE_product_to_standard_form_l773_77308


namespace NUMINAMATH_CALUDE_product_pure_imaginary_l773_77328

theorem product_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*Complex.I
  let z₂ : ℂ := 2 + Complex.I
  (∃ b : ℝ, z₁ * z₂ = b * Complex.I) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_l773_77328


namespace NUMINAMATH_CALUDE_sum_of_products_l773_77352

theorem sum_of_products (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = 9)
  (h2 : y^2 + y*z + z^2 = 16)
  (h3 : z^2 + z*x + x^2 = 25) :
  x*y + y*z + z*x = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l773_77352


namespace NUMINAMATH_CALUDE_girls_fraction_l773_77300

theorem girls_fraction (total : ℕ) (middle : ℕ) 
  (h_total : total = 800)
  (h_middle : middle = 330)
  (h_primary : total - middle = 470) :
  ∃ (girls boys : ℕ),
    girls + boys = total ∧
    (7 : ℚ) / 10 * girls + (2 : ℚ) / 5 * boys = total - middle ∧
    (girls : ℚ) / total = 5 / 8 :=
by sorry

end NUMINAMATH_CALUDE_girls_fraction_l773_77300


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l773_77303

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def quadratic_roots_prime (k : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 105 ∧ p * q = k

theorem unique_k_for_prime_roots : ∃! k : ℕ, quadratic_roots_prime k :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l773_77303


namespace NUMINAMATH_CALUDE_function_identity_l773_77379

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : 
  ∀ n : ℕ+, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l773_77379


namespace NUMINAMATH_CALUDE_unique_lcm_solution_l773_77358

theorem unique_lcm_solution : ∃! (n : ℕ), n > 0 ∧ Nat.lcm n (n - 30) = n + 1320 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_lcm_solution_l773_77358


namespace NUMINAMATH_CALUDE_dog_distance_theorem_l773_77356

/-- The problem of calculating the distance run by a dog between two people --/
theorem dog_distance_theorem 
  (anderson_speed baxter_speed dog_speed : ℝ)
  (head_start : ℝ)
  (h_anderson_speed : anderson_speed = 2)
  (h_baxter_speed : baxter_speed = 4)
  (h_dog_speed : dog_speed = 10)
  (h_head_start : head_start = 1) :
  let initial_distance := anderson_speed * head_start
  let relative_speed := baxter_speed - anderson_speed
  let catch_up_time := initial_distance / relative_speed
  dog_speed * catch_up_time = 10 := by sorry

end NUMINAMATH_CALUDE_dog_distance_theorem_l773_77356


namespace NUMINAMATH_CALUDE_abs_plus_one_nonzero_l773_77309

theorem abs_plus_one_nonzero (a : ℚ) : |a| + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_plus_one_nonzero_l773_77309


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l773_77342

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n where a_4 = 4, prove that a_2 * a_6 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geo : IsGeometricSequence a) (h_a4 : a 4 = 4) :
  a 2 * a 6 = 16 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l773_77342


namespace NUMINAMATH_CALUDE_other_communities_count_l773_77389

theorem other_communities_count (total_boys : ℕ) 
  (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 850 →
  muslim_percent = 40 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  ↑⌊(1 - (muslim_percent + hindu_percent + sikh_percent)) * total_boys⌋ = 187 :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l773_77389


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_2_l773_77323

theorem factorization_of_2x_squared_minus_2 (x : ℝ) : 2*x^2 - 2 = 2*(x+1)*(x-1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_2_l773_77323


namespace NUMINAMATH_CALUDE_perfect_square_3_6_4_5_5_4_l773_77398

theorem perfect_square_3_6_4_5_5_4 : ∃ n : ℕ, n ^ 2 = 3^6 * 4^5 * 5^4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_3_6_4_5_5_4_l773_77398


namespace NUMINAMATH_CALUDE_expression_evaluation_l773_77312

theorem expression_evaluation : (2^8 + 4^5) * (1^3 - (-1)^3)^2 = 5120 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l773_77312


namespace NUMINAMATH_CALUDE_mean_is_seven_l773_77361

def pull_up_data : List (ℕ × ℕ) := [(9, 2), (8, 3), (6, 3), (5, 2)]

def total_students : ℕ := 10

theorem mean_is_seven :
  let sum := (pull_up_data.map (λ p => p.1 * p.2)).sum
  sum / total_students = 7 := by sorry

end NUMINAMATH_CALUDE_mean_is_seven_l773_77361


namespace NUMINAMATH_CALUDE_comparison_of_expressions_l773_77306

theorem comparison_of_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 / b + b^2 / a ≥ a + b := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_expressions_l773_77306


namespace NUMINAMATH_CALUDE_factorization_proof_l773_77359

theorem factorization_proof (x : ℝ) : (x + 3)^2 - (x + 3) = (x + 3) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l773_77359


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l773_77301

/-- Two points are symmetric with respect to the y-axis if their y-coordinates are equal
    and their x-coordinates are negatives of each other. -/
def symmetric_y_axis (p q : ℝ × ℝ) : Prop :=
  p.2 = q.2 ∧ p.1 = -q.1

theorem symmetric_points_sum (a b : ℝ) :
  symmetric_y_axis (3, a) (b, 2) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l773_77301


namespace NUMINAMATH_CALUDE_x_minus_y_values_l773_77345

theorem x_minus_y_values (x y : ℝ) (h1 : x^2 = 4) (h2 : |y| = 3) (h3 : x + y < 0) :
  x - y = 1 ∨ x - y = 5 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l773_77345
