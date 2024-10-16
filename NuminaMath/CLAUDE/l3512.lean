import Mathlib

namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3512_351204

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The theorem to be proved -/
theorem parallel_lines_a_value :
  ∀ a : ℝ,
  let l1 : Line := ⟨a - 1, 2, 3⟩
  let l2 : Line := ⟨1, a, 3⟩
  parallel l1 l2 → a = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3512_351204


namespace NUMINAMATH_CALUDE_smallest_n_for_perfect_square_sums_l3512_351291

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def validPermutation (p : List ℕ) : Prop :=
  ∀ i : ℕ, i < p.length - 1 → isPerfectSquare (p[i]! + p[i+1]!)

def consecutiveIntegers (n : ℕ) : List ℕ := List.range n

theorem smallest_n_for_perfect_square_sums : 
  ∀ n : ℕ, n > 1 →
    (∃ p : List ℕ, p.length = n ∧ p.toFinset = (consecutiveIntegers n).toFinset ∧ validPermutation p) 
    ↔ n ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_perfect_square_sums_l3512_351291


namespace NUMINAMATH_CALUDE_intersection_midpoint_distance_l3512_351248

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 - (Real.sqrt 3 / 2) * t, t / 2)

-- Define the curve C
def curve_C (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.sqrt 3 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point P
def point_P : ℝ × ℝ := (Real.sqrt 3, 0)

-- Theorem statement
theorem intersection_midpoint_distance : 
  ∃ (t₁ t₂ : ℝ), 
    let A := line_l t₁
    let B := line_l t₂
    let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)  -- Midpoint of AB
    curve_C (Real.arctan (A.2 / A.1)) = A ∧     -- A is on curve C
    curve_C (Real.arctan (B.2 / B.1)) = B ∧     -- B is on curve C
    Real.sqrt ((D.1 - point_P.1)^2 + (D.2 - point_P.2)^2) = (3 + Real.sqrt 3) / 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_intersection_midpoint_distance_l3512_351248


namespace NUMINAMATH_CALUDE_max_log_sum_and_min_reciprocal_sum_l3512_351295

open Real

theorem max_log_sum_and_min_reciprocal_sum (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + 5 * y = 20) :
  (∃ (u : ℝ), u = log x + log y ∧ ∀ (v : ℝ), v = log x + log y → v ≤ u) ∧
  u = 1 ∧
  (∃ (w : ℝ), w = 1/x + 1/y ∧ ∀ (z : ℝ), z = 1/x + 1/y → w ≤ z) ∧
  w = (7 + 2 * sqrt 10) / 20 := by
  sorry

end NUMINAMATH_CALUDE_max_log_sum_and_min_reciprocal_sum_l3512_351295


namespace NUMINAMATH_CALUDE_investment_interest_rate_l3512_351289

/-- Proves that given the specified conditions, the interest rate for the second part of an investment is 5% -/
theorem investment_interest_rate 
  (total_investment : ℕ)
  (first_part : ℕ)
  (first_rate : ℚ)
  (total_interest : ℕ)
  (h1 : total_investment = 3400)
  (h2 : first_part = 1300)
  (h3 : first_rate = 3 / 100)
  (h4 : total_interest = 144) :
  let second_part := total_investment - first_part
  let first_interest := (first_part : ℚ) * first_rate
  let second_interest := (total_interest : ℚ) - first_interest
  let second_rate := second_interest / (second_part : ℚ)
  second_rate = 5 / 100 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l3512_351289


namespace NUMINAMATH_CALUDE_bruce_mangoes_l3512_351252

/-- Calculates the quantity of mangoes purchased given the total amount paid,
    the quantity and price of grapes, and the price of mangoes. -/
def mangoes_purchased (total_paid : ℕ) (grape_qty : ℕ) (grape_price : ℕ) (mango_price : ℕ) : ℕ :=
  ((total_paid - grape_qty * grape_price) / mango_price : ℕ)

/-- Proves that Bruce purchased 9 kg of mangoes given the problem conditions. -/
theorem bruce_mangoes :
  mangoes_purchased 1055 8 70 55 = 9 := by
  sorry

end NUMINAMATH_CALUDE_bruce_mangoes_l3512_351252


namespace NUMINAMATH_CALUDE_line_equation_l3512_351264

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := x + 2*y - 5 = 0
def l₂ (x y : ℝ) : Prop := x + 2*y - 3 = 0
def l₃ (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the point P
def P : ℝ × ℝ := (-1, 1)

-- Define the property that l passes through P
def passes_through_P (l : ℝ → ℝ) : Prop := l P.1 = P.2

-- Define the property that M is the midpoint of M₁M₂
def is_midpoint (M : ℝ × ℝ) (M₁ M₂ : ℝ × ℝ) : Prop :=
  M.1 = (M₁.1 + M₂.1) / 2 ∧ M.2 = (M₁.2 + M₂.2) / 2

-- Main theorem
theorem line_equation (l : ℝ → ℝ) 
  (h₁ : passes_through_P l)
  (h₂ : ∃ M₁ M₂ : ℝ × ℝ, l₁ M₁.1 M₁.2 ∧ l₂ M₂.1 M₂.2)
  (h₃ : ∃ M : ℝ × ℝ, l₃ M.1 M.2 ∧ ∀ M₁ M₂ : ℝ × ℝ, l₁ M₁.1 M₁.2 → l₂ M₂.1 M₂.2 → is_midpoint M M₁ M₂) :
  ∀ x : ℝ, l x = 1 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3512_351264


namespace NUMINAMATH_CALUDE_actual_time_greater_than_planned_l3512_351267

/-- Represents the running competition scenario -/
structure RunningCompetition where
  V : ℝ  -- Planned constant speed
  D : ℝ  -- Total distance
  V1 : ℝ := 1.25 * V  -- Increased speed for first half
  V2 : ℝ := 0.80 * V  -- Decreased speed for second half

/-- Theorem stating that the actual time is greater than the planned time -/
theorem actual_time_greater_than_planned (rc : RunningCompetition) 
  (h_positive_speed : rc.V > 0) (h_positive_distance : rc.D > 0) : 
  (rc.D / (2 * rc.V1) + rc.D / (2 * rc.V2)) > (rc.D / rc.V) :=
by sorry

end NUMINAMATH_CALUDE_actual_time_greater_than_planned_l3512_351267


namespace NUMINAMATH_CALUDE_jerry_weekly_earnings_l3512_351290

/-- Jerry's weekly earnings calculation --/
theorem jerry_weekly_earnings
  (rate_per_task : ℕ)
  (hours_per_task : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (h1 : rate_per_task = 40)
  (h2 : hours_per_task = 2)
  (h3 : hours_per_day = 10)
  (h4 : days_per_week = 7) :
  (rate_per_task * (hours_per_day / hours_per_task) * days_per_week : ℕ) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_jerry_weekly_earnings_l3512_351290


namespace NUMINAMATH_CALUDE_cubic_inequality_l3512_351246

theorem cubic_inequality (a b c : ℝ) 
  (h : ∃ x₁ x₂ x₃ : ℝ, 
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧
    x₁^3 + a*x₁^2 + b*x₁ + c = 0 ∧
    x₂^3 + a*x₂^2 + b*x₂ + c = 0 ∧
    x₃^3 + a*x₃^2 + b*x₃ + c = 0 ∧
    x₁ + x₂ + x₃ ≤ 1) :
  a^3*(1 + a + b) - 9*c*(3 + 3*a + a^2) ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3512_351246


namespace NUMINAMATH_CALUDE_josh_marbles_calculation_l3512_351249

theorem josh_marbles_calculation (initial_marbles : ℕ) : 
  initial_marbles = 16 → 
  (initial_marbles * 3 * 3 / 4 : ℕ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_calculation_l3512_351249


namespace NUMINAMATH_CALUDE_unique_digit_product_l3512_351207

theorem unique_digit_product (A M C : ℕ) : 
  A < 10 → M < 10 → C < 10 →
  (100 * A + 10 * M + C) * (A + M + C) = 2008 →
  A = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_product_l3512_351207


namespace NUMINAMATH_CALUDE_junior_score_l3512_351282

theorem junior_score (n : ℝ) (junior_proportion : ℝ) (senior_proportion : ℝ) 
  (overall_average : ℝ) (senior_average : ℝ) :
  junior_proportion = 0.3 →
  senior_proportion = 0.7 →
  overall_average = 79 →
  senior_average = 75 →
  junior_proportion + senior_proportion = 1 →
  let junior_score := (overall_average - senior_average * senior_proportion) / junior_proportion
  junior_score = 88 := by
  sorry

end NUMINAMATH_CALUDE_junior_score_l3512_351282


namespace NUMINAMATH_CALUDE_all_sheep_can_be_blue_not_all_sheep_can_be_red_or_green_l3512_351245

/-- Represents the count of sheep of each color -/
structure SheepCounts where
  blue : Nat
  red : Nat
  green : Nat

/-- Represents a transformation of sheep colors -/
inductive SheepTransform
  | BlueRedToGreen
  | BlueGreenToRed
  | RedGreenToBlue

/-- Applies a transformation to the sheep counts -/
def applyTransform (counts : SheepCounts) (transform : SheepTransform) : SheepCounts :=
  match transform with
  | SheepTransform.BlueRedToGreen => 
      ⟨counts.blue - 1, counts.red - 1, counts.green + 2⟩
  | SheepTransform.BlueGreenToRed => 
      ⟨counts.blue - 1, counts.red + 2, counts.green - 1⟩
  | SheepTransform.RedGreenToBlue => 
      ⟨counts.blue + 2, counts.red - 1, counts.green - 1⟩

/-- The initial counts of sheep -/
def initialCounts : SheepCounts := ⟨22, 18, 15⟩

/-- Theorem stating that it's possible for all sheep to become blue -/
theorem all_sheep_can_be_blue :
  ∃ (transforms : List SheepTransform), 
    let finalCounts := transforms.foldl applyTransform initialCounts
    finalCounts.red = 0 ∧ finalCounts.green = 0 ∧ finalCounts.blue > 0 :=
sorry

/-- Theorem stating that it's impossible for all sheep to become red or green -/
theorem not_all_sheep_can_be_red_or_green :
  ¬∃ (transforms : List SheepTransform), 
    let finalCounts := transforms.foldl applyTransform initialCounts
    (finalCounts.blue = 0 ∧ finalCounts.green = 0 ∧ finalCounts.red > 0) ∨
    (finalCounts.blue = 0 ∧ finalCounts.red = 0 ∧ finalCounts.green > 0) :=
sorry

end NUMINAMATH_CALUDE_all_sheep_can_be_blue_not_all_sheep_can_be_red_or_green_l3512_351245


namespace NUMINAMATH_CALUDE_highest_x_value_l3512_351237

theorem highest_x_value (x : ℝ) :
  (((15 * x^2 - 40 * x + 18) / (4 * x - 3) + 7 * x = 9 * x - 2) →
   x ≤ 4) ∧
  (∃ y : ℝ, ((15 * y^2 - 40 * y + 18) / (4 * y - 3) + 7 * y = 9 * y - 2) ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_highest_x_value_l3512_351237


namespace NUMINAMATH_CALUDE_students_practicing_both_sports_l3512_351236

theorem students_practicing_both_sports :
  -- Define variables
  ∀ (F B x : ℕ),
  -- Condition 1: One-fifth of footballers play basketball
  F / 5 = x →
  -- Condition 2: One-seventh of basketball players play football
  B / 7 = x →
  -- Condition 3: 110 students practice exactly one sport
  (F - x) + (B - x) = 110 →
  -- Conclusion: x (students practicing both sports) = 11
  x = 11 := by
sorry

end NUMINAMATH_CALUDE_students_practicing_both_sports_l3512_351236


namespace NUMINAMATH_CALUDE_remainder_when_divided_by_fifteen_l3512_351257

theorem remainder_when_divided_by_fifteen (r : ℕ) (h : r / 15 = 82 / 10) : r % 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_when_divided_by_fifteen_l3512_351257


namespace NUMINAMATH_CALUDE_fraction_zero_l3512_351298

theorem fraction_zero (x : ℝ) (h : x ≠ 1) : (x^2 - 1) / (x - 1) = 0 ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_l3512_351298


namespace NUMINAMATH_CALUDE_number_of_stools_l3512_351281

/-- Represents the number of legs on a stool -/
def stool_legs : ℕ := 3

/-- Represents the number of legs on a chair -/
def chair_legs : ℕ := 4

/-- Represents the total number of legs in the room when people sit on all furniture -/
def total_legs : ℕ := 39

/-- Theorem stating that the number of three-legged stools is 3 -/
theorem number_of_stools (x y : ℕ) 
  (h : stool_legs * x + chair_legs * y = total_legs) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_stools_l3512_351281


namespace NUMINAMATH_CALUDE_complex_argument_bounds_l3512_351241

variable (b : ℝ) (hb : b ≠ 0)
variable (y : ℂ)

theorem complex_argument_bounds :
  (Complex.abs (b * y + y⁻¹) = Real.sqrt 2) →
  (Complex.arg y = π / 4 ∨ Complex.arg y = 7 * π / 4) ∧
  (∀ z : ℂ, Complex.abs (b * z + z⁻¹) = Real.sqrt 2 →
    π / 4 ≤ Complex.arg z ∧ Complex.arg z ≤ 7 * π / 4) :=
by sorry

end NUMINAMATH_CALUDE_complex_argument_bounds_l3512_351241


namespace NUMINAMATH_CALUDE_bus_driver_regular_rate_l3512_351217

/-- Represents the compensation structure and work details of a bus driver --/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeMultiplier : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Calculates the total compensation based on the given compensation structure --/
def calculateTotalCompensation (c : BusDriverCompensation) : ℝ :=
  c.regularRate * c.regularHours + c.regularRate * c.overtimeMultiplier * c.overtimeHours

/-- Theorem stating that the regular rate of $16 per hour satisfies the given conditions --/
theorem bus_driver_regular_rate :
  ∃ (c : BusDriverCompensation),
    c.regularRate = 16 ∧
    c.overtimeMultiplier = 1.75 ∧
    c.regularHours = 40 ∧
    c.overtimeHours = 12 ∧
    c.totalCompensation = 976 ∧
    calculateTotalCompensation c = c.totalCompensation :=
  sorry

end NUMINAMATH_CALUDE_bus_driver_regular_rate_l3512_351217


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l3512_351255

/-- Given a 2x2 matrix A with inverse [[3, -1], [1, 1]], 
    prove that the inverse of A^3 is [[20, -12], [12, -4]] -/
theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = ![![3, -1], ![1, 1]]) : 
  (A^3)⁻¹ = ![![20, -12], ![12, -4]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l3512_351255


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3512_351215

/-- The line equation is of the form (a-1)x - y + 2a + 1 = 0 where a is a real number -/
def line_equation (a x y : ℝ) : Prop :=
  (a - 1) * x - y + 2 * a + 1 = 0

/-- Theorem: The line always passes through the point (-2, 3) for all real values of a -/
theorem line_passes_through_fixed_point :
  ∀ a : ℝ, line_equation a (-2) 3 :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3512_351215


namespace NUMINAMATH_CALUDE_least_b_value_l3512_351277

def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem least_b_value (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h_a_factors : num_factors a = 4) 
  (h_b_factors : num_factors b = a) 
  (h_b_div_a : a ∣ b) : 
  ∀ c, c > 0 ∧ num_factors c = a ∧ a ∣ c → b ≤ c ∧ b = 12 := by
  sorry

end NUMINAMATH_CALUDE_least_b_value_l3512_351277


namespace NUMINAMATH_CALUDE_max_puzzles_in_club_l3512_351230

/-- Represents a math club with members solving puzzles -/
structure MathClub where
  members : ℕ
  average_puzzles : ℕ
  min_puzzles : ℕ

/-- Calculates the maximum number of puzzles one member can solve -/
def max_puzzles_by_one (club : MathClub) : ℕ :=
  club.members * club.average_puzzles - (club.members - 1) * club.min_puzzles

/-- Theorem stating the maximum number of puzzles solved by one member in the given conditions -/
theorem max_puzzles_in_club (club : MathClub) 
  (h_members : club.members = 40)
  (h_average : club.average_puzzles = 6)
  (h_min : club.min_puzzles = 2) :
  max_puzzles_by_one club = 162 := by
  sorry

#eval max_puzzles_by_one ⟨40, 6, 2⟩

end NUMINAMATH_CALUDE_max_puzzles_in_club_l3512_351230


namespace NUMINAMATH_CALUDE_salad_dressing_weight_l3512_351239

theorem salad_dressing_weight (bowl_capacity : ℝ) (oil_fraction vinegar_fraction : ℝ)
  (oil_density vinegar_density lemon_juice_density : ℝ) :
  bowl_capacity = 200 →
  oil_fraction = 3/5 →
  vinegar_fraction = 1/4 →
  oil_density = 5 →
  vinegar_density = 4 →
  lemon_juice_density = 2.5 →
  let lemon_juice_fraction : ℝ := 1 - oil_fraction - vinegar_fraction
  let oil_volume : ℝ := bowl_capacity * oil_fraction
  let vinegar_volume : ℝ := bowl_capacity * vinegar_fraction
  let lemon_juice_volume : ℝ := bowl_capacity * lemon_juice_fraction
  let total_weight : ℝ := oil_volume * oil_density + vinegar_volume * vinegar_density + lemon_juice_volume * lemon_juice_density
  total_weight = 875 := by
sorry


end NUMINAMATH_CALUDE_salad_dressing_weight_l3512_351239


namespace NUMINAMATH_CALUDE_solve_system_l3512_351243

theorem solve_system (a b : ℝ) 
  (eq1 : a * (a - 4) = 5)
  (eq2 : b * (b - 4) = 5)
  (neq : a ≠ b)
  (sum : a + b = 4) :
  a = -1 := by sorry

end NUMINAMATH_CALUDE_solve_system_l3512_351243


namespace NUMINAMATH_CALUDE_second_year_interest_rate_l3512_351238

/-- Calculates the interest rate for the second year given the initial principal,
    first year interest rate, and final amount after two years. -/
theorem second_year_interest_rate
  (initial_principal : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_principal = 8000)
  (h2 : first_year_rate = 0.04)
  (h3 : final_amount = 8736) :
  let first_year_amount := initial_principal * (1 + first_year_rate)
  let second_year_rate := (final_amount / first_year_amount) - 1
  second_year_rate = 0.05 := by
sorry

end NUMINAMATH_CALUDE_second_year_interest_rate_l3512_351238


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_range_of_a_l3512_351288

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem for the range of a where f(x) ≥ a^2 - a - 2 for all x in ℝ
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 - a - 2) ↔ -1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_range_of_a_l3512_351288


namespace NUMINAMATH_CALUDE_simplify_expression_l3512_351200

theorem simplify_expression (y : ℝ) : 
  y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8) = 4 * y^3 - 6 * y^2 + 15 * y - 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3512_351200


namespace NUMINAMATH_CALUDE_initial_juice_amount_l3512_351263

theorem initial_juice_amount (total : ℚ) : 
  (total * (1 - 1/6) * (1 - 2/5) * (1 - 2/3) = 120) → total = 720 := by
  sorry

end NUMINAMATH_CALUDE_initial_juice_amount_l3512_351263


namespace NUMINAMATH_CALUDE_estimate_fish_population_l3512_351296

/-- Estimates the total number of fish in a lake using the mark and recapture method. -/
theorem estimate_fish_population (m n k : ℕ) (h : k > 0) :
  let estimated_total := m * n / k
  ∃ x : ℚ, x = estimated_total ∧ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l3512_351296


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3512_351283

/-- A regular polygon with exterior angles measuring 18° has 20 sides -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 / n = 18) → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3512_351283


namespace NUMINAMATH_CALUDE_students_passed_both_tests_l3512_351227

theorem students_passed_both_tests 
  (total : ℕ) 
  (passed_long_jump : ℕ) 
  (passed_shot_put : ℕ) 
  (failed_both : ℕ) 
  (h1 : total = 50)
  (h2 : passed_long_jump = 40)
  (h3 : passed_shot_put = 31)
  (h4 : failed_both = 4) :
  ∃ (passed_both : ℕ), 
    passed_both = 25 ∧ 
    total = passed_both + (passed_long_jump - passed_both) + (passed_shot_put - passed_both) + failed_both :=
by sorry

end NUMINAMATH_CALUDE_students_passed_both_tests_l3512_351227


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l3512_351271

/-- The coefficient of x^2 in the expansion of (1/x - √x)^10 is 45 -/
theorem coefficient_x_squared_expansion (x : ℝ) : 
  (Finset.range 11).sum (fun k => (-1)^k * (Nat.choose 10 k : ℝ) * x^((3*k:ℤ)/2 - 5)) = 45 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l3512_351271


namespace NUMINAMATH_CALUDE_logarithm_calculation_l3512_351209

theorem logarithm_calculation : (Real.log 128 / Real.log 2) / (Real.log 64 / Real.log 2) - (Real.log 256 / Real.log 2) / (Real.log 16 / Real.log 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_calculation_l3512_351209


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l3512_351278

theorem fourth_term_of_geometric_progression (a b c : ℝ) :
  a = Real.sqrt 2 →
  b = Real.rpow 2 (1/3) →
  c = Real.rpow 2 (1/6) →
  b / a = c / b →
  c * (b / a) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l3512_351278


namespace NUMINAMATH_CALUDE_smallest_x_value_l3512_351247

theorem smallest_x_value (x : ℝ) : 
  (4 * x / 10 + 1 / (4 * x) = 5 / 8) → 
  x ≥ (25 - Real.sqrt 1265) / 32 ∧ 
  ∃ y : ℝ, y = (25 - Real.sqrt 1265) / 32 ∧ 4 * y / 10 + 1 / (4 * y) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3512_351247


namespace NUMINAMATH_CALUDE_log_identity_l3512_351250

theorem log_identity (a b P : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≠ 1) (ha1 : a ≠ 1) :
  (Real.log P / Real.log a) / (Real.log P / Real.log (a * b)) = 1 + Real.log b / Real.log a :=
by sorry

end NUMINAMATH_CALUDE_log_identity_l3512_351250


namespace NUMINAMATH_CALUDE_minimal_fraction_sum_l3512_351273

theorem minimal_fraction_sum (a b : ℕ+) (h : (9:ℚ)/22 < (a:ℚ)/b ∧ (a:ℚ)/b < 5/11) :
  (∃ (c d : ℕ+), (9:ℚ)/22 < (c:ℚ)/d ∧ (c:ℚ)/d < 5/11 ∧ c.val + d.val < a.val + b.val) ∨ (a = 3 ∧ b = 7) :=
sorry

end NUMINAMATH_CALUDE_minimal_fraction_sum_l3512_351273


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l3512_351293

-- Define the siblings
inductive Sibling : Type
| Emily : Sibling
| Sam : Sibling
| Nora : Sibling
| Oliver : Sibling
| Jack : Sibling

-- Define the pizza consumption for each sibling
def pizza_consumption (s : Sibling) : Rat :=
  match s with
  | Sibling.Emily => 1/6
  | Sibling.Sam => 1/4
  | Sibling.Nora => 1/3
  | Sibling.Oliver => 1/8
  | Sibling.Jack => 1 - (1/6 + 1/4 + 1/3 + 1/8)

-- Define a function to compare pizza consumption
def consumes_more (s1 s2 : Sibling) : Prop :=
  pizza_consumption s1 > pizza_consumption s2

-- State the theorem
theorem pizza_consumption_order :
  consumes_more Sibling.Nora Sibling.Sam ∧
  consumes_more Sibling.Sam Sibling.Emily ∧
  consumes_more Sibling.Emily Sibling.Jack ∧
  consumes_more Sibling.Jack Sibling.Oliver :=
by sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l3512_351293


namespace NUMINAMATH_CALUDE_sufficient_condition_sum_greater_than_double_l3512_351219

theorem sufficient_condition_sum_greater_than_double (a b c : ℝ) :
  a > c ∧ b > c → a + b > 2 * c := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_sum_greater_than_double_l3512_351219


namespace NUMINAMATH_CALUDE_solutions_difference_squared_l3512_351225

theorem solutions_difference_squared (α β : ℝ) : 
  α ≠ β ∧ 
  α^2 - 3*α + 1 = 0 ∧ 
  β^2 - 3*β + 1 = 0 → 
  (α - β)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_solutions_difference_squared_l3512_351225


namespace NUMINAMATH_CALUDE_general_term_is_arithmetic_l3512_351232

-- Define the sequence a_n and its sum S_n
def S (n : ℕ) : ℕ := n^2 + 2*n

def a : ℕ → ℕ := fun n => S n - S (n-1)

-- Theorem 1: The general term of the sequence
theorem general_term : ∀ n : ℕ, n > 0 → a n = 2*n + 1 :=
sorry

-- Definition of arithmetic sequence
def is_arithmetic_sequence (f : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, n > 1 → f n - f (n-1) = d

-- Theorem 2: The sequence is arithmetic
theorem is_arithmetic : is_arithmetic_sequence a :=
sorry

end NUMINAMATH_CALUDE_general_term_is_arithmetic_l3512_351232


namespace NUMINAMATH_CALUDE_f_minimum_l3512_351274

noncomputable def f (a x : ℝ) : ℝ := x^2 + |x - a| - 1

theorem f_minimum (a : ℝ) : 
  (∀ x, f a x ≥ -a - 5/4 ∧ ∃ x, f a x = -a - 5/4) ∨
  (∀ x, f a x ≥ a^2 - 1 ∧ ∃ x, f a x = a^2 - 1) ∨
  (∀ x, f a x ≥ a - 5/4 ∧ ∃ x, f a x = a - 5/4) :=
by sorry

end NUMINAMATH_CALUDE_f_minimum_l3512_351274


namespace NUMINAMATH_CALUDE_hospital_nurse_count_l3512_351244

/-- Given a hospital with doctors and nurses, calculate the number of nurses -/
theorem hospital_nurse_count 
  (total : ℕ) -- Total number of doctors and nurses
  (doc_ratio : ℕ) -- Ratio part for doctors
  (nurse_ratio : ℕ) -- Ratio part for nurses
  (h_total : total = 200) -- Total is 200
  (h_ratio : doc_ratio = 4 ∧ nurse_ratio = 6) -- Ratio is 4:6
  : (nurse_ratio : ℚ) / (doc_ratio + nurse_ratio) * total = 120 := by
  sorry

end NUMINAMATH_CALUDE_hospital_nurse_count_l3512_351244


namespace NUMINAMATH_CALUDE_work_completion_theorem_l3512_351222

/-- Represents the number of days it takes for a person to complete the work alone -/
structure WorkRate :=
  (days : ℝ)
  (positive : days > 0)

/-- Represents the state of the work project -/
structure WorkProject :=
  (rate_a : WorkRate)
  (rate_b : WorkRate)
  (total_days : ℝ)
  (a_left_before : Bool)

/-- Calculate the number of days A left before completion -/
def days_a_left_before (project : WorkProject) : ℝ :=
  sorry

theorem work_completion_theorem (project : WorkProject) 
  (h1 : project.rate_a.days = 10)
  (h2 : project.rate_b.days = 20)
  (h3 : project.total_days = 10)
  (h4 : project.a_left_before = true) :
  days_a_left_before project = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l3512_351222


namespace NUMINAMATH_CALUDE_function_equality_up_to_constant_l3512_351258

theorem function_equality_up_to_constant 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g) 
  (h : ∀ x, deriv f x = deriv g x) : 
  ∃ C, ∀ x, f x = g x + C :=
sorry

end NUMINAMATH_CALUDE_function_equality_up_to_constant_l3512_351258


namespace NUMINAMATH_CALUDE_total_interest_after_ten_years_l3512_351216

/-- Calculate the total interest after 10 years given the conditions -/
theorem total_interest_after_ten_years
  (P : ℝ) -- Principal amount
  (R : ℝ) -- Interest rate (in percentage per annum)
  (h1 : P * R * 10 / 100 = 600) -- Simple interest on P for 10 years is Rs. 600
  : P * R * 5 / 100 + 3 * P * R * 5 / 100 = 1200 := by
  sorry

#check total_interest_after_ten_years

end NUMINAMATH_CALUDE_total_interest_after_ten_years_l3512_351216


namespace NUMINAMATH_CALUDE_division_remainder_l3512_351284

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 725) (h2 : divisor = 36) (h3 : quotient = 20) : 
  dividend % divisor = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l3512_351284


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3512_351213

/-- Given a line L1 with equation 6x - 3y = 9 and a point P (1, -2),
    prove that the line L2 passing through P and parallel to L1
    has the equation y = 2x - 4 in slope-intercept form. -/
theorem parallel_line_through_point (x y : ℝ) :
  (6 * x - 3 * y = 9) →  -- equation of L1
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ 6 * x - 3 * y = 9) →  -- L1 in slope-intercept form
  (∀ m b : ℝ, y = m * x + b ∧ 6 * x - 3 * y = 9 → m = 2) →  -- slope of L1 is 2
  (∃ m b : ℝ, y = m * x + b ∧ y = 2 * x + b ∧ -2 = 2 * 1 + b) →  -- L2 passes through (1, -2) and has slope 2
  y = 2 * x - 4  -- equation of L2
  := by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3512_351213


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l3512_351253

theorem polygon_exterior_angles (n : ℕ) (h : n > 2) :
  (n : ℝ) * 60 = 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l3512_351253


namespace NUMINAMATH_CALUDE_school_C_sample_size_l3512_351251

/-- Represents the number of teachers in each school -/
structure SchoolPopulation where
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ

/-- Calculates the sample size for a given school in stratified sampling -/
def stratifiedSampleSize (totalSample : ℕ) (schoolPop : SchoolPopulation) (schoolSize : ℕ) : ℕ :=
  (schoolSize * totalSample) / (schoolPop.schoolA + schoolPop.schoolB + schoolPop.schoolC)

/-- Theorem: The stratified sample size for school C is 10 -/
theorem school_C_sample_size :
  let totalSample : ℕ := 60
  let schoolPop : SchoolPopulation := { schoolA := 180, schoolB := 270, schoolC := 90 }
  stratifiedSampleSize totalSample schoolPop schoolPop.schoolC = 10 := by
  sorry


end NUMINAMATH_CALUDE_school_C_sample_size_l3512_351251


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3512_351218

theorem quadratic_equation_roots (x y : ℝ) : 
  x + y = 10 ∧ |x - y| = 12 → x^2 - 10*x - 11 = 0 ∨ y^2 - 10*y - 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3512_351218


namespace NUMINAMATH_CALUDE_tiffany_lives_l3512_351235

theorem tiffany_lives (initial_lives lost_lives gained_lives final_lives : ℕ) : 
  lost_lives = 14 →
  gained_lives = 27 →
  final_lives = 56 →
  final_lives = initial_lives - lost_lives + gained_lives →
  initial_lives = 43 :=
by sorry

end NUMINAMATH_CALUDE_tiffany_lives_l3512_351235


namespace NUMINAMATH_CALUDE_park_tree_increase_l3512_351265

/-- Represents the state of trees in the park -/
structure ParkState where
  maples : ℕ
  lindens : ℕ

/-- Calculates the total number of trees in the park -/
def total_trees (state : ParkState) : ℕ := state.maples + state.lindens

/-- Calculates the percentage of maples in the park -/
def maple_percentage (state : ParkState) : ℚ :=
  state.maples / (total_trees state)

/-- The initial state of the park -/
def initial_state : ParkState := sorry

/-- The state after planting lindens in spring -/
def spring_state : ParkState := sorry

/-- The final state after planting maples in autumn -/
def autumn_state : ParkState := sorry

theorem park_tree_increase :
  maple_percentage initial_state = 3/5 →
  maple_percentage spring_state = 1/5 →
  maple_percentage autumn_state = 3/5 →
  total_trees autumn_state = 6 * total_trees initial_state :=
sorry

end NUMINAMATH_CALUDE_park_tree_increase_l3512_351265


namespace NUMINAMATH_CALUDE_min_value_theorem_l3512_351285

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 2*a*b + b^2 + 3*c^2 ≥ 324 ∧ ∃ (a' b' c' : ℝ), a'^2 + 2*a'*b' + b'^2 + 3*c'^2 = 324 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3512_351285


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3512_351202

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, k * x^2 + x + k > 0) ↔ k > (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3512_351202


namespace NUMINAMATH_CALUDE_blue_tetrahedron_volume_l3512_351272

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem blue_tetrahedron_volume (s : ℝ) (h : s = 8) :
  let cube_volume := s^3
  let small_tetrahedron_volume := (1/6) * s^3
  cube_volume - 4 * small_tetrahedron_volume = (512:ℝ)/3 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_tetrahedron_volume_l3512_351272


namespace NUMINAMATH_CALUDE_expression_evaluation_l3512_351208

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = 3) : 
  (3*x^2 + y)^2 - (3*x^2 - y)^2 = 144 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3512_351208


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_perpendicular_parallel_l3512_351221

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations for parallel and perpendicular
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (planeparallel : Plane → Plane → Prop)
variable (planeperpendicular : Plane → Plane → Prop)
variable (lineplaneparallel : Line → Plane → Prop)
variable (lineplaneperpendicular : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ " => parallel
local infix:50 " ⊥ " => perpendicular
local infix:50 " ∥ₚ " => planeparallel
local infix:50 " ⊥ₚ " => planeperpendicular
local infix:50 " ∥ₗₚ " => lineplaneparallel
local infix:50 " ⊥ₗₚ " => lineplaneperpendicular

-- Theorem statements
theorem perpendicular_transitivity 
  (m n : Line) (α β : Plane) :
  m ⊥ₗₚ α → n ⊥ₗₚ β → α ⊥ₚ β → m ⊥ n :=
sorry

theorem perpendicular_parallel 
  (m n : Line) (α β : Plane) :
  m ⊥ₗₚ α → n ∥ₗₚ β → α ∥ₚ β → m ⊥ n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_perpendicular_parallel_l3512_351221


namespace NUMINAMATH_CALUDE_red_marbles_count_l3512_351299

theorem red_marbles_count :
  ∀ (total blue red yellow : ℕ),
    total = 85 →
    blue = 3 * red →
    yellow = 29 →
    total = red + blue + yellow →
    red = 14 := by
  sorry

end NUMINAMATH_CALUDE_red_marbles_count_l3512_351299


namespace NUMINAMATH_CALUDE_volume_of_cubes_l3512_351203

/-- Given two cubes where the ratio of their edges is 3:1 and the volume of the smaller cube is 8 units,
    the volume of the larger cube is 216 units. -/
theorem volume_of_cubes (a b : ℝ) (h1 : a / b = 3) (h2 : b^3 = 8) : a^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_cubes_l3512_351203


namespace NUMINAMATH_CALUDE_valid_pairs_l3512_351270

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def valid_pair (a b : ℕ) : Prop :=
  is_integer ((a^2 + b) / (b^2 - a)) ∧
  is_integer ((b^2 + a) / (a^2 - b))

theorem valid_pairs :
  ∀ a b : ℕ, valid_pair a b ↔
    ((a = 1 ∧ b = 2) ∨
     (a = 2 ∧ b = 1) ∨
     (a = 2 ∧ b = 2) ∨
     (a = 2 ∧ b = 3) ∨
     (a = 3 ∧ b = 2) ∨
     (a = 3 ∧ b = 3)) :=
sorry

end NUMINAMATH_CALUDE_valid_pairs_l3512_351270


namespace NUMINAMATH_CALUDE_polynomial_value_l3512_351211

/-- Given a polynomial p(x) = a(x³ - x² + 3x) + b(2x² + x) + x³ - 5,
    if p(2) = -17, then p(-2) = -1 -/
theorem polynomial_value (a b : ℝ) : 
  let p : ℝ → ℝ := λ x => a*(x^3 - x^2 + 3*x) + b*(2*x^2 + x) + x^3 - 5
  (p 2 = -17) → (p (-2) = -1) := by
  sorry


end NUMINAMATH_CALUDE_polynomial_value_l3512_351211


namespace NUMINAMATH_CALUDE_project_completion_equivalence_l3512_351210

/-- Represents the time taken to complete a project given the number of workers -/
def project_completion_time (num_workers : ℕ) (days : ℚ) : Prop :=
  num_workers * days = 120 * 7

theorem project_completion_equivalence :
  project_completion_time 120 7 → project_completion_time 80 (21/2) := by
  sorry

end NUMINAMATH_CALUDE_project_completion_equivalence_l3512_351210


namespace NUMINAMATH_CALUDE_circle_max_values_l3512_351201

theorem circle_max_values (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ - x₀ = Real.sqrt 6 - 2) ∧
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀^2 + y₀^2 = 7 + 4*Real.sqrt 3) ∧
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀ ≠ 0 ∧ y₀ / x₀ = Real.sqrt 3) ∧
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀ + y₀ = 2 + Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_max_values_l3512_351201


namespace NUMINAMATH_CALUDE_students_watching_l3512_351266

theorem students_watching (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 33 → 
  total = boys + girls → 
  (2 * boys + 2 * girls) / 3 = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_students_watching_l3512_351266


namespace NUMINAMATH_CALUDE_population_change_l3512_351297

theorem population_change (k m : ℝ) : 
  let decrease_factor : ℝ := 1 - k / 100
  let increase_factor : ℝ := 1 + m / 100
  let total_factor : ℝ := decrease_factor * increase_factor
  total_factor = 1 + (m - k - k * m / 100) / 100 := by sorry

end NUMINAMATH_CALUDE_population_change_l3512_351297


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3512_351231

/-- The parabola equation: x = 3y^2 + 5y - 4 -/
def parabola (x y : ℝ) : Prop := x = 3 * y^2 + 5 * y - 4

/-- The line equation: x = k -/
def line (x k : ℝ) : Prop := x = k

/-- The condition for a single intersection point -/
def single_intersection (k : ℝ) : Prop :=
  ∃! y, parabola k y ∧ line k k

theorem parabola_line_intersection :
  ∀ k : ℝ, single_intersection k ↔ k = -23/12 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3512_351231


namespace NUMINAMATH_CALUDE_volunteer_comprehensive_score_l3512_351228

/-- Calculates the comprehensive score of a volunteer guide based on test scores and weights -/
def comprehensive_score (written_score trial_score interview_score : ℝ)
  (written_weight trial_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + trial_score * trial_weight + interview_score * interview_weight

/-- Theorem stating that the comprehensive score of the volunteer guide is 92.4 points -/
theorem volunteer_comprehensive_score :
  comprehensive_score 90 94 92 0.3 0.5 0.2 = 92.4 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_comprehensive_score_l3512_351228


namespace NUMINAMATH_CALUDE_equation_solution_l3512_351261

theorem equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  let x : ℝ := (5 * m^2 + 18 * m * n + 18 * n^2) / (10 * m + 6 * n)
  (x + 2 * m)^2 - (x - 3 * n)^2 = 9 * (m + n)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3512_351261


namespace NUMINAMATH_CALUDE_park_planting_problem_l3512_351214

/-- The number of short bushes to be planted in a park -/
def short_bushes_to_plant (current_short_bushes total_short_bushes_after : ℕ) : ℕ :=
  total_short_bushes_after - current_short_bushes

/-- Theorem stating that 20 short bushes will be planted -/
theorem park_planting_problem :
  short_bushes_to_plant 37 57 = 20 := by
  sorry

end NUMINAMATH_CALUDE_park_planting_problem_l3512_351214


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3512_351276

theorem sum_of_a_and_b (a b : ℚ) (h1 : 3 * a + 7 * b = 12) (h2 : 9 * a + 2 * b = 23) :
  a + b = 176 / 57 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3512_351276


namespace NUMINAMATH_CALUDE_ab_over_c_equals_two_l3512_351279

theorem ab_over_c_equals_two 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_eq1 : a * b - c = 3) 
  (h_eq2 : a * b * c = 18) : 
  a * b / c = 2 := by
sorry

end NUMINAMATH_CALUDE_ab_over_c_equals_two_l3512_351279


namespace NUMINAMATH_CALUDE_notebook_cost_proof_l3512_351233

theorem notebook_cost_proof :
  ∀ (s c n : ℕ),
    s ≤ 36 →                     -- number of students who bought notebooks
    s > 36 / 2 →                 -- at least half of the students
    n > 2 →                      -- more than 2 notebooks per student
    c > n →                      -- cost in cents greater than number of notebooks
    s * c * n = 3969 →           -- total cost in cents
    c = 27 :=                    -- cost per notebook is 27 cents
by sorry

end NUMINAMATH_CALUDE_notebook_cost_proof_l3512_351233


namespace NUMINAMATH_CALUDE_points_in_quadrant_I_l3512_351294

theorem points_in_quadrant_I (x y : ℝ) : 
  y > -x + 6 ∧ y > 3*x - 2 → x > 0 ∧ y > 0 := by
sorry

end NUMINAMATH_CALUDE_points_in_quadrant_I_l3512_351294


namespace NUMINAMATH_CALUDE_remainder_3_2015_mod_13_l3512_351268

theorem remainder_3_2015_mod_13 : ∃ k : ℤ, 3^2015 = 13 * k + 9 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_3_2015_mod_13_l3512_351268


namespace NUMINAMATH_CALUDE_complex_subtraction_l3512_351242

theorem complex_subtraction (c d : ℂ) (h1 : c = 5 - 3*I) (h2 : d = 2 - I) : 
  c - 3*d = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3512_351242


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l3512_351262

/-- The number of unique arrangements of letters in "BANANA" -/
def banana_arrangements : ℕ := 
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of "BANANA" is 60 -/
theorem banana_arrangements_count : banana_arrangements = 60 := by
  sorry

#eval banana_arrangements

end NUMINAMATH_CALUDE_banana_arrangements_count_l3512_351262


namespace NUMINAMATH_CALUDE_opposite_sides_iff_in_set_l3512_351256

/-- The set of real numbers a for which points A and B lie on opposite sides of the line 3x - y = 4 -/
def opposite_sides_set : Set ℝ :=
  {a | a < -1 ∨ (-1/3 < a ∧ a < 0) ∨ a > 8/3}

/-- Point A coordinates satisfy the given equation -/
def point_A (a x y : ℝ) : Prop :=
  26 * a^2 - 22 * a * x - 20 * a * y + 5 * x^2 + 8 * x * y + 4 * y^2 = 0

/-- Parabola equation with vertex at point B -/
def parabola (a x y : ℝ) : Prop :=
  a * x^2 + 2 * a^2 * x - a * y + a^3 + 1 = 0

/-- Line equation -/
def line (x y : ℝ) : Prop :=
  3 * x - y = 4

/-- Main theorem: A and B lie on opposite sides of the line if and only if a is in the opposite_sides_set -/
theorem opposite_sides_iff_in_set (a : ℝ) :
  (∃ x_a y_a x_b y_b : ℝ,
    point_A a x_a y_a ∧
    parabola a x_b y_b ∧
    ¬line x_a y_a ∧
    ¬line x_b y_b ∧
    (3 * x_a - y_a - 4) * (3 * x_b - y_b - 4) < 0) ↔
  a ∈ opposite_sides_set := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_iff_in_set_l3512_351256


namespace NUMINAMATH_CALUDE_max_cos_diff_l3512_351234

theorem max_cos_diff (x y : Real) (h : Real.sin x - Real.sin y = 3/4) :
  ∃ (max_val : Real), max_val = 23/32 ∧ 
    ∀ (z w : Real), Real.sin z - Real.sin w = 3/4 → Real.cos (z - w) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_cos_diff_l3512_351234


namespace NUMINAMATH_CALUDE_horner_method_correct_l3512_351286

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^6 - 5x^5 + 6x^4 + x^2 + 0.3x + 2 -/
def f : ℝ → ℝ := fun x => x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

theorem horner_method_correct :
  let coeffs := [1, -5, 6, 0, 1, 0.3, 2]
  horner_eval coeffs (-2) = f (-2) ∧ f (-2) = 325.4 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_correct_l3512_351286


namespace NUMINAMATH_CALUDE_excess_calories_is_770_l3512_351240

/-- Calculates the excess calories consumed by James after snacking and exercising -/
def excess_calories : ℕ :=
  let cheezit_bags : ℕ := 3
  let cheezit_oz_per_bag : ℕ := 2
  let cheezit_cal_per_oz : ℕ := 150
  let chocolate_bars : ℕ := 2
  let chocolate_cal_per_bar : ℕ := 250
  let popcorn_cal : ℕ := 500
  let run_minutes : ℕ := 40
  let run_cal_per_minute : ℕ := 12
  let swim_minutes : ℕ := 30
  let swim_cal_per_minute : ℕ := 15
  let cycle_minutes : ℕ := 20
  let cycle_cal_per_minute : ℕ := 10

  let total_calories_consumed : ℕ := 
    cheezit_bags * cheezit_oz_per_bag * cheezit_cal_per_oz +
    chocolate_bars * chocolate_cal_per_bar +
    popcorn_cal

  let total_calories_burned : ℕ := 
    run_minutes * run_cal_per_minute +
    swim_minutes * swim_cal_per_minute +
    cycle_minutes * cycle_cal_per_minute

  total_calories_consumed - total_calories_burned

theorem excess_calories_is_770 : excess_calories = 770 := by
  sorry

end NUMINAMATH_CALUDE_excess_calories_is_770_l3512_351240


namespace NUMINAMATH_CALUDE_incorrect_statement_l3512_351275

theorem incorrect_statement : ¬(∀ (p q : Prop), (¬p ∧ ¬q) → (p ∧ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l3512_351275


namespace NUMINAMATH_CALUDE_fish_population_estimate_l3512_351224

/-- Estimates the number of fish in a pond based on a capture-recapture method. -/
def estimate_fish_population (tagged_fish : ℕ) (second_catch : ℕ) (recaptured : ℕ) : ℕ :=
  (tagged_fish * second_catch) / recaptured

/-- Theorem stating that given the specific conditions of the problem, 
    the estimated fish population is 600. -/
theorem fish_population_estimate :
  let tagged_fish : ℕ := 30
  let second_catch : ℕ := 40
  let recaptured : ℕ := 2
  estimate_fish_population tagged_fish second_catch recaptured = 600 := by
  sorry

#eval estimate_fish_population 30 40 2

end NUMINAMATH_CALUDE_fish_population_estimate_l3512_351224


namespace NUMINAMATH_CALUDE_num_winning_configurations_l3512_351223

/-- Represents a 4x4 tic-tac-toe board -/
def Board := Fin 4 → Fin 4 → Option Bool

/-- Represents a 3x3 section of the 4x4 board -/
def Section := Fin 3 → Fin 3 → Option Bool

/-- The number of 3x3 sections in a 4x4 board -/
def numSections : Nat := 4

/-- The number of ways to place X's in a winning 3x3 section for horizontal or vertical wins -/
def numXPlacementsRowCol : Nat := 18

/-- The number of ways to place X's in a winning 3x3 section for diagonal wins -/
def numXPlacementsDiag : Nat := 20

/-- The number of rows or columns in a 3x3 section -/
def numRowsOrCols : Nat := 6

/-- The number of diagonals in a 3x3 section -/
def numDiagonals : Nat := 2

/-- Calculates the total number of winning configurations in one 3x3 section -/
def winsIn3x3Section : Nat :=
  numRowsOrCols * numXPlacementsRowCol + numDiagonals * numXPlacementsDiag

/-- The main theorem: proves that the number of possible board configurations after Carl wins is 592 -/
theorem num_winning_configurations :
  (numSections * winsIn3x3Section) = 592 := by sorry

end NUMINAMATH_CALUDE_num_winning_configurations_l3512_351223


namespace NUMINAMATH_CALUDE_remainder_of_b_mod_13_l3512_351206

/-- Given that b ≡ (2^(-1) + 3^(-1) + 5^(-1))^(-1) (mod 13), prove that b ≡ 6 (mod 13) -/
theorem remainder_of_b_mod_13 :
  (((2 : ZMod 13)⁻¹ + (3 : ZMod 13)⁻¹ + (5 : ZMod 13)⁻¹)⁻¹ : ZMod 13) = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_b_mod_13_l3512_351206


namespace NUMINAMATH_CALUDE_nancys_savings_in_euros_l3512_351287

/-- Calculates the amount of money Nancy has in euros given her savings and the exchange rate. -/
def nancys_euros_savings (quarters : ℕ) (five_dollar_bills : ℕ) (dimes : ℕ) (exchange_rate : ℚ) : ℚ :=
  let dollars : ℚ := (quarters * (1 / 4) + five_dollar_bills * 5 + dimes * (1 / 10))
  dollars / exchange_rate

/-- Proves that Nancy has €18.21 in euros given her savings and the exchange rate. -/
theorem nancys_savings_in_euros :
  nancys_euros_savings 12 3 24 (112 / 100) = 1821 / 100 := by
  sorry

end NUMINAMATH_CALUDE_nancys_savings_in_euros_l3512_351287


namespace NUMINAMATH_CALUDE_optimal_price_is_160_l3512_351260

/-- Represents the price and occupancy data for a hotel room --/
structure PriceOccupancy where
  price : ℝ
  occupancy : ℝ

/-- Calculates the daily income for a given price and occupancy --/
def dailyIncome (po : PriceOccupancy) (totalRooms : ℝ) : ℝ :=
  po.price * po.occupancy * totalRooms

/-- Theorem: The optimal price for maximizing daily income is 160 yuan --/
theorem optimal_price_is_160 (totalRooms : ℝ) 
  (priceOccupancyData : List PriceOccupancy) 
  (h1 : totalRooms = 100)
  (h2 : priceOccupancyData = [
    ⟨200, 0.65⟩, 
    ⟨180, 0.75⟩, 
    ⟨160, 0.85⟩, 
    ⟨140, 0.95⟩
  ]) : 
  ∃ (optimalPO : PriceOccupancy), 
    optimalPO ∈ priceOccupancyData ∧ 
    optimalPO.price = 160 ∧
    ∀ (po : PriceOccupancy), 
      po ∈ priceOccupancyData → 
      dailyIncome optimalPO totalRooms ≥ dailyIncome po totalRooms :=
by sorry

end NUMINAMATH_CALUDE_optimal_price_is_160_l3512_351260


namespace NUMINAMATH_CALUDE_horner_method_v3_l3512_351254

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

def horner_v3 (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * x + 0
  let v2 := v1 * x + 2
  v2 * x + 3

theorem horner_method_v3 :
  horner_v3 f 3 = 36 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3512_351254


namespace NUMINAMATH_CALUDE_no_subset_with_unique_finite_sum_representation_l3512_351226

-- Define the set S as rational numbers in (0,1)
def S : Set ℚ := {q : ℚ | 0 < q ∧ q < 1}

-- Define the property for subset T
def has_unique_finite_sum_representation (T : Set ℚ) : Prop :=
  ∀ s ∈ S, ∃! (finite_sum : List ℚ),
    (∀ t ∈ finite_sum, t ∈ T) ∧
    (∀ t ∈ finite_sum, ∀ u ∈ finite_sum, t ≠ u → t ≠ u) ∧
    (s = finite_sum.sum)

-- Theorem statement
theorem no_subset_with_unique_finite_sum_representation :
  ¬ ∃ (T : Set ℚ), T ⊆ S ∧ has_unique_finite_sum_representation T := by
  sorry

end NUMINAMATH_CALUDE_no_subset_with_unique_finite_sum_representation_l3512_351226


namespace NUMINAMATH_CALUDE_max_third_side_of_triangle_l3512_351292

theorem max_third_side_of_triangle (a b c : ℝ) : 
  a = 7 → b = 10 → c > 0 → a + b + c ≤ 30 → 
  a + b > c → a + c > b → b + c > a → 
  ∀ n : ℕ, (n : ℝ) > c → n ≤ 13 :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_of_triangle_l3512_351292


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3512_351229

/-- 
A quadratic equation x^2 + 3x - k = 0 has two equal real roots if and only if k = -9/4.
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x - k = 0 ∧ (∀ y : ℝ, y^2 + 3*y - k = 0 → y = x)) ↔ k = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3512_351229


namespace NUMINAMATH_CALUDE_probability_of_event_A_l3512_351259

theorem probability_of_event_A (P_B P_AB P_AUB : ℝ) 
  (hB : P_B = 0.4)
  (hAB : P_AB = 0.25)
  (hAUB : P_AUB = 0.6)
  : ∃ P_A : ℝ, P_A = 0.45 ∧ P_AUB = P_A + P_B - P_AB :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_event_A_l3512_351259


namespace NUMINAMATH_CALUDE_PQ_length_l3512_351269

-- Define the point R
def R : ℝ × ℝ := (10, 8)

-- Define the lines
def line1 (x y : ℝ) : Prop := 7 * y = 24 * x
def line2 (x y : ℝ) : Prop := 13 * y = 5 * x

-- Define P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- State that P is on line1
axiom P_on_line1 : line1 P.1 P.2

-- State that Q is on line2
axiom Q_on_line2 : line2 Q.1 Q.2

-- R is the midpoint of PQ
axiom R_midpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Theorem to prove
theorem PQ_length : 
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 4648 / 277 := by sorry

end NUMINAMATH_CALUDE_PQ_length_l3512_351269


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3512_351220

theorem units_digit_of_product (n : ℕ) : n % 10 = (2^101 * 7^1002 * 3^1004) % 10 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3512_351220


namespace NUMINAMATH_CALUDE_hyperbola_point_relationship_l3512_351212

/-- Given points on a hyperbola, prove their y-coordinate relationship -/
theorem hyperbola_point_relationship (k : ℝ) (y₁ y₂ y₃ : ℝ)
  (h_pos : k > 0)
  (h_y₁ : y₁ = k / (-5))
  (h_y₂ : y₂ = k / (-1))
  (h_y₃ : y₃ = k / 2) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_point_relationship_l3512_351212


namespace NUMINAMATH_CALUDE_proportion_with_reciprocals_l3512_351280

theorem proportion_with_reciprocals (a b c d : ℝ) : 
  a / b = c / d →  -- proportion
  b * c = 1 →      -- inner terms are reciprocals
  a = 0.2 →        -- one outer term is 0.2
  d = 5 :=         -- prove the other outer term is 5
by sorry

end NUMINAMATH_CALUDE_proportion_with_reciprocals_l3512_351280


namespace NUMINAMATH_CALUDE_sine_graph_shift_l3512_351205

theorem sine_graph_shift (x : ℝ) :
  2 * Real.sin (2 * (x + π/8) - π/4) = 2 * Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l3512_351205
