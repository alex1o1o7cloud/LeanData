import Mathlib

namespace NUMINAMATH_CALUDE_congruence_problem_l2012_201251

theorem congruence_problem (x : ℤ) : 
  (5 * x + 3) % 16 = 4 → (4 * x + 5) % 16 = 9 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2012_201251


namespace NUMINAMATH_CALUDE_intersection_locus_is_ellipse_l2012_201243

theorem intersection_locus_is_ellipse :
  ∀ (x y u : ℝ),
  (2 * u * x - 3 * y - 4 * u = 0) →
  (x - 3 * u * y + 4 = 0) →
  (x^2 / 16 + y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_is_ellipse_l2012_201243


namespace NUMINAMATH_CALUDE_parabola_properties_l2012_201295

/-- Represents a parabola of the form y = ax^2 - 2ax + 3 -/
structure Parabola where
  a : ℝ
  h : a ≠ 0

/-- The axis of symmetry of the parabola -/
def axisOfSymmetry (p : Parabola) : ℝ := 1

/-- The shifted parabola's vertex is on the x-axis -/
def vertexOnXAxis (p : Parabola) : Prop :=
  p.a = 3/4 ∨ p.a = -3/2

theorem parabola_properties (p : Parabola) :
  (axisOfSymmetry p = 1) ∧
  (vertexOnXAxis p ↔ (p.a = 3/4 ∨ p.a = -3/2)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2012_201295


namespace NUMINAMATH_CALUDE_sophia_reading_progress_l2012_201271

theorem sophia_reading_progress (total_pages : ℕ) (pages_finished : ℚ) : 
  total_pages = 270 → pages_finished = 2/3 → 
  (pages_finished * total_pages : ℚ) - ((1 - pages_finished) * total_pages : ℚ) = 90 := by
  sorry


end NUMINAMATH_CALUDE_sophia_reading_progress_l2012_201271


namespace NUMINAMATH_CALUDE_runners_meet_at_6000_seconds_l2012_201299

/-- The time at which three runners meet again on a circular track -/
def runners_meeting_time (track_length : ℝ) (speed1 speed2 speed3 : ℝ) : ℝ :=
  let t := 6000
  t

/-- Theorem stating that the runners meet after 6000 seconds -/
theorem runners_meet_at_6000_seconds (track_length : ℝ) (speed1 speed2 speed3 : ℝ)
  (h_track : track_length = 600)
  (h_speed1 : speed1 = 4.4)
  (h_speed2 : speed2 = 4.9)
  (h_speed3 : speed3 = 5.1) :
  runners_meeting_time track_length speed1 speed2 speed3 = 6000 := by
  sorry

#check runners_meet_at_6000_seconds

end NUMINAMATH_CALUDE_runners_meet_at_6000_seconds_l2012_201299


namespace NUMINAMATH_CALUDE_car_travel_time_l2012_201277

/-- Given a truck and car with specific conditions, prove the car's travel time --/
theorem car_travel_time (truck_distance : ℝ) (truck_time : ℝ) (speed_difference : ℝ) (distance_difference : ℝ) :
  truck_distance = 296 →
  truck_time = 8 →
  speed_difference = 18 →
  distance_difference = 6.5 →
  let truck_speed := truck_distance / truck_time
  let car_speed := truck_speed + speed_difference
  let car_distance := truck_distance + distance_difference
  car_distance / car_speed = 5.5 := by sorry

end NUMINAMATH_CALUDE_car_travel_time_l2012_201277


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2012_201293

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point :
  ∀ (given_line : Line),
  given_line.a = 1 ∧ given_line.b = -2 ∧ given_line.c = -2 →
  ∃ (parallel_line : Line),
    parallel parallel_line given_line ∧
    point_on_line 1 0 parallel_line ∧
    parallel_line.a = 1 ∧ parallel_line.b = -2 ∧ parallel_line.c = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2012_201293


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l2012_201279

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, 
    (10000 ≤ m ∧ m < 100000) →
    (∃ x : ℕ, m = x^2) →
    (∃ y : ℕ, m = y^3) →
    n ≤ m) ∧                  -- least such number
  n = 15625 := by
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l2012_201279


namespace NUMINAMATH_CALUDE_rectangle_y_value_l2012_201228

theorem rectangle_y_value (y : ℝ) (h1 : y > 0) : 
  let vertices := [(0, y), (10, y), (0, 4), (10, 4)]
  let area := 90
  let length := 10
  let height := y - 4
  (length * height = area) → y = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l2012_201228


namespace NUMINAMATH_CALUDE_pascal_contest_average_age_l2012_201247

/-- Represents an age in years and months -/
structure Age where
  years : ℕ
  months : ℕ
  h : months < 12

/-- Converts an Age to total months -/
def ageToMonths (a : Age) : ℕ := a.years * 12 + a.months

/-- Converts total months to an Age -/
def monthsToAge (m : ℕ) : Age :=
  { years := m / 12
  , months := m % 12
  , h := by sorry }

/-- The average age of three contestants in the Pascal Contest -/
theorem pascal_contest_average_age (a1 a2 a3 : Age)
  (h1 : a1 = { years := 14, months := 9, h := by sorry })
  (h2 : a2 = { years := 15, months := 1, h := by sorry })
  (h3 : a3 = { years := 14, months := 8, h := by sorry }) :
  monthsToAge ((ageToMonths a1 + ageToMonths a2 + ageToMonths a3) / 3) =
  { years := 14, months := 10, h := by sorry } :=
by sorry

end NUMINAMATH_CALUDE_pascal_contest_average_age_l2012_201247


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2012_201222

/-- The infinite series ∑(n=1 to ∞) (n³ + 2n² - n) / (n+3)! converges to 1/6 -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (n^3 + 2*n^2 - n : ℚ) / (Nat.factorial (n+3)) = 1/6 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2012_201222


namespace NUMINAMATH_CALUDE_number_equation_solution_l2012_201215

theorem number_equation_solution :
  ∃ x : ℝ, 0.5 * x = 0.1667 * x + 10 ∧ x = 30 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2012_201215


namespace NUMINAMATH_CALUDE_weight_sequence_l2012_201281

theorem weight_sequence (a : ℕ → ℝ) : 
  (∀ n, a n < a (n + 1)) →  -- weights are in increasing order
  (∀ k, k ≤ 29 → a k + a (k + 3) = a (k + 1) + a (k + 2)) →  -- balancing condition
  a 3 = 9 →  -- third weight is 9 grams
  a 9 = 33 →  -- ninth weight is 33 grams
  a 33 = 257 :=  -- 33rd weight is 257 grams
by
  sorry


end NUMINAMATH_CALUDE_weight_sequence_l2012_201281


namespace NUMINAMATH_CALUDE_set_operations_l2012_201242

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 2)}
def N : Set ℝ := {x | x < 1 ∨ x > 3}

-- State the theorem
theorem set_operations :
  (M ∪ N = {x | x < 1 ∨ x ≥ 2}) ∧
  (M ∩ (Nᶜ) = {x | 2 ≤ x ∧ x ≤ 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2012_201242


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2012_201234

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- Sum of squares condition
  c = 25 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2012_201234


namespace NUMINAMATH_CALUDE_max_d_is_four_l2012_201285

/-- A function that constructs a 6-digit number of the form 6d6,33e -/
def construct_number (d e : ℕ) : ℕ := 
  600000 + d * 10000 + 6 * 1000 + 300 + 30 + e

/-- Proposition: The maximum value of d is 4 -/
theorem max_d_is_four :
  ∃ (d e : ℕ),
    d ≤ 9 ∧ e ≤ 9 ∧
    (construct_number d e) % 33 = 0 ∧
    d + e = 4 ∧
    ∀ (d' e' : ℕ), d' ≤ 9 ∧ e' ≤ 9 ∧ 
      (construct_number d' e') % 33 = 0 ∧ 
      d' + e' = 4 → 
      d' ≤ d :=
by
  sorry

end NUMINAMATH_CALUDE_max_d_is_four_l2012_201285


namespace NUMINAMATH_CALUDE_distance_comparisons_l2012_201278

-- Define the driving conditions
def joseph_speed1 : ℝ := 48
def joseph_time1 : ℝ := 2.5
def joseph_speed2 : ℝ := 60
def joseph_time2 : ℝ := 1.5

def kyle_speed1 : ℝ := 70
def kyle_time1 : ℝ := 2
def kyle_speed2 : ℝ := 63
def kyle_time2 : ℝ := 2.5

def emily_speed : ℝ := 65
def emily_time : ℝ := 3

-- Define the distances driven
def joseph_distance : ℝ := joseph_speed1 * joseph_time1 + joseph_speed2 * joseph_time2
def kyle_distance : ℝ := kyle_speed1 * kyle_time1 + kyle_speed2 * kyle_time2
def emily_distance : ℝ := emily_speed * emily_time

-- Theorem to prove the distance comparisons
theorem distance_comparisons :
  (joseph_distance = 210) ∧
  (kyle_distance = 297.5) ∧
  (emily_distance = 195) ∧
  (joseph_distance - kyle_distance = -87.5) ∧
  (emily_distance - joseph_distance = -15) ∧
  (emily_distance - kyle_distance = -102.5) :=
by sorry

end NUMINAMATH_CALUDE_distance_comparisons_l2012_201278


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2012_201291

theorem parallelogram_base_length
  (area : ℝ) (base : ℝ) (altitude : ℝ)
  (h1 : area = 288)
  (h2 : altitude = 2 * base)
  (h3 : area = base * altitude) :
  base = 12 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2012_201291


namespace NUMINAMATH_CALUDE_grid_coloring_inequality_l2012_201253

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the grid -/
def Grid (n : ℕ) := Fin n → Fin n → Cell

/-- Counts the number of black cells adjacent to a vertex -/
def countAdjacentBlack (g : Grid n) (i j : Fin n) : ℕ := sorry

/-- Determines if a vertex is red -/
def isRed (g : Grid n) (i j : Fin n) : Bool :=
  Odd (countAdjacentBlack g i j)

/-- Counts the number of red vertices in the grid -/
def countRedVertices (g : Grid n) : ℕ := sorry

/-- Represents an operation to change colors in a rectangle -/
structure Operation (n : ℕ) where
  topLeft : Fin n × Fin n
  bottomRight : Fin n × Fin n

/-- Applies an operation to the grid -/
def applyOperation (g : Grid n) (op : Operation n) : Grid n := sorry

/-- Checks if the grid is entirely white -/
def isAllWhite (g : Grid n) : Bool := sorry

/-- The minimum number of operations to make the grid white -/
noncomputable def minOperations (g : Grid n) : ℕ := sorry

theorem grid_coloring_inequality (n : ℕ) (g : Grid n) :
  let Y := countRedVertices g
  let X := minOperations g
  Y / 4 ≤ X ∧ X ≤ Y / 2 := by sorry

end NUMINAMATH_CALUDE_grid_coloring_inequality_l2012_201253


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l2012_201213

/-- Represents an ellipse with given properties -/
structure Ellipse where
  e : ℝ  -- eccentricity
  ab_length : ℝ  -- length of AB

/-- Represents a line that intersects the ellipse -/
structure IntersectingLine where
  k : ℝ  -- slope of the line y = kx + 2

/-- Main theorem about the ellipse and intersecting line -/
theorem ellipse_and_line_properties
  (ell : Ellipse)
  (line : IntersectingLine)
  (h_e : ell.e = Real.sqrt 6 / 3)
  (h_ab : ell.ab_length = 2 * Real.sqrt 3 / 3) :
  (∃ (a b : ℝ), a^2 / 3 + b^2 = 1) ∧  -- Ellipse equation
  (∃ (x y : ℝ), x^2 / 3 + y^2 = 1 ∧ y = line.k * x + 2) ∧  -- Line intersects ellipse
  (∃ (c d : ℝ × ℝ),
    (c.1 - d.1)^2 + (c.2 - d.2)^2 = (-1 - c.1)^2 + (0 - c.2)^2 ∧  -- Circle condition
    (c.1 - d.1)^2 + (c.2 - d.2)^2 = (-1 - d.1)^2 + (0 - d.2)^2 ∧
    c.2 = line.k * c.1 + 2 ∧
    d.2 = line.k * d.1 + 2) →
  line.k = 7 / 6 := by
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l2012_201213


namespace NUMINAMATH_CALUDE_delta_problem_l2012_201236

-- Define the Δ operation
def delta (a b : ℕ) : ℕ := a^2 + b

-- State the theorem
theorem delta_problem : delta (3^(delta 2 6)) (4^(delta 4 2)) = 72201960037 := by
  sorry

end NUMINAMATH_CALUDE_delta_problem_l2012_201236


namespace NUMINAMATH_CALUDE_washer_dryer_price_ratio_l2012_201269

theorem washer_dryer_price_ratio :
  ∀ (washer_price dryer_price : ℕ),
    washer_price + dryer_price = 600 →
    ∃ k : ℕ, washer_price = k * dryer_price →
    dryer_price = 150 →
    washer_price / dryer_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_washer_dryer_price_ratio_l2012_201269


namespace NUMINAMATH_CALUDE_expand_product_l2012_201265

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2012_201265


namespace NUMINAMATH_CALUDE_monika_beans_purchase_l2012_201238

def mall_cost : ℚ := 250
def movie_cost : ℚ := 24
def num_movies : ℕ := 3
def bean_cost : ℚ := 1.25
def total_spent : ℚ := 347

theorem monika_beans_purchase :
  (total_spent - (mall_cost + movie_cost * num_movies)) / bean_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_monika_beans_purchase_l2012_201238


namespace NUMINAMATH_CALUDE_allison_extra_glue_sticks_l2012_201209

/-- Represents the number of items bought by a person -/
structure Items where
  glue_sticks : ℕ
  construction_paper : ℕ

/-- The problem setup -/
def craft_store_problem (allison marie : Items) : Prop :=
  allison.glue_sticks > marie.glue_sticks ∧
  marie.construction_paper = 6 * allison.construction_paper ∧
  marie.glue_sticks = 15 ∧
  marie.construction_paper = 30 ∧
  allison.glue_sticks + allison.construction_paper = 28

/-- The theorem to prove -/
theorem allison_extra_glue_sticks (allison marie : Items) 
  (h : craft_store_problem allison marie) : 
  allison.glue_sticks - marie.glue_sticks = 8 := by
  sorry


end NUMINAMATH_CALUDE_allison_extra_glue_sticks_l2012_201209


namespace NUMINAMATH_CALUDE_shopping_mall_escalator_problem_l2012_201231

/-- Represents the escalator and staircase system in the shopping mall -/
structure EscalatorSystem where
  escalator_speed : ℝ
  a_step_rate : ℝ
  b_step_rate : ℝ
  a_steps_up : ℕ
  b_steps_up : ℕ

/-- Represents the result of the problem -/
structure ProblemResult where
  exposed_steps : ℕ
  catchup_location : Bool  -- true if on staircase, false if on escalator
  steps_walked : ℕ

/-- The main theorem that proves the result of the problem -/
theorem shopping_mall_escalator_problem (sys : EscalatorSystem) 
  (h1 : sys.a_step_rate = 2 * sys.b_step_rate)
  (h2 : sys.a_steps_up = 24)
  (h3 : sys.b_steps_up = 16) :
  ∃ (result : ProblemResult), 
    result.exposed_steps = 48 ∧ 
    result.catchup_location = true ∧ 
    result.steps_walked = 176 :=
by
  sorry

end NUMINAMATH_CALUDE_shopping_mall_escalator_problem_l2012_201231


namespace NUMINAMATH_CALUDE_mutual_fund_yield_range_theorem_l2012_201294

/-- Represents the range of annual yields for mutual funds -/
structure YieldRange where
  last_year : ℝ
  improvement_rate : ℝ

/-- Calculates the new range of annual yields after improvement -/
def new_range (yr : YieldRange) : ℝ :=
  yr.last_year * (1 + yr.improvement_rate)

theorem mutual_fund_yield_range_theorem (yr : YieldRange) 
  (h1 : yr.last_year = 10000)
  (h2 : yr.improvement_rate = 0.15) : 
  new_range yr = 11500 := by
  sorry

#check mutual_fund_yield_range_theorem

end NUMINAMATH_CALUDE_mutual_fund_yield_range_theorem_l2012_201294


namespace NUMINAMATH_CALUDE_expenditure_representation_l2012_201270

def represent_income (amount : ℝ) : ℝ := amount

theorem expenditure_representation (amount : ℝ) :
  (represent_income amount = amount) →
  (∃ (f : ℝ → ℝ), f amount = -amount) :=
by sorry

end NUMINAMATH_CALUDE_expenditure_representation_l2012_201270


namespace NUMINAMATH_CALUDE_travel_time_calculation_l2012_201297

/-- Given a constant rate of travel where 1 mile takes 4 minutes,
    prove that the time required to travel 5 miles is 20 minutes. -/
theorem travel_time_calculation (rate : ℝ) (distance : ℝ) :
  rate = 1 / 4 → distance = 5 → rate * distance = 20 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l2012_201297


namespace NUMINAMATH_CALUDE_binomial_coefficient_10_3_l2012_201296

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_10_3_l2012_201296


namespace NUMINAMATH_CALUDE_square_one_fifth_equals_point_zero_four_l2012_201256

theorem square_one_fifth_equals_point_zero_four (ε : ℝ) :
  ∃ ε > 0, (1 / 5 : ℝ)^2 = 0.04 + ε ∧ ε < 0.00000000000000001 := by
  sorry

end NUMINAMATH_CALUDE_square_one_fifth_equals_point_zero_four_l2012_201256


namespace NUMINAMATH_CALUDE_power_product_equality_l2012_201212

theorem power_product_equality : (-4 : ℝ)^2010 * (-0.25 : ℝ)^2011 = -0.25 := by sorry

end NUMINAMATH_CALUDE_power_product_equality_l2012_201212


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_positive_n_for_unique_solution_l2012_201254

theorem unique_solution_quadratic (n : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + n * x + 16 = 0) ↔ n = 24 ∨ n = -24 :=
by sorry

theorem positive_n_for_unique_solution :
  ∃ n : ℝ, n > 0 ∧ (∃! x : ℝ, 9 * x^2 + n * x + 16 = 0) ∧ n = 24 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_positive_n_for_unique_solution_l2012_201254


namespace NUMINAMATH_CALUDE_evaluate_expression_l2012_201235

theorem evaluate_expression : 3000 * (3000^1500) = 3000^1501 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2012_201235


namespace NUMINAMATH_CALUDE_binary_1010_is_10_l2012_201214

/-- Converts a binary number represented as a list of bits (0s and 1s) to its decimal equivalent -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

theorem binary_1010_is_10 : binary_to_decimal [0, 1, 0, 1] = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_is_10_l2012_201214


namespace NUMINAMATH_CALUDE_machine_comparison_l2012_201276

def machine_A : List ℕ := [0, 2, 1, 0, 3, 0, 2, 1, 2, 4]
def machine_B : List ℕ := [2, 1, 1, 2, 1, 0, 2, 1, 3, 2]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let avg := average l
  (l.map (λ x => ((x : ℚ) - avg) ^ 2)).sum / l.length

theorem machine_comparison :
  average machine_A = average machine_B ∧
  variance machine_B < variance machine_A :=
sorry

end NUMINAMATH_CALUDE_machine_comparison_l2012_201276


namespace NUMINAMATH_CALUDE_plane_equivalence_l2012_201230

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parametric equation of a plane -/
def parametric_plane (s t : ℝ) : Point3D :=
  { x := 2 + 2*s - t
    y := 1 - 2*s
    z := 4 - s + 3*t }

/-- Represents the Cartesian equation of a plane -/
def cartesian_plane (p : Point3D) : Prop :=
  6 * p.x + 5 * p.y + 2 * p.z - 25 = 0

/-- Theorem stating the equivalence of the parametric and Cartesian representations -/
theorem plane_equivalence :
  ∀ (p : Point3D), (∃ (s t : ℝ), p = parametric_plane s t) ↔ cartesian_plane p :=
sorry

end NUMINAMATH_CALUDE_plane_equivalence_l2012_201230


namespace NUMINAMATH_CALUDE_speed_ratio_l2012_201288

def equidistant_points (vA vB : ℝ) : Prop :=
  ∃ (t : ℝ), t * vA = |(-800 + t * vB)|

theorem speed_ratio : ∃ (vA vB : ℝ),
  vA > 0 ∧ vB > 0 ∧
  equidistant_points vA vB ∧
  equidistant_points (3 * vA) (3 * vB) ∧
  vA / vB = 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_speed_ratio_l2012_201288


namespace NUMINAMATH_CALUDE_distance_between_points_l2012_201289

/-- The distance between two points (2, -7) and (-8, 4) is √221. -/
theorem distance_between_points : Real.sqrt 221 = Real.sqrt ((2 - (-8))^2 + ((-7) - 4)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2012_201289


namespace NUMINAMATH_CALUDE_alice_ice_cream_count_l2012_201286

/-- The number of pints of ice cream Alice had on Wednesday -/
def ice_cream_on_wednesday (sunday_pints : ℕ) : ℕ :=
  let monday_pints := 3 * sunday_pints
  let tuesday_pints := monday_pints / 3
  let total_before_wednesday := sunday_pints + monday_pints + tuesday_pints
  let returned_pints := tuesday_pints / 2
  total_before_wednesday - returned_pints

/-- Theorem stating that Alice had 18 pints of ice cream on Wednesday -/
theorem alice_ice_cream_count : ice_cream_on_wednesday 4 = 18 := by
  sorry

#eval ice_cream_on_wednesday 4

end NUMINAMATH_CALUDE_alice_ice_cream_count_l2012_201286


namespace NUMINAMATH_CALUDE_expression_value_l2012_201237

theorem expression_value : 3 * 4^2 - (8 / 2) = 44 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2012_201237


namespace NUMINAMATH_CALUDE_vanaspati_percentage_after_addition_l2012_201229

/-- Calculates the percentage of vanaspati in a ghee mixture after adding pure ghee -/
theorem vanaspati_percentage_after_addition
  (original_quantity : ℝ)
  (original_pure_ghee_percentage : ℝ)
  (original_vanaspati_percentage : ℝ)
  (added_pure_ghee : ℝ)
  (h1 : original_quantity = 30)
  (h2 : original_pure_ghee_percentage = 50)
  (h3 : original_vanaspati_percentage = 50)
  (h4 : added_pure_ghee = 20)
  (h5 : original_pure_ghee_percentage + original_vanaspati_percentage = 100) :
  let original_vanaspati := original_quantity * (original_vanaspati_percentage / 100)
  let new_total_quantity := original_quantity + added_pure_ghee
  (original_vanaspati / new_total_quantity) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_vanaspati_percentage_after_addition_l2012_201229


namespace NUMINAMATH_CALUDE_configurations_count_l2012_201248

/-- The number of squares in the set -/
def total_squares : ℕ := 8

/-- The number of squares to be placed -/
def squares_to_place : ℕ := 2

/-- The number of distinct sides on which squares can be placed -/
def distinct_sides : ℕ := 2

/-- The number of configurations that can be formed -/
def num_configurations : ℕ := total_squares * (total_squares - 1)

theorem configurations_count :
  num_configurations = 56 :=
sorry

end NUMINAMATH_CALUDE_configurations_count_l2012_201248


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2012_201272

theorem problem_1 : (-1)^2 + (Real.pi - 2022)^0 + 2 * Real.sin (Real.pi / 3) - |1 - Real.sqrt 3| = 3 := by
  sorry

theorem problem_2 : ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → (2 / (x + 1) + 1 = x / (x - 1) ↔ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2012_201272


namespace NUMINAMATH_CALUDE_leahs_coins_value_l2012_201232

/-- Represents the value of a coin in cents -/
inductive Coin
| Penny : Coin
| Nickel : Coin

/-- The value of a coin in cents -/
def coin_value : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5

/-- A collection of coins -/
structure CoinCollection :=
  (pennies : Nat)
  (nickels : Nat)

/-- The total number of coins in a collection -/
def total_coins (c : CoinCollection) : Nat :=
  c.pennies + c.nickels

/-- The total value of coins in a collection in cents -/
def total_value (c : CoinCollection) : Nat :=
  c.pennies * coin_value Coin.Penny + c.nickels * coin_value Coin.Nickel

/-- The main theorem -/
theorem leahs_coins_value (c : CoinCollection) :
  total_coins c = 15 ∧
  c.pennies = c.nickels + 2 →
  total_value c = 44 := by
  sorry


end NUMINAMATH_CALUDE_leahs_coins_value_l2012_201232


namespace NUMINAMATH_CALUDE_mean_books_read_l2012_201283

def readers_3 : ℕ := 4
def books_3 : ℕ := 3
def readers_5 : ℕ := 5
def books_5 : ℕ := 5
def readers_7 : ℕ := 2
def books_7 : ℕ := 7
def readers_10 : ℕ := 1
def books_10 : ℕ := 10

def total_readers : ℕ := readers_3 + readers_5 + readers_7 + readers_10
def total_books : ℕ := readers_3 * books_3 + readers_5 * books_5 + readers_7 * books_7 + readers_10 * books_10

theorem mean_books_read :
  (total_books : ℚ) / (total_readers : ℚ) = 61 / 12 :=
sorry

end NUMINAMATH_CALUDE_mean_books_read_l2012_201283


namespace NUMINAMATH_CALUDE_coin_division_l2012_201223

theorem coin_division (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5)
  (h3 : ∀ m : ℕ, m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) :
  n % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_coin_division_l2012_201223


namespace NUMINAMATH_CALUDE_max_profit_rate_l2012_201266

def f (x : ℕ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 20 then 1
  else if 21 ≤ x ∧ x ≤ 60 then x / 10
  else 0

def g (x : ℕ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 20 then 1 / (x + 80)
  else if 21 ≤ x ∧ x ≤ 60 then (2 * x) / (x^2 - x + 1600)
  else 0

theorem max_profit_rate :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 60 → g x ≤ 2/79 ∧ g 40 = 2/79 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_rate_l2012_201266


namespace NUMINAMATH_CALUDE_rs_fraction_l2012_201263

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the altitude CH
def altitude (t : Triangle) : ℝ × ℝ := sorry

-- Define the points R and S
def R (t : Triangle) : ℝ × ℝ := sorry
def S (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem rs_fraction (t : Triangle) :
  distance (t.A) (t.B) = 2023 →
  distance (t.A) (t.C) = 2022 →
  distance (t.B) (t.C) = 2021 →
  distance (R t) (S t) = 2021 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_rs_fraction_l2012_201263


namespace NUMINAMATH_CALUDE_max_perimeter_of_third_rectangle_l2012_201273

-- Define the rectangles
structure Rectangle where
  width : ℕ
  height : ℕ

-- Define the problem setup
def rectangle1 : Rectangle := ⟨70, 110⟩
def rectangle2 : Rectangle := ⟨40, 80⟩

-- Function to calculate perimeter
def perimeter (r : Rectangle) : ℕ :=
  2 * (r.width + r.height)

-- Function to check if three rectangles can form a larger rectangle
def canFormLargerRectangle (r1 r2 r3 : Rectangle) : Prop :=
  (r1.width + r2.width = r3.width ∧ max r1.height r2.height = r3.height) ∨
  (r1.height + r2.height = r3.height ∧ max r1.width r2.width = r3.width) ∨
  (r1.width + r2.height = r3.width ∧ r1.height + r2.width = r3.height) ∨
  (r1.height + r2.width = r3.width ∧ r1.width + r2.height = r3.height)

-- Theorem statement
theorem max_perimeter_of_third_rectangle :
  ∃ (r3 : Rectangle), canFormLargerRectangle rectangle1 rectangle2 r3 ∧
    perimeter r3 = 300 ∧
    ∀ (r : Rectangle), canFormLargerRectangle rectangle1 rectangle2 r →
      perimeter r ≤ 300 := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_of_third_rectangle_l2012_201273


namespace NUMINAMATH_CALUDE_special_sequence_property_l2012_201258

/-- A sequence of natural numbers with specific properties -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ k, a (k + 1) - a k ∈ ({0, 1} : Set ℕ)

theorem special_sequence_property (a : ℕ → ℕ) (m : ℕ) :
  SpecialSequence a →
  (∃ m, a m = m / 1000) →
  ∃ n, a n = n / 500 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_property_l2012_201258


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2012_201290

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), Int.floor a = a → a^2 * b = 0) ↔
  (∃ (a b : ℝ), Int.floor a = a ∧ a^2 * b ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2012_201290


namespace NUMINAMATH_CALUDE_range_of_m_l2012_201200

-- Define the necessary condition p
def p (x m : ℝ) : Prop := (x - m)^2 > 3*(x - m)

-- Define the condition q
def q (x : ℝ) : Prop := x^2 - 3*x - 4 ≤ 0

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (m : ℝ) : Prop :=
  (∀ x, q x → p x m) ∧ (∃ x, p x m ∧ ¬q x)

-- Theorem statement
theorem range_of_m :
  {m : ℝ | necessary_but_not_sufficient m} = {m | m < -4 ∨ m > 4} :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2012_201200


namespace NUMINAMATH_CALUDE_garden_length_l2012_201218

/-- Proves that a rectangular garden with width 5 m and area 60 m² has a length of 12 m -/
theorem garden_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 5 → area = 60 → area = length * width → length = 12 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l2012_201218


namespace NUMINAMATH_CALUDE_no_prime_with_consecutive_squares_l2012_201204

theorem no_prime_with_consecutive_squares (n : ℕ) : 
  Prime n → ¬(∃ a b : ℕ, (2 * n + 1 = a^2) ∧ (3 * n + 1 = b^2)) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_with_consecutive_squares_l2012_201204


namespace NUMINAMATH_CALUDE_integer_root_values_l2012_201284

theorem integer_root_values (a : ℤ) : 
  (∃ x : ℤ, x^3 + 3*x^2 + a*x + 7 = 0) ↔ a ∈ ({-71, -27, -11, 9} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_root_values_l2012_201284


namespace NUMINAMATH_CALUDE_custom_op_7_neg3_custom_op_not_commutative_l2012_201217

-- Define the custom operation ※
def custom_op (a b : ℤ) : ℤ := (a + 2) * 2 - b

-- Theorem 1: 7 ※ (-3) = 21
theorem custom_op_7_neg3 : custom_op 7 (-3) = 21 := by sorry

-- Theorem 2: 7 ※ (-3) ≠ (-3) ※ 7
theorem custom_op_not_commutative : custom_op 7 (-3) ≠ custom_op (-3) 7 := by sorry

end NUMINAMATH_CALUDE_custom_op_7_neg3_custom_op_not_commutative_l2012_201217


namespace NUMINAMATH_CALUDE_parabola_focal_chord_property_l2012_201257

-- Define the parabola
def parabola (p : ℝ) : Set (ℝ × ℝ) := {xy | xy.2^2 = 2*p*xy.1}

-- Define the focal chord
def is_focal_chord (p : ℝ) (P Q : ℝ × ℝ) : Prop :=
  P ∈ parabola p ∧ Q ∈ parabola p

-- Define the directrix
def directrix (p : ℝ) : Set (ℝ × ℝ) := {xy | xy.1 = -p}

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) := {xy | xy.2 = 0}

-- Define perpendicularity
def perpendicular (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

theorem parabola_focal_chord_property (p : ℝ) (P Q M N : ℝ × ℝ) :
  p > 0 →
  is_focal_chord p P Q →
  N ∈ directrix p →
  N ∈ x_axis →
  perpendicular P Q N Q →
  perpendicular P M M (0, 0) →
  M.2 = 0 →
  abs (P.1 - M.1) = abs (M.1 - Q.1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focal_chord_property_l2012_201257


namespace NUMINAMATH_CALUDE_alice_paid_fifteen_per_acorn_l2012_201255

/-- The price Alice paid for each acorn -/
def alice_price_per_acorn (alice_acorn_count : ℕ) (alice_bob_price_ratio : ℕ) (bob_total_price : ℕ) : ℚ :=
  (alice_bob_price_ratio * bob_total_price : ℚ) / alice_acorn_count

/-- Theorem stating that Alice paid $15 for each acorn -/
theorem alice_paid_fifteen_per_acorn :
  alice_price_per_acorn 3600 9 6000 = 15 := by
  sorry

end NUMINAMATH_CALUDE_alice_paid_fifteen_per_acorn_l2012_201255


namespace NUMINAMATH_CALUDE_book_arrangements_eq_1440_l2012_201261

/-- The number of ways to arrange 8 books (3 Russian, 2 French, and 3 Italian) on a shelf,
    keeping the Russian books together and the French books together. -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 8
  let russian_books : ℕ := 3
  let french_books : ℕ := 2
  let italian_books : ℕ := 3
  let russian_unit : ℕ := 1
  let french_unit : ℕ := 1
  let total_units : ℕ := russian_unit + french_unit + italian_books
  Nat.factorial total_units * Nat.factorial russian_books * Nat.factorial french_books

/-- Theorem stating that the number of book arrangements is 1440. -/
theorem book_arrangements_eq_1440 : book_arrangements = 1440 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangements_eq_1440_l2012_201261


namespace NUMINAMATH_CALUDE_select_books_result_l2012_201205

/-- The number of ways to select one book from each of two bags of science books -/
def select_books (bag1_count : ℕ) (bag2_count : ℕ) : ℕ :=
  bag1_count * bag2_count

/-- Theorem: The number of ways to select one book from each of two bags,
    where one bag contains 4 different books and the other contains 5 different books,
    is equal to 20. -/
theorem select_books_result : select_books 4 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_select_books_result_l2012_201205


namespace NUMINAMATH_CALUDE_percentage_to_pass_l2012_201202

/-- Given a test with maximum marks and a student's performance, 
    calculate the percentage needed to pass the test. -/
theorem percentage_to_pass (max_marks student_marks shortfall : ℕ) :
  max_marks = 400 →
  student_marks = 80 →
  shortfall = 40 →
  (((student_marks + shortfall) : ℚ) / max_marks) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l2012_201202


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l2012_201287

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 1)

theorem magnitude_of_vector_sum : 
  ‖(2 • a.1 + b.1, 2 • a.2 + b.2)‖ = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l2012_201287


namespace NUMINAMATH_CALUDE_mia_speed_theorem_l2012_201292

def eugene_speed : ℚ := 5
def carlos_ratio : ℚ := 3/4
def mia_ratio : ℚ := 4/3

theorem mia_speed_theorem : 
  mia_ratio * (carlos_ratio * eugene_speed) = eugene_speed := by
  sorry

end NUMINAMATH_CALUDE_mia_speed_theorem_l2012_201292


namespace NUMINAMATH_CALUDE_maurice_age_proof_l2012_201280

/-- Ron's current age -/
def ron_current_age : ℕ := 43

/-- Maurice's current age -/
def maurice_current_age : ℕ := 7

/-- Theorem stating that Maurice's current age is 7 years -/
theorem maurice_age_proof :
  (ron_current_age + 5 = 4 * (maurice_current_age + 5)) →
  maurice_current_age = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_maurice_age_proof_l2012_201280


namespace NUMINAMATH_CALUDE_new_student_weight_l2012_201219

/-- Given a group of 15 students where replacing a 150 kg student with a new student
    decreases the average weight by 8 kg, the weight of the new student is 30 kg. -/
theorem new_student_weight (total_weight : ℝ) (new_weight : ℝ) : 
  (15 : ℝ) * (total_weight / 15 - (total_weight - 150 + new_weight) / 15) = 8 →
  new_weight = 30 := by
sorry

end NUMINAMATH_CALUDE_new_student_weight_l2012_201219


namespace NUMINAMATH_CALUDE_red_balls_count_l2012_201227

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The probability of drawing a red ball -/
def red_probability : ℝ := 0.85

/-- The number of red balls in the bag -/
def red_balls : ℕ := 17

theorem red_balls_count :
  (red_balls : ℝ) / (red_balls + black_balls) = red_probability :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l2012_201227


namespace NUMINAMATH_CALUDE_flea_can_reach_all_naturals_l2012_201241

def jump_length (k : ℕ) : ℕ := 2^k + 1

theorem flea_can_reach_all_naturals :
  ∀ n : ℕ, ∃ (jumps : List (ℕ × Bool)), 
    (jumps.foldl (λ acc (len, dir) => if dir then acc + len else acc - len) 0 : ℤ) = n ∧
    ∀ k, k < jumps.length → (jumps.get ⟨k, by sorry⟩).1 = jump_length k :=
by sorry

end NUMINAMATH_CALUDE_flea_can_reach_all_naturals_l2012_201241


namespace NUMINAMATH_CALUDE_hall_area_is_450_l2012_201240

/-- Represents a rectangular hall with specific properties. -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  width_half_length : width = length / 2
  length_width_diff : length - width = 15

/-- Calculates the area of a rectangular hall. -/
def area (hall : RectangularHall) : ℝ := hall.length * hall.width

/-- Theorem stating that a rectangular hall with the given properties has an area of 450 square units. -/
theorem hall_area_is_450 (hall : RectangularHall) : area hall = 450 := by
  sorry

end NUMINAMATH_CALUDE_hall_area_is_450_l2012_201240


namespace NUMINAMATH_CALUDE_weight_replacement_l2012_201274

theorem weight_replacement (initial_count : ℕ) (average_increase : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  average_increase = 2.5 →
  new_weight = 55 →
  ∃ (replaced_weight : ℝ),
    replaced_weight = new_weight - (initial_count * average_increase) ∧
    replaced_weight = 35 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l2012_201274


namespace NUMINAMATH_CALUDE_two_numbers_product_l2012_201246

theorem two_numbers_product (n : ℕ) (h : n = 34) : ∃ x y : ℕ, 
  x ∈ Finset.range (n + 1) ∧ 
  y ∈ Finset.range (n + 1) ∧ 
  x ≠ y ∧
  (Finset.sum (Finset.range (n + 1)) id) - x - y = 22 * (y - x) ∧
  x * y = 416 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_product_l2012_201246


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l2012_201226

/-- The polynomial with coefficients p and q -/
def polynomial (p q : ℚ) (x : ℚ) : ℚ :=
  p * x^4 + q * x^3 + 20 * x^2 - 10 * x + 15

/-- The factor of the polynomial -/
def factor (x : ℚ) : ℚ :=
  5 * x^2 - 3 * x + 3

theorem polynomial_factor_implies_coefficients (p q : ℚ) :
  (∃ (a b : ℚ), ∀ x, polynomial p q x = factor x * (a * x^2 + b * x + 5)) →
  p = 0 ∧ q = 25/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l2012_201226


namespace NUMINAMATH_CALUDE_function_has_max_and_min_l2012_201259

/-- The function f(x) = x^3 - ax^2 + ax has both a maximum and a minimum value 
    if and only if a is in the range (-∞, 0) ∪ (3, +∞) -/
theorem function_has_max_and_min (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^3 - a*x^2 + a*x ≤ x₁^3 - a*x₁^2 + a*x₁) ∧
    (∀ x : ℝ, x^3 - a*x^2 + a*x ≥ x₂^3 - a*x₂^2 + a*x₂)) ↔ 
  (a < 0 ∨ a > 3) := by
  sorry

#check function_has_max_and_min

end NUMINAMATH_CALUDE_function_has_max_and_min_l2012_201259


namespace NUMINAMATH_CALUDE_smallest_n_congruence_four_satisfies_congruence_four_is_smallest_l2012_201252

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 7^n % 5 = n^4 % 5) → n ≥ 4 :=
by sorry

theorem four_satisfies_congruence : 
  7^4 % 5 = 4^4 % 5 :=
by sorry

theorem four_is_smallest : 
  ∀ m : ℕ, m > 0 ∧ m < 4 → 7^m % 5 ≠ m^4 % 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_four_satisfies_congruence_four_is_smallest_l2012_201252


namespace NUMINAMATH_CALUDE_heaviest_tv_weight_difference_l2012_201207

/-- Represents the dimensions of a TV screen -/
structure TVDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the area of a TV screen given its dimensions -/
def screenArea (d : TVDimensions) : ℕ := d.width * d.height

/-- Calculates the weight of a TV in ounces given its screen area -/
def tvWeight (area : ℕ) : ℕ := area * 4

/-- Converts weight from ounces to pounds -/
def ouncesToPounds (oz : ℕ) : ℕ := oz / 16

theorem heaviest_tv_weight_difference (bill_tv bob_tv steve_tv : TVDimensions) 
    (h1 : bill_tv = ⟨48, 100⟩)
    (h2 : bob_tv = ⟨70, 60⟩)
    (h3 : steve_tv = ⟨84, 92⟩) :
  ouncesToPounds (tvWeight (screenArea steve_tv)) - 
  (ouncesToPounds (tvWeight (screenArea bill_tv)) + ouncesToPounds (tvWeight (screenArea bob_tv))) = 318 := by
  sorry


end NUMINAMATH_CALUDE_heaviest_tv_weight_difference_l2012_201207


namespace NUMINAMATH_CALUDE_reece_climbs_l2012_201267

-- Define constants
def keaton_ladder_feet : ℕ := 30
def keaton_climbs : ℕ := 20
def ladder_difference_feet : ℕ := 4
def total_climbed_inches : ℕ := 11880

-- Define functions
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

def reece_ladder_feet : ℕ := keaton_ladder_feet - ladder_difference_feet

-- Theorem statement
theorem reece_climbs : 
  (feet_to_inches keaton_ladder_feet * keaton_climbs + 
   feet_to_inches reece_ladder_feet * 15 = total_climbed_inches) := by
sorry

end NUMINAMATH_CALUDE_reece_climbs_l2012_201267


namespace NUMINAMATH_CALUDE_curve_transformation_l2012_201250

/-- Given a curve C: (x-y)^2 + y^2 = 1 transformed by matrix A = [[2, -2], [0, 1]],
    prove that the resulting curve C' has the equation x^2/4 + y^2 = 1 -/
theorem curve_transformation (x₀ y₀ x y : ℝ) : 
  (x₀ - y₀)^2 + y₀^2 = 1 →
  x = 2*x₀ - 2*y₀ →
  y = y₀ →
  x^2/4 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_curve_transformation_l2012_201250


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_geq_two_l2012_201224

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 5

-- State the theorem
theorem decreasing_quadratic_implies_a_geq_two :
  ∀ a : ℝ, (∀ x ≤ 2, ∀ y ≤ 2, x < y → f a x > f a y) → a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_geq_two_l2012_201224


namespace NUMINAMATH_CALUDE_p_days_correct_l2012_201264

/-- The number of days it takes for q to do the work alone -/
def q_days : ℝ := 10

/-- The fraction of work left after p and q work together for 2 days -/
def work_left : ℝ := 0.7

/-- The number of days it takes for p to do the work alone -/
def p_days : ℝ := 20

/-- Theorem stating that p_days is correct given the conditions -/
theorem p_days_correct : 
  2 * (1 / p_days + 1 / q_days) = 1 - work_left := by
  sorry

end NUMINAMATH_CALUDE_p_days_correct_l2012_201264


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2012_201210

theorem imaginary_part_of_complex_number :
  let z : ℂ := -1/2 + (1/2) * Complex.I
  Complex.im z = 1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2012_201210


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2012_201262

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 2 + a 4 = 5) :
  a 5 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2012_201262


namespace NUMINAMATH_CALUDE_sin_135_degrees_l2012_201249

theorem sin_135_degrees :
  Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l2012_201249


namespace NUMINAMATH_CALUDE_first_plane_speed_calculation_l2012_201221

/-- The speed of the first plane in kilometers per hour -/
def first_plane_speed : ℝ := 110

/-- The speed of the second plane in kilometers per hour -/
def second_plane_speed : ℝ := 90

/-- The time taken for the planes to be 800 km apart in hours -/
def time : ℝ := 4.84848484848

/-- The distance between the planes after the given time in kilometers -/
def distance : ℝ := 800

theorem first_plane_speed_calculation :
  (first_plane_speed + second_plane_speed) * time = distance := by
  sorry

end NUMINAMATH_CALUDE_first_plane_speed_calculation_l2012_201221


namespace NUMINAMATH_CALUDE_AMC9_paths_l2012_201298

-- Define the grid structure
structure Grid :=
  (has_A : Bool)
  (has_M_left : Bool)
  (has_M_right : Bool)
  (C_count_left : Nat)
  (C_count_right : Nat)
  (nine_count_per_C : Nat)

-- Define the path counting function
def count_paths (g : Grid) : Nat :=
  let left_paths := if g.has_M_left then g.C_count_left * g.nine_count_per_C else 0
  let right_paths := if g.has_M_right then g.C_count_right * g.nine_count_per_C else 0
  left_paths + right_paths

-- Theorem statement
theorem AMC9_paths (g : Grid) 
  (h1 : g.has_A)
  (h2 : g.has_M_left)
  (h3 : g.has_M_right)
  (h4 : g.C_count_left = 4)
  (h5 : g.C_count_right = 2)
  (h6 : g.nine_count_per_C = 2) :
  count_paths g = 24 := by
  sorry


end NUMINAMATH_CALUDE_AMC9_paths_l2012_201298


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2012_201282

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x, a * x^2 + 2 * a * x + 1 > 0) →
  (0 ≤ a ∧ a < 2) ∧
  ¬(0 ≤ a ∧ a < 2 → ∀ x, a * x^2 + 2 * a * x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2012_201282


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2012_201211

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : a > 0) (k : b > 0) :
  (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b) →
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2012_201211


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2012_201268

/-- A complex number z is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Definition of the complex number z in terms of m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 - m - 2) (m^2 - 3*m - 2)

/-- m = -1 is a sufficient but not necessary condition for z to be purely imaginary -/
theorem sufficient_not_necessary_condition :
  (isPurelyImaginary (z (-1))) ∧
  (∃ m : ℝ, m ≠ -1 ∧ isPurelyImaginary (z m)) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2012_201268


namespace NUMINAMATH_CALUDE_no_prime_square_product_l2012_201275

theorem no_prime_square_product (p q r : Nat) (hp : Prime p) (hq : Prime q) (hr : Prime r) :
  ¬∃ n : Nat, (p^2 + p) * (q^2 + q) * (r^2 + r) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_square_product_l2012_201275


namespace NUMINAMATH_CALUDE_kamal_present_age_l2012_201201

/-- Represents the present age of Kamal -/
def kamal_age : ℕ := sorry

/-- Represents the present age of Kamal's son -/
def son_age : ℕ := sorry

/-- Kamal was 4 times as old as his son 8 years ago -/
axiom condition1 : kamal_age - 8 = 4 * (son_age - 8)

/-- After 8 years, Kamal will be twice as old as his son -/
axiom condition2 : kamal_age + 8 = 2 * (son_age + 8)

/-- Theorem stating that Kamal's present age is 40 years -/
theorem kamal_present_age : kamal_age = 40 := by sorry

end NUMINAMATH_CALUDE_kamal_present_age_l2012_201201


namespace NUMINAMATH_CALUDE_archery_competition_scores_l2012_201208

/-- Represents an archer's score distribution --/
structure ArcherScore where
  bullseye : Nat
  ring39 : Nat
  ring24 : Nat
  ring23 : Nat
  ring17 : Nat
  ring16 : Nat

/-- Calculates the total score for an archer --/
def totalScore (score : ArcherScore) : Nat :=
  40 * score.bullseye + 39 * score.ring39 + 24 * score.ring24 +
  23 * score.ring23 + 17 * score.ring17 + 16 * score.ring16

/-- Calculates the total number of arrows used --/
def totalArrows (score : ArcherScore) : Nat :=
  score.bullseye + score.ring39 + score.ring24 + score.ring23 + score.ring17 + score.ring16

theorem archery_competition_scores :
  ∃ (dora reggie finch : ArcherScore),
    totalScore dora = 120 ∧
    totalScore reggie = 110 ∧
    totalScore finch = 100 ∧
    totalArrows dora = 6 ∧
    totalArrows reggie = 6 ∧
    totalArrows finch = 6 ∧
    dora.bullseye + reggie.bullseye + finch.bullseye = 1 ∧
    dora = { bullseye := 1, ring39 := 0, ring24 := 0, ring23 := 0, ring17 := 0, ring16 := 5 } ∧
    reggie = { bullseye := 0, ring39 := 0, ring24 := 0, ring23 := 2, ring17 := 0, ring16 := 4 } ∧
    finch = { bullseye := 0, ring39 := 0, ring24 := 0, ring23 := 0, ring17 := 4, ring16 := 2 } :=
by
  sorry


end NUMINAMATH_CALUDE_archery_competition_scores_l2012_201208


namespace NUMINAMATH_CALUDE_largest_C_for_divisibility_by_4_l2012_201206

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem largest_C_for_divisibility_by_4 :
  ∃ (B : ℕ) (h_B : B < 10),
    ∀ (C : ℕ) (h_C : C < 10),
      is_divisible_by_4 (4000000 + 600000 + B * 100000 + 41800 + C) →
      C ≤ 8 ∧
      is_divisible_by_4 (4000000 + 600000 + B * 100000 + 41800 + 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_C_for_divisibility_by_4_l2012_201206


namespace NUMINAMATH_CALUDE_maxwell_current_age_l2012_201203

/-- Maxwell's current age --/
def maxwell_age : ℕ := 6

/-- Maxwell's sister's current age --/
def sister_age : ℕ := 2

/-- Years into the future when Maxwell will be twice his sister's age --/
def years_future : ℕ := 2

theorem maxwell_current_age :
  maxwell_age = 6 ∧
  sister_age = 2 ∧
  maxwell_age + years_future = 2 * (sister_age + years_future) :=
by sorry

end NUMINAMATH_CALUDE_maxwell_current_age_l2012_201203


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2012_201245

theorem complex_fraction_calculation :
  |-(7/2)| * (12/7) / (4/3) / (-3)^2 = 1/2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2012_201245


namespace NUMINAMATH_CALUDE_sine_graph_shift_l2012_201244

theorem sine_graph_shift (x : ℝ) :
  (3 * Real.sin (2 * (x + π / 8))) = (3 * Real.sin (2 * x + π / 4)) :=
by sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l2012_201244


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2012_201216

/-- Simple interest calculation -/
theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) : 
  interest = 4016.25 → 
  rate = 0.14 → 
  time = 5 → 
  principal = interest / (rate * time) → 
  principal = 5737.5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2012_201216


namespace NUMINAMATH_CALUDE_A_obtuse_sufficient_not_necessary_l2012_201260

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define an obtuse angle
def is_obtuse (angle : Real) : Prop := angle > 90

-- Define an obtuse triangle
def is_obtuse_triangle (t : Triangle) : Prop :=
  is_obtuse t.A ∨ is_obtuse t.B ∨ is_obtuse t.C

-- Theorem statement
theorem A_obtuse_sufficient_not_necessary (t : Triangle) :
  (is_obtuse t.A → is_obtuse_triangle t) ∧
  ∃ (t' : Triangle), is_obtuse_triangle t' ∧ ¬is_obtuse t'.A :=
sorry

end NUMINAMATH_CALUDE_A_obtuse_sufficient_not_necessary_l2012_201260


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2012_201225

theorem fraction_equivalence : 
  (20 / 16 : ℚ) = 10 / 8 ∧
  (1 + 6 / 24 : ℚ) = 10 / 8 ∧
  (1 + 2 / 8 : ℚ) = 10 / 8 ∧
  (1 + 40 / 160 : ℚ) = 10 / 8 ∧
  (1 + 4 / 8 : ℚ) ≠ 10 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2012_201225


namespace NUMINAMATH_CALUDE_associate_professor_items_l2012_201220

def CommitteeMeeting (associate_count : ℕ) (assistant_count : ℕ) 
  (total_pencils : ℕ) (total_charts : ℕ) : Prop :=
  associate_count + assistant_count = 9 ∧
  assistant_count = 11 ∧
  2 * assistant_count = 16 ∧
  total_pencils = 11 ∧
  total_charts = 16

theorem associate_professor_items :
  ∃! (associate_count : ℕ), CommitteeMeeting associate_count (9 - associate_count) 11 16 ∧
  associate_count = 1 ∧
  11 = 9 - associate_count ∧
  16 = 2 * (9 - associate_count) :=
sorry

end NUMINAMATH_CALUDE_associate_professor_items_l2012_201220


namespace NUMINAMATH_CALUDE_student_ticket_price_is_318_l2012_201239

/-- Calculates the price of a student ticket given the total number of tickets sold,
    total revenue, adult ticket price, number of adult tickets sold, and number of student tickets sold. -/
def student_ticket_price (total_tickets : ℕ) (total_revenue : ℚ) (adult_price : ℚ) 
                         (adult_tickets : ℕ) (student_tickets : ℕ) : ℚ :=
  (total_revenue - (adult_price * adult_tickets)) / student_tickets

/-- Proves that the student ticket price is $3.18 given the specified conditions. -/
theorem student_ticket_price_is_318 :
  student_ticket_price 846 3846 6 410 436 = 318/100 := by
  sorry

end NUMINAMATH_CALUDE_student_ticket_price_is_318_l2012_201239


namespace NUMINAMATH_CALUDE_equal_numbers_exist_l2012_201233

/-- A quadratic polynomial -/
def QuadraticPolynomial (α : Type*) [Field α] := α → α

/-- Condition for a quadratic polynomial -/
def IsQuadratic {α : Type*} [Field α] (f : QuadraticPolynomial α) :=
  ∃ a b c, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem equal_numbers_exist (f : QuadraticPolynomial ℝ) (l t v : ℝ)
    (hf : IsQuadratic f)
    (hl : f l = t + v)
    (ht : f t = l + v)
    (hv : f v = l + t) :
    l = t ∨ l = v ∨ t = v := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_exist_l2012_201233
