import Mathlib

namespace NUMINAMATH_CALUDE_largest_cube_filling_box_l3271_327161

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the greatest common divisor of three natural numbers -/
def gcd3 (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

/-- Calculates the volume of a rectangular box -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the volume of a cube -/
def cubeVolume (edge : ℕ) : ℕ :=
  edge * edge * edge

/-- The main theorem about the largest cube that can fill the box -/
theorem largest_cube_filling_box (box : BoxDimensions) 
  (h_box : box = { length := 102, width := 255, height := 170 }) :
  let maxEdge := gcd3 box.length box.width box.height
  let numCubes := boxVolume box / cubeVolume maxEdge
  maxEdge = 17 ∧ numCubes = 900 := by sorry

end NUMINAMATH_CALUDE_largest_cube_filling_box_l3271_327161


namespace NUMINAMATH_CALUDE_terminal_side_quadrant_l3271_327154

theorem terminal_side_quadrant (α : Real) :
  let P : ℝ × ℝ := (Real.sin 2, Real.cos 2)
  (∃ k : ℝ, k > 0 ∧ P = (k * Real.sin α, k * Real.cos α)) →
  Real.sin α > 0 ∧ Real.cos α < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_terminal_side_quadrant_l3271_327154


namespace NUMINAMATH_CALUDE_divisible_by_seventeen_l3271_327119

theorem divisible_by_seventeen (n : ℕ) : 
  (6^(2*n) + 2^(n+2) + 12 * 2^n) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seventeen_l3271_327119


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3271_327162

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The problem statement -/
theorem line_through_point_parallel_to_line :
  let l1 : Line := { a := 3, b := 1, c := -2 }
  let l2 : Line := { a := 3, b := 1, c := -5 }
  l2.contains 2 (-1) ∧ Line.parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3271_327162


namespace NUMINAMATH_CALUDE_troy_computer_purchase_l3271_327156

/-- The amount of money Troy needs to buy a new computer -/
def additional_money_needed (new_computer_cost saved_amount old_computer_value : ℕ) : ℕ :=
  new_computer_cost - (saved_amount + old_computer_value)

/-- Theorem stating the amount Troy needs to buy the new computer -/
theorem troy_computer_purchase (new_computer_cost saved_amount old_computer_value : ℕ) 
  (h1 : new_computer_cost = 1200)
  (h2 : saved_amount = 450)
  (h3 : old_computer_value = 150) :
  additional_money_needed new_computer_cost saved_amount old_computer_value = 600 := by
  sorry

#eval additional_money_needed 1200 450 150

end NUMINAMATH_CALUDE_troy_computer_purchase_l3271_327156


namespace NUMINAMATH_CALUDE_black_lambs_count_l3271_327182

theorem black_lambs_count (total : ℕ) (white : ℕ) (h1 : total = 6048) (h2 : white = 193) :
  total - white = 5855 := by
  sorry

end NUMINAMATH_CALUDE_black_lambs_count_l3271_327182


namespace NUMINAMATH_CALUDE_line_equation_l3271_327177

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line passing through (0, 2)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 2

-- Define a point on both the line and the parabola
def intersection_point (k x y : ℝ) : Prop :=
  parabola x y ∧ line k x y

-- Define the condition for a circle passing through three points
def circle_condition (x1 y1 x2 y2 : ℝ) : Prop :=
  x1*x2 + y1*y2 = 0

theorem line_equation :
  ∀ k : ℝ,
  (∃ x1 y1 x2 y2 : ℝ,
    x1 ≠ x2 ∧
    intersection_point k x1 y1 ∧
    intersection_point k x2 y2 ∧
    circle_condition x1 y1 x2 y2) →
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3271_327177


namespace NUMINAMATH_CALUDE_tan_expression_equality_l3271_327106

theorem tan_expression_equality (θ : Real) (h : Real.tan θ = 3) :
  let k : Real := 1/2
  (1 - k * Real.cos θ) / Real.sin θ - (2 * Real.sin θ) / (1 + Real.cos θ) =
  (20 - Real.sqrt 10) / (3 * Real.sqrt 10) - (6 * Real.sqrt 10) / (10 + Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_tan_expression_equality_l3271_327106


namespace NUMINAMATH_CALUDE_max_x_value_l3271_327180

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |2*x - a|

-- State the theorem
theorem max_x_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∃ a : ℝ, ∀ x : ℝ, f x a ≤ 1/m + 4/n) →
  (∃ x : ℝ, ∀ y : ℝ, |y| ≤ |x| → ∃ a : ℝ, f y a ≤ 1/m + 4/n) ∧
  (∀ x : ℝ, (∀ y : ℝ, |y| ≤ |x| → ∃ a : ℝ, f y a ≤ 1/m + 4/n) → |x| ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_max_x_value_l3271_327180


namespace NUMINAMATH_CALUDE_issacs_pens_l3271_327141

theorem issacs_pens (total : ℕ) (pens : ℕ) (pencils : ℕ) : 
  total = 108 →
  total = pens + pencils →
  pencils = 5 * pens + 12 →
  pens = 16 := by
  sorry

end NUMINAMATH_CALUDE_issacs_pens_l3271_327141


namespace NUMINAMATH_CALUDE_inequality_proof_l3271_327102

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b + 3/4) * (b^2 + c + 3/4) * (c^2 + a + 3/4) ≥ (2*a + 1/2) * (2*b + 1/2) * (2*c + 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3271_327102


namespace NUMINAMATH_CALUDE_water_left_after_experiment_l3271_327120

/-- 
Proves that if Jori starts with 3 gallons of water and uses 5/4 gallons, 
she will have 7/4 gallons left.
-/
theorem water_left_after_experiment (initial_water : ℚ) (used_water : ℚ) 
  (h1 : initial_water = 3)
  (h2 : used_water = 5/4) : 
  initial_water - used_water = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_water_left_after_experiment_l3271_327120


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l3271_327125

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := x^2 + 6*x

-- Define the general form of a quadratic expression
def general_form (a h k x : ℝ) : ℝ := a*(x - h)^2 + k

-- Theorem statement
theorem quadratic_completion_of_square :
  ∃ (a h k : ℝ), ∀ x, quadratic_expr x = general_form a h k x → k = -9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l3271_327125


namespace NUMINAMATH_CALUDE_andy_twice_rahims_age_l3271_327129

def rahims_current_age : ℕ := 6
def andys_age_difference : ℕ := 1

theorem andy_twice_rahims_age (x : ℕ) : 
  (rahims_current_age + andys_age_difference + x = 2 * rahims_current_age) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_andy_twice_rahims_age_l3271_327129


namespace NUMINAMATH_CALUDE_floor_sqrt_75_l3271_327117

theorem floor_sqrt_75 : ⌊Real.sqrt 75⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_75_l3271_327117


namespace NUMINAMATH_CALUDE_equilateral_triangle_circumradius_ratio_l3271_327137

/-- Given two equilateral triangles with side lengths B and b (B ≠ b) and circumradii S and s respectively,
    the ratio of their circumradii S/s is always equal to the ratio of their side lengths B/b. -/
theorem equilateral_triangle_circumradius_ratio 
  (B b S s : ℝ) 
  (hB : B > 0) 
  (hb : b > 0) 
  (hne : B ≠ b) 
  (hS : S = B * Real.sqrt 3 / 3) 
  (hs : s = b * Real.sqrt 3 / 3) : 
  S / s = B / b := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circumradius_ratio_l3271_327137


namespace NUMINAMATH_CALUDE_vector_at_t_5_l3271_327176

def line_parameterization (t : ℝ) : ℝ × ℝ := sorry

theorem vector_at_t_5 (h1 : line_parameterization 1 = (2, 7))
                      (h2 : line_parameterization 4 = (8, -5)) :
  line_parameterization 5 = (10, -9) := by sorry

end NUMINAMATH_CALUDE_vector_at_t_5_l3271_327176


namespace NUMINAMATH_CALUDE_e_is_random_error_l3271_327185

/-- Linear regression model -/
structure LinearRegressionModel where
  x : ℝ
  y : ℝ
  a : ℝ
  b : ℝ
  e : ℝ
  model_equation : y = b * x + a + e

/-- Definition of random error in linear regression -/
def is_random_error (model : LinearRegressionModel) : Prop :=
  ∃ (error_term : ℝ), 
    error_term = model.e ∧ 
    model.y = model.b * model.x + model.a + error_term

/-- Theorem: In the linear regression model, e is the random error -/
theorem e_is_random_error (model : LinearRegressionModel) : 
  is_random_error model :=
sorry

end NUMINAMATH_CALUDE_e_is_random_error_l3271_327185


namespace NUMINAMATH_CALUDE_exists_monochromatic_parallelepiped_l3271_327149

-- Define the set A as points in ℤ³
def A : Set (ℤ × ℤ × ℤ) := Set.univ

-- Define a color assignment function
def colorAssignment (p : ℕ) : (ℤ × ℤ × ℤ) → Fin p := sorry

-- Define a rectangular parallelepiped
def isRectangularParallelepiped (vertices : Finset (ℤ × ℤ × ℤ)) : Prop := sorry

-- Theorem statement
theorem exists_monochromatic_parallelepiped (p : ℕ) (hp : p > 0) :
  ∃ (vertices : Finset (ℤ × ℤ × ℤ)),
    vertices.card = 8 ∧
    isRectangularParallelepiped vertices ∧
    ∃ (c : Fin p), ∀ v ∈ vertices, colorAssignment p v = c :=
  sorry

end NUMINAMATH_CALUDE_exists_monochromatic_parallelepiped_l3271_327149


namespace NUMINAMATH_CALUDE_forum_posts_per_day_l3271_327103

/-- A forum with questions and answers -/
structure Forum where
  members : ℕ
  questionsPerHour : ℕ
  answerRatio : ℕ

/-- Calculate the total posts (questions and answers) per day -/
def totalPostsPerDay (f : Forum) : ℕ :=
  let questionsPerDay := f.members * (f.questionsPerHour * 24)
  let answersPerDay := questionsPerDay * f.answerRatio
  questionsPerDay + answersPerDay

/-- Theorem: Given the conditions, the forum has 57600 posts per day -/
theorem forum_posts_per_day :
  ∀ (f : Forum),
    f.members = 200 →
    f.questionsPerHour = 3 →
    f.answerRatio = 3 →
    totalPostsPerDay f = 57600 := by
  sorry

end NUMINAMATH_CALUDE_forum_posts_per_day_l3271_327103


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l3271_327167

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x y, -1 < x ∧ x < y ∧ y < 11 → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l3271_327167


namespace NUMINAMATH_CALUDE_M_simplification_M_specific_value_l3271_327126

/-- Given expressions for A and B -/
def A (x y : ℝ) : ℝ := x^2 - 3*x*y - y^2
def B (x y : ℝ) : ℝ := x^2 - 3*x*y - 3*y^2

/-- The expression M defined as 2A - B -/
def M (x y : ℝ) : ℝ := 2 * A x y - B x y

/-- Theorem stating that M simplifies to x^2 - 3xy + y^2 -/
theorem M_simplification (x y : ℝ) : M x y = x^2 - 3*x*y + y^2 := by
  sorry

/-- Theorem stating that M equals 11 when x = -2 and y = 1 -/
theorem M_specific_value : M (-2) 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_M_simplification_M_specific_value_l3271_327126


namespace NUMINAMATH_CALUDE_minimum_balls_to_draw_l3271_327110

theorem minimum_balls_to_draw (red green yellow blue white black : ℕ) 
  (h_red : red = 35) (h_green : green = 25) (h_yellow : yellow = 22) 
  (h_blue : blue = 15) (h_white : white = 14) (h_black : black = 12) :
  let total := red + green + yellow + blue + white + black
  let threshold := 18
  ∃ n : ℕ, n = 93 ∧ 
    (∀ m : ℕ, m < n → 
      ∃ (r g y b w k : ℕ), r + g + y + b + w + k = m ∧
        r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black ∧
        r < threshold ∧ g < threshold ∧ y < threshold ∧ 
        b < threshold ∧ w < threshold ∧ k < threshold) ∧
    (∀ m : ℕ, m ≥ n → 
      ∀ (r g y b w k : ℕ), r + g + y + b + w + k = m →
        r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black →
        r ≥ threshold ∨ g ≥ threshold ∨ y ≥ threshold ∨ 
        b ≥ threshold ∨ w ≥ threshold ∨ k ≥ threshold) :=
by sorry

end NUMINAMATH_CALUDE_minimum_balls_to_draw_l3271_327110


namespace NUMINAMATH_CALUDE_simplify_linear_expression_l3271_327170

theorem simplify_linear_expression (x : ℝ) : 5*x + 2*x + 7*x = 14*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_linear_expression_l3271_327170


namespace NUMINAMATH_CALUDE_scooter_profit_theorem_l3271_327101

def scooter_profit_problem (cost_price : ℝ) : Prop :=
  let repair_cost : ℝ := 500
  let profit : ℝ := 1100
  let selling_price : ℝ := cost_price + profit
  (0.1 * cost_price = repair_cost) ∧
  ((profit / cost_price) * 100 = 22)

theorem scooter_profit_theorem :
  ∃ (cost_price : ℝ), scooter_profit_problem cost_price := by
  sorry

end NUMINAMATH_CALUDE_scooter_profit_theorem_l3271_327101


namespace NUMINAMATH_CALUDE_ship_journey_theorem_l3271_327164

/-- A ship's journey over three days -/
structure ShipJourney where
  day1_distance : ℝ
  day2_multiplier : ℝ
  day3_additional : ℝ
  total_distance : ℝ

/-- The solution to the ship's journey problem -/
def ship_journey_solution (j : ShipJourney) : Prop :=
  j.day1_distance = 100 ∧
  j.day2_multiplier = 3 ∧
  j.total_distance = 810 ∧
  j.total_distance = j.day1_distance + (j.day2_multiplier * j.day1_distance) + 
                     (j.day2_multiplier * j.day1_distance + j.day3_additional) ∧
  j.day3_additional = 110

/-- Theorem stating the solution to the ship's journey problem -/
theorem ship_journey_theorem (j : ShipJourney) :
  ship_journey_solution j → j.day3_additional = 110 :=
by
  sorry


end NUMINAMATH_CALUDE_ship_journey_theorem_l3271_327164


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3271_327168

/-- A line in 2D space defined by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if a line intersects a circle at exactly one point -/
def intersectsAtOnePoint (l : ParametricLine) (c : Circle) : Prop := sorry

/-- The main theorem -/
theorem line_circle_intersection (m : ℝ) :
  let l : ParametricLine := { x := λ t => 3 * t, y := λ t => 4 * t + m }
  let c : Circle := { center := (1, 0), radius := 1 }
  intersectsAtOnePoint l c → m = 1/3 ∨ m = -3 := by
  sorry


end NUMINAMATH_CALUDE_line_circle_intersection_l3271_327168


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3271_327128

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x * y / (x^2 + y^2 + 2 * z^2)) +
  Real.sqrt (y * z / (y^2 + z^2 + 2 * x^2)) +
  Real.sqrt (z * x / (z^2 + x^2 + 2 * y^2)) ≤ 3 / 2 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x * y / (x^2 + y^2 + 2 * z^2)) +
  Real.sqrt (y * z / (y^2 + z^2 + 2 * x^2)) +
  Real.sqrt (z * x / (z^2 + x^2 + 2 * y^2)) = 3 / 2 ↔
  x = y ∧ y = z :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3271_327128


namespace NUMINAMATH_CALUDE_divisibility_condition_l3271_327193

theorem divisibility_condition (a : ℕ) : 
  (a^2 + a + 1) ∣ (a^7 + 3*a^6 + 3*a^5 + 3*a^4 + a^3 + a^2 + 3) ↔ a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3271_327193


namespace NUMINAMATH_CALUDE_impossible_grid_arrangement_l3271_327134

/-- A type representing a 6x7 grid of natural numbers -/
def Grid := Fin 6 → Fin 7 → ℕ

/-- Predicate to check if a grid contains all numbers from 1 to 42 exactly once -/
def contains_all_numbers (g : Grid) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 42 → ∃! (i : Fin 6) (j : Fin 7), g i j = n

/-- Predicate to check if all vertical 1x2 rectangles in a grid have even sum -/
def all_vertical_sums_even (g : Grid) : Prop :=
  ∀ (i : Fin 5) (j : Fin 7), Even (g i j + g (i.succ) j)

/-- Theorem stating the impossibility of the desired grid arrangement -/
theorem impossible_grid_arrangement : 
  ¬ ∃ (g : Grid), contains_all_numbers g ∧ all_vertical_sums_even g :=
sorry

end NUMINAMATH_CALUDE_impossible_grid_arrangement_l3271_327134


namespace NUMINAMATH_CALUDE_job_selection_probability_l3271_327179

theorem job_selection_probability 
  (jamie_prob : ℚ) 
  (tom_prob : ℚ) 
  (h1 : jamie_prob = 2/3) 
  (h2 : tom_prob = 5/7) : 
  jamie_prob * tom_prob = 10/21 := by
sorry

end NUMINAMATH_CALUDE_job_selection_probability_l3271_327179


namespace NUMINAMATH_CALUDE_pineapple_profit_l3271_327150

/-- Calculates Jonah's profit from selling pineapples --/
theorem pineapple_profit : 
  let num_pineapples : ℕ := 6
  let price_per_pineapple : ℚ := 3
  let discount_rate : ℚ := 0.2
  let discount_threshold : ℕ := 4
  let rings_per_pineapple : ℕ := 10
  let price_per_two_rings : ℚ := 5
  let price_per_four_ring_set : ℚ := 16

  let total_cost : ℚ := if num_pineapples > discount_threshold
    then num_pineapples * price_per_pineapple * (1 - discount_rate)
    else num_pineapples * price_per_pineapple

  let total_rings : ℕ := num_pineapples * rings_per_pineapple
  let revenue_from_two_rings : ℚ := price_per_two_rings
  let remaining_rings : ℕ := total_rings - 2
  let full_sets : ℕ := remaining_rings / 4
  let revenue_from_sets : ℚ := full_sets * price_per_four_ring_set

  let total_revenue : ℚ := revenue_from_two_rings + revenue_from_sets
  let profit : ℚ := total_revenue - total_cost

  profit = 219.6 := by sorry

end NUMINAMATH_CALUDE_pineapple_profit_l3271_327150


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_is_18_l3271_327143

theorem absolute_value_equation_solution_difference : ℝ → Prop :=
  fun difference =>
    ∃ x₁ x₂ : ℝ,
      (|2 * x₁ - 3| = 18) ∧
      (|2 * x₂ - 3| = 18) ∧
      (x₁ ≠ x₂) ∧
      (difference = |x₁ - x₂|) ∧
      (difference = 18)

-- The proof goes here
theorem absolute_value_equation_solution_difference_is_18 :
  absolute_value_equation_solution_difference 18 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_is_18_l3271_327143


namespace NUMINAMATH_CALUDE_symmetric_point_to_origin_l3271_327144

/-- Given a point M with coordinates (-3, -5), proves that the coordinates of the point symmetric to M with respect to the origin are (3, 5). -/
theorem symmetric_point_to_origin (M : ℝ × ℝ) (h : M = (-3, -5)) :
  (- M.1, - M.2) = (3, 5) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_to_origin_l3271_327144


namespace NUMINAMATH_CALUDE_least_lcm_ac_l3271_327121

theorem least_lcm_ac (a b c : ℕ+) (h1 : Nat.lcm a b = 18) (h2 : Nat.lcm b c = 20) :
  ∃ (a' c' : ℕ+), Nat.lcm a' b = 18 ∧ Nat.lcm b c' = 20 ∧ 
    Nat.lcm a' c' = 90 ∧ ∀ (x y : ℕ+), Nat.lcm x b = 18 → Nat.lcm b y = 20 → 
      Nat.lcm x y ≥ 90 := by
sorry

end NUMINAMATH_CALUDE_least_lcm_ac_l3271_327121


namespace NUMINAMATH_CALUDE_same_last_three_digits_l3271_327172

theorem same_last_three_digits (N : ℕ) (h1 : N > 0) :
  (∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ 
   N % 1000 = 100 * a + 10 * b + c ∧
   (N^2) % 1000 = 100 * a + 10 * b + c) →
  N % 1000 = 873 :=
by sorry

end NUMINAMATH_CALUDE_same_last_three_digits_l3271_327172


namespace NUMINAMATH_CALUDE_complement_equivalence_l3271_327123

def U (a : ℝ) := {x : ℕ | 0 < x ∧ x ≤ ⌊a⌋}
def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {4, 5, 6}

theorem complement_equivalence (a : ℝ) :
  (6 ≤ a ∧ a < 7) ↔ (U a \ P = Q) :=
sorry

end NUMINAMATH_CALUDE_complement_equivalence_l3271_327123


namespace NUMINAMATH_CALUDE_linear_equation_solution_comparison_l3271_327178

theorem linear_equation_solution_comparison
  (c c' d d' : ℝ)
  (hc_pos : c > 0)
  (hc'_pos : c' > 0)
  (hc_gt_c' : c > c') :
  ((-d) / c < (-d') / c') ↔ (c * d' < c' * d) := by
sorry

end NUMINAMATH_CALUDE_linear_equation_solution_comparison_l3271_327178


namespace NUMINAMATH_CALUDE_parabola_equation_l3271_327114

-- Define the parabola C
def C (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line x = 2
def Line (x : ℝ) : Prop := x = 2

-- Define the intersection points D and E
def Intersect (p : ℝ) (D E : ℝ × ℝ) : Prop :=
  C p D.1 D.2 ∧ C p E.1 E.2 ∧ Line D.1 ∧ Line E.1

-- Define the orthogonality condition
def Orthogonal (O D E : ℝ × ℝ) : Prop :=
  (D.1 - O.1) * (E.1 - O.1) + (D.2 - O.2) * (E.2 - O.2) = 0

-- The main theorem
theorem parabola_equation (p : ℝ) (D E : ℝ × ℝ) :
  C p D.1 D.2 ∧ C p E.1 E.2 ∧ Line D.1 ∧ Line E.1 ∧ 
  Orthogonal (0, 0) D E →
  ∀ x y : ℝ, C p x y ↔ y^2 = 2*x :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3271_327114


namespace NUMINAMATH_CALUDE_lowest_number_of_students_smallest_multiple_is_120_l3271_327196

theorem lowest_number_of_students (n : ℕ) : n > 0 ∧ 15 ∣ n ∧ 24 ∣ n → n ≥ 120 := by
  sorry

theorem smallest_multiple_is_120 : ∃ n : ℕ, n > 0 ∧ 15 ∣ n ∧ 24 ∣ n ∧ n = 120 := by
  sorry

end NUMINAMATH_CALUDE_lowest_number_of_students_smallest_multiple_is_120_l3271_327196


namespace NUMINAMATH_CALUDE_quadratic_max_condition_l3271_327108

/-- Given a quadratic function y = ax² - 2ax + c with a ≠ 0 and maximum value 2,
    prove that c - a = 2 and a < 0 -/
theorem quadratic_max_condition (a c : ℝ) (h1 : a ≠ 0) :
  (∀ x, a * x^2 - 2*a*x + c ≤ 2) ∧ 
  (∃ x, a * x^2 - 2*a*x + c = 2) →
  c - a = 2 ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_condition_l3271_327108


namespace NUMINAMATH_CALUDE_tangent_line_slope_intercept_difference_l3271_327132

/-- A line passing through two points and tangent to a circle -/
structure TangentLine where
  a : ℝ
  b : ℝ
  passes_through_first : 7 = a * 5 + b
  passes_through_second : 20 = a * 9 + b
  tangent_at : (5, 7) ∈ {(x, y) | y = a * x + b}

/-- The difference between the slope and y-intercept of the tangent line -/
def slope_intercept_difference (line : TangentLine) : ℝ := line.a - line.b

/-- Theorem stating that the difference between slope and y-intercept is 12.5 -/
theorem tangent_line_slope_intercept_difference :
  ∀ (line : TangentLine), slope_intercept_difference line = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_intercept_difference_l3271_327132


namespace NUMINAMATH_CALUDE_minas_age_l3271_327115

/-- Given the ages of Minho, Suhong, and Mina, prove that Mina is 10 years old -/
theorem minas_age (suhong minho mina : ℕ) : 
  minho = 3 * suhong →  -- Minho's age is three times Suhong's age
  mina = 2 * suhong - 2 →  -- Mina's age is two years younger than twice Suhong's age
  suhong + minho + mina = 34 →  -- The sum of the ages of the three is 34
  mina = 10 := by
sorry


end NUMINAMATH_CALUDE_minas_age_l3271_327115


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_l3271_327171

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones and a sphere -/
structure ConeSphereProblem where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ
  sphereRadius : ℝ

/-- The specific problem setup -/
def problemSetup : ConeSphereProblem :=
  { cone1 := { baseRadius := 4, height := 10 }
  , cone2 := { baseRadius := 4, height := 10 }
  , intersectionDistance := 2
  , sphereRadius := 0  -- To be maximized
  }

/-- The theorem stating the maximal value of r^2 -/
theorem max_sphere_radius_squared (setup : ConeSphereProblem) :
  setup.cone1 = setup.cone2 →
  setup.cone1.baseRadius = 4 →
  setup.cone1.height = 10 →
  setup.intersectionDistance = 2 →
  ∃ (r : ℝ), r^2 ≤ 144/29 ∧
    ∀ (s : ℝ), (∃ (config : ConeSphereProblem),
      config.cone1 = setup.cone1 ∧
      config.cone2 = setup.cone2 ∧
      config.intersectionDistance = setup.intersectionDistance ∧
      config.sphereRadius = s) →
    s^2 ≤ r^2 :=
by
  sorry


end NUMINAMATH_CALUDE_max_sphere_radius_squared_l3271_327171


namespace NUMINAMATH_CALUDE_equal_cupcake_distribution_l3271_327194

theorem equal_cupcake_distribution (total_cupcakes : ℕ) (num_children : ℕ) 
  (h1 : total_cupcakes = 96) (h2 : num_children = 8) :
  total_cupcakes / num_children = 12 := by
  sorry

end NUMINAMATH_CALUDE_equal_cupcake_distribution_l3271_327194


namespace NUMINAMATH_CALUDE_workshop_workers_count_l3271_327153

/-- The total number of workers in a workshop given specific salary conditions -/
theorem workshop_workers_count : ℕ :=
  let average_salary : ℚ := 1000
  let technician_salary : ℚ := 1200
  let other_salary : ℚ := 820
  let technician_count : ℕ := 10
  let total_workers : ℕ := 21

  have h1 : average_salary * total_workers = 
    technician_salary * technician_count + other_salary * (total_workers - technician_count) := by sorry

  total_workers


end NUMINAMATH_CALUDE_workshop_workers_count_l3271_327153


namespace NUMINAMATH_CALUDE_line_through_tangent_intersections_l3271_327138

/-- The equation of an ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- A point inside the ellipse -/
def M : ℝ × ℝ := (3, 2)

/-- A line intersecting the ellipse -/
structure IntersectingLine where
  a : ℝ × ℝ
  b : ℝ × ℝ
  ha : is_ellipse a.1 a.2
  hb : is_ellipse b.1 b.2

/-- The intersection point of tangent lines -/
structure TangentIntersection where
  p : ℝ × ℝ
  line : IntersectingLine
  -- Additional properties for tangent intersection could be added here

/-- The theorem statement -/
theorem line_through_tangent_intersections 
  (ab cd : IntersectingLine) 
  (p q : TangentIntersection) 
  (hp : p.line = ab) 
  (hq : q.line = cd) 
  (hab : ab.a.1 * M.1 / 25 + ab.a.2 * M.2 / 9 = 1)
  (hcd : cd.a.1 * M.1 / 25 + cd.a.2 * M.2 / 9 = 1) :
  ∃ (k : ℝ), ∀ (x y : ℝ), y = k * x + (1 - 3 * k / 25) * 9 / 2 ↔ 3 * x / 25 + 2 * y / 9 = 1 :=
sorry

end NUMINAMATH_CALUDE_line_through_tangent_intersections_l3271_327138


namespace NUMINAMATH_CALUDE_gcd_problem_l3271_327122

theorem gcd_problem (p : Nat) (h : Prime p) :
  Nat.gcd (p^7 + 1) (p^7 + p^3 + 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_problem_l3271_327122


namespace NUMINAMATH_CALUDE_shaded_perimeter_value_l3271_327158

/-- The perimeter of the shaded region formed by four quarter-circle arcs in a unit square --/
def shadedPerimeter : ℝ := sorry

/-- The square PQRS with side length 1 --/
def unitSquare : Set (ℝ × ℝ) := sorry

/-- Arc TRU with center P --/
def arcTRU : Set (ℝ × ℝ) := sorry

/-- Arc VPW with center R --/
def arcVPW : Set (ℝ × ℝ) := sorry

/-- Arc UV with center S --/
def arcUV : Set (ℝ × ℝ) := sorry

/-- Arc WT with center Q --/
def arcWT : Set (ℝ × ℝ) := sorry

/-- The theorem stating that the perimeter of the shaded region is (2√2 - 1)π --/
theorem shaded_perimeter_value : shadedPerimeter = (2 * Real.sqrt 2 - 1) * Real.pi := by sorry

end NUMINAMATH_CALUDE_shaded_perimeter_value_l3271_327158


namespace NUMINAMATH_CALUDE_suitable_land_size_l3271_327198

def previous_property : ℝ := 2
def land_multiplier : ℝ := 10
def pond_size : ℝ := 1

theorem suitable_land_size :
  let new_property := previous_property * land_multiplier
  let suitable_land := new_property - pond_size
  suitable_land = 19 := by sorry

end NUMINAMATH_CALUDE_suitable_land_size_l3271_327198


namespace NUMINAMATH_CALUDE_fraction_equality_l3271_327124

theorem fraction_equality : (1722^2 - 1715^2) / (1731^2 - 1708^2) = (7 * 3437) / (23 * 3439) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3271_327124


namespace NUMINAMATH_CALUDE_direct_proportional_function_inequality_l3271_327111

/-- A direct proportional function satisfying f[f(x)] ≥ x - 3 for all real x
    must be either f(x) = -x or f(x) = x -/
theorem direct_proportional_function_inequality 
  (f : ℝ → ℝ) 
  (h_prop : ∃ (a : ℝ), ∀ x, f x = a * x) 
  (h_ineq : ∀ x, f (f x) ≥ x - 3) :
  (∀ x, f x = -x) ∨ (∀ x, f x = x) := by
sorry

end NUMINAMATH_CALUDE_direct_proportional_function_inequality_l3271_327111


namespace NUMINAMATH_CALUDE_shopping_trip_result_l3271_327184

def shopping_trip (initial_amount : ℝ) (video_game_price : ℝ) (video_game_discount : ℝ)
  (goggles_percent : ℝ) (goggles_tax : ℝ) (jacket_price : ℝ) (jacket_discount : ℝ)
  (book_percent : ℝ) (book_tax : ℝ) (gift_card : ℝ) : ℝ :=
  let video_game_cost := video_game_price * (1 - video_game_discount)
  let remaining_after_game := initial_amount - video_game_cost
  let goggles_cost := remaining_after_game * goggles_percent * (1 + goggles_tax)
  let remaining_after_goggles := remaining_after_game - goggles_cost
  let jacket_cost := jacket_price * (1 - jacket_discount)
  let remaining_after_jacket := remaining_after_goggles - jacket_cost
  let book_cost := remaining_after_jacket * book_percent * (1 + book_tax)
  remaining_after_jacket - book_cost

theorem shopping_trip_result :
  shopping_trip 200 60 0.15 0.20 0.08 80 0.25 0.10 0.05 20 = 50.85 := by
  sorry

#eval shopping_trip 200 60 0.15 0.20 0.08 80 0.25 0.10 0.05 20

end NUMINAMATH_CALUDE_shopping_trip_result_l3271_327184


namespace NUMINAMATH_CALUDE_largest_number_l3271_327148

theorem largest_number (S : Set ℤ) (h : S = {0, 2, -1, -2}) : 
  ∃ m ∈ S, ∀ x ∈ S, x ≤ m ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3271_327148


namespace NUMINAMATH_CALUDE_shaded_area_sum_l3271_327100

/-- Represents the shaded area in each step of the square division pattern -/
def shadedAreaSeries : ℕ → ℚ
  | 0 => 1/4
  | n + 1 => (shadedAreaSeries n) / 16

/-- The sum of the infinite geometric series representing the total shaded area -/
def totalShadedArea : ℚ := 4/15

theorem shaded_area_sum :
  (∑' n, shadedAreaSeries n) = totalShadedArea := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_sum_l3271_327100


namespace NUMINAMATH_CALUDE_unique_x_for_all_y_l3271_327183

theorem unique_x_for_all_y : ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 18 * y + x - 2 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_x_for_all_y_l3271_327183


namespace NUMINAMATH_CALUDE_power_function_properties_l3271_327139

def f (m : ℕ) (x : ℝ) : ℝ := x^(3*m - 5)

theorem power_function_properties (m : ℕ) :
  (∀ x y, 0 < x ∧ x < y → f m y < f m x) ∧
  (∀ x, f m (-x) = f m x) →
  m = 1 := by sorry

end NUMINAMATH_CALUDE_power_function_properties_l3271_327139


namespace NUMINAMATH_CALUDE_total_spent_is_two_dollars_l3271_327188

/-- The price of a single pencil in cents -/
def pencil_price : ℕ := 20

/-- The number of pencils Tolu wants -/
def tolu_pencils : ℕ := 3

/-- The number of pencils Robert wants -/
def robert_pencils : ℕ := 5

/-- The number of pencils Melissa wants -/
def melissa_pencils : ℕ := 2

/-- Theorem: The total amount spent by the students is $2.00 -/
theorem total_spent_is_two_dollars :
  (tolu_pencils * pencil_price + robert_pencils * pencil_price + melissa_pencils * pencil_price) / 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_two_dollars_l3271_327188


namespace NUMINAMATH_CALUDE_intersection_chord_length_l3271_327131

/-- Line in parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Circle in polar form -/
structure PolarCircle where
  ρ : ℝ → ℝ

/-- Chord length calculation -/
def chordLength (l : ParametricLine) (c : PolarCircle) : ℝ := sorry

/-- Main theorem -/
theorem intersection_chord_length :
  let l : ParametricLine := { x := fun t => t + 1, y := fun t => t - 3 }
  let c : PolarCircle := { ρ := fun θ => 4 * Real.cos θ }
  chordLength l c = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l3271_327131


namespace NUMINAMATH_CALUDE_square_sum_value_l3271_327118

theorem square_sum_value (m n : ℝ) :
  (m^2 + 3*n^2)^2 - 4*(m^2 + 3*n^2) - 12 = 0 →
  m^2 + 3*n^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l3271_327118


namespace NUMINAMATH_CALUDE_special_line_equation_l3271_327166

/-- A line passing through point M(3, -4) with intercepts on the coordinate axes that are opposite numbers -/
structure SpecialLine where
  -- The slope of the line
  slope : ℝ
  -- The y-intercept of the line
  y_intercept : ℝ
  -- The line passes through point M(3, -4)
  passes_through_M : slope * 3 + y_intercept = -4
  -- The intercepts on the coordinate axes are opposite numbers
  opposite_intercepts : (y_intercept = 0 ∧ -y_intercept / slope = 0) ∨ 
                        (y_intercept ≠ 0 ∧ -y_intercept / slope = -y_intercept)

/-- The equation of the special line is either x + y = -1 or 4x + 3y = 0 -/
theorem special_line_equation (l : SpecialLine) : 
  (l.slope = -1 ∧ l.y_intercept = -1) ∨ (l.slope = -4/3 ∧ l.y_intercept = 0) := by
  sorry

#check special_line_equation

end NUMINAMATH_CALUDE_special_line_equation_l3271_327166


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_for_solution_l3271_327155

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2 * |x - 1|

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x > 5} = {x : ℝ | x < -1/3 ∨ x > 3} :=
sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_solution :
  {a : ℝ | ∃ x, f a x - |x - 1| ≤ |a - 2|} = {a : ℝ | a ≤ 3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_for_solution_l3271_327155


namespace NUMINAMATH_CALUDE_gwens_birthday_money_l3271_327151

/-- The amount of money Gwen spent -/
def amount_spent : ℕ := 8

/-- The amount of money Gwen has left -/
def amount_left : ℕ := 6

/-- The total amount of money Gwen received for her birthday -/
def total_amount : ℕ := amount_spent + amount_left

theorem gwens_birthday_money : total_amount = 14 := by
  sorry

end NUMINAMATH_CALUDE_gwens_birthday_money_l3271_327151


namespace NUMINAMATH_CALUDE_sin_315_degrees_l3271_327145

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l3271_327145


namespace NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l3271_327142

variables (P S M t n : ℝ)

/-- The margin M can be expressed in terms of the selling price S, given the production cost P, tax rate t, and a constant n. -/
theorem margin_in_terms_of_selling_price
  (h1 : S = P * (1 + t/100))  -- Selling price including tax
  (h2 : M = P / n)            -- Margin definition
  (h3 : n > 0)                -- n is positive (implied by the context)
  (h4 : t ≥ 0)                -- Tax rate is non-negative (implied by the context)
  : M = S / (n * (1 + t/100)) :=
sorry

end NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l3271_327142


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l3271_327186

def satisfiesConditions (n : Nat) : Prop :=
  n % 43 = 0 ∧
  n < 43 * 9 ∧
  n % 2 = 1 ∧
  n % 3 = 1 ∧
  n % 4 = 1 ∧
  n % 5 = 1 ∧
  n % 6 = 1

theorem smallest_satisfying_number :
  satisfiesConditions 301 ∧
  ∀ m : Nat, m < 301 → ¬satisfiesConditions m :=
by sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l3271_327186


namespace NUMINAMATH_CALUDE_paint_needed_l3271_327174

theorem paint_needed (total_needed : ℕ) (existing : ℕ) (newly_bought : ℕ) 
  (h1 : total_needed = 70)
  (h2 : existing = 36)
  (h3 : newly_bought = 23) :
  total_needed - (existing + newly_bought) = 11 := by
  sorry

end NUMINAMATH_CALUDE_paint_needed_l3271_327174


namespace NUMINAMATH_CALUDE_farm_animals_l3271_327112

theorem farm_animals (horses cows : ℕ) : 
  horses = 5 * cows →                           -- Initial ratio of horses to cows is 5:1
  (horses - 15) = 17 * (cows + 15) / 7 →        -- New ratio after transaction is 17:7
  horses - 15 - (cows + 15) = 50 := by          -- Difference after transaction is 50
sorry

end NUMINAMATH_CALUDE_farm_animals_l3271_327112


namespace NUMINAMATH_CALUDE_sale_price_for_55_percent_profit_l3271_327181

/-- Proves that the sale price for making a 55% profit is $2792, given the conditions. -/
theorem sale_price_for_55_percent_profit 
  (equal_profit_loss : ∀ (cp sp_profit : ℝ), sp_profit - cp = cp - 448)
  (profit_amount : ∀ (cp : ℝ), 0.55 * cp = 992) :
  ∃ (cp : ℝ), cp + 992 = 2792 :=
by sorry

end NUMINAMATH_CALUDE_sale_price_for_55_percent_profit_l3271_327181


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l3271_327140

def arrange_books (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n

theorem book_arrangement_proof :
  arrange_books 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l3271_327140


namespace NUMINAMATH_CALUDE_max_blank_squares_l3271_327107

/-- Represents a grid of unit squares -/
structure Grid :=
  (size : ℕ)

/-- Represents a triangle placement on the grid -/
structure TrianglePlacement :=
  (grid : Grid)
  (covers_all_segments : Prop)

/-- Represents the count of squares without triangles -/
def blank_squares (tp : TrianglePlacement) : ℕ := sorry

/-- The main theorem: maximum number of blank squares in a 100x100 grid -/
theorem max_blank_squares :
  ∀ (tp : TrianglePlacement),
    tp.grid.size = 100 →
    tp.covers_all_segments →
    blank_squares tp ≤ 2450 :=
by sorry

end NUMINAMATH_CALUDE_max_blank_squares_l3271_327107


namespace NUMINAMATH_CALUDE_angle_value_in_plane_figure_l3271_327175

theorem angle_value_in_plane_figure (x : ℝ) : 
  x > 0 ∧ 
  x + x + 140 = 360 → 
  x = 110 := by
sorry

end NUMINAMATH_CALUDE_angle_value_in_plane_figure_l3271_327175


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3271_327173

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 864 →
  (∃ (side_length : ℝ), 
    surface_area = 6 * side_length^2 ∧
    volume = side_length^3) →
  volume = 1728 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3271_327173


namespace NUMINAMATH_CALUDE_triangle_problem_l3271_327104

theorem triangle_problem (a b c A B C : ℝ) : 
  -- Conditions
  (2 * Real.sin (7 * π / 6) * Real.sin (π / 6 + C) + Real.cos C = -1 / 2) →
  (c = Real.sqrt 13) →
  (1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3) →
  -- Conclusions
  (C = π / 3) ∧ 
  (Real.sin A + Real.sin B = 7 * Real.sqrt 39 / 26) := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3271_327104


namespace NUMINAMATH_CALUDE_at_least_one_positive_l3271_327190

theorem at_least_one_positive (a b c : ℝ) :
  (a > 0 ∨ b > 0 ∨ c > 0) ↔ ¬(a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_positive_l3271_327190


namespace NUMINAMATH_CALUDE_quadratic_roots_root_of_two_two_as_only_root_l3271_327165

/-- The quadratic equation x^2 - 2px + q = 0 -/
def quadratic_equation (p q x : ℝ) : Prop :=
  x^2 - 2*p*x + q = 0

/-- The discriminant of the quadratic equation -/
def discriminant (p q : ℝ) : ℝ :=
  4*p^2 - 4*q

theorem quadratic_roots (p q : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation p q x ∧ quadratic_equation p q y) ↔ q < p^2 :=
sorry

theorem root_of_two (p q : ℝ) :
  quadratic_equation p q 2 ↔ q = 4*p - 4 :=
sorry

theorem two_as_only_root (p q : ℝ) :
  (∀ x : ℝ, quadratic_equation p q x ↔ x = 2) ↔ (p = 2 ∧ q = 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_root_of_two_two_as_only_root_l3271_327165


namespace NUMINAMATH_CALUDE_sum_geq_three_l3271_327135

theorem sum_geq_three (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : a + b + c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_three_l3271_327135


namespace NUMINAMATH_CALUDE_three_player_rotation_l3271_327113

/-- Represents the number of games played by each player in a three-player table tennis rotation. -/
structure GameCount where
  player1 : ℕ
  player2 : ℕ
  player3 : ℕ

/-- 
Theorem: In a three-player table tennis rotation where the losing player is replaced by the non-participating player,
if Player 1 played 10 games and Player 2 played 21 games, then Player 3 must have played 11 games.
-/
theorem three_player_rotation (gc : GameCount) 
  (h1 : gc.player1 = 10)
  (h2 : gc.player2 = 21)
  (h_total : gc.player1 + gc.player2 + gc.player3 = 2 * gc.player2) :
  gc.player3 = 11 := by
  sorry


end NUMINAMATH_CALUDE_three_player_rotation_l3271_327113


namespace NUMINAMATH_CALUDE_calculate_first_train_length_l3271_327152

/-- The length of the first train given the specified conditions -/
def first_train_length (first_train_speed second_train_speed : ℝ)
                       (second_train_length : ℝ)
                       (crossing_time : ℝ) : ℝ :=
  (first_train_speed - second_train_speed) * crossing_time - second_train_length

/-- Theorem stating the length of the first train under given conditions -/
theorem calculate_first_train_length :
  first_train_length 72 36 300 69.99440044796417 = 399.9440044796417 := by
  sorry

end NUMINAMATH_CALUDE_calculate_first_train_length_l3271_327152


namespace NUMINAMATH_CALUDE_least_value_quadratic_l3271_327169

theorem least_value_quadratic (a : ℝ) :
  (∀ x : ℝ, x^2 - 12*x + 35 ≤ 0 → a ≤ x) ↔ a = 5 := by sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l3271_327169


namespace NUMINAMATH_CALUDE_gain_amount_theorem_l3271_327197

/-- The amount on which a gain was made, given the gain and gain percent -/
def amount_with_gain (gain : ℚ) (gain_percent : ℚ) : ℚ :=
  gain / (gain_percent / 100)

/-- Theorem: The amount on which a gain of 0.70 rupees was made, given a gain percent of 1%, is equal to 70 rupees -/
theorem gain_amount_theorem : amount_with_gain 0.70 1 = 70 := by
  sorry

end NUMINAMATH_CALUDE_gain_amount_theorem_l3271_327197


namespace NUMINAMATH_CALUDE_ball_ratio_proof_l3271_327127

theorem ball_ratio_proof (a b x : ℕ) : 
  (a / (a + b + x) = 1/4) →
  ((a + x) / (b + x) = 2/3) →
  (3*a - b = x) →
  (2*b - 3*a = x) →
  (a / b = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ball_ratio_proof_l3271_327127


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_18_l3271_327160

theorem factorization_of_2m_squared_minus_18 (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_18_l3271_327160


namespace NUMINAMATH_CALUDE_stone_length_is_four_dm_l3271_327159

/-- Represents the dimensions of a rectangular hall in meters -/
structure HallDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a rectangular stone in decimeters -/
structure StoneDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : HallDimensions) : ℝ := d.length * d.width

/-- Converts meters to decimeters -/
def meterToDecimeter (m : ℝ) : ℝ := m * 10

/-- Theorem: Given a hall of 36m x 15m, 2700 stones required, and stone width of 5dm,
    prove that the length of each stone is 4 dm -/
theorem stone_length_is_four_dm (hall : HallDimensions) (stone : StoneDimensions) 
    (num_stones : ℕ) : 
    hall.length = 36 → 
    hall.width = 15 → 
    num_stones = 2700 → 
    stone.width = 5 → 
    stone.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_stone_length_is_four_dm_l3271_327159


namespace NUMINAMATH_CALUDE_abs_square_eq_neg_cube_l3271_327105

theorem abs_square_eq_neg_cube (a b : ℤ) : |a|^2 = -(b^3) → a = -8 ∧ b = -4 :=
by sorry

end NUMINAMATH_CALUDE_abs_square_eq_neg_cube_l3271_327105


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3271_327191

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (3*m + 2)*x + 2*(m + 6)

-- Define the property of having two real roots greater than 3
def has_two_roots_greater_than_three (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 3 ∧ x₂ > 3 ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0

-- Theorem statement
theorem quadratic_roots_condition (m : ℝ) :
  has_two_roots_greater_than_three m ↔ 4/3 < m ∧ m < 15/7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3271_327191


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l3271_327195

/-- Given a rhombus with area 80 cm² and one diagonal 16 cm, prove the other diagonal is 10 cm -/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) : 
  area = 80 → d1 = 16 → area = (d1 * d2) / 2 → d2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l3271_327195


namespace NUMINAMATH_CALUDE_tens_digit_of_3_to_2023_l3271_327147

theorem tens_digit_of_3_to_2023 : ∃ n : ℕ, 3^2023 ≡ 20 + n [ZMOD 100] :=
by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_to_2023_l3271_327147


namespace NUMINAMATH_CALUDE_system_solution_l3271_327136

theorem system_solution (a b c : ℝ) : 
  a^2 + a*b + c^2 = 31 ∧ 
  b^2 + a*b - c^2 = 18 ∧ 
  a^2 - b^2 = 7 → 
  c = Real.sqrt 3 ∨ c = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3271_327136


namespace NUMINAMATH_CALUDE_sum_is_parabola_l3271_327133

-- Define the original parabola
def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the reflected parabola
def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 - b * x + c

-- Define the translated original parabola (f)
def f (a b c : ℝ) (x : ℝ) : ℝ := original_parabola a b c x + 3

-- Define the translated reflected parabola (g)
def g (a b c : ℝ) (x : ℝ) : ℝ := reflected_parabola a b c x - 2

-- Theorem: The sum of f and g is a parabola
theorem sum_is_parabola (a b c : ℝ) :
  ∃ (A C : ℝ), ∀ x, f a b c x + g a b c x = A * x^2 + C :=
sorry

end NUMINAMATH_CALUDE_sum_is_parabola_l3271_327133


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3271_327163

/-- A cubic function with a local maximum at x = -1 and a local minimum at x = 3 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties :
  ∃ (a b c : ℝ),
    (f_deriv a b (-1) = 0) ∧
    (f_deriv a b 3 = 0) ∧
    (f a b c (-1) = 7) ∧
    (a = -3) ∧
    (b = -9) ∧
    (c = 2) ∧
    (f a b c 3 = -25) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3271_327163


namespace NUMINAMATH_CALUDE_keith_picked_six_apples_l3271_327146

/-- The number of apples picked by Mike -/
def mike_apples : ℕ := 7

/-- The number of apples picked by Nancy -/
def nancy_apples : ℕ := 3

/-- The total number of apples picked -/
def total_apples : ℕ := 16

/-- The number of apples picked by Keith -/
def keith_apples : ℕ := total_apples - (mike_apples + nancy_apples)

theorem keith_picked_six_apples : keith_apples = 6 := by
  sorry

end NUMINAMATH_CALUDE_keith_picked_six_apples_l3271_327146


namespace NUMINAMATH_CALUDE_order_of_numbers_l3271_327199

theorem order_of_numbers : 70.3 > 0.37 ∧ 0.37 > Real.log 0.3 := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3271_327199


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_44_l3271_327116

theorem smallest_four_digit_divisible_by_44 : 
  ∀ n : ℕ, 1000 ≤ n → n < 10000 → n % 44 = 0 → 1023 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_44_l3271_327116


namespace NUMINAMATH_CALUDE_square_eq_necessary_condition_l3271_327157

theorem square_eq_necessary_condition (x h k : ℝ) :
  (x + h)^2 = k → k ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_eq_necessary_condition_l3271_327157


namespace NUMINAMATH_CALUDE_gcd_1721_1733_l3271_327109

theorem gcd_1721_1733 : Nat.gcd 1721 1733 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1721_1733_l3271_327109


namespace NUMINAMATH_CALUDE_sine_addition_formula_l3271_327192

theorem sine_addition_formula (α β : Real) : 
  Real.sin (α - β) * Real.cos β + Real.cos (α - β) * Real.sin β = Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_sine_addition_formula_l3271_327192


namespace NUMINAMATH_CALUDE_tan_alpha_two_implies_expression_value_l3271_327189

theorem tan_alpha_two_implies_expression_value (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_implies_expression_value_l3271_327189


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3271_327130

theorem trigonometric_equation_solution :
  ∀ x : ℝ,
  (Real.sin (2019 * x))^4 + (Real.cos (2022 * x))^2019 * (Real.cos (2019 * x))^2018 = 1 ↔
  (∃ n : ℤ, x = π / 4038 + π * n / 2019) ∨ (∃ k : ℤ, x = π * k / 3) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3271_327130


namespace NUMINAMATH_CALUDE_david_average_marks_l3271_327187

def david_marks : List ℕ := [74, 65, 82, 67, 90]

theorem david_average_marks :
  (david_marks.sum : ℚ) / david_marks.length = 75.6 := by sorry

end NUMINAMATH_CALUDE_david_average_marks_l3271_327187
