import Mathlib

namespace NUMINAMATH_CALUDE_exactly_two_integers_satisfy_l3762_376279

-- Define the circle
def circle_center : ℝ × ℝ := (3, -3)
def circle_radius : ℝ := 8

-- Define the point (x, x+2)
def point (x : ℤ) : ℝ × ℝ := (x, x + 2)

-- Define the condition for a point to be inside or on the circle
def inside_or_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 ≤ circle_radius^2

-- Theorem statement
theorem exactly_two_integers_satisfy :
  ∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x : ℤ, x ∈ s ↔ inside_or_on_circle (point x) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_integers_satisfy_l3762_376279


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l3762_376237

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)

/-- The theorem stating that functions satisfying the equation are either constant zero or square. -/
theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l3762_376237


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_three_one_forty_one_satisfies_conditions_main_result_l3762_376228

theorem greatest_integer_with_gcf_three (n : ℕ) : n < 150 ∧ Nat.gcd n 24 = 3 → n ≤ 141 :=
by
  sorry

theorem one_forty_one_satisfies_conditions : 141 < 150 ∧ Nat.gcd 141 24 = 3 :=
by
  sorry

theorem main_result : ∃ (n : ℕ), n < 150 ∧ Nat.gcd n 24 = 3 ∧ 
  ∀ (m : ℕ), m < 150 ∧ Nat.gcd m 24 = 3 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_three_one_forty_one_satisfies_conditions_main_result_l3762_376228


namespace NUMINAMATH_CALUDE_simplify_expression_l3762_376218

theorem simplify_expression (a b : ℝ) : 6*a - 8*b - 2*(3*a + b) = -10*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3762_376218


namespace NUMINAMATH_CALUDE_grid_solution_l3762_376236

/-- A 3x3 grid represented as a function from (Fin 3 × Fin 3) to ℕ -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if two cells are adjacent in the grid -/
def adjacent (a b : Fin 3 × Fin 3) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ b.2.val + 1 = a.2.val)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ b.1.val + 1 = a.1.val))

/-- The given grid with known values -/
def given_grid : Grid :=
  fun i j =>
    if i = 0 ∧ j = 1 then 1
    else if i = 0 ∧ j = 2 then 9
    else if i = 1 ∧ j = 0 then 3
    else if i = 1 ∧ j = 1 then 5
    else if i = 2 ∧ j = 2 then 7
    else 0  -- placeholder for unknown values

theorem grid_solution :
  ∀ g : Grid,
  (∀ i j, g i j ∈ Finset.range 10) →  -- all numbers are from 1 to 9
  (∀ a b, adjacent a b → g a.1 a.2 + g b.1 b.2 < 12) →  -- sum of adjacent cells < 12
  (∀ i j, given_grid i j ≠ 0 → g i j = given_grid i j) →  -- matches given values
  g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_grid_solution_l3762_376236


namespace NUMINAMATH_CALUDE_overtake_scenario_l3762_376240

/-- Represents the scenario where three people travel at different speeds and overtake each other -/
structure TravelScenario where
  speed_a : ℝ
  speed_b : ℝ
  speed_k : ℝ
  b_delay : ℝ
  overtake_time : ℝ
  k_start_time : ℝ

/-- The theorem statement based on the given problem -/
theorem overtake_scenario (s : TravelScenario) 
  (h1 : s.speed_a = 30)
  (h2 : s.speed_b = 40)
  (h3 : s.speed_k = 60)
  (h4 : s.b_delay = 5)
  (h5 : s.speed_a * s.overtake_time = s.speed_b * (s.overtake_time - s.b_delay))
  (h6 : s.speed_a * s.overtake_time = s.speed_k * s.k_start_time) :
  s.k_start_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_overtake_scenario_l3762_376240


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l3762_376288

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m : Line) (α β : Plane) :
  perpendicular α β → 
  perpendicularLP m β → 
  ¬subset m α → 
  parallel m α :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l3762_376288


namespace NUMINAMATH_CALUDE_steve_total_cost_theorem_l3762_376262

def steve_total_cost (mike_dvd_price : ℝ) (steve_extra_dvd_price : ℝ) 
  (steve_extra_dvd_count : ℕ) (shipping_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let steve_favorite_dvd_price := 2 * mike_dvd_price
  let steve_extra_dvds_cost := steve_extra_dvd_count * steve_extra_dvd_price
  let total_dvds_cost := steve_favorite_dvd_price + steve_extra_dvds_cost
  let shipping_cost := shipping_rate * total_dvds_cost
  let subtotal := total_dvds_cost + shipping_cost
  let tax := tax_rate * subtotal
  subtotal + tax

theorem steve_total_cost_theorem :
  steve_total_cost 5 7 2 0.8 0.1 = 47.52 := by
  sorry

end NUMINAMATH_CALUDE_steve_total_cost_theorem_l3762_376262


namespace NUMINAMATH_CALUDE_tangent_line_slope_l3762_376294

/-- Given a line y = kx + 1 tangent to the curve y = 1/x at point (a, 1/a) and passing through (0, 1), k equals -1/4 -/
theorem tangent_line_slope (k a : ℝ) : 
  (∀ x, x ≠ 0 → (k * x + 1) = 1 / x ∨ (k * x + 1) > 1 / x) → -- tangent condition
  (k * 0 + 1 = 1) →                                         -- passes through (0, 1)
  (k * a + 1 = 1 / a) →                                     -- point of tangency
  (k = -1 / (a^2)) →                                        -- slope at point of tangency
  (k = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l3762_376294


namespace NUMINAMATH_CALUDE_f_monotonic_intervals_f_inequality_solution_f_max_value_l3762_376259

-- Define the function f(x) = x|x-2|
def f (x : ℝ) : ℝ := x * abs (x - 2)

-- Theorem for monotonic intervals
theorem f_monotonic_intervals :
  (∀ x y, x ≤ y ∧ y ≤ 1 → f x ≤ f y) ∧
  (∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f x ≥ f y) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f x ≤ f y) :=
sorry

-- Theorem for the inequality solution
theorem f_inequality_solution :
  ∀ x, f x < 3 ↔ x < 3 :=
sorry

-- Theorem for the maximum value
theorem f_max_value (a : ℝ) (h : 0 < a ∧ a ≤ 2) :
  (∀ x, 0 ≤ x ∧ x ≤ a → f x ≤ (if a ≤ 1 then a * (2 - a) else 1)) ∧
  (∃ x, 0 ≤ x ∧ x ≤ a ∧ f x = (if a ≤ 1 then a * (2 - a) else 1)) :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_intervals_f_inequality_solution_f_max_value_l3762_376259


namespace NUMINAMATH_CALUDE_jons_laundry_loads_l3762_376249

/-- Represents the laundry machine and Jon's clothes -/
structure LaundryProblem where
  machine_capacity : ℝ
  shirt_weight : ℝ
  pants_weight : ℝ
  sock_weight : ℝ
  jacket_weight : ℝ
  shirt_count : ℕ
  pants_count : ℕ
  sock_count : ℕ
  jacket_count : ℕ

/-- Calculates the minimum number of loads required -/
def minimum_loads (problem : LaundryProblem) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of loads for Jon's laundry is 5 -/
theorem jons_laundry_loads :
  let problem : LaundryProblem :=
    { machine_capacity := 8
    , shirt_weight := 1/4
    , pants_weight := 1/2
    , sock_weight := 1/6
    , jacket_weight := 2
    , shirt_count := 20
    , pants_count := 20
    , sock_count := 18
    , jacket_count := 6
    }
  minimum_loads problem = 5 := by
  sorry

end NUMINAMATH_CALUDE_jons_laundry_loads_l3762_376249


namespace NUMINAMATH_CALUDE_symmetry_axis_shifted_even_function_l3762_376242

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define what it means for a function to have an axis of symmetry
def has_axis_of_symmetry (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- State the theorem
theorem symmetry_axis_shifted_even_function :
  is_even (λ x => f (x + 2)) → has_axis_of_symmetry f 2 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axis_shifted_even_function_l3762_376242


namespace NUMINAMATH_CALUDE_largest_lcm_with_18_l3762_376232

theorem largest_lcm_with_18 :
  (List.map (lcm 18) [3, 6, 9, 12, 15, 18]).maximum? = some 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_18_l3762_376232


namespace NUMINAMATH_CALUDE_pokemon_cards_total_l3762_376263

/-- The number of people with Pokemon cards -/
def num_people : ℕ := 4

/-- The number of Pokemon cards each person has -/
def cards_per_person : ℕ := 14

/-- The total number of Pokemon cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem pokemon_cards_total : total_cards = 56 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_total_l3762_376263


namespace NUMINAMATH_CALUDE_lindas_savings_l3762_376231

theorem lindas_savings (savings : ℝ) : 
  (3 / 5 : ℝ) * savings + 400 = savings → savings = 1000 := by
sorry

end NUMINAMATH_CALUDE_lindas_savings_l3762_376231


namespace NUMINAMATH_CALUDE_sam_drew_age_problem_l3762_376253

/-- The combined age of Sam and Drew given Sam's age and the relation between their ages -/
def combinedAge (samAge : ℕ) (drewAge : ℕ) : ℕ := samAge + drewAge

theorem sam_drew_age_problem :
  let samAge : ℕ := 18
  let drewAge : ℕ := 2 * samAge
  combinedAge samAge drewAge = 54 := by
  sorry

end NUMINAMATH_CALUDE_sam_drew_age_problem_l3762_376253


namespace NUMINAMATH_CALUDE_g_eval_l3762_376292

/-- The function g(x) = 3x^2 - 6x + 8 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 8

/-- Theorem: 4g(2) + 2g(-2) = 96 -/
theorem g_eval : 4 * g 2 + 2 * g (-2) = 96 := by
  sorry

end NUMINAMATH_CALUDE_g_eval_l3762_376292


namespace NUMINAMATH_CALUDE_complex_modulus_l3762_376296

theorem complex_modulus (z : ℂ) : z = -1 + Complex.I * Real.sqrt 3 → Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3762_376296


namespace NUMINAMATH_CALUDE_overlapping_triangle_area_l3762_376274

/-- Given a rectangle with length 8 and width 4, when folded along its diagonal,
    the area of the overlapping triangle is 10. -/
theorem overlapping_triangle_area (length width : ℝ) (h1 : length = 8) (h2 : width = 4) :
  let diagonal := Real.sqrt (length ^ 2 + width ^ 2)
  let overlap_base := (length ^ 2 + width ^ 2) / (2 * length)
  let overlap_height := width
  let overlap_area := (1 / 2) * overlap_base * overlap_height
  overlap_area = 10 := by
sorry

end NUMINAMATH_CALUDE_overlapping_triangle_area_l3762_376274


namespace NUMINAMATH_CALUDE_sam_above_average_l3762_376270

/-- The number of shooting stars counted by Bridget -/
def bridget_count : ℕ := 14

/-- The number of shooting stars counted by Reginald -/
def reginald_count : ℕ := bridget_count - 2

/-- The number of shooting stars counted by Sam -/
def sam_count : ℕ := reginald_count + 4

/-- The average number of shooting stars counted by the three observers -/
def average_count : ℚ := (bridget_count + reginald_count + sam_count) / 3

/-- Theorem stating that Sam counted 2 more shooting stars than the average -/
theorem sam_above_average : sam_count - average_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_sam_above_average_l3762_376270


namespace NUMINAMATH_CALUDE_positive_integer_solutions_m_value_when_x_equals_y_fixed_solution_l3762_376246

-- Define the system of equations
def equation1 (x y : ℤ) : Prop := 2*x + y - 6 = 0
def equation2 (x y m : ℤ) : Prop := 2*x - 2*y + m*y + 8 = 0

-- Theorem for part 1
theorem positive_integer_solutions :
  ∀ x y : ℤ, x > 0 ∧ y > 0 ∧ equation1 x y ↔ (x = 2 ∧ y = 2) ∨ (x = 1 ∧ y = 4) :=
sorry

-- Theorem for part 2
theorem m_value_when_x_equals_y :
  ∃ m : ℤ, ∀ x y : ℤ, x = y ∧ equation1 x y ∧ equation2 x y m → m = -4 :=
sorry

-- Theorem for part 3
theorem fixed_solution :
  ∀ m : ℤ, equation2 (-4) 0 m :=
sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_m_value_when_x_equals_y_fixed_solution_l3762_376246


namespace NUMINAMATH_CALUDE_soft_drinks_bought_l3762_376248

theorem soft_drinks_bought (soft_drink_cost : ℕ) (candy_bars : ℕ) (candy_bar_cost : ℕ) (total_spent : ℕ) : 
  soft_drink_cost = 4 →
  candy_bars = 5 →
  candy_bar_cost = 4 →
  total_spent = 28 →
  ∃ (num_soft_drinks : ℕ), num_soft_drinks * soft_drink_cost + candy_bars * candy_bar_cost = total_spent ∧ num_soft_drinks = 2 :=
by sorry

end NUMINAMATH_CALUDE_soft_drinks_bought_l3762_376248


namespace NUMINAMATH_CALUDE_unit_vector_AB_l3762_376206

/-- Given two points A and B in a 2D plane, prove that the unit vector parallel to vector AB is (3/5, -4/5) -/
theorem unit_vector_AB (A B : ℝ × ℝ) (h : A = (1, 3) ∧ B = (4, -1)) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let magnitude := Real.sqrt (AB.1^2 + AB.2^2)
  (AB.1 / magnitude, AB.2 / magnitude) = (3/5, -4/5) := by
sorry

end NUMINAMATH_CALUDE_unit_vector_AB_l3762_376206


namespace NUMINAMATH_CALUDE_points_not_on_any_circle_l3762_376227

-- Define the circle equation
def circle_equation (x y α β : ℝ) : Prop :=
  α * ((x - 2)^2 + y^2 - 1) + β * ((x + 2)^2 + y^2 - 1) = 0

-- Define the set of points not on any circle
def points_not_on_circles : Set (ℝ × ℝ) :=
  {p | p.1 = 0 ∨ (p.1 = Real.sqrt 3 ∧ p.2 = 0) ∨ (p.1 = -Real.sqrt 3 ∧ p.2 = 0)}

-- Theorem statement
theorem points_not_on_any_circle :
  ∀ (p : ℝ × ℝ), p ∈ points_not_on_circles →
  ∀ (α β : ℝ), ¬(circle_equation p.1 p.2 α β) :=
by sorry

end NUMINAMATH_CALUDE_points_not_on_any_circle_l3762_376227


namespace NUMINAMATH_CALUDE_max_value_implies_a_l3762_376201

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + a

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2) ∧ 
  (∃ x ∈ Set.Icc (-1) 1, f a x = 2) →
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_max_value_implies_a_l3762_376201


namespace NUMINAMATH_CALUDE_top_z_conference_teams_l3762_376222

theorem top_z_conference_teams (n : ℕ) : n * (n - 1) / 2 = 45 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_top_z_conference_teams_l3762_376222


namespace NUMINAMATH_CALUDE_complex_number_and_imaginary_root_l3762_376220

theorem complex_number_and_imaginary_root (z : ℂ) (m : ℂ) : 
  (∃ (r : ℝ), z + Complex.I = r) →
  (∃ (s : ℝ), z / (1 - Complex.I) = s) →
  (∃ (t : ℝ), m = Complex.I * t) →
  (∃ (x : ℝ), (x^2 : ℂ) + x * (1 + z) - (3 * m - 1) * Complex.I = 0) →
  z = 1 - Complex.I ∧ m = -Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_and_imaginary_root_l3762_376220


namespace NUMINAMATH_CALUDE_ivanna_dorothy_ratio_l3762_376230

/-- Represents the scores of the three students -/
structure Scores where
  tatuya : ℚ
  ivanna : ℚ
  dorothy : ℚ

/-- The conditions of the quiz scores -/
def quiz_conditions (s : Scores) : Prop :=
  s.dorothy = 90 ∧
  (s.tatuya + s.ivanna + s.dorothy) / 3 = 84 ∧
  s.tatuya = 2 * s.ivanna ∧
  ∃ x : ℚ, 0 < x ∧ x < 1 ∧ s.ivanna = x * s.dorothy

/-- The theorem stating the ratio of Ivanna's score to Dorothy's score -/
theorem ivanna_dorothy_ratio (s : Scores) (h : quiz_conditions s) :
  s.ivanna / s.dorothy = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ivanna_dorothy_ratio_l3762_376230


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3762_376272

theorem cylinder_surface_area (V : Real) (d : Real) (h : Real) : 
  V = 500 * Real.pi / 3 →  -- Volume of the sphere
  d = 8 →                  -- Diameter of the cylinder base
  h = 6 →                  -- Height of the cylinder (derived from the problem)
  2 * Real.pi * (d/2) * h + 2 * Real.pi * (d/2)^2 = 80 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3762_376272


namespace NUMINAMATH_CALUDE_circle_intersection_range_l3762_376250

theorem circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - a)^2 = 8 ∧ x^2 + y^2 = 2) ↔ 
  (a < -1 ∧ a > -3) ∨ (a > 1 ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l3762_376250


namespace NUMINAMATH_CALUDE_reverse_sum_divisibility_l3762_376238

def reverse_number (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem reverse_sum_divisibility (n : ℕ) (m : ℕ) (h1 : n ≥ 10^(m-1)) (h2 : n < 10^m) :
  (81 ∣ (n + reverse_number n)) ↔ (81 ∣ sum_of_digits n) := by sorry

end NUMINAMATH_CALUDE_reverse_sum_divisibility_l3762_376238


namespace NUMINAMATH_CALUDE_distance_between_parallel_lines_l3762_376268

/-- Two lines in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Distance between two parallel lines -/
noncomputable def distance (l1 l2 : Line2D) : ℝ :=
  abs (l1.c / l2.a - l2.c / l2.a) / Real.sqrt (l1.a^2 + l1.b^2)

theorem distance_between_parallel_lines :
  let l1 : Line2D := ⟨1, -2, 1⟩
  let l2 : Line2D := ⟨2, a, -2⟩
  parallel l1 l2 → distance l1 l2 = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_parallel_lines_l3762_376268


namespace NUMINAMATH_CALUDE_white_l_shapes_count_l3762_376229

/-- Represents a 5x5 grid with white and non-white squares -/
def Grid := Matrix (Fin 5) (Fin 5) Bool

/-- Represents an "L" shape composed of three squares -/
def LShape := List (Fin 5 × Fin 5)

/-- Returns true if all squares in the L-shape are white -/
def isWhite (g : Grid) (l : LShape) : Bool :=
  l.all (fun (i, j) => g i j)

/-- Returns the number of distinct all-white L-shapes in the grid -/
def countWhiteLShapes (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that there are 24 distinct ways to choose an all-white L-shape -/
theorem white_l_shapes_count (g : Grid) : countWhiteLShapes g = 24 := by
  sorry

end NUMINAMATH_CALUDE_white_l_shapes_count_l3762_376229


namespace NUMINAMATH_CALUDE_lcm_48_180_l3762_376203

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_180_l3762_376203


namespace NUMINAMATH_CALUDE_water_drinkers_l3762_376282

theorem water_drinkers (total : ℕ) (juice_percent : ℚ) (water_percent : ℚ) (juice_count : ℕ) :
  juice_percent = 2/5 →
  water_percent = 3/10 →
  juice_count = 100 →
  ∃ water_count : ℕ, water_count = 75 ∧ (water_count : ℚ) / total = water_percent :=
by sorry

end NUMINAMATH_CALUDE_water_drinkers_l3762_376282


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3762_376202

theorem inequality_solution_set (x : ℝ) : 
  (3 * x - 5 > 11 - 2 * x) ↔ (x > 16 / 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3762_376202


namespace NUMINAMATH_CALUDE_q_is_false_l3762_376210

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q :=
sorry

end NUMINAMATH_CALUDE_q_is_false_l3762_376210


namespace NUMINAMATH_CALUDE_f_negative_iff_x_in_unit_interval_l3762_376283

/-- The function f(x) = x^2 - x^(1/2) is negative if and only if x is in the open interval (0, 1) -/
theorem f_negative_iff_x_in_unit_interval (x : ℝ) :
  x^2 - x^(1/2) < 0 ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_iff_x_in_unit_interval_l3762_376283


namespace NUMINAMATH_CALUDE_book_distribution_ways_l3762_376293

-- Define the number of books and students
def num_books : ℕ := 5
def num_students : ℕ := 4

-- Define a function to calculate the number of ways to distribute books
def distribute_books (books : ℕ) (students : ℕ) : ℕ :=
  -- Implementation details are not provided here
  sorry

-- Theorem statement
theorem book_distribution_ways :
  distribute_books num_books num_students = 240 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_ways_l3762_376293


namespace NUMINAMATH_CALUDE_common_factor_proof_l3762_376221

variables (a b c : ℕ+)

theorem common_factor_proof : Nat.gcd (4 * a^2 * b^2 * c) (6 * a * b^3) = 2 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_common_factor_proof_l3762_376221


namespace NUMINAMATH_CALUDE_function_value_proof_l3762_376215

theorem function_value_proof (f : ℝ → ℝ) (h : ∀ x, f (1 - 2*x) = 1 / x^2) :
  f (1/2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_function_value_proof_l3762_376215


namespace NUMINAMATH_CALUDE_triangle_properties_l3762_376226

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle satisfying certain conditions -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.a + t.c) * Real.cos t.B + t.b * Real.cos t.C = 0)
  (h2 : t.b = Real.sqrt 13)
  (h3 : t.a + t.c = 4) :
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3762_376226


namespace NUMINAMATH_CALUDE_min_parts_for_triangle_flip_l3762_376255

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a part of a triangle that can be flipped -/
structure TrianglePart where
  vertices : List Point

/-- A function that determines if a list of triangle parts can recreate the original triangle when flipped -/
def canReconstructTriangle (t : Triangle) (parts : List TrianglePart) : Prop :=
  sorry

/-- The theorem stating that the minimum number of parts to divide a triangle for flipping reconstruction is 3 -/
theorem min_parts_for_triangle_flip (t : Triangle) :
  ∃ (parts : List TrianglePart),
    parts.length = 3 ∧
    canReconstructTriangle t parts ∧
    ∀ (smaller_parts : List TrianglePart),
      smaller_parts.length < 3 →
      ¬(canReconstructTriangle t smaller_parts) :=
by sorry

end NUMINAMATH_CALUDE_min_parts_for_triangle_flip_l3762_376255


namespace NUMINAMATH_CALUDE_log_sum_problem_l3762_376275

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sum_problem (x y z : ℝ) 
  (hx : log 3 (log 4 (log 5 x)) = 0)
  (hy : log 4 (log 5 (log 3 y)) = 0)
  (hz : log 5 (log 3 (log 4 z)) = 0) :
  x + y + z = 932 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_problem_l3762_376275


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3762_376267

/-- Given a geometric sequence {a_n} with common ratio q and S_n as the sum of the first n terms,
    if a_3 = S_3 + 1, then q = 3 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum formula for geometric sequence
  a 3 = S 3 + 1 →
  q = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3762_376267


namespace NUMINAMATH_CALUDE_programmer_work_hours_l3762_376290

theorem programmer_work_hours (flow_chart_time : ℚ) (coding_time : ℚ) (debug_time : ℚ) 
  (h1 : flow_chart_time = 1/4)
  (h2 : coding_time = 3/8)
  (h3 : debug_time = 1 - (flow_chart_time + coding_time))
  (h4 : debug_time * 48 = 18) :
  48 = 48 := by sorry

end NUMINAMATH_CALUDE_programmer_work_hours_l3762_376290


namespace NUMINAMATH_CALUDE_train_length_calculation_l3762_376252

/-- Calculates the length of a train given the speeds of two trains traveling in opposite directions and the time taken for one train to pass an observer in the other train. -/
theorem train_length_calculation (woman_speed goods_speed : ℝ) (passing_time : ℝ) 
  (woman_speed_pos : 0 < woman_speed)
  (goods_speed_pos : 0 < goods_speed)
  (passing_time_pos : 0 < passing_time)
  (h_woman_speed : woman_speed = 25)
  (h_goods_speed : goods_speed = 142.986561075114)
  (h_passing_time : passing_time = 3) :
  ∃ (train_length : ℝ), abs (train_length - 38.932) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3762_376252


namespace NUMINAMATH_CALUDE_complex_fraction_product_l3762_376269

theorem complex_fraction_product (a b : ℝ) :
  (1 + 7 * Complex.I) / (2 - Complex.I) = Complex.mk a b →
  a * b = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_product_l3762_376269


namespace NUMINAMATH_CALUDE_solution_set_characterization_l3762_376224

-- Define the properties of function f
def IsOddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsDecreasingFunction (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

-- Define the solution set
def SolutionSet (f : ℝ → ℝ) : Set ℝ := {a | f (a^2) + f (2*a) > 0}

-- State the theorem
theorem solution_set_characterization 
  (f : ℝ → ℝ) 
  (h_odd : IsOddFunction f) 
  (h_decreasing : IsDecreasingFunction f) : 
  SolutionSet f = Set.Ioo (-2) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l3762_376224


namespace NUMINAMATH_CALUDE_seven_boys_without_calculators_l3762_376278

/-- Represents the number of boys who didn't bring calculators to Mrs. Luna's math class -/
def boys_without_calculators (total_boys : ℕ) (total_with_calculators : ℕ) (girls_with_calculators : ℕ) : ℕ :=
  total_boys - (total_with_calculators - girls_with_calculators)

/-- Theorem stating that 7 boys didn't bring their calculators to Mrs. Luna's math class -/
theorem seven_boys_without_calculators :
  boys_without_calculators 20 28 15 = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_boys_without_calculators_l3762_376278


namespace NUMINAMATH_CALUDE_bridgette_dog_baths_l3762_376244

/-- The number of times Bridgette bathes her dogs each month -/
def dog_baths_per_month : ℕ := sorry

/-- The number of dogs Bridgette has -/
def num_dogs : ℕ := 2

/-- The number of cats Bridgette has -/
def num_cats : ℕ := 3

/-- The number of birds Bridgette has -/
def num_birds : ℕ := 4

/-- The number of times Bridgette bathes her cats each month -/
def cat_baths_per_month : ℕ := 1

/-- The number of times Bridgette bathes her birds each month -/
def bird_baths_per_month : ℚ := 1/4

/-- The total number of baths Bridgette gives in a year -/
def total_baths_per_year : ℕ := 96

/-- The number of months in a year -/
def months_per_year : ℕ := 12

theorem bridgette_dog_baths : 
  dog_baths_per_month = 2 :=
by sorry

end NUMINAMATH_CALUDE_bridgette_dog_baths_l3762_376244


namespace NUMINAMATH_CALUDE_square_area_ratio_l3762_376212

theorem square_area_ratio (side_C side_D : ℝ) (h1 : side_C = 48) (h2 : side_D = 60) :
  (side_C ^ 2) / (side_D ^ 2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3762_376212


namespace NUMINAMATH_CALUDE_circles_intersect_l3762_376257

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle_O2 (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9

-- Define the centers and radii
def center_O1 : ℝ × ℝ := (1, 0)
def center_O2 : ℝ × ℝ := (0, 3)
def radius_O1 : ℝ := 1
def radius_O2 : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect : 
  let d := Real.sqrt ((center_O1.1 - center_O2.1)^2 + (center_O1.2 - center_O2.2)^2)
  (radius_O2 - radius_O1 < d) ∧ (d < radius_O1 + radius_O2) := by
  sorry


end NUMINAMATH_CALUDE_circles_intersect_l3762_376257


namespace NUMINAMATH_CALUDE_trivia_game_points_per_question_l3762_376234

theorem trivia_game_points_per_question 
  (first_half_correct : ℕ) 
  (second_half_correct : ℕ) 
  (final_score : ℕ) 
  (h1 : first_half_correct = 6)
  (h2 : second_half_correct = 4)
  (h3 : final_score = 30) :
  final_score / (first_half_correct + second_half_correct) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_points_per_question_l3762_376234


namespace NUMINAMATH_CALUDE_incorrect_expression_l3762_376254

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 6) :
  ¬((3 * x + 3 * y) / x = 18 / 5) := by
sorry

end NUMINAMATH_CALUDE_incorrect_expression_l3762_376254


namespace NUMINAMATH_CALUDE_equal_squares_from_sum_product_l3762_376233

theorem equal_squares_from_sum_product (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h : a * (b + c + d) = b * (a + c + d) ∧ 
       b * (a + c + d) = c * (a + b + d) ∧ 
       c * (a + b + d) = d * (a + b + c)) : 
  a^2 = b^2 ∧ b^2 = c^2 ∧ c^2 = d^2 := by
sorry

end NUMINAMATH_CALUDE_equal_squares_from_sum_product_l3762_376233


namespace NUMINAMATH_CALUDE_matrix_product_equality_l3762_376216

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -3; 1, 2]
def B : Matrix (Fin 2) (Fin 1) ℝ := !![4; -6]
def result : Matrix (Fin 2) (Fin 1) ℝ := !![26; -8]

theorem matrix_product_equality : A * B = result := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l3762_376216


namespace NUMINAMATH_CALUDE_history_class_grades_l3762_376286

theorem history_class_grades (total_students : ℕ) 
  (prob_A : ℚ) (prob_B : ℚ) (prob_C : ℚ) :
  total_students = 31 →
  prob_A = 0.7 * prob_B →
  prob_C = 1.4 * prob_B →
  prob_A + prob_B + prob_C = 1 →
  (total_students : ℚ) * prob_B = 10 := by
  sorry

end NUMINAMATH_CALUDE_history_class_grades_l3762_376286


namespace NUMINAMATH_CALUDE_circle_C_properties_l3762_376266

-- Define the circles and points
def circle_M (r : ℝ) (x y : ℝ) : Prop := (x + 2)^2 + (y + 2)^2 = r^2
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2
def point_P : ℝ × ℝ := (1, 1)
def point_Q (x y : ℝ) : Prop := circle_C x y

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the vector dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Define the theorem
theorem circle_C_properties
  (r : ℝ)
  (h_r : r > 0)
  (h_symmetry : ∀ x y, circle_C x y ↔ 
    ∃ x' y', circle_M r x' y' ∧ symmetry_line ((x + x')/2) ((y + y')/2))
  (h_P_on_C : circle_C point_P.1 point_P.2)
  (h_complementary_slopes : ∀ A B : ℝ × ℝ, 
    circle_C A.1 A.2 → circle_C B.1 B.2 → 
    (A.2 - point_P.2) * (B.2 - point_P.2) = -(A.1 - point_P.1) * (B.1 - point_P.1)) :
  (∀ x y, circle_C x y ↔ x^2 + y^2 = 2) ∧
  (∀ Q : ℝ × ℝ, point_Q Q.1 Q.2 → 
    dot_product (Q.1 - point_P.1, Q.2 - point_P.2) (Q.1 + 2, Q.2 + 2) ≥ -4) ∧
  (∀ A B : ℝ × ℝ, circle_C A.1 A.2 → circle_C B.1 B.2 → A ≠ B →
    (A.2 - B.2) * point_P.1 = (A.1 - B.1) * point_P.2) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l3762_376266


namespace NUMINAMATH_CALUDE_gathering_drinks_l3762_376211

/-- Represents the number of people who took both wine and soda at a gathering -/
def people_took_both (total : ℕ) (wine : ℕ) (soda : ℕ) : ℕ :=
  wine + soda - total

theorem gathering_drinks (total : ℕ) (wine : ℕ) (soda : ℕ) 
  (h_total : total = 31) 
  (h_wine : wine = 26) 
  (h_soda : soda = 22) :
  people_took_both total wine soda = 17 := by
  sorry

#eval people_took_both 31 26 22

end NUMINAMATH_CALUDE_gathering_drinks_l3762_376211


namespace NUMINAMATH_CALUDE_bank_teller_problem_l3762_376213

theorem bank_teller_problem (total_bills : ℕ) (total_value : ℕ) 
  (h1 : total_bills = 54)
  (h2 : total_value = 780) :
  ∃ (five_dollar_bills twenty_dollar_bills : ℕ),
    five_dollar_bills + twenty_dollar_bills = total_bills ∧
    5 * five_dollar_bills + 20 * twenty_dollar_bills = total_value ∧
    five_dollar_bills = 20 := by
  sorry

end NUMINAMATH_CALUDE_bank_teller_problem_l3762_376213


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3762_376260

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3762_376260


namespace NUMINAMATH_CALUDE_polynomial_inverse_property_l3762_376289

-- Define the polynomials p and P
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def P (A B C : ℝ) (x : ℝ) : ℝ := A * x^2 + B * x + C

-- State the theorem
theorem polynomial_inverse_property 
  (a b c A B C : ℝ) : 
  (∀ x : ℝ, P A B C (p a b c x) = x) → 
  (∀ x : ℝ, p a b c (P A B C x) = x) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_inverse_property_l3762_376289


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l3762_376239

theorem last_two_digits_sum (n : ℕ) : (7^30 + 13^30) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l3762_376239


namespace NUMINAMATH_CALUDE_largest_b_value_l3762_376277

theorem largest_b_value (b : ℝ) (h : (3*b + 4)*(b - 2) = 7*b) :
  b ≤ (9 + Real.sqrt 177) / 6 ∧
  ∃ (b : ℝ), (3*b + 4)*(b - 2) = 7*b ∧ b = (9 + Real.sqrt 177) / 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_b_value_l3762_376277


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3762_376245

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x - 1) - 5 / (x - 2) + 5 / (x - 3) - 2 / (x - 4) < 1 / 15) ↔ 
  (x < -3/2 ∨ (-1 < x ∧ x < 1) ∨ (2 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3762_376245


namespace NUMINAMATH_CALUDE_grunters_win_probability_l3762_376208

theorem grunters_win_probability (n : ℕ) (p : ℚ) (h1 : n = 6) (h2 : p = 3/5) :
  p ^ n = 729 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l3762_376208


namespace NUMINAMATH_CALUDE_range_of_a_l3762_376295

theorem range_of_a (p q : Prop) (hp : ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0)
  (hq : ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) (hpq : p ∧ q) :
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3762_376295


namespace NUMINAMATH_CALUDE_cat_kibble_problem_l3762_376219

/-- Represents the amount of kibble eaten by a cat in a given time -/
def kibble_eaten (eating_rate : ℚ) (time : ℚ) : ℚ :=
  (time / 4) * eating_rate

/-- Represents the amount of kibble left in the bowl after some time -/
def kibble_left (initial_amount : ℚ) (eating_rate : ℚ) (time : ℚ) : ℚ :=
  initial_amount - kibble_eaten eating_rate time

theorem cat_kibble_problem :
  let initial_amount : ℚ := 3
  let eating_rate : ℚ := 1
  let time : ℚ := 8
  kibble_left initial_amount eating_rate time = 1 := by sorry

end NUMINAMATH_CALUDE_cat_kibble_problem_l3762_376219


namespace NUMINAMATH_CALUDE_max_value_of_f_l3762_376214

theorem max_value_of_f (x y z u v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0) (hv : v > 0) :
  (x * y + y * z + z * u + u * v) / (2 * x^2 + y^2 + 2 * z^2 + u^2 + 2 * v^2) ≤ Real.sqrt 6 / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3762_376214


namespace NUMINAMATH_CALUDE_total_distance_is_12_17_l3762_376235

def walking_time : ℚ := 30 / 60
def walking_rate : ℚ := 3
def running_time : ℚ := 20 / 60
def running_rate : ℚ := 8
def cycling_time : ℚ := 40 / 60
def cycling_rate : ℚ := 12

def total_distance : ℚ :=
  walking_time * walking_rate +
  running_time * running_rate +
  cycling_time * cycling_rate

theorem total_distance_is_12_17 :
  total_distance = 12.17 := by sorry

end NUMINAMATH_CALUDE_total_distance_is_12_17_l3762_376235


namespace NUMINAMATH_CALUDE_number_of_boys_l3762_376204

theorem number_of_boys (total_amount : ℕ) (total_people : ℕ) (boy_amount : ℕ) (girl_amount : ℕ) 
  (h1 : total_amount = 460)
  (h2 : total_people = 41)
  (h3 : boy_amount = 12)
  (h4 : girl_amount = 8) :
  ∃ (boys : ℕ), boys = 33 ∧ 
  boys * boy_amount + (total_people - boys) * girl_amount = total_amount :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l3762_376204


namespace NUMINAMATH_CALUDE_photos_to_cover_poster_l3762_376251

def poster_length : ℕ := 3
def poster_width : ℕ := 5
def photo_length : ℕ := 3
def photo_width : ℕ := 5
def inches_per_foot : ℕ := 12

theorem photos_to_cover_poster :
  (poster_length * inches_per_foot * poster_width * inches_per_foot) / (photo_length * photo_width) = 144 := by
  sorry

end NUMINAMATH_CALUDE_photos_to_cover_poster_l3762_376251


namespace NUMINAMATH_CALUDE_fraction_value_l3762_376223

theorem fraction_value (x y : ℝ) (hx : x = 4) (hy : y = -3) :
  (x - 2*y) / (x + y) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3762_376223


namespace NUMINAMATH_CALUDE_basketball_free_throws_l3762_376298

theorem basketball_free_throws (total_players : Nat) (captains : Nat) 
  (h1 : total_players = 15)
  (h2 : captains = 2)
  (h3 : captains ≤ total_players) :
  (total_players - 1) * captains = 28 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l3762_376298


namespace NUMINAMATH_CALUDE_sin_cos_equation_solutions_l3762_376241

theorem sin_cos_equation_solutions (x : ℝ) :
  (0 ≤ x ∧ x < 2 * Real.pi) ∧ (Real.sin x - Real.cos x = Real.sqrt 3 / 2) ↔
  (x = Real.arcsin (Real.sqrt 6 / 4) - Real.pi / 4 ∨
   x = Real.pi - Real.arcsin (Real.sqrt 6 / 4) - Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solutions_l3762_376241


namespace NUMINAMATH_CALUDE_equation_solutions_l3762_376205

theorem equation_solutions :
  (∃ x : ℝ, (x + 1)^3 = 64 ∧ x = 3) ∧
  (∃ x : ℝ, (2*x + 1)^2 = 81 ∧ (x = 4 ∨ x = -5)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3762_376205


namespace NUMINAMATH_CALUDE_tomato_price_proof_l3762_376209

/-- The original price per pound of tomatoes -/
def original_price : ℝ := 0.80

/-- The proportion of tomatoes remaining after discarding ruined ones -/
def remaining_proportion : ℝ := 0.90

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.12

/-- The selling price per pound of remaining tomatoes -/
def selling_price : ℝ := 0.9956

theorem tomato_price_proof :
  selling_price * remaining_proportion = original_price * (1 + profit_percentage) := by
  sorry


end NUMINAMATH_CALUDE_tomato_price_proof_l3762_376209


namespace NUMINAMATH_CALUDE_eight_people_line_up_with_pair_l3762_376271

/-- The number of ways to arrange n people in a line. -/
def linearArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line, 
    with 2 specific people always standing together. -/
def arrangementsWithPair (n : ℕ) : ℕ :=
  2 * linearArrangements (n - 1)

/-- Theorem: There are 10080 ways for 8 people to line up
    with 2 specific people always standing together. -/
theorem eight_people_line_up_with_pair : 
  arrangementsWithPair 8 = 10080 := by
  sorry


end NUMINAMATH_CALUDE_eight_people_line_up_with_pair_l3762_376271


namespace NUMINAMATH_CALUDE_cubic_roots_difference_l3762_376200

-- Define the cubic polynomial
def cubic_poly (x : ℝ) : ℝ := 27 * x^3 - 81 * x^2 + 63 * x - 14

-- Define a predicate for roots in geometric progression
def roots_in_geometric_progression (r₁ r₂ r₃ : ℝ) : Prop :=
  ∃ (a r : ℝ), r₁ = a ∧ r₂ = a * r ∧ r₃ = a * r^2

-- Theorem statement
theorem cubic_roots_difference (r₁ r₂ r₃ : ℝ) :
  cubic_poly r₁ = 0 ∧ cubic_poly r₂ = 0 ∧ cubic_poly r₃ = 0 →
  roots_in_geometric_progression r₁ r₂ r₃ →
  (max r₁ (max r₂ r₃))^2 - (min r₁ (min r₂ r₃))^2 = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_difference_l3762_376200


namespace NUMINAMATH_CALUDE_karinas_brother_birth_year_l3762_376281

/-- Proves the birth year of Karina's brother given the conditions of the problem -/
theorem karinas_brother_birth_year 
  (karina_birth_year : ℕ) 
  (karina_current_age : ℕ) 
  (h1 : karina_birth_year = 1970)
  (h2 : karina_current_age = 40)
  (h3 : karina_current_age = 2 * (karina_current_age - (karina_birth_year - brother_birth_year)))
  : brother_birth_year = 1990 := by
  sorry


end NUMINAMATH_CALUDE_karinas_brother_birth_year_l3762_376281


namespace NUMINAMATH_CALUDE_min_transportation_cost_l3762_376207

/-- Transportation problem between two cities and two towns -/
structure TransportationProblem where
  cityA_goods : ℕ
  cityB_goods : ℕ
  townA_needs : ℕ
  townB_needs : ℕ
  costA_to_A : ℕ
  costA_to_B : ℕ
  costB_to_A : ℕ
  costB_to_B : ℕ

/-- Define the specific problem instance -/
def problem : TransportationProblem := {
  cityA_goods := 120
  cityB_goods := 130
  townA_needs := 140
  townB_needs := 110
  costA_to_A := 300
  costA_to_B := 150
  costB_to_A := 200
  costB_to_B := 100
}

/-- Total transportation cost function -/
def total_cost (p : TransportationProblem) (x : ℕ) : ℕ :=
  p.costA_to_A * x + p.costA_to_B * (p.cityA_goods - x) +
  p.costB_to_A * (p.townA_needs - x) + p.costB_to_B * (p.townB_needs - p.cityA_goods + x)

/-- Theorem: The minimum total transportation cost is 45500 yuan -/
theorem min_transportation_cost :
  ∃ x, x ≥ 10 ∧ x ≤ 120 ∧
  (∀ y, y ≥ 10 → y ≤ 120 → total_cost problem x ≤ total_cost problem y) ∧
  total_cost problem x = 45500 :=
sorry

end NUMINAMATH_CALUDE_min_transportation_cost_l3762_376207


namespace NUMINAMATH_CALUDE_return_journey_time_l3762_376217

/-- Proves that given a round trip with specified conditions, the return journey takes 7 hours -/
theorem return_journey_time 
  (total_distance : ℝ) 
  (outbound_time : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_distance = 2000) 
  (h2 : outbound_time = 10) 
  (h3 : average_speed = 142.85714285714286) : 
  (total_distance / average_speed) - outbound_time = 7 := by
  sorry

end NUMINAMATH_CALUDE_return_journey_time_l3762_376217


namespace NUMINAMATH_CALUDE_no_nontrivial_solutions_l3762_376258

theorem no_nontrivial_solutions (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1)
  (h2p1 : Nat.Prime (2 * p + 1)) :
  ∀ x y z : ℤ, x^p + 2*y^p + 5*z^p = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nontrivial_solutions_l3762_376258


namespace NUMINAMATH_CALUDE_circle_assignment_exists_l3762_376243

structure Circle where
  value : ℕ

structure Graph where
  A : Circle
  B : Circle
  C : Circle
  D : Circle

def connected (x y : Circle) (g : Graph) : Prop :=
  (x = g.A ∧ y = g.B) ∨ (x = g.B ∧ y = g.A) ∨
  (x = g.A ∧ y = g.D) ∨ (x = g.D ∧ y = g.A) ∨
  (x = g.B ∧ y = g.C) ∨ (x = g.C ∧ y = g.B)

def ratio (x y : Circle) : ℚ :=
  (x.value : ℚ) / (y.value : ℚ)

theorem circle_assignment_exists : ∃ g : Graph,
  (∀ x y : Circle, connected x y g → (ratio x y = 3 ∨ ratio x y = 9)) ∧
  (∀ x y : Circle, ¬connected x y g → (ratio x y ≠ 3 ∧ ratio x y ≠ 9)) :=
sorry

end NUMINAMATH_CALUDE_circle_assignment_exists_l3762_376243


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_ellipse_parametric_to_cartesian_line_l3762_376297

-- Equation 1
theorem parametric_to_cartesian_ellipse (x y φ : ℝ) :
  x = 5 * Real.cos φ ∧ y = 4 * Real.sin φ ↔ x^2 / 25 + y^2 / 16 = 1 :=
sorry

-- Equation 2
theorem parametric_to_cartesian_line (x y t : ℝ) :
  x = 1 - 3 * t^2 ∧ y = 4 * t^2 ↔ 4 * x + 3 * y - 4 = 0 ∧ x ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_ellipse_parametric_to_cartesian_line_l3762_376297


namespace NUMINAMATH_CALUDE_total_revenue_is_3610_l3762_376287

/-- Represents the quantity and price information for a fruit --/
structure Fruit where
  quantity : ℕ
  originalPrice : ℚ
  priceChange : ℚ

/-- Calculates the total revenue for a single fruit type --/
def calculateFruitRevenue (fruit : Fruit) : ℚ :=
  fruit.quantity * (fruit.originalPrice * (1 + fruit.priceChange))

/-- Theorem stating that the total revenue from all fruits is $3610 --/
theorem total_revenue_is_3610 
  (lemons : Fruit)
  (grapes : Fruit)
  (oranges : Fruit)
  (apples : Fruit)
  (kiwis : Fruit)
  (pineapples : Fruit)
  (h1 : lemons = { quantity := 80, originalPrice := 8, priceChange := 0.5 })
  (h2 : grapes = { quantity := 140, originalPrice := 7, priceChange := 0.25 })
  (h3 : oranges = { quantity := 60, originalPrice := 5, priceChange := 0.1 })
  (h4 : apples = { quantity := 100, originalPrice := 4, priceChange := 0.2 })
  (h5 : kiwis = { quantity := 50, originalPrice := 6, priceChange := -0.15 })
  (h6 : pineapples = { quantity := 30, originalPrice := 12, priceChange := 0 }) :
  calculateFruitRevenue lemons + calculateFruitRevenue grapes + 
  calculateFruitRevenue oranges + calculateFruitRevenue apples + 
  calculateFruitRevenue kiwis + calculateFruitRevenue pineapples = 3610 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_3610_l3762_376287


namespace NUMINAMATH_CALUDE_coin_exchange_problem_l3762_376280

theorem coin_exchange_problem :
  ∃! (one_cent two_cent five_cent ten_cent : ℕ),
    two_cent = (3 * one_cent) / 5 ∧
    five_cent = (3 * two_cent) / 5 ∧
    ten_cent = (3 * five_cent) / 5 - 7 ∧
    50 < (one_cent + 2 * two_cent + 5 * five_cent + 10 * ten_cent) / 100 ∧
    (one_cent + 2 * two_cent + 5 * five_cent + 10 * ten_cent) / 100 < 100 ∧
    one_cent = 1375 ∧
    two_cent = 825 ∧
    five_cent = 495 ∧
    ten_cent = 290 := by
  sorry

end NUMINAMATH_CALUDE_coin_exchange_problem_l3762_376280


namespace NUMINAMATH_CALUDE_student_selection_l3762_376256

/-- The number of ways to select 3 students from a group of 4 boys and 3 girls, 
    including both boys and girls. -/
theorem student_selection (boys : Nat) (girls : Nat) : 
  boys = 4 → girls = 3 → Nat.choose boys 2 * Nat.choose girls 1 + 
                         Nat.choose boys 1 * Nat.choose girls 2 = 30 := by
  sorry

#eval Nat.choose 4 2 * Nat.choose 3 1 + Nat.choose 4 1 * Nat.choose 3 2

end NUMINAMATH_CALUDE_student_selection_l3762_376256


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l3762_376247

/-- Represents a pentagon with angles P, Q, R, S, and T -/
structure Pentagon where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ

/-- The sum of angles in a pentagon is 540° -/
axiom pentagon_angle_sum (p : Pentagon) : p.P + p.Q + p.R + p.S + p.T = 540

/-- Theorem: In a pentagon PQRST where P = 70°, Q = 110°, R = S, and T = 3R + 20°,
    the measure of the largest angle is 224° -/
theorem largest_angle_in_pentagon (p : Pentagon)
  (h1 : p.P = 70)
  (h2 : p.Q = 110)
  (h3 : p.R = p.S)
  (h4 : p.T = 3 * p.R + 20) :
  max p.P (max p.Q (max p.R (max p.S p.T))) = 224 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l3762_376247


namespace NUMINAMATH_CALUDE_fraction_comparison_l3762_376225

theorem fraction_comparison (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a / b > (a + 1) / (b + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3762_376225


namespace NUMINAMATH_CALUDE_power_function_m_value_l3762_376285

/-- A function of the form y = ax^n where a and n are constants and a ≠ 0 -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- The main theorem: if y = (2m-1)x^(m^2) is a power function, then m = 1 -/
theorem power_function_m_value (m : ℝ) :
  isPowerFunction (fun x => (2*m - 1) * x^(m^2)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_m_value_l3762_376285


namespace NUMINAMATH_CALUDE_range_of_m_l3762_376276

/-- Proposition p: x < -2 or x > 10 -/
def p (x : ℝ) : Prop := x < -2 ∨ x > 10

/-- Proposition q: 1-m ≤ x ≤ 1+m^2 -/
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m^2

/-- ¬p is a sufficient but not necessary condition for q -/
def suff_not_nec (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → q x m) ∧ ∃ x, q x m ∧ p x

theorem range_of_m :
  {m : ℝ | suff_not_nec m} = {m : ℝ | m ≥ 3} :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3762_376276


namespace NUMINAMATH_CALUDE_inequality_linear_iff_k_eq_two_l3762_376273

/-- The inequality (k+2)x^(|k|-1) + 5 < 0 is linear in x if and only if k = 2 -/
theorem inequality_linear_iff_k_eq_two (k : ℝ) : 
  (∃ (a b : ℝ), ∀ x, ((k + 2) * x^(|k| - 1) + 5 < 0) ↔ (a * x + b < 0)) ↔ k = 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_linear_iff_k_eq_two_l3762_376273


namespace NUMINAMATH_CALUDE_symmetry_problem_l3762_376265

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_wrt_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

/-- Given that z₁ = 1 - 2i and z₁ and z₂ are symmetric with respect to the imaginary axis,
    prove that z₂ = -1 - 2i. -/
theorem symmetry_problem (z₁ z₂ : ℂ) 
    (h₁ : z₁ = 1 - 2*I) 
    (h₂ : symmetric_wrt_imaginary_axis z₁ z₂) : 
  z₂ = -1 - 2*I :=
sorry

end NUMINAMATH_CALUDE_symmetry_problem_l3762_376265


namespace NUMINAMATH_CALUDE_percent_less_problem_l3762_376299

theorem percent_less_problem (w x y z : ℝ) : 
  x = y * (1 - z / 100) →
  y = 1.4 * w →
  x = 5 * w / 4 →
  z = 10.71 := by
sorry

end NUMINAMATH_CALUDE_percent_less_problem_l3762_376299


namespace NUMINAMATH_CALUDE_carrots_taken_l3762_376284

theorem carrots_taken (initial_carrots remaining_carrots : ℕ) :
  initial_carrots = 6 →
  remaining_carrots = 3 →
  initial_carrots - remaining_carrots = 3 :=
by sorry

end NUMINAMATH_CALUDE_carrots_taken_l3762_376284


namespace NUMINAMATH_CALUDE_quadratic_monotonic_condition_l3762_376264

/-- A quadratic function f(x) = x^2 - 2mx + 3 is monotonic on [2, 3] if and only if m ∈ (-∞, 2] ∪ [3, +∞) -/
theorem quadratic_monotonic_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 2 3, Monotone (fun x => x^2 - 2*m*x + 3)) ↔ 
  (m ≤ 2 ∨ m ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotonic_condition_l3762_376264


namespace NUMINAMATH_CALUDE_court_case_fraction_l3762_376291

theorem court_case_fraction (total : ℕ) (dismissed : ℕ) (delayed : ℕ) (guilty : ℕ) 
  (h_total : total = 17)
  (h_dismissed : dismissed = 2)
  (h_delayed : delayed = 1)
  (h_guilty : guilty = 4)
  : (total - dismissed - delayed - guilty : ℚ) / (total - dismissed : ℚ) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_court_case_fraction_l3762_376291


namespace NUMINAMATH_CALUDE_new_rectangle_area_l3762_376261

def rectangle_area_increase (original_area : ℝ) (length_increase : ℝ) (width_increase : ℝ) : ℝ :=
  original_area * (1 + length_increase) * (1 + width_increase)

theorem new_rectangle_area :
  let original_area : ℝ := 576
  let length_increase : ℝ := 0.2
  let width_increase : ℝ := 0.05
  let new_area := rectangle_area_increase original_area length_increase width_increase
  Int.floor (new_area + 0.5) = 726 := by
  sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l3762_376261
