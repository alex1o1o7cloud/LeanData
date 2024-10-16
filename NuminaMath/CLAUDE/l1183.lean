import Mathlib

namespace NUMINAMATH_CALUDE_transformation_matrix_exists_and_unique_l1183_118376

open Matrix

theorem transformation_matrix_exists_and_unique :
  ∃! N : Matrix (Fin 2) (Fin 2) ℝ, 
    ∀ A : Matrix (Fin 2) (Fin 2) ℝ, 
      N * A = !![4 * A 0 0, 4 * A 0 1; A 1 0, A 1 1] := by
  sorry

end NUMINAMATH_CALUDE_transformation_matrix_exists_and_unique_l1183_118376


namespace NUMINAMATH_CALUDE_min_y_difference_parabola_l1183_118377

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  vertex : ℝ × ℝ

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4 * p.a * (x - p.vertex.1)

/-- Line passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: Minimum value of |y₁ - 4y₂| for points on parabola y² = 4x -/
theorem min_y_difference_parabola (p : Parabola) 
  (h_p : p.a = 1 ∧ p.vertex = (0, 0))
  (l : Line) 
  (h_l : l.a * 1 + l.b * 0 + l.c = 0)  -- Line passes through focus (1, 0)
  (A B : ParabolaPoint p)
  (h_A : A.x ≥ 0 ∧ A.y ≥ 0)  -- A is in the first quadrant
  (h_line : l.a * A.x + l.b * A.y + l.c = 0 ∧ 
            l.a * B.x + l.b * B.y + l.c = 0)  -- A and B are on the line
  : ∃ (y₁ y₂ : ℝ), |y₁ - 4*y₂| ≥ 8 ∧ 
    (∃ (A' B' : ParabolaPoint p), 
      |A'.y - 4*B'.y| = 8 ∧ 
      l.a * A'.x + l.b * A'.y + l.c = 0 ∧ 
      l.a * B'.x + l.b * B'.y + l.c = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_y_difference_parabola_l1183_118377


namespace NUMINAMATH_CALUDE_system_solution_existence_l1183_118390

/-- The system of equations has at least one solution for some 'a' if and only if 'b' is in [-11, 2) -/
theorem system_solution_existence (b : ℝ) : 
  (∃ a x y : ℝ, x^2 + y^2 + 2*b*(b - x + y) = 4 ∧ y = 9 / ((x + a)^2 + 1)) ↔ 
  -11 ≤ b ∧ b < 2 := by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l1183_118390


namespace NUMINAMATH_CALUDE_star_equation_solution_l1183_118332

def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

theorem star_equation_solution :
  ∀ x : ℝ, star 3 x = 15 → x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1183_118332


namespace NUMINAMATH_CALUDE_vegetable_ghee_weight_l1183_118384

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 395

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 950

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def mixture_ratio : ℚ := 3 / 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3640

theorem vegetable_ghee_weight : 
  weight_a * (mixture_ratio * total_volume / (1 + mixture_ratio)) + 
  weight_b * (total_volume / (1 + mixture_ratio)) = total_weight := by
  sorry

#check vegetable_ghee_weight

end NUMINAMATH_CALUDE_vegetable_ghee_weight_l1183_118384


namespace NUMINAMATH_CALUDE_max_visible_cubes_l1183_118317

/-- Represents a transparent cube made of unit cubes --/
structure TransparentCube where
  size : Nat
  deriving Repr

/-- Calculates the number of visible unit cubes from a single point --/
def visibleUnitCubes (cube : TransparentCube) : Nat :=
  let fullFace := cube.size * cube.size
  let surfaceFaces := 2 * (cube.size * cube.size - (cube.size - 2) * (cube.size - 2))
  let sharedEdges := 3 * cube.size
  fullFace + surfaceFaces - sharedEdges + 1

/-- Theorem stating that the maximum number of visible unit cubes is 181 for a 12x12x12 cube --/
theorem max_visible_cubes (cube : TransparentCube) (h : cube.size = 12) :
  visibleUnitCubes cube = 181 := by
  sorry

#eval visibleUnitCubes { size := 12 }

end NUMINAMATH_CALUDE_max_visible_cubes_l1183_118317


namespace NUMINAMATH_CALUDE_julia_tag_game_l1183_118365

theorem julia_tag_game (monday_kids tuesday_kids : ℕ) : 
  monday_kids = 22 → 
  monday_kids = tuesday_kids + 8 → 
  tuesday_kids = 14 := by
sorry

end NUMINAMATH_CALUDE_julia_tag_game_l1183_118365


namespace NUMINAMATH_CALUDE_eddys_spider_plant_babies_l1183_118369

/-- A spider plant that produces baby plants -/
structure SpiderPlant where
  /-- Number of baby plants produced per cycle -/
  babies_per_cycle : ℕ
  /-- Number of cycles per year -/
  cycles_per_year : ℕ

/-- Calculate the total number of baby plants produced over a given number of years -/
def total_babies (plant : SpiderPlant) (years : ℕ) : ℕ :=
  plant.babies_per_cycle * plant.cycles_per_year * years

/-- Theorem: Eddy's spider plant produces 16 baby plants after 4 years -/
theorem eddys_spider_plant_babies :
  ∃ (plant : SpiderPlant), plant.babies_per_cycle = 2 ∧ plant.cycles_per_year = 2 ∧ total_babies plant 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_eddys_spider_plant_babies_l1183_118369


namespace NUMINAMATH_CALUDE_intersection_distance_to_pole_l1183_118319

-- Define the polar coordinate system
def PolarCoordinate := ℝ × ℝ

-- Define the distance function in polar coordinates
def distance_to_pole (p : PolarCoordinate) : ℝ := p.1

-- Define the curves
def curve1 (ρ θ : ℝ) : Prop := ρ = 2 * θ + 1
def curve2 (ρ θ : ℝ) : Prop := ρ * θ = 1

theorem intersection_distance_to_pole :
  ∀ (p : PolarCoordinate),
    p.1 > 0 →
    curve1 p.1 p.2 →
    curve2 p.1 p.2 →
    distance_to_pole p = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_to_pole_l1183_118319


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l1183_118305

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 6) 
  (h2 : sum_first_two = 9/2) : 
  ∃ a : ℝ, (a = 9 ∨ a = 3) ∧ 
  ∃ r : ℝ, (r = 1/2 ∨ r = -1/2) ∧ 
  S = a / (1 - r) ∧ 
  sum_first_two = a + a * r :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l1183_118305


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_l1183_118321

/-- A quadratic function f(x) = ax² + bx + c has exactly two distinct real zeros when a·c < 0 -/
theorem quadratic_two_zeros (a b c : ℝ) (h : a * c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_l1183_118321


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1183_118335

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ (r₁ r₂ : ℝ), r₁ + r₂ = -p ∧ r₁ * r₂ = m ∧
   (3 * r₁) + (3 * r₂) = -m ∧ (3 * r₁) * (3 * r₂) = n) →
  n / p = -9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1183_118335


namespace NUMINAMATH_CALUDE_estate_area_calculation_l1183_118360

/-- Represents the scale of the map in miles per inch -/
def scale : ℝ := 300

/-- Represents the length of the estate on the map in inches -/
def map_length : ℝ := 6

/-- Represents the width of the estate on the map in inches -/
def map_width : ℝ := 4

/-- Calculates the actual length of the estate in miles -/
def actual_length : ℝ := scale * map_length

/-- Calculates the actual width of the estate in miles -/
def actual_width : ℝ := scale * map_width

/-- Calculates the area of the estate in square miles -/
def estate_area : ℝ := actual_length * actual_width

theorem estate_area_calculation : estate_area = 2160000 := by
  sorry

end NUMINAMATH_CALUDE_estate_area_calculation_l1183_118360


namespace NUMINAMATH_CALUDE_equidistant_circles_count_l1183_118325

-- Define a point in a 2D plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle in a 2D plane
structure Circle2D where
  center : Point2D
  radius : ℝ

-- Function to check if a circle is equidistant from a set of points
def isEquidistant (c : Circle2D) (points : List Point2D) : Prop :=
  ∀ p ∈ points, (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Theorem statement
theorem equidistant_circles_count 
  (A B C D : Point2D) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  ∃! (circles : List Circle2D), 
    circles.length = 7 ∧ 
    ∀ c ∈ circles, isEquidistant c [A, B, C, D] :=
sorry

end NUMINAMATH_CALUDE_equidistant_circles_count_l1183_118325


namespace NUMINAMATH_CALUDE_certain_number_proof_l1183_118329

theorem certain_number_proof (y : ℝ) : 
  (0.25 * 660 = 0.12 * y - 15) → y = 1500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1183_118329


namespace NUMINAMATH_CALUDE_min_value_expression_l1183_118385

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1183_118385


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1183_118373

/-- The line equation ax + (2a-1)y + a-3 = 0 passes through the point (5, -3) for all values of a. -/
theorem fixed_point_on_line (a : ℝ) : a * 5 + (2 * a - 1) * (-3) + a - 3 = 0 := by
  sorry

#check fixed_point_on_line

end NUMINAMATH_CALUDE_fixed_point_on_line_l1183_118373


namespace NUMINAMATH_CALUDE_steve_markers_l1183_118374

/-- Given the number of markers for Alia, Austin, and Steve, 
    where Alia has 2 times as many markers as Austin, 
    Austin has one-third as many markers as Steve, 
    and Alia has 40 markers, prove that Steve has 60 markers. -/
theorem steve_markers (alia austin steve : ℕ) 
  (h1 : alia = 2 * austin) 
  (h2 : austin = steve / 3)
  (h3 : alia = 40) : 
  steve = 60 := by sorry

end NUMINAMATH_CALUDE_steve_markers_l1183_118374


namespace NUMINAMATH_CALUDE_clock_solution_l1183_118342

/-- The time in minutes after 9:00 when the minute hand will be exactly opposite
    the place where the hour hand was two minutes ago, five minutes from now. -/
def clock_problem : ℝ → Prop := λ t =>
  0 < t ∧ t < 60 ∧  -- Time is between 9:00 and 10:00
  abs (6 * (t + 5) - (270 + 0.5 * (t - 2))) = 180  -- Opposite hands condition

theorem clock_solution : ∃ t, clock_problem t ∧ t = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_clock_solution_l1183_118342


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1183_118381

theorem nested_fraction_evaluation :
  1 / (2 - 1 / (2 - 1 / (2 - 1 / 3))) = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1183_118381


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_find_b_find_c_find_d_l1183_118322

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

end NUMINAMATH_CALUDE_sum_of_reciprocals_find_b_find_c_find_d_l1183_118322


namespace NUMINAMATH_CALUDE_initial_shoe_pairs_l1183_118375

theorem initial_shoe_pairs (remaining_pairs : ℕ) (lost_shoes : ℕ) : 
  remaining_pairs = 19 → lost_shoes = 9 → 
  (2 * remaining_pairs + lost_shoes + 1) / 2 = 23 := by
sorry

end NUMINAMATH_CALUDE_initial_shoe_pairs_l1183_118375


namespace NUMINAMATH_CALUDE_doughnuts_per_staff_member_l1183_118324

theorem doughnuts_per_staff_member 
  (total_doughnuts : ℕ) 
  (staff_members : ℕ) 
  (doughnuts_left : ℕ) 
  (h1 : total_doughnuts = 50) 
  (h2 : staff_members = 19) 
  (h3 : doughnuts_left = 12) : 
  (total_doughnuts - doughnuts_left) / staff_members = 2 :=
sorry

end NUMINAMATH_CALUDE_doughnuts_per_staff_member_l1183_118324


namespace NUMINAMATH_CALUDE_journey_time_change_l1183_118355

/-- Given a journey that takes 5 hours at 80 miles per hour, prove that the same journey at 50 miles per hour will take 8 hours. -/
theorem journey_time_change (initial_time initial_speed new_speed : ℝ) 
  (h1 : initial_time = 5)
  (h2 : initial_speed = 80)
  (h3 : new_speed = 50) :
  (initial_time * initial_speed) / new_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_change_l1183_118355


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1183_118343

/-- Given a > 0 and a ≠ 1, prove that (-2, 2) is a fixed point of f(x) = a^(x+2) + 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x + 2) + 1
  f (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1183_118343


namespace NUMINAMATH_CALUDE_governor_addresses_ratio_l1183_118327

theorem governor_addresses_ratio (S : ℕ) : 
  S + S / 2 + (S + 10) = 40 → S / (S / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_governor_addresses_ratio_l1183_118327


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l1183_118320

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l1183_118320


namespace NUMINAMATH_CALUDE_remaining_expenses_l1183_118372

def base_8_to_10 (n : ℕ) : ℕ := 
  5 * 8^3 + 4 * 8^2 + 3 * 8^1 + 2 * 8^0

def savings : ℕ := base_8_to_10 5432
def ticket_cost : ℕ := 1200

theorem remaining_expenses : savings - ticket_cost = 1642 := by
  sorry

end NUMINAMATH_CALUDE_remaining_expenses_l1183_118372


namespace NUMINAMATH_CALUDE_min_value_2sin_x_l1183_118338

theorem min_value_2sin_x (x : Real) (h : π/3 ≤ x ∧ x ≤ 5*π/6) : 
  ∃ (y : Real), y = 2 * Real.sin x ∧ y ≥ 1 ∧ ∀ z, (∃ t, π/3 ≤ t ∧ t ≤ 5*π/6 ∧ z = 2 * Real.sin t) → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_2sin_x_l1183_118338


namespace NUMINAMATH_CALUDE_train_length_l1183_118323

/-- Given a train that crosses a platform in 39 seconds, crosses a signal pole in 18 seconds,
    and the platform length is 175 meters, the length of the train is 150 meters. -/
theorem train_length (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
    (h1 : platform_crossing_time = 39)
    (h2 : pole_crossing_time = 18)
    (h3 : platform_length = 175) :
    ∃ train_length : ℝ, train_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1183_118323


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l1183_118392

/-- Given a cube with volume 8x cubic units and surface area 4x square units, x = 216 -/
theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 4*x) → x = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l1183_118392


namespace NUMINAMATH_CALUDE_line_through_point_l1183_118330

/-- Given a line equation 1 - 3kx = 7y and a point (-2/3, 3) on the line, prove that k = 10 -/
theorem line_through_point (k : ℝ) : 
  (1 - 3 * k * (-2/3) = 7 * 3) → k = 10 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1183_118330


namespace NUMINAMATH_CALUDE_average_service_hours_l1183_118336

theorem average_service_hours (n : ℕ) (h1 h2 h3 : ℕ) (s1 s2 s3 : ℕ) :
  n = 10 →
  h1 = 15 →
  h2 = 16 →
  h3 = 20 →
  s1 = 2 →
  s2 = 5 →
  s3 = 3 →
  s1 + s2 + s3 = n →
  (h1 * s1 + h2 * s2 + h3 * s3) / n = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_average_service_hours_l1183_118336


namespace NUMINAMATH_CALUDE_systematic_sampling_fourth_element_l1183_118361

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

end NUMINAMATH_CALUDE_systematic_sampling_fourth_element_l1183_118361


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1183_118326

theorem simplify_and_evaluate (a : ℝ) : 
  (a - 3)^2 - (a - 1) * (a + 1) + 2 * (a + 3) = 4 ↔ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1183_118326


namespace NUMINAMATH_CALUDE_cosine_inequality_l1183_118397

theorem cosine_inequality (y : ℝ) (hy : 0 ≤ y ∧ y ≤ 2 * Real.pi) :
  ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → Real.cos (x - y) ≥ Real.cos x - Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_l1183_118397


namespace NUMINAMATH_CALUDE_second_meeting_time_l1183_118318

/-- The time in seconds for Racing Magic to complete one lap -/
def racing_magic_lap_time : ℕ := 150

/-- The number of laps Charging Bull completes in one hour -/
def charging_bull_laps_per_hour : ℕ := 40

/-- The time in minutes when both vehicles meet at the starting point for the second time -/
def meeting_time : ℕ := 15

/-- Theorem stating that the vehicles meet at the starting point for the second time after 15 minutes -/
theorem second_meeting_time :
  let racing_magic_lap_time_min : ℚ := racing_magic_lap_time / 60
  let charging_bull_lap_time_min : ℚ := 60 / charging_bull_laps_per_hour
  Nat.lcm (Nat.ceil (racing_magic_lap_time_min * 2)) (Nat.ceil (charging_bull_lap_time_min * 2)) / 2 = meeting_time :=
sorry

end NUMINAMATH_CALUDE_second_meeting_time_l1183_118318


namespace NUMINAMATH_CALUDE_fraction_simplification_l1183_118337

theorem fraction_simplification 
  (a b x y : ℝ) : 
  (3*b*x*(a^3*x^3 + 3*a^2*y^2 + 2*b^2*y^2) + 2*a*y*(2*a^2*x^2 + 3*b^2*x^2 + b^3*y^3)) / (3*b*x + 2*a*y) 
  = a^3*x^3 + 3*a^2*x*y + 2*b^2*y^2 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1183_118337


namespace NUMINAMATH_CALUDE_intersection_of_logarithmic_curves_l1183_118389

theorem intersection_of_logarithmic_curves :
  ∃! x : ℝ, x > 0 ∧ 3 * Real.log x = Real.log (3 * x) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_logarithmic_curves_l1183_118389


namespace NUMINAMATH_CALUDE_derivative_y_wrt_x_l1183_118347

noncomputable section

variable (t : ℝ)

def x : ℝ := Real.arcsin (Real.sin t)
def y : ℝ := Real.arccos (Real.cos t)

theorem derivative_y_wrt_x : 
  deriv (fun x => y x) (x t) = 1 :=
sorry

end NUMINAMATH_CALUDE_derivative_y_wrt_x_l1183_118347


namespace NUMINAMATH_CALUDE_a_steps_equals_b_steps_l1183_118313

/-- The number of steps in a stationary escalator -/
def stationary_steps : ℕ := 100

/-- The number of steps B takes -/
def b_steps : ℕ := 75

/-- The relative speed of A compared to B -/
def relative_speed : ℚ := 1/3

theorem a_steps_equals_b_steps :
  ∃ (a : ℕ) (e : ℚ),
    -- A's steps plus escalator movement equals total steps
    a + e = stationary_steps ∧
    -- B's steps plus escalator movement equals total steps
    b_steps + e = stationary_steps ∧
    -- A's speed is 1/3 of B's speed
    a * relative_speed = b_steps * (1 : ℚ) →
    a = b_steps :=
by sorry

end NUMINAMATH_CALUDE_a_steps_equals_b_steps_l1183_118313


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l1183_118348

/-- A line passing through (1, 2) with equal intercepts on both coordinate axes -/
structure EqualInterceptLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (1, 2)
  passes_through : 2 = m * 1 + b
  -- The line has equal intercepts on both axes
  equal_intercepts : m ≠ -1 → b = b / m

/-- The equation of an EqualInterceptLine is either 2x - y = 0 or x + y - 3 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 2 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 3) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l1183_118348


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l1183_118357

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/2)^(x-2)

theorem zero_point_in_interval :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 2 3 ∧ f x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l1183_118357


namespace NUMINAMATH_CALUDE_impossible_all_even_impossible_all_divisible_by_three_l1183_118354

-- Define the cube structure
structure Cube :=
  (vertices : Fin 8 → ℕ)

-- Define the initial state of the cube
def initial_cube : Cube :=
  { vertices := λ i => if i = 0 then 1 else 0 }

-- Define the operation of adding 1 to both ends of an edge
def add_to_edge (c : Cube) (v1 v2 : Fin 8) : Cube :=
  { vertices := λ i => if i = v1 || i = v2 then c.vertices i + 1 else c.vertices i }

-- Define the property of all numbers being divisible by 2
def all_even (c : Cube) : Prop :=
  ∀ i, c.vertices i % 2 = 0

-- Define the property of all numbers being divisible by 3
def all_divisible_by_three (c : Cube) : Prop :=
  ∀ i, c.vertices i % 3 = 0

-- Theorem stating it's impossible to make all numbers even
theorem impossible_all_even :
  ¬ ∃ (operations : List (Fin 8 × Fin 8)), 
    all_even (operations.foldl (λ c (v1, v2) => add_to_edge c v1 v2) initial_cube) :=
sorry

-- Theorem stating it's impossible to make all numbers divisible by 3
theorem impossible_all_divisible_by_three :
  ¬ ∃ (operations : List (Fin 8 × Fin 8)), 
    all_divisible_by_three (operations.foldl (λ c (v1, v2) => add_to_edge c v1 v2) initial_cube) :=
sorry

end NUMINAMATH_CALUDE_impossible_all_even_impossible_all_divisible_by_three_l1183_118354


namespace NUMINAMATH_CALUDE_multiples_of_ten_range_l1183_118309

theorem multiples_of_ten_range (start : ℕ) : 
  (∃ n : ℕ, n = 991) ∧ 
  (start ≤ 10000) ∧
  (∀ x ∈ Set.Icc start 10000, x % 10 = 0 → x ∈ Finset.range 992) ∧
  (10000 ∈ Finset.range 992) →
  start = 90 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_ten_range_l1183_118309


namespace NUMINAMATH_CALUDE_no_simultaneous_greater_value_l1183_118358

theorem no_simultaneous_greater_value : ¬∃ x : ℝ, (x + 3) / 5 > 2 * x + 3 ∧ (x + 3) / 5 > 1 - x := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_greater_value_l1183_118358


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1183_118396

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  x / (x^2 - 1) / (1 + 1 / (x - 1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1183_118396


namespace NUMINAMATH_CALUDE_train_speed_l1183_118344

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length time : ℝ) (h1 : length = 2500) (h2 : time = 100) :
  length / time = 25 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1183_118344


namespace NUMINAMATH_CALUDE_arabella_first_step_time_l1183_118346

/-- Represents the time spent learning dance steps -/
structure DanceSteps where
  first : ℝ
  second : ℝ
  third : ℝ

/-- The conditions for Arabella's dance step learning -/
def arabella_dance_conditions (steps : DanceSteps) : Prop :=
  steps.second = steps.first / 2 ∧
  steps.third = steps.first + steps.second ∧
  steps.first + steps.second + steps.third = 90

/-- Theorem stating that under the given conditions, the time spent on the first step is 30 minutes -/
theorem arabella_first_step_time (steps : DanceSteps) 
  (h : arabella_dance_conditions steps) : steps.first = 30 := by
  sorry

end NUMINAMATH_CALUDE_arabella_first_step_time_l1183_118346


namespace NUMINAMATH_CALUDE_floor_x_floor_x_eq_48_l1183_118307

theorem floor_x_floor_x_eq_48 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 48 ↔ 48 / 7 ≤ x ∧ x < 49 / 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_x_floor_x_eq_48_l1183_118307


namespace NUMINAMATH_CALUDE_remaining_money_proof_l1183_118371

def calculate_remaining_money (initial_amount : ℝ) (sparkling_water_count : ℕ) 
  (sparkling_water_price : ℝ) (sparkling_water_discount : ℝ) 
  (still_water_price : ℝ) (still_water_multiplier : ℕ) 
  (cheddar_weight : ℝ) (cheddar_price : ℝ) 
  (swiss_weight : ℝ) (swiss_price : ℝ) 
  (cheese_tax_rate : ℝ) : ℝ :=
  let sparkling_water_cost := sparkling_water_count * sparkling_water_price * (1 - sparkling_water_discount)
  let still_water_count := sparkling_water_count * still_water_multiplier
  let still_water_paid_bottles := (still_water_count / 3) * 2
  let still_water_cost := still_water_paid_bottles * still_water_price
  let cheese_cost := cheddar_weight * cheddar_price + swiss_weight * swiss_price
  let cheese_tax := cheese_cost * cheese_tax_rate
  let total_cost := sparkling_water_cost + still_water_cost + cheese_cost + cheese_tax
  initial_amount - total_cost

theorem remaining_money_proof :
  calculate_remaining_money 200 4 3 0.1 2.5 3 2.5 8.5 1.75 11 0.05 = 126.67 := by
  sorry

#eval calculate_remaining_money 200 4 3 0.1 2.5 3 2.5 8.5 1.75 11 0.05

end NUMINAMATH_CALUDE_remaining_money_proof_l1183_118371


namespace NUMINAMATH_CALUDE_total_lives_l1183_118350

theorem total_lives (num_friends : ℕ) (lives_per_friend : ℕ) 
  (h1 : num_friends = 8) (h2 : lives_per_friend = 8) : 
  num_friends * lives_per_friend = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_l1183_118350


namespace NUMINAMATH_CALUDE_binomial_constant_term_l1183_118303

/-- The constant term in the binomial expansion of (x - a/(3x))^8 -/
def constantTerm (a : ℝ) : ℝ := ((-1)^6 * a^6) * (Nat.choose 8 6)

theorem binomial_constant_term (a : ℝ) : 
  constantTerm a = 28 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_constant_term_l1183_118303


namespace NUMINAMATH_CALUDE_smallest_odd_digit_multiple_of_9_l1183_118383

/-- A function that checks if a number has only odd digits -/
def hasOnlyOddDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

/-- The smallest positive integer less than 10,000 with only odd digits that is a multiple of 9 -/
def smallestOddDigitMultipleOf9 : ℕ := 1117

theorem smallest_odd_digit_multiple_of_9 :
  smallestOddDigitMultipleOf9 < 10000 ∧
  hasOnlyOddDigits smallestOddDigitMultipleOf9 ∧
  smallestOddDigitMultipleOf9 % 9 = 0 ∧
  ∀ n : ℕ, n < 10000 → hasOnlyOddDigits n → n % 9 = 0 → smallestOddDigitMultipleOf9 ≤ n :=
by sorry

#eval smallestOddDigitMultipleOf9

end NUMINAMATH_CALUDE_smallest_odd_digit_multiple_of_9_l1183_118383


namespace NUMINAMATH_CALUDE_house_count_l1183_118364

/-- The number of houses in a development with specific features -/
theorem house_count (G P GP N : ℕ) (hG : G = 50) (hP : P = 40) (hGP : GP = 35) (hN : N = 10) :
  G + P - GP + N = 65 := by
  sorry

#check house_count

end NUMINAMATH_CALUDE_house_count_l1183_118364


namespace NUMINAMATH_CALUDE_quadratic_sum_equals_seven_l1183_118302

def quadratic (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 6

theorem quadratic_sum_equals_seven :
  ∃ (a h k : ℝ), (∀ x, quadratic x = a * (x - h)^2 + k) ∧ (a + h + k = 7) := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_equals_seven_l1183_118302


namespace NUMINAMATH_CALUDE_triangle_inequality_l1183_118399

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b + b * c + c * a ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1183_118399


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1183_118306

/-- The radius of the inscribed circle in a triangle with sides 9, 10, and 11 is 2√2 -/
theorem inscribed_circle_radius (a b c : ℝ) (h_a : a = 9) (h_b : b = 10) (h_c : c = 11) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1183_118306


namespace NUMINAMATH_CALUDE_brookes_initial_balloons_l1183_118378

theorem brookes_initial_balloons :
  ∀ (b : ℕ), -- Brooke's initial number of balloons
  let brooke_final := b + 8 -- Brooke's final number of balloons
  let tracy_initial := 6 -- Tracy's initial number of balloons
  let tracy_added := 24 -- Number of balloons Tracy adds
  let tracy_final := (tracy_initial + tracy_added) / 2 -- Tracy's final number of balloons after popping half
  brooke_final + tracy_final = 35 → b = 12 :=
by
  sorry

#check brookes_initial_balloons

end NUMINAMATH_CALUDE_brookes_initial_balloons_l1183_118378


namespace NUMINAMATH_CALUDE_ratio_cube_square_l1183_118334

theorem ratio_cube_square (x y : ℝ) (h : x / y = 7 / 5) : 
  x^3 / y^2 = 343 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_cube_square_l1183_118334


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1183_118359

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (2, m)
  parallel a b → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1183_118359


namespace NUMINAMATH_CALUDE_expansion_coefficient_constraint_l1183_118311

theorem expansion_coefficient_constraint (k : ℕ+) :
  (15 : ℝ) * (k : ℝ)^4 < 120 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_constraint_l1183_118311


namespace NUMINAMATH_CALUDE_final_balance_l1183_118314

def bank_account_balance (initial_savings withdrawal deposit : ℕ) : ℕ :=
  initial_savings - withdrawal + deposit

theorem final_balance (initial_savings withdrawal : ℕ) 
  (h1 : initial_savings = 230)
  (h2 : withdrawal = 60)
  (h3 : bank_account_balance initial_savings withdrawal (2 * withdrawal) = 290) : 
  bank_account_balance initial_savings withdrawal (2 * withdrawal) = 290 := by
  sorry

end NUMINAMATH_CALUDE_final_balance_l1183_118314


namespace NUMINAMATH_CALUDE_integer_values_less_than_sqrt2_l1183_118388

theorem integer_values_less_than_sqrt2 (x : ℤ) : 
  (|x| : ℝ) < Real.sqrt 2 → x = -1 ∨ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_values_less_than_sqrt2_l1183_118388


namespace NUMINAMATH_CALUDE_geometric_mean_point_existence_l1183_118366

theorem geometric_mean_point_existence (A B C : ℝ) :
  ∃ (D : ℝ), 0 ≤ D ∧ D ≤ 1 ∧
  (Real.sin A * Real.sin B ≤ Real.sin (C / 2) ^ 2) ↔
  ∃ (CD AD DB : ℝ), CD ^ 2 = AD * DB ∧ AD + DB = 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_mean_point_existence_l1183_118366


namespace NUMINAMATH_CALUDE_square_land_side_length_l1183_118351

/-- Given a square-shaped land plot with an area of 625 square units,
    prove that the length of one side is 25 units. -/
theorem square_land_side_length :
  ∀ (side : ℝ), side > 0 → side * side = 625 → side = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l1183_118351


namespace NUMINAMATH_CALUDE_wendys_bouquets_l1183_118315

/-- Calculates the number of bouquets that can be made given the initial number of flowers,
    number of wilted flowers, and number of flowers per bouquet. -/
def calculateBouquets (initialFlowers : ℕ) (wiltedFlowers : ℕ) (flowersPerBouquet : ℕ) : ℕ :=
  (initialFlowers - wiltedFlowers) / flowersPerBouquet

/-- Proves that Wendy can make 2 bouquets with the given conditions. -/
theorem wendys_bouquets :
  calculateBouquets 45 35 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_wendys_bouquets_l1183_118315


namespace NUMINAMATH_CALUDE_green_ball_probability_l1183_118301

-- Define the containers and their contents
def containerA : ℕ × ℕ := (4, 6)  -- (red, green)
def containerB : ℕ × ℕ := (6, 4)
def containerC : ℕ × ℕ := (6, 4)

-- Define the probability of selecting each container
def containerProb : ℚ := 1 / 3

-- Function to calculate the probability of selecting a green ball from a container
def greenBallProb (container : ℕ × ℕ) : ℚ :=
  container.2 / (container.1 + container.2)

-- Theorem statement
theorem green_ball_probability :
  containerProb * greenBallProb containerA +
  containerProb * greenBallProb containerB +
  containerProb * greenBallProb containerC = 7 / 15 := by
  sorry


end NUMINAMATH_CALUDE_green_ball_probability_l1183_118301


namespace NUMINAMATH_CALUDE_geometric_series_sum_proof_l1183_118398

/-- The sum of the infinite geometric series 5/3 - 5/6 + 5/18 - 5/54 + ... -/
def geometric_series_sum : ℚ := 10/9

/-- The first term of the geometric series -/
def a : ℚ := 5/3

/-- The common ratio of the geometric series -/
def r : ℚ := -1/2

theorem geometric_series_sum_proof :
  geometric_series_sum = a / (1 - r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_proof_l1183_118398


namespace NUMINAMATH_CALUDE_cars_in_first_section_l1183_118331

/-- The number of rows in the first section -/
def first_section_rows : ℕ := 15

/-- The number of cars per row in the first section -/
def first_section_cars_per_row : ℕ := 10

/-- The number of rows in the second section -/
def second_section_rows : ℕ := 20

/-- The number of cars per row in the second section -/
def second_section_cars_per_row : ℕ := 9

/-- The number of cars Nate can walk past per minute -/
def cars_passed_per_minute : ℕ := 11

/-- The number of minutes Nate spent searching -/
def search_time_minutes : ℕ := 30

/-- The theorem stating the number of cars in the first section -/
theorem cars_in_first_section :
  first_section_rows * first_section_cars_per_row = 150 := by
  sorry

end NUMINAMATH_CALUDE_cars_in_first_section_l1183_118331


namespace NUMINAMATH_CALUDE_sin_150_degrees_l1183_118310

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l1183_118310


namespace NUMINAMATH_CALUDE_abc_expression_value_l1183_118395

theorem abc_expression_value (a b c : ℚ) 
  (ha : a^2 = 9)
  (hb : abs b = 4)
  (hc : c^3 = 27)
  (hab : a * b < 0)
  (hbc : b * c > 0) :
  a * b - b * c + c * a = -33 := by
sorry

end NUMINAMATH_CALUDE_abc_expression_value_l1183_118395


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1183_118341

theorem pure_imaginary_complex_number (m : ℝ) : 
  (((m^2 - m - 2) : ℂ) + (m + 1) * Complex.I).re = 0 ∧ 
  (((m^2 - m - 2) : ℂ) + (m + 1) * Complex.I).im ≠ 0 → 
  m = 2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1183_118341


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l1183_118367

theorem log_equality_implies_golden_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.log a / Real.log 8 = Real.log b / Real.log 18) ∧
  (Real.log a / Real.log 8 = Real.log (a + b) / Real.log 32) →
  b / a = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l1183_118367


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l1183_118328

-- Define the circle and its properties
def circle_radius : ℝ := 10
def central_angle : ℝ := 270

-- Theorem statement
theorem shaded_region_perimeter :
  let perimeter := 2 * circle_radius + (central_angle / 360) * (2 * Real.pi * circle_radius)
  perimeter = 20 + 15 * Real.pi := by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l1183_118328


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1183_118393

theorem algebraic_expression_value (a : ℝ) (h : 2 * a^2 - 3 * a = 1) :
  9 * a + 7 - 6 * a^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1183_118393


namespace NUMINAMATH_CALUDE_triangle_side_length_l1183_118316

theorem triangle_side_length (a b c : ℝ) (A : ℝ) : 
  a = 3 * Real.sqrt 2 →
  b = 6 →
  A = π / 6 →
  c^2 - 6 * Real.sqrt 3 * c + 18 = 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1183_118316


namespace NUMINAMATH_CALUDE_square_tiles_count_l1183_118333

/-- Represents the number of edges for each tile type -/
def edges_per_tile : Fin 3 → ℕ
| 0 => 3  -- Triangle
| 1 => 4  -- Square
| 2 => 5  -- Rectangle

/-- Proves that given the conditions, the number of square tiles is 10 -/
theorem square_tiles_count 
  (total_tiles : ℕ) 
  (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 32)
  (h_total_edges : total_edges = 114) :
  ∃ (triangles squares rectangles : ℕ),
    triangles + squares + rectangles = total_tiles ∧
    3 * triangles + 4 * squares + 5 * rectangles = total_edges ∧
    squares = 10 :=
by sorry

end NUMINAMATH_CALUDE_square_tiles_count_l1183_118333


namespace NUMINAMATH_CALUDE_trapezoid_larger_base_length_l1183_118382

/-- A trapezoid with a midline of length 10 and a diagonal that divides the midline
    into two parts with a difference of 3 has a larger base of length 13. -/
theorem trapezoid_larger_base_length (x y : ℝ) 
  (h1 : (x + y) / 2 = 10)  -- midline length is 10
  (h2 : x - y = 6)         -- difference between parts of divided midline is 3 * 2
  : x = 13 := by  -- x represents the larger base
  sorry

end NUMINAMATH_CALUDE_trapezoid_larger_base_length_l1183_118382


namespace NUMINAMATH_CALUDE_given_number_scientific_notation_l1183_118370

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The given number to be converted -/
def given_number : ℝ := 0.00000164

/-- The expected scientific notation representation -/
def expected_notation : ScientificNotation := {
  coefficient := 1.64,
  exponent := -6,
  is_valid := by sorry
}

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_scientific_notation : given_number = expected_notation.coefficient * (10 : ℝ) ^ expected_notation.exponent := by
  sorry

end NUMINAMATH_CALUDE_given_number_scientific_notation_l1183_118370


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l1183_118362

theorem triangle_side_ratio (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 2 * a →
  b / a = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l1183_118362


namespace NUMINAMATH_CALUDE_two_successes_in_four_trials_l1183_118356

def probability_of_two_successes_in_four_trials (p : ℝ) : ℝ :=
  6 * p^2 * (1 - p)^2

theorem two_successes_in_four_trials :
  probability_of_two_successes_in_four_trials 0.6 = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_two_successes_in_four_trials_l1183_118356


namespace NUMINAMATH_CALUDE_inverse_proportion_l1183_118379

theorem inverse_proportion (x₁ x₂ y₁ y₂ : ℝ) (h1 : x₁ ≠ 0) (h2 : x₂ ≠ 0) (h3 : y₁ ≠ 0) (h4 : y₂ ≠ 0)
  (h5 : ∃ k : ℝ, ∀ x y : ℝ, x * y = k) (h6 : x₁ / x₂ = 4 / 5) :
  y₁ / y₂ = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_l1183_118379


namespace NUMINAMATH_CALUDE_nine_digit_number_not_prime_l1183_118352

/-- A function that checks if a number is a three-digit prime -/
def isThreeDigitPrime (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ Nat.Prime n

/-- A function that forms a nine-digit number from three three-digit numbers -/
def concatenateThreeNumbers (a b c : ℕ) : ℕ :=
  a * 1000000 + b * 1000 + c

/-- The main theorem -/
theorem nine_digit_number_not_prime 
  (a b c : ℕ) 
  (h1 : isThreeDigitPrime a) 
  (h2 : isThreeDigitPrime b) 
  (h3 : isThreeDigitPrime c) 
  (h4 : ∃ (d : ℤ), b = a + d ∧ c = b + d) : 
  ¬ Nat.Prime (concatenateThreeNumbers a b c) := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_number_not_prime_l1183_118352


namespace NUMINAMATH_CALUDE_original_number_proof_l1183_118391

theorem original_number_proof : ∃ (n : ℕ), n ≥ 129 ∧ (n - 30) % 99 = 0 ∧ ∀ (m : ℕ), m < 129 → (m - 30) % 99 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l1183_118391


namespace NUMINAMATH_CALUDE_derivative_at_one_l1183_118394

def f (x : ℝ) : ℝ := (x - 2)^2

theorem derivative_at_one :
  deriv f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1183_118394


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l1183_118387

theorem sphere_surface_area_from_rectangular_solid (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) : 
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  4 * Real.pi * radius^2 = 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l1183_118387


namespace NUMINAMATH_CALUDE_max_value_on_parabola_l1183_118308

/-- The maximum value of m + n where (m,n) lies on y = -x^2 + 3 is 13/4 -/
theorem max_value_on_parabola :
  ∀ m n : ℝ, n = -m^2 + 3 → (∀ x y : ℝ, y = -x^2 + 3 → m + n ≥ x + y) → m + n = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_parabola_l1183_118308


namespace NUMINAMATH_CALUDE_stating_solutions_depend_on_angle_l1183_118386

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

end NUMINAMATH_CALUDE_stating_solutions_depend_on_angle_l1183_118386


namespace NUMINAMATH_CALUDE_alex_marbles_l1183_118339

theorem alex_marbles (lorin_black : ℕ) (jimmy_yellow : ℕ) 
  (h1 : lorin_black = 4)
  (h2 : jimmy_yellow = 22)
  (alex_black : ℕ) (alex_yellow : ℕ)
  (h3 : alex_black = 2 * lorin_black)
  (h4 : alex_yellow = jimmy_yellow / 2) :
  alex_black + alex_yellow = 19 := by
sorry

end NUMINAMATH_CALUDE_alex_marbles_l1183_118339


namespace NUMINAMATH_CALUDE_smallest_w_l1183_118340

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 → 
  is_factor (2^5) (936 * w) → 
  is_factor (3^3) (936 * w) → 
  is_factor (14^2) (936 * w) → 
  w ≥ 1764 :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l1183_118340


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1183_118353

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 + a 2 + a 3 + a 4 = 30) →
  (a 2 + a 3 = 15) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1183_118353


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l1183_118345

theorem arccos_equation_solution (x : ℝ) : 
  Real.arccos (3 * x) - Real.arccos (2 * x) = π / 6 →
  x = 1 / (2 * Real.sqrt (12 - 6 * Real.sqrt 3)) ∨
  x = -1 / (2 * Real.sqrt (12 - 6 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l1183_118345


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l1183_118312

def is_arithmetic_sequence (seq : Fin 4 → ℝ) : Prop :=
  ∃ (a d : ℝ), seq 0 = a - d ∧ seq 1 = a ∧ seq 2 = a + d ∧ seq 3 = a + 2*d

def sum_is_26 (seq : Fin 4 → ℝ) : Prop :=
  (seq 0) + (seq 1) + (seq 2) + (seq 3) = 26

def middle_product_is_40 (seq : Fin 4 → ℝ) : Prop :=
  (seq 1) * (seq 2) = 40

theorem arithmetic_sequence_theorem (seq : Fin 4 → ℝ) :
  is_arithmetic_sequence seq ∧ sum_is_26 seq ∧ middle_product_is_40 seq →
  (seq 0 = 2 ∧ seq 1 = 5 ∧ seq 2 = 8 ∧ seq 3 = 11) ∨
  (seq 0 = 11 ∧ seq 1 = 8 ∧ seq 2 = 5 ∧ seq 3 = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l1183_118312


namespace NUMINAMATH_CALUDE_square_sum_plus_product_squares_l1183_118380

theorem square_sum_plus_product_squares : (3 + 9)^2 + 3^2 * 9^2 = 873 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_plus_product_squares_l1183_118380


namespace NUMINAMATH_CALUDE_expression_undefined_at_nine_l1183_118363

/-- The expression (3x^3 + 4) / (x^2 - 18x + 81) is undefined when x = 9 -/
theorem expression_undefined_at_nine :
  ∀ x : ℝ, x = 9 → (x^2 - 18*x + 81 = 0) := by
  sorry

end NUMINAMATH_CALUDE_expression_undefined_at_nine_l1183_118363


namespace NUMINAMATH_CALUDE_digit_equation_solution_l1183_118304

theorem digit_equation_solution :
  ∀ (A M C : ℕ),
  (A ≤ 9 ∧ M ≤ 9 ∧ C ≤ 9) →
  (10 * A^2 + 10 * M + C) * (A + M^2 + C^2) = 1050 →
  A = 2 := by
sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l1183_118304


namespace NUMINAMATH_CALUDE_abs_x_minus_two_iff_x_in_range_l1183_118368

theorem abs_x_minus_two_iff_x_in_range (x : ℝ) : |x - 2| ≤ 5 ↔ -3 ≤ x ∧ x ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_two_iff_x_in_range_l1183_118368


namespace NUMINAMATH_CALUDE_chicken_infection_probabilities_l1183_118300

def vaccine_effectiveness : ℝ := 0.8
def num_chickens : ℕ := 5

theorem chicken_infection_probabilities :
  let p_none_infected := vaccine_effectiveness ^ num_chickens
  let p_one_infected := num_chickens * vaccine_effectiveness ^ (num_chickens - 1) * (1 - vaccine_effectiveness)
  (p_none_infected = 1024 / 3125) ∧ (p_one_infected = 256 / 625) := by
  sorry

end NUMINAMATH_CALUDE_chicken_infection_probabilities_l1183_118300


namespace NUMINAMATH_CALUDE_employees_using_public_transportation_l1183_118349

theorem employees_using_public_transportation 
  (total_employees : ℕ) 
  (drive_percentage : ℚ) 
  (public_transport_fraction : ℚ) :
  total_employees = 100 →
  drive_percentage = 60 / 100 →
  public_transport_fraction = 1 / 2 →
  (total_employees : ℚ) * (1 - drive_percentage) * public_transport_fraction = 20 := by
  sorry

end NUMINAMATH_CALUDE_employees_using_public_transportation_l1183_118349
