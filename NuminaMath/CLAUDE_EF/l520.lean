import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_140_l520_52044

-- Define the given constants
noncomputable def train_length : ℝ := 360
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def time_to_pass : ℝ := 40

-- Define the conversion factor from km/h to m/s
noncomputable def km_h_to_m_s : ℝ := 1000 / 3600

-- Define the function to calculate the platform length
noncomputable def platform_length : ℝ :=
  train_speed_kmh * km_h_to_m_s * time_to_pass - train_length

-- State the theorem
theorem platform_length_is_140 : platform_length = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_140_l520_52044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_and_profit_l520_52024

/-- Represents the profit model for a product with given cost and pricing parameters. -/
structure ProfitModel where
  cost : ℝ  -- Cost price in yuan
  initial_price : ℝ  -- Initial selling price in yuan
  initial_sales : ℝ  -- Initial sales in units
  increase_effect : ℝ  -- Units lost per yuan of price increase
  decrease_effect : ℝ  -- Units gained per yuan of price decrease

/-- Calculates the profit for a given price change. -/
noncomputable def profit (model : ProfitModel) (price_change : ℝ) : ℝ :=
  if price_change ≥ 0 then
    (model.initial_price + price_change - model.cost) * (model.initial_sales - model.increase_effect * price_change)
  else
    (model.initial_price + price_change - model.cost) * (model.initial_sales - model.decrease_effect * price_change)

/-- Theorem stating the optimal price and maximum profit for the given model. -/
theorem optimal_price_and_profit (model : ProfitModel) 
  (h_cost : model.cost = 100)
  (h_initial_price : model.initial_price = 120)
  (h_initial_sales : model.initial_sales = 300)
  (h_increase_effect : model.increase_effect = 10)
  (h_decrease_effect : model.decrease_effect = -30) :
  ∃ (optimal_change : ℝ),
    optimal_change = -5 ∧
    profit model optimal_change = 6750 ∧
    ∀ (x : ℝ), profit model x ≤ profit model optimal_change :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_and_profit_l520_52024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_point_l520_52016

open Real

-- Define the function f(x) = sin(2x) - x
noncomputable def f (x : ℝ) := Real.sin (2 * x) - x

-- State the theorem
theorem max_value_point :
  ∃ (c : ℝ), c = π / 6 ∧ c ∈ Set.Ioo 0 π ∧ ∀ y ∈ Set.Ioo 0 π, f y ≤ f c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_point_l520_52016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l520_52005

-- Define the equation
noncomputable def f (x a : ℝ) : ℝ := 2 * (1/4)^(-x) - (1/2)^(-x) + a

-- State the theorem
theorem range_of_a :
  (∃ x ∈ Set.Icc (-1 : ℝ) 0, f x a = 0) →
  a ∈ Set.Icc (-1 : ℝ) 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l520_52005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_problem_l520_52099

noncomputable def A : ℝ × ℝ × ℝ := (-2, 8, 12)
noncomputable def C : ℝ × ℝ × ℝ := (4, 4, 10)
noncomputable def B : ℝ × ℝ × ℝ := (1, 29/4, 43/4)

def plane_equation (p : ℝ × ℝ × ℝ) : Prop :=
  2 * p.1 + p.2.1 + p.2.2 = 18

def is_reflection (incident : ℝ × ℝ × ℝ) (point : ℝ × ℝ × ℝ) (reflected : ℝ × ℝ × ℝ) : Prop :=
  let normal : ℝ × ℝ × ℝ := (2, 1, 1)
  let v1 := (point.1 - incident.1, point.2.1 - incident.2.1, point.2.2 - incident.2.2)
  let v2 := (reflected.1 - point.1, reflected.2.1 - point.2.1, reflected.2.2 - point.2.2)
  let dot_product := v1.1 * normal.1 + v1.2.1 * normal.2.1 + v1.2.2 * normal.2.2
  let normal_squared := normal.1^2 + normal.2.1^2 + normal.2.2^2
  v2 = (v1.1 - 2 * dot_product / normal_squared * normal.1,
        v1.2.1 - 2 * dot_product / normal_squared * normal.2.1,
        v1.2.2 - 2 * dot_product / normal_squared * normal.2.2)

theorem reflection_problem :
  plane_equation B ∧ is_reflection A B C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_problem_l520_52099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_second_quadrant_l520_52023

theorem cos_minus_sin_second_quadrant (α : ℝ) : 
  (π / 2 < α ∧ α < π) →  -- α is in the second quadrant
  Real.sin (2 * α) = -24 / 25 → 
  Real.cos α - Real.sin α = -7 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_second_quadrant_l520_52023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_proof_l520_52098

theorem x_value_proof (x : ℝ) (h1 : x > 0) (h2 : x * Int.floor x = 24) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_proof_l520_52098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_l520_52002

/-- The function for which we're finding the vertical asymptote -/
noncomputable def f (x : ℝ) : ℝ := (2*x + 3) / (7*x - 9)

/-- The x-value of the vertical asymptote -/
noncomputable def asymptote : ℝ := 9/7

theorem vertical_asymptote :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - asymptote| ∧ |x - asymptote| < δ → |f x| > 1/ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_l520_52002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_l520_52054

noncomputable section

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l
def lineL (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the fixed point P
def P : ℝ × ℝ := (-2, 2)

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem min_distance :
  ∃ (min : ℝ), min = Real.sqrt 37 - 1 ∧
  ∀ (A B : ℝ × ℝ),
    circleC A.1 A.2 →
    lineL B.1 B.2 →
    distance P B + distance A B ≥ min := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_l520_52054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_ten_l520_52081

/-- Represents a scientist's observation -/
structure Observation where
  start_time : ℝ
  end_time : ℝ
  distance : ℝ

/-- Represents the snail's journey -/
structure Journey where
  duration : ℝ
  observations : List Observation

/-- Helper function to sum distances of observations -/
def sum_distances (observations : List Observation) : ℝ :=
  observations.foldl (λ acc o => acc + o.distance) 0

/-- Conditions for a valid journey -/
def is_valid_journey (j : Journey) : Prop :=
  j.duration = 6 ∧
  (∀ t, 0 ≤ t ∧ t ≤ j.duration → ∃ o ∈ j.observations, o.start_time ≤ t ∧ t ≤ o.end_time) ∧
  (∀ o ∈ j.observations, o.end_time - o.start_time = 1 ∧ o.distance = 1)

/-- The theorem to prove -/
theorem max_distance_is_ten (j : Journey) (h : is_valid_journey j) :
  (∃ d : ℝ, d ≤ 10 ∧ ∀ d' : ℝ, (∃ j' : Journey, is_valid_journey j' ∧ d' = sum_distances j'.observations) → d' ≤ d) :=
by
  sorry -- Proof to be completed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_ten_l520_52081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_npq_l520_52031

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Midpoint of a line segment -/
def is_midpoint (M X Y : Point) : Prop :=
  M.x = (X.x + Y.x) / 2 ∧ M.y = (X.y + Y.y) / 2

/-- Area of a triangle -/
noncomputable def area_triangle (X Y Z : Point) : ℝ :=
  abs ((X.x - Z.x) * (Y.y - Z.y) - (Y.x - Z.x) * (X.y - Z.y)) / 2

/-- Given a triangle XYZ with area 120, prove that the area of triangle NPQ is 7.5,
    where M is the midpoint of XZ, N is the midpoint of XY, P is the midpoint of MZ,
    and Q is the midpoint of MY. -/
theorem area_of_npq (X Y Z : Point) (M N P Q : Point)
  (h_area : area_triangle X Y Z = 120)
  (h_M : is_midpoint M X Z)
  (h_N : is_midpoint N X Y)
  (h_P : is_midpoint P M Z)
  (h_Q : is_midpoint Q M Y) :
  area_triangle N P Q = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_npq_l520_52031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_car_theorem_l520_52089

/-- Represents the movement of a toy car, where positive values are eastward and negative values are westward. -/
def ToyCarMovements : List Int := [15, -25, 20, -35]

/-- Calculates the final position of the toy car relative to its starting point. -/
def finalPosition (movements : List Int) : Int :=
  movements.sum

/-- Calculates the total distance traveled by the toy car. -/
def totalDistance (movements : List Int) : Nat :=
  movements.map Int.natAbs |>.sum

theorem toy_car_theorem (movements : List Int := ToyCarMovements) :
  finalPosition movements = -25 ∧ totalDistance movements = 95 := by
  sorry

#eval finalPosition ToyCarMovements
#eval totalDistance ToyCarMovements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_car_theorem_l520_52089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_implies_a_range_l520_52006

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := 2^x - 2/x - a

-- Theorem statement
theorem zero_point_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Ioo 1 2, f x a = 0) → a ∈ Set.Ioo 0 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_implies_a_range_l520_52006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_pay_calculation_l520_52009

/-- The weekly pay of employee Y when the total pay for X and Y is 580 units
    and X is paid 120% of Y's pay. -/
noncomputable def employee_y_pay (total_pay : ℝ) (x_percentage : ℝ) : ℝ :=
  total_pay / (1 + x_percentage)

theorem employee_pay_calculation :
  let total_pay : ℝ := 580
  let x_percentage : ℝ := 1.2
  abs (employee_y_pay total_pay x_percentage - 263.64) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_pay_calculation_l520_52009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_100_equals_505_minus_fraction_l520_52096

open BigOperators

def series_sum (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, (2 + (k + 1) * 10) / 3^(n - k)

theorem series_sum_100_equals_505_minus_fraction : 
  series_sum 100 = 505 - 5 / (2 * 9^49) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_100_equals_505_minus_fraction_l520_52096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_g_l520_52010

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

noncomputable def g (x : ℝ) : ℝ := 3^x

-- State the theorem
theorem f_inverse_of_g (x : ℝ) (h : x > 0) :
  f (g x) = x ∧ g (f x) = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_of_g_l520_52010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_equals_pascal_l520_52020

/-- Pascal's triangle as a function from row and position to natural number -/
def pascal (n k : ℕ) : ℕ :=
  match n, k with
  | _, 0 => 1
  | 0, _ => 0
  | n+1, k+1 => pascal n k + pascal n (k+1)

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ :=
  if k ≤ n then
    Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  else
    0

theorem binomial_equals_pascal (n k : ℕ) (h : k ≤ n) :
  binomial n k = pascal n k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_equals_pascal_l520_52020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_disk_areas_l520_52056

/-- The number of disks -/
def n : ℕ := 15

/-- The radius of the large circle -/
def R : ℝ := 1

/-- A function that returns true if the disks cover the circle, don't overlap, and are tangent to their neighbors -/
def valid_arrangement (disk_radius : ℝ) : Prop :=
  disk_radius > 0 ∧ 
  n * (2 * disk_radius) ≥ 2 * Real.pi * R ∧
  2 * n * disk_radius ≤ 2 * Real.pi * R

/-- The theorem stating the sum of the areas of the disks -/
theorem sum_of_disk_areas :
  ∃ (disk_radius : ℝ), 
    valid_arrangement disk_radius → 
    n * Real.pi * disk_radius^2 = Real.pi * (105 - 60 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_disk_areas_l520_52056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1994_l520_52029

noncomputable def sequenceA (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => (sequenceA a n * Real.sqrt 3 + 1) / (Real.sqrt 3 - sequenceA a n)

theorem sequence_1994 (a : ℝ) : 
  sequenceA a 1994 = (a + Real.sqrt 3) / (1 - a * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1994_l520_52029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_ball_on_torus_l520_52037

/-- The radius of the largest spherical ball that can be placed atop the center of a torus -/
noncomputable def largest_ball_radius (inner_radius outer_radius : ℝ) 
  (circle_center : ℝ × ℝ × ℝ) (circle_radius : ℝ) : ℝ :=
  81 / 32

/-- Theorem stating that the radius of the largest spherical ball is 81/32 for the given torus -/
theorem largest_ball_on_torus :
  let inner_radius : ℝ := 3
  let outer_radius : ℝ := 6
  let circle_center : ℝ × ℝ × ℝ := (4.5, 0, 2)
  let circle_radius : ℝ := 2
  largest_ball_radius inner_radius outer_radius circle_center circle_radius = 81 / 32 := by
  sorry

#check largest_ball_on_torus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_ball_on_torus_l520_52037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_third_not_first_quadrant_l520_52052

-- Define the fourth quadrant
def fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, α > 2 * k * Real.pi + 3 * Real.pi / 2 ∧ α < 2 * k * Real.pi + 2 * Real.pi

-- Define the first quadrant
def first_quadrant (α : Real) : Prop :=
  ∃ k : Int, α > 2 * k * Real.pi ∧ α < 2 * k * Real.pi + Real.pi / 2

theorem alpha_third_not_first_quadrant (α : Real) :
  fourth_quadrant α → ¬ first_quadrant (α / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_third_not_first_quadrant_l520_52052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disaster_relief_team_selection_l520_52085

def total_doctors : ℕ := 20
def internal_doctors : ℕ := 12
def surgeons : ℕ := 8
def team_size : ℕ := 5

-- A is an internal medicine doctor, B is a surgeon
-- Remove the Fin type and use ℕ instead
def A : ℕ := 0
def B : ℕ := 0

def selection_with_A_and_B : ℕ := Nat.choose (total_doctors - 2) (team_size - 2)

def selection_without_A_and_B : ℕ := Nat.choose (total_doctors - 2) team_size

def selection_with_at_least_A_or_B : ℕ := 
  Nat.choose 2 1 * Nat.choose (total_doctors - 2) (team_size - 1) + 
  Nat.choose (total_doctors - 2) (team_size - 2)

def selection_with_at_least_one_of_each : ℕ := 
  Nat.choose internal_doctors 1 * Nat.choose surgeons 4 +
  Nat.choose internal_doctors 2 * Nat.choose surgeons 3 +
  Nat.choose internal_doctors 3 * Nat.choose surgeons 2 +
  Nat.choose internal_doctors 4 * Nat.choose surgeons 1

theorem disaster_relief_team_selection :
  selection_with_A_and_B = 816 ∧
  selection_without_A_and_B = 8568 ∧
  selection_with_at_least_A_or_B = 6936 ∧
  selection_with_at_least_one_of_each = 14656 := by
  sorry

#eval selection_with_A_and_B
#eval selection_without_A_and_B
#eval selection_with_at_least_A_or_B
#eval selection_with_at_least_one_of_each

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disaster_relief_team_selection_l520_52085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_octahedron_tiles_space_l520_52075

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the polyhedron formed by truncating an octahedron -/
structure TruncatedOctahedron where
  original_side : ℝ
  truncated_edge : ℝ

/-- Checks if a point has all even or all odd coordinates -/
def hasConsistentParity (p : Point3D) : Prop :=
  (Int.mod (Int.floor p.x) 2 = Int.mod (Int.floor p.y) 2) ∧ 
  (Int.mod (Int.floor p.y) 2 = Int.mod (Int.floor p.z) 2)

/-- Represents the tiling of 3D space -/
def SpaceTiling (poly : TruncatedOctahedron) :=
  ∃ (tiling : Set Point3D),
    (∀ p : Point3D, p ∈ tiling → hasConsistentParity p) ∧
    (∀ p q : Point3D, p ∈ tiling → q ∈ tiling → p ≠ q →
      ∃ (plane : Point3D → ℝ), 
        (∀ x : Point3D, plane x = 0 → 
          (x.x - (p.x + q.x)/2)^2 + (x.y - (p.y + q.y)/2)^2 + (x.z - (p.z + q.z)/2)^2 = 
          ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2) / 4))

/-- The main theorem stating that the truncated octahedron can tile 3D space -/
theorem truncated_octahedron_tiles_space (poly : TruncatedOctahedron) 
  (h1 : poly.original_side = 1) 
  (h2 : poly.truncated_edge = 1/3) : 
  SpaceTiling poly := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_octahedron_tiles_space_l520_52075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_after_cube_submersion_l520_52027

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the new water height after submerging a cube in a tank -/
noncomputable def newWaterHeight (tank : TankDimensions) (initialWaterHeight : ℝ) (cubeSideLength : ℝ) : ℝ :=
  (initialWaterHeight * tank.width * tank.length + cubeSideLength ^ 3) / (tank.width * tank.length)

/-- Theorem stating the new water height after submerging a cube in the given tank -/
theorem water_height_after_cube_submersion 
  (tank : TankDimensions)
  (initialWaterHeight : ℝ)
  (cubeSideLength : ℝ)
  (h1 : tank.width = 50)
  (h2 : tank.length = 16)
  (h3 : tank.height = 25)
  (h4 : initialWaterHeight = 15)
  (h5 : cubeSideLength = 10) :
  newWaterHeight tank initialWaterHeight cubeSideLength = 16.25 := by
  sorry

-- Remove the #eval line as it's not necessary for the proof and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_after_cube_submersion_l520_52027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_of_PFQ_l520_52048

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, -1)

-- Define point P on the parabola
def P : ℝ × ℝ := (-4, -4)

-- Define point Q as the intersection of the tangent line at P with the x-axis
def Q : ℝ × ℝ := (-2, 0)

-- Theorem statement
theorem circumcircle_of_PFQ : 
  ∀ (x y : ℝ),
  parabola P.1 P.2 →
  ((x + 2)^2 + (y + 5/2)^2 = 25/4) ↔ 
  ((x - P.1)^2 + (y - P.2)^2 = 
   (x - focus.1)^2 + (y - focus.2)^2 ∧
   (x - focus.1)^2 + (y - focus.2)^2 = 
   (x - Q.1)^2 + (y - Q.2)^2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_of_PFQ_l520_52048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characteristic_equation_of_B_l520_52040

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 2, 1],
    ![2, 0, 2],
    ![1, 2, 0]]

theorem characteristic_equation_of_B :
  ∃ (a b c : ℤ), 
    B^3 + a • B^2 + b • B + c • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 ∧
    a = 0 ∧ b = -10 ∧ c = -32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characteristic_equation_of_B_l520_52040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_tea_cost_in_july_l520_52013

-- Define the cost per pound of green tea and coffee in June
noncomputable def june_cost : ℝ := sorry

-- Define the cost per pound of coffee in July
noncomputable def july_coffee_cost : ℝ := 2 * june_cost

-- Define the cost per pound of green tea in July
noncomputable def july_green_tea_cost : ℝ := 0.3 * june_cost

-- Define the cost of the mixture in July
def mixture_cost : ℝ := 3.45

-- Theorem statement
theorem green_tea_cost_in_july :
  july_green_tea_cost = 0.30 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_tea_cost_in_july_l520_52013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_derivative_at_pi_l520_52065

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

theorem f_sum_derivative_at_pi :
  f π + deriv f (π / 2) = -3 / π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_derivative_at_pi_l520_52065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_unit_circle_l520_52095

theorem cos_double_angle_unit_circle (α : ℝ) (y₀ : ℝ) :
  (1/2)^2 + y₀^2 = 1 → Real.cos (2 * α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_unit_circle_l520_52095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_range_of_m_l520_52030

-- Define set A
def A : Set ℝ := {x | x ∈ Set.Icc (-2 : ℝ) 5}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | x^2 - 3*m*x + 2*m^2 - m - 1 < 0}

-- Define the natural numbers in A
def A_nat : Set ℕ := {x | x ∈ Set.Icc 0 5}

-- Theorem 1: Number of non-empty proper subsets of A_nat
theorem number_of_subsets : Finset.card (Finset.powerset (Finset.Icc 0 5) \ {∅, Finset.Icc 0 5}) = 62 := by sorry

-- Theorem 2: Range of m for A ∩ B = B
theorem range_of_m (m : ℝ) : A ∩ B m = B m ↔ (m ∈ Set.Icc (-1 : ℝ) 2 ∨ m = -2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_range_of_m_l520_52030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l520_52014

-- Define set A
def A : Set ℝ := {x | -12 < 4 * x ∧ 4 * x < 8}

-- Define set B
def B : Set ℝ := {x | x ≠ 0 ∧ 1 / x < 1}

-- Define the expected intersection
def expected_intersection : Set ℝ := Set.Ioo (-3) 0 ∪ Set.Ioo 1 2

-- Theorem statement
theorem set_intersection_theorem : A ∩ B = expected_intersection := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l520_52014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_uniqueness_l520_52097

theorem remainder_uniqueness (p : ℕ) (a b : ℤ) (hp : Nat.Prime p) (ha : ¬(p : ℤ) ∣ a) (hb : ¬(p : ℤ) ∣ b) :
  ∃! r : ℕ, r < p ∧ ∃ (k : ℤ), a = k * (p : ℤ) + r :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_uniqueness_l520_52097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_blend_price_l520_52082

/-- Calculates the price per pound of a coffee blend -/
noncomputable def blendPrice (price1 price2 : ℝ) (weight1 weight2 : ℝ) : ℝ :=
  (price1 * weight1 + price2 * weight2) / (weight1 + weight2)

theorem coffee_blend_price :
  let price1 := (9 : ℝ)  -- Price of first blend ($/lb)
  let price2 := (8 : ℝ)  -- Price of second blend ($/lb)
  let totalWeight := (20 : ℝ)  -- Total weight of new blend (lb)
  let weight2 := (12 : ℝ)  -- Weight of second blend used (lb)
  let weight1 := totalWeight - weight2  -- Weight of first blend used (lb)
  blendPrice price1 price2 weight1 weight2 = 8.4
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_blend_price_l520_52082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_g_eq_5_has_three_solutions_l520_52004

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 1 then -2 * x + 6 else 3 * x - 7

-- State the theorem
theorem g_g_eq_5_has_three_solutions :
  ∃ (s : Finset ℝ), (∀ x ∈ s, g (g x) = 5) ∧ (Finset.card s = 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_g_eq_5_has_three_solutions_l520_52004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_one_f_is_monotone_increasing_k_upper_bound_l520_52053

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m - 2 / (2^x + 1)

-- Theorem 1
theorem odd_function_implies_m_equals_one (m : ℝ) :
  (∀ x, f m x = -f m (-x)) → m = 1 := by sorry

-- Theorem 2
theorem f_is_monotone_increasing (m : ℝ) :
  Monotone (f m) := by sorry

-- Theorem 3
theorem k_upper_bound (k : ℝ) :
  (∀ x, f 1 x = -f 1 (-x)) →
  (∀ x, f 1 (k * 3^x) + f 1 (3^x - 9^x - 2) < 0) →
  k < 2 * Real.sqrt 2 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_one_f_is_monotone_increasing_k_upper_bound_l520_52053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_solutions_l520_52087

theorem no_positive_integer_solutions :
  ¬∃ (x y n : ℕ), x > 0 ∧ y > 0 ∧ n > 0 ∧ x^2 + y^2 + 40 = 2^n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_solutions_l520_52087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l520_52007

theorem min_value_of_function (x : ℝ) 
  (h : π / 4 ≤ x ∧ x ≤ π / 2) : 
  ∃ (m : ℝ), m = 2 ∧ 
  (∀ y ∈ Set.Icc (π / 4) (π / 2), 
    2 * (Real.sin (π / 4 + y))^2 - Real.sqrt 3 * Real.cos (2 * y) ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l520_52007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farey_sequence_property_l520_52080

def is_irreducible (a b : ℤ) : Prop := Int.gcd a b = 1

def in_farey_sequence (a b n : ℤ) : Prop :=
  0 < a ∧ a < b ∧ b ≤ n ∧ is_irreducible a b

theorem farey_sequence_property (n a b c d : ℤ) :
  in_farey_sequence a b n →
  in_farey_sequence c d n →
  (a : ℚ) / b < (c : ℚ) / d →
  (∀ x y : ℤ, in_farey_sequence x y n → (a : ℚ) / b < (x : ℚ) / y → (x : ℚ) / y < (c : ℚ) / d → False) →
  |b * c - a * d| = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farey_sequence_property_l520_52080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l520_52069

theorem tan_ratio_from_sin_sum_diff (p q : ℝ) 
  (h1 : Real.sin (p + q) = 5/8) 
  (h2 : Real.sin (p - q) = 3/8) : 
  Real.tan p / Real.tan q = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l520_52069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_centroid_distance_sum_l520_52078

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y^2 = x -/
def OnParabola (p : Point) : Prop :=
  p.y^2 = p.x

/-- The focus of the parabola y^2 = x -/
noncomputable def FocusOfParabola : Point :=
  { x := 1/4, y := 0 }

/-- Distance between two points -/
noncomputable def Distance (p1 p2 : Point) : ℝ :=
  |p1.x - p2.x|

/-- Centroid of a triangle -/
noncomputable def Centroid (a b c : Point) : Point :=
  { x := (a.x + b.x + c.x) / 3,
    y := (a.y + b.y + c.y) / 3 }

/-- The main theorem -/
theorem parabola_focus_centroid_distance_sum
  (a b c : Point)
  (ha : OnParabola a)
  (hb : OnParabola b)
  (hc : OnParabola c)
  (hf : FocusOfParabola = Centroid a b c) :
  Distance a FocusOfParabola + Distance b FocusOfParabola + Distance c FocusOfParabola = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_centroid_distance_sum_l520_52078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l520_52012

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 3)

theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x : ℝ, f ω (x + Real.pi / (2 * ω)) = f ω x) : 
  ω = 2 ∧ 
  Set.Icc (-Real.sqrt 3) 2 = Set.image (f 2) (Set.Icc (-Real.pi / 6) (Real.pi / 2)) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l520_52012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_norm_bound_l520_52041

open RealInnerProductSpace NormedSpace

/-- Given two non-zero vectors in a real normed space satisfying certain conditions,
    the sum of their norms is bounded above. -/
theorem vector_norm_bound {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (m n : V) 
    (hm : m ≠ 0) (hn : n ≠ 0) (h1 : ‖m‖ = 2) (h2 : ‖m + 2 • n‖ = 2) :
    ‖n‖ + ‖2 • m + n‖ ≤ 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_norm_bound_l520_52041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2023_of_8_19th_l520_52086

def decimal_expansion (n d : ℕ) : List ℕ := sorry

theorem digit_2023_of_8_19th : 
  let expansion := decimal_expansion 8 19
  (expansion.get? 2022).isSome ∧ (expansion.get? 2022).get! = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2023_of_8_19th_l520_52086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_divisors_1800_l520_52034

def σ (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum (λ x => x^2)

theorem sum_squares_divisors_1800 : σ 1800 = 5035485 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_divisors_1800_l520_52034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_is_correct_l520_52022

/-- Definition of a line -/
structure Line where
  -- We'll use a placeholder definition for Line
  dummy : Unit

/-- Definition of parallel lines -/
def parallel (l₁ l₂ : Line) : Prop := sorry

/-- Definition of interior alternate angles -/
def interior_alternate_angles_equal (l₁ l₂ : Line) : Prop := sorry

/-- The original proposition -/
def original_proposition : Prop :=
  ∀ l₁ l₂ : Line, parallel l₁ l₂ → interior_alternate_angles_equal l₁ l₂

/-- The inverse proposition -/
def inverse_proposition : Prop :=
  ∀ l₁ l₂ : Line, interior_alternate_angles_equal l₁ l₂ → parallel l₁ l₂

/-- Theorem stating that the inverse proposition is correct -/
theorem inverse_proposition_is_correct :
  inverse_proposition = (¬ original_proposition → False) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_is_correct_l520_52022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_some_students_not_chess_members_l520_52001

-- Define the sets
variable (Student Athlete ChessClubMember : Type)

-- Define the membership relation
variable (IsAthlete : Student → Prop)
variable (IsChessMember : Student → Prop)

-- Define the conditions
variable (some_students_not_athletes : ∃ s : Student, ¬IsAthlete s)
variable (all_chess_members_are_athletes : ∀ s : Student, IsChessMember s → IsAthlete s)

-- Theorem to prove
theorem some_students_not_chess_members :
  ∃ s : Student, ¬IsChessMember s :=
by
  -- Extract a student who is not an athlete
  obtain ⟨s, not_athlete⟩ := some_students_not_athletes
  -- Show that this student is not a chess club member
  use s
  intro is_chess_member
  -- If they were a chess member, they would be an athlete
  have athlete := all_chess_members_are_athletes s is_chess_member
  -- But we know they're not an athlete
  contradiction


end NUMINAMATH_CALUDE_ERRORFEEDBACK_some_students_not_chess_members_l520_52001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_105_106_l520_52083

def g (x : ℤ) : ℤ := x^2 - x + 2502

theorem gcd_g_105_106 : Int.gcd (g 105) (g 106) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_105_106_l520_52083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_read_l520_52025

def reading_speed : ℕ := 120
def pages_per_book : ℕ := 360
def available_time : ℕ := 8

theorem books_read : 
  (reading_speed * available_time) / pages_per_book = 2 := by
  norm_num
  rfl

#eval (reading_speed * available_time) / pages_per_book

end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_read_l520_52025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_periodicity_l520_52088

/-- Represents the configuration of cards in the game -/
def Configuration (n : ℕ) := Fin n → Fin 3 → Fin (3*n)

/-- The game setup and rules -/
structure Game (n : ℕ) (h : n ≥ 3) where
  initial_config : Configuration n
  next_config : Configuration n → Configuration n
  T : ℕ → Configuration n

/-- The theorem to be proved -/
theorem game_periodicity (n : ℕ) (h : n ≥ 3) (game : Game n h) :
  (∃ m : ℕ, ∀ r ≥ m, game.T r = game.T (r + n)) ∧
  (∃ m : ℕ, m = n - 1 ∧ game.T m = game.T (m + n) ∧
    ∀ k < m, game.T k ≠ game.T (k + n)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_periodicity_l520_52088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_per_dozen_eggs_l520_52062

/-- Calculates the price per dozen eggs given the farm's parameters -/
theorem price_per_dozen_eggs 
  (num_chickens : ℕ) 
  (eggs_per_chicken_per_week : ℕ) 
  (total_revenue : ℚ) 
  (num_weeks : ℕ) 
  (h1 : num_chickens = 10)
  (h2 : eggs_per_chicken_per_week = 6)
  (h3 : total_revenue = 20)
  (h4 : num_weeks = 2) :
  (total_revenue / ((num_chickens * eggs_per_chicken_per_week * num_weeks) / 12 : ℚ)) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_per_dozen_eggs_l520_52062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l520_52093

theorem trigonometric_identities (α β γ : ℝ) 
  (h : α + β + γ = Real.pi) : 
  (Real.sin α * Real.cos β * Real.cos γ + 
   Real.sin β * Real.cos α * Real.cos γ + 
   Real.sin γ * Real.cos α * Real.cos β = 
   Real.sin α * Real.sin β * Real.sin γ) ∧
  ((Real.tan (π/2 - β/2) + Real.tan (π/2 - γ/2)) / 
   (Real.tan (π/2 - α/2) + Real.tan (π/2 - γ/2)) = 
   Real.sin α / Real.sin β) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l520_52093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_one_is_local_minimum_l520_52076

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x)

theorem negative_one_is_local_minimum :
  ∃ δ > 0, ∀ x ∈ Set.Ioo (-1 - δ) (-1 + δ), f (-1) ≤ f x := by
  sorry

#check negative_one_is_local_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_one_is_local_minimum_l520_52076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l520_52035

/-- Two circles are externally tangent. -/
def externally_tangent (r₁ r₂ : ℝ) : Prop :=
  sorry

/-- A chord is a common external tangent of two circles with respect to a larger circle. -/
def is_common_external_tangent (chord r₁ r₂ r₃ : ℝ) : Prop :=
  sorry

/-- Given three circles with radii 4, 8, and 12, where the circles of radius 4 and 8 are externally
    tangent to each other and both are externally tangent to the circle of radius 12, the square of
    the length of the chord of the circle with radius 12 that is a common external tangent of the
    other two circles is equal to 3584/9. -/
theorem chord_length_squared (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : r₃ = 12)
    (h_tangent_small : externally_tangent r₁ r₂)
    (h_tangent_large₁ : externally_tangent r₁ r₃)
    (h_tangent_large₂ : externally_tangent r₂ r₃)
    (chord : ℝ) (h_chord : is_common_external_tangent chord r₁ r₂ r₃) :
    chord ^ 2 = 3584 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l520_52035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_eq_A_l520_52045

noncomputable def A (a : ℝ) : Set ℝ := {x | a * x^2 + 3 * x - 2 * a = 0}

def B : Set ℝ := {x | 2 * x^2 - 5 * x - 42 ≤ 0}

theorem intersection_eq_A (a : ℝ) (ha : a ≠ 0) :
  A a ∩ B = A a ↔ a ∈ Set.Iic (-9/17) ∪ Set.Ici (42/41) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_eq_A_l520_52045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_component_product_l520_52094

/-- Given two vectors a and b in R³, if they are parallel and have specific components,
    then the product of their first and third components is -3. -/
theorem parallel_vectors_component_product (m r : ℝ) : 
  let a : ℝ × ℝ × ℝ := (m, 5, -1)
  let b : ℝ × ℝ × ℝ := (3, 1, r)
  (∃ (k : ℝ), a = k • b) → m * r = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_component_product_l520_52094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l520_52043

noncomputable def f (x : ℝ) : ℝ := Real.sin x - 4 * (Real.sin (x/2))^3 * Real.cos (x/2)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l520_52043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_selling_price_is_675_l520_52021

/-- The final selling price of an item, given its original cost, loss percentage, and sales tax percentage. -/
noncomputable def finalSellingPrice (originalCost : ℚ) (lossPercentage : ℚ) (salesTaxPercentage : ℚ) : ℚ :=
  let sellingPriceAfterLoss := originalCost * (1 - lossPercentage / 100)
  let salesTax := originalCost * (salesTaxPercentage / 100)
  sellingPriceAfterLoss + salesTax

/-- Theorem stating that the final selling price of an item with given conditions is 675. -/
theorem final_selling_price_is_675 :
  finalSellingPrice 750 20 10 = 675 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_selling_price_is_675_l520_52021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l520_52068

open Real

-- Define the setup
noncomputable def angle_XOY : ℝ := 90 * (π / 180)  -- Convert 90° to radians
noncomputable def angle_XOP : ℝ := 30 * (π / 180)  -- Convert 30° to radians
def OP : ℝ := 1

-- Define the function to be maximized
noncomputable def f (M N : ℝ × ℝ) : ℝ :=
  sqrt ((M.1)^2 + (M.2)^2) +  -- OM
  sqrt ((N.1)^2 + (N.2)^2) -  -- ON
  sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)  -- MN

-- State the theorem
theorem max_value_theorem :
  ∃ (max_value : ℝ), 
    (∀ M N : ℝ × ℝ, f M N ≤ max_value) ∧ 
    (max_value = sqrt 3 + 1 - (12 : ℝ)^(1/4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l520_52068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_increasing_l520_52042

noncomputable def f (x : ℝ) := Real.log (abs x) / Real.log 2

theorem f_is_even_and_increasing : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by
  sorry

#check f_is_even_and_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_increasing_l520_52042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_range_l520_52057

def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - 2)^2 = 9

def point : ℝ × ℝ := (-2, 3)

def has_two_tangents (a : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ × ℝ), t₁ ≠ t₂ ∧ circle_equation a t₁.1 t₁.2 ∧ circle_equation a t₂.1 t₂.2

noncomputable def range_of_a : Set ℝ := {a | a < -2 - 2 * Real.sqrt 2 ∨ a > -2 + 2 * Real.sqrt 2}

theorem tangent_range :
  ∀ a : ℝ, has_two_tangents a ↔ a ∈ range_of_a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_range_l520_52057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l520_52077

/-- Represents the time needed for a worker to complete the job alone -/
structure Worker where
  time : ℚ
  time_pos : time > 0

/-- Represents a group of workers -/
structure WorkerGroup where
  alpha : Worker
  delta : Worker
  omega : Worker
  zeta : Worker
  all_together : ℚ
  all_together_pos : all_together > 0

/-- The conditions of the problem -/
def job_conditions (wg : WorkerGroup) : Prop :=
  wg.all_together = wg.alpha.time - 4 ∧
  wg.all_together = wg.delta.time - 2 ∧
  wg.all_together = wg.omega.time - 15 ∧
  wg.zeta.time = wg.omega.time / 2

/-- The time needed for Alpha and Delta to complete the job together -/
noncomputable def time_alpha_delta (wg : WorkerGroup) : ℚ :=
  1 / (1 / wg.alpha.time + 1 / wg.delta.time)

/-- The main theorem to prove -/
theorem job_completion_time (wg : WorkerGroup) (h : job_conditions wg) :
  time_alpha_delta wg = 323 / 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l520_52077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l520_52008

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m + Real.exp (x * Real.log 2)

-- Define the locally odd property
def locally_odd (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x ∈ Set.Icc a b, f (-x) = -f x

-- Define proposition p
def p (m : ℝ) : Prop := locally_odd (f m) (-1) 1

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (5*m + 1)*x₁ + 1 = 0 ∧ x₂^2 + (5*m + 1)*x₂ + 1 = 0

-- State the theorem
theorem range_of_m :
  (∀ m : ℝ, ¬(p m ∧ q m)) ∧ (∀ m : ℝ, p m ∨ q m) →
  ∀ m : ℝ, m < -5/4 ∨ (-1 < m ∧ m < -3/5) ∨ m > 1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l520_52008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l520_52026

def charter_cost : ℤ := 30000

def max_people : ℕ := 75

def fee (x : ℕ) : ℤ :=
  if 1 ≤ x ∧ x ≤ 30 then 1800
  else if 30 < x ∧ x ≤ 75 then -20 * x + 2400
  else 0

def profit (x : ℕ) : ℤ :=
  if 1 ≤ x ∧ x ≤ 30 then (fee x * x) - charter_cost
  else if 30 < x ∧ x ≤ 75 then (fee x * x) - charter_cost
  else 0

theorem max_profit :
  ∃ (x : ℕ), x ≤ max_people ∧
    (∀ (y : ℕ), y ≤ max_people → profit y ≤ profit x) ∧
    x = 60 ∧ profit x = 42000 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l520_52026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_assignment_proof_l520_52032

-- Define the set of possible grades
inductive Grade : Type where
  | three : Grade
  | four : Grade
  | five : Grade

-- Define the set of students
inductive Student : Type where
  | alekseev : Student
  | vasiliev : Student
  | sergeev : Student

-- Define a function type that assigns grades to students
def GradeAssignment := Student → Grade

-- Define the constraints based on the teacher's initial statements
def teacherStatements (g : GradeAssignment) : Prop :=
  g Student.sergeev ≠ Grade.five ∧
  g Student.vasiliev ≠ Grade.four ∧
  g Student.alekseev = Grade.four

-- Define the condition that all students have different grades
def allDifferentGrades (g : GradeAssignment) : Prop :=
  g Student.alekseev ≠ g Student.vasiliev ∧
  g Student.alekseev ≠ g Student.sergeev ∧
  g Student.vasiliev ≠ g Student.sergeev

-- Define the correct grade assignment
def correctAssignment (g : GradeAssignment) : Prop :=
  g Student.vasiliev = Grade.five ∧
  g Student.sergeev = Grade.three ∧
  g Student.alekseev = Grade.four

-- Theorem statement
theorem grade_assignment_proof :
  ∃ (g : GradeAssignment),
    allDifferentGrades g ∧
    (teacherStatements g = false) ∧
    correctAssignment g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_assignment_proof_l520_52032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_seven_800_l520_52036

def contains_seven (n : Nat) : Bool :=
  let digits := n.repr.data
  digits.any (· == '7')

def count_numbers_with_seven (upper_bound : Nat) : Nat :=
  (List.range upper_bound).filter contains_seven |>.length

theorem count_numbers_with_seven_800 : count_numbers_with_seven 800 = 152 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_seven_800_l520_52036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_asymptotes_l520_52071

/-- The equation of the first hyperbola -/
def hyperbola1 (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1

/-- The equation of the second hyperbola -/
def hyperbola2 (x y : ℝ) : Prop := x^2 / 4 - y^2 / 16 = 1

/-- The asymptotes of a hyperbola -/
def asymptotes (f : ℝ → ℝ → Prop) : Set (ℝ → ℝ) :=
  {g | ∃ (k : ℝ), (∀ x, g x = k * x) ∨ (∀ x, g x = -k * x)}

/-- Theorem stating that the two hyperbolas have the same asymptotes -/
theorem hyperbolas_same_asymptotes :
  asymptotes hyperbola1 = asymptotes hyperbola2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_asymptotes_l520_52071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_theorem_l520_52070

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + 3

-- State the theorem
theorem function_value_theorem (a : ℝ) (h1 : a > 0) :
  (∀ x, f (2^x) = x + 3) → f a = 5 → a = 4 := by
  intro h2 h3
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- Example usage of the theorem (optional)
example (a : ℝ) (h1 : a > 0) (h2 : ∀ x, f (2^x) = x + 3) (h3 : f a = 5) : a = 4 := by
  exact function_value_theorem a h1 h2 h3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_theorem_l520_52070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_iff_c_eq_zero_l520_52091

noncomputable section

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x in the domain of f -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = 1/x + cx^2 where c is a constant -/
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := 1/x + c*x^2

/-- Theorem: c = 0 is the necessary and sufficient condition for f(x) to be an odd function -/
theorem odd_function_iff_c_eq_zero (c : ℝ) :
  IsOdd (f c) ↔ c = 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_iff_c_eq_zero_l520_52091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_3_root_3_l520_52058

-- Define the vertices of the triangle
def A : ℝ × ℝ × ℝ := (1, 8, 11)
def B : ℝ × ℝ × ℝ := (0, 7, 9)
def C : ℝ × ℝ × ℝ := (-3, 10, 9)

-- Define a function to calculate the distance between two points in 3D space
noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, z₁) := p
  let (x₂, y₂, z₂) := q
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

-- Define the side lengths of the triangle
noncomputable def AB : ℝ := distance A B
noncomputable def BC : ℝ := distance B C
noncomputable def AC : ℝ := distance A C

-- Theorem statement
theorem triangle_area_is_3_root_3 : 
  (1/2 : ℝ) * AB * BC = 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_3_root_3_l520_52058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_N_for_Q_less_than_seven_ninths_l520_52019

def is_positive_multiple_of_four (n : ℕ) : Prop :=
  n > 0 ∧ n % 4 = 0

noncomputable def Q (N : ℕ) : ℚ :=
  sorry  -- Definition of Q(N) based on the problem description

theorem least_N_for_Q_less_than_seven_ninths :
  ∃ N : ℕ, is_positive_multiple_of_four N ∧
    Q N < 7/9 ∧
    (∀ M : ℕ, is_positive_multiple_of_four M ∧ M < N → Q M ≥ 7/9) ∧
    N = 76 := by
  sorry

#check least_N_for_Q_less_than_seven_ninths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_N_for_Q_less_than_seven_ninths_l520_52019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₂_properties_l520_52074

-- Define the set A
noncomputable def A : Set (ℝ → ℝ) :=
  {f | ∀ x, x ≥ 0 → f x ∈ Set.Icc (-2) 4 ∧ StrictMono f}

-- Define the function f₂
noncomputable def f₂ (x : ℝ) : ℝ := 4 - 6 * (1/2)^x

-- Theorem statement
theorem f₂_properties :
  f₂ ∈ A ∧ ∀ x, x ≥ 0 → f₂ x + f₂ (x + 2) < 2 * f₂ (x + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₂_properties_l520_52074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l520_52017

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define the foci
def left_focus (a b : ℝ) : ℝ × ℝ :=
  (-Real.sqrt (a^2 - b^2), 0)

def right_focus (a b : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 - b^2), 0)

-- Define a point on the ellipse
def point_on_ellipse : ℝ × ℝ :=
  (2, Real.sqrt 3)

-- Define the perpendicular bisector condition
def perp_bisector_condition (a b : ℝ) : Prop :=
  let f₁ := left_focus a b
  let f₂ := right_focus a b
  let p := point_on_ellipse
  (f₂.1 - p.1)^2 + (f₂.2 - p.2)^2 = (f₁.1 - p.1)^2 + (f₁.2 - p.2)^2

-- Define the intersection line
def intersection_line (k m : ℝ) : ℝ → ℝ :=
  λ x ↦ k * x + m

-- Main theorem
theorem ellipse_theorem (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  (eccentricity a b = Real.sqrt 2 / 2) →
  (perp_bisector_condition a b) →
  (∀ k m : ℝ, ∃ x₁ x₂ : ℝ,
    ellipse a b x₁ (intersection_line k m x₁) ∧
    ellipse a b x₂ (intersection_line k m x₂) ∧
    x₁ ≠ x₂ →
    (∃ α β : ℝ,
      α + β = Real.pi ∧
      -- Here we would define the angle condition, but it's complex to express in Lean
      intersection_line k m 2 = 0)) →
  (ellipse a b = λ x y ↦ x^2 / 2 + y^2 = 1) ∧
  (∀ k m : ℝ, ∃ x₁ x₂ : ℝ,
    ellipse a b x₁ (intersection_line k m x₁) ∧
    ellipse a b x₂ (intersection_line k m x₂) ∧
    x₁ ≠ x₂ →
    intersection_line k m 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l520_52017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_l520_52084

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

def equation (x : ℝ) : Prop := ⌊x⌋ = 7 + 150 * (frac x)

theorem largest_solution :
  ∃ (x : ℝ), equation x ∧ x = 156.9933 ∧ ∀ (y : ℝ), equation y → y ≤ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_l520_52084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_planes_intersection_l520_52000

-- Define the types for lines and planes
variable (Line Plane Point : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpPlane : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- Define the "in plane" relation
variable (inPlane : Line → Plane → Prop)

-- Define the intersection of planes
variable (intersect : Plane → Plane → Prop)

-- Define the intersection line of two planes
variable (intersectionLine : Plane → Plane → Line)

-- Define a predicate for a point being on a line
variable (onLine : Point → Line → Prop)

-- State the theorem
theorem skew_lines_planes_intersection
  (m n l : Line) (α β : Plane)
  (h1 : ¬ para m n ∧ ¬ (∃ p : Point, onLine p m ∧ onLine p n)) -- m and n are skew
  (h2 : perpPlane m α)
  (h3 : perpPlane n β)
  (h4 : perp l m)
  (h5 : perp l n)
  (h6 : ¬ inPlane l α)
  (h7 : ¬ inPlane l β) :
  intersect α β ∧ para l (intersectionLine α β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_planes_intersection_l520_52000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_jogger_l520_52038

/-- The time taken for a train to pass a jogger given their speeds and initial positions --/
theorem train_passes_jogger (jogger_speed train_speed : ℝ) (train_length initial_distance : ℝ) : 
  jogger_speed = 8 * (1000 / 3600) →
  train_speed = 50 * (1000 / 3600) →
  train_length = 180 →
  initial_distance = 360 →
  ∃ t : ℝ, (t ≥ 46.24 ∧ t ≤ 46.26) ∧ 
    t = (initial_distance + train_length) / (train_speed - jogger_speed) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_jogger_l520_52038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_sales_revenue_l520_52011

noncomputable def f (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 40 then (1/4 : ℝ) * t + 10
  else if 40 < t ∧ t ≤ 90 then (t : ℝ) - 20
  else 0

noncomputable def g (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 40 then -10 * t + 630
  else if 40 < t ∧ t ≤ 90 then -(1/10 : ℝ) * (t : ℝ)^2 + 10 * t - 10
  else 0

noncomputable def S (t : ℕ) : ℝ := f t * g t

theorem max_daily_sales_revenue :
  ∃ t : ℕ, 1 ≤ t ∧ t ≤ 90 ∧ S t = 53045/8 ∧ ∀ t' : ℕ, 1 ≤ t' ∧ t' ≤ 90 → S t' ≤ S t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_daily_sales_revenue_l520_52011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_sum_of_distances_l520_52051

/-- A triangle with side lengths 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_eq : a = 3
  b_eq : b = 4
  c_eq : c = 5
  right_angle : a^2 + b^2 = c^2

/-- The sum of distances from P to each side of the triangle -/
def sumOfDistances (T : RightTriangle) (P : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the greatest possible sum of distances -/
theorem greatest_sum_of_distances (T : RightTriangle) :
  ∀ P, sumOfDistances T P ≤ 12/5 ∧ ∃ P, sumOfDistances T P = 12/5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_sum_of_distances_l520_52051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_4_minus_alpha_l520_52063

theorem cos_pi_4_minus_alpha (α : ℝ) :
  Real.sin (π / 4 + α) = 2 / 3 → Real.cos (π / 4 - α) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_4_minus_alpha_l520_52063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_QT_length_is_3825_div_481_l520_52072

/-- Triangle PQR with points S and T on QR -/
structure TrianglePQR where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- Length of RS -/
  RS : ℝ
  /-- Point S is on QR -/
  S_on_QR : RS ≤ QR
  /-- Point T is on QR -/
  T_on_QR : ℝ
  /-- Angle PQT equals angle PSR -/
  angle_equality : T_on_QR ≤ QR

/-- The length of QT in the given triangle configuration -/
noncomputable def QT_length (t : TrianglePQR) : ℝ :=
  3825 / 481

/-- Theorem stating that QT length is 3825/481 for the given triangle configuration -/
theorem QT_length_is_3825_div_481 (t : TrianglePQR) 
    (h1 : t.PQ = 15) 
    (h2 : t.QR = 17) 
    (h3 : t.PR = 16) 
    (h4 : t.RS = 7) :
    QT_length t = 3825 / 481 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_QT_length_is_3825_div_481_l520_52072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_minimum_value_increasing_condition_l520_52033

-- Define the function f(x) = x^2 - a*ln(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * Real.log x

-- Theorem 1: Tangent line when a = 1
theorem tangent_line_at_one :
  (deriv (f 1)) 1 = 1 ∧ f 1 1 = 1 := by sorry

-- Theorem 2: Minimum value for a > 0
theorem minimum_value (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x > 0 ∧ (deriv (f a)) x = 0 ∧
  (f a) x = a/2 - a/2 * Real.log (a/2) ∧
  ∀ y > 0, (f a) y ≥ (f a) x := by sorry

-- Theorem 3: Condition for f to be increasing on (2, +∞)
theorem increasing_condition (a : ℝ) :
  (∀ x > 2, (deriv (f a)) x ≥ 0) ↔ a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_minimum_value_increasing_condition_l520_52033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l520_52090

noncomputable def circle1_center : ℝ × ℝ := (-6, 2)
noncomputable def circle1_radius : ℝ := 12
noncomputable def circle2_center : ℝ × ℝ := (3, 9)
noncomputable def circle2_radius : ℝ := Real.sqrt 65

def intersection_line (x y : ℝ) : Prop := x + y = 6

theorem intersection_line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ((x₁ + 6)^2 + (y₁ - 2)^2 = circle1_radius^2) ∧
    ((x₁ - 3)^2 + (y₁ - 9)^2 = circle2_radius^2) ∧
    ((x₂ + 6)^2 + (y₂ - 2)^2 = circle1_radius^2) ∧
    ((x₂ - 3)^2 + (y₂ - 9)^2 = circle2_radius^2) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    intersection_line x₁ y₁ ∧
    intersection_line x₂ y₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l520_52090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l520_52055

/-- A function f(x) = e^(kx) where k is positive and x is real. -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp (k * x)

/-- Theorem stating properties of the function f. -/
theorem f_properties (k : ℝ) (hk : k > 0) :
  (∀ x : ℝ, f k x > 0) ∧
  (f k 0 = 1) ∧
  (∀ x : ℝ, f k (-x) = 1 / f k x) ∧
  (∀ x : ℝ, f k x = (f k (4 * x)) ^ (1/4)) ∧
  (∀ x y : ℝ, y > x → f k y > f k x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l520_52055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_a_always_one_l520_52066

def a (n : ℕ) : ℚ := (5^n - 1) / 4

theorem gcd_a_always_one (n : ℕ) :
  Nat.gcd (Int.natAbs ((a n).num)) (Int.natAbs ((a (n + 1)).num)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_a_always_one_l520_52066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_b_l520_52039

/-- The weight of person a -/
def A : ℝ := sorry

/-- The weight of person b -/
def B : ℝ := sorry

/-- The weight of person c -/
def C : ℝ := sorry

/-- The average weight of a, b, and c is 45 kg -/
axiom avg_abc : (A + B + C) / 3 = 45

/-- The average weight of a and b is 40 kg -/
axiom avg_ab : (A + B) / 2 = 40

/-- The average weight of b and c is 43 kg -/
axiom avg_bc : (B + C) / 2 = 43

/-- Theorem: Given the conditions, the weight of b is 31 kg -/
theorem weight_of_b : B = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_b_l520_52039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l520_52049

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 1) / (x^2 - 4*x + 3)

-- Define the solution set
def solution_set : Set ℝ :=
  {x | x > 3 ∨ (1 < x ∧ x ≤ 2) ∨ x ≤ 1/2}

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | f x ≥ -1 ∧ x ≠ 1 ∧ x ≠ 3} = solution_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l520_52049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_l520_52060

noncomputable section

-- Define the polynomial
def f (x : ℝ) : ℝ := Real.sqrt 100 * x^3 - 201 * x^2 + 3

-- Define the roots
variable (x₁ x₂ x₃ : ℝ)

-- The roots satisfy the equation
axiom root_eq₁ : f x₁ = 0
axiom root_eq₂ : f x₂ = 0
axiom root_eq₃ : f x₃ = 0

-- The roots are ordered
axiom root_order : x₁ < x₂ ∧ x₂ < x₃

-- Theorem to prove
theorem root_product : x₂ * (x₁ + x₃) = 398 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_l520_52060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l520_52047

-- Define the shapes
inductive Shape
  | Triangle
  | Circle
  | Square

-- Define the position of a shape on the circle
structure Position :=
  (angle : Real)

-- Define the figure as a function mapping shapes to their positions
def Figure := Shape → Position

-- Define the rotation function
noncomputable def rotate (fig : Figure) (angle : Real) : Figure :=
  λ s => ⟨(fig s).angle + angle⟩

-- Define the original configuration
noncomputable def original_config : Figure :=
  λ s => match s with
    | Shape.Triangle => ⟨0⟩
    | Shape.Circle => ⟨2 * Real.pi / 3⟩
    | Shape.Square => ⟨4 * Real.pi / 3⟩

-- Define the rotated configuration
noncomputable def rotated_config : Figure := rotate original_config (150 * Real.pi / 180)

-- Theorem stating the result of the rotation
theorem rotation_result :
  ∀ (s : Shape),
    ∃ (s1 s2 : Shape),
      s ≠ s1 ∧ s ≠ s2 ∧ s1 ≠ s2 ∧
      (rotated_config s).angle > (original_config s1).angle ∧
      (rotated_config s).angle < (original_config s2).angle :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l520_52047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_marks_correct_l520_52028

/-- Calculates the maximum marks given the passing threshold, student score, and failure margin. -/
def max_marks_calculation (passing_threshold : ℚ) (student_score : ℕ) (failure_margin : ℕ) : ℕ :=
  let passing_score := student_score + failure_margin
  let max_marks : ℚ := passing_score / passing_threshold
  ⌈max_marks⌉.toNat

#eval max_marks_calculation (45/100) 155 75

theorem max_marks_correct :
  max_marks_calculation (45/100) 155 75 = 512 := by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_marks_correct_l520_52028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l520_52092

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + Real.exp (2 * x) - m

noncomputable def f_derivative (m : ℝ) (x : ℝ) : ℝ := 1 + 2 * Real.exp (2 * x)

noncomputable def tangent_line (m : ℝ) (x : ℝ) : ℝ := (f_derivative m 0) * x + f m 0

noncomputable def x_intercept (m : ℝ) : ℝ := (m - 1) / (f_derivative m 0)

noncomputable def y_intercept (m : ℝ) : ℝ := f m 0

noncomputable def triangle_area (m : ℝ) : ℝ := (1/2) * abs (x_intercept m * y_intercept m)

theorem tangent_triangle_area (m : ℝ) : 
  triangle_area m = 1/6 → m = 2 ∨ m = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l520_52092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_color_probability_l520_52059

theorem cube_color_probability : 
  let num_faces : ℕ := 6
  let num_colors : ℕ := 3
  let total_arrangements : ℕ := num_colors^num_faces
  let favorable_arrangements : ℕ := 
    -- All faces same color
    num_colors + 
    -- Five faces same color, one different
    (num_colors * (num_faces.choose 5)) +
    -- Four vertical faces same, top and bottom different from each other and sides
    (Nat.choose 3 1 * Nat.choose 2 1)
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_color_probability_l520_52059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_three_distinct_zeros_l520_52018

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x - a

-- State the theorem
theorem range_of_a_for_three_distinct_zeros :
  ∀ a : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  -9 < a ∧ a < 5/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_three_distinct_zeros_l520_52018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_theorem_l520_52067

/-- A function that checks if all positive integers less than or equal to n 
    and relatively prime to n are pairwise coprime -/
def satisfies_property (n : ℕ+) : Prop :=
  ∀ a b : ℕ+, a ≤ n → b ≤ n → Nat.Coprime a.val n.val → Nat.Coprime b.val n.val → 
    a ≠ b → Nat.Coprime a.val b.val

/-- The set of all positive integers that satisfy the property -/
def special_set : Set ℕ+ :=
  {1, 2, 3, 4, 6, 8, 12, 18, 24, 30}

/-- Theorem stating the equivalence between satisfying the property
    and being in the special set -/
theorem characterization_theorem (n : ℕ+) :
  satisfies_property n ↔ n ∈ special_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_theorem_l520_52067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l520_52061

theorem triangle_area (line : ℝ → ℝ) : 
  (∀ x, line x = 3 * x - 5) →
  (1 / 2 : ℝ) * (5 / 3) * |(-5)| = 25 / 6 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l520_52061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_of_g_7_l520_52079

-- Define the functions u and g
noncomputable def u (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)
noncomputable def g (x : ℝ) : ℝ := 7 - u x

-- State the theorem
theorem u_of_g_7 : u (g 7) = Real.sqrt (30 - 4 * Real.sqrt 30) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_of_g_7_l520_52079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l520_52050

/-- The function g(x) = sin(log x) -/
noncomputable def g (x : ℝ) : ℝ := Real.sin (Real.log x)

/-- Theorem: There are infinitely many values of x in (0, 1) where g(x) = 1 -/
theorem infinitely_many_solutions :
  ∃ (S : Set ℝ), Set.Infinite S ∧ S ⊆ Set.Ioo 0 1 ∧ ∀ x ∈ S, g x = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l520_52050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_condition_l520_52003

/-- A parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A point on a parabola -/
structure ParabolaPoint (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- The point E with coordinates (-lambda, 0) -/
structure PointE where
  lambda : ℝ
  h_lambda_pos : lambda > 0

/-- The vector from E to a point on the parabola -/
def vector_EM (E : PointE) (M : ParabolaPoint C) : ℝ × ℝ :=
  (M.x + E.lambda, M.y)

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The theorem to be proved -/
theorem parabola_tangent_condition (C : Parabola) (E : PointE) :
  (∀ (M N : ParabolaPoint C), 
    (∀ M' N' : ParabolaPoint C, 
      dot_product (vector_EM E M') (vector_EM E N') ≥ 
      dot_product (vector_EM E M) (vector_EM E N)) →
    dot_product (vector_EM E M) (vector_EM E N) = 0) →
  E.lambda = C.p / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_condition_l520_52003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l520_52064

theorem pyramid_volume (a : ℝ) (a_pos : a > 0) :
  let pyramid_volume := (81 - 32 * Real.pi) / 486 * a^3
  let base_side := a
  let height := -a / 2
  let sphere_radius := a / 3
  pyramid_volume = (81 - 32 * Real.pi) / 486 * a^3 := by
  -- The proof would go here
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l520_52064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_implies_a_value_l520_52046

/-- Circle C in the xy-plane -/
def circleC (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + 1 = 0

/-- Line l in the xy-plane -/
def lineL (x y : ℝ) (a : ℝ) : Prop :=
  x + a*y + 1 = 0

/-- The length of chord AB -/
def chord_length : ℝ := 4

/-- Theorem stating that if the chord AB formed by the intersection of circle C and line l
    has length 4, then a = -1 -/
theorem chord_intersection_implies_a_value :
  ∀ (a : ℝ), (∃ (x y : ℝ), circleC x y ∧ lineL x y a) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circleC x₁ y₁ ∧ circleC x₂ y₂ ∧ 
    lineL x₁ y₁ a ∧ lineL x₂ y₂ a ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2) →
  a = -1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_implies_a_value_l520_52046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_dry_l520_52015

/-- A person in the room -/
structure Person where
  position : ℝ × ℝ × ℝ

/-- The distance between two people -/
noncomputable def distance (p q : Person) : ℝ :=
  let (x₁, y₁, z₁) := p.position
  let (x₂, y₂, z₂) := q.position
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

/-- The theorem stating that at least one person remains dry -/
theorem at_least_one_dry (n : ℕ) (people : Fin n → Person) :
  Odd n →
  (∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → distance (people i) (people j) ≠ distance (people j) (people k)) →
  ∃ i : Fin n, ∀ j : Fin n, j ≠ i → ∃ k : Fin n, k ≠ j ∧ distance (people j) (people k) < distance (people j) (people i) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_dry_l520_52015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_third_term_l520_52073

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => a₁ * q^(n + 1)

theorem geometric_sequence_third_term (a₁ q : ℝ) :
  let a := geometric_sequence a₁ q
  (a 1 + a 2) / (a 0 + a 1) = 2 →
  a 3 = 8 →
  a 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_third_term_l520_52073
