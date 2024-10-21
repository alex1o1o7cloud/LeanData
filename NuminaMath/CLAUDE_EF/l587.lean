import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_cube_surface_area_equals_expected_l587_58768

noncomputable section

-- Define the cube edge length
def cube_edge : ℝ := 3

-- Define the hole diameter
def hole_diameter : ℝ := 1.5

-- Define the hole radius
def hole_radius : ℝ := hole_diameter / 2

-- Function to calculate the surface area of the modified cube
noncomputable def modified_cube_surface_area : ℝ :=
  -- Original surface area of the cube
  6 * cube_edge^2 +
  -- Area of cylindrical walls exposed by holes
  6 * (2 * Real.pi * hole_radius * cube_edge) -
  -- Area removed by circular holes
  6 * (Real.pi * hole_radius^2)

-- Theorem statement
theorem modified_cube_surface_area_equals_expected : 
  modified_cube_surface_area = 54 + 16.4 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_cube_surface_area_equals_expected_l587_58768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_seven_eq_28_l587_58725

/-- The number obtained by writing the integers from 1 to n in base 8, from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The count of b_k divisible by 7 for 1 ≤ k ≤ 100 -/
def count_divisible_by_seven : ℕ := 
  Finset.card (Finset.filter (λ k => (b (k + 1) % 7) = 0) (Finset.range 100))

theorem count_divisible_by_seven_eq_28 : count_divisible_by_seven = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_seven_eq_28_l587_58725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l587_58710

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 - 3*x - 10)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -2 ∧ x ≠ 5} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l587_58710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_in_square_distance_l587_58794

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A square with side length 2 -/
def square : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 2 ∧ 0 ≤ p.y ∧ p.y ≤ 2}

theorem six_points_in_square_distance (points : Finset Point) :
  points.card = 6 → (∀ p, p ∈ points → p ∈ square) →
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_in_square_distance_l587_58794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_gas_volume_l587_58703

/-- Represents the volume of gas at a given temperature -/
structure GasVolume where
  temperature : ℚ
  volume : ℚ

/-- Calculates the volume change for a given temperature change -/
def volumeChange (temperatureChange : ℚ) : ℚ :=
  (temperatureChange / 5) * 4

/-- The theorem stating the final volume of gas after temperature changes -/
theorem final_gas_volume 
  (initial : GasVolume)
  (h1 : initial.temperature = 30)
  (h2 : initial.volume = 36)
  (h3 : volumeChange 5 = 4) :
  let intermediate := GasVolume.mk 40 (initial.volume + volumeChange 10)
  let final := GasVolume.mk 25 (intermediate.volume - volumeChange 15)
  final.volume = 32 := by
  sorry

#check final_gas_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_gas_volume_l587_58703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_theorem_l587_58743

/-- A cone with two spheres inside it -/
structure ConeWithSpheres where
  r : ℝ  -- radius of smaller sphere s
  R : ℝ  -- radius of larger sphere S
  h : ℝ  -- height from cone vertex to center of s
  d : ℝ  -- distance from center of s to its circle of contact with cone
  D : ℝ  -- distance from cone vertex to circle of contact of S

/-- Conditions for the spheres and cone configuration -/
def valid_configuration (c : ConeWithSpheres) : Prop :=
  c.r > 0 ∧ c.R > c.r ∧ c.h > 0 ∧ c.d > 0 ∧ c.D > 0 ∧
  c.h = (2 * c.r^2) / (c.R - c.r) ∧
  c.d = (2 * c.r^2) / (c.R + c.r) ∧
  c.D = (2 * c.r * c.R) / (c.R + c.r)

/-- Volume of the region between the spheres inside the cone -/
noncomputable def volume_between_spheres (c : ConeWithSpheres) : ℝ :=
  (4 * Real.pi * c.r^2 * c.R^2) / (3 * (c.R + c.r))

/-- Theorem stating the volume of the region between the spheres -/
theorem volume_theorem (c : ConeWithSpheres) (h : valid_configuration c) :
  volume_between_spheres c = (4 * Real.pi * c.r^2 * c.R^2) / (3 * (c.R + c.r)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_theorem_l587_58743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_apex_angle_for_specific_spheres_cone_apex_angle_l587_58705

/-- The angle at the apex of a cone that externally touches three spheres --/
noncomputable def coneApexAngle (r1 r2 r3 : ℝ) : ℝ :=
  2 * Real.arctan (1 / 72)

/-- Theorem: The angle at the apex of a cone that externally touches three spheres with radii 2, 2, and 5 --/
theorem cone_apex_angle_for_specific_spheres :
  coneApexAngle 2 2 5 = 2 * Real.arctan (1 / 72) := by
  sorry

/-- Axiom: The cone touches all spheres externally --/
axiom cone_touches_spheres_externally (r1 r2 r3 : ℝ) : Prop

/-- Axiom: The apex of the cone is located in the middle between the points of contact of the identical spheres with the table --/
axiom cone_apex_midpoint (r1 r2 r3 : ℝ) : Prop

/-- Theorem: The angle at the apex of a cone that externally touches three spheres with radii r1, r2, and r3 --/
theorem cone_apex_angle (r1 r2 r3 : ℝ) 
  (h1 : r1 = 2) (h2 : r2 = 2) (h3 : r3 = 5)
  (h4 : cone_touches_spheres_externally r1 r2 r3)
  (h5 : cone_apex_midpoint r1 r2 r3) :
  coneApexAngle r1 r2 r3 = 2 * Real.arctan (1 / 72) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_apex_angle_for_specific_spheres_cone_apex_angle_l587_58705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_A_and_B_l587_58750

/-- Represents a 3x3 grid with numbers 1, 2, and 3 --/
def Grid := Fin 3 → Fin 3 → Fin 3

/-- Checks if a grid is valid according to the rules --/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j = 1 ∨ g i j = 2 ∨ g i j = 3) ∧
  (∀ i, (g i 0 ≠ g i 1) ∧ (g i 0 ≠ g i 2) ∧ (g i 1 ≠ g i 2)) ∧
  (∀ j, (g 0 j ≠ g 1 j) ∧ (g 0 j ≠ g 2 j) ∧ (g 1 j ≠ g 2 j))

/-- The theorem to be proved --/
theorem sum_of_A_and_B (g : Grid) (h : is_valid_grid g) 
  (h1 : g 0 0 = 2) (h2 : g 1 2 = g 2 0) : g 1 2 + g 2 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_A_and_B_l587_58750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l587_58795

theorem sin_alpha_value (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : π/2 < β ∧ β < π)
  (h3 : Real.sin (α + β) = 33/65)
  (h4 : Real.cos β = -5/13) : 
  Real.sin α = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l587_58795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_coin_problem_l587_58732

/-- Represents the denominations of coins available -/
inductive Coin : Type
  | ten : Coin
  | fifteen : Coin
  | twenty : Coin

/-- The bus fare in kopecks -/
def busFare : ℕ := 5

/-- A function that calculates the minimum number of coins required for k people -/
def minCoins (k : ℕ) : ℕ := k + (k + 3) / 4

/-- Convert a Coin to its Natural number value -/
def Coin.toNat : Coin → ℕ
  | Coin.ten => 10
  | Coin.fifteen => 15
  | Coin.twenty => 20

theorem bus_coin_problem (k : ℕ) :
  ∀ (coins : Finset Coin),
  (∀ person : Fin k, ∃ (payment : Coin), payment ∈ coins ∧ 
    (Coin.toNat payment - busFare) ≥ 0) →
  (minCoins k) ≤ Finset.card coins :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_coin_problem_l587_58732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l587_58701

/-- A geometric sequence with first term a₁ and common ratio q -/
noncomputable def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ := λ n ↦ a₁ * q ^ (n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_first_term
  (a₁ q : ℝ) (hq : q ≠ 0) :
  let a := geometric_sequence a₁ q
  let S := geometric_sum a₁ q
  (S 3 = a 2 + 10 * a 1) →
  (a 5 = 9) →
  a₁ = 1/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l587_58701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_derivative_implies_max_value_l587_58798

theorem even_derivative_implies_max_value (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^3 + a*x^2 + (a-3)*x) :
  (∀ x, (deriv f) x = (deriv f) (-x)) → (∃ x, f x = 2 ∧ ∀ y, f y ≤ 2) := by
  sorry

#check even_derivative_implies_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_derivative_implies_max_value_l587_58798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equality_log_equation_ratios_l587_58715

-- Problem 1
theorem power_sum_equality : (-2)^101 + (-2)^100 = -2^100 := by sorry

-- Problem 2
theorem log_equation_ratios {x y : ℝ} (h : x > 0 ∧ y > 0) :
  Real.log (x + y) + Real.log (2*x + 3*y) - Real.log 3 = Real.log 4 + Real.log x + Real.log y →
  x / y = 1/2 ∨ x / y = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equality_log_equation_ratios_l587_58715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l587_58740

theorem angle_properties (α : Real) 
  (h1 : α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) 
  (h2 : Real.sin α = -Real.sqrt 5 / 5) : 
  Real.cos α = -2 * Real.sqrt 5 / 5 ∧ 
  Real.tan α = 1 / 2 ∧
  (Real.cos (Real.pi / 2 + α) * Real.sin (-Real.pi - α)) / 
  (Real.cos (11 * Real.pi / 2 - α) * Real.sin (9 * Real.pi / 2 + α)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l587_58740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pdf_of_Y_squared_l587_58718

/-- The probability density function of X -/
noncomputable def p (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 2 then 1 else 0

/-- Y as a function of X -/
def Y (x : ℝ) : ℝ := x^2

/-- The probability density function of Y -/
noncomputable def g (y : ℝ) : ℝ :=
  if 1 ≤ y ∧ y ≤ 4 then 1 / (2 * Real.sqrt y) else 0

/-- Theorem stating that g is the correct probability density function for Y -/
theorem pdf_of_Y_squared (x : ℝ) : 
  g (Y x) = p x * |1 / (2 * x)| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pdf_of_Y_squared_l587_58718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_RPQ_is_90_degrees_l587_58762

-- Define the points
variable (P Q R S : EuclideanSpace ℝ (Fin 2))

-- Define the angle measure function
noncomputable def angle_measure (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Define the conditions
variable (h1 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • R + t • S)
variable (h2 : angle_measure S Q R = 2 * angle_measure R Q P)
variable (h3 : dist P Q = dist P R)
variable (h4 : ∃ y : ℝ, angle_measure R S Q = 3 * y)
variable (h5 : ∃ y : ℝ, angle_measure R P Q = 4 * y)

-- State the theorem
theorem angle_RPQ_is_90_degrees :
  angle_measure R P Q = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_RPQ_is_90_degrees_l587_58762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_interval_minimizes_cost_l587_58709

/-- Represents the optimal purchase interval for flour that minimizes the average daily total cost -/
noncomputable def optimal_purchase_interval : ℝ := 10

/-- Daily flour requirement in tons -/
def daily_flour_requirement : ℝ := 6

/-- Price per ton of flour in yuan -/
def price_per_ton : ℝ := 1800

/-- Daily storage cost per ton of flour in yuan -/
def daily_storage_cost : ℝ := 3

/-- Shipping fee per purchase in yuan -/
def shipping_fee : ℝ := 900

/-- Average daily total cost as a function of purchase interval -/
noncomputable def average_daily_cost (t : ℝ) : ℝ :=
  (daily_flour_requirement * price_per_ton * t + shipping_fee) / t +
  daily_storage_cost * daily_flour_requirement * t / 2

/-- Theorem stating that the optimal purchase interval minimizes the average daily total cost -/
theorem optimal_interval_minimizes_cost :
  ∀ t > 0, average_daily_cost optimal_purchase_interval ≤ average_daily_cost t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_interval_minimizes_cost_l587_58709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_set_cardinality_l587_58786

/-- The base-n representation of 536537 as a function of n -/
def base_n_representation (n : ℕ) : ℕ := 5*n^5 + 3*n^4 + 6*n^3 + 5*n^2 + 3*n + 7

/-- The set of integers n between 3 and 200 (inclusive) for which 536537₍ₙ₎ is divisible by 13 -/
def divisible_set : Set ℕ :=
  {n : ℕ | 3 ≤ n ∧ n ≤ 200 ∧ (base_n_representation n) % 13 = 0}

/-- Predicate to check if a number is in the divisible set -/
def is_in_divisible_set (n : ℕ) : Prop :=
  3 ≤ n ∧ n ≤ 200 ∧ (base_n_representation n) % 13 = 0

/-- Proof that the predicate is decidable -/
instance : DecidablePred is_in_divisible_set :=
  fun n => And.decidable

theorem divisible_set_cardinality :
  Finset.card (Finset.filter (λ n => is_in_divisible_set n) (Finset.range 201)) = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_set_cardinality_l587_58786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_vertex_distance_l587_58760

noncomputable def distance_to_plane (a b c d x y z : ℝ) : ℝ :=
  (a * x + b * y + c * z + d) / Real.sqrt (a^2 + b^2 + c^2)

theorem cube_vertex_distance (r s t : ℕ) (hr : r > 0) (hs : s > 0) (ht : t > 0) 
    (hsum : r + s + t < 1200) :
  (∃ (a b c d : ℝ),
    a^2 + b^2 + c^2 = 1 ∧
    distance_to_plane a b c d 12 0 0 = 13 ∧
    distance_to_plane a b c d 0 12 0 = 14 ∧
    distance_to_plane a b c d 0 0 12 = 15 ∧
    distance_to_plane a b c d 0 0 0 = (r - Real.sqrt s) / t) →
  r + s + t = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_vertex_distance_l587_58760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_coloring_possible_l587_58799

theorem house_coloring_possible (n : ℕ) (π : Equiv.Perm (Fin n)) :
  ∃ (f : Fin n → Fin 3), ∀ i : Fin n, f i ≠ f (π i) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_coloring_possible_l587_58799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_sector_l587_58749

/-- Given a sector of a circle with radius 6 and central angle 60°, 
    the length of the arc is 2π. -/
theorem arc_length_of_sector (r θ_deg l : ℝ) : 
  r = 6 → θ_deg = 60 → l = r * (θ_deg * π / 180) → l = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_sector_l587_58749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_distance_formula_l587_58739

/-- A truncated pyramid with regular triangular bases -/
structure TruncatedPyramid where
  l : ℝ  -- Length of the longer lateral edge
  α : ℝ  -- Angle between the longer lateral edge and the base plane

/-- The length of the segment connecting the centers of the upper and lower bases -/
noncomputable def centerDistance (tp : TruncatedPyramid) : ℝ :=
  (tp.l / 3) * Real.sqrt (5 - 4 * Real.cos (2 * tp.α))

/-- Theorem stating the length of the segment connecting the centers of the bases -/
theorem center_distance_formula (tp : TruncatedPyramid) :
  centerDistance tp = (tp.l / 3) * Real.sqrt (5 - 4 * Real.cos (2 * tp.α)) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_distance_formula_l587_58739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_is_two_elevenths_l587_58766

/-- Represents the dimensions and lap times of a rectangular park -/
structure Park where
  length : ℚ
  width : ℚ
  inside_lap_time : ℚ
  outside_lap_time_diff : ℚ
  outside_edge_width : ℚ

/-- Calculates the speed of a person walking around the park -/
def calculate_speed (p : Park) : ℚ :=
  let inside_perimeter := 2 * (p.length + p.width)
  let outside_perimeter := 2 * (p.length + 2 * p.outside_edge_width + p.width + 2 * p.outside_edge_width)
  (outside_perimeter - inside_perimeter) / p.outside_lap_time_diff

theorem speed_is_two_elevenths (p : Park) 
  (h1 : p.length = 100)
  (h2 : p.width = 50)
  (h3 : p.inside_lap_time = 200)
  (h4 : p.outside_lap_time_diff = 220)
  (h5 : p.outside_edge_width = 5) :
  calculate_speed p = 2 / 11 := by
  sorry

def example_park : Park := {
  length := 100,
  width := 50,
  inside_lap_time := 200,
  outside_lap_time_diff := 220,
  outside_edge_width := 5
}

#eval calculate_speed example_park

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_is_two_elevenths_l587_58766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_change_l587_58708

/-- Represents the volume change of a gas with temperature under constant pressure -/
structure GasVolume where
  initialTemp : ℚ
  initialVolume : ℚ
  finalTemp : ℚ
  volumeChangeRate : ℚ
  tempChangeRate : ℚ

/-- Calculates the final volume of a gas given temperature change -/
def finalVolume (g : GasVolume) : ℚ :=
  g.initialVolume + (g.finalTemp - g.initialTemp) / g.tempChangeRate * g.volumeChangeRate

/-- Theorem stating that under the given conditions, the final volume is 63 cm³ -/
theorem gas_volume_change (g : GasVolume) 
    (h1 : g.initialTemp = 22)
    (h2 : g.initialVolume = 45)
    (h3 : g.finalTemp = 34)
    (h4 : g.volumeChangeRate = 3)
    (h5 : g.tempChangeRate = 2) :
  finalVolume g = 63 := by
  -- Unfold the definition of finalVolume
  unfold finalVolume
  -- Substitute the given values
  rw [h1, h2, h3, h4, h5]
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_change_l587_58708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basement_pump_out_time_l587_58727

/-- Calculates the time required to pump out water from a flooded basement. -/
noncomputable def pumpOutTime (length width depth : ℝ) (numPumps pumpRate : ℝ) (gallonsPerCubicFoot : ℝ) : ℝ :=
  let volumeCubicFeet := length * width * (depth / 12)
  let volumeGallons := volumeCubicFeet * gallonsPerCubicFoot
  let totalPumpRate := numPumps * pumpRate
  volumeGallons / totalPumpRate

/-- Theorem stating the time required to pump out the basement. -/
theorem basement_pump_out_time :
  pumpOutTime 30 36 24 3 10 7.5 = 540 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basement_pump_out_time_l587_58727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_120_meters_l587_58746

/-- The speed of each train in km/hr -/
noncomputable def train_speed : ℝ := 108

/-- The time it takes for the trains to cross each other in seconds -/
noncomputable def crossing_time : ℝ := 4

/-- The speed conversion factor from km/hr to m/s -/
noncomputable def speed_conversion : ℝ := 1000 / 3600

/-- Calculates the length of each train in meters -/
noncomputable def train_length : ℝ := 
  (train_speed * speed_conversion * crossing_time) / 2

theorem train_length_is_120_meters : 
  train_length = 120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_120_meters_l587_58746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medicine_expiration_expiration_date_is_may_11_2034_l587_58767

-- Define the starting date
def start_date : Nat := 0  -- Representing March 5, 2019 as 0

-- Define the expiration time in seconds
def expiration_time : Nat := Nat.factorial 12

-- Define the function to calculate the expiration date
def calculate_expiration_date (start : Nat) (duration : Nat) : Nat :=
  start + duration

-- Theorem statement
theorem medicine_expiration :
  calculate_expiration_date start_date expiration_time = 479001600 := by
  sorry

-- Helper theorem to show the result corresponds to May 11, 2034
theorem expiration_date_is_may_11_2034 :
  479001600 / 86400 = 5541 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_medicine_expiration_expiration_date_is_may_11_2034_l587_58767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_samovar_cools_faster_l587_58731

/-- Represents a samovar with its volume and surface area. -/
structure Samovar where
  volume : ℝ
  surfaceArea : ℝ
  (volume_positive : volume > 0)
  (surface_area_positive : surfaceArea > 0)

/-- The cooling rate of a samovar is proportional to its surface area. -/
noncomputable def coolingRate (s : Samovar) : ℝ := s.surfaceArea

/-- The cooling rate per unit volume of a samovar. -/
noncomputable def coolingRatePerUnitVolume (s : Samovar) : ℝ := coolingRate s / s.volume

/-- Given two samovars with the same shape but different sizes,
    the smaller samovar has a higher cooling rate per unit volume. -/
theorem smaller_samovar_cools_faster (small large : Samovar)
    (h_shape : ∃ (n : ℝ), n > 1 ∧ large.volume = n^3 * small.volume ∧ large.surfaceArea = n^2 * small.surfaceArea) :
    coolingRatePerUnitVolume small > coolingRatePerUnitVolume large := by
  sorry

#check smaller_samovar_cools_faster

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_samovar_cools_faster_l587_58731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_correct_prove_systematic_sampling_l587_58792

def is_arithmetic_sequence (seq : List Nat) : Prop :=
  seq.length > 1 ∧ ∃ d, ∀ i, i < seq.length - 1 → seq[i + 1]! - seq[i]! = d

def is_evenly_spread (seq : List Nat) (total : Nat) : Prop :=
  seq.length > 1 ∧ ∃ k, ∀ i, i < seq.length - 1 → seq[i + 1]! - seq[i]! = k ∧ k * seq.length = total

theorem systematic_sampling_correct (students : Nat) (sample_size : Nat) (sample : List Nat) : Prop :=
  students = 20 ∧
  sample_size = 4 ∧
  sample = [5, 10, 15, 20] ∧
  is_arithmetic_sequence sample ∧
  is_evenly_spread sample students ∧
  ∀ n ∈ sample, 1 ≤ n ∧ n ≤ students

theorem prove_systematic_sampling : systematic_sampling_correct 20 4 [5, 10, 15, 20] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_correct_prove_systematic_sampling_l587_58792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_c_for_inequality_l587_58735

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |log x|

-- Define the theorem
theorem range_of_c_for_inequality :
  ∃ a b : ℝ, (a = -1 ∧ b = 1 / (ℯ - 1)) ∧
  (∀ c : ℝ, (∀ x ∈ Set.Ioo 0 ℯ, f x - f 1 ≥ c * (x - 1)) → c ∈ Set.Ico a b) ∧
  (∀ ε > 0, ∃ c ∈ Set.Ico a b, ∃ x ∈ Set.Ioo 0 ℯ, f x - f 1 < c * (x - 1) + ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_c_for_inequality_l587_58735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_ratio_theorem_l587_58734

open Complex

theorem complex_ratio_theorem : 
  let z₁ : ℂ := (Real.sqrt 3 / 2) + (1 / 2) * I
  let z₂ : ℂ := 3 + 4 * I
  Complex.abs (z₁^2016) / Complex.abs z₂ = 1 / 5 := by
    -- Proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_ratio_theorem_l587_58734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_l587_58783

noncomputable def g (x : ℝ) : ℝ := (6 * x^2 + 11 * x + 17) / (7 * (2 + x))

theorem g_min_value (x : ℝ) (h : x ≥ 0) : g x ≥ 127/24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_min_value_l587_58783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l587_58765

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define eccentricity for a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 + b^2) / a

-- Theorem statement
theorem hyperbola_eccentricity : 
  eccentricity 4 3 = 5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l587_58765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_3x_gt_4y_l587_58730

/-- The width of the rectangle -/
noncomputable def w : ℝ := 2021

/-- The height of the rectangle -/
noncomputable def h : ℝ := 2022

/-- The area of the rectangle -/
noncomputable def rectangle_area : ℝ := w * h

/-- The y-coordinate of the intersection point of y = (3/4)x with x = w -/
noncomputable def intersection_y : ℝ := (3/4) * w

/-- The area of the triangle formed by the points (0,0), (w,0), and (w,intersection_y) -/
noncomputable def triangle_area : ℝ := (1/2) * w * intersection_y

/-- The probability that 3x > 4y for a randomly chosen point (x,y) in the rectangle -/
noncomputable def probability : ℝ := triangle_area / rectangle_area

theorem probability_3x_gt_4y : probability = 1515750 / 4044 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_3x_gt_4y_l587_58730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l587_58796

/-- Calculates the time (in seconds) for a train to pass a person moving in the opposite direction. -/
noncomputable def timeToPast (trainLength : ℝ) (trainSpeed : ℝ) (personSpeed : ℝ) : ℝ :=
  trainLength / (trainSpeed + personSpeed)

/-- Converts speed from km/h to m/s. -/
noncomputable def kmhToMs (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

theorem train_passing_time :
  let trainLength : ℝ := 250
  let trainSpeed : ℝ := 80
  let personSpeed : ℝ := 12
  abs (timeToPast trainLength (kmhToMs trainSpeed) (kmhToMs personSpeed) - 9.79) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l587_58796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_point_y_value_l587_58745

theorem angle_point_y_value (θ : Real) (y : Int) :
  -- The vertex of angle θ is at the origin and its initial side is the non-negative half of the x-axis
  -- P(4,y) is a point on the terminal side of angle θ
  (4 : Real) * (Real.sin θ) = y →
  -- sin θ = -2√5/5
  Real.sin θ = -2 * Real.sqrt 5 / 5 →
  -- The value of y is -8
  y = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_point_y_value_l587_58745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_equation_l587_58724

/-- Represents the total amount of work to be done -/
def W : ℝ := sorry

/-- Represents the work rate of one woman per day -/
def Ww : ℝ := sorry

/-- Represents the number of days it takes for 10 men and 15 women to complete the work -/
def D : ℝ := sorry

/-- Theorem stating the relationship between work, work rates, and time -/
theorem work_completion_equation : 
  (10 * (W / 100) + 15 * Ww) * D = W := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_equation_l587_58724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_factorial_product_l587_58761

theorem smallest_n_for_factorial_product (a b c : ℕ) (m n : ℕ) : 
  a + b + c = 2010 →
  b = 2 * a →
  (Nat.factorial a * Nat.factorial b * Nat.factorial c : ℕ) = m * (10 : ℕ)^n →
  ¬(10 ∣ m) →
  (∀ k : ℕ, k < n → ∃ (m' : ℕ), (Nat.factorial a * Nat.factorial b * Nat.factorial c : ℕ) = m' * (10 : ℕ)^k ∧ (10 ∣ m')) →
  n = 589 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_factorial_product_l587_58761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_k_values_l587_58780

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+1) + 1
noncomputable def g (k x : ℝ) : ℝ := k * 2^x

-- State the theorem
theorem intersection_points_k_values :
  ∀ k : ℝ,
  (∀ x, x ∈ Set.Icc 1 2 → f x ≤ 9 ∧ f x ≥ 1) →
  (∃ x₁ x₂, x₁ ∈ Set.Icc (-1) 2 ∧ x₂ ∈ Set.Icc (-1) 2 ∧ x₁ ≠ x₂ ∧ f x₁ = g k x₁ ∧ f x₂ = g k x₂) →
  (k = 1/4 ∨ k = 1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_k_values_l587_58780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_machines_count_l587_58736

/-- The number of machines initially working on the job -/
noncomputable def initial_machines : ℕ := 8

/-- The time taken by the initial number of machines to complete the job -/
noncomputable def initial_time : ℝ := 6

/-- The number of machines in the alternative scenario -/
noncomputable def alt_machines : ℕ := 5

/-- The time taken by the alternative number of machines to complete the job -/
noncomputable def alt_time : ℝ := 9.6

/-- The work rate of a single machine (jobs per hour) -/
noncomputable def work_rate : ℝ := 1 / (alt_machines * alt_time)

theorem initial_machines_count : 
  initial_machines * work_rate * initial_time = 1 ∧
  alt_machines * work_rate * alt_time = 1 →
  initial_machines = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_machines_count_l587_58736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mona_joined_nine_groups_l587_58754

/-- Represents the number of groups Mona joined -/
def num_groups : ℕ := sorry

/-- Represents the number of other players in each group -/
def players_per_group : ℕ := 4

/-- Represents the total number of unique players Mona grouped with -/
def total_unique_players : ℕ := 33

/-- Represents the number of non-unique players in the first special group -/
def non_unique_players_group1 : ℕ := 2

/-- Represents the number of non-unique players in the second special group -/
def non_unique_players_group2 : ℕ := 1

/-- Theorem stating that the number of groups Mona joined is 9 -/
theorem mona_joined_nine_groups :
  num_groups * players_per_group - (non_unique_players_group1 + non_unique_players_group2) = total_unique_players →
  num_groups = 9 := by
  sorry

#check mona_joined_nine_groups

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mona_joined_nine_groups_l587_58754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_squared_at_5_l587_58784

noncomputable def g (x : ℝ) : ℝ := 25 / (4 + 2*x)

theorem inverse_g_squared_at_5 : (Function.invFun g 5)^2 = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_squared_at_5_l587_58784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_equiv_x_range_l587_58772

noncomputable def f (x : ℝ) := x * (2^x - 1/(2^x))

theorem f_inequality_equiv_x_range (x : ℝ) :
  f (x - 1) > f x ↔ x < (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_equiv_x_range_l587_58772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowella_to_pamela_increase_l587_58770

/-- Represents the exam and scores of three students -/
structure ExamScores where
  total_items : ℕ
  lowella_percentage : ℝ
  mandy_score : ℕ
  pamela_score : ℕ

/-- Calculates the percentage increase between two scores -/
noncomputable def percentage_increase (initial : ℝ) (final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

/-- Theorem stating the percentage increase from Lowella to Pamela -/
theorem lowella_to_pamela_increase (exam : ExamScores) 
  (h1 : exam.total_items = 100)
  (h2 : exam.lowella_percentage = 35)
  (h3 : exam.mandy_score = 84)
  (h4 : exam.mandy_score = 2 * exam.pamela_score) :
  percentage_increase exam.lowella_percentage 
    ((exam.pamela_score : ℝ) / (exam.total_items : ℝ) * 100) = 20 := by
  sorry

#eval "Theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowella_to_pamela_increase_l587_58770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_element_in_S_unique_element_is_116_l587_58755

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 10 - (1 / 2) * (Real.log x / Real.log 3)

-- Define the set S
def S : Set ℤ := {n : ℤ | f (n^2 - 214*n - 1998 : ℝ) ≥ 0}

-- Statement to prove
theorem unique_element_in_S : ∃! n : ℤ, n ∈ S := by
  -- The proof goes here
  sorry

-- Additional theorem to show the unique element is 116
theorem unique_element_is_116 : ∀ n : ℤ, n ∈ S ↔ n = 116 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_element_in_S_unique_element_is_116_l587_58755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_remainder_l587_58788

theorem modular_remainder (y : ℕ) (h : (7 * y) % 31 = 1) : (15 + 3 * y) % 31 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_remainder_l587_58788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_theorem_l587_58790

/-- Calculates the total sample mean for a stratified sample with two groups of equal size. -/
noncomputable def totalSampleMean (maleMean femMean : ℝ) : ℝ :=
  (maleMean + femMean) / 2

/-- Calculates the total sample variance for a stratified sample with two groups of equal size. -/
noncomputable def totalSampleVariance (maleMean femMean maleVar femVar : ℝ) : ℝ :=
  let totalMean := totalSampleMean maleMean femMean
  (((maleMean - totalMean)^2 + maleVar) + ((femMean - totalMean)^2 + femVar)) / 2

/-- 
Theorem: For a stratified sample with two groups of equal size (100 each),
where the male group has a sample mean of 170 and sample variance of 22,
and the female group has a sample mean of 160 and sample variance of 38,
the total sample mean is 165 and the total sample variance is 55.
-/
theorem stratified_sample_theorem :
  let maleMean : ℝ := 170
  let femMean : ℝ := 160
  let maleVar : ℝ := 22
  let femVar : ℝ := 38
  totalSampleMean maleMean femMean = 165 ∧
  totalSampleVariance maleMean femMean maleVar femVar = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_theorem_l587_58790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l587_58756

def is_arithmetic_sequence (s : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, s (n + 1) - s n = d

theorem sequence_property (a : ℕ → ℕ) (S : ℕ → ℕ)
    (h1 : ∀ n : ℕ, n > 0 → S n = a (n + 1) - 2^(n + 1) + 1)
    (h2 : a 1 = 1) :
    (is_arithmetic_sequence (λ n ↦ (a n : ℚ) / 2^(n - 1))) ∧
    (∀ n : ℕ, n > 0 → a n = n * 2^(n - 1)) := by
  sorry

#check sequence_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l587_58756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colored_cells_l587_58753

/-- A colored cell in a 9x9 grid --/
structure ColoredCell where
  x : Fin 9
  y : Fin 9

/-- The distance between two colored cells --/
noncomputable def distance (c1 c2 : ColoredCell) : ℝ :=
  Real.sqrt (((c1.x : ℝ) - (c2.x : ℝ))^2 + ((c1.y : ℝ) - (c2.y : ℝ))^2)

/-- A valid coloring of the grid --/
def ValidColoring (cells : List ColoredCell) : Prop :=
  ∀ c1 c2, c1 ∈ cells → c2 ∈ cells → c1 ≠ c2 → distance c1 c2 > 2

/-- The main theorem: maximum number of colored cells is 17 --/
theorem max_colored_cells :
  ∀ (cells : List ColoredCell), ValidColoring cells → cells.length ≤ 17 := by
  sorry

#check max_colored_cells

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colored_cells_l587_58753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_is_nine_halves_l587_58771

-- Define the curve and line functions
def curve (x : ℝ) : ℝ := 9 - x^2
def line (x : ℝ) : ℝ := x + 7

-- Define the area enclosed by the curve and line
noncomputable def enclosed_area : ℝ :=
  ∫ x in Set.Icc (-2) 1, (curve x - line x)

-- Theorem statement
theorem area_enclosed_is_nine_halves :
  enclosed_area = 9/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_is_nine_halves_l587_58771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_numbers_l587_58714

theorem opposite_numbers : -(-(5 : ℤ)) = -(-|5|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_numbers_l587_58714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_theorem_l587_58764

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  area : ℝ
  acute_angle : ℝ
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- The sides of the trapezoid given its area and acute angle -/
def trapezoid_sides (t : IsoscelesTrapezoid) : Prop :=
  t.area = 32 ∧
  t.acute_angle = 30 * Real.pi / 180 ∧
  t.side1 = 8 ∧
  t.side2 = 8 - 4 * Real.sqrt 3 ∧
  t.side3 = 8 + 4 * Real.sqrt 3

theorem trapezoid_theorem (t : IsoscelesTrapezoid) :
  trapezoid_sides t := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_theorem_l587_58764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_three_halves_l587_58758

/-- The distance from a point to a line in polar coordinates --/
noncomputable def distance_point_to_line (r : ℝ) (θ : ℝ) (l : ℝ → ℝ → ℝ) : ℝ :=
  -- Definition of distance calculation
  sorry

/-- Polar equation of the line --/
noncomputable def line_equation (ρ θ : ℝ) : ℝ :=
  ρ * Real.sin (θ + Real.pi/3) - 1/2

theorem distance_point_to_line_is_three_halves :
  distance_point_to_line 2 (Real.pi/6) line_equation = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_three_halves_l587_58758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_completes_in_4_days_l587_58759

/-- The number of days it takes P to complete the job alone -/
noncomputable def P_days : ℝ := sorry

/-- The daily work rate of P -/
noncomputable def P_rate : ℝ := 1 / P_days

/-- The daily work rate of Q -/
noncomputable def Q_rate : ℝ := (1 / 3) * P_rate

/-- The time it takes P and Q to complete the job together -/
def PQ_time : ℝ := 3

theorem P_completes_in_4_days :
  P_rate + Q_rate = 1 / PQ_time → P_days = 4 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_completes_in_4_days_l587_58759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l587_58751

open Real

theorem trigonometric_identities :
  (∀ α : ℝ, (sin (α + 3/2*π) * sin (-α + π) * cos (α + π/2)) / 
    (cos (-α - π) * cos (α - π/2) * tan (α + π)) = cos α) ∧
  (tan (675 * π/180) + sin (-330 * π/180) + cos (960 * π/180) = -1) := by
  constructor
  · intro α
    -- Proof for the first part
    sorry
  · -- Proof for the second part
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l587_58751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_bounds_l587_58729

theorem vector_magnitude_bounds (a b c : ℝ × ℝ) (l : ℝ) : 
  (‖a‖ = 1) → 
  (‖b‖ = 2) → 
  (‖c‖ = 3) → 
  (0 < l) → 
  (l < 1) → 
  (b • c = 0) → 
  ((6 / Real.sqrt 13) - 1 ≤ ‖a - l • b - (1 - l) • c‖) ∧ 
  (‖a - l • b - (1 - l) • c‖ ≤ 4) := by
  sorry

#check vector_magnitude_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_bounds_l587_58729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_chair_price_l587_58769

-- Define the sale price and discount rate
noncomputable def sale_price : ℝ := 59.95
noncomputable def discount_rate : ℝ := 0.2309

-- Define the original price calculation
noncomputable def original_price : ℝ := sale_price / (1 - discount_rate)

-- Theorem statement
theorem lawn_chair_price :
  ∀ ε > 0, |original_price - 77.95| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_chair_price_l587_58769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_digits_correct_l587_58741

/-- The first three digits after the decimal point of (10^987 + 1)^(8/3) -/
def first_three_digits : ℕ := 666

/-- The expression (10^987 + 1)^(8/3) -/
noncomputable def expression : ℝ := (10^987 + 1)^(8/3)

/-- The binomial expansion of the expression up to the third term -/
noncomputable def binomial_expansion : ℝ := 
  10^3296 + (8/3) * 10^2309 + (8/3 * 5/3) / 2 * 10^1322

theorem first_three_digits_correct :
  ∃ (n : ℕ), ∃ (r : ℝ), expression = (n : ℝ) + (first_three_digits : ℝ) / 1000 + r ∧ 0 ≤ r ∧ r < 1/1000 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_digits_correct_l587_58741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_theorem_l587_58747

/-- Represents the properties of a train crossing a platform and a signal pole. -/
structure TrainCrossing where
  train_length : ℝ
  platform_length : ℝ
  time_cross_platform : ℝ

/-- Calculates the time taken for a train to cross a signal pole. -/
noncomputable def time_cross_signal_pole (tc : TrainCrossing) : ℝ :=
  tc.train_length / ((tc.train_length + tc.platform_length) / tc.time_cross_platform)

/-- Theorem stating that for a train of length 300 m crossing a platform of length 450 m
    in 45 seconds, the time taken to cross a signal pole is 18 seconds. -/
theorem train_crossing_time_theorem :
  let tc : TrainCrossing := {
    train_length := 300,
    platform_length := 450,
    time_cross_platform := 45
  }
  time_cross_signal_pole tc = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_theorem_l587_58747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_l587_58704

-- Define the curve
def curve (x y : ℝ) : Prop := x * y = 2

-- Define the circle (we don't know its equation, but we know it exists)
def circle_set : Set (ℝ × ℝ) := sorry

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | p ∈ circle_set ∧ curve p.1 p.2}

-- State the theorem
theorem fourth_intersection_point :
  (4, 1/2) ∈ intersection_points ∧
  (-6, -1/3) ∈ intersection_points ∧
  (1/4, 8) ∈ intersection_points ∧
  (∃ (s : Finset (ℝ × ℝ)), s.toSet = intersection_points ∧ s.card = 4) →
  (-1/6, -12) ∈ intersection_points :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_l587_58704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_larger_distance_l587_58789

-- Define a type for points in a plane
def Point := ℝ × ℝ

-- Define a distance function between two points
noncomputable def distance (p q : Point) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the theorem
theorem subset_with_larger_distance 
  (S : Finset Point) 
  (h : ∀ (p q : Point), p ∈ S → q ∈ S → p ≠ q → distance p q ≥ 1) :
  ∃ (T : Finset Point), 
    T ⊆ S ∧ 
    T.card ≥ S.card / 7 ∧ 
    ∀ (p q : Point), p ∈ T → q ∈ T → p ≠ q → distance p q ≥ Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_larger_distance_l587_58789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_20_l587_58782

noncomputable section

-- Define the clock and its properties
def clock_degrees : ℝ := 360
def hours_on_clock : ℕ := 12
def minutes_in_hour : ℕ := 60

-- Define the time
def hour : ℕ := 3
def minute : ℕ := 20

-- Calculate degrees per hour and per minute
def degrees_per_hour : ℝ := clock_degrees / hours_on_clock
def degrees_per_minute : ℝ := clock_degrees / minutes_in_hour

-- Calculate initial positions of hands at 3:00
def initial_hour_position : ℝ := hour * degrees_per_hour
def minute_position : ℝ := minute * degrees_per_minute

-- Calculate additional movement of hour hand
def hour_hand_additional_movement : ℝ := (minute : ℝ) * (degrees_per_hour / minutes_in_hour)

-- Calculate final position of hour hand
def hour_hand_position : ℝ := initial_hour_position + hour_hand_additional_movement

-- Define the function to calculate the smaller angle
def smaller_angle (pos1 : ℝ) (pos2 : ℝ) : ℝ :=
  min (abs (pos1 - pos2)) (clock_degrees - abs (pos1 - pos2))

-- Theorem statement
theorem clock_angle_at_3_20 :
  smaller_angle hour_hand_position minute_position = 20 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_20_l587_58782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_right_triangle_l587_58774

/-- A right triangle with hypotenuse 4 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  hypotenuse : a^2 + b^2 = 16

/-- The area of a right triangle -/
noncomputable def area (t : RightTriangle) : ℝ := (1/2) * t.a * t.b

theorem max_area_right_triangle :
  (∀ t : RightTriangle, area t ≤ 4) ∧ 
  (∃ t : RightTriangle, area t = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_right_triangle_l587_58774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_inequality_l587_58700

theorem cubic_root_inequality (x : ℝ) :
  (x ^ (1/3 : ℝ)) + 2 / ((x ^ (1/3 : ℝ)) + 3) ≤ 0 ↔ x ∈ Set.Iic (-27) ∪ Set.Icc (-8) (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_inequality_l587_58700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lineup_arrangements_l587_58712

theorem lineup_arrangements (n : ℕ) (h : n = 6) : 
  -- 1. Six people form a circle
  Nat.factorial (n - 1) = 120 ∧
  -- 2. All stand in a row, with A and B must be adjacent
  2 * Nat.factorial (n - 1) = 240 ∧
  -- 3. All stand in a row, with A and B not adjacent
  Nat.factorial (n - 2) * (n - 1) = 480 ∧
  -- 4. All stand in a row, with A, B, and C in order from left to right
  Nat.factorial (n - 3) = 120 ∧
  -- 5. All stand in a row, with A and B at the ends
  2 * Nat.factorial (n - 2) = 48 ∧
  -- 6. All stand in a row, with A not at the left end and B not at the right end
  Nat.factorial n - 2 * Nat.factorial (n - 1) + Nat.factorial (n - 2) = 504 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lineup_arrangements_l587_58712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_annual_income_l587_58791

/-- Represents the monthly income of a person in Rupees -/
structure MonthlyIncome where
  value : ℕ

/-- Calculates the annual income given a monthly income -/
def annualIncome (monthlyIncome : MonthlyIncome) : ℕ :=
  12 * monthlyIncome.value

theorem a_annual_income
  (a_income b_income c_income : MonthlyIncome)
  (h1 : a_income.value * 2 = b_income.value * 5)
  (h2 : b_income.value * 100 = c_income.value * 112)
  (h3 : c_income.value = 15000) :
  annualIncome a_income = 504000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_annual_income_l587_58791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_neg_one_l587_58776

noncomputable def geometric_sequence (a q : ℝ) : ℕ → ℝ := λ n => a * q^(n-1)

noncomputable def geometric_sum (a q : ℝ) : ℕ → ℝ := λ n =>
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_ratio_neg_one (a q : ℝ) :
  let a_n := geometric_sequence a q
  let S_n := geometric_sum a q
  a_n 2 + S_n 3 = 0 → q = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_neg_one_l587_58776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l587_58728

open Set

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def U : Set ℝ := univ

theorem problem_solution :
  (A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3}) ∧
  (A ∪ (U \ B) = {x : ℝ | x ≤ 3}) ∧
  (∀ C : Set ℝ, C.Nonempty → C ⊆ A → ∀ a : ℝ, C = {x | 1 < x ∧ x < a} → 1 < a ∧ a ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l587_58728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l587_58723

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (2, 4)

theorem perpendicular_vectors (l : ℝ) : 
  (a.1 - l * b.1) * b.1 + (a.2 - l * b.2) * b.2 = 0 → l = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l587_58723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l587_58719

/-- The function f --/
noncomputable def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x

/-- The function g with parameter a --/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/3)*x^3 - ((a+1)/2)*x^2 + a*x - 1/3

/-- The theorem stating the range of a --/
theorem range_of_a :
  ∀ (a : ℝ), a > 1 →
  (∀ (x₁ : ℝ), x₁ ∈ Set.Icc 0 4 →
    ∃ (x₂ : ℝ), x₂ ∈ Set.Icc 0 4 ∧ f x₁ = g a x₂) →
  a ∈ Set.Ioo 1 (9/4) ∪ Set.Ici 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l587_58719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_60_degree_angle_l587_58787

noncomputable def angle_to_x_axis (P : ℝ × ℝ) : ℝ :=
  Real.arctan (P.2 / P.1)

theorem point_on_60_degree_angle (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = 4 ∧ P.2 = a ∧ angle_to_x_axis P = π/3) → a = 4 * Real.sqrt 3 := by
  sorry

/- Explanation of the changes:
   - We keep the broad import of Mathlib
   - We define 'angle_to_x_axis' as a noncomputable function
   - We use 'by sorry' to skip the proof
   - The structure of the theorem remains the same
-/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_60_degree_angle_l587_58787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_height_for_specific_triangle_l587_58721

/-- Triangle PQR with sides PQ, QR, PR --/
structure Triangle where
  PQ : ℝ
  QR : ℝ
  PR : ℝ

/-- The maximum height of a table constructed from a triangle --/
noncomputable def max_table_height (t : Triangle) : ℝ :=
  30 * Real.sqrt 1287 / 58

/-- Theorem stating the maximum height of the table for the given triangle --/
theorem max_height_for_specific_triangle :
  let t : Triangle := ⟨24, 26, 28⟩
  max_table_height t = 30 * Real.sqrt 1287 / 58 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_height_for_specific_triangle_l587_58721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_theorem_l587_58707

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Checks if a rectangle intersects with a circle -/
def intersects (rect : Rectangle) (p : ℝ × ℝ) (circle : Circle) : Prop :=
  sorry

/-- The main theorem statement -/
theorem garden_theorem (circles : Finset Circle) (large_rect : Rectangle) : 
  large_rect.width = 120 ∧ 
  large_rect.height = 100 ∧ 
  circles.card = 9 ∧ 
  (∀ c ∈ circles, c.radius = 2.5) →
  ∃ (small_rect : Rectangle) (p : ℝ × ℝ), 
    small_rect.width = 25 ∧ 
    small_rect.height = 35 ∧
    (∀ c ∈ circles, ¬ intersects small_rect p c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_theorem_l587_58707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l587_58773

theorem angle_sum_theorem (α β : ℝ) 
  (h1 : Real.sin (π / 4 - α) = -Real.sqrt 5 / 5)
  (h2 : Real.sin (π / 4 + β) = 3 * Real.sqrt 10 / 10)
  (h3 : 0 < α) (h4 : α < π / 2)
  (h5 : π / 4 < β) (h6 : β < π / 2) :
  α + β = 3 * π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l587_58773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_quantity_is_fifty_liters_l587_58722

/-- Represents the contents of a can with milk and water -/
structure Can where
  milk : ℚ
  water : ℚ

/-- Calculates the ratio of milk to water in a can -/
noncomputable def ratio (c : Can) : ℚ := c.milk / c.water

theorem initial_quantity_is_fifty_liters 
  (initial : Can) 
  (final : Can) 
  (h1 : ratio initial = 3) 
  (h2 : ratio final = 1/3) 
  (h3 : final.water = initial.water + 100) 
  (h4 : final.milk = initial.milk) : 
  initial.milk + initial.water = 50 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_quantity_is_fifty_liters_l587_58722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decaf_percentage_after_purchase_l587_58713

/-- Represents the percentage of decaffeinated coffee in the total stock after a new purchase --/
noncomputable def decaf_percentage (initial_stock : ℝ) (type_a_percent : ℝ) (type_b_percent : ℝ) (type_c_percent : ℝ)
  (type_a_decaf : ℝ) (type_b_decaf : ℝ) (type_c_decaf : ℝ)
  (additional_stock : ℝ) (new_type_a_percent : ℝ) (new_type_b_percent : ℝ) (new_type_c_percent : ℝ)
  (new_type_a_decaf : ℝ) (new_type_b_decaf : ℝ) (new_type_c_decaf : ℝ) : ℝ :=
  let initial_decaf := initial_stock * (type_a_percent * type_a_decaf + type_b_percent * type_b_decaf + type_c_percent * type_c_decaf)
  let additional_decaf := additional_stock * (new_type_a_percent * new_type_a_decaf + new_type_b_percent * new_type_b_decaf + new_type_c_percent * new_type_c_decaf)
  let total_stock := initial_stock + additional_stock
  let total_decaf := initial_decaf + additional_decaf
  (total_decaf / total_stock) * 100

/-- Theorem stating that the percentage of decaffeinated coffee in the total stock after the new purchase is approximately 26.18% --/
theorem decaf_percentage_after_purchase :
  ∃ ε > 0, abs (decaf_percentage 800 0.40 0.35 0.25 0.20 0.50 0 300 0.50 0.30 0.20 0.25 0.45 0.10 - 26.18) < ε :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decaf_percentage_after_purchase_l587_58713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jogging_speed_l587_58781

/-- Represents a rectangular track with semicircular ends -/
structure Track where
  straight_length : ℝ
  width : ℝ

/-- Calculates the inner perimeter of the track -/
noncomputable def inner_perimeter (t : Track) : ℝ :=
  2 * t.straight_length + 2 * Real.pi * (t.width / 2)

/-- Calculates the outer perimeter of the track -/
noncomputable def outer_perimeter (t : Track) : ℝ :=
  2 * t.straight_length + 2 * Real.pi * (t.width * 3 / 2)

/-- The theorem to be proved -/
theorem jogging_speed (t : Track) (time_diff : ℝ) : 
  t.straight_length = 80 ∧ t.width = 10 ∧ time_diff = 60 →
  ∃ (speed : ℝ), speed = Real.pi / 3 ∧ 
    outer_perimeter t / speed = inner_perimeter t / speed + time_diff :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jogging_speed_l587_58781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_in_still_water_l587_58748

/-- The speed of a man in still water, given his upstream and downstream speeds -/
noncomputable def speed_in_still_water (upstream_speed downstream_speed : ℝ) : ℝ :=
  (upstream_speed + downstream_speed) / 2

/-- Theorem: The speed of a man in still water is 40 kmph -/
theorem man_speed_in_still_water :
  speed_in_still_water 20 60 = 40 := by
  -- Unfold the definition of speed_in_still_water
  unfold speed_in_still_water
  -- Simplify the arithmetic
  simp [add_div]
  -- Check that 80 / 2 = 40
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_in_still_water_l587_58748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_range_of_a_theorem_l587_58757

-- Define the function f
def f (x : ℝ) : ℝ := abs (2 * x - 1)

-- Theorem 1
theorem solution_set_theorem (x : ℝ) :
  x ∈ Set.Icc (-1/4 : ℝ) (9/4 : ℝ) ↔ f x ≤ 5 - f (x - 1) :=
sorry

-- Theorem 2
theorem range_of_a_theorem (a : ℝ) :
  (∀ x ∈ Set.Ioo (1/2 : ℝ) (1 : ℝ), f x ≤ f (x + a) - abs (x - a)) →
  a ∈ Set.Icc (-1 : ℝ) (5/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_theorem_range_of_a_theorem_l587_58757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_l587_58737

/-- The golden ratio minus 1 -/
noncomputable def φ : ℝ := (Real.sqrt 5 - 1) / 2

/-- The sequence a_n -/
noncomputable def a (n : ℕ) : ℝ := φ^n

theorem unique_sequence :
  (a 0 = 1) ∧
  (∀ n, a n > 0) ∧
  (∀ n, a n - a (n + 1) = a (n + 2)) ∧
  (∀ b : ℕ → ℝ, (b 0 = 1 ∧ (∀ n, b n > 0) ∧ (∀ n, b n - b (n + 1) = b (n + 2))) → b = a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_l587_58737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_transformation_l587_58711

/-- A function that, when transformed, results in y = 1/2 * sin(x) -/
noncomputable def f (x : ℝ) : ℝ := 1/2 * Real.sin (2*x - Real.pi/2)

/-- The transformation applied to f -/
noncomputable def transform (g : ℝ → ℝ) (x : ℝ) : ℝ := g (x/2 + Real.pi/2)

theorem f_transformation :
  ∀ x, transform f x = 1/2 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_transformation_l587_58711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deluxe_stereo_production_time_fraction_l587_58777

theorem deluxe_stereo_production_time_fraction :
  ∀ (T : ℕ) (B : ℝ),
    T > 0 →
    B > 0 →
    (let basic_count : ℝ := (2 / 3) * T
     let deluxe_count : ℝ := T - basic_count
     let basic_time : ℝ := B
     let deluxe_time : ℝ := (7 / 5) * B
     let total_basic_time : ℝ := basic_count * basic_time
     let total_deluxe_time : ℝ := deluxe_count * deluxe_time
     let total_time : ℝ := total_basic_time + total_deluxe_time
     total_deluxe_time / total_time = 7 / 17) := by
  intros T B hT hB
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_deluxe_stereo_production_time_fraction_l587_58777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l587_58706

noncomputable def f (x : ℝ) := Real.cos (4 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f x = Real.cos (4 * x + Real.pi / 3)) ∧
  (∃ p, p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q) ∧
  (let p := Real.pi / 2; p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q) ∧
  (∀ x, f (-(x + Real.pi / 12)) = f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l587_58706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_truth_tellers_is_eight_l587_58738

/-- Represents a person on the island -/
structure Person where
  number : ℤ
  truthTeller : Bool

/-- The statements made by a person -/
def validStatements (p : Person) : Prop :=
  ∃ (n m : ℕ), n ≤ 10 ∧ m ≤ 10 ∧
    (p.truthTeller → (p.number > n ∧ p.number < m))

/-- The maximum number of truth-tellers possible -/
def maxTruthTellers (people : List Person) : ℕ :=
  (people.filter (λ p => p.truthTeller)).length

/-- The main theorem -/
theorem max_truth_tellers_is_eight :
  ∀ (people : List Person),
    people.length = 10 →
    (∀ p ∈ people, validStatements p) →
    maxTruthTellers people ≤ 8 := by
  sorry

#check max_truth_tellers_is_eight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_truth_tellers_is_eight_l587_58738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_fourth_square_l587_58775

/-- Given a square with side length s, returns the side length of the next square
    formed by bisecting the sides of the given square. -/
noncomputable def next_square_side (s : ℝ) : ℝ := s * Real.sqrt 2 / 2

/-- Given an initial square S1 with area 25, calculates the area of the fourth square S4
    in the sequence where each subsequent square is formed by bisecting the sides of
    the previous square. -/
theorem area_of_fourth_square (s1 : ℝ) (h1 : s1^2 = 25) : 
  (next_square_side (next_square_side (next_square_side s1)))^2 = 3.125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_fourth_square_l587_58775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_expression_identification_l587_58752

/-- A fractional expression is an expression where the denominator contains variables. -/
def IsFractionalExpression (numerator denominator : ℝ → ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, denominator x y ≠ 0 ∧ ¬(∃ c : ℝ, ∀ x y, denominator x y = c)

theorem fractional_expression_identification :
  let expr1 := fun (x : ℝ) (_ : ℝ) => (8 * x) / (3 * Real.pi)
  let expr2 := fun (x : ℝ) (y : ℝ) => (x^2 - y^2) / (x - y)
  let expr3 := fun (x : ℝ) (y : ℝ) => (x - y) / 5
  let expr4 := fun (_ : ℝ) (_ : ℝ) => 5 / 8
  IsFractionalExpression (fun x y => x^2 - y^2) (fun x y => x - y) ∧
  ¬IsFractionalExpression (fun x _ => 8 * x) (fun _ _ => 3 * Real.pi) ∧
  ¬IsFractionalExpression (fun x y => x - y) (fun _ _ => 5) ∧
  ¬IsFractionalExpression (fun _ _ => 5) (fun _ _ => 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_expression_identification_l587_58752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l587_58779

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x + Real.sin (2 * x)

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ (t x : ℝ), a * Real.sin t + 2 * a + 1 ≥ f x) → a ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l587_58779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l587_58726

def parabola (c d x : ℝ) : ℝ := -x^2 + c*x + d

theorem parabola_vertex (c d : ℝ) :
  (∀ x, parabola c d x ≤ 0 ↔ x ∈ Set.Ici 7 ∪ Set.Iic (-1)) →
  ∃ (h : ℝ), parabola c d 3 = 16 ∧ ∀ x, parabola c d x ≤ parabola c d 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_l587_58726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l587_58716

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - x) + Real.sqrt (x - 1)

-- State the theorem
theorem f_properties :
  (∀ x, f x ≠ 0 → 1 ≤ x ∧ x ≤ 4) ∧
  (∀ a, (∃ x ∈ Set.Icc a (a + 1), ∀ y ∈ Set.Icc a (a + 1), f y ≤ f x) ↔ 3/2 < a ∧ a ≤ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l587_58716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_little_theorem_l587_58733

theorem fermats_little_theorem (a n : ℕ) (h1 : a ≠ 0) (h2 : n ≠ 0) (h3 : Nat.Coprime a n) :
  n ∣ a^(Nat.totient n) - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermats_little_theorem_l587_58733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheepdog_catch_time_l587_58763

/-- The time it takes for a sheepdog to catch a sheep -/
noncomputable def catchTime (sheepSpeed dogSpeed initialDistance : ℝ) : ℝ :=
  initialDistance / (dogSpeed - sheepSpeed)

/-- Theorem: The sheepdog catches the sheep in 20 seconds -/
theorem sheepdog_catch_time :
  let sheepSpeed : ℝ := 12
  let dogSpeed : ℝ := 20
  let initialDistance : ℝ := 160
  catchTime sheepSpeed dogSpeed initialDistance = 20 := by
  -- Unfold the definition of catchTime
  unfold catchTime
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheepdog_catch_time_l587_58763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_alpha_l587_58720

theorem tan_pi_minus_alpha (α : ℝ) (h : Real.sin (α - Real.pi) = 3 * Real.cos α) : 
  Real.tan (Real.pi - α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_alpha_l587_58720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l587_58702

theorem problem_statement : -1^2023 + Real.sqrt 4 - (8 : ℝ)^(1/3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l587_58702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremal_value_theorem_l587_58785

-- Define the coordinate system and points
variable (a b c : ℝ)
variable (A B C : ℝ × ℝ)

-- Define the conditions
axiom A_on_x : A = (a, 0)
axiom B_on_x : B = (b, 0)
axiom C_on_y : C = (0, c)
axiom special_case : c^2 = a * b

-- Define the function for (MA * MB) / MC²
noncomputable def f (x : ℝ) : ℝ := ((x - a) * (x - b)) / (x^2 + c^2)

-- State the theorem
theorem extremal_value_theorem :
  ∃ m : ℝ, (∀ x : ℝ, f a b c x ≤ m ∨ f a b c x ≥ m) ∧ 
  (m = 1 + (a + b) / (2 * Real.sqrt (a * b)) ∨
   m = 1 - (a + b) / (2 * Real.sqrt (a * b))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremal_value_theorem_l587_58785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_l587_58717

/-- Check if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given sticks -/
def stick1 : ℝ := 2
def stick2 : ℝ := 5

/-- Possible third stick lengths -/
def options : List ℝ := [2, 3, 5, 7]

theorem triangle_formation :
  ∃! x, x ∈ options ∧ can_form_triangle stick1 stick2 x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_l587_58717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_equality_l587_58797

theorem root_difference_equality (p q a b : ℝ) (m n : ℕ+) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0) →  -- x^2 + px + q has two distinct roots
  (a^2 - 4*b).sqrt = (b^2 - 4*a).sqrt →  -- positive difference between roots of x^2 + ax + b and x^2 + bx + a are equal
  (a^2 - 4*b).sqrt = 2 * |a - b| →  -- twice the positive difference between roots of x^2 + px + q equals the other differences
  q = m / n →  -- q can be expressed as m/n
  Nat.Coprime m n →  -- m and n are coprime
  m + n = 21 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_equality_l587_58797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crushing_load_for_given_dimensions_l587_58742

/-- The crushing load formula for cylindrical pillars -/
noncomputable def crushing_load (T : ℝ) (H : ℝ) : ℝ := 30 * T^3 / H

/-- Proof that the crushing load is 375 for given thickness and height -/
theorem crushing_load_for_given_dimensions :
  let T : ℝ := 5
  let H : ℝ := 10
  crushing_load T H = 375 := by
  -- Unfold the definition of crushing_load
  unfold crushing_load
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crushing_load_for_given_dimensions_l587_58742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_APPLES_paths_l587_58793

/-- Represents the triangular arrangement of letters --/
structure LetterArrangement where
  rows : ℕ
  letters : List (List Char)

/-- Represents a path in the letter arrangement --/
inductive CustomPath
  | empty : CustomPath
  | cons : Char → CustomPath → CustomPath

/-- Checks if a path spells "APPLES" --/
def spellsAPPLES : CustomPath → Bool :=
  sorry

/-- Counts the number of valid paths spelling "APPLES" --/
def countValidPaths (arrangement : LetterArrangement) : ℕ :=
  sorry

/-- The specific letter arrangement given in the problem --/
def givenArrangement : LetterArrangement :=
  { rows := 7,
    letters := [
      ['A'],
      ['A', 'P', 'A'],
      ['A', 'P', 'P', 'P', 'A'],
      ['A', 'P', 'P', 'L', 'P', 'P', 'A'],
      ['A', 'P', 'P', 'L', 'E', 'L', 'P', 'P', 'A'],
      ['A', 'P', 'P', 'L', 'E', 'S', 'E', 'L', 'P', 'P', 'A'],
      ['A', 'P', 'P', 'L', 'E', 'S', 'H', 'S', 'E', 'L', 'P', 'P', 'A']
    ] }

theorem count_APPLES_paths :
  countValidPaths givenArrangement = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_APPLES_paths_l587_58793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_problem_l587_58778

/-- Distance formula between two points in a plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Simplified distance formula for points on a line parallel to y-axis -/
def distance_parallel_y (y₁ y₂ : ℝ) : ℝ :=
  |y₂ - y₁|

theorem distance_problem :
  (distance 2 4 (-3) (-8) = 13) ∧
  (distance_parallel_y 5 (-1) = 6) := by
  sorry

#check distance_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_problem_l587_58778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_n_values_l587_58744

def is_valid_arrangement (arr : Matrix (Fin 4) (Fin 4) ℕ) : Prop :=
  ∀ i j, arr i j ∈ Finset.range 16

def row_sum (arr : Matrix (Fin 4) (Fin 4) ℕ) (i : Fin 4) : ℕ :=
  (Finset.univ : Finset (Fin 4)).sum (λ j ↦ arr i j)

def col_sum (arr : Matrix (Fin 4) (Fin 4) ℕ) (j : Fin 4) : ℕ :=
  (Finset.univ : Finset (Fin 4)).sum (λ i ↦ arr i j)

def all_sums (arr : Matrix (Fin 4) (Fin 4) ℕ) : Finset ℕ :=
  (Finset.univ : Finset (Fin 4)).image (row_sum arr) ∪ (Finset.univ : Finset (Fin 4)).image (col_sum arr)

theorem valid_n_values (arr : Matrix (Fin 4) (Fin 4) ℕ) (n : ℕ) :
  is_valid_arrangement arr →
  n ≥ 2 →
  (∀ s ∈ all_sums arr, s % n = 0) →
  (all_sums arr).card = 8 →
  n = 2 ∨ n = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_n_values_l587_58744
