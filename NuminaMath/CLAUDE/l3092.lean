import Mathlib

namespace NUMINAMATH_CALUDE_equation_has_integer_solution_l3092_309285

theorem equation_has_integer_solution (a b : ℤ) : ∃ x : ℤ, (x - a) * (x - b) * (x - 3) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_integer_solution_l3092_309285


namespace NUMINAMATH_CALUDE_distance_maximum_at_halfway_l3092_309203

-- Define a square in 2D space
structure Square :=
  (side : ℝ)
  (center : ℝ × ℝ)

-- Define a runner's position on the square
structure RunnerPosition :=
  (square : Square)
  (t : ℝ)  -- Parameter representing time or position along the path (0 ≤ t ≤ 4)

-- Function to calculate the runner's coordinates
def runnerCoordinates (pos : RunnerPosition) : ℝ × ℝ :=
  sorry

-- Function to calculate the straight-line distance from the starting point
def distanceFromStart (pos : RunnerPosition) : ℝ :=
  sorry

theorem distance_maximum_at_halfway (s : Square) :
  ∃ (t_max : ℝ), t_max = 2 ∧
  ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 4 →
    distanceFromStart ⟨s, t⟩ ≤ distanceFromStart ⟨s, t_max⟩ :=
sorry

end NUMINAMATH_CALUDE_distance_maximum_at_halfway_l3092_309203


namespace NUMINAMATH_CALUDE_popsicle_sticks_problem_l3092_309217

theorem popsicle_sticks_problem (steve sid sam : ℕ) : 
  sid = 2 * steve →
  sam = 3 * sid →
  steve + sid + sam = 108 →
  steve = 12 := by
sorry

end NUMINAMATH_CALUDE_popsicle_sticks_problem_l3092_309217


namespace NUMINAMATH_CALUDE_sequence_a1_value_l3092_309216

theorem sequence_a1_value (p q : ℝ) (a : ℕ → ℝ) 
  (hp : p > 0) (hq : q > 0)
  (ha_pos : ∀ n, a n > 0)
  (ha_0 : a 0 = 1)
  (ha_rec : ∀ n, a (n + 2) = p * a n - q * a (n + 1)) :
  a 1 = (-q + Real.sqrt (q^2 + 4*p)) / 2 :=
sorry

end NUMINAMATH_CALUDE_sequence_a1_value_l3092_309216


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3092_309283

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  (∃ b : ℝ, b ≠ 2 ∧ (b - 1) * (b - 2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3092_309283


namespace NUMINAMATH_CALUDE_expression_factorization_l3092_309296

theorem expression_factorization (x : ℝ) :
  (10 * x^3 + 50 * x^2 - 4) - (3 * x^3 - 5 * x^2 + 2) = 7 * x^3 + 55 * x^2 - 6 :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l3092_309296


namespace NUMINAMATH_CALUDE_puppy_weight_l3092_309220

/-- Represents the weight of animals in pounds -/
structure AnimalWeights where
  puppy : ℝ
  smaller_cat : ℝ
  larger_cat : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (w : AnimalWeights) : Prop :=
  w.puppy + 2 * w.smaller_cat + w.larger_cat = 38 ∧
  w.puppy + w.larger_cat = 3 * w.smaller_cat ∧
  w.puppy + 2 * w.smaller_cat = w.larger_cat

/-- The theorem stating the puppy's weight -/
theorem puppy_weight (w : AnimalWeights) (h : satisfies_conditions w) : w.puppy = 3.8 := by
  sorry

#check puppy_weight

end NUMINAMATH_CALUDE_puppy_weight_l3092_309220


namespace NUMINAMATH_CALUDE_paul_cookie_price_l3092_309266

/-- Represents a cookie baker -/
structure Baker where
  name : String
  num_cookies : ℕ
  price_per_cookie : ℚ

/-- The total amount of dough used by all bakers -/
def total_dough : ℝ := 120

theorem paul_cookie_price 
  (art paul : Baker)
  (h1 : art.name = "Art")
  (h2 : paul.name = "Paul")
  (h3 : art.num_cookies = 10)
  (h4 : paul.num_cookies = 20)
  (h5 : art.price_per_cookie = 1/2)
  (h6 : (total_dough / art.num_cookies) = (total_dough / paul.num_cookies)) :
  paul.price_per_cookie = 1/4 := by
sorry

end NUMINAMATH_CALUDE_paul_cookie_price_l3092_309266


namespace NUMINAMATH_CALUDE_percentage_owning_only_cats_l3092_309257

/-- The percentage of students owning only cats in a survey. -/
theorem percentage_owning_only_cats
  (total_students : ℕ)
  (cat_owners : ℕ)
  (dog_owners : ℕ)
  (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 120)
  (h3 : dog_owners = 200)
  (h4 : both_owners = 40) :
  (cat_owners - both_owners) / total_students * 100 = 16 :=
by sorry

end NUMINAMATH_CALUDE_percentage_owning_only_cats_l3092_309257


namespace NUMINAMATH_CALUDE_seven_lines_regions_l3092_309265

/-- The number of regions created by n lines in a plane, where no two are parallel and no three are concurrent -/
def regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1)) / 2

/-- Seven lines in a plane with no two parallel and no three concurrent -/
def seven_lines : ℕ := 7

theorem seven_lines_regions :
  regions seven_lines = 29 := by sorry

end NUMINAMATH_CALUDE_seven_lines_regions_l3092_309265


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l3092_309234

/-- Given a mixture of milk and water, prove that the initial volume is 60 litres -/
theorem initial_mixture_volume
  (initial_ratio : ℚ) -- Initial ratio of milk to water
  (final_ratio : ℚ) -- Final ratio of milk to water
  (added_water : ℚ) -- Amount of water added to achieve final ratio
  (h1 : initial_ratio = 2 / 1) -- Initial ratio is 2:1
  (h2 : final_ratio = 1 / 2) -- Final ratio is 1:2
  (h3 : added_water = 60) -- 60 litres of water is added
  : ℚ :=
by
  sorry

#check initial_mixture_volume

end NUMINAMATH_CALUDE_initial_mixture_volume_l3092_309234


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l3092_309299

theorem ratio_sum_theorem (a b c : ℝ) 
  (h : ∃ k : ℝ, a = 2*k ∧ b = 3*k ∧ c = 5*k) : (a + b) / c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l3092_309299


namespace NUMINAMATH_CALUDE_cos_angle_through_point_l3092_309224

/-- Given an angle α whose initial side is the positive x-axis and whose terminal side
    passes through the point (4, -3), prove that cos(α) = 4/5 -/
theorem cos_angle_through_point :
  ∀ α : ℝ,
  (∃ (x y : ℝ), x = 4 ∧ y = -3 ∧ 
    (Real.cos α * x - Real.sin α * y = x) ∧
    (Real.sin α * x + Real.cos α * y = y)) →
  Real.cos α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_through_point_l3092_309224


namespace NUMINAMATH_CALUDE_cereal_eating_time_l3092_309260

def fat_rate : ℚ := 1 / 20
def thin_rate : ℚ := 1 / 30
def average_rate : ℚ := 1 / 24
def total_cereal : ℚ := 5

theorem cereal_eating_time :
  let combined_rate := fat_rate + thin_rate + average_rate
  (total_cereal / combined_rate) = 40 := by sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l3092_309260


namespace NUMINAMATH_CALUDE_travis_apple_sales_l3092_309246

/-- Calculates the total money Travis takes home from selling apples -/
def total_money (total_apples : ℕ) (apples_per_box : ℕ) (price_per_box : ℕ) : ℕ :=
  (total_apples / apples_per_box) * price_per_box

/-- Proves that Travis will take home $7000 -/
theorem travis_apple_sales : total_money 10000 50 35 = 7000 := by
  sorry

end NUMINAMATH_CALUDE_travis_apple_sales_l3092_309246


namespace NUMINAMATH_CALUDE_dvds_per_season_l3092_309249

theorem dvds_per_season (total_dvds : ℕ) (num_seasons : ℕ) 
  (h1 : total_dvds = 40) (h2 : num_seasons = 5) : 
  total_dvds / num_seasons = 8 := by
  sorry

end NUMINAMATH_CALUDE_dvds_per_season_l3092_309249


namespace NUMINAMATH_CALUDE_part1_part2_l3092_309238

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 5 - a|

-- Part 1
theorem part1 (a : ℝ) :
  (∀ x, x ∈ Set.Icc (-5 : ℝ) (-1) ↔ f x a - |x - a| ≤ 2) →
  a = 2 := by sorry

-- Part 2
theorem part2 (a m : ℝ) :
  (∃ x₀, f x₀ a < 4 * m + m^2) →
  m ∈ Set.Ioi 1 ∪ Set.Iio (-5) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3092_309238


namespace NUMINAMATH_CALUDE_cycle_not_divisible_by_three_l3092_309261

/-- A graph is a type with an edge relation -/
class Graph (V : Type) :=
  (adj : V → V → Prop)

/-- The degree of a vertex in a graph is the number of adjacent vertices -/
def degree {V : Type} [Graph V] (v : V) : ℕ := sorry

/-- A path in a graph is a list of vertices where each consecutive pair is adjacent -/
def is_path {V : Type} [Graph V] (p : List V) : Prop := sorry

/-- A cycle in a graph is a path where the first and last vertices are the same -/
def is_cycle {V : Type} [Graph V] (c : List V) : Prop := sorry

/-- The length of a path or cycle is the number of edges it contains -/
def length {V : Type} [Graph V] (p : List V) : ℕ := sorry

theorem cycle_not_divisible_by_three 
  {V : Type} [Graph V] 
  (h : ∀ v : V, degree v ≥ 3) : 
  ∃ c : List V, is_cycle c ∧ ¬(length c % 3 = 0) := by sorry

end NUMINAMATH_CALUDE_cycle_not_divisible_by_three_l3092_309261


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l3092_309227

theorem walnut_trees_after_planting 
  (initial_trees : ℕ) 
  (new_trees : ℕ) 
  (h1 : initial_trees = 107) 
  (h2 : new_trees = 104) : 
  initial_trees + new_trees = 211 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l3092_309227


namespace NUMINAMATH_CALUDE_volleyball_team_theorem_l3092_309247

def volleyball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) : ℕ :=
  quadruplets * (Nat.choose (total_players - quadruplets) (starters - 1))

theorem volleyball_team_theorem :
  volleyball_team_selection 16 4 6 = 3168 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_theorem_l3092_309247


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3092_309225

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- State the theorem
theorem union_of_A_and_B : 
  A ∪ B = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3092_309225


namespace NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l3092_309277

theorem sqrt_inequality_solution_set (x : ℝ) :
  (x + 3 ≥ 0) → (∃ y, y > 1 ∧ x = y) ↔ (Real.sqrt (x + 3) > 3 - x) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l3092_309277


namespace NUMINAMATH_CALUDE_boat_speed_proof_l3092_309208

/-- The speed of the boat in still water -/
def boat_speed : ℝ := 10

/-- The speed of the stream -/
def stream_speed : ℝ := 2

/-- The distance traveled -/
def distance : ℝ := 36

/-- The time difference between upstream and downstream travel -/
def time_difference : ℝ := 1.5

theorem boat_speed_proof :
  (distance / (boat_speed - stream_speed) - distance / (boat_speed + stream_speed) = time_difference) ∧
  (boat_speed > stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_proof_l3092_309208


namespace NUMINAMATH_CALUDE_square_difference_of_even_integers_l3092_309295

theorem square_difference_of_even_integers (x y : ℕ) : 
  Even x → Even y → x > y → x + y = 68 → x - y = 20 → x^2 - y^2 = 1360 :=
by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_even_integers_l3092_309295


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3092_309222

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, ∃ r : ℝ, r ≠ 0 ∧ a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3092_309222


namespace NUMINAMATH_CALUDE_line_parallel_to_polar_axis_l3092_309235

/-- Given a point P in polar coordinates (r, θ), prove that the equation r * sin(θ) = 1
    represents a line that passes through P and is parallel to the polar axis. -/
theorem line_parallel_to_polar_axis 
  (r : ℝ) (θ : ℝ) (h1 : r = 2) (h2 : θ = π / 6) :
  r * Real.sin θ = 1 := by sorry

end NUMINAMATH_CALUDE_line_parallel_to_polar_axis_l3092_309235


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3092_309204

theorem quadratic_expression_value (x : ℝ) : 
  x = -2 → x^2 + 6*x - 8 = -16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3092_309204


namespace NUMINAMATH_CALUDE_circle_equation_l3092_309293

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: For a circle with center (4, -6) and radius 3,
    any point (x, y) on the circle satisfies (x - 4)^2 + (y + 6)^2 = 9 -/
theorem circle_equation (c : Circle) (p : Point) :
  c.h = 4 ∧ c.k = -6 ∧ c.r = 3 →
  (p.x - c.h)^2 + (p.y - c.k)^2 = c.r^2 →
  (p.x - 4)^2 + (p.y + 6)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l3092_309293


namespace NUMINAMATH_CALUDE_front_axle_wheels_l3092_309229

/-- The toll formula for a truck crossing a bridge -/
def toll (x : ℕ) : ℚ :=
  3.5 + 0.5 * (x - 2)

/-- The number of axles for an 18-wheel truck with f wheels on the front axle -/
def num_axles (f : ℕ) : ℕ :=
  1 + (18 - f) / 4

theorem front_axle_wheels :
  ∃ (f : ℕ), f > 0 ∧ f < 18 ∧ 
  toll (num_axles f) = 5 ∧
  f = 2 := by
  sorry

end NUMINAMATH_CALUDE_front_axle_wheels_l3092_309229


namespace NUMINAMATH_CALUDE_min_sum_of_equal_multiples_l3092_309218

theorem min_sum_of_equal_multiples (x y z : ℕ+) (h : (4 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val) :
  ∃ (m : ℕ+), ∀ (a b c : ℕ+), ((4 : ℕ) * a.val = (5 : ℕ) * b.val ∧ (5 : ℕ) * b.val = (6 : ℕ) * c.val) →
    m.val ≤ a.val + b.val + c.val ∧ m.val = x.val + y.val + z.val ∧ m.val = 37 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_equal_multiples_l3092_309218


namespace NUMINAMATH_CALUDE_election_majority_l3092_309212

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 800 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = 320 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_l3092_309212


namespace NUMINAMATH_CALUDE_power_equation_l3092_309255

theorem power_equation (m n : ℕ) (h1 : 2^m = 3) (h2 : 2^n = 4) : 2^(3*m - 2*n) = 27/16 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l3092_309255


namespace NUMINAMATH_CALUDE_green_light_probability_l3092_309242

def traffic_light_cycle : ℕ := 60
def red_light_duration : ℕ := 30
def green_light_duration : ℕ := 25
def yellow_light_duration : ℕ := 5

theorem green_light_probability :
  (green_light_duration : ℚ) / traffic_light_cycle = 5 / 12 := by
sorry

end NUMINAMATH_CALUDE_green_light_probability_l3092_309242


namespace NUMINAMATH_CALUDE_convex_lattice_pentagon_contains_lattice_point_l3092_309240

/-- A point in the 2D integer lattice -/
def LatticePoint := (ℤ × ℤ)

/-- A pentagon in the 2D plane -/
structure Pentagon where
  vertices : Fin 5 → ℝ × ℝ

/-- A convex pentagon is one where all interior angles are less than or equal to 180 degrees -/
def IsConvex (p : Pentagon) : Prop := sorry

/-- A lattice pentagon is one where all vertices are lattice points -/
def IsLatticePentagon (p : Pentagon) : Prop :=
  ∀ i : Fin 5, ∃ (x y : ℤ), p.vertices i = (↑x, ↑y)

/-- The interior of a pentagon includes all points strictly inside the pentagon -/
def Interior (p : Pentagon) : Set (ℝ × ℝ) := sorry

/-- The boundary of a pentagon includes all points on the edges of the pentagon, excluding the vertices -/
def Boundary (p : Pentagon) : Set (ℝ × ℝ) := sorry

/-- Main theorem: Any convex lattice pentagon contains a lattice point in its interior or on its boundary -/
theorem convex_lattice_pentagon_contains_lattice_point (p : Pentagon) 
  (h_convex : IsConvex p) (h_lattice : IsLatticePentagon p) :
  ∃ (point : LatticePoint), (↑point.1, ↑point.2) ∈ Interior p ∪ Boundary p :=
sorry

end NUMINAMATH_CALUDE_convex_lattice_pentagon_contains_lattice_point_l3092_309240


namespace NUMINAMATH_CALUDE_balloon_problem_solution_l3092_309282

/-- The total number of balloons Brooke and Tracy have after Tracy pops half of hers -/
def total_balloons (brooke_initial : ℕ) (brooke_added : ℕ) (tracy_initial : ℕ) (tracy_added : ℕ) : ℕ :=
  let brooke_total := brooke_initial + brooke_added
  let tracy_before_popping := tracy_initial + tracy_added
  let tracy_after_popping := tracy_before_popping / 2
  brooke_total + tracy_after_popping

/-- Theorem stating that the total number of balloons is 35 given the problem conditions -/
theorem balloon_problem_solution :
  total_balloons 12 8 6 24 = 35 := by
  sorry

end NUMINAMATH_CALUDE_balloon_problem_solution_l3092_309282


namespace NUMINAMATH_CALUDE_cake_box_width_l3092_309288

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

theorem cake_box_width :
  let carton := BoxDimensions.mk 25 42 60
  let cakeBox := BoxDimensions.mk 8 W 5
  let maxBoxes := 210
  boxVolume carton = maxBoxes * boxVolume cakeBox →
  W = 7.5 := by
sorry

end NUMINAMATH_CALUDE_cake_box_width_l3092_309288


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3092_309289

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x < -1 ∨ 4 < x}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3092_309289


namespace NUMINAMATH_CALUDE_sin_120_degrees_l3092_309258

theorem sin_120_degrees : Real.sin (2 * π / 3) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l3092_309258


namespace NUMINAMATH_CALUDE_max_xy_geometric_mean_l3092_309209

theorem max_xy_geometric_mean (x y : ℝ) : 
  x^2 = (1 + 2*y) * (1 - 2*y) → 
  ∃ (k : ℝ), k = x*y ∧ k ≤ (1/4 : ℝ) ∧ 
  ∃ (x₀ y₀ : ℝ), x₀^2 = (1 + 2*y₀) * (1 - 2*y₀) ∧ x₀ * y₀ = (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_xy_geometric_mean_l3092_309209


namespace NUMINAMATH_CALUDE_complex_sum_real_imag_l3092_309210

theorem complex_sum_real_imag (a : ℝ) : 
  let z : ℂ := a / (2 + Complex.I) + (2 + Complex.I) / 5
  (z.re + z.im = 1) → a = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_real_imag_l3092_309210


namespace NUMINAMATH_CALUDE_trapezoid_cd_length_l3092_309233

structure Trapezoid (A B C D : ℝ × ℝ) :=
  (parallel : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1))
  (bd_length : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 2)
  (angle_dbc : Real.arccos ((B.1 - D.1) * (C.1 - B.1) + (B.2 - D.2) * (C.2 - B.2)) / 
    (Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) * Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)) = 36 * π / 180)
  (angle_bda : Real.arccos ((B.1 - D.1) * (A.1 - D.1) + (B.2 - D.2) * (A.2 - D.2)) / 
    (Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) * Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)) = 72 * π / 180)
  (ratio : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) / Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 5/3)

theorem trapezoid_cd_length (A B C D : ℝ × ℝ) (t : Trapezoid A B C D) :
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_cd_length_l3092_309233


namespace NUMINAMATH_CALUDE_vector_dot_product_equation_l3092_309236

/-- Given vectors a, b, and c satisfying certain conditions, prove that x = 4 -/
theorem vector_dot_product_equation (a b c : ℝ × ℝ) (x : ℝ) 
  (ha : a = (1, 1))
  (hb : b = (2, 5))
  (hc : c = (3, x))
  (h_dot : ((8 • a - b) • c) = 30) :
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_equation_l3092_309236


namespace NUMINAMATH_CALUDE_f_always_negative_iff_a_in_range_l3092_309214

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 1

theorem f_always_negative_iff_a_in_range :
  (∀ x : ℝ, f a x < 0) ↔ -4 < a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_f_always_negative_iff_a_in_range_l3092_309214


namespace NUMINAMATH_CALUDE_staircase_region_perimeter_l3092_309230

/-- Represents the staircase-shaped region with an adjoined right triangle -/
structure StaircaseRegion where
  staircase_side_length : ℝ
  staircase_side_count : ℕ
  triangle_leg1 : ℝ
  triangle_leg2 : ℝ
  total_area : ℝ

/-- Calculates the perimeter of the StaircaseRegion -/
def calculate_perimeter (region : StaircaseRegion) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific StaircaseRegion -/
theorem staircase_region_perimeter :
  let region : StaircaseRegion := {
    staircase_side_length := 2,
    staircase_side_count := 10,
    triangle_leg1 := 3,
    triangle_leg2 := 4,
    total_area := 150
  }
  calculate_perimeter region = 81.77 := by
  sorry

end NUMINAMATH_CALUDE_staircase_region_perimeter_l3092_309230


namespace NUMINAMATH_CALUDE_max_value_quadratic_inequality_l3092_309278

/-- Given that the solution set of ax^2 + 2x + c ≤ 0 is {x | x = -1/a} and a > c,
    prove that the maximum value of (a-c)/(a^2+c^2) is √2/4 -/
theorem max_value_quadratic_inequality (a c : ℝ) 
    (h1 : ∀ x, a * x^2 + 2 * x + c ≤ 0 ↔ x = -1/a)
    (h2 : a > c) :
    (∀ a' c', a' > c' → (a' - c') / (a'^2 + c'^2) ≤ Real.sqrt 2 / 4) ∧ 
    (∃ a' c', a' > c' ∧ (a' - c') / (a'^2 + c'^2) = Real.sqrt 2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_inequality_l3092_309278


namespace NUMINAMATH_CALUDE_average_headcount_proof_l3092_309232

def fall_headcount_03_04 : ℕ := 11500
def fall_headcount_04_05 : ℕ := 11600
def fall_headcount_05_06 : ℕ := 11300

def average_headcount : ℕ := 
  (fall_headcount_03_04 + fall_headcount_04_05 + fall_headcount_05_06 + 1) / 3

theorem average_headcount_proof :
  average_headcount = 11467 := by sorry

end NUMINAMATH_CALUDE_average_headcount_proof_l3092_309232


namespace NUMINAMATH_CALUDE_total_travel_time_l3092_309284

def distance_washington_idaho : ℝ := 640
def distance_idaho_nevada : ℝ := 550
def speed_washington_idaho : ℝ := 80
def speed_idaho_nevada : ℝ := 50

theorem total_travel_time :
  (distance_washington_idaho / speed_washington_idaho) +
  (distance_idaho_nevada / speed_idaho_nevada) = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_l3092_309284


namespace NUMINAMATH_CALUDE_digitSum5_125th_l3092_309274

/-- The sum of digits of a natural number --/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence of natural numbers with digit sum 5 in ascending order --/
def digitSum5Seq : ℕ → ℕ := sorry

/-- The 125th number in the sequence of natural numbers with digit sum 5 --/
theorem digitSum5_125th :
  digitSum5Seq 125 = 41000 ∧ sumOfDigits (digitSum5Seq 125) = 5 := by sorry

end NUMINAMATH_CALUDE_digitSum5_125th_l3092_309274


namespace NUMINAMATH_CALUDE_hockey_season_games_l3092_309280

/-- Calculate the number of games in a hockey season -/
theorem hockey_season_games (n : ℕ) (m : ℕ) (h1 : n = 16) (h2 : m = 10) :
  (n * (n - 1) / 2) * m = 2400 := by
  sorry

end NUMINAMATH_CALUDE_hockey_season_games_l3092_309280


namespace NUMINAMATH_CALUDE_ice_cream_cost_l3092_309271

theorem ice_cream_cost (total_spent : ℚ) (apple_extra_cost : ℚ) 
  (h1 : total_spent = 25)
  (h2 : apple_extra_cost = 10) : 
  ∃ (ice_cream_cost : ℚ), 
    ice_cream_cost + (ice_cream_cost + apple_extra_cost) = total_spent ∧ 
    ice_cream_cost = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l3092_309271


namespace NUMINAMATH_CALUDE_amy_school_year_hours_l3092_309276

/-- Amy's summer work and earnings information -/
structure SummerWork where
  hours_per_week : ℕ
  weeks : ℕ
  total_earnings : ℕ

/-- Amy's school year work plan -/
structure SchoolYearPlan where
  weeks : ℕ
  target_earnings : ℕ

/-- Calculate required weekly hours for school year -/
def required_weekly_hours (summer : SummerWork) (school : SchoolYearPlan) : ℕ :=
  15

/-- Theorem: Amy must work 15 hours per week during the school year -/
theorem amy_school_year_hours 
  (summer : SummerWork) 
  (school : SchoolYearPlan) 
  (h1 : summer.hours_per_week = 45)
  (h2 : summer.weeks = 8)
  (h3 : summer.total_earnings = 3600)
  (h4 : school.weeks = 24)
  (h5 : school.target_earnings = 3600) :
  required_weekly_hours summer school = 15 := by
  sorry

#check amy_school_year_hours

end NUMINAMATH_CALUDE_amy_school_year_hours_l3092_309276


namespace NUMINAMATH_CALUDE_complex_z_value_l3092_309292

def is_negative_real (z : ℂ) : Prop := ∃ (r : ℝ), r < 0 ∧ z = r

def is_purely_imaginary (z : ℂ) : Prop := ∃ (r : ℝ), z = r * Complex.I

theorem complex_z_value (z : ℂ) 
  (h1 : is_negative_real ((z - 3*Complex.I) / (z + Complex.I)))
  (h2 : is_purely_imaginary ((z - 3) / (z + 1))) :
  z = Complex.I * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_z_value_l3092_309292


namespace NUMINAMATH_CALUDE_smallest_multiple_1_to_10_l3092_309243

theorem smallest_multiple_1_to_10 : ∀ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) → n ≥ 2520 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_1_to_10_l3092_309243


namespace NUMINAMATH_CALUDE_alternatingArrangements_4_3_l3092_309250

/-- The number of ways to arrange 4 men and 3 women in a row, such that no two men or two women are adjacent -/
def alternatingArrangements (numMen : Nat) (numWomen : Nat) : Nat :=
  Nat.factorial numMen * 
  (Nat.choose (numMen + 1) numWomen) * 
  Nat.factorial numWomen

/-- Theorem stating that the number of alternating arrangements of 4 men and 3 women is 1440 -/
theorem alternatingArrangements_4_3 : 
  alternatingArrangements 4 3 = 1440 := by
  sorry

#eval alternatingArrangements 4 3

end NUMINAMATH_CALUDE_alternatingArrangements_4_3_l3092_309250


namespace NUMINAMATH_CALUDE_true_discount_calculation_l3092_309279

/-- Given a present worth and banker's discount, calculate the true discount -/
theorem true_discount_calculation (present_worth banker_discount : ℚ) :
  present_worth = 400 →
  banker_discount = 21 →
  ∃ true_discount : ℚ,
    banker_discount = true_discount + (true_discount * banker_discount / present_worth) ∧
    true_discount = 8400 / 421 :=
by sorry

end NUMINAMATH_CALUDE_true_discount_calculation_l3092_309279


namespace NUMINAMATH_CALUDE_largest_common_term_less_than_800_l3092_309263

def arithmetic_progression_1 (n : ℕ) : ℤ := 4 + 5 * n
def arithmetic_progression_2 (m : ℕ) : ℤ := 7 + 8 * m

def is_common_term (a : ℤ) : Prop :=
  ∃ n m : ℕ, arithmetic_progression_1 n = a ∧ arithmetic_progression_2 m = a

theorem largest_common_term_less_than_800 :
  ∃ a : ℤ, is_common_term a ∧ a < 800 ∧ ∀ b : ℤ, is_common_term b ∧ b < 800 → b ≤ a :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_less_than_800_l3092_309263


namespace NUMINAMATH_CALUDE_three_hour_classes_count_l3092_309239

/-- Represents the data analytics course structure and duration --/
structure CourseStructure where
  duration : ℕ            -- Course duration in weeks
  fourHourClasses : ℕ     -- Number of four-hour classes per week
  homeworkHours : ℕ       -- Hours spent on homework per week
  totalHours : ℕ          -- Total hours spent on the course
  threeHourClasses : ℕ    -- Number of three-hour classes per week (to be proved)

/-- Theorem stating the number of three-hour classes per week --/
theorem three_hour_classes_count (c : CourseStructure) 
  (h1 : c.duration = 24)
  (h2 : c.fourHourClasses = 1)
  (h3 : c.homeworkHours = 4)
  (h4 : c.totalHours = 336) :
  c.threeHourClasses = 2 := by
  sorry

#check three_hour_classes_count

end NUMINAMATH_CALUDE_three_hour_classes_count_l3092_309239


namespace NUMINAMATH_CALUDE_inequality_problem_l3092_309207

/-- Given positive real numbers a and b such that 1/a + 1/b = 2√2, 
    prove the minimum value of a² + b² and the value of ab under certain conditions. -/
theorem inequality_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h : 1/a + 1/b = 2 * Real.sqrt 2) : 
  (∃ (min : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 2 * Real.sqrt 2 → x^2 + y^2 ≥ min ∧ 
    a^2 + b^2 = min) ∧ 
  ((a - b)^2 ≥ 4 * (a*b)^3 → a * b = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l3092_309207


namespace NUMINAMATH_CALUDE_evaluate_expression_l3092_309298

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3092_309298


namespace NUMINAMATH_CALUDE_unique_prime_product_l3092_309281

theorem unique_prime_product (p q r : Nat) : 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  p * q * r = 7802 ∧
  p + q + r = 1306 →
  ∀ (p1 p2 p3 : Nat), 
    Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    p1 * p2 * p3 ≠ 7802 ∧
    p1 + p2 + p3 = 1306 →
    False :=
by sorry

#check unique_prime_product

end NUMINAMATH_CALUDE_unique_prime_product_l3092_309281


namespace NUMINAMATH_CALUDE_range_of_f_l3092_309231

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3092_309231


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3092_309252

def f (x : ℝ) : ℝ := x^2 - 3*x + 1

theorem arithmetic_sequence_formula (a : ℝ) (a_n : ℕ → ℝ) :
  (∀ n, a_n (n + 2) - a_n (n + 1) = a_n (n + 1) - a_n n) →  -- arithmetic sequence
  a_n 1 = f (a + 1) →
  a_n 2 = 0 →
  a_n 3 = f (a - 1) →
  ((a = 1 ∧ ∀ n, a_n n = n - 2) ∨ (a = 2 ∧ ∀ n, a_n n = 2 - n)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3092_309252


namespace NUMINAMATH_CALUDE_parabola_m_value_l3092_309200

/-- A parabola with equation x² = my and a point M(x₀, -3) on it. -/
structure Parabola where
  m : ℝ
  x₀ : ℝ
  eq : x₀^2 = m * (-3)

/-- The distance from a point to the focus of the parabola. -/
def distance_to_focus (p : Parabola) : ℝ := 5

/-- Theorem: If a point M(x₀, -3) on the parabola x² = my has a distance of 5 to the focus, then m = -8. -/
theorem parabola_m_value (p : Parabola) (h : distance_to_focus p = 5) : p.m = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_m_value_l3092_309200


namespace NUMINAMATH_CALUDE_other_number_difference_l3092_309272

theorem other_number_difference (x : ℕ) (h1 : x + 42 = 96) : x = 54 := by
  sorry

#check other_number_difference

end NUMINAMATH_CALUDE_other_number_difference_l3092_309272


namespace NUMINAMATH_CALUDE_distinct_roots_condition_root_condition_l3092_309244

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 + 2*x + k

-- Theorem for part 1
theorem distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic x k = 0 ∧ quadratic y k = 0) ↔ k < 1 :=
sorry

-- Theorem for part 2
theorem root_condition (m k : ℝ) :
  quadratic m k = 0 ∧ m^2 + 2*m = 2 → k = -2 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_condition_root_condition_l3092_309244


namespace NUMINAMATH_CALUDE_meeting_2015_same_as_first_l3092_309219

/-- Represents a point on a line segment --/
structure Point :=
  (position : ℝ)

/-- Represents a person moving on the line segment --/
structure Person :=
  (startPoint : Point)
  (speed : ℝ)
  (startTime : ℝ)

/-- Represents a meeting between two people --/
def Meeting := ℕ → Point

/-- The movement pattern of two people as described in the problem --/
def movementPattern (person1 person2 : Person) : Meeting :=
  sorry

/-- Theorem stating that the 2015th meeting point is the same as the first meeting point --/
theorem meeting_2015_same_as_first 
  (person1 person2 : Person) (pattern : Meeting := movementPattern person1 person2) :
  pattern 2015 = pattern 1 :=
sorry

end NUMINAMATH_CALUDE_meeting_2015_same_as_first_l3092_309219


namespace NUMINAMATH_CALUDE_triathlete_average_rate_l3092_309223

/-- The average rate of a triathlete's round trip -/
theorem triathlete_average_rate 
  (total_distance : ℝ) 
  (running_distance : ℝ) 
  (swimming_distance : ℝ) 
  (running_speed : ℝ) 
  (swimming_speed : ℝ) 
  (h1 : total_distance = 6) 
  (h2 : running_distance = total_distance / 2) 
  (h3 : swimming_distance = total_distance / 2) 
  (h4 : running_speed = 10) 
  (h5 : swimming_speed = 6) : 
  (total_distance / ((running_distance / running_speed + swimming_distance / swimming_speed) * 60)) = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_triathlete_average_rate_l3092_309223


namespace NUMINAMATH_CALUDE_g_13_equals_205_l3092_309262

def g (n : ℕ) : ℕ := n^2 + n + 23

theorem g_13_equals_205 : g 13 = 205 := by
  sorry

end NUMINAMATH_CALUDE_g_13_equals_205_l3092_309262


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3092_309211

theorem max_value_of_expression (x y : ℝ) (h : x^2 + y^2 ≠ 0) :
  (3*x^2 + 16*x*y + 15*y^2) / (x^2 + y^2) ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3092_309211


namespace NUMINAMATH_CALUDE_product_formula_l3092_309248

theorem product_formula (a b : ℕ) :
  (100 - a) * (100 + b) = ((b + (200 - a) - 100) * 100) - a * b := by
  sorry

end NUMINAMATH_CALUDE_product_formula_l3092_309248


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l3092_309254

theorem units_digit_of_seven_to_six_to_five (n : ℕ) : n = 7^(6^5) → n % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l3092_309254


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l3092_309273

theorem simplify_complex_fraction (a : ℝ) (h : a ≠ 1 ∧ a ≠ -1) :
  1 - (1 / (1 + a^2 / (1 - a^2))) = a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l3092_309273


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l3092_309237

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l3092_309237


namespace NUMINAMATH_CALUDE_coles_return_speed_l3092_309215

/-- Calculates the average speed of the return journey given the conditions of Cole's trip -/
theorem coles_return_speed (total_time : Real) (outbound_time : Real) (outbound_speed : Real) :
  total_time = 2 ∧ outbound_time = 72 / 60 ∧ outbound_speed = 70 →
  (2 * outbound_speed * outbound_time) / (total_time - outbound_time) = 105 := by
  sorry

#check coles_return_speed

end NUMINAMATH_CALUDE_coles_return_speed_l3092_309215


namespace NUMINAMATH_CALUDE_function_property_l3092_309290

theorem function_property (f : ℕ → ℝ) :
  f 1 = 3/2 ∧
  (∀ x y : ℕ, f (x + y) = (1 + y / (x + 1 : ℝ)) * f x + (1 + x / (y + 1 : ℝ)) * f y + x^2 * y + x * y + x * y^2) →
  ∀ x : ℕ, f x = (1/4 : ℝ) * x * (x + 1) * (2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l3092_309290


namespace NUMINAMATH_CALUDE_remainder_97_power_51_mod_100_l3092_309213

theorem remainder_97_power_51_mod_100 : 97^51 % 100 = 39 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_power_51_mod_100_l3092_309213


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_is_correct_l3092_309201

/-- The smallest positive integer divisible by all integers from 1 to 10 -/
def smallest_divisible_by_1_to_10 : ℕ := 27720

/-- Proposition: smallest_divisible_by_1_to_10 is the smallest positive integer 
    divisible by all integers from 1 to 10 -/
theorem smallest_divisible_by_1_to_10_is_correct :
  (∀ n : ℕ, n > 0 ∧ n < smallest_divisible_by_1_to_10 → 
    ∃ m : ℕ, m ∈ Finset.range 10 ∧ n % (m + 1) ≠ 0) ∧
  (∀ m : ℕ, m ∈ Finset.range 10 → smallest_divisible_by_1_to_10 % (m + 1) = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_is_correct_l3092_309201


namespace NUMINAMATH_CALUDE_second_tea_price_l3092_309206

/-- Represents the price of tea varieties and their mixture --/
structure TeaPrices where
  first : ℝ
  second : ℝ
  third : ℝ
  mixture : ℝ

/-- Theorem stating the price of the second tea variety --/
theorem second_tea_price (p : TeaPrices)
  (h1 : p.first = 126)
  (h2 : p.third = 177.5)
  (h3 : p.mixture = 154)
  (h4 : p.mixture * 4 = p.first + p.second + 2 * p.third) :
  p.second = 135 := by
  sorry

end NUMINAMATH_CALUDE_second_tea_price_l3092_309206


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l3092_309275

-- Define the rectangle
def rectangle_width : ℝ := 11
def rectangle_height : ℝ := 7

-- Define the circles
def circle_diameter : ℝ := rectangle_height

-- Theorem statement
theorem distance_between_circle_centers : 
  let circle_radius : ℝ := circle_diameter / 2
  let distance : ℝ := rectangle_width - 2 * circle_radius
  distance = 4 := by sorry

end NUMINAMATH_CALUDE_distance_between_circle_centers_l3092_309275


namespace NUMINAMATH_CALUDE_multiply_by_0_064_l3092_309294

theorem multiply_by_0_064 (x : ℝ) (h : 13.26 * x = 132.6) : 0.064 * x = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_0_064_l3092_309294


namespace NUMINAMATH_CALUDE_triangle_inequality_for_given_sides_l3092_309253

theorem triangle_inequality_for_given_sides (x : ℝ) (h : x > 1) :
  let a := x^4 + x^3 + 2*x^2 + x + 1
  let b := 2*x^3 + x^2 + 2*x + 1
  let c := x^4 - 1
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_for_given_sides_l3092_309253


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l3092_309270

theorem sum_of_powers_of_i (i : ℂ) (h : i^2 = -1) :
  (Finset.range 2017).sum (fun k => i^(k + 1)) = i := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l3092_309270


namespace NUMINAMATH_CALUDE_tens_digit_of_11_pow_2045_l3092_309291

theorem tens_digit_of_11_pow_2045 : ∃ k : ℕ, 11^2045 ≡ 50 + k [ZMOD 100] :=
by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_11_pow_2045_l3092_309291


namespace NUMINAMATH_CALUDE_gamma_max_success_ratio_l3092_309251

theorem gamma_max_success_ratio 
  (alpha_day1_score alpha_day1_total : ℕ)
  (alpha_day2_score alpha_day2_total : ℕ)
  (gamma_day1_score gamma_day1_total : ℕ)
  (gamma_day2_score gamma_day2_total : ℕ)
  (h1 : alpha_day1_score = 170)
  (h2 : alpha_day1_total = 280)
  (h3 : alpha_day2_score = 150)
  (h4 : alpha_day2_total = 220)
  (h5 : gamma_day1_total < alpha_day1_total)
  (h6 : gamma_day1_score > 0)
  (h7 : gamma_day2_score > 0)
  (h8 : (gamma_day1_score : ℚ) / gamma_day1_total < (alpha_day1_score : ℚ) / alpha_day1_total)
  (h9 : (gamma_day2_score : ℚ) / gamma_day2_total < (alpha_day2_score : ℚ) / alpha_day2_total)
  (h10 : gamma_day1_total + gamma_day2_total = 500)
  (h11 : (alpha_day1_score + alpha_day2_score : ℚ) / (alpha_day1_total + alpha_day2_total) = 320 / 500) :
  (gamma_day1_score + gamma_day2_score : ℚ) / 500 ≤ 170 / 500 :=
by sorry

end NUMINAMATH_CALUDE_gamma_max_success_ratio_l3092_309251


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3092_309269

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x^2 - 2*x ≤ 0 → -1 ≤ x ∧ x ≤ 3) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ x^2 - 2*x > 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3092_309269


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l3092_309241

theorem quadratic_distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 6 * x + 9 = 0 ∧ k * y^2 - 6 * y + 9 = 0) ↔ 
  (k < 1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l3092_309241


namespace NUMINAMATH_CALUDE_savings_ratio_l3092_309287

theorem savings_ratio (initial_savings : ℝ) (final_savings : ℝ) (months : ℕ) 
  (h1 : initial_savings = 10)
  (h2 : final_savings = 160)
  (h3 : months = 5) :
  ∃ (ratio : ℝ), ratio = 2 ∧ final_savings = initial_savings * ratio ^ (months - 1) :=
sorry

end NUMINAMATH_CALUDE_savings_ratio_l3092_309287


namespace NUMINAMATH_CALUDE_benny_turnips_l3092_309221

theorem benny_turnips (melanie_turnips benny_turnips total_turnips : ℕ) : 
  melanie_turnips = 139 → total_turnips = 252 → benny_turnips = total_turnips - melanie_turnips → 
  benny_turnips = 113 := by
  sorry

end NUMINAMATH_CALUDE_benny_turnips_l3092_309221


namespace NUMINAMATH_CALUDE_auto_credit_percentage_l3092_309259

/-- Given that automobile finance companies extended $57 billion of credit, which is 1/3 of the
    total automobile installment credit, and the total consumer installment credit outstanding
    is $855 billion, prove that automobile installment credit accounts for 20% of all outstanding
    consumer installment credit. -/
theorem auto_credit_percentage (finance_company_credit : ℝ) (total_consumer_credit : ℝ)
    (h1 : finance_company_credit = 57)
    (h2 : total_consumer_credit = 855)
    (h3 : finance_company_credit = (1/3) * (3 * finance_company_credit)) :
    (3 * finance_company_credit) / total_consumer_credit = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_auto_credit_percentage_l3092_309259


namespace NUMINAMATH_CALUDE_repeating_digit_divisible_by_101_l3092_309205

/-- A 9-digit integer where the first three digits are the same as the middle three and last three digits -/
def RepeatingDigitInteger (x y z : ℕ) : ℕ :=
  100100100 * x + 10010010 * y + 1001001 * z

/-- Theorem stating that 101 is a factor of any RepeatingDigitInteger -/
theorem repeating_digit_divisible_by_101 (x y z : ℕ) (h : 0 < x ∧ x < 10 ∧ y < 10 ∧ z < 10) :
  101 ∣ RepeatingDigitInteger x y z := by
  sorry

#check repeating_digit_divisible_by_101

end NUMINAMATH_CALUDE_repeating_digit_divisible_by_101_l3092_309205


namespace NUMINAMATH_CALUDE_melanie_balloons_l3092_309264

def joan_balloons : ℕ := 40
def total_balloons : ℕ := 81

theorem melanie_balloons : total_balloons - joan_balloons = 41 := by
  sorry

end NUMINAMATH_CALUDE_melanie_balloons_l3092_309264


namespace NUMINAMATH_CALUDE_box_volume_l3092_309286

-- Define the set of possible volumes
def possibleVolumes : Set ℕ := {180, 240, 300, 360, 450}

-- Theorem statement
theorem box_volume (a b c : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : ∃ (k : ℕ), k > 0 ∧ a = 2 * k ∧ b = 3 * k ∧ c = 5 * k) :
  (a * b * c) ∈ possibleVolumes ↔ a * b * c = 240 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l3092_309286


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3092_309268

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S seq n + seq.a (n + 1)

theorem arithmetic_sequence_property (seq : ArithmeticSequence) (m : ℕ) 
  (h1 : S seq (m - 1) = -2)
  (h2 : S seq m = 0)
  (h3 : S seq (m + 1) = 3) :
  seq.d = 1 ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3092_309268


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3092_309256

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ), 
    P = 2 / 15 ∧ Q = 1 / 3 ∧ R = 0 ∧
    ∀ (x : ℚ), x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 5*x + 6) / ((x - 1)*(x - 4)*(x - 6)) = 
      P / (x - 1) + Q / (x - 4) + R / (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3092_309256


namespace NUMINAMATH_CALUDE_count_auspicious_dragon_cards_l3092_309202

/-- The number of ways to select 4 digits from 0 to 9 and arrange them in ascending order -/
def auspicious_dragon_cards : ℕ := sorry

/-- Theorem stating that the number of Auspicious Dragon Cards is 210 -/
theorem count_auspicious_dragon_cards : auspicious_dragon_cards = 210 := by sorry

end NUMINAMATH_CALUDE_count_auspicious_dragon_cards_l3092_309202


namespace NUMINAMATH_CALUDE_percentage_problem_l3092_309297

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 24 + 0.1 * 40 = 5.92 ↔ P = 8 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3092_309297


namespace NUMINAMATH_CALUDE_triangle_centroid_product_l3092_309267

theorem triangle_centroid_product (AP PD BP PE CP PF : ℝ) 
  (h : AP / PD + BP / PE + CP / PF = 90) : 
  AP / PD * BP / PE * CP / PF = 94 := by
  sorry

end NUMINAMATH_CALUDE_triangle_centroid_product_l3092_309267


namespace NUMINAMATH_CALUDE_smallest_integer_divisible_l3092_309245

theorem smallest_integer_divisible (x : ℤ) : x = 36629 ↔ 
  (∀ y : ℤ, y < x → ¬(∃ k₁ k₂ k₃ k₄ : ℤ, 
    2 * y + 2 = 33 * k₁ ∧ 
    2 * y + 2 = 44 * k₂ ∧ 
    2 * y + 2 = 55 * k₃ ∧ 
    2 * y + 2 = 666 * k₄)) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℤ, 
    2 * x + 2 = 33 * k₁ ∧ 
    2 * x + 2 = 44 * k₂ ∧ 
    2 * x + 2 = 55 * k₃ ∧ 
    2 * x + 2 = 666 * k₄) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_divisible_l3092_309245


namespace NUMINAMATH_CALUDE_sequence_may_or_may_not_be_arithmetic_l3092_309228

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def is_arithmetic (s : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

/-- The first five terms of the sequence are 1, 2, 3, 4, 5. -/
def first_five_terms (s : Sequence) : Prop :=
  s 0 = 1 ∧ s 1 = 2 ∧ s 2 = 3 ∧ s 3 = 4 ∧ s 4 = 5

theorem sequence_may_or_may_not_be_arithmetic :
  ∃ s₁ s₂ : Sequence, first_five_terms s₁ ∧ first_five_terms s₂ ∧
    is_arithmetic s₁ ∧ ¬is_arithmetic s₂ := by
  sorry

end NUMINAMATH_CALUDE_sequence_may_or_may_not_be_arithmetic_l3092_309228


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l3092_309226

/-- Given vectors a and b, and their linear combinations u and v, 
    prove that if u is parallel to v, then x = 1/2 -/
theorem parallel_vectors_imply_x_value 
  (a b u v : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (x, 1))
  (h3 : u = a + 2 • b)
  (h4 : v = 2 • a - b)
  (h5 : ∃ (k : ℝ), u = k • v)
  : x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l3092_309226
