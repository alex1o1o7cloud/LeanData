import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3041_304196

theorem sqrt_equation_solution : ∃! z : ℚ, Real.sqrt (10 + 3 * z) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3041_304196


namespace NUMINAMATH_CALUDE_absolute_value_sum_lower_bound_l3041_304103

theorem absolute_value_sum_lower_bound :
  ∀ x : ℝ, |x - 4| + |x + 3| ≥ 7 ∧ ∃ y : ℝ, |y - 4| + |y + 3| = 7 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_lower_bound_l3041_304103


namespace NUMINAMATH_CALUDE_certain_value_problem_l3041_304147

theorem certain_value_problem (n : ℤ) (v : ℤ) (h1 : n = -7) (h2 : 3 * n = 2 * n - v) : v = 7 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_problem_l3041_304147


namespace NUMINAMATH_CALUDE_problem_solution_l3041_304116

-- Define proposition p
def p : Prop := ∀ a b : ℝ, (a > b ∧ b > 0) → (1/a < 1/b)

-- Define proposition q
def q : Prop := ∀ f : ℝ → ℝ, (∀ x : ℝ, f (x - 1) = f (-(x - 1))) → 
  (∀ x : ℝ, f x = f (2 - x))

-- Theorem to prove
theorem problem_solution : p ∨ ¬q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3041_304116


namespace NUMINAMATH_CALUDE_square_sum_of_linear_equations_l3041_304132

theorem square_sum_of_linear_equations (x y : ℝ) 
  (eq1 : 3 * x + 4 * y = 30) 
  (eq2 : x + 2 * y = 13) : 
  x^2 + y^2 = 145/4 := by sorry

end NUMINAMATH_CALUDE_square_sum_of_linear_equations_l3041_304132


namespace NUMINAMATH_CALUDE_polygon_sides_l3041_304138

theorem polygon_sides (n : ℕ) (h : n > 2) : 
  (180 * (n - 2) : ℝ) = 3 * 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3041_304138


namespace NUMINAMATH_CALUDE_degree_to_radian_300_l3041_304150

theorem degree_to_radian_300 : 
  (300 : ℝ) * (π / 180) = (5 * π) / 3 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_300_l3041_304150


namespace NUMINAMATH_CALUDE_rectangle_sides_l3041_304191

theorem rectangle_sides (S d : ℝ) (h1 : S > 0) (h2 : d ≥ 0) :
  let a := Real.sqrt (S + d^2 / 4) + d / 2
  let b := Real.sqrt (S + d^2 / 4) - d / 2
  a * b = S ∧ a - b = d ∧ a > 0 ∧ b > 0 := by sorry

end NUMINAMATH_CALUDE_rectangle_sides_l3041_304191


namespace NUMINAMATH_CALUDE_exists_n_in_sequence_l3041_304197

theorem exists_n_in_sequence (a : ℕ → ℕ) : (∀ n, a n = n^2 + n) → ∃ n, a n = 30 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_in_sequence_l3041_304197


namespace NUMINAMATH_CALUDE_volume_of_specific_prism_l3041_304195

/-- A regular triangular prism inscribed in a sphere -/
structure InscribedPrism where
  /-- Radius of the sphere -/
  R : ℝ
  /-- Length of AD, where D is on the diameter CD -/
  AD : ℝ

/-- The volume of the inscribed prism -/
def prism_volume (p : InscribedPrism) : ℝ := sorry

/-- Theorem: The volume of the specific inscribed prism is 48√15 -/
theorem volume_of_specific_prism :
  let p : InscribedPrism := { R := 6, AD := 4 * Real.sqrt 6 }
  prism_volume p = 48 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_prism_l3041_304195


namespace NUMINAMATH_CALUDE_trapezoid_bases_solutions_l3041_304178

theorem trapezoid_bases_solutions :
  let area : ℕ := 1800
  let altitude : ℕ := 60
  let base_sum : ℕ := 2 * area / altitude
  let valid_base_pair := λ b₁ b₂ : ℕ =>
    b₁ % 10 = 0 ∧ b₂ % 10 = 0 ∧ b₁ + b₂ = base_sum
  (∃! (solutions : Finset (ℕ × ℕ)), solutions.card = 4 ∧
    ∀ pair : ℕ × ℕ, pair ∈ solutions ↔ valid_base_pair pair.1 pair.2) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_bases_solutions_l3041_304178


namespace NUMINAMATH_CALUDE_remaining_length_is_21_l3041_304110

/-- A polygon with perpendicular adjacent sides -/
structure PerpendicularPolygon where
  left : ℝ
  top : ℝ
  right : ℝ
  bottom_removed : List ℝ

/-- The total length of remaining segments after removal -/
def remaining_length (p : PerpendicularPolygon) : ℝ :=
  p.left + p.top + p.right

theorem remaining_length_is_21 (p : PerpendicularPolygon)
  (h1 : p.left = 10)
  (h2 : p.top = 3)
  (h3 : p.right = 8)
  (h4 : p.bottom_removed = [2, 1, 2]) :
  remaining_length p = 21 := by
  sorry

end NUMINAMATH_CALUDE_remaining_length_is_21_l3041_304110


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l3041_304119

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem sum_of_powers_of_i :
  i^1520 + i^1521 + i^1522 + i^1523 + i^1524 = (2 : ℂ) := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l3041_304119


namespace NUMINAMATH_CALUDE_chord_line_equation_l3041_304159

/-- Given an ellipse and a chord midpoint, prove the equation of the line containing the chord -/
theorem chord_line_equation (x y : ℝ) :
  (x^2 / 4 + y^2 / 3 = 1) →  -- Ellipse equation
  (∃ x1 y1 x2 y2 : ℝ,        -- Endpoints of the chord
    x1^2 / 4 + y1^2 / 3 = 1 ∧
    x2^2 / 4 + y2^2 / 3 = 1 ∧
    (x1 + x2) / 2 = -1 ∧     -- Midpoint x-coordinate
    (y1 + y2) / 2 = 1) →     -- Midpoint y-coordinate
  (∃ a b c : ℝ,              -- Line equation coefficients
    a * x + b * y + c = 0 ∧  -- General form of line equation
    a = 3 ∧ b = -4 ∧ c = 7)  -- Specific coefficients for the answer
  := by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l3041_304159


namespace NUMINAMATH_CALUDE_refrigeratorSample_is_valid_l3041_304126

/-- Represents a systematic sample -/
structure SystematicSample (N : ℕ) (n : ℕ) where
  start : ℕ
  sequence : Fin n → ℕ
  valid : ∀ i : Fin n, sequence i = start + i.val * (N / n)

/-- The specific systematic sample for the refrigerator problem -/
def refrigeratorSample : SystematicSample 60 6 :=
  { start := 3,
    sequence := λ i => 3 + i.val * 10,
    valid := sorry }

/-- Theorem stating that the refrigeratorSample is valid -/
theorem refrigeratorSample_is_valid :
  ∀ i : Fin 6, refrigeratorSample.sequence i ≤ 60 :=
by sorry

end NUMINAMATH_CALUDE_refrigeratorSample_is_valid_l3041_304126


namespace NUMINAMATH_CALUDE_k_value_l3041_304172

def A (k : ℕ) : Set ℕ := {1, 2, k}
def B : Set ℕ := {2, 5}

theorem k_value : ∀ k : ℕ, A k ∪ B = {1, 2, 3, 5} → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l3041_304172


namespace NUMINAMATH_CALUDE_final_short_tree_count_l3041_304120

/-- Represents the number of trees of a specific type -/
structure TreeCount where
  short_oak : ℕ
  short_pine : ℕ
  short_maple : ℕ

/-- Calculates the total number of short trees -/
def total_short_trees (tc : TreeCount) : ℕ :=
  tc.short_oak + tc.short_pine + tc.short_maple

/-- The initial count of trees in the park -/
def initial_trees : TreeCount :=
  { short_oak := 3
  , short_pine := 4
  , short_maple := 5 }

/-- The count of trees to be planted -/
def trees_to_plant : TreeCount :=
  { short_oak := 9
  , short_pine := 6
  , short_maple := 4 }

/-- The final count of trees after planting -/
def final_trees : TreeCount :=
  { short_oak := initial_trees.short_oak + trees_to_plant.short_oak
  , short_pine := initial_trees.short_pine + trees_to_plant.short_pine
  , short_maple := initial_trees.short_maple + trees_to_plant.short_maple }

theorem final_short_tree_count :
  total_short_trees final_trees = 31 := by
  sorry

end NUMINAMATH_CALUDE_final_short_tree_count_l3041_304120


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3041_304187

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3041_304187


namespace NUMINAMATH_CALUDE_arithmetic_sequence_x_value_l3041_304121

/-- Given an arithmetic sequence with first three terms 2x-3, 3x, and 5x+1, prove that x = 2 -/
theorem arithmetic_sequence_x_value :
  ∀ x : ℝ,
  let a₁ := 2*x - 3
  let a₂ := 3*x
  let a₃ := 5*x + 1
  (a₂ - a₁ = a₃ - a₂) → x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_x_value_l3041_304121


namespace NUMINAMATH_CALUDE_hidden_dots_count_l3041_304162

/-- Represents a standard six-sided die -/
def StandardDie := Fin 6

/-- The sum of dots on all faces of a standard die -/
def sumOfDots : ℕ := (List.range 6).sum + 6

/-- The list of visible face values -/
def visibleFaces : List ℕ := [1, 2, 3, 4, 5, 4, 6, 5, 3]

/-- The number of dice in the stack -/
def numberOfDice : ℕ := 4

/-- The number of visible faces -/
def numberOfVisibleFaces : ℕ := 9

theorem hidden_dots_count :
  (numberOfDice * sumOfDots) - visibleFaces.sum = 51 := by
  sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l3041_304162


namespace NUMINAMATH_CALUDE_rice_price_decrease_l3041_304102

theorem rice_price_decrease (original_price : ℝ) (h : original_price > 0) :
  let new_price := (20 / 25) * original_price
  let percentage_decrease := (original_price - new_price) / original_price * 100
  percentage_decrease = 20 := by
sorry

end NUMINAMATH_CALUDE_rice_price_decrease_l3041_304102


namespace NUMINAMATH_CALUDE_smaller_angle_at_4_oclock_l3041_304127

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees between each hour on a clock face -/
def degrees_per_hour : ℕ := full_circle_degrees / clock_hours

/-- The number of hour spaces between 12 and 4 on a clock face -/
def spaces_12_to_4 : ℕ := 4

/-- The smaller angle formed by the hands of a clock at 4 o'clock -/
def clock_angle_at_4 : ℕ := spaces_12_to_4 * degrees_per_hour

theorem smaller_angle_at_4_oclock :
  clock_angle_at_4 = 120 :=
sorry

end NUMINAMATH_CALUDE_smaller_angle_at_4_oclock_l3041_304127


namespace NUMINAMATH_CALUDE_x_plus_y_equals_four_l3041_304180

-- Define the conditions
def conditions (x y : ℝ) : Prop :=
  x ≥ -2 ∧ 
  y ≥ -3 ∧ 
  x - 2 * Real.sqrt (x + 2) = 2 * Real.sqrt (y + 3) - y

-- Theorem statement
theorem x_plus_y_equals_four (x y : ℝ) (h : conditions x y) : x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_four_l3041_304180


namespace NUMINAMATH_CALUDE_largest_number_l3041_304114

-- Define a function to convert a number from base b to decimal
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

-- Define the numbers in their respective bases
def num_A : Nat := to_decimal [2, 1, 1] 6
def num_B : Nat := 41
def num_C : Nat := to_decimal [6, 4] 9
def num_D : Nat := to_decimal [11, 2] 16

-- State the theorem
theorem largest_number :
  num_A > num_B ∧ num_A > num_C ∧ num_A > num_D :=
sorry

end NUMINAMATH_CALUDE_largest_number_l3041_304114


namespace NUMINAMATH_CALUDE_total_turtles_l3041_304140

theorem total_turtles (kristen_turtles : ℕ) (kris_turtles : ℕ) (trey_turtles : ℕ) :
  kristen_turtles = 12 →
  kris_turtles = kristen_turtles / 4 →
  trey_turtles = 5 * kris_turtles →
  kristen_turtles + kris_turtles + trey_turtles = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_turtles_l3041_304140


namespace NUMINAMATH_CALUDE_lowest_temp_is_harbin_l3041_304156

def harbin_temp : ℤ := -20
def beijing_temp : ℤ := -10
def hangzhou_temp : ℤ := 0
def jinhua_temp : ℤ := 2

def city_temps : List ℤ := [harbin_temp, beijing_temp, hangzhou_temp, jinhua_temp]

theorem lowest_temp_is_harbin :
  List.minimum city_temps = some harbin_temp := by
  sorry

end NUMINAMATH_CALUDE_lowest_temp_is_harbin_l3041_304156


namespace NUMINAMATH_CALUDE_black_cube_difference_l3041_304145

/-- Represents a 3x3x3 cube built with unit cubes -/
structure Cube :=
  (size : Nat)
  (total_cubes : Nat)
  (surface_area : Nat)

/-- Represents the distribution of colors on the cube's surface -/
structure SurfaceColor :=
  (black : Nat)
  (grey : Nat)
  (white : Nat)

/-- Defines a valid 3x3x3 cube with equal surface color distribution -/
def valid_cube (c : Cube) (sc : SurfaceColor) : Prop :=
  c.size = 3 ∧
  c.total_cubes = 27 ∧
  c.surface_area = 54 ∧
  sc.black = sc.grey ∧
  sc.grey = sc.white ∧
  sc.black + sc.grey + sc.white = c.surface_area

/-- The minimum number of black cubes that can be used -/
def min_black_cubes (c : Cube) (sc : SurfaceColor) : Nat :=
  sorry

/-- The maximum number of black cubes that can be used -/
def max_black_cubes (c : Cube) (sc : SurfaceColor) : Nat :=
  sorry

/-- Theorem stating the difference between max and min black cubes -/
theorem black_cube_difference (c : Cube) (sc : SurfaceColor) :
  valid_cube c sc → max_black_cubes c sc - min_black_cubes c sc = 7 :=
  sorry

end NUMINAMATH_CALUDE_black_cube_difference_l3041_304145


namespace NUMINAMATH_CALUDE_stating_simultaneous_ring_theorem_l3041_304106

/-- The time interval (in minutes) between bell rings for the post office -/
def post_office_interval : ℕ := 18

/-- The time interval (in minutes) between bell rings for the train station -/
def train_station_interval : ℕ := 24

/-- The time interval (in minutes) between bell rings for the town hall -/
def town_hall_interval : ℕ := 30

/-- The time (in minutes) after which all bells ring simultaneously again -/
def simultaneous_ring_time : ℕ := 360

/-- 
Theorem stating that the time after which all bells ring simultaneously
is the least common multiple of their individual intervals
-/
theorem simultaneous_ring_theorem :
  simultaneous_ring_time = Nat.lcm post_office_interval (Nat.lcm train_station_interval town_hall_interval) :=
by sorry

end NUMINAMATH_CALUDE_stating_simultaneous_ring_theorem_l3041_304106


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l3041_304188

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 5) 
  (h1 : f 1 = 1) 
  (h2 : f 2 = 2) : 
  f 23 + f (-14) = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l3041_304188


namespace NUMINAMATH_CALUDE_f_composition_half_l3041_304100

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_half : f (f (1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_half_l3041_304100


namespace NUMINAMATH_CALUDE_geric_bills_count_geric_bills_proof_l3041_304144

theorem geric_bills_count : ℕ → ℕ → ℕ → Prop :=
  fun geric_bills kyla_bills jessa_bills =>
    (geric_bills = 2 * kyla_bills) ∧
    (kyla_bills = jessa_bills - 2) ∧
    (jessa_bills - 3 = 7) →
    geric_bills = 16

-- The proof goes here
theorem geric_bills_proof : ∃ g k j, geric_bills_count g k j :=
  sorry

end NUMINAMATH_CALUDE_geric_bills_count_geric_bills_proof_l3041_304144


namespace NUMINAMATH_CALUDE_prob_ride_all_cars_l3041_304151

/-- The number of cars in the roller coaster -/
def num_cars : ℕ := 4

/-- The number of times the passenger rides the roller coaster -/
def num_rides : ℕ := 4

/-- The probability of choosing any specific car for a single ride -/
def prob_single_car : ℚ := 1 / num_cars

/-- The probability of riding in each of the 4 cars exactly once in 4 rides -/
def prob_all_cars : ℚ := 3 / 32

/-- Theorem stating that the probability of riding in each car exactly once is 3/32 -/
theorem prob_ride_all_cars : 
  prob_all_cars = (num_cars.factorial : ℚ) / num_cars ^ num_rides :=
sorry

end NUMINAMATH_CALUDE_prob_ride_all_cars_l3041_304151


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l3041_304105

-- Define the given circle
def givenCircle (x y : ℝ) : Prop := (x - 6)^2 + y^2 = 64

-- Define the point P
def P : ℝ × ℝ := (2, 0)

-- Define the point Q (center of the given circle)
def Q : ℝ × ℝ := (6, 0)

-- Define a circle passing through P and tangent to the given circle
def passingCircle (a b r : ℝ) : Prop :=
  (a - P.1)^2 + (b - P.2)^2 = r^2 ∧
  ∃ (x y : ℝ), givenCircle x y ∧ (a - x)^2 + (b - y)^2 = r^2 ∧
  (a - Q.1)^2 + (b - Q.2)^2 = (8 - r)^2

-- Define the locus of centers
def locus (a b : ℝ) : Prop :=
  ∃ (r : ℝ), passingCircle a b r

-- Theorem statement
theorem locus_is_ellipse :
  ∀ (a b : ℝ), locus a b ↔ 
    (a - P.1)^2 + (b - P.2)^2 + (a - Q.1)^2 + (b - Q.2)^2 = 8^2 :=
sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l3041_304105


namespace NUMINAMATH_CALUDE_perpendicular_condition_l3041_304192

-- Define the lines l₁ and l₂
def l₁ (x y a : ℝ) : Prop := x + a * y - 2 = 0
def l₂ (x y a : ℝ) : Prop := x - a * y - 1 = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop := 1 + a * (-a) = 0

-- Define sufficient condition
def sufficient (P Q : Prop) : Prop := P → Q

-- Define necessary condition
def necessary (P Q : Prop) : Prop := Q → P

theorem perpendicular_condition (a : ℝ) :
  sufficient (a = -1) (perpendicular a) ∧
  ¬ necessary (a = -1) (perpendicular a) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l3041_304192


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3041_304198

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (5 * x + 12) = 13 - x →
  x = (31 + Real.sqrt 333) / 2 ∨ x = (31 - Real.sqrt 333) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3041_304198


namespace NUMINAMATH_CALUDE_james_delivery_l3041_304175

/-- Calculates the number of bags delivered by James in a given number of days -/
def bags_delivered (bags_per_trip : ℕ) (trips_per_day : ℕ) (days : ℕ) : ℕ :=
  bags_per_trip * trips_per_day * days

/-- Theorem stating that James delivers 1000 bags in 5 days -/
theorem james_delivery : bags_delivered 10 20 5 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_james_delivery_l3041_304175


namespace NUMINAMATH_CALUDE_five_digit_base10_to_base2_sum_l3041_304163

theorem five_digit_base10_to_base2_sum : ∃ (min max : ℕ),
  (∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 →
    min ≤ (Nat.log 2 n + 1) ∧ (Nat.log 2 n + 1) ≤ max) ∧
  (max - min + 1) * (min + max) / 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_base10_to_base2_sum_l3041_304163


namespace NUMINAMATH_CALUDE_intersection_point_when_a_is_one_parallel_when_a_is_three_halves_l3041_304161

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := x + a * y - a + 2 = 0
def l₂ (a x y : ℝ) : Prop := 2 * a * x + (a + 3) * y + a - 5 = 0

-- Theorem for the intersection point when a = 1
theorem intersection_point_when_a_is_one :
  ∃ (x y : ℝ), l₁ 1 x y ∧ l₂ 1 x y ∧ x = -4 ∧ y = 3 :=
sorry

-- Definition of parallel lines
def parallel (a : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧
  (1 : ℝ) / (a : ℝ) = k * (2 * a) / (a + 3) ∧
  (a ≠ -3)

-- Theorem for parallel lines when a = 3/2
theorem parallel_when_a_is_three_halves :
  parallel (3/2) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_when_a_is_one_parallel_when_a_is_three_halves_l3041_304161


namespace NUMINAMATH_CALUDE_equilateral_triangle_count_is_twenty_l3041_304167

/-- Represents a point in the hexagonal lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The hexagonal lattice with 19 points -/
def HexagonalLattice : Set LatticePoint :=
  sorry

/-- Predicate to check if three points form an equilateral triangle -/
def IsEquilateralTriangle (p1 p2 p3 : LatticePoint) : Prop :=
  sorry

/-- The number of equilateral triangles in the lattice -/
def EquilateralTriangleCount : ℕ :=
  sorry

/-- Theorem stating that there are exactly 20 equilateral triangles in the lattice -/
theorem equilateral_triangle_count_is_twenty :
  EquilateralTriangleCount = 20 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_count_is_twenty_l3041_304167


namespace NUMINAMATH_CALUDE_solution_set_part1_value_of_a_part2_l3041_304182

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | |x - 2| ≥ 7 - |x - 1|} = {x : ℝ | x ≤ -2 ∨ x ≥ 5} :=
sorry

-- Part 2
theorem value_of_a_part2 (a : ℝ) :
  {x : ℝ | |x - a| ≤ 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} → a = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_value_of_a_part2_l3041_304182


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3041_304168

theorem sum_of_cubes_of_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^3 + x₂^3 = 95/8 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3041_304168


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_product_l3041_304184

theorem repeating_decimal_fraction_product : ∃ (n d : ℕ), 
  (n ≠ 0 ∧ d ≠ 0) ∧ 
  (∀ (k : ℕ), (0.027 + 0.027 / (1000 ^ k - 1) : ℚ) = n / d) ∧
  (∀ (n' d' : ℕ), n' ≠ 0 ∧ d' ≠ 0 → (∀ (k : ℕ), (0.027 + 0.027 / (1000 ^ k - 1) : ℚ) = n' / d') → n ≤ n' ∧ d ≤ d') ∧
  n * d = 37 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_product_l3041_304184


namespace NUMINAMATH_CALUDE_johns_snack_spending_l3041_304152

theorem johns_snack_spending (initial_amount : ℝ) (remaining_amount : ℝ) 
  (snack_fraction : ℝ) (necessity_fraction : ℝ) :
  initial_amount = 20 →
  remaining_amount = 4 →
  necessity_fraction = 3/4 →
  remaining_amount = initial_amount * (1 - snack_fraction) * (1 - necessity_fraction) →
  snack_fraction = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_johns_snack_spending_l3041_304152


namespace NUMINAMATH_CALUDE_inequality_proof_l3041_304136

theorem inequality_proof (a : ℝ) (ha : a > 0) : 
  Real.sqrt (a^2 + 1/a^2) + 2 ≥ a + 1/a + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3041_304136


namespace NUMINAMATH_CALUDE_sqrt_not_defined_for_negative_one_l3041_304153

theorem sqrt_not_defined_for_negative_one :
  ¬ (∃ (y : ℝ), y^2 = -1) :=
sorry

end NUMINAMATH_CALUDE_sqrt_not_defined_for_negative_one_l3041_304153


namespace NUMINAMATH_CALUDE_beths_shopping_multiple_l3041_304148

/-- The problem of Beth's shopping for peas and corn -/
theorem beths_shopping_multiple (peas corn : ℕ) (multiple : ℚ) 
  (h1 : peas = corn * multiple + 15)
  (h2 : peas = 35)
  (h3 : corn = 10) :
  multiple = 2 := by
  sorry

end NUMINAMATH_CALUDE_beths_shopping_multiple_l3041_304148


namespace NUMINAMATH_CALUDE_two_week_training_hours_l3041_304128

/-- Calculates the total training hours for two weeks given daily maximum hours -/
def totalTrainingHours (week1MaxHours : ℕ) (week2MaxHours : ℕ) : ℕ :=
  7 * week1MaxHours + 7 * week2MaxHours

/-- Proves that training for 2 hours max per day in week 1 and 3 hours max per day in week 2 results in 35 total hours -/
theorem two_week_training_hours : totalTrainingHours 2 3 = 35 := by
  sorry

#eval totalTrainingHours 2 3

end NUMINAMATH_CALUDE_two_week_training_hours_l3041_304128


namespace NUMINAMATH_CALUDE_high_school_total_students_l3041_304112

/-- Represents a high school with three grades -/
structure HighSchool where
  freshman : ℕ
  sophomore : ℕ
  senior : ℕ

/-- Represents a stratified sample from the high school -/
structure StratifiedSample where
  freshman : ℕ
  sophomore : ℕ
  senior : ℕ
  total : ℕ

theorem high_school_total_students (hs : HighSchool) (sample : StratifiedSample) : 
  hs.senior = 1000 →
  sample.freshman = 75 →
  sample.sophomore = 60 →
  sample.total = 185 →
  hs.freshman + hs.sophomore + hs.senior = 3700 := by
  sorry

#check high_school_total_students

end NUMINAMATH_CALUDE_high_school_total_students_l3041_304112


namespace NUMINAMATH_CALUDE_max_value_of_even_quadratic_function_l3041_304134

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem max_value_of_even_quadratic_function (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 1) (2 * a), f a b x = f a b (-x)) →
  (∃ x ∈ Set.Icc (a - 1) (2 * a), ∀ y ∈ Set.Icc (a - 1) (2 * a), f a b y ≤ f a b x) →
  (∃ x ∈ Set.Icc (a - 1) (2 * a), f a b x = 31 / 27) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_even_quadratic_function_l3041_304134


namespace NUMINAMATH_CALUDE_sample_size_theorem_l3041_304111

theorem sample_size_theorem (frequency_sum : ℝ) (frequency_ratio : ℝ) 
  (h1 : frequency_sum = 20) 
  (h2 : frequency_ratio = 0.4) : 
  frequency_sum / frequency_ratio = 50 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_theorem_l3041_304111


namespace NUMINAMATH_CALUDE_unique_n_for_divisibility_by_15_l3041_304179

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem unique_n_for_divisibility_by_15 : 
  ∃! n : ℕ, n < 10 ∧ is_divisible_by (80000 + 10000 * n + 945) 15 :=
sorry

end NUMINAMATH_CALUDE_unique_n_for_divisibility_by_15_l3041_304179


namespace NUMINAMATH_CALUDE_fruit_draw_ways_l3041_304189

/-- The number of fruits in the basket -/
def num_fruits : ℕ := 5

/-- The number of draws -/
def num_draws : ℕ := 2

/-- The number of ways to draw a fruit twice from a basket of 5 distinct fruits, considering the order -/
def num_ways : ℕ := num_fruits * (num_fruits - 1)

theorem fruit_draw_ways :
  num_ways = 20 :=
by sorry

end NUMINAMATH_CALUDE_fruit_draw_ways_l3041_304189


namespace NUMINAMATH_CALUDE_even_five_digit_numbers_l3041_304199

def set1 : Finset ℕ := {1, 3, 5}
def set2 : Finset ℕ := {2, 4, 6, 8}

def is_valid_selection (s : Finset ℕ) : Prop :=
  s.card = 5 ∧ (s ∩ set1).card = 2 ∧ (s ∩ set2).card = 3

def is_even (n : ℕ) : Prop := n % 2 = 0

def count_even_numbers : ℕ := sorry

theorem even_five_digit_numbers :
  count_even_numbers = 864 :=
sorry

end NUMINAMATH_CALUDE_even_five_digit_numbers_l3041_304199


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l3041_304104

-- Equation 1
theorem solve_equation_one : 
  ∀ x : ℝ, x^2 - 10*x + 16 = 0 ↔ x = 8 ∨ x = 2 := by sorry

-- Equation 2
theorem solve_equation_two :
  ∀ x : ℝ, x*(x-3) = 6-2*x ↔ x = 3 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l3041_304104


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l3041_304130

/-- Theorem: For a right circular cone and a sphere with the same radius,
    if the volume of the cone is one-third that of the sphere,
    then the ratio of the altitude of the cone to its base radius is 4/3. -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) :
  (1 / 3) * ((4 / 3) * π * r^3) = (1 / 3) * π * r^2 * h →
  h / r = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l3041_304130


namespace NUMINAMATH_CALUDE_people_in_room_l3041_304186

/-- Given a room with chairs and people, prove the total number of people -/
theorem people_in_room (chairs : ℕ) (people : ℕ) : 
  (3 : ℚ) / 5 * people = (4 : ℚ) / 5 * chairs ∧ 
  chairs - (4 : ℚ) / 5 * chairs = 8 →
  people = 54 := by sorry

end NUMINAMATH_CALUDE_people_in_room_l3041_304186


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l3041_304131

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 2 + a 8 = 16)
  (h_a4 : a 4 = 1) : 
  a 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l3041_304131


namespace NUMINAMATH_CALUDE_a_lt_neg_four_sufficient_not_necessary_l3041_304169

/-- The function f(x) = ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

/-- The condition for f to have a zero point on [-1,1] -/
def has_zero_point (a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a x = 0

/-- The statement that a < -4 is sufficient but not necessary for f to have a zero point on [-1,1] -/
theorem a_lt_neg_four_sufficient_not_necessary :
  (∀ a : ℝ, a < -4 → has_zero_point a) ∧
  ¬(∀ a : ℝ, has_zero_point a → a < -4) :=
sorry

end NUMINAMATH_CALUDE_a_lt_neg_four_sufficient_not_necessary_l3041_304169


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3041_304124

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3041_304124


namespace NUMINAMATH_CALUDE_factorization_equality_l3041_304158

theorem factorization_equality (m a : ℝ) : 3 * m * a^2 - 6 * m * a + 3 * m = 3 * m * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3041_304158


namespace NUMINAMATH_CALUDE_coin_arrangement_count_l3041_304146

/-- Represents a coin with its type and orientation -/
inductive Coin
| Gold : Bool → Coin
| Silver : Bool → Coin

/-- Checks if two adjacent coins are not face to face -/
def notFaceToFace (c1 c2 : Coin) : Prop := sorry

/-- Checks if three consecutive coins do not have the same orientation -/
def notSameOrientation (c1 c2 c3 : Coin) : Prop := sorry

/-- Represents a valid arrangement of coins -/
def ValidArrangement (arrangement : List Coin) : Prop :=
  arrangement.length = 10 ∧
  (arrangement.filter (λ c => match c with | Coin.Gold _ => true | _ => false)).length = 5 ∧
  (arrangement.filter (λ c => match c with | Coin.Silver _ => true | _ => false)).length = 5 ∧
  (∀ i, i < 9 → notFaceToFace (arrangement.get ⟨i, sorry⟩) (arrangement.get ⟨i+1, sorry⟩)) ∧
  (∀ i, i < 8 → notSameOrientation (arrangement.get ⟨i, sorry⟩) (arrangement.get ⟨i+1, sorry⟩) (arrangement.get ⟨i+2, sorry⟩))

/-- The number of valid arrangements -/
def numValidArrangements : ℕ := sorry

theorem coin_arrangement_count :
  numValidArrangements = 8568 := by sorry

end NUMINAMATH_CALUDE_coin_arrangement_count_l3041_304146


namespace NUMINAMATH_CALUDE_line_param_solution_l3041_304174

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = -x + 3

/-- The parameterization of the line -/
def parameterization (u v m : ℝ) (x y : ℝ) : Prop :=
  x = 2 + u * m ∧ y = v + u * 8

/-- Theorem stating that v = 1 and m = -8 satisfy the line equation and parameterization -/
theorem line_param_solution :
  ∃ (v m : ℝ), v = 1 ∧ m = -8 ∧
  (∀ (x y u : ℝ), parameterization u v m x y → line_equation x y) :=
sorry

end NUMINAMATH_CALUDE_line_param_solution_l3041_304174


namespace NUMINAMATH_CALUDE_hannah_stocking_stuffers_l3041_304101

/-- The number of candy canes per stocking -/
def candy_canes_per_stocking : ℕ := 4

/-- The number of beanie babies per stocking -/
def beanie_babies_per_stocking : ℕ := 2

/-- The number of books per stocking -/
def books_per_stocking : ℕ := 1

/-- The number of kids Hannah has -/
def number_of_kids : ℕ := 3

/-- The total number of stocking stuffers Hannah buys -/
def total_stocking_stuffers : ℕ :=
  (candy_canes_per_stocking + beanie_babies_per_stocking + books_per_stocking) * number_of_kids

theorem hannah_stocking_stuffers :
  total_stocking_stuffers = 21 := by
  sorry

end NUMINAMATH_CALUDE_hannah_stocking_stuffers_l3041_304101


namespace NUMINAMATH_CALUDE_brothers_difference_l3041_304193

theorem brothers_difference (aaron_brothers : ℕ) (bennett_brothers : ℕ) : 
  aaron_brothers = 4 → bennett_brothers = 6 → 2 * aaron_brothers - bennett_brothers = 2 := by
  sorry

end NUMINAMATH_CALUDE_brothers_difference_l3041_304193


namespace NUMINAMATH_CALUDE_min_b_minus_a_l3041_304108

open Real

noncomputable def f (x : ℝ) : ℝ := log x - 1 / x

noncomputable def g (a b x : ℝ) : ℝ := -a * x + b

def is_tangent_line (f g : ℝ → ℝ) : Prop :=
  ∃ x₀, (∀ x, g x = f x₀ + (deriv f x₀) * (x - x₀))

theorem min_b_minus_a (a b : ℝ) :
  (∀ x, x > 0 → f x = f x) →
  is_tangent_line f (g a b) →
  b - a ≥ -1 ∧ ∃ a₀ b₀, b₀ - a₀ = -1 :=
sorry

end NUMINAMATH_CALUDE_min_b_minus_a_l3041_304108


namespace NUMINAMATH_CALUDE_lemonade_glasses_served_l3041_304164

/-- The number of glasses of lemonade that can be served from a given number of pitchers. -/
def glasses_served (glasses_per_pitcher : ℕ) (num_pitchers : ℕ) : ℕ :=
  glasses_per_pitcher * num_pitchers

/-- Theorem stating that 6 pitchers of lemonade, each serving 5 glasses, can serve 30 glasses in total. -/
theorem lemonade_glasses_served :
  glasses_served 5 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_glasses_served_l3041_304164


namespace NUMINAMATH_CALUDE_apple_delivery_proof_l3041_304117

/-- Represents the number of apples delivered by the truck -/
def apples_delivered : ℕ → ℕ → ℕ → ℕ
  | initial_green, initial_red, final_green_excess =>
    final_green_excess + initial_red - initial_green

theorem apple_delivery_proof :
  let initial_green := 32
  let initial_red := initial_green + 200
  let final_green_excess := 140
  apples_delivered initial_green initial_red final_green_excess = 340 := by
sorry

#eval apples_delivered 32 232 140

end NUMINAMATH_CALUDE_apple_delivery_proof_l3041_304117


namespace NUMINAMATH_CALUDE_Q_four_roots_implies_d_value_l3041_304157

/-- The polynomial Q(x) -/
def Q (d x : ℂ) : ℂ := (x^2 - 3*x + 3) * (x^2 - d*x + 5) * (x^2 - 5*x + 15)

/-- The theorem stating that if Q(x) has exactly 4 distinct roots, then |d| = 13/2 -/
theorem Q_four_roots_implies_d_value (d : ℂ) :
  (∃ (s : Finset ℂ), s.card = 4 ∧ (∀ x ∈ s, Q d x = 0) ∧ (∀ x, Q d x = 0 → x ∈ s)) →
  Complex.abs d = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_Q_four_roots_implies_d_value_l3041_304157


namespace NUMINAMATH_CALUDE_winnie_balloons_l3041_304171

/-- The number of balloons Winnie keeps for herself when distributing balloons among friends -/
theorem winnie_balloons (total_balloons : ℕ) (num_friends : ℕ) (h1 : total_balloons = 226) (h2 : num_friends = 11) :
  total_balloons % num_friends = 6 := by
  sorry

end NUMINAMATH_CALUDE_winnie_balloons_l3041_304171


namespace NUMINAMATH_CALUDE_solve_equation_l3041_304122

theorem solve_equation (x : ℝ) (h : (40 / x) - 1 = 19) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3041_304122


namespace NUMINAMATH_CALUDE_fast_clock_accuracy_l3041_304133

/-- Represents time in minutes since midnight -/
def Time := ℕ

/-- Converts hours and minutes to total minutes -/
def toMinutes (hours minutes : ℕ) : Time :=
  hours * 60 + minutes

/-- A fast-running clock that gains time at a constant rate -/
structure FastClock where
  /-- The rate at which the clock gains time, represented as (gained_minutes, real_minutes) -/
  rate : ℕ × ℕ
  /-- The current time shown on the fast clock -/
  current_time : Time

/-- Calculates the actual time given a FastClock -/
def actualTime (clock : FastClock) (start_time : Time) : Time :=
  sorry

theorem fast_clock_accuracy (start_time : Time) (end_time : Time) :
  let initial_clock : FastClock := { rate := (15, 45), current_time := start_time }
  let final_clock : FastClock := { rate := (15, 45), current_time := end_time }
  start_time = toMinutes 15 0 →
  end_time = toMinutes 23 0 →
  actualTime final_clock start_time = toMinutes 23 15 :=
  sorry

end NUMINAMATH_CALUDE_fast_clock_accuracy_l3041_304133


namespace NUMINAMATH_CALUDE_product_equals_3408_decimal_product_l3041_304160

theorem product_equals_3408 : 213 * 16 = 3408 := by
  sorry

-- Additional fact (not used in the proof)
theorem decimal_product : 0.16 * 2.13 = 0.3408 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_3408_decimal_product_l3041_304160


namespace NUMINAMATH_CALUDE_complex_number_modulus_l3041_304115

theorem complex_number_modulus (b : ℝ) : 
  let z : ℂ := (3 - b * Complex.I) / (2 + Complex.I)
  (z.re = z.im) → Complex.abs z = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l3041_304115


namespace NUMINAMATH_CALUDE_train_length_l3041_304137

/-- The length of a train given its speed and time to cross a pole. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 36 → time = 9 → speed * time * (5 / 18) = 90 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3041_304137


namespace NUMINAMATH_CALUDE_least_time_six_horses_at_start_l3041_304107

def horse_lap_time (k : ℕ) : ℕ := 2 * k - 1

def is_at_start (t : ℕ) (k : ℕ) : Prop :=
  t % (horse_lap_time k) = 0

def at_least_six_at_start (t : ℕ) : Prop :=
  ∃ (s : Finset ℕ), s.card ≥ 6 ∧ s ⊆ Finset.range 12 ∧ ∀ k ∈ s, is_at_start t (k + 1)

theorem least_time_six_horses_at_start :
  ∃! t : ℕ, t > 0 ∧ at_least_six_at_start t ∧ ∀ s, s > 0 ∧ s < t → ¬(at_least_six_at_start s) :=
by sorry

end NUMINAMATH_CALUDE_least_time_six_horses_at_start_l3041_304107


namespace NUMINAMATH_CALUDE_root_in_interval_l3041_304177

theorem root_in_interval : ∃ x : ℝ, 2 < x ∧ x < 3 ∧ Real.log x + x - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l3041_304177


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3041_304143

/-- The radius of the inscribed circle of a triangle with side lengths 5, 12, and 13 is 2 -/
theorem inscribed_circle_radius (a b c r : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) :
  r = (a + b - c) / 2 → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3041_304143


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3041_304154

open Set

theorem intersection_of_sets : 
  let A : Set ℝ := {x | 1 < x ∧ x < 8}
  let B : Set ℝ := {1, 3, 5, 6, 7}
  A ∩ B = {3, 5, 6, 7} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3041_304154


namespace NUMINAMATH_CALUDE_triangle_properties_l3041_304113

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (A + B + C = π) →
  -- Side lengths are positive
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  -- Law of cosines
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →
  -- Prove the three properties
  ((A > B ↔ Real.sin A > Real.sin B) ∧
   (B = π/3 ∧ b^2 = a*c → A = π/3 ∧ B = π/3 ∧ C = π/3) ∧
   (b = a * Real.cos C + c * Real.sin A → A = π/4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3041_304113


namespace NUMINAMATH_CALUDE_sarahs_pool_depth_is_five_l3041_304139

/-- The depth of Sarah's pool in feet -/
def sarahs_pool_depth : ℝ := 5

/-- The depth of John's pool in feet -/
def johns_pool_depth : ℝ := 15

/-- Theorem stating that Sarah's pool depth is 5 feet -/
theorem sarahs_pool_depth_is_five :
  sarahs_pool_depth = 5 ∧
  johns_pool_depth = 2 * sarahs_pool_depth + 5 :=
by sorry

end NUMINAMATH_CALUDE_sarahs_pool_depth_is_five_l3041_304139


namespace NUMINAMATH_CALUDE_transformation_has_integer_root_intermediate_l3041_304142

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Checks if a quadratic equation has integer roots -/
def has_integer_root (eq : QuadraticEquation) : Prop :=
  ∃ x : ℤ, eq.a * x^2 + eq.b * x + eq.c = 0

/-- Represents a single step in the transformation process -/
inductive TransformationStep
  | IncreaseP
  | DecreaseP
  | IncreaseQ
  | DecreaseQ

/-- Applies a transformation step to a quadratic equation -/
def apply_step (eq : QuadraticEquation) (step : TransformationStep) : QuadraticEquation :=
  match step with
  | TransformationStep.IncreaseP => ⟨eq.a, eq.b + 1, eq.c⟩
  | TransformationStep.DecreaseP => ⟨eq.a, eq.b - 1, eq.c⟩
  | TransformationStep.IncreaseQ => ⟨eq.a, eq.b, eq.c + 1⟩
  | TransformationStep.DecreaseQ => ⟨eq.a, eq.b, eq.c - 1⟩

theorem transformation_has_integer_root_intermediate 
  (initial : QuadraticEquation) 
  (final : QuadraticEquation) 
  (h_initial : initial = ⟨1, -2013, -13⟩) 
  (h_final : final = ⟨1, 13, 2013⟩) :
  ∀ steps : List TransformationStep, 
    (List.foldl apply_step initial steps = final) → 
    (∃ intermediate : QuadraticEquation, 
      intermediate ∈ List.scanl apply_step initial steps ∧ 
      has_integer_root intermediate) :=
sorry

end NUMINAMATH_CALUDE_transformation_has_integer_root_intermediate_l3041_304142


namespace NUMINAMATH_CALUDE_group_morphism_identity_or_inverse_l3041_304149

variable {G : Type*} [Group G]

theorem group_morphism_identity_or_inverse
  (no_order_4 : ∀ g : G, g^4 = 1 → g = 1)
  (f : G → G)
  (f_hom : ∀ x y : G, f (x * y) = f x * f y)
  (f_property : ∀ x : G, f x = x ∨ f x = x⁻¹) :
  (∀ x : G, f x = x) ∨ (∀ x : G, f x = x⁻¹) := by
  sorry

end NUMINAMATH_CALUDE_group_morphism_identity_or_inverse_l3041_304149


namespace NUMINAMATH_CALUDE_problem_solution_l3041_304125

theorem problem_solution : ∃ x : ℝ, (5 * 12) / (180 / 3) + x = 65 ∧ x = 64 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3041_304125


namespace NUMINAMATH_CALUDE_intersection_point_l3041_304123

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 3 = 0

-- Define a point on the y-axis
def on_y_axis (x y : ℝ) : Prop := x = 0

-- Theorem statement
theorem intersection_point :
  ∃ (y : ℝ), line_equation 0 y ∧ on_y_axis 0 y ∧ y = 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l3041_304123


namespace NUMINAMATH_CALUDE_train_speed_l3041_304173

/-- Proves that a train with given length and time to cross a pole has a specific speed in km/h -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 300) (h2 : crossing_time = 15) :
  (train_length / crossing_time) * 3.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3041_304173


namespace NUMINAMATH_CALUDE_base_10_to_12_conversion_l3041_304118

/-- Represents a digit in base 12 -/
inductive Base12Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

/-- Converts a Base12Digit to its corresponding natural number -/
def Base12Digit.toNat : Base12Digit → Nat
| D0 => 0
| D1 => 1
| D2 => 2
| D3 => 3
| D4 => 4
| D5 => 5
| D6 => 6
| D7 => 7
| D8 => 8
| D9 => 9
| A => 10
| B => 11

/-- Represents a number in base 12 -/
def Base12Number := List Base12Digit

/-- Converts a Base12Number to its corresponding natural number -/
def Base12Number.toNat : Base12Number → Nat
| [] => 0
| d::ds => d.toNat * (12 ^ ds.length) + Base12Number.toNat ds

theorem base_10_to_12_conversion :
  Base12Number.toNat [Base12Digit.B, Base12Digit.D5] = 173 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_12_conversion_l3041_304118


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_l3041_304181

def total_players : ℕ := 16
def lineup_size : ℕ := 7
def num_twins : ℕ := 2

theorem volleyball_lineup_count : 
  (Nat.choose total_players lineup_size) - 
  (Nat.choose (total_players - num_twins) lineup_size) = 8008 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_l3041_304181


namespace NUMINAMATH_CALUDE_bridge_length_l3041_304170

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 160 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 215 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3041_304170


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_fourth_root_l3041_304190

theorem arithmetic_mean_geq_geometric_mean_fourth_root
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) / 4 ≥ (a * b * c * d) ^ (1/4) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_fourth_root_l3041_304190


namespace NUMINAMATH_CALUDE_average_difference_l3041_304176

-- Define the number of students and teachers
def num_students : ℕ := 120
def num_teachers : ℕ := 6

-- Define the class enrollments
def class_enrollments : List ℕ := [40, 40, 20, 10, 5, 5]

-- Define t (average number of students per teacher)
def t : ℚ := (num_students : ℚ) / num_teachers

-- Define s (average number of students per student)
def s : ℚ := (List.sum (List.map (fun x => x * x) class_enrollments) : ℚ) / num_students

-- Theorem to prove
theorem average_difference : t - s = -11.25 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l3041_304176


namespace NUMINAMATH_CALUDE_math_team_combinations_l3041_304194

def number_of_teams (n_girls m_boys k_girls l_boys : ℕ) : ℕ :=
  Nat.choose n_girls k_girls * Nat.choose m_boys l_boys

theorem math_team_combinations :
  number_of_teams 5 7 3 2 = 210 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l3041_304194


namespace NUMINAMATH_CALUDE_square_land_area_l3041_304109

/-- A square land plot with side length 40 units has an area of 1600 square units. -/
theorem square_land_area : 
  ∀ (side_length area : ℝ), 
  side_length = 40 → 
  area = side_length ^ 2 → 
  area = 1600 :=
by sorry

end NUMINAMATH_CALUDE_square_land_area_l3041_304109


namespace NUMINAMATH_CALUDE_speed_equivalence_l3041_304155

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in m/s -/
def given_speed_mps : ℝ := 15.001199999999999

/-- The calculated speed in km/h -/
def calculated_speed_kmph : ℝ := 54.004319999999996

/-- Theorem stating that the calculated speed in km/h is equivalent to the given speed in m/s -/
theorem speed_equivalence : calculated_speed_kmph = given_speed_mps * mps_to_kmph := by
  sorry

#check speed_equivalence

end NUMINAMATH_CALUDE_speed_equivalence_l3041_304155


namespace NUMINAMATH_CALUDE_coin_value_equality_l3041_304165

theorem coin_value_equality (n : ℕ) : 
  (15 * 25 + 20 * 10 = 5 * 25 + n * 10) → n = 45 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_equality_l3041_304165


namespace NUMINAMATH_CALUDE_probability_not_all_same_dice_five_dice_not_all_same_l3041_304135

theorem probability_not_all_same_dice (n : ℕ) (s : ℕ) : 
  n > 0 → s > 1 → (1 - s / s^n : ℚ) = (s^n - s) / s^n := by sorry

-- The probability that five fair 6-sided dice don't all show the same number
theorem five_dice_not_all_same : 
  (1 - (6 : ℚ) / 6^5) = 1295 / 1296 := by
  have h : (1 - 6 / 6^5 : ℚ) = (6^5 - 6) / 6^5 := 
    probability_not_all_same_dice 5 6 (by norm_num) (by norm_num)
  rw [h]
  norm_num


end NUMINAMATH_CALUDE_probability_not_all_same_dice_five_dice_not_all_same_l3041_304135


namespace NUMINAMATH_CALUDE_trig_simplification_l3041_304183

theorem trig_simplification :
  1 / Real.sin (70 * π / 180) - Real.sqrt 3 / Real.cos (70 * π / 180) = -4 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3041_304183


namespace NUMINAMATH_CALUDE_complex_magnitude_calculation_l3041_304141

theorem complex_magnitude_calculation (ω : ℂ) (h : ω = 7 + 3*I) :
  Complex.abs (ω^2 + 5*ω + 50) = Real.sqrt 18874 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_calculation_l3041_304141


namespace NUMINAMATH_CALUDE_two_x_plus_three_equals_nine_l3041_304129

theorem two_x_plus_three_equals_nine (x : ℝ) (h : x = 3) : 2 * x + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_x_plus_three_equals_nine_l3041_304129


namespace NUMINAMATH_CALUDE_triangle_theorem_l3041_304185

-- Define the triangle ABC
structure Triangle (α : Type*) [Field α] where
  a : α
  b : α
  c : α

-- Define the existence of point P
def exists_unique_point (t : Triangle ℝ) : Prop :=
  t.c ≠ t.a ∧ t.a ≠ t.b

-- Define the angle BAC
noncomputable def angle_BAC (t : Triangle ℝ) : ℝ :=
  Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))

-- Main theorem
theorem triangle_theorem (t : Triangle ℝ) :
  (exists_unique_point t) ∧
  (∃ (P : ℝ × ℝ), (angle_BAC t < Real.pi / 3)) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3041_304185


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l3041_304166

theorem greatest_integer_radius (A : ℝ) (h : A < 100 * Real.pi) :
  ∃ (r : ℕ), r^2 * Real.pi ≤ A ∧ ∀ (s : ℕ), s^2 * Real.pi ≤ A → s ≤ r ∧ r = 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l3041_304166
