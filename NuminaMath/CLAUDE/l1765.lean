import Mathlib

namespace NUMINAMATH_CALUDE_smallest_odd_angle_in_right_triangle_l1765_176511

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem smallest_odd_angle_in_right_triangle :
  ∀ y : ℕ, 
    (is_odd y) →
    (∃ x : ℕ, 
      (is_even x) ∧ 
      (x + y = 90) ∧ 
      (x > y)) →
    y ≥ 31 :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_angle_in_right_triangle_l1765_176511


namespace NUMINAMATH_CALUDE_sum_of_two_primes_52_l1765_176547

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem sum_of_two_primes_52 :
  ∃! (count : ℕ), ∃ (pairs : List (ℕ × ℕ)),
    (∀ (p q : ℕ), (p, q) ∈ pairs → is_prime p ∧ is_prime q ∧ p + q = 52) ∧
    (∀ (p q : ℕ), is_prime p → is_prime q → p + q = 52 → (p, q) ∈ pairs ∨ (q, p) ∈ pairs) ∧
    count = pairs.length ∧
    count = 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_two_primes_52_l1765_176547


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1765_176563

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1765_176563


namespace NUMINAMATH_CALUDE_percent_equation_l1765_176572

theorem percent_equation (x y : ℝ) (P : ℝ) 
  (h1 : 0.25 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.25 * x) : 
  P = 15 := by
  sorry

end NUMINAMATH_CALUDE_percent_equation_l1765_176572


namespace NUMINAMATH_CALUDE_odd_function_implies_a_equals_negative_one_l1765_176594

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = (x+a+1)(x^2+a-1) -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  (x + a + 1) * (x^2 + a - 1)

theorem odd_function_implies_a_equals_negative_one :
  ∀ a : ℝ, IsOdd (f a) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_equals_negative_one_l1765_176594


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1765_176503

theorem sqrt_equation_solution :
  ∀ x : ℝ, (Real.sqrt ((1 + Real.sqrt 2) ^ x) + Real.sqrt ((1 - Real.sqrt 2) ^ x) = 2) ↔ (x = 0) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1765_176503


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l1765_176509

theorem complex_magnitude_equation (t : ℝ) : 
  (t > 0 ∧ Complex.abs (8 + t * Complex.I) = 15) ↔ t = Real.sqrt 161 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l1765_176509


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x3_prism_l1765_176527

/-- Represents a rectangular prism -/
structure RectangularPrism :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Counts the number of unpainted cubes in a painted rectangular prism -/
def count_unpainted_cubes (prism : RectangularPrism) : ℕ :=
  if prism.height ≤ 2 then 0
  else (prism.length - 2) * (prism.width - 2)

/-- Theorem stating that a 6 × 6 × 3 painted prism has 16 unpainted cubes -/
theorem unpainted_cubes_in_6x6x3_prism :
  let prism : RectangularPrism := ⟨6, 6, 3⟩
  count_unpainted_cubes prism = 16 := by
  sorry

#eval count_unpainted_cubes ⟨6, 6, 3⟩

end NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x3_prism_l1765_176527


namespace NUMINAMATH_CALUDE_cross_shape_surface_area_l1765_176595

/-- Represents a 3D shape made of unit cubes -/
structure CubeShape where
  num_cubes : ℕ
  exposed_faces : ℕ

/-- The cross-like shape made of 5 unit cubes -/
def cross_shape : CubeShape :=
  { num_cubes := 5,
    exposed_faces := 22 }

/-- Theorem stating that the surface area of the cross-shape is 22 square units -/
theorem cross_shape_surface_area :
  cross_shape.exposed_faces = 22 := by
  sorry

end NUMINAMATH_CALUDE_cross_shape_surface_area_l1765_176595


namespace NUMINAMATH_CALUDE_remainder_of_product_l1765_176584

theorem remainder_of_product (a b c : ℕ) (hc : c ≥ 3) 
  (ha : a % c = 1) (hb : b % c = 2) : (a * b) % c = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_l1765_176584


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1765_176599

/-- Given a rhombus with area 24 and one diagonal of length 6, its perimeter is 20. -/
theorem rhombus_perimeter (area : ℝ) (diagonal1 : ℝ) (perimeter : ℝ) : 
  area = 24 → diagonal1 = 6 → perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1765_176599


namespace NUMINAMATH_CALUDE_number_with_one_third_equal_to_twelve_l1765_176552

theorem number_with_one_third_equal_to_twelve (x : ℝ) : (1 / 3 : ℝ) * x = 12 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_with_one_third_equal_to_twelve_l1765_176552


namespace NUMINAMATH_CALUDE_product_of_difference_of_squares_l1765_176585

theorem product_of_difference_of_squares (a b x1 y1 x2 y2 : ℤ) 
  (ha : a = x1^2 - 5*y1^2) (hb : b = x2^2 - 5*y2^2) :
  ∃ u v : ℤ, a * b = u^2 - 5*v^2 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_of_squares_l1765_176585


namespace NUMINAMATH_CALUDE_discount_difference_l1765_176557

theorem discount_difference (bill : ℝ) (d1 d2 d3 : ℝ) : 
  bill = 20000 ∧ d1 = 0.3 ∧ d2 = 0.25 ∧ d3 = 0.05 →
  bill * (1 - d2) * (1 - d3) - bill * (1 - d1) = 250 :=
by sorry

end NUMINAMATH_CALUDE_discount_difference_l1765_176557


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1765_176593

/-- Given a boat that travels 6 km/hr along a stream and 2 km/hr against the same stream,
    its speed in still water is 4 km/hr. -/
theorem boat_speed_in_still_water (b s : ℝ) 
    (h1 : b + s = 6)  -- Speed along the stream
    (h2 : b - s = 2)  -- Speed against the stream
    : b = 4 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1765_176593


namespace NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_result_l1765_176504

/-- Calculates the length of a bridge given train specifications --/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length

/-- The length of the bridge is approximately 299.95 meters --/
theorem bridge_length_result : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |bridge_length_calculation 200 45 40 - 299.95| < ε :=
sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_result_l1765_176504


namespace NUMINAMATH_CALUDE_distribution_theorem_l1765_176541

-- Define the number of books and students
def num_books : ℕ := 5
def num_students : ℕ := 3

-- Define a function to calculate the number of distribution methods
def distribution_methods (n_books : ℕ) (n_students : ℕ) : ℕ :=
  -- Implementation details are not provided as per the instructions
  sorry

-- Theorem statement
theorem distribution_theorem :
  distribution_methods num_books num_students = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribution_theorem_l1765_176541


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_difference_l1765_176526

theorem cube_root_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (x₁ ≠ x₂) ∧ 
  ((9 - x₁^2 / 4)^(1/3 : ℝ) = -3) ∧ 
  ((9 - x₂^2 / 4)^(1/3 : ℝ) = -3) ∧ 
  (abs (x₁ - x₂) = 24) := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_difference_l1765_176526


namespace NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l1765_176536

-- Define the parabola function
def f (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- Theorem statement
theorem parabola_has_one_x_intercept :
  ∃! x : ℝ, ∃ y : ℝ, f y = x ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l1765_176536


namespace NUMINAMATH_CALUDE_tom_spent_seven_tickets_on_hat_l1765_176501

/-- The number of tickets Tom spent on the hat -/
def tickets_spent_on_hat (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (remaining_tickets : ℕ) : ℕ :=
  whack_a_mole_tickets + skee_ball_tickets - remaining_tickets

/-- Proof that Tom spent 7 tickets on the hat -/
theorem tom_spent_seven_tickets_on_hat :
  tickets_spent_on_hat 32 25 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_spent_seven_tickets_on_hat_l1765_176501


namespace NUMINAMATH_CALUDE_choose_four_from_fifteen_l1765_176581

theorem choose_four_from_fifteen (n : ℕ) (k : ℕ) : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_fifteen_l1765_176581


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_is_32_l1765_176551

/-- The area of the right triangle formed by the lines y = x, x = -8, and the x-axis -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    let line1 : ℝ → ℝ → Prop := fun x y => y = x
    let line2 : ℝ → Prop := fun x => x = -8
    let x_axis : ℝ → Prop := fun y => y = 0
    let intersection_point : ℝ × ℝ := (-8, -8)
    let base : ℝ := 8
    let height : ℝ := 8
    (∀ x y, line1 x y → line2 x → (x, y) = intersection_point) ∧
    (∀ x, line2 x → x_axis 0) ∧
    (area = (1/2) * base * height) →
    area = 32

theorem triangle_area_is_32 : triangle_area 32 := by sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_is_32_l1765_176551


namespace NUMINAMATH_CALUDE_m_range_l1765_176591

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the parabola
def on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 2 * P.1

-- Define the distance ratio condition
def satisfies_distance_ratio (P : ℝ × ℝ) (m : ℝ) : Prop :=
  (P.1 + 1)^2 + P.2^2 = m^2 * ((P.1 - 1)^2 + P.2^2)

-- Main theorem
theorem m_range (P : ℝ × ℝ) (m : ℝ) 
  (h1 : on_parabola P) 
  (h2 : satisfies_distance_ratio P m) : 
  1 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1765_176591


namespace NUMINAMATH_CALUDE_cow_count_is_16_l1765_176564

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs for a given animal count -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads for a given animal count -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem stating that if the total number of legs is 32 more than twice the number of heads,
    then the number of cows is 16 -/
theorem cow_count_is_16 (count : AnimalCount) :
    totalLegs count = 2 * totalHeads count + 32 → count.cows = 16 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_is_16_l1765_176564


namespace NUMINAMATH_CALUDE_square_root_equation_l1765_176520

theorem square_root_equation (n : ℝ) : 
  Real.sqrt (10 + n) = 9 → n = 71 := by sorry

end NUMINAMATH_CALUDE_square_root_equation_l1765_176520


namespace NUMINAMATH_CALUDE_triangle_properties_l1765_176582

open Real

theorem triangle_properties (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / sin A = b / sin B ∧ b / sin B = c / sin C ∧
  1 + (tan C / tan B) = 2 * a / b ∧
  cos (B + π/6) = 1/3 ∧
  (a + b)^2 - c^2 = 4 →
  C = π/3 ∧ 
  sin A = (2 * sqrt 6 + 1) / 6 ∧
  ∀ x y, x > 0 ∧ y > 0 ∧ (x + y)^2 - c^2 = 4 → 3*x + y ≥ 4 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1765_176582


namespace NUMINAMATH_CALUDE_average_sum_abs_diff_l1765_176524

/-- A permutation of integers from 1 to 12 -/
def Permutation := Fin 12 → Fin 12

/-- The sum of absolute differences for a given permutation -/
def sumAbsDiff (p : Permutation) : ℚ :=
  |p 0 - p 1| + |p 2 - p 3| + |p 4 - p 5| + |p 6 - p 7| + |p 8 - p 9| + |p 10 - p 11|

/-- The set of all permutations of integers from 1 to 12 -/
def allPermutations : Finset Permutation := sorry

/-- The average value of sumAbsDiff over all permutations -/
def averageValue : ℚ := (allPermutations.sum sumAbsDiff) / allPermutations.card

theorem average_sum_abs_diff : averageValue = 143 / 33 := by sorry

end NUMINAMATH_CALUDE_average_sum_abs_diff_l1765_176524


namespace NUMINAMATH_CALUDE_average_sq_feet_per_person_closest_to_500000_l1765_176531

/-- The population of the United States in 1980 -/
def us_population : ℕ := 226504825

/-- The area of the United States in square miles -/
def us_area : ℕ := 3615122

/-- The number of square feet in one square mile -/
def sq_feet_per_sq_mile : ℕ := 5280 * 5280

/-- The options for the average number of square feet per person -/
def options : List ℕ := [5000, 10000, 50000, 100000, 500000]

/-- The theorem stating that the average number of square feet per person
    is closest to 500,000 among the given options -/
theorem average_sq_feet_per_person_closest_to_500000 :
  let total_sq_feet : ℕ := us_area * sq_feet_per_sq_mile
  let avg_sq_feet_per_person : ℚ := (total_sq_feet : ℚ) / us_population
  (500000 : ℚ) = options.argmin (fun x => |avg_sq_feet_per_person - x|) := by
  sorry

end NUMINAMATH_CALUDE_average_sq_feet_per_person_closest_to_500000_l1765_176531


namespace NUMINAMATH_CALUDE_planes_with_three_common_points_l1765_176590

-- Define a plane in 3D space
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define a point in 3D space
def Point : Type := ℝ × ℝ × ℝ

-- Define what it means for a point to be on a plane
def on_plane (p : Point) (plane : Plane) : Prop :=
  let (x, y, z) := p
  plane x y z

-- Define what it means for two planes to intersect
def intersect (p1 p2 : Plane) : Prop :=
  ∃ (p : Point), on_plane p p1 ∧ on_plane p p2

-- Define what it means for two planes to coincide
def coincide (p1 p2 : Plane) : Prop :=
  ∀ (p : Point), on_plane p p1 ↔ on_plane p p2

-- Theorem statement
theorem planes_with_three_common_points 
  (p1 p2 : Plane) (a b c : Point)
  (h1 : on_plane a p1 ∧ on_plane a p2)
  (h2 : on_plane b p1 ∧ on_plane b p2)
  (h3 : on_plane c p1 ∧ on_plane c p2) :
  intersect p1 p2 ∨ coincide p1 p2 :=
sorry

end NUMINAMATH_CALUDE_planes_with_three_common_points_l1765_176590


namespace NUMINAMATH_CALUDE_valid_sequences_count_l1765_176559

def is_valid_sequence (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a ≤ 10 ∧ b ≤ 10 ∧ c ≤ 10 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  12 ≤ a + b + c ∧ a + b + c ≤ 30 ∧
  b - a = c - b

def count_valid_sequences : ℕ := sorry

theorem valid_sequences_count :
  count_valid_sequences = 34 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_count_l1765_176559


namespace NUMINAMATH_CALUDE_box_makers_solution_l1765_176592

/-- Represents the possible makers of the boxes -/
inductive Maker
| Bellini
| BelliniSon
| Cellini

/-- Represents the two boxes -/
inductive Box
| Gold
| Silver

/-- The inscription on the gold box -/
def gold_inscription (gold_maker silver_maker : Maker) : Prop :=
  (gold_maker = Maker.Bellini ∨ gold_maker = Maker.BelliniSon) → silver_maker = Maker.Cellini

/-- The inscription on the silver box -/
def silver_inscription (gold_maker : Maker) : Prop :=
  gold_maker = Maker.BelliniSon

/-- The theorem stating the solution to the problem -/
theorem box_makers_solution :
  ∃ (gold_maker silver_maker : Maker),
    gold_inscription gold_maker silver_maker ∧
    silver_inscription gold_maker ∧
    gold_maker = Maker.Bellini ∧
    silver_maker = Maker.Cellini :=
sorry

end NUMINAMATH_CALUDE_box_makers_solution_l1765_176592


namespace NUMINAMATH_CALUDE_sqrt_15_div_sqrt_5_eq_sqrt_3_l1765_176579

theorem sqrt_15_div_sqrt_5_eq_sqrt_3 : Real.sqrt 15 / Real.sqrt 5 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_div_sqrt_5_eq_sqrt_3_l1765_176579


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1765_176530

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1765_176530


namespace NUMINAMATH_CALUDE_square_sum_ge_twice_product_l1765_176515

theorem square_sum_ge_twice_product (x y : ℝ) : x^2 + y^2 ≥ 2*x*y := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_twice_product_l1765_176515


namespace NUMINAMATH_CALUDE_complement_of_N_l1765_176538

-- Define the universal set M
def M : Set Nat := {1, 2, 3, 4, 5}

-- Define the set N
def N : Set Nat := {2, 4}

-- State the theorem
theorem complement_of_N : (M \ N) = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_N_l1765_176538


namespace NUMINAMATH_CALUDE_labor_costs_l1765_176521

/-- Calculate the overall labor costs for one day given the salaries of different workers -/
theorem labor_costs (worker_salary : ℕ) : 
  worker_salary = 100 →
  (2 * worker_salary) + (2 * worker_salary) + (5/2 * worker_salary) = 650 := by
  sorry

#check labor_costs

end NUMINAMATH_CALUDE_labor_costs_l1765_176521


namespace NUMINAMATH_CALUDE_points_on_line_equidistant_l1765_176542

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the line 4x + 7y = 28 -/
def onLine (p : Point) : Prop :=
  4 * p.x + 7 * p.y = 28

/-- Defines the condition of being equidistant from coordinate axes -/
def equidistant (p : Point) : Prop :=
  |p.x| = |p.y|

/-- Defines the condition of being in quadrant I -/
def inQuadrantI (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Defines the condition of being in quadrant II -/
def inQuadrantII (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Defines the condition of being in quadrant III -/
def inQuadrantIII (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Defines the condition of being in quadrant IV -/
def inQuadrantIV (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

theorem points_on_line_equidistant :
  ∀ p : Point, onLine p ∧ equidistant p →
    (inQuadrantI p ∨ inQuadrantII p) ∧
    ¬(inQuadrantIII p ∨ inQuadrantIV p) :=
by sorry

end NUMINAMATH_CALUDE_points_on_line_equidistant_l1765_176542


namespace NUMINAMATH_CALUDE_pyramid_stack_balls_l1765_176596

/-- Represents a pyramid-shaped stack of balls -/
structure PyramidStack where
  top_layer : Nat
  layer_diff : Nat
  bottom_layer : Nat

/-- Calculates the number of layers in the pyramid stack -/
def num_layers (p : PyramidStack) : Nat :=
  (p.bottom_layer - p.top_layer) / p.layer_diff + 1

/-- Calculates the total number of balls in the pyramid stack -/
def total_balls (p : PyramidStack) : Nat :=
  let n := num_layers p
  n * (p.top_layer + p.bottom_layer) / 2

/-- Theorem: The total number of balls in the given pyramid stack is 247 -/
theorem pyramid_stack_balls :
  let p := PyramidStack.mk 1 3 37
  total_balls p = 247 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_stack_balls_l1765_176596


namespace NUMINAMATH_CALUDE_inequality_proof_l1765_176505

theorem inequality_proof (x₁ x₂ x₃ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) :
  (x₁^2 + x₂^2 + x₃^2)^3 ≤ 3 * (x₁^3 + x₂^3 + x₃^3)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1765_176505


namespace NUMINAMATH_CALUDE_franks_daily_cookie_consumption_l1765_176598

/-- Proves that Frank eats 1 cookie each day given the conditions of the problem -/
theorem franks_daily_cookie_consumption :
  let days : ℕ := 6
  let trays_per_day : ℕ := 2
  let cookies_per_tray : ℕ := 12
  let ted_cookies : ℕ := 4
  let cookies_left : ℕ := 134
  let total_baked : ℕ := days * trays_per_day * cookies_per_tray
  let franks_total_consumption : ℕ := total_baked - ted_cookies - cookies_left
  franks_total_consumption / days = 1 := by
  sorry

end NUMINAMATH_CALUDE_franks_daily_cookie_consumption_l1765_176598


namespace NUMINAMATH_CALUDE_sock_shoe_permutations_l1765_176510

def num_legs : ℕ := 10

def total_items : ℕ := 2 * num_legs

def valid_permutations : ℕ := Nat.factorial total_items / (2^num_legs)

theorem sock_shoe_permutations :
  valid_permutations = Nat.factorial total_items / (2^num_legs) :=
by sorry

end NUMINAMATH_CALUDE_sock_shoe_permutations_l1765_176510


namespace NUMINAMATH_CALUDE_parabola_tangent_intercept_l1765_176517

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line on a 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of the specific parabola x² = 4y -/
def parabola_C : Parabola :=
  { equation := fun x y => x^2 = 4*y }

/-- Focus of the parabola -/
def F : Point :=
  { x := 0, y := 1 }

/-- Point E on the y-axis -/
def E : Point :=
  { x := 0, y := 3 }

/-- Origin -/
def O : Point :=
  { x := 0, y := 0 }

/-- Theorem statement -/
theorem parabola_tangent_intercept :
  ∀ (M : Point),
    parabola_C.equation M.x M.y →
    M.x ≠ 0 →
    (∃ (l : Line),
      -- l is tangent to the parabola at M
      (∀ (P : Point), P.y = l.slope * P.x + l.intercept → parabola_C.equation P.x P.y) →
      -- l passes through M
      M.y = l.slope * M.x + l.intercept →
      -- l is perpendicular to ME
      l.slope * ((M.y - E.y) / (M.x - E.x)) = -1 →
      -- y-intercept of l is -1
      l.intercept = -1) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_intercept_l1765_176517


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1765_176532

/-- The repeating decimal 0.4̄67 as a real number -/
def repeating_decimal : ℚ := 0.4 + (2/3) / 100

/-- The fraction 4621/9900 as a rational number -/
def target_fraction : ℚ := 4621 / 9900

/-- Theorem stating that the repeating decimal 0.4̄67 is equal to the fraction 4621/9900 -/
theorem repeating_decimal_equals_fraction :
  repeating_decimal = target_fraction := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1765_176532


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l1765_176555

/-- A quadratic radical is considered "simple" if it cannot be simplified further. -/
def IsSimpleQuadraticRadical (x : ℝ) : Prop :=
  x ≥ 0 ∧ ∀ y z : ℝ, x = y * y * z → y = 1 ∨ z < 0

/-- The set of quadratic radicals to consider -/
def QuadraticRadicals : Set ℝ := {4, 7, 12, 0.5}

theorem simplest_quadratic_radical :
  ∃ (x : ℝ), x ∈ QuadraticRadicals ∧
    IsSimpleQuadraticRadical (Real.sqrt x) ∧
    ∀ y ∈ QuadraticRadicals, IsSimpleQuadraticRadical (Real.sqrt y) → y = x :=
by
  sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l1765_176555


namespace NUMINAMATH_CALUDE_x_fifth_plus_64x_l1765_176586

theorem x_fifth_plus_64x (x : ℝ) (h : x^2 + 4*x = 8) : x^5 + 64*x = 768*x - 1024 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_plus_64x_l1765_176586


namespace NUMINAMATH_CALUDE_plan_A_first_9_minutes_charge_l1765_176506

/- Define the charge per minute after the first 9 minutes for plan A -/
def plan_A_rate : ℚ := 6 / 100

/- Define the charge per minute for plan B -/
def plan_B_rate : ℚ := 8 / 100

/- Define the duration at which both plans charge the same amount -/
def equal_duration : ℚ := 3

/- Theorem statement -/
theorem plan_A_first_9_minutes_charge : 
  ∃ (charge : ℚ), 
    charge = plan_B_rate * equal_duration ∧ 
    charge = 24 / 100 := by
  sorry

end NUMINAMATH_CALUDE_plan_A_first_9_minutes_charge_l1765_176506


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l1765_176568

/-- An ellipse with equation x²/m + y²/5 = 1 and focal length 2 has m = 4 -/
theorem ellipse_focal_length (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m + y^2 / 5 = 1) →  -- Ellipse equation
  2 = 2 * (Real.sqrt (5 - m)) →         -- Focal length is 2
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l1765_176568


namespace NUMINAMATH_CALUDE_correct_equation_l1765_176537

/-- Represents the situation described in the problem -/
structure Situation where
  x : ℕ  -- number of people
  total_cost : ℕ  -- total cost of the item

/-- The condition when each person contributes 8 coins -/
def condition_8 (s : Situation) : Prop :=
  8 * s.x = s.total_cost + 3

/-- The condition when each person contributes 7 coins -/
def condition_7 (s : Situation) : Prop :=
  7 * s.x + 4 = s.total_cost

/-- The theorem stating that the equation 8x - 3 = 7x + 4 correctly represents the situation -/
theorem correct_equation (s : Situation) :
  condition_8 s ∧ condition_7 s ↔ 8 * s.x - 3 = 7 * s.x + 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l1765_176537


namespace NUMINAMATH_CALUDE_evaluate_expression_l1765_176576

theorem evaluate_expression : (24^36) / (72^18) = 8^18 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1765_176576


namespace NUMINAMATH_CALUDE_ratio_equality_l1765_176589

theorem ratio_equality (x y a b : ℝ) 
  (h1 : x / y = 3)
  (h2 : (2 * a - x) / (3 * b - y) = 3) :
  a / b = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1765_176589


namespace NUMINAMATH_CALUDE_system_solution_l1765_176546

theorem system_solution (x y z : ℝ) 
  (eq1 : y + z = 10 - 4*x)
  (eq2 : x + z = -16 - 4*y)
  (eq3 : x + y = 9 - 4*z) :
  3*x + 3*y + 3*z = 1.5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1765_176546


namespace NUMINAMATH_CALUDE_equation_solution_l1765_176508

theorem equation_solution (x : ℝ) :
  (Real.sqrt (2 * x + 7)) / (Real.sqrt (8 * x + 10)) = 2 / Real.sqrt 5 →
  x = -5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1765_176508


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1765_176583

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, ax^2 + bx - 2 = 0 ↔ x = -2 ∨ x = -1/4) → 
  a + b = -13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1765_176583


namespace NUMINAMATH_CALUDE_parallelogram_area_l1765_176523

theorem parallelogram_area (side1 side2 angle : ℝ) (h_side1 : side1 = 20) (h_side2 : side2 = 30) (h_angle : angle = 40 * π / 180) :
  let height := side1 * Real.sin angle
  let area := side2 * height
  ∃ ε > 0, |area - 385.68| < ε :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1765_176523


namespace NUMINAMATH_CALUDE_car_distance_l1765_176562

theorem car_distance (efficiency : ℝ) (gas : ℝ) (distance : ℝ) :
  efficiency = 20 →
  gas = 5 →
  distance = efficiency * gas →
  distance = 100 := by sorry

end NUMINAMATH_CALUDE_car_distance_l1765_176562


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1765_176549

theorem trigonometric_identity (c : ℝ) (h : c = 2 * Real.pi / 13) :
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c) * Real.sin (16 * c) * Real.sin (20 * c)) /
  (Real.sin c * Real.sin (2 * c) * Real.sin (3 * c) * Real.sin (4 * c) * Real.sin (5 * c)) =
  1 / Real.sin (2 * Real.pi / 13) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1765_176549


namespace NUMINAMATH_CALUDE_grayson_speed_l1765_176570

/-- Grayson's motorboat trip -/
structure GraysonTrip where
  speed1 : ℝ  -- Speed during the first hour
  time1 : ℝ   -- Time of the first part (1 hour)
  speed2 : ℝ  -- Speed during the second part (20 mph)
  time2 : ℝ   -- Time of the second part (0.5 hours)

/-- Rudy's rowboat trip -/
structure RudyTrip where
  speed : ℝ   -- Rudy's speed (10 mph)
  time : ℝ    -- Rudy's travel time (3 hours)

/-- The main theorem -/
theorem grayson_speed (g : GraysonTrip) (r : RudyTrip) 
  (h1 : g.time1 = 1)
  (h2 : g.time2 = 0.5)
  (h3 : g.speed2 = 20)
  (h4 : r.speed = 10)
  (h5 : r.time = 3)
  (h6 : g.speed1 * g.time1 + g.speed2 * g.time2 = r.speed * r.time + 5) :
  g.speed1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_grayson_speed_l1765_176570


namespace NUMINAMATH_CALUDE_set_game_combinations_l1765_176540

theorem set_game_combinations (n : ℕ) (k : ℕ) (h1 : n = 81) (h2 : k = 3) :
  Nat.choose n k = 85320 := by
  sorry

end NUMINAMATH_CALUDE_set_game_combinations_l1765_176540


namespace NUMINAMATH_CALUDE_determine_c_l1765_176587

theorem determine_c (b c : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_determine_c_l1765_176587


namespace NUMINAMATH_CALUDE_gcd_2023_1991_l1765_176574

theorem gcd_2023_1991 : Nat.gcd 2023 1991 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2023_1991_l1765_176574


namespace NUMINAMATH_CALUDE_set_operations_correctness_l1765_176573

variable {α : Type*}
variable (A B C : Set α)

theorem set_operations_correctness :
  (A ∪ B = B ∪ A) ∧
  (A ∪ (B ∪ C) = (A ∪ B) ∪ C) ∧
  (A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_correctness_l1765_176573


namespace NUMINAMATH_CALUDE_both_hit_probability_l1765_176566

def prob_both_hit (prob_A prob_B : ℝ) : ℝ := prob_A * prob_B

theorem both_hit_probability :
  let prob_A : ℝ := 0.8
  let prob_B : ℝ := 0.7
  prob_both_hit prob_A prob_B = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_both_hit_probability_l1765_176566


namespace NUMINAMATH_CALUDE_lcm_of_36_and_125_l1765_176565

theorem lcm_of_36_and_125 : Nat.lcm 36 125 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_125_l1765_176565


namespace NUMINAMATH_CALUDE_x_one_value_l1765_176550

theorem x_one_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h_eq : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + (x₃ - x₄)^2 + x₄^2 = 1/5) :
  x₁ = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_x_one_value_l1765_176550


namespace NUMINAMATH_CALUDE_faster_speed_calculation_l1765_176548

/-- Proves that given a 1200-mile trip, if driving at a certain speed saves 4 hours
    compared to driving at 50 miles per hour, then that certain speed is 60 miles per hour. -/
theorem faster_speed_calculation (trip_distance : ℝ) (original_speed : ℝ) (time_saved : ℝ) 
    (faster_speed : ℝ) : 
    trip_distance = 1200 → 
    original_speed = 50 → 
    time_saved = 4 → 
    trip_distance / original_speed - trip_distance / faster_speed = time_saved → 
    faster_speed = 60 := by
  sorry

#check faster_speed_calculation

end NUMINAMATH_CALUDE_faster_speed_calculation_l1765_176548


namespace NUMINAMATH_CALUDE_timothy_total_cost_l1765_176518

/-- Calculates the total cost of Timothy's purchases --/
def total_cost (land_acres : ℕ) (land_price_per_acre : ℕ) (house_price : ℕ) 
  (num_cows : ℕ) (cow_price : ℕ) (num_chickens : ℕ) (chicken_price : ℕ)
  (solar_install_hours : ℕ) (solar_install_rate : ℕ) (solar_equipment_price : ℕ) : ℕ :=
  land_acres * land_price_per_acre +
  house_price +
  num_cows * cow_price +
  num_chickens * chicken_price +
  solar_install_hours * solar_install_rate + solar_equipment_price

/-- Theorem stating that the total cost of Timothy's purchases is $147,700 --/
theorem timothy_total_cost :
  total_cost 30 20 120000 20 1000 100 5 6 100 6000 = 147700 := by
  sorry

end NUMINAMATH_CALUDE_timothy_total_cost_l1765_176518


namespace NUMINAMATH_CALUDE_no_valid_partition_l1765_176545

-- Define a partition type
def Partition := ℤ → Fin 3

-- Define the property that n, n-50, and n+1987 belong to different subsets
def ValidPartition (p : Partition) : Prop :=
  ∀ n : ℤ, p n ≠ p (n - 50) ∧ p n ≠ p (n + 1987) ∧ p (n - 50) ≠ p (n + 1987)

-- Theorem statement
theorem no_valid_partition : ¬∃ p : Partition, ValidPartition p := by
  sorry

end NUMINAMATH_CALUDE_no_valid_partition_l1765_176545


namespace NUMINAMATH_CALUDE_excluded_students_average_mark_l1765_176580

theorem excluded_students_average_mark 
  (N : ℕ) 
  (A : ℚ) 
  (E : ℕ) 
  (A_remaining : ℚ) 
  (h1 : N = 25)
  (h2 : A = 80)
  (h3 : E = 5)
  (h4 : A_remaining = 95) :
  let A_excluded := ((N : ℚ) * A - (N - E : ℚ) * A_remaining) / E
  A_excluded = 20 := by
sorry

end NUMINAMATH_CALUDE_excluded_students_average_mark_l1765_176580


namespace NUMINAMATH_CALUDE_regular_tetrahedron_on_sphere_volume_l1765_176558

/-- A regular tetrahedron with vertices on a unit sphere -/
structure RegularTetrahedronOnSphere where
  /-- The tetrahedron is regular -/
  is_regular : Bool
  /-- The vertices are on the surface of a unit sphere -/
  vertices_on_sphere : Bool
  /-- The base vertices are on a great circle of the sphere -/
  base_on_great_circle : Bool

/-- The volume of a regular tetrahedron on a unit sphere -/
def volume (t : RegularTetrahedronOnSphere) : ℝ :=
  sorry

/-- Theorem: The volume of a regular tetrahedron with vertices on a unit sphere is √3/4 -/
theorem regular_tetrahedron_on_sphere_volume 
  (t : RegularTetrahedronOnSphere) 
  (h1 : t.is_regular = true) 
  (h2 : t.vertices_on_sphere = true) 
  (h3 : t.base_on_great_circle = true) : 
  volume t = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_on_sphere_volume_l1765_176558


namespace NUMINAMATH_CALUDE_trigonometric_equalities_l1765_176519

theorem trigonometric_equalities (α : ℝ) (h : 2 * Real.sin α + Real.cos α = 0) :
  (2 * Real.cos α - Real.sin α) / (Real.sin α + Real.cos α) = 5 ∧
  Real.sin α / (Real.sin α ^ 3 - Real.cos α ^ 3) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equalities_l1765_176519


namespace NUMINAMATH_CALUDE_sqrt_less_than_3x_plus_1_l1765_176553

theorem sqrt_less_than_3x_plus_1 (x : ℝ) (hx : x > 0) : Real.sqrt x < 3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_less_than_3x_plus_1_l1765_176553


namespace NUMINAMATH_CALUDE_inheritance_problem_l1765_176575

theorem inheritance_problem (total_amount : ℕ) (amount_per_person : ℕ) 
  (h1 : total_amount = 527500)
  (h2 : amount_per_person = 105500) :
  total_amount / amount_per_person = 5 :=
by sorry

end NUMINAMATH_CALUDE_inheritance_problem_l1765_176575


namespace NUMINAMATH_CALUDE_max_colored_pages_l1765_176569

/-- The cost in cents to print a colored page -/
def cost_per_page : ℕ := 4

/-- The budget in dollars -/
def budget : ℕ := 30

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The maximum number of colored pages that can be printed -/
def max_pages : ℕ := (budget * cents_per_dollar) / cost_per_page

theorem max_colored_pages : max_pages = 750 := by
  sorry

end NUMINAMATH_CALUDE_max_colored_pages_l1765_176569


namespace NUMINAMATH_CALUDE_puzzle_solutions_l1765_176560

def is_valid_solution (a b : Nat) : Prop :=
  a ≠ b ∧
  a ≥ 1 ∧ a ≤ 9 ∧
  b ≥ 1 ∧ b ≤ 9 ∧
  a^b = 10*b + a ∧
  10*b + a ≠ b*a

theorem puzzle_solutions :
  {(a, b) : Nat × Nat | is_valid_solution a b} =
  {(2, 5), (6, 2), (4, 3)} :=
sorry

end NUMINAMATH_CALUDE_puzzle_solutions_l1765_176560


namespace NUMINAMATH_CALUDE_perimeter_is_120_inches_l1765_176513

/-- The perimeter of a figure formed by cutting an equilateral triangle from a square and rotating it -/
def rotated_triangle_perimeter (square_side : ℝ) (triangle_side : ℝ) : ℝ :=
  3 * square_side + 3 * triangle_side

/-- Theorem: The perimeter of the new figure is 120 inches -/
theorem perimeter_is_120_inches :
  let square_side := (20 : ℝ)
  let triangle_side := (20 : ℝ)
  rotated_triangle_perimeter square_side triangle_side = 120 := by
  sorry

#eval rotated_triangle_perimeter 20 20

end NUMINAMATH_CALUDE_perimeter_is_120_inches_l1765_176513


namespace NUMINAMATH_CALUDE_negative_exponent_equality_l1765_176507

theorem negative_exponent_equality : -5^3 = -(5^3) := by
  sorry

end NUMINAMATH_CALUDE_negative_exponent_equality_l1765_176507


namespace NUMINAMATH_CALUDE_student_grade_problem_l1765_176525

/-- Given a student's grades in three subjects, prove that if the second subject is 70%,
    the third subject is 90%, and the overall average is 70%, then the first subject must be 50%. -/
theorem student_grade_problem (grade1 grade2 grade3 : ℝ) : 
  grade2 = 70 → grade3 = 90 → (grade1 + grade2 + grade3) / 3 = 70 → grade1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_student_grade_problem_l1765_176525


namespace NUMINAMATH_CALUDE_c_is_winner_l1765_176571

-- Define the candidates
inductive Candidate
| A
| B
| C

-- Define the election result
structure ElectionResult where
  total_voters : Nat
  votes : Candidate → Nat
  vote_count_valid : votes Candidate.A + votes Candidate.B + votes Candidate.C = total_voters

-- Define the winner selection rule
def is_winner (result : ElectionResult) (c : Candidate) : Prop :=
  ∀ other : Candidate, result.votes c ≥ result.votes other

-- Theorem statement
theorem c_is_winner (result : ElectionResult) 
  (h_total : result.total_voters = 30)
  (h_a : result.votes Candidate.A = 12)
  (h_b : result.votes Candidate.B = 3)
  (h_c : result.votes Candidate.C = 15) :
  is_winner result Candidate.C :=
by sorry

end NUMINAMATH_CALUDE_c_is_winner_l1765_176571


namespace NUMINAMATH_CALUDE_least_number_satisfying_conditions_l1765_176578

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def leaves_remainder_2 (n : ℕ) (d : ℕ) : Prop := n % d = 2

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by_9 n ∧
  (∀ d : ℕ, 3 ≤ d ∧ d ≤ 7 → leaves_remainder_2 n d)

theorem least_number_satisfying_conditions :
  satisfies_conditions 6302 ∧
  ∀ m : ℕ, m < 6302 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_least_number_satisfying_conditions_l1765_176578


namespace NUMINAMATH_CALUDE_cube_volume_from_doubled_cuboid_edges_l1765_176533

theorem cube_volume_from_doubled_cuboid_edges (l w h : ℝ) : 
  l * w * h = 36 → (2 * l) * (2 * w) * (2 * h) = 288 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_doubled_cuboid_edges_l1765_176533


namespace NUMINAMATH_CALUDE_face_D_opposite_Y_l1765_176597

-- Define the faces of the cube
inductive Face
| A | B | C | D | E | Y

-- Define the structure of the net
structure Net :=
  (faces : List Face)
  (adjacent : Face → Face → Bool)

-- Define the structure of the cube
structure Cube :=
  (faces : List Face)
  (opposite : Face → Face)

-- Define the folding operation
def fold (net : Net) : Cube :=
  sorry

-- The theorem to prove
theorem face_D_opposite_Y (net : Net) (cube : Cube) :
  net.faces = [Face.A, Face.B, Face.C, Face.D, Face.E, Face.Y] →
  cube = fold net →
  cube.opposite Face.Y = Face.D :=
sorry

end NUMINAMATH_CALUDE_face_D_opposite_Y_l1765_176597


namespace NUMINAMATH_CALUDE_inequality_proof_l1765_176556

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * (a + b)) + Real.sqrt (b * c * (b + c)) + Real.sqrt (c * a * (c + a)) >
  Real.sqrt ((a + b) * (b + c) * (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1765_176556


namespace NUMINAMATH_CALUDE_covid_cases_after_growth_l1765_176567

/-- Calculates the total number of COVID-19 cases in New York, California, and Texas after one month of growth --/
theorem covid_cases_after_growth (new_york_initial : ℕ) 
  (h1 : new_york_initial = 2000)
  (h2 : ∃ california_initial : ℕ, california_initial = new_york_initial / 2)
  (h3 : ∃ texas_initial : ℕ, ∃ california_initial : ℕ, 
    california_initial = new_york_initial / 2 ∧ 
    california_initial = texas_initial + 400)
  (h4 : ∃ new_york_growth : ℚ, new_york_growth = 25 / 100)
  (h5 : ∃ california_growth : ℚ, california_growth = 15 / 100)
  (h6 : ∃ texas_growth : ℚ, texas_growth = 30 / 100) :
  ∃ total_cases : ℕ, total_cases = 4430 := by
sorry

end NUMINAMATH_CALUDE_covid_cases_after_growth_l1765_176567


namespace NUMINAMATH_CALUDE_area_triangle_PTU_l1765_176543

/-- Regular octagon with side length 3 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Triangle formed by vertices P, T, and U in the regular octagon -/
def triangle_PTU (octagon : RegularOctagon) : Set (Fin 3 → ℝ × ℝ) := sorry

/-- Area of a triangle -/
def triangle_area (t : Set (Fin 3 → ℝ × ℝ)) : ℝ := sorry

/-- Main theorem: Area of triangle PTU in a regular octagon with side length 3 -/
theorem area_triangle_PTU (octagon : RegularOctagon) :
  triangle_area (triangle_PTU octagon) = (9 * Real.sqrt 2 + 9) / 2 := by sorry

end NUMINAMATH_CALUDE_area_triangle_PTU_l1765_176543


namespace NUMINAMATH_CALUDE_house_transaction_problem_l1765_176514

/-- Represents a person's assets -/
structure Assets where
  cash : Int
  has_house : Bool

/-- Represents a transaction between two people -/
def transaction (seller buyer : Assets) (price : Int) : Assets × Assets :=
  ({ cash := seller.cash + price, has_house := false },
   { cash := buyer.cash - price, has_house := true })

/-- The problem statement -/
theorem house_transaction_problem :
  let initial_a : Assets := { cash := 15000, has_house := true }
  let initial_b : Assets := { cash := 20000, has_house := false }
  let house_value := 15000

  let (a1, b1) := transaction initial_a initial_b 18000
  let (b2, a2) := transaction b1 a1 12000
  let (a3, b3) := transaction a2 b2 16000

  (a3.cash - initial_a.cash = 22000) ∧
  (b3.cash + house_value - initial_b.cash = -7000) := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_problem_l1765_176514


namespace NUMINAMATH_CALUDE_count_valid_n_l1765_176502

theorem count_valid_n : ∃! (s : Finset ℕ), 
  (∀ n ∈ s, 0 < n ∧ n < 35 ∧ ∃ k : ℕ, k > 0 ∧ n = k * (35 - n)) ∧ 
  s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_n_l1765_176502


namespace NUMINAMATH_CALUDE_neighborhood_households_l1765_176512

theorem neighborhood_households (no_car_no_bike : ℕ) (car_and_bike : ℕ) (total_with_car : ℕ) (bike_only : ℕ) :
  no_car_no_bike = 11 →
  car_and_bike = 18 →
  total_with_car = 44 →
  bike_only = 35 →
  no_car_no_bike + car_and_bike + (total_with_car - car_and_bike) + bike_only = 90 :=
by sorry

end NUMINAMATH_CALUDE_neighborhood_households_l1765_176512


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_8191_l1765_176588

def greatest_prime_divisor (n : Nat) : Nat :=
  sorry

def sum_of_digits (n : Nat) : Nat :=
  sorry

theorem sum_of_digits_of_greatest_prime_divisor_8191 :
  sum_of_digits (greatest_prime_divisor 8191) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_8191_l1765_176588


namespace NUMINAMATH_CALUDE_marias_gum_count_l1765_176535

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem marias_gum_count 
  (x y z : ℕ) 
  (hx : is_two_digit x) 
  (hy : is_two_digit y) 
  (hz : is_two_digit z) : 
  58 + x + y + z = 58 + x + y + z :=
by sorry

end NUMINAMATH_CALUDE_marias_gum_count_l1765_176535


namespace NUMINAMATH_CALUDE_proposition_truth_l1765_176577

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 0 → x^2 - 2*x + 1 > 0

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 2*x₀ + 1 ≤ 0

-- Theorem to prove
theorem proposition_truth : ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_proposition_truth_l1765_176577


namespace NUMINAMATH_CALUDE_det_equality_l1765_176516

variables (a b c d : ℝ)

/-- The determinant of a 2x2 matrix -/
def det2 (m : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  m 0 0 * m 1 1 - m 0 1 * m 1 0

theorem det_equality (h : det2 !![a, b; c, d] = 7) :
  det2 !![a + 2*c, b + 2*d; c, d] = 7 := by
  sorry

end NUMINAMATH_CALUDE_det_equality_l1765_176516


namespace NUMINAMATH_CALUDE_root_of_equation_l1765_176554

theorem root_of_equation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x > 0, (Real.sqrt (a * b * x * (a + b + x)) + 
             Real.sqrt (b * c * x * (b + c + x)) + 
             Real.sqrt (c * a * x * (c + a + x)) = 
             Real.sqrt (a * b * c * (a + b + c))) ∧
           (x = (a * b * c) / (a * b + b * c + c * a + 2 * Real.sqrt (a * b * c * (a + b + c)))) :=
by sorry

end NUMINAMATH_CALUDE_root_of_equation_l1765_176554


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1765_176529

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1 → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x > 1 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1765_176529


namespace NUMINAMATH_CALUDE_largest_n_divisibility_n_890_divisibility_n_890_largest_l1765_176500

theorem largest_n_divisibility : ∀ n : ℕ, n > 890 → ¬(n + 10 ∣ n^3 + 100) :=
by
  sorry

theorem n_890_divisibility : (890 + 10 ∣ 890^3 + 100) :=
by
  sorry

theorem n_890_largest : ∀ n : ℕ, n > 0 → (n + 10 ∣ n^3 + 100) → n ≤ 890 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_n_890_divisibility_n_890_largest_l1765_176500


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l1765_176539

theorem isosceles_right_triangle (a b c : ℝ) :
  a = 2 * Real.sqrt 6 ∧ 
  b = 2 * Real.sqrt 3 ∧ 
  c = 2 * Real.sqrt 3 →
  (a^2 = b^2 + c^2) ∧ (b = c) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_l1765_176539


namespace NUMINAMATH_CALUDE_second_plan_fee_calculation_l1765_176534

/-- The monthly fee for the first plan -/
def first_plan_monthly_fee : ℚ := 22

/-- The per-minute fee for the first plan -/
def first_plan_per_minute : ℚ := 13 / 100

/-- The monthly fee for the second plan -/
def second_plan_monthly_fee : ℚ := 8

/-- The number of minutes at which both plans cost the same -/
def equal_cost_minutes : ℚ := 280

/-- The per-minute fee for the second plan -/
def second_plan_per_minute : ℚ := 18 / 100

theorem second_plan_fee_calculation :
  first_plan_monthly_fee + first_plan_per_minute * equal_cost_minutes =
  second_plan_monthly_fee + second_plan_per_minute * equal_cost_minutes := by
  sorry

end NUMINAMATH_CALUDE_second_plan_fee_calculation_l1765_176534


namespace NUMINAMATH_CALUDE_map_distance_twenty_cm_distance_l1765_176522

-- Define the scale of the map
def map_scale (cm : ℝ) (km : ℝ) : Prop := cm * (54 / 9) = km

-- Theorem statement
theorem map_distance (cm : ℝ) :
  map_scale 9 54 → map_scale cm (cm * 6) :=
by
  sorry

-- The specific case for 20 cm
theorem twenty_cm_distance :
  map_scale 9 54 → map_scale 20 120 :=
by
  sorry

end NUMINAMATH_CALUDE_map_distance_twenty_cm_distance_l1765_176522


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l1765_176528

theorem rectangle_measurement_error (L W : ℝ) (L_excess W_deficit : ℝ) 
  (h1 : L_excess = 1.20)  -- 20% excess on first side
  (h2 : W_deficit > 0)    -- deficit percentage is positive
  (h3 : W_deficit < 1)    -- deficit percentage is less than 100%
  (h4 : L_excess * (1 - W_deficit) = 1.08)  -- 8% error in area
  : W_deficit = 0.10 :=   -- 10% deficit on second side
by sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l1765_176528


namespace NUMINAMATH_CALUDE_pauls_remaining_crayons_l1765_176544

/-- The number of crayons Paul had initially -/
def initial_crayons : ℕ := 479

/-- The number of crayons Paul lost or gave away -/
def lost_crayons : ℕ := 345

/-- The number of crayons Paul had left -/
def remaining_crayons : ℕ := initial_crayons - lost_crayons

theorem pauls_remaining_crayons : remaining_crayons = 134 := by
  sorry

end NUMINAMATH_CALUDE_pauls_remaining_crayons_l1765_176544


namespace NUMINAMATH_CALUDE_equation_solution_l1765_176561

theorem equation_solution : ∃! x : ℝ, (1 / (x + 12) + 1 / (x + 10) = 1 / (x + 13) + 1 / (x + 9)) ∧ x = -11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1765_176561
