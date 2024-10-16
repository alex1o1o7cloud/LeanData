import Mathlib

namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l3775_377594

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 4 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 4 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l3775_377594


namespace NUMINAMATH_CALUDE_carla_bob_payment_difference_l3775_377564

/-- Represents the pizza and its properties -/
structure Pizza :=
  (total_slices : ℕ)
  (vegetarian_slices : ℕ)
  (plain_cost : ℚ)
  (vegetarian_extra_cost : ℚ)

/-- Calculates the cost per slice of the pizza -/
def cost_per_slice (p : Pizza) : ℚ :=
  (p.plain_cost + p.vegetarian_extra_cost) / p.total_slices

/-- Calculates the cost for a given number of slices -/
def cost_for_slices (p : Pizza) (slices : ℕ) : ℚ :=
  (cost_per_slice p) * slices

/-- The main theorem to prove -/
theorem carla_bob_payment_difference
  (p : Pizza)
  (carla_slices bob_slices : ℕ)
  : p.total_slices = 12 →
    p.vegetarian_slices = 6 →
    p.plain_cost = 10 →
    p.vegetarian_extra_cost = 3 →
    carla_slices = 8 →
    bob_slices = 3 →
    (cost_for_slices p carla_slices) - (cost_for_slices p bob_slices) = 5.42 := by
  sorry

end NUMINAMATH_CALUDE_carla_bob_payment_difference_l3775_377564


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l3775_377527

-- Define the properties for m and n
def has_two_divisors (x : ℕ) : Prop := (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = 2

def has_four_divisors (x : ℕ) : Prop := (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = 4

def is_smallest_with_two_divisors (m : ℕ) : Prop :=
  has_two_divisors m ∧ ∀ k < m, ¬has_two_divisors k

def is_largest_under_200_with_four_divisors (n : ℕ) : Prop :=
  n < 200 ∧ has_four_divisors n ∧ ∀ k > n, k < 200 → ¬has_four_divisors k

-- State the theorem
theorem sum_of_special_numbers :
  ∃ (m n : ℕ), is_smallest_with_two_divisors m ∧ is_largest_under_200_with_four_divisors n ∧ m + n = 127 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l3775_377527


namespace NUMINAMATH_CALUDE_height_equality_l3775_377539

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  p : ℝ -- semiperimeter
  ha : ℝ -- height corresponding to side a

-- State the theorem
theorem height_equality (t : Triangle) : 
  t.ha = (2 * (t.p - t.a) * Real.cos (t.β / 2) * Real.cos (t.γ / 2)) / Real.cos (t.α / 2) ∧
  t.ha = (2 * (t.p - t.b) * Real.sin (t.β / 2) * Real.cos (t.γ / 2)) / Real.sin (t.α / 2) := by
  sorry

end NUMINAMATH_CALUDE_height_equality_l3775_377539


namespace NUMINAMATH_CALUDE_line_points_equation_l3775_377513

/-- Given a line and two points on it, prove an equation relating to the x-coordinate of the first point -/
theorem line_points_equation (m n : ℝ) : 
  (∀ x y, x - 5/2 * y + 1 = 0 → 
    ((x = m ∧ y = n) ∨ (x = m + 1/2 ∧ y = n + 1)) → 
      m + 1 = m - 3) := by
  sorry

end NUMINAMATH_CALUDE_line_points_equation_l3775_377513


namespace NUMINAMATH_CALUDE_triangle_angle_sum_equivalent_to_parallel_postulate_l3775_377535

-- Define Euclidean geometry
axiom EuclideanGeometry : Type

-- Define the parallel postulate
axiom parallel_postulate : EuclideanGeometry → Prop

-- Define the triangle angle sum theorem
axiom triangle_angle_sum : EuclideanGeometry → Prop

-- Theorem statement
theorem triangle_angle_sum_equivalent_to_parallel_postulate :
  ∀ (E : EuclideanGeometry), triangle_angle_sum E ↔ parallel_postulate E :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_equivalent_to_parallel_postulate_l3775_377535


namespace NUMINAMATH_CALUDE_odd_fraction_in_multiplication_table_l3775_377559

def table_size : Nat := 16

theorem odd_fraction_in_multiplication_table :
  let total_products := table_size * table_size
  let odd_products := (table_size / 2) * (table_size / 2)
  (odd_products : ℚ) / total_products = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_fraction_in_multiplication_table_l3775_377559


namespace NUMINAMATH_CALUDE_gcd_84_126_l3775_377534

theorem gcd_84_126 : Nat.gcd 84 126 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_84_126_l3775_377534


namespace NUMINAMATH_CALUDE_trajectory_and_m_value_l3775_377538

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 7

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 7

-- Define the line that intersects C
def intersecting_line (x y m : ℝ) : Prop := x + y - m = 0

-- Define the property that a circle passes through the origin
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem trajectory_and_m_value :
  ∀ (x₀ y₀ x y x₁ y₁ x₂ y₂ m : ℝ),
  (3/2, 0) = ((x₀ + x)/2, (y₀ + y)/2) →  -- A is midpoint of BM
  circle_O x₀ y₀ →  -- B is on circle O
  trajectory_C x y →  -- M is on trajectory C
  intersecting_line x₁ y₁ m →  -- P is on the intersecting line
  intersecting_line x₂ y₂ m →  -- Q is on the intersecting line
  trajectory_C x₁ y₁ →  -- P is on trajectory C
  trajectory_C x₂ y₂ →  -- Q is on trajectory C
  circle_through_origin x₁ y₁ x₂ y₂ →  -- Circle with PQ as diameter passes through origin
  (∀ x y, trajectory_C x y ↔ (x - 3)^2 + y^2 = 7) ∧  -- Trajectory equation is correct
  (m = 1 ∨ m = 2)  -- m value is correct
  := by sorry

end NUMINAMATH_CALUDE_trajectory_and_m_value_l3775_377538


namespace NUMINAMATH_CALUDE_circle_equation_l3775_377553

theorem circle_equation (x y : ℝ) : 
  (∃ c : ℝ, x^2 + (y - c)^2 = 1 ∧ 1^2 + (2 - c)^2 = 1) → 
  x^2 + (y - 2)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l3775_377553


namespace NUMINAMATH_CALUDE_pie_chart_most_suitable_l3775_377571

-- Define the available graph types
inductive GraphType
| PieChart
| BarGraph
| LineGraph

-- Define the expenditure categories
inductive ExpenditureCategory
| Education
| Clothing
| Food
| Other

-- Define a function to determine if a graph type is suitable for representing percentages
def isSuitableForPercentages (g : GraphType) : Prop :=
  match g with
  | GraphType.PieChart => True
  | _ => False

-- Define a function to check if a graph type can effectively show parts of a whole
def showsPartsOfWhole (g : GraphType) : Prop :=
  match g with
  | GraphType.PieChart => True
  | _ => False

-- Theorem stating that a pie chart is the most suitable graph type
theorem pie_chart_most_suitable (categories : List ExpenditureCategory) 
  (h1 : categories.length > 1) 
  (h2 : categories.length ≤ 4) : 
  ∃ (g : GraphType), isSuitableForPercentages g ∧ showsPartsOfWhole g :=
by
  sorry

end NUMINAMATH_CALUDE_pie_chart_most_suitable_l3775_377571


namespace NUMINAMATH_CALUDE_largest_x_absolute_value_equation_l3775_377581

theorem largest_x_absolute_value_equation : 
  (∃ x : ℝ, |5*x - 3| = 28) → 
  (∃ max_x : ℝ, |5*max_x - 3| = 28 ∧ ∀ y : ℝ, |5*y - 3| = 28 → y ≤ max_x) → 
  (∃ x : ℝ, |5*x - 3| = 28 ∧ ∀ y : ℝ, |5*y - 3| = 28 → y ≤ 31/5) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_absolute_value_equation_l3775_377581


namespace NUMINAMATH_CALUDE_workout_days_l3775_377574

/-- Represents the number of squats performed on a given day -/
def squats_on_day (initial_squats : ℕ) (daily_increase : ℕ) (day : ℕ) : ℕ :=
  initial_squats + (day - 1) * daily_increase

/-- The problem of determining the number of consecutive workout days -/
theorem workout_days (initial_squats : ℕ) (daily_increase : ℕ) (goal_squats : ℕ) : 
  initial_squats = 30 → 
  daily_increase = 5 → 
  goal_squats = 45 → 
  squats_on_day initial_squats daily_increase 4 = goal_squats := by
  sorry

#eval squats_on_day 30 5 4  -- Should evaluate to 45

end NUMINAMATH_CALUDE_workout_days_l3775_377574


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3775_377557

theorem complex_equation_solution (x y z : ℂ) (h_real : x.im = 0)
  (h_sum : x + y + z = 5)
  (h_prod_sum : x * y + y * z + z * x = 5)
  (h_prod : x * y * z = 5) :
  x = 1 + (4 : ℂ) ^ (1/3 : ℂ) :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3775_377557


namespace NUMINAMATH_CALUDE_range_of_x_l3775_377529

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem range_of_x (a : ℝ) (h1 : a > 1) (h2 : ∀ x, f x a ≥ 3) (h3 : ∃ x, f x a = 3) :
  ∀ x, f x a ≤ 5 ↔ 3 ≤ x ∧ x ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l3775_377529


namespace NUMINAMATH_CALUDE_range_of_g_l3775_377537

theorem range_of_g (x : ℝ) : 3/4 ≤ Real.cos x ^ 4 + Real.sin x ^ 2 ∧ Real.cos x ^ 4 + Real.sin x ^ 2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l3775_377537


namespace NUMINAMATH_CALUDE_complement_of_M_l3775_377507

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}

theorem complement_of_M : (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l3775_377507


namespace NUMINAMATH_CALUDE_pizza_slices_remaining_l3775_377547

theorem pizza_slices_remaining (total_slices : ℕ) (given_to_first_group : ℕ) (given_to_second_group : ℕ) :
  total_slices = 8 →
  given_to_first_group = 3 →
  given_to_second_group = 4 →
  total_slices - (given_to_first_group + given_to_second_group) = 1 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_remaining_l3775_377547


namespace NUMINAMATH_CALUDE_min_value_fraction_l3775_377569

theorem min_value_fraction (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (hsum : p + q + r = 2) : 
  (p + q) / (p * q * r) ≥ 9 ∧ ∃ p q r, p > 0 ∧ q > 0 ∧ r > 0 ∧ p + q + r = 2 ∧ (p + q) / (p * q * r) = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3775_377569


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3775_377576

/-- Given a geometric sequence {a_n} with common ratio q, if a₃ = 2S₂ + 1 and a₄ = 2S₃ + 1, then q = 3 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- Definition of sum of geometric sequence
  a 3 = 2 * S 2 + 1 →
  a 4 = 2 * S 3 + 1 →
  q = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3775_377576


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3775_377505

theorem geometric_sequence_sum (a : ℕ → ℚ) (x y : ℚ) :
  (∀ n, a (n + 1) = a n * (1/3)) →
  a 3 = x →
  a 4 = y →
  a 5 = 1 →
  a 6 = (1/3) →
  x + y = 36 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3775_377505


namespace NUMINAMATH_CALUDE_faster_train_distance_and_time_l3775_377593

/-- Represents the speed and distance of a train -/
structure Train where
  speed : ℝ
  distance : ℝ

/-- Proves the distance covered by a faster train and the time taken -/
theorem faster_train_distance_and_time 
  (old_train : Train)
  (new_train : Train)
  (speed_increase_percent : ℝ)
  (h1 : old_train.distance = 300)
  (h2 : new_train.speed = old_train.speed * (1 + speed_increase_percent))
  (h3 : speed_increase_percent = 0.3)
  (h4 : new_train.speed = 120) : 
  new_train.distance = 390 ∧ (new_train.distance / new_train.speed) = 3.25 := by
  sorry

#check faster_train_distance_and_time

end NUMINAMATH_CALUDE_faster_train_distance_and_time_l3775_377593


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3775_377583

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ 
  a ∈ Set.Ioc (-3/5) 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3775_377583


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3775_377565

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 10 = 0) : 
  x^3 - 3*x^2 - 10*x + 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3775_377565


namespace NUMINAMATH_CALUDE_condition_relationship_l3775_377500

theorem condition_relationship (x : ℝ) : 
  (∀ x, x = 1 → x^2 - 3*x + 2 = 0) ∧ 
  (∃ x, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l3775_377500


namespace NUMINAMATH_CALUDE_root_zero_implies_a_half_l3775_377541

theorem root_zero_implies_a_half (a : ℝ) : 
  (∀ x : ℝ, x^2 + x + 2*a - 1 = 0 → x = 0 ∨ x ≠ 0) →
  (0^2 + 0 + 2*a - 1 = 0) →
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_root_zero_implies_a_half_l3775_377541


namespace NUMINAMATH_CALUDE_second_longest_piece_length_l3775_377533

/-- The length of the second longest piece of rope when a 142.75-inch rope is cut into five pieces
    in the ratio (√2):6:(4/3):(3^2):(1/2) is approximately 46.938 inches. -/
theorem second_longest_piece_length (total_length : ℝ) (piece1 piece2 piece3 piece4 piece5 : ℝ)
  (h1 : total_length = 142.75)
  (h2 : piece1 / (Real.sqrt 2) = piece2 / 6)
  (h3 : piece2 / 6 = piece3 / (4/3))
  (h4 : piece3 / (4/3) = piece4 / 9)
  (h5 : piece4 / 9 = piece5 / (1/2))
  (h6 : piece1 + piece2 + piece3 + piece4 + piece5 = total_length) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |piece2 - 46.938| < ε ∧
  (piece2 > piece1 ∨ piece2 > piece3 ∨ piece2 > piece5) ∧
  (piece4 > piece2 ∨ piece4 = piece2) :=
by sorry

end NUMINAMATH_CALUDE_second_longest_piece_length_l3775_377533


namespace NUMINAMATH_CALUDE_factorization_equality_l3775_377599

theorem factorization_equality (m n : ℝ) : 2*m*n^2 - 12*m*n + 18*m = 2*m*(n-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3775_377599


namespace NUMINAMATH_CALUDE_point_moved_right_l3775_377542

def move_right (x y d : ℝ) : ℝ × ℝ := (x + d, y)

theorem point_moved_right :
  let A : ℝ × ℝ := (2, -1)
  let d : ℝ := 3
  move_right A.1 A.2 d = (5, -1) := by sorry

end NUMINAMATH_CALUDE_point_moved_right_l3775_377542


namespace NUMINAMATH_CALUDE_converse_not_always_true_l3775_377589

theorem converse_not_always_true : ¬(∀ a b c : ℝ, a > b → a * c^2 > b * c^2) := by
  sorry

end NUMINAMATH_CALUDE_converse_not_always_true_l3775_377589


namespace NUMINAMATH_CALUDE_equation_solution_l3775_377520

theorem equation_solution (a : ℝ) : 
  (2*a + 4*(-1) = (-1) + 5*a) → 
  (a = -1) ∧ 
  (∀ y : ℝ, (-1)*y + 6 = 6*(-1) + 2*y → y = 4) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3775_377520


namespace NUMINAMATH_CALUDE_Q_equals_N_l3775_377567

-- Define the sets Q and N
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {x : ℝ | x ≥ 1}

-- Theorem statement
theorem Q_equals_N : Q = N := by sorry

end NUMINAMATH_CALUDE_Q_equals_N_l3775_377567


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3775_377560

/-- The original polynomial expression -/
def original_expression (x : ℝ) : ℝ := 3 * (x^3 - 4*x^2 + 3*x - 1) - 5 * (2*x^3 - x^2 + x + 2)

/-- The simplified polynomial expression -/
def simplified_expression (x : ℝ) : ℝ := -7*x^3 - 7*x^2 + 4*x - 13

/-- Coefficients of the simplified expression -/
def coefficients : List ℝ := [-7, -7, 4, -13]

theorem sum_of_squared_coefficients :
  (original_expression = simplified_expression) →
  (List.sum (List.map (λ x => x^2) coefficients) = 283) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3775_377560


namespace NUMINAMATH_CALUDE_monotone_increasing_range_l3775_377570

/-- A function g(x) = ax³ + ax² + x is monotonically increasing on ℝ -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (a * x^3 + a * x^2 + x) < (a * y^3 + a * y^2 + y)

/-- The range of a for which g(x) = ax³ + ax² + x is monotonically increasing on ℝ -/
theorem monotone_increasing_range :
  ∀ a : ℝ, is_monotone_increasing a ↔ (0 ≤ a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_monotone_increasing_range_l3775_377570


namespace NUMINAMATH_CALUDE_exists_unique_decomposition_l3775_377563

def sequence_decomposition (x : ℕ → ℝ) : Prop :=
  ∃! (y z : ℕ → ℝ), 
    (∀ n : ℕ, x n = y n - z n) ∧
    (∀ n : ℕ, y n ≥ 0) ∧
    (∀ n : ℕ, n > 0 → z n ≥ z (n-1)) ∧
    (∀ n : ℕ, n > 0 → y n * (z n - z (n-1)) = 0) ∧
    (z 0 = 0)

theorem exists_unique_decomposition (x : ℕ → ℝ) : sequence_decomposition x := by
  sorry

end NUMINAMATH_CALUDE_exists_unique_decomposition_l3775_377563


namespace NUMINAMATH_CALUDE_jeds_speed_jeds_speed_is_66_l3775_377528

def fine_per_mph : ℝ := 16
def total_fine : ℝ := 256
def speed_limit : ℝ := 50

theorem jeds_speed : ℝ :=
  speed_limit + total_fine / fine_per_mph

theorem jeds_speed_is_66 : jeds_speed = 66 := by sorry

end NUMINAMATH_CALUDE_jeds_speed_jeds_speed_is_66_l3775_377528


namespace NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l3775_377579

/-- Given a sphere and a right circular cone where:
    1. The cone's height is twice the sphere's radius
    2. The cone's base radius is equal to the sphere's radius
    Prove that the ratio of the cone's volume to the sphere's volume is 1/2 -/
theorem cone_sphere_volume_ratio (r : ℝ) (h : ℝ) (h_pos : r > 0) (h_eq : h = 2 * r) :
  (1 / 3 * π * r^2 * h) / (4 / 3 * π * r^3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l3775_377579


namespace NUMINAMATH_CALUDE_sum_ten_consecutive_naturals_odd_l3775_377525

theorem sum_ten_consecutive_naturals_odd (n : ℕ) : ∃ k : ℕ, (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8) + (n + 9)) = 2 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_ten_consecutive_naturals_odd_l3775_377525


namespace NUMINAMATH_CALUDE_expand_difference_of_squares_l3775_377597

theorem expand_difference_of_squares (x y : ℝ) : 
  (x - y + 1) * (x - y - 1) = x^2 - 2*x*y + y^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expand_difference_of_squares_l3775_377597


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3775_377506

/-- Given a real number a, proves that "1, a, 16 form a geometric sequence" 
    is a necessary but not sufficient condition for "a=4" -/
theorem geometric_sequence_condition (a : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ a = r * 1 ∧ 16 = r * a) → 
  (a = 4 → ∃ r : ℝ, r ≠ 0 ∧ a = r * 1 ∧ 16 = r * a) ∧ 
  ¬(∃ r : ℝ, r ≠ 0 ∧ a = r * 1 ∧ 16 = r * a → a = 4) :=
by sorry


end NUMINAMATH_CALUDE_geometric_sequence_condition_l3775_377506


namespace NUMINAMATH_CALUDE_diamond_roof_diagonal_l3775_377573

/-- Given a diamond-shaped roof with area A and diagonals d1 and d2, 
    prove that if A = 80 and d1 = 16, then d2 = 10 -/
theorem diamond_roof_diagonal (A d1 d2 : ℝ) 
  (h_area : A = 80) 
  (h_diagonal : d1 = 16) 
  (h_shape : A = (d1 * d2) / 2) : 
  d2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_diamond_roof_diagonal_l3775_377573


namespace NUMINAMATH_CALUDE_total_apples_eaten_l3775_377596

def simone_daily_consumption : ℚ := 1/2
def simone_days : ℕ := 16
def lauri_daily_consumption : ℚ := 1/3
def lauri_days : ℕ := 15

theorem total_apples_eaten :
  simone_daily_consumption * simone_days + lauri_daily_consumption * lauri_days = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_eaten_l3775_377596


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l3775_377598

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : (m^2 + 2^m) % 10 = 3 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l3775_377598


namespace NUMINAMATH_CALUDE_arrangement_exists_for_23_l3775_377508

/-- Definition of the Fibonacci-like sequence -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of an arrangement for P = 23 -/
theorem arrangement_exists_for_23 : ∃ (F : ℕ → ℤ), F 12 % 23 = 0 ∧
  (∀ n, F (n + 2) = 3 * F (n + 1) - F n) ∧ F 0 = 0 ∧ F 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_arrangement_exists_for_23_l3775_377508


namespace NUMINAMATH_CALUDE_abcdefg_over_defghij_l3775_377551

theorem abcdefg_over_defghij (a b c d e f g h i j : ℚ)
  (h1 : a / b = -7 / 3)
  (h2 : b / c = -5 / 2)
  (h3 : c / d = 2)
  (h4 : d / e = -3 / 2)
  (h5 : e / f = 4 / 3)
  (h6 : f / g = -1 / 4)
  (h7 : g / h = 3 / -5)
  (h8 : i ≠ 0) -- Additional hypothesis to avoid division by zero
  : a * b * c * d * e * f * g / (d * e * f * g * h * i * j) = (-21 / 16) * (c / i) := by
  sorry

end NUMINAMATH_CALUDE_abcdefg_over_defghij_l3775_377551


namespace NUMINAMATH_CALUDE_vector_addition_theorem_l3775_377552

/-- Given vectors a and b, prove that 2a + b equals the specified result -/
theorem vector_addition_theorem (a b : ℝ × ℝ × ℝ) :
  a = (1, 2, -3) →
  b = (5, -7, 8) →
  (2 : ℝ) • a + b = (7, -3, 2) := by sorry

end NUMINAMATH_CALUDE_vector_addition_theorem_l3775_377552


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l3775_377532

theorem abs_sum_minimum (x : ℝ) : 
  |x + 3| + |x + 5| + |x + 6| ≥ 1 ∧ ∃ y : ℝ, |y + 3| + |y + 5| + |y + 6| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l3775_377532


namespace NUMINAMATH_CALUDE_only_propositions_3_and_4_are_correct_l3775_377501

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the relations between planes and lines
def perpendicular (p q : Plane) : Prop := sorry
def parallel (p q : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def intersection (p q : Plane) : Plane := sorry

-- Define the planes and lines
def α : Plane := sorry
def β : Plane := sorry
def γ : Plane := sorry
def l : Line := sorry
def m : Line := sorry
def n : Line := sorry

-- Define the propositions
def proposition_1 : Prop :=
  (perpendicular α γ ∧ perpendicular β γ) → parallel α β

def proposition_2 : Prop :=
  (parallel_line_plane m β ∧ parallel_line_plane n β) → parallel α β

def proposition_3 : Prop :=
  (line_in_plane l α ∧ parallel α β) → parallel_line_plane l β

def proposition_4 : Prop :=
  (intersection α β = γ ∧ intersection β γ = m ∧ intersection γ α = n ∧ parallel_line_plane l m) →
  parallel_line_plane m n

-- Theorem to prove
theorem only_propositions_3_and_4_are_correct :
  ¬proposition_1 ∧ ¬proposition_2 ∧ proposition_3 ∧ proposition_4 :=
sorry

end NUMINAMATH_CALUDE_only_propositions_3_and_4_are_correct_l3775_377501


namespace NUMINAMATH_CALUDE_smallest_divisible_by_10_11_12_l3775_377550

theorem smallest_divisible_by_10_11_12 : ∃ (n : ℕ), n > 0 ∧ 
  10 ∣ n ∧ 11 ∣ n ∧ 12 ∣ n ∧ 
  ∀ (m : ℕ), m > 0 ∧ 10 ∣ m ∧ 11 ∣ m ∧ 12 ∣ m → n ≤ m :=
by
  use 660
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_10_11_12_l3775_377550


namespace NUMINAMATH_CALUDE_cone_base_circumference_l3775_377509

/-- The circumference of the base of a right circular cone formed by removing a 180° sector from a circle with radius 6 inches is equal to 6π inches. -/
theorem cone_base_circumference (r : ℝ) (θ : ℝ) : 
  r = 6 → θ = 180 → 2 * π * r * (θ / 360) = 6 * π := by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l3775_377509


namespace NUMINAMATH_CALUDE_quadratic_distinct_rational_roots_l3775_377584

theorem quadratic_distinct_rational_roots (a b c : ℚ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hsum : a + b + c = 0) : 
  ∃ (x y : ℚ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_rational_roots_l3775_377584


namespace NUMINAMATH_CALUDE_beach_towel_loads_l3775_377586

/-- The number of laundry loads required for beach towels during a vacation. -/
def laundry_loads (num_families : ℕ) (people_per_family : ℕ) (days : ℕ) 
                  (towels_per_person_per_day : ℕ) (towels_per_load : ℕ) : ℕ :=
  (num_families * people_per_family * days * towels_per_person_per_day) / towels_per_load

theorem beach_towel_loads : 
  laundry_loads 7 6 10 2 10 = 84 := by sorry

end NUMINAMATH_CALUDE_beach_towel_loads_l3775_377586


namespace NUMINAMATH_CALUDE_range_of_a_l3775_377519

-- Define the propositions p and q
def p (x : ℝ) : Prop := |4 - x| ≤ 6
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≥ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, ¬(p x) → q x a) →
  (∃ x : ℝ, p x ∧ q x a) →
  (0 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3775_377519


namespace NUMINAMATH_CALUDE_expression_factorization_l3775_377504

theorem expression_factorization (x : ℝ) :
  (4 * x^3 - 64 * x^2 + 52) - (-3 * x^3 - 2 * x^2 + 52) = x^2 * (7 * x - 62) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3775_377504


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3775_377561

theorem average_speed_calculation (speed1 speed2 : ℝ) (h1 : speed1 = 70) (h2 : speed2 = 90) :
  (speed1 + speed2) / 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3775_377561


namespace NUMINAMATH_CALUDE_mindys_tax_rate_l3775_377546

/-- Given Mork's tax rate, Mindy's income relative to Mork's, and their combined tax rate,
    calculate Mindy's tax rate. -/
theorem mindys_tax_rate
  (morks_tax_rate : ℝ)
  (mindys_income_ratio : ℝ)
  (combined_tax_rate : ℝ)
  (h1 : morks_tax_rate = 0.40)
  (h2 : mindys_income_ratio = 3)
  (h3 : combined_tax_rate = 0.325) :
  let mindys_tax_rate := (combined_tax_rate * (1 + mindys_income_ratio) - morks_tax_rate) / mindys_income_ratio
  mindys_tax_rate = 0.30 :=
by sorry

end NUMINAMATH_CALUDE_mindys_tax_rate_l3775_377546


namespace NUMINAMATH_CALUDE_largest_negative_and_smallest_absolute_l3775_377587

theorem largest_negative_and_smallest_absolute : ∃ (a b : ℤ),
  (∀ x : ℤ, x < 0 → x ≤ a) ∧
  (∀ y : ℤ, |b| ≤ |y|) ∧
  b - 4*a = 4 :=
sorry

end NUMINAMATH_CALUDE_largest_negative_and_smallest_absolute_l3775_377587


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3775_377526

-- Define set A
def A : Set ℕ := {4, 5, 6, 7}

-- Define set B
def B : Set ℕ := {x | 3 ≤ x ∧ x < 6}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3775_377526


namespace NUMINAMATH_CALUDE_final_number_of_boys_l3775_377540

/-- Given the initial number of boys and additional boys in a school, 
    prove that the final number of boys is the sum of these two numbers. -/
theorem final_number_of_boys 
  (initial_boys : ℕ) 
  (additional_boys : ℕ) : 
  initial_boys + additional_boys = initial_boys + additional_boys :=
by sorry

end NUMINAMATH_CALUDE_final_number_of_boys_l3775_377540


namespace NUMINAMATH_CALUDE_correct_sqrt_product_incorrect_sqrt_sum_incorrect_sqrt_diff_incorrect_sqrt_div_only_sqrt_product_correct_l3775_377548

theorem correct_sqrt_product : ∀ a b : ℝ, a > 0 → b > 0 → (Real.sqrt a) * (Real.sqrt b) = Real.sqrt (a * b) := by sorry

theorem incorrect_sqrt_sum : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (Real.sqrt a) + (Real.sqrt b) ≠ Real.sqrt (a + b) := by sorry

theorem incorrect_sqrt_diff : ∃ a : ℝ, a > 0 ∧ 3 * (Real.sqrt a) - (Real.sqrt a) ≠ 3 := by sorry

theorem incorrect_sqrt_div : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (Real.sqrt a) / (Real.sqrt b) ≠ 2 := by sorry

theorem only_sqrt_product_correct :
  (∀ a b : ℝ, a > 0 → b > 0 → (Real.sqrt a) * (Real.sqrt b) = Real.sqrt (a * b)) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (Real.sqrt a) + (Real.sqrt b) ≠ Real.sqrt (a + b)) ∧
  (∃ a : ℝ, a > 0 ∧ 3 * (Real.sqrt a) - (Real.sqrt a) ≠ 3) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (Real.sqrt a) / (Real.sqrt b) ≠ 2) := by sorry

end NUMINAMATH_CALUDE_correct_sqrt_product_incorrect_sqrt_sum_incorrect_sqrt_diff_incorrect_sqrt_div_only_sqrt_product_correct_l3775_377548


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3775_377577

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 - 4*I) / (1 - 2*I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3775_377577


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3775_377558

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) :
  1 / x + 1 / y = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3775_377558


namespace NUMINAMATH_CALUDE_sum_due_calculation_l3775_377578

/-- Represents the relationship between banker's discount, true discount, and face value -/
def banker_discount_relation (banker_discount true_discount face_value : ℚ) : Prop :=
  banker_discount = true_discount + (true_discount * banker_discount) / face_value

/-- Proves that given a banker's discount of 576 and a true discount of 480, the sum due (face value) is 2880 -/
theorem sum_due_calculation (banker_discount true_discount : ℚ) 
  (h1 : banker_discount = 576)
  (h2 : true_discount = 480) :
  ∃ face_value : ℚ, face_value = 2880 ∧ banker_discount_relation banker_discount true_discount face_value :=
by
  sorry

end NUMINAMATH_CALUDE_sum_due_calculation_l3775_377578


namespace NUMINAMATH_CALUDE_josh_string_cheese_cost_l3775_377531

/-- The total cost of Josh's string cheese purchase, including tax and discount -/
def total_cost (pack1 pack2 pack3 : ℕ) (price_per_cheese : ℚ) (discount_rate tax_rate : ℚ) : ℚ :=
  let total_cheese := pack1 + pack2 + pack3
  let subtotal := (total_cheese : ℚ) * price_per_cheese
  let discounted_price := subtotal * (1 - discount_rate)
  let tax := discounted_price * tax_rate
  discounted_price + tax

/-- Theorem stating the total cost of Josh's purchase -/
theorem josh_string_cheese_cost :
  total_cost 18 22 24 (10 / 100) (5 / 100) (12 / 100) = (681 / 100) := by
  sorry

end NUMINAMATH_CALUDE_josh_string_cheese_cost_l3775_377531


namespace NUMINAMATH_CALUDE_sequence_general_term_l3775_377585

theorem sequence_general_term (p : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n^2 + p*n) →
  (∃ r, a 2 * r = a 5 ∧ a 5 * r = a 10) →
  ∃ k, ∀ n, a n = 2*n + k :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3775_377585


namespace NUMINAMATH_CALUDE_club_simplifier_probability_l3775_377549

def probability_more_wins_than_losses (num_matches : ℕ) 
  (prob_win prob_lose prob_tie : ℚ) : ℚ :=
  sorry

theorem club_simplifier_probability :
  probability_more_wins_than_losses 3 (1/2) (1/4) (1/4) = 25/64 :=
by sorry

end NUMINAMATH_CALUDE_club_simplifier_probability_l3775_377549


namespace NUMINAMATH_CALUDE_pairing_probability_l3775_377544

/-- Represents a student in the classroom -/
structure Student :=
  (name : String)

/-- The probability of a specific event occurring in a random pairing scenario -/
def probability_of_pairing (total_students : ℕ) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / (total_students - 1)

/-- The classroom setup -/
def classroom_setup : Prop :=
  ∃ (students : Finset Student) (margo irma jess kurt : Student),
    students.card = 50 ∧
    margo ∈ students ∧
    irma ∈ students ∧
    jess ∈ students ∧
    kurt ∈ students ∧
    margo ≠ irma ∧ margo ≠ jess ∧ margo ≠ kurt

theorem pairing_probability (h : classroom_setup) :
  probability_of_pairing 50 3 = 3 / 49 := by
  sorry

end NUMINAMATH_CALUDE_pairing_probability_l3775_377544


namespace NUMINAMATH_CALUDE_negative_nine_less_than_negative_sqrt_80_l3775_377575

theorem negative_nine_less_than_negative_sqrt_80 : -9 < -Real.sqrt 80 := by
  sorry

end NUMINAMATH_CALUDE_negative_nine_less_than_negative_sqrt_80_l3775_377575


namespace NUMINAMATH_CALUDE_lowest_fifth_score_for_target_average_l3775_377566

def number_of_tests : ℕ := 5
def max_score : ℕ := 100
def target_average : ℕ := 85

def first_three_scores : List ℕ := [76, 94, 87]
def fourth_score : ℕ := 92

def total_needed_score : ℕ := number_of_tests * target_average

theorem lowest_fifth_score_for_target_average :
  ∃ (fifth_score : ℕ),
    fifth_score = total_needed_score - (first_three_scores.sum + fourth_score) ∧
    fifth_score = 76 ∧
    (∀ (x : ℕ), x < fifth_score →
      (first_three_scores.sum + fourth_score + x) / number_of_tests < target_average) :=
by sorry

end NUMINAMATH_CALUDE_lowest_fifth_score_for_target_average_l3775_377566


namespace NUMINAMATH_CALUDE_odd_function_value_l3775_377502

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value (f : ℝ → ℝ) (h_odd : IsOdd f)
    (h_periodic : ∀ x, f (x + 4) = f x + f 2) (h_f_neg_one : f (-1) = -2) :
    f 2013 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l3775_377502


namespace NUMINAMATH_CALUDE_melody_reading_pages_l3775_377523

def english_pages : ℕ := 20
def science_pages : ℕ := 16
def civics_pages : ℕ := 8
def chinese_pages : ℕ := 12

def pages_to_read (total_pages : ℕ) : ℕ := total_pages / 4

theorem melody_reading_pages : 
  pages_to_read english_pages + 
  pages_to_read science_pages + 
  pages_to_read civics_pages + 
  pages_to_read chinese_pages = 14 := by
  sorry

end NUMINAMATH_CALUDE_melody_reading_pages_l3775_377523


namespace NUMINAMATH_CALUDE_a_plus_b_value_l3775_377595

theorem a_plus_b_value (a b : ℝ) (h1 : |a| = 3) (h2 : b^2 = 25) (h3 : a*b < 0) :
  a + b = 2 ∨ a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l3775_377595


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l3775_377580

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ := 4 * s.side

/-- Theorem: The perimeter of a large rectangle composed of three identical squares
    and three identical small rectangles is 52, given the conditions. -/
theorem large_rectangle_perimeter : 
  ∀ (s : Square) (r : Rectangle),
    s.perimeter = 24 →
    r.perimeter = 16 →
    (3 * s.side = r.length) →
    (s.side + r.width = 3 * r.length + 3 * r.width) →
    Rectangle.perimeter { length := 3 * s.side, width := s.side + r.width } = 52 := by
  sorry

end NUMINAMATH_CALUDE_large_rectangle_perimeter_l3775_377580


namespace NUMINAMATH_CALUDE_find_k_l3775_377572

-- Define the sets A and B
def A (k : ℝ) : Set ℝ := {x | 1 < x ∧ x < k}
def B (k : ℝ) : Set ℝ := {y | ∃ x ∈ A k, y = 2*x - 5}

-- Define the intersection set
def intersection_set : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem find_k (k : ℝ) : A k ∩ B k = intersection_set → k = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3775_377572


namespace NUMINAMATH_CALUDE_sin_510_degrees_l3775_377582

theorem sin_510_degrees : Real.sin (510 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_510_degrees_l3775_377582


namespace NUMINAMATH_CALUDE_exact_two_support_probability_l3775_377543

/-- The probability of a voter supporting the law -/
def p_support : ℝ := 0.6

/-- The probability of a voter not supporting the law -/
def p_oppose : ℝ := 1 - p_support

/-- The number of voters selected -/
def n : ℕ := 5

/-- The number of voters supporting the law in our target scenario -/
def k : ℕ := 2

/-- The binomial coefficient for choosing k items from n items -/
def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of exactly k out of n voters supporting the law -/
def prob_exact_support (n k : ℕ) (p : ℝ) : ℝ :=
  (binom_coeff n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exact_two_support_probability :
  prob_exact_support n k p_support = 0.2304 := by sorry

end NUMINAMATH_CALUDE_exact_two_support_probability_l3775_377543


namespace NUMINAMATH_CALUDE_assignment_ways_20_3_l3775_377522

/-- The number of ways to assign 3 distinct items from a set of 20 items -/
def assignmentWays (n : ℕ) (k : ℕ) : ℕ :=
  n * (n - 1) * (n - 2)

/-- Theorem: The number of ways to assign 3 distinct items from a set of 20 items is 6840 -/
theorem assignment_ways_20_3 :
  assignmentWays 20 3 = 6840 := by
  sorry

#eval assignmentWays 20 3

end NUMINAMATH_CALUDE_assignment_ways_20_3_l3775_377522


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3775_377591

-- Define the conditions
def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3775_377591


namespace NUMINAMATH_CALUDE_triangle_side_ratio_bound_l3775_377503

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  height_a : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  height_condition : height_a = a

-- Theorem statement
theorem triangle_side_ratio_bound (t : Triangle) : 
  2 ≤ (t.b / t.c + t.c / t.b) ∧ (t.b / t.c + t.c / t.b) ≤ Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_bound_l3775_377503


namespace NUMINAMATH_CALUDE_average_weight_increase_l3775_377517

/-- Given a group of 8 people, prove that replacing a person weighing 40 kg with a person weighing 60 kg increases the average weight by 2.5 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 8 * initial_average
  let new_total := initial_total - 40 + 60
  let new_average := new_total / 8
  new_average - initial_average = 2.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3775_377517


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3775_377588

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
   x1^2 - x1 + 2*m - 2 = 0 ∧ 
   x2^2 - x2 + 2*m - 2 = 0) 
  ↔ m ≤ 9/8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3775_377588


namespace NUMINAMATH_CALUDE_jimmy_has_more_sheets_l3775_377516

/-- Represents the number of sheets each person has -/
structure Sheets where
  jimmy : ℕ
  tommy : ℕ
  ashton : ℕ

/-- The initial state of sheet distribution -/
def initial : Sheets where
  jimmy := 58
  tommy := 58 + 25
  ashton := 85

/-- The state after Ashton gives sheets to Jimmy -/
def final (s : Sheets) : Sheets where
  jimmy := s.jimmy + s.ashton
  tommy := s.tommy
  ashton := 0

/-- Theorem stating that Jimmy will have 60 more sheets than Tommy after receiving sheets from Ashton -/
theorem jimmy_has_more_sheets : (final initial).jimmy - (final initial).tommy = 60 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_has_more_sheets_l3775_377516


namespace NUMINAMATH_CALUDE_solve_bodyguard_problem_l3775_377518

def bodyguard_problem (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (total_cost : ℕ) : Prop :=
  let daily_cost := num_bodyguards * hourly_rate * hours_per_day
  ∃ (days : ℕ), days * daily_cost = total_cost ∧ days = 7

theorem solve_bodyguard_problem :
  bodyguard_problem 2 20 8 2240 :=
sorry

end NUMINAMATH_CALUDE_solve_bodyguard_problem_l3775_377518


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3775_377555

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 = 2 → a 2 + a 5 = 13 → a 5 + a 6 + a 7 = 33 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3775_377555


namespace NUMINAMATH_CALUDE_angle_BCD_measure_l3775_377511

-- Define a pentagon
structure Pentagon :=
  (A B C D E : ℝ)

-- Define the conditions
def pentagon_conditions (p : Pentagon) : Prop :=
  p.A = 100 ∧ p.D = 120 ∧ p.E = 80 ∧ p.A + p.B + p.C + p.D + p.E = 540

-- Theorem statement
theorem angle_BCD_measure (p : Pentagon) :
  pentagon_conditions p → p.B = 140 → p.C = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_BCD_measure_l3775_377511


namespace NUMINAMATH_CALUDE_square_perimeter_from_p_shape_l3775_377515

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Represents the P shape formed by rectangles -/
structure PShape where
  rectangles : Fin 4 → Rectangle

theorem square_perimeter_from_p_shape 
  (s : Square) 
  (p : PShape) 
  (h1 : ∀ i, p.rectangles i = ⟨s.side / 5, 4 * s.side / 5⟩) 
  (h2 : (6 * (4 * s.side / 5) + 4 * (s.side / 5) : ℝ) = 56) :
  4 * s.side = 40 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_from_p_shape_l3775_377515


namespace NUMINAMATH_CALUDE_smallest_c_for_positive_quadratic_l3775_377530

theorem smallest_c_for_positive_quadratic : 
  ∃ c : ℤ, (∀ x : ℝ, x^2 + c*x + 15 > 0) ∧ 
  (∀ d : ℤ, d < c → ∃ x : ℝ, x^2 + d*x + 15 ≤ 0) ∧ 
  c = -7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_for_positive_quadratic_l3775_377530


namespace NUMINAMATH_CALUDE_sqrt_x_plus_five_equals_two_l3775_377545

theorem sqrt_x_plus_five_equals_two (x : ℝ) (h : x = -1) : Real.sqrt (x + 5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_five_equals_two_l3775_377545


namespace NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l3775_377562

theorem sin_negative_thirty_degrees : 
  Real.sin (-(30 * π / 180)) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_thirty_degrees_l3775_377562


namespace NUMINAMATH_CALUDE_fraction_operations_l3775_377521

theorem fraction_operations :
  let a := 2
  let b := 9
  let c := 5
  let d := 11
  (a / b) * (c / d) = 10 / 99 ∧ (a / b) + (c / d) = 67 / 99 := by
  sorry

end NUMINAMATH_CALUDE_fraction_operations_l3775_377521


namespace NUMINAMATH_CALUDE_range_of_m_l3775_377514

theorem range_of_m (x y m : ℝ) 
  (hx : 1 < x ∧ x < 3) 
  (hy : -3 < y ∧ y < 1) 
  (hm : m = x - 3*y) : 
  -2 < m ∧ m < 12 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3775_377514


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3775_377554

/-- Two points are symmetric about the x-axis if their x-coordinates are equal and 
    their y-coordinates are equal in magnitude but opposite in sign -/
def symmetric_about_x_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = x₂ ∧ y₁ = -y₂

/-- Given two points A(2,m) and B(n,-3) that are symmetric about the x-axis,
    prove that m + n = 5 -/
theorem symmetric_points_sum (m n : ℝ) 
  (h : symmetric_about_x_axis 2 m n (-3)) : m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3775_377554


namespace NUMINAMATH_CALUDE_fraction_value_at_three_l3775_377556

theorem fraction_value_at_three :
  let x : ℝ := 3
  (x^8 + 16*x^4 + 64) / (x^4 + 8) = 89 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_at_three_l3775_377556


namespace NUMINAMATH_CALUDE_fencing_required_l3775_377536

/-- Calculates the fencing required for a rectangular field with one side uncovered -/
theorem fencing_required (length width area : ℝ) (h1 : length = 34) (h2 : area = 680) 
  (h3 : area = length * width) : 2 * width + length = 74 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l3775_377536


namespace NUMINAMATH_CALUDE_circle_radius_l3775_377568

/-- The radius of a circle given by the equation x^2 + 10x + y^2 - 8y + 25 = 0 is 4 -/
theorem circle_radius (x y : ℝ) : x^2 + 10*x + y^2 - 8*y + 25 = 0 → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 4^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3775_377568


namespace NUMINAMATH_CALUDE_positive_integer_divisible_by_15_with_sqrt_between_33_and_33_5_l3775_377512

theorem positive_integer_divisible_by_15_with_sqrt_between_33_and_33_5 :
  ∃ n : ℕ+, 
    (∃ k : ℕ, n = 15 * k) ∧ 
    (33 * 33 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) < (33.5 * 33.5) ∧
    (n = 1095 ∨ n = 1110) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_divisible_by_15_with_sqrt_between_33_and_33_5_l3775_377512


namespace NUMINAMATH_CALUDE_recipe_salt_amount_l3775_377592

def recipe_salt (total_flour sugar flour_added : ℕ) : ℕ :=
  let remaining_flour := total_flour - flour_added
  remaining_flour - 3

theorem recipe_salt_amount :
  recipe_salt 12 14 2 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_recipe_salt_amount_l3775_377592


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l3775_377524

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 11*x - 42 = 0 → 
  (∃ y : ℝ, y ≠ x ∧ y^2 - 11*y - 42 = 0) → 
  (x ≤ 14 ∧ (∀ z : ℝ, z^2 - 11*z - 42 = 0 → z ≤ 14)) :=
sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l3775_377524


namespace NUMINAMATH_CALUDE_find_A_value_l3775_377590

theorem find_A_value : ∃! A : ℕ, ∃ B : ℕ, 
  (A < 10 ∧ B < 10) ∧ 
  (500 + 10 * A + 8) - (100 * B + 14) = 364 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_find_A_value_l3775_377590


namespace NUMINAMATH_CALUDE_complement_of_A_l3775_377510

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | x < -1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3775_377510
