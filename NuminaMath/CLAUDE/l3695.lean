import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_problem_l3695_369579

theorem complex_number_problem (z₁ z₂ z : ℂ) : 
  z₁ = 1 - 2*I →
  z₂ = 4 + 3*I →
  Complex.abs z = 2 →
  Complex.im z = Complex.re (3*z₁ - z₂) →
  Complex.re z < 0 ∧ Complex.im z < 0 →
  z = -Real.sqrt 2 - I * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3695_369579


namespace NUMINAMATH_CALUDE_leahs_coins_value_l3695_369578

/-- Represents the value of a coin in cents -/
def coinValue (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | _ => 0

/-- Calculates the total value of coins given their quantities -/
def totalValue (pennies nickels dimes : ℕ) : ℕ :=
  pennies * coinValue "penny" + nickels * coinValue "nickel" + dimes * coinValue "dime"

theorem leahs_coins_value :
  ∀ (pennies nickels dimes : ℕ),
    pennies + nickels + dimes = 17 →
    nickels + 2 = pennies →
    totalValue pennies nickels dimes = 68 :=
by sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l3695_369578


namespace NUMINAMATH_CALUDE_complex_distance_l3695_369527

theorem complex_distance (z : ℂ) : z = 1 - 2*I → Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_distance_l3695_369527


namespace NUMINAMATH_CALUDE_min_sum_of_distances_l3695_369564

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab_eq_ac : dist A B = dist A C)
  (ab_eq_5 : dist A B = 5)
  (bc_eq_6 : dist B C = 6)

/-- Point on the sides of the triangle -/
def PointOnSides (t : Triangle) : Set (ℝ × ℝ) :=
  {P | ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧
    (P = (1 - s) • t.A + s • t.B ∨
     P = (1 - s) • t.B + s • t.C ∨
     P = (1 - s) • t.C + s • t.A)}

/-- Sum of distances from P to vertices -/
def SumOfDistances (t : Triangle) (P : ℝ × ℝ) : ℝ :=
  dist P t.A + dist P t.B + dist P t.C

/-- Theorem: Minimum sum of distances is 16 -/
theorem min_sum_of_distances (t : Triangle) :
  ∀ P ∈ PointOnSides t, SumOfDistances t P ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_distances_l3695_369564


namespace NUMINAMATH_CALUDE_marked_nodes_on_circle_l3695_369588

/-- Represents a node in the hexagon grid -/
structure Node where
  x : ℤ
  y : ℤ

/-- Represents a circle in the hexagon grid -/
structure Circle where
  center : Node
  radius : ℕ

/-- The side length of the regular hexagon -/
def hexagon_side_length : ℕ := 5

/-- The side length of the equilateral triangles -/
def triangle_side_length : ℕ := 1

/-- The total number of nodes in the hexagon -/
def total_nodes : ℕ := 91

/-- A function that determines if a node is marked -/
def is_marked : Node → Prop := sorry

/-- A function that determines if a node lies on a given circle -/
def on_circle : Node → Circle → Prop := sorry

/-- The main theorem to be proved -/
theorem marked_nodes_on_circle :
  (∃ (marked_nodes : Finset Node), 
    (∀ n ∈ marked_nodes, is_marked n) ∧ 
    (marked_nodes.card > total_nodes / 2)) →
  (∃ (c : Circle) (five_nodes : Finset Node),
    five_nodes.card = 5 ∧
    (∀ n ∈ five_nodes, is_marked n ∧ on_circle n c)) :=
by sorry

end NUMINAMATH_CALUDE_marked_nodes_on_circle_l3695_369588


namespace NUMINAMATH_CALUDE_second_plan_fee_calculation_l3695_369541

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

end NUMINAMATH_CALUDE_second_plan_fee_calculation_l3695_369541


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3695_369562

theorem sqrt_equation_solution : ∃ x : ℝ, x = 2401 / 100 ∧ Real.sqrt x + Real.sqrt (x + 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3695_369562


namespace NUMINAMATH_CALUDE_olympic_quiz_probability_l3695_369560

theorem olympic_quiz_probability (A B C : ℝ) 
  (hA : A = 3/4)
  (hAC : (1 - A) * (1 - C) = 1/12)
  (hBC : B * C = 1/4) :
  A * B * (1 - C) + A * (1 - B) * C + (1 - A) * B * C = 15/32 := by
  sorry

end NUMINAMATH_CALUDE_olympic_quiz_probability_l3695_369560


namespace NUMINAMATH_CALUDE_max_sum_xy_l3695_369577

theorem max_sum_xy (x y a b : ℝ) (hx : x > 0) (hy : y > 0)
  (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ y)
  (h1 : a^2 + y^2 = 2) (h2 : b^2 + x^2 = 1) (h3 : a*x + b*y = 1) :
  x + y ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 2 ∧
    ∃ (a₀ b₀ : ℝ), 0 ≤ a₀ ∧ a₀ ≤ x₀ ∧ 0 ≤ b₀ ∧ b₀ ≤ y₀ ∧
      a₀^2 + y₀^2 = 2 ∧ b₀^2 + x₀^2 = 1 ∧ a₀*x₀ + b₀*y₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_xy_l3695_369577


namespace NUMINAMATH_CALUDE_unique_prime_triple_l3695_369590

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 0 → d ∣ n → d = 1 ∨ d = n

theorem unique_prime_triple :
  ∃! (p q r : ℕ), isPrime p ∧ isPrime q ∧ isPrime r ∧ p = q + 2 ∧ q = r + 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l3695_369590


namespace NUMINAMATH_CALUDE_tank_capacity_l3695_369550

theorem tank_capacity : 
  ∀ (initial_fraction final_fraction added_water : ℚ),
  initial_fraction = 1/8 →
  final_fraction = 2/3 →
  added_water = 150 →
  ∃ (total_capacity : ℚ),
  (final_fraction - initial_fraction) * total_capacity = added_water ∧
  total_capacity = 3600/13 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l3695_369550


namespace NUMINAMATH_CALUDE_train_speed_l3695_369518

/-- The average speed of a train without stoppages, given its speed with stoppages and stop time -/
theorem train_speed (speed_with_stops : ℝ) (stop_time : ℝ) : 
  speed_with_stops = 200 → stop_time = 20 → 
  (speed_with_stops * 60) / (60 - stop_time) = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3695_369518


namespace NUMINAMATH_CALUDE_check_error_l3695_369593

theorem check_error (x y : ℤ) 
  (h1 : 10 ≤ x ∧ x ≤ 99) 
  (h2 : 10 ≤ y ∧ y ≤ 99) 
  (h3 : 100 * y + x - (100 * x + y) = 2046) 
  (h4 : x = (3 * y) / 2) : 
  x = 66 := by sorry

end NUMINAMATH_CALUDE_check_error_l3695_369593


namespace NUMINAMATH_CALUDE_youngest_child_age_l3695_369532

def is_valid_age (x : ℕ) : Prop :=
  Nat.Prime x ∧
  Nat.Prime (x + 2) ∧
  Nat.Prime (x + 6) ∧
  Nat.Prime (x + 8) ∧
  Nat.Prime (x + 12) ∧
  Nat.Prime (x + 14)

theorem youngest_child_age :
  ∃ (x : ℕ), is_valid_age x ∧ ∀ (y : ℕ), y < x → ¬is_valid_age y :=
by sorry

end NUMINAMATH_CALUDE_youngest_child_age_l3695_369532


namespace NUMINAMATH_CALUDE_factorial_10_mod_13_l3695_369545

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_10_mod_13 : factorial 10 % 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_10_mod_13_l3695_369545


namespace NUMINAMATH_CALUDE_sara_purse_value_l3695_369509

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes quarters : ℕ) : ℕ :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  dimes * coin_value "dime" +
  quarters * coin_value "quarter"

/-- Converts a number of cents to a percentage of a dollar -/
def cents_to_percentage (cents : ℕ) : ℚ :=
  (cents : ℚ) / 100

theorem sara_purse_value :
  cents_to_percentage (total_value 3 2 1 2) = 73 / 100 := by
  sorry

end NUMINAMATH_CALUDE_sara_purse_value_l3695_369509


namespace NUMINAMATH_CALUDE_no_solution_trigonometric_equation_l3695_369500

open Real

theorem no_solution_trigonometric_equation (m : ℝ) :
  ¬ ∃ x : ℝ, (sin (3 * x) * cos (π / 3 - x) + 1) / (sin (π / 3 - 7 * x) - cos (π / 6 + x) + m) = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_trigonometric_equation_l3695_369500


namespace NUMINAMATH_CALUDE_optimal_garden_dimensions_l3695_369594

/-- Represents the dimensions of a rectangular garden --/
structure GardenDimensions where
  perpendicular_side : ℝ
  parallel_side : ℝ

/-- Calculates the area of the garden given its dimensions --/
def garden_area (d : GardenDimensions) : ℝ :=
  d.perpendicular_side * d.parallel_side

/-- Represents the constraints of the garden problem --/
structure GardenConstraints where
  wall_length : ℝ
  fence_cost_per_foot : ℝ
  total_fence_cost : ℝ

/-- Theorem stating that the optimal garden dimensions maximize the area --/
theorem optimal_garden_dimensions (c : GardenConstraints)
  (h1 : c.wall_length = 300)
  (h2 : c.fence_cost_per_foot = 10)
  (h3 : c.total_fence_cost = 1500) :
  ∃ (d : GardenDimensions),
    d.parallel_side = 75 ∧
    ∀ (d' : GardenDimensions),
      d'.perpendicular_side + d'.perpendicular_side + d'.parallel_side = c.total_fence_cost / c.fence_cost_per_foot →
      garden_area d ≥ garden_area d' :=
sorry

end NUMINAMATH_CALUDE_optimal_garden_dimensions_l3695_369594


namespace NUMINAMATH_CALUDE_fifty_square_divisible_by_one_by_four_strips_l3695_369502

/-- Represents a rectangular strip --/
structure Strip where
  width : ℕ
  length : ℕ

/-- Represents a square --/
structure Square where
  side : ℕ

/-- Checks if a square can be divided into strips --/
def isDivisible (s : Square) (strip : Strip) : Prop :=
  s.side * s.side % (strip.width * strip.length) = 0

/-- Theorem: A 50x50 square can be divided into 1x4 strips --/
theorem fifty_square_divisible_by_one_by_four_strips :
  isDivisible (Square.mk 50) (Strip.mk 1 4) := by
  sorry

end NUMINAMATH_CALUDE_fifty_square_divisible_by_one_by_four_strips_l3695_369502


namespace NUMINAMATH_CALUDE_hardey_fitness_center_ratio_l3695_369501

theorem hardey_fitness_center_ratio :
  ∀ (f m : ℕ) (f_avg m_avg total_avg : ℝ),
  f_avg = 55 →
  m_avg = 80 →
  total_avg = 70 →
  (f_avg * f + m_avg * m) / (f + m) = total_avg →
  (f : ℝ) / m = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hardey_fitness_center_ratio_l3695_369501


namespace NUMINAMATH_CALUDE_hyperbola_center_l3695_369599

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 81 * x - 16 * y^2 + 64 * y + 144 = 0

-- Define the center of a hyperbola
def is_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), eq x y ↔ eq (x - c.1) (y - c.2)

-- Theorem statement
theorem hyperbola_center :
  is_center (9/2, 2) hyperbola_eq :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l3695_369599


namespace NUMINAMATH_CALUDE_cannot_construct_configuration_l3695_369552

/-- Represents a rhombus figure with two colors -/
structure ColoredRhombus where
  white_part : Set (ℝ × ℝ)
  gray_part : Set (ℝ × ℝ)
  is_rhombus : white_part ∪ gray_part = unit_rhombus
  no_overlap : white_part ∩ gray_part = ∅

/-- Represents a configuration of multiple rhombuses -/
def Configuration := Set (ColoredRhombus × (ℝ × ℝ))

/-- Rotates a point around the origin -/
def rotate (θ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Translates a point -/
def translate (v : ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Applies rotation and translation to a ColoredRhombus -/
def transform (r : ColoredRhombus) (θ : ℝ) (v : ℝ × ℝ) : ColoredRhombus := sorry

/-- Checks if a configuration can be constructed from a given rhombus -/
def is_constructible (r : ColoredRhombus) (c : Configuration) : Prop := sorry

/-- The specific configuration that we claim is impossible to construct -/
def impossible_configuration : Configuration := sorry

/-- The main theorem stating that the impossible configuration cannot be constructed -/
theorem cannot_construct_configuration (r : ColoredRhombus) : 
  ¬(is_constructible r impossible_configuration) := by sorry

end NUMINAMATH_CALUDE_cannot_construct_configuration_l3695_369552


namespace NUMINAMATH_CALUDE_white_spotted_mushrooms_count_l3695_369597

/-- The number of white-spotted mushrooms gathered by Bill and Ted -/
def white_spotted_mushrooms : ℕ :=
  let bill_red := 12
  let bill_brown := 6
  let ted_blue := 6
  let red_with_spots := (2 * bill_red) / 3
  let brown_with_spots := bill_brown
  let blue_with_spots := ted_blue / 2
  red_with_spots + brown_with_spots + blue_with_spots

/-- Theorem stating that the total number of white-spotted mushrooms is 17 -/
theorem white_spotted_mushrooms_count : white_spotted_mushrooms = 17 := by
  sorry

end NUMINAMATH_CALUDE_white_spotted_mushrooms_count_l3695_369597


namespace NUMINAMATH_CALUDE_book_shelf_problem_l3695_369566

theorem book_shelf_problem (paperbacks hardbacks : ℕ) 
  (h1 : paperbacks = 2)
  (h2 : hardbacks = 6)
  (h3 : Nat.choose paperbacks 1 * Nat.choose hardbacks 2 + 
        Nat.choose paperbacks 2 * Nat.choose hardbacks 1 = 36) :
  paperbacks + hardbacks = 8 := by
sorry

end NUMINAMATH_CALUDE_book_shelf_problem_l3695_369566


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_relation_l3695_369553

theorem quadratic_equation_roots_relation (p : ℚ) : 
  (∃ x1 x2 : ℚ, 3 * x1^2 - 5*(p-1)*x1 + p^2 + 2 = 0 ∧
                3 * x2^2 - 5*(p-1)*x2 + p^2 + 2 = 0 ∧
                x1 + 4*x2 = 14) ↔ 
  (p = 742/127 ∨ p = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_relation_l3695_369553


namespace NUMINAMATH_CALUDE_star_calculation_l3695_369569

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 - 2*y

-- State the theorem
theorem star_calculation :
  let a := star 5 14
  let b := star 4 6
  star (2^a) (4^b) = -512.421875 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l3695_369569


namespace NUMINAMATH_CALUDE_trajectory_equation_l3695_369592

-- Define the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- Define the distance difference condition
def distance_difference : ℝ := 4

-- Define the trajectory of point C
def trajectory_of_C (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1 ∧ x ≥ 2

-- State the theorem
theorem trajectory_equation :
  ∀ (C : ℝ × ℝ),
    (Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) -
     Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = distance_difference) →
    trajectory_of_C C.1 C.2 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l3695_369592


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3695_369528

def A : Set ℝ := {x | x^2 - 2*x = 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3695_369528


namespace NUMINAMATH_CALUDE_cow_count_is_16_l3695_369555

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

end NUMINAMATH_CALUDE_cow_count_is_16_l3695_369555


namespace NUMINAMATH_CALUDE_baseball_team_groups_l3695_369505

theorem baseball_team_groups (new_players returning_players players_per_group : ℕ) 
  (h1 : new_players = 48)
  (h2 : returning_players = 6)
  (h3 : players_per_group = 6) :
  (new_players + returning_players) / players_per_group = 9 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_groups_l3695_369505


namespace NUMINAMATH_CALUDE_division_problem_l3695_369513

theorem division_problem (x : ℝ) (h : 0.009 / x = 0.03) : x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3695_369513


namespace NUMINAMATH_CALUDE_circle_equation_m_range_l3695_369598

theorem circle_equation_m_range (m : ℝ) :
  (∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) →
  m < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_m_range_l3695_369598


namespace NUMINAMATH_CALUDE_equal_distance_trajectory_length_l3695_369554

/-- Rectilinear distance between two points -/
def rectilinearDistance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

/-- The set of points C(x, y) with equal rectilinear distance to A and B -/
def equalDistancePoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               0 ≤ x ∧ x ≤ 10 ∧ 0 ≤ y ∧ y ≤ 10 ∧
               rectilinearDistance x y 1 3 = rectilinearDistance x y 6 9}

/-- The sum of the lengths of the trajectories of all points in equalDistancePoints -/
noncomputable def trajectoryLength : ℝ :=
  5 * (Real.sqrt 2 + 1)

theorem equal_distance_trajectory_length :
  trajectoryLength = 5 * (Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_equal_distance_trajectory_length_l3695_369554


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l3695_369522

theorem smallest_n_for_inequality : ∀ n : ℕ, n ≥ 5 → 2^n > n^2 ∧ ∀ k : ℕ, k < 5 → 2^k ≤ k^2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l3695_369522


namespace NUMINAMATH_CALUDE_cubic_equation_result_l3695_369568

theorem cubic_equation_result (a : ℝ) (h : a^3 + 2*a = -2) :
  3*a^6 + 12*a^4 - a^3 + 12*a^2 - 2*a - 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_result_l3695_369568


namespace NUMINAMATH_CALUDE_S_is_infinite_l3695_369547

-- Define the set of points satisfying the conditions
def S : Set (ℚ × ℚ) :=
  {p : ℚ × ℚ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + 2 * p.2 ≤ 10}

-- Theorem stating that the set S is infinite
theorem S_is_infinite : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_S_is_infinite_l3695_369547


namespace NUMINAMATH_CALUDE_max_value_theorem_l3695_369558

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  2 * a * c * Real.sqrt 2 + 2 * a * b ≤ Real.sqrt 3 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ 
  a'^2 + b'^2 + c'^2 = 1 ∧ 
  2 * a' * c' * Real.sqrt 2 + 2 * a' * b' = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3695_369558


namespace NUMINAMATH_CALUDE_unique_solution_l3695_369508

/-- A function y is a solution to the differential equation y' - y = cos x - sin x
    and is bounded as x approaches positive infinity -/
def IsSolution (y : ℝ → ℝ) : Prop :=
  (∀ x, (deriv y x) - y x = Real.cos x - Real.sin x) ∧
  (∃ M, ∀ x, x ≥ 0 → |y x| ≤ M)

/-- The unique solution to the differential equation y' - y = cos x - sin x
    that is bounded as x approaches positive infinity is y = - cos x -/
theorem unique_solution :
  ∃! y, IsSolution y ∧ (∀ x, y x = - Real.cos x) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3695_369508


namespace NUMINAMATH_CALUDE_quadratic_roots_integrality_l3695_369525

theorem quadratic_roots_integrality (q : ℤ) :
  (q > 0 → ∃ (p : ℤ), ∃ (x₁ x₂ x₃ x₄ : ℤ),
    x₁^2 - p*x₁ + q = 0 ∧
    x₂^2 - p*x₂ + q = 0 ∧
    x₃^2 - (p+1)*x₃ + q = 0 ∧
    x₄^2 - (p+1)*x₄ + q = 0) ∧
  (q < 0 → ¬∃ (p : ℤ), ∃ (x₁ x₂ x₃ x₄ : ℤ),
    x₁^2 - p*x₁ + q = 0 ∧
    x₂^2 - p*x₂ + q = 0 ∧
    x₃^2 - (p+1)*x₃ + q = 0 ∧
    x₄^2 - (p+1)*x₄ + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_integrality_l3695_369525


namespace NUMINAMATH_CALUDE_positive_A_value_l3695_369582

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + 3*B^2

-- Theorem statement
theorem positive_A_value :
  ∃ A : ℝ, A > 0 ∧ hash A 6 = 270 ∧ A = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l3695_369582


namespace NUMINAMATH_CALUDE_max_draws_until_white_l3695_369534

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : Nat
  white : Nat

/-- Represents the process of drawing balls from the bag -/
def drawUntilWhite (bag : BagContents) : Nat :=
  sorry

/-- Theorem stating the maximum number of draws needed -/
theorem max_draws_until_white (bag : BagContents) 
  (h1 : bag.red = 6) 
  (h2 : bag.white = 5) : 
  drawUntilWhite bag ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_draws_until_white_l3695_369534


namespace NUMINAMATH_CALUDE_third_week_vegetable_intake_l3695_369587

/-- Represents the daily vegetable intake in pounds -/
structure DailyIntake where
  asparagus : ℝ
  broccoli : ℝ
  cauliflower : ℝ
  spinach : ℝ
  kale : ℝ
  zucchini : ℝ

/-- Calculates the total daily intake -/
def totalDailyIntake (intake : DailyIntake) : ℝ :=
  intake.asparagus + intake.broccoli + intake.cauliflower + intake.spinach + intake.kale + intake.zucchini

/-- Initial daily intake -/
def initialIntake : DailyIntake :=
  { asparagus := 0.25, broccoli := 0.25, cauliflower := 0.5, spinach := 0, kale := 0, zucchini := 0 }

/-- Daily intake after second week changes -/
def secondWeekIntake : DailyIntake :=
  { asparagus := initialIntake.asparagus * 2,
    broccoli := initialIntake.broccoli * 3,
    cauliflower := initialIntake.cauliflower * 1.75,
    spinach := 0.5,
    kale := 0,
    zucchini := 0 }

/-- Daily intake in the third week -/
def thirdWeekIntake : DailyIntake :=
  { asparagus := secondWeekIntake.asparagus,
    broccoli := secondWeekIntake.broccoli,
    cauliflower := secondWeekIntake.cauliflower,
    spinach := secondWeekIntake.spinach,
    kale := 0.5,  -- 1 pound every two days
    zucchini := 0.15 }  -- 0.3 pounds every two days

theorem third_week_vegetable_intake :
  totalDailyIntake thirdWeekIntake * 7 = 22.925 := by
  sorry

end NUMINAMATH_CALUDE_third_week_vegetable_intake_l3695_369587


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3695_369586

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 4 → b = 4 → c = 2 →
  (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  a + b > c ∧ b + c > a ∧ a + c > b →  -- triangle inequality
  a + b + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3695_369586


namespace NUMINAMATH_CALUDE_nine_pointed_star_sum_tip_angles_l3695_369585

/-- A 9-pointed star formed by connecting nine evenly spaced points on a circle -/
structure NinePointedStar where
  /-- The measure of the angle at each tip of the star -/
  tip_angle : ℝ
  /-- The number of points on the circle -/
  num_points : ℕ
  /-- The points are evenly spaced on the circle -/
  evenly_spaced : num_points = 9
  /-- The measure of the arc between two consecutive points -/
  arc_measure : ℝ
  /-- The arc measure is 360° divided by the number of points -/
  arc_measure_def : arc_measure = 360 / num_points
  /-- Each tip angle subtends an arc that spans 3 consecutive points -/
  tip_angle_subtends_three_arcs : tip_angle = 3 * arc_measure / 2

/-- The sum of the measures of all tip angles in a 9-pointed star is 540° -/
theorem nine_pointed_star_sum_tip_angles (star : NinePointedStar) :
  star.num_points * star.tip_angle = 540 := by
  sorry

end NUMINAMATH_CALUDE_nine_pointed_star_sum_tip_angles_l3695_369585


namespace NUMINAMATH_CALUDE_light_bulb_survey_not_appropriate_l3695_369584

-- Define the types of surveys
inductive SurveyMethod
| Sampling
| Comprehensive

-- Define the characteristics of a survey subject
structure SurveySubject where
  population_size : Nat
  requires_destruction : Bool

-- Define when a survey method is appropriate
def is_appropriate (method : SurveyMethod) (subject : SurveySubject) : Prop :=
  match method with
  | SurveyMethod.Sampling => subject.population_size > 100 ∨ subject.requires_destruction
  | SurveyMethod.Comprehensive => subject.population_size ≤ 100 ∧ ¬subject.requires_destruction

-- Theorem statement
theorem light_bulb_survey_not_appropriate :
  let light_bulbs : SurveySubject := ⟨1000, true⟩
  ¬(is_appropriate SurveyMethod.Comprehensive light_bulbs) :=
by sorry

end NUMINAMATH_CALUDE_light_bulb_survey_not_appropriate_l3695_369584


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3695_369530

theorem inequality_solution_set (x : ℝ) : -x + 1 > 7*x - 3 ↔ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3695_369530


namespace NUMINAMATH_CALUDE_function_symmetry_l3695_369512

/-- For any function f(x) = x^5 - ax^3 + bx + 2, f(x) + f(-x) = 4 for all real x -/
theorem function_symmetry (a b : ℝ) :
  let f := fun (x : ℝ) => x^5 - a*x^3 + b*x + 2
  ∀ x, f x + f (-x) = 4 := by sorry

end NUMINAMATH_CALUDE_function_symmetry_l3695_369512


namespace NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l3695_369596

theorem linear_coefficient_of_quadratic (m : ℝ) : 
  m^2 - 2*m - 1 = 2 → 
  m - 3 ≠ 0 → 
  ∃ a b c, (m - 3)*x + 4*m^2 - 2*m - 1 - m*x + 6 = a*x^2 + b*x + c ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l3695_369596


namespace NUMINAMATH_CALUDE_find_y_when_x_is_12_l3695_369565

-- Define the inverse proportionality constant
def k : ℝ := 675

-- Define the relationship between x and y
def inverse_proportional (x y : ℝ) : Prop := x * y = k

-- State the theorem
theorem find_y_when_x_is_12 (x y : ℝ) 
  (h1 : inverse_proportional x y) 
  (h2 : x + y = 60) 
  (h3 : x = 3 * y) :
  x = 12 → y = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_find_y_when_x_is_12_l3695_369565


namespace NUMINAMATH_CALUDE_ninth_square_difference_l3695_369574

/-- The number of tiles in the n-th square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := (2 * n) ^ 2

/-- The difference in tiles between the n-th and (n-1)-th squares -/
def tile_difference (n : ℕ) : ℕ := tiles_in_square n - tiles_in_square (n - 1)

theorem ninth_square_difference : tile_difference 9 = 68 := by
  sorry

end NUMINAMATH_CALUDE_ninth_square_difference_l3695_369574


namespace NUMINAMATH_CALUDE_roller_coaster_friends_l3695_369524

theorem roller_coaster_friends (tickets_per_ride : ℕ) (total_tickets : ℕ) (num_friends : ℕ) : 
  tickets_per_ride = 6 → total_tickets = 48 → num_friends * tickets_per_ride = total_tickets → num_friends = 8 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_friends_l3695_369524


namespace NUMINAMATH_CALUDE_largest_difference_l3695_369572

def A : ℕ := 3 * 2023^2024
def B : ℕ := 2023^2024
def C : ℕ := 2022 * 2023^2023
def D : ℕ := 3 * 2023^2023
def E : ℕ := 2023^2023
def F : ℕ := 2023^2022

theorem largest_difference : 
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) := by
  sorry

end NUMINAMATH_CALUDE_largest_difference_l3695_369572


namespace NUMINAMATH_CALUDE_triangle_count_l3695_369511

/-- Calculates the total number of triangles in a triangular figure composed of n rows of small isosceles triangles. -/
def totalTriangles (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The number of rows in our specific triangular figure -/
def numRows : ℕ := 7

/-- Theorem stating that the total number of triangles in our specific figure is 28 -/
theorem triangle_count : totalTriangles numRows = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l3695_369511


namespace NUMINAMATH_CALUDE_value_of_y_minus_x_l3695_369583

theorem value_of_y_minus_x (x y : ℚ) 
  (h1 : x + y = 8) 
  (h2 : y - 3 * x = 7) : 
  y - x = 7.5 := by
sorry

end NUMINAMATH_CALUDE_value_of_y_minus_x_l3695_369583


namespace NUMINAMATH_CALUDE_area_equals_perimeter_count_l3695_369571

/-- A structure representing a rectangle with integer sides -/
structure Rectangle where
  a : ℕ
  b : ℕ

/-- A structure representing a right triangle with integer sides -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The area of a rectangle is equal to its perimeter -/
def Rectangle.areaEqualsPerimeter (r : Rectangle) : Prop :=
  r.a * r.b = 2 * (r.a + r.b)

/-- The area of a right triangle is equal to its perimeter -/
def RightTriangle.areaEqualsPerimeter (t : RightTriangle) : Prop :=
  t.a * t.b = 2 * (t.a + t.b + t.c)

/-- The sides of a right triangle satisfy the Pythagorean theorem -/
def RightTriangle.isPythagorean (t : RightTriangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

/-- The main theorem stating the number of rectangles and right triangles that satisfy the conditions -/
theorem area_equals_perimeter_count :
  (∃! (rs : Finset Rectangle), ∀ r ∈ rs, r.areaEqualsPerimeter ∧ rs.card = 2) ∧
  (∃! (ts : Finset RightTriangle), ∀ t ∈ ts, t.areaEqualsPerimeter ∧ t.isPythagorean ∧ ts.card = 1) := by
  sorry


end NUMINAMATH_CALUDE_area_equals_perimeter_count_l3695_369571


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l3695_369573

/-- The number of second-year students in the chess tournament --/
def n : ℕ := 7

/-- The total number of participants in the tournament --/
def total_participants : ℕ := n + 2

/-- The total number of games played in the tournament --/
def total_games : ℕ := (total_participants * (total_participants - 1)) / 2

/-- The total points scored in the tournament --/
def total_points : ℕ := total_games

/-- The points scored by the two freshmen --/
def freshman_points : ℕ := 8

/-- The points scored by all second-year students --/
def secondyear_points : ℕ := total_points - freshman_points

/-- The points scored by each second-year student --/
def points_per_secondyear : ℕ := secondyear_points / n

theorem chess_tournament_participants :
  n > 0 ∧
  total_participants = n + 2 ∧
  total_games = (total_participants * (total_participants - 1)) / 2 ∧
  total_points = total_games ∧
  freshman_points = 8 ∧
  secondyear_points = total_points - freshman_points ∧
  points_per_secondyear = secondyear_points / n ∧
  points_per_secondyear * n = secondyear_points ∧
  (∀ m : ℕ, m ≠ n → (m > 0 → 
    (m + 2) * (m + 1) / 2 - 8 ≠ ((m + 2) * (m + 1) / 2 - 8) / m * m)) :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l3695_369573


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3695_369516

/-- Two numbers are inversely proportional if their product is constant -/
def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) :
  inversely_proportional x y →
  (∃ x₀ y₀ : ℝ, x₀ + y₀ = 60 ∧ x₀ = 3 * y₀ ∧ inversely_proportional x₀ y₀) →
  (x = -10 → y = -67.5) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3695_369516


namespace NUMINAMATH_CALUDE_no_delightful_eight_digit_integers_l3695_369559

/-- Represents an 8-digit positive integer as a list of its digits -/
def EightDigitInteger := List Nat

/-- Checks if a list of digits forms a valid 8-digit integer -/
def isValid (n : EightDigitInteger) : Prop :=
  n.length = 8 ∧ n.toFinset = Finset.range 9 \ {0}

/-- Checks if the sum of the first k digits is divisible by k for all k from 1 to 8 -/
def isDelightful (n : EightDigitInteger) : Prop :=
  ∀ k : Nat, k ∈ Finset.range 9 \ {0} → (n.take k).sum % k = 0

/-- The main theorem: there are no delightful 8-digit integers -/
theorem no_delightful_eight_digit_integers :
  ¬∃ n : EightDigitInteger, isValid n ∧ isDelightful n := by
  sorry

end NUMINAMATH_CALUDE_no_delightful_eight_digit_integers_l3695_369559


namespace NUMINAMATH_CALUDE_yaras_ship_speed_l3695_369580

/-- Prove that Yara's ship speed is 30 nautical miles per hour -/
theorem yaras_ship_speed (theons_speed : ℝ) (distance : ℝ) (time_difference : ℝ) :
  theons_speed = 15 →
  distance = 90 →
  time_difference = 3 →
  distance / (distance / theons_speed - time_difference) = 30 :=
by sorry

end NUMINAMATH_CALUDE_yaras_ship_speed_l3695_369580


namespace NUMINAMATH_CALUDE_waiter_customers_l3695_369519

theorem waiter_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) :
  num_tables = 6 →
  women_per_table = 3 →
  men_per_table = 5 →
  num_tables * (women_per_table + men_per_table) = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l3695_369519


namespace NUMINAMATH_CALUDE_base_conversion_1729_l3695_369537

theorem base_conversion_1729 :
  (2 * 9^3 + 3 * 9^2 + 3 * 9^1 + 1 * 9^0) = 1729 := by
  sorry

#eval 2 * 9^3 + 3 * 9^2 + 3 * 9^1 + 1 * 9^0

end NUMINAMATH_CALUDE_base_conversion_1729_l3695_369537


namespace NUMINAMATH_CALUDE_rackets_sold_l3695_369503

def total_sales : ℝ := 588
def average_price : ℝ := 9.8

theorem rackets_sold (pairs : ℝ) : pairs = total_sales / average_price → pairs = 60 := by
  sorry

end NUMINAMATH_CALUDE_rackets_sold_l3695_369503


namespace NUMINAMATH_CALUDE_vector_parallel_implies_m_equals_two_l3695_369563

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem vector_parallel_implies_m_equals_two (m : ℝ) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (2, 1)
  parallel (a.1 - 2 * b.1, a.2 - 2 * b.2) b →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_m_equals_two_l3695_369563


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l3695_369575

-- Define the hyperbola equation
def hyperbola_equation (m : ℝ) (x y : ℝ) : Prop := x^2 - m*y^2 = 1

-- Define the condition for axis lengths
def axis_length_condition (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a^2 = 1 ∧ b^2 = 1/m ∧ 2*a = 2*(2*b)

-- Theorem statement
theorem hyperbola_m_value (m : ℝ) :
  (∀ x y : ℝ, hyperbola_equation m x y) →
  axis_length_condition m →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l3695_369575


namespace NUMINAMATH_CALUDE_empty_quadratic_inequality_solution_set_l3695_369589

theorem empty_quadratic_inequality_solution_set
  (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4*a*c ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_empty_quadratic_inequality_solution_set_l3695_369589


namespace NUMINAMATH_CALUDE_diego_payment_is_9800_l3695_369539

def total_payment : ℝ := 50000
def celina_payment (diego_payment : ℝ) : ℝ := 1000 + 4 * diego_payment

theorem diego_payment_is_9800 :
  ∃ (diego_payment : ℝ),
    diego_payment + celina_payment diego_payment = total_payment ∧
    diego_payment = 9800 :=
by sorry

end NUMINAMATH_CALUDE_diego_payment_is_9800_l3695_369539


namespace NUMINAMATH_CALUDE_problem_solution_l3695_369567

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 10

theorem problem_solution (m : ℝ) (h_m : m > 1) :
  (∀ x, f m x = x^2 - 2*m*x + 10) →
  (f m m = 1 → ∀ x, f m x = x^2 - 6*x + 10) ∧
  (((∀ x ≤ 2, ∀ y ≤ 2, x < y → f m x > f m y) ∧
    (∀ x ∈ Set.Icc 1 (m + 1), ∀ y ∈ Set.Icc 1 (m + 1), |f m x - f m y| ≤ 9)) →
   m ∈ Set.Icc 2 4) ∧
  ((∃ x ∈ Set.Icc 3 5, f m x = 0) →
   m ∈ Set.Icc (Real.sqrt 10) (7/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3695_369567


namespace NUMINAMATH_CALUDE_star_four_eight_two_l3695_369538

-- Define the ⋆ operation
def star (a b c : ℕ+) : ℚ := (a * b + c) / (a + b + c)

-- Theorem statement
theorem star_four_eight_two :
  star 4 8 2 = 17 / 7 := by sorry

end NUMINAMATH_CALUDE_star_four_eight_two_l3695_369538


namespace NUMINAMATH_CALUDE_jills_gifts_and_charity_l3695_369551

/-- Calculates the amount Jill uses for gifts and charitable causes --/
def gifts_and_charity (net_salary : ℚ) : ℚ :=
  let discretionary_income := (1 / 5) * net_salary
  let vacation_fund := (30 / 100) * discretionary_income
  let savings := (20 / 100) * discretionary_income
  let eating_out := (35 / 100) * discretionary_income
  discretionary_income - (vacation_fund + savings + eating_out)

/-- Theorem stating that Jill uses $99 for gifts and charitable causes --/
theorem jills_gifts_and_charity :
  gifts_and_charity 3300 = 99 := by
  sorry

end NUMINAMATH_CALUDE_jills_gifts_and_charity_l3695_369551


namespace NUMINAMATH_CALUDE_cube_volume_from_doubled_cuboid_edges_l3695_369540

theorem cube_volume_from_doubled_cuboid_edges (l w h : ℝ) : 
  l * w * h = 36 → (2 * l) * (2 * w) * (2 * h) = 288 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_doubled_cuboid_edges_l3695_369540


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l3695_369510

theorem quadratic_equation_root (x : ℝ) : x^2 + 6*x - 4 = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l3695_369510


namespace NUMINAMATH_CALUDE_radian_measure_of_negative_120_degrees_l3695_369517

theorem radian_measure_of_negative_120_degrees :
  let degree_to_radian (d : ℝ) := d * (π / 180)
  degree_to_radian (-120) = -(2 * π / 3) := by sorry

end NUMINAMATH_CALUDE_radian_measure_of_negative_120_degrees_l3695_369517


namespace NUMINAMATH_CALUDE_water_speed_proof_l3695_369561

/-- Proves that the speed of the water is 2 km/h, given the conditions of the swimming problem. -/
theorem water_speed_proof (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : still_water_speed = 4) (h2 : distance = 10) (h3 : time = 5) :
  ∃ water_speed : ℝ, water_speed = 2 ∧ still_water_speed - water_speed = distance / time :=
by sorry

end NUMINAMATH_CALUDE_water_speed_proof_l3695_369561


namespace NUMINAMATH_CALUDE_tan_half_alpha_l3695_369520

theorem tan_half_alpha (α : ℝ) (h1 : π < α) (h2 : α < 3*π/2) 
  (h3 : Real.sin (3*π/2 + α) = 4/5) : Real.tan (α/2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_alpha_l3695_369520


namespace NUMINAMATH_CALUDE_fettuccine_tortellini_ratio_l3695_369591

/-- The ratio of students preferring fettuccine to those preferring tortellini -/
theorem fettuccine_tortellini_ratio 
  (total_students : ℕ) 
  (fettuccine_preference : ℕ) 
  (tortellini_preference : ℕ) 
  (h1 : total_students = 800)
  (h2 : fettuccine_preference = 200)
  (h3 : tortellini_preference = 160) : 
  (fettuccine_preference : ℚ) / tortellini_preference = 5 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_fettuccine_tortellini_ratio_l3695_369591


namespace NUMINAMATH_CALUDE_a_necessary_for_c_l3695_369557

theorem a_necessary_for_c (A B C : Prop) 
  (h1 : ¬A ↔ ¬B) (h2 : ¬B → ¬C) : C → A := by
  sorry

end NUMINAMATH_CALUDE_a_necessary_for_c_l3695_369557


namespace NUMINAMATH_CALUDE_cubic_equation_one_root_strategy_l3695_369531

theorem cubic_equation_one_root_strategy :
  ∃ (strategy : ℝ → ℝ → ℝ),
    ∀ (a b c : ℝ),
      ∃ (root : ℝ),
        (root^3 + a*root^2 + b*root + c = 0) ∧
        (∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 → x = root) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_one_root_strategy_l3695_369531


namespace NUMINAMATH_CALUDE_alex_age_problem_l3695_369581

/-- Alex's age problem -/
theorem alex_age_problem (A M : ℝ) : 
  (A - M = 3 * (A - 4 * M)) → A / M = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_alex_age_problem_l3695_369581


namespace NUMINAMATH_CALUDE_range_of_power_function_l3695_369533

theorem range_of_power_function (m : ℝ) (h : m > 0) :
  Set.range (fun x : ℝ => x ^ m) ∩ Set.Ioo 0 1 = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_power_function_l3695_369533


namespace NUMINAMATH_CALUDE_anion_and_salt_identification_l3695_369521

/-- Represents an anion with an O-O bond -/
structure AnionWithOOBond where
  has_oo_bond : Bool

/-- Represents a salt formed during anodic oxidation of bisulfate -/
structure SaltFromBisulfateOxidation where
  is_sulfate_based : Bool

/-- Theorem stating that an anion with an O-O bond is a peroxide ion and 
    the salt formed from bisulfate oxidation is sulfate-based -/
theorem anion_and_salt_identification 
  (anion : AnionWithOOBond) 
  (salt : SaltFromBisulfateOxidation) : 
  (anion.has_oo_bond → (∃ x : String, x = "O₂²⁻")) ∧ 
  (salt.is_sulfate_based → (∃ y : String, y = "K₂SO₄")) := by
  sorry

end NUMINAMATH_CALUDE_anion_and_salt_identification_l3695_369521


namespace NUMINAMATH_CALUDE_math_books_prob_theorem_l3695_369570

/-- The probability of all three mathematics textbooks ending up in the same box -/
def math_books_same_box_prob (total_books n_math_books : ℕ) 
  (box_sizes : Fin 3 → ℕ) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem math_books_prob_theorem :
  let total_books : ℕ := 15
  let n_math_books : ℕ := 3
  let box_sizes : Fin 3 → ℕ := ![4, 5, 6]
  math_books_same_box_prob total_books n_math_books box_sizes = 9 / 121 :=
sorry

end NUMINAMATH_CALUDE_math_books_prob_theorem_l3695_369570


namespace NUMINAMATH_CALUDE_lcm_problem_l3695_369544

theorem lcm_problem (A B : ℕ+) (h1 : A * B = 45276) (h2 : Nat.gcd A B = 22) :
  Nat.lcm A B = 2058 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3695_369544


namespace NUMINAMATH_CALUDE_roots_of_equation_number_of_roots_l3695_369535

def f (x : ℝ) : ℝ := x + |x^2 - 1|

theorem roots_of_equation (k : ℝ) :
  (∀ x, f x ≠ k) ∨
  (∃! x, f x = k) ∨
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = k ∧ f x₂ = k ∧ ∀ x, f x = k → x = x₁ ∨ x = x₂) ∨
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃) ∨
  (∃ x₁ x₂ x₃ x₄, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧ f x₄ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) :=
by
  sorry

theorem number_of_roots (k : ℝ) :
  (k < -1 → ∀ x, f x ≠ k) ∧
  (k = -1 → ∃! x, f x = k) ∧
  (-1 < k ∧ k < 1 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = k ∧ f x₂ = k ∧ ∀ x, f x = k → x = x₁ ∨ x = x₂) ∧
  (k = 1 → ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (1 < k ∧ k < 5/4 → ∃ x₁ x₂ x₃ x₄, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧ f x₄ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) ∧
  (k = 5/4 → ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = k ∧ f x₂ = k ∧ f x₃ = k ∧
    ∀ x, f x = k → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (k > 5/4 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = k ∧ f x₂ = k ∧ ∀ x, f x = k → x = x₁ ∨ x = x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_number_of_roots_l3695_369535


namespace NUMINAMATH_CALUDE_train_speed_l3695_369506

/-- Given a train of length 200 meters that takes 5 seconds to cross an electric pole,
    prove that its speed is 40 meters per second. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 200) (h2 : crossing_time = 5) :
  train_length / crossing_time = 40 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l3695_369506


namespace NUMINAMATH_CALUDE_min_disks_to_cover_l3695_369536

/-- Represents a disk in 2D space -/
structure Disk where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a covering of a disk by smaller disks -/
def DiskCovering (large : Disk) (small : List Disk) : Prop :=
  ∀ p : ℝ × ℝ, (p.1 - large.center.1)^2 + (p.2 - large.center.2)^2 ≤ large.radius^2 →
    ∃ d ∈ small, (p.1 - d.center.1)^2 + (p.2 - d.center.2)^2 ≤ d.radius^2

/-- The theorem stating that 7 is the minimum number of smaller disks needed -/
theorem min_disks_to_cover (large : Disk) (small : List Disk) :
  large.radius = 1 →
  (∀ d ∈ small, d.radius = 1/2) →
  DiskCovering large small →
  small.length ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_min_disks_to_cover_l3695_369536


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3695_369548

-- Define the vectors
def a : ℝ × ℝ := (3, 1)
def b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem perpendicular_vectors_m_value :
  ∀ m : ℝ, dot_product a (b m) = 0 → m = -6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3695_369548


namespace NUMINAMATH_CALUDE_range_of_a_l3695_369507

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (2 * (1 - x^2) - 3 * x > 0 → x > a) ∧ 
  (∃ y : ℝ, y > a ∧ 2 * (1 - y^2) - 3 * y ≤ 0)) → 
  a ∈ Set.Iic (-2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3695_369507


namespace NUMINAMATH_CALUDE_projectile_trajectory_area_l3695_369523

theorem projectile_trajectory_area (v₀ g : ℝ) (h₁ : v₀ > 0) (h₂ : g > 0) :
  let v := fun t => v₀ + t * v₀  -- v varies from v₀ to 2v₀
  let x := fun t => (v t)^2 / (2 * g)
  let y := fun t => (v t)^2 / (4 * g)
  let area := ∫ t in (0)..(1), y (v t) * (x (v 1) - x (v 0))
  area = 3 * v₀^4 / (8 * g^2) :=
by sorry

end NUMINAMATH_CALUDE_projectile_trajectory_area_l3695_369523


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3695_369543

/-- Given two perpendicular vectors a and b in ℝ², where a = (3, x) and b = (y, 1),
    prove that x = -7/4 -/
theorem perpendicular_vectors_x_value (x y : ℝ) :
  let a : Fin 2 → ℝ := ![3, x]
  let b : Fin 2 → ℝ := ![y, 1]
  (∀ i j, a i * b j = 0) → x = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3695_369543


namespace NUMINAMATH_CALUDE_extreme_value_at_one_l3695_369576

-- Define the function f(x) = x³ - ax
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem extreme_value_at_one (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 3 := by
  sorry


end NUMINAMATH_CALUDE_extreme_value_at_one_l3695_369576


namespace NUMINAMATH_CALUDE_meal_base_cost_is_28_l3695_369529

/-- Represents the cost structure of a meal --/
structure MealCost where
  baseCost : ℝ
  taxRate : ℝ
  tipRate : ℝ
  totalCost : ℝ

/-- Calculates the total cost of a meal given its base cost, tax rate, and tip rate --/
def calculateTotalCost (m : MealCost) : ℝ :=
  m.baseCost * (1 + m.taxRate + m.tipRate)

/-- Theorem stating that given the specified conditions, the base cost of the meal is $28 --/
theorem meal_base_cost_is_28 (m : MealCost) 
  (h1 : m.taxRate = 0.08)
  (h2 : m.tipRate = 0.18)
  (h3 : m.totalCost = 35.20)
  (h4 : calculateTotalCost m = m.totalCost) :
  m.baseCost = 28 := by
  sorry

#eval (28 : ℚ) * (1 + 0.08 + 0.18)

end NUMINAMATH_CALUDE_meal_base_cost_is_28_l3695_369529


namespace NUMINAMATH_CALUDE_trig_expression_value_l3695_369546

theorem trig_expression_value (α : Real) 
  (h : (Real.tan α - 3) * (Real.sin α + Real.cos α + 3) = 0) : 
  2 + 2/3 * (Real.sin α)^2 + 1/4 * (Real.cos α)^2 = 21/8 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l3695_369546


namespace NUMINAMATH_CALUDE_marias_gum_count_l3695_369542

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem marias_gum_count 
  (x y z : ℕ) 
  (hx : is_two_digit x) 
  (hy : is_two_digit y) 
  (hz : is_two_digit z) : 
  58 + x + y + z = 58 + x + y + z :=
by sorry

end NUMINAMATH_CALUDE_marias_gum_count_l3695_369542


namespace NUMINAMATH_CALUDE_vacation_cost_division_l3695_369514

theorem vacation_cost_division (total_cost : ℝ) (initial_people : ℕ) (cost_reduction : ℝ) (n : ℕ) :
  total_cost = 1000 →
  initial_people = 4 →
  (total_cost / initial_people) - (total_cost / n) = cost_reduction →
  cost_reduction = 50 →
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l3695_369514


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_l3695_369515

theorem complex_magnitude_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_l3695_369515


namespace NUMINAMATH_CALUDE_planes_lines_relations_l3695_369526

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem planes_lines_relations 
  (α β : Plane) (l m : Line) 
  (h1 : perpendicular l α) 
  (h2 : contained_in m β) :
  (parallel α β → line_perpendicular l m) ∧ 
  (line_parallel l m → plane_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_planes_lines_relations_l3695_369526


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l3695_369549

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def rightmost_digit (n : ℕ) : ℕ := n % 10

def move_rightmost_to_leftmost (n : ℕ) : ℕ :=
  (n / 10) + (rightmost_digit n * 100000)

theorem unique_six_digit_number : 
  ∃! n : ℕ, is_six_digit n ∧ 
            rightmost_digit n = 2 ∧
            move_rightmost_to_leftmost n = 2 * n + 2 :=
by
  use 105262
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l3695_369549


namespace NUMINAMATH_CALUDE_lcm_of_36_and_125_l3695_369556

theorem lcm_of_36_and_125 : Nat.lcm 36 125 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_125_l3695_369556


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3695_369595

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_1 + a_3 = 10 and a_4 + a_6 = 5/4, then q = 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_sum1 : a 1 + a 3 = 10) 
  (h_sum2 : a 4 + a 6 = 5/4) : 
  q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3695_369595


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3695_369504

theorem absolute_value_inequality (x y z : ℝ) :
  |x| + |y| + |z| - |x + y| - |y + z| - |z + x| + |x + y + z| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3695_369504
