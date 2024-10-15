import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3465_346597

def A : Set ℝ := {x | (1 : ℝ) / (x - 1) ≤ 1}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3465_346597


namespace NUMINAMATH_CALUDE_combination_properties_l3465_346573

theorem combination_properties (n m : ℕ+) (h : n > m) :
  (Nat.choose n m = Nat.choose n (n - m)) ∧
  (Nat.choose n m + Nat.choose n (m - 1) = Nat.choose (n + 1) m) := by
  sorry

end NUMINAMATH_CALUDE_combination_properties_l3465_346573


namespace NUMINAMATH_CALUDE_standard_deviation_of_scores_l3465_346566

def scores : List ℝ := [10, 10, 10, 9, 10, 8, 8, 10, 10, 8]

theorem standard_deviation_of_scores :
  let n : ℕ := scores.length
  let mean : ℝ := (scores.sum) / n
  let variance : ℝ := (scores.map (λ x => (x - mean)^2)).sum / n
  Real.sqrt variance = 0.9 := by sorry

end NUMINAMATH_CALUDE_standard_deviation_of_scores_l3465_346566


namespace NUMINAMATH_CALUDE_current_speed_l3465_346526

/-- Proves that given a man's speed with and against the current, the speed of the current can be determined. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 21)
  (h2 : speed_against_current = 12.4) :
  ∃ (current_speed : ℝ), current_speed = 4.3 := by
  sorry


end NUMINAMATH_CALUDE_current_speed_l3465_346526


namespace NUMINAMATH_CALUDE_solution_exists_in_interval_l3465_346534

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- State the theorem
theorem solution_exists_in_interval :
  ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_in_interval_l3465_346534


namespace NUMINAMATH_CALUDE_remaining_jellybeans_l3465_346559

/-- Calculates the number of jelly beans remaining in a container after distribution --/
def jellybeans_remaining (initial_count : ℕ) (people : ℕ) (first_group : ℕ) (last_group : ℕ) (last_group_beans : ℕ) : ℕ :=
  initial_count - (first_group * 2 * last_group_beans + last_group * last_group_beans)

/-- Theorem stating the number of jelly beans remaining in the container --/
theorem remaining_jellybeans : 
  jellybeans_remaining 8000 10 6 4 400 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_remaining_jellybeans_l3465_346559


namespace NUMINAMATH_CALUDE_bobby_candy_left_l3465_346513

theorem bobby_candy_left (initial_candy : ℕ) (eaten_candy : ℕ) (h1 : initial_candy = 30) (h2 : eaten_candy = 23) :
  initial_candy - eaten_candy = 7 := by
sorry

end NUMINAMATH_CALUDE_bobby_candy_left_l3465_346513


namespace NUMINAMATH_CALUDE_function_monotonicity_l3465_346537

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

-- Define monotonically increasing function
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem function_monotonicity (a : ℝ) :
  MonotonicallyIncreasing (f a) ↔ (9/4 ≤ a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_function_monotonicity_l3465_346537


namespace NUMINAMATH_CALUDE_razorback_tshirt_revenue_l3465_346575

/-- The amount of money made from selling t-shirts at the Razorback shop -/
theorem razorback_tshirt_revenue :
  let profit_per_tshirt : ℕ := 62
  let tshirts_sold : ℕ := 183
  let total_profit : ℕ := profit_per_tshirt * tshirts_sold
  total_profit = 11346 := by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_revenue_l3465_346575


namespace NUMINAMATH_CALUDE_sequence_sum_l3465_346545

theorem sequence_sum (a b c d : ℕ+) : 
  (∃ r : ℚ, b = a * r ∧ c = a * r^2) →  -- geometric progression
  (d = a + 40) →                        -- arithmetic progression and difference
  a + b + c + d = 110 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3465_346545


namespace NUMINAMATH_CALUDE_first_player_wins_l3465_346593

/-- Represents a stick with a certain length -/
structure Stick :=
  (length : ℝ)

/-- Represents the state of the game -/
structure GameState :=
  (sticks : List Stick)

/-- Represents a player's move, breaking a stick into two parts -/
def breakStick (s : Stick) : Stick × Stick :=
  sorry

/-- Checks if three sticks can form a triangle -/
def canFormTriangle (s1 s2 s3 : Stick) : Prop :=
  sorry

/-- Represents a player's strategy -/
def Strategy := GameState → Option (Stick × (Stick × Stick))

/-- The first player's strategy -/
def firstPlayerStrategy : Strategy :=
  sorry

/-- The second player's strategy -/
def secondPlayerStrategy : Strategy :=
  sorry

/-- Simulates the game for three moves -/
def gameSimulation (s1 : Strategy) (s2 : Strategy) : GameState :=
  sorry

/-- Checks if the given game state allows forming two triangles -/
def canFormTwoTriangles (gs : GameState) : Prop :=
  sorry

/-- The main theorem stating that the first player can guarantee a win -/
theorem first_player_wins :
  ∀ (s2 : Strategy),
  ∃ (s1 : Strategy),
  canFormTwoTriangles (gameSimulation s1 s2) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l3465_346593


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l3465_346532

/-- Represents a truncated cone with horizontal bases -/
structure TruncatedCone where
  bottom_radius : ℝ
  top_radius : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Checks if a sphere is tangent to the truncated cone -/
def is_tangent (cone : TruncatedCone) (sphere : Sphere) : Prop :=
  -- This is a placeholder for the tangency condition
  True

theorem sphere_radius_in_truncated_cone (cone : TruncatedCone) (sphere : Sphere) :
  cone.bottom_radius = 10 ∧ 
  cone.top_radius = 3 ∧ 
  is_tangent cone sphere → 
  sphere.radius = Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l3465_346532


namespace NUMINAMATH_CALUDE_square_perimeter_l3465_346510

/-- The perimeter of a square is equal to four times its side length. -/
theorem square_perimeter (side : ℝ) (h : side = 13) : 4 * side = 52 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3465_346510


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3465_346542

theorem quadratic_minimum (x : ℝ) : 
  let f : ℝ → ℝ := fun x => x^2 - 12*x + 35
  ∃ (min_x : ℝ), ∀ y, f y ≥ f min_x ∧ min_x = 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3465_346542


namespace NUMINAMATH_CALUDE_square_difference_equality_l3465_346560

theorem square_difference_equality : (45 + 15)^2 - (45^2 + 15^2) = 1350 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3465_346560


namespace NUMINAMATH_CALUDE_circumscribed_circle_equation_l3465_346538

/-- The equation of a circle passing through three given points -/
def CircleEquation (x y : ℝ) := x^2 + y^2 - 6*x + 4 = 0

/-- Point A coordinates -/
def A : ℝ × ℝ := (1, 1)

/-- Point B coordinates -/
def B : ℝ × ℝ := (4, 2)

/-- Point C coordinates -/
def C : ℝ × ℝ := (2, -2)

/-- Theorem stating that the given equation represents the circumscribed circle of triangle ABC -/
theorem circumscribed_circle_equation :
  CircleEquation A.1 A.2 ∧
  CircleEquation B.1 B.2 ∧
  CircleEquation C.1 C.2 :=
sorry

end NUMINAMATH_CALUDE_circumscribed_circle_equation_l3465_346538


namespace NUMINAMATH_CALUDE_min_route_length_5x5_city_l3465_346554

/-- Represents a square grid city -/
structure City where
  size : ℕ
  streets : ℕ

/-- Calculates the minimum route length for an Eulerian circuit in the city -/
def minRouteLength (c : City) : ℕ :=
  2 * c.streets + 8

theorem min_route_length_5x5_city :
  ∃ (c : City), c.size = 5 ∧ c.streets = 30 ∧ minRouteLength c = 68 :=
by sorry

end NUMINAMATH_CALUDE_min_route_length_5x5_city_l3465_346554


namespace NUMINAMATH_CALUDE_valid_tiling_conditions_l3465_346541

/-- Represents a tile shape -/
inductive TileShape
  | L  -- L-shaped tile covering 3 squares
  | T  -- T-shaped tile covering 4 squares

/-- Represents a valid tiling of an n×n board -/
def ValidTiling (n : ℕ) : Prop :=
  ∃ (tiling : List (TileShape × ℕ × ℕ)), 
    (∀ (t : TileShape) (x y : ℕ), (t, x, y) ∈ tiling → x < n ∧ y < n) ∧ 
    (∀ (x y : ℕ), x < n ∧ y < n → ∃! (t : TileShape), (t, x, y) ∈ tiling)

/-- The main theorem stating the conditions for a valid tiling -/
theorem valid_tiling_conditions (n : ℕ) : 
  ValidTiling n ↔ (n % 4 = 0 ∧ n > 4) :=
sorry

end NUMINAMATH_CALUDE_valid_tiling_conditions_l3465_346541


namespace NUMINAMATH_CALUDE_intersecting_sphere_yz_radius_l3465_346588

/-- A sphere intersecting two planes -/
structure IntersectingSphere where
  /-- Center of the intersection circle with xy-plane -/
  xy_center : ℝ × ℝ × ℝ
  /-- Radius of the intersection circle with xy-plane -/
  xy_radius : ℝ
  /-- Center of the intersection circle with yz-plane -/
  yz_center : ℝ × ℝ × ℝ
  /-- Radius of the intersection circle with yz-plane -/
  yz_radius : ℝ

/-- The theorem stating the radius of the yz-plane intersection -/
theorem intersecting_sphere_yz_radius (sphere : IntersectingSphere) 
  (h1 : sphere.xy_center = (3, 5, 0))
  (h2 : sphere.xy_radius = 2)
  (h3 : sphere.yz_center = (0, 5, -8)) :
  sphere.yz_radius = Real.sqrt 59 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_sphere_yz_radius_l3465_346588


namespace NUMINAMATH_CALUDE_first_term_is_seven_l3465_346585

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 2) + (-1)^n * a n = 3 * n - 1

/-- The sum of the first 16 terms of the sequence equals 540 -/
def SumCondition (a : ℕ → ℚ) : Prop :=
  (Finset.range 16).sum a = 540

/-- The theorem stating that a₁ = 7 for the given conditions -/
theorem first_term_is_seven
    (a : ℕ → ℚ)
    (h_recurrence : RecurrenceSequence a)
    (h_sum : SumCondition a) :
    a 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_seven_l3465_346585


namespace NUMINAMATH_CALUDE_extreme_value_inequality_l3465_346523

/-- A function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - (1/2) * a * x^2 + (4-a) * x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 4 / x - a * x + (4-a)

theorem extreme_value_inequality (a : ℝ) (x₀ x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx₀ : x₀ > 0) 
  (hx₁ : x₁ > 0) 
  (hx₂ : x₂ > 0) 
  (h_order : x₁ < x₂) 
  (h_extreme : ∃ x, x > 0 ∧ ∀ y, y > 0 → f a x ≥ f a y) 
  (h_mean_value : f a x₁ - f a x₂ = f_deriv a x₀ * (x₁ - x₂)) :
  x₁ + x₂ > 2 * x₀ := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_inequality_l3465_346523


namespace NUMINAMATH_CALUDE_constant_term_quadratic_l3465_346539

theorem constant_term_quadratic (x : ℝ) : 
  (2 * x^2 = x + 4) → 
  (∃ a b : ℝ, 2 * x^2 - x - 4 = a * x^2 + b * x + (-4)) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_quadratic_l3465_346539


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3465_346512

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1)) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3465_346512


namespace NUMINAMATH_CALUDE_max_value_relationship_l3465_346530

theorem max_value_relationship (x y : ℝ) : 
  (∀ a b : ℝ, 2005 - (a + b)^2 ≤ 2005 - (x + y)^2) → x = -y := by
  sorry

end NUMINAMATH_CALUDE_max_value_relationship_l3465_346530


namespace NUMINAMATH_CALUDE_max_value_ab_l3465_346581

theorem max_value_ab (a b : ℝ) (g : ℝ → ℝ) (ha : a > 0) (hb : b > 0)
  (hg : ∀ x, g x = 2^x) (h_prod : g a * g b = 2) :
  ∀ x y, x > 0 → y > 0 → g x * g y = 2 → x * y ≤ (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_value_ab_l3465_346581


namespace NUMINAMATH_CALUDE_binomial_difference_divisibility_l3465_346576

theorem binomial_difference_divisibility (k : ℕ) (h : k ≥ 2) :
  ∃ n : ℕ, (Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1))) = n * 2^(3*k) ∧
           (Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1))) % 2^(3*k+1) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_binomial_difference_divisibility_l3465_346576


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3465_346553

theorem rectangular_field_area (m : ℕ) : 
  (3 * m + 8) * (m - 3) = 76 → m = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3465_346553


namespace NUMINAMATH_CALUDE_regular_polygon_angle_relation_l3465_346556

theorem regular_polygon_angle_relation (m : ℕ) : m ≥ 3 →
  (120 : ℝ) = 4 * (360 / m) → m = 12 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_angle_relation_l3465_346556


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_nine_over_x_l3465_346520

theorem min_value_of_x_plus_nine_over_x (x : ℝ) (hx : x > 0) :
  x + 9 / x ≥ 6 ∧ (x + 9 / x = 6 ↔ x = 3) := by sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_nine_over_x_l3465_346520


namespace NUMINAMATH_CALUDE_max_value_constrained_l3465_346515

theorem max_value_constrained (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  (∃ x y z : ℝ, 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 ∧ 8 * x + 3 * y + 5 * z > 8 * a + 3 * b + 5 * c) ∨
  8 * a + 3 * b + 5 * c = 7 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_constrained_l3465_346515


namespace NUMINAMATH_CALUDE_largest_solution_bound_l3465_346586

theorem largest_solution_bound (x : ℝ) : 
  3 * (9 * x^2 + 15 * x + 20) = x * (9 * x - 60) →
  x ≤ -0.642 ∧ x > -0.643 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_bound_l3465_346586


namespace NUMINAMATH_CALUDE_students_in_cars_l3465_346583

def total_students : ℕ := 375
def num_buses : ℕ := 7
def students_per_bus : ℕ := 53

theorem students_in_cars : 
  total_students - (num_buses * students_per_bus) = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_in_cars_l3465_346583


namespace NUMINAMATH_CALUDE_cubic_tangent_perpendicular_l3465_346547

/-- Given a cubic function f(x) = ax³ + x + 1, if its tangent line at x = 1 is
    perpendicular to the line x + 4y = 0, then a = 1. -/
theorem cubic_tangent_perpendicular (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + x + 1
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 1
  (f' 1) * (-1/4) = -1 →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_cubic_tangent_perpendicular_l3465_346547


namespace NUMINAMATH_CALUDE_arc_length_for_given_angle_l3465_346578

theorem arc_length_for_given_angle (r : ℝ) (α : ℝ) (h1 : r = 2) (h2 : α = π / 7) :
  r * α = 2 * π / 7 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_given_angle_l3465_346578


namespace NUMINAMATH_CALUDE_total_soaking_time_l3465_346524

/-- Calculates the total soaking time for clothes with grass and marinara stains -/
theorem total_soaking_time
  (grass_stain_time : ℕ)
  (marinara_stain_time : ℕ)
  (grass_stain_count : ℕ)
  (marinara_stain_count : ℕ)
  (h1 : grass_stain_time = 4)
  (h2 : marinara_stain_time = 7)
  (h3 : grass_stain_count = 3)
  (h4 : marinara_stain_count = 1) :
  grass_stain_time * grass_stain_count + marinara_stain_time * marinara_stain_count = 19 :=
by
  sorry


end NUMINAMATH_CALUDE_total_soaking_time_l3465_346524


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l3465_346533

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l3465_346533


namespace NUMINAMATH_CALUDE_sweet_potatoes_theorem_l3465_346544

def sweet_potatoes_problem (total_harvested sold_to_adams sold_to_lenon : ℕ) : Prop :=
  total_harvested - (sold_to_adams + sold_to_lenon) = 45

theorem sweet_potatoes_theorem :
  sweet_potatoes_problem 80 20 15 := by
  sorry

end NUMINAMATH_CALUDE_sweet_potatoes_theorem_l3465_346544


namespace NUMINAMATH_CALUDE_green_peppers_weight_equal_pepper_weights_l3465_346580

/-- The weight of green peppers bought by Dale's Vegetarian Restaurant -/
def green_peppers : ℝ := 2.8333333335

/-- The total weight of peppers bought by Dale's Vegetarian Restaurant -/
def total_peppers : ℝ := 5.666666667

/-- Theorem stating that the weight of green peppers is half the total weight of peppers -/
theorem green_peppers_weight :
  green_peppers = total_peppers / 2 :=
by sorry

/-- Theorem stating that the weight of green peppers is equal to the weight of red peppers -/
theorem equal_pepper_weights :
  green_peppers = total_peppers - green_peppers :=
by sorry

end NUMINAMATH_CALUDE_green_peppers_weight_equal_pepper_weights_l3465_346580


namespace NUMINAMATH_CALUDE_three_distinct_roots_l3465_346595

/-- The polynomial Q(x) with parameter p -/
def Q (p : ℝ) (x : ℝ) : ℝ := x^3 + p * x^2 - p * x - 1

/-- Theorem stating the condition for Q(x) to have three distinct real roots -/
theorem three_distinct_roots (p : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    Q p x = 0 ∧ Q p y = 0 ∧ Q p z = 0) ↔ 
  (p > 1 ∨ p < -3) :=
sorry

end NUMINAMATH_CALUDE_three_distinct_roots_l3465_346595


namespace NUMINAMATH_CALUDE_train_length_l3465_346507

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 15 → speed * (5/18) * time = 375 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3465_346507


namespace NUMINAMATH_CALUDE_linear_function_proof_l3465_346552

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

theorem linear_function_proof (f : ℝ → ℝ) 
  (h1 : is_linear f) 
  (h2 : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) : 
  ∀ x, f x = 2 * x + 7 := by sorry

end NUMINAMATH_CALUDE_linear_function_proof_l3465_346552


namespace NUMINAMATH_CALUDE_tray_height_l3465_346550

/-- The height of a tray formed from a square paper with specific cuts -/
theorem tray_height (side_length : ℝ) (cut_start : ℝ) (cut_angle : ℝ) : 
  side_length = 50 →
  cut_start = Real.sqrt 5 →
  cut_angle = π / 4 →
  (Real.sqrt 10) / 2 = 
    cut_start * Real.sin (cut_angle / 2) := by
  sorry

end NUMINAMATH_CALUDE_tray_height_l3465_346550


namespace NUMINAMATH_CALUDE_sum_of_root_pairs_is_124_l3465_346598

def root_pairs : List (Nat × Nat) := [(1, 48), (2, 24), (3, 16), (4, 12), (6, 8)]

def sum_of_pairs (pairs : List (Nat × Nat)) : Nat :=
  pairs.map (fun (a, b) => a + b) |> List.sum

theorem sum_of_root_pairs_is_124 : sum_of_pairs root_pairs = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_root_pairs_is_124_l3465_346598


namespace NUMINAMATH_CALUDE_shortest_paths_correct_l3465_346529

def shortest_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

theorem shortest_paths_correct (m n : ℕ) : 
  shortest_paths m n = Nat.choose (m + n) m :=
by sorry

end NUMINAMATH_CALUDE_shortest_paths_correct_l3465_346529


namespace NUMINAMATH_CALUDE_expected_wins_equal_l3465_346591

/-- The total number of balls in the lottery box -/
def total_balls : ℕ := 8

/-- The number of red balls in the lottery box -/
def red_balls : ℕ := 4

/-- The number of black balls in the lottery box -/
def black_balls : ℕ := 4

/-- The number of draws made -/
def num_draws : ℕ := 2

/-- Represents the outcome of a single lottery draw -/
inductive DrawResult
| Red
| Black

/-- Represents the result of two draws -/
inductive TwoDrawResult
| Win  -- Two balls of the same color
| Lose -- Two balls of different colors

/-- The probability of winning in a single draw with replacement -/
def prob_win_with_replacement : ℚ :=
  (red_balls.choose 2 + black_balls.choose 2) / total_balls.choose 2

/-- The expected number of wins with replacement -/
def expected_wins_with_replacement : ℚ :=
  num_draws * prob_win_with_replacement

/-- The probability of winning in a single draw without replacement -/
def prob_win_without_replacement : ℚ :=
  (red_balls.choose 2 + black_balls.choose 2) / total_balls.choose 2

/-- The expected number of wins without replacement -/
def expected_wins_without_replacement : ℚ :=
  (0 * (12 / 35) + 1 * (16 / 35) + 2 * (7 / 35))

/-- Theorem stating that the expected number of wins is 6/7 for both cases -/
theorem expected_wins_equal :
  expected_wins_with_replacement = 6/7 ∧
  expected_wins_without_replacement = 6/7 := by
  sorry


end NUMINAMATH_CALUDE_expected_wins_equal_l3465_346591


namespace NUMINAMATH_CALUDE_sum_reciprocals_and_diff_squares_l3465_346511

theorem sum_reciprocals_and_diff_squares (x y : ℝ) 
  (sum_eq : x + y = 12) 
  (prod_eq : x * y = 32) : 
  (1 / x + 1 / y = 3 / 8) ∧ (x^2 - y^2 = 48 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_and_diff_squares_l3465_346511


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l3465_346565

theorem profit_percent_calculation (selling_price cost_price : ℝ) :
  cost_price = 0.25 * selling_price →
  (selling_price - cost_price) / cost_price * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l3465_346565


namespace NUMINAMATH_CALUDE_vanilla_to_cream_cheese_ratio_l3465_346521

-- Define the ratios and quantities
def sugar_to_cream_cheese_ratio : ℚ := 1 / 4
def vanilla_to_eggs_ratio : ℚ := 1 / 2
def sugar_used : ℚ := 2
def eggs_used : ℚ := 8
def teaspoons_per_cup : ℚ := 48

-- Theorem to prove
theorem vanilla_to_cream_cheese_ratio :
  let cream_cheese := sugar_used / sugar_to_cream_cheese_ratio
  let vanilla := eggs_used * vanilla_to_eggs_ratio
  let cream_cheese_teaspoons := cream_cheese * teaspoons_per_cup
  vanilla / cream_cheese_teaspoons = 1 / 96 :=
by sorry

end NUMINAMATH_CALUDE_vanilla_to_cream_cheese_ratio_l3465_346521


namespace NUMINAMATH_CALUDE_initial_clean_and_jerk_was_80kg_l3465_346562

/-- Represents John's weightlifting progress --/
structure Weightlifting where
  initial_snatch : ℝ
  initial_clean_and_jerk : ℝ
  new_combined_total : ℝ

/-- Calculates the new Snatch weight after an 80% increase --/
def new_snatch (w : Weightlifting) : ℝ :=
  w.initial_snatch * 1.8

/-- Calculates the new Clean & Jerk weight after doubling --/
def new_clean_and_jerk (w : Weightlifting) : ℝ :=
  w.initial_clean_and_jerk * 2

/-- Theorem stating that John's initial Clean & Jerk weight was 80 kg --/
theorem initial_clean_and_jerk_was_80kg (w : Weightlifting) 
  (h1 : w.initial_snatch = 50)
  (h2 : new_snatch w + new_clean_and_jerk w = w.new_combined_total)
  (h3 : w.new_combined_total = 250) : 
  w.initial_clean_and_jerk = 80 := by
  sorry


end NUMINAMATH_CALUDE_initial_clean_and_jerk_was_80kg_l3465_346562


namespace NUMINAMATH_CALUDE_correct_guess_probability_l3465_346522

/-- Represents a six-digit password with an unknown last digit -/
structure Password :=
  (first_five : Nat)
  (last_digit : Nat)

/-- The set of possible last digits -/
def possible_last_digits : Finset Nat := Finset.range 10

/-- The probability of guessing the correct password on the first try -/
def guess_probability (p : Password) : ℚ :=
  1 / (Finset.card possible_last_digits : ℚ)

theorem correct_guess_probability (p : Password) :
  p.last_digit ∈ possible_last_digits →
  guess_probability p = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_guess_probability_l3465_346522


namespace NUMINAMATH_CALUDE_fruit_display_total_l3465_346543

/-- Represents the number of fruits on a display -/
structure FruitDisplay where
  bananas : ℕ
  oranges : ℕ
  apples : ℕ
  lemons : ℕ

/-- Calculates the total number of fruits on the display -/
def totalFruits (d : FruitDisplay) : ℕ :=
  d.bananas + d.oranges + d.apples + d.lemons

/-- Theorem stating the total number of fruits on the display -/
theorem fruit_display_total (d : FruitDisplay) 
  (h1 : d.bananas = 5)
  (h2 : d.oranges = 2 * d.bananas)
  (h3 : d.apples = 2 * d.oranges)
  (h4 : d.lemons = (d.apples + d.bananas) / 2) :
  totalFruits d = 47 := by
  sorry

#eval totalFruits { bananas := 5, oranges := 10, apples := 20, lemons := 12 }

end NUMINAMATH_CALUDE_fruit_display_total_l3465_346543


namespace NUMINAMATH_CALUDE_pairing_theorem_l3465_346536

/-- Represents the pairing of boys and girls in the school event. -/
structure Pairing where
  boys : ℕ
  girls : ℕ
  first_pairing : ℕ := 3
  pairing_increment : ℕ := 2

/-- The relationship between boys and girls in the pairing. -/
def pairing_relationship (p : Pairing) : Prop :=
  p.boys = (p.girls - 1) / 2

/-- Theorem stating the relationship between boys and girls in the pairing. -/
theorem pairing_theorem (p : Pairing) : pairing_relationship p := by
  sorry

#check pairing_theorem

end NUMINAMATH_CALUDE_pairing_theorem_l3465_346536


namespace NUMINAMATH_CALUDE_sum_of_digits_9cd_l3465_346517

def c : ℕ := 10^1984 + 6

def d : ℕ := 7 * (10^1984 - 1) / 9 + 4

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_9cd : sum_of_digits (9 * c * d) = 33728 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_9cd_l3465_346517


namespace NUMINAMATH_CALUDE_contracting_schemes_l3465_346568

def number_of_projects : ℕ := 6
def projects_for_A : ℕ := 3
def projects_for_B : ℕ := 2
def projects_for_C : ℕ := 1

theorem contracting_schemes :
  (number_of_projects.choose projects_for_A) *
  ((number_of_projects - projects_for_A).choose projects_for_B) *
  ((number_of_projects - projects_for_A - projects_for_B).choose projects_for_C) = 60 := by
  sorry

end NUMINAMATH_CALUDE_contracting_schemes_l3465_346568


namespace NUMINAMATH_CALUDE_intersection_not_solution_quadratic_solution_l3465_346551

theorem intersection_not_solution : ∀ x y : ℝ,
  (y = x ∧ y = x - 4) → (x ≠ 2 ∨ y ≠ 2) :=
by sorry

theorem quadratic_solution : ∀ x : ℝ,
  x^2 - 4*x + 4 = 0 → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_not_solution_quadratic_solution_l3465_346551


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3465_346508

theorem fraction_sum_equality : 
  (1 : ℚ) / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 2 / 15 = -2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3465_346508


namespace NUMINAMATH_CALUDE_area_between_circles_and_xaxis_l3465_346574

/-- The area of the region bound by two circles and the x-axis -/
theorem area_between_circles_and_xaxis :
  let c1_center : ℝ × ℝ := (3, 5)
  let c2_center : ℝ × ℝ := (9, 5)
  let radius : ℝ := 3
  let rectangle_area : ℝ := (c2_center.1 - c1_center.1) * radius
  let sector_area : ℝ := (1/4) * π * radius^2
  rectangle_area - 2 * sector_area = 18 - (9/2) * π := by sorry

end NUMINAMATH_CALUDE_area_between_circles_and_xaxis_l3465_346574


namespace NUMINAMATH_CALUDE_table_tennis_outcomes_count_l3465_346540

/-- Represents the number of possible outcomes in a table tennis match -/
def table_tennis_outcomes : ℕ := 30

/-- The winning condition for the match -/
def winning_games : ℕ := 3

/-- Theorem stating that the number of possible outcomes in a table tennis match
    where the first to win 3 games wins the match is 30 -/
theorem table_tennis_outcomes_count :
  (∀ (match_length : ℕ), match_length ≥ winning_games →
    (∃ (winner_games loser_games : ℕ),
      winner_games = winning_games ∧
      winner_games + loser_games = match_length ∧
      winner_games > loser_games)) →
  table_tennis_outcomes = 30 := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_outcomes_count_l3465_346540


namespace NUMINAMATH_CALUDE_floor_sqrt_sum_eq_floor_sqrt_4n_plus_2_l3465_346531

theorem floor_sqrt_sum_eq_floor_sqrt_4n_plus_2 (n : ℕ+) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_sum_eq_floor_sqrt_4n_plus_2_l3465_346531


namespace NUMINAMATH_CALUDE_inside_implies_intersects_on_implies_tangent_outside_implies_no_intersection_l3465_346506

-- Define the circle C
def Circle (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = r^2}

-- Define the line l
def Line (x y : ℝ) : Set (ℝ × ℝ) := {p | x * p.1 + y * p.2 = x^2 + y^2}

-- Define point inside circle
def IsInside (p : ℝ × ℝ) (r : ℝ) : Prop := p.1^2 + p.2^2 < r^2

-- Define point on circle
def IsOn (p : ℝ × ℝ) (r : ℝ) : Prop := p.1^2 + p.2^2 = r^2

-- Define point outside circle
def IsOutside (p : ℝ × ℝ) (r : ℝ) : Prop := p.1^2 + p.2^2 > r^2

-- Define line intersects circle
def Intersects (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := ∃ p, p ∈ l ∧ p ∈ c

-- Define line tangent to circle
def IsTangent (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := ∃! p, p ∈ l ∧ p ∈ c

-- Define line does not intersect circle
def DoesNotIntersect (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop := ∀ p, p ∈ l → p ∉ c

-- Theorem 1
theorem inside_implies_intersects (x y r : ℝ) (h1 : IsInside (x, y) r) (h2 : (x, y) ≠ (0, 0)) :
  Intersects (Line x y) (Circle r) := by sorry

-- Theorem 2
theorem on_implies_tangent (x y r : ℝ) (h : IsOn (x, y) r) :
  IsTangent (Line x y) (Circle r) := by sorry

-- Theorem 3
theorem outside_implies_no_intersection (x y r : ℝ) (h : IsOutside (x, y) r) :
  DoesNotIntersect (Line x y) (Circle r) := by sorry

end NUMINAMATH_CALUDE_inside_implies_intersects_on_implies_tangent_outside_implies_no_intersection_l3465_346506


namespace NUMINAMATH_CALUDE_water_drinkers_l3465_346561

theorem water_drinkers (total : ℕ) (juice_percent : ℚ) (water_percent : ℚ) (juice_drinkers : ℕ) : ℕ :=
  let water_drinkers : ℕ := 60
  have h1 : juice_percent = 70 / 100 := by sorry
  have h2 : water_percent = 30 / 100 := by sorry
  have h3 : juice_percent + water_percent = 1 := by sorry
  have h4 : juice_drinkers = 140 := by sorry
  have h5 : ↑juice_drinkers / ↑total = juice_percent := by sorry
  have h6 : ↑water_drinkers / ↑total = water_percent := by sorry
  water_drinkers

#check water_drinkers

end NUMINAMATH_CALUDE_water_drinkers_l3465_346561


namespace NUMINAMATH_CALUDE_l_shaped_area_l3465_346582

/-- The area of an L-shaped region formed by subtracting three squares from a larger square -/
theorem l_shaped_area (total_side : ℝ) (small_side1 small_side2 large_side : ℝ) :
  total_side = 7 ∧ 
  small_side1 = 2 ∧ 
  small_side2 = 2 ∧ 
  large_side = 5 →
  total_side^2 - (small_side1^2 + small_side2^2 + large_side^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_l3465_346582


namespace NUMINAMATH_CALUDE_julia_total_kids_l3465_346518

/-- The number of kids Julia played with on each day of the week -/
structure WeeklyKids where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Calculate the total number of kids Julia played with throughout the week -/
def totalKids (w : WeeklyKids) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday + w.sunday

/-- The conditions given in the problem -/
def juliaWeek : WeeklyKids where
  monday := 15
  tuesday := 18
  wednesday := 25
  thursday := 30
  friday := 30 + (30 * 20 / 100)
  saturday := (30 + (30 * 20 / 100)) - ((30 + (30 * 20 / 100)) * 30 / 100)
  sunday := 15 * 2

/-- Theorem stating that the total number of kids Julia played with is 180 -/
theorem julia_total_kids : totalKids juliaWeek = 180 := by
  sorry

end NUMINAMATH_CALUDE_julia_total_kids_l3465_346518


namespace NUMINAMATH_CALUDE_kiran_money_l3465_346505

/-- Given the ratios of money between Ravi, Giri, and Kiran, and Ravi's amount of money,
    prove that Kiran has $105. -/
theorem kiran_money (ravi giri kiran : ℚ) : 
  (ravi / giri = 6 / 7) →
  (giri / kiran = 6 / 15) →
  (ravi = 36) →
  (kiran = 105) := by
sorry

end NUMINAMATH_CALUDE_kiran_money_l3465_346505


namespace NUMINAMATH_CALUDE_no_positive_abc_with_all_roots_l3465_346502

theorem no_positive_abc_with_all_roots : ¬ ∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (b^2 - 4*a*c ≥ 0) ∧ 
  (c^2 - 4*b*a ≥ 0) ∧ 
  (a^2 - 4*b*c ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_abc_with_all_roots_l3465_346502


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_l3465_346527

theorem sum_of_last_two_digits (n : ℕ) : (7^25 + 13^25) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_l3465_346527


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3465_346504

/-- A quadratic function with graph opening upwards and vertex at (1, -2) -/
def quadratic_function (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem quadratic_function_properties :
  ∃ (a : ℝ), a > 0 ∧
  (∀ x : ℝ, quadratic_function x = a * (x - 1)^2 - 2) ∧
  (∀ x : ℝ, quadratic_function x ≥ -2) ∧
  quadratic_function 1 = -2 := by
sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l3465_346504


namespace NUMINAMATH_CALUDE_smartphone_savings_proof_l3465_346563

/-- Calculates the required weekly savings to reach a target amount. -/
def weekly_savings (smartphone_cost : ℚ) (current_savings : ℚ) (saving_weeks : ℕ) : ℚ :=
  (smartphone_cost - current_savings) / saving_weeks

/-- Proves that the weekly savings required to buy a $160 smartphone
    with $40 current savings over 8 weeks is $15. -/
theorem smartphone_savings_proof :
  let smartphone_cost : ℚ := 160
  let current_savings : ℚ := 40
  let saving_weeks : ℕ := 8
  weekly_savings smartphone_cost current_savings saving_weeks = 15 := by
sorry

end NUMINAMATH_CALUDE_smartphone_savings_proof_l3465_346563


namespace NUMINAMATH_CALUDE_parabola_solution_l3465_346501

/-- The parabola C: y^2 = 6x with focus F, and a point M(x,y) on C where |MF| = 5/2 and y > 0 -/
def parabola_problem (x y : ℝ) : Prop :=
  y^2 = 6*x ∧ y > 0 ∧ (x + 3/2)^2 + y^2 = (5/2)^2

/-- The coordinates of point M are (1, √6) -/
theorem parabola_solution :
  ∀ x y : ℝ, parabola_problem x y → x = 1 ∧ y = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_parabola_solution_l3465_346501


namespace NUMINAMATH_CALUDE_minimum_garden_width_minimum_garden_width_is_ten_l3465_346557

theorem minimum_garden_width (w : ℝ) : w > 0 → w * (w + 10) ≥ 150 → w ≥ 10 := by
  sorry

theorem minimum_garden_width_is_ten : ∃ w : ℝ, w > 0 ∧ w * (w + 10) ≥ 150 ∧ ∀ x : ℝ, x > 0 → x * (x + 10) ≥ 150 → x ≥ w := by
  sorry

end NUMINAMATH_CALUDE_minimum_garden_width_minimum_garden_width_is_ten_l3465_346557


namespace NUMINAMATH_CALUDE_total_wash_time_l3465_346535

def wash_time_normal : ℕ := 4 + 7 + 4 + 9

def wash_time_suv : ℕ := 2 * wash_time_normal

def wash_time_minivan : ℕ := (3 * wash_time_normal) / 2

def num_normal_cars : ℕ := 3

def num_suvs : ℕ := 2

def num_minivans : ℕ := 1

def break_time : ℕ := 5

def total_vehicles : ℕ := num_normal_cars + num_suvs + num_minivans

theorem total_wash_time : 
  num_normal_cars * wash_time_normal + 
  num_suvs * wash_time_suv + 
  num_minivans * wash_time_minivan + 
  (total_vehicles - 1) * break_time = 229 := by
  sorry

end NUMINAMATH_CALUDE_total_wash_time_l3465_346535


namespace NUMINAMATH_CALUDE_train_length_calculation_l3465_346500

/-- Proves that a train with given speed, crossing a bridge of known length in a specific time, has a particular length. -/
theorem train_length_calculation (train_speed : Real) (bridge_length : Real) (crossing_time : Real) (train_length : Real) : 
  train_speed = 36 → -- speed in km/hr
  bridge_length = 132 → -- bridge length in meters
  crossing_time = 24.198064154867613 → -- time to cross the bridge in seconds
  train_length = 109.98064154867613 → -- train length in meters
  (train_speed * 1000 / 3600) * crossing_time = bridge_length + train_length := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l3465_346500


namespace NUMINAMATH_CALUDE_regression_line_properties_l3465_346564

-- Define random variables x and y
variable (x y : ℝ → ℝ)

-- Define that x and y are correlated
variable (h_correlated : Correlated x y)

-- Define the mean of x and y
def x_mean : ℝ := sorry
def y_mean : ℝ := sorry

-- Define the regression line
def regression_line (x y : ℝ → ℝ) : ℝ → ℝ := sorry

-- Define the slope and intercept of the regression line
def a : ℝ := 0.2
def b : ℝ := 12

theorem regression_line_properties :
  (∀ t : ℝ, regression_line x y t = a * t + b) →
  (regression_line x y x_mean = y_mean) ∧
  (∀ δ : ℝ, regression_line x y (x_mean + δ) - regression_line x y x_mean = a * δ) :=
sorry

end NUMINAMATH_CALUDE_regression_line_properties_l3465_346564


namespace NUMINAMATH_CALUDE_sandra_beignets_l3465_346570

/-- The number of beignets Sandra eats in 16 weeks -/
def total_beignets : ℕ := 336

/-- The number of weeks -/
def num_weeks : ℕ := 16

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of beignets Sandra eats every morning -/
def beignets_per_morning : ℕ := total_beignets / (num_weeks * days_per_week)

theorem sandra_beignets : beignets_per_morning = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandra_beignets_l3465_346570


namespace NUMINAMATH_CALUDE_machine_x_production_rate_l3465_346555

/-- Production rates and times for two machines -/
structure MachineProduction where
  x_rate : ℝ  -- Production rate of Machine X (widgets per hour)
  y_rate : ℝ  -- Production rate of Machine Y (widgets per hour)
  x_time : ℝ  -- Time taken by Machine X to produce 1080 widgets
  y_time : ℝ  -- Time taken by Machine Y to produce 1080 widgets

/-- Theorem stating the production rate of Machine X -/
theorem machine_x_production_rate (m : MachineProduction) :
  m.x_rate = 18 :=
by
  have h1 : m.x_time = m.y_time + 10 := by sorry
  have h2 : m.y_rate = 1.2 * m.x_rate := by sorry
  have h3 : m.x_rate * m.x_time = 1080 := by sorry
  have h4 : m.y_rate * m.y_time = 1080 := by sorry
  sorry

end NUMINAMATH_CALUDE_machine_x_production_rate_l3465_346555


namespace NUMINAMATH_CALUDE_b_savings_l3465_346594

/-- Given two people a and b with monthly incomes and expenditures, calculate b's savings -/
theorem b_savings (income_a income_b expenditure_a expenditure_b savings_a : ℕ) 
  (h1 : income_a * 6 = income_b * 5)  -- income ratio 5:6
  (h2 : expenditure_a * 4 = expenditure_b * 3)  -- expenditure ratio 3:4
  (h3 : income_a - expenditure_a = savings_a)
  (h4 : savings_a = 1800)
  (h5 : income_b = 7200) :
  income_b - expenditure_b = 1600 := by sorry

end NUMINAMATH_CALUDE_b_savings_l3465_346594


namespace NUMINAMATH_CALUDE_salary_degrees_in_circle_graph_l3465_346503

-- Define the percentages for each category
def transportation_percent : ℝ := 15
def research_dev_percent : ℝ := 9
def utilities_percent : ℝ := 5
def equipment_percent : ℝ := 4
def supplies_percent : ℝ := 2

-- Define the total degrees in a circle
def total_degrees : ℝ := 360

-- Theorem statement
theorem salary_degrees_in_circle_graph :
  let other_categories_percent := transportation_percent + research_dev_percent + 
                                  utilities_percent + equipment_percent + supplies_percent
  let salary_percent := 100 - other_categories_percent
  let salary_degrees := (salary_percent / 100) * total_degrees
  salary_degrees = 234 := by
sorry


end NUMINAMATH_CALUDE_salary_degrees_in_circle_graph_l3465_346503


namespace NUMINAMATH_CALUDE_divisibility_of_T_members_l3465_346516

/-- The set of all numbers which are the sum of the squares of four consecutive integers -/
def T : Set ℤ := {x | ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2}

theorem divisibility_of_T_members :
  (∃ x ∈ T, 5 ∣ x) ∧ (∀ x ∈ T, ¬(7 ∣ x)) := by sorry

end NUMINAMATH_CALUDE_divisibility_of_T_members_l3465_346516


namespace NUMINAMATH_CALUDE_cosine_angle_OAB_l3465_346587

/-- Given points A and B in a 2D Cartesian coordinate system with O as the origin,
    prove that the cosine of angle OAB is equal to -√2/10. -/
theorem cosine_angle_OAB (A B : ℝ × ℝ) (h_A : A = (-3, -4)) (h_B : B = (5, -12)) :
  let O : ℝ × ℝ := (0, 0)
  let AO : ℝ × ℝ := (O.1 - A.1, O.2 - A.2)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let dot_product := AO.1 * AB.1 + AO.2 * AB.2
  let magnitude_AO := Real.sqrt (AO.1^2 + AO.2^2)
  let magnitude_AB := Real.sqrt (AB.1^2 + AB.2^2)
  dot_product / (magnitude_AO * magnitude_AB) = -Real.sqrt 2 / 10 := by
sorry

end NUMINAMATH_CALUDE_cosine_angle_OAB_l3465_346587


namespace NUMINAMATH_CALUDE_john_savings_period_l3465_346572

/-- Calculates the number of years saved given monthly savings, recent expense, and remaining balance -/
def years_saved (monthly_saving : ℕ) (recent_expense : ℕ) (remaining_balance : ℕ) : ℚ :=
  (recent_expense + remaining_balance) / (monthly_saving * 12)

theorem john_savings_period :
  let monthly_saving : ℕ := 25
  let recent_expense : ℕ := 400
  let remaining_balance : ℕ := 200
  years_saved monthly_saving recent_expense remaining_balance = 2 := by sorry

end NUMINAMATH_CALUDE_john_savings_period_l3465_346572


namespace NUMINAMATH_CALUDE_first_month_sale_is_5921_l3465_346596

/-- Calculates the sale in the first month given the sales for months 2 to 6 and the average sale -/
def first_month_sale (sales_2_to_5 : List ℕ) (sale_6 : ℕ) (average : ℕ) : ℕ :=
  6 * average - (sales_2_to_5.sum + sale_6)

/-- Theorem stating that the sale in the first month is 5921 -/
theorem first_month_sale_is_5921 :
  first_month_sale [5468, 5568, 6088, 6433] 5922 5900 = 5921 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_is_5921_l3465_346596


namespace NUMINAMATH_CALUDE_ethanol_percentage_in_fuel_A_l3465_346519

/-- Proves that the percentage of ethanol in fuel A is 12%, given the specified conditions. -/
theorem ethanol_percentage_in_fuel_A : ∀ (tank_capacity fuel_A_volume fuel_B_ethanol_percent total_ethanol : ℝ),
  tank_capacity = 204 →
  fuel_A_volume = 66 →
  fuel_B_ethanol_percent = 16 / 100 →
  total_ethanol = 30 →
  ∃ (fuel_A_ethanol_percent : ℝ),
    fuel_A_ethanol_percent * fuel_A_volume + 
    fuel_B_ethanol_percent * (tank_capacity - fuel_A_volume) = total_ethanol ∧
    fuel_A_ethanol_percent = 12 / 100 :=
by sorry

end NUMINAMATH_CALUDE_ethanol_percentage_in_fuel_A_l3465_346519


namespace NUMINAMATH_CALUDE_A_value_l3465_346558

theorem A_value (a : ℝ) (h : a * (a + 2) = 8 ∨ a^2 + a = 8 - a) :
  2 / (a^2 - 4) - 1 / (a * (a - 2)) = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_A_value_l3465_346558


namespace NUMINAMATH_CALUDE_kendys_initial_balance_l3465_346584

/-- Proves that Kendy's initial account balance was $190 given the conditions of her transfers --/
theorem kendys_initial_balance :
  let mom_transfer : ℕ := 60
  let sister_transfer : ℕ := mom_transfer / 2
  let remaining_balance : ℕ := 100
  let initial_balance : ℕ := remaining_balance + mom_transfer + sister_transfer
  initial_balance = 190 := by
  sorry

end NUMINAMATH_CALUDE_kendys_initial_balance_l3465_346584


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3465_346599

theorem simplify_and_rationalize :
  (Real.sqrt 6 / Real.sqrt 5) * (Real.sqrt 8 / Real.sqrt 9) * (Real.sqrt 10 / Real.sqrt 11) = 4 * Real.sqrt 66 / 33 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3465_346599


namespace NUMINAMATH_CALUDE_tims_income_percentage_l3465_346579

theorem tims_income_percentage (tim mart juan : ℝ) 
  (h1 : mart = tim + 0.6 * tim) 
  (h2 : mart = 0.9599999999999999 * juan) : 
  tim = 0.6 * juan := by sorry

end NUMINAMATH_CALUDE_tims_income_percentage_l3465_346579


namespace NUMINAMATH_CALUDE_integer_tuple_solution_l3465_346528

theorem integer_tuple_solution : 
  ∀ (a b c : ℤ), (a - b)^3 * (a + b)^2 = c^2 + 2*(a - b) + 1 ↔ (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) := by
  sorry

end NUMINAMATH_CALUDE_integer_tuple_solution_l3465_346528


namespace NUMINAMATH_CALUDE_product_quotient_puzzle_l3465_346567

theorem product_quotient_puzzle :
  ∃ (x y t : ℕ+),
    100 ≤ (x * y : ℕ) ∧ (x * y : ℕ) ≤ 999 ∧
    x * y = t^3 ∧
    (x : ℚ) / y = t^2 ∧
    x = 243 ∧ y = 3 :=
by sorry

end NUMINAMATH_CALUDE_product_quotient_puzzle_l3465_346567


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l3465_346592

theorem greatest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, 8 - 3 * (2 * y + 1) > 26 → y ≤ x) ∧ (8 - 3 * (2 * x + 1) > 26) ↔ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l3465_346592


namespace NUMINAMATH_CALUDE_complex_power_equality_l3465_346525

theorem complex_power_equality : (((1 - Complex.I) / Real.sqrt 2) ^ 44 : ℂ) = -1 := by sorry

end NUMINAMATH_CALUDE_complex_power_equality_l3465_346525


namespace NUMINAMATH_CALUDE_college_student_count_l3465_346590

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The total number of students in the college -/
def College.total (c : College) : ℕ := c.boys + c.girls

/-- Theorem: In a college where the ratio of boys to girls is 8:5 and there are 300 girls, 
    the total number of students is 780 -/
theorem college_student_count : 
  ∀ (c : College), 
  c.boys * 5 = c.girls * 8 → 
  c.girls = 300 → 
  c.total = 780 := by
sorry

end NUMINAMATH_CALUDE_college_student_count_l3465_346590


namespace NUMINAMATH_CALUDE_seokgi_candies_l3465_346548

theorem seokgi_candies :
  ∀ (original : ℕ),
  (original : ℚ) * (1/2 : ℚ) * (2/3 : ℚ) = 12 →
  original = 36 := by
  sorry

end NUMINAMATH_CALUDE_seokgi_candies_l3465_346548


namespace NUMINAMATH_CALUDE_decimal_equivalent_of_one_tenth_squared_l3465_346577

theorem decimal_equivalent_of_one_tenth_squared : (1 / 10 : ℚ) ^ 2 = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_of_one_tenth_squared_l3465_346577


namespace NUMINAMATH_CALUDE_find_novel_cost_l3465_346514

def novel_cost (initial_amount lunch_cost remaining_amount : ℚ) : Prop :=
  ∃ (novel_cost : ℚ),
    novel_cost > 0 ∧
    lunch_cost = 2 * novel_cost ∧
    initial_amount - (novel_cost + lunch_cost) = remaining_amount

theorem find_novel_cost :
  novel_cost 50 (2 * 7) 29 :=
sorry

end NUMINAMATH_CALUDE_find_novel_cost_l3465_346514


namespace NUMINAMATH_CALUDE_travel_cost_for_twenty_days_l3465_346569

/-- Calculate the total travel cost for a given number of working days and one-way trip cost. -/
def totalTravelCost (workingDays : ℕ) (oneWayCost : ℕ) : ℕ :=
  workingDays * (2 * oneWayCost)

/-- Theorem: The total travel cost for 20 working days with a one-way cost of $24 is $960. -/
theorem travel_cost_for_twenty_days :
  totalTravelCost 20 24 = 960 := by
  sorry

end NUMINAMATH_CALUDE_travel_cost_for_twenty_days_l3465_346569


namespace NUMINAMATH_CALUDE_trick_decks_total_spent_l3465_346509

/-- The total amount spent by Frank and his friend on trick decks -/
def total_spent (deck_price : ℕ) (frank_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  deck_price * frank_decks + deck_price * friend_decks

/-- Theorem stating the total amount spent by Frank and his friend -/
theorem trick_decks_total_spent :
  total_spent 7 3 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_trick_decks_total_spent_l3465_346509


namespace NUMINAMATH_CALUDE_combined_salaries_of_four_l3465_346589

/-- Given 5 individuals with an average monthly salary and one known salary, 
    prove the sum of the other four salaries. -/
theorem combined_salaries_of_four (average_salary : ℕ) (known_salary : ℕ) 
  (h1 : average_salary = 9000)
  (h2 : known_salary = 5000) :
  4 * average_salary - known_salary = 40000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_of_four_l3465_346589


namespace NUMINAMATH_CALUDE_gcd_10010_15015_l3465_346549

theorem gcd_10010_15015 : Nat.gcd 10010 15015 = 5005 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10010_15015_l3465_346549


namespace NUMINAMATH_CALUDE_collinear_points_sum_l3465_346546

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p₁ p₂ p₃ : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), p₃ = (1 - t₁ - t₂) • p₁ + t₁ • p₂ + t₂ • p₃

/-- If the points (2, a, b), (a, 3, b), and (a, b, 4) are collinear, then a + b = 6. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l3465_346546


namespace NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l3465_346571

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of decks used -/
def num_decks : ℕ := 2

/-- The total number of cards in the combined deck -/
def total_cards : ℕ := standard_deck_size * num_decks

/-- The number of red cards in the combined deck -/
def red_cards : ℕ := standard_deck_size

/-- The expected number of pairs of adjacent red cards -/
def expected_red_pairs : ℚ := 2652 / 103

theorem expected_adjacent_red_pairs :
  let p := red_cards / total_cards
  expected_red_pairs = red_cards * (red_cards - 1) / (total_cards - 1) := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l3465_346571
