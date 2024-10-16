import Mathlib

namespace NUMINAMATH_CALUDE_chicken_cost_l1396_139622

def initial_amount : Int := 55
def banana_packs : Int := 2
def banana_cost : Int := 4
def pear_cost : Int := 2
def asparagus_cost : Int := 6
def remaining_amount : Int := 28

theorem chicken_cost : 
  initial_amount - (banana_packs * banana_cost + pear_cost + asparagus_cost) - remaining_amount = 11 := by
  sorry

end NUMINAMATH_CALUDE_chicken_cost_l1396_139622


namespace NUMINAMATH_CALUDE_fraction_addition_l1396_139638

theorem fraction_addition (d : ℝ) : (5 + 4 * d) / 8 + 3 = (29 + 4 * d) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1396_139638


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_composable_l1396_139688

/-- A fourth-degree polynomial -/
structure FourthDegreePolynomial where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  A_nonzero : A ≠ 0

/-- Condition for a fourth-degree polynomial to be expressible as a composition of two quadratic polynomials -/
def is_composable (f : FourthDegreePolynomial) : Prop :=
  f.D = (f.B * f.C) / (2 * f.A) - (f.B^3) / (8 * f.A^2)

/-- Theorem stating the necessary and sufficient condition for a fourth-degree polynomial 
    to be expressible as a composition of two quadratic polynomials -/
theorem fourth_degree_polynomial_composable (f : FourthDegreePolynomial) :
  (∃ (p q : ℝ → ℝ), (∀ x, f.A * x^4 + f.B * x^3 + f.C * x^2 + f.D * x + f.E = p (q x)) ∧
                     (∃ a b c r s t, p x = a * x^2 + b * x + c ∧
                                     q x = r * x^2 + s * x + t)) ↔
  is_composable f :=
sorry

end NUMINAMATH_CALUDE_fourth_degree_polynomial_composable_l1396_139688


namespace NUMINAMATH_CALUDE_particle_movement_ways_l1396_139694

/-- The number of distinct ways a particle can move on a number line -/
def distinct_ways (total_steps : ℕ) (final_distance : ℕ) : ℕ :=
  Nat.choose total_steps ((total_steps + final_distance) / 2) +
  Nat.choose total_steps ((total_steps - final_distance) / 2)

/-- Theorem stating the number of distinct ways for the given conditions -/
theorem particle_movement_ways :
  distinct_ways 10 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_particle_movement_ways_l1396_139694


namespace NUMINAMATH_CALUDE_store_earnings_is_120_l1396_139617

/-- Represents the store's sales policy and outcomes -/
structure StoreSales where
  pencil_count : ℕ
  eraser_per_pencil : ℕ
  eraser_price : ℚ
  pencil_price : ℚ

/-- Calculates the total earnings from pencil and eraser sales -/
def total_earnings (s : StoreSales) : ℚ :=
  s.pencil_count * s.pencil_price + 
  s.pencil_count * s.eraser_per_pencil * s.eraser_price

/-- Theorem stating that the store's earnings are $120 given the specified conditions -/
theorem store_earnings_is_120 (s : StoreSales) 
  (h1 : s.eraser_per_pencil = 2)
  (h2 : s.eraser_price = 1)
  (h3 : s.pencil_price = 2 * s.eraser_per_pencil * s.eraser_price)
  (h4 : s.pencil_count = 20) : 
  total_earnings s = 120 := by
  sorry

#eval total_earnings { pencil_count := 20, eraser_per_pencil := 2, eraser_price := 1, pencil_price := 4 }

end NUMINAMATH_CALUDE_store_earnings_is_120_l1396_139617


namespace NUMINAMATH_CALUDE_probability_on_square_l1396_139620

/-- A square with 12 evenly spaced points -/
structure SquareWithPoints :=
  (side_length : ℕ)
  (num_points : ℕ)
  (interval : ℝ)

/-- The probability of selecting two points one unit apart -/
def probability_one_unit_apart (s : SquareWithPoints) : ℚ :=
  sorry

/-- The main theorem -/
theorem probability_on_square (s : SquareWithPoints) 
  (h1 : s.side_length = 3)
  (h2 : s.num_points = 12)
  (h3 : s.interval = 1) :
  probability_one_unit_apart s = 2 / 11 :=
sorry

end NUMINAMATH_CALUDE_probability_on_square_l1396_139620


namespace NUMINAMATH_CALUDE_max_cubes_is_six_l1396_139639

/-- Represents a stack of identical wooden cubes -/
structure CubeStack where
  front_view : Nat
  side_view : Nat
  top_view : Nat

/-- The maximum number of cubes in a stack given its views -/
def max_cubes (stack : CubeStack) : Nat :=
  2 * stack.top_view

/-- Theorem stating that the maximum number of cubes in the stack is 6 -/
theorem max_cubes_is_six (stack : CubeStack) 
  (h_top : stack.top_view = 3) : max_cubes stack = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_is_six_l1396_139639


namespace NUMINAMATH_CALUDE_solution_set_of_equations_l1396_139631

theorem solution_set_of_equations (a b c d : ℝ) : 
  (a * b * c + d = 2 ∧
   b * c * d + a = 2 ∧
   c * d * a + b = 2 ∧
   d * a * b + c = 2) ↔ 
  ((a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
   (a = 3 ∧ b = -1 ∧ c = -1 ∧ d = -1) ∨
   (a = -1 ∧ b = 3 ∧ c = -1 ∧ d = -1)) :=
by sorry

#check solution_set_of_equations

end NUMINAMATH_CALUDE_solution_set_of_equations_l1396_139631


namespace NUMINAMATH_CALUDE_fruit_cost_calculation_l1396_139699

/-- The cost of a single orange in dollars -/
def orange_cost : ℚ := (1.27 - 2 * 0.21) / 5

/-- The total cost of six apples and three oranges in dollars -/
def total_cost : ℚ := 6 * 0.21 + 3 * orange_cost

theorem fruit_cost_calculation :
  (2 * 0.21 + 5 * orange_cost = 1.27) →
  total_cost = 1.77 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_calculation_l1396_139699


namespace NUMINAMATH_CALUDE_basketball_league_games_l1396_139614

/-- Calculates the total number of games in a basketball league -/
def total_games (n : ℕ) (regular_games_per_pair : ℕ) (knockout_games_per_team : ℕ) : ℕ :=
  let regular_season_games := (n * (n - 1) / 2) * regular_games_per_pair
  let knockout_games := n * knockout_games_per_team / 2
  regular_season_games + knockout_games

/-- Theorem: In a 12-team basketball league where each team plays 4 games against every other team
    and participates in 2 knockout matches, the total number of games is 276 -/
theorem basketball_league_games :
  total_games 12 4 2 = 276 := by
  sorry

end NUMINAMATH_CALUDE_basketball_league_games_l1396_139614


namespace NUMINAMATH_CALUDE_shape_cuttable_and_rearrangeable_l1396_139676

/-- A shape that can be cut into identical parts and rearranged --/
structure Cuttable where
  area : ℕ
  parts : ℕ
  width : ℕ
  height : ℕ

/-- Predicate to check if a shape is cuttable into equal parts and rearrangeable --/
def is_cuttable_and_rearrangeable (s : Cuttable) : Prop :=
  s.area % s.parts = 0 ∧ 
  s.area = s.width * s.height ∧
  s.area / s.parts * s.parts = s.area

/-- Theorem stating that the given shape can be cut and rearranged as required --/
theorem shape_cuttable_and_rearrangeable : 
  ∃ (s : Cuttable), s.area = 30 ∧ s.parts = 6 ∧ s.width = 5 ∧ s.height = 6 ∧ 
  is_cuttable_and_rearrangeable s :=
sorry

end NUMINAMATH_CALUDE_shape_cuttable_and_rearrangeable_l1396_139676


namespace NUMINAMATH_CALUDE_circle_op_example_l1396_139635

/-- Custom binary operation on real numbers -/
def circle_op (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- The main theorem to prove -/
theorem circle_op_example : circle_op 9 (circle_op 4 3) = 32 := by
  sorry

end NUMINAMATH_CALUDE_circle_op_example_l1396_139635


namespace NUMINAMATH_CALUDE_journey_time_increase_l1396_139674

theorem journey_time_increase (total_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 640 ∧ 
  first_half_speed = 80 ∧ 
  average_speed = 40 →
  let first_half_time := (total_distance / 2) / first_half_speed
  let total_time := total_distance / average_speed
  let second_half_time := total_time - first_half_time
  (second_half_time - first_half_time) / first_half_time = 2 := by
sorry

end NUMINAMATH_CALUDE_journey_time_increase_l1396_139674


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1396_139680

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (b / a = Real.sqrt 3) → 
  (∃ c : ℝ, c = 4 ∧ c^2 = a^2 + b^2) → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 / 12 = 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1396_139680


namespace NUMINAMATH_CALUDE_min_value_problem_l1396_139661

theorem min_value_problem (a b c : ℝ) 
  (eq1 : 3 * a + 2 * b + c = 5)
  (eq2 : 2 * a + b - 3 * c = 1)
  (nonneg_a : a ≥ 0)
  (nonneg_b : b ≥ 0)
  (nonneg_c : c ≥ 0) :
  (∀ a' b' c' : ℝ, 
    3 * a' + 2 * b' + c' = 5 → 
    2 * a' + b' - 3 * c' = 1 → 
    a' ≥ 0 → b' ≥ 0 → c' ≥ 0 → 
    3 * a + b - 7 * c ≤ 3 * a' + b' - 7 * c') ∧
  (3 * a + b - 7 * c = -5/7) :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1396_139661


namespace NUMINAMATH_CALUDE_quadratic_function_bounds_l1396_139625

theorem quadratic_function_bounds (a c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 - c
  (-4 : ℝ) ≤ f 1 ∧ f 1 ≤ -1 ∧ (-1 : ℝ) ≤ f 2 ∧ f 2 ≤ 5 →
  (-1 : ℝ) ≤ f 3 ∧ f 3 ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bounds_l1396_139625


namespace NUMINAMATH_CALUDE_yellow_flowers_killed_correct_l1396_139618

/-- Represents the number of flowers of each color --/
structure FlowerCounts where
  red : ℕ
  yellow : ℕ
  orange : ℕ
  purple : ℕ

/-- Represents the problem parameters --/
structure BouquetProblem where
  seeds_per_color : ℕ
  flowers_per_bouquet : ℕ
  total_bouquets : ℕ
  killed_flowers : FlowerCounts

def yellow_flowers_killed (problem : BouquetProblem) : ℕ :=
  problem.seeds_per_color -
    (problem.total_bouquets * problem.flowers_per_bouquet -
      (problem.seeds_per_color - problem.killed_flowers.red +
       problem.seeds_per_color - problem.killed_flowers.orange +
       problem.seeds_per_color - problem.killed_flowers.purple))

theorem yellow_flowers_killed_correct (problem : BouquetProblem) :
  problem.seeds_per_color = 125 →
  problem.flowers_per_bouquet = 9 →
  problem.total_bouquets = 36 →
  problem.killed_flowers.red = 45 →
  problem.killed_flowers.orange = 30 →
  problem.killed_flowers.purple = 40 →
  yellow_flowers_killed problem = 61 := by
  sorry

#eval yellow_flowers_killed {
  seeds_per_color := 125,
  flowers_per_bouquet := 9,
  total_bouquets := 36,
  killed_flowers := {
    red := 45,
    yellow := 0,  -- This value doesn't affect the calculation
    orange := 30,
    purple := 40
  }
}

end NUMINAMATH_CALUDE_yellow_flowers_killed_correct_l1396_139618


namespace NUMINAMATH_CALUDE_percentage_problem_l1396_139651

theorem percentage_problem (P : ℝ) (x : ℝ) : 
  x = 840 → P * x = 0.15 * 1500 - 15 → P = 0.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1396_139651


namespace NUMINAMATH_CALUDE_sum_of_roots_product_polynomials_l1396_139673

theorem sum_of_roots_product_polynomials :
  let p₁ : Polynomial ℝ := 3 * X^3 + 3 * X^2 - 9 * X + 27
  let p₂ : Polynomial ℝ := 4 * X^3 - 16 * X^2 + 5
  (p₁ * p₂).roots.sum = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_product_polynomials_l1396_139673


namespace NUMINAMATH_CALUDE_equation_solution_l1396_139697

theorem equation_solution : ∃! x : ℚ, 3 * (x - 2) + 1 = x - (2 * x - 1) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1396_139697


namespace NUMINAMATH_CALUDE_car_speed_difference_l1396_139629

theorem car_speed_difference (distance : ℝ) (speed_R : ℝ) : 
  distance = 750 ∧ 
  speed_R = 56.44102863722254 → 
  ∃ (speed_P : ℝ), 
    distance / speed_P = distance / speed_R - 2 ∧ 
    speed_P > speed_R ∧ 
    speed_P - speed_R = 10 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_difference_l1396_139629


namespace NUMINAMATH_CALUDE_cosine_equality_l1396_139604

theorem cosine_equality (n : ℤ) : 0 ≤ n ∧ n ≤ 180 → n = 38 → Real.cos (n * π / 180) = Real.cos (758 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l1396_139604


namespace NUMINAMATH_CALUDE_litter_collection_weight_l1396_139670

/-- The total weight of litter collected by Gina and her neighbors -/
def total_litter_weight (gina_glass_bags : ℕ) (gina_plastic_bags : ℕ)
  (glass_weight : ℕ) (plastic_weight : ℕ) (neighbor_glass_factor : ℕ)
  (neighbor_plastic_factor : ℕ) : ℕ :=
  let gina_glass := gina_glass_bags * glass_weight
  let gina_plastic := gina_plastic_bags * plastic_weight
  let neighbor_glass := neighbor_glass_factor * gina_glass
  let neighbor_plastic := neighbor_plastic_factor * gina_plastic
  gina_glass + gina_plastic + neighbor_glass + neighbor_plastic

/-- Theorem stating the total weight of litter collected -/
theorem litter_collection_weight :
  total_litter_weight 5 3 7 4 120 80 = 5207 := by
  sorry

end NUMINAMATH_CALUDE_litter_collection_weight_l1396_139670


namespace NUMINAMATH_CALUDE_softball_team_savings_l1396_139653

/-- Calculates the total savings for a softball team's uniform purchase with group discount --/
theorem softball_team_savings
  (team_size : ℕ)
  (brand_a_shirt_cost brand_a_pants_cost brand_a_socks_cost : ℚ)
  (brand_b_shirt_cost brand_b_pants_cost brand_b_socks_cost : ℚ)
  (brand_a_customization_cost brand_b_customization_cost : ℚ)
  (brand_a_group_shirt_cost brand_a_group_pants_cost brand_a_group_socks_cost : ℚ)
  (brand_b_group_shirt_cost brand_b_group_pants_cost brand_b_group_socks_cost : ℚ)
  (individual_socks_players non_customized_shirts_players brand_b_socks_players : ℕ)
  (h1 : team_size = 12)
  (h2 : brand_a_shirt_cost = 7.5)
  (h3 : brand_a_pants_cost = 15)
  (h4 : brand_a_socks_cost = 4.5)
  (h5 : brand_b_shirt_cost = 10)
  (h6 : brand_b_pants_cost = 20)
  (h7 : brand_b_socks_cost = 6)
  (h8 : brand_a_customization_cost = 6)
  (h9 : brand_b_customization_cost = 8)
  (h10 : brand_a_group_shirt_cost = 6.5)
  (h11 : brand_a_group_pants_cost = 13)
  (h12 : brand_a_group_socks_cost = 4)
  (h13 : brand_b_group_shirt_cost = 8.5)
  (h14 : brand_b_group_pants_cost = 17)
  (h15 : brand_b_group_socks_cost = 5)
  (h16 : individual_socks_players = 3)
  (h17 : non_customized_shirts_players = 2)
  (h18 : brand_b_socks_players = 1) :
  (team_size * (brand_a_shirt_cost + brand_a_customization_cost + brand_b_pants_cost + brand_a_socks_cost)) -
  (team_size * (brand_a_group_shirt_cost + brand_a_customization_cost + brand_b_group_pants_cost + brand_a_group_socks_cost) +
   individual_socks_players * (brand_a_socks_cost - brand_a_group_socks_cost) -
   non_customized_shirts_players * brand_a_customization_cost +
   brand_b_socks_players * (brand_b_socks_cost - brand_a_group_socks_cost)) = 46.5 := by
  sorry


end NUMINAMATH_CALUDE_softball_team_savings_l1396_139653


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1396_139608

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (2*x + 7 > 3*x + 2 ∧ 2*x - 2 < 2*m) ↔ x < 5) →
  m ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1396_139608


namespace NUMINAMATH_CALUDE_trig_identity_l1396_139666

theorem trig_identity : 4 * Real.cos (15 * π / 180) * Real.cos (75 * π / 180) - Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1396_139666


namespace NUMINAMATH_CALUDE_smallest_five_times_decrease_five_times_decrease_form_no_twelve_times_decrease_divisible_by_k_condition_l1396_139693

def is_valid_number (N : ℕ) : Prop :=
  ∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n)

theorem smallest_five_times_decrease (N : ℕ) :
  is_valid_number N →
  (∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n) ∧ N = 5 * (N % 10^n)) →
  N ≥ 25 :=
sorry

theorem five_times_decrease_form (N : ℕ) :
  is_valid_number N →
  (∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n) ∧ N = 5 * (N % 10^n)) →
  ∃ (m : ℕ), N = 12 * 10^m ∨ N = 24 * 10^m ∨ N = 36 * 10^m ∨ N = 48 * 10^m :=
sorry

theorem no_twelve_times_decrease (N : ℕ) :
  is_valid_number N →
  ¬(∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n) ∧ N = 12 * (N % 10^n)) :=
sorry

theorem divisible_by_k_condition (k : ℕ) :
  (∃ (N : ℕ), is_valid_number N ∧ 
   ∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n) ∧ k ∣ (N % 10^n)) ↔
  ∃ (x a b : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ a + b > 0 ∧ k = x * 2^a * 5^b :=
sorry

end NUMINAMATH_CALUDE_smallest_five_times_decrease_five_times_decrease_form_no_twelve_times_decrease_divisible_by_k_condition_l1396_139693


namespace NUMINAMATH_CALUDE_triangle_inequality_l1396_139602

/-- Given a triangle ABC with area t, perimeter k, and circumradius R, 
    prove that 4tR ≤ (k/3)³ -/
theorem triangle_inequality (t k R : ℝ) (h_positive : t > 0 ∧ k > 0 ∧ R > 0) :
  4 * t * R ≤ (k / 3) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1396_139602


namespace NUMINAMATH_CALUDE_train_platform_length_l1396_139691

/-- The length of the platform given the conditions of the train problem -/
theorem train_platform_length 
  (train_speed : ℝ) 
  (opposite_train_speed : ℝ) 
  (crossing_time : ℝ) 
  (platform_passing_time : ℝ) 
  (h1 : train_speed = 48) 
  (h2 : opposite_train_speed = 42) 
  (h3 : crossing_time = 12) 
  (h4 : platform_passing_time = 45) : 
  ∃ (platform_length : ℝ), 
    (abs (platform_length - 600) < 1) ∧ 
    (platform_length = train_speed * (5/18) * platform_passing_time) :=
sorry


end NUMINAMATH_CALUDE_train_platform_length_l1396_139691


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l1396_139677

/-- Given an ellipse defined by 16(x-2)^2 + 4y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 16 * (x - 2)^2 + 4 * y^2 = 64 ↔ 
      (x - 2)^2 / 4 + y^2 / 16 = 1) →
    (C.1 - 2)^2 / 4 + C.2^2 / 16 = 1 →
    (D.1 - 2)^2 / 4 + D.2^2 / 16 = 1 →
    C.2 = 4 ∨ C.2 = -4 →
    D.1 = 4 ∨ D.1 = 0 →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l1396_139677


namespace NUMINAMATH_CALUDE_warehouse_optimization_l1396_139627

/-- Represents the warehouse dimensions and costs -/
structure Warehouse where
  x : ℝ  -- length of the iron fence (front)
  y : ℝ  -- length of one brick wall (side)
  iron_cost : ℝ := 40  -- cost per meter of iron fence
  brick_cost : ℝ := 45  -- cost per meter of brick wall
  top_cost : ℝ := 20  -- cost per square meter of the top
  budget : ℝ := 3200  -- total budget

/-- The total cost of the warehouse -/
def total_cost (w : Warehouse) : ℝ :=
  w.iron_cost * w.x + 2 * w.brick_cost * w.y + w.top_cost * w.x * w.y

/-- The area of the warehouse -/
def area (w : Warehouse) : ℝ :=
  w.x * w.y

/-- Theorem stating the maximum area and optimal dimensions -/
theorem warehouse_optimization (w : Warehouse) :
  (∀ w' : Warehouse, total_cost w' ≤ w.budget → area w' ≤ 100) ∧
  (∃ w' : Warehouse, total_cost w' ≤ w.budget ∧ area w' = 100) ∧
  (area w = 100 → total_cost w ≤ w.budget → w.x = 15) :=
sorry

end NUMINAMATH_CALUDE_warehouse_optimization_l1396_139627


namespace NUMINAMATH_CALUDE_congruence_solution_l1396_139641

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 14589 [ZMOD 15] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1396_139641


namespace NUMINAMATH_CALUDE_complex_product_real_implies_a_equals_negative_one_l1396_139679

theorem complex_product_real_implies_a_equals_negative_one (a : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (↑(1 + a * Complex.I) * ↑(1 + Complex.I) : ℂ).im = 0 →
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_implies_a_equals_negative_one_l1396_139679


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1396_139665

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n a n / sum_n b n = (7 * n + 1 : ℚ) / (4 * n + 27)) →
  a.a 6 / b.a 6 = 78 / 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1396_139665


namespace NUMINAMATH_CALUDE_greatest_y_value_l1396_139663

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -9) : 
  ∀ z : ℤ, (∃ w : ℤ, w * z + 3 * w + 2 * z = -9) → z ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_y_value_l1396_139663


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l1396_139654

/-- The volume of a cube with space diagonal 3√3 is 27 -/
theorem cube_volume_from_space_diagonal :
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 3 * Real.sqrt 3 → s^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l1396_139654


namespace NUMINAMATH_CALUDE_compute_expression_l1396_139659

theorem compute_expression : 16 * (125 / 2 + 25 / 4 + 9 / 16 + 1) = 1125 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1396_139659


namespace NUMINAMATH_CALUDE_newer_truck_distance_l1396_139685

theorem newer_truck_distance (old_distance : ℝ) (percentage_increase : ℝ) : 
  old_distance = 150 → percentage_increase = 0.3 → 
  old_distance * (1 + percentage_increase) = 195 := by
  sorry

end NUMINAMATH_CALUDE_newer_truck_distance_l1396_139685


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1396_139626

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (5 * x + 13) = 15 → x = 212 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1396_139626


namespace NUMINAMATH_CALUDE_fish_distribution_l1396_139647

-- Define the number of people
def num_people : ℕ := 3

-- Define the number of eyes Oomyapeck eats
def oomyapeck_eyes : ℕ := 22

-- Define the number of eyes given to the dog
def dog_eyes : ℕ := 2

-- Define the number of eyes per fish
def eyes_per_fish : ℕ := 2

-- Theorem statement
theorem fish_distribution (total_eyes : ℕ) (total_fish : ℕ) (fish_per_person : ℕ) :
  total_eyes = oomyapeck_eyes + dog_eyes →
  total_fish = total_eyes / eyes_per_fish →
  fish_per_person = total_fish / num_people →
  fish_per_person = 4 := by
  sorry


end NUMINAMATH_CALUDE_fish_distribution_l1396_139647


namespace NUMINAMATH_CALUDE_sugar_spill_ratio_l1396_139657

/-- Proves that the ratio of sugar that fell to the ground to the sugar in the torn bag before it fell is 1:2 -/
theorem sugar_spill_ratio (initial_sugar : ℕ) (num_bags : ℕ) (remaining_sugar : ℕ) : 
  initial_sugar = 24 →
  num_bags = 4 →
  remaining_sugar = 21 →
  (initial_sugar - remaining_sugar) * 2 = initial_sugar / num_bags :=
by
  sorry

#check sugar_spill_ratio

end NUMINAMATH_CALUDE_sugar_spill_ratio_l1396_139657


namespace NUMINAMATH_CALUDE_arbor_day_saplings_l1396_139633

theorem arbor_day_saplings 
  (rate_A rate_B : ℚ) 
  (saplings_A saplings_B : ℕ) : 
  rate_A = (3 : ℚ) / 4 * rate_B → 
  saplings_B = saplings_A + 36 → 
  saplings_A + saplings_B = 252 := by
sorry

end NUMINAMATH_CALUDE_arbor_day_saplings_l1396_139633


namespace NUMINAMATH_CALUDE_intersection_at_diametrically_opposite_points_l1396_139678

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure ThreeCircles where
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle
  tangent_point : ℝ × ℝ

def are_touching (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  dist c1.center p = c1.radius ∧
  dist c2.center p = c2.radius ∧
  dist c1.center c2.center = c1.radius + c2.radius

def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  dist c.center p = c.radius

def are_diametrically_opposite (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  dist p1 p2 = 2 * c.radius

theorem intersection_at_diametrically_opposite_points
  (tc : ThreeCircles)
  (h1 : tc.circle1.radius = tc.circle2.radius)
  (h2 : tc.circle2.radius = tc.circle3.radius)
  (h3 : are_touching tc.circle1 tc.circle2 tc.tangent_point)
  (h4 : passes_through tc.circle3 tc.tangent_point) :
  ∃ (p1 p2 : ℝ × ℝ),
    p1 ≠ tc.tangent_point ∧
    p2 ≠ tc.tangent_point ∧
    passes_through tc.circle1 p1 ∧
    passes_through tc.circle2 p2 ∧
    passes_through tc.circle3 p1 ∧
    passes_through tc.circle3 p2 ∧
    are_diametrically_opposite tc.circle3 p1 p2 :=
  sorry

end NUMINAMATH_CALUDE_intersection_at_diametrically_opposite_points_l1396_139678


namespace NUMINAMATH_CALUDE_total_gift_combinations_l1396_139646

/-- The number of different gift packaging combinations -/
def gift_combinations (wrapping_paper : ℕ) (ribbon : ℕ) (gift_card : ℕ) (bow : ℕ) : ℕ :=
  wrapping_paper * ribbon * gift_card * bow

/-- Theorem stating that the total number of gift packaging combinations is 400 -/
theorem total_gift_combinations :
  gift_combinations 10 4 5 2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_total_gift_combinations_l1396_139646


namespace NUMINAMATH_CALUDE_parabola_shift_l1396_139686

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The shifted parabola function -/
def shifted_parabola (x : ℝ) : ℝ := (x-4)^2 - 2

/-- Theorem stating that the shifted parabola is equivalent to 
    shifting the original parabola 1 unit right and 2 units up -/
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 1) + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l1396_139686


namespace NUMINAMATH_CALUDE_line_through_point_l1396_139607

theorem line_through_point (b : ℝ) : 
  (∀ x y : ℝ, b*x + (b-1)*y = b^2 - 1 → (x = 2 ∧ y = -5)) → 
  (b = (-3 + Real.sqrt 33) / 2 ∨ b = (-3 - Real.sqrt 33) / 2) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_l1396_139607


namespace NUMINAMATH_CALUDE_correct_observation_value_l1396_139615

theorem correct_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (wrong_value : ℝ) 
  (corrected_mean : ℝ) 
  (h1 : n = 50) 
  (h2 : initial_mean = 36) 
  (h3 : wrong_value = 23) 
  (h4 : corrected_mean = 36.5) : 
  ∃ (correct_value : ℝ), 
    (n : ℝ) * corrected_mean = (n : ℝ) * initial_mean - wrong_value + correct_value ∧ 
    correct_value = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_observation_value_l1396_139615


namespace NUMINAMATH_CALUDE_distance_between_Disney_and_London_l1396_139669

/-- The distance between lake Disney and lake London --/
def distance_Disney_London : ℝ := 60

/-- The number of migrating birds --/
def num_birds : ℕ := 20

/-- The distance between lake Jim and lake Disney --/
def distance_Jim_Disney : ℝ := 50

/-- The combined distance traveled by all birds in two seasons --/
def total_distance : ℝ := 2200

theorem distance_between_Disney_and_London :
  distance_Disney_London = 
    (total_distance - num_birds * distance_Jim_Disney) / num_birds :=
by sorry

end NUMINAMATH_CALUDE_distance_between_Disney_and_London_l1396_139669


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l1396_139684

/-- The radius of a circle tangent to an ellipse --/
theorem tangent_circle_radius (a b c : ℝ) (h1 : a = 7) (h2 : b = 5) (h3 : c = (a^2 - b^2).sqrt) : 
  ∃ r : ℝ, r = 4 ∧ 
  (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) → ((x - c)^2 + y^2 ≤ r^2)) ∧
  (∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ (x - c)^2 + y^2 = r^2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l1396_139684


namespace NUMINAMATH_CALUDE_inverse_proportion_quadrants_l1396_139613

/-- An inverse proportion function passing through (3, -2) lies in the second and fourth quadrants -/
theorem inverse_proportion_quadrants :
  ∀ (k : ℝ), k ≠ 0 →
  (∃ (f : ℝ → ℝ), (∀ x, x ≠ 0 → f x = k / x) ∧ f 3 = -2) →
  (∀ x y, (x > 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_quadrants_l1396_139613


namespace NUMINAMATH_CALUDE_gcd_of_16434_24651_43002_l1396_139611

theorem gcd_of_16434_24651_43002 : Nat.gcd 16434 (Nat.gcd 24651 43002) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_16434_24651_43002_l1396_139611


namespace NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l1396_139648

theorem triangle_perimeter_impossibility (a b x : ℝ) (h1 : a = 12) (h2 : b = 30) : 
  (a + b + x = 58 ∨ a + b + x = 85) → ¬(a + b > x ∧ a + x > b ∧ b + x > a) :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l1396_139648


namespace NUMINAMATH_CALUDE_strawberry_harvest_l1396_139644

/-- Proves that the number of strawberries harvested from each plant is 14 --/
theorem strawberry_harvest (
  strawberry_plants : ℕ)
  (tomato_plants : ℕ)
  (tomatoes_per_plant : ℕ)
  (fruits_per_basket : ℕ)
  (strawberry_basket_price : ℕ)
  (tomato_basket_price : ℕ)
  (total_revenue : ℕ)
  (h1 : strawberry_plants = 5)
  (h2 : tomato_plants = 7)
  (h3 : tomatoes_per_plant = 16)
  (h4 : fruits_per_basket = 7)
  (h5 : strawberry_basket_price = 9)
  (h6 : tomato_basket_price = 6)
  (h7 : total_revenue = 186) :
  (total_revenue - tomato_basket_price * (tomato_plants * tomatoes_per_plant / fruits_per_basket)) / strawberry_basket_price * fruits_per_basket / strawberry_plants = 14 :=
by sorry

end NUMINAMATH_CALUDE_strawberry_harvest_l1396_139644


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angles_divisible_by_nine_l1396_139603

theorem regular_polygon_interior_angles_divisible_by_nine :
  (∃ (S : Finset ℕ), S.card = 5 ∧
    (∀ n ∈ S, 3 ≤ n ∧ n ≤ 15 ∧ (180 - 360 / n) % 9 = 0) ∧
    (∀ n, 3 ≤ n → n ≤ 15 → (180 - 360 / n) % 9 = 0 → n ∈ S)) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angles_divisible_by_nine_l1396_139603


namespace NUMINAMATH_CALUDE_gcd_with_35_is_7_l1396_139642

theorem gcd_with_35_is_7 : 
  ∃ (s : Finset Nat), s = {n : Nat | 70 < n ∧ n < 90 ∧ Nat.gcd 35 n = 7} ∧ s = {77, 84} := by
  sorry

end NUMINAMATH_CALUDE_gcd_with_35_is_7_l1396_139642


namespace NUMINAMATH_CALUDE_circle_tangent_and_secant_l1396_139640

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 3 = 0

-- Define point M
def point_M : ℝ × ℝ := (4, -8)

-- Define the length of AB
def length_AB : ℝ := 4

theorem circle_tangent_and_secant :
  ∃ (tangent_length : ℝ) (line_DE : ℝ → ℝ → Prop) (line_AB : ℝ → ℝ → Prop),
    -- The length of the tangent from M to C is 3√5
    tangent_length = 3 * Real.sqrt 5 ∧
    -- The equation of line DE is 2x-7y-19=0
    (∀ x y, line_DE x y ↔ 2*x - 7*y - 19 = 0) ∧
    -- The equation of line AB is either 45x+28y+44=0 or x=4
    (∀ x y, line_AB x y ↔ (45*x + 28*y + 44 = 0 ∨ x = 4)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_and_secant_l1396_139640


namespace NUMINAMATH_CALUDE_math_problems_l1396_139675

theorem math_problems :
  (270 * 9 = 2430) ∧
  (735 / 7 = 105) ∧
  (99 * 9 = 891) :=
by sorry

end NUMINAMATH_CALUDE_math_problems_l1396_139675


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l1396_139619

theorem cube_sum_theorem (p q r : ℝ) 
  (h1 : p + q + r = 4) 
  (h2 : p * q + q * r + r * p = 7) 
  (h3 : p * q * r = -10) : 
  p^3 + q^3 + r^3 = 154 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l1396_139619


namespace NUMINAMATH_CALUDE_slant_height_neq_base_side_l1396_139656

/-- Represents a regular hexagonal pyramid --/
structure RegularHexagonalPyramid where
  r : ℝ  -- side length of each equilateral triangle in the base
  h : ℝ  -- height of the pyramid
  l : ℝ  -- slant height (lateral edge) of the pyramid
  r_pos : r > 0
  h_pos : h > 0
  l_pos : l > 0
  pythagorean : h^2 + r^2 = l^2

/-- Theorem: In a regular hexagonal pyramid, the slant height cannot be equal to the side length of the base hexagon --/
theorem slant_height_neq_base_side (p : RegularHexagonalPyramid) : p.l ≠ p.r := by
  sorry


end NUMINAMATH_CALUDE_slant_height_neq_base_side_l1396_139656


namespace NUMINAMATH_CALUDE_divisibility_implies_power_l1396_139687

theorem divisibility_implies_power (m n : ℕ+) 
  (h : (m * n) ∣ (m ^ 2010 + n ^ 2010 + n)) :
  ∃ k : ℕ+, n = k ^ 2010 := by sorry

end NUMINAMATH_CALUDE_divisibility_implies_power_l1396_139687


namespace NUMINAMATH_CALUDE_share_face_value_l1396_139681

/-- Given a share with a 9% dividend rate and a market value of Rs. 42,
    prove that the face value is Rs. 56 if an investor wants a 12% return. -/
theorem share_face_value (dividend_rate : ℝ) (market_value : ℝ) (desired_return : ℝ) :
  dividend_rate = 0.09 →
  market_value = 42 →
  desired_return = 0.12 →
  ∃ (face_value : ℝ), face_value = 56 ∧ dividend_rate * face_value = desired_return * market_value :=
by
  sorry

#check share_face_value

end NUMINAMATH_CALUDE_share_face_value_l1396_139681


namespace NUMINAMATH_CALUDE_alex_chocolates_l1396_139672

theorem alex_chocolates : 
  ∀ n : ℕ, n ≥ 150 ∧ n % 19 = 17 → n ≥ 150 := by
  sorry

end NUMINAMATH_CALUDE_alex_chocolates_l1396_139672


namespace NUMINAMATH_CALUDE_intersection_line_circle_l1396_139696

theorem intersection_line_circle (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 + A.2 = a) ∧ (B.1 + B.2 = a) ∧
    (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧
    ((A.1 + B.1)^2 + (A.2 + B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2)) →
  (a = 2 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l1396_139696


namespace NUMINAMATH_CALUDE_parabola_roots_and_point_below_axis_l1396_139609

/-- A parabola with a point below the x-axis has two distinct real roots, and the x-coordinate of the point is between these roots. -/
theorem parabola_roots_and_point_below_axis 
  (p q x₀ : ℝ) 
  (h_below : x₀^2 + p*x₀ + q < 0) :
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 + p*x₁ + q = 0) ∧ 
    (x₂^2 + p*x₂ + q = 0) ∧ 
    (x₁ < x₀) ∧ 
    (x₀ < x₂) ∧ 
    (x₁ ≠ x₂) := by
  sorry

end NUMINAMATH_CALUDE_parabola_roots_and_point_below_axis_l1396_139609


namespace NUMINAMATH_CALUDE_solve_for_n_l1396_139637

variable (n : ℝ)

def f (x : ℝ) : ℝ := x^2 - 3*x + n
def g (x : ℝ) : ℝ := x^2 - 3*x + 5*n

theorem solve_for_n : 3 * f 3 = 2 * g 3 → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_n_l1396_139637


namespace NUMINAMATH_CALUDE_circumradius_special_triangle_l1396_139695

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at vertex A
def angle_at_A (t : Triangle) : ℝ := sorry

-- Define the radius of the circumscribed circle
def circumradius (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem circumradius_special_triangle (t : Triangle) :
  angle_at_A t = π / 3 ∧
  distance t.B (incenter t) = 3 ∧
  distance t.C (incenter t) = 4 →
  circumradius t = Real.sqrt (37 / 3) := by
  sorry

end NUMINAMATH_CALUDE_circumradius_special_triangle_l1396_139695


namespace NUMINAMATH_CALUDE_lagrange_four_square_theorem_l1396_139664

-- Define the property of being expressible as the sum of four squares
def SumOfFourSquares (n : ℕ) : Prop :=
  ∃ a b c d : ℤ, n = a^2 + b^2 + c^2 + d^2

-- State Lagrange's Four Square Theorem
theorem lagrange_four_square_theorem :
  ∀ n : ℕ, SumOfFourSquares n :=
by
  sorry

-- State the given conditions
axiom odd_prime_four_squares :
  ∀ p : ℕ, Nat.Prime p → p % 2 = 1 → SumOfFourSquares p

axiom two_four_squares : SumOfFourSquares 2

axiom product_four_squares :
  ∀ a b : ℕ, SumOfFourSquares a → SumOfFourSquares b → SumOfFourSquares (a * b)

end NUMINAMATH_CALUDE_lagrange_four_square_theorem_l1396_139664


namespace NUMINAMATH_CALUDE_inequality_not_always_hold_l1396_139623

theorem inequality_not_always_hold (a b : ℝ) (h : a > b ∧ b > 0) : 
  ¬ ∀ c : ℝ, a * c > b * c :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_hold_l1396_139623


namespace NUMINAMATH_CALUDE_solve_equation_l1396_139649

theorem solve_equation (B : ℝ) : 
  80 - (5 - (6 + 2 * (B - 8 - 5))) = 89 ↔ B = 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1396_139649


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1396_139698

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) / (1 - 2 * Complex.I) = Complex.I * b) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1396_139698


namespace NUMINAMATH_CALUDE_quadratic_root_value_l1396_139601

theorem quadratic_root_value (k : ℝ) : 
  (∀ x : ℂ, 5 * x^2 + 7 * x + k = 0 ↔ x = Complex.mk (-7/10) ((Real.sqrt 399)/10) ∨ x = Complex.mk (-7/10) (-(Real.sqrt 399)/10)) →
  k = 22.4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l1396_139601


namespace NUMINAMATH_CALUDE_miss_adamson_paper_usage_l1396_139624

/-- Calculates the total number of sheets of paper used by a teacher for all students --/
def total_sheets_of_paper (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) : ℕ :=
  num_classes * students_per_class * sheets_per_student

/-- Proves that Miss Adamson will use 400 sheets of paper for all her students --/
theorem miss_adamson_paper_usage :
  total_sheets_of_paper 4 20 5 = 400 := by
  sorry

#eval total_sheets_of_paper 4 20 5

end NUMINAMATH_CALUDE_miss_adamson_paper_usage_l1396_139624


namespace NUMINAMATH_CALUDE_point_M_coordinates_l1396_139632

/-- Given a line MN with slope 2, point N at (1, -1), and point M on the line y = x + 1,
    prove that the coordinates of point M are (4, 5). -/
theorem point_M_coordinates :
  let slope_MN : ℝ := 2
  let N : ℝ × ℝ := (1, -1)
  let M : ℝ × ℝ := (x, y)
  (y = x + 1) →  -- M lies on y = x + 1
  ((y - N.2) / (x - N.1) = slope_MN) →  -- slope formula
  M = (4, 5) := by
sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l1396_139632


namespace NUMINAMATH_CALUDE_propositions_true_l1396_139605

-- Define reciprocals
def reciprocals (x y : ℝ) : Prop := x * y = 1

-- Define the quadratic equation
def has_real_roots (b : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 2*b*x + b^2 + b = 0

theorem propositions_true :
  (∀ x y : ℝ, reciprocals x y → x * y = 1) ∧
  (∀ b : ℝ, ¬(has_real_roots b) → b > -1) :=
sorry

end NUMINAMATH_CALUDE_propositions_true_l1396_139605


namespace NUMINAMATH_CALUDE_certain_number_problem_l1396_139616

theorem certain_number_problem (x : ℝ) (certain_number : ℝ) 
  (h1 : certain_number * x = 675)
  (h2 : x = 27) : 
  certain_number = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1396_139616


namespace NUMINAMATH_CALUDE_cricketer_average_after_22nd_inning_l1396_139689

/-- Represents the average score of a cricketer before the 22nd inning -/
def initial_average : ℝ := sorry

/-- The score made by the cricketer in the 22nd inning -/
def score_22nd_inning : ℝ := 134

/-- The increase in average after the 22nd inning -/
def average_increase : ℝ := 3.5

/-- The number of innings played before the 22nd inning -/
def previous_innings : ℕ := 21

/-- The total number of innings including the 22nd inning -/
def total_innings : ℕ := 22

/-- Calculates the new average after the 22nd inning -/
def new_average : ℝ := initial_average + average_increase

/-- Theorem stating that the new average after the 22nd inning is 60.5 -/
theorem cricketer_average_after_22nd_inning : 
  (previous_innings : ℝ) * initial_average + score_22nd_inning = 
    new_average * (total_innings : ℝ) ∧ new_average = 60.5 := by sorry

end NUMINAMATH_CALUDE_cricketer_average_after_22nd_inning_l1396_139689


namespace NUMINAMATH_CALUDE_survey_selection_count_l1396_139636

/-- Represents the total number of households selected in a stratified sampling survey. -/
def total_selected (total_households : ℕ) (middle_income : ℕ) (low_income : ℕ) (high_income_selected : ℕ) : ℕ :=
  (high_income_selected * total_households) / (total_households - middle_income - low_income)

/-- Theorem stating that the total number of households selected in the survey is 24. -/
theorem survey_selection_count :
  total_selected 480 200 160 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_survey_selection_count_l1396_139636


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l1396_139612

theorem sqrt_sum_comparison : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l1396_139612


namespace NUMINAMATH_CALUDE_valid_divisions_count_l1396_139690

/-- Represents a rectangle on the grid -/
structure Rectangle where
  x : Nat
  y : Nat
  width : Nat
  height : Nat

/-- Represents a division of the grid into 5 rectangles -/
structure GridDivision where
  center : Rectangle
  top : Rectangle
  bottom : Rectangle
  left : Rectangle
  right : Rectangle

/-- Checks if a rectangle touches the perimeter of an 11x11 grid -/
def touchesPerimeter (r : Rectangle) : Bool :=
  r.x = 0 || r.y = 0 || r.x + r.width = 11 || r.y + r.height = 11

/-- Checks if a grid division is valid -/
def isValidDivision (d : GridDivision) : Bool :=
  d.center.x > 0 && d.center.y > 0 && 
  d.center.x + d.center.width < 11 && 
  d.center.y + d.center.height < 11 &&
  touchesPerimeter d.top &&
  touchesPerimeter d.bottom &&
  touchesPerimeter d.left &&
  touchesPerimeter d.right

/-- Counts the number of valid grid divisions -/
def countValidDivisions : Nat :=
  sorry

theorem valid_divisions_count : countValidDivisions = 81 := by
  sorry

end NUMINAMATH_CALUDE_valid_divisions_count_l1396_139690


namespace NUMINAMATH_CALUDE_skittles_theorem_l1396_139682

def skittles_problem (brandon_initial bonnie_initial brandon_loss : ℕ) : Prop :=
  let brandon_after_loss := brandon_initial - brandon_loss
  let combined := brandon_after_loss + bonnie_initial
  let each_share := combined / 4
  let chloe_initial := each_share
  let dylan_initial := each_share
  let chloe_to_dylan := chloe_initial / 2
  let dylan_after_chloe := dylan_initial + chloe_to_dylan
  let dylan_to_bonnie := dylan_after_chloe / 3
  let dylan_final := dylan_after_chloe - dylan_to_bonnie
  dylan_final = 22

theorem skittles_theorem : skittles_problem 96 4 9 := by sorry

end NUMINAMATH_CALUDE_skittles_theorem_l1396_139682


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_1720_l1396_139630

theorem sqrt_product_plus_one_equals_1720 : 
  Real.sqrt ((43 : ℝ) * 42 * 41 * 40 + 1) = 1720 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_1720_l1396_139630


namespace NUMINAMATH_CALUDE_andys_school_distance_l1396_139650

/-- The distance between Andy's house and school, given the total distance walked and the distance to the market. -/
theorem andys_school_distance (total_distance : ℕ) (market_distance : ℕ) (h1 : total_distance = 140) (h2 : market_distance = 40) : 
  let school_distance := (total_distance - market_distance) / 2
  school_distance = 50 := by sorry

end NUMINAMATH_CALUDE_andys_school_distance_l1396_139650


namespace NUMINAMATH_CALUDE_apple_distribution_l1396_139671

theorem apple_distribution (total_apples : ℕ) (num_people : ℕ) (apples_per_person : ℕ) :
  total_apples = 15 →
  num_people = 3 →
  apples_per_person * num_people ≤ total_apples →
  apples_per_person = total_apples / num_people →
  apples_per_person = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1396_139671


namespace NUMINAMATH_CALUDE_lumberjack_trees_l1396_139610

theorem lumberjack_trees (logs_per_tree : ℕ) (firewood_per_log : ℕ) (total_firewood : ℕ) :
  logs_per_tree = 4 →
  firewood_per_log = 5 →
  total_firewood = 500 →
  total_firewood / (logs_per_tree * firewood_per_log) = 25 :=
by sorry

end NUMINAMATH_CALUDE_lumberjack_trees_l1396_139610


namespace NUMINAMATH_CALUDE_hyperbola_real_semiaxis_range_l1396_139600

/-- The range of values for the length of the real semi-axis of a hyperbola -/
theorem hyperbola_real_semiaxis_range (a b : ℝ) (c : ℝ) :
  a > 0 →
  b > 0 →
  c = 4 →
  c^2 = a^2 + b^2 →
  b / a < Real.sqrt 3 →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y / x = Real.tan (60 * π / 180)) →
  2 < a ∧ a < 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_real_semiaxis_range_l1396_139600


namespace NUMINAMATH_CALUDE_f_5_solution_set_l1396_139658

def f (x : ℝ) : ℝ := x^2 + 12*x + 30

def f_5 (x : ℝ) : ℝ := f (f (f (f (f x))))

theorem f_5_solution_set :
  ∀ x : ℝ, f_5 x = 0 ↔ x = -6 - (6 : ℝ)^(1/32) ∨ x = -6 + (6 : ℝ)^(1/32) := by
  sorry

end NUMINAMATH_CALUDE_f_5_solution_set_l1396_139658


namespace NUMINAMATH_CALUDE_triangle_minimum_shortest_side_l1396_139643

theorem triangle_minimum_shortest_side :
  ∀ a b : ℕ,
  a < b ∧ b < 3 * a →  -- Condition for unequal sides
  a + b + 3 * a = 120 →  -- Total number of matches
  a ≥ 18 →  -- Minimum value of shortest side
  ∃ (a₀ : ℕ), a₀ = 18 ∧ 
    ∃ (b₀ : ℕ), a₀ < b₀ ∧ b₀ < 3 * a₀ ∧ 
    a₀ + b₀ + 3 * a₀ = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_minimum_shortest_side_l1396_139643


namespace NUMINAMATH_CALUDE_max_value_inequality_l1396_139652

theorem max_value_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + 2*b + 3*c = 6) : 
  Real.sqrt (a + 1) + Real.sqrt (2*b + 1) + Real.sqrt (3*c + 1) ≤ 3 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1396_139652


namespace NUMINAMATH_CALUDE_divisible_by_133_l1396_139621

theorem divisible_by_133 (n : ℕ) : ∃ k : ℤ, (11 : ℤ)^(n+2) + (12 : ℤ)^(2*n+1) = 133 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_133_l1396_139621


namespace NUMINAMATH_CALUDE_solution_p_percentage_l1396_139645

/-- Represents a solution with lemonade and carbonated water. -/
structure Solution where
  lemonade : ℝ
  carbonated_water : ℝ
  sum_to_one : lemonade + carbonated_water = 1

/-- Represents a mixture of two solutions. -/
structure Mixture (p q : Solution) where
  p_volume : ℝ
  q_volume : ℝ
  p_volume_nonneg : 0 ≤ p_volume
  q_volume_nonneg : 0 ≤ q_volume
  total_carbonated_water_percent : ℝ
  total_carbonated_water_eq : 
    total_carbonated_water_percent * (p_volume + q_volume) = 
    p.carbonated_water * p_volume + q.carbonated_water * q_volume

/-- The main theorem stating the percentage of solution P in the mixture. -/
theorem solution_p_percentage 
  (p : Solution) 
  (q : Solution) 
  (mixture : Mixture p q) 
  (hp : p.carbonated_water = 0.8) 
  (hq : q.carbonated_water = 0.55) 
  (hm : mixture.total_carbonated_water_percent = 0.65) :
  mixture.p_volume / (mixture.p_volume + mixture.q_volume) = 0.4 := by
  sorry


end NUMINAMATH_CALUDE_solution_p_percentage_l1396_139645


namespace NUMINAMATH_CALUDE_area_of_encompassing_rectangle_l1396_139606

/-- Given two identical rectangles with intersecting extended sides, prove the area of the encompassing rectangle. -/
theorem area_of_encompassing_rectangle 
  (area_BNHM : ℝ) 
  (area_MBCK : ℝ) 
  (area_MLGH : ℝ) 
  (h1 : area_BNHM = 12)
  (h2 : area_MBCK = 63)
  (h3 : area_MLGH = 28) : 
  ∃ (area_IFJD : ℝ), area_IFJD = 418 := by
sorry

end NUMINAMATH_CALUDE_area_of_encompassing_rectangle_l1396_139606


namespace NUMINAMATH_CALUDE_minimize_y_l1396_139692

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^3 + (x - b)^3

/-- The theorem stating that (a+b)/2 minimizes y -/
theorem minimize_y (a b : ℝ) :
  ∃ (x : ℝ), ∀ (z : ℝ), y z a b ≥ y x a b ∧ x = (a + b) / 2 :=
sorry

end NUMINAMATH_CALUDE_minimize_y_l1396_139692


namespace NUMINAMATH_CALUDE_lines_symmetric_about_y_axis_l1396_139668

/-- Two lines are symmetric about the y-axis if and only if their coefficients satisfy certain conditions -/
theorem lines_symmetric_about_y_axis 
  (m n p : ℝ) : 
  (∀ x y : ℝ, x + m * y + 5 = 0 ↔ x + n * y + p = 0) ∧ 
  (∀ x y : ℝ, x + m * y + 5 = 0 ↔ -x + n * y + p = 0) ↔ 
  m = -n ∧ p = -5 := by sorry

end NUMINAMATH_CALUDE_lines_symmetric_about_y_axis_l1396_139668


namespace NUMINAMATH_CALUDE_a_fourth_plus_b_fourth_l1396_139628

theorem a_fourth_plus_b_fourth (a b : ℝ) 
  (h1 : a^2 - b^2 = 5) 
  (h2 : a * b = 2) : 
  a^4 + b^4 = 833 := by sorry

end NUMINAMATH_CALUDE_a_fourth_plus_b_fourth_l1396_139628


namespace NUMINAMATH_CALUDE_date_books_ordered_l1396_139660

theorem date_books_ordered (calendar_cost date_book_cost : ℚ) 
  (total_items : ℕ) (total_spent : ℚ) :
  calendar_cost = 3/4 →
  date_book_cost = 1/2 →
  total_items = 500 →
  total_spent = 300 →
  ∃ (calendars date_books : ℕ),
    calendars + date_books = total_items ∧
    calendar_cost * calendars + date_book_cost * date_books = total_spent ∧
    date_books = 300 := by
sorry

end NUMINAMATH_CALUDE_date_books_ordered_l1396_139660


namespace NUMINAMATH_CALUDE_frank_miles_proof_l1396_139662

/-- The number of miles Jim ran in 2 hours -/
def jim_miles : ℝ := 16

/-- The number of hours Jim and Frank ran -/
def total_hours : ℝ := 2

/-- The difference in miles per hour between Frank and Jim -/
def frank_jim_diff : ℝ := 2

/-- Frank's total miles run in 2 hours -/
def frank_total_miles : ℝ := 20

theorem frank_miles_proof :
  frank_total_miles = (jim_miles / total_hours + frank_jim_diff) * total_hours :=
by sorry

end NUMINAMATH_CALUDE_frank_miles_proof_l1396_139662


namespace NUMINAMATH_CALUDE_swim_time_proof_l1396_139634

/-- Proves that the time taken to swim downstream and upstream is 6 hours each -/
theorem swim_time_proof (downstream_distance : ℝ) (upstream_distance : ℝ) 
  (still_water_speed : ℝ) (h1 : downstream_distance = 30) 
  (h2 : upstream_distance = 18) (h3 : still_water_speed = 4) :
  ∃ (t : ℝ) (current_speed : ℝ), 
    t = downstream_distance / (still_water_speed + current_speed) ∧
    t = upstream_distance / (still_water_speed - current_speed) ∧
    t = 6 := by
  sorry

#check swim_time_proof

end NUMINAMATH_CALUDE_swim_time_proof_l1396_139634


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l1396_139683

theorem quadratic_complete_square (a b c : ℝ) (h : a = 4 ∧ b = -16 ∧ c = -200) :
  ∃ q t : ℝ, (∀ x, a * x^2 + b * x + c = 0 ↔ (x + q)^2 = t) ∧ t = 54 :=
sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l1396_139683


namespace NUMINAMATH_CALUDE_paper_goods_cost_l1396_139667

/-- Given that 100 paper plates and 200 paper cups cost $6.00, 
    prove that 20 paper plates and 40 paper cups cost $1.20 -/
theorem paper_goods_cost (plate_cost cup_cost : ℝ) 
    (h : 100 * plate_cost + 200 * cup_cost = 6) :
  20 * plate_cost + 40 * cup_cost = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_paper_goods_cost_l1396_139667


namespace NUMINAMATH_CALUDE_lightest_box_weight_l1396_139655

/-- Given three boxes with pairwise sums of weights 83 kg, 85 kg, and 86 kg,
    the weight of the lightest box is 41 kg. -/
theorem lightest_box_weight (s m l : ℝ) : 
  s ≤ m ∧ m ≤ l ∧ 
  m + s = 83 ∧ 
  l + s = 85 ∧ 
  l + m = 86 → 
  s = 41 := by
sorry

end NUMINAMATH_CALUDE_lightest_box_weight_l1396_139655
