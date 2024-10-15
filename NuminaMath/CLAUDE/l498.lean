import Mathlib

namespace NUMINAMATH_CALUDE_fence_cost_for_square_plot_l498_49821

theorem fence_cost_for_square_plot (area : Real) (price_per_foot : Real) : 
  area = 289 → price_per_foot = 55 → 4 * Real.sqrt area * price_per_foot = 3740 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_for_square_plot_l498_49821


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l498_49874

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  2 * X^4 + 10 * X^3 - 35 * X^2 - 40 * X + 12 = 
  (X^2 + 7 * X - 8) * q + (-135 * X + 84) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l498_49874


namespace NUMINAMATH_CALUDE_lcm_48_180_l498_49895

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_180_l498_49895


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l498_49878

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the foci
def focus1 : ℝ × ℝ := sorry
def focus2 : ℝ × ℝ := sorry

-- Define points P and Q on the ellipse
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define the line passing through F₁, P, and Q
def line_through_F1PQ : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter :
  ellipse P.1 P.2 ∧ ellipse Q.1 Q.2 ∧ 
  P ∈ line_through_F1PQ ∧ Q ∈ line_through_F1PQ ∧ focus1 ∈ line_through_F1PQ →
  (dist P Q + dist Q focus2 + dist P focus2 = 20) :=
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l498_49878


namespace NUMINAMATH_CALUDE_game_ends_in_56_rounds_l498_49825

/-- Represents a player in the token game -/
structure Player where
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : Fin 4 → Player
  rounds : ℕ

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game is over (any player has 0 tokens) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Plays the game until it's over -/
def playGame (initialState : GameState) : GameState :=
  sorry

/-- Theorem stating the game ends in 56 rounds -/
theorem game_ends_in_56_rounds :
  let initialState : GameState := {
    players := λ i =>
      match i with
      | 0 => ⟨17⟩
      | 1 => ⟨16⟩
      | 2 => ⟨15⟩
      | 3 => ⟨14⟩
    rounds := 0
  }
  let finalState := playGame initialState
  finalState.rounds = 56 ∧ 
  finalState.players 3 = ⟨0⟩ ∧
  isGameOver finalState = true :=
by
  sorry

end NUMINAMATH_CALUDE_game_ends_in_56_rounds_l498_49825


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l498_49823

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1 : ℝ) * tree_distance

/-- Theorem: A yard with 26 equally spaced trees, 13 meters apart, is 325 meters long -/
theorem yard_length_26_trees : 
  yard_length 26 13 = 325 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l498_49823


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l498_49846

/-- Two lines are parallel if their slopes are equal but they are not the same line -/
def parallel (a b c d e f : ℝ) : Prop :=
  a / b = d / e ∧ a / b ≠ c / f

theorem parallel_lines_a_value (a : ℝ) :
  parallel a 2 0 3 (a + 1) 1 → a = -3 ∨ a = 2 := by
  sorry

#check parallel_lines_a_value

end NUMINAMATH_CALUDE_parallel_lines_a_value_l498_49846


namespace NUMINAMATH_CALUDE_money_division_l498_49803

theorem money_division (a b c : ℝ) 
  (h1 : a = (1/3) * (b + c))
  (h2 : b = (2/7) * (a + c))
  (h3 : a = b + 15) :
  a + b + c = 540 := by
sorry

end NUMINAMATH_CALUDE_money_division_l498_49803


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l498_49801

theorem smallest_constant_inequality (x y z : ℝ) (h : x + y + z = -1) :
  ∃ C : ℝ, (∀ x y z : ℝ, x + y + z = -1 → |x^3 + y^3 + z^3 + 1| ≤ C * |x^5 + y^5 + z^5 + 1|) ∧
  C = 9/10 ∧
  ∀ C' : ℝ, (∀ x y z : ℝ, x + y + z = -1 → |x^3 + y^3 + z^3 + 1| ≤ C' * |x^5 + y^5 + z^5 + 1|) → C' ≥ 9/10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l498_49801


namespace NUMINAMATH_CALUDE_h_not_prime_l498_49898

def h (n : ℕ+) : ℤ := n.val ^ 4 - 500 * n.val ^ 2 + 625

theorem h_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (h n)) := by
  sorry

end NUMINAMATH_CALUDE_h_not_prime_l498_49898


namespace NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l498_49865

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem second_term_of_geometric_sequence
  (a : ℕ → ℚ)
  (h_geometric : geometric_sequence a)
  (h_third : a 3 = 12)
  (h_fourth : a 4 = 18) :
  a 2 = 8 :=
sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l498_49865


namespace NUMINAMATH_CALUDE_base4_10201_equals_289_l498_49822

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base4_10201_equals_289 :
  base4_to_decimal [1, 0, 2, 0, 1] = 289 := by
  sorry

end NUMINAMATH_CALUDE_base4_10201_equals_289_l498_49822


namespace NUMINAMATH_CALUDE_probability_even_sum_is_5_11_l498_49812

def ball_numbers : Finset ℕ := Finset.range 12

theorem probability_even_sum_is_5_11 :
  let total_outcomes := ball_numbers.card * (ball_numbers.card - 1)
  let favorable_outcomes := 
    (ball_numbers.filter (λ x => x % 2 = 0)).card * ((ball_numbers.filter (λ x => x % 2 = 0)).card - 1) +
    (ball_numbers.filter (λ x => x % 2 = 1)).card * ((ball_numbers.filter (λ x => x % 2 = 1)).card - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_is_5_11_l498_49812


namespace NUMINAMATH_CALUDE_real_y_condition_l498_49883

theorem real_y_condition (x y : ℝ) : 
  (∃ y, 2 * y^2 + 3 * x * y - x + 8 = 0) ↔ (x ≤ -23/9 ∨ x ≥ 5/3) :=
by sorry

end NUMINAMATH_CALUDE_real_y_condition_l498_49883


namespace NUMINAMATH_CALUDE_reference_city_stores_l498_49827

/-- The number of stores in the reference city -/
def stores : ℕ := sorry

/-- The number of hospitals in the reference city -/
def hospitals : ℕ := 500

/-- The number of schools in the reference city -/
def schools : ℕ := 200

/-- The number of police stations in the reference city -/
def police_stations : ℕ := 20

/-- The total number of buildings in the new city -/
def new_city_buildings : ℕ := 2175

theorem reference_city_stores :
  stores / 2 + 2 * hospitals + (schools - 50) + (police_stations + 5) = new_city_buildings →
  stores = 2000 := by
  sorry

end NUMINAMATH_CALUDE_reference_city_stores_l498_49827


namespace NUMINAMATH_CALUDE_lisa_walking_time_l498_49848

/-- Given Lisa's walking speed and total distance over two days, prove she walks for 1 hour each day -/
theorem lisa_walking_time 
  (speed : ℝ)              -- Lisa's walking speed in meters per minute
  (total_distance : ℝ)     -- Total distance Lisa walks in two days
  (h1 : speed = 10)        -- Lisa walks 10 meters each minute
  (h2 : total_distance = 1200) -- Lisa walks 1200 meters in two days
  : (total_distance / 2) / speed / 60 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lisa_walking_time_l498_49848


namespace NUMINAMATH_CALUDE_pen_count_problem_l498_49849

theorem pen_count_problem (total_pens : ℕ) (difference : ℕ) 
  (h1 : total_pens = 140)
  (h2 : difference = 20) :
  ∃ (ballpoint_pens fountain_pens : ℕ),
    ballpoint_pens + fountain_pens = total_pens ∧
    ballpoint_pens + difference = fountain_pens ∧
    ballpoint_pens = 60 ∧
    fountain_pens = 80 := by
  sorry

end NUMINAMATH_CALUDE_pen_count_problem_l498_49849


namespace NUMINAMATH_CALUDE_rectangle_point_characterization_l498_49893

/-- A point in the Cartesian coordinate system representing a rectangle's perimeter and area -/
structure RectanglePoint where
  k : ℝ  -- perimeter
  t : ℝ  -- area

/-- Characterizes the region of valid RectanglePoints -/
def is_valid_rectangle_point (p : RectanglePoint) : Prop :=
  p.k > 0 ∧ p.t > 0 ∧ p.t ≤ (p.k^2) / 16

/-- Theorem stating the characterization of valid rectangle points -/
theorem rectangle_point_characterization (p : RectanglePoint) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ p.k = 2*(x + y) ∧ p.t = x*y) ↔ 
  is_valid_rectangle_point p :=
sorry

end NUMINAMATH_CALUDE_rectangle_point_characterization_l498_49893


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_500_l498_49810

def hulk_jump (n : ℕ) : ℝ := 2 * (2 ^ (n - 1))

theorem hulk_jump_exceeds_500 :
  (∀ k < 9, hulk_jump k ≤ 500) ∧ hulk_jump 9 > 500 := by
  sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_500_l498_49810


namespace NUMINAMATH_CALUDE_max_value_of_function_l498_49817

theorem max_value_of_function (x : ℝ) (h : x ∈ Set.Ioo 0 (1/4)) :
  x * (1 - 4*x) ≤ (1/8) * (1 - 4*(1/8)) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l498_49817


namespace NUMINAMATH_CALUDE_tangent_circles_count_l498_49832

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are tangent if the distance between their centers equals the sum or difference of their radii -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2 ∨
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius - c2.radius)^2

/-- The theorem statement -/
theorem tangent_circles_count 
  (C1 C2 : Circle)
  (h1 : C1.radius = 2)
  (h2 : C2.radius = 2)
  (h3 : are_tangent C1 C2) :
  ∃! (s : Finset Circle), 
    (∀ c ∈ s, c.radius = 4 ∧ are_tangent c C1 ∧ are_tangent c C2) ∧ 
    s.card = 4 :=
  sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l498_49832


namespace NUMINAMATH_CALUDE_sphere_triangle_distance_l498_49844

theorem sphere_triangle_distance (r : ℝ) (a b : ℝ) (hr : r = 13) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let d := Real.sqrt (r^2 - (c/2)^2)
  d = 12 := by sorry

end NUMINAMATH_CALUDE_sphere_triangle_distance_l498_49844


namespace NUMINAMATH_CALUDE_y_value_l498_49824

theorem y_value : ∀ y : ℚ, (1 / 3 - 1 / 4 : ℚ) = 4 / y → y = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l498_49824


namespace NUMINAMATH_CALUDE_unique_number_exists_l498_49806

theorem unique_number_exists : ∃! N : ℚ, (1 / 3) * N = 8 ∧ (1 / 8) * N = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_exists_l498_49806


namespace NUMINAMATH_CALUDE_total_cost_of_gifts_l498_49809

def parents : ℕ := 2
def brothers : ℕ := 4
def sister : ℕ := 1
def brothers_spouses : ℕ := 4
def children_of_brothers : ℕ := 12
def sister_spouse : ℕ := 1
def children_of_sister : ℕ := 2
def grandparents : ℕ := 2
def cousins : ℕ := 3
def cost_per_package : ℕ := 7

theorem total_cost_of_gifts :
  parents + brothers + sister + brothers_spouses + children_of_brothers +
  sister_spouse + children_of_sister + grandparents + cousins = 31 →
  (parents + brothers + sister + brothers_spouses + children_of_brothers +
  sister_spouse + children_of_sister + grandparents + cousins) * cost_per_package = 217 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_of_gifts_l498_49809


namespace NUMINAMATH_CALUDE_unanswered_test_completion_ways_l498_49897

/-- Represents a multiple choice test. -/
structure MCTest where
  num_questions : Nat
  choices_per_question : Nat

/-- Defines an unanswered test. -/
def unanswered_test (test : MCTest) : Nat := 1

/-- Theorem stating that for a test with 4 questions and 5 choices per question,
    there is only one way to complete it if all questions are unanswered. -/
theorem unanswered_test_completion_ways :
  let test : MCTest := { num_questions := 4, choices_per_question := 5 }
  unanswered_test test = 1 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_test_completion_ways_l498_49897


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l498_49841

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 20 (2*x - 1) = Nat.choose 20 (x + 3)) → (x = 4 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l498_49841


namespace NUMINAMATH_CALUDE_inequality_holds_for_all_reals_l498_49855

theorem inequality_holds_for_all_reals (x : ℝ) : 4 * x / (x^2 + 4) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_all_reals_l498_49855


namespace NUMINAMATH_CALUDE_apples_left_ella_apples_left_l498_49852

theorem apples_left (bags_20 : Nat) (apples_per_bag_20 : Nat) 
                    (bags_25 : Nat) (apples_per_bag_25 : Nat) 
                    (sold : Nat) : Nat :=
  let total_20 := bags_20 * apples_per_bag_20
  let total_25 := bags_25 * apples_per_bag_25
  let total := total_20 + total_25
  total - sold

theorem ella_apples_left : apples_left 4 20 6 25 200 = 30 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_ella_apples_left_l498_49852


namespace NUMINAMATH_CALUDE_transportation_problem_l498_49834

/-- Represents a transportation plan -/
structure TransportPlan where
  a_trucks : ℕ
  b_trucks : ℕ

/-- Represents the problem parameters -/
structure ProblemParams where
  a_capacity : ℕ
  b_capacity : ℕ
  a_cost : ℕ
  b_cost : ℕ
  total_goods : ℕ

def is_valid_plan (params : ProblemParams) (plan : TransportPlan) : Prop :=
  params.a_capacity * plan.a_trucks + params.b_capacity * plan.b_trucks = params.total_goods

def plan_cost (params : ProblemParams) (plan : TransportPlan) : ℕ :=
  params.a_cost * plan.a_trucks + params.b_cost * plan.b_trucks

def is_most_cost_effective (params : ProblemParams) (plan : TransportPlan) : Prop :=
  is_valid_plan params plan ∧
  ∀ other_plan : TransportPlan, is_valid_plan params other_plan →
    plan_cost params plan ≤ plan_cost params other_plan

theorem transportation_problem :
  ∃ (params : ProblemParams) (best_plan : TransportPlan),
    params.a_capacity = 20 ∧
    params.b_capacity = 15 ∧
    params.a_cost = 500 ∧
    params.b_cost = 400 ∧
    params.total_goods = 190 ∧
    best_plan.a_trucks = 8 ∧
    best_plan.b_trucks = 2 ∧
    plan_cost params best_plan = 4800 ∧
    (1 * params.a_capacity + 2 * params.b_capacity = 50) ∧
    (5 * params.a_capacity + 4 * params.b_capacity = 160) ∧
    is_most_cost_effective params best_plan :=
by
  sorry

end NUMINAMATH_CALUDE_transportation_problem_l498_49834


namespace NUMINAMATH_CALUDE_m_range_l498_49887

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → m ∈ Set.Icc (-3) 5 := by
sorry

end NUMINAMATH_CALUDE_m_range_l498_49887


namespace NUMINAMATH_CALUDE_polynomial_always_positive_l498_49860

theorem polynomial_always_positive (m : ℝ) : m^6 - m^5 + m^4 + m^2 - m + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_always_positive_l498_49860


namespace NUMINAMATH_CALUDE_triangle_coordinate_l498_49885

/-- Given a triangle with vertices A(0, 3), B(4, 0), and C(x, 5), where 0 < x < 4,
    if the area of the triangle is 8 square units, then x = 8/3. -/
theorem triangle_coordinate (x : ℝ) : 
  0 < x → x < 4 → 
  (1/2 : ℝ) * |0 * (0 - 5) + 4 * (5 - 3) + x * (3 - 0)| = 8 → 
  x = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_coordinate_l498_49885


namespace NUMINAMATH_CALUDE_money_distribution_l498_49876

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (ac_sum : A + C = 200)
  (bc_sum : B + C = 330) :
  C = 30 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l498_49876


namespace NUMINAMATH_CALUDE_lemonade_solution_water_content_l498_49882

theorem lemonade_solution_water_content 
  (L : ℝ) -- Amount of lemonade syrup
  (W : ℝ) -- Amount of water
  (removed : ℝ) -- Amount of solution removed and replaced with water
  (h1 : L = 7) -- 7 parts of lemonade syrup
  (h2 : removed = 2.1428571428571423) -- Amount removed and replaced
  (h3 : L / (L + W - removed + removed) = 0.4) -- 40% concentration after replacement
  : W = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_solution_water_content_l498_49882


namespace NUMINAMATH_CALUDE_clown_count_l498_49830

/-- The number of clown mobiles -/
def num_mobiles : ℕ := 5

/-- The number of clowns in each mobile -/
def clowns_per_mobile : ℕ := 28

/-- The total number of clowns in all mobiles -/
def total_clowns : ℕ := num_mobiles * clowns_per_mobile

theorem clown_count : total_clowns = 140 := by
  sorry

end NUMINAMATH_CALUDE_clown_count_l498_49830


namespace NUMINAMATH_CALUDE_initial_average_height_l498_49853

/-- Given a class of boys with an incorrect height measurement, prove the initially calculated average height. -/
theorem initial_average_height
  (n : ℕ) -- number of boys
  (height_difference : ℝ) -- difference between incorrect and correct height
  (actual_average : ℝ) -- actual average height after correction
  (h_n : n = 35) -- there are 35 boys
  (h_diff : height_difference = 60) -- the height difference is 60 cm
  (h_actual : actual_average = 183) -- the actual average height is 183 cm
  : ∃ (initial_average : ℝ), initial_average = 181 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_height_l498_49853


namespace NUMINAMATH_CALUDE_quadratic_roots_l498_49847

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -1 ∧ x₂ = 2) ∧ 
  (∀ x : ℝ, x * (x - 2) = 2 - x ↔ x = x₁ ∨ x = x₂) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l498_49847


namespace NUMINAMATH_CALUDE_distribute_over_sum_diff_l498_49814

theorem distribute_over_sum_diff (a b c : ℝ) : a * (a + b - c) = a^2 + a*b - a*c := by
  sorry

end NUMINAMATH_CALUDE_distribute_over_sum_diff_l498_49814


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l498_49859

theorem arithmetic_calculation : (40 * 30 + (12 + 8) * 3) / 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l498_49859


namespace NUMINAMATH_CALUDE_min_value_reciprocal_product_l498_49839

theorem min_value_reciprocal_product (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_sum : 2*a + b = 4) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = 4 → 1/(x*y) ≥ 1/2) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = 4 ∧ 1/(x*y) = 1/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_product_l498_49839


namespace NUMINAMATH_CALUDE_x_range_l498_49875

theorem x_range (x : ℝ) : 
  let p := x^2 - 2*x - 3 ≥ 0
  let q := 0 < x ∧ x < 4
  (¬q) → (p ∨ q) → (x ≤ -1 ∨ x ≥ 4) :=
by
  sorry

end NUMINAMATH_CALUDE_x_range_l498_49875


namespace NUMINAMATH_CALUDE_percentage_problem_l498_49837

theorem percentage_problem (n : ℝ) : 0.15 * 0.30 * 0.50 * n = 90 → n = 4000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l498_49837


namespace NUMINAMATH_CALUDE_quadratic_root_one_l498_49813

theorem quadratic_root_one (a b c : ℝ) (h1 : a - b + c = 0) (h2 : a ≠ 0) :
  ∃ x : ℝ, a * x^2 - b * x + c = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_one_l498_49813


namespace NUMINAMATH_CALUDE_function_range_l498_49804

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the domain
def domain : Set ℝ := { x | -3 ≤ x ∧ x ≤ 2 }

-- State the theorem
theorem function_range :
  { y | ∃ x ∈ domain, f x = y } = { y | 1 ≤ y ∧ y ≤ 17 } := by sorry

end NUMINAMATH_CALUDE_function_range_l498_49804


namespace NUMINAMATH_CALUDE_inequality_proof_l498_49835

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 : ℝ) / (b - c) > (1 : ℝ) / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l498_49835


namespace NUMINAMATH_CALUDE_missing_number_proof_l498_49871

theorem missing_number_proof (a b x : ℕ) (h1 : a = 105) (h2 : b = 147) 
  (h3 : a^3 = x * 25 * 15 * b) : x = 21 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l498_49871


namespace NUMINAMATH_CALUDE_min_flowers_for_bouquets_l498_49881

/-- The number of different types of flowers in the box -/
def num_flower_types : ℕ := 6

/-- The number of flowers needed to make one bouquet -/
def flowers_per_bouquet : ℕ := 5

/-- The number of bouquets we want to guarantee -/
def target_bouquets : ℕ := 10

/-- The minimum number of flowers needed to guarantee the target number of bouquets -/
def min_flowers_needed : ℕ := 70

theorem min_flowers_for_bouquets :
  ∀ (n : ℕ), n ≥ min_flowers_needed →
  ∀ (f : Fin n → Fin num_flower_types),
  ∃ (S : Finset (Fin n)),
  S.card = target_bouquets * flowers_per_bouquet ∧
  ∀ (t : Fin num_flower_types),
  (S.filter (fun i => f i = t)).card ≥ target_bouquets * flowers_per_bouquet / num_flower_types :=
by sorry

end NUMINAMATH_CALUDE_min_flowers_for_bouquets_l498_49881


namespace NUMINAMATH_CALUDE_similarity_circle_theorem_l498_49879

-- Define the types for figures, lines, and points
variable (F : Type) (L : Type) (P : Type)

-- Define the properties and relations
variable (similar : F → F → Prop)
variable (corresponding_line : F → L → Prop)
variable (intersect_at : L → L → L → P → Prop)
variable (lies_on_circumcircle : P → F → F → F → Prop)
variable (passes_through : L → P → Prop)

-- Theorem statement
theorem similarity_circle_theorem 
  (F₁ F₂ F₃ : F) 
  (l₁ l₂ l₃ : L) 
  (W : P) :
  similar F₁ F₂ ∧ similar F₂ F₃ ∧ similar F₁ F₃ →
  corresponding_line F₁ l₁ ∧ corresponding_line F₂ l₂ ∧ corresponding_line F₃ l₃ →
  intersect_at l₁ l₂ l₃ W →
  lies_on_circumcircle W F₁ F₂ F₃ ∧
  ∃ (J₁ J₂ J₃ : P),
    lies_on_circumcircle J₁ F₁ F₂ F₃ ∧
    lies_on_circumcircle J₂ F₁ F₂ F₃ ∧
    lies_on_circumcircle J₃ F₁ F₂ F₃ ∧
    passes_through l₁ J₁ ∧
    passes_through l₂ J₂ ∧
    passes_through l₃ J₃ :=
by sorry

end NUMINAMATH_CALUDE_similarity_circle_theorem_l498_49879


namespace NUMINAMATH_CALUDE_first_trail_length_is_20_l498_49858

/-- The length of the first trail in miles -/
def first_trail_length : ℝ := 20

/-- The speed of hiking the first trail in miles per hour -/
def first_trail_speed : ℝ := 5

/-- The length of the second trail in miles -/
def second_trail_length : ℝ := 12

/-- The speed of hiking the second trail in miles per hour -/
def second_trail_speed : ℝ := 3

/-- The duration of the break during the second trail in hours -/
def break_duration : ℝ := 1

/-- The time difference between the two trails in hours -/
def time_difference : ℝ := 1

theorem first_trail_length_is_20 :
  first_trail_length = 20 ∧
  first_trail_length / first_trail_speed = 
    (second_trail_length / second_trail_speed + break_duration) - time_difference :=
by sorry

end NUMINAMATH_CALUDE_first_trail_length_is_20_l498_49858


namespace NUMINAMATH_CALUDE_cole_math_classes_l498_49868

/-- Represents the number of students in Ms. Cole's sixth-level math class -/
def sixth_level_students : ℕ := sorry

/-- Represents the number of students in Ms. Cole's fourth-level math class -/
def fourth_level_students : ℕ := sorry

/-- Represents the number of students in Ms. Cole's seventh-level math class -/
def seventh_level_students : ℕ := sorry

/-- The total number of students Ms. Cole teaches -/
def total_students : ℕ := 520

theorem cole_math_classes :
  (fourth_level_students = 4 * sixth_level_students) ∧
  (seventh_level_students = 2 * fourth_level_students) ∧
  (total_students = sixth_level_students + fourth_level_students + seventh_level_students) →
  sixth_level_students = 40 := by
  sorry

end NUMINAMATH_CALUDE_cole_math_classes_l498_49868


namespace NUMINAMATH_CALUDE_intersection_P_Q_l498_49843

-- Define the sets P and Q
def P : Set ℝ := {x | 1 < x ∧ x < 3}
def Q : Set ℝ := {x | x > 2}

-- Define the open interval (2, 3)
def open_interval_2_3 : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem intersection_P_Q : P ∩ Q = open_interval_2_3 := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l498_49843


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l498_49816

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l498_49816


namespace NUMINAMATH_CALUDE_balloon_problem_l498_49818

-- Define the variables
def x : ℕ := 10
def y : ℕ := 46
def z : ℕ := 10
def d : ℕ := 16

-- Define the total initial number of balloons
def total_initial : ℕ := x + y

-- Define the final remaining number of balloons
def final_remaining : ℕ := total_initial - d

-- Define the total amount spent
def total_spent : ℕ := total_initial * z

-- Theorem statement
theorem balloon_problem :
  final_remaining = 40 ∧ total_spent = 560 := by
  sorry

end NUMINAMATH_CALUDE_balloon_problem_l498_49818


namespace NUMINAMATH_CALUDE_x_gt_1_necessary_not_sufficient_l498_49867

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := ![x - 3, 2]
def b : Fin 2 → ℝ := ![1, 1]

-- Define the dot product of two 2D vectors
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

-- Define what it means for the angle between two vectors to be acute
def is_acute_angle (u v : Fin 2 → ℝ) : Prop := dot_product u v > 0

-- State the theorem
theorem x_gt_1_necessary_not_sufficient :
  (∀ x : ℝ, is_acute_angle (a x) b → x > 1) ∧
  ¬(∀ x : ℝ, x > 1 → is_acute_angle (a x) b) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_1_necessary_not_sufficient_l498_49867


namespace NUMINAMATH_CALUDE_tank_filling_flow_rate_l498_49854

/-- The flow rate of a pipe filling a tank, given specific conditions --/
theorem tank_filling_flow_rate : ℝ := by
  -- Define the tank capacity
  let tank_capacity : ℝ := 1000

  -- Define the initial water level (half-full)
  let initial_water : ℝ := tank_capacity / 2

  -- Define the drain rates
  let drain_rate_1 : ℝ := 1000 / 4  -- 1 kiloliter every 4 minutes
  let drain_rate_2 : ℝ := 1000 / 6  -- 1 kiloliter every 6 minutes

  -- Define the total drain rate
  let total_drain_rate : ℝ := drain_rate_1 + drain_rate_2

  -- Define the time to fill the tank completely
  let fill_time : ℝ := 6

  -- Define the volume of water added
  let water_added : ℝ := tank_capacity - initial_water

  -- Define the flow rate of the pipe
  let flow_rate : ℝ := (water_added / fill_time) + total_drain_rate

  -- Prove that the flow rate is 500 liters per minute
  have h : flow_rate = 500 := by sorry

  exact 500


end NUMINAMATH_CALUDE_tank_filling_flow_rate_l498_49854


namespace NUMINAMATH_CALUDE_BF_length_is_150_l498_49808

/-- Square ABCD with side length 500 and points E, F on AB satisfying given conditions -/
structure SquareEF where
  /-- Side length of the square -/
  side_length : ℝ
  /-- Length of EF -/
  EF_length : ℝ
  /-- Angle EOF in degrees -/
  angle_EOF : ℝ
  /-- E is between A and F -/
  E_between_A_F : Prop
  /-- AE is less than BF -/
  AE_less_than_BF : Prop
  /-- Side length is 500 -/
  side_length_eq : side_length = 500
  /-- EF length is 300 -/
  EF_length_eq : EF_length = 300
  /-- Angle EOF is 45 degrees -/
  angle_EOF_eq : angle_EOF = 45

/-- The length of BF in the given square configuration -/
def BF_length (s : SquareEF) : ℝ := sorry

/-- Theorem stating that BF length is 150 -/
theorem BF_length_is_150 (s : SquareEF) : BF_length s = 150 := sorry

end NUMINAMATH_CALUDE_BF_length_is_150_l498_49808


namespace NUMINAMATH_CALUDE_perfect_square_condition_l498_49829

theorem perfect_square_condition (x : ℤ) : 
  (∃ k : ℤ, x^2 - 14*x - 256 = k^2) ↔ (x = 15 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l498_49829


namespace NUMINAMATH_CALUDE_distinct_sequences_is_256_l498_49807

/-- Represents the number of coin flips -/
def total_flips : ℕ := 10

/-- Represents the number of fixed heads at the start -/
def fixed_heads : ℕ := 2

/-- Represents the number of possible outcomes for each flip -/
def outcomes_per_flip : ℕ := 2

/-- Calculates the number of distinct sequences -/
def distinct_sequences : ℕ := outcomes_per_flip ^ (total_flips - fixed_heads)

/-- Theorem stating that the number of distinct sequences is 256 -/
theorem distinct_sequences_is_256 : distinct_sequences = 256 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sequences_is_256_l498_49807


namespace NUMINAMATH_CALUDE_fox_distribution_l498_49826

/-- Fox Distribution Problem -/
theorem fox_distribution (m : ℕ) (a : ℝ) (x y : ℝ) :
  (∀ n : ℕ, n * a + (x - (n - 1) * y - n * a) / m = y) →
  x = (m - 1)^2 * a ∧ 
  y = (m - 1) * a ∧ 
  m - 1 > 0 := by
sorry

end NUMINAMATH_CALUDE_fox_distribution_l498_49826


namespace NUMINAMATH_CALUDE_opposite_of_neg_two_is_two_l498_49863

/-- The opposite number of a real number x is the number y such that x + y = 0 -/
def opposite (x : ℝ) : ℝ := -x

/-- Theorem: The opposite number of -2 is 2 -/
theorem opposite_of_neg_two_is_two : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_two_is_two_l498_49863


namespace NUMINAMATH_CALUDE_angle_between_vectors_l498_49888

/-- The measure of the angle between vectors a = (1, √3) and b = (√3, 1) is π/6 -/
theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (1, Real.sqrt 3) →
  b = (Real.sqrt 3, 1) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l498_49888


namespace NUMINAMATH_CALUDE_quadratic_inequality_l498_49890

/-- Given a quadratic function f(x) = x^2 + bx + c, if f(-1) = f(3), then f(1) < c < f(3) -/
theorem quadratic_inequality (b c : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + b*x + c
  f (-1) = f 3 → f 1 < c ∧ c < f 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l498_49890


namespace NUMINAMATH_CALUDE_trigonometric_sum_equals_sqrt_two_l498_49872

theorem trigonometric_sum_equals_sqrt_two :
  (Real.cos (2 * π / 180)) / (Real.sin (47 * π / 180)) +
  (Real.cos (88 * π / 180)) / (Real.sin (133 * π / 180)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equals_sqrt_two_l498_49872


namespace NUMINAMATH_CALUDE_john_bought_three_spools_l498_49866

/-- The number of spools John bought given the conditions of the problem -/
def spools_bought (spool_length : ℕ) (wire_per_necklace : ℕ) (necklaces_made : ℕ) : ℕ :=
  (wire_per_necklace * necklaces_made) / spool_length

/-- Theorem stating that John bought 3 spools of wire -/
theorem john_bought_three_spools :
  spools_bought 20 4 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_bought_three_spools_l498_49866


namespace NUMINAMATH_CALUDE_marble_percentage_theorem_l498_49873

/-- Represents a set of marbles -/
structure MarbleSet where
  total : ℕ
  broken : ℕ

/-- The problem setup -/
def marbleProblem : Prop :=
  ∃ (set1 set2 : MarbleSet),
    set1.total = 50 ∧
    set2.total = 60 ∧
    set1.broken = (10 : ℕ) * set1.total / 100 ∧
    set1.broken + set2.broken = 17 ∧
    set2.broken * 100 / set2.total = 20

/-- The theorem to prove -/
theorem marble_percentage_theorem : marbleProblem := by
  sorry

#check marble_percentage_theorem

end NUMINAMATH_CALUDE_marble_percentage_theorem_l498_49873


namespace NUMINAMATH_CALUDE_append_self_perfect_square_l498_49845

theorem append_self_perfect_square :
  ∃ (A : ℕ) (n : ℕ), 
    (10^n ≤ A) ∧ (A < 10^(n+1)) ∧ 
    ∃ (k : ℕ), ((10^n + 1) * A = k^2) := by
  sorry

end NUMINAMATH_CALUDE_append_self_perfect_square_l498_49845


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_m_range_l498_49870

theorem quadratic_inequality_implies_m_range (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*m*x + 1 ≥ 0) → -1 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_m_range_l498_49870


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l498_49856

/-- An arithmetic sequence and its partial sums -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  sum_def : ∀ n : ℕ+, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ := seq.a 2 - seq.a 1

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h1 : seq.S 5 < seq.S 6)
    (h2 : seq.S 6 = seq.S 7)
    (h3 : seq.S 7 > seq.S 8) :
  (common_difference seq < 0) ∧ 
  (seq.a 7 = 0) ∧ 
  (seq.S 9 ≤ seq.S 5) ∧
  (∀ n : ℕ+, seq.S n ≤ seq.S 6) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l498_49856


namespace NUMINAMATH_CALUDE_right_triangle_third_side_square_l498_49819

theorem right_triangle_third_side_square (a b c : ℝ) : 
  a = 4 → b = 5 → c > 0 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c^2 = 9 ∨ c^2 = 41 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_square_l498_49819


namespace NUMINAMATH_CALUDE_girls_candies_contradiction_l498_49864

theorem girls_candies_contradiction (cM cK cL cO : ℕ) : 
  (cM + cK = cL + cO + 12) → (cK + cL = cM + cO - 7) → False :=
by
  sorry

end NUMINAMATH_CALUDE_girls_candies_contradiction_l498_49864


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_mean_55_l498_49857

theorem max_ratio_two_digit_mean_55 :
  ∀ x y : ℕ,
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  (x + y) / 2 = 55 →
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  (a + b) / 2 = 55 →
  x / y ≤ 9 ∧ x / y ≥ a / b :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_mean_55_l498_49857


namespace NUMINAMATH_CALUDE_fraction_difference_squared_l498_49838

theorem fraction_difference_squared (a b : ℝ) (h : 1/a - 1/b = 1/(a + b)) :
  1/a^2 - 1/b^2 = 1/(a*b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_squared_l498_49838


namespace NUMINAMATH_CALUDE_a_profit_calculation_l498_49831

def business_profit (a_investment b_investment total_profit : ℕ) : ℕ :=
  let total_investment := a_investment + b_investment
  let management_fee := total_profit / 10
  let remaining_profit := total_profit - management_fee
  let a_share := remaining_profit * a_investment / total_investment
  management_fee + a_share

theorem a_profit_calculation (a_investment b_investment total_profit : ℕ) :
  a_investment = 15000 →
  b_investment = 25000 →
  total_profit = 9600 →
  business_profit a_investment b_investment total_profit = 4200 := by
sorry

#eval business_profit 15000 25000 9600

end NUMINAMATH_CALUDE_a_profit_calculation_l498_49831


namespace NUMINAMATH_CALUDE_max_sum_xyz_l498_49802

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorial (x : ℕ) : ℕ := (List.range x).map factorial |>.sum

theorem max_sum_xyz (x y z : ℕ) (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_eq : sum_factorial x = y^2) :
  x + y + z ≤ 8 := by sorry

end NUMINAMATH_CALUDE_max_sum_xyz_l498_49802


namespace NUMINAMATH_CALUDE_line_circle_intersection_condition_l498_49896

/-- The line equation: ax + y + a + 1 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y + a + 1 = 0

/-- The circle equation: x^2 + y^2 - 2x - 2y + b = 0 -/
def circle_equation (b x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + b = 0

/-- The line intersects the circle for all real a -/
def line_intersects_circle (b : ℝ) : Prop :=
  ∀ a : ℝ, ∃ x y : ℝ, line_equation a x y ∧ circle_equation b x y

theorem line_circle_intersection_condition :
  ∀ b : ℝ, line_intersects_circle b ↔ b < -6 := by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_condition_l498_49896


namespace NUMINAMATH_CALUDE_horner_operation_count_l498_49861

/-- Polynomial coefficients in descending order of degree -/
def polynomial : List ℝ := [3, 4, 5, 6, 7, 8, 1]

/-- Number of multiplication operations in Horner's method -/
def horner_mult_ops (p : List ℝ) : ℕ := p.length - 1

/-- Number of addition operations in Horner's method -/
def horner_add_ops (p : List ℝ) : ℕ := p.length - 1

theorem horner_operation_count :
  horner_mult_ops polynomial = 6 ∧ horner_add_ops polynomial = 6 := by
  sorry

end NUMINAMATH_CALUDE_horner_operation_count_l498_49861


namespace NUMINAMATH_CALUDE_bananas_lost_l498_49850

theorem bananas_lost (initial_bananas final_bananas : ℕ) 
  (h1 : initial_bananas = 47) 
  (h2 : final_bananas = 2) : 
  initial_bananas - final_bananas = 45 := by
sorry

end NUMINAMATH_CALUDE_bananas_lost_l498_49850


namespace NUMINAMATH_CALUDE_power_of_two_in_product_l498_49820

theorem power_of_two_in_product (w : ℕ+) : 
  (3^3 ∣ (1452 * w)) ∧ 
  (13^3 ∣ (1452 * w)) ∧ 
  (∀ x : ℕ+, (3^3 ∣ (1452 * x)) ∧ (13^3 ∣ (1452 * x)) → w ≤ x) ∧
  w = 468 →
  ∃ (n : ℕ), 2^2 * n = 1452 * w ∧ ¬(2 ∣ n) := by
sorry

end NUMINAMATH_CALUDE_power_of_two_in_product_l498_49820


namespace NUMINAMATH_CALUDE_x_squared_eq_two_is_quadratic_l498_49899

/-- Definition of a quadratic equation in x -/
def is_quadratic_in_x (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 2 -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- Theorem: x^2 = 2 is a quadratic equation in x -/
theorem x_squared_eq_two_is_quadratic : is_quadratic_in_x f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_eq_two_is_quadratic_l498_49899


namespace NUMINAMATH_CALUDE_sum_of_integers_l498_49884

theorem sum_of_integers (a b : ℕ+) 
  (h1 : a.val^2 + b.val^2 = 585)
  (h2 : Nat.gcd a.val b.val + Nat.lcm a.val b.val = 87) :
  a.val + b.val = 33 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_l498_49884


namespace NUMINAMATH_CALUDE_white_tshirts_per_package_l498_49842

theorem white_tshirts_per_package (total_tshirts : ℕ) (num_packages : ℕ) 
  (h1 : total_tshirts = 426) (h2 : num_packages = 71) :
  total_tshirts / num_packages = 6 := by
  sorry

end NUMINAMATH_CALUDE_white_tshirts_per_package_l498_49842


namespace NUMINAMATH_CALUDE_group_total_sum_l498_49800

/-- The total sum spent by a group of friends on dinner and a gift -/
def total_sum (num_friends : ℕ) (additional_payment : ℚ) (gift_cost : ℚ) : ℚ :=
  let dinner_cost := (num_friends - 1) * additional_payment * 10
  dinner_cost + gift_cost

/-- Proof of the total sum spent by the group -/
theorem group_total_sum :
  total_sum 10 3 15 = 285 := by
  sorry

end NUMINAMATH_CALUDE_group_total_sum_l498_49800


namespace NUMINAMATH_CALUDE_fabric_price_system_l498_49894

/-- Represents the price per foot of damask fabric in cents -/
def damask_price : ℝ := sorry

/-- Represents the price per foot of gauze fabric in cents -/
def gauze_price : ℝ := sorry

/-- The length of the damask fabric in feet -/
def damask_length : ℝ := 7

/-- The length of the gauze fabric in feet -/
def gauze_length : ℝ := 9

/-- The price difference per foot between damask and gauze fabrics in cents -/
def price_difference : ℝ := 36

theorem fabric_price_system :
  (damask_length * damask_price = gauze_length * gauze_price) ∧
  (damask_price - gauze_price = price_difference) := by sorry

end NUMINAMATH_CALUDE_fabric_price_system_l498_49894


namespace NUMINAMATH_CALUDE_sin_cos_relation_l498_49851

theorem sin_cos_relation (x : Real) (h : Real.sin x = 4 * Real.cos x) :
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_relation_l498_49851


namespace NUMINAMATH_CALUDE_acute_triangle_existence_l498_49869

/-- Given n positive real numbers satisfying the max-min condition, 
    there exist three that form an acute triangle for all n ≥ 13 -/
theorem acute_triangle_existence (n : ℕ) (hn : n ≥ 13) 
  (a : Fin n → ℝ) (ha : ∀ i, a i > 0) 
  (hmax : ∀ i j, a i ≤ n * a j) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    a i ^ 2 + a j ^ 2 > a k ^ 2 ∧
    a j ^ 2 + a k ^ 2 > a i ^ 2 ∧
    a k ^ 2 + a i ^ 2 > a j ^ 2 :=
sorry

end NUMINAMATH_CALUDE_acute_triangle_existence_l498_49869


namespace NUMINAMATH_CALUDE_system_solution_l498_49891

theorem system_solution :
  let solutions : List (ℝ × ℝ) := [
    (-Real.sqrt (2/5), 2 * Real.sqrt (2/5)),
    (Real.sqrt (2/5), 2 * Real.sqrt (2/5)),
    (Real.sqrt (2/5), -2 * Real.sqrt (2/5)),
    (-Real.sqrt (2/5), -2 * Real.sqrt (2/5))
  ]
  ∀ x y : ℝ,
    (x^2 + y^2 ≤ 2 ∧
     x^4 - 8*x^2*y^2 + 16*y^4 - 20*x^2 - 80*y^2 + 100 = 0) ↔
    (x, y) ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l498_49891


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l498_49805

theorem quadratic_equation_equivalence :
  ∃ (m n : ℝ), (∀ x, 4 * x^2 + 8 * x - 448 = 0 ↔ (x + m)^2 = n) ∧ m = 1 ∧ n = 113 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l498_49805


namespace NUMINAMATH_CALUDE_f_period_and_range_l498_49836

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + Real.sqrt 3 / 2

theorem f_period_and_range :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-Real.sqrt 3 / 2) 1 ↔
    ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_f_period_and_range_l498_49836


namespace NUMINAMATH_CALUDE_football_water_cooler_problem_l498_49815

/-- Represents the number of skill position players who must wait for a refill --/
def skillPlayersWaiting (coolerCapacity : ℕ) (numLinemen : ℕ) (numSkillPlayers : ℕ) 
  (linemenConsumption : ℕ) (skillPlayerConsumption : ℕ) : ℕ :=
  let waterLeftForSkillPlayers := coolerCapacity - numLinemen * linemenConsumption
  let skillPlayersThatCanDrink := waterLeftForSkillPlayers / skillPlayerConsumption
  numSkillPlayers - skillPlayersThatCanDrink

theorem football_water_cooler_problem :
  skillPlayersWaiting 126 12 10 8 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_football_water_cooler_problem_l498_49815


namespace NUMINAMATH_CALUDE_flower_pot_cost_difference_l498_49828

theorem flower_pot_cost_difference (n : Nat) (largest_cost difference total_cost : ℚ) : 
  n = 6 ∧ 
  largest_cost = 175/100 ∧ 
  difference = 15/100 ∧
  total_cost = (n : ℚ) * largest_cost - ((n - 1) * n / 2 : ℚ) * difference →
  total_cost = 825/100 := by
sorry

end NUMINAMATH_CALUDE_flower_pot_cost_difference_l498_49828


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l498_49833

theorem diophantine_equation_solution (x y : ℤ) : x^4 - 2*y^2 = 1 → x = 1 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l498_49833


namespace NUMINAMATH_CALUDE_division_remainder_l498_49840

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 22)
  (h2 : divisor = 3)
  (h3 : quotient = 7)
  (h4 : dividend = divisor * quotient + remainder) :
  remainder = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l498_49840


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l498_49811

theorem sum_of_roots_quadratic_equation (α β : ℝ) : 
  (α^2 - 4*α + 3 = 0) → (β^2 - 4*β + 3 = 0) → α + β = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l498_49811


namespace NUMINAMATH_CALUDE_asymptote_sum_l498_49886

/-- Given a function g(x) = (x - 3) / (x^2 + cx + d) with vertical asymptotes at x = 2 and x = -1,
    the sum of c and d is -3. -/
theorem asymptote_sum (c d : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -1 → 
    ∃ g : ℝ → ℝ, g x = (x - 3) / (x^2 + c*x + d)) ∧
  (x^2 + c*x + d = 0 ↔ x = 2 ∨ x = -1) →
  c + d = -3 :=
by sorry

end NUMINAMATH_CALUDE_asymptote_sum_l498_49886


namespace NUMINAMATH_CALUDE_sqrt_fraction_equivalence_l498_49862

theorem sqrt_fraction_equivalence (x : ℝ) (h : x < 0) :
  Real.sqrt (x / (1 - (x - 4) / x)) = -x / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equivalence_l498_49862


namespace NUMINAMATH_CALUDE_complex_fourth_power_magnitude_l498_49880

theorem complex_fourth_power_magnitude : 
  Complex.abs ((5 + 2 * Complex.I * Real.sqrt 3) ^ 4) = 1369 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_magnitude_l498_49880


namespace NUMINAMATH_CALUDE_log_equation_solutions_are_integers_l498_49892

theorem log_equation_solutions_are_integers : ∃ (a b : ℤ), 
  (Real.log (a^2 - 8*a + 20) = 3) ∧ 
  (Real.log (b^2 - 8*b + 20) = 3) ∧ 
  (a ≠ b) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solutions_are_integers_l498_49892


namespace NUMINAMATH_CALUDE_sum_of_digits_square_of_ones_l498_49889

/-- Given a natural number n, construct a number consisting of n ones -/
def number_of_ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

/-- Theorem: For a number consisting of n ones, the sum of digits of its square equals n^2 -/
theorem sum_of_digits_square_of_ones (n : ℕ) :
  sum_of_digits ((number_of_ones n)^2) = n^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_of_ones_l498_49889


namespace NUMINAMATH_CALUDE_sin_plus_cos_eq_one_solutions_l498_49877

theorem sin_plus_cos_eq_one_solutions (x : Real) : 
  0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0 ∨ x = Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_eq_one_solutions_l498_49877
