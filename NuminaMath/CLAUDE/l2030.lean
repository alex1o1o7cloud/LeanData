import Mathlib

namespace NUMINAMATH_CALUDE_min_length_GH_l2030_203026

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the vertices A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define a point P on the ellipse above the x-axis
def P (x y : ℝ) : Prop := ellipse_C x y ∧ y > 0

-- Define the line y = 3
def line_y_3 (x y : ℝ) : Prop := y = 3

-- Define the intersection points G and H
def G (x y : ℝ) : Prop := ∃ (k : ℝ), y = k * (x + 2) ∧ line_y_3 x y
def H (x y : ℝ) : Prop := ∃ (k : ℝ), y = -1/(4*k) * (x - 2) ∧ line_y_3 x y

-- Theorem statement
theorem min_length_GH :
  ∀ (x_p y_p x_g y_g x_h y_h : ℝ),
    P x_p y_p →
    G x_g y_g →
    H x_h y_h →
    ∀ (l : ℝ), l = |x_g - x_h| →
    ∃ (min_l : ℝ), min_l = 8 ∧ l ≥ min_l :=
sorry

end NUMINAMATH_CALUDE_min_length_GH_l2030_203026


namespace NUMINAMATH_CALUDE_field_path_area_and_cost_l2030_203000

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

/-- Theorem: For a 60m x 55m field with a 2.5m wide path, the path area is 600 sq m
    and the construction cost at Rs. 2 per sq m is Rs. 1200 -/
theorem field_path_area_and_cost :
  let field_length : ℝ := 60
  let field_width : ℝ := 55
  let path_width : ℝ := 2.5
  let cost_per_unit : ℝ := 2
  (path_area field_length field_width path_width = 600) ∧
  (construction_cost (path_area field_length field_width path_width) cost_per_unit = 1200) :=
by sorry

end NUMINAMATH_CALUDE_field_path_area_and_cost_l2030_203000


namespace NUMINAMATH_CALUDE_solve_inequality_system_simplify_expression_l2030_203009

-- Problem 1
theorem solve_inequality_system (x : ℝ) :
  (10 - 3 * x < -5 ∧ x / 3 ≥ 4 - (x - 2) / 2) ↔ x ≥ 6 := by sorry

-- Problem 2
theorem simplify_expression (a : ℝ) (h : a ≠ 0 ∧ a ≠ 1 ∧ a ≠ -1) :
  2 / (a + 1) - (a - 2) / (a^2 - 1) / (a * (a - 2) / (a^2 - 2*a + 1)) = 1 / a := by sorry

end NUMINAMATH_CALUDE_solve_inequality_system_simplify_expression_l2030_203009


namespace NUMINAMATH_CALUDE_unique_digits_for_multiple_of_99_l2030_203055

def is_divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

theorem unique_digits_for_multiple_of_99 :
  ∀ α β : ℕ,
  0 ≤ α ∧ α ≤ 9 →
  0 ≤ β ∧ β ≤ 9 →
  is_divisible_by_99 (62 * 10000 + α * 1000 + β * 100 + 427) →
  α = 2 ∧ β = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_digits_for_multiple_of_99_l2030_203055


namespace NUMINAMATH_CALUDE_bug_probability_l2030_203063

/-- The probability of the bug being at its starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - Q n)

/-- The problem statement -/
theorem bug_probability : Q 8 = 547 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_l2030_203063


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l2030_203058

/-- The distance between two cities given a map distance and scale -/
def real_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Theorem: The distance between Stockholm and Uppsala is 1200 km -/
theorem stockholm_uppsala_distance :
  let map_distance : ℝ := 120
  let scale : ℝ := 10
  real_distance map_distance scale = 1200 := by
sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l2030_203058


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l2030_203013

def n : ℕ := 4016

def k : ℕ := n^2 + 2^n

theorem units_digit_of_k_squared_plus_two_to_k (n : ℕ) (k : ℕ) :
  n = 4016 →
  k = n^2 + 2^n →
  (k^2 + 2^k) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l2030_203013


namespace NUMINAMATH_CALUDE_no_perfect_square_131_base_n_l2030_203065

theorem no_perfect_square_131_base_n : 
  ¬ ∃ (n : ℤ), 4 ≤ n ∧ n ≤ 12 ∧ ∃ (k : ℤ), n^2 + 3*n + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_131_base_n_l2030_203065


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2030_203056

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 8 = 10) : 
  3 * a 5 + a 7 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2030_203056


namespace NUMINAMATH_CALUDE_one_fifths_in_nine_fifths_l2030_203048

theorem one_fifths_in_nine_fifths : (9 : ℚ) / 5 / (1 / 5) = 9 := by
  sorry

end NUMINAMATH_CALUDE_one_fifths_in_nine_fifths_l2030_203048


namespace NUMINAMATH_CALUDE_intersection_area_is_4pi_l2030_203047

-- Define the rectangle
def rectangle_vertices : List (ℝ × ℝ) := [(2, 3), (2, 15), (13, 3), (13, 15)]

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 13)^2 + (y - 3)^2 = 16

-- Define the area of intersection
def intersection_area (rect : List (ℝ × ℝ)) (circle : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Theorem statement
theorem intersection_area_is_4pi :
  intersection_area rectangle_vertices circle_equation = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_4pi_l2030_203047


namespace NUMINAMATH_CALUDE_committee_formation_count_l2030_203093

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of Republicans in the city council -/
def num_republicans : ℕ := 10

/-- The number of Democrats in the city council -/
def num_democrats : ℕ := 7

/-- The number of Republicans needed in the committee -/
def committee_republicans : ℕ := 4

/-- The number of Democrats needed in the committee -/
def committee_democrats : ℕ := 3

/-- The total number of ways to form the committee -/
def total_committee_formations : ℕ := 
  binomial num_republicans committee_republicans * binomial num_democrats committee_democrats

theorem committee_formation_count : total_committee_formations = 7350 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l2030_203093


namespace NUMINAMATH_CALUDE_social_practice_arrangements_l2030_203023

def number_of_teachers : Nat := 2
def number_of_students : Nat := 6
def teachers_per_group : Nat := 1
def students_per_group : Nat := 3
def number_of_groups : Nat := 2

theorem social_practice_arrangements :
  (number_of_teachers.choose teachers_per_group) *
  (number_of_students.choose students_per_group) = 40 := by
  sorry

end NUMINAMATH_CALUDE_social_practice_arrangements_l2030_203023


namespace NUMINAMATH_CALUDE_alice_score_l2030_203061

theorem alice_score : 
  let correct_answers : ℕ := 15
  let incorrect_answers : ℕ := 5
  let unattempted : ℕ := 10
  let correct_points : ℚ := 1
  let incorrect_penalty : ℚ := 1/4
  correct_answers * correct_points - incorrect_answers * incorrect_penalty = 13.75 := by
  sorry

end NUMINAMATH_CALUDE_alice_score_l2030_203061


namespace NUMINAMATH_CALUDE_polygon_equidistant_point_l2030_203099

-- Define a convex polygon
def ConvexPolygon (V : Type*) := V → ℝ × ℝ

-- Define a point inside the polygon
def InsidePoint (P : ConvexPolygon V) (O : ℝ × ℝ) : Prop := sorry

-- Define the property of forming isosceles triangles
def FormsIsoscelesTriangles (P : ConvexPolygon V) (O : ℝ × ℝ) : Prop :=
  ∀ (v1 v2 : V), v1 ≠ v2 → ‖P v1 - O‖ = ‖P v2 - O‖

-- Define the property of being equidistant from all vertices
def EquidistantFromVertices (P : ConvexPolygon V) (O : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (v : V), ‖P v - O‖ = r

-- State the theorem
theorem polygon_equidistant_point {V : Type*} (P : ConvexPolygon V) (O : ℝ × ℝ) :
  InsidePoint P O → FormsIsoscelesTriangles P O → EquidistantFromVertices P O :=
sorry

end NUMINAMATH_CALUDE_polygon_equidistant_point_l2030_203099


namespace NUMINAMATH_CALUDE_train_length_problem_l2030_203072

theorem train_length_problem (speed1 speed2 : ℝ) (pass_time : ℝ) (h1 : speed1 = 55) (h2 : speed2 = 50) (h3 : pass_time = 11.657142857142858) :
  let relative_speed := (speed1 + speed2) * (5 / 18)
  let total_distance := relative_speed * pass_time
  let train_length := total_distance / 2
  train_length = 170 := by sorry

end NUMINAMATH_CALUDE_train_length_problem_l2030_203072


namespace NUMINAMATH_CALUDE_smallest_cookie_containers_six_satisfies_condition_smallest_n_is_six_l2030_203059

theorem smallest_cookie_containers (n : ℕ) : (∃ k : ℕ, 15 * n - 2 = 11 * k) → n ≥ 6 := by
  sorry

theorem six_satisfies_condition : ∃ k : ℕ, 15 * 6 - 2 = 11 * k := by
  sorry

theorem smallest_n_is_six : (∃ n : ℕ, (∃ k : ℕ, 15 * n - 2 = 11 * k) ∧ (∀ m : ℕ, m < n → ¬(∃ k : ℕ, 15 * m - 2 = 11 * k))) ∧ (∃ k : ℕ, 15 * 6 - 2 = 11 * k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_containers_six_satisfies_condition_smallest_n_is_six_l2030_203059


namespace NUMINAMATH_CALUDE_initial_travel_time_l2030_203094

theorem initial_travel_time (distance : ℝ) (new_speed : ℝ) :
  distance = 540 ∧ new_speed = 60 →
  ∃ initial_time : ℝ,
    distance = new_speed * (3/4 * initial_time) ∧
    initial_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_travel_time_l2030_203094


namespace NUMINAMATH_CALUDE_tan_half_implies_expression_eight_l2030_203018

theorem tan_half_implies_expression_eight (x : ℝ) (h : Real.tan x = 1 / 2) :
  (2 * Real.sin x + 3 * Real.cos x) / (Real.cos x - Real.sin x) = 8 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_implies_expression_eight_l2030_203018


namespace NUMINAMATH_CALUDE_train_length_calculation_l2030_203096

/-- Calculates the length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length_calculation (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  speed = 72 → platform_length = 250 → crossing_time = 30 →
  (speed * (5/18) * crossing_time) - platform_length = 350 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2030_203096


namespace NUMINAMATH_CALUDE_cube_root_21600_l2030_203019

theorem cube_root_21600 : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (∀ (a' b' : ℕ), a' > 0 → b' > 0 → a'^3 * b' = 21600 → b ≤ b') ∧ a^3 * b = 21600 ∧ a + b = 106 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_21600_l2030_203019


namespace NUMINAMATH_CALUDE_strategy_D_is_best_l2030_203004

/-- Represents an investment strategy --/
inductive Strategy
| A  -- Six 1-year terms
| B  -- Three 2-year terms
| C  -- Two 3-year terms
| D  -- One 5-year term followed by one 1-year term

/-- Calculates the final amount for a given strategy --/
def calculate_return (strategy : Strategy) : ℝ :=
  match strategy with
  | Strategy.A => 10000 * (1 + 0.0225)^6
  | Strategy.B => 10000 * (1 + 0.025 * 2)^3
  | Strategy.C => 10000 * (1 + 0.028 * 3)^2
  | Strategy.D => 10000 * (1 + 0.03 * 5) * (1 + 0.0225)

/-- Theorem stating that Strategy D yields the highest return --/
theorem strategy_D_is_best :
  ∀ s : Strategy, calculate_return Strategy.D ≥ calculate_return s :=
by sorry


end NUMINAMATH_CALUDE_strategy_D_is_best_l2030_203004


namespace NUMINAMATH_CALUDE_sine_amplitude_l2030_203040

/-- Given a sinusoidal function y = a * sin(bx + c) + d with positive constants a, b, c, and d,
    if the maximum value of the function is 3 and the minimum value is -1, then a = 2. -/
theorem sine_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hmax : ∀ x, a * Real.sin (b * x + c) + d ≤ 3)
  (hmin : ∀ x, a * Real.sin (b * x + c) + d ≥ -1)
  (hreach_max : ∃ x, a * Real.sin (b * x + c) + d = 3)
  (hreach_min : ∃ x, a * Real.sin (b * x + c) + d = -1) :
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_amplitude_l2030_203040


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2030_203077

def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -2)

theorem parallel_vectors_x_value :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ a + b x = k • (2 • a - b x)) → x = -4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2030_203077


namespace NUMINAMATH_CALUDE_complex_on_real_axis_l2030_203001

theorem complex_on_real_axis (a : ℝ) : 
  let z : ℂ := (a - Complex.I) * (1 + Complex.I)
  z.im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_real_axis_l2030_203001


namespace NUMINAMATH_CALUDE_union_of_intervals_l2030_203074

open Set

theorem union_of_intervals (M N : Set ℝ) : 
  M = {x : ℝ | -1 < x ∧ x < 3} → 
  N = {x : ℝ | x ≥ 1} → 
  M ∪ N = {x : ℝ | x > -1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_intervals_l2030_203074


namespace NUMINAMATH_CALUDE_janet_lives_calculation_l2030_203034

theorem janet_lives_calculation (initial_lives current_lives gained_lives : ℕ) :
  initial_lives = 47 →
  current_lives = initial_lives - 23 →
  gained_lives = 46 →
  current_lives + gained_lives = 70 := by
  sorry

end NUMINAMATH_CALUDE_janet_lives_calculation_l2030_203034


namespace NUMINAMATH_CALUDE_meaningful_reciprocal_l2030_203036

theorem meaningful_reciprocal (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_reciprocal_l2030_203036


namespace NUMINAMATH_CALUDE_nancy_homework_l2030_203076

def homework_problem (math_problems : ℝ) (problems_per_hour : ℝ) (total_hours : ℝ) : Prop :=
  let total_problems := problems_per_hour * total_hours
  let spelling_problems := total_problems - math_problems
  spelling_problems = 15.0

theorem nancy_homework :
  homework_problem 17.0 8.0 4.0 := by
  sorry

end NUMINAMATH_CALUDE_nancy_homework_l2030_203076


namespace NUMINAMATH_CALUDE_equation_solution_l2030_203051

theorem equation_solution (x : ℝ) : 
  x ≠ 8 → x ≠ 7 → 
  ((x + 7) / (x - 8) - 6 = (5 * x - 55) / (7 - x)) ↔ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2030_203051


namespace NUMINAMATH_CALUDE_M_subset_N_l2030_203070

-- Define the set M
def M : Set ℝ := {x | ∃ k : ℤ, x = (k / 2 : ℝ) * 180 + 45}

-- Define the set N
def N : Set ℝ := {x | ∃ k : ℤ, x = (k / 4 : ℝ) * 180 + 45}

-- Theorem stating that M is a subset of N
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l2030_203070


namespace NUMINAMATH_CALUDE_max_product_of_three_l2030_203052

def S : Finset Int := {-10, -5, -3, 0, 2, 6, 8}

theorem max_product_of_three (a b c : Int) :
  a ∈ S → b ∈ S → c ∈ S →
  a ≠ b → b ≠ c → a ≠ c →
  a * b * c ≤ 400 ∧ 
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x * y * z = 400 :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_three_l2030_203052


namespace NUMINAMATH_CALUDE_path_area_calculation_l2030_203090

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Proves that the area of the path around the given field is 675 sq m -/
theorem path_area_calculation (field_length field_width path_width : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5) :
  path_area field_length field_width path_width = 675 := by
  sorry

#eval path_area 75 55 2.5

end NUMINAMATH_CALUDE_path_area_calculation_l2030_203090


namespace NUMINAMATH_CALUDE_max_value_of_z_in_D_l2030_203041

-- Define the triangular region D
def D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 0 ∧ p.1 ≤ 1 ∧ p.1 ≥ 0}

-- Define the objective function
def z (p : ℝ × ℝ) : ℝ := p.1 - 2*p.2 + 5

-- Theorem statement
theorem max_value_of_z_in_D :
  ∃ (max_z : ℝ), max_z = 8 ∧ ∀ p ∈ D, z p ≤ max_z :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_in_D_l2030_203041


namespace NUMINAMATH_CALUDE_winner_depends_on_n_l2030_203043

/-- Represents a player in the game -/
inductive Player
| Bela
| Jenn

/-- Represents the game state -/
structure GameState where
  n : ℕ
  choices : List ℝ

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (move : ℝ) : Prop :=
  0 ≤ move ∧ move ≤ state.n ∧ ∀ c ∈ state.choices, |move - c| > 1.5

/-- Determines if the game is over -/
def is_game_over (state : GameState) : Prop :=
  ∀ move, ¬(is_valid_move state move)

/-- Determines the winner of the game -/
def winner (state : GameState) : Player :=
  if state.choices.length % 2 = 0 then Player.Jenn else Player.Bela

/-- The main theorem stating that the winner depends on the specific value of n -/
theorem winner_depends_on_n :
  ∃ n m : ℕ,
    n > 5 ∧ m > 5 ∧
    (∃ state1 : GameState, state1.n = n ∧ is_game_over state1 ∧ winner state1 = Player.Bela) ∧
    (∃ state2 : GameState, state2.n = m ∧ is_game_over state2 ∧ winner state2 = Player.Jenn) :=
  sorry


end NUMINAMATH_CALUDE_winner_depends_on_n_l2030_203043


namespace NUMINAMATH_CALUDE_equation_positive_root_l2030_203084

/-- Given an equation (x / (x - 5) = 3 - a / (x - 5)) with a positive root, prove that a = -5 --/
theorem equation_positive_root (x a : ℝ) (h : x > 0) 
  (eq : x / (x - 5) = 3 - a / (x - 5)) : a = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_l2030_203084


namespace NUMINAMATH_CALUDE_farm_animals_percentage_l2030_203064

theorem farm_animals_percentage (cows ducks pigs : ℕ) : 
  cows = 20 →
  pigs = (ducks + cows) / 5 →
  cows + ducks + pigs = 60 →
  (ducks - cows : ℚ) / cows * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_percentage_l2030_203064


namespace NUMINAMATH_CALUDE_q_age_is_40_l2030_203021

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Given two people P and Q, proves that Q's age is 40 years
    under the specified conditions -/
theorem q_age_is_40 (P Q : Person) :
  (∃ (y : ℕ), P.age = 3 * (Q.age - y) ∧ P.age - y = Q.age) →
  P.age + Q.age = 100 →
  Q.age = 40 := by
sorry


end NUMINAMATH_CALUDE_q_age_is_40_l2030_203021


namespace NUMINAMATH_CALUDE_picnic_difference_l2030_203012

/-- Proves that the difference between adults and children at a picnic is 20 --/
theorem picnic_difference (total : ℕ) (men : ℕ) (women : ℕ) (adults : ℕ) (children : ℕ) : 
  total = 200 → 
  men = women + 20 → 
  men = 65 → 
  total = adults + children → 
  adults = men + women → 
  adults - children = 20 := by
sorry

end NUMINAMATH_CALUDE_picnic_difference_l2030_203012


namespace NUMINAMATH_CALUDE_distance_is_150_l2030_203095

/-- The distance between point A and point B in kilometers. -/
def distance : ℝ := 150

/-- The original speed of the car in kilometers per hour. -/
def original_speed : ℝ := sorry

/-- The original travel time in hours. -/
def original_time : ℝ := sorry

/-- Condition 1: If the car's speed is increased by 20%, the car can arrive 25 minutes earlier. -/
axiom condition1 : distance / (original_speed * 1.2) = original_time - 25 / 60

/-- Condition 2: If the car travels 100 kilometers at the original speed and then increases its speed by 25%, the car can arrive 10 minutes earlier. -/
axiom condition2 : 100 / original_speed + (distance - 100) / (original_speed * 1.25) = original_time - 10 / 60

/-- The theorem stating that the distance between point A and point B is 150 kilometers. -/
theorem distance_is_150 : distance = 150 := by sorry

end NUMINAMATH_CALUDE_distance_is_150_l2030_203095


namespace NUMINAMATH_CALUDE_unique_number_l2030_203032

theorem unique_number : ∃! x : ℕ, 
  x > 0 ∧ 
  (∃ k : ℕ, 10 * x + 4 = k * (x + 4)) ∧
  (10 * x + 4) / (x + 4) = x + 4 - 27 ∧
  x = 32 := by
sorry

end NUMINAMATH_CALUDE_unique_number_l2030_203032


namespace NUMINAMATH_CALUDE_cos_squared_pi_sixth_plus_half_alpha_l2030_203014

theorem cos_squared_pi_sixth_plus_half_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (π / 6 + α / 2) ^ 2 = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_squared_pi_sixth_plus_half_alpha_l2030_203014


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l2030_203082

theorem sqrt_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l2030_203082


namespace NUMINAMATH_CALUDE_arcsin_neg_half_equals_neg_pi_sixth_l2030_203073

theorem arcsin_neg_half_equals_neg_pi_sixth : 
  Real.arcsin (-1/2) = -π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_neg_half_equals_neg_pi_sixth_l2030_203073


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l2030_203015

theorem min_product_of_three_numbers (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1)
  (ordered : x ≤ y ∧ y ≤ z)
  (max_twice_min : z ≤ 2 * x) :
  x * y * z ≥ 1 / 32 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b + c = 1 ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 2 * a ∧ a * b * c = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l2030_203015


namespace NUMINAMATH_CALUDE_problem_solution_l2030_203029

/-- S(n, k) denotes the number of coefficients in the expansion of (x+1)^n that are not divisible by k -/
def S (n k : ℕ) : ℕ := sorry

theorem problem_solution :
  (S 2012 3 = 324) ∧ (2012 ∣ S (2012^2011) 2011) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2030_203029


namespace NUMINAMATH_CALUDE_base_2_representation_of_96_l2030_203075

theorem base_2_representation_of_96 :
  ∃ (a b c d e f g : Nat),
    96 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 ∧
    a = 1 ∧ b = 1 ∧ c = 0 ∧ d = 0 ∧ e = 0 ∧ f = 0 ∧ g = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_96_l2030_203075


namespace NUMINAMATH_CALUDE_daffodil_cost_is_65_cents_l2030_203038

/-- Represents the cost of bulbs and garden space --/
structure BulbGarden where
  totalSpace : ℕ
  crocusCost : ℚ
  totalBudget : ℚ
  crocusCount : ℕ

/-- Calculates the cost of each daffodil bulb --/
def daffodilCost (g : BulbGarden) : ℚ :=
  let crocusTotalCost := g.crocusCost * g.crocusCount
  let remainingBudget := g.totalBudget - crocusTotalCost
  let daffodilCount := g.totalSpace - g.crocusCount
  remainingBudget / daffodilCount

/-- Theorem stating the cost of each daffodil bulb --/
theorem daffodil_cost_is_65_cents (g : BulbGarden)
  (h1 : g.totalSpace = 55)
  (h2 : g.crocusCost = 35/100)
  (h3 : g.totalBudget = 2915/100)
  (h4 : g.crocusCount = 22) :
  daffodilCost g = 65/100 := by
  sorry

#eval daffodilCost { totalSpace := 55, crocusCost := 35/100, totalBudget := 2915/100, crocusCount := 22 }

end NUMINAMATH_CALUDE_daffodil_cost_is_65_cents_l2030_203038


namespace NUMINAMATH_CALUDE_min_absolute_value_at_20_l2030_203050

/-- An arithmetic sequence with first term 14 and common difference -3/4 -/
def arithmeticSequence (n : ℕ) : ℚ :=
  14 + (n - 1 : ℚ) * (-3/4)

/-- The absolute value of the nth term of the arithmetic sequence -/
def absoluteValue (n : ℕ) : ℚ :=
  |arithmeticSequence n|

theorem min_absolute_value_at_20 :
  ∀ n : ℕ, n ≠ 0 → absoluteValue 20 ≤ absoluteValue n :=
sorry

end NUMINAMATH_CALUDE_min_absolute_value_at_20_l2030_203050


namespace NUMINAMATH_CALUDE_min_value_expression_l2030_203083

theorem min_value_expression (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : c ≠ 0) :
  ((a - c)^2 + (c - b)^2 + (b - a)^2) / c^2 ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2030_203083


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2030_203006

/-- The longest segment in a cylinder. -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 3) (hh : h = 8) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2030_203006


namespace NUMINAMATH_CALUDE_arithmetic_mean_value_l2030_203008

/-- A normal distribution with given properties -/
structure NormalDistribution where
  σ : ℝ  -- standard deviation
  x : ℝ  -- value 2 standard deviations below the mean
  h : x = μ - 2 * σ  -- relation between x, μ, and σ

/-- The arithmetic mean of a normal distribution satisfying given conditions -/
def arithmetic_mean (d : NormalDistribution) : ℝ := 
  d.x + 2 * d.σ

/-- Theorem stating the arithmetic mean of the specific normal distribution -/
theorem arithmetic_mean_value (d : NormalDistribution) 
  (h_σ : d.σ = 1.5) (h_x : d.x = 11) : arithmetic_mean d = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_value_l2030_203008


namespace NUMINAMATH_CALUDE_simplify_and_find_ratio_l2030_203078

theorem simplify_and_find_ratio (k : ℝ) : 
  (6 * k + 18) / 6 = k + 3 ∧ (1 : ℝ) / 3 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_find_ratio_l2030_203078


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l2030_203085

theorem quadratic_roots_expression (m n : ℝ) : 
  m^2 + 3*m + 1 = 0 → n^2 + 3*n + 1 = 0 → m * n = 1 → (3*m + 1) / (m^3 * n) = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l2030_203085


namespace NUMINAMATH_CALUDE_triangle_problem_l2030_203042

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π → 
  0 < B ∧ B < π → 
  0 < C ∧ C < π →
  A + B + C = π →
  2 * c - 2 * a * Real.cos B = b →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4 →
  c^2 + a * b * Real.cos C + a^2 = 4 →
  A = π/3 ∧ a = Real.sqrt 7 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2030_203042


namespace NUMINAMATH_CALUDE_fraction_simplification_l2030_203045

theorem fraction_simplification : (252 : ℚ) / 21 * 7 / 168 * 12 / 4 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2030_203045


namespace NUMINAMATH_CALUDE_algebraic_identities_l2030_203054

theorem algebraic_identities (m n x y z : ℝ) : 
  ((m + 2*n) - (m - 2*n) = 4*n) ∧
  (2*(x - 3) - (-x + 4) = 3*x - 10) ∧
  (2*x - 3*(x - 2*y + 3*x) + 2*(3*x - 3*y + 2*z) = -4*x + 4*z) ∧
  (8*m^2 - (4*m^2 - 2*m - 4*(2*m^2 - 5*m)) = 12*m^2 - 18*m) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l2030_203054


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_l2030_203089

theorem smallest_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (∃ k : ℕ, n = 3 * k + 1) ∧
  (∃ k : ℕ, n = 4 * k + 1) ∧
  (∃ k : ℕ, n = 5 * k + 1) ∧
  (∃ k : ℕ, n = 7 * k + 1) ∧
  (∀ m : ℕ, m < n → 
    (m < 100 ∨ m ≥ 1000 ∨
    (∀ k : ℕ, m ≠ 3 * k + 1) ∨
    (∀ k : ℕ, m ≠ 4 * k + 1) ∨
    (∀ k : ℕ, m ≠ 5 * k + 1) ∨
    (∀ k : ℕ, m ≠ 7 * k + 1))) ∧
  n = 421 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_l2030_203089


namespace NUMINAMATH_CALUDE_total_berries_l2030_203022

/-- The number of berries each person has -/
structure Berries where
  stacy : ℕ
  steve : ℕ
  skylar : ℕ

/-- The conditions of the berry distribution -/
def berry_conditions (b : Berries) : Prop :=
  b.stacy = 4 * b.steve ∧ 
  b.steve = 2 * b.skylar ∧ 
  b.stacy = 800

/-- The theorem stating the total number of berries -/
theorem total_berries (b : Berries) (h : berry_conditions b) : 
  b.stacy + b.steve + b.skylar = 1100 := by
  sorry

end NUMINAMATH_CALUDE_total_berries_l2030_203022


namespace NUMINAMATH_CALUDE_a_equals_base_conversion_l2030_203002

/-- Convert a natural number to a different base representation --/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- Interpret a list of digits in a given base as a natural number --/
def fromBase (digits : List ℕ) (base : ℕ) : ℕ :=
  sorry

/-- Check if a list of numbers forms an arithmetic sequence --/
def isArithmeticSequence (seq : List ℕ) : Bool :=
  sorry

/-- Define the sequence a_n as described in the problem --/
def a (p : ℕ) : ℕ → ℕ
  | n => if n < p - 1 then n else
    sorry -- Find the least positive integer not forming an arithmetic sequence

/-- Main theorem to prove --/
theorem a_equals_base_conversion {p : ℕ} (hp : Nat.Prime p) (hodd : Odd p) :
  ∀ n, a p n = fromBase (toBase n (p - 1)) p :=
  sorry

end NUMINAMATH_CALUDE_a_equals_base_conversion_l2030_203002


namespace NUMINAMATH_CALUDE_outfit_combinations_l2030_203035

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (excluded_combinations : ℕ) : 
  shirts = 5 → pants = 6 → excluded_combinations = 1 →
  shirts * pants - excluded_combinations = 29 := by
sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2030_203035


namespace NUMINAMATH_CALUDE_equation_solution_l2030_203010

theorem equation_solution (x : ℝ) : 
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 2) * (x - 1) / 
  ((x - 4) * (x - 2) * (x - 1)) = 1 →
  x = (9 + Real.sqrt 5) / 2 ∨ x = (9 - Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2030_203010


namespace NUMINAMATH_CALUDE_tan_value_from_sin_cos_equation_l2030_203088

theorem tan_value_from_sin_cos_equation (x : ℝ) 
  (h1 : 0 < x ∧ x < Real.pi / 2) 
  (h2 : Real.sin x ^ 4 / 42 + Real.cos x ^ 4 / 75 = 1 / 117) : 
  Real.tan x = Real.sqrt 14 / 5 := by
sorry

end NUMINAMATH_CALUDE_tan_value_from_sin_cos_equation_l2030_203088


namespace NUMINAMATH_CALUDE_seashells_after_giving_away_starfish_count_indeterminate_l2030_203081

/-- Proves that the number of seashells after giving some away is correct -/
theorem seashells_after_giving_away 
  (initial_seashells : ℕ) 
  (seashells_given_away : ℕ) 
  (final_seashells : ℕ) 
  (h1 : initial_seashells = 49)
  (h2 : seashells_given_away = 13)
  (h3 : final_seashells = 36) :
  final_seashells = initial_seashells - seashells_given_away :=
by sorry

/-- The number of starfish cannot be determined from the given information -/
theorem starfish_count_indeterminate 
  (initial_seashells : ℕ) 
  (seashells_given_away : ℕ) 
  (final_seashells : ℕ) 
  (starfish : ℕ) 
  (h1 : initial_seashells = 49)
  (h2 : seashells_given_away = 13)
  (h3 : final_seashells = 36) :
  ∃ (n : ℕ), starfish = n :=
by sorry

end NUMINAMATH_CALUDE_seashells_after_giving_away_starfish_count_indeterminate_l2030_203081


namespace NUMINAMATH_CALUDE_bernardo_receives_345_l2030_203003

/-- The distribution pattern for Bernardo: 2, 5, 8, 11, ... -/
def bernardoSequence (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- The sum of the first n terms in Bernardo's sequence -/
def bernardoSum (n : ℕ) : ℕ := n * (2 * 2 + (n - 1) * 3) / 2

/-- The total amount distributed -/
def totalAmount : ℕ := 1000

theorem bernardo_receives_345 :
  ∃ n : ℕ, bernardoSum n ≤ totalAmount ∧ 
  bernardoSum (n + 1) > totalAmount ∧ 
  bernardoSum n = 345 := by sorry

end NUMINAMATH_CALUDE_bernardo_receives_345_l2030_203003


namespace NUMINAMATH_CALUDE_joseph_running_distance_l2030_203067

/-- Calculates the total distance run over a number of days with a given initial distance and daily increase. -/
def totalDistance (initialDistance : ℕ) (dailyIncrease : ℕ) (days : ℕ) : ℕ :=
  (days * (2 * initialDistance + (days - 1) * dailyIncrease)) / 2

/-- Proves that given an initial distance of 900 meters, a daily increase of 200 meters,
    and running for 3 days, the total distance run is 3300 meters. -/
theorem joseph_running_distance :
  totalDistance 900 200 3 = 3300 := by
  sorry

end NUMINAMATH_CALUDE_joseph_running_distance_l2030_203067


namespace NUMINAMATH_CALUDE_triangle_properties_l2030_203098

/-- Triangle ABC with vertices A(0, 3), B(-2, -1), and C(4, 3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The given triangle -/
def givenTriangle : Triangle where
  A := (0, 3)
  B := (-2, -1)
  C := (4, 3)

/-- Equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The altitude from side AB -/
def altitudeAB (t : Triangle) : Line :=
  sorry

/-- The point symmetric to C with respect to line AB -/
def symmetricPointC (t : Triangle) : Point :=
  sorry

theorem triangle_properties (t : Triangle) (h : t = givenTriangle) :
  (altitudeAB t = Line.mk 1 2 (-10)) ∧
  (symmetricPointC t = Point.mk (-12/5) (31/5)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2030_203098


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2030_203027

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2030_203027


namespace NUMINAMATH_CALUDE_total_cupcakes_l2030_203066

theorem total_cupcakes (cupcakes_per_event : ℕ) (number_of_events : ℕ) 
  (h1 : cupcakes_per_event = 156) 
  (h2 : number_of_events = 12) : 
  cupcakes_per_event * number_of_events = 1872 := by
sorry

end NUMINAMATH_CALUDE_total_cupcakes_l2030_203066


namespace NUMINAMATH_CALUDE_nephews_count_l2030_203062

/-- The number of nephews Alden had 10 years ago -/
def alden_nephews_10_years_ago : ℕ := 50

/-- The number of nephews Alden has now -/
def alden_nephews_now : ℕ := 2 * alden_nephews_10_years_ago

/-- The number of additional nephews Vihaan has compared to Alden -/
def vihaan_additional_nephews : ℕ := 60

/-- The number of nephews Vihaan has -/
def vihaan_nephews : ℕ := alden_nephews_now + vihaan_additional_nephews

/-- The total number of nephews Alden and Vihaan have together -/
def total_nephews : ℕ := alden_nephews_now + vihaan_nephews

theorem nephews_count : total_nephews = 260 := by
  sorry

end NUMINAMATH_CALUDE_nephews_count_l2030_203062


namespace NUMINAMATH_CALUDE_solve_equation_l2030_203087

theorem solve_equation : ∃! x : ℝ, (x - 4)^4 = (1/16)⁻¹ := by
  use 6
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2030_203087


namespace NUMINAMATH_CALUDE_probability_product_not_odd_l2030_203020

/-- Represents a standard six-sided die -/
def Die : Type := Fin 6

/-- The set of all possible outcomes when rolling two dice -/
def TwoDiceOutcomes : Type := Die × Die

/-- Predicate to check if a number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- Predicate to check if the product of two die rolls is not odd -/
def productNotOdd (roll : TwoDiceOutcomes) : Prop :=
  ¬isOdd ((roll.1.val + 1) * (roll.2.val + 1))

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := 36

/-- The number of outcomes where the product is not odd -/
def favorableOutcomes : ℕ := 27

theorem probability_product_not_odd :
  (favorableOutcomes : ℚ) / totalOutcomes = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_product_not_odd_l2030_203020


namespace NUMINAMATH_CALUDE_josh_marbles_difference_l2030_203053

theorem josh_marbles_difference (initial_marbles : ℕ) (lost_marbles : ℕ) (found_marbles : ℕ) 
  (num_friends : ℕ) (marbles_per_friend : ℕ) : 
  initial_marbles = 85 → 
  lost_marbles = 46 → 
  found_marbles = 130 → 
  num_friends = 12 → 
  marbles_per_friend = 3 → 
  found_marbles - (lost_marbles + num_friends * marbles_per_friend) = 48 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_difference_l2030_203053


namespace NUMINAMATH_CALUDE_max_prime_factors_l2030_203028

theorem max_prime_factors (a b : ℕ+) 
  (h_gcd : (Finset.card (Nat.primeFactors (Nat.gcd a b))) = 8)
  (h_lcm : (Finset.card (Nat.primeFactors (Nat.lcm a b))) = 30)
  (h_fewer : (Finset.card (Nat.primeFactors a)) < (Finset.card (Nat.primeFactors b))) :
  (Finset.card (Nat.primeFactors a)) ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_max_prime_factors_l2030_203028


namespace NUMINAMATH_CALUDE_divisibility_by_36_l2030_203091

theorem divisibility_by_36 (n : ℤ) (h1 : n ≥ 5) (h2 : ¬ 2 ∣ n) (h3 : ¬ 3 ∣ n) : 
  36 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_36_l2030_203091


namespace NUMINAMATH_CALUDE_complex_division_l2030_203016

theorem complex_division (i : ℂ) : i^2 = -1 → (2 + 4*i) / i = 4 - 2*i := by sorry

end NUMINAMATH_CALUDE_complex_division_l2030_203016


namespace NUMINAMATH_CALUDE_ellipse_equation_l2030_203060

theorem ellipse_equation (a b : ℝ) (h1 : a = 6) (h2 : b = 5) (h3 : a > b) :
  ∃ (x y : ℝ), x^2 / 25 + y^2 / 36 = 1 ∧ 
  ∀ (u v : ℝ), (u^2 / b^2 + v^2 / a^2 = 1 ↔ x^2 / 25 + y^2 / 36 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2030_203060


namespace NUMINAMATH_CALUDE_conference_seating_arrangement_l2030_203079

theorem conference_seating_arrangement :
  ∃! y : ℕ, ∃ x : ℕ,
    (9 * x + 10 * y = 73) ∧
    (0 < x) ∧ (0 < y) ∧
    y = 1 := by
  sorry

end NUMINAMATH_CALUDE_conference_seating_arrangement_l2030_203079


namespace NUMINAMATH_CALUDE_ladder_problem_l2030_203005

theorem ladder_problem (ladder_length height base : ℝ) :
  ladder_length = 15 ∧ height = 12 ∧ ladder_length^2 = height^2 + base^2 → base = 9 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l2030_203005


namespace NUMINAMATH_CALUDE_quadratic_touches_x_axis_l2030_203092

/-- A quadratic function that touches the x-axis -/
def touches_x_axis (a : ℝ) : Prop :=
  ∃ x : ℝ, 2 * x^2 - 8 * x + a = 0 ∧
  ∀ y : ℝ, 2 * y^2 - 8 * y + a ≥ 0

/-- The value of 'a' for which the quadratic function touches the x-axis is 8 -/
theorem quadratic_touches_x_axis :
  ∃! a : ℝ, touches_x_axis a ∧ a = 8 :=
sorry

end NUMINAMATH_CALUDE_quadratic_touches_x_axis_l2030_203092


namespace NUMINAMATH_CALUDE_empty_cube_exists_l2030_203011

/-- Represents a 3D coordinate within the cube -/
structure Coord where
  x : Fin 5
  y : Fin 5
  z : Fin 5

/-- Represents the state of a unit cube -/
inductive CubeState
  | Occupied
  | Empty

/-- The type of the cube, mapping coordinates to cube states -/
def Cube := Coord → CubeState

/-- Checks if two coordinates are adjacent -/
def isAdjacent (c1 c2 : Coord) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ (c1.z = c2.z + 1 ∨ c1.z + 1 = c2.z)) ∨
  (c1.x = c2.x ∧ c1.z = c2.z ∧ (c1.y = c2.y + 1 ∨ c1.y + 1 = c2.y)) ∨
  (c1.y = c2.y ∧ c1.z = c2.z ∧ (c1.x = c2.x + 1 ∨ c1.x + 1 = c2.x))

/-- A function representing the movement of objects -/
def moveObjects (initial : Cube) : Cube :=
  sorry

theorem empty_cube_exists (initial : Cube) :
  (∀ c : Coord, initial c = CubeState.Occupied) →
  ∃ c : Coord, (moveObjects initial) c = CubeState.Empty :=
sorry

end NUMINAMATH_CALUDE_empty_cube_exists_l2030_203011


namespace NUMINAMATH_CALUDE_largest_k_is_correct_l2030_203044

/-- The largest natural number k for which there exists a natural number n 
    satisfying the inequality sin(n + 1) < sin(n + 2) < sin(n + 3) < ... < sin(n + k) -/
def largest_k : ℕ := 3

/-- Predicate that checks if the sine inequality holds for a given n and k -/
def sine_inequality (n k : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → Real.sin (n + i) < Real.sin (n + j)

theorem largest_k_is_correct :
  (∃ n : ℕ, sine_inequality n largest_k) ∧
  (∀ k : ℕ, k > largest_k → ¬∃ n : ℕ, sine_inequality n k) :=
sorry

end NUMINAMATH_CALUDE_largest_k_is_correct_l2030_203044


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2030_203071

/-- Given a geometric sequence with common ratio greater than 1,
    if the difference between the 5th and 1st term is 15,
    and the difference between the 4th and 2nd term is 6,
    then the 3rd term is 4. -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)  -- The sequence
  (q : ℝ)      -- Common ratio
  (h_geom : ∀ n, a (n + 1) = a n * q)  -- Geometric sequence property
  (h_q : q > 1)  -- Common ratio > 1
  (h_diff1 : a 5 - a 1 = 15)  -- Given condition
  (h_diff2 : a 4 - a 2 = 6)   -- Given condition
  : a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2030_203071


namespace NUMINAMATH_CALUDE_anne_final_caps_l2030_203007

def anne_initial_caps : ℕ := 10
def anne_found_caps : ℕ := 5

theorem anne_final_caps : anne_initial_caps + anne_found_caps = 15 := by
  sorry

end NUMINAMATH_CALUDE_anne_final_caps_l2030_203007


namespace NUMINAMATH_CALUDE_sports_enjoyment_misreporting_l2030_203025

theorem sports_enjoyment_misreporting (total : ℝ) (total_pos : 0 < total) :
  let enjoy := 0.7 * total
  let not_enjoy := 0.3 * total
  let enjoy_say_enjoy := 0.75 * enjoy
  let enjoy_say_not := 0.25 * enjoy
  let not_enjoy_say_not := 0.85 * not_enjoy
  let not_enjoy_say_enjoy := 0.15 * not_enjoy
  enjoy_say_not / (enjoy_say_not + not_enjoy_say_not) = 7 / 17 := by
sorry

end NUMINAMATH_CALUDE_sports_enjoyment_misreporting_l2030_203025


namespace NUMINAMATH_CALUDE_count_special_numbers_l2030_203057

/-- A function that returns the set of all divisors of a natural number -/
def divisors (n : ℕ) : Finset ℕ :=
  sorry

/-- A function that counts the number of divisors of a natural number -/
def divisor_count (n : ℕ) : ℕ :=
  sorry

/-- A function that counts the number of divisors of a natural number that are less than or equal to 10 -/
def divisors_leq_10_count (n : ℕ) : ℕ :=
  sorry

/-- The set of natural numbers from 1 to 100 with exactly four divisors, 
    at least three of which do not exceed 10 -/
def special_numbers : Finset ℕ :=
  sorry

theorem count_special_numbers : special_numbers.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_l2030_203057


namespace NUMINAMATH_CALUDE_pencil_distribution_l2030_203030

theorem pencil_distribution (total_students : ℕ) (total_pencils : ℕ) 
  (h1 : total_students = 36)
  (h2 : total_pencils = 50)
  (h3 : ∃ (a b c : ℕ), a + b + c = total_students ∧ a + 2*b + 3*c = total_pencils ∧ a = 2*(b + c)) :
  ∃ (a b c : ℕ), a + b + c = total_students ∧ a + 2*b + 3*c = total_pencils ∧ a = 2*(b + c) ∧ b = 10 := by
  sorry

#check pencil_distribution

end NUMINAMATH_CALUDE_pencil_distribution_l2030_203030


namespace NUMINAMATH_CALUDE_point_3_4_in_first_quadrant_l2030_203024

/-- A point is in the first quadrant if both its x and y coordinates are positive. -/
def is_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The point (3,4) lies in the first quadrant. -/
theorem point_3_4_in_first_quadrant : is_first_quadrant 3 4 := by
  sorry

end NUMINAMATH_CALUDE_point_3_4_in_first_quadrant_l2030_203024


namespace NUMINAMATH_CALUDE_mail_difference_l2030_203069

theorem mail_difference (monday tuesday wednesday thursday : ℕ) : 
  monday = 65 →
  tuesday = monday + 10 →
  wednesday < tuesday →
  thursday = wednesday + 15 →
  monday + tuesday + wednesday + thursday = 295 →
  tuesday - wednesday = 5 := by
sorry

end NUMINAMATH_CALUDE_mail_difference_l2030_203069


namespace NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l2030_203097

/-- A line in the 2D plane represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The fourth quadrant of the 2D plane -/
def fourth_quadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0}

/-- Theorem: A line Ax + By + C = 0 where AB < 0 and BC < 0 does not pass through the fourth quadrant -/
theorem line_not_in_fourth_quadrant (l : Line) 
    (h1 : l.A * l.B < 0) 
    (h2 : l.B * l.C < 0) : 
    ∀ p ∈ fourth_quadrant, l.A * p.1 + l.B * p.2 + l.C ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l2030_203097


namespace NUMINAMATH_CALUDE_total_yellow_balloons_l2030_203033

theorem total_yellow_balloons (fred sam mary : ℕ) 
  (h1 : fred = 5) 
  (h2 : sam = 6) 
  (h3 : mary = 7) : 
  fred + sam + mary = 18 := by
sorry

end NUMINAMATH_CALUDE_total_yellow_balloons_l2030_203033


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2030_203039

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 12,
    prove that the fourth term is 6. -/
theorem arithmetic_sequence_fourth_term (a d : ℝ) 
  (h : (a + 2*d) + (a + 4*d) = 12) : a + 3*d = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2030_203039


namespace NUMINAMATH_CALUDE_magnified_tissue_diameter_l2030_203046

/-- The diameter of a magnified image given the actual diameter and magnification factor -/
def magnified_diameter (actual_diameter : ℝ) (magnification_factor : ℝ) : ℝ :=
  actual_diameter * magnification_factor

/-- Theorem stating that for an actual diameter of 0.005 cm and a magnification factor of 1000,
    the magnified diameter is 5 cm -/
theorem magnified_tissue_diameter :
  magnified_diameter 0.005 1000 = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnified_tissue_diameter_l2030_203046


namespace NUMINAMATH_CALUDE_problem_statement_l2030_203037

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_abc : a * b * c = 1)
  (h_a_c : a + 1 / c = 7)
  (h_b_a : b + 1 / a = 16) :
  c + 1 / b = 25 / 111 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2030_203037


namespace NUMINAMATH_CALUDE_snowman_volume_l2030_203080

theorem snowman_volume (π : ℝ) (h : π > 0) : 
  let sphere_volume (r : ℝ) := (4 / 3) * π * r^3
  sphere_volume 4 + sphere_volume 6 + sphere_volume 8 = (3168 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_snowman_volume_l2030_203080


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l2030_203068

theorem number_exceeding_percentage : ∃ x : ℝ, x = 0.35 * x + 245 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l2030_203068


namespace NUMINAMATH_CALUDE_m_union_n_eq_n_l2030_203017

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def N : Set ℝ := {x : ℝ | x^2 < 2}

-- State the theorem
theorem m_union_n_eq_n : M ∪ N = N := by sorry

end NUMINAMATH_CALUDE_m_union_n_eq_n_l2030_203017


namespace NUMINAMATH_CALUDE_minimum_days_to_find_poisoned_apple_l2030_203031

def number_of_apples : ℕ := 2021

theorem minimum_days_to_find_poisoned_apple :
  ∀ (n : ℕ), n = number_of_apples →
  (∃ (k : ℕ), 2^k ≥ n ∧ ∀ (m : ℕ), 2^m ≥ n → k ≤ m) →
  (∃ (k : ℕ), k = 11 ∧ 2^k ≥ n ∧ ∀ (m : ℕ), 2^m ≥ n → k ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_minimum_days_to_find_poisoned_apple_l2030_203031


namespace NUMINAMATH_CALUDE_coin_collection_value_l2030_203086

theorem coin_collection_value (total_coins : ℕ) (two_dollar_coins : ℕ) : 
  total_coins = 275 →
  two_dollar_coins = 148 →
  (total_coins - two_dollar_coins) * 1 + two_dollar_coins * 2 = 423 := by
sorry

end NUMINAMATH_CALUDE_coin_collection_value_l2030_203086


namespace NUMINAMATH_CALUDE_solution_set_l2030_203049

theorem solution_set (x : ℝ) :
  (1 / Real.pi) ^ (-x + 1) > (1 / Real.pi) ^ (x^2 - x) ↔ x < -1 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l2030_203049
