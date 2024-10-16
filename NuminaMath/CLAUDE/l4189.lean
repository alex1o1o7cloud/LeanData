import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_product_zero_l4189_418975

theorem polynomial_product_zero (a : ℚ) (h : a = 5/3) :
  (6*a^3 - 11*a^2 + 3*a - 2) * (3*a - 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_zero_l4189_418975


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l4189_418956

theorem meaningful_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l4189_418956


namespace NUMINAMATH_CALUDE_total_wheels_calculation_l4189_418955

/-- The number of wheels on a four-wheeler -/
def wheels_per_four_wheeler : ℕ := 4

/-- The number of four-wheelers parked -/
def num_four_wheelers : ℕ := 13

/-- The total number of wheels for all four-wheelers -/
def total_wheels : ℕ := num_four_wheelers * wheels_per_four_wheeler

theorem total_wheels_calculation :
  total_wheels = 52 := by sorry

end NUMINAMATH_CALUDE_total_wheels_calculation_l4189_418955


namespace NUMINAMATH_CALUDE_back_seat_ticket_cost_l4189_418938

/-- Proves that the cost of back seat tickets is $45 given the concert conditions -/
theorem back_seat_ticket_cost
  (total_seats : ℕ)
  (main_seat_cost : ℕ)
  (total_revenue : ℕ)
  (back_seat_sold : ℕ)
  (h_total_seats : total_seats = 20000)
  (h_main_seat_cost : main_seat_cost = 55)
  (h_total_revenue : total_revenue = 955000)
  (h_back_seat_sold : back_seat_sold = 14500) :
  (total_revenue - (total_seats - back_seat_sold) * main_seat_cost) / back_seat_sold = 45 :=
by sorry

end NUMINAMATH_CALUDE_back_seat_ticket_cost_l4189_418938


namespace NUMINAMATH_CALUDE_ellipse_condition_l4189_418904

/-- A non-degenerate ellipse is represented by the equation x^2 + 9y^2 - 6x + 27y = b
    if and only if b > -145/4 -/
theorem ellipse_condition (b : ℝ) :
  (∃ (x y : ℝ), x^2 + 9*y^2 - 6*x + 27*y = b) ∧
  (∀ (x y : ℝ), x^2 + 9*y^2 - 6*x + 27*y = b → (x, y) ≠ (0, 0)) ↔
  b > -145/4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l4189_418904


namespace NUMINAMATH_CALUDE_chess_tournament_games_l4189_418984

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournamentGames (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 8 players, where each player plays twice with every other player, the total number of games played is 112 -/
theorem chess_tournament_games :
  tournamentGames 8 * 2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l4189_418984


namespace NUMINAMATH_CALUDE_P_has_negative_and_positive_roots_l4189_418921

def P (x : ℝ) : ℝ := x^7 - 2*x^6 - 7*x^4 - x^2 + 9

theorem P_has_negative_and_positive_roots :
  (∃ (a : ℝ), a < 0 ∧ P a = 0) ∧ (∃ (b : ℝ), b > 0 ∧ P b = 0) := by sorry

end NUMINAMATH_CALUDE_P_has_negative_and_positive_roots_l4189_418921


namespace NUMINAMATH_CALUDE_branches_per_tree_is_100_l4189_418900

/-- Represents the farm with trees and their branches -/
structure Farm where
  num_trees : ℕ
  leaves_per_subbranch : ℕ
  subbranches_per_branch : ℕ
  total_leaves : ℕ

/-- Calculates the number of branches per tree on the farm -/
def branches_per_tree (f : Farm) : ℕ :=
  f.total_leaves / (f.num_trees * f.subbranches_per_branch * f.leaves_per_subbranch)

/-- Theorem stating that the number of branches per tree is 100 -/
theorem branches_per_tree_is_100 (f : Farm) 
  (h1 : f.num_trees = 4)
  (h2 : f.leaves_per_subbranch = 60)
  (h3 : f.subbranches_per_branch = 40)
  (h4 : f.total_leaves = 96000) : 
  branches_per_tree f = 100 := by
  sorry

#eval branches_per_tree { num_trees := 4, leaves_per_subbranch := 60, subbranches_per_branch := 40, total_leaves := 96000 }

end NUMINAMATH_CALUDE_branches_per_tree_is_100_l4189_418900


namespace NUMINAMATH_CALUDE_circle_equation_and_line_intersection_l4189_418981

/-- Represents a circle with center on the x-axis -/
structure CircleOnXAxis where
  center : ℤ
  radius : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def is_tangent (circle : CircleOnXAxis) (line : Line) : Prop :=
  (|line.a * circle.center + line.c| / Real.sqrt (line.a^2 + line.b^2)) = circle.radius

def intersects_circle (circle : CircleOnXAxis) (line : Line) : Prop :=
  ∃ x y : ℝ, line.a * x + line.b * y + line.c = 0 ∧
             (x - circle.center)^2 + y^2 = circle.radius^2

theorem circle_equation_and_line_intersection
  (circle : CircleOnXAxis)
  (tangent_line : Line)
  (h_radius : circle.radius = 5)
  (h_tangent : is_tangent circle tangent_line)
  (h_tangent_eq : tangent_line.a = 4 ∧ tangent_line.b = 3 ∧ tangent_line.c = -29) :
  (∃ equation : ℝ → ℝ → Prop, ∀ x y, equation x y ↔ (x - 1)^2 + y^2 = 25) ∧
  (∀ a : ℝ, a > 0 →
    let intersecting_line : Line := { a := a, b := -1, c := 5 }
    intersects_circle circle intersecting_line ↔ a > 5/12) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_and_line_intersection_l4189_418981


namespace NUMINAMATH_CALUDE_fourth_sample_is_twenty_l4189_418963

/-- Represents a systematic sampling scheme. -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  first_sample : ℕ
  interval : ℕ

/-- Generates the nth sample number in a systematic sampling scheme. -/
def nth_sample (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first_sample + (n - 1) * s.interval

/-- The theorem stating that 20 is the fourth sample number. -/
theorem fourth_sample_is_twenty
  (total_students : ℕ)
  (h_total : total_students = 56)
  (sample : SystematicSample)
  (h_population : sample.population = total_students)
  (h_sample_size : sample.sample_size = 4)
  (h_first_sample : sample.first_sample = 6)
  (h_interval : sample.interval = total_students / sample.sample_size)
  (h_third_sample : nth_sample sample 3 = 34)
  (h_fourth_sample : nth_sample sample 4 = 48) :
  nth_sample sample 2 = 20 := by
  sorry


end NUMINAMATH_CALUDE_fourth_sample_is_twenty_l4189_418963


namespace NUMINAMATH_CALUDE_product_sum_of_three_numbers_l4189_418982

theorem product_sum_of_three_numbers 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 99) 
  (h2 : a + b + c = 19) : 
  a*b + b*c + a*c = 131 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_of_three_numbers_l4189_418982


namespace NUMINAMATH_CALUDE_x_needs_18_days_l4189_418970

/-- The time needed for x to finish the remaining work after y leaves -/
def remaining_time_for_x (x_time y_time y_worked : ℚ) : ℚ :=
  (1 - y_worked / y_time) * x_time

/-- Proof that x needs 18 days to finish the remaining work -/
theorem x_needs_18_days (x_time y_time y_worked : ℚ) 
  (hx : x_time = 36)
  (hy : y_time = 24)
  (hw : y_worked = 12) :
  remaining_time_for_x x_time y_time y_worked = 18 := by
  sorry

#eval remaining_time_for_x 36 24 12

end NUMINAMATH_CALUDE_x_needs_18_days_l4189_418970


namespace NUMINAMATH_CALUDE_combined_weight_calculation_l4189_418973

def combined_new_weight (orange_weight apple_weight : ℝ) 
  (orange_water_content apple_water_content : ℝ)
  (orange_water_loss apple_water_loss : ℝ) : ℝ :=
  let orange_water := orange_weight * orange_water_content
  let apple_water := apple_weight * apple_water_content
  let orange_pulp := orange_weight - orange_water
  let apple_pulp := apple_weight - apple_water
  let new_orange_water := orange_water * (1 - orange_water_loss)
  let new_apple_water := apple_water * (1 - apple_water_loss)
  (orange_pulp + new_orange_water) + (apple_pulp + new_apple_water)

theorem combined_weight_calculation :
  combined_new_weight 5 3 0.95 0.90 0.07 0.04 = 7.5595 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_calculation_l4189_418973


namespace NUMINAMATH_CALUDE_circle_symmetry_l4189_418998

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

-- Define Circle C₁
def circle_C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define Circle C₂
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

-- Define symmetry with respect to a line
def symmetric_points (x1 y1 x2 y2 : ℝ) : Prop :=
  line_of_symmetry ((x1 + x2) / 2) ((y1 + y2) / 2) ∧
  (x2 - x1) * (x2 - x1) = (y2 - y1) * (y2 - y1)

-- Theorem statement
theorem circle_symmetry :
  ∀ x1 y1 x2 y2 : ℝ,
  circle_C1 x1 y1 →
  circle_C2 x2 y2 →
  symmetric_points x1 y1 x2 y2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l4189_418998


namespace NUMINAMATH_CALUDE_grass_field_length_l4189_418961

/-- Represents a rectangular grass field with a surrounding path. -/
structure GrassField where
  length : ℝ
  width : ℝ
  pathWidth : ℝ

/-- Calculates the area of the path surrounding the grass field. -/
def pathArea (field : GrassField) : ℝ :=
  (field.length + 2 * field.pathWidth) * (field.width + 2 * field.pathWidth) - field.length * field.width

/-- Theorem stating the length of the grass field given specific conditions. -/
theorem grass_field_length : 
  ∀ (field : GrassField),
  field.width = 55 →
  field.pathWidth = 2.5 →
  pathArea field = 1250 →
  field.length = 190 := by
sorry

end NUMINAMATH_CALUDE_grass_field_length_l4189_418961


namespace NUMINAMATH_CALUDE_solve_equation_l4189_418983

/-- Given the equation 19(x + y) + 17 = 19(-x + y) - z, where x = 1, prove that z = -55 -/
theorem solve_equation (y : ℝ) : 
  ∃ (z : ℝ), 19 * (1 + y) + 17 = 19 * (-1 + y) - z ∧ z = -55 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4189_418983


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l4189_418916

/-- A line given by parametric equations is tangent to a circle. -/
theorem line_tangent_to_circle (α : Real) (h1 : α > π / 2) :
  (∃ t : Real, ∀ φ : Real,
    let x_line := t * Real.cos α
    let y_line := t * Real.sin α
    let x_circle := 4 + 2 * Real.cos φ
    let y_circle := 2 * Real.sin φ
    (x_line - x_circle)^2 + (y_line - y_circle)^2 = 4) →
  α = 5 * π / 6 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l4189_418916


namespace NUMINAMATH_CALUDE_arithmetic_sequence_square_root_l4189_418971

theorem arithmetic_sequence_square_root (x : ℝ) :
  x > 0 →
  (∃ d : ℝ, 2^2 + d = x^2 ∧ x^2 + d = 5^2) →
  x = Real.sqrt 14.5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_square_root_l4189_418971


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l4189_418914

theorem quadratic_roots_sum_bound (a b : ℤ) 
  (ha : a ≠ -1) (hb : b ≠ -1) 
  (h_roots : ∃ x y : ℤ, x ≠ y ∧ 
    x^2 + a*b*x + (a + b) = 0 ∧ 
    y^2 + a*b*y + (a + b) = 0) : 
  a + b ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l4189_418914


namespace NUMINAMATH_CALUDE_joint_equation_solver_l4189_418923

/-- Given two equations and two solutions, prove the value of a specific expression --/
theorem joint_equation_solver (a b : ℤ) :
  (a * (-3) + 5 * (-1) = 15) →
  (4 * (-3) - b * (-1) = -2) →
  (a * 5 + 5 * 4 = 15) →
  (4 * 5 - b * 4 = -2) →
  a^2018 + (-1/10 * b : ℚ)^2019 = 0 := by
  sorry

end NUMINAMATH_CALUDE_joint_equation_solver_l4189_418923


namespace NUMINAMATH_CALUDE_two_mice_boring_l4189_418977

/-- The sum of distances bored by two mice in n days -/
def S (n : ℕ) : ℚ :=
  let big_mouse := 2^n - 1  -- Sum of geometric sequence with a₁ = 1, r = 2
  let small_mouse := 2 - 1 / 2^(n-1)  -- Sum of geometric sequence with a₁ = 1, r = 1/2
  big_mouse + small_mouse

theorem two_mice_boring (n : ℕ) : S n = 2^n - 1/2^(n-1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_two_mice_boring_l4189_418977


namespace NUMINAMATH_CALUDE_problems_left_to_solve_l4189_418952

def math_test (total_problems : ℕ) (first_20min : ℕ) (second_20min : ℕ) : Prop :=
  total_problems = 75 ∧
  first_20min = 10 ∧
  second_20min = 2 * first_20min ∧
  total_problems - (first_20min + second_20min) = 45

theorem problems_left_to_solve :
  ∀ (total_problems first_20min second_20min : ℕ),
    math_test total_problems first_20min second_20min →
    total_problems - (first_20min + second_20min) = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_problems_left_to_solve_l4189_418952


namespace NUMINAMATH_CALUDE_spade_calculation_l4189_418950

-- Define the ⋄ operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_calculation : spade (spade 1 2) (spade 9 (spade 5 4)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l4189_418950


namespace NUMINAMATH_CALUDE_cube_frame_wire_ratio_l4189_418939

/-- The ratio of wire lengths used by two people constructing cube frames -/
theorem cube_frame_wire_ratio : 
  ∀ (wire_a wire_b : ℕ) (pieces_a : ℕ) (volume : ℕ),
  wire_a = 8 →
  pieces_a = 12 →
  wire_b = 2 →
  volume = wire_a^3 →
  (wire_a * pieces_a) / (wire_b * 12 * volume) = 1 / 128 :=
by sorry

end NUMINAMATH_CALUDE_cube_frame_wire_ratio_l4189_418939


namespace NUMINAMATH_CALUDE_hyperbola_center_trajectory_l4189_418908

/-- The equation of the trajectory of the center of a hyperbola -/
theorem hyperbola_center_trajectory 
  (x y m : ℝ) 
  (h : x^2 - y^2 - 6*m*x - 4*m*y + 5*m^2 - 1 = 0) : 
  2*x + 3*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_trajectory_l4189_418908


namespace NUMINAMATH_CALUDE_two_possible_products_l4189_418941

theorem two_possible_products (a b : ℝ) (ha : |a| = 5) (hb : |b| = 3) :
  ∃ (x y : ℝ), (∀ z, a * b = z → z = x ∨ z = y) ∧ x ≠ y :=
sorry

end NUMINAMATH_CALUDE_two_possible_products_l4189_418941


namespace NUMINAMATH_CALUDE_odd_products_fraction_l4189_418953

/-- The number of integers from 0 to 15 inclusive -/
def table_size : ℕ := 16

/-- The count of odd numbers from 0 to 15 inclusive -/
def odd_count : ℕ := 8

/-- The total number of entries in the multiplication table -/
def total_entries : ℕ := table_size * table_size

/-- The number of odd products in the multiplication table -/
def odd_products : ℕ := odd_count * odd_count

/-- The fraction of odd products in the multiplication table -/
def odd_fraction : ℚ := odd_products / total_entries

theorem odd_products_fraction :
  odd_fraction = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_odd_products_fraction_l4189_418953


namespace NUMINAMATH_CALUDE_profit_percentage_at_marked_price_l4189_418980

theorem profit_percentage_at_marked_price 
  (cost_price : ℝ) 
  (marked_price : ℝ) 
  (h1 : marked_price > 0) 
  (h2 : cost_price > 0)
  (h3 : 0.8 * marked_price = 1.2 * cost_price) : 
  (marked_price - cost_price) / cost_price = 0.5 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_at_marked_price_l4189_418980


namespace NUMINAMATH_CALUDE_frisbee_price_problem_l4189_418967

theorem frisbee_price_problem (total_frisbees : ℕ) (total_receipts : ℕ) (price_a : ℕ) (min_frisbees_b : ℕ) :
  total_frisbees = 60 →
  price_a = 3 →
  total_receipts = 204 →
  min_frisbees_b = 24 →
  ∃ (frisbees_a frisbees_b price_b : ℕ),
    frisbees_a + frisbees_b = total_frisbees ∧
    frisbees_b ≥ min_frisbees_b ∧
    price_a * frisbees_a + price_b * frisbees_b = total_receipts ∧
    price_b = 4 := by
  sorry

#check frisbee_price_problem

end NUMINAMATH_CALUDE_frisbee_price_problem_l4189_418967


namespace NUMINAMATH_CALUDE_equation_solution_l4189_418989

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = -4/3 ∧ x₂ = 3 ∧
  ∀ (x : ℝ), x ≠ 2/3 → x ≠ -4/3 →
  ((6*x + 4) / (3*x^2 + 6*x - 8) = (3*x) / (3*x - 2) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4189_418989


namespace NUMINAMATH_CALUDE_inequality_problem_l4189_418957

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1|

-- State the theorem
theorem inequality_problem (a : ℝ) :
  (∀ x, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 2) →
  (a = 2 ∧
   ∀ k, (∃ x, (f a x + f a (-x)) / 3 < |k|) ↔ k < -2/3 ∨ k > 2/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l4189_418957


namespace NUMINAMATH_CALUDE_binomial_coefficient_identity_a_l4189_418997

theorem binomial_coefficient_identity_a (r m k : ℕ) (h1 : k ≤ m) (h2 : m ≤ r) :
  Nat.choose r m * Nat.choose m k = Nat.choose r k * Nat.choose (r - k) (m - k) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_identity_a_l4189_418997


namespace NUMINAMATH_CALUDE_seating_arrangement_l4189_418962

theorem seating_arrangement (total_people : ℕ) (row_sizes : List ℕ) : 
  total_people = 65 →
  (∀ x ∈ row_sizes, x = 7 ∨ x = 8 ∨ x = 9) →
  (List.sum row_sizes = total_people) →
  (List.count 9 row_sizes = 1) :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangement_l4189_418962


namespace NUMINAMATH_CALUDE_tank_width_proof_l4189_418907

/-- Proves that a tank with given dimensions and plastering cost has a width of 12 meters -/
theorem tank_width_proof (length depth : ℝ) (cost_per_sqm total_cost : ℝ) :
  length = 25 →
  depth = 6 →
  cost_per_sqm = 0.30 →
  total_cost = 223.2 →
  ∃ width : ℝ,
    width = 12 ∧
    total_cost = cost_per_sqm * (length * width + 2 * (length * depth + width * depth)) :=
by sorry

end NUMINAMATH_CALUDE_tank_width_proof_l4189_418907


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l4189_418925

theorem gain_percent_calculation (C S : ℝ) (h : C > 0) :
  50 * C = 45 * S → (S - C) / C * 100 = 100 / 9 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l4189_418925


namespace NUMINAMATH_CALUDE_angle_in_complete_rotation_l4189_418919

theorem angle_in_complete_rotation (y : ℝ) : y + 90 = 360 → y = 270 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_complete_rotation_l4189_418919


namespace NUMINAMATH_CALUDE_kevins_cards_l4189_418992

theorem kevins_cards (x y : ℕ) : x + y = 8 * x → y = 7 * x := by
  sorry

end NUMINAMATH_CALUDE_kevins_cards_l4189_418992


namespace NUMINAMATH_CALUDE_min_draws_for_even_product_l4189_418966

theorem min_draws_for_even_product (n : ℕ) (h : n = 14) :
  let S := Finset.range n
  let evens := S.filter (λ x => x % 2 = 0)
  let odds := S.filter (λ x => x % 2 ≠ 0)
  odds.card + 1 = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_even_product_l4189_418966


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l4189_418988

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x < 2) ↔ (∀ x : ℝ, x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l4189_418988


namespace NUMINAMATH_CALUDE_cube_monotone_l4189_418976

theorem cube_monotone (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_monotone_l4189_418976


namespace NUMINAMATH_CALUDE_factor_tree_problem_l4189_418928

theorem factor_tree_problem (X Y Z W : ℕ) : 
  X = Y * Z ∧ 
  Y = 7 * 11 ∧ 
  Z = 2 * W ∧ 
  W = 3 * 2 → 
  X = 924 := by sorry

end NUMINAMATH_CALUDE_factor_tree_problem_l4189_418928


namespace NUMINAMATH_CALUDE_greatest_c_not_in_range_l4189_418934

def f (c : ℝ) (x : ℝ) : ℝ := x^2 + c*x + 15

theorem greatest_c_not_in_range : 
  ∀ c : ℤ, (∀ x : ℝ, f c x ≠ -9) ↔ c ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_c_not_in_range_l4189_418934


namespace NUMINAMATH_CALUDE_minimum_additional_candies_l4189_418986

theorem minimum_additional_candies 
  (initial_candies : ℕ) 
  (num_students : ℕ) 
  (additional_candies : ℕ) : 
  initial_candies = 237 →
  num_students = 31 →
  additional_candies = 11 →
  (∃ (candies_per_student : ℕ), 
    (initial_candies + additional_candies) = num_students * candies_per_student) ∧
  (∀ (x : ℕ), x < additional_candies →
    ¬(∃ (y : ℕ), (initial_candies + x) = num_students * y)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_additional_candies_l4189_418986


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4189_418927

def M : Set ℝ := {x | x^2 - 1 ≤ 0}
def N : Set ℝ := {x | x^2 - 3*x > 0}

theorem intersection_of_M_and_N : ∀ x : ℝ, x ∈ M ∩ N ↔ -1 ≤ x ∧ x < 0 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4189_418927


namespace NUMINAMATH_CALUDE_ruby_apples_left_l4189_418936

/-- The number of apples Ruby has left after Emily takes some away -/
def applesLeft (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

theorem ruby_apples_left :
  applesLeft 6357912 2581435 = 3776477 := by
  sorry

end NUMINAMATH_CALUDE_ruby_apples_left_l4189_418936


namespace NUMINAMATH_CALUDE_giraffe_zebra_ratio_is_three_to_one_l4189_418965

/-- Represents the zoo layout and animal distribution --/
structure Zoo where
  tiger_enclosures : ℕ
  zebra_enclosures_per_tiger : ℕ
  tigers_per_enclosure : ℕ
  zebras_per_enclosure : ℕ
  giraffes_per_enclosure : ℕ
  total_animals : ℕ

/-- Calculates the ratio of giraffe enclosures to zebra enclosures --/
def giraffe_zebra_enclosure_ratio (zoo : Zoo) : ℚ :=
  let total_zebra_enclosures := zoo.tiger_enclosures * zoo.zebra_enclosures_per_tiger
  let total_tigers := zoo.tiger_enclosures * zoo.tigers_per_enclosure
  let total_zebras := total_zebra_enclosures * zoo.zebras_per_enclosure
  let total_giraffes := zoo.total_animals - total_tigers - total_zebras
  let giraffe_enclosures := total_giraffes / zoo.giraffes_per_enclosure
  giraffe_enclosures / total_zebra_enclosures

/-- The main theorem stating the ratio of giraffe enclosures to zebra enclosures --/
theorem giraffe_zebra_ratio_is_three_to_one (zoo : Zoo)
  (h1 : zoo.tiger_enclosures = 4)
  (h2 : zoo.zebra_enclosures_per_tiger = 2)
  (h3 : zoo.tigers_per_enclosure = 4)
  (h4 : zoo.zebras_per_enclosure = 10)
  (h5 : zoo.giraffes_per_enclosure = 2)
  (h6 : zoo.total_animals = 144) :
  giraffe_zebra_enclosure_ratio zoo = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_giraffe_zebra_ratio_is_three_to_one_l4189_418965


namespace NUMINAMATH_CALUDE_intersection_A_B_l4189_418949

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l4189_418949


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l4189_418969

theorem arithmetic_calculation : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l4189_418969


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l4189_418945

def A : Set ℝ := {x | 2 * x - x^2 > 0}
def B : Set ℝ := {x | x > 1}

theorem set_intersection_theorem : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l4189_418945


namespace NUMINAMATH_CALUDE_not_not_or_implies_or_at_least_one_true_l4189_418924

theorem not_not_or_implies_or (p q : Prop) : ¬¬(p ∨ q) → (p ∨ q) := by
  sorry

theorem at_least_one_true (p q : Prop) : ¬¬(p ∨ q) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_not_or_implies_or_at_least_one_true_l4189_418924


namespace NUMINAMATH_CALUDE_sqrt_inequality_and_sum_of_squares_l4189_418935

theorem sqrt_inequality_and_sum_of_squares (a b c : ℝ) : 
  (Real.sqrt 6 + Real.sqrt 10 > 2 * Real.sqrt 3 + 2) ∧ 
  (a^2 + b^2 + c^2 ≥ a*b + b*c + a*c) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_and_sum_of_squares_l4189_418935


namespace NUMINAMATH_CALUDE_both_pipes_open_time_l4189_418951

/-- The time it takes for pipe p to fill the cistern alone -/
def p_time : ℚ := 12

/-- The time it takes for pipe q to fill the cistern alone -/
def q_time : ℚ := 15

/-- The additional time it takes for pipe q to fill the cistern after pipe p is turned off -/
def additional_time : ℚ := 6

/-- The theorem stating that the time both pipes are open together is 4 minutes -/
theorem both_pipes_open_time : 
  ∃ (t : ℚ), 
    t * (1 / p_time + 1 / q_time) + additional_time * (1 / q_time) = 1 ∧ 
    t = 4 := by
  sorry

end NUMINAMATH_CALUDE_both_pipes_open_time_l4189_418951


namespace NUMINAMATH_CALUDE_expression_evaluation_l4189_418979

theorem expression_evaluation (x : ℝ) (h : x < 2) :
  Real.sqrt ((x - 2) / (1 - (x - 3) / (x - 2))) = (2 - x) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4189_418979


namespace NUMINAMATH_CALUDE_jacks_recycling_l4189_418917

/-- Proves the number of cans Jack recycled given the deposit amounts and quantities of other items --/
theorem jacks_recycling
  (bottle_deposit : ℚ)
  (can_deposit : ℚ)
  (glass_deposit : ℚ)
  (num_bottles : ℕ)
  (num_glass : ℕ)
  (total_earnings : ℚ)
  (h1 : bottle_deposit = 10 / 100)
  (h2 : can_deposit = 5 / 100)
  (h3 : glass_deposit = 15 / 100)
  (h4 : num_bottles = 80)
  (h5 : num_glass = 50)
  (h6 : total_earnings = 25) :
  (total_earnings - (num_bottles * bottle_deposit + num_glass * glass_deposit)) / can_deposit = 190 := by
  sorry

end NUMINAMATH_CALUDE_jacks_recycling_l4189_418917


namespace NUMINAMATH_CALUDE_solve_for_y_l4189_418909

theorem solve_for_y (x y : ℚ) (h1 : x = 103) (h2 : x^3 * y - 2 * x^2 * y + x * y = 103030) : y = 10 / 103 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l4189_418909


namespace NUMINAMATH_CALUDE_minimum_value_and_range_l4189_418974

theorem minimum_value_and_range (a b : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, (|2*a + b| + |2*a - b|) / |a| ≥ 4) ∧
  (∀ x : ℝ, |2*a + b| + |2*a - b| ≥ |a| * (|2 + x| + |2 - x|) → -2 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_and_range_l4189_418974


namespace NUMINAMATH_CALUDE_meeting_arrangements_l4189_418999

def number_of_schools : ℕ := 3
def members_per_school : ℕ := 6
def total_members : ℕ := 18
def host_representatives : ℕ := 3
def other_representatives : ℕ := 1

def arrange_meeting : ℕ := 
  number_of_schools * 
  (Nat.choose members_per_school host_representatives) * 
  (Nat.choose members_per_school other_representatives) * 
  (Nat.choose members_per_school other_representatives)

theorem meeting_arrangements :
  arrange_meeting = 2160 :=
sorry

end NUMINAMATH_CALUDE_meeting_arrangements_l4189_418999


namespace NUMINAMATH_CALUDE_a_total_share_l4189_418958

def total_profit : ℚ := 9600
def a_investment : ℚ := 15000
def b_investment : ℚ := 25000
def management_fee_percentage : ℚ := 10 / 100

def total_investment : ℚ := a_investment + b_investment

def management_fee (profit : ℚ) : ℚ := management_fee_percentage * profit

def remaining_profit (profit : ℚ) : ℚ := profit - management_fee profit

def a_share_ratio : ℚ := a_investment / total_investment

theorem a_total_share :
  management_fee total_profit + (a_share_ratio * remaining_profit total_profit) = 4200 := by
  sorry

end NUMINAMATH_CALUDE_a_total_share_l4189_418958


namespace NUMINAMATH_CALUDE_cube_planes_parallel_l4189_418903

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Given two planes in a cube, determines if they are parallel -/
def are_planes_parallel (cube : Cube) (plane1 plane2 : Plane) : Prop :=
  -- The definition of parallel planes
  ∃ (k : ℝ), k ≠ 0 ∧ plane1.normal = k • plane2.normal

/-- Constructs the plane AB1D1 in the cube -/
def plane_AB1D1 (cube : Cube) : Plane :=
  -- Definition of plane AB1D1
  sorry

/-- Constructs the plane BC1D in the cube -/
def plane_BC1D (cube : Cube) : Plane :=
  -- Definition of plane BC1D
  sorry

/-- Theorem stating that in a cube, plane AB1D1 is parallel to plane BC1D -/
theorem cube_planes_parallel (cube : Cube) : 
  are_planes_parallel cube (plane_AB1D1 cube) (plane_BC1D cube) := by
  sorry

end NUMINAMATH_CALUDE_cube_planes_parallel_l4189_418903


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l4189_418920

theorem purely_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := m + 2 + (m - 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l4189_418920


namespace NUMINAMATH_CALUDE_kylie_jewelry_beads_l4189_418913

/-- The number of beaded necklaces Kylie makes on Monday -/
def monday_necklaces : ℕ := 10

/-- The number of beaded necklaces Kylie makes on Tuesday -/
def tuesday_necklaces : ℕ := 2

/-- The number of beaded bracelets Kylie makes on Wednesday -/
def wednesday_bracelets : ℕ := 5

/-- The number of beaded earrings Kylie makes on Wednesday -/
def wednesday_earrings : ℕ := 7

/-- The number of beads needed to make one beaded necklace -/
def beads_per_necklace : ℕ := 20

/-- The number of beads needed to make one beaded bracelet -/
def beads_per_bracelet : ℕ := 10

/-- The number of beads needed to make one beaded earring -/
def beads_per_earring : ℕ := 5

/-- The total number of beads Kylie uses to make her jewelry -/
def total_beads : ℕ := 
  (monday_necklaces + tuesday_necklaces) * beads_per_necklace +
  wednesday_bracelets * beads_per_bracelet +
  wednesday_earrings * beads_per_earring

theorem kylie_jewelry_beads : total_beads = 325 := by
  sorry

end NUMINAMATH_CALUDE_kylie_jewelry_beads_l4189_418913


namespace NUMINAMATH_CALUDE_kim_money_amount_l4189_418933

theorem kim_money_amount (sal phil : ℝ) (h1 : sal = 0.8 * phil) (h2 : sal + phil = 1.8) : 
  1.4 * sal = 1.12 := by
  sorry

end NUMINAMATH_CALUDE_kim_money_amount_l4189_418933


namespace NUMINAMATH_CALUDE_shifted_function_sum_l4189_418948

-- Define the original function
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- Define the shifted function
def g (x : ℝ) : ℝ := f (x + 6)

-- Define a, b, and c
def a : ℝ := 3
def b : ℝ := 38
def c : ℝ := 115

theorem shifted_function_sum (x : ℝ) : g x = a * x^2 + b * x + c ∧ a + b + c = 156 := by
  sorry

end NUMINAMATH_CALUDE_shifted_function_sum_l4189_418948


namespace NUMINAMATH_CALUDE_range_of_x_l4189_418960

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the interval [-1, 1]
def I : Set ℝ := Set.Icc (-1 : ℝ) 1

-- Define the theorem
theorem range_of_x (h1 : Monotone f) (h2 : Set.MapsTo f I I) 
  (h3 : ∀ x, f (x - 2) < f (1 - x)) :
  ∃ S : Set ℝ, S = Set.Ico 1 (3/2) ∧ ∀ x, x ∈ S ↔ 
    (x - 2 ∈ I ∧ 1 - x ∈ I ∧ f (x - 2) < f (1 - x)) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l4189_418960


namespace NUMINAMATH_CALUDE_cookie_count_indeterminate_l4189_418944

theorem cookie_count_indeterminate (total_bananas : ℕ) (num_boxes : ℕ) (bananas_per_box : ℕ) 
  (h1 : total_bananas = 40)
  (h2 : num_boxes = 8)
  (h3 : bananas_per_box = 5)
  (h4 : total_bananas = num_boxes * bananas_per_box) :
  ¬ ∃ (cookie_count : ℕ), ∀ (n : ℕ), cookie_count = n :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_count_indeterminate_l4189_418944


namespace NUMINAMATH_CALUDE_student_walking_distance_l4189_418991

theorem student_walking_distance 
  (total_distance : ℝ)
  (walking_speed : ℝ)
  (bus_speed_with_students : ℝ)
  (empty_bus_speed : ℝ)
  (h1 : total_distance = 1)
  (h2 : walking_speed = 4)
  (h3 : bus_speed_with_students = 40)
  (h4 : empty_bus_speed = 60)
  (h5 : ∀ x : ℝ, 0 < x ∧ x < 1 → 
    x / walking_speed = 
    (1 - x) / bus_speed_with_students + 
    (1 - 2*x) / empty_bus_speed) :
  ∃ x : ℝ, x = 5 / 37 ∧ 
    x / walking_speed = 
    (1 - x) / bus_speed_with_students + 
    (1 - 2*x) / empty_bus_speed :=
sorry

end NUMINAMATH_CALUDE_student_walking_distance_l4189_418991


namespace NUMINAMATH_CALUDE_missing_items_count_l4189_418922

def initial_tshirts : ℕ := 9

def initial_sweaters (t : ℕ) : ℕ := 2 * t

def final_sweaters : ℕ := 3

def final_tshirts (t : ℕ) : ℕ := 3 * t

def missing_items (init_t init_s final_t final_s : ℕ) : ℕ :=
  if final_t > init_t
  then init_s - final_s
  else (init_t - final_t) + (init_s - final_s)

theorem missing_items_count :
  missing_items initial_tshirts (initial_sweaters initial_tshirts) 
                (final_tshirts initial_tshirts) final_sweaters = 15 := by
  sorry

end NUMINAMATH_CALUDE_missing_items_count_l4189_418922


namespace NUMINAMATH_CALUDE_line_equation_l4189_418918

-- Define the point P
def P : ℝ × ℝ := (3, 0)

-- Define the two given lines
def line1 (x y : ℝ) : Prop := 2*x - y - 2 = 0
def line2 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define a general line passing through P
def line_through_P (m : ℝ) (x y : ℝ) : Prop := y - P.2 = m * (x - P.1)

-- Define the condition for P being the midpoint of AB
def P_is_midpoint (A B : ℝ × ℝ) : Prop := 
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- State the theorem
theorem line_equation : 
  ∃ (A B : ℝ × ℝ) (m : ℝ),
    line1 A.1 A.2 ∧ 
    line2 B.1 B.2 ∧ 
    line_through_P m A.1 A.2 ∧ 
    line_through_P m B.1 B.2 ∧ 
    P_is_midpoint A B ∧ 
    ∀ (x y : ℝ), line_through_P m x y ↔ 4*x - 5*y = 12 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l4189_418918


namespace NUMINAMATH_CALUDE_fraction_evaluation_l4189_418926

theorem fraction_evaluation : (5 * 6 + 4) / 8 = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l4189_418926


namespace NUMINAMATH_CALUDE_airport_distance_solution_l4189_418964

/-- Represents the problem of calculating the distance to the airport --/
def AirportDistanceProblem (initial_speed : ℝ) (speed_increase : ℝ) (stop_time : ℝ) (early_arrival : ℝ) : Prop :=
  ∃ (distance : ℝ) (initial_time : ℝ),
    -- The first portion is driven at the initial speed
    initial_speed * initial_time = 40 ∧
    -- The total time includes the initial drive, stop time, and the rest of the journey
    (distance - 40) / (initial_speed + speed_increase) + initial_time + stop_time = 
      (distance / initial_speed) - early_arrival ∧
    -- The total distance is 190 miles
    distance = 190

/-- Theorem stating that the solution to the problem is 190 miles --/
theorem airport_distance_solution :
  AirportDistanceProblem 40 20 0.25 0.25 := by
  sorry

#check airport_distance_solution

end NUMINAMATH_CALUDE_airport_distance_solution_l4189_418964


namespace NUMINAMATH_CALUDE_spaceship_travel_distance_l4189_418931

def earth_to_x : ℝ := 0.5
def x_to_y : ℝ := 0.1
def y_to_earth : ℝ := 0.1

theorem spaceship_travel_distance :
  earth_to_x + x_to_y + y_to_earth = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_spaceship_travel_distance_l4189_418931


namespace NUMINAMATH_CALUDE_dividing_trapezoid_mn_length_l4189_418915

/-- A trapezoid with bases a and b, and a segment MN parallel to the bases that divides the area in half -/
structure DividingTrapezoid (a b : ℝ) where
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (mn_length : ℝ)
  (mn_divides_area : mn_length ^ 2 = (a ^ 2 + b ^ 2) / 2)

/-- The length of MN in a DividingTrapezoid is √((a² + b²) / 2) -/
theorem dividing_trapezoid_mn_length (a b : ℝ) (t : DividingTrapezoid a b) :
  t.mn_length = Real.sqrt ((a ^ 2 + b ^ 2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_dividing_trapezoid_mn_length_l4189_418915


namespace NUMINAMATH_CALUDE_inequality_proof_l4189_418978

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4189_418978


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l4189_418940

theorem triangle_third_side_length 
  (a b : ℝ) 
  (cos_C : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 3) 
  (h3 : cos_C = -3/5) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*cos_C ∧ c = 2 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l4189_418940


namespace NUMINAMATH_CALUDE_prime_difference_theorem_l4189_418912

theorem prime_difference_theorem (m n : ℕ) : 
  Nat.Prime m → Nat.Prime n → m - n^2 = 2007 → m * n = 4022 := by
  sorry

end NUMINAMATH_CALUDE_prime_difference_theorem_l4189_418912


namespace NUMINAMATH_CALUDE_bill_harry_nuts_ratio_l4189_418932

theorem bill_harry_nuts_ratio : 
  ∀ (sue_nuts harry_nuts bill_nuts : ℕ),
    sue_nuts = 48 →
    harry_nuts = 2 * sue_nuts →
    bill_nuts + harry_nuts = 672 →
    bill_nuts = 6 * harry_nuts :=
by
  sorry

end NUMINAMATH_CALUDE_bill_harry_nuts_ratio_l4189_418932


namespace NUMINAMATH_CALUDE_complement_B_union_A_C_subset_B_implies_a_range_l4189_418993

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem for part (1)
theorem complement_B_union_A :
  (Set.univ \ B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ 9 ≤ x} :=
sorry

-- Theorem for part (2)
theorem C_subset_B_implies_a_range (a : ℝ) :
  C a ⊆ B → 2 ≤ a ∧ a ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_complement_B_union_A_C_subset_B_implies_a_range_l4189_418993


namespace NUMINAMATH_CALUDE_toy_blocks_difference_l4189_418996

theorem toy_blocks_difference (red_blocks yellow_blocks blue_blocks : ℕ) : 
  red_blocks = 18 →
  yellow_blocks = red_blocks + 7 →
  red_blocks + yellow_blocks + blue_blocks = 75 →
  blue_blocks - red_blocks = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_blocks_difference_l4189_418996


namespace NUMINAMATH_CALUDE_hotel_fee_proof_l4189_418968

/-- The flat fee for the first night in a hotel -/
def flat_fee : ℝ := 87.5

/-- The nightly fee for each subsequent night -/
def nightly_fee : ℝ := 52.5

/-- Alice's total cost for a 4-night stay -/
def alice_cost : ℝ := 245

/-- Bob's total cost for a 6-night stay -/
def bob_cost : ℝ := 350

/-- The number of nights in Alice's stay -/
def alice_nights : ℕ := 4

/-- The number of nights in Bob's stay -/
def bob_nights : ℕ := 6

theorem hotel_fee_proof :
  (flat_fee + (alice_nights - 1 : ℝ) * nightly_fee = alice_cost) ∧
  (flat_fee + (bob_nights - 1 : ℝ) * nightly_fee = bob_cost) :=
by sorry

#check hotel_fee_proof

end NUMINAMATH_CALUDE_hotel_fee_proof_l4189_418968


namespace NUMINAMATH_CALUDE_cookies_sum_l4189_418942

/-- The number of cookies Mona brought -/
def mona_cookies : ℕ := 20

/-- The number of cookies Jasmine brought -/
def jasmine_cookies : ℕ := mona_cookies - 5

/-- The number of cookies Rachel brought -/
def rachel_cookies : ℕ := jasmine_cookies + 10

/-- The total number of cookies brought by Mona, Jasmine, and Rachel -/
def total_cookies : ℕ := mona_cookies + jasmine_cookies + rachel_cookies

theorem cookies_sum : total_cookies = 60 := by
  sorry

end NUMINAMATH_CALUDE_cookies_sum_l4189_418942


namespace NUMINAMATH_CALUDE_trajectory_and_minimum_distance_l4189_418929

-- Define the points M and N
def M : ℝ × ℝ := (4, 0)
def N : ℝ × ℝ := (1, 0)

-- Define the line l
def l (x y : ℝ) : ℝ := x + 2*y - 12

-- Define the condition for point P
def P_condition (x y : ℝ) : Prop :=
  let MP := (x - M.1, y - M.2)
  let MN := (N.1 - M.1, N.2 - M.2)
  let NP := (x - N.1, y - N.2)
  MN.1 * MP.1 + MN.2 * MP.2 = 6 * Real.sqrt (NP.1^2 + NP.2^2)

-- State the theorem
theorem trajectory_and_minimum_distance :
  ∃ (Q : ℝ × ℝ),
    (∀ (x y : ℝ), P_condition x y ↔ x^2/4 + y^2/3 = 1) ∧
    Q = (1, 3/2) ∧
    (∀ (P : ℝ × ℝ), P_condition P.1 P.2 →
      |l P.1 P.2| / Real.sqrt 5 ≥ 8/5) ∧
    |l Q.1 Q.2| / Real.sqrt 5 = 8/5 :=
  sorry

end NUMINAMATH_CALUDE_trajectory_and_minimum_distance_l4189_418929


namespace NUMINAMATH_CALUDE_small_painting_price_l4189_418987

/-- Represents the price of paintings and sales data for Noah's art business -/
structure PaintingSales where
  large_price : ℕ
  small_price : ℕ
  last_month_large : ℕ
  last_month_small : ℕ
  this_month_total : ℕ

/-- Theorem stating that given the conditions, the price of a small painting is $30 -/
theorem small_painting_price (sales : PaintingSales) 
  (h1 : sales.large_price = 60)
  (h2 : sales.last_month_large = 8)
  (h3 : sales.last_month_small = 4)
  (h4 : sales.this_month_total = 1200) :
  sales.small_price = 30 := by
  sorry

#check small_painting_price

end NUMINAMATH_CALUDE_small_painting_price_l4189_418987


namespace NUMINAMATH_CALUDE_reflection_of_M_across_x_axis_l4189_418902

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The point M with coordinates (1, 2) -/
def M : ℝ × ℝ := (1, 2)

theorem reflection_of_M_across_x_axis :
  reflect_x M = (1, -2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_M_across_x_axis_l4189_418902


namespace NUMINAMATH_CALUDE_least_integer_square_triple_plus_80_l4189_418959

theorem least_integer_square_triple_plus_80 :
  ∃ x : ℤ, (∀ y : ℤ, y^2 = 3*y + 80 → x ≤ y) ∧ x^2 = 3*x + 80 :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_square_triple_plus_80_l4189_418959


namespace NUMINAMATH_CALUDE_rachel_albums_count_l4189_418990

/-- The number of songs per album -/
def songs_per_album : ℕ := 2

/-- The total number of songs Rachel bought -/
def total_songs : ℕ := 16

/-- The number of albums Rachel bought -/
def albums_bought : ℕ := total_songs / songs_per_album

theorem rachel_albums_count : albums_bought = 8 := by
  sorry

end NUMINAMATH_CALUDE_rachel_albums_count_l4189_418990


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4189_418937

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_formula : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The common difference of an arithmetic sequence is 3 given S_2 = 4 and S_4 = 20 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h2 : seq.S 2 = 4) 
  (h4 : seq.S 4 = 20) : 
  seq.a 2 - seq.a 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4189_418937


namespace NUMINAMATH_CALUDE_nested_radical_solution_l4189_418911

theorem nested_radical_solution :
  ∃ x : ℝ, x = Real.sqrt (3 - x) → x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_solution_l4189_418911


namespace NUMINAMATH_CALUDE_second_point_x_coordinate_l4189_418946

/-- Given two points on a line, prove the x-coordinate of the second point -/
theorem second_point_x_coordinate 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) -- First point (m, n) satisfies the line equation
  (h2 : m + 2 = 2 * (n + 1) + 5) -- Second point (m+2, n+1) satisfies the line equation
  : m + 2 = 2 * n + 7 := by
  sorry

end NUMINAMATH_CALUDE_second_point_x_coordinate_l4189_418946


namespace NUMINAMATH_CALUDE_vasya_drove_two_fifths_l4189_418906

/-- Represents the fraction of total distance driven by each person -/
structure DistanceFractions where
  anton : ℝ
  vasya : ℝ
  sasha : ℝ
  dima : ℝ

/-- Conditions of the driving problem -/
def driving_conditions (d : DistanceFractions) : Prop :=
  d.anton = d.vasya / 2 ∧
  d.sasha = d.anton + d.dima ∧
  d.dima = 1 / 10 ∧
  d.anton + d.vasya + d.sasha + d.dima = 1

/-- Theorem stating that Vasya drove 2/5 of the total distance -/
theorem vasya_drove_two_fifths :
  ∀ d : DistanceFractions, driving_conditions d → d.vasya = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_vasya_drove_two_fifths_l4189_418906


namespace NUMINAMATH_CALUDE_power_of_2_ending_probabilities_l4189_418905

/-- The probability that 2^n ends with the digit 2, where n is a randomly chosen positive integer -/
def prob_ends_with_2 : ℚ := 1 / 4

/-- The probability that 2^n ends with the digits 12, where n is a randomly chosen positive integer -/
def prob_ends_with_12 : ℚ := 1 / 20

/-- Theorem stating the probabilities for 2^n ending with 2 and 12 -/
theorem power_of_2_ending_probabilities :
  (prob_ends_with_2 = 1 / 4) ∧ (prob_ends_with_12 = 1 / 20) := by
  sorry

end NUMINAMATH_CALUDE_power_of_2_ending_probabilities_l4189_418905


namespace NUMINAMATH_CALUDE_felicity_gasoline_usage_l4189_418901

/-- Represents the fuel consumption and distance data for a road trip. -/
structure RoadTripData where
  felicity_mpg : ℝ
  adhira_mpg : ℝ
  benjamin_ethanol_mpg : ℝ
  benjamin_biodiesel_mpg : ℝ
  total_distance : ℝ
  adhira_felicity_diff : ℝ
  felicity_benjamin_diff : ℝ
  ethanol_ratio : ℝ
  biodiesel_ratio : ℝ
  felicity_adhira_fuel_ratio : ℝ
  benjamin_adhira_fuel_diff : ℝ

/-- Calculates the amount of gasoline used by Felicity given the road trip data. -/
def calculate_felicity_gasoline (data : RoadTripData) : ℝ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that Felicity used 56 gallons of gasoline on her trip. -/
theorem felicity_gasoline_usage (data : RoadTripData) 
    (h1 : data.felicity_mpg = 35)
    (h2 : data.adhira_mpg = 25)
    (h3 : data.benjamin_ethanol_mpg = 30)
    (h4 : data.benjamin_biodiesel_mpg = 20)
    (h5 : data.total_distance = 1750)
    (h6 : data.adhira_felicity_diff = 150)
    (h7 : data.felicity_benjamin_diff = 50)
    (h8 : data.ethanol_ratio = 0.35)
    (h9 : data.biodiesel_ratio = 0.65)
    (h10 : data.felicity_adhira_fuel_ratio = 2)
    (h11 : data.benjamin_adhira_fuel_diff = 5) :
  calculate_felicity_gasoline data = 56 := by
  sorry

end NUMINAMATH_CALUDE_felicity_gasoline_usage_l4189_418901


namespace NUMINAMATH_CALUDE_queen_heart_jack_probability_l4189_418947

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of queens in a standard deck -/
def QueenCount : ℕ := 4

/-- Number of hearts in a standard deck -/
def HeartCount : ℕ := 13

/-- Number of jacks in a standard deck -/
def JackCount : ℕ := 4

/-- Probability of drawing a queen as the first card, a heart as the second card, 
    and a jack as the third card from a standard 52-card deck -/
def probabilityQueenHeartJack : ℚ := 1 / 663

theorem queen_heart_jack_probability :
  probabilityQueenHeartJack = 
    (QueenCount / StandardDeck) * 
    (HeartCount / (StandardDeck - 1)) * 
    (JackCount / (StandardDeck - 2)) := by
  sorry

end NUMINAMATH_CALUDE_queen_heart_jack_probability_l4189_418947


namespace NUMINAMATH_CALUDE_ratio_problem_l4189_418954

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 5) :
  d / a = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l4189_418954


namespace NUMINAMATH_CALUDE_chads_birthday_money_l4189_418910

/-- Chad's savings problem -/
theorem chads_birthday_money (
  savings_rate : ℝ)
  (other_earnings : ℝ)
  (total_savings : ℝ)
  (birthday_money : ℝ) :
  savings_rate = 0.4 →
  other_earnings = 900 →
  total_savings = 460 →
  savings_rate * (other_earnings + birthday_money) = total_savings →
  birthday_money = 250 := by
  sorry

end NUMINAMATH_CALUDE_chads_birthday_money_l4189_418910


namespace NUMINAMATH_CALUDE_ryan_bus_trips_l4189_418943

/-- Represents Ryan's commuting schedule and times --/
structure CommuteSchedule where
  bike_time : ℕ
  bus_time : ℕ
  friend_time : ℕ
  bike_frequency : ℕ
  friend_frequency : ℕ
  total_time : ℕ

/-- Calculates the number of bus trips given a CommuteSchedule --/
def calculate_bus_trips (schedule : CommuteSchedule) : ℕ :=
  (schedule.total_time - 
   (schedule.bike_time * schedule.bike_frequency + 
    schedule.friend_time * schedule.friend_frequency)) / 
  schedule.bus_time

/-- Ryan's actual commute schedule --/
def ryan_schedule : CommuteSchedule :=
  { bike_time := 30
  , bus_time := 40
  , friend_time := 10
  , bike_frequency := 1
  , friend_frequency := 1
  , total_time := 160 }

/-- Theorem stating that Ryan takes the bus 3 times a week --/
theorem ryan_bus_trips : calculate_bus_trips ryan_schedule = 3 := by
  sorry

end NUMINAMATH_CALUDE_ryan_bus_trips_l4189_418943


namespace NUMINAMATH_CALUDE_right_triangle_squares_problem_l4189_418985

theorem right_triangle_squares_problem (x : ℝ) : 
  (3 * x)^2 + (6 * x)^2 + (1/2 * 3 * x * 6 * x) = 1200 → x = (10 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_squares_problem_l4189_418985


namespace NUMINAMATH_CALUDE_problem_solution_l4189_418995

theorem problem_solution (a b : ℝ) : 
  (∀ x : ℝ, (x + a) * (x + 8) = x^2 + b*x + 24) → 
  a + b = 14 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4189_418995


namespace NUMINAMATH_CALUDE_max_money_earned_is_zero_l4189_418994

/-- Represents the state of the three piles of stones -/
structure PileState where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ

/-- Represents a single move of a stone from one pile to another -/
inductive Move
  | oneToTwo
  | oneToThree
  | twoToOne
  | twoToThree
  | threeToOne
  | threeToTwo

/-- Applies a move to a pile state, returning the new state and the money earned -/
def applyMove (state : PileState) (move : Move) : PileState × ℤ := sorry

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to an initial state, returning the final state and total money earned -/
def applyMoves (initial : PileState) (moves : MoveSequence) : PileState × ℤ := sorry

/-- The main theorem: the maximum money that can be earned is 0 -/
theorem max_money_earned_is_zero (initial : PileState) :
  (∀ moves : MoveSequence, applyMoves initial moves = (initial, 0)) → 
  (∀ moves : MoveSequence, (applyMoves initial moves).2 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_max_money_earned_is_zero_l4189_418994


namespace NUMINAMATH_CALUDE_prob_reroll_two_dice_l4189_418930

-- Define a die as a natural number between 1 and 6
def Die := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define a roll of four dice
def FourDiceRoll := Die × Die × Die × Die

-- Function to calculate the sum of four dice
def diceSum (roll : FourDiceRoll) : ℕ := roll.1 + roll.2.1 + roll.2.2.1 + roll.2.2.2

-- Function to determine if a roll is a win (sum is 9)
def isWin (roll : FourDiceRoll) : Prop := diceSum roll = 9

-- Function to calculate the probability of winning by rerolling all four dice
def probWinRerollAll : ℚ := 56 / 1296

-- Function to calculate the probability of winning by rerolling two dice
def probWinRerollTwo (keptSum : ℕ) : ℚ :=
  if keptSum ≤ 7 then (9 - keptSum - 1) / 36
  else (13 - (9 - keptSum)) / 36

-- Theorem: The probability of Jason choosing to reroll exactly two dice is 1/18
theorem prob_reroll_two_dice : 
  (∀ roll : FourDiceRoll, ∃ (keptSum : ℕ), 
    (keptSum ≤ 8 ∧ probWinRerollTwo keptSum > probWinRerollAll) ∨
    (keptSum > 8 ∧ probWinRerollAll ≥ probWinRerollTwo keptSum)) →
  (2 : ℚ) / 36 = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_prob_reroll_two_dice_l4189_418930


namespace NUMINAMATH_CALUDE_eighteen_cubed_nine_cubed_l4189_418972

theorem eighteen_cubed_nine_cubed (L M : ℕ) : 18^3 * 9^3 = 2^L * 3^M → L = 3 ∧ M = 12 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_cubed_nine_cubed_l4189_418972
