import Mathlib

namespace NUMINAMATH_CALUDE_two_digit_multiple_of_eight_l448_44884

theorem two_digit_multiple_of_eight (A : Nat) : 
  (30 ≤ 10 * 3 + A) ∧ (10 * 3 + A < 40) ∧ (10 * 3 + A) % 8 = 0 → A = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_multiple_of_eight_l448_44884


namespace NUMINAMATH_CALUDE_one_in_set_implies_x_one_or_neg_one_l448_44833

theorem one_in_set_implies_x_one_or_neg_one (x : ℝ) :
  (1 ∈ ({x, x^2} : Set ℝ)) → (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_one_in_set_implies_x_one_or_neg_one_l448_44833


namespace NUMINAMATH_CALUDE_price_increase_to_equality_l448_44859

theorem price_increase_to_equality (price_B : ℝ) (price_A : ℝ) 
    (h1 : price_A = price_B * 0.8) : 
  (price_B - price_A) / price_A * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_to_equality_l448_44859


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l448_44878

/-- The maximum marks for an exam -/
def maximum_marks : ℕ := 467

/-- The passing percentage as a rational number -/
def passing_percentage : ℚ := 60 / 100

/-- The marks scored by the student -/
def student_marks : ℕ := 200

/-- The marks by which the student failed -/
def failing_margin : ℕ := 80

theorem exam_maximum_marks :
  (↑maximum_marks * passing_percentage : ℚ).ceil = student_marks + failing_margin ∧
  (↑maximum_marks * passing_percentage : ℚ).ceil < ↑maximum_marks ∧
  (↑(maximum_marks - 1) * passing_percentage : ℚ).ceil < student_marks + failing_margin :=
sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l448_44878


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l448_44823

theorem smallest_n_for_inequality :
  ∃ (n : ℕ), n = 3 ∧ 
  (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z : ℝ), (x^2 + y^2 + z^2)^2 > m * (x^4 + y^4 + z^4)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l448_44823


namespace NUMINAMATH_CALUDE_units_digit_of_M_M12_is_1_l448_44828

/-- Modified Lucas sequence -/
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | (n + 2) => M (n + 1) + M n

/-- The 12th term of the Modified Lucas sequence -/
def M12 : ℕ := M 12

/-- Theorem stating that the units digit of M_{M₁₂} is 1 -/
theorem units_digit_of_M_M12_is_1 : M M12 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_M_M12_is_1_l448_44828


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l448_44867

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the path length of vertex C when rotating a triangle along a rectangle -/
def pathLengthC (t : Triangle) (r : Rectangle) : ℝ :=
  sorry

theorem triangle_rotation_path_length :
  let t : Triangle := { a := 2, b := 3, c := 4 }
  let r : Rectangle := { width := 8, height := 6 }
  pathLengthC t r = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l448_44867


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l448_44873

/-- A line passing through (2, 1) with equal intercepts on x and y axes -/
structure EqualInterceptLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (2, 1)
  point_condition : 1 = 2 * m + b
  -- The line has equal intercepts on x and y axes
  equal_intercepts : (m ≠ -1 → -b / (1 + m) = -b / m) ∧ (m = -1 → b = 0)

/-- The equation of the line is either x+y-3=0 or y = 1/2x -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = -1 ∧ l.b = 3) ∨ (l.m = 1/2 ∧ l.b = 0) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l448_44873


namespace NUMINAMATH_CALUDE_find_number_l448_44846

theorem find_number : ∃! x : ℝ, ((x * 2) - 37 + 25) / 8 = 5 := by sorry

end NUMINAMATH_CALUDE_find_number_l448_44846


namespace NUMINAMATH_CALUDE_family_buffet_employees_l448_44806

theorem family_buffet_employees (total : ℕ) (dining : ℕ) (snack : ℕ) (two_restaurants : ℕ) (all_restaurants : ℕ) : 
  total = 39 →
  dining = 18 →
  snack = 12 →
  two_restaurants = 4 →
  all_restaurants = 3 →
  ∃ family : ℕ, family = 20 ∧ 
    family + dining + snack - two_restaurants - 2 * all_restaurants + all_restaurants = total :=
by sorry

end NUMINAMATH_CALUDE_family_buffet_employees_l448_44806


namespace NUMINAMATH_CALUDE_specific_plot_fencing_cost_l448_44813

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  fencingCostPerMeter : ℝ

/-- Calculates the total fencing cost for a rectangular plot -/
def totalFencingCost (plot : RectangularPlot) : ℝ :=
  plot.perimeter * plot.fencingCostPerMeter

/-- Theorem stating the total fencing cost for a specific rectangular plot -/
theorem specific_plot_fencing_cost :
  ∃ (plot : RectangularPlot),
    plot.length = plot.width + 10 ∧
    plot.perimeter = 180 ∧
    plot.fencingCostPerMeter = 6.5 ∧
    totalFencingCost plot = 1170 := by
  sorry

end NUMINAMATH_CALUDE_specific_plot_fencing_cost_l448_44813


namespace NUMINAMATH_CALUDE_inequality_proof_l448_44899

theorem inequality_proof (n : ℕ+) (x : ℝ) (hx : x > 0) :
  x + (n : ℝ)^(n : ℕ) / x^(n : ℕ) ≥ (n : ℝ) + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l448_44899


namespace NUMINAMATH_CALUDE_platform_length_relation_l448_44860

/-- Given a train of length L that passes a pole in t seconds and a platform in 3.5t seconds
    at a constant velocity, prove that the length of the platform is 2.5 times the length of the train. -/
theorem platform_length_relation (L t : ℝ) (h1 : L > 0) (h2 : t > 0) :
  ∃ (P : ℝ), P = 2.5 * L ∧ L / t = (L + P) / (3.5 * t) := by
  sorry

end NUMINAMATH_CALUDE_platform_length_relation_l448_44860


namespace NUMINAMATH_CALUDE_alice_bracelet_profit_l448_44815

/-- Alice's bracelet sale profit calculation -/
theorem alice_bracelet_profit :
  ∀ (total_bracelets : ℕ) 
    (material_cost given_away price : ℚ),
  total_bracelets = 52 →
  material_cost = 3 →
  given_away = 8 →
  price = 1/4 →
  (total_bracelets - given_away : ℚ) * price - material_cost = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_alice_bracelet_profit_l448_44815


namespace NUMINAMATH_CALUDE_container_capacity_l448_44894

/-- Given that 8 liters is 20% of a container's capacity, prove that 40 such containers have a total capacity of 1600 liters. -/
theorem container_capacity (container_capacity : ℝ) 
  (h1 : 8 = 0.2 * container_capacity) 
  (h2 : container_capacity > 0) : 
  40 * container_capacity = 1600 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l448_44894


namespace NUMINAMATH_CALUDE_soccer_league_games_l448_44895

/-- The number of games played in a soccer league where each team plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a soccer league with 10 teams, where each team plays every other team once, 
    the total number of games played is 45 -/
theorem soccer_league_games : games_played 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l448_44895


namespace NUMINAMATH_CALUDE_triangle_altitude_segment_l448_44844

theorem triangle_altitude_segment (a b c h x : ℝ) : 
  a = 40 → b = 90 → c = 100 → 
  a^2 = x^2 + h^2 → 
  b^2 = (c - x)^2 + h^2 → 
  c - x = 82.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_segment_l448_44844


namespace NUMINAMATH_CALUDE_roles_assignment_count_l448_44847

/-- The number of ways to assign n distinct roles to n different people. -/
def assignRoles (n : ℕ) : ℕ := Nat.factorial n

/-- There are four team members. -/
def numTeamMembers : ℕ := 4

/-- There are four different roles. -/
def numRoles : ℕ := 4

/-- Each person can only take one role. -/
axiom one_role_per_person : numTeamMembers = numRoles

theorem roles_assignment_count :
  assignRoles numTeamMembers = 24 :=
sorry

end NUMINAMATH_CALUDE_roles_assignment_count_l448_44847


namespace NUMINAMATH_CALUDE_no_double_application_function_l448_44861

theorem no_double_application_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2013 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l448_44861


namespace NUMINAMATH_CALUDE_binomial_sum_equality_l448_44889

theorem binomial_sum_equality (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) : 
  (∑' d, Nat.choose (n - r + 1) d * Nat.choose (r - 1) (d - 1)) = Nat.choose n r :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_equality_l448_44889


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l448_44870

theorem quadratic_no_real_roots
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : c > 0)
  (h3 : |a - b| < c) :
  ∀ x : ℝ, a^2 * x^2 + (b^2 + a^2 - c^2) * x + b^2 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l448_44870


namespace NUMINAMATH_CALUDE_student_can_escape_l448_44881

-- Define the square pond
structure SquarePond where
  side : ℝ
  side_positive : side > 0

-- Define the student and teacher
structure Participant where
  swim_speed : ℝ
  run_speed : ℝ
  swim_speed_positive : swim_speed > 0
  run_speed_positive : run_speed > 0

-- Define the problem setup
def setup (pond : SquarePond) (student teacher : Participant) : Prop :=
  teacher.run_speed = 4 * student.swim_speed ∧
  student.run_speed > teacher.run_speed

-- Theorem statement
theorem student_can_escape (pond : SquarePond) (student teacher : Participant) 
  (h : setup pond student teacher) :
  (pond.side / (student.swim_speed * Real.sqrt 2)) < (pond.side / (2 * teacher.run_speed)) :=
sorry

end NUMINAMATH_CALUDE_student_can_escape_l448_44881


namespace NUMINAMATH_CALUDE_braking_distance_problems_l448_44839

/-- Braking distance formula -/
def braking_distance (t v k : ℝ) : ℝ := t * v + k * v^2

/-- Braking coefficient -/
def k : ℝ := 0.1

/-- Initial reaction time before alcohol consumption -/
def t_initial : ℝ := 0.5

theorem braking_distance_problems :
  /- (1) -/
  braking_distance t_initial 10 k = 15 ∧
  /- (2) -/
  ∃ t : ℝ, braking_distance t 15 k = 52.5 ∧ t = 2 ∧
  /- (3) -/
  braking_distance 2 10 k = 30 ∧
  /- (4) -/
  braking_distance 2 10 k - braking_distance t_initial 10 k = 15 ∧
  /- (5) -/
  ∀ t : ℝ, braking_distance t 12 k < 42 → t < 2.3 :=
by sorry

end NUMINAMATH_CALUDE_braking_distance_problems_l448_44839


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l448_44842

theorem right_triangle_perimeter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_area : (1/2) * a * b = 150) (h_leg : a = 30) : 
  a + b + c = 40 + 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l448_44842


namespace NUMINAMATH_CALUDE_angle_bisector_equation_l448_44855

/-- Given three lines y = x, y = 3x, and y = -x intersecting at the origin,
    the angle bisector of the smallest acute angle that passes through (1, 1)
    has the equation y = (2 - √11/2)x -/
theorem angle_bisector_equation (x y : ℝ) :
  let line1 : ℝ → ℝ := λ t => t
  let line2 : ℝ → ℝ := λ t => 3 * t
  let line3 : ℝ → ℝ := λ t => -t
  let bisector : ℝ → ℝ := λ t => (2 - Real.sqrt 11 / 2) * t
  (∀ t, line1 t = t ∧ line2 t = 3 * t ∧ line3 t = -t) →
  (bisector 0 = 0) →
  (bisector 1 = 1) →
  (∀ t, bisector t = (2 - Real.sqrt 11 / 2) * t) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_equation_l448_44855


namespace NUMINAMATH_CALUDE_tan_sum_product_fifteen_thirty_l448_44858

theorem tan_sum_product_fifteen_thirty : 
  Real.tan (15 * π / 180) + Real.tan (30 * π / 180) + Real.tan (15 * π / 180) * Real.tan (30 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_fifteen_thirty_l448_44858


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l448_44886

theorem quadratic_roots_problem (α β b : ℝ) : 
  (∀ x, x^2 + b*x - 1 = 0 ↔ x = α ∨ x = β) →
  α * β - 2*α - 2*β = -11 →
  b = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l448_44886


namespace NUMINAMATH_CALUDE_distance_to_point_distance_from_origin_to_point_l448_44897

theorem distance_to_point : ℝ → ℝ → ℝ
  | x, y => Real.sqrt (x^2 + y^2)

theorem distance_from_origin_to_point : distance_to_point (-8) 15 = 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_distance_from_origin_to_point_l448_44897


namespace NUMINAMATH_CALUDE_equation_solution_l448_44804

theorem equation_solution (b c : ℝ) (θ : ℝ) :
  let x := (b^2 - c^2 * Real.sin θ^2) / (2 * b)
  x^2 + c^2 * Real.sin θ^2 = (b - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l448_44804


namespace NUMINAMATH_CALUDE_arthur_dinner_cost_theorem_l448_44834

/-- Calculates the total cost of Arthur's dinner, including tips --/
def arthurDinnerCost (appetizer_cost dessert_cost entree_cost wine_cost : ℝ)
  (entree_discount appetizer_discount dessert_discount bill_discount tax_rate waiter_tip_rate busser_tip_rate : ℝ) : ℝ :=
  let discounted_entree := entree_cost * (1 - entree_discount)
  let subtotal := discounted_entree + 2 * wine_cost
  let discounted_subtotal := subtotal * (1 - bill_discount)
  let tax := discounted_subtotal * tax_rate
  let total_with_tax := discounted_subtotal + tax
  let original_cost := appetizer_cost + entree_cost + 2 * wine_cost + dessert_cost
  let original_with_tax := original_cost * (1 + tax_rate)
  let waiter_tip := original_with_tax * waiter_tip_rate
  let total_with_waiter_tip := total_with_tax + waiter_tip
  let busser_tip := total_with_waiter_tip * busser_tip_rate
  total_with_waiter_tip + busser_tip

/-- Theorem stating that Arthur's dinner cost is $38.556 --/
theorem arthur_dinner_cost_theorem :
  arthurDinnerCost 8 7 30 4 0.4 1 1 0.1 0.08 0.2 0.05 = 38.556 := by
  sorry


end NUMINAMATH_CALUDE_arthur_dinner_cost_theorem_l448_44834


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l448_44817

theorem quadratic_root_ratio (c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (4 * x₁^2 - 5 * x₁ + c = 0) ∧ 
    (4 * x₂^2 - 5 * x₂ + c = 0) ∧ 
    (x₁ / x₂ = -3/4)) →
  c = -75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l448_44817


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l448_44892

/-- Determines if the equation x²/(k-4) - y²/(k+4) = 1 represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop := (k - 4) * (k + 4) > 0

theorem sufficient_but_not_necessary :
  (∀ k : ℝ, k ≤ -5 → is_hyperbola k) ∧
  (∃ k : ℝ, k > -5 ∧ is_hyperbola k) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l448_44892


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l448_44812

/-- A right triangle with perimeter 40 and area 30 has a hypotenuse of length 74/4 -/
theorem right_triangle_hypotenuse : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  a + b + c = 40 →   -- perimeter condition
  a * b / 2 = 30 →   -- area condition
  c = 74 / 4 := by
    sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l448_44812


namespace NUMINAMATH_CALUDE_find_x_value_l448_44857

theorem find_x_value (x y z : ℝ) 
  (h1 : x ≠ 0)
  (h2 : x / 3 = z + 2 * y^2)
  (h3 : x / 6 = 3 * z - y) :
  x = 168 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l448_44857


namespace NUMINAMATH_CALUDE_intersection_condition_chord_length_condition_l448_44811

-- Define the ellipse and line
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1
def line (x y m : ℝ) : Prop := y = x + m

-- Theorem for intersection condition
theorem intersection_condition (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 5 / 2 ≤ m ∧ m ≤ Real.sqrt 5 / 2 :=
sorry

-- Theorem for chord length condition
theorem chord_length_condition (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line x₁ y₁ m ∧ line x₂ y₂ m ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 * Real.sqrt 2 / 5) →
  m = 1/2 ∨ m = -1/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_chord_length_condition_l448_44811


namespace NUMINAMATH_CALUDE_graph_connectivity_probability_l448_44827

def num_vertices : Nat := 20
def num_edges_removed : Nat := 35

theorem graph_connectivity_probability :
  let total_edges := num_vertices * (num_vertices - 1) / 2
  let remaining_edges := total_edges - num_edges_removed
  let prob_connected := 1 - (num_vertices * Nat.choose remaining_edges (remaining_edges - num_vertices + 1)) / Nat.choose total_edges num_edges_removed
  prob_connected = 1 - (20 * Nat.choose 171 16) / Nat.choose 190 35 := by
  sorry

end NUMINAMATH_CALUDE_graph_connectivity_probability_l448_44827


namespace NUMINAMATH_CALUDE_min_cost_halloween_bags_l448_44852

/-- Represents the cost calculation for Halloween goodie bags --/
def halloween_bags_cost (total_students : ℕ) (vampire_count : ℕ) (pumpkin_count : ℕ) 
  (pack_size : ℕ) (pack_cost : ℕ) (individual_cost : ℕ) : ℕ := 
  let vampire_packs := vampire_count / pack_size
  let vampire_individuals := vampire_count % pack_size
  let pumpkin_packs := pumpkin_count / pack_size
  let pumpkin_individuals := pumpkin_count % pack_size
  vampire_packs * pack_cost + vampire_individuals * individual_cost +
  pumpkin_packs * pack_cost + pumpkin_individuals * individual_cost

/-- Theorem stating the minimum cost for Halloween goodie bags --/
theorem min_cost_halloween_bags : 
  halloween_bags_cost 25 11 14 5 3 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_halloween_bags_l448_44852


namespace NUMINAMATH_CALUDE_power_of_8_mod_100_l448_44872

theorem power_of_8_mod_100 : 8^2023 % 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_8_mod_100_l448_44872


namespace NUMINAMATH_CALUDE_rectangle_in_square_l448_44848

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the arrangement of rectangles in the square -/
def arrangement (r : Rectangle) : ℝ := 2 * r.length + 2 * r.width

/-- The theorem stating the properties of the rectangles in the square -/
theorem rectangle_in_square (r : Rectangle) : 
  arrangement r = 18 ∧ 3 * r.length = 18 → r.length = 6 ∧ r.width = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_in_square_l448_44848


namespace NUMINAMATH_CALUDE_quarters_fraction_l448_44818

/-- The number of state quarters in Stephanie's collection -/
def total_quarters : ℕ := 25

/-- The number of states that joined the union from 1800 to 1809 -/
def states_1800_1809 : ℕ := 8

/-- The fraction of quarters representing states that joined from 1800 to 1809 -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_quarters

theorem quarters_fraction :
  fraction_1800_1809 = 8 / 25 := by sorry

end NUMINAMATH_CALUDE_quarters_fraction_l448_44818


namespace NUMINAMATH_CALUDE_y_bound_l448_44829

theorem y_bound (x y : ℝ) (hx : x = 7) (heq : (x - 2*y)^y = 0.001) : 
  0 < y ∧ y < 7/2 := by
  sorry

end NUMINAMATH_CALUDE_y_bound_l448_44829


namespace NUMINAMATH_CALUDE_pepperoni_pizza_coverage_l448_44832

theorem pepperoni_pizza_coverage (pizza_diameter : ℝ) (pepperoni_count : ℕ) 
  (pepperoni_across : ℕ) : 
  pizza_diameter = 12 →
  pepperoni_across = 8 →
  pepperoni_count = 32 →
  (pepperoni_count * (pizza_diameter / pepperoni_across / 2)^2) / 
  (pizza_diameter / 2)^2 = 1 / 2 := by
  sorry

#check pepperoni_pizza_coverage

end NUMINAMATH_CALUDE_pepperoni_pizza_coverage_l448_44832


namespace NUMINAMATH_CALUDE_benjamin_walks_158_miles_l448_44803

/-- Calculates the total miles Benjamin walks in a week -/
def total_miles_walked : ℕ :=
  let work_distance := 8
  let work_days := 5
  let dog_walk_distance := 3
  let dog_walks_per_day := 2
  let days_in_week := 7
  let friend_distance := 5
  let friend_visits := 1
  let store_distance := 4
  let store_visits := 2
  let hike_distance := 10

  let work_miles := work_distance * 2 * work_days
  let dog_miles := dog_walk_distance * dog_walks_per_day * days_in_week
  let friend_miles := friend_distance * 2 * friend_visits
  let store_miles := store_distance * 2 * store_visits
  let hike_miles := hike_distance

  work_miles + dog_miles + friend_miles + store_miles + hike_miles

theorem benjamin_walks_158_miles : total_miles_walked = 158 := by
  sorry

end NUMINAMATH_CALUDE_benjamin_walks_158_miles_l448_44803


namespace NUMINAMATH_CALUDE_cos_angle_F₁PF₂_l448_44805

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the foci (we don't know their exact coordinates, so we leave them abstract)
variables (F₁ F₂ : ℝ × ℝ)

-- Define point P on the ellipse
variable (P : ℝ × ℝ)

-- State that P is on the ellipse
axiom P_on_ellipse : ellipse P.1 P.2

-- Define the distance between O and P
axiom OP_distance : Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = Real.sqrt 3

-- Theorem to prove
theorem cos_angle_F₁PF₂ : 
  ∃ (F₁ F₂ : ℝ × ℝ), 
    (F₁ ≠ F₂) ∧ 
    (∀ Q : ℝ × ℝ, ellipse Q.1 Q.2 → 
      Real.sqrt ((Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2) +
      Real.sqrt ((Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2) = 
      Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)) →
    ((P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2)) / 
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
     Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_cos_angle_F₁PF₂_l448_44805


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l448_44836

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 50) (h2 : x * y = 25) :
  1 / x + 1 / y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l448_44836


namespace NUMINAMATH_CALUDE_inequality_preservation_l448_44871

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l448_44871


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l448_44868

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 2) :
  (1/x + 1/y + 1/z) ≥ 4.5 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 ∧ 1/x + 1/y + 1/z = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l448_44868


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l448_44898

/-- Represents a quadratic equation ax^2 + 4x + c = 0 with exactly one solution -/
structure UniqueQuadratic where
  a : ℝ
  c : ℝ
  has_unique_solution : (4^2 - 4*a*c) = 0
  sum_constraint : a + c = 5
  order_constraint : a < c

theorem unique_quadratic_solution (q : UniqueQuadratic) : (q.a, q.c) = (1, 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l448_44898


namespace NUMINAMATH_CALUDE_rosy_fish_count_l448_44814

def lilly_fish : ℕ := 10
def total_fish : ℕ := 24

theorem rosy_fish_count : ∃ (rosy_fish : ℕ), rosy_fish = total_fish - lilly_fish ∧ rosy_fish = 14 := by
  sorry

end NUMINAMATH_CALUDE_rosy_fish_count_l448_44814


namespace NUMINAMATH_CALUDE_cubic_equation_root_l448_44885

theorem cubic_equation_root (h : ℝ) : 
  (2 : ℝ)^3 + h * 2 + 10 = 0 → h = -9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l448_44885


namespace NUMINAMATH_CALUDE_sqrt_expression_l448_44843

theorem sqrt_expression : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_l448_44843


namespace NUMINAMATH_CALUDE_museum_pictures_l448_44824

theorem museum_pictures (zoo_pics : ℕ) (deleted_pics : ℕ) (remaining_pics : ℕ) :
  zoo_pics = 41 →
  deleted_pics = 15 →
  remaining_pics = 55 →
  ∃ museum_pics : ℕ, zoo_pics + museum_pics = remaining_pics + deleted_pics ∧ museum_pics = 29 :=
by sorry

end NUMINAMATH_CALUDE_museum_pictures_l448_44824


namespace NUMINAMATH_CALUDE_candice_arrival_time_l448_44869

/-- Represents the driving scenario of Candice --/
structure DrivingScenario where
  initial_speed : ℕ
  final_speed : ℕ
  total_distance : ℚ
  drive_time : ℕ

/-- The conditions of Candice's drive --/
def candice_drive : DrivingScenario :=
  { initial_speed := 10,
    final_speed := 6,
    total_distance := 2/3,
    drive_time := 5 }

/-- Theorem stating that Candice arrives home at 5:05 PM --/
theorem candice_arrival_time (d : DrivingScenario) 
  (h1 : d.initial_speed > d.final_speed)
  (h2 : d.drive_time > 0)
  (h3 : d.total_distance = (d.initial_speed + d.final_speed + 1) * (d.initial_speed - d.final_speed) / 120) :
  d.drive_time = 5 ∧ d = candice_drive :=
sorry

end NUMINAMATH_CALUDE_candice_arrival_time_l448_44869


namespace NUMINAMATH_CALUDE_product_equals_zero_l448_44896

theorem product_equals_zero : (3 * 5 * 7 + 4 * 6 * 8) * (2 * 12 * 5 - 20 * 3 * 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l448_44896


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l448_44810

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

theorem geometric_sequence_sum (a₁ r : ℝ) (h₁ : a₁ = 1) (h₂ : r = -2) :
  let a := geometric_sequence a₁ r
  (a 1) + |a 2| + (a 3) + |a 4| = 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l448_44810


namespace NUMINAMATH_CALUDE_problem_statement_l448_44808

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (abs (x + 2*y) + abs (x - y) ≤ 5/2 ↔ 1/6 ≤ x ∧ x < 1) ∧
  ((1/x^2 - 1) * (1/y^2 - 1) ≥ 9) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l448_44808


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l448_44825

theorem complex_product_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*Complex.I
  let z₂ : ℂ := 2 + Complex.I
  (z₁ * z₂).re = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l448_44825


namespace NUMINAMATH_CALUDE_tom_gave_two_seashells_l448_44840

/-- The number of seashells Tom gave to Jessica -/
def seashells_given (initial_seashells current_seashells : ℕ) : ℕ :=
  initial_seashells - current_seashells

/-- Theorem stating that Tom gave 2 seashells to Jessica -/
theorem tom_gave_two_seashells :
  seashells_given 5 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tom_gave_two_seashells_l448_44840


namespace NUMINAMATH_CALUDE_triangle_median_sum_bounds_l448_44876

/-- For any triangle, the sum of its medians is greater than 3/4 of its perimeter
    but less than its perimeter. -/
theorem triangle_median_sum_bounds (a b c m_a m_b m_c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : m_a^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_median_b : m_b^2 = (2*c^2 + 2*a^2 - b^2) / 4)
  (h_median_c : m_c^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  3/4 * (a + b + c) < m_a + m_b + m_c ∧ m_a + m_b + m_c < a + b + c := by
sorry

end NUMINAMATH_CALUDE_triangle_median_sum_bounds_l448_44876


namespace NUMINAMATH_CALUDE_pawsitive_training_center_dogs_l448_44850

/-- Represents the number of dogs that can perform a specific combination of tricks -/
structure DogTricks where
  sit : ℕ
  stay : ℕ
  fetch : ℕ
  roll_over : ℕ
  sit_stay : ℕ
  sit_fetch : ℕ
  sit_roll : ℕ
  stay_fetch : ℕ
  stay_roll : ℕ
  fetch_roll : ℕ
  sit_stay_fetch : ℕ
  sit_stay_roll : ℕ
  sit_fetch_roll : ℕ
  stay_fetch_roll : ℕ
  all_four : ℕ
  none : ℕ

/-- Calculates the total number of dogs at the Pawsitive Training Center -/
def total_dogs (d : DogTricks) : ℕ := sorry

/-- Theorem stating that given the conditions, the total number of dogs is 135 -/
theorem pawsitive_training_center_dogs :
  let d : DogTricks := {
    sit := 60, stay := 35, fetch := 45, roll_over := 40,
    sit_stay := 20, sit_fetch := 15, sit_roll := 10,
    stay_fetch := 5, stay_roll := 8, fetch_roll := 6,
    sit_stay_fetch := 4, sit_stay_roll := 3,
    sit_fetch_roll := 2, stay_fetch_roll := 1,
    all_four := 2, none := 12
  }
  total_dogs d = 135 := by sorry

end NUMINAMATH_CALUDE_pawsitive_training_center_dogs_l448_44850


namespace NUMINAMATH_CALUDE_rectangles_not_necessarily_similar_l448_44863

-- Define a rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  length_positive : length > 0
  width_positive : width > 0

-- Define similarity for rectangles
def are_similar (r1 r2 : Rectangle) : Prop :=
  r1.length / r1.width = r2.length / r2.width

-- Theorem stating that rectangles are not necessarily similar
theorem rectangles_not_necessarily_similar :
  ∃ (r1 r2 : Rectangle), ¬(are_similar r1 r2) :=
sorry

end NUMINAMATH_CALUDE_rectangles_not_necessarily_similar_l448_44863


namespace NUMINAMATH_CALUDE_triangle_properties_l448_44826

/-- Triangle ABC with side lengths a, b, c corresponding to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 3 * Real.sin t.B - Real.cos t.B = 2 * Real.sin (t.B - π / 6))
  (h2 : t.b = 1)
  (h3 : t.A = 5 * π / 12) :
  (t.c = Real.sqrt 6 / 3) ∧ 
  (∀ h : ℝ, h ≤ Real.sqrt 3 / 2 → 
    ∃ (a c : ℝ), 
      a > 0 ∧ c > 0 ∧ 
      a * c ≤ 1 ∧ 
      h = Real.sqrt 3 / 2 * a * c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l448_44826


namespace NUMINAMATH_CALUDE_range_of_a_l448_44809

theorem range_of_a (a : ℝ) 
  (p : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x)
  (q : ∃ x₀ : ℝ, x₀^2 + 4*x₀ + a = 0) :
  e ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l448_44809


namespace NUMINAMATH_CALUDE_michael_needs_additional_money_l448_44800

def michael_money : ℝ := 50
def cake_cost : ℝ := 20
def bouquet_cost : ℝ := 36
def balloons_cost : ℝ := 5
def perfume_cost_gbp : ℝ := 30
def photo_album_cost_eur : ℝ := 25
def gbp_to_usd : ℝ := 1.4
def eur_to_usd : ℝ := 1.2

theorem michael_needs_additional_money :
  let perfume_cost_usd := perfume_cost_gbp * gbp_to_usd
  let photo_album_cost_usd := photo_album_cost_eur * eur_to_usd
  let total_cost := cake_cost + bouquet_cost + balloons_cost + perfume_cost_usd + photo_album_cost_usd
  total_cost - michael_money = 83 := by sorry

end NUMINAMATH_CALUDE_michael_needs_additional_money_l448_44800


namespace NUMINAMATH_CALUDE_chloe_min_nickels_l448_44838

/-- The minimum number of nickels Chloe needs to afford the book -/
def min_nickels : ℕ := 120

/-- The cost of the book in cents -/
def book_cost : ℕ := 4850

/-- The value of Chloe's $10 bills in cents -/
def ten_dollar_bills : ℕ := 4000

/-- The value of Chloe's quarters in cents -/
def quarters : ℕ := 250

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

theorem chloe_min_nickels :
  ∀ n : ℕ, n ≥ min_nickels →
  ten_dollar_bills + quarters + n * nickel_value ≥ book_cost :=
by sorry

end NUMINAMATH_CALUDE_chloe_min_nickels_l448_44838


namespace NUMINAMATH_CALUDE_wire_service_reporters_l448_44849

theorem wire_service_reporters (x y both_local non_local_politics international_only : ℝ) 
  (hx : x = 35)
  (hy : y = 25)
  (hboth : both_local = 20)
  (hnon_local : non_local_politics = 30)
  (hinter : international_only = 15) :
  100 - ((x + y - both_local) + non_local_politics + international_only) = 75 := by
  sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l448_44849


namespace NUMINAMATH_CALUDE_four_integers_sum_l448_44801

theorem four_integers_sum (a b c d : ℤ) :
  a + b + c = 6 ∧
  a + b + d = 7 ∧
  a + c + d = 8 ∧
  b + c + d = 9 →
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := by
sorry

end NUMINAMATH_CALUDE_four_integers_sum_l448_44801


namespace NUMINAMATH_CALUDE_trig_identity_l448_44888

theorem trig_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l448_44888


namespace NUMINAMATH_CALUDE_solution_set_part_I_value_of_a_part_II_l448_44841

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 1

-- Part I
theorem solution_set_part_I (a : ℝ) (h : a > 1) :
  let f := f a
  a = 2 →
  {x : ℝ | f x ≥ 4 - |x - 4|} = {x : ℝ | x ≥ 11/2 ∨ x ≤ 1/2} :=
sorry

-- Part II
theorem value_of_a_part_II (a : ℝ) (h : a > 1) :
  let f := f a
  ({x : ℝ | |f (2*x + a) - 2*f x| ≤ 1} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 1}) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_I_value_of_a_part_II_l448_44841


namespace NUMINAMATH_CALUDE_inequality_upper_bound_upper_bound_tight_l448_44822

theorem inequality_upper_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) ≤ 2 + Real.sqrt 5 := by
  sorry

theorem upper_bound_tight : 
  ∀ ε > 0, ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ 
  (2 + Real.sqrt 5) - (Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1)) < ε := by
  sorry

end NUMINAMATH_CALUDE_inequality_upper_bound_upper_bound_tight_l448_44822


namespace NUMINAMATH_CALUDE_abc_is_50_l448_44866

def repeating_decimal (a b c : ℕ) : ℚ :=
  1 + (100 * a + 10 * b + c : ℚ) / 999

theorem abc_is_50 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  12 * (repeating_decimal a b c - (1 + (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000)) = 0.6 →
  100 * a + 10 * b + c = 50 := by
sorry

end NUMINAMATH_CALUDE_abc_is_50_l448_44866


namespace NUMINAMATH_CALUDE_lcm_of_ratio_3_4_l448_44883

/-- Given two natural numbers with a ratio of 3:4, where one number is 45 and the other is 60, 
    their least common multiple (LCM) is 180. -/
theorem lcm_of_ratio_3_4 (a b : ℕ) (h_ratio : 3 * b = 4 * a) (h_a : a = 45) (h_b : b = 60) : 
  Nat.lcm a b = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_3_4_l448_44883


namespace NUMINAMATH_CALUDE_sushi_cost_l448_44882

theorem sushi_cost (j e : ℝ) (h1 : j + e = 200) (h2 : e = 9 * j) : e = 180 := by
  sorry

end NUMINAMATH_CALUDE_sushi_cost_l448_44882


namespace NUMINAMATH_CALUDE_box_surface_area_is_744_l448_44807

/-- The surface area of an open box formed by removing square corners from a rectangular sheet --/
def boxSurfaceArea (length width cornerSize : ℕ) : ℕ :=
  length * width - 4 * (cornerSize * cornerSize)

/-- Theorem stating that the surface area of the specified box is 744 square units --/
theorem box_surface_area_is_744 :
  boxSurfaceArea 40 25 8 = 744 := by
  sorry

end NUMINAMATH_CALUDE_box_surface_area_is_744_l448_44807


namespace NUMINAMATH_CALUDE_fixed_salary_is_1000_l448_44835

/-- Represents the earnings structure and goal of a sales executive -/
structure SalesExecutive where
  commissionRate : Float
  targetEarnings : Float
  targetSales : Float

/-- Calculates the fixed salary for a sales executive -/
def calculateFixedSalary (exec : SalesExecutive) : Float :=
  exec.targetEarnings - exec.commissionRate * exec.targetSales

/-- Theorem: The fixed salary for the given sales executive is $1000 -/
theorem fixed_salary_is_1000 :
  let exec : SalesExecutive := {
    commissionRate := 0.05,
    targetEarnings := 5000,
    targetSales := 80000
  }
  calculateFixedSalary exec = 1000 := by
  sorry

#eval calculateFixedSalary {
  commissionRate := 0.05,
  targetEarnings := 5000,
  targetSales := 80000
}

end NUMINAMATH_CALUDE_fixed_salary_is_1000_l448_44835


namespace NUMINAMATH_CALUDE_yearly_subscription_cost_proof_l448_44830

/-- The yearly subscription cost to professional magazines, given that a 50% reduction
    in the budget results in spending $470 less. -/
def yearly_subscription_cost : ℝ := 940

theorem yearly_subscription_cost_proof :
  yearly_subscription_cost - yearly_subscription_cost / 2 = 470 := by
  sorry

end NUMINAMATH_CALUDE_yearly_subscription_cost_proof_l448_44830


namespace NUMINAMATH_CALUDE_sharp_constant_is_20_l448_44854

/-- The function # defined for any real number -/
def sharp (C : ℝ) (p : ℝ) : ℝ := 2 * p - C

/-- Theorem stating that the constant in the sharp function is 20 -/
theorem sharp_constant_is_20 : ∃ C : ℝ, 
  (sharp C (sharp C (sharp C 18.25)) = 6) ∧ C = 20 := by
  sorry

end NUMINAMATH_CALUDE_sharp_constant_is_20_l448_44854


namespace NUMINAMATH_CALUDE_plot_width_calculation_l448_44877

/-- Calculates the width of a rectangular plot given its length and fence specifications. -/
theorem plot_width_calculation (length width : ℝ) (num_poles : ℕ) (pole_distance : ℝ) : 
  length = 90 ∧ 
  num_poles = 28 ∧ 
  pole_distance = 10 ∧ 
  (num_poles - 1) * pole_distance = 2 * (length + width) →
  width = 45 := by
sorry

end NUMINAMATH_CALUDE_plot_width_calculation_l448_44877


namespace NUMINAMATH_CALUDE_germination_probability_convergence_l448_44891

/-- Represents the experimental data for rice seed germination --/
structure GerminationData where
  n : ℕ  -- number of grains per batch
  m : ℕ  -- number of germinations
  h : m ≤ n

/-- The list of experimental data --/
def experimentalData : List GerminationData := [
  ⟨50, 47, sorry⟩,
  ⟨100, 89, sorry⟩,
  ⟨200, 188, sorry⟩,
  ⟨500, 461, sorry⟩,
  ⟨1000, 892, sorry⟩,
  ⟨2000, 1826, sorry⟩,
  ⟨3000, 2733, sorry⟩
]

/-- The germination frequency for a given experiment --/
def germinationFrequency (data : GerminationData) : ℚ :=
  data.m / data.n

/-- The estimated probability of germination --/
def estimatedProbability : ℚ := 91 / 100

/-- Theorem stating that the germination frequency approaches the estimated probability as sample size increases --/
theorem germination_probability_convergence :
  ∀ ε > 0, ∃ N, ∀ data ∈ experimentalData, data.n ≥ N →
    |germinationFrequency data - estimatedProbability| < ε :=
sorry

end NUMINAMATH_CALUDE_germination_probability_convergence_l448_44891


namespace NUMINAMATH_CALUDE_simplify_quadratic_expression_l448_44862

theorem simplify_quadratic_expression (a : ℝ) : -2 * a^2 + 4 * a^2 = 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_quadratic_expression_l448_44862


namespace NUMINAMATH_CALUDE_towel_area_decrease_l448_44821

theorem towel_area_decrease (L B : ℝ) (h_positive : L > 0 ∧ B > 0) : 
  let original_area := L * B
  let new_length := 0.8 * L
  let new_breadth := 0.8 * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.36 := by
sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l448_44821


namespace NUMINAMATH_CALUDE_M_properties_l448_44853

def M (n : ℕ) : ℤ := (-2) ^ n

theorem M_properties :
  (M 5 + M 6 = 32) ∧
  (2 * M 2015 + M 2016 = 0) ∧
  (∀ n : ℕ, 2 * M n + M (n + 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_M_properties_l448_44853


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l448_44865

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + m = 0) → m ≤ 9/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l448_44865


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l448_44831

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (a : Line) (M N : Plane)
  (h1 : perpendicular a M)
  (h2 : parallel a N) :
  perp_planes M N :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l448_44831


namespace NUMINAMATH_CALUDE_ab_gt_1_sufficient_not_necessary_for_a_plus_b_gt_2_l448_44890

theorem ab_gt_1_sufficient_not_necessary_for_a_plus_b_gt_2 :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y > 1 → x + y > 2) ∧
  (∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ c + d > 2 ∧ c * d ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ab_gt_1_sufficient_not_necessary_for_a_plus_b_gt_2_l448_44890


namespace NUMINAMATH_CALUDE_onion_saute_time_l448_44864

def calzone_problem (onion_time : ℝ) : Prop :=
  let garlic_pepper_time := (1/4) * onion_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1/10) * (knead_time + rest_time)
  onion_time + garlic_pepper_time + knead_time + rest_time + assemble_time = 124

theorem onion_saute_time :
  ∃ (t : ℝ), calzone_problem t ∧ t = 20 := by
  sorry

end NUMINAMATH_CALUDE_onion_saute_time_l448_44864


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a4_l448_44845

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_a4 (seq : ArithmeticSequence) 
    (h1 : seq.S 6 = 24) (h2 : seq.S 9 = 63) : seq.a 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a4_l448_44845


namespace NUMINAMATH_CALUDE_binary_1011_equals_11_l448_44819

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1011_equals_11 :
  binary_to_decimal [true, true, false, true] = 11 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011_equals_11_l448_44819


namespace NUMINAMATH_CALUDE_division_problem_l448_44887

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 190 →
  divisor = 21 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l448_44887


namespace NUMINAMATH_CALUDE_compute_expression_l448_44820

theorem compute_expression : 
  18 * (140 / 2 + 30 / 4 + 12 / 20 + 2 / 3) = 1417.8 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l448_44820


namespace NUMINAMATH_CALUDE_triangle_count_theorem_l448_44802

/-- The number of trees planted in a triangular shape -/
def num_trees : ℕ := 21

/-- The number of ways to choose 3 trees from num_trees -/
def total_choices : ℕ := Nat.choose num_trees 3

/-- The number of ways to choose 3 collinear trees -/
def collinear_choices : ℕ := 114

/-- The number of ways to choose 3 trees to form a non-degenerate triangle -/
def non_degenerate_triangles : ℕ := total_choices - collinear_choices

theorem triangle_count_theorem : non_degenerate_triangles = 1216 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_theorem_l448_44802


namespace NUMINAMATH_CALUDE_small_pond_green_percentage_l448_44893

def total_ducks : ℕ := 100
def small_pond_ducks : ℕ := 20
def large_pond_ducks : ℕ := 80
def large_pond_green_percentage : ℚ := 15 / 100
def total_green_percentage : ℚ := 16 / 100

theorem small_pond_green_percentage :
  ∃ x : ℚ,
    x * small_pond_ducks + large_pond_green_percentage * large_pond_ducks =
    total_green_percentage * total_ducks ∧
    x = 20 / 100 := by
  sorry

end NUMINAMATH_CALUDE_small_pond_green_percentage_l448_44893


namespace NUMINAMATH_CALUDE_no_ten_consecutive_power_of_two_values_l448_44879

theorem no_ten_consecutive_power_of_two_values (a b : ℝ) : 
  ¬ ∃ (k : ℤ → ℕ) (x₀ : ℤ), ∀ x : ℤ, x₀ ≤ x ∧ x < x₀ + 10 → 
    x^2 + a*x + b = 2^(k x) :=
sorry

end NUMINAMATH_CALUDE_no_ten_consecutive_power_of_two_values_l448_44879


namespace NUMINAMATH_CALUDE_inequality_system_solution_l448_44856

theorem inequality_system_solution (x : ℝ) :
  (x - 1) * Real.log 2 + Real.log (2^(x + 1) + 1) < Real.log (7 * 2^x + 12) →
  Real.log (x + 2) / Real.log x > 2 →
  1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l448_44856


namespace NUMINAMATH_CALUDE_stating_wrapping_paper_area_theorem_l448_44851

/-- Represents a rectangular box. -/
structure Box where
  a : ℝ
  b : ℝ
  h : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  h_pos : 0 < h

/-- Calculates the area of the square wrapping paper needed for a given box. -/
def wrappingPaperArea (box : Box) : ℝ :=
  (box.a + 2 * box.h) ^ 2

/-- 
Theorem stating that the area of the square wrapping paper for a rectangular box
with base dimensions a × b and height h, wrapped as described in the problem,
is (a + 2h)².
-/
theorem wrapping_paper_area_theorem (box : Box) :
  wrappingPaperArea box = (box.a + 2 * box.h) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_stating_wrapping_paper_area_theorem_l448_44851


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_3_pow_5_l448_44875

def units_digit_pattern : ℕ → ℕ
| 0 => 7
| 1 => 9
| 2 => 3
| 3 => 1
| n + 4 => units_digit_pattern n

def power_mod (base exponent modulus : ℕ) : ℕ :=
  (base ^ exponent) % modulus

theorem units_digit_of_7_pow_3_pow_5 :
  units_digit_pattern (power_mod 3 5 4) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_3_pow_5_l448_44875


namespace NUMINAMATH_CALUDE_probability_two_red_correct_l448_44874

def bag_red_balls : ℕ := 9
def bag_white_balls : ℕ := 3
def total_balls : ℕ := bag_red_balls + bag_white_balls
def drawn_balls : ℕ := 4

def probability_two_red : ℚ :=
  (Nat.choose bag_red_balls 2 * Nat.choose bag_white_balls 2) / Nat.choose total_balls drawn_balls

theorem probability_two_red_correct :
  probability_two_red = (Nat.choose bag_red_balls 2 * Nat.choose bag_white_balls 2) / Nat.choose total_balls drawn_balls :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_correct_l448_44874


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l448_44816

/-- The repeating decimal 0.3535... expressed as a real number -/
def repeating_decimal : ℚ := 35 / 99

theorem repeating_decimal_equals_fraction : repeating_decimal = 35 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l448_44816


namespace NUMINAMATH_CALUDE_pythagorean_triple_parity_l448_44880

theorem pythagorean_triple_parity (m n : ℤ) 
  (h_consecutive : m = n + 1 ∨ n = m + 1)
  (a b c : ℤ)
  (h_a : a = m^2 - n^2)
  (h_b : b = 2*m*n)
  (h_c : c = m^2 + n^2)
  (h_coprime : ¬(2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c)) : 
  Odd c ∧ Even b ∧ Odd a := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_parity_l448_44880


namespace NUMINAMATH_CALUDE_geometric_sequence_decreasing_l448_44837

/-- A geometric sequence with first term a₁ and common ratio q. -/
def GeometricSequence (a₁ q : ℝ) : ℕ → ℝ := fun n ↦ a₁ * q ^ (n - 1)

/-- A sequence is decreasing if each term is less than the previous term. -/
def IsDecreasing (s : ℕ → ℝ) : Prop := ∀ n : ℕ, s (n + 1) < s n

theorem geometric_sequence_decreasing (a₁ q : ℝ) (h1 : a₁ * (q - 1) < 0) (h2 : q > 0) :
  IsDecreasing (GeometricSequence a₁ q) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_decreasing_l448_44837
