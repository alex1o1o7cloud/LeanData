import Mathlib

namespace bruce_total_payment_l2596_259608

def grape_quantity : ℝ := 8
def grape_rate : ℝ := 70
def grape_discount : ℝ := 0.1
def mango_quantity : ℝ := 10
def mango_rate : ℝ := 55

theorem bruce_total_payment :
  let grape_cost := grape_quantity * grape_rate
  let grape_discount_amount := grape_cost * grape_discount
  let final_grape_cost := grape_cost - grape_discount_amount
  let mango_cost := mango_quantity * mango_rate
  final_grape_cost + mango_cost = 1054 := by sorry

end bruce_total_payment_l2596_259608


namespace grocer_banana_profit_l2596_259618

/-- Represents the profit calculation for a grocer selling bananas -/
theorem grocer_banana_profit : 
  let purchase_rate : ℚ := 3 / 0.5  -- 3 pounds per $0.50
  let sell_rate : ℚ := 4 / 1        -- 4 pounds per $1.00
  let total_pounds : ℚ := 108       -- Total pounds purchased
  let cost_price := total_pounds / purchase_rate
  let sell_price := total_pounds / sell_rate
  let profit := sell_price - cost_price
  profit = 9 := by sorry

end grocer_banana_profit_l2596_259618


namespace increasing_function_property_l2596_259651

-- Define an increasing function on the real line
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- State the theorem
theorem increasing_function_property (f : ℝ → ℝ) (h : IncreasingFunction f) :
  (∀ a b : ℝ, a + b ≥ 0 → f a + f b ≥ f (-a) + f (-b)) ∧
  (∀ a b : ℝ, f a + f b ≥ f (-a) + f (-b) → a + b ≥ 0) := by
  sorry

end increasing_function_property_l2596_259651


namespace problem_solution_l2596_259663

theorem problem_solution (m n x y : ℝ) 
  (h1 : m - n = 8) 
  (h2 : x + y = 1) : 
  (n + x) - (m - y) = -7 := by
sorry

end problem_solution_l2596_259663


namespace equilateral_triangle_dot_product_l2596_259659

/-- Equilateral triangle ABC with side length 2 -/
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ‖A - B‖ = 2 ∧ ‖B - C‖ = 2 ∧ ‖C - A‖ = 2

/-- Vector from point P to point Q -/
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := Q - P

theorem equilateral_triangle_dot_product 
  (A B C D E : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_BC : vec B C = 2 • vec B D)
  (h_CA : vec C A = 3 • vec C E)
  (a b : ℝ × ℝ)
  (h_a : a = vec A B)
  (h_b : b = vec A C)
  (h_norm_a : ‖a‖ = 2)
  (h_norm_b : ‖b‖ = 2)
  (h_dot : a • b = 2) :
  (1/2 • (a + b)) • (2/3 • b - a) = -1 :=
sorry

end equilateral_triangle_dot_product_l2596_259659


namespace horner_method_proof_l2596_259630

def f (x : ℝ) : ℝ := 3*x^5 - 4*x^4 + 6*x^3 - 2*x^2 - 5*x - 2

theorem horner_method_proof : f 5 = 7548 := by
  sorry

end horner_method_proof_l2596_259630


namespace problem_statement_l2596_259650

theorem problem_statement (a b c d : ℝ) : 
  (a * b > 0 ∧ b * c - a * d > 0 → c / a - d / b > 0) ∧
  (a * b > 0 ∧ c / a - d / b > 0 → b * c - a * d > 0) :=
sorry

end problem_statement_l2596_259650


namespace tangent_point_coordinates_l2596_259656

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the condition for the tangent line to be parallel to y = 4x - 1
def tangent_parallel (x : ℝ) : Prop := f' x = 4

-- Define the point P₀
structure Point_P₀ where
  x : ℝ
  y : ℝ
  on_curve : f x = y
  tangent_parallel : tangent_parallel x

-- State the theorem
theorem tangent_point_coordinates :
  ∀ p : Point_P₀, (p.x = 1 ∧ p.y = 0) ∨ (p.x = -1 ∧ p.y = -4) :=
sorry

end tangent_point_coordinates_l2596_259656


namespace base_approximation_l2596_259684

/-- The base value we're looking for -/
def base : ℝ := 21.5

/-- The function representing the left side of the inequality -/
def f (b : ℝ) : ℝ := 2.134 * b^3

theorem base_approximation :
  ∀ b : ℝ, f b < 21000 → b ≤ base :=
sorry

end base_approximation_l2596_259684


namespace only_cylinder_quadrilateral_l2596_259603

-- Define the types of geometric solids
inductive GeometricSolid
  | Cone
  | Sphere
  | Cylinder

-- Define the possible shapes of plane sections
inductive PlaneSection
  | Circle
  | Ellipse
  | Parabola
  | Triangle
  | Quadrilateral

-- Function to determine possible plane sections for each solid
def possibleSections (solid : GeometricSolid) : Set PlaneSection :=
  match solid with
  | GeometricSolid.Cone => {PlaneSection.Circle, PlaneSection.Ellipse, PlaneSection.Parabola, PlaneSection.Triangle}
  | GeometricSolid.Sphere => {PlaneSection.Circle}
  | GeometricSolid.Cylinder => {PlaneSection.Circle, PlaneSection.Ellipse, PlaneSection.Quadrilateral}

-- Theorem stating that only a cylinder can produce a quadrilateral section
theorem only_cylinder_quadrilateral :
  ∀ (solid : GeometricSolid),
    PlaneSection.Quadrilateral ∈ possibleSections solid ↔ solid = GeometricSolid.Cylinder :=
by sorry

end only_cylinder_quadrilateral_l2596_259603


namespace gcd_1994_powers_and_product_l2596_259631

theorem gcd_1994_powers_and_product : 
  Nat.gcd (1994^1994 + 1994^1995) (1994 * 1995) = 1994 * 1995 := by
  sorry

end gcd_1994_powers_and_product_l2596_259631


namespace cistern_width_l2596_259698

/-- Calculates the width of a rectangular cistern given its length, depth, and total wet surface area. -/
theorem cistern_width (length depth area : ℝ) (h1 : length = 5) (h2 : depth = 1.25) (h3 : area = 42.5) :
  ∃ width : ℝ, width = 4 ∧ 
  area = length * width + 2 * (depth * length) + 2 * (depth * width) :=
by sorry

end cistern_width_l2596_259698


namespace prime_solution_equation_l2596_259658

theorem prime_solution_equation : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 ∧ p = 17 ∧ q = 3 := by
  sorry

end prime_solution_equation_l2596_259658


namespace max_value_quadratic_inequality_l2596_259612

/-- Given that the solution set of ax^2 + 2x + c ≤ 0 is {x | x = -1/a} and a > c,
    prove that the maximum value of (a-c)/(a^2+c^2) is √2/4 -/
theorem max_value_quadratic_inequality (a c : ℝ) 
    (h1 : ∀ x, a * x^2 + 2 * x + c ≤ 0 ↔ x = -1/a)
    (h2 : a > c) :
    (∀ a' c', a' > c' → (a' - c') / (a'^2 + c'^2) ≤ Real.sqrt 2 / 4) ∧ 
    (∃ a' c', a' > c' ∧ (a' - c') / (a'^2 + c'^2) = Real.sqrt 2 / 4) :=
by sorry

end max_value_quadratic_inequality_l2596_259612


namespace quadratic_roots_relation_l2596_259675

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∀ x, x^2 + m*x + n = 0 ↔ ∃ y, y^2 + p*y + m = 0 ∧ x = 3*y) →
  n / p = 27 :=
by sorry

end quadratic_roots_relation_l2596_259675


namespace car_fuel_efficiency_improvement_l2596_259610

/-- Proves the increase in travel distance after modifying a car's fuel efficiency -/
theorem car_fuel_efficiency_improvement (initial_mpg : ℝ) (tank_capacity : ℝ) 
  (fuel_reduction_factor : ℝ) (h1 : initial_mpg = 28) (h2 : tank_capacity = 15) 
  (h3 : fuel_reduction_factor = 0.8) : 
  (initial_mpg / fuel_reduction_factor - initial_mpg) * tank_capacity = 84 := by
  sorry

end car_fuel_efficiency_improvement_l2596_259610


namespace m_range_l2596_259644

def f (x : ℝ) := x^3 + x

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m < 1 :=
sorry

end m_range_l2596_259644


namespace cylinder_volume_ratio_l2596_259692

/-- 
Given two cylinders with the same height and radii in the ratio 1:3,
if the volume of the larger cylinder is 360 cc, then the volume of the smaller cylinder is 40 cc.
-/
theorem cylinder_volume_ratio (h : ℝ) (r : ℝ) : 
  h > 0 → r > 0 → π * (3 * r)^2 * h = 360 → π * r^2 * h = 40 := by
  sorry

end cylinder_volume_ratio_l2596_259692


namespace slope_intercept_sum_l2596_259690

/-- Given points A, B, C, and D (midpoint of AB), prove that the sum of 
    the slope and y-intercept of the line passing through C and D is 27/5 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 2) →
  C = (10, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := D.2
  m + b = 27 / 5 := by
  sorry

end slope_intercept_sum_l2596_259690


namespace tan_fifteen_ratio_equals_sqrt_three_l2596_259634

theorem tan_fifteen_ratio_equals_sqrt_three :
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end tan_fifteen_ratio_equals_sqrt_three_l2596_259634


namespace sum_of_powers_of_i_l2596_259619

theorem sum_of_powers_of_i (i : ℂ) (h : i^2 = -1) :
  (Finset.range 2017).sum (fun k => i^(k + 1)) = i := by
  sorry

end sum_of_powers_of_i_l2596_259619


namespace miss_world_contest_l2596_259607

-- Define the total number of girls
def total_girls : ℕ := 48

-- Define the number of girls with blue eyes and white skin
def blue_eyes_white_skin : ℕ := 12

-- Define the number of girls with light black skin
def light_black_skin : ℕ := 28

-- Define the number of girls with brown eyes
def brown_eyes : ℕ := 15

-- Define a as the number of girls with brown eyes and light black skin
def a : ℕ := sorry

-- Define b as the number of girls with white skin and brown eyes
def b : ℕ := sorry

-- Theorem to prove
theorem miss_world_contest :
  a = 7 ∧ b = 8 := by sorry

end miss_world_contest_l2596_259607


namespace M_minus_N_equals_closed_open_l2596_259623

-- Definition of set difference
def set_difference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Definition of set M
def M : Set ℝ := {x | |x + 1| ≤ 2}

-- Definition of set N
def N : Set ℝ := {x | ∃ α : ℝ, x = |Real.sin α|}

-- Theorem statement
theorem M_minus_N_equals_closed_open :
  set_difference M N = Set.Ico (-3) 0 := by sorry

end M_minus_N_equals_closed_open_l2596_259623


namespace proper_subset_of_singleton_l2596_259668

-- Define the set P
def P : Set ℕ := {0}

-- State the theorem
theorem proper_subset_of_singleton :
  ∀ (S : Set ℕ), S ⊂ P → S = ∅ :=
sorry

end proper_subset_of_singleton_l2596_259668


namespace parallel_planes_from_perpendicular_lines_l2596_259652

/-- Two lines are parallel -/
def parallel_lines (a b : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are parallel -/
def parallel_planes (p q : Plane) : Prop := sorry

/-- A line is distinct from another line -/
def distinct_lines (a b : Line) : Prop := sorry

/-- A plane is distinct from another plane -/
def distinct_planes (p q : Plane) : Prop := sorry

theorem parallel_planes_from_perpendicular_lines 
  (a b : Line) (α β : Plane) 
  (h1 : distinct_lines a b) 
  (h2 : distinct_planes α β) 
  (h3 : perpendicular_line_plane a α) 
  (h4 : perpendicular_line_plane b β) 
  (h5 : parallel_lines a b) : 
  parallel_planes α β := 
sorry

end parallel_planes_from_perpendicular_lines_l2596_259652


namespace tan_alpha_eq_neg_five_l2596_259642

theorem tan_alpha_eq_neg_five (α : ℝ) :
  (Real.cos (π / 2 - α) - 3 * Real.cos α) / (Real.sin α - Real.cos (π + α)) = 2 →
  Real.tan α = -5 := by
sorry

end tan_alpha_eq_neg_five_l2596_259642


namespace factorial_sum_equality_l2596_259671

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 2 * Nat.factorial 5 = 36120 := by
  sorry

end factorial_sum_equality_l2596_259671


namespace routes_on_3x2_grid_l2596_259633

/-- The number of routes on a grid from (0, m) to (n, 0) moving only right or down -/
def num_routes (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- The dimensions of the grid -/
def grid_width : ℕ := 3
def grid_height : ℕ := 2

theorem routes_on_3x2_grid :
  num_routes grid_height grid_width = Nat.choose (grid_height + grid_width) grid_height :=
by sorry

end routes_on_3x2_grid_l2596_259633


namespace solution_set_g_range_of_m_l2596_259626

-- Define the functions f and g
def f (x : ℝ) := x^2 - 2*x - 8
def g (x : ℝ) := 2*x^2 - 4*x - 16

-- Theorem for the solution set of g(x) < 0
theorem solution_set_g (x : ℝ) : g x < 0 ↔ -2 < x ∧ x < 4 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∀ x > 1, f x ≥ (m + 2)*x - m - 15) → m ≤ 4 := by sorry

end solution_set_g_range_of_m_l2596_259626


namespace distance_A_to_y_axis_l2596_259638

def point_A : ℝ × ℝ := (-2, 1)

def distance_to_y_axis (p : ℝ × ℝ) : ℝ := |p.1|

theorem distance_A_to_y_axis :
  distance_to_y_axis point_A = 2 := by sorry

end distance_A_to_y_axis_l2596_259638


namespace correct_statements_count_l2596_259662

/-- Represents a statement about sampling methods -/
inductive SamplingStatement
| SimpleRandomSmallPopulation
| SystematicSamplingMethod
| LotteryDrawingLots
| SystematicSamplingEqualProbability

/-- Checks if a given sampling statement is correct -/
def is_correct (s : SamplingStatement) : Bool :=
  match s with
  | SamplingStatement.SimpleRandomSmallPopulation => true
  | SamplingStatement.SystematicSamplingMethod => false
  | SamplingStatement.LotteryDrawingLots => true
  | SamplingStatement.SystematicSamplingEqualProbability => true

/-- The list of all sampling statements -/
def all_statements : List SamplingStatement :=
  [SamplingStatement.SimpleRandomSmallPopulation,
   SamplingStatement.SystematicSamplingMethod,
   SamplingStatement.LotteryDrawingLots,
   SamplingStatement.SystematicSamplingEqualProbability]

/-- Theorem stating that the number of correct sampling statements is 3 -/
theorem correct_statements_count :
  (all_statements.filter is_correct).length = 3 := by sorry

end correct_statements_count_l2596_259662


namespace ellies_calculation_l2596_259611

theorem ellies_calculation (x y z : ℝ) 
  (h1 : x - (y + z) = 18) 
  (h2 : x - y - z = 6) : 
  x - y = 12 := by
sorry

end ellies_calculation_l2596_259611


namespace rebate_calculation_l2596_259639

theorem rebate_calculation (polo_price necklace_price game_price : ℕ)
  (polo_count necklace_count : ℕ) (total_after_rebate : ℕ) :
  polo_price = 26 →
  necklace_price = 83 →
  game_price = 90 →
  polo_count = 3 →
  necklace_count = 2 →
  total_after_rebate = 322 →
  (polo_price * polo_count + necklace_price * necklace_count + game_price) - total_after_rebate = 12 := by
  sorry

end rebate_calculation_l2596_259639


namespace simplify_trig_expression_l2596_259636

theorem simplify_trig_expression (h : π / 2 < 2 ∧ 2 < π) :
  2 * Real.sqrt (1 + Real.sin 4) + Real.sqrt (2 + 2 * Real.cos 4) = 2 * Real.sin 2 := by
  sorry

end simplify_trig_expression_l2596_259636


namespace shaded_area_rectangle_minus_circles_l2596_259622

/-- The shaded area in a rectangle after subtracting two circles -/
theorem shaded_area_rectangle_minus_circles 
  (rectangle_length : ℝ) 
  (rectangle_width : ℝ)
  (circle1_radius : ℝ)
  (circle2_radius : ℝ)
  (h1 : rectangle_length = 16)
  (h2 : rectangle_width = 8)
  (h3 : circle1_radius = 4)
  (h4 : circle2_radius = 2) :
  rectangle_length * rectangle_width - π * (circle1_radius^2 + circle2_radius^2) = 128 - 20 * π := by
  sorry

end shaded_area_rectangle_minus_circles_l2596_259622


namespace four_students_in_all_activities_l2596_259657

/-- The number of students participating in all three activities in a summer camp. -/
def students_in_all_activities (total_students : ℕ) 
  (swimming_students : ℕ) (archery_students : ℕ) (chess_students : ℕ) 
  (at_least_two_activities : ℕ) : ℕ :=
  let a := swimming_students + archery_students + chess_students - at_least_two_activities - total_students
  a

/-- Theorem stating that 4 students participate in all three activities. -/
theorem four_students_in_all_activities : 
  students_in_all_activities 25 15 17 10 12 = 4 := by
  sorry

end four_students_in_all_activities_l2596_259657


namespace number_of_divisors_sum_of_divisors_l2596_259624

def n : ℕ := 120

-- Number of positive divisors
theorem number_of_divisors : (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 16 := by sorry

-- Sum of positive divisors
theorem sum_of_divisors : (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id = 3240 := by sorry

end number_of_divisors_sum_of_divisors_l2596_259624


namespace investment_interest_rate_proof_l2596_259637

theorem investment_interest_rate_proof 
  (total_investment : ℝ)
  (first_part : ℝ)
  (first_rate : ℝ)
  (total_interest : ℝ)
  (h1 : total_investment = 3500)
  (h2 : first_part = 1549.9999999999998)
  (h3 : first_rate = 3)
  (h4 : total_interest = 144)
  (h5 : first_part * (first_rate / 100) + (total_investment - first_part) * (second_rate / 100) = total_interest) :
  second_rate = 5 := by
  sorry


end investment_interest_rate_proof_l2596_259637


namespace processing_box_function_is_assignment_and_calculation_l2596_259602

/-- Represents the possible functions of a processing box in an algorithm -/
inductive ProcessingBoxFunction
  | startIndicator
  | inputIndicator
  | assignmentAndCalculation
  | conditionJudgment

/-- The function of a processing box -/
def processingBoxFunction : ProcessingBoxFunction :=
  ProcessingBoxFunction.assignmentAndCalculation

/-- Theorem stating that the function of a processing box is assignment and calculation -/
theorem processing_box_function_is_assignment_and_calculation :
  processingBoxFunction = ProcessingBoxFunction.assignmentAndCalculation :=
by sorry

end processing_box_function_is_assignment_and_calculation_l2596_259602


namespace both_vegan_and_kosher_l2596_259695

/-- Represents the meal delivery scenario -/
structure MealDelivery where
  total : ℕ
  vegan : ℕ
  kosher : ℕ
  neither : ℕ

/-- Theorem stating the number of clients needing both vegan and kosher meals -/
theorem both_vegan_and_kosher (m : MealDelivery) 
  (h_total : m.total = 30)
  (h_vegan : m.vegan = 7)
  (h_kosher : m.kosher = 8)
  (h_neither : m.neither = 18) :
  m.total - m.neither - (m.vegan + m.kosher - (m.total - m.neither)) = 3 := by
  sorry

#check both_vegan_and_kosher

end both_vegan_and_kosher_l2596_259695


namespace prism_sides_plus_two_l2596_259693

/-- A prism is a polyhedron with two congruent parallel faces (bases) and rectangular sides. -/
structure Prism where
  sides : ℕ

/-- The number of edges in a prism. -/
def Prism.edges (p : Prism) : ℕ := 3 * p.sides

/-- The number of vertices in a prism. -/
def Prism.vertices (p : Prism) : ℕ := 2 * p.sides

/-- Theorem: For a prism where the sum of its edges and vertices is 30,
    the number of sides plus 2 equals 8. -/
theorem prism_sides_plus_two (p : Prism) 
    (h : p.edges + p.vertices = 30) : p.sides + 2 = 8 := by
  sorry


end prism_sides_plus_two_l2596_259693


namespace matrix_power_50_l2596_259687

theorem matrix_power_50 (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = ![![1, 1], ![0, 1]] →
  A ^ 50 = ![![1, 50], ![0, 1]] := by
  sorry

end matrix_power_50_l2596_259687


namespace total_points_after_perfect_games_l2596_259627

/-- The number of points in a perfect score -/
def perfect_score : ℕ := 21

/-- The number of perfect games played -/
def games_played : ℕ := 11

/-- Theorem: The total points scored after playing 11 perfect games,
    where a perfect score is 21 points, is equal to 231 points. -/
theorem total_points_after_perfect_games :
  perfect_score * games_played = 231 := by
  sorry

end total_points_after_perfect_games_l2596_259627


namespace gcd_of_779_209_589_l2596_259694

theorem gcd_of_779_209_589 : Nat.gcd 779 (Nat.gcd 209 589) = 19 := by
  sorry

end gcd_of_779_209_589_l2596_259694


namespace t_leq_s_l2596_259600

theorem t_leq_s (a b : ℝ) (t s : ℝ) (ht : t = a + 2*b) (hs : s = a + b^2 + 1) : t ≤ s := by
  sorry

end t_leq_s_l2596_259600


namespace balloon_distribution_l2596_259605

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) (balloons_returned : ℕ) : 
  total_balloons = 250 → 
  num_friends = 5 → 
  balloons_returned = 11 → 
  (total_balloons / num_friends) - balloons_returned = 39 := by
sorry

end balloon_distribution_l2596_259605


namespace cat_leash_max_distance_l2596_259648

theorem cat_leash_max_distance :
  let center : ℝ × ℝ := (6, 2)
  let radius : ℝ := 15
  let origin : ℝ × ℝ := (0, 0)
  let max_distance := radius + Real.sqrt ((center.1 - origin.1)^2 + (center.2 - origin.2)^2)
  max_distance = 15 + 2 * Real.sqrt 10 := by
  sorry

end cat_leash_max_distance_l2596_259648


namespace ice_cream_cost_l2596_259620

theorem ice_cream_cost (total_spent : ℚ) (apple_extra_cost : ℚ) 
  (h1 : total_spent = 25)
  (h2 : apple_extra_cost = 10) : 
  ∃ (ice_cream_cost : ℚ), 
    ice_cream_cost + (ice_cream_cost + apple_extra_cost) = total_spent ∧ 
    ice_cream_cost = 7.5 := by
  sorry

end ice_cream_cost_l2596_259620


namespace library_book_sorting_l2596_259616

theorem library_book_sorting (damaged : ℕ) (obsolete : ℕ) : 
  obsolete = 6 * damaged - 8 →
  damaged + obsolete = 69 →
  damaged = 11 := by
sorry

end library_book_sorting_l2596_259616


namespace second_tea_price_l2596_259609

/-- Represents the price of tea varieties and their mixture --/
structure TeaPrices where
  first : ℝ
  second : ℝ
  third : ℝ
  mixture : ℝ

/-- Theorem stating the price of the second tea variety --/
theorem second_tea_price (p : TeaPrices)
  (h1 : p.first = 126)
  (h2 : p.third = 177.5)
  (h3 : p.mixture = 154)
  (h4 : p.mixture * 4 = p.first + p.second + 2 * p.third) :
  p.second = 135 := by
  sorry

end second_tea_price_l2596_259609


namespace equation_solution_l2596_259674

theorem equation_solution : ∃ x : ℝ, x = 25 ∧ Real.sqrt (1 + Real.sqrt (2 + x^2)) = (3 + Real.sqrt x) ^ (1/3 : ℝ) :=
  sorry

end equation_solution_l2596_259674


namespace power_product_inequality_l2596_259654

/-- Given positive real numbers a, b, and c, 
    a^a * b^b * c^c ≥ (a * b * c)^((a+b+c)/3) -/
theorem power_product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) := by
  sorry

end power_product_inequality_l2596_259654


namespace megan_cupcakes_l2596_259606

theorem megan_cupcakes (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (h1 : todd_ate = 32)
  (h2 : packages = 6)
  (h3 : cupcakes_per_package = 6) :
  todd_ate + packages * cupcakes_per_package = 68 := by
  sorry

end megan_cupcakes_l2596_259606


namespace sum_of_squares_l2596_259614

theorem sum_of_squares (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (sum_products : a * b + a * c + b * c = -3)
  (product : a * b * c = 2) :
  a^2 + b^2 + c^2 = 6 := by
sorry

end sum_of_squares_l2596_259614


namespace gcf_of_lcms_l2596_259680

theorem gcf_of_lcms : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 15 := by
  sorry

end gcf_of_lcms_l2596_259680


namespace complex_equation_solution_l2596_259682

theorem complex_equation_solution (b : ℝ) : (1 + b * Complex.I) * Complex.I = 1 + Complex.I → b = -1 := by
  sorry

end complex_equation_solution_l2596_259682


namespace polynomial_identity_coefficients_l2596_259601

theorem polynomial_identity_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x : ℝ, x^5 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + a₅*(x+2)^5) : 
  a₃ = 40 ∧ a₀ + a₁ + a₂ + a₄ + a₅ = -41 := by
  sorry

end polynomial_identity_coefficients_l2596_259601


namespace road_travel_rate_l2596_259632

/-- The rate per square meter for traveling roads on a rectangular lawn -/
theorem road_travel_rate (lawn_length lawn_width road_width total_cost : ℕ) :
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  total_cost = 5200 →
  (total_cost : ℚ) / ((lawn_length * road_width + lawn_width * road_width - road_width * road_width) : ℚ) = 4 := by
  sorry

end road_travel_rate_l2596_259632


namespace area_of_inscribing_square_l2596_259672

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 4*y = 12

/-- The circle is inscribed in a square with sides parallel to y-axis -/
axiom inscribed_in_square : ∃ (side : ℝ), ∀ (x y : ℝ), 
  circle_equation x y → (0 ≤ x ∧ x ≤ side) ∧ (0 ≤ y ∧ y ≤ side)

/-- The area of the square inscribing the circle -/
def square_area : ℝ := 68

/-- Theorem: The area of the square inscribing the circle is 68 square units -/
theorem area_of_inscribing_square : 
  ∃ (side : ℝ), (∀ (x y : ℝ), circle_equation x y → 
    (0 ≤ x ∧ x ≤ side) ∧ (0 ≤ y ∧ y ≤ side)) ∧ side^2 = square_area := by
  sorry

end area_of_inscribing_square_l2596_259672


namespace coin_toss_sequences_count_l2596_259670

/-- The number of ways to distribute n indistinguishable balls into k distinguishable urns -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of different sequences of 18 coin tosses with specific subsequence counts -/
def coinTossSequences : ℕ :=
  let numTosses := 18
  let numHH := 3
  let numHT := 4
  let numTH := 5
  let numTT := 6
  let numTGaps := numTH + 1
  let numHGaps := numHT + 1
  let tDistributions := starsAndBars numTT numTGaps
  let hDistributions := starsAndBars numHH numHGaps
  tDistributions * hDistributions

theorem coin_toss_sequences_count : coinTossSequences = 4200 := by
  sorry

end coin_toss_sequences_count_l2596_259670


namespace max_value_xyz_expression_l2596_259689

theorem max_value_xyz_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  xyz * (x + y + z) / ((x + y)^2 * (x + z)^2) ≤ (1 : ℝ) / 4 ∧
  (xyz * (x + y + z) / ((x + y)^2 * (x + z)^2) = (1 : ℝ) / 4 ↔ x = y ∧ y = z) :=
by sorry

end max_value_xyz_expression_l2596_259689


namespace triangle_side_length_l2596_259696

theorem triangle_side_length (a b c : ℝ) (R : ℝ) :
  a > 0 → b > 0 → c > 0 → R > 0 →
  (a^2 / (b * c)) - (c / b) - (b / c) = Real.sqrt 3 →
  R = 3 →
  a = 3 := by
sorry

end triangle_side_length_l2596_259696


namespace simplify_expression_l2596_259666

theorem simplify_expression (y : ℝ) :
  4 * y - 6 * y^2 + 8 - (3 + 5 * y - 9 * y^2) = 3 * y^2 - y + 5 :=
by sorry

end simplify_expression_l2596_259666


namespace simple_interest_principal_l2596_259685

/-- Simple interest calculation -/
theorem simple_interest_principal
  (rate : ℚ)
  (time : ℚ)
  (interest : ℚ)
  (h_rate : rate = 4 / 100)
  (h_time : time = 1)
  (h_interest : interest = 400) :
  interest = (10000 : ℚ) * rate * time :=
sorry

end simple_interest_principal_l2596_259685


namespace weight_difference_l2596_259697

theorem weight_difference (rachel jimmy adam : ℝ) 
  (h1 : rachel = 75)
  (h2 : rachel < jimmy)
  (h3 : rachel = adam + 15)
  (h4 : (rachel + jimmy + adam) / 3 = 72) :
  jimmy - rachel = 6 := by
sorry

end weight_difference_l2596_259697


namespace candy_distribution_l2596_259691

theorem candy_distribution (hugh tommy melany lily : ℝ) 
  (h_hugh : hugh = 8.5)
  (h_tommy : tommy = 6.75)
  (h_melany : melany = 7.25)
  (h_lily : lily = 5.5) :
  let total := hugh + tommy + melany + lily
  let num_people := 4
  (total / num_people) = 7 := by sorry

end candy_distribution_l2596_259691


namespace two_std_dev_below_mean_example_l2596_259661

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : 0 < std_dev

/-- The value that is exactly 2 standard deviations less than the mean -/
def two_std_dev_below_mean (d : NormalDistribution) : ℝ :=
  d.mean - 2 * d.std_dev

/-- Theorem: For a normal distribution with mean 15.5 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 12.5 -/
theorem two_std_dev_below_mean_example :
  let d : NormalDistribution := ⟨15.5, 1.5, by norm_num⟩
  two_std_dev_below_mean d = 12.5 := by sorry

end two_std_dev_below_mean_example_l2596_259661


namespace danes_daughters_flowers_l2596_259669

def flowers_per_basket (people : ℕ) (flowers_per_person : ℕ) (additional_growth : ℕ) (died : ℕ) (baskets : ℕ) : ℕ :=
  ((people * flowers_per_person + additional_growth - died) / baskets)

theorem danes_daughters_flowers :
  flowers_per_basket 2 5 20 10 5 = 4 := by
  sorry

end danes_daughters_flowers_l2596_259669


namespace emilys_initial_lives_l2596_259676

theorem emilys_initial_lives :
  ∀ (initial_lives : ℕ),
  initial_lives - 25 + 24 = 41 →
  initial_lives = 42 :=
by sorry

end emilys_initial_lives_l2596_259676


namespace largest_circle_area_l2596_259653

theorem largest_circle_area (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) -- Right triangle condition
  (h5 : π * (a/2)^2 + π * (b/2)^2 + π * (c/2)^2 = 338 * π) : -- Sum of circle areas
  π * (c/2)^2 = 169 * π := by
sorry

end largest_circle_area_l2596_259653


namespace cube_root_three_equation_l2596_259660

theorem cube_root_three_equation (s : ℝ) : 
  s = 1 / (2 - Real.rpow 3 (1/3)) → 
  s = ((2 + Real.rpow 3 (1/3)) * (4 + Real.sqrt 3)) / 13 := by
sorry

end cube_root_three_equation_l2596_259660


namespace zero_not_equivalent_to_intersection_l2596_259643

-- Define a real-valued function
def RealFunction := ℝ → ℝ

-- Define the zero of a function
def is_zero_of_function (f : RealFunction) (x : ℝ) : Prop := f x = 0

-- Define the intersection point of a function's graph and the x-axis
def is_intersection_with_x_axis (f : RealFunction) (x : ℝ) : Prop :=
  f x = 0 ∧ ∀ y : ℝ, y ≠ 0 → f x ≠ y

-- Theorem stating that these concepts are not equivalent
theorem zero_not_equivalent_to_intersection :
  ¬ (∀ (f : RealFunction) (x : ℝ), is_zero_of_function f x ↔ is_intersection_with_x_axis f x) :=
sorry

end zero_not_equivalent_to_intersection_l2596_259643


namespace solve_equation_l2596_259640

theorem solve_equation (y : ℝ) : 3/4 + 1/y = 7/8 ↔ y = 8 := by sorry

end solve_equation_l2596_259640


namespace rex_driving_lessons_l2596_259681

theorem rex_driving_lessons 
  (total_hours : ℕ) 
  (hours_per_week : ℕ) 
  (remaining_weeks : ℕ) 
  (h1 : total_hours = 40)
  (h2 : hours_per_week = 4)
  (h3 : remaining_weeks = 4) :
  total_hours - (remaining_weeks * hours_per_week) = 6 * hours_per_week :=
by sorry

end rex_driving_lessons_l2596_259681


namespace percent_of_y_l2596_259649

theorem percent_of_y (y : ℝ) (h : y > 0) : ((6 * y) / 20 + (3 * y) / 10) / y = 60 / 100 := by
  sorry

end percent_of_y_l2596_259649


namespace cream_cheese_price_l2596_259621

-- Define variables for bagel and cream cheese prices
variable (B : ℝ) -- Price of one bag of bagels
variable (C : ℝ) -- Price of one package of cream cheese

-- Define the equations from the problem
def monday_equation : Prop := 2 * B + 3 * C = 12
def friday_equation : Prop := 4 * B + 2 * C = 14

-- Theorem statement
theorem cream_cheese_price 
  (h1 : monday_equation B C) 
  (h2 : friday_equation B C) : 
  C = 2.5 := by sorry

end cream_cheese_price_l2596_259621


namespace first_podium_height_calculation_l2596_259673

/-- The height of the second prize podium in centimeters -/
def second_podium_height : ℚ := 53 + 7 / 10

/-- Hyeonjoo's measured height on the second prize podium in centimeters -/
def height_on_second_podium : ℚ := 190

/-- Hyeonjoo's measured height on the first prize podium in centimeters -/
def height_on_first_podium : ℚ := 232 + 5 / 10

/-- The height of the first prize podium in centimeters -/
def first_podium_height : ℚ := height_on_first_podium - (height_on_second_podium - second_podium_height)

theorem first_podium_height_calculation :
  first_podium_height = 96.2 := by sorry

end first_podium_height_calculation_l2596_259673


namespace auntie_em_parking_probability_l2596_259645

def total_spaces : ℕ := 18
def parked_cars : ℕ := 12
def suv_spaces : ℕ := 2

theorem auntie_em_parking_probability :
  let total_configurations := Nat.choose total_spaces parked_cars
  let unfavorable_configurations := Nat.choose (parked_cars + 1) parked_cars
  (total_configurations - unfavorable_configurations : ℚ) / total_configurations = 1403 / 1546 :=
by sorry

end auntie_em_parking_probability_l2596_259645


namespace parabola_y_intercepts_l2596_259628

/-- The number of y-intercepts for the parabola x = 3y^2 - 4y + 5 -/
theorem parabola_y_intercepts :
  let f (y : ℝ) := 3 * y^2 - 4 * y + 5
  ∃! x : ℝ, (∀ y : ℝ, f y = x) ∧ (¬ ∃ y : ℝ, f y = 0) :=
by sorry

end parabola_y_intercepts_l2596_259628


namespace product_of_repeating_decimal_and_eight_l2596_259629

theorem product_of_repeating_decimal_and_eight :
  ∃ (x : ℚ), (∃ (n : ℕ), x = (456 : ℚ) / (10^n - 1)) ∧ 8 * x = 1216 / 333 := by
  sorry

end product_of_repeating_decimal_and_eight_l2596_259629


namespace rohans_salary_l2596_259635

/-- Rohan's monthly salary in rupees -/
def monthly_salary : ℝ := 7500

/-- Percentage of salary spent on food -/
def food_percentage : ℝ := 40

/-- Percentage of salary spent on house rent -/
def rent_percentage : ℝ := 20

/-- Percentage of salary spent on entertainment -/
def entertainment_percentage : ℝ := 10

/-- Percentage of salary spent on conveyance -/
def conveyance_percentage : ℝ := 10

/-- Rohan's savings at the end of the month in rupees -/
def savings : ℝ := 1500

theorem rohans_salary :
  monthly_salary = 7500 ∧
  food_percentage + rent_percentage + entertainment_percentage + conveyance_percentage + (savings / monthly_salary * 100) = 100 :=
by sorry

end rohans_salary_l2596_259635


namespace factorization_problems_l2596_259646

theorem factorization_problems (x y : ℝ) : 
  (x^2 - 6*x + 9 = (x - 3)^2) ∧ 
  (x^2*(y - 2) - 4*(y - 2) = (y - 2)*(x + 2)*(x - 2)) := by
  sorry

end factorization_problems_l2596_259646


namespace no_positive_integer_triples_l2596_259641

theorem no_positive_integer_triples : 
  ¬∃ (a b c : ℕ+), (Nat.factorial a.val + b.val^3 = 18 + c.val^3) := by
  sorry

end no_positive_integer_triples_l2596_259641


namespace power_seven_eight_mod_hundred_l2596_259613

theorem power_seven_eight_mod_hundred : 7^8 % 100 = 1 := by
  sorry

end power_seven_eight_mod_hundred_l2596_259613


namespace decreasing_power_function_l2596_259683

/-- A power function y = ax^b is decreasing on (0, +∞) if and only if b = -3 -/
theorem decreasing_power_function (a : ℝ) (b : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → a * x₁^b > a * x₂^b) ↔ b = -3 :=
by sorry

end decreasing_power_function_l2596_259683


namespace school_weeks_l2596_259655

/-- Proves that the number of school weeks is 36 given the conditions --/
theorem school_weeks (sandwiches_per_week : ℕ) (missed_days : ℕ) (total_sandwiches : ℕ) 
  (h1 : sandwiches_per_week = 2)
  (h2 : missed_days = 3)
  (h3 : total_sandwiches = 69) : 
  (total_sandwiches + missed_days) / sandwiches_per_week = 36 := by
  sorry

#check school_weeks

end school_weeks_l2596_259655


namespace f_period_l2596_259665

open Real

noncomputable def f (x : ℝ) : ℝ := 
  (sin (2 * x) + sin (2 * x + π / 3)) / (cos (2 * x) + cos (2 * x + π / 3))

theorem f_period : 
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ 
  T = π / 2 :=
sorry

end f_period_l2596_259665


namespace upper_limit_of_set_A_l2596_259679

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def SetA : Set ℕ := {n : ℕ | isPrime n ∧ n > 15}

theorem upper_limit_of_set_A (lower_bound : ℕ) (h1 : lower_bound ∈ SetA) 
  (h2 : ∀ x ∈ SetA, x ≥ lower_bound) 
  (h3 : ∃ upper_bound : ℕ, upper_bound ∈ SetA ∧ upper_bound - lower_bound = 14) :
  ∃ max_element : ℕ, max_element ∈ SetA ∧ max_element = 31 :=
sorry

end upper_limit_of_set_A_l2596_259679


namespace ab_bounds_l2596_259615

theorem ab_bounds (a b c : ℝ) (h1 : a ≠ b) (h2 : c > 0)
  (h3 : a^4 - 2019*a = c) (h4 : b^4 - 2019*b = c) :
  -Real.sqrt c < a * b ∧ a * b < 0 := by
  sorry

end ab_bounds_l2596_259615


namespace counterexample_exists_l2596_259625

theorem counterexample_exists : ∃ n : ℕ, 15 ≤ n ∧ n ≤ 30 ∧ ¬(Nat.Prime n) ∧ Nat.Prime (n - 5) := by
  sorry

end counterexample_exists_l2596_259625


namespace oil_barrels_problem_l2596_259664

/-- The minimum number of barrels needed to contain a given amount of oil -/
def min_barrels (total_oil : ℕ) (barrel_capacity : ℕ) : ℕ :=
  (total_oil + barrel_capacity - 1) / barrel_capacity

/-- Proof that at least 7 barrels are needed for 250 kg of oil with 40 kg capacity barrels -/
theorem oil_barrels_problem :
  min_barrels 250 40 = 7 := by
  sorry

end oil_barrels_problem_l2596_259664


namespace vote_count_proof_l2596_259699

theorem vote_count_proof (T : ℕ) (F : ℕ) (A : ℕ) : 
  F = A + 68 →  -- 68 more votes in favor than against
  A = (40 * T) / 100 →  -- 40% of total votes were against
  T = F + A →  -- total votes is sum of for and against
  T = 340 :=
by sorry

end vote_count_proof_l2596_259699


namespace fred_gave_233_marbles_l2596_259686

/-- The number of black marbles Fred gave to Sara -/
def marbles_from_fred (initial_marbles final_marbles : ℕ) : ℕ :=
  final_marbles - initial_marbles

/-- Theorem stating that Fred gave Sara 233 black marbles -/
theorem fred_gave_233_marbles :
  let initial_marbles : ℕ := 792
  let final_marbles : ℕ := 1025
  marbles_from_fred initial_marbles final_marbles = 233 := by
  sorry

end fred_gave_233_marbles_l2596_259686


namespace equation_solution_l2596_259688

theorem equation_solution :
  ∃ k : ℚ, (3 * k - 4) / (k + 7) = 2 / 5 ↔ k = 34 / 13 := by
  sorry

end equation_solution_l2596_259688


namespace cube_edge_length_is_twelve_l2596_259678

/-- Represents a cube with integer edge length -/
structure Cube where
  edge_length : ℕ

/-- Calculates the number of small cubes with three painted faces -/
def three_painted_faces (c : Cube) : ℕ := 8

/-- Calculates the number of small cubes with two painted faces -/
def two_painted_faces (c : Cube) : ℕ := 12 * (c.edge_length - 2)

/-- Theorem stating that when the number of small cubes with two painted faces
    is 15 times the number of small cubes with three painted faces,
    the edge length of the cube must be 12 -/
theorem cube_edge_length_is_twelve (c : Cube) :
  two_painted_faces c = 15 * three_painted_faces c → c.edge_length = 12 := by
  sorry


end cube_edge_length_is_twelve_l2596_259678


namespace tablet_screen_area_difference_l2596_259677

theorem tablet_screen_area_difference : 
  let diagonal_8 : ℝ := 8
  let diagonal_7 : ℝ := 7
  let area_8 : ℝ := (diagonal_8^2) / 2
  let area_7 : ℝ := (diagonal_7^2) / 2
  area_8 - area_7 = 7.5 := by sorry

end tablet_screen_area_difference_l2596_259677


namespace least_divisible_by_three_l2596_259617

theorem least_divisible_by_three (x : ℕ) : (∃ y : ℕ, y > 0 ∧ 23 * y % 3 = 0) → x ≥ 1 :=
sorry

end least_divisible_by_three_l2596_259617


namespace passengers_off_in_texas_l2596_259604

/-- Represents the number of passengers at different stages of the flight --/
structure FlightPassengers where
  initial : ℕ
  texas_off : ℕ
  texas_on : ℕ
  nc_off : ℕ
  nc_on : ℕ
  final : ℕ

/-- Theorem stating that 48 passengers got off in Texas --/
theorem passengers_off_in_texas (fp : FlightPassengers) 
  (h1 : fp.initial = 124)
  (h2 : fp.texas_on = 24)
  (h3 : fp.nc_off = 47)
  (h4 : fp.nc_on = 14)
  (h5 : fp.final = 67)
  (h6 : fp.initial - fp.texas_off + fp.texas_on - fp.nc_off + fp.nc_on = fp.final) :
  fp.texas_off = 48 := by
  sorry


end passengers_off_in_texas_l2596_259604


namespace expected_intersection_value_l2596_259667

/-- A subset of consecutive integers from {1,2,3,4,5,6,7,8} -/
def ConsecutiveSubset := List ℕ

/-- The set of all possible consecutive subsets -/
def allSubsets : Finset ConsecutiveSubset :=
  sorry

/-- The probability of an element x being in a randomly chosen subset -/
def P (x : ℕ) : ℚ :=
  sorry

/-- The expected number of elements in the intersection of three independently chosen subsets -/
def expectedIntersection : ℚ :=
  sorry

theorem expected_intersection_value :
  expectedIntersection = 178 / 243 := by
  sorry

end expected_intersection_value_l2596_259667


namespace square_area_from_corners_l2596_259647

/-- The area of a square with adjacent corners at (4, -1) and (-1, 3) on a Cartesian coordinate plane is 41. -/
theorem square_area_from_corners : 
  let p1 : ℝ × ℝ := (4, -1)
  let p2 : ℝ × ℝ := (-1, 3)
  let side_length := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  side_length^2 = 41 := by
  sorry

end square_area_from_corners_l2596_259647
