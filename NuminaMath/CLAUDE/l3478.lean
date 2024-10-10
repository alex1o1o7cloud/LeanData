import Mathlib

namespace union_of_sets_l3478_347851

theorem union_of_sets : 
  let A : Set Int := {-1, 1, 2, 4}
  let B : Set Int := {-1, 0, 2}
  A ∪ B = {-1, 0, 1, 2, 4} := by
sorry

end union_of_sets_l3478_347851


namespace trig_expression_equality_l3478_347858

theorem trig_expression_equality : 
  (Real.sin (15 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (19 * π / 180) * Real.cos (11 * π / 180) + 
   Real.cos (161 * π / 180) * Real.cos (101 * π / 180)) = 
  Real.sin (5 * π / 180) / Real.sin (8 * π / 180) := by sorry

end trig_expression_equality_l3478_347858


namespace complement_A_intersect_B_l3478_347882

-- Define the sets A and B
def A : Set ℝ := {x | x + 1 < 0}
def B : Set ℝ := {x | x - 3 < 0}

-- Define the complement of A in ℝ
def complementA : Set ℝ := {x | x ∉ A}

-- State the theorem
theorem complement_A_intersect_B :
  complementA ∩ B = {x : ℝ | -1 ≤ x ∧ x < 3} := by sorry

end complement_A_intersect_B_l3478_347882


namespace opposite_to_gold_is_yellow_l3478_347880

/-- Represents the colors used on the cube faces -/
inductive Color
  | Blue
  | Yellow
  | Orange
  | Black
  | Silver
  | Gold

/-- Represents the positions of faces on the cube -/
inductive Position
  | Top
  | Bottom
  | Front
  | Back
  | Left
  | Right

/-- Represents a view of the cube, showing top, front, and right faces -/
structure CubeView where
  top : Color
  front : Color
  right : Color

/-- Represents a cube with colored faces -/
structure Cube where
  faces : Position → Color

/-- The three views of the cube given in the problem -/
def givenViews : List CubeView := [
  { top := Color.Blue, front := Color.Yellow, right := Color.Orange },
  { top := Color.Blue, front := Color.Black,  right := Color.Orange },
  { top := Color.Blue, front := Color.Silver, right := Color.Orange }
]

/-- Theorem stating that the face opposite to gold is yellow -/
theorem opposite_to_gold_is_yellow (cube : Cube) 
    (h1 : ∀ view ∈ givenViews, 
      cube.faces Position.Top = view.top ∧ 
      cube.faces Position.Right = view.right ∧ 
      (cube.faces Position.Front = view.front ∨ 
       cube.faces Position.Left = view.front ∨ 
       cube.faces Position.Bottom = view.front))
    (h2 : ∃! pos, cube.faces pos = Color.Gold) :
    cube.faces Position.Front = Color.Yellow :=
  sorry

end opposite_to_gold_is_yellow_l3478_347880


namespace simplify_and_evaluate_l3478_347898

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 16 + Real.tan (45 * π / 180)) :
  (m + 2 + 5 / (2 - m)) * ((2 * m - 4) / (3 - m)) = -16 := by
  sorry

end simplify_and_evaluate_l3478_347898


namespace major_preference_stronger_than_gender_l3478_347838

/-- Represents the observed K^2 value for gender preference --/
def k1 : ℝ := 1.010

/-- Represents the observed K^2 value for major preference --/
def k2 : ℝ := 9.090

/-- Theorem stating that the observed K^2 value for major preference is greater than the observed K^2 value for gender preference --/
theorem major_preference_stronger_than_gender : k2 > k1 := by sorry

end major_preference_stronger_than_gender_l3478_347838


namespace division_problem_l3478_347863

theorem division_problem (a b c : ℝ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/3) : 
  c / a = 1/2 := by sorry

end division_problem_l3478_347863


namespace ice_problem_solution_l3478_347867

def ice_problem (tray_a_initial tray_a_added : ℕ) : ℕ :=
  let tray_a := tray_a_initial + tray_a_added
  let tray_b := tray_a / 3
  let tray_c := 2 * tray_a
  tray_a + tray_b + tray_c

theorem ice_problem_solution :
  ice_problem 2 7 = 30 := by
  sorry

end ice_problem_solution_l3478_347867


namespace contrapositive_not_always_false_l3478_347803

theorem contrapositive_not_always_false :
  ∃ (p q : Prop), (p → q) ∧ ¬(¬q → ¬p) → False :=
sorry

end contrapositive_not_always_false_l3478_347803


namespace relay_race_distance_per_member_l3478_347818

theorem relay_race_distance_per_member 
  (total_distance : ℝ) 
  (team_members : ℕ) 
  (h1 : total_distance = 150) 
  (h2 : team_members = 5) : 
  total_distance / team_members = 30 := by
sorry

end relay_race_distance_per_member_l3478_347818


namespace geometric_arithmetic_geometric_sequence_l3478_347822

theorem geometric_arithmetic_geometric_sequence :
  ∀ a q : ℝ,
  (∀ x y z : ℝ, x = a ∧ y = a * q ∧ z = a * q^2 →
    (2 * (a * q + 8) = a + a * q^2) ∧
    ((a * q + 8)^2 = a * (a * q^2 + 64))) →
  ((a = 4 ∧ q = 3) ∨ (a = 4/9 ∧ q = -5)) :=
by sorry

end geometric_arithmetic_geometric_sequence_l3478_347822


namespace fraction_problem_l3478_347842

theorem fraction_problem : 
  let x : ℚ := 2/3
  (3/4 : ℚ) * (4/5 : ℚ) * x = (2/5 : ℚ) := by
  sorry

end fraction_problem_l3478_347842


namespace hyperbola_equation_l3478_347813

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_asymptote : b / a = Real.sqrt 5 / 2
  h_shared_focus : ∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c^2 = 3

/-- The equation of the hyperbola is x^2/4 - y^2/5 = 1 -/
theorem hyperbola_equation (C : Hyperbola) : C.a^2 = 4 ∧ C.b^2 = 5 := by
  sorry

end hyperbola_equation_l3478_347813


namespace right_triangle_sinC_l3478_347887

theorem right_triangle_sinC (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : B = Real.pi / 2) (h3 : Real.tan A = 3 / 4) : Real.sin C = 4 / 5 := by
  sorry

end right_triangle_sinC_l3478_347887


namespace symmetric_function_k_range_l3478_347897

/-- A function f is symmetric if it's monotonic on its domain D and there exists an interval [a,b] ⊆ D such that the range of f on [a,b] is [-b,-a] -/
def IsSymmetric (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  Monotone f ∧ ∃ a b, a < b ∧ Set.Icc a b ⊆ D ∧ Set.image f (Set.Icc a b) = Set.Icc (-b) (-a)

/-- The main theorem stating that if f(x) = √(2 - x) - k is symmetric on (-∞, 2], then k ∈ [2, 9/4) -/
theorem symmetric_function_k_range :
  ∀ k : ℝ, IsSymmetric (fun x ↦ Real.sqrt (2 - x) - k) (Set.Iic 2) →
  k ∈ Set.Icc 2 (9/4) := by sorry

end symmetric_function_k_range_l3478_347897


namespace inequality_proof_l3478_347853

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end inequality_proof_l3478_347853


namespace quadratic_root_range_l3478_347873

theorem quadratic_root_range (m : ℝ) (α β : ℝ) : 
  (∃ x, x^2 - m*x + 1 = 0) ∧ 
  (α^2 - m*α + 1 = 0) ∧ 
  (β^2 - m*β + 1 = 0) ∧ 
  (0 < α) ∧ (α < 1) ∧ 
  (1 < β) ∧ (β < 2) →
  (2 < m) ∧ (m < 5/2) := by
sorry

end quadratic_root_range_l3478_347873


namespace scientific_notation_of_120000_l3478_347827

theorem scientific_notation_of_120000 :
  (120000 : ℝ) = 1.2 * (10 ^ 5) := by
  sorry

end scientific_notation_of_120000_l3478_347827


namespace geometric_sequence_ratio_l3478_347826

/-- For a geometric sequence with common ratio -1/3, the ratio of the sum of odd-indexed terms
    to the sum of even-indexed terms (up to the 8th term) is -3. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h : q = -1/3) :
  (∀ n : ℕ, a (n + 1) = q * a n) →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by sorry

end geometric_sequence_ratio_l3478_347826


namespace eight_queens_exists_l3478_347896

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Checks if two positions are on the same diagonal -/
def sameDiagonal (p1 p2 : Position) : Prop :=
  (p1.row.val : Int) - (p1.col.val : Int) = (p2.row.val : Int) - (p2.col.val : Int) ∨
  (p1.row.val : Int) + (p1.col.val : Int) = (p2.row.val : Int) + (p2.col.val : Int)

/-- Checks if two queens threaten each other -/
def threaten (p1 p2 : Position) : Prop :=
  p1.row = p2.row ∨ p1.col = p2.col ∨ sameDiagonal p1 p2

/-- Represents an arrangement of eight queens on the chessboard -/
def QueenArrangement := Fin 8 → Position

/-- Checks if a queen arrangement is valid (no queens threaten each other) -/
def validArrangement (arrangement : QueenArrangement) : Prop :=
  ∀ i j : Fin 8, i ≠ j → ¬threaten (arrangement i) (arrangement j)

/-- Theorem: There exists a valid arrangement of eight queens on an 8x8 chessboard -/
theorem eight_queens_exists : ∃ arrangement : QueenArrangement, validArrangement arrangement :=
sorry

end eight_queens_exists_l3478_347896


namespace unique_solution_l3478_347831

/-- A 3x3 matrix with special properties -/
structure SpecialMatrix where
  a : Matrix (Fin 3) (Fin 3) ℝ
  all_positive : ∀ i j, 0 < a i j
  row_sum_one : ∀ i, (Finset.univ.sum (λ j => a i j)) = 1
  col_sum_one : ∀ j, (Finset.univ.sum (λ i => a i j)) = 1
  diagonal_half : ∀ i, a i i = 1/2

/-- The system of equations -/
def system (m : SpecialMatrix) (x y z : ℝ) : Prop :=
  m.a 0 0 * x + m.a 0 1 * y + m.a 0 2 * z = 0 ∧
  m.a 1 0 * x + m.a 1 1 * y + m.a 1 2 * z = 0 ∧
  m.a 2 0 * x + m.a 2 1 * y + m.a 2 2 * z = 0

theorem unique_solution (m : SpecialMatrix) :
  ∀ x y z, system m x y z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end unique_solution_l3478_347831


namespace solution_set_part1_range_of_a_part2_l3478_347890

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end solution_set_part1_range_of_a_part2_l3478_347890


namespace modular_arithmetic_problem_l3478_347843

theorem modular_arithmetic_problem :
  ∃ (a b : ℕ), a < 65 ∧ b < 65 ∧ (4 * a) % 65 = 1 ∧ (13 * b) % 65 = 1 ∧
  (3 * a + 7 * b) % 65 = 47 := by
  sorry

end modular_arithmetic_problem_l3478_347843


namespace equation_solutions_l3478_347859

theorem equation_solutions (n : ℕ) : 
  (∃! (solutions : Finset (ℕ × ℕ × ℕ)), 
    solutions.card = 10 ∧ 
    ∀ (x y z : ℕ), (x, y, z) ∈ solutions ↔ 
      (x > 0 ∧ y > 0 ∧ z > 0 ∧ 4*x + 6*y + 2*z = n)) ↔ 
  (n = 32 ∨ n = 33) :=
sorry

end equation_solutions_l3478_347859


namespace unique_root_quadratic_root_l3478_347821

/-- A quadratic polynomial with exactly one root -/
structure UniqueRootQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  has_unique_root : (b ^ 2 - 4 * a * c) = 0

/-- The theorem stating that the root of the quadratic polynomial is -11 -/
theorem unique_root_quadratic_root (f : UniqueRootQuadratic) 
  (h : ∃ g : UniqueRootQuadratic, 
    g.a = -f.a ∧ 
    g.b = (f.b - 30 * f.a) ∧ 
    g.c = (17 * f.a - 7 * f.b + f.c)) :
  (f.a ≠ 0) → (-f.b / (2 * f.a)) = -11 :=
by sorry

end unique_root_quadratic_root_l3478_347821


namespace ed_hotel_stay_l3478_347807

def hotel_problem (night_rate : ℚ) (morning_rate : ℚ) (initial_money : ℚ) (night_hours : ℚ) (money_left : ℚ) : ℚ :=
  let total_spent := initial_money - money_left
  let night_cost := night_rate * night_hours
  let morning_spent := total_spent - night_cost
  morning_spent / morning_rate

theorem ed_hotel_stay :
  hotel_problem 1.5 2 80 6 63 = 4 := by
  sorry

end ed_hotel_stay_l3478_347807


namespace overtime_hours_calculation_l3478_347871

/-- Calculates overtime hours given regular pay rate, regular hours, and total pay -/
def calculate_overtime_hours (regular_rate : ℚ) (regular_hours : ℚ) (total_pay : ℚ) : ℚ :=
  let regular_pay := regular_rate * regular_hours
  let overtime_rate := 2 * regular_rate
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate

/-- Theorem stating that given the problem conditions, the overtime hours are 11 -/
theorem overtime_hours_calculation :
  let regular_rate : ℚ := 3
  let regular_hours : ℚ := 40
  let total_pay : ℚ := 186
  calculate_overtime_hours regular_rate regular_hours total_pay = 11 := by
  sorry

end overtime_hours_calculation_l3478_347871


namespace cos_sin_difference_equals_sqrt3_over_2_l3478_347837

theorem cos_sin_difference_equals_sqrt3_over_2 :
  Real.cos (10 * π / 180) * Real.sin (70 * π / 180) -
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end cos_sin_difference_equals_sqrt3_over_2_l3478_347837


namespace budget_circle_graph_l3478_347808

theorem budget_circle_graph (transportation research_development utilities equipment supplies : ℝ)
  (h1 : transportation = 15)
  (h2 : research_development = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : supplies = 2)
  (h6 : transportation + research_development + utilities + equipment + supplies < 100) :
  let salaries := 100 - (transportation + research_development + utilities + equipment + supplies)
  (salaries / 100) * 360 = 234 := by
sorry

end budget_circle_graph_l3478_347808


namespace exists_linear_bound_l3478_347829

def Color := Bool

def is_valid_coloring (coloring : ℕ+ → Color) : Prop :=
  ∀ n : ℕ+, coloring n = true ∨ coloring n = false

structure ColoredIntegerFunction where
  f : ℕ+ → ℕ+
  coloring : ℕ+ → Color
  is_valid_coloring : is_valid_coloring coloring
  monotone : ∀ x y : ℕ+, x ≤ y → f x ≤ f y
  color_additive : ∀ x y z : ℕ+, 
    coloring x = coloring y ∧ coloring y = coloring z → 
    x + y = z → f x + f y = f z

theorem exists_linear_bound (cf : ColoredIntegerFunction) : 
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℕ+, (cf.f x : ℝ) ≤ a * x :=
sorry

end exists_linear_bound_l3478_347829


namespace two_trains_problem_l3478_347810

/-- Two trains problem -/
theorem two_trains_problem (train_length : ℝ) (time_first_train : ℝ) (time_crossing : ℝ) :
  train_length = 120 →
  time_first_train = 12 →
  time_crossing = 16 →
  ∃ time_second_train : ℝ,
    time_second_train = 24 ∧
    train_length / time_first_train + train_length / time_second_train = 2 * train_length / time_crossing :=
by sorry

end two_trains_problem_l3478_347810


namespace car_license_combinations_l3478_347862

def letter_choices : ℕ := 2
def digit_choices : ℕ := 10
def num_digits : ℕ := 6

def total_license_combinations : ℕ := letter_choices * digit_choices ^ num_digits

theorem car_license_combinations :
  total_license_combinations = 2000000 := by
  sorry

end car_license_combinations_l3478_347862


namespace emilys_cards_l3478_347847

theorem emilys_cards (initial_cards : ℕ) (cards_per_apple : ℕ) (bruce_apples : ℕ) :
  initial_cards + cards_per_apple * bruce_apples = 
  initial_cards + cards_per_apple * bruce_apples := by
  sorry

#check emilys_cards 63 7 13

end emilys_cards_l3478_347847


namespace simplify_fraction_l3478_347811

theorem simplify_fraction (b : ℚ) (h : b = 4) : 18 * b^4 / (27 * b^3) = 8 / 3 := by
  sorry

end simplify_fraction_l3478_347811


namespace equal_intercept_line_equation_l3478_347868

/-- A line passing through the point (3, -4) with equal intercepts on the coordinate axes -/
structure EqualInterceptLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (3, -4)
  point_condition : -4 = m * 3 + b
  -- The line has equal intercepts on both axes
  equal_intercepts : m ≠ -1 → b / (1 + m) = -b / m

/-- The equation of an EqualInterceptLine is either 4x + 3y = 0 or x + y + 1 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (4 * l.m + 3 = 0 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = -1) :=
sorry

end equal_intercept_line_equation_l3478_347868


namespace fraction_value_l3478_347832

theorem fraction_value : (20 + 24) / (20 - 24) = -11 := by sorry

end fraction_value_l3478_347832


namespace cube_root_125_times_fourth_root_256_times_sixth_root_64_l3478_347844

theorem cube_root_125_times_fourth_root_256_times_sixth_root_64 :
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/6) = 40 := by
  sorry

end cube_root_125_times_fourth_root_256_times_sixth_root_64_l3478_347844


namespace fractional_equation_solution_l3478_347830

theorem fractional_equation_solution : 
  ∃ x : ℝ, (x * (x - 2) ≠ 0) ∧ (5 / (x - 2) = 3 / x) ∧ (x = -3) := by
  sorry

end fractional_equation_solution_l3478_347830


namespace marble_sculpture_second_week_cut_l3478_347878

/-- Proves that the percentage of marble cut away in the second week is 20% --/
theorem marble_sculpture_second_week_cut (
  original_weight : ℝ)
  (first_week_cut_percent : ℝ)
  (third_week_cut_percent : ℝ)
  (final_weight : ℝ)
  (h1 : original_weight = 250)
  (h2 : first_week_cut_percent = 30)
  (h3 : third_week_cut_percent = 25)
  (h4 : final_weight = 105)
  : ∃ (second_week_cut_percent : ℝ),
    second_week_cut_percent = 20 ∧
    final_weight = original_weight *
      (1 - first_week_cut_percent / 100) *
      (1 - second_week_cut_percent / 100) *
      (1 - third_week_cut_percent / 100) :=
by sorry

end marble_sculpture_second_week_cut_l3478_347878


namespace sin_angle_RPT_l3478_347802

theorem sin_angle_RPT (RPQ : Real) (h : Real.sin RPQ = 3/5) : 
  Real.sin (2 * Real.pi - RPQ) = 3/5 := by
  sorry

end sin_angle_RPT_l3478_347802


namespace sin_plus_cos_equals_one_fifth_l3478_347804

/-- Given that the terminal side of angle α passes through the point (3a, -4a) where a < 0,
    prove that sin α + cos α = 1/5 -/
theorem sin_plus_cos_equals_one_fifth 
  (α : Real) (a : Real) (h1 : a < 0) 
  (h2 : ∃ (t : Real), t > 0 ∧ Real.cos α = 3 * a / t ∧ Real.sin α = -4 * a / t) : 
  Real.sin α + Real.cos α = 1 / 5 := by
  sorry

end sin_plus_cos_equals_one_fifth_l3478_347804


namespace f_properties_when_a_is_1_f_minimum_on_interval_l3478_347864

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Part 1
theorem f_properties_when_a_is_1 :
  let a := 1
  ∀ x y : ℝ, x < y ∧ y ≤ 1 → f a x > f a y ∧
  ∀ z : ℝ, f a z ≥ 1 ∧ ∃ w : ℝ, f a w = 1 := by sorry

-- Part 2
theorem f_minimum_on_interval (a : ℝ) (h : a ≥ -1) :
  let min_value := if a < 1 then -a^2 + 2 else 3 - 2*a
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a x ≥ min_value ∧
  ∃ y : ℝ, y ∈ Set.Icc (-1) 1 ∧ f a y = min_value := by sorry

end f_properties_when_a_is_1_f_minimum_on_interval_l3478_347864


namespace interest_problem_l3478_347899

/-- Given compound and simple interest conditions, prove the principal amount -/
theorem interest_problem (P R : ℝ) : 
  P * ((1 + R / 100) ^ 2 - 1) = 11730 →
  (P * R * 2) / 100 = 10200 →
  P = 34000 := by
  sorry

end interest_problem_l3478_347899


namespace problem_solution_l3478_347870

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1 - x) / (a * x)

theorem problem_solution (a : ℝ) (h : a > 0) :
  (∀ x ≥ 1, Monotone (f a) → a ≥ 1) ∧
  (∀ x ∈ Set.Icc 1 2,
    (0 < a ∧ a ≤ 1/2 → f a x ≥ Real.log 2 - 1/(2*a)) ∧
    (1/2 < a ∧ a < 1 → f a x ≥ Real.log (1/a) + 1 - 1/a) ∧
    (a ≥ 1 → f a x ≥ 0)) ∧
  (∀ n : ℕ, n > 1 → Real.log n > (Finset.range (n-1)).sum (λ i => 1 / (i + 2))) :=
sorry

end problem_solution_l3478_347870


namespace fifth_power_sum_equality_l3478_347869

theorem fifth_power_sum_equality : ∃ n : ℕ+, 120^5 + 97^5 + 79^5 + 44^5 = n^5 ∧ n = 144 := by
  sorry

end fifth_power_sum_equality_l3478_347869


namespace quadratic_inequality_solution_l3478_347819

theorem quadratic_inequality_solution (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + c > 0 ↔ x < -1 ∨ x > 2) → 
  b + c = -3 := by
sorry

end quadratic_inequality_solution_l3478_347819


namespace fair_coin_probability_l3478_347828

/-- A coin toss with two possible outcomes -/
inductive CoinOutcome
  | heads
  | tails

/-- The probability of a coin toss outcome -/
def probability (outcome : CoinOutcome) : Real :=
  0.5

/-- Theorem stating that the probability of getting heads or tails is 0.5 -/
theorem fair_coin_probability :
  ∀ (outcome : CoinOutcome), probability outcome = 0.5 := by
  sorry

end fair_coin_probability_l3478_347828


namespace f_extrema_l3478_347893

def f (x : ℝ) := x^2 - 2*x + 2

def A₁ : Set ℝ := Set.Icc (-2) 0
def A₂ : Set ℝ := Set.Icc 2 3

theorem f_extrema :
  (∀ x ∈ A₁, f x ≤ 10 ∧ f x ≥ 2) ∧
  (∃ x₁ ∈ A₁, f x₁ = 10) ∧
  (∃ x₂ ∈ A₁, f x₂ = 2) ∧
  (∀ x ∈ A₂, f x ≤ 5 ∧ f x ≥ 2) ∧
  (∃ x₃ ∈ A₂, f x₃ = 5) ∧
  (∃ x₄ ∈ A₂, f x₄ = 2) :=
sorry

end f_extrema_l3478_347893


namespace factor_expression_l3478_347806

theorem factor_expression (a : ℝ) : 37 * a^2 + 111 * a = 37 * a * (a + 3) := by
  sorry

end factor_expression_l3478_347806


namespace arithmetic_sequence_n_l3478_347854

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_n (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 1 = 1/3 →
  a 2 + a 5 = 4 →
  ∃ n : ℕ, a n = 33 →
  ∃ n : ℕ, a n = 33 ∧ n = 50 :=
by sorry

end arithmetic_sequence_n_l3478_347854


namespace a_eq_one_sufficient_not_necessary_l3478_347874

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 1) (a - 2)

-- Statement of the theorem
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → z a = Complex.I * Complex.im (z a)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ z a = Complex.I * Complex.im (z a)) :=
by sorry

end a_eq_one_sufficient_not_necessary_l3478_347874


namespace dawn_savings_l3478_347845

/-- Dawn's financial situation --/
def dawn_finances : Prop :=
  let annual_income : ℝ := 48000
  let monthly_income : ℝ := annual_income / 12
  let tax_rate : ℝ := 0.20
  let variable_expense_rate : ℝ := 0.30
  let stock_investment_rate : ℝ := 0.05
  let retirement_contribution_rate : ℝ := 0.15
  let savings_rate : ℝ := 0.10
  let after_tax_income : ℝ := monthly_income * (1 - tax_rate)
  let variable_expenses : ℝ := after_tax_income * variable_expense_rate
  let stock_investment : ℝ := after_tax_income * stock_investment_rate
  let retirement_contribution : ℝ := after_tax_income * retirement_contribution_rate
  let total_deductions : ℝ := variable_expenses + stock_investment + retirement_contribution
  let remaining_income : ℝ := after_tax_income - total_deductions
  let monthly_savings : ℝ := remaining_income * savings_rate
  monthly_savings = 160

theorem dawn_savings : dawn_finances := by
  sorry

end dawn_savings_l3478_347845


namespace jerry_average_additional_hours_l3478_347895

def tom_total_hours : ℝ := 10
def jerry_daily_differences : List ℝ := [-2, 1, -2, 2, 2, 1]

theorem jerry_average_additional_hours :
  let jerry_total_hours := tom_total_hours + jerry_daily_differences.sum
  let total_difference := jerry_total_hours - tom_total_hours
  let num_days := jerry_daily_differences.length
  total_difference / num_days = 1/3 := by sorry

end jerry_average_additional_hours_l3478_347895


namespace zero_function_equals_derivative_l3478_347834

theorem zero_function_equals_derivative : ∃ f : ℝ → ℝ, ∀ x, f x = 0 ∧ (deriv f) x = f x := by
  sorry

end zero_function_equals_derivative_l3478_347834


namespace ones_divisibility_l3478_347848

theorem ones_divisibility (d : ℕ) (h1 : d > 0) (h2 : ¬ 2 ∣ d) (h3 : ¬ 5 ∣ d) :
  ∃ n : ℕ, d ∣ ((10^n - 1) / 9) :=
sorry

end ones_divisibility_l3478_347848


namespace physics_group_size_l3478_347846

theorem physics_group_size (total : ℕ) (math_ratio physics_ratio chem_ratio : ℕ) : 
  total = 135 → 
  math_ratio = 6 →
  physics_ratio = 5 →
  chem_ratio = 4 →
  (physics_ratio : ℚ) / (math_ratio + physics_ratio + chem_ratio : ℚ) * total = 45 :=
by
  sorry

end physics_group_size_l3478_347846


namespace alyssa_future_games_l3478_347817

/-- The number of soccer games Alyssa attended this year -/
def games_this_year : ℕ := 11

/-- The number of soccer games Alyssa missed this year -/
def games_missed_this_year : ℕ := 12

/-- The number of soccer games Alyssa attended last year -/
def games_last_year : ℕ := 13

/-- The total number of soccer games Alyssa will attend over three years -/
def total_games : ℕ := 39

/-- The number of games Alyssa plans to attend next year -/
def games_next_year : ℕ := total_games - (games_this_year + games_last_year)

theorem alyssa_future_games : games_next_year = 15 := by
  sorry

end alyssa_future_games_l3478_347817


namespace tan_45_deg_eq_one_l3478_347824

/-- Tangent of 45 degrees is 1 -/
theorem tan_45_deg_eq_one :
  Real.tan (π / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l3478_347824


namespace robie_initial_cards_robie_initial_cards_proof_l3478_347825

theorem robie_initial_cards (cards_per_box : ℕ) (loose_cards : ℕ) (boxes_given : ℕ) 
  (boxes_returned : ℕ) (current_boxes : ℕ) (cards_bought : ℕ) (cards_traded : ℕ) : ℕ :=
  let initial_boxes := current_boxes - boxes_returned + boxes_given
  let boxed_cards := initial_boxes * cards_per_box
  let total_cards := boxed_cards + loose_cards
  let initial_cards := total_cards - cards_bought
  initial_cards

theorem robie_initial_cards_proof :
  robie_initial_cards 30 18 8 2 15 21 12 = 627 := by
  sorry

end robie_initial_cards_robie_initial_cards_proof_l3478_347825


namespace negative_sqrt_two_squared_l3478_347833

theorem negative_sqrt_two_squared : (-Real.sqrt 2)^2 = 2 := by
  sorry

end negative_sqrt_two_squared_l3478_347833


namespace function_range_theorem_l3478_347801

-- Define the function types
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def OddFunction (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Define the main theorem
theorem function_range_theorem (f g : ℝ → ℝ) (a : ℝ) :
  EvenFunction f →
  OddFunction g →
  (∀ x, f x + g x = 2^(x + 1)) →
  (∀ x, a * f (2*x) + g x ≤ 25/8 + a * f (2*0) + g 0) →
  (∀ x, a * f (2*x) + g x ≥ a * f (2*0) + g 0 - 25/8) →
  -2 ≤ a ∧ a ≤ 13/18 :=
by sorry

end function_range_theorem_l3478_347801


namespace quadratic_sum_l3478_347885

theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ), 
  (-6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) ∧ (a + b + c = 261) := by
  sorry

end quadratic_sum_l3478_347885


namespace negation_of_universal_proposition_l3478_347852

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ x) ↔ (∃ x : ℝ, x^2 + 1 < x) :=
by sorry

end negation_of_universal_proposition_l3478_347852


namespace roots_sum_of_powers_l3478_347841

theorem roots_sum_of_powers (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^4 + p^3*q^2 + p^2*q^3 + q^4 = 241 := by
  sorry

end roots_sum_of_powers_l3478_347841


namespace tv_diagonal_problem_l3478_347805

theorem tv_diagonal_problem (larger_diagonal smaller_diagonal : ℝ) :
  larger_diagonal = 24 →
  larger_diagonal ^ 2 / 2 - smaller_diagonal ^ 2 / 2 = 143.5 →
  smaller_diagonal = 17 := by
sorry

end tv_diagonal_problem_l3478_347805


namespace odd_ceiling_factorial_fraction_l3478_347815

theorem odd_ceiling_factorial_fraction (n : ℕ) (h1 : n > 6) (h2 : Nat.Prime (n + 1)) :
  Odd (⌈(Nat.factorial (n - 1) : ℚ) / (n * (n + 1))⌉) := by
  sorry

end odd_ceiling_factorial_fraction_l3478_347815


namespace john_pushups_l3478_347877

def zachary_pushups : ℕ := 51
def david_pushups_difference : ℕ := 22
def john_pushups_difference : ℕ := 4

theorem john_pushups : 
  zachary_pushups + david_pushups_difference - john_pushups_difference = 69 :=
by
  sorry

end john_pushups_l3478_347877


namespace two_A_minus_four_B_y_value_when_independent_of_x_l3478_347884

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y - 2 * x
def B (x y : ℝ) : ℝ := x^2 - x * y + 1

-- Theorem 1: 2A - 4B = 10xy - 4x - 4
theorem two_A_minus_four_B (x y : ℝ) :
  2 * A x y - 4 * B x y = 10 * x * y - 4 * x - 4 := by sorry

-- Theorem 2: When 2A - 4B is independent of x, y = 2/5
theorem y_value_when_independent_of_x (y : ℝ) :
  (∀ x : ℝ, 2 * A x y - 4 * B x y = 10 * x * y - 4 * x - 4) →
  y = 2 / 5 := by sorry

end two_A_minus_four_B_y_value_when_independent_of_x_l3478_347884


namespace invisible_dots_count_l3478_347892

-- Define the number of dice
def num_dice : ℕ := 4

-- Define the numbers on a single die
def die_numbers : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define the visible numbers
def visible_numbers : List ℕ := [2, 2, 3, 4, 4, 5, 6, 6]

-- Theorem to prove
theorem invisible_dots_count :
  (num_dice * (die_numbers.sum)) - (visible_numbers.sum) = 52 := by
  sorry

end invisible_dots_count_l3478_347892


namespace jason_stored_23_bales_l3478_347839

/-- The number of bales Jason stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Theorem: Jason stored 23 bales in the barn -/
theorem jason_stored_23_bales (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 73) 
  (h2 : final_bales = 96) : 
  bales_stored initial_bales final_bales = 23 := by
  sorry

end jason_stored_23_bales_l3478_347839


namespace g_53_l3478_347891

/-- A function satisfying g(xy) = yg(x) for all real x and y, with g(1) = 15 -/
def g : ℝ → ℝ :=
  sorry

/-- The functional equation for g -/
axiom g_eq (x y : ℝ) : g (x * y) = y * g x

/-- The value of g at 1 -/
axiom g_one : g 1 = 15

/-- The theorem to be proved -/
theorem g_53 : g 53 = 795 :=
  sorry

end g_53_l3478_347891


namespace percentage_problem_l3478_347840

theorem percentage_problem (p : ℝ) : 
  (p / 100) * 180 - (1 / 3) * ((p / 100) * 180) = 18 ↔ p = 15 := by
  sorry

end percentage_problem_l3478_347840


namespace max_value_sqrt_sum_l3478_347876

theorem max_value_sqrt_sum (x : ℝ) (h : 0 ≤ x ∧ x ≤ 18) :
  Real.sqrt (35 - x) + Real.sqrt x + Real.sqrt (18 - x) ≤ Real.sqrt 35 + Real.sqrt 18 := by
  sorry

end max_value_sqrt_sum_l3478_347876


namespace union_of_intervals_l3478_347889

open Set

theorem union_of_intervals (A B : Set ℝ) : 
  A = {x : ℝ | 3 < x ∧ x ≤ 7} →
  B = {x : ℝ | 4 < x ∧ x ≤ 10} →
  A ∪ B = {x : ℝ | 3 < x ∧ x ≤ 10} :=
by sorry

end union_of_intervals_l3478_347889


namespace imaginary_part_of_z_l3478_347835

theorem imaginary_part_of_z (z : ℂ) : 
  z * (1 + Complex.I) * Complex.I^3 / (1 - Complex.I) = 1 - Complex.I →
  z.im = -1 := by
sorry

end imaginary_part_of_z_l3478_347835


namespace percentile_rank_between_90_and_91_l3478_347836

/-- Represents a student's rank in a class -/
structure StudentRank where
  total_students : ℕ
  rank : ℕ
  h_rank_valid : rank ≤ total_students

/-- Calculates the percentile rank of a student -/
def percentile_rank (sr : StudentRank) : ℚ :=
  (sr.total_students - sr.rank : ℚ) / sr.total_students * 100

/-- Theorem stating that a student ranking 5th in a class of 48 has a percentile rank between 90 and 91 -/
theorem percentile_rank_between_90_and_91 (sr : StudentRank) 
  (h_total : sr.total_students = 48) 
  (h_rank : sr.rank = 5) : 
  90 < percentile_rank sr ∧ percentile_rank sr < 91 := by
  sorry

#eval percentile_rank ⟨48, 5, by norm_num⟩

end percentile_rank_between_90_and_91_l3478_347836


namespace inequality_problem_l3478_347816

theorem inequality_problem (s x y : ℝ) (h1 : s > 0) (h2 : x^2 + y^2 ≠ 0) (h3 : x * s^2 < y * s^2) :
  ¬(-x^2 < -y^2) ∧ ¬(-x^2 < y^2) ∧ ¬(x^2 < -y^2) ∧ ¬(x^2 > y^2) := by
  sorry

end inequality_problem_l3478_347816


namespace monica_wednesday_study_time_l3478_347886

/-- Represents the study schedule of Monica over five days -/
structure StudySchedule where
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  weekend : ℝ
  total : ℝ

/-- The study schedule satisfies the given conditions -/
def validSchedule (s : StudySchedule) : Prop :=
  s.thursday = 3 * s.wednesday ∧
  s.friday = 1.5 * s.wednesday ∧
  s.weekend = 5.5 * s.wednesday ∧
  s.total = 22 ∧
  s.total = s.wednesday + s.thursday + s.friday + s.weekend

/-- Theorem stating that Monica studied 2 hours on Wednesday -/
theorem monica_wednesday_study_time (s : StudySchedule) 
  (h : validSchedule s) : s.wednesday = 2 := by
  sorry

end monica_wednesday_study_time_l3478_347886


namespace biff_voting_percentage_l3478_347812

theorem biff_voting_percentage (total_polled : ℕ) (marty_votes : ℕ) (undecided_percent : ℚ) :
  total_polled = 200 →
  marty_votes = 94 →
  undecided_percent = 8 / 100 →
  (↑(total_polled - marty_votes - (undecided_percent * ↑total_polled).num) / ↑total_polled : ℚ) = 45 / 100 := by
  sorry

end biff_voting_percentage_l3478_347812


namespace permutations_of_six_distinct_objects_l3478_347881

theorem permutations_of_six_distinct_objects : Nat.factorial 6 = 720 := by
  sorry

end permutations_of_six_distinct_objects_l3478_347881


namespace angle_sum_is_pi_over_two_l3478_347894

theorem angle_sum_is_pi_over_two (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end angle_sum_is_pi_over_two_l3478_347894


namespace tony_puzzle_time_l3478_347875

/-- The total time Tony spent solving puzzles -/
def total_puzzle_time (warm_up_time : ℝ) : ℝ :=
  let challenging_puzzle_time := 3 * warm_up_time
  let set_puzzle1_time := 0.5 * warm_up_time
  let set_puzzle2_time := 2 * set_puzzle1_time
  let set_puzzle3_time := set_puzzle1_time + set_puzzle2_time + 2
  let set_puzzle4_time := 1.5 * set_puzzle3_time
  warm_up_time + 2 * challenging_puzzle_time + set_puzzle1_time + set_puzzle2_time + set_puzzle3_time + set_puzzle4_time

/-- Theorem stating that Tony spent 127.5 minutes solving puzzles -/
theorem tony_puzzle_time : total_puzzle_time 10 = 127.5 := by
  sorry

end tony_puzzle_time_l3478_347875


namespace cos_alpha_minus_pi_sixth_l3478_347823

theorem cos_alpha_minus_pi_sixth (α : ℝ) (h : Real.sin (α + π/3) = 4/5) : 
  Real.cos (α - π/6) = 4/5 := by
sorry

end cos_alpha_minus_pi_sixth_l3478_347823


namespace number_of_children_l3478_347883

theorem number_of_children (total : ℕ) (adults : ℕ) (children : ℕ) : 
  total = 42 → 
  children = 2 * adults → 
  total = adults + children →
  children = 28 := by
sorry

end number_of_children_l3478_347883


namespace rational_sum_product_equality_l3478_347856

theorem rational_sum_product_equality : ∃ (a b : ℚ), a ≠ b ∧ a + b = a * b ∧ a = 3/2 ∧ b = 3 := by
  sorry

end rational_sum_product_equality_l3478_347856


namespace triangle_perimeter_not_88_l3478_347861

theorem triangle_perimeter_not_88 (a b x : ℝ) (h1 : a = 18) (h2 : b = 25) (h3 : a + b > x) (h4 : a + x > b) (h5 : b + x > a) : a + b + x ≠ 88 :=
sorry

end triangle_perimeter_not_88_l3478_347861


namespace fuji_ratio_l3478_347857

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  totalTrees : ℕ
  pureFuji : ℕ
  pureGala : ℕ
  crossPollinated : ℕ

/-- The conditions of the orchard as described in the problem -/
def orchardConditions (o : Orchard) : Prop :=
  o.crossPollinated = o.totalTrees / 10 ∧
  o.pureFuji + o.crossPollinated = 204 ∧
  o.pureGala = 36 ∧
  o.totalTrees = o.pureFuji + o.pureGala + o.crossPollinated

/-- The theorem stating the ratio of pure Fuji trees to all trees -/
theorem fuji_ratio (o : Orchard) (h : orchardConditions o) :
  3 * o.totalTrees = 4 * o.pureFuji := by
  sorry


end fuji_ratio_l3478_347857


namespace ham_bread_percentage_l3478_347800

def bread_cost : ℕ := 50
def ham_cost : ℕ := 150
def cake_cost : ℕ := 200

def total_cost : ℕ := bread_cost + ham_cost + cake_cost
def ham_bread_cost : ℕ := bread_cost + ham_cost

theorem ham_bread_percentage :
  (ham_bread_cost : ℚ) / (total_cost : ℚ) * 100 = 50 := by
  sorry

end ham_bread_percentage_l3478_347800


namespace teacher_age_l3478_347850

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) : 
  num_students = 19 →
  student_avg_age = 20 →
  new_avg_age = student_avg_age + 1 →
  (num_students + 1) * new_avg_age - num_students * student_avg_age = 40 :=
by
  sorry

end teacher_age_l3478_347850


namespace descent_time_specific_garage_l3478_347855

/-- Represents a parking garage with specified characteristics -/
structure ParkingGarage where
  floors : ℕ
  gateInterval : ℕ
  gateTime : ℕ
  floorDistance : ℕ
  drivingSpeed : ℕ

/-- Calculates the total time to descend the parking garage -/
def descentTime (garage : ParkingGarage) : ℕ :=
  let drivingTime := (garage.floors - 1) * (garage.floorDistance / garage.drivingSpeed)
  let gateCount := (garage.floors - 1) / garage.gateInterval
  let gateTime := gateCount * garage.gateTime
  drivingTime + gateTime

/-- The theorem stating the total descent time for the specific garage -/
theorem descent_time_specific_garage :
  let garage : ParkingGarage := {
    floors := 12,
    gateInterval := 3,
    gateTime := 120,
    floorDistance := 800,
    drivingSpeed := 10
  }
  descentTime garage = 1240 := by sorry

end descent_time_specific_garage_l3478_347855


namespace parabola_coefficients_l3478_347820

def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_coefficients :
  ∃ (a b c : ℝ),
    (∀ x, parabola a b c x = parabola a b c (4 - x)) ∧  -- Vertical axis of symmetry at x = 2
    (parabola a b c 2 = 3) ∧                            -- Vertex at (2, 3)
    (parabola a b c 0 = 1) ∧                            -- Passes through (0, 1)
    (a = -1/2 ∧ b = 2 ∧ c = 1) :=
by
  sorry

end parabola_coefficients_l3478_347820


namespace problem_statement_l3478_347809

theorem problem_statement (a b c d e : ℕ+) : 
  a * b * c * d * e = 362880 →
  a * b + a + b = 728 →
  b * c + b + c = 342 →
  c * d + c + d = 464 →
  d * e + d + e = 780 →
  (a : ℤ) - (e : ℤ) = 172 := by
  sorry

end problem_statement_l3478_347809


namespace list_price_is_40_l3478_347888

/-- The list price of the item. -/
def list_price : ℝ := 40

/-- Alice's selling price. -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price. -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Alice's commission rate. -/
def alice_rate : ℝ := 0.15

/-- Bob's commission rate. -/
def bob_rate : ℝ := 0.25

/-- Alice's commission. -/
def alice_commission (x : ℝ) : ℝ := alice_rate * alice_price x

/-- Bob's commission. -/
def bob_commission (x : ℝ) : ℝ := bob_rate * bob_price x

theorem list_price_is_40 :
  alice_commission list_price = bob_commission list_price ∧
  list_price = 40 := by
  sorry

end list_price_is_40_l3478_347888


namespace pool_capacity_l3478_347865

theorem pool_capacity (C : ℝ) 
  (h1 : 0.8 * C = 0.5 * C + 300) 
  (h2 : 300 = 0.3 * C) : 
  C = 1000 := by
sorry

end pool_capacity_l3478_347865


namespace copy_pages_proof_l3478_347872

/-- Given a cost per page in cents and a budget in dollars, 
    calculates the maximum number of pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (budget_dollars : ℕ) : ℕ :=
  (budget_dollars * 100) / cost_per_page

/-- Proves that with a cost of 3 cents per page and a budget of $15,
    the maximum number of pages that can be copied is 500. -/
theorem copy_pages_proof :
  max_pages_copied 3 15 = 500 := by
  sorry

#eval max_pages_copied 3 15

end copy_pages_proof_l3478_347872


namespace overlap_length_l3478_347866

/-- Given red line segments with equal lengths and overlaps, prove the length of each overlap. -/
theorem overlap_length (total_length : ℝ) (edge_to_edge : ℝ) (num_overlaps : ℕ) 
  (h1 : total_length = 98)
  (h2 : edge_to_edge = 83)
  (h3 : num_overlaps = 6)
  (h4 : total_length - edge_to_edge = num_overlaps * (total_length - edge_to_edge) / num_overlaps) :
  (total_length - edge_to_edge) / num_overlaps = 2.5 := by
  sorry

end overlap_length_l3478_347866


namespace shopping_remaining_amount_l3478_347860

def initial_amount : ℚ := 74
def sweater_cost : ℚ := 9
def tshirt_cost : ℚ := 11
def shoes_cost : ℚ := 30
def refund_percentage : ℚ := 90 / 100

theorem shopping_remaining_amount :
  initial_amount - (sweater_cost + tshirt_cost + shoes_cost * (1 - refund_percentage)) = 51 := by
  sorry

end shopping_remaining_amount_l3478_347860


namespace N_divisible_by_2027_l3478_347849

theorem N_divisible_by_2027 : ∃ k : ℤ, (7 * 9 * 13 + 2020 * 2018 * 2014) = 2027 * k := by
  sorry

end N_divisible_by_2027_l3478_347849


namespace infinite_sum_equals_two_l3478_347814

theorem infinite_sum_equals_two :
  (∑' n : ℕ, (4 * n - 2) / (3 : ℝ)^n) = 2 := by sorry

end infinite_sum_equals_two_l3478_347814


namespace circle_center_and_radius_l3478_347879

/-- Given a circle with equation x^2 + y^2 - 2x = 0, its center is (1,0) and its radius is 1 -/
theorem circle_center_and_radius :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 2*x = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, 0) ∧ radius = 1 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l3478_347879
