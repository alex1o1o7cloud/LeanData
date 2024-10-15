import Mathlib

namespace NUMINAMATH_CALUDE_residue_negative_1237_mod_37_l3653_365309

theorem residue_negative_1237_mod_37 : ∃ k : ℤ, -1237 = 37 * k + 21 ∧ (0 ≤ 21 ∧ 21 < 37) := by sorry

end NUMINAMATH_CALUDE_residue_negative_1237_mod_37_l3653_365309


namespace NUMINAMATH_CALUDE_rational_fraction_representation_l3653_365390

def is_rational (f : ℕ+ → ℚ) : Prop :=
  ∀ x : ℕ+, ∃ p q : ℤ, f x = p / q ∧ q ≠ 0

theorem rational_fraction_representation
  (a b : ℚ) (h : is_rational (λ x : ℕ+ => (a * x + b) / x)) :
  ∃ A B C : ℤ, ∀ x : ℕ+, (a * x + b) / x = (A * x + B) / (C * x) :=
sorry

end NUMINAMATH_CALUDE_rational_fraction_representation_l3653_365390


namespace NUMINAMATH_CALUDE_average_score_two_classes_l3653_365359

theorem average_score_two_classes (n1 n2 : ℕ) (s1 s2 : ℝ) :
  n1 > 0 → n2 > 0 →
  s1 = 80 → s2 = 70 →
  n1 = 20 → n2 = 30 →
  (n1 * s1 + n2 * s2) / (n1 + n2 : ℝ) = 74 := by
  sorry

end NUMINAMATH_CALUDE_average_score_two_classes_l3653_365359


namespace NUMINAMATH_CALUDE_linear_equation_solve_l3653_365397

theorem linear_equation_solve (x y : ℝ) :
  2 * x - 7 * y = 5 → y = (2 * x - 5) / 7 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solve_l3653_365397


namespace NUMINAMATH_CALUDE_paint_cost_rectangular_floor_l3653_365364

/-- The cost to paint a rectangular floor given its length and the ratio of length to breadth -/
theorem paint_cost_rectangular_floor 
  (length : ℝ) 
  (length_to_breadth_ratio : ℝ) 
  (paint_rate : ℝ) 
  (h1 : length = 15.491933384829668)
  (h2 : length_to_breadth_ratio = 3)
  (h3 : paint_rate = 3) : 
  ⌊length * (length / length_to_breadth_ratio) * paint_rate⌋ = 240 := by
sorry

end NUMINAMATH_CALUDE_paint_cost_rectangular_floor_l3653_365364


namespace NUMINAMATH_CALUDE_problem_1_l3653_365325

theorem problem_1 : 
  Real.sqrt 48 / Real.sqrt 3 - 4 * Real.sqrt (1/5) * Real.sqrt 30 + (2 * Real.sqrt 2 + Real.sqrt 3)^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3653_365325


namespace NUMINAMATH_CALUDE_ellipse_max_ratio_l3653_365313

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    where a > b > 0, prove that the maximum value of |FA|/|OH| is 1/4, 
    where F is the right focus, A is the right vertex, O is the center, 
    and H is the intersection of the right directrix with the x-axis. -/
theorem ellipse_max_ratio (a b : ℝ) (h : a > b ∧ b > 0) : 
  let e := Real.sqrt (1 - b^2 / a^2)  -- eccentricity
  ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    (∀ (x' y' : ℝ), x'^2 / a^2 + y'^2 / b^2 = 1 → 
      (a - a * e) / (a^2 / (a * e)) ≤ 1/4) ∧
    (a - a * e) / (a^2 / (a * e)) = 1/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_max_ratio_l3653_365313


namespace NUMINAMATH_CALUDE_regular_polygon_with_20_degree_exterior_angle_l3653_365384

theorem regular_polygon_with_20_degree_exterior_angle (n : ℕ) : 
  n > 2 → (360 : ℝ) / n = 20 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_20_degree_exterior_angle_l3653_365384


namespace NUMINAMATH_CALUDE_max_plus_min_of_f_l3653_365334

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem max_plus_min_of_f : 
  ∃ (m n : ℝ), (∀ x, f x ≤ m) ∧ (∃ x₁, f x₁ = m) ∧ 
               (∀ x, n ≤ f x) ∧ (∃ x₂, f x₂ = n) ∧ 
               m + n = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_plus_min_of_f_l3653_365334


namespace NUMINAMATH_CALUDE_max_value_a_sqrt_1_plus_b_sq_l3653_365336

theorem max_value_a_sqrt_1_plus_b_sq (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (heq : a^2 / 2 + b^2 = 4) :
  ∃ (max : ℝ), max = (5 * Real.sqrt 2) / 2 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → x^2 / 2 + y^2 = 4 → 
  x * Real.sqrt (1 + y^2) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_sqrt_1_plus_b_sq_l3653_365336


namespace NUMINAMATH_CALUDE_log_xy_value_l3653_365337

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x^3 * y^2) = 2) (h2 : Real.log (x^2 * y^3) = 2) : 
  Real.log (x * y) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_log_xy_value_l3653_365337


namespace NUMINAMATH_CALUDE_professional_doctors_percentage_l3653_365360

theorem professional_doctors_percentage 
  (leaders_percent : ℝ)
  (nurses_percent : ℝ)
  (h1 : leaders_percent = 4)
  (h2 : nurses_percent = 56)
  (h3 : ∃ (doctors_percent psychologists_percent : ℝ), 
    leaders_percent + nurses_percent + doctors_percent + psychologists_percent = 100) :
  ∃ (doctors_percent : ℝ), doctors_percent = 40 := by
  sorry

end NUMINAMATH_CALUDE_professional_doctors_percentage_l3653_365360


namespace NUMINAMATH_CALUDE_function_equation_solution_l3653_365329

theorem function_equation_solution (f : ℤ → ℤ) :
  (∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))) →
  (∃ c : ℤ, ∀ x : ℤ, f x = 0 ∨ f x = 2 * x + c) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l3653_365329


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l3653_365372

/-- Calculates the profit percentage for a cricket bat sale --/
theorem cricket_bat_profit_percentage 
  (selling_price : ℝ) 
  (initial_profit : ℝ) 
  (tax_rate : ℝ) 
  (discount_rate : ℝ) 
  (h1 : selling_price = 850)
  (h2 : initial_profit = 255)
  (h3 : tax_rate = 0.07)
  (h4 : discount_rate = 0.05) : 
  ∃ (profit_percentage : ℝ), abs (profit_percentage - 25.71) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l3653_365372


namespace NUMINAMATH_CALUDE_profit_percentage_l3653_365324

theorem profit_percentage (selling_price cost_price : ℝ) 
  (h : cost_price = 0.92 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100 / 92 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l3653_365324


namespace NUMINAMATH_CALUDE_expression_evaluation_l3653_365319

theorem expression_evaluation : (-5)^5 / 5^3 + 3^4 - 6^1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3653_365319


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3653_365308

theorem quadratic_equation_solution :
  let a : ℝ := 5
  let b : ℝ := -2 * Real.sqrt 15
  let c : ℝ := -2
  let x₁ : ℝ := -1 + Real.sqrt 15 / 5
  let x₂ : ℝ := 1 + Real.sqrt 15 / 5
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) ∧
  (a * x₁^2 + b * x₁ + c = 0) ∧
  (a * x₂^2 + b * x₂ + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3653_365308


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3653_365305

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - p * x + q = 0 → 
    (∃ r₁ r₂ : ℝ, r₁ + r₂ = 9 ∧ r₁ * r₂ = 15 ∧ 
      (3 * r₁^2 - p * r₁ + q = 0) ∧ (3 * r₂^2 - p * r₂ + q = 0))) →
  p + q = 72 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3653_365305


namespace NUMINAMATH_CALUDE_sin_2theta_plus_pi_4_l3653_365383

theorem sin_2theta_plus_pi_4 (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ + Real.pi / 4) = Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_plus_pi_4_l3653_365383


namespace NUMINAMATH_CALUDE_other_solution_of_quadratic_equation_l3653_365314

theorem other_solution_of_quadratic_equation :
  let equation := fun (x : ℚ) => 72 * x^2 + 43 = 113 * x - 12
  equation (3/8) → ∃ x : ℚ, x ≠ 3/8 ∧ equation x ∧ x = 43/36 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_of_quadratic_equation_l3653_365314


namespace NUMINAMATH_CALUDE_pyramid_theorem_l3653_365348

structure Pyramid where
  S₁ : ℝ  -- Area of face ABD
  S₂ : ℝ  -- Area of face BCD
  S₃ : ℝ  -- Area of face CAD
  Q : ℝ   -- Area of face ABC
  α : ℝ   -- Dihedral angle at edge AB
  β : ℝ   -- Dihedral angle at edge BC
  γ : ℝ   -- Dihedral angle at edge AC
  h₁ : S₁ > 0
  h₂ : S₂ > 0
  h₃ : S₃ > 0
  h₄ : Q > 0
  h₅ : 0 < α ∧ α < π
  h₆ : 0 < β ∧ β < π
  h₇ : 0 < γ ∧ γ < π
  h₈ : Real.cos α = S₁ / Q
  h₉ : Real.cos β = S₂ / Q
  h₁₀ : Real.cos γ = S₃ / Q

theorem pyramid_theorem (p : Pyramid) : 
  p.S₁^2 + p.S₂^2 + p.S₃^2 = p.Q^2 ∧ 
  Real.cos (2 * p.α) + Real.cos (2 * p.β) + Real.cos (2 * p.γ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_theorem_l3653_365348


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3653_365382

theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  (1 + 1 / (x + 1)) / ((x^2 + 4*x + 4) / (x + 1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3653_365382


namespace NUMINAMATH_CALUDE_det_specific_matrix_l3653_365393

theorem det_specific_matrix : 
  Matrix.det !![2, 0, 4; 3, -1, 5; 1, 2, 3] = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_specific_matrix_l3653_365393


namespace NUMINAMATH_CALUDE_expand_expression_l3653_365322

theorem expand_expression (x y z : ℝ) : 
  (x + 10 + y) * (2 * z + 10) = 2 * x * z + 2 * y * z + 10 * x + 10 * y + 20 * z + 100 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3653_365322


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l3653_365343

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (h : x > 0) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l3653_365343


namespace NUMINAMATH_CALUDE_boys_cannot_score_double_l3653_365386

/-- Represents a player in the chess tournament -/
inductive Player
| Boy
| Girl

/-- Represents the outcome of a chess game -/
inductive GameResult
| Win
| Draw
| Loss

/-- The number of players in the tournament -/
def numPlayers : Nat := 6

/-- The number of boys in the tournament -/
def numBoys : Nat := 2

/-- The number of girls in the tournament -/
def numGirls : Nat := 4

/-- The number of games each player plays -/
def gamesPerPlayer : Nat := numPlayers - 1

/-- The total number of games in the tournament -/
def totalGames : Nat := (numPlayers * gamesPerPlayer) / 2

/-- The points awarded for each game result -/
def pointsForResult (result : GameResult) : Rat :=
  match result with
  | GameResult.Win => 1
  | GameResult.Draw => 1/2
  | GameResult.Loss => 0

/-- A function representing the total score of a group of players -/
def groupScore (players : List Player) (results : List (Player × Player × GameResult)) : Rat :=
  sorry

/-- The main theorem stating that boys cannot score twice as many points as girls -/
theorem boys_cannot_score_double :
  ¬∃ (results : List (Player × Player × GameResult)),
    (results.length = totalGames) ∧
    (groupScore [Player.Boy, Player.Boy] results = 2 * groupScore [Player.Girl, Player.Girl, Player.Girl, Player.Girl] results) :=
  sorry

end NUMINAMATH_CALUDE_boys_cannot_score_double_l3653_365386


namespace NUMINAMATH_CALUDE_a_squared_minus_b_squared_eq_zero_l3653_365302

def first_seven_multiples_of_seven : List ℕ := [7, 14, 21, 28, 35, 42, 49]

def first_three_multiples_of_fourteen : List ℕ := [14, 28, 42]

def a : ℚ := (first_seven_multiples_of_seven.sum : ℚ) / 7

def b : ℕ := first_three_multiples_of_fourteen[1]

theorem a_squared_minus_b_squared_eq_zero : a^2 - (b^2 : ℚ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_squared_minus_b_squared_eq_zero_l3653_365302


namespace NUMINAMATH_CALUDE_tangent_line_to_ellipse_l3653_365388

/-- Represents an ellipse with semi-major axis √8 and semi-minor axis √2 -/
def Ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

/-- Represents a point on the ellipse -/
def PointOnEllipse : ℝ × ℝ := (2, 1)

/-- Represents the equation of a line -/
def Line (x y : ℝ) : Prop := x / 4 + y / 2 = 1

theorem tangent_line_to_ellipse :
  Line PointOnEllipse.1 PointOnEllipse.2 ∧
  Ellipse PointOnEllipse.1 PointOnEllipse.2 ∧
  ∀ (x y : ℝ), Ellipse x y → Line x y ∨ (x = PointOnEllipse.1 ∧ y = PointOnEllipse.2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_ellipse_l3653_365388


namespace NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l3653_365371

theorem equidistant_point_on_x_axis : ∃ x : ℝ, 
  (x^2 + 6*x + 9 = x^2 + 25) ∧ (x = 8/3) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l3653_365371


namespace NUMINAMATH_CALUDE_equality_conditions_l3653_365366

theorem equality_conditions (a b c : ℝ) : 
  ((a + (a * b * c) / (a - b * c + b)) / (b + (a * b * c) / (a - a * c + b)) = 
   (a - (a * b) / (a + 2 * b)) / (b - (a * b) / (2 * a + b)) ∧
   (a - (a * b) / (a + 2 * b)) / (b - (a * b) / (2 * a + b)) = 
   ((2 * a * b) / (a - b) + a) / ((2 * a * b) / (a - b) - b) ∧
   ((2 * a * b) / (a - b) + a) / ((2 * a * b) / (a - b) - b) = a / b) ↔
  (a = 0 ∧ b ≠ 0 ∧ c ≠ 1) :=
by sorry


end NUMINAMATH_CALUDE_equality_conditions_l3653_365366


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l3653_365369

theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 6*x - 7 = 0 ↔ (x - 3)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l3653_365369


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3653_365385

theorem necessary_but_not_sufficient (x : ℝ) :
  ((x - 5) / (2 - x) > 0 → abs (x - 1) < 4) ∧
  (∃ y : ℝ, abs (y - 1) < 4 ∧ ¬((y - 5) / (2 - y) > 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3653_365385


namespace NUMINAMATH_CALUDE_painting_area_l3653_365303

/-- The area of a rectangular painting inside a border -/
theorem painting_area (outer_width outer_height border_width : ℕ) : 
  outer_width = 100 ∧ outer_height = 150 ∧ border_width = 15 →
  (outer_width - 2 * border_width) * (outer_height - 2 * border_width) = 8400 :=
by sorry

end NUMINAMATH_CALUDE_painting_area_l3653_365303


namespace NUMINAMATH_CALUDE_vector_equation_l3653_365344

def a : ℝ × ℝ := (3, -2)
def b : ℝ × ℝ := (-2, 1)
def c : ℝ × ℝ := (-12, 7)

theorem vector_equation (m n : ℝ) (h : c = m • a + n • b) : m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_l3653_365344


namespace NUMINAMATH_CALUDE_min_value_2sin_x_l3653_365346

theorem min_value_2sin_x (x : Real) (h : π/3 ≤ x ∧ x ≤ 5*π/6) : 
  ∃ (y : Real), y = 2 * Real.sin x ∧ y ≥ 1 ∧ ∀ z, (∃ t, π/3 ≤ t ∧ t ≤ 5*π/6 ∧ z = 2 * Real.sin t) → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_2sin_x_l3653_365346


namespace NUMINAMATH_CALUDE_first_part_segments_second_part_segments_l3653_365315

/-- Number of segments after cutting loops in a Chinese knot --/
def segments_after_cutting (loops : ℕ) (wings : ℕ := 1) : ℕ :=
  (loops * 2 * wings + wings) / wings

/-- Theorem for the first part of the problem --/
theorem first_part_segments : segments_after_cutting 5 = 6 := by sorry

/-- Theorem for the second part of the problem --/
theorem second_part_segments : segments_after_cutting 7 2 = 15 := by sorry

end NUMINAMATH_CALUDE_first_part_segments_second_part_segments_l3653_365315


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3653_365387

/-- Proves that a train with given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : Real) 
  (train_speed_kmh : Real) 
  (bridge_length : Real) : 
  train_length = 170 → 
  train_speed_kmh = 45 → 
  bridge_length = 205 → 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry


end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3653_365387


namespace NUMINAMATH_CALUDE_james_two_point_shots_l3653_365376

/-- Represents the number of 2-point shots scored by James -/
def two_point_shots : ℕ := sorry

/-- Represents the number of 3-point shots scored by James -/
def three_point_shots : ℕ := 13

/-- Represents the total points scored by James -/
def total_points : ℕ := 79

/-- Theorem stating that James scored 20 two-point shots -/
theorem james_two_point_shots : 
  two_point_shots = 20 ∧ 
  2 * two_point_shots + 3 * three_point_shots = total_points := by
  sorry

end NUMINAMATH_CALUDE_james_two_point_shots_l3653_365376


namespace NUMINAMATH_CALUDE_student_sister_weight_ratio_l3653_365363

theorem student_sister_weight_ratio : 
  ∀ (student_weight sister_weight : ℝ),
    student_weight = 90 →
    student_weight + sister_weight = 132 →
    (student_weight - 6) / sister_weight = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_student_sister_weight_ratio_l3653_365363


namespace NUMINAMATH_CALUDE_sarah_wide_reflections_correct_l3653_365310

/-- The number of times Sarah sees her reflection in the room with tall mirrors -/
def sarah_tall_reflections : ℕ := 10

/-- The number of times Ellie sees her reflection in the room with tall mirrors -/
def ellie_tall_reflections : ℕ := 6

/-- The number of times Ellie sees her reflection in the room with wide mirrors -/
def ellie_wide_reflections : ℕ := 3

/-- The number of times they both passed through the room with tall mirrors -/
def tall_mirror_passes : ℕ := 3

/-- The number of times they both passed through the room with wide mirrors -/
def wide_mirror_passes : ℕ := 5

/-- The total number of reflections seen by both Sarah and Ellie -/
def total_reflections : ℕ := 88

/-- The number of times Sarah sees her reflection in the room with wide mirrors -/
def sarah_wide_reflections : ℕ := 5

theorem sarah_wide_reflections_correct :
  sarah_tall_reflections * tall_mirror_passes +
  sarah_wide_reflections * wide_mirror_passes +
  ellie_tall_reflections * tall_mirror_passes +
  ellie_wide_reflections * wide_mirror_passes = total_reflections :=
by sorry

end NUMINAMATH_CALUDE_sarah_wide_reflections_correct_l3653_365310


namespace NUMINAMATH_CALUDE_sprint_tournament_races_l3653_365323

theorem sprint_tournament_races (total_sprinters : ℕ) (lanes_per_race : ℕ) : 
  total_sprinters = 320 →
  lanes_per_race = 8 →
  (∃ (num_races : ℕ), 
    num_races = 46 ∧
    num_races = (total_sprinters - 1) / (lanes_per_race - 1) + 
      (if (total_sprinters - 1) % (lanes_per_race - 1) = 0 then 0 else 1)) :=
by
  sorry

#check sprint_tournament_races

end NUMINAMATH_CALUDE_sprint_tournament_races_l3653_365323


namespace NUMINAMATH_CALUDE_sunzi_carriage_problem_l3653_365367

/-- 
Given a number of carriages and people satisfying the conditions from 
"The Mathematical Classic of Sunzi", prove that the number of carriages 
satisfies the equation 3(x-2) = 2x + 9.
-/
theorem sunzi_carriage_problem (x : ℕ) (people : ℕ) :
  (3 * (x - 2) = people) →  -- Three people per carriage, two empty
  (2 * x + 9 = people) →    -- Two people per carriage, nine walking
  3 * (x - 2) = 2 * x + 9 := by
sorry

end NUMINAMATH_CALUDE_sunzi_carriage_problem_l3653_365367


namespace NUMINAMATH_CALUDE_x_minus_y_equals_four_l3653_365395

theorem x_minus_y_equals_four (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_four_l3653_365395


namespace NUMINAMATH_CALUDE_sachin_lending_rate_l3653_365307

/-- Calculates simple interest --/
def simpleInterest (principal time rate : ℚ) : ℚ :=
  principal * rate * time / 100

theorem sachin_lending_rate :
  let borrowed_amount : ℚ := 5000
  let borrowed_time : ℚ := 2
  let borrowed_rate : ℚ := 4
  let sachin_gain_per_year : ℚ := 112.5
  let borrowed_interest := simpleInterest borrowed_amount borrowed_time borrowed_rate
  let total_gain := sachin_gain_per_year * borrowed_time
  let total_interest_from_rahul := borrowed_interest + total_gain
  let rahul_rate := (total_interest_from_rahul * 100) / (borrowed_amount * borrowed_time)
  rahul_rate = 6.25 := by sorry

end NUMINAMATH_CALUDE_sachin_lending_rate_l3653_365307


namespace NUMINAMATH_CALUDE_peter_has_320_dollars_l3653_365352

-- Define the friends' money amounts
def john_money : ℝ := 160
def peter_money : ℝ := 2 * john_money
def quincy_money : ℝ := peter_money + 20
def andrew_money : ℝ := 1.15 * quincy_money

-- Define the total money and expenses
def total_money : ℝ := john_money + peter_money + quincy_money + andrew_money
def item_cost : ℝ := 1200
def money_left : ℝ := 11

-- Theorem to prove
theorem peter_has_320_dollars :
  peter_money = 320 ∧
  john_money + peter_money + quincy_money + andrew_money = item_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_peter_has_320_dollars_l3653_365352


namespace NUMINAMATH_CALUDE_max_large_chips_l3653_365391

/-- The smallest composite number -/
def smallest_composite : ℕ := 4

/-- Represents the problem of finding the maximum number of large chips -/
def chip_problem (total : ℕ) (small : ℕ) (large : ℕ) : Prop :=
  total = 60 ∧
  small + large = total ∧
  ∃ c : ℕ, c ≥ smallest_composite ∧ small = large + c

/-- The theorem stating the maximum number of large chips -/
theorem max_large_chips :
  ∀ total small large,
  chip_problem total small large →
  large ≤ 28 :=
sorry

end NUMINAMATH_CALUDE_max_large_chips_l3653_365391


namespace NUMINAMATH_CALUDE_x_gt_y_iff_exp_and_cbrt_l3653_365320

theorem x_gt_y_iff_exp_and_cbrt (x y : ℝ) : 
  x > y ↔ (Real.exp x > Real.exp y ∧ x^(1/3) > y^(1/3)) :=
sorry

end NUMINAMATH_CALUDE_x_gt_y_iff_exp_and_cbrt_l3653_365320


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3653_365326

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  mantissa_bounds : 1 ≤ mantissa ∧ mantissa < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 274000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { mantissa := 2.74
  , exponent := 8
  , mantissa_bounds := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.mantissa * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3653_365326


namespace NUMINAMATH_CALUDE_correct_propositions_l3653_365304

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| SpecificToGeneral
| GeneralToSpecific
| SpecificToSpecific
| GeneralToGeneral

-- Define a function to check if a statement about reasoning is correct
def isCorrectStatement (rt : ReasoningType) (rd : ReasoningDirection) : Prop :=
  match rt, rd with
  | ReasoningType.Inductive, ReasoningDirection.SpecificToGeneral => True
  | ReasoningType.Deductive, ReasoningDirection.GeneralToSpecific => True
  | ReasoningType.Analogical, ReasoningDirection.SpecificToSpecific => True
  | _, _ => False

-- Define the five propositions
def proposition1 := isCorrectStatement ReasoningType.Inductive ReasoningDirection.SpecificToGeneral
def proposition2 := isCorrectStatement ReasoningType.Inductive ReasoningDirection.GeneralToGeneral
def proposition3 := isCorrectStatement ReasoningType.Deductive ReasoningDirection.GeneralToSpecific
def proposition4 := isCorrectStatement ReasoningType.Analogical ReasoningDirection.SpecificToGeneral
def proposition5 := isCorrectStatement ReasoningType.Analogical ReasoningDirection.SpecificToSpecific

-- Theorem to prove
theorem correct_propositions :
  {n : Nat | n ∈ [1, 3, 5]} = {n : Nat | n ∈ [1, 2, 3, 4, 5] ∧ 
    match n with
    | 1 => proposition1
    | 2 => proposition2
    | 3 => proposition3
    | 4 => proposition4
    | 5 => proposition5
    | _ => False} :=
by sorry

end NUMINAMATH_CALUDE_correct_propositions_l3653_365304


namespace NUMINAMATH_CALUDE_least_sum_of_equal_multiples_l3653_365357

theorem least_sum_of_equal_multiples (x y z : ℕ+) (h : (2 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (8 : ℕ) * z.val) :
  x.val + y.val + z.val ≥ 33 ∧ ∃ (a b c : ℕ+), (2 : ℕ) * a.val = (5 : ℕ) * b.val ∧ (5 : ℕ) * b.val = (8 : ℕ) * c.val ∧ a.val + b.val + c.val = 33 :=
by
  sorry

#check least_sum_of_equal_multiples

end NUMINAMATH_CALUDE_least_sum_of_equal_multiples_l3653_365357


namespace NUMINAMATH_CALUDE_unique_solution_l3653_365356

/-- Represents a cell in the 5x5 grid --/
structure Cell :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents the 5x5 grid --/
def Grid := Cell → Fin 5

/-- Check if two cells are in the same row --/
def same_row (c1 c2 : Cell) : Prop := c1.row = c2.row

/-- Check if two cells are in the same column --/
def same_column (c1 c2 : Cell) : Prop := c1.col = c2.col

/-- Check if two cells are in the same block --/
def same_block (c1 c2 : Cell) : Prop :=
  (c1.row / 3 = c2.row / 3) ∧ (c1.col / 3 = c2.col / 3)

/-- Check if two cells are diagonally adjacent --/
def diag_adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row + 1 ∧ c1.col = c2.col + 1) ∨
  (c1.row = c2.row + 1 ∧ c1.col = c2.col - 1) ∨
  (c1.row = c2.row - 1 ∧ c1.col = c2.col + 1) ∨
  (c1.row = c2.row - 1 ∧ c1.col = c2.col - 1)

/-- Check if a grid is valid according to the rules --/
def valid_grid (g : Grid) : Prop :=
  ∀ c1 c2 : Cell, c1 ≠ c2 →
    (same_row c1 c2 ∨ same_column c1 c2 ∨ same_block c1 c2 ∨ diag_adjacent c1 c2) →
    g c1 ≠ g c2

/-- The unique solution to the puzzle --/
theorem unique_solution (g : Grid) (h : valid_grid g) :
  (g ⟨0, 0⟩ = 5) ∧ (g ⟨0, 1⟩ = 3) ∧ (g ⟨0, 2⟩ = 1) ∧ (g ⟨0, 3⟩ = 2) ∧ (g ⟨0, 4⟩ = 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3653_365356


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l3653_365362

theorem point_in_third_quadrant (x y : ℝ) (h1 : x + y < 0) (h2 : x * y > 0) : x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l3653_365362


namespace NUMINAMATH_CALUDE_janet_action_figures_l3653_365361

/-- Calculates the final number of action figures Janet has --/
def final_action_figures (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  let after_selling := initial - sold
  let after_buying := after_selling + bought
  let brothers_collection := 2 * after_buying
  after_buying + brothers_collection

/-- Theorem stating that Janet ends up with 24 action figures --/
theorem janet_action_figures :
  final_action_figures 10 6 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_janet_action_figures_l3653_365361


namespace NUMINAMATH_CALUDE_property_P_implies_m_range_l3653_365377

open Real

/-- Property P(a) for a function f -/
def has_property_P (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ h : ℝ → ℝ, (∀ x > 1, h x > 0) ∧
    (∀ x > 1, deriv f x = h x * (x^2 - a*x + 1))

theorem property_P_implies_m_range
  (g : ℝ → ℝ) (hg : has_property_P g 2)
  (x₁ x₂ : ℝ) (hx : 1 < x₁ ∧ x₁ < x₂)
  (m : ℝ) (α β : ℝ)
  (hα : α = m*x₁ + (1-m)*x₂)
  (hβ : β = (1-m)*x₁ + m*x₂)
  (hαβ : α > 1 ∧ β > 1)
  (hineq : |g α - g β| < |g x₁ - g x₂|) :
  0 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_property_P_implies_m_range_l3653_365377


namespace NUMINAMATH_CALUDE_complex_subtraction_equality_l3653_365339

theorem complex_subtraction_equality : ((1 - 1) - 1) - ((1 - (1 - 1))) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_equality_l3653_365339


namespace NUMINAMATH_CALUDE_concave_iff_m_nonneg_l3653_365379

/-- A function f is concave on a set A if for any x₁, x₂ ∈ A,
    f((x₁ + x₂)/2) ≤ (1/2)[f(x₁) + f(x₂)] -/
def IsConcave (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f ((x₁ + x₂) / 2) ≤ (f x₁ + f x₂) / 2

/-- The function f(x) = mx² + x -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x

theorem concave_iff_m_nonneg (m : ℝ) :
  IsConcave (f m) ↔ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_concave_iff_m_nonneg_l3653_365379


namespace NUMINAMATH_CALUDE_lorenzo_board_test_l3653_365368

/-- The number of boards Lorenzo tested -/
def boards_tested : ℕ := 120

/-- The total number of thumbtacks Lorenzo started with -/
def total_thumbtacks : ℕ := 450

/-- The number of cans of thumbtacks -/
def number_of_cans : ℕ := 3

/-- The number of thumbtacks remaining in each can at the end of the day -/
def remaining_thumbtacks_per_can : ℕ := 30

/-- The number of thumbtacks used per board -/
def thumbtacks_per_board : ℕ := 3

theorem lorenzo_board_test :
  boards_tested = (total_thumbtacks - number_of_cans * remaining_thumbtacks_per_can) / thumbtacks_per_board :=
by sorry

end NUMINAMATH_CALUDE_lorenzo_board_test_l3653_365368


namespace NUMINAMATH_CALUDE_consecutive_non_prime_powers_l3653_365317

/-- A number is a prime power if it can be expressed as p^k where p is prime and k ≥ 1 -/
def IsPrimePower (n : ℕ) : Prop :=
  ∃ (p k : ℕ), Prime p ∧ k ≥ 1 ∧ n = p^k

theorem consecutive_non_prime_powers (N : ℕ) (h : N > 0) :
  ∃ (M : ℤ), ∀ (i : ℕ), i < N → ¬IsPrimePower (Int.toNat (M + i)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_powers_l3653_365317


namespace NUMINAMATH_CALUDE_problem_solution_l3653_365331

theorem problem_solution (x y : ℝ) (h1 : x + 2*y = 14) (h2 : y = 3) : 2*x + 3*y = 25 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3653_365331


namespace NUMINAMATH_CALUDE_line_through_points_l3653_365316

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the line passing through two given points -/
def line_equation (p₀ p₁ : Point) : ℝ → Prop :=
  fun y => y = p₀.y

/-- The theorem states that the line equation y = 2 passes through the given points -/
theorem line_through_points :
  let p₀ : Point := ⟨1, 2⟩
  let p₁ : Point := ⟨3, 2⟩
  let eq := line_equation p₀ p₁
  (eq 2) ∧ (p₀.y = 2) ∧ (p₁.y = 2) := by sorry

end NUMINAMATH_CALUDE_line_through_points_l3653_365316


namespace NUMINAMATH_CALUDE_sin_alpha_abs_value_l3653_365332

/-- Theorem: If point P(3a, 4a) lies on the terminal side of angle α, where a ≠ 0, then |sin α| = 4/5 -/
theorem sin_alpha_abs_value (a : ℝ) (α : ℝ) (ha : a ≠ 0) :
  let P : ℝ × ℝ := (3 * a, 4 * a)
  (P.1 = 3 * a ∧ P.2 = 4 * a) → |Real.sin α| = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_abs_value_l3653_365332


namespace NUMINAMATH_CALUDE_mice_eaten_in_decade_l3653_365338

/-- Represents the number of weeks in a year -/
def weeksInYear : ℕ := 52

/-- Represents the eating frequency (in weeks) for the snake in its first year -/
def firstYearFrequency : ℕ := 4

/-- Represents the eating frequency (in weeks) for the snake in its second year -/
def secondYearFrequency : ℕ := 3

/-- Represents the eating frequency (in weeks) for the snake after its second year -/
def laterYearsFrequency : ℕ := 2

/-- Calculates the number of mice eaten in the first year -/
def miceEatenFirstYear : ℕ := weeksInYear / firstYearFrequency

/-- Calculates the number of mice eaten in the second year -/
def miceEatenSecondYear : ℕ := weeksInYear / secondYearFrequency

/-- Calculates the number of mice eaten in one year after the second year -/
def miceEatenPerLaterYear : ℕ := weeksInYear / laterYearsFrequency

/-- Represents the number of years in a decade -/
def yearsInDecade : ℕ := 10

/-- Theorem stating the total number of mice eaten over a decade -/
theorem mice_eaten_in_decade : 
  miceEatenFirstYear + miceEatenSecondYear + (yearsInDecade - 2) * miceEatenPerLaterYear = 238 := by
  sorry

end NUMINAMATH_CALUDE_mice_eaten_in_decade_l3653_365338


namespace NUMINAMATH_CALUDE_log_inequality_l3653_365312

theorem log_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let f : ℝ → ℝ := fun x ↦ |Real.log x / Real.log a|
  f (1/4) > f (1/3) ∧ f (1/3) > f 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3653_365312


namespace NUMINAMATH_CALUDE_homework_question_not_proposition_l3653_365365

-- Define what a proposition is
def is_proposition (s : String) : Prop := 
  ∃ (b : Bool), (s = "true") ∨ (s = "false")

-- Define the statement in question
def homework_question : String :=
  "Have you finished your homework?"

-- Theorem to prove
theorem homework_question_not_proposition :
  ¬ (is_proposition homework_question) := by
  sorry

end NUMINAMATH_CALUDE_homework_question_not_proposition_l3653_365365


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3653_365381

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {0, 1, 2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3653_365381


namespace NUMINAMATH_CALUDE_alloy_composition_theorem_l3653_365321

/-- Represents the composition of an alloy -/
structure AlloyComposition where
  copper : ℝ
  tin : ℝ
  zinc : ℝ
  sum_to_one : copper + tin + zinc = 1

/-- The conditions given in the problem -/
def satisfies_conditions (c : AlloyComposition) : Prop :=
  c.copper - c.tin = 1/10 ∧ c.tin - c.zinc = 3/10

/-- The theorem to be proved -/
theorem alloy_composition_theorem :
  ∃ (c : AlloyComposition),
    satisfies_conditions c ∧
    c.copper = 0.5 ∧ c.tin = 0.4 ∧ c.zinc = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_alloy_composition_theorem_l3653_365321


namespace NUMINAMATH_CALUDE_largest_product_bound_l3653_365398

theorem largest_product_bound (a : Fin 1985 → ℕ) 
  (h_perm : Function.Bijective a) 
  (h_range : ∀ i, a i ∈ Finset.range 1986) : 
  (Finset.range 1985).sup (λ k => (k + 1) * a k) ≥ 993^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_product_bound_l3653_365398


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3653_365375

theorem inequality_equivalence (x : ℝ) : x - 1 > 0 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3653_365375


namespace NUMINAMATH_CALUDE_mildred_oranges_l3653_365327

/-- The number of oranges Mildred ends up with after a series of operations -/
def final_oranges (initial : ℕ) : ℕ :=
  let from_father := 3 * initial
  let after_father := initial + from_father
  let after_sister := after_father - 174
  2 * after_sister

/-- Theorem stating that given an initial collection of 215 oranges, 
    Mildred ends up with 1372 oranges after the described operations -/
theorem mildred_oranges : final_oranges 215 = 1372 := by
  sorry

end NUMINAMATH_CALUDE_mildred_oranges_l3653_365327


namespace NUMINAMATH_CALUDE_circumradius_inradius_ratio_irrational_l3653_365353

-- Define a lattice point
def LatticePoint := ℤ × ℤ

-- Define a triangle with lattice points as vertices
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

-- Define a square-free natural number
def SquareFree (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m * m ∣ n → m = 1

-- Define the property that one side of the triangle has length √n
def HasSqrtNSide (t : LatticeTriangle) (n : ℕ) : Prop :=
  SquareFree n ∧
  (((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 : ℚ) = n ∨
   ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2 : ℚ) = n ∨
   ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2 : ℚ) = n)

-- Define the circumradius and inradius of a triangle
noncomputable def circumradius (t : LatticeTriangle) : ℝ := sorry
noncomputable def inradius (t : LatticeTriangle) : ℝ := sorry

-- The main theorem
theorem circumradius_inradius_ratio_irrational (t : LatticeTriangle) (n : ℕ) :
  HasSqrtNSide t n → ¬ (∃ q : ℚ, (circumradius t / inradius t : ℝ) = q) :=
sorry

end NUMINAMATH_CALUDE_circumradius_inradius_ratio_irrational_l3653_365353


namespace NUMINAMATH_CALUDE_polynomial_inequality_l3653_365389

theorem polynomial_inequality (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1/2) →
  (∀ x : ℝ, |x| ≥ 1 → |a * x^2 + b * x + c| ≤ x^2 - 1/2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l3653_365389


namespace NUMINAMATH_CALUDE_initial_points_count_l3653_365328

theorem initial_points_count (k : ℕ) : 
  k > 0 → 4 * k - 3 = 101 → k = 26 := by
  sorry

end NUMINAMATH_CALUDE_initial_points_count_l3653_365328


namespace NUMINAMATH_CALUDE_max_large_sculptures_l3653_365358

theorem max_large_sculptures (total_blocks : ℕ) (small_sculptures large_sculptures : ℕ) : 
  total_blocks = 30 →
  small_sculptures > large_sculptures →
  small_sculptures + 3 * large_sculptures + (small_sculptures + large_sculptures) / 2 ≤ total_blocks →
  large_sculptures ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_large_sculptures_l3653_365358


namespace NUMINAMATH_CALUDE_student_average_age_l3653_365311

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (new_average : ℝ) 
  (h1 : num_students = 10)
  (h2 : teacher_age = 26)
  (h3 : new_average = 16)
  (h4 : (num_students : ℝ) * new_average = (num_students + 1 : ℝ) * new_average - teacher_age) :
  (num_students : ℝ) * new_average - teacher_age = num_students * 15 := by
sorry

end NUMINAMATH_CALUDE_student_average_age_l3653_365311


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_problem_solution_l3653_365301

/-- An arithmetic sequence with the property that the sequence of products of consecutive terms
    forms a geometric progression, and the first term is 1, is constant with all terms equal to 1. -/
theorem arithmetic_geometric_sequence_property (a : ℕ → ℝ) : 
  (∀ n, ∃ d, a (n + 1) - a n = d) →  -- arithmetic sequence
  (∀ n, ∃ r, (a (n + 1) * a (n + 2)) / (a n * a (n + 1)) = r) →  -- geometric progression of products
  a 1 = 1 →  -- first term is 1
  ∀ n, a n = 1 :=  -- all terms are 1
by sorry

/-- The 2017th term of the sequence described in the problem is 1. -/
theorem problem_solution (a : ℕ → ℝ) :
  (∀ n, ∃ d, a (n + 1) - a n = d) →  -- arithmetic sequence
  (∀ n, ∃ r, (a (n + 1) * a (n + 2)) / (a n * a (n + 1)) = r) →  -- geometric progression of products
  a 1 = 1 →  -- first term is 1
  a 2017 = 1 :=  -- 2017th term is 1
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_problem_solution_l3653_365301


namespace NUMINAMATH_CALUDE_alex_marbles_l3653_365347

theorem alex_marbles (lorin_black : ℕ) (jimmy_yellow : ℕ) 
  (h1 : lorin_black = 4)
  (h2 : jimmy_yellow = 22)
  (alex_black : ℕ) (alex_yellow : ℕ)
  (h3 : alex_black = 2 * lorin_black)
  (h4 : alex_yellow = jimmy_yellow / 2) :
  alex_black + alex_yellow = 19 := by
sorry

end NUMINAMATH_CALUDE_alex_marbles_l3653_365347


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_zero_l3653_365373

theorem negation_of_forall_geq_zero :
  (¬ ∀ x : ℝ, 2 * x + 4 ≥ 0) ↔ (∃ x : ℝ, 2 * x + 4 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_zero_l3653_365373


namespace NUMINAMATH_CALUDE_circle_C2_equation_l3653_365392

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

-- Define circle C1
def circle_C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define circle C2
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

-- Define symmetry relation
def symmetric_point (x y x' y' : ℝ) : Prop :=
  line_of_symmetry ((x + x') / 2) ((y + y') / 2) ∧
  (x - x')^2 + (y - y')^2 = 2 * ((x - y - 1)^2)

-- Theorem statement
theorem circle_C2_equation :
  ∀ x y : ℝ, circle_C2 x y ↔
  ∃ x' y' : ℝ, circle_C1 x' y' ∧ symmetric_point x y x' y' :=
sorry

end NUMINAMATH_CALUDE_circle_C2_equation_l3653_365392


namespace NUMINAMATH_CALUDE_range_of_x_l3653_365354

def is_monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f y ≤ f x

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem range_of_x (f : ℝ → ℝ) 
  (h1 : is_monotone_decreasing f)
  (h2 : is_odd_function f)
  (h3 : f 1 = -1)
  (h4 : ∀ x, -1 ≤ f (x - 2) ∧ f (x - 2) ≤ 1) :
  ∀ x, -1 ≤ f (x - 2) ∧ f (x - 2) ≤ 1 → 1 ≤ x ∧ x ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l3653_365354


namespace NUMINAMATH_CALUDE_triangle_inequality_expression_l3653_365378

theorem triangle_inequality_expression (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 - 2*a*b + b^2 - c^2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_expression_l3653_365378


namespace NUMINAMATH_CALUDE_digit_of_fraction_l3653_365394

/-- The fraction we're considering -/
def f : ℚ := 66 / 1110

/-- The index of the digit we're looking for (0-indexed) -/
def n : ℕ := 221

/-- The function that returns the nth digit after the decimal point
    in the decimal representation of a rational number -/
noncomputable def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem digit_of_fraction :
  nth_digit_after_decimal f n = 5 := by sorry

end NUMINAMATH_CALUDE_digit_of_fraction_l3653_365394


namespace NUMINAMATH_CALUDE_valid_selections_eq_48_l3653_365396

/-- The number of ways to select k items from n items -/
def arrangements (n k : ℕ) : ℕ := sorry

/-- The number of valid selections given the problem constraints -/
def valid_selections : ℕ :=
  arrangements 5 3 - arrangements 4 2

theorem valid_selections_eq_48 : valid_selections = 48 := by sorry

end NUMINAMATH_CALUDE_valid_selections_eq_48_l3653_365396


namespace NUMINAMATH_CALUDE_exists_perfect_square_with_digit_sum_2011_l3653_365350

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a perfect square with sum of digits 2011 -/
theorem exists_perfect_square_with_digit_sum_2011 : 
  ∃ n : ℕ, sum_of_digits (n^2) = 2011 := by
sorry

end NUMINAMATH_CALUDE_exists_perfect_square_with_digit_sum_2011_l3653_365350


namespace NUMINAMATH_CALUDE_digit_equation_solution_l3653_365333

theorem digit_equation_solution :
  ∀ (A M C : ℕ),
  (A ≤ 9 ∧ M ≤ 9 ∧ C ≤ 9) →
  (10 * A^2 + 10 * M + C) * (A + M^2 + C^2) = 1050 →
  A = 2 := by
sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l3653_365333


namespace NUMINAMATH_CALUDE_probability_increasing_maxima_correct_l3653_365318

/-- The probability that the maximum numbers in each row of a triangular array
    are in strictly increasing order. -/
def probability_increasing_maxima (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / (n + 1).factorial

/-- Theorem stating that the probability of increasing maxima in a triangular array
    with n rows is equal to 2^n / (n+1)! -/
theorem probability_increasing_maxima_correct (n : ℕ) :
  let array_size := n * (n + 1) / 2
  probability_increasing_maxima n =
    (2 ^ n : ℚ) / (n + 1).factorial :=
by sorry

end NUMINAMATH_CALUDE_probability_increasing_maxima_correct_l3653_365318


namespace NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l3653_365370

theorem divisibility_implies_multiple_of_three (a b : ℤ) : 
  (9 ∣ a^2 + a*b + b^2) → (3 ∣ a) ∧ (3 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l3653_365370


namespace NUMINAMATH_CALUDE_remainder_evaluation_l3653_365335

-- Define the remainder function
def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

-- State the theorem
theorem remainder_evaluation :
  rem (-1/7 : ℚ) (1/3 : ℚ) = 4/21 := by
  sorry

end NUMINAMATH_CALUDE_remainder_evaluation_l3653_365335


namespace NUMINAMATH_CALUDE_two_std_dev_below_value_l3653_365300

/-- Represents a normal distribution --/
structure NormalDistribution where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation

/-- The value that is exactly 2 standard deviations less than the mean --/
def twoStdDevBelow (nd : NormalDistribution) : ℝ :=
  nd.μ - 2 * nd.σ

theorem two_std_dev_below_value :
  let nd : NormalDistribution := { μ := 14.0, σ := 1.5 }
  twoStdDevBelow nd = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_value_l3653_365300


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l3653_365306

theorem quadratic_inequality_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l3653_365306


namespace NUMINAMATH_CALUDE_credit_card_balance_l3653_365399

theorem credit_card_balance (B : ℝ) : 
  (1.44 * B + 24 = 96) → B = 50 := by
sorry

end NUMINAMATH_CALUDE_credit_card_balance_l3653_365399


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3653_365345

/-- Given that i² = -1, prove that w = -2i/3 is the solution to the equation 3 - iw = 1 + 2iw -/
theorem complex_equation_solution (i : ℂ) (h : i^2 = -1) :
  let w : ℂ := -2*i/3
  3 - i*w = 1 + 2*i*w := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3653_365345


namespace NUMINAMATH_CALUDE_max_tangent_segments_2017_l3653_365349

/-- Given a number of circles, calculates the maximum number of tangent segments -/
def max_tangent_segments (n : ℕ) : ℕ := 3 * (n * (n - 1)) / 2

/-- Theorem: The maximum number of tangent segments for 2017 circles is 6,051,252 -/
theorem max_tangent_segments_2017 :
  max_tangent_segments 2017 = 6051252 := by
  sorry

#eval max_tangent_segments 2017

end NUMINAMATH_CALUDE_max_tangent_segments_2017_l3653_365349


namespace NUMINAMATH_CALUDE_triple_composition_even_l3653_365341

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem triple_composition_even (f : ℝ → ℝ) (h : IsEven f) :
  ∀ x, f (f (f (-x))) = f (f (f x)) := by sorry

end NUMINAMATH_CALUDE_triple_composition_even_l3653_365341


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3653_365351

/-- A line passing through (1, 2) with equal intercepts on both coordinate axes -/
structure EqualInterceptLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (1, 2)
  passes_through : 2 = m * 1 + b
  -- The line has equal intercepts on both axes
  equal_intercepts : m ≠ -1 → b = b / m

/-- The equation of an EqualInterceptLine is either 2x - y = 0 or x + y - 3 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 2 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 3) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3653_365351


namespace NUMINAMATH_CALUDE_circle_equation_l3653_365374

/-- A circle C with given properties -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (center_in_first_quadrant : center.1 > 0 ∧ center.2 > 0)
  (tangent_to_line : |4 * center.1 - 3 * center.2| = 5 * radius)
  (tangent_to_x_axis : center.2 = radius)
  (radius_is_one : radius = 1)

/-- The standard equation of a circle -/
def standard_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Theorem: The standard equation of the circle C is (x-2)^2 + (y-1)^2 = 1 -/
theorem circle_equation (c : Circle) :
  ∀ x y : ℝ, standard_equation c x y ↔ (x - 2)^2 + (y - 1)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3653_365374


namespace NUMINAMATH_CALUDE_quadratic_function_continuous_l3653_365330

/-- A quadratic function f(x) = ax^2 + bx + c is continuous at any point x ∈ ℝ,
    where a, b, and c are real constants. -/
theorem quadratic_function_continuous (a b c : ℝ) :
  Continuous (fun x : ℝ => a * x^2 + b * x + c) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_continuous_l3653_365330


namespace NUMINAMATH_CALUDE_complex_inequalities_l3653_365380

theorem complex_inequalities :
  (∀ z w : ℂ, Complex.abs z + Complex.abs w ≤ Complex.abs (z + w) + Complex.abs (z - w)) ∧
  (∀ z₁ z₂ z₃ z₄ : ℂ, 
    Complex.abs z₁ + Complex.abs z₂ + Complex.abs z₃ + Complex.abs z₄ ≤
    Complex.abs (z₁ + z₂) + Complex.abs (z₁ + z₃) + Complex.abs (z₁ + z₄) +
    Complex.abs (z₂ + z₃) + Complex.abs (z₂ + z₄) + Complex.abs (z₃ + z₄)) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequalities_l3653_365380


namespace NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l3653_365340

/-- A linear function f(x) = -x - 2 -/
def f (x : ℝ) : ℝ := -x - 2

/-- The first quadrant of the coordinate plane -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem linear_function_not_in_first_quadrant :
  ∀ x : ℝ, ¬(first_quadrant x (f x)) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l3653_365340


namespace NUMINAMATH_CALUDE_coefficient_of_sixth_power_l3653_365355

theorem coefficient_of_sixth_power (x : ℝ) :
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
    (2 - x)^6 = a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5 + a₆*(1+x)^6 ∧
    a₆ = 1 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_sixth_power_l3653_365355


namespace NUMINAMATH_CALUDE_pencil_length_l3653_365342

theorem pencil_length (length1 length2 total_length : ℕ) : 
  length1 = length2 → 
  length1 + length2 = 24 → 
  length1 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l3653_365342
