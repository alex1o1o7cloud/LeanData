import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_and_function_property_l2782_278240

def f (x : ℝ) : ℝ := |x + 1|

theorem inequality_solution_and_function_property :
  (∃ (S : Set ℝ), S = {x : ℝ | x ≤ -10 ∨ x ≥ 0} ∧
    ∀ x, x ∈ S ↔ f (x + 8) ≥ 10 - f x) ∧
  (∀ x y, |x| > 1 → |y| < 1 → f y < |x| * f (y / x^2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_and_function_property_l2782_278240


namespace NUMINAMATH_CALUDE_line_circle_properties_l2782_278211

-- Define the line l and circle C
def line_l (m x y : ℝ) : Prop := (m + 1) * x + 2 * y - m - 3 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 4 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 2)

-- Theorem statement
theorem line_circle_properties (m : ℝ) :
  (∀ x y, line_l m x y → (x = 1 ∧ y = 1)) ∧
  (∃ x y, line_l m x y ∧ circle_C x y) ∧
  (∃ x y, line_l m x y ∧ 
    Real.sqrt ((x - circle_center.1)^2 + (y - circle_center.2)^2) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_properties_l2782_278211


namespace NUMINAMATH_CALUDE_problem_statement_l2782_278208

theorem problem_statement (a b : ℝ) (h1 : a * b = 2) (h2 : a + b = 3) :
  a^2 * b + a * b^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2782_278208


namespace NUMINAMATH_CALUDE_sequence_general_term_l2782_278238

/-- Given a sequence {a_n} where S_n is the sum of the first n terms 
    and S_n = (1/2)(1 - a_n), prove that a_n = (1/3)^n -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = (1/2) * (1 - a n)) :
  ∀ n, a n = (1/3)^n := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2782_278238


namespace NUMINAMATH_CALUDE_equality_multiplication_negative_two_l2782_278272

theorem equality_multiplication_negative_two (m n : ℝ) : m = n → -2 * m = -2 * n := by
  sorry

end NUMINAMATH_CALUDE_equality_multiplication_negative_two_l2782_278272


namespace NUMINAMATH_CALUDE_tangent_line_power_l2782_278232

/-- Given a curve y = x^2 + ax + b with a tangent line at (0, b) of equation x - y + 1 = 0, prove a^b = 1 -/
theorem tangent_line_power (a b : ℝ) : 
  (∀ x, x^2 + a*x + b = 0 → x - (x^2 + a*x + b) + 1 = 0) → 
  a^b = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_power_l2782_278232


namespace NUMINAMATH_CALUDE_parallel_plane_intersection_theorem_l2782_278218

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Theorem statement
theorem parallel_plane_intersection_theorem
  (α β γ : Plane) (a b : Line)
  (h1 : parallel_planes α β)
  (h2 : intersect α γ = a)
  (h3 : intersect β γ = b) :
  parallel_lines a b :=
sorry

end NUMINAMATH_CALUDE_parallel_plane_intersection_theorem_l2782_278218


namespace NUMINAMATH_CALUDE_check_cashing_error_l2782_278271

theorem check_cashing_error (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  x > y →
  100 * y + x - (100 * x + y) = 2187 →
  x - y = 22 := by
sorry

end NUMINAMATH_CALUDE_check_cashing_error_l2782_278271


namespace NUMINAMATH_CALUDE_dans_age_problem_l2782_278247

theorem dans_age_problem (x : ℝ) : (8 + 20 : ℝ) = 7 * (8 - x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_dans_age_problem_l2782_278247


namespace NUMINAMATH_CALUDE_expression_simplification_l2782_278261

theorem expression_simplification (x y : ℚ) (hx : x = -2) (hy : y = -1) :
  (2 * (x - 2*y) * (2*x + y) - (x + 2*y)^2 + x * (8*y - 3*x)) / (6*y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2782_278261


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2782_278235

theorem complex_modulus_problem (z : ℂ) : z = (1 - 3*I) / (1 + 2*I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2782_278235


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l2782_278286

/-- The probability of selecting either a blue or purple jelly bean from a bag -/
theorem jelly_bean_probability : 
  let red : ℕ := 4
  let green : ℕ := 5
  let yellow : ℕ := 9
  let blue : ℕ := 7
  let purple : ℕ := 10
  let total : ℕ := red + green + yellow + blue + purple
  (blue + purple : ℚ) / total = 17 / 35 := by sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l2782_278286


namespace NUMINAMATH_CALUDE_sphere_volume_from_parallel_planes_l2782_278249

theorem sphere_volume_from_parallel_planes (R : ℝ) :
  R > 0 →
  ∃ (h : ℝ),
    h > 0 ∧
    h < R ∧
    (h^2 + 9^2 = R^2) ∧
    ((h + 3)^2 + 12^2 = R^2) →
    (4 / 3 * Real.pi * R^3 = 4050 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_parallel_planes_l2782_278249


namespace NUMINAMATH_CALUDE_equation_solutions_no_other_solutions_l2782_278251

/-- Definition of the factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The main theorem stating the solutions to the equation -/
theorem equation_solutions :
  ∀ x y z : ℕ, 2^x + 3^y + 7 = factorial z ↔ (x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5) :=
by sorry

/-- Auxiliary theorem: There are no other solutions -/
theorem no_other_solutions :
  ∀ x y z : ℕ, 2^x + 3^y + 7 = factorial z →
  (x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_no_other_solutions_l2782_278251


namespace NUMINAMATH_CALUDE_tangent_sum_l2782_278258

theorem tangent_sum (x y a b : Real) 
  (h1 : Real.sin (2 * x) + Real.sin (2 * y) = a)
  (h2 : Real.cos (2 * x) + Real.cos (2 * y) = b)
  (h3 : a^2 + b^2 ≤ 4)
  (h4 : a^2 + b^2 + 2*b ≠ 0) :
  Real.tan x + Real.tan y = 4 * a / (a^2 + b^2 + 2*b) :=
sorry

end NUMINAMATH_CALUDE_tangent_sum_l2782_278258


namespace NUMINAMATH_CALUDE_count_five_digit_even_divisible_by_five_l2782_278287

def is_even_digit (d : Nat) : Bool :=
  d % 2 = 0

def has_only_even_digits (n : Nat) : Bool :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

theorem count_five_digit_even_divisible_by_five : 
  (Finset.filter (λ n : Nat => 
    10000 ≤ n ∧ n ≤ 99999 ∧ 
    has_only_even_digits n ∧ 
    n % 5 = 0
  ) (Finset.range 100000)).card = 500 := by
  sorry

end NUMINAMATH_CALUDE_count_five_digit_even_divisible_by_five_l2782_278287


namespace NUMINAMATH_CALUDE_correct_log_values_l2782_278289

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define variables a, b, c
variable (a b c : ℝ)

-- Define the given correct logarithmic values
axiom log_3 : log 3 = 2*a - b
axiom log_5 : log 5 = a + c
axiom log_9 : log 9 = 4*a - 2*b
axiom log_0_27 : log 0.27 = 6*a - 3*b - 2
axiom log_8 : log 8 = 3 - 3*a - 3*c
axiom log_6 : log 6 = 1 + a - b - c

-- State the theorem to be proved
theorem correct_log_values :
  log 1.5 = 3*a - b + c - 1 ∧ log 7 = 2*b + c :=
sorry

end NUMINAMATH_CALUDE_correct_log_values_l2782_278289


namespace NUMINAMATH_CALUDE_student_weights_l2782_278203

theorem student_weights (A B C D : ℕ) : 
  A < B ∧ B < C ∧ C < D →
  A + B = 45 →
  A + C = 49 →
  B + C = 54 →
  B + D = 60 →
  C + D = 64 →
  D = 35 := by
sorry

end NUMINAMATH_CALUDE_student_weights_l2782_278203


namespace NUMINAMATH_CALUDE_set_operations_l2782_278256

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 3}

-- State the theorem
theorem set_operations :
  (Aᶜ : Set ℝ) = {x : ℝ | x ≥ 3 ∨ x ≤ -2} ∧
  (A ∩ B : Set ℝ) = {x : ℝ | -2 < x ∧ x < 3} ∧
  ((A ∩ B)ᶜ : Set ℝ) = {x : ℝ | x ≥ 3 ∨ x ≤ -2} ∧
  (Aᶜ ∩ B : Set ℝ) = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ x = 3} :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l2782_278256


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2782_278297

-- Define the equation
def equation (x : ℝ) : Prop := |x - 1| = 3 * |x + 3|

-- Define the set of solutions
def solution_set : Set ℝ := {x : ℝ | equation x}

-- State the theorem
theorem sum_of_solutions :
  ∃ (s : Finset ℝ), s.toSet = solution_set ∧ s.sum id = -7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2782_278297


namespace NUMINAMATH_CALUDE_max_y_value_l2782_278255

theorem max_y_value (x y : ℝ) :
  (Real.log (x + y) / Real.log (x^2 + y^2) ≥ 1) →
  y ≤ 1/2 + Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_y_value_l2782_278255


namespace NUMINAMATH_CALUDE_simplify_expression_l2782_278222

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^7 = 1280 * 16^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2782_278222


namespace NUMINAMATH_CALUDE_money_distribution_l2782_278207

/-- The problem of distributing money among boys -/
theorem money_distribution (total_amount : ℕ) (extra_per_boy : ℕ) : 
  (total_amount = 5040) →
  (extra_per_boy = 80) →
  ∃ (x : ℕ), 
    (x * (total_amount / 18 + extra_per_boy) = total_amount) ∧
    (x = 14) := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l2782_278207


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l2782_278221

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem circle_and_line_intersection :
  ∀ m : ℝ,
  (∀ x y : ℝ, circle_equation x y m → x^2 + y^2 = (x - 1)^2 + (y - 2)^2 + (5 - m)) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ m ∧ 
    circle_equation x₂ y₂ m ∧ 
    line_equation x₁ y₁ ∧ 
    line_equation x₂ y₂ ∧ 
    perpendicular x₁ y₁ x₂ y₂) →
  (m < 5 ∧ m = 8/5 ∧ 
   ∀ x y : ℝ, x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔ 
   ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
   x = (1-t)*x₁ + t*x₂ ∧ 
   y = (1-t)*y₁ + t*y₂) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l2782_278221


namespace NUMINAMATH_CALUDE_math_homework_percentage_l2782_278292

/-- Proves that the percentage of time spent on math homework is 30%, given the total homework time,
    time spent on science, and time spent on other subjects. -/
theorem math_homework_percentage
  (total_time : ℝ)
  (science_percentage : ℝ)
  (other_subjects_time : ℝ)
  (h1 : total_time = 150)
  (h2 : science_percentage = 0.4)
  (h3 : other_subjects_time = 45)
  : (total_time - science_percentage * total_time - other_subjects_time) / total_time = 0.3 := by
  sorry

#check math_homework_percentage

end NUMINAMATH_CALUDE_math_homework_percentage_l2782_278292


namespace NUMINAMATH_CALUDE_poster_wall_width_l2782_278216

/-- Calculates the minimum wall width required to attach a given number of posters with specified width and overlap. -/
def minimumWallWidth (posterWidth : ℕ) (overlap : ℕ) (numPosters : ℕ) : ℕ :=
  posterWidth + (numPosters - 1) * (posterWidth - overlap)

/-- Theorem stating that 15 posters of width 30 cm, overlapping by 2 cm, require a wall width of 422 cm. -/
theorem poster_wall_width :
  minimumWallWidth 30 2 15 = 422 := by
  sorry

end NUMINAMATH_CALUDE_poster_wall_width_l2782_278216


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l2782_278282

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^2 - 6*x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 9

-- Define the line
def line (x y : ℝ) (a : ℝ) : Prop := x - y + a = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem
theorem circle_and_line_intersection :
  ∃ (x1 y1 x2 y2 : ℝ),
    -- The curve intersects the coordinate axes at points on circle C
    (curve 0 y1 ∧ circle_C 0 y1) ∧
    (curve x1 0 ∧ circle_C x1 0) ∧
    (curve x2 0 ∧ circle_C x2 0) ∧
    -- Circle C intersects the line at A(x1, y1) and B(x2, y2)
    (circle_C x1 y1 ∧ line x1 y1 (-1)) ∧
    (circle_C x2 y2 ∧ line x2 y2 (-1)) ∧
    -- OA ⊥ OB
    perpendicular x1 y1 x2 y2 :=
  sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l2782_278282


namespace NUMINAMATH_CALUDE_increase_in_average_weight_l2782_278285

/-- Theorem: Increase in average weight when replacing a person in a group -/
theorem increase_in_average_weight 
  (group_size : ℕ) 
  (old_weight new_weight : ℝ) 
  (h1 : group_size = 8)
  (h2 : old_weight = 55)
  (h3 : new_weight = 87) :
  (new_weight - old_weight) / group_size = 4 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_average_weight_l2782_278285


namespace NUMINAMATH_CALUDE_limit_at_one_l2782_278296

def f (x : ℝ) : ℝ := 2 * x^3

theorem limit_at_one (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f (1 + Δx) - f 1) / Δx) - 6| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_at_one_l2782_278296


namespace NUMINAMATH_CALUDE_midpoint_fraction_l2782_278264

theorem midpoint_fraction : 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 7
  let midpoint := (a + b) / 2
  midpoint = (41 : ℚ) / 56 := by
sorry

end NUMINAMATH_CALUDE_midpoint_fraction_l2782_278264


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2782_278281

theorem cube_root_equation_solution :
  ∃ x : ℝ, (2 * x * (x^3)^(1/2))^(1/3) = 6 ∧ x = 108^(2/5) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2782_278281


namespace NUMINAMATH_CALUDE_height_for_weight_35_l2782_278280

/-- Linear regression equation relating height to weight -/
def linear_regression (x : ℝ) : ℝ := 0.1 * x + 20

/-- Theorem stating that a person weighing 35 kg has a height of 150 cm
    according to the given linear regression equation -/
theorem height_for_weight_35 :
  ∃ x : ℝ, linear_regression x = 35 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_height_for_weight_35_l2782_278280


namespace NUMINAMATH_CALUDE_book_pages_equation_l2782_278215

theorem book_pages_equation (x : ℝ) : 
  x > 0 → 
  20 + (1/2) * (x - 20) + 15 = x := by
  sorry

end NUMINAMATH_CALUDE_book_pages_equation_l2782_278215


namespace NUMINAMATH_CALUDE_greatest_base8_digit_sum_l2782_278294

/-- Represents a positive integer in base 8 --/
structure Base8Int where
  digits : List Nat
  positive : digits ≠ []
  valid : ∀ d ∈ digits, d < 8

/-- Converts a Base8Int to its decimal representation --/
def toDecimal (n : Base8Int) : Nat :=
  sorry

/-- Computes the sum of digits of a Base8Int --/
def digitSum (n : Base8Int) : Nat :=
  sorry

/-- The theorem to be proved --/
theorem greatest_base8_digit_sum :
  (∃ (n : Base8Int), toDecimal n < 1728 ∧
    ∀ (m : Base8Int), toDecimal m < 1728 → digitSum m ≤ digitSum n) ∧
  (∀ (n : Base8Int), toDecimal n < 1728 → digitSum n ≤ 23) :=
sorry

end NUMINAMATH_CALUDE_greatest_base8_digit_sum_l2782_278294


namespace NUMINAMATH_CALUDE_function_value_problem_l2782_278239

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f x = 2^x - 5)
  (h2 : f m = 3) : 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_function_value_problem_l2782_278239


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l2782_278242

/-- Represents the state of the game -/
structure GameState :=
  (dominoes : Finset Nat)
  (current_score : Nat)
  (last_move : Nat)

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (move : Nat) : Prop :=
  move ∈ state.dominoes ∧ move ≠ state.last_move

/-- Defines the winning condition -/
def is_winning_state (state : GameState) : Prop :=
  state.current_score = 37 ∨ state.current_score > 37

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Nat

/-- Defines a winning strategy for the first player -/
def winning_strategy (s : Strategy) : Prop :=
  ∀ (initial_state : GameState),
    initial_state.dominoes = {1, 2, 3, 4, 5} →
    initial_state.current_score = 0 →
    ∃ (final_state : GameState),
      is_winning_state final_state ∧
      (∀ (opponent_move : Nat),
        valid_move initial_state opponent_move →
        valid_move (GameState.mk 
          initial_state.dominoes
          (initial_state.current_score + opponent_move)
          opponent_move) 
        (s (GameState.mk 
          initial_state.dominoes
          (initial_state.current_score + opponent_move)
          opponent_move)))

/-- Theorem stating that there exists a winning strategy for the first player -/
theorem first_player_winning_strategy :
  ∃ (s : Strategy), winning_strategy s :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l2782_278242


namespace NUMINAMATH_CALUDE_maximize_f_l2782_278277

-- Define the function f
def f (a b c x y z : ℝ) : ℝ := a * x + b * y + c * z

-- State the theorem
theorem maximize_f (a b c : ℝ) :
  (∀ x y z : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ -5 ≤ y ∧ y ≤ 5 ∧ -5 ≤ z ∧ z ≤ 5) →
  f a b c 3 1 1 > f a b c 2 1 1 →
  f a b c 2 2 3 > f a b c 2 3 4 →
  f a b c 3 3 4 > f a b c 3 3 3 →
  ∀ x y z : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ -5 ≤ y ∧ y ≤ 5 ∧ -5 ≤ z ∧ z ≤ 5 →
  f a b c x y z ≤ f a b c 5 (-5) 5 :=
by sorry

end NUMINAMATH_CALUDE_maximize_f_l2782_278277


namespace NUMINAMATH_CALUDE_johns_actual_marks_l2782_278246

theorem johns_actual_marks (total : ℝ) (n : ℕ) (wrong_mark : ℝ) (increase : ℝ) :
  n = 80 →
  wrong_mark = 82 →
  increase = 1/2 →
  (total + wrong_mark) / n = (total + johns_mark) / n + increase →
  johns_mark = 42 :=
by sorry

end NUMINAMATH_CALUDE_johns_actual_marks_l2782_278246


namespace NUMINAMATH_CALUDE_min_value_expression_l2782_278295

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  y / x + 16 * x / (2 * x + y) ≥ 6 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ y₀ / x₀ + 16 * x₀ / (2 * x₀ + y₀) = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2782_278295


namespace NUMINAMATH_CALUDE_a_4_value_l2782_278205

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_4_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 = 2 ∨ a 2 = 32) →
  (a 6 = 2 ∨ a 6 = 32) →
  a 2 * a 6 = 64 →
  a 2 + a 6 = 34 →
  a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_a_4_value_l2782_278205


namespace NUMINAMATH_CALUDE_matrix_equality_l2782_278236

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]

def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]

theorem matrix_equality (x y z w : ℝ) (h1 : A * B x y z w = B x y z w * A) (h2 : 4 * y ≠ z) :
  (x - w) / (z - 4 * y) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_l2782_278236


namespace NUMINAMATH_CALUDE_tangent_line_values_l2782_278283

/-- A line y = kx + b is tangent to two circles -/
def is_tangent_to_circles (k b : ℝ) : Prop :=
  k > 0 ∧
  ∃ (x₁ y₁ : ℝ), x₁^2 + y₁^2 = 1 ∧ y₁ = k * x₁ + b ∧
  ∃ (x₂ y₂ : ℝ), (x₂ - 4)^2 + y₂^2 = 1 ∧ y₂ = k * x₂ + b

/-- The unique values of k and b for a line tangent to both circles -/
theorem tangent_line_values :
  ∀ k b : ℝ, is_tangent_to_circles k b →
  k = Real.sqrt 3 / 3 ∧ b = -(2 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_values_l2782_278283


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2782_278220

/-- The surface area of a sphere circumscribed around a rectangular solid -/
theorem circumscribed_sphere_surface_area 
  (length width height : ℝ) 
  (h_length : length = 2)
  (h_width : width = 2)
  (h_height : height = 2 * Real.sqrt 2) : 
  4 * Real.pi * ((length^2 + width^2 + height^2) / 4) = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2782_278220


namespace NUMINAMATH_CALUDE_two_zeros_implies_m_values_l2782_278269

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x + m

/-- The statement that f has exactly two zeros -/
def has_two_zeros (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧ ∀ z : ℝ, f m z = 0 → z = x ∨ z = y

/-- The theorem stating that if f has exactly two zeros, then m = -2 or m = 2 -/
theorem two_zeros_implies_m_values (m : ℝ) :
  has_two_zeros m → m = -2 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_zeros_implies_m_values_l2782_278269


namespace NUMINAMATH_CALUDE_sum_of_coordinates_A_l2782_278273

/-- Given three points A, B, and C in a plane satisfying certain conditions,
    prove that the sum of coordinates of A is 22. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (C.2 - A.2) / (B.2 - A.2) = 1/3 →
  B = (2, 8) →
  C = (5, 11) →
  A.1 + A.2 = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_A_l2782_278273


namespace NUMINAMATH_CALUDE_book_pages_count_l2782_278244

/-- The number of pages read each night -/
def pages_per_night : ℕ := 12

/-- The number of nights needed to finish the book -/
def nights_to_finish : ℕ := 10

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_per_night * nights_to_finish

theorem book_pages_count : total_pages = 120 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l2782_278244


namespace NUMINAMATH_CALUDE_apple_difference_is_two_l2782_278248

/-- The number of apples Jackie has -/
def jackies_apples : ℕ := 10

/-- The number of apples Adam has -/
def adams_apples : ℕ := 8

/-- The difference in apples between Jackie and Adam -/
def apple_difference : ℕ := jackies_apples - adams_apples

theorem apple_difference_is_two : apple_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_is_two_l2782_278248


namespace NUMINAMATH_CALUDE_fish_in_pond_l2782_278250

theorem fish_in_pond (tagged_fish : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  tagged_fish = 60 →
  second_catch = 50 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = tagged_fish / (1500 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_fish_in_pond_l2782_278250


namespace NUMINAMATH_CALUDE_factorization_of_four_a_squared_minus_one_l2782_278270

theorem factorization_of_four_a_squared_minus_one (a : ℝ) : 4 * a^2 - 1 = (2*a - 1) * (2*a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_four_a_squared_minus_one_l2782_278270


namespace NUMINAMATH_CALUDE_equality_of_gcd_lcm_sets_l2782_278223

theorem equality_of_gcd_lcm_sets (a b c : ℕ) :
  ({Nat.gcd a b, Nat.gcd b c, Nat.gcd a c} : Set ℕ) =
  ({Nat.lcm a b, Nat.lcm b c, Nat.lcm a c} : Set ℕ) →
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_equality_of_gcd_lcm_sets_l2782_278223


namespace NUMINAMATH_CALUDE_equation_solution_complex_l2782_278209

theorem equation_solution_complex (a b : ℂ) : 
  a ≠ 0 → 
  a + b ≠ 0 → 
  (a + b) / a = 3 * b / (a + b) → 
  (¬(a.im = 0) ∧ b.im = 0) ∨ (a.im = 0 ∧ ¬(b.im = 0)) ∨ (¬(a.im = 0) ∧ ¬(b.im = 0)) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_complex_l2782_278209


namespace NUMINAMATH_CALUDE_matts_bike_ride_l2782_278265

/-- Given Matt's bike ride scenario, prove the remaining distance after the second sign. -/
theorem matts_bike_ride (total_distance : ℕ) (distance_to_first_sign : ℕ) (distance_between_signs : ℕ)
  (h1 : total_distance = 1000)
  (h2 : distance_to_first_sign = 350)
  (h3 : distance_between_signs = 375) :
  total_distance - (distance_to_first_sign + distance_between_signs) = 275 := by
  sorry

end NUMINAMATH_CALUDE_matts_bike_ride_l2782_278265


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2782_278243

theorem solve_exponential_equation :
  ∃ x : ℝ, (9 : ℝ)^x * (9 : ℝ)^x * (9 : ℝ)^x = (81 : ℝ)^3 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2782_278243


namespace NUMINAMATH_CALUDE_train_length_calculation_l2782_278257

/-- Calculates the length of a train given its speed and time to cross a pole. -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 240 → time_s = 21 → 
  ∃ (length_m : ℝ), abs (length_m - 1400.07) < 0.01 ∧ length_m = speed_kmh * (1000 / 3600) * time_s := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2782_278257


namespace NUMINAMATH_CALUDE_problem_solution_l2782_278275

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2*a| + |x - a|

theorem problem_solution :
  ∀ a : ℝ, a ≠ 0 →
  (∀ x : ℝ, f a x ≥ |x - 2*a| + |x - a|) →
  (∀ x : ℝ, (f 1 x > 2 ↔ x < 1/2 ∨ x > 5/2)) ∧
  (∀ b : ℝ, b ≠ 0 → f a b ≥ f a a) ∧
  (∀ b : ℝ, b ≠ 0 → (f a b = f a a ↔ (2*a - b)*(b - a) ≥ 0 ∨ 2*a - b = 0 ∨ b - a = 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2782_278275


namespace NUMINAMATH_CALUDE_total_payment_after_discounts_l2782_278288

def shirt_price : ℝ := 80
def pants_price : ℝ := 100
def shirt_discount : ℝ := 0.15
def pants_discount : ℝ := 0.10
def coupon_discount : ℝ := 0.05

theorem total_payment_after_discounts :
  let discounted_shirt := shirt_price * (1 - shirt_discount)
  let discounted_pants := pants_price * (1 - pants_discount)
  let total_before_coupon := discounted_shirt + discounted_pants
  let final_amount := total_before_coupon * (1 - coupon_discount)
  final_amount = 150.10 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_after_discounts_l2782_278288


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_l2782_278254

-- Define the piecewise function g(x)
noncomputable def g (x a b : ℝ) : ℝ :=
  if x > 1 then b * x + 1
  else if -3 ≤ x ∧ x ≤ 1 then 2 * x - 4
  else 3 * x - a

-- State the theorem
theorem continuous_piecewise_function (a b : ℝ) :
  Continuous g → a + b = -2 :=
by sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_l2782_278254


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2782_278291

theorem contrapositive_equivalence :
  (∀ x : ℝ, x ≤ -3 → x < 0) ↔ (∀ x : ℝ, x ≥ 0 → x > -3) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2782_278291


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2782_278228

theorem root_sum_theorem (x : ℝ) : 
  (1/x + 1/(x + 4) - 1/(x + 8) - 1/(x + 12) - 1/(x + 16) - 1/(x + 20) + 1/(x + 24) + 1/(x + 28) = 0) →
  (∃ (a b c d : ℕ), 
    (x = -a + Real.sqrt (b + c * Real.sqrt d) ∨ x = -a - Real.sqrt (b + c * Real.sqrt d) ∨
     x = -a + Real.sqrt (b - c * Real.sqrt d) ∨ x = -a - Real.sqrt (b - c * Real.sqrt d)) ∧
    a + b + c + d = 123) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2782_278228


namespace NUMINAMATH_CALUDE_inheritance_calculation_l2782_278212

/-- The inheritance amount in dollars -/
def inheritance : ℝ := 33000

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The state tax rate as a decimal -/
def state_tax_rate : ℝ := 0.15

/-- The additional fee in dollars -/
def additional_fee : ℝ := 50

/-- The total amount paid for taxes and fee in dollars -/
def total_paid : ℝ := 12000

theorem inheritance_calculation :
  federal_tax_rate * inheritance + additional_fee +
  state_tax_rate * ((1 - federal_tax_rate) * inheritance - additional_fee) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l2782_278212


namespace NUMINAMATH_CALUDE_smallest_a_for_quadratic_roots_l2782_278202

theorem smallest_a_for_quadratic_roots (a : ℕ) (b c : ℝ) : 
  (∃ x y : ℝ, 
    x ≠ y ∧ 
    0 < x ∧ x ≤ 1/1000 ∧ 
    0 < y ∧ y ≤ 1/1000 ∧ 
    a * x^2 + b * x + c = 0 ∧ 
    a * y^2 + b * y + c = 0) →
  a ≥ 1001000 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_quadratic_roots_l2782_278202


namespace NUMINAMATH_CALUDE_log_identities_l2782_278262

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem statement
theorem log_identities (a P : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  (log (a^2) P = (log a P) / 2) ∧
  (log (Real.sqrt a) P = 2 * log a P) ∧
  (log (1/a) P = -(log a P)) := by
  sorry

end NUMINAMATH_CALUDE_log_identities_l2782_278262


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2782_278226

theorem rectangle_perimeter (x y : ℝ) (h1 : x^2 + y^2 = 900) (h2 : x * y = 200) :
  2 * (x + y) = 20 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2782_278226


namespace NUMINAMATH_CALUDE_no_solution_l2782_278245

/-- P(n) denotes the greatest prime factor of n -/
def P (n : ℕ) : ℕ := sorry

/-- The theorem states that there are no positive integers n > 1 satisfying both conditions -/
theorem no_solution :
  ¬ ∃ (n : ℕ), n > 1 ∧ (P n : ℝ) = Real.sqrt n ∧ (P (n + 60) : ℝ) = Real.sqrt (n + 60) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l2782_278245


namespace NUMINAMATH_CALUDE_b_performance_conditions_l2782_278224

/-- Represents a person's shooting performance -/
structure ShootingPerformance where
  average : ℝ
  variance : ℝ

/-- The shooting performances of A, C, and D -/
def performances : List ShootingPerformance := [
  ⟨9.7, 0.25⟩,  -- A
  ⟨9.3, 0.28⟩,  -- C
  ⟨9.6, 0.27⟩   -- D
]

/-- B's performance is the best and most stable -/
def b_is_best (m n : ℝ) : Prop :=
  ∀ p ∈ performances, m > p.average ∧ n < p.variance

/-- Theorem stating the conditions for B's performance -/
theorem b_performance_conditions (m n : ℝ) 
  (h : b_is_best m n) : m > 9.7 ∧ n < 0.25 := by
  sorry

#check b_performance_conditions

end NUMINAMATH_CALUDE_b_performance_conditions_l2782_278224


namespace NUMINAMATH_CALUDE_f_inequality_range_l2782_278279

/-- The function f(x) = |x-a| + |x+2| -/
def f (a x : ℝ) : ℝ := |x - a| + |x + 2|

/-- The theorem stating the range of a values for which ∃x₀ ∈ ℝ such that f(x₀) ≤ |2a+1| -/
theorem f_inequality_range (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ ≤ |2*a + 1|) ↔ a ≤ -1 ∨ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_range_l2782_278279


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2782_278201

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if a circle with diameter equal to the distance between its foci
    intersects one of its asymptotes at the point (3,4),
    then the equation of the hyperbola is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ 3^2 + 4^2 = c^2) →
  (∃ (k : ℝ), k = b/a ∧ 4/3 = k) →
  a^2 = 9 ∧ b^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2782_278201


namespace NUMINAMATH_CALUDE_not_divides_power_diff_l2782_278237

theorem not_divides_power_diff (m n : ℕ) : 
  m ≥ 3 → n ≥ 3 → Odd m → Odd n → ¬(2^m - 1 ∣ 3^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_diff_l2782_278237


namespace NUMINAMATH_CALUDE_output_for_three_l2782_278214

/-- Represents the output of the program based on the input x -/
def program_output (x : ℤ) : ℤ :=
  if x < 0 then -1
  else if x = 0 then 0
  else 1

/-- Theorem stating that when x = 3, the program outputs 1 -/
theorem output_for_three : program_output 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_output_for_three_l2782_278214


namespace NUMINAMATH_CALUDE_anita_apples_l2782_278253

/-- The number of apples Anita has, given the number of students and apples per student -/
def total_apples (num_students : ℕ) (apples_per_student : ℕ) : ℕ :=
  num_students * apples_per_student

/-- Theorem: Anita has 360 apples -/
theorem anita_apples : total_apples 60 6 = 360 := by
  sorry

end NUMINAMATH_CALUDE_anita_apples_l2782_278253


namespace NUMINAMATH_CALUDE_square_difference_l2782_278229

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2782_278229


namespace NUMINAMATH_CALUDE_minibuses_needed_l2782_278267

theorem minibuses_needed (students : ℕ) (teacher : ℕ) (capacity : ℕ) : 
  students = 48 → teacher = 1 → capacity = 8 → 
  (students + teacher + capacity - 1) / capacity = 7 := by
  sorry

end NUMINAMATH_CALUDE_minibuses_needed_l2782_278267


namespace NUMINAMATH_CALUDE_inverse_f_at_3_l2782_278298

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the domain of f
def f_domain (x : ℝ) : Prop := -2 ≤ x ∧ x < 0

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ), 
    (∀ x, f_domain x → f_inv (f x) = x) ∧
    (∀ y, (∃ x, f_domain x ∧ f x = y) → f (f_inv y) = y) ∧
    f_inv 3 = -1 :=
sorry

end NUMINAMATH_CALUDE_inverse_f_at_3_l2782_278298


namespace NUMINAMATH_CALUDE_quadratic_root_implies_u_l2782_278293

theorem quadratic_root_implies_u (u : ℝ) : 
  (4 * (((-15 + Real.sqrt 205) / 8) ^ 2) + 15 * ((-15 + Real.sqrt 205) / 8) + u = 0) → 
  u = 5/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_u_l2782_278293


namespace NUMINAMATH_CALUDE_median_and_mode_of_scores_l2782_278210

def student_scores : List Nat := [3, 6, 4, 6, 4, 3, 6, 5, 7]

def median (l : List Nat) : Nat := sorry

def mode (l : List Nat) : Nat := sorry

theorem median_and_mode_of_scores : 
  median student_scores = 5 ∧ mode student_scores = 6 := by sorry

end NUMINAMATH_CALUDE_median_and_mode_of_scores_l2782_278210


namespace NUMINAMATH_CALUDE_divisibility_by_three_l2782_278227

/-- A sequence of integers satisfying the recurrence relation -/
def SatisfiesRecurrence (a : ℕ → ℤ) (k : ℕ+) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n * n = a (n - 1) + n^(k : ℕ)

/-- The main theorem -/
theorem divisibility_by_three (k : ℕ+) (a : ℕ → ℤ) 
  (h : SatisfiesRecurrence a k) : 
  3 ∣ (k : ℤ) - 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l2782_278227


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l2782_278276

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (36 - 9) = Real.sqrt 170 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l2782_278276


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2782_278259

theorem cost_price_calculation (selling_price : ℚ) (profit_percentage : ℚ) : 
  selling_price = 120 → profit_percentage = 25 → 
  ∃ (cost_price : ℚ), cost_price = 96 ∧ selling_price = cost_price * (1 + profit_percentage / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2782_278259


namespace NUMINAMATH_CALUDE_exists_nth_root_product_in_disc_l2782_278219

/-- A closed disc in the complex plane -/
structure ClosedDisc where
  center : ℂ
  radius : ℝ
  radius_nonneg : 0 ≤ radius

/-- A point is in a closed disc if its distance from the center is at most the radius -/
def in_closed_disc (z : ℂ) (D : ClosedDisc) : Prop :=
  Complex.abs (z - D.center) ≤ D.radius

/-- The main theorem -/
theorem exists_nth_root_product_in_disc (D : ClosedDisc) (n : ℕ) (h_n : 0 < n) 
    (z_list : List ℂ) (h_z_list : ∀ z ∈ z_list, in_closed_disc z D) :
    ∃ z : ℂ, in_closed_disc z D ∧ z^n = z_list.prod := by
  sorry

end NUMINAMATH_CALUDE_exists_nth_root_product_in_disc_l2782_278219


namespace NUMINAMATH_CALUDE_final_ratio_is_two_to_one_l2782_278206

/-- Represents the ratio of milk to water in a mixture -/
structure Ratio where
  milk : ℕ
  water : ℕ

/-- Represents a can containing a mixture of milk and water -/
structure Can where
  capacity : ℕ
  current_volume : ℕ
  mixture : Ratio

def add_milk (can : Can) (amount : ℕ) : Can :=
  { can with
    current_volume := can.current_volume + amount
    mixture := Ratio.mk (can.mixture.milk + amount) can.mixture.water
  }

theorem final_ratio_is_two_to_one
  (initial_can : Can)
  (h1 : initial_can.mixture = Ratio.mk 4 3)
  (h2 : initial_can.capacity = 36)
  (h3 : (add_milk initial_can 8).current_volume = initial_can.capacity) :
  (add_milk initial_can 8).mixture = Ratio.mk 2 1 := by
  sorry

end NUMINAMATH_CALUDE_final_ratio_is_two_to_one_l2782_278206


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_less_than_hundred_l2782_278200

theorem greatest_multiple_of_four_less_than_hundred : 
  ∀ n : ℕ, n % 4 = 0 ∧ n < 100 → n ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_less_than_hundred_l2782_278200


namespace NUMINAMATH_CALUDE_complex_sum_as_polar_l2782_278252

open Complex

theorem complex_sum_as_polar : ∃ (r θ : ℝ),
  7 * exp (3 * π * I / 14) - 7 * exp (10 * π * I / 21) = r * exp (θ * I) ∧
  r = Real.sqrt (2 - Real.sqrt 3 / 2) ∧
  θ = 29 * π / 84 + Real.arctan (-2 / (Real.sqrt 3 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_as_polar_l2782_278252


namespace NUMINAMATH_CALUDE_min_buildings_correct_min_buildings_20x20_min_buildings_50x90_l2782_278284

/-- Represents a rectangular city grid -/
structure City where
  rows : ℕ
  cols : ℕ

/-- Calculates the minimum number of buildings after renovation -/
def min_buildings (city : City) : ℕ :=
  ((city.rows * city.cols + 15) / 16 : ℕ)

/-- Theorem: The minimum number of buildings after renovation is correct -/
theorem min_buildings_correct (city : City) :
  min_buildings city = ⌈(city.rows * city.cols : ℚ) / 16⌉ :=
sorry

/-- Corollary: For a 20x20 grid, the minimum number of buildings is 25 -/
theorem min_buildings_20x20 :
  min_buildings { rows := 20, cols := 20 } = 25 :=
sorry

/-- Corollary: For a 50x90 grid, the minimum number of buildings is 282 -/
theorem min_buildings_50x90 :
  min_buildings { rows := 50, cols := 90 } = 282 :=
sorry

end NUMINAMATH_CALUDE_min_buildings_correct_min_buildings_20x20_min_buildings_50x90_l2782_278284


namespace NUMINAMATH_CALUDE_bathroom_floor_space_l2782_278260

/-- Calculates the available floor space in an L-shaped bathroom with a pillar -/
theorem bathroom_floor_space
  (main_width : ℕ) (main_length : ℕ)
  (alcove_width : ℕ) (alcove_depth : ℕ)
  (pillar_width : ℕ) (pillar_length : ℕ)
  (tile_size : ℚ) :
  main_width = 15 →
  main_length = 25 →
  alcove_width = 10 →
  alcove_depth = 8 →
  pillar_width = 3 →
  pillar_length = 5 →
  tile_size = 1/2 →
  (main_width * main_length * tile_size^2 +
   alcove_width * alcove_depth * tile_size^2 -
   pillar_width * pillar_length * tile_size^2) = 110 :=
by sorry

end NUMINAMATH_CALUDE_bathroom_floor_space_l2782_278260


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l2782_278213

/-- The fraction of homes on Gotham Street that are termite-ridden -/
def termite_ridden_fraction : ℚ := 1/3

/-- The fraction of termite-ridden homes that are collapsing -/
def collapsing_fraction : ℚ := 7/10

/-- Theorem: The fraction of homes that are termite-ridden but not collapsing is 1/10 -/
theorem termite_ridden_not_collapsing : 
  termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l2782_278213


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2782_278231

theorem solution_set_inequality (x : ℝ) :
  (x - 1) * |x + 2| ≥ 0 ↔ x ≥ 1 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2782_278231


namespace NUMINAMATH_CALUDE_adult_ticket_price_l2782_278233

theorem adult_ticket_price 
  (child_price : ℕ)
  (total_attendance : ℕ)
  (total_collection : ℕ)
  (children_attendance : ℕ) :
  child_price = 25 →
  total_attendance = 280 →
  total_collection = 140 * 100 →
  children_attendance = 80 →
  (total_attendance - children_attendance) * 60 + children_attendance * child_price = total_collection :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l2782_278233


namespace NUMINAMATH_CALUDE_simplify_tan_product_l2782_278230

theorem simplify_tan_product : (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_tan_product_l2782_278230


namespace NUMINAMATH_CALUDE_circle_on_y_axis_through_point_one_two_l2782_278274

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_on_y_axis_through_point_one_two :
  ∃ (c : Circle),
    c.center.1 = 0 ∧
    c.radius = 1 ∧
    circle_equation c 1 2 ∧
    ∀ (x y : ℝ), circle_equation c x y ↔ x^2 + (y - 2)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_on_y_axis_through_point_one_two_l2782_278274


namespace NUMINAMATH_CALUDE_ship_supplies_l2782_278204

theorem ship_supplies (x : ℝ) : 
  x > 0 →
  (x - 2/5 * x) * (1 - 3/5) = 96 →
  x = 400 :=
by sorry

end NUMINAMATH_CALUDE_ship_supplies_l2782_278204


namespace NUMINAMATH_CALUDE_student_score_problem_l2782_278266

theorem student_score_problem (total_questions : ℕ) (score : ℤ) 
  (h1 : total_questions = 100)
  (h2 : score = 73) :
  ∃ (correct incorrect : ℕ),
    correct + incorrect = total_questions ∧
    (correct : ℤ) - 2 * (incorrect : ℤ) = score ∧
    correct = 91 := by
  sorry

end NUMINAMATH_CALUDE_student_score_problem_l2782_278266


namespace NUMINAMATH_CALUDE_root_implies_constant_value_l2782_278234

theorem root_implies_constant_value (c : ℝ) : 
  ((-5 : ℝ)^2 = c^2) → (c = 5 ∨ c = -5) := by
  sorry

end NUMINAMATH_CALUDE_root_implies_constant_value_l2782_278234


namespace NUMINAMATH_CALUDE_fraction_product_equals_one_l2782_278290

theorem fraction_product_equals_one : 
  (4 + 6 + 8) / (3 + 5 + 7) * (3 + 5 + 7) / (4 + 6 + 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_one_l2782_278290


namespace NUMINAMATH_CALUDE_arithmetic_problem_l2782_278225

theorem arithmetic_problem : 300 + 5 * 8 = 340 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l2782_278225


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2782_278217

theorem polynomial_factorization (m : ℝ) : 
  -4 * m^3 + 4 * m^2 - m = -m * (2*m - 1)^2 := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2782_278217


namespace NUMINAMATH_CALUDE_absent_present_probability_l2782_278299

theorem absent_present_probability (p : ℝ) (h1 : p = 2/30) :
  let q := 1 - p
  2 * (p * q) = 28/225 := by sorry

end NUMINAMATH_CALUDE_absent_present_probability_l2782_278299


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l2782_278241

theorem price_decrease_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 1300)
  (h2 : new_price = 988) :
  (original_price - new_price) / original_price * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l2782_278241


namespace NUMINAMATH_CALUDE_vinay_position_from_right_l2782_278268

/-- Represents the position of a boy in a row. -/
structure Position where
  fromLeft : Nat
  fromRight : Nat
  total : Nat
  valid : fromLeft + fromRight = total + 1

/-- Given the conditions of the problem, calculate Vinay's position. -/
def vinayPosition (totalBoys : Nat) (rajanFromLeft : Nat) (betweenRajanAndVinay : Nat) : Position :=
  { fromLeft := rajanFromLeft + betweenRajanAndVinay + 1,
    fromRight := totalBoys - (rajanFromLeft + betweenRajanAndVinay),
    total := totalBoys,
    valid := by sorry }

/-- The main theorem to be proved. -/
theorem vinay_position_from_right :
  let p := vinayPosition 24 6 8
  p.fromRight = 9 := by sorry

end NUMINAMATH_CALUDE_vinay_position_from_right_l2782_278268


namespace NUMINAMATH_CALUDE_equation_holds_for_negative_eight_l2782_278278

theorem equation_holds_for_negative_eight :
  let t : ℝ := -8
  let f (x : ℝ) : ℝ := (2 / (x + 3)) + (x / (x + 3)) - (4 / (x + 3))
  f t = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_holds_for_negative_eight_l2782_278278


namespace NUMINAMATH_CALUDE_triangle_proof_l2782_278263

-- Define a triangle structure
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the property of being acute
def isAcute (t : Triangle) : Prop :=
  t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90

-- Theorem statement
theorem triangle_proof (t : Triangle) 
  (h1 : t.angle1 = 48)
  (h2 : t.angle2 = 52)
  (h3 : t.angle1 + t.angle2 + t.angle3 = 180) : 
  t.angle3 = 80 ∧ isAcute t := by
  sorry

end NUMINAMATH_CALUDE_triangle_proof_l2782_278263
