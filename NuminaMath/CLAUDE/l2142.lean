import Mathlib

namespace NUMINAMATH_CALUDE_prime_value_theorem_l2142_214264

theorem prime_value_theorem (n : ℕ+) (h : Nat.Prime (n^4 - 16*n^2 + 100)) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_value_theorem_l2142_214264


namespace NUMINAMATH_CALUDE_f_range_and_max_value_l2142_214290

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 1)*x - 3

theorem f_range_and_max_value :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f 2 x ∈ Set.Icc (-21/4 : ℝ) 15) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc (-1 : ℝ) 3, f a x ≤ 1) ∧ 
            (∃ x ∈ Set.Icc (-1 : ℝ) 3, f a x = 1) →
            a = -1/3 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_f_range_and_max_value_l2142_214290


namespace NUMINAMATH_CALUDE_digit_sum_congruence_l2142_214288

/-- The digit sum of n in base r -/
noncomputable def digit_sum (r n : ℕ) : ℕ := sorry

theorem digit_sum_congruence :
  (∀ r > 2, ∃ p : ℕ, Nat.Prime p ∧ ∀ n > 0, digit_sum r n ≡ n [MOD p]) ∧
  (∀ r > 1, ∀ p : ℕ, Nat.Prime p → ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, digit_sum r n ≡ n [MOD p]) :=
sorry

end NUMINAMATH_CALUDE_digit_sum_congruence_l2142_214288


namespace NUMINAMATH_CALUDE_quadrilateral_sum_of_squares_bounds_l2142_214205

/-- Represents a point on the side of a rectangle -/
structure SidePoint where
  side : Fin 4  -- 0: top, 1: right, 2: bottom, 3: left
  position : ℝ
  h_position : 0 ≤ position ∧ position ≤ match side with
    | 0 | 2 => 3  -- top and bottom sides
    | 1 | 3 => 4  -- right and left sides

/-- The quadrilateral formed by four points on the sides of a 3x4 rectangle -/
def Quadrilateral (p₁ p₂ p₃ p₄ : SidePoint) : Prop :=
  p₁.side ≠ p₂.side ∧ p₂.side ≠ p₃.side ∧ p₃.side ≠ p₄.side ∧ p₄.side ≠ p₁.side

/-- The side length of the quadrilateral between two points -/
def sideLength (p₁ p₂ : SidePoint) : ℝ :=
  sorry  -- Definition of side length calculation

/-- The sum of squares of side lengths of the quadrilateral -/
def sumOfSquares (p₁ p₂ p₃ p₄ : SidePoint) : ℝ :=
  (sideLength p₁ p₂)^2 + (sideLength p₂ p₃)^2 + (sideLength p₃ p₄)^2 + (sideLength p₄ p₁)^2

/-- The main theorem -/
theorem quadrilateral_sum_of_squares_bounds
  (p₁ p₂ p₃ p₄ : SidePoint)
  (h : Quadrilateral p₁ p₂ p₃ p₄) :
  25 ≤ sumOfSquares p₁ p₂ p₃ p₄ ∧ sumOfSquares p₁ p₂ p₃ p₄ ≤ 50 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_sum_of_squares_bounds_l2142_214205


namespace NUMINAMATH_CALUDE_overtime_threshold_is_40_l2142_214277

/-- Represents Janet's work and financial situation -/
structure JanetWorkSituation where
  regularRate : ℝ  -- Regular hourly rate
  weeklyHours : ℝ  -- Total weekly work hours
  overtimeMultiplier : ℝ  -- Overtime pay multiplier
  carCost : ℝ  -- Cost of the car
  weeksToSave : ℝ  -- Number of weeks to save for the car

/-- Calculates the weekly earnings given a threshold for overtime hours -/
def weeklyEarnings (j : JanetWorkSituation) (threshold : ℝ) : ℝ :=
  threshold * j.regularRate + (j.weeklyHours - threshold) * j.regularRate * j.overtimeMultiplier

/-- Theorem stating that the overtime threshold is 40 hours -/
theorem overtime_threshold_is_40 (j : JanetWorkSituation) 
    (h1 : j.regularRate = 20)
    (h2 : j.weeklyHours = 52)
    (h3 : j.overtimeMultiplier = 1.5)
    (h4 : j.carCost = 4640)
    (h5 : j.weeksToSave = 4)
    : ∃ (threshold : ℝ), 
      threshold = 40 ∧ 
      weeklyEarnings j threshold ≥ j.carCost / j.weeksToSave ∧
      ∀ t, t > threshold → weeklyEarnings j t < j.carCost / j.weeksToSave :=
by
  sorry

end NUMINAMATH_CALUDE_overtime_threshold_is_40_l2142_214277


namespace NUMINAMATH_CALUDE_equation_solutions_l2142_214203

theorem equation_solutions : 
  ∀ x : ℝ, x * (2 * x - 4) = 3 * (2 * x - 4) ↔ x = 3 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2142_214203


namespace NUMINAMATH_CALUDE_inequality_proof_l2142_214294

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^4 + b^4 > 2*a*b^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2142_214294


namespace NUMINAMATH_CALUDE_y_value_approximation_l2142_214249

noncomputable def x : ℝ := 3.87

theorem y_value_approximation :
  let y := 2 * (Real.log x)^3 - (5 / 3)
  ∃ ε > 0, |y + 1.2613| < ε ∧ ε < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_y_value_approximation_l2142_214249


namespace NUMINAMATH_CALUDE_hyperbola_iff_k_in_range_l2142_214242

/-- A curve is defined by the equation (x^2)/(k+4) + (y^2)/(k-1) = 1 -/
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k + 4) + y^2 / (k - 1) = 1 ∧ (k + 4) * (k - 1) < 0

/-- The range of k values for which the curve represents a hyperbola -/
def hyperbola_k_range : Set ℝ := {k | -4 < k ∧ k < 1}

/-- Theorem stating that the curve represents a hyperbola if and only if k is in the range (-4, 1) -/
theorem hyperbola_iff_k_in_range (k : ℝ) :
  is_hyperbola k ↔ k ∈ hyperbola_k_range :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_iff_k_in_range_l2142_214242


namespace NUMINAMATH_CALUDE_coffee_shrink_problem_l2142_214272

def shrink_ray_effect : ℝ := 0.5

theorem coffee_shrink_problem (num_cups : ℕ) (remaining_coffee : ℝ) 
  (h1 : num_cups = 5)
  (h2 : remaining_coffee = 20) : 
  (remaining_coffee / shrink_ray_effect) / num_cups = 8 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shrink_problem_l2142_214272


namespace NUMINAMATH_CALUDE_line_through_first_and_third_quadrants_l2142_214248

theorem line_through_first_and_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x y : ℝ, y = k * x → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))) → k > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_first_and_third_quadrants_l2142_214248


namespace NUMINAMATH_CALUDE_min_socks_theorem_l2142_214258

/-- Represents a collection of socks with at least 5 different colors -/
structure SockCollection where
  colors : Nat
  min_socks_per_color : Nat
  colors_ge_5 : colors ≥ 5
  min_socks_ge_40 : min_socks_per_color ≥ 40

/-- The smallest number of socks that must be selected to guarantee at least 15 pairs -/
def min_socks_for_15_pairs (sc : SockCollection) : Nat :=
  38

theorem min_socks_theorem (sc : SockCollection) :
  min_socks_for_15_pairs sc = 38 := by
  sorry

#check min_socks_theorem

end NUMINAMATH_CALUDE_min_socks_theorem_l2142_214258


namespace NUMINAMATH_CALUDE_circle_area_doubling_l2142_214207

theorem circle_area_doubling (r : ℝ) (h : r > 0) : 
  π * (2 * r)^2 = 4 * (π * r^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_doubling_l2142_214207


namespace NUMINAMATH_CALUDE_exponent_equality_l2142_214274

theorem exponent_equality (n : ℕ) (x : ℕ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^x) 
  (h2 : n = 27) : 
  x = 28 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l2142_214274


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2142_214231

-- Define the conditions
def p (m : ℝ) : Prop := -2 < m ∧ m < -1

def q (m : ℝ) : Prop := 
  ∃ (x y : ℝ), x^2 / (2 + m) - y^2 / (m + 1) = 1 ∧ 
  2 + m > 0 ∧ m + 1 < 0

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ m : ℝ, q m → p m) ∧ 
  (∃ m : ℝ, p m ∧ ¬q m) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2142_214231


namespace NUMINAMATH_CALUDE_total_shared_amount_l2142_214261

/-- Proves that the total amount shared between x, y, and z is 925, given the specified conditions. -/
theorem total_shared_amount (z : ℚ) (y : ℚ) (x : ℚ) : 
  z = 250 →
  y = 1.2 * z →
  x = 1.25 * y →
  x + y + z = 925 := by
  sorry

end NUMINAMATH_CALUDE_total_shared_amount_l2142_214261


namespace NUMINAMATH_CALUDE_paper_cutting_theorem_l2142_214267

/-- Represents a polygon --/
structure Polygon where
  vertices : ℕ

/-- Represents the state of the paper after cuts --/
structure PaperState where
  polygons : List Polygon
  totalVertices : ℕ

/-- Initial state of the rectangular paper --/
def initialState : PaperState :=
  { polygons := [{ vertices := 4 }], totalVertices := 4 }

/-- Perform a single cut on the paper state --/
def performCut (state : PaperState) : PaperState :=
  { polygons := state.polygons ++ [{ vertices := 3 }],
    totalVertices := state.totalVertices + 2 }

/-- Perform n cuts on the paper state --/
def performCuts (n : ℕ) (state : PaperState) : PaperState :=
  match n with
  | 0 => state
  | n + 1 => performCuts n (performCut state)

/-- The main theorem to prove --/
theorem paper_cutting_theorem :
  (performCuts 100 initialState).totalVertices ≠ 302 :=
by sorry

end NUMINAMATH_CALUDE_paper_cutting_theorem_l2142_214267


namespace NUMINAMATH_CALUDE_log_50_between_integers_l2142_214287

theorem log_50_between_integers : ∃ c d : ℤ, (c : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < (d : ℝ) ∧ c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_50_between_integers_l2142_214287


namespace NUMINAMATH_CALUDE_sum_of_absolute_values_l2142_214211

theorem sum_of_absolute_values (a b : ℤ) : 
  (abs a = 5 ∧ abs b = 3) → 
  (a + b = 8 ∨ a + b = 2 ∨ a + b = -2 ∨ a + b = -8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_absolute_values_l2142_214211


namespace NUMINAMATH_CALUDE_smallest_common_factor_40_90_l2142_214216

theorem smallest_common_factor_40_90 : 
  ∃ (a : ℕ), a > 0 ∧ Nat.gcd a 40 > 1 ∧ Nat.gcd a 90 > 1 ∧ 
  ∀ (b : ℕ), b > 0 → Nat.gcd b 40 > 1 → Nat.gcd b 90 > 1 → a ≤ b :=
by
  use 2
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_40_90_l2142_214216


namespace NUMINAMATH_CALUDE_jack_morning_emails_l2142_214254

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 8

/-- The difference between afternoon and morning emails -/
def email_difference : ℕ := 2

theorem jack_morning_emails : 
  morning_emails = 6 ∧ 
  afternoon_emails = morning_emails + email_difference := by
  sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l2142_214254


namespace NUMINAMATH_CALUDE_midpoint_parallelogram_l2142_214204

/-- A quadrilateral in 2D plane represented by its vertices -/
structure Quadrilateral where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Points that divide the sides of a quadrilateral in ratio r -/
def divisionPoints (q : Quadrilateral) (r : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let (x1, y1) := q.v1
  let (x2, y2) := q.v2
  let (x3, y3) := q.v3
  let (x4, y4) := q.v4
  ( ((x2 - x1) * r + x1, (y2 - y1) * r + y1),
    ((x3 - x2) * r + x2, (y3 - y2) * r + y2),
    ((x4 - x3) * r + x3, (y4 - y3) * r + y3),
    ((x1 - x4) * r + x4, (y1 - y4) * r + y4) )

/-- Check if the quadrilateral formed by the division points is a parallelogram -/
def isParallelogram (points : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) := points
  (x3 - x1 = x4 - x2) ∧ (y3 - y1 = y4 - y2)

/-- The main theorem: only midpoints (r = 1/2) form a parallelogram for all quadrilaterals -/
theorem midpoint_parallelogram (q : Quadrilateral) :
    ∀ r : ℝ, (∀ q' : Quadrilateral, isParallelogram (divisionPoints q' r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_parallelogram_l2142_214204


namespace NUMINAMATH_CALUDE_equation_solution_l2142_214200

theorem equation_solution (x : ℝ) :
  (5.31 * Real.tan (6 * x) * Real.cos (2 * x) - Real.sin (2 * x) - 2 * Real.sin (4 * x) = 0) ↔
  (∃ k : ℤ, x = k * π / 2) ∨ (∃ k : ℤ, x = π / 18 * (6 * k + 1) ∨ x = π / 18 * (6 * k - 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2142_214200


namespace NUMINAMATH_CALUDE_unique_power_inequality_l2142_214295

theorem unique_power_inequality : ∃! (n : ℕ), n > 0 ∧ ∀ (m : ℕ), m > 0 → n^m ≥ m^n :=
by sorry

end NUMINAMATH_CALUDE_unique_power_inequality_l2142_214295


namespace NUMINAMATH_CALUDE_union_A_B_l2142_214214

def A : Set ℝ := {x | (x - 2) / (x + 1) ≤ 0}

def B : Set ℝ := {x | -2 * x^2 + 7 * x + 4 > 0}

theorem union_A_B : A ∪ B = Set.Ioo (-1 : ℝ) 4 := by sorry

end NUMINAMATH_CALUDE_union_A_B_l2142_214214


namespace NUMINAMATH_CALUDE_kelly_games_giveaway_l2142_214226

theorem kelly_games_giveaway (initial_games : ℕ) (remaining_games : ℕ) : 
  initial_games = 50 → remaining_games = 35 → initial_games - remaining_games = 15 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_giveaway_l2142_214226


namespace NUMINAMATH_CALUDE_triangle_side_ratio_bound_l2142_214268

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  height_a : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  height_condition : height_a = a

-- Theorem statement
theorem triangle_side_ratio_bound (t : Triangle) : 
  2 ≤ (t.b / t.c + t.c / t.b) ∧ (t.b / t.c + t.c / t.b) ≤ Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_bound_l2142_214268


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_implies_n_l2142_214265

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the r-th term in the binomial expansion of (√x - 1/2x)^n -/
def coefficient (n r : ℕ) : ℚ :=
  (binomial n r : ℚ) * (-1/2)^r

theorem fourth_term_coefficient_implies_n (n : ℕ) :
  coefficient n 3 = -7 → n = 8 := by sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_implies_n_l2142_214265


namespace NUMINAMATH_CALUDE_min_value_a_l2142_214233

theorem min_value_a (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + a/y ≥ 16/(x+y)) → a ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l2142_214233


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2142_214283

theorem polynomial_evaluation :
  let x : ℤ := -2
  x^4 + x^3 + x^2 + x + 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2142_214283


namespace NUMINAMATH_CALUDE_friends_fireworks_count_l2142_214292

/-- The number of fireworks Henry bought -/
def henrys_fireworks : ℕ := 2

/-- The number of fireworks saved from last year -/
def saved_fireworks : ℕ := 6

/-- The total number of fireworks they have now -/
def total_fireworks : ℕ := 11

/-- The number of fireworks Henry's friend bought -/
def friends_fireworks : ℕ := total_fireworks - (henrys_fireworks + saved_fireworks)

theorem friends_fireworks_count : friends_fireworks = 3 := by
  sorry

end NUMINAMATH_CALUDE_friends_fireworks_count_l2142_214292


namespace NUMINAMATH_CALUDE_james_printing_sheets_l2142_214218

theorem james_printing_sheets (num_books : ℕ) (pages_per_book : ℕ) (pages_per_side : ℕ) :
  num_books = 2 →
  pages_per_book = 600 →
  pages_per_side = 4 →
  (num_books * pages_per_book) / (2 * pages_per_side) = 150 := by
  sorry

end NUMINAMATH_CALUDE_james_printing_sheets_l2142_214218


namespace NUMINAMATH_CALUDE_dart_probability_l2142_214269

theorem dart_probability (square_side : Real) (circle_area : Real) 
  (h1 : square_side = 1)
  (h2 : circle_area = Real.pi / 4) :
  1 - (circle_area / (square_side * square_side)) = 1 - Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_dart_probability_l2142_214269


namespace NUMINAMATH_CALUDE_equation_study_path_and_future_l2142_214293

/-- Represents the steps in the study path of equations -/
inductive StudyPath
  | Definition
  | Solution
  | Solving
  | Application

/-- Represents types of equations that may be studied in the future -/
inductive FutureEquation
  | LinearQuadratic
  | LinearCubic
  | SystemQuadratic

/-- Represents an example of a future equation -/
def futureEquationExample : ℝ → ℝ := fun x => x^3 + 2*x + 1

/-- Theorem stating the study path of equations and future equations to be studied -/
theorem equation_study_path_and_future :
  (∃ (path : List StudyPath), path = [StudyPath.Definition, StudyPath.Solution, StudyPath.Solving, StudyPath.Application]) ∧
  (∃ (future : List FutureEquation), future = [FutureEquation.LinearQuadratic, FutureEquation.LinearCubic, FutureEquation.SystemQuadratic]) ∧
  (∃ (x : ℝ), futureEquationExample x = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_study_path_and_future_l2142_214293


namespace NUMINAMATH_CALUDE_system_equiv_line_l2142_214221

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x - 2*y = 1 ∧ x^3 - 6*x*y - 8*y^3 = 1

/-- The line representing the solution -/
def solution_line (x y : ℝ) : Prop :=
  y = (x - 1) / 2

/-- Theorem stating that the system of equations is equivalent to the solution line -/
theorem system_equiv_line : 
  ∀ x y : ℝ, system x y ↔ solution_line x y :=
sorry

end NUMINAMATH_CALUDE_system_equiv_line_l2142_214221


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2142_214278

theorem sum_of_reciprocals (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^4 = 1 →
  ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) : ℂ) = 4 / ω^2 →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) : ℝ) = -1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2142_214278


namespace NUMINAMATH_CALUDE_survey_result_l2142_214279

theorem survey_result (total : ℕ) (thought_diseases : ℕ) (said_rabies : ℕ) :
  (thought_diseases : ℚ) / total = 3 / 4 →
  (said_rabies : ℚ) / thought_diseases = 1 / 2 →
  said_rabies = 18 →
  total = 48 :=
by sorry

end NUMINAMATH_CALUDE_survey_result_l2142_214279


namespace NUMINAMATH_CALUDE_height_prediction_at_10_l2142_214262

/-- Represents a linear regression model for height based on age -/
structure HeightModel where
  slope : ℝ
  intercept : ℝ

/-- Predicts the height for a given age using the model -/
def predict_height (model : HeightModel) (age : ℝ) : ℝ :=
  model.slope * age + model.intercept

/-- Theorem stating that the predicted height at age 10 is approximately 145.83cm -/
theorem height_prediction_at_10 (model : HeightModel)
  (h_slope : model.slope = 7.19)
  (h_intercept : model.intercept = 73.93) :
  ∃ ε > 0, |predict_height model 10 - 145.83| < ε :=
sorry

#check height_prediction_at_10

end NUMINAMATH_CALUDE_height_prediction_at_10_l2142_214262


namespace NUMINAMATH_CALUDE_only_four_and_eight_satisfy_l2142_214270

/-- A natural number is a proper divisor of another natural number if it divides the number, is greater than 1, and is not equal to the number itself. -/
def IsProperDivisor (d n : ℕ) : Prop :=
  d ∣ n ∧ d > 1 ∧ d ≠ n

/-- The set of proper divisors of a natural number. -/
def ProperDivisors (n : ℕ) : Set ℕ :=
  {d | IsProperDivisor d n}

/-- The property that all proper divisors of n, when increased by 1, form the set of proper divisors of m. -/
def SatisfiesProperty (n m : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), f = (· + 1) ∧
  (ProperDivisors m) = f '' (ProperDivisors n)

/-- The theorem stating that only 4 and 8 satisfy the given property. -/
theorem only_four_and_eight_satisfy :
  ∀ n : ℕ, (∃ m : ℕ, SatisfiesProperty n m) ↔ n = 4 ∨ n = 8 := by
  sorry


end NUMINAMATH_CALUDE_only_four_and_eight_satisfy_l2142_214270


namespace NUMINAMATH_CALUDE_min_sum_factors_l2142_214251

theorem min_sum_factors (n : ℕ) (hn : n = 2025) :
  ∃ (a b : ℕ), a * b = n ∧ a > 0 ∧ b > 0 ∧
  (∀ (x y : ℕ), x * y = n → x > 0 → y > 0 → a + b ≤ x + y) ∧
  a + b = 90 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_factors_l2142_214251


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_unknown_blanket_rate_eq_175_l2142_214220

/-- The unknown rate of two blankets given the following conditions:
    - 1 blanket purchased at Rs. 100
    - 5 blankets purchased at Rs. 150 each
    - 2 blankets purchased at an unknown rate
    - The average price of all blankets is Rs. 150
-/
theorem unknown_blanket_rate : ℕ :=
  let num_blankets_1 : ℕ := 1
  let price_1 : ℕ := 100
  let num_blankets_2 : ℕ := 5
  let price_2 : ℕ := 150
  let num_blankets_3 : ℕ := 2
  let total_blankets : ℕ := num_blankets_1 + num_blankets_2 + num_blankets_3
  let average_price : ℕ := 150
  let total_cost : ℕ := average_price * total_blankets
  let known_cost : ℕ := num_blankets_1 * price_1 + num_blankets_2 * price_2
  let unknown_rate : ℕ := (total_cost - known_cost) / num_blankets_3
  unknown_rate

theorem unknown_blanket_rate_eq_175 : unknown_blanket_rate = 175 := by
  sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_unknown_blanket_rate_eq_175_l2142_214220


namespace NUMINAMATH_CALUDE_third_median_length_l2142_214212

/-- A triangle with two known medians and area -/
structure TriangleWithMedians where
  -- The length of the first median
  median1 : ℝ
  -- The length of the second median
  median2 : ℝ
  -- The area of the triangle
  area : ℝ

/-- The theorem stating the length of the third median -/
theorem third_median_length (t : TriangleWithMedians) 
  (h1 : t.median1 = 5)
  (h2 : t.median2 = 4)
  (h3 : t.area = 6 * Real.sqrt 5) :
  ∃ (median3 : ℝ), median3 = 3 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_third_median_length_l2142_214212


namespace NUMINAMATH_CALUDE_power_of_three_equality_l2142_214206

theorem power_of_three_equality : 3^1999 - 3^1998 - 3^1997 + 3^1996 = 16 * 3^1996 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equality_l2142_214206


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l2142_214297

theorem complex_square_one_plus_i : (1 + Complex.I) ^ 2 = 2 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l2142_214297


namespace NUMINAMATH_CALUDE_top_square_is_14_l2142_214253

/-- Represents a position in the 4x4 grid -/
structure Position :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the 4x4 grid -/
def Grid := Fin 4 → Fin 4 → ℕ

/-- Initial configuration of the grid -/
def initialGrid : Grid :=
  λ i j => i.val * 4 + j.val + 1

/-- Fold right half over left half -/
def foldRight (g : Grid) : Grid :=
  λ i j => g i (3 - j)

/-- Fold bottom half over top half -/
def foldBottom (g : Grid) : Grid :=
  λ i j => g (3 - i) j

/-- Apply all folding operations -/
def applyFolds (g : Grid) : Grid :=
  foldRight (foldBottom (foldRight g))

/-- The position of the top square after folding -/
def topPosition : Position :=
  ⟨0, 0⟩

/-- Theorem: After folding, the top square was originally numbered 14 -/
theorem top_square_is_14 :
  applyFolds initialGrid topPosition.row topPosition.col = 14 := by
  sorry

end NUMINAMATH_CALUDE_top_square_is_14_l2142_214253


namespace NUMINAMATH_CALUDE_smallest_c_for_all_real_domain_l2142_214257

theorem smallest_c_for_all_real_domain : ∃ c : ℤ, 
  (∀ x : ℝ, (x^2 + c*x + 15 ≠ 0)) ∧ 
  (∀ k : ℤ, k < c → ∃ x : ℝ, x^2 + k*x + 15 = 0) ∧
  c = -7 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_all_real_domain_l2142_214257


namespace NUMINAMATH_CALUDE_stating_orthogonal_parallelepiped_angle_properties_l2142_214273

/-- 
Represents the angles formed between the diagonal and the edges of an orthogonal parallelepiped.
-/
structure ParallelepipedAngles where
  α₁ : ℝ
  α₂ : ℝ
  α₃ : ℝ

/-- 
Theorem stating the properties of angles in an orthogonal parallelepiped.
-/
theorem orthogonal_parallelepiped_angle_properties (angles : ParallelepipedAngles) :
  Real.sin angles.α₁ ^ 2 + Real.sin angles.α₂ ^ 2 + Real.sin angles.α₃ ^ 2 = 1 ∧
  Real.cos angles.α₁ ^ 2 + Real.cos angles.α₂ ^ 2 + Real.cos angles.α₃ ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_stating_orthogonal_parallelepiped_angle_properties_l2142_214273


namespace NUMINAMATH_CALUDE_sisters_get_five_bars_l2142_214275

/-- Calculates the number of granola bars each sister receives when splitting the remaining bars evenly -/
def granola_bars_per_sister (total : ℕ) (set_aside : ℕ) (traded : ℕ) (num_sisters : ℕ) : ℕ :=
  (total - set_aside - traded) / num_sisters

/-- Proves that given the specific conditions, each sister receives 5 granola bars -/
theorem sisters_get_five_bars :
  let total := 20
  let set_aside := 7
  let traded := 3
  let num_sisters := 2
  granola_bars_per_sister total set_aside traded num_sisters = 5 := by
  sorry

#eval granola_bars_per_sister 20 7 3 2

end NUMINAMATH_CALUDE_sisters_get_five_bars_l2142_214275


namespace NUMINAMATH_CALUDE_point_on_y_axis_l2142_214291

/-- A point M with coordinates (a+2, 2a-5) lies on the y-axis. -/
theorem point_on_y_axis (a : ℝ) : (a + 2 = 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l2142_214291


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_l2142_214208

-- Define the parabola
def parabola (a x : ℝ) : ℝ := x^3 - 3*a*x + a^3

-- Define the line
def line (a x : ℝ) : ℝ := x + 2*a

-- Define the derivative of the parabola with respect to x
def parabola_derivative (a x : ℝ) : ℝ := 3*x^2 - 3*a

-- Theorem statement
theorem line_through_parabola_vertex :
  ∃! (a : ℝ), ∃ (x : ℝ),
    (parabola_derivative a x = 0) ∧
    (line a x = parabola a x) :=
sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_l2142_214208


namespace NUMINAMATH_CALUDE_smallest_n_not_divisible_l2142_214284

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_n_not_divisible : ∃ (n : ℕ), n = 124 ∧ 
  ¬(factorial 1999 ∣ 34^n * factorial n) ∧ 
  ∀ (m : ℕ), m < n → (factorial 1999 ∣ 34^m * factorial m) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_not_divisible_l2142_214284


namespace NUMINAMATH_CALUDE_inequality_solution_l2142_214271

theorem inequality_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) (h4 : x ≠ 4) :
  (1 / (x - 1) - 3 / (x - 2) + 3 / (x - 3) - 1 / (x - 4) < 1 / 24) ↔
  (x < -2 ∨ (-1 < x ∧ x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 6)) :=
by sorry


end NUMINAMATH_CALUDE_inequality_solution_l2142_214271


namespace NUMINAMATH_CALUDE_annie_distance_equals_22_l2142_214282

def base_fare : ℚ := 2.5
def per_mile_rate : ℚ := 0.25
def mike_distance : ℚ := 42
def annie_toll : ℚ := 5

theorem annie_distance_equals_22 :
  ∃ (annie_distance : ℚ),
    base_fare + per_mile_rate * mike_distance =
    base_fare + annie_toll + per_mile_rate * annie_distance ∧
    annie_distance = 22 := by
  sorry

end NUMINAMATH_CALUDE_annie_distance_equals_22_l2142_214282


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l2142_214246

/-- Given two points A (-2, 0) and B (0, 4), prove that the equation x + 2y - 3 = 0
    represents the perpendicular bisector of the line segment AB. -/
theorem perpendicular_bisector_equation (A B : ℝ × ℝ) (h1 : A = (-2, 0)) (h2 : B = (0, 4)) :
  ∀ (x y : ℝ), (x + 2*y - 3 = 0) ↔ 
    (x - (-2))^2 + (y - 0)^2 = (x - 0)^2 + (y - 4)^2 ∧ 
    (x + 1) * (4 - 0) + (y - 2) * (0 - (-2)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l2142_214246


namespace NUMINAMATH_CALUDE_largest_package_size_l2142_214236

theorem largest_package_size (mary_markers luis_markers ali_markers : ℕ) 
  (h1 : mary_markers = 36)
  (h2 : luis_markers = 45)
  (h3 : ali_markers = 75) :
  Nat.gcd mary_markers (Nat.gcd luis_markers ali_markers) = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l2142_214236


namespace NUMINAMATH_CALUDE_big_eight_football_league_games_l2142_214296

theorem big_eight_football_league_games (num_divisions : Nat) (teams_per_division : Nat) : 
  num_divisions = 3 → 
  teams_per_division = 4 → 
  (num_divisions * teams_per_division * (teams_per_division - 1) + 
   num_divisions * teams_per_division * (num_divisions - 1) * teams_per_division / 2) = 228 := by
  sorry

#check big_eight_football_league_games

end NUMINAMATH_CALUDE_big_eight_football_league_games_l2142_214296


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l2142_214241

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a

theorem max_value_implies_a_equals_one :
  (∃ (M : ℝ), M = 1 ∧ ∀ x ∈ Set.Icc 0 2, f a x ≤ M) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l2142_214241


namespace NUMINAMATH_CALUDE_total_cards_packed_l2142_214280

/-- The number of cards in a standard playing card deck -/
def playing_cards_per_deck : ℕ := 52

/-- The number of cards in a Pinochle deck -/
def pinochle_cards_per_deck : ℕ := 48

/-- The number of cards in a Tarot deck -/
def tarot_cards_per_deck : ℕ := 78

/-- The number of cards in an Uno deck -/
def uno_cards_per_deck : ℕ := 108

/-- The number of playing card decks Elijah packed -/
def playing_card_decks : ℕ := 6

/-- The number of Pinochle decks Elijah packed -/
def pinochle_decks : ℕ := 4

/-- The number of Tarot decks Elijah packed -/
def tarot_decks : ℕ := 2

/-- The number of Uno decks Elijah packed -/
def uno_decks : ℕ := 3

/-- Theorem stating the total number of cards Elijah packed -/
theorem total_cards_packed : 
  playing_card_decks * playing_cards_per_deck + 
  pinochle_decks * pinochle_cards_per_deck + 
  tarot_decks * tarot_cards_per_deck + 
  uno_decks * uno_cards_per_deck = 984 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_packed_l2142_214280


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2142_214281

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0)
  (h2 : a 1 = 2 * a 8 - 3 * a 4)
  (h3 : ∀ n : ℕ, a (n + 1) = a n + d) :
  let S : ℕ → ℝ := λ n ↦ (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2
  S 8 / S 16 = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2142_214281


namespace NUMINAMATH_CALUDE_student_distribution_l2142_214202

theorem student_distribution (total : ℕ) (h_total : total > 0) :
  let third_year := (30 : ℕ) * total / 100
  let not_second_year := (90 : ℕ) * total / 100
  let second_year := total - not_second_year
  let not_third_year := total - third_year
  (second_year : ℚ) / not_third_year = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_student_distribution_l2142_214202


namespace NUMINAMATH_CALUDE_log_problem_l2142_214259

theorem log_problem (x k : ℝ) 
  (h1 : Real.log x * (Real.log 10 / Real.log k) = 4)
  (h2 : k^2 = 100) : 
  x = 10000 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2142_214259


namespace NUMINAMATH_CALUDE_parabola_coefficients_l2142_214263

/-- A parabola with vertex at (-2, 5) passing through (2, 9) has coefficients a = 1/4, b = 1, c = 6 -/
theorem parabola_coefficients :
  ∀ (a b c : ℝ),
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (5 = a * (-2)^2 + b * (-2) + c) →
  (∀ x : ℝ, a * (x + 2)^2 + 5 = a * x^2 + b * x + c) →
  (9 = a * 2^2 + b * 2 + c) →
  (a = 1/4 ∧ b = 1 ∧ c = 6) :=
by sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l2142_214263


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l2142_214276

theorem quadratic_roots_to_coefficients :
  ∀ (p q : ℝ),
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I : ℂ) ^ 2 + p * (2 + Complex.I) + q = 0 →
  (2 - Complex.I : ℂ) ^ 2 + p * (2 - Complex.I) + q = 0 →
  p = -4 ∧ q = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l2142_214276


namespace NUMINAMATH_CALUDE_puppies_brought_in_l2142_214215

-- Define the given conditions
def initial_puppies : ℕ := 5
def adopted_per_day : ℕ := 8
def days_to_adopt_all : ℕ := 5

-- Define the theorem
theorem puppies_brought_in :
  ∃ (brought_in : ℕ), 
    initial_puppies + brought_in = adopted_per_day * days_to_adopt_all :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_puppies_brought_in_l2142_214215


namespace NUMINAMATH_CALUDE_circle_tangent_line_equation_non_intersecting_line_condition_intersecting_line_orthogonal_condition_l2142_214223

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y + 3 * Real.sqrt 2 + 1 = 0

-- Define the non-intersecting line
def non_intersecting_line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the intersecting line
def intersecting_line (m : ℝ) (x y : ℝ) : Prop := y = x + m

theorem circle_tangent_line_equation :
  ∀ x y : ℝ, circle_C x y ↔ (x - 1)^2 + (y + 2)^2 = 9 := by sorry

theorem non_intersecting_line_condition :
  ∀ k : ℝ, (∀ x y : ℝ, ¬(circle_C x y ∧ non_intersecting_line k x y)) ↔ (0 < k ∧ k < 3/4) := by sorry

theorem intersecting_line_orthogonal_condition :
  ∀ m : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
    intersecting_line m x₁ y₁ ∧ intersecting_line m x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0) ↔ (m = 1 ∨ m = -4) := by sorry

end NUMINAMATH_CALUDE_circle_tangent_line_equation_non_intersecting_line_condition_intersecting_line_orthogonal_condition_l2142_214223


namespace NUMINAMATH_CALUDE_three_true_propositions_l2142_214219

theorem three_true_propositions : 
  (¬∀ (a b c : ℝ), a > b → a * c < b * c) ∧ 
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b : ℝ), a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) ∧
  (∀ (a b : ℝ), a > b ∧ 1 / a > 1 / b → a > 0 ∧ b < 0) :=
by sorry

end NUMINAMATH_CALUDE_three_true_propositions_l2142_214219


namespace NUMINAMATH_CALUDE_g_sum_equal_164_l2142_214209

def g (x : ℝ) : ℝ := 2 * x^6 - 5 * x^4 + 7 * x^2 + 6

theorem g_sum_equal_164 (h : g 15 = 82) : g 15 + g (-15) = 164 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_equal_164_l2142_214209


namespace NUMINAMATH_CALUDE_pick_two_different_suits_standard_deck_l2142_214289

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    h_total := rfl }

/-- The number of ways to pick two cards from different suits -/
def pick_two_different_suits (d : Deck) : Nat :=
  d.total_cards * (d.total_cards - d.cards_per_suit)

theorem pick_two_different_suits_standard_deck :
  pick_two_different_suits standard_deck = 2028 := by
  sorry

#eval pick_two_different_suits standard_deck

end NUMINAMATH_CALUDE_pick_two_different_suits_standard_deck_l2142_214289


namespace NUMINAMATH_CALUDE_complex_product_one_plus_i_one_minus_i_l2142_214252

theorem complex_product_one_plus_i_one_minus_i : (1 + Complex.I) * (1 - Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_one_plus_i_one_minus_i_l2142_214252


namespace NUMINAMATH_CALUDE_asterisk_replacement_l2142_214224

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 20) * (x / 80) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l2142_214224


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2142_214256

/-- The area of the shaded region in a square with circles at its vertices -/
theorem shaded_area_square_with_circles (s : ℝ) (r : ℝ) (h1 : s = 12) (h2 : r = 2) :
  s^2 - (4 * (π / 2 * r^2) + 4 * (r^2 / 2)) = 136 - 2 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2142_214256


namespace NUMINAMATH_CALUDE_vector_on_line_and_parallel_l2142_214234

def line_param (t : ℝ) : ℝ × ℝ := (3 * t + 1, 2 * t + 3)

def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_on_line_and_parallel : 
  ∃ t : ℝ, line_param t = (16, 32/3) ∧ 
  is_parallel (16, 32/3) (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_and_parallel_l2142_214234


namespace NUMINAMATH_CALUDE_failed_students_l2142_214255

/-- The number of students who failed an examination, given the total number of students and the percentage who passed. -/
theorem failed_students (total : ℕ) (pass_percent : ℚ) : 
  total = 700 → pass_percent = 35 / 100 → 
  (total : ℚ) * (1 - pass_percent) = 455 := by
  sorry

end NUMINAMATH_CALUDE_failed_students_l2142_214255


namespace NUMINAMATH_CALUDE_abs_gt_one_iff_square_minus_one_gt_zero_l2142_214222

theorem abs_gt_one_iff_square_minus_one_gt_zero :
  ∀ x : ℝ, |x| > 1 ↔ x^2 - 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_gt_one_iff_square_minus_one_gt_zero_l2142_214222


namespace NUMINAMATH_CALUDE_problem_solution_l2142_214266

theorem problem_solution : 
  (-(3^3) * (-1/3) + |(-2)| / ((-1/2)^2) = 17) ∧ 
  (7 - 12 * (2/3 - 3/4 + 5/6) = -2) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2142_214266


namespace NUMINAMATH_CALUDE_cid_earnings_l2142_214232

/-- Represents the earnings from Cid's mechanic shop --/
def mechanic_earnings (oil_change_price : ℕ) (repair_price : ℕ) (car_wash_price : ℕ)
  (oil_changes : ℕ) (repairs : ℕ) (car_washes : ℕ) : ℕ :=
  oil_change_price * oil_changes + repair_price * repairs + car_wash_price * car_washes

/-- Theorem stating that Cid's earnings are $475 given the specific prices and services --/
theorem cid_earnings : 
  mechanic_earnings 20 30 5 5 10 15 = 475 := by
  sorry

end NUMINAMATH_CALUDE_cid_earnings_l2142_214232


namespace NUMINAMATH_CALUDE_min_seats_for_adjacent_seating_l2142_214230

/-- Represents a seating arrangement in a row of seats. -/
structure SeatingArrangement where
  total_seats : ℕ
  min_gap : ℕ
  occupied_seats : ℕ

/-- Checks if a seating arrangement is valid according to the rules. -/
def is_valid_arrangement (sa : SeatingArrangement) : Prop :=
  sa.total_seats = 150 ∧ sa.min_gap = 2 ∧ sa.occupied_seats ≤ sa.total_seats

/-- Checks if the next person must sit next to someone in the given arrangement. -/
def forces_adjacent_seating (sa : SeatingArrangement) : Prop :=
  ∀ (new_seat : ℕ), new_seat ≤ sa.total_seats →
    (∃ (occupied : ℕ), occupied ≤ sa.total_seats ∧
      (new_seat = occupied + 1 ∨ new_seat = occupied - 1))

/-- The main theorem stating the minimum number of occupied seats. -/
theorem min_seats_for_adjacent_seating :
  ∃ (sa : SeatingArrangement),
    is_valid_arrangement sa ∧
    forces_adjacent_seating sa ∧
    sa.occupied_seats = 74 ∧
    (∀ (sa' : SeatingArrangement),
      is_valid_arrangement sa' ∧
      forces_adjacent_seating sa' →
      sa'.occupied_seats ≥ 74) :=
  sorry

end NUMINAMATH_CALUDE_min_seats_for_adjacent_seating_l2142_214230


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_fourths_l2142_214286

theorem opposite_of_negative_three_fourths :
  ∀ x : ℚ, x + (-3/4) = 0 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_fourths_l2142_214286


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l2142_214235

theorem volleyball_team_selection (total_players : Nat) (quadruplets : Nat) (starters : Nat) :
  total_players = 18 →
  quadruplets = 4 →
  starters = 8 →
  (total_players.choose (starters - quadruplets)) = 1001 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l2142_214235


namespace NUMINAMATH_CALUDE_sin_72_cos_18_plus_cos_72_sin_18_l2142_214244

theorem sin_72_cos_18_plus_cos_72_sin_18 : 
  Real.sin (72 * π / 180) * Real.cos (18 * π / 180) + 
  Real.cos (72 * π / 180) * Real.sin (18 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_72_cos_18_plus_cos_72_sin_18_l2142_214244


namespace NUMINAMATH_CALUDE_birthday_problem_l2142_214250

/-- The number of trainees -/
def n : ℕ := 62

/-- The number of days in a year -/
def d : ℕ := 365

/-- The probability of at least two trainees sharing a birthday -/
noncomputable def prob_shared_birthday : ℝ :=
  1 - (d.factorial / (d - n).factorial : ℝ) / d ^ n

theorem birthday_problem :
  ∃ (p : ℝ), prob_shared_birthday = p ∧ p > 0.9959095 ∧ p < 0.9959096 :=
sorry

end NUMINAMATH_CALUDE_birthday_problem_l2142_214250


namespace NUMINAMATH_CALUDE_equation_solution_l2142_214240

theorem equation_solution : ∃! x : ℝ, Real.sqrt (7 * x - 3) + Real.sqrt (x^3 - 1) = 3 :=
  by sorry

end NUMINAMATH_CALUDE_equation_solution_l2142_214240


namespace NUMINAMATH_CALUDE_factorization_equality_l2142_214243

theorem factorization_equality (a x y : ℝ) : a*x^2 + 2*a*x*y + a*y^2 = a*(x+y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2142_214243


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l2142_214239

theorem quadratic_two_real_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 3*x + m = 0 ∧ y^2 + 3*y + m = 0) ↔ m ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l2142_214239


namespace NUMINAMATH_CALUDE_oil_ratio_proof_l2142_214228

theorem oil_ratio_proof (small_tank_capacity large_tank_capacity initial_large_tank_oil additional_oil_needed : ℕ) 
  (h1 : small_tank_capacity = 4000)
  (h2 : large_tank_capacity = 20000)
  (h3 : initial_large_tank_oil = 3000)
  (h4 : additional_oil_needed = 4000)
  (h5 : initial_large_tank_oil + (small_tank_capacity - (small_tank_capacity - x)) + additional_oil_needed = large_tank_capacity / 2)
  : (small_tank_capacity - (small_tank_capacity - x)) / small_tank_capacity = 3 / 4 := by
  sorry

#check oil_ratio_proof

end NUMINAMATH_CALUDE_oil_ratio_proof_l2142_214228


namespace NUMINAMATH_CALUDE_ellipse_equation_proof_l2142_214247

def original_ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

def new_ellipse (x y : ℝ) : Prop := x^2/15 + y^2/10 = 1

def same_foci (e1 e2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ c : ℝ, (∀ x y : ℝ, e1 x y ↔ (x - c)^2/(9 - 4) + y^2/4 = 1) ∧
           (∀ x y : ℝ, e2 x y ↔ (x - c)^2/(15 - 10) + y^2/10 = 1)

theorem ellipse_equation_proof :
  (new_ellipse 3 (-2)) ∧
  (same_foci original_ellipse new_ellipse) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_proof_l2142_214247


namespace NUMINAMATH_CALUDE_g_zero_value_l2142_214299

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the relationship between h, f, and g
axiom h_eq_f_mul_g : h = f * g

-- Define the constant term of f
axiom f_const_term : f.coeff 0 = 6

-- Define the constant term of h
axiom h_const_term : h.coeff 0 = -18

-- Theorem to prove
theorem g_zero_value : g.coeff 0 = -3 := by sorry

end NUMINAMATH_CALUDE_g_zero_value_l2142_214299


namespace NUMINAMATH_CALUDE_number_of_observations_l2142_214245

theorem number_of_observations (original_mean new_mean : ℝ) 
  (original_value new_value : ℝ) (n : ℕ) : 
  original_mean = 36 → 
  new_mean = 36.5 → 
  original_value = 23 → 
  new_value = 44 → 
  n * original_mean + (new_value - original_value) = n * new_mean → 
  n = 42 := by
sorry

end NUMINAMATH_CALUDE_number_of_observations_l2142_214245


namespace NUMINAMATH_CALUDE_roots_and_m_value_l2142_214229

theorem roots_and_m_value (a b c m : ℝ) : 
  (a + b = 4 ∧ a * b = m) →  -- roots of x^2 - 4x + m = 0
  (b + c = 8 ∧ b * c = 5 * m) →  -- roots of x^2 - 8x + 5m = 0
  m = 0 ∨ m = 3 := by
sorry

end NUMINAMATH_CALUDE_roots_and_m_value_l2142_214229


namespace NUMINAMATH_CALUDE_prime_roots_equation_l2142_214227

theorem prime_roots_equation (p q : ℕ) : 
  (∃ x y : ℕ, Prime x ∧ Prime y ∧ 
   x ≠ y ∧
   (p * x^2 - q * x + 1985 = 0) ∧ 
   (p * y^2 - q * y + 1985 = 0)) →
  12 * p^2 + q = 414 := by
sorry

end NUMINAMATH_CALUDE_prime_roots_equation_l2142_214227


namespace NUMINAMATH_CALUDE_complement_of_M_l2142_214237

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}

theorem complement_of_M : (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2142_214237


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l2142_214238

theorem ceiling_sum_sqrt : ⌈Real.sqrt 5⌉ + ⌈Real.sqrt 50⌉ + ⌈Real.sqrt 500⌉ + ⌈Real.sqrt 1000⌉ = 66 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l2142_214238


namespace NUMINAMATH_CALUDE_milk_buckets_l2142_214210

theorem milk_buckets (bucket_capacity : ℝ) (total_milk : ℝ) : 
  bucket_capacity = 15 → total_milk = 147 → ⌈total_milk / bucket_capacity⌉ = 10 := by
  sorry

end NUMINAMATH_CALUDE_milk_buckets_l2142_214210


namespace NUMINAMATH_CALUDE_sleepy_squirrel_stockpile_l2142_214260

/-- The number of busy squirrels -/
def busy_squirrels : ℕ := 2

/-- The number of nuts stockpiled per day by each busy squirrel -/
def nuts_per_busy_squirrel : ℕ := 30

/-- The number of days the squirrels have been stockpiling -/
def days_stockpiling : ℕ := 40

/-- The total number of nuts in Mason's car -/
def total_nuts : ℕ := 3200

/-- The number of nuts stockpiled per day by the sleepy squirrel -/
def sleepy_squirrel_nuts : ℕ := 20

theorem sleepy_squirrel_stockpile :
  busy_squirrels * nuts_per_busy_squirrel * days_stockpiling + 
  sleepy_squirrel_nuts * days_stockpiling = total_nuts :=
by sorry

end NUMINAMATH_CALUDE_sleepy_squirrel_stockpile_l2142_214260


namespace NUMINAMATH_CALUDE_ratio_problem_l2142_214298

theorem ratio_problem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_ratio : ∃ (k : ℝ), a = 6*k ∧ b = 3*k ∧ c = k) :
  3 * b^2 / (2 * a^2 + b * c) = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2142_214298


namespace NUMINAMATH_CALUDE_super_ball_distance_l2142_214213

def bounce_height (initial_height : ℝ) (bounce_factor : ℝ) (n : ℕ) : ℝ :=
  initial_height * (bounce_factor ^ n)

def total_distance (initial_height : ℝ) (bounce_factor : ℝ) (num_bounces : ℕ) : ℝ :=
  let descents := initial_height + (Finset.sum (Finset.range num_bounces) (λ i => bounce_height initial_height bounce_factor i))
  let ascents := Finset.sum (Finset.range (num_bounces + 1)) (λ i => bounce_height initial_height bounce_factor i)
  descents + ascents

theorem super_ball_distance :
  total_distance 20 0.6 4 = 69.632 := by
  sorry

end NUMINAMATH_CALUDE_super_ball_distance_l2142_214213


namespace NUMINAMATH_CALUDE_min_m_for_log_triangle_l2142_214217

/-- The minimum value of M such that for any right-angled triangle with sides a, b, c > M,
    the logarithms of these sides also form a triangle. -/
theorem min_m_for_log_triangle : ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ (a b c : ℝ), a > M → b > M → c > M →
    a^2 + b^2 = c^2 →
    Real.log a + Real.log b > Real.log c) ∧
  (∀ (M' : ℝ), M' < M →
    ∃ (a b c : ℝ), a > M' ∧ b > M' ∧ c > M' ∧
      a^2 + b^2 = c^2 ∧
      Real.log a + Real.log b ≤ Real.log c) :=
by sorry

end NUMINAMATH_CALUDE_min_m_for_log_triangle_l2142_214217


namespace NUMINAMATH_CALUDE_intersection_range_chord_length_l2142_214225

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The line equation -/
def line (x y m : ℝ) : Prop := y = x + m

/-- The range of m for which the line intersects the ellipse -/
theorem intersection_range (m : ℝ) : 
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 := by sorry

/-- The length of the chord when the line passes through (1,0) -/
theorem chord_length : 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line x₁ y₁ (-1) ∧ line x₂ y₂ (-1) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = (4 / 3) * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_intersection_range_chord_length_l2142_214225


namespace NUMINAMATH_CALUDE_apple_lovers_problem_l2142_214201

theorem apple_lovers_problem (total_apples : ℕ) (initial_per_person : ℕ) (decrease : ℕ) 
  (h1 : total_apples = 1430)
  (h2 : initial_per_person = 22)
  (h3 : decrease = 9) :
  ∃ (initial_people new_people : ℕ),
    initial_people * initial_per_person = total_apples ∧
    (initial_people + new_people) * (initial_per_person - decrease) = total_apples ∧
    new_people = 45 := by
  sorry

end NUMINAMATH_CALUDE_apple_lovers_problem_l2142_214201


namespace NUMINAMATH_CALUDE_angle_between_diagonals_is_133_l2142_214285

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the angles of the quadrilateral
def angle_ABC (q : Quadrilateral) : ℝ := 116
def angle_ADC (q : Quadrilateral) : ℝ := 64
def angle_CAB (q : Quadrilateral) : ℝ := 35
def angle_CAD (q : Quadrilateral) : ℝ := 52

-- Define the angle between diagonals subtended by side AB
def angle_between_diagonals (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem angle_between_diagonals_is_133 (q : Quadrilateral) : 
  angle_between_diagonals q = 133 := by sorry

end NUMINAMATH_CALUDE_angle_between_diagonals_is_133_l2142_214285
