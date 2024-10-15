import Mathlib

namespace NUMINAMATH_CALUDE_treehouse_paint_calculation_l2802_280210

/-- The amount of paint needed for a treehouse project, including paint loss. -/
def total_paint_needed (white_paint green_paint brown_paint blue_paint : Real)
  (paint_loss_percentage : Real) (oz_to_liter_conversion : Real) : Real :=
  let total_oz := white_paint + green_paint + brown_paint + blue_paint
  let total_oz_with_loss := total_oz * (1 + paint_loss_percentage)
  total_oz_with_loss * oz_to_liter_conversion

/-- Theorem stating the total amount of paint needed is approximately 2.635 liters. -/
theorem treehouse_paint_calculation :
  let white_paint := 20
  let green_paint := 15
  let brown_paint := 34
  let blue_paint := 12
  let paint_loss_percentage := 0.1
  let oz_to_liter_conversion := 0.0295735
  ∃ ε > 0, |total_paint_needed white_paint green_paint brown_paint blue_paint
    paint_loss_percentage oz_to_liter_conversion - 2.635| < ε :=
by sorry

end NUMINAMATH_CALUDE_treehouse_paint_calculation_l2802_280210


namespace NUMINAMATH_CALUDE_english_to_maths_ratio_l2802_280293

/-- Represents the marks obtained in different subjects -/
structure Marks where
  english : ℕ
  science : ℕ
  maths : ℕ

/-- Represents the ratio between two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of English to Maths marks -/
theorem english_to_maths_ratio (m : Marks) : 
  m.science = 17 ∧ 
  m.english = 3 * m.science ∧ 
  m.english + m.science + m.maths = 170 → 
  ∃ r : Ratio, r.numerator = 1 ∧ r.denominator = 2 ∧ 
    r.numerator * m.maths = r.denominator * m.english :=
by sorry

end NUMINAMATH_CALUDE_english_to_maths_ratio_l2802_280293


namespace NUMINAMATH_CALUDE_bijective_function_decomposition_l2802_280240

theorem bijective_function_decomposition
  (f : ℤ → ℤ) (hf : Function.Bijective f) :
  ∃ (u v : ℤ → ℤ), Function.Bijective u ∧ Function.Bijective v ∧ (∀ x, f x = u x + v x) := by
  sorry

end NUMINAMATH_CALUDE_bijective_function_decomposition_l2802_280240


namespace NUMINAMATH_CALUDE_f_simplification_symmetry_condition_g_maximum_condition_l2802_280272

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x / 2) * Real.cos (x / 2) - 2 * Real.sqrt 3 * Real.sin (x / 2) ^ 2 + Real.sqrt 3

noncomputable def g (x : ℝ) : ℝ := f x + Real.sin x

theorem f_simplification (x : ℝ) : f x = 2 * Real.sin (x + π / 3) := by sorry

theorem symmetry_condition (φ : ℝ) :
  (∃ k : ℤ, π / 3 + φ + π / 3 = k * π) → φ = π / 3 := by sorry

theorem g_maximum_condition (θ : ℝ) :
  (∀ x : ℝ, g x ≤ g θ) → Real.cos θ = Real.sqrt 3 / Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_f_simplification_symmetry_condition_g_maximum_condition_l2802_280272


namespace NUMINAMATH_CALUDE_license_plate_count_l2802_280289

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 5

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 3

/-- The number of distinct letters that are repeated -/
def repeated_letters : ℕ := 2

/-- The number of license plate combinations -/
def license_plate_combinations : ℕ := 7776000

theorem license_plate_count :
  (Nat.choose alphabet_size repeated_letters) *
  (alphabet_size - repeated_letters) *
  (Nat.choose letter_positions repeated_letters) *
  (Nat.choose (letter_positions - repeated_letters) repeated_letters) *
  (Nat.factorial digit_positions) = license_plate_combinations :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l2802_280289


namespace NUMINAMATH_CALUDE_new_average_weight_l2802_280274

theorem new_average_weight (initial_count : ℕ) (initial_average : ℝ) (new_student_weight : ℝ) :
  initial_count = 29 →
  initial_average = 28 →
  new_student_weight = 22 →
  (initial_count * initial_average + new_student_weight) / (initial_count + 1) = 27.8 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l2802_280274


namespace NUMINAMATH_CALUDE_gcd_105_88_l2802_280241

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_88_l2802_280241


namespace NUMINAMATH_CALUDE_find_number_l2802_280245

theorem find_number : ∃ x : ℝ, 0.20 * x + 0.25 * 60 = 23 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2802_280245


namespace NUMINAMATH_CALUDE_smallest_n_and_y_over_x_l2802_280253

theorem smallest_n_and_y_over_x :
  ∃ (n : ℕ+) (x y : ℝ), 
    x > 0 ∧ y > 0 ∧
    (Complex.I : ℂ)^2 = -1 ∧
    (x + 2*y*Complex.I)^(n:ℕ) = (x - 2*y*Complex.I)^(n:ℕ) ∧
    (∀ (m : ℕ+), m < n → ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (a + 2*b*Complex.I)^(m:ℕ) = (a - 2*b*Complex.I)^(m:ℕ)) ∧
    n = 3 ∧
    y / x = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_and_y_over_x_l2802_280253


namespace NUMINAMATH_CALUDE_dice_probability_l2802_280259

def first_die : Finset ℕ := {1, 3, 5, 6}
def second_die : Finset ℕ := {1, 2, 4, 5, 7, 9}

def sum_in_range (x : ℕ) (y : ℕ) : Bool :=
  let sum := x + y
  8 ≤ sum ∧ sum ≤ 10

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (first_die.product second_die).filter (fun (x, y) ↦ sum_in_range x y)

def total_outcomes : ℕ := (first_die.card * second_die.card : ℕ)

theorem dice_probability :
  (favorable_outcomes.card : ℚ) / total_outcomes = 7 / 18 := by
  sorry

#eval favorable_outcomes
#eval total_outcomes

end NUMINAMATH_CALUDE_dice_probability_l2802_280259


namespace NUMINAMATH_CALUDE_ship_speed_problem_l2802_280255

theorem ship_speed_problem (speed_diff : ℝ) (time : ℝ) (final_distance : ℝ) :
  speed_diff = 3 →
  time = 2 →
  final_distance = 174 →
  ∃ (speed1 speed2 : ℝ),
    speed2 = speed1 + speed_diff ∧
    (speed1 * time)^2 + (speed2 * time)^2 = final_distance^2 ∧
    speed1 = 60 ∧
    speed2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_ship_speed_problem_l2802_280255


namespace NUMINAMATH_CALUDE_runner_ends_in_quadrant_A_l2802_280238

/-- Represents the quadrants of the circular track -/
inductive Quadrant
  | A
  | B
  | C
  | D

/-- Represents a point on the circular track -/
structure Point where
  angle : ℝ  -- Angle in radians from the starting point S

/-- The circular track -/
structure Track where
  circumference : ℝ
  start : Point

/-- A runner on the track -/
structure Runner where
  position : Point
  distance_run : ℝ

/-- Function to determine which quadrant a point is in -/
def point_to_quadrant (p : Point) : Quadrant :=
  sorry

/-- Function to update a runner's position after running a certain distance -/
def update_position (r : Runner) (d : ℝ) (t : Track) : Runner :=
  sorry

/-- Main theorem: After running one mile, the runner ends up in quadrant A -/
theorem runner_ends_in_quadrant_A (t : Track) (r : Runner) :
  t.circumference = 60 ∧ 
  r.position = t.start ∧
  (update_position r 5280 t).position = t.start →
  point_to_quadrant ((update_position r 5280 t).position) = Quadrant.A :=
  sorry

end NUMINAMATH_CALUDE_runner_ends_in_quadrant_A_l2802_280238


namespace NUMINAMATH_CALUDE_simplify_fraction_l2802_280235

theorem simplify_fraction (x : ℝ) (h : x ≠ -1) :
  (x + 1) / (x^2 + 2*x + 1) = 1 / (x + 1) := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2802_280235


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2802_280224

theorem complex_equation_solution (z : ℂ) (h : (3 + z) * Complex.I = 1) : z = -3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2802_280224


namespace NUMINAMATH_CALUDE_find_s_value_l2802_280203

/-- Given a relationship between R, S, and T, prove that S = 3/2 when R = 18 and T = 2 -/
theorem find_s_value (k : ℝ) : 
  (2 = k * 1^2 / 8) →  -- When R = 2, S = 1, and T = 8
  (18 = k * S^2 / 2) →  -- When R = 18 and T = 2
  S = 3/2 := by sorry

end NUMINAMATH_CALUDE_find_s_value_l2802_280203


namespace NUMINAMATH_CALUDE_algebraic_expression_proof_l2802_280233

-- Define the condition
theorem algebraic_expression_proof (a b : ℝ) (h : a - b + 3 = Real.sqrt 2) :
  (2*a - 2*b + 6)^4 = 64 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_proof_l2802_280233


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_squares_not_divisible_by_5_or_13_l2802_280229

theorem sum_of_four_consecutive_squares_not_divisible_by_5_or_13 (n : ℤ) :
  ∃ (k : ℤ), k ≠ 0 ∧ ((n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2) % 5 = k ∧
              ((n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2) % 13 = k :=
by sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_squares_not_divisible_by_5_or_13_l2802_280229


namespace NUMINAMATH_CALUDE_cactus_jump_difference_l2802_280258

theorem cactus_jump_difference (num_cacti : ℕ) (total_distance : ℝ) 
  (derek_hops_per_gap : ℕ) (rory_jumps_per_gap : ℕ) 
  (h1 : num_cacti = 31) 
  (h2 : total_distance = 3720) 
  (h3 : derek_hops_per_gap = 30) 
  (h4 : rory_jumps_per_gap = 10) : 
  ∃ (diff : ℝ), abs (diff - 8.27) < 0.01 ∧ 
  diff = (total_distance / ((num_cacti - 1) * rory_jumps_per_gap)) - 
         (total_distance / ((num_cacti - 1) * derek_hops_per_gap)) :=
by sorry

end NUMINAMATH_CALUDE_cactus_jump_difference_l2802_280258


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l2802_280277

/-- The height of a tree that a monkey can climb in 15 hours, 
    given that it hops 3 ft up and slips 2 ft back each hour except for the last hour. -/
def tree_height : ℕ :=
  let hop_distance : ℕ := 3
  let slip_distance : ℕ := 2
  let total_hours : ℕ := 15
  let net_progress_per_hour : ℕ := hop_distance - slip_distance
  let height_before_last_hour : ℕ := net_progress_per_hour * (total_hours - 1)
  height_before_last_hour + hop_distance

theorem monkey_climb_theorem : tree_height = 17 := by
  sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l2802_280277


namespace NUMINAMATH_CALUDE_nested_sqrt_fraction_l2802_280290

/-- Given a real number x satisfying the equation x = 2 + √3 / x,
    prove that 1 / ((x + 2)(x - 3)) = (√3 + 5) / (-22) -/
theorem nested_sqrt_fraction (x : ℝ) (hx : x = 2 + Real.sqrt 3 / x) :
  1 / ((x + 2) * (x - 3)) = (Real.sqrt 3 + 5) / (-22) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_fraction_l2802_280290


namespace NUMINAMATH_CALUDE_uniform_count_l2802_280296

theorem uniform_count (pants_cost shirt_cost tie_cost socks_cost total_spend : ℚ) 
  (h1 : pants_cost = 20)
  (h2 : shirt_cost = 2 * pants_cost)
  (h3 : tie_cost = shirt_cost / 5)
  (h4 : socks_cost = 3)
  (h5 : total_spend = 355) :
  (total_spend / (pants_cost + shirt_cost + tie_cost + socks_cost) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_uniform_count_l2802_280296


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2802_280286

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 + Real.sqrt x) = 4 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2802_280286


namespace NUMINAMATH_CALUDE_at_least_two_equations_have_real_solutions_l2802_280248

theorem at_least_two_equations_have_real_solutions (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  let eq1 := fun x => (x - a) * (x - b) = x - c
  let eq2 := fun x => (x - c) * (x - b) = x - a
  let eq3 := fun x => (x - a) * (x - c) = x - b
  let has_real_solution := fun f => ∃ x : ℝ, f x
  (has_real_solution eq1 ∧ has_real_solution eq2) ∨
  (has_real_solution eq1 ∧ has_real_solution eq3) ∨
  (has_real_solution eq2 ∧ has_real_solution eq3) :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_equations_have_real_solutions_l2802_280248


namespace NUMINAMATH_CALUDE_parallel_vectors_subtraction_l2802_280247

def a : ℝ × ℝ := (-1, -3)
def b (t : ℝ) : ℝ × ℝ := (2, t)

theorem parallel_vectors_subtraction :
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b t →
  a - b t = (-3, -9) := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_subtraction_l2802_280247


namespace NUMINAMATH_CALUDE_angle_ABC_is_30_l2802_280242

-- Define the angles
def angle_CBD : ℝ := 90
def angle_ABD : ℝ := 60

-- Theorem statement
theorem angle_ABC_is_30 :
  ∀ (angle_ABC : ℝ),
  angle_ABD + angle_ABC + angle_CBD = 180 →
  angle_ABC = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_ABC_is_30_l2802_280242


namespace NUMINAMATH_CALUDE_log_identity_l2802_280220

theorem log_identity : Real.log 2 ^ 3 + 3 * Real.log 2 * Real.log 5 + Real.log 5 ^ 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l2802_280220


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2802_280298

/-- Parabola defined by y² = 2x -/
def parabola (x y : ℝ) : Prop := y^2 = 2*x

/-- Line defined by y = -1/2x + b -/
def line (x y b : ℝ) : Prop := y = -1/2*x + b

/-- Point on both parabola and line -/
def intersection_point (x y b : ℝ) : Prop :=
  parabola x y ∧ line x y b

/-- Circle with diameter AB is tangent to x-axis -/
def circle_tangent_to_x_axis (xA yA xB yB : ℝ) : Prop :=
  (yA + yB) / 2 = (xB - xA) / 4

theorem parabola_line_intersection (b : ℝ) :
  (∃ xA yA xB yB : ℝ,
    intersection_point xA yA b ∧
    intersection_point xB yB b ∧
    xA ≠ xB ∧
    circle_tangent_to_x_axis xA yA xB yB) →
  b = -4/5 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2802_280298


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2802_280282

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, y^2 = x^3 + 2*x^2 + 2*x + 1 ↔ (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = -1) ∨ (x = 0 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2802_280282


namespace NUMINAMATH_CALUDE_zhang_bing_special_year_l2802_280264

/-- Given that Zhang Bing was born in 1953, this theorem proves the existence and uniqueness of a year between 1953 and 2023 where his age is both a multiple of 9 and equal to the sum of the digits of that year. -/
theorem zhang_bing_special_year : 
  ∃! Y : ℕ, 1953 < Y ∧ Y < 2023 ∧ 
  (∃ k : ℕ, Y - 1953 = 9 * k) ∧
  (Y - 1953 = (Y / 1000) + ((Y % 1000) / 100) + ((Y % 100) / 10) + (Y % 10)) :=
by sorry

end NUMINAMATH_CALUDE_zhang_bing_special_year_l2802_280264


namespace NUMINAMATH_CALUDE_cross_flag_center_area_ratio_l2802_280214

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  side : ℝ
  crossWidth : ℝ
  crossArea : ℝ
  centerArea : ℝ
  crossSymmetric : Bool
  crossUniformWidth : Bool
  crossAreaRatio : crossArea = 0.49 * side * side

/-- Theorem: If the cross occupies 49% of the flag's area, then the center square occupies 25.14% of the flag's area -/
theorem cross_flag_center_area_ratio (flag : CrossFlag) :
  flag.crossSymmetric ∧ flag.crossUniformWidth →
  flag.centerArea / (flag.side * flag.side) = 0.2514 := by
  sorry

end NUMINAMATH_CALUDE_cross_flag_center_area_ratio_l2802_280214


namespace NUMINAMATH_CALUDE_fitness_center_membership_ratio_l2802_280285

theorem fitness_center_membership_ratio :
  ∀ (f m : ℕ) (f_avg m_avg total_avg : ℚ),
    f_avg = 45 →
    m_avg = 20 →
    total_avg = 28 →
    (f_avg * f + m_avg * m) / (f + m) = total_avg →
    (f : ℚ) / m = 8 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fitness_center_membership_ratio_l2802_280285


namespace NUMINAMATH_CALUDE_faye_pencil_rows_l2802_280263

/-- The number of rows that can be made with a given number of pencils and pencils per row. -/
def number_of_rows (total_pencils : ℕ) (pencils_per_row : ℕ) : ℕ :=
  total_pencils / pencils_per_row

/-- Theorem: Faye can make 6 rows with 30 pencils, placing 5 pencils in each row. -/
theorem faye_pencil_rows : number_of_rows 30 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencil_rows_l2802_280263


namespace NUMINAMATH_CALUDE_history_score_is_84_percent_l2802_280271

/-- Given a student's scores in math and a third subject, along with a desired overall average,
    this function calculates the required score in history. -/
def calculate_history_score (math_score : ℚ) (third_subject_score : ℚ) (desired_average : ℚ) : ℚ :=
  3 * desired_average - math_score - third_subject_score

/-- Theorem stating that given the specific scores and desired average,
    the calculated history score is 84%. -/
theorem history_score_is_84_percent :
  calculate_history_score 72 69 75 = 84 := by
  sorry

#eval calculate_history_score 72 69 75

end NUMINAMATH_CALUDE_history_score_is_84_percent_l2802_280271


namespace NUMINAMATH_CALUDE_point_on_y_axis_has_zero_x_coordinate_l2802_280223

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on the y-axis -/
def lies_on_y_axis (p : Point) : Prop := p.x = 0

/-- Theorem: If a point lies on the y-axis, its x-coordinate is zero -/
theorem point_on_y_axis_has_zero_x_coordinate (m n : ℝ) :
  lies_on_y_axis (Point.mk m n) → m = 0 := by
  sorry


end NUMINAMATH_CALUDE_point_on_y_axis_has_zero_x_coordinate_l2802_280223


namespace NUMINAMATH_CALUDE_salt_production_average_l2802_280257

/-- The salt production problem --/
theorem salt_production_average (initial_production : ℕ) (monthly_increase : ℕ) (months : ℕ) (days_in_year : ℕ) :
  let total_production := initial_production + (monthly_increase * (months * (months - 1)) / 2)
  (total_production : ℚ) / days_in_year = 121.1 := by
  sorry

#check salt_production_average 3000 100 12 365

end NUMINAMATH_CALUDE_salt_production_average_l2802_280257


namespace NUMINAMATH_CALUDE_factorization_sum_l2802_280265

theorem factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 17*x + 72 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 8*x - 63 = (x + b)*(x - c)) →
  a + b + c = 24 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l2802_280265


namespace NUMINAMATH_CALUDE_exist_mutual_wins_l2802_280256

/-- Represents a football tournament --/
structure Tournament :=
  (num_teams : Nat)
  (scores_round1 : Fin num_teams → Nat)
  (scores_round2 : Fin num_teams → Nat)

/-- Properties of the tournament --/
def TournamentProperties (t : Tournament) : Prop :=
  t.num_teams = 20 ∧
  (∀ i j, i ≠ j → t.scores_round1 i ≠ t.scores_round1 j) ∧
  (∃ s, ∀ i, t.scores_round2 i = s)

/-- Theorem stating the existence of two teams that each won one game against the other --/
theorem exist_mutual_wins (t : Tournament) (h : TournamentProperties t) :
  ∃ i j, i ≠ j ∧ 
    t.scores_round2 i - t.scores_round1 i = 2 ∧
    t.scores_round2 j - t.scores_round1 j = 2 :=
by sorry

end NUMINAMATH_CALUDE_exist_mutual_wins_l2802_280256


namespace NUMINAMATH_CALUDE_unique_fixed_point_l2802_280215

noncomputable def F (a b c : ℝ) (x y z : ℝ) : ℝ × ℝ × ℝ :=
  ((Real.sqrt (c^2 + z^2) - z + Real.sqrt (c^2 + y^2) - y) / 2,
   (Real.sqrt (b^2 + z^2) - z + Real.sqrt (b^2 + x^2) - x) / 2,
   (Real.sqrt (a^2 + x^2) - x + Real.sqrt (a^2 + y^2) - y) / 2)

theorem unique_fixed_point (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! p : ℝ × ℝ × ℝ, F a b c p.1 p.2.1 p.2.2 = p ∧ p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_fixed_point_l2802_280215


namespace NUMINAMATH_CALUDE_sandy_shirt_cost_l2802_280295

/-- The amount Sandy spent on clothes, in cents -/
def total_spent : ℕ := 3356

/-- The cost of shorts, in cents -/
def shorts_cost : ℕ := 1399

/-- The cost of jacket, in cents -/
def jacket_cost : ℕ := 743

/-- The cost of shirt, in cents -/
def shirt_cost : ℕ := total_spent - (shorts_cost + jacket_cost)

theorem sandy_shirt_cost : shirt_cost = 1214 := by
  sorry

end NUMINAMATH_CALUDE_sandy_shirt_cost_l2802_280295


namespace NUMINAMATH_CALUDE_feb_7_is_saturday_l2802_280206

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February -/
structure FebruaryDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that February 14 is a Saturday, February 7 is also a Saturday -/
theorem feb_7_is_saturday (feb14 : FebruaryDate) 
    (h14 : feb14.day = 14 ∧ feb14.dayOfWeek = DayOfWeek.Saturday) :
    ∃ (feb7 : FebruaryDate), feb7.day = 7 ∧ feb7.dayOfWeek = DayOfWeek.Saturday := by
  sorry

end NUMINAMATH_CALUDE_feb_7_is_saturday_l2802_280206


namespace NUMINAMATH_CALUDE_rectangle_height_l2802_280243

/-- Given a rectangle with width 32 cm and area divided by diagonal 576 cm², prove its height is 36 cm. -/
theorem rectangle_height (w h : ℝ) (area_div_diagonal : ℝ) : 
  w = 32 → 
  area_div_diagonal = 576 →
  (w * h) / 2 = area_div_diagonal →
  h = 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_height_l2802_280243


namespace NUMINAMATH_CALUDE_divisibility_of_group_difference_l2802_280284

/-- Represents a person in the circle, either a boy or a girl -/
inductive Person
| Boy
| Girl

/-- The circle of people -/
def Circle := List Person

/-- Count the number of groups of 3 consecutive people with exactly one boy -/
def countGroupsWithOneBoy (circle : Circle) : Nat :=
  sorry

/-- Count the number of groups of 3 consecutive people with exactly one girl -/
def countGroupsWithOneGirl (circle : Circle) : Nat :=
  sorry

theorem divisibility_of_group_difference (n : Nat) (circle : Circle) 
    (h1 : n ≥ 3)
    (h2 : circle.length = n) :
  let a := countGroupsWithOneBoy circle
  let b := countGroupsWithOneGirl circle
  3 ∣ (a - b) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_group_difference_l2802_280284


namespace NUMINAMATH_CALUDE_pelicans_remaining_l2802_280217

/-- Represents the number of pelicans in Shark Bite Cove -/
def original_pelicans : ℕ := 30

/-- Represents the number of sharks in Pelican Bay -/
def sharks : ℕ := 60

/-- Represents the fraction of pelicans that moved from Shark Bite Cove to Pelican Bay -/
def moved_fraction : ℚ := 1/3

/-- The theorem stating the number of pelicans remaining in Shark Bite Cove -/
theorem pelicans_remaining : 
  sharks = 2 * original_pelicans ∧ 
  (original_pelicans : ℚ) * (1 - moved_fraction) = 20 :=
sorry

end NUMINAMATH_CALUDE_pelicans_remaining_l2802_280217


namespace NUMINAMATH_CALUDE_train_crossing_time_l2802_280216

/-- Time for a train to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 350 → train_speed_kmh = 144 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2802_280216


namespace NUMINAMATH_CALUDE_arithmetic_progression_equality_l2802_280218

theorem arithmetic_progression_equality (n : ℕ) 
  (a b : Fin n → ℕ) 
  (h_n : n ≥ 2018) 
  (h_distinct_a : ∀ i j : Fin n, i ≠ j → a i ≠ a j)
  (h_distinct_b : ∀ i j : Fin n, i ≠ j → b i ≠ b j)
  (h_bound_a : ∀ i : Fin n, a i ≤ 5*n)
  (h_bound_b : ∀ i : Fin n, b i ≤ 5*n)
  (h_positive_a : ∀ i : Fin n, a i > 0)
  (h_positive_b : ∀ i : Fin n, b i > 0)
  (h_arithmetic : ∃ d : ℚ, ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) - (a j : ℚ) / (b j : ℚ) = (i.val - j.val : ℚ) * d) :
  ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) = (a j : ℚ) / (b j : ℚ) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_equality_l2802_280218


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2802_280204

/-- Given a sum at simple interest for 10 years, if increasing the interest rate by 5%
    results in Rs. 200 more interest, then the original sum is Rs. 2000. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 200 → P = 2000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2802_280204


namespace NUMINAMATH_CALUDE_money_left_is_five_l2802_280269

/-- The cost of the gift in dollars -/
def gift_cost : ℕ := 250

/-- Erika's savings in dollars -/
def erika_savings : ℕ := 155

/-- The cost of the cake in dollars -/
def cake_cost : ℕ := 25

/-- Rick's savings in dollars, defined as half of the gift cost -/
def rick_savings : ℕ := gift_cost / 2

/-- The total savings of Erika and Rick -/
def total_savings : ℕ := erika_savings + rick_savings

/-- The total cost of the gift and cake -/
def total_cost : ℕ := gift_cost + cake_cost

/-- The amount of money left after buying the gift and cake -/
def money_left : ℕ := total_savings - total_cost

theorem money_left_is_five : money_left = 5 := by
  sorry

end NUMINAMATH_CALUDE_money_left_is_five_l2802_280269


namespace NUMINAMATH_CALUDE_student_selection_l2802_280200

theorem student_selection (boys girls : ℕ) (ways : ℕ) : 
  boys = 15 → 
  girls = 10 → 
  ways = 1050 → 
  ways = (girls.choose 1) * (boys.choose 2) →
  1 + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_l2802_280200


namespace NUMINAMATH_CALUDE_min_sum_of_cubes_for_sum_eight_l2802_280202

theorem min_sum_of_cubes_for_sum_eight :
  ∀ x y : ℝ, x + y = 8 →
  x^3 + y^3 ≥ 2 * 4^3 ∧
  (x^3 + y^3 = 2 * 4^3 ↔ x = 4 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_cubes_for_sum_eight_l2802_280202


namespace NUMINAMATH_CALUDE_wills_hourly_rate_l2802_280207

/-- Proof of Will's hourly rate given his work hours and total earnings -/
theorem wills_hourly_rate (monday_hours tuesday_hours total_earnings : ℕ) 
  (h1 : monday_hours = 8)
  (h2 : tuesday_hours = 2)
  (h3 : total_earnings = 80) :
  total_earnings / (monday_hours + tuesday_hours) = 8 := by
  sorry

#check wills_hourly_rate

end NUMINAMATH_CALUDE_wills_hourly_rate_l2802_280207


namespace NUMINAMATH_CALUDE_circle_equation_l2802_280244

theorem circle_equation (x y θ : ℝ) : 
  (x = 3 + 4 * Real.cos θ ∧ y = -2 + 4 * Real.sin θ) → 
  (x - 3)^2 + (y + 2)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l2802_280244


namespace NUMINAMATH_CALUDE_lcm_gcd_relation_l2802_280228

theorem lcm_gcd_relation (n : ℕ+) : 
  Nat.lcm n.val 180 = Nat.gcd n.val 180 + 360 → n.val = 450 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_relation_l2802_280228


namespace NUMINAMATH_CALUDE_original_number_is_75_l2802_280221

theorem original_number_is_75 (x : ℝ) : ((x / 2.5) - 10.5) * 0.3 = 5.85 → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_75_l2802_280221


namespace NUMINAMATH_CALUDE_school_supplies_expenditure_l2802_280266

theorem school_supplies_expenditure (winnings : ℚ) : 
  (winnings / 2 : ℚ) + -- Amount spent on supplies
  ((winnings - winnings / 2) * 3 / 8 : ℚ) + -- Amount saved
  (2500 : ℚ) -- Remaining amount
  = winnings →
  (winnings / 2 : ℚ) = 4000 := by sorry

end NUMINAMATH_CALUDE_school_supplies_expenditure_l2802_280266


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l2802_280276

/-- The width of a rectangular prism with given dimensions and diagonal length -/
theorem rectangular_prism_width (l h d : ℝ) (hl : l = 5) (hh : h = 15) (hd : d = 17) :
  ∃ w : ℝ, w ^ 2 = 39 ∧ d ^ 2 = l ^ 2 + w ^ 2 + h ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l2802_280276


namespace NUMINAMATH_CALUDE_daily_forfeit_is_25_l2802_280222

/-- Calculates the daily forfeit amount for idle days given work conditions --/
def calculate_daily_forfeit (daily_pay : ℕ) (total_days : ℕ) (net_earnings : ℕ) (worked_days : ℕ) : ℕ :=
  let idle_days := total_days - worked_days
  let total_possible_earnings := daily_pay * total_days
  let total_forfeit := total_possible_earnings - net_earnings
  total_forfeit / idle_days

/-- Proves that the daily forfeit amount is 25 dollars given the specific work conditions --/
theorem daily_forfeit_is_25 :
  calculate_daily_forfeit 20 25 450 23 = 25 := by
  sorry

end NUMINAMATH_CALUDE_daily_forfeit_is_25_l2802_280222


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l2802_280209

-- Define a heptagon
def Heptagon : Nat := 7

-- Define the formula for the number of diagonals in a polygon
def numDiagonals (n : Nat) : Nat := n * (n - 3) / 2

-- Theorem: The number of diagonals in a heptagon is 14
theorem heptagon_diagonals : numDiagonals Heptagon = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l2802_280209


namespace NUMINAMATH_CALUDE_mirror_height_for_full_body_view_l2802_280268

/-- 
Theorem: For a person standing upright in front of a vertical mirror, 
the minimum mirror height required to see their full body is exactly 
half of their height.
-/
theorem mirror_height_for_full_body_view 
  (h : ℝ) -- height of the person
  (m : ℝ) -- height of the mirror
  (h_pos : h > 0) -- person's height is positive
  (m_pos : m > 0) -- mirror's height is positive
  (full_view : m ≥ h / 2) -- condition for full body view
  (minimal : ∀ m' : ℝ, m' > 0 → m' < m → ¬(m' ≥ h / 2)) -- m is minimal
  : m = h / 2 := by sorry

end NUMINAMATH_CALUDE_mirror_height_for_full_body_view_l2802_280268


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l2802_280250

theorem units_digit_of_7_power_2023 : ∃ n : ℕ, 7^2023 ≡ 3 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l2802_280250


namespace NUMINAMATH_CALUDE_goats_count_l2802_280252

/-- Represents the number of animals on a farm --/
structure Farm where
  goats : ℕ
  cows : ℕ
  pigs : ℕ
  chickens : ℕ
  ducks : ℕ

/-- Represents the conditions given in the problem --/
def farm_conditions (f : Farm) : Prop :=
  f.pigs = 2 * f.cows ∧
  f.cows = f.goats + 4 ∧
  f.chickens = 3 * f.pigs ∧
  f.ducks = (f.cows + f.goats) / 2 ∧
  f.goats + f.cows + f.pigs + f.chickens + f.ducks = 172

/-- The theorem to be proved --/
theorem goats_count (f : Farm) (h : farm_conditions f) : f.goats = 12 := by
  sorry


end NUMINAMATH_CALUDE_goats_count_l2802_280252


namespace NUMINAMATH_CALUDE_trillion_scientific_notation_l2802_280225

theorem trillion_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), a = 1 ∧ n = 12 ∧ 1000000000000 = a * (10 : ℝ) ^ n :=
by
  sorry

end NUMINAMATH_CALUDE_trillion_scientific_notation_l2802_280225


namespace NUMINAMATH_CALUDE_undefined_rational_function_l2802_280262

theorem undefined_rational_function (x : ℝ) :
  (x^2 - 12*x + 36 = 0) → ¬∃y, y = (3*x^3 + 5) / (x^2 - 12*x + 36) :=
by
  sorry

end NUMINAMATH_CALUDE_undefined_rational_function_l2802_280262


namespace NUMINAMATH_CALUDE_gcd_lcm_8951_4267_l2802_280211

theorem gcd_lcm_8951_4267 : 
  (Nat.gcd 8951 4267 = 1) ∧ 
  (Nat.lcm 8951 4267 = 38212917) := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_8951_4267_l2802_280211


namespace NUMINAMATH_CALUDE_cube_root_eight_plus_negative_two_power_zero_l2802_280246

theorem cube_root_eight_plus_negative_two_power_zero : 
  (8 : ℝ) ^ (1/3) + (-2 : ℝ) ^ 0 = 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_eight_plus_negative_two_power_zero_l2802_280246


namespace NUMINAMATH_CALUDE_composite_29n_plus_11_l2802_280279

theorem composite_29n_plus_11 (n : ℕ) 
  (h1 : ∃ a : ℕ, 3 * n + 1 = a ^ 2) 
  (h2 : ∃ b : ℕ, 10 * n + 1 = b ^ 2) : 
  ¬(Nat.Prime (29 * n + 11)) :=
sorry

end NUMINAMATH_CALUDE_composite_29n_plus_11_l2802_280279


namespace NUMINAMATH_CALUDE_base_five_digits_of_3125_l2802_280236

theorem base_five_digits_of_3125 : ∃ n : ℕ, n = 6 ∧ 
  (∀ k : ℕ, 5^k ≤ 3125 → k + 1 ≤ n) ∧
  (∀ m : ℕ, (∀ k : ℕ, 5^k ≤ 3125 → k + 1 ≤ m) → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_base_five_digits_of_3125_l2802_280236


namespace NUMINAMATH_CALUDE_geometric_series_product_l2802_280249

theorem geometric_series_product (y : ℝ) : y = 9 ↔ 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/y)^n := by sorry

end NUMINAMATH_CALUDE_geometric_series_product_l2802_280249


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l2802_280297

theorem trigonometric_expression_evaluation :
  (Real.tan (150 * π / 180)) * (Real.cos (-210 * π / 180)) * (Real.sin (-420 * π / 180)) /
  ((Real.sin (1050 * π / 180)) * (Real.cos (-600 * π / 180))) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l2802_280297


namespace NUMINAMATH_CALUDE_f_range_f_range_complete_l2802_280273

noncomputable def f (x : ℝ) : ℝ :=
  |Real.sin x| / Real.sin x + Real.cos x / |Real.cos x| + |Real.tan x| / Real.tan x

theorem f_range :
  ∀ x : ℝ, Real.sin x ≠ 0 ∧ Real.cos x ≠ 0 →
    f x = -1 ∨ f x = 3 :=
by sorry

theorem f_range_complete :
  ∃ x y : ℝ, Real.sin x ≠ 0 ∧ Real.cos x ≠ 0 ∧
             Real.sin y ≠ 0 ∧ Real.cos y ≠ 0 ∧
             f x = -1 ∧ f y = 3 :=
by sorry

end NUMINAMATH_CALUDE_f_range_f_range_complete_l2802_280273


namespace NUMINAMATH_CALUDE_coeff_x_squared_expansion_l2802_280283

open Polynomial

/-- The coefficient of x^2 in the expansion of (1-2x)^5(1+3x)^4 is -26 -/
theorem coeff_x_squared_expansion : 
  (coeff ((1 - 2 * X) ^ 5 * (1 + 3 * X) ^ 4) 2) = -26 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x_squared_expansion_l2802_280283


namespace NUMINAMATH_CALUDE_equation_solutions_l2802_280231

theorem equation_solutions :
  (∃ x : ℝ, 2 * (x + 6) = 3 * (x - 1) ∧ x = 15) ∧
  (∃ x : ℝ, (x - 7) / 2 - (1 + x) / 3 = 1 ∧ x = 29) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2802_280231


namespace NUMINAMATH_CALUDE_central_cell_value_l2802_280294

theorem central_cell_value (n : ℕ) (h1 : n = 29) :
  let total_sum := n * (n * (n + 1) / 2)
  let above_diagonal_sum := 3 * ((total_sum - n * (n + 1) / 2) / 2)
  let below_diagonal_sum := (total_sum - n * (n + 1) / 2) / 2
  let diagonal_sum := total_sum - above_diagonal_sum - below_diagonal_sum
  above_diagonal_sum = 3 * below_diagonal_sum →
  diagonal_sum / n = 15 := by
  sorry

#check central_cell_value

end NUMINAMATH_CALUDE_central_cell_value_l2802_280294


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2802_280201

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2802_280201


namespace NUMINAMATH_CALUDE_increase_and_subtract_l2802_280287

theorem increase_and_subtract (initial : ℝ) (increase_percent : ℝ) (subtract_amount : ℝ) : 
  initial = 837 → 
  increase_percent = 135 → 
  subtract_amount = 250 → 
  (initial * (1 + increase_percent / 100) - subtract_amount) = 1717.95 := by
  sorry

end NUMINAMATH_CALUDE_increase_and_subtract_l2802_280287


namespace NUMINAMATH_CALUDE_all_real_roots_condition_l2802_280227

theorem all_real_roots_condition (k : ℝ) : 
  (∀ x : ℂ, x^4 - 4*x^3 + 4*x^2 + k*x - 4 = 0 → x.im = 0) ↔ k = -8 := by
sorry

end NUMINAMATH_CALUDE_all_real_roots_condition_l2802_280227


namespace NUMINAMATH_CALUDE_remaining_movies_to_watch_l2802_280226

theorem remaining_movies_to_watch 
  (total_movies : ℕ) 
  (watched_movies : ℕ) 
  (total_books : ℕ) 
  (read_books : ℕ) 
  (h1 : total_movies = 12) 
  (h2 : watched_movies = 6) 
  (h3 : total_books = 21) 
  (h4 : read_books = 7) 
  (h5 : watched_movies ≤ total_movies) : 
  total_movies - watched_movies = 6 := by
sorry

end NUMINAMATH_CALUDE_remaining_movies_to_watch_l2802_280226


namespace NUMINAMATH_CALUDE_catrionas_aquarium_l2802_280237

/-- The number of goldfish in Catriona's aquarium -/
def num_goldfish : ℕ := 8

/-- The number of angelfish in Catriona's aquarium -/
def num_angelfish : ℕ := num_goldfish + 4

/-- The number of guppies in Catriona's aquarium -/
def num_guppies : ℕ := 2 * num_angelfish

/-- The total number of fish in Catriona's aquarium -/
def total_fish : ℕ := num_goldfish + num_angelfish + num_guppies

theorem catrionas_aquarium : total_fish = 44 := by
  sorry

end NUMINAMATH_CALUDE_catrionas_aquarium_l2802_280237


namespace NUMINAMATH_CALUDE_line_properties_l2802_280208

-- Define the lines
def line (A B C : ℝ) := {(x, y) : ℝ × ℝ | A * x + B * y + C = 0}

-- Define when two lines intersect
def intersect (l1 l2 : Set (ℝ × ℝ)) := ∃ p, p ∈ l1 ∧ p ∈ l2

-- Define when two lines are perpendicular
def perpendicular (A1 B1 A2 B2 : ℝ) := A1 * A2 + B1 * B2 = 0

-- Theorem statement
theorem line_properties (A1 B1 C1 A2 B2 C2 : ℝ) :
  (A1 * B2 - A2 * B1 ≠ 0 → intersect (line A1 B1 C1) (line A2 B2 C2)) ∧
  (perpendicular A1 B1 A2 B2 → 
    ∃ (x1 y1 x2 y2 : ℝ), 
      (x1, y1) ∈ line A1 B1 C1 ∧ 
      (x2, y2) ∈ line A2 B2 C2 ∧ 
      (x2 - x1) * (y2 - y1) = 0) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l2802_280208


namespace NUMINAMATH_CALUDE_gcd_2814_1806_l2802_280291

theorem gcd_2814_1806 : Nat.gcd 2814 1806 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2814_1806_l2802_280291


namespace NUMINAMATH_CALUDE_tommy_balloons_l2802_280251

/-- The number of balloons Tommy initially had -/
def initial_balloons : ℕ := 71

/-- The number of balloons Tommy's mom gave him -/
def mom_balloons : ℕ := 34

/-- The number of balloons Tommy gave to his friends -/
def friend_balloons : ℕ := 15

/-- The number of teddy bears Tommy got after exchanging balloons -/
def teddy_bears : ℕ := 30

/-- The exchange rate of balloons to teddy bears -/
def exchange_rate : ℕ := 3

theorem tommy_balloons : 
  initial_balloons + mom_balloons - friend_balloons = teddy_bears * exchange_rate := by
  sorry

end NUMINAMATH_CALUDE_tommy_balloons_l2802_280251


namespace NUMINAMATH_CALUDE_calculation_proof_l2802_280275

theorem calculation_proof : (3.14 - 1) ^ 0 * (-1/4) ^ (-2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2802_280275


namespace NUMINAMATH_CALUDE_sin_45_is_proposition_l2802_280288

-- Define what a proposition is in this context
def is_proposition (s : String) : Prop := 
  ∃ (truth_value : Bool), (s ≠ "") ∧ (truth_value = true ∨ truth_value = false)

-- State the theorem
theorem sin_45_is_proposition : 
  is_proposition "sin(45°) = 1" := by
  sorry

end NUMINAMATH_CALUDE_sin_45_is_proposition_l2802_280288


namespace NUMINAMATH_CALUDE_taxi_problem_l2802_280280

def taxi_distances : List Int := [9, -3, -5, 4, 8, 6, 3, -6, -4, 10]
def price_per_km : ℝ := 2.4

theorem taxi_problem (distances : List Int) (price : ℝ) 
  (h_distances : distances = taxi_distances) (h_price : price = price_per_km) :
  (distances.sum = 22) ∧ 
  ((distances.map Int.natAbs).sum * price = 139.2) := by
  sorry

end NUMINAMATH_CALUDE_taxi_problem_l2802_280280


namespace NUMINAMATH_CALUDE_problem_statement_l2802_280232

open Real

theorem problem_statement : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 3^x₀ + x₀ = 2016) ∧ 
  ¬(∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, |x| - a*x = |-x| - a*(-x)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2802_280232


namespace NUMINAMATH_CALUDE_banana_proportion_after_adding_l2802_280239

/-- Represents a fruit basket with apples and bananas -/
structure FruitBasket where
  apples : ℕ
  bananas : ℕ

/-- Calculates the fraction of bananas in the basket -/
def bananaProportion (basket : FruitBasket) : ℚ :=
  basket.bananas / (basket.apples + basket.bananas)

/-- The initial basket -/
def initialBasket : FruitBasket := ⟨12, 15⟩

/-- The basket after adding 3 bananas -/
def finalBasket : FruitBasket := ⟨initialBasket.apples, initialBasket.bananas + 3⟩

/-- Theorem stating that the proportion of bananas in the final basket is 3/5 -/
theorem banana_proportion_after_adding : bananaProportion finalBasket = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_banana_proportion_after_adding_l2802_280239


namespace NUMINAMATH_CALUDE_probability_seven_tails_l2802_280219

/-- The probability of flipping exactly k tails in n flips of an unfair coin -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of flipping exactly 7 tails in 10 flips of an unfair coin with 2/3 probability of tails -/
theorem probability_seven_tails : 
  binomial_probability 10 7 (2/3) = 5120/19683 := by
  sorry

end NUMINAMATH_CALUDE_probability_seven_tails_l2802_280219


namespace NUMINAMATH_CALUDE_max_k_is_19_l2802_280281

/-- Represents a two-digit number -/
def TwoDigitNumber (a b : Nat) : Prop :=
  a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9

/-- Represents a three-digit number formed by inserting a digit between two others -/
def ThreeDigitNumber (a c b : Nat) : Prop :=
  a ≠ 0 ∧ a ≤ 9 ∧ c ≤ 9 ∧ b ≤ 9

/-- The value of a two-digit number -/
def twoDigitValue (a b : Nat) : Nat :=
  10 * a + b

/-- The value of a three-digit number -/
def threeDigitValue (a c b : Nat) : Nat :=
  100 * a + 10 * c + b

/-- The theorem stating that the maximum value of k is 19 -/
theorem max_k_is_19 :
  ∀ a b c k : Nat,
  TwoDigitNumber a b →
  ThreeDigitNumber a c b →
  threeDigitValue a c b = k * twoDigitValue a b →
  k ≤ 19 :=
sorry

end NUMINAMATH_CALUDE_max_k_is_19_l2802_280281


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l2802_280212

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  num_tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to a configuration -/
def add_tiles (initial : TileConfiguration) (added_tiles : ℕ) : TileConfiguration :=
  { num_tiles := initial.num_tiles + added_tiles,
    perimeter := initial.perimeter } -- Placeholder, actual calculation would depend on tile placement

/-- The theorem to be proved -/
theorem perimeter_after_adding_tiles :
  ∃ (final : TileConfiguration),
    let initial : TileConfiguration := { num_tiles := 9, perimeter := 16 }
    let with_added_tiles := add_tiles initial 3
    with_added_tiles.perimeter = 18 :=
sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l2802_280212


namespace NUMINAMATH_CALUDE_game_wheel_probability_l2802_280261

theorem game_wheel_probability : 
  ∀ (p_A p_B p_C p_D p_E : ℚ),
    p_A = 2/7 →
    p_B = 1/7 →
    p_C = p_D →
    p_C = p_E →
    p_A + p_B + p_C + p_D + p_E = 1 →
    p_C = 4/21 := by
  sorry

end NUMINAMATH_CALUDE_game_wheel_probability_l2802_280261


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l2802_280260

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 21 cm and height 11 cm is 231 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 21 11 = 231 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l2802_280260


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2802_280299

-- Define the quadratic equation
def quadratic_equation (x a : ℝ) : Prop := x^2 + 3*x - a = 0

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation x a ∧ quadratic_equation y a

-- Theorem statement
theorem quadratic_roots_condition (a : ℝ) :
  has_two_distinct_real_roots a ↔ a > -9/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2802_280299


namespace NUMINAMATH_CALUDE_no_perfect_square_in_sequence_l2802_280205

def sequence_a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => sequence_a n + 2 / sequence_a n

theorem no_perfect_square_in_sequence :
  ∀ n : ℕ, ¬∃ q : ℚ, sequence_a n = q ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_sequence_l2802_280205


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l2802_280254

/-- Given a rectangular prism with dimensions 10 × 5 × 24 inches, 
    prove that a cube with the same volume has a surface area of approximately 678 square inches. -/
theorem cube_surface_area_equal_volume (ε : ℝ) (hε : ε > 0) : ∃ (s : ℝ), 
  s^3 = 10 * 5 * 24 ∧ 
  abs (6 * s^2 - 678) < ε :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l2802_280254


namespace NUMINAMATH_CALUDE_function_properties_l2802_280230

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

theorem function_properties (a b : ℝ) :
  (∀ x ∈ Set.Ioo (-3) 2, f a b x > 0) ∧
  (∀ x ∈ Set.Iic (-3) ∪ Set.Ici 2, f a b x < 0) →
  (∃ a₀ b₀ : ℝ, ∀ x, f a b x = -3 * x^2 - 3 * x + 18) ∧
  (∀ c : ℝ, (∀ x : ℝ, a * x^2 + b * x + c ≤ 0) ↔ c ≤ -25/12) ∧
  (∃ M : ℝ, M = -3 ∧ ∀ x > -1, (f a b x - 21) / (x + 1) ≤ M) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2802_280230


namespace NUMINAMATH_CALUDE_similar_polygons_perimeter_l2802_280270

theorem similar_polygons_perimeter (A₁ A₂ P₁ P₂ : ℝ) : 
  A₁ / A₂ = 1 / 16 →  -- ratio of areas
  P₂ - P₁ = 9 →       -- difference in perimeters
  P₁ = 3 :=           -- perimeter of smaller polygon
by sorry

end NUMINAMATH_CALUDE_similar_polygons_perimeter_l2802_280270


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sequence_angles_l2802_280292

theorem triangle_arithmetic_sequence_angles (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- angles are positive
  A + B + C = 180 →  -- sum of angles in a triangle
  ∃ (d : ℝ), C - B = B - A →  -- arithmetic sequence condition
  B = 60 := by sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sequence_angles_l2802_280292


namespace NUMINAMATH_CALUDE_train_crossing_time_l2802_280213

/-- Given a train crossing two platforms of different lengths, prove the time taken to cross the second platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform1_length platform2_length : ℝ)
  (time1 : ℝ) 
  (h1 : train_length = 30)
  (h2 : platform1_length = 180)
  (h3 : platform2_length = 250)
  (h4 : time1 = 15)
  (h5 : (train_length + platform1_length) / time1 = (train_length + platform2_length) / (20 : ℝ)) :
  (train_length + platform2_length) / ((train_length + platform1_length) / time1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2802_280213


namespace NUMINAMATH_CALUDE_fraction_sum_greater_than_sum_fraction_l2802_280278

theorem fraction_sum_greater_than_sum_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a + 1 / b > 1 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_greater_than_sum_fraction_l2802_280278


namespace NUMINAMATH_CALUDE_probability_divisible_by_five_l2802_280267

/- Define the spinner outcomes -/
def spinner : Finset ℕ := {1, 2, 4, 5}

/- Define a function to check if a number is divisible by 5 -/
def divisible_by_five (n : ℕ) : Bool :=
  n % 5 = 0

/- Define a function to create a three-digit number from three spins -/
def make_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

/- Main theorem -/
theorem probability_divisible_by_five :
  (Finset.filter (fun n => divisible_by_five (make_number n.1 n.2.1 n.2.2))
    (spinner.product (spinner.product spinner))).card /
  (spinner.product (spinner.product spinner)).card = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_divisible_by_five_l2802_280267


namespace NUMINAMATH_CALUDE_book_pages_calculation_l2802_280234

theorem book_pages_calculation (pages_per_day : ℕ) (days_to_finish : ℕ) : 
  pages_per_day = 8 → days_to_finish = 72 → pages_per_day * days_to_finish = 576 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l2802_280234
