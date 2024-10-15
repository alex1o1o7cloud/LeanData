import Mathlib

namespace NUMINAMATH_CALUDE_abc_unique_solution_l2415_241531

/-- Represents a base-7 number with two digits --/
def Base7TwoDigit (a b : ℕ) : ℕ := 7 * a + b

/-- Converts a three-digit decimal number to its numeric value --/
def ThreeDigitToNum (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem abc_unique_solution :
  ∀ A B C : ℕ,
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A < 7 → B < 7 → C < 7 →
    A ≠ B → B ≠ C → A ≠ C →
    Base7TwoDigit A B + C = Base7TwoDigit C 0 →
    Base7TwoDigit A B + Base7TwoDigit B A = Base7TwoDigit B 6 →
    ThreeDigitToNum A B C = 425 :=
by sorry

end NUMINAMATH_CALUDE_abc_unique_solution_l2415_241531


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2415_241594

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (x + 8) = 10 → x = 92 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2415_241594


namespace NUMINAMATH_CALUDE_ninas_run_l2415_241530

theorem ninas_run (x : ℝ) : x + x + 0.67 = 0.83 → x = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_ninas_run_l2415_241530


namespace NUMINAMATH_CALUDE_movie_shelf_problem_l2415_241580

/-- The minimum number of additional movies needed to satisfy the conditions -/
def additional_movies_needed (current_movies : ℕ) (num_shelves : ℕ) : ℕ :=
  let target := num_shelves * (current_movies / num_shelves + 1)
  target - current_movies

theorem movie_shelf_problem :
  let current_movies := 9
  let num_shelves := 2
  let result := additional_movies_needed current_movies num_shelves
  (result = 1 ∧
   (current_movies + result) % 2 = 0 ∧
   (current_movies + result) / num_shelves % 2 = 1 ∧
   ∀ (shelf : ℕ), shelf < num_shelves →
     (current_movies + result) / num_shelves = (current_movies + result - shelf * ((current_movies + result) / num_shelves)) / (num_shelves - shelf)) :=
by sorry

end NUMINAMATH_CALUDE_movie_shelf_problem_l2415_241580


namespace NUMINAMATH_CALUDE_terrys_spending_l2415_241518

/-- Terry's spending problem -/
theorem terrys_spending (monday : ℕ) : 
  monday = 6 →
  let tuesday := 2 * monday
  let wednesday := 2 * (monday + tuesday)
  monday + tuesday + wednesday = 54 := by
  sorry

end NUMINAMATH_CALUDE_terrys_spending_l2415_241518


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2415_241535

theorem sum_of_fractions : (3 : ℚ) / 7 + (5 : ℚ) / 14 = (11 : ℚ) / 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2415_241535


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l2415_241576

theorem simplify_sqrt_sum : 
  Real.sqrt 0.5 + Real.sqrt (0.5 + 1.5) + Real.sqrt (0.5 + 1.5 + 2.5) + 
  Real.sqrt (0.5 + 1.5 + 2.5 + 3.5) = Real.sqrt 0.5 + 3 * Real.sqrt 2 + Real.sqrt 4.5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l2415_241576


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l2415_241578

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y ≥ 9) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ Real.sqrt 2 = Real.sqrt (4^x * 2^y) ∧ 2/x + 1/y = 9) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l2415_241578


namespace NUMINAMATH_CALUDE_solution_to_inequalities_l2415_241566

theorem solution_to_inequalities :
  let x : ℚ := -1/3
  let y : ℚ := 2/3
  (11 * x^2 + 8 * x * y + 8 * y^2 ≤ 3) ∧ (x - 4 * y ≤ -3) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_inequalities_l2415_241566


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l2415_241556

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its division -/
structure Plywood where
  length : ℝ
  width : ℝ
  num_pieces : ℕ

/-- Represents a way of cutting the plywood -/
structure CutMethod where
  piece : Rectangle
  is_valid : Bool

theorem plywood_cut_perimeter_difference 
  (p : Plywood) 
  (h1 : p.length = 10 ∧ p.width = 5)
  (h2 : p.num_pieces = 5) :
  ∃ (m1 m2 : CutMethod), 
    m1.is_valid ∧ m2.is_valid ∧ 
    perimeter m1.piece - perimeter m2.piece = 8 ∧
    ∀ (m : CutMethod), m.is_valid → 
      perimeter m.piece ≤ perimeter m1.piece ∧
      perimeter m.piece ≥ perimeter m2.piece := by
  sorry


end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l2415_241556


namespace NUMINAMATH_CALUDE_no_win_prob_at_least_two_no_win_prob_l2415_241522

-- Define the probability of winning for a single bottle
def win_prob : ℚ := 1/6

-- Define the number of students
def num_students : ℕ := 3

-- Theorem 1: Probability that none of the three students win a prize
theorem no_win_prob : 
  (1 - win_prob) ^ num_students = 125/216 := by sorry

-- Theorem 2: Probability that at least two of the three students do not win a prize
theorem at_least_two_no_win_prob : 
  1 - (Nat.choose num_students 2 * win_prob^2 * (1 - win_prob) + win_prob^num_students) = 25/27 := by sorry

end NUMINAMATH_CALUDE_no_win_prob_at_least_two_no_win_prob_l2415_241522


namespace NUMINAMATH_CALUDE_max_terms_of_arithmetic_sequence_l2415_241550

/-- An arithmetic sequence with common difference 4 and real-valued terms -/
def ArithmeticSequence (a₁ : ℝ) (n : ℕ) : ℕ → ℝ :=
  fun k => a₁ + (k - 1) * 4

/-- The sum of terms from the second to the nth term -/
def SumOfRemainingTerms (a₁ : ℝ) (n : ℕ) : ℝ :=
  (n - 1) * (a₁ + 2 * n)

/-- The condition that the square of the first term plus the sum of remaining terms does not exceed 100 -/
def SequenceCondition (a₁ : ℝ) (n : ℕ) : Prop :=
  a₁^2 + SumOfRemainingTerms a₁ n ≤ 100

theorem max_terms_of_arithmetic_sequence :
  ∀ a₁ : ℝ, ∀ n : ℕ, SequenceCondition a₁ n → n ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_terms_of_arithmetic_sequence_l2415_241550


namespace NUMINAMATH_CALUDE_triangle_lines_theorem_l2415_241536

/-- Triangle ABC with vertices A(-3,5), B(5,7), and C(5,1) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The given triangle -/
def ABC : Triangle := { A := (-3, 5), B := (5, 7), C := (5, 1) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The median line on side AB -/
def medianAB : LineEquation := { a := 5, b := 4, c := -29 }

/-- The line through A with equal x-axis and y-axis intercepts -/
def lineA : LineEquation := { a := 1, b := 1, c := -2 }

theorem triangle_lines_theorem (t : Triangle) (m : LineEquation) (l : LineEquation) : 
  t = ABC → m = medianAB → l = lineA → True := by sorry

end NUMINAMATH_CALUDE_triangle_lines_theorem_l2415_241536


namespace NUMINAMATH_CALUDE_race_length_is_1000_l2415_241548

/-- The length of a race given Aubrey's and Violet's positions -/
def race_length (violet_distance_covered : ℕ) (violet_distance_to_finish : ℕ) : ℕ :=
  violet_distance_covered + violet_distance_to_finish

/-- Theorem stating that the race length is 1000 meters -/
theorem race_length_is_1000 :
  race_length 721 279 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_race_length_is_1000_l2415_241548


namespace NUMINAMATH_CALUDE_triangle_side_b_l2415_241564

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_side_b (t : Triangle) 
  (h1 : t.a = Real.sqrt 3)
  (h2 : t.A = 60 * π / 180)
  (h3 : t.C = 75 * π / 180)
  : t.b = Real.sqrt 2 := by
  sorry

-- Note: We use radians for angles in Lean, so we convert degrees to radians

end NUMINAMATH_CALUDE_triangle_side_b_l2415_241564


namespace NUMINAMATH_CALUDE_seating_arrangements_l2415_241597

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem seating_arrangements (n : ℕ) (k : ℕ) (h : n = 10 ∧ k = 3) :
  factorial n - factorial (n - k + 1) * factorial k = 3598560 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2415_241597


namespace NUMINAMATH_CALUDE_volume_right_triangular_prism_l2415_241555

/-- The volume of a right triangular prism given its lateral face areas and lateral edge length -/
theorem volume_right_triangular_prism
  (M N P l : ℝ)
  (hM : M > 0)
  (hN : N > 0)
  (hP : P > 0)
  (hl : l > 0) :
  let V := (1 / (4 * l)) * Real.sqrt ((N + M + P) * (N + P - M) * (N + M - P) * (M + P - N))
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    M = a * l ∧
    N = b * l ∧
    P = c * l ∧
    V = (1 / 2) * l * Real.sqrt ((-a + b + c) * (a - b + c) * (a + b - c) * (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_volume_right_triangular_prism_l2415_241555


namespace NUMINAMATH_CALUDE_angle_between_lines_l2415_241572

theorem angle_between_lines (r₁ r₂ r₃ : ℝ) (shaded_ratio : ℝ) :
  r₁ = 3 ∧ r₂ = 2 ∧ r₃ = 1 ∧ shaded_ratio = 8/13 →
  ∃ θ : ℝ, 
    θ > 0 ∧ 
    θ < π/2 ∧
    (6 * θ + 4 * π = 24 * π / 7) ∧
    θ = π/7 :=
sorry

end NUMINAMATH_CALUDE_angle_between_lines_l2415_241572


namespace NUMINAMATH_CALUDE_min_sum_distances_on_BC_l2415_241593

/-- Four distinct points on a line -/
structure FourPoints where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  ordered : A < B ∧ B < C ∧ C < D

/-- Sum of distances from a point X to A, B, C, and D -/
def sumOfDistances (fp : FourPoints) (X : ℝ) : ℝ :=
  |X - fp.A| + |X - fp.B| + |X - fp.C| + |X - fp.D|

/-- The point that minimizes the sum of distances is on the segment BC -/
theorem min_sum_distances_on_BC (fp : FourPoints) :
  ∃ (X : ℝ), fp.B ≤ X ∧ X ≤ fp.C ∧
  ∀ (Y : ℝ), sumOfDistances fp X ≤ sumOfDistances fp Y :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_on_BC_l2415_241593


namespace NUMINAMATH_CALUDE_find_sets_A_and_B_l2415_241532

def I : Set ℕ := {x | x ≤ 8 ∧ x > 0}

theorem find_sets_A_and_B 
  (h1 : A ∪ (I \ B) = {1, 3, 4, 5, 6, 7})
  (h2 : (I \ A) ∪ B = {1, 2, 4, 5, 6, 8})
  (h3 : (I \ A) ∩ (I \ B) = {1, 5, 6}) :
  A = {3, 4, 7} ∧ B = {2, 4, 8} := by
sorry

end NUMINAMATH_CALUDE_find_sets_A_and_B_l2415_241532


namespace NUMINAMATH_CALUDE_stairs_calculation_l2415_241586

/-- The number of stairs run up and down one way during a football team's exercise routine. -/
def stairs_one_way : ℕ := 32

/-- The number of times players run up and down the bleachers. -/
def num_runs : ℕ := 40

/-- The number of calories burned per stair. -/
def calories_per_stair : ℕ := 2

/-- The total number of calories burned during the exercise. -/
def total_calories_burned : ℕ := 5120

/-- Theorem stating that the number of stairs run up and down one way is 32,
    given the conditions of the exercise routine. -/
theorem stairs_calculation :
  stairs_one_way = 32 ∧
  num_runs * (2 * stairs_one_way) * calories_per_stair = total_calories_burned :=
by sorry

end NUMINAMATH_CALUDE_stairs_calculation_l2415_241586


namespace NUMINAMATH_CALUDE_min_balls_to_draw_l2415_241544

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  white : ℕ
  black : ℕ

/-- The minimum number of balls needed to guarantee a specific count of one color -/
def minBallsForGuarantee (counts : BallCounts) (targetCount : ℕ) : ℕ :=
  (min counts.red (targetCount - 1)) +
  (min counts.green (targetCount - 1)) +
  (min counts.yellow (targetCount - 1)) +
  (min counts.blue (targetCount - 1)) +
  (min counts.white (targetCount - 1)) +
  (min counts.black (targetCount - 1)) + 1

/-- Theorem stating the minimum number of balls to draw for the given problem -/
theorem min_balls_to_draw (counts : BallCounts)
    (h_red : counts.red = 35)
    (h_green : counts.green = 25)
    (h_yellow : counts.yellow = 22)
    (h_blue : counts.blue = 15)
    (h_white : counts.white = 14)
    (h_black : counts.black = 12) :
    minBallsForGuarantee counts 18 = 93 := by
  sorry


end NUMINAMATH_CALUDE_min_balls_to_draw_l2415_241544


namespace NUMINAMATH_CALUDE_min_value_and_max_t_l2415_241529

-- Define the function f
def f (a b x : ℝ) : ℝ := |x + a| + |2*x - b|

-- State the theorem
theorem min_value_and_max_t (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (m : ℝ), m = 1 ∧ ∀ x, f a b x ≥ m) →
  (2*a + b = 2) ∧
  (∀ t, (a + 2*b ≥ t*a*b) → t ≤ 9/2) ∧
  (∃ t₀, t₀ = 9/2 ∧ a + 2*b ≥ t₀*a*b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_max_t_l2415_241529


namespace NUMINAMATH_CALUDE_all_pollywogs_gone_pollywogs_present_before_44_l2415_241540

/-- Represents the number of pollywogs in the pond after a given number of days -/
def pollywogs_remaining (days : ℕ) : ℕ :=
  if days ≤ 20 then
    2400 - 60 * days
  else
    2400 - 60 * 20 - 50 * (days - 20)

/-- The theorem states that after 44 days, no pollywogs remain in the pond -/
theorem all_pollywogs_gone : pollywogs_remaining 44 = 0 := by
  sorry

/-- The theorem states that before 44 days, there are still pollywogs in the pond -/
theorem pollywogs_present_before_44 (d : ℕ) (h : d < 44) : pollywogs_remaining d > 0 := by
  sorry

end NUMINAMATH_CALUDE_all_pollywogs_gone_pollywogs_present_before_44_l2415_241540


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l2415_241573

theorem greatest_integer_fraction_inequality : 
  (∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l2415_241573


namespace NUMINAMATH_CALUDE_rani_cycling_speed_difference_l2415_241553

/-- Rani's cycling speed as a girl in miles per minute -/
def girl_speed : ℚ := 20 / (2 * 60 + 45)

/-- Rani's cycling speed as an older woman in miles per minute -/
def woman_speed : ℚ := 12 / (3 * 60)

/-- The difference in minutes per mile between Rani's cycling speed as an older woman and as a girl -/
def speed_difference : ℚ := (1 / woman_speed) - (1 / girl_speed)

theorem rani_cycling_speed_difference :
  speed_difference = 6.75 := by sorry

end NUMINAMATH_CALUDE_rani_cycling_speed_difference_l2415_241553


namespace NUMINAMATH_CALUDE_given_expression_is_proper_algebraic_notation_l2415_241560

/-- A predicate that determines if an expression meets the requirements for algebraic notation -/
def is_proper_algebraic_notation (expression : String) : Prop := 
  expression = "(3πm)/4"

/-- The given expression -/
def given_expression : String := "(3πm)/4"

/-- Theorem stating that the given expression meets the requirements for algebraic notation -/
theorem given_expression_is_proper_algebraic_notation : 
  is_proper_algebraic_notation given_expression := by
  sorry

end NUMINAMATH_CALUDE_given_expression_is_proper_algebraic_notation_l2415_241560


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2415_241583

/-- The x-coordinate of a point on the curve for a given t -/
def x (t : ℝ) : ℝ := 2 * t^2 - 3

/-- The y-coordinate of a point on the curve for a given t -/
def y (t : ℝ) : ℝ := 2 * t^4 - 9 * t^2 + 6

/-- Theorem stating that (-1, -1) is a self-intersection point of the curve -/
theorem curve_self_intersection :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ x t₁ = -1 ∧ y t₁ = -1 := by
  sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2415_241583


namespace NUMINAMATH_CALUDE_parabola_rotation_l2415_241596

/-- A parabola is defined by its coefficients a, h, and k in the form y = a(x - h)² + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Rotate a parabola by 180° around the origin -/
def rotate180 (p : Parabola) : Parabola :=
  { a := -p.a, h := -p.h, k := -p.k }

theorem parabola_rotation :
  let p := Parabola.mk 2 1 2
  rotate180 p = Parabola.mk (-2) (-1) (-2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_rotation_l2415_241596


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l2415_241528

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, y < x → ¬(|2*y + 7| ≤ 16)) ∧ (|2*x + 7| ≤ 16) → x = -11 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l2415_241528


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2415_241524

/-- A quadratic inequality with parameter m has exactly 3 integer solutions -/
def has_three_integer_solutions (m : ℝ) : Prop :=
  ∃! (a b c : ℤ), (m : ℝ) * (a : ℝ)^2 + (2 - m) * (a : ℝ) - 2 > 0 ∧
                   (m : ℝ) * (b : ℝ)^2 + (2 - m) * (b : ℝ) - 2 > 0 ∧
                   (m : ℝ) * (c : ℝ)^2 + (2 - m) * (c : ℝ) - 2 > 0 ∧
                   ∀ (x : ℤ), (m : ℝ) * (x : ℝ)^2 + (2 - m) * (x : ℝ) - 2 > 0 → (x = a ∨ x = b ∨ x = c)

/-- The main theorem -/
theorem quadratic_inequality_solution_range (m : ℝ) :
  has_three_integer_solutions m → -1/2 < m ∧ m ≤ -2/5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2415_241524


namespace NUMINAMATH_CALUDE_square_area_equals_one_l2415_241523

theorem square_area_equals_one (w l : ℝ) (h1 : l = 2 * w) (h2 : w * l = 8 / 9) :
  ∃ s : ℝ, s > 0 ∧ 4 * s = 6 * w ∧ s^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equals_one_l2415_241523


namespace NUMINAMATH_CALUDE_division_theorem_l2415_241533

theorem division_theorem (A : ℕ) : 14 = 3 * A + 2 → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_theorem_l2415_241533


namespace NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_sum_proof_l2415_241527

/-- The sum of the present ages of a father and son, given specific age relationships -/
theorem father_son_age_sum : ℕ → ℕ → Prop :=
  fun son_age father_age =>
    (father_age - 18 = 3 * (son_age - 18)) ∧  -- 18 years ago relationship
    (father_age = 2 * son_age) →              -- current relationship
    son_age + father_age = 108                -- sum of present ages

/-- Proof of the father_son_age_sum theorem -/
theorem father_son_age_sum_proof : ∃ (son_age father_age : ℕ), father_son_age_sum son_age father_age := by
  sorry

#check father_son_age_sum
#check father_son_age_sum_proof

end NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_sum_proof_l2415_241527


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2415_241502

/-- Represents a parallelogram with given properties -/
structure Parallelogram where
  area : ℝ
  base : ℝ
  altitude : ℝ
  angle : ℝ
  area_eq : area = 200
  altitude_eq : altitude = 2 * base
  angle_eq : angle = 60

/-- Theorem: The base of the parallelogram with given properties is 10 meters -/
theorem parallelogram_base_length (p : Parallelogram) : p.base = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2415_241502


namespace NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_times_three_l2415_241517

theorem five_fourths_of_twelve_fifths_times_three (x : ℚ) : x = 12 / 5 → (5 / 4 * x) * 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_times_three_l2415_241517


namespace NUMINAMATH_CALUDE_platform_length_l2415_241543

/-- Given a train and platform with specific properties, prove the platform length --/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 30)
  (h3 : pole_crossing_time = 18) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 200 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2415_241543


namespace NUMINAMATH_CALUDE_smaug_copper_coins_l2415_241590

/-- Represents the number of coins of each type in Smaug's hoard -/
structure DragonHoard where
  gold : ℕ
  silver : ℕ
  copper : ℕ

/-- Calculates the total value of a hoard in copper coins -/
def hoardValue (h : DragonHoard) (silverValue copperValue : ℕ) : ℕ :=
  h.gold * silverValue * copperValue + h.silver * copperValue + h.copper

/-- Theorem stating that Smaug has 33 copper coins -/
theorem smaug_copper_coins :
  ∃ (h : DragonHoard),
    h.gold = 100 ∧
    h.silver = 60 ∧
    hoardValue h 3 8 = 2913 ∧
    h.copper = 33 := by
  sorry

end NUMINAMATH_CALUDE_smaug_copper_coins_l2415_241590


namespace NUMINAMATH_CALUDE_water_bottles_sold_l2415_241520

/-- The number of water bottles sold in a store, given the prices and quantities of other drinks --/
theorem water_bottles_sold : ℕ := by
  -- Define the prices of drinks
  let cola_price : ℚ := 3
  let juice_price : ℚ := 3/2
  let water_price : ℚ := 1

  -- Define the quantities of cola and juice sold
  let cola_quantity : ℕ := 15
  let juice_quantity : ℕ := 12

  -- Define the total earnings
  let total_earnings : ℚ := 88

  -- Define the function to calculate the number of water bottles
  let water_bottles (x : ℕ) : Prop :=
    cola_price * cola_quantity + juice_price * juice_quantity + water_price * x = total_earnings

  -- Prove that the number of water bottles sold is 25
  have h : water_bottles 25 := by sorry

  exact 25

end NUMINAMATH_CALUDE_water_bottles_sold_l2415_241520


namespace NUMINAMATH_CALUDE_probability_king_hearts_then_ace_in_standard_deck_l2415_241503

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of King of Hearts in a standard deck -/
def KingOfHearts : ℕ := 1

/-- Number of Aces in a standard deck -/
def Aces : ℕ := 4

/-- Probability of drawing King of Hearts first and any Ace second -/
def probability_king_hearts_then_ace (deck : ℕ) (king_of_hearts : ℕ) (aces : ℕ) : ℚ :=
  (king_of_hearts : ℚ) / deck * (aces : ℚ) / (deck - 1)

theorem probability_king_hearts_then_ace_in_standard_deck :
  probability_king_hearts_then_ace StandardDeck KingOfHearts Aces = 1 / 663 := by
  sorry

end NUMINAMATH_CALUDE_probability_king_hearts_then_ace_in_standard_deck_l2415_241503


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2415_241595

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ),
    (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 →
      (D * x - 17) / (x^2 - 8*x + 15) = C / (x - 3) + 5 / (x - 5)) ∧
    C + D = 29/5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2415_241595


namespace NUMINAMATH_CALUDE_pet_shop_dogs_l2415_241569

theorem pet_shop_dogs (birds : ℕ) (snakes : ℕ) (spider : ℕ) (total_legs : ℕ) :
  birds = 3 → snakes = 4 → spider = 1 → total_legs = 34 →
  ∃ dogs : ℕ, dogs = 5 ∧ total_legs = birds * 2 + dogs * 4 + spider * 8 :=
by sorry

end NUMINAMATH_CALUDE_pet_shop_dogs_l2415_241569


namespace NUMINAMATH_CALUDE_workmen_efficiency_ratio_l2415_241509

/-- Given two workmen with different efficiencies, prove their efficiency ratio -/
theorem workmen_efficiency_ratio 
  (combined_time : ℝ) 
  (b_alone_time : ℝ) 
  (ha : combined_time = 18) 
  (hb : b_alone_time = 54) : 
  (1 / combined_time - 1 / b_alone_time) / (1 / b_alone_time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_workmen_efficiency_ratio_l2415_241509


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2415_241571

theorem sum_of_solutions (x : ℝ) : 
  (5 * x^2 - 3 * x - 2 = 0) → 
  (∃ y : ℝ, 5 * y^2 - 3 * y - 2 = 0 ∧ x + y = 3/5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2415_241571


namespace NUMINAMATH_CALUDE_apple_weight_average_l2415_241504

theorem apple_weight_average (standard_weight : ℝ) (deviations : List ℝ) : 
  standard_weight = 30 →
  deviations = [0.4, -0.2, -0.8, -0.4, 1, 0.3, 0.5, -2, 0.5, -0.1] →
  (standard_weight + (deviations.sum / deviations.length)) = 29.92 := by
  sorry

end NUMINAMATH_CALUDE_apple_weight_average_l2415_241504


namespace NUMINAMATH_CALUDE_translated_function_coefficient_sum_l2415_241516

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

def translation (h : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + h)

theorem translated_function_coefficient_sum :
  ∃ (a b c : ℝ),
    (∀ x, translation 3 f x = a * x^2 + b * x + c) ∧
    a + b + c = 51 :=
by sorry

end NUMINAMATH_CALUDE_translated_function_coefficient_sum_l2415_241516


namespace NUMINAMATH_CALUDE_remainder_theorem_l2415_241562

-- Define the polynomial q(x)
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 8

-- State the theorem
theorem remainder_theorem (D E F : ℝ) :
  (∃ p : ℝ → ℝ, ∀ x, q D E F x = (x - 2) * p x + 12) →
  (∃ p : ℝ → ℝ, ∀ x, q D E F x = (x + 2) * p x + 4) :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2415_241562


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_including_31_l2415_241579

theorem unique_number_with_three_prime_divisors_including_31 (x n : ℕ) :
  x = 8^n - 1 →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 31 ∧ q ≠ 31 ∧
    x = p * q * 31 ∧ 
    ∀ r : ℕ, Prime r → r ∣ x → (r = p ∨ r = q ∨ r = 31)) →
  x = 32767 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_including_31_l2415_241579


namespace NUMINAMATH_CALUDE_f_one_when_m_three_max_value_when_even_max_value_attained_when_even_l2415_241546

-- Define the function f(x) with parameter m
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x + 2

-- Theorem 1: When m = 3, f(1) = 4
theorem f_one_when_m_three : f 3 1 = 4 := by sorry

-- Define what it means for f to be an even function
def is_even_function (m : ℝ) : Prop := ∀ x, f m (-x) = f m x

-- Theorem 2: If f is an even function, its maximum value is 2
theorem max_value_when_even :
  ∀ m, is_even_function m → ∀ x, f m x ≤ 2 := by sorry

-- Theorem 3: The maximum value 2 is attained when f is an even function
theorem max_value_attained_when_even :
  ∃ m, is_even_function m ∧ ∃ x, f m x = 2 := by sorry

end NUMINAMATH_CALUDE_f_one_when_m_three_max_value_when_even_max_value_attained_when_even_l2415_241546


namespace NUMINAMATH_CALUDE_max_area_quadrilateral_in_circle_l2415_241585

theorem max_area_quadrilateral_in_circle (d : Real) 
  (h1 : 0 ≤ d) (h2 : d < 1) : 
  ∃ (max_area : Real),
    (d < Real.sqrt 2 / 2 → max_area = 2 * Real.sqrt (1 - d^2)) ∧
    (Real.sqrt 2 / 2 ≤ d → max_area = 1 / d) ∧
    ∀ (area : Real), area ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_max_area_quadrilateral_in_circle_l2415_241585


namespace NUMINAMATH_CALUDE_van_helsing_earnings_l2415_241561

/-- Van Helsing's vampire and werewolf removal earnings problem -/
theorem van_helsing_earnings : ∀ (v w : ℕ),
  w = 4 * v →  -- There were 4 times as many werewolves as vampires
  w = 8 →      -- 8 werewolves were removed
  5 * (v / 2) + 10 * 8 = 85  -- Total earnings calculation
  := by sorry

end NUMINAMATH_CALUDE_van_helsing_earnings_l2415_241561


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2415_241537

def P (x : ℝ) : ℝ := 5*x^6 - 3*x^5 + 4*x^4 - x^3 + 6*x^2 - 5*x + 7

theorem polynomial_remainder_theorem :
  ∃ (Q : ℝ → ℝ), P = λ x ↦ (x - 3) * Q x + 3259 :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2415_241537


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_values_l2415_241549

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

theorem unique_solution_implies_a_values (a : ℝ) : (∃! x, x ∈ A a) → a = 0 ∨ a = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_values_l2415_241549


namespace NUMINAMATH_CALUDE_chord_intersection_theorem_l2415_241521

theorem chord_intersection_theorem (r : ℝ) (PT OT : ℝ) : 
  r = 7 → OT = 3 → PT = 8 → 
  ∃ (RS : ℝ), RS = 16 ∧ 
  ∃ (x : ℝ), x * (RS - x) = PT * PT ∧ 
  ∃ (n : ℕ), x * (RS - x) = n^2 :=
by sorry

end NUMINAMATH_CALUDE_chord_intersection_theorem_l2415_241521


namespace NUMINAMATH_CALUDE_ellipse_equation_form_l2415_241541

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_axes : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  sorry

theorem ellipse_equation_form (e : Ellipse) 
  (h_center : e.center = (0, 0))
  (h_foci : e.foci_on_axes = true)
  (h_eccentricity : e.eccentricity = Real.sqrt 3 / 2)
  (h_point : e.passes_through = (2, 0)) :
  (ellipse_equation e = fun x y => x^2 + 4*y^2 = 4) ∨
  (ellipse_equation e = fun x y => 4*x^2 + y^2 = 16) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_form_l2415_241541


namespace NUMINAMATH_CALUDE_fisher_min_score_l2415_241568

/-- Represents a student's scores and eligibility requirements -/
structure StudentScores where
  algebra_sem1 : ℝ
  algebra_sem2 : ℝ
  statistics : ℝ
  algebra_required_avg : ℝ
  statistics_required : ℝ

/-- Determines if a student is eligible for the geometry class -/
def is_eligible (s : StudentScores) : Prop :=
  (s.algebra_sem1 + s.algebra_sem2) / 2 ≥ s.algebra_required_avg ∧
  s.statistics ≥ s.statistics_required

/-- Calculates the minimum score needed in the second semester of Algebra -/
def min_algebra_sem2_score (s : StudentScores) : ℝ :=
  2 * s.algebra_required_avg - s.algebra_sem1

/-- Theorem stating the minimum score Fisher needs in the second semester of Algebra -/
theorem fisher_min_score (fisher : StudentScores)
  (h1 : fisher.algebra_required_avg = 85)
  (h2 : fisher.statistics_required = 80)
  (h3 : fisher.algebra_sem1 = 84)
  (h4 : fisher.statistics = 82) :
  min_algebra_sem2_score fisher = 86 ∧ 
  is_eligible { fisher with algebra_sem2 := min_algebra_sem2_score fisher } :=
sorry


end NUMINAMATH_CALUDE_fisher_min_score_l2415_241568


namespace NUMINAMATH_CALUDE_log_equation_implies_c_eq_a_to_three_halves_l2415_241515

/-- Given the equation relating logarithms of x with bases c and a, prove that c = a^(3/2) -/
theorem log_equation_implies_c_eq_a_to_three_halves
  (a c x : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_c : 0 < c)
  (h_pos_x : 0 < x)
  (h_eq : 2 * (Real.log x / Real.log c)^2 + 5 * (Real.log x / Real.log a)^2 = 12 * (Real.log x)^2 / (Real.log a * Real.log c)) :
  c = a^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_implies_c_eq_a_to_three_halves_l2415_241515


namespace NUMINAMATH_CALUDE_root_difference_zero_l2415_241513

/-- The nonnegative difference between the roots of x^2 + 40x + 300 = -100 is 0 -/
theorem root_difference_zero : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 40*x + 300 + 100
  let roots := {x : ℝ | f x = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_zero_l2415_241513


namespace NUMINAMATH_CALUDE_number_of_divisors_180_l2415_241501

theorem number_of_divisors_180 : Nat.card (Nat.divisors 180) = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_180_l2415_241501


namespace NUMINAMATH_CALUDE_adams_school_schedule_l2415_241519

/-- Represents the number of lessons Adam had on Tuesday -/
def tuesday_lessons : ℕ := 3

theorem adams_school_schedule :
  let monday_hours : ℝ := 3
  let tuesday_hours : ℝ := tuesday_lessons
  let wednesday_hours : ℝ := 2 * tuesday_hours
  let total_hours : ℝ := 12
  monday_hours + tuesday_hours + wednesday_hours = total_hours :=
by sorry


end NUMINAMATH_CALUDE_adams_school_schedule_l2415_241519


namespace NUMINAMATH_CALUDE_johnson_smith_tied_may_l2415_241506

/-- Represents the months of a baseball season --/
inductive Month
| Jan | Feb | Mar | Apr | May | Jul | Aug | Sep

/-- Represents a baseball player --/
structure Player where
  name : String
  homeRuns : Month → Nat

def johnson : Player :=
  { name := "Johnson"
  , homeRuns := fun
    | Month.Jan => 2
    | Month.Feb => 12
    | Month.Mar => 15
    | Month.Apr => 8
    | Month.May => 14
    | Month.Jul => 11
    | Month.Aug => 9
    | Month.Sep => 16 }

def smith : Player :=
  { name := "Smith"
  , homeRuns := fun
    | Month.Jan => 5
    | Month.Feb => 9
    | Month.Mar => 10
    | Month.Apr => 12
    | Month.May => 15
    | Month.Jul => 12
    | Month.Aug => 10
    | Month.Sep => 17 }

def totalHomeRunsUpTo (p : Player) (m : Month) : Nat :=
  match m with
  | Month.Jan => p.homeRuns Month.Jan
  | Month.Feb => p.homeRuns Month.Jan + p.homeRuns Month.Feb
  | Month.Mar => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar
  | Month.Apr => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr
  | Month.May => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr + p.homeRuns Month.May
  | Month.Jul => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr + p.homeRuns Month.May + p.homeRuns Month.Jul
  | Month.Aug => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr + p.homeRuns Month.May + p.homeRuns Month.Jul + p.homeRuns Month.Aug
  | Month.Sep => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr + p.homeRuns Month.May + p.homeRuns Month.Jul + p.homeRuns Month.Aug + p.homeRuns Month.Sep

theorem johnson_smith_tied_may :
  totalHomeRunsUpTo johnson Month.May = totalHomeRunsUpTo smith Month.May :=
by sorry

end NUMINAMATH_CALUDE_johnson_smith_tied_may_l2415_241506


namespace NUMINAMATH_CALUDE_cousins_arrangement_l2415_241559

/-- The number of ways to arrange n indistinguishable objects into k distinct boxes -/
def arrange (n k : ℕ) : ℕ := sorry

/-- There are 4 rooms available -/
def num_rooms : ℕ := 4

/-- There are 5 cousins to arrange -/
def num_cousins : ℕ := 5

/-- The number of arrangements of 5 cousins in 4 rooms is 76 -/
theorem cousins_arrangement : arrange num_cousins num_rooms = 76 := by sorry

end NUMINAMATH_CALUDE_cousins_arrangement_l2415_241559


namespace NUMINAMATH_CALUDE_square_root_of_1024_l2415_241581

theorem square_root_of_1024 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1024_l2415_241581


namespace NUMINAMATH_CALUDE_sum_six_consecutive_integers_l2415_241567

/-- The sum of six consecutive integers starting from n is equal to 6n + 15 -/
theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_consecutive_integers_l2415_241567


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2415_241563

theorem complex_equation_solution : 
  ∃ (z : ℂ), z / (1 - Complex.I) = Complex.I ^ 2019 → z = -1 - Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2415_241563


namespace NUMINAMATH_CALUDE_min_colors_theorem_l2415_241598

/-- A coloring of positive integers -/
def Coloring (k : ℕ) := ℕ+ → Fin k

/-- A function from positive integers to positive integers -/
def IntFunction := ℕ+ → ℕ+

/-- The property that f(m+n) = f(m) + f(n) for integers of the same color -/
def SameColorAdditive (f : IntFunction) (c : Coloring k) : Prop :=
  ∀ m n : ℕ+, c m = c n → f (m + n) = f m + f n

/-- The property that there exist m and n such that f(m+n) ≠ f(m) + f(n) -/
def ExistsDifferentSum (f : IntFunction) : Prop :=
  ∃ m n : ℕ+, f (m + n) ≠ f m + f n

/-- The main theorem -/
theorem min_colors_theorem :
  (∃ k : ℕ+, ∃ c : Coloring k, ∃ f : IntFunction,
    SameColorAdditive f c ∧ ExistsDifferentSum f) ∧
  (∀ k : ℕ+, k < 3 → ¬∃ c : Coloring k, ∃ f : IntFunction,
    SameColorAdditive f c ∧ ExistsDifferentSum f) :=
sorry

end NUMINAMATH_CALUDE_min_colors_theorem_l2415_241598


namespace NUMINAMATH_CALUDE_parabola_vertex_l2415_241557

/-- 
Given a parabola y = -x^2 + px + q where the solution to y ≤ 0 is (-∞, -4] ∪ [6, ∞),
prove that the vertex of the parabola is (1, 25).
-/
theorem parabola_vertex (p q : ℝ) : 
  (∀ x, -x^2 + p*x + q ≤ 0 ↔ x ≤ -4 ∨ x ≥ 6) →
  ∃ x y, x = 1 ∧ y = 25 ∧ ∀ t, -t^2 + p*t + q ≤ y := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2415_241557


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_half_l2415_241570

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem parallel_vectors_imply_x_half :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ a + 2 • b x = k • (2 • a - 2 • b x)) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_half_l2415_241570


namespace NUMINAMATH_CALUDE_drawing_pie_satisfies_hunger_is_impossible_l2415_241500

/-- An event that involves drawing a pie and satisfying hunger -/
def drawing_pie_to_satisfy_hunger : Set (Nat × Nat) := sorry

/-- Definition of an impossible event -/
def impossible_event (E : Set (Nat × Nat)) : Prop :=
  E = ∅

/-- Theorem: Drawing a pie to satisfy hunger is an impossible event -/
theorem drawing_pie_satisfies_hunger_is_impossible :
  impossible_event drawing_pie_to_satisfy_hunger := by sorry

end NUMINAMATH_CALUDE_drawing_pie_satisfies_hunger_is_impossible_l2415_241500


namespace NUMINAMATH_CALUDE_log_216_simplification_l2415_241512

theorem log_216_simplification :
  (216 : ℝ) = 6^3 →
  Real.log 216 = 3 * (Real.log 2 + Real.log 3) := by
sorry

end NUMINAMATH_CALUDE_log_216_simplification_l2415_241512


namespace NUMINAMATH_CALUDE_square_root_problem_l2415_241551

theorem square_root_problem (a b : ℝ) : 
  ((2 * a - 1)^2 = 4) → (b = 1) → (2 * a - b = 2 ∨ 2 * a - b = -2) := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l2415_241551


namespace NUMINAMATH_CALUDE_multiply_125_3_2_25_solve_equation_l2415_241539

-- Part 1: Prove that 125 × 3.2 × 25 = 10000
theorem multiply_125_3_2_25 : 125 * 3.2 * 25 = 10000 := by sorry

-- Part 2: Prove that the solution to 24(x-12) = 16(x-4) is x = 28
theorem solve_equation : ∃ x : ℝ, 24 * (x - 12) = 16 * (x - 4) ∧ x = 28 := by sorry

end NUMINAMATH_CALUDE_multiply_125_3_2_25_solve_equation_l2415_241539


namespace NUMINAMATH_CALUDE_sasha_picked_24_leaves_l2415_241565

/-- The number of apple trees along the road. -/
def apple_trees : ℕ := 17

/-- The number of poplar trees along the road. -/
def poplar_trees : ℕ := 20

/-- The index of the apple tree from which Sasha starts picking leaves. -/
def start_tree : ℕ := 8

/-- The total number of trees along the road. -/
def total_trees : ℕ := apple_trees + poplar_trees

/-- The number of leaves Sasha picked. -/
def leaves_picked : ℕ := total_trees - (start_tree - 1)

theorem sasha_picked_24_leaves : leaves_picked = 24 := by
  sorry

end NUMINAMATH_CALUDE_sasha_picked_24_leaves_l2415_241565


namespace NUMINAMATH_CALUDE_exponent_division_l2415_241511

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^7 / a^3 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2415_241511


namespace NUMINAMATH_CALUDE_charity_event_arrangements_l2415_241526

theorem charity_event_arrangements (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 5 → k = 4 → m = 3 →
  (Nat.choose n 2) * (Nat.choose (n - 2) 1) * (Nat.choose (n - 3) 1) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_charity_event_arrangements_l2415_241526


namespace NUMINAMATH_CALUDE_age_difference_of_children_l2415_241542

/-- Proves that the age difference between two children is 2 years given the family conditions --/
theorem age_difference_of_children (initial_members : ℕ) (initial_avg_age : ℕ) 
  (years_passed : ℕ) (current_members : ℕ) (current_avg_age : ℕ) (youngest_child_age : ℕ) :
  initial_members = 4 →
  initial_avg_age = 24 →
  years_passed = 10 →
  current_members = 6 →
  current_avg_age = 24 →
  youngest_child_age = 3 →
  ∃ (older_child_age : ℕ), 
    older_child_age - youngest_child_age = 2 ∧
    older_child_age + youngest_child_age = 
      current_members * current_avg_age - initial_members * (initial_avg_age + years_passed) :=
by
  sorry

#check age_difference_of_children

end NUMINAMATH_CALUDE_age_difference_of_children_l2415_241542


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l2415_241599

theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45) 
  (h2 : (b + c) / 2 = 50) : 
  c - a = 10 := by
sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l2415_241599


namespace NUMINAMATH_CALUDE_two_dice_prime_probability_l2415_241538

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def dice_outcomes (n : ℕ) : ℕ := 6^n

def prime_outcomes (n : ℕ) : ℕ := 
  if n = 2 then 15 else 0  -- We only define it for 2 dice as per the problem

theorem two_dice_prime_probability :
  (prime_outcomes 2 : ℚ) / (dice_outcomes 2 : ℚ) = 5/12 :=
sorry

end NUMINAMATH_CALUDE_two_dice_prime_probability_l2415_241538


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l2415_241558

noncomputable def f (t : ℝ) : ℝ := Real.exp t + 1

noncomputable def g (t : ℝ) : ℝ := 2 * t - 1

theorem min_distance_between_curves :
  ∃ (t_min : ℝ), ∀ (t : ℝ), |f t - g t| ≥ |f t_min - g t_min| ∧ 
  |f t_min - g t_min| = 4 - 2 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l2415_241558


namespace NUMINAMATH_CALUDE_fraction_of_silver_knights_with_shields_l2415_241547

theorem fraction_of_silver_knights_with_shields :
  ∀ (total_knights : ℕ) (silver_knights : ℕ) (golden_knights : ℕ) (knights_with_shields : ℕ)
    (silver_knights_with_shields : ℕ) (golden_knights_with_shields : ℕ),
  total_knights > 0 →
  silver_knights + golden_knights = total_knights →
  silver_knights = (3 * total_knights) / 8 →
  knights_with_shields = total_knights / 4 →
  silver_knights_with_shields + golden_knights_with_shields = knights_with_shields →
  silver_knights_with_shields * golden_knights = 3 * golden_knights_with_shields * silver_knights →
  silver_knights_with_shields * 7 = silver_knights * 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_silver_knights_with_shields_l2415_241547


namespace NUMINAMATH_CALUDE_sequence_periodicity_l2415_241591

theorem sequence_periodicity (a : ℕ → ℚ) (h1 : ∀ n : ℕ, |a (n + 1) - 2 * a n| = 2)
    (h2 : ∀ n : ℕ, |a n| ≤ 2) :
  ∃ k l : ℕ, k < l ∧ a k = a l :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l2415_241591


namespace NUMINAMATH_CALUDE_initial_persons_count_l2415_241508

/-- The number of persons initially in the group. -/
def n : ℕ := sorry

/-- The average weight increase when a new person replaces one person. -/
def avg_weight_increase : ℚ := 5/2

/-- The weight of the replaced person. -/
def old_weight : ℕ := 65

/-- The weight of the new person. -/
def new_weight : ℕ := 85

/-- Theorem stating that the initial number of persons is 8. -/
theorem initial_persons_count : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_persons_count_l2415_241508


namespace NUMINAMATH_CALUDE_min_stamps_proof_l2415_241507

/-- Represents the number of stamps of each denomination -/
structure StampCombination where
  three_cent : ℕ
  four_cent : ℕ
  five_cent : ℕ

/-- Calculates the total value of stamps in cents -/
def total_value (s : StampCombination) : ℕ :=
  3 * s.three_cent + 4 * s.four_cent + 5 * s.five_cent

/-- Calculates the total number of stamps -/
def total_stamps (s : StampCombination) : ℕ :=
  s.three_cent + s.four_cent + s.five_cent

/-- Checks if a stamp combination is valid (totals 50 cents) -/
def is_valid (s : StampCombination) : Prop :=
  total_value s = 50

/-- The minimum number of stamps needed -/
def min_stamps : ℕ := 10

theorem min_stamps_proof :
  (∀ s : StampCombination, is_valid s → total_stamps s ≥ min_stamps) ∧
  (∃ s : StampCombination, is_valid s ∧ total_stamps s = min_stamps) := by
  sorry

#check min_stamps_proof

end NUMINAMATH_CALUDE_min_stamps_proof_l2415_241507


namespace NUMINAMATH_CALUDE_average_problem_l2415_241525

theorem average_problem (x y : ℝ) :
  (7 + 9 + x + y + 17) / 5 = 10 →
  ((x + 3) + (x + 5) + (y + 2) + 8 + (y + 18)) / 5 = 14 :=
by sorry

end NUMINAMATH_CALUDE_average_problem_l2415_241525


namespace NUMINAMATH_CALUDE_profit_is_120_l2415_241514

/-- Calculates the profit from book sales given the selling price, number of customers,
    production cost, and books per customer. -/
def calculate_profit (selling_price : ℕ) (num_customers : ℕ) (production_cost : ℕ) (books_per_customer : ℕ) : ℕ :=
  let total_books := num_customers * books_per_customer
  let revenue := selling_price * total_books
  let total_cost := production_cost * total_books
  revenue - total_cost

/-- Proves that the profit is $120 given the specified conditions. -/
theorem profit_is_120 :
  let selling_price := 20
  let num_customers := 4
  let production_cost := 5
  let books_per_customer := 2
  calculate_profit selling_price num_customers production_cost books_per_customer = 120 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_120_l2415_241514


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2415_241552

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 1 → x^2 + x - 2 > 0) ∧ 
  (∃ x : ℝ, x^2 + x - 2 > 0 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2415_241552


namespace NUMINAMATH_CALUDE_equation_solutions_count_l2415_241587

theorem equation_solutions_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 + x - 12)^2 = 81) ∧ s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l2415_241587


namespace NUMINAMATH_CALUDE_boat_speed_l2415_241505

/-- The speed of a boat in still water, given its speeds with and against a stream -/
theorem boat_speed (along_stream : ℝ) (against_stream : ℝ) 
  (h1 : along_stream = 38) 
  (h2 : against_stream = 16) : 
  (along_stream + against_stream) / 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l2415_241505


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l2415_241510

theorem students_taking_one_subject (total_students : ℕ) 
  (algebra_and_drafting : ℕ) (algebra_total : ℕ) (only_drafting : ℕ) 
  (neither_subject : ℕ) :
  algebra_and_drafting = 22 →
  algebra_total = 40 →
  only_drafting = 15 →
  neither_subject = 8 →
  total_students = algebra_total + only_drafting + neither_subject →
  (algebra_total - algebra_and_drafting) + only_drafting = 33 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l2415_241510


namespace NUMINAMATH_CALUDE_machine_productivity_problem_l2415_241534

theorem machine_productivity_problem 
  (productivity_second : ℝ) 
  (productivity_first : ℝ := 1.4 * productivity_second) 
  (hours_first : ℝ := 6) 
  (hours_second : ℝ := 8) 
  (total_parts : ℕ := 820) :
  productivity_first * hours_first + productivity_second * hours_second = total_parts → 
  (productivity_first * hours_first = 420 ∧ productivity_second * hours_second = 400) :=
by
  sorry

end NUMINAMATH_CALUDE_machine_productivity_problem_l2415_241534


namespace NUMINAMATH_CALUDE_hippo_ratio_l2415_241575

/-- Represents the number of female hippos -/
def F : ℕ := sorry

/-- The initial number of elephants -/
def initial_elephants : ℕ := 20

/-- The initial number of hippos -/
def initial_hippos : ℕ := 35

/-- The number of baby hippos born per female hippo -/
def babies_per_hippo : ℕ := 5

/-- The total number of animals after births -/
def total_animals : ℕ := 315

theorem hippo_ratio :
  let newborn_hippos := F * babies_per_hippo
  let newborn_elephants := newborn_hippos + 10
  let total_hippos := initial_hippos + newborn_hippos
  (F : ℚ) / total_hippos = 5 / 32 :=
by sorry

end NUMINAMATH_CALUDE_hippo_ratio_l2415_241575


namespace NUMINAMATH_CALUDE_integral_exp_plus_2x_equals_e_l2415_241584

theorem integral_exp_plus_2x_equals_e :
  ∫ x in (0:ℝ)..1, (Real.exp x + 2 * x) = Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_integral_exp_plus_2x_equals_e_l2415_241584


namespace NUMINAMATH_CALUDE_tomato_field_area_l2415_241574

/-- Given a rectangular field with length 3.6 meters and width 2.5 times the length,
    the area of half of this field is 16.2 square meters. -/
theorem tomato_field_area :
  let length : ℝ := 3.6
  let width : ℝ := 2.5 * length
  let total_area : ℝ := length * width
  let tomato_area : ℝ := total_area / 2
  tomato_area = 16.2 := by
sorry

end NUMINAMATH_CALUDE_tomato_field_area_l2415_241574


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l2415_241577

theorem right_triangle_acute_angles (α : Real) 
  (h1 : 0 < α ∧ α < 90) 
  (h2 : (90 - α / 2) / (45 + α / 2) = 13 / 17) : 
  α = 63 ∧ 90 - α = 27 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l2415_241577


namespace NUMINAMATH_CALUDE_expression_value_l2415_241545

theorem expression_value (x y : ℝ) (h1 : x / (2 * y) = 3 / 2) (h2 : y ≠ 0) :
  (7 * x + 4 * y) / (x - 2 * y) = 25 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2415_241545


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2415_241589

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_fourth : a 4 = 5)
  (h_sum : a 5 + a 6 = 11) :
  a 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2415_241589


namespace NUMINAMATH_CALUDE_speaking_orders_count_l2415_241554

def total_people : Nat := 7
def speakers : Nat := 4
def special_people : Nat := 2  -- A and B

theorem speaking_orders_count : 
  (total_people.choose speakers * speakers.factorial - 
   (total_people - special_people).choose speakers * speakers.factorial) = 720 := by
  sorry

end NUMINAMATH_CALUDE_speaking_orders_count_l2415_241554


namespace NUMINAMATH_CALUDE_trophy_cost_l2415_241582

theorem trophy_cost (x y : ℕ) (hx : x < 10) (hy : y < 10) :
  let total_cents : ℕ := 1000 * x + 9990 + y
  (72 ∣ total_cents) →
  (total_cents : ℚ) / (72 * 100) = 11.11 := by
sorry

end NUMINAMATH_CALUDE_trophy_cost_l2415_241582


namespace NUMINAMATH_CALUDE_circle_radii_order_l2415_241592

/-- Given three circles A, B, and C with the following properties:
    - Circle A has a circumference of 6π
    - Circle B has an area of 16π
    - Circle C has a radius of 2
    Prove that the radii of the circles are ordered as r_C < r_A < r_B -/
theorem circle_radii_order (r_A r_B r_C : ℝ) : 
  (2 * π * r_A = 6 * π) →  -- Circumference of A
  (π * r_B^2 = 16 * π) →   -- Area of B
  (r_C = 2) →              -- Radius of C
  r_C < r_A ∧ r_A < r_B := by
sorry

end NUMINAMATH_CALUDE_circle_radii_order_l2415_241592


namespace NUMINAMATH_CALUDE_all_parameterizations_valid_l2415_241588

def line_equation (x y : ℝ) : Prop := y = 2 * x - 4

def valid_parameterization (p : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line_equation (p.1 + t * v.1) (p.2 + t * v.2)

theorem all_parameterizations_valid :
  valid_parameterization (3, -2) (1, 2) ∧
  valid_parameterization (4, 0) (2, 4) ∧
  valid_parameterization (0, -4) (1, 2) ∧
  valid_parameterization (1, -1) (0.5, 1) ∧
  valid_parameterization (-1, -6) (-2, -4) :=
sorry

end NUMINAMATH_CALUDE_all_parameterizations_valid_l2415_241588
