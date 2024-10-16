import Mathlib

namespace NUMINAMATH_CALUDE_shoe_box_problem_l2454_245437

theorem shoe_box_problem (num_pairs : ℕ) (prob : ℚ) (total_shoes : ℕ) : 
  num_pairs = 12 → 
  prob = 1 / 23 → 
  prob = num_pairs / (total_shoes.choose 2) → 
  total_shoes = 24 := by
sorry

end NUMINAMATH_CALUDE_shoe_box_problem_l2454_245437


namespace NUMINAMATH_CALUDE_min_value_ab_l2454_245464

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a * b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = a₀ + b₀ + 3 ∧ a₀ * b₀ = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_ab_l2454_245464


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l2454_245489

/-- An angle is in the third quadrant if it's between 180° and 270° (or equivalent in radians) -/
def is_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 2 * Real.pi + Real.pi < α ∧ α < k * 2 * Real.pi + 3 * Real.pi / 2

/-- An angle is in the second or fourth quadrant if it's between 90° and 180° or between 270° and 360° (or equivalent in radians) -/
def is_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, (k * 2 * Real.pi + Real.pi / 2 < α ∧ α < k * 2 * Real.pi + Real.pi) ∨
           (k * 2 * Real.pi + 3 * Real.pi / 2 < α ∧ α < (k + 1) * 2 * Real.pi)

theorem half_angle_quadrant (α : Real) :
  is_third_quadrant α → is_second_or_fourth_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l2454_245489


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2454_245468

theorem arithmetic_sequence_problem (a : ℕ → ℤ) (d : ℤ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 4 + a 5 + a 6 + a 7 = 56 →  -- sum condition
  a 4 * a 7 = 187 →             -- product condition
  ((a 1 = 5 ∧ d = 2) ∨ (a 1 = 23 ∧ d = -2)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2454_245468


namespace NUMINAMATH_CALUDE_is_arithmetic_sequence_l2454_245446

def S (n : ℕ) : ℝ := 2 * n + 1

theorem is_arithmetic_sequence :
  ∀ n : ℕ, S (n + 1) - S n = S 1 - S 0 :=
by
  sorry

end NUMINAMATH_CALUDE_is_arithmetic_sequence_l2454_245446


namespace NUMINAMATH_CALUDE_ways_to_top_center_l2454_245457

/-- Number of ways to reach the center square of the topmost row in a grid -/
def numWaysToTopCenter (n : ℕ) : ℕ :=
  2^(n-1)

/-- Theorem: The number of ways to reach the center square of the topmost row
    in a rectangular grid with n rows and 3 columns, starting from the bottom
    left corner and moving either one square right or simultaneously one square
    left and one square up at each step, is equal to 2^(n-1). -/
theorem ways_to_top_center (n : ℕ) (h : n > 0) :
  numWaysToTopCenter n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_ways_to_top_center_l2454_245457


namespace NUMINAMATH_CALUDE_race_time_difference_l2454_245436

/-- Proves that given the speeds of A and B are in the ratio 3:4, and A takes 2 hours to reach the destination, A takes 30 minutes more than B to reach the destination. -/
theorem race_time_difference (speed_a speed_b : ℝ) (time_a : ℝ) : 
  speed_a / speed_b = 3 / 4 →
  time_a = 2 →
  (time_a - (speed_a * time_a / speed_b)) * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l2454_245436


namespace NUMINAMATH_CALUDE_percent_of_self_equal_sixteen_l2454_245409

theorem percent_of_self_equal_sixteen (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 16) : x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_self_equal_sixteen_l2454_245409


namespace NUMINAMATH_CALUDE_division_problem_l2454_245479

theorem division_problem : (96 / 6) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2454_245479


namespace NUMINAMATH_CALUDE_cos_95_cos_25_minus_sin_95_sin_25_l2454_245483

theorem cos_95_cos_25_minus_sin_95_sin_25 :
  Real.cos (95 * π / 180) * Real.cos (25 * π / 180) - 
  Real.sin (95 * π / 180) * Real.sin (25 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_95_cos_25_minus_sin_95_sin_25_l2454_245483


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2454_245416

/-- Sum of a geometric series with n terms -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℚ := 1/3
  let r : ℚ := -1/2
  let n : ℕ := 6
  geometric_sum a r n = 7/32 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2454_245416


namespace NUMINAMATH_CALUDE_cos_30_degrees_l2454_245473

theorem cos_30_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_30_degrees_l2454_245473


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l2454_245445

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (initial_distance : ℝ)
  (train_length : ℝ)
  (h1 : jogger_speed = 9 * 1000 / 3600) -- 9 km/hr in m/s
  (h2 : train_speed = 45 * 1000 / 3600) -- 45 km/hr in m/s
  (h3 : initial_distance = 240)
  (h4 : train_length = 110) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 35 := by
sorry


end NUMINAMATH_CALUDE_train_passing_jogger_time_l2454_245445


namespace NUMINAMATH_CALUDE_exam_average_proof_l2454_245466

theorem exam_average_proof (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) (h₁ : n₁ = 15) (h₂ : n₂ = 10)
  (h₃ : avg₁ = 75/100) (h₄ : avg_total = 81/100) (h₅ : n₁ + n₂ = 25) :
  let avg₂ := (((n₁ + n₂ : ℚ) * avg_total) - (n₁ * avg₁)) / n₂
  avg₂ = 90/100 := by
sorry

end NUMINAMATH_CALUDE_exam_average_proof_l2454_245466


namespace NUMINAMATH_CALUDE_part_I_part_II_l2454_245414

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := { x | 0 < 2*x + a ∧ 2*x + a ≤ 3 }
def B : Set ℝ := { x | -1/2 < x ∧ x < 2 }

-- Part I
theorem part_I : 
  (Set.univ \ B) ∪ (A 1) = { x | x ≤ 1 ∨ x ≥ 2 } := by sorry

-- Part II
theorem part_II : 
  ∀ a : ℝ, (A a) ∩ B = A a ↔ -1 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l2454_245414


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2454_245440

theorem exponent_multiplication (b : ℝ) : b * b^3 = b^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2454_245440


namespace NUMINAMATH_CALUDE_tan_half_product_l2454_245431

theorem tan_half_product (a b : ℝ) :
  5 * (Real.cos a + Real.cos b) + 4 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 3 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -3) :=
by sorry

end NUMINAMATH_CALUDE_tan_half_product_l2454_245431


namespace NUMINAMATH_CALUDE_diagonal_length_of_regular_hexagon_l2454_245482

/-- A regular hexagon with side length 12 units -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 12)

/-- The length of a diagonal in a regular hexagon -/
def diagonal_length (h : RegularHexagon) : ℝ := 2 * h.side_length

/-- Theorem: The diagonal length of a regular hexagon with side length 12 is 24 -/
theorem diagonal_length_of_regular_hexagon (h : RegularHexagon) :
  diagonal_length h = 24 := by
  sorry

#check diagonal_length_of_regular_hexagon

end NUMINAMATH_CALUDE_diagonal_length_of_regular_hexagon_l2454_245482


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_factorials_l2454_245430

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem greatest_prime_factor_of_sum_factorials :
  ∃ p : ℕ, is_prime p ∧ p ∣ (factorial 15 + factorial 17) ∧
    ∀ q : ℕ, is_prime q → q ∣ (factorial 15 + factorial 17) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_factorials_l2454_245430


namespace NUMINAMATH_CALUDE_student_marks_calculation_l2454_245429

/-- Calculates the marks obtained by a student who failed an exam. -/
theorem student_marks_calculation
  (total_marks : ℕ)
  (passing_percentage : ℚ)
  (failing_margin : ℕ)
  (h_total : total_marks = 500)
  (h_passing : passing_percentage = 40 / 100)
  (h_failing : failing_margin = 50) :
  (total_marks : ℚ) * passing_percentage - failing_margin = 150 :=
by sorry

end NUMINAMATH_CALUDE_student_marks_calculation_l2454_245429


namespace NUMINAMATH_CALUDE_remainder_97_pow_45_mod_100_l2454_245484

theorem remainder_97_pow_45_mod_100 : 97^45 % 100 = 57 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_pow_45_mod_100_l2454_245484


namespace NUMINAMATH_CALUDE_abigail_expenses_l2454_245478

def initial_amount : ℝ := 200

def food_expense_percentage : ℝ := 0.60

def phone_bill_percentage : ℝ := 0.25

def entertainment_expense : ℝ := 20

def remaining_amount (initial : ℝ) (food_percent : ℝ) (phone_percent : ℝ) (entertainment : ℝ) : ℝ :=
  let after_food := initial * (1 - food_percent)
  let after_phone := after_food * (1 - phone_percent)
  after_phone - entertainment

theorem abigail_expenses :
  remaining_amount initial_amount food_expense_percentage phone_bill_percentage entertainment_expense = 40 := by
  sorry

end NUMINAMATH_CALUDE_abigail_expenses_l2454_245478


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l2454_245470

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n ≤ 99999) ∧
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n = b^3) ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m ≤ 99999 ∧ (∃ x : ℕ, m = x^2) ∧ (∃ y : ℕ, m = y^3) → m ≥ n) ∧
  n = 15625 :=
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l2454_245470


namespace NUMINAMATH_CALUDE_complex_number_with_conditions_l2454_245425

theorem complex_number_with_conditions (z : ℂ) :
  Complex.abs z = 1 →
  ∃ (y : ℝ), (3 + 4*I) * z = y*I →
  z = 4/5 - 3/5*I ∨ z = -4/5 + 3/5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_with_conditions_l2454_245425


namespace NUMINAMATH_CALUDE_lines_intersect_at_point_l2454_245403

/-- Represents a 2D point --/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Represents a line parameterization --/
structure LineParam where
  base : Point2D
  direction : Point2D

/-- First line parameterization --/
def line1 : LineParam := {
  base := { x := 2, y := 3 },
  direction := { x := 1, y := -4 }
}

/-- Second line parameterization --/
def line2 : LineParam := {
  base := { x := 4, y := -6 },
  direction := { x := 5, y := 3 }
}

/-- The intersection point of the two lines --/
def intersection_point : Point2D := {
  x := 185 / 23,
  y := 21 / 23
}

/-- Theorem stating that the given point is the intersection of the two lines --/
theorem lines_intersect_at_point :
  ∃ (t u : ℚ),
    (line1.base.x + t * line1.direction.x = intersection_point.x) ∧
    (line1.base.y + t * line1.direction.y = intersection_point.y) ∧
    (line2.base.x + u * line2.direction.x = intersection_point.x) ∧
    (line2.base.y + u * line2.direction.y = intersection_point.y) := by
  sorry

end NUMINAMATH_CALUDE_lines_intersect_at_point_l2454_245403


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l2454_245485

theorem triangle_area_inequality (a b c S_triangle : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hS : S_triangle > 0)
  (h_triangle : S_triangle = Real.sqrt ((a + b + c) * (a + b - c) * (b + c - a) * (c + a - b) / 16)) :
  1 - (8 * ((a - b)^2 + (b - c)^2 + (c - a)^2)) / (a + b + c)^2 
  ≤ (432 * S_triangle^2) / (a + b + c)^4 
  ∧ (432 * S_triangle^2) / (a + b + c)^4 
  ≤ 1 - (2 * ((a - b)^2 + (b - c)^2 + (c - a)^2)) / (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l2454_245485


namespace NUMINAMATH_CALUDE_sin_870_degrees_l2454_245490

theorem sin_870_degrees : Real.sin (870 * Real.pi / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_870_degrees_l2454_245490


namespace NUMINAMATH_CALUDE_vessel_width_calculation_l2454_245492

/-- Proves that the width of a rectangular vessel's base is 5 cm when a cube of edge 5 cm is
    immersed, causing a 2.5 cm rise in water level, given that the vessel's base length is 10 cm. -/
theorem vessel_width_calculation (cube_edge : ℝ) (vessel_length : ℝ) (water_rise : ℝ) :
  cube_edge = 5 →
  vessel_length = 10 →
  water_rise = 2.5 →
  ∃ (vessel_width : ℝ),
    vessel_width = 5 ∧
    cube_edge ^ 3 = vessel_length * vessel_width * water_rise :=
by sorry

end NUMINAMATH_CALUDE_vessel_width_calculation_l2454_245492


namespace NUMINAMATH_CALUDE_finance_club_probability_l2454_245413

theorem finance_club_probability (total_members : ℕ) (interested_ratio : ℚ) : 
  total_members = 20 →
  interested_ratio = 3 / 4 →
  let interested_members := (interested_ratio * total_members).num
  let not_interested_members := total_members - interested_members
  let prob_neither_interested := (not_interested_members / total_members) * ((not_interested_members - 1) / (total_members - 1))
  1 - prob_neither_interested = 18 / 19 := by
sorry

end NUMINAMATH_CALUDE_finance_club_probability_l2454_245413


namespace NUMINAMATH_CALUDE_veranda_width_l2454_245471

/-- Given a rectangular room with length 19 m and width 12 m, surrounded by a veranda on all sides
    with an area of 140 m², prove that the width of the veranda is 2 m. -/
theorem veranda_width (room_length : ℝ) (room_width : ℝ) (veranda_area : ℝ) :
  room_length = 19 →
  room_width = 12 →
  veranda_area = 140 →
  ∃ (w : ℝ), w = 2 ∧
    (room_length + 2 * w) * (room_width + 2 * w) - room_length * room_width = veranda_area :=
by sorry

end NUMINAMATH_CALUDE_veranda_width_l2454_245471


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2454_245450

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 16 * X + 38 = (X - 4) * q + 22 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2454_245450


namespace NUMINAMATH_CALUDE_reciprocal_square_roots_l2454_245463

theorem reciprocal_square_roots (a b c d : ℂ) : 
  (a^4 - a^2 - 5 = 0 ∧ b^4 - b^2 - 5 = 0 ∧ c^4 - c^2 - 5 = 0 ∧ d^4 - d^2 - 5 = 0) →
  (5 * (1/a)^4 + (1/a)^2 - 1 = 0 ∧ 5 * (1/b)^4 + (1/b)^2 - 1 = 0 ∧
   5 * (1/c)^4 + (1/c)^2 - 1 = 0 ∧ 5 * (1/d)^4 + (1/d)^2 - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_square_roots_l2454_245463


namespace NUMINAMATH_CALUDE_area_of_ABCD_l2454_245498

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The composed rectangle ABCD -/
def ABCD : Rectangle := { width := 10, height := 15 }

/-- One of the smaller identical rectangles -/
def SmallRect : Rectangle := { width := 5, height := 10 }

theorem area_of_ABCD : ABCD.area = 150 := by sorry

end NUMINAMATH_CALUDE_area_of_ABCD_l2454_245498


namespace NUMINAMATH_CALUDE_b2_a2_minus_a1_value_l2454_245438

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  a₂ - 4 = 4 - a₁ ∧ 1 - a₂ = a₂ - 4

-- Define the geometric sequence
def geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  4 / b₁ = b₂ / 4 ∧ b₂ / 4 = 1 / b₂ ∧ 1 / b₂ = b₃ / 1

theorem b2_a2_minus_a1_value (a₁ a₂ b₁ b₂ b₃ : ℝ) :
  arithmetic_sequence a₁ a₂ → geometric_sequence b₁ b₂ b₃ →
  (b₂ * (a₂ - a₁) = 6 ∨ b₂ * (a₂ - a₁) = -6) :=
by sorry

end NUMINAMATH_CALUDE_b2_a2_minus_a1_value_l2454_245438


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2454_245422

theorem sum_of_reciprocals (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : 1/x - 1/y = -3) : 
  x + y = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2454_245422


namespace NUMINAMATH_CALUDE_cookie_remainder_percentage_l2454_245461

/-- Proves that given 600 initial cookies, if Nicole eats 2/5 of the total and Eduardo eats 3/5 of the remaining, then 24% of the original cookies remain. -/
theorem cookie_remainder_percentage (initial_cookies : ℕ) (nicole_fraction : ℚ) (eduardo_fraction : ℚ)
  (h_initial : initial_cookies = 600)
  (h_nicole : nicole_fraction = 2 / 5)
  (h_eduardo : eduardo_fraction = 3 / 5) :
  (initial_cookies - nicole_fraction * initial_cookies - eduardo_fraction * (initial_cookies - nicole_fraction * initial_cookies)) / initial_cookies = 24 / 100 := by
  sorry

#check cookie_remainder_percentage

end NUMINAMATH_CALUDE_cookie_remainder_percentage_l2454_245461


namespace NUMINAMATH_CALUDE_median_in_80_84_interval_l2454_245454

/-- Represents the score intervals --/
inductive ScoreInterval
  | interval_65_69
  | interval_70_74
  | interval_75_79
  | interval_80_84
  | interval_85_89
  | interval_90_94

/-- The number of students in each score interval --/
def studentCount (interval : ScoreInterval) : Nat :=
  match interval with
  | .interval_65_69 => 6
  | .interval_70_74 => 10
  | .interval_75_79 => 25
  | .interval_80_84 => 30
  | .interval_85_89 => 20
  | .interval_90_94 => 10

/-- The total number of students --/
def totalStudents : Nat := 101

/-- The position of the median in the dataset --/
def medianPosition : Nat := (totalStudents + 1) / 2

/-- Theorem stating that the median score is in the 80-84 interval --/
theorem median_in_80_84_interval :
  ∃ k, k ≤ medianPosition ∧
       k > (studentCount ScoreInterval.interval_90_94 +
            studentCount ScoreInterval.interval_85_89) ∧
       k ≤ (studentCount ScoreInterval.interval_90_94 +
            studentCount ScoreInterval.interval_85_89 +
            studentCount ScoreInterval.interval_80_84) :=
  sorry

end NUMINAMATH_CALUDE_median_in_80_84_interval_l2454_245454


namespace NUMINAMATH_CALUDE_existence_of_common_element_l2454_245488

theorem existence_of_common_element (ε : ℝ) (h_ε_pos : 0 < ε) (h_ε_bound : ε < 1/2) :
  ∃ m : ℕ+, ∀ x : ℝ, ∃ i : ℕ+, ∃ k : ℤ, i.val ≤ m.val ∧ |i.val • x - k| ≤ ε :=
sorry

end NUMINAMATH_CALUDE_existence_of_common_element_l2454_245488


namespace NUMINAMATH_CALUDE_f_unique_zero_g_max_increasing_param_l2454_245415

noncomputable section

def f (x : ℝ) := (x - 2) * Real.log x + 2 * x - 3

def g (a : ℝ) (x : ℝ) := (x - a) * Real.log x + a * (x - 1) / x

theorem f_unique_zero :
  ∃! x : ℝ, x ≥ 1 ∧ f x = 0 :=
sorry

theorem g_max_increasing_param :
  (∃ (a : ℤ), ∀ x ≥ 1, Monotone (g a)) ∧
  (∀ a : ℤ, a > 6 → ∃ x ≥ 1, ¬Monotone (g a)) :=
sorry

end NUMINAMATH_CALUDE_f_unique_zero_g_max_increasing_param_l2454_245415


namespace NUMINAMATH_CALUDE_sum_of_squares_equality_l2454_245434

theorem sum_of_squares_equality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a^2 + b^2 + c^2 = a*b + b*c + c*a) :
  (a^2*b^2)/((a^2+b*c)*(b^2+a*c)) + (a^2*c^2)/((a^2+b*c)*(c^2+a*b)) + (b^2*c^2)/((b^2+a*c)*(c^2+a*b)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equality_l2454_245434


namespace NUMINAMATH_CALUDE_remainder_98_102_div_11_l2454_245476

theorem remainder_98_102_div_11 : (98 * 102) % 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_102_div_11_l2454_245476


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2454_245401

/-- Given a real number a and the imaginary unit i, if (2+ai)/(1+i) = 3+i, then a = 4 -/
theorem complex_equation_solution (a : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (2 + a * Complex.I) / (1 + Complex.I) = (3 : ℂ) + Complex.I →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2454_245401


namespace NUMINAMATH_CALUDE_at_least_one_good_certain_l2454_245444

def total_products : ℕ := 12
def good_products : ℕ := 10
def defective_products : ℕ := 2
def picked_products : ℕ := 3

theorem at_least_one_good_certain :
  Fintype.card {s : Finset (Fin total_products) // s.card = picked_products ∧ ∃ x ∈ s, x.val < good_products} =
  Fintype.card {s : Finset (Fin total_products) // s.card = picked_products} :=
sorry

end NUMINAMATH_CALUDE_at_least_one_good_certain_l2454_245444


namespace NUMINAMATH_CALUDE_angle_inequality_equivalence_l2454_245465

theorem angle_inequality_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 →
    x^2 * Real.sin θ - x * (1 - 2*x) + (1 - 3*x)^2 * Real.cos θ > 0) ↔
  (0 < θ ∧ θ < Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_equivalence_l2454_245465


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2454_245428

/-- Right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13
  right_angle : a^2 + b^2 = c^2

/-- Square inscribed in the first triangle with vertex at right angle -/
def square_at_vertex (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x ≤ t.b ∧ x / t.a = x / t.b

/-- Square inscribed in the second triangle with side on hypotenuse -/
def square_on_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.c ∧ t.a * y / t.c = y

/-- The main theorem -/
theorem inscribed_squares_ratio (t1 t2 : RightTriangle) (x y : ℝ) 
    (hx : square_at_vertex t1 x) (hy : square_on_hypotenuse t2 y) : 
    x / y = 144 / 221 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2454_245428


namespace NUMINAMATH_CALUDE_tiling_count_is_27_l2454_245475

/-- Represents a 2 × 24 grid divided into three 3 × 8 subrectangles -/
structure Grid :=
  (subrectangles : Fin 3 → Unit)

/-- Represents the number of ways to tile a single 3 × 8 subrectangle -/
def subrectangle_tiling_count : ℕ := 3

/-- Calculates the total number of ways to tile the entire 2 × 24 grid -/
def total_tiling_count (g : Grid) : ℕ :=
  subrectangle_tiling_count ^ 3

/-- Theorem stating that the total number of tiling ways is 27 -/
theorem tiling_count_is_27 (g : Grid) : total_tiling_count g = 27 := by
  sorry

end NUMINAMATH_CALUDE_tiling_count_is_27_l2454_245475


namespace NUMINAMATH_CALUDE_equation_proof_l2454_245408

theorem equation_proof (a b : ℝ) (h : a - 2 * b = 4) : 3 - a + 2 * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2454_245408


namespace NUMINAMATH_CALUDE_wasted_fruits_and_vegetables_is_15_l2454_245491

/-- Calculates the amount of wasted fruits and vegetables in pounds -/
def wasted_fruits_and_vegetables (meat_pounds : ℕ) (meat_price : ℚ) 
  (bread_pounds : ℕ) (bread_price : ℚ) (janitor_hours : ℕ) (janitor_normal_wage : ℚ)
  (minimum_wage : ℚ) (work_hours : ℕ) (fruit_veg_price : ℚ) : ℚ :=
  let meat_cost := meat_pounds * meat_price
  let bread_cost := bread_pounds * bread_price
  let janitor_cost := janitor_hours * (janitor_normal_wage * 1.5)
  let total_earnings := work_hours * minimum_wage
  let remaining_cost := total_earnings - (meat_cost + bread_cost + janitor_cost)
  remaining_cost / fruit_veg_price

theorem wasted_fruits_and_vegetables_is_15 :
  wasted_fruits_and_vegetables 20 5 60 (3/2) 10 10 8 50 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_wasted_fruits_and_vegetables_is_15_l2454_245491


namespace NUMINAMATH_CALUDE_curve_and_line_properties_l2454_245449

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 1 = 0

-- Define the distance ratio condition
def distance_ratio (x y : ℝ) : Prop :=
  ((x + 1)^2 + y^2) = 2 * ((x - 1)^2 + y^2)

-- Define the line l
def line_l (x y : ℝ) : Prop := x = 1 ∨ y = 2

-- Main theorem
theorem curve_and_line_properties :
  ∀ x y : ℝ,
  distance_ratio x y →
  (C x y ∧
   (∃ x' y' : ℝ, C x' y' ∧ line_l x' y' ∧
    (x' - 1)^2 + (y' - 2)^2 = 4)) →
  line_l x y :=
sorry

end NUMINAMATH_CALUDE_curve_and_line_properties_l2454_245449


namespace NUMINAMATH_CALUDE_mean_of_seven_numbers_l2454_245405

theorem mean_of_seven_numbers (x y : ℝ) :
  (6 + 14 + x + 17 + 9 + y + 10) / 7 = 13 → x + y = 35 := by
sorry

end NUMINAMATH_CALUDE_mean_of_seven_numbers_l2454_245405


namespace NUMINAMATH_CALUDE_profit_difference_l2454_245421

-- Define the types of statues
inductive StatueType
| Giraffe
| Elephant
| Rhinoceros

-- Define the properties of each statue type
def jade_required (s : StatueType) : ℕ :=
  match s with
  | StatueType.Giraffe => 120
  | StatueType.Elephant => 240
  | StatueType.Rhinoceros => 180

def original_price (s : StatueType) : ℕ :=
  match s with
  | StatueType.Giraffe => 150
  | StatueType.Elephant => 350
  | StatueType.Rhinoceros => 250

-- Define the bulk discount
def bulk_discount : ℚ := 0.9

-- Define the total jade available
def total_jade : ℕ := 1920

-- Calculate the number of statues that can be made
def num_statues (s : StatueType) : ℕ :=
  total_jade / jade_required s

-- Calculate the revenue for a statue type
def revenue (s : StatueType) : ℚ :=
  if num_statues s > 3 then
    (num_statues s : ℚ) * (original_price s : ℚ) * bulk_discount
  else
    (num_statues s : ℚ) * (original_price s : ℚ)

-- Theorem to prove
theorem profit_difference : 
  revenue StatueType.Elephant - revenue StatueType.Rhinoceros = 270 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_l2454_245421


namespace NUMINAMATH_CALUDE_max_coins_distribution_l2454_245458

theorem max_coins_distribution (n : ℕ) (h1 : n < 150) 
  (h2 : ∃ k : ℕ, n = 8 * k + 4) : n ≤ 148 := by
  sorry

end NUMINAMATH_CALUDE_max_coins_distribution_l2454_245458


namespace NUMINAMATH_CALUDE_age_difference_l2454_245448

theorem age_difference (a b c : ℕ) : 
  b = 2 * c → 
  a + b + c = 22 → 
  b = 8 → 
  a - b = 2 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l2454_245448


namespace NUMINAMATH_CALUDE_at_least_one_positive_negation_l2454_245469

theorem at_least_one_positive_negation (a b c : ℝ) :
  (¬ (a > 0 ∨ b > 0 ∨ c > 0)) ↔ (a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_positive_negation_l2454_245469


namespace NUMINAMATH_CALUDE_cos_equality_problem_l2454_245499

theorem cos_equality_problem (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → (Real.cos (n * π / 180) = Real.cos (832 * π / 180) ↔ n = 112) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_problem_l2454_245499


namespace NUMINAMATH_CALUDE_toys_problem_toys_problem_unique_l2454_245439

/-- Given the number of toys for Kamari, calculates the total number of toys for all three children. -/
def total_toys (kamari_toys : ℕ) : ℕ :=
  kamari_toys + (kamari_toys + 30) + (2 * kamari_toys)

/-- Theorem stating that given the conditions, the total number of toys is 290. -/
theorem toys_problem : ∃ (k : ℕ), k + (k + 30) = 160 ∧ total_toys k = 290 :=
  sorry

/-- Corollary: The solution to the problem exists and is unique. -/
theorem toys_problem_unique : ∃! (k : ℕ), k + (k + 30) = 160 ∧ total_toys k = 290 :=
  sorry

end NUMINAMATH_CALUDE_toys_problem_toys_problem_unique_l2454_245439


namespace NUMINAMATH_CALUDE_sqrt_triangle_exists_abs_diff_plus_one_triangle_exists_l2454_245442

-- Define a triangle with sides a, b, and c
structure Triangle :=
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : c > 0)
  (triangle_inequality₁ : a + b > c)
  (triangle_inequality₂ : b + c > a)
  (triangle_inequality₃ : c + a > b)

-- Theorem 1: A triangle with sides √a, √b, and √c always exists
theorem sqrt_triangle_exists (t : Triangle) : 
  ∃ (t' : Triangle), t'.a = Real.sqrt t.a ∧ t'.b = Real.sqrt t.b ∧ t'.c = Real.sqrt t.c :=
sorry

-- Theorem 2: A triangle with sides |a-b|+1, |b-c|+1, and |c-a|+1 always exists
theorem abs_diff_plus_one_triangle_exists (t : Triangle) :
  ∃ (t' : Triangle), t'.a = |t.a - t.b| + 1 ∧ t'.b = |t.b - t.c| + 1 ∧ t'.c = |t.c - t.a| + 1 :=
sorry

end NUMINAMATH_CALUDE_sqrt_triangle_exists_abs_diff_plus_one_triangle_exists_l2454_245442


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2454_245497

theorem fraction_to_decimal : (13 : ℚ) / 200 = (52 : ℚ) / 100 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2454_245497


namespace NUMINAMATH_CALUDE_c_rent_share_l2454_245406

/-- Represents the usage of the pasture by a person -/
structure Usage where
  oxen : ℕ
  months : ℕ

/-- Calculates the ox-months for a given usage -/
def oxMonths (u : Usage) : ℕ := u.oxen * u.months

/-- Represents the rental situation of the pasture -/
structure PastureRental where
  a : Usage
  b : Usage
  c : Usage
  totalRent : ℕ

/-- Calculates the total ox-months for all users -/
def totalOxMonths (r : PastureRental) : ℕ :=
  oxMonths r.a + oxMonths r.b + oxMonths r.c

/-- Calculates the share of rent for a given usage -/
def rentShare (r : PastureRental) (u : Usage) : ℚ :=
  (oxMonths u : ℚ) / (totalOxMonths r : ℚ) * (r.totalRent : ℚ)

theorem c_rent_share (r : PastureRental) : 
  r.a = { oxen := 10, months := 7 } →
  r.b = { oxen := 12, months := 5 } →
  r.c = { oxen := 15, months := 3 } →
  r.totalRent = 245 →
  rentShare r r.c = 63 := by
  sorry

end NUMINAMATH_CALUDE_c_rent_share_l2454_245406


namespace NUMINAMATH_CALUDE_probability_is_seven_ninety_sixths_l2454_245410

/-- Triangle PQR with given side lengths -/
structure Triangle :=
  (PQ : ℝ)
  (QR : ℝ)
  (PR : ℝ)

/-- The specific triangle in the problem -/
def problemTriangle : Triangle :=
  { PQ := 7,
    QR := 24,
    PR := 25 }

/-- A point randomly selected inside the triangle -/
def S : Type := Unit

/-- The midpoint of side QR -/
def M (t : Triangle) : ℝ × ℝ := sorry

/-- Function to determine if a point is closer to M than to P or R -/
def closerToM (t : Triangle) (s : S) : Prop := sorry

/-- The probability of the event -/
def probability (t : Triangle) : ℝ := sorry

/-- The main theorem -/
theorem probability_is_seven_ninety_sixths :
  probability problemTriangle = 7 / 96 := by sorry

end NUMINAMATH_CALUDE_probability_is_seven_ninety_sixths_l2454_245410


namespace NUMINAMATH_CALUDE_evaluate_expression_l2454_245417

theorem evaluate_expression : 8^7 + 8^7 + 8^7 - 8^7 = 8^8 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2454_245417


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l2454_245433

/-- The number of ways to distribute 10 colored balls into two boxes -/
def distribute_balls : ℕ :=
  Nat.choose 10 4

/-- The total number of balls -/
def total_balls : ℕ := 10

/-- The number of red balls -/
def red_balls : ℕ := 5

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of green balls -/
def green_balls : ℕ := 2

/-- The capacity of the smaller box -/
def small_box_capacity : ℕ := 4

/-- The capacity of the larger box -/
def large_box_capacity : ℕ := 6

theorem ball_distribution_theorem :
  distribute_balls = 210 ∧
  total_balls = red_balls + white_balls + green_balls ∧
  total_balls = small_box_capacity + large_box_capacity :=
sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l2454_245433


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_and_height_l2454_245418

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (a b c d : Point3D) : ℝ := sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (a b c d : Point3D) : ℝ := sorry

/-- Theorem stating the volume and height of a specific tetrahedron -/
theorem specific_tetrahedron_volume_and_height :
  let a₁ : Point3D := ⟨-3, -5, 6⟩
  let a₂ : Point3D := ⟨2, 1, -4⟩
  let a₃ : Point3D := ⟨0, -3, -1⟩
  let a₄ : Point3D := ⟨-5, 2, -8⟩
  tetrahedronVolume a₁ a₂ a₃ a₄ = 191 / 6 ∧
  tetrahedronHeight a₄ a₁ a₂ a₃ = Real.sqrt (191 / 3) := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_and_height_l2454_245418


namespace NUMINAMATH_CALUDE_max_value_of_f_l2454_245424

open Real

noncomputable def f (θ : ℝ) : ℝ := tan (θ / 2) * (1 - sin θ)

theorem max_value_of_f :
  ∃ (θ_max : ℝ), 
    -π/2 < θ_max ∧ θ_max < π/2 ∧
    θ_max = 2 * arctan ((-2 + Real.sqrt 7) / 3) ∧
    ∀ (θ : ℝ), -π/2 < θ ∧ θ < π/2 → f θ ≤ f θ_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2454_245424


namespace NUMINAMATH_CALUDE_dark_tiles_three_fourths_l2454_245411

/-- Represents a tiled floor with a repeating pattern -/
structure TiledFloor :=
  (pattern_size : Nat)
  (corner_dark_tiles : Nat)
  (corner_size : Nat)

/-- The fraction of dark tiles in the entire floor -/
def dark_tile_fraction (floor : TiledFloor) : Rat :=
  floor.corner_dark_tiles / (floor.corner_size * floor.corner_size)

/-- Theorem stating that for a floor with a 4x4 repeating pattern and 3 dark tiles
    in a 2x2 corner section, 3/4 of the entire floor is made of darker tiles -/
theorem dark_tiles_three_fourths (floor : TiledFloor)
  (h1 : floor.pattern_size = 4)
  (h2 : floor.corner_size = 2)
  (h3 : floor.corner_dark_tiles = 3) :
  dark_tile_fraction floor = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_dark_tiles_three_fourths_l2454_245411


namespace NUMINAMATH_CALUDE_gcd_consecutive_b_terms_is_one_l2454_245462

def b (n : ℕ) : ℕ := 2 * n.factorial + n

theorem gcd_consecutive_b_terms_is_one (n : ℕ) : 
  Nat.gcd (b n) (b (n + 1)) = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_consecutive_b_terms_is_one_l2454_245462


namespace NUMINAMATH_CALUDE_farm_ratio_change_l2454_245467

def farm_problem (initial_horses initial_cows : ℕ) : Prop :=
  let horses_after := initial_horses - 15
  let cows_after := initial_cows + 15
  initial_horses = 5 * initial_cows ∧
  horses_after = cows_after + 50 ∧
  horses_after / 5 = 17 ∧
  cows_after / 5 = 7

theorem farm_ratio_change :
  ∃ (initial_horses initial_cows : ℕ), farm_problem initial_horses initial_cows :=
by
  sorry

end NUMINAMATH_CALUDE_farm_ratio_change_l2454_245467


namespace NUMINAMATH_CALUDE_profit_ratio_l2454_245477

def investment_p : ℕ := 500000
def investment_q : ℕ := 1000000

theorem profit_ratio (p q : ℕ) (h : p = investment_p ∧ q = investment_q) :
  (p : ℚ) / (p + q : ℚ) = 1 / 3 ∧ (q : ℚ) / (p + q : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_ratio_l2454_245477


namespace NUMINAMATH_CALUDE_john_ray_difference_l2454_245443

/-- The number of chickens each person took -/
structure ChickenCount where
  mary : ℕ
  john : ℕ
  ray : ℕ

/-- The conditions of the chicken problem -/
def chicken_problem (c : ChickenCount) : Prop :=
  c.john = c.mary + 5 ∧
  c.mary = c.ray + 6 ∧
  c.ray = 10

/-- The theorem stating John took 11 more chickens than Ray -/
theorem john_ray_difference (c : ChickenCount) 
  (h : chicken_problem c) : c.john - c.ray = 11 := by
  sorry


end NUMINAMATH_CALUDE_john_ray_difference_l2454_245443


namespace NUMINAMATH_CALUDE_greening_problem_l2454_245420

/-- The greening problem -/
theorem greening_problem 
  (total_area : ℝ) 
  (team_a_speed : ℝ) 
  (team_b_speed : ℝ) 
  (team_a_cost : ℝ) 
  (team_b_cost : ℝ) 
  (max_cost : ℝ) 
  (h1 : total_area = 1800) 
  (h2 : team_a_speed = 2 * team_b_speed) 
  (h3 : 400 / team_a_speed + 4 = 400 / team_b_speed) 
  (h4 : team_a_cost = 0.4) 
  (h5 : team_b_cost = 0.25) 
  (h6 : max_cost = 8) :
  ∃ (team_a_area team_b_area min_days : ℝ),
    team_a_area = 100 ∧ 
    team_b_area = 50 ∧ 
    min_days = 10 ∧
    (∀ y : ℝ, y ≥ min_days → 
      team_a_cost * y + team_b_cost * ((total_area - team_a_area * y) / team_b_area) ≤ max_cost) := by
  sorry

end NUMINAMATH_CALUDE_greening_problem_l2454_245420


namespace NUMINAMATH_CALUDE_campground_distance_l2454_245451

/-- The distance traveled by Sue's family to the campground -/
def distance_to_campground (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: The distance to the campground is 300 miles -/
theorem campground_distance : 
  distance_to_campground 60 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_campground_distance_l2454_245451


namespace NUMINAMATH_CALUDE_quadratic_roots_max_reciprocal_sum_l2454_245481

theorem quadratic_roots_max_reciprocal_sum (t q r₁ r₂ : ℝ) : 
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 1003 → r₁^n + r₂^n = r₁ + r₂) →
  r₁ * r₂ = q →
  r₁ + r₂ = t →
  r₁ ≠ 0 →
  r₂ ≠ 0 →
  (1 / r₁^1004 + 1 / r₂^1004) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_max_reciprocal_sum_l2454_245481


namespace NUMINAMATH_CALUDE_remove_parentheses_l2454_245480

theorem remove_parentheses (x y z : ℝ) : -(x - (y - z)) = -x + y - z := by
  sorry

end NUMINAMATH_CALUDE_remove_parentheses_l2454_245480


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2454_245447

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (10 + 3 * z) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2454_245447


namespace NUMINAMATH_CALUDE_binomial_parameters_unique_l2454_245423

/-- A random variable following a binomial distribution -/
structure BinomialRandomVariable where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (ξ : BinomialRandomVariable) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRandomVariable) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_parameters_unique 
  (ξ : BinomialRandomVariable) 
  (h_exp : expectation ξ = 2.4)
  (h_var : variance ξ = 1.44) : 
  ξ.n = 6 ∧ ξ.p = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_parameters_unique_l2454_245423


namespace NUMINAMATH_CALUDE_family_gathering_problem_l2454_245419

theorem family_gathering_problem (P : ℕ) : 
  P / 2 = P - 10 →
  P / 2 + P / 4 + (P - (P / 2 + P / 4)) = P →
  P = 20 := by
sorry

end NUMINAMATH_CALUDE_family_gathering_problem_l2454_245419


namespace NUMINAMATH_CALUDE_zorg_vamp_and_not_wook_l2454_245404

-- Define the types
variable (U : Type) -- Universe set
variable (Zorg Xyon Wook Vamp : Set U)

-- Define the conditions
variable (h1 : Zorg ⊆ Xyon)
variable (h2 : Wook ⊆ Xyon)
variable (h3 : Vamp ⊆ Zorg)
variable (h4 : Wook ⊆ Vamp)
variable (h5 : Zorg ∩ Wook = ∅)

-- Theorem to prove
theorem zorg_vamp_and_not_wook : 
  Zorg ⊆ Vamp ∧ Zorg ∩ Wook = ∅ :=
by sorry

end NUMINAMATH_CALUDE_zorg_vamp_and_not_wook_l2454_245404


namespace NUMINAMATH_CALUDE_min_value_inequality_l2454_245496

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 1/b = Real.sqrt 6) : 1/a^2 + 2/b^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2454_245496


namespace NUMINAMATH_CALUDE_minimum_value_quadratic_l2454_245493

theorem minimum_value_quadratic (a : ℝ) : 
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ 
    (∀ (y : ℝ), y ∈ Set.Icc 0 1 → x^2 - 2*a*x + a - 1 ≤ y^2 - 2*a*y + a - 1) ∧
    x^2 - 2*a*x + a - 1 = -2) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_quadratic_l2454_245493


namespace NUMINAMATH_CALUDE_parabola_intersection_range_l2454_245453

/-- The parabola function -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - (4*m + 1)*x + 2*m - 1

theorem parabola_intersection_range (m : ℝ) :
  (∃ x₁ x₂, x₁ < 2 ∧ x₂ > 2 ∧ f m x₁ = 0 ∧ f m x₂ = 0) →  -- Intersects x-axis at two points
  (f m 0 < -1/2) →  -- Intersects y-axis below (0, -1/2)
  1/6 < m ∧ m < 1/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_range_l2454_245453


namespace NUMINAMATH_CALUDE_carlos_class_size_l2454_245452

theorem carlos_class_size :
  ∃! b : ℕ, 80 < b ∧ b < 150 ∧
  ∃ k₁ : ℕ, b = 3 * k₁ - 2 ∧
  ∃ k₂ : ℕ, b = 4 * k₂ - 3 ∧
  ∃ k₃ : ℕ, b = 5 * k₃ - 4 ∧
  b = 121 := by
sorry

end NUMINAMATH_CALUDE_carlos_class_size_l2454_245452


namespace NUMINAMATH_CALUDE_gcd_2703_1113_l2454_245426

theorem gcd_2703_1113 : Nat.gcd 2703 1113 = 159 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2703_1113_l2454_245426


namespace NUMINAMATH_CALUDE_delta_value_l2454_245456

theorem delta_value (Δ : ℤ) (h : 4 * (-3) = Δ + 3) : Δ = -15 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l2454_245456


namespace NUMINAMATH_CALUDE_counterexample_exists_l2454_245441

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ Nat.Prime (n - 2) :=
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2454_245441


namespace NUMINAMATH_CALUDE_annual_fixed_costs_satisfy_profit_equation_l2454_245407

/-- Represents the annual fixed costs for Model X -/
def annual_fixed_costs : ℝ := 50200000

/-- Represents the desired annual profit -/
def desired_profit : ℝ := 30500000

/-- Represents the selling price per unit -/
def selling_price : ℝ := 9035

/-- Represents the variable cost per unit -/
def variable_cost : ℝ := 5000

/-- Represents the number of units sold -/
def units_sold : ℝ := 20000

/-- The profit equation -/
def profit_equation (fixed_costs : ℝ) : ℝ :=
  selling_price * units_sold - variable_cost * units_sold - fixed_costs

/-- Theorem stating that the annual fixed costs satisfy the profit equation -/
theorem annual_fixed_costs_satisfy_profit_equation :
  profit_equation annual_fixed_costs = desired_profit := by
  sorry

end NUMINAMATH_CALUDE_annual_fixed_costs_satisfy_profit_equation_l2454_245407


namespace NUMINAMATH_CALUDE_parabolas_intersection_l2454_245472

-- Define the parabolas
def parabola1 (a x y : ℝ) : Prop := y = x^2 + x + a
def parabola2 (a x y : ℝ) : Prop := x = 4*y^2 + 3*y + a

-- Define the condition of four intersection points
def has_four_intersections (a : ℝ) : Prop := ∃ x1 y1 x2 y2 x3 y3 x4 y4 : ℝ,
  (parabola1 a x1 y1 ∧ parabola2 a x1 y1) ∧
  (parabola1 a x2 y2 ∧ parabola2 a x2 y2) ∧
  (parabola1 a x3 y3 ∧ parabola2 a x3 y3) ∧
  (parabola1 a x4 y4 ∧ parabola2 a x4 y4) ∧
  (x1 ≠ x2 ∨ y1 ≠ y2) ∧ (x1 ≠ x3 ∨ y1 ≠ y3) ∧ (x1 ≠ x4 ∨ y1 ≠ y4) ∧
  (x2 ≠ x3 ∨ y2 ≠ y3) ∧ (x2 ≠ x4 ∨ y2 ≠ y4) ∧ (x3 ≠ x4 ∨ y3 ≠ y4)

-- Define the range of a
def a_range (a : ℝ) : Prop := (a < -1/2 ∨ (-1/2 < a ∧ a < -7/16))

-- Define the condition for points being concyclic
def concyclic (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : Prop :=
  ∃ cx cy r : ℝ, 
    (x1 - cx)^2 + (y1 - cy)^2 = r^2 ∧
    (x2 - cx)^2 + (y2 - cy)^2 = r^2 ∧
    (x3 - cx)^2 + (y3 - cy)^2 = r^2 ∧
    (x4 - cx)^2 + (y4 - cy)^2 = r^2

-- The main theorem
theorem parabolas_intersection (a : ℝ) :
  has_four_intersections a →
  (a_range a ∧
   ∃ x1 y1 x2 y2 x3 y3 x4 y4 : ℝ,
     (parabola1 a x1 y1 ∧ parabola2 a x1 y1) ∧
     (parabola1 a x2 y2 ∧ parabola2 a x2 y2) ∧
     (parabola1 a x3 y3 ∧ parabola2 a x3 y3) ∧
     (parabola1 a x4 y4 ∧ parabola2 a x4 y4) ∧
     concyclic x1 y1 x2 y2 x3 y3 x4 y4 ∧
     ∃ cx cy : ℝ, cx = -3/8 ∧ cy = 1/8) :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l2454_245472


namespace NUMINAMATH_CALUDE_max_distance_sum_l2454_245494

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define a line passing through F₁
def line_through_F₁ (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x + 1)

-- Define the intersection points A and B
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ ellipse x y ∧ line_through_F₁ m x y}

-- Statement of the theorem
theorem max_distance_sum :
  ∀ (m : ℝ), ∃ (A B : ℝ × ℝ),
    A ∈ intersection_points m ∧
    B ∈ intersection_points m ∧
    A ≠ B ∧
    (∀ (A' B' : ℝ × ℝ),
      A' ∈ intersection_points m →
      B' ∈ intersection_points m →
      A' ≠ B' →
      dist A' F₂ + dist B' F₂ ≤ 5) ∧
    (∃ (m' : ℝ), ∃ (A' B' : ℝ × ℝ),
      A' ∈ intersection_points m' ∧
      B' ∈ intersection_points m' ∧
      A' ≠ B' ∧
      dist A' F₂ + dist B' F₂ = 5) :=
sorry


end NUMINAMATH_CALUDE_max_distance_sum_l2454_245494


namespace NUMINAMATH_CALUDE_find_number_l2454_245486

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 69 ∧ x = 7 := by sorry

end NUMINAMATH_CALUDE_find_number_l2454_245486


namespace NUMINAMATH_CALUDE_ad_difference_l2454_245432

/-- Represents the number of ads on each web page -/
structure WebPageAds where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The conditions of the problem -/
def adConditions (w : WebPageAds) : Prop :=
  w.first = 12 ∧
  w.second = 2 * w.first ∧
  w.third > w.second ∧
  w.fourth = (3 * w.second) / 4 ∧
  (2 * (w.first + w.second + w.third + w.fourth)) / 3 = 68

theorem ad_difference (w : WebPageAds) (h : adConditions w) : 
  w.third - w.second = 24 := by
sorry

end NUMINAMATH_CALUDE_ad_difference_l2454_245432


namespace NUMINAMATH_CALUDE_percentage_problem_l2454_245474

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.6 * x = 240 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2454_245474


namespace NUMINAMATH_CALUDE_total_coins_is_21_l2454_245495

/-- Represents the coin distribution pattern between Pete and Paul -/
def coin_distribution (x : ℕ) : Prop :=
  ∃ (paul_coins : ℕ) (pete_coins : ℕ),
    paul_coins = x ∧
    pete_coins = 6 * x ∧
    pete_coins = x * (x + 1) * (x + 2) / 6

/-- The total number of coins is 21 -/
theorem total_coins_is_21 : ∃ (x : ℕ), coin_distribution x ∧ x + 6 * x = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_is_21_l2454_245495


namespace NUMINAMATH_CALUDE_common_tangent_existence_l2454_245427

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = x^2 + 169/100

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = y^2 + 49/4

/-- Common tangent line L -/
def L (a b c : ℕ) (x y : ℝ) : Prop := a * x + b * y = c

theorem common_tangent_existence :
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (Nat.gcd a (Nat.gcd b c) = 1) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      P₁ x₁ y₁ ∧ P₂ x₂ y₂ ∧
      L a b c x₁ y₁ ∧ L a b c x₂ y₂) ∧
    a + b + c = 52 :=
by sorry

end NUMINAMATH_CALUDE_common_tangent_existence_l2454_245427


namespace NUMINAMATH_CALUDE_gcd_sum_lcm_l2454_245400

theorem gcd_sum_lcm (a b : ℤ) : Nat.gcd (a + b).natAbs (Nat.lcm a.natAbs b.natAbs) = Nat.gcd a.natAbs b.natAbs := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_lcm_l2454_245400


namespace NUMINAMATH_CALUDE_profit_loss_ratio_l2454_245402

theorem profit_loss_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_profit_loss_ratio_l2454_245402


namespace NUMINAMATH_CALUDE_pythagorean_triple_for_eleven_l2454_245412

theorem pythagorean_triple_for_eleven : ∃ (b c : ℕ), 11^2 + b^2 = c^2 ∧ c = 61 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_for_eleven_l2454_245412


namespace NUMINAMATH_CALUDE_combined_work_time_is_14_minutes_l2454_245460

/-- Represents the time taken to complete a job when working together, given individual work rates -/
def combined_work_time (george_rate : ℚ) (abe_rate : ℚ) (carla_rate : ℚ) : ℚ :=
  1 / (george_rate + abe_rate + carla_rate)

/-- Theorem stating that given the individual work rates, the combined work time is 14 minutes -/
theorem combined_work_time_is_14_minutes :
  combined_work_time (1/70) (1/30) (1/42) = 14 := by
  sorry

#eval combined_work_time (1/70) (1/30) (1/42)

end NUMINAMATH_CALUDE_combined_work_time_is_14_minutes_l2454_245460


namespace NUMINAMATH_CALUDE_minimum_values_l2454_245459

theorem minimum_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x^2 + y^2 ≥ 1/2) ∧ (1/x + 1/y + 1/(x*y) ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_minimum_values_l2454_245459


namespace NUMINAMATH_CALUDE_not_divisible_by_81_l2454_245487

theorem not_divisible_by_81 (n : ℤ) : ¬ (81 ∣ (n^3 - 9*n + 27)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_81_l2454_245487


namespace NUMINAMATH_CALUDE_soda_machine_leak_time_difference_l2454_245455

/-- 
Given a machine that normally fills a barrel of soda in 3 minutes, 
but takes 5 minutes when leaking, prove that it will take 2n minutes 
longer to fill n barrels when leaking, given that it takes 24 minutes 
longer for 12 barrels.
-/
theorem soda_machine_leak_time_difference (n : ℕ) : 
  (3 : ℝ) = normal_fill_time_per_barrel →
  (5 : ℝ) = leaking_fill_time_per_barrel →
  24 = 12 * (leaking_fill_time_per_barrel - normal_fill_time_per_barrel) →
  2 * n = n * (leaking_fill_time_per_barrel - normal_fill_time_per_barrel) :=
by sorry


end NUMINAMATH_CALUDE_soda_machine_leak_time_difference_l2454_245455


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2454_245435

theorem arithmetic_calculations :
  (3.21 - 1.05 - 1.95 = 0.21) ∧
  (15 - (2.95 + 8.37) = 3.68) ∧
  (14.6 * 2 - 0.6 * 2 = 28) ∧
  (0.25 * 1.25 * 32 = 10) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2454_245435
