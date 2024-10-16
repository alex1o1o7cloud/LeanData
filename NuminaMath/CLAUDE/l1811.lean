import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l1811_181173

-- Define the base 10 logarithm
noncomputable def log (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem problem_statement : (2/3)^0 + log 2 + log 5 = 2 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1811_181173


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l1811_181194

theorem complex_magnitude_equation (t : ℝ) (h : t > 0) : 
  Complex.abs (8 + t * Complex.I) = 15 → t = Real.sqrt 161 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l1811_181194


namespace NUMINAMATH_CALUDE_solution_product_l1811_181189

theorem solution_product (a b : ℝ) : 
  (a - 3) * (3 * a + 7) = a^2 - 16 * a + 55 →
  (b - 3) * (3 * b + 7) = b^2 - 16 * b + 55 →
  a ≠ b →
  (a + 2) * (b + 2) = -54 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l1811_181189


namespace NUMINAMATH_CALUDE_two_and_three_digit_number_product_l1811_181164

theorem two_and_three_digit_number_product : ∃! (x y : ℕ), 
  10 ≤ x ∧ x < 100 ∧ 
  100 ≤ y ∧ y < 1000 ∧ 
  1000 * x + y = 9 * x * y ∧
  x + y = 126 := by
sorry

end NUMINAMATH_CALUDE_two_and_three_digit_number_product_l1811_181164


namespace NUMINAMATH_CALUDE_work_completion_time_l1811_181146

theorem work_completion_time 
  (n : ℕ) -- number of persons
  (t : ℝ) -- time to complete the work
  (h : t = 12) -- given condition that work is completed in 12 days
  : (2 * n) * (3 : ℝ) = n * t / 2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1811_181146


namespace NUMINAMATH_CALUDE_nicoles_age_l1811_181109

theorem nicoles_age (nicole_age sally_age : ℕ) : 
  nicole_age = 3 * sally_age →
  nicole_age + sally_age + 8 = 40 →
  nicole_age = 24 := by
sorry

end NUMINAMATH_CALUDE_nicoles_age_l1811_181109


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l1811_181150

noncomputable def f (x : ℝ) := 2 * x^3 - 2 * x^2 + 3

theorem f_derivative_at_2 : 
  (deriv f) 2 = 16 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l1811_181150


namespace NUMINAMATH_CALUDE_joan_driving_speed_l1811_181112

theorem joan_driving_speed 
  (total_distance : ℝ) 
  (total_trip_time : ℝ) 
  (lunch_break : ℝ) 
  (bathroom_break : ℝ) 
  (num_bathroom_breaks : ℕ) :
  total_distance = 480 →
  total_trip_time = 9 →
  lunch_break = 0.5 →
  bathroom_break = 0.25 →
  num_bathroom_breaks = 2 →
  let total_break_time := lunch_break + num_bathroom_breaks * bathroom_break
  let driving_time := total_trip_time - total_break_time
  let speed := total_distance / driving_time
  speed = 60 := by sorry

end NUMINAMATH_CALUDE_joan_driving_speed_l1811_181112


namespace NUMINAMATH_CALUDE_range_of_a_l1811_181134

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |1/2 * x^3 - a*x| ≤ 1) ↔ -1/2 ≤ a ∧ a ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1811_181134


namespace NUMINAMATH_CALUDE_opposite_sign_power_l1811_181195

theorem opposite_sign_power (x y : ℝ) : 
  (|x + 3| * (y - 2)^2 ≤ 0 ∧ |x + 3| + (y - 2)^2 = 0) → x^y = 9 := by sorry

end NUMINAMATH_CALUDE_opposite_sign_power_l1811_181195


namespace NUMINAMATH_CALUDE_special_circle_equation_l1811_181183

/-- A circle with specific properties -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_in_first_quadrant : 0 < center.1 ∧ 0 < center.2
  tangent_to_line : |4 * center.1 - 3 * center.2| = 5 * radius
  tangent_to_x_axis : center.2 = radius
  radius_is_one : radius = 1

/-- The standard equation of a circle given its center and radius -/
def circle_equation (c : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  (x - c.1)^2 + (y - c.2)^2 = r^2

/-- Theorem stating that a SpecialCircle has the standard equation (x-2)^2 + (y-1)^2 = 1 -/
theorem special_circle_equation (C : SpecialCircle) (x y : ℝ) :
  circle_equation (2, 1) 1 x y ↔ circle_equation C.center C.radius x y :=
sorry

end NUMINAMATH_CALUDE_special_circle_equation_l1811_181183


namespace NUMINAMATH_CALUDE_book_distribution_l1811_181138

theorem book_distribution (x : ℕ) : x > 0 → (x / 3 : ℚ) + 2 = (x - 9 : ℚ) / 2 :=
  sorry

end NUMINAMATH_CALUDE_book_distribution_l1811_181138


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l1811_181126

theorem quadratic_roots_difference_squared : 
  ∀ Θ θ : ℝ, 
  (Θ^2 - 3*Θ + 1 = 0) → 
  (θ^2 - 3*θ + 1 = 0) → 
  (Θ ≠ θ) → 
  (Θ - θ)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l1811_181126


namespace NUMINAMATH_CALUDE_brick_length_calculation_brick_length_is_20cm_l1811_181111

/-- Given a courtyard and brick specifications, calculate the length of each brick. -/
theorem brick_length_calculation (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_width : ℝ) (total_bricks : ℕ) : ℝ :=
  let courtyard_area := courtyard_length * courtyard_width * 10000 -- Convert to cm²
  let brick_area := courtyard_area / total_bricks
  brick_area / brick_width

/-- Prove that for the given specifications, the brick length is 20 cm. -/
theorem brick_length_is_20cm : 
  brick_length_calculation 30 16 10 24000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_calculation_brick_length_is_20cm_l1811_181111


namespace NUMINAMATH_CALUDE_current_age_ratio_l1811_181180

def age_ratio (p q : ℕ) : ℚ := p / q

theorem current_age_ratio :
  ∀ (p q : ℕ),
  (∃ k : ℕ, p = k * q) →
  (p + 11 = 2 * (q + 11)) →
  (p = 30 + 3) →
  age_ratio p q = 3 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_current_age_ratio_l1811_181180


namespace NUMINAMATH_CALUDE_unique_solution_power_sum_square_l1811_181198

theorem unique_solution_power_sum_square :
  ∃! (x y z : ℕ+), 2^(x.val) + 3^(y.val) = z.val^2 ∧ x = 4 ∧ y = 2 ∧ z = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_sum_square_l1811_181198


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l1811_181105

theorem recurring_decimal_sum : 
  (1 : ℚ) / 3 + 4 / 99 + 5 / 999 = 1134 / 2997 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l1811_181105


namespace NUMINAMATH_CALUDE_quadratic_symmetry_inequality_l1811_181177

/-- A quadratic function with axis of symmetry at x = 1 satisfies c > 2b -/
theorem quadratic_symmetry_inequality (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (2 - x)^2 + b * (2 - x) + c) →
  c > 2 * b := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_inequality_l1811_181177


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1811_181159

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - a ≤ -3) → 
  (a ≤ -6 ∨ a ≥ 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1811_181159


namespace NUMINAMATH_CALUDE_sin_squared_plus_sin_minus_two_range_l1811_181116

theorem sin_squared_plus_sin_minus_two_range :
  ∀ x : ℝ, -9/4 ≤ Real.sin x ^ 2 + Real.sin x - 2 ∧
  (∃ x : ℝ, Real.sin x ^ 2 + Real.sin x - 2 = -9/4) ∧
  (∃ x : ℝ, Real.sin x ^ 2 + Real.sin x - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_plus_sin_minus_two_range_l1811_181116


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1811_181102

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal -/
def symmetricYAxis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetric_points_sum (a b : ℝ) :
  symmetricYAxis (a, 3) (4, b) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1811_181102


namespace NUMINAMATH_CALUDE_time_sum_after_duration_l1811_181141

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time and returns the resulting time -/
def addDuration (initial : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

/-- Converts a Time to its 12-hour clock representation -/
def to12HourClock (t : Time) : Time :=
  sorry

/-- Calculates the sum of digits in a Time -/
def sumDigits (t : Time) : Nat :=
  sorry

theorem time_sum_after_duration :
  let initialTime : Time := ⟨15, 0, 0⟩  -- 3:00:00 PM
  let finalTime := to12HourClock (addDuration initialTime 317 58 33)
  sumDigits finalTime = 99 := by
  sorry

end NUMINAMATH_CALUDE_time_sum_after_duration_l1811_181141


namespace NUMINAMATH_CALUDE_blending_markers_count_l1811_181155

/-- Proof that the number of drawings made with blending markers is 7 -/
theorem blending_markers_count (total : ℕ) (colored_pencils : ℕ) (charcoal : ℕ) 
  (h1 : total = 25)
  (h2 : colored_pencils = 14)
  (h3 : charcoal = 4) :
  total - (colored_pencils + charcoal) = 7 := by
  sorry

end NUMINAMATH_CALUDE_blending_markers_count_l1811_181155


namespace NUMINAMATH_CALUDE_john_age_proof_l1811_181123

/-- John's current age -/
def john_age : ℕ := 18

/-- Proposition: John's current age satisfies the given condition -/
theorem john_age_proof : 
  (john_age - 5 : ℤ) = (john_age + 8 : ℤ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_john_age_proof_l1811_181123


namespace NUMINAMATH_CALUDE_no_solution_l1811_181104

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := 3*x - 4*y + z = 10
def equation2 (x y z : ℝ) : Prop := 6*x - 8*y + 2*z = 5
def equation3 (x y z : ℝ) : Prop := 2*x - y - z = 4

-- Theorem stating that the system has no solution
theorem no_solution : ¬∃ (x y z : ℝ), equation1 x y z ∧ equation2 x y z ∧ equation3 x y z :=
sorry

end NUMINAMATH_CALUDE_no_solution_l1811_181104


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l1811_181188

theorem triangle_angle_sum (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- Ensures positive angles
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  A = 25 →  -- Given angle A
  B = 55 →  -- Given angle B
  C = 100 :=  -- Conclusion: angle C
by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l1811_181188


namespace NUMINAMATH_CALUDE_diamonds_in_figure_l1811_181147

-- Define the sequence of figures
def F (n : ℕ) : ℕ :=
  2 * n^2 - 2 * n + 1

-- State the theorem
theorem diamonds_in_figure (n : ℕ) (h : n ≥ 1) : 
  F n = 2 * n^2 - 2 * n + 1 :=
by sorry

-- Verify the result for F_20
example : F 20 = 761 :=
by sorry

end NUMINAMATH_CALUDE_diamonds_in_figure_l1811_181147


namespace NUMINAMATH_CALUDE_line_slope_point_sum_l1811_181128

/-- Represents a line in the form y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

/-- Checks if a point (x, y) is on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.c

theorem line_slope_point_sum (l : Line) :
  l.m = -5 →
  l.contains 2 3 →
  l.m + l.c = 8 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_point_sum_l1811_181128


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_proof_l1811_181151

/-- The value of a chair in dollars -/
def chair_value : ℕ := 240

/-- The value of a table in dollars -/
def table_value : ℕ := 180

/-- A function that checks if a debt can be resolved using chairs and tables -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (c t : ℤ), debt = chair_value * c + table_value * t

/-- The smallest positive debt that can be resolved -/
def smallest_resolvable_debt : ℕ := 60

theorem smallest_resolvable_debt_proof :
  (∀ d : ℕ, d > 0 → d < smallest_resolvable_debt → ¬ is_resolvable d) ∧
  is_resolvable smallest_resolvable_debt :=
sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_proof_l1811_181151


namespace NUMINAMATH_CALUDE_multiplication_addition_difference_l1811_181135

theorem multiplication_addition_difference : 
  (2 : ℚ) / 3 * (3 : ℚ) / 2 - ((2 : ℚ) / 3 + (3 : ℚ) / 2) = -(7 : ℚ) / 6 :=
by sorry

end NUMINAMATH_CALUDE_multiplication_addition_difference_l1811_181135


namespace NUMINAMATH_CALUDE_square_inequality_l1811_181133

theorem square_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 3 / (x - y)) : x^2 > y^2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l1811_181133


namespace NUMINAMATH_CALUDE_three_people_eight_seats_l1811_181120

/-- The number of ways 3 people can sit in 8 seats with empty seats between them -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  let available_positions := total_seats - 2 * people + 1
  (available_positions.choose people) * (Nat.factorial people)

/-- Theorem stating that there are 24 ways for 3 people to sit in 8 seats with empty seats between them -/
theorem three_people_eight_seats : seating_arrangements 8 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_three_people_eight_seats_l1811_181120


namespace NUMINAMATH_CALUDE_green_hat_cost_l1811_181176

/-- Proves that the cost of each green hat is $1 given the conditions of the problem -/
theorem green_hat_cost (total_hats : ℕ) (blue_hat_cost : ℕ) (total_price : ℕ) (green_hats : ℕ) :
  total_hats = 85 →
  blue_hat_cost = 6 →
  total_price = 600 →
  green_hats = 90 →
  ∃ (green_hat_cost : ℕ), green_hat_cost = 1 ∧
    total_price = blue_hat_cost * (total_hats - green_hats) + green_hat_cost * green_hats :=
by
  sorry


end NUMINAMATH_CALUDE_green_hat_cost_l1811_181176


namespace NUMINAMATH_CALUDE_triangle_properties_l1811_181144

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def triangle_conditions (t : Triangle) : Prop :=
  acute_triangle t ∧
  t.a = 2 * t.b * Real.sin t.A ∧
  t.a = 3 * Real.sqrt 3 ∧
  t.c = 5

-- State the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = Real.pi/6 ∧ 
  t.b = Real.sqrt 7 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B) = (15 * Real.sqrt 3)/4 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1811_181144


namespace NUMINAMATH_CALUDE_inequality_proof_l1811_181152

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  1 ≤ (x / (1 + y*z) + y / (1 + z*x) + z / (1 + x*y)) ∧ 
  (x / (1 + y*z) + y / (1 + z*x) + z / (1 + x*y)) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1811_181152


namespace NUMINAMATH_CALUDE_parabola_distance_theorem_l1811_181148

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -2

-- Define the condition for Q being on the parabola
def Q_on_parabola (Q : ℝ × ℝ) : Prop := parabola Q.1 Q.2

-- Define the vector relation
def vector_relation (P Q : ℝ × ℝ) : Prop :=
  (Q.1 - focus.1, Q.2 - focus.2) = (-4 * (focus.1 - P.1), -4 * (focus.2 - P.2))

-- The theorem to prove
theorem parabola_distance_theorem (P Q : ℝ × ℝ) :
  directrix P.1 →
  Q_on_parabola Q →
  vector_relation P Q →
  Real.sqrt ((Q.1 - focus.1)^2 + (Q.2 - focus.2)^2) = 20 :=
by sorry

end NUMINAMATH_CALUDE_parabola_distance_theorem_l1811_181148


namespace NUMINAMATH_CALUDE_michaela_needs_20_oranges_l1811_181165

/-- The number of oranges Michaela needs to eat until she gets full. -/
def michaela_oranges : ℕ := 20

/-- The number of oranges Cassandra needs to eat until she gets full. -/
def cassandra_oranges : ℕ := 2 * michaela_oranges

/-- The total number of oranges picked. -/
def total_oranges : ℕ := 90

/-- The number of oranges remaining after both ate until full. -/
def remaining_oranges : ℕ := 30

/-- Proves that Michaela needs 20 oranges to get full given the conditions. -/
theorem michaela_needs_20_oranges : 
  michaela_oranges = 20 ∧ 
  cassandra_oranges = 2 * michaela_oranges ∧
  total_oranges = 90 ∧
  remaining_oranges = 30 ∧
  total_oranges = michaela_oranges + cassandra_oranges + remaining_oranges :=
by sorry

end NUMINAMATH_CALUDE_michaela_needs_20_oranges_l1811_181165


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1811_181113

theorem regular_polygon_sides (interior_angle : ℝ) : 
  interior_angle = 150 → (360 / (180 - interior_angle) : ℝ) = 12 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1811_181113


namespace NUMINAMATH_CALUDE_A_equals_6x_squared_A_plus_2B_A_plus_2B_at_negative_one_l1811_181192

-- Define polynomials A and B
def B (x : ℝ) : ℝ := 4 * x^2 - 5 * x - 7

axiom A_minus_2B (x : ℝ) : ∃ A : ℝ → ℝ, A x - 2 * (B x) = -2 * x^2 + 10 * x + 14

-- Theorem 1: A = 6x^2
theorem A_equals_6x_squared : ∃ A : ℝ → ℝ, ∀ x : ℝ, A x = 6 * x^2 := by sorry

-- Theorem 2: A + 2B = 14x^2 - 10x - 14
theorem A_plus_2B (x : ℝ) : ∃ A : ℝ → ℝ, A x + 2 * (B x) = 14 * x^2 - 10 * x - 14 := by sorry

-- Theorem 3: When x = -1, A + 2B = 10
theorem A_plus_2B_at_negative_one : ∃ A : ℝ → ℝ, A (-1) + 2 * (B (-1)) = 10 := by sorry

end NUMINAMATH_CALUDE_A_equals_6x_squared_A_plus_2B_A_plus_2B_at_negative_one_l1811_181192


namespace NUMINAMATH_CALUDE_cookie_boxes_l1811_181118

theorem cookie_boxes (n : Nat) (h : n = 392) : 
  (Finset.filter (fun p => 1 < p ∧ p < n ∧ n / p > 3) (Finset.range (n + 1))).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookie_boxes_l1811_181118


namespace NUMINAMATH_CALUDE_smallest_n_congruence_two_satisfies_congruence_two_is_smallest_smallest_positive_integer_congruence_l1811_181101

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 527 * n ≡ 1083 * n [ZMOD 30] → n ≥ 2 :=
by sorry

theorem two_satisfies_congruence : 527 * 2 ≡ 1083 * 2 [ZMOD 30] :=
by sorry

theorem two_is_smallest : ∀ m : ℕ, m > 0 ∧ 527 * m ≡ 1083 * m [ZMOD 30] → m ≥ 2 :=
by sorry

theorem smallest_positive_integer_congruence : 
  ∃! n : ℕ, n > 0 ∧ 527 * n ≡ 1083 * n [ZMOD 30] ∧ ∀ m : ℕ, (m > 0 ∧ 527 * m ≡ 1083 * m [ZMOD 30] → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_two_satisfies_congruence_two_is_smallest_smallest_positive_integer_congruence_l1811_181101


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1811_181190

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1811_181190


namespace NUMINAMATH_CALUDE_polynomial_monomial_degree_l1811_181178

/-- Given a sixth-degree polynomial and a monomial, prove the values of m and n -/
theorem polynomial_monomial_degree (m n : ℕ) : 
  (2 + (m + 1) = 6) ∧ (2*n + (5 - m) = 6) → m = 3 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_monomial_degree_l1811_181178


namespace NUMINAMATH_CALUDE_sqrt_sum_power_inequality_l1811_181145

theorem sqrt_sum_power_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (Real.sqrt x + Real.sqrt y)^8 ≥ 64 * x * y * (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_power_inequality_l1811_181145


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1811_181103

/-- Given a hyperbola with asymptote x + √3y = 0 and one focus at (4, 0),
    its standard equation is x²/12 - y²/4 = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧
    (x + Real.sqrt 3 * y = 0 → y = -(1 / Real.sqrt 3) * x) ∧  -- Asymptote condition
    c = 4 ∧                                                   -- Focus condition
    c^2 = a^2 + b^2 ∧                                         -- Hyperbola property
    b/a = Real.sqrt 3 / 3) →                                  -- Derived from asymptote
  x^2 / 12 - y^2 / 4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1811_181103


namespace NUMINAMATH_CALUDE_break_even_point_correct_l1811_181136

/-- The cost to mold each handle in dollars -/
def moldCost : ℝ := 0.60

/-- The fixed cost to run the molding machine per week in dollars -/
def fixedCost : ℝ := 7640

/-- The selling price per handle in dollars -/
def sellingPrice : ℝ := 4.60

/-- The number of handles needed to break even -/
def breakEvenPoint : ℕ := 1910

/-- Theorem stating that the calculated break-even point is correct -/
theorem break_even_point_correct :
  ↑breakEvenPoint * (sellingPrice - moldCost) = fixedCost :=
sorry

end NUMINAMATH_CALUDE_break_even_point_correct_l1811_181136


namespace NUMINAMATH_CALUDE_workers_per_block_l1811_181140

/-- Given a company with the following conditions:
  - The total amount for gifts is $6000
  - Each gift costs $2
  - There are 15 blocks in the company
  This theorem proves that there are 200 workers in each block. -/
theorem workers_per_block (total_amount : ℕ) (gift_worth : ℕ) (num_blocks : ℕ)
  (h1 : total_amount = 6000)
  (h2 : gift_worth = 2)
  (h3 : num_blocks = 15) :
  total_amount / gift_worth / num_blocks = 200 := by
  sorry

end NUMINAMATH_CALUDE_workers_per_block_l1811_181140


namespace NUMINAMATH_CALUDE_solutions_count_no_solutions_when_a_less_than_neg_one_one_solution_when_a_eq_neg_one_two_solutions_when_a_between_neg_one_and_zero_one_solution_when_a_greater_than_zero_l1811_181149

/-- The number of real solutions to the equation √(3a - 2x) + x = a depends on the value of a -/
theorem solutions_count (a : ℝ) :
  (∀ x, ¬ (Real.sqrt (3 * a - 2 * x) + x = a)) ∨
  (∃! x, Real.sqrt (3 * a - 2 * x) + x = a) ∨
  (∃ x y, x ≠ y ∧ Real.sqrt (3 * a - 2 * x) + x = a ∧ Real.sqrt (3 * a - 2 * y) + y = a) :=
by
  sorry

/-- For a < -1, there are no real solutions -/
theorem no_solutions_when_a_less_than_neg_one (a : ℝ) (h : a < -1) :
  ∀ x, ¬ (Real.sqrt (3 * a - 2 * x) + x = a) :=
by
  sorry

/-- For a = -1, there is exactly one real solution -/
theorem one_solution_when_a_eq_neg_one (a : ℝ) (h : a = -1) :
  ∃! x, Real.sqrt (3 * a - 2 * x) + x = a :=
by
  sorry

/-- For -1 < a ≤ 0, there are exactly two real solutions -/
theorem two_solutions_when_a_between_neg_one_and_zero (a : ℝ) (h1 : -1 < a) (h2 : a ≤ 0) :
  ∃ x y, x ≠ y ∧ Real.sqrt (3 * a - 2 * x) + x = a ∧ Real.sqrt (3 * a - 2 * y) + y = a :=
by
  sorry

/-- For a > 0, there is exactly one real solution -/
theorem one_solution_when_a_greater_than_zero (a : ℝ) (h : a > 0) :
  ∃! x, Real.sqrt (3 * a - 2 * x) + x = a :=
by
  sorry

end NUMINAMATH_CALUDE_solutions_count_no_solutions_when_a_less_than_neg_one_one_solution_when_a_eq_neg_one_two_solutions_when_a_between_neg_one_and_zero_one_solution_when_a_greater_than_zero_l1811_181149


namespace NUMINAMATH_CALUDE_expression_simplification_l1811_181197

theorem expression_simplification (x y : ℝ) (hx : x = 4) (hy : y = -2) :
  1 - (x - y) / (x + 2*y) / ((x^2 - y^2) / (x^2 + 4*x*y + 4*y^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1811_181197


namespace NUMINAMATH_CALUDE_reading_percentage_third_night_l1811_181130

/-- Theorem: Reading percentage on the third night
Given:
- A book with 500 pages
- 20% read on the first night
- 20% read on the second night
- 150 pages left after three nights of reading
Prove: The percentage read on the third night is 30%
-/
theorem reading_percentage_third_night
  (total_pages : ℕ)
  (first_night_percentage : ℚ)
  (second_night_percentage : ℚ)
  (pages_left : ℕ)
  (h1 : total_pages = 500)
  (h2 : first_night_percentage = 20 / 100)
  (h3 : second_night_percentage = 20 / 100)
  (h4 : pages_left = 150) :
  let pages_read_first_two_nights := (first_night_percentage + second_night_percentage) * total_pages
  let total_pages_read := total_pages - pages_left
  let pages_read_third_night := total_pages_read - pages_read_first_two_nights
  pages_read_third_night / total_pages = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_reading_percentage_third_night_l1811_181130


namespace NUMINAMATH_CALUDE_wall_height_calculation_l1811_181161

/-- Calculates the height of a wall given brick dimensions and number of bricks --/
theorem wall_height_calculation (brick_length brick_width brick_height : ℝ)
                                (wall_length wall_width : ℝ)
                                (num_bricks : ℝ) :
  brick_length = 25 →
  brick_width = 11 →
  brick_height = 6 →
  wall_length = 200 →
  wall_width = 2 →
  num_bricks = 72.72727272727273 →
  ∃ (wall_height : ℝ), abs (wall_height - 436.3636363636364) < 0.0001 :=
by
  sorry

#check wall_height_calculation

end NUMINAMATH_CALUDE_wall_height_calculation_l1811_181161


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1811_181124

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  c = Real.sqrt 2 →
  Real.cos C = 3/4 →
  2 * c * Real.sin A = b * Real.sin C →
  -- Conclusions
  b = 2 ∧
  Real.sin A = Real.sqrt 14 / 8 ∧
  Real.sin (2 * A + π/6) = (5 * Real.sqrt 21 + 9) / 32 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1811_181124


namespace NUMINAMATH_CALUDE_min_value_inequality_l1811_181157

theorem min_value_inequality (a : ℝ) : 
  (∀ x y : ℝ, |x| + |y| ≤ 1 → |2*x - 3*y + 3/2| + |y - 1| + |2*y - x - 3| ≤ a) ↔ 
  23/2 ≤ a :=
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1811_181157


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l1811_181143

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 > 0}
def B : Set ℝ := {x : ℝ | x / (x - 1) < 0}

-- State the theorem
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l1811_181143


namespace NUMINAMATH_CALUDE_white_washing_cost_per_square_foot_l1811_181191

/-- Calculates the cost per square foot for white washing a room --/
theorem white_washing_cost_per_square_foot
  (room_length room_width room_height : ℝ)
  (door_length door_width : ℝ)
  (window_length window_width : ℝ)
  (num_windows : ℕ)
  (total_cost : ℝ)
  (h_room_length : room_length = 25)
  (h_room_width : room_width = 15)
  (h_room_height : room_height = 12)
  (h_door_length : door_length = 6)
  (h_door_width : door_width = 3)
  (h_window_length : window_length = 4)
  (h_window_width : window_width = 3)
  (h_num_windows : num_windows = 3)
  (h_total_cost : total_cost = 8154) :
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_length * door_width
  let window_area := num_windows * (window_length * window_width)
  let net_area := wall_area - door_area - window_area
  let cost_per_square_foot := total_cost / net_area
  cost_per_square_foot = 9 :=
sorry

end NUMINAMATH_CALUDE_white_washing_cost_per_square_foot_l1811_181191


namespace NUMINAMATH_CALUDE_solve_for_y_l1811_181172

theorem solve_for_y (x y : ℝ) (h1 : x + 2*y = 12) (h2 : x = 6) : y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1811_181172


namespace NUMINAMATH_CALUDE_seating_arrangement_count_l1811_181131

-- Define the number of people
def num_people : ℕ := 10

-- Define the number of seats in each row
def seats_per_row : ℕ := 5

-- Define a function to calculate the number of valid seating arrangements
def valid_seating_arrangements (n : ℕ) (s : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

-- State the theorem
theorem seating_arrangement_count :
  valid_seating_arrangements num_people seats_per_row = 518400 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangement_count_l1811_181131


namespace NUMINAMATH_CALUDE_product_not_negative_l1811_181184

theorem product_not_negative (x y : ℝ) (n : ℕ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x^(2*n) - y^(2*n) > x) (h2 : y^(2*n) - x^(2*n) > y) :
  x * y > 0 :=
sorry

end NUMINAMATH_CALUDE_product_not_negative_l1811_181184


namespace NUMINAMATH_CALUDE_equation_solution_l1811_181169

theorem equation_solution : ∃ x : ℝ, (12 - 2 * x = 6) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1811_181169


namespace NUMINAMATH_CALUDE_square_side_length_l1811_181187

theorem square_side_length (area : ℚ) (side : ℚ) : 
  area = 9 / 16 → side * side = area → side = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1811_181187


namespace NUMINAMATH_CALUDE_painted_cells_theorem_l1811_181115

theorem painted_cells_theorem (k l : ℕ) : 
  k * l = 74 →
  ((2 * k + 1) * (2 * l + 1) - k * l = 373) ∨
  ((2 * k + 1) * (2 * l + 1) - k * l = 301) := by
sorry

end NUMINAMATH_CALUDE_painted_cells_theorem_l1811_181115


namespace NUMINAMATH_CALUDE_sqrt_function_theorem_linear_function_theorem_l1811_181179

-- Problem 1
theorem sqrt_function_theorem (f : ℝ → ℝ) :
  (∀ x ≥ 0, f (Real.sqrt x) = x - 1) →
  (∀ x ≥ 0, f x = x^2 - 1) :=
by sorry

-- Problem 2
theorem linear_function_theorem (f : ℝ → ℝ) :
  (∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b) →
  (∀ x, f (f x) = f x + 2) →
  (∀ x, f x = x + 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_function_theorem_linear_function_theorem_l1811_181179


namespace NUMINAMATH_CALUDE_current_short_trees_count_l1811_181122

/-- The number of short trees currently in the park -/
def current_short_trees : ℕ := 41

/-- The number of short trees to be planted today -/
def trees_to_plant : ℕ := 57

/-- The total number of short trees after planting -/
def total_short_trees : ℕ := 98

/-- Theorem stating that the number of short trees currently in the park is 41 -/
theorem current_short_trees_count :
  current_short_trees + trees_to_plant = total_short_trees :=
by sorry

end NUMINAMATH_CALUDE_current_short_trees_count_l1811_181122


namespace NUMINAMATH_CALUDE_quadball_play_time_l1811_181193

/-- Represents the quadball game setup -/
structure QuadballGame where
  total_duration : ℕ
  players_per_game : ℕ
  total_players : ℕ
  play_time_per_player : ℕ

/-- Theorem stating the play time for each player in the quadball game -/
theorem quadball_play_time (game : QuadballGame) 
  (h1 : game.total_duration = 120)
  (h2 : game.players_per_game = 4)
  (h3 : game.total_players = 8)
  (h4 : game.play_time_per_player * game.total_players = game.total_duration * game.players_per_game) :
  game.play_time_per_player = 60 := by
  sorry

#check quadball_play_time

end NUMINAMATH_CALUDE_quadball_play_time_l1811_181193


namespace NUMINAMATH_CALUDE_max_share_is_18200_l1811_181163

/-- Represents the profit share of a partner -/
structure PartnerShare where
  ratio : Nat
  bonus : Bool

/-- Calculates the maximum share given the total profit, bonus amount, and partner shares -/
def maxShare (totalProfit : ℚ) (bonusAmount : ℚ) (shares : List PartnerShare) : ℚ :=
  sorry

/-- The main theorem -/
theorem max_share_is_18200 :
  let shares := [
    ⟨4, false⟩,
    ⟨3, false⟩,
    ⟨2, true⟩,
    ⟨6, false⟩
  ]
  maxShare 45000 500 shares = 18200 := by sorry

end NUMINAMATH_CALUDE_max_share_is_18200_l1811_181163


namespace NUMINAMATH_CALUDE_f_sum_symmetric_max_a_bound_l1811_181137

noncomputable def f (x : ℝ) : ℝ := (3 * Real.exp x) / (1 + Real.exp x)

theorem f_sum_symmetric (x : ℝ) : f x + f (-x) = 3 := by sorry

theorem max_a_bound (a : ℝ) :
  (∀ x > 0, f (4 - a * x) + f (x^2) ≥ 3) ↔ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_max_a_bound_l1811_181137


namespace NUMINAMATH_CALUDE_parabola_points_range_l1811_181121

-- Define the parabola
def parabola (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + b

-- Define the theorem
theorem parabola_points_range (a b y₁ y₂ n : ℝ) :
  a > 0 →
  y₁ < y₂ →
  parabola a b (2 * n + 3) = y₁ →
  parabola a b (n - 1) = y₂ →
  (2 * n + 3 - 1) * (n - 1 - 1) < 0 →  -- Opposite sides of axis of symmetry
  -1 < n ∧ n < 0 := by
sorry

end NUMINAMATH_CALUDE_parabola_points_range_l1811_181121


namespace NUMINAMATH_CALUDE_sum_of_inverse_cubes_of_roots_l1811_181162

theorem sum_of_inverse_cubes_of_roots (r s : ℝ) : 
  (3 * r^2 + 5 * r + 2 = 0) → 
  (3 * s^2 + 5 * s + 2 = 0) → 
  (r ≠ s) →
  (1 / r^3 + 1 / s^3 = 25 / 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_inverse_cubes_of_roots_l1811_181162


namespace NUMINAMATH_CALUDE_sequence_property_l1811_181168

/-- Given a sequence where the nth term is of the form 32000+n + m/n = (2000+n) 3(m/n),
    prove that when n = 2016, (n³)/(n²) = 2016 -/
theorem sequence_property : 
  ∀ n : ℕ, n = 2016 → (n^3 : ℚ) / (n^2 : ℚ) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l1811_181168


namespace NUMINAMATH_CALUDE_function_transformation_l1811_181119

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_transformation (h : f 1 = -2) : f (-(-1)) + 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l1811_181119


namespace NUMINAMATH_CALUDE_watch_cost_price_l1811_181167

/-- Proves that the cost price of a watch is 280 Rs. given specific selling conditions -/
theorem watch_cost_price (selling_price : ℝ) : 
  (selling_price = 0.54 * 280) →  -- Sold at 46% loss
  (selling_price + 140 = 1.04 * 280) →  -- If sold for 140 more, 4% gain
  280 = 280 := by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1811_181167


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1811_181107

-- Define the universal set U
def U : Set ℤ := {x : ℤ | x^2 - 5*x - 6 < 0}

-- Define set A
def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 2}

-- Define set B
def B : Set ℤ := {2, 3, 5}

-- Theorem statement
theorem complement_A_intersect_B : (U \ A) ∩ B = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1811_181107


namespace NUMINAMATH_CALUDE_difference_A_B_l1811_181158

def A : ℕ → ℕ
  | 0 => 41
  | n + 1 => (2*n + 1) * (2*n + 2) + A n

def B : ℕ → ℕ
  | 0 => 1
  | n + 1 => (2*n) * (2*n + 1) + B n

theorem difference_A_B : A 20 - B 20 = 380 := by
  sorry

end NUMINAMATH_CALUDE_difference_A_B_l1811_181158


namespace NUMINAMATH_CALUDE_smallest_m_congruence_l1811_181108

theorem smallest_m_congruence : ∃ m : ℕ+, 
  (∀ k : ℕ+, k < m → ¬(790 * k.val ≡ 1430 * k.val [ZMOD 30])) ∧ 
  (790 * m.val ≡ 1430 * m.val [ZMOD 30]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_congruence_l1811_181108


namespace NUMINAMATH_CALUDE_sin_graph_translation_l1811_181174

open Real

theorem sin_graph_translation (a : ℝ) (h1 : 0 < a) (h2 : a < π) :
  (∀ x, sin (2 * (x - a) + π / 3) = sin (2 * x)) → a = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_sin_graph_translation_l1811_181174


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1811_181125

theorem sphere_volume_from_surface_area (S : ℝ) (h : S = 36 * Real.pi) :
  (4 / 3 : ℝ) * Real.pi * ((S / (4 * Real.pi)) ^ (3 / 2 : ℝ)) = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1811_181125


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l1811_181186

theorem arithmetic_mean_of_first_four_primes_reciprocals :
  let first_four_primes := [2, 3, 5, 7]
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / 4) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l1811_181186


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l1811_181181

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_diagonal_length 
  (q : Quadrilateral) 
  (hConvex : isConvex q) 
  (hAB : distance q.A q.B = 8)
  (hCD : distance q.C q.D = 18)
  (hAC : distance q.A q.C = 20)
  (E : Point)
  (hE : E = lineIntersection q.A q.C q.B q.D)
  (hAreas : triangleArea q.A E q.D = triangleArea q.B E q.C) :
  distance q.A E = 80 / 13 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l1811_181181


namespace NUMINAMATH_CALUDE_triangle_inscribed_circle_properties_l1811_181182

/-- Given a triangle ABC with semi-perimeter s, inscribed circle center O₁,
    inscribed circle radius r₁, and circumscribed circle radius R -/
theorem triangle_inscribed_circle_properties 
  (A B C O₁ : ℝ × ℝ) (r₁ R s a b c : ℝ) :
  let AO₁ := Real.sqrt ((A.1 - O₁.1)^2 + (A.2 - O₁.2)^2)
  let BO₁ := Real.sqrt ((B.1 - O₁.1)^2 + (B.2 - O₁.2)^2)
  let CO₁ := Real.sqrt ((C.1 - O₁.1)^2 + (C.2 - O₁.2)^2)
  -- Conditions
  s = (a + b + c) / 2 →
  -- Theorem statements
  AO₁^2 = (s / (s - a)) * b * c ∧
  AO₁ * BO₁ * CO₁ = 4 * R * r₁^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inscribed_circle_properties_l1811_181182


namespace NUMINAMATH_CALUDE_simplify_and_sum_l1811_181117

theorem simplify_and_sum : ∃ (a b : ℕ), 
  (a > 0) ∧ (b > 0) ∧ 
  ((2^10 * 5^2)^(1/4) : ℝ) = a * (b^(1/4) : ℝ) ∧ 
  a + b = 104 := by sorry

end NUMINAMATH_CALUDE_simplify_and_sum_l1811_181117


namespace NUMINAMATH_CALUDE_cookies_milk_ratio_l1811_181154

/-- Proof that 5 cookies require 20/3 pints of milk given the established ratio and conversion rates -/
theorem cookies_milk_ratio (cookies_base : ℕ) (milk_base : ℕ) (cookies_target : ℕ) :
  cookies_base = 18 →
  milk_base = 3 →
  cookies_target = 5 →
  (milk_base * 4 * 2 : ℚ) / cookies_base * cookies_target = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_milk_ratio_l1811_181154


namespace NUMINAMATH_CALUDE_equation_standard_form_and_coefficients_l1811_181110

theorem equation_standard_form_and_coefficients :
  ∀ x : ℝ, x * (x + 1) = 2 * x - 1 ↔ x^2 - x + 1 = 0 ∧
  1 = 1 ∧ -1 = -1 ∧ 1 = 1 := by sorry

end NUMINAMATH_CALUDE_equation_standard_form_and_coefficients_l1811_181110


namespace NUMINAMATH_CALUDE_refrigerator_loss_percentage_l1811_181153

/-- Represents the loss percentage on the refrigerator -/
def loss_percentage : ℝ := 4

/-- Represents the cost price of the refrigerator in Rupees -/
def refrigerator_cp : ℝ := 15000

/-- Represents the cost price of the mobile phone in Rupees -/
def mobile_cp : ℝ := 8000

/-- Represents the profit percentage on the mobile phone -/
def mobile_profit_percentage : ℝ := 9

/-- Represents the overall profit in Rupees -/
def overall_profit : ℝ := 120

/-- Theorem stating that given the conditions, the loss percentage on the refrigerator is 4% -/
theorem refrigerator_loss_percentage :
  let mobile_sp := mobile_cp * (1 + mobile_profit_percentage / 100)
  let total_cp := refrigerator_cp + mobile_cp
  let total_sp := total_cp + overall_profit
  let refrigerator_sp := total_sp - mobile_sp
  let loss := refrigerator_cp - refrigerator_sp
  loss_percentage = (loss / refrigerator_cp) * 100 := by
  sorry


end NUMINAMATH_CALUDE_refrigerator_loss_percentage_l1811_181153


namespace NUMINAMATH_CALUDE_space_division_by_five_spheres_l1811_181175

/-- Maximum number of regions into which a sphere can be divided by n circles -/
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => a (n + 2) + 2 * (n + 2)

/-- Maximum number of regions into which space can be divided by n spheres -/
def b : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => b (n + 2) + a (n + 2)

theorem space_division_by_five_spheres :
  b 5 = 30 := by sorry

end NUMINAMATH_CALUDE_space_division_by_five_spheres_l1811_181175


namespace NUMINAMATH_CALUDE_cos_ninety_degrees_l1811_181156

theorem cos_ninety_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_ninety_degrees_l1811_181156


namespace NUMINAMATH_CALUDE_circle_square_area_l1811_181199

/-- A circle described by the equation 2x^2 = -2y^2 + 8x - 8y + 28 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 = -2 * p.2^2 + 8 * p.1 - 8 * p.2 + 28}

/-- The square that circumscribes the circle with sides parallel to the axes -/
def CircumscribingSquare (c : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), (x, y) ∈ c ∧ 
    (p.1 = x ∨ p.1 = -x) ∧ (p.2 = y ∨ p.2 = -y)}

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem circle_square_area : area (CircumscribingSquare Circle) = 88 := by sorry

end NUMINAMATH_CALUDE_circle_square_area_l1811_181199


namespace NUMINAMATH_CALUDE_total_hours_worked_is_48_l1811_181139

/-- Calculates the total hours worked given basic pay, overtime rate, and total wage -/
def totalHoursWorked (basicPay : ℚ) (basicHours : ℚ) (overtimeRate : ℚ) (totalWage : ℚ) : ℚ :=
  let basicHourlyRate := basicPay / basicHours
  let overtimeHourlyRate := basicHourlyRate * (1 + overtimeRate)
  let overtimePay := totalWage - basicPay
  let overtimeHours := overtimePay / overtimeHourlyRate
  basicHours + overtimeHours

/-- Theorem stating that under given conditions, the total hours worked is 48 -/
theorem total_hours_worked_is_48 :
  totalHoursWorked 20 40 (1/4) 25 = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_worked_is_48_l1811_181139


namespace NUMINAMATH_CALUDE_unknown_cube_edge_length_l1811_181100

theorem unknown_cube_edge_length 
  (edge1 : ℝ) (edge2 : ℝ) (edge_unknown : ℝ) (edge_new : ℝ)
  (h1 : edge1 = 6)
  (h2 : edge2 = 8)
  (h3 : edge_new = 12)
  (h4 : edge1^3 + edge2^3 + edge_unknown^3 = edge_new^3) :
  edge_unknown = 10 := by sorry

end NUMINAMATH_CALUDE_unknown_cube_edge_length_l1811_181100


namespace NUMINAMATH_CALUDE_volunteer_schedule_lcm_l1811_181129

theorem volunteer_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_schedule_lcm_l1811_181129


namespace NUMINAMATH_CALUDE_median_salary_is_45000_l1811_181160

structure Position :=
  (title : String)
  (count : ℕ)
  (salary : ℕ)

def company_data : List Position := [
  ⟨"CEO", 1, 150000⟩,
  ⟨"Senior Manager", 4, 95000⟩,
  ⟨"Manager", 15, 70000⟩,
  ⟨"Assistant Manager", 20, 45000⟩,
  ⟨"Clerk", 40, 18000⟩
]

def total_employees : ℕ := (company_data.map Position.count).sum

def median_salary (data : List Position) : ℕ := 
  if total_employees % 2 = 0 
  then 45000  -- As both (total_employees / 2) and (total_employees / 2 + 1) fall under Assistant Manager
  else 45000  -- As (total_employees / 2 + 1) falls under Assistant Manager

theorem median_salary_is_45000 : 
  median_salary company_data = 45000 := by sorry

end NUMINAMATH_CALUDE_median_salary_is_45000_l1811_181160


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l1811_181114

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of nitrogen atoms in the compound -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in the compound -/
def num_O : ℕ := 5

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := (num_N : ℝ) * atomic_weight_N + (num_O : ℝ) * atomic_weight_O

theorem molecular_weight_calculation :
  molecular_weight = 108.02 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l1811_181114


namespace NUMINAMATH_CALUDE_circle_division_theorem_l1811_181185

/-- A type representing a straight cut through a circle -/
structure Cut where
  -- We don't need to define the internal structure of a cut for this statement

/-- A type representing a circle -/
structure Circle where
  -- We don't need to define the internal structure of a circle for this statement

/-- A function that counts the number of regions created by cuts in a circle -/
def count_regions (circle : Circle) (cuts : List Cut) : ℕ := sorry

/-- Theorem stating that a circle can be divided into 4, 5, 6, and 7 parts using three straight cuts -/
theorem circle_division_theorem (circle : Circle) :
  ∃ (cuts₁ cuts₂ cuts₃ cuts₄ : List Cut),
    (cuts₁.length = 3 ∧ count_regions circle cuts₁ = 4) ∧
    (cuts₂.length = 3 ∧ count_regions circle cuts₂ = 5) ∧
    (cuts₃.length = 3 ∧ count_regions circle cuts₃ = 6) ∧
    (cuts₄.length = 3 ∧ count_regions circle cuts₄ = 7) :=
  sorry

end NUMINAMATH_CALUDE_circle_division_theorem_l1811_181185


namespace NUMINAMATH_CALUDE_second_largest_is_eleven_l1811_181132

def numbers : Finset ℕ := {10, 11, 12}

theorem second_largest_is_eleven :
  ∃ (a b : ℕ), a ∈ numbers ∧ b ∈ numbers ∧ a ≠ b ∧
  (∀ x ∈ numbers, x ≤ a) ∧
  (∃ y ∈ numbers, y > 11) ∧
  (∀ z ∈ numbers, z > 11 → z ≥ a) :=
sorry

end NUMINAMATH_CALUDE_second_largest_is_eleven_l1811_181132


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l1811_181196

theorem add_preserves_inequality (x y : ℝ) (h : x < y) : x + 6 < y + 6 := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l1811_181196


namespace NUMINAMATH_CALUDE_orange_apple_difference_l1811_181106

def apples : ℕ := 14
def dozen : ℕ := 12
def oranges : ℕ := 2 * dozen

theorem orange_apple_difference : oranges - apples = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_apple_difference_l1811_181106


namespace NUMINAMATH_CALUDE_sibling_ages_sum_l1811_181170

theorem sibling_ages_sum (a b c : ℕ+) 
  (h_order : c < b ∧ b < a) 
  (h_product : a * b * c = 72) : 
  a + b + c = 13 := by
  sorry

end NUMINAMATH_CALUDE_sibling_ages_sum_l1811_181170


namespace NUMINAMATH_CALUDE_distance_between_tangent_circles_l1811_181142

/-- The distance between centers of two internally tangent circles -/
def distance_between_centers (r₁ r₂ : ℝ) : ℝ := |r₂ - r₁|

/-- Theorem: The distance between centers of two internally tangent circles
    with radii 3 and 4 is 1 -/
theorem distance_between_tangent_circles :
  let r₁ : ℝ := 3
  let r₂ : ℝ := 4
  distance_between_centers r₁ r₂ = 1 := by sorry

end NUMINAMATH_CALUDE_distance_between_tangent_circles_l1811_181142


namespace NUMINAMATH_CALUDE_cross_figure_perimeter_l1811_181171

/-- A cross-shaped figure formed by five identical squares -/
structure CrossFigure where
  /-- The side length of each square in the figure -/
  side_length : ℝ
  /-- The total area of the figure is 125 cm² -/
  total_area_eq : 5 * side_length^2 = 125

/-- The perimeter of a cross-shaped figure -/
def perimeter (f : CrossFigure) : ℝ :=
  16 * f.side_length

/-- Theorem: The perimeter of the cross-shaped figure is 80 cm -/
theorem cross_figure_perimeter (f : CrossFigure) : perimeter f = 80 := by
  sorry

end NUMINAMATH_CALUDE_cross_figure_perimeter_l1811_181171


namespace NUMINAMATH_CALUDE_no_counterexamples_l1811_181166

def sum_of_digits (n : ℕ) : ℕ := sorry

def has_no_zero_digit (n : ℕ) : Prop := sorry

theorem no_counterexamples :
  ¬ ∃ N : ℕ, 
    (sum_of_digits N = 5) ∧ 
    (has_no_zero_digit N) ∧ 
    (Nat.Prime N) ∧ 
    (N % 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_counterexamples_l1811_181166


namespace NUMINAMATH_CALUDE_gunther_typing_words_l1811_181127

-- Define the typing speeds and durations
def first_phase_speed : ℕ := 160
def first_phase_duration : ℕ := 2 * 60
def second_phase_speed : ℕ := 200
def second_phase_duration : ℕ := 3 * 60
def third_phase_speed : ℕ := 140
def third_phase_duration : ℕ := 4 * 60

-- Define the interval duration (in minutes)
def interval_duration : ℕ := 3

-- Function to calculate words typed in a phase
def words_in_phase (speed : ℕ) (duration : ℕ) : ℕ :=
  (duration / interval_duration) * speed

-- Theorem statement
theorem gunther_typing_words :
  words_in_phase first_phase_speed first_phase_duration +
  words_in_phase second_phase_speed second_phase_duration +
  words_in_phase third_phase_speed third_phase_duration = 29600 := by
  sorry

end NUMINAMATH_CALUDE_gunther_typing_words_l1811_181127
