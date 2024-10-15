import Mathlib

namespace NUMINAMATH_CALUDE_surface_a_properties_surface_b_properties_surface_c_properties_l326_32675

-- Part (a)
def surface_a1 (x y z : ℝ) : Prop := 2 * y = x^2 + z^2
def surface_a2 (x y z : ℝ) : Prop := x^2 + z^2 = 1

theorem surface_a_properties (x y z : ℝ) :
  surface_a1 x y z ∧ surface_a2 x y z → y ≥ 0 :=
sorry

-- Part (b)
def surface_b1 (x y z : ℝ) : Prop := z = 0
def surface_b2 (x y z : ℝ) : Prop := y + z = 2
def surface_b3 (x y z : ℝ) : Prop := y = x^2

theorem surface_b_properties (x y z : ℝ) :
  surface_b1 x y z ∧ surface_b2 x y z ∧ surface_b3 x y z → 
  y ≤ 2 ∧ y ≥ 0 ∧ z ≤ 2 ∧ z ≥ 0 :=
sorry

-- Part (c)
def surface_c1 (x y z : ℝ) : Prop := z = 6 - x^2 - y^2
def surface_c2 (x y z : ℝ) : Prop := x^2 + y^2 - z^2 = 0

theorem surface_c_properties (x y z : ℝ) :
  surface_c1 x y z ∧ surface_c2 x y z → 
  z ≤ 3 ∧ z ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_surface_a_properties_surface_b_properties_surface_c_properties_l326_32675


namespace NUMINAMATH_CALUDE_smallest_perimeter_l326_32637

/-- Triangle PQR with intersection point J of angle bisectors of ∠Q and ∠R -/
structure TrianglePQR where
  PQ : ℕ+
  QR : ℕ+
  QJ : ℕ+
  isIsosceles : PQ = PQ
  angleIntersection : QJ = 10

/-- The perimeter of triangle PQR -/
def perimeter (t : TrianglePQR) : ℕ := 2 * t.PQ + t.QR

/-- The smallest possible perimeter of triangle PQR satisfying the given conditions -/
theorem smallest_perimeter :
  ∀ t : TrianglePQR, perimeter t ≥ 40 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l326_32637


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l326_32622

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - 3*a = 0 ∧ x = -2) → 
  (∃ y : ℝ, y^2 - a*y - 3*a = 0 ∧ y = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l326_32622


namespace NUMINAMATH_CALUDE_expression_bounds_l326_32651

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) : 
  4 + 2 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
                        Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ∧
  Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
  Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l326_32651


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l326_32676

theorem consecutive_integers_problem (n : ℕ) (x : ℤ) : 
  n > 0 → 
  x + n - 1 = 23 → 
  (n : ℝ) * 20 = (n / 2 : ℝ) * (2 * x + n - 1) → 
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l326_32676


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l326_32620

theorem quadratic_inequality_solution_set (a : ℝ) (h : a > 1) :
  let solution_set := {x : ℝ | (a - 1) * x^2 - a * x + 1 > 0}
  (a = 2 → solution_set = {x : ℝ | x ≠ 1}) ∧
  (1 < a ∧ a < 2 → solution_set = {x : ℝ | x < 1 ∨ x > 1 / (a - 1)}) ∧
  (a > 2 → solution_set = {x : ℝ | x < 1 / (a - 1) ∨ x > 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l326_32620


namespace NUMINAMATH_CALUDE_divide_seven_students_three_groups_l326_32668

/-- The number of ways to divide students into groups and send them to different places -/
def divideAndSend (n : ℕ) (k : ℕ) (ratio : List ℕ) : ℕ :=
  sorry

/-- The theorem stating the correct number of ways for the given problem -/
theorem divide_seven_students_three_groups : divideAndSend 7 3 [3, 2, 2] = 630 := by
  sorry

end NUMINAMATH_CALUDE_divide_seven_students_three_groups_l326_32668


namespace NUMINAMATH_CALUDE_cos_450_degrees_eq_zero_l326_32635

theorem cos_450_degrees_eq_zero : Real.cos (450 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_450_degrees_eq_zero_l326_32635


namespace NUMINAMATH_CALUDE_inequality_solution_l326_32634

theorem inequality_solution (x : ℝ) :
  x ≠ 1 ∧ x ≠ 2 →
  (x^3 - x^2 - 6*x) / (x^2 - 3*x + 2) > 0 ↔ (-2 < x ∧ x < 0) ∨ (1 < x ∧ x < 2) ∨ (3 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l326_32634


namespace NUMINAMATH_CALUDE_line_not_in_first_quadrant_l326_32650

/-- Given that mx + 3 = 4 has the solution x = 1, prove that y = (m-2)x - 3 does not pass through the first quadrant -/
theorem line_not_in_first_quadrant (m : ℝ) (h : m * 1 + 3 = 4) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y ≠ (m - 2) * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_first_quadrant_l326_32650


namespace NUMINAMATH_CALUDE_volleyball_team_math_count_l326_32680

theorem volleyball_team_math_count (total_players : ℕ) (physics_players : ℕ) (both_players : ℕ) 
  (h1 : total_players = 25)
  (h2 : physics_players = 10)
  (h3 : both_players = 6)
  (h4 : physics_players ≥ both_players)
  (h5 : ∀ player, player ∈ Set.range (Fin.val : Fin total_players → ℕ) → 
    (player ∈ Set.range (Fin.val : Fin physics_players → ℕ) ∨ 
     player ∈ Set.range (Fin.val : Fin (total_players - physics_players + both_players) → ℕ))) :
  total_players - physics_players + both_players = 21 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_math_count_l326_32680


namespace NUMINAMATH_CALUDE_circle_equation_l326_32619

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 2)

-- Define the radius of the circle
def radius : ℝ := 2

-- State the theorem
theorem circle_equation (x y : ℝ) :
  ((x - center.1)^2 + (y - center.2)^2 = radius^2) ↔
  ((x + 1)^2 + (y - 2)^2 = 4) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l326_32619


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l326_32624

theorem polynomial_sum_of_coefficients 
  (f : ℂ → ℂ) 
  (a b c d : ℝ) :
  (∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d) →
  f (2*I) = 0 →
  f (2 + I) = 0 →
  a + b + c + d = 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l326_32624


namespace NUMINAMATH_CALUDE_calculation_proof_l326_32663

theorem calculation_proof : 
  let tan30 := Real.sqrt 3 / 3
  let π := 3.14
  (1/3)⁻¹ - Real.sqrt 27 + 3 * tan30 + (π - 3.14)^0 = 4 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l326_32663


namespace NUMINAMATH_CALUDE_third_offense_percentage_increase_l326_32698

/-- Calculates the percentage increase for a third offense in a burglary case -/
theorem third_offense_percentage_increase
  (base_rate : ℚ)  -- Base sentence rate in years per $5000
  (stolen_value : ℚ)  -- Total value of stolen goods in dollars
  (additional_penalty : ℚ)  -- Additional penalty in years
  (total_sentence : ℚ)  -- Total sentence in years
  (h1 : base_rate = 1)  -- Base rate is 1 year per $5000
  (h2 : stolen_value = 40000)  -- $40,000 worth of goods stolen
  (h3 : additional_penalty = 2)  -- 2 years additional penalty
  (h4 : total_sentence = 12)  -- Total sentence is 12 years
  : (total_sentence - additional_penalty - (stolen_value / 5000 * base_rate)) / (stolen_value / 5000 * base_rate) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_third_offense_percentage_increase_l326_32698


namespace NUMINAMATH_CALUDE_train_length_l326_32685

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 18 → ∃ (length_m : ℝ), 
    (length_m ≥ 300.05 ∧ length_m ≤ 300.07) ∧ 
    length_m = speed_kmh * (1000 / 3600) * time_s :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l326_32685


namespace NUMINAMATH_CALUDE_system_solution_l326_32693

theorem system_solution : 
  ∃! (x y : ℝ), x + y = 5 ∧ 2 * x + 5 * y = 28 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_l326_32693


namespace NUMINAMATH_CALUDE_change_per_bill_l326_32647

/-- Proves that the value of each bill given as change is $5 -/
theorem change_per_bill (num_games : ℕ) (cost_per_game : ℕ) (payment : ℕ) (num_bills : ℕ) :
  num_games = 6 →
  cost_per_game = 15 →
  payment = 100 →
  num_bills = 2 →
  (payment - num_games * cost_per_game) / num_bills = 5 := by
  sorry

end NUMINAMATH_CALUDE_change_per_bill_l326_32647


namespace NUMINAMATH_CALUDE_composition_zero_iff_rank_sum_eq_dim_l326_32656

variable {V : Type*} [AddCommGroup V] [Module ℝ V] [FiniteDimensional ℝ V]
variable (T U : V →ₗ[ℝ] V)

theorem composition_zero_iff_rank_sum_eq_dim (h : Function.Bijective (T + U)) :
  (T.comp U = 0 ∧ U.comp T = 0) ↔ LinearMap.rank T + LinearMap.rank U = FiniteDimensional.finrank ℝ V :=
sorry

end NUMINAMATH_CALUDE_composition_zero_iff_rank_sum_eq_dim_l326_32656


namespace NUMINAMATH_CALUDE_fourth_column_unique_l326_32636

/-- Represents a 9x9 Sudoku grid -/
def SudokuGrid := Fin 9 → Fin 9 → Fin 9

/-- Checks if a number is valid in a given position -/
def isValid (grid : SudokuGrid) (row col num : Fin 9) : Prop :=
  (∀ i : Fin 9, grid i col ≠ num) ∧
  (∀ j : Fin 9, grid row j ≠ num) ∧
  (∀ i j : Fin 3, grid (3 * (row / 3) + i) (3 * (col / 3) + j) ≠ num)

/-- Checks if the entire grid is valid -/
def isValidGrid (grid : SudokuGrid) : Prop :=
  ∀ row col : Fin 9, isValid grid row col (grid row col)

/-- Represents the pre-filled numbers in the 4th column -/
def fourthColumnPrefilled : Fin 9 → Option (Fin 9)
  | 0 => some 3
  | 1 => some 2
  | 3 => some 4
  | 7 => some 5
  | 8 => some 1
  | _ => none

/-- The theorem to be proved -/
theorem fourth_column_unique (grid : SudokuGrid) :
  isValidGrid grid →
  (∀ row : Fin 9, (fourthColumnPrefilled row).map (grid row 3) = fourthColumnPrefilled row) →
  (∀ row : Fin 9, grid row 3 = match row with
    | 0 => 3 | 1 => 2 | 2 => 7 | 3 => 4 | 4 => 6 | 5 => 8 | 6 => 9 | 7 => 5 | 8 => 1) :=
by sorry

end NUMINAMATH_CALUDE_fourth_column_unique_l326_32636


namespace NUMINAMATH_CALUDE_cube_inequality_l326_32661

theorem cube_inequality (x y a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l326_32661


namespace NUMINAMATH_CALUDE_dance_troupe_average_age_l326_32610

theorem dance_troupe_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (h1 : num_females = 12) 
  (h2 : num_males = 18) 
  (h3 : avg_age_females = 25) 
  (h4 : avg_age_males = 30) : 
  (num_females * avg_age_females + num_males * avg_age_males) / (num_females + num_males) = 28 := by
  sorry

end NUMINAMATH_CALUDE_dance_troupe_average_age_l326_32610


namespace NUMINAMATH_CALUDE_twelve_chairs_adjacent_subsets_l326_32696

/-- The number of chairs arranged in a circle. -/
def n : ℕ := 12

/-- A function that calculates the number of subsets containing at least three adjacent chairs
    for a given number of chairs arranged in a circle. -/
def subsets_with_adjacent_chairs (num_chairs : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs arranged in a circle, 
    the number of subsets containing at least three adjacent chairs is 2066. -/
theorem twelve_chairs_adjacent_subsets : subsets_with_adjacent_chairs n = 2066 := by sorry

end NUMINAMATH_CALUDE_twelve_chairs_adjacent_subsets_l326_32696


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l326_32611

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l326_32611


namespace NUMINAMATH_CALUDE_max_segment_length_through_centroid_l326_32666

/-- Given a triangle ABC with vertex A at (0,0), B at (b, 0), and C at (c_x, c_y),
    the maximum length of a line segment starting from A and passing through the centroid
    is equal to the distance between A and the centroid. -/
theorem max_segment_length_through_centroid (b c_x c_y : ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (b, 0)
  let C : ℝ × ℝ := (c_x, c_y)
  let centroid : ℝ × ℝ := ((b + c_x) / 3, c_y / 3)
  let max_length := Real.sqrt (((b + c_x) / 3)^2 + (c_y / 3)^2)
  ∃ (segment : ℝ × ℝ → ℝ × ℝ),
    (segment 0 = A) ∧
    (∃ t, segment t = centroid) ∧
    (∀ t, ‖segment t - A‖ ≤ max_length) ∧
    (∃ t, ‖segment t - A‖ = max_length) :=
by sorry


end NUMINAMATH_CALUDE_max_segment_length_through_centroid_l326_32666


namespace NUMINAMATH_CALUDE_fish_catch_total_l326_32678

def fish_problem (bass : ℕ) (trout : ℕ) (blue_gill : ℕ) : Prop :=
  (bass = 32) ∧
  (trout = bass / 4) ∧
  (blue_gill = 2 * bass) ∧
  (bass + trout + blue_gill = 104)

theorem fish_catch_total :
  ∀ (bass trout blue_gill : ℕ), fish_problem bass trout blue_gill :=
by
  sorry

end NUMINAMATH_CALUDE_fish_catch_total_l326_32678


namespace NUMINAMATH_CALUDE_three_digit_number_times_seven_l326_32660

theorem three_digit_number_times_seven (n : ℕ) : 
  (100 ≤ n ∧ n < 1000) ∧ (∃ k : ℕ, 7 * n = 1000 * k + 638) ↔ n = 234 :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_times_seven_l326_32660


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l326_32621

/-- The quadratic equation (2kx^2 + 7kx + 2) = 0 has equal roots when k = 16/49 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + 7 * k * x + 2 = 0) → 
  (∃! r : ℝ, 2 * k * r^2 + 7 * k * r + 2 = 0) → 
  k = 16/49 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l326_32621


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reversal_l326_32641

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def reverse_digits (n : ℕ) : ℕ :=
  let units := n % 10
  let tens := n / 10
  units * 10 + tens

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def tens_digit_is_two (n : ℕ) : Prop := (n / 10) % 10 = 2

theorem smallest_two_digit_prime_with_composite_reversal :
  ∃ n : ℕ,
    is_two_digit n ∧
    tens_digit_is_two n ∧
    is_prime n ∧
    is_composite (reverse_digits n) ∧
    (∀ m : ℕ, is_two_digit m → tens_digit_is_two m → is_prime m → 
      is_composite (reverse_digits m) → n ≤ m) ∧
    n = 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reversal_l326_32641


namespace NUMINAMATH_CALUDE_whitney_total_spent_l326_32653

def whale_books : ℕ := 9
def fish_books : ℕ := 7
def magazines : ℕ := 3
def book_cost : ℕ := 11
def magazine_cost : ℕ := 1

theorem whitney_total_spent : 
  (whale_books + fish_books) * book_cost + magazines * magazine_cost = 179 := by
  sorry

end NUMINAMATH_CALUDE_whitney_total_spent_l326_32653


namespace NUMINAMATH_CALUDE_collinear_points_problem_l326_32613

/-- Given three collinear points A, B, C in a plane with position vectors
    OA = (-2, m), OB = (n, 1), OC = (5, -1), and OA perpendicular to OB,
    prove that m = 6 and n = 3. -/
theorem collinear_points_problem (m n : ℝ) : 
  let OA : ℝ × ℝ := (-2, m)
  let OB : ℝ × ℝ := (n, 1)
  let OC : ℝ × ℝ := (5, -1)
  let AC : ℝ × ℝ := (OC.1 - OA.1, OC.2 - OA.2)
  let BC : ℝ × ℝ := (OC.1 - OB.1, OC.2 - OB.2)
  (∃ (k : ℝ), AC = k • BC) →  -- collinearity condition
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) →  -- perpendicularity condition
  m = 6 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_problem_l326_32613


namespace NUMINAMATH_CALUDE_power_of_power_l326_32648

theorem power_of_power (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l326_32648


namespace NUMINAMATH_CALUDE_unshaded_area_of_intersecting_rectangles_l326_32606

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the intersection of two rectangles -/
structure Intersection where
  width : ℝ
  height : ℝ

theorem unshaded_area_of_intersecting_rectangles
  (r1 : Rectangle)
  (r2 : Rectangle)
  (i : Intersection)
  (h1 : r1.width = 4 ∧ r1.height = 12)
  (h2 : r2.width = 5 ∧ r2.height = 10)
  (h3 : i.width = 4 ∧ i.height = 5) :
  area r1 + area r2 - (area r1 + area r2 - i.width * i.height) = 20 :=
sorry

end NUMINAMATH_CALUDE_unshaded_area_of_intersecting_rectangles_l326_32606


namespace NUMINAMATH_CALUDE_division_problem_l326_32669

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 686)
  (h2 : divisor = 36)
  (h3 : remainder = 2)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 19 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l326_32669


namespace NUMINAMATH_CALUDE_tangency_points_form_circular_arc_l326_32602

structure Segment where
  A : Point
  B : Point

structure Circle where
  center : Point
  radius : ℝ

def TangentCirclePair (s : Segment) (c1 c2 : Circle) : Prop :=
  -- Definition of tangent circle pair inscribed in segment
  sorry

def TangencyPoint (s : Segment) (c1 c2 : Circle) : Point :=
  -- Definition of tangency point between two circles
  sorry

def CircularArc (A B : Point) : Set Point :=
  -- Definition of a circular arc with endpoints A and B
  sorry

def AngleBisector (s : Segment) (arc : Set Point) : Set Point :=
  -- Definition of angle bisector between chord AB and segment arc
  sorry

theorem tangency_points_form_circular_arc (s : Segment) :
  ∃ (arc : Set Point), 
    (arc = CircularArc s.A s.B) ∧ 
    (arc = AngleBisector s arc) ∧
    (∀ (c1 c2 : Circle), TangentCirclePair s c1 c2 → 
      TangencyPoint s c1 c2 ∈ arc) := by
  sorry

end NUMINAMATH_CALUDE_tangency_points_form_circular_arc_l326_32602


namespace NUMINAMATH_CALUDE_intersection_k_range_l326_32692

-- Define the line and hyperbola equations
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 2
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 6

-- Define the condition for intersection points
def intersects_right_branch (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
  hyperbola x₁ (line k x₁) ∧ hyperbola x₂ (line k x₂)

-- Theorem statement
theorem intersection_k_range :
  ∀ k : ℝ, intersects_right_branch k ↔ -Real.sqrt 15 / 3 < k ∧ k < -1 :=
sorry

end NUMINAMATH_CALUDE_intersection_k_range_l326_32692


namespace NUMINAMATH_CALUDE_square_triangulation_l326_32605

/-- A planar graph representing the configuration of points and lines in a square -/
structure SquareGraph where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces (regions)

/-- The number of triangles formed in a square with 20 internal points -/
def num_triangles (g : SquareGraph) : ℕ := g.F - 1

theorem square_triangulation :
  ∀ g : SquareGraph,
  g.V = 24 →  -- 20 internal points + 4 vertices of the square
  2 * g.E = 3 * g.F + 1 →  -- Relation between edges and faces
  g.V - g.E + g.F = 2 →  -- Euler's formula for planar graphs
  num_triangles g = 42 := by
sorry

end NUMINAMATH_CALUDE_square_triangulation_l326_32605


namespace NUMINAMATH_CALUDE_C₂_is_symmetric_to_C₁_l326_32664

/-- Circle C₁ with equation (x+1)²+(y-1)²=1 -/
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

/-- The line of symmetry x-y-1=0 -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- The symmetric point of (x, y) with respect to the line x-y-1=0 -/
def symmetric_point (x y : ℝ) : ℝ × ℝ := (y + 1, x - 1)

/-- Circle C₂, symmetric to C₁ with respect to the line x-y-1=0 -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

/-- Theorem stating that C₂ is indeed symmetric to C₁ with respect to the given line -/
theorem C₂_is_symmetric_to_C₁ :
  ∀ x y : ℝ, C₂ x y ↔ C₁ (symmetric_point x y).1 (symmetric_point x y).2 :=
sorry

end NUMINAMATH_CALUDE_C₂_is_symmetric_to_C₁_l326_32664


namespace NUMINAMATH_CALUDE_cuboid_area_and_volume_l326_32673

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℝ
  breadth : ℝ
  height : ℝ

/-- Calculate the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.breadth + c.length * c.height + c.breadth * c.height)

/-- Calculate the volume of a cuboid -/
def volume (c : Cuboid) : ℝ :=
  c.length * c.breadth * c.height

/-- Theorem stating the surface area and volume of a specific cuboid -/
theorem cuboid_area_and_volume :
  let c : Cuboid := ⟨10, 8, 6⟩
  surfaceArea c = 376 ∧ volume c = 480 := by
  sorry

#check cuboid_area_and_volume

end NUMINAMATH_CALUDE_cuboid_area_and_volume_l326_32673


namespace NUMINAMATH_CALUDE_circle_center_l326_32652

/-- Given a circle with equation x^2 + y^2 - 2x + 4y - 4 = 0, 
    its center coordinates are (1, -2) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 4*y - 4 = 0) → 
  ∃ (h : ℝ), (x - 1)^2 + (y + 2)^2 = h^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l326_32652


namespace NUMINAMATH_CALUDE_scientific_notation_of_14900_l326_32654

theorem scientific_notation_of_14900 : 
  14900 = 1.49 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_14900_l326_32654


namespace NUMINAMATH_CALUDE_fraction_of_puppies_sold_l326_32646

/-- Proves that the fraction of puppies sold is 3/8 given the problem conditions --/
theorem fraction_of_puppies_sold (total_puppies : ℕ) (price_per_puppy : ℕ) (total_received : ℕ) :
  total_puppies = 20 →
  price_per_puppy = 200 →
  total_received = 3000 →
  (total_received / price_per_puppy : ℚ) / total_puppies = 3 / 8 := by
  sorry

#check fraction_of_puppies_sold

end NUMINAMATH_CALUDE_fraction_of_puppies_sold_l326_32646


namespace NUMINAMATH_CALUDE_min_value_expression_l326_32609

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 + c^2) / (a*b + 2*b*c) ≥ 2 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l326_32609


namespace NUMINAMATH_CALUDE_line_through_ellipse_vertex_l326_32689

/-- The value of 'a' when a line passes through the right vertex of an ellipse --/
theorem line_through_ellipse_vertex (t θ : ℝ) (a : ℝ) : 
  (∀ t, ∃ x y, x = t ∧ y = t - a) →  -- Line equation
  (∀ θ, ∃ x y, x = 3 * Real.cos θ ∧ y = 2 * Real.sin θ) →  -- Ellipse equation
  (∃ t, t = 3 ∧ t - a = 0) →  -- Line passes through right vertex (3, 0)
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_line_through_ellipse_vertex_l326_32689


namespace NUMINAMATH_CALUDE_at_most_two_out_of_three_l326_32631

-- Define the probability of a single event
def p : ℚ := 3 / 5

-- Define the number of events
def n : ℕ := 3

-- Define the maximum number of events we want to occur
def k : ℕ := 2

-- Theorem statement
theorem at_most_two_out_of_three (p : ℚ) (n : ℕ) (k : ℕ) 
  (h1 : p = 3 / 5) 
  (h2 : n = 3) 
  (h3 : k = 2) : 
  1 - p^n = 98 / 125 := by
  sorry

#check at_most_two_out_of_three p n k

end NUMINAMATH_CALUDE_at_most_two_out_of_three_l326_32631


namespace NUMINAMATH_CALUDE_wood_length_after_sawing_l326_32691

theorem wood_length_after_sawing (original_length sawed_off_length : ℝ) 
  (h1 : original_length = 0.41)
  (h2 : sawed_off_length = 0.33) :
  original_length - sawed_off_length = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_after_sawing_l326_32691


namespace NUMINAMATH_CALUDE_ab_value_l326_32671

theorem ab_value (a b : ℝ) (h : 48 * (a * b) = (a * b) * 65) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l326_32671


namespace NUMINAMATH_CALUDE_valid_seq_equals_fib_prob_no_consecutive_ones_l326_32623

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Sequence of valid arrangements -/
def validSeq : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validSeq (n + 1) + validSeq n

theorem valid_seq_equals_fib (n : ℕ) : validSeq n = fib (n + 2) := by
  sorry

theorem prob_no_consecutive_ones : 
  (validSeq 12 : ℚ) / 2^12 = 377 / 4096 := by
  sorry

#eval fib 14 + 4096  -- Should output 4473


end NUMINAMATH_CALUDE_valid_seq_equals_fib_prob_no_consecutive_ones_l326_32623


namespace NUMINAMATH_CALUDE_paper_cutting_theorem_l326_32690

/-- Represents the number of pieces after a series of cutting operations -/
def num_pieces (n : ℕ) : ℕ := 4 * n + 1

/-- The result we want to prove is valid -/
def target_result : ℕ := 1993

theorem paper_cutting_theorem :
  ∃ (n : ℕ), num_pieces n = target_result ∧
  ∀ (m : ℕ), ∃ (k : ℕ), num_pieces k = m → m = target_result ∨ m ≠ target_result :=
by sorry

end NUMINAMATH_CALUDE_paper_cutting_theorem_l326_32690


namespace NUMINAMATH_CALUDE_shaded_area_l326_32684

/-- The area of the shaded region in a square with two non-overlapping rectangles --/
theorem shaded_area (total_area : ℝ) (rect1_area rect2_area : ℝ) :
  total_area = 16 →
  rect1_area = 6 →
  rect2_area = 2 →
  total_area - (rect1_area + rect2_area) = 8 := by
sorry


end NUMINAMATH_CALUDE_shaded_area_l326_32684


namespace NUMINAMATH_CALUDE_vampire_survival_l326_32642

/-- The number of pints in a gallon -/
def pints_per_gallon : ℕ := 8

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The amount of blood (in gallons) a vampire needs per week -/
def blood_needed_per_week : ℕ := 7

/-- The amount of blood (in pints) a vampire sucks from each person -/
def blood_per_person : ℕ := 2

/-- The number of people a vampire needs to suck from each day to survive -/
def people_per_day : ℕ := 4

theorem vampire_survival :
  (blood_needed_per_week * pints_per_gallon) / days_per_week / blood_per_person = people_per_day :=
sorry

end NUMINAMATH_CALUDE_vampire_survival_l326_32642


namespace NUMINAMATH_CALUDE_quadratic_inequality_l326_32644

/-- Quadratic function f(x) = -2(x+1)^2 + k -/
def f (k : ℝ) (x : ℝ) : ℝ := -2 * (x + 1)^2 + k

/-- Theorem stating the relationship between f(x) values at x = 2, -3, and -0.5 -/
theorem quadratic_inequality (k : ℝ) : f k 2 < f k (-3) ∧ f k (-3) < f k (-0.5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l326_32644


namespace NUMINAMATH_CALUDE_right_triangle_cosine_l326_32616

theorem right_triangle_cosine (X Y Z : ℝ) (h1 : X = 90) (h2 : Real.sin Z = 4/5) : 
  Real.cos Z = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cosine_l326_32616


namespace NUMINAMATH_CALUDE_middle_number_problem_l326_32657

theorem middle_number_problem (x y z : ℕ) 
  (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 20) (h4 : x + z = 25) (h5 : y + z = 29) (h6 : z - x = 11) :
  y = 13 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_problem_l326_32657


namespace NUMINAMATH_CALUDE_greatest_integer_gcd_4_with_12_l326_32640

theorem greatest_integer_gcd_4_with_12 : 
  ∃ n : ℕ, n < 100 ∧ Nat.gcd n 12 = 4 ∧ ∀ m : ℕ, m < 100 → Nat.gcd m 12 = 4 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_gcd_4_with_12_l326_32640


namespace NUMINAMATH_CALUDE_street_lights_on_triangular_playground_l326_32670

theorem street_lights_on_triangular_playground (side_length : ℝ) (interval : ℝ) :
  side_length = 10 ∧ interval = 3 →
  (3 * side_length) / interval = 10 := by
sorry

end NUMINAMATH_CALUDE_street_lights_on_triangular_playground_l326_32670


namespace NUMINAMATH_CALUDE_turkey_cost_per_kg_turkey_cost_is_two_l326_32681

/-- Given Dabbie's turkey purchase scenario, prove the cost per kilogram of turkey. -/
theorem turkey_cost_per_kg : ℝ → Prop :=
  fun cost_per_kg =>
    let first_turkey_weight := 6
    let second_turkey_weight := 9
    let third_turkey_weight := 2 * second_turkey_weight
    let total_weight := first_turkey_weight + second_turkey_weight + third_turkey_weight
    let total_cost := 66
    cost_per_kg = total_cost / total_weight

/-- The cost per kilogram of turkey is $2. -/
theorem turkey_cost_is_two : turkey_cost_per_kg 2 := by
  sorry

end NUMINAMATH_CALUDE_turkey_cost_per_kg_turkey_cost_is_two_l326_32681


namespace NUMINAMATH_CALUDE_binary_1101101_equals_decimal_109_l326_32626

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define the binary number 1101101
def binary_number : List Bool := [true, false, true, true, false, true, true]

-- Theorem statement
theorem binary_1101101_equals_decimal_109 :
  binary_to_decimal binary_number = 109 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101101_equals_decimal_109_l326_32626


namespace NUMINAMATH_CALUDE_complex_number_location_l326_32628

theorem complex_number_location : ∃ (z : ℂ), z = (5 - 6*I) + (-2 - I) - (3 + 4*I) ∧ z = -11*I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l326_32628


namespace NUMINAMATH_CALUDE_sin_cos_power_relation_l326_32638

theorem sin_cos_power_relation (x : ℝ) :
  (Real.sin x)^10 + (Real.cos x)^10 = 11/36 →
  (Real.sin x)^12 + (Real.cos x)^12 = 5/18 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_power_relation_l326_32638


namespace NUMINAMATH_CALUDE_cyclic_sum_factorization_l326_32630

theorem cyclic_sum_factorization (a b c : ℝ) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = (a - b) * (b - c) * (c - a) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_factorization_l326_32630


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l326_32608

theorem cow_chicken_problem (C H : ℕ) : 4*C + 2*H = 2*(C + H) + 10 → C = 5 :=
by sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l326_32608


namespace NUMINAMATH_CALUDE_trenton_earning_goal_l326_32612

/-- Calculates the earning goal for a salesperson given their fixed weekly earnings,
    commission rate, and sales amount. -/
def earning_goal (fixed_earnings : ℝ) (commission_rate : ℝ) (sales : ℝ) : ℝ :=
  fixed_earnings + commission_rate * sales

/-- Proves that Trenton's earning goal for the week is $500 given the specified conditions. -/
theorem trenton_earning_goal :
  let fixed_earnings : ℝ := 190
  let commission_rate : ℝ := 0.04
  let sales : ℝ := 7750
  earning_goal fixed_earnings commission_rate sales = 500 := by
  sorry


end NUMINAMATH_CALUDE_trenton_earning_goal_l326_32612


namespace NUMINAMATH_CALUDE_supplementary_angle_of_10_degrees_l326_32649

def is_supplementary (a b : ℝ) : Prop :=
  (a + b) % 360 = 180

theorem supplementary_angle_of_10_degrees (k : ℤ) :
  is_supplementary 10 (k * 360 + 250) :=
sorry

end NUMINAMATH_CALUDE_supplementary_angle_of_10_degrees_l326_32649


namespace NUMINAMATH_CALUDE_equation_solution_expression_simplification_l326_32658

-- Part 1
theorem equation_solution :
  ∃ x : ℝ, (x / (2*x - 3) + 5 / (3 - 2*x) = 4) ∧ (x = 1) :=
sorry

-- Part 2
theorem expression_simplification (a : ℝ) (h : a ≠ 2 ∧ a ≠ -2) :
  (a - 2 - 4 / (a - 2)) / ((a - 4) / (a^2 - 4)) = a^2 + 2*a :=
sorry

end NUMINAMATH_CALUDE_equation_solution_expression_simplification_l326_32658


namespace NUMINAMATH_CALUDE_coin_count_l326_32603

theorem coin_count (total_amount : ℕ) (five_dollar_count : ℕ) : 
  total_amount = 125 →
  five_dollar_count = 15 →
  ∃ (two_dollar_count : ℕ), 
    two_dollar_count * 2 + five_dollar_count * 5 = total_amount ∧
    two_dollar_count + five_dollar_count = 40 :=
by sorry

end NUMINAMATH_CALUDE_coin_count_l326_32603


namespace NUMINAMATH_CALUDE_number_of_persons_l326_32683

theorem number_of_persons (total_amount : ℕ) (amount_per_person : ℕ) 
  (h1 : total_amount = 42900) 
  (h2 : amount_per_person = 1950) : 
  total_amount / amount_per_person = 22 := by
  sorry

end NUMINAMATH_CALUDE_number_of_persons_l326_32683


namespace NUMINAMATH_CALUDE_integral_reciprocal_plus_x_l326_32617

theorem integral_reciprocal_plus_x : ∫ x in (1:ℝ)..(2:ℝ), (1/x + x) = Real.log 2 + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_plus_x_l326_32617


namespace NUMINAMATH_CALUDE_water_drip_relationship_faucet_left_on_time_l326_32645

/-- Represents the water drip rate in mL per second -/
def drip_rate : ℝ := 2 * 0.05

/-- Represents the relationship between time (in hours) and water volume (in mL) -/
def water_volume (time : ℝ) : ℝ := (3600 * drip_rate) * time

theorem water_drip_relationship (time : ℝ) (volume : ℝ) (h : time ≥ 0) :
  water_volume time = 360 * time :=
sorry

theorem faucet_left_on_time (volume : ℝ) (h : volume = 1620) :
  ∃ (time : ℝ), water_volume time = volume ∧ time = 4.5 :=
sorry

end NUMINAMATH_CALUDE_water_drip_relationship_faucet_left_on_time_l326_32645


namespace NUMINAMATH_CALUDE_sasha_sticker_problem_l326_32665

theorem sasha_sticker_problem (m n : ℕ) (t : ℝ) : 
  0 < m ∧ m < n ∧ 1 < t ∧ 
  m * t + n = 100 ∧ 
  m + n * t = 101 → 
  n = 34 ∨ n = 66 := by
sorry

end NUMINAMATH_CALUDE_sasha_sticker_problem_l326_32665


namespace NUMINAMATH_CALUDE_james_money_theorem_l326_32627

/-- The amount of money James has now, given the conditions -/
def jamesTotal (billsFound : ℕ) (billValue : ℕ) (initialWallet : ℕ) : ℕ :=
  billsFound * billValue + initialWallet

/-- Theorem stating that James has $135 given the problem conditions -/
theorem james_money_theorem :
  jamesTotal 3 20 75 = 135 := by
  sorry

end NUMINAMATH_CALUDE_james_money_theorem_l326_32627


namespace NUMINAMATH_CALUDE_second_smallest_packs_l326_32667

/-- The number of hot dogs in each pack -/
def hot_dogs_per_pack : ℕ := 9

/-- The number of buns in each pack -/
def buns_per_pack : ℕ := 7

/-- The number of hot dogs left over after the barbecue -/
def leftover_hot_dogs : ℕ := 6

/-- 
Theorem: The second smallest number of packs of hot dogs that satisfies 
the conditions of the barbecue problem is 10.
-/
theorem second_smallest_packs : 
  (∃ m : ℕ, m < 10 ∧ hot_dogs_per_pack * m ≡ leftover_hot_dogs [MOD buns_per_pack]) ∧
  (∀ k : ℕ, k < 10 → hot_dogs_per_pack * k ≡ leftover_hot_dogs [MOD buns_per_pack] → 
    ∃ m : ℕ, m < k ∧ hot_dogs_per_pack * m ≡ leftover_hot_dogs [MOD buns_per_pack]) ∧
  hot_dogs_per_pack * 10 ≡ leftover_hot_dogs [MOD buns_per_pack] := by
  sorry


end NUMINAMATH_CALUDE_second_smallest_packs_l326_32667


namespace NUMINAMATH_CALUDE_monkey_climb_time_l326_32687

/-- A monkey climbing a tree with specific conditions -/
def monkey_climb (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) : ℕ :=
  let effective_climb := hop_distance - slip_distance
  let full_climbs := (tree_height - 1) / effective_climb
  let remaining_distance := (tree_height - 1) % effective_climb
  full_climbs + if remaining_distance > 0 then 1 else 0

/-- Theorem stating the time taken by the monkey to climb the tree -/
theorem monkey_climb_time :
  monkey_climb 17 3 2 = 17 := by sorry

end NUMINAMATH_CALUDE_monkey_climb_time_l326_32687


namespace NUMINAMATH_CALUDE_linda_jeans_sold_l326_32618

/-- The number of jeans sold by Linda -/
def jeans_sold : ℕ := 4

/-- The price of a pair of jeans in dollars -/
def jeans_price : ℕ := 11

/-- The price of a tee in dollars -/
def tees_price : ℕ := 8

/-- The number of tees sold -/
def tees_sold : ℕ := 7

/-- The total revenue in dollars -/
def total_revenue : ℕ := 100

theorem linda_jeans_sold :
  jeans_sold * jeans_price + tees_sold * tees_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_linda_jeans_sold_l326_32618


namespace NUMINAMATH_CALUDE_max_area_triangle_OPQ_l326_32672

/-- Parabola in Cartesian coordinates -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point on a parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  h : x^2 = 2 * c.p * y

/-- Line intersecting a parabola -/
structure IntersectingLine (c : Parabola) where
  k : ℝ
  b : ℝ
  h : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 = 2 * c.p * (k * x₁ + b) ∧ x₂^2 = 2 * c.p * (k * x₂ + b)

/-- Theorem: Maximum area of triangle OPQ -/
theorem max_area_triangle_OPQ (c : Parabola) (a : PointOnParabola c) (l : IntersectingLine c) :
  a.x^2 + a.y^2 = (3/2)^2 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧
    x₁^2 = 2 * c.p * y₁ ∧ x₂^2 = 2 * c.p * y₂ ∧
    y₁ = l.k * x₁ + l.b ∧ y₂ = l.k * x₂ + l.b ∧
    (y₁ + y₂) / 2 = 1) →
  (∃ (area : ℝ), area ≤ 2 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ →
      x₁^2 = 2 * c.p * y₁ → x₂^2 = 2 * c.p * y₂ →
      y₁ = l.k * x₁ + l.b → y₂ = l.k * x₂ + l.b →
      (y₁ + y₂) / 2 = 1 →
      area ≥ abs (x₁ * y₂ - x₂ * y₁) / 2) :=
sorry

end NUMINAMATH_CALUDE_max_area_triangle_OPQ_l326_32672


namespace NUMINAMATH_CALUDE_trig_fraction_simplification_l326_32625

theorem trig_fraction_simplification (α : ℝ) : 
  (Real.cos (π + α) * Real.sin (α + 2*π)) / (Real.sin (-α - π) * Real.cos (-π - α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_simplification_l326_32625


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l326_32632

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l326_32632


namespace NUMINAMATH_CALUDE_quadratic_point_relation_l326_32604

/-- The quadratic function f(x) = x^2 + x + 1 -/
def f (x : ℝ) : ℝ := x^2 + x + 1

theorem quadratic_point_relation :
  let y₁ := f (-3)
  let y₂ := f 2
  let y₃ := f (1/2)
  y₃ < y₁ ∧ y₁ = y₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_point_relation_l326_32604


namespace NUMINAMATH_CALUDE_catch_turtle_certain_l326_32674

-- Define the type for idioms
inductive Idiom
| CatchTurtle
| CarveBoat
| WaitRabbit
| FishMoon

-- Define a function to determine if an idiom represents a certain event
def isCertainEvent (i : Idiom) : Prop :=
  match i with
  | Idiom.CatchTurtle => True
  | _ => False

-- Theorem statement
theorem catch_turtle_certain :
  ∀ i : Idiom, isCertainEvent i ↔ i = Idiom.CatchTurtle :=
by
  sorry


end NUMINAMATH_CALUDE_catch_turtle_certain_l326_32674


namespace NUMINAMATH_CALUDE_sum_of_powers_of_two_l326_32695

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 = 2^5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_two_l326_32695


namespace NUMINAMATH_CALUDE_bears_on_shelves_l326_32662

/-- Given an initial stock of bears, a new shipment, and a number of bears per shelf,
    calculate the number of shelves required to store all bears. -/
def shelves_required (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

/-- Theorem stating that with 17 initial bears, 10 new bears, and 9 bears per shelf,
    3 shelves are required. -/
theorem bears_on_shelves :
  shelves_required 17 10 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bears_on_shelves_l326_32662


namespace NUMINAMATH_CALUDE_some_value_proof_l326_32688

theorem some_value_proof (a : ℝ) : 
  (∀ x : ℝ, |x - a| = 100 → (a + 100) + (a - 100) = 24) → 
  a = 12 := by
  sorry

end NUMINAMATH_CALUDE_some_value_proof_l326_32688


namespace NUMINAMATH_CALUDE_stickers_remaining_proof_l326_32639

/-- Calculates the number of stickers remaining after losing a page. -/
def stickers_remaining (stickers_per_page : ℕ) (initial_pages : ℕ) (pages_lost : ℕ) : ℕ :=
  (initial_pages - pages_lost) * stickers_per_page

/-- Proves that the number of stickers remaining is 220. -/
theorem stickers_remaining_proof :
  stickers_remaining 20 12 1 = 220 := by
  sorry

end NUMINAMATH_CALUDE_stickers_remaining_proof_l326_32639


namespace NUMINAMATH_CALUDE_right_triangles_on_circle_l326_32677

theorem right_triangles_on_circle (n : ℕ) (h : n = 100) :
  ¬ (∃ (k : ℕ), k = 1000 ∧ k = (n / 2) * (n - 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangles_on_circle_l326_32677


namespace NUMINAMATH_CALUDE_arc_length_sixty_degrees_l326_32633

theorem arc_length_sixty_degrees (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 1 → θ = 60 → l = (θ * π * r) / 180 → l = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_sixty_degrees_l326_32633


namespace NUMINAMATH_CALUDE_fraction_comparison_l326_32643

theorem fraction_comparison : 
  (10 / 8 : ℚ) = 5 / 4 ∧ 
  (5 / 4 : ℚ) = 5 / 4 ∧ 
  (15 / 12 : ℚ) = 5 / 4 ∧ 
  (6 / 5 : ℚ) ≠ 5 / 4 ∧ 
  (50 / 40 : ℚ) = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l326_32643


namespace NUMINAMATH_CALUDE_percentage_relation_l326_32694

/-- Given that b as a percentage of x is equal to x as a percentage of (a + b),
    and this percentage is 61.80339887498949%, prove that a = b * (38.1966/61.8034) -/
theorem percentage_relation (a b x : ℝ) 
  (h1 : b / x = x / (a + b)) 
  (h2 : b / x = 61.80339887498949 / 100) : 
  a = b * (38.1966 / 61.8034) := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l326_32694


namespace NUMINAMATH_CALUDE_frequency_of_six_is_nineteen_hundredths_l326_32629

/-- Represents the outcome of rolling a fair six-sided die multiple times -/
structure DieRollOutcome where
  total_rolls : ℕ
  sixes_count : ℕ

/-- Calculates the frequency of rolling a 6 -/
def frequency_of_six (outcome : DieRollOutcome) : ℚ :=
  outcome.sixes_count / outcome.total_rolls

/-- Theorem stating that for the given die roll outcome, the frequency of rolling a 6 is 0.19 -/
theorem frequency_of_six_is_nineteen_hundredths 
  (outcome : DieRollOutcome) 
  (h1 : outcome.total_rolls = 100) 
  (h2 : outcome.sixes_count = 19) : 
  frequency_of_six outcome = 19 / 100 := by
  sorry

end NUMINAMATH_CALUDE_frequency_of_six_is_nineteen_hundredths_l326_32629


namespace NUMINAMATH_CALUDE_art_math_supplies_cost_l326_32614

-- Define the prices of items
def folder_price : ℚ := 3.5
def notebook_price : ℚ := 3
def binder_price : ℚ := 5
def pencil_price : ℚ := 1
def eraser_price : ℚ := 0.75
def highlighter_price : ℚ := 3.25
def marker_price : ℚ := 3.5
def sticky_note_price : ℚ := 2.5
def calculator_price : ℚ := 10.5
def sketchbook_price : ℚ := 4.5
def paint_set_price : ℚ := 18
def color_pencil_price : ℚ := 7

-- Define the quantities
def num_classes : ℕ := 12
def folders_per_class : ℕ := 1
def notebooks_per_class : ℕ := 2
def binders_per_class : ℕ := 1
def pencils_per_class : ℕ := 3
def erasers_per_6_pencils : ℕ := 2

-- Define the total spent
def total_spent : ℚ := 210

-- Theorem statement
theorem art_math_supplies_cost : 
  paint_set_price + color_pencil_price + calculator_price + sketchbook_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_art_math_supplies_cost_l326_32614


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l326_32600

theorem quadratic_inequality_solution_set (m : ℝ) (h : m > 1) :
  {x : ℝ | x^2 + (m - 1) * x - m ≥ 0} = {x : ℝ | x ≤ -m ∨ x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l326_32600


namespace NUMINAMATH_CALUDE_eight_power_plus_six_divisible_by_seven_l326_32615

theorem eight_power_plus_six_divisible_by_seven (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℤ, (8 : ℤ)^n + 6 = 7 * k :=
by sorry

end NUMINAMATH_CALUDE_eight_power_plus_six_divisible_by_seven_l326_32615


namespace NUMINAMATH_CALUDE_money_distribution_l326_32679

/-- The problem of distributing money among five people with specific conditions -/
theorem money_distribution (a b c d e : ℕ) : 
  a + b + c + d + e = 1010 ∧
  (a - 25) / 4 = (b - 10) / 3 ∧
  (a - 25) / 4 = (c - 15) / 6 ∧
  (a - 25) / 4 = (d - 20) / 2 ∧
  (a - 25) / 4 = (e - 30) / 5 →
  c = 288 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l326_32679


namespace NUMINAMATH_CALUDE_buddy_system_l326_32659

theorem buddy_system (s n : ℕ) (h1 : n ≠ 0) (h2 : s ≠ 0) : 
  (n / 4 : ℚ) = (s / 2 : ℚ) → 
  ((n / 4 + s / 2) / (n + s) : ℚ) = (1 / 3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_buddy_system_l326_32659


namespace NUMINAMATH_CALUDE_trapezoid_ratio_l326_32682

/-- Represents a trapezoid ABCD with a point P inside -/
structure Trapezoid :=
  (AB CD : ℝ)
  (height : ℝ)
  (area_PCD area_PAD area_PBC area_PAB : ℝ)

/-- The theorem stating the ratio of parallel sides in the trapezoid -/
theorem trapezoid_ratio (T : Trapezoid) : 
  T.AB > T.CD →
  T.height = 8 →
  T.area_PCD = 4 →
  T.area_PAD = 6 →
  T.area_PBC = 5 →
  T.area_PAB = 7 →
  T.AB / T.CD = 4 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_ratio_l326_32682


namespace NUMINAMATH_CALUDE_T_2021_2022_2023_even_l326_32686

def T : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | 2 => 2
  | n + 3 => T (n + 2) + T (n + 1) + T n

theorem T_2021_2022_2023_even :
  Even (T 2021) ∧ Even (T 2022) ∧ Even (T 2023) := by sorry

end NUMINAMATH_CALUDE_T_2021_2022_2023_even_l326_32686


namespace NUMINAMATH_CALUDE_range_of_f_l326_32699

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2^x - 5 else 3 * Real.sin x

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ioc (-5) 3 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l326_32699


namespace NUMINAMATH_CALUDE_pages_used_per_day_l326_32607

/-- Given 5 notebooks with 40 pages each, lasting for 50 days, prove that 4 pages are used per day. -/
theorem pages_used_per_day (num_notebooks : ℕ) (pages_per_notebook : ℕ) (days_lasted : ℕ) :
  num_notebooks = 5 →
  pages_per_notebook = 40 →
  days_lasted = 50 →
  (num_notebooks * pages_per_notebook) / days_lasted = 4 :=
by sorry

end NUMINAMATH_CALUDE_pages_used_per_day_l326_32607


namespace NUMINAMATH_CALUDE_riverdale_high_quiz_l326_32601

theorem riverdale_high_quiz (total_contestants : ℕ) (total_students : ℕ) 
  (h1 : total_contestants = 234) 
  (h2 : total_students = 420) : 
  ∃ (freshmen juniors : ℕ), 
    freshmen + juniors = total_students ∧
    (3 * freshmen) / 7 + (3 * juniors) / 4 = total_contestants ∧
    freshmen = 64 ∧ 
    juniors = 356 := by
  sorry

end NUMINAMATH_CALUDE_riverdale_high_quiz_l326_32601


namespace NUMINAMATH_CALUDE_bananas_left_l326_32697

/-- The number of bananas in a dozen -/
def dozen : ℕ := 12

/-- The number of bananas Elizabeth ate -/
def eaten : ℕ := 4

/-- Theorem: If Elizabeth bought a dozen bananas and ate 4 of them, then 8 bananas are left -/
theorem bananas_left (bought : ℕ) (ate : ℕ) (h1 : bought = dozen) (h2 : ate = eaten) :
  bought - ate = 8 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_l326_32697


namespace NUMINAMATH_CALUDE_expression_value_l326_32655

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x = 1 / y) (hzy : z = 1 / y) : (x + 1 / x) * (z - 1 / z) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l326_32655
