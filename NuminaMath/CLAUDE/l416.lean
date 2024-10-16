import Mathlib

namespace NUMINAMATH_CALUDE_jackson_chairs_count_l416_41616

/-- The number of chairs Jackson needs to buy for his restaurant -/
def total_chairs (four_seat_tables six_seat_tables : ℕ) (seats_per_four_seat_table seats_per_six_seat_table : ℕ) : ℕ :=
  four_seat_tables * seats_per_four_seat_table + six_seat_tables * seats_per_six_seat_table

/-- Proof that Jackson needs to buy 96 chairs for his restaurant -/
theorem jackson_chairs_count :
  total_chairs 6 12 4 6 = 96 := by
  sorry

end NUMINAMATH_CALUDE_jackson_chairs_count_l416_41616


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_4_real_l416_41646

theorem sqrt_2x_minus_4_real (x : ℝ) : (∃ (y : ℝ), y ^ 2 = 2 * x - 4) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_4_real_l416_41646


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l416_41651

theorem abs_inequality_solution_set (x : ℝ) :
  |x - 2| < |2*x + 1| ↔ x < -3 ∨ x > 1/3 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l416_41651


namespace NUMINAMATH_CALUDE_ellen_initial_legos_l416_41649

/-- The number of Legos Ellen lost -/
def lost_legos : ℕ := 17

/-- The number of Legos Ellen currently has -/
def current_legos : ℕ := 2063

/-- The initial number of Legos Ellen had -/
def initial_legos : ℕ := current_legos + lost_legos

theorem ellen_initial_legos : initial_legos = 2080 := by
  sorry

end NUMINAMATH_CALUDE_ellen_initial_legos_l416_41649


namespace NUMINAMATH_CALUDE_michael_and_anna_ages_l416_41675

theorem michael_and_anna_ages :
  ∀ (michael anna : ℕ),
  michael = anna + 8 →
  michael + 12 = 3 * (anna - 6) →
  michael + anna = 46 :=
by sorry

end NUMINAMATH_CALUDE_michael_and_anna_ages_l416_41675


namespace NUMINAMATH_CALUDE_reciprocal_squares_sum_l416_41679

theorem reciprocal_squares_sum (a b : ℕ) (h : a * b = 3) :
  (1 : ℚ) / a^2 + 1 / b^2 = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_squares_sum_l416_41679


namespace NUMINAMATH_CALUDE_f_increasing_condition_and_range_f_range_on_interval_l416_41699

/-- The function f(x) = x^2 - 4x -/
def f (x : ℝ) : ℝ := x^2 - 4*x

theorem f_increasing_condition_and_range (a : ℝ) :
  (∀ x ≥ 2*a - 1, MonotoneOn f (Set.Ici (2*a - 1))) → a ≥ 3/2 :=
sorry

theorem f_range_on_interval :
  Set.image f (Set.Icc 1 7) = Set.Icc (-4) 21 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_condition_and_range_f_range_on_interval_l416_41699


namespace NUMINAMATH_CALUDE_shirt_cost_l416_41697

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 86) : S = 24 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l416_41697


namespace NUMINAMATH_CALUDE_largest_k_inequality_l416_41676

theorem largest_k_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ k : ℝ, k > 174960 → ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2*d)^5) < k * a * b * c * d^3) ∧
  (∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 →
    (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2*d)^5) ≥ 174960 * a * b * c * d^3) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_inequality_l416_41676


namespace NUMINAMATH_CALUDE_cone_max_section_area_cone_max_section_area_condition_l416_41629

/-- Given a cone with lateral surface that unfolds into a sector with radius 2 and central angle 5π/3,
    the maximum area of any section determined by two generatrices is 2. -/
theorem cone_max_section_area :
  ∀ (r : ℝ) (l : ℝ) (a : ℝ),
  l = 2 →
  2 * π * r = 2 * (5 * π / 3) →
  0 < a →
  a ≤ 2 * r →
  (a / 2) * Real.sqrt (4 - a^2 / 4) ≤ 2 :=
by sorry

/-- The maximum area is achieved when a = 2√2 -/
theorem cone_max_section_area_condition (r : ℝ) (l : ℝ) :
  l = 2 →
  2 * π * r = 2 * (5 * π / 3) →
  (2 * Real.sqrt 2 / 2) * Real.sqrt (4 - (2 * Real.sqrt 2)^2 / 4) = 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_max_section_area_cone_max_section_area_condition_l416_41629


namespace NUMINAMATH_CALUDE_minute_hand_rotation_l416_41612

theorem minute_hand_rotation (start_position : ℕ) (rotation_angle : ℝ) : 
  start_position = 12 → 
  rotation_angle = 120 → 
  (rotation_angle / (360 / 12) + start_position) % 12 = 4 :=
by sorry

end NUMINAMATH_CALUDE_minute_hand_rotation_l416_41612


namespace NUMINAMATH_CALUDE_profit_ratio_problem_l416_41653

/-- The profit ratio problem -/
theorem profit_ratio_problem (profit_3_shirts profit_7_shirts_3_sandals : ℚ) 
  (h1 : profit_3_shirts = 21)
  (h2 : profit_7_shirts_3_sandals = 175) :
  (2 * profit_3_shirts) = ((profit_7_shirts_3_sandals - (7 / 3) * profit_3_shirts) / 3 * 2) :=
by sorry

end NUMINAMATH_CALUDE_profit_ratio_problem_l416_41653


namespace NUMINAMATH_CALUDE_train_speed_l416_41624

/-- Proves that a train of length 400 meters crossing a pole in 12 seconds has a speed of 120 km/hr -/
theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) : 
  length = 400 ∧ time = 12 → speed = (length / 1000) / (time / 3600) → speed = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l416_41624


namespace NUMINAMATH_CALUDE_vector_equation_l416_41648

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (-2, 1)

theorem vector_equation (x y : ℝ) (h : c = x • a + y • b) : x - y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_l416_41648


namespace NUMINAMATH_CALUDE_intersection_points_count_l416_41670

/-- The number of points with positive x-coordinates that lie on at least two of the graphs
    y = log₂x, y = 1/log₂x, y = -log₂x, and y = -1/log₂x -/
theorem intersection_points_count : ℕ := by
  sorry

#check intersection_points_count

end NUMINAMATH_CALUDE_intersection_points_count_l416_41670


namespace NUMINAMATH_CALUDE_complement_of_A_l416_41602

-- Define the universal set U
def U : Set Int := {-1, 0, 2}

-- Define set A
def A : Set Int := {-1, 0}

-- Define the complement operation
def complement (S : Set Int) : Set Int :=
  {x | x ∈ U ∧ x ∉ S}

-- Theorem statement
theorem complement_of_A : complement A = {2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l416_41602


namespace NUMINAMATH_CALUDE_family_gathering_handshakes_l416_41684

theorem family_gathering_handshakes :
  let num_twin_sets : ℕ := 10
  let num_quadruplet_sets : ℕ := 5
  let num_twins : ℕ := num_twin_sets * 2
  let num_quadruplets : ℕ := num_quadruplet_sets * 4
  let twin_handshakes : ℕ := num_twins * (num_twins - 2)
  let quadruplet_handshakes : ℕ := num_quadruplets * (num_quadruplets - 4)
  let twin_to_quadruplet : ℕ := num_twins * (2 * num_quadruplets / 3)
  let quadruplet_to_twin : ℕ := num_quadruplets * (3 * num_twins / 4)
  let total_handshakes : ℕ := (twin_handshakes + quadruplet_handshakes + twin_to_quadruplet + quadruplet_to_twin) / 2
  total_handshakes = 620 :=
by sorry

end NUMINAMATH_CALUDE_family_gathering_handshakes_l416_41684


namespace NUMINAMATH_CALUDE_sum_fraction_positive_l416_41642

theorem sum_fraction_positive (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_neg : a * b * c < 0) : 
  (a^2 + b^2) / c + (b^2 + c^2) / a + (c^2 + a^2) / b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_positive_l416_41642


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l416_41667

/-- Given a train of length 250 meters that passes a pole in 10 seconds
    and a platform in 60 seconds, prove that the time taken to pass
    only the platform is 50 seconds. -/
theorem train_platform_passing_time
  (train_length : ℝ)
  (pole_passing_time : ℝ)
  (platform_total_passing_time : ℝ)
  (h1 : train_length = 250)
  (h2 : pole_passing_time = 10)
  (h3 : platform_total_passing_time = 60) :
  let train_speed := train_length / pole_passing_time
  let platform_length := train_speed * platform_total_passing_time - train_length
  platform_length / train_speed = 50 := by
sorry

end NUMINAMATH_CALUDE_train_platform_passing_time_l416_41667


namespace NUMINAMATH_CALUDE_student_number_problem_l416_41654

theorem student_number_problem :
  ∃ x : ℝ, 2 * x - 138 = 104 ∧ x = 121 := by sorry

end NUMINAMATH_CALUDE_student_number_problem_l416_41654


namespace NUMINAMATH_CALUDE_inequality_solution_min_value_theorem_min_value_achieved_l416_41611

-- Define the function f
def f (a b x : ℝ) : ℝ := |x - a| + |x + b|

-- Theorem 1
theorem inequality_solution (x : ℝ) :
  f 1 2 x ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := by sorry

-- Theorem 2
theorem min_value_theorem (a b : ℝ) :
  a > 0 → b > 0 → (∀ x, f a b x ≥ 3) → (∃ x, f a b x = 3) →
  a^2 / b + b^2 / a ≥ 3 := by sorry

-- Corollary: The minimum value is achieved
theorem min_value_achieved (a b : ℝ) :
  a > 0 → b > 0 → (∀ x, f a b x ≥ 3) → (∃ x, f a b x = 3) →
  ∃ a b, a > 0 ∧ b > 0 ∧ a^2 / b + b^2 / a = 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_min_value_theorem_min_value_achieved_l416_41611


namespace NUMINAMATH_CALUDE_exists_x_in_interval_l416_41678

theorem exists_x_in_interval (x : ℝ) : 
  ∃ x, x ∈ Set.Icc (-1 : ℝ) (3/10) ∧ x^2 + 3*x - 1 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_in_interval_l416_41678


namespace NUMINAMATH_CALUDE_train_speed_proof_l416_41605

/-- Proves that the new train speed is 256 km/h given the problem conditions -/
theorem train_speed_proof (distance : ℝ) (speed_multiplier : ℝ) (time_reduction : ℝ) 
  (h1 : distance = 1280)
  (h2 : speed_multiplier = 3.2)
  (h3 : time_reduction = 11)
  (h4 : ∀ x : ℝ, distance / x - distance / (speed_multiplier * x) = time_reduction) :
  speed_multiplier * (distance / (distance / speed_multiplier + time_reduction)) = 256 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_proof_l416_41605


namespace NUMINAMATH_CALUDE_ellipse_line_theorem_l416_41636

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (x : ℝ) : Prop := x = -2

-- Define a line passing through a point
def line_through_point (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the perpendicular bisector of a line segment
def perpendicular_bisector (k : ℝ) (x y : ℝ) : Prop := 
  y + k / (1 + 2*k^2) = -(1/k) * (x - 2*k^2 / (1 + 2*k^2))

-- Define the theorem
theorem ellipse_line_theorem (k : ℝ) (x₁ y₁ x₂ y₂ xp yp xc yc : ℝ) : 
  ellipse x₁ y₁ → 
  ellipse x₂ y₂ → 
  line_through_point k x₁ y₁ → 
  line_through_point k x₂ y₂ → 
  perpendicular_bisector k xp yp → 
  perpendicular_bisector k xc yc → 
  line_l xp → 
  (xc - 1)^2 + yc^2 = ((x₂ - x₁)^2 + (y₂ - y₁)^2) / 4 → 
  (xp - xc)^2 + (yp - yc)^2 = 4 * ((x₂ - x₁)^2 + (y₂ - y₁)^2) → 
  (k = 1 ∨ k = -1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_theorem_l416_41636


namespace NUMINAMATH_CALUDE_smallest_fruit_distribution_l416_41603

theorem smallest_fruit_distribution (N : ℕ) : N = 79 ↔ 
  N > 0 ∧
  (N - 1) % 3 = 0 ∧
  (2 * (N - 1) / 3 - 1) % 3 = 0 ∧
  ((2 * N - 5) / 3 - 1) % 3 = 0 ∧
  ((4 * N - 28) / 9 - 1) % 3 = 0 ∧
  ((8 * N - 56) / 27 - 1) % 3 = 0 ∧
  ∀ (M : ℕ), M < N → 
    (M > 0 ∧
    (M - 1) % 3 = 0 ∧
    (2 * (M - 1) / 3 - 1) % 3 = 0 ∧
    ((2 * M - 5) / 3 - 1) % 3 = 0 ∧
    ((4 * M - 28) / 9 - 1) % 3 = 0 ∧
    ((8 * M - 56) / 27 - 1) % 3 = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_smallest_fruit_distribution_l416_41603


namespace NUMINAMATH_CALUDE_l_shapes_on_8x8_chessboard_l416_41690

/-- Represents a chessboard --/
structure Chessboard where
  size : ℕ
  size_pos : size > 0

/-- Represents an L-shaped pattern on a chessboard --/
structure LShape where
  board : Chessboard

/-- Count of L-shaped patterns on a given chessboard --/
def count_l_shapes (board : Chessboard) : ℕ :=
  sorry

theorem l_shapes_on_8x8_chessboard :
  ∃ (board : Chessboard), board.size = 8 ∧ count_l_shapes board = 196 :=
sorry

end NUMINAMATH_CALUDE_l_shapes_on_8x8_chessboard_l416_41690


namespace NUMINAMATH_CALUDE_blue_chip_percentage_l416_41672

theorem blue_chip_percentage
  (total : ℕ)
  (blue : ℕ)
  (white : ℕ)
  (green : ℕ)
  (h1 : blue = 3)
  (h2 : white = total / 2)
  (h3 : green = 12)
  (h4 : total = blue + white + green) :
  (blue : ℚ) / total * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_blue_chip_percentage_l416_41672


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficients_l416_41677

theorem quadratic_inequality_coefficients 
  (a b : ℝ) 
  (h1 : Set.Ioo (-2 : ℝ) (-1/4 : ℝ) = {x : ℝ | 5 - x > 7 * |x + 1|})
  (h2 : Set.Ioo (-2 : ℝ) (-1/4 : ℝ) = {x : ℝ | a * x^2 + b * x - 2 > 0}) :
  a = -4 ∧ b = -9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficients_l416_41677


namespace NUMINAMATH_CALUDE_intersection_unique_l416_41671

/-- Two lines in 2D space -/
def line1 (t : ℝ) : ℝ × ℝ := (4 + 3 * t, 1 - 2 * t)
def line2 (u : ℝ) : ℝ × ℝ := (-2 + 4 * u, 5 - u)

/-- The point of intersection -/
def intersection_point : ℝ × ℝ := (-2, 5)

/-- Theorem stating that the intersection_point is the unique point of intersection for the two lines -/
theorem intersection_unique :
  (∃ (t : ℝ), line1 t = intersection_point) ∧
  (∃ (u : ℝ), line2 u = intersection_point) ∧
  (∀ (p : ℝ × ℝ), (∃ (t : ℝ), line1 t = p) ∧ (∃ (u : ℝ), line2 u = p) → p = intersection_point) :=
by sorry

end NUMINAMATH_CALUDE_intersection_unique_l416_41671


namespace NUMINAMATH_CALUDE_two_pump_filling_time_l416_41645

/-- Given two pumps with different filling rates, calculate the time taken to fill a tank when both pumps work together. -/
theorem two_pump_filling_time 
  (small_pump_rate : ℝ) 
  (large_pump_rate : ℝ) 
  (h1 : small_pump_rate = 1 / 4) 
  (h2 : large_pump_rate = 2) : 
  1 / (small_pump_rate + large_pump_rate) = 4 / 9 := by
  sorry

#check two_pump_filling_time

end NUMINAMATH_CALUDE_two_pump_filling_time_l416_41645


namespace NUMINAMATH_CALUDE_buttons_problem_l416_41614

theorem buttons_problem (sue kendra mari : ℕ) : 
  sue = kendra / 2 →
  sue = 6 →
  mari = 5 * kendra + 4 →
  mari = 64 := by
sorry

end NUMINAMATH_CALUDE_buttons_problem_l416_41614


namespace NUMINAMATH_CALUDE_january_salary_is_5300_l416_41674

/-- Represents monthly salaries -/
structure MonthlySalaries where
  J : ℕ  -- January
  F : ℕ  -- February
  M : ℕ  -- March
  A : ℕ  -- April
  Ma : ℕ -- May
  Ju : ℕ -- June

/-- Theorem stating the conditions and the result to be proved -/
theorem january_salary_is_5300 (s : MonthlySalaries) : 
  (s.J + s.F + s.M + s.A) / 4 = 8000 →
  (s.F + s.M + s.A + s.Ma) / 4 = 8300 →
  (s.M + s.A + s.Ma + s.Ju) / 4 = 8600 →
  s.Ma = 6500 →
  s.J = 5300 := by
  sorry

#check january_salary_is_5300

end NUMINAMATH_CALUDE_january_salary_is_5300_l416_41674


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l416_41631

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 6) ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l416_41631


namespace NUMINAMATH_CALUDE_mary_story_characters_l416_41632

theorem mary_story_characters (total : ℕ) (a b c g d e f h : ℕ) : 
  total = 360 →
  a = total / 3 →
  b = (total - a) / 4 →
  c = (total - a - b) / 5 →
  g = (total - a - b - c) / 6 →
  d + e + f + h = total - a - b - c - g →
  d = 3 * e →
  f = 2 * e →
  h = f →
  d = 45 :=
by sorry

end NUMINAMATH_CALUDE_mary_story_characters_l416_41632


namespace NUMINAMATH_CALUDE_stream_speed_l416_41662

theorem stream_speed (rowing_speed : ℝ) (total_time : ℝ) (distance : ℝ) (stream_speed : ℝ) : 
  rowing_speed = 10 →
  total_time = 5 →
  distance = 24 →
  (distance / (rowing_speed - stream_speed) + distance / (rowing_speed + stream_speed) = total_time) →
  stream_speed = 2 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l416_41662


namespace NUMINAMATH_CALUDE_exponential_inequality_l416_41644

theorem exponential_inequality (x y a : ℝ) (hx : x > y) (hy : y > 1) (ha : 0 < a) (ha1 : a < 1) :
  a^x < a^y := by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l416_41644


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l416_41683

theorem min_value_sum_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 5) : 
  (9 / x + 4 / y + 25 / z) ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l416_41683


namespace NUMINAMATH_CALUDE_sum_of_constants_l416_41688

/-- Given constants a, b, and c satisfying the conditions, prove that a + 2b + 3c = 65 -/
theorem sum_of_constants (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ x < -4 ∨ |x - 25| ≤ 2)
  (h2 : a < b) : 
  a + 2*b + 3*c = 65 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_constants_l416_41688


namespace NUMINAMATH_CALUDE_andrew_may_cookie_spending_l416_41628

/-- Andrew's cookie spending in May -/
def andrew_cookie_spending (days_in_may : ℕ) (cookies_per_day : ℕ) (cost_per_cookie : ℕ) : ℕ :=
  days_in_may * cookies_per_day * cost_per_cookie

/-- Theorem: Andrew's cookie spending in May is 1395 dollars -/
theorem andrew_may_cookie_spending :
  andrew_cookie_spending 31 3 15 = 1395 :=
by sorry

end NUMINAMATH_CALUDE_andrew_may_cookie_spending_l416_41628


namespace NUMINAMATH_CALUDE_division_remainder_and_double_l416_41659

theorem division_remainder_and_double : 
  let dividend := 4509
  let divisor := 98
  let remainder := dividend % divisor
  let doubled_remainder := 2 * remainder
  remainder = 1 ∧ doubled_remainder = 2 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_and_double_l416_41659


namespace NUMINAMATH_CALUDE_eventually_all_zero_l416_41625

/-- Represents a quadruple of integers -/
structure Quadruple where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Generates the next quadruple in the sequence -/
def nextQuadruple (q : Quadruple) : Quadruple := {
  a := |q.a - q.b|
  b := |q.b - q.c|
  c := |q.c - q.d|
  d := |q.d - q.a|
}

/-- Checks if all elements in a quadruple are zero -/
def isAllZero (q : Quadruple) : Prop :=
  q.a = 0 ∧ q.b = 0 ∧ q.c = 0 ∧ q.d = 0

/-- Theorem: The sequence will eventually reach all zeros -/
theorem eventually_all_zero (q₀ : Quadruple) : 
  ∃ n : ℕ, isAllZero ((nextQuadruple^[n]) q₀) :=
sorry


end NUMINAMATH_CALUDE_eventually_all_zero_l416_41625


namespace NUMINAMATH_CALUDE_increasing_function_a_range_l416_41695

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4*a*x else (2*a+3)*x - 4*a + 5

theorem increasing_function_a_range (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) → a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_a_range_l416_41695


namespace NUMINAMATH_CALUDE_perfect_square_triples_l416_41664

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem perfect_square_triples :
  ∀ a b c : ℕ,
    (is_perfect_square (a^2 + 2*b + c) ∧
     is_perfect_square (b^2 + 2*c + a) ∧
     is_perfect_square (c^2 + 2*a + b)) →
    ((a = 0 ∧ b = 0 ∧ c = 0) ∨
     (a = 1 ∧ b = 1 ∧ c = 1) ∨
     (a = 43 ∧ b = 127 ∧ c = 106)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_triples_l416_41664


namespace NUMINAMATH_CALUDE_correct_average_l416_41621

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 21 ∧ 
  wrong_num = 26 ∧ 
  correct_num = 36 →
  (n : ℚ) * incorrect_avg + (correct_num - wrong_num) = n * 22 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l416_41621


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l416_41652

theorem complex_modulus_sqrt_5 (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (eq : a + 2 * i = 1 - b * i) : 
  Complex.abs (a + b * i) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l416_41652


namespace NUMINAMATH_CALUDE_sequence_completeness_l416_41655

theorem sequence_completeness (a : ℕ → ℤ) :
  (∀ n : ℕ, n > 0 → (Finset.range n).card = (Finset.image (λ i => a i % n) (Finset.range n)).card) →
  ∀ k : ℤ, ∃! i : ℕ, a i = k :=
sorry

end NUMINAMATH_CALUDE_sequence_completeness_l416_41655


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l416_41668

theorem least_positive_integer_with_remainder_one (n : ℕ) : 
  (n > 1) →
  (n % 3 = 1) →
  (n % 4 = 1) →
  (n % 5 = 1) →
  (n % 6 = 1) →
  (n % 7 = 1) →
  (n % 10 = 1) →
  (n % 11 = 1) →
  (∀ m : ℕ, m > 1 → 
    (m % 3 = 1) →
    (m % 4 = 1) →
    (m % 5 = 1) →
    (m % 6 = 1) →
    (m % 7 = 1) →
    (m % 10 = 1) →
    (m % 11 = 1) →
    (n ≤ m)) →
  n = 4621 :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l416_41668


namespace NUMINAMATH_CALUDE_correct_rounding_sum_l416_41689

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  (n + 50) / 100 * 100

theorem correct_rounding_sum : round_to_nearest_hundred (125 + 96) = 200 := by
  sorry

end NUMINAMATH_CALUDE_correct_rounding_sum_l416_41689


namespace NUMINAMATH_CALUDE_fixed_circle_theorem_l416_41606

noncomputable section

-- Define the hyperbola C
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / (3 * a^2) = 1

-- Define the foci F₁ and F₂
def F₁ (a : ℝ) : ℝ × ℝ := (-2, 0)
def F₂ (a : ℝ) : ℝ × ℝ := (2, 0)

-- Define the distance from F₂ to the asymptote
def distance_to_asymptote (a : ℝ) : ℝ := Real.sqrt 3

-- Define a line passing through the left vertex and not coinciding with x-axis
def line_through_left_vertex (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Define the intersection point B
def point_B (a k : ℝ) : ℝ × ℝ := ((3 + k^2) / (3 - k^2), 6 * k / (3 - k^2))

-- Define the intersection point P
def point_P (k : ℝ) : ℝ × ℝ := (1/2, k * 3/2)

-- Define the line parallel to PF₂ passing through F₁
def parallel_line (k : ℝ) (x : ℝ) : ℝ := -k * (x + 2)

-- Define the theorem
theorem fixed_circle_theorem (a : ℝ) (k : ℝ) :
  a > 0 →
  ∀ Q : ℝ × ℝ,
  (∃ x, Q.1 = x ∧ Q.2 = line_through_left_vertex k x) →
  (∃ x, Q.1 = x ∧ Q.2 = parallel_line k x) →
  (Q.1 - (F₂ a).1)^2 + (Q.2 - (F₂ a).2)^2 = 16 :=
by sorry

end

end NUMINAMATH_CALUDE_fixed_circle_theorem_l416_41606


namespace NUMINAMATH_CALUDE_luke_good_games_l416_41694

theorem luke_good_games (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (non_working_games : ℕ) :
  games_from_friend = 2 →
  games_from_garage_sale = 2 →
  non_working_games = 2 →
  games_from_friend + games_from_garage_sale - non_working_games = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_luke_good_games_l416_41694


namespace NUMINAMATH_CALUDE_factor_divisor_statements_l416_41650

theorem factor_divisor_statements :
  (∃ n : ℕ, 24 = 4 * n) ∧
  (∃ n : ℕ, 200 = 10 * n) ∧
  (¬ ∃ n : ℕ, 133 = 19 * n ∨ ∃ n : ℕ, 57 = 19 * n) ∧
  (∃ n : ℕ, 90 = 30 * n ∨ ∃ n : ℕ, 65 = 30 * n) ∧
  (¬ ∃ n : ℕ, 49 = 7 * n ∨ ∃ n : ℕ, 98 = 7 * n) :=
by sorry

end NUMINAMATH_CALUDE_factor_divisor_statements_l416_41650


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l416_41681

/-- The number of wheels on a car -/
def car_wheels : ℕ := 4

/-- The number of wheels on a bike -/
def bike_wheels : ℕ := 2

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 14

/-- The number of bikes in the parking lot -/
def num_bikes : ℕ := 5

/-- The total number of wheels in the parking lot -/
def total_wheels : ℕ := num_cars * car_wheels + num_bikes * bike_wheels

theorem parking_lot_wheels : total_wheels = 66 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_wheels_l416_41681


namespace NUMINAMATH_CALUDE_inheritance_interest_rate_proof_l416_41600

theorem inheritance_interest_rate_proof (inheritance : ℝ) (amount_first : ℝ) (rate_first : ℝ) (total_interest : ℝ) :
  inheritance = 12000 →
  amount_first = 5000 →
  rate_first = 0.06 →
  total_interest = 860 →
  let amount_second := inheritance - amount_first
  let interest_first := amount_first * rate_first
  let interest_second := total_interest - interest_first
  let rate_second := interest_second / amount_second
  rate_second = 0.08 := by sorry

end NUMINAMATH_CALUDE_inheritance_interest_rate_proof_l416_41600


namespace NUMINAMATH_CALUDE_volume_change_l416_41666

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Parallelepiped where
  length : ℝ
  breadth : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular parallelepiped -/
def volume (p : Parallelepiped) : ℝ := p.length * p.breadth * p.height

/-- Applies the given changes to the dimensions of a parallelepiped -/
def apply_changes (p : Parallelepiped) : Parallelepiped :=
  { length := p.length * 1.5,
    breadth := p.breadth * 0.7,
    height := p.height * 1.2 }

theorem volume_change (p : Parallelepiped) :
  volume (apply_changes p) = 1.26 * volume p := by
  sorry

end NUMINAMATH_CALUDE_volume_change_l416_41666


namespace NUMINAMATH_CALUDE_equation_equivalence_l416_41610

theorem equation_equivalence (x y : ℝ) 
  (eq1 : 2 * x^2 + 6 * x + 5 * y + 1 = 0) 
  (eq2 : 2 * x + y + 3 = 0) : 
  y^2 + 10 * y - 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l416_41610


namespace NUMINAMATH_CALUDE_movie_of_the_year_requirement_l416_41618

/-- The number of members in the cinematic academy -/
def academy_members : ℕ := 795

/-- The fraction of lists a film must appear on to be considered for "movie of the year" -/
def required_fraction : ℚ := 1 / 4

/-- The smallest number of lists a film can appear on to be considered for "movie of the year" -/
def min_lists : ℕ := 199

/-- Theorem stating the smallest number of lists a film must appear on -/
theorem movie_of_the_year_requirement :
  min_lists = ⌈(required_fraction * academy_members : ℚ)⌉ :=
sorry

end NUMINAMATH_CALUDE_movie_of_the_year_requirement_l416_41618


namespace NUMINAMATH_CALUDE_clothes_cost_l416_41615

def total_spending : ℕ := 10000

def adidas_cost : ℕ := 800

def nike_cost : ℕ := 2 * adidas_cost

def skechers_cost : ℕ := 4 * adidas_cost

def puma_cost : ℕ := nike_cost / 2

def total_sneakers_cost : ℕ := adidas_cost + nike_cost + skechers_cost + puma_cost

theorem clothes_cost : total_spending - total_sneakers_cost = 3600 := by
  sorry

end NUMINAMATH_CALUDE_clothes_cost_l416_41615


namespace NUMINAMATH_CALUDE_fifty_numbers_with_negative_products_l416_41639

theorem fifty_numbers_with_negative_products (total : Nat) (neg_products : Nat) 
  (h1 : total = 50) (h2 : neg_products = 500) : 
  ∃ (m n p : Nat), m + n + p = total ∧ m * p = neg_products ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifty_numbers_with_negative_products_l416_41639


namespace NUMINAMATH_CALUDE_increasing_function_condition_l416_41640

/-- The function f(x) = x^2 + a/x is increasing on (1, +∞) when 0 < a < 2 -/
theorem increasing_function_condition (a : ℝ) :
  (0 < a ∧ a < 2) →
  ∃ (f : ℝ → ℝ), (∀ x > 1, f x = x^2 + a/x) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (¬ ∀ a, (∃ (f : ℝ → ℝ), (∀ x > 1, f x = x^2 + a/x) ∧
    (∀ x y, 1 < x ∧ x < y → f x < f y)) → (0 < a ∧ a < 2)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l416_41640


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l416_41663

/-- Represents the ages of Arun and Deepak -/
structure Ages where
  arun : ℕ
  deepak : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of two natural numbers -/
def calculateRatio (a b : ℕ) : Ratio :=
  let gcd := Nat.gcd a b
  { numerator := a / gcd, denominator := b / gcd }

/-- Theorem stating the ratio of Arun's and Deepak's ages -/
theorem age_ratio_theorem (ages : Ages) : 
  ages.deepak = 42 → 
  ages.arun + 6 = 36 → 
  calculateRatio ages.arun ages.deepak = Ratio.mk 5 7 := by
  sorry

#check age_ratio_theorem

end NUMINAMATH_CALUDE_age_ratio_theorem_l416_41663


namespace NUMINAMATH_CALUDE_dog_eaten_cost_l416_41691

/-- Represents the cost of ingredients for a cake -/
structure CakeIngredients where
  flour : Float
  sugar : Float
  eggs : Float
  butter : Float

/-- Represents the cake and its consumption -/
structure Cake where
  ingredients : CakeIngredients
  totalSlices : Nat
  slicesEatenByMother : Nat

def totalCost (c : CakeIngredients) : Float :=
  c.flour + c.sugar + c.eggs + c.butter

def costPerSlice (cake : Cake) : Float :=
  totalCost cake.ingredients / cake.totalSlices.toFloat

def slicesEatenByDog (cake : Cake) : Nat :=
  cake.totalSlices - cake.slicesEatenByMother

theorem dog_eaten_cost (cake : Cake) 
  (h1 : cake.ingredients = { flour := 4, sugar := 2, eggs := 0.5, butter := 2.5 })
  (h2 : cake.totalSlices = 6)
  (h3 : cake.slicesEatenByMother = 2) :
  costPerSlice cake * (slicesEatenByDog cake).toFloat = 6 := by
  sorry

#check dog_eaten_cost

end NUMINAMATH_CALUDE_dog_eaten_cost_l416_41691


namespace NUMINAMATH_CALUDE_cos_triple_angle_l416_41620

theorem cos_triple_angle (θ : Real) (x : Real) (h : x = Real.cos θ) :
  Real.cos (3 * θ) = 4 * x^3 - 3 * x := by sorry

end NUMINAMATH_CALUDE_cos_triple_angle_l416_41620


namespace NUMINAMATH_CALUDE_multiplication_increase_l416_41686

theorem multiplication_increase (x : ℝ) : 18 * x = 18 + 198 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_increase_l416_41686


namespace NUMINAMATH_CALUDE_additional_combinations_l416_41657

theorem additional_combinations (original_set1 original_set3 : ℕ)
  (original_set2 original_set4 : ℕ)
  (added_set1 added_set3 : ℕ) :
  original_set1 = 4 →
  original_set2 = 2 →
  original_set3 = 3 →
  original_set4 = 3 →
  added_set1 = 2 →
  added_set3 = 1 →
  (original_set1 + added_set1) * original_set2 * (original_set3 + added_set3) * original_set4 -
  original_set1 * original_set2 * original_set3 * original_set4 = 72 := by
  sorry

#check additional_combinations

end NUMINAMATH_CALUDE_additional_combinations_l416_41657


namespace NUMINAMATH_CALUDE_davids_age_l416_41604

/-- Given that Yuan is 14 years old and twice David's age, prove that David is 7 years old. -/
theorem davids_age (yuan_age : ℕ) (david_age : ℕ) 
  (h1 : yuan_age = 14) 
  (h2 : yuan_age = 2 * david_age) : 
  david_age = 7 := by
  sorry

end NUMINAMATH_CALUDE_davids_age_l416_41604


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l416_41673

/-- Given a point P(-3, 1), its symmetric point with respect to the x-axis has coordinates (-3, -1) -/
theorem symmetric_point_wrt_x_axis :
  let P : ℝ × ℝ := (-3, 1)
  let symmetric_point := (P.1, -P.2)
  symmetric_point = (-3, -1) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l416_41673


namespace NUMINAMATH_CALUDE_stool_height_is_53_l416_41682

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height (ceiling_height floor_dip alice_height hat_height reach_above_head light_bulb_distance : ℝ) : ℝ :=
  ceiling_height * 100 - light_bulb_distance - (alice_height * 100 + hat_height + reach_above_head - floor_dip)

/-- Theorem stating that the stool height is 53 cm given the problem conditions -/
theorem stool_height_is_53 :
  stool_height 2.8 3 1.6 5 50 15 = 53 := by
  sorry

end NUMINAMATH_CALUDE_stool_height_is_53_l416_41682


namespace NUMINAMATH_CALUDE_sphere_in_cube_volume_ratio_l416_41660

theorem sphere_in_cube_volume_ratio (cube_side : ℝ) (h : cube_side = 8) :
  let sphere_volume := (4 / 3) * Real.pi * (cube_side / 2)^3
  let cube_volume := cube_side^3
  sphere_volume / cube_volume = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_sphere_in_cube_volume_ratio_l416_41660


namespace NUMINAMATH_CALUDE_only_81_satisfies_l416_41609

/-- A function that returns true if a number is two-digit -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that swaps the digits of a two-digit number -/
def swapDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The main theorem -/
theorem only_81_satisfies : ∃! n : ℕ, isTwoDigit n ∧ (swapDigits n)^2 = 4 * n :=
  sorry

end NUMINAMATH_CALUDE_only_81_satisfies_l416_41609


namespace NUMINAMATH_CALUDE_ball_probabilities_l416_41613

theorem ball_probabilities (total_balls : ℕ) (p_red p_black p_yellow : ℚ) :
  total_balls = 12 →
  p_red + p_black + p_yellow = 1 →
  p_red = 1/3 →
  p_black - p_yellow = 1/6 →
  p_black = 5/12 ∧ p_yellow = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l416_41613


namespace NUMINAMATH_CALUDE_marie_sold_700_reading_materials_l416_41633

/-- The number of magazines Marie sold -/
def magazines : ℕ := 425

/-- The number of newspapers Marie sold -/
def newspapers : ℕ := 275

/-- The total number of reading materials Marie sold -/
def total_reading_materials : ℕ := magazines + newspapers

/-- Proof that Marie sold 700 reading materials -/
theorem marie_sold_700_reading_materials : total_reading_materials = 700 := by
  sorry

end NUMINAMATH_CALUDE_marie_sold_700_reading_materials_l416_41633


namespace NUMINAMATH_CALUDE_fred_bought_two_tickets_l416_41656

def ticket_cost : ℚ := 592 / 100
def movie_cost : ℚ := 679 / 100
def paid_amount : ℚ := 20
def change_received : ℚ := 137 / 100

theorem fred_bought_two_tickets :
  (paid_amount - change_received - movie_cost) / ticket_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_fred_bought_two_tickets_l416_41656


namespace NUMINAMATH_CALUDE_min_value_theorem_l416_41692

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ∃ (m : ℝ), m = 3 ∧ ∀ x y, x > 0 → y > 0 → x + y = 4 → (y / x + 4 / y) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l416_41692


namespace NUMINAMATH_CALUDE_square_field_area_l416_41693

/-- The area of a square field with a diagonal of 20 meters is 200 square meters. -/
theorem square_field_area (diagonal : Real) (area : Real) :
  diagonal = 20 →
  area = diagonal^2 / 2 →
  area = 200 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l416_41693


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_equation_l416_41665

theorem arithmetic_geometric_mean_equation (a b : ℝ) :
  (a + b) / 2 = 10 ∧ Real.sqrt (a * b) = 10 →
  ∀ x, x^2 - 20*x + 100 = 0 ↔ x = a ∨ x = b :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_equation_l416_41665


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_properties_l416_41617

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The asymptote of a hyperbola -/
def asymptote (h : Hyperbola) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x

/-- The focus of the parabola y^2 = 16x -/
def parabola_focus : ℝ × ℝ := (4, 0)

/-- Theorem: If a hyperbola has the given properties, its equation is x^2/4 - y^2/12 = 1 -/
theorem hyperbola_equation_from_properties (h : Hyperbola) :
  (∃ x y : ℝ, asymptote h x y) →
  (∃ x y : ℝ, hyperbola_equation h x y ∧ (x, y) = parabola_focus) →
  (∀ x y : ℝ, hyperbola_equation h x y ↔ x^2/4 - y^2/12 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_properties_l416_41617


namespace NUMINAMATH_CALUDE_total_limbs_is_108_l416_41658

/-- The total number of legs, arms, and tentacles of Daniel's animals -/
def total_limbs : ℕ :=
  let horses := 2
  let dogs := 5
  let cats := 7
  let turtles := 3
  let goats := 1
  let snakes := 4
  let spiders := 2
  let birds := 3
  let starfish_arms := 5
  let octopus_tentacles := 6
  let three_legged_dog := 1

  horses * 4 + 
  dogs * 4 + 
  cats * 4 + 
  turtles * 4 + 
  goats * 4 + 
  snakes * 0 + 
  spiders * 8 + 
  birds * 2 + 
  starfish_arms + 
  octopus_tentacles + 
  three_legged_dog * 3

theorem total_limbs_is_108 : total_limbs = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_limbs_is_108_l416_41658


namespace NUMINAMATH_CALUDE_xanadu_license_plates_l416_41619

/-- The number of possible letters in each letter position of a Xanadu license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in each digit position of a Xanadu license plate. -/
def num_digits : ℕ := 10

/-- The total number of valid license plates in Xanadu. -/
def total_license_plates : ℕ := num_letters^4 * num_digits^2

/-- Theorem stating the total number of valid license plates in Xanadu. -/
theorem xanadu_license_plates : total_license_plates = 45697600 := by
  sorry

end NUMINAMATH_CALUDE_xanadu_license_plates_l416_41619


namespace NUMINAMATH_CALUDE_decimal_difference_l416_41680

/-- The value of the repeating decimal 0.737373... -/
def repeating_decimal : ℚ := 73 / 99

/-- The value of the terminating decimal 0.73 -/
def terminating_decimal : ℚ := 73 / 100

/-- The difference between the repeating decimal 0.737373... and the terminating decimal 0.73 -/
def difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference : difference = 73 / 9900 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l416_41680


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l416_41647

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 42 →
  3 * girls = 4 * boys →
  total_students = boys + girls →
  girls - boys = 6 := by
sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l416_41647


namespace NUMINAMATH_CALUDE_sixteen_is_counterexample_l416_41669

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def is_counterexample (n : Nat) : Prop :=
  ¬(is_prime n) ∧ (is_prime (n - 2) ∨ is_prime (n + 2))

theorem sixteen_is_counterexample : is_counterexample 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_is_counterexample_l416_41669


namespace NUMINAMATH_CALUDE_otimes_equation_solution_l416_41643

/-- Custom binary operator ⊗ -/
def otimes (a b : ℝ) : ℝ := -2 * a + b

/-- Theorem stating that if x ⊗ (-5) = 3, then x = -4 -/
theorem otimes_equation_solution (x : ℝ) (h : otimes x (-5) = 3) : x = -4 := by
  sorry

end NUMINAMATH_CALUDE_otimes_equation_solution_l416_41643


namespace NUMINAMATH_CALUDE_inverse_of_exponential_l416_41630

noncomputable def g (x : ℝ) : ℝ := 3^x

theorem inverse_of_exponential (f : ℝ → ℝ) :
  (∀ x, f (g x) = x ∧ g (f x) = x) → f 3 = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_exponential_l416_41630


namespace NUMINAMATH_CALUDE_equation_solution_l416_41687

theorem equation_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y + 4 = 2 * (Real.sqrt (2*x+1) + Real.sqrt (2*y+1))) : 
  x = 1 + Real.sqrt 2 ∧ y = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l416_41687


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l416_41601

theorem quadratic_inequality_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2)*x - k + 8 > 0) ↔ k > -2*Real.sqrt 7 ∧ k < 2*Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l416_41601


namespace NUMINAMATH_CALUDE_triangle_properties_l416_41627

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * (Real.cos (t.C / 2))^2 + t.c * (Real.cos (t.A / 2))^2 = 3 * t.b / 2) :
  -- Part I: Prove that a, b, c form an arithmetic sequence
  t.a + t.c = 2 * t.b ∧
  -- Part II: If angle B = 60° and b = 4, find the area of triangle ABC
  (t.B = π / 3 ∧ t.b = 4 → 
    t.a * t.c * Real.sin t.B / 2 = 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l416_41627


namespace NUMINAMATH_CALUDE_triangle_inequality_l416_41623

/-- For any triangle with sides a, b, c, semi-perimeter p, inradius r, and area S,
    where S = √(p(p-a)(p-b)(p-c)) and r = S/p, the following inequality holds:
    1/(p-a)² + 1/(p-b)² + 1/(p-c)² ≥ 1/r² -/
theorem triangle_inequality (a b c p r S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : p = (a + b + c) / 2)
  (h5 : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h6 : r = S / p) :
  1 / (p - a)^2 + 1 / (p - b)^2 + 1 / (p - c)^2 ≥ 1 / r^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l416_41623


namespace NUMINAMATH_CALUDE_periodic_trig_function_l416_41608

theorem periodic_trig_function (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 2015 = -1 → f 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_trig_function_l416_41608


namespace NUMINAMATH_CALUDE_insertion_sort_comparison_bounds_l416_41607

/-- Insertion sort comparison count bounds -/
theorem insertion_sort_comparison_bounds (n : ℕ) :
  ∀ (list : List ℕ), list.length = n →
  ∃ (comparisons : ℕ),
    (n - 1 : ℝ) ≤ comparisons ∧ comparisons ≤ (n * (n - 1) : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_insertion_sort_comparison_bounds_l416_41607


namespace NUMINAMATH_CALUDE_star_equation_solution_l416_41626

def star (a b : ℝ) : ℝ := a * (a + b) + b

theorem star_equation_solution :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧ 
  star a₁ 2.5 = 28.5 ∧ 
  star a₂ 2.5 = 28.5 ∧ 
  a₁ = 4 ∧ 
  a₂ = -13/2 :=
sorry

end NUMINAMATH_CALUDE_star_equation_solution_l416_41626


namespace NUMINAMATH_CALUDE_sports_camp_coach_age_l416_41698

theorem sports_camp_coach_age (total_members : ℕ) (total_average_age : ℕ)
  (num_girls num_boys num_coaches : ℕ) (girls_average_age boys_average_age : ℕ)
  (h1 : total_members = 50)
  (h2 : total_average_age = 20)
  (h3 : num_girls = 30)
  (h4 : num_boys = 15)
  (h5 : num_coaches = 5)
  (h6 : girls_average_age = 18)
  (h7 : boys_average_age = 19)
  (h8 : total_members = num_girls + num_boys + num_coaches) :
  (total_members * total_average_age - num_girls * girls_average_age - num_boys * boys_average_age) / num_coaches = 35 := by
sorry


end NUMINAMATH_CALUDE_sports_camp_coach_age_l416_41698


namespace NUMINAMATH_CALUDE_green_pill_cost_l416_41634

/-- The cost of Al's pills for three weeks of treatment --/
def total_cost : ℚ := 1092

/-- The number of days in the treatment period --/
def treatment_days : ℕ := 21

/-- The number of times Al takes a blue pill --/
def blue_pill_count : ℕ := 10

/-- The cost difference between a green pill and a pink pill --/
def green_pink_diff : ℚ := 2

/-- The cost of a pink pill --/
def pink_cost : ℚ := 1050 / 62

/-- The cost of a green pill --/
def green_cost : ℚ := pink_cost + green_pink_diff

/-- Theorem stating the cost of a green pill --/
theorem green_pill_cost : green_cost = 587 / 31 := by sorry

end NUMINAMATH_CALUDE_green_pill_cost_l416_41634


namespace NUMINAMATH_CALUDE_car_distance_theorem_l416_41685

/-- Calculates the total distance traveled by a car with increasing speed over a given number of hours -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  (hours * (2 * initialSpeed + (hours - 1) * speedIncrease)) / 2

/-- Theorem stating that a car with given initial speed and speed increase travels a specific distance in 12 hours -/
theorem car_distance_theorem (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) :
  initialSpeed = 40 ∧ speedIncrease = 2 ∧ hours = 12 →
  totalDistance initialSpeed speedIncrease hours = 606 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l416_41685


namespace NUMINAMATH_CALUDE_sin_2x_equals_plus_minus_one_l416_41638

/-- Given vectors a and b, if a is a non-zero scalar multiple of b, then sin(2x) = ±1 -/
theorem sin_2x_equals_plus_minus_one (x : ℝ) :
  let a : ℝ × ℝ := (Real.cos x, -Real.sin x)
  let b : ℝ × ℝ := (-Real.cos (π/2 - x), Real.cos x)
  ∀ t : ℝ, t ≠ 0 → a = t • b → Real.sin (2*x) = 1 ∨ Real.sin (2*x) = -1 :=
by sorry

end NUMINAMATH_CALUDE_sin_2x_equals_plus_minus_one_l416_41638


namespace NUMINAMATH_CALUDE_consecutive_four_plus_one_is_square_l416_41641

theorem consecutive_four_plus_one_is_square (a : ℕ) (h : a ≥ 1) :
  a * (a + 1) * (a + 2) * (a + 3) + 1 = (a^2 + 3*a + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_four_plus_one_is_square_l416_41641


namespace NUMINAMATH_CALUDE_inequality_proof_l416_41635

theorem inequality_proof (a : ℝ) : 2 * a^4 + 2 * a^2 - 1 ≥ (3/2) * (a^2 + a - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l416_41635


namespace NUMINAMATH_CALUDE_units_digit_of_30_factorial_l416_41622

theorem units_digit_of_30_factorial (n : ℕ) : n = 30 → n.factorial % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_30_factorial_l416_41622


namespace NUMINAMATH_CALUDE_squared_difference_of_quadratic_roots_l416_41696

theorem squared_difference_of_quadratic_roots : ∀ p q : ℝ,
  (2 * p^2 + 7 * p - 15 = 0) →
  (2 * q^2 + 7 * q - 15 = 0) →
  (p - q)^2 = 169 / 4 := by
  sorry

end NUMINAMATH_CALUDE_squared_difference_of_quadratic_roots_l416_41696


namespace NUMINAMATH_CALUDE_set_operations_and_range_l416_41661

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

-- State the theorem
theorem set_operations_and_range :
  (∃ (a : ℝ),
    (A ∩ B = {x | 2 ≤ x ∧ x < 3}) ∧
    ((U \ A) ∪ B = {x | x < -1 ∨ x ≥ 2}) ∧
    (B ∪ C a = C a → a > 4)) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l416_41661


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_47_l416_41637

theorem gcd_of_powers_of_47 : Nat.gcd (47^5 + 1) (47^5 + 47^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_47_l416_41637
