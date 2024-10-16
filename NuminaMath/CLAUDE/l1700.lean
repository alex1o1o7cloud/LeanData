import Mathlib

namespace NUMINAMATH_CALUDE_simplify_fraction_l1700_170058

theorem simplify_fraction : (150 : ℚ) / 6000 * 75 = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1700_170058


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l1700_170071

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (α : Plane) :
  m ≠ n →
  parallel n m →
  perpendicular n α →
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l1700_170071


namespace NUMINAMATH_CALUDE_system_solution_l1700_170015

theorem system_solution :
  ∃ (x y : ℚ), (7 * x = -5 - 3 * y) ∧ (4 * x = 5 * y - 34) :=
by
  use (-127/47), (218/47)
  sorry

end NUMINAMATH_CALUDE_system_solution_l1700_170015


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l1700_170018

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line equation passing through the intersection points of two circles --/
def intersectionLine (c1 c2 : Circle) : ℝ → ℝ → Prop :=
  fun x y => x + y = 23/8

theorem intersection_line_of_circles :
  let c1 : Circle := { center := (0, 0), radius := 5 }
  let c2 : Circle := { center := (4, 4), radius := 3 }
  ∀ x y : ℝ, (x^2 + y^2 = c1.radius^2) ∧ ((x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2) →
    intersectionLine c1 c2 x y := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l1700_170018


namespace NUMINAMATH_CALUDE_problem_solution_l1700_170068

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x, x^2 + m*x + 1 ≠ 0

def q (m : ℝ) : Prop := m > 0

-- Define the condition that either p or q is true, but not both
def condition (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬(p m ∧ q m)

-- Define the set of m values that satisfy the condition
def solution_set : Set ℝ := {m | condition m}

-- Theorem statement
theorem problem_solution : 
  solution_set = {m | m ∈ Set.Ioo (-2) 0 ∪ Set.Ioi 2} :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1700_170068


namespace NUMINAMATH_CALUDE_a_in_open_interval_l1700_170043

/-- The set A defined as {x | |x-a| ≤ 1} -/
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}

/-- The set B defined as {x | x^2-5x+4 ≥ 0} -/
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

/-- Theorem stating that if A intersect B is empty, then a is in the open interval (2, 3) -/
theorem a_in_open_interval (a : ℝ) (h : A a ∩ B = ∅) : a ∈ Set.Ioo 2 3 := by
  sorry

#check a_in_open_interval

end NUMINAMATH_CALUDE_a_in_open_interval_l1700_170043


namespace NUMINAMATH_CALUDE_books_in_bargain_bin_l1700_170020

theorem books_in_bargain_bin 
  (initial_books : ℕ) 
  (books_sold : ℕ) 
  (books_added : ℕ) 
  (h1 : initial_books ≥ books_sold) : 
  initial_books - books_sold + books_added = 
    initial_books + books_added - books_sold :=
by sorry

end NUMINAMATH_CALUDE_books_in_bargain_bin_l1700_170020


namespace NUMINAMATH_CALUDE_math_problem_time_l1700_170030

-- Define the number of problems for each subject
def math_problems : ℕ := 15
def social_studies_problems : ℕ := 6
def science_problems : ℕ := 10

-- Define the time taken for social studies and science problems (in minutes)
def social_studies_time : ℚ := 30 / 60
def science_time : ℚ := 1.5

-- Define the total time taken for all homework (in minutes)
def total_time : ℚ := 48

-- Theorem to prove
theorem math_problem_time :
  ∃ (x : ℚ), 
    x * math_problems + 
    social_studies_time * social_studies_problems + 
    science_time * science_problems = total_time ∧
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_math_problem_time_l1700_170030


namespace NUMINAMATH_CALUDE_remainder_theorem_l1700_170032

theorem remainder_theorem (x : ℕ+) (h : (7 * x.val) % 29 = 1) :
  (13 + x.val) % 29 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1700_170032


namespace NUMINAMATH_CALUDE_tank_fill_time_xy_l1700_170019

/-- Represents the time (in hours) to fill a tank given specific valve configurations -/
structure TankFillTime where
  all : ℝ
  xz : ℝ
  yz : ℝ

/-- Proves that given specific fill times for different valve configurations, 
    the time to fill the tank with only valves X and Y open is 2.4 hours -/
theorem tank_fill_time_xy (t : TankFillTime) 
  (h_all : t.all = 2)
  (h_xz : t.xz = 3)
  (h_yz : t.yz = 4) :
  1 / (1 / t.all - 1 / t.yz) + 1 / (1 / t.all - 1 / t.xz) = 2.4 := by
  sorry

#check tank_fill_time_xy

end NUMINAMATH_CALUDE_tank_fill_time_xy_l1700_170019


namespace NUMINAMATH_CALUDE_dandelion_game_strategy_l1700_170041

/-- The dandelion mowing game -/
def has_winning_strategy (m n : ℕ+) : Prop :=
  (m.val + n.val) % 2 = 1 ∨ min m.val n.val = 1

theorem dandelion_game_strategy (m n : ℕ+) :
  has_winning_strategy m n ↔ (m.val + n.val) % 2 = 1 ∨ min m.val n.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_game_strategy_l1700_170041


namespace NUMINAMATH_CALUDE_power_zero_minus_one_equals_zero_l1700_170066

theorem power_zero_minus_one_equals_zero : 2^0 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_minus_one_equals_zero_l1700_170066


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l1700_170036

def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (6, -5)
  let reflected_center : ℝ × ℝ := reflect_over_y_eq_x original_center
  reflected_center = (-5, 6) := by sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l1700_170036


namespace NUMINAMATH_CALUDE_willie_initial_stickers_l1700_170013

/-- The number of stickers Willie gave to Emily -/
def stickers_given : ℕ := 7

/-- The number of stickers Willie had left after giving some to Emily -/
def stickers_left : ℕ := 29

/-- The initial number of stickers Willie had -/
def initial_stickers : ℕ := stickers_given + stickers_left

theorem willie_initial_stickers : initial_stickers = 36 := by
  sorry

end NUMINAMATH_CALUDE_willie_initial_stickers_l1700_170013


namespace NUMINAMATH_CALUDE_sarah_weed_pulling_l1700_170064

def tuesday_weeds : ℕ := 25

def wednesday_weeds : ℕ := 3 * tuesday_weeds

def thursday_weeds : ℕ := wednesday_weeds / 5

def friday_weeds : ℕ := thursday_weeds - 10

def total_weeds : ℕ := tuesday_weeds + wednesday_weeds + thursday_weeds + friday_weeds

theorem sarah_weed_pulling :
  total_weeds = 120 :=
sorry

end NUMINAMATH_CALUDE_sarah_weed_pulling_l1700_170064


namespace NUMINAMATH_CALUDE_division_problem_l1700_170085

theorem division_problem (N D Q R : ℕ) : 
  D = 5 → Q = 4 → R = 3 → N = D * Q + R → N = 23 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1700_170085


namespace NUMINAMATH_CALUDE_sum_less_than_addends_implies_negative_l1700_170008

theorem sum_less_than_addends_implies_negative (a b : ℝ) :
  (a + b < a ∧ a + b < b) → (a < 0 ∧ b < 0) := by sorry

end NUMINAMATH_CALUDE_sum_less_than_addends_implies_negative_l1700_170008


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1700_170003

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

-- Theorem stating the center and radius of the circle
theorem circle_center_and_radius :
  (∃ (x₀ y₀ r : ℝ), (∀ x y : ℝ, C x y ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧ x₀ = 2 ∧ y₀ = -1 ∧ r = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1700_170003


namespace NUMINAMATH_CALUDE_count_pairs_eq_32_l1700_170087

/-- The number of pairs of positive integers (m,n) satisfying m^2 + n^2 < 50 -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + p.2^2 < 50) (Finset.product (Finset.range 50) (Finset.range 50))).card

theorem count_pairs_eq_32 : count_pairs = 32 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_32_l1700_170087


namespace NUMINAMATH_CALUDE_solve_equation_l1700_170090

theorem solve_equation (a : ℚ) (h : a + a/4 - 1/2 = 10/5) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1700_170090


namespace NUMINAMATH_CALUDE_coordinate_axes_characterization_l1700_170014

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set of all points on the coordinate axes -/
def CoordinateAxesPoints : Set Point :=
  {p : Point | p.x * p.y = 0}

/-- Predicate to check if a point is on a coordinate axis -/
def IsOnAxis (p : Point) : Prop :=
  p.x = 0 ∨ p.y = 0

theorem coordinate_axes_characterization :
  ∀ p : Point, p ∈ CoordinateAxesPoints ↔ IsOnAxis p :=
by sorry

end NUMINAMATH_CALUDE_coordinate_axes_characterization_l1700_170014


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1700_170093

/-- A line passing through a point and perpendicular to another line -/
def perpendicular_line (x₀ y₀ a b c : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ 
    (∀ x y : ℝ, (y - y₀ = k * (x - x₀)) ↔ (a * x + b * y + c = 0)) ∧
    k * (a / b) = -1

theorem perpendicular_line_equation :
  perpendicular_line 1 3 2 (-5) 1 →
  ∀ x y : ℝ, 5 * x + 2 * y - 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1700_170093


namespace NUMINAMATH_CALUDE_f_one_lower_bound_l1700_170089

/-- A function f(x) = 4x^2 - mx + 5 that is increasing on [-2, +∞) -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- The property that f is increasing on [-2, +∞) -/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, -2 ≤ x → x < y → f m x < f m y

theorem f_one_lower_bound (m : ℝ) (h : is_increasing_on_interval m) : f m 1 ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_f_one_lower_bound_l1700_170089


namespace NUMINAMATH_CALUDE_boxed_divisibility_boxed_27_divisibility_l1700_170037

def boxed (n : ℕ+) : ℕ := (10^n.val - 1) / 9

theorem boxed_divisibility (m : ℕ) :
  ∃ k : ℕ, boxed (3^m : ℕ+) = k * 3^m ∧ 
  ∀ l : ℕ, boxed (3^m : ℕ+) ≠ l * 3^(m+1) :=
sorry

theorem boxed_27_divisibility (n : ℕ+) :
  27 ∣ n ↔ 27 ∣ boxed n :=
sorry

end NUMINAMATH_CALUDE_boxed_divisibility_boxed_27_divisibility_l1700_170037


namespace NUMINAMATH_CALUDE_smallest_valid_n_l1700_170039

def is_valid (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  let tens := n / 10
  let ones := n % 10
  2 * n = 10 * ones + tens + 3

theorem smallest_valid_n :
  is_valid 12 ∧ ∀ m : ℕ, is_valid m → 12 ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l1700_170039


namespace NUMINAMATH_CALUDE_nth_equation_pattern_l1700_170045

theorem nth_equation_pattern (n : ℕ) :
  (-n : ℚ) * (n / (n + 1)) = -n + (n / (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_pattern_l1700_170045


namespace NUMINAMATH_CALUDE_inequality_not_always_preserved_l1700_170033

theorem inequality_not_always_preserved (a b : ℝ) (h : a < b) :
  ∃ m : ℝ, ¬(m * a > m * b) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_preserved_l1700_170033


namespace NUMINAMATH_CALUDE_x_squared_is_quadratic_l1700_170026

/-- A quadratic equation in one variable is of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x² is a quadratic equation in one variable -/
theorem x_squared_is_quadratic : is_quadratic_equation (λ x : ℝ => x^2) := by
  sorry

#check x_squared_is_quadratic

end NUMINAMATH_CALUDE_x_squared_is_quadratic_l1700_170026


namespace NUMINAMATH_CALUDE_roger_tray_collection_l1700_170076

/-- The number of trips required to collect trays -/
def numTrips (capacity traysTable1 traysTable2 : ℕ) : ℕ :=
  (traysTable1 + traysTable2 + capacity - 1) / capacity

theorem roger_tray_collection (capacity traysTable1 traysTable2 : ℕ) 
  (h1 : capacity = 4) 
  (h2 : traysTable1 = 10) 
  (h3 : traysTable2 = 2) : 
  numTrips capacity traysTable1 traysTable2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_roger_tray_collection_l1700_170076


namespace NUMINAMATH_CALUDE_exam_pass_probability_l1700_170028

/-- The probability of passing an exam given the following conditions:
  - There are 5 total questions
  - The candidate is familiar with 3 questions
  - The candidate randomly selects 3 questions to answer
  - The candidate needs to answer 2 questions correctly to pass
-/
theorem exam_pass_probability :
  let total_questions : ℕ := 5
  let familiar_questions : ℕ := 3
  let selected_questions : ℕ := 3
  let required_correct : ℕ := 2
  let pass_probability : ℚ := 7 / 10
  (Nat.choose familiar_questions selected_questions +
   Nat.choose familiar_questions (selected_questions - 1) * Nat.choose (total_questions - familiar_questions) 1) /
  Nat.choose total_questions selected_questions = pass_probability :=
by sorry

end NUMINAMATH_CALUDE_exam_pass_probability_l1700_170028


namespace NUMINAMATH_CALUDE_curve_fixed_point_l1700_170049

/-- The curve C: x^2 + y^2 + 2kx + (4k+10)y + 10k + 20 = 0 passes through the fixed point (1, -3) for all k ≠ -1 -/
theorem curve_fixed_point (k : ℝ) (h : k ≠ -1) :
  let C (x y : ℝ) := x^2 + y^2 + 2*k*x + (4*k+10)*y + 10*k + 20
  C 1 (-3) = 0 := by sorry

end NUMINAMATH_CALUDE_curve_fixed_point_l1700_170049


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_implies_b_bound_l1700_170050

theorem quadratic_always_real_roots_implies_b_bound (b : ℝ) :
  (∀ a : ℝ, ∃ x : ℝ, x^2 - 2*a*x - a + 2*b = 0) →
  b ≤ -1/8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_implies_b_bound_l1700_170050


namespace NUMINAMATH_CALUDE_complex_24th_power_of_cube_root_of_unity_l1700_170012

theorem complex_24th_power_of_cube_root_of_unity (z : ℂ) : z = (1 + Complex.I * Real.sqrt 3) / 2 → z^24 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_24th_power_of_cube_root_of_unity_l1700_170012


namespace NUMINAMATH_CALUDE_hare_tortoise_race_l1700_170022

theorem hare_tortoise_race (v : ℝ) (x : ℝ) (y : ℝ) (h_v_pos : v > 0) :
  v > 0 ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  x + y = 25 ∧ 
  x^2 + 5^2 = y^2 →
  y = 13 :=
by sorry

end NUMINAMATH_CALUDE_hare_tortoise_race_l1700_170022


namespace NUMINAMATH_CALUDE_equation_solution_l1700_170004

theorem equation_solution :
  ∃ x : ℚ, x + 5/6 = 7/18 - 2/9 ∧ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1700_170004


namespace NUMINAMATH_CALUDE_same_side_inequality_l1700_170006

/-- Given that point P (a, b) and point Q (1, 2) are on the same side of the line 3x + 2y - 8 = 0,
    prove that 3a + 2b - 8 > 0 -/
theorem same_side_inequality (a b : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (3*a + 2*b - 8) * (3*1 + 2*2 - 8) = k * (3*a + 2*b - 8)^2) →
  3*a + 2*b - 8 > 0 := by
sorry

end NUMINAMATH_CALUDE_same_side_inequality_l1700_170006


namespace NUMINAMATH_CALUDE_five_people_seven_chairs_l1700_170047

/-- The number of ways to arrange n people in r chairs -/
def arrangements (n r : ℕ) : ℕ := sorry

/-- The number of ways to arrange n people in r chairs,
    with one person restricted to m specific chairs -/
def restricted_arrangements (n r m : ℕ) : ℕ := sorry

/-- Theorem: Five people can be arranged in a row of seven chairs in 2160 ways,
    given that the oldest must sit in one of the three chairs at the end of the row -/
theorem five_people_seven_chairs : restricted_arrangements 5 7 3 = 2160 := by sorry

end NUMINAMATH_CALUDE_five_people_seven_chairs_l1700_170047


namespace NUMINAMATH_CALUDE_divisor_expression_l1700_170055

theorem divisor_expression (N D Y : ℕ) : 
  N = 45 * D + 13 → N = 6 * Y + 4 → D = (2 * Y - 3) / 15 := by
  sorry

end NUMINAMATH_CALUDE_divisor_expression_l1700_170055


namespace NUMINAMATH_CALUDE_book_chapters_l1700_170031

/-- A problem about determining the number of chapters in a book based on reading rate. -/
theorem book_chapters (chapters_read : ℕ) (hours_read : ℕ) (hours_remaining : ℕ) : 
  chapters_read = 2 →
  hours_read = 3 →
  hours_remaining = 9 →
  ∃ (total_chapters : ℕ), 
    total_chapters = chapters_read + (chapters_read * hours_remaining / hours_read) ∧
    total_chapters = 8 := by
  sorry


end NUMINAMATH_CALUDE_book_chapters_l1700_170031


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_one_l1700_170070

theorem no_solution_iff_n_eq_neg_one (n : ℝ) : 
  (∀ x y z : ℝ, nx + y = 1 ∧ ny + z = 1 ∧ x + nz = 1 → False) ↔ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_one_l1700_170070


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1700_170072

/-- The eccentricity of the hyperbola x^2 - y^2 = 1 is √2 -/
theorem hyperbola_eccentricity :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2 = 1
  ∃ e : ℝ, e = Real.sqrt 2 ∧ ∀ x y : ℝ, h x y → e = (Real.sqrt (x^2 + y^2)) / (Real.sqrt (x^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1700_170072


namespace NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l1700_170095

theorem sqrt_50_plus_sqrt_32 : Real.sqrt 50 + Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l1700_170095


namespace NUMINAMATH_CALUDE_remainder_sum_l1700_170001

theorem remainder_sum (n : ℤ) (h : n % 18 = 11) : (n % 2 + n % 9 = 3) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1700_170001


namespace NUMINAMATH_CALUDE_max_shaded_area_trapezoid_l1700_170009

/-- Given a trapezoid ABCD with bases of length a and b, and area 1,
    the maximum area of the shaded region formed by moving points on the bases is ab / (a+b)^2 -/
theorem max_shaded_area_trapezoid (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let trapezoid_area : ℝ := 1
  ∃ (max_area : ℝ), max_area = a * b / (a + b)^2 ∧
    ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b →
      (x * y / ((a + b) * (x + y)) + (a - x) * (b - y) / ((a + b) * (a + b - x - y))) / (a + b) ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_max_shaded_area_trapezoid_l1700_170009


namespace NUMINAMATH_CALUDE_smallest_multiple_45_60_not_25_l1700_170074

theorem smallest_multiple_45_60_not_25 : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (45 ∣ n) ∧ 
  (60 ∣ n) ∧ 
  ¬(25 ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ (45 ∣ m) ∧ (60 ∣ m) ∧ ¬(25 ∣ m) → n ≤ m) ∧
  n = 180 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_45_60_not_25_l1700_170074


namespace NUMINAMATH_CALUDE_function_range_l1700_170025

def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem function_range :
  {y | ∃ x ∈ Set.Ioo (-1) 2, f x = y} = Set.Icc (-4) 0 := by sorry

end NUMINAMATH_CALUDE_function_range_l1700_170025


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1700_170094

/-- A circle with equation x^2 + y^2 = n is tangent to the line x + y + 1 = 0 if and only if n = 1/2 -/
theorem circle_tangent_to_line (n : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = n ∧ x + y + 1 = 0 ∧ 
    ∀ (x' y' : ℝ), x'^2 + y'^2 = n → x' + y' + 1 ≠ 0 ∨ (x' = x ∧ y' = y)) ↔ 
  n = 1/2 := by
sorry


end NUMINAMATH_CALUDE_circle_tangent_to_line_l1700_170094


namespace NUMINAMATH_CALUDE_quadratic_polynomial_functional_equation_l1700_170075

theorem quadratic_polynomial_functional_equation 
  (P : ℝ → ℝ) 
  (h_quadratic : ∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) 
  (f : ℝ → ℝ) 
  (h_add : ∀ x y, f (x + y) = f x + f y) 
  (h_poly : ∀ x, f (P x) = f x) : 
  ∀ x, f x = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_functional_equation_l1700_170075


namespace NUMINAMATH_CALUDE_lauras_workout_speed_l1700_170079

theorem lauras_workout_speed :
  ∃! x : ℝ, x > 0 ∧ (25 / (3 * x + 2)) + (8 / x) = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_lauras_workout_speed_l1700_170079


namespace NUMINAMATH_CALUDE_range_of_a_l1700_170027

theorem range_of_a (x a : ℝ) : 
  (∀ x, -x^2 + 5*x - 6 > 0 → |x - a| < 4) ∧ 
  (∃ x, |x - a| < 4 ∧ -x^2 + 5*x - 6 ≤ 0) →
  a ∈ Set.Icc (-1 : ℝ) 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1700_170027


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1700_170056

theorem smallest_n_congruence :
  ∃ (n : ℕ), n > 0 ∧ 5 * n ≡ 1723 [MOD 26] ∧
  ∀ (m : ℕ), m > 0 ∧ 5 * m ≡ 1723 [MOD 26] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1700_170056


namespace NUMINAMATH_CALUDE_linear_regression_coefficient_l1700_170040

theorem linear_regression_coefficient
  (x : Fin 4 → ℝ)
  (y : Fin 4 → ℝ)
  (h_x : x = ![6, 8, 10, 12])
  (h_y : y = ![6, 5, 3, 2])
  (a : ℝ)
  (h_reg : ∀ i, y i = a * x i + 10.3) :
  a = -0.7 := by
sorry

end NUMINAMATH_CALUDE_linear_regression_coefficient_l1700_170040


namespace NUMINAMATH_CALUDE_irreducible_fractions_exist_l1700_170073

theorem irreducible_fractions_exist : ∃ (a b : ℕ), 
  Nat.gcd a b = 1 ∧ Nat.gcd (a + 1) b = 1 ∧ Nat.gcd (a + 1) (b + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fractions_exist_l1700_170073


namespace NUMINAMATH_CALUDE_integer_root_characterization_l1700_170060

def polynomial (x b : ℤ) : ℤ := x^4 + 4*x^3 + 4*x^2 + b*x + 12

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_characterization (b : ℤ) :
  has_integer_root b ↔ b ∈ ({-38, -21, -2, 10, 13, 34} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_root_characterization_l1700_170060


namespace NUMINAMATH_CALUDE_investment_rate_proof_l1700_170096

def total_investment : ℝ := 3000
def high_interest_amount : ℝ := 800
def high_interest_rate : ℝ := 0.1
def total_interest : ℝ := 256

def remaining_investment : ℝ := total_investment - high_interest_amount
def high_interest : ℝ := high_interest_amount * high_interest_rate
def remaining_interest : ℝ := total_interest - high_interest

theorem investment_rate_proof :
  remaining_interest / remaining_investment = 0.08 :=
sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l1700_170096


namespace NUMINAMATH_CALUDE_simplify_expression_l1700_170069

theorem simplify_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
  a * (1 / b + 1 / c) + b * (1 / a + 1 / c) + c * (1 / a + 1 / b) = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1700_170069


namespace NUMINAMATH_CALUDE_min_sum_squares_l1700_170088

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 2*x₂ + 3*x₃ = 120) : 
  ∀ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 → y₁ + 2*y₂ + 3*y₃ = 120 → 
  x₁^2 + x₂^2 + x₃^2 ≤ y₁^2 + y₂^2 + y₃^2 ∧ 
  ∃ x₁' x₂' x₃' : ℝ, x₁'^2 + x₂'^2 + x₃'^2 = 1400 ∧ 
                    x₁' > 0 ∧ x₂' > 0 ∧ x₃' > 0 ∧ 
                    x₁' + 2*x₂' + 3*x₃' = 120 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1700_170088


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1700_170053

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := sorry

/-- The common ratio of the geometric sequence -/
def q (a : ℕ → ℝ) : ℝ := sorry

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  S 4 a = -5 →
  S 6 a = 21 * S 2 a →
  S 8 a = -85 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1700_170053


namespace NUMINAMATH_CALUDE_equal_egg_distribution_l1700_170007

theorem equal_egg_distribution (total_eggs : ℕ) (num_siblings : ℕ) : 
  total_eggs = 2 * 12 → 
  num_siblings = 3 → 
  (total_eggs / (num_siblings + 1) : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_equal_egg_distribution_l1700_170007


namespace NUMINAMATH_CALUDE_b_completes_job_in_20_days_l1700_170052

/-- The number of days it takes A to complete the job -/
def days_A : ℝ := 15

/-- The number of days A and B work together -/
def days_together : ℝ := 5

/-- The fraction of the job left after A and B work together -/
def fraction_left : ℝ := 0.41666666666666663

/-- The number of days it takes B to complete the job -/
def days_B : ℝ := 20

theorem b_completes_job_in_20_days :
  (days_together * (1 / days_A + 1 / days_B) = 1 - fraction_left) ∧
  (days_B = 20) := by sorry

end NUMINAMATH_CALUDE_b_completes_job_in_20_days_l1700_170052


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt_40_l1700_170005

theorem closest_integer_to_sqrt_40 :
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |m - Real.sqrt 40| ≥ |n - Real.sqrt 40| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt_40_l1700_170005


namespace NUMINAMATH_CALUDE_guessing_game_l1700_170046

theorem guessing_game (G C : ℕ) (h1 : G = 33) (h2 : 3 * G = 2 * C - 3) : C = 51 := by
  sorry

end NUMINAMATH_CALUDE_guessing_game_l1700_170046


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1700_170081

/-- Given Ronaldo's current age and the future age ratio, prove the past age ratio -/
theorem age_ratio_proof (ronaldo_current_age : ℕ) (future_ratio : ℚ) : 
  ronaldo_current_age = 36 → 
  future_ratio = 7 / 8 → 
  ∃ (roonie_current_age : ℕ), 
    (roonie_current_age + 4 : ℚ) / (ronaldo_current_age + 4 : ℚ) = future_ratio ∧ 
    (roonie_current_age - 1 : ℚ) / (ronaldo_current_age - 1 : ℚ) = 6 / 7 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1700_170081


namespace NUMINAMATH_CALUDE_leap_year_classification_l1700_170065

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

theorem leap_year_classification :
  let leap_years : Set ℕ := {1992, 2040}
  let common_years : Set ℕ := {1800, 1994}
  (∀ y ∈ leap_years, is_leap_year y) ∧
  (∀ y ∈ common_years, ¬is_leap_year y) ∧
  (leap_years ∪ common_years = {1800, 1992, 1994, 2040}) :=
by sorry

end NUMINAMATH_CALUDE_leap_year_classification_l1700_170065


namespace NUMINAMATH_CALUDE_roots_sum_absolute_values_l1700_170035

theorem roots_sum_absolute_values (m : ℤ) (a b c : ℤ) : 
  (∀ x : ℤ, x^3 - 2013*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  abs a + abs b + abs c = 94 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_absolute_values_l1700_170035


namespace NUMINAMATH_CALUDE_poison_frog_count_l1700_170091

theorem poison_frog_count (total : ℕ) (tree : ℕ) (wood : ℕ) (poison : ℕ) :
  total = 78 →
  tree = 55 →
  wood = 13 →
  poison = total - (tree + wood) →
  poison = 10 := by
  sorry

end NUMINAMATH_CALUDE_poison_frog_count_l1700_170091


namespace NUMINAMATH_CALUDE_high_school_population_change_l1700_170057

/-- Represents the number of students in a high school --/
structure SchoolPopulation where
  boys : ℕ
  girls : ℕ

/-- Represents the ratio of boys to girls --/
structure Ratio where
  boys : ℕ
  girls : ℕ

def SchoolPopulation.ratio (pop : SchoolPopulation) : Ratio :=
  { boys := pop.boys, girls := pop.girls }

theorem high_school_population_change 
  (initial_ratio : Ratio)
  (final_ratio : Ratio)
  (boys_left : ℕ)
  (girls_left : ℕ)
  (h1 : initial_ratio.boys = 3 ∧ initial_ratio.girls = 4)
  (h2 : final_ratio.boys = 4 ∧ final_ratio.girls = 5)
  (h3 : boys_left = 10)
  (h4 : girls_left = 20)
  (h5 : girls_left = 2 * boys_left) :
  ∃ (initial_pop : SchoolPopulation),
    initial_pop.ratio = initial_ratio ∧
    initial_pop.boys = 90 ∧
    let final_pop : SchoolPopulation :=
      { boys := initial_pop.boys - boys_left,
        girls := initial_pop.girls - girls_left }
    final_pop.ratio = final_ratio :=
  sorry


end NUMINAMATH_CALUDE_high_school_population_change_l1700_170057


namespace NUMINAMATH_CALUDE_travel_methods_count_l1700_170054

/-- The number of transportation options from Shijiazhuang to Qingdao -/
def shijiazhuang_to_qingdao : Nat := 3

/-- The number of transportation options from Qingdao to Guangzhou -/
def qingdao_to_guangzhou : Nat := 4

/-- The total number of travel methods for the entire journey -/
def total_travel_methods : Nat := shijiazhuang_to_qingdao * qingdao_to_guangzhou

theorem travel_methods_count : total_travel_methods = 12 := by
  sorry

end NUMINAMATH_CALUDE_travel_methods_count_l1700_170054


namespace NUMINAMATH_CALUDE_rounded_expression_smaller_l1700_170051

theorem rounded_expression_smaller (a b c : ℕ+) :
  let exact_value := (a.val^2 : ℚ) / b.val + c.val^3
  let rounded_a := (a.val + 1 : ℚ)
  let rounded_b := (b.val + 1 : ℚ)
  let rounded_c := (c.val - 1 : ℚ)
  let rounded_value := rounded_a^2 / rounded_b + rounded_c^3
  rounded_value < exact_value :=
by sorry

end NUMINAMATH_CALUDE_rounded_expression_smaller_l1700_170051


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l1700_170092

theorem greatest_integer_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 5*x - 4 < 3 - 2*x := by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l1700_170092


namespace NUMINAMATH_CALUDE_li_ming_father_age_l1700_170002

theorem li_ming_father_age :
  ∃! age : ℕ, 
    18 ≤ age ∧ age ≤ 70 ∧
    ∃ (month day : ℕ), 
      1 ≤ month ∧ month ≤ 12 ∧
      1 ≤ day ∧ day ≤ 31 ∧
      age * month * day = 2975 ∧
      age = 35 := by
sorry

end NUMINAMATH_CALUDE_li_ming_father_age_l1700_170002


namespace NUMINAMATH_CALUDE_F_is_integer_exists_valid_s_and_t_l1700_170084

/-- Given a four-digit number, swap the thousands and tens digits, and the hundreds and units digits -/
def swap_digits (n : Nat) : Nat :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 * c + 100 * d + 10 * a + b

/-- The "wholehearted number" function -/
def F (n : Nat) : Nat :=
  (n + swap_digits n) / 101

/-- Theorem: F(n) is an integer for any four-digit number n -/
theorem F_is_integer (n : Nat) (h : 1000 ≤ n ∧ n < 10000) : ∃ k : Nat, F n = k := by
  sorry

/-- Helper function to check if a number is divisible by 8 -/
def is_divisible_by_8 (n : Int) : Prop :=
  ∃ k : Int, n = 8 * k

/-- Function to generate s given a and b -/
def s (a b : Nat) : Nat :=
  3800 + 10 * a + b

/-- Function to generate t given a and b -/
def t (a b : Nat) : Nat :=
  1000 * b + 100 * a + 13

/-- Theorem: There exist values of a and b such that 3F(t) - F(s) is divisible by 8 -/
theorem exists_valid_s_and_t :
  ∃ (a b : Nat), 1 ≤ a ∧ a ≤ 5 ∧ 5 ≤ b ∧ b ≤ 9 ∧ is_divisible_by_8 (3 * (F (t a b)) - (F (s a b))) := by
  sorry

end NUMINAMATH_CALUDE_F_is_integer_exists_valid_s_and_t_l1700_170084


namespace NUMINAMATH_CALUDE_area_between_circles_l1700_170080

/-- The area of the region inside a large circle and outside eight congruent circles forming a ring --/
theorem area_between_circles (R : ℝ) (h : R = 40) : ∃ L : ℝ,
  (∃ (r : ℝ), 
    -- Eight congruent circles with radius r
    -- Each circle is externally tangent to its two adjacent circles
    -- All eight circles are internally tangent to a larger circle with radius R
    r > 0 ∧ r = R / 3 ∧
    -- L is the area of the region inside the large circle and outside all eight circles
    L = π * R^2 - 8 * π * r^2) ∧
  L = 1600 * π :=
sorry

end NUMINAMATH_CALUDE_area_between_circles_l1700_170080


namespace NUMINAMATH_CALUDE_unused_sector_angle_l1700_170011

/-- Given a cone with radius 10 cm and volume 250π cm³, 
    prove that the angle of the sector not used to form the cone is 72°. -/
theorem unused_sector_angle (r h : ℝ) (volume : ℝ) : 
  r = 10 → 
  volume = 250 * Real.pi → 
  (1/3) * Real.pi * r^2 * h = volume → 
  Real.sqrt (r^2 + h^2) = 12.5 → 
  360 - (360 * ((2 * Real.pi * r) / (2 * Real.pi * 12.5))) = 72 := by
  sorry

#check unused_sector_angle

end NUMINAMATH_CALUDE_unused_sector_angle_l1700_170011


namespace NUMINAMATH_CALUDE_sum_of_variables_l1700_170021

theorem sum_of_variables (x y z : ℚ) 
  (eq1 : y + z = 18 - 4*x)
  (eq2 : x + z = 16 - 4*y)
  (eq3 : x + y = 9 - 4*z) :
  3*x + 3*y + 3*z = 43/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l1700_170021


namespace NUMINAMATH_CALUDE_fixed_points_theorem_l1700_170077

/-- A function f(x) = ax^2 + (b+1)x + (b-1) where a ≠ 0 -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + (b - 1)

/-- x0 is a fixed point of f if f(x0) = x0 -/
def is_fixed_point (a b x0 : ℝ) : Prop := f a b x0 = x0

/-- The function has two distinct fixed points -/
def has_two_distinct_fixed_points (a b : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point a b x1 ∧ is_fixed_point a b x2

/-- The fixed points are symmetric with respect to the line y = kx + 1/(2a^2 + 1) -/
def fixed_points_symmetric (a b : ℝ) : Prop :=
  ∃ x1 x2 k : ℝ, x1 ≠ x2 ∧ is_fixed_point a b x1 ∧ is_fixed_point a b x2 ∧
    (f a b x1 + f a b x2) / 2 = k * (x1 + x2) / 2 + 1 / (2 * a^2 + 1)

/-- Main theorem -/
theorem fixed_points_theorem (a b : ℝ) (ha : a ≠ 0) :
  (has_two_distinct_fixed_points a b ↔ 0 < a ∧ a < 1) ∧
  (0 < a ∧ a < 1 → ∃ b_min : ℝ, ∀ b : ℝ, fixed_points_symmetric a b → b ≥ b_min) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_theorem_l1700_170077


namespace NUMINAMATH_CALUDE_lindas_cookies_l1700_170063

theorem lindas_cookies (classmates : ℕ) (cookies_per_student : ℕ) (cookies_per_batch : ℕ) 
  (chocolate_chip_batches : ℕ) (remaining_batches : ℕ) :
  classmates = 24 →
  cookies_per_student = 10 →
  cookies_per_batch = 48 →
  chocolate_chip_batches = 2 →
  remaining_batches = 2 →
  (classmates * cookies_per_student - chocolate_chip_batches * cookies_per_batch) / cookies_per_batch - remaining_batches = 1 :=
by sorry

end NUMINAMATH_CALUDE_lindas_cookies_l1700_170063


namespace NUMINAMATH_CALUDE_sector_area_l1700_170017

theorem sector_area (θ : Real) (r : Real) (h1 : θ = 135) (h2 : r = 20) :
  (θ * π * r^2) / 360 = 150 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1700_170017


namespace NUMINAMATH_CALUDE_smallest_multiple_eighty_is_solution_eighty_is_smallest_l1700_170044

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 540 * x % 800 = 0 → x ≥ 80 := by
  sorry

theorem eighty_is_solution : 540 * 80 % 800 = 0 := by
  sorry

theorem eighty_is_smallest : ∀ y : ℕ, y > 0 ∧ 540 * y % 800 = 0 → y ≥ 80 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_eighty_is_solution_eighty_is_smallest_l1700_170044


namespace NUMINAMATH_CALUDE_linear_regression_change_specific_regression_change_l1700_170098

/-- Given a linear regression equation y = a + bx, this theorem proves
    that when x increases by 1 unit, y changes by b units. -/
theorem linear_regression_change (a b : ℝ) :
  let y : ℝ → ℝ := λ x ↦ a + b * x
  ∀ x : ℝ, y (x + 1) - y x = b := by
  sorry

/-- For the specific linear regression equation y = 2 - 3.5x,
    this theorem proves that when x increases by 1 unit, y decreases by 3.5 units. -/
theorem specific_regression_change :
  let y : ℝ → ℝ := λ x ↦ 2 - 3.5 * x
  ∀ x : ℝ, y (x + 1) - y x = -3.5 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_change_specific_regression_change_l1700_170098


namespace NUMINAMATH_CALUDE_inequality_solution_l1700_170034

theorem inequality_solution (x : ℝ) : (3*x + 7)/5 + 1 > x ↔ x < 6 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1700_170034


namespace NUMINAMATH_CALUDE_range_of_m_l1700_170059

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) m, f x ≤ 10) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) m, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) m, f x = 10) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) m, f x = 1) →
  m ∈ Set.Icc 2 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1700_170059


namespace NUMINAMATH_CALUDE_kenny_trumpet_practice_l1700_170062

/-- Given Kenny's activities and their durations, prove that he practiced trumpet for 40 hours. -/
theorem kenny_trumpet_practice (x y z w : ℕ) : 
  let basketball : ℕ := 10
  let running : ℕ := 2 * basketball
  let trumpet : ℕ := 2 * running
  let other_activities : ℕ := x + y + z + w
  other_activities = basketball + running + trumpet - 5
  → trumpet = 40 := by
sorry

end NUMINAMATH_CALUDE_kenny_trumpet_practice_l1700_170062


namespace NUMINAMATH_CALUDE_equation_solution_l1700_170048

theorem equation_solution (x : ℝ) : 
  (8 * x^2 + 120 * x + 7) / (3 * x + 10) = 4 * x + 2 ↔ 
  -4 * x^2 + 74 * x - 13 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1700_170048


namespace NUMINAMATH_CALUDE_lemonade_water_quarts_l1700_170078

/-- Proves the number of quarts of water needed for a special lemonade recipe -/
theorem lemonade_water_quarts : 
  let total_parts : ℚ := 5 + 3
  let water_parts : ℚ := 5
  let total_gallons : ℚ := 5
  let quarts_per_gallon : ℚ := 4
  (water_parts / total_parts) * total_gallons * quarts_per_gallon = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_water_quarts_l1700_170078


namespace NUMINAMATH_CALUDE_frieda_hop_probability_l1700_170099

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Checks if a position is a corner -/
def is_corner (p : Position) : Bool :=
  (p.x = 0 ∧ p.y = 0) ∨ (p.x = 0 ∧ p.y = 3) ∨
  (p.x = 3 ∧ p.y = 0) ∨ (p.x = 3 ∧ p.y = 3)

/-- Represents a single hop direction -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Applies a hop in the given direction, considering wrap-around rules -/
def apply_hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x, (p.y + 1) % 4⟩
  | Direction.Down => ⟨p.x, (p.y - 1 + 4) % 4⟩
  | Direction.Left => ⟨(p.x - 1 + 4) % 4, p.y⟩
  | Direction.Right => ⟨(p.x + 1) % 4, p.y⟩

/-- Calculates the probability of reaching a corner in at most n hops -/
def prob_reach_corner (start : Position) (n : Nat) : ℚ :=
  sorry  -- Proof implementation goes here

/-- The main theorem to prove -/
theorem frieda_hop_probability :
  prob_reach_corner ⟨0, 1⟩ 3 = 21 / 32 :=
by sorry  -- Proof goes here

end NUMINAMATH_CALUDE_frieda_hop_probability_l1700_170099


namespace NUMINAMATH_CALUDE_storeroom_contains_912_blocks_l1700_170000

/-- Calculates the number of blocks in a rectangular storeroom with given dimensions and wall thickness -/
def storeroom_blocks (length width height wall_thickness : ℕ) : ℕ :=
  let total_volume := length * width * height
  let internal_length := length - 2 * wall_thickness
  let internal_width := width - 2 * wall_thickness
  let internal_height := height - wall_thickness
  let internal_volume := internal_length * internal_width * internal_height
  total_volume - internal_volume

/-- Theorem stating that a storeroom with given dimensions contains 912 blocks -/
theorem storeroom_contains_912_blocks :
  storeroom_blocks 15 12 8 2 = 912 := by
  sorry

#eval storeroom_blocks 15 12 8 2

end NUMINAMATH_CALUDE_storeroom_contains_912_blocks_l1700_170000


namespace NUMINAMATH_CALUDE_store_holiday_customers_l1700_170097

/-- The number of customers entering a store during holiday season -/
def holiday_customers (normal_rate : ℕ) (hours : ℕ) : ℕ :=
  2 * normal_rate * hours

/-- Theorem: Given the conditions, the store will see 2800 customers in 8 hours during the holiday season -/
theorem store_holiday_customers :
  holiday_customers 175 8 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_store_holiday_customers_l1700_170097


namespace NUMINAMATH_CALUDE_december_sales_multiple_l1700_170042

theorem december_sales_multiple (A : ℝ) (M : ℝ) (h1 : M > 0) :
  M * A = 0.3125 * (11 * A + M * A) → M = 5 := by
sorry

end NUMINAMATH_CALUDE_december_sales_multiple_l1700_170042


namespace NUMINAMATH_CALUDE_arithmetic_progression_poly_j_value_l1700_170038

/-- A polynomial of degree 4 with four distinct real roots in arithmetic progression -/
structure ArithmeticProgressionPoly where
  j : ℝ
  k : ℝ
  roots_distinct : True
  roots_real : True
  roots_arithmetic : True

/-- The value of j in the polynomial x^4 + jx^2 + kx + 81 with four distinct real roots in arithmetic progression is -10 -/
theorem arithmetic_progression_poly_j_value (p : ArithmeticProgressionPoly) : p.j = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_poly_j_value_l1700_170038


namespace NUMINAMATH_CALUDE_magic_wheel_product_l1700_170024

def is_even (n : ℕ) : Prop := n % 2 = 0

def has_two_odd_between (a d : ℕ) : Prop :=
  ∃ b c : ℕ, a < b ∧ b < c ∧ c < d ∧
  ¬(is_even b) ∧ ¬(is_even c) ∧
  (d - a) % 16 = 3

theorem magic_wheel_product :
  ∀ a d : ℕ,
  1 ≤ a ∧ a ≤ 16 ∧
  1 ≤ d ∧ d ≤ 16 ∧
  is_even a ∧
  is_even d ∧
  has_two_odd_between a d →
  a * d = 120 :=
sorry

end NUMINAMATH_CALUDE_magic_wheel_product_l1700_170024


namespace NUMINAMATH_CALUDE_penalty_kick_test_l1700_170083

/-- The probability of scoring a single penalty kick -/
def p_score : ℚ := 2/3

/-- The probability of missing a single penalty kick -/
def p_miss : ℚ := 1 - p_score

/-- The probability of being admitted in the penalty kick test -/
def p_admitted : ℚ := 
  p_score * p_score + 
  p_miss * p_score * p_score + 
  p_miss * p_miss * p_score * p_score + 
  p_score * p_miss * p_score * p_score

/-- The expected number of goals scored in the penalty kick test -/
def expected_goals : ℚ := 
  0 * (p_miss * p_miss * p_miss) + 
  1 * (2 * p_score * p_miss * p_miss + p_miss * p_miss * p_score * p_miss) + 
  2 * (p_score * p_score + p_miss * p_score * p_score + p_miss * p_miss * p_score * p_score + p_score * p_miss * p_score * p_miss) + 
  3 * (p_score * p_miss * p_score * p_score)

theorem penalty_kick_test :
  p_admitted = 20/27 ∧ expected_goals = 50/27 := by
  sorry

end NUMINAMATH_CALUDE_penalty_kick_test_l1700_170083


namespace NUMINAMATH_CALUDE_female_democrat_ratio_l1700_170082

theorem female_democrat_ratio (total_participants male_participants female_participants female_democrats : ℕ) 
  (h1 : total_participants = 720)
  (h2 : male_participants + female_participants = total_participants)
  (h3 : male_participants / 4 = total_participants / 3 - female_democrats)
  (h4 : total_participants / 3 = 240)
  (h5 : female_democrats = 120) :
  female_democrats / female_participants = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_female_democrat_ratio_l1700_170082


namespace NUMINAMATH_CALUDE_min_value_problem_l1700_170086

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1 / x^2) + (1 / y^2) + (1 / (x * y)) ≥ 3 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ (1 / a^2) + (1 / b^2) + (1 / (a * b)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1700_170086


namespace NUMINAMATH_CALUDE_added_amount_l1700_170023

theorem added_amount (x : ℝ) (y : ℝ) (h1 : x = 6) (h2 : 2 / 3 * x + y = 10) : y = 6 := by
  sorry

end NUMINAMATH_CALUDE_added_amount_l1700_170023


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1700_170016

theorem complex_modulus_problem (a b : ℝ) :
  (Complex.mk a 1) * (Complex.mk 1 (-1)) = Complex.mk 3 b →
  Complex.abs (Complex.mk a b) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1700_170016


namespace NUMINAMATH_CALUDE_min_abs_sum_l1700_170029

theorem min_abs_sum (a b c : ℝ) (h1 : a + b + c = -2) (h2 : a * b * c = -4) :
  ∀ x y z : ℝ, x + y + z = -2 → x * y * z = -4 → |a| + |b| + |c| ≤ |x| + |y| + |z| ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀ + b₀ + c₀ = -2 ∧ a₀ * b₀ * c₀ = -4 ∧ |a₀| + |b₀| + |c₀| = 6 := by
sorry

end NUMINAMATH_CALUDE_min_abs_sum_l1700_170029


namespace NUMINAMATH_CALUDE_parabola_point_order_l1700_170010

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Theorem: Given a quadratic function y = ax^2 + bx - 1 and a linear function y = ax
    intersecting at (-2, 1), with a = -1/2 and b = -2, and points A(2, y1), B(-2, y2),
    C(3/2, y3) on the parabola, prove that y1 < y3 < y2 holds. -/
theorem parabola_point_order
  (q : QuadraticFunction)
  (l : LinearFunction)
  (h1 : q.a = -1/2)
  (h2 : q.b = -2)
  (h3 : q.c = -1)
  (h4 : l.m = q.a)
  (h5 : l.b = 0)
  (h6 : q.a * (-2)^2 + q.b * (-2) + q.c = 1)
  (h7 : l.m * (-2) + l.b = 1)
  (y1 y2 y3 : ℝ)
  (h8 : y1 = q.a * 2^2 + q.b * 2 + q.c)
  (h9 : y2 = q.a * (-2)^2 + q.b * (-2) + q.c)
  (h10 : y3 = q.a * (3/2)^2 + q.b * (3/2) + q.c)
  : y1 < y3 ∧ y3 < y2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_order_l1700_170010


namespace NUMINAMATH_CALUDE_alexander_exhibition_problem_l1700_170067

/-- The number of pictures at each new gallery -/
def pictures_per_new_gallery (
  original_pictures : ℕ
  ) (new_galleries : ℕ
  ) (pencils_per_picture : ℕ
  ) (pencils_for_signing : ℕ
  ) (total_pencils : ℕ
  ) : ℕ :=
  let total_exhibitions := new_galleries + 1
  let pencils_for_drawing := total_pencils - (total_exhibitions * pencils_for_signing)
  let total_pictures := pencils_for_drawing / pencils_per_picture
  let new_gallery_pictures := total_pictures - original_pictures
  new_gallery_pictures / new_galleries

theorem alexander_exhibition_problem :
  pictures_per_new_gallery 9 5 4 2 88 = 2 := by
  sorry

end NUMINAMATH_CALUDE_alexander_exhibition_problem_l1700_170067


namespace NUMINAMATH_CALUDE_line_l_line_l_l1700_170061

/-- The equation of line l -/
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

/-- The equation of line l' that passes through (-1, 3) and is parallel to l -/
def line_l'_parallel (x y : ℝ) : Prop := 3 * x + 4 * y - 9 = 0

/-- The equation of line l' that is symmetric to l about the y-axis -/
def line_l'_symmetric (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0

/-- Point (-1, 3) -/
def point : ℝ × ℝ := (-1, 3)

theorem line_l'_parallel_correct :
  (∀ x y, line_l'_parallel x y ↔ (∃ k, y - point.2 = k * (x - point.1) ∧
    ∀ x₁ y₁ x₂ y₂, line_l x₁ y₁ → line_l x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = k)) ∧
  line_l'_parallel point.1 point.2 :=
sorry

theorem line_l'_symmetric_correct :
  ∀ x y, line_l'_symmetric x y ↔ line_l (-x) y :=
sorry

end NUMINAMATH_CALUDE_line_l_line_l_l1700_170061
