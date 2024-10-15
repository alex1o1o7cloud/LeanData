import Mathlib

namespace NUMINAMATH_CALUDE_triangle_line_equations_l29_2992

/-- Triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given triangle ABC, returns the equation of line AB -/
def line_AB (t : Triangle) : LineEquation :=
  { a := 3, b := 8, c := 15 }

/-- Given triangle ABC, returns the equation of the altitude from C to AB -/
def altitude_C (t : Triangle) : LineEquation :=
  { a := 8, b := -3, c := 6 }

theorem triangle_line_equations (t : Triangle) 
  (h1 : t.A = (-5, 0)) 
  (h2 : t.B = (3, -3)) 
  (h3 : t.C = (0, 2)) : 
  (line_AB t = { a := 3, b := 8, c := 15 }) ∧ 
  (altitude_C t = { a := 8, b := -3, c := 6 }) := by
  sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l29_2992


namespace NUMINAMATH_CALUDE_sam_distance_l29_2988

/-- Given Marguerite's travel details and Sam's driving time, prove Sam's distance traveled. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) :
  (marguerite_distance / marguerite_time) * sam_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_sam_distance_l29_2988


namespace NUMINAMATH_CALUDE_exponent_division_l29_2994

theorem exponent_division (a : ℝ) : a^10 / a^5 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l29_2994


namespace NUMINAMATH_CALUDE_game_result_l29_2920

theorem game_result (x : ℝ) : ((x + 90) - 27 - x) * 11 / 3 = 231 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l29_2920


namespace NUMINAMATH_CALUDE_ava_finishes_on_monday_l29_2915

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def days_after (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => next_day (days_after d n)

def reading_time (n : ℕ) : ℕ := 2 * n - 1

def total_reading_time (n : ℕ) : ℕ := 
  (List.range n).map reading_time |>.sum

theorem ava_finishes_on_monday : 
  days_after DayOfWeek.Sunday (total_reading_time 20) = DayOfWeek.Monday := by
  sorry


end NUMINAMATH_CALUDE_ava_finishes_on_monday_l29_2915


namespace NUMINAMATH_CALUDE_simplify_expression_l29_2946

theorem simplify_expression :
  1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1)) =
  ((Real.sqrt 3 - 2 * Real.sqrt 5 - 1) * (-16 - 2 * Real.sqrt 3)) / 244 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l29_2946


namespace NUMINAMATH_CALUDE_no_upper_bound_for_y_l29_2939

-- Define the equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - (y - 3)^2 / 9 = 1

-- Theorem stating that there is no upper bound for y
theorem no_upper_bound_for_y :
  ∀ M : ℝ, ∃ x y : ℝ, hyperbola_equation x y ∧ y > M :=
by sorry

end NUMINAMATH_CALUDE_no_upper_bound_for_y_l29_2939


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_l29_2907

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a four-digit integer -/
def isFourDigit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible :
  ∃ (p : ℕ),
    isFourDigit p ∧
    isFourDigit (reverseDigits p) ∧
    p % 99 = 0 ∧
    (reverseDigits p) % 99 = 0 ∧
    p % 7 = 0 ∧
    p = 7623 ∧
    ∀ (q : ℕ),
      isFourDigit q ∧
      isFourDigit (reverseDigits q) ∧
      q % 99 = 0 ∧
      (reverseDigits q) % 99 = 0 ∧
      q % 7 = 0 →
      q ≤ 7623 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_l29_2907


namespace NUMINAMATH_CALUDE_modified_fibonacci_series_sum_l29_2952

/-- Modified Fibonacci sequence -/
def G : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | (n + 2) => G (n + 1) + G n

/-- The sum of the series G_n / 5^n from n = 0 to infinity -/
noncomputable def series_sum : ℚ := ∑' n, G n / (5 : ℚ) ^ n

/-- Theorem stating that the sum of the series G_n / 5^n from n = 0 to infinity equals 50/19 -/
theorem modified_fibonacci_series_sum : series_sum = 50 / 19 := by sorry

end NUMINAMATH_CALUDE_modified_fibonacci_series_sum_l29_2952


namespace NUMINAMATH_CALUDE_dana_beth_same_money_l29_2906

-- Define the set of individuals
inductive Person : Type
  | Abby : Person
  | Beth : Person
  | Cindy : Person
  | Dana : Person
  | Eve : Person

-- Define the money function
variable (money : Person → ℕ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q ∨ (p = Person.Dana ∧ q = Person.Beth) ∨ (p = Person.Beth ∧ q = Person.Dana)
axiom abby_more_than_cindy : money Person.Abby > money Person.Cindy
axiom beth_less_than_eve : money Person.Beth < money Person.Eve
axiom beth_more_than_dana : money Person.Beth > money Person.Dana
axiom abby_less_than_dana : money Person.Abby < money Person.Dana
axiom dana_not_most : ∃ (p : Person), money p > money Person.Dana
axiom cindy_more_than_beth : money Person.Cindy > money Person.Beth

-- Theorem to prove
theorem dana_beth_same_money : money Person.Dana = money Person.Beth :=
  sorry

end NUMINAMATH_CALUDE_dana_beth_same_money_l29_2906


namespace NUMINAMATH_CALUDE_min_distance_to_line_l29_2989

theorem min_distance_to_line (x y : ℝ) : 
  3 * x + 4 * y = 24 → x ≥ 0 → 
  ∃ (min_val : ℝ), min_val = 24 / 5 ∧ 
    ∀ (x' y' : ℝ), 3 * x' + 4 * y' = 24 → x' ≥ 0 → 
      Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l29_2989


namespace NUMINAMATH_CALUDE_squirrel_calories_proof_l29_2936

/-- The number of squirrels Brandon can catch in 1 hour -/
def squirrels_per_hour : ℕ := 6

/-- The number of rabbits Brandon can catch in 1 hour -/
def rabbits_per_hour : ℕ := 2

/-- The number of calories in each rabbit -/
def calories_per_rabbit : ℕ := 800

/-- The additional calories Brandon gets from catching squirrels instead of rabbits in 1 hour -/
def additional_calories : ℕ := 200

/-- The number of calories in each squirrel -/
def calories_per_squirrel : ℕ := 300

theorem squirrel_calories_proof :
  squirrels_per_hour * calories_per_squirrel = 
  rabbits_per_hour * calories_per_rabbit + additional_calories :=
by sorry

end NUMINAMATH_CALUDE_squirrel_calories_proof_l29_2936


namespace NUMINAMATH_CALUDE_cuboid_properties_l29_2928

/-- Represents a cuboid with length, width, and height -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the sum of all edge lengths of a cuboid -/
def sumEdgeLengths (c : Cuboid) : ℝ :=
  4 * (c.length + c.width + c.height)

/-- Calculates the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ :=
  c.length * c.width * c.height

/-- Theorem about a specific cuboid's properties -/
theorem cuboid_properties :
  ∃ c : Cuboid,
    c.length = 2 * c.width ∧
    c.width = c.height ∧
    sumEdgeLengths c = 48 ∧
    surfaceArea c = 90 ∧
    volume c = 54 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_properties_l29_2928


namespace NUMINAMATH_CALUDE_factorial_800_trailing_zeros_l29_2960

def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem factorial_800_trailing_zeros :
  trailingZeros 800 = 199 := by
  sorry

end NUMINAMATH_CALUDE_factorial_800_trailing_zeros_l29_2960


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l29_2987

/-- Given two vectors a and b in R², where a = (1, m) and b = (3, -2),
    if a + b is perpendicular to b, then m = 8. -/
theorem vector_perpendicular_condition (m : ℝ) :
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (3, -2)
  (a.1 + b.1, a.2 + b.2) • b = 0 → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l29_2987


namespace NUMINAMATH_CALUDE_irrational_pair_sum_six_l29_2944

theorem irrational_pair_sum_six : ∃ (x y : ℝ), Irrational x ∧ Irrational y ∧ x + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_irrational_pair_sum_six_l29_2944


namespace NUMINAMATH_CALUDE_perimeter_difference_is_six_l29_2900

/-- Calculate the perimeter of a rectangle --/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Calculate the perimeter of a divided rectangle --/
def divided_rectangle_perimeter (length width divisions : ℕ) : ℕ :=
  rectangle_perimeter length width + 2 * divisions

/-- The positive difference between the perimeters of two specific rectangles --/
def perimeter_difference : ℕ :=
  Int.natAbs (rectangle_perimeter 7 3 - divided_rectangle_perimeter 6 2 5)

theorem perimeter_difference_is_six : perimeter_difference = 6 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_is_six_l29_2900


namespace NUMINAMATH_CALUDE_volleyball_net_max_removable_edges_l29_2912

/-- Represents a volleyball net graph -/
structure VolleyballNet where
  rows : Nat
  cols : Nat

/-- Calculates the number of vertices in the volleyball net graph -/
def VolleyballNet.vertexCount (net : VolleyballNet) : Nat :=
  (net.rows + 1) * (net.cols + 1) + net.rows * net.cols

/-- Calculates the total number of edges in the volleyball net graph -/
def VolleyballNet.edgeCount (net : VolleyballNet) : Nat :=
  -- This is a placeholder. The actual calculation would be more complex.
  4 * net.rows * net.cols + net.rows * (net.cols - 1) + net.cols * (net.rows - 1)

/-- Theorem: The maximum number of edges that can be removed without disconnecting
    the graph for a 10x20 volleyball net is 800 -/
theorem volleyball_net_max_removable_edges :
  let net : VolleyballNet := { rows := 10, cols := 20 }
  ∃ (removable : Nat), removable = net.edgeCount - (net.vertexCount - 1) ∧ removable = 800 := by
  sorry


end NUMINAMATH_CALUDE_volleyball_net_max_removable_edges_l29_2912


namespace NUMINAMATH_CALUDE_max_yellow_apples_max_total_apples_l29_2958

/-- Represents the number of apples of each color in the basket -/
structure Basket :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Represents the number of apples taken from the basket -/
structure TakenApples :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Checks if the condition for stopping is met -/
def stoppingCondition (taken : TakenApples) : Prop :=
  taken.green < taken.yellow ∧ taken.yellow < taken.red

/-- The initial basket of apples -/
def initialBasket : Basket :=
  { green := 10, yellow := 13, red := 18 }

/-- Theorem for the maximum number of yellow apples that can be taken -/
theorem max_yellow_apples :
  ∃ (taken : TakenApples),
    taken.yellow = 13 ∧
    taken.yellow ≤ taken.red ∧
    taken.green ≤ taken.yellow ∧
    ∀ (other : TakenApples),
      other.yellow > 13 →
      other.yellow > other.red ∨ other.green > other.yellow :=
sorry

/-- Theorem for the maximum number of apples that can be taken in total -/
theorem max_total_apples :
  ∃ (taken : TakenApples),
    taken.green + taken.yellow + taken.red = 39 ∧
    ¬(stoppingCondition taken) ∧
    ∀ (other : TakenApples),
      other.green + other.yellow + other.red > 39 →
      stoppingCondition other :=
sorry

end NUMINAMATH_CALUDE_max_yellow_apples_max_total_apples_l29_2958


namespace NUMINAMATH_CALUDE_board_sum_possible_l29_2943

theorem board_sum_possible : ∃ (a b : ℕ), 
  a ≤ 10 ∧ b ≤ 11 ∧ 
  (10 - a : ℝ) * 1.11 + (11 - b : ℝ) * 1.01 = 20.19 := by
sorry

end NUMINAMATH_CALUDE_board_sum_possible_l29_2943


namespace NUMINAMATH_CALUDE_no_real_solutions_l29_2962

theorem no_real_solutions :
  ¬∃ (x : ℝ), (2*x - 6)^2 + 4 = -2*|x| := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l29_2962


namespace NUMINAMATH_CALUDE_base_spheres_in_triangular_pyramid_l29_2914

/-- The number of spheres in the nth layer of a regular triangular pyramid -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of spheres in a regular triangular pyramid with n layers -/
def total_spheres (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- 
Given a regular triangular pyramid of tightly packed identical spheres,
if the total number of spheres is 120, then the number of spheres in the base is 36.
-/
theorem base_spheres_in_triangular_pyramid :
  ∃ n : ℕ, total_spheres n = 120 ∧ triangular_number n = 36 :=
by sorry

end NUMINAMATH_CALUDE_base_spheres_in_triangular_pyramid_l29_2914


namespace NUMINAMATH_CALUDE_smallest_two_digit_reverse_diff_perfect_square_l29_2997

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_two_digit_reverse_diff_perfect_square :
  ∃ N : ℕ, is_two_digit N ∧
    is_perfect_square (N - reverse_digits N) ∧
    (N - reverse_digits N > 0) ∧
    (∀ M : ℕ, is_two_digit M →
      is_perfect_square (M - reverse_digits M) →
      (M - reverse_digits M > 0) →
      N ≤ M) ∧
    N = 90 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_reverse_diff_perfect_square_l29_2997


namespace NUMINAMATH_CALUDE_train_speed_calculation_l29_2964

/-- The speed of a train given the lengths of two trains, the speed of one train, and the time they take to cross each other when moving in opposite directions. -/
theorem train_speed_calculation (length_train1 length_train2 speed_train2 crossing_time : ℝ) 
  (h1 : length_train1 = 150)
  (h2 : length_train2 = 350.04)
  (h3 : speed_train2 = 80)
  (h4 : crossing_time = 9)
  : ∃ (speed_train1 : ℝ), abs (speed_train1 - 120.016) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l29_2964


namespace NUMINAMATH_CALUDE_jongkooks_milk_consumption_l29_2925

/-- Converts liters to milliliters -/
def liters_to_ml (l : ℚ) : ℚ := 1000 * l

/-- Represents the amount of milk drunk in milliliters for each day -/
structure MilkConsumption where
  day1 : ℚ
  day2 : ℚ
  day3 : ℚ

/-- Calculates the total milk consumption in milliliters -/
def total_consumption (mc : MilkConsumption) : ℚ :=
  mc.day1 + mc.day2 + mc.day3

theorem jongkooks_milk_consumption :
  ∃ (mc : MilkConsumption),
    mc.day1 = liters_to_ml 3 + 7 ∧
    mc.day3 = 840 ∧
    total_consumption mc = liters_to_ml 6 + 30 ∧
    mc.day2 = 2183 := by
  sorry

end NUMINAMATH_CALUDE_jongkooks_milk_consumption_l29_2925


namespace NUMINAMATH_CALUDE_least_months_to_double_debt_l29_2982

def initial_amount : ℝ := 1200
def interest_rate : ℝ := 0.06

def compound_factor : ℝ := 1 + interest_rate

theorem least_months_to_double_debt : 
  (∀ n : ℕ, n < 12 → compound_factor ^ n ≤ 2) ∧ 
  compound_factor ^ 12 > 2 := by
  sorry

end NUMINAMATH_CALUDE_least_months_to_double_debt_l29_2982


namespace NUMINAMATH_CALUDE_clock_face_ratio_l29_2903

theorem clock_face_ratio (s : ℝ) (h : s = 4) : 
  let r : ℝ := s / 2
  let circle_area : ℝ := π * r^2
  let triangle_area : ℝ := r^2
  let sector_area : ℝ := circle_area / 12
  sector_area / triangle_area = π / 2 := by sorry

end NUMINAMATH_CALUDE_clock_face_ratio_l29_2903


namespace NUMINAMATH_CALUDE_toms_books_l29_2929

/-- Given that Joan has 10 books and together with Tom they have 48 books,
    prove that Tom has 38 books. -/
theorem toms_books (joan_books : ℕ) (total_books : ℕ) (h1 : joan_books = 10) (h2 : total_books = 48) :
  total_books - joan_books = 38 := by
  sorry

end NUMINAMATH_CALUDE_toms_books_l29_2929


namespace NUMINAMATH_CALUDE_quiz_mcq_count_l29_2961

theorem quiz_mcq_count :
  ∀ (n : ℕ),
  (((1 : ℚ) / 3) ^ n * ((1 : ℚ) / 2) ^ 2 = (1 : ℚ) / 12) →
  n = 1 :=
by sorry

end NUMINAMATH_CALUDE_quiz_mcq_count_l29_2961


namespace NUMINAMATH_CALUDE_marks_lost_per_wrong_answer_l29_2978

/-- Prove that the number of marks lost for each wrong answer is 1 -/
theorem marks_lost_per_wrong_answer
  (total_questions : ℕ)
  (marks_per_correct : ℕ)
  (total_marks : ℕ)
  (correct_answers : ℕ)
  (h1 : total_questions = 60)
  (h2 : marks_per_correct = 4)
  (h3 : total_marks = 130)
  (h4 : correct_answers = 38) :
  ∃ (marks_lost : ℕ), 
    marks_lost = 1 ∧ 
    total_marks = correct_answers * marks_per_correct - (total_questions - correct_answers) * marks_lost :=
by
  sorry

end NUMINAMATH_CALUDE_marks_lost_per_wrong_answer_l29_2978


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l29_2976

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {0, 1}

-- Define set N
def N : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl M ∩ N) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l29_2976


namespace NUMINAMATH_CALUDE_min_complex_value_is_zero_l29_2953

theorem min_complex_value_is_zero 
  (a b c : ℤ) 
  (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_omega_power : ω^4 = 1) 
  (h_omega_not_one : ω ≠ 1) :
  ∃ (a' b' c' : ℤ) (h_distinct' : a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c'),
    Complex.abs (a' + b' * ω + c' * ω^3) = 0 ∧
    ∀ (x y z : ℤ) (h_xyz_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z),
      Complex.abs (x + y * ω + z * ω^3) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_complex_value_is_zero_l29_2953


namespace NUMINAMATH_CALUDE_three_points_determine_plane_line_and_point_determine_plane_trapezoid_determines_plane_circle_points_not_always_determine_plane_l29_2985

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- A circle in 3D space -/
structure Circle3D where
  center : Point3D
  radius : ℝ
  normal : Point3D

/-- Three points are collinear if they lie on the same line -/
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

/-- A point lies on a line -/
def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

/-- A point lies on a plane -/
def point_on_plane (p : Point3D) (plane : Plane3D) : Prop := sorry

/-- A point lies on a circle -/
def point_on_circle (p : Point3D) (c : Circle3D) : Prop := sorry

/-- Three points determine a unique plane -/
theorem three_points_determine_plane (p1 p2 p3 : Point3D) (h : ¬collinear p1 p2 p3) : 
  ∃! plane : Plane3D, point_on_plane p1 plane ∧ point_on_plane p2 plane ∧ point_on_plane p3 plane :=
sorry

/-- A line and a point not on the line determine a unique plane -/
theorem line_and_point_determine_plane (l : Line3D) (p : Point3D) (h : ¬point_on_line p l) :
  ∃! plane : Plane3D, (∀ q : Point3D, point_on_line q l → point_on_plane q plane) ∧ point_on_plane p plane :=
sorry

/-- A trapezoid determines a unique plane -/
theorem trapezoid_determines_plane (p1 p2 p3 p4 : Point3D) 
  (h1 : ∃ l1 l2 : Line3D, point_on_line p1 l1 ∧ point_on_line p2 l1 ∧ point_on_line p3 l2 ∧ point_on_line p4 l2) 
  (h2 : l1 ≠ l2) :
  ∃! plane : Plane3D, point_on_plane p1 plane ∧ point_on_plane p2 plane ∧ point_on_plane p3 plane ∧ point_on_plane p4 plane :=
sorry

/-- The center and two points on a circle do not always determine a unique plane -/
theorem circle_points_not_always_determine_plane :
  ∃ (c : Circle3D) (p1 p2 : Point3D), 
    point_on_circle p1 c ∧ point_on_circle p2 c ∧ 
    ¬(∃! plane : Plane3D, point_on_plane c.center plane ∧ point_on_plane p1 plane ∧ point_on_plane p2 plane) :=
sorry

end NUMINAMATH_CALUDE_three_points_determine_plane_line_and_point_determine_plane_trapezoid_determines_plane_circle_points_not_always_determine_plane_l29_2985


namespace NUMINAMATH_CALUDE_min_sum_squares_l29_2908

theorem min_sum_squares (a b : ℝ) : 
  (∃! x, x^2 - 2*a*x + a^2 - a*b + 4 ≤ 0) → 
  (∀ c d : ℝ, (∃! x, x^2 - 2*c*x + c^2 - c*d + 4 ≤ 0) → a^2 + b^2 ≤ c^2 + d^2) →
  a^2 + b^2 = 8 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l29_2908


namespace NUMINAMATH_CALUDE_toothfairy_money_is_11_90_l29_2979

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The number of each type of coin Joan received -/
def coin_count : ℕ := 14

/-- The total value of coins Joan received from the toothfairy -/
def toothfairy_money : ℚ := coin_count * quarter_value + coin_count * half_dollar_value + coin_count * dime_value

theorem toothfairy_money_is_11_90 : toothfairy_money = 11.90 := by
  sorry

end NUMINAMATH_CALUDE_toothfairy_money_is_11_90_l29_2979


namespace NUMINAMATH_CALUDE_joe_haircut_time_l29_2957

/-- The time it takes to cut different types of hair and the number of haircuts Joe performed --/
structure HaircutData where
  womenTime : ℕ  -- Time to cut a woman's hair
  menTime : ℕ    -- Time to cut a man's hair
  kidsTime : ℕ   -- Time to cut a kid's hair
  womenCount : ℕ -- Number of women's haircuts
  menCount : ℕ   -- Number of men's haircuts
  kidsCount : ℕ  -- Number of kids' haircuts

/-- Calculate the total time Joe spent cutting hair --/
def totalHaircutTime (data : HaircutData) : ℕ :=
  data.womenTime * data.womenCount +
  data.menTime * data.menCount +
  data.kidsTime * data.kidsCount

/-- Theorem stating that Joe spent 255 minutes cutting hair --/
theorem joe_haircut_time :
  let data : HaircutData := {
    womenTime := 50,
    menTime := 15,
    kidsTime := 25,
    womenCount := 3,
    menCount := 2,
    kidsCount := 3
  }
  totalHaircutTime data = 255 := by
  sorry


end NUMINAMATH_CALUDE_joe_haircut_time_l29_2957


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_not_five_l29_2941

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits of a perfect square cannot be 5 -/
theorem sum_of_digits_of_square_not_five (n : ℕ) : 
  ∃ m : ℕ, n = m^2 → sumOfDigits n ≠ 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_not_five_l29_2941


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l29_2949

/-- Parabola function -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- Theorem stating the relationship between y-coordinates of three points on the parabola -/
theorem parabola_point_relationship : 
  let y₁ := f (-5)
  let y₂ := f 1
  let y₃ := f 12
  y₂ < y₁ ∧ y₁ < y₃ := by sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l29_2949


namespace NUMINAMATH_CALUDE_h1n1_vaccine_scientific_notation_l29_2965

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- Checks if two ScientificNotation values are equal up to a certain number of significant figures -/
def equalUpToSigFigs (a b : ScientificNotation) (sigFigs : ℕ) : Prop :=
  sorry

theorem h1n1_vaccine_scientific_notation :
  equalUpToSigFigs (toScientificNotation (25.06 * 1000000) 3) 
                   { coefficient := 2.51, exponent := 7, is_valid := by sorry } 3 := by
  sorry

end NUMINAMATH_CALUDE_h1n1_vaccine_scientific_notation_l29_2965


namespace NUMINAMATH_CALUDE_simplify_expression_l29_2981

theorem simplify_expression (b : ℝ) (h : b ≠ 2) :
  2 - 2 / (2 + b / (2 - b)) = 4 / (4 - b) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l29_2981


namespace NUMINAMATH_CALUDE_doughnuts_left_l29_2995

theorem doughnuts_left (total_doughnuts : ℕ) (staff_count : ℕ) (doughnuts_per_staff : ℕ) :
  total_doughnuts = 50 →
  staff_count = 19 →
  doughnuts_per_staff = 2 →
  total_doughnuts - (staff_count * doughnuts_per_staff) = 12 :=
by sorry

end NUMINAMATH_CALUDE_doughnuts_left_l29_2995


namespace NUMINAMATH_CALUDE_first_year_rate_is_two_percent_l29_2901

/-- Given an initial amount, time period, second year interest rate, and final amount,
    calculate the first year interest rate. -/
def calculate_first_year_rate (initial_amount : ℝ) (time_period : ℕ) 
                               (second_year_rate : ℝ) (final_amount : ℝ) : ℝ :=
  sorry

/-- Theorem: Given the specific conditions, the first year interest rate is 2% -/
theorem first_year_rate_is_two_percent :
  let initial_amount : ℝ := 5000
  let time_period : ℕ := 2
  let second_year_rate : ℝ := 0.03
  let final_amount : ℝ := 5253
  calculate_first_year_rate initial_amount time_period second_year_rate final_amount = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_first_year_rate_is_two_percent_l29_2901


namespace NUMINAMATH_CALUDE_circles_position_l29_2998

theorem circles_position (r₁ r₂ : ℝ) (h₁ : r₁ * r₂ = 3) (h₂ : r₁ + r₂ = 5) (h₃ : (r₁ - r₂)^2 = 13/4) :
  let d := 3
  r₁ + r₂ > d ∧ |r₁ - r₂| > d :=
by sorry

end NUMINAMATH_CALUDE_circles_position_l29_2998


namespace NUMINAMATH_CALUDE_star_equation_solution_l29_2910

-- Define the star operation
noncomputable def star (x y : ℝ) : ℝ :=
  x + Real.sqrt (y + Real.sqrt (y + Real.sqrt y))

-- State the theorem
theorem star_equation_solution :
  ∃ h : ℝ, star 3 h = 8 ∧ h = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l29_2910


namespace NUMINAMATH_CALUDE_last_digit_of_difference_l29_2977

/-- A function that returns the last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- A function that checks if a number is a power of 10 -/
def isPowerOfTen (n : ℕ) : Prop := ∃ k : ℕ, n = 10^k

theorem last_digit_of_difference (p q : ℕ) 
  (hp : p > 0) (hq : q > 0) 
  (hpq : p > q)
  (hpLast : lastDigit p ≠ 0) 
  (hqLast : lastDigit q ≠ 0)
  (hProduct : isPowerOfTen (p * q)) : 
  lastDigit (p - q) ≠ 5 := by
sorry

end NUMINAMATH_CALUDE_last_digit_of_difference_l29_2977


namespace NUMINAMATH_CALUDE_max_value_implies_a_l29_2930

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + a

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∃ x ∈ Set.Icc 0 4, f a x = 3) ∧
  (∀ x ∈ Set.Icc 0 4, f a x ≤ 3) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l29_2930


namespace NUMINAMATH_CALUDE_diagonals_bisect_implies_parallelogram_l29_2968

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- A diagonal of a quadrilateral is a line segment connecting two non-adjacent vertices. -/
def Quadrilateral.diagonal (q : Quadrilateral) (i j : Fin 4) : ℝ × ℝ → ℝ × ℝ :=
  sorry

/-- Two line segments bisect each other if they intersect at their midpoints. -/
def bisect (seg1 seg2 : ℝ × ℝ → ℝ × ℝ) : Prop :=
  sorry

/-- A parallelogram is a quadrilateral with two pairs of parallel sides. -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  sorry

/-- If the diagonals of a quadrilateral bisect each other, then it is a parallelogram. -/
theorem diagonals_bisect_implies_parallelogram (q : Quadrilateral) :
  (∃ (i j k l : Fin 4), i ≠ j ∧ k ≠ l ∧ 
    bisect (q.diagonal i k) (q.diagonal j l)) →
  is_parallelogram q :=
sorry

end NUMINAMATH_CALUDE_diagonals_bisect_implies_parallelogram_l29_2968


namespace NUMINAMATH_CALUDE_no_base_with_final_digit_one_l29_2927

theorem no_base_with_final_digit_one : 
  ∀ b : ℕ, 2 ≤ b ∧ b ≤ 9 → ¬(∃ k : ℕ, 360 = k * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_base_with_final_digit_one_l29_2927


namespace NUMINAMATH_CALUDE_average_after_removal_l29_2975

def originalList : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def removedNumber : ℕ := 1

def remainingList : List ℕ := originalList.filter (· ≠ removedNumber)

theorem average_after_removal :
  (remainingList.sum : ℚ) / remainingList.length = 15/2 := by sorry

end NUMINAMATH_CALUDE_average_after_removal_l29_2975


namespace NUMINAMATH_CALUDE_ian_painted_faces_l29_2951

/-- The number of faces of a cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The number of cuboids Ian painted -/
def number_of_cuboids : ℕ := 8

/-- The total number of faces painted by Ian -/
def total_faces_painted : ℕ := faces_per_cuboid * number_of_cuboids

theorem ian_painted_faces :
  total_faces_painted = 48 :=
by sorry

end NUMINAMATH_CALUDE_ian_painted_faces_l29_2951


namespace NUMINAMATH_CALUDE_sum_of_squares_remainder_l29_2991

theorem sum_of_squares_remainder (a b c d e : ℕ) (ha : a = 445876) (hb : b = 985420) (hc : c = 215546) (hd : d = 656452) (he : e = 387295) :
  (a^2 + b^2 + c^2 + d^2 + e^2) % 8 = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_remainder_l29_2991


namespace NUMINAMATH_CALUDE_john_ate_12_ounces_l29_2911

/-- The amount of steak John ate given the original weight, burned portion, and eating percentage -/
def steak_eaten (original_weight : ℝ) (burned_portion : ℝ) (eating_percentage : ℝ) : ℝ :=
  (1 - burned_portion) * original_weight * eating_percentage

/-- Theorem stating that John ate 12 ounces of steak -/
theorem john_ate_12_ounces : 
  let original_weight : ℝ := 30
  let burned_portion : ℝ := 1/2
  let eating_percentage : ℝ := 0.8
  steak_eaten original_weight burned_portion eating_percentage = 12 := by
  sorry

end NUMINAMATH_CALUDE_john_ate_12_ounces_l29_2911


namespace NUMINAMATH_CALUDE_complex_multiplication_l29_2931

theorem complex_multiplication : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 → 
  (2 + Complex.I) * (1 - 3 * Complex.I) = 5 - 5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l29_2931


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l29_2956

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 = 80) :
  a 1 + a 13 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l29_2956


namespace NUMINAMATH_CALUDE_triangle_side_length_l29_2921

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  A = π / 3 →  -- 60° in radians
  B = π / 4 →  -- 45° in radians
  a = 3 →
  b = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l29_2921


namespace NUMINAMATH_CALUDE_martha_coffee_spending_l29_2996

/-- The cost of an iced coffee that satisfies Martha's coffee spending reduction --/
def iced_coffee_cost : ℚ := by sorry

/-- Proves that the cost of an iced coffee is $2.00 --/
theorem martha_coffee_spending :
  let latte_cost : ℚ := 4
  let lattes_per_week : ℕ := 5
  let iced_coffees_per_week : ℕ := 3
  let weeks_per_year : ℕ := 52
  let spending_reduction_ratio : ℚ := 1 / 4
  let spending_reduction_amount : ℚ := 338

  let annual_latte_spending : ℚ := latte_cost * lattes_per_week * weeks_per_year
  let annual_iced_coffee_spending : ℚ := iced_coffee_cost * iced_coffees_per_week * weeks_per_year
  let total_annual_spending : ℚ := annual_latte_spending + annual_iced_coffee_spending

  (1 - spending_reduction_ratio) * total_annual_spending = total_annual_spending - spending_reduction_amount →
  iced_coffee_cost = 2 := by sorry

end NUMINAMATH_CALUDE_martha_coffee_spending_l29_2996


namespace NUMINAMATH_CALUDE_birthday_money_calculation_l29_2980

def playstation_cost : ℝ := 500
def christmas_money : ℝ := 150
def game_price : ℝ := 7.5
def games_to_sell : ℕ := 20

theorem birthday_money_calculation :
  let total_from_games : ℝ := game_price * (games_to_sell : ℝ)
  let remaining_money_needed : ℝ := playstation_cost - christmas_money - total_from_games
  remaining_money_needed = 200 := by sorry

end NUMINAMATH_CALUDE_birthday_money_calculation_l29_2980


namespace NUMINAMATH_CALUDE_stadium_height_l29_2942

/-- The height of a rectangular stadium given its length, width, and the length of the longest pole that can fit diagonally. -/
theorem stadium_height (length width diagonal : ℝ) (h1 : length = 24) (h2 : width = 18) (h3 : diagonal = 34) :
  Real.sqrt (diagonal^2 - length^2 - width^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_stadium_height_l29_2942


namespace NUMINAMATH_CALUDE_stream_speed_l29_2938

/-- Proves that the speed of a stream is 8 kmph given the conditions of a boat's travel --/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) 
  (h1 : boat_speed = 24)
  (h2 : downstream_distance = 64)
  (h3 : upstream_distance = 32)
  (h4 : downstream_distance / (boat_speed + x) = upstream_distance / (boat_speed - x)) :
  x = 8 := by
  sorry

#check stream_speed

end NUMINAMATH_CALUDE_stream_speed_l29_2938


namespace NUMINAMATH_CALUDE_alyssa_total_games_l29_2904

/-- The total number of soccer games Alyssa will attend -/
def total_games (this_year last_year next_year : ℕ) : ℕ :=
  this_year + last_year + next_year

/-- Proof that Alyssa will attend 39 soccer games in total -/
theorem alyssa_total_games : 
  ∃ (this_year last_year next_year : ℕ),
    this_year = 11 ∧ 
    last_year = 13 ∧ 
    next_year = 15 ∧ 
    total_games this_year last_year next_year = 39 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_total_games_l29_2904


namespace NUMINAMATH_CALUDE_base_b_square_l29_2934

theorem base_b_square (b : ℕ) : b > 0 → (3 * b + 3)^2 = b^3 + 2 * b^2 + 3 * b ↔ b = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_b_square_l29_2934


namespace NUMINAMATH_CALUDE_sticker_distribution_l29_2972

/-- Represents the share of stickers each winner should receive -/
structure Share where
  al : Rat
  bert : Rat
  carl : Rat
  dan : Rat

/-- Calculates the remaining fraction of stickers after all winners have taken their perceived shares -/
def remaining_stickers (s : Share) : Rat :=
  let total := 1
  let bert_sees := total - s.al
  let carl_sees := bert_sees - (s.bert * bert_sees)
  let dan_sees := carl_sees - (s.carl * carl_sees)
  total - (s.al + s.bert * bert_sees + s.carl * carl_sees + s.dan * dan_sees)

/-- The theorem to be proved -/
theorem sticker_distribution (s : Share) 
  (h1 : s.al = 4/10)
  (h2 : s.bert = 3/10)
  (h3 : s.carl = 2/10)
  (h4 : s.dan = 1/10) :
  remaining_stickers s = 2844/10000 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l29_2972


namespace NUMINAMATH_CALUDE_expand_expression_l29_2917

theorem expand_expression (y : ℝ) : (11 * y + 18) * (3 * y) = 33 * y^2 + 54 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l29_2917


namespace NUMINAMATH_CALUDE_gretchen_to_rachelle_ratio_l29_2983

def pennies_problem (rachelle gretchen rocky : ℕ) : Prop :=
  rachelle = 180 ∧
  rocky = gretchen / 3 ∧
  rachelle + gretchen + rocky = 300

theorem gretchen_to_rachelle_ratio :
  ∀ rachelle gretchen rocky : ℕ,
  pennies_problem rachelle gretchen rocky →
  gretchen * 2 = rachelle :=
by
  sorry

end NUMINAMATH_CALUDE_gretchen_to_rachelle_ratio_l29_2983


namespace NUMINAMATH_CALUDE_age_difference_proof_l29_2918

/-- Proves that the difference between twice John's current age and Tim's age is 15 years -/
theorem age_difference_proof (james_age_past : ℕ) (john_age_past : ℕ) (tim_age : ℕ) 
  (h1 : james_age_past = 23)
  (h2 : john_age_past = 35)
  (h3 : tim_age = 79)
  (h4 : ∃ (x : ℕ), tim_age + x = 2 * (john_age_past + (john_age_past - james_age_past))) :
  2 * (john_age_past + (john_age_past - james_age_past)) - tim_age = 15 := by
  sorry


end NUMINAMATH_CALUDE_age_difference_proof_l29_2918


namespace NUMINAMATH_CALUDE_fourth_sample_seat_number_l29_2940

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  first_sample : ℕ

/-- Calculates the nth element in a systematic sample -/
def nth_sample (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first_sample + (n - 1) * (s.population_size / s.sample_size)

theorem fourth_sample_seat_number 
  (sample : SystematicSample)
  (h1 : sample.population_size = 56)
  (h2 : sample.sample_size = 4)
  (h3 : sample.first_sample = 4)
  (h4 : nth_sample sample 2 = 18)
  (h5 : nth_sample sample 3 = 46) :
  nth_sample sample 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_fourth_sample_seat_number_l29_2940


namespace NUMINAMATH_CALUDE_weight_of_b_l29_2924

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 42)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) :
  b = 40 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l29_2924


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l29_2990

def a : Fin 2 → ℝ := ![1, -3]
def b : Fin 2 → ℝ := ![2, 1]

theorem parallel_vectors_k_value : 
  ∃ (k : ℝ), ∃ (c : ℝ), c ≠ 0 ∧ 
    (∀ i : Fin 2, (k * a i + b i) = c * (a i - 2 * b i)) → 
    k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l29_2990


namespace NUMINAMATH_CALUDE_exists_decreasing_function_always_ge_one_l29_2937

theorem exists_decreasing_function_always_ge_one :
  ∃ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f x > f y) ∧ (∀ x : ℝ, f x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_decreasing_function_always_ge_one_l29_2937


namespace NUMINAMATH_CALUDE_burn_time_3x5_grid_l29_2913

/-- Represents a grid of toothpicks -/
structure ToothpickGrid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the time taken for a toothpick grid to burn completely -/
def burnTime (grid : ToothpickGrid) (burnTimePerToothpick : ℕ) : ℕ :=
  sorry

/-- Theorem stating that a 3x5 grid burns in 65 seconds -/
theorem burn_time_3x5_grid :
  let grid := ToothpickGrid.mk 3 5
  let burnTimePerToothpick := 10
  burnTime grid burnTimePerToothpick = 65 :=
sorry

end NUMINAMATH_CALUDE_burn_time_3x5_grid_l29_2913


namespace NUMINAMATH_CALUDE_divisibility_condition_l29_2970

theorem divisibility_condition (n : ℕ) : n ≥ 1 → (n ^ 2 ∣ 2 ^ n + 1) ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l29_2970


namespace NUMINAMATH_CALUDE_complement_intersection_equals_five_l29_2902

def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

theorem complement_intersection_equals_five :
  (I \ A) ∩ B = {5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_five_l29_2902


namespace NUMINAMATH_CALUDE_oblique_drawing_area_relation_original_triangle_area_l29_2905

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Represents the oblique drawing method transformation -/
def obliqueDrawing (t : Triangle) : Triangle := sorry

theorem oblique_drawing_area_relation (t : Triangle) :
  area (obliqueDrawing t) / area t = Real.sqrt 2 / 4 := sorry

/-- The main theorem proving the area of the original triangle -/
theorem original_triangle_area (t : Triangle) 
  (h1 : area (obliqueDrawing t) = 3) :
  area t = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_oblique_drawing_area_relation_original_triangle_area_l29_2905


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l29_2933

theorem quadratic_inequality_solution_set :
  {x : ℝ | 3 * x^2 - 5 * x - 2 < 0} = {x : ℝ | -1/3 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l29_2933


namespace NUMINAMATH_CALUDE_lindas_savings_l29_2947

theorem lindas_savings (savings : ℝ) : 
  (3 / 4 : ℝ) * savings + 210 = savings → savings = 840 :=
by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l29_2947


namespace NUMINAMATH_CALUDE_arrangements_theorem_l29_2967

/-- The number of arrangements of 5 people in a row with exactly 1 person between A and B -/
def arrangements_count : ℕ := 36

/-- The total number of people in the row -/
def total_people : ℕ := 5

/-- The number of people between A and B -/
def people_between : ℕ := 1

/-- Theorem stating that the number of arrangements is 36 -/
theorem arrangements_theorem :
  (arrangements_count = 36) ∧
  (total_people = 5) ∧
  (people_between = 1) :=
sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l29_2967


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l29_2935

-- Define the types for lines and planes
def Line : Type := ℝ × ℝ × ℝ → Prop
def Plane : Type := ℝ × ℝ × ℝ → Prop

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p1 : Plane) (p2 : Plane) : Prop := sorry

-- State the theorem
theorem line_plane_perpendicularity (l : Line) (α β : Plane) :
  perpendicular l α → parallel l β → perpendicular_planes α β := by sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l29_2935


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l29_2923

/-- A hyperbola with equation x^2 + (k-1)y^2 = k+1 and foci on the x-axis -/
structure Hyperbola (k : ℝ) where
  eq : ∀ (x y : ℝ), x^2 + (k-1)*y^2 = k+1
  foci_on_x : True  -- This is a placeholder for the foci condition

/-- The range of k values for which the hyperbola is well-defined -/
def valid_k_range : Set ℝ := {k | ∃ h : Hyperbola k, True}

theorem hyperbola_k_range :
  valid_k_range = Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l29_2923


namespace NUMINAMATH_CALUDE_count_numbers_with_seven_800_l29_2969

def contains_seven (n : Nat) : Bool :=
  let digits := n.digits 10
  7 ∈ digits

def count_numbers_with_seven (upper_bound : Nat) : Nat :=
  (List.range upper_bound).filter contains_seven |>.length

theorem count_numbers_with_seven_800 :
  count_numbers_with_seven 800 = 152 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_seven_800_l29_2969


namespace NUMINAMATH_CALUDE_max_value_constraint_l29_2948

theorem max_value_constraint (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) :
  a*b + b*c + c*d + d*a + a*c + 4*b*d ≤ 5/2 := by
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l29_2948


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l29_2999

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∃ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l29_2999


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l29_2986

/-- 
Given an ellipse with equation mx^2 + ny^2 + mn = 0, where m < n < 0,
prove that the coordinates of its foci are (0, ±√(n-m)).
-/
theorem ellipse_foci_coordinates 
  (m n : ℝ) 
  (h1 : m < n) 
  (h2 : n < 0) : 
  let equation := fun (x y : ℝ) => m * x^2 + n * y^2 + m * n
  ∃ c : ℝ, c > 0 ∧ 
    (∀ x y : ℝ, equation x y = 0 → 
      ((x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)) ∧ 
      c^2 = n - m) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l29_2986


namespace NUMINAMATH_CALUDE_percentage_calculation_l29_2973

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 100 → 
  (P / 100) * (3 / 5 * N) = 36 → 
  P = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l29_2973


namespace NUMINAMATH_CALUDE_elf_nuts_problem_l29_2932

theorem elf_nuts_problem (nuts : Fin 10 → ℕ) 
  (h1 : (nuts 0) + (nuts 2) = 110)
  (h2 : (nuts 1) + (nuts 3) = 120)
  (h3 : (nuts 2) + (nuts 4) = 130)
  (h4 : (nuts 3) + (nuts 5) = 140)
  (h5 : (nuts 4) + (nuts 6) = 150)
  (h6 : (nuts 5) + (nuts 7) = 160)
  (h7 : (nuts 6) + (nuts 8) = 170)
  (h8 : (nuts 7) + (nuts 9) = 180)
  (h9 : (nuts 8) + (nuts 0) = 190)
  (h10 : (nuts 9) + (nuts 1) = 200) :
  nuts 5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_elf_nuts_problem_l29_2932


namespace NUMINAMATH_CALUDE_girls_in_class_l29_2971

theorem girls_in_class (boys : ℕ) (ways : ℕ) : boys = 15 → ways = 1050 → ∃ girls : ℕ,
  girls * (boys.choose 2) = ways ∧ girls = 10 := by sorry

end NUMINAMATH_CALUDE_girls_in_class_l29_2971


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_1000_l29_2984

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits of all numbers from 1 to 1000 is 13501 -/
theorem sum_of_digits_up_to_1000 : sumOfDigitsUpTo 1000 = 13501 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_1000_l29_2984


namespace NUMINAMATH_CALUDE_no_integers_product_sum_20182017_l29_2993

theorem no_integers_product_sum_20182017 : ¬∃ (a b : ℤ), a * b * (a + b) = 20182017 := by
  sorry

end NUMINAMATH_CALUDE_no_integers_product_sum_20182017_l29_2993


namespace NUMINAMATH_CALUDE_basketball_shooting_test_l29_2945

-- Define the probabilities of making a basket for students A and B
def prob_A : ℚ := 1/2
def prob_B : ℚ := 2/3

-- Define the number of shots for Part I
def shots_part_I : ℕ := 3

-- Define the number of chances for Part II
def chances_part_II : ℕ := 4

-- Define the function to calculate the probability of exactly k successes in n trials
def binomial_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- Define the probability of student A meeting the standard in Part I
def prob_A_meets_standard : ℚ :=
  binomial_probability shots_part_I 2 prob_A + binomial_probability shots_part_I 3 prob_A

-- Define the probability distribution of X (number of shots taken by B) in Part II
def prob_X (x : ℕ) : ℚ :=
  if x = 2 then prob_B^2
  else if x = 3 then prob_B * (1-prob_B) * prob_B + prob_B^2 * (1-prob_B) + (1-prob_B)^3
  else if x = 4 then (1-prob_B) * prob_B^2 + prob_B * (1-prob_B) * prob_B
  else 0

-- Define the expected value of X
def expected_X : ℚ :=
  2 * prob_X 2 + 3 * prob_X 3 + 4 * prob_X 4

-- Theorem statement
theorem basketball_shooting_test :
  prob_A_meets_standard = 1/2 ∧ expected_X = 25/9 := by sorry

end NUMINAMATH_CALUDE_basketball_shooting_test_l29_2945


namespace NUMINAMATH_CALUDE_hyperbola_condition_l29_2955

-- Define the equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (k - 5) - y^2 / (k + 2) = 1 ∧ 
  (k - 5 > 0 ∧ k + 2 > 0)

-- State the theorem
theorem hyperbola_condition (k : ℝ) :
  (is_hyperbola k → k > 5) ∧ 
  ¬(k > 5 → is_hyperbola k) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l29_2955


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l29_2974

/-- Given a square tile with side length 10 dm, containing four identical rectangles and a small square,
    where the perimeter of the small square is five times smaller than the perimeter of the entire square,
    prove that the dimensions of the rectangles are 4 dm × 6 dm. -/
theorem rectangle_dimensions (tile_side : ℝ) (small_square_side : ℝ) (rect_short_side : ℝ) (rect_long_side : ℝ) :
  tile_side = 10 →
  small_square_side * 4 = tile_side * 4 / 5 →
  tile_side = small_square_side + 2 * rect_short_side →
  tile_side = rect_short_side + rect_long_side →
  rect_short_side = 4 ∧ rect_long_side = 6 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l29_2974


namespace NUMINAMATH_CALUDE_barrel_division_exists_l29_2959

/-- Represents the fill state of a barrel -/
inductive BarrelState
  | Empty
  | Half
  | Full

/-- Represents a distribution of barrels to an heir -/
structure Distribution where
  empty : Nat
  half : Nat
  full : Nat

/-- Calculates the total wine in a distribution -/
def wineAmount (d : Distribution) : Nat :=
  d.full * 2 + d.half

/-- Checks if a distribution is valid (8 barrels total) -/
def isValidDistribution (d : Distribution) : Prop :=
  d.empty + d.half + d.full = 8

/-- Represents a complete division of barrels among three heirs -/
structure BarrelDivision where
  heir1 : Distribution
  heir2 : Distribution
  heir3 : Distribution

/-- Checks if a barrel division is valid -/
def isValidDivision (div : BarrelDivision) : Prop :=
  isValidDistribution div.heir1 ∧
  isValidDistribution div.heir2 ∧
  isValidDistribution div.heir3 ∧
  div.heir1.empty + div.heir2.empty + div.heir3.empty = 8 ∧
  div.heir1.half + div.heir2.half + div.heir3.half = 8 ∧
  div.heir1.full + div.heir2.full + div.heir3.full = 8 ∧
  wineAmount div.heir1 = wineAmount div.heir2 ∧
  wineAmount div.heir2 = wineAmount div.heir3

theorem barrel_division_exists : ∃ (div : BarrelDivision), isValidDivision div := by
  sorry

end NUMINAMATH_CALUDE_barrel_division_exists_l29_2959


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l29_2909

/-- Given a nonreal cube root of unity ω, prove that (ω - 2ω^2 + 2)^4 + (2 + 2ω - ω^2)^4 = -257 -/
theorem cube_root_unity_sum (ω : ℂ) : 
  ω ≠ 1 → ω^3 = 1 → (ω - 2*ω^2 + 2)^4 + (2 + 2*ω - ω^2)^4 = -257 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l29_2909


namespace NUMINAMATH_CALUDE_first_number_in_sequence_l29_2926

def sequence_product (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → n ≤ 10 → a n = a (n-1) * a (n-2)

theorem first_number_in_sequence 
  (a : ℕ → ℚ) 
  (h_seq : sequence_product a) 
  (h_8 : a 8 = 36) 
  (h_9 : a 9 = 324) 
  (h_10 : a 10 = 11664) : 
  a 1 = 59049 / 65536 := by
sorry

end NUMINAMATH_CALUDE_first_number_in_sequence_l29_2926


namespace NUMINAMATH_CALUDE_tissue_box_price_l29_2954

-- Define the quantities and prices
def toilet_paper_rolls : ℕ := 10
def paper_towel_rolls : ℕ := 7
def tissue_boxes : ℕ := 3
def toilet_paper_price : ℚ := 1.5
def paper_towel_price : ℚ := 2
def total_cost : ℚ := 35

-- Theorem to prove
theorem tissue_box_price : 
  (total_cost - (toilet_paper_rolls * toilet_paper_price + paper_towel_rolls * paper_towel_price)) / tissue_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_tissue_box_price_l29_2954


namespace NUMINAMATH_CALUDE_equation_with_72_l29_2966

/-- The first term of the nth equation in the sequence -/
def first_term (n : ℕ) : ℕ := 2 * n^2

/-- The equation number in which 72 appears as the first term -/
theorem equation_with_72 : {k : ℕ | first_term k = 72} = {6} := by sorry

end NUMINAMATH_CALUDE_equation_with_72_l29_2966


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l29_2963

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_ne_zero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_first_term
  (seq : ArithmeticSequence)
  (h1 : seq.a 2 * seq.a 3 = seq.a 4 * seq.a 5)
  (h2 : sum_n seq 9 = 1) :
  seq.a 1 = -5/27 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l29_2963


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_2_l29_2919

theorem factorization_of_2x_squared_minus_2 (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_2_l29_2919


namespace NUMINAMATH_CALUDE_sum_of_union_equals_31_l29_2922

def A : Finset ℕ := {2, 0, 1, 8}

def B : Finset ℕ := Finset.image (· * 2) A

theorem sum_of_union_equals_31 : (A ∪ B).sum id = 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_union_equals_31_l29_2922


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l29_2916

theorem quadratic_root_condition (a b c : ℝ) (h : a ≠ 0) :
  (∃ r s : ℝ, r * s = -1 ∧ a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) →
  c = -a :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l29_2916


namespace NUMINAMATH_CALUDE_platform_length_l29_2950

/-- The length of a platform given train specifications -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 500)
  (h2 : time_platform = 45)
  (h3 : time_pole = 25) :
  let train_speed := train_length / time_pole
  let platform_length := train_speed * time_platform - train_length
  platform_length = 400 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l29_2950
