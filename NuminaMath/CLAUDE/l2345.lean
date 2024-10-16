import Mathlib

namespace NUMINAMATH_CALUDE_machinery_expense_l2345_234573

/-- Proves that the amount spent on machinery is $1000 --/
theorem machinery_expense (total : ℝ) (raw_materials : ℝ) (cash_percentage : ℝ) :
  total = 5714.29 →
  raw_materials = 3000 →
  cash_percentage = 0.30 →
  ∃ (machinery : ℝ),
    machinery = 1000 ∧
    total = raw_materials + machinery + (cash_percentage * total) :=
by
  sorry


end NUMINAMATH_CALUDE_machinery_expense_l2345_234573


namespace NUMINAMATH_CALUDE_nine_five_dollar_bills_equal_45_l2345_234511

/-- Calculates the total amount of money given the number of five-dollar bills -/
def total_money (num_bills : ℕ) : ℕ := 5 * num_bills

/-- Theorem stating that 9 five-dollar bills equal $45 -/
theorem nine_five_dollar_bills_equal_45 :
  total_money 9 = 45 := by sorry

end NUMINAMATH_CALUDE_nine_five_dollar_bills_equal_45_l2345_234511


namespace NUMINAMATH_CALUDE_rectangle_area_l2345_234559

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : ℝ × ℝ)

-- Define the diagonal BD
def diagonal (rect : Rectangle) : ℝ × ℝ := (rect.B.1 - rect.D.1, rect.B.2 - rect.D.2)

-- Define points E and F on the diagonal
structure PerpendicularPoints (rect : Rectangle) :=
  (E F : ℝ × ℝ)

-- Define the perpendicularity condition
def isPerpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- State the theorem
theorem rectangle_area (rect : Rectangle) (perp : PerpendicularPoints rect) :
  isPerpendicular (rect.A.1 - perp.E.1, rect.A.2 - perp.E.2) (diagonal rect) →
  isPerpendicular (rect.C.1 - perp.F.1, rect.C.2 - perp.F.2) (diagonal rect) →
  (perp.E.1 - rect.B.1)^2 + (perp.E.2 - rect.B.2)^2 = 1 →
  (perp.F.1 - perp.E.1)^2 + (perp.F.2 - perp.E.2)^2 = 4 →
  (rect.B.1 - rect.A.1) * (rect.D.2 - rect.A.2) = 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2345_234559


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l2345_234570

/-- A trapezoid with angles in arithmetic sequence -/
structure ArithmeticTrapezoid where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference

/-- The properties of an arithmetic trapezoid -/
def ArithmeticTrapezoid.valid (t : ArithmeticTrapezoid) : Prop :=
  -- Sum of interior angles is 360°
  t.a + (t.a + t.d) + (t.a + 2*t.d) + (t.a + 3*t.d) = 360 ∧
  -- Largest angle is 150°
  t.a + 3*t.d = 150

theorem smallest_angle_measure (t : ArithmeticTrapezoid) (h : t.valid) :
  t.a = 15 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l2345_234570


namespace NUMINAMATH_CALUDE_rectangle_area_l2345_234548

theorem rectangle_area (c d w : ℝ) (h1 : w > 0) (h2 : w + 3 > w) 
  (h3 : (c + d)^2 = w^2 + (w + 3)^2) : w * (w + 3) = w^2 + 3*w := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2345_234548


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l2345_234557

theorem quadratic_roots_difference_squared :
  ∀ α β : ℝ, 
  (α^2 - 3*α + 1 = 0) → 
  (β^2 - 3*β + 1 = 0) → 
  (α ≠ β) →
  (α - β)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l2345_234557


namespace NUMINAMATH_CALUDE_school_population_l2345_234556

/-- Given a school with boys and girls, prove the total number of students is 900 -/
theorem school_population (total boys girls : ℕ) : 
  total = boys + girls →
  boys = 90 →
  girls = (90 * total) / 100 →
  total = 900 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l2345_234556


namespace NUMINAMATH_CALUDE_xyz_sum_l2345_234532

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : x * z + y = 47) : 
  x + y + z = 48 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l2345_234532


namespace NUMINAMATH_CALUDE_spending_recorded_as_negative_l2345_234590

/-- Represents a WeChat payment record -/
structure WeChatPayment where
  amount : ℝ

/-- Records a receipt in WeChat payment system -/
def recordReceipt (x : ℝ) : WeChatPayment :=
  ⟨x⟩

/-- Records spending in WeChat payment system -/
def recordSpending (x : ℝ) : WeChatPayment :=
  ⟨-x⟩

theorem spending_recorded_as_negative :
  recordSpending 10.6 = WeChatPayment.mk (-10.6) := by
  sorry

end NUMINAMATH_CALUDE_spending_recorded_as_negative_l2345_234590


namespace NUMINAMATH_CALUDE_derivative_of_cube_root_l2345_234597

theorem derivative_of_cube_root (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.sqrt (x^3)) x = (3/2) * Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_cube_root_l2345_234597


namespace NUMINAMATH_CALUDE_ellipse_intersection_ratio_l2345_234591

/-- An ellipse intersecting with a line -/
structure EllipseIntersection where
  m : ℝ
  n : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  ellipse_eq₁ : m * x₁^2 + n * y₁^2 = 1
  ellipse_eq₂ : m * x₂^2 + n * y₂^2 = 1
  line_eq₁ : y₁ = 1 - x₁
  line_eq₂ : y₂ = 1 - x₂

/-- The theorem stating the relationship between m and n -/
theorem ellipse_intersection_ratio (e : EllipseIntersection) 
  (h : (y₁ + y₂) / (x₁ + x₂) = Real.sqrt 2) : 
  e.m / e.n = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_ratio_l2345_234591


namespace NUMINAMATH_CALUDE_tennis_uniform_numbers_l2345_234560

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem tennis_uniform_numbers 
  (e f g h : ℕ) 
  (e_birthday today_date g_birthday : ℕ)
  (h_two_digit_prime : is_two_digit_prime e ∧ is_two_digit_prime f ∧ is_two_digit_prime g ∧ is_two_digit_prime h)
  (h_sum_all : e + f + g + h = e_birthday)
  (h_sum_ef : e + f = today_date)
  (h_sum_gf : g + f = g_birthday)
  (h_sum_hg : h + g = e_birthday) :
  h = 19 := by
  sorry

end NUMINAMATH_CALUDE_tennis_uniform_numbers_l2345_234560


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_five_sqrt_two_over_two_l2345_234584

theorem sqrt_sum_equals_five_sqrt_two_over_two :
  Real.sqrt 8 + Real.sqrt (1/2) = (5 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_five_sqrt_two_over_two_l2345_234584


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l2345_234504

theorem fraction_sum_equals_one (x y : ℝ) 
  (h1 : 3 * x + 2 * y ≠ 0) (h2 : 3 * x - 2 * y ≠ 0) : 
  (7 * x - 5 * y) / (3 * x + 2 * y) + 
  (5 * x - 8 * y) / (3 * x - 2 * y) - 
  (x - 9 * y) / (3 * x + 2 * y) - 
  (8 * x - 10 * y) / (3 * x - 2 * y) = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l2345_234504


namespace NUMINAMATH_CALUDE_ring_cost_l2345_234506

theorem ring_cost (total_cost : ℝ) (num_rings : ℕ) (h1 : total_cost = 24) (h2 : num_rings = 2) :
  total_cost / num_rings = 12 :=
by sorry

end NUMINAMATH_CALUDE_ring_cost_l2345_234506


namespace NUMINAMATH_CALUDE_average_weight_increase_l2345_234594

/-- Proves that the average weight increase is 200 grams when a 45 kg student leaves a group of 60 students, resulting in a new average of 57 kg for the remaining 59 students. -/
theorem average_weight_increase (initial_count : ℕ) (left_weight : ℝ) (remaining_count : ℕ) (new_average : ℝ) : 
  initial_count = 60 → 
  left_weight = 45 → 
  remaining_count = 59 → 
  new_average = 57 → 
  (new_average - (initial_count * new_average - left_weight) / initial_count) * 1000 = 200 := by
  sorry

#check average_weight_increase

end NUMINAMATH_CALUDE_average_weight_increase_l2345_234594


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_pattern_l2345_234529

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

def consecutive_pair (a b : ℕ) : Prop := b = a + 1

theorem least_integer_with_divisibility_pattern : 
  ∃ (p q : ℕ), 18 ≤ p ∧ p < 25 ∧ consecutive_pair p q ∧
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 30 ∧ k ≠ p ∧ k ≠ q → is_divisible 659375723440 k) ∧
  ¬(is_divisible 659375723440 p) ∧ ¬(is_divisible 659375723440 q) ∧
  (∀ (n : ℕ), n < 659375723440 → 
    ¬(∃ (r s : ℕ), 18 ≤ r ∧ r < 25 ∧ consecutive_pair r s ∧
    (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 30 ∧ k ≠ r ∧ k ≠ s → is_divisible n k) ∧
    ¬(is_divisible n r) ∧ ¬(is_divisible n s))) :=
by sorry

end NUMINAMATH_CALUDE_least_integer_with_divisibility_pattern_l2345_234529


namespace NUMINAMATH_CALUDE_quiz_win_probability_l2345_234513

def num_questions : ℕ := 4
def num_choices : ℕ := 4
def min_correct : ℕ := 3

def prob_correct_one : ℚ := 1 / num_choices

def prob_all_correct : ℚ := prob_correct_one ^ num_questions

def prob_three_correct : ℚ := num_questions * (prob_correct_one ^ 3) * (1 - prob_correct_one)

theorem quiz_win_probability :
  prob_all_correct + prob_three_correct = 13 / 256 := by
  sorry

end NUMINAMATH_CALUDE_quiz_win_probability_l2345_234513


namespace NUMINAMATH_CALUDE_quiz_logic_l2345_234540

theorem quiz_logic (x y z w u v : ℝ) : 
  (x > y → z < w) → 
  (z > w → u < v) → 
  ¬((x < y → u < v) ∨ 
    (u < v → x < y) ∨ 
    (u > v → x > y) ∨ 
    (x > y → u > v)) := by
  sorry

end NUMINAMATH_CALUDE_quiz_logic_l2345_234540


namespace NUMINAMATH_CALUDE_unique_solution_cube_difference_prime_l2345_234572

theorem unique_solution_cube_difference_prime (x y z : ℕ+) : 
  Nat.Prime y.val ∧ 
  ¬(3 ∣ z.val) ∧ 
  ¬(y.val ∣ z.val) ∧ 
  x.val^3 - y.val^3 = z.val^2 →
  x = 8 ∧ y = 7 ∧ z = 13 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_cube_difference_prime_l2345_234572


namespace NUMINAMATH_CALUDE_min_value_of_sum_absolute_differences_l2345_234526

theorem min_value_of_sum_absolute_differences (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, |x - 2| + |x - 3| + |x - 4| < a) → a > 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_absolute_differences_l2345_234526


namespace NUMINAMATH_CALUDE_bird_families_flew_away_l2345_234563

/-- Given the initial number of bird families and the number of families left,
    calculate the number of families that flew away. -/
theorem bird_families_flew_away (initial : ℕ) (left : ℕ) (flew_away : ℕ) 
    (h1 : initial = 41)
    (h2 : left = 14)
    (h3 : flew_away = initial - left) :
  flew_away = 27 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_flew_away_l2345_234563


namespace NUMINAMATH_CALUDE_integer_solution_inequality_l2345_234546

theorem integer_solution_inequality (x : ℤ) : 
  (3 * |2 * x + 1| + 6 < 24) ↔ x ∈ ({-3, -2, -1, 0, 1, 2} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_inequality_l2345_234546


namespace NUMINAMATH_CALUDE_alice_sold_120_oranges_l2345_234508

/-- The number of oranges Emily sold -/
def emily_oranges : ℕ := sorry

/-- The number of oranges Alice sold -/
def alice_oranges : ℕ := 2 * emily_oranges

/-- The total number of oranges sold -/
def total_oranges : ℕ := 180

/-- Theorem stating that Alice sold 120 oranges -/
theorem alice_sold_120_oranges : alice_oranges = 120 :=
  by
    sorry

/-- The condition that the total oranges sold is the sum of Alice's and Emily's oranges -/
axiom total_is_sum : total_oranges = alice_oranges + emily_oranges

end NUMINAMATH_CALUDE_alice_sold_120_oranges_l2345_234508


namespace NUMINAMATH_CALUDE_non_parallel_diagonals_32gon_l2345_234505

/-- The number of diagonals not parallel to any side in a regular n-gon -/
def non_parallel_diagonals (n : ℕ) : ℕ :=
  let total_diagonals := n * (n - 3) / 2
  let parallel_pairs := n / 2
  let diagonals_per_pair := (n - 4) / 2
  let parallel_diagonals := parallel_pairs * diagonals_per_pair
  total_diagonals - parallel_diagonals

/-- Theorem: In a regular 32-gon, the number of diagonals not parallel to any of its sides is 240 -/
theorem non_parallel_diagonals_32gon :
  non_parallel_diagonals 32 = 240 := by
  sorry


end NUMINAMATH_CALUDE_non_parallel_diagonals_32gon_l2345_234505


namespace NUMINAMATH_CALUDE_basket_weight_proof_l2345_234569

def basket_problem (num_pears : ℕ) (pear_weight : ℝ) (total_weight : ℝ) : Prop :=
  let pears_weight := num_pears * pear_weight
  let basket_weight := total_weight - pears_weight
  basket_weight = 0.46

theorem basket_weight_proof :
  basket_problem 30 0.36 11.26 := by
  sorry

end NUMINAMATH_CALUDE_basket_weight_proof_l2345_234569


namespace NUMINAMATH_CALUDE_min_value_theorem_l2345_234510

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 1) :
  (1 / a + 1 / b ≥ 2 * Real.sqrt 2) ∧ (b / a^3 + a / b^3 ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2345_234510


namespace NUMINAMATH_CALUDE_problem_part1_problem_part2_l2345_234551

-- Part 1
theorem problem_part1 (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = |a * x - 1|) 
  (h2 : Set.Icc (-3) 1 = {x | f x ≤ 2}) : a = -1 := by sorry

-- Part 2
theorem problem_part2 (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) 
  (m : ℝ) (h2 : ∃ x, f (2 * x + 1) - f (x - 1) ≤ 3 - 2 * m) : m ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_problem_part1_problem_part2_l2345_234551


namespace NUMINAMATH_CALUDE_evaluate_expression_l2345_234579

theorem evaluate_expression : (33 + 12)^2 - (12^2 + 33^2) = 792 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2345_234579


namespace NUMINAMATH_CALUDE_percentage_of_unsold_books_l2345_234595

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem percentage_of_unsold_books :
  abs (percentage_not_sold - 80.57) < 0.01 := by sorry

end NUMINAMATH_CALUDE_percentage_of_unsold_books_l2345_234595


namespace NUMINAMATH_CALUDE_coin_stack_solution_l2345_234517

/-- Thickness of a nickel in millimeters -/
def nickel_thickness : ℚ := 2.05

/-- Thickness of a quarter in millimeters -/
def quarter_thickness : ℚ := 1.65

/-- Height of the stack in millimeters -/
def stack_height : ℚ := 16.5

theorem coin_stack_solution :
  ∃! (n q : ℕ), 
    n * nickel_thickness + q * quarter_thickness = stack_height ∧
    n + q = 9 := by sorry

end NUMINAMATH_CALUDE_coin_stack_solution_l2345_234517


namespace NUMINAMATH_CALUDE_truncated_tetrahedron_edge_count_l2345_234527

/-- A tetrahedron with truncated vertices -/
structure TruncatedTetrahedron where
  /-- The number of truncated vertices -/
  truncatedVertices : ℕ
  /-- Assertion that all vertices are truncated -/
  all_truncated : truncatedVertices = 4
  /-- Assertion that truncations are distinct and non-intersecting -/
  distinct_truncations : True

/-- The number of edges in a truncated tetrahedron -/
def edgeCount (t : TruncatedTetrahedron) : ℕ := sorry

/-- Theorem stating that a truncated tetrahedron has 18 edges -/
theorem truncated_tetrahedron_edge_count (t : TruncatedTetrahedron) : 
  edgeCount t = 18 := by sorry

end NUMINAMATH_CALUDE_truncated_tetrahedron_edge_count_l2345_234527


namespace NUMINAMATH_CALUDE_problem_solution_l2345_234542

noncomputable section

def f (x : ℝ) : ℝ := Real.log ((2 / (x + 1)) - 1) / Real.log 10

def g (a x : ℝ) : ℝ := Real.sqrt (1 - a^2 - 2*a*x - x^2)

def A : Set ℝ := {x : ℝ | (2 / (x + 1)) - 1 > 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | 1 - a^2 - 2*a*x - x^2 ≥ 0}

theorem problem_solution (a : ℝ) :
  (f (1/2013) + f (-1/2013) = 0) ∧
  (∀ a, a ≥ 2 → A ∩ B a = ∅) ∧
  (∃ a, a < 2 ∧ A ∩ B a = ∅) :=
sorry

end

end NUMINAMATH_CALUDE_problem_solution_l2345_234542


namespace NUMINAMATH_CALUDE_cos_double_angle_tan_4_l2345_234586

theorem cos_double_angle_tan_4 (α : Real) (h : Real.tan α = 4) : 
  Real.cos (2 * α) = -15 / 17 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_tan_4_l2345_234586


namespace NUMINAMATH_CALUDE_star_two_three_l2345_234582

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 * b^2 - a + 2

-- Theorem statement
theorem star_two_three : star 2 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_star_two_three_l2345_234582


namespace NUMINAMATH_CALUDE_square_count_is_22_l2345_234503

/-- Represents a square grid with some line segments possibly erased -/
structure ErasedGrid where
  size : Nat
  erased_segments : List (Nat × Nat × Nat × Nat)

/-- Counts the number of squares in an erased grid -/
def count_squares (grid : ErasedGrid) : Nat :=
  sorry

/-- The specific 4x4 grid with two erased segments as described in the problem -/
def problem_grid : ErasedGrid :=
  { size := 4,
    erased_segments := [(1, 1, 1, 2), (2, 2, 3, 2)] }

theorem square_count_is_22 : count_squares problem_grid = 22 := by
  sorry

end NUMINAMATH_CALUDE_square_count_is_22_l2345_234503


namespace NUMINAMATH_CALUDE_multiple_of_2007_cube_difference_l2345_234554

theorem multiple_of_2007_cube_difference (k : ℕ+) :
  (∃ a : ℤ, ∃ m : ℤ, (a + k.val : ℤ)^3 - a^3 = 2007 * m) ↔ ∃ n : ℕ, k.val = 669 * n :=
sorry

end NUMINAMATH_CALUDE_multiple_of_2007_cube_difference_l2345_234554


namespace NUMINAMATH_CALUDE_point_p_coordinates_l2345_234545

/-- A point in the fourth quadrant with specific distances from axes -/
structure PointP where
  x : ℝ
  y : ℝ
  in_fourth_quadrant : x > 0 ∧ y < 0
  distance_to_x_axis : |y| = 1
  distance_to_y_axis : |x| = 2

/-- The coordinates of point P are (2, -1) -/
theorem point_p_coordinates (p : PointP) : p.x = 2 ∧ p.y = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_p_coordinates_l2345_234545


namespace NUMINAMATH_CALUDE_overtaking_time_l2345_234537

/-- The problem of determining when person B starts walking to overtake person A --/
theorem overtaking_time (speed_A speed_B overtake_time : ℝ) (h1 : speed_A = 5)
  (h2 : speed_B = 5.555555555555555) (h3 : overtake_time = 1.8) :
  let start_time_diff := overtake_time * speed_B / speed_A - overtake_time
  start_time_diff = 0.2 := by sorry

end NUMINAMATH_CALUDE_overtaking_time_l2345_234537


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l2345_234581

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = -6 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 4 → 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l2345_234581


namespace NUMINAMATH_CALUDE_add_9999_seconds_to_10_15_30_l2345_234500

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time -/
def initialTime : Time :=
  { hours := 10, minutes := 15, seconds := 30 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 9999

/-- The expected final time -/
def expectedFinalTime : Time :=
  { hours := 13, minutes := 2, seconds := 9 }

theorem add_9999_seconds_to_10_15_30 :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_add_9999_seconds_to_10_15_30_l2345_234500


namespace NUMINAMATH_CALUDE_bike_clamps_theorem_l2345_234589

/-- The number of bike clamps given away with each bicycle sale -/
def clamps_per_bike : ℕ := 2

/-- The number of bikes sold in the morning -/
def morning_sales : ℕ := 19

/-- The number of bikes sold in the afternoon -/
def afternoon_sales : ℕ := 27

/-- The total number of bike clamps given away -/
def total_clamps : ℕ := clamps_per_bike * (morning_sales + afternoon_sales)

theorem bike_clamps_theorem : total_clamps = 92 := by
  sorry

end NUMINAMATH_CALUDE_bike_clamps_theorem_l2345_234589


namespace NUMINAMATH_CALUDE_work_completion_time_l2345_234538

/-- Represents the number of days it takes for B to complete the entire work -/
def days_for_B (days_for_A days_A_worked days_B_remaining : ℕ) : ℚ :=
  (4 * days_for_A * days_B_remaining) / (3 * days_for_A - 3 * days_A_worked)

theorem work_completion_time :
  days_for_B 40 10 45 = 60 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2345_234538


namespace NUMINAMATH_CALUDE_total_molecular_weight_l2345_234552

/-- Atomic weight in g/mol -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "Ca" => 40.08
  | "I"  => 126.90
  | "Na" => 22.99
  | "Cl" => 35.45
  | "K"  => 39.10
  | "S"  => 32.06
  | "O"  => 16.00
  | _    => 0  -- Default case

/-- Molecular weight of a compound in g/mol -/
def molecular_weight (compound : String) : ℝ :=
  match compound with
  | "CaI2" => atomic_weight "Ca" + 2 * atomic_weight "I"
  | "NaCl" => atomic_weight "Na" + atomic_weight "Cl"
  | "K2SO4" => 2 * atomic_weight "K" + atomic_weight "S" + 4 * atomic_weight "O"
  | _      => 0  -- Default case

/-- Total weight of a given number of moles of a compound in grams -/
def total_weight (compound : String) (moles : ℝ) : ℝ :=
  moles * molecular_weight compound

/-- Theorem: The total molecular weight of 10 moles of CaI2, 7 moles of NaCl, and 15 moles of K2SO4 is 5961.78 grams -/
theorem total_molecular_weight : 
  total_weight "CaI2" 10 + total_weight "NaCl" 7 + total_weight "K2SO4" 15 = 5961.78 := by
  sorry

end NUMINAMATH_CALUDE_total_molecular_weight_l2345_234552


namespace NUMINAMATH_CALUDE_exactly_three_solutions_l2345_234578

-- Define the greatest integer function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (floor x : ℝ) / x else 0

-- State the theorem
theorem exactly_three_solutions (a : ℝ) (h : 3/4 < a ∧ a ≤ 4/5) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, x > 0 ∧ f x = a :=
sorry

end NUMINAMATH_CALUDE_exactly_three_solutions_l2345_234578


namespace NUMINAMATH_CALUDE_car_cyclist_problem_solution_l2345_234562

/-- Represents the speeds and meeting point of a car and cyclist problem -/
structure CarCyclistProblem where
  car_speed : ℝ
  cyclist_speed : ℝ
  meeting_distance_from_A : ℝ

/-- Checks if the given speeds and meeting point satisfy the problem conditions -/
def is_valid_solution (p : CarCyclistProblem) : Prop :=
  let total_distance := 80
  let time_to_meet := 1.5
  let distance_after_one_hour := 24
  let car_distance_one_hour := p.car_speed
  let cyclist_distance_one_hour := p.cyclist_speed
  let car_total_distance := p.car_speed * time_to_meet
  let cyclist_total_distance := p.cyclist_speed * 1.25  -- Cyclist rests for 1 hour

  -- Condition 1: After one hour, they are 24 km apart
  (total_distance - (car_distance_one_hour + cyclist_distance_one_hour) = distance_after_one_hour) ∧
  -- Condition 2: They meet after 90 minutes
  (car_total_distance + cyclist_total_distance = total_distance) ∧
  -- Condition 3: Meeting point is correct
  (p.meeting_distance_from_A = car_total_distance)

/-- The theorem stating that the given solution satisfies the problem conditions -/
theorem car_cyclist_problem_solution :
  is_valid_solution ⟨40, 16, 60⟩ := by
  sorry

end NUMINAMATH_CALUDE_car_cyclist_problem_solution_l2345_234562


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2345_234528

def A : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2345_234528


namespace NUMINAMATH_CALUDE_estate_value_l2345_234544

def estate_problem (total_estate : ℚ) : Prop :=
  let daughters_son_share := (3 : ℚ) / 5 * total_estate
  let first_daughter := (5 : ℚ) / 10 * daughters_son_share
  let second_daughter := (3 : ℚ) / 10 * daughters_son_share
  let son := (2 : ℚ) / 10 * daughters_son_share
  let husband := 2 * son
  let gardener := 600
  let charity := 800
  total_estate = first_daughter + second_daughter + son + husband + gardener + charity

theorem estate_value : 
  ∃ (total_estate : ℚ), estate_problem total_estate ∧ total_estate = 35000 := by
  sorry

end NUMINAMATH_CALUDE_estate_value_l2345_234544


namespace NUMINAMATH_CALUDE_sum_angles_less_than_1100_l2345_234555

/-- Represents the angle measurement scenario with a car and a fence -/
structure AngleMeasurement where
  carSpeed : ℝ  -- Car speed in km/h
  fenceLength : ℝ  -- Fence length in meters
  measurementInterval : ℝ  -- Measurement interval in seconds

/-- Calculates the sum of angles measured -/
def sumOfAngles (scenario : AngleMeasurement) : ℝ :=
  sorry  -- Proof omitted

/-- Theorem stating that the sum of angles is less than 1100 degrees -/
theorem sum_angles_less_than_1100 (scenario : AngleMeasurement) 
  (h1 : scenario.carSpeed = 60)
  (h2 : scenario.fenceLength = 100)
  (h3 : scenario.measurementInterval = 1) :
  sumOfAngles scenario < 1100 := by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_sum_angles_less_than_1100_l2345_234555


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angles_l2345_234592

-- Define an isosceles triangle with one angle of 150°
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ a = b ∧ c = 150

-- Theorem statement
theorem isosceles_triangle_base_angles 
  (a b c : ℝ) (h : IsoscelesTriangle a b c) : a = 15 ∧ b = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angles_l2345_234592


namespace NUMINAMATH_CALUDE_game_show_probability_l2345_234541

theorem game_show_probability (total_doors : ℕ) (prize_doors : ℕ) 
  (opened_doors : ℕ) (opened_prize_doors : ℕ) :
  total_doors = 7 →
  prize_doors = 2 →
  opened_doors = 3 →
  opened_prize_doors = 1 →
  (total_doors - opened_doors - 1 : ℚ) / (total_doors - opened_doors) * 
  (prize_doors - opened_prize_doors) / (total_doors - opened_doors - 1) = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_game_show_probability_l2345_234541


namespace NUMINAMATH_CALUDE_problem_solution_l2345_234567

theorem problem_solution : 
  (101 * 99 = 9999) ∧ 
  (32 * 2^2 + 14 * 2^3 + 10 * 2^4 = 400) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2345_234567


namespace NUMINAMATH_CALUDE_range_of_sin6_plus_cos4_l2345_234568

theorem range_of_sin6_plus_cos4 :
  ∀ x : ℝ, 0 ≤ Real.sin x ^ 6 + Real.cos x ^ 4 ∧
  Real.sin x ^ 6 + Real.cos x ^ 4 ≤ 1 ∧
  (∃ y : ℝ, Real.sin y ^ 6 + Real.cos y ^ 4 = 0) ∧
  (∃ z : ℝ, Real.sin z ^ 6 + Real.cos z ^ 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_sin6_plus_cos4_l2345_234568


namespace NUMINAMATH_CALUDE_hex_grid_half_path_l2345_234580

/-- Represents a point in the hexagonal grid -/
structure HexPoint where
  x : ℤ
  y : ℤ

/-- Represents a direction in the hexagonal grid -/
inductive HexDirection
  | Horizontal
  | LeftDiagonal
  | RightDiagonal

/-- Calculates the distance between two points in the hexagonal grid -/
def hexDistance (a b : HexPoint) : ℕ :=
  sorry

/-- Represents a path in the hexagonal grid -/
def HexPath := List HexDirection

/-- Checks if a path is valid (follows the grid lines) -/
def isValidPath (path : HexPath) (start finish : HexPoint) : Prop :=
  sorry

/-- Calculates the length of a path -/
def pathLength (path : HexPath) : ℕ :=
  sorry

/-- Checks if a path is the shortest between two points -/
def isShortestPath (path : HexPath) (start finish : HexPoint) : Prop :=
  isValidPath path start finish ∧
  pathLength path = hexDistance start finish

/-- Counts the number of steps in a single direction -/
def countDirectionSteps (path : HexPath) (direction : HexDirection) : ℕ :=
  sorry

theorem hex_grid_half_path (a b : HexPoint) (path : HexPath) :
  isShortestPath path a b →
  hexDistance a b = 100 →
  ∃ (direction : HexDirection), countDirectionSteps path direction = 50 :=
sorry

end NUMINAMATH_CALUDE_hex_grid_half_path_l2345_234580


namespace NUMINAMATH_CALUDE_zero_in_interval_l2345_234524

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2345_234524


namespace NUMINAMATH_CALUDE_roll_sum_less_than_12_prob_value_l2345_234521

def roll_sum_less_than_12_prob : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := total_outcomes - 15
  favorable_outcomes / total_outcomes

theorem roll_sum_less_than_12_prob_value : 
  roll_sum_less_than_12_prob = 49 / 64 := by sorry

end NUMINAMATH_CALUDE_roll_sum_less_than_12_prob_value_l2345_234521


namespace NUMINAMATH_CALUDE_range_of_m_l2345_234547

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : x + 2 * y = 4)
  (h_solution : ∃ m : ℝ, m^2 + (1/3) * m > 2/x + 1/(y+1)) :
  ∃ m : ℝ, (m < -4/3 ∨ m > 1) ∧ m^2 + (1/3) * m > 2/x + 1/(y+1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2345_234547


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l2345_234516

/-- Represents a large cube composed of unit cubes -/
structure LargeCube where
  side_length : ℕ
  total_units : ℕ
  painted_on_opposite_faces : ℕ
  painted_on_other_faces : ℕ

/-- Calculates the number of unpainted unit cubes in the large cube -/
def unpainted_cubes (c : LargeCube) : ℕ :=
  c.total_units - (2 * c.painted_on_opposite_faces + 4 * c.painted_on_other_faces - 8)

/-- The theorem to be proved -/
theorem unpainted_cubes_count (c : LargeCube) 
  (h1 : c.side_length = 6)
  (h2 : c.total_units = 216)
  (h3 : c.painted_on_opposite_faces = 16)
  (h4 : c.painted_on_other_faces = 9) :
  unpainted_cubes c = 156 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_count_l2345_234516


namespace NUMINAMATH_CALUDE_negative_two_m_cubed_squared_l2345_234539

theorem negative_two_m_cubed_squared (m : ℝ) : (-2 * m^3)^2 = 4 * m^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_m_cubed_squared_l2345_234539


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l2345_234522

theorem complex_product_magnitude : 
  Complex.abs ((3 - 4 * Complex.I) * (5 + 12 * Complex.I) * (2 - 7 * Complex.I)) = 65 * Real.sqrt 53 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l2345_234522


namespace NUMINAMATH_CALUDE_james_ali_difference_l2345_234585

def total_amount : ℕ := 250
def james_amount : ℕ := 145

theorem james_ali_difference :
  ∀ (ali_amount : ℕ),
  ali_amount + james_amount = total_amount →
  james_amount > ali_amount →
  james_amount - ali_amount = 40 :=
by sorry

end NUMINAMATH_CALUDE_james_ali_difference_l2345_234585


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l2345_234535

/-- Given a regular pentagon and a rectangle with the same perimeter,
    where the rectangle's length is twice its width,
    prove that the ratio of the pentagon's side length to the rectangle's width is 6/5 -/
theorem pentagon_rectangle_ratio (p w : ℝ) (h1 : 5 * p = 30) (h2 : 6 * w = 30) : p / w = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l2345_234535


namespace NUMINAMATH_CALUDE_system_solutions_l2345_234509

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  4 * x^2 / (1 + 4 * x^2) = y ∧
  4 * y^2 / (1 + 4 * y^2) = z ∧
  4 * z^2 / (1 + 4 * z^2) = x

-- Theorem statement
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2345_234509


namespace NUMINAMATH_CALUDE_nancy_games_this_month_l2345_234514

/-- Represents the number of football games Nancy attended or plans to attend -/
structure FootballGames where
  lastMonth : ℕ
  thisMonth : ℕ
  nextMonth : ℕ
  total : ℕ

/-- Theorem stating that Nancy attended 9 games this month -/
theorem nancy_games_this_month (g : FootballGames)
  (h1 : g.lastMonth = 8)
  (h2 : g.nextMonth = 7)
  (h3 : g.total = 24)
  (h4 : g.total = g.lastMonth + g.thisMonth + g.nextMonth) :
  g.thisMonth = 9 := by
  sorry


end NUMINAMATH_CALUDE_nancy_games_this_month_l2345_234514


namespace NUMINAMATH_CALUDE_candy_division_l2345_234515

theorem candy_division (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (hpq : p < q) (hqr : q < r) :
  (∃ n : ℕ, n > 0 ∧ n * (r + q - 2 * p) = 39) →
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x + y + (r - p) = 10) →
  (∃ z : ℕ, z > 0 ∧ 18 - 3 * p = 9) →
  (p = 3 ∧ q = 6 ∧ r = 13) := by
sorry

end NUMINAMATH_CALUDE_candy_division_l2345_234515


namespace NUMINAMATH_CALUDE_frank_to_betty_bill_ratio_l2345_234531

/-- The number of oranges Betty picked -/
def betty_oranges : ℕ := 15

/-- The number of oranges Bill picked -/
def bill_oranges : ℕ := 12

/-- The number of seeds Frank planted from each orange -/
def seeds_per_orange : ℕ := 2

/-- The number of oranges on each tree -/
def oranges_per_tree : ℕ := 5

/-- The total number of oranges Philip can pick -/
def philip_oranges : ℕ := 810

/-- Theorem stating the ratio of Frank's oranges to Betty and Bill's combined oranges -/
theorem frank_to_betty_bill_ratio :
  ∃ (frank_oranges : ℕ),
    frank_oranges > 0 ∧
    philip_oranges = frank_oranges * seeds_per_orange * oranges_per_tree ∧
    frank_oranges = 3 * (betty_oranges + bill_oranges) := by
  sorry

end NUMINAMATH_CALUDE_frank_to_betty_bill_ratio_l2345_234531


namespace NUMINAMATH_CALUDE_part1_solution_set_part2_minimum_value_l2345_234558

-- Define the function f
def f (m n x : ℝ) : ℝ := |x - m| + |x - n|

-- Part 1
theorem part1_solution_set (x : ℝ) :
  (f 2 (-5) x > 9) ↔ (x < -6 ∨ x > 3) := by sorry

-- Part 2
theorem part2_minimum_value (a : ℝ) (h : a ≠ 0) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x : ℝ), f a (-1/a) x ≥ min := by sorry

end NUMINAMATH_CALUDE_part1_solution_set_part2_minimum_value_l2345_234558


namespace NUMINAMATH_CALUDE_total_area_smaller_than_4pi_R_squared_l2345_234533

variable (R x y z : ℝ)

/-- Three circles with radii x, y, and z touch each other externally -/
axiom circles_touch_externally : True

/-- The centers of the three circles lie on a fourth circle with radius R -/
axiom centers_on_fourth_circle : True

/-- The radius R of the fourth circle is related to x, y, and z by Heron's formula -/
axiom heron_formula : R = (x + y) * (y + z) * (z + x) / (4 * Real.sqrt ((x + y + z) * x * y * z))

/-- The total area of the three circle disks is smaller than 4πR² -/
theorem total_area_smaller_than_4pi_R_squared :
  x^2 + y^2 + z^2 < 4 * R^2 := by sorry

end NUMINAMATH_CALUDE_total_area_smaller_than_4pi_R_squared_l2345_234533


namespace NUMINAMATH_CALUDE_find_Y_l2345_234534

theorem find_Y : ∃ Y : ℚ, (19 + Y / 151) * 151 = 2912 → Y = 43 := by
  sorry

end NUMINAMATH_CALUDE_find_Y_l2345_234534


namespace NUMINAMATH_CALUDE_perpendicular_planes_necessary_not_sufficient_l2345_234574

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perp_line_plane (m : Line) (α : Plane) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def perp_plane_plane (α β : Plane) : Prop := sorry

/-- Definition of necessary but not sufficient condition -/
def necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem perpendicular_planes_necessary_not_sufficient 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) (h2 : α ≠ β) 
  (h3 : perp_line_plane m α) (h4 : line_in_plane n β) :
  necessary_not_sufficient (perp_plane_plane α β) (parallel m n) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_planes_necessary_not_sufficient_l2345_234574


namespace NUMINAMATH_CALUDE_sum_of_ages_l2345_234530

/-- Given the age relationships between Paula, Karl, and Jane at different points in time, 
    prove that the sum of their current ages is 63 years. -/
theorem sum_of_ages (P K J : ℚ) : 
  (P - 7 = 4 * (K - 7)) →  -- 7 years ago, Paula was 4 times as old as Karl
  (J - 7 = (P - 7) / 2) →  -- 7 years ago, Jane was half as old as Paula
  (P + 8 = 2 * (K + 8)) →  -- In 8 years, Paula will be twice as old as Karl
  (J + 8 = K + 5) →        -- In 8 years, Jane will be 3 years younger than Karl
  P + K + J = 63 :=        -- The sum of their current ages is 63
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2345_234530


namespace NUMINAMATH_CALUDE_division_problem_l2345_234588

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2345_234588


namespace NUMINAMATH_CALUDE_congruence_from_equation_l2345_234564

theorem congruence_from_equation (a b : ℕ+) (h : a^(b : ℕ) - b^(a : ℕ) = 1008) :
  a ≡ b [ZMOD 1008] := by
  sorry

end NUMINAMATH_CALUDE_congruence_from_equation_l2345_234564


namespace NUMINAMATH_CALUDE_intersection_of_ranges_equality_of_ranges_equality_of_functions_l2345_234543

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 4*x + 1

def A₁ : Set ℝ := Set.Icc 1 2
def S₁ : Set ℝ := Set.image f A₁
def T₁ : Set ℝ := Set.image g A₁

def A₂ (m : ℝ) : Set ℝ := Set.Icc 0 m
def S₂ (m : ℝ) : Set ℝ := Set.image f (A₂ m)
def T₂ (m : ℝ) : Set ℝ := Set.image g (A₂ m)

theorem intersection_of_ranges : S₁ ∩ T₁ = {5} := by sorry

theorem equality_of_ranges (m : ℝ) : S₂ m = T₂ m → m = 4 := by sorry

theorem equality_of_functions : 
  {A : Set ℝ | ∀ x ∈ A, f x = g x} ⊆ {{0}, {4}, {0, 4}} := by sorry

end NUMINAMATH_CALUDE_intersection_of_ranges_equality_of_ranges_equality_of_functions_l2345_234543


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l2345_234519

theorem geometric_mean_of_4_and_9 :
  ∃ G : ℝ, (4 / G = G / 9) ∧ (G = 6 ∨ G = -6) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l2345_234519


namespace NUMINAMATH_CALUDE_range_of_a_l2345_234587

-- Define the statements p and q
def p (a : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (x₁^2 / a + (k * x₁ + 1)^2 = 1) ∧
    (x₂^2 / a + (k * x₂ + 1)^2 = 1)

def q (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, 4^x₀ - 2^x₀ - a ≤ 0

-- State the theorem
theorem range_of_a :
  ∃ a : ℝ, -1/4 ≤ a ∧ a ≤ 1 ∧
  ¬(¬p a ∧ ¬q a) ∧ (p a ∨ q a) ∧
  ∀ b : ℝ, (¬(¬p b ∧ ¬q b) ∧ (p b ∨ q b)) → -1/4 ≤ b ∧ b ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2345_234587


namespace NUMINAMATH_CALUDE_charles_and_jen_whistles_l2345_234501

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The difference in whistles between Sean and Charles -/
def sean_charles_diff : ℕ := 32

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := sean_whistles - sean_charles_diff

/-- The difference in whistles between Jen and Charles -/
def jen_charles_diff : ℕ := 15

/-- The number of whistles Jen has -/
def jen_whistles : ℕ := charles_whistles + jen_charles_diff

/-- The total number of whistles Charles and Jen have -/
def total_whistles : ℕ := charles_whistles + jen_whistles

theorem charles_and_jen_whistles : total_whistles = 41 := by
  sorry

end NUMINAMATH_CALUDE_charles_and_jen_whistles_l2345_234501


namespace NUMINAMATH_CALUDE_problem_solution_l2345_234507

theorem problem_solution (x y : ℝ) : 
  y - Real.sqrt (x - 2022) = Real.sqrt (2022 - x) - 2023 →
  (x + y) ^ 2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2345_234507


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2345_234523

/-- 
Given the equation 2x(x+5) = 10, this theorem states that when converted to 
general form ax² + bx + c = 0, the coefficients a, b, and c are 2, 10, and -10 respectively.
-/
theorem quadratic_equation_coefficients : 
  ∃ (a b c : ℝ), (∀ x, 2*x*(x+5) = 10 ↔ a*x^2 + b*x + c = 0) ∧ a = 2 ∧ b = 10 ∧ c = -10 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2345_234523


namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l2345_234502

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l2345_234502


namespace NUMINAMATH_CALUDE_student_average_age_l2345_234518

/-- Given a class of students and a staff member, if including the staff's age
    increases the average age by 1 year, then we can determine the average age of the students. -/
theorem student_average_age
  (num_students : ℕ)
  (staff_age : ℕ)
  (avg_increase : ℝ)
  (h1 : num_students = 32)
  (h2 : staff_age = 49)
  (h3 : avg_increase = 1) :
  (num_students * (staff_age - num_students - 1 : ℝ)) / num_students = 16 := by
  sorry

end NUMINAMATH_CALUDE_student_average_age_l2345_234518


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l2345_234565

theorem slower_speed_calculation (distance : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  distance = 50 →
  faster_speed = 14 →
  extra_distance = 20 →
  ∃ slower_speed : ℝ,
    slower_speed > 0 ∧
    distance / slower_speed = distance / faster_speed + extra_distance / faster_speed ∧
    slower_speed = 10 :=
by sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l2345_234565


namespace NUMINAMATH_CALUDE_exists_valid_painting_33_exists_valid_painting_32_l2345_234577

/-- Represents a cell on the board -/
structure Cell :=
  (x : Fin 7)
  (y : Fin 7)

/-- Checks if two cells are adjacent -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.x = c2.x ∧ c1.y.val + 1 = c2.y.val) ∨
  (c1.x = c2.x ∧ c1.y.val = c2.y.val + 1) ∨
  (c1.x.val + 1 = c2.x.val ∧ c1.y = c2.y) ∨
  (c1.x.val = c2.x.val + 1 ∧ c1.y = c2.y)

/-- A valid painting sequence -/
def ValidPainting (seq : List Cell) : Prop :=
  seq.length > 0 ∧
  ∀ i j, 0 < i ∧ i < seq.length → 0 ≤ j ∧ j < i - 1 →
    (adjacent (seq.get ⟨i, sorry⟩) (seq.get ⟨i-1, sorry⟩) ∧
     ¬adjacent (seq.get ⟨i, sorry⟩) (seq.get ⟨j, sorry⟩))

/-- Main theorem: There exists a valid painting of 33 cells -/
theorem exists_valid_painting_33 :
  ∃ (seq : List Cell), seq.length = 33 ∧ ValidPainting seq :=
sorry

/-- Corollary: There exists a valid painting of 32 cells -/
theorem exists_valid_painting_32 :
  ∃ (seq : List Cell), seq.length = 32 ∧ ValidPainting seq :=
sorry

end NUMINAMATH_CALUDE_exists_valid_painting_33_exists_valid_painting_32_l2345_234577


namespace NUMINAMATH_CALUDE_square_in_triangle_angle_sum_l2345_234553

/-- The sum of angles in a triangle --/
def triangle_angle_sum : ℝ := 180

/-- The interior angle of an equilateral triangle --/
def equilateral_triangle_angle : ℝ := 60

/-- The sum of angles on a straight line --/
def straight_line_angle_sum : ℝ := 180

/-- The angle of a right angle (in a square) --/
def right_angle : ℝ := 90

/-- Configuration of a square inscribed in an equilateral triangle --/
structure SquareInTriangle where
  x : ℝ  -- Angle between square side and triangle side
  y : ℝ  -- Angle between square side and triangle side
  p : ℝ  -- Complementary angle to x in the triangle
  q : ℝ  -- Complementary angle to y in the triangle

/-- Theorem: The sum of x and y in the SquareInTriangle configuration is 150° --/
theorem square_in_triangle_angle_sum (config : SquareInTriangle) : 
  config.x + config.y = 150 := by
  sorry

end NUMINAMATH_CALUDE_square_in_triangle_angle_sum_l2345_234553


namespace NUMINAMATH_CALUDE_clara_alice_pen_ratio_l2345_234566

def alice_pens : ℕ := 60
def alice_age : ℕ := 20
def clara_future_age : ℕ := 61
def years_to_future : ℕ := 5

theorem clara_alice_pen_ratio :
  ∃ (clara_pens : ℕ) (clara_age : ℕ),
    clara_age > alice_age ∧
    clara_age + years_to_future = clara_future_age ∧
    clara_age - alice_age = alice_pens - clara_pens ∧
    clara_pens * 5 = alice_pens * 2 :=
by sorry

end NUMINAMATH_CALUDE_clara_alice_pen_ratio_l2345_234566


namespace NUMINAMATH_CALUDE_penelope_candy_count_l2345_234576

/-- Given a ratio of M&M candies to Starbursts candies and a number of Starbursts,
    calculate the number of M&M candies. -/
def calculate_mm_candies (mm_ratio : ℕ) (starburst_ratio : ℕ) (starburst_count : ℕ) : ℕ :=
  (starburst_count / starburst_ratio) * mm_ratio

/-- Theorem stating that given the specific ratio and Starburst count,
    the number of M&M candies is 25. -/
theorem penelope_candy_count :
  calculate_mm_candies 5 3 15 = 25 := by
  sorry

end NUMINAMATH_CALUDE_penelope_candy_count_l2345_234576


namespace NUMINAMATH_CALUDE_one_third_of_four_equals_two_l2345_234596

-- Define the country's multiplication operation
noncomputable def country_mul (a b : ℚ) : ℚ := sorry

-- Define the property that 1/8 of 4 equals 3 in this system
axiom country_property : country_mul (1/8) 4 = 3

-- Theorem statement
theorem one_third_of_four_equals_two : 
  country_mul (1/3) 4 = 2 := by sorry

end NUMINAMATH_CALUDE_one_third_of_four_equals_two_l2345_234596


namespace NUMINAMATH_CALUDE_chord_length_theorem_l2345_234512

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c2.radius - c1.radius)^2

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem chord_length_theorem (c1 c2 c3 : Circle) 
  (h1 : c1.radius = 6)
  (h2 : c2.radius = 12)
  (h3 : are_externally_tangent c1 c2)
  (h4 : is_internally_tangent c1 c3)
  (h5 : is_internally_tangent c2 c3)
  (h6 : are_collinear c1.center c2.center c3.center) :
  ∃ (chord_length : ℝ), 
    chord_length = (144 * Real.sqrt 26) / 5 ∧ 
    chord_length^2 = 4 * (c3.radius^2 - (c3.radius - c1.radius - c2.radius)^2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l2345_234512


namespace NUMINAMATH_CALUDE_price_difference_is_500_l2345_234550

/-- The price difference between enhanced and basic computers -/
def price_difference (total_basic_printer : ℝ) (price_basic : ℝ) : ℝ :=
  let price_printer := total_basic_printer - price_basic
  let price_enhanced := 8 * price_printer - price_printer
  price_enhanced - price_basic

/-- Theorem stating the price difference between enhanced and basic computers -/
theorem price_difference_is_500 :
  price_difference 2500 2125 = 500 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_is_500_l2345_234550


namespace NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_digits_of_large_number_l2345_234561

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number has exactly ten billion digits -/
def has_ten_billion_digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_sum_of_digits_of_large_number 
  (A : ℕ) 
  (h1 : has_ten_billion_digits A) 
  (h2 : A % 9 = 0) : 
  let B := sum_of_digits A
  let C := sum_of_digits B
  sum_of_digits C = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_digits_of_large_number_l2345_234561


namespace NUMINAMATH_CALUDE_prime_mod_8_not_sum_of_three_squares_l2345_234583

theorem prime_mod_8_not_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (hmod : p % 8 = 7) :
  ¬ ∃ (a b c : ℤ), (a ^ 2 + b ^ 2 + c ^ 2 : ℤ) = p := by
  sorry

end NUMINAMATH_CALUDE_prime_mod_8_not_sum_of_three_squares_l2345_234583


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2345_234536

theorem tan_alpha_plus_pi_fourth (α β : Real) 
  (h1 : Real.tan (α + β) = 3/7)
  (h2 : Real.tan (β - π/4) = -1/3) :
  Real.tan (α + π/4) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2345_234536


namespace NUMINAMATH_CALUDE_tea_customers_l2345_234525

/-- Proves that the number of tea customers is 8 given the conditions of the problem -/
theorem tea_customers (coffee_price : ℕ) (tea_price : ℕ) (coffee_customers : ℕ) (total_revenue : ℕ) :
  coffee_price = 5 →
  tea_price = 4 →
  coffee_customers = 7 →
  total_revenue = 67 →
  ∃ tea_customers : ℕ, 
    tea_customers = 8 ∧ 
    coffee_price * coffee_customers + tea_price * tea_customers = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_tea_customers_l2345_234525


namespace NUMINAMATH_CALUDE_custom_equation_solution_l2345_234599

-- Define the custom operation *
def star (a b : ℚ) : ℚ := 4 * a - 2 * b

-- State the theorem
theorem custom_equation_solution :
  ∃! x : ℚ, star 3 (star 6 x) = -2 ∧ x = 17/2 := by sorry

end NUMINAMATH_CALUDE_custom_equation_solution_l2345_234599


namespace NUMINAMATH_CALUDE_power_mean_inequality_l2345_234598

theorem power_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) :
  (a^4 + b^4) / 2 ≥ ((a + b) / 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_power_mean_inequality_l2345_234598


namespace NUMINAMATH_CALUDE_sqrt_15_bounds_l2345_234593

theorem sqrt_15_bounds : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_bounds_l2345_234593


namespace NUMINAMATH_CALUDE_hyperbola_standard_form_l2345_234520

/-- A hyperbola with given asymptote and point -/
structure Hyperbola where
  -- Asymptote equation: 3x + 4y = 0
  asymptote_slope : ℝ
  asymptote_slope_eq : asymptote_slope = -3/4
  -- Point on the hyperbola
  point : ℝ × ℝ
  point_eq : point = (4, 6)

/-- The standard form of a hyperbola -/
def standard_form (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem stating the standard form of the hyperbola -/
theorem hyperbola_standard_form (h : Hyperbola) :
  ∃ (a b : ℝ), a^2 = 48 ∧ b^2 = 27 ∧ 
  ∀ (x y : ℝ), standard_form a b x y ↔ 
    (∃ (t : ℝ), x = 3*t ∧ y = -4*t) ∨ (x, y) = h.point :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_form_l2345_234520


namespace NUMINAMATH_CALUDE_multiplication_of_powers_l2345_234549

theorem multiplication_of_powers (b : ℝ) : 3 * b^3 * (2 * b^2) = 6 * b^5 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_powers_l2345_234549


namespace NUMINAMATH_CALUDE_opposite_numbers_l2345_234575

theorem opposite_numbers : ((-5)^2 : ℤ) = -(-5^2) :=
sorry

end NUMINAMATH_CALUDE_opposite_numbers_l2345_234575


namespace NUMINAMATH_CALUDE_larger_number_problem_l2345_234571

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1365) (h3 : L = 6 * S + 15) : L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2345_234571
