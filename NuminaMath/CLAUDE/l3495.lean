import Mathlib

namespace NUMINAMATH_CALUDE_constant_term_is_180_l3495_349563

/-- The binomial expansion of (√x + 2/x²)^10 has its largest coefficient in the sixth term -/
axiom largest_coeff_sixth_term : ∃ k, k = 6 ∧ ∀ j, j ≠ k → 
  Nat.choose 10 (k-1) * 2^(k-1) ≥ Nat.choose 10 (j-1) * 2^(j-1)

/-- The constant term in the expansion of (√x + 2/x²)^10 -/
def constant_term : ℕ := Nat.choose 10 2 * 2^2

theorem constant_term_is_180 : constant_term = 180 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_is_180_l3495_349563


namespace NUMINAMATH_CALUDE_clock_equivalent_square_l3495_349545

theorem clock_equivalent_square : 
  ∃ (h : ℕ), h > 10 ∧ h ≤ 12 ∧ (h - h^2) % 12 = 0 ∧ 
  ∀ (k : ℕ), k > 10 ∧ k < h → (k - k^2) % 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_clock_equivalent_square_l3495_349545


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3495_349561

/-- 
Given a line L1 with equation 2x - y + 1 = 0, and a point P (1, 1),
prove that the line L2 passing through P and parallel to L1 has the equation 2x - y - 1 = 0.
-/
theorem parallel_line_through_point (x y : ℝ) : 
  (2 * x - y + 1 = 0) →  -- L1 equation
  (∃ c : ℝ, 2 * x - y + c = 0 ∧ 2 * 1 - 1 + c = 0) →  -- L2 passes through (1, 1)
  (2 * x - y - 1 = 0)  -- L2 equation
:= by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3495_349561


namespace NUMINAMATH_CALUDE_blaine_fish_count_l3495_349591

theorem blaine_fish_count :
  ∀ (blaine_fish keith_fish : ℕ),
    blaine_fish > 0 →
    keith_fish = 2 * blaine_fish →
    blaine_fish + keith_fish = 15 →
    blaine_fish = 5 := by
  sorry

end NUMINAMATH_CALUDE_blaine_fish_count_l3495_349591


namespace NUMINAMATH_CALUDE_student_congress_sample_size_l3495_349581

/-- Represents a school with classes and a student congress. -/
structure School where
  num_classes : ℕ
  students_per_class : ℕ
  students_sent_per_class : ℕ

/-- Calculates the sample size for the Student Congress. -/
def sample_size (s : School) : ℕ :=
  s.num_classes * s.students_sent_per_class

/-- Theorem: The sample size for the given school is 120. -/
theorem student_congress_sample_size :
  let s : School := {
    num_classes := 40,
    students_per_class := 50,
    students_sent_per_class := 3
  }
  sample_size s = 120 := by
  sorry


end NUMINAMATH_CALUDE_student_congress_sample_size_l3495_349581


namespace NUMINAMATH_CALUDE_equation_solution_l3495_349523

theorem equation_solution : ∃ x : ℝ, (((32 : ℝ) ^ (x - 2) / (8 : ℝ) ^ (x - 2)) ^ 2 = (1024 : ℝ) ^ (2 * x - 1)) ∧ x = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3495_349523


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3495_349538

/-- Given a hyperbola with equation x²/p² - y²/q² = 1 where p > q,
    if the angle between its asymptotes is 45°, then p/q = √2 - 1 -/
theorem hyperbola_asymptote_angle (p q : ℝ) (h1 : p > q) (h2 : q > 0) :
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / p^2 - (y t)^2 / q^2 = 1) →
  (∃ (m : ℝ), m = q / p ∧ 
    Real.tan (45 * π / 180) = |((m - (-m)) / (1 + m * (-m)))|) →
  p / q = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3495_349538


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3495_349502

/-- The surface area of a cylinder with diameter 4 units and height 3 units is 20π square units. -/
theorem cylinder_surface_area : 
  let d : ℝ := 4  -- diameter
  let h : ℝ := 3  -- height
  let r : ℝ := d / 2  -- radius
  let surface_area : ℝ := 2 * Real.pi * r^2 + 2 * Real.pi * r * h
  surface_area = 20 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3495_349502


namespace NUMINAMATH_CALUDE_range_of_a_l3495_349553

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∪ B a = B a → a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3495_349553


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3495_349527

theorem trigonometric_simplification (α : ℝ) :
  (-Real.sin (π + α) + Real.sin (-α) - Real.tan (2*π + α)) /
  (Real.tan (α + π) + Real.cos (-α) + Real.cos (π - α)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3495_349527


namespace NUMINAMATH_CALUDE_fraction_equality_l3495_349575

theorem fraction_equality : (2015 : ℤ) / (2015^2 - 2016 * 2014) = 2015 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3495_349575


namespace NUMINAMATH_CALUDE_min_sum_of_probabilities_l3495_349536

theorem min_sum_of_probabilities (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let p_a := 4 / x
  let p_b := 1 / y
  (p_a + p_b = 1) → (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 / x' + 1 / y' = 1 → x' + y' ≥ x + y) →
  x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_probabilities_l3495_349536


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l3495_349548

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 8 * X^3 + 16 * X^2 - 7 * X + 4
  let divisor : Polynomial ℚ := 2 * X + 5
  let quotient : Polynomial ℚ := 4 * X^2 - 2 * X + (3/2)
  dividend = divisor * quotient + (-7/2) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l3495_349548


namespace NUMINAMATH_CALUDE_project_time_calculation_l3495_349597

/-- Calculates the remaining time for writing a report given the total time available and time spent on research and proposal. -/
def remaining_time (total_time research_time proposal_time : ℕ) : ℕ :=
  total_time - (research_time + proposal_time)

/-- Proves that given 20 hours total, 10 hours for research, and 2 hours for proposal, 
    the remaining time for writing the report is 8 hours. -/
theorem project_time_calculation :
  remaining_time 20 10 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_project_time_calculation_l3495_349597


namespace NUMINAMATH_CALUDE_three_zeros_sin_minus_one_l3495_349567

/-- The function f(x) = sin(ωx) - 1 has exactly 3 zeros in [0, 2π] iff ω ∈ [9/4, 13/4) -/
theorem three_zeros_sin_minus_one (ω : ℝ) : ω > 0 →
  (∃! (s : Finset ℝ), s.card = 3 ∧ (∀ x ∈ s, x ∈ Set.Icc 0 (2 * Real.pi) ∧ Real.sin (ω * x) = 1)) ↔
  ω ∈ Set.Icc (9 / 4) (13 / 4) := by
  sorry

#check three_zeros_sin_minus_one

end NUMINAMATH_CALUDE_three_zeros_sin_minus_one_l3495_349567


namespace NUMINAMATH_CALUDE_fruit_purchase_theorem_l3495_349556

/-- Calculates the total cost of a fruit purchase with a quantity-based discount --/
def fruitPurchaseCost (lemonPrice papayaPrice mangoPrice : ℕ) 
                      (lemonQty papayaQty mangoQty : ℕ) 
                      (discountPerFruits : ℕ) 
                      (discountAmount : ℕ) : ℕ :=
  let totalFruits := lemonQty + papayaQty + mangoQty
  let totalCost := lemonPrice * lemonQty + papayaPrice * papayaQty + mangoPrice * mangoQty
  let discountCount := totalFruits / discountPerFruits
  let totalDiscount := discountCount * discountAmount
  totalCost - totalDiscount

theorem fruit_purchase_theorem : 
  fruitPurchaseCost 2 1 4 6 4 2 4 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_theorem_l3495_349556


namespace NUMINAMATH_CALUDE_roselyn_initial_books_l3495_349595

/-- The number of books Roselyn initially had -/
def initial_books : ℕ := 220

/-- The number of books Rebecca received -/
def rebecca_books : ℕ := 40

/-- The number of books Mara received -/
def mara_books : ℕ := 3 * rebecca_books

/-- The number of books Roselyn remained with -/
def remaining_books : ℕ := 60

/-- Theorem stating that the initial number of books Roselyn had is 220 -/
theorem roselyn_initial_books :
  initial_books = mara_books + rebecca_books + remaining_books :=
by sorry

end NUMINAMATH_CALUDE_roselyn_initial_books_l3495_349595


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3495_349569

def A : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}
def B : Set (ℝ × ℝ) := {p | p.2 = -2 * p.1 + 4}

theorem intersection_of_A_and_B : A ∩ B = {(1, 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3495_349569


namespace NUMINAMATH_CALUDE_march_first_is_sunday_l3495_349513

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific March with given properties -/
structure SpecificMarch where
  daysInMonth : Nat
  wednesdayCount : Nat
  saturdayCount : Nat
  firstDay : DayOfWeek

/-- Theorem stating that March 1st is a Sunday given the conditions -/
theorem march_first_is_sunday (march : SpecificMarch) 
  (h1 : march.daysInMonth = 31)
  (h2 : march.wednesdayCount = 4)
  (h3 : march.saturdayCount = 4) :
  march.firstDay = DayOfWeek.Sunday := by
  sorry

#check march_first_is_sunday

end NUMINAMATH_CALUDE_march_first_is_sunday_l3495_349513


namespace NUMINAMATH_CALUDE_reciprocal_equals_self_l3495_349501

theorem reciprocal_equals_self (x : ℝ) : (1 / x = x) → (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equals_self_l3495_349501


namespace NUMINAMATH_CALUDE_trajectory_of_B_l3495_349570

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space of the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a given line -/
def Point.isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Defines a parallelogram ABCD -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point
  is_parallelogram : (B.x - A.x = D.x - C.x) ∧ (B.y - A.y = D.y - C.y)

/-- Theorem: Trajectory of point B in a parallelogram ABCD -/
theorem trajectory_of_B (ABCD : Parallelogram)
  (hA : ABCD.A = Point.mk (-1) 3)
  (hC : ABCD.C = Point.mk (-3) 2)
  (hD : ABCD.D.isOnLine (Line.mk 1 (-3) 1)) :
  ABCD.B.isOnLine (Line.mk 1 (-3) 20) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_B_l3495_349570


namespace NUMINAMATH_CALUDE_function_equality_l3495_349585

/-- Given a function f such that f(2x) = 2 / (2 + x) for all x > 0,
    prove that 2f(x) = 8 / (4 + x) -/
theorem function_equality (f : ℝ → ℝ) 
    (h : ∀ x > 0, f (2 * x) = 2 / (2 + x)) :
  ∀ x > 0, 2 * f x = 8 / (4 + x) := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l3495_349585


namespace NUMINAMATH_CALUDE_half_cutting_line_exists_l3495_349528

/-- Triangle ABC with vertices A(0, 10), B(4, 0), and C(10, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- A line represented by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

/-- The area of a triangle given its vertices -/
def triangle_area (t : Triangle) : ℝ := sorry

/-- Check if a line cuts a triangle in half -/
def cuts_in_half (l : Line) (t : Triangle) : Prop := sorry

/-- The theorem stating the existence of a line that cuts the triangle in half
    and the sum of its slope and y-intercept -/
theorem half_cutting_line_exists (t : Triangle) 
  (h1 : t.A = (0, 10))
  (h2 : t.B = (4, 0))
  (h3 : t.C = (10, 0)) :
  ∃ l : Line, cuts_in_half l t ∧ l.slope + l.y_intercept = 5.625 := by
  sorry

end NUMINAMATH_CALUDE_half_cutting_line_exists_l3495_349528


namespace NUMINAMATH_CALUDE_iphone_savings_l3495_349526

def iphone_cost : ℝ := 600
def discount_rate : ℝ := 0.05
def num_phones : ℕ := 3

def individual_cost : ℝ := iphone_cost * num_phones
def discounted_cost : ℝ := individual_cost * (1 - discount_rate)
def savings : ℝ := individual_cost - discounted_cost

theorem iphone_savings : savings = 90 := by
  sorry

end NUMINAMATH_CALUDE_iphone_savings_l3495_349526


namespace NUMINAMATH_CALUDE_vehicle_value_fraction_l3495_349543

theorem vehicle_value_fraction (value_this_year value_last_year : ℚ) 
  (h1 : value_this_year = 16000)
  (h2 : value_last_year = 20000) :
  value_this_year / value_last_year = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_vehicle_value_fraction_l3495_349543


namespace NUMINAMATH_CALUDE_range_of_x_less_than_6_range_of_a_for_f_less_than_a_l3495_349510

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 3| + |x + 1|

-- Theorem for part I
theorem range_of_x_less_than_6 :
  ∀ x : ℝ, f x < 6 ↔ x ∈ Set.Ioo (-2 : ℝ) 4 :=
sorry

-- Theorem for part II
theorem range_of_a_for_f_less_than_a :
  ∀ a : ℝ, (∃ x : ℝ, f x < a) ↔ a ∈ Set.Ioi 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_less_than_6_range_of_a_for_f_less_than_a_l3495_349510


namespace NUMINAMATH_CALUDE_angle_D_is_100_l3495_349560

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.D + t.E + t.F = 180

def angle_E_is_30 (t : Triangle) : Prop :=
  t.E = 30

def angle_D_twice_F (t : Triangle) : Prop :=
  t.D = 2 * t.F

-- Theorem statement
theorem angle_D_is_100 (t : Triangle) 
  (h1 : is_right_triangle t) 
  (h2 : angle_E_is_30 t) 
  (h3 : angle_D_twice_F t) : 
  t.D = 100 :=
sorry

end NUMINAMATH_CALUDE_angle_D_is_100_l3495_349560


namespace NUMINAMATH_CALUDE_sqrt_x_minus_8_real_l3495_349566

theorem sqrt_x_minus_8_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 8) ↔ x ≥ 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_8_real_l3495_349566


namespace NUMINAMATH_CALUDE_salary_decrease_percentage_l3495_349531

theorem salary_decrease_percentage (x : ℝ) : 
  (100 - x) / 100 * 130 / 100 = 65 / 100 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_salary_decrease_percentage_l3495_349531


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3495_349505

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The left vertex of the hyperbola -/
def left_vertex (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- An asymptote of the hyperbola -/
def asymptote (h : Hyperbola a b) : Set (ℝ × ℝ) := sorry

/-- The projection of a point onto a line -/
def project (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The area of a triangle formed by three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) :
  let A := left_vertex h
  let F := right_focus h
  let asym := asymptote h
  let B := project A asym
  let Q := project F asym
  let O := (0, 0)
  triangle_area A B O / triangle_area F Q O = 1 / 2 →
  eccentricity h = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3495_349505


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l3495_349534

/-- Calculates the time for a monkey to climb a tree given the tree height,
    hop distance, slip distance, and net climb rate per hour. -/
def monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) (net_climb_rate : ℕ) : ℕ :=
  (tree_height - hop_distance) / net_climb_rate + 1

theorem monkey_climb_theorem (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) :
  tree_height = 22 →
  hop_distance = 3 →
  slip_distance = 2 →
  monkey_climb_time tree_height hop_distance slip_distance (hop_distance - slip_distance) = 20 := by
  sorry

#eval monkey_climb_time 22 3 2 1

end NUMINAMATH_CALUDE_monkey_climb_theorem_l3495_349534


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l3495_349533

theorem unique_five_digit_number : ∀ N : ℕ,
  (10000 ≤ N ∧ N < 100000) →
  let P := 200000 + N
  let Q := 10 * N + 2
  Q = 3 * P →
  N = 85714 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l3495_349533


namespace NUMINAMATH_CALUDE_problem_statement_l3495_349542

theorem problem_statement :
  (∀ x : ℝ, x^2 - 4*x + 5 > 0) ∧
  (∃ x : ℤ, 3*x^2 - 2*x - 1 = 0) ∧
  (¬ ∃ x : ℚ, x^2 = 5) ∧
  (¬ ∀ x : ℝ, x + 1/x > 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3495_349542


namespace NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_for_x_squared_gt_9_l3495_349512

theorem x_gt_3_sufficient_not_necessary_for_x_squared_gt_9 :
  (∀ x : ℝ, x > 3 → x^2 > 9) ∧ (∃ x : ℝ, x^2 > 9 ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_for_x_squared_gt_9_l3495_349512


namespace NUMINAMATH_CALUDE_salary_spending_problem_l3495_349517

/-- The problem statement about salaries and spending --/
theorem salary_spending_problem 
  (total_salary : ℝ)
  (a_salary : ℝ)
  (a_spend_percent : ℝ)
  (ha_total : total_salary = 6000)
  (ha_salary : a_salary = 4500)
  (ha_spend : a_spend_percent = 0.95)
  (h_equal_savings : a_salary * (1 - a_spend_percent) = (total_salary - a_salary) - ((total_salary - a_salary) * (85 / 100))) :
  (((total_salary - a_salary) - ((total_salary - a_salary) * (1 - 85 / 100))) / (total_salary - a_salary)) * 100 = 85 := by
sorry


end NUMINAMATH_CALUDE_salary_spending_problem_l3495_349517


namespace NUMINAMATH_CALUDE_ping_pong_theorem_l3495_349520

/-- Represents a ping-pong match result between two players -/
inductive MatchResult
| Win
| Lose

/-- Represents a ping-pong team -/
def Team := Fin 1000

/-- Represents the result of all matches between two teams -/
def MatchResults := Team → Team → MatchResult

theorem ping_pong_theorem (results : MatchResults) : 
  ∃ (winning_team : Bool) (subset : Finset Team),
    subset.card ≤ 10 ∧ 
    ∀ (player : Team), 
      ∃ (winner : Team), winner ∈ subset ∧ 
        (if winning_team then 
          results winner player = MatchResult.Win
        else
          results player winner = MatchResult.Lose) :=
sorry

end NUMINAMATH_CALUDE_ping_pong_theorem_l3495_349520


namespace NUMINAMATH_CALUDE_integer_sum_l3495_349522

theorem integer_sum (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_l3495_349522


namespace NUMINAMATH_CALUDE_solution_set_l3495_349558

-- Define the variables
variable (a b : ℝ)

-- Define the conditions
def condition1 : Prop := ∀ x : ℝ, (a - b) * x + a + 2 * b > 0 ↔ x > 1 / 2
def condition2 : Prop := a > 0

-- Define the theorem
theorem solution_set (h1 : condition1 a b) (h2 : condition2 a) :
  ∀ x : ℝ, a * x < b ↔ x < -1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_l3495_349558


namespace NUMINAMATH_CALUDE_min_value_theorem_l3495_349539

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ 1/x + 1/y = 1 → 1/(a-1) + 9/(b-1) ≤ 1/(x-1) + 9/(y-1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3495_349539


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3495_349503

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x : ℝ, (x - 2)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) → 
  a₀ + a₂ + a₄ = -122 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3495_349503


namespace NUMINAMATH_CALUDE_max_distance_circle_ellipse_l3495_349506

/-- The maximum distance between any point on the circle x^2 + (y-6)^2 = 2 
    and any point on the ellipse x^2/10 + y^2 = 1 is 6√2 -/
theorem max_distance_circle_ellipse : 
  ∃ (max_dist : ℝ), max_dist = 6 * Real.sqrt 2 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + (y₁ - 6)^2 = 2) →  -- Point on the circle
    (x₂^2 / 10 + y₂^2 = 1) →   -- Point on the ellipse
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_ellipse_l3495_349506


namespace NUMINAMATH_CALUDE_value_of_a_l3495_349577

theorem value_of_a (A B : Set ℝ) (a : ℝ) : 
  A = {0, 2, a} → 
  B = {1, a^2} → 
  A ∪ B = {0, 1, 2, 4, 16} → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3495_349577


namespace NUMINAMATH_CALUDE_unique_number_between_zero_and_two_l3495_349544

theorem unique_number_between_zero_and_two : 
  ∃! (n : ℕ), n ≤ 9 ∧ n > 0 ∧ n < 2 := by sorry

end NUMINAMATH_CALUDE_unique_number_between_zero_and_two_l3495_349544


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l3495_349508

theorem min_value_squared_sum (a b t s : ℝ) (h1 : a + b = t) (h2 : a - b = s) :
  a^2 + b^2 = (t^2 + s^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l3495_349508


namespace NUMINAMATH_CALUDE_magic_square_sum_l3495_349568

/-- Represents a 3x3 magic square with given values and variables -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  sum : ℕ
  row1_eq : sum = 20 + e + 18
  row2_eq : sum = 15 + c + d
  row3_eq : sum = a + 25 + b
  col1_eq : sum = 20 + 15 + a
  col2_eq : sum = e + c + 25
  col3_eq : sum = 18 + d + b
  diag1_eq : sum = 20 + c + b
  diag2_eq : sum = a + c + 18

/-- Theorem: In the given magic square, d + e = 42 -/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 42 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l3495_349568


namespace NUMINAMATH_CALUDE_square_root_of_nine_l3495_349521

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l3495_349521


namespace NUMINAMATH_CALUDE_password_probability_l3495_349555

/-- Represents the probability of using password A in week k -/
def P (k : ℕ) : ℚ :=
  3/4 * (-1/3)^(k-1) + 1/4

/-- The problem statement -/
theorem password_probability : P 7 = 61/243 := by
  sorry

end NUMINAMATH_CALUDE_password_probability_l3495_349555


namespace NUMINAMATH_CALUDE_factors_of_210_l3495_349550

def number_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factors_of_210 : number_of_factors 210 = 16 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_210_l3495_349550


namespace NUMINAMATH_CALUDE_max_gcd_sum_1980_l3495_349529

theorem max_gcd_sum_1980 :
  ∃ (a b : ℕ+), a + b = 1980 ∧
  ∀ (c d : ℕ+), c + d = 1980 → Nat.gcd c d ≤ Nat.gcd a b ∧
  Nat.gcd a b = 990 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1980_l3495_349529


namespace NUMINAMATH_CALUDE_smallest_multiple_36_45_not_11_l3495_349552

theorem smallest_multiple_36_45_not_11 : ∃ (n : ℕ), 
  n > 0 ∧ 
  36 ∣ n ∧ 
  45 ∣ n ∧ 
  ¬(11 ∣ n) ∧ 
  ∀ (m : ℕ), m > 0 ∧ 36 ∣ m ∧ 45 ∣ m ∧ ¬(11 ∣ m) → n ≤ m :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_36_45_not_11_l3495_349552


namespace NUMINAMATH_CALUDE_roots_sum_of_sixth_powers_l3495_349588

theorem roots_sum_of_sixth_powers (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 3 + 1 = 0 →
  s^2 - 2*s*Real.sqrt 3 + 1 = 0 →
  r ≠ s →
  r^6 + s^6 = 970 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_of_sixth_powers_l3495_349588


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3495_349590

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (10/9) * x^2 + (4/9) * x + 4/9

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q (-2) = 4 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

#eval q (-2)
#eval q 1
#eval q 3

end NUMINAMATH_CALUDE_q_satisfies_conditions_l3495_349590


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3495_349549

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ (p : ℚ) / q < 5 / 8 →
  q ≥ 13 ∧ (q = 13 → p = 8) :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3495_349549


namespace NUMINAMATH_CALUDE_fraction_comparison_l3495_349582

theorem fraction_comparison : (5 / 8 : ℚ) - (1 / 16 : ℚ) > (5 / 9 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3495_349582


namespace NUMINAMATH_CALUDE_sequence_non_positive_l3495_349537

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n ≥ 3)
  (h_start : a 1 = 0)
  (h_end : a n = 0)
  (h_ineq : ∀ k : ℕ, 2 ≤ k ∧ k ≤ n - 1 → a (k - 1) + a (k + 1) ≥ 2 * a k) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l3495_349537


namespace NUMINAMATH_CALUDE_remainder_of_2583156_div_4_l3495_349500

theorem remainder_of_2583156_div_4 : 2583156 % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2583156_div_4_l3495_349500


namespace NUMINAMATH_CALUDE_sum_of_powers_l3495_349511

theorem sum_of_powers (w : ℂ) (hw : w^3 + w^2 + 1 = 0) :
  w^100 + w^101 + w^102 + w^103 + w^104 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l3495_349511


namespace NUMINAMATH_CALUDE_chess_group_players_l3495_349535

/-- The number of players in a chess group -/
def num_players : ℕ := 8

/-- The total number of games played -/
def total_games : ℕ := 28

/-- Calculates the number of games played given the number of players -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_group_players :
  (games_played num_players = total_games) ∧ 
  (∀ m : ℕ, m ≠ num_players → games_played m ≠ total_games) :=
sorry

end NUMINAMATH_CALUDE_chess_group_players_l3495_349535


namespace NUMINAMATH_CALUDE_horner_method_v1_l3495_349540

def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

def horner_v1 (a : ℝ) : ℝ := 3 * a + 0

theorem horner_method_v1 :
  let x : ℝ := 10
  horner_v1 x = 30 := by sorry

end NUMINAMATH_CALUDE_horner_method_v1_l3495_349540


namespace NUMINAMATH_CALUDE_cinema_rows_l3495_349573

def base8_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem cinema_rows :
  let total_seats : ℕ := base8_to_decimal 351
  let seats_per_row : ℕ := 3
  (total_seats / seats_per_row : ℕ) = 77 := by
sorry

end NUMINAMATH_CALUDE_cinema_rows_l3495_349573


namespace NUMINAMATH_CALUDE_total_cans_of_peas_l3495_349571

-- Define the number of cans per box
def cans_per_box : ℕ := 4

-- Define the number of boxes ordered
def boxes_ordered : ℕ := 203

-- Theorem to prove
theorem total_cans_of_peas : cans_per_box * boxes_ordered = 812 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_of_peas_l3495_349571


namespace NUMINAMATH_CALUDE_b_age_is_ten_l3495_349559

/-- Given the ages of three people a, b, and c, prove that b is 10 years old. -/
theorem b_age_is_ten (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 27 → 
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_b_age_is_ten_l3495_349559


namespace NUMINAMATH_CALUDE_min_value_of_f_l3495_349518

/-- The quadratic function we want to minimize -/
def f (x y : ℝ) : ℝ := 3*x^2 + 4*x*y + 2*y^2 - 6*x + 8*y + 10

/-- The theorem stating the minimum value of the function -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -2/3 ∧ ∀ (x y : ℝ), f x y ≥ min := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3495_349518


namespace NUMINAMATH_CALUDE_chloe_treasures_l3495_349562

theorem chloe_treasures (points_per_treasure : ℕ) (second_level_treasures : ℕ) (total_score : ℕ) 
  (h1 : points_per_treasure = 9)
  (h2 : second_level_treasures = 3)
  (h3 : total_score = 81) :
  total_score = points_per_treasure * (second_level_treasures + 6) :=
by sorry

end NUMINAMATH_CALUDE_chloe_treasures_l3495_349562


namespace NUMINAMATH_CALUDE_shoes_total_price_l3495_349546

/-- Given the conditions of Jeff's purchase, prove the total price of shoes. -/
theorem shoes_total_price (total_cost : ℕ) (shoe_pairs : ℕ) (jerseys : ℕ) 
  (h1 : total_cost = 560)
  (h2 : shoe_pairs = 6)
  (h3 : jerseys = 4)
  (h4 : ∃ (shoe_price : ℚ), total_cost = shoe_pairs * shoe_price + jerseys * (shoe_price / 4)) :
  shoe_pairs * (total_cost / (shoe_pairs + jerseys / 4 : ℚ)) = 480 := by
  sorry

#check shoes_total_price

end NUMINAMATH_CALUDE_shoes_total_price_l3495_349546


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3495_349515

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 3 + a^(x + 2)
  f (-2) = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3495_349515


namespace NUMINAMATH_CALUDE_point_on_line_with_given_y_l3495_349504

/-- A straight line in the xy-plane with given slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem point_on_line_with_given_y (l : Line) (p : Point) :
  l.slope = 4 →
  l.yIntercept = 100 →
  p.y = 300 →
  pointOnLine l p →
  p.x = 50 := by
  sorry

#check point_on_line_with_given_y

end NUMINAMATH_CALUDE_point_on_line_with_given_y_l3495_349504


namespace NUMINAMATH_CALUDE_upstream_rate_calculation_l3495_349599

/-- Represents the rowing rates and current speed in kilometers per hour. -/
structure RowingScenario where
  downstream_rate : ℝ
  still_water_rate : ℝ
  current_rate : ℝ

/-- Calculates the upstream rate given a RowingScenario. -/
def upstream_rate (scenario : RowingScenario) : ℝ :=
  scenario.still_water_rate - scenario.current_rate

/-- Theorem stating that for the given scenario, the upstream rate is 10 kmph. -/
theorem upstream_rate_calculation (scenario : RowingScenario) 
  (h1 : scenario.downstream_rate = 30)
  (h2 : scenario.still_water_rate = 20)
  (h3 : scenario.current_rate = 10) :
  upstream_rate scenario = 10 := by
  sorry

#check upstream_rate_calculation

end NUMINAMATH_CALUDE_upstream_rate_calculation_l3495_349599


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3495_349579

theorem unique_triple_solution : 
  ∃! (n p q : ℕ), n ≥ 2 ∧ n^p + n^q = n^2010 ∧ p = 2009 ∧ q = 2009 ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3495_349579


namespace NUMINAMATH_CALUDE_problem_solution_l3495_349516

theorem problem_solution (x : ℝ) : 3 * x = (26 - x) + 10 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3495_349516


namespace NUMINAMATH_CALUDE_root_property_l3495_349598

theorem root_property (a : ℝ) (h : 2 * a^2 - 3 * a - 5 = 0) : -4 * a^2 + 6 * a = -10 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l3495_349598


namespace NUMINAMATH_CALUDE_function_value_proof_l3495_349584

/-- Given a function f(x) = x^5 - ax^3 + bx - 6 where f(-2) = 10, prove that f(2) = -22 -/
theorem function_value_proof (a b : ℝ) : 
  let f := λ x : ℝ => x^5 - a*x^3 + b*x - 6
  f (-2) = 10 → f 2 = -22 := by
sorry

end NUMINAMATH_CALUDE_function_value_proof_l3495_349584


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3495_349572

theorem infinitely_many_solutions (d : ℝ) : 
  (∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) ↔ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l3495_349572


namespace NUMINAMATH_CALUDE_copper_alloy_impossibility_l3495_349564

/-- Proves the impossibility of creating a specific copper alloy mixture --/
theorem copper_alloy_impossibility : ∀ (x : ℝ),
  0 ≤ x ∧ x ≤ 100 →
  32 * 0.25 + 8 * (x / 100) ≠ 40 * 0.45 :=
by
  sorry

#check copper_alloy_impossibility

end NUMINAMATH_CALUDE_copper_alloy_impossibility_l3495_349564


namespace NUMINAMATH_CALUDE_gcd_problem_l3495_349551

theorem gcd_problem :
  ∃! n : ℕ, 80 ≤ n ∧ n ≤ 100 ∧ Nat.gcd n 27 = 9 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3495_349551


namespace NUMINAMATH_CALUDE_simplify_sqrt_seven_simplify_sqrt_fraction_simplify_sqrt_sum_simplify_sqrt_expression_l3495_349574

-- Problem 1
theorem simplify_sqrt_seven : 2 * Real.sqrt 7 - 6 * Real.sqrt 7 = -4 * Real.sqrt 7 := by sorry

-- Problem 2
theorem simplify_sqrt_fraction : Real.sqrt (2/3) / Real.sqrt (8/27) = 3/2 := by sorry

-- Problem 3
theorem simplify_sqrt_sum : Real.sqrt 18 + Real.sqrt 98 - Real.sqrt 27 = 10 * Real.sqrt 2 - 3 * Real.sqrt 3 := by sorry

-- Problem 4
theorem simplify_sqrt_expression : 
  (Real.sqrt 0.5 + Real.sqrt 6) - (Real.sqrt (1/8) - Real.sqrt 24) = Real.sqrt 2 / 4 + 3 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_seven_simplify_sqrt_fraction_simplify_sqrt_sum_simplify_sqrt_expression_l3495_349574


namespace NUMINAMATH_CALUDE_final_milk_water_ratio_l3495_349524

/- Given conditions -/
def initial_ratio : Rat := 1 / 5
def can_capacity : ℝ := 8
def additional_milk : ℝ := 2

/- Theorem to prove -/
theorem final_milk_water_ratio :
  let initial_mixture := can_capacity - additional_milk
  let initial_milk := initial_mixture * (initial_ratio / (1 + initial_ratio))
  let initial_water := initial_mixture * (1 / (1 + initial_ratio))
  let final_milk := initial_milk + additional_milk
  let final_water := initial_water
  (final_milk / final_water) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_final_milk_water_ratio_l3495_349524


namespace NUMINAMATH_CALUDE_sequence_property_l3495_349554

/-- Given a sequence a_n and S_n where a_{n+1} = 3S_n for all n ≥ 1,
    prove that a_n can be arithmetic but not geometric -/
theorem sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) = 3 * S n) :
    (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
    (¬ ∃ r : ℝ, r ≠ 1 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = r * a n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l3495_349554


namespace NUMINAMATH_CALUDE_angle_X_measure_l3495_349593

-- Define the angles in the configuration
def angle_Y : ℝ := 130
def angle_60 : ℝ := 60
def right_angle : ℝ := 90

-- Theorem statement
theorem angle_X_measure :
  ∀ (angle_X : ℝ),
  -- Conditions
  (angle_Y + (180 - angle_Y) = 180) →  -- Y and Z form a linear pair
  (angle_X + angle_60 + right_angle = 180) →  -- Sum of angles in the smaller triangle
  -- Conclusion
  angle_X = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_X_measure_l3495_349593


namespace NUMINAMATH_CALUDE_valid_solutions_l3495_349586

-- Define digits as natural numbers from 0 to 9
def Digit : Type := { n : ℕ // n ≤ 9 }

-- Define the conditions for each case
def case1_conditions (x y z : Digit) : Prop :=
  (10 * x.val + y.val = 3 * (10 * y.val + z.val)) ∧
  (x.val + y.val = y.val + z.val + 3)

def case2_conditions (x y z : Digit) : Prop :=
  (10 * x.val + y.val = 3 * (10 * z.val + x.val)) ∧
  (x.val + y.val = x.val + z.val + 3 ∨ x.val + y.val = x.val + z.val - 3)

def case3_conditions (x y z : Digit) : Prop :=
  (10 * x.val + y.val = 3 * (10 * x.val + z.val)) ∧
  (x.val + y.val = x.val + z.val + 3 ∨ x.val + y.val = x.val + z.val - 3)

def case4_conditions (x y z : Digit) : Prop :=
  (10 * x.val + y.val = 3 * (10 * z.val + y.val)) ∧
  (x.val + y.val = z.val + y.val + 3 ∨ x.val + y.val = z.val + y.val - 3)

-- Main theorem
theorem valid_solutions :
  ∀ (a b : ℕ) (x y z : Digit),
    a > b →
    (case1_conditions x y z ∨ case2_conditions x y z ∨ case3_conditions x y z ∨ case4_conditions x y z) →
    ((a = 72 ∧ b = 24) ∨ (a = 45 ∧ b = 15)) := by
  sorry

end NUMINAMATH_CALUDE_valid_solutions_l3495_349586


namespace NUMINAMATH_CALUDE_shirt_problem_l3495_349589

/-- Represents the problem of determining the number of shirts and minimum selling price --/
theorem shirt_problem (first_batch_cost second_batch_cost : ℕ) 
  (h1 : first_batch_cost = 13200)
  (h2 : second_batch_cost = 28800)
  (h3 : ∃ x : ℕ, x > 0 ∧ second_batch_cost / (2 * x) = first_batch_cost / x + 10)
  (h4 : ∃ y : ℕ, y > 0 ∧ 350 * y ≥ (first_batch_cost + second_batch_cost) * 125 / 100) :
  (∃ x : ℕ, x = 120 ∧ second_batch_cost / (2 * x) = first_batch_cost / x + 10) ∧
  (∃ y : ℕ, y = 150 ∧ 350 * y ≥ (first_batch_cost + second_batch_cost) * 125 / 100) :=
by sorry


end NUMINAMATH_CALUDE_shirt_problem_l3495_349589


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3495_349592

/-- The imaginary part of (2+i)/i is -2 -/
theorem imaginary_part_of_complex_fraction : Complex.im ((2 : Complex) + Complex.I) / Complex.I = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3495_349592


namespace NUMINAMATH_CALUDE_percentage_not_french_l3495_349530

def total_students : ℕ := 200
def french_and_english : ℕ := 25
def french_not_english : ℕ := 65

theorem percentage_not_french : 
  (total_students - (french_and_english + french_not_english)) * 100 / total_students = 55 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_french_l3495_349530


namespace NUMINAMATH_CALUDE_custom_op_value_l3495_349519

/-- Custom operation * for non-zero integers -/
def custom_op (a b : ℤ) : ℚ := 1 / a + 1 / b

theorem custom_op_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 12) (h4 : a * b = 32) :
  custom_op a b = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_value_l3495_349519


namespace NUMINAMATH_CALUDE_ping_pong_game_ratio_l3495_349578

/-- Given that Frankie and Carla played 30 games of ping pong,
    and Carla won 20 games, prove that the ratio of games
    Frankie won to games Carla won is 1:2. -/
theorem ping_pong_game_ratio :
  let total_games : ℕ := 30
  let carla_wins : ℕ := 20
  let frankie_wins : ℕ := total_games - carla_wins
  (frankie_wins : ℚ) / (carla_wins : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_game_ratio_l3495_349578


namespace NUMINAMATH_CALUDE_boyfriend_texts_l3495_349594

theorem boyfriend_texts (total : ℕ) (grocery : ℕ) : 
  total = 33 → 
  grocery + 5 * grocery + (grocery + 5 * grocery) / 10 = total → 
  grocery = 5 := by
  sorry

end NUMINAMATH_CALUDE_boyfriend_texts_l3495_349594


namespace NUMINAMATH_CALUDE_expression_value_l3495_349576

theorem expression_value : 100 * (100 - 3) - (100 * 100 - 3) = -297 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3495_349576


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l3495_349541

theorem triangle_determinant_zero (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) : 
  let M : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l3495_349541


namespace NUMINAMATH_CALUDE_function_non_negative_iff_a_leq_four_l3495_349583

theorem function_non_negative_iff_a_leq_four (a : ℝ) :
  (∀ x : ℝ, 2^(2*x) - a * 2^x + 4 ≥ 0) ↔ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_function_non_negative_iff_a_leq_four_l3495_349583


namespace NUMINAMATH_CALUDE_residue_of_negative_1001_mod_37_l3495_349514

theorem residue_of_negative_1001_mod_37 :
  -1001 ≡ 35 [ZMOD 37] := by sorry

end NUMINAMATH_CALUDE_residue_of_negative_1001_mod_37_l3495_349514


namespace NUMINAMATH_CALUDE_sqrt_square_abs_l3495_349565

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_l3495_349565


namespace NUMINAMATH_CALUDE_horner_method_for_f_at_2_l3495_349587

def f (x : ℝ) : ℝ := 2 * x^5 + 3 * x^4 + 2 * x^3 - 4 * x + 5

theorem horner_method_for_f_at_2 : f 2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_for_f_at_2_l3495_349587


namespace NUMINAMATH_CALUDE_min_distance_sum_l3495_349580

noncomputable def parabola (x y : ℝ) : Prop := x^2 = 4*y

def focus : ℝ × ℝ := (0, 1)

def point_A : ℝ × ℝ := (2, 3)

theorem min_distance_sum (P : ℝ × ℝ) :
  parabola P.1 P.2 →
  Real.sqrt ((P.1 - point_A.1)^2 + (P.2 - point_A.2)^2) +
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_sum_l3495_349580


namespace NUMINAMATH_CALUDE_committee_selection_count_l3495_349596

theorem committee_selection_count : Nat.choose 12 7 = 792 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_count_l3495_349596


namespace NUMINAMATH_CALUDE_simplify_expressions_l3495_349525

theorem simplify_expressions :
  ∀ (x a b : ℝ),
    (x^2 + (3*x - 5) - (4*x - 1) = x^2 - x - 4) ∧
    (7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3495_349525


namespace NUMINAMATH_CALUDE_students_in_all_workshops_l3495_349547

theorem students_in_all_workshops (total : ℕ) (robotics dance music : ℕ) (at_least_two : ℕ) 
  (h_total : total = 25)
  (h_robotics : robotics = 15)
  (h_dance : dance = 12)
  (h_music : music = 10)
  (h_at_least_two : at_least_two = 11)
  (h_sum : robotics + dance + music - 2 * at_least_two ≤ total) :
  ∃ (only_one only_two all_three : ℕ),
    only_one + only_two + all_three = total ∧
    only_two + 3 * all_three = at_least_two ∧
    all_three = 1 :=
by sorry

end NUMINAMATH_CALUDE_students_in_all_workshops_l3495_349547


namespace NUMINAMATH_CALUDE_darias_initial_savings_l3495_349557

def couch_price : ℕ := 750
def table_price : ℕ := 100
def lamp_price : ℕ := 50
def remaining_debt : ℕ := 400

def total_furniture_cost : ℕ := couch_price + table_price + lamp_price

theorem darias_initial_savings : total_furniture_cost - remaining_debt = 500 := by
  sorry

end NUMINAMATH_CALUDE_darias_initial_savings_l3495_349557


namespace NUMINAMATH_CALUDE_equation_solution_l3495_349507

theorem equation_solution (x y : ℚ) : 
  (0.009 / x = 0.01 / y) → (x + y = 50) → (x = 450 / 19 ∧ y = 500 / 19) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3495_349507


namespace NUMINAMATH_CALUDE_ramu_car_price_l3495_349532

/-- Proves that given the conditions of Ramu's car purchase, repair, and sale,
    the original price he paid for the car is 42000 rupees. -/
theorem ramu_car_price :
  let repair_cost : ℝ := 12000
  let selling_price : ℝ := 64900
  let profit_percent : ℝ := 20.185185185185187
  let original_price : ℝ := 42000
  (selling_price = original_price + repair_cost + (original_price + repair_cost) * (profit_percent / 100)) →
  original_price = 42000 :=
by
  sorry

#check ramu_car_price

end NUMINAMATH_CALUDE_ramu_car_price_l3495_349532


namespace NUMINAMATH_CALUDE_total_diagonals_specific_prism_l3495_349509

/-- A rectangular prism with edge lengths a, b, and c. -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of face diagonals in a rectangular prism. -/
def face_diagonals (p : RectangularPrism) : ℕ := 12

/-- The number of space diagonals in a rectangular prism. -/
def space_diagonals (p : RectangularPrism) : ℕ := 4

/-- The total number of diagonals in a rectangular prism. -/
def total_diagonals (p : RectangularPrism) : ℕ :=
  face_diagonals p + space_diagonals p

/-- Theorem: The total number of diagonals in a rectangular prism
    with edge lengths 4, 6, and 8 is 16. -/
theorem total_diagonals_specific_prism :
  ∃ p : RectangularPrism, p.a = 4 ∧ p.b = 6 ∧ p.c = 8 ∧ total_diagonals p = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_diagonals_specific_prism_l3495_349509
