import Mathlib

namespace NUMINAMATH_CALUDE_distance_2_5_distance_neg2_5_distance_x_3_solutions_abs_x_minus_1_int_solutions_sum_distances_min_value_sum_distances_l245_24554

-- Define the distance function
def distance (a b : ℚ) : ℚ := |a - b|

-- Theorem 1: Distance between 2 and 5 is 3
theorem distance_2_5 : distance 2 5 = 3 := by sorry

-- Theorem 2: Distance between -2 and 5 is 7
theorem distance_neg2_5 : distance (-2) 5 = 7 := by sorry

-- Theorem 3: |x-3| represents the distance between x and 3
theorem distance_x_3 (x : ℚ) : |x - 3| = distance x 3 := by sorry

-- Theorem 4: Solutions of |x-1| = 3
theorem solutions_abs_x_minus_1 (x : ℚ) : |x - 1| = 3 ↔ x = 4 ∨ x = -2 := by sorry

-- Theorem 5: Integer solutions of |x-1| + |x+2| = 3
theorem int_solutions_sum_distances (x : ℤ) : 
  |x - 1| + |x + 2| = 3 ↔ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 := by sorry

-- Theorem 6: Minimum value of |x+8| + |x-3| + |x-6|
theorem min_value_sum_distances :
  ∃ (x : ℚ), ∀ (y : ℚ), |x + 8| + |x - 3| + |x - 6| ≤ |y + 8| + |y - 3| + |y - 6| ∧
  |x + 8| + |x - 3| + |x - 6| = 14 := by sorry

end NUMINAMATH_CALUDE_distance_2_5_distance_neg2_5_distance_x_3_solutions_abs_x_minus_1_int_solutions_sum_distances_min_value_sum_distances_l245_24554


namespace NUMINAMATH_CALUDE_james_purchase_cost_l245_24553

def shirts_count : ℕ := 10
def shirt_price : ℕ := 6
def pants_price : ℕ := 8

def pants_count : ℕ := shirts_count / 2

def total_cost : ℕ := shirts_count * shirt_price + pants_count * pants_price

theorem james_purchase_cost : total_cost = 100 := by
  sorry

end NUMINAMATH_CALUDE_james_purchase_cost_l245_24553


namespace NUMINAMATH_CALUDE_angle_215_in_third_quadrant_l245_24533

def angle_in_third_quadrant (angle : ℝ) : Prop :=
  180 < angle ∧ angle ≤ 270

theorem angle_215_in_third_quadrant :
  angle_in_third_quadrant 215 :=
sorry

end NUMINAMATH_CALUDE_angle_215_in_third_quadrant_l245_24533


namespace NUMINAMATH_CALUDE_checker_arrangement_count_l245_24503

/-- The number of ways to arrange white and black checkers on a chessboard -/
def checker_arrangements : ℕ := 
  let total_squares : ℕ := 32
  let white_checkers : ℕ := 12
  let black_checkers : ℕ := 12
  Nat.factorial total_squares / (Nat.factorial white_checkers * Nat.factorial black_checkers * Nat.factorial (total_squares - white_checkers - black_checkers))

/-- Theorem stating that the number of ways to arrange 12 white and 12 black checkers
    on 32 black squares of a chessboard is equal to (32! / (12! * 12! * 8!)) -/
theorem checker_arrangement_count : 
  checker_arrangements = Nat.factorial 32 / (Nat.factorial 12 * Nat.factorial 12 * Nat.factorial 8) :=
by sorry

end NUMINAMATH_CALUDE_checker_arrangement_count_l245_24503


namespace NUMINAMATH_CALUDE_student_count_l245_24502

theorem student_count : ∃! n : ℕ, n < 50 ∧ n % 8 = 5 ∧ n % 4 = 1 ∧ n = 45 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l245_24502


namespace NUMINAMATH_CALUDE_annas_cupcake_sales_l245_24591

/-- Anna's cupcake sales problem -/
theorem annas_cupcake_sales (num_trays : ℕ) (cupcakes_per_tray : ℕ) (price_per_cupcake : ℕ) (sold_fraction : ℚ) : 
  num_trays = 4 →
  cupcakes_per_tray = 20 →
  price_per_cupcake = 2 →
  sold_fraction = 3 / 5 →
  (num_trays * cupcakes_per_tray * sold_fraction * price_per_cupcake : ℚ) = 96 :=
by sorry

end NUMINAMATH_CALUDE_annas_cupcake_sales_l245_24591


namespace NUMINAMATH_CALUDE_floor_of_4_7_l245_24528

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l245_24528


namespace NUMINAMATH_CALUDE_august_matches_l245_24539

/-- Calculates the number of matches played in August given the initial and final winning percentages and the number of additional matches won. -/
def matches_in_august (initial_percentage : ℚ) (final_percentage : ℚ) (additional_wins : ℕ) : ℕ :=
  sorry

theorem august_matches :
  matches_in_august (22 / 100) (52 / 100) 75 = 120 :=
sorry

end NUMINAMATH_CALUDE_august_matches_l245_24539


namespace NUMINAMATH_CALUDE_basketball_cost_l245_24532

/-- The cost of each basketball given the total cost and soccer ball cost -/
theorem basketball_cost (total_cost : ℕ) (soccer_cost : ℕ) : 
  total_cost = 920 ∧ soccer_cost = 65 → (total_cost - 8 * soccer_cost) / 5 = 80 := by
  sorry

#check basketball_cost

end NUMINAMATH_CALUDE_basketball_cost_l245_24532


namespace NUMINAMATH_CALUDE_blocks_added_l245_24593

/-- 
Given:
- initial_blocks: The initial number of blocks in Adolfo's tower
- final_blocks: The final number of blocks in Adolfo's tower

Prove that the number of blocks added is equal to the difference between 
the final and initial number of blocks.
-/
theorem blocks_added (initial_blocks final_blocks : ℕ) 
  (h1 : initial_blocks = 35)
  (h2 : final_blocks = 65) : 
  final_blocks - initial_blocks = 30 := by
  sorry

end NUMINAMATH_CALUDE_blocks_added_l245_24593


namespace NUMINAMATH_CALUDE_f_value_at_2_l245_24534

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_value_at_2 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
    (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l245_24534


namespace NUMINAMATH_CALUDE_math_books_count_l245_24566

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℕ) :
  total_books = 90 →
  math_cost = 4 →
  history_cost = 5 →
  total_price = 396 →
  ∃ (math_books : ℕ), 
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧ 
    math_books = 54 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l245_24566


namespace NUMINAMATH_CALUDE_min_value_a_b_squared_l245_24575

/-- Given that the ratio of the absolute values of the coefficients of x² and x³ terms
    in the expansion of (1/a + ax)⁵ - (1/b + bx)⁵ is 1:6, 
    the minimum value of a² + b² is 12 -/
theorem min_value_a_b_squared (a b : ℝ) (h : ∃ k : ℝ, k > 0 ∧ 
  |5 * (1/a^2 - 1/b^2)| = k ∧ |10 * (a - b)| = 6*k) : 
  a^2 + b^2 ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_b_squared_l245_24575


namespace NUMINAMATH_CALUDE_four_corner_holes_l245_24519

/-- Represents the state of a rectangular paper. -/
structure Paper where
  folded : Bool
  holes : List (Nat × Nat)

/-- Represents the folding operations. -/
inductive FoldOperation
  | BottomToTop
  | LeftToRight
  | TopToBottom

/-- Folds the paper according to the given operation. -/
def fold (p : Paper) (op : FoldOperation) : Paper :=
  { p with folded := true }

/-- Punches a hole in the top left corner of the folded paper. -/
def punchHole (p : Paper) : Paper :=
  { p with holes := (0, 0) :: p.holes }

/-- Unfolds the paper and calculates the final hole positions. -/
def unfold (p : Paper) : Paper :=
  { p with 
    folded := false,
    holes := [(0, 0), (0, 1), (1, 0), (1, 1)] }

/-- The main theorem stating that after folding, punching, and unfolding, 
    the paper will have four holes, one in each corner. -/
theorem four_corner_holes (p : Paper) :
  let p1 := fold p FoldOperation.BottomToTop
  let p2 := fold p1 FoldOperation.LeftToRight
  let p3 := fold p2 FoldOperation.TopToBottom
  let p4 := punchHole p3
  let final := unfold p4
  final.holes = [(0, 0), (0, 1), (1, 0), (1, 1)] :=
by sorry

end NUMINAMATH_CALUDE_four_corner_holes_l245_24519


namespace NUMINAMATH_CALUDE_circle_M_equation_l245_24573

-- Define the line on which point M lies
def line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 5

-- Define that points (3,0) and (0,1) lie on circle M
def points_on_circle : Prop := circle_M 3 0 ∧ circle_M 0 1

-- Theorem statement
theorem circle_M_equation : 
  ∃ (x y : ℝ), line x y ∧ points_on_circle → circle_M x y :=
sorry

end NUMINAMATH_CALUDE_circle_M_equation_l245_24573


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l245_24572

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 + 6*p^3 + 11*p^2 + 6*p + 3 = 0) →
  (q^4 + 6*q^3 + 11*q^2 + 6*q + 3 = 0) →
  (r^4 + 6*r^3 + 11*r^2 + 6*r + 3 = 0) →
  (s^4 + 6*s^3 + 11*s^2 + 6*s + 3 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l245_24572


namespace NUMINAMATH_CALUDE_smallest_prime_angle_in_right_triangle_l245_24585

/-- Checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Theorem: Smallest angle b in a right triangle with prime angles -/
theorem smallest_prime_angle_in_right_triangle :
  ∀ a b : ℕ,
  (a : ℝ) + (b : ℝ) = 90 →
  isPrime a →
  isPrime b →
  (a : ℝ) > (b : ℝ) + 2 →
  b ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_angle_in_right_triangle_l245_24585


namespace NUMINAMATH_CALUDE_bedroom_set_price_l245_24516

def original_price : ℝ := 2000
def gift_card : ℝ := 200
def first_discount_rate : ℝ := 0.15
def second_discount_rate : ℝ := 0.10

def final_price : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  price_after_second_discount - gift_card

theorem bedroom_set_price : final_price = 1330 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_set_price_l245_24516


namespace NUMINAMATH_CALUDE_train_length_l245_24597

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 30 → time = 24 → ∃ length : ℝ, abs (length - 199.92) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l245_24597


namespace NUMINAMATH_CALUDE_division_problem_l245_24599

theorem division_problem (x : ℝ) : 75 / x = 1500 → x = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l245_24599


namespace NUMINAMATH_CALUDE_applicant_age_standard_deviation_l245_24510

theorem applicant_age_standard_deviation
  (average_age : ℝ)
  (max_different_ages : ℕ)
  (h_average : average_age = 31)
  (h_max_ages : max_different_ages = 11) :
  let standard_deviation := (max_different_ages - 1) / 2
  standard_deviation = 5 := by
  sorry

end NUMINAMATH_CALUDE_applicant_age_standard_deviation_l245_24510


namespace NUMINAMATH_CALUDE_polynomial_independent_implies_m_plus_n_squared_l245_24550

/-- A polynomial that is independent of x -/
def polynomial (m n x y : ℝ) : ℝ := 4*m*x^2 + 5*x - 2*y^2 + 8*x^2 - n*x + y - 1

/-- The polynomial is independent of x -/
def independent_of_x (m n : ℝ) : Prop :=
  ∀ x y : ℝ, ∃ c : ℝ, ∀ x' : ℝ, polynomial m n x' y = c

/-- The main theorem -/
theorem polynomial_independent_implies_m_plus_n_squared (m n : ℝ) :
  independent_of_x m n → (m + n)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_independent_implies_m_plus_n_squared_l245_24550


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_l245_24556

theorem smallest_multiple_of_seven (x y : ℤ) 
  (h1 : (x + 1) % 7 = 0) 
  (h2 : (y - 5) % 7 = 0) : 
  (∃ n : ℕ+, (x^2 + x*y + y^2 + 3*n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 + x*y + y^2 + 3*m) % 7 = 0 → n ≤ m) → 
  (∃ n : ℕ+, (x^2 + x*y + y^2 + 3*n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 + x*y + y^2 + 3*m) % 7 = 0 → n ≤ m) ∧ 
  (∃ n : ℕ+, (x^2 + x*y + y^2 + 3*n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 + x*y + y^2 + 3*m) % 7 = 0 → n ≤ m) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_l245_24556


namespace NUMINAMATH_CALUDE_cubic_identity_l245_24513

theorem cubic_identity (a b c : ℝ) : 
  (b + c - 2 * a)^3 + (c + a - 2 * b)^3 + (a + b - 2 * c)^3 = 
  (b + c - 2 * a) * (c + a - 2 * b) * (a + b - 2 * c) := by sorry

end NUMINAMATH_CALUDE_cubic_identity_l245_24513


namespace NUMINAMATH_CALUDE_tank_capacity_is_21600_l245_24549

/-- The capacity of a tank with specific inlet and outlet pipe properties -/
def tank_capacity : ℝ := by
  -- Define the time to empty the tank with only the outlet pipe open
  let outlet_time : ℝ := 10

  -- Define the inlet pipe rate in litres per minute
  let inlet_rate_per_minute : ℝ := 16

  -- Define the time to empty the tank with both pipes open
  let both_pipes_time : ℝ := 18

  -- Calculate the inlet rate in litres per hour
  let inlet_rate_per_hour : ℝ := inlet_rate_per_minute * 60

  -- The capacity of the tank
  exact 21600

/-- Theorem stating that the tank capacity is 21,600 litres -/
theorem tank_capacity_is_21600 : tank_capacity = 21600 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_is_21600_l245_24549


namespace NUMINAMATH_CALUDE_max_colors_upper_bound_l245_24526

/-- 
Given a positive integer n ≥ 2, an n × n × n cube is divided into n³ unit cubes, 
each colored with one color. For each n × n × 1 rectangular prism (in 3 orientations), 
consider the set of colors appearing in this prism. For any color set in one group, 
it also appears in each of the other two groups. 
This theorem states the upper bound for the maximum number of colors.
-/
theorem max_colors_upper_bound (n : ℕ) (h : n ≥ 2) : 
  ∃ C : ℕ, C ≤ n * (n + 1) * (2 * n + 1) / 6 ∧ 
  (∀ D : ℕ, D ≤ n * (n + 1) * (2 * n + 1) / 6) :=
by sorry

end NUMINAMATH_CALUDE_max_colors_upper_bound_l245_24526


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l245_24541

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), (1056 + n) % 25 = 0 ∧ 
  ∀ (m : ℕ), m < n → (1056 + m) % 25 ≠ 0 :=
by
  use 19
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l245_24541


namespace NUMINAMATH_CALUDE_total_tape_theorem_l245_24524

/-- The amount of tape needed for a rectangular box -/
def tape_for_rect_box (length width : ℕ) : ℕ := 2 * width + length

/-- The amount of tape needed for a square box -/
def tape_for_square_box (side : ℕ) : ℕ := 3 * side

/-- The total amount of tape needed for multiple boxes -/
def total_tape_needed (rect_boxes square_boxes : ℕ) (rect_length rect_width square_side : ℕ) : ℕ :=
  rect_boxes * tape_for_rect_box rect_length rect_width +
  square_boxes * tape_for_square_box square_side

theorem total_tape_theorem :
  total_tape_needed 5 2 30 15 40 = 540 :=
by sorry

end NUMINAMATH_CALUDE_total_tape_theorem_l245_24524


namespace NUMINAMATH_CALUDE_farm_animals_difference_l245_24588

theorem farm_animals_difference (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 6 * initial_cows →
  (initial_horses - 30) = 4 * (initial_cows + 30) →
  (initial_horses - 30) - (initial_cows + 30) = 315 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_difference_l245_24588


namespace NUMINAMATH_CALUDE_sum_of_twos_and_threes_1800_l245_24570

/-- The number of ways to represent a positive integer as a sum of 2s and 3s -/
def waysToSum (n : ℕ) : ℕ :=
  (n / 6 + 1)

/-- 1800 can be represented as a sum of 2s and 3s in 301 ways -/
theorem sum_of_twos_and_threes_1800 : waysToSum 1800 = 301 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_twos_and_threes_1800_l245_24570


namespace NUMINAMATH_CALUDE_phi_value_l245_24596

theorem phi_value : ∃ (Φ : ℕ), 504 / Φ = 40 + 3 * Φ ∧ 0 ≤ Φ ∧ Φ ≤ 9 ∧ Φ = 8 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_l245_24596


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l245_24500

theorem complex_number_quadrant : ∀ z : ℂ, 
  (3 - Complex.I) * z = 1 - 2 * Complex.I →
  0 < z.re ∧ z.im < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l245_24500


namespace NUMINAMATH_CALUDE_fred_total_cents_l245_24531

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of dimes Fred has -/
def fred_dimes : ℕ := 9

/-- Theorem: Fred's total cents is 90 -/
theorem fred_total_cents : fred_dimes * dime_value = 90 := by
  sorry

end NUMINAMATH_CALUDE_fred_total_cents_l245_24531


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l245_24586

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l245_24586


namespace NUMINAMATH_CALUDE_multiple_of_ten_implies_multiple_of_five_l245_24507

theorem multiple_of_ten_implies_multiple_of_five 
  (h1 : ∀ n : ℕ, 10 ∣ n → 5 ∣ n) 
  (a : ℕ) 
  (h2 : 10 ∣ a) : 
  5 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_ten_implies_multiple_of_five_l245_24507


namespace NUMINAMATH_CALUDE_unique_function_theorem_l245_24520

/-- A function from rational numbers to rational numbers -/
def RationalFunction := ℚ → ℚ

/-- The property that a function satisfies the given conditions -/
def SatisfiesConditions (f : RationalFunction) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1

/-- The theorem statement -/
theorem unique_function_theorem :
  ∀ f : RationalFunction, SatisfiesConditions f → ∀ x : ℚ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l245_24520


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_zero_l245_24587

theorem ceiling_floor_sum_zero : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_zero_l245_24587


namespace NUMINAMATH_CALUDE_centroid_property_l245_24547

/-- Given a triangle PQR with vertices P(-2,4), Q(6,3), and R(2,-5),
    prove that if S(x,y) is the centroid of the triangle, then 7x + 3y = 16 -/
theorem centroid_property (P Q R S : ℝ × ℝ) (x y : ℝ) :
  P = (-2, 4) →
  Q = (6, 3) →
  R = (2, -5) →
  S = (x, y) →
  S = ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3) →
  7 * x + 3 * y = 16 := by
sorry

end NUMINAMATH_CALUDE_centroid_property_l245_24547


namespace NUMINAMATH_CALUDE_warrens_event_capacity_l245_24589

theorem warrens_event_capacity :
  let total_tables : ℕ := 252
  let large_tables : ℕ := 93
  let medium_tables : ℕ := 97
  let small_tables : ℕ := total_tables - large_tables - medium_tables
  let unusable_small_tables : ℕ := 20
  let usable_small_tables : ℕ := small_tables - unusable_small_tables
  let large_table_capacity : ℕ := 6
  let medium_table_capacity : ℕ := 5
  let small_table_capacity : ℕ := 4
  
  large_tables * large_table_capacity +
  medium_tables * medium_table_capacity +
  usable_small_tables * small_table_capacity = 1211 :=
by
  sorry

#eval
  let total_tables : ℕ := 252
  let large_tables : ℕ := 93
  let medium_tables : ℕ := 97
  let small_tables : ℕ := total_tables - large_tables - medium_tables
  let unusable_small_tables : ℕ := 20
  let usable_small_tables : ℕ := small_tables - unusable_small_tables
  let large_table_capacity : ℕ := 6
  let medium_table_capacity : ℕ := 5
  let small_table_capacity : ℕ := 4
  
  large_tables * large_table_capacity +
  medium_tables * medium_table_capacity +
  usable_small_tables * small_table_capacity

end NUMINAMATH_CALUDE_warrens_event_capacity_l245_24589


namespace NUMINAMATH_CALUDE_bus_trip_distance_l245_24505

/-- The distance of a bus trip given specific speed conditions -/
theorem bus_trip_distance : ∃ (d : ℝ), 
  (d / 45 = d / 50 + 1) ∧ d = 450 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_distance_l245_24505


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l245_24537

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l245_24537


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l245_24544

/-- Given two vectors m and n in ℝ², if m + n is perpendicular to m, then the second component of n is -3. -/
theorem vector_perpendicular_condition (m n : ℝ × ℝ) :
  m = (1, 2) →
  n.1 = a →
  n.2 = -1 →
  (m + n) • m = 0 →
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l245_24544


namespace NUMINAMATH_CALUDE_point_movement_l245_24578

/-- Given point A(-1, 3), moving it 5 units down and 2 units to the left results in point B(-3, -2) -/
theorem point_movement (A B : ℝ × ℝ) : 
  A = (-1, 3) → 
  B.1 = A.1 - 2 → 
  B.2 = A.2 - 5 → 
  B = (-3, -2) := by
sorry

end NUMINAMATH_CALUDE_point_movement_l245_24578


namespace NUMINAMATH_CALUDE_fixed_points_subset_stable_points_quadratic_no_fixed_points_implies_no_stable_points_l245_24565

/-- Fixed points of a function -/
def fixed_points (f : ℝ → ℝ) : Set ℝ := {x | f x = x}

/-- Stable points of a function -/
def stable_points (f : ℝ → ℝ) : Set ℝ := {x | f (f x) = x}

theorem fixed_points_subset_stable_points (f : ℝ → ℝ) :
  fixed_points f ⊆ stable_points f := by sorry

theorem quadratic_no_fixed_points_implies_no_stable_points
  (a b c : ℝ) (h : a ≠ 0) (f : ℝ → ℝ) (hf : ∀ x, f x = a * x^2 + b * x + c) :
  fixed_points f = ∅ → stable_points f = ∅ := by sorry

end NUMINAMATH_CALUDE_fixed_points_subset_stable_points_quadratic_no_fixed_points_implies_no_stable_points_l245_24565


namespace NUMINAMATH_CALUDE_coefficient_d_nonzero_l245_24536

/-- A polynomial of degree 5 with specific properties -/
def Q (a b c d f : ℝ) (x : ℝ) : ℝ :=
  x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + f

/-- Theorem: For a polynomial Q with five distinct x-intercepts, including (0,0) and (1,0), 
    the coefficient d must be non-zero -/
theorem coefficient_d_nonzero 
  (a b c d f : ℝ) 
  (h1 : Q a b c d f 0 = 0)
  (h2 : Q a b c d f 1 = 0)
  (h3 : ∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ 
       p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ p ≠ 1 ∧ q ≠ 1 ∧ r ≠ 1 ∧
       ∀ x : ℝ, Q a b c d f x = x * (x - 1) * (x - p) * (x - q) * (x - r)) : 
  d ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_coefficient_d_nonzero_l245_24536


namespace NUMINAMATH_CALUDE_parallelogram_product_l245_24523

structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ
  x : ℝ
  z : ℝ
  h_EF : EF = 46
  h_FG : FG z = 4 * z^3 + 1
  h_GH : GH x = 3 * x + 6
  h_HE : HE = 35
  h_opposite_sides_equal : EF = GH x ∧ FG z = HE

theorem parallelogram_product (p : Parallelogram) :
  p.x * p.z = (40/3) * Real.rpow 8.5 (1/3) := by sorry

end NUMINAMATH_CALUDE_parallelogram_product_l245_24523


namespace NUMINAMATH_CALUDE_line_equations_l245_24576

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ

-- Define the point A
def A : Point := (1, -3)

-- Define the reference line
def reference_line : Line := λ x y ↦ 2*x - y + 4

-- Define the properties of lines l and m
def parallel (l1 l2 : Line) : Prop := ∃ k : ℝ, ∀ x y, l1 x y = k * l2 x y
def perpendicular (l1 l2 : Line) : Prop := ∃ k : ℝ, ∀ x y, l1 x y * l2 x y = -k

-- Define the y-intercept of a line
def y_intercept (l : Line) : ℝ := l 0 1

-- State the theorem
theorem line_equations (l m : Line) : 
  (∃ k : ℝ, l A.fst A.snd = 0) →  -- l passes through A
  parallel l reference_line →     -- l is parallel to reference_line
  perpendicular l m →             -- m is perpendicular to l
  y_intercept m = 3 →             -- m has y-intercept 3
  (∀ x y, l x y = 2*x - y - 5) ∧  -- equation of l
  (∀ x y, m x y = x + 2*y - 6)    -- equation of m
  := by sorry

end NUMINAMATH_CALUDE_line_equations_l245_24576


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l245_24580

theorem power_fraction_simplification :
  (3^100 + 3^98) / (3^100 - 3^98) = 5/4 := by
sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l245_24580


namespace NUMINAMATH_CALUDE_money_distribution_ratio_l245_24509

def distribute_money (total : ℝ) (p q r s : ℝ) : Prop :=
  p + q + r + s = total ∧
  p = 2 * q ∧
  q = r ∧
  s - p = 250

theorem money_distribution_ratio :
  ∀ (total p q r s : ℝ),
    total = 1000 →
    distribute_money total p q r s →
    s / r = 4 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_ratio_l245_24509


namespace NUMINAMATH_CALUDE_bus_catch_probability_l245_24598

/-- The probability of catching a bus within 5 minutes -/
theorem bus_catch_probability 
  (p3 : ℝ) -- Probability of bus No. 3 arriving
  (p6 : ℝ) -- Probability of bus No. 6 arriving
  (h1 : p3 = 0.20) -- Given probability for bus No. 3
  (h2 : p6 = 0.60) -- Given probability for bus No. 6
  (h3 : 0 ≤ p3 ∧ p3 ≤ 1) -- p3 is a valid probability
  (h4 : 0 ≤ p6 ∧ p6 ≤ 1) -- p6 is a valid probability
  : p3 + p6 = 0.80 := by
  sorry

end NUMINAMATH_CALUDE_bus_catch_probability_l245_24598


namespace NUMINAMATH_CALUDE_value_of_expression_l245_24535

theorem value_of_expression : 8 + 2 * (3^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l245_24535


namespace NUMINAMATH_CALUDE_steven_more_peaches_l245_24582

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 17

/-- The number of apples Steven has -/
def steven_apples : ℕ := 16

/-- Jake has 6 fewer peaches than Steven -/
def jake_peaches : ℕ := steven_peaches - 6

/-- Jake has 8 more apples than Steven -/
def jake_apples : ℕ := steven_apples + 8

/-- Theorem: Steven has 1 more peach than apples -/
theorem steven_more_peaches : steven_peaches - steven_apples = 1 := by
  sorry

end NUMINAMATH_CALUDE_steven_more_peaches_l245_24582


namespace NUMINAMATH_CALUDE_total_toys_is_15_8_l245_24530

-- Define the initial number of toys and daily changes
def initial_toys : ℝ := 5.3
def tuesday_remaining_percent : ℝ := 0.605
def tuesday_new_toys : ℝ := 3.6
def wednesday_loss_percent : ℝ := 0.502
def wednesday_new_toys : ℝ := 2.4
def thursday_loss_percent : ℝ := 0.308
def thursday_new_toys : ℝ := 4.5

-- Define the function to calculate the total number of toys
def total_toys : ℝ :=
  let tuesday_toys := initial_toys * tuesday_remaining_percent + tuesday_new_toys
  let wednesday_toys := tuesday_toys * (1 - wednesday_loss_percent) + wednesday_new_toys
  let thursday_toys := wednesday_toys * (1 - thursday_loss_percent) + thursday_new_toys
  let lost_tuesday := initial_toys - initial_toys * tuesday_remaining_percent
  let lost_wednesday := tuesday_toys - tuesday_toys * (1 - wednesday_loss_percent)
  let lost_thursday := wednesday_toys - wednesday_toys * (1 - thursday_loss_percent)
  thursday_toys + lost_tuesday + lost_wednesday + lost_thursday

-- Theorem statement
theorem total_toys_is_15_8 : total_toys = 15.8 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_is_15_8_l245_24530


namespace NUMINAMATH_CALUDE_minimum_packaging_volume_l245_24511

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the packaging problem parameters -/
structure PackagingProblem where
  boxDimensions : BoxDimensions
  costPerBox : ℝ
  minTotalCost : ℝ

theorem minimum_packaging_volume (p : PackagingProblem) 
  (h1 : p.boxDimensions.length = 20)
  (h2 : p.boxDimensions.width = 20)
  (h3 : p.boxDimensions.height = 12)
  (h4 : p.costPerBox = 0.4)
  (h5 : p.minTotalCost = 200) :
  (p.minTotalCost / p.costPerBox) * boxVolume p.boxDimensions = 2400000 := by
  sorry

#check minimum_packaging_volume

end NUMINAMATH_CALUDE_minimum_packaging_volume_l245_24511


namespace NUMINAMATH_CALUDE_gig_song_ratio_l245_24555

/-- Proves that the ratio of the length of the last song to the length of the first two songs is 3:1 --/
theorem gig_song_ratio :
  let days_in_two_weeks : ℕ := 14
  let gigs_in_two_weeks : ℕ := days_in_two_weeks / 2
  let songs_per_gig : ℕ := 3
  let length_of_first_two_songs : ℕ := 2 * 5
  let total_playing_time : ℕ := 280
  let total_length_first_two_songs : ℕ := gigs_in_two_weeks * length_of_first_two_songs
  let total_length_third_song : ℕ := total_playing_time - total_length_first_two_songs
  let length_third_song_per_gig : ℕ := total_length_third_song / gigs_in_two_weeks
  length_third_song_per_gig / length_of_first_two_songs = 3 := by
  sorry

end NUMINAMATH_CALUDE_gig_song_ratio_l245_24555


namespace NUMINAMATH_CALUDE_child_running_speed_l245_24521

/-- Verify the child's running speed on a still sidewalk -/
theorem child_running_speed 
  (distance_with : ℝ)
  (distance_against : ℝ)
  (time_against : ℝ)
  (speed_still : ℝ)
  (h1 : distance_with = 372)
  (h2 : distance_against = 165)
  (h3 : time_against = 3)
  (h4 : speed_still = 74)
  (h5 : ∃ t, t > 0 ∧ (speed_still + (distance_against / time_against - speed_still)) * t = distance_with)
  (h6 : (speed_still - (distance_against / time_against - speed_still)) * time_against = distance_against) :
  speed_still = 74 := by
sorry

end NUMINAMATH_CALUDE_child_running_speed_l245_24521


namespace NUMINAMATH_CALUDE_smallest_sum_4x4x4_cube_l245_24538

/-- Represents a 4x4x4 cube made of dice -/
structure LargeCube where
  size : Nat
  dice_count : Nat
  opposite_sides_sum : Nat

/-- Calculates the smallest possible sum of visible faces on the large cube -/
def smallest_visible_sum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the smallest possible sum for a 4x4x4 cube of dice -/
theorem smallest_sum_4x4x4_cube (cube : LargeCube) 
  (h1 : cube.size = 4)
  (h2 : cube.dice_count = 64)
  (h3 : cube.opposite_sides_sum = 7) :
  smallest_visible_sum cube = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_4x4x4_cube_l245_24538


namespace NUMINAMATH_CALUDE_only_cat_owners_count_l245_24512

/-- The number of people owning only cats in a pet ownership scenario. -/
def num_only_cat_owners : ℕ := 
  let total_pet_owners : ℕ := 59
  let only_dog_owners : ℕ := 15
  let cat_and_dog_owners : ℕ := 5
  let cat_dog_snake_owners : ℕ := 3
  total_pet_owners - (only_dog_owners + cat_and_dog_owners + cat_dog_snake_owners)

/-- Theorem stating that the number of people owning only cats is 36. -/
theorem only_cat_owners_count : num_only_cat_owners = 36 := by
  sorry

end NUMINAMATH_CALUDE_only_cat_owners_count_l245_24512


namespace NUMINAMATH_CALUDE_cos_greater_when_sin_greater_in_second_quadrant_l245_24594

theorem cos_greater_when_sin_greater_in_second_quadrant 
  (α β : Real) 
  (h1 : π/2 < α ∧ α < π) 
  (h2 : π/2 < β ∧ β < π) 
  (h3 : Real.sin α > Real.sin β) : 
  Real.cos α > Real.cos β := by
sorry

end NUMINAMATH_CALUDE_cos_greater_when_sin_greater_in_second_quadrant_l245_24594


namespace NUMINAMATH_CALUDE_abs_negative_2023_l245_24563

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l245_24563


namespace NUMINAMATH_CALUDE_complement_A_in_U_equals_open_interval_l245_24517

-- Define the set U
def U : Set ℝ := {x | (x - 2) / x ≤ 1}

-- Define the set A
def A : Set ℝ := {x | 2 - x ≤ 1}

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := U \ A

-- Theorem statement
theorem complement_A_in_U_equals_open_interval :
  complement_A_in_U = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_equals_open_interval_l245_24517


namespace NUMINAMATH_CALUDE_closest_to_neg_sqrt_two_l245_24501

theorem closest_to_neg_sqrt_two :
  let options : List ℝ := [-2, -1, 0, 1]
  ∀ x ∈ options, |(-1) - (-Real.sqrt 2)| ≤ |x - (-Real.sqrt 2)| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_to_neg_sqrt_two_l245_24501


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l245_24577

-- Define ω as a complex number
variable (ω : ℂ)

-- Define the conditions
def omega_condition := ω^5 = 1 ∧ ω ≠ 1

-- Define α and β
def α := ω + ω^2
def β := ω^3 + ω^4

-- Define the theorem
theorem quadratic_coefficients (h : omega_condition ω) : 
  ∃ (p : ℝ × ℝ), p.1 = 0 ∧ p.2 = 2 ∧ 
  (α ω)^2 + p.1 * (α ω) + p.2 = 0 ∧ 
  (β ω)^2 + p.1 * (β ω) + p.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l245_24577


namespace NUMINAMATH_CALUDE_fraction_simplification_l245_24545

theorem fraction_simplification (x y : ℚ) (hx : x = 4/6) (hy : y = 8/12) :
  (6*x + 8*y) / (48*x*y) = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l245_24545


namespace NUMINAMATH_CALUDE_x_percent_of_z_l245_24558

-- Define the variables
variable (x y z : ℝ)

-- Define the conditions
def condition1 : Prop := x = 1.20 * y
def condition2 : Prop := y = 0.40 * z

-- State the theorem
theorem x_percent_of_z (h1 : condition1 x y) (h2 : condition2 y z) : x = 0.48 * z := by
  sorry

end NUMINAMATH_CALUDE_x_percent_of_z_l245_24558


namespace NUMINAMATH_CALUDE_limit_cosine_ratio_l245_24584

theorem limit_cosine_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |(1 - Real.cos (2*x)) / (Real.cos (7*x) - Real.cos (3*x)) + (1/10)| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_cosine_ratio_l245_24584


namespace NUMINAMATH_CALUDE_ascendant_function_theorem_l245_24542

/-- A function is ascendant if it is non-decreasing --/
def Ascendant (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem ascendant_function_theorem (f : ℝ → ℝ) 
  (h1 : Ascendant (fun x => f x - 3 * x))
  (h2 : Ascendant (fun x => f x - x^3)) :
  Ascendant (fun x => f x - x^2 - x) :=
sorry

end NUMINAMATH_CALUDE_ascendant_function_theorem_l245_24542


namespace NUMINAMATH_CALUDE_total_length_of_T_l245_24574

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; |(|(|x| - 3)| - 1)| + |(|(|y| - 3)| - 1)| = 2}

-- Define the total length function
def totalLength (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem total_length_of_T : totalLength T = 32 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_total_length_of_T_l245_24574


namespace NUMINAMATH_CALUDE_marie_age_proof_l245_24562

/-- Marie's age in years -/
def marie_age : ℚ := 8/3

/-- Liam's age in years -/
def liam_age : ℚ := 4 * marie_age

/-- Oliver's age in years -/
def oliver_age : ℚ := marie_age + 8

theorem marie_age_proof :
  (liam_age = 4 * marie_age) ∧
  (oliver_age = marie_age + 8) ∧
  (liam_age = oliver_age) →
  marie_age = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_marie_age_proof_l245_24562


namespace NUMINAMATH_CALUDE_beanie_baby_ratio_l245_24590

/-- The number of beanie babies Lori has -/
def lori_babies : ℕ := 300

/-- The total number of beanie babies Lori and Sydney have together -/
def total_babies : ℕ := 320

/-- The number of beanie babies Sydney has -/
def sydney_babies : ℕ := total_babies - lori_babies

/-- The ratio of Lori's beanie babies to Sydney's beanie babies -/
def beanie_ratio : ℚ := lori_babies / sydney_babies

theorem beanie_baby_ratio : beanie_ratio = 15 := by
  sorry

end NUMINAMATH_CALUDE_beanie_baby_ratio_l245_24590


namespace NUMINAMATH_CALUDE_saltwater_solution_volume_l245_24557

/-- Proves that the initial volume of a saltwater solution is 100 gallons, given the conditions stated in the problem. -/
theorem saltwater_solution_volume : ∃ (x : ℝ), 
  -- Initial salt concentration is 20%
  (0.2 * x = x * 0.2) ∧ 
  -- After evaporation, total volume is 3/4 of initial
  (3/4 * x = x * 3/4) ∧ 
  -- Final salt concentration is 33 1/3%
  ((0.2 * x + 10) / (3/4 * x + 15) = 1/3) ∧ 
  -- Initial volume is 100 gallons
  (x = 100) := by
  sorry

end NUMINAMATH_CALUDE_saltwater_solution_volume_l245_24557


namespace NUMINAMATH_CALUDE_rectangle_area_l245_24595

theorem rectangle_area (w : ℝ) (h : w > 0) : 
  w^2 + (3*w)^2 = 16^2 → w * (3*w) = 76.8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l245_24595


namespace NUMINAMATH_CALUDE_min_q_geq_half_l245_24504

def q (a : ℕ) : ℚ := ((48 - a) * (47 - a) + (a - 1) * (a - 2)) / (2 * 1653)

theorem min_q_geq_half (n : ℕ) (h : n ≥ 1 ∧ n ≤ 60) :
  (∀ a : ℕ, a ≥ 1 ∧ a ≤ 60 → q a ≥ 1/2 → a ≥ n) →
  q n ≥ 1/2 →
  n = 10 :=
sorry

end NUMINAMATH_CALUDE_min_q_geq_half_l245_24504


namespace NUMINAMATH_CALUDE_trig_expression_equals_negative_four_l245_24592

theorem trig_expression_equals_negative_four :
  (Real.sqrt 3 / Real.cos (10 * π / 180)) - (1 / Real.sin (10 * π / 180)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_negative_four_l245_24592


namespace NUMINAMATH_CALUDE_find_unknown_number_l245_24548

theorem find_unknown_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((60 + 35 + x) / 3) + 5 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_number_l245_24548


namespace NUMINAMATH_CALUDE_total_pens_l245_24540

theorem total_pens (black_pens blue_pens : ℕ) : 
  black_pens = 4 → blue_pens = 4 → black_pens + blue_pens = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_pens_l245_24540


namespace NUMINAMATH_CALUDE_area_of_specific_region_l245_24522

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bound by two circles and the x-axis -/
def areaOfRegion (c1 c2 : Circle) : ℝ :=
  sorry

theorem area_of_specific_region :
  let c1 : Circle := { center := (5, 5), radius := 5 }
  let c2 : Circle := { center := (10, 5), radius := 3 }
  areaOfRegion c1 c2 = 25 - 17 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_specific_region_l245_24522


namespace NUMINAMATH_CALUDE_fraction_equality_implies_value_l245_24525

theorem fraction_equality_implies_value (a : ℝ) : 
  a / (a + 45) = 0.82 → a = 205 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_value_l245_24525


namespace NUMINAMATH_CALUDE_gene_mutation_not_valid_reason_l245_24515

/-- Represents a genotype --/
inductive Genotype
  | AA
  | Aa
  | BB
  | Bb
  | AaBB
  | AaBb
  | AAB
  | AaB
  | AABb

/-- Represents possible reasons for missing genes --/
inductive MissingGeneReason
  | GeneMutation
  | ChromosomeNumberVariation
  | ChromosomeStructureVariation
  | MaleSexLinked

/-- Defines the genotypes of individuals A and B --/
def individualA : Genotype := Genotype.AaB
def individualB : Genotype := Genotype.AABb

/-- Determines if a reason is valid for explaining the missing gene --/
def isValidReason (reason : MissingGeneReason) (genotypeA : Genotype) (genotypeB : Genotype) : Prop :=
  match reason with
  | MissingGeneReason.GeneMutation => False
  | _ => True

/-- Theorem stating that gene mutation is not a valid reason for the missing gene --/
theorem gene_mutation_not_valid_reason :
  ¬(isValidReason MissingGeneReason.GeneMutation individualA individualB) := by
  sorry


end NUMINAMATH_CALUDE_gene_mutation_not_valid_reason_l245_24515


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l245_24564

theorem imaginary_part_of_z (z : ℂ) : (3 - 4*I)*z = Complex.abs (4 + 3*I) → Complex.im z = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l245_24564


namespace NUMINAMATH_CALUDE_sequence_problem_l245_24569

def S (n : ℕ) (k : ℕ) : ℚ := -1/2 * n^2 + k*n

theorem sequence_problem (k : ℕ) (h1 : k > 0) 
  (h2 : ∀ n : ℕ, S n k ≤ 8) 
  (h3 : ∃ n : ℕ, S n k = 8) :
  k = 4 ∧ ∀ n : ℕ, n ≥ 1 → ((-1/2 : ℚ) * n^2 + 4*n) - ((-1/2 : ℚ) * (n-1)^2 + 4*(n-1)) = 9/2 - n :=
sorry

end NUMINAMATH_CALUDE_sequence_problem_l245_24569


namespace NUMINAMATH_CALUDE_q_of_one_equals_five_l245_24529

/-- Given a function q : ℝ → ℝ that passes through the point (1, 5), prove that q(1) = 5 -/
theorem q_of_one_equals_five (q : ℝ → ℝ) (h : q 1 = 5) : q 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_q_of_one_equals_five_l245_24529


namespace NUMINAMATH_CALUDE_truth_probability_l245_24546

theorem truth_probability (pA pB pAB : ℝ) : 
  pA = 0.7 →
  pAB = 0.42 →
  pAB = pA * pB →
  pB = 0.6 :=
by
  sorry

end NUMINAMATH_CALUDE_truth_probability_l245_24546


namespace NUMINAMATH_CALUDE_abc_sum_888_l245_24581

theorem abc_sum_888 : 
  ∃! (a b c : Nat), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    (100 * a + 10 * b + c) + (100 * a + 10 * b + c) + (100 * a + 10 * b + c) = 888 ∧
    100 * a + 10 * b + c = 296 :=
by sorry

end NUMINAMATH_CALUDE_abc_sum_888_l245_24581


namespace NUMINAMATH_CALUDE_smallest_19_factor_number_is_78732_l245_24561

/-- A function that returns the number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The smallest positive integer with exactly 19 factors -/
def smallest_19_factor_number : ℕ+ := sorry

/-- Theorem stating that the smallest positive integer with exactly 19 factors is 78732 -/
theorem smallest_19_factor_number_is_78732 : 
  smallest_19_factor_number = 78732 ∧ num_factors smallest_19_factor_number = 19 := by sorry

end NUMINAMATH_CALUDE_smallest_19_factor_number_is_78732_l245_24561


namespace NUMINAMATH_CALUDE_expression_equality_l245_24571

theorem expression_equality (a b : ℝ) :
  (-a * b^2)^3 + a * b^2 * (a * b)^2 * (-2 * b)^2 = 3 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l245_24571


namespace NUMINAMATH_CALUDE_function_is_even_l245_24506

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_is_even (f : ℝ → ℝ) 
  (h : ∀ x y, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)) : 
  IsEven f := by
  sorry

end NUMINAMATH_CALUDE_function_is_even_l245_24506


namespace NUMINAMATH_CALUDE_polynomial_roots_equivalence_l245_24514

theorem polynomial_roots_equivalence :
  let p (x : ℝ) := 7 * x^4 - 48 * x^3 + 93 * x^2 - 48 * x + 7
  let y (x : ℝ) := x + 2 / x
  let q (y : ℝ) := 7 * y^2 - 48 * y + 47
  ∀ x : ℝ, x ≠ 0 →
    (p x = 0 ↔ ∃ y : ℝ, q y = 0 ∧ (x + 2 / x = y ∨ x + 2 / x = y)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_equivalence_l245_24514


namespace NUMINAMATH_CALUDE_equation_solution_l245_24583

theorem equation_solution : ∃! x : ℚ, 5 * (x - 4) = 3 * (3 - 3 * x) + 6 ∧ x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l245_24583


namespace NUMINAMATH_CALUDE_station_A_relay_ways_l245_24560

/-- Represents a communication station -/
inductive Station : Type
| A | B | C | D

/-- The number of stations excluding A -/
def num_other_stations : Nat := 3

/-- The total number of ways station A can relay the message -/
def total_relay_ways : Nat := 16

/-- Theorem stating the number of ways station A can relay the message -/
theorem station_A_relay_ways :
  (∀ s₁ s₂ : Station, s₁ ≠ s₂ → (∃ t : Nat, t > 0)) →  -- Stations can communicate pairwise
  (∀ s : Station, ∃ t : Nat, t > 0) →  -- Space station can send to any station
  (∀ s : Station, ∀ n : Nat, n > 1 → ¬∃ t : Nat, t > 0) →  -- No simultaneous transmissions
  (∃ n : Nat, n = 3) →  -- Three transmissions occurred
  (∀ s : Station, ∃ m : Nat, m > 0) →  -- All stations received the message
  total_relay_ways = (2^num_other_stations - 1) + num_other_stations * 2^(num_other_stations - 1) :=
by sorry

end NUMINAMATH_CALUDE_station_A_relay_ways_l245_24560


namespace NUMINAMATH_CALUDE_prime_pair_equation_solution_l245_24527

theorem prime_pair_equation_solution :
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    p^3 - q^5 = (p + q)^2 → 
    (p = 7 ∧ q = 3) := by
  sorry

end NUMINAMATH_CALUDE_prime_pair_equation_solution_l245_24527


namespace NUMINAMATH_CALUDE_average_cost_is_1_85_l245_24552

/-- Calculates the average cost per fruit given the prices and quantities of fruits, applying special offers --/
def average_cost_per_fruit (apple_price banana_price orange_price : ℚ) 
  (apple_qty banana_qty orange_qty : ℕ) : ℚ :=
  let apple_cost := apple_price * (apple_qty.div 10 * 10)
  let banana_cost := banana_price * banana_qty
  let orange_cost := orange_price * (orange_qty.div 3 * 3)
  let total_cost := apple_cost + banana_cost + orange_cost
  let total_fruits := apple_qty + banana_qty + orange_qty
  total_cost / total_fruits

/-- The average cost per fruit is $1.85 given the specified prices, quantities, and offers --/
theorem average_cost_is_1_85 :
  average_cost_per_fruit 2 1 3 12 4 4 = 37/20 := by
  sorry

end NUMINAMATH_CALUDE_average_cost_is_1_85_l245_24552


namespace NUMINAMATH_CALUDE_eulers_theorem_parallelepiped_l245_24567

/-- Represents a parallelepiped with edges a, b, c meeting at a vertex,
    face diagonals d, e, f, and space diagonal g. -/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  g : ℝ

/-- Euler's theorem for parallelepipeds:
    The sum of the squares of the edges and the space diagonal at one vertex
    is equal to the sum of the squares of the face diagonals. -/
theorem eulers_theorem_parallelepiped (p : Parallelepiped) :
  p.a^2 + p.b^2 + p.c^2 + p.g^2 = p.d^2 + p.e^2 + p.f^2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_theorem_parallelepiped_l245_24567


namespace NUMINAMATH_CALUDE_kennel_problem_l245_24559

theorem kennel_problem (total : ℕ) (long_fur : ℕ) (brown : ℕ) (neither : ℕ) 
  (h_total : total = 45)
  (h_long_fur : long_fur = 26)
  (h_brown : brown = 22)
  (h_neither : neither = 8) :
  long_fur + brown - (total - neither) = 11 :=
by sorry

end NUMINAMATH_CALUDE_kennel_problem_l245_24559


namespace NUMINAMATH_CALUDE_third_year_sample_size_l245_24508

/-- The number of third-year students to be sampled in a stratified sampling scenario -/
theorem third_year_sample_size 
  (total_students : ℕ) 
  (first_year_students : ℕ) 
  (sophomore_probability : ℚ) 
  (sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : first_year_students = 760)
  (h3 : sophomore_probability = 37/100)
  (h4 : sample_size = 20) :
  let sophomore_students : ℕ := (sophomore_probability * total_students).num.toNat
  let third_year_students : ℕ := total_students - first_year_students - sophomore_students
  (sample_size * third_year_students) / total_students = 5 :=
by sorry

end NUMINAMATH_CALUDE_third_year_sample_size_l245_24508


namespace NUMINAMATH_CALUDE_space_filling_tetrahedrons_octahedrons_l245_24543

/-- A regular tetrahedron -/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (edge_length_pos : edge_length > 0)

/-- A regular octahedron -/
structure RegularOctahedron :=
  (edge_length : ℝ)
  (edge_length_pos : edge_length > 0)

/-- A space-filling arrangement -/
structure SpaceFillingArrangement :=
  (tetrahedrons : Set RegularTetrahedron)
  (octahedrons : Set RegularOctahedron)

/-- No gaps or overlaps in the arrangement -/
def NoGapsOrOverlaps (arrangement : SpaceFillingArrangement) : Prop :=
  sorry

/-- All polyhedra in the arrangement are congruent and have equal edge lengths -/
def CongruentWithEqualEdges (arrangement : SpaceFillingArrangement) : Prop :=
  sorry

/-- The main theorem: There exists a space-filling arrangement of congruent regular tetrahedrons
    and regular octahedrons with equal edge lengths, without gaps or overlaps -/
theorem space_filling_tetrahedrons_octahedrons :
  ∃ (arrangement : SpaceFillingArrangement),
    CongruentWithEqualEdges arrangement ∧ NoGapsOrOverlaps arrangement :=
sorry

end NUMINAMATH_CALUDE_space_filling_tetrahedrons_octahedrons_l245_24543


namespace NUMINAMATH_CALUDE_squares_below_specific_line_l245_24518

/-- Represents a line in the coordinate plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of unit squares below a line in the first quadrant -/
def countSquaresBelowLine (l : Line) : ℕ :=
  sorry

theorem squares_below_specific_line :
  let l : Line := { a := 5, b := 195, c := 975 }
  countSquaresBelowLine l = 388 := by sorry

end NUMINAMATH_CALUDE_squares_below_specific_line_l245_24518


namespace NUMINAMATH_CALUDE_minimum_students_in_class_l245_24579

theorem minimum_students_in_class (boys girls : ℕ) : 
  boys > 0 → girls > 0 →
  2 * (boys / 2) = 3 * (girls / 3) →
  boys + girls ≥ 7 :=
by
  sorry

#check minimum_students_in_class

end NUMINAMATH_CALUDE_minimum_students_in_class_l245_24579


namespace NUMINAMATH_CALUDE_greatest_integer_gcd_eighteen_l245_24568

theorem greatest_integer_gcd_eighteen : ∃ n : ℕ, n < 200 ∧ n.gcd 18 = 6 ∧ ∀ m : ℕ, m < 200 → m.gcd 18 = 6 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_gcd_eighteen_l245_24568


namespace NUMINAMATH_CALUDE_cos_A_minus_sin_C_range_l245_24551

theorem cos_A_minus_sin_C_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle
  A + B + C = π ∧          -- Sum of angles in a triangle
  a = 2 * b * Real.sin A → -- Given condition
  -Real.sqrt 3 / 2 < Real.cos A - Real.sin C ∧ 
  Real.cos A - Real.sin C < 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_A_minus_sin_C_range_l245_24551
