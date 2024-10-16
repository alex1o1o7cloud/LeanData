import Mathlib

namespace NUMINAMATH_CALUDE_integer_fraction_characterization_l1276_127638

theorem integer_fraction_characterization (a b : ℕ) :
  (∃ k : ℤ, (a^3 + 1 : ℤ) = k * (2*a*b^2 + 1)) ↔
  (∃ n : ℕ, a = 2*n^2 + 1 ∧ b = n) :=
sorry

end NUMINAMATH_CALUDE_integer_fraction_characterization_l1276_127638


namespace NUMINAMATH_CALUDE_shifted_function_point_l1276_127623

-- Define a function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem shifted_function_point (h : f 1 = 1) : 
  f (5 - 4) = 1 := by sorry

end NUMINAMATH_CALUDE_shifted_function_point_l1276_127623


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1276_127639

-- Define the sets M and N
def M : Set ℝ := {x | -4 < x ∧ x < -2}
def N : Set ℝ := {x | x^2 + 5*x + 6 < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -3 < x ∧ x < -2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1276_127639


namespace NUMINAMATH_CALUDE_jessica_candy_distribution_l1276_127679

/-- The number of candies Jessica must remove to distribute them equally among her friends -/
def candies_to_remove (total : Nat) (friends : Nat) : Nat :=
  total % friends

theorem jessica_candy_distribution :
  candies_to_remove 30 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_candy_distribution_l1276_127679


namespace NUMINAMATH_CALUDE_parabola_focus_l1276_127693

/-- The parabola is defined by the equation y = (1/4)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

/-- The focus of a parabola with equation y = ax^2 has coordinates (0, 1/(4a)) -/
def is_focus (a x y : ℝ) : Prop := x = 0 ∧ y = 1 / (4 * a)

/-- Prove that the focus of the parabola y = (1/4)x^2 has coordinates (0, 1) -/
theorem parabola_focus :
  is_focus (1/4) 0 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1276_127693


namespace NUMINAMATH_CALUDE_time_after_2017_minutes_l1276_127656

def add_minutes (hours minutes add_minutes : ℕ) : ℕ × ℕ :=
  let total_minutes := hours * 60 + minutes + add_minutes
  let new_hours := (total_minutes / 60) % 24
  let new_minutes := total_minutes % 60
  (new_hours, new_minutes)

theorem time_after_2017_minutes : 
  add_minutes 20 17 2017 = (5, 54) := by
sorry

end NUMINAMATH_CALUDE_time_after_2017_minutes_l1276_127656


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_l1276_127648

/-- The value of p for which the focus of the parabola y² = 2px coincides with 
    the right focus of the hyperbola x²/4 - y²/5 = 1 -/
theorem parabola_hyperbola_focus (p : ℝ) : 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2/4 - y^2/5 = 1 ∧ 
   x = (Real.sqrt (4 + 5 : ℝ)) ∧ y = 0) → 
  p = 6 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_l1276_127648


namespace NUMINAMATH_CALUDE_range_of_a_range_of_g_l1276_127650

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 2*a + 12

-- Define the function g
def g (a : ℝ) : ℝ := (a + 1) * (|a - 1| + 2)

-- Theorem 1: Range of a
theorem range_of_a (h : ∀ x : ℝ, f a x ≥ 0) : a ∈ Set.Icc (-3/2) 2 :=
sorry

-- Theorem 2: Range of g(a)
theorem range_of_g : Set.range g = Set.Icc (-9/4) 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_g_l1276_127650


namespace NUMINAMATH_CALUDE_heartsuit_nested_equals_fourteen_l1276_127657

-- Define the ⊛ operation for positive real numbers
def heartsuit (x y : ℝ) : ℝ := x + 2 * y

-- State the theorem
theorem heartsuit_nested_equals_fourteen :
  heartsuit 2 (heartsuit 2 2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_nested_equals_fourteen_l1276_127657


namespace NUMINAMATH_CALUDE_function_not_linear_plus_integer_l1276_127603

theorem function_not_linear_plus_integer : 
  ∃ (f : ℚ → ℚ), 
    (∀ x y : ℚ, ∃ z : ℤ, f (x + y) - f x - f y = ↑z) ∧ 
    (¬ ∃ c : ℚ, ∀ x : ℚ, ∃ z : ℤ, f x - c * x = ↑z) := by
  sorry

end NUMINAMATH_CALUDE_function_not_linear_plus_integer_l1276_127603


namespace NUMINAMATH_CALUDE_smallest_number_proof_l1276_127677

def smallest_number : ℕ := 910314816600

theorem smallest_number_proof :
  (∀ i ∈ Finset.range 28, smallest_number % (i + 1) = 0) ∧
  smallest_number % 29 ≠ 0 ∧
  smallest_number % 30 ≠ 0 ∧
  (∀ n < smallest_number, 
    (∀ i ∈ Finset.range 28, n % (i + 1) = 0) →
    (n % 29 = 0 ∨ n % 30 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l1276_127677


namespace NUMINAMATH_CALUDE_interior_point_distance_l1276_127664

-- Define the rectangle and point
def Rectangle (E F G H : ℝ × ℝ) : Prop := sorry

def InteriorPoint (P : ℝ × ℝ) (E F G H : ℝ × ℝ) : Prop := 
  Rectangle E F G H ∧ sorry

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem interior_point_distance 
  (E F G H P : ℝ × ℝ) 
  (h_rect : Rectangle E F G H)
  (h_interior : InteriorPoint P E F G H)
  (h_PE : distance P E = 5)
  (h_PH : distance P H = 12)
  (h_PG : distance P G = 13) :
  distance P F = 12 := by
  sorry

end NUMINAMATH_CALUDE_interior_point_distance_l1276_127664


namespace NUMINAMATH_CALUDE_a_minus_b_squared_l1276_127661

theorem a_minus_b_squared (a b : ℝ) (h1 : (a + b)^2 = 49) (h2 : a * b = 6) : 
  (a - b)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_squared_l1276_127661


namespace NUMINAMATH_CALUDE_golden_rabbit_cards_l1276_127617

def total_cards : ℕ := 10000
def digits_without_6_8 : ℕ := 8

theorem golden_rabbit_cards :
  (total_cards - digits_without_6_8^4 : ℕ) = 5904 := by sorry

end NUMINAMATH_CALUDE_golden_rabbit_cards_l1276_127617


namespace NUMINAMATH_CALUDE_cookies_calculation_l1276_127624

/-- The number of people receiving cookies -/
def num_people : ℕ := 14

/-- The number of cookies each person receives -/
def cookies_per_person : ℕ := 30

/-- The total number of cookies prepared -/
def total_cookies : ℕ := num_people * cookies_per_person

theorem cookies_calculation : total_cookies = 420 := by
  sorry

end NUMINAMATH_CALUDE_cookies_calculation_l1276_127624


namespace NUMINAMATH_CALUDE_election_result_l1276_127608

theorem election_result (total_voters : ℝ) (rep_percent : ℝ) (dem_percent : ℝ) 
  (dem_x_vote_percent : ℝ) (x_win_margin : ℝ) :
  rep_percent / dem_percent = 3 / 2 →
  rep_percent + dem_percent = 100 →
  dem_x_vote_percent = 25 →
  x_win_margin = 16.000000000000014 →
  ∃ (rep_x_vote_percent : ℝ),
    rep_x_vote_percent * rep_percent + dem_x_vote_percent * dem_percent = 
    (100 + x_win_margin) / 2 ∧
    rep_x_vote_percent = 80 :=
by sorry

end NUMINAMATH_CALUDE_election_result_l1276_127608


namespace NUMINAMATH_CALUDE_roots_sum_of_powers_l1276_127614

theorem roots_sum_of_powers (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 3 + 1 = 0 →
  s^2 - 2*s*Real.sqrt 3 + 1 = 0 →
  r^12 + s^12 = 940802 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_powers_l1276_127614


namespace NUMINAMATH_CALUDE_match_processes_count_l1276_127668

def number_of_match_processes : ℕ := 2 * Nat.choose 13 6

theorem match_processes_count :
  number_of_match_processes = 3432 :=
by sorry

end NUMINAMATH_CALUDE_match_processes_count_l1276_127668


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1276_127695

-- Problem 1
theorem problem_1 : (-2)^2 + |-4| - 18 * (-1/3) = 14 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : 2*(3*a^2*b - 2*a*b^2) - 4*(-a*b^2 + a^2*b) = 2*a^2*b := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1276_127695


namespace NUMINAMATH_CALUDE_factorization_a4_2a3_1_l1276_127611

theorem factorization_a4_2a3_1 (a : ℝ) : 
  a^4 + 2*a^3 + 1 = (a + 1) * (a^3 + a^2 - a + 1) := by sorry

end NUMINAMATH_CALUDE_factorization_a4_2a3_1_l1276_127611


namespace NUMINAMATH_CALUDE_age_difference_l1276_127615

-- Define the ages of the siblings
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := 16

-- Theorem to prove
theorem age_difference : greg_age - marcia_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1276_127615


namespace NUMINAMATH_CALUDE_sin_thirty_degrees_l1276_127627

theorem sin_thirty_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirty_degrees_l1276_127627


namespace NUMINAMATH_CALUDE_rahim_pillows_l1276_127634

/-- The number of pillows Rahim bought initially -/
def initial_pillows : ℕ := 4

/-- The initial average cost of pillows -/
def initial_avg_cost : ℚ := 5

/-- The price of the fifth pillow -/
def fifth_pillow_price : ℚ := 10

/-- The new average price of 5 pillows -/
def new_avg_price : ℚ := 6

/-- Proof that the number of pillows Rahim bought initially is 4 -/
theorem rahim_pillows :
  (initial_avg_cost * initial_pillows + fifth_pillow_price) / (initial_pillows + 1) = new_avg_price :=
by sorry

end NUMINAMATH_CALUDE_rahim_pillows_l1276_127634


namespace NUMINAMATH_CALUDE_find_k_l1276_127665

-- Define the circles and points
def larger_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 100}
def smaller_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 25}
def P : ℝ × ℝ := (6, 8)
def S (k : ℝ) : ℝ × ℝ := (0, k)
def QR : ℝ := 5

-- Theorem statement
theorem find_k :
  P ∈ larger_circle ∧
  ∀ k, S k ∈ smaller_circle →
  QR = 5 →
  ∃ k, S k ∈ smaller_circle ∧ k = 5 := by
sorry

end NUMINAMATH_CALUDE_find_k_l1276_127665


namespace NUMINAMATH_CALUDE_inequality_proof_l1276_127643

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 3) :
  (a * b / Real.sqrt (c^2 + 3)) + 
  (b * c / Real.sqrt (a^2 + 3)) + 
  (c * a / Real.sqrt (b^2 + 3)) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1276_127643


namespace NUMINAMATH_CALUDE_no_solution_exponential_equation_l1276_127688

theorem no_solution_exponential_equation :
  ¬ ∃ y : ℝ, (16 : ℝ) ^ (3 * y) = (64 : ℝ) ^ (2 * y + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exponential_equation_l1276_127688


namespace NUMINAMATH_CALUDE_mailman_theorem_l1276_127692

def mailman_problem (total_mail : ℕ) (mail_per_block : ℕ) : ℕ :=
  total_mail / mail_per_block

theorem mailman_theorem : 
  mailman_problem 192 48 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mailman_theorem_l1276_127692


namespace NUMINAMATH_CALUDE_point_direction_form_equation_l1276_127632

/-- The point-direction form equation of a line with direction vector (2, -3) passing through the point (1, 0) -/
theorem point_direction_form_equation (x y : ℝ) : 
  let direction_vector : ℝ × ℝ := (2, -3)
  let point : ℝ × ℝ := (1, 0)
  let line_equation := (x - point.1) / direction_vector.1 = y / direction_vector.2
  line_equation = ((x - 1) / 2 = y / (-3))
  := by sorry

end NUMINAMATH_CALUDE_point_direction_form_equation_l1276_127632


namespace NUMINAMATH_CALUDE_train_arrival_time_correct_l1276_127659

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a train journey -/
structure TrainJourney where
  distance : Nat  -- in miles
  speed : Nat     -- in miles per hour
  departure : Time
  timeDifference : Int  -- time difference between departure and arrival time zones

def arrivalTime (journey : TrainJourney) : Time :=
  sorry

theorem train_arrival_time_correct (journey : TrainJourney) 
  (h1 : journey.distance = 480)
  (h2 : journey.speed = 60)
  (h3 : journey.departure = ⟨10, 0⟩)
  (h4 : journey.timeDifference = -1) :
  arrivalTime journey = ⟨17, 0⟩ :=
  sorry

end NUMINAMATH_CALUDE_train_arrival_time_correct_l1276_127659


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l1276_127645

theorem opposite_of_negative_six : 
  -((-6) : ℤ) = 6 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l1276_127645


namespace NUMINAMATH_CALUDE_train_length_l1276_127628

/-- The length of a train given its speed, bridge length, and crossing time -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 255.03 →
  crossing_time = 30 →
  train_speed * crossing_time - bridge_length = 119.97 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1276_127628


namespace NUMINAMATH_CALUDE_money_ratio_proof_l1276_127674

/-- Proves that the ratio of Nataly's money to Raquel's money is 3:1 given the problem conditions -/
theorem money_ratio_proof (tom nataly raquel : ℚ) : 
  tom = (1 / 4) * nataly →  -- Tom has 1/4 as much money as Nataly
  nataly = raquel * (nataly / raquel) →  -- Nataly has a certain multiple of Raquel's money
  tom + raquel + nataly = 190 →  -- Total money is $190
  raquel = 40 →  -- Raquel has $40
  nataly / raquel = 3 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_proof_l1276_127674


namespace NUMINAMATH_CALUDE_mrs_sheridan_fish_count_l1276_127636

/-- The number of fish Mrs. Sheridan initially had -/
def initial_fish : ℕ := 22

/-- The number of fish Mrs. Sheridan's sister gave her -/
def additional_fish : ℕ := 47

/-- The total number of fish Mrs. Sheridan has now -/
def total_fish : ℕ := initial_fish + additional_fish

theorem mrs_sheridan_fish_count : total_fish = 69 := by
  sorry

end NUMINAMATH_CALUDE_mrs_sheridan_fish_count_l1276_127636


namespace NUMINAMATH_CALUDE_reflection_problem_l1276_127607

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x - y + 7 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define point N
def N : ℝ × ℝ := (1, 0)

-- Define the intersection point M
def M : ℝ × ℝ := (-2, 1)

-- Define the symmetric point P
def P : ℝ × ℝ := (-2, -1)

-- Define line l₃
def l₃ (x y : ℝ) : Prop := y = (1/3) * x - (1/3)

-- Define the parallel lines at distance √10 from l₃
def parallel_line₁ (x y : ℝ) : Prop := y = (1/3) * x + 3
def parallel_line₂ (x y : ℝ) : Prop := y = (1/3) * x - (11/3)

theorem reflection_problem :
  (∀ x y, l₁ x y ∧ l₂ x y → (x, y) = M) ∧
  P = (-2, -1) ∧
  (∀ x y, l₃ x y ↔ y = (1/3) * x - (1/3)) ∧
  (∀ x y, (parallel_line₁ x y ∨ parallel_line₂ x y) ↔
    ∃ d, d = Real.sqrt 10 ∧ 
    (y - ((1/3) * x - (1/3)))^2 / (1 + (1/3)^2) = d^2) :=
by sorry

end NUMINAMATH_CALUDE_reflection_problem_l1276_127607


namespace NUMINAMATH_CALUDE_diagonal_intersection_theorem_l1276_127621

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem diagonal_intersection_theorem (ABCD : Quadrilateral) (E : Point) :
  isConvex ABCD →
  distance ABCD.A ABCD.B = 9 →
  distance ABCD.C ABCD.D = 12 →
  distance ABCD.A ABCD.C = 14 →
  E = lineIntersection ABCD.A ABCD.C ABCD.B ABCD.D →
  triangleArea ABCD.A E ABCD.D = triangleArea ABCD.B E ABCD.C →
  distance ABCD.A E = 6 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersection_theorem_l1276_127621


namespace NUMINAMATH_CALUDE_agricultural_product_prices_l1276_127629

/-- Given two linear equations representing the cost of agricultural products A and B,
    prove that the unique solution for the prices of A and B is (120, 150). -/
theorem agricultural_product_prices (x y : ℚ) : 
  (2 * x + 3 * y = 690) ∧ (x + 4 * y = 720) → x = 120 ∧ y = 150 := by
  sorry

end NUMINAMATH_CALUDE_agricultural_product_prices_l1276_127629


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l1276_127612

-- Define the operation ⋈
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a^2 + Real.sqrt (b + 2 * Real.sqrt (b + 3 * Real.sqrt b))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ x : ℝ, bowtie 3 x = 18 ∧ x = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l1276_127612


namespace NUMINAMATH_CALUDE_largest_valid_three_digit_number_l1276_127618

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Converts a ThreeDigitNumber to its numerical value -/
def toNumber (n : ThreeDigitNumber) : Nat :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- Calculates the sum of digits of a ThreeDigitNumber -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.1 + n.2.1 + n.2.2

/-- Checks if a ThreeDigitNumber satisfies all conditions -/
def isValid (n : ThreeDigitNumber) : Prop :=
  n.1 ≠ 0 ∧  -- Ensures it's a three-digit number
  n.1 = n.2.2 ∧  -- First digit matches third digit
  n.1 ≠ n.2.1 ∧  -- First digit doesn't match second digit
  (toNumber n) % (digitSum n) = 0  -- Number is divisible by sum of its digits

theorem largest_valid_three_digit_number :
  ∀ n : ThreeDigitNumber, isValid n → toNumber n ≤ 828 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_three_digit_number_l1276_127618


namespace NUMINAMATH_CALUDE_routes_2x2_grid_proof_l1276_127680

/-- The number of routes on a 2x2 grid from top-left to bottom-right -/
def routes_2x2_grid : ℕ := 6

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem routes_2x2_grid_proof :
  routes_2x2_grid = choose 4 2 :=
by sorry

end NUMINAMATH_CALUDE_routes_2x2_grid_proof_l1276_127680


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l1276_127654

theorem log_equality_implies_golden_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 8 = Real.log q / Real.log 18) ∧
  (Real.log q / Real.log 18 = Real.log (p + q) / Real.log 32) →
  q / p = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l1276_127654


namespace NUMINAMATH_CALUDE_either_odd_or_even_l1276_127616

theorem either_odd_or_even (n : ℤ) : (Odd (2*n - 1)) ∨ (Even (2*n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_either_odd_or_even_l1276_127616


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1276_127619

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ,
    5 - 2*x ≥ 1 →
    x + 3 > 0 →
    x + 1 ≠ 0 →
    (2 + x) * (2 - x) ≠ 0 →
    (x^2 - 4*x + 4) / (x + 1) / ((3 / (x + 1)) - x + 1) = (2 - x) / (2 + x) ∧
    (2 - 0) / (2 + 0) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1276_127619


namespace NUMINAMATH_CALUDE_distance_between_lines_l1276_127694

/-- A circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- Radius of the circle -/
  r : ℝ
  /-- Distance between adjacent parallel lines -/
  d : ℝ
  /-- Length of the first chord -/
  chord1 : ℝ
  /-- Length of the second chord -/
  chord2 : ℝ
  /-- Length of the third chord -/
  chord3 : ℝ
  /-- The first and third chords are equal -/
  chord1_eq_chord3 : chord1 = chord3
  /-- The first chord has length 42 -/
  chord1_eq_42 : chord1 = 42
  /-- The second chord has length 36 -/
  chord2_eq_36 : chord2 = 36

/-- The distance between adjacent parallel lines is 7.65 -/
theorem distance_between_lines (c : CircleWithParallelLines) : c.d = 7.65 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_lines_l1276_127694


namespace NUMINAMATH_CALUDE_custom_operation_results_l1276_127698

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := a^2 - (a + b) + a*b

-- State the theorem
theorem custom_operation_results :
  (customOp 2 (-3) = -1) ∧ (customOp 4 (customOp 2 (-3)) = 7) := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_results_l1276_127698


namespace NUMINAMATH_CALUDE_square_root_sum_equals_absolute_value_sum_l1276_127672

theorem square_root_sum_equals_absolute_value_sum (x : ℝ) :
  Real.sqrt (x^2 + 4*x + 4) + Real.sqrt (x^2 - 6*x + 9) = |x + 2| + |x - 3| := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_absolute_value_sum_l1276_127672


namespace NUMINAMATH_CALUDE_number_composition_l1276_127670

/-- A number composed of hundreds, tens, ones, and hundredths -/
def compose_number (hundreds tens ones hundredths : ℕ) : ℚ :=
  (hundreds * 100 + tens * 10 + ones : ℚ) + (hundredths : ℚ) / 100

theorem number_composition :
  compose_number 3 4 6 8 = 346.08 := by
  sorry

end NUMINAMATH_CALUDE_number_composition_l1276_127670


namespace NUMINAMATH_CALUDE_green_minus_blue_disks_l1276_127697

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green
  | Red

/-- Represents the ratio of disks of each color -/
def colorRatio : Fin 4 → Nat
  | 0 => 3  -- Blue
  | 1 => 7  -- Yellow
  | 2 => 8  -- Green
  | 3 => 4  -- Red

/-- The total number of disks in the bag -/
def totalDisks : Nat := 176

/-- Calculates the number of disks of a given color based on the ratio and total disks -/
def disksOfColor (color : Fin 4) : Nat :=
  (colorRatio color * totalDisks) / (colorRatio 0 + colorRatio 1 + colorRatio 2 + colorRatio 3)

/-- Theorem: There are 40 more green disks than blue disks in the bag -/
theorem green_minus_blue_disks : disksOfColor 2 - disksOfColor 0 = 40 := by
  sorry

end NUMINAMATH_CALUDE_green_minus_blue_disks_l1276_127697


namespace NUMINAMATH_CALUDE_wilson_sledding_l1276_127690

/-- The number of times Wilson sleds down each tall hill -/
def T : ℕ := sorry

/-- The number of times Wilson sleds down each small hill -/
def S : ℕ := sorry

/-- There are 2 tall hills and 3 small hills -/
axiom hill_counts : 2 * T + 3 * S = 14

/-- The number of times he sleds down each small hill is half the number of times he sleds down each tall hill -/
axiom small_hill_frequency : S = T / 2

theorem wilson_sledding :
  T = 4 := by sorry

end NUMINAMATH_CALUDE_wilson_sledding_l1276_127690


namespace NUMINAMATH_CALUDE_hash_composition_l1276_127604

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.75 * N + 2

-- State the theorem
theorem hash_composition : hash (hash (hash 72)) = 35 := by sorry

end NUMINAMATH_CALUDE_hash_composition_l1276_127604


namespace NUMINAMATH_CALUDE_lantern_probability_l1276_127642

def total_large_lanterns : ℕ := 360
def total_small_lanterns : ℕ := 1200

def large_with_two_small (x : ℕ) : Prop := 
  x * 2 + (total_large_lanterns - x) * 4 = total_small_lanterns

def large_with_four_small (x : ℕ) : ℕ := total_large_lanterns - x

def total_combinations : ℕ := total_large_lanterns.choose 2

def favorable_outcomes (x : ℕ) : ℕ := 
  (large_with_four_small x).choose 2 + (large_with_four_small x).choose 1 * x.choose 1

theorem lantern_probability (x : ℕ) (h : large_with_two_small x) : 
  (favorable_outcomes x : ℚ) / total_combinations = 958 / 1077 := by sorry

end NUMINAMATH_CALUDE_lantern_probability_l1276_127642


namespace NUMINAMATH_CALUDE_shoe_discount_percentage_l1276_127675

def original_price : ℝ := 62.50 + 3.75
def amount_saved : ℝ := 3.75
def amount_spent : ℝ := 62.50

theorem shoe_discount_percentage : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
  |((amount_saved / original_price) * 100 - 6)| < ε :=
sorry

end NUMINAMATH_CALUDE_shoe_discount_percentage_l1276_127675


namespace NUMINAMATH_CALUDE_set_equality_implies_x_zero_l1276_127637

theorem set_equality_implies_x_zero (x : ℝ) : 
  ({1, x^2} : Set ℝ) = ({1, x} : Set ℝ) → x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_x_zero_l1276_127637


namespace NUMINAMATH_CALUDE_max_value_constraint_l1276_127683

theorem max_value_constraint (w x y z : ℝ) (h : 9*w^2 + 4*x^2 + y^2 + 25*z^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 201 ∧ 
  (∀ w' x' y' z' : ℝ, 9*w'^2 + 4*x'^2 + y'^2 + 25*z'^2 = 1 → 
    9*w' + 4*x' + 2*y' + 10*z' ≤ max) ∧
  (∃ w'' x'' y'' z'' : ℝ, 9*w''^2 + 4*x''^2 + y''^2 + 25*z''^2 = 1 ∧
    9*w'' + 4*x'' + 2*y'' + 10*z'' = max) := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1276_127683


namespace NUMINAMATH_CALUDE_local_extrema_sum_l1276_127630

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 27

-- State the theorem
theorem local_extrema_sum (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b (-1) ≥ f a b x) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3 - ε) (3 + ε), f a b 3 ≤ f a b x) →
  a + b = -12 := by
  sorry

end NUMINAMATH_CALUDE_local_extrema_sum_l1276_127630


namespace NUMINAMATH_CALUDE_max_necklaces_is_five_l1276_127640

/-- Represents the number of beads of each color required for a single necklace -/
structure NecklacePattern where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Represents the total number of beads available for each color -/
structure AvailableBeads where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Calculates the maximum number of complete necklaces that can be made -/
def maxNecklaces (pattern : NecklacePattern) (available : AvailableBeads) : ℕ :=
  min (available.green / pattern.green)
      (min (available.white / pattern.white)
           (available.orange / pattern.orange))

/-- Theorem stating that given the specific bead counts, the maximum number of necklaces is 5 -/
theorem max_necklaces_is_five :
  let pattern := NecklacePattern.mk 9 6 3
  let available := AvailableBeads.mk 45 45 45
  maxNecklaces pattern available = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_necklaces_is_five_l1276_127640


namespace NUMINAMATH_CALUDE_actual_length_is_320_l1276_127676

/-- Blueprint scale factor -/
def scale_factor : ℝ := 20

/-- Measured length on the blueprint in cm -/
def measured_length : ℝ := 16

/-- Actual length of the part in cm -/
def actual_length : ℝ := measured_length * scale_factor

/-- Theorem stating that the actual length is 320cm -/
theorem actual_length_is_320 : actual_length = 320 := by
  sorry

end NUMINAMATH_CALUDE_actual_length_is_320_l1276_127676


namespace NUMINAMATH_CALUDE_tangent_line_circle_m_value_l1276_127699

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  equation : ℝ → ℝ → ℝ → Prop

/-- Predicate to check if a line is tangent to a circle -/
def IsTangent (l : Line) (c : Circle) : Prop := sorry

/-- The main theorem -/
theorem tangent_line_circle_m_value (m : ℝ) :
  let c : Circle := ⟨λ x y m => x^2 + y^2 = m⟩
  let l : Line := ⟨λ x y m => x + y + m = 0⟩
  IsTangent l c → m = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_m_value_l1276_127699


namespace NUMINAMATH_CALUDE_fourth_player_wins_probability_l1276_127658

def roll_probability : ℚ := 1 / 6

def other_roll_probability : ℚ := 1 - roll_probability

def num_players : ℕ := 4

def first_cycle_probability : ℚ := (other_roll_probability ^ (num_players - 1)) * roll_probability

def cycle_continuation_probability : ℚ := other_roll_probability ^ num_players

theorem fourth_player_wins_probability :
  let a := first_cycle_probability
  let r := cycle_continuation_probability
  (a / (1 - r)) = 125 / 671 := by sorry

end NUMINAMATH_CALUDE_fourth_player_wins_probability_l1276_127658


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1276_127605

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1276_127605


namespace NUMINAMATH_CALUDE_painter_problem_l1276_127691

/-- Given a painting job with a total number of rooms, rooms already painted, and time per room,
    calculate the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (painted_rooms : ℕ) (time_per_room : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

theorem painter_problem :
  let total_rooms : ℕ := 9
  let painted_rooms : ℕ := 5
  let time_per_room : ℕ := 8
  time_to_paint_remaining total_rooms painted_rooms time_per_room = 32 := by
  sorry

end NUMINAMATH_CALUDE_painter_problem_l1276_127691


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l1276_127651

theorem unique_solution_to_equation : 
  ∃! (x y : ℕ+), (x.val : ℝ)^4 * (y.val : ℝ)^4 - 16 * (x.val : ℝ)^2 * (y.val : ℝ)^2 + 15 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l1276_127651


namespace NUMINAMATH_CALUDE_seashells_count_l1276_127609

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The total number of seashells found by Tom and Fred -/
def total_seashells : ℕ := tom_seashells + fred_seashells

theorem seashells_count : total_seashells = 58 := by
  sorry

end NUMINAMATH_CALUDE_seashells_count_l1276_127609


namespace NUMINAMATH_CALUDE_function_properties_l1276_127663

/-- Given that y+6 is directly proportional to x+1 and when x=3, y=2 -/
def proportional_function (x y : ℝ) : Prop :=
  ∃ (k : ℝ), y + 6 = k * (x + 1) ∧ 2 + 6 = k * (3 + 1)

theorem function_properties :
  ∀ x y m : ℝ,
  proportional_function x y →
  (y = 2*x - 4 ∧
   (proportional_function m (-2) → m = 1) ∧
   ¬proportional_function 1 (-3)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1276_127663


namespace NUMINAMATH_CALUDE_min_value_theorem_l1276_127633

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + 3 * b = 5 * a * b → 3 * a + 4 * b ≥ 3 * x + 4 * y) →
  3 * x + 4 * y = 5 ∧ x + 4 * y = 3 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1276_127633


namespace NUMINAMATH_CALUDE_min_value_f_zeros_of_g_inequality_holds_l1276_127666

/-- The minimum value of f(x) = 1/2x²·a·ln(x) is -a/(4e) for a > 0 and x > 0 -/
theorem min_value_f (a : ℝ) (ha : a > 0) :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → (1/2 * y^2 * a * Real.log y) ≥ -a / (4 * Real.exp 1) :=
sorry

/-- The function g(x) = 1/2x² - a·ln(x) + (a - 1)x has two zeros in (1/e, e) 
    if and only if (2e - 1)/(2e² + 2e) < a < 1/2 -/
theorem zeros_of_g (a : ℝ) :
  (∃ (x y : ℝ), 1/Real.exp 1 < x ∧ x < y ∧ y < Real.exp 1 ∧
    1/2 * x^2 - a * Real.log x + (a - 1) * x = 0 ∧
    1/2 * y^2 - a * Real.log y + (a - 1) * y = 0) ↔
  ((2 * Real.exp 1 - 1) / (2 * Real.exp 1^2 + 2 * Real.exp 1) < a ∧ a < 1/2) :=
sorry

/-- For all x > 0, ln(x) + 3/(4x²) - 1/eˣ > 0 -/
theorem inequality_holds (x : ℝ) (hx : x > 0) :
  Real.log x + 3 / (4 * x^2) - 1 / Real.exp x > 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_zeros_of_g_inequality_holds_l1276_127666


namespace NUMINAMATH_CALUDE_equation_solution_l1276_127685

theorem equation_solution : ∃ x : ℝ, x + 1 - 2 * (x - 1) = 1 - 3 * x ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1276_127685


namespace NUMINAMATH_CALUDE_two_lines_intersecting_at_distance_5_l1276_127613

/-- Given a line and a point, find two lines passing through the point and intersecting the given line at a distance of 5 from the given point. -/
theorem two_lines_intersecting_at_distance_5 :
  ∃ (l₂₁ l₂₂ : ℝ → ℝ → Prop),
    (∀ x y, l₂₁ x y ↔ x = 1) ∧
    (∀ x y, l₂₂ x y ↔ 3 * x + 4 * y + 1 = 0) ∧
    (∀ x y, l₂₁ x y → l₂₁ 1 (-1)) ∧
    (∀ x y, l₂₂ x y → l₂₂ 1 (-1)) ∧
    (∃ x₁ y₁, l₂₁ x₁ y₁ ∧ 2 * x₁ + y₁ - 6 = 0 ∧ (x₁ - 1)^2 + (y₁ + 1)^2 = 5^2) ∧
    (∃ x₂ y₂, l₂₂ x₂ y₂ ∧ 2 * x₂ + y₂ - 6 = 0 ∧ (x₂ - 1)^2 + (y₂ + 1)^2 = 5^2) :=
by sorry


end NUMINAMATH_CALUDE_two_lines_intersecting_at_distance_5_l1276_127613


namespace NUMINAMATH_CALUDE_tom_fruit_purchase_l1276_127696

theorem tom_fruit_purchase (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) :
  apple_quantity = 8 →
  apple_rate = 70 →
  mango_quantity = 9 →
  mango_rate = 75 →
  apple_quantity * apple_rate + mango_quantity * mango_rate = 1235 := by
  sorry

end NUMINAMATH_CALUDE_tom_fruit_purchase_l1276_127696


namespace NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l1276_127625

/-- Given two planes α and β with normal vectors (x, 1, -2) and (-1, y, 1/2) respectively,
    if α is parallel to β, then x + y = 15/4 -/
theorem parallel_planes_normal_vectors (x y : ℝ) :
  let n1 : ℝ × ℝ × ℝ := (x, 1, -2)
  let n2 : ℝ × ℝ × ℝ := (-1, y, 1/2)
  (∃ (k : ℝ), n1 = k • n2) →
  x + y = 15/4 := by
sorry

end NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l1276_127625


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_product_l1276_127669

-- Part 1
theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  1 / x + 1 / y ≥ 3 + 2 * Real.sqrt 2 := by sorry

-- Part 2
theorem min_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x * y ≥ 32 := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_product_l1276_127669


namespace NUMINAMATH_CALUDE_termite_ridden_homes_l1276_127686

theorem termite_ridden_homes (total_homes : ℝ) (termite_ridden_homes : ℝ) 
  (h1 : termite_ridden_homes > 0)
  (h2 : termite_ridden_homes ≤ total_homes)
  (h3 : (7 / 10) * termite_ridden_homes + 0.1 * total_homes = termite_ridden_homes) : 
  termite_ridden_homes / total_homes = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_termite_ridden_homes_l1276_127686


namespace NUMINAMATH_CALUDE_pizza_slices_left_is_three_l1276_127646

/-- Calculates the number of pizza slices left after John and Sam eat -/
def pizza_slices_left (total : ℕ) (john_ate : ℕ) (sam_ate_multiplier : ℕ) : ℕ :=
  total - (john_ate + sam_ate_multiplier * john_ate)

/-- Theorem: The number of pizza slices left is 3 -/
theorem pizza_slices_left_is_three :
  pizza_slices_left 12 3 2 = 3 := by
  sorry

#eval pizza_slices_left 12 3 2

end NUMINAMATH_CALUDE_pizza_slices_left_is_three_l1276_127646


namespace NUMINAMATH_CALUDE_sin_negative_270_degrees_l1276_127655

theorem sin_negative_270_degrees : Real.sin ((-270 : ℝ) * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_270_degrees_l1276_127655


namespace NUMINAMATH_CALUDE_saree_final_price_l1276_127644

/-- Calculates the final sale price of a saree after discounts, tax, and custom fee. -/
def saree_sale_price (original_price : ℝ) (discount1 discount2 discount3 tax : ℝ) (custom_fee : ℝ) : ℝ :=
  let price_after_discounts := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  let price_after_tax := price_after_discounts * (1 + tax)
  price_after_tax + custom_fee

/-- Theorem stating that the final sale price of the saree is 773.2 -/
theorem saree_final_price :
  saree_sale_price 1200 0.25 0.20 0.15 0.10 100 = 773.2 := by
  sorry

#eval saree_sale_price 1200 0.25 0.20 0.15 0.10 100

end NUMINAMATH_CALUDE_saree_final_price_l1276_127644


namespace NUMINAMATH_CALUDE_smallest_product_l1276_127647

def digits : List Nat := [4, 5, 6, 7]

def valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat, valid_arrangement a b c d →
    product a b c d ≥ 2622 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l1276_127647


namespace NUMINAMATH_CALUDE_sum_of_integers_l1276_127662

theorem sum_of_integers (x y : ℕ+) (h1 : x.val - y.val = 8) (h2 : x.val * y.val = 120) : 
  x.val + y.val = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1276_127662


namespace NUMINAMATH_CALUDE_min_abs_z_plus_two_l1276_127641

open Complex

theorem min_abs_z_plus_two (z : ℂ) (h : (z * (1 + I)).im = 0) :
  ∃ (min : ℝ), min = Real.sqrt 2 ∧ ∀ (w : ℂ), (w * (1 + I)).im = 0 → Complex.abs (w + 2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_plus_two_l1276_127641


namespace NUMINAMATH_CALUDE_focal_length_of_hyperbola_C_l1276_127681

-- Define the hyperbola C
def hyperbola_C (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 = 1

-- Define the asymptote of C
def asymptote_C (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 3 * x + m * y = 0

-- State the theorem
theorem focal_length_of_hyperbola_C (m : ℝ) (h1 : m > 0) :
  (∀ x y : ℝ, hyperbola_C m x y ↔ asymptote_C m x y) →
  2 * Real.sqrt (m + m) = 4 := by sorry

end NUMINAMATH_CALUDE_focal_length_of_hyperbola_C_l1276_127681


namespace NUMINAMATH_CALUDE_ln_cube_relation_l1276_127600

theorem ln_cube_relation (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x > Real.log y → x^3 > y^3) ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^3 > b^3 ∧ ¬(Real.log a > Real.log b) :=
sorry

end NUMINAMATH_CALUDE_ln_cube_relation_l1276_127600


namespace NUMINAMATH_CALUDE_rectangle_area_equality_l1276_127653

theorem rectangle_area_equality (x y : ℝ) : 
  x * y = (x + 4) * (y - 3) ∧ 
  x * y = (x + 8) * (y - 4) → 
  x + y = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_equality_l1276_127653


namespace NUMINAMATH_CALUDE_f_recursive_relation_l1276_127678

def f (n : ℕ) : ℕ := (Finset.range (2 * n + 1)).sum (λ i => i ^ 2)

theorem f_recursive_relation (k : ℕ) : f (k + 1) = f k + (2 * k + 1) ^ 2 + (2 * k + 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_f_recursive_relation_l1276_127678


namespace NUMINAMATH_CALUDE_caitlins_number_l1276_127626

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem caitlins_number (a b c : ℕ) 
  (h1 : is_two_digit_prime a)
  (h2 : is_two_digit_prime b)
  (h3 : is_two_digit_prime c)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h5 : 1 ≤ a + b ∧ a + b ≤ 31)
  (h6 : a + c < a + b)
  (h7 : b + c > a + b) :
  c = 11 := by
sorry

end NUMINAMATH_CALUDE_caitlins_number_l1276_127626


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_nonzero_l1276_127684

theorem x_positive_sufficient_not_necessary_for_x_nonzero :
  (∃ x : ℝ, x > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_nonzero_l1276_127684


namespace NUMINAMATH_CALUDE_average_marks_of_combined_classes_l1276_127622

theorem average_marks_of_combined_classes 
  (class1_size : ℕ) (class1_avg : ℝ) 
  (class2_size : ℕ) (class2_avg : ℝ) : 
  class1_size = 30 → 
  class1_avg = 40 → 
  class2_size = 50 → 
  class2_avg = 60 → 
  let total_students := class1_size + class2_size
  let total_marks := class1_size * class1_avg + class2_size * class2_avg
  (total_marks / total_students : ℝ) = 52.5 := by
sorry

end NUMINAMATH_CALUDE_average_marks_of_combined_classes_l1276_127622


namespace NUMINAMATH_CALUDE_partial_square_division_l1276_127687

/-- Represents a square with a side length and a removed portion. -/
structure PartialSquare where
  side_length : ℝ
  removed_fraction : ℝ

/-- Represents a division of the remaining area into parts. -/
structure AreaDivision where
  num_parts : ℕ
  area_per_part : ℝ

/-- Theorem stating that a square with side length 4 and one fourth removed
    can be divided into four equal parts with area 3 each. -/
theorem partial_square_division (s : PartialSquare)
  (h1 : s.side_length = 4)
  (h2 : s.removed_fraction = 1/4) :
  ∃ (d : AreaDivision), 
    d.num_parts = 4 ∧ 
    d.area_per_part = 3 ∧
    d.num_parts * d.area_per_part = s.side_length^2 - s.side_length^2 * s.removed_fraction :=
by sorry

end NUMINAMATH_CALUDE_partial_square_division_l1276_127687


namespace NUMINAMATH_CALUDE_linear_function_x_axis_intersection_l1276_127652

/-- A linear function passing through (-1, 2) with y-intercept 4 -/
def f (x : ℝ) : ℝ := 2 * x + 4

theorem linear_function_x_axis_intersection :
  ∃ (x : ℝ), f x = 0 ∧ x = -2 := by
  sorry

#check linear_function_x_axis_intersection

end NUMINAMATH_CALUDE_linear_function_x_axis_intersection_l1276_127652


namespace NUMINAMATH_CALUDE_subtract_131_6_minus_35_6_l1276_127667

/-- Converts a number from base 6 to base 10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a number from base 10 to base 6 --/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec loop (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc else loop (n / 6) ((n % 6) :: acc)
  loop n []

/-- Subtracts two numbers in base 6 --/
def subtractBase6 (a b : List Nat) : List Nat :=
  toBase6 (toBase10 a - toBase10 b)

theorem subtract_131_6_minus_35_6 :
  subtractBase6 [1, 3, 1] [3, 5] = [5, 2] := by sorry

end NUMINAMATH_CALUDE_subtract_131_6_minus_35_6_l1276_127667


namespace NUMINAMATH_CALUDE_product_of_five_terms_l1276_127631

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_five_terms
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a3 : a 3 = -1) :
  a 1 * a 2 * a 3 * a 4 * a 5 = -1 :=
by sorry

end NUMINAMATH_CALUDE_product_of_five_terms_l1276_127631


namespace NUMINAMATH_CALUDE_lcm_problem_l1276_127602

theorem lcm_problem (a b c : ℕ+) (h1 : b = 30) (h2 : c = 40) (h3 : Nat.lcm (Nat.lcm a.val b.val) c.val = 120) : a = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l1276_127602


namespace NUMINAMATH_CALUDE_nancy_books_l1276_127620

theorem nancy_books (alyssa_books : ℕ) (nancy_multiplier : ℕ) : 
  alyssa_books = 36 → nancy_multiplier = 7 → alyssa_books * nancy_multiplier = 252 := by
  sorry

end NUMINAMATH_CALUDE_nancy_books_l1276_127620


namespace NUMINAMATH_CALUDE_average_price_23_story_building_l1276_127660

/-- The average price per square meter of a 23-story building with specific pricing conditions. -/
theorem average_price_23_story_building (a₁ a₂ a : ℝ) : 
  let floor_prices : List ℝ := 
    [a₁] ++ (List.range 21).map (λ i => a + i * (a / 100)) ++ [a₂]
  (floor_prices.sum / 23 : ℝ) = (a₁ + a₂ + 23.1 * a) / 23 := by
  sorry

end NUMINAMATH_CALUDE_average_price_23_story_building_l1276_127660


namespace NUMINAMATH_CALUDE_sum_of_powers_l1276_127601

theorem sum_of_powers (x : ℝ) (h1 : x^2023 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^2022 + x^2021 + x^2020 + x^2019 + x^2018 + x^2017 + x^2016 + x^2015 + x^2014 + x^2013 +
  x^2012 + x^2011 + x^2010 + x^2009 + x^2008 + x^2007 + x^2006 + x^2005 + x^2004 + x^2003 +
  x^2002 + x^2001 + x^2000 + x^1999 + x^1998 + x^1997 + x^1996 + x^1995 + x^1994 + x^1993 +
  x^1992 + x^1991 + x^1990 + x^1989 + x^1988 + x^1987 + x^1986 + x^1985 + x^1984 + x^1983 +
  x^1982 + x^1981 + x^1980 + x^1979 + x^1978 + x^1977 + x^1976 + x^1975 + x^1974 + x^1973 +
  -- ... (omitting middle terms for brevity)
  x^50 + x^49 + x^48 + x^47 + x^46 + x^45 + x^44 + x^43 + x^42 + x^41 +
  x^40 + x^39 + x^38 + x^37 + x^36 + x^35 + x^34 + x^33 + x^32 + x^31 +
  x^30 + x^29 + x^28 + x^27 + x^26 + x^25 + x^24 + x^23 + x^22 + x^21 +
  x^20 + x^19 + x^18 + x^17 + x^16 + x^15 + x^14 + x^13 + x^12 + x^11 +
  x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1276_127601


namespace NUMINAMATH_CALUDE_symmetry_of_M_and_N_l1276_127671

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the z-axis -/
def symmetricAboutZAxis (p q : Point3D) : Prop :=
  p.x = -q.x ∧ p.y = -q.y ∧ p.z = q.z

theorem symmetry_of_M_and_N :
  let M : Point3D := ⟨1, -2, 3⟩
  let N : Point3D := ⟨-1, 2, 3⟩
  symmetricAboutZAxis M N := by sorry

end NUMINAMATH_CALUDE_symmetry_of_M_and_N_l1276_127671


namespace NUMINAMATH_CALUDE_linear_system_solution_l1276_127606

theorem linear_system_solution (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (eq1 : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ = 0)
  (eq2 : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ = 10)
  (eq3 : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ = 100) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ = 210 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1276_127606


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1276_127649

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1276_127649


namespace NUMINAMATH_CALUDE_one_slice_left_l1276_127682

/-- Represents the number of slices in a whole pizza after cutting -/
def total_slices : ℕ := 8

/-- Represents the number of friends who receive 1 slice each -/
def friends_one_slice : ℕ := 3

/-- Represents the number of friends who receive 2 slices each -/
def friends_two_slices : ℕ := 2

/-- Represents the number of slices given to friends -/
def slices_given : ℕ := friends_one_slice * 1 + friends_two_slices * 2

/-- Theorem stating that there is 1 slice left after distribution -/
theorem one_slice_left : total_slices - slices_given = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_slice_left_l1276_127682


namespace NUMINAMATH_CALUDE_product_of_sum_of_squares_l1276_127610

theorem product_of_sum_of_squares (a b c d : ℤ) : ∃ x y : ℤ, (a^2 + b^2) * (c^2 + d^2) = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_squares_l1276_127610


namespace NUMINAMATH_CALUDE_sams_age_l1276_127689

theorem sams_age (sam drew alex jordan : ℕ) : 
  sam + drew + alex + jordan = 142 →
  sam = drew / 2 →
  alex = sam + 3 →
  jordan = 2 * alex →
  sam = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_sams_age_l1276_127689


namespace NUMINAMATH_CALUDE_circle_point_marking_l1276_127635

/-- The number of points on the circle -/
def n : ℕ := 2021

/-- 
Given n points on a circle, prove that the smallest positive integer b 
such that b(b+1)/2 is divisible by n is 67.
-/
theorem circle_point_marking (b : ℕ) : 
  (∀ k < b, ¬(2 ∣ k * (k + 1) ∧ n ∣ k * (k + 1))) ∧ 
  (2 ∣ b * (b + 1) ∧ n ∣ b * (b + 1)) → 
  b = 67 := by sorry

end NUMINAMATH_CALUDE_circle_point_marking_l1276_127635


namespace NUMINAMATH_CALUDE_fruit_difference_l1276_127673

theorem fruit_difference (total : ℕ) (apples : ℕ) : 
  total = 913 → apples = 514 → apples - (total - apples) = 115 := by
  sorry

end NUMINAMATH_CALUDE_fruit_difference_l1276_127673
