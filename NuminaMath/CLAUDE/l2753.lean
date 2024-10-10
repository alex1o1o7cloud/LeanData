import Mathlib

namespace bisection_interval_valid_l2753_275341

-- Define the function f(x) = x^3 + 5
def f (x : ℝ) : ℝ := x^3 + 5

-- Theorem statement
theorem bisection_interval_valid :
  f (-2) * f 1 < 0 := by sorry

end bisection_interval_valid_l2753_275341


namespace monotonic_unique_zero_l2753_275320

/-- A function f is monotonic on (a, b) -/
def Monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ f y < f x)

/-- f has exactly one zero in [a, b] -/
def HasUniqueZero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0

theorem monotonic_unique_zero (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : Monotonic f a b) (h2 : f a * f b < 0) :
  HasUniqueZero f a b :=
sorry

end monotonic_unique_zero_l2753_275320


namespace root_equations_l2753_275374

/-- Given two constants c and d, prove that they satisfy the given conditions -/
theorem root_equations (c d : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (x + c) * (x + d) * (x + 8) = 0 ∧
    (y + c) * (y + d) * (y + 8) = 0 ∧
    (z + c) * (z + d) * (z + 8) = 0 ∧
    (x + 2) ≠ 0 ∧ (y + 2) ≠ 0 ∧ (z + 2) ≠ 0) ∧
  (∃! w : ℝ, (w + 3*c) * (w + 2) * (w + 4) = 0 ∧
    (w + d) ≠ 0 ∧ (w + 8) ≠ 0) →
  c = 2/3 ∧ d = 4 := by
sorry

end root_equations_l2753_275374


namespace arithmetic_sequence_13th_term_l2753_275331

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_13th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_5th : a 5 = 3)
  (h_9th : a 9 = 6) :
  a 13 = 9 := by
sorry

end arithmetic_sequence_13th_term_l2753_275331


namespace cubic_root_sum_l2753_275399

theorem cubic_root_sum (c d : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + c * (Complex.I * Real.sqrt 2 + 2) + d = 0 →
  c + d = 14 := by
sorry

end cubic_root_sum_l2753_275399


namespace remainder_of_y_l2753_275350

theorem remainder_of_y (y : ℤ) 
  (h1 : (4 + y) % 8 = 3^2 % 8)
  (h2 : (6 + y) % 27 = 2^3 % 27)
  (h3 : (8 + y) % 125 = 3^3 % 125) :
  y % 30 = 4 := by
sorry

end remainder_of_y_l2753_275350


namespace total_workers_l2753_275376

theorem total_workers (monkeys termites : ℕ) 
  (h1 : monkeys = 239) 
  (h2 : termites = 622) : 
  monkeys + termites = 861 := by
  sorry

end total_workers_l2753_275376


namespace trigonometric_identity_l2753_275347

theorem trigonometric_identity (x y : ℝ) :
  Real.sin (x - y + π/6) * Real.cos (y + π/6) + Real.cos (x - y + π/6) * Real.sin (y + π/6) = Real.sin (x + π/3) := by
  sorry

end trigonometric_identity_l2753_275347


namespace three_number_problem_l2753_275317

theorem three_number_problem (x y z : ℚ) : 
  x + (1/3) * z = y ∧ 
  y + (1/3) * x = z ∧ 
  z - x = 10 → 
  x = 10 ∧ y = 50/3 ∧ z = 20 := by
sorry

end three_number_problem_l2753_275317


namespace even_function_implies_f_2_equals_3_l2753_275371

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_f_2_equals_3 (a : ℝ) 
  (h : ∀ x, f a x = f a (-x)) : 
  f a 2 = 3 := by
sorry

end even_function_implies_f_2_equals_3_l2753_275371


namespace brianna_marbles_l2753_275365

/-- The number of marbles Brianna has remaining after a series of events -/
def remaining_marbles (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost - (2 * lost) - (lost / 2)

/-- Theorem stating that Brianna has 10 marbles remaining -/
theorem brianna_marbles : remaining_marbles 24 4 = 10 := by
  sorry

end brianna_marbles_l2753_275365


namespace parallel_lines_k_value_l2753_275349

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of the first line y = 5x + 3 -/
def slope1 : ℝ := 5

/-- The slope of the second line y = (3k)x + 7 -/
def slope2 (k : ℝ) : ℝ := 3 * k

/-- Theorem: If the lines y = 5x + 3 and y = (3k)x + 7 are parallel, then k = 5/3 -/
theorem parallel_lines_k_value (k : ℝ) :
  parallel slope1 (slope2 k) → k = 5/3 := by
  sorry

end parallel_lines_k_value_l2753_275349


namespace eggs_in_box_l2753_275378

/-- Given an initial count of eggs and a number of whole eggs added, 
    calculate the total number of whole eggs, ignoring fractional parts. -/
def total_whole_eggs (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that with 7 initial eggs and 3 added whole eggs, 
    the total number of whole eggs is 10. -/
theorem eggs_in_box : total_whole_eggs 7 3 = 10 := by
  sorry

end eggs_in_box_l2753_275378


namespace second_class_end_time_l2753_275364

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60
  , minutes := totalMinutes % 60 }

theorem second_class_end_time :
  let start_time : Time := { hours := 9, minutes := 25 }
  let class_duration : Nat := 35
  let end_time := addMinutes start_time class_duration
  end_time = { hours := 10, minutes := 0 } := by
  sorry

end second_class_end_time_l2753_275364


namespace exists_farther_point_l2753_275328

/-- A rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- A point on the surface of the box -/
inductive SurfacePoint (b : Box)
  | front (x y : ℝ) : 0 ≤ x ∧ x ≤ b.width → 0 ≤ y ∧ y ≤ b.height → SurfacePoint b
  | back (x y : ℝ) : 0 ≤ x ∧ x ≤ b.width → 0 ≤ y ∧ y ≤ b.height → SurfacePoint b
  | left (y z : ℝ) : 0 ≤ y ∧ y ≤ b.height → 0 ≤ z ∧ z ≤ b.length → SurfacePoint b
  | right (y z : ℝ) : 0 ≤ y ∧ y ≤ b.height → 0 ≤ z ∧ z ≤ b.length → SurfacePoint b
  | top (x z : ℝ) : 0 ≤ x ∧ x ≤ b.width → 0 ≤ z ∧ z ≤ b.length → SurfacePoint b
  | bottom (x z : ℝ) : 0 ≤ x ∧ x ≤ b.width → 0 ≤ z ∧ z ≤ b.length → SurfacePoint b

/-- The distance between two points on the surface of the box -/
def surfaceDistance (b : Box) (p q : SurfacePoint b) : ℝ := sorry

/-- The opposite corner of a given corner -/
def oppositeCorner (b : Box) (p : SurfacePoint b) : SurfacePoint b := sorry

/-- Theorem: There exists a point on the surface farther from a corner than the opposite corner -/
theorem exists_farther_point (b : Box) :
  ∃ (corner : SurfacePoint b) (p : SurfacePoint b),
    surfaceDistance b corner p > surfaceDistance b corner (oppositeCorner b corner) := by sorry

end exists_farther_point_l2753_275328


namespace function_properties_l2753_275392

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x) - 2 * Real.sqrt 3 * (Real.sin (ω * x))^2 + Real.sqrt 3

def is_symmetry_axis (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem function_properties (ω : ℝ) (h_ω : ω > 0) 
  (h_symmetry : ∃ x₁ x₂, is_symmetry_axis (f ω) x₁ ∧ is_symmetry_axis (f ω) x₂)
  (h_min_dist : ∃ x₁ x₂, is_symmetry_axis (f ω) x₁ ∧ is_symmetry_axis (f ω) x₂ ∧ |x₁ - x₂| ≥ π/2 ∧ 
    ∀ y₁ y₂, is_symmetry_axis (f ω) y₁ ∧ is_symmetry_axis (f ω) y₂ → |y₁ - y₂| ≥ |x₁ - x₂|) :
  (ω = 1) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (-5*π/12 + k*π) (π/12 + k*π), 
    ∀ y ∈ Set.Icc (-5*π/12 + k*π) (π/12 + k*π), x < y → (f ω x) < (f ω y)) ∧
  (∀ α : ℝ, f ω α = 2/3 → Real.sin (5*π/6 - 4*α) = -7/9) :=
sorry

end function_properties_l2753_275392


namespace min_value_expression_l2753_275359

theorem min_value_expression (x : ℝ) (hx : x > 0) : 3 * x + 2 / x^5 + 3 / x ≥ 8 ∧
  (3 * x + 2 / x^5 + 3 / x = 8 ↔ x = 1) := by
  sorry

end min_value_expression_l2753_275359


namespace max_ones_on_board_l2753_275368

/-- The operation that replaces two numbers with their GCD and LCM -/
def replace_with_gcd_lcm (a b : ℕ) : List ℕ :=
  [Nat.gcd a b, Nat.lcm a b]

/-- The set of numbers on the board -/
def board : Set ℕ := Finset.range 2014

/-- A sequence of operations on the board -/
def operation_sequence := List (ℕ × ℕ)

/-- Apply a sequence of operations to the board -/
def apply_operations (ops : operation_sequence) (b : Set ℕ) : Set ℕ :=
  sorry

/-- Count the number of 1's in a set of natural numbers -/
def count_ones (s : Set ℕ) : ℕ :=
  sorry

/-- The theorem stating the maximum number of 1's obtainable -/
theorem max_ones_on_board :
  ∃ (ops : operation_sequence),
    ∀ (ops' : operation_sequence),
      count_ones (apply_operations ops board) ≥ count_ones (apply_operations ops' board) ∧
      count_ones (apply_operations ops board) = 1007 :=
  sorry

end max_ones_on_board_l2753_275368


namespace pauls_diner_cost_l2753_275348

/-- Represents the pricing and discount policy at Paul's Diner -/
structure PaulsDiner where
  sandwich_price : ℕ
  soda_price : ℕ
  discount_threshold : ℕ
  discount_amount : ℕ

/-- Calculates the total cost for a purchase at Paul's Diner -/
def total_cost (diner : PaulsDiner) (sandwiches : ℕ) (sodas : ℕ) : ℕ :=
  let sandwich_cost := diner.sandwich_price * sandwiches
  let soda_cost := diner.soda_price * sodas
  let subtotal := sandwich_cost + soda_cost
  if sandwiches > diner.discount_threshold then
    subtotal - diner.discount_amount
  else
    subtotal

/-- Theorem stating that the total cost for 6 sandwiches and 3 sodas is 29 -/
theorem pauls_diner_cost :
  ∃ (d : PaulsDiner), total_cost d 6 3 = 29 :=
by
  -- The proof goes here
  sorry

end pauls_diner_cost_l2753_275348


namespace marys_bag_check_time_l2753_275333

/-- Represents the time in minutes for Mary's trip to the airport -/
structure AirportTrip where
  uberToHouse : ℕ
  uberToAirport : ℕ
  bagCheck : ℕ
  security : ℕ
  waitForBoarding : ℕ
  waitForTakeoff : ℕ

/-- The total trip time in minutes -/
def totalTripTime (trip : AirportTrip) : ℕ :=
  trip.uberToHouse + trip.uberToAirport + trip.bagCheck + trip.security + trip.waitForBoarding + trip.waitForTakeoff

/-- Mary's airport trip satisfies the given conditions -/
def marysTrip (trip : AirportTrip) : Prop :=
  trip.uberToHouse = 10 ∧
  trip.uberToAirport = 5 * trip.uberToHouse ∧
  trip.security = 3 * trip.bagCheck ∧
  trip.waitForBoarding = 20 ∧
  trip.waitForTakeoff = 2 * trip.waitForBoarding ∧
  totalTripTime trip = 180  -- 3 hours in minutes

theorem marys_bag_check_time (trip : AirportTrip) (h : marysTrip trip) : trip.bagCheck = 15 := by
  sorry

end marys_bag_check_time_l2753_275333


namespace f_satisfies_all_points_l2753_275358

/-- The relation between x and y --/
def f (x : ℝ) : ℝ := -50 * x + 200

/-- The set of points from the given table --/
def points : List (ℝ × ℝ) := [(0, 200), (1, 150), (2, 100), (3, 50), (4, 0)]

/-- Theorem stating that the function f satisfies all points in the given table --/
theorem f_satisfies_all_points : ∀ (p : ℝ × ℝ), p ∈ points → f p.1 = p.2 := by
  sorry

end f_satisfies_all_points_l2753_275358


namespace y_squared_value_l2753_275383

theorem y_squared_value (x y : ℤ) 
  (eq1 : 4 * x + y = 34) 
  (eq2 : 2 * x - y = 20) : 
  y ^ 2 = 4 := by
  sorry

end y_squared_value_l2753_275383


namespace odd_factorials_equal_sum_factorial_l2753_275388

def product_of_odd_factorials (m : ℕ) : ℕ :=
  (List.range m).foldl (λ acc i => acc * Nat.factorial (2 * i + 1)) 1

def sum_of_first_n (m : ℕ) : ℕ :=
  m * (m + 1) / 2

theorem odd_factorials_equal_sum_factorial (m : ℕ) :
  (product_of_odd_factorials m = Nat.factorial (sum_of_first_n m)) ↔ (m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4) := by
  sorry

end odd_factorials_equal_sum_factorial_l2753_275388


namespace tangent_line_to_ln_l2753_275386

-- Define the natural logarithm function
noncomputable def ln : ℝ → ℝ := Real.log

-- Define the tangent line
def tangent_line (a : ℝ) (x : ℝ) : ℝ := x + a

-- State the theorem
theorem tangent_line_to_ln (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 
    tangent_line a x = ln x ∧ 
    (∀ y : ℝ, y > 0 → tangent_line a y ≥ ln y)) →
  a = -1 :=
sorry

end tangent_line_to_ln_l2753_275386


namespace framed_painting_ratio_l2753_275366

theorem framed_painting_ratio :
  ∀ (x : ℝ),
    x > 0 →
    (20 + 2*x) * (30 + 6*x) - 20 * 30 = 20 * 30 * (3/4) →
    (min (20 + 2*x) (30 + 6*x)) / (max (20 + 2*x) (30 + 6*x)) = 1 :=
by sorry

end framed_painting_ratio_l2753_275366


namespace base9_to_base10_653_l2753_275314

/-- Converts a base-9 number to base 10 --/
def base9_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

/-- The base-9 representation of the number --/
def base9_number : List Nat := [3, 5, 6]

theorem base9_to_base10_653 :
  base9_to_base10 base9_number = 534 := by
  sorry

end base9_to_base10_653_l2753_275314


namespace unique_root_condition_l2753_275391

/-- The equation ln(x+a) - 4(x+a)^2 + a = 0 has a unique root if and only if a = (3 ln 2 + 1) / 2 -/
theorem unique_root_condition (a : ℝ) :
  (∃! x : ℝ, Real.log (x + a) - 4 * (x + a)^2 + a = 0) ↔ 
  a = (3 * Real.log 2 + 1) / 2 := by
sorry

end unique_root_condition_l2753_275391


namespace f_3_equals_6_l2753_275396

def f : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * f n

theorem f_3_equals_6 : f 3 = 6 := by
  sorry

end f_3_equals_6_l2753_275396


namespace lashawn_double_kymbrea_after_25_months_l2753_275307

/-- Represents the number of comic books in a collection after a given number of months. -/
def comic_books (initial : ℕ) (rate : ℕ) (months : ℕ) : ℕ :=
  initial + rate * months

theorem lashawn_double_kymbrea_after_25_months :
  let kymbrea_initial := 30
  let kymbrea_rate := 2
  let lashawn_initial := 10
  let lashawn_rate := 6
  let months := 25
  comic_books lashawn_initial lashawn_rate months = 
    2 * comic_books kymbrea_initial kymbrea_rate months := by
  sorry

end lashawn_double_kymbrea_after_25_months_l2753_275307


namespace min_value_problem_l2753_275309

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 9/b = 6) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 9/y = 6 → (a + 1) * (b + 9) ≤ (x + 1) * (y + 9) ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 6 ∧ (x + 1) * (y + 9) = 16 :=
by sorry

end min_value_problem_l2753_275309


namespace face_value_calculation_l2753_275390

/-- Given a company's dividend rate, an investor's return on investment, and the purchase price of shares, 
    calculate the face value of the shares. -/
theorem face_value_calculation (dividend_rate : ℝ) (roi : ℝ) (purchase_price : ℝ) :
  dividend_rate = 0.185 →
  roi = 0.25 →
  purchase_price = 37 →
  ∃ (face_value : ℝ), face_value * dividend_rate = purchase_price * roi ∧ face_value = 50 := by
  sorry

end face_value_calculation_l2753_275390


namespace problem_statement_l2753_275360

theorem problem_statement (a b : ℝ) : 
  let M := {b/a, 1}
  let N := {a, 0}
  (∃ f : ℝ → ℝ, f = id ∧ f '' M ⊆ N) →
  a + b = 1 := by
sorry

end problem_statement_l2753_275360


namespace sqrt_difference_equality_l2753_275380

theorem sqrt_difference_equality : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - Real.sqrt 27 := by
  sorry

end sqrt_difference_equality_l2753_275380


namespace cookie_sales_l2753_275337

theorem cookie_sales (n : ℕ) (a : ℕ) (h1 : n = 10) (h2 : 1 ≤ a) (h3 : a < n) (h4 : 1 + a < n) : a ≤ 8 := by
  sorry

end cookie_sales_l2753_275337


namespace train_delivery_wood_cars_l2753_275316

/-- Represents the train's cargo and delivery parameters -/
structure TrainDelivery where
  coal_cars : ℕ
  iron_cars : ℕ
  station_distance : ℕ
  travel_time : ℕ
  max_coal_deposit : ℕ
  max_iron_deposit : ℕ
  max_wood_deposit : ℕ
  total_delivery_time : ℕ

/-- Calculates the initial number of wood cars -/
def initial_wood_cars (td : TrainDelivery) : ℕ :=
  (td.total_delivery_time / td.travel_time) * td.max_wood_deposit

/-- Theorem stating that given the problem conditions, the initial number of wood cars is 4 -/
theorem train_delivery_wood_cars :
  let td : TrainDelivery := {
    coal_cars := 6,
    iron_cars := 12,
    station_distance := 6,
    travel_time := 25,
    max_coal_deposit := 2,
    max_iron_deposit := 3,
    max_wood_deposit := 1,
    total_delivery_time := 100
  }
  initial_wood_cars td = 4 := by
  sorry


end train_delivery_wood_cars_l2753_275316


namespace rectangle_dimension_change_l2753_275327

theorem rectangle_dimension_change (L B : ℝ) (x : ℝ) (h_pos_L : L > 0) (h_pos_B : B > 0) :
  let new_length := L * (1 + x / 100)
  let new_breadth := B * 0.9
  let new_area := new_length * new_breadth
  let original_area := L * B
  new_area = original_area * 1.035 → x = 15 := by
sorry

end rectangle_dimension_change_l2753_275327


namespace rachel_apple_picking_l2753_275361

theorem rachel_apple_picking (num_trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ) : 
  num_trees = 4 → 
  apples_per_tree = 7 → 
  remaining_apples = 29 → 
  num_trees * apples_per_tree = 28 :=
by sorry

end rachel_apple_picking_l2753_275361


namespace investment_problem_l2753_275356

/-- The investment problem with three partners A, B, and C. -/
theorem investment_problem (investment_B investment_C : ℕ) 
  (profit_B : ℕ) (profit_diff_A_C : ℕ) (investment_A : ℕ) : 
  investment_B = 8000 →
  investment_C = 10000 →
  profit_B = 1000 →
  profit_diff_A_C = 500 →
  (investment_A : ℚ) / investment_B = ((profit_B : ℚ) + profit_diff_A_C) / profit_B →
  (investment_A : ℚ) / investment_C = ((profit_B : ℚ) + profit_diff_A_C) / profit_B →
  investment_A = 12000 := by
  sorry

end investment_problem_l2753_275356


namespace convex_ngon_regions_l2753_275362

/-- The number of regions into which the diagonals of a convex n-gon divide it -/
def f (n : ℕ) : ℕ := (n - 1) * (n - 2) * (n^2 - 3*n + 12) / 24

/-- A convex n-gon is divided into f(n) regions by its diagonals, 
    given that no three diagonals intersect at a single point -/
theorem convex_ngon_regions (n : ℕ) (h : n ≥ 3) : 
  f n = (n - 1) * (n - 2) * (n^2 - 3*n + 12) / 24 := by
  sorry

end convex_ngon_regions_l2753_275362


namespace vector_problem_l2753_275367

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-1, -1)

theorem vector_problem :
  let magnitude := Real.sqrt ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2)
  magnitude = 3 * Real.sqrt 2 ∧
  (let angle := Real.arccos ((a.1 + b.1) * (2 * a.1 - b.1) + (a.2 + b.2) * (2 * a.2 - b.2)) /
    (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) * Real.sqrt ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2))
   angle = π / 4 → a.1 * b.1 + a.2 * b.2 = 1) := by
  sorry

end vector_problem_l2753_275367


namespace bracelet_profit_l2753_275343

/-- Given the following conditions:
    - Total bracelets made
    - Number of bracelets given away
    - Cost of materials
    - Selling price per bracelet
    Prove that the profit equals $8.00 -/
theorem bracelet_profit 
  (total_bracelets : ℕ) 
  (given_away : ℕ) 
  (material_cost : ℚ) 
  (price_per_bracelet : ℚ) 
  (h1 : total_bracelets = 52)
  (h2 : given_away = 8)
  (h3 : material_cost = 3)
  (h4 : price_per_bracelet = 1/4) : 
  (total_bracelets - given_away : ℚ) * price_per_bracelet - material_cost = 8 := by
  sorry

end bracelet_profit_l2753_275343


namespace claires_remaining_balance_l2753_275318

/-- Calculates the remaining balance on Claire's gift card after a week of purchases --/
def remaining_balance (gift_card latte_price croissant_price bagel_price holiday_drink_price cookie_price : ℚ)
  (days bagel_occasions cookies : ℕ) : ℚ :=
  let daily_total := latte_price + croissant_price
  let weekly_total := daily_total * days
  let bagel_total := bagel_price * bagel_occasions
  let friday_treats := holiday_drink_price + cookie_price * cookies
  let friday_adjustment := friday_treats - latte_price
  let total_expenses := weekly_total + bagel_total + friday_adjustment
  gift_card - total_expenses

/-- Theorem stating that Claire's remaining balance is $35.50 --/
theorem claires_remaining_balance :
  remaining_balance 100 3.75 3.50 2.25 4.50 1.25 7 3 5 = 35.50 := by
  sorry

end claires_remaining_balance_l2753_275318


namespace geometric_sequence_property_l2753_275304

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a n > 0 ∧ a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : is_positive_geometric_sequence a)
  (h_prod : a 2 * a 8 = 4) :
  a 5 = 2 := by
sorry

end geometric_sequence_property_l2753_275304


namespace newlyGrownUneatenCorrect_l2753_275375

/-- Represents the number of potatoes in Mary's garden -/
structure PotatoGarden where
  initial : ℕ
  current : ℕ

/-- Calculates the number of newly grown potatoes left uneaten -/
def newlyGrownUneaten (garden : PotatoGarden) : ℕ :=
  garden.current - garden.initial

theorem newlyGrownUneatenCorrect (garden : PotatoGarden) 
  (h1 : garden.initial = 8) 
  (h2 : garden.current = 11) : 
  newlyGrownUneaten garden = 3 := by
  sorry

end newlyGrownUneatenCorrect_l2753_275375


namespace range_of_a_l2753_275373

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x > a^2 - a - 3) → a ∈ Set.Ioo (-1 : ℝ) 2 :=
by sorry

end range_of_a_l2753_275373


namespace perpendicular_line_parallel_planes_l2753_275335

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_parallel_planes 
  (l m n : Line) (α β : Plane) 
  (h1 : l ≠ m) (h2 : l ≠ n) (h3 : m ≠ n) (h4 : α ≠ β)
  (h5 : perpendicularToPlane l α) 
  (h6 : parallel α β) 
  (h7 : containedIn m β) : 
  perpendicular l m :=
sorry

end perpendicular_line_parallel_planes_l2753_275335


namespace probability_sum_13_l2753_275302

def die1 : Finset ℕ := {1, 2, 3, 7, 8, 9}
def die2 : Finset ℕ := {4, 5, 6, 10, 11, 12}

def sumTo13 : Finset (ℕ × ℕ) :=
  (die1.product die2).filter (fun p => p.1 + p.2 = 13)

theorem probability_sum_13 :
  (sumTo13.card : ℚ) / ((die1.card * die2.card) : ℚ) = 1 / 6 := by
  sorry

end probability_sum_13_l2753_275302


namespace band_to_orchestra_ratio_l2753_275369

/-- The number of male musicians in the orchestra -/
def orchestra_males : ℕ := 11

/-- The number of female musicians in the orchestra -/
def orchestra_females : ℕ := 12

/-- The number of male musicians in the choir -/
def choir_males : ℕ := 12

/-- The number of female musicians in the choir -/
def choir_females : ℕ := 17

/-- The total number of musicians in all groups -/
def total_musicians : ℕ := 98

/-- The number of musicians in the orchestra -/
def orchestra_total : ℕ := orchestra_males + orchestra_females

/-- The number of musicians in the choir -/
def choir_total : ℕ := choir_males + choir_females

theorem band_to_orchestra_ratio :
  ∃ (band_musicians : ℕ),
    band_musicians = 2 * orchestra_total ∧
    orchestra_total + band_musicians + choir_total = total_musicians :=
by sorry

end band_to_orchestra_ratio_l2753_275369


namespace friends_money_distribution_l2753_275338

structure Friend :=
  (name : String)
  (initialMoney : ℚ)

def giveMoneyTo (giver receiver : Friend) (fraction : ℚ) : ℚ :=
  giver.initialMoney * fraction

theorem friends_money_distribution (loki moe nick ott pam : Friend) 
  (h1 : ott.initialMoney = 0)
  (h2 : pam.initialMoney = 0)
  (h3 : giveMoneyTo moe ott (1/6) = giveMoneyTo loki ott (1/5))
  (h4 : giveMoneyTo moe ott (1/6) = giveMoneyTo nick ott (1/4))
  (h5 : giveMoneyTo moe pam (1/6) = giveMoneyTo loki pam (1/5))
  (h6 : giveMoneyTo moe pam (1/6) = giveMoneyTo nick pam (1/4)) :
  let totalInitialMoney := loki.initialMoney + moe.initialMoney + nick.initialMoney
  let moneyReceivedByOttAndPam := 2 * (giveMoneyTo moe ott (1/6) + giveMoneyTo loki ott (1/5) + giveMoneyTo nick ott (1/4))
  moneyReceivedByOttAndPam / totalInitialMoney = 2/5 := by
    sorry

#check friends_money_distribution

end friends_money_distribution_l2753_275338


namespace school_teacher_count_l2753_275397

/-- Represents the number of students and teachers in a grade --/
structure GradeData where
  students : ℕ
  teachers : ℕ

/-- Proves that given the conditions, the number of teachers in grade A is 8 and in grade B is 26 --/
theorem school_teacher_count 
  (gradeA gradeB : GradeData)
  (ratioA : gradeA.students = 30 * gradeA.teachers)
  (ratioB : gradeB.students = 40 * gradeB.teachers)
  (newRatioA : gradeA.students + 60 = 25 * (gradeA.teachers + 4))
  (newRatioB : gradeB.students + 80 = 35 * (gradeB.teachers + 6))
  : gradeA.teachers = 8 ∧ gradeB.teachers = 26 := by
  sorry

#check school_teacher_count

end school_teacher_count_l2753_275397


namespace range_of_a_l2753_275325

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (A ∪ B a = {x | x < 1}) ↔ (-1 < a ∧ a ≤ 1) :=
sorry

end range_of_a_l2753_275325


namespace vector_param_validity_l2753_275389

/-- A vector parameterization of a line -/
structure VectorParam where
  x0 : ℝ
  y0 : ℝ
  dx : ℝ
  dy : ℝ

/-- The line equation y = -3/2x - 5 -/
def line_equation (x y : ℝ) : Prop := y = -3/2 * x - 5

/-- Predicate for a valid vector parameterization -/
def is_valid_param (p : VectorParam) : Prop :=
  line_equation p.x0 p.y0 ∧ p.dy = -3/2 * p.dx

theorem vector_param_validity (p : VectorParam) :
  is_valid_param p ↔ ∀ t : ℝ, line_equation (p.x0 + t * p.dx) (p.y0 + t * p.dy) :=
sorry

end vector_param_validity_l2753_275389


namespace student_arrangements_l2753_275308

/-- The number of students in the row -/
def n : ℕ := 7

/-- The number of arrangements where students A and B must stand next to each other -/
def arrangements_adjacent : ℕ := 1440

/-- The number of arrangements where students A, B, and C must not stand next to each other -/
def arrangements_not_adjacent : ℕ := 1440

/-- The number of arrangements where student A is not at the head and student B is not at the tail -/
def arrangements_not_head_tail : ℕ := 3720

theorem student_arrangements :
  (arrangements_adjacent = 1440) ∧
  (arrangements_not_adjacent = 1440) ∧
  (arrangements_not_head_tail = 3720) := by
  sorry

end student_arrangements_l2753_275308


namespace apartment_buildings_count_l2753_275336

/-- The number of floors in each apartment building -/
def floors_per_building : ℕ := 12

/-- The number of apartments on each floor -/
def apartments_per_floor : ℕ := 6

/-- The number of doors needed for each apartment -/
def doors_per_apartment : ℕ := 7

/-- The total number of doors needed to be bought -/
def total_doors : ℕ := 1008

/-- The number of apartment buildings being constructed -/
def num_buildings : ℕ := total_doors / (floors_per_building * apartments_per_floor * doors_per_apartment)

theorem apartment_buildings_count : num_buildings = 2 := by
  sorry

end apartment_buildings_count_l2753_275336


namespace abc_inequality_l2753_275384

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) :=
by sorry

end abc_inequality_l2753_275384


namespace smith_payment_l2753_275329

-- Define the original balance
def original_balance : ℝ := 150

-- Define the finance charge rate
def finance_charge_rate : ℝ := 0.02

-- Define the finance charge calculation
def finance_charge : ℝ := original_balance * finance_charge_rate

-- Define the total payment calculation
def total_payment : ℝ := original_balance + finance_charge

-- Theorem to prove
theorem smith_payment : total_payment = 153 := by
  sorry

end smith_payment_l2753_275329


namespace function_characterization_l2753_275354

theorem function_characterization (f : ℕ → ℕ) :
  (∀ m n : ℕ, (m^2 + f n) ∣ (m * f m + n)) →
  (∀ n : ℕ, f n = n) := by
sorry

end function_characterization_l2753_275354


namespace product_325_4_base_7_l2753_275346

/-- Converts a number from base 7 to base 10 -/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a number from base 10 to base 7 -/
def to_base_7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- Multiplies two numbers in base 7 -/
def mult_base_7 (a b : List Nat) : List Nat :=
  to_base_7 (to_base_10 a * to_base_10 b)

theorem product_325_4_base_7 :
  mult_base_7 [5, 2, 3] [4] = [6, 3, 6, 1] := by sorry

end product_325_4_base_7_l2753_275346


namespace value_of_a_l2753_275372

def A (a : ℝ) : Set ℝ := {a + 2, 2 * a^2 + a}

theorem value_of_a : ∀ a : ℝ, 3 ∈ A a → a = -3/2 := by
  sorry

end value_of_a_l2753_275372


namespace merry_go_round_revolutions_l2753_275315

theorem merry_go_round_revolutions 
  (r₁ : ℝ) (r₂ : ℝ) (rev₁ : ℝ) 
  (h₁ : r₁ = 36) 
  (h₂ : r₂ = 12) 
  (h₃ : rev₁ = 40) 
  (h₄ : r₁ > 0) 
  (h₅ : r₂ > 0) : 
  ∃ rev₂ : ℝ, rev₂ * r₂ = rev₁ * r₁ ∧ rev₂ = 120 :=
sorry

end merry_go_round_revolutions_l2753_275315


namespace sequence_general_term_l2753_275330

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := 2 * n.val - a n

theorem sequence_general_term (a : ℕ+ → ℚ)
  (h : ∀ n : ℕ+, S n a = (n.val : ℚ)) :
  ∀ n : ℕ+, a n = (2^n.val - 1) / 2^(n.val - 1) :=
sorry

end sequence_general_term_l2753_275330


namespace no_real_solutions_for_inequality_l2753_275394

theorem no_real_solutions_for_inequality :
  ¬ ∃ x : ℝ, -x^2 + 2*x - 3 > 0 := by
sorry

end no_real_solutions_for_inequality_l2753_275394


namespace percentage_of_males_l2753_275342

theorem percentage_of_males (total_employees : ℕ) (males_below_50 : ℕ) 
  (h1 : total_employees = 2200)
  (h2 : males_below_50 = 616)
  (h3 : (70 : ℚ) / 100 * (males_below_50 / ((70 : ℚ) / 100)) = males_below_50) :
  (males_below_50 / ((70 : ℚ) / 100)) / total_employees = (40 : ℚ) / 100 := by
  sorry

end percentage_of_males_l2753_275342


namespace weight_of_nine_moles_972_l2753_275398

/-- The weight of a compound given its number of moles and molecular weight -/
def weight_of_compound (moles : ℝ) (molecular_weight : ℝ) : ℝ :=
  moles * molecular_weight

/-- Theorem: The weight of 9 moles of a compound with molecular weight 972 g/mol is 8748 grams -/
theorem weight_of_nine_moles_972 : 
  weight_of_compound 9 972 = 8748 := by
  sorry

end weight_of_nine_moles_972_l2753_275398


namespace grid_midpoint_theorem_l2753_275387

theorem grid_midpoint_theorem (points : Finset (ℤ × ℤ)) :
  points.card = 5 →
  ∃ p1 p2 : ℤ × ℤ, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧
  (∃ x y : ℤ, ((p1.1 + p2.1) / 2 : ℚ) = x ∧ ((p1.2 + p2.2) / 2 : ℚ) = y) :=
by sorry

end grid_midpoint_theorem_l2753_275387


namespace angle_CAD_measure_l2753_275334

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the properties of the triangle and square
def is_right_triangle (A B C : ℝ × ℝ) : Prop := sorry
def is_isosceles (A B C : ℝ × ℝ) : Prop := sorry
def is_square (B C D E : ℝ × ℝ) : Prop := sorry

-- Define angle measurement function
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_CAD_measure 
  (h_right : is_right_triangle A B C)
  (h_isosceles : is_isosceles A B C)
  (h_square : is_square B C D E) :
  angle_measure C A D = 22.5 := by sorry

end angle_CAD_measure_l2753_275334


namespace integers_with_consecutive_twos_l2753_275303

def fibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def valid_integers (n : ℕ) : ℕ := 2^n

def integers_without_consecutive_twos (n : ℕ) : ℕ := fibonacci (n - 1)

theorem integers_with_consecutive_twos (n : ℕ) : 
  valid_integers n - integers_without_consecutive_twos n = 880 → n = 10 := by
  sorry

#eval valid_integers 10 - integers_without_consecutive_twos 10

end integers_with_consecutive_twos_l2753_275303


namespace girls_in_school_l2753_275353

/-- Proves the number of girls in a school given stratified sampling conditions -/
theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) (girls_boys_diff : ℕ) :
  total_students = 2400 →
  sample_size = 200 →
  girls_boys_diff = 10 →
  ∃ (girls_in_sample : ℕ) (girls_in_school : ℕ),
    girls_in_sample + (girls_in_sample + girls_boys_diff) = sample_size ∧
    (girls_in_sample : ℚ) / sample_size = (girls_in_school : ℚ) / total_students ∧
    girls_in_school = 1140 :=
by sorry

end girls_in_school_l2753_275353


namespace diamonds_G6_l2753_275381

/-- The k-th triangular number -/
def T (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The number of diamonds in the n-th figure -/
def diamonds (n : ℕ) : ℕ :=
  1 + 4 * (Finset.sum (Finset.range (n - 1)) (λ i => T (i + 1)))

/-- The theorem stating that the number of diamonds in G_6 is 141 -/
theorem diamonds_G6 : diamonds 6 = 141 := by
  sorry

end diamonds_G6_l2753_275381


namespace maxim_method_correct_only_for_24_l2753_275344

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≤ 9
  h_ones : ones ≤ 9 ∧ ones ≥ 1

/-- Maxim's division method -/
def maximMethod (A : ℚ) (N : TwoDigitNumber) : ℚ :=
  A / (N.tens + N.ones : ℚ) - A / (N.tens * N.ones : ℚ)

/-- The theorem stating that 24 is the only two-digit number for which Maxim's method works -/
theorem maxim_method_correct_only_for_24 :
  ∀ (N : TwoDigitNumber),
    (∀ (A : ℚ), maximMethod A N = A / (10 * N.tens + N.ones : ℚ)) ↔
    (N.tens = 2 ∧ N.ones = 4) :=
sorry

end maxim_method_correct_only_for_24_l2753_275344


namespace complex_number_quadrant_l2753_275382

theorem complex_number_quadrant (z : ℂ) (h : (1 + 2*Complex.I)*z = 3 + Complex.I*z) :
  z.re > 0 ∧ z.im < 0 := by
  sorry

end complex_number_quadrant_l2753_275382


namespace jinho_ribbon_length_l2753_275326

/-- The number of students in Minsu's class -/
def minsu_students : ℕ := 8

/-- The number of students in Jinho's class -/
def jinho_students : ℕ := minsu_students + 1

/-- The total length of ribbon in meters -/
def total_ribbon_m : ℝ := 3.944

/-- The length of ribbon given to each student in Minsu's class in centimeters -/
def ribbon_per_minsu_student_cm : ℝ := 29.05

/-- Conversion factor from meters to centimeters -/
def m_to_cm : ℝ := 100

theorem jinho_ribbon_length :
  let total_ribbon_cm := total_ribbon_m * m_to_cm
  let minsu_total_ribbon_cm := ribbon_per_minsu_student_cm * minsu_students
  let remaining_ribbon_cm := total_ribbon_cm - minsu_total_ribbon_cm
  remaining_ribbon_cm / jinho_students = 18 := by sorry

end jinho_ribbon_length_l2753_275326


namespace sqrt_equation_solution_l2753_275385

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (5 * x + 9) = 11 → x = 22.4 := by
  sorry

end sqrt_equation_solution_l2753_275385


namespace walker_speed_l2753_275355

-- Define the track properties
def track_A_width : ℝ := 6
def track_B_width : ℝ := 8
def track_A_time_diff : ℝ := 36
def track_B_time_diff : ℝ := 48

-- Define the theorem
theorem walker_speed (speed : ℝ) : 
  (2 * Real.pi * track_A_width = speed * track_A_time_diff) →
  (2 * Real.pi * track_B_width = speed * track_B_time_diff) →
  speed = Real.pi / 3 := by
  sorry

end walker_speed_l2753_275355


namespace digit_puzzle_l2753_275310

theorem digit_puzzle (c o u n t s : ℕ) 
  (h1 : c + o = u)
  (h2 : u + n = t)
  (h3 : t + c = s)
  (h4 : o + n + s = 12)
  (h5 : c ≠ 0)
  (h6 : o ≠ 0)
  (h7 : u ≠ 0)
  (h8 : n ≠ 0)
  (h9 : t ≠ 0)
  (h10 : s ≠ 0) :
  t = 6 := by
sorry

end digit_puzzle_l2753_275310


namespace machine_tool_supervision_probability_l2753_275352

theorem machine_tool_supervision_probability :
  let p_no_supervision : ℝ := 0.8000
  let n_tools : ℕ := 4
  let p_at_most_two_require_supervision : ℝ := 1 - (Nat.choose n_tools 3 * (1 - p_no_supervision)^3 * p_no_supervision + Nat.choose n_tools 4 * (1 - p_no_supervision)^4)
  p_at_most_two_require_supervision = 0.9728 := by
sorry

end machine_tool_supervision_probability_l2753_275352


namespace expression_simplification_l2753_275305

theorem expression_simplification :
  (4 * 6) / (12 * 14) * (3 * 5 * 7 * 9) / (4 * 6 * 8) * 7 = 45 / 8 := by
  sorry

end expression_simplification_l2753_275305


namespace correct_oranges_to_remove_l2753_275321

/-- Represents the fruit selection problem -/
structure FruitSelection where
  applePrice : ℚ  -- Price of each apple in cents
  orangePrice : ℚ  -- Price of each orange in cents
  totalFruits : ℕ  -- Total number of fruits initially selected
  initialAvgPrice : ℚ  -- Initial average price of all fruits
  desiredAvgPrice : ℚ  -- Desired average price after removing oranges

/-- Calculates the number of oranges to remove -/
def orangesToRemove (fs : FruitSelection) : ℕ :=
  sorry

/-- Theorem stating the correct number of oranges to remove -/
theorem correct_oranges_to_remove (fs : FruitSelection) 
  (h1 : fs.applePrice = 40/100)
  (h2 : fs.orangePrice = 60/100)
  (h3 : fs.totalFruits = 20)
  (h4 : fs.initialAvgPrice = 56/100)
  (h5 : fs.desiredAvgPrice = 52/100) :
  orangesToRemove fs = 10 := by sorry

end correct_oranges_to_remove_l2753_275321


namespace min_difference_theorem_l2753_275301

noncomputable section

def f (x : ℝ) : ℝ := Real.exp (2 * x)
def g (x : ℝ) : ℝ := Real.log x + 1/2

theorem min_difference_theorem :
  ∃ (h : ℝ → ℝ), ∀ (x₁ : ℝ),
    (∃ (x₂ : ℝ), x₂ > 0 ∧ f x₁ = g x₂) ∧
    (∀ (x₂ : ℝ), x₂ > 0 → f x₁ = g x₂ → h x₁ ≤ x₂ - x₁) ∧
    (∃ (x₁ x₂ : ℝ), x₂ > 0 ∧ f x₁ = g x₂ ∧ h x₁ = x₂ - x₁) ∧
    (∀ (x : ℝ), h x = 1 + Real.log 2 / 2) :=
sorry

end

end min_difference_theorem_l2753_275301


namespace sequence_difference_l2753_275377

theorem sequence_difference (a : ℕ → ℤ) (h : ∀ n, a (n + 1) - a n - n = 0) : 
  a 2017 - a 2016 = 2016 := by
sorry

end sequence_difference_l2753_275377


namespace original_price_from_loss_and_selling_price_l2753_275300

/-- Proves that if an item is sold at a 20% loss for 960 units, then its original price was 1200 units. -/
theorem original_price_from_loss_and_selling_price 
  (loss_percentage : ℝ) 
  (selling_price : ℝ) : 
  loss_percentage = 20 → 
  selling_price = 960 → 
  (1 - loss_percentage / 100) * (selling_price / (1 - loss_percentage / 100)) = 1200 :=
by sorry

end original_price_from_loss_and_selling_price_l2753_275300


namespace retirement_fund_decrease_l2753_275322

/-- Proves that the decrease in Kate's retirement fund is $12 --/
theorem retirement_fund_decrease (previous_value current_value : ℕ) 
  (h1 : previous_value = 1472)
  (h2 : current_value = 1460) : 
  previous_value - current_value = 12 := by
  sorry

end retirement_fund_decrease_l2753_275322


namespace percentage_relation_l2753_275393

theorem percentage_relation (x : ℝ) (h : 0.4 * x = 160) : 0.6 * x = 240 := by
  sorry

end percentage_relation_l2753_275393


namespace ceramic_cup_price_l2753_275363

theorem ceramic_cup_price 
  (total_cups : ℕ) 
  (total_revenue : ℚ) 
  (plastic_cup_price : ℚ) 
  (ceramic_cups_sold : ℕ) 
  (plastic_cups_sold : ℕ) :
  total_cups = 400 →
  total_revenue = 1458 →
  plastic_cup_price = (7/2) →
  ceramic_cups_sold = 284 →
  plastic_cups_sold = 116 →
  (total_revenue - (plastic_cup_price * plastic_cups_sold)) / ceramic_cups_sold = (37/10) := by
  sorry

end ceramic_cup_price_l2753_275363


namespace valid_solution_l2753_275379

def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem valid_solution :
  let A : Set ℕ := {1, 2, 3, 4, 5}
  let B : Set ℕ := {4, 5}
  set_difference A B = {1, 2, 3} := by sorry

end valid_solution_l2753_275379


namespace min_omega_value_l2753_275370

theorem min_omega_value (ω : ℝ) (n : ℤ) : 
  ω > 0 ∧ (4 * π / 3 = n * (2 * π / ω)) → ω ≥ 3 / 2 :=
by sorry

end min_omega_value_l2753_275370


namespace alison_has_4000_l2753_275351

-- Define the amounts of money for each person
def kent_money : ℕ := 1000
def brooke_money : ℕ := 2 * kent_money
def brittany_money : ℕ := 4 * brooke_money
def alison_money : ℕ := brittany_money / 2

-- Theorem statement
theorem alison_has_4000 : alison_money = 4000 := by
  sorry

end alison_has_4000_l2753_275351


namespace lattice_points_count_is_35_l2753_275395

/-- The number of lattice points in the region bounded by the x-axis, 
    the line x=4, and the parabola y=x^2 -/
def lattice_points_count : ℕ :=
  (Finset.range 5).sum (λ x => x^2 + 1)

/-- The theorem stating that the number of lattice points in the specified region is 35 -/
theorem lattice_points_count_is_35 : lattice_points_count = 35 := by
  sorry

end lattice_points_count_is_35_l2753_275395


namespace square_one_on_top_l2753_275323

/-- Represents the possible positions of a square after folding and rotation. -/
inductive Position
  | TopLeft | TopMiddle | TopRight
  | MiddleLeft | Center | MiddleRight
  | BottomLeft | BottomMiddle | BottomRight

/-- Represents the state of the grid after each operation. -/
structure GridState :=
  (positions : Fin 9 → Position)

/-- Folds the right half over the left half. -/
def foldRightOverLeft (state : GridState) : GridState := sorry

/-- Folds the top half over the bottom half. -/
def foldTopOverBottom (state : GridState) : GridState := sorry

/-- Folds the left half over the right half. -/
def foldLeftOverRight (state : GridState) : GridState := sorry

/-- Rotates the entire grid 90 degrees clockwise. -/
def rotateClockwise (state : GridState) : GridState := sorry

/-- The initial state of the grid. -/
def initialState : GridState :=
  { positions := λ i => match i with
    | 0 => Position.TopLeft
    | 1 => Position.TopMiddle
    | 2 => Position.TopRight
    | 3 => Position.MiddleLeft
    | 4 => Position.Center
    | 5 => Position.MiddleRight
    | 6 => Position.BottomLeft
    | 7 => Position.BottomMiddle
    | 8 => Position.BottomRight }

theorem square_one_on_top :
  (rotateClockwise (foldLeftOverRight (foldTopOverBottom (foldRightOverLeft initialState)))).positions 0 = Position.TopLeft := by
  sorry

end square_one_on_top_l2753_275323


namespace parallelogram_height_l2753_275306

/-- The height of a parallelogram with given area and base -/
theorem parallelogram_height (area base height : ℝ) 
  (h_area : area = 864) 
  (h_base : base = 36) 
  (h_formula : area = base * height) : 
  height = 24 := by
  sorry

end parallelogram_height_l2753_275306


namespace netGainDifference_l2753_275313

/-- Represents a job candidate with their associated costs and revenue --/
structure Candidate where
  salary : ℕ
  revenue : ℕ
  trainingMonths : ℕ
  trainingCostPerMonth : ℕ
  hiringBonusPercent : ℕ

/-- Calculates the net gain for the company from a candidate --/
def netGain (c : Candidate) : ℕ :=
  c.revenue - c.salary - (c.trainingMonths * c.trainingCostPerMonth) - (c.salary * c.hiringBonusPercent / 100)

/-- The two candidates as described in the problem --/
def candidate1 : Candidate :=
  { salary := 42000
    revenue := 93000
    trainingMonths := 3
    trainingCostPerMonth := 1200
    hiringBonusPercent := 0 }

def candidate2 : Candidate :=
  { salary := 45000
    revenue := 92000
    trainingMonths := 0
    trainingCostPerMonth := 0
    hiringBonusPercent := 1 }

/-- Theorem stating the difference in net gain between the two candidates --/
theorem netGainDifference : netGain candidate1 - netGain candidate2 = 850 := by
  sorry

end netGainDifference_l2753_275313


namespace equation_solution_l2753_275332

theorem equation_solution : ∃ x : ℝ, 
  6 * ((1/2) * x - 4) + 2 * x = 7 - ((1/3) * x - 1) ∧ x = 6 := by
  sorry

end equation_solution_l2753_275332


namespace coins_problem_l2753_275319

theorem coins_problem (a b c d : ℕ) : 
  a = 21 →                  -- A has 21 coins
  a = b + 9 →               -- A has 9 more coins than B
  c = b + 17 →              -- C has 17 more coins than B
  a + b = c + d - 5 →       -- Sum of A and B is 5 less than sum of C and D
  d = 9 :=                  -- D has 9 coins
by sorry

end coins_problem_l2753_275319


namespace inequality_proof_l2753_275345

theorem inequality_proof (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  (x + y) / (x^2 - x*y + y^2) ≤ (2 * Real.sqrt 2) / Real.sqrt (x^2 + y^2) := by
  sorry

end inequality_proof_l2753_275345


namespace shorter_diagonal_length_l2753_275311

/-- Represents a trapezoid EFGH with given properties -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  side1 : ℝ
  side2 : ℝ
  acute_angles : Bool

/-- The shorter diagonal of the trapezoid -/
def shorter_diagonal (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that for a trapezoid with specific measurements, 
    the shorter diagonal has length 27 -/
theorem shorter_diagonal_length :
  ∀ t : Trapezoid, 
    t.EF = 40 ∧ 
    t.GH = 28 ∧ 
    t.side1 = 13 ∧ 
    t.side2 = 15 ∧ 
    t.acute_angles = true →
    shorter_diagonal t = 27 :=
by
  sorry

end shorter_diagonal_length_l2753_275311


namespace value_of_a_l2753_275340

def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem value_of_a : ∀ a : ℝ, A a ∪ B a = {0, 1, 2, 4, 16} → a = 4 := by
  sorry

end value_of_a_l2753_275340


namespace problem_solution_l2753_275357

-- Define proposition p
def p (k : ℝ) : Prop := k^2 - 8*k - 20 ≤ 0

-- Define proposition q
def q (k : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b < 0 ∧ a = 4 - k ∧ b = 1 - k

-- Define the range of k
def k_range (k : ℝ) : Prop := (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10)

-- Theorem statement
theorem problem_solution (k : ℝ) : (p k ∨ q k) ∧ ¬(p k ∧ q k) → k_range k := by
  sorry

end problem_solution_l2753_275357


namespace divisibility_properties_l2753_275312

theorem divisibility_properties :
  (∃ k : ℤ, 2^41 + 1 = 83 * k) ∧
  (∃ m : ℤ, 2^70 + 3^70 = 13 * m) ∧
  (∃ n : ℤ, 2^60 - 1 = 20801 * n) := by
  sorry

end divisibility_properties_l2753_275312


namespace circle_radius_from_chords_and_midpoint_distance_l2753_275339

theorem circle_radius_from_chords_and_midpoint_distance 
  (chord1 : ℝ) (chord2 : ℝ) (midpoint_distance : ℝ) (radius : ℝ) : 
  chord1 = 10 → 
  chord2 = 12 → 
  midpoint_distance = 4 → 
  (8 * (2 * radius - 8) = 6 * 6) → 
  radius = 6.25 := by sorry

end circle_radius_from_chords_and_midpoint_distance_l2753_275339


namespace bus_journey_l2753_275324

theorem bus_journey (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ)
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 6) :
  ∃ (distance1 : ℝ), 
    distance1 / speed1 + (total_distance - distance1) / speed2 = total_time ∧
    distance1 = 220 := by
  sorry

end bus_journey_l2753_275324
