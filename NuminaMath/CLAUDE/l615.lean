import Mathlib

namespace NUMINAMATH_CALUDE_data_set_average_l615_61514

theorem data_set_average (a : ℝ) : 
  (2 + 3 + 3 + 4 + a) / 5 = 3 → a = 3 := by
sorry

end NUMINAMATH_CALUDE_data_set_average_l615_61514


namespace NUMINAMATH_CALUDE_digit_difference_of_82_l615_61593

theorem digit_difference_of_82 :
  let n : ℕ := 82
  let tens : ℕ := n / 10
  let ones : ℕ := n % 10
  (tens + ones = 10) → (tens - ones = 6) := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_of_82_l615_61593


namespace NUMINAMATH_CALUDE_grass_field_width_l615_61533

theorem grass_field_width (length width path_width cost_per_sqm total_cost : ℝ) :
  length = 95 →
  path_width = 2.5 →
  cost_per_sqm = 2 →
  total_cost = 1550 →
  (length + 2 * path_width) * (width + 2 * path_width) - length * width = total_cost / cost_per_sqm →
  width = 55 := by
sorry

end NUMINAMATH_CALUDE_grass_field_width_l615_61533


namespace NUMINAMATH_CALUDE_acute_triangle_existence_l615_61511

/-- Given n positive real numbers satisfying the max-min relation,
    there exist three that form an acute triangle when n ≥ 13 -/
theorem acute_triangle_existence (n : ℕ) (h : n ≥ 13) :
  ∀ (a : Fin n → ℝ),
  (∀ i, a i > 0) →
  (∀ i j, a i ≤ n * a j) →
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    a i ^ 2 + a j ^ 2 > a k ^ 2 ∧
    a i ^ 2 + a k ^ 2 > a j ^ 2 ∧
    a j ^ 2 + a k ^ 2 > a i ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_existence_l615_61511


namespace NUMINAMATH_CALUDE_congruence_solution_l615_61571

theorem congruence_solution (x a m : ℕ) : m ≥ 2 → a < m → (15 * x + 2) % 20 = 7 % 20 → x % m = a % m → a + m = 7 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l615_61571


namespace NUMINAMATH_CALUDE_sum_positive_differences_equals_754152_l615_61550

-- Define the set S
def S : Finset ℕ := Finset.range 11

-- Define the function to calculate 3^n
def pow3 (n : ℕ) : ℕ := 3^n

-- Define the sum of positive differences
def sumPositiveDifferences : ℕ :=
  Finset.sum S (fun i =>
    Finset.sum S (fun j =>
      if pow3 j > pow3 i then pow3 j - pow3 i else 0
    )
  )

-- Theorem statement
theorem sum_positive_differences_equals_754152 :
  sumPositiveDifferences = 754152 := by sorry

end NUMINAMATH_CALUDE_sum_positive_differences_equals_754152_l615_61550


namespace NUMINAMATH_CALUDE_ned_good_games_l615_61505

/-- The number of good games Ned ended up with -/
def good_games (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (non_working_games : ℕ) : ℕ :=
  games_from_friend + games_from_garage_sale - non_working_games

/-- Proof that Ned ended up with 3 good games -/
theorem ned_good_games :
  good_games 50 27 74 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ned_good_games_l615_61505


namespace NUMINAMATH_CALUDE_max_pens_with_ten_dollars_l615_61516

/-- Represents the maximum number of pens that can be bought with a given budget. -/
def maxPens (budget : ℕ) : ℕ :=
  let singlePenCost : ℕ := 1
  let fourPackCost : ℕ := 3
  let sevenPackCost : ℕ := 4
  -- The actual calculation would go here
  16

/-- Theorem stating that with a $10 budget, the maximum number of pens that can be bought is 16. -/
theorem max_pens_with_ten_dollars :
  maxPens 10 = 16 := by
  sorry

#eval maxPens 10

end NUMINAMATH_CALUDE_max_pens_with_ten_dollars_l615_61516


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l615_61565

/-- The minimum number of additional coins needed -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (friends * (friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for Alex's problem -/
theorem alex_coin_distribution (friends : ℕ) (initial_coins : ℕ)
  (h1 : friends = 15)
  (h2 : initial_coins = 105) :
  min_additional_coins friends initial_coins = 15 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l615_61565


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l615_61543

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2*x - 3}

-- Define set N
def N : Set ℝ := {x | -5 ≤ x ∧ x ≤ 2}

-- Define the complement of N with respect to ℝ
def complement_N : Set ℝ := {x | x < -5 ∨ 2 < x}

-- Theorem statement
theorem intersection_M_complement_N : M ∩ complement_N = {y | y > 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l615_61543


namespace NUMINAMATH_CALUDE_sine_square_sum_condition_l615_61551

theorem sine_square_sum_condition (α β : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : 0 < β) (h4 : β < π/2) : 
  (Real.sin α)^2 + (Real.sin β)^2 = (Real.sin (α + β))^2 ↔ α + β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_sine_square_sum_condition_l615_61551


namespace NUMINAMATH_CALUDE_opposite_sign_roots_l615_61595

theorem opposite_sign_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - (a + 3) * x + 2 = 0 ∧ 
               a * y^2 - (a + 3) * y + 2 = 0 ∧ 
               x * y < 0) ↔ 
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_opposite_sign_roots_l615_61595


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l615_61523

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I) = 4 - 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l615_61523


namespace NUMINAMATH_CALUDE_linear_function_proof_l615_61583

/-- Given a linear function f(x) = kx passing through (2,4), prove f(-2) = -4 -/
theorem linear_function_proof (k : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = k * x) →  -- Function definition
  f 2 = 4 →             -- Point (2,4) lies on the graph
  f (-2) = -4 :=        -- Prove f(-2) = -4
by sorry

end NUMINAMATH_CALUDE_linear_function_proof_l615_61583


namespace NUMINAMATH_CALUDE_inequality_proof_l615_61531

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x + y + z) * (x + y - z) * (x - y + z) / (x * y * z) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l615_61531


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_difference_l615_61564

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := n + 1

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

-- Define the geometric sequence b_n
def b (n : ℕ) : ℚ := 32 * (1/2)^(n-1)

-- Define the sum of the first n terms of a_n - b_n
def T (n : ℕ) : ℚ := n * (n + 3) / 2 + 2^(6-n) - 64

theorem arithmetic_geometric_sum_difference 
  (h1 : a 1 = 2) 
  (h2 : S 5 = 20) 
  (h3 : a 4 + b 4 = 9) :
  ∀ n : ℕ, T n = (S n) - (b 1 * (1 - (1/2)^n) / (1 - 1/2)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_difference_l615_61564


namespace NUMINAMATH_CALUDE_inequality_proof_l615_61548

theorem inequality_proof (x y z : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) :
  (1 + 2*x + 2*y + 2*z) * (1 + 2*y + 2*z) * (1 + 2*x + 2*z) * (1 + 2*x + 2*y) ≥ 
  (1 + 3*x + 3*y) * (1 + 3*y + 3*z) * (1 + 3*x) * (1 + 3*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l615_61548


namespace NUMINAMATH_CALUDE_two_digit_divisor_of_2701_l615_61570

/-- Two-digit number type -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n ≤ 99 }

/-- Sum of squares of digits of a two-digit number -/
def sumOfSquaresOfDigits (x : TwoDigitNumber) : ℕ :=
  let tens := x.val / 10
  let ones := x.val % 10
  tens * tens + ones * ones

/-- Main theorem -/
theorem two_digit_divisor_of_2701 (x : TwoDigitNumber) : 
  (2701 % x.val = 0) ↔ (sumOfSquaresOfDigits x = 58) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_divisor_of_2701_l615_61570


namespace NUMINAMATH_CALUDE_line_parametric_equation_l615_61582

theorem line_parametric_equation :
  ∀ (t : ℝ), 2 * (1 - t) - (3 - 2 * t) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_parametric_equation_l615_61582


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l615_61537

theorem quadratic_equation_solution :
  let f (x : ℝ) := (2*x + 1)^2 - (2*x + 1)*(x - 1)
  ∀ x : ℝ, f x = 0 ↔ x = -1/2 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l615_61537


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l615_61562

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| > 3} = {x : ℝ | x < -4 ∨ x > 2} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l615_61562


namespace NUMINAMATH_CALUDE_wine_barrel_system_l615_61529

/-- Represents the capacity of a large barrel in hu -/
def large_barrel_capacity : ℝ := sorry

/-- Represents the capacity of a small barrel in hu -/
def small_barrel_capacity : ℝ := sorry

/-- The total capacity of 6 large barrels and 4 small barrels is 48 hu -/
axiom first_equation : 6 * large_barrel_capacity + 4 * small_barrel_capacity = 48

/-- The total capacity of 5 large barrels and 3 small barrels is 38 hu -/
axiom second_equation : 5 * large_barrel_capacity + 3 * small_barrel_capacity = 38

/-- The system of equations representing the wine barrel problem -/
theorem wine_barrel_system :
  (6 * large_barrel_capacity + 4 * small_barrel_capacity = 48) ∧
  (5 * large_barrel_capacity + 3 * small_barrel_capacity = 38) := by
  sorry

end NUMINAMATH_CALUDE_wine_barrel_system_l615_61529


namespace NUMINAMATH_CALUDE_mrs_hilt_picture_frame_perimeter_l615_61576

/-- The perimeter of a rectangular picture frame. -/
def perimeter_picture_frame (height length : ℕ) : ℕ :=
  2 * (height + length)

/-- Theorem: The perimeter of Mrs. Hilt's picture frame is 44 inches. -/
theorem mrs_hilt_picture_frame_perimeter :
  perimeter_picture_frame 12 10 = 44 := by
  sorry

#eval perimeter_picture_frame 12 10

end NUMINAMATH_CALUDE_mrs_hilt_picture_frame_perimeter_l615_61576


namespace NUMINAMATH_CALUDE_sequence_sum_properties_l615_61538

/-- Given a sequence {a_n} with sum of first n terms S_n = n^2 - 3,
    prove the first term and general term. -/
theorem sequence_sum_properties (a : ℕ → ℤ) (S : ℕ → ℤ) 
    (h : ∀ n, S n = n^2 - 3) :
  (a 1 = -2) ∧ 
  (∀ n ≥ 2, a n = 2*n - 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_properties_l615_61538


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l615_61569

theorem smallest_integer_solution (x : ℤ) : 
  (3 - 5 * x > 24) ↔ (x ≤ -5) :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l615_61569


namespace NUMINAMATH_CALUDE_minimize_fuel_consumption_l615_61549

theorem minimize_fuel_consumption
  (total_cargo : ℕ)
  (large_capacity small_capacity : ℕ)
  (large_fuel small_fuel : ℕ)
  (h1 : total_cargo = 157)
  (h2 : large_capacity = 5)
  (h3 : small_capacity = 2)
  (h4 : large_fuel = 20)
  (h5 : small_fuel = 10) :
  ∃ (large_trucks small_trucks : ℕ),
    large_trucks * large_capacity + small_trucks * small_capacity ≥ total_cargo ∧
    ∀ (x y : ℕ),
      x * large_capacity + y * small_capacity ≥ total_cargo →
      x * large_fuel + y * small_fuel ≥ large_trucks * large_fuel + small_trucks * small_fuel →
      x = large_trucks ∧ y = small_trucks :=
by sorry

end NUMINAMATH_CALUDE_minimize_fuel_consumption_l615_61549


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l615_61526

def f (x : ℝ) := x^3 + 1.1*x^2 + 0.9*x - 1.4

theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo 0.625 0.6875, f c = 0 :=
by
  have h1 : f 0.625 < 0 := by sorry
  have h2 : f 0.6875 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l615_61526


namespace NUMINAMATH_CALUDE_least_value_quadratic_l615_61560

theorem least_value_quadratic (x : ℝ) : 
  (5 * x^2 + 7 * x + 3 = 6) → x ≥ (-7 - Real.sqrt 109) / 10 := by
  sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l615_61560


namespace NUMINAMATH_CALUDE_basketball_games_played_l615_61577

theorem basketball_games_played (x : ℕ) : 
  (3 : ℚ) / 4 * x + (1 : ℚ) / 4 * x = x ∧ 
  (2 : ℚ) / 3 * (x + 12) = (3 : ℚ) / 4 * x + 6 ∧ 
  (1 : ℚ) / 3 * (x + 12) = (1 : ℚ) / 4 * x + 6 → 
  x = 24 := by sorry

end NUMINAMATH_CALUDE_basketball_games_played_l615_61577


namespace NUMINAMATH_CALUDE_total_shells_is_245_l615_61541

/-- The number of shells each person has -/
structure Shells :=
  (david : ℕ)
  (mia : ℕ)
  (ava : ℕ)
  (alice : ℕ)
  (liam : ℕ)

/-- The conditions of the problem -/
def shellConditions (s : Shells) : Prop :=
  s.david = 15 ∧
  s.mia = 4 * s.david ∧
  s.ava = s.mia + 20 ∧
  s.alice = s.ava / 2 ∧
  s.liam = 2 * (s.alice - s.david)

/-- The total number of shells -/
def totalShells (s : Shells) : ℕ :=
  s.david + s.mia + s.ava + s.alice + s.liam

/-- Theorem: Given the conditions, the total number of shells is 245 -/
theorem total_shells_is_245 (s : Shells) (h : shellConditions s) : totalShells s = 245 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_is_245_l615_61541


namespace NUMINAMATH_CALUDE_tricia_age_l615_61552

-- Define the ages of all people as natural numbers
def Vincent : ℕ := 22
def Rupert : ℕ := Vincent - 2
def Khloe : ℕ := Rupert - 10
def Eugene : ℕ := Khloe * 3
def Yorick : ℕ := Eugene * 2
def Selena : ℕ := Yorick - 5
def Amilia : ℕ := Selena - 3
def Tricia : ℕ := Amilia / 3

-- Theorem to prove Tricia's age
theorem tricia_age : Tricia = 17 := by
  sorry

end NUMINAMATH_CALUDE_tricia_age_l615_61552


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l615_61522

/-- The coordinates of a point with respect to the origin are the same as its definition. -/
theorem point_coordinates_wrt_origin (x y : ℝ) : 
  let P : ℝ × ℝ := (x, y)
  (P.1, P.2) = (x, y) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l615_61522


namespace NUMINAMATH_CALUDE_solve_equation_l615_61563

theorem solve_equation : ∃ y : ℚ, (2 * y + 3 * y = 500 - (4 * y + 6 * y)) ∧ y = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l615_61563


namespace NUMINAMATH_CALUDE_sunday_newspaper_cost_l615_61527

/-- The cost of the Sunday edition of a newspaper -/
def sunday_cost (weekday_cost : ℚ) (total_cost : ℚ) (num_weeks : ℕ) : ℚ :=
  (total_cost - 3 * weekday_cost * num_weeks) / num_weeks

/-- Theorem stating that the Sunday edition costs $2.00 -/
theorem sunday_newspaper_cost : 
  let weekday_cost : ℚ := 1/2
  let total_cost : ℚ := 28
  let num_weeks : ℕ := 8
  sunday_cost weekday_cost total_cost num_weeks = 2 := by
sorry

end NUMINAMATH_CALUDE_sunday_newspaper_cost_l615_61527


namespace NUMINAMATH_CALUDE_tan_alpha_value_l615_61540

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l615_61540


namespace NUMINAMATH_CALUDE_train_length_calculation_l615_61555

/-- Given a train crossing a bridge, calculate its length. -/
theorem train_length_calculation 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (train_speed : ℝ) 
  (h1 : bridge_length = 480) 
  (h2 : crossing_time = 55) 
  (h3 : train_speed = 39.27272727272727) : 
  train_speed * crossing_time - bridge_length = 1680 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l615_61555


namespace NUMINAMATH_CALUDE_truck_transport_problem_l615_61508

/-- Represents the problem of determining the number of trucks needed to transport goods --/
theorem truck_transport_problem (truck_capacity : ℕ) (partial_load : ℕ) (remaining_goods : ℕ) :
  truck_capacity = 8 →
  partial_load = 4 →
  remaining_goods = 20 →
  ∃ (num_trucks : ℕ) (total_goods : ℕ),
    num_trucks = 6 ∧
    total_goods = 44 ∧
    partial_load * num_trucks + remaining_goods = total_goods ∧
    0 < total_goods - truck_capacity * (num_trucks - 1) ∧
    total_goods - truck_capacity * (num_trucks - 1) < truck_capacity :=
by sorry

end NUMINAMATH_CALUDE_truck_transport_problem_l615_61508


namespace NUMINAMATH_CALUDE_remainder_2468135790_mod_99_l615_61553

theorem remainder_2468135790_mod_99 :
  2468135790 % 99 = 54 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2468135790_mod_99_l615_61553


namespace NUMINAMATH_CALUDE_infinite_twin_pretty_numbers_l615_61510

/-- A positive integer is a "pretty number" if each of its prime factors appears with an exponent of at least 2 in its prime factorization. -/
def is_pretty_number (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (∃ k : ℕ, k ≥ 2 ∧ p^k ∣ n)

/-- Two consecutive positive integers that are both "pretty numbers" are called "twin pretty numbers." -/
def is_twin_pretty_numbers (n : ℕ) : Prop :=
  is_pretty_number n ∧ is_pretty_number (n + 1)

/-- For any pair of twin pretty numbers, there exists a larger pair of twin pretty numbers. -/
theorem infinite_twin_pretty_numbers :
  ∀ n : ℕ, is_twin_pretty_numbers n →
    ∃ m : ℕ, m > n + 1 ∧ is_twin_pretty_numbers m :=
sorry

end NUMINAMATH_CALUDE_infinite_twin_pretty_numbers_l615_61510


namespace NUMINAMATH_CALUDE_greatest_gcd_triangular_number_l615_61559

/-- The nth triangular number -/
def T (n : ℕ+) : ℕ := n.val * (n.val + 1) / 2

/-- The greatest possible value of gcd(6T_n, n-2) is 12 -/
theorem greatest_gcd_triangular_number :
  ∃ (n : ℕ+), Nat.gcd (6 * T n) (n.val - 2) = 12 ∧
  ∀ (m : ℕ+), Nat.gcd (6 * T m) (m.val - 2) ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_greatest_gcd_triangular_number_l615_61559


namespace NUMINAMATH_CALUDE_marbles_after_sharing_undetermined_l615_61512

/-- Represents the items Carolyn has -/
structure CarolynItems where
  marbles : ℕ
  oranges : ℕ

/-- Represents the sharing action -/
def share (items : CarolynItems) (shared : ℕ) : Prop :=
  shared ≤ items.marbles + items.oranges

/-- Theorem stating that the number of marbles Carolyn ends up with is undetermined -/
theorem marbles_after_sharing_undetermined 
  (initial : CarolynItems) 
  (shared : ℕ) 
  (h1 : initial.marbles = 47)
  (h2 : initial.oranges = 6)
  (h3 : share initial shared)
  (h4 : shared = 42) :
  ∃ (final : CarolynItems), final.marbles ≤ initial.marbles ∧ 
    final.marbles + final.oranges = initial.marbles + initial.oranges - shared :=
sorry

end NUMINAMATH_CALUDE_marbles_after_sharing_undetermined_l615_61512


namespace NUMINAMATH_CALUDE_solution_l615_61575

def problem (x : ℝ) (number : ℝ) : Prop :=
  number = 3639 + 11.95 - x

theorem solution (x : ℝ) (number : ℝ) 
  (h1 : problem x number) 
  (h2 : x = 596.95) : 
  number = 3054 := by
  sorry

end NUMINAMATH_CALUDE_solution_l615_61575


namespace NUMINAMATH_CALUDE_right_angled_triangle_obtuse_triangle_l615_61521

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = 180

-- Define types of triangles
def is_right_angled (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

def is_obtuse (t : Triangle) : Prop :=
  t.A > 90 ∨ t.B > 90 ∨ t.C > 90

-- Theorem for the first part
theorem right_angled_triangle (t : Triangle) (h1 : t.A = 30) (h2 : t.B = 60) :
  is_right_angled t :=
by sorry

-- Theorem for the second part
theorem obtuse_triangle (t : Triangle) (h : t.A / t.B = 1 / 3 ∧ t.B / t.C = 3 / 5) :
  is_obtuse t :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_obtuse_triangle_l615_61521


namespace NUMINAMATH_CALUDE_decision_not_basic_l615_61509

-- Define the type for flowchart structures
inductive FlowchartStructure
  | Sequence
  | Condition
  | Loop
  | Decision

-- Define the set of basic logic structures
def basic_logic_structures : Set FlowchartStructure :=
  {FlowchartStructure.Sequence, FlowchartStructure.Condition, FlowchartStructure.Loop}

-- Theorem: Decision structure is not in the set of basic logic structures
theorem decision_not_basic : FlowchartStructure.Decision ∉ basic_logic_structures := by
  sorry

end NUMINAMATH_CALUDE_decision_not_basic_l615_61509


namespace NUMINAMATH_CALUDE_khali_snow_volume_l615_61557

/-- The volume of snow on Khali's driveway -/
def snow_volume (length width height : ℚ) : ℚ := length * width * height

theorem khali_snow_volume :
  snow_volume 30 4 (3/4) = 90 := by
  sorry

end NUMINAMATH_CALUDE_khali_snow_volume_l615_61557


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l615_61544

theorem quadratic_form_equivalence (b : ℝ) (h1 : b > 0) :
  (∃ n : ℝ, ∀ x : ℝ, x^2 + b*x + 36 = (x + n)^2 + 20) → b = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l615_61544


namespace NUMINAMATH_CALUDE_art_gallery_sculptures_l615_61586

theorem art_gallery_sculptures (total_pieces : ℕ) 
  (h1 : total_pieces = 2700)
  (h2 : ∃ (displayed : ℕ), displayed = total_pieces / 3)
  (h3 : ∃ (displayed_sculptures : ℕ), 
    displayed_sculptures = (total_pieces / 3) / 6)
  (h4 : ∃ (not_displayed_paintings : ℕ), 
    not_displayed_paintings = (total_pieces * 2 / 3) / 3)
  (h5 : ∃ (not_displayed_sculptures : ℕ), 
    not_displayed_sculptures > 0) : 
  ∃ (sculptures_not_displayed : ℕ), 
    sculptures_not_displayed = 1200 := by
sorry

end NUMINAMATH_CALUDE_art_gallery_sculptures_l615_61586


namespace NUMINAMATH_CALUDE_sine_theorem_l615_61573

theorem sine_theorem (a b c α β γ : ℝ) 
  (h1 : a / Real.sin α = b / Real.sin β)
  (h2 : b / Real.sin β = c / Real.sin γ)
  (h3 : α + β + γ = Real.pi) : 
  (a = b * Real.cos γ + c * Real.cos β) ∧ 
  (b = c * Real.cos α + a * Real.cos γ) ∧ 
  (c = a * Real.cos β + b * Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_sine_theorem_l615_61573


namespace NUMINAMATH_CALUDE_angle_relationship_l615_61585

theorem angle_relationship (α β : Real) 
  (h1 : 0 < α) 
  (h2 : α < 2 * β) 
  (h3 : 2 * β ≤ π / 2)
  (h4 : 2 * Real.cos (α + β) * Real.cos β = -1 + 2 * Real.sin (α + β) * Real.sin β) : 
  α + 2 * β = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_relationship_l615_61585


namespace NUMINAMATH_CALUDE_larger_number_problem_l615_61546

theorem larger_number_problem (L S : ℕ) (hL : L > S) : 
  L - S = 2415 → L = 21 * S + 15 → L = 2535 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l615_61546


namespace NUMINAMATH_CALUDE_cuboid_volume_l615_61567

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 3) (h2 : a * c = 5) (h3 : b * c = 15) : 
  a * b * c = 15 := by
sorry

end NUMINAMATH_CALUDE_cuboid_volume_l615_61567


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l615_61594

/-- Given p, q, r are roots of x³ - 3x - 2 = 0, prove p(q - r)² + q(r - p)² + r(p - q)² = -6 -/
theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 - 3*p - 2 = 0) → 
  (q^3 - 3*q - 2 = 0) → 
  (r^3 - 3*r - 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l615_61594


namespace NUMINAMATH_CALUDE_right_angled_triangle_l615_61542

theorem right_angled_triangle (A B C : Real) (h : A + B + C = Real.pi) 
  (eq : (Real.sin A)^2 + (Real.sin B)^2 + (Real.sin C)^2 = 
        2 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2)) : 
  A = Real.pi/2 ∨ B = Real.pi/2 ∨ C = Real.pi/2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l615_61542


namespace NUMINAMATH_CALUDE_tangent_relations_l615_61532

theorem tangent_relations (α : Real) (h : Real.tan (α / 2) = 2) :
  Real.tan α = -4/3 ∧
  Real.tan (α + π/4) = -1/7 ∧
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_relations_l615_61532


namespace NUMINAMATH_CALUDE_square_of_102_l615_61592

theorem square_of_102 : 102 * 102 = 10404 := by
  sorry

end NUMINAMATH_CALUDE_square_of_102_l615_61592


namespace NUMINAMATH_CALUDE_largest_root_ratio_l615_61507

def f (x : ℝ) : ℝ := 1 - x - 4*x^2 + x^4

def g (x : ℝ) : ℝ := 16 - 8*x - 16*x^2 + x^4

theorem largest_root_ratio :
  ∃ (x₁ x₂ : ℝ),
    (∀ y, f y = 0 → y ≤ x₁) ∧
    (f x₁ = 0) ∧
    (∀ z, g z = 0 → z ≤ x₂) ∧
    (g x₂ = 0) ∧
    x₁ / x₂ = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_root_ratio_l615_61507


namespace NUMINAMATH_CALUDE_three_valid_starting_days_l615_61574

/-- Represents the days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the number of occurrences of a specific weekday in a 30-day month starting on a given day -/
def countWeekday (start : Weekday) (day : Weekday) : Nat :=
  sorry

/-- Checks if the number of Tuesdays and Fridays are equal in a 30-day month starting on a given day -/
def equalTuesdaysFridays (start : Weekday) : Prop :=
  countWeekday start Weekday.Tuesday = countWeekday start Weekday.Friday

/-- The set of all possible starting days that result in equal Tuesdays and Fridays -/
def validStartingDays : Finset Weekday :=
  sorry

/-- Theorem stating that there are exactly 3 valid starting days for a 30-day month with equal Tuesdays and Fridays -/
theorem three_valid_starting_days :
  Finset.card validStartingDays = 3 :=
sorry

end NUMINAMATH_CALUDE_three_valid_starting_days_l615_61574


namespace NUMINAMATH_CALUDE_problem_solution_l615_61580

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2^x + a*x

theorem problem_solution (a : ℝ) : f a (f a 1) = 4 * a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l615_61580


namespace NUMINAMATH_CALUDE_disjoint_subset_count_is_three_pow_l615_61591

/-- The number of ways to select two disjoint subsets from a set with n elements -/
def disjointSubsetCount (n : ℕ) : ℕ := 3^n

/-- Theorem: The number of ways to select two disjoint subsets from a set with n elements is 3^n -/
theorem disjoint_subset_count_is_three_pow (n : ℕ) : disjointSubsetCount n = 3^n := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subset_count_is_three_pow_l615_61591


namespace NUMINAMATH_CALUDE_complex_argument_range_l615_61554

theorem complex_argument_range (z : ℂ) (h : Complex.abs (2 * z + 1 / z) = 1) :
  ∃ (θ : ℝ), Complex.arg z = θ ∧
  ((θ ∈ Set.Icc (Real.pi / 2 - 1 / 2 * Real.arccos (3 / 4)) (Real.pi / 2 + 1 / 2 * Real.arccos (3 / 4))) ∨
   (θ ∈ Set.Icc (3 * Real.pi / 2 - 1 / 2 * Real.arccos (3 / 4)) (3 * Real.pi / 2 + 1 / 2 * Real.arccos (3 / 4)))) :=
by sorry

end NUMINAMATH_CALUDE_complex_argument_range_l615_61554


namespace NUMINAMATH_CALUDE_refrigerator_price_correct_l615_61500

/-- The purchase price of a refrigerator, given specific conditions on its sale and the sale of a mobile phone --/
def refrigerator_price : ℝ :=
  let mobile_price : ℝ := 8000
  let refrigerator_loss_percent : ℝ := 0.04
  let mobile_profit_percent : ℝ := 0.11
  let total_profit : ℝ := 280
  
  -- Define the equation to solve
  let equation (x : ℝ) : Prop :=
    (mobile_price * (1 + mobile_profit_percent) - mobile_price) - 
    (x - x * (1 - refrigerator_loss_percent)) = total_profit
  
  -- The solution (to be proved)
  15000

theorem refrigerator_price_correct : 
  refrigerator_price = 15000 := by sorry

end NUMINAMATH_CALUDE_refrigerator_price_correct_l615_61500


namespace NUMINAMATH_CALUDE_no_correlation_iff_deterministic_l615_61524

/-- Represents a pair of variables -/
inductive VariablePair
  | HeightEyesight
  | PointCoordinates
  | RiceYieldFertilizer
  | ExamScoreReviewTime
  | DistanceTime
  | IncomeHouseholdTax
  | SalesAdvertising

/-- Defines whether a pair of variables has a deterministic relationship -/
def isDeterministic (pair : VariablePair) : Prop :=
  match pair with
  | VariablePair.HeightEyesight => true
  | VariablePair.PointCoordinates => true
  | VariablePair.RiceYieldFertilizer => false
  | VariablePair.ExamScoreReviewTime => false
  | VariablePair.DistanceTime => true
  | VariablePair.IncomeHouseholdTax => false
  | VariablePair.SalesAdvertising => false

/-- Defines whether a pair of variables exhibits a correlation relationship -/
def isCorrelated (pair : VariablePair) : Prop :=
  ¬(isDeterministic pair)

/-- The main theorem stating that the pairs without correlation are exactly those with deterministic relationships -/
theorem no_correlation_iff_deterministic :
  ∀ (pair : VariablePair), ¬(isCorrelated pair) ↔ isDeterministic pair := by
  sorry

end NUMINAMATH_CALUDE_no_correlation_iff_deterministic_l615_61524


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l615_61520

def product : ℕ := 30 * 31 * 32 * 33 * 34 * 35
def denominator : ℕ := 10000

theorem units_digit_of_fraction :
  (product / denominator) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l615_61520


namespace NUMINAMATH_CALUDE_litter_collection_weight_l615_61503

/-- The total weight of litter collected by Gina and her neighbors -/
def total_litter_weight (gina_glass_bags : ℕ) (gina_plastic_bags : ℕ)
  (glass_weight : ℕ) (plastic_weight : ℕ) (neighbor_glass_factor : ℕ)
  (neighbor_plastic_factor : ℕ) : ℕ :=
  let gina_glass := gina_glass_bags * glass_weight
  let gina_plastic := gina_plastic_bags * plastic_weight
  let neighbor_glass := neighbor_glass_factor * gina_glass
  let neighbor_plastic := neighbor_plastic_factor * gina_plastic
  gina_glass + gina_plastic + neighbor_glass + neighbor_plastic

/-- Theorem stating the total weight of litter collected -/
theorem litter_collection_weight :
  total_litter_weight 5 3 7 4 120 80 = 5207 := by
  sorry

end NUMINAMATH_CALUDE_litter_collection_weight_l615_61503


namespace NUMINAMATH_CALUDE_mom_tshirt_packages_l615_61568

theorem mom_tshirt_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) : 
  total_tshirts = 51 → tshirts_per_package = 3 → total_tshirts / tshirts_per_package = 17 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_packages_l615_61568


namespace NUMINAMATH_CALUDE_parabola_translation_original_to_result_l615_61528

/-- Represents a parabola in the form y = (x - h)^2 + k, where (h, k) is the vertex --/
structure Parabola where
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { h := p.h - dx, k := p.k + dy }

theorem parabola_translation (p : Parabola) (dx dy : ℝ) :
  translate p dx dy = { h := p.h - dx, k := p.k + dy } := by sorry

theorem original_to_result :
  let original := Parabola.mk 2 (-8)
  let result := translate original 3 5
  result = Parabola.mk (-1) (-3) := by sorry

end NUMINAMATH_CALUDE_parabola_translation_original_to_result_l615_61528


namespace NUMINAMATH_CALUDE_complex_norm_squared_l615_61558

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.normSq z = 5 - (2*I)^2) : 
  Complex.normSq z = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_norm_squared_l615_61558


namespace NUMINAMATH_CALUDE_integer_root_cubic_equation_l615_61539

theorem integer_root_cubic_equation :
  ∀ x : ℤ, x^3 - 4*x^2 - 11*x + 24 = 0 ↔ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_integer_root_cubic_equation_l615_61539


namespace NUMINAMATH_CALUDE_equation_describes_ellipse_l615_61545

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y - 4)^2) + Real.sqrt ((x - 8)^2 + (y + 1)^2) = 15

-- Define what it means for a point to be on the conic
def point_on_conic (x y : ℝ) : Prop := conic_equation x y

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (2, 4)
def focus2 : ℝ × ℝ := (8, -1)

-- Theorem stating that the equation describes an ellipse
theorem equation_describes_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), point_on_conic x y ↔
    (x - (focus1.1 + focus2.1) / 2)^2 / a^2 +
    (y - (focus1.2 + focus2.2) / 2)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_equation_describes_ellipse_l615_61545


namespace NUMINAMATH_CALUDE_constant_function_proof_l615_61596

theorem constant_function_proof (f : ℤ × ℤ → ℝ) 
  (h1 : ∀ (x y : ℤ), 0 ≤ f (x, y) ∧ f (x, y) ≤ 1)
  (h2 : ∀ (x y : ℤ), f (x, y) = (f (x - 1, y) + f (x, y - 1)) / 2) :
  ∃ (c : ℝ), ∀ (x y : ℤ), f (x, y) = c :=
sorry

end NUMINAMATH_CALUDE_constant_function_proof_l615_61596


namespace NUMINAMATH_CALUDE_water_addition_changes_ratio_l615_61534

/-- Proves that adding 21 litres of water to a 45-litre mixture with initial milk to water ratio of 4:1 results in a new mixture with milk to water ratio of 1.2 -/
theorem water_addition_changes_ratio :
  let initial_volume : ℝ := 45
  let initial_milk_ratio : ℝ := 4
  let initial_water_ratio : ℝ := 1
  let added_water : ℝ := 21
  let final_ratio : ℝ := 1.2

  let initial_milk : ℝ := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water : ℝ := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let final_water : ℝ := initial_water + added_water
  let final_volume : ℝ := initial_volume + added_water

  initial_milk / final_water = final_ratio := by sorry

end NUMINAMATH_CALUDE_water_addition_changes_ratio_l615_61534


namespace NUMINAMATH_CALUDE_alternating_number_divisibility_l615_61579

/-- Represents a number in the form 1010101...0101 -/
def AlternatingNumber (n : ℕ) : ℕ := sorry

/-- The count of ones in an AlternatingNumber -/
def CountOnes (n : ℕ) : ℕ := sorry

theorem alternating_number_divisibility (n : ℕ) :
  (∃ k : ℕ, CountOnes n = 198 * k) ↔ AlternatingNumber n % 9999 = 0 := by sorry

end NUMINAMATH_CALUDE_alternating_number_divisibility_l615_61579


namespace NUMINAMATH_CALUDE_exactly_two_triples_l615_61556

/-- Least common multiple of two positive integers -/
def lcm (r s : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a,b,c) satisfying the given conditions -/
def count_triples : ℕ := sorry

/-- Theorem stating that there are exactly 2 ordered triples satisfying the conditions -/
theorem exactly_two_triples : 
  count_triples = 2 ∧ 
  ∀ a b c : ℕ+, 
    (lcm a b = 1250 ∧ lcm b c = 2500 ∧ lcm c a = 2500) → 
    (a, b, c) ∈ {x | count_triples > 0} :=
sorry

end NUMINAMATH_CALUDE_exactly_two_triples_l615_61556


namespace NUMINAMATH_CALUDE_x_value_when_z_64_l615_61598

/-- Given that x is directly proportional to y^4 and y is inversely proportional to z^2,
    prove that x = 1/4 when z = 64, given that x = 4 when z = 16. -/
theorem x_value_when_z_64 
  (h1 : ∃ (k₁ : ℝ), ∀ (x y : ℝ), x = k₁ * y^4)
  (h2 : ∃ (k₂ : ℝ), ∀ (y z : ℝ), y * z^2 = k₂)
  (h3 : ∃ (x y : ℝ), x = 4 ∧ y^4 = 1/16^4)
  : ∃ (x y : ℝ), x = 1/4 ∧ y^4 = 1/64^4 :=
sorry

end NUMINAMATH_CALUDE_x_value_when_z_64_l615_61598


namespace NUMINAMATH_CALUDE_problem_statement_l615_61588

theorem problem_statement :
  (¬ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 < 0) ∨ (∀ a b c : ℝ, b > c → a * b > a * c)) ∧
  (¬ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 < 0) ∧ ¬ (∀ a b c : ℝ, b > c → a * b > a * c)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l615_61588


namespace NUMINAMATH_CALUDE_matrix_fourth_power_l615_61525

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_fourth_power :
  A ^ 4 = !![(-4 : ℤ), 6; -6, 5] := by sorry

end NUMINAMATH_CALUDE_matrix_fourth_power_l615_61525


namespace NUMINAMATH_CALUDE_xyz_sum_sqrt_l615_61597

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 16) 
  (h2 : z + x = 18) 
  (h3 : x + y = 20) : 
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 18711 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_sqrt_l615_61597


namespace NUMINAMATH_CALUDE_apple_distribution_l615_61504

theorem apple_distribution (total_apples : ℕ) (num_people : ℕ) (apples_per_person : ℕ) :
  total_apples = 15 →
  num_people = 3 →
  apples_per_person * num_people ≤ total_apples →
  apples_per_person = total_apples / num_people →
  apples_per_person = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l615_61504


namespace NUMINAMATH_CALUDE_equation_proof_l615_61589

theorem equation_proof : 484 + 2 * 22 * 7 + 49 = 841 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l615_61589


namespace NUMINAMATH_CALUDE_pams_bank_account_l615_61590

def initial_balance : ℝ := 400
def withdrawal : ℝ := 250
def current_balance : ℝ := 950

theorem pams_bank_account :
  initial_balance * 3 - withdrawal = current_balance :=
by sorry

end NUMINAMATH_CALUDE_pams_bank_account_l615_61590


namespace NUMINAMATH_CALUDE_construct_equilateral_triangle_l615_61536

/-- A triangle with angles 80°, 50°, and 50° -/
structure DraftingTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_angles : angle1 + angle2 + angle3 = 180
  angle_values : angle1 = 80 ∧ angle2 = 50 ∧ angle3 = 50

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  all_sides_equal : side1 = side2 ∧ side2 = side3

/-- Theorem stating that an equilateral triangle can be constructed using the drafting triangle -/
theorem construct_equilateral_triangle (d : DraftingTriangle) : 
  ∃ (e : EquilateralTriangle), True := by sorry

end NUMINAMATH_CALUDE_construct_equilateral_triangle_l615_61536


namespace NUMINAMATH_CALUDE_smallest_batch_size_l615_61502

theorem smallest_batch_size (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ (21 * N)) : 
  (∀ M : ℕ, M > 70 ∧ 70 ∣ (21 * M) → N ≤ M) → N = 80 :=
by sorry

end NUMINAMATH_CALUDE_smallest_batch_size_l615_61502


namespace NUMINAMATH_CALUDE_julie_can_print_100_newspapers_l615_61578

/-- The number of boxes of paper Julie bought -/
def boxes : ℕ := 2

/-- The number of packages in each box -/
def packages_per_box : ℕ := 5

/-- The number of sheets in each package -/
def sheets_per_package : ℕ := 250

/-- The number of sheets required to print one newspaper -/
def sheets_per_newspaper : ℕ := 25

/-- The total number of newspapers Julie can print -/
def newspapers_printed : ℕ := 
  (boxes * packages_per_box * sheets_per_package) / sheets_per_newspaper

theorem julie_can_print_100_newspapers : newspapers_printed = 100 := by
  sorry

end NUMINAMATH_CALUDE_julie_can_print_100_newspapers_l615_61578


namespace NUMINAMATH_CALUDE_problem_solution_l615_61506

theorem problem_solution (a b m n : ℝ) : 
  (a + b - 1)^2 = -|a + 2| → mn = 1 → a^b + mn = -7 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l615_61506


namespace NUMINAMATH_CALUDE_rope_length_problem_l615_61599

theorem rope_length_problem (total_ropes : ℕ) (avg_length : ℝ) 
  (subset_ropes : ℕ) (subset_avg_length : ℝ) 
  (ratio_a ratio_b ratio_c : ℝ) :
  total_ropes = 9 →
  avg_length = 90 →
  subset_ropes = 3 →
  subset_avg_length = 70 →
  ratio_a = 2 ∧ ratio_b = 3 ∧ ratio_c = 5 →
  let remaining_ropes := total_ropes - subset_ropes
  let total_length := total_ropes * avg_length
  let subset_length := subset_ropes * subset_avg_length
  let remaining_length := total_length - subset_length
  (remaining_length / remaining_ropes : ℝ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_problem_l615_61599


namespace NUMINAMATH_CALUDE_chinese_character_equation_l615_61535

theorem chinese_character_equation : ∃! (math love i : ℕ),
  (math ≠ love ∧ math ≠ i ∧ love ≠ i) ∧
  (math > 0 ∧ love > 0 ∧ i > 0) ∧
  (math * (love * 1000 + math) = i * 1000 + love * 100 + math) ∧
  (math = 25 ∧ love = 125 ∧ i = 3) := by
  sorry

end NUMINAMATH_CALUDE_chinese_character_equation_l615_61535


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l615_61517

def U : Set Nat := {1,2,3,4,5,6}
def P : Set Nat := {1,3,5}
def Q : Set Nat := {1,2,4}

theorem complement_P_intersect_Q :
  (U \ P) ∩ Q = {2,4} := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l615_61517


namespace NUMINAMATH_CALUDE_star_equation_solution_l615_61547

/-- Custom operation  defined for integers -/
def star (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that if x star 9 = 160, then x = 21 -/
theorem star_equation_solution :
  ∀ x : ℤ, star x 9 = 160 → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l615_61547


namespace NUMINAMATH_CALUDE_intersecting_circles_radius_range_l615_61581

/-- Given two intersecting circles, prove that the radius of the first circle falls within a specific range -/
theorem intersecting_circles_radius_range :
  ∀ r : ℝ,
  r > 0 →
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ (x+3)^2 + (y-4)^2 = 36) →
  1 < r ∧ r < 11 := by
sorry

end NUMINAMATH_CALUDE_intersecting_circles_radius_range_l615_61581


namespace NUMINAMATH_CALUDE_no_triangle_with_given_conditions_l615_61501

theorem no_triangle_with_given_conditions (a b c : ℕ+) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (gcd_one : Nat.gcd a.val (Nat.gcd b.val c.val) = 1)
  (div_a : a.val ∣ (b.val - c.val)^2)
  (div_b : b.val ∣ (c.val - a.val)^2)
  (div_c : c.val ∣ (a.val - b.val)^2) :
  ¬(a.val < b.val + c.val ∧ b.val < a.val + c.val ∧ c.val < a.val + b.val) :=
sorry

end NUMINAMATH_CALUDE_no_triangle_with_given_conditions_l615_61501


namespace NUMINAMATH_CALUDE_f_properties_l615_61561

/-- The function f(x) = 2^x / (2^x + 1) + a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (2^x + 1) + a

/-- Main theorem about the properties of f -/
theorem f_properties (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ∧
  (∀ x : ℝ, f a (-x) = -(f a x) → a = -1/2) ∧
  (∀ x : ℝ, f a (-x) = -(f a x) → 
    (∀ x k : ℝ, f a (x^2 - 2*x) + f a (2*x^2 - k) > 0 → k < -1/3)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l615_61561


namespace NUMINAMATH_CALUDE_triangle_perimeter_with_tangent_circles_l615_61587

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle type
structure Triangle where
  vertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

-- Define a function to check if circles are tangent to each other
def areTangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define a function to check if a circle is tangent to two sides of a triangle
def isTangentToTriangleSides (c : Circle) (t : Triangle) : Prop :=
  sorry -- Implementation details omitted for brevity

-- Theorem statement
theorem triangle_perimeter_with_tangent_circles 
  (X Y Z : Circle) (DEF : Triangle) :
  X.radius = 2 ∧ Y.radius = 2 ∧ Z.radius = 2 →
  areTangent X Y ∧ areTangent Y Z ∧ areTangent Z X →
  isTangentToTriangleSides X DEF ∧ 
  isTangentToTriangleSides Y DEF ∧ 
  isTangentToTriangleSides Z DEF →
  let (D, E, F) := DEF.vertices
  let perimeter := Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) +
                   Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) +
                   Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  perimeter = 12 * Real.sqrt 3 :=
by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_triangle_perimeter_with_tangent_circles_l615_61587


namespace NUMINAMATH_CALUDE_cubic_expansion_equality_l615_61584

theorem cubic_expansion_equality : 27^3 + 9*(27^2) + 27*(9^2) + 9^3 = (27 + 9)^3 := by sorry

end NUMINAMATH_CALUDE_cubic_expansion_equality_l615_61584


namespace NUMINAMATH_CALUDE_square_root_calculations_l615_61566

theorem square_root_calculations :
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    Real.sqrt (a / b) * Real.sqrt c / Real.sqrt b = Real.sqrt (a * c / (b * b))) ∧
  (Real.sqrt (1 / 6) * Real.sqrt 96 / Real.sqrt 6 = 2 * Real.sqrt 6 / 3) ∧
  (Real.sqrt 80 - Real.sqrt 8 - Real.sqrt 45 + 4 * Real.sqrt (1 / 2) = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_square_root_calculations_l615_61566


namespace NUMINAMATH_CALUDE_complex_expression_equality_l615_61518

theorem complex_expression_equality : 
  (64 : ℝ) ^ (1/3) - 4 * Real.cos (45 * π / 180) + (1 - Real.sqrt 3) ^ 0 - abs (-Real.sqrt 2) = 5 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l615_61518


namespace NUMINAMATH_CALUDE_calculate_expression_l615_61530

theorem calculate_expression : 9^6 * 3^3 / 27^4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l615_61530


namespace NUMINAMATH_CALUDE_pizza_slices_l615_61572

theorem pizza_slices (coworkers : ℕ) (pizzas : ℕ) (slices_per_person : ℕ) :
  coworkers = 12 →
  pizzas = 3 →
  slices_per_person = 2 →
  (coworkers * slices_per_person) / pizzas = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l615_61572


namespace NUMINAMATH_CALUDE_weight_of_b_l615_61513

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) : 
  b = 31 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l615_61513


namespace NUMINAMATH_CALUDE_work_completion_proof_l615_61515

/-- The number of days it takes for b to complete the work alone -/
def b_days : ℝ := 26.25

/-- The number of days it takes for a to complete the work alone -/
def a_days : ℝ := 24

/-- The number of days it takes for c to complete the work alone -/
def c_days : ℝ := 40

/-- The total number of days it took to complete the work -/
def total_days : ℝ := 11

/-- The number of days c worked before leaving -/
def c_work_days : ℝ := total_days - 4

theorem work_completion_proof :
  7 * (1 / a_days + 1 / b_days + 1 / c_days) + 4 * (1 / a_days + 1 / b_days) = 1 := by
  sorry

#check work_completion_proof

end NUMINAMATH_CALUDE_work_completion_proof_l615_61515


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l615_61519

/-- Given a curve y = x^3 - 2x + 1 and a point (-1, 2) on this curve,
    if the tangent line at this point is perpendicular to the line ax + y + 1 = 0,
    then a = 1. -/
theorem tangent_line_perpendicular (a : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 2*x + 1
  let point : ℝ × ℝ := (-1, 2)
  let tangent_slope : ℝ := (deriv f) point.1
  let perpendicular_line : ℝ → ℝ := λ x => -a*x - 1
  f point.1 = point.2 ∧ 
  tangent_slope * (perpendicular_line point.1 - perpendicular_line (-1)) / (point.1 - (-1)) = -1 
  → a = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l615_61519
