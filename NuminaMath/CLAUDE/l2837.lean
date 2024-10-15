import Mathlib

namespace NUMINAMATH_CALUDE_fraction_problem_l2837_283776

theorem fraction_problem (x : ℚ) : 
  (5 / 6 : ℚ) * 576 = x * 576 + 300 → x = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2837_283776


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2837_283730

theorem inequality_solution_set :
  {x : ℝ | (1 : ℝ) / x < (1 : ℝ) / 2} = {x : ℝ | x < 0 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2837_283730


namespace NUMINAMATH_CALUDE_no_integer_solution_l2837_283751

theorem no_integer_solution :
  ¬ ∃ (m n k : ℕ+), ∀ (x y : ℝ),
    (x + 1)^2 + y^2 = (m : ℝ)^2 ∧
    (x - 1)^2 + y^2 = (n : ℝ)^2 ∧
    x^2 + (y - Real.sqrt 3)^2 = (k : ℝ)^2 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2837_283751


namespace NUMINAMATH_CALUDE_determine_down_speed_man_down_speed_l2837_283704

/-- The speed of a man traveling up and down a hill -/
structure TravelSpeed where
  up : ℝ
  down : ℝ
  average : ℝ

/-- Theorem stating that given the up speed and average speed, we can determine the down speed -/
theorem determine_down_speed (s : TravelSpeed) (h1 : s.up = 24) (h2 : s.average = 28.8) :
  s.down = 36 := by
  sorry

/-- Main theorem proving the specific case in the problem -/
theorem man_down_speed :
  ∃ s : TravelSpeed, s.up = 24 ∧ s.average = 28.8 ∧ s.down = 36 := by
  sorry

end NUMINAMATH_CALUDE_determine_down_speed_man_down_speed_l2837_283704


namespace NUMINAMATH_CALUDE_divisors_540_multiple_of_two_l2837_283717

/-- The number of positive divisors of 540 that are multiples of 2 -/
def divisors_multiple_of_two (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d > 0 ∧ n % d = 0 ∧ d % 2 = 0) (Finset.range (n + 1))).card

/-- Theorem stating that the number of positive divisors of 540 that are multiples of 2 is 16 -/
theorem divisors_540_multiple_of_two :
  divisors_multiple_of_two 540 = 16 := by
  sorry

end NUMINAMATH_CALUDE_divisors_540_multiple_of_two_l2837_283717


namespace NUMINAMATH_CALUDE_smallest_winning_k_l2837_283766

/-- Represents the game board -/
def Board := Fin 8 → Fin 8 → Option Char

/-- Checks if a sequence "HMM" or "MMH" exists horizontally or vertically -/
def winning_sequence (board : Board) : Prop :=
  ∃ (i j : Fin 8), 
    (board i j = some 'H' ∧ board i (j+1) = some 'M' ∧ board i (j+2) = some 'M') ∨
    (board i j = some 'M' ∧ board i (j+1) = some 'M' ∧ board i (j+2) = some 'H') ∨
    (board i j = some 'H' ∧ board (i+1) j = some 'M' ∧ board (i+2) j = some 'M') ∨
    (board i j = some 'M' ∧ board (i+1) j = some 'M' ∧ board (i+2) j = some 'H')

/-- Mike's strategy for placing 'M's -/
def mike_strategy (k : ℕ) : Board := sorry

/-- Harry's strategy for placing 'H's -/
def harry_strategy (k : ℕ) (mike_board : Board) : Board := sorry

/-- The main theorem stating that 16 is the smallest k for which Mike has a winning strategy -/
theorem smallest_winning_k : 
  (∀ (k : ℕ), k < 16 → ∃ (harry_board : Board), 
    harry_board = harry_strategy k (mike_strategy k) ∧ ¬winning_sequence harry_board) ∧ 
  (∀ (harry_board : Board), 
    harry_board = harry_strategy 16 (mike_strategy 16) → winning_sequence harry_board) :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_k_l2837_283766


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l2837_283748

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l2837_283748


namespace NUMINAMATH_CALUDE_armband_break_even_l2837_283753

/-- The cost of an individual ticket in dollars -/
def individual_ticket_cost : ℚ := 3/4

/-- The cost of an armband in dollars -/
def armband_cost : ℚ := 15

/-- The number of rides at which the armband cost equals the individual ticket cost -/
def break_even_rides : ℕ := 20

theorem armband_break_even :
  (individual_ticket_cost * break_even_rides : ℚ) = armband_cost :=
sorry

end NUMINAMATH_CALUDE_armband_break_even_l2837_283753


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_length_l2837_283750

/-- A circle passes through points A(1, 3), B(4, 2), and C(1, -7). 
    The segment MN is formed by the intersection of this circle with the y-axis. -/
theorem circle_y_axis_intersection_length :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    let circle := {(x, y) : ℝ × ℝ | (x - center.1)^2 + (y - center.2)^2 = radius^2}
    (1, 3) ∈ circle ∧ (4, 2) ∈ circle ∧ (1, -7) ∈ circle →
    let y_intersections := {y : ℝ | (0, y) ∈ circle}
    ∃ (m n : ℝ), m ∈ y_intersections ∧ n ∈ y_intersections ∧ m ≠ n ∧ 
    |m - n| = 4 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_length_l2837_283750


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l2837_283741

-- Define the number in base 5
def base_5_number : Nat := 200220220

-- Define the function to convert from base 5 to base 10
def base_5_to_10 (n : Nat) : Nat :=
  let digits := n.digits 5
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (5^i)) 0

-- Define the number in base 10
def number : Nat := base_5_to_10 base_5_number

-- Statement to prove
theorem largest_prime_divisor :
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ number ∧ ∀ (q : Nat), Nat.Prime q → q ∣ number → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l2837_283741


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l2837_283731

/-- The line equation kx - y - 2k = 0 -/
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y - 2 * k = 0

/-- The hyperbola equation x^2 - y^2 = 2 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

/-- The asymptotes of the hyperbola -/
def asymptotes (x y : ℝ) : Prop := y = x ∨ y = -x

/-- The theorem stating that if the line and hyperbola have only one common point, then k = 1 or k = -1 -/
theorem line_hyperbola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, line k p.1 p.2 ∧ hyperbola p.1 p.2) → 
  k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l2837_283731


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2837_283743

/-- Given two arithmetic sequences {an} and {bn} with the specified conditions, 
    prove that a5 + b5 = 35 -/
theorem arithmetic_sequence_sum (a b : ℕ → ℕ) 
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)  -- a is an arithmetic sequence
  (h2 : ∀ n, b (n + 1) - b n = b 2 - b 1)  -- b is an arithmetic sequence
  (h3 : a 1 + b 1 = 7)                     -- first condition
  (h4 : a 3 + b 3 = 21)                    -- second condition
  : a 5 + b 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2837_283743


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2837_283784

/-- Given a line segment with midpoint (3, -3) and one endpoint (7, 4),
    prove that the other endpoint is (-1, -10). -/
theorem line_segment_endpoint
  (midpoint : ℝ × ℝ)
  (endpoint1 : ℝ × ℝ)
  (h_midpoint : midpoint = (3, -3))
  (h_endpoint1 : endpoint1 = (7, 4)) :
  ∃ (endpoint2 : ℝ × ℝ),
    endpoint2 = (-1, -10) ∧
    midpoint = (
      (endpoint1.1 + endpoint2.1) / 2,
      (endpoint1.2 + endpoint2.2) / 2
    ) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2837_283784


namespace NUMINAMATH_CALUDE_pinecrest_academy_ratio_l2837_283756

theorem pinecrest_academy_ratio (j s : ℕ) (h1 : 3 * s = 6 * j) : s / j = 1 / 2 := by
  sorry

#check pinecrest_academy_ratio

end NUMINAMATH_CALUDE_pinecrest_academy_ratio_l2837_283756


namespace NUMINAMATH_CALUDE_expand_expression_l2837_283749

theorem expand_expression (x y z : ℝ) : 
  (2*x + 5) * (3*y + 15 + 4*z) = 6*x*y + 30*x + 8*x*z + 15*y + 20*z + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2837_283749


namespace NUMINAMATH_CALUDE_ned_garage_sale_games_l2837_283791

/-- The number of games Ned bought from a friend -/
def games_from_friend : ℕ := 50

/-- The number of games that didn't work -/
def bad_games : ℕ := 74

/-- The number of good games Ned ended up with -/
def good_games : ℕ := 3

/-- The number of games Ned bought at the garage sale -/
def games_from_garage_sale : ℕ := (good_games + bad_games) - games_from_friend

theorem ned_garage_sale_games :
  games_from_garage_sale = 27 := by sorry

end NUMINAMATH_CALUDE_ned_garage_sale_games_l2837_283791


namespace NUMINAMATH_CALUDE_solve_journey_l2837_283725

def journey_problem (total_distance : ℝ) (cycling_speed : ℝ) (walking_speed : ℝ) (total_time : ℝ) : Prop :=
  let cycling_distance : ℝ := (2/3) * total_distance
  let walking_distance : ℝ := total_distance - cycling_distance
  let cycling_time : ℝ := cycling_distance / cycling_speed
  let walking_time : ℝ := walking_distance / walking_speed
  (cycling_time + walking_time = total_time) → (walking_distance = 6)

theorem solve_journey :
  journey_problem 18 20 4 (70/60) := by
  sorry

end NUMINAMATH_CALUDE_solve_journey_l2837_283725


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2837_283709

/-- Proves that the cost price of one metre of cloth is 66.25,
    given the selling conditions of a cloth trader. -/
theorem cloth_cost_price
  (meters_sold : ℕ)
  (selling_price : ℚ)
  (profit_per_meter : ℚ)
  (h_meters : meters_sold = 80)
  (h_price : selling_price = 6900)
  (h_profit : profit_per_meter = 20) :
  (selling_price - meters_sold * profit_per_meter) / meters_sold = 66.25 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l2837_283709


namespace NUMINAMATH_CALUDE_linear_function_difference_l2837_283755

-- Define the properties of the linear function g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, ∃ a b : ℝ, g x = a * x + b) ∧ 
  (∀ d : ℝ, g (d + 2) - g d = 4)

-- State the theorem
theorem linear_function_difference 
  (g : ℝ → ℝ) 
  (h : g_properties g) : 
  g 4 - g 8 = -8 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_difference_l2837_283755


namespace NUMINAMATH_CALUDE_system_solution_l2837_283789

theorem system_solution (x y : ℝ) (m n : ℤ) : 
  (4 * (Real.cos x)^2 * (Real.sin (x/6))^2 + 4 * Real.sin (x/6) - 4 * (Real.sin x)^2 * Real.sin (x/6) + 1 = 0 ∧
   Real.sin (x/4) = Real.sqrt (Real.cos y)) ↔ 
  ((x = 11 * Real.pi + 24 * Real.pi * ↑m ∧ (y = Real.pi/3 + 2 * Real.pi * ↑n ∨ y = -Real.pi/3 + 2 * Real.pi * ↑n)) ∨
   (x = -5 * Real.pi + 24 * Real.pi * ↑m ∧ (y = Real.pi/3 + 2 * Real.pi * ↑n ∨ y = -Real.pi/3 + 2 * Real.pi * ↑n))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2837_283789


namespace NUMINAMATH_CALUDE_reappearance_is_lcm_reappearance_is_twenty_l2837_283721

/-- The cycle length of the letter sequence -/
def letter_cycle_length : ℕ := 5

/-- The cycle length of the digit sequence -/
def digit_cycle_length : ℕ := 4

/-- The line number where the original sequences reappear together -/
def reappearance_line : ℕ := 20

/-- Theorem stating that the reappearance line is the LCM of the cycle lengths -/
theorem reappearance_is_lcm :
  reappearance_line = Nat.lcm letter_cycle_length digit_cycle_length := by
  sorry

/-- Theorem stating that the reappearance line is 20 -/
theorem reappearance_is_twenty : reappearance_line = 20 := by
  sorry

end NUMINAMATH_CALUDE_reappearance_is_lcm_reappearance_is_twenty_l2837_283721


namespace NUMINAMATH_CALUDE_permutations_of_six_objects_l2837_283713

theorem permutations_of_six_objects : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_objects_l2837_283713


namespace NUMINAMATH_CALUDE_quadratic_root_proof_l2837_283712

theorem quadratic_root_proof (v : ℝ) : 
  v = 7 → (5 * (((-21 - Real.sqrt 301) / 10) ^ 2) + 21 * ((-21 - Real.sqrt 301) / 10) + v = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_proof_l2837_283712


namespace NUMINAMATH_CALUDE_expression_not_simplifiable_to_AD_l2837_283793

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (A B C D M : V)

theorem expression_not_simplifiable_to_AD :
  ∃ (BM DA MB : V), -BM - DA + MB ≠ (A - D) :=
by sorry

end NUMINAMATH_CALUDE_expression_not_simplifiable_to_AD_l2837_283793


namespace NUMINAMATH_CALUDE_base_6_addition_l2837_283724

/-- Converts a base-6 number to base-10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base-10 number to base-6 --/
def to_base_6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

theorem base_6_addition :
  to_base_6 (to_base_10 [4, 2, 5, 3] + to_base_10 [2, 4, 4, 2]) = [0, 1, 4, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_6_addition_l2837_283724


namespace NUMINAMATH_CALUDE_taxi_fare_equality_l2837_283767

/-- Taxi fare calculation and comparison -/
theorem taxi_fare_equality (mike_miles : ℝ) : 
  (2.5 + 0.25 * mike_miles = 2.5 + 5 + 0.25 * 26) → mike_miles = 46 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_equality_l2837_283767


namespace NUMINAMATH_CALUDE_jar_balls_count_l2837_283740

theorem jar_balls_count (initial_blue : ℕ) (removed : ℕ) (prob : ℚ) :
  initial_blue = 6 →
  removed = 3 →
  prob = 1/5 →
  (initial_blue - removed : ℚ) / ((initial_blue - removed : ℚ) + (18 - initial_blue : ℚ)) = prob →
  18 = initial_blue + (18 - initial_blue) :=
by sorry

end NUMINAMATH_CALUDE_jar_balls_count_l2837_283740


namespace NUMINAMATH_CALUDE_coordinate_axes_characterization_l2837_283738

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

end NUMINAMATH_CALUDE_coordinate_axes_characterization_l2837_283738


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2837_283772

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 3 = 2) →
  (a 3 + a 5 = 4) →
  (a 5 + a 7 = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2837_283772


namespace NUMINAMATH_CALUDE_two_books_different_genres_l2837_283733

theorem two_books_different_genres (n : ℕ) (h : n = 4) : 
  (n.choose 2) * n * n = 96 :=
by sorry

end NUMINAMATH_CALUDE_two_books_different_genres_l2837_283733


namespace NUMINAMATH_CALUDE_cost_difference_l2837_283754

-- Define the monthly costs
def rental_cost : ℕ := 20
def new_car_cost : ℕ := 30

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Define the total costs for a year
def total_rental_cost : ℕ := rental_cost * months_in_year
def total_new_car_cost : ℕ := new_car_cost * months_in_year

-- Theorem statement
theorem cost_difference :
  total_new_car_cost - total_rental_cost = 120 :=
by sorry

end NUMINAMATH_CALUDE_cost_difference_l2837_283754


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l2837_283757

theorem complex_pure_imaginary (a : ℝ) : 
  (a^2 - 3*a + 2 : ℂ) + (a - 1 : ℂ) * Complex.I = Complex.I * (b : ℝ) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l2837_283757


namespace NUMINAMATH_CALUDE_curve_to_line_equation_l2837_283727

theorem curve_to_line_equation (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 * t + 5) (h2 : y = 5 * t - 3) : 
  y = (5 * x - 34) / 3 := by
  sorry

end NUMINAMATH_CALUDE_curve_to_line_equation_l2837_283727


namespace NUMINAMATH_CALUDE_rachel_cookies_l2837_283781

theorem rachel_cookies (mona jasmine rachel : ℕ) : 
  mona = 20 →
  jasmine = mona - 5 →
  rachel > jasmine →
  mona + jasmine + rachel = 60 →
  rachel = 25 := by
sorry

end NUMINAMATH_CALUDE_rachel_cookies_l2837_283781


namespace NUMINAMATH_CALUDE_at_least_one_product_leq_one_l2837_283742

theorem at_least_one_product_leq_one (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_product_leq_one_l2837_283742


namespace NUMINAMATH_CALUDE_min_socks_for_pairs_l2837_283783

/-- Represents the number of colors of socks in the drawer -/
def num_colors : ℕ := 4

/-- Represents the number of pairs we want to guarantee -/
def required_pairs : ℕ := 10

/-- Theorem: The minimum number of socks to guarantee the required pairs -/
theorem min_socks_for_pairs :
  ∀ (sock_counts : Fin num_colors → ℕ),
  (∀ i, sock_counts i > 0) →
  ∃ (n : ℕ),
    n = num_colors + 2 * required_pairs ∧
    ∀ (m : ℕ), m ≥ n →
      ∀ (selection : Fin m → Fin num_colors),
      ∃ (pairs : Fin required_pairs → Fin m × Fin m),
        ∀ i, 
          (pairs i).1 < (pairs i).2 ∧
          selection (pairs i).1 = selection (pairs i).2 ∧
          ∀ j, i ≠ j → 
            ({(pairs i).1, (pairs i).2} : Set (Fin m)) ∩ {(pairs j).1, (pairs j).2} = ∅ :=
by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_pairs_l2837_283783


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l2837_283770

theorem difference_c_minus_a (a b c d k : ℝ) : 
  (a + b) / 2 = 45 →
  (b + c) / 2 = 50 →
  (a + c + d) / 3 = 60 →
  a^2 + b^2 + c^2 + d^2 = k →
  c - a = 10 := by
sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l2837_283770


namespace NUMINAMATH_CALUDE_marble_arrangements_mod_1000_l2837_283786

/-- The number of blue marbles --/
def blue_marbles : ℕ := 6

/-- The maximum number of yellow marbles that maintains the balance --/
def yellow_marbles : ℕ := 18

/-- The total number of marbles --/
def total_marbles : ℕ := blue_marbles + yellow_marbles

/-- The number of different arrangements --/
def arrangements : ℕ := Nat.choose total_marbles blue_marbles

theorem marble_arrangements_mod_1000 :
  arrangements % 1000 = 564 := by sorry

end NUMINAMATH_CALUDE_marble_arrangements_mod_1000_l2837_283786


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l2837_283711

theorem no_solution_fractional_equation :
  ¬∃ (x : ℝ), (x - 2) / (2 * x - 1) + 1 = 3 / (2 - 4 * x) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l2837_283711


namespace NUMINAMATH_CALUDE_equal_angles_not_always_vertical_l2837_283700

-- Define the concept of an angle
def Angle : Type := ℝ

-- Define the concept of vertical angles
def are_vertical_angles (a b : Angle) : Prop := sorry

-- Define the equality of angles
def angle_equal (a b : Angle) : Prop := a = b

-- Theorem stating that equal angles are not necessarily vertical angles
theorem equal_angles_not_always_vertical :
  ∃ (a b : Angle), angle_equal a b ∧ ¬(are_vertical_angles a b) := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_not_always_vertical_l2837_283700


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l2837_283746

theorem factorization_of_quadratic (a : ℝ) : a^2 + 2*a = a*(a + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l2837_283746


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_mean_of_fifteen_numbers_l2837_283735

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) 
                                  (set2_count : ℕ) (set2_mean : ℚ) : ℚ :=
  let total_count := set1_count + set2_count
  let combined_sum := set1_count * set1_mean + set2_count * set2_mean
  combined_sum / total_count

theorem mean_of_fifteen_numbers : 
  combined_mean_of_two_sets 7 15 8 22 = 281 / 15 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_mean_of_fifteen_numbers_l2837_283735


namespace NUMINAMATH_CALUDE_even_painted_faces_count_l2837_283798

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with an even number of painted faces in a block -/
def countEvenPaintedFaces (b : Block) : ℕ :=
  sorry

/-- The main theorem stating that a 6x3x2 block has 16 cubes with even number of painted faces -/
theorem even_painted_faces_count : 
  let b : Block := { length := 6, width := 3, height := 2 }
  countEvenPaintedFaces b = 16 := by
  sorry

end NUMINAMATH_CALUDE_even_painted_faces_count_l2837_283798


namespace NUMINAMATH_CALUDE_min_value_of_f_l2837_283799

/-- The function f(x) = 3x^2 - 18x + 2023 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2023

theorem min_value_of_f :
  ∃ (m : ℝ), m = 1996 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2837_283799


namespace NUMINAMATH_CALUDE_no_upper_bound_expression_l2837_283726

/-- The expression has no upper bound -/
theorem no_upper_bound_expression (a b c d : ℝ) (h : a * d - b * c = 1) :
  ∀ M : ℝ, ∃ a' b' c' d' : ℝ, 
    a' * d' - b' * c' = 1 ∧ 
    a'^2 + b'^2 + c'^2 + d'^2 + a' * b' + c' * d' > M :=
by sorry

end NUMINAMATH_CALUDE_no_upper_bound_expression_l2837_283726


namespace NUMINAMATH_CALUDE_coordinates_of_B_l2837_283714

/-- Given a line segment AB parallel to the y-axis, with A(1, -2) and AB = 8,
    the coordinates of B are either (1, -10) or (1, 6). -/
theorem coordinates_of_B (A B : ℝ × ℝ) : 
  A = (1, -2) →
  (B.1 = A.1) →
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 8 →
  (B = (1, -10) ∨ B = (1, 6)) :=
by sorry

end NUMINAMATH_CALUDE_coordinates_of_B_l2837_283714


namespace NUMINAMATH_CALUDE_inequality_solution_l2837_283745

theorem inequality_solution (x : ℝ) : 
  (x * (x - 1)) / ((x - 5)^2) ≥ 15 ↔ 
  (x ≤ 4.09 ∨ x ≥ 6.56) ∧ x ≠ 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2837_283745


namespace NUMINAMATH_CALUDE_unique_k_for_pythagorean_like_equation_l2837_283707

theorem unique_k_for_pythagorean_like_equation :
  ∃! k : ℕ+, ∃ a b : ℕ+, a^2 + b^2 = k * a * b := by sorry

end NUMINAMATH_CALUDE_unique_k_for_pythagorean_like_equation_l2837_283707


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l2837_283797

theorem min_value_of_expression (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by sorry

theorem min_value_attained : 
  ∃ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l2837_283797


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2837_283785

theorem geometric_series_sum : 
  let a : ℚ := 1/3
  let r : ℚ := -1/4
  let n : ℕ := 6
  let series_sum : ℚ := a * (1 - r^n) / (1 - r)
  series_sum = 4095/30720 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2837_283785


namespace NUMINAMATH_CALUDE_inequality_proof_l2837_283759

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b = 3 + b - a) : (3 / b) + (1 / a) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2837_283759


namespace NUMINAMATH_CALUDE_f_and_g_odd_and_increasing_l2837_283728

-- Define the functions
def f (x : ℝ) := x * |x|
def g (x : ℝ) := x^3

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Theorem statement
theorem f_and_g_odd_and_increasing :
  (is_odd f ∧ is_increasing f) ∧ (is_odd g ∧ is_increasing g) :=
sorry

end NUMINAMATH_CALUDE_f_and_g_odd_and_increasing_l2837_283728


namespace NUMINAMATH_CALUDE_vitamin_boxes_count_l2837_283715

/-- Given the total number of medicine boxes and the number of supplement boxes,
    prove that the number of vitamin boxes is 472. -/
theorem vitamin_boxes_count (total_medicine : ℕ) (supplements : ℕ) 
    (h1 : total_medicine = 760)
    (h2 : supplements = 288)
    (h3 : ∃ vitamins : ℕ, total_medicine = vitamins + supplements) :
  ∃ vitamins : ℕ, vitamins = 472 ∧ total_medicine = vitamins + supplements :=
by
  sorry

end NUMINAMATH_CALUDE_vitamin_boxes_count_l2837_283715


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2837_283732

theorem quadratic_equal_roots (a b : ℝ) (h : b^2 = 4*a) :
  (a * b^2) / (a^2 - 4*a + b^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2837_283732


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l2837_283736

/-- The product of fractions from 10/5 to 2520/2515 -/
def fraction_product : ℕ → ℚ
  | 0 => 2 -- 10/5
  | n + 1 => fraction_product n * ((5 * (n + 2)) / (5 * (n + 1)))

theorem fraction_product_simplification :
  fraction_product 502 = 504 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l2837_283736


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2837_283723

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (2 * a^3 - 3 * a^2 + 165 * a - 4 = 0) →
  (2 * b^3 - 3 * b^2 + 165 * b - 4 = 0) →
  (2 * c^3 - 3 * c^2 + 165 * c - 4 = 0) →
  (a + b - 1)^3 + (b + c - 1)^3 + (c + a - 1)^3 = 117 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l2837_283723


namespace NUMINAMATH_CALUDE_taxi_charge_per_segment_l2837_283729

/-- Calculates the additional charge per 2/5 of a mile for a taxi service -/
theorem taxi_charge_per_segment (initial_fee : ℚ) (total_distance : ℚ) (total_charge : ℚ) :
  initial_fee = 2.25 →
  total_distance = 3.6 →
  total_charge = 3.60 →
  (total_charge - initial_fee) / (total_distance / (2/5)) = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_per_segment_l2837_283729


namespace NUMINAMATH_CALUDE_square_area_from_corners_l2837_283705

/-- The area of a square with adjacent corners at (1, 2) and (-2, 2) is 9 -/
theorem square_area_from_corners : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (-2, 2)
  let side_length := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let area := side_length^2
  area = 9 := by sorry

end NUMINAMATH_CALUDE_square_area_from_corners_l2837_283705


namespace NUMINAMATH_CALUDE_stone_volume_l2837_283780

/-- Represents a rectangular cuboid bowl -/
structure Bowl where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the volume of water in the bowl given its height -/
def water_volume (b : Bowl) (water_height : ℝ) : ℝ :=
  b.width * b.length * water_height

theorem stone_volume (b : Bowl) (initial_water_height final_water_height : ℝ) :
  b.width = 16 →
  b.length = 14 →
  b.height = 9 →
  initial_water_height = 4 →
  final_water_height = 9 →
  water_volume b final_water_height - water_volume b initial_water_height = 1120 :=
by sorry

end NUMINAMATH_CALUDE_stone_volume_l2837_283780


namespace NUMINAMATH_CALUDE_sum_of_a_and_a1_is_nine_l2837_283762

theorem sum_of_a_and_a1_is_nine (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x + 1)^2 + (x + 1)^11 = a + a₁*(x + 2) + a₂*(x + 2)^2 + a₃*(x + 2)^3 + 
   a₄*(x + 2)^4 + a₅*(x + 2)^5 + a₆*(x + 2)^6 + a₇*(x + 2)^7 + a₈*(x + 2)^8 + 
   a₉*(x + 2)^9 + a₁₀*(x + 2)^10 + a₁₁*(x + 2)^11) →
  a + a₁ = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_a1_is_nine_l2837_283762


namespace NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_percentage_increase_l2837_283706

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l) * (1.2 * w) = 1.56 * (l * w) := by
  sorry

theorem rectangle_area_percentage_increase :
  (1.56 - 1) * 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_percentage_increase_l2837_283706


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l2837_283708

theorem arithmetic_sequence_count : 
  ∀ (a d last : ℕ) (n : ℕ),
    a = 2 →
    d = 4 →
    last = 2018 →
    last = a + (n - 1) * d →
    n = 505 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l2837_283708


namespace NUMINAMATH_CALUDE_catherine_stationery_l2837_283771

theorem catherine_stationery (initial_pens initial_pencils pens_given pencils_given remaining_pens remaining_pencils : ℕ) :
  initial_pens = initial_pencils →
  pens_given = 36 →
  pencils_given = 16 →
  remaining_pens = 36 →
  remaining_pencils = 28 →
  initial_pens - pens_given = remaining_pens →
  initial_pencils - pencils_given = remaining_pencils →
  initial_pens = 72 ∧ initial_pencils = 72 := by
sorry

end NUMINAMATH_CALUDE_catherine_stationery_l2837_283771


namespace NUMINAMATH_CALUDE_angle_sum_in_triangle_l2837_283790

theorem angle_sum_in_triangle (A B C : ℝ) : 
  A + B + C = 180 →
  A + B = 150 →
  C = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_triangle_l2837_283790


namespace NUMINAMATH_CALUDE_area_YPW_is_8_l2837_283747

/-- Represents a rectangle XYZW with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a point P that divides the diagonal XW of a rectangle -/
structure DiagonalPoint where
  ratio_XP : ℝ
  ratio_PW : ℝ

/-- Calculates the area of triangle YPW in the given rectangle with the given diagonal point -/
def area_YPW (rect : Rectangle) (p : DiagonalPoint) : ℝ :=
  sorry

/-- Theorem stating that for a rectangle with length 8 and width 6, 
    if P divides XW in ratio 2:1, then area of YPW is 8 -/
theorem area_YPW_is_8 (rect : Rectangle) (p : DiagonalPoint) :
  rect.length = 8 →
  rect.width = 6 →
  p.ratio_XP = 2 →
  p.ratio_PW = 1 →
  area_YPW rect p = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_YPW_is_8_l2837_283747


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l2837_283719

theorem smallest_x_for_equation : 
  ∃ (x : ℝ), x ≠ 6 ∧ x ≠ -4 ∧
  (x^2 - 3*x - 18) / (x - 6) = 5 / (x + 4) ∧
  ∀ (y : ℝ), y ≠ 6 ∧ y ≠ -4 ∧ (y^2 - 3*y - 18) / (y - 6) = 5 / (y + 4) → x ≤ y ∧
  x = (-7 - Real.sqrt 21) / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l2837_283719


namespace NUMINAMATH_CALUDE_semicircle_radius_l2837_283702

/-- The radius of a semi-circle with perimeter 198 cm is 198 / (π + 2) cm. -/
theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 198) : 
  perimeter / (Real.pi + 2) = 198 / (Real.pi + 2) := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l2837_283702


namespace NUMINAMATH_CALUDE_parallelogram_height_l2837_283701

theorem parallelogram_height (area base height : ℝ) : 
  area = 120 ∧ base = 12 ∧ area = base * height → height = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2837_283701


namespace NUMINAMATH_CALUDE_oak_trees_after_planting_l2837_283778

/-- The number of oak trees in the park after planting -/
def total_oak_trees (initial : ℕ) (new : ℕ) : ℕ :=
  initial + new

/-- Theorem: The total number of oak trees after planting is 11 -/
theorem oak_trees_after_planting :
  total_oak_trees 9 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_after_planting_l2837_283778


namespace NUMINAMATH_CALUDE_constant_path_mapping_l2837_283734

/-- Given two segments AB and A'B' with their respective midpoints D and D', 
    prove that for any point P on AB with distance x from D, 
    and its associated point P' on A'B' with distance y from D', x + y = 6.5 -/
theorem constant_path_mapping (AB A'B' : ℝ) (D D' x y : ℝ) : 
  AB = 5 →
  A'B' = 8 →
  D = AB / 2 →
  D' = A'B' / 2 →
  x + y + D + D' = AB + A'B' →
  x + y = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_constant_path_mapping_l2837_283734


namespace NUMINAMATH_CALUDE_sum_of_digits_of_k_l2837_283768

def k : ℕ := 10^30 - 36

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_k : sum_of_digits k = 262 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_k_l2837_283768


namespace NUMINAMATH_CALUDE_ant_problem_l2837_283795

theorem ant_problem (abe_ants cece_ants duke_ants beth_ants : ℕ) 
  (total_ants : ℕ) (beth_percentage : ℚ) :
  abe_ants = 4 →
  cece_ants = 2 * abe_ants →
  duke_ants = abe_ants / 2 →
  beth_ants = abe_ants + (beth_percentage / 100) * abe_ants →
  total_ants = abe_ants + beth_ants + cece_ants + duke_ants →
  total_ants = 20 →
  beth_percentage = 50 := by
sorry

end NUMINAMATH_CALUDE_ant_problem_l2837_283795


namespace NUMINAMATH_CALUDE_triangle_inequality_l2837_283760

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2837_283760


namespace NUMINAMATH_CALUDE_company_sugar_usage_l2837_283764

/-- The amount of sugar (in grams) used by a chocolate company in two minutes -/
def sugar_used_in_two_minutes (sugar_per_bar : ℝ) (bars_per_minute : ℝ) : ℝ :=
  2 * (sugar_per_bar * bars_per_minute)

/-- Theorem stating that the company uses 108 grams of sugar in two minutes -/
theorem company_sugar_usage :
  sugar_used_in_two_minutes 1.5 36 = 108 := by
  sorry

end NUMINAMATH_CALUDE_company_sugar_usage_l2837_283764


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l2837_283744

theorem stratified_sampling_male_count 
  (total_employees : ℕ) 
  (female_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 120) 
  (h2 : female_employees = 72) 
  (h3 : sample_size = 15) :
  (total_employees - female_employees) * sample_size / total_employees = 6 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l2837_283744


namespace NUMINAMATH_CALUDE_cube_root_equation_l2837_283758

theorem cube_root_equation (x : ℝ) : (9 * x + 8) ^ (1/3 : ℝ) = 4 → x = 56 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_l2837_283758


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_inequality_l2837_283794

theorem sqrt_equality_implies_inequality (x y α : ℝ) : 
  Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α) → x + y ≥ 2 * α := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_inequality_l2837_283794


namespace NUMINAMATH_CALUDE_average_age_combined_l2837_283718

theorem average_age_combined (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  num_students = 40 →
  num_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 40 →
  ((num_students : ℚ) * avg_age_students + (num_parents : ℚ) * avg_age_parents) / 
    ((num_students : ℚ) + (num_parents : ℚ)) = 28.8 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l2837_283718


namespace NUMINAMATH_CALUDE_value_of_a_l2837_283788

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {a, b, 2}
def B (a b : ℝ) : Set ℝ := {2, b^2, 2*a}

-- State the theorem
theorem value_of_a (a b : ℝ) :
  A a b = B a b → a = 0 ∨ a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2837_283788


namespace NUMINAMATH_CALUDE_ferry_distance_ratio_l2837_283752

/-- The ratio of the distance covered by ferry Q to the distance covered by ferry P -/
theorem ferry_distance_ratio :
  let speed_p : ℝ := 8
  let time_p : ℝ := 3
  let speed_q : ℝ := speed_p + 1
  let time_q : ℝ := time_p + 5
  let distance_p : ℝ := speed_p * time_p
  let distance_q : ℝ := speed_q * time_q
  (distance_q / distance_p : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ferry_distance_ratio_l2837_283752


namespace NUMINAMATH_CALUDE_proportional_function_expression_l2837_283716

/-- Given a proportional function y = kx (k ≠ 0), if y = 6 when x = 4, 
    then the function can be expressed as y = (3/2)x -/
theorem proportional_function_expression (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x y, y = k * x) → (6 : ℝ) = k * 4 → 
  ∀ x y, y = k * x ↔ y = (3/2) * x := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_expression_l2837_283716


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2837_283792

theorem sqrt_inequality (a : ℝ) : (0 < a ∧ a < 1) ↔ a < Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2837_283792


namespace NUMINAMATH_CALUDE_parabola_slope_l2837_283710

/-- The slope of line MF for a parabola y² = 2px with point M(3, m) at distance 4 from focus -/
theorem parabola_slope (p m : ℝ) : p > 0 → m > 0 → m^2 = 6*p → (3 + p/2)^2 + m^2 = 16 → 
  (m / (3 - p/2) : ℝ) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_slope_l2837_283710


namespace NUMINAMATH_CALUDE_polynomial_B_value_l2837_283720

def polynomial (z A B C D : ℤ) : ℤ := z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 144

def roots : List ℤ := [3, 3, 2, 2, 1, 1]

theorem polynomial_B_value :
  ∀ (A B C D : ℤ),
  (∀ r ∈ roots, polynomial r A B C D = 0) →
  B = -126 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_B_value_l2837_283720


namespace NUMINAMATH_CALUDE_rsa_factorization_l2837_283775

theorem rsa_factorization :
  ∃ (p q : ℕ), 
    400000001 = p * q ∧ 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p = 20201 ∧ 
    q = 19801 := by
  sorry

end NUMINAMATH_CALUDE_rsa_factorization_l2837_283775


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2837_283779

theorem polynomial_factorization (a b x : ℝ) : 
  a + (a+b)*x + (a+2*b)*x^2 + (a+3*b)*x^3 + 3*b*x^4 + 2*b*x^5 + b*x^6 = 
  (1 + x)*(1 + x^2)*(a + b*x + b*x^2 + b*x^3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2837_283779


namespace NUMINAMATH_CALUDE_sandwich_cost_l2837_283773

theorem sandwich_cost (num_sandwiches num_drinks drink_cost total_cost : ℕ) 
  (h1 : num_sandwiches = 3)
  (h2 : num_drinks = 2)
  (h3 : drink_cost = 4)
  (h4 : total_cost = 26) :
  ∃ (sandwich_cost : ℕ), 
    num_sandwiches * sandwich_cost + num_drinks * drink_cost = total_cost ∧ 
    sandwich_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_l2837_283773


namespace NUMINAMATH_CALUDE_root_difference_range_l2837_283763

noncomputable section

variables (a b c d : ℝ) (x₁ x₂ : ℝ)

def g (x : ℝ) := a * x^3 + b * x^2 + c * x + d

def f (x : ℝ) := 3 * a * x^2 + 2 * b * x + c

theorem root_difference_range (ha : a ≠ 0) 
  (h_sum : a + b + c = 0) 
  (h_prod : f 0 * f 1 > 0) 
  (h_roots : f x₁ = 0 ∧ f x₂ = 0) :
  ∃ (l u : ℝ), l = Real.sqrt 3 / 3 ∧ u = 2 / 3 ∧ 
  l ≤ |x₁ - x₂| ∧ |x₁ - x₂| < u :=
sorry

end NUMINAMATH_CALUDE_root_difference_range_l2837_283763


namespace NUMINAMATH_CALUDE_dinner_arrangements_l2837_283782

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- There are 5 people in the group -/
def total_people : ℕ := 5

/-- The number of people who cook -/
def cooks : ℕ := 2

theorem dinner_arrangements :
  choose total_people cooks = 10 := by
  sorry

end NUMINAMATH_CALUDE_dinner_arrangements_l2837_283782


namespace NUMINAMATH_CALUDE_x_is_25_percent_greater_than_88_l2837_283796

theorem x_is_25_percent_greater_than_88 (x : ℝ) : 
  x = 88 * (1 + 0.25) → x = 110 := by
  sorry

end NUMINAMATH_CALUDE_x_is_25_percent_greater_than_88_l2837_283796


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2837_283703

theorem polynomial_factorization (a x : ℝ) : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2837_283703


namespace NUMINAMATH_CALUDE_sqrt_ceil_floor_sum_l2837_283774

theorem sqrt_ceil_floor_sum : 
  ⌈Real.sqrt 50⌉ + ⌈Real.sqrt 150⌉ + ⌊Real.sqrt 350⌋ = 39 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ceil_floor_sum_l2837_283774


namespace NUMINAMATH_CALUDE_circle_tangent_ratio_l2837_283722

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the basic geometric relations
variable (on_circle : Point → Circle → Prop)
variable (inside_circle : Circle → Circle → Prop)
variable (concentric : Circle → Circle → Prop)
variable (tangent_to : Point → Point → Circle → Prop)
variable (intersects : Point → Point → Circle → Point → Prop)
variable (midpoint : Point → Point → Point → Prop)
variable (line_through : Point → Point → Point → Prop)
variable (perp_bisector : Point → Point → Point → Point → Prop)
variable (ratio : Point → Point → Point → ℚ → Prop)

-- State the theorem
theorem circle_tangent_ratio 
  (Γ₁ Γ₂ : Circle) 
  (A B C D E F M : Point) :
  concentric Γ₁ Γ₂ →
  inside_circle Γ₂ Γ₁ →
  on_circle A Γ₁ →
  on_circle B Γ₂ →
  tangent_to A B Γ₂ →
  intersects A B Γ₁ C →
  midpoint D A B →
  line_through A E F →
  on_circle E Γ₂ →
  on_circle F Γ₂ →
  perp_bisector D E M B →
  perp_bisector C F M B →
  ratio A M C (3/2) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_ratio_l2837_283722


namespace NUMINAMATH_CALUDE_positive_root_range_l2837_283739

theorem positive_root_range : ∃ x : ℝ, x^2 - 2*x - 1 = 0 ∧ x > 0 ∧ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_root_range_l2837_283739


namespace NUMINAMATH_CALUDE_original_number_proof_l2837_283777

theorem original_number_proof (x : ℝ) : 1 + 1/x = 8/3 → x = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2837_283777


namespace NUMINAMATH_CALUDE_range_of_a_l2837_283737

def P (a : ℝ) : Set ℝ := {x | a - 4 < x ∧ x < a + 4}
def Q : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Q, x ∈ P a) → -1 < a ∧ a < 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2837_283737


namespace NUMINAMATH_CALUDE_percentage_increase_l2837_283761

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 100 → final = 110 → (final - initial) / initial * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l2837_283761


namespace NUMINAMATH_CALUDE_xiao_ming_english_score_l2837_283765

/-- Calculates the weighted average score given three component scores and their weights -/
def weighted_average (listening_score language_score written_score : ℚ) 
  (listening_weight language_weight written_weight : ℕ) : ℚ :=
  (listening_score * listening_weight + language_score * language_weight + written_score * written_weight) / 
  (listening_weight + language_weight + written_weight)

/-- Theorem stating that Xiao Ming's English score is 92.6 given his component scores and the weighting ratio -/
theorem xiao_ming_english_score : 
  weighted_average 92 90 95 3 3 4 = 92.6 := by sorry

end NUMINAMATH_CALUDE_xiao_ming_english_score_l2837_283765


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_squared_l2837_283787

theorem square_plus_reciprocal_squared (x : ℝ) (h : x ≠ 0) :
  x^2 + 1/x^2 = 2 → x^4 + 1/x^4 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_squared_l2837_283787


namespace NUMINAMATH_CALUDE_difference_largest_smallest_n_l2837_283769

-- Define a convex n-gon
def ConvexNGon (n : ℕ) := n ≥ 3

-- Define an odd prime number
def OddPrime (p : ℕ) := Nat.Prime p ∧ p % 2 = 1

-- Define the condition that all interior angles are odd primes
def AllAnglesOddPrime (n : ℕ) (angles : Fin n → ℕ) :=
  ∀ i, OddPrime (angles i)

-- Define the sum of interior angles of an n-gon
def InteriorAngleSum (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the condition that the sum of angles equals the interior angle sum
def AnglesSumToInteriorSum (n : ℕ) (angles : Fin n → ℕ) :=
  (Finset.univ.sum angles) = InteriorAngleSum n

-- Main theorem
theorem difference_largest_smallest_n :
  ∃ (n_min n_max : ℕ),
    (ConvexNGon n_min ∧
     ∃ angles_min, AllAnglesOddPrime n_min angles_min ∧ AnglesSumToInteriorSum n_min angles_min) ∧
    (ConvexNGon n_max ∧
     ∃ angles_max, AllAnglesOddPrime n_max angles_max ∧ AnglesSumToInteriorSum n_max angles_max) ∧
    (∀ n, ConvexNGon n → 
      (∃ angles, AllAnglesOddPrime n angles ∧ AnglesSumToInteriorSum n angles) →
      n_min ≤ n ∧ n ≤ n_max) ∧
    n_max - n_min = 356 :=
sorry

end NUMINAMATH_CALUDE_difference_largest_smallest_n_l2837_283769
