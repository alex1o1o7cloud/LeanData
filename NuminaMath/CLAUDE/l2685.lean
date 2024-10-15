import Mathlib

namespace NUMINAMATH_CALUDE_expansion_distinct_terms_l2685_268596

/-- The number of distinct terms in the expansion of (x+y+z+w)(p+q+r+s+t) -/
def distinctTerms (x y z w p q r s t : ℝ) : ℕ :=
  4 * 5

theorem expansion_distinct_terms (x y z w p q r s t : ℝ) 
  (h : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
       p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t) :
  distinctTerms x y z w p q r s t = 20 := by
  sorry

end NUMINAMATH_CALUDE_expansion_distinct_terms_l2685_268596


namespace NUMINAMATH_CALUDE_total_cost_is_36_l2685_268519

-- Define the cost per dose for each antibiotic
def cost_a : ℚ := 3
def cost_b : ℚ := 4.5

-- Define the number of doses per week for each antibiotic
def doses_a : ℕ := 3 * 2  -- 3 days, twice a day
def doses_b : ℕ := 4 * 1  -- 4 days, once a day

-- Define the discount rate and the number of doses required for the discount
def discount_rate : ℚ := 0.2
def discount_doses : ℕ := 10

-- Define the total cost function
def total_cost : ℚ :=
  min (doses_a * cost_a) (discount_doses * cost_a * (1 - discount_rate)) +
  doses_b * cost_b

-- Theorem statement
theorem total_cost_is_36 : total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_36_l2685_268519


namespace NUMINAMATH_CALUDE_equation_solution_l2685_268560

theorem equation_solution : ∃! x : ℝ, (2 / (x - 1) = 3 / (x - 2)) ∧ (x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2685_268560


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2685_268563

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2685_268563


namespace NUMINAMATH_CALUDE_decimal_expression_simplification_l2685_268597

theorem decimal_expression_simplification :
  (0.00001 * (0.01)^2 * 1000) / 0.001 = 10^(-3) := by
  sorry

end NUMINAMATH_CALUDE_decimal_expression_simplification_l2685_268597


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2685_268568

theorem division_remainder_proof (dividend : Nat) (divisor : Nat) (quotient : Nat) 
    (h1 : dividend = 131)
    (h2 : divisor = 14)
    (h3 : quotient = 9)
    (h4 : dividend = divisor * quotient + (dividend % divisor)) :
  dividend % divisor = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2685_268568


namespace NUMINAMATH_CALUDE_cube_root_27_times_sixth_root_64_times_sqrt_9_l2685_268590

theorem cube_root_27_times_sixth_root_64_times_sqrt_9 :
  (27 : ℝ) ^ (1/3) * (64 : ℝ) ^ (1/6) * (9 : ℝ) ^ (1/2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_times_sixth_root_64_times_sqrt_9_l2685_268590


namespace NUMINAMATH_CALUDE_initial_interest_rate_is_45_percent_l2685_268558

/-- Given an initial deposit amount and two interest scenarios, 
    prove that the initial interest rate is 45% --/
theorem initial_interest_rate_is_45_percent 
  (P : ℝ) -- Principal amount (initial deposit)
  (r : ℝ) -- Initial interest rate (as a percentage)
  (h1 : P * r / 100 = 405) -- Interest at initial rate is 405
  (h2 : P * (r + 5) / 100 = 450) -- Interest at (r + 5)% is 450
  : r = 45 := by
sorry

end NUMINAMATH_CALUDE_initial_interest_rate_is_45_percent_l2685_268558


namespace NUMINAMATH_CALUDE_no_fib_rectangle_decomposition_l2685_268517

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- A square with side length that is a Fibonacci number -/
structure FibSquare where
  side : ℕ
  is_fib : ∃ n, fib n = side

/-- A rectangle composed of Fibonacci squares -/
structure FibRectangle where
  squares : List FibSquare
  different_sizes : ∀ i j, i ≠ j → (squares.get i).side ≠ (squares.get j).side
  at_least_two : squares.length ≥ 2

/-- The theorem stating that a rectangle cannot be composed of different-sized Fibonacci squares -/
theorem no_fib_rectangle_decomposition : ¬ ∃ (r : FibRectangle), True := by
  sorry

end NUMINAMATH_CALUDE_no_fib_rectangle_decomposition_l2685_268517


namespace NUMINAMATH_CALUDE_valid_schedules_l2685_268541

/-- Number of periods in a day -/
def total_periods : ℕ := 8

/-- Number of periods in the morning -/
def morning_periods : ℕ := 5

/-- Number of periods in the afternoon -/
def afternoon_periods : ℕ := 3

/-- Number of classes to teach -/
def classes_to_teach : ℕ := 3

/-- Calculate the number of ways to arrange n items taken k at a time -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of valid teaching schedules -/
theorem valid_schedules : 
  arrange total_periods classes_to_teach - 
  (morning_periods * arrange morning_periods classes_to_teach) - 
  arrange afternoon_periods classes_to_teach = 312 := by sorry

end NUMINAMATH_CALUDE_valid_schedules_l2685_268541


namespace NUMINAMATH_CALUDE_number_difference_l2685_268593

theorem number_difference (L S : ℕ) : L = 1495 → L = 5 * S + 4 → L - S = 1197 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2685_268593


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l2685_268579

def digit_sum (n : ℕ) : ℕ := sorry

def is_prime (n : ℕ) : Prop := sorry

theorem smallest_prime_with_digit_sum_23 :
  (∀ p : ℕ, is_prime p ∧ digit_sum p = 23 → p ≥ 599) ∧
  is_prime 599 ∧
  digit_sum 599 = 23 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l2685_268579


namespace NUMINAMATH_CALUDE_equilateral_triangle_minimum_rotation_angle_l2685_268520

/-- An equilateral triangle is a polygon with three equal sides and three equal angles. -/
structure EquilateralTriangle where
  -- We don't need to define the structure completely for this problem

/-- A figure is rotationally symmetric if it can be rotated around a fixed point by a certain angle
    and coincide with its initial position. -/
class RotationallySymmetric (α : Type*) where
  is_rotationally_symmetric : α → Prop

/-- The minimum rotation angle is the smallest non-zero angle by which a rotationally symmetric
    figure can be rotated to coincide with itself. -/
def minimum_rotation_angle (α : Type*) [RotationallySymmetric α] (figure : α) : ℝ :=
  sorry

theorem equilateral_triangle_minimum_rotation_angle 
  (triangle : EquilateralTriangle) 
  [RotationallySymmetric EquilateralTriangle] 
  (h : RotationallySymmetric.is_rotationally_symmetric triangle) : 
  minimum_rotation_angle EquilateralTriangle triangle = 120 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_minimum_rotation_angle_l2685_268520


namespace NUMINAMATH_CALUDE_bubble_radius_l2685_268574

/-- Given a hemisphere with radius 4∛2 cm that has the same volume as a spherical bubble,
    the radius of the original bubble is 4 cm. -/
theorem bubble_radius (r : ℝ) (R : ℝ) : 
  r = 4 * Real.rpow 2 (1/3) → -- radius of hemisphere
  (2/3) * Real.pi * r^3 = (4/3) * Real.pi * R^3 → -- volume equality
  R = 4 := by
sorry

end NUMINAMATH_CALUDE_bubble_radius_l2685_268574


namespace NUMINAMATH_CALUDE_divisibility_from_point_distribution_l2685_268506

theorem divisibility_from_point_distribution (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_k_le_n : k ≤ n)
  (points : Finset ℝ) (h_card : points.card = n)
  (h_divisible : ∀ x ∈ points, (points.filter (λ y => |y - x| ≤ 1)).card % k = 0) :
  k ∣ n := by
sorry

end NUMINAMATH_CALUDE_divisibility_from_point_distribution_l2685_268506


namespace NUMINAMATH_CALUDE_test_questions_l2685_268576

theorem test_questions (total_questions : ℕ) 
  (h1 : total_questions / 2 = (13 : ℕ) + (total_questions - 20) / 4)
  (h2 : total_questions ≥ 20) : total_questions = 32 := by
  sorry

end NUMINAMATH_CALUDE_test_questions_l2685_268576


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2685_268527

/-- The polynomial we're considering -/
def p (x : ℝ) : ℝ := x^2 - 9

/-- The proposed factorization of the polynomial -/
def f (x : ℝ) : ℝ := (x - 3) * (x + 3)

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, p x = f x) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2685_268527


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2685_268551

theorem sqrt_inequality (C : ℝ) (h : C > 1) :
  Real.sqrt (C + 1) - Real.sqrt C < Real.sqrt C - Real.sqrt (C - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2685_268551


namespace NUMINAMATH_CALUDE_special_line_equation_l2685_268575

/-- A line passing through a point with x-axis and y-axis intercepts that are opposite numbers -/
structure SpecialLine where
  -- The point through which the line passes
  point : ℝ × ℝ
  -- The equation of the line, represented as a function ℝ² → ℝ
  equation : ℝ → ℝ → ℝ
  -- Condition: The line passes through the given point
  passes_through_point : equation point.1 point.2 = 0
  -- Condition: The line has intercepts on x-axis and y-axis that are opposite numbers
  opposite_intercepts : ∃ (a : ℝ), (equation a 0 = 0 ∧ equation 0 (-a) = 0) ∨ 
                                   (equation (-a) 0 = 0 ∧ equation 0 a = 0)

/-- Theorem: The equation of the special line is either x - y - 7 = 0 or 2x + 5y = 0 -/
theorem special_line_equation (l : SpecialLine) (h : l.point = (5, -2)) :
  (l.equation = fun x y => x - y - 7) ∨ (l.equation = fun x y => 2*x + 5*y) := by
  sorry

end NUMINAMATH_CALUDE_special_line_equation_l2685_268575


namespace NUMINAMATH_CALUDE_monkey_swing_theorem_l2685_268585

/-- The distance a monkey swings in a given time -/
def monkey_swing_distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time * 60

/-- Theorem: A monkey swinging at 1.2 m/s for 30 minutes travels 2160 meters -/
theorem monkey_swing_theorem :
  monkey_swing_distance 1.2 30 = 2160 :=
by sorry

end NUMINAMATH_CALUDE_monkey_swing_theorem_l2685_268585


namespace NUMINAMATH_CALUDE_problem_solution_l2685_268589

theorem problem_solution : 
  let expr := (1 / (1 + 24 / 4) - 5 / 9) * (3 / (2 + 5 / 7)) / (2 / (3 + 3 / 4)) + 2.25
  ∀ A : ℝ, expr = 4 → (1 / (1 + 24 / A) - 5 / 9 = 1 / (1 + 24 / 4) - 5 / 9) → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2685_268589


namespace NUMINAMATH_CALUDE_line_through_point_l2685_268578

/-- Given a line with equation 3 - 2kx = -4y that contains the point (5, -2),
    prove that the value of k is -0.5 -/
theorem line_through_point (k : ℝ) : 
  (3 - 2 * k * 5 = -4 * (-2)) → k = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2685_268578


namespace NUMINAMATH_CALUDE_cube_root_and_square_root_l2685_268548

theorem cube_root_and_square_root (x y : ℝ) 
  (h1 : (x - 1) ^ (1/3 : ℝ) = 2) 
  (h2 : (y + 2) ^ (1/2 : ℝ) = 3) : 
  x - 2*y = -5 := by sorry

end NUMINAMATH_CALUDE_cube_root_and_square_root_l2685_268548


namespace NUMINAMATH_CALUDE_gridiron_club_members_l2685_268565

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost of a helmet in dollars -/
def helmet_cost : ℕ := 2 * tshirt_cost

/-- The cost of equipment for one member in dollars -/
def member_cost : ℕ := sock_cost + tshirt_cost + helmet_cost

/-- The total expenditure for all members in dollars -/
def total_expenditure : ℕ := 4680

/-- The number of members in the club -/
def club_members : ℕ := total_expenditure / member_cost

theorem gridiron_club_members :
  club_members = 104 :=
sorry

end NUMINAMATH_CALUDE_gridiron_club_members_l2685_268565


namespace NUMINAMATH_CALUDE_hotel_room_allocation_l2685_268592

theorem hotel_room_allocation (total_people : ℕ) (small_room_capacity : ℕ) 
  (num_small_rooms : ℕ) (h1 : total_people = 26) (h2 : small_room_capacity = 2) 
  (h3 : num_small_rooms = 1) :
  ∃ (large_room_capacity : ℕ),
    large_room_capacity = 12 ∧
    large_room_capacity > 0 ∧
    (total_people - num_small_rooms * small_room_capacity) % large_room_capacity = 0 ∧
    ∀ (x : ℕ), x > large_room_capacity → 
      (total_people - num_small_rooms * small_room_capacity) % x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_hotel_room_allocation_l2685_268592


namespace NUMINAMATH_CALUDE_chocolate_division_l2685_268502

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_for_shaina : ℕ) :
  total_chocolate = 72 / 7 →
  num_piles = 6 →
  piles_for_shaina = 2 →
  (total_chocolate / num_piles) * piles_for_shaina = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l2685_268502


namespace NUMINAMATH_CALUDE_total_games_is_32_l2685_268566

/-- The number of games won by Jerry -/
def jerry_wins : ℕ := 7

/-- The number of games won by Dave -/
def dave_wins : ℕ := jerry_wins + 3

/-- The number of games won by Ken -/
def ken_wins : ℕ := dave_wins + 5

/-- The total number of games played -/
def total_games : ℕ := jerry_wins + dave_wins + ken_wins

theorem total_games_is_32 : total_games = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_games_is_32_l2685_268566


namespace NUMINAMATH_CALUDE_kim_payment_amount_l2685_268569

def meal_cost : ℝ := 10
def drink_cost : ℝ := 2.5
def tip_percentage : ℝ := 0.2
def change_received : ℝ := 5

theorem kim_payment_amount :
  let total_before_tip := meal_cost + drink_cost
  let tip := tip_percentage * total_before_tip
  let total_with_tip := total_before_tip + tip
  let payment_amount := total_with_tip + change_received
  payment_amount = 20 := by sorry

end NUMINAMATH_CALUDE_kim_payment_amount_l2685_268569


namespace NUMINAMATH_CALUDE_seashells_found_joan_seashells_l2685_268530

theorem seashells_found (given_to_mike : ℕ) (has_now : ℕ) : ℕ :=
  given_to_mike + has_now

theorem joan_seashells : seashells_found 63 16 = 79 := by
  sorry

end NUMINAMATH_CALUDE_seashells_found_joan_seashells_l2685_268530


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l2685_268513

theorem inscribed_cube_volume (large_cube_edge : ℝ) (small_cube_edge : ℝ) (small_cube_volume : ℝ) : 
  large_cube_edge = 12 →
  small_cube_edge * Real.sqrt 3 = large_cube_edge →
  small_cube_volume = small_cube_edge ^ 3 →
  small_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l2685_268513


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2685_268561

/-- Given an arithmetic sequence where the eighth term is 20 and the common difference is 3,
    prove that the sum of the first three terms is 6. -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  (∀ n, a (n + 1) - a n = 3) →  -- Common difference is 3
  a 8 = 20 →                   -- Eighth term is 20
  a 1 + a 2 + a 3 = 6 :=        -- Sum of first three terms is 6
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2685_268561


namespace NUMINAMATH_CALUDE_equation_solution_l2685_268521

theorem equation_solution (a b c : ℤ) : 
  (∀ x : ℝ, (x - a) * (x - 10) + 5 = (x + b) * (x + c)) → (a = 4 ∨ a = 16) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2685_268521


namespace NUMINAMATH_CALUDE_factorial_ratio_eleven_nine_l2685_268549

theorem factorial_ratio_eleven_nine : Nat.factorial 11 / Nat.factorial 9 = 110 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_eleven_nine_l2685_268549


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2685_268535

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - 2 * i) / (2 + 5 * i) = (-4 : ℝ) / 29 - (19 : ℝ) / 29 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2685_268535


namespace NUMINAMATH_CALUDE_enemy_plane_hit_probability_l2685_268544

/-- The probability of A hitting the enemy plane -/
def prob_A_hit : ℝ := 0.6

/-- The probability of B hitting the enemy plane -/
def prob_B_hit : ℝ := 0.5

/-- The probability that the enemy plane is hit by at least one of A or B -/
def prob_plane_hit : ℝ := 1 - (1 - prob_A_hit) * (1 - prob_B_hit)

theorem enemy_plane_hit_probability :
  prob_plane_hit = 0.8 :=
sorry

end NUMINAMATH_CALUDE_enemy_plane_hit_probability_l2685_268544


namespace NUMINAMATH_CALUDE_gum_distribution_l2685_268507

theorem gum_distribution (num_cousins : ℕ) (gum_per_cousin : ℕ) : 
  num_cousins = 4 → gum_per_cousin = 5 → num_cousins * gum_per_cousin = 20 := by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l2685_268507


namespace NUMINAMATH_CALUDE_tina_career_result_l2685_268509

def boxer_career (initial_wins : ℕ) (additional_wins1 : ℕ) (triple_factor : ℕ) (additional_wins2 : ℕ) (double_factor : ℕ) : ℕ × ℕ :=
  let wins1 := initial_wins + additional_wins1
  let wins2 := wins1 * triple_factor
  let wins3 := wins2 + additional_wins2
  let final_wins := wins3 * double_factor
  let losses := 3
  (final_wins, losses)

theorem tina_career_result :
  let (wins, losses) := boxer_career 10 5 3 7 2
  wins - losses = 131 := by sorry

end NUMINAMATH_CALUDE_tina_career_result_l2685_268509


namespace NUMINAMATH_CALUDE_first_pass_bubble_sort_l2685_268534

def bubbleSortPass (list : List Int) : List Int :=
  list.zipWith (λ a b => if a > b then b else a) (list.drop 1 ++ [0])

theorem first_pass_bubble_sort :
  bubbleSortPass [8, 23, 12, 14, 39, 11] = [8, 12, 14, 23, 11, 39] := by
  sorry

end NUMINAMATH_CALUDE_first_pass_bubble_sort_l2685_268534


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2685_268543

theorem quadratic_inequality_solution_sets 
  (a b : ℝ) 
  (h : Set.Icc (-2 : ℝ) 1 = {x : ℝ | a * x^2 - x + b ≥ 0}) : 
  Set.Icc (-1/2 : ℝ) 1 = {x : ℝ | b * x^2 - x + a ≤ 0} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2685_268543


namespace NUMINAMATH_CALUDE_coefficient_d_value_l2685_268550

-- Define the polynomial Q(x)
def Q (x d : ℝ) : ℝ := x^4 + 3*x^3 + 2*x^2 + d*x + 15

-- State the theorem
theorem coefficient_d_value :
  ∃ d : ℝ, (∀ x : ℝ, Q x d = 0 → x = -3) ∧ d = 11 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_d_value_l2685_268550


namespace NUMINAMATH_CALUDE_sum_a4_a5_a6_l2685_268523

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_a2_a3 : a 2 + a 3 = 13
  a1_eq_2 : a 1 = 2

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem sum_a4_a5_a6 (seq : ArithmeticSequence) : seq.a 4 + seq.a 5 + seq.a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_a4_a5_a6_l2685_268523


namespace NUMINAMATH_CALUDE_error_percentage_l2685_268570

theorem error_percentage (y : ℝ) (h : y > 0) : 
  (|5 * y - y / 4| / (5 * y)) * 100 = 95 := by
  sorry

end NUMINAMATH_CALUDE_error_percentage_l2685_268570


namespace NUMINAMATH_CALUDE_projection_magnitude_l2685_268524

def a : ℝ × ℝ := (2, 1)
def b (k : ℝ) : ℝ × ℝ := (k, 3)

theorem projection_magnitude (k : ℝ) 
  (h : (a.1 + (b k).1, a.2 + (b k).2) • a = 0) : 
  |(a.1 * (b k).1 + a.2 * (b k).2) / Real.sqrt ((b k).1^2 + (b k).2^2)| = 1 :=
sorry

end NUMINAMATH_CALUDE_projection_magnitude_l2685_268524


namespace NUMINAMATH_CALUDE_smallest_integer_in_special_set_l2685_268533

theorem smallest_integer_in_special_set : ∃ (n : ℤ),
  (n + 6 > 2 * ((7 * n + 21) / 7)) ∧
  (∀ (m : ℤ), m < n → ¬(m + 6 > 2 * ((7 * m + 21) / 7))) →
  n = -1 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_special_set_l2685_268533


namespace NUMINAMATH_CALUDE_remaining_cakes_l2685_268554

def cakes_per_day : ℕ := 4
def baking_days : ℕ := 6
def eating_frequency : ℕ := 2

def total_baked (cakes_per_day baking_days : ℕ) : ℕ :=
  cakes_per_day * baking_days

def cakes_eaten (baking_days eating_frequency : ℕ) : ℕ :=
  baking_days / eating_frequency

theorem remaining_cakes :
  total_baked cakes_per_day baking_days - cakes_eaten baking_days eating_frequency = 21 :=
by sorry

end NUMINAMATH_CALUDE_remaining_cakes_l2685_268554


namespace NUMINAMATH_CALUDE_book_pyramid_theorem_l2685_268511

/-- Represents a book pyramid with a given number of levels -/
structure BookPyramid where
  levels : ℕ
  top_level_books : ℕ
  ratio : ℚ
  total_books : ℕ

/-- Calculates the total number of books in the pyramid -/
def calculate_total (p : BookPyramid) : ℚ :=
  p.top_level_books * (1 - p.ratio ^ p.levels) / (1 - p.ratio)

/-- Theorem stating the properties of the specific book pyramid -/
theorem book_pyramid_theorem (p : BookPyramid) 
  (h1 : p.levels = 4)
  (h2 : p.ratio = 4/5)
  (h3 : p.total_books = 369) :
  p.top_level_books = 64 := by
  sorry


end NUMINAMATH_CALUDE_book_pyramid_theorem_l2685_268511


namespace NUMINAMATH_CALUDE_class_size_with_error_l2685_268564

/-- Represents a class with a marking error -/
structure ClassWithError where
  n : ℕ  -- number of pupils
  S : ℕ  -- correct sum of marks
  wrong_mark : ℕ  -- wrongly entered mark
  correct_mark : ℕ  -- correct mark

/-- The conditions of the problem -/
def problem_conditions (c : ClassWithError) : Prop :=
  c.wrong_mark = 79 ∧
  c.correct_mark = 45 ∧
  (c.S + (c.wrong_mark - c.correct_mark)) / c.n = 3/2 * (c.S / c.n)

/-- The theorem stating the solution -/
theorem class_size_with_error (c : ClassWithError) :
  problem_conditions c → c.n = 68 :=
by sorry

end NUMINAMATH_CALUDE_class_size_with_error_l2685_268564


namespace NUMINAMATH_CALUDE_distance_ratio_in_pyramid_l2685_268532

/-- A regular square pyramid with vertex P and base ABCD -/
structure RegularSquarePyramid where
  base_side_length : ℝ
  height : ℝ

/-- A point inside the base of the pyramid -/
structure PointInBase where
  x : ℝ
  y : ℝ

/-- Sum of distances from a point to all faces of the pyramid -/
def sum_distances_to_faces (p : RegularSquarePyramid) (e : PointInBase) : ℝ := sorry

/-- Sum of distances from a point to all edges of the base -/
def sum_distances_to_base_edges (p : RegularSquarePyramid) (e : PointInBase) : ℝ := sorry

/-- The main theorem stating the ratio of distances -/
theorem distance_ratio_in_pyramid (p : RegularSquarePyramid) (e : PointInBase) 
  (h_centroid : e ≠ PointInBase.mk (p.base_side_length / 2) (p.base_side_length / 2)) :
  sum_distances_to_faces p e / sum_distances_to_base_edges p e = 
    8 * Real.sqrt (p.height^2 + p.base_side_length^2 / 2) / p.base_side_length := by
  sorry

end NUMINAMATH_CALUDE_distance_ratio_in_pyramid_l2685_268532


namespace NUMINAMATH_CALUDE_digit_2000th_position_l2685_268598

/-- The sequence of digits formed by concatenating consecutive positive integers starting from 1 -/
def concatenatedSequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => (concatenatedSequence n) * 10 + ((n + 1) % 10)

/-- The digit at a given position in the concatenated sequence -/
def digitAtPosition (pos : ℕ) : ℕ :=
  (concatenatedSequence pos) % 10

theorem digit_2000th_position :
  digitAtPosition 1999 = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_2000th_position_l2685_268598


namespace NUMINAMATH_CALUDE_equation_solution_l2685_268559

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 6*x*y - 18*y + 3*x - 9 = 0) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2685_268559


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l2685_268515

theorem quadratic_inequality_empty_solution (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l2685_268515


namespace NUMINAMATH_CALUDE_divisible_by_24_l2685_268504

theorem divisible_by_24 (n : ℕ+) : ∃ k : ℤ, (n : ℤ)^4 + 2*(n : ℤ)^3 + 11*(n : ℤ)^2 + 10*(n : ℤ) = 24*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l2685_268504


namespace NUMINAMATH_CALUDE_equation_solution_l2685_268581

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x * (x - 3)^2 * (5 + x)
  {x : ℝ | f x = 0} = {0, 3, -5} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2685_268581


namespace NUMINAMATH_CALUDE_rectangle_circles_radii_sum_l2685_268536

theorem rectangle_circles_radii_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (4 * b^2 + a^2) / (4 * b) + (4 * a^2 + b^2) / (4 * a) ≥ 5 * (a + b) / 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circles_radii_sum_l2685_268536


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2685_268539

theorem complex_equation_solution (z : ℂ) : (z - 1) / (z + 1) = I → z = I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2685_268539


namespace NUMINAMATH_CALUDE_train_length_calculation_l2685_268594

/-- Given a train traveling at a certain speed that crosses a bridge of known length in a specific time, this theorem calculates the length of the train. -/
theorem train_length_calculation (train_speed : Real) (bridge_length : Real) (crossing_time : Real) :
  train_speed = 90 * (1000 / 3600) → -- Convert 90 km/hr to m/s
  bridge_length = 275 →
  crossing_time = 30 →
  (train_speed * crossing_time) - bridge_length = 475 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l2685_268594


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2685_268586

theorem quadratic_inequality_solution (a : ℝ) :
  let solution_set := {x : ℝ | 12 * x^2 - a * x - a^2 < 0}
  if a > 0 then
    solution_set = {x : ℝ | -a/4 < x ∧ x < a/3}
  else if a = 0 then
    solution_set = ∅
  else
    solution_set = {x : ℝ | a/3 < x ∧ x < -a/4} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2685_268586


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l2685_268557

theorem min_value_quadratic_form (x y : ℝ) : 2 * x^2 + 3 * x * y + 4 * y^2 + 5 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l2685_268557


namespace NUMINAMATH_CALUDE_average_habitable_land_per_person_approx_l2685_268580

-- Define the given constants
def total_population : ℕ := 281000000
def total_land_area : ℝ := 3797000
def habitable_land_percentage : ℝ := 0.8
def feet_per_mile : ℕ := 5280

-- Theorem statement
theorem average_habitable_land_per_person_approx :
  let habitable_land_area : ℝ := total_land_area * habitable_land_percentage
  let total_habitable_sq_feet : ℝ := habitable_land_area * (feet_per_mile ^ 2 : ℝ)
  let avg_sq_feet_per_person : ℝ := total_habitable_sq_feet / total_population
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1000 ∧ |avg_sq_feet_per_person - 300000| < ε :=
by sorry

end NUMINAMATH_CALUDE_average_habitable_land_per_person_approx_l2685_268580


namespace NUMINAMATH_CALUDE_gala_hat_count_l2685_268571

theorem gala_hat_count (total_attendees : ℕ) 
  (women_fraction : ℚ) (women_hat_percent : ℚ) (men_hat_percent : ℚ)
  (h1 : total_attendees = 2400)
  (h2 : women_fraction = 2/3)
  (h3 : women_hat_percent = 30/100)
  (h4 : men_hat_percent = 12/100) : 
  ↑⌊women_fraction * total_attendees * women_hat_percent⌋ + 
  ↑⌊(1 - women_fraction) * total_attendees * men_hat_percent⌋ = 576 :=
by sorry

end NUMINAMATH_CALUDE_gala_hat_count_l2685_268571


namespace NUMINAMATH_CALUDE_helen_to_betsy_win_ratio_l2685_268508

/-- The ratio of Helen's wins to Betsy's wins in a Monopoly game scenario -/
theorem helen_to_betsy_win_ratio :
  ∀ (helen_wins : ℕ),
  let betsy_wins : ℕ := 5
  let susan_wins : ℕ := 3 * betsy_wins
  let total_wins : ℕ := 30
  (betsy_wins + helen_wins + susan_wins = total_wins) →
  (helen_wins : ℚ) / betsy_wins = 2 := by
    sorry

end NUMINAMATH_CALUDE_helen_to_betsy_win_ratio_l2685_268508


namespace NUMINAMATH_CALUDE_sales_goals_calculation_l2685_268538

/-- Represents the sales data for a candy store employee over three days. -/
structure SalesData :=
  (jetBarGoal : ℕ)
  (zippyBarGoal : ℕ)
  (candyCloudGoal : ℕ)
  (mondayJetBars : ℕ)
  (mondayZippyBars : ℕ)
  (mondayCandyClouds : ℕ)
  (tuesdayJetBarsDiff : ℤ)
  (tuesdayZippyBarsDiff : ℕ)
  (wednesdayCandyCloudsMultiplier : ℕ)

/-- Calculates the remaining sales needed to reach the weekly goals. -/
def remainingSales (data : SalesData) : ℤ × ℤ × ℤ :=
  let totalJetBars := data.mondayJetBars + (data.mondayJetBars : ℤ) + data.tuesdayJetBarsDiff
  let totalZippyBars := data.mondayZippyBars + data.mondayZippyBars + data.tuesdayZippyBarsDiff
  let totalCandyClouds := data.mondayCandyClouds + data.mondayCandyClouds * data.wednesdayCandyCloudsMultiplier
  ((data.jetBarGoal : ℤ) - totalJetBars,
   (data.zippyBarGoal : ℤ) - (totalZippyBars : ℤ),
   (data.candyCloudGoal : ℤ) - (totalCandyClouds : ℤ))

theorem sales_goals_calculation (data : SalesData)
  (h1 : data.jetBarGoal = 90)
  (h2 : data.zippyBarGoal = 70)
  (h3 : data.candyCloudGoal = 50)
  (h4 : data.mondayJetBars = 45)
  (h5 : data.mondayZippyBars = 34)
  (h6 : data.mondayCandyClouds = 16)
  (h7 : data.tuesdayJetBarsDiff = -16)
  (h8 : data.tuesdayZippyBarsDiff = 8)
  (h9 : data.wednesdayCandyCloudsMultiplier = 2) :
  remainingSales data = (16, -6, 2) :=
by sorry


end NUMINAMATH_CALUDE_sales_goals_calculation_l2685_268538


namespace NUMINAMATH_CALUDE_probability_is_9_128_l2685_268545

/-- Four points chosen uniformly at random on a circle -/
def random_points_on_circle : Type := Fin 4 → ℝ × ℝ

/-- The circle's center -/
def circle_center : ℝ × ℝ := (0, 0)

/-- Checks if three points form an obtuse triangle -/
def is_obtuse_triangle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- The probability of no two points forming an obtuse triangle with the center -/
def probability_no_obtuse_triangle (points : random_points_on_circle) : ℝ := sorry

/-- Main theorem: The probability is 9/128 -/
theorem probability_is_9_128 :
  ∀ points : random_points_on_circle,
  probability_no_obtuse_triangle points = 9 / 128 := by sorry

end NUMINAMATH_CALUDE_probability_is_9_128_l2685_268545


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l2685_268573

theorem max_product_sum_2000 :
  ∃ (x : ℤ), ∀ (y : ℤ), x * (2000 - x) ≥ y * (2000 - y) ∧ x * (2000 - x) = 1000000 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l2685_268573


namespace NUMINAMATH_CALUDE_investment_ratio_l2685_268562

/-- Prove that given two equal investments of $12000, one at 11% and one at 9%,
    the ratio of these investments is 1:1 if the total interest after 1 year is $2400. -/
theorem investment_ratio (investment_11 investment_9 : ℝ) 
  (h1 : investment_11 = 12000)
  (h2 : investment_9 = 12000)
  (h3 : 0.11 * investment_11 + 0.09 * investment_9 = 2400) :
  investment_11 / investment_9 = 1 := by
sorry

end NUMINAMATH_CALUDE_investment_ratio_l2685_268562


namespace NUMINAMATH_CALUDE_samantha_buys_four_toys_l2685_268584

/-- Represents the price of a dog toy in cents -/
def toy_price : ℕ := 1200

/-- Represents the total amount spent on dog toys in cents -/
def total_spent : ℕ := 3600

/-- Calculates the cost of a pair of toys under the "buy one get one half off" promotion -/
def pair_cost : ℕ := toy_price + toy_price / 2

/-- Represents the number of toys Samantha buys -/
def num_toys : ℕ := (total_spent / pair_cost) * 2

theorem samantha_buys_four_toys : num_toys = 4 := by
  sorry

end NUMINAMATH_CALUDE_samantha_buys_four_toys_l2685_268584


namespace NUMINAMATH_CALUDE_product_equals_three_l2685_268567

theorem product_equals_three : 
  (∀ a b c : ℝ, a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) → 
  6 * 15 * 3 = 3 := by sorry

end NUMINAMATH_CALUDE_product_equals_three_l2685_268567


namespace NUMINAMATH_CALUDE_greater_number_proof_l2685_268505

theorem greater_number_proof (x y : ℝ) (h1 : x > y) (h2 : x > 0) (h3 : y > 0)
  (h4 : x * y = 2688) (h5 : (x + y) - (x - y) = 64) : x = 84 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l2685_268505


namespace NUMINAMATH_CALUDE_tan_alpha_eq_neg_one_l2685_268501

theorem tan_alpha_eq_neg_one (α : ℝ) (h : Real.sin (π/6 - α) = Real.cos (π/6 + α)) : 
  Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_neg_one_l2685_268501


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l2685_268500

/-- Given two lines l₁ and l₂ in the plane, and a third line l₃,
    this theorem proves that the line ax + by + c = 0 passes through
    the intersection point of l₁ and l₂, and is perpendicular to l₃. -/
theorem intersection_and_perpendicular_line
  (l₁ : Real → Real → Prop) (l₂ : Real → Real → Prop) (l₃ : Real → Real → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ 2 * x - 3 * y + 10 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 3 * x + 4 * y - 2 = 0)
  (h₃ : ∀ x y, l₃ x y ↔ 3 * x - 2 * y + 5 = 0)
  : ∃ x y, l₁ x y ∧ l₂ x y ∧ 2 * x + 3 * y - 2 = 0 ∧
    (∀ x₁ y₁ x₂ y₂, l₃ x₁ y₁ → l₃ x₂ y₂ → (y₂ - y₁) * (3 * (x₂ - x₁)) = -2 * (y₂ - y₁)) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l2685_268500


namespace NUMINAMATH_CALUDE_lottery_tax_percentage_l2685_268595

/-- Proves that the percentage of lottery winnings paid for tax is 20% given the specified conditions --/
theorem lottery_tax_percentage (winnings : ℝ) (processing_fee : ℝ) (take_home : ℝ) : 
  winnings = 50 → processing_fee = 5 → take_home = 35 → 
  (winnings - (take_home + processing_fee)) / winnings * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_lottery_tax_percentage_l2685_268595


namespace NUMINAMATH_CALUDE_wire_division_l2685_268537

theorem wire_division (total_feet : ℕ) (total_inches : ℕ) (num_parts : ℕ) 
  (h1 : total_feet = 5) 
  (h2 : total_inches = 4) 
  (h3 : num_parts = 4) 
  (h4 : ∀ (feet : ℕ), feet * 12 = feet * (1 : ℕ) * 12) :
  (total_feet * 12 + total_inches) / num_parts = 16 := by
  sorry

end NUMINAMATH_CALUDE_wire_division_l2685_268537


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2685_268588

theorem quadratic_discriminant (a b c : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, a * x^2 + b * x + c = a * (x - x₁) * (x - x₂)) →
  |x₂ - x₁| = 2 →
  b^2 - 4*a*c = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2685_268588


namespace NUMINAMATH_CALUDE_sequence_general_term_l2685_268526

-- Define the sequence and its partial sum
def S (n : ℕ) : ℤ := n^2 - 4*n

-- Define the general term of the sequence
def a (n : ℕ) : ℤ := 2*n - 5

-- Theorem statement
theorem sequence_general_term (n : ℕ) : 
  n ≥ 1 → S n - S (n-1) = a n :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2685_268526


namespace NUMINAMATH_CALUDE_dumpling_selection_probability_l2685_268503

/-- The number of dumplings of each kind in the pot -/
def dumplings_per_kind : ℕ := 5

/-- The number of different kinds of dumplings -/
def kinds_of_dumplings : ℕ := 3

/-- The total number of dumplings in the pot -/
def total_dumplings : ℕ := dumplings_per_kind * kinds_of_dumplings

/-- The number of dumplings to be selected -/
def selected_dumplings : ℕ := 4

/-- The probability of selecting at least one dumpling of each kind -/
def probability_at_least_one_of_each : ℚ := 50 / 91

theorem dumpling_selection_probability :
  (Nat.choose total_dumplings selected_dumplings *
   probability_at_least_one_of_each : ℚ) =
  (Nat.choose kinds_of_dumplings 1 *
   Nat.choose dumplings_per_kind 2 *
   Nat.choose dumplings_per_kind 1 *
   Nat.choose dumplings_per_kind 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_dumpling_selection_probability_l2685_268503


namespace NUMINAMATH_CALUDE_angle_bisector_of_lines_l2685_268547

-- Define the two lines
def L₁ (x y : ℝ) : Prop := 4 * x - 3 * y + 1 = 0
def L₂ (x y : ℝ) : Prop := 12 * x + 5 * y + 13 = 0

-- Define the angle bisector
def angle_bisector (x y : ℝ) : Prop := 56 * x - 7 * y + 39 = 0

-- Theorem statement
theorem angle_bisector_of_lines :
  ∀ x y : ℝ, angle_bisector x y ↔ (L₁ x y ∧ L₂ x y → ∃ t : ℝ, t > 0 ∧ 
    abs ((4 * x - 3 * y + 1) / (12 * x + 5 * y + 13)) = t) :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_of_lines_l2685_268547


namespace NUMINAMATH_CALUDE_ratio_equality_solution_l2685_268546

theorem ratio_equality_solution (x : ℝ) : (0.75 / x = 5 / 9) → x = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_solution_l2685_268546


namespace NUMINAMATH_CALUDE_deepak_age_l2685_268510

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 10 = 26 →
  deepak_age = 12 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l2685_268510


namespace NUMINAMATH_CALUDE_guppy_count_theorem_l2685_268514

/-- Calculates the total number of guppies given initial count and two batches of baby guppies -/
def total_guppies (initial : ℕ) (first_batch : ℕ) (second_batch : ℕ) : ℕ :=
  initial + first_batch + second_batch

/-- Converts dozens to individual count -/
def dozens_to_count (dozens : ℕ) : ℕ :=
  dozens * 12

theorem guppy_count_theorem (initial : ℕ) (first_batch_dozens : ℕ) (second_batch : ℕ) 
  (h1 : initial = 7)
  (h2 : first_batch_dozens = 3)
  (h3 : second_batch = 9) :
  total_guppies initial (dozens_to_count first_batch_dozens) second_batch = 52 := by
  sorry

end NUMINAMATH_CALUDE_guppy_count_theorem_l2685_268514


namespace NUMINAMATH_CALUDE_macey_savings_l2685_268542

/-- The amount Macey has already saved is equal to the cost of the shirt minus the amount she will save in the next 3 weeks. -/
theorem macey_savings (shirt_cost : ℝ) (weeks_left : ℕ) (weekly_savings : ℝ) 
  (h1 : shirt_cost = 3)
  (h2 : weeks_left = 3)
  (h3 : weekly_savings = 0.5) :
  shirt_cost - (weeks_left : ℝ) * weekly_savings = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_macey_savings_l2685_268542


namespace NUMINAMATH_CALUDE_unique_solution_values_l2685_268599

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop := a * x^2 - 2 * x + 1 = 0

-- Define the property of having exactly one solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x, quadratic_equation a x

-- Theorem statement
theorem unique_solution_values :
  ∀ a : ℝ, has_unique_solution a ↔ (a = 0 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_values_l2685_268599


namespace NUMINAMATH_CALUDE_want_is_correct_choice_l2685_268572

/-- Represents the possible word choices for the sentence --/
inductive WordChoice
  | hope
  | search
  | want
  | charge

/-- Represents the context of the situation --/
structure Situation where
  duration : Nat
  location : String
  isSnowstorm : Bool
  lackOfSupplies : Bool

/-- Defines the correct word choice given a situation --/
def correctWordChoice (s : Situation) : WordChoice :=
  if s.duration ≥ 5 && s.location = "station" && s.isSnowstorm && s.lackOfSupplies then
    WordChoice.want
  else
    WordChoice.hope  -- Default choice, not relevant for this problem

/-- Theorem stating that 'want' is the correct word choice for the given situation --/
theorem want_is_correct_choice (s : Situation) 
  (h1 : s.duration = 5)
  (h2 : s.location = "station")
  (h3 : s.isSnowstorm = true)
  (h4 : s.lackOfSupplies = true) :
  correctWordChoice s = WordChoice.want := by
  sorry


end NUMINAMATH_CALUDE_want_is_correct_choice_l2685_268572


namespace NUMINAMATH_CALUDE_temperature_calculation_l2685_268591

theorem temperature_calculation (T₁ T₂ : ℝ) : 
  2.24 * T₁ = 1.1 * 2 * 298 ∧ 1.76 * T₂ = 1.1 * 2 * 298 → 
  T₁ = 292.7 ∧ T₂ = 372.5 := by
  sorry

end NUMINAMATH_CALUDE_temperature_calculation_l2685_268591


namespace NUMINAMATH_CALUDE_smallest_max_sum_l2685_268512

theorem smallest_max_sum (p q r s t : ℕ+) 
  (sum_constraint : p + q + r + s + t = 2015) : 
  (∃ (N : ℕ), 
    N = max (p + q) (max (q + r) (max (r + s) (s + t))) ∧ 
    N = 1005 ∧
    ∀ (M : ℕ), (M = max (p + q) (max (q + r) (max (r + s) (s + t))) → M ≥ N)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l2685_268512


namespace NUMINAMATH_CALUDE_snow_cone_price_is_0_875_l2685_268531

/-- Calculates the price of a snow cone given the conditions of Todd's snow-cone stand. -/
def snow_cone_price (borrowed : ℚ) (repay : ℚ) (ingredients_cost : ℚ) (num_sold : ℕ) (leftover : ℚ) : ℚ :=
  (repay + leftover) / num_sold

/-- Proves that the price of each snow cone is $0.875 under the given conditions. -/
theorem snow_cone_price_is_0_875 :
  snow_cone_price 100 110 75 200 65 = 0.875 := by
  sorry

end NUMINAMATH_CALUDE_snow_cone_price_is_0_875_l2685_268531


namespace NUMINAMATH_CALUDE_xy_value_l2685_268577

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x + y) = 16)
  (h2 : (27 : ℝ)^(x + y) / (9 : ℝ)^(5 * y) = 729) :
  x * y = 96 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l2685_268577


namespace NUMINAMATH_CALUDE_stating_ball_falls_in_hole_iff_rational_ratio_l2685_268583

/-- Represents a rectangular pool table with sides of lengths a and b -/
structure PoolTable where
  a : ℝ
  b : ℝ

/-- Predicate to check if a ratio is rational -/
def isRational (x : ℝ) : Prop := ∃ (m n : ℤ), n ≠ 0 ∧ x = m / n

/-- 
  Theorem stating that a ball shot from one corner along the angle bisector 
  will eventually fall into a hole at one of the other three corners 
  if and only if the ratio of side lengths is rational
-/
theorem ball_falls_in_hole_iff_rational_ratio (table : PoolTable) : 
  (∃ (k l : ℤ), k ≠ 0 ∧ l ≠ 0 ∧ table.a * k = table.b * l) ↔ isRational (table.a / table.b) := by
  sorry


end NUMINAMATH_CALUDE_stating_ball_falls_in_hole_iff_rational_ratio_l2685_268583


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2685_268556

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2685_268556


namespace NUMINAMATH_CALUDE_student_distribution_count_l2685_268518

/-- The number of ways to distribute 5 students into three groups -/
def distribute_students : ℕ :=
  -- The actual distribution logic would go here
  80

/-- The conditions for the distribution -/
def valid_distribution (a b c : ℕ) : Prop :=
  a + b + c = 5 ∧ a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 1

theorem student_distribution_count :
  ∃ (a b c : ℕ), valid_distribution a b c ∧
  (∀ (x y z : ℕ), valid_distribution x y z → x + y + z = 5) ∧
  distribute_students = 80 :=
sorry

end NUMINAMATH_CALUDE_student_distribution_count_l2685_268518


namespace NUMINAMATH_CALUDE_property_price_reduction_l2685_268540

/-- Represents the price reduction scenario of a property over two years -/
theorem property_price_reduction (x : ℝ) : 
  (20000 : ℝ) * (1 - x)^2 = 16200 ↔ 
  (∃ (initial_price final_price : ℝ), 
    initial_price = 20000 ∧ 
    final_price = 16200 ∧ 
    final_price = initial_price * (1 - x)^2 ∧ 
    0 ≤ x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_property_price_reduction_l2685_268540


namespace NUMINAMATH_CALUDE_no_snow_no_rain_probability_l2685_268587

theorem no_snow_no_rain_probability 
  (prob_snow : ℚ) 
  (prob_rain : ℚ) 
  (days : ℕ) 
  (h1 : prob_snow = 2/3) 
  (h2 : prob_rain = 1/2) 
  (h3 : days = 5) : 
  (1 - prob_snow) * (1 - prob_rain) ^ days = 1/7776 :=
sorry

end NUMINAMATH_CALUDE_no_snow_no_rain_probability_l2685_268587


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_two_fifths_l2685_268553

noncomputable def g (x : ℝ) : ℝ := 3 - x^2

noncomputable def f (x : ℝ) : ℝ := 
  if x = 0 then 0 else (3 - (g⁻¹ x)^2) / (g⁻¹ x)^2

theorem f_neg_two_eq_neg_two_fifths : f (-2) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_two_fifths_l2685_268553


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_implies_a_negative_l2685_268552

/-- A point is in the third quadrant if both its x and y coordinates are negative -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- If point A(a, a-1) is in the third quadrant, then a < 0 -/
theorem point_in_third_quadrant_implies_a_negative (a : ℝ) : 
  in_third_quadrant a (a - 1) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_implies_a_negative_l2685_268552


namespace NUMINAMATH_CALUDE_inner_circles_radii_l2685_268582

/-- An isosceles triangle with a 120° angle and an inscribed circle of radius R -/
structure IsoscelesTriangle120 where
  R : ℝ
  R_pos : R > 0

/-- Two equal circles inside the triangle that touch each other,
    where each circle touches one leg of the triangle and the inscribed circle -/
structure InnerCircles (t : IsoscelesTriangle120) where
  radius : ℝ
  radius_pos : radius > 0

/-- The theorem stating the possible radii of the inner circles -/
theorem inner_circles_radii (t : IsoscelesTriangle120) (c : InnerCircles t) :
  c.radius = t.R / 3 ∨ c.radius = (3 - 2 * Real.sqrt 2) / 3 * t.R :=
by sorry

end NUMINAMATH_CALUDE_inner_circles_radii_l2685_268582


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_l2685_268555

/-- The area of a geometric figure formed by a rectangle and an additional triangle -/
theorem rectangle_triangle_area (a b : ℝ) (h : 0 < a ∧ a < b) :
  let diagonal := Real.sqrt (a^2 + b^2)
  let triangle_area := (b * diagonal) / 4
  let total_area := a * b + triangle_area
  total_area = a * b + (b * Real.sqrt (a^2 + b^2)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_l2685_268555


namespace NUMINAMATH_CALUDE_shaded_area_of_tiled_floor_l2685_268528

/-- Calculates the shaded area of a floor with specific tiling pattern -/
theorem shaded_area_of_tiled_floor (floor_length floor_width tile_size : ℝ)
  (quarter_circle_radius : ℝ) :
  floor_length = 8 →
  floor_width = 10 →
  tile_size = 1 →
  quarter_circle_radius = 1/2 →
  (floor_length * floor_width) * (tile_size^2 - π * quarter_circle_radius^2) = 80 - 20 * π :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_tiled_floor_l2685_268528


namespace NUMINAMATH_CALUDE_goldfish_preference_total_l2685_268525

theorem goldfish_preference_total : 
  let johnson_class := 30
  let johnson_ratio := (1 : ℚ) / 6
  let feldstein_class := 45
  let feldstein_ratio := (2 : ℚ) / 3
  let henderson_class := 36
  let henderson_ratio := (1 : ℚ) / 5
  let dias_class := 50
  let dias_ratio := (3 : ℚ) / 5
  let norris_class := 25
  let norris_ratio := (2 : ℚ) / 5
  ⌊johnson_class * johnson_ratio⌋ +
  ⌊feldstein_class * feldstein_ratio⌋ +
  ⌊henderson_class * henderson_ratio⌋ +
  ⌊dias_class * dias_ratio⌋ +
  ⌊norris_class * norris_ratio⌋ = 82 :=
by sorry


end NUMINAMATH_CALUDE_goldfish_preference_total_l2685_268525


namespace NUMINAMATH_CALUDE_final_tomato_count_l2685_268529

def cherry_tomatoes (initial : ℕ) : ℕ :=
  let after_first_birds := initial - (initial / 3)
  let after_second_birds := after_first_birds - (after_first_birds * 2 / 5)
  let after_growth := after_second_birds + (after_second_birds / 2)
  let after_more_growth := after_growth + 4
  after_more_growth - (after_more_growth / 4)

theorem final_tomato_count : cherry_tomatoes 21 = 13 := by
  sorry

end NUMINAMATH_CALUDE_final_tomato_count_l2685_268529


namespace NUMINAMATH_CALUDE_polynomial_nonnegative_implies_a_range_a_range_implies_polynomial_nonnegative_l2685_268522

/-- A real coefficient polynomial f(x) = x^4 + (a-1)x^2 + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + (a-1)*x^2 + 1

/-- Theorem: If f(x) is non-negative for all real x, then a ≥ -1 -/
theorem polynomial_nonnegative_implies_a_range (a : ℝ) 
  (h : ∀ x : ℝ, f a x ≥ 0) : a ≥ -1 := by
  sorry

/-- Theorem: If a ≥ -1, then f(x) is non-negative for all real x -/
theorem a_range_implies_polynomial_nonnegative (a : ℝ) 
  (h : a ≥ -1) : ∀ x : ℝ, f a x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_implies_a_range_a_range_implies_polynomial_nonnegative_l2685_268522


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2685_268516

theorem simplify_fraction_product : (2 / (2 + Real.sqrt 3)) * (2 / (2 - Real.sqrt 3)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2685_268516
