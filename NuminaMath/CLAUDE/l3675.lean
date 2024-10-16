import Mathlib

namespace NUMINAMATH_CALUDE_remainder_theorem_l3675_367565

def f (x : ℝ) : ℝ := x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_theorem : ∃ (Q : ℝ → ℝ), ∀ x, f (x^10) = f x * Q x + 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3675_367565


namespace NUMINAMATH_CALUDE_optimal_quadruple_l3675_367538

def is_valid_quadruple (k l m n : ℕ) : Prop :=
  k > l ∧ l > m ∧ m > n

def sum_inverse (k l m n : ℕ) : ℚ :=
  1 / k + 1 / l + 1 / m + 1 / n

theorem optimal_quadruple :
  ∀ k l m n : ℕ,
    is_valid_quadruple k l m n →
    sum_inverse k l m n < 1 →
    sum_inverse k l m n ≤ sum_inverse 43 7 3 2 :=
by sorry

end NUMINAMATH_CALUDE_optimal_quadruple_l3675_367538


namespace NUMINAMATH_CALUDE_range_of_a_l3675_367574

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y + 4 = 2*x*y → 
    x^2 + 2*x*y + y^2 - a*x - a*y + 1 ≥ 0) → 
  a ≤ 17/4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3675_367574


namespace NUMINAMATH_CALUDE_jalapeno_slices_per_pepper_l3675_367537

/-- The number of jalapeno strips required per sandwich -/
def strips_per_sandwich : ℕ := 4

/-- The time in minutes between serving each sandwich -/
def minutes_per_sandwich : ℕ := 5

/-- The number of hours the shop operates per day -/
def operating_hours : ℕ := 8

/-- The number of jalapeno peppers required for a full day of operation -/
def peppers_per_day : ℕ := 48

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem jalapeno_slices_per_pepper : 
  (operating_hours * minutes_per_hour / minutes_per_sandwich) * strips_per_sandwich / peppers_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_jalapeno_slices_per_pepper_l3675_367537


namespace NUMINAMATH_CALUDE_cone_base_radius_l3675_367560

/-- Given a cone whose lateral surface is a semicircle with radius 2,
    prove that the radius of the base of the cone is 1. -/
theorem cone_base_radius (r : ℝ) (h : r > 0) : r = 1 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3675_367560


namespace NUMINAMATH_CALUDE_min_even_integers_l3675_367578

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 30 →
  a + b + c + d = 50 →
  a + b + c + d + e + f = 70 →
  ∃ (x y z w u v : ℤ), 
    x + y = 30 ∧
    x + y + z + w = 50 ∧
    x + y + z + w + u + v = 70 ∧
    Even x ∧ Even y ∧ Even z ∧ Even w ∧ Even u ∧ Even v :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l3675_367578


namespace NUMINAMATH_CALUDE_fixed_point_satisfies_line_fixed_point_unique_l3675_367577

/-- A line that passes through a fixed point for all values of m -/
def line (m x y : ℝ) : Prop :=
  (3*m + 4)*x + (5 - 2*m)*y + 7*m - 6 = 0

/-- The fixed point through which the line always passes -/
def fixed_point : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the fixed point satisfies the line equation for all m -/
theorem fixed_point_satisfies_line :
  ∀ m : ℝ, line m (fixed_point.1) (fixed_point.2) :=
by sorry

/-- Theorem stating that the fixed point is unique -/
theorem fixed_point_unique :
  ∀ x y : ℝ, (∀ m : ℝ, line m x y) → (x, y) = fixed_point :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_satisfies_line_fixed_point_unique_l3675_367577


namespace NUMINAMATH_CALUDE_exists_perpendicular_line_l3675_367514

-- Define a plane
variable (α : Set (ℝ × ℝ × ℝ))

-- Define a line
variable (l : Set (ℝ × ℝ × ℝ))

-- Define a predicate for a line being in a plane
def LineInPlane (line : Set (ℝ × ℝ × ℝ)) (plane : Set (ℝ × ℝ × ℝ)) : Prop :=
  line ⊆ plane

-- Define a predicate for two lines being perpendicular
def Perpendicular (line1 line2 : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry -- Definition of perpendicularity

-- Theorem statement
theorem exists_perpendicular_line (α : Set (ℝ × ℝ × ℝ)) (l : Set (ℝ × ℝ × ℝ)) :
  ∃ m : Set (ℝ × ℝ × ℝ), LineInPlane m α ∧ Perpendicular m l :=
sorry

end NUMINAMATH_CALUDE_exists_perpendicular_line_l3675_367514


namespace NUMINAMATH_CALUDE_quinary_444_equals_octal_174_l3675_367593

/-- Converts a quinary (base-5) number to decimal (base-10) --/
def quinary_to_decimal (q : ℕ) : ℕ := 
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

/-- Converts a decimal (base-10) number to octal (base-8) --/
def decimal_to_octal (d : ℕ) : ℕ := 
  1 * 8^2 + 7 * 8^1 + 4 * 8^0

/-- Theorem stating that 444₅ in quinary is equal to 174₈ in octal --/
theorem quinary_444_equals_octal_174 : 
  quinary_to_decimal 444 = decimal_to_octal 174 := by
  sorry

end NUMINAMATH_CALUDE_quinary_444_equals_octal_174_l3675_367593


namespace NUMINAMATH_CALUDE_average_weight_calculation_l3675_367553

theorem average_weight_calculation (total_boys : ℕ) (group1_boys : ℕ) (group2_boys : ℕ)
  (group2_avg_weight : ℝ) (total_avg_weight : ℝ) :
  total_boys = group1_boys + group2_boys →
  group2_boys = 8 →
  group2_avg_weight = 45.15 →
  total_avg_weight = 48.55 →
  let group1_avg_weight := (total_boys * total_avg_weight - group2_boys * group2_avg_weight) / group1_boys
  group1_avg_weight = 50.25 := by
sorry

end NUMINAMATH_CALUDE_average_weight_calculation_l3675_367553


namespace NUMINAMATH_CALUDE_alice_bob_meet_l3675_367531

/-- The number of points on the circular path -/
def n : ℕ := 18

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 11

/-- The number of turns after which Alice and Bob meet -/
def meeting_turns : ℕ := 9

/-- Function to calculate the position after a certain number of moves -/
def position_after_moves (start : ℕ) (move : ℕ) (turns : ℕ) : ℕ :=
  (start + move * turns - 1) % n + 1

theorem alice_bob_meet :
  position_after_moves n alice_move meeting_turns =
  position_after_moves n (n - bob_move) meeting_turns :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l3675_367531


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3675_367527

/-- An isosceles triangle with perimeter 26 and one side 12 has the other side length either 12 or 7 -/
theorem isosceles_triangle_side_length (a b c : ℝ) : 
  a + b + c = 26 → -- perimeter is 26
  (a = b ∨ b = c ∨ a = c) → -- isosceles condition
  (a = 12 ∨ b = 12 ∨ c = 12) → -- one side is 12
  (a = 7 ∨ b = 7 ∨ c = 7) ∨ (a = 12 ∧ b = 12) ∨ (b = 12 ∧ c = 12) ∨ (a = 12 ∧ c = 12) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3675_367527


namespace NUMINAMATH_CALUDE_no_rain_probability_l3675_367516

def probability_of_rain : ℚ := 2/3

theorem no_rain_probability (days : ℕ) (h : days = 5) :
  (1 - probability_of_rain) ^ days = 1/243 :=
sorry

end NUMINAMATH_CALUDE_no_rain_probability_l3675_367516


namespace NUMINAMATH_CALUDE_composite_number_l3675_367539

theorem composite_number : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (2^17 + 2^5 - 1 = a * b) := by
  sorry

end NUMINAMATH_CALUDE_composite_number_l3675_367539


namespace NUMINAMATH_CALUDE_allan_plum_count_l3675_367587

/-- The number of plums Sharon has -/
def sharon_plums : ℕ := 7

/-- The difference between Sharon's plums and Allan's plums -/
def plum_difference : ℕ := 3

/-- The number of plums Allan has -/
def allan_plums : ℕ := sharon_plums - plum_difference

theorem allan_plum_count : allan_plums = 4 := by
  sorry

end NUMINAMATH_CALUDE_allan_plum_count_l3675_367587


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l3675_367556

def letter_count : ℕ := 26
def digit_count : ℕ := 10
def plate_length : ℕ := 4

def is_palindrome (s : List α) : Prop :=
  s = s.reverse

def prob_palindrome_letters : ℚ :=
  (letter_count ^ 2 : ℚ) / (letter_count ^ plate_length)

def prob_palindrome_digits : ℚ :=
  (digit_count ^ 2 : ℚ) / (digit_count ^ plate_length)

theorem license_plate_palindrome_probability :
  let prob := prob_palindrome_letters + prob_palindrome_digits - 
              prob_palindrome_letters * prob_palindrome_digits
  prob = 775 / 67600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l3675_367556


namespace NUMINAMATH_CALUDE_fourth_power_sum_l3675_367594

theorem fourth_power_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (sum_squares : a^2 + b^2 + c^2 = 5)
  (sum_cubes : a^3 + b^3 + c^3 = 8) : 
  a^4 + b^4 + c^4 = 18.5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l3675_367594


namespace NUMINAMATH_CALUDE_trivia_team_selection_l3675_367541

/-- The number of students not picked for a trivia team --/
def students_not_picked (total : ℕ) (groups : ℕ) (per_group : ℕ) : ℕ :=
  total - (groups * per_group)

/-- Theorem: Given 65 total students, 8 groups, and 6 students per group,
    17 students were not picked for the trivia team --/
theorem trivia_team_selection :
  students_not_picked 65 8 6 = 17 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_selection_l3675_367541


namespace NUMINAMATH_CALUDE_first_digit_389_base4_is_1_l3675_367508

-- Define a function to convert a number to its base-4 representation
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

-- Theorem statement
theorem first_digit_389_base4_is_1 :
  (toBase4 389).reverse.head? = some 1 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_389_base4_is_1_l3675_367508


namespace NUMINAMATH_CALUDE_prob_two_qualified_bottles_l3675_367507

/-- The probability of a single bottle of beverage being qualified -/
def qualified_rate : ℝ := 0.8

/-- The probability of two bottles both being qualified -/
def both_qualified_prob : ℝ := qualified_rate * qualified_rate

/-- Theorem: The probability of drinking two qualified bottles is 0.64 -/
theorem prob_two_qualified_bottles : both_qualified_prob = 0.64 := by sorry

end NUMINAMATH_CALUDE_prob_two_qualified_bottles_l3675_367507


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l3675_367528

theorem set_equality_implies_values (a b : ℝ) : 
  ({1, a, b} : Set ℝ) = {a, a^2, a*b} → a = -1 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l3675_367528


namespace NUMINAMATH_CALUDE_solve_pocket_money_problem_l3675_367558

def pocket_money_problem (P : ℝ) : Prop :=
  let tteokbokki_cost : ℝ := P / 2
  let remaining_after_tteokbokki : ℝ := P - tteokbokki_cost
  let pencil_cost : ℝ := (3 / 8) * remaining_after_tteokbokki
  let final_remaining : ℝ := remaining_after_tteokbokki - pencil_cost
  (final_remaining = 2500) → (tteokbokki_cost = 4000)

theorem solve_pocket_money_problem :
  ∃ P : ℝ, pocket_money_problem P :=
sorry

end NUMINAMATH_CALUDE_solve_pocket_money_problem_l3675_367558


namespace NUMINAMATH_CALUDE_girls_in_class_l3675_367570

theorem girls_in_class (total : ℕ) (prob : ℚ) (boys : ℕ) (girls : ℕ) : 
  total = 25 →
  prob = 3 / 25 →
  boys + girls = total →
  (boys.choose 2 : ℚ) / (total.choose 2 : ℚ) = prob →
  girls = 16 :=
sorry

end NUMINAMATH_CALUDE_girls_in_class_l3675_367570


namespace NUMINAMATH_CALUDE_multiply_sum_power_l3675_367588

theorem multiply_sum_power (n : ℕ) (h : n > 0) :
  n * (n^n + 1) = n^(n + 1) + n :=
by sorry

end NUMINAMATH_CALUDE_multiply_sum_power_l3675_367588


namespace NUMINAMATH_CALUDE_smallest_integer_with_conditions_l3675_367569

/-- Represents a natural number as a list of its digits in reverse order -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  have : n / 10 < n := sorry
  (n % 10) :: digits (n / 10)

/-- Checks if the digits of a number are in strictly increasing order -/
def increasing_digits (n : ℕ) : Prop :=
  List.Pairwise (· < ·) (digits n)

/-- Calculates the sum of squares of digits of a number -/
def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  (digits n).map (λ d => d * d) |> List.sum

/-- Calculates the product of digits of a number -/
def product_of_digits (n : ℕ) : ℕ :=
  (digits n).prod

/-- The main theorem -/
theorem smallest_integer_with_conditions :
  ∃ n : ℕ,
    (∀ m : ℕ, m < n →
      (sum_of_squares_of_digits m ≠ 85 ∨
       ¬increasing_digits m)) ∧
    sum_of_squares_of_digits n = 85 ∧
    increasing_digits n ∧
    product_of_digits n = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_conditions_l3675_367569


namespace NUMINAMATH_CALUDE_correct_arrangements_l3675_367580

/-- The number of ways to arrange 4 teachers and 4 students in a line. -/
def arrangement_count (n m : ℕ) : ℕ × ℕ × ℕ := sorry

/-- The correct arrangement counts for 4 teachers and 4 students. -/
theorem correct_arrangements :
  arrangement_count 4 4 = (2880, 2880, 1152) := by sorry

end NUMINAMATH_CALUDE_correct_arrangements_l3675_367580


namespace NUMINAMATH_CALUDE_no_rain_probability_l3675_367563

theorem no_rain_probability (p : ℝ) (n : ℕ) (h1 : p = 2/3) (h2 : n = 5) :
  (1 - p)^n = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l3675_367563


namespace NUMINAMATH_CALUDE_det_of_specific_matrix_l3675_367521

theorem det_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![6, -2; -3, 5]
  Matrix.det A = 24 := by
sorry

end NUMINAMATH_CALUDE_det_of_specific_matrix_l3675_367521


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_less_than_one_l3675_367582

/-- A function f: ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def Decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The function f(x) = (a-1)x + 1 -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ (a - 1) * x + 1

theorem decreasing_function_implies_a_less_than_one (a : ℝ) :
  Decreasing (f a) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_less_than_one_l3675_367582


namespace NUMINAMATH_CALUDE_travel_time_calculation_l3675_367517

/-- Given a person travels 2 miles in 8 minutes, prove they will travel 5 miles in 20 minutes at the same rate. -/
theorem travel_time_calculation (distance_1 : ℝ) (time_1 : ℝ) (distance_2 : ℝ) 
  (h1 : distance_1 = 2) 
  (h2 : time_1 = 8) 
  (h3 : distance_2 = 5) :
  (distance_2 / (distance_1 / time_1)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l3675_367517


namespace NUMINAMATH_CALUDE_not_all_perfect_squares_l3675_367599

theorem not_all_perfect_squares (a b c : ℕ+) : 
  ¬(∃ (x y z : ℕ), x^2 = a^2 + b + c ∧ y^2 = b^2 + c + a ∧ z^2 = c^2 + a + b) := by
  sorry

end NUMINAMATH_CALUDE_not_all_perfect_squares_l3675_367599


namespace NUMINAMATH_CALUDE_common_roots_product_l3675_367540

theorem common_roots_product (A B : ℝ) : 
  (∃ p q r s : ℂ, 
    (p^3 + A*p + 10 = 0) ∧ 
    (q^3 + A*q + 10 = 0) ∧ 
    (r^3 + A*r + 10 = 0) ∧
    (p^3 + B*p^2 + 50 = 0) ∧ 
    (q^3 + B*q^2 + 50 = 0) ∧ 
    (s^3 + B*s^2 + 50 = 0) ∧
    (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧ (p ≠ s) ∧ (q ≠ s)) →
  (∃ p q : ℂ, 
    (p^3 + A*p + 10 = 0) ∧ 
    (q^3 + A*q + 10 = 0) ∧
    (p^3 + B*p^2 + 50 = 0) ∧ 
    (q^3 + B*q^2 + 50 = 0) ∧
    (p*q = 5 * (4^(1/3)))) := by
sorry

end NUMINAMATH_CALUDE_common_roots_product_l3675_367540


namespace NUMINAMATH_CALUDE_smallest_coin_count_l3675_367512

def is_valid_coin_combination (dimes quarters : ℕ) : Prop :=
  dimes * 10 + quarters * 25 = 265 ∧ dimes > quarters

def coin_count (dimes quarters : ℕ) : ℕ :=
  dimes + quarters

theorem smallest_coin_count : 
  (∃ d q : ℕ, is_valid_coin_combination d q) ∧ 
  (∀ d q : ℕ, is_valid_coin_combination d q → coin_count d q ≥ 16) ∧
  (∃ d q : ℕ, is_valid_coin_combination d q ∧ coin_count d q = 16) :=
sorry

end NUMINAMATH_CALUDE_smallest_coin_count_l3675_367512


namespace NUMINAMATH_CALUDE_factorial_sum_equals_natural_sum_squared_l3675_367523

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (k : ℕ) : ℕ := (List.range k).map factorial |>.sum

def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

theorem factorial_sum_equals_natural_sum_squared :
  ∀ k n : ℕ, sum_of_factorials k = (sum_of_naturals n)^2 ↔ (k = 1 ∧ n = 1) ∨ (k = 3 ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_natural_sum_squared_l3675_367523


namespace NUMINAMATH_CALUDE_max_roses_for_680_l3675_367572

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : ℚ  -- Price of an individual rose
  dozen : ℚ       -- Price of a dozen roses
  twoDozen : ℚ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased given a budget and pricing options -/
def maxRoses (budget : ℚ) (pricing : RosePricing) : ℕ :=
  sorry

/-- The specific pricing for the problem -/
def problemPricing : RosePricing :=
  { individual := 5.3
  , dozen := 36
  , twoDozen := 50 }

theorem max_roses_for_680 :
  maxRoses 680 problemPricing = 317 := by
  sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l3675_367572


namespace NUMINAMATH_CALUDE_paint_for_similar_statues_l3675_367551

-- Define the height and paint amount for the original statue
def original_height : ℝ := 8
def original_paint : ℝ := 2

-- Define the height and number of new statues
def new_height : ℝ := 2
def num_new_statues : ℕ := 360

-- Theorem statement
theorem paint_for_similar_statues :
  let surface_area_ratio := (new_height / original_height) ^ 2
  let paint_per_new_statue := original_paint * surface_area_ratio
  let total_paint := num_new_statues * paint_per_new_statue
  total_paint = 45 := by sorry

end NUMINAMATH_CALUDE_paint_for_similar_statues_l3675_367551


namespace NUMINAMATH_CALUDE_dodgeball_team_size_l3675_367505

/-- Given a dodgeball team with the following conditions:
  * The team scored 39 points total
  * One player (Emily) scored 23 points
  * Everyone else scored 2 points each
  This theorem proves that the total number of players on the team is 9. -/
theorem dodgeball_team_size :
  ∀ (total_points : ℕ) (emily_points : ℕ) (points_per_other : ℕ),
    total_points = 39 →
    emily_points = 23 →
    points_per_other = 2 →
    ∃ (team_size : ℕ),
      team_size = (total_points - emily_points) / points_per_other + 1 ∧
      team_size = 9 :=
by sorry

end NUMINAMATH_CALUDE_dodgeball_team_size_l3675_367505


namespace NUMINAMATH_CALUDE_partition_five_elements_l3675_367595

/-- The number of ways to partition a set of 5 elements into two non-empty subsets, 
    where two specific elements must be in the same subset -/
def partitionWays : ℕ := 6

/-- A function that calculates the number of ways to partition a set of n elements into two non-empty subsets,
    where two specific elements must be in the same subset -/
def partitionFunction (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 2)

theorem partition_five_elements :
  partitionWays = partitionFunction 5 :=
by sorry

end NUMINAMATH_CALUDE_partition_five_elements_l3675_367595


namespace NUMINAMATH_CALUDE_otimes_self_otimes_self_l3675_367564

def otimes (x y : ℝ) : ℝ := x^2 - y^2

theorem otimes_self_otimes_self (h : ℝ) : otimes h (otimes h h) = h^2 := by
  sorry

end NUMINAMATH_CALUDE_otimes_self_otimes_self_l3675_367564


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l3675_367598

/-- Represents the profit function for a product with given pricing conditions -/
def profit_function (x : ℝ) : ℝ := -x^2 + 190*x - 7800

/-- The optimal selling price that maximizes profit -/
def optimal_price : ℝ := 95

/-- Theorem stating that the optimal price maximizes the profit function -/
theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, 60 ≤ x ∧ x ≤ 130 → profit_function x ≤ profit_function optimal_price :=
by sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l3675_367598


namespace NUMINAMATH_CALUDE_sandwiches_bought_l3675_367591

theorem sandwiches_bought (sandwich_cost : ℝ) (soda_count : ℕ) (soda_cost : ℝ) (total_cost : ℝ)
  (h1 : sandwich_cost = 2.44)
  (h2 : soda_count = 4)
  (h3 : soda_cost = 0.87)
  (h4 : total_cost = 8.36)
  : ∃ (sandwich_count : ℕ), 
    sandwich_count * sandwich_cost + soda_count * soda_cost = total_cost ∧ 
    sandwich_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_bought_l3675_367591


namespace NUMINAMATH_CALUDE_complex_value_theorem_l3675_367530

theorem complex_value_theorem (z : ℂ) (h : (1 - z) / (1 + z) = I) : 
  Complex.abs (z + 1) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_value_theorem_l3675_367530


namespace NUMINAMATH_CALUDE_hank_total_donation_l3675_367552

def carwash_earnings : ℝ := 100
def carwash_donation_percentage : ℝ := 0.90
def bake_sale_earnings : ℝ := 80
def bake_sale_donation_percentage : ℝ := 0.75
def lawn_mowing_earnings : ℝ := 50
def lawn_mowing_donation_percentage : ℝ := 1.00

def total_donation : ℝ := 
  carwash_earnings * carwash_donation_percentage +
  bake_sale_earnings * bake_sale_donation_percentage +
  lawn_mowing_earnings * lawn_mowing_donation_percentage

theorem hank_total_donation : total_donation = 200 := by
  sorry

end NUMINAMATH_CALUDE_hank_total_donation_l3675_367552


namespace NUMINAMATH_CALUDE_parabola_directrix_l3675_367501

/-- Given a parabola with equation y = ax² and directrix y = -1, prove that a = 1/4 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 → (∃ k : ℝ, y = -1/4/k ∧ k = a)) → 
  a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3675_367501


namespace NUMINAMATH_CALUDE_soccer_team_wins_l3675_367568

/-- Given a soccer team that played 140 games and won 50 percent of them, 
    prove that the number of games won is 70. -/
theorem soccer_team_wins (total_games : ℕ) (win_percentage : ℚ) (games_won : ℕ) : 
  total_games = 140 → 
  win_percentage = 1/2 → 
  games_won = (total_games : ℚ) * win_percentage → 
  games_won = 70 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_wins_l3675_367568


namespace NUMINAMATH_CALUDE_consecutive_draw_probability_l3675_367585

def num_purple_chips : ℕ := 4
def num_orange_chips : ℕ := 3
def num_green_chips : ℕ := 5
def total_chips : ℕ := num_purple_chips + num_orange_chips + num_green_chips

def probability_consecutive_draw : ℚ :=
  (Nat.factorial 2 * Nat.factorial num_purple_chips * Nat.factorial num_orange_chips * Nat.factorial num_green_chips) /
  Nat.factorial total_chips

theorem consecutive_draw_probability :
  probability_consecutive_draw = 1 / 13860 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_draw_probability_l3675_367585


namespace NUMINAMATH_CALUDE_power_of_power_l3675_367536

theorem power_of_power (a : ℝ) : (a^4)^3 = a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3675_367536


namespace NUMINAMATH_CALUDE_quiz_probability_theorem_l3675_367554

/-- The number of questions in the quiz -/
def total_questions : ℕ := 30

/-- The number of answer choices for each question -/
def choices_per_question : ℕ := 6

/-- The number of questions Emily guesses randomly -/
def guessed_questions : ℕ := 5

/-- The probability of guessing a single question correctly -/
def prob_correct : ℚ := 1 / choices_per_question

/-- The probability of guessing a single question incorrectly -/
def prob_incorrect : ℚ := 1 - prob_correct

/-- The probability of guessing at least two out of five questions correctly -/
def prob_at_least_two_correct : ℚ := 763 / 3888

theorem quiz_probability_theorem :
  (1 : ℚ) - (prob_incorrect ^ guessed_questions + 
    (guessed_questions : ℚ) * prob_correct * prob_incorrect ^ (guessed_questions - 1)) = 
  prob_at_least_two_correct :=
sorry

end NUMINAMATH_CALUDE_quiz_probability_theorem_l3675_367554


namespace NUMINAMATH_CALUDE_min_nSn_l3675_367504

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℤ  -- The sequence
  S : ℕ+ → ℤ  -- Sum of first n terms
  h4 : S 4 = -2
  h5 : S 5 = 0
  h6 : S 6 = 3

/-- The product of n and S_n -/
def nSn (seq : ArithmeticSequence) (n : ℕ+) : ℤ :=
  n * seq.S n

theorem min_nSn (seq : ArithmeticSequence) :
  ∃ (m : ℕ+), ∀ (n : ℕ+), nSn seq m ≤ nSn seq n ∧ nSn seq m = -9 :=
sorry

end NUMINAMATH_CALUDE_min_nSn_l3675_367504


namespace NUMINAMATH_CALUDE_men_in_second_group_l3675_367515

/-- Given the conditions of the problem, prove that the number of men in the second group is 9 -/
theorem men_in_second_group : 
  let first_group_men : ℕ := 4
  let first_group_hours_per_day : ℕ := 10
  let first_group_earnings : ℕ := 1200
  let second_group_hours_per_day : ℕ := 6
  let second_group_earnings : ℕ := 1620
  let days_per_week : ℕ := 7
  
  ∃ (second_group_men : ℕ),
    second_group_men * second_group_hours_per_day * days_per_week * first_group_earnings = 
    first_group_men * first_group_hours_per_day * days_per_week * second_group_earnings ∧
    second_group_men = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_men_in_second_group_l3675_367515


namespace NUMINAMATH_CALUDE_hikers_speed_hikers_speed_specific_l3675_367533

/-- The problem of determining a hiker's speed given specific conditions involving a cyclist -/
theorem hikers_speed (cyclist_speed : ℝ) (cyclist_travel_time : ℝ) (hiker_catch_up_time : ℝ) : ℝ :=
  let hiker_speed := (cyclist_speed * cyclist_travel_time) / hiker_catch_up_time
  by
    -- Assuming:
    -- 1. The hiker walks at a constant rate.
    -- 2. A cyclist passes the hiker, traveling in the same direction at 'cyclist_speed'.
    -- 3. The cyclist stops after 'cyclist_travel_time'.
    -- 4. The hiker continues walking at her constant rate.
    -- 5. The cyclist waits 'hiker_catch_up_time' until the hiker catches up.
    
    -- Prove: hiker_speed = 20/3

    sorry

/-- The specific instance of the hiker's speed problem -/
theorem hikers_speed_specific : hikers_speed 20 (1/12) (1/4) = 20/3 :=
  by sorry

end NUMINAMATH_CALUDE_hikers_speed_hikers_speed_specific_l3675_367533


namespace NUMINAMATH_CALUDE_parabola_vertex_l3675_367596

/-- The vertex of a parabola defined by y^2 + 8y + 4x + 5 = 0 is (11/4, -4) -/
theorem parabola_vertex : 
  let f (x y : ℝ) := y^2 + 8*y + 4*x + 5
  ∃! (vx vy : ℝ), (∀ (x y : ℝ), f x y = 0 → (x - vx)^2 ≥ 0) ∧ vx = 11/4 ∧ vy = -4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3675_367596


namespace NUMINAMATH_CALUDE_g_sum_zero_l3675_367597

def g (x : ℝ) : ℝ := x^2 - 2013*x

theorem g_sum_zero (a b : ℝ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_zero_l3675_367597


namespace NUMINAMATH_CALUDE_zoo_trip_result_l3675_367506

def zoo_trip (initial_students_class1 initial_students_class2 parent_chaperones teachers students_left chaperones_left : ℕ) : ℕ :=
  let total_initial_students := initial_students_class1 + initial_students_class2
  let total_initial_adults := parent_chaperones + teachers
  let remaining_students := total_initial_students - students_left
  let remaining_chaperones := parent_chaperones - chaperones_left
  remaining_students + remaining_chaperones + teachers

theorem zoo_trip_result :
  zoo_trip 10 10 5 2 10 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_result_l3675_367506


namespace NUMINAMATH_CALUDE_simons_age_is_45_l3675_367548

/-- Simon's age in 2010, given Jorge's age in 2005 and the age difference between Simon and Jorge -/
def simons_age_2010 (jorges_age_2005 : ℕ) (age_difference : ℕ) : ℕ :=
  jorges_age_2005 + (2010 - 2005) + age_difference

/-- Theorem stating that Simon's age in 2010 is 45 years old -/
theorem simons_age_is_45 :
  simons_age_2010 16 24 = 45 := by
  sorry

end NUMINAMATH_CALUDE_simons_age_is_45_l3675_367548


namespace NUMINAMATH_CALUDE_paint_stones_l3675_367549

def canPaintAllBlack (k : Nat) : Prop :=
  1 ≤ k ∧ k ≤ 50 ∧ Nat.gcd 100 (k - 1) = 1

theorem paint_stones (k : Nat) :
  canPaintAllBlack k ↔ ¬∃m : Nat, m ∈ Finset.range 13 ∧ k = 4 * m + 1 :=
by sorry

end NUMINAMATH_CALUDE_paint_stones_l3675_367549


namespace NUMINAMATH_CALUDE_smallest_advantageous_discount_l3675_367557

def is_more_advantageous (n : ℕ) : Prop :=
  (1 - n / 100 : ℝ) < (1 - 0.2)^2 ∧
  (1 - n / 100 : ℝ) < (1 - 0.15)^3 ∧
  (1 - n / 100 : ℝ) < (1 - 0.3) * (1 - 0.1)

theorem smallest_advantageous_discount : 
  (∀ m : ℕ, m < 39 → ¬(is_more_advantageous m)) ∧ 
  is_more_advantageous 39 := by
  sorry

end NUMINAMATH_CALUDE_smallest_advantageous_discount_l3675_367557


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3675_367543

/-- The scale factor of the map, representing km per cm -/
def scale : ℝ := 10

/-- The distance between Stockholm and Uppsala on the map in cm -/
def map_distance : ℝ := 35

/-- The actual distance between Stockholm and Uppsala in km -/
def actual_distance : ℝ := map_distance * scale

theorem stockholm_uppsala_distance : actual_distance = 350 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3675_367543


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l3675_367584

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2018 = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l3675_367584


namespace NUMINAMATH_CALUDE_cycle_selling_price_l3675_367502

/-- The selling price of a cycle after applying successive discounts -/
def selling_price (original_price : ℝ) (discount1 discount2 discount3 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)

/-- Theorem: The selling price of a cycle originally priced at Rs. 3,600, 
    after applying successive discounts of 15%, 10%, and 5%, is equal to Rs. 2,616.30 -/
theorem cycle_selling_price :
  selling_price 3600 0.15 0.10 0.05 = 2616.30 := by
  sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l3675_367502


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3675_367589

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given condition for the arithmetic sequence -/
def SequenceCondition (a : ℕ → ℝ) : Prop :=
  a 2 + 2 * a 6 + a 10 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SequenceCondition a) : 
  a 3 + a 9 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3675_367589


namespace NUMINAMATH_CALUDE_root_sum_product_l3675_367550

def complex_plane : Type := ℂ

def coordinates (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem root_sum_product (z : ℂ) (p q : ℝ) :
  coordinates z = (-1, 3) →
  (z^2 + p*z + q = 0) →
  p + q = 12 := by sorry

end NUMINAMATH_CALUDE_root_sum_product_l3675_367550


namespace NUMINAMATH_CALUDE_equation_solution_l3675_367575

theorem equation_solution (p m z : ℤ) : 
  Prime p ∧ m > 0 ∧ z < 0 ∧ p^3 + p*m + 2*z*m = m^2 + p*z + z^2 ↔ 
  (p = 2 ∧ m = 4 + z ∧ (z = -1 ∨ z = -2 ∨ z = -3)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3675_367575


namespace NUMINAMATH_CALUDE_complex_power_2019_l3675_367590

theorem complex_power_2019 (i : ℂ) (h : i^2 = -1) : i^2019 = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_2019_l3675_367590


namespace NUMINAMATH_CALUDE_complement_A_in_U_l3675_367555

open Set

-- Define the universal set U
def U : Set ℝ := {x | x^2 > 1}

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Theorem statement
theorem complement_A_in_U :
  (U \ A) = {x : ℝ | x ≥ 3 ∨ x < -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l3675_367555


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3675_367567

-- Define the inequality
def inequality (x k : ℝ) : Prop := x^2 > (k+1)*x - k

-- Define the solution set
def solution_set (k : ℝ) : Set ℝ :=
  if k > 1 then {x : ℝ | x < 1 ∨ x > k}
  else if k = 1 then {x : ℝ | x ≠ 1}
  else {x : ℝ | x < k ∨ x > 1}

-- Theorem statement
theorem inequality_solution_set (k : ℝ) :
  {x : ℝ | inequality x k} = solution_set k :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3675_367567


namespace NUMINAMATH_CALUDE_min_value_inequality_l3675_367518

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1/a + 1/b + 2 * Real.sqrt (a * b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3675_367518


namespace NUMINAMATH_CALUDE_base8_sum_l3675_367511

/-- Base 8 representation of a three-digit number -/
def base8Rep (x y z : ℕ) : ℕ := 64 * x + 8 * y + z

/-- Proposition: If X, Y, and Z are non-zero distinct digits in base 8 such that 
    XYZ₈ + YZX₈ + ZXY₈ = XXX0₈, then Y + Z = 7₈ -/
theorem base8_sum (X Y Z : ℕ) 
  (h1 : X ≠ 0 ∧ Y ≠ 0 ∧ Z ≠ 0)
  (h2 : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z)
  (h3 : X < 8 ∧ Y < 8 ∧ Z < 8)
  (h4 : base8Rep X Y Z + base8Rep Y Z X + base8Rep Z X Y = 8 * base8Rep X X X) :
  Y + Z = 7 := by
sorry

end NUMINAMATH_CALUDE_base8_sum_l3675_367511


namespace NUMINAMATH_CALUDE_joan_video_game_spending_l3675_367535

/-- The cost of the basketball game Joan purchased -/
def basketball_cost : ℚ := 5.2

/-- The cost of the racing game Joan purchased -/
def racing_cost : ℚ := 4.23

/-- The total amount Joan spent on video games -/
def total_spent : ℚ := basketball_cost + racing_cost

/-- Theorem stating that the total amount Joan spent on video games is $9.43 -/
theorem joan_video_game_spending :
  total_spent = 9.43 := by sorry

end NUMINAMATH_CALUDE_joan_video_game_spending_l3675_367535


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3675_367562

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- {an} is a geometric sequence with common ratio q
  q > 0 →                       -- q is positive
  a 2 = 1 →                     -- a2 = 1
  a 4 = 4 →                     -- a4 = 4
  q = 2 :=                      -- prove q = 2
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3675_367562


namespace NUMINAMATH_CALUDE_odd_function_property_l3675_367503

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f : ℝ → ℝ is increasing on [a,b] if x₁ ≤ x₂ implies f x₁ ≤ f x₂ for all x₁, x₂ in [a,b] -/
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ b → f x₁ ≤ f x₂

theorem odd_function_property (f : ℝ → ℝ) :
  IsOdd f →
  IncreasingOn f 3 7 →
  (∀ x ∈ Set.Icc 3 6, f x ≤ 8) →
  (∀ x ∈ Set.Icc 3 6, 1 ≤ f x) →
  f (-3) + 2 * f 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3675_367503


namespace NUMINAMATH_CALUDE_binary_arithmetic_proof_l3675_367571

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

theorem binary_arithmetic_proof :
  let a := [true, true, false, true, true]  -- 11011₂
  let b := [true, false, true]              -- 101₂
  let c := [false, true, false, true]       -- 1010₂
  let product := binary_to_decimal a * binary_to_decimal b
  let result := product - binary_to_decimal c
  decimal_to_binary result = [true, false, true, true, true, true, true] -- 1111101₂
  := by sorry

end NUMINAMATH_CALUDE_binary_arithmetic_proof_l3675_367571


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3675_367513

theorem polynomial_coefficient_sum (a b c d : ℤ) :
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 - 2*x^3 + 3*x^2 + 4*x - 10) →
  a + b + c + d = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3675_367513


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3675_367524

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^5 + 2*x^3 + x + 3) % (x - 2) = 53 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3675_367524


namespace NUMINAMATH_CALUDE_preimage_of_20_l3675_367566

def A : Set ℕ := sorry
def B : Set ℕ := sorry

def f (n : ℕ) : ℕ := 2 * n^2

theorem preimage_of_20 : ∃ (n : ℕ), n ∈ A ∧ f n = 20 ∧ (∀ (m : ℕ), m ∈ A ∧ f m = 20 → m = n) :=
  sorry

end NUMINAMATH_CALUDE_preimage_of_20_l3675_367566


namespace NUMINAMATH_CALUDE_selection_theorem_l3675_367526

/-- Represents the number of students with each skill -/
structure StudentGroup where
  total : ℕ
  singers : ℕ
  dancers : ℕ
  both : ℕ

/-- Represents the selection requirements -/
structure SelectionRequirement where
  singersToSelect : ℕ
  dancersToSelect : ℕ

/-- Calculates the number of ways to select students given a student group and selection requirements -/
def numberOfWaysToSelect (group : StudentGroup) (req : SelectionRequirement) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem selection_theorem (group : StudentGroup) (req : SelectionRequirement) :
  group.total = 6 ∧ 
  group.singers = 3 ∧ 
  group.dancers = 2 ∧ 
  group.both = 1 ∧
  req.singersToSelect = 2 ∧
  req.dancersToSelect = 1 →
  numberOfWaysToSelect group req = 15 :=
by sorry

end NUMINAMATH_CALUDE_selection_theorem_l3675_367526


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_eight_l3675_367592

theorem sum_of_solutions_eq_eight : 
  ∃ (x y : ℝ), x * (x - 8) = 7 ∧ y * (y - 8) = 7 ∧ x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_eight_l3675_367592


namespace NUMINAMATH_CALUDE_income_left_percentage_man_income_left_l3675_367586

/-- Given a man's spending pattern, calculate the percentage of income left --/
theorem income_left_percentage (total_income : ℝ) (food_percent : ℝ) (education_percent : ℝ) 
  (transport_percent : ℝ) (rent_percent : ℝ) : ℝ :=
  let initial_expenses := food_percent + education_percent + transport_percent
  let remaining_after_initial := 100 - initial_expenses
  let rent_amount := rent_percent * remaining_after_initial / 100
  let total_expenses := initial_expenses + rent_amount
  100 - total_expenses

/-- Prove that the man is left with 12.6% of his income --/
theorem man_income_left :
  income_left_percentage 100 42 18 12 55 = 12.6 := by
  sorry

end NUMINAMATH_CALUDE_income_left_percentage_man_income_left_l3675_367586


namespace NUMINAMATH_CALUDE_no_prime_intercept_lines_through_point_l3675_367532

-- Define a prime number
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define a line with intercepts
def Line (a b : ℕ) := {(x, y) : ℝ × ℝ | x / a + y / b = 1}

-- Theorem statement
theorem no_prime_intercept_lines_through_point :
  ¬∃ (a b : ℕ), isPrime a ∧ isPrime b ∧ (6, 5) ∈ Line a b := by
  sorry

end NUMINAMATH_CALUDE_no_prime_intercept_lines_through_point_l3675_367532


namespace NUMINAMATH_CALUDE_union_subset_implies_m_leq_three_l3675_367583

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 6}
def C (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m}

theorem union_subset_implies_m_leq_three (m : ℝ) :
  B ∪ C m = B → m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_subset_implies_m_leq_three_l3675_367583


namespace NUMINAMATH_CALUDE_little_john_friends_money_l3675_367522

/-- Calculates the amount given to each friend by Little John --/
theorem little_john_friends_money 
  (initial_amount : ℚ) 
  (sweets_cost : ℚ) 
  (num_friends : ℕ) 
  (remaining_amount : ℚ) 
  (h1 : initial_amount = 8.5)
  (h2 : sweets_cost = 1.25)
  (h3 : num_friends = 2)
  (h4 : remaining_amount = 4.85) :
  (initial_amount - remaining_amount - sweets_cost) / num_friends = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_little_john_friends_money_l3675_367522


namespace NUMINAMATH_CALUDE_expansion_theorem_l3675_367525

-- Define the sum of coefficients for (3x + √x)^n
def sumCoefficients (n : ℕ) : ℝ := 4^n

-- Define the sum of binomial coefficients
def sumBinomialCoefficients (n : ℕ) : ℝ := 2^n

-- Define the condition M - N = 240
def conditionSatisfied (n : ℕ) : Prop :=
  sumCoefficients n - sumBinomialCoefficients n = 240

-- Define the rational terms in the expansion
def rationalTerms (n : ℕ) : List (ℝ × ℕ) :=
  [(81, 4), (54, 3), (1, 2)]

theorem expansion_theorem :
  ∃ n : ℕ, conditionSatisfied n ∧ 
  n = 4 ∧
  rationalTerms n = [(81, 4), (54, 3), (1, 2)] :=
sorry

end NUMINAMATH_CALUDE_expansion_theorem_l3675_367525


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3675_367542

/-- The polynomial z^5 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^5 - z^3 + 1

/-- n-th roots of unity -/
def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

/-- All roots of f are n-th roots of unity -/
def all_roots_are_nth_roots_of_unity (n : ℕ) : Prop :=
  ∀ z : ℂ, f z = 0 → is_nth_root_of_unity z n

theorem smallest_n_for_roots_of_unity :
  (∃ n : ℕ, n > 0 ∧ all_roots_are_nth_roots_of_unity n) ∧
  (∀ m : ℕ, m > 0 ∧ all_roots_are_nth_roots_of_unity m → m ≥ 30) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3675_367542


namespace NUMINAMATH_CALUDE_water_evaporation_rate_l3675_367561

/-- Proves that given a glass filled with 10 ounces of water, and 6% of the water
    evaporating over a 30-day period, the amount of water evaporated each day is 0.02 ounces. -/
theorem water_evaporation_rate (initial_water : ℝ) (days : ℕ) (evaporation_percentage : ℝ) :
  initial_water = 10 →
  days = 30 →
  evaporation_percentage = 6 →
  (initial_water * evaporation_percentage / 100) / days = 0.02 := by
  sorry


end NUMINAMATH_CALUDE_water_evaporation_rate_l3675_367561


namespace NUMINAMATH_CALUDE_prob_six_odd_in_eight_rolls_l3675_367529

/-- A fair 6-sided die -/
def fair_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability of rolling an odd number on a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The number of odd results we're interested in -/
def target_odd : ℕ := 6

/-- The probability of getting exactly 6 odd results in 8 rolls of a fair 6-sided die -/
theorem prob_six_odd_in_eight_rolls :
  (Nat.choose num_rolls target_odd : ℚ) * prob_odd^target_odd * (1 - prob_odd)^(num_rolls - target_odd) = 28/256 := by
  sorry

end NUMINAMATH_CALUDE_prob_six_odd_in_eight_rolls_l3675_367529


namespace NUMINAMATH_CALUDE_sets_are_equal_l3675_367573

def M : Set ℝ := {y | ∃ x, y = x^2 + 3}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 3)}

theorem sets_are_equal : M = N := by sorry

end NUMINAMATH_CALUDE_sets_are_equal_l3675_367573


namespace NUMINAMATH_CALUDE_browser_usage_inconsistency_l3675_367547

theorem browser_usage_inconsistency (total_A : ℕ) (total_B : ℕ) (both : ℕ) (only_one : ℕ) :
  total_A = 316 →
  total_B = 478 →
  both = 104 →
  only_one = 567 →
  (total_A - both) + (total_B - both) ≠ only_one :=
by
  sorry

end NUMINAMATH_CALUDE_browser_usage_inconsistency_l3675_367547


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3675_367510

theorem cubic_root_sum (α β γ : ℂ) : 
  α^3 - α - 1 = 0 → β^3 - β - 1 = 0 → γ^3 - γ - 1 = 0 →
  (1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3675_367510


namespace NUMINAMATH_CALUDE_beavers_working_l3675_367520

theorem beavers_working (initial_beavers : ℕ) (swimming_beavers : ℕ) : 
  initial_beavers = 2 → swimming_beavers = 1 → initial_beavers - swimming_beavers = 1 := by
  sorry

end NUMINAMATH_CALUDE_beavers_working_l3675_367520


namespace NUMINAMATH_CALUDE_covered_boards_l3675_367546

/-- Represents a modified checkerboard with one corner removed. -/
structure ModifiedBoard :=
  (rows : Nat)
  (cols : Nat)

/-- Checks if a modified board can be completely covered by dominoes. -/
def can_be_covered (board : ModifiedBoard) : Prop :=
  let total_squares := board.rows * board.cols - 1
  (total_squares % 2 = 0) ∧ 
  (board.rows ≥ 2) ∧ 
  (board.cols ≥ 2)

/-- Theorem stating which modified boards can be covered. -/
theorem covered_boards :
  (can_be_covered ⟨5, 5⟩) ∧
  (can_be_covered ⟨7, 3⟩) ∧
  ¬(can_be_covered ⟨4, 5⟩) ∧
  ¬(can_be_covered ⟨6, 5⟩) ∧
  ¬(can_be_covered ⟨5, 4⟩) :=
sorry

end NUMINAMATH_CALUDE_covered_boards_l3675_367546


namespace NUMINAMATH_CALUDE_greg_situps_l3675_367579

/-- 
Given:
- For every sit-up Peter does, Greg does 4.
- Peter did 24 sit-ups.

Prove that Greg did 96 sit-ups.
-/
theorem greg_situps (peter_situps : ℕ) (greg_ratio : ℕ) : 
  peter_situps = 24 → greg_ratio = 4 → peter_situps * greg_ratio = 96 := by
  sorry

end NUMINAMATH_CALUDE_greg_situps_l3675_367579


namespace NUMINAMATH_CALUDE_festival_fruit_prices_l3675_367509

/-- Proves that given the conditions from the problem, the cost per kg of oranges is 2.2 yuan and the cost per kg of bananas is 5.4 yuan -/
theorem festival_fruit_prices :
  let orange_price : ℚ := x
  let pear_price : ℚ := x
  let apple_price : ℚ := y
  let banana_price : ℚ := y
  ∀ x y : ℚ,
  (9 * x + 10 * y = 73.8) →
  (17 * x + 6 * y = 69.8) →
  (x = 2.2 ∧ y = 5.4) :=
by
  sorry

end NUMINAMATH_CALUDE_festival_fruit_prices_l3675_367509


namespace NUMINAMATH_CALUDE_hijk_is_square_l3675_367544

-- Define the points
variable (A B C D E F G H I J K : EuclideanSpace ℝ (Fin 2))

-- Define the squares
def is_square (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the midpoint
def is_midpoint (M P Q : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem hijk_is_square 
  (h1 : is_square A B C D)
  (h2 : is_square D E F G)
  (h3 : A ≠ D ∧ B ≠ D ∧ C ≠ D ∧ E ≠ D ∧ F ≠ D ∧ G ≠ D)
  (h4 : is_midpoint H A G)
  (h5 : is_midpoint I G E)
  (h6 : is_midpoint J E C)
  (h7 : is_midpoint K C A) :
  is_square H I J K := by sorry

end NUMINAMATH_CALUDE_hijk_is_square_l3675_367544


namespace NUMINAMATH_CALUDE_banana_arrangements_l3675_367545

theorem banana_arrangements : 
  let total_letters : ℕ := 6
  let freq_b : ℕ := 1
  let freq_n : ℕ := 2
  let freq_a : ℕ := 3
  (total_letters = freq_b + freq_n + freq_a) →
  (Nat.factorial total_letters / (Nat.factorial freq_b * Nat.factorial freq_n * Nat.factorial freq_a) = 60) := by
sorry

end NUMINAMATH_CALUDE_banana_arrangements_l3675_367545


namespace NUMINAMATH_CALUDE_pen_price_calculation_l3675_367534

theorem pen_price_calculation (num_pens num_pencils total_cost pencil_avg_price : ℝ) 
  (h1 : num_pens = 30)
  (h2 : num_pencils = 75)
  (h3 : total_cost = 750)
  (h4 : pencil_avg_price = 2) : 
  (total_cost - num_pencils * pencil_avg_price) / num_pens = 20 := by
  sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l3675_367534


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_implies_fourth_plus_inverse_fourth_l3675_367559

theorem square_plus_inverse_square_implies_fourth_plus_inverse_fourth (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x^2) = 2 → x^4 + (1/x^4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_implies_fourth_plus_inverse_fourth_l3675_367559


namespace NUMINAMATH_CALUDE_expression_evaluation_l3675_367581

theorem expression_evaluation :
  let expr := 3 * 15 + 20 / 4 + 1
  let max_expr := 3 * (15 + 20 / 4 + 1)
  let min_expr := (3 * 15 + 20) / (4 + 1)
  (expr = 51) ∧ 
  (max_expr = 63) ∧ 
  (min_expr = 13) ∧
  (∀ x : ℤ, (∃ e : ℤ → ℤ, e expr = x) → (x ≤ max_expr ∧ x ≥ min_expr)) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3675_367581


namespace NUMINAMATH_CALUDE_floor_a_equals_four_l3675_367576

theorem floor_a_equals_four (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x ≥ 0) (h3 : y ≥ 0) (h4 : z ≥ 0) :
  let a := Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1)
  ⌊a⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_a_equals_four_l3675_367576


namespace NUMINAMATH_CALUDE_no_linear_term_implies_a_eq_neg_two_l3675_367500

/-- If the simplified result of (3x+2)(3x+a) does not contain a linear term of x, then a = -2 -/
theorem no_linear_term_implies_a_eq_neg_two (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, (3*x + 2) * (3*x + a) = b*x^2 + c) → a = -2 := by
sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_a_eq_neg_two_l3675_367500


namespace NUMINAMATH_CALUDE_total_pieces_eq_59_l3675_367519

/-- The number of pieces of clothing in the first load -/
def first_load : ℕ := 32

/-- The number of equal loads for the remaining clothing -/
def num_equal_loads : ℕ := 9

/-- The number of pieces of clothing in each of the equal loads -/
def pieces_per_equal_load : ℕ := 3

/-- The total number of pieces of clothing Will had to wash -/
def total_pieces : ℕ := first_load + num_equal_loads * pieces_per_equal_load

theorem total_pieces_eq_59 : total_pieces = 59 := by sorry

end NUMINAMATH_CALUDE_total_pieces_eq_59_l3675_367519
