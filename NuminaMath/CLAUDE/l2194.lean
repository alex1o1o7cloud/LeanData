import Mathlib

namespace isabellas_hair_growth_l2194_219459

/-- Isabella's hair growth problem -/
theorem isabellas_hair_growth (initial_length : ℝ) : 
  initial_length + 6 = 24 → initial_length = 18 := by
  sorry

end isabellas_hair_growth_l2194_219459


namespace square_perimeter_l2194_219486

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (2 * s = 32) → (4 * s = 64) := by
  sorry

end square_perimeter_l2194_219486


namespace unique_divisible_by_20_l2194_219406

def is_divisible_by_20 (n : ℕ) : Prop := ∃ k : ℕ, n = 20 * k

def four_digit_number (x : ℕ) : ℕ := 1000 * x + 480 + x

theorem unique_divisible_by_20 :
  ∃! x : ℕ, x < 10 ∧ is_divisible_by_20 (four_digit_number x) :=
by sorry

end unique_divisible_by_20_l2194_219406


namespace hall_volume_l2194_219464

/-- Proves that a rectangular hall with given dimensions and area equality has a volume of 972 cubic meters -/
theorem hall_volume (length width height : ℝ) : 
  length = 18 ∧ 
  width = 9 ∧ 
  2 * (length * width) = 2 * (length * height) + 2 * (width * height) → 
  length * width * height = 972 := by
  sorry

end hall_volume_l2194_219464


namespace horner_rule_v3_l2194_219485

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 + x - 3

def horner_v3 (x : ℝ) : ℝ := 
  let v0 := 2*x
  let v1 := v0*x - 3
  let v2 := v1*x + 2
  v2*x + 1

theorem horner_rule_v3 : horner_v3 2 = 12 := by sorry

end horner_rule_v3_l2194_219485


namespace symmetric_points_x_axis_l2194_219442

/-- Given two points A and B that are symmetric with respect to the x-axis,
    prove that the y-coordinate of A determines m to be 1. -/
theorem symmetric_points_x_axis (m : ℝ) : 
  let A : ℝ × ℝ := (-3, 2*m - 1)
  let B : ℝ × ℝ := (-3, -1)
  (A.1 = B.1 ∧ A.2 = -B.2) → m = 1 :=
by
  sorry

end symmetric_points_x_axis_l2194_219442


namespace range_of_f_l2194_219479

def f (x : ℕ) : ℤ := 2 * x - 3

def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 5}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 1, 3, 5, 7} := by sorry

end range_of_f_l2194_219479


namespace intersection_not_in_third_quadrant_l2194_219495

/-- The intersection point of y = 2x + m and y = -x + 3 cannot be in the third quadrant -/
theorem intersection_not_in_third_quadrant (m : ℝ) : 
  ∀ x y : ℝ, y = 2*x + m ∧ y = -x + 3 → ¬(x < 0 ∧ y < 0) :=
by sorry

end intersection_not_in_third_quadrant_l2194_219495


namespace min_value_expression_l2194_219452

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  let M := Real.sqrt (1 + 2 * a^2) + 2 * Real.sqrt ((5/12)^2 + b^2)
  ∀ x y, x > 0 ∧ y > 0 ∧ x + y = 1 →
    Real.sqrt (1 + 2 * x^2) + 2 * Real.sqrt ((5/12)^2 + y^2) ≥ 5 * Real.sqrt 34 / 12 :=
by sorry

end min_value_expression_l2194_219452


namespace right_triangle_side_length_l2194_219407

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (hyp : a^2 + b^2 = c^2) -- Pythagorean theorem
  (hyp_length : c = 5) -- Hypotenuse length
  (side_length : a = 3) -- Known side length
  : b = 4 := by
sorry

end right_triangle_side_length_l2194_219407


namespace target_probability_l2194_219403

/-- The probability of hitting a target in one shot. -/
def p : ℝ := 0.6

/-- The number of shots taken. -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in three shots. -/
def prob_at_least_two : ℝ := 3 * p^2 * (1 - p) + p^3

theorem target_probability :
  prob_at_least_two = 0.648 := by
  sorry

end target_probability_l2194_219403


namespace min_sum_fraction_l2194_219432

theorem min_sum_fraction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ 3 / Real.rpow 162 (1/3) :=
by sorry

end min_sum_fraction_l2194_219432


namespace simplify_fraction_with_sqrt_two_l2194_219427

theorem simplify_fraction_with_sqrt_two : 
  (1 / (1 + Real.sqrt 2)) * (1 / (1 - Real.sqrt 2)) = -1 := by sorry

end simplify_fraction_with_sqrt_two_l2194_219427


namespace power_multiplication_l2194_219404

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end power_multiplication_l2194_219404


namespace prime_sums_count_l2194_219455

/-- Sequence of prime numbers -/
def primes : List Nat := sorry

/-- Function to generate sums by adding primes and skipping every third -/
def generateSums (n : Nat) : List Nat :=
  sorry

/-- Check if a number is prime -/
def isPrime (n : Nat) : Bool :=
  sorry

/-- Count prime sums in the first n generated sums -/
def countPrimeSums (n : Nat) : Nat :=
  sorry

/-- Main theorem: The number of prime sums among the first 12 generated sums is 5 -/
theorem prime_sums_count : countPrimeSums 12 = 5 := by
  sorry

end prime_sums_count_l2194_219455


namespace platyfish_count_l2194_219456

/-- The number of goldfish in the tank -/
def num_goldfish : ℕ := 3

/-- The number of red balls each goldfish plays with -/
def red_balls_per_goldfish : ℕ := 10

/-- The number of white balls each platyfish plays with -/
def white_balls_per_platyfish : ℕ := 5

/-- The total number of balls in the fish tank -/
def total_balls : ℕ := 80

/-- The number of platyfish in the tank -/
def num_platyfish : ℕ := (total_balls - num_goldfish * red_balls_per_goldfish) / white_balls_per_platyfish

theorem platyfish_count : num_platyfish = 10 := by
  sorry

end platyfish_count_l2194_219456


namespace expression_equality_l2194_219401

theorem expression_equality : 4 + (-8) / (-4) - (-1) = 7 := by
  sorry

end expression_equality_l2194_219401


namespace unique_valid_quintuple_l2194_219473

/-- A quintuple of nonnegative real numbers satisfying the given conditions -/
structure ValidQuintuple where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  nonneg_a : 0 ≤ a
  nonneg_b : 0 ≤ b
  nonneg_c : 0 ≤ c
  nonneg_d : 0 ≤ d
  nonneg_e : 0 ≤ e
  condition1 : a^2 + b^2 + c^3 + d^3 + e^3 = 5
  condition2 : (a + b + c + d + e) * (a^3 + b^3 + c^2 + d^2 + e^2) = 25

/-- There exists exactly one valid quintuple -/
theorem unique_valid_quintuple : ∃! q : ValidQuintuple, True :=
  sorry

end unique_valid_quintuple_l2194_219473


namespace modulus_of_2_plus_i_times_i_l2194_219453

theorem modulus_of_2_plus_i_times_i : Complex.abs ((2 + Complex.I) * Complex.I) = Real.sqrt 5 := by
  sorry

end modulus_of_2_plus_i_times_i_l2194_219453


namespace square_equal_area_rectangle_l2194_219497

theorem square_equal_area_rectangle (rectangle_length rectangle_width square_side : ℝ) :
  rectangle_length = 25 ∧ 
  rectangle_width = 9 ∧ 
  square_side = 15 →
  rectangle_length * rectangle_width = square_side * square_side :=
by sorry

end square_equal_area_rectangle_l2194_219497


namespace one_quarter_of_6_75_l2194_219447

theorem one_quarter_of_6_75 : (6.75 : ℚ) / 4 = 27 / 16 := by sorry

end one_quarter_of_6_75_l2194_219447


namespace min_value_x_plus_y_l2194_219494

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y - x - 2 * y = 0) :
  x + y ≥ 3 + 2 * Real.sqrt 2 := by
sorry

end min_value_x_plus_y_l2194_219494


namespace hiker_route_length_l2194_219454

theorem hiker_route_length (rate_up : ℝ) (days_up : ℝ) (rate_down_factor : ℝ) : 
  rate_up = 7 →
  days_up = 2 →
  rate_down_factor = 1.5 →
  (rate_up * days_up) * rate_down_factor = 21 := by
  sorry

end hiker_route_length_l2194_219454


namespace unique_positive_solution_l2194_219400

theorem unique_positive_solution :
  ∃! x : ℚ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 :=
by
  -- The proof would go here
  sorry

end unique_positive_solution_l2194_219400


namespace marble_fraction_after_tripling_l2194_219435

theorem marble_fraction_after_tripling (total : ℝ) (h : total > 0) :
  let initial_green := (3/4 : ℝ) * total
  let initial_yellow := (1/4 : ℝ) * total
  let new_yellow := 3 * initial_yellow
  let new_total := initial_green + new_yellow
  new_yellow / new_total = (1/2 : ℝ) := by
  sorry

end marble_fraction_after_tripling_l2194_219435


namespace median_divided_triangle_area_l2194_219409

/-- Given a triangle with sides 13, 14, and 15 cm, the area of each smaller triangle
    formed by its medians is 14 cm². -/
theorem median_divided_triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) :
  let s := (a + b + c) / 2
  let total_area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  total_area / 6 = 14 := by
  sorry

end median_divided_triangle_area_l2194_219409


namespace p_arithmetic_fibonacci_subsequence_l2194_219498

/-- Definition of a p-arithmetic Fibonacci sequence -/
def pArithmeticFibonacci (p : ℕ) (v : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, v (n + 2) = v (n + 1) + v n

/-- Theorem: The terms of a p-arithmetic Fibonacci sequence whose indices are divisible by p
    form another arithmetic Fibonacci sequence -/
theorem p_arithmetic_fibonacci_subsequence (p : ℕ) (v : ℕ → ℕ) 
    (h : pArithmeticFibonacci p v) :
  ∀ n : ℕ, n ≥ 1 → v ((n - 1) * p) + v (n * p) = v ((n + 1) * p) :=
by sorry

end p_arithmetic_fibonacci_subsequence_l2194_219498


namespace solution_is_i_div_3_l2194_219414

/-- The imaginary unit i, where i^2 = -1 -/
noncomputable def i : ℂ := Complex.I

/-- The equation to be solved -/
def equation (x : ℂ) : Prop := 3 + i * x = 5 - 2 * i * x

/-- The theorem stating that i/3 is the solution to the equation -/
theorem solution_is_i_div_3 : equation (i / 3) := by
  sorry

end solution_is_i_div_3_l2194_219414


namespace rectangle_division_l2194_219437

theorem rectangle_division (w₁ h₁ w₂ h₂ : ℝ) :
  w₁ > 0 ∧ h₁ > 0 ∧ w₂ > 0 ∧ h₂ > 0 →
  w₁ * h₁ = 6 →
  w₂ * h₁ = 15 →
  w₂ * h₂ = 25 →
  w₁ * h₂ = 10 :=
by
  sorry

end rectangle_division_l2194_219437


namespace ordering_abc_l2194_219410

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem ordering_abc : a > c ∧ c > b := by sorry

end ordering_abc_l2194_219410


namespace three_digit_number_problem_l2194_219463

theorem three_digit_number_problem (A B : ℝ) : 
  (100 ≤ A ∧ A < 1000) →  -- A is a three-digit number
  (B = A / 10 ∨ B = A / 100 ∨ B = A / 1000) →  -- B is obtained by placing a decimal point in front of one of A's digits
  (A - B = 478.8) →  -- Given condition
  A = 532 := by
sorry

end three_digit_number_problem_l2194_219463


namespace problem_1_problem_2_l2194_219481

-- Problem 1
theorem problem_1 : (4 - Real.pi) ^ 0 + (1/3)⁻¹ - 2 * Real.cos (45 * π / 180) = 4 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : 
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = x + 1 := by
  sorry

end problem_1_problem_2_l2194_219481


namespace trailer_homes_count_l2194_219418

/-- Represents the number of new trailer homes added -/
def new_homes : ℕ := 17

/-- The initial number of trailer homes -/
def initial_homes : ℕ := 25

/-- The initial average age of trailer homes (in years) -/
def initial_avg_age : ℕ := 15

/-- The time elapsed since the initial state (in years) -/
def time_elapsed : ℕ := 3

/-- The current average age of all trailer homes (in years) -/
def current_avg_age : ℕ := 12

theorem trailer_homes_count :
  (initial_homes * (initial_avg_age + time_elapsed) + new_homes * time_elapsed) / 
  (initial_homes + new_homes) = current_avg_age := by sorry

end trailer_homes_count_l2194_219418


namespace smallest_a_correct_l2194_219482

/-- The smallest natural number a such that there are exactly 50 perfect squares in the interval (a, 3a) -/
def smallest_a : ℕ := 4486

/-- The number of perfect squares in the interval (a, 3a) -/
def count_squares (a : ℕ) : ℕ :=
  (Nat.sqrt (3 * a) - Nat.sqrt a).pred

theorem smallest_a_correct :
  (∀ b < smallest_a, count_squares b ≠ 50) ∧
  count_squares smallest_a = 50 :=
sorry

#eval smallest_a
#eval count_squares smallest_a

end smallest_a_correct_l2194_219482


namespace quadrilateral_inequality_l2194_219441

-- Define the points
variable (A B C D O M N : ℝ × ℝ)

-- Define the triangle area function
def triangle_area (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem quadrilateral_inequality 
  (h_convex : sorry) -- ABCD is a convex quadrilateral
  (h_intersect : sorry) -- AC and BD intersect at O
  (h_line : sorry) -- Line through O intersects AB at M and CD at N
  (h_ineq1 : triangle_area O M B > triangle_area O N D)
  (h_ineq2 : triangle_area O C N > triangle_area O A M) :
  triangle_area O A M + triangle_area O B C + triangle_area O N D >
  triangle_area O D A + triangle_area O M B + triangle_area O C N :=
sorry

end quadrilateral_inequality_l2194_219441


namespace min_sum_squares_l2194_219416

def S : Finset Int := {-11, -8, -6, -1, 1, 5, 7, 12}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (∀ a b c d e f g h : Int, 
    a ∈ S → b ∈ S → c ∈ S → d ∈ S → e ∈ S → f ∈ S → g ∈ S → h ∈ S →
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → a ≠ g → a ≠ h →
    b ≠ c → b ≠ d → b ≠ e → b ≠ f → b ≠ g → b ≠ h →
    c ≠ d → c ≠ e → c ≠ f → c ≠ g → c ≠ h →
    d ≠ e → d ≠ f → d ≠ g → d ≠ h →
    e ≠ f → e ≠ g → e ≠ h →
    f ≠ g → f ≠ h →
    g ≠ h →
    (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 1) ∧
  (p + q + r + s)^2 + (t + u + v + w)^2 = 1 := by sorry

end min_sum_squares_l2194_219416


namespace min_value_of_sum_l2194_219424

theorem min_value_of_sum (x y : ℝ) (h : x^2 - 2*x*y + y^2 - Real.sqrt 2*x - Real.sqrt 2*y + 6 = 0) :
  ∃ (u : ℝ), u = x + y ∧ u ≥ 3 * Real.sqrt 2 ∧ ∀ (v : ℝ), v = x + y → v ≥ u := by
  sorry

end min_value_of_sum_l2194_219424


namespace evaluate_polynomial_l2194_219413

theorem evaluate_polynomial (a b : ℤ) (h : b = a + 2) :
  b^3 - a*b^2 - a^2*b + a^3 = 8*(a + 1) := by
  sorry

end evaluate_polynomial_l2194_219413


namespace cookies_remaining_l2194_219412

theorem cookies_remaining (total_taken : ℕ) (h1 : total_taken = 11) 
  (h2 : total_taken * 2 = total_taken + total_taken) : 
  total_taken = total_taken * 2 - total_taken := by
  sorry

end cookies_remaining_l2194_219412


namespace prob_odd_score_is_35_72_l2194_219402

/-- Represents the dartboard with given dimensions and point values -/
structure Dartboard :=
  (outer_radius : ℝ)
  (inner_radius : ℝ)
  (inner_points : Fin 3 → ℕ)
  (outer_points : Fin 3 → ℕ)

/-- Calculates the probability of scoring an odd sum with two darts -/
def prob_odd_score (db : Dartboard) : ℚ :=
  sorry

/-- The specific dartboard described in the problem -/
def problem_dartboard : Dartboard :=
  { outer_radius := 8
  , inner_radius := 4
  , inner_points := ![3, 4, 4]
  , outer_points := ![4, 3, 3] }

theorem prob_odd_score_is_35_72 :
  prob_odd_score problem_dartboard = 35 / 72 :=
sorry

end prob_odd_score_is_35_72_l2194_219402


namespace club_choices_l2194_219471

/-- Represents a club with boys and girls -/
structure Club where
  boys : ℕ
  girls : ℕ

/-- The number of ways to choose a president and vice-president of the same gender -/
def sameGenderChoices (c : Club) : ℕ :=
  c.boys * (c.boys - 1) + c.girls * (c.girls - 1)

/-- Theorem stating that for a club with 10 boys and 10 girls, 
    there are 180 ways to choose a president and vice-president of the same gender -/
theorem club_choices (c : Club) (h1 : c.boys = 10) (h2 : c.girls = 10) :
  sameGenderChoices c = 180 := by
  sorry

#check club_choices

end club_choices_l2194_219471


namespace equation_equivalence_l2194_219489

-- Define the original equation
def original_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + y^2) + Real.sqrt ((x + 4)^2 + y^2) = 10

-- Define the simplified equation
def simplified_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

-- Theorem statement
theorem equation_equivalence :
  ∀ x y : ℝ, original_equation x y ↔ simplified_equation x y :=
by sorry

end equation_equivalence_l2194_219489


namespace diane_gingerbreads_l2194_219467

/-- Proves that given Diane's baking conditions, each of the four trays contains 25 gingerbreads -/
theorem diane_gingerbreads :
  ∀ (x : ℕ),
  (4 * x + 3 * 20 = 160) →
  x = 25 :=
by
  sorry

end diane_gingerbreads_l2194_219467


namespace sin_cos_difference_equals_half_l2194_219436

theorem sin_cos_difference_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) - 
  Real.sin (13 * π / 180) * Real.cos (43 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_difference_equals_half_l2194_219436


namespace pairings_of_six_items_l2194_219443

/-- The number of possible pairings between two sets of 6 distinct items -/
def num_pairings (n : ℕ) : ℕ := n * n

/-- Theorem: The number of possible pairings between two sets of 6 distinct items is 36 -/
theorem pairings_of_six_items :
  num_pairings 6 = 36 := by
  sorry

end pairings_of_six_items_l2194_219443


namespace junk_mail_per_block_l2194_219426

/-- Given that a mailman distributes junk mail to blocks with the following conditions:
  1. The mailman gives 8 mails to each house in a block.
  2. There are 4 houses in a block.
Prove that the number of pieces of junk mail given to each block is 32. -/
theorem junk_mail_per_block (mails_per_house : ℕ) (houses_per_block : ℕ) 
  (h1 : mails_per_house = 8) (h2 : houses_per_block = 4) : 
  mails_per_house * houses_per_block = 32 := by
  sorry

#check junk_mail_per_block

end junk_mail_per_block_l2194_219426


namespace dave_earnings_l2194_219430

/-- Calculates the total money earned from selling video games -/
def total_money_earned (action_games adventure_games roleplaying_games : ℕ) 
  (action_price adventure_price roleplaying_price : ℕ) : ℕ :=
  action_games * action_price + 
  adventure_games * adventure_price + 
  roleplaying_games * roleplaying_price

/-- Proves that Dave earns $49 by selling all working games -/
theorem dave_earnings : 
  total_money_earned 3 2 3 6 5 7 = 49 := by
  sorry

end dave_earnings_l2194_219430


namespace inequality_equivalence_l2194_219462

theorem inequality_equivalence (x : ℝ) : 
  |((8 - x) / 4)|^2 < 4 ↔ 0 < x ∧ x < 16 :=
by sorry

end inequality_equivalence_l2194_219462


namespace power_six_sum_l2194_219439

theorem power_six_sum (x : ℝ) (h : x + 1/x = 4) : x^6 + 1/x^6 = 2702 := by
  sorry

end power_six_sum_l2194_219439


namespace divisibility_by_six_l2194_219431

theorem divisibility_by_six (y : ℕ) : y < 10 → (62000 + y * 100 + 16) % 6 = 0 ↔ y = 3 := by
  sorry

end divisibility_by_six_l2194_219431


namespace min_balls_to_draw_for_given_counts_l2194_219421

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat

/-- The minimum number of balls to draw to guarantee the desired outcome -/
def minBallsToDraw (counts : BallCounts) : Nat :=
  sorry

/-- The theorem stating the minimum number of balls to draw for the given problem -/
theorem min_balls_to_draw_for_given_counts :
  let counts := BallCounts.mk 30 25 20 15 10
  minBallsToDraw counts = 81 := by
  sorry

end min_balls_to_draw_for_given_counts_l2194_219421


namespace monotonic_increase_interval_l2194_219438

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the composition function g(x) = f(|x+2|)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (|x + 2|)

-- State the theorem
theorem monotonic_increase_interval
  (f : ℝ → ℝ) (h : DecreasingFunction f) :
  StrictMonoOn (g f) (Set.Iio (-2)) :=
sorry

end monotonic_increase_interval_l2194_219438


namespace complex_square_l2194_219419

-- Define the complex number i
axiom i : ℂ
axiom i_squared : i * i = -1

-- State the theorem
theorem complex_square : (1 + i) * (1 + i) = 2 * i := by sorry

end complex_square_l2194_219419


namespace bank_profit_maximization_l2194_219440

/-- The bank's profit maximization problem -/
theorem bank_profit_maximization
  (k : ℝ) -- Proportionality constant
  (h_k_pos : k > 0) -- k is positive
  (loan_rate : ℝ := 0.048) -- Loan interest rate
  (deposit_rate : ℝ) -- Deposit interest rate
  (h_deposit_rate : deposit_rate > 0 ∧ deposit_rate < loan_rate) -- Deposit rate is between 0 and loan rate
  (deposit_amount : ℝ := k * deposit_rate^2) -- Deposit amount formula
  (profit : ℝ → ℝ := λ x => loan_rate * k * x^2 - k * x^3) -- Profit function
  : (∀ x, x > 0 ∧ x < loan_rate → profit x ≤ profit 0.032) :=
by sorry

end bank_profit_maximization_l2194_219440


namespace fraction_equality_l2194_219461

theorem fraction_equality : (1 : ℚ) / 2 = 4 / 8 := by sorry

end fraction_equality_l2194_219461


namespace right_triangle_roots_l2194_219411

-- Define the equation
def equation (m x : ℝ) : Prop := x^2 - (2*m + 1)*x + m^2 + m = 0

-- Define the roots
def roots (m : ℝ) : Set ℝ := {x | equation m x}

theorem right_triangle_roots (m : ℝ) :
  let a := (2*m + 1 + 1) / 2
  let b := (2*m + 1 - 1) / 2
  (∀ x ∈ roots m, x = a ∨ x = b) →
  a^2 + b^2 = 5^2 →
  m = 3 := by sorry

end right_triangle_roots_l2194_219411


namespace square_with_quarter_circles_area_l2194_219488

theorem square_with_quarter_circles_area (π : Real) : 
  let square_side : Real := 4
  let quarter_circle_radius : Real := square_side / 2
  let square_area : Real := square_side ^ 2
  let quarter_circle_area : Real := π * quarter_circle_radius ^ 2 / 4
  let total_quarter_circles_area : Real := 4 * quarter_circle_area
  square_area - total_quarter_circles_area = 16 - 4 * π := by sorry

end square_with_quarter_circles_area_l2194_219488


namespace heart_ratio_eq_half_l2194_219434

def heart (n m : ℕ) : ℕ := n^3 * m^2

theorem heart_ratio_eq_half :
  (heart 2 4) / (heart 4 2) = 1/2 := by sorry

end heart_ratio_eq_half_l2194_219434


namespace class_test_problem_l2194_219460

theorem class_test_problem (first_correct : Real) (second_correct : Real) (both_correct : Real)
  (h1 : first_correct = 0.75)
  (h2 : second_correct = 0.65)
  (h3 : both_correct = 0.60) :
  1 - (first_correct + second_correct - both_correct) = 0.20 := by
  sorry

end class_test_problem_l2194_219460


namespace parallelogram_area_l2194_219417

/-- The area of a parallelogram with given dimensions -/
theorem parallelogram_area (base slant_height : ℝ) (angle : ℝ) : 
  base = 20 → 
  slant_height = 10 → 
  angle = 30 * π / 180 → 
  base * (slant_height * Real.sin angle) = 100 := by
  sorry

end parallelogram_area_l2194_219417


namespace cans_per_carton_l2194_219475

theorem cans_per_carton (total_cartons : ℕ) (loaded_cartons : ℕ) (remaining_cans : ℕ) :
  total_cartons = 50 →
  loaded_cartons = 40 →
  remaining_cans = 200 →
  (total_cartons - loaded_cartons) * (remaining_cans / (total_cartons - loaded_cartons)) = remaining_cans :=
by sorry

end cans_per_carton_l2194_219475


namespace min_value_reciprocal_sum_l2194_219428

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  ∃ (min : ℝ), min = 3 + 2 * Real.sqrt 2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 2 * x + y = 1 → 1 / x + 1 / y ≥ min :=
sorry

end min_value_reciprocal_sum_l2194_219428


namespace middle_digit_zero_l2194_219490

theorem middle_digit_zero (a b c : Nat) (M : Nat) :
  (0 ≤ a ∧ a < 6) →
  (0 ≤ b ∧ b < 6) →
  (0 ≤ c ∧ c < 6) →
  M = 36 * a + 6 * b + c →
  M = 64 * a + 8 * b + c →
  b = 0 := by
  sorry

end middle_digit_zero_l2194_219490


namespace sin_52pi_over_3_l2194_219492

theorem sin_52pi_over_3 : Real.sin (52 * π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end sin_52pi_over_3_l2194_219492


namespace library_book_purchase_l2194_219422

theorem library_book_purchase (initial_books : ℕ) (current_books : ℕ) (last_year_purchase : ℕ) : 
  initial_books = 100 →
  current_books = 300 →
  current_books = initial_books + last_year_purchase + 3 * last_year_purchase →
  last_year_purchase = 50 := by
  sorry

end library_book_purchase_l2194_219422


namespace plan_y_cheaper_at_min_usage_l2194_219408

/-- Cost of Plan X in cents for z MB of data usage -/
def cost_plan_x (z : ℕ) : ℕ := 15 * z

/-- Cost of Plan Y in cents for z MB of data usage, without discount -/
def cost_plan_y_no_discount (z : ℕ) : ℕ := 3000 + 7 * z

/-- Cost of Plan Y in cents for z MB of data usage, with discount -/
def cost_plan_y_with_discount (z : ℕ) : ℕ := 
  if z > 500 then cost_plan_y_no_discount z - 1000 else cost_plan_y_no_discount z

/-- The minimum usage in MB where Plan Y becomes cheaper than Plan X -/
def min_usage : ℕ := 501

theorem plan_y_cheaper_at_min_usage : 
  cost_plan_y_with_discount min_usage < cost_plan_x min_usage ∧
  ∀ z : ℕ, z < min_usage → cost_plan_x z ≤ cost_plan_y_with_discount z :=
by sorry


end plan_y_cheaper_at_min_usage_l2194_219408


namespace angle_sum_in_circle_l2194_219449

theorem angle_sum_in_circle (x : ℝ) : 
  (6 * x + 3 * x + 2 * x + x = 360) → x = 30 := by
  sorry

end angle_sum_in_circle_l2194_219449


namespace inequality_solution_set_l2194_219468

theorem inequality_solution_set (x : ℝ) : 
  (5 / (x + 2) ≥ 1 ∧ x + 2 ≠ 0) ↔ -2 < x ∧ x ≤ 3 :=
by sorry

end inequality_solution_set_l2194_219468


namespace reciprocal_sum_pairs_l2194_219483

theorem reciprocal_sum_pairs : 
  ∃! (count : ℕ), ∃ (pairs : Finset (ℕ × ℕ)),
    pairs.card = count ∧
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > 0 ∧ n > 0 ∧ (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 3) ∧
    count = 3 :=
by sorry

end reciprocal_sum_pairs_l2194_219483


namespace problem_statement_l2194_219429

open Real

theorem problem_statement :
  (∀ x > 0, exp x - 2 > x - 1 ∧ x - 1 ≥ log x) ∧
  (∀ m : ℤ, m < 1 → ¬∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ exp x - log x - m - 2 = 0 ∧ exp y - log y - m - 2 = 0) ∧
  (∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ exp x - log x - 1 - 2 = 0 ∧ exp y - log y - 1 - 2 = 0) := by
  sorry

end problem_statement_l2194_219429


namespace marathon_completion_time_l2194_219450

/-- The time to complete a marathon given the distance and average pace -/
theorem marathon_completion_time 
  (distance : ℕ) 
  (avg_pace : ℕ) 
  (h1 : distance = 24)  -- marathon distance in miles
  (h2 : avg_pace = 9)   -- average pace in minutes per mile
  : distance * avg_pace = 216 := by
  sorry

end marathon_completion_time_l2194_219450


namespace increase_when_multiplied_l2194_219472

theorem increase_when_multiplied (n : ℕ) (m : ℕ) (increase : ℕ) : n = 14 → m = 15 → increase = m * n - n → increase = 196 := by
  sorry

end increase_when_multiplied_l2194_219472


namespace special_ellipse_eccentricity_special_ellipse_equation_l2194_219465

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_foci : F₁.1 < F₂.1 -- F₁ is left focus, F₂ is right focus
  h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ ({A, B} : Set (ℝ × ℝ))
  h_line : A.2 - F₁.2 = A.1 - F₁.1 ∧ B.2 - F₁.2 = B.1 - F₁.1 -- Line through F₁ with slope 1
  h_arithmetic : ∃ (d : ℝ), dist A F₂ + d = dist A B ∧ dist A B + d = dist B F₂
  h_circle : ∃ (r : ℝ), dist A (-2, 0) = r ∧ dist B (-2, 0) = r

/-- The eccentricity of the special ellipse is √2/2 -/
theorem special_ellipse_eccentricity (E : SpecialEllipse) : 
  (E.a^2 - E.b^2) / E.a^2 = 1/2 := by sorry

/-- The equation of the special ellipse is x²/72 + y²/36 = 1 -/
theorem special_ellipse_equation (E : SpecialEllipse) : 
  E.a^2 = 72 ∧ E.b^2 = 36 := by sorry

end special_ellipse_eccentricity_special_ellipse_equation_l2194_219465


namespace sine_equality_proof_l2194_219458

theorem sine_equality_proof (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * π / 180) = Real.sin (720 * π / 180) → n = 0 :=
by sorry

end sine_equality_proof_l2194_219458


namespace max_quarters_l2194_219446

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The total amount Sasha has in dollars -/
def total_amount : ℚ := 4.50

/-- Proves that the maximum number of quarters (and dimes) Sasha can have is 12 -/
theorem max_quarters : 
  ∀ q : ℕ, 
    (q : ℚ) * (quarter_value + dime_value) ≤ total_amount → 
    q ≤ 12 := by
  sorry

#check max_quarters

end max_quarters_l2194_219446


namespace candy_problem_l2194_219484

theorem candy_problem (x : ℚ) : 
  (2/9 * x - 2/3 - 4 = 8) → x = 57 := by
  sorry

end candy_problem_l2194_219484


namespace channel_transmission_theorem_l2194_219480

/-- Channel transmission probabilities -/
structure ChannelProb where
  α : ℝ
  β : ℝ
  h_α_pos : 0 < α
  h_α_lt_one : α < 1
  h_β_pos : 0 < β
  h_β_lt_one : β < 1

/-- Single transmission probability for sequence 1, 0, 1 -/
def single_trans_prob (cp : ChannelProb) : ℝ := (1 - cp.α) * (1 - cp.β)^2

/-- Triple transmission probability for decoding 0 as 0 -/
def triple_trans_prob_0 (cp : ChannelProb) : ℝ :=
  (1 - cp.α)^3 + 3 * cp.α * (1 - cp.α)^2

/-- Single transmission probability for decoding 0 as 0 -/
def single_trans_prob_0 (cp : ChannelProb) : ℝ := 1 - cp.α

theorem channel_transmission_theorem (cp : ChannelProb) :
  single_trans_prob cp = (1 - cp.α) * (1 - cp.β)^2 ∧
  (cp.α < 1/2 → triple_trans_prob_0 cp > single_trans_prob_0 cp) := by sorry

end channel_transmission_theorem_l2194_219480


namespace eriks_mother_money_l2194_219448

/-- The amount of money Erik's mother gave him. -/
def money_from_mother : ℕ := sorry

/-- The number of loaves of bread Erik bought. -/
def bread_loaves : ℕ := 3

/-- The number of cartons of orange juice Erik bought. -/
def juice_cartons : ℕ := 3

/-- The cost of one loaf of bread in dollars. -/
def bread_cost : ℕ := 3

/-- The cost of one carton of orange juice in dollars. -/
def juice_cost : ℕ := 6

/-- The amount of money Erik has left in dollars. -/
def money_left : ℕ := 59

/-- Theorem stating that the amount of money Erik's mother gave him is $86. -/
theorem eriks_mother_money : money_from_mother = 86 := by sorry

end eriks_mother_money_l2194_219448


namespace hostel_provisions_l2194_219469

/-- The number of men initially in the hostel -/
def initial_men : ℕ := 250

/-- The number of days the provisions last initially -/
def initial_days : ℕ := 40

/-- The number of men who leave the hostel -/
def men_who_leave : ℕ := 50

/-- The number of days the provisions last after some men leave -/
def days_after_leaving : ℕ := 50

theorem hostel_provisions :
  initial_men * initial_days = (initial_men - men_who_leave) * days_after_leaving :=
by sorry

#check hostel_provisions

end hostel_provisions_l2194_219469


namespace probability_triangle_or_circle_l2194_219451

theorem probability_triangle_or_circle :
  let total_figures : ℕ := 10
  let triangle_count : ℕ := 4
  let circle_count : ℕ := 3
  let target_count : ℕ := triangle_count + circle_count
  (target_count : ℚ) / total_figures = 7 / 10 :=
by sorry

end probability_triangle_or_circle_l2194_219451


namespace triangle_max_area_l2194_219420

/-- Given a triangle ABC with sides a, b, c and area S, 
    if S = a² - (b-c)² and b + c = 8, 
    then the maximum possible value of S is 64/17 -/
theorem triangle_max_area (a b c S : ℝ) : 
  S = a^2 - (b-c)^2 → b + c = 8 → (∀ S' : ℝ, S' = a'^2 - (b'-c')^2 ∧ b' + c' = 8 → S' ≤ S) → S = 64/17 :=
by sorry


end triangle_max_area_l2194_219420


namespace min_trapezium_perimeter_l2194_219493

/-- A right-angled isosceles triangle with hypotenuse √2 cm -/
structure RightIsoscelesTriangle where
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = Real.sqrt 2

/-- A trapezium formed by assembling right-angled isosceles triangles -/
structure Trapezium where
  triangles : List RightIsoscelesTriangle
  is_trapezium : Bool  -- This should be a predicate ensuring the shape is a trapezium

/-- The perimeter of a trapezium -/
def trapezium_perimeter (t : Trapezium) : ℝ := sorry

/-- Theorem stating the minimum perimeter of a trapezium formed by right-angled isosceles triangles -/
theorem min_trapezium_perimeter :
  ∀ t : Trapezium, trapezium_perimeter t ≥ 4 + 2 * Real.sqrt 2 := by sorry

end min_trapezium_perimeter_l2194_219493


namespace sin_30_degrees_l2194_219474

theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end sin_30_degrees_l2194_219474


namespace soccer_penalty_kicks_l2194_219457

theorem soccer_penalty_kicks (total_players : ℕ) (goalkeepers : ℕ) : 
  total_players = 16 → goalkeepers = 2 → (total_players - goalkeepers) * goalkeepers = 30 := by
  sorry

end soccer_penalty_kicks_l2194_219457


namespace range_of_m_l2194_219491

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (Set.compl (A m) ∩ B = ∅) → m ∈ Set.Ici 2 := by
  sorry

end range_of_m_l2194_219491


namespace salazar_oranges_l2194_219405

theorem salazar_oranges (initial_oranges : ℕ) (sold_fraction : ℚ) 
  (rotten_oranges : ℕ) (remaining_oranges : ℕ) :
  initial_oranges = 7 * 12 →
  sold_fraction = 3 / 7 →
  rotten_oranges = 4 →
  remaining_oranges = 32 →
  ∃ (f : ℚ), 
    0 ≤ f ∧ f ≤ 1 ∧
    (1 - f) * initial_oranges - sold_fraction * ((1 - f) * initial_oranges) - rotten_oranges = remaining_oranges ∧
    f = 1 / 4 := by
  sorry

end salazar_oranges_l2194_219405


namespace certain_number_bound_l2194_219496

theorem certain_number_bound (x y z : ℤ) (N : ℝ) 
  (h1 : x < y ∧ y < z)
  (h2 : (y - x : ℝ) > N)
  (h3 : Even x)
  (h4 : Odd y ∧ Odd z)
  (h5 : ∀ (a b : ℤ), (Even a ∧ Odd b ∧ a < b) → (b - a ≥ 7) → (z - x ≤ b - a)) :
  N < 3 := by
sorry

end certain_number_bound_l2194_219496


namespace minervas_stamps_l2194_219444

/-- Given that Lizette has 813 stamps and 125 more stamps than Minerva,
    prove that Minerva has 688 stamps. -/
theorem minervas_stamps (lizette_stamps : ℕ) (difference : ℕ) 
  (h1 : lizette_stamps = 813)
  (h2 : difference = 125)
  (h3 : lizette_stamps = difference + minerva_stamps) :
  minerva_stamps = 688 := by
  sorry

end minervas_stamps_l2194_219444


namespace water_consumption_proof_l2194_219466

/-- Calculates the total water consumption for horses over a given number of days -/
def total_water_consumption (initial_horses : ℕ) (added_horses : ℕ) (drinking_water : ℕ) (bathing_water : ℕ) (days : ℕ) : ℕ :=
  (initial_horses + added_horses) * (drinking_water + bathing_water) * days

/-- Proves that under given conditions, the total water consumption for 28 days is 1568 liters -/
theorem water_consumption_proof :
  total_water_consumption 3 5 5 2 28 = 1568 := by
  sorry

#eval total_water_consumption 3 5 5 2 28

end water_consumption_proof_l2194_219466


namespace triangle_area_l2194_219445

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, prove that if 4S = √3(a² + b² - c²) and 
    f(x) = 4sin(x)cos(x + π/6) + 1 attains its maximum value b when x = A,
    then the area S of the triangle is √3/2. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  4 * S = Real.sqrt 3 * (a^2 + b^2 - c^2) →
  (∀ x, 4 * Real.sin x * Real.cos (x + π/6) + 1 ≤ b) →
  (4 * Real.sin A * Real.cos (A + π/6) + 1 = b) →
  S = Real.sqrt 3 / 2 := by
  sorry

end triangle_area_l2194_219445


namespace polynomial_factorization_l2194_219433

theorem polynomial_factorization (a b : ℝ) : 
  a^2 - b^2 + 2*a + 1 = (a - b + 1) * (a + b + 1) := by
  sorry

end polynomial_factorization_l2194_219433


namespace intersection_of_A_and_B_l2194_219470

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l2194_219470


namespace work_time_problem_l2194_219415

/-- The work time problem for Mr. Willson -/
theorem work_time_problem (total_time tuesday wednesday thursday friday : ℚ) :
  total_time = 4 ∧
  tuesday = 1/2 ∧
  wednesday = 2/3 ∧
  thursday = 5/6 ∧
  friday = 75/60 →
  total_time - (tuesday + wednesday + thursday + friday) = 3/4 := by
sorry

end work_time_problem_l2194_219415


namespace triangle_ratio_l2194_219477

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →  -- 60° in radians
  a = Real.sqrt 13 →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end triangle_ratio_l2194_219477


namespace negation_of_p_l2194_219476

/-- Proposition p: a and b are both even numbers -/
def p (a b : ℤ) : Prop := Even a ∧ Even b

/-- The negation of proposition p -/
theorem negation_of_p (a b : ℤ) : ¬(p a b) ↔ ¬(Even a ∧ Even b) := by sorry

end negation_of_p_l2194_219476


namespace parabola_normal_min_area_l2194_219425

noncomputable def min_y_coordinate : ℝ := (-3 + Real.sqrt 33) / 24

theorem parabola_normal_min_area (x₀ : ℝ) :
  let y₀ := x₀^2
  let normal_slope := -1 / (2 * x₀)
  let x₁ := -1 / (2 * x₀) - x₀
  let y₁ := x₁^2
  let triangle_area := (1/2) * (x₀ - x₁) * (y₀ + 1/2)
  (∀ x : ℝ, triangle_area ≤ ((1/2) * (x - (-1 / (2 * x) - x)) * (x^2 + 1/2))) →
  y₀ = min_y_coordinate := by
sorry

end parabola_normal_min_area_l2194_219425


namespace product_and_reciprocal_relation_l2194_219487

theorem product_and_reciprocal_relation (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 16 → 1 / x = 3 * (1 / y) → |x - y| = (8 * Real.sqrt 3) / 3 := by
  sorry

end product_and_reciprocal_relation_l2194_219487


namespace inequality_solution_inequality_proof_l2194_219499

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part (I)
theorem inequality_solution (x : ℝ) :
  f (x - 1) + f (x + 3) ≥ 6 ↔ x ≤ -3 ∨ x ≥ 3 :=
sorry

-- Theorem for part (II)
theorem inequality_proof (a b : ℝ) (h1 : |a| < 1) (h2 : |b| < 1) (h3 : a ≠ 0) :
  f (a * b) > |a| * f (b / a) :=
sorry

end inequality_solution_inequality_proof_l2194_219499


namespace pretzels_eaten_difference_l2194_219423

/-- The number of pretzels Marcus ate compared to John -/
def pretzels_difference (total : ℕ) (john : ℕ) (alan : ℕ) (marcus : ℕ) : ℕ :=
  marcus - john

/-- Theorem stating the difference in pretzels eaten between Marcus and John -/
theorem pretzels_eaten_difference 
  (total : ℕ) 
  (john : ℕ) 
  (alan : ℕ) 
  (marcus : ℕ) 
  (h1 : total = 95)
  (h2 : john = 28)
  (h3 : alan = john - 9)
  (h4 : marcus > john)
  (h5 : marcus = 40) :
  pretzels_difference total john alan marcus = 12 := by
  sorry

end pretzels_eaten_difference_l2194_219423


namespace geometric_sequence_sum_range_l2194_219478

/-- Given real numbers a, b, c forming a geometric sequence with sum 1,
    prove that a + c is non-negative and unbounded above. -/
theorem geometric_sequence_sum_range (a b c : ℝ) : 
  (∃ r : ℝ, a = r ∧ b = r^2 ∧ c = r^3) →  -- geometric sequence condition
  a + b + c = 1 →                        -- sum condition
  (a + c ≥ 0 ∧ ∀ M : ℝ, ∃ x y z : ℝ, 
    (∃ r : ℝ, x = r ∧ y = r^2 ∧ z = r^3) ∧ 
    x + y + z = 1 ∧ 
    x + z > M) := by
  sorry

end geometric_sequence_sum_range_l2194_219478
