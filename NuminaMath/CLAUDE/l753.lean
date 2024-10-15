import Mathlib

namespace NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l753_75386

theorem conditional_probability_rain_given_east_wind
  (p_east_wind : ℚ)
  (p_rain : ℚ)
  (p_both : ℚ)
  (h1 : p_east_wind = 3 / 10)
  (h2 : p_rain = 11 / 30)
  (h3 : p_both = 8 / 30) :
  p_both / p_east_wind = 8 / 9 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l753_75386


namespace NUMINAMATH_CALUDE_distance_after_7km_l753_75348

/-- Regular hexagon with side length 3 km -/
structure RegularHexagon where
  side_length : ℝ
  is_regular : side_length = 3

/-- Point on the perimeter of the hexagon -/
structure PerimeterPoint (h : RegularHexagon) where
  distance_from_start : ℝ
  on_perimeter : distance_from_start ≥ 0 ∧ distance_from_start ≤ 6 * h.side_length

/-- The distance from the starting point to a point on the perimeter -/
def distance_to_start (h : RegularHexagon) (p : PerimeterPoint h) : ℝ :=
  sorry

theorem distance_after_7km (h : RegularHexagon) (p : PerimeterPoint h) 
  (h_distance : p.distance_from_start = 7) :
  distance_to_start h p = 2 :=
sorry

end NUMINAMATH_CALUDE_distance_after_7km_l753_75348


namespace NUMINAMATH_CALUDE_polynomial_divisibility_implies_perfect_powers_l753_75304

/-- Given a polynomial ax³ + 3bx² + 3cx + d that is divisible by ax² + 2bx + c,
    prove that it's a perfect cube and the divisor is a perfect square. -/
theorem polynomial_divisibility_implies_perfect_powers
  (a b c d : ℝ) (h : a ≠ 0) :
  (∃ (q : ℝ → ℝ), ∀ x, a * x^3 + 3*b * x^2 + 3*c * x + d = (a * x^2 + 2*b * x + c) * q x) →
  (∃ y, a * x^3 + 3*b * x^2 + 3*c * x + d = (a * x + y)^3) ∧
  (∃ z, a * x^2 + 2*b * x + c = (a * x + z)^2) ∧
  c = 2 * b^2 / a ∧
  d = 2 * b^3 / a^2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_implies_perfect_powers_l753_75304


namespace NUMINAMATH_CALUDE_tom_neither_soccer_nor_test_l753_75391

theorem tom_neither_soccer_nor_test (soccer_prob : ℚ) (test_prob : ℚ) 
  (h_soccer : soccer_prob = 5 / 8)
  (h_test : test_prob = 1 / 4)
  (h_independent : True) -- Assumption of independence
  : (1 - soccer_prob) * (1 - test_prob) = 9 / 32 := by
  sorry

end NUMINAMATH_CALUDE_tom_neither_soccer_nor_test_l753_75391


namespace NUMINAMATH_CALUDE_third_shift_participation_rate_l753_75319

-- Define the total number of employees in each shift
def first_shift : ℕ := 60
def second_shift : ℕ := 50
def third_shift : ℕ := 40

-- Define the participation rates for the first two shifts
def first_shift_rate : ℚ := 1/5
def second_shift_rate : ℚ := 2/5

-- Define the total participation rate
def total_participation_rate : ℚ := 6/25

-- Theorem statement
theorem third_shift_participation_rate :
  let total_employees := first_shift + second_shift + third_shift
  let total_participants := total_employees * total_participation_rate
  let first_shift_participants := first_shift * first_shift_rate
  let second_shift_participants := second_shift * second_shift_rate
  let third_shift_participants := total_participants - first_shift_participants - second_shift_participants
  third_shift_participants / third_shift = 1/10 := by
sorry

end NUMINAMATH_CALUDE_third_shift_participation_rate_l753_75319


namespace NUMINAMATH_CALUDE_die_roll_frequency_l753_75396

/-- The frequency of an event in an experiment -/
def frequency (occurrences : ℕ) (totalTrials : ℕ) : ℚ :=
  occurrences / totalTrials

/-- The number of times the die was rolled -/
def totalRolls : ℕ := 100

/-- The number of times "even numbers facing up" occurred -/
def evenOccurrences : ℕ := 47

/-- The expected frequency of "even numbers facing up" -/
def expectedFrequency : ℚ := 47 / 100

theorem die_roll_frequency :
  frequency evenOccurrences totalRolls = expectedFrequency := by
  sorry

end NUMINAMATH_CALUDE_die_roll_frequency_l753_75396


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l753_75370

/-- Given a mixture of milk and water with an initial ratio of 2:1, 
    prove that if 60 litres of water is added to change the ratio to 1:2, 
    the initial volume of the mixture was 60 litres. -/
theorem initial_mixture_volume 
  (initial_milk : ℝ) 
  (initial_water : ℝ) 
  (h1 : initial_milk = 2 * initial_water) 
  (h2 : initial_milk = (initial_water + 60) / 2) : 
  initial_milk + initial_water = 60 := by
  sorry

#check initial_mixture_volume

end NUMINAMATH_CALUDE_initial_mixture_volume_l753_75370


namespace NUMINAMATH_CALUDE_train_departure_time_difference_l753_75331

/-- Proves that Train A leaves 40 minutes before Train B, given their speeds and overtake time --/
theorem train_departure_time_difference 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (overtake_time : ℝ) 
  (h1 : speed_A = 60) 
  (h2 : speed_B = 80) 
  (h3 : overtake_time = 120) :
  ∃ (time_diff : ℝ), 
    time_diff = 40 ∧ 
    speed_A * (time_diff / 60 + overtake_time / 60) = speed_B * (overtake_time / 60) := by
  sorry


end NUMINAMATH_CALUDE_train_departure_time_difference_l753_75331


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_reciprocal_l753_75321

theorem imaginary_part_of_complex_reciprocal (z : ℂ) (h : z = 1 + 2*I) : 
  Complex.im (z⁻¹) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_reciprocal_l753_75321


namespace NUMINAMATH_CALUDE_wait_time_difference_l753_75344

/-- Proves that the difference in wait times between swings and slide is 270 seconds -/
theorem wait_time_difference : 
  let kids_on_swings : ℕ := 3
  let kids_on_slide : ℕ := 2 * kids_on_swings
  let swing_wait_time : ℕ := 2 * 60  -- 2 minutes in seconds
  let slide_wait_time : ℕ := 15      -- 15 seconds
  let total_swing_wait : ℕ := kids_on_swings * swing_wait_time
  let total_slide_wait : ℕ := kids_on_slide * slide_wait_time
  total_swing_wait - total_slide_wait = 270 := by
sorry


end NUMINAMATH_CALUDE_wait_time_difference_l753_75344


namespace NUMINAMATH_CALUDE_power_function_decreasing_first_quadrant_l753_75381

/-- A power function with negative exponent is decreasing in the first quadrant -/
theorem power_function_decreasing_first_quadrant (n : ℝ) (h : n < 0) :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂^n < x₁^n :=
by sorry


end NUMINAMATH_CALUDE_power_function_decreasing_first_quadrant_l753_75381


namespace NUMINAMATH_CALUDE_expression_evaluation_l753_75373

theorem expression_evaluation : (16 : ℝ) * 0.5 - (4.5 - 0.125 * 8) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l753_75373


namespace NUMINAMATH_CALUDE_john_remaining_money_l753_75382

theorem john_remaining_money (initial_amount : ℕ) (spent_amount : ℕ) : 
  initial_amount = 1600 →
  initial_amount - spent_amount = spent_amount - 600 →
  initial_amount - spent_amount = 500 :=
by sorry

end NUMINAMATH_CALUDE_john_remaining_money_l753_75382


namespace NUMINAMATH_CALUDE_at_least_n_prime_divisors_l753_75364

theorem at_least_n_prime_divisors (n : ℕ) :
  ∃ (S : Finset Nat), (S.card ≥ n) ∧ (∀ p ∈ S, Nat.Prime p ∧ p ∣ (2^(2^n) + 2^(2^(n-1)) + 1)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_n_prime_divisors_l753_75364


namespace NUMINAMATH_CALUDE_area_of_triangle_pqr_l753_75398

-- Define the square pyramid
structure SquarePyramid where
  base_side : ℝ
  altitude : ℝ

-- Define points P, Q, R
structure PyramidPoints where
  p_ratio : ℝ
  q_ratio : ℝ
  r_ratio : ℝ

-- Define the theorem
theorem area_of_triangle_pqr 
  (pyramid : SquarePyramid) 
  (points : PyramidPoints) 
  (h1 : pyramid.base_side = 4) 
  (h2 : pyramid.altitude = 8) 
  (h3 : points.p_ratio = 1/4) 
  (h4 : points.q_ratio = 1/4) 
  (h5 : points.r_ratio = 3/4) : 
  ∃ (area : ℝ), area = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_pqr_l753_75398


namespace NUMINAMATH_CALUDE_more_difficult_than_easy_l753_75325

/-- Represents the number of problems solved by a specific number of people -/
structure ProblemCounts where
  total : ℕ
  solvedByOne : ℕ
  solvedByTwo : ℕ
  solvedByThree : ℕ

/-- The total number of problems solved by each person -/
def problemsPerPerson : ℕ := 60

theorem more_difficult_than_easy (p : ProblemCounts) :
  p.total = 100 →
  p.solvedByOne + p.solvedByTwo + p.solvedByThree = p.total →
  p.solvedByOne + 3 * p.solvedByThree + 2 * p.solvedByTwo = 3 * problemsPerPerson →
  p.solvedByOne = p.solvedByThree + 20 :=
by
  sorry

#check more_difficult_than_easy

end NUMINAMATH_CALUDE_more_difficult_than_easy_l753_75325


namespace NUMINAMATH_CALUDE_f_properties_l753_75390

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x / 2) * cos (x / 2) - 2 * Real.sqrt 3 * sin (x / 2) ^ 2 + Real.sqrt 3

theorem f_properties (α : ℝ) (h1 : α ∈ Set.Ioo (π / 6) (2 * π / 3)) (h2 : f α = 6 / 5) :
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (2 * k * π + π / 6) (2 * k * π + 7 * π / 6))) ∧
  f (α - π / 6) = (4 + 3 * Real.sqrt 3) / 5 := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l753_75390


namespace NUMINAMATH_CALUDE_cubic_parabola_collinearity_l753_75309

/-- Represents a point on a cubic parabola -/
structure CubicPoint where
  x : ℝ
  y : ℝ

/-- Represents a cubic parabola y = x^3 + a₁x^2 + a₂x + a₃ -/
structure CubicParabola where
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ

/-- Check if a point lies on the cubic parabola -/
def onCubicParabola (p : CubicPoint) (c : CubicParabola) : Prop :=
  p.y = p.x^3 + c.a₁ * p.x^2 + c.a₂ * p.x + c.a₃

/-- Check if three points are collinear -/
def areCollinear (p q r : CubicPoint) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- Main theorem: Given a cubic parabola and three points on it with x-coordinates summing to -a₁, the points are collinear -/
theorem cubic_parabola_collinearity (c : CubicParabola) (p q r : CubicPoint)
    (h_p : onCubicParabola p c)
    (h_q : onCubicParabola q c)
    (h_r : onCubicParabola r c)
    (h_sum : p.x + q.x + r.x = -c.a₁) :
    areCollinear p q r := by
  sorry

end NUMINAMATH_CALUDE_cubic_parabola_collinearity_l753_75309


namespace NUMINAMATH_CALUDE_racing_game_cost_l753_75366

/-- The cost of the racing game given the total spent and the cost of the basketball game -/
theorem racing_game_cost (total_spent basketball_cost : ℚ) 
  (h1 : total_spent = 9.43)
  (h2 : basketball_cost = 5.20) : 
  total_spent - basketball_cost = 4.23 := by
  sorry

end NUMINAMATH_CALUDE_racing_game_cost_l753_75366


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l753_75312

/-- A point in a 2D coordinate system. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis. -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The theorem stating that reflecting P(3, -4) across the y-axis results in P'(-3, -4). -/
theorem reflection_across_y_axis :
  let P : Point := { x := 3, y := -4 }
  let P' : Point := reflectAcrossYAxis P
  P'.x = -3 ∧ P'.y = -4 := by sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l753_75312


namespace NUMINAMATH_CALUDE_f_shifted_l753_75314

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_shifted (x : ℝ) : f (x + 1) = x^2 + 2*x → f (x - 1) = x^2 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_f_shifted_l753_75314


namespace NUMINAMATH_CALUDE_sequence_properties_l753_75368

/-- Given a sequence {a_n} with the sum formula S_n = 2n^2 - 26n -/
def S (n : ℕ) : ℤ := 2 * n^2 - 26 * n

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℤ := 4 * n - 28

theorem sequence_properties :
  (∀ n : ℕ, a n = S (n + 1) - S n) ∧
  (∀ n : ℕ, a (n + 1) - a n = 4) ∧
  (∃ n : ℕ, n = 6 ∨ n = 7) ∧ (∀ m : ℕ, S m ≥ S 6 ∧ S m ≥ S 7) := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l753_75368


namespace NUMINAMATH_CALUDE_rectangle_length_l753_75346

/-- Represents a rectangle with perimeter P, width W, length L, and area A. -/
structure Rectangle where
  P : ℝ  -- Perimeter
  W : ℝ  -- Width
  L : ℝ  -- Length
  A : ℝ  -- Area
  h1 : P = 2 * (L + W)  -- Perimeter formula
  h2 : A = L * W        -- Area formula
  h3 : P / W = 5        -- Given ratio
  h4 : A = 150          -- Given area

/-- Proves that a rectangle with the given properties has a length of 15. -/
theorem rectangle_length (rect : Rectangle) : rect.L = 15 := by
  sorry

#check rectangle_length

end NUMINAMATH_CALUDE_rectangle_length_l753_75346


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l753_75306

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from one base to another -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- Number of digits in a number in a given base -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_dual_base_palindrome :
  let n := 10001 -- In base 3
  ∀ m : ℕ,
    (numDigits n 3 = 5) →
    (isPalindrome n 3) →
    (∃ b : ℕ, b > 3 ∧ isPalindrome (baseConvert n 3 b) b ∧ numDigits (baseConvert n 3 b) b = 4) →
    (numDigits m 3 = 5) →
    (isPalindrome m 3) →
    (∃ b : ℕ, b > 3 ∧ isPalindrome (baseConvert m 3 b) b ∧ numDigits (baseConvert m 3 b) b = 4) →
    m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l753_75306


namespace NUMINAMATH_CALUDE_evaluate_expression_l753_75354

theorem evaluate_expression : ((4^4 - 4*(4-2)^4)^4) = 136048896 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l753_75354


namespace NUMINAMATH_CALUDE_sachin_age_l753_75302

theorem sachin_age (sachin_age rahul_age : ℕ) 
  (h1 : rahul_age = sachin_age + 7)
  (h2 : sachin_age * 12 = rahul_age * 5) : 
  sachin_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_sachin_age_l753_75302


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_five_l753_75300

def sequence_u (n : ℕ) : ℚ :=
  sorry

theorem sum_of_coefficients_is_five :
  ∃ (a b c : ℚ),
    (∀ n : ℕ, sequence_u n = a * n^2 + b * n + c) ∧
    (sequence_u 1 = 5) ∧
    (∀ n : ℕ, sequence_u (n + 1) - sequence_u n = 3 + 4 * (n - 1)) ∧
    (a + b + c = 5) :=
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_five_l753_75300


namespace NUMINAMATH_CALUDE_photo_gallery_total_l753_75310

theorem photo_gallery_total (initial_photos : ℕ) 
  (h1 : initial_photos = 1200) 
  (first_day : ℕ) 
  (h2 : first_day = initial_photos * 3 / 5) 
  (second_day : ℕ) 
  (h3 : second_day = first_day + 230) : 
  initial_photos + first_day + second_day = 2870 := by
  sorry

end NUMINAMATH_CALUDE_photo_gallery_total_l753_75310


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l753_75363

theorem geometric_sequence_properties (a b c : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ -1 = -r^4 ∧ a = -r^3 ∧ b = r^2 ∧ c = -r ∧ -9 = 1) →
  b = -3 ∧ a * c = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l753_75363


namespace NUMINAMATH_CALUDE_sum_max_l753_75356

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  sum_odd : a 1 + a 3 + a 5 = 156
  sum_even : a 2 + a 4 + a 6 = 147

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (List.range n).map seq.a |>.sum

/-- The theorem stating that the sum reaches its maximum at n = 20 -/
theorem sum_max (seq : ArithmeticSequence) :
  ∀ k : ℕ, sum_n seq 20 ≥ sum_n seq k :=
sorry

end NUMINAMATH_CALUDE_sum_max_l753_75356


namespace NUMINAMATH_CALUDE_voucher_draw_theorem_l753_75389

/-- The number of apple cards in the bag -/
def num_apple : ℕ := 4

/-- The number of pear cards in the bag -/
def num_pear : ℕ := 4

/-- The total number of cards in the bag -/
def total_cards : ℕ := num_apple + num_pear

/-- The number of cards drawn -/
def cards_drawn : ℕ := 4

/-- The voucher amount random variable -/
inductive VoucherAmount : Type
  | zero : VoucherAmount
  | five : VoucherAmount
  | ten : VoucherAmount

/-- The probability of drawing 4 apple cards -/
def prob_four_apples : ℚ := 1 / 70

/-- The probability distribution of the voucher amount -/
def prob_distribution (x : VoucherAmount) : ℚ :=
  match x with
  | VoucherAmount.zero => 18 / 35
  | VoucherAmount.five => 16 / 35
  | VoucherAmount.ten => 1 / 35

/-- The expected value of the voucher amount -/
def expected_value : ℚ := 18 / 7

/-- Theorem stating the correctness of the probability and expected value calculations -/
theorem voucher_draw_theorem :
  (prob_four_apples = 1 / 70) ∧
  (∀ x, prob_distribution x = match x with
    | VoucherAmount.zero => 18 / 35
    | VoucherAmount.five => 16 / 35
    | VoucherAmount.ten => 1 / 35) ∧
  (expected_value = 18 / 7) := by sorry

end NUMINAMATH_CALUDE_voucher_draw_theorem_l753_75389


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_33_l753_75362

theorem consecutive_integers_sqrt_33 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 33) → (Real.sqrt 33 < b) → (a + b = 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_33_l753_75362


namespace NUMINAMATH_CALUDE_cape_may_august_sharks_l753_75358

/-- The number of sharks in Daytona Beach in July -/
def daytona_july : ℕ := 23

/-- The number of sharks in Cape May in July -/
def cape_may_july : ℕ := 2 * daytona_july

/-- The number of sharks in Daytona Beach in August -/
def daytona_august : ℕ := daytona_july

/-- The number of sharks in Cape May in August -/
def cape_may_august : ℕ := 5 + 3 * daytona_august

theorem cape_may_august_sharks : cape_may_august = 74 := by
  sorry

end NUMINAMATH_CALUDE_cape_may_august_sharks_l753_75358


namespace NUMINAMATH_CALUDE_exists_polyhedron_no_three_same_sided_faces_l753_75378

/-- A face of a polyhedron --/
structure Face where
  sides : ℕ

/-- A polyhedron --/
structure Polyhedron where
  faces : List Face

/-- Predicate to check if a polyhedron has no three faces with the same number of sides --/
def has_no_three_same_sided_faces (p : Polyhedron) : Prop :=
  ∀ n : ℕ, (p.faces.filter (λ f => f.sides = n)).length < 3

/-- Theorem stating the existence of a polyhedron with no three faces having the same number of sides --/
theorem exists_polyhedron_no_three_same_sided_faces :
  ∃ p : Polyhedron, has_no_three_same_sided_faces p ∧ p.faces.length = 6 :=
sorry

end NUMINAMATH_CALUDE_exists_polyhedron_no_three_same_sided_faces_l753_75378


namespace NUMINAMATH_CALUDE_square_commutes_with_multiplication_l753_75311

theorem square_commutes_with_multiplication (m n : ℝ) : m^2 * n - n * m^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_commutes_with_multiplication_l753_75311


namespace NUMINAMATH_CALUDE_consecutive_six_product_not_776965920_l753_75395

theorem consecutive_six_product_not_776965920 (n : ℕ) : 
  n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ≠ 776965920 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_six_product_not_776965920_l753_75395


namespace NUMINAMATH_CALUDE_valid_parameterization_l753_75335

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a vector parameterization of a line -/
structure VectorParam where
  point : Vector2D
  direction : Vector2D

def isOnLine (v : Vector2D) : Prop :=
  v.y = 3 * v.x + 5

def isParallel (v : Vector2D) : Prop :=
  ∃ k : ℝ, v.x = k * 1 ∧ v.y = k * 3

theorem valid_parameterization (param : VectorParam) :
  (isOnLine param.point ∧ isParallel param.direction) ↔
  ∀ t : ℝ, isOnLine (Vector2D.mk
    (param.point.x + t * param.direction.x)
    (param.point.y + t * param.direction.y)) :=
sorry

end NUMINAMATH_CALUDE_valid_parameterization_l753_75335


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_diff_smallest_positive_integer_l753_75328

theorem smallest_n_for_sqrt_diff (n : ℕ) : n ≥ 10001 ↔ Real.sqrt n - Real.sqrt (n - 1) < 0.005 := by
  sorry

theorem smallest_positive_integer : ∀ m : ℕ, m < 10001 → Real.sqrt m - Real.sqrt (m - 1) ≥ 0.005 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_diff_smallest_positive_integer_l753_75328


namespace NUMINAMATH_CALUDE_nine_by_nine_min_unoccupied_l753_75365

/-- Represents a chessboard with grasshoppers -/
structure Chessboard :=
  (size : Nat)
  (initial_grasshoppers : Nat)
  (diagonal_jump : Bool)

/-- Calculates the minimum number of unoccupied squares after jumps -/
def min_unoccupied_squares (board : Chessboard) : Nat :=
  sorry

/-- Theorem stating the minimum number of unoccupied squares for a 9x9 board -/
theorem nine_by_nine_min_unoccupied (board : Chessboard) : 
  board.size = 9 ∧ 
  board.initial_grasshoppers = 9 * 9 ∧ 
  board.diagonal_jump = true →
  min_unoccupied_squares board = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_by_nine_min_unoccupied_l753_75365


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l753_75323

/-- Calculates the length of a bridge given a person's walking speed and time to cross -/
theorem bridge_length_calculation (speed : ℝ) (time_minutes : ℝ) : 
  speed = 6 → time_minutes = 15 → speed * (time_minutes / 60) = 1.5 := by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l753_75323


namespace NUMINAMATH_CALUDE_min_value_expression_l753_75320

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 4) * (b^2 + 4*b + 4) * (c^2 + 4*c + 4) / (a*b*c) ≥ 64 ∧
  (∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    (a₀^2 + 4*a₀ + 4) * (b₀^2 + 4*b₀ + 4) * (c₀^2 + 4*c₀ + 4) / (a₀*b₀*c₀) = 64) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l753_75320


namespace NUMINAMATH_CALUDE_scientific_notation_125000_l753_75322

theorem scientific_notation_125000 : 
  125000 = 1.25 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_125000_l753_75322


namespace NUMINAMATH_CALUDE_tax_percentage_proof_l753_75305

/-- 
Given:
- total_income: The total annual income
- after_tax_income: The income left after paying taxes

Prove that the percentage of income paid in taxes is 18%
-/
theorem tax_percentage_proof (total_income after_tax_income : ℝ) 
  (h1 : total_income = 60000)
  (h2 : after_tax_income = 49200) :
  (total_income - after_tax_income) / total_income * 100 = 18 := by
  sorry


end NUMINAMATH_CALUDE_tax_percentage_proof_l753_75305


namespace NUMINAMATH_CALUDE_exactly_21_numbers_reach_one_in_8_steps_l753_75324

def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 1

def reachesOneIn (steps : ℕ) (n : ℕ) : Prop :=
  ∃ (sequence : Fin (steps + 1) → ℕ),
    sequence 0 = n ∧
    sequence steps = 1 ∧
    ∀ i : Fin steps, sequence (i + 1) = operation (sequence i)

theorem exactly_21_numbers_reach_one_in_8_steps :
  ∃! (s : Finset ℕ), s.card = 21 ∧ ∀ n, n ∈ s ↔ reachesOneIn 8 n :=
sorry

end NUMINAMATH_CALUDE_exactly_21_numbers_reach_one_in_8_steps_l753_75324


namespace NUMINAMATH_CALUDE_jean_price_satisfies_conditions_l753_75329

/-- The price of a jean that satisfies the given conditions -/
def jean_price : ℝ := 11

/-- The price of a tee -/
def tee_price : ℝ := 8

/-- The number of tees sold -/
def tees_sold : ℕ := 7

/-- The number of jeans sold -/
def jeans_sold : ℕ := 4

/-- The total revenue -/
def total_revenue : ℝ := 100

/-- Theorem stating that the jean price satisfies the given conditions -/
theorem jean_price_satisfies_conditions :
  tee_price * tees_sold + jean_price * jeans_sold = total_revenue := by
  sorry

#check jean_price_satisfies_conditions

end NUMINAMATH_CALUDE_jean_price_satisfies_conditions_l753_75329


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l753_75316

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l753_75316


namespace NUMINAMATH_CALUDE_building_units_count_l753_75361

/-- Represents the number of units in a building -/
structure Building where
  oneBedroom : ℕ
  twoBedroom : ℕ

/-- The total cost of all units in the building -/
def totalCost (b : Building) : ℕ := 360 * b.oneBedroom + 450 * b.twoBedroom

/-- The total number of units in the building -/
def totalUnits (b : Building) : ℕ := b.oneBedroom + b.twoBedroom

theorem building_units_count :
  ∃ (b : Building),
    totalCost b = 4950 ∧
    b.twoBedroom = 7 ∧
    totalUnits b = 12 := by
  sorry

end NUMINAMATH_CALUDE_building_units_count_l753_75361


namespace NUMINAMATH_CALUDE_reciprocal_sum_fractions_l753_75385

theorem reciprocal_sum_fractions : (((3 : ℚ) / 4 + (1 : ℚ) / 6)⁻¹) = (12 : ℚ) / 11 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_fractions_l753_75385


namespace NUMINAMATH_CALUDE_definite_integral_equals_twenty_minus_six_pi_l753_75353

theorem definite_integral_equals_twenty_minus_six_pi :
  let f : ℝ → ℝ := λ x => x^4 / ((16 - x^2) * Real.sqrt (16 - x^2))
  let a : ℝ := 0
  let b : ℝ := 2 * Real.sqrt 2
  ∫ x in a..b, f x = 20 - 6 * Real.pi := by sorry

end NUMINAMATH_CALUDE_definite_integral_equals_twenty_minus_six_pi_l753_75353


namespace NUMINAMATH_CALUDE_sum_squares_and_products_ge_ten_l753_75317

theorem sum_squares_and_products_ge_ten (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (prod_eq_one : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_and_products_ge_ten_l753_75317


namespace NUMINAMATH_CALUDE_ceiling_square_fraction_plus_eighth_l753_75355

theorem ceiling_square_fraction_plus_eighth : ⌈(-7/4)^2 + 1/8⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_square_fraction_plus_eighth_l753_75355


namespace NUMINAMATH_CALUDE_restaurant_glasses_count_l753_75384

theorem restaurant_glasses_count :
  ∀ (x y : ℕ),
  -- x is the number of small boxes (12 glasses each)
  -- y is the number of large boxes (16 glasses each)
  y = x + 16 →  -- There are 16 more large boxes
  (12 * x + 16 * y) / (x + y) = 15 →  -- Average number of glasses per box is 15
  12 * x + 16 * y = 480  -- Total number of glasses
  := by sorry

end NUMINAMATH_CALUDE_restaurant_glasses_count_l753_75384


namespace NUMINAMATH_CALUDE_two_digit_product_4536_l753_75357

theorem two_digit_product_4536 (a b : ℕ) 
  (h1 : 10 ≤ a ∧ a < 100) 
  (h2 : 10 ≤ b ∧ b < 100) 
  (h3 : a * b = 4536) 
  (h4 : a ≤ b) : 
  a = 21 := by
sorry

end NUMINAMATH_CALUDE_two_digit_product_4536_l753_75357


namespace NUMINAMATH_CALUDE_set_relationship_l753_75376

def M : Set ℝ := {x : ℝ | ∃ m : ℤ, x = m + 1/6}
def S : Set ℝ := {x : ℝ | ∃ s : ℤ, x = 1/2 * s - 1/3}
def P : Set ℝ := {x : ℝ | ∃ p : ℤ, x = 1/2 * p + 1/6}

theorem set_relationship : M ⊆ S ∧ S = P := by sorry

end NUMINAMATH_CALUDE_set_relationship_l753_75376


namespace NUMINAMATH_CALUDE_f_inequality_implies_a_range_l753_75393

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x^2 + 4*x else Real.log (x + 1)

-- State the theorem
theorem f_inequality_implies_a_range :
  (∀ x, |f x| ≥ a * x) → a ∈ Set.Icc (-4) 0 :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_implies_a_range_l753_75393


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l753_75349

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.35 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l753_75349


namespace NUMINAMATH_CALUDE_wrong_value_correction_l753_75372

theorem wrong_value_correction (n : ℕ) (initial_mean correct_mean correct_value : ℝ) 
  (h1 : n = 20)
  (h2 : initial_mean = 150)
  (h3 : correct_mean = 151.25)
  (h4 : correct_value = 160) :
  ∃ x : ℝ, n * initial_mean - x + correct_value = n * correct_mean ∧ x = 135 := by
  sorry

end NUMINAMATH_CALUDE_wrong_value_correction_l753_75372


namespace NUMINAMATH_CALUDE_postman_pete_miles_l753_75308

/-- Represents a pedometer with a maximum step count before resetting --/
structure Pedometer where
  max_steps : ℕ
  resets : ℕ
  final_reading : ℕ

/-- Calculates the total steps recorded by a pedometer --/
def total_steps (p : Pedometer) : ℕ :=
  p.max_steps * (p.resets + 1) + p.final_reading

/-- Converts steps to miles, rounded to the nearest mile --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  (steps + steps_per_mile / 2) / steps_per_mile

/-- Theorem stating the total miles walked by Postman Pete --/
theorem postman_pete_miles :
  let p : Pedometer := { max_steps := 100000, resets := 48, final_reading := 25000 }
  let steps_per_mile : ℕ := 1600
  steps_to_miles (total_steps p) steps_per_mile = 3016 := by
  sorry

end NUMINAMATH_CALUDE_postman_pete_miles_l753_75308


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odds_l753_75338

theorem largest_divisor_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  ∃ (k : ℕ), k = 105 ∧ 
  (∀ (d : ℕ), d ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) → d ≤ k) ∧
  k ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odds_l753_75338


namespace NUMINAMATH_CALUDE_smallest_y_value_l753_75345

theorem smallest_y_value (x y : ℝ) 
  (h1 : 2 < x ∧ x < y)
  (h2 : 2 + x ≤ y)
  (h3 : 1 / x + 1 / y ≤ 1) :
  y ≥ 2 + Real.sqrt 2 ∧ ∀ z, (2 < z ∧ 2 + z ≤ y ∧ 1 / z + 1 / y ≤ 1) → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_value_l753_75345


namespace NUMINAMATH_CALUDE_expand_polynomial_l753_75397

theorem expand_polynomial (x : ℝ) : (x + 3) * (2*x^2 - x + 4) = 2*x^3 + 5*x^2 + x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l753_75397


namespace NUMINAMATH_CALUDE_smallest_a_in_special_progression_l753_75301

theorem smallest_a_in_special_progression (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : 2 * b = a + c)  -- arithmetic progression
  (h4 : a * a = c * b)  -- geometric progression
  : a ≥ 1 ∧ ∃ (a₀ b₀ c₀ : ℤ), a₀ = 1 ∧ 
    a₀ < b₀ ∧ b₀ < c₀ ∧ 
    2 * b₀ = a₀ + c₀ ∧
    a₀ * a₀ = c₀ * b₀ := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_in_special_progression_l753_75301


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l753_75336

/-- The repeating decimal 0.8̄23 as a rational number -/
def repeating_decimal : ℚ := 0.8 + 23 / 99

/-- The expected fraction representation of 0.8̄23 -/
def expected_fraction : ℚ := 511 / 495

/-- Theorem stating that the repeating decimal 0.8̄23 is equal to 511/495 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = expected_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l753_75336


namespace NUMINAMATH_CALUDE_largest_integer_l753_75334

theorem largest_integer (a b c d : ℤ) 
  (sum1 : a + b + c = 210)
  (sum2 : a + b + d = 230)
  (sum3 : a + c + d = 245)
  (sum4 : b + c + d = 260) :
  max a (max b (max c d)) = 105 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_l753_75334


namespace NUMINAMATH_CALUDE_leftover_value_l753_75318

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents a person's coin collection --/
structure CoinCollection where
  quarters : Nat
  dimes : Nat

/-- Calculates the dollar value of a given number of quarters and dimes --/
def dollarValue (quarters dimes : Nat) : ℚ :=
  (quarters : ℚ) * (1 / 4) + (dimes : ℚ) * (1 / 10)

/-- Theorem stating the dollar value of leftover coins --/
theorem leftover_value (roll_size : RollSize) (ana_coins ben_coins : CoinCollection) :
  roll_size.quarters = 30 →
  roll_size.dimes = 40 →
  ana_coins.quarters = 95 →
  ana_coins.dimes = 183 →
  ben_coins.quarters = 104 →
  ben_coins.dimes = 219 →
  dollarValue 
    ((ana_coins.quarters + ben_coins.quarters) % roll_size.quarters)
    ((ana_coins.dimes + ben_coins.dimes) % roll_size.dimes) = 695 / 100 := by
  sorry

#eval dollarValue 19 22

end NUMINAMATH_CALUDE_leftover_value_l753_75318


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l753_75394

theorem simplify_fraction_product : 
  (36 : ℚ) / 51 * 35 / 24 * 68 / 49 = 20 / 7 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l753_75394


namespace NUMINAMATH_CALUDE_plane_relations_l753_75339

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (in_plane : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem plane_relations (a b : Plane) (h : a ≠ b) :
  (∀ (l : Line), in_plane l a → 
    (∀ (m : Line), in_plane m b → perpendicular l m) → 
    perpendicular_planes a b) ∧
  (∀ (l : Line), in_plane l a → 
    parallel_line_plane l b → 
    parallel_planes a b) ∧
  (parallel_planes a b → 
    ∀ (l : Line), in_plane l a → 
    parallel_line_plane l b) :=
by sorry

end NUMINAMATH_CALUDE_plane_relations_l753_75339


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l753_75352

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, x < 2 ∧ ¬(1 < x ∧ x < 2)) ∧ 
  (∀ x : ℝ, 1 < x ∧ x < 2 → x < 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l753_75352


namespace NUMINAMATH_CALUDE_combined_weight_of_acids_l753_75360

/-- The atomic mass of carbon in g/mol -/
def carbon_mass : ℝ := 12.01

/-- The atomic mass of hydrogen in g/mol -/
def hydrogen_mass : ℝ := 1.01

/-- The atomic mass of oxygen in g/mol -/
def oxygen_mass : ℝ := 16.00

/-- The atomic mass of sulfur in g/mol -/
def sulfur_mass : ℝ := 32.07

/-- The molar mass of C6H8O7 in g/mol -/
def citric_acid_mass : ℝ := 6 * carbon_mass + 8 * hydrogen_mass + 7 * oxygen_mass

/-- The molar mass of H2SO4 in g/mol -/
def sulfuric_acid_mass : ℝ := 2 * hydrogen_mass + sulfur_mass + 4 * oxygen_mass

/-- The number of moles of C6H8O7 -/
def citric_acid_moles : ℝ := 8

/-- The number of moles of H2SO4 -/
def sulfuric_acid_moles : ℝ := 4

/-- The combined weight of C6H8O7 and H2SO4 in grams -/
def combined_weight : ℝ := citric_acid_moles * citric_acid_mass + sulfuric_acid_moles * sulfuric_acid_mass

theorem combined_weight_of_acids : combined_weight = 1929.48 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_of_acids_l753_75360


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l753_75340

theorem absolute_value_inequality (x : ℝ) : |2*x + 1| < 3 ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l753_75340


namespace NUMINAMATH_CALUDE_yolanda_rate_l753_75392

def total_distance : ℝ := 31
def bob_distance : ℝ := 20
def bob_rate : ℝ := 2

theorem yolanda_rate (total_distance : ℝ) (bob_distance : ℝ) (bob_rate : ℝ) :
  total_distance = 31 →
  bob_distance = 20 →
  bob_rate = 2 →
  ∃ yolanda_rate : ℝ,
    yolanda_rate = (total_distance - bob_distance) / (bob_distance / bob_rate) ∧
    yolanda_rate = 1.1 :=
by sorry

end NUMINAMATH_CALUDE_yolanda_rate_l753_75392


namespace NUMINAMATH_CALUDE_function_characterization_l753_75371

def SatisfiesEquation (f : ℤ → ℤ) : Prop :=
  ∀ a b c : ℤ, a + b + c = 0 →
    f a ^ 2 + f b ^ 2 + f c ^ 2 = 2 * f a * f b + 2 * f b * f c + 2 * f c * f a

def IsZeroFunction (f : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, f x = 0

def IsQuadraticFunction (f : ℤ → ℤ) : Prop :=
  ∃ k : ℤ, ∀ x : ℤ, f x = k * x ^ 2

def IsEvenOddFunction (f : ℤ → ℤ) : Prop :=
  ∃ k : ℤ, ∀ x : ℤ, 
    (Even x → f x = 0) ∧ 
    (Odd x → f x = k)

def IsModFourFunction (f : ℤ → ℤ) : Prop :=
  ∃ k : ℤ, ∀ x : ℤ,
    (x % 4 = 0 → f x = 0) ∧
    (x % 4 = 1 → f x = k) ∧
    (x % 4 = 2 → f x = 4 * k)

theorem function_characterization (f : ℤ → ℤ) : 
  SatisfiesEquation f → 
    IsZeroFunction f ∨ 
    IsQuadraticFunction f ∨ 
    IsEvenOddFunction f ∨ 
    IsModFourFunction f := by
  sorry

end NUMINAMATH_CALUDE_function_characterization_l753_75371


namespace NUMINAMATH_CALUDE_parabola_sum_l753_75347

/-- Represents a parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum (p : Parabola) : 
  p.x_coord (-6) = 7 → p.x_coord (-4) = 5 → p.a + p.b + p.c = -42 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l753_75347


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l753_75313

theorem ratio_sum_theorem (a b : ℕ) (h1 : a * 4 = b * 3) (h2 : a = 180) : a + b = 420 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l753_75313


namespace NUMINAMATH_CALUDE_sin_330_degrees_l753_75387

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l753_75387


namespace NUMINAMATH_CALUDE_mikes_video_game_earnings_l753_75307

theorem mikes_video_game_earnings :
  let working_game_prices : List ℕ := [5, 7, 12, 9, 6, 15, 11, 10]
  List.sum working_game_prices = 75 := by
sorry

end NUMINAMATH_CALUDE_mikes_video_game_earnings_l753_75307


namespace NUMINAMATH_CALUDE_luke_fish_catching_l753_75341

theorem luke_fish_catching (days : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ) :
  days = 30 →
  fillets_per_fish = 2 →
  total_fillets = 120 →
  total_fillets / (days * fillets_per_fish) = 2 :=
by sorry

end NUMINAMATH_CALUDE_luke_fish_catching_l753_75341


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l753_75333

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156) 
  (h2 : a*b + b*c + c*a = 50) : 
  a + b + c = 16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l753_75333


namespace NUMINAMATH_CALUDE_amy_money_calculation_l753_75330

theorem amy_money_calculation (initial : ℕ) (chores : ℕ) (birthday : ℕ) : 
  initial = 2 → chores = 13 → birthday = 3 → initial + chores + birthday = 18 := by
  sorry

end NUMINAMATH_CALUDE_amy_money_calculation_l753_75330


namespace NUMINAMATH_CALUDE_juan_running_time_l753_75380

/-- Given Juan's running distance and speed, prove that his running time is 8 hours. -/
theorem juan_running_time (distance : ℝ) (speed : ℝ) (h1 : distance = 80) (h2 : speed = 10) :
  distance / speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_juan_running_time_l753_75380


namespace NUMINAMATH_CALUDE_transformed_dataset_properties_l753_75350

/-- Represents a dataset with its average and variance -/
structure Dataset where
  average : ℝ
  variance : ℝ

/-- Represents a linear transformation of a dataset -/
structure LinearTransform where
  scale : ℝ
  shift : ℝ

/-- Theorem stating the properties of a transformed dataset -/
theorem transformed_dataset_properties (original : Dataset) (transform : LinearTransform) :
  original.average = 3 ∧ 
  original.variance = 4 ∧ 
  transform.scale = 3 ∧ 
  transform.shift = -1 →
  ∃ (transformed : Dataset),
    transformed.average = 8 ∧
    transformed.variance = 36 := by
  sorry

end NUMINAMATH_CALUDE_transformed_dataset_properties_l753_75350


namespace NUMINAMATH_CALUDE_point_on_graph_and_coordinate_sum_l753_75303

theorem point_on_graph_and_coordinate_sum 
  (f : ℝ → ℝ) 
  (h : f 6 = 10) : 
  ∃ (x y : ℝ), 
    x = 2 ∧ 
    y = 28.5 ∧ 
    2 * y = 5 * f (3 * x) + 7 ∧ 
    x + y = 30.5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_and_coordinate_sum_l753_75303


namespace NUMINAMATH_CALUDE_complex_multiplication_l753_75374

theorem complex_multiplication : (Complex.I : ℂ) * (1 - Complex.I) = 1 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l753_75374


namespace NUMINAMATH_CALUDE_book_writing_time_difference_l753_75342

/-- The time difference in months between Ivanka's and Woody's book writing time -/
def time_difference (ivanka_time woody_time : ℕ) : ℕ :=
  ivanka_time - woody_time

/-- Proof that the time difference is 3 months given the conditions -/
theorem book_writing_time_difference :
  ∀ (ivanka_time woody_time : ℕ),
    woody_time = 18 →
    ivanka_time + woody_time = 39 →
    time_difference ivanka_time woody_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_book_writing_time_difference_l753_75342


namespace NUMINAMATH_CALUDE_number_division_problem_l753_75379

theorem number_division_problem (x : ℝ) (h : (x - 5) / 7 = 7) :
  ∃ y : ℝ, (x - 34) / y = 2 ∧ y = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_division_problem_l753_75379


namespace NUMINAMATH_CALUDE_right_triangle_existence_condition_l753_75369

/-- A right triangle with hypotenuse c and median s_a to one of the legs. -/
structure RightTriangle (c s_a : ℝ) :=
  (hypotenuse_positive : c > 0)
  (median_positive : s_a > 0)
  (right_angle : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2)
  (median_property : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 ∧ s_a^2 = (a/2)^2 + (c/2)^2)

/-- The existence condition for a right triangle with given hypotenuse and median. -/
theorem right_triangle_existence_condition (c s_a : ℝ) :
  (∃ (t : RightTriangle c s_a), True) ↔ (c/2 < s_a ∧ s_a < c) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_existence_condition_l753_75369


namespace NUMINAMATH_CALUDE_all_positive_integers_in_A_l753_75375

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define the properties of set A
def HasPropertyA (A : Set ℕ) : Prop :=
  A ⊆ PositiveIntegers ∧
  (∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  (∀ m : ℕ, m ∈ A → ∀ d : ℕ, d > 0 ∧ m % d = 0 → d ∈ A) ∧
  (∀ b c : ℕ, b ∈ A → c ∈ A → 1 < b → b < c → (1 + b * c) ∈ A)

-- Theorem statement
theorem all_positive_integers_in_A (A : Set ℕ) (h : HasPropertyA A) :
  A = PositiveIntegers := by
  sorry

end NUMINAMATH_CALUDE_all_positive_integers_in_A_l753_75375


namespace NUMINAMATH_CALUDE_ball_probability_l753_75388

theorem ball_probability (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h1 : total = 60) 
  (h2 : red = 6) 
  (h3 : purple = 9) : 
  (total - (red + purple)) / total = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l753_75388


namespace NUMINAMATH_CALUDE_water_used_l753_75332

theorem water_used (total_liquid oil : ℝ) (h1 : total_liquid = 1.33) (h2 : oil = 0.17) :
  total_liquid - oil = 1.16 := by
  sorry

end NUMINAMATH_CALUDE_water_used_l753_75332


namespace NUMINAMATH_CALUDE_max_value_of_f_on_I_l753_75377

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval
def I : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem max_value_of_f_on_I :
  ∃ (m : ℝ), m = 2 ∧ ∀ x ∈ I, f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_I_l753_75377


namespace NUMINAMATH_CALUDE_recycling_project_weight_l753_75359

-- Define the number of items collected by each person
def marcus_bottles : ℕ := 25
def marcus_cans : ℕ := 30
def john_bottles : ℕ := 20
def john_cans : ℕ := 25
def sophia_bottles : ℕ := 15
def sophia_cans : ℕ := 35

-- Define the weight of each item
def bottle_weight : ℚ := 0.5
def can_weight : ℚ := 0.025

-- Define the total weight function
def total_weight : ℚ :=
  (marcus_bottles + john_bottles + sophia_bottles) * bottle_weight +
  (marcus_cans + john_cans + sophia_cans) * can_weight

-- Theorem statement
theorem recycling_project_weight :
  total_weight = 32.25 := by sorry

end NUMINAMATH_CALUDE_recycling_project_weight_l753_75359


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_third_l753_75351

theorem cos_alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/6) = 1/3) :
  Real.cos (α + π/3) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_third_l753_75351


namespace NUMINAMATH_CALUDE_expression_multiple_of_six_l753_75326

theorem expression_multiple_of_six (n : ℕ) (h : n ≥ 10) :
  ∃ k : ℤ, ((n + 3).factorial - (n + 1).factorial) / n.factorial = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_expression_multiple_of_six_l753_75326


namespace NUMINAMATH_CALUDE_coin_distribution_l753_75327

theorem coin_distribution (x y : ℕ) : 
  x + y = 16 → 
  x^2 - y^2 = 16 * (x - y) → 
  x = 8 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_l753_75327


namespace NUMINAMATH_CALUDE_min_value_theorem_l753_75343

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 4) :
  (2/x + 1/y) ≥ 2 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y = 4 ∧ 2/x + 1/y = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l753_75343


namespace NUMINAMATH_CALUDE_negation_equivalence_l753_75399

theorem negation_equivalence : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 1 > 0)) ↔ (∀ x : ℝ, x > 0 → x^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l753_75399


namespace NUMINAMATH_CALUDE_max_value_when_m_2_range_of_sum_when_parallel_tangents_l753_75367

noncomputable section

def f (m : ℝ) (x : ℝ) : ℝ := (m + 1/m) * Real.log x + 1/x - x

theorem max_value_when_m_2 :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 2 x ≥ f 2 y ∧ f 2 x = 5/2 * Real.log 2 - 3/2 := by sorry

theorem range_of_sum_when_parallel_tangents :
  ∀ (m : ℝ), m ≥ 3 →
    ∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ →
      (deriv (f m) x₁ = deriv (f m) x₂) →
        x₁ + x₂ > 6/5 ∧ ∀ (ε : ℝ), ε > 0 →
          ∃ (y₁ y₂ : ℝ), y₁ > 0 ∧ y₂ > 0 ∧ y₁ ≠ y₂ ∧
            (deriv (f m) y₁ = deriv (f m) y₂) ∧
            y₁ + y₂ < 6/5 + ε := by sorry

end NUMINAMATH_CALUDE_max_value_when_m_2_range_of_sum_when_parallel_tangents_l753_75367


namespace NUMINAMATH_CALUDE_central_angle_values_l753_75337

/-- A circular sector with given perimeter and area -/
structure CircularSector where
  perimeter : ℝ
  area : ℝ

/-- The central angle of a circular sector in radians -/
def central_angle (s : CircularSector) : Set ℝ :=
  {θ : ℝ | ∃ r : ℝ, 
    s.area = 1/2 * r^2 * θ ∧ 
    s.perimeter = 2 * r + r * θ}

/-- Theorem: For a circular sector with perimeter 3 cm and area 1/2 cm², 
    the central angle is either 1 or 4 radians -/
theorem central_angle_values (s : CircularSector) 
  (h_perimeter : s.perimeter = 3)
  (h_area : s.area = 1/2) : 
  central_angle s = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_central_angle_values_l753_75337


namespace NUMINAMATH_CALUDE_find_other_number_l753_75383

theorem find_other_number (A B : ℕ+) (hA : A = 24) (hHCF : Nat.gcd A B = 13) (hLCM : Nat.lcm A B = 312) : B = 169 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l753_75383


namespace NUMINAMATH_CALUDE_expression_simplification_l753_75315

theorem expression_simplification (x : ℝ) (hx : x > 0) :
  (x - 1) / (x^(3/4) + x^(1/2)) * (x^(1/2) + x^(1/4)) / (x^(1/2) + 1) * x^(1/4) + 1 = x^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l753_75315
