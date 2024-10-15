import Mathlib

namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l3106_310661

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → a^2002 + b^2003 = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l3106_310661


namespace NUMINAMATH_CALUDE_square_root_of_10_factorial_div_210_l3106_310630

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem square_root_of_10_factorial_div_210 :
  ∃ (x : ℝ), x > 0 ∧ x^2 = (factorial 10 : ℝ) / 210 ∧ x = 24 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_10_factorial_div_210_l3106_310630


namespace NUMINAMATH_CALUDE_convex_polygon_symmetry_l3106_310668

-- Define a convex polygon
structure ConvexPolygon where
  -- (Add necessary fields for a convex polygon)

-- Define a point inside the polygon
structure InnerPoint (P : ConvexPolygon) where
  point : ℝ × ℝ
  isInside : Bool -- Predicate to check if the point is inside the polygon

-- Define a line passing through a point
structure Line (P : ℝ × ℝ) where
  slope : ℝ
  -- The line is represented by y = slope * (x - P.1) + P.2

-- Function to check if a line divides the polygon into equal areas
def dividesEqualAreas (P : ConvexPolygon) (O : InnerPoint P) (l : Line O.point) : Prop :=
  -- (Add logic to check if the line divides the polygon into equal areas)
  sorry

-- Function to check if a point is the center of symmetry
def isCenterOfSymmetry (P : ConvexPolygon) (O : InnerPoint P) : Prop :=
  -- (Add logic to check if O is the center of symmetry)
  sorry

-- The main theorem
theorem convex_polygon_symmetry (P : ConvexPolygon) (O : InnerPoint P) :
  (∀ l : Line O.point, dividesEqualAreas P O l) → isCenterOfSymmetry P O :=
by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_symmetry_l3106_310668


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3106_310620

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  h1 : a 2 + a 6 = 6
  h2 : (5 * (a 1 + a 5)) / 2 = 35 / 3

/-- The sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n * (seq.a 1 + seq.a n)) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = (2 / 3) * n + 1 / 3) ∧
  (∀ n : ℕ, S seq n ≥ 1) ∧
  (S seq 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3106_310620


namespace NUMINAMATH_CALUDE_intersection_exists_l3106_310613

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2 / 9 - x^2 / 4 = 1

/-- The line equation -/
def line (k x y : ℝ) : Prop := y = k * x

/-- The theorem statement -/
theorem intersection_exists : ∃ k : ℝ, 0 < k ∧ k < 2 ∧ 
  ∃ x y : ℝ, hyperbola x y ∧ line k x y :=
sorry

end NUMINAMATH_CALUDE_intersection_exists_l3106_310613


namespace NUMINAMATH_CALUDE_attitude_gender_relationship_expected_value_X_l3106_310649

-- Define the survey data
def total_sample : ℕ := 200
def male_agree : ℕ := 70
def male_disagree : ℕ := 30
def female_agree : ℕ := 50
def female_disagree : ℕ := 50

-- Define the chi-square function
def chi_square (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% confidence
def critical_value : ℚ := 6635 / 1000

-- Define the probability of agreeing
def p_agree : ℚ := (male_agree + female_agree) / total_sample

-- Theorem 1: Relationship between attitudes and gender
theorem attitude_gender_relationship :
  chi_square total_sample male_agree female_agree male_disagree female_disagree > critical_value :=
sorry

-- Theorem 2: Expected value of X
theorem expected_value_X :
  (3 : ℚ) * p_agree = 9 / 5 :=
sorry

end NUMINAMATH_CALUDE_attitude_gender_relationship_expected_value_X_l3106_310649


namespace NUMINAMATH_CALUDE_total_money_is_305_l3106_310607

/-- The value of a gold coin in dollars -/
def gold_coin_value : ℕ := 50

/-- The value of a silver coin in dollars -/
def silver_coin_value : ℕ := 25

/-- The number of gold coins -/
def num_gold_coins : ℕ := 3

/-- The number of silver coins -/
def num_silver_coins : ℕ := 5

/-- The amount of cash in dollars -/
def cash : ℕ := 30

/-- The total amount of money in dollars -/
def total_money : ℕ := gold_coin_value * num_gold_coins + silver_coin_value * num_silver_coins + cash

theorem total_money_is_305 : total_money = 305 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_305_l3106_310607


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l3106_310673

/-- Given a triangle PQR with angles 3x, x, and 6x, prove that the largest angle is 108° -/
theorem largest_angle_in_triangle (x : ℝ) : 
  x > 0 ∧ 3*x + x + 6*x = 180 → 
  max (3*x) (max x (6*x)) = 108 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l3106_310673


namespace NUMINAMATH_CALUDE_calculate_unknown_interest_rate_l3106_310632

/-- Proves that for a given principal, time period, and interest rate difference, 
    the unknown rate can be calculated. -/
theorem calculate_unknown_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (known_rate : ℝ) 
  (interest_difference : ℝ) 
  (unknown_rate : ℝ)
  (h1 : principal = 7000)
  (h2 : time = 2)
  (h3 : known_rate = 18)
  (h4 : interest_difference = 840)
  (h5 : principal * (known_rate / 100) * time - principal * (unknown_rate / 100) * time = interest_difference) :
  unknown_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_calculate_unknown_interest_rate_l3106_310632


namespace NUMINAMATH_CALUDE_fraction_addition_l3106_310638

theorem fraction_addition : 
  (7 : ℚ) / 12 + (11 : ℚ) / 16 = (61 : ℚ) / 48 :=
by sorry

end NUMINAMATH_CALUDE_fraction_addition_l3106_310638


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l3106_310667

theorem smallest_angle_measure (ABC ABD : ℝ) (h1 : ABC = 24) (h2 : ABD = 20) :
  ∃ CBD : ℝ, CBD = ABC - ABD ∧ CBD = 4 ∧ ∀ x : ℝ, x ≥ 0 → x ≥ CBD := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l3106_310667


namespace NUMINAMATH_CALUDE_tan_sum_pi_fractions_l3106_310653

theorem tan_sum_pi_fractions : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_fractions_l3106_310653


namespace NUMINAMATH_CALUDE_largest_prime_factor_is_29_l3106_310651

def numbers : List Nat := [145, 187, 221, 299, 169]

/-- Returns the largest prime factor of a natural number -/
def largestPrimeFactor (n : Nat) : Nat :=
  sorry

theorem largest_prime_factor_is_29 : 
  ∀ n ∈ numbers, largestPrimeFactor n ≤ 29 ∧ 
  ∃ m ∈ numbers, largestPrimeFactor m = 29 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_is_29_l3106_310651


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l3106_310659

theorem no_valid_arrangement :
  ¬ ∃ (x y : ℕ), 
    90 = x * y ∧ 
    5 ≤ x ∧ x ≤ 20 ∧ 
    Even y :=
by sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l3106_310659


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l3106_310689

-- Define the arithmetic square root function
noncomputable def arithmeticSqrt (x : ℝ) : ℝ := 
  Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_of_nine : arithmeticSqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l3106_310689


namespace NUMINAMATH_CALUDE_michael_digging_time_l3106_310606

/-- Given the conditions of Michael's and his father's hole digging, prove that Michael will take 700 hours to dig his hole. -/
theorem michael_digging_time (father_rate : ℝ) (father_time : ℝ) (michael_depth_diff : ℝ) :
  father_rate = 4 →
  father_time = 400 →
  michael_depth_diff = 400 →
  (2 * (father_rate * father_time) - michael_depth_diff) / father_rate = 700 :=
by sorry

end NUMINAMATH_CALUDE_michael_digging_time_l3106_310606


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_has_odd_factors_196_is_less_than_200_196_greatest_number_less_than_200_with_odd_factors_l3106_310691

def has_odd_number_of_factors (n : ℕ) : Prop :=
  Odd (Finset.card (Finset.filter (·∣n) (Finset.range (n + 1))))

theorem greatest_number_with_odd_factors : 
  ∀ n : ℕ, n < 200 → has_odd_number_of_factors n → n ≤ 196 :=
by sorry

theorem has_odd_factors_196 : has_odd_number_of_factors 196 :=
by sorry

theorem is_less_than_200_196 : 196 < 200 :=
by sorry

theorem greatest_number_less_than_200_with_odd_factors :
  ∃ n : ℕ, n < 200 ∧ has_odd_number_of_factors n ∧
  ∀ m : ℕ, m < 200 → has_odd_number_of_factors m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_has_odd_factors_196_is_less_than_200_196_greatest_number_less_than_200_with_odd_factors_l3106_310691


namespace NUMINAMATH_CALUDE_special_ap_sums_l3106_310617

/-- An arithmetic progression with special properties -/
structure SpecialAP where
  m : ℕ
  n : ℕ
  sum_m_terms : ℕ
  sum_n_terms : ℕ
  h1 : sum_m_terms = n
  h2 : sum_n_terms = m

/-- The sum of (m+n) terms and (m-n) terms for a SpecialAP -/
def special_sums (ap : SpecialAP) : ℤ × ℚ :=
  (-(ap.m + ap.n : ℤ), (ap.m - ap.n : ℚ) * (2 * ap.n + ap.m) / ap.m)

/-- Theorem stating the sums of (m+n) and (m-n) terms for a SpecialAP -/
theorem special_ap_sums (ap : SpecialAP) :
  special_sums ap = (-(ap.m + ap.n : ℤ), (ap.m - ap.n : ℚ) * (2 * ap.n + ap.m) / ap.m) := by
  sorry

#check special_ap_sums

end NUMINAMATH_CALUDE_special_ap_sums_l3106_310617


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3106_310684

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / a + 4 / b ≥ 9 / 2 := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3106_310684


namespace NUMINAMATH_CALUDE_rotate_triangle_forms_cone_l3106_310662

/-- A right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ
  hypotenuse : ℝ
  right_angle : base^2 + height^2 = hypotenuse^2

/-- A cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- The solid formed by rotating a right-angled triangle around one of its right-angle sides -/
def rotateTriangle (t : RightTriangle) : Cone :=
  { radius := t.base, height := t.height }

/-- Theorem: Rotating a right-angled triangle around one of its right-angle sides forms a cone -/
theorem rotate_triangle_forms_cone (t : RightTriangle) :
  ∃ (c : Cone), rotateTriangle t = c :=
sorry

end NUMINAMATH_CALUDE_rotate_triangle_forms_cone_l3106_310662


namespace NUMINAMATH_CALUDE_distance_P_to_xaxis_l3106_310637

/-- The distance from a point to the x-axis is the absolute value of its y-coordinate -/
def distanceToXAxis (p : ℝ × ℝ) : ℝ := |p.2|

/-- Point P with coordinates (-3, 1) -/
def P : ℝ × ℝ := (-3, 1)

/-- Theorem: The distance from point P to the x-axis is 1 -/
theorem distance_P_to_xaxis : distanceToXAxis P = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_xaxis_l3106_310637


namespace NUMINAMATH_CALUDE_peanut_butter_jars_l3106_310665

/-- Given the total amount of peanut butter and jar sizes, calculate the number of jars. -/
def number_of_jars (total_ounces : ℕ) (jar_sizes : List ℕ) : ℕ :=
  if jar_sizes.length = 0 then 0
  else
    let jars_per_size := total_ounces / (jar_sizes.sum)
    jars_per_size * jar_sizes.length

/-- Theorem stating that given 252 ounces of peanut butter in equal numbers of 16, 28, and 40 ounce jars, the total number of jars is 9. -/
theorem peanut_butter_jars :
  number_of_jars 252 [16, 28, 40] = 9 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_jars_l3106_310665


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3106_310634

theorem unique_positive_solution : 
  ∃! (x : ℝ), x > 0 ∧ x^101 + 100^99 = x^99 + 100^101 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3106_310634


namespace NUMINAMATH_CALUDE_rays_number_l3106_310629

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def reverse_digits (n : ℕ) : ℕ := 10 * (n % 10) + (n / 10)

theorem rays_number :
  ∃ n : ℕ,
    is_two_digit n ∧
    n > 4 * (sum_of_digits n) + 3 ∧
    n + 18 = reverse_digits n ∧
    n = 35 := by
  sorry

end NUMINAMATH_CALUDE_rays_number_l3106_310629


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l3106_310694

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def has_tens_digit_2 (n : ℕ) : Prop := n ≥ 20 ∧ n < 30

theorem smallest_two_digit_prime_with_composite_reverse :
  ∃ n : ℕ, is_prime n ∧ 
           is_composite (reverse_digits n) ∧ 
           has_tens_digit_2 n ∧
           (∀ m : ℕ, m < n → ¬(is_prime m ∧ is_composite (reverse_digits m) ∧ has_tens_digit_2 m)) ∧
           n = 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l3106_310694


namespace NUMINAMATH_CALUDE_bob_pie_count_l3106_310647

/-- The radius of Tom's circular pies in cm -/
def tom_radius : ℝ := 8

/-- The number of pies Tom can make in one batch -/
def tom_batch_size : ℕ := 6

/-- The length of one leg of Bob's right-angled triangular pies in cm -/
def bob_leg1 : ℝ := 6

/-- The length of the other leg of Bob's right-angled triangular pies in cm -/
def bob_leg2 : ℝ := 8

/-- The number of pies Bob can make with the same amount of dough as Tom -/
def bob_batch_size : ℕ := 50

theorem bob_pie_count :
  bob_batch_size = ⌊(tom_radius^2 * Real.pi * tom_batch_size) / (bob_leg1 * bob_leg2 / 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_bob_pie_count_l3106_310647


namespace NUMINAMATH_CALUDE_train_crossing_time_l3106_310601

/-- Calculates the time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 100 →
  train_speed = 54.99520038396929 →
  man_speed = 5 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3106_310601


namespace NUMINAMATH_CALUDE_power_calculations_l3106_310603

theorem power_calculations :
  ((-2 : ℤ) ^ (0 : ℕ) = 1) ∧
  ((-3 : ℚ) ^ (-3 : ℤ) = -1/27) := by
  sorry

end NUMINAMATH_CALUDE_power_calculations_l3106_310603


namespace NUMINAMATH_CALUDE_james_and_louise_ages_l3106_310666

theorem james_and_louise_ages :
  ∀ (james louise : ℕ),
  james = louise + 6 →
  james + 8 = 4 * (louise - 4) →
  james + louise = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_james_and_louise_ages_l3106_310666


namespace NUMINAMATH_CALUDE_second_player_strategy_exists_first_player_strategy_exists_l3106_310619

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a 4-digit number -/
def FourDigitNumber := Fin 10000

/-- The game state, representing the current partially filled subtraction problem -/
structure GameState where
  minuend : FourDigitNumber
  subtrahend : FourDigitNumber

/-- A player's move, either calling out a digit or placing a digit -/
inductive Move
  | CallDigit : Digit → Move
  | PlaceDigit : Digit → Nat → Move

/-- The result of the game -/
def gameResult (finalState : GameState) : Int :=
  (finalState.minuend.val : Int) - (finalState.subtrahend.val : Int)

/-- A strategy for a player -/
def Strategy := GameState → Move

/-- Theorem: There exists a strategy for the second player to keep the difference ≤ 4000 -/
theorem second_player_strategy_exists : 
  ∃ (s : Strategy), ∀ (g : GameState), gameResult g ≤ 4000 := by sorry

/-- Theorem: There exists a strategy for the first player to keep the difference ≥ 4000 -/
theorem first_player_strategy_exists :
  ∃ (s : Strategy), ∀ (g : GameState), gameResult g ≥ 4000 := by sorry

end NUMINAMATH_CALUDE_second_player_strategy_exists_first_player_strategy_exists_l3106_310619


namespace NUMINAMATH_CALUDE_sqrt_difference_product_l3106_310655

theorem sqrt_difference_product : (Real.sqrt 6 + Real.sqrt 11) * (Real.sqrt 6 - Real.sqrt 11) = -5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_product_l3106_310655


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l3106_310652

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- State the theorem
theorem f_decreasing_interval :
  ∀ f : ℝ → ℝ, (∀ x, deriv f x = f' x) →
  ∀ x ∈ Set.Ioo 0 2, deriv (fun y ↦ f (y + 1)) x < 0 :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l3106_310652


namespace NUMINAMATH_CALUDE_set_equivalence_l3106_310635

theorem set_equivalence : 
  {x : ℕ+ | x - 3 < 2} = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_set_equivalence_l3106_310635


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3106_310618

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 - 3 * x) ↔ x ≤ 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3106_310618


namespace NUMINAMATH_CALUDE_eight_factorial_equals_product_l3106_310682

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem eight_factorial_equals_product : 4 * 6 * 3 * 560 = factorial 8 := by
  sorry


end NUMINAMATH_CALUDE_eight_factorial_equals_product_l3106_310682


namespace NUMINAMATH_CALUDE_apples_given_away_l3106_310699

/-- Given that Joan picked a certain number of apples and now has fewer,
    prove that the number of apples she gave away is the difference between
    the initial and current number of apples. -/
theorem apples_given_away (initial current : ℕ) (h : current ≤ initial) :
  initial - current = initial - current := by sorry

end NUMINAMATH_CALUDE_apples_given_away_l3106_310699


namespace NUMINAMATH_CALUDE_double_iced_subcubes_count_l3106_310681

/-- Represents a 3D cube with icing on some faces -/
structure IcedCube where
  size : Nat
  top_iced : Bool
  front_iced : Bool
  right_iced : Bool

/-- Counts the number of 1x1x1 subcubes with icing on exactly two faces -/
def count_double_iced_subcubes (cube : IcedCube) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem double_iced_subcubes_count (cake : IcedCube) : 
  cake.size = 5 ∧ cake.top_iced ∧ cake.front_iced ∧ cake.right_iced →
  count_double_iced_subcubes cake = 32 :=
by sorry

end NUMINAMATH_CALUDE_double_iced_subcubes_count_l3106_310681


namespace NUMINAMATH_CALUDE_john_squat_difference_l3106_310605

/-- Given John's raw squat weight, the weight added by sleeves, and the percentage added by wraps,
    calculate the difference between the weight added by wraps and sleeves. -/
def weight_difference (raw_squat : ℝ) (sleeve_addition : ℝ) (wrap_percentage : ℝ) : ℝ :=
  raw_squat * wrap_percentage - sleeve_addition

/-- Prove that the difference between the weight added by wraps and sleeves to John's squat is 120 pounds. -/
theorem john_squat_difference :
  weight_difference 600 30 0.25 = 120 := by
  sorry

end NUMINAMATH_CALUDE_john_squat_difference_l3106_310605


namespace NUMINAMATH_CALUDE_peony_count_l3106_310674

theorem peony_count (n : ℕ) 
  (h1 : ∃ (x : ℕ), n = 4*x + 2*x + 6*x) 
  (h2 : ∃ (y : ℕ), 6*y - 4*y = 30) 
  (h3 : ∃ (z : ℕ), 4 + 2 + 6 = 12) : n = 180 := by
  sorry

end NUMINAMATH_CALUDE_peony_count_l3106_310674


namespace NUMINAMATH_CALUDE_hyperbola_sum_l3106_310636

/-- Given a hyperbola with center (-2, 0), one focus at (-2 + √41, 0), and one vertex at (-7, 0),
    prove that h + k + a + b = 7, where (h, k) is the center, a is the distance from the center
    to a vertex, and b is the length of the conjugate axis. -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -2 ∧ 
  k = 0 ∧ 
  (h + Real.sqrt 41 - h)^2 = c^2 ∧
  (h - 5 - h)^2 = a^2 ∧
  c^2 = a^2 + b^2 →
  h + k + a + b = 7 := by sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l3106_310636


namespace NUMINAMATH_CALUDE_f_composition_eq_one_l3106_310627

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 1 else x - 3

theorem f_composition_eq_one (x : ℝ) :
  f (f x) = 1 ↔ x ∈ Set.union (Set.Icc 0 1) (Set.union (Set.Icc 3 4) {7}) := by
  sorry

end NUMINAMATH_CALUDE_f_composition_eq_one_l3106_310627


namespace NUMINAMATH_CALUDE_wallpaper_three_layers_l3106_310679

/-- Given wallpaper covering conditions, prove the area covered by three layers -/
theorem wallpaper_three_layers
  (total_area : ℝ)
  (wall_area : ℝ)
  (two_layer_area : ℝ)
  (h1 : total_area = 300)
  (h2 : wall_area = 180)
  (h3 : two_layer_area = 30)
  : ∃ (three_layer_area : ℝ),
    three_layer_area = total_area - (wall_area - two_layer_area + two_layer_area) ∧
    three_layer_area = 120 :=
by sorry

end NUMINAMATH_CALUDE_wallpaper_three_layers_l3106_310679


namespace NUMINAMATH_CALUDE_max_silver_tokens_l3106_310688

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules --/
inductive ExchangeRule
  | RedToSilver : ExchangeRule  -- 3 red → 2 silver + 1 blue
  | BlueToSilver : ExchangeRule -- 4 blue → 1 silver + 2 red

/-- Applies an exchange rule to a token count --/
def applyExchange (tc : TokenCount) (rule : ExchangeRule) : Option TokenCount :=
  match rule with
  | ExchangeRule.RedToSilver =>
      if tc.red ≥ 3 then
        some ⟨tc.red - 3, tc.blue + 1, tc.silver + 2⟩
      else
        none
  | ExchangeRule.BlueToSilver =>
      if tc.blue ≥ 4 then
        some ⟨tc.red + 2, tc.blue - 4, tc.silver + 1⟩
      else
        none

/-- Determines if any exchange is possible --/
def canExchange (tc : TokenCount) : Bool :=
  tc.red ≥ 3 ∨ tc.blue ≥ 4

/-- The main theorem to prove --/
theorem max_silver_tokens :
  ∃ (final : TokenCount),
    final.silver = 113 ∧
    ¬(canExchange final) ∧
    (∀ (tc : TokenCount),
      tc.red = 100 ∧ tc.blue = 50 ∧ tc.silver = 0 →
      (∃ (exchanges : List ExchangeRule),
        (exchanges.foldl (λ acc rule => (applyExchange acc rule).getD acc) tc) = final)) :=
  sorry


end NUMINAMATH_CALUDE_max_silver_tokens_l3106_310688


namespace NUMINAMATH_CALUDE_sqrt_81_equals_3_to_m_l3106_310639

theorem sqrt_81_equals_3_to_m (m : ℝ) : (81 : ℝ)^(1/2) = 3^m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_equals_3_to_m_l3106_310639


namespace NUMINAMATH_CALUDE_min_value_problem_l3106_310645

theorem min_value_problem (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 2) :
  (1/3 : ℝ) * x^3 + y^2 + z ≥ 13/12 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l3106_310645


namespace NUMINAMATH_CALUDE_furniture_cost_l3106_310640

/-- Prove that the cost of the furniture is $400, given the conditions of Emma's spending. -/
theorem furniture_cost (initial_amount : ℝ) (remaining_amount : ℝ) 
  (h1 : initial_amount = 2000)
  (h2 : remaining_amount = 400)
  (h3 : ∃ (furniture_cost : ℝ), remaining_amount = (1/4) * (initial_amount - furniture_cost)) :
  ∃ (furniture_cost : ℝ), furniture_cost = 400 := by
  sorry

end NUMINAMATH_CALUDE_furniture_cost_l3106_310640


namespace NUMINAMATH_CALUDE_sam_seashells_l3106_310656

theorem sam_seashells (initial_seashells : ℕ) (given_away : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 35)
  (h2 : given_away = 18)
  (h3 : remaining_seashells = initial_seashells - given_away) :
  remaining_seashells = 17 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l3106_310656


namespace NUMINAMATH_CALUDE_two_books_cost_l3106_310622

/-- The cost of two books, where one is sold at a loss and the other at a gain --/
theorem two_books_cost (C₁ C₂ : ℝ) (h1 : C₁ = 274.1666666666667) 
  (h2 : C₁ * 0.85 = C₂ * 1.19) : 
  abs (C₁ + C₂ - 470) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_two_books_cost_l3106_310622


namespace NUMINAMATH_CALUDE_smallest_p_is_three_l3106_310642

theorem smallest_p_is_three (p q s r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime s → Nat.Prime r →
  p + q + s = r →
  2 < p → p < q → q < s →
  ∀ p' : ℕ, (Nat.Prime p' ∧ 
             (∃ q' s' r' : ℕ, Nat.Prime q' ∧ Nat.Prime s' ∧ Nat.Prime r' ∧
                              p' + q' + s' = r' ∧
                              2 < p' ∧ p' < q' ∧ q' < s')) →
            p' ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_p_is_three_l3106_310642


namespace NUMINAMATH_CALUDE_gift_wrap_sales_l3106_310692

theorem gift_wrap_sales (solid_price print_price total_rolls total_amount : ℝ) 
  (h1 : solid_price = 4)
  (h2 : print_price = 6)
  (h3 : total_rolls = 480)
  (h4 : total_amount = 2340)
  : ∃ (solid_rolls print_rolls : ℝ),
    solid_rolls + print_rolls = total_rolls ∧
    solid_price * solid_rolls + print_price * print_rolls = total_amount ∧
    print_rolls = 210 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrap_sales_l3106_310692


namespace NUMINAMATH_CALUDE_triangles_in_ten_point_config_l3106_310693

/-- Represents a configuration of points on a circle with chords --/
structure CircleConfiguration where
  numPoints : ℕ
  numChords : ℕ
  numIntersections : ℕ

/-- Calculates the number of triangles formed by chord intersections --/
def numTriangles (config : CircleConfiguration) : ℕ :=
  sorry

/-- The specific configuration for our problem --/
def tenPointConfig : CircleConfiguration :=
  { numPoints := 10
  , numChords := 45
  , numIntersections := 210 }

/-- Theorem stating that the number of triangles in the given configuration is 120 --/
theorem triangles_in_ten_point_config :
  numTriangles tenPointConfig = 120 :=
sorry

end NUMINAMATH_CALUDE_triangles_in_ten_point_config_l3106_310693


namespace NUMINAMATH_CALUDE_marias_trip_distance_l3106_310686

/-- Proves that the total distance of Maria's trip is 450 miles -/
theorem marias_trip_distance (D : ℝ) 
  (first_stop : D - D/3 = 2/3 * D)
  (second_stop : 2/3 * D - 1/4 * (2/3 * D) = 1/2 * D)
  (third_stop : 1/2 * D - 1/5 * (1/2 * D) = 2/5 * D)
  (final_distance : 2/5 * D = 180) :
  D = 450 := by sorry

end NUMINAMATH_CALUDE_marias_trip_distance_l3106_310686


namespace NUMINAMATH_CALUDE_log_product_equals_one_third_l3106_310663

theorem log_product_equals_one_third :
  Real.log 2 / Real.log 3 *
  Real.log 3 / Real.log 4 *
  Real.log 4 / Real.log 5 *
  Real.log 5 / Real.log 6 *
  Real.log 6 / Real.log 7 *
  Real.log 7 / Real.log 8 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_one_third_l3106_310663


namespace NUMINAMATH_CALUDE_max_product_sum_2004_l3106_310616

theorem max_product_sum_2004 :
  (∃ (a b : ℤ), a + b = 2004 ∧ a * b = 1004004) ∧
  (∀ (x y : ℤ), x + y = 2004 → x * y ≤ 1004004) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2004_l3106_310616


namespace NUMINAMATH_CALUDE_purple_pants_count_l3106_310670

/-- Represents the number of shirts Teairra has -/
def total_shirts : ℕ := 5

/-- Represents the number of pants Teairra has -/
def total_pants : ℕ := 24

/-- Represents the number of plaid shirts Teairra has -/
def plaid_shirts : ℕ := 3

/-- Represents the number of items that are neither plaid nor purple -/
def neither_plaid_nor_purple : ℕ := 21

/-- Represents the number of purple pants Teairra has -/
def purple_pants : ℕ := total_pants - (neither_plaid_nor_purple - (total_shirts - plaid_shirts))

theorem purple_pants_count : purple_pants = 5 := by
  sorry

end NUMINAMATH_CALUDE_purple_pants_count_l3106_310670


namespace NUMINAMATH_CALUDE_dasha_ate_one_bowl_l3106_310610

/-- The number of bowls of porridge eaten by each monkey -/
structure MonkeyPorridge where
  masha : ℕ
  dasha : ℕ
  glasha : ℕ
  natasha : ℕ

/-- The conditions of the monkey porridge problem -/
def MonkeyPorridgeConditions (mp : MonkeyPorridge) : Prop :=
  mp.masha + mp.dasha + mp.glasha + mp.natasha = 16 ∧
  mp.glasha + mp.natasha = 9 ∧
  mp.masha > mp.dasha ∧
  mp.masha > mp.glasha ∧
  mp.masha > mp.natasha

theorem dasha_ate_one_bowl (mp : MonkeyPorridge) 
  (h : MonkeyPorridgeConditions mp) : mp.dasha = 1 := by
  sorry

end NUMINAMATH_CALUDE_dasha_ate_one_bowl_l3106_310610


namespace NUMINAMATH_CALUDE_log_simplification_l3106_310633

theorem log_simplification (u v w t : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) (ht : t > 0) :
  Real.log (u / v) + Real.log (v / (2 * w)) + Real.log (w / (4 * t)) - Real.log (u / t) = Real.log (1 / 8) := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l3106_310633


namespace NUMINAMATH_CALUDE_net_amount_is_2550_l3106_310695

/-- Calculates the net amount received from selling puppies given the specified conditions -/
def calculate_net_amount (first_litter : ℕ) (second_litter : ℕ) 
  (first_price : ℕ) (second_price : ℕ) (raising_cost : ℕ) : ℕ :=
  let sold_first := (first_litter * 3) / 4
  let sold_second := (second_litter * 3) / 4
  let revenue := sold_first * first_price + sold_second * second_price
  let expenses := (first_litter + second_litter) * raising_cost
  revenue - expenses

/-- The net amount received from selling puppies under the given conditions is $2550 -/
theorem net_amount_is_2550 : 
  calculate_net_amount 10 12 200 250 50 = 2550 := by
  sorry

end NUMINAMATH_CALUDE_net_amount_is_2550_l3106_310695


namespace NUMINAMATH_CALUDE_disjunction_implies_conjunction_false_l3106_310604

theorem disjunction_implies_conjunction_false : 
  ¬(∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_implies_conjunction_false_l3106_310604


namespace NUMINAMATH_CALUDE_unique_assignment_l3106_310612

-- Define the students and authors as enums
inductive Student : Type
| ZhangBoyuan : Student
| GaoJiaming : Student
| LiuYuheng : Student

inductive Author : Type
| Shakespeare : Author
| Hugo : Author
| CaoXueqin : Author

-- Define the assignment of authors to students
def Assignment := Student → Author

-- Define the condition that each student has a different author
def all_different (a : Assignment) : Prop :=
  ∀ s1 s2 : Student, s1 ≠ s2 → a s1 ≠ a s2

-- Define Teacher Liu's guesses
def guess1 (a : Assignment) : Prop := a Student.ZhangBoyuan = Author.Shakespeare
def guess2 (a : Assignment) : Prop := a Student.LiuYuheng ≠ Author.CaoXueqin
def guess3 (a : Assignment) : Prop := a Student.GaoJiaming ≠ Author.Shakespeare

-- Define the condition that only one guess is correct
def only_one_correct (a : Assignment) : Prop :=
  (guess1 a ∧ ¬guess2 a ∧ ¬guess3 a) ∨
  (¬guess1 a ∧ guess2 a ∧ ¬guess3 a) ∨
  (¬guess1 a ∧ ¬guess2 a ∧ guess3 a)

-- The main theorem
theorem unique_assignment :
  ∃! a : Assignment,
    all_different a ∧
    only_one_correct a ∧
    a Student.ZhangBoyuan = Author.CaoXueqin ∧
    a Student.GaoJiaming = Author.Shakespeare ∧
    a Student.LiuYuheng = Author.Hugo :=
  sorry

end NUMINAMATH_CALUDE_unique_assignment_l3106_310612


namespace NUMINAMATH_CALUDE_carmen_sculpture_height_l3106_310660

/-- Represents a measurement in feet and inches -/
structure FeetInches where
  feet : ℕ
  inches : ℕ
  h_valid : inches < 12

/-- Converts inches to a FeetInches measurement -/
def inchesToFeetInches (totalInches : ℕ) : FeetInches :=
  { feet := totalInches / 12,
    inches := totalInches % 12,
    h_valid := by sorry }

/-- Adds two FeetInches measurements -/
def addFeetInches (a b : FeetInches) : FeetInches :=
  inchesToFeetInches (a.feet * 12 + a.inches + b.feet * 12 + b.inches)

theorem carmen_sculpture_height :
  let rectangular_prism_height : ℕ := 8
  let cylinder_height : ℕ := 15
  let pyramid_height : ℕ := 10
  let base_height : ℕ := 10
  let sculpture_height := rectangular_prism_height + cylinder_height + pyramid_height
  let sculpture_feet_inches := inchesToFeetInches sculpture_height
  let base_feet_inches := inchesToFeetInches base_height
  let combined_height := addFeetInches sculpture_feet_inches base_feet_inches
  combined_height = { feet := 3, inches := 7, h_valid := by sorry } := by sorry

end NUMINAMATH_CALUDE_carmen_sculpture_height_l3106_310660


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_intersection_empty_iff_l3106_310621

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

-- Part 1
theorem intersection_and_union_when_m_is_3 :
  (A ∩ B 3 = {x | 2 ≤ x ∧ x ≤ 5}) ∧
  ((Set.univ \ A) ∪ B 3 = {x | x < -2 ∨ x ≥ 2}) := by sorry

-- Part 2
theorem intersection_empty_iff :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m < -3/2 ∨ m > 6 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_is_3_intersection_empty_iff_l3106_310621


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_arithmetic_sequence_l3106_310615

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- State the theorem
theorem tenth_term_of_specific_arithmetic_sequence : 
  ∃ (a d : ℝ), 
    arithmetic_sequence a d 3 = 10 ∧ 
    arithmetic_sequence a d 6 = 16 ∧ 
    arithmetic_sequence a d 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_arithmetic_sequence_l3106_310615


namespace NUMINAMATH_CALUDE_set_operation_result_l3106_310648

def X : Set ℕ := {0, 1, 2, 4, 5, 7}
def Y : Set ℕ := {1, 3, 6, 8, 9}
def Z : Set ℕ := {3, 7, 8}

theorem set_operation_result : (X ∩ Y) ∪ Z = {1, 3, 7, 8} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l3106_310648


namespace NUMINAMATH_CALUDE_marys_potatoes_l3106_310697

/-- 
Given that Mary has some initial number of potatoes, rabbits ate 3 potatoes, 
and Mary now has 5 potatoes left, prove that Mary initially had 8 potatoes.
-/
theorem marys_potatoes (initial : ℕ) (eaten : ℕ) (remaining : ℕ) : 
  eaten = 3 → remaining = 5 → initial = eaten + remaining → initial = 8 := by
sorry

end NUMINAMATH_CALUDE_marys_potatoes_l3106_310697


namespace NUMINAMATH_CALUDE_triangle_properties_l3106_310678

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given condition
  2 * a = Real.sqrt 3 * c * Real.sin A - a * Real.cos C →
  -- Part 1: Prove C = 2π/3
  C = 2 * π / 3 ∧
  -- Part 2: Prove maximum area is √3/4 when c = √3
  (c = Real.sqrt 3 →
    ∀ (a' b' : ℝ), 
      0 < a' ∧ 0 < b' ∧
      2 * a' = Real.sqrt 3 * c * Real.sin A - a' * Real.cos C →
      1/2 * a' * b' * Real.sin C ≤ Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3106_310678


namespace NUMINAMATH_CALUDE_floor_length_percentage_l3106_310625

/-- Proves that for a rectangular floor with length 20 meters, if the total cost to paint the floor
    at 3 currency units per square meter is 400 currency units, then the length is 200% more than
    the breadth. -/
theorem floor_length_percentage (breadth : ℝ) (percentage : ℝ) : 
  breadth > 0 →
  percentage > 0 →
  20 = breadth * (1 + percentage / 100) →
  400 = 3 * (20 * breadth) →
  percentage = 200 := by
sorry

end NUMINAMATH_CALUDE_floor_length_percentage_l3106_310625


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3106_310690

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ -- Semi-major axis
  b : ℝ -- Semi-minor axis
  f1 : Point -- Focus 1
  f2 : Point -- Focus 2

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop := sorry

theorem ellipse_triangle_perimeter 
  (e : Ellipse) 
  (A B : Point) 
  (h1 : e.a = 5)
  (h2 : isOnEllipse e A)
  (h3 : isOnEllipse e B) :
  distance A B + distance A e.f2 + distance B e.f2 = 4 * e.a := by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3106_310690


namespace NUMINAMATH_CALUDE_total_cans_collected_l3106_310611

def saturday_bags : ℕ := 3
def sunday_bags : ℕ := 4
def cans_per_bag : ℕ := 9

theorem total_cans_collected :
  saturday_bags * cans_per_bag + sunday_bags * cans_per_bag = 63 :=
by sorry

end NUMINAMATH_CALUDE_total_cans_collected_l3106_310611


namespace NUMINAMATH_CALUDE_fruit_filled_mooncake_probability_l3106_310657

def num_fruits : ℕ := 5
def num_meats : ℕ := 4

def combinations (n : ℕ) : ℕ := n * (n - 1) / 2

theorem fruit_filled_mooncake_probability :
  let total_combinations := combinations num_fruits + combinations num_meats
  let fruit_combinations := combinations num_fruits
  (fruit_combinations : ℚ) / total_combinations = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_fruit_filled_mooncake_probability_l3106_310657


namespace NUMINAMATH_CALUDE_cut_to_square_iff_perfect_square_l3106_310698

/-- Represents a figure on a grid -/
structure GridFigure where
  area : ℕ

/-- Represents a cut of the figure -/
inductive Cut
  | Line : Cut

/-- Represents the result of cutting the figure -/
structure CutResult where
  parts : Fin 3 → GridFigure

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- Can form a square from the cut parts -/
def can_form_square (cr : CutResult) : Prop :=
  ∃ side : ℕ, (cr.parts 0).area + (cr.parts 1).area + (cr.parts 2).area = side * side

/-- The main theorem: a figure can be cut into three parts to form a square
    if and only if its area is a perfect square -/
theorem cut_to_square_iff_perfect_square (f : GridFigure) :
  (∃ cuts : List Cut, ∃ cr : CutResult, can_form_square cr) ↔ is_perfect_square f.area :=
sorry

end NUMINAMATH_CALUDE_cut_to_square_iff_perfect_square_l3106_310698


namespace NUMINAMATH_CALUDE_rogers_money_l3106_310623

theorem rogers_money (x : ℤ) : 
  x - 20 + 46 = 71 → x = 45 := by
sorry

end NUMINAMATH_CALUDE_rogers_money_l3106_310623


namespace NUMINAMATH_CALUDE_zero_only_number_unchanged_by_integer_multiplication_l3106_310602

theorem zero_only_number_unchanged_by_integer_multiplication :
  ∀ n : ℤ, (∀ m : ℤ, n * m = n) → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_only_number_unchanged_by_integer_multiplication_l3106_310602


namespace NUMINAMATH_CALUDE_arman_sister_age_ratio_l3106_310654

/-- Given Arman and his sister's ages at different points in time, prove the ratio of their current ages -/
theorem arman_sister_age_ratio :
  ∀ (sister_age_4_years_ago : ℕ) (arman_age_4_years_future : ℕ),
    sister_age_4_years_ago = 2 →
    arman_age_4_years_future = 40 →
    (arman_age_4_years_future - 4) / (sister_age_4_years_ago + 4) = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_arman_sister_age_ratio_l3106_310654


namespace NUMINAMATH_CALUDE_banana_kiwi_equivalence_l3106_310671

-- Define the cost relationship between fruits
def cost_relation (banana pear kiwi : ℕ) : Prop :=
  4 * banana = 3 * pear ∧ 9 * pear = 6 * kiwi

-- Theorem statement
theorem banana_kiwi_equivalence :
  ∀ (banana pear kiwi : ℕ), cost_relation banana pear kiwi → 24 * banana = 12 * kiwi :=
by
  sorry

end NUMINAMATH_CALUDE_banana_kiwi_equivalence_l3106_310671


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_square_sum_l3106_310646

theorem arithmetic_geometric_mean_square_sum (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 125) :
  x^2 + y^2 = 1350 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_square_sum_l3106_310646


namespace NUMINAMATH_CALUDE_distance_against_current_equals_swimming_speed_l3106_310683

/-- The distance swam against the current given swimming speed in still water and current speed -/
def distanceAgainstCurrent (swimmingSpeed currentSpeed : ℝ) : ℝ :=
  swimmingSpeed

theorem distance_against_current_equals_swimming_speed
  (swimmingSpeed currentSpeed : ℝ)
  (h1 : swimmingSpeed = 12)
  (h2 : currentSpeed = 10)
  (h3 : swimmingSpeed > currentSpeed) :
  distanceAgainstCurrent swimmingSpeed currentSpeed = 12 := by
  sorry

#eval distanceAgainstCurrent 12 10

end NUMINAMATH_CALUDE_distance_against_current_equals_swimming_speed_l3106_310683


namespace NUMINAMATH_CALUDE_sequence_sum_bound_l3106_310624

/-- Given a sequence of positive integers satisfying certain conditions, 
    prove that the sum of its first n terms is at most n². -/
theorem sequence_sum_bound (n : ℕ) (a : ℕ → ℕ) : n > 0 →
  (∀ i, a (i + n) = a i) →
  (∀ i ∈ Finset.range n, a i > 0) →
  (∀ i ∈ Finset.range (n - 1), a i ≤ a (i + 1)) →
  a n ≤ a 1 + n →
  (∀ i ∈ Finset.range n, a (a i) ≤ n + i) →
  (Finset.range n).sum a ≤ n^2 := by
  sorry


end NUMINAMATH_CALUDE_sequence_sum_bound_l3106_310624


namespace NUMINAMATH_CALUDE_periodic_points_measure_l3106_310609

open MeasureTheory

theorem periodic_points_measure (f : ℝ → ℝ) (hf : Continuous f) (hf0 : f 0 = 0) (hf1 : f 1 = 0) :
  let A := {h ∈ Set.Icc 0 1 | ∃ x ∈ Set.Icc 0 1, f (x + h) = f x}
  Measurable A ∧ volume A ≥ 1/2 := by
sorry

end NUMINAMATH_CALUDE_periodic_points_measure_l3106_310609


namespace NUMINAMATH_CALUDE_square_perimeter_l3106_310643

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (3 * s = 40) → (4 * s = 160 / 3) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3106_310643


namespace NUMINAMATH_CALUDE_problem_solution_l3106_310685

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x else Real.log x / x

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem problem_solution (a : ℕ → ℝ) : 
  is_geometric_sequence a →
  a 3 * a 4 * a 5 = 1 →
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1 →
  a 1 = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3106_310685


namespace NUMINAMATH_CALUDE_proportion_fourth_term_l3106_310696

theorem proportion_fourth_term (x y : ℝ) : 
  (0.75 : ℝ) / 1.2 = 5 / y → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_proportion_fourth_term_l3106_310696


namespace NUMINAMATH_CALUDE_smallest_five_digit_number_with_conditions_l3106_310676

theorem smallest_five_digit_number_with_conditions : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- 5-digit number
  (n % 32 = 0) ∧              -- divisible by 32
  (n % 45 = 0) ∧              -- divisible by 45
  (n % 54 = 0) ∧              -- divisible by 54
  (30 % n = 0) ∧              -- factor of 30
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ (m % 32 = 0) ∧ (m % 45 = 0) ∧ (m % 54 = 0) ∧ (30 % m = 0) → n ≤ m) ∧
  n = 12960 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_number_with_conditions_l3106_310676


namespace NUMINAMATH_CALUDE_explicit_formula_l3106_310675

noncomputable section

variable (f : ℝ → ℝ)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = (deriv f 1) * Real.exp (x - 1) - (f 0) * x + (1/2) * x^2

theorem explicit_formula (h : satisfies_condition f) :
  ∀ x, f x = Real.exp x - x + (1/2) * x^2 := by
  sorry

end

end NUMINAMATH_CALUDE_explicit_formula_l3106_310675


namespace NUMINAMATH_CALUDE_carlton_zoo_total_l3106_310600

/-- Represents the number of animals in each zoo -/
structure ZooAnimals :=
  (rhinoceroses : ℕ)
  (elephants : ℕ)
  (lions : ℕ)
  (monkeys : ℕ)
  (penguins : ℕ)

/-- Defines the relationship between Bell Zoo and Carlton Zoo -/
def zoo_relationship (bell : ZooAnimals) (carlton : ZooAnimals) : Prop :=
  bell.rhinoceroses = carlton.lions ∧
  bell.elephants = carlton.lions + 3 ∧
  bell.elephants = carlton.rhinoceroses ∧
  carlton.elephants = carlton.rhinoceroses + 2 ∧
  carlton.monkeys = 2 * (carlton.rhinoceroses + carlton.elephants + carlton.lions) ∧
  carlton.penguins = carlton.monkeys + 2 ∧
  bell.monkeys = 2 * carlton.penguins / 3 ∧
  bell.penguins = bell.monkeys + 2 ∧
  bell.lions * 2 = bell.penguins ∧
  bell.rhinoceroses + bell.elephants + bell.lions + bell.monkeys + bell.penguins = 48

theorem carlton_zoo_total (bell : ZooAnimals) (carlton : ZooAnimals) 
  (h : zoo_relationship bell carlton) : 
  carlton.rhinoceroses + carlton.elephants + carlton.lions + carlton.monkeys + carlton.penguins = 57 :=
by sorry


end NUMINAMATH_CALUDE_carlton_zoo_total_l3106_310600


namespace NUMINAMATH_CALUDE_employee_hire_year_l3106_310677

/-- Represents the rule of 70 provision for retirement eligibility -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- Calculates the year an employee was hired given their retirement eligibility year and years of employment -/
def hire_year (retirement_eligibility_year : ℕ) (years_employed : ℕ) : ℕ :=
  retirement_eligibility_year - years_employed

theorem employee_hire_year :
  ∀ (retirement_eligibility_year : ℕ) (hire_age : ℕ),
    hire_age = 32 →
    retirement_eligibility_year = 2009 →
    (∃ (years_employed : ℕ), rule_of_70 (hire_age + years_employed) years_employed) →
    hire_year retirement_eligibility_year (retirement_eligibility_year - (hire_age + 32)) = 1971 :=
by sorry

end NUMINAMATH_CALUDE_employee_hire_year_l3106_310677


namespace NUMINAMATH_CALUDE_non_intersecting_to_concentric_l3106_310631

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- An inversion transformation --/
structure Inversion where
  center : ℝ × ℝ
  power : ℝ
  power_pos : power > 0

/-- Two circles are non-intersecting --/
def non_intersecting (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 > (c1.radius + c2.radius)^2

/-- Two circles are concentric --/
def concentric (c1 c2 : Circle) : Prop :=
  c1.center = c2.center

/-- The image of a circle under inversion --/
def inversion_image (i : Inversion) (c : Circle) : Circle :=
  sorry

/-- The main theorem --/
theorem non_intersecting_to_concentric :
  ∀ (S1 S2 : Circle), non_intersecting S1 S2 →
  ∃ (i : Inversion), concentric (inversion_image i S1) (inversion_image i S2) :=
sorry

end NUMINAMATH_CALUDE_non_intersecting_to_concentric_l3106_310631


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l3106_310614

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x + 2*y, x - 2*y)

theorem preimage_of_3_1 (x y : ℝ) :
  f (x, y) = (3, 1) → (x, y) = (2, 1/2) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l3106_310614


namespace NUMINAMATH_CALUDE_smallest_m_satisfying_conditions_l3106_310628

theorem smallest_m_satisfying_conditions : ∃ m : ℕ,
  (100 ≤ m ∧ m < 1000) ∧  -- m is a three-digit number
  (∃ k : ℤ, m + 7 = 9 * k) ∧  -- m + 7 is divisible by 9
  (∃ l : ℤ, m - 9 = 7 * l) ∧  -- m - 9 is divisible by 7
  (∀ n : ℕ, (100 ≤ n ∧ n < 1000 ∧
    (∃ p : ℤ, n + 7 = 9 * p) ∧
    (∃ q : ℤ, n - 9 = 7 * q)) → m ≤ n) ∧
  m = 128 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_satisfying_conditions_l3106_310628


namespace NUMINAMATH_CALUDE_stirling_duality_l3106_310669

/-- Stirling number of the second kind -/
def stirling2 (N n : ℕ) : ℕ := sorry

/-- Stirling number of the first kind -/
def stirling1 (n M : ℕ) : ℤ := sorry

/-- Kronecker delta -/
def kroneckerDelta (N M : ℕ) : ℕ :=
  if N = M then 1 else 0

/-- The duality property of Stirling numbers -/
theorem stirling_duality (N M : ℕ) :
  (∑' n, (stirling2 N n : ℤ) * stirling1 n M) = kroneckerDelta N M := by
  sorry

end NUMINAMATH_CALUDE_stirling_duality_l3106_310669


namespace NUMINAMATH_CALUDE_common_factor_l3106_310664

def expression (m n : ℕ) : ℤ := 4 * m^3 * n - 9 * m * n^3

theorem common_factor (m n : ℕ) : 
  ∃ (k : ℤ), expression m n = m * n * k ∧ 
  ¬∃ (l : ℤ), l ≠ 1 ∧ l ≠ -1 ∧ 
  ∃ (p : ℤ), expression m n = (m * n * l) * p :=
sorry

end NUMINAMATH_CALUDE_common_factor_l3106_310664


namespace NUMINAMATH_CALUDE_initial_distance_is_one_mile_l3106_310626

/-- Two boats moving towards each other -/
structure BoatSystem where
  boat1_speed : ℝ
  boat2_speed : ℝ
  distance_before_collision : ℝ
  time_before_collision : ℝ

/-- The initial distance between the boats -/
def initial_distance (bs : BoatSystem) : ℝ :=
  bs.distance_before_collision + (bs.boat1_speed + bs.boat2_speed) * bs.time_before_collision

/-- Theorem stating the initial distance between the boats -/
theorem initial_distance_is_one_mile :
  ∀ (bs : BoatSystem),
    bs.boat1_speed = 5 ∧
    bs.boat2_speed = 25 ∧
    bs.distance_before_collision = 0.5 ∧
    bs.time_before_collision = 1 / 60 →
    initial_distance bs = 1 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_is_one_mile_l3106_310626


namespace NUMINAMATH_CALUDE_range_of_m_l3106_310644

def p (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ a = 16 - m ∧ b = m - 4 ∧ a > 0 ∧ b > 0

def q (m : ℝ) : Prop :=
  (m - 10)^2 + 3^2 < 13

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  (4 < m ∧ m ≤ 8) ∨ (10 ≤ m ∧ m < 12) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3106_310644


namespace NUMINAMATH_CALUDE_maxwell_age_proof_l3106_310650

/-- Maxwell's current age --/
def maxwell_age : ℕ := 6

/-- Maxwell's sister's current age --/
def sister_age : ℕ := 2

/-- Years into the future when the age relationship holds --/
def years_future : ℕ := 2

theorem maxwell_age_proof :
  maxwell_age = 6 ∧
  sister_age = 2 ∧
  maxwell_age + years_future = 2 * (sister_age + years_future) :=
by sorry

end NUMINAMATH_CALUDE_maxwell_age_proof_l3106_310650


namespace NUMINAMATH_CALUDE_prime_divisibility_condition_l3106_310641

theorem prime_divisibility_condition (p : ℕ) (x : ℕ) :
  Prime p →
  1 ≤ x ∧ x ≤ 2 * p →
  (x^(p-1) ∣ (p-1)^x + 1) ↔ 
  ((p = 2 ∧ x = 2) ∨ (p = 3 ∧ x = 3) ∨ (x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_prime_divisibility_condition_l3106_310641


namespace NUMINAMATH_CALUDE_arrangements_eq_combinations_l3106_310658

/-- The number of ways to arrange nine 1s and four 0s in a row, where no two 0s are adjacent -/
def arrangements : ℕ := sorry

/-- The number of ways to choose 4 items from 10 items -/
def combinations : ℕ := Nat.choose 10 4

/-- Theorem stating that the number of arrangements is equal to the number of combinations -/
theorem arrangements_eq_combinations : arrangements = combinations := by sorry

end NUMINAMATH_CALUDE_arrangements_eq_combinations_l3106_310658


namespace NUMINAMATH_CALUDE_beluga_breath_interval_proof_l3106_310672

/-- The average time (in minutes) between a bottle-nosed dolphin's air breaths -/
def dolphin_breath_interval : ℝ := 3

/-- The number of minutes in a 24-hour period -/
def minutes_per_day : ℝ := 24 * 60

/-- The ratio of dolphin breaths to beluga whale breaths in a 24-hour period -/
def breath_ratio : ℝ := 2.5

/-- The average time (in minutes) between a beluga whale's air breaths -/
def beluga_breath_interval : ℝ := 7.5

theorem beluga_breath_interval_proof :
  (minutes_per_day / dolphin_breath_interval) = breath_ratio * (minutes_per_day / beluga_breath_interval) :=
by sorry

end NUMINAMATH_CALUDE_beluga_breath_interval_proof_l3106_310672


namespace NUMINAMATH_CALUDE_line_through_point_l3106_310687

/-- Given a line described by the equation 2 - kx = -4y that contains the point (3, 1),
    prove that k = 2. -/
theorem line_through_point (k : ℝ) : 
  (2 - k * 3 = -4 * 1) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l3106_310687


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3106_310608

theorem smallest_fraction_between (a b c d : ℕ) (h1 : a < b) (h2 : c < d) :
  ∃ (x y : ℕ), 
    (x : ℚ) / y > (a : ℚ) / b ∧ 
    (x : ℚ) / y < (c : ℚ) / d ∧ 
    (∀ (p q : ℕ), (p : ℚ) / q > (a : ℚ) / b ∧ (p : ℚ) / q < (c : ℚ) / d → y ≤ q) ∧
    x = 2 ∧ y = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3106_310608


namespace NUMINAMATH_CALUDE_equation_solution_l3106_310680

theorem equation_solution (x : ℝ) : 
  (2*x - 3) / (x + 4) = (3*x + 1) / (2*x - 5) ↔ 
  x = (29 + Real.sqrt 797) / 2 ∨ x = (29 - Real.sqrt 797) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3106_310680
