import Mathlib

namespace debby_total_texts_l334_33400

def texts_before_noon : ℕ := 21
def initial_texts_after_noon : ℕ := 2
def hours_after_noon : ℕ := 12

def texts_after_noon (n : ℕ) : ℕ := initial_texts_after_noon * 2^n

def total_texts : ℕ := texts_before_noon + (Finset.sum (Finset.range hours_after_noon) texts_after_noon)

theorem debby_total_texts : total_texts = 8211 := by sorry

end debby_total_texts_l334_33400


namespace cylinder_radius_calculation_l334_33497

/-- Regular prism with a cylinder -/
structure PrismWithCylinder where
  -- Base side length of the prism
  base_side : ℝ
  -- Lateral edge length of the prism
  lateral_edge : ℝ
  -- Distance between cylinder axis and line AB₁
  axis_distance : ℝ
  -- Radius of the cylinder
  cylinder_radius : ℝ

/-- Theorem stating the radius of the cylinder given the prism dimensions -/
theorem cylinder_radius_calculation (p : PrismWithCylinder) 
  (h1 : p.base_side = 1)
  (h2 : p.lateral_edge = 1 / Real.sqrt 3)
  (h3 : p.axis_distance = 1 / 4) :
  p.cylinder_radius = Real.sqrt 7 / 4 := by
  sorry

end cylinder_radius_calculation_l334_33497


namespace hyperbola_equation_l334_33419

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (m n : ℝ) (h : m < 0 ∧ n > 0) :
  (∀ x y : ℝ, x^2 / m + y^2 / n = 1) →  -- Hyperbola equation
  (2 * Real.sqrt 3 / 3 : ℝ) = 2 / Real.sqrt n →  -- Eccentricity condition
  (∃ c : ℝ, c = 2 ∧ ∀ x y : ℝ, x^2 = 8*y → y = c/2) →  -- Shared focus with parabola
  (∀ x y : ℝ, y^2 / 3 - x^2 = 1) :=
by sorry

end hyperbola_equation_l334_33419


namespace greatest_integer_with_gcd_four_exists_148_with_gcd_four_less_than_150_max_integer_with_gcd_four_l334_33403

theorem greatest_integer_with_gcd_four (n : ℕ) : n < 150 ∧ Nat.gcd n 12 = 4 → n ≤ 148 :=
by sorry

theorem exists_148_with_gcd_four : Nat.gcd 148 12 = 4 :=
by sorry

theorem less_than_150 : 148 < 150 :=
by sorry

theorem max_integer_with_gcd_four :
  ∀ m : ℕ, m < 150 ∧ Nat.gcd m 12 = 4 → m ≤ 148 :=
by sorry

end greatest_integer_with_gcd_four_exists_148_with_gcd_four_less_than_150_max_integer_with_gcd_four_l334_33403


namespace a_less_than_neg_one_sufficient_not_necessary_l334_33471

theorem a_less_than_neg_one_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x < -1 → x + 1/x < -2) ∧
  (∃ y : ℝ, y ≥ -1 ∧ y + 1/y < -2) :=
sorry

end a_less_than_neg_one_sufficient_not_necessary_l334_33471


namespace function_inequality_implies_range_l334_33450

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + Real.log (x + Real.sqrt (x^2 + 1))

theorem function_inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 2 → f (x^2 + 2) + f (-2*a*x) ≥ 0) →
  a ≤ 3/2 :=
by sorry

end function_inequality_implies_range_l334_33450


namespace compute_expression_l334_33440

theorem compute_expression : 6^2 - 4*5 + 4^2 = 32 := by
  sorry

end compute_expression_l334_33440


namespace solution_set_abs_inequality_l334_33486

theorem solution_set_abs_inequality :
  {x : ℝ | |1 - 2*x| < 3} = Set.Ioo (-1) 2 := by
  sorry

end solution_set_abs_inequality_l334_33486


namespace lottery_probability_l334_33492

/-- The number of people participating in the lottery drawing event -/
def num_people : ℕ := 5

/-- The total number of tickets in the box -/
def total_tickets : ℕ := 5

/-- The number of winning tickets -/
def winning_tickets : ℕ := 3

/-- The probability of drawing exactly 2 winning tickets in the first 3 draws
    and the last winning ticket on the 4th draw -/
def event_probability : ℚ := 3 / 10

/-- Theorem stating that the probability of the event ending exactly after
    the 4th person has drawn is 3/10 -/
theorem lottery_probability :
  (num_people = 5) →
  (total_tickets = 5) →
  (winning_tickets = 3) →
  (event_probability = 3 / 10) :=
by
  sorry

end lottery_probability_l334_33492


namespace minimize_reciprocal_sum_l334_33489

theorem minimize_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 30) :
  (1 / a + 1 / b) ≥ 1 / 5 + 1 / 20 ∧
  (1 / a + 1 / b = 1 / 5 + 1 / 20 ↔ a = 5 ∧ b = 20) :=
by sorry

end minimize_reciprocal_sum_l334_33489


namespace union_of_A_and_B_l334_33418

-- Define set A
def A : Set Int := {x | (x + 2) * (x - 1) < 0}

-- Define set B
def B : Set Int := {-2, -1}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {-2, -1, 0} := by sorry

end union_of_A_and_B_l334_33418


namespace adolfo_tower_blocks_l334_33439

-- Define the variables
def initial_blocks : ℕ := sorry
def added_blocks : ℝ := 65.0
def total_blocks : ℕ := 100

-- State the theorem
theorem adolfo_tower_blocks : initial_blocks = 35 := by
  sorry

end adolfo_tower_blocks_l334_33439


namespace border_area_l334_33412

/-- Given a rectangular photograph with a frame, calculate the area of the border. -/
theorem border_area (photo_height photo_width border_width : ℝ) 
  (h1 : photo_height = 8)
  (h2 : photo_width = 10)
  (h3 : border_width = 2) : 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 88 := by
  sorry

#check border_area

end border_area_l334_33412


namespace smallest_fraction_between_l334_33422

theorem smallest_fraction_between (r s : ℕ+) : 
  (7 : ℚ)/11 < r/s ∧ r/s < (5 : ℚ)/8 ∧ 
  (∀ r' s' : ℕ+, (7 : ℚ)/11 < r'/s' ∧ r'/s' < (5 : ℚ)/8 → s ≤ s') →
  s - r = 10 := by
sorry

end smallest_fraction_between_l334_33422


namespace divisible_by_six_percentage_l334_33461

theorem divisible_by_six_percentage (n : ℕ) (h : n = 150) : 
  (Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))).card / n = 1 / 6 := by
  sorry

end divisible_by_six_percentage_l334_33461


namespace unfair_die_expected_value_is_nine_eighths_l334_33415

def unfair_die_expected_value (p1 p2 p3 p4 p5 : ℚ) : ℚ :=
  let p6 := 1 - (p1 + p2 + p3 + p4 + p5)
  1 * p1 + 2 * p2 + 3 * p3 + 4 * p4 + 5 * p5 + 6 * p6

theorem unfair_die_expected_value_is_nine_eighths :
  unfair_die_expected_value (1/6) (1/8) (1/12) (1/12) (1/12) = 9/8 := by
  sorry

#eval unfair_die_expected_value (1/6) (1/8) (1/12) (1/12) (1/12)

end unfair_die_expected_value_is_nine_eighths_l334_33415


namespace students_not_taking_test_l334_33420

theorem students_not_taking_test
  (total_students : ℕ)
  (correct_q1 : ℕ)
  (correct_q2 : ℕ)
  (h1 : total_students = 25)
  (h2 : correct_q1 = 22)
  (h3 : correct_q2 = 20)
  : total_students - max correct_q1 correct_q2 = 3 := by
  sorry

end students_not_taking_test_l334_33420


namespace three_fifths_of_ten_x_minus_three_l334_33423

theorem three_fifths_of_ten_x_minus_three (x : ℝ) : 
  (3 / 5) * (10 * x - 3) = 6 * x - 9 / 5 := by
  sorry

end three_fifths_of_ten_x_minus_three_l334_33423


namespace intersection_A_B_l334_33472

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x * (x - 2) < 0}

def B : Set ℝ := {x | x - 1 > 0}

theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end intersection_A_B_l334_33472


namespace proportional_sum_equation_l334_33494

theorem proportional_sum_equation (x y z a : ℝ) : 
  (∃ (k : ℝ), x = 2*k ∧ y = 3*k ∧ z = 5*k) →  -- x, y, z are proportional to 2, 3, 5
  x + y + z = 100 →                           -- sum is 100
  y = a*x - 10 →                              -- equation for y
  a = 2 :=                                    -- conclusion: a = 2
by
  sorry

end proportional_sum_equation_l334_33494


namespace gcd_15893_35542_l334_33444

theorem gcd_15893_35542 : Nat.gcd 15893 35542 = 1 := by
  sorry

end gcd_15893_35542_l334_33444


namespace line_OF_equation_l334_33410

/-- Given a triangle ABC with vertices A(0,a), B(b,0), C(c,0), and a point P(0,p) on line segment AO
    (not an endpoint), where a, b, c, and p are non-zero real numbers, prove that the equation of
    line OF is (1/c - 1/b)x + (1/p - 1/a)y = 0, where F is the intersection of lines CP and AB. -/
theorem line_OF_equation (a b c p : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hp : p ≠ 0)
    (hp_between : 0 < p ∧ p < a) : 
    ∃ (x y : ℝ), (1 / c - 1 / b) * x + (1 / p - 1 / a) * y = 0 ↔ 
    (∃ (t : ℝ), x = t * c ∧ y = t * p) ∧ (∃ (s : ℝ), x = s * b ∧ y = s * a) := by
  sorry

end line_OF_equation_l334_33410


namespace first_video_length_l334_33426

/-- Given information about Kimiko's YouTube watching --/
structure YoutubeWatching where
  total_time : ℕ
  second_video_length : ℕ
  last_video_length : ℕ

/-- The theorem stating the length of the first video --/
theorem first_video_length (info : YoutubeWatching)
  (h1 : info.total_time = 510)
  (h2 : info.second_video_length = 270)
  (h3 : info.last_video_length = 60) :
  510 - info.second_video_length - 2 * info.last_video_length = 120 := by
  sorry

#check first_video_length

end first_video_length_l334_33426


namespace new_person_weight_l334_33488

theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) :
  n = 7 ∧ avg_increase = 3.5 ∧ old_weight = 75 →
  (n : ℝ) * avg_increase + old_weight = 99.5 :=
by sorry

end new_person_weight_l334_33488


namespace line_x_intercept_l334_33474

theorem line_x_intercept (t : ℝ) (h : t ∈ Set.Icc 0 (2 * Real.pi)) :
  let x := 2 * Real.cos t + 3
  let y := -1 + 5 * Real.sin t
  y = 0 → Real.sin t = 1/5 ∧ x = 2 * Real.cos (Real.arcsin (1/5)) + 3 := by
  sorry

end line_x_intercept_l334_33474


namespace probability_six_distinct_numbers_l334_33438

theorem probability_six_distinct_numbers (n : ℕ) (h : n = 6) :
  (Nat.factorial n : ℚ) / (n ^ n : ℚ) = 5 / 324 := by
  sorry

end probability_six_distinct_numbers_l334_33438


namespace triangle_area_proof_l334_33441

theorem triangle_area_proof (A B C : Real) (a b c : Real) (f : Real → Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  b^2 + c^2 - a^2 = b*c →
  -- a = 2
  a = 2 →
  -- Definition of function f
  (∀ x, f x = Real.sqrt 3 * Real.sin (x/2) * Real.cos (x/2) + Real.cos (x/2)^2) →
  -- f reaches maximum at B
  (∀ x, f x ≤ f B) →
  -- Conclusion: area of triangle is √3
  (1/2) * a^2 * Real.sin A = Real.sqrt 3 :=
by sorry

end triangle_area_proof_l334_33441


namespace blueberry_count_l334_33428

theorem blueberry_count (total : ℕ) (raspberries : ℕ) (blackberries : ℕ) (blueberries : ℕ)
  (h1 : total = 42)
  (h2 : raspberries = total / 2)
  (h3 : blackberries = total / 3)
  (h4 : total = raspberries + blackberries + blueberries) :
  blueberries = 7 := by
  sorry

end blueberry_count_l334_33428


namespace squirrel_acorns_l334_33459

theorem squirrel_acorns (total_acorns : ℕ) (num_months : ℕ) (acorns_per_month : ℕ) :
  total_acorns = 210 →
  num_months = 3 →
  acorns_per_month = 60 →
  total_acorns - num_months * acorns_per_month = 30 :=
by sorry

end squirrel_acorns_l334_33459


namespace coefficient_x_squared_in_expansion_l334_33402

theorem coefficient_x_squared_in_expansion : 
  let expansion := (X - 2 / X) ^ 4
  ∃ a b c d e : ℤ, 
    expansion = a * X^4 + b * X^3 + c * X^2 + d * X + e * X^0 ∧ 
    c = -8
  := by sorry

end coefficient_x_squared_in_expansion_l334_33402


namespace fixed_point_sum_l334_33457

theorem fixed_point_sum (a : ℝ) (m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : n = a * (m - 1) + 2) : m + n = 4 := by
  sorry

end fixed_point_sum_l334_33457


namespace loss_percentage_calculation_l334_33498

theorem loss_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 1500 → 
  selling_price = 1335 → 
  (cost_price - selling_price) / cost_price * 100 = 11 := by
sorry

end loss_percentage_calculation_l334_33498


namespace max_attached_squares_l334_33404

/-- Represents a square in 2D space -/
structure Square :=
  (side_length : ℝ)
  (center : ℝ × ℝ)

/-- Checks if two squares are touching but not overlapping -/
def are_touching (s1 s2 : Square) : Prop :=
  sorry

/-- Checks if a square is touching the perimeter of another square -/
def is_touching_perimeter (s1 s2 : Square) : Prop :=
  sorry

/-- The configuration of squares attached to a given square -/
structure SquareConfiguration :=
  (given_square : Square)
  (attached_squares : List Square)

/-- Checks if a configuration is valid according to the problem conditions -/
def is_valid_configuration (config : SquareConfiguration) : Prop :=
  ∀ s ∈ config.attached_squares,
    is_touching_perimeter s config.given_square ∧
    ∀ t ∈ config.attached_squares, s ≠ t → ¬(are_touching s t)

/-- The main theorem: maximum number of attached squares is 8 -/
theorem max_attached_squares (config : SquareConfiguration) :
  is_valid_configuration config →
  config.attached_squares.length ≤ 8 :=
sorry

end max_attached_squares_l334_33404


namespace max_additional_plates_l334_33453

def first_set : Finset Char := {'B', 'F', 'J', 'M', 'S'}
def second_set : Finset Char := {'E', 'U', 'Y'}
def third_set : Finset Char := {'G', 'K', 'R', 'Z'}

theorem max_additional_plates :
  ∃ (new_first : Char) (new_third : Char),
    new_first ∉ first_set ∧
    new_third ∉ third_set ∧
    (first_set.card + 1) * second_set.card * (third_set.card + 1) -
    first_set.card * second_set.card * third_set.card = 30 ∧
    ∀ (a : Char) (c : Char),
      a ∉ first_set →
      c ∉ third_set →
      (first_set.card + 1) * second_set.card * (third_set.card + 1) -
      first_set.card * second_set.card * third_set.card ≤ 30 :=
by sorry

end max_additional_plates_l334_33453


namespace unique_sums_count_l334_33460

/-- Represents the set of available coins -/
def CoinSet : Finset ℕ := {1, 2, 5, 100, 100, 100, 100, 500, 500}

/-- Generates all possible sums using the given coin set -/
def PossibleSums (coins : Finset ℕ) : Finset ℕ :=
  sorry

/-- The number of unique sums that can be formed using the given coin set -/
theorem unique_sums_count : (PossibleSums CoinSet).card = 119 := by
  sorry

end unique_sums_count_l334_33460


namespace max_x_plus_y_max_x_plus_y_achieved_l334_33467

theorem max_x_plus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x - y) :
  x + y ≤ 1 / Real.sqrt 2 :=
by sorry

theorem max_x_plus_y_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, 3 * (x^2 + y^2) = x - y ∧ x + y > 1 / Real.sqrt 2 - ε :=
by sorry

end max_x_plus_y_max_x_plus_y_achieved_l334_33467


namespace sum_has_five_digits_l334_33483

/-- A nonzero digit is a natural number between 1 and 9, inclusive. -/
def NonzeroDigit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The function that constructs the second number (A75) from a nonzero digit A. -/
def secondNumber (A : NonzeroDigit) : ℕ := A.val * 100 + 75

/-- The function that constructs the third number (5B2) from a nonzero digit B. -/
def thirdNumber (B : NonzeroDigit) : ℕ := 500 + B.val * 10 + 2

/-- The theorem stating that the sum of the three numbers always has 5 digits. -/
theorem sum_has_five_digits (A B : NonzeroDigit) :
  ∃ n : ℕ, 10000 ≤ 9643 + secondNumber A + thirdNumber B ∧
           9643 + secondNumber A + thirdNumber B < 100000 := by
  sorry

end sum_has_five_digits_l334_33483


namespace sin_theta_plus_7pi_6_l334_33433

theorem sin_theta_plus_7pi_6 (θ : ℝ) 
  (h : Real.cos (θ - π/6) + Real.sin θ = 4 * Real.sqrt 3 / 5) : 
  Real.sin (θ + 7*π/6) = -4/5 := by
  sorry

end sin_theta_plus_7pi_6_l334_33433


namespace B_power_97_l334_33411

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_97 : B^97 = B := by sorry

end B_power_97_l334_33411


namespace second_point_y_coordinate_l334_33430

/-- Given two points on a line, prove the y-coordinate of the second point -/
theorem second_point_y_coordinate
  (m n k : ℝ)
  (h1 : m = 2 * n + 3)  -- First point (m, n) satisfies line equation
  (h2 : m + 2 = 2 * (n + k) + 3)  -- Second point (m + 2, n + k) satisfies line equation
  (h3 : k = 1)  -- Given condition
  : n + k = n + 1 := by
  sorry

end second_point_y_coordinate_l334_33430


namespace joohee_ate_17_chocolates_l334_33414

-- Define the total number of chocolates
def total_chocolates : ℕ := 25

-- Define the relationship between Joo-hee's and Jun-seong's chocolates
def joohee_chocolates (junseong_chocolates : ℕ) : ℕ :=
  2 * junseong_chocolates + 1

-- Theorem statement
theorem joohee_ate_17_chocolates :
  ∃ (junseong_chocolates : ℕ),
    junseong_chocolates + joohee_chocolates junseong_chocolates = total_chocolates ∧
    joohee_chocolates junseong_chocolates = 17 :=
  sorry

end joohee_ate_17_chocolates_l334_33414


namespace employed_females_percentage_l334_33465

theorem employed_females_percentage (total_population : ℝ) 
  (h1 : total_population > 0) 
  (employed_percentage : ℝ) 
  (h2 : employed_percentage = 60) 
  (employed_males_percentage : ℝ) 
  (h3 : employed_males_percentage = 45) : 
  (employed_percentage - employed_males_percentage) / employed_percentage * 100 = 25 := by
sorry

end employed_females_percentage_l334_33465


namespace sum_of_squares_l334_33473

theorem sum_of_squares (a b c : ℝ) : 
  (a * b + b * c + a * c = 50) → 
  (a + b + c = 16) → 
  (a^2 + b^2 + c^2 = 156) := by
sorry

end sum_of_squares_l334_33473


namespace sixth_term_of_geometric_sequence_l334_33490

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sixth_term_of_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4)^2 - 8*(a 4) + 9 = 0 →
  (a 8)^2 - 8*(a 8) + 9 = 0 →
  a 6 = 3 := by
  sorry

end sixth_term_of_geometric_sequence_l334_33490


namespace palace_rotation_l334_33462

theorem palace_rotation (x : ℕ) : 
  (x % 30 = 15 ∧ x % 50 = 25 ∧ x % 70 = 35) → x ≥ 525 :=
by sorry

end palace_rotation_l334_33462


namespace ellipse_ratio_squared_l334_33475

/-- For an ellipse with semi-major axis a, semi-minor axis b, and distance from center to focus c,
    if b/a = a/c and c^2 = a^2 - b^2, then (b/a)^2 = 1/2 -/
theorem ellipse_ratio_squared (a b c : ℝ) (h1 : b / a = a / c) (h2 : c^2 = a^2 - b^2) :
  (b / a)^2 = 1 / 2 := by
  sorry

end ellipse_ratio_squared_l334_33475


namespace interest_rate_calculation_l334_33466

/-- Proves that given the conditions of simple and compound interest, the interest rate is 18.50% -/
theorem interest_rate_calculation (P : ℝ) (R : ℝ) : 
  P * R * 2 / 100 = 55 →
  P * ((1 + R / 100)^2 - 1) = 56.375 →
  R = 18.50 := by
sorry

end interest_rate_calculation_l334_33466


namespace product_divisible_by_3_probability_l334_33434

/-- A standard die has 6 sides -/
def standard_die : ℕ := 6

/-- The number of rolls -/
def num_rolls : ℕ := 8

/-- The probability of rolling a number divisible by 3 on a standard die -/
def prob_divisible_by_3 : ℚ := 1 / 3

/-- The probability of rolling a number not divisible by 3 on a standard die -/
def prob_not_divisible_by_3 : ℚ := 2 / 3

/-- The probability that the product of all rolls is divisible by 3 -/
def prob_product_divisible_by_3 : ℚ := 6305 / 6561

theorem product_divisible_by_3_probability :
  prob_product_divisible_by_3 = 1 - (prob_not_divisible_by_3 ^ num_rolls) :=
sorry

end product_divisible_by_3_probability_l334_33434


namespace solve_for_a_l334_33446

theorem solve_for_a (a x : ℝ) : 
  (3/10) * a + (2*x + 4)/2 = 4*(x - 1) ∧ x = 3 → a = 10 := by
  sorry

end solve_for_a_l334_33446


namespace discount_percentage_l334_33484

theorem discount_percentage (M : ℝ) (C : ℝ) (S : ℝ) 
  (h1 : C = 0.64 * M) 
  (h2 : S = C * 1.28125) : 
  (M - S) / M * 100 = 18.08 := by
  sorry

end discount_percentage_l334_33484


namespace total_pencils_l334_33456

/-- Given that each child has 2 pencils and there are 9 children, prove that the total number of pencils is 18. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) (h1 : pencils_per_child = 2) (h2 : num_children = 9) :
  pencils_per_child * num_children = 18 := by
  sorry

end total_pencils_l334_33456


namespace sum_of_roots_l334_33406

theorem sum_of_roots (α β : ℝ) 
  (h1 : α^3 - 3*α^2 + 5*α - 17 = 0)
  (h2 : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := by sorry

end sum_of_roots_l334_33406


namespace carters_increased_baking_l334_33425

def usual_cheesecakes : ℕ := 6
def usual_muffins : ℕ := 5
def usual_redvelvet : ℕ := 8
def tripling_factor : ℕ := 3

theorem carters_increased_baking :
  (usual_cheesecakes + usual_muffins + usual_redvelvet) * tripling_factor -
  (usual_cheesecakes + usual_muffins + usual_redvelvet) = 38 :=
by sorry

end carters_increased_baking_l334_33425


namespace track_circumference_l334_33447

/-- The circumference of a circular track given specific conditions -/
theorem track_circumference : 
  ∀ (circumference : ℝ) (distance_B_first_meet : ℝ) (distance_A_second_meet : ℝ),
  distance_B_first_meet = 100 →
  distance_A_second_meet = circumference - 60 →
  (circumference / 2 - distance_B_first_meet) / distance_B_first_meet = 
    distance_A_second_meet / (circumference + 60) →
  circumference = 480 := by
sorry

end track_circumference_l334_33447


namespace range_of_alpha_plus_three_beta_l334_33470

theorem range_of_alpha_plus_three_beta 
  (h1 : ∀ α β : ℝ, -1 ≤ α + β ∧ α + β ≤ 1 → 1 ≤ α + 2*β ∧ α + 2*β ≤ 3) :
  ∀ α β : ℝ, (-1 ≤ α + β ∧ α + β ≤ 1) → (1 ≤ α + 2*β ∧ α + 2*β ≤ 3) → 
  (1 ≤ α + 3*β ∧ α + 3*β ≤ 7) := by
sorry

end range_of_alpha_plus_three_beta_l334_33470


namespace intersection_with_complement_l334_33436

open Set

def U : Finset ℕ := {0, 1, 2, 3, 4, 5}
def A : Finset ℕ := {0, 1, 3}
def B : Finset ℕ := {2, 3, 5}

theorem intersection_with_complement :
  A ∩ (U \ B) = {0, 1} := by sorry

end intersection_with_complement_l334_33436


namespace jessica_money_difference_l334_33401

/-- Proves that Jessica has 90 dollars more than Rodney given the stated conditions. -/
theorem jessica_money_difference (jessica_money : ℕ) (lily_money : ℕ) (ian_money : ℕ) (rodney_money : ℕ) :
  jessica_money = 150 ∧
  jessica_money = lily_money + 30 ∧
  lily_money = 3 * ian_money ∧
  ian_money + 20 = rodney_money →
  jessica_money - rodney_money = 90 :=
by sorry

end jessica_money_difference_l334_33401


namespace choose_captains_l334_33442

theorem choose_captains (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  Nat.choose n k = 1365 := by
  sorry

end choose_captains_l334_33442


namespace diana_apollo_dice_probability_l334_33435

def roll_die := Finset.range 6

def favorable_outcomes : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 >= p.2) (roll_die.product roll_die)

theorem diana_apollo_dice_probability :
  (favorable_outcomes.card : ℚ) / (roll_die.card * roll_die.card) = 7 / 12 := by
  sorry

end diana_apollo_dice_probability_l334_33435


namespace three_possible_values_for_sum_l334_33493

theorem three_possible_values_for_sum (x y : ℤ) 
  (h : x^2 + y^2 + 1 ≤ 2*x + 2*y) : 
  ∃ (S : Finset ℤ), (Finset.card S = 3) ∧ ((x + y) ∈ S) :=
sorry

end three_possible_values_for_sum_l334_33493


namespace smallest_n_for_inequality_l334_33477

theorem smallest_n_for_inequality : 
  (∃ n : ℕ, ∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧ 
  (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4)) ∧
  (∀ m : ℕ, m < 3 → ∃ x y z : ℝ, (x^2 + y^2 + z^2)^2 > m * (x^4 + y^4 + z^4)) :=
by sorry

#check smallest_n_for_inequality

end smallest_n_for_inequality_l334_33477


namespace second_quarter_profit_l334_33427

theorem second_quarter_profit 
  (annual_profit : ℕ)
  (first_quarter_profit : ℕ)
  (third_quarter_profit : ℕ)
  (fourth_quarter_profit : ℕ)
  (h1 : annual_profit = 8000)
  (h2 : first_quarter_profit = 1500)
  (h3 : third_quarter_profit = 3000)
  (h4 : fourth_quarter_profit = 2000) :
  annual_profit - (first_quarter_profit + third_quarter_profit + fourth_quarter_profit) = 1500 :=
by
  sorry

end second_quarter_profit_l334_33427


namespace rectangle_area_from_equilateral_triangle_l334_33408

theorem rectangle_area_from_equilateral_triangle (triangle_area : ℝ) : 
  triangle_area = 9 * Real.sqrt 3 →
  ∃ (triangle_side : ℝ), 
    triangle_area = (Real.sqrt 3 / 4) * triangle_side^2 ∧
    ∃ (rect_width rect_length : ℝ),
      rect_width = triangle_side ∧
      rect_length = 3 * rect_width ∧
      rect_width * rect_length = 108 := by
sorry

end rectangle_area_from_equilateral_triangle_l334_33408


namespace range_of_x_l334_33443

theorem range_of_x (x : ℝ) (h1 : 1 / x ≤ 4) (h2 : 1 / x ≥ -2) : x ≥ 1 / 4 ∨ x ≤ -1 / 2 := by
  sorry

end range_of_x_l334_33443


namespace direct_proportion_quadrants_l334_33452

/-- A direct proportion function in a plane rectangular coordinate system -/
structure DirectProportionFunction where
  n : ℝ
  f : ℝ → ℝ
  h : ∀ x, f x = (n - 1) * x

/-- Predicate to check if a point (x, y) is in the first or third quadrant -/
def isInFirstOrThirdQuadrant (x y : ℝ) : Prop :=
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)

/-- Predicate to check if the graph of a function passes through the first and third quadrants -/
def passesFirstAndThirdQuadrants (f : ℝ → ℝ) : Prop :=
  ∀ x, isInFirstOrThirdQuadrant x (f x)

/-- Theorem: If a direct proportion function's graph passes through the first and third quadrants,
    then n > 1 -/
theorem direct_proportion_quadrants (dpf : DirectProportionFunction)
    (h : passesFirstAndThirdQuadrants dpf.f) : dpf.n > 1 := by
  sorry

end direct_proportion_quadrants_l334_33452


namespace inequality_solution_set_l334_33445

theorem inequality_solution_set (x : ℝ) :
  (((x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4) ↔ 
   (x ∈ Set.Ioc 0 (1/2) ∪ Set.Ioo (3/2) 2)) :=
by sorry

end inequality_solution_set_l334_33445


namespace brick_height_is_6cm_l334_33432

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The dimensions of the wall in centimeters -/
def wallDimensions : Dimensions :=
  { length := 800, width := 22.5, height := 600 }

/-- The known dimensions of a brick in centimeters (height is unknown) -/
def brickDimensions (h : ℝ) : Dimensions :=
  { length := 50, width := 11.25, height := h }

/-- The number of bricks needed to build the wall -/
def numberOfBricks : ℕ := 3200

/-- Theorem stating that the height of each brick is 6 cm -/
theorem brick_height_is_6cm :
  ∃ (h : ℝ), h = 6 ∧
    (volume wallDimensions = ↑numberOfBricks * volume (brickDimensions h)) := by
  sorry

end brick_height_is_6cm_l334_33432


namespace fourth_root_closest_to_6700_l334_33429

def n : ℕ := 2001200120012001

def options : List ℕ := [2001, 6700, 21000, 12000, 2100]

theorem fourth_root_closest_to_6700 :
  ∃ (x : ℝ), x^4 = n ∧ 
  ∀ y ∈ options, |x - 6700| ≤ |x - y| :=
sorry

end fourth_root_closest_to_6700_l334_33429


namespace cars_produced_in_europe_l334_33469

def cars_north_america : ℕ := 3884
def total_cars : ℕ := 6755

theorem cars_produced_in_europe : 
  total_cars - cars_north_america = 2871 := by sorry

end cars_produced_in_europe_l334_33469


namespace simplify_and_evaluate_l334_33468

theorem simplify_and_evaluate : 
  ∀ x : ℝ, x ≠ 2 → x ≠ -2 → x ≠ 0 → x = 1 →
  (3 * x / (x - 2) - x / (x + 2)) * (x^2 - 4) / x = 10 := by
  sorry

end simplify_and_evaluate_l334_33468


namespace outbound_speed_calculation_l334_33496

theorem outbound_speed_calculation (distance : ℝ) (return_speed : ℝ) (total_time : ℝ) :
  distance = 19.999999999999996 →
  return_speed = 4 →
  total_time = 5.8 →
  ∃ outbound_speed : ℝ, 
    outbound_speed = 25 ∧
    distance / outbound_speed + distance / return_speed = total_time :=
by sorry

end outbound_speed_calculation_l334_33496


namespace quadratic_function_property_l334_33480

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_property
  (f : ℝ → ℝ)
  (h_quad : is_quadratic f)
  (h_pos : ∃ a b c : ℝ, a > 0 ∧ ∀ x, f x = a * x^2 + b * x + c)
  (h_sym : ∀ x : ℝ, f x = f (4 - x))
  (h_ineq : ∀ a : ℝ, f (2 - a^2) < f (1 + a - a^2)) :
  ∀ a : ℝ, a < 1 :=
sorry

end quadratic_function_property_l334_33480


namespace orange_cost_calculation_l334_33407

theorem orange_cost_calculation (family_size : ℕ) (planned_spending : ℚ) (savings_percentage : ℚ) (oranges_received : ℕ) : 
  family_size = 4 → 
  planned_spending = 15 → 
  savings_percentage = 40 / 100 → 
  oranges_received = family_size →
  (planned_spending * savings_percentage) / oranges_received = 3/2 := by
sorry

end orange_cost_calculation_l334_33407


namespace baby_panda_eats_50_pounds_l334_33431

/-- The amount of bamboo (in pounds) an adult panda eats per day -/
def adult_panda_daily : ℕ := 138

/-- The total amount of bamboo (in pounds) eaten by both adult and baby pandas in a week -/
def total_weekly : ℕ := 1316

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The amount of bamboo (in pounds) a baby panda eats per day -/
def baby_panda_daily : ℕ := (total_weekly - adult_panda_daily * days_per_week) / days_per_week

theorem baby_panda_eats_50_pounds : baby_panda_daily = 50 := by
  sorry

end baby_panda_eats_50_pounds_l334_33431


namespace normal_dist_prob_l334_33495

-- Define a random variable following normal distribution
def normal_dist (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def P (X : normal_dist 1 σ) (event : Set ℝ) : ℝ := sorry

-- State the theorem
theorem normal_dist_prob (σ : ℝ) (ξ : normal_dist 1 σ) 
  (h : P ξ {x | x < 0} = 0.4) : 
  P ξ {x | x < 2} = 0.6 := by sorry

end normal_dist_prob_l334_33495


namespace angle_equality_l334_33416

-- Define the problem statement
theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (5 * π / 180) = Real.cos θ - Real.sin θ) : 
  θ = 40 * π / 180 := by
  sorry

end angle_equality_l334_33416


namespace satisfy_equation_l334_33454

theorem satisfy_equation : ∀ (x y : ℝ), x = 1 ∧ y = 2 → 2 * x + 3 * y = 8 := by
  sorry

end satisfy_equation_l334_33454


namespace max_value_of_g_l334_33463

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^3

-- State the theorem
theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ 2 → g y ≤ g x) ∧
  g x = 16 * Real.sqrt 3 / 9 :=
sorry

end max_value_of_g_l334_33463


namespace largest_integer_negative_quadratic_l334_33499

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 28 < 0 → n ≤ 6 :=
by sorry

end largest_integer_negative_quadratic_l334_33499


namespace video_votes_l334_33481

theorem video_votes (net_score : ℚ) (like_percentage : ℚ) (dislike_percentage : ℚ) :
  net_score = 75 →
  like_percentage = 55 / 100 →
  dislike_percentage = 45 / 100 →
  like_percentage + dislike_percentage = 1 →
  ∃ (total_votes : ℚ),
    total_votes * (like_percentage - dislike_percentage) = net_score ∧
    total_votes = 750 :=
by sorry

end video_votes_l334_33481


namespace negation_equivalence_l334_33424

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 2 ∨ x ≤ -1) ↔ (∀ x : ℝ, -1 < x ∧ x ≤ 2) :=
by sorry

end negation_equivalence_l334_33424


namespace johns_payment_is_1500_l334_33405

/-- Calculates the personal payment for hearing aids given insurance details --/
def calculate_personal_payment (cost_per_aid : ℕ) (num_aids : ℕ) (deductible : ℕ) 
  (coverage_percent : ℚ) (coverage_limit : ℕ) : ℕ :=
  let total_cost := cost_per_aid * num_aids
  let after_deductible := total_cost - deductible
  let insurance_payment := min (coverage_limit) (↑(Nat.floor (coverage_percent * ↑after_deductible)))
  total_cost - insurance_payment

/-- Theorem stating that John's personal payment for hearing aids is $1500 --/
theorem johns_payment_is_1500 : 
  calculate_personal_payment 2500 2 500 (4/5) 3500 = 1500 := by
  sorry

end johns_payment_is_1500_l334_33405


namespace tan_sum_alpha_beta_l334_33409

-- Define the line l
def line_l (x y : ℝ) (α β : ℝ) : Prop :=
  x * Real.tan α - y - 3 * Real.tan β = 0

-- Define the normal vector
def normal_vector : ℝ × ℝ := (2, -1)

-- Theorem statement
theorem tan_sum_alpha_beta (α β : ℝ) :
  line_l 0 1 α β ∧ 
  normal_vector = (2, -1) →
  Real.tan (α + β) = 1 :=
by sorry

end tan_sum_alpha_beta_l334_33409


namespace sums_are_equal_l334_33417

def S₁ : ℕ := 1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999

def S₂ : ℕ := 9 + 98 + 987 + 9876 + 98765 + 987654 + 9876543 + 98765432 + 987654321

theorem sums_are_equal : S₁ = S₂ := by
  sorry

end sums_are_equal_l334_33417


namespace tomato_land_area_l334_33455

/-- Represents the farm land allocation -/
structure FarmLand where
  total : ℝ
  cleared_percentage : ℝ
  barley_percentage : ℝ
  potato_percentage : ℝ

/-- Calculates the area of land planted with tomato -/
def tomato_area (farm : FarmLand) : ℝ :=
  let cleared_land := farm.total * farm.cleared_percentage
  let barley_land := cleared_land * farm.barley_percentage
  let potato_land := cleared_land * farm.potato_percentage
  cleared_land - (barley_land + potato_land)

/-- Theorem stating the area of land planted with tomato -/
theorem tomato_land_area : 
  let farm := FarmLand.mk 1000 0.9 0.8 0.1
  tomato_area farm = 90 := by
  sorry


end tomato_land_area_l334_33455


namespace problem_solution_l334_33448

theorem problem_solution (A B : ℝ) : 
  (A^2 = 0.012345678987654321 * (List.sum (List.range 9) + List.sum (List.reverse (List.range 9)))) →
  (B^2 = 0.012345679012345679) →
  9 * (10^9 : ℝ) * (1 - |A|) * B = 0 := by
  sorry

end problem_solution_l334_33448


namespace max_ratio_of_two_digit_integers_l334_33464

theorem max_ratio_of_two_digit_integers (x y : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →  -- x is a two-digit positive integer
  (10 ≤ y ∧ y ≤ 99) →  -- y is a two-digit positive integer
  x > y →              -- x is greater than y
  (x + y) / 2 = 70 →   -- their mean is 70
  (∀ a b : ℕ, (10 ≤ a ∧ a ≤ 99) → (10 ≤ b ∧ b ≤ 99) → a > b → (a + b) / 2 = 70 → x / y ≥ a / b) →
  x / y = 99 / 41 :=
by sorry

end max_ratio_of_two_digit_integers_l334_33464


namespace andy_gave_five_to_brother_l334_33413

/-- The number of cookies Andy had at the start -/
def initial_cookies : ℕ := 72

/-- The number of cookies Andy ate -/
def andy_ate : ℕ := 3

/-- The number of players in Andy's basketball team -/
def team_size : ℕ := 8

/-- The number of cookies taken by the i-th player -/
def player_cookies (i : ℕ) : ℕ := 2 * i - 1

/-- The sum of cookies taken by all team members -/
def team_total : ℕ := (team_size * (player_cookies 1 + player_cookies team_size)) / 2

/-- The number of cookies Andy gave to his little brother -/
def brother_cookies : ℕ := initial_cookies - andy_ate - team_total

theorem andy_gave_five_to_brother : brother_cookies = 5 := by
  sorry

end andy_gave_five_to_brother_l334_33413


namespace sum_of_roots_l334_33482

theorem sum_of_roots (c d : ℝ) 
  (hc : c^3 - 21*c^2 + 28*c - 70 = 0) 
  (hd : 10*d^3 - 75*d^2 - 350*d + 3225 = 0) : 
  c + d = 21/2 := by
sorry

end sum_of_roots_l334_33482


namespace fruit_shopping_cost_l334_33458

/-- Calculates the price per unit of fruit given the number of fruits and their total price in cents. -/
def price_per_unit (num_fruits : ℕ) (total_price : ℕ) : ℚ :=
  total_price / num_fruits

/-- Determines the cheaper fruit given their prices per unit. -/
def cheaper_fruit (apple_price : ℚ) (orange_price : ℚ) : ℚ :=
  min apple_price orange_price

theorem fruit_shopping_cost :
  let apple_price := price_per_unit 10 200  -- 10 apples for $2 (200 cents)
  let orange_price := price_per_unit 5 150  -- 5 oranges for $1.50 (150 cents)
  let cheaper_price := cheaper_fruit apple_price orange_price
  (12 : ℕ) * (cheaper_price : ℚ) = 240
  := by sorry

end fruit_shopping_cost_l334_33458


namespace brother_money_distribution_l334_33451

theorem brother_money_distribution (older_initial younger_initial difference transfer : ℕ) :
  older_initial = 2800 →
  younger_initial = 1500 →
  difference = 360 →
  transfer = 470 →
  (older_initial - transfer) = (younger_initial + transfer + difference) :=
by
  sorry

end brother_money_distribution_l334_33451


namespace parabola_intersection_l334_33487

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 8 * x + 2
def g (x : ℝ) : ℝ := 6 * x^2 + 4 * x + 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(-4, 82), (0, 2)}

-- Theorem statement
theorem parabola_intersection :
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) ∈ intersection_points :=
by sorry

end parabola_intersection_l334_33487


namespace inequality_range_l334_33478

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 - Real.log x / Real.log a < 0) ↔ a ∈ Set.Ioo (1/16) 1 := by
  sorry

end inequality_range_l334_33478


namespace constant_sum_implies_parallelogram_l334_33437

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Additional condition to ensure convexity

-- Define a function to calculate the distance from a point to a line
def distanceToLine (p : ℝ × ℝ) (l : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ := sorry

-- Define a function to check if a point is inside the quadrilateral
def isInsideQuadrilateral (q : ConvexQuadrilateral) (p : ℝ × ℝ) : Prop := sorry

-- Define a function to calculate the sum of distances from a point to all sides
def sumOfDistances (q : ConvexQuadrilateral) (p : ℝ × ℝ) : ℝ :=
  distanceToLine p (q.vertices 0, q.vertices 1) +
  distanceToLine p (q.vertices 1, q.vertices 2) +
  distanceToLine p (q.vertices 2, q.vertices 3) +
  distanceToLine p (q.vertices 3, q.vertices 0)

-- Define what it means for a quadrilateral to be a parallelogram
def isParallelogram (q : ConvexQuadrilateral) : Prop := sorry

-- The main theorem
theorem constant_sum_implies_parallelogram (q : ConvexQuadrilateral) :
  (∃ k : ℝ, ∀ p : ℝ × ℝ, isInsideQuadrilateral q p → sumOfDistances q p = k) →
  isParallelogram q := by sorry

end constant_sum_implies_parallelogram_l334_33437


namespace min_distances_2019_points_l334_33421

/-- The minimum number of distinct distances between pairs of points in a set of n points in a plane -/
noncomputable def min_distinct_distances (n : ℕ) : ℝ :=
  Real.sqrt (n - 3/4 : ℝ) - 1/2

/-- Theorem: For 2019 distinct points in a plane, the number of distinct distances between pairs of points is at least 44 -/
theorem min_distances_2019_points :
  ⌈min_distinct_distances 2019⌉ ≥ 44 := by sorry

end min_distances_2019_points_l334_33421


namespace elenas_earnings_l334_33449

/-- Calculates the total earnings given an hourly wage and number of hours worked -/
def totalEarnings (hourlyWage : ℚ) (hoursWorked : ℚ) : ℚ :=
  hourlyWage * hoursWorked

/-- Proves that Elena's earnings for 4 hours at $13.25 per hour is $53.00 -/
theorem elenas_earnings :
  totalEarnings (13.25 : ℚ) (4 : ℚ) = (53 : ℚ) := by
  sorry

end elenas_earnings_l334_33449


namespace box_surface_area_l334_33491

/-- Calculates the surface area of the interior of an open box formed by removing square corners from a rectangular sheet and folding up the sides. -/
def interior_surface_area (sheet_length : ℕ) (sheet_width : ℕ) (corner_size : ℕ) : ℕ :=
  let modified_area := sheet_length * sheet_width
  let corner_area := corner_size * corner_size
  let total_removed_area := 4 * corner_area
  modified_area - total_removed_area

/-- Theorem stating that the surface area of the interior of the box is 804 square units. -/
theorem box_surface_area :
  interior_surface_area 25 40 7 = 804 := by
  sorry

#eval interior_surface_area 25 40 7

end box_surface_area_l334_33491


namespace basement_water_pump_time_l334_33476

/-- Calculates the time required to pump water out of a flooded basement. -/
theorem basement_water_pump_time
  (basement_length : ℝ)
  (basement_width : ℝ)
  (water_depth_inches : ℝ)
  (num_pumps : ℕ)
  (pump_rate : ℝ)
  (cubic_foot_to_gallon : ℝ)
  (h1 : basement_length = 30)
  (h2 : basement_width = 40)
  (h3 : water_depth_inches = 24)
  (h4 : num_pumps = 4)
  (h5 : pump_rate = 10)
  (h6 : cubic_foot_to_gallon = 7.5) :
  (basement_length * basement_width * (water_depth_inches / 12) * cubic_foot_to_gallon) /
  (num_pumps * pump_rate) = 450 := by
  sorry

#check basement_water_pump_time

end basement_water_pump_time_l334_33476


namespace distribute_four_balls_three_boxes_l334_33479

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 3 ways to distribute 4 indistinguishable balls into 3 distinguishable boxes,
    with each box containing at least one ball -/
theorem distribute_four_balls_three_boxes :
  distribute_balls 4 3 = 3 := by sorry

end distribute_four_balls_three_boxes_l334_33479


namespace lune_area_l334_33485

/-- The area of a lune formed by two overlapping semicircles -/
theorem lune_area (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 4) : 
  (π * r₂^2 / 2) - (π * r₁^2 / 2) = 3.5 * π := by sorry

end lune_area_l334_33485
