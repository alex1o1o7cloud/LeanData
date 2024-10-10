import Mathlib

namespace unfair_die_expected_value_l1226_122647

/-- An unfair eight-sided die with given probabilities -/
structure UnfairDie where
  /-- The probability of rolling an 8 -/
  prob_eight : ℝ
  /-- The probability of rolling any number from 1 to 7 -/
  prob_others : ℝ
  /-- The probability of rolling an 8 is 3/8 -/
  h1 : prob_eight = 3/8
  /-- The probability of rolling any number from 1 to 7 is 5/56 -/
  h2 : prob_others = 5/56
  /-- The sum of all probabilities is 1 -/
  h3 : prob_eight + 7 * prob_others = 1

/-- The expected value of rolling the unfair die -/
def expected_value (d : UnfairDie) : ℝ :=
  d.prob_others * (1 + 2 + 3 + 4 + 5 + 6 + 7) + d.prob_eight * 8

/-- Theorem stating that the expected value of rolling the unfair die is 5.5 -/
theorem unfair_die_expected_value (d : UnfairDie) :
  expected_value d = 5.5 := by
  sorry

end unfair_die_expected_value_l1226_122647


namespace problem_solution_l1226_122611

theorem problem_solution (x : ℝ) : 
  3.5 * ((3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5)) = 2800.0000000000005 → x = 0.3 := by
  sorry

end problem_solution_l1226_122611


namespace inequality_implication_l1226_122694

theorem inequality_implication (m n : ℝ) : m > 0 → n > m → 1/m - 1/n > 0 := by
  sorry

end inequality_implication_l1226_122694


namespace sqrt_36_times_sqrt_16_l1226_122635

theorem sqrt_36_times_sqrt_16 : Real.sqrt (36 * Real.sqrt 16) = 12 := by
  sorry

end sqrt_36_times_sqrt_16_l1226_122635


namespace sector_area_proof_l1226_122667

-- Define the given conditions
def circle_arc_length : ℝ := 2
def central_angle : ℝ := 2

-- Define the theorem
theorem sector_area_proof :
  let radius := circle_arc_length / central_angle
  let sector_area := (1 / 2) * radius^2 * central_angle
  sector_area = 1 := by sorry

end sector_area_proof_l1226_122667


namespace base8_digit_product_l1226_122670

/-- Convert a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculate the product of a list of natural numbers -/
def product (l : List ℕ) : ℕ :=
  sorry

/-- The product of the digits in the base 8 representation of 8927 is 126 -/
theorem base8_digit_product : product (toBase8 8927) = 126 :=
  sorry

end base8_digit_product_l1226_122670


namespace ellipse_standard_equation_l1226_122651

def ellipse_equation (e : ℝ) (l : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) => x^2 / 32 + y^2 / 16 = 1

theorem ellipse_standard_equation (e l : ℝ) 
  (h1 : e = Real.sqrt 2 / 2) 
  (h2 : l = 8) : 
  ellipse_equation e l = fun (x, y) => x^2 / 32 + y^2 / 16 = 1 := by
sorry

end ellipse_standard_equation_l1226_122651


namespace line_ellipse_intersection_range_l1226_122634

/-- The range of m for which the line y = kx + 2 (k ∈ ℝ) always intersects the ellipse x² + y²/m = 1 -/
theorem line_ellipse_intersection_range :
  ∀ (k : ℝ), ∃ (x y : ℝ), 
    y = k * x + 2 ∧ 
    x^2 + y^2 / m = 1 ↔ 
    m ∈ Set.Ici (4 : ℝ) :=
by sorry

end line_ellipse_intersection_range_l1226_122634


namespace units_digit_sum_factorials_9999_l1226_122660

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The sum of factorials from 1 to n -/
def sumFactorials (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => factorial (n + 1) + sumFactorials n

theorem units_digit_sum_factorials_9999 :
  unitsDigit (sumFactorials 9999) = 3 := by
  sorry

end units_digit_sum_factorials_9999_l1226_122660


namespace stating_discount_calculation_l1226_122620

/-- Represents the profit percentage after discount -/
def profit_after_discount : ℝ := 25

/-- Represents the profit percentage without discount -/
def profit_without_discount : ℝ := 38.89

/-- Represents the discount percentage -/
def discount_percentage : ℝ := 10

/-- 
Theorem stating that given the profit percentages with and without discount, 
the discount percentage is 10%
-/
theorem discount_calculation (cost : ℝ) (cost_positive : cost > 0) :
  let selling_price := cost * (1 + profit_after_discount / 100)
  let marked_price := cost * (1 + profit_without_discount / 100)
  selling_price = marked_price * (1 - discount_percentage / 100) :=
by
  sorry


end stating_discount_calculation_l1226_122620


namespace determinant_is_zero_l1226_122662

-- Define the polynomial and its roots
variable (p q r : ℝ)
variable (a b c d : ℝ)

-- Define the condition that a, b, c, d are roots of the polynomial
def are_roots (a b c d p q r : ℝ) : Prop :=
  a^4 + 2*a^3 + p*a^2 + q*a + r = 0 ∧
  b^4 + 2*b^3 + p*b^2 + q*b + r = 0 ∧
  c^4 + 2*c^3 + p*c^2 + q*c + r = 0 ∧
  d^4 + 2*d^3 + p*d^2 + q*d + r = 0

-- Define the matrix
def matrix (a b c d : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![a, b, c, d;
     b, c, d, a;
     c, d, a, b;
     d, a, b, c]

-- State the theorem
theorem determinant_is_zero (p q r : ℝ) (a b c d : ℝ) 
  (h : are_roots a b c d p q r) : 
  Matrix.det (matrix a b c d) = 0 := by
  sorry

end determinant_is_zero_l1226_122662


namespace college_student_count_l1226_122627

/-- Represents the number of students in each category -/
structure StudentCount where
  boys : ℕ
  girls : ℕ
  nonBinary : ℕ

/-- Calculates the total number of students -/
def totalStudents (s : StudentCount) : ℕ :=
  s.boys + s.girls + s.nonBinary

/-- Theorem: Given the ratio and number of girls, prove the total number of students -/
theorem college_student_count :
  ∀ (s : StudentCount),
    s.boys * 5 = s.girls * 8 →
    s.nonBinary * 5 = s.girls * 3 →
    s.girls = 400 →
    totalStudents s = 1280 := by
  sorry


end college_student_count_l1226_122627


namespace lcm_perfect_square_l1226_122614

theorem lcm_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a*b) % (a*b*(a - b)) = 0) : 
  ∃ k : ℕ, Nat.lcm a b = k^2 := by
  sorry

end lcm_perfect_square_l1226_122614


namespace friend_spent_more_l1226_122668

theorem friend_spent_more (total : ℕ) (friend_spent : ℕ) (you_spent : ℕ) : 
  total = 11 → friend_spent = 7 → total = friend_spent + you_spent → friend_spent > you_spent →
  friend_spent - you_spent = 3 := by
sorry

end friend_spent_more_l1226_122668


namespace twice_product_of_sum_and_difference_l1226_122607

theorem twice_product_of_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 80) 
  (diff_eq : x - y = 10) : 
  2 * x * y = 3150 := by
sorry

end twice_product_of_sum_and_difference_l1226_122607


namespace total_balls_l1226_122633

/-- Given 2 boxes, each containing 3 balls, the total number of balls is 6. -/
theorem total_balls (num_boxes : ℕ) (balls_per_box : ℕ) (h1 : num_boxes = 2) (h2 : balls_per_box = 3) :
  num_boxes * balls_per_box = 6 := by
  sorry

end total_balls_l1226_122633


namespace tom_golf_performance_l1226_122690

/-- Represents a round of golf --/
structure GolfRound where
  holes : ℕ
  averageStrokes : ℚ
  parValue : ℕ

/-- Calculates the total strokes for a round --/
def totalStrokes (round : GolfRound) : ℚ :=
  round.averageStrokes * round.holes

/-- Calculates the par for a round --/
def parForRound (round : GolfRound) : ℕ :=
  round.parValue * round.holes

theorem tom_golf_performance :
  let rounds : List GolfRound := [
    { holes := 9, averageStrokes := 4, parValue := 3 },
    { holes := 9, averageStrokes := 3.5, parValue := 3 },
    { holes := 9, averageStrokes := 5, parValue := 3 },
    { holes := 9, averageStrokes := 3, parValue := 3 },
    { holes := 9, averageStrokes := 4.5, parValue := 3 }
  ]
  let totalStrokesTaken := (rounds.map totalStrokes).sum
  let totalPar := (rounds.map parForRound).sum
  totalStrokesTaken - totalPar = 45 := by sorry

end tom_golf_performance_l1226_122690


namespace corn_acres_l1226_122649

theorem corn_acres (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : beans_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) : 
  (total_land * corn_ratio) / (beans_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end corn_acres_l1226_122649


namespace basketball_game_scores_l1226_122622

/-- Represents the scores of a team in a four-quarter basketball game -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the given scores form an increasing geometric sequence -/
def isGeometricSequence (scores : TeamScores) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ 
    scores.q2 = scores.q1 * r ∧
    scores.q3 = scores.q2 * r ∧
    scores.q4 = scores.q3 * r

/-- Checks if the given scores form an increasing arithmetic sequence -/
def isArithmeticSequence (scores : TeamScores) : Prop :=
  ∃ (d : ℕ), d > 0 ∧
    scores.q2 = scores.q1 + d ∧
    scores.q3 = scores.q2 + d ∧
    scores.q4 = scores.q3 + d

/-- The main theorem representing the basketball game scenario -/
theorem basketball_game_scores 
  (tigers lions : TeamScores)
  (h1 : tigers.q1 = lions.q1)  -- Tied at the end of first quarter
  (h2 : isGeometricSequence tigers)
  (h3 : isArithmeticSequence lions)
  (h4 : tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 = 
        lions.q1 + lions.q2 + lions.q3 + lions.q4 + 4)  -- Tigers won by 4 points
  (h5 : tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 ≤ 120)  -- Max score constraint for Tigers
  (h6 : lions.q1 + lions.q2 + lions.q3 + lions.q4 ≤ 120)  -- Max score constraint for Lions
  : tigers.q1 + tigers.q2 + lions.q1 + lions.q2 = 23 :=
by
  sorry  -- Proof omitted as per instructions


end basketball_game_scores_l1226_122622


namespace rectangle_ratio_l1226_122624

theorem rectangle_ratio (w : ℝ) (h1 : w > 0) (h2 : 2 * w + 2 * 10 = 36) : w / 10 = 4 / 5 := by
  sorry

end rectangle_ratio_l1226_122624


namespace odd_function_property_l1226_122626

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_even : is_even (fun x ↦ f (x + 2))) 
  (h_f_neg_one : f (-1) = 1) : 
  f 2017 + f 2016 = -1 := by
  sorry

end odd_function_property_l1226_122626


namespace min_abs_z_plus_2i_l1226_122691

theorem min_abs_z_plus_2i (z : ℂ) (h : Complex.abs (z^2 - 3) = Complex.abs (z * (z - 3*I))) :
  Complex.abs (z + 2*I) ≥ (7 + Real.sqrt 3) / 2 := by
  sorry

end min_abs_z_plus_2i_l1226_122691


namespace remainder_problem_l1226_122644

theorem remainder_problem : (55^55 + 15) % 8 = 4 := by
  sorry

end remainder_problem_l1226_122644


namespace julia_tuesday_kids_l1226_122645

/-- The number of kids Julia played with on Tuesday -/
def kids_on_tuesday (total kids_monday kids_wednesday : ℕ) : ℕ :=
  total - kids_monday - kids_wednesday

theorem julia_tuesday_kids :
  kids_on_tuesday 34 17 2 = 15 := by
  sorry

end julia_tuesday_kids_l1226_122645


namespace complex_modulus_l1226_122674

theorem complex_modulus (z : ℂ) (h : z * (2 - Complex.I) = 1 + Complex.I) : Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end complex_modulus_l1226_122674


namespace intersecting_lines_sum_l1226_122639

/-- Two lines intersecting at a point -/
structure IntersectingLines where
  m : ℝ
  b : ℝ
  intersect_x : ℝ
  intersect_y : ℝ
  eq1 : intersect_y = 2 * m * intersect_x + 5
  eq2 : intersect_y = 4 * intersect_x + b

/-- The sum of b and m for two intersecting lines -/
def sum_b_m (lines : IntersectingLines) : ℝ :=
  lines.b + lines.m

/-- Theorem: For two lines y = 2mx + 5 and y = 4x + b intersecting at (4, 17), b + m = 2.5 -/
theorem intersecting_lines_sum (lines : IntersectingLines)
    (h1 : lines.intersect_x = 4)
    (h2 : lines.intersect_y = 17) :
    sum_b_m lines = 2.5 := by
  sorry


end intersecting_lines_sum_l1226_122639


namespace problem_solution_l1226_122613

theorem problem_solution : ∃ Y : ℚ, 
  let A : ℚ := 2010 / 3
  let B : ℚ := A / 3
  Y = A + B ∧ Y = 893 + 1/3 := by
sorry

end problem_solution_l1226_122613


namespace no_infinite_prime_sequence_l1226_122612

theorem no_infinite_prime_sequence :
  ¬∃ (p : ℕ → ℕ), (∀ n, Prime (p n)) ∧
    (∀ k > 0, p k = 2 * p (k - 1) + 1 ∨ p k = 2 * p (k - 1) - 1) ∧
    (∀ m, ∃ n > m, p n ≠ 0) :=
by sorry

end no_infinite_prime_sequence_l1226_122612


namespace ring_price_calculation_l1226_122628

def total_revenue : ℕ := 80
def necklace_price : ℕ := 12
def necklaces_sold : ℕ := 4
def rings_sold : ℕ := 8

theorem ring_price_calculation : 
  ∃ (ring_price : ℕ), 
    necklaces_sold * necklace_price + rings_sold * ring_price = total_revenue ∧ 
    ring_price = 4 := by
  sorry

end ring_price_calculation_l1226_122628


namespace two_trucks_meeting_problem_l1226_122685

/-- The problem of two trucks meeting under different conditions -/
theorem two_trucks_meeting_problem 
  (t : ℝ) -- Time of meeting in normal conditions
  (s : ℝ) -- Length of the route AB
  (v1 v2 : ℝ) -- Speeds of trucks from A and B respectively
  (h1 : t = 8 + 40/60) -- Meeting time is 8 hours 40 minutes
  (h2 : v1 * t = s - 62/5) -- Distance traveled by first truck in normal conditions
  (h3 : v2 * t = 62/5) -- Distance traveled by second truck in normal conditions
  (h4 : v1 * (t - 1/12) = 62/5) -- Distance traveled by first truck in modified conditions
  (h5 : v2 * (t + 1/8) = s - 62/5) -- Distance traveled by second truck in modified conditions
  : v1 = 38.4 ∧ v2 = 25.6 ∧ s = 16 := by
  sorry


end two_trucks_meeting_problem_l1226_122685


namespace macaroon_weight_l1226_122648

theorem macaroon_weight (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) :
  total_macaroons = 12 →
  weight_per_macaroon = 5 →
  num_bags = 4 →
  total_macaroons % num_bags = 0 →
  (total_macaroons - total_macaroons / num_bags) * weight_per_macaroon = 45 := by
  sorry

end macaroon_weight_l1226_122648


namespace cubic_equation_third_root_l1226_122654

theorem cubic_equation_third_root 
  (a b : ℚ) 
  (h1 : a * (-1)^3 + (a + 3*b) * (-1)^2 + (2*b - 4*a) * (-1) + (10 - a) = 0)
  (h2 : a * 4^3 + (a + 3*b) * 4^2 + (2*b - 4*a) * 4 + (10 - a) = 0)
  : ∃ (x : ℚ), x = -62/19 ∧ 
    a * x^3 + (a + 3*b) * x^2 + (2*b - 4*a) * x + (10 - a) = 0 :=
by sorry

end cubic_equation_third_root_l1226_122654


namespace dividend_percentage_calculation_l1226_122698

theorem dividend_percentage_calculation (face_value : ℝ) (purchase_price : ℝ) (return_on_investment : ℝ) :
  face_value = 40 →
  purchase_price = 20 →
  return_on_investment = 0.25 →
  (purchase_price * return_on_investment) / face_value = 0.125 :=
by sorry

end dividend_percentage_calculation_l1226_122698


namespace jasmine_purchase_cost_l1226_122664

/-- The cost of Jasmine's purchase -/
def total_cost (coffee_pounds : ℕ) (milk_gallons : ℕ) (coffee_price : ℚ) (milk_price : ℚ) : ℚ :=
  coffee_pounds * coffee_price + milk_gallons * milk_price

/-- Proof that Jasmine's purchase costs $17 -/
theorem jasmine_purchase_cost :
  total_cost 4 2 (5/2) (7/2) = 17 := by
  sorry

end jasmine_purchase_cost_l1226_122664


namespace product_ab_l1226_122696

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def complex_equation (a b : ℝ) : Prop :=
  (1 + 7 * i) / (2 - i) = (a : ℂ) + b * i

-- Theorem statement
theorem product_ab (a b : ℝ) (h : complex_equation a b) : a * b = -5 := by
  sorry

end product_ab_l1226_122696


namespace smallest_integer_in_set_l1226_122658

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7)) →
  n ≥ -1 :=
by sorry

end smallest_integer_in_set_l1226_122658


namespace smallest_multiple_of_6_and_15_l1226_122659

theorem smallest_multiple_of_6_and_15 : 
  ∃ (b : ℕ), b > 0 ∧ 6 ∣ b ∧ 15 ∣ b ∧ ∀ (x : ℕ), x > 0 → 6 ∣ x → 15 ∣ x → b ≤ x :=
by sorry

end smallest_multiple_of_6_and_15_l1226_122659


namespace solution_system_equations_l1226_122609

theorem solution_system_equations (w x y z : ℝ) 
  (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : w + x + y + z = 12)
  (h2 : w * x * y * z = w * x + w * y + w * z + x * y + x * z + y * z + 27) :
  w = 3 ∧ x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end solution_system_equations_l1226_122609


namespace wendy_second_level_treasures_l1226_122693

def points_per_treasure : ℕ := 5
def treasures_first_level : ℕ := 4
def total_score : ℕ := 35

theorem wendy_second_level_treasures :
  (total_score - points_per_treasure * treasures_first_level) / points_per_treasure = 3 := by
  sorry

end wendy_second_level_treasures_l1226_122693


namespace wage_restoration_l1226_122625

theorem wage_restoration (original_wage : ℝ) (h : original_wage > 0) :
  let reduced_wage := 0.7 * original_wage
  let raise_percentage := 100 * (1 / 0.7 - 1)
  reduced_wage * (1 + raise_percentage / 100) = original_wage := by
sorry

end wage_restoration_l1226_122625


namespace rafael_earnings_l1226_122652

def hours_monday : ℕ := 10
def hours_tuesday : ℕ := 8
def hours_left : ℕ := 20
def hourly_rate : ℕ := 20

theorem rafael_earnings : 
  (hours_monday + hours_tuesday + hours_left) * hourly_rate = 760 := by
  sorry

end rafael_earnings_l1226_122652


namespace triangle_inequality_l1226_122604

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) (S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0)
  (h_sum_angles : A + B + C = Real.pi)
  (h_area : S = (1/2) * a * b * Real.sin C)
  (h_side_a : a = b * Real.sin C / Real.sin A)
  (h_side_b : b = c * Real.sin A / Real.sin B)
  (h_side_c : c = a * Real.sin B / Real.sin C) :
  a^2 * Real.tan (A/2) + b^2 * Real.tan (B/2) + c^2 * Real.tan (C/2) ≥ 4 * S :=
by sorry

end triangle_inequality_l1226_122604


namespace michael_remaining_yards_l1226_122642

/-- Represents the length of an ultra-marathon in miles and yards -/
structure UltraMarathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def yards_per_mile : ℕ := 1760

def ultra_marathon : UltraMarathon := ⟨50, 800⟩

def michael_marathons : ℕ := 5

theorem michael_remaining_yards :
  ∃ (m : ℕ) (y : ℕ), 
    y < yards_per_mile ∧
    (michael_marathons * ultra_marathon.miles * yards_per_mile + 
     michael_marathons * ultra_marathon.yards) = 
    (m * yards_per_mile + y) ∧
    y = 480 := by
  sorry

end michael_remaining_yards_l1226_122642


namespace fox_alice_numbers_l1226_122699

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divisible_by_at_least_three (n : ℕ) : Prop :=
  (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 = 0) ∨
  (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∨
  (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 6 = 0) ∨
  (n % 2 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0) ∨
  (n % 2 = 0 ∧ n % 4 = 0 ∧ n % 6 = 0) ∨
  (n % 2 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0) ∨
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 6 = 0) ∨
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
  (n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0)

def not_divisible_by_exactly_two (n : ℕ) : Prop :=
  ¬((n % 2 ≠ 0 ∧ n % 3 ≠ 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
    (n % 2 ≠ 0 ∧ n % 3 = 0 ∧ n % 4 ≠ 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
    (n % 2 ≠ 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 ≠ 0 ∧ n % 6 = 0) ∨
    (n % 2 ≠ 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 ≠ 0) ∨
    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 ≠ 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 = 0 ∧ n % 5 ≠ 0 ∧ n % 6 = 0) ∨
    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 ≠ 0) ∨
    (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 ≠ 0 ∧ n % 5 ≠ 0 ∧ n % 6 = 0) ∨
    (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 ≠ 0 ∧ n % 5 = 0 ∧ n % 6 ≠ 0) ∨
    (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 ≠ 0 ∧ n % 6 ≠ 0))

theorem fox_alice_numbers :
  ∃! (s : Finset ℕ), ∀ n, n ∈ s ↔ 
    is_two_digit n ∧ 
    divisible_by_at_least_three n ∧ 
    not_divisible_by_exactly_two n ∧
    s.card = 8 := by sorry

end fox_alice_numbers_l1226_122699


namespace fifteenth_odd_multiple_of_5_l1226_122697

/-- The nth odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- Theorem stating that the 15th positive integer that is both odd and a multiple of 5 is 145 -/
theorem fifteenth_odd_multiple_of_5 : nthOddMultipleOf5 15 = 145 := by
  sorry

end fifteenth_odd_multiple_of_5_l1226_122697


namespace super_eighteen_total_games_l1226_122684

/-- Calculates the total number of games in the Super Eighteen Football League -/
def super_eighteen_games (num_divisions : ℕ) (teams_per_division : ℕ) : ℕ :=
  let intra_division_games := num_divisions * teams_per_division * (teams_per_division - 1)
  let inter_division_games := num_divisions * teams_per_division * teams_per_division
  intra_division_games + inter_division_games

/-- Theorem stating that the Super Eighteen Football League schedules 450 games -/
theorem super_eighteen_total_games :
  super_eighteen_games 2 9 = 450 := by
  sorry

end super_eighteen_total_games_l1226_122684


namespace certain_number_problem_l1226_122638

theorem certain_number_problem : 
  ∃ x : ℝ, (0.1 * x + 0.15 * 50 = 10.5) ∧ x = 30 := by
  sorry

end certain_number_problem_l1226_122638


namespace intersection_P_Q_l1226_122616

def P : Set ℝ := {-3, 0, 2, 4}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by sorry

end intersection_P_Q_l1226_122616


namespace max_value_theorem_l1226_122637

theorem max_value_theorem (x y : ℝ) (h : x^2/4 + y^2 = 1) :
  ∃ (max_val : ℝ), max_val = (1 + Real.sqrt 2) / 2 ∧
  ∀ (z : ℝ), z = x*y/(x + 2*y - 2) → z ≤ max_val :=
sorry

end max_value_theorem_l1226_122637


namespace ticket_sales_total_l1226_122689

/-- Calculates the total amount collected from ticket sales given the ticket prices, total tickets sold, and number of children attending. -/
def total_amount_collected (child_price adult_price total_tickets children_count : ℕ) : ℕ :=
  let adult_count := total_tickets - children_count
  child_price * children_count + adult_price * adult_count

/-- Theorem stating that the total amount collected from ticket sales is $1875 given the specified conditions. -/
theorem ticket_sales_total :
  total_amount_collected 6 9 225 50 = 1875 := by
  sorry

end ticket_sales_total_l1226_122689


namespace P_range_l1226_122655

theorem P_range (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let P := (a^2 / (a^2 + b^2 + c^2)) + (b^2 / (b^2 + c^2 + d^2)) +
           (c^2 / (c^2 + d^2 + a^2)) + (d^2 / (d^2 + a^2 + b^2))
  1 < P ∧ P < 2 := by
  sorry

end P_range_l1226_122655


namespace cone_base_radius_l1226_122631

/-- A cone formed by a semicircle with radius 2 cm has a base circle with radius 1 cm -/
theorem cone_base_radius (r : ℝ) (h : r = 2) : 
  (2 * Real.pi * r / 2) / (2 * Real.pi) = 1 := by sorry

end cone_base_radius_l1226_122631


namespace perimeter_difference_l1226_122680

/-- Represents a figure made of unit squares -/
structure UnitSquareFigure where
  perimeter : ℕ

/-- The first figure in the problem -/
def figure1 : UnitSquareFigure :=
  { perimeter := 24 }

/-- The second figure in the problem -/
def figure2 : UnitSquareFigure :=
  { perimeter := 33 }

/-- The theorem stating the difference between the perimeters of the two figures -/
theorem perimeter_difference :
  (figure2.perimeter - figure1.perimeter : ℤ) = 9 := by
  sorry

end perimeter_difference_l1226_122680


namespace inequality_proof_l1226_122683

theorem inequality_proof (x : ℝ) (n : ℕ) (h : x > 0) :
  1 + x^(n+1) ≥ (2*x)^n / (1+x)^(n-1) := by
  sorry

end inequality_proof_l1226_122683


namespace quadratic_transformation_l1226_122650

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := 2 * (x - 175) * (x - 176) + 6

/-- The transformed quadratic function -/
def g (x : ℝ) : ℝ := f x - 6

/-- The roots of the transformed function -/
def root1 : ℝ := 175
def root2 : ℝ := 176

theorem quadratic_transformation :
  (g root1 = 0) ∧ 
  (g root2 = 0) ∧ 
  (root2 - root1 = 1) := by
  sorry

end quadratic_transformation_l1226_122650


namespace average_of_data_set_l1226_122643

def data_set : List ℝ := [9.8, 9.9, 10, 10.1, 10.2]

theorem average_of_data_set :
  (List.sum data_set) / (List.length data_set) = 10 := by
  sorry

end average_of_data_set_l1226_122643


namespace daves_phone_files_l1226_122605

theorem daves_phone_files :
  let initial_apps : ℕ := 15
  let initial_files : ℕ := 24
  let final_apps : ℕ := 21
  let app_file_difference : ℕ := 17
  let files_left : ℕ := final_apps - app_file_difference
  files_left = 4 := by sorry

end daves_phone_files_l1226_122605


namespace binomial_12_9_l1226_122619

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end binomial_12_9_l1226_122619


namespace line_relations_l1226_122602

-- Define the structure for a line
structure Line where
  slope : ℝ
  angle_of_inclination : ℝ

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem line_relations (l1 l2 : Line) (h_distinct : l1 ≠ l2) :
  (parallel l1 l2 → l1.slope = l2.slope) ∧
  (l1.slope = l2.slope → parallel l1 l2) ∧
  (parallel l1 l2 → l1.angle_of_inclination = l2.angle_of_inclination) ∧
  (l1.angle_of_inclination = l2.angle_of_inclination → parallel l1 l2) := by
  sorry

end line_relations_l1226_122602


namespace baker_cake_difference_l1226_122681

/-- Given Baker's cake inventory and transactions, prove the difference between sold and bought cakes. -/
theorem baker_cake_difference (initial_cakes bought_cakes sold_cakes : ℚ) 
  (h1 : initial_cakes = 8.5)
  (h2 : bought_cakes = 139.25)
  (h3 : sold_cakes = 145.75) :
  sold_cakes - bought_cakes = 6.5 := by
  sorry

#eval (145.75 : ℚ) - (139.25 : ℚ)

end baker_cake_difference_l1226_122681


namespace cubic_sum_over_product_l1226_122606

theorem cubic_sum_over_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) :
  (a^3 + b^3 + c^3) / (a * b * c) = (1 + 3*(a - b)^2) / (a * b * (1 - a - b)) := by
sorry

end cubic_sum_over_product_l1226_122606


namespace inflection_point_is_center_of_symmetry_l1226_122617

/-- Represents a cubic function of the form ax³ + bx² + cx + d -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_nonzero : a ≠ 0

/-- The given cubic function x³ - 3x² + 3x -/
def f : CubicFunction := {
  a := 1
  b := -3
  c := 3
  d := 0
  a_nonzero := by norm_num
}

/-- Evaluates a cubic function at a given x -/
def evaluate (f : CubicFunction) (x : ℝ) : ℝ :=
  f.a * x^3 + f.b * x^2 + f.c * x + f.d

/-- Computes the second derivative of a cubic function -/
def secondDerivative (f : CubicFunction) (x : ℝ) : ℝ :=
  6 * f.a * x + 2 * f.b

/-- An inflection point of a cubic function -/
structure InflectionPoint (f : CubicFunction) where
  x : ℝ
  y : ℝ
  is_inflection : secondDerivative f x = 0
  on_curve : y = evaluate f x

theorem inflection_point_is_center_of_symmetry :
  ∃ (p : InflectionPoint f), p.x = 1 ∧ p.y = 1 := by sorry

end inflection_point_is_center_of_symmetry_l1226_122617


namespace only_coin_toss_is_random_l1226_122608

-- Define the type for events
inductive Event
  | CoinToss : Event
  | ChargeAttraction : Event
  | WaterFreeze : Event

-- Define a predicate for random events
def is_random_event : Event → Prop :=
  fun e => match e with
    | Event.CoinToss => True
    | _ => False

-- Theorem statement
theorem only_coin_toss_is_random :
  (is_random_event Event.CoinToss) ∧
  (¬ is_random_event Event.ChargeAttraction) ∧
  (¬ is_random_event Event.WaterFreeze) :=
by sorry

end only_coin_toss_is_random_l1226_122608


namespace average_annual_cost_reduction_l1226_122657

theorem average_annual_cost_reduction (total_reduction : Real) 
  (h : total_reduction = 0.36) : 
  ∃ x : Real, x > 0 ∧ x < 1 ∧ (1 - x)^2 = 1 - total_reduction :=
sorry

end average_annual_cost_reduction_l1226_122657


namespace min_brilliant_product_l1226_122661

/-- A triple of integers (a, b, c) is brilliant if:
    1. a > b > c are prime numbers
    2. a = b + 2c
    3. a + b + c is a perfect square number -/
def is_brilliant (a b c : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧
  a > b ∧ b > c ∧
  a = b + 2 * c ∧
  ∃ k, a + b + c = k * k

/-- The minimum value of abc for a brilliant triple (a, b, c) is 35651 -/
theorem min_brilliant_product :
  (∀ a b c : ℕ, is_brilliant a b c → a * b * c ≥ 35651) ∧
  ∃ a b c : ℕ, is_brilliant a b c ∧ a * b * c = 35651 :=
sorry

end min_brilliant_product_l1226_122661


namespace constant_distance_l1226_122665

/-- Represents an ellipse centered at the origin with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h : 0 < b ∧ b < a
  h_e : e = Real.sqrt 2 / 2
  h_eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : ∃ (x y : ℝ), x^2 / E.a^2 + y^2 / E.b^2 = 1 ∧ y = k * x + m

/-- The theorem to be proved -/
theorem constant_distance (E : Ellipse) (l : IntersectingLine E) :
  ∃ (P Q : ℝ × ℝ) (d : ℝ),
    P ∈ {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1} ∧
    Q ∈ {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1} ∧
    P.1 * Q.1 + P.2 * Q.2 = 0 ∧
    d = Real.sqrt 6 / 3 ∧
    d = abs m / Real.sqrt (l.k^2 + 1) :=
  sorry

end constant_distance_l1226_122665


namespace shadow_length_ratio_l1226_122600

theorem shadow_length_ratio (α β : Real) 
  (h1 : Real.tan (α - β) = 1 / 3)
  (h2 : Real.tan β = 1) :
  Real.tan α = 2 := by
  sorry

end shadow_length_ratio_l1226_122600


namespace largest_fraction_l1226_122671

theorem largest_fraction : 
  let f1 := 8 / 15
  let f2 := 5 / 11
  let f3 := 19 / 37
  let f4 := 101 / 199
  let f5 := 153 / 305
  (f1 > f2 ∧ f1 > f3 ∧ f1 > f4 ∧ f1 > f5) := by
  sorry

end largest_fraction_l1226_122671


namespace log4_20_approximation_l1226_122621

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.300
def log10_5_approx : ℝ := 0.699

-- Define the target approximation
def target_approx : ℚ := 13/6

-- State the theorem
theorem log4_20_approximation : 
  ∀ (ε : ℝ), ε > 0 → 
  ∃ (δ : ℝ), δ > 0 ∧ 
  (∀ (x y : ℝ), 
    |x - log10_2_approx| < δ ∧ 
    |y - log10_5_approx| < δ → 
    |((1 + x) / (2 * x)) - target_approx| < ε) :=
sorry

end log4_20_approximation_l1226_122621


namespace sixth_quiz_score_l1226_122632

def john_scores : List ℕ := [85, 88, 90, 92, 83]
def target_mean : ℕ := 90
def num_quizzes : ℕ := 6

theorem sixth_quiz_score (score : ℕ) : 
  (john_scores.sum + score) / num_quizzes = target_mean ↔ score = 102 := by
  sorry

end sixth_quiz_score_l1226_122632


namespace set_c_forms_triangle_set_a_not_triangle_set_b_not_triangle_set_d_not_triangle_triangle_formation_result_l1226_122663

/-- A function that checks if three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the set (4, 5, 7) can form a triangle --/
theorem set_c_forms_triangle : can_form_triangle 4 5 7 := by sorry

/-- Theorem stating that the set (1, 3, 4) cannot form a triangle --/
theorem set_a_not_triangle : ¬ can_form_triangle 1 3 4 := by sorry

/-- Theorem stating that the set (2, 2, 7) cannot form a triangle --/
theorem set_b_not_triangle : ¬ can_form_triangle 2 2 7 := by sorry

/-- Theorem stating that the set (3, 3, 6) cannot form a triangle --/
theorem set_d_not_triangle : ¬ can_form_triangle 3 3 6 := by sorry

/-- Main theorem combining all results --/
theorem triangle_formation_result :
  can_form_triangle 4 5 7 ∧
  ¬ can_form_triangle 1 3 4 ∧
  ¬ can_form_triangle 2 2 7 ∧
  ¬ can_form_triangle 3 3 6 := by sorry

end set_c_forms_triangle_set_a_not_triangle_set_b_not_triangle_set_d_not_triangle_triangle_formation_result_l1226_122663


namespace correct_multiplication_result_l1226_122601

theorem correct_multiplication_result (result : ℕ) (wrong_digits : List ℕ) :
  result = 867559827931 ∧
  wrong_digits = [8, 6, 7, 5, 2, 7, 9] ∧
  (∃ n : ℕ, n * 98765 = result) →
  ∃ m : ℕ, m * 98765 = 888885 := by
  sorry

end correct_multiplication_result_l1226_122601


namespace circle_tangent_area_zero_l1226_122676

-- Define the circle struct
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line struct
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

def CircleTangentToLine (c : Circle) (l : Line) : Prop := sorry

def CircleInternallyTangent (c1 c2 : Circle) : Prop := sorry

def PointBetween (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

def TriangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem circle_tangent_area_zero 
  (P Q R : Circle)
  (l : Line)
  (P' Q' R' : ℝ × ℝ)
  (h1 : P.radius = 2)
  (h2 : Q.radius = 3)
  (h3 : R.radius = 4)
  (h4 : CircleTangentToLine P l)
  (h5 : CircleTangentToLine Q l)
  (h6 : CircleTangentToLine R l)
  (h7 : P'.1 = P.center.1 ∧ P'.2 = P.center.2 + P.radius)
  (h8 : Q'.1 = Q.center.1 ∧ Q'.2 = Q.center.2 + Q.radius)
  (h9 : R'.1 = R.center.1 ∧ R'.2 = R.center.2 + R.radius)
  (h10 : PointBetween P' Q' R')
  (h11 : CircleInternallyTangent Q P)
  (h12 : CircleInternallyTangent Q R) :
  TriangleArea P.center Q.center R.center = 0 := by sorry

end circle_tangent_area_zero_l1226_122676


namespace committee_formation_count_l1226_122641

/-- The number of ways to choose k elements from n elements --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players in the basketball team --/
def total_players : ℕ := 13

/-- The size of the committee to be formed --/
def committee_size : ℕ := 4

/-- The number of players to be chosen after including player A --/
def remaining_to_choose : ℕ := committee_size - 1

/-- The number of players to choose from after excluding player A --/
def players_to_choose_from : ℕ := total_players - 1

theorem committee_formation_count :
  choose players_to_choose_from remaining_to_choose = 220 := by
  sorry

end committee_formation_count_l1226_122641


namespace arithmetic_mean_two_digit_multiples_of_8_l1226_122656

/-- The first two-digit multiple of 8 -/
def first_multiple : Nat := 16

/-- The last two-digit multiple of 8 -/
def last_multiple : Nat := 96

/-- The common difference between consecutive multiples of 8 -/
def common_difference : Nat := 8

/-- The number of two-digit multiples of 8 -/
def num_multiples : Nat := (last_multiple - first_multiple) / common_difference + 1

/-- The arithmetic mean of all positive two-digit multiples of 8 is 56 -/
theorem arithmetic_mean_two_digit_multiples_of_8 :
  (first_multiple + last_multiple) * num_multiples / (2 * num_multiples) = 56 := by
  sorry


end arithmetic_mean_two_digit_multiples_of_8_l1226_122656


namespace polynomial_simplification_l1226_122687

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + 6 * y^10 + 4 + 2 * y^9) =
  15 * y^13 - y^12 + 12 * y^11 - 6 * y^10 - 4 * y^9 + 12 * y - 8 := by
  sorry

end polynomial_simplification_l1226_122687


namespace at_least_one_sum_of_primes_l1226_122618

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a number is the sum of two primes
def isSumOfTwoPrimes (n : ℕ) : Prop :=
  ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ n = p + q

-- Theorem statement
theorem at_least_one_sum_of_primes (n : ℕ) (h : n > 1) :
  isSumOfTwoPrimes (2*n) ∨ isSumOfTwoPrimes (2*n + 2) ∨ isSumOfTwoPrimes (2*n + 4) :=
sorry

end at_least_one_sum_of_primes_l1226_122618


namespace union_of_A_and_B_complement_of_intersection_A_and_B_l1226_122678

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | x - 3 ≥ 0}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {x | x ≥ -2} := by sorry

-- Theorem for Ā ∩ B
theorem complement_of_intersection_A_and_B : (A ∩ B)ᶜ = {x | x < 3 ∨ x ≥ 4} := by sorry

end union_of_A_and_B_complement_of_intersection_A_and_B_l1226_122678


namespace ellipse_line_intersection_range_l1226_122629

-- Define the line equation
def line (k x y : ℝ) : Prop := 2 * k * x - y + 1 = 0

-- Define the ellipse equation
def ellipse (x y m : ℝ) : Prop := (x^2 / 9) + (y^2 / m) = 1

-- State the theorem
theorem ellipse_line_intersection_range (k : ℝ) :
  (∀ m : ℝ, (∀ x y : ℝ, line k x y → ellipse x y m → (∃ x' y' : ℝ, line k x' y' ∧ ellipse x' y' m))) →
  (∃ S : Set ℝ, S = {m : ℝ | m ∈ Set.Icc 1 9 ∪ Set.Ioi 9}) :=
sorry

end ellipse_line_intersection_range_l1226_122629


namespace second_order_size_l1226_122692

/-- Proves that given the specified production rates and average output, the second order contains 60 cogs. -/
theorem second_order_size
  (initial_rate : ℝ)
  (initial_order : ℝ)
  (second_rate : ℝ)
  (average_output : ℝ)
  (h1 : initial_rate = 36)
  (h2 : initial_order = 60)
  (h3 : second_rate = 60)
  (h4 : average_output = 45) :
  ∃ (second_order : ℝ),
    (initial_order + second_order) / ((initial_order / initial_rate) + (second_order / second_rate)) = average_output ∧
    second_order = 60 :=
by sorry

end second_order_size_l1226_122692


namespace square_perimeter_ratio_l1226_122677

theorem square_perimeter_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_ratio : a / b = 36 / 49) :
  (4 * Real.sqrt a) / (4 * Real.sqrt b) = 6 / 7 := by
sorry

end square_perimeter_ratio_l1226_122677


namespace triangle_area_triple_altitude_l1226_122666

theorem triangle_area_triple_altitude (b h : ℝ) (h_pos : 0 < h) :
  let A := (1/2) * b * h
  let A' := (1/2) * b * (3*h)
  A' = 3 * A := by sorry

end triangle_area_triple_altitude_l1226_122666


namespace f_minimum_value_l1226_122640

noncomputable def f (x : ℝ) : ℝ := ((2 * x - 1) * Real.exp x) / (x - 1)

theorem f_minimum_value :
  ∃ (x_min : ℝ), x_min = 3 / 2 ∧
  (∀ x : ℝ, x ≠ 1 → f x ≥ f x_min) ∧
  f x_min = 4 * Real.exp (3 / 2) :=
sorry

end f_minimum_value_l1226_122640


namespace doubled_average_l1226_122679

theorem doubled_average (n : ℕ) (original_average : ℝ) (h1 : n = 30) (h2 : original_average = 45) :
  let new_average := 2 * original_average
  new_average = 90 := by
sorry

end doubled_average_l1226_122679


namespace unique_m_for_solution_set_minimum_a_for_inequality_l1226_122610

-- Define the function f
def f (x : ℝ) := |2 * x - 1|

-- Part 1
theorem unique_m_for_solution_set :
  ∃! m : ℝ, m > 0 ∧
  (∀ x : ℝ, |2 * (x + 1/2) - 1| ≤ 2 * m + 1 ↔ x ≤ -2 ∨ x ≥ 2) ∧
  m = 3/2 :=
sorry

-- Part 2
theorem minimum_a_for_inequality :
  ∃ a : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) ∧
  (∀ a' : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a'/(2^y) + |2*x + 3|) → a ≤ a') ∧
  a = 4 :=
sorry

end unique_m_for_solution_set_minimum_a_for_inequality_l1226_122610


namespace floor_sum_inequality_l1226_122636

theorem floor_sum_inequality (x y : ℝ) :
  (⌊x⌋ : ℝ) + ⌊y⌋ ≤ ⌊x + y⌋ ∧ ⌊x + y⌋ ≤ (⌊x⌋ : ℝ) + ⌊y⌋ + 1 ∧
  ((⌊x + y⌋ = (⌊x⌋ : ℝ) + ⌊y⌋) ∨ (⌊x + y⌋ = (⌊x⌋ : ℝ) + ⌊y⌋ + 1)) ∧
  ¬((⌊x + y⌋ = (⌊x⌋ : ℝ) + ⌊y⌋) ∧ (⌊x + y⌋ = (⌊x⌋ : ℝ) + ⌊y⌋ + 1)) :=
by sorry


end floor_sum_inequality_l1226_122636


namespace fraction_multiplication_l1226_122615

theorem fraction_multiplication (x : ℚ) : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5060 = 759 := by
  sorry

end fraction_multiplication_l1226_122615


namespace vector_decomposition_l1226_122688

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![3, 1, 3]
def p : Fin 3 → ℝ := ![2, 1, 0]
def q : Fin 3 → ℝ := ![1, 0, 1]
def r : Fin 3 → ℝ := ![4, 2, 1]

/-- The decomposition of x in terms of p, q, and r -/
theorem vector_decomposition : x = (-3 : ℝ) • p + q + (2 : ℝ) • r := by sorry

end vector_decomposition_l1226_122688


namespace function_properties_l1226_122695

noncomputable section

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * Real.exp x - x - 1

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the range of y
def y_range (x₁ x₂ : ℝ) (m : ℝ) : ℝ := 
  (Real.exp x₂ - Real.exp x₁) * ((Real.exp x₂ + Real.exp x₁)⁻¹ - m)

-- Theorem statement
theorem function_properties (m : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0) →
  (∀ (x : ℝ), f m x > 0) →
  (f m 0 = 1) →
  (tangent_line 0 1) ∧
  (∀ (y : ℝ), y < 0 → ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ y = y_range x₁ x₂ m) ∧
  (1 < m ∧ m < Real.exp 1 → Real.exp (m - 1) < m ^ (Real.exp 1 - 1)) ∧
  (m = Real.exp 1 → Real.exp (m - 1) = m ^ (Real.exp 1 - 1)) ∧
  (m > Real.exp 1 → Real.exp (m - 1) > m ^ (Real.exp 1 - 1)) := by
  sorry

end

end function_properties_l1226_122695


namespace units_digit_difference_l1226_122673

def is_positive_even_integer (p : ℕ) : Prop := p > 0 ∧ p % 2 = 0

def has_positive_units_digit (p : ℕ) : Prop := p % 10 > 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_difference (p : ℕ) 
  (h1 : is_positive_even_integer p) 
  (h2 : has_positive_units_digit p) 
  (h3 : units_digit (p + 5) = 1) : 
  units_digit (p^3) - units_digit (p^2) = 0 := by
sorry

end units_digit_difference_l1226_122673


namespace unread_books_l1226_122682

theorem unread_books (total : ℕ) (read : ℕ) (h1 : total = 21) (h2 : read = 13) :
  total - read = 8 := by
  sorry

end unread_books_l1226_122682


namespace inverse_of_A_squared_l1226_122675

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = ![![-4, 1], ![0, 2]] →
  (A^2)⁻¹ = ![![16, -2], ![0, 4]] := by
sorry

end inverse_of_A_squared_l1226_122675


namespace problem_solution_l1226_122653

def set_product (A B : Set ℝ) : Set ℝ :=
  {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

def A : Set ℝ := {0, 2}
def B : Set ℝ := {1, 3}
def C : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem problem_solution :
  (set_product A B) ∩ (set_product B C) = {2, 6} := by
  sorry

end problem_solution_l1226_122653


namespace sqrt_a_div_sqrt_b_l1226_122630

theorem sqrt_a_div_sqrt_b (a b : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*a)/(53*b)) :
  Real.sqrt a / Real.sqrt b = 5/2 := by sorry

end sqrt_a_div_sqrt_b_l1226_122630


namespace train_passengers_l1226_122603

theorem train_passengers (initial : ℕ) 
  (first_off first_on second_off second_on final : ℕ) : 
  first_off = 29 → 
  first_on = 17 → 
  second_off = 27 → 
  second_on = 35 → 
  final = 116 → 
  initial = 120 → 
  initial - first_off + first_on - second_off + second_on = final :=
by sorry

end train_passengers_l1226_122603


namespace ordering_abc_l1226_122686

theorem ordering_abc : 
  let a : ℝ := 0.1 * Real.exp 0.1
  let b : ℝ := 1 / 9
  let c : ℝ := -Real.log 0.9
  c < a ∧ a < b := by sorry

end ordering_abc_l1226_122686


namespace river_depth_ratio_l1226_122669

/-- Given the depths of a river at different times, prove the ratio of depths -/
theorem river_depth_ratio 
  (depth_may : ℝ) 
  (increase_june : ℝ) 
  (depth_july : ℝ) 
  (h1 : depth_may = 5)
  (h2 : depth_july = 45)
  (h3 : depth_may + increase_june = depth_may + 10) :
  depth_july / (depth_may + increase_june) = 3 := by
  sorry

end river_depth_ratio_l1226_122669


namespace parabola_standard_equation_l1226_122646

/-- A parabola with directrix y = 1/2 has the standard equation x^2 = -2y -/
theorem parabola_standard_equation (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, y = 1/2 → (x^2 = -2*p*y ↔ y = -x^2/(2*p))) →
  p = 1 :=
by sorry

#check parabola_standard_equation

end parabola_standard_equation_l1226_122646


namespace savings_fraction_is_one_seventh_l1226_122623

/-- A worker's monthly financial situation -/
structure WorkerFinances where
  P : ℝ  -- Monthly take-home pay
  S : ℝ  -- Fraction of take-home pay saved
  E : ℝ  -- Fraction of take-home pay for expenses
  T : ℝ  -- Monthly taxes
  h_positive_pay : 0 < P
  h_valid_fractions : 0 ≤ S ∧ 0 ≤ E ∧ S + E ≤ 1

/-- The theorem stating that if total yearly savings equals twice the monthly amount not saved,
    then the savings fraction is 1/7 -/
theorem savings_fraction_is_one_seventh (w : WorkerFinances) 
    (h_savings_equality : 12 * w.P * w.S = 2 * w.P * (1 - w.S)) : 
    w.S = 1 / 7 := by
  sorry

end savings_fraction_is_one_seventh_l1226_122623


namespace connie_initial_marbles_l1226_122672

/-- The number of marbles Connie initially had -/
def initial_marbles : ℕ := 241

/-- The number of marbles Connie bought -/
def bought_marbles : ℕ := 45

/-- The number of marbles Connie gave to Juan -/
def given_to_juan : ℕ := 73

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

theorem connie_initial_marbles :
  (initial_marbles + bought_marbles) / 2 - given_to_juan = marbles_left :=
by sorry

end connie_initial_marbles_l1226_122672
