import Mathlib

namespace exam_attendance_l3198_319860

theorem exam_attendance (passed_percentage : ℝ) (failed_count : ℕ) : 
  passed_percentage = 35 →
  failed_count = 481 →
  (100 - passed_percentage) / 100 * 740 = failed_count :=
by
  sorry

end exam_attendance_l3198_319860


namespace initial_marbles_calculation_l3198_319881

/-- The number of marbles Connie initially had -/
def initial_marbles : ℝ := 972.1

/-- The number of marbles Connie gave to Juan -/
def marbles_to_juan : ℝ := 183.5

/-- The number of marbles Connie gave to Maria -/
def marbles_to_maria : ℝ := 245.7

/-- The number of marbles Connie received from Mike -/
def marbles_from_mike : ℝ := 50.3

/-- The number of marbles Connie has left -/
def marbles_left : ℝ := 593.2

/-- Theorem stating that the initial number of marbles is equal to the sum of
    the current marbles, marbles given away, minus marbles received -/
theorem initial_marbles_calculation :
  initial_marbles = marbles_left + marbles_to_juan + marbles_to_maria - marbles_from_mike :=
by sorry

end initial_marbles_calculation_l3198_319881


namespace angle_A_is_90_l3198_319823

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧ t.A + t.B + t.C = 180

-- Define the specific conditions of our triangle
def our_triangle (t : Triangle) : Prop :=
  is_valid_triangle t ∧ t.C = 2 * t.B ∧ t.B = 30

-- Theorem statement
theorem angle_A_is_90 (t : Triangle) (h : our_triangle t) : t.A = 90 := by
  sorry

end angle_A_is_90_l3198_319823


namespace boys_girls_difference_l3198_319804

/-- The number of girls on the playground -/
def num_girls : ℝ := 28.0

/-- The number of boys on the playground -/
def num_boys : ℝ := 35.0

/-- The difference between the number of boys and girls -/
def difference : ℝ := num_boys - num_girls

theorem boys_girls_difference : difference = 7.0 := by
  sorry

end boys_girls_difference_l3198_319804


namespace initial_commission_rate_l3198_319800

theorem initial_commission_rate 
  (unchanged_income : ℝ → ℝ → ℝ → ℝ → Prop)
  (new_rate : ℝ)
  (slump_percentage : ℝ) :
  let initial_rate := 4
  let slump_factor := 1 - slump_percentage / 100
  unchanged_income initial_rate new_rate slump_factor 1 →
  new_rate = 5 →
  slump_percentage = 20.000000000000007 →
  initial_rate = 4 := by
sorry

end initial_commission_rate_l3198_319800


namespace quadratic_equation_solution_l3198_319898

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 = 5*x ↔ x = 0 ∨ x = 5 := by sorry

end quadratic_equation_solution_l3198_319898


namespace range_a_characterization_l3198_319870

/-- The range of values for a where "p or q" is true and "p and q" is false -/
def range_a : Set ℝ := Set.union (Set.Ioc 0 0.5) (Set.Ico 1 2)

/-- p is true when 0 < a < 1 -/
def p_true (a : ℝ) : Prop := 0 < a ∧ a < 1

/-- q is true when 0.5 < a < 2 -/
def q_true (a : ℝ) : Prop := 0.5 < a ∧ a < 2

theorem range_a_characterization (a : ℝ) (h : a > 0) :
  a ∈ range_a ↔ (p_true a ∨ q_true a) ∧ ¬(p_true a ∧ q_true a) :=
by sorry

end range_a_characterization_l3198_319870


namespace equation_solutions_l3198_319863

theorem equation_solutions :
  (∃ x : ℚ, (3 - x) / (x + 4) = 1 / 2 ∧ x = 2 / 3) ∧
  (∃ x : ℚ, x / (x - 1) - 2 * x / (3 * x - 3) = 1 ∧ x = 3 / 2) := by
  sorry

end equation_solutions_l3198_319863


namespace modern_pentathlon_theorem_l3198_319877

/-- Represents a competitor in the Modern Pentathlon --/
inductive Competitor
| A
| B
| C

/-- Represents an event in the Modern Pentathlon --/
inductive Event
| Shooting
| Fencing
| Swimming
| Equestrian
| CrossCountryRunning

/-- Represents the place a competitor finished in an event --/
inductive Place
| First
| Second
| Third

/-- The scoring system for the Modern Pentathlon --/
structure ScoringSystem where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  a_gt_b : a > b
  b_gt_c : b > c

/-- The results of the Modern Pentathlon --/
def ModernPentathlonResults (s : ScoringSystem) :=
  Competitor → Event → Place

/-- Calculate the total score for a competitor given the results --/
def totalScore (s : ScoringSystem) (results : ModernPentathlonResults s) (competitor : Competitor) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem modern_pentathlon_theorem (s : ScoringSystem) 
  (results : ModernPentathlonResults s)
  (total_A : totalScore s results Competitor.A = 22)
  (total_B : totalScore s results Competitor.B = 9)
  (total_C : totalScore s results Competitor.C = 9)
  (B_first_equestrian : results Competitor.B Event.Equestrian = Place.First) :
  (∃ r : ModernPentathlonResults s, 
    (totalScore s r Competitor.A = 22) ∧ 
    (totalScore s r Competitor.B = 9) ∧ 
    (totalScore s r Competitor.C = 9) ∧
    (r Competitor.B Event.Equestrian = Place.First) ∧
    (r Competitor.B Event.Swimming = Place.Third)) ∧
  (∃ r : ModernPentathlonResults s, 
    (totalScore s r Competitor.A = 22) ∧ 
    (totalScore s r Competitor.B = 9) ∧ 
    (totalScore s r Competitor.C = 9) ∧
    (r Competitor.B Event.Equestrian = Place.First) ∧
    (r Competitor.C Event.Swimming = Place.Third)) :=
  sorry

end modern_pentathlon_theorem_l3198_319877


namespace largest_710_triple_l3198_319839

/-- Converts a natural number to its base-7 representation as a list of digits -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 7) :: aux (m / 7)
  aux n |>.reverse

/-- Interprets a list of digits as a base-10 number -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is a 7-10 triple -/
def is710Triple (n : ℕ) : Prop :=
  fromDigits (toBase7 n) = 3 * n

/-- States that 1422 is the largest 7-10 triple -/
theorem largest_710_triple :
  is710Triple 1422 ∧ ∀ m : ℕ, m > 1422 → ¬is710Triple m :=
sorry

end largest_710_triple_l3198_319839


namespace tank_emptying_time_l3198_319899

/-- Represents the time (in minutes) it takes to empty a water tank -/
def empty_tank_time (initial_fill : ℚ) (fill_rate : ℚ) (empty_rate : ℚ) : ℚ :=
  initial_fill / (empty_rate - fill_rate)

theorem tank_emptying_time :
  let initial_fill : ℚ := 4/5
  let fill_pipe_rate : ℚ := 1/10
  let empty_pipe_rate : ℚ := 1/6
  empty_tank_time initial_fill fill_pipe_rate empty_pipe_rate = 12 := by
sorry

end tank_emptying_time_l3198_319899


namespace abs_of_negative_three_l3198_319811

theorem abs_of_negative_three :
  ∀ x : ℝ, x = -3 → |x| = 3 := by
  sorry

end abs_of_negative_three_l3198_319811


namespace equation_solution_l3198_319835

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 5 → (x + 60 / (x - 5) = -12 ↔ x = 0 ∨ x = -7) :=
by sorry

end equation_solution_l3198_319835


namespace company_kw_price_l3198_319866

theorem company_kw_price (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (1.2 * a = 0.75 * (a + b)) → (1.2 * a = 2 * b) := by
  sorry

end company_kw_price_l3198_319866


namespace common_element_exists_l3198_319882

-- Define a type for the index of sets (1 to 2011)
def SetIndex := Fin 2011

-- Define the property of being a set of consecutive integers
def IsConsecutiveSet (S : Set ℤ) : Prop :=
  ∃ a b : ℤ, a ≤ b ∧ S = Finset.Ico a (b + 1)

-- Define the main theorem
theorem common_element_exists
  (S : SetIndex → Set ℤ)
  (h_nonempty : ∀ i, (S i).Nonempty)
  (h_consecutive : ∀ i, IsConsecutiveSet (S i))
  (h_common : ∀ i j, i ≠ j → (S i ∩ S j).Nonempty) :
  ∃ n : ℤ, n > 0 ∧ ∀ i, n ∈ S i :=
sorry

end common_element_exists_l3198_319882


namespace concave_quadrilateral_perimeter_theorem_l3198_319883

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle in 2D space -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Check if a point is inside a rectangle -/
def isInsideRectangle (p : Point) (r : Rectangle) : Prop :=
  r.topLeft.x ≤ p.x ∧ p.x ≤ r.bottomRight.x ∧
  r.bottomRight.y ≤ p.y ∧ p.y ≤ r.topLeft.y

/-- Check if a point is inside a triangle formed by three points -/
def isInsideTriangle (p : Point) (a b c : Point) : Prop :=
  sorry  -- Definition of point inside triangle

/-- Calculate the perimeter of a quadrilateral -/
def quadrilateralPerimeter (a b c d : Point) : ℝ :=
  sorry  -- Definition of quadrilateral perimeter

/-- Calculate the perimeter of a rectangle -/
def rectanglePerimeter (r : Rectangle) : ℝ :=
  sorry  -- Definition of rectangle perimeter

theorem concave_quadrilateral_perimeter_theorem 
  (r : Rectangle) (a x y z : Point) :
  isInsideRectangle a r ∧ 
  isInsideRectangle x r ∧ 
  isInsideRectangle y r ∧
  isInsideTriangle z a x y →
  (quadrilateralPerimeter a x y z < rectanglePerimeter r) ∨
  (quadrilateralPerimeter a x z y < rectanglePerimeter r) ∨
  (quadrilateralPerimeter a y z x < rectanglePerimeter r) :=
by sorry

end concave_quadrilateral_perimeter_theorem_l3198_319883


namespace team_points_distribution_l3198_319897

theorem team_points_distribution (x : ℝ) (y : ℕ) : 
  (1/3 : ℝ) * x + (3/8 : ℝ) * x + 18 + y = x ∧ 
  y ≤ 24 ∧ 
  ∀ (z : ℕ), z ≤ 8 → (y : ℝ) / 8 ≤ 3 →
  y = 17 :=
sorry

end team_points_distribution_l3198_319897


namespace vincent_stickers_l3198_319891

theorem vincent_stickers (yesterday : ℕ) (today_extra : ℕ) : 
  yesterday = 15 → today_extra = 10 → yesterday + (yesterday + today_extra) = 40 :=
by
  sorry

end vincent_stickers_l3198_319891


namespace midnight_temperature_l3198_319832

def morning_temp : Int := 7
def noon_rise : Int := 2
def midnight_drop : Int := 10

theorem midnight_temperature : 
  morning_temp + noon_rise - midnight_drop = -1 := by sorry

end midnight_temperature_l3198_319832


namespace subset_M_proof_l3198_319838

def M : Set ℝ := {x | x ≤ 2 * Real.sqrt 3}

theorem subset_M_proof (b : ℝ) (hb : b ∈ Set.Ioo 0 1) :
  {Real.sqrt (11 + b)} ⊆ M := by
  sorry

end subset_M_proof_l3198_319838


namespace prob_odd_divisor_15_factorial_l3198_319840

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of positive integer divisors of n -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- The number of odd positive integer divisors of n -/
def numOddDivisors (n : ℕ) : ℕ := sorry

/-- The probability of randomly choosing an odd divisor from the positive integer divisors of n -/
def probOddDivisor (n : ℕ) : ℚ := (numOddDivisors n : ℚ) / (numDivisors n : ℚ)

theorem prob_odd_divisor_15_factorial :
  probOddDivisor (factorial 15) = 1 / 12 := by sorry

end prob_odd_divisor_15_factorial_l3198_319840


namespace flight_chess_starting_position_l3198_319842

theorem flight_chess_starting_position (x : ℤ) :
  x - 5 + 4 + 2 - 3 + 1 = 6 → x = 7 := by
  sorry

end flight_chess_starting_position_l3198_319842


namespace sin_cos_pi_12_l3198_319854

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end sin_cos_pi_12_l3198_319854


namespace adams_average_score_l3198_319848

/-- Given Adam's total score and number of rounds played, calculate the average points per round --/
theorem adams_average_score (total_score : ℕ) (num_rounds : ℕ) 
  (h1 : total_score = 283) (h2 : num_rounds = 4) :
  ∃ (avg : ℚ), avg = (total_score : ℚ) / (num_rounds : ℚ) ∧ 
  ∃ (rounded : ℕ), rounded = round avg ∧ rounded = 71 := by
  sorry

end adams_average_score_l3198_319848


namespace function_is_identity_l3198_319803

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (a : ℝ), f (f x - y) = f x + f (f y - f a) + x

theorem function_is_identity (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f x = x :=
sorry

end function_is_identity_l3198_319803


namespace ababab_divisible_by_seven_l3198_319845

/-- Given two digits a and b, the function forms the number ababab -/
def formNumber (a b : ℕ) : ℕ := 
  100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b

/-- Theorem stating that for any two digits, the number formed as ababab is divisible by 7 -/
theorem ababab_divisible_by_seven (a b : ℕ) (ha : a < 10) (hb : b < 10) : 
  7 ∣ formNumber a b := by
  sorry

#eval formNumber 2 3  -- To check if the function works correctly

end ababab_divisible_by_seven_l3198_319845


namespace greatest_integer_less_than_negative_fraction_l3198_319846

theorem greatest_integer_less_than_negative_fraction :
  ⌊-22/5⌋ = -5 := by sorry

end greatest_integer_less_than_negative_fraction_l3198_319846


namespace system_solution_l3198_319810

theorem system_solution :
  ∃ (x y z : ℝ),
    (x + y + z = 26) ∧
    (3 * x - 2 * y + z = 3) ∧
    (x - 4 * y - 2 * z = -13) ∧
    (x = -32.2) ∧
    (y = -13.8) ∧
    (z = 72) := by
  sorry

end system_solution_l3198_319810


namespace ratio_composition_l3198_319826

theorem ratio_composition (a b c : ℚ) 
  (hab : a / b = 4 / 3) 
  (hbc : b / c = 1 / 5) : 
  a / c = 4 / 15 := by
sorry

end ratio_composition_l3198_319826


namespace monomial_condition_and_expression_evaluation_l3198_319813

/-- Given that -2a^2 * b^(y+3) and 4a^x * b^2 form a monomial when added together,
    prove that x = 2 and y = -1, and that under these conditions,
    2(x^2*y - 3*y^3 + 2*x) - 3(x + x^2*y - 2*y^3) - x = 4 -/
theorem monomial_condition_and_expression_evaluation 
  (a b : ℝ) (x y : ℤ) 
  (h : ∃ k, -2 * a^2 * b^(y+3) + 4 * a^x * b^2 = k * a^2 * b^2) :
  x = 2 ∧ y = -1 ∧ 
  2 * (x^2 * y - 3 * y^3 + 2 * x) - 3 * (x + x^2 * y - 2 * y^3) - x = 4 :=
by sorry

end monomial_condition_and_expression_evaluation_l3198_319813


namespace two_valid_B_values_l3198_319836

/-- Represents a single digit (1 to 9) -/
def SingleDigit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- Converts a single digit B to the two-digit number B1 -/
def toTwoDigit (B : SingleDigit) : ℕ := B.val * 10 + 1

/-- Checks if the equation x^2 - (1B)x + B1 = 0 has positive integer solutions -/
def hasPositiveIntegerSolutions (B : SingleDigit) : Prop :=
  ∃ x : ℕ, x > 0 ∧ x^2 - (10 + B.val) * x + toTwoDigit B = 0

/-- The main theorem stating that exactly two single-digit B values satisfy the condition -/
theorem two_valid_B_values :
  ∃! (S : Finset SingleDigit), S.card = 2 ∧ ∀ B, B ∈ S ↔ hasPositiveIntegerSolutions B :=
sorry

end two_valid_B_values_l3198_319836


namespace mean_height_of_players_l3198_319859

def player_heights : List ℝ := [47, 48, 50, 50, 54, 55, 57, 59, 63, 63, 64, 65]

theorem mean_height_of_players : 
  (player_heights.sum / player_heights.length : ℝ) = 56.25 := by
  sorry

end mean_height_of_players_l3198_319859


namespace inequality_proof_l3198_319817

theorem inequality_proof (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (b - c)^2011 * (b + c)^2011 * (c - b)^2011 ≥ (b^2011 - c^2011) * (b^2011 + c^2011) * (c^2011 - b^2011) := by
  sorry

end inequality_proof_l3198_319817


namespace complex_sum_real_part_l3198_319805

theorem complex_sum_real_part (a b : ℝ) : 
  (1 + Complex.I) / Complex.I + (1 + Complex.I * Real.sqrt 3) ^ 2 = Complex.mk a b →
  a + b = 2 * Real.sqrt 3 - 2 :=
sorry

end complex_sum_real_part_l3198_319805


namespace matrix_sum_theorem_l3198_319847

def M (a b c d : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]]

theorem matrix_sum_theorem (a b c d : ℝ) :
  ¬(IsUnit (M a b c d).det) →
  (a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c) = 4 / 3) :=
by sorry

end matrix_sum_theorem_l3198_319847


namespace complex_exponential_to_rectangular_l3198_319887

theorem complex_exponential_to_rectangular : 
  ∃ (z : ℂ), z = Real.sqrt 2 * Complex.exp (Complex.I * (13 * Real.pi / 6)) ∧ 
             z = (Real.sqrt 6 / 2 : ℂ) + Complex.I * (Real.sqrt 2 / 2 : ℂ) := by
  sorry

end complex_exponential_to_rectangular_l3198_319887


namespace certain_number_problem_l3198_319885

theorem certain_number_problem (x y : ℝ) (hx : x = 4) (hy : y + y * x = 48) : y = 9.6 := by
  sorry

end certain_number_problem_l3198_319885


namespace least_number_divisible_by_3_4_5_7_8_l3198_319814

theorem least_number_divisible_by_3_4_5_7_8 : ∀ n : ℕ, n > 0 → (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (8 ∣ n) → n ≥ 840 := by
  sorry

#check least_number_divisible_by_3_4_5_7_8

end least_number_divisible_by_3_4_5_7_8_l3198_319814


namespace defective_from_factory1_l3198_319851

/-- The probability of a product coming from the first factory -/
def p_factory1 : ℝ := 0.20

/-- The probability of a product coming from the second factory -/
def p_factory2 : ℝ := 0.46

/-- The probability of a product coming from the third factory -/
def p_factory3 : ℝ := 0.34

/-- The probability of a defective item from the first factory -/
def p_defective1 : ℝ := 0.03

/-- The probability of a defective item from the second factory -/
def p_defective2 : ℝ := 0.02

/-- The probability of a defective item from the third factory -/
def p_defective3 : ℝ := 0.01

/-- The probability that a randomly selected defective item was produced at the first factory -/
theorem defective_from_factory1 : 
  (p_defective1 * p_factory1) / (p_defective1 * p_factory1 + p_defective2 * p_factory2 + p_defective3 * p_factory3) = 0.322 := by
sorry

end defective_from_factory1_l3198_319851


namespace lunks_needed_for_bananas_l3198_319807

/-- Exchange rate of lunks to kunks -/
def lunk_to_kunk_rate : ℚ := 2 / 3

/-- Exchange rate of kunks to bananas -/
def kunk_to_banana_rate : ℚ := 5 / 6

/-- Number of bananas to purchase -/
def bananas_to_buy : ℕ := 20

/-- Theorem stating the number of lunks needed to buy the specified number of bananas -/
theorem lunks_needed_for_bananas : 
  ⌈(bananas_to_buy : ℚ) / (kunk_to_banana_rate * lunk_to_kunk_rate)⌉ = 36 := by
  sorry


end lunks_needed_for_bananas_l3198_319807


namespace two_circles_distance_formula_l3198_319871

/-- Two circles with radii R and r, whose centers are at distance d apart,
    and whose common internal tangents define four points of tangency
    that form a quadrilateral circumscribed around a circle. -/
structure TwoCirclesConfig where
  R : ℝ
  r : ℝ
  d : ℝ

/-- The theorem stating the relationship between the radii and the distance between centers -/
theorem two_circles_distance_formula (config : TwoCirclesConfig) :
  config.d ^ 2 = (config.R + config.r) ^ 2 + 4 * config.R * config.r :=
sorry

end two_circles_distance_formula_l3198_319871


namespace rectangle_area_l3198_319864

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end rectangle_area_l3198_319864


namespace retirement_fund_increment_l3198_319874

theorem retirement_fund_increment (y k : ℝ) 
  (h1 : k * Real.sqrt (y + 3) = k * Real.sqrt y + 15)
  (h2 : k * Real.sqrt (y + 5) = k * Real.sqrt y + 27)
  : k * Real.sqrt y = 810 := by
  sorry

end retirement_fund_increment_l3198_319874


namespace solution_set_when_m_is_5_range_of_m_for_inequality_l3198_319815

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 6|

-- Part I
theorem solution_set_when_m_is_5 :
  {x : ℝ | f 5 x ≤ 12} = {x : ℝ | -13/2 ≤ x ∧ x ≤ 11/2} := by sorry

-- Part II
theorem range_of_m_for_inequality :
  {m : ℝ | ∀ x : ℝ, f m x ≥ 7} = {m : ℝ | m ≤ -13 ∨ m ≥ 1} := by sorry

end solution_set_when_m_is_5_range_of_m_for_inequality_l3198_319815


namespace exists_x_in_interval_l3198_319858

theorem exists_x_in_interval (x : ℝ) : 
  ∃ x, x ∈ Set.Icc (-1 : ℝ) (3/10) ∧ x^2 + 3*x - 1 ≤ 0 := by
  sorry

end exists_x_in_interval_l3198_319858


namespace ellipse_slope_bound_l3198_319875

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    and points A(-a, 0), B(a, 0), and P(x, y) on the ellipse such that
    P ≠ A, P ≠ B, and |AP| = |OA|, prove that the absolute value of
    the slope of line OP is greater than √3. -/
theorem ellipse_slope_bound (a b x y : ℝ) :
  a > b ∧ b > 0 ∧
  x^2 / a^2 + y^2 / b^2 = 1 ∧
  (x ≠ -a ∨ y ≠ 0) ∧ (x ≠ a ∨ y ≠ 0) ∧
  (x + a)^2 + y^2 = 4 * a^2 →
  abs (y / x) > Real.sqrt 3 :=
by sorry

end ellipse_slope_bound_l3198_319875


namespace angle_double_supplement_is_120_l3198_319862

-- Define the angle measure
def angle_measure : ℝ → Prop := λ x => 
  -- The angle measure is double its supplement
  x = 2 * (180 - x) ∧ 
  -- The angle measure is positive and less than or equal to 180
  0 < x ∧ x ≤ 180

-- Theorem statement
theorem angle_double_supplement_is_120 : 
  ∃ x : ℝ, angle_measure x ∧ x = 120 :=
sorry

end angle_double_supplement_is_120_l3198_319862


namespace y_intercept_of_line_l3198_319890

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x - 5 * y = 10

-- Define y-intercept
def is_y_intercept (y : ℝ) : Prop :=
  line_equation 0 y

-- Theorem statement
theorem y_intercept_of_line :
  is_y_intercept (-2) :=
sorry

end y_intercept_of_line_l3198_319890


namespace markup_percentage_of_selling_price_l3198_319895

theorem markup_percentage_of_selling_price 
  (cost selling_price markup : ℝ) 
  (h1 : markup = 0.1 * cost) 
  (h2 : selling_price = cost + markup) :
  markup / selling_price = 100 / 11 / 100 := by
sorry

end markup_percentage_of_selling_price_l3198_319895


namespace recurring_decimal_sum_diff_main_theorem_l3198_319801

/-- Represents a recurring decimal of the form 0.nnn... where n is a single digit -/
def recurring_decimal (n : ℕ) : ℚ := n / 9

theorem recurring_decimal_sum_diff (a b c : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) :
  recurring_decimal a + recurring_decimal b - recurring_decimal c = (a + b - c : ℚ) / 9 := by sorry

theorem main_theorem : 
  recurring_decimal 6 + recurring_decimal 2 - recurring_decimal 4 = 4 / 9 := by sorry

end recurring_decimal_sum_diff_main_theorem_l3198_319801


namespace identify_fake_bag_l3198_319809

/-- Represents a bag of coins -/
structure CoinBag where
  id : Nat
  isFake : Bool

/-- Represents the collection of all coin bags -/
def allBags : Finset CoinBag := sorry

/-- The weight of a real coin in grams -/
def realCoinWeight : Nat := 10

/-- The weight of a fake coin in grams -/
def fakeCoinWeight : Nat := 9

/-- The number of bags -/
def numBags : Nat := 10

/-- The expected total weight if all coins were real -/
def expectedTotalWeight : Nat := 550

/-- Calculates the weight of coins taken from a bag -/
def bagWeight (bag : CoinBag) : Nat :=
  if bag.isFake then
    bag.id * fakeCoinWeight
  else
    bag.id * realCoinWeight

/-- Calculates the total weight of all selected coins -/
def totalWeight : Nat := (allBags.sum bagWeight)

/-- Theorem stating that the bag number with fake coins is equal to the difference
    between the expected total weight and the actual total weight -/
theorem identify_fake_bag :
  ∃ (fakeBag : CoinBag), fakeBag ∈ allBags ∧ fakeBag.isFake ∧
    fakeBag.id = expectedTotalWeight - totalWeight := by sorry

end identify_fake_bag_l3198_319809


namespace hyperbola_directrices_distance_l3198_319880

/-- Given a hyperbola with foci at (±√26, 0) and asymptotes y = ±(3/2)x,
    prove that the distance between its two directrices is (8√26)/13 -/
theorem hyperbola_directrices_distance (a b c : ℝ) : 
  (c = Real.sqrt 26) →                  -- focus distance
  (b / a = 3 / 2) →                     -- asymptote slope
  (a^2 + b^2 = 26) →                    -- relation between a, b, and c
  (2 * (a^2 / c)) = (8 * Real.sqrt 26) / 13 := by
  sorry

#check hyperbola_directrices_distance

end hyperbola_directrices_distance_l3198_319880


namespace janet_waiting_time_l3198_319892

/-- Proves that Janet waits 3 hours for her sister to cross the lake -/
theorem janet_waiting_time (lake_width : ℝ) (janet_speed : ℝ) (sister_speed : ℝ) :
  lake_width = 60 →
  janet_speed = 30 →
  sister_speed = 12 →
  (lake_width / sister_speed) - (lake_width / janet_speed) = 3 := by
  sorry

end janet_waiting_time_l3198_319892


namespace ten_point_circle_triangles_l3198_319819

/-- Represents a circle with points and chords -/
structure CircleWithChords where
  numPoints : ℕ
  noTripleIntersection : Bool

/-- Calculates the number of triangles formed inside the circle -/
def trianglesInsideCircle (c : CircleWithChords) : ℕ :=
  sorry

/-- The main theorem stating that for 10 points on a circle with the given conditions,
    the number of triangles formed inside is 105 -/
theorem ten_point_circle_triangles :
  ∀ (c : CircleWithChords),
    c.numPoints = 10 →
    c.noTripleIntersection = true →
    trianglesInsideCircle c = 105 :=
by sorry

end ten_point_circle_triangles_l3198_319819


namespace subset_union_inclusion_fraction_inequality_ac_squared_eq_bc_squared_not_sufficient_negation_of_all_positive_real_l3198_319812

-- 1. Set inclusion property
theorem subset_union_inclusion (M N : Set α) : M ⊆ N → M ⊆ (M ∪ N) := by sorry

-- 2. Fraction inequality
theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  (b + m) / (a + m) > b / a := by sorry

-- 3. Counterexample for ac² = bc² implying a = b
theorem ac_squared_eq_bc_squared_not_sufficient :
  ∃ (a b c : ℝ), a * c^2 = b * c^2 ∧ a ≠ b := by sorry

-- 4. Negation of universal quantifier
theorem negation_of_all_positive_real :
  ¬(∀ (x : ℝ), x > 0) ≠ (∃ (x : ℝ), x < 0) := by sorry

end subset_union_inclusion_fraction_inequality_ac_squared_eq_bc_squared_not_sufficient_negation_of_all_positive_real_l3198_319812


namespace eight_bead_bracelet_arrangements_l3198_319844

/-- The number of distinct ways to arrange n distinct beads on a bracelet, 
    considering rotations and reflections as equivalent -/
def bracelet_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of distinct arrangements 
    for 8 beads is 2520 -/
theorem eight_bead_bracelet_arrangements : 
  bracelet_arrangements 8 = 2520 := by
  sorry

end eight_bead_bracelet_arrangements_l3198_319844


namespace factorization_implies_k_value_l3198_319822

theorem factorization_implies_k_value (k : ℝ) :
  (∃ (a b c d e f : ℝ), ∀ x y : ℝ,
    x^3 + 3*x^2 - 2*x*y - k*x - 4*y = (a*x + b*y + c) * (d*x^2 + e*x*y + f*y)) →
  k = -2 :=
by sorry

end factorization_implies_k_value_l3198_319822


namespace lemons_for_twenty_gallons_l3198_319896

/-- Calculates the number of lemons needed for a given volume of lemonade -/
def lemons_needed (base_lemons : ℕ) (base_gallons : ℕ) (target_gallons : ℕ) : ℕ :=
  let base_ratio := base_lemons / base_gallons
  let base_lemons_needed := base_ratio * target_gallons
  let additional_lemons := target_gallons / 10
  base_lemons_needed + additional_lemons

theorem lemons_for_twenty_gallons :
  lemons_needed 40 50 20 = 18 := by
  sorry

#eval lemons_needed 40 50 20

end lemons_for_twenty_gallons_l3198_319896


namespace roots_sum_and_product_l3198_319816

theorem roots_sum_and_product (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∀ x, x^2 - p*x - 2*q = 0 ↔ x = p ∨ x = q) :
  p + q = p ∧ p * q = -2*q := by
  sorry

end roots_sum_and_product_l3198_319816


namespace min_value_reciprocal_sum_l3198_319868

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  1 / x + 4 / y ≥ 9 / 4 := by
  sorry

end min_value_reciprocal_sum_l3198_319868


namespace wedding_decoration_cost_l3198_319833

/-- Calculates the total cost of decorations for Nathan's wedding reception --/
def total_decoration_cost (num_tables : ℕ) 
  (tablecloth_cost service_charge place_setting_cost place_settings_per_table : ℝ)
  (roses_per_centerpiece rose_cost rose_discount : ℝ)
  (lilies_per_centerpiece lily_cost lily_discount : ℝ)
  (daisies_per_centerpiece daisy_cost : ℝ)
  (sunflowers_per_centerpiece sunflower_cost : ℝ)
  (lighting_cost : ℝ) : ℝ :=
  let tablecloth_total := num_tables * tablecloth_cost * (1 + service_charge)
  let place_settings_total := num_tables * place_settings_per_table * place_setting_cost
  let centerpiece_cost := 
    (roses_per_centerpiece * rose_cost * (1 - rose_discount)) +
    (lilies_per_centerpiece * lily_cost * (1 - lily_discount)) +
    (daisies_per_centerpiece * daisy_cost) +
    (sunflowers_per_centerpiece * sunflower_cost)
  let centerpiece_total := num_tables * centerpiece_cost
  tablecloth_total + place_settings_total + centerpiece_total + lighting_cost

/-- Theorem stating the total cost of decorations for Nathan's wedding reception --/
theorem wedding_decoration_cost : 
  total_decoration_cost 30 25 0.15 12 6 15 6 0.1 20 5 0.05 5 3 3 4 450 = 9562.50 := by
  sorry

end wedding_decoration_cost_l3198_319833


namespace odot_computation_l3198_319855

def odot (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem odot_computation : odot 2 (odot 3 (odot 4 5)) = 7/8 := by
  sorry

end odot_computation_l3198_319855


namespace probability_both_odd_l3198_319865

def m : ℕ := 7
def n : ℕ := 9

def is_odd (k : ℕ) : Prop := k % 2 = 1

def count_odd (k : ℕ) : ℕ := (k + 1) / 2

theorem probability_both_odd : 
  (count_odd m * count_odd n : ℚ) / (m * n : ℚ) = 20 / 63 := by sorry

end probability_both_odd_l3198_319865


namespace line_properties_l3198_319869

/-- A line in the 2D plane represented by the equation y = k(x-1) --/
structure Line where
  k : ℝ

/-- The point (1,0) in the 2D plane --/
def point : ℝ × ℝ := (1, 0)

/-- Checks if a given line passes through the point (1,0) --/
def passes_through_point (l : Line) : Prop :=
  0 = l.k * (point.1 - 1)

/-- Checks if a given line is not perpendicular to the x-axis --/
def not_perpendicular_to_x_axis (l : Line) : Prop :=
  l.k ≠ 0

/-- Theorem stating that all lines represented by y = k(x-1) pass through (1,0) and are not perpendicular to the x-axis --/
theorem line_properties (l : Line) : 
  passes_through_point l ∧ not_perpendicular_to_x_axis l :=
sorry

end line_properties_l3198_319869


namespace v_sum_zero_l3198_319806

noncomputable def v (x : ℝ) : ℝ := -x + (3/2) * Real.sin (x * Real.pi / 2)

theorem v_sum_zero : v (-3.14) + v (-1) + v 1 + v 3.14 = 0 := by sorry

end v_sum_zero_l3198_319806


namespace tangent_line_distance_l3198_319856

/-- A line with slope 1 is tangent to y = e^x and y^2 = 4x at two different points. -/
theorem tangent_line_distance : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  -- The line is tangent to y = e^x at (x₁, y₁)
  (Real.exp x₁ = y₁) ∧ 
  (Real.exp x₁ = 1) ∧
  -- The line is tangent to y^2 = 4x at (x₂, y₂)
  (y₂^2 = 4 * x₂) ∧
  (y₂ = 2 * Real.sqrt x₂) ∧
  -- Both points lie on a line with slope 1
  (y₂ - y₁ = x₂ - x₁) ∧
  -- The distance between the two points is √2
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = Real.sqrt 2 :=
by sorry


end tangent_line_distance_l3198_319856


namespace furniture_reimbursement_l3198_319843

/-- Calculates the reimbursement amount for an overcharged furniture purchase -/
theorem furniture_reimbursement
  (total_paid : ℕ)
  (num_pieces : ℕ)
  (cost_per_piece : ℕ)
  (h1 : total_paid = 20700)
  (h2 : num_pieces = 150)
  (h3 : cost_per_piece = 134) :
  total_paid - (num_pieces * cost_per_piece) = 600 := by
sorry

end furniture_reimbursement_l3198_319843


namespace square_land_side_length_l3198_319824

theorem square_land_side_length 
  (area : ℝ) 
  (h : area = Real.sqrt 100) : 
  ∃ (side : ℝ), side * side = area ∧ side = 10 :=
sorry

end square_land_side_length_l3198_319824


namespace calc_complex_fraction_l3198_319853

theorem calc_complex_fraction : (3^2 * 5^4 * 7^2) / 7 = 39375 := by
  sorry

end calc_complex_fraction_l3198_319853


namespace mike_picked_seven_apples_l3198_319825

/-- The number of apples Mike picked -/
def mike_apples : ℝ := 7.0

/-- The number of apples Nancy ate -/
def nancy_ate : ℝ := 3.0

/-- The number of apples Keith picked -/
def keith_picked : ℝ := 6.0

/-- The number of apples left -/
def apples_left : ℝ := 10.0

/-- Theorem stating that Mike picked 7.0 apples -/
theorem mike_picked_seven_apples : 
  mike_apples = mike_apples - nancy_ate + keith_picked - apples_left + apples_left :=
by sorry

end mike_picked_seven_apples_l3198_319825


namespace line_through_point_and_circle_center_l3198_319873

/-- A line passing through two points on a plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in a plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The theorem stating that the equation of a line passing through a given point and the center of a given circle is x-2=0. -/
theorem line_through_point_and_circle_center 
  (M : ℝ × ℝ) 
  (C : Circle) 
  (h1 : M.1 = 2 ∧ M.2 = 3) 
  (h2 : C.center = (2, -3)) 
  (h3 : C.radius = 3) : 
  ∃ (l : Line), l.a = 1 ∧ l.b = 0 ∧ l.c = -2 :=
sorry

end line_through_point_and_circle_center_l3198_319873


namespace square_sum_theorem_l3198_319872

theorem square_sum_theorem (x y : ℝ) (h1 : x + 2*y = 4) (h2 : x*y = -8) : 
  x^2 + 4*y^2 = 48 := by
sorry

end square_sum_theorem_l3198_319872


namespace set_operations_and_subset_condition_l3198_319861

def A : Set ℝ := {x | -4 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

theorem set_operations_and_subset_condition :
  (∀ x, x ∈ (A ∪ B 1) ↔ -4 < x ∧ x ≤ 3) ∧
  (∀ x, x ∈ (A ∩ (Set.univ \ B 1)) ↔ -4 < x ∧ x < 0) ∧
  (∀ a, B a ⊆ A ↔ -3 < a ∧ a < -1) :=
by sorry

end set_operations_and_subset_condition_l3198_319861


namespace max_sum_given_constraints_l3198_319849

theorem max_sum_given_constraints (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) :
  x + y ≤ 6 * Real.sqrt 5 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 100 ∧ x₀ * y₀ = 40 ∧ x₀ + y₀ = 6 * Real.sqrt 5 := by
  sorry

end max_sum_given_constraints_l3198_319849


namespace kelly_initial_games_l3198_319821

/-- The number of Nintendo games Kelly gave away -/
def games_given_away : ℕ := 64

/-- The number of Nintendo games Kelly has left -/
def games_left : ℕ := 42

/-- The initial number of Nintendo games Kelly had -/
def initial_games : ℕ := games_given_away + games_left

theorem kelly_initial_games : initial_games = 106 := by
  sorry

end kelly_initial_games_l3198_319821


namespace magical_stack_with_201_fixed_l3198_319878

/-- Definition of a magical stack of cards -/
def is_magical_stack (n : ℕ) : Prop :=
  ∃ (card_from_A card_from_B : ℕ), 
    card_from_A ≤ n ∧ 
    card_from_B > n ∧ 
    card_from_B ≤ 2*n ∧
    (card_from_A = 2 * ((card_from_A + 1) / 2) - 1 ∨
     card_from_B = 2 * (card_from_B / 2))

/-- Theorem stating the number of cards in a magical stack where card 201 retains its position -/
theorem magical_stack_with_201_fixed :
  ∃ (n : ℕ), 
    is_magical_stack n ∧ 
    201 ≤ n ∧
    201 = 2 * ((201 + 1) / 2) - 1 ∧
    n = 201 ∧
    2 * n = 402 := by
  sorry

end magical_stack_with_201_fixed_l3198_319878


namespace square_remainders_l3198_319894

theorem square_remainders (n : ℤ) : 
  (∃ r : ℤ, r ∈ ({0, 1} : Set ℤ) ∧ n^2 ≡ r [ZMOD 3]) ∧
  (∃ r : ℤ, r ∈ ({0, 1} : Set ℤ) ∧ n^2 ≡ r [ZMOD 4]) ∧
  (∃ r : ℤ, r ∈ ({0, 1, 4} : Set ℤ) ∧ n^2 ≡ r [ZMOD 5]) ∧
  (∃ r : ℤ, r ∈ ({0, 1, 4} : Set ℤ) ∧ n^2 ≡ r [ZMOD 8]) :=
by sorry

end square_remainders_l3198_319894


namespace equation_solution_l3198_319867

theorem equation_solution (x : ℚ) : 
  (5 * x - 3) / (6 * x - 12) = 4 / 3 → x = 13 / 3 := by
  sorry

end equation_solution_l3198_319867


namespace cubic_equation_root_l3198_319893

theorem cubic_equation_root (a b : ℚ) : 
  (1 + Real.sqrt 5)^3 + a * (1 + Real.sqrt 5)^2 + b * (1 + Real.sqrt 5) - 60 = 0 → 
  b = 26 := by
sorry

end cubic_equation_root_l3198_319893


namespace right_triangle_sets_l3198_319828

/-- A function to check if three line segments can form a right triangle -/
def isRightTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- Theorem stating that among the given sets, only {2, √2, √2} forms a right triangle -/
theorem right_triangle_sets :
  ¬ isRightTriangle 2 3 4 ∧
  ¬ isRightTriangle 1 1 2 ∧
  ¬ isRightTriangle 4 5 6 ∧
  isRightTriangle 2 (Real.sqrt 2) (Real.sqrt 2) :=
by sorry

end right_triangle_sets_l3198_319828


namespace sum_reciprocal_product_20_l3198_319886

/-- Given a sequence {a_n} where the sum of its first n terms is S_n = 6n - n^2,
    this function returns the sum of the first k terms of the sequence {1/(a_n * a_{n+1})} -/
def sum_reciprocal_product (k : ℕ) : ℚ :=
  let S : ℕ → ℚ := λ n => 6 * n - n^2
  let a : ℕ → ℚ := λ n => S n - S (n-1)
  let term : ℕ → ℚ := λ n => 1 / (a n * a (n+1))
  (Finset.range k).sum term

/-- The main theorem stating that the sum of the first 20 terms of the 
    sequence {1/(a_n * a_{n+1})} is equal to -4/35 -/
theorem sum_reciprocal_product_20 : sum_reciprocal_product 20 = -4/35 := by
  sorry

end sum_reciprocal_product_20_l3198_319886


namespace swimming_pool_volume_l3198_319820

/-- A swimming pool with trapezoidal cross-section -/
structure SwimmingPool where
  width : ℝ
  length : ℝ
  shallow_depth : ℝ
  deep_depth : ℝ

/-- Calculate the volume of a swimming pool with trapezoidal cross-section -/
def pool_volume (pool : SwimmingPool) : ℝ :=
  0.5 * (pool.shallow_depth + pool.deep_depth) * pool.width * pool.length

/-- Theorem stating that the volume of the given swimming pool is 270 cubic meters -/
theorem swimming_pool_volume :
  let pool : SwimmingPool := {
    width := 9,
    length := 12,
    shallow_depth := 1,
    deep_depth := 4
  }
  pool_volume pool = 270 := by sorry

end swimming_pool_volume_l3198_319820


namespace card_arrangement_probability_l3198_319834

theorem card_arrangement_probability : 
  let n : ℕ := 8  -- total number of cards
  let k : ℕ := 3  -- number of identical cards (О in this case)
  let total_permutations : ℕ := n.factorial
  let favorable_permutations : ℕ := k.factorial
  (favorable_permutations : ℚ) / total_permutations = 1 / 6720 :=
by sorry

end card_arrangement_probability_l3198_319834


namespace fraction_simplification_l3198_319889

theorem fraction_simplification (x : ℝ) : (2*x + 3)/4 + (5 - 4*x)/3 = (-10*x + 29)/12 := by
  sorry

end fraction_simplification_l3198_319889


namespace sports_love_distribution_l3198_319829

theorem sports_love_distribution (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (boys_not_love_sports : ℕ) (total_not_love_sports : ℕ) :
  total_students = 50 →
  total_boys = 30 →
  total_girls = 20 →
  boys_not_love_sports = 12 →
  total_not_love_sports = 24 →
  ∃ (boys_love_sports : ℕ) (total_love_sports : ℕ),
    boys_love_sports = total_boys - boys_not_love_sports ∧
    total_love_sports = total_students - total_not_love_sports ∧
    boys_love_sports = 18 ∧
    total_love_sports = 26 := by
  sorry

end sports_love_distribution_l3198_319829


namespace shared_root_quadratic_equation_l3198_319827

theorem shared_root_quadratic_equation (a b p q : ℝ) (h : a ≠ p ∧ b ≠ q) :
  ∃ (α β γ : ℝ),
    (α^2 + a*α + b = 0 ∧ α^2 + p*α + q = 0) →
    (β^2 + a*β + b = 0 ∧ β ≠ α) →
    (γ^2 + p*γ + q = 0 ∧ γ ≠ α) →
    (x^2 - (-p - (b - q)/(p - a))*x + (b*q*(p - a)^2)/(b - q)^2 = (x - β)*(x - γ)) := by
  sorry

end shared_root_quadratic_equation_l3198_319827


namespace area_under_curve_l3198_319850

-- Define the curve
def curve (x : ℝ) : ℝ := 3 * x^2

-- Define the bounds of the region
def lower_bound : ℝ := 0
def upper_bound : ℝ := 1

-- Theorem statement
theorem area_under_curve :
  ∫ x in lower_bound..upper_bound, curve x = 1 := by sorry

end area_under_curve_l3198_319850


namespace bus_fare_payment_possible_l3198_319888

/-- Represents a person with their initial money and final payment -/
structure Person where
  initial_money : ℕ
  final_payment : ℕ

/-- Represents the bus fare payment scenario -/
def BusFareScenario (fare : ℕ) (people : List Person) : Prop :=
  (people.length = 3) ∧
  (∀ p ∈ people, p.final_payment = fare) ∧
  (∃ total : ℕ, total = people.foldl (λ sum person => sum + person.initial_money) 0) ∧
  (∃ payer : Person, payer ∈ people ∧ payer.initial_money ≥ 3 * fare)

/-- Theorem stating that it's possible to pay the bus fare -/
theorem bus_fare_payment_possible (fare : ℕ) (people : List Person) 
  (h : BusFareScenario fare people) : 
  ∃ (final_money : List ℕ), 
    final_money.length = people.length ∧ 
    final_money.sum = people.foldl (λ sum person => sum + person.initial_money) 0 - 3 * fare :=
sorry

end bus_fare_payment_possible_l3198_319888


namespace greatest_gcd_value_l3198_319841

def S (n : ℕ+) : ℕ := n^2

theorem greatest_gcd_value (n : ℕ+) :
  (∃ m : ℕ+, Nat.gcd (2 * S m + 10 * m) (m - 3) = 42) ∧
  (∀ k : ℕ+, Nat.gcd (2 * S k + 10 * k) (k - 3) ≤ 42) :=
sorry

end greatest_gcd_value_l3198_319841


namespace investment_rate_proof_l3198_319879

/-- Proves that the unknown investment rate is 0.18 given the problem conditions --/
theorem investment_rate_proof (total_investment : ℝ) (known_rate : ℝ) (total_interest : ℝ) (unknown_rate_investment : ℝ) 
  (h1 : total_investment = 22000)
  (h2 : known_rate = 0.14)
  (h3 : total_interest = 3360)
  (h4 : unknown_rate_investment = 7000)
  (h5 : unknown_rate_investment * r + (total_investment - unknown_rate_investment) * known_rate = total_interest) :
  r = 0.18 := by
  sorry

end investment_rate_proof_l3198_319879


namespace total_money_available_l3198_319802

/-- Represents the cost of a single gumdrop in cents -/
def cost_per_gumdrop : ℕ := 4

/-- Represents the number of gumdrops that can be purchased -/
def num_gumdrops : ℕ := 20

/-- Theorem stating that the total amount of money available is 80 cents -/
theorem total_money_available : cost_per_gumdrop * num_gumdrops = 80 := by
  sorry

end total_money_available_l3198_319802


namespace min_value_and_integer_solutions_l3198_319876

theorem min_value_and_integer_solutions (x y : ℝ) : 
  x + y + 2*x*y = 5 →
  (∀ (x y : ℝ), x > 0 ∧ y > 0 → x + y ≥ Real.sqrt 11 - 1) ∧
  (∃ (x y : ℤ), x + y + 2*x*y = 5 ∧ (x + y = 5 ∨ x + y = -7)) := by
  sorry

end min_value_and_integer_solutions_l3198_319876


namespace f_inequality_l3198_319831

/-- The number of ways to represent a positive integer as a sum of non-decreasing positive integers -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating that f(n+1) is less than or equal to the average of f(n) and f(n+2) for any positive integer n -/
theorem f_inequality (n : ℕ) (h : n > 0) : f (n + 1) ≤ (f n + f (n + 2)) / 2 := by sorry

end f_inequality_l3198_319831


namespace gravel_weight_in_specific_mixture_l3198_319830

/-- A cement mixture with sand, water, and gravel -/
structure CementMixture where
  total_weight : ℝ
  sand_fraction : ℝ
  water_fraction : ℝ

/-- Calculate the weight of gravel in a cement mixture -/
def gravel_weight (m : CementMixture) : ℝ :=
  m.total_weight * (1 - m.sand_fraction - m.water_fraction)

/-- Theorem stating the weight of gravel in the specific mixture -/
theorem gravel_weight_in_specific_mixture :
  let m : CementMixture := {
    total_weight := 120,
    sand_fraction := 1/5,
    water_fraction := 3/4
  }
  gravel_weight m = 6 := by
  sorry

end gravel_weight_in_specific_mixture_l3198_319830


namespace saree_original_price_l3198_319818

/-- The original price of sarees given successive discounts -/
theorem saree_original_price (final_price : ℝ) 
  (h1 : final_price = 380.16) 
  (h2 : final_price = 0.9 * 0.8 * original_price) : 
  original_price = 528 :=
by
  sorry

#check saree_original_price

end saree_original_price_l3198_319818


namespace max_profit_theorem_l3198_319837

/-- Represents the store's lamp purchasing problem -/
structure LampProblem where
  cost_diff : ℕ  -- Cost difference between type A and B lamps
  budget_A : ℕ   -- Budget for type A lamps
  budget_B : ℕ   -- Budget for type B lamps
  total_lamps : ℕ -- Total number of lamps to purchase
  max_budget : ℕ  -- Maximum total budget
  price_A : ℕ    -- Selling price of type A lamp
  price_B : ℕ    -- Selling price of type B lamp

/-- Calculates the maximum profit for the given LampProblem -/
def max_profit (p : LampProblem) : ℕ :=
  let cost_A := p.budget_A * 2 / 5  -- Cost of type A lamp
  let cost_B := cost_A - p.cost_diff -- Cost of type B lamp
  let max_A := (p.max_budget - cost_B * p.total_lamps) / (cost_A - cost_B)
  let profit := (p.price_A - cost_A) * max_A + (p.price_B - cost_B) * (p.total_lamps - max_A)
  profit

/-- Theorem stating the maximum profit for the given problem -/
theorem max_profit_theorem (p : LampProblem) : 
  p.cost_diff = 40 ∧ 
  p.budget_A = 2000 ∧ 
  p.budget_B = 1600 ∧ 
  p.total_lamps = 80 ∧ 
  p.max_budget = 14550 ∧ 
  p.price_A = 300 ∧ 
  p.price_B = 200 →
  max_profit p = 5780 ∧ 
  (p.max_budget - 160 * p.total_lamps) / 40 = 43 :=
by sorry


end max_profit_theorem_l3198_319837


namespace monotone_increasing_condition_l3198_319808

-- Define the function f(x) = x² - mx + 1
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 1

-- Define the property of being monotonically increasing on an interval
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- State the theorem
theorem monotone_increasing_condition (m : ℝ) :
  MonotonicallyIncreasing (f m) 3 8 → m ≤ 6 ∨ m ≥ 16 := by
  sorry

end monotone_increasing_condition_l3198_319808


namespace sugar_solution_concentration_l3198_319852

theorem sugar_solution_concentration (W : ℝ) (X : ℝ) : 
  W > 0 → -- W is positive (total weight of solution)
  0.08 * W = 0.08 * W - 0.02 * W + X * W / 400 → -- Sugar balance equation
  0.16 * W = 0.06 * W + X * W / 400 → -- Final concentration equation
  X = 40 := by
sorry

end sugar_solution_concentration_l3198_319852


namespace cycle_gain_percent_l3198_319884

theorem cycle_gain_percent (cost_price selling_price : ℚ) (h1 : cost_price = 900) (h2 : selling_price = 1100) :
  (selling_price - cost_price) / cost_price * 100 = (2 : ℚ) / 9 * 100 := by
  sorry

end cycle_gain_percent_l3198_319884


namespace sandwich_cost_l3198_319857

theorem sandwich_cost (N B J : ℕ) (h1 : N > 1) (h2 : B > 0) (h3 : J > 0)
  (h4 : N * (3 * B + 6 * J) = 306) : 6 * N * J = 288 := by
  sorry

end sandwich_cost_l3198_319857
