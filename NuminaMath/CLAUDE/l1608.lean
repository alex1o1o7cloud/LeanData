import Mathlib

namespace weighted_average_calculation_l1608_160868

/-- Calculates the weighted average of exam scores for a class -/
theorem weighted_average_calculation (total_students : ℕ) 
  (math_perfect_scores math_zero_scores : ℕ)
  (science_perfect_scores science_zero_scores : ℕ)
  (math_average_rest science_average_rest : ℚ)
  (math_weight science_weight : ℚ) :
  total_students = 30 →
  math_perfect_scores = 3 →
  math_zero_scores = 4 →
  math_average_rest = 50 →
  science_perfect_scores = 2 →
  science_zero_scores = 5 →
  science_average_rest = 60 →
  math_weight = 2/5 →
  science_weight = 3/5 →
  (((math_perfect_scores * 100 + 
    (total_students - math_perfect_scores - math_zero_scores) * math_average_rest) * math_weight +
   (science_perfect_scores * 100 + 
    (total_students - science_perfect_scores - science_zero_scores) * science_average_rest) * science_weight) / total_students) = 1528/30 :=
by sorry

end weighted_average_calculation_l1608_160868


namespace total_height_calculation_l1608_160866

-- Define the heights in inches
def sculpture_height_inches : ℚ := 34
def base_height_inches : ℚ := 2

-- Define the conversion factor from inches to centimeters
def inches_to_cm : ℚ := 2.54

-- Define the total height in centimeters
def total_height_cm : ℚ := (sculpture_height_inches + base_height_inches) * inches_to_cm

-- Theorem statement
theorem total_height_calculation :
  total_height_cm = 91.44 := by sorry

end total_height_calculation_l1608_160866


namespace pelican_shark_ratio_l1608_160818

/-- Given that one-third of the Pelicans in Shark Bite Cove moved away, 
    20 Pelicans remain in Shark Bite Cove, and there are 60 sharks in Pelican Bay, 
    prove that the ratio of sharks in Pelican Bay to the original number of 
    Pelicans in Shark Bite Cove is 2:1. -/
theorem pelican_shark_ratio 
  (remaining_pelicans : ℕ) 
  (sharks : ℕ) 
  (h1 : remaining_pelicans = 20)
  (h2 : sharks = 60)
  (h3 : remaining_pelicans = (2/3 : ℚ) * (remaining_pelicans + remaining_pelicans / 2)) :
  (sharks : ℚ) / (remaining_pelicans + remaining_pelicans / 2) = 2 := by
  sorry

#check pelican_shark_ratio

end pelican_shark_ratio_l1608_160818


namespace bill_bathroom_visits_l1608_160872

/-- The number of times Bill goes to the bathroom daily -/
def bathroom_visits : ℕ := 3

/-- The number of squares of toilet paper Bill uses per bathroom visit -/
def squares_per_visit : ℕ := 5

/-- The number of rolls of toilet paper Bill has -/
def total_rolls : ℕ := 1000

/-- The number of squares of toilet paper per roll -/
def squares_per_roll : ℕ := 300

/-- The number of days Bill's toilet paper supply will last -/
def supply_duration : ℕ := 20000

theorem bill_bathroom_visits :
  bathroom_visits * squares_per_visit * supply_duration = total_rolls * squares_per_roll := by
  sorry

end bill_bathroom_visits_l1608_160872


namespace calculation_result_l1608_160813

theorem calculation_result : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |0.00067 * 0.338 - (75 * 0.00000102 / 0.00338 * 0.042) - 0.0008| < ε :=
sorry

end calculation_result_l1608_160813


namespace negative_one_third_squared_l1608_160870

theorem negative_one_third_squared : (-1/3 : ℚ)^2 = 1/9 := by
  sorry

end negative_one_third_squared_l1608_160870


namespace jane_mean_score_l1608_160865

def jane_scores : List ℕ := [95, 88, 94, 86, 92, 91]

theorem jane_mean_score :
  (jane_scores.sum / jane_scores.length : ℚ) = 91 := by sorry

end jane_mean_score_l1608_160865


namespace balloon_permutations_count_l1608_160839

/-- The number of distinct permutations of the letters in "BALLOON" -/
def balloon_permutations : ℕ := 1260

/-- The total number of letters in "BALLOON" -/
def total_letters : ℕ := 7

/-- The number of repeated 'L's in "BALLOON" -/
def repeated_L : ℕ := 2

/-- The number of repeated 'O's in "BALLOON" -/
def repeated_O : ℕ := 2

/-- Theorem stating that the number of distinct permutations of the letters in "BALLOON" is 1260 -/
theorem balloon_permutations_count :
  balloon_permutations = Nat.factorial total_letters / (Nat.factorial repeated_L * Nat.factorial repeated_O) :=
by sorry

end balloon_permutations_count_l1608_160839


namespace factorization_equality_l1608_160833

theorem factorization_equality (x y : ℝ) : 1 - 2*(x - y) + (x - y)^2 = (1 - x + y)^2 := by
  sorry

end factorization_equality_l1608_160833


namespace stating_count_testing_methods_proof_l1608_160888

/-- The number of different products -/
def total_products : ℕ := 7

/-- The number of defective products -/
def defective_products : ℕ := 4

/-- The number of non-defective products -/
def non_defective_products : ℕ := 3

/-- The test number on which the third defective product is identified -/
def third_defective_test : ℕ := 4

/-- 
  The number of testing methods where the third defective product 
  is exactly identified on the 4th test, given 7 total products 
  with 4 defective and 3 non-defective ones.
-/
def count_testing_methods : ℕ := 1080

/-- 
  Theorem stating that the number of testing methods where the third defective product 
  is exactly identified on the 4th test is equal to 1080, given the problem conditions.
-/
theorem count_testing_methods_proof : 
  count_testing_methods = 1080 ∧
  total_products = 7 ∧
  defective_products = 4 ∧
  non_defective_products = 3 ∧
  third_defective_test = 4 :=
by sorry

end stating_count_testing_methods_proof_l1608_160888


namespace christen_peeled_twenty_l1608_160898

/-- The number of potatoes Christen peeled --/
def christenPotatoes (initialPile : ℕ) (homerRate christenRate : ℕ) (timeBeforeJoin : ℕ) : ℕ :=
  let homerPotatoes := homerRate * timeBeforeJoin
  let remainingPotatoes := initialPile - homerPotatoes
  let combinedRate := homerRate + christenRate
  let timeAfterJoin := remainingPotatoes / combinedRate
  christenRate * timeAfterJoin

/-- Theorem stating that Christen peeled 20 potatoes --/
theorem christen_peeled_twenty :
  christenPotatoes 60 4 5 6 = 20 := by
  sorry

#eval christenPotatoes 60 4 5 6

end christen_peeled_twenty_l1608_160898


namespace binomial_10_choose_5_l1608_160816

theorem binomial_10_choose_5 : Nat.choose 10 5 = 252 := by
  sorry

end binomial_10_choose_5_l1608_160816


namespace triangle_angle_inequality_l1608_160809

theorem triangle_angle_inequality (A B C : Real) 
  (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) 
  (h4 : A + B + C = π) : A * Real.cos B + Real.sin A * Real.sin C > 0 := by
  sorry

end triangle_angle_inequality_l1608_160809


namespace smallest_integer_solution_two_is_smallest_l1608_160891

theorem smallest_integer_solution (y : ℤ) : (10 - 5 * y < 5) ↔ y ≥ 2 := by sorry

theorem two_is_smallest : ∃ (y : ℤ), (10 - 5 * y < 5) ∧ (∀ (z : ℤ), 10 - 5 * z < 5 → z ≥ y) ∧ y = 2 := by sorry

end smallest_integer_solution_two_is_smallest_l1608_160891


namespace speed_against_current_l1608_160893

def distance : ℝ := 30
def time_downstream : ℝ := 2
def time_upstream : ℝ := 3

def speed_downstream (v_m v_c : ℝ) : ℝ := v_m + v_c
def speed_upstream (v_m v_c : ℝ) : ℝ := v_m - v_c

theorem speed_against_current :
  ∃ (v_m v_c : ℝ),
    distance = speed_downstream v_m v_c * time_downstream ∧
    distance = speed_upstream v_m v_c * time_upstream ∧
    speed_upstream v_m v_c = 10 :=
by sorry

end speed_against_current_l1608_160893


namespace ball_box_problem_l1608_160848

/-- The number of ways to put n different balls into m different boxes -/
def ways_to_put_balls (n m : ℕ) : ℕ := m^n

/-- The number of ways to put n different balls into m different boxes with exactly k boxes left empty -/
def ways_with_empty_boxes (n m k : ℕ) : ℕ := sorry

theorem ball_box_problem :
  (ways_to_put_balls 4 4 = 256) ∧
  (ways_with_empty_boxes 4 4 1 = 144) ∧
  (ways_with_empty_boxes 4 4 2 = 84) := by sorry

end ball_box_problem_l1608_160848


namespace f_simplification_f_specific_value_l1608_160844

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

theorem f_specific_value : f (-31 * Real.pi / 3) = -1 / 2 := by sorry

end f_simplification_f_specific_value_l1608_160844


namespace x_squared_y_squared_value_l1608_160801

theorem x_squared_y_squared_value (x y : ℝ) 
  (h1 : x + y = 25)
  (h2 : x^2 + y^2 = 169)
  (h3 : x^3*y^3 + y^3*x^3 = 243) :
  x^2 * y^2 = 51984 := by
  sorry

end x_squared_y_squared_value_l1608_160801


namespace quadratic_root_sum_product_l1608_160852

theorem quadratic_root_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 9 ∧ x * y = 14) → 
  m + n = 69 := by
sorry

end quadratic_root_sum_product_l1608_160852


namespace product_sum_relation_l1608_160820

theorem product_sum_relation (a b c N : ℕ) : 
  0 < a → 0 < b → 0 < c →
  a < b → b < c →
  c = a + b →
  N = a * b * c →
  N = 8 * (a + b + c) →
  N = 160 := by
sorry

end product_sum_relation_l1608_160820


namespace table_length_l1608_160814

/-- Proves that a rectangular table with an area of 54 square meters and a width of 600 centimeters has a length of 900 centimeters. -/
theorem table_length (area : ℝ) (width : ℝ) (length : ℝ) : 
  area = 54 → 
  width = 6 →
  area = length * width →
  length * 100 = 900 := by
  sorry

#check table_length

end table_length_l1608_160814


namespace bear_weight_gain_l1608_160824

def bear_weight_problem (total_weight : ℝ) 
  (berry_fraction : ℝ) (insect_fraction : ℝ) 
  (acorn_multiplier : ℝ) (honey_multiplier : ℝ) 
  (salmon_fraction : ℝ) : Prop :=
  let berry_weight := berry_fraction * total_weight
  let insect_weight := insect_fraction * total_weight
  let acorn_weight := acorn_multiplier * berry_weight
  let honey_weight := honey_multiplier * insect_weight
  let gained_weight := berry_weight + insect_weight + acorn_weight + honey_weight
  gained_weight = total_weight →
  total_weight - gained_weight = 0 →
  total_weight - (berry_weight + insect_weight + acorn_weight + honey_weight) = 0

theorem bear_weight_gain :
  bear_weight_problem 1200 (1/5) (1/10) 2 3 (1/4) →
  0 = 0 := by sorry

end bear_weight_gain_l1608_160824


namespace even_function_condition_l1608_160819

theorem even_function_condition (a : ℝ) :
  (∀ x : ℝ, (x - 1) * (x - a) = (-x - 1) * (-x - a)) → a = -1 := by
  sorry

end even_function_condition_l1608_160819


namespace initial_violet_balloons_count_l1608_160885

/-- The number of violet balloons Jason had initially -/
def initial_violet_balloons : ℕ := sorry

/-- The number of violet balloons Jason lost -/
def lost_violet_balloons : ℕ := 3

/-- The number of violet balloons Jason has now -/
def current_violet_balloons : ℕ := 4

/-- Theorem stating that the initial number of violet balloons is 7 -/
theorem initial_violet_balloons_count : initial_violet_balloons = 7 := by
  sorry

end initial_violet_balloons_count_l1608_160885


namespace simplify_expression_l1608_160845

theorem simplify_expression :
  (Real.sqrt 2 + Real.sqrt 3)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = 6 + 2 * Real.sqrt 6 := by
  sorry

end simplify_expression_l1608_160845


namespace largest_proper_fraction_and_ratio_l1608_160817

theorem largest_proper_fraction_and_ratio :
  let fractional_unit : ℚ := 1 / 5
  let largest_proper_fraction : ℚ := 4 / 5
  let reciprocal_of_ten : ℚ := 1 / 10
  (∀ n : ℕ, n < 5 → n / 5 ≤ largest_proper_fraction) ∧
  (largest_proper_fraction / reciprocal_of_ten = 8) := by
  sorry

end largest_proper_fraction_and_ratio_l1608_160817


namespace candy_sales_l1608_160850

theorem candy_sales (x y z : ℝ) : 
  x + y + z = 100 →
  20 * x + 25 * y + 30 * z = 2570 →
  25 * y + 30 * z = 1970 →
  y = 26 := by
sorry

end candy_sales_l1608_160850


namespace isosceles_triangle_quadratic_roots_l1608_160881

theorem isosceles_triangle_quadratic_roots (a b c m : ℝ) : 
  a = 5 →
  b ≠ c →
  (b = a ∨ c = a) →
  b > 0 ∧ c > 0 →
  (b * b + (m + 2) * b + (6 - m) = 0) ∧ 
  (c * c + (m + 2) * c + (6 - m) = 0) →
  m = -10 := by sorry

end isosceles_triangle_quadratic_roots_l1608_160881


namespace tech_gadget_cost_conversion_l1608_160834

/-- Proves that a tech gadget costing 160 Namibian dollars is equivalent to 100 Indian rupees given the exchange rates. -/
theorem tech_gadget_cost_conversion :
  -- Define the exchange rates
  let usd_to_namibian : ℚ := 8
  let usd_to_indian : ℚ := 5
  -- Define the cost in Namibian dollars
  let cost_namibian : ℚ := 160
  -- Define the function to convert Namibian dollars to Indian rupees
  let namibian_to_indian (n : ℚ) : ℚ := n / usd_to_namibian * usd_to_indian
  -- State the theorem
  namibian_to_indian cost_namibian = 100 := by
  sorry

end tech_gadget_cost_conversion_l1608_160834


namespace tan_graph_product_l1608_160805

theorem tan_graph_product (a b : ℝ) : 
  a > 0 → b > 0 → 
  (π / b = 2 * π / 3) →
  (a * Real.tan (b * (π / 6)) = 2) →
  a * b = 3 := by
  sorry

end tan_graph_product_l1608_160805


namespace no_four_digit_perfect_square_with_condition_l1608_160895

theorem no_four_digit_perfect_square_with_condition : ¬ ∃ (abcd : ℕ), 
  (1000 ≤ abcd ∧ abcd ≤ 9999) ∧  -- four-digit number
  (∃ (n : ℕ), abcd = n^2) ∧  -- perfect square
  (∃ (ab cd : ℕ), 
    (10 ≤ ab ∧ ab ≤ 99) ∧  -- ab is two-digit
    (10 ≤ cd ∧ cd ≤ 99) ∧  -- cd is two-digit
    (abcd = 100 * ab + cd) ∧  -- abcd is composed of ab and cd
    (ab = cd / 4)) :=  -- given condition
by sorry


end no_four_digit_perfect_square_with_condition_l1608_160895


namespace max_quotient_value_l1608_160869

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) :
  (∀ x y, 100 ≤ x ∧ x ≤ 300 → 500 ≤ y ∧ y ≤ 1500 → y^2 / x^2 ≤ 225) ∧
  (∃ x y, 100 ≤ x ∧ x ≤ 300 ∧ 500 ≤ y ∧ y ≤ 1500 ∧ y^2 / x^2 = 225) :=
by sorry

end max_quotient_value_l1608_160869


namespace arithmetic_geometric_ratio_l1608_160884

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  h_d_nonzero : d ≠ 0

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1 : ℝ) * seq.d

theorem arithmetic_geometric_ratio 
  (seq : ArithmeticSequence)
  (h_geom : seq.nthTerm 7 ^ 2 = seq.nthTerm 4 * seq.nthTerm 16) :
  ∃ q : ℝ, q ^ 2 = 3 ∧ 
    (seq.nthTerm 7 / seq.nthTerm 4 = q ∨ seq.nthTerm 7 / seq.nthTerm 4 = -q) :=
by sorry

end arithmetic_geometric_ratio_l1608_160884


namespace curve_circle_intersection_l1608_160847

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^2 - 4*x + 3

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Define the line
def line (x y : ℝ) (m : ℝ) : Prop := x + y + m = 0

-- Define perpendicularity of OA and OB
def perpendicular (xA yA xB yB : ℝ) : Prop := xA * xB + yA * yB = 0

-- Main theorem
theorem curve_circle_intersection (m : ℝ) :
  ∃ (x1 y1 x2 y2 : ℝ),
    curve 0 3 ∧ curve 1 0 ∧ curve 3 0 ∧  -- Curve intersects axes at (0,3), (1,0), and (3,0)
    circle_C 0 3 ∧ circle_C 1 0 ∧ circle_C 3 0 ∧  -- These points lie on circle C
    circle_C x1 y1 ∧ circle_C x2 y2 ∧  -- A and B lie on circle C
    line x1 y1 m ∧ line x2 y2 m ∧  -- A and B lie on the line
    perpendicular x1 y1 x2 y2  -- OA is perpendicular to OB
  →
    (m = -1 ∨ m = -3) :=
sorry

end curve_circle_intersection_l1608_160847


namespace three_numbers_sum_l1608_160871

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a + 15 → 
  (a + b + c) / 3 = c - 25 → 
  a + b + c = 60 := by
sorry

end three_numbers_sum_l1608_160871


namespace rabbit_population_solution_l1608_160825

/-- Represents the rabbit population in a park --/
structure RabbitPopulation where
  yesterday : ℕ
  brown : ℕ
  white : ℕ
  male : ℕ
  female : ℕ

/-- Conditions for the rabbit population problem --/
def rabbitProblem (pop : RabbitPopulation) : Prop :=
  -- Today's total is triple yesterday's
  pop.brown + pop.white = 3 * pop.yesterday
  -- 13 + 7 = 1/3 of brown rabbits
  ∧ 20 = pop.brown / 3
  -- White rabbits relation to brown
  ∧ pop.white = pop.brown / 2 - 2
  -- Male to female ratio is 5:3
  ∧ 5 * pop.female = 3 * pop.male
  -- Total rabbits is sum of male and female
  ∧ pop.male + pop.female = pop.brown + pop.white

/-- Theorem stating the solution to the rabbit population problem --/
theorem rabbit_population_solution :
  ∃ (pop : RabbitPopulation),
    rabbitProblem pop ∧ 
    pop.brown = 60 ∧ 
    pop.white = 28 ∧ 
    pop.male = 55 ∧ 
    pop.female = 33 :=
by
  sorry

end rabbit_population_solution_l1608_160825


namespace min_value_sum_min_value_achievable_l1608_160880

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a)) ≥ 3 / Real.rpow 162 (1/3) :=
by sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / (3 * b) + b / (6 * c) + c / (9 * a)) = 3 / Real.rpow 162 (1/3) :=
by sorry

end min_value_sum_min_value_achievable_l1608_160880


namespace rachel_solved_sixteen_at_lunch_l1608_160892

/-- Represents the number of math problems Rachel solved. -/
structure RachelsMathProblems where
  problems_per_minute : ℕ
  minutes_before_bed : ℕ
  total_problems : ℕ

/-- Calculates the number of math problems Rachel solved at lunch. -/
def problems_solved_at_lunch (r : RachelsMathProblems) : ℕ :=
  r.total_problems - (r.problems_per_minute * r.minutes_before_bed)

/-- Theorem stating that Rachel solved 16 math problems at lunch. -/
theorem rachel_solved_sixteen_at_lunch :
  let r : RachelsMathProblems := ⟨5, 12, 76⟩
  problems_solved_at_lunch r = 16 := by sorry

end rachel_solved_sixteen_at_lunch_l1608_160892


namespace max_remainder_dividend_l1608_160838

theorem max_remainder_dividend (divisor quotient : ℕ) (h1 : divisor = 8) (h2 : quotient = 10) : 
  quotient * divisor + (divisor - 1) = 87 := by
  sorry

end max_remainder_dividend_l1608_160838


namespace completing_square_equivalence_l1608_160827

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 + 4*x - 1 = 0) ↔ ((x + 2)^2 = 5) := by
sorry

end completing_square_equivalence_l1608_160827


namespace angle_function_value_l1608_160883

theorem angle_function_value (α : Real) : 
  ((-4 : Real), (3 : Real)) ∈ {(x, y) | x = r * Real.cos α ∧ y = r * Real.sin α ∧ r > 0} →
  (Real.cos (π/2 + α) * Real.sin (3*π/2 - α)) / Real.tan (-π + α) = 16/25 := by
sorry

end angle_function_value_l1608_160883


namespace seating_arrangement_theorem_l1608_160822

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  total_people : ℕ
  rows_with_ten : ℕ
  rows_with_nine : ℕ

/-- Checks if a seating arrangement is valid --/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.total_people = 67 ∧
  s.rows_with_ten * 10 + s.rows_with_nine * 9 = s.total_people

theorem seating_arrangement_theorem :
  ∃ (s : SeatingArrangement), is_valid_arrangement s ∧ s.rows_with_ten = 4 := by
  sorry

end seating_arrangement_theorem_l1608_160822


namespace slope_of_BF_l1608_160837

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 
  ∃ m : ℝ, y + 2 = m * (x + 3)

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem slope_of_BF (B : ℝ × ℝ) :
  parabola B.1 B.2 →
  tangent_line B.1 B.2 →
  second_quadrant B.1 B.2 →
  (B.2 - focus.2) / (B.1 - focus.1) = -3/4 :=
sorry

end slope_of_BF_l1608_160837


namespace geometric_sequence_inequality_l1608_160836

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- For any geometric sequence, the sum of squares of the first and third terms is greater than or equal to twice the square of the second term. -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (h : IsGeometricSequence a) :
    a 1 ^ 2 + a 3 ^ 2 ≥ 2 * (a 2 ^ 2) :=
  sorry


end geometric_sequence_inequality_l1608_160836


namespace g_equation_holds_l1608_160857

-- Define the polynomial g(x)
noncomputable def g (x : ℝ) : ℝ := -4*x^5 + 4*x^3 - 5*x^2 + 2*x + 4

-- State the theorem
theorem g_equation_holds (x : ℝ) : 4*x^5 + 3*x^3 - 2*x + g x = 7*x^3 - 5*x^2 + 4 := by
  sorry

end g_equation_holds_l1608_160857


namespace tim_score_l1608_160897

/-- Represents the scores of players in a basketball game -/
structure BasketballScores where
  joe : ℕ
  tim : ℕ
  ken : ℕ

/-- Theorem: Tim's score is 30 points given the conditions of the basketball game -/
theorem tim_score (scores : BasketballScores) : scores.tim = 30 :=
  by
  have h1 : scores.tim = scores.joe + 20 := by sorry
  have h2 : scores.tim * 2 = scores.ken := by sorry
  have h3 : scores.joe + scores.tim + scores.ken = 100 := by sorry
  sorry

#check tim_score

end tim_score_l1608_160897


namespace gathering_attendance_l1608_160855

theorem gathering_attendance (empty_chairs : ℕ) 
  (h1 : empty_chairs = 9)
  (h2 : ∃ (total_chairs seated_people total_people : ℕ),
    empty_chairs = total_chairs / 3 ∧
    seated_people = 2 * total_chairs / 3 ∧
    seated_people = 3 * total_people / 5) :
  ∃ (total_people : ℕ), total_people = 30 :=
by sorry

end gathering_attendance_l1608_160855


namespace fractional_equation_solution_l1608_160879

theorem fractional_equation_solution :
  ∃! x : ℚ, x ≠ 1 ∧ x ≠ -1 ∧ (x / (x + 1) - 1 = 3 / (x - 1)) :=
by
  use (-1/2)
  sorry

end fractional_equation_solution_l1608_160879


namespace sixth_number_in_sequence_l1608_160843

theorem sixth_number_in_sequence (numbers : List ℝ) 
  (h_count : numbers.length = 11)
  (h_sum_all : numbers.sum = 660)
  (h_sum_first_six : (numbers.take 6).sum = 588)
  (h_sum_last_six : (numbers.drop 5).sum = 390) :
  numbers[5] = 159 :=
by sorry

end sixth_number_in_sequence_l1608_160843


namespace triangle_area_is_3_2_l1608_160808

/-- The area of the triangle bounded by the y-axis and two lines -/
def triangle_area : ℝ :=
  let line1 : ℝ → ℝ → Prop := fun x y ↦ y - 2*x = 1
  let line2 : ℝ → ℝ → Prop := fun x y ↦ 2*y + x = 10
  let y_axis : ℝ → ℝ → Prop := fun x _ ↦ x = 0
  3.2

/-- The area of the triangle is 3.2 -/
theorem triangle_area_is_3_2 : triangle_area = 3.2 := by
  sorry

end triangle_area_is_3_2_l1608_160808


namespace problem_statement_l1608_160899

theorem problem_statement (x y : ℝ) (h : |x + 2| + Real.sqrt (y - 3) = 0) :
  (x + y) ^ 2023 = 1 := by sorry

end problem_statement_l1608_160899


namespace imaginary_part_of_complex_fraction_l1608_160807

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (5 - I) / (1 - I)
  (z.im : ℝ) = 2 := by
  sorry

end imaginary_part_of_complex_fraction_l1608_160807


namespace eighth_term_of_arithmetic_sequence_l1608_160803

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the second term is 17 and the fifth term is 19,
    the eighth term is 21. -/
theorem eighth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_second_term : a 2 = 17)
  (h_fifth_term : a 5 = 19) :
  a 8 = 21 := by
  sorry


end eighth_term_of_arithmetic_sequence_l1608_160803


namespace solution_value_l1608_160821

theorem solution_value (a b : ℝ) (h : 2 * a - 3 * b - 1 = 0) : 5 - 4 * a + 6 * b = 3 := by
  sorry

end solution_value_l1608_160821


namespace isosceles_trapezoid_area_l1608_160863

/-- An isosceles trapezoid with perpendicular diagonals -/
structure IsoscelesTrapezoid where
  /-- The length of the midsegment of the trapezoid -/
  midsegment : ℝ
  /-- The diagonals of the trapezoid are perpendicular -/
  diagonals_perpendicular : Bool
  /-- The trapezoid is isosceles -/
  isosceles : Bool

/-- The area of an isosceles trapezoid with perpendicular diagonals -/
def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem: The area of an isosceles trapezoid with perpendicular diagonals 
    and midsegment of length 5 is 25 -/
theorem isosceles_trapezoid_area 
  (t : IsoscelesTrapezoid) 
  (h1 : t.midsegment = 5) 
  (h2 : t.diagonals_perpendicular = true) 
  (h3 : t.isosceles = true) : 
  area t = 25 := by sorry

end isosceles_trapezoid_area_l1608_160863


namespace log_expression_equality_l1608_160854

theorem log_expression_equality : 
  4 * Real.log 3 / Real.log 2 - Real.log (81 / 4) / Real.log 2 - (5 : ℝ) ^ (Real.log 3 / Real.log 5) + Real.log (Real.sqrt 3) / Real.log 9 = -3/4 := by
  sorry

end log_expression_equality_l1608_160854


namespace smallest_angle_SQR_l1608_160829

-- Define the angles
def angle_PQR : ℝ := 40
def angle_PQS : ℝ := 28

-- Theorem statement
theorem smallest_angle_SQR : 
  let angle_SQR := angle_PQR - angle_PQS
  angle_SQR = 12 := by sorry

end smallest_angle_SQR_l1608_160829


namespace article_original_price_l1608_160862

theorem article_original_price (profit_percentage : ℝ) (profit_amount : ℝ) (original_price : ℝ) : 
  profit_percentage = 35 →
  profit_amount = 1080 →
  original_price = profit_amount / (profit_percentage / 100) →
  ⌊original_price⌋ = 3085 :=
by
  sorry

end article_original_price_l1608_160862


namespace complex_equation_solution_l1608_160841

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution (a : ℂ) :
  a / (1 - i) = (1 + i) / i → a = -2 * i := by
  sorry

end complex_equation_solution_l1608_160841


namespace smallest_prime_after_six_nonprimes_l1608_160858

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that returns true if there are six consecutive nonprime numbers before n -/
def sixConsecutiveNonprimes (n : ℕ) : Prop :=
  ∀ k : ℕ, n - 6 ≤ k → k < n → ¬(isPrime k)

/-- Theorem stating that 89 is the smallest prime number after six consecutive nonprimes -/
theorem smallest_prime_after_six_nonprimes :
  isPrime 89 ∧ sixConsecutiveNonprimes 89 ∧
  ∀ m : ℕ, m < 89 → ¬(isPrime m ∧ sixConsecutiveNonprimes m) :=
sorry

end smallest_prime_after_six_nonprimes_l1608_160858


namespace no_valid_a_l1608_160846

theorem no_valid_a : ∀ a : ℝ, a > 0 → ∃ x : ℝ, 
  |Real.cos x| + |Real.cos (a * x)| ≤ Real.sin x + Real.sin (a * x) := by
  sorry

end no_valid_a_l1608_160846


namespace writing_stats_theorem_l1608_160804

/-- Represents the writing statistics of an author -/
structure WritingStats where
  total_words : ℕ
  total_hours : ℕ
  first_half_hours : ℕ
  first_half_words : ℕ

/-- Calculates the average words per hour -/
def average_words_per_hour (words : ℕ) (hours : ℕ) : ℚ :=
  (words : ℚ) / (hours : ℚ)

/-- Theorem about the writing statistics -/
theorem writing_stats_theorem (stats : WritingStats) 
  (h1 : stats.total_words = 60000)
  (h2 : stats.total_hours = 150)
  (h3 : stats.first_half_hours = 50)
  (h4 : stats.first_half_words = stats.total_words / 2) :
  average_words_per_hour stats.total_words stats.total_hours = 400 ∧
  average_words_per_hour stats.first_half_words stats.first_half_hours = 600 := by
  sorry


end writing_stats_theorem_l1608_160804


namespace ferry_speed_proof_l1608_160840

/-- The speed of ferry P in km/h -/
def speed_P : ℝ := 8

/-- The speed of ferry Q in km/h -/
def speed_Q : ℝ := speed_P + 4

/-- The time taken by ferry P in hours -/
def time_P : ℝ := 2

/-- The time taken by ferry Q in hours -/
def time_Q : ℝ := time_P + 2

/-- The distance traveled by ferry P in km -/
def distance_P : ℝ := speed_P * time_P

/-- The distance traveled by ferry Q in km -/
def distance_Q : ℝ := 3 * distance_P

theorem ferry_speed_proof :
  speed_P = 8 ∧
  speed_Q = speed_P + 4 ∧
  time_P = 2 ∧
  time_Q = time_P + 2 ∧
  distance_Q = 3 * distance_P ∧
  distance_Q = speed_Q * time_Q ∧
  distance_P = speed_P * time_P :=
by
  sorry

#check ferry_speed_proof

end ferry_speed_proof_l1608_160840


namespace product_minus_sum_of_first_45_primes_l1608_160830

def first_n_primes (n : ℕ) : List ℕ :=
  (List.range 1000).filter Nat.Prime |> List.take n

theorem product_minus_sum_of_first_45_primes :
  ∃ x : ℕ, (List.prod (first_n_primes 45) - List.sum (first_n_primes 45) = x) :=
by
  sorry

end product_minus_sum_of_first_45_primes_l1608_160830


namespace friends_distribution_unique_solution_l1608_160823

/-- The number of friends that satisfies the given conditions -/
def number_of_friends : ℕ := 20

/-- The total amount of money distributed (in rupees) -/
def total_amount : ℕ := 100

/-- Theorem stating that the number of friends satisfies the given conditions -/
theorem friends_distribution (n : ℕ) (h : n = number_of_friends) :
  (total_amount / n : ℚ) - (total_amount / (n + 5) : ℚ) = 1 := by
  sorry

/-- Theorem proving that the number of friends is unique -/
theorem unique_solution (n : ℕ) :
  (total_amount / n : ℚ) - (total_amount / (n + 5) : ℚ) = 1 → n = number_of_friends := by
  sorry

end friends_distribution_unique_solution_l1608_160823


namespace triangle_inequality_special_l1608_160878

theorem triangle_inequality_special (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x^2 + y^2 - x*y) + Real.sqrt (y^2 + z^2 - y*z) ≥ Real.sqrt (z^2 + x^2 - z*x) := by
  sorry

end triangle_inequality_special_l1608_160878


namespace housing_price_growth_equation_l1608_160832

/-- 
Given:
- initial_price: The initial housing price in January 2016
- final_price: The final housing price in March 2016
- x: The average monthly growth rate over the two-month period

Prove that the equation initial_price * (1 + x)² = final_price holds.
-/
theorem housing_price_growth_equation 
  (initial_price final_price : ℝ) 
  (x : ℝ) 
  (h_initial : initial_price = 8300)
  (h_final : final_price = 8700) :
  initial_price * (1 + x)^2 = final_price := by
sorry

end housing_price_growth_equation_l1608_160832


namespace sum_of_k_values_l1608_160835

theorem sum_of_k_values : ∃ (S : Finset ℕ), 
  (∀ k ∈ S, ∃ j : ℕ, j > 0 ∧ k > 0 ∧ 1 / j + 1 / k = 1 / 4) ∧ 
  (∀ k : ℕ, k > 0 → (∃ j : ℕ, j > 0 ∧ 1 / j + 1 / k = 1 / 4) → k ∈ S) ∧
  (S.sum id = 51) := by
sorry

end sum_of_k_values_l1608_160835


namespace polynomial_simplification_l1608_160856

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + 2 * x^4 + x + 15) - (x^6 + 4 * x^5 + x^4 - x^3 + 20) =
  x^6 - x^5 + x^4 + x^3 + x - 5 := by
  sorry

end polynomial_simplification_l1608_160856


namespace min_x_prime_factorization_l1608_160853

theorem min_x_prime_factorization (x y : ℕ+) (h : 5 * x^7 = 13 * y^11) :
  ∃ (a b c d : ℕ), 
    (x = a^c * b^d) ∧ 
    (x ≥ 13^6 * 5^7) ∧
    (x = 13^6 * 5^7 → a + b + c + d = 31) :=
sorry

end min_x_prime_factorization_l1608_160853


namespace inequality_equivalence_l1608_160890

theorem inequality_equivalence (a : ℝ) : 
  (∀ x, (4*x + a)/3 > 1 ↔ -((2*x + 1)/2) < 0) → a ≤ 5 := by
sorry

end inequality_equivalence_l1608_160890


namespace jacksons_vacuuming_time_l1608_160826

/-- Represents the problem of calculating Jackson's vacuuming time --/
theorem jacksons_vacuuming_time (vacuum_time : ℝ) : 
  vacuum_time = 2 :=
by
  have hourly_rate : ℝ := 5
  have dish_washing_time : ℝ := 0.5
  have bathroom_cleaning_time : ℝ := 3 * dish_washing_time
  have total_earnings : ℝ := 30
  have total_chore_time : ℝ := 2 * vacuum_time + dish_washing_time + bathroom_cleaning_time
  
  have h1 : hourly_rate * total_chore_time = total_earnings :=
    sorry
  
  -- The proof would go here
  sorry

end jacksons_vacuuming_time_l1608_160826


namespace angle_C_is_120_degrees_max_area_is_sqrt_3_l1608_160806

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem angle_C_is_120_degrees (t : Triangle) 
  (h : t.a * (Real.cos t.C)^2 + 2 * t.c * Real.cos t.A * Real.cos t.C + t.a + t.b = 0) :
  t.C = 2 * π / 3 := by sorry

theorem max_area_is_sqrt_3 (t : Triangle) (h : t.b = 4 * Real.sin t.B) :
  (∀ u : Triangle, u.b = 4 * Real.sin u.B → t.a * t.b * Real.sin t.C / 2 ≥ u.a * u.b * Real.sin u.C / 2) ∧
  t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3 := by sorry

end angle_C_is_120_degrees_max_area_is_sqrt_3_l1608_160806


namespace triangle_acuteness_l1608_160874

theorem triangle_acuteness (a b c : ℝ) (n : ℕ) (h1 : n > 2) 
  (h2 : a > 0) (h3 : b > 0) (h4 : c > 0)
  (h5 : a + b > c) (h6 : b + c > a) (h7 : c + a > b)
  (h8 : a^n + b^n = c^n) : 
  a^2 + b^2 > c^2 := by
  sorry

#check triangle_acuteness

end triangle_acuteness_l1608_160874


namespace largest_number_hcf_lcm_l1608_160849

theorem largest_number_hcf_lcm (a b : ℕ+) : 
  (Nat.gcd a b = 62) →
  (∃ (x y : ℕ+), x = 11 ∧ y = 12 ∧ Nat.lcm a b = 62 * x * y) →
  max a b = 744 := by
sorry

end largest_number_hcf_lcm_l1608_160849


namespace fraction_percent_of_x_l1608_160896

theorem fraction_percent_of_x (x : ℝ) (h : x > 0) : (x / 10 + x / 25) / x * 100 = 14 := by
  sorry

end fraction_percent_of_x_l1608_160896


namespace even_function_range_l1608_160851

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_function_range (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (h_f_neg_two : f (-2) = 1) :
  {x : ℝ | f (x - 2) ≤ 1} = Set.Icc 0 4 := by
  sorry

end even_function_range_l1608_160851


namespace parabola_reflection_l1608_160800

/-- Reflects a point (x, y) over the point (1, 1) -/
def reflect (x y : ℝ) : ℝ × ℝ := (2 - x, 2 - y)

/-- The original parabola y = x^2 -/
def original_parabola (x y : ℝ) : Prop := y = x^2

/-- The reflected parabola y = -x^2 + 4x - 2 -/
def reflected_parabola (x y : ℝ) : Prop := y = -x^2 + 4*x - 2

theorem parabola_reflection :
  ∀ x y : ℝ, original_parabola x y ↔ reflected_parabola (reflect x y).1 (reflect x y).2 :=
sorry

end parabola_reflection_l1608_160800


namespace no_three_distinct_solutions_l1608_160873

theorem no_three_distinct_solutions : 
  ¬∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    a * (a - 4) = 12 ∧ b * (b - 4) = 12 ∧ c * (c - 4) = 12 := by
  sorry

end no_three_distinct_solutions_l1608_160873


namespace quadratic_equation_general_form_quadratic_equation_coefficients_l1608_160860

/-- Given a quadratic equation 2x^2 - 1 = 6x, prove its general form and coefficients --/
theorem quadratic_equation_general_form :
  ∀ x : ℝ, (2 * x^2 - 1 = 6 * x) ↔ (2 * x^2 - 6 * x - 1 = 0) :=
by sorry

/-- Prove the coefficients of the general form ax^2 + bx + c = 0 --/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = 2 * x^2 - 6 * x - 1) ∧ 
    (a = 2 ∧ b = -6 ∧ c = -1) :=
by sorry

end quadratic_equation_general_form_quadratic_equation_coefficients_l1608_160860


namespace initial_lives_count_l1608_160886

/-- Proves that if a person loses 6 lives, then gains 37 lives, and ends up with 41 lives, they must have started with 10 lives. -/
theorem initial_lives_count (initial_lives : ℕ) : 
  initial_lives - 6 + 37 = 41 → initial_lives = 10 := by
  sorry

#check initial_lives_count

end initial_lives_count_l1608_160886


namespace subtraction_multiplication_problem_l1608_160864

theorem subtraction_multiplication_problem (x : ℝ) : 
  8.9 - x = 3.1 → (x * 3.1) * 2.5 = 44.95 := by
  sorry

end subtraction_multiplication_problem_l1608_160864


namespace sine_cosine_transform_l1608_160802

theorem sine_cosine_transform (x : ℝ) : 
  Real.sqrt 3 * Real.sin (3 * x) + Real.cos (3 * x) = 2 * Real.sin (3 * x + π / 6) := by
  sorry

end sine_cosine_transform_l1608_160802


namespace number_of_people_l1608_160889

theorem number_of_people (average_age : ℝ) (youngest_age : ℝ) (average_age_at_birth : ℝ) :
  average_age = 30 →
  youngest_age = 3 →
  average_age_at_birth = 27 →
  ∃ n : ℕ, n = 7 ∧ 
    average_age * n = youngest_age + average_age_at_birth * (n - 1) :=
by sorry

end number_of_people_l1608_160889


namespace log_inequality_l1608_160812

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + 1/x) > 1/(1 + x) := by
  sorry

end log_inequality_l1608_160812


namespace fifth_element_row_20_l1608_160842

-- Define Pascal's triangle function
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem statement
theorem fifth_element_row_20 : pascal 20 4 = 4845 := by
  sorry

end fifth_element_row_20_l1608_160842


namespace arccos_zero_equals_pi_half_l1608_160810

theorem arccos_zero_equals_pi_half : Real.arccos 0 = π / 2 := by
  sorry

end arccos_zero_equals_pi_half_l1608_160810


namespace calculation_proof_l1608_160811

theorem calculation_proof : 
  |(-7)| / ((2 / 3) - (1 / 5)) - (1 / 2) * ((-4)^2) = 7 := by
  sorry

end calculation_proof_l1608_160811


namespace cube_root_equation_solution_l1608_160867

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x)^(1/3 : ℝ) = 1 :=
by
  sorry

end cube_root_equation_solution_l1608_160867


namespace z_in_third_quadrant_l1608_160875

/-- The complex number under consideration -/
def z : ℂ := Complex.I * (-2 + 3 * Complex.I)

/-- A complex number is in the third quadrant if its real and imaginary parts are both negative -/
def is_in_third_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im < 0

/-- Theorem stating that z is in the third quadrant -/
theorem z_in_third_quadrant : is_in_third_quadrant z := by
  sorry

end z_in_third_quadrant_l1608_160875


namespace circle_with_same_center_and_radius_2_l1608_160877

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the center of a circle
def center (f : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

-- Define a new circle with given center and radius
def new_circle (c : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  (x - c.1)^2 + (y - c.2)^2 = r^2

-- Theorem statement
theorem circle_with_same_center_and_radius_2 :
  ∀ (x y : ℝ),
  new_circle (center given_circle) 2 x y ↔ (x + 1)^2 + y^2 = 4 :=
by sorry

end circle_with_same_center_and_radius_2_l1608_160877


namespace problem_statement_l1608_160815

theorem problem_statement (P Q : Prop) (h_P : P ↔ (2 + 2 = 5)) (h_Q : Q ↔ (3 > 2)) :
  (P ∨ Q) ∧ ¬(¬Q) := by
  sorry

end problem_statement_l1608_160815


namespace peanut_problem_l1608_160894

theorem peanut_problem (a b c d : ℕ) : 
  b = a + 6 ∧ 
  c = b + 6 ∧ 
  d = c + 6 ∧ 
  a + b + c + d = 120 → 
  d = 39 := by
sorry

end peanut_problem_l1608_160894


namespace green_candies_count_l1608_160887

/-- Proves the number of green candies in a bag given the number of blue and red candies and the probability of picking a blue candy. -/
theorem green_candies_count (blue : ℕ) (red : ℕ) (prob_blue : ℚ) (green : ℕ) : 
  blue = 3 → red = 4 → prob_blue = 1/4 → 
  (blue : ℚ) / ((green : ℚ) + (blue : ℚ) + (red : ℚ)) = prob_blue →
  green = 5 := by sorry

end green_candies_count_l1608_160887


namespace complex_equality_problem_l1608_160876

theorem complex_equality_problem (a : ℝ) : 
  let z₁ : ℂ := a + Complex.I
  let z₂ : ℂ := 2 - Complex.I
  Complex.abs z₁ = Complex.abs z₂ → (a = 2 ∨ a = -2) := by
sorry

end complex_equality_problem_l1608_160876


namespace width_of_specific_box_l1608_160831

/-- A rectangular box with given dimensions -/
structure RectangularBox where
  height : ℝ
  length : ℝ
  width : ℝ
  diagonal : ℝ
  height_positive : height > 0
  length_eq_twice_height : length = 2 * height
  diagonal_formula : diagonal^2 = length^2 + width^2 + height^2

/-- Theorem stating the width of a specific rectangular box -/
theorem width_of_specific_box :
  ∀ (box : RectangularBox),
    box.height = 8 ∧ 
    box.diagonal = 20 →
    box.width = 4 * Real.sqrt 5 := by
  sorry

end width_of_specific_box_l1608_160831


namespace sue_total_items_l1608_160882

def initial_books : ℕ := 15
def initial_movies : ℕ := 6
def returned_books : ℕ := 8
def checked_out_books : ℕ := 9

def remaining_books : ℕ := initial_books - returned_books + checked_out_books
def remaining_movies : ℕ := initial_movies - (initial_movies / 3)

theorem sue_total_items : remaining_books + remaining_movies = 20 := by
  sorry

end sue_total_items_l1608_160882


namespace base_7_representation_l1608_160828

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list of digits are consecutive -/
def isConsecutive (digits : List ℕ) : Bool :=
  sorry

theorem base_7_representation :
  let base7Digits := toBase7 143
  base7Digits = [2, 6, 3] ∧
  base7Digits.length = 3 ∧
  isConsecutive base7Digits = true :=
by sorry

end base_7_representation_l1608_160828


namespace light_travel_distance_l1608_160861

/-- The speed of light in miles per second -/
def speed_of_light : ℝ := 186282

/-- The number of seconds light travels -/
def travel_time : ℝ := 500

/-- The conversion factor from miles to kilometers -/
def mile_to_km : ℝ := 1.609

/-- The distance light travels in kilometers -/
def light_distance : ℝ := speed_of_light * travel_time * mile_to_km

theorem light_travel_distance :
  ∃ ε > 0, |light_distance - 1.498e8| < ε :=
sorry

end light_travel_distance_l1608_160861


namespace alyssas_soccer_games_l1608_160859

theorem alyssas_soccer_games (games_this_year games_next_year total_games : ℕ) 
  (h1 : games_this_year = 11)
  (h2 : games_next_year = 15)
  (h3 : total_games = 39) :
  total_games - games_this_year - games_next_year = 13 := by
  sorry

end alyssas_soccer_games_l1608_160859
