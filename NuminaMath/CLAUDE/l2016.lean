import Mathlib

namespace x_coordinate_C_l2016_201695

/-- Parabola function -/
def parabola (x : ℝ) : ℝ := x^2

/-- Triangle ABC with vertices on parabola y = x^2 -/
structure TriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A.2 = parabola A.1
  h_B : B.2 = parabola B.1
  h_C : C.2 = parabola C.1
  h_A_origin : A = (0, 0)
  h_B_coords : B = (-3, 9)
  h_C_positive : C.1 > 0
  h_BC_parallel : B.2 = C.2
  h_area : (1/2) * |C.1 + 3| * C.2 = 45

/-- The x-coordinate of vertex C is 7 -/
theorem x_coordinate_C (t : TriangleABC) : t.C.1 = 7 := by
  sorry

end x_coordinate_C_l2016_201695


namespace geometric_sequence_problem_l2016_201691

/-- A geometric sequence with first term 2 and satisfying a₄a₆ = 4a₇² has a₃ = 1 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : a 1 = 2) 
  (h2 : a 4 * a 6 = 4 * (a 7)^2) (h3 : ∀ n : ℕ, n ≥ 1 → ∃ q : ℝ, a (n + 1) = a n * q) : 
  a 3 = 1 := by
  sorry

end geometric_sequence_problem_l2016_201691


namespace exists_number_satisfying_equation_l2016_201677

theorem exists_number_satisfying_equation : ∃ x : ℝ, (x * 7) / (10 * 17) = 10000 := by
  sorry

end exists_number_satisfying_equation_l2016_201677


namespace rosas_initial_flowers_l2016_201661

/-- The problem of finding Rosa's initial number of flowers -/
theorem rosas_initial_flowers :
  ∀ (initial_flowers additional_flowers total_flowers : ℕ),
    additional_flowers = 23 →
    total_flowers = 90 →
    total_flowers = initial_flowers + additional_flowers →
    initial_flowers = 67 := by
  sorry

end rosas_initial_flowers_l2016_201661


namespace total_amount_correct_l2016_201675

/-- The total amount earned from selling notebooks -/
def total_amount (a b : ℝ) : ℝ :=
  70 * (1 + 0.2) * a + 30 * (a - b)

/-- Proof that the total amount is correct -/
theorem total_amount_correct (a b : ℝ) :
  let total_notebooks : ℕ := 100
  let first_batch : ℕ := 70
  let price_increase : ℝ := 0.2
  total_amount a b = first_batch * (1 + price_increase) * a + (total_notebooks - first_batch) * (a - b) :=
by sorry

end total_amount_correct_l2016_201675


namespace a_investment_is_800_l2016_201680

/-- Represents the investment and profit scenario of three business partners -/
structure BusinessScenario where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  investment_period : ℝ
  total_profit : ℝ
  c_profit_share : ℝ

/-- The business scenario with given conditions -/
def given_scenario : BusinessScenario :=
  { a_investment := 0,  -- Unknown, to be solved
    b_investment := 1000,
    c_investment := 1200,
    investment_period := 2,
    total_profit := 1000,
    c_profit_share := 400 }

/-- Theorem stating that a's investment in the given scenario is 800 -/
theorem a_investment_is_800 (scenario : BusinessScenario) 
  (h1 : scenario = given_scenario) :
  scenario.a_investment = 800 := by
  sorry


end a_investment_is_800_l2016_201680


namespace quadratic_intercepts_l2016_201614

/-- Given a quadratic function y = x^2 + bx - 3 that passes through the point (3,0),
    prove that b = -2 and the other x-intercept is at (-1,0) -/
theorem quadratic_intercepts (b : ℝ) : 
  (3^2 + 3*b - 3 = 0) → 
  (b = -2 ∧ (-1)^2 + (-1)*b - 3 = 0) :=
by sorry

end quadratic_intercepts_l2016_201614


namespace picture_frame_length_l2016_201634

/-- Given a rectangular frame with perimeter 30 cm and width 10 cm, its length is 5 cm. -/
theorem picture_frame_length (perimeter width : ℝ) (h1 : perimeter = 30) (h2 : width = 10) :
  let length := (perimeter - 2 * width) / 2
  length = 5 := by sorry

end picture_frame_length_l2016_201634


namespace sum_of_perimeters_equals_expected_l2016_201639

/-- Calculates the sum of perimeters of triangles formed by repeatedly
    connecting points 1/3 of the distance along each side of an initial
    equilateral triangle, for a given number of iterations. -/
def sumOfPerimeters (initialSideLength : ℚ) (iterations : ℕ) : ℚ :=
  let rec perimeter (sideLength : ℚ) (n : ℕ) : ℚ :=
    if n = 0 then 0
    else 3 * sideLength + perimeter (sideLength / 3) (n - 1)
  perimeter initialSideLength (iterations + 1)

/-- Theorem stating that the sum of perimeters of triangles formed by
    repeatedly connecting points 1/3 of the distance along each side of
    an initial equilateral triangle with side length 18 units, for 4
    iterations, is equal to 80 2/3 units. -/
theorem sum_of_perimeters_equals_expected :
  sumOfPerimeters 18 4 = 80 + 2/3 := by
  sorry

end sum_of_perimeters_equals_expected_l2016_201639


namespace speaker_combinations_l2016_201603

/-- Represents the number of representatives for each company -/
def company_reps : List ℕ := [2, 1, 1, 1, 1]

/-- The total number of companies -/
def num_companies : ℕ := company_reps.length

/-- The number of speakers required -/
def num_speakers : ℕ := 3

/-- Calculates the number of ways to choose speakers from different companies -/
def choose_speakers (reps : List ℕ) (k : ℕ) : ℕ := sorry

theorem speaker_combinations :
  choose_speakers company_reps num_speakers = 16 := by sorry

end speaker_combinations_l2016_201603


namespace range_of_a_for_increasing_f_l2016_201662

/-- A function f is increasing on an interval [a, b) if for any x, y in [a, b) with x < y, f(x) < f(y) -/
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y < b → f x < f y

/-- The quadratic function f(x) = ax^2 + 2x - 2a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 2 * a

theorem range_of_a_for_increasing_f :
  {a : ℝ | IsIncreasing (f a) (-1) 2} = Set.Icc (-1/2) 1 := by
  sorry

end range_of_a_for_increasing_f_l2016_201662


namespace sqrt_sum_eq_two_l2016_201632

theorem sqrt_sum_eq_two (x₁ x₂ : ℝ) 
  (h1 : x₁ ≥ x₂) 
  (h2 : x₂ ≥ 0) 
  (h3 : x₁ + x₂ = 2) : 
  Real.sqrt (x₁ + Real.sqrt (x₁^2 - x₂^2)) + Real.sqrt (x₁ - Real.sqrt (x₁^2 - x₂^2)) = 2 := by
  sorry

end sqrt_sum_eq_two_l2016_201632


namespace water_remaining_l2016_201678

theorem water_remaining (poured_out : ℚ) (h : poured_out = 45 / 100) :
  1 - poured_out = 55 / 100 := by
  sorry

end water_remaining_l2016_201678


namespace product_mod_800_l2016_201635

theorem product_mod_800 : (2431 * 1587) % 800 = 397 := by
  sorry

end product_mod_800_l2016_201635


namespace exponent_and_logarithm_equalities_l2016_201619

theorem exponent_and_logarithm_equalities :
  (3 : ℝ) ^ 64 = 4 ∧ (4 : ℝ) ^ (Real.log 3 / Real.log 2) = 9 := by
  sorry

end exponent_and_logarithm_equalities_l2016_201619


namespace ralphSockPurchase_l2016_201642

/-- Represents the number of socks bought at each price point -/
structure SockPurchase where
  oneDollar : Nat
  twoDollar : Nat
  fourDollar : Nat

/-- Checks if the SockPurchase satisfies the problem conditions -/
def isValidPurchase (p : SockPurchase) : Prop :=
  p.oneDollar + p.twoDollar + p.fourDollar = 10 ∧
  p.oneDollar + 2 * p.twoDollar + 4 * p.fourDollar = 30 ∧
  p.oneDollar ≥ 1 ∧ p.twoDollar ≥ 1 ∧ p.fourDollar ≥ 1

/-- Theorem stating that the only valid purchase has 2 pairs of $1 socks -/
theorem ralphSockPurchase :
  ∀ p : SockPurchase, isValidPurchase p → p.oneDollar = 2 :=
by sorry

end ralphSockPurchase_l2016_201642


namespace min_a_for_parabola_l2016_201640

/-- Given a parabola y = ax^2 + bx + c with vertex at (1/4, -9/8), 
    where a > 0 and a + b + c is an integer, 
    the minimum possible value of a is 2/9 -/
theorem min_a_for_parabola (a b c : ℝ) : 
  a > 0 ∧ 
  (∃ k : ℤ, a + b + c = k) ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = a * (x - 1/4)^2 - 9/8) → 
  (∀ a' : ℝ, a' > 0 ∧ 
    (∃ b' c' : ℝ, (∃ k : ℤ, a' + b' + c' = k) ∧ 
      (∀ x : ℝ, a' * x^2 + b' * x + c' = a' * (x - 1/4)^2 - 9/8)) → 
    a' ≥ 2/9) ∧ 
  a = 2/9 := by
sorry

end min_a_for_parabola_l2016_201640


namespace rachel_apple_tree_l2016_201621

/-- The number of apples on Rachel's tree after picking some and new ones growing. -/
def final_apples (initial : ℕ) (picked : ℕ) (new_grown : ℕ) : ℕ :=
  initial - picked + new_grown

/-- Theorem stating that the final number of apples is correct. -/
theorem rachel_apple_tree (initial : ℕ) (picked : ℕ) (new_grown : ℕ) 
    (h1 : initial = 4) (h2 : picked = 2) (h3 : new_grown = 3) : 
  final_apples initial picked new_grown = 5 := by
  sorry

end rachel_apple_tree_l2016_201621


namespace increasing_function_inequality_l2016_201671

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h : is_increasing f) : f (a^2 + 1) > f a := by
  sorry

end increasing_function_inequality_l2016_201671


namespace no_lower_grade_possible_l2016_201651

/-- Represents Lisa's quiz performance and goal --/
structure QuizPerformance where
  total_quizzes : ℕ
  goal_percentage : ℚ
  completed_quizzes : ℕ
  as_earned : ℕ

/-- Theorem stating that Lisa cannot earn a grade lower than A on any remaining quiz --/
theorem no_lower_grade_possible (perf : QuizPerformance) 
  (h1 : perf.total_quizzes = 60)
  (h2 : perf.goal_percentage = 85 / 100)
  (h3 : perf.completed_quizzes = 35)
  (h4 : perf.as_earned = 25) :
  (perf.total_quizzes - perf.completed_quizzes : ℚ) - 
  (↑⌈perf.goal_percentage * perf.total_quizzes⌉ - perf.as_earned) ≤ 0 := by
  sorry

#eval ⌈(85 : ℚ) / 100 * 60⌉ -- Expected output: 51

end no_lower_grade_possible_l2016_201651


namespace company_survey_l2016_201610

theorem company_survey (total employees_with_tool employees_with_training employees_with_both : ℕ)
  (h_total : total = 150)
  (h_tool : employees_with_tool = 90)
  (h_training : employees_with_training = 60)
  (h_both : employees_with_both = 30) :
  (↑(total - (employees_with_tool + employees_with_training - employees_with_both)) / ↑total) * 100 = 20 :=
by sorry

end company_survey_l2016_201610


namespace total_litter_weight_l2016_201605

/-- The amount of litter collected by Gina and her neighborhood --/
def litterCollection (gina_bags : ℕ) (neighborhood_multiplier : ℕ) (weight_per_bag : ℕ) : ℕ :=
  let total_bags := gina_bags + gina_bags * neighborhood_multiplier
  total_bags * weight_per_bag

/-- Theorem stating the total weight of litter collected --/
theorem total_litter_weight :
  litterCollection 2 82 4 = 664 := by
  sorry

end total_litter_weight_l2016_201605


namespace find_number_l2016_201607

theorem find_number : ∃ x : ℝ, x / 2 = 9 ∧ x = 18 := by sorry

end find_number_l2016_201607


namespace cubic_function_derivative_condition_l2016_201664

/-- Given a function f(x) = x^3 - mx + 3, if f'(1) = 0, then m = 3 -/
theorem cubic_function_derivative_condition (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - m*x + 3
  (∀ x, (deriv f) x = 3*x^2 - m) → (deriv f) 1 = 0 → m = 3 := by
  sorry

end cubic_function_derivative_condition_l2016_201664


namespace equivalent_operations_l2016_201615

theorem equivalent_operations (x : ℝ) : (x * (4/5)) / (4/7) = x * (7/5) := by
  sorry

end equivalent_operations_l2016_201615


namespace circle_area_ratio_l2016_201683

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * Real.pi * r₁)) = (48 / 360 * (2 * Real.pi * r₂)) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 16 / 25 := by
sorry

end circle_area_ratio_l2016_201683


namespace expression_simplification_and_evaluation_l2016_201692

theorem expression_simplification_and_evaluation :
  let x : ℝ := 3
  let expr := (1 / (x - 2) - 1 / (x + 1)) / (3 / (x^2 - 1))
  expr = 2 := by sorry

end expression_simplification_and_evaluation_l2016_201692


namespace unique_intersection_point_l2016_201623

/-- Two equations y = x^2 and y = 2x + k intersect at exactly one point if and only if k = 0 -/
theorem unique_intersection_point (k : ℝ) : 
  (∃! x : ℝ, x^2 = 2*x + k) ↔ k = 0 :=
by sorry

end unique_intersection_point_l2016_201623


namespace isabellas_hourly_rate_l2016_201696

/-- Calculates Isabella's hourly rate given her work schedule and total earnings -/
theorem isabellas_hourly_rate 
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (total_weeks : ℕ)
  (total_earnings : ℕ)
  (h1 : hours_per_day = 5)
  (h2 : days_per_week = 6)
  (h3 : total_weeks = 7)
  (h4 : total_earnings = 1050) :
  total_earnings / (hours_per_day * days_per_week * total_weeks) = 5 := by
sorry

end isabellas_hourly_rate_l2016_201696


namespace cube_face_sum_l2016_201685

/-- Represents the numbers on the faces of a cube -/
structure CubeFaces where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- The sum of vertex products for a given set of cube faces -/
def vertexProductSum (faces : CubeFaces) : ℕ :=
  faces.a * faces.b * faces.c +
  faces.a * faces.e * faces.c +
  faces.a * faces.b * faces.f +
  faces.a * faces.e * faces.f +
  faces.d * faces.b * faces.c +
  faces.d * faces.e * faces.c +
  faces.d * faces.b * faces.f +
  faces.d * faces.e * faces.f

/-- The sum of all face values -/
def faceSum (faces : CubeFaces) : ℕ :=
  faces.a + faces.b + faces.c + faces.d + faces.e + faces.f

theorem cube_face_sum (faces : CubeFaces) :
  vertexProductSum faces = 1008 → faceSum faces = 173 := by
  sorry


end cube_face_sum_l2016_201685


namespace cabinet_price_l2016_201630

theorem cabinet_price (P : ℝ) (discount_rate : ℝ) (discounted_price : ℝ) : 
  discount_rate = 0.15 →
  discounted_price = 1020 →
  discounted_price = P * (1 - discount_rate) →
  P = 1200 := by
sorry

end cabinet_price_l2016_201630


namespace average_score_is_94_l2016_201655

/-- The average math test score of Clyde's four children -/
def average_score (june_score patty_score josh_score henry_score : ℕ) : ℚ :=
  (june_score + patty_score + josh_score + henry_score : ℚ) / 4

/-- Theorem stating that the average math test score of Clyde's four children is 94 -/
theorem average_score_is_94 :
  average_score 97 85 100 94 = 94 := by sorry

end average_score_is_94_l2016_201655


namespace sum_of_three_element_subset_sums_l2016_201648

def A : Finset ℕ := Finset.range 10

def three_element_subsets (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ subset => subset.card = 3)

def subset_sum (subset : Finset ℕ) : ℕ :=
  subset.sum id

theorem sum_of_three_element_subset_sums : 
  (three_element_subsets A).sum subset_sum = 1980 := by
  sorry

end sum_of_three_element_subset_sums_l2016_201648


namespace existence_of_non_coprime_pair_l2016_201609

theorem existence_of_non_coprime_pair :
  ∃ m : ℤ, (Nat.gcd (100 + 101 * m).natAbs (101 - 100 * m).natAbs) ≠ 1 := by
  sorry

end existence_of_non_coprime_pair_l2016_201609


namespace gcd_lcm_sum_231_4620_l2016_201658

theorem gcd_lcm_sum_231_4620 : Nat.gcd 231 4620 + Nat.lcm 231 4620 = 4851 := by sorry

end gcd_lcm_sum_231_4620_l2016_201658


namespace total_amount_paid_l2016_201682

def apple_quantity : ℕ := 8
def apple_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 70

theorem total_amount_paid : 
  apple_quantity * apple_rate + mango_quantity * mango_rate = 1190 := by
  sorry

end total_amount_paid_l2016_201682


namespace profit_calculation_l2016_201667

/-- Calculates the total profit of a business given the investments and one partner's share of the profit -/
def calculate_total_profit (investment_A investment_B investment_C share_A : ℕ) : ℕ :=
  let ratio_A := investment_A / (Nat.gcd investment_A (Nat.gcd investment_B investment_C))
  let ratio_B := investment_B / (Nat.gcd investment_A (Nat.gcd investment_B investment_C))
  let ratio_C := investment_C / (Nat.gcd investment_A (Nat.gcd investment_B investment_C))
  let total_ratio := ratio_A + ratio_B + ratio_C
  (share_A * total_ratio) / ratio_A

theorem profit_calculation (investment_A investment_B investment_C share_A : ℕ) 
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : share_A = 3900) :
  calculate_total_profit investment_A investment_B investment_C share_A = 13000 := by
  sorry

end profit_calculation_l2016_201667


namespace water_consumption_l2016_201652

theorem water_consumption (morning_amount : ℝ) (afternoon_multiplier : ℝ) : 
  morning_amount = 1.5 → 
  afternoon_multiplier = 3 → 
  morning_amount + (afternoon_multiplier * morning_amount) = 6 := by
  sorry

end water_consumption_l2016_201652


namespace least_common_multiple_of_pack_sizes_l2016_201653

theorem least_common_multiple_of_pack_sizes (tulip_pack_size daffodil_pack_size : ℕ) 
  (h1 : tulip_pack_size = 15) 
  (h2 : daffodil_pack_size = 16) : 
  Nat.lcm tulip_pack_size daffodil_pack_size = 240 := by
  sorry

end least_common_multiple_of_pack_sizes_l2016_201653


namespace square_with_trees_theorem_l2016_201668

/-- Represents a square with trees at its vertices -/
structure SquareWithTrees where
  side_length : ℝ
  height_A : ℝ
  height_B : ℝ
  height_C : ℝ
  height_D : ℝ

/-- Checks if there exists a point equidistant from all tree tops -/
def has_equidistant_point (s : SquareWithTrees) : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ x < s.side_length ∧ 0 < y ∧ y < s.side_length ∧
  (s.height_A^2 + x^2 + y^2 = s.height_B^2 + (s.side_length - x)^2 + y^2) ∧
  (s.height_A^2 + x^2 + y^2 = s.height_C^2 + (s.side_length - x)^2 + (s.side_length - y)^2) ∧
  (s.height_A^2 + x^2 + y^2 = s.height_D^2 + x^2 + (s.side_length - y)^2)

/-- The main theorem about the square with trees -/
theorem square_with_trees_theorem (s : SquareWithTrees) 
  (h1 : s.height_A = 7)
  (h2 : s.height_B = 13)
  (h3 : s.height_C = 17)
  (h4 : has_equidistant_point s) :
  s.side_length > Real.sqrt 120 ∧ s.height_D = 13 := by
  sorry

end square_with_trees_theorem_l2016_201668


namespace prob_same_length_is_17_35_l2016_201612

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of segments (sides and diagonals) in a regular hexagon -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The number of pairs of segments with the same length -/
def same_length_pairs : ℕ := (num_sides.choose 2) + (num_diagonals.choose 2)

/-- The total number of possible pairs of segments -/
def total_pairs : ℕ := total_segments.choose 2

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ := same_length_pairs / total_pairs

theorem prob_same_length_is_17_35 : prob_same_length = 17 / 35 := by sorry

end prob_same_length_is_17_35_l2016_201612


namespace sara_golf_balls_l2016_201686

def dozen : ℕ := 12

theorem sara_golf_balls (total_balls : ℕ) (h : total_balls = 192) :
  total_balls / dozen = 16 := by
  sorry

end sara_golf_balls_l2016_201686


namespace roots_properties_l2016_201637

theorem roots_properties (r s t : ℝ) : 
  (∀ x : ℝ, x * (x - 2) * (3 * x - 7) = 2 ↔ x = r ∨ x = s ∨ x = t) →
  (r > 0 ∧ s > 0 ∧ t > 0) ∧
  (Real.arctan r + Real.arctan s + Real.arctan t = 3 * π / 4) := by
sorry

end roots_properties_l2016_201637


namespace student_multiplication_problem_l2016_201699

theorem student_multiplication_problem (chosen_number : ℕ) (final_result : ℕ) (subtracted_amount : ℕ) :
  chosen_number = 125 →
  final_result = 112 →
  subtracted_amount = 138 →
  ∃ x : ℚ, chosen_number * x - subtracted_amount = final_result ∧ x = 2 :=
by sorry

end student_multiplication_problem_l2016_201699


namespace second_day_percentage_l2016_201646

def puzzle_pieces : ℕ := 1000
def first_day_percentage : ℚ := 10 / 100
def third_day_percentage : ℚ := 30 / 100
def pieces_left_after_third_day : ℕ := 504

theorem second_day_percentage :
  ∃ (p : ℚ),
    p > 0 ∧
    p < 1 ∧
    (puzzle_pieces * (1 - first_day_percentage) * (1 - p) * (1 - third_day_percentage) : ℚ) =
      pieces_left_after_third_day ∧
    p = 20 / 100 := by
  sorry

end second_day_percentage_l2016_201646


namespace complement_A_intersect_B_l2016_201617

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 6 > 0}
def B : Set ℝ := {x | x - 1 > 0}

-- Define the complement of A in ℝ
def C_R_A : Set ℝ := (Set.univ : Set ℝ) \ A

-- State the theorem
theorem complement_A_intersect_B : C_R_A ∩ B = Set.Ioo 1 3 := by sorry

end complement_A_intersect_B_l2016_201617


namespace trapezoid_height_theorem_l2016_201698

/-- Represents a trapezoid with given diagonal lengths and midsegment length -/
structure Trapezoid where
  diag1 : ℝ
  diag2 : ℝ
  midsegment : ℝ

/-- Calculates the height of a trapezoid given its properties -/
def trapezoidHeight (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that a trapezoid with diagonals 6 and 8 and midsegment 5 has height 4.8 -/
theorem trapezoid_height_theorem (t : Trapezoid) 
  (h1 : t.diag1 = 6) 
  (h2 : t.diag2 = 8) 
  (h3 : t.midsegment = 5) : 
  trapezoidHeight t = 4.8 := by
  sorry

end trapezoid_height_theorem_l2016_201698


namespace midpoint_octagon_area_ratio_l2016_201674

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- An octagon formed by connecting midpoints of another octagon's sides -/
def MidpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of an octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The ratio of the area of the midpoint octagon to the area of the original regular octagon is 1/2 -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (MidpointOctagon o) / area o = 1 / 2 :=
sorry

end midpoint_octagon_area_ratio_l2016_201674


namespace non_working_video_games_l2016_201665

theorem non_working_video_games (total : ℕ) (price : ℕ) (earnings : ℕ) : 
  total = 10 → price = 6 → earnings = 12 → total - (earnings / price) = 8 := by
  sorry

end non_working_video_games_l2016_201665


namespace count_decimals_near_three_elevenths_l2016_201627

theorem count_decimals_near_three_elevenths :
  let lower_bound : ℚ := 2614 / 10000
  let upper_bound : ℚ := 2792 / 10000
  let count := (upper_bound * 10000).floor.toNat - (lower_bound * 10000).ceil.toNat + 1
  (∀ s : ℚ, lower_bound ≤ s → s ≤ upper_bound →
    (∃ w x y z : ℕ, w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧
      s = (w * 1000 + x * 100 + y * 10 + z) / 10000) →
    (∀ n d : ℕ, n ≤ 3 → 0 < d → |s - n / d| ≥ |s - 3 / 11|)) →
  count = 179 := by
sorry

end count_decimals_near_three_elevenths_l2016_201627


namespace bobby_shoe_cost_l2016_201656

theorem bobby_shoe_cost (mold_cost labor_rate hours discount_rate : ℝ) : 
  mold_cost = 250 →
  labor_rate = 75 →
  hours = 8 →
  discount_rate = 0.8 →
  mold_cost + (labor_rate * hours * discount_rate) = 730 := by
sorry

end bobby_shoe_cost_l2016_201656


namespace integer_sum_problem_l2016_201638

theorem integer_sum_problem (x y : ℕ) (h1 : x > y) (h2 : x - y = 16) (h3 : x * y = 63) :
  x + y = 2 * Real.sqrt 127 := by
  sorry

end integer_sum_problem_l2016_201638


namespace sequence_representation_l2016_201649

theorem sequence_representation (a : ℕ → ℝ) 
  (h0 : a 0 = 4)
  (h1 : a 1 = 22)
  (h_rec : ∀ n : ℕ, n ≥ 2 → a n - 6 * a (n - 1) + a (n - 2) = 0) :
  ∃ x y : ℕ → ℕ, ∀ n : ℕ, a n = (y n ^ 2 + 7) / (x n - y n) := by
sorry

end sequence_representation_l2016_201649


namespace triangle_cosine_sum_less_than_two_l2016_201606

theorem triangle_cosine_sum_less_than_two (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) : 
  Real.cos α + Real.cos β + Real.cos γ < 2 := by
  sorry

end triangle_cosine_sum_less_than_two_l2016_201606


namespace exam_max_marks_l2016_201689

theorem exam_max_marks (percentage : ℝ) (scored_marks : ℝ) (max_marks : ℝ) : 
  percentage = 0.95 → scored_marks = 285 → percentage * max_marks = scored_marks → max_marks = 300 := by
  sorry

end exam_max_marks_l2016_201689


namespace rectangle_sequence_area_stage_6_l2016_201647

/-- Calculates the area of a rectangle sequence up to a given stage -/
def rectangleSequenceArea (stage : ℕ) : ℕ :=
  let baseWidth := 2
  let length := 3
  List.range stage |>.map (fun i => (baseWidth + i) * length) |>.sum

/-- The area of the rectangle sequence at Stage 6 is 81 square inches -/
theorem rectangle_sequence_area_stage_6 :
  rectangleSequenceArea 6 = 81 := by
  sorry

end rectangle_sequence_area_stage_6_l2016_201647


namespace dinosaur_egg_theft_l2016_201657

theorem dinosaur_egg_theft (total_eggs : ℕ) (claimed_max : ℕ) : 
  total_eggs = 20 → 
  claimed_max = 7 → 
  ¬(∃ (a b : ℕ), 
    a + b + claimed_max = total_eggs ∧ 
    a ≠ b ∧ 
    a ≠ claimed_max ∧ 
    b ≠ claimed_max ∧
    a < claimed_max ∧ 
    b < claimed_max) := by
  sorry

end dinosaur_egg_theft_l2016_201657


namespace homework_problem_l2016_201684

theorem homework_problem (total_problems : ℕ) (finished_problems : ℕ) (remaining_pages : ℕ) 
  (x y : ℕ) (h1 : total_problems = 450) (h2 : finished_problems = 185) (h3 : remaining_pages = 15) :
  ∃ (odd_pages even_pages : ℕ), 
    odd_pages + even_pages = remaining_pages ∧ 
    odd_pages * x + even_pages * y = total_problems - finished_problems :=
by sorry

end homework_problem_l2016_201684


namespace binomial_18_6_l2016_201613

theorem binomial_18_6 : Nat.choose 18 6 = 18564 := by
  sorry

end binomial_18_6_l2016_201613


namespace rug_inner_length_is_four_l2016_201618

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular region -/
def area (dim : RectDimensions) : ℝ := dim.length * dim.width

/-- Represents the three regions of the rug -/
structure RugRegions where
  inner : RectDimensions
  middle : RectDimensions
  outer : RectDimensions

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem rug_inner_length_is_four
  (rug : RugRegions)
  (inner_width_two : rug.inner.width = 2)
  (middle_wider_by_two : rug.middle.length = rug.inner.length + 4 ∧ rug.middle.width = rug.inner.width + 4)
  (outer_wider_by_two : rug.outer.length = rug.middle.length + 4 ∧ rug.outer.width = rug.middle.width + 4)
  (areas_in_arithmetic_progression : isArithmeticProgression (area rug.inner) (area rug.middle) (area rug.outer)) :
  rug.inner.length = 4 := by
  sorry

end rug_inner_length_is_four_l2016_201618


namespace percentage_of_boys_from_school_A_l2016_201673

theorem percentage_of_boys_from_school_A (total_boys : ℕ) 
  (boys_A_not_science : ℕ) (science_percentage : ℚ) :
  total_boys = 400 →
  boys_A_not_science = 56 →
  science_percentage = 30 / 100 →
  (boys_A_not_science : ℚ) / ((1 - science_percentage) * total_boys) = 20 / 100 :=
by sorry

end percentage_of_boys_from_school_A_l2016_201673


namespace sequential_search_element_count_l2016_201687

/-- Represents a sequential search in an unordered array -/
structure SequentialSearch where
  n : ℕ  -- number of elements in the array
  avg_comparisons : ℕ  -- average number of comparisons

/-- 
  Theorem: If the average number of comparisons in a sequential search 
  of an unordered array is 100, and the searched element is not in the array, 
  then the number of elements in the array is 200.
-/
theorem sequential_search_element_count 
  (search : SequentialSearch) 
  (h1 : search.avg_comparisons = 100) 
  (h2 : search.avg_comparisons = search.n / 2) : 
  search.n = 200 := by
  sorry

#check sequential_search_element_count

end sequential_search_element_count_l2016_201687


namespace complex_division_result_l2016_201622

theorem complex_division_result : (3 - Complex.I) / Complex.I = -1 - 3 * Complex.I := by
  sorry

end complex_division_result_l2016_201622


namespace number_separation_l2016_201690

theorem number_separation (a b : ℝ) (h1 : a = 50) (h2 : 0.40 * a = 0.625 * b + 10) : a + b = 66 := by
  sorry

end number_separation_l2016_201690


namespace ice_cube_water_cost_l2016_201641

/-- The cost of 1 ounce of water in Pauly's ice cube production --/
theorem ice_cube_water_cost : 
  let pounds_needed : ℝ := 10
  let ounces_per_cube : ℝ := 2
  let pound_per_cube : ℝ := 1/16
  let cubes_per_hour : ℝ := 10
  let cost_per_hour : ℝ := 1.5
  let total_cost : ℝ := 56
  
  let num_cubes : ℝ := pounds_needed / pound_per_cube
  let hours_needed : ℝ := num_cubes / cubes_per_hour
  let ice_maker_cost : ℝ := hours_needed * cost_per_hour
  let water_cost : ℝ := total_cost - ice_maker_cost
  let total_ounces : ℝ := num_cubes * ounces_per_cube
  let cost_per_ounce : ℝ := water_cost / total_ounces
  
  cost_per_ounce = 0.1 := by sorry

end ice_cube_water_cost_l2016_201641


namespace marias_number_l2016_201625

theorem marias_number : ∃ x : ℚ, (((3 * x - 6) * 5) / 2 = 94) ∧ (x = 218 / 15) := by
  sorry

end marias_number_l2016_201625


namespace largest_inscribed_square_l2016_201611

theorem largest_inscribed_square (outer_square_side : ℝ) (triangle_side : ℝ) 
  (h1 : outer_square_side = 8)
  (h2 : triangle_side = outer_square_side)
  (h3 : 0 < outer_square_side) :
  let triangle_height : ℝ := triangle_side * (Real.sqrt 3) / 2
  let center_to_midpoint : ℝ := triangle_height / 2
  let inscribed_square_side : ℝ := 2 * center_to_midpoint
  inscribed_square_side = 4 * Real.sqrt 3 := by
  sorry

end largest_inscribed_square_l2016_201611


namespace acute_angle_relationship_l2016_201654

theorem acute_angle_relationship (α β : Real) : 
  0 < α ∧ α < π / 2 →
  0 < β ∧ β < π / 2 →
  2 * Real.sin α = Real.sin α * Real.cos β + Real.cos α * Real.sin β →
  α < β := by
sorry

end acute_angle_relationship_l2016_201654


namespace triangle_construction_pieces_l2016_201650

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Calculates the sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Properties of the triangle construction -/
structure TriangleConstruction where
  rodRows : ℕ
  firstRowRods : ℕ
  rodIncrease : ℕ
  connectorRows : ℕ

/-- Theorem statement for the triangle construction problem -/
theorem triangle_construction_pieces 
  (t : TriangleConstruction) 
  (h1 : t.rodRows = 10)
  (h2 : t.firstRowRods = 4)
  (h3 : t.rodIncrease = 4)
  (h4 : t.connectorRows = t.rodRows + 1) :
  arithmeticSum t.firstRowRods t.rodIncrease t.rodRows + triangularNumber t.connectorRows = 286 := by
  sorry

end triangle_construction_pieces_l2016_201650


namespace intersection_distance_l2016_201679

/-- A cube with vertices at (0,0,0), (6,0,0), (6,6,0), (0,6,0), (0,0,6), (6,0,6), (6,6,6), and (0,6,6) -/
def cube : Set (Fin 3 → ℝ) :=
  {v | ∀ i, v i ∈ ({0, 6} : Set ℝ)}

/-- The plane cutting the cube -/
def plane (x y z : ℝ) : Prop :=
  -3 * x + 10 * y + 4 * z = 30

/-- The plane cuts the edges of the cube at these points -/
axiom plane_cuts : plane 0 3 0 ∧ plane 6 0 3 ∧ plane 2 6 6

/-- The intersection point on the edge from (0,0,0) to (0,0,6) -/
def U : Fin 3 → ℝ := ![0, 0, 3]

/-- The intersection point on the edge from (6,6,0) to (6,6,6) -/
def V : Fin 3 → ℝ := ![6, 6, 3]

/-- The theorem to be proved -/
theorem intersection_distance : 
  U ∈ cube ∧ V ∈ cube ∧ plane (U 0) (U 1) (U 2) ∧ plane (V 0) (V 1) (V 2) →
  Real.sqrt (((U 0 - V 0)^2 + (U 1 - V 1)^2 + (U 2 - V 2)^2) : ℝ) = 6 * Real.sqrt 2 := by
  sorry

end intersection_distance_l2016_201679


namespace coffee_shop_revenue_l2016_201600

theorem coffee_shop_revenue : 
  let coffee_orders : ℕ := 7
  let tea_orders : ℕ := 8
  let coffee_price : ℕ := 5
  let tea_price : ℕ := 4
  let total_revenue := coffee_orders * coffee_price + tea_orders * tea_price
  total_revenue = 67 := by sorry

end coffee_shop_revenue_l2016_201600


namespace three_pipes_fill_time_l2016_201624

-- Define the tank volume and pipe rates
variable (T : ℝ) -- Tank volume
variable (X Y Z : ℝ) -- Filling rates of pipes X, Y, and Z

-- Define the conditions
axiom fill_XY : T = 3 * (X + Y)
axiom fill_XZ : T = 6 * (X + Z)
axiom fill_YZ : T = 4.5 * (Y + Z)

-- Define the theorem
theorem three_pipes_fill_time : 
  T / (X + Y + Z) = 36 / 11 := by sorry

end three_pipes_fill_time_l2016_201624


namespace intersection_distance_product_l2016_201636

/-- Given an ellipse and a hyperbola sharing the same foci, the product of distances
    from their intersection point to the foci is equal to the difference of their
    respective parameters. -/
theorem intersection_distance_product (a b m n : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a > b ∧ b > 0 ∧ m > 0 ∧ n > 0 →
  (P.1^2 / a + P.2^2 / b = 1) →
  (P.1^2 / m - P.2^2 / n = 1) →
  (∀ Q : ℝ × ℝ, Q.1^2 / a + Q.2^2 / b = 1 → dist Q F₁ + dist Q F₂ = 2 * Real.sqrt a) →
  (∀ R : ℝ × ℝ, R.1^2 / m - R.2^2 / n = 1 → |dist R F₁ - dist R F₂| = 2 * Real.sqrt m) →
  dist P F₁ * dist P F₂ = a - m :=
by sorry

end intersection_distance_product_l2016_201636


namespace unique_arrangement_l2016_201601

def is_valid_arrangement (A B C D E F : ℕ) : Prop :=
  A ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  B ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  C ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  D ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  F ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F ∧
  A + D + E = 15 ∧
  7 + C + E = 15 ∧
  9 + C + A = 15 ∧
  A + 8 + F = 15 ∧
  7 + D + F = 15 ∧
  9 + D + B = 15

theorem unique_arrangement :
  ∀ A B C D E F : ℕ,
  is_valid_arrangement A B C D E F →
  A = 4 ∧ B = 1 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 3 :=
by sorry

end unique_arrangement_l2016_201601


namespace expected_condition_sufferers_l2016_201676

theorem expected_condition_sufferers (total_sample : ℕ) (condition_rate : ℚ) : 
  total_sample = 450 → condition_rate = 1/3 → 
  (condition_rate * total_sample : ℚ) = 150 := by
sorry

end expected_condition_sufferers_l2016_201676


namespace enclosed_area_equals_eight_thirds_l2016_201660

-- Define the functions f and g
def f (x : ℝ) : ℝ := -2 * x^2 + 7 * x - 6
def g (x : ℝ) : ℝ := -x

-- Define the theorem
theorem enclosed_area_equals_eight_thirds :
  ∃ (a b : ℝ), a < b ∧
  (∫ x in a..b, f x - g x) = 8/3 :=
sorry

end enclosed_area_equals_eight_thirds_l2016_201660


namespace total_difference_across_age_groups_l2016_201620

/-- Represents the number of children in each category for an age group -/
structure AgeGroup where
  camp : ℕ
  home : ℕ

/-- Calculates the difference between camp and home for an age group -/
def difference (group : AgeGroup) : ℤ :=
  group.camp - group.home

/-- The given data for each age group -/
def group_5_10 : AgeGroup := ⟨245785, 197680⟩
def group_11_15 : AgeGroup := ⟨287279, 253425⟩
def group_16_18 : AgeGroup := ⟨285994, 217173⟩

/-- The theorem to be proved -/
theorem total_difference_across_age_groups :
  difference group_5_10 + difference group_11_15 + difference group_16_18 = 150780 := by
  sorry

end total_difference_across_age_groups_l2016_201620


namespace max_automobile_weight_l2016_201644

/-- Represents the capacity of the ferry in tons -/
def ferry_capacity : ℝ := 50

/-- Represents the maximum number of automobiles the ferry can carry -/
def max_automobiles : ℝ := 62.5

/-- Represents the conversion factor from tons to pounds -/
def tons_to_pounds : ℝ := 2000

/-- Theorem stating that the maximum weight of an automobile is 1600 pounds -/
theorem max_automobile_weight :
  (ferry_capacity * tons_to_pounds) / max_automobiles = 1600 := by
  sorry

end max_automobile_weight_l2016_201644


namespace sqrt_equation_solvability_l2016_201693

theorem sqrt_equation_solvability (a : ℝ) :
  (∃ x : ℝ, Real.sqrt x - Real.sqrt (x - a) = 2) ↔ a ≥ 4 := by
sorry

end sqrt_equation_solvability_l2016_201693


namespace students_passing_both_tests_l2016_201628

theorem students_passing_both_tests 
  (total : ℕ) 
  (passed_long_jump : ℕ) 
  (passed_shot_put : ℕ) 
  (failed_both : ℕ) 
  (h1 : total = 50) 
  (h2 : passed_long_jump = 40) 
  (h3 : passed_shot_put = 31) 
  (h4 : failed_both = 4) :
  total - failed_both = passed_long_jump + passed_shot_put - 25 := by
sorry

end students_passing_both_tests_l2016_201628


namespace negation_of_existence_l2016_201608

theorem negation_of_existence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) := by sorry

end negation_of_existence_l2016_201608


namespace dans_eggs_l2016_201669

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Dan bought -/
def dans_dozens : ℕ := 9

/-- Theorem: Dan bought 108 eggs -/
theorem dans_eggs : dans_dozens * eggs_per_dozen = 108 := by
  sorry

end dans_eggs_l2016_201669


namespace mechanic_job_hours_l2016_201643

theorem mechanic_job_hours (hourly_rate parts_cost total_bill : ℕ) : 
  hourly_rate = 45 → parts_cost = 225 → total_bill = 450 → 
  ∃ hours : ℕ, hours * hourly_rate + parts_cost = total_bill ∧ hours = 5 := by
  sorry

end mechanic_job_hours_l2016_201643


namespace standard_deviation_is_2_l2016_201663

def data : List ℝ := [51, 54, 55, 57, 53]

theorem standard_deviation_is_2 :
  let mean := (data.sum) / (data.length : ℝ)
  let variance := (data.map (λ x => (x - mean) ^ 2)).sum / (data.length : ℝ)
  Real.sqrt variance = 2 := by sorry

end standard_deviation_is_2_l2016_201663


namespace min_distance_complex_circle_l2016_201616

open Complex

theorem min_distance_complex_circle (Z : ℂ) (h : abs (Z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ (W : ℂ), abs (W + 2 - 2*I) = 1 → abs (W - 2 - 2*I) ≥ min_val :=
sorry

end min_distance_complex_circle_l2016_201616


namespace factorization_proof_l2016_201659

theorem factorization_proof (a b : ℝ) : a * b^2 - 9 * a = a * (b + 3) * (b - 3) := by
  sorry

end factorization_proof_l2016_201659


namespace sum_of_squares_l2016_201688

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 10)
  (eq2 : y^2 + 5*z = -10)
  (eq3 : z^2 + 7*x = -21) : 
  x^2 + y^2 + z^2 = 83/4 := by
  sorry

end sum_of_squares_l2016_201688


namespace jordans_income_l2016_201694

/-- Represents the state income tax calculation and Jordan's specific case -/
theorem jordans_income (q : ℝ) : 
  ∃ (I : ℝ),
    I > 35000 ∧
    0.01 * q * 35000 + 0.01 * (q + 3) * (I - 35000) = (0.01 * q + 0.004) * I ∧
    I = 40000 := by
  sorry

end jordans_income_l2016_201694


namespace f_inequality_l2016_201697

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.log (x + 1) - 1 / (1 + x^2)
  else Real.log (-x + 1) - 1 / (1 + x^2)

theorem f_inequality (a : ℝ) :
  f (a - 2) < f (4 - a^2) ↔ a > 2 ∨ a < -3 ∨ (-1 < a ∧ a < 2) :=
sorry

end f_inequality_l2016_201697


namespace inscribed_parallelogram_exists_l2016_201666

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (P : Point)
  (Q : Point)
  (R : Point)
  (S : Point)

/-- Checks if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a parallelogram is inscribed in a quadrilateral -/
def Parallelogram.inscribed_in (p : Parallelogram) (q : Quadrilateral) : Prop :=
  (p.P.on_line (Line.mk q.A.x q.A.y (-1)) ∨ p.P.on_line (Line.mk q.A.x q.B.y (-1)) ∨
   p.P.on_line (Line.mk q.B.x q.B.y (-1)) ∨ p.P.on_line (Line.mk q.C.x q.C.y (-1))) ∧
  (p.Q.on_line (Line.mk q.A.x q.A.y (-1)) ∨ p.Q.on_line (Line.mk q.A.x q.B.y (-1)) ∨
   p.Q.on_line (Line.mk q.B.x q.B.y (-1)) ∨ p.Q.on_line (Line.mk q.C.x q.C.y (-1))) ∧
  (p.R.on_line (Line.mk q.B.x q.B.y (-1)) ∨ p.R.on_line (Line.mk q.B.x q.C.y (-1)) ∨
   p.R.on_line (Line.mk q.C.x q.C.y (-1)) ∨ p.R.on_line (Line.mk q.D.x q.D.y (-1))) ∧
  (p.S.on_line (Line.mk q.A.x q.A.y (-1)) ∨ p.S.on_line (Line.mk q.C.x q.D.y (-1)) ∨
   p.S.on_line (Line.mk q.D.x q.D.y (-1)) ∨ p.S.on_line (Line.mk q.A.x q.D.y (-1)))

/-- Checks if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem inscribed_parallelogram_exists (ABCD : Quadrilateral) 
  (E : Point) (F : Point) (BF CE : Line) :
  E.on_line (Line.mk ABCD.A.x ABCD.B.y (-1)) →
  F.on_line (Line.mk ABCD.C.x ABCD.D.y (-1)) →
  ∃ (PQRS : Parallelogram),
    PQRS.inscribed_in ABCD ∧
    Line.parallel (Line.mk PQRS.P.x PQRS.Q.y (-1)) BF ∧
    Line.parallel (Line.mk PQRS.Q.x PQRS.R.y (-1)) CE :=
  sorry

end inscribed_parallelogram_exists_l2016_201666


namespace equation_solutions_l2016_201672

theorem equation_solutions : 
  (∃ (x : ℝ), (x + 8) * (x + 1) = -12 ↔ (x = -4 ∨ x = -5)) ∧
  (∃ (x : ℝ), (2 * x - 3)^2 = 5 * (2 * x - 3) ↔ (x = 3/2 ∨ x = 4)) := by
  sorry

end equation_solutions_l2016_201672


namespace line_of_sight_not_blocked_l2016_201602

/-- The curve C: y = 2x^2 -/
def C : ℝ → ℝ := λ x ↦ 2 * x^2

/-- Point A: (0, -2) -/
def A : ℝ × ℝ := (0, -2)

/-- Point B: (3, a), where a is a parameter -/
def B (a : ℝ) : ℝ × ℝ := (3, a)

/-- The line of sight from A to B(a) is not blocked by C if and only if a < 10 -/
theorem line_of_sight_not_blocked (a : ℝ) : 
  (∀ x ∈ Set.Icc A.1 (B a).1, (B a).2 - A.2 > (C x - A.2) * ((B a).1 - A.1) / (x - A.1)) ↔ 
  a < 10 :=
sorry

end line_of_sight_not_blocked_l2016_201602


namespace inequality_system_solution_l2016_201670

theorem inequality_system_solution (x : ℝ) :
  (2 * (x - 1) < x + 2) → ((x + 1) / 2 < x) → (1 < x ∧ x < 4) := by
  sorry

end inequality_system_solution_l2016_201670


namespace sqrt_20_minus_1_range_l2016_201604

theorem sqrt_20_minus_1_range : 3 < Real.sqrt 20 - 1 ∧ Real.sqrt 20 - 1 < 4 := by
  sorry

end sqrt_20_minus_1_range_l2016_201604


namespace power_of_product_l2016_201633

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by sorry

end power_of_product_l2016_201633


namespace f_neg_three_gt_f_neg_pi_l2016_201645

/-- A function f satisfying the given condition -/
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (x₁ - x₂) * (f x₁ - f x₂) > 0

/-- Theorem stating that f(-3) > f(-π) given the condition -/
theorem f_neg_three_gt_f_neg_pi (f : ℝ → ℝ) (h : StrictlyIncreasing f) :
  f (-3) > f (-Real.pi) := by
  sorry

end f_neg_three_gt_f_neg_pi_l2016_201645


namespace common_tangents_count_l2016_201629

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 9
def circle2 (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

-- Define a function to count common tangent lines
noncomputable def count_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents circle1 circle2 = 3 := by sorry

end common_tangents_count_l2016_201629


namespace larger_number_is_42_l2016_201626

theorem larger_number_is_42 (x y : ℝ) (sum_eq : x + y = 77) (ratio_eq : 5 * x = 6 * y) :
  max x y = 42 := by
sorry

end larger_number_is_42_l2016_201626


namespace age_difference_proof_l2016_201631

theorem age_difference_proof (man_age son_age : ℕ) : 
  man_age > son_age →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 26 →
  man_age - son_age = 28 := by
sorry

end age_difference_proof_l2016_201631


namespace alpha_tan_beta_gt_beta_tan_alpha_l2016_201681

theorem alpha_tan_beta_gt_beta_tan_alpha (α β : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) : 
  α * Real.tan β > β * Real.tan α := by
  sorry

end alpha_tan_beta_gt_beta_tan_alpha_l2016_201681
