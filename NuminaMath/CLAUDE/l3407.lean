import Mathlib

namespace tangent_parallel_points_l3407_340759

/-- The function f(x) = x^3 - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_parallel_points :
  ∀ x : ℝ, (f' x = 2) ↔ (x = 1 ∨ x = -1) := by sorry

end tangent_parallel_points_l3407_340759


namespace unused_types_count_l3407_340788

/-- The number of natural resources --/
def num_resources : ℕ := 6

/-- The number of developed types of nature use --/
def developed_types : ℕ := 23

/-- The number of unused types of nature use --/
def unused_types : ℕ := 2^num_resources - 1 - developed_types

theorem unused_types_count : unused_types = 40 := by
  sorry

end unused_types_count_l3407_340788


namespace hyperbola_sum_l3407_340794

theorem hyperbola_sum (h k a b c : ℝ) : 
  (h = -2) →
  (k = 0) →
  (c = Real.sqrt 34) →
  (a = 3) →
  (c^2 = a^2 + b^2) →
  (h + k + a + b = 6) := by
  sorry

end hyperbola_sum_l3407_340794


namespace lockridge_marching_band_max_size_l3407_340716

theorem lockridge_marching_band_max_size :
  ∀ n : ℕ,
  (22 * n ≡ 2 [ZMOD 24]) →
  (22 * n < 1000) →
  (∀ m : ℕ, (22 * m ≡ 2 [ZMOD 24]) → (22 * m < 1000) → (22 * m ≤ 22 * n)) →
  22 * n = 770 :=
by sorry

end lockridge_marching_band_max_size_l3407_340716


namespace apple_sale_revenue_is_408_l3407_340714

/-- Calculates the money brought in from selling apples in bags -/
def apple_sale_revenue (total_harvest : ℕ) (juice_weight : ℕ) (restaurant_weight : ℕ) (bag_weight : ℕ) (price_per_bag : ℕ) : ℕ :=
  let remaining_weight := total_harvest - juice_weight - restaurant_weight
  let num_bags := remaining_weight / bag_weight
  num_bags * price_per_bag

/-- Theorem stating that the apple sale revenue is $408 given the problem conditions -/
theorem apple_sale_revenue_is_408 :
  apple_sale_revenue 405 90 60 5 8 = 408 := by
  sorry

end apple_sale_revenue_is_408_l3407_340714


namespace interest_difference_l3407_340753

/-- Calculate the difference between compound interest and simple interest -/
theorem interest_difference (P : ℝ) (r : ℝ) (t : ℝ) (n : ℝ) : 
  P = 6000.000000000128 →
  r = 0.05 →
  t = 2 →
  n = 1 →
  let CI := P * (1 + r/n)^(n*t) - P
  let SI := P * r * t
  abs (CI - SI - 15.0000000006914) < 1e-10 := by
  sorry

end interest_difference_l3407_340753


namespace emails_left_theorem_l3407_340708

/-- Calculates the number of emails left in the inbox after a series of moves -/
def emailsLeftInInbox (initialEmails : ℕ) : ℕ :=
  let afterTrash := initialEmails / 2
  let afterWork := afterTrash - (afterTrash * 2 / 5)
  let afterPersonal := afterWork - (afterWork / 4)
  afterPersonal - (afterPersonal / 10)

/-- Theorem stating that given 500 initial emails, after a series of moves, 102 emails are left in the inbox -/
theorem emails_left_theorem :
  emailsLeftInInbox 500 = 102 := by
  sorry

end emails_left_theorem_l3407_340708


namespace integer_root_of_cubic_l3407_340756

/-- A cubic polynomial with rational coefficients -/
def cubic_polynomial (a b c : ℚ) (x : ℝ) : ℝ :=
  x^3 + a*x^2 + b*x + c

theorem integer_root_of_cubic (a b c : ℚ) :
  (∃ (r : ℤ), cubic_polynomial a b c r = 0) →
  cubic_polynomial a b c (3 - Real.sqrt 5) = 0 →
  ∃ (r : ℤ), cubic_polynomial a b c r = 0 ∧ r = 0 := by
  sorry

end integer_root_of_cubic_l3407_340756


namespace stock_market_value_l3407_340702

/-- Prove that for a stock with an 8% dividend rate and a 20% yield, the market value is 40% of the face value. -/
theorem stock_market_value (face_value : ℝ) (dividend_rate : ℝ) (yield : ℝ) :
  dividend_rate = 0.08 →
  yield = 0.20 →
  (dividend_rate * face_value) / yield = 0.40 * face_value :=
by sorry

end stock_market_value_l3407_340702


namespace inequality_solution_set_l3407_340732

def solution_set (a : ℝ) : Set ℝ :=
  if a > 0 then { x | x < -a/4 ∨ x > a/3 }
  else if a = 0 then { x | x ≠ 0 }
  else { x | x > -a/4 ∨ x < a/3 }

theorem inequality_solution_set (a : ℝ) :
  { x : ℝ | 12 * x^2 - a * x > a^2 } = solution_set a := by sorry

end inequality_solution_set_l3407_340732


namespace baby_guppies_count_is_36_l3407_340798

/-- The number of baby guppies Amber saw several days after buying 7 guppies,
    given that she later saw 9 more baby guppies and now has 52 guppies in total. -/
def baby_guppies_count : ℕ := by sorry

/-- The initial number of guppies Amber bought. -/
def initial_guppies : ℕ := 7

/-- The number of additional baby guppies Amber saw two days after the first group. -/
def additional_baby_guppies : ℕ := 9

/-- The total number of guppies Amber has now. -/
def total_guppies : ℕ := 52

theorem baby_guppies_count_is_36 :
  baby_guppies_count = 36 ∧
  initial_guppies + baby_guppies_count + additional_baby_guppies = total_guppies := by
  sorry

end baby_guppies_count_is_36_l3407_340798


namespace inequality_proof_l3407_340769

theorem inequality_proof (a : ℝ) : (a^2 + a + 2) / Real.sqrt (a^2 + a + 1) ≥ 2 := by
  sorry

end inequality_proof_l3407_340769


namespace equation_d_is_quadratic_l3407_340705

/-- A polynomial equation in x is quadratic if it can be written in the form ax² + bx + c = 0,
    where a ≠ 0 and a, b, c are constants. -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation 3(x+1)² = 2(x+1) is a quadratic equation in terms of x. -/
theorem equation_d_is_quadratic :
  is_quadratic_equation (λ x => 3 * (x + 1)^2 - 2 * (x + 1)) :=
sorry

end equation_d_is_quadratic_l3407_340705


namespace statue_cost_l3407_340722

theorem statue_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) : 
  selling_price = 750 ∧ 
  profit_percentage = 35 ∧ 
  selling_price = original_cost * (1 + profit_percentage / 100) →
  original_cost = 555.56 := by
sorry

end statue_cost_l3407_340722


namespace two_digit_numbers_problem_l3407_340734

theorem two_digit_numbers_problem : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧  -- a is a two-digit number
  10 ≤ b ∧ b < 100 ∧  -- b is a two-digit number
  a = 2 * b ∧         -- a is double b
  (a / 10 ≠ b / 10) ∧ (a / 10 ≠ b % 10) ∧ (a % 10 ≠ b / 10) ∧ (a % 10 ≠ b % 10) ∧  -- no common digits
  (a / 10 + a % 10 = b / 10) ∧  -- sum of digits of a equals tens digit of b
  (a % 10 - a / 10 = b % 10) ∧  -- difference of digits of a equals ones digit of b
  a = 34 ∧ b = 17 :=
by
  sorry

end two_digit_numbers_problem_l3407_340734


namespace unique_prime_in_sequence_l3407_340762

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def seven_digit_number (B : ℕ) : ℕ := 9031511 * 10 + B

theorem unique_prime_in_sequence :
  ∃! B : ℕ, B < 10 ∧ is_prime (seven_digit_number B) ∧ seven_digit_number B = 9031517 :=
sorry

end unique_prime_in_sequence_l3407_340762


namespace sum_of_features_l3407_340791

/-- A pentagonal prism with a pyramid added to one pentagonal face -/
structure PentagonalPrismWithPyramid where
  /-- Number of faces of the original pentagonal prism -/
  prism_faces : ℕ
  /-- Number of vertices of the original pentagonal prism -/
  prism_vertices : ℕ
  /-- Number of edges of the original pentagonal prism -/
  prism_edges : ℕ
  /-- Number of faces added by the pyramid -/
  pyramid_faces : ℕ
  /-- Number of vertices added by the pyramid -/
  pyramid_vertices : ℕ
  /-- Number of edges added by the pyramid -/
  pyramid_edges : ℕ
  /-- The pentagonal prism has 7 faces -/
  prism_faces_eq : prism_faces = 7
  /-- The pentagonal prism has 10 vertices -/
  prism_vertices_eq : prism_vertices = 10
  /-- The pentagonal prism has 15 edges -/
  prism_edges_eq : prism_edges = 15
  /-- The pyramid adds 5 faces -/
  pyramid_faces_eq : pyramid_faces = 5
  /-- The pyramid adds 1 vertex -/
  pyramid_vertices_eq : pyramid_vertices = 1
  /-- The pyramid adds 5 edges -/
  pyramid_edges_eq : pyramid_edges = 5

/-- The sum of exterior faces, vertices, and edges of the resulting shape is 42 -/
theorem sum_of_features (shape : PentagonalPrismWithPyramid) :
  (shape.prism_faces + shape.pyramid_faces - 1) +
  (shape.prism_vertices + shape.pyramid_vertices) +
  (shape.prism_edges + shape.pyramid_edges) = 42 := by
  sorry

end sum_of_features_l3407_340791


namespace card_rotation_result_l3407_340761

-- Define the positions on the card
inductive Position
  | TopLeft
  | TopRight
  | BottomLeft
  | BottomRight

-- Define the colors of the triangles
inductive Color
  | LightGrey
  | DarkGrey

-- Define the card as a function mapping colors to positions
def Card := Color → Position

-- Define the initial card configuration
def initialCard : Card :=
  fun c => match c with
    | Color.LightGrey => Position.BottomRight
    | Color.DarkGrey => Position.BottomLeft

-- Define the rotation about the lower edge
def rotateLowerEdge (card : Card) : Card :=
  fun c => match card c with
    | Position.BottomLeft => Position.TopLeft
    | Position.BottomRight => Position.TopRight
    | p => p

-- Define the rotation about the right-hand edge
def rotateRightEdge (card : Card) : Card :=
  fun c => match card c with
    | Position.TopRight => Position.TopLeft
    | Position.BottomRight => Position.BottomLeft
    | p => p

-- Theorem statement
theorem card_rotation_result :
  let finalCard := rotateRightEdge (rotateLowerEdge initialCard)
  finalCard Color.LightGrey = Position.TopLeft ∧
  finalCard Color.DarkGrey = Position.TopRight := by
  sorry

end card_rotation_result_l3407_340761


namespace smallest_number_with_digit_sum_1981_l3407_340786

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that constructs a number with 1 followed by n nines -/
def oneFollowedByNines (n : ℕ) : ℕ := sorry

/-- The theorem stating that the smallest natural number whose digits sum to 1981
    is 1 followed by 220 nines -/
theorem smallest_number_with_digit_sum_1981 :
  ∀ n : ℕ, sumOfDigits n = 1981 → n ≥ oneFollowedByNines 220 :=
sorry

end smallest_number_with_digit_sum_1981_l3407_340786


namespace sqrt_12_between_3_and_4_l3407_340711

theorem sqrt_12_between_3_and_4 : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end sqrt_12_between_3_and_4_l3407_340711


namespace blue_shoes_count_l3407_340736

theorem blue_shoes_count (total : ℕ) (purple : ℕ) (h1 : total = 1250) (h2 : purple = 355) :
  ∃ (blue green : ℕ), blue + green + purple = total ∧ green = purple ∧ blue = 540 := by
  sorry

end blue_shoes_count_l3407_340736


namespace probability_of_nine_in_three_elevenths_l3407_340787

def decimal_representation (n d : ℕ) : List ℕ := sorry

def contains_digit (l : List ℕ) (digit : ℕ) : Bool := sorry

def probability_of_digit (n d digit : ℕ) : ℚ := sorry

theorem probability_of_nine_in_three_elevenths :
  probability_of_digit 3 11 9 = 0 := by sorry

end probability_of_nine_in_three_elevenths_l3407_340787


namespace coaches_schedule_lcm_l3407_340746

theorem coaches_schedule_lcm : Nat.lcm 5 (Nat.lcm 3 (Nat.lcm 9 8)) = 360 := by
  sorry

end coaches_schedule_lcm_l3407_340746


namespace equation_solutions_l3407_340728

theorem equation_solutions :
  let f (x : ℝ) := (x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)
  let g (x : ℝ) := (x - 5) * (x - 6) * (x - 5)
  ∀ x : ℝ, x ≠ 5 ∧ x ≠ 6 →
    (f x / g x = 1 ↔ x = 2 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
by sorry

end equation_solutions_l3407_340728


namespace prob_at_most_one_success_in_three_trials_l3407_340723

/-- The probability of at most one success in three independent trials -/
theorem prob_at_most_one_success_in_three_trials (p : ℝ) (h : p = 1/3) :
  p^0 * (1-p)^3 + 3 * p^1 * (1-p)^2 = 20/27 := by
  sorry

end prob_at_most_one_success_in_three_trials_l3407_340723


namespace decimal_to_fraction_l3407_340745

theorem decimal_to_fraction : (2.375 : ℚ) = 19 / 8 := by sorry

end decimal_to_fraction_l3407_340745


namespace additional_emails_per_day_l3407_340737

theorem additional_emails_per_day 
  (initial_emails_per_day : ℕ)
  (total_days : ℕ)
  (subscription_day : ℕ)
  (total_emails : ℕ)
  (h1 : initial_emails_per_day = 20)
  (h2 : total_days = 30)
  (h3 : subscription_day = 15)
  (h4 : total_emails = 675) :
  ∃ (additional_emails : ℕ),
    additional_emails = 5 ∧
    total_emails = initial_emails_per_day * subscription_day + 
      (initial_emails_per_day + additional_emails) * (total_days - subscription_day) :=
by sorry

end additional_emails_per_day_l3407_340737


namespace calculate_expression_l3407_340747

-- Define the variables and their relationships
def x : ℝ := 70 * (1 + 0.11)
def y : ℝ := x * (1 + 0.15)
def z : ℝ := y * (1 - 0.20)

-- State the theorem
theorem calculate_expression : 3 * z - 2 * x + y = 148.407 := by
  sorry

end calculate_expression_l3407_340747


namespace smallest_square_containing_circle_l3407_340727

theorem smallest_square_containing_circle (r : ℝ) (h : r = 5) :
  (2 * r) ^ 2 = 100 := by
  sorry

end smallest_square_containing_circle_l3407_340727


namespace yellow_heavier_than_green_l3407_340719

/-- The weight difference between two blocks -/
def weight_difference (yellow_weight green_weight : Real) : Real :=
  yellow_weight - green_weight

/-- Theorem: The yellow block weighs 0.2 pounds more than the green block -/
theorem yellow_heavier_than_green :
  let yellow_weight : Real := 0.6
  let green_weight : Real := 0.4
  weight_difference yellow_weight green_weight = 0.2 := by
  sorry

end yellow_heavier_than_green_l3407_340719


namespace apples_handed_out_correct_l3407_340752

/-- Represents the cafeteria's apple distribution problem -/
def apples_handed_out (initial_apples : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - (num_pies * apples_per_pie)

/-- Proves that the number of apples handed out is correct -/
theorem apples_handed_out_correct (initial_apples : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) :
  apples_handed_out initial_apples num_pies apples_per_pie =
  initial_apples - (num_pies * apples_per_pie) :=
by
  sorry

#eval apples_handed_out 47 5 4

end apples_handed_out_correct_l3407_340752


namespace consecutive_integers_sqrt_13_l3407_340755

theorem consecutive_integers_sqrt_13 (m n : ℤ) : 
  (n = m + 1) → (m < Real.sqrt 13) → (Real.sqrt 13 < n) → m * n = 12 := by
  sorry

end consecutive_integers_sqrt_13_l3407_340755


namespace gcd_75_100_l3407_340772

theorem gcd_75_100 : Nat.gcd 75 100 = 25 := by
  sorry

end gcd_75_100_l3407_340772


namespace line_through_point_parallel_to_line_line_equation_proof_l3407_340749

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel_lines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (given_line : Line) 
  (given_point : Point) 
  (result_line : Line) : Prop :=
  given_line.a = 1 ∧ 
  given_line.b = -2 ∧ 
  given_line.c = 3 ∧
  given_point.x = -1 ∧ 
  given_point.y = 3 ∧
  result_line.a = 1 ∧ 
  result_line.b = -2 ∧ 
  result_line.c = 7 →
  point_on_line given_point result_line ∧ 
  parallel_lines given_line result_line

-- The proof goes here
theorem line_equation_proof : line_through_point_parallel_to_line 
  (Line.mk 1 (-2) 3) 
  (Point.mk (-1) 3) 
  (Line.mk 1 (-2) 7) := by
  sorry

end line_through_point_parallel_to_line_line_equation_proof_l3407_340749


namespace sin_shift_l3407_340770

theorem sin_shift (x : ℝ) :
  let f (x : ℝ) := Real.sin (4 * x)
  let g (x : ℝ) := f (x + π / 12)
  let h (x : ℝ) := Real.sin (4 * x + π / 3)
  g = h :=
by sorry

end sin_shift_l3407_340770


namespace square_sum_bound_l3407_340710

theorem square_sum_bound (a b c : ℝ) :
  (|a^2 + b + c| + |a + b^2 - c| ≤ 1) → (a^2 + b^2 + c^2 < 100) := by
  sorry

end square_sum_bound_l3407_340710


namespace complex_equation_solution_l3407_340793

theorem complex_equation_solution (z : ℂ) : z / (1 - 2*I) = I → z = 2 + I := by
  sorry

end complex_equation_solution_l3407_340793


namespace total_score_l3407_340758

theorem total_score (darius_score marius_score matt_score : ℕ) : 
  darius_score = 10 →
  marius_score = darius_score + 3 →
  matt_score = darius_score + 5 →
  darius_score + marius_score + matt_score = 38 := by
sorry

end total_score_l3407_340758


namespace range_of_m_l3407_340775

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : x + 2*y - x*y = 0) 
  (h_ineq : ∀ m : ℝ, x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
  sorry

end range_of_m_l3407_340775


namespace min_value_theorem_l3407_340797

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^4 + y^4 + z^4 = 1) :
  (x^3 / (1 - x^8)) + (y^3 / (1 - y^8)) + (z^3 / (1 - z^8)) ≥ 9 * (3^(1/4)) / 8 :=
by sorry

end min_value_theorem_l3407_340797


namespace similar_triangle_perimeter_l3407_340721

theorem similar_triangle_perimeter :
  ∀ (small_side : ℝ) (small_base : ℝ) (large_base : ℝ),
    small_side > 0 → small_base > 0 → large_base > 0 →
    small_side + small_side + small_base = 7 + 7 + 12 →
    large_base = 36 →
    large_base / small_base = 36 / 12 →
    (2 * small_side * (large_base / small_base) + large_base) = 78 :=
by
  sorry

#check similar_triangle_perimeter

end similar_triangle_perimeter_l3407_340721


namespace all_sheep_with_one_peasant_l3407_340741

/-- Represents the state of sheep distribution among peasants -/
structure SheepDistribution where
  peasants : List Nat
  deriving Repr

/-- Represents a single expropriation event -/
def expropriation (dist : SheepDistribution) : SheepDistribution :=
  sorry

/-- The total number of sheep -/
def totalSheep : Nat := 128

/-- Theorem: After 7 expropriations, all sheep end up with one peasant -/
theorem all_sheep_with_one_peasant 
  (initial : SheepDistribution) 
  (h_total : initial.peasants.sum = totalSheep) 
  (h_expropriations : ∃ (d : SheepDistribution), 
    d = (expropriation^[7]) initial ∧ 
    d.peasants.length > 0) : 
  ∃ (final : SheepDistribution), 
    final = (expropriation^[7]) initial ∧ 
    final.peasants = [totalSheep] :=
sorry

end all_sheep_with_one_peasant_l3407_340741


namespace tamtam_orange_shells_l3407_340709

/-- The number of orange shells in Tamtam's collection --/
def orange_shells (total purple pink yellow blue : ℕ) : ℕ :=
  total - (purple + pink + yellow + blue)

/-- Theorem stating the number of orange shells in Tamtam's collection --/
theorem tamtam_orange_shells :
  orange_shells 65 13 8 18 12 = 14 := by
  sorry

end tamtam_orange_shells_l3407_340709


namespace simplified_fraction_sum_l3407_340764

theorem simplified_fraction_sum (a b : ℕ) (h : a = 49 ∧ b = 84) :
  let (n, d) := (a / gcd a b, b / gcd a b)
  n + d = 19 := by
  sorry

end simplified_fraction_sum_l3407_340764


namespace chewing_gum_revenue_comparison_l3407_340701

theorem chewing_gum_revenue_comparison 
  (last_year_revenue : ℝ) 
  (projected_increase_rate : ℝ) 
  (actual_decrease_rate : ℝ) 
  (h1 : projected_increase_rate = 0.25)
  (h2 : actual_decrease_rate = 0.25) :
  (last_year_revenue * (1 - actual_decrease_rate)) / 
  (last_year_revenue * (1 + projected_increase_rate)) = 0.6 := by
sorry

end chewing_gum_revenue_comparison_l3407_340701


namespace M_when_a_is_one_M_subset_N_iff_a_in_range_N_explicit_M_cases_l3407_340706

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem 1: When a = 1, M = {x | 0 < x < 2}
theorem M_when_a_is_one : M 1 = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem 2: M ⊆ N if and only if a ∈ [-2, 2]
theorem M_subset_N_iff_a_in_range : 
  ∀ a : ℝ, M a ⊆ N ↔ a ∈ Set.Icc (-2) 2 := by sorry

-- Additional helper theorems to establish the relationship
theorem N_explicit : N = Set.Icc (-1) 3 := by sorry

theorem M_cases (a : ℝ) : 
  (a < -1 → M a = {x | a + 1 < x ∧ x < 0}) ∧
  (a = -1 → M a = ∅) ∧
  (a > -1 → M a = {x | 0 < x ∧ x < a + 1}) := by sorry

end M_when_a_is_one_M_subset_N_iff_a_in_range_N_explicit_M_cases_l3407_340706


namespace dogsled_race_distance_l3407_340731

/-- The distance of a dogsled race course given the speeds and time differences of two teams. -/
theorem dogsled_race_distance
  (team_e_speed : ℝ)
  (team_a_speed_diff : ℝ)
  (team_a_time_diff : ℝ)
  (h1 : team_e_speed = 20)
  (h2 : team_a_speed_diff = 5)
  (h3 : team_a_time_diff = 3) :
  let team_a_speed := team_e_speed + team_a_speed_diff
  let team_e_time := (team_a_speed * team_a_time_diff) / (team_a_speed - team_e_speed)
  let distance := team_e_speed * team_e_time
  distance = 300 := by
  sorry

end dogsled_race_distance_l3407_340731


namespace range_of_a_minus_abs_b_l3407_340754

theorem range_of_a_minus_abs_b (a b : ℝ) (ha : 1 < a ∧ a < 3) (hb : -4 < b ∧ b < 2) :
  ∀ x : ℝ, x = a - |b| → -3 < x ∧ x < 3 := by
sorry

end range_of_a_minus_abs_b_l3407_340754


namespace cos_120_degrees_l3407_340704

theorem cos_120_degrees : Real.cos (2 * π / 3) = -(1 / 2) := by
  sorry

end cos_120_degrees_l3407_340704


namespace fraction_modification_l3407_340774

theorem fraction_modification (a b c d x y : ℚ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a + x) / (b + y) = c / d) : 
  x = (b * c - a * d + y * c) / d := by
  sorry

end fraction_modification_l3407_340774


namespace marco_running_time_l3407_340730

-- Define the constants
def laps : ℕ := 7
def track_length : ℝ := 500
def first_segment : ℝ := 150
def second_segment : ℝ := 350
def speed_first : ℝ := 3
def speed_second : ℝ := 4

-- Define the theorem
theorem marco_running_time :
  let time_first := first_segment / speed_first
  let time_second := second_segment / speed_second
  let time_per_lap := time_first + time_second
  laps * time_per_lap = 962.5 := by sorry

end marco_running_time_l3407_340730


namespace circle_radius_is_three_l3407_340773

/-- Given a circle where the product of three inches and its circumference
    equals twice its area, prove that its radius is 3 inches. -/
theorem circle_radius_is_three (r : ℝ) (h : 3 * (2 * π * r) = 2 * (π * r^2)) : r = 3 := by
  sorry

end circle_radius_is_three_l3407_340773


namespace fish_pond_population_l3407_340742

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) 
  (h1 : initial_tagged = 70)
  (h2 : second_catch = 50)
  (h3 : tagged_in_second = 2)
  (h4 : (tagged_in_second : ℚ) / second_catch = initial_tagged / total_fish) :
  total_fish = 1750 := by
  sorry

#check fish_pond_population

end fish_pond_population_l3407_340742


namespace positive_number_problem_l3407_340784

theorem positive_number_problem (n : ℝ) : n > 0 ∧ 5 * (n^2 + n) = 780 → n = 12 := by
  sorry

end positive_number_problem_l3407_340784


namespace mans_age_twice_sons_l3407_340785

/-- 
Proves that the number of years it takes for a man's age to be twice his son's age is 2,
given the initial conditions.
-/
theorem mans_age_twice_sons (
  son_age : ℕ) -- Present age of the son
  (age_diff : ℕ) -- Age difference between man and son
  (h1 : son_age = 27) -- The son's present age is 27
  (h2 : age_diff = 29) -- The man is 29 years older than his son
  : ∃ (years : ℕ), years = 2 ∧ (son_age + years + age_diff = 2 * (son_age + years)) :=
by sorry

end mans_age_twice_sons_l3407_340785


namespace homework_policy_for_25_points_l3407_340792

def homework_assignments (n : ℕ) : ℕ :=
  if n ≤ 3 then 0
  else ((n - 3 - 1) / 5 + 1)

def total_assignments (total_points : ℕ) : ℕ :=
  (List.range total_points).map homework_assignments |>.sum

theorem homework_policy_for_25_points :
  total_assignments 25 = 60 := by
  sorry

end homework_policy_for_25_points_l3407_340792


namespace park_available_spaces_l3407_340765

/-- Calculates the number of available spaces in a park given the number of benches, 
    capacity per bench, and number of people currently sitting. -/
def available_spaces (num_benches : ℕ) (capacity_per_bench : ℕ) (people_sitting : ℕ) : ℕ :=
  num_benches * capacity_per_bench - people_sitting

/-- Theorem stating that in a park with 50 benches, each with a capacity of 4 people, 
    and 80 people currently sitting, there are 120 available spaces. -/
theorem park_available_spaces : 
  available_spaces 50 4 80 = 120 := by
  sorry

end park_available_spaces_l3407_340765


namespace lamplighter_monkey_distance_l3407_340767

/-- Calculates the total distance traveled by a Lamplighter monkey under specific conditions. -/
theorem lamplighter_monkey_distance (initial_swing_speed initial_run_speed : ℝ)
  (wind_resistance_factor branch_weight_factor : ℝ)
  (run_time swing_time : ℝ) :
  initial_swing_speed = 10 →
  initial_run_speed = 15 →
  wind_resistance_factor = 0.9 →
  branch_weight_factor = 1.05 →
  run_time = 5 →
  swing_time = 10 →
  let adjusted_swing_speed := initial_swing_speed * wind_resistance_factor
  let adjusted_run_speed := initial_run_speed * branch_weight_factor
  let total_distance := adjusted_run_speed * run_time + adjusted_swing_speed * swing_time
  total_distance = 168.75 := by
sorry


end lamplighter_monkey_distance_l3407_340767


namespace z_tetromino_placement_count_l3407_340712

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)

/-- Represents a tetromino -/
structure Tetromino :=
  (shape : String)

/-- Calculates the number of ways to place a rectangle on a chessboard -/
def placeRectangle (board : Chessboard) (width : Nat) (height : Nat) : Nat :=
  (board.size - width + 1) * (board.size - height + 1)

/-- Calculates the total number of ways to place a Z-shaped tetromino on a chessboard -/
def placeZTetromino (board : Chessboard) (tetromino : Tetromino) : Nat :=
  2 * (placeRectangle board 2 3 + placeRectangle board 3 2)

/-- The main theorem stating the number of ways to place a Z-shaped tetromino on an 8x8 chessboard -/
theorem z_tetromino_placement_count :
  let board : Chessboard := ⟨8⟩
  let tetromino : Tetromino := ⟨"Z"⟩
  placeZTetromino board tetromino = 168 := by
  sorry


end z_tetromino_placement_count_l3407_340712


namespace slope_condition_l3407_340700

-- Define the function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x

-- Define the theorem
theorem slope_condition (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f m x₁ = y₁)
  (h2 : f m x₂ = y₂)
  (h3 : x₁ > x₂)
  (h4 : y₁ > y₂) :
  m > 2 := by
  sorry

end slope_condition_l3407_340700


namespace set_equality_implies_sum_l3407_340713

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2013 + b^2013 = -1 := by
  sorry

end set_equality_implies_sum_l3407_340713


namespace class_trip_problem_l3407_340733

theorem class_trip_problem (x y : ℕ) : 
  ((x + 5) * (y + 6) = x * y + 792) ∧ 
  ((x - 4) * (y + 4) = x * y - 388) → 
  (x = 27 ∧ y = 120) := by
sorry

end class_trip_problem_l3407_340733


namespace shirts_made_over_two_days_l3407_340757

/-- Calculates the total number of shirts made by an industrial machine over two days -/
theorem shirts_made_over_two_days 
  (shirts_per_minute : ℕ) -- Number of shirts the machine can make per minute
  (minutes_worked_yesterday : ℕ) -- Number of minutes the machine worked yesterday
  (shirts_made_today : ℕ) -- Number of shirts made today
  (h1 : shirts_per_minute = 6)
  (h2 : minutes_worked_yesterday = 12)
  (h3 : shirts_made_today = 14) :
  shirts_per_minute * minutes_worked_yesterday + shirts_made_today = 86 :=
by
  sorry

#check shirts_made_over_two_days

end shirts_made_over_two_days_l3407_340757


namespace projection_vector_a_on_b_l3407_340740

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 2)

theorem projection_vector_a_on_b :
  let proj_b_a := ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b
  proj_b_a = (-3/5, 6/5) := by sorry

end projection_vector_a_on_b_l3407_340740


namespace distance_condition_l3407_340783

theorem distance_condition (a : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → 
      (x - a)^2 + (1/x - a)^2 ≤ (y - a)^2 + (1/y - a)^2) ∧
    (x - a)^2 + (1/x - a)^2 = 8) ↔ 
  (a = -1 ∨ a = Real.sqrt 10) :=
sorry

end distance_condition_l3407_340783


namespace quadratic_factorization_l3407_340782

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end quadratic_factorization_l3407_340782


namespace system_solution_ratio_l3407_340760

theorem system_solution_ratio (k x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + k*y + 5*z = 0 →
  4*x + k*y - 3*z = 0 →
  3*x + 5*y - 4*z = 0 →
  x*z / (y^2) = 2/15 := by
sorry

end system_solution_ratio_l3407_340760


namespace one_nonnegative_solution_l3407_340799

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -5*x :=
by sorry

end one_nonnegative_solution_l3407_340799


namespace angle_between_skew_lines_range_l3407_340768

-- Define skew lines
structure SkewLine where
  -- We don't need to define the internal structure of a skew line for this problem

-- Define the angle between two skew lines
def angle_between_skew_lines (a b : SkewLine) : ℝ :=
  sorry -- The actual implementation is not needed for the statement

-- Theorem statement
theorem angle_between_skew_lines_range (a b : SkewLine) :
  let θ := angle_between_skew_lines a b
  0 < θ ∧ θ ≤ π/2 :=
sorry

end angle_between_skew_lines_range_l3407_340768


namespace geometric_sequence_general_term_l3407_340776

/-- A geometric sequence {a_n} satisfying the given conditions -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧
  a 6 = 8 * a 3 ∧
  a 6 = 8 * (a 2)^2

/-- The theorem stating the general term of the geometric sequence -/
theorem geometric_sequence_general_term {a : ℕ → ℝ} (h : geometric_sequence a) :
  ∀ n : ℕ, a n = 2^(n - 1) := by
  sorry

end geometric_sequence_general_term_l3407_340776


namespace sum_inequality_l3407_340796

theorem sum_inequality (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y)
  (hax : a * x ≤ 5) (hay : a * y ≤ 10) (hbx : b * x ≤ 10) (hby : b * y ≤ 10) :
  a * x + a * y + b * x + b * y ≤ 30 := by
  sorry

end sum_inequality_l3407_340796


namespace ceiling_floor_sum_l3407_340778

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by sorry

end ceiling_floor_sum_l3407_340778


namespace min_distance_sum_l3407_340779

theorem min_distance_sum (x y : ℝ) :
  Real.sqrt ((x - 1)^2 + y^2) + Real.sqrt ((x + 1)^2 + y^2) + |2 - y| ≥ 2 + Real.sqrt 3 :=
by sorry

end min_distance_sum_l3407_340779


namespace jumping_contest_l3407_340718

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) : 
  grasshopper_jump = 19 →
  frog_jump = grasshopper_jump + 10 →
  mouse_jump = grasshopper_jump + 30 →
  mouse_jump - frog_jump = 20 := by
  sorry

end jumping_contest_l3407_340718


namespace line_parameterization_l3407_340717

/-- Given a line y = 2x - 30 parameterized by (x,y) = (f(t), 20t - 10),
    prove that f(t) = 10t + 10 -/
theorem line_parameterization (f : ℝ → ℝ) :
  (∀ x y, y = 2 * x - 30 ↔ ∃ t, x = f t ∧ y = 20 * t - 10) →
  f = fun t => 10 * t + 10 := by
sorry

end line_parameterization_l3407_340717


namespace interest_rate_is_four_percent_l3407_340735

/-- Proves that the rate of interest is 4% per annum given the conditions of the problem -/
theorem interest_rate_is_four_percent (principal : ℝ) (simple_interest : ℝ) (time : ℝ) 
  (h1 : simple_interest = principal - 2080)
  (h2 : principal = 2600)
  (h3 : time = 5)
  (h4 : simple_interest = (principal * rate * time) / 100) : rate = 4 := by
  sorry

#check interest_rate_is_four_percent

end interest_rate_is_four_percent_l3407_340735


namespace symmetry_composition_l3407_340744

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry operations
def symmetryXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

def symmetryYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

-- Theorem statement
theorem symmetry_composition (a b : ℝ) :
  let M : Point2D := { x := a, y := b }
  let N : Point2D := symmetryXAxis M
  let P : Point2D := symmetryYAxis N
  let Q : Point2D := symmetryXAxis P
  let R : Point2D := symmetryYAxis Q
  R = M := by sorry

end symmetry_composition_l3407_340744


namespace area_of_rectangle_l3407_340763

-- Define the points
def P : ℝ × ℝ := (0, 4)
def Q : ℝ × ℝ := (3, 4)
def R : ℝ × ℝ := (3, 0)

-- Define the length of PR
def PR_length : ℝ := 5

-- Define the property that PQR is a right triangle
def is_right_triangle (P Q R : ℝ × ℝ) (PR_length : ℝ) : Prop :=
  (Q.1 - P.1)^2 + (R.2 - Q.2)^2 = PR_length^2

-- Define the area of the rectangle
def rectangle_area (P Q R : ℝ × ℝ) : ℝ :=
  (Q.1 - P.1) * (Q.2 - R.2)

-- The theorem to be proved
theorem area_of_rectangle : 
  is_right_triangle P Q R PR_length → rectangle_area P Q R = 12 :=
by sorry

end area_of_rectangle_l3407_340763


namespace power_product_rule_l3407_340707

theorem power_product_rule (a : ℝ) : a^3 * a^5 = a^8 := by
  sorry

end power_product_rule_l3407_340707


namespace manuscript_pages_l3407_340780

/-- Represents the typing service cost structure and manuscript details -/
structure ManuscriptTyping where
  first_time_cost : ℕ
  revision_cost : ℕ
  pages_revised_once : ℕ
  pages_revised_twice : ℕ
  total_cost : ℕ

/-- Calculates the total number of pages in the manuscript -/
def total_pages (mt : ManuscriptTyping) : ℕ :=
  sorry

/-- Theorem stating that the total number of pages is 100 -/
theorem manuscript_pages (mt : ManuscriptTyping) 
  (h1 : mt.first_time_cost = 5)
  (h2 : mt.revision_cost = 3)
  (h3 : mt.pages_revised_once = 30)
  (h4 : mt.pages_revised_twice = 20)
  (h5 : mt.total_cost = 710) :
  total_pages mt = 100 := by
  sorry

end manuscript_pages_l3407_340780


namespace benny_attended_games_l3407_340720

/-- 
Given:
- The total number of baseball games is 39.
- Benny missed 25 games.

Prove that the number of games Benny attended is 14.
-/
theorem benny_attended_games (total_games : ℕ) (missed_games : ℕ) 
  (h1 : total_games = 39)
  (h2 : missed_games = 25) :
  total_games - missed_games = 14 := by
  sorry

end benny_attended_games_l3407_340720


namespace unique_solution_and_sum_l3407_340795

theorem unique_solution_and_sum : ∃! (a b c : ℕ), 
  ({a, b, c} : Set ℕ) = {0, 1, 2} ∧ 
  (((a ≠ 2) ∧ (b ≠ 2) ∧ (c = 0)) ∨ 
   ((a = 2) ∧ (b = 2) ∧ (c ≠ 0)) ∨ 
   ((a = 2) ∧ (b ≠ 2) ∧ (c ≠ 0))) ∧
  a = 2 ∧ b = 0 ∧ c = 1 ∧ 
  100 * c + 10 * b + a = 102 :=
sorry

end unique_solution_and_sum_l3407_340795


namespace trigonometric_identity_l3407_340725

theorem trigonometric_identity (α : ℝ) :
  Real.sin (4 * α) - Real.sin (5 * α) - Real.sin (6 * α) + Real.sin (7 * α) =
  -4 * Real.sin (α / 2) * Real.sin α * Real.sin (11 * α / 2) := by
  sorry

end trigonometric_identity_l3407_340725


namespace regional_math_competition_l3407_340738

theorem regional_math_competition (initial_contestants : ℕ) : 
  (initial_contestants : ℚ) * (2/5) * (1/2) = 30 → initial_contestants = 150 := by
  sorry

end regional_math_competition_l3407_340738


namespace zero_neither_positive_nor_negative_l3407_340715

-- Define the property of being a positive number
def IsPositive (x : ℚ) : Prop := x > 0

-- Define the property of being a negative number
def IsNegative (x : ℚ) : Prop := x < 0

-- Theorem statement
theorem zero_neither_positive_nor_negative : 
  ¬(IsPositive 0) ∧ ¬(IsNegative 0) :=
sorry

end zero_neither_positive_nor_negative_l3407_340715


namespace parabola_intersection_midpoint_l3407_340743

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with vertex at origin and focus at (p/2, 0) -/
structure Parabola where
  p : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a parabola -/
def onParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Check if a point lies on a line -/
def onLine (point : Point) (line : Line) : Prop :=
  line.a * point.x + line.b * point.y + line.c = 0

/-- Check if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (p : Point) (q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

theorem parabola_intersection_midpoint 
  (parabola : Parabola)
  (C : Point)
  : 
  parabola.p = 2 ∧ C.x = 2 ∧ C.y = 1 →
  ∃ (l : Line) (M N : Point),
    l.a = 2 ∧ l.b = -1 ∧ l.c = -3 ∧
    onLine C l ∧
    onLine M l ∧ onLine N l ∧
    onParabola M parabola ∧ onParabola N parabola ∧
    isMidpoint C M N := by
  sorry

end parabola_intersection_midpoint_l3407_340743


namespace gcf_seven_eight_factorial_l3407_340777

theorem gcf_seven_eight_factorial : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcf_seven_eight_factorial_l3407_340777


namespace flight_duration_sum_l3407_340789

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDiffMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + t2.minutes - t1.minutes

/-- Theorem: Flight duration calculation -/
theorem flight_duration_sum (departureLA : Time) (arrivalNY : Time) (h m : ℕ) :
  departureLA.hours = 9 ∧ departureLA.minutes = 15 ∧
  arrivalNY.hours = 18 ∧ arrivalNY.minutes = 25 ∧
  0 < m ∧ m < 60 ∧
  timeDiffMinutes departureLA { hours := arrivalNY.hours - 3, minutes := arrivalNY.minutes, valid := sorry } = h * 60 + m →
  h + m = 16 := by
  sorry

#check flight_duration_sum

end flight_duration_sum_l3407_340789


namespace functional_equation_solution_l3407_340750

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (f x - y) + 4 * f x * y

/-- The main theorem stating that any function satisfying the equation
    must be of the form f(x) = x² + C for some constant C -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ C : ℝ, ∀ x : ℝ, f x = x^2 + C := by
  sorry

end functional_equation_solution_l3407_340750


namespace solution_set_of_inequality_system_l3407_340726

def inequality_system (x : ℝ) : Prop :=
  2 * x - 4 ≥ 2 ∧ 3 * x - 7 < 8

theorem solution_set_of_inequality_system :
  {x : ℝ | inequality_system x} = {x : ℝ | 3 ≤ x ∧ x < 5} :=
by sorry

end solution_set_of_inequality_system_l3407_340726


namespace value_of_expression_l3407_340748

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 4 = 22 := by
  sorry

end value_of_expression_l3407_340748


namespace lemonade_parts_l3407_340739

/-- Punch mixture properties -/
structure PunchMixture where
  lemonade : ℕ
  cranberry : ℕ
  total_volume : ℕ
  ratio : ℚ

/-- Theorem: The number of parts of lemonade in the punch mixture is 12 -/
theorem lemonade_parts (p : PunchMixture) : p.lemonade = 12 :=
  by
  have h1 : p.cranberry = p.lemonade + 18 := sorry
  have h2 : p.ratio = p.lemonade / 5 := sorry
  have h3 : p.total_volume = 72 := sorry
  have h4 : p.lemonade + p.cranberry = p.total_volume := sorry
  sorry

#check lemonade_parts

end lemonade_parts_l3407_340739


namespace sally_picked_42_peaches_l3407_340703

/-- The number of peaches Sally picked -/
def peaches_picked (initial peaches_now : ℕ) : ℕ := peaches_now - initial

/-- Theorem stating that Sally picked 42 peaches -/
theorem sally_picked_42_peaches : peaches_picked 13 55 = 42 := by
  sorry

end sally_picked_42_peaches_l3407_340703


namespace circle_radius_proof_l3407_340724

theorem circle_radius_proof (r : ℝ) : r > 0 → 3 * (2 * π * r) = 2 * (π * r^2) → r = 3 := by
  sorry

end circle_radius_proof_l3407_340724


namespace max_water_depth_l3407_340766

/-- The maximum water depth during a swim, given the swimmer's height,
    the ratio of water depth to height, and the wave increase percentage. -/
theorem max_water_depth
  (height : ℝ)
  (depth_ratio : ℝ)
  (wave_increase : ℝ)
  (h1 : height = 6)
  (h2 : depth_ratio = 10)
  (h3 : wave_increase = 0.25)
  : height * depth_ratio * (1 + wave_increase) = 75 := by
  sorry

#check max_water_depth

end max_water_depth_l3407_340766


namespace soccer_substitutions_l3407_340771

theorem soccer_substitutions (total_players : ℕ) (starting_players : ℕ) (non_playing_players : ℕ) :
  total_players = 24 →
  starting_players = 11 →
  non_playing_players = 7 →
  ∃ (first_half_subs : ℕ),
    first_half_subs = 2 ∧
    total_players = starting_players + first_half_subs + 2 * first_half_subs + non_playing_players :=
by sorry

end soccer_substitutions_l3407_340771


namespace solution_set_theorem_l3407_340781

-- Define the solution sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem solution_set_theorem (a b : ℝ) : 
  ({x : ℝ | x^2 + a*x + b < 0} = A_intersect_B) → a + b = -3 :=
by sorry

end solution_set_theorem_l3407_340781


namespace largest_multiple_of_15_under_500_l3407_340790

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n * 15 < 500 → n * 15 ≤ 495 :=
by
  sorry

end largest_multiple_of_15_under_500_l3407_340790


namespace expression_value_l3407_340751

theorem expression_value : ((2^2 - 3*2 - 10) / (2 - 5)) = 4 := by
  sorry

end expression_value_l3407_340751


namespace circle_diameter_from_area_and_custom_definition_l3407_340729

/-- The diameter of a circle with area 400π cm² is 1600 cm, given that the diameter is defined as four times the square of the radius. -/
theorem circle_diameter_from_area_and_custom_definition :
  ∀ (r d : ℝ),
  r > 0 →
  400 * Real.pi = Real.pi * r^2 →
  d = 4 * r^2 →
  d = 1600 := by
sorry

end circle_diameter_from_area_and_custom_definition_l3407_340729
