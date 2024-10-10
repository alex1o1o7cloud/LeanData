import Mathlib

namespace circle_proof_l2495_249542

/-- The equation of the given circle -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 6*y + 5 = 0

/-- The equation of the circle we want to prove -/
def our_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

/-- Point A -/
def point_A : ℝ × ℝ := (4, -1)

/-- Point B -/
def point_B : ℝ × ℝ := (1, 2)

/-- Two circles are tangent if they intersect at exactly one point -/
def tangent (c1 c2 : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop :=
  c1 p.1 p.2 ∧ c2 p.1 p.2 ∧ ∀ x y, c1 x y ∧ c2 x y → (x, y) = p

theorem circle_proof :
  our_circle point_A.1 point_A.2 ∧
  tangent given_circle our_circle point_B :=
sorry

end circle_proof_l2495_249542


namespace michaels_brothers_age_multiple_prove_michaels_brothers_age_multiple_l2495_249526

theorem michaels_brothers_age_multiple : ℕ → ℕ → ℕ → Prop :=
  fun michael_age older_brother_age younger_brother_age =>
    let k : ℚ := (older_brother_age - 1 : ℚ) / (michael_age - 1 : ℚ)
    younger_brother_age = 5 ∧
    older_brother_age = 3 * younger_brother_age ∧
    michael_age + older_brother_age + younger_brother_age = 28 ∧
    older_brother_age = k * (michael_age - 1) + 1 →
    k = 2

theorem prove_michaels_brothers_age_multiple :
  ∃ (michael_age older_brother_age younger_brother_age : ℕ),
    michaels_brothers_age_multiple michael_age older_brother_age younger_brother_age :=
by
  sorry

end michaels_brothers_age_multiple_prove_michaels_brothers_age_multiple_l2495_249526


namespace probability_between_lines_l2495_249508

-- Define the lines
def line_l (x : ℝ) : ℝ := -2 * x + 8
def line_m (x : ℝ) : ℝ := -3 * x + 9

-- Define the region of interest
def region_of_interest (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ y ≤ line_l x ∧ y ≥ line_m x

-- Define the area calculation function
def area_between_lines : ℝ := 2.5

-- Define the total area under line l in the first quadrant
def total_area : ℝ := 16

-- Theorem statement
theorem probability_between_lines :
  (area_between_lines / total_area) = 0.15625 :=
sorry

end probability_between_lines_l2495_249508


namespace sqrt_193_between_13_and_14_l2495_249554

theorem sqrt_193_between_13_and_14 : 13 < Real.sqrt 193 ∧ Real.sqrt 193 < 14 := by
  sorry

end sqrt_193_between_13_and_14_l2495_249554


namespace monotonic_decreasing_interval_f_l2495_249537

/-- The function f(x) = -x^2 + 2x + 1 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 1

/-- The monotonic decreasing interval of f(x) = -x^2 + 2x + 1 is [1, +∞) -/
theorem monotonic_decreasing_interval_f :
  {x : ℝ | ∀ y, x ≤ y → f x ≥ f y} = Set.Ici 1 := by sorry

end monotonic_decreasing_interval_f_l2495_249537


namespace prob_diff_colors_our_deck_l2495_249515

/-- A deck of cards -/
structure Deck :=
  (red : ℕ)
  (black : ℕ)

/-- The probability of drawing two cards of different colors -/
def prob_diff_colors (d : Deck) : ℚ :=
  if d.red + d.black < 2 then 0
  else (d.red * d.black) / ((d.red + d.black) * (d.red + d.black - 1) / 2)

/-- The deck in our problem -/
def our_deck : Deck := ⟨2, 2⟩

theorem prob_diff_colors_our_deck :
  prob_diff_colors our_deck = 2/3 := by
  sorry

#eval prob_diff_colors our_deck

end prob_diff_colors_our_deck_l2495_249515


namespace stella_stamps_count_l2495_249535

/-- The number of stamps in Stella's album -/
def total_stamps : ℕ :=
  let total_pages : ℕ := 50
  let first_pages : ℕ := 10
  let stamps_per_row : ℕ := 30
  let rows_per_first_page : ℕ := 5
  let stamps_per_remaining_page : ℕ := 50
  
  let stamps_in_first_pages : ℕ := first_pages * rows_per_first_page * stamps_per_row
  let remaining_pages : ℕ := total_pages - first_pages
  let stamps_in_remaining_pages : ℕ := remaining_pages * stamps_per_remaining_page
  
  stamps_in_first_pages + stamps_in_remaining_pages

/-- Theorem stating that the total number of stamps in Stella's album is 3500 -/
theorem stella_stamps_count : total_stamps = 3500 := by
  sorry

end stella_stamps_count_l2495_249535


namespace scientific_notation_2310000_l2495_249514

theorem scientific_notation_2310000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 2310000 = a * (10 : ℝ) ^ n ∧ a = 2.31 ∧ n = 6 := by
  sorry

end scientific_notation_2310000_l2495_249514


namespace units_digit_of_p_plus_5_l2495_249575

/-- A function that returns the units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A predicate that checks if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

theorem units_digit_of_p_plus_5 (p : ℕ) 
  (h1 : p > 0)
  (h2 : isEven p)
  (h3 : unitsDigit p > 0)
  (h4 : unitsDigit (p^3) - unitsDigit (p^2) = 0) :
  unitsDigit (p + 5) = 1 := by
sorry

end units_digit_of_p_plus_5_l2495_249575


namespace science_price_relation_spending_condition_min_literature_books_proof_l2495_249534

-- Define the prices of books
def literature_price : ℚ := 5
def science_price : ℚ := 15 / 2

-- Define the condition that science book price is half higher than literature book price
theorem science_price_relation : science_price = literature_price * (3/2) := by sorry

-- Define the spending condition
theorem spending_condition (lit_count science_count : ℕ) : 
  lit_count * literature_price + science_count * science_price = 15 ∧ lit_count = science_count + 1 := by sorry

-- Define the budget condition
def total_books : ℕ := 10
def total_budget : ℚ := 60

-- Define the function to calculate the minimum number of literature books
def min_literature_books : ℕ := 6

-- Theorem to prove the minimum number of literature books
theorem min_literature_books_proof :
  ∀ m : ℕ, m * literature_price + (total_books - m) * science_price ≤ total_budget → m ≥ min_literature_books := by sorry

end science_price_relation_spending_condition_min_literature_books_proof_l2495_249534


namespace product_inequality_l2495_249506

theorem product_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  (1 + x) * (1 + y) * (1 + z) ≥ 2 * (1 + (y / x)^(1/3) + (z / y)^(1/3) + (x / z)^(1/3)) := by
  sorry

end product_inequality_l2495_249506


namespace polynomial_factorization_l2495_249597

theorem polynomial_factorization (m n : ℝ) :
  (m + n)^2 - 10*(m + n) + 25 = (m + n - 5)^2 := by
  sorry

end polynomial_factorization_l2495_249597


namespace car_speed_proof_l2495_249596

/-- Proves that a car covering 400 meters in 12 seconds has a speed of 120 kilometers per hour -/
theorem car_speed_proof (distance : ℝ) (time : ℝ) (speed_mps : ℝ) (speed_kmph : ℝ) : 
  distance = 400 ∧ time = 12 ∧ speed_mps = distance / time ∧ speed_kmph = speed_mps * 3.6 →
  speed_kmph = 120 := by
  sorry

end car_speed_proof_l2495_249596


namespace rectangle_shorter_side_l2495_249544

/-- A rectangle with perimeter 46 and area 108 has a shorter side of 9 -/
theorem rectangle_shorter_side : ∃ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧  -- positive sides
  a ≥ b ∧          -- a is the longer side
  2 * (a + b) = 46 ∧  -- perimeter condition
  a * b = 108 ∧    -- area condition
  b = 9 :=
by sorry

end rectangle_shorter_side_l2495_249544


namespace smaller_root_of_quadratic_l2495_249566

theorem smaller_root_of_quadratic (x : ℚ) : 
  (x - 4/5) * (x - 4/5) + (x - 4/5) * (x - 2/3) + 1/15 = 0 →
  (x = 11/15 ∨ x = 4/5) ∧ 11/15 < 4/5 :=
sorry

end smaller_root_of_quadratic_l2495_249566


namespace geometric_sequence_property_l2495_249538

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r ≠ 0 ∧ a (n + 1) = a n * r

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 1 * a 9 = 16) :
  a 2 * a 5 * a 8 = 64 := by
  sorry

end geometric_sequence_property_l2495_249538


namespace sum_of_cubes_equals_cube_l2495_249503

theorem sum_of_cubes_equals_cube : 57^6 + 95^6 + 109^6 = 228^6 := by
  sorry

end sum_of_cubes_equals_cube_l2495_249503


namespace problem_solution_l2495_249564

theorem problem_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := by
  sorry

end problem_solution_l2495_249564


namespace sum_of_squared_coefficients_l2495_249518

/-- The original polynomial expression -/
def original_expression (x : ℝ) : ℝ := 3 * (x^3 - 4*x^2 + 3*x - 1) - 5 * (2*x^3 - x^2 + x + 2)

/-- The simplified polynomial expression -/
def simplified_expression (x : ℝ) : ℝ := -7*x^3 - 7*x^2 + 4*x - 13

/-- Coefficients of the simplified expression -/
def coefficients : List ℝ := [-7, -7, 4, -13]

theorem sum_of_squared_coefficients :
  (original_expression = simplified_expression) →
  (List.sum (List.map (λ x => x^2) coefficients) = 283) := by
  sorry

end sum_of_squared_coefficients_l2495_249518


namespace evenness_of_k_l2495_249555

theorem evenness_of_k (a b n k : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) (hk : 0 < k)
  (h1 : 2^n - 1 = a * b)
  (h2 : (a * b + a - b - 1) % 2^k = 0)
  (h3 : (a * b + a - b - 1) % 2^(k+1) ≠ 0) :
  Even k := by sorry

end evenness_of_k_l2495_249555


namespace polynomial_remainder_l2495_249577

theorem polynomial_remainder : ∀ x : ℝ, 
  (4 * x^8 - 3 * x^6 - 6 * x^4 + x^3 + 5 * x^2 - 9) = 
  (x - 1) * (4 * x^7 + 4 * x^6 + x^5 - 2 * x^4 - 2 * x^3 + 4 * x^2 + 4 * x + 4) + (-9) := by
  sorry

end polynomial_remainder_l2495_249577


namespace find_a_min_value_g_l2495_249520

-- Define the function f(x) = |x - a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1: Prove that a = 3
theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a x ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) → a = 3 := by sorry

-- Define g(x) = f(2x) + f(x + 2) where f(x) = |x - 3|
def g (x : ℝ) : ℝ := |2*x - 3| + |x + 2 - 3|

-- Theorem 2: Prove that the minimum value of g(x) is 1/2
theorem min_value_g : 
  ∀ x : ℝ, g x ≥ 1/2 ∧ ∃ y : ℝ, g y = 1/2 := by sorry

end find_a_min_value_g_l2495_249520


namespace rectangle_area_problem_l2495_249585

theorem rectangle_area_problem (square_area : Real) (rectangle_breadth : Real) :
  square_area = 784 →
  rectangle_breadth = 5 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := circle_radius / 4
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area = 35 := by sorry

end rectangle_area_problem_l2495_249585


namespace sin_difference_product_l2495_249576

theorem sin_difference_product (a b : ℝ) :
  Real.sin (2 * a + b) - Real.sin b = 2 * Real.cos (a + b) * Real.sin a := by
  sorry

end sin_difference_product_l2495_249576


namespace correct_matching_probability_l2495_249582

/-- The number of Earthly Branches and zodiac signs -/
def n : ℕ := 12

/-- The number of cards selected from each color -/
def k : ℕ := 3

/-- The probability of correctly matching the selected cards -/
def matching_probability : ℚ := 1 / (n.choose k)

/-- Theorem stating the probability of correctly matching the selected cards -/
theorem correct_matching_probability :
  matching_probability = 1 / 220 :=
by sorry

end correct_matching_probability_l2495_249582


namespace charge_account_interest_l2495_249524

/-- Calculates the total amount owed after one year given an initial charge and simple annual interest rate -/
def total_amount_owed (initial_charge : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_charge * (1 + interest_rate)

/-- Proves that the total amount owed after one year for a $60 charge at 6% simple annual interest is $63.60 -/
theorem charge_account_interest :
  total_amount_owed 60 0.06 = 63.60 := by
  sorry

end charge_account_interest_l2495_249524


namespace shaded_area_square_circles_l2495_249557

/-- The shaded area between a square and four circles --/
theorem shaded_area_square_circles (s : ℝ) (r : ℝ) (h1 : s = 10) (h2 : r = 3 * Real.sqrt 3) :
  s^2 - 4 * (π * r^2 / 4) - 8 * (s / 2 * Real.sqrt ((3 * Real.sqrt 3)^2 - (s / 2)^2) / 2) = 
    100 - 27 * π - 20 * Real.sqrt 2 :=
by sorry

end shaded_area_square_circles_l2495_249557


namespace arithmetic_sequence_difference_l2495_249581

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_difference 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 5 = 8) 
  (h2 : seq.S 3 = 6) : 
  common_difference seq = 2 := by
sorry

end arithmetic_sequence_difference_l2495_249581


namespace line_y_coordinate_at_x_10_l2495_249527

/-- Given a line passing through points (4, 0) and (-4, -4), 
    prove that the y-coordinate of the point on this line with x-coordinate 10 is 3. -/
theorem line_y_coordinate_at_x_10 (L : Set (ℝ × ℝ)) :
  ((4 : ℝ), 0) ∈ L →
  ((-4 : ℝ), -4) ∈ L →
  ∃ m b : ℝ, ∀ x y : ℝ, (x, y) ∈ L ↔ y = m * x + b →
  (10, 3) ∈ L := by
sorry

end line_y_coordinate_at_x_10_l2495_249527


namespace sin_to_cos_l2495_249574

theorem sin_to_cos (α : Real) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := by
  sorry

end sin_to_cos_l2495_249574


namespace smaller_number_problem_l2495_249591

theorem smaller_number_problem (x y : ℤ) (h1 : x + y = 62) (h2 : y = x + 12) : x = 25 := by
  sorry

end smaller_number_problem_l2495_249591


namespace coefficient_of_x_cubed_l2495_249579

def polynomial (x : ℝ) : ℝ := 4 * (x^3 - 2*x^2) + 3 * (x^2 - x^3 + 2*x^4) - 5 * (x^4 - 2*x^3)

theorem coefficient_of_x_cubed :
  ∃ (a b c d : ℝ), ∀ x, polynomial x = a*x^4 + 11*x^3 + b*x^2 + c*x + d :=
by sorry

end coefficient_of_x_cubed_l2495_249579


namespace set_inclusion_implies_a_values_l2495_249532

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 ≠ 1}

-- State the theorem
theorem set_inclusion_implies_a_values :
  ∀ a : ℝ, (B a ⊆ A) ↔ (a ≤ -1 ∨ a = 1) :=
sorry

end set_inclusion_implies_a_values_l2495_249532


namespace f_derivative_at_pi_half_l2495_249593

noncomputable def f (x : ℝ) : ℝ := x / Real.sin x

theorem f_derivative_at_pi_half :
  deriv f (π / 2) = 1 := by
  sorry

end f_derivative_at_pi_half_l2495_249593


namespace car_speed_proof_l2495_249513

theorem car_speed_proof (reduced_speed : ℝ) (distance : ℝ) (time : ℝ) (actual_speed : ℝ) : 
  reduced_speed = 5 / 7 * actual_speed →
  distance = 42 →
  time = 42 / 25 →
  reduced_speed = distance / time →
  actual_speed = 35 := by
sorry


end car_speed_proof_l2495_249513


namespace linear_function_properties_l2495_249560

/-- A linear function passing through two given points -/
structure LinearFunction where
  b : ℝ
  k : ℝ
  point1 : b * (-2) + k = -3
  point2 : b * 1 + k = 3

/-- Theorem stating the properties of the linear function -/
theorem linear_function_properties (f : LinearFunction) :
  f.k = 1 ∧ f.b = 2 ∧ f.b * (-2) + f.k ≠ 3 := by sorry

end linear_function_properties_l2495_249560


namespace solution_set_is_open_interval_l2495_249500

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_increasing_neg : ∀ x y, x ≤ y → y ≤ 0 → f x ≤ f y

-- Define the solution set
def solution_set := {x : ℝ | f (3 - 2*x) > f 1}

-- State the theorem
theorem solution_set_is_open_interval :
  solution_set = Set.Ioo 1 2 := by sorry

end solution_set_is_open_interval_l2495_249500


namespace distribute_and_simplify_l2495_249572

theorem distribute_and_simplify (a b : ℝ) : 3*a*(2*a - b) = 6*a^2 - 3*a*b := by
  sorry

end distribute_and_simplify_l2495_249572


namespace tournament_committee_count_l2495_249517

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members the host team contributes to the committee -/
def host_contribution : ℕ := 3

/-- The number of members each non-host team contributes to the committee -/
def non_host_contribution : ℕ := 2

/-- The total number of members in the tournament committee -/
def committee_size : ℕ := 11

/-- The number of possible tournament committees -/
def num_committees : ℕ := 172043520

theorem tournament_committee_count :
  (num_teams * (Nat.choose team_size host_contribution) * 
   (Nat.choose team_size non_host_contribution)^(num_teams - 1)) = num_committees := by
  sorry

end tournament_committee_count_l2495_249517


namespace union_of_M_and_N_l2495_249588

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by
  sorry

end union_of_M_and_N_l2495_249588


namespace max_discount_l2495_249584

/-- Given a product with a marked price and markup percentage, calculate the maximum discount --/
theorem max_discount (marked_price : ℝ) (markup_percent : ℝ) (min_markup_percent : ℝ) : 
  marked_price = 360 ∧ 
  markup_percent = 0.8 ∧ 
  min_markup_percent = 0.2 →
  marked_price - (marked_price / (1 + markup_percent) * (1 + min_markup_percent)) = 120 := by
  sorry

end max_discount_l2495_249584


namespace green_pill_cost_calculation_l2495_249578

def green_pill_cost (total_cost : ℚ) (days : ℕ) (green_daily : ℕ) (pink_daily : ℕ) : ℚ :=
  (total_cost / days + 2 * pink_daily) / (green_daily + pink_daily)

theorem green_pill_cost_calculation :
  let total_cost : ℚ := 600
  let days : ℕ := 10
  let green_daily : ℕ := 2
  let pink_daily : ℕ := 1
  green_pill_cost total_cost days green_daily pink_daily = 62/3 := by
sorry

end green_pill_cost_calculation_l2495_249578


namespace tangent_parallel_to_BC_l2495_249559

/-- Two circles in a plane -/
structure TwoCircles where
  circle1 : Set (ℝ × ℝ)
  circle2 : Set (ℝ × ℝ)

/-- Points of intersection and other significant points -/
structure CirclePoints (tc : TwoCircles) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  on_circle1_P : P ∈ tc.circle1
  on_circle2_P : P ∈ tc.circle2
  on_circle1_Q : Q ∈ tc.circle1
  on_circle2_Q : Q ∈ tc.circle2
  on_circle1_A : A ∈ tc.circle1
  on_circle2_B : B ∈ tc.circle2
  on_circle2_C : C ∈ tc.circle2

/-- Line represented by two points -/
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Tangent line to a circle at a point -/
def TangentLine (circle : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Two lines are parallel -/
def Parallel (l1 l2 : Set (ℝ × ℝ)) : Prop := sorry

/-- Main theorem -/
theorem tangent_parallel_to_BC (tc : TwoCircles) (cp : CirclePoints tc) : 
  Parallel (TangentLine tc.circle1 cp.A) (Line cp.B cp.C) := by sorry

end tangent_parallel_to_BC_l2495_249559


namespace inequality_proof_l2495_249507

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a + b + c ≥ (a * (b * c + c + 1)) / (c * a + a + 1) +
              (b * (c * a + a + 1)) / (a * b + b + 1) +
              (c * (a * b + b + 1)) / (b * c + c + 1) := by
  sorry

end inequality_proof_l2495_249507


namespace smallest_five_digit_multiple_of_18_l2495_249571

theorem smallest_five_digit_multiple_of_18 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ 18 ∣ n → 10008 ≤ n :=
by
  sorry

end smallest_five_digit_multiple_of_18_l2495_249571


namespace speed_conversion_correct_l2495_249558

/-- Converts meters per second to kilometers per hour -/
def mps_to_kmh (speed_mps : ℚ) : ℚ :=
  speed_mps * 3.6

theorem speed_conversion_correct : 
  mps_to_kmh (13/48) = 39/40 :=
by sorry

end speed_conversion_correct_l2495_249558


namespace trapezoid_area_division_l2495_249587

/-- Given a trapezoid with specific properties, prove that the largest integer less than x^2/50 is 72 -/
theorem trapezoid_area_division (b : ℝ) (h : ℝ) (x : ℝ) : 
  b > 0 ∧ h > 0 ∧
  (b + 12.5) / (b + 37.5) = 3 / 5 ∧
  x > 0 ∧
  (25 + x) * ((x - 25) / 50) = 50 ∧
  x^2 - 75*x + 3125 = 0 →
  ⌊x^2 / 50⌋ = 72 :=
by sorry

end trapezoid_area_division_l2495_249587


namespace f_properties_l2495_249568

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x + 8

-- Define the interval
def interval : Set ℝ := Set.Icc (-1) 3

-- Theorem for extreme values and max/min in the interval
theorem f_properties :
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x) ∧
  (∀ (x : ℝ), x ∈ interval → f x ≤ 14) ∧
  (∀ (x : ℝ), x ∈ interval → f x ≥ -15) ∧
  (f 1 = 13 ∧ f 2 = 12) ∧
  (∀ (x : ℝ), f x ≥ 12 → x = 1 ∨ x = 2) :=
by sorry

end f_properties_l2495_249568


namespace divisor_log_sum_l2495_249586

theorem divisor_log_sum (n : ℕ) : (n * (n + 1)^2) / 2 = 1080 ↔ n = 12 := by sorry

end divisor_log_sum_l2495_249586


namespace circle_center_and_radius_l2495_249590

/-- Given a circle and a line with specific properties, prove the center and radius of the circle -/
theorem circle_center_and_radius 
  (x y : ℝ) 
  (m : ℝ) 
  (circle_eq : x^2 + y^2 + x - 6*y + m = 0) 
  (line_eq : x + 2*y - 3 = 0) 
  (P Q : ℝ × ℝ) 
  (intersect : (P.1^2 + P.2^2 + P.1 - 6*P.2 + m = 0 ∧ P.1 + 2*P.2 - 3 = 0) ∧ 
               (Q.1^2 + Q.2^2 + Q.1 - 6*Q.2 + m = 0 ∧ Q.1 + 2*Q.2 - 3 = 0)) 
  (perpendicular : P.1 * Q.1 + P.2 * Q.2 = 0) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (-1/2, 3) ∧ radius = 5/2 := by
  sorry

end circle_center_and_radius_l2495_249590


namespace closest_integer_to_cube_root_l2495_249504

theorem closest_integer_to_cube_root : ∃ (n : ℤ), 
  n = 10 ∧ ∀ (m : ℤ), |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| :=
by sorry

end closest_integer_to_cube_root_l2495_249504


namespace right_triangle_seven_units_contains_28_triangles_l2495_249570

/-- Represents a right-angled triangle on a grid -/
structure GridTriangle where
  leg_length : ℕ
  is_right_angled : Bool

/-- Calculates the maximum number of triangles that can be formed within a GridTriangle -/
def max_triangles (t : GridTriangle) : ℕ :=
  if t.is_right_angled && t.leg_length > 0 then
    (t.leg_length + 1).choose 2
  else
    0

/-- Theorem stating that a right-angled triangle with legs of 7 units on a grid contains 28 triangles -/
theorem right_triangle_seven_units_contains_28_triangles :
  let t : GridTriangle := { leg_length := 7, is_right_angled := true }
  max_triangles t = 28 := by
  sorry

end right_triangle_seven_units_contains_28_triangles_l2495_249570


namespace distance_to_origin_l2495_249521

theorem distance_to_origin (x y : ℝ) (h1 : y = 16) (h2 : x > 3) 
  (h3 : Real.sqrt ((x - 3)^2 + (y - 6)^2) = 14) : 
  Real.sqrt (x^2 + y^2) = 19 + 12 * Real.sqrt 6 := by
  sorry

end distance_to_origin_l2495_249521


namespace paper_folding_l2495_249509

theorem paper_folding (n : ℕ) : 2^n = 128 → n = 7 := by sorry

end paper_folding_l2495_249509


namespace rental_cost_calculation_l2495_249589

/-- Calculates the total cost of renting a truck given the daily rate, mileage rate, number of days, and miles driven. -/
def total_rental_cost (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mileage_rate * miles

/-- Proves that the total rental cost for the given conditions is $230. -/
theorem rental_cost_calculation :
  let daily_rate : ℚ := 35
  let mileage_rate : ℚ := 1/4
  let days : ℕ := 3
  let miles : ℕ := 500
  total_rental_cost daily_rate mileage_rate days miles = 230 := by
sorry


end rental_cost_calculation_l2495_249589


namespace sum_of_fractions_l2495_249556

theorem sum_of_fractions : 
  let a := 1 + 3 + 5
  let b := 2 + 4 + 6
  (a / b) + (b / a) = 25 / 12 := by
  sorry

end sum_of_fractions_l2495_249556


namespace total_paths_is_4312_l2495_249529

/-- Represents the number of paths between different points in the lattice --/
structure LatticePathCounts where
  a_to_red1 : Nat
  a_to_red2 : Nat
  red1_to_blue : Nat
  red2_to_blue : Nat
  blue12_to_green : Nat
  blue34_to_green : Nat
  green_to_b : Nat
  green_to_c : Nat

/-- Calculates the total number of distinct paths to reach points B and C --/
def totalPaths (counts : LatticePathCounts) : Nat :=
  let paths_to_blue := counts.a_to_red1 * counts.red1_to_blue * 2 + 
                       counts.a_to_red2 * counts.red2_to_blue * 2
  let paths_to_green := paths_to_blue * (counts.blue12_to_green * 2 + counts.blue34_to_green * 2)
  paths_to_green * (counts.green_to_b + counts.green_to_c)

/-- The theorem stating that the total number of distinct paths is 4312 --/
theorem total_paths_is_4312 (counts : LatticePathCounts)
  (h1 : counts.a_to_red1 = 1)
  (h2 : counts.a_to_red2 = 2)
  (h3 : counts.red1_to_blue = 3)
  (h4 : counts.red2_to_blue = 4)
  (h5 : counts.blue12_to_green = 5)
  (h6 : counts.blue34_to_green = 6)
  (h7 : counts.green_to_b = 3)
  (h8 : counts.green_to_c = 4) :
  totalPaths counts = 4312 := by
  sorry


end total_paths_is_4312_l2495_249529


namespace problem_grid_paths_l2495_249567

/-- Represents a grid with forbidden segments -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (forbidden_segments : List (ℕ × ℕ × ℕ × ℕ))

/-- Calculates the number of paths in a grid with forbidden segments -/
def count_paths (g : Grid) : ℕ :=
  sorry

/-- The specific grid described in the problem -/
def problem_grid : Grid :=
  { rows := 4
  , cols := 7
  , forbidden_segments := [(1, 2, 3, 4), (2, 3, 5, 6)] }

/-- Theorem stating that the number of paths in the problem grid is 64 -/
theorem problem_grid_paths :
  count_paths problem_grid = 64 :=
sorry

end problem_grid_paths_l2495_249567


namespace all_propositions_incorrect_l2495_249563

/-- Represents a proposition with potential flaws in statistical reasoning --/
structure Proposition where
  hasTemporalityIgnorance : Bool
  hasSpeciesCharacteristicsIgnorance : Bool
  hasCausalityMisinterpretation : Bool
  hasIncorrectUsageRange : Bool

/-- Determines if a proposition is incorrect based on its flaws --/
def isIncorrect (p : Proposition) : Bool :=
  p.hasTemporalityIgnorance ∨ 
  p.hasSpeciesCharacteristicsIgnorance ∨ 
  p.hasCausalityMisinterpretation ∨ 
  p.hasIncorrectUsageRange

/-- Counts the number of incorrect propositions in a list --/
def countIncorrectPropositions (props : List Proposition) : Nat :=
  props.filter isIncorrect |>.length

/-- The main theorem stating that all given propositions are incorrect --/
theorem all_propositions_incorrect (props : List Proposition) 
  (h1 : props.length = 4)
  (h2 : ∀ p ∈ props, isIncorrect p = true) : 
  countIncorrectPropositions props = 4 := by
  sorry

#check all_propositions_incorrect

end all_propositions_incorrect_l2495_249563


namespace sqrt_12_similar_to_sqrt_3_l2495_249536

/-- Two quadratic radicals are similar if they have the same radicand when simplified. -/
def similar_radicals (a b : ℝ) : Prop :=
  ∃ (k₁ k₂ : ℝ), k₁ > 0 ∧ k₂ > 0 ∧ a = k₁^2 * b

/-- √12 is of the same type as √3 -/
theorem sqrt_12_similar_to_sqrt_3 : similar_radicals 12 3 := by
  sorry

end sqrt_12_similar_to_sqrt_3_l2495_249536


namespace proposition_equivalence_l2495_249525

theorem proposition_equivalence (p q : Prop) :
  (¬p → ¬q) ↔ (p → q) := by sorry

end proposition_equivalence_l2495_249525


namespace dormitory_students_l2495_249531

theorem dormitory_students (total : ℝ) (total_pos : 0 < total) : 
  let first_year := total / 2
  let second_year := total / 2
  let first_year_undeclared := 4 / 5 * first_year
  let first_year_declared := first_year - first_year_undeclared
  let second_year_declared := 1 / 3 * (first_year_declared / first_year) * second_year
  let second_year_undeclared := second_year - second_year_declared
  second_year_undeclared / total = 7 / 15 := by
sorry

end dormitory_students_l2495_249531


namespace Q_equals_N_l2495_249523

-- Define the sets Q and N
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {x : ℝ | x ≥ 1}

-- Theorem statement
theorem Q_equals_N : Q = N := by sorry

end Q_equals_N_l2495_249523


namespace unique_prime_with_prime_successors_l2495_249541

theorem unique_prime_with_prime_successors :
  ∀ p : ℕ, Prime p ∧ Prime (p + 4) ∧ Prime (p + 8) → p = 3 := by
  sorry

end unique_prime_with_prime_successors_l2495_249541


namespace number_ratio_l2495_249550

theorem number_ratio (x y z : ℝ) (k : ℝ) : 
  x = 18 →
  y = k * x →
  z = 2 * y →
  (x + y + z) / 3 = 78 →
  y / x = 4 := by
sorry

end number_ratio_l2495_249550


namespace distribution_counterexample_l2495_249595

-- Define a type for random variables
def RandomVariable := Real → Real

-- Define a type for distribution functions
def DistributionFunction := Real → Real

-- Function to get the distribution function of a random variable
def getDistribution (X : RandomVariable) : DistributionFunction := sorry

-- Function to check if two distribution functions are identical
def distributionsIdentical (F G : DistributionFunction) : Prop := sorry

-- Function to multiply two random variables
def multiply (X Y : RandomVariable) : RandomVariable := sorry

theorem distribution_counterexample :
  ∃ (ξ η ζ : RandomVariable),
    distributionsIdentical (getDistribution ξ) (getDistribution η) ∧
    ¬distributionsIdentical (getDistribution (multiply ξ ζ)) (getDistribution (multiply η ζ)) := by
  sorry

end distribution_counterexample_l2495_249595


namespace sum_of_integers_l2495_249561

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 145)
  (h2 : x * y = 40) : 
  x + y = 15 := by sorry

end sum_of_integers_l2495_249561


namespace treasure_value_proof_l2495_249583

def base7ToBase10 (n : Nat) : Nat :=
  let digits := n.digits 7
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (7 ^ i)) 0

theorem treasure_value_proof :
  let diamonds := 5643
  let silver := 1652
  let spices := 236
  (base7ToBase10 diamonds) + (base7ToBase10 silver) + (base7ToBase10 spices) = 2839 := by
  sorry

end treasure_value_proof_l2495_249583


namespace line_division_theorem_l2495_249553

/-- A line on a plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three lines divide the plane into six parts -/
def divides_into_six_parts (l1 l2 l3 : Line) : Prop :=
  sorry

/-- The set of k values that satisfy the condition -/
def k_values : Set ℝ := {0, -1, -2}

/-- Theorem stating the relationship between the lines and k values -/
theorem line_division_theorem (k : ℝ) :
  let l1 : Line := ⟨1, -2, 1⟩
  let l2 : Line := ⟨1, 0, -1⟩
  let l3 : Line := ⟨1, k, 0⟩
  divides_into_six_parts l1 l2 l3 → k ∈ k_values := by
  sorry

end line_division_theorem_l2495_249553


namespace sin_cos_sum_equals_sqrt3_over_2_l2495_249539

theorem sin_cos_sum_equals_sqrt3_over_2 :
  Real.sin (36 * π / 180) * Real.cos (24 * π / 180) +
  Real.cos (36 * π / 180) * Real.sin (156 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_sum_equals_sqrt3_over_2_l2495_249539


namespace least_common_multiple_first_ten_l2495_249547

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∧
  n = 2520 := by
sorry

end least_common_multiple_first_ten_l2495_249547


namespace specific_circle_diameter_l2495_249552

/-- A circle tangent to the y-axis and a line, passing through a specific point -/
structure TangentCircle where
  /-- The circle is tangent to the y-axis -/
  tangent_y_axis : Bool
  /-- The slope of the line the circle is tangent to -/
  line_slope : ℝ
  /-- The x-coordinate of the point the circle passes through -/
  point_x : ℝ
  /-- The y-coordinate of the point the circle passes through -/
  point_y : ℝ

/-- The diameter of a TangentCircle -/
def circle_diameter (c : TangentCircle) : Set ℝ :=
  {d : ℝ | d = 2 ∨ d = 14/3}

/-- Theorem stating the diameter of the specific TangentCircle -/
theorem specific_circle_diameter :
  let c : TangentCircle := {
    tangent_y_axis := true,
    line_slope := Real.sqrt 3 / 3,
    point_x := 2,
    point_y := Real.sqrt 3
  }
  ∀ d ∈ circle_diameter c, d = 2 ∨ d = 14/3 := by
  sorry

end specific_circle_diameter_l2495_249552


namespace irrationality_of_root_sum_squares_l2495_249530

theorem irrationality_of_root_sum_squares (a b c : ℤ) (r : ℝ) 
  (h1 : a * r^2 + b * r + c = 0)
  (h2 : a * c ≠ 0) : 
  Irrational (Real.sqrt (r^2 + c^2)) :=
by sorry

end irrationality_of_root_sum_squares_l2495_249530


namespace milk_cartons_consumption_l2495_249512

theorem milk_cartons_consumption (total_cartons : ℕ) 
  (younger_sister_fraction : ℚ) (older_sister_fraction : ℚ) :
  total_cartons = 24 →
  younger_sister_fraction = 1 / 8 →
  older_sister_fraction = 3 / 8 →
  (younger_sister_fraction * total_cartons : ℚ) = 3 ∧
  (older_sister_fraction * total_cartons : ℚ) = 9 := by
  sorry

end milk_cartons_consumption_l2495_249512


namespace min_value_sum_of_reciprocals_l2495_249546

theorem min_value_sum_of_reciprocals (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 3) :
  (1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x)) ≥ 3/4 :=
sorry

end min_value_sum_of_reciprocals_l2495_249546


namespace linear_function_property_l2495_249594

/-- A linear function is a function of the form f(x) = mx + b where m and b are constants. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- Given a linear function g where g(8) - g(3) = 15, prove that g(20) - g(3) = 51. -/
theorem linear_function_property (g : ℝ → ℝ) (h1 : LinearFunction g) (h2 : g 8 - g 3 = 15) :
  g 20 - g 3 = 51 := by
  sorry

end linear_function_property_l2495_249594


namespace company_layoff_payment_l2495_249502

theorem company_layoff_payment (total_employees : ℕ) (salary : ℕ) (layoff_fraction : ℚ) : 
  total_employees = 450 →
  salary = 2000 →
  layoff_fraction = 1/3 →
  (total_employees : ℚ) * (1 - layoff_fraction) * salary = 600000 := by
sorry

end company_layoff_payment_l2495_249502


namespace four_integer_average_l2495_249519

theorem four_integer_average (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  d = 90 ∧                 -- Largest integer is 90
  a ≥ 13 →                 -- Smallest integer is at least 13
  (a + b + c + d) / 4 = 33 -- Average is 33
  := by sorry

end four_integer_average_l2495_249519


namespace sticks_left_in_yard_l2495_249565

def sticks_picked_up : ℕ := 14
def difference : ℕ := 10

theorem sticks_left_in_yard : sticks_picked_up - difference = 4 := by
  sorry

end sticks_left_in_yard_l2495_249565


namespace units_digit_of_m_squared_plus_two_to_m_l2495_249510

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : (m^2 + 2^m) % 10 = 3 :=
sorry

end units_digit_of_m_squared_plus_two_to_m_l2495_249510


namespace equal_probabilities_decreasing_probabilities_l2495_249505

/-- Represents the probability of finding a specific item -/
def item_probability : ℝ := 0.1

/-- Represents the total number of items in the collection -/
def total_items : ℕ := 10

/-- Represents the probability that the second collection is missing exactly k items when the first collection is completed -/
noncomputable def p (k : ℕ) : ℝ := sorry

/-- The probability of missing 1 item equals the probability of missing 2 items -/
theorem equal_probabilities : p 1 = p 2 := by sorry

/-- The probabilities form a strictly decreasing sequence for k from 2 to 10 -/
theorem decreasing_probabilities : ∀ k ∈ Finset.range 9, p (k + 2) > p (k + 3) := by sorry

end equal_probabilities_decreasing_probabilities_l2495_249505


namespace equilibrium_concentration_Ca_OH_2_l2495_249599

-- Define the reaction components
inductive Species
| CaO
| H2O
| Ca_OH_2

-- Define the reaction
def reaction : List Species := [Species.CaO, Species.H2O, Species.Ca_OH_2]

-- Define the equilibrium constant
def Kp : ℝ := 0.02

-- Define the equilibrium concentration function
noncomputable def equilibrium_concentration (s : Species) : ℝ :=
  match s with
  | Species.CaO => 0     -- Not applicable (solid)
  | Species.H2O => 0     -- Not applicable (liquid)
  | Species.Ca_OH_2 => Kp -- Equilibrium concentration equals Kp

-- Theorem statement
theorem equilibrium_concentration_Ca_OH_2 :
  equilibrium_concentration Species.Ca_OH_2 = Kp := by sorry

end equilibrium_concentration_Ca_OH_2_l2495_249599


namespace complex_distance_sum_l2495_249511

theorem complex_distance_sum (z : ℂ) (h : Complex.abs (z - (3 - 2*I)) = 4) :
  Complex.abs (z + (1 - I))^2 + Complex.abs (z - (7 - 3*I))^2 = 94 := by
  sorry

end complex_distance_sum_l2495_249511


namespace nes_sale_price_l2495_249569

/-- The sale price of an NES given trade-in and cash transactions -/
theorem nes_sale_price 
  (snes_value : ℝ) 
  (trade_in_percentage : ℝ) 
  (cash_given : ℝ) 
  (change_received : ℝ) 
  (game_value : ℝ) 
  (h1 : snes_value = 150)
  (h2 : trade_in_percentage = 0.8)
  (h3 : cash_given = 80)
  (h4 : change_received = 10)
  (h5 : game_value = 30) :
  snes_value * trade_in_percentage + cash_given - change_received - game_value = 160 :=
by sorry

end nes_sale_price_l2495_249569


namespace line_through_A_equal_intercepts_line_BC_equation_l2495_249501

-- Define the point A
def A : ℝ × ℝ := (2, 1)

-- Part 1: Line through A with equal intercepts
theorem line_through_A_equal_intercepts :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (a * A.1 + b * A.2 + c = 0) ∧
  (a + b + c = 0) ∧
  (a = 1 ∧ b = 1 ∧ c = -3) := by sorry

-- Part 2: Triangle ABC
theorem line_BC_equation (B C : ℝ × ℝ) :
  -- Given conditions
  (B.1 - B.2 = 0) →  -- B is on the line x - y = 0
  (2 * ((A.1 + B.1) / 2) + ((A.2 + B.2) / 2) - 1 = 0) →  -- CM is on 2x + y - 1 = 0
  (C.1 + C.2 - 3 = 0) →  -- C is on x + y - 3 = 0
  (2 * C.1 + C.2 - 1 = 0) →  -- C is on 2x + y - 1 = 0
  -- Conclusion
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (a * B.1 + b * B.2 + c = 0) ∧
  (a * C.1 + b * C.2 + c = 0) ∧
  (a = 6 ∧ b = 1 ∧ c = 7) := by sorry

end line_through_A_equal_intercepts_line_BC_equation_l2495_249501


namespace conditional_probability_B_given_A_l2495_249598

-- Define the set of numbers
def S : Finset ℕ := Finset.range 7

-- Define a type for a selection of 5 numbers
def Selection := {s : Finset ℕ // s.card = 5 ∧ s ⊆ S}

-- Define the median of a selection
def median (sel : Selection) : ℚ :=
  sorry

-- Define the average of a selection
def average (sel : Selection) : ℚ :=
  sorry

-- Define event A: median is 4
def eventA (sel : Selection) : Prop :=
  median sel = 4

-- Define event B: average is 4
def eventB (sel : Selection) : Prop :=
  average sel = 4

-- Define the probability measure
noncomputable def P : Set Selection → ℝ :=
  sorry

-- State the theorem
theorem conditional_probability_B_given_A :
  P {sel : Selection | eventB sel ∧ eventA sel} / P {sel : Selection | eventA sel} = 1/3 :=
sorry

end conditional_probability_B_given_A_l2495_249598


namespace optimal_garden_is_best_l2495_249580

/-- Represents a rectangular garden with one side against a wall --/
structure Garden where
  width : ℝ  -- Width of the garden (perpendicular to the wall)
  length : ℝ  -- Length of the garden (parallel to the wall)

/-- The wall length --/
def wall_length : ℝ := 600

/-- The fence cost per foot --/
def fence_cost_per_foot : ℝ := 6

/-- The total budget for fencing --/
def fence_budget : ℝ := 1800

/-- The minimum required area --/
def min_area : ℝ := 6000

/-- Calculate the area of the garden --/
def area (g : Garden) : ℝ := g.width * g.length

/-- Calculate the perimeter of the garden --/
def perimeter (g : Garden) : ℝ := 2 * g.width + g.length + wall_length

/-- Check if the garden satisfies the budget constraint --/
def satisfies_budget (g : Garden) : Prop :=
  (2 * g.width + g.length) * fence_cost_per_foot ≤ fence_budget

/-- Check if the garden satisfies the area constraint --/
def satisfies_area (g : Garden) : Prop :=
  area g ≥ min_area

/-- The optimal garden dimensions --/
def optimal_garden : Garden :=
  { width := 75, length := 150 }

/-- Theorem stating that the optimal garden maximizes perimeter while satisfying constraints --/
theorem optimal_garden_is_best :
  satisfies_budget optimal_garden ∧
  satisfies_area optimal_garden ∧
  ∀ g : Garden, satisfies_budget g → satisfies_area g →
    perimeter g ≤ perimeter optimal_garden :=
by sorry

end optimal_garden_is_best_l2495_249580


namespace baker_bought_two_boxes_of_baking_soda_l2495_249516

/-- The number of boxes of baking soda bought by the baker -/
def baking_soda_boxes : ℕ :=
  let flour_cost : ℕ := 3 * 3
  let eggs_cost : ℕ := 3 * 10
  let milk_cost : ℕ := 7 * 5
  let total_cost : ℕ := 80
  let baking_soda_unit_cost : ℕ := 3
  (total_cost - (flour_cost + eggs_cost + milk_cost)) / baking_soda_unit_cost

theorem baker_bought_two_boxes_of_baking_soda :
  baking_soda_boxes = 2 := by sorry

end baker_bought_two_boxes_of_baking_soda_l2495_249516


namespace dandan_age_problem_l2495_249548

theorem dandan_age_problem (dandan_age : ℕ) (father_age : ℕ) (a : ℕ) :
  dandan_age = 4 →
  father_age = 28 →
  father_age + a = 3 * (dandan_age + a) →
  a = 8 :=
by sorry

end dandan_age_problem_l2495_249548


namespace room_length_to_perimeter_ratio_l2495_249545

/-- The ratio of a rectangular room's length to its perimeter -/
theorem room_length_to_perimeter_ratio :
  let length : ℚ := 23
  let width : ℚ := 13
  let perimeter : ℚ := 2 * (length + width)
  (length : ℚ) / perimeter = 23 / 72 := by sorry

end room_length_to_perimeter_ratio_l2495_249545


namespace common_number_in_list_l2495_249592

theorem common_number_in_list (list : List ℝ) : 
  list.length = 7 →
  (list.take 4).sum / 4 = 7 →
  (list.drop 3).sum / 4 = 10 →
  list.sum / 7 = 8 →
  ∃ x ∈ list.take 4 ∩ list.drop 3, x = 12 :=
by sorry

end common_number_in_list_l2495_249592


namespace cubic_equation_solution_l2495_249528

theorem cubic_equation_solution :
  ∃! x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ∧ x = 6 := by
  sorry

end cubic_equation_solution_l2495_249528


namespace billys_book_pages_l2495_249540

/-- Proves that given Billy's reading habits and time allocation, each book he reads contains 80 pages. -/
theorem billys_book_pages : 
  -- Billy's free time per day
  (free_time_per_day : ℕ) →
  -- Number of weekend days
  (weekend_days : ℕ) →
  -- Percentage of time spent on video games
  (video_game_percentage : ℚ) →
  -- Pages Billy can read per hour
  (pages_per_hour : ℕ) →
  -- Number of books Billy reads
  (number_of_books : ℕ) →
  -- Conditions
  (free_time_per_day = 8) →
  (weekend_days = 2) →
  (video_game_percentage = 3/4) →
  (pages_per_hour = 60) →
  (number_of_books = 3) →
  -- Conclusion: each book contains 80 pages
  (∃ (pages_per_book : ℕ), pages_per_book = 80 ∧ 
    pages_per_book * number_of_books = 
      (1 - video_game_percentage) * (free_time_per_day * weekend_days : ℚ) * pages_per_hour) :=
by
  sorry


end billys_book_pages_l2495_249540


namespace exact_blue_marbles_probability_l2495_249543

def total_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 4
def num_draws : ℕ := 7
def target_blue : ℕ := 4

theorem exact_blue_marbles_probability :
  (Nat.choose num_draws target_blue : ℚ) * (blue_marbles ^ target_blue * red_marbles ^ (num_draws - target_blue)) / (total_marbles ^ num_draws) = 35 * (16 : ℚ) / 2187 := by
  sorry

end exact_blue_marbles_probability_l2495_249543


namespace calculation_proof_l2495_249533

theorem calculation_proof : 
  Real.rpow 27 (1/3) + (Real.sqrt 2 - 1)^2 - (1/2)⁻¹ + 2 / (Real.sqrt 2 - 1) = 6 := by
  sorry

end calculation_proof_l2495_249533


namespace video_votes_l2495_249551

theorem video_votes (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ) (score : ℤ) : 
  total_votes = likes + dislikes →
  likes = (6 : ℕ) * total_votes / 10 →
  dislikes = (4 : ℕ) * total_votes / 10 →
  score = likes - dislikes →
  score = 150 →
  total_votes = 750 := by
sorry


end video_votes_l2495_249551


namespace chessboard_rectangle_same_color_l2495_249562

-- Define the chessboard as a 4x7 matrix of booleans (true for black, false for white)
def Chessboard := Matrix (Fin 4) (Fin 7) Bool

-- Define a rectangle on the chessboard
def Rectangle (board : Chessboard) (r1 r2 : Fin 4) (c1 c2 : Fin 7) : Prop :=
  r1 < r2 ∧ c1 < c2

-- Define the property of a rectangle having all corners of the same color
def SameColorCorners (board : Chessboard) (r1 r2 : Fin 4) (c1 c2 : Fin 7) : Prop :=
  Rectangle board r1 r2 c1 c2 ∧
  board r1 c1 = board r1 c2 ∧
  board r1 c1 = board r2 c1 ∧
  board r1 c1 = board r2 c2

-- The main theorem
theorem chessboard_rectangle_same_color (board : Chessboard) :
  ∃ (r1 r2 : Fin 4) (c1 c2 : Fin 7), SameColorCorners board r1 r2 c1 c2 := by
  sorry


end chessboard_rectangle_same_color_l2495_249562


namespace lowest_fifth_score_for_target_average_l2495_249522

def number_of_tests : ℕ := 5
def max_score : ℕ := 100
def target_average : ℕ := 85

def first_three_scores : List ℕ := [76, 94, 87]
def fourth_score : ℕ := 92

def total_needed_score : ℕ := number_of_tests * target_average

theorem lowest_fifth_score_for_target_average :
  ∃ (fifth_score : ℕ),
    fifth_score = total_needed_score - (first_three_scores.sum + fourth_score) ∧
    fifth_score = 76 ∧
    (∀ (x : ℕ), x < fifth_score →
      (first_three_scores.sum + fourth_score + x) / number_of_tests < target_average) :=
by sorry

end lowest_fifth_score_for_target_average_l2495_249522


namespace fathers_age_three_times_xiaojuns_l2495_249573

theorem fathers_age_three_times_xiaojuns (xiaojun_age : ℕ) (father_age : ℕ) (years_passed : ℕ) :
  xiaojun_age = 5 →
  father_age = 31 →
  years_passed = 8 →
  father_age + years_passed = 3 * (xiaojun_age + years_passed) :=
by
  sorry

#check fathers_age_three_times_xiaojuns

end fathers_age_three_times_xiaojuns_l2495_249573


namespace largest_integer_for_negative_quadratic_l2495_249549

theorem largest_integer_for_negative_quadratic :
  ∀ m : ℤ, m^2 - 11*m + 24 < 0 → m ≤ 7 ∧ 
  ∃ n : ℤ, n^2 - 11*n + 24 < 0 ∧ n = 7 :=
by sorry

end largest_integer_for_negative_quadratic_l2495_249549
