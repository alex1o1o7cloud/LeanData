import Mathlib

namespace unique_divisible_by_396_l1690_169046

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧
    n = x * 100000 + y * 10000 + 2 * 1000 + 4 * 100 + 3 * 10 + z

theorem unique_divisible_by_396 :
  ∃! n : ℕ, is_valid_number n ∧ n % 396 = 0 :=
by
  -- The proof goes here
  sorry

end unique_divisible_by_396_l1690_169046


namespace solve_for_m_l1690_169051

theorem solve_for_m (x y m : ℝ) (h1 : x = 2) (h2 : y = m) (h3 : 3 * x + 2 * y = 10) : m = 2 := by
  sorry

end solve_for_m_l1690_169051


namespace equation_solution_l1690_169024

theorem equation_solution (x y : ℝ) (hx : x ≠ 0) (hxy : x + y ≠ 0) :
  (x + y) / x = (y + 1) / (x + y) →
  (x = (-y + Real.sqrt (4 - 3 * y^2)) / 2 ∨ x = (-y - Real.sqrt (4 - 3 * y^2)) / 2) ∧
  -2 / Real.sqrt 3 ≤ y ∧ y ≤ 2 / Real.sqrt 3 :=
by sorry

end equation_solution_l1690_169024


namespace incorrect_statement_l1690_169094

theorem incorrect_statement : ¬(∀ (p q : Prop), (¬p ∧ ¬q) → (p ∧ q = False)) := by
  sorry

end incorrect_statement_l1690_169094


namespace parallel_line_not_through_point_l1690_169001

-- Define a line in 2D space
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

def is_point_on_line (p : Point) (l : Line) : Prop :=
  l.A * p.x + l.B * p.y + l.C = 0

def are_lines_parallel (l1 l2 : Line) : Prop :=
  l1.A * l2.B = l1.B * l2.A

theorem parallel_line_not_through_point 
  (L : Line) (P : Point) (h : ¬ is_point_on_line P L) :
  ∃ (L' : Line), 
    are_lines_parallel L' L ∧ 
    ¬ is_point_on_line P L' ∧
    L'.A = L.A ∧ 
    L'.B = L.B ∧ 
    L'.C = L.C + (L.A * P.x + L.B * P.y + L.C) := by
  sorry

end parallel_line_not_through_point_l1690_169001


namespace first_day_pages_l1690_169071

/-- Proves that given the specific writing pattern and remaining pages, the writer wrote 25 pages on the first day -/
theorem first_day_pages (total_pages remaining_pages day_4_pages : ℕ) 
  (h1 : total_pages = 500)
  (h2 : remaining_pages = 315)
  (h3 : day_4_pages = 10) : 
  ∃ x : ℕ, x + 2*x + 4*x + day_4_pages = total_pages - remaining_pages ∧ x = 25 := by
  sorry

end first_day_pages_l1690_169071


namespace younger_person_age_l1690_169023

theorem younger_person_age (y e : ℕ) : 
  e = y + 20 → 
  e - 4 = 5 * (y - 4) → 
  y = 9 := by
sorry

end younger_person_age_l1690_169023


namespace line_segment_endpoint_l1690_169021

theorem line_segment_endpoint (y : ℝ) : 
  y > 0 → 
  Real.sqrt ((3 - (-5))^2 + (y - 4)^2) = 12 → 
  y = 4 + 4 * Real.sqrt 5 := by
sorry

end line_segment_endpoint_l1690_169021


namespace vector_equation_y_axis_l1690_169054

/-- Given points O, A, and B in the plane, and a vector equation for OP,
    prove that if P is on the y-axis, then m = 2/3 -/
theorem vector_equation_y_axis (O A B P : ℝ × ℝ) (m : ℝ) :
  O = (0, 0) →
  A = (-1, 3) →
  B = (2, -4) →
  P.1 = 0 →
  P = (2 * A.1 + m * (B.1 - A.1), 2 * A.2 + m * (B.2 - A.2)) →
  m = 2/3 := by
  sorry

end vector_equation_y_axis_l1690_169054


namespace total_cargo_after_loading_l1690_169030

def initial_cargo : ℕ := 5973
def loaded_cargo : ℕ := 8723

theorem total_cargo_after_loading : initial_cargo + loaded_cargo = 14696 := by
  sorry

end total_cargo_after_loading_l1690_169030


namespace cubic_root_sum_l1690_169034

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 6*p - 3 = 0 →
  q^3 - 8*q^2 + 6*q - 3 = 0 →
  r^3 - 8*r^2 + 6*r - 3 = 0 →
  p/(q*r-1) + q/(p*r-1) + r/(p*q-1) = -14 := by
sorry

end cubic_root_sum_l1690_169034


namespace max_intersections_l1690_169050

/-- Represents a polynomial of degree 5 or less -/
def Polynomial5 := Fin 6 → ℝ

/-- The set of ten 5-degree polynomials -/
def TenPolynomials := Fin 10 → Polynomial5

/-- A linear function representing an arithmetic sequence -/
def ArithmeticSequence := ℝ → ℝ

/-- The number of intersections between a polynomial and a linear function -/
def intersections (p : Polynomial5) (f : ArithmeticSequence) : ℕ :=
  sorry

/-- The total number of intersections between ten polynomials and a linear function -/
def totalIntersections (polynomials : TenPolynomials) (f : ArithmeticSequence) : ℕ :=
  sorry

theorem max_intersections (polynomials : TenPolynomials) (f : ArithmeticSequence) :
  totalIntersections polynomials f ≤ 50 :=
sorry

end max_intersections_l1690_169050


namespace lawn_length_is_70_l1690_169077

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  roadWidth : ℝ
  gravelCostPerSquareMeter : ℝ
  totalGravelCost : ℝ

/-- Calculates the total area of the roads -/
def roadArea (l : LawnWithRoads) : ℝ :=
  l.length * l.roadWidth + l.width * l.roadWidth - l.roadWidth * l.roadWidth

/-- Theorem stating that given the conditions, the length of the lawn must be 70 m -/
theorem lawn_length_is_70 (l : LawnWithRoads)
    (h1 : l.width = 30)
    (h2 : l.roadWidth = 5)
    (h3 : l.gravelCostPerSquareMeter = 4)
    (h4 : l.totalGravelCost = 1900)
    (h5 : l.totalGravelCost = l.gravelCostPerSquareMeter * roadArea l) :
    l.length = 70 := by
  sorry

end lawn_length_is_70_l1690_169077


namespace highlighter_count_l1690_169043

/-- The number of highlighters in Kaya's teacher's desk -/
theorem highlighter_count : 
  let pink : ℕ := 12
  let yellow : ℕ := 15
  let blue : ℕ := 8
  let green : ℕ := 6
  let orange : ℕ := 4
  pink + yellow + blue + green + orange = 45 := by
  sorry

end highlighter_count_l1690_169043


namespace john_annual_profit_l1690_169017

def annual_profit (num_subletters : ℕ) (subletter_rent : ℕ) (monthly_expense : ℕ) (months_per_year : ℕ) : ℕ :=
  (num_subletters * subletter_rent - monthly_expense) * months_per_year

theorem john_annual_profit :
  annual_profit 3 400 900 12 = 3600 := by
  sorry

end john_annual_profit_l1690_169017


namespace gcd_of_B_is_five_l1690_169091

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_five : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d) ∧ d = 5 := by
  sorry

end gcd_of_B_is_five_l1690_169091


namespace f_min_at_negative_seven_l1690_169093

/-- The quadratic function f(x) = x^2 + 14x - 12 -/
def f (x : ℝ) : ℝ := x^2 + 14*x - 12

/-- The point where f attains its minimum -/
def min_point : ℝ := -7

theorem f_min_at_negative_seven :
  ∀ x : ℝ, f x ≥ f min_point := by
sorry

end f_min_at_negative_seven_l1690_169093


namespace min_friend_pairs_2000_users_min_friend_pairs_within_bounds_l1690_169085

/-- Represents a social network with a fixed number of users and invitations per user. -/
structure SocialNetwork where
  numUsers : ℕ
  invitationsPerUser : ℕ

/-- Calculates the minimum number of friend pairs in a social network where friendships
    are formed only when invitations are mutual. -/
def minFriendPairs (network : SocialNetwork) : ℕ :=
  (network.numUsers * network.invitationsPerUser) / 2

/-- Theorem stating that in a social network with 2000 users, each inviting 1000 others,
    the minimum number of friend pairs is 1000. -/
theorem min_friend_pairs_2000_users :
  let network : SocialNetwork := { numUsers := 2000, invitationsPerUser := 1000 }
  minFriendPairs network = 1000 := by
  sorry

/-- Verifies that the calculated minimum number of friend pairs does not exceed
    the maximum possible number of pairs given the number of users. -/
theorem min_friend_pairs_within_bounds (network : SocialNetwork) :
  minFriendPairs network ≤ (network.numUsers.choose 2) := by
  sorry

end min_friend_pairs_2000_users_min_friend_pairs_within_bounds_l1690_169085


namespace range_of_m_for_quadratic_equation_l1690_169073

theorem range_of_m_for_quadratic_equation (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) ↔ 
  (m < -2 ∨ m > 2) :=
sorry

end range_of_m_for_quadratic_equation_l1690_169073


namespace triangle_cosine_C_l1690_169011

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A ∧ A < π / 2)
  (h2 : 0 < B ∧ B < π / 2)
  (h3 : 0 < C ∧ C < π / 2)
  (h4 : A + B + C = π)

-- State the theorem
theorem triangle_cosine_C (t : Triangle) 
  (h5 : f t.A = 3 / 2)
  (h6 : ∃ (D : ℝ), D = Real.sqrt 2 ∧ D * Real.sin (t.B / 2) = t.c * Real.sin (t.A / 2))
  (h7 : ∃ (D : ℝ), D = 2 ∧ D * Real.sin (t.A / 2) = t.b * Real.sin (t.C / 2)) :
  Real.cos t.C = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end

end triangle_cosine_C_l1690_169011


namespace white_balls_count_l1690_169022

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) 
  (h_total : total = 60)
  (h_green : green = 18)
  (h_yellow : yellow = 8)
  (h_red : red = 5)
  (h_purple : purple = 7)
  (h_prob : prob_not_red_purple = 4/5) :
  total - (green + yellow + red + purple) = 22 := by
sorry

end white_balls_count_l1690_169022


namespace game_ends_in_finite_steps_l1690_169060

/-- Represents the state of the game at any point -/
structure GameState where
  m : ℕ+  -- Player A's number
  n : ℕ+  -- Player B's number
  t : ℤ   -- The other number written by the umpire
  k : ℕ   -- The current question number

/-- Represents whether a player knows the other's number -/
def knows (state : GameState) (player : Bool) : Prop :=
  if player 
  then state.t ≤ state.m + state.n - state.n / state.k
  else state.t ≥ state.m + state.n + state.n / state.k

/-- The main theorem stating that the game will end after a finite number of questions -/
theorem game_ends_in_finite_steps : 
  ∀ (initial_state : GameState), 
  ∃ (final_state : GameState), 
  (knows final_state true ∨ knows final_state false) ∧ 
  final_state.k ≥ initial_state.k :=
sorry

end game_ends_in_finite_steps_l1690_169060


namespace max_value_a_l1690_169061

theorem max_value_a (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b = 1 - a)
  (h4 : ∀ x : ℝ, x ∈ Set.Icc 0 1 → Real.exp x ≤ (1 + a * x) / (1 - b * x)) :
  a ≤ (1 : ℝ) / 2 ∧ ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ Real.exp x = (1 + (1/2) * x) / (1 - (1/2) * x) :=
by sorry

end max_value_a_l1690_169061


namespace count_12_digit_numbers_with_consecutive_ones_l1690_169037

/-- The sequence of counts of n-digit numbers with digits 1, 2, or 3 without two consecutive 1's -/
def F : ℕ → ℕ
| 0 => 1
| 1 => 3
| (n+2) => 2 * F (n+1) + F n

/-- The count of n-digit numbers with digits 1, 2, or 3 -/
def total_count (n : ℕ) : ℕ := 3^n

/-- The count of n-digit numbers with digits 1, 2, or 3 and at least two consecutive 1's -/
def count_with_consecutive_ones (n : ℕ) : ℕ := total_count n - F n

theorem count_12_digit_numbers_with_consecutive_ones : 
  count_with_consecutive_ones 12 = 530456 := by
  sorry

end count_12_digit_numbers_with_consecutive_ones_l1690_169037


namespace correct_statements_count_l1690_169027

/-- A circle in a plane. -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in a plane. -/
structure Line where
  point1 : Point
  point2 : Point

/-- A statement about circle geometry. -/
inductive CircleStatement
  | perpRadiusTangent : CircleStatement
  | centerPerpTangentThruPoint : CircleStatement
  | tangentPerpThruCenterPoint : CircleStatement
  | radiusEndPerpTangent : CircleStatement
  | chordTangentMidpoint : CircleStatement

/-- Determines if a circle statement is correct. -/
def isCorrectStatement (s : CircleStatement) : Bool :=
  match s with
  | CircleStatement.perpRadiusTangent => false
  | CircleStatement.centerPerpTangentThruPoint => true
  | CircleStatement.tangentPerpThruCenterPoint => true
  | CircleStatement.radiusEndPerpTangent => false
  | CircleStatement.chordTangentMidpoint => true

/-- The list of all circle statements. -/
def allStatements : List CircleStatement :=
  [CircleStatement.perpRadiusTangent,
   CircleStatement.centerPerpTangentThruPoint,
   CircleStatement.tangentPerpThruCenterPoint,
   CircleStatement.radiusEndPerpTangent,
   CircleStatement.chordTangentMidpoint]

/-- Counts the number of correct statements. -/
def countCorrectStatements (statements : List CircleStatement) : Nat :=
  statements.filter isCorrectStatement |>.length

theorem correct_statements_count :
  countCorrectStatements allStatements = 3 := by
  sorry

end correct_statements_count_l1690_169027


namespace correct_product_l1690_169086

theorem correct_product (x y : ℚ) (z : ℕ) (h1 : x = 63 / 10000) (h2 : y = 385 / 100) (h3 : z = 24255) :
  x * y = 24255 / 1000000 :=
sorry

end correct_product_l1690_169086


namespace boys_usual_time_to_school_l1690_169009

theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) : 
  usual_rate > 0 → 
  usual_time > 0 → 
  usual_rate * usual_time = (6/5 * usual_rate) * (usual_time - 4) → 
  usual_time = 24 := by
sorry

end boys_usual_time_to_school_l1690_169009


namespace house_painting_cost_is_1900_l1690_169063

/-- The cost of painting a house given the contributions of three individuals. -/
def housePaintingCost (judsonContribution : ℕ) : ℕ :=
  let kennyContribution := judsonContribution + judsonContribution / 5
  let camiloContribution := kennyContribution + 200
  judsonContribution + kennyContribution + camiloContribution

/-- Theorem stating that the total cost of painting the house is $1900. -/
theorem house_painting_cost_is_1900 : housePaintingCost 500 = 1900 := by
  sorry

end house_painting_cost_is_1900_l1690_169063


namespace simplify_cubic_root_sum_exponents_l1690_169084

-- Define the expression
def radicand : ℕ → ℕ → ℕ → ℕ → ℕ
  | a, b, c, d => 60 * a^5 * b^7 * c^8 * d^2

-- Define the function to calculate the sum of exponents outside the radical
def sum_exponents_outside_radical : ℕ → ℕ → ℕ → ℕ → ℕ
  | a, b, c, d => 5

-- Theorem statement
theorem simplify_cubic_root_sum_exponents
  (a b c d : ℕ) :
  sum_exponents_outside_radical a b c d = 5 :=
by sorry

end simplify_cubic_root_sum_exponents_l1690_169084


namespace andrew_cookie_cost_l1690_169012

/-- The cost of cookies purchased by Andrew in May --/
def total_cost : ℕ := 1395

/-- The number of days in May --/
def days_in_may : ℕ := 31

/-- The number of cookies Andrew purchased each day --/
def cookies_per_day : ℕ := 3

/-- The total number of cookies Andrew purchased in May --/
def total_cookies : ℕ := days_in_may * cookies_per_day

/-- The cost of each cookie --/
def cookie_cost : ℚ := total_cost / total_cookies

theorem andrew_cookie_cost : cookie_cost = 15 := by
  sorry

end andrew_cookie_cost_l1690_169012


namespace rikkis_earnings_l1690_169040

/-- Represents Rikki's poetry writing and selling scenario -/
structure PoetryScenario where
  price_per_word : ℚ
  words_per_unit : ℕ
  minutes_per_unit : ℕ
  total_hours : ℕ

/-- Calculates the expected earnings for a given poetry scenario -/
def expected_earnings (scenario : PoetryScenario) : ℚ :=
  let total_minutes : ℕ := scenario.total_hours * 60
  let total_units : ℕ := total_minutes / scenario.minutes_per_unit
  let total_words : ℕ := total_units * scenario.words_per_unit
  (total_words : ℚ) * scenario.price_per_word

/-- Rikki's specific poetry scenario -/
def rikkis_scenario : PoetryScenario :=
  { price_per_word := 1 / 100
  , words_per_unit := 25
  , minutes_per_unit := 5
  , total_hours := 2 }

theorem rikkis_earnings :
  expected_earnings rikkis_scenario = 6 := by
  sorry

end rikkis_earnings_l1690_169040


namespace value_of_7a_plus_3b_l1690_169070

-- Define the function g
def g (x : ℝ) : ℝ := 7 * x - 4

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem value_of_7a_plus_3b 
  (a b : ℝ) 
  (h1 : ∀ x, g x = (Function.invFun (f a b)) x - 5) 
  (h2 : Function.Injective (f a b)) :
  7 * a + 3 * b = 4 / 7 := by
  sorry

end value_of_7a_plus_3b_l1690_169070


namespace log_equation_solution_l1690_169057

theorem log_equation_solution : 
  ∃! x : ℝ, x > 0 ∧ 2 * Real.log x = Real.log 192 + Real.log 3 - Real.log 4 :=
by
  -- The unique solution is x = 12
  use 12
  constructor
  · -- Prove that x = 12 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check log_equation_solution

end log_equation_solution_l1690_169057


namespace perpendicular_vectors_x_value_l1690_169066

theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) :
  a = (1, 2) →
  b = (2, x) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  x = -1 := by
sorry

end perpendicular_vectors_x_value_l1690_169066


namespace other_root_of_quadratic_l1690_169018

theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x, 3 * x^2 + m * x = 5) → 
  (3 * 5^2 + m * 5 = 5) → 
  (3 * (-1/3)^2 + m * (-1/3) = 5) :=
by
  sorry

end other_root_of_quadratic_l1690_169018


namespace pastry_combinations_linda_pastry_purchase_l1690_169000

theorem pastry_combinations : ℕ → ℕ → ℕ
  | n, k => Nat.choose (n + k - 1) (k - 1)

theorem linda_pastry_purchase : pastry_combinations 4 4 = 35 := by
  sorry

end pastry_combinations_linda_pastry_purchase_l1690_169000


namespace quadratic_factorization_sum_l1690_169062

theorem quadratic_factorization_sum (p q r : ℤ) : 
  (∀ x, x^2 + 16*x + 63 = (x + p) * (x + q)) →
  (∀ x, x^2 - 15*x + 56 = (x - q) * (x - r)) →
  p + q + r = 22 := by
sorry

end quadratic_factorization_sum_l1690_169062


namespace hyperbola_equation_l1690_169053

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    x = Real.sqrt 2 ∧ y = Real.sqrt 3) →
  (Real.sqrt (1 + b^2 / a^2) = 2) →
  (∀ x y : ℝ, x^2 - y^2 / 3 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end hyperbola_equation_l1690_169053


namespace no_solution_for_specific_a_l1690_169005

/-- The equation 7|x-4a|+|x-a²|+6x-2a=0 has no solution when a ∈ (-∞, -18) ∪ (0, +∞) -/
theorem no_solution_for_specific_a (a : ℝ) : 
  (a < -18 ∨ a > 0) → ¬∃ x : ℝ, 7*|x - 4*a| + |x - a^2| + 6*x - 2*a = 0 := by
  sorry

end no_solution_for_specific_a_l1690_169005


namespace pascal_interior_sum_l1690_169041

def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_interior_sum : 
  interior_sum 6 = 30 → interior_sum 8 + interior_sum 9 = 380 := by
sorry

end pascal_interior_sum_l1690_169041


namespace complex_equation_solution_l1690_169047

theorem complex_equation_solution :
  ∀ z : ℂ, -Complex.I * z = (3 + 2 * Complex.I) * (1 - Complex.I) → z = 1 + 5 * Complex.I :=
by
  sorry

end complex_equation_solution_l1690_169047


namespace quadratic_equation_with_root_one_l1690_169039

theorem quadratic_equation_with_root_one (a : ℝ) (h : a ≠ 0) :
  ∃ f : ℝ → ℝ, (∀ x, f x = a * x^2 - a) ∧ f 1 = 0 := by
  sorry

end quadratic_equation_with_root_one_l1690_169039


namespace sphere_radius_regular_tetrahedron_l1690_169081

/-- The radius of a sphere touching all edges of a regular tetrahedron with edge length √2 --/
theorem sphere_radius_regular_tetrahedron : 
  ∀ (tetrahedron_edge : ℝ) (sphere_radius : ℝ),
  tetrahedron_edge = Real.sqrt 2 →
  sphere_radius = 
    (1 / 2) * ((tetrahedron_edge * Real.sqrt 6) / 3) →
  sphere_radius = 1 := by
sorry

end sphere_radius_regular_tetrahedron_l1690_169081


namespace binomial_properties_l1690_169007

variable (X : Nat → ℝ)

def binomial_distribution (n : Nat) (p : ℝ) (X : Nat → ℝ) : Prop :=
  ∀ k, 0 ≤ k ∧ k ≤ n → X k = (n.choose k : ℝ) * p^k * (1-p)^(n-k)

def expectation (X : Nat → ℝ) : ℝ := sorry
def variance (X : Nat → ℝ) : ℝ := sorry

theorem binomial_properties :
  binomial_distribution 8 (1/2) X →
  expectation X = 4 ∧
  variance X = 2 ∧
  X 3 = X 5 := by sorry

end binomial_properties_l1690_169007


namespace four_isosceles_triangles_l1690_169098

/-- A triangle represented by three points on a 2D plane. -/
structure Triangle :=
  (a b c : ℕ × ℕ)

/-- Checks if a triangle is isosceles. -/
def isIsosceles (t : Triangle) : Bool :=
  let d1 := ((t.a.1 - t.b.1)^2 + (t.a.2 - t.b.2)^2 : ℕ)
  let d2 := ((t.b.1 - t.c.1)^2 + (t.b.2 - t.c.2)^2 : ℕ)
  let d3 := ((t.c.1 - t.a.1)^2 + (t.c.2 - t.a.2)^2 : ℕ)
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

def triangles : List Triangle := [
  ⟨(0, 6), (2, 6), (1, 4)⟩,
  ⟨(3, 4), (3, 6), (5, 4)⟩,
  ⟨(0, 1), (3, 2), (6, 1)⟩,
  ⟨(7, 4), (6, 6), (9, 4)⟩,
  ⟨(8, 1), (9, 3), (10, 0)⟩
]

theorem four_isosceles_triangles :
  (triangles.filter isIsosceles).length = 4 := by
  sorry


end four_isosceles_triangles_l1690_169098


namespace jacqueline_erasers_l1690_169078

-- Define the given quantities
def cases : ℕ := 7
def boxes_per_case : ℕ := 12
def erasers_per_box : ℕ := 25

-- Define the total number of erasers
def total_erasers : ℕ := cases * boxes_per_case * erasers_per_box

-- Theorem to prove
theorem jacqueline_erasers : total_erasers = 2100 := by
  sorry

end jacqueline_erasers_l1690_169078


namespace product_difference_equals_two_tenths_l1690_169056

theorem product_difference_equals_two_tenths : 0.5 * 0.8 - 0.2 = 0.2 := by
  sorry

end product_difference_equals_two_tenths_l1690_169056


namespace binomial_sum_l1690_169042

theorem binomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + a₃ + a₅ = 123 := by
  sorry

end binomial_sum_l1690_169042


namespace smallest_positive_and_largest_negative_integer_l1690_169065

theorem smallest_positive_and_largest_negative_integer :
  (∀ n : ℤ, n > 0 → n ≥ 1) ∧ (∀ m : ℤ, m < 0 → m ≤ -1) := by
  sorry

end smallest_positive_and_largest_negative_integer_l1690_169065


namespace matchstick_20th_term_l1690_169025

/-- Arithmetic sequence with first term 4 and common difference 3 -/
def matchstick_sequence (n : ℕ) : ℕ := 4 + 3 * (n - 1)

/-- The 20th term of the matchstick sequence is 61 -/
theorem matchstick_20th_term : matchstick_sequence 20 = 61 := by
  sorry

end matchstick_20th_term_l1690_169025


namespace student_count_l1690_169016

/-- In a class, given a student who is both the 30th best and 30th worst, 
    the total number of students in the class is 59. -/
theorem student_count (n : ℕ) (rob : ℕ) 
  (h1 : rob = 30)  -- Rob's position from the top
  (h2 : rob = n - 29) : -- Rob's position from the bottom
  n = 59 := by
  sorry

end student_count_l1690_169016


namespace parabola_equation_l1690_169082

/-- A parabola with vertex at the origin and focus on the line x - 2y - 2 = 0 --/
structure Parabola where
  /-- The focus of the parabola lies on this line --/
  focus_line : {(x, y) : ℝ × ℝ | x - 2*y - 2 = 0}
  /-- The axis of symmetry is either the x-axis or y-axis --/
  symmetry_axis : (Unit → Prop) ⊕ (Unit → Prop)

/-- The standard equation of the parabola is either y² = 8x or x² = -4y --/
theorem parabola_equation (p : Parabola) :
  (∃ (x y : ℝ), y^2 = 8*x) ∨ (∃ (x y : ℝ), x^2 = -4*y) :=
sorry

end parabola_equation_l1690_169082


namespace problem_statement_l1690_169064

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a / (1 + a)) + (b / (1 + b)) = 1) : 
  (a / (1 + b^2)) - (b / (1 + a^2)) = a - b := by
  sorry

end problem_statement_l1690_169064


namespace sum_of_a_and_b_l1690_169095

theorem sum_of_a_and_b (a b : ℚ) (h1 : 3 * a + 7 * b = 12) (h2 : 9 * a + 2 * b = 23) :
  a + b = 176 / 57 := by sorry

end sum_of_a_and_b_l1690_169095


namespace inequality_proof_l1690_169048

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b + b * c + a * c)^2 ≥ 3 * a * b * c * (a + b + c) := by
  sorry

end inequality_proof_l1690_169048


namespace otimes_four_two_l1690_169074

-- Define the new operation ⊗
def otimes (a b : ℝ) : ℝ := 4 * a + 5 * b

-- Theorem to prove
theorem otimes_four_two : otimes 4 2 = 26 := by
  sorry

end otimes_four_two_l1690_169074


namespace charlie_bobby_age_difference_l1690_169089

theorem charlie_bobby_age_difference :
  ∀ (jenny charlie bobby : ℕ),
  jenny = charlie + 5 →
  ∃ (x : ℕ), charlie + x = 11 ∧ jenny + x = 2 * (bobby + x) →
  charlie = bobby + 3 :=
by
  sorry

end charlie_bobby_age_difference_l1690_169089


namespace opposite_expression_implies_ab_zero_l1690_169097

/-- Given that for all x, ax + bx^2 = -(a(-x) + b(-x)^2), prove that ab = 0 -/
theorem opposite_expression_implies_ab_zero (a b : ℝ) 
  (h : ∀ x : ℝ, a * x + b * x^2 = -(a * (-x) + b * (-x)^2)) : 
  a * b = 0 := by
  sorry

end opposite_expression_implies_ab_zero_l1690_169097


namespace remainder_233_divided_by_d_l1690_169020

theorem remainder_233_divided_by_d (a b c d : ℕ) : 
  1 < a → a < b → b < c → a + c = 13 → d = a * b * c → 
  233 % d = 53 := by sorry

end remainder_233_divided_by_d_l1690_169020


namespace johns_candy_store_spending_l1690_169002

theorem johns_candy_store_spending (allowance : ℝ) (arcade_fraction : ℝ) (toy_store_fraction : ℝ)
  (h1 : allowance = 3.375)
  (h2 : arcade_fraction = 3/5)
  (h3 : toy_store_fraction = 1/3) :
  allowance * (1 - arcade_fraction) * (1 - toy_store_fraction) = 0.90 := by
  sorry

end johns_candy_store_spending_l1690_169002


namespace max_value_theorem_l1690_169028

theorem max_value_theorem (a b c d : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a / b + b / c + c / d + d / a = 4) (h_prod : a * c = b * d) :
  ∃ (max : ℝ), max = -12 ∧ ∀ (x : ℝ), x ≤ max ∧ (∃ (a' b' c' d' : ℝ), 
    a' / b' + b' / c' + c' / d' + d' / a' = 4 ∧ 
    a' * c' = b' * d' ∧
    x = a' / c' + b' / d' + c' / a' + d' / b') :=
sorry

end max_value_theorem_l1690_169028


namespace triangle_angle_measure_l1690_169035

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Valid angle measures
  A + B + C = π →  -- Angle sum in a triangle
  2 * b * Real.cos B = a * Real.cos C + c * Real.cos A →  -- Given condition
  b^2 = 3 * a * c →  -- Given condition
  A = π/12 ∨ A = 7*π/12 := by
sorry

end triangle_angle_measure_l1690_169035


namespace line_parallel_implies_plane_perpendicular_l1690_169069

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_implies_plane_perpendicular
  (l : Line) (m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β)
  (h3 : parallel l m) :
  plane_perpendicular α β :=
sorry

end line_parallel_implies_plane_perpendicular_l1690_169069


namespace arithmetic_sequence_problem_l1690_169033

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₃ = 3 and a₅ = -3, prove a₇ = -9 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h3 : a 3 = 3) 
  (h5 : a 5 = -3) : 
  a 7 = -9 := by
  sorry


end arithmetic_sequence_problem_l1690_169033


namespace tenth_term_is_44_l1690_169076

/-- Arithmetic sequence with first term 8 and common difference 4 -/
def arithmetic_sequence (n : ℕ) : ℕ := 8 + 4 * (n - 1)

/-- The 10th term of the arithmetic sequence is 44 -/
theorem tenth_term_is_44 : arithmetic_sequence 10 = 44 := by
  sorry

end tenth_term_is_44_l1690_169076


namespace hyperbola_focal_distance_property_l1690_169014

/-- A hyperbola in a 2D plane -/
structure Hyperbola where
  -- Add necessary fields to define a hyperbola
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Foci of the hyperbola -/
def foci (h : Hyperbola) : (Point × Point) := sorry

/-- Check if a point is on the hyperbola -/
def is_on_hyperbola (h : Hyperbola) (p : Point) : Prop := sorry

/-- Diameter of the director circle -/
def director_circle_diameter (h : Hyperbola) : ℝ := sorry

theorem hyperbola_focal_distance_property (h : Hyperbola) (p : Point) :
  is_on_hyperbola h p →
  let (f1, f2) := foci h
  |distance p f1 - distance p f2| = director_circle_diameter h := by
  sorry

end hyperbola_focal_distance_property_l1690_169014


namespace log_expression_equality_l1690_169044

theorem log_expression_equality : 2 * Real.log 2 - Real.log (1 / 25) = 2 := by
  sorry

end log_expression_equality_l1690_169044


namespace rectangular_prism_ratios_l1690_169008

/-- A rectangular prism with edges a, b, c, and free surface ratios p, q, r -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ
  q : ℝ
  r : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  p_pos : 0 < p
  q_pos : 0 < q
  r_pos : 0 < r

/-- The theorem stating the edge ratios and conditions for p, q, r -/
theorem rectangular_prism_ratios (prism : RectangularPrism) :
  (prism.a : ℝ) / (prism.b : ℝ) = (2 * prism.p - 3 * prism.q + 2 * prism.r) / (-3 * prism.p + 2 * prism.q + 2 * prism.r) ∧
  (prism.b : ℝ) / (prism.c : ℝ) = (2 * prism.p + 2 * prism.q - 3 * prism.r) / (2 * prism.p - 3 * prism.q + 2 * prism.r) ∧
  (prism.c : ℝ) / (prism.a : ℝ) = (-3 * prism.p + 2 * prism.q + 2 * prism.r) / (2 * prism.p + 2 * prism.q - 3 * prism.r) ∧
  2 * prism.p + 2 * prism.r > 3 * prism.q ∧
  2 * prism.p + 2 * prism.q > 3 * prism.r ∧
  2 * prism.q + 2 * prism.r > 3 * prism.p := by
  sorry

end rectangular_prism_ratios_l1690_169008


namespace geometric_sequence_third_term_l1690_169006

/-- A geometric sequence with a_1 = 1/9 and a_5 = 9 has a_3 = 1 -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 / 9 →
  a 5 = 9 →
  a 3 = 1 := by
sorry

end geometric_sequence_third_term_l1690_169006


namespace distance_moonbase_to_skyhaven_l1690_169099

theorem distance_moonbase_to_skyhaven :
  let moonbase : ℂ := 0
  let skyhaven : ℂ := 900 + 1200 * I
  Complex.abs (skyhaven - moonbase) = 1500 := by
  sorry

end distance_moonbase_to_skyhaven_l1690_169099


namespace factorization_equality_l1690_169058

theorem factorization_equality (m n : ℝ) : 
  2*m^2 - m*n + 2*m + n - n^2 = (2*m + n)*(m - n + 1) := by
  sorry

end factorization_equality_l1690_169058


namespace jessica_journey_length_l1690_169031

/-- Represents Jessica's journey in miles -/
def journey_distance : ℝ → Prop :=
  λ total_distance =>
    ∃ (rough_trail tunnel bridge : ℝ),
      -- The journey consists of three parts
      total_distance = rough_trail + tunnel + bridge ∧
      -- The rough trail is one-quarter of the total distance
      rough_trail = (1/4) * total_distance ∧
      -- The tunnel is 25 miles long
      tunnel = 25 ∧
      -- The bridge is one-fourth of the total distance
      bridge = (1/4) * total_distance

/-- Theorem stating that Jessica's journey is 50 miles long -/
theorem jessica_journey_length :
  journey_distance 50 := by
  sorry

end jessica_journey_length_l1690_169031


namespace class_election_combinations_l1690_169038

/-- The number of candidates for class president -/
def president_candidates : ℕ := 3

/-- The number of candidates for vice president -/
def vice_president_candidates : ℕ := 5

/-- The total number of ways to choose one class president and one vice president -/
def total_ways : ℕ := president_candidates * vice_president_candidates

theorem class_election_combinations :
  total_ways = 15 :=
by sorry

end class_election_combinations_l1690_169038


namespace ellipse_line_intersection_l1690_169052

/-- Given an ellipse mx^2 + ny^2 = 1 intersecting with a line x + y - 1 = 0,
    if the slope of the line passing through the origin and the midpoint of
    the intersection points is √2/2, then n/m = √2 -/
theorem ellipse_line_intersection (m n : ℝ) :
  (∃ A B : ℝ × ℝ,
    m * A.1^2 + n * A.2^2 = 1 ∧
    m * B.1^2 + n * B.2^2 = 1 ∧
    A.1 + A.2 = 1 ∧
    B.1 + B.2 = 1 ∧
    (A ≠ B) ∧
    ((A.2 + B.2)/2) / ((A.1 + B.1)/2) = Real.sqrt 2 / 2) →
  n / m = Real.sqrt 2 := by
  sorry

end ellipse_line_intersection_l1690_169052


namespace quadratic_integer_roots_l1690_169096

theorem quadratic_integer_roots (n : ℕ+) :
  (∃ x : ℤ, x^2 - 4*x + n.val = 0) ↔ (n = 3 ∨ n = 4) := by
  sorry

end quadratic_integer_roots_l1690_169096


namespace natural_growth_determined_by_birth_and_death_rates_l1690_169010

/-- Represents the rate of change in a population -/
structure PopulationRate :=
  (value : ℝ)

/-- The natural growth rate of a population -/
def naturalGrowthRate (birthRate deathRate : PopulationRate) : PopulationRate :=
  ⟨birthRate.value - deathRate.value⟩

/-- Theorem stating that the natural growth rate is determined by both birth and death rates -/
theorem natural_growth_determined_by_birth_and_death_rates 
  (birthRate deathRate : PopulationRate) :
  ∃ (f : PopulationRate → PopulationRate → PopulationRate), 
    naturalGrowthRate birthRate deathRate = f birthRate deathRate :=
by
  sorry


end natural_growth_determined_by_birth_and_death_rates_l1690_169010


namespace A_work_days_l1690_169075

/-- The number of days B takes to finish the work alone -/
def B_days : ℕ := 15

/-- The total wages when A and B work together -/
def total_wages : ℕ := 3100

/-- A's share of the wages when working together with B -/
def A_wages : ℕ := 1860

/-- The number of days A takes to finish the work alone -/
def A_days : ℕ := 10

theorem A_work_days :
  B_days = 15 ∧
  total_wages = 3100 ∧
  A_wages = 1860 →
  A_days = 10 :=
by sorry

end A_work_days_l1690_169075


namespace triangle_side_range_l1690_169049

theorem triangle_side_range (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 2 →
  a * Real.cos C = c * Real.sin A →
  (∃ (a₁ b₁ : ℝ) (A₁ B₁ : ℝ), a₁ ≠ a ∨ b₁ ≠ b ∨ A₁ ≠ A ∨ B₁ ≠ B) →
  Real.sqrt 2 < b ∧ b < 2 :=
by sorry

end triangle_side_range_l1690_169049


namespace min_vertical_distance_l1690_169067

/-- The absolute value function -/
def abs_func (x : ℝ) : ℝ := |x - 1|

/-- The quadratic function -/
def quad_func (x : ℝ) : ℝ := -x^2 - 4*x - 3

/-- The vertical distance between the two functions -/
def vertical_distance (x : ℝ) : ℝ := abs_func x - quad_func x

theorem min_vertical_distance :
  ∃ (min_dist : ℝ), min_dist = 7/4 ∧
  ∀ (x : ℝ), vertical_distance x ≥ min_dist :=
sorry

end min_vertical_distance_l1690_169067


namespace complex_equation_solution_l1690_169088

theorem complex_equation_solution : 
  ∃ z : ℂ, z * (1 - Complex.I) = 2 * Complex.I ∧ z = 1 + Complex.I := by sorry

end complex_equation_solution_l1690_169088


namespace valid_numbers_l1690_169055

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 730000 + 10000 * a + 1000 * b + 100 * c + 6 ∧
    b < 4 ∧
    n % 56 = 0 ∧
    (a % 40 = a % 61) ∧ (a % 61 = a % 810)

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ (n = 731136 ∨ n = 737016 ∨ n = 737296) :=
by sorry

end valid_numbers_l1690_169055


namespace arithmetic_sequence_length_example_l1690_169003

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ aₙ d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: An arithmetic sequence starting with 2, ending with 1007, 
    and having a common difference of 5, contains 202 terms. -/
theorem arithmetic_sequence_length_example : 
  arithmetic_sequence_length 2 1007 5 = 202 := by
  sorry

end arithmetic_sequence_length_example_l1690_169003


namespace second_half_duration_percentage_l1690_169029

/-- Proves that the second half of a trip takes 200% longer than the first half
    given specific conditions about distance and speed. -/
theorem second_half_duration_percentage (total_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  average_speed = 40 →
  let first_half_time := (total_distance / 2) / first_half_speed
  let total_time := total_distance / average_speed
  let second_half_time := total_time - first_half_time
  (second_half_time - first_half_time) / first_half_time * 100 = 200 := by
  sorry

end second_half_duration_percentage_l1690_169029


namespace fraction_calculation_l1690_169019

theorem fraction_calculation (x y : ℚ) 
  (hx : x = 7/8) 
  (hy : y = 5/6) 
  (hx_nonzero : x ≠ 0) 
  (hy_nonzero : y ≠ 0) : 
  (4*x - 6*y) / (60*x*y) = -6/175 := by
  sorry

end fraction_calculation_l1690_169019


namespace smallest_integer_power_l1690_169004

theorem smallest_integer_power (x : ℕ) : (∀ y : ℕ, y < x → 27^y ≤ 3^24) ∧ 27^x > 3^24 ↔ x = 9 := by
  sorry

end smallest_integer_power_l1690_169004


namespace quadratic_root_problem_l1690_169090

theorem quadratic_root_problem (p : ℝ) : 
  (∃ x : ℂ, 3 * x^2 + p * x - 8 = 0 ∧ x = 2 + Complex.I) → p = -12 := by
  sorry

end quadratic_root_problem_l1690_169090


namespace equation_solution_l1690_169015

theorem equation_solution (x : ℝ) :
  x > 6 →
  (Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 6)) - 3) ↔
  x ≥ 18 :=
by sorry

end equation_solution_l1690_169015


namespace exponent_equation_l1690_169080

theorem exponent_equation (x s : ℕ) (h : (2^x) * (25^s) = 5 * (10^16)) : x = 16 := by
  sorry

end exponent_equation_l1690_169080


namespace alpha_value_l1690_169092

theorem alpha_value (α : Real) 
  (h1 : -π/2 < α ∧ α < π/2) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 2 / 2) : 
  α = -π/12 := by
sorry

end alpha_value_l1690_169092


namespace reciprocal_sum_is_one_l1690_169087

theorem reciprocal_sum_is_one :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1 :=
by sorry

end reciprocal_sum_is_one_l1690_169087


namespace extremum_at_one_l1690_169079

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

theorem extremum_at_one (a : ℝ) : 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) → 
  a = 3 := by
sorry

end extremum_at_one_l1690_169079


namespace quadratic_rewrite_sum_l1690_169036

theorem quadratic_rewrite_sum (x : ℝ) :
  ∃ (a b c : ℝ),
    (-4 * x^2 + 16 * x + 128 = a * (x + b)^2 + c) ∧
    (a + b + c = 138) := by
  sorry

end quadratic_rewrite_sum_l1690_169036


namespace acute_angles_sum_l1690_169013

theorem acute_angles_sum (x y : Real) : 
  0 < x ∧ x < π/2 →
  0 < y ∧ y < π/2 →
  4 * (Real.cos x)^2 + 3 * (Real.cos y)^2 = 1 →
  4 * Real.cos (2*x) - 3 * Real.cos (2*y) = 0 →
  x + 3*y = π/2 := by
sorry

end acute_angles_sum_l1690_169013


namespace contradiction_assumptions_l1690_169083

theorem contradiction_assumptions :
  (∀ p q : ℝ, (p^3 + q^3 = 2) → (¬(p + q ≤ 2) ↔ p + q > 2)) ∧
  (∀ a b : ℝ, |a| + |b| < 1 →
    ∃ x₁ : ℝ, x₁^2 + a*x₁ + b = 0 ∧ |x₁| ≥ 1 →
      ∃ x₂ : ℝ, x₂^2 + a*x₂ + b = 0 ∧ |x₂| ≥ 1) := by
  sorry

end contradiction_assumptions_l1690_169083


namespace count_non_adjacent_arrangements_l1690_169026

/-- The number of arrangements of 5 letters where two specific letters are not adjacent to a third specific letter -/
def non_adjacent_arrangements : ℕ :=
  let total_letters := 5
  let non_adjacent_three := 12  -- arrangements where a, b, c are not adjacent
  let adjacent_pair_not_third := 24  -- arrangements where a and b are adjacent, but not to c
  non_adjacent_three + adjacent_pair_not_third

/-- Theorem stating that the number of arrangements of a, b, c, d, e where both a and b are not adjacent to c is 36 -/
theorem count_non_adjacent_arrangements :
  non_adjacent_arrangements = 36 := by
  sorry

end count_non_adjacent_arrangements_l1690_169026


namespace akeno_extra_expenditure_l1690_169068

def akeno_expenditure : ℕ := 2985

def lev_expenditure (akeno : ℕ) : ℕ := akeno / 3

def ambrocio_expenditure (lev : ℕ) : ℕ := lev - 177

theorem akeno_extra_expenditure (akeno lev ambrocio : ℕ) 
  (h1 : akeno = akeno_expenditure)
  (h2 : lev = lev_expenditure akeno)
  (h3 : ambrocio = ambrocio_expenditure lev) :
  akeno - (lev + ambrocio) = 1172 := by
  sorry

end akeno_extra_expenditure_l1690_169068


namespace quiz_probabilities_l1690_169045

/-- Represents a quiz with multiple-choice and true/false questions -/
structure Quiz where
  total_questions : ℕ
  multiple_choice : ℕ
  true_false : ℕ
  h_total : total_questions = multiple_choice + true_false

/-- Calculates the probability of an event in a quiz draw -/
def probability (q : Quiz) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / (q.total_questions * (q.total_questions - 1))

theorem quiz_probabilities (q : Quiz) 
    (h_total : q.total_questions = 5)
    (h_mc : q.multiple_choice = 3)
    (h_tf : q.true_false = 2) :
  let p1 := probability q (q.true_false * q.multiple_choice)
  let p2 := 1 - probability q (q.true_false * (q.true_false - 1))
  p1 = 3/10 ∧ p2 = 9/10 := by
  sorry


end quiz_probabilities_l1690_169045


namespace function_monotonicity_l1690_169059

open Set
open Function

theorem function_monotonicity (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x > 0, f x > -x * deriv f x) :
  Monotone (fun x => x * f x) := by
sorry

end function_monotonicity_l1690_169059


namespace two_year_inflation_rate_real_yield_bank_deposit_l1690_169032

-- Define the annual inflation rate
def annual_inflation_rate : ℝ := 0.015

-- Define the nominal annual interest rate
def nominal_interest_rate : ℝ := 0.07

-- Theorem for two-year inflation rate
theorem two_year_inflation_rate : 
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 3.0225 := by sorry

-- Theorem for real yield of bank deposit
theorem real_yield_bank_deposit : 
  ((1 + nominal_interest_rate)^2 / (1 + ((1 + annual_inflation_rate)^2 - 1)) - 1) * 100 = 11.13 := by sorry

end two_year_inflation_rate_real_yield_bank_deposit_l1690_169032


namespace expected_red_balls_l1690_169072

/-- The expected number of red balls selected when choosing 2 balls from a box containing 4 black, 3 red, and 2 white balls -/
theorem expected_red_balls (total_balls : ℕ) (red_balls : ℕ) (selected_balls : ℕ) 
  (h_total : total_balls = 9)
  (h_red : red_balls = 3)
  (h_selected : selected_balls = 2) :
  (red_balls : ℚ) * selected_balls / total_balls = 2/3 := by
  sorry

end expected_red_balls_l1690_169072
