import Mathlib

namespace toad_ratio_is_25_to_1_l2222_222218

/-- Represents the number of toads per acre -/
structure ToadPopulation where
  green : ℕ
  brown : ℕ
  spotted_brown : ℕ

/-- The ratio of brown toads to green toads -/
def brown_to_green_ratio (pop : ToadPopulation) : ℚ :=
  pop.brown / pop.green

theorem toad_ratio_is_25_to_1 (pop : ToadPopulation) 
  (h1 : pop.green = 8)
  (h2 : pop.spotted_brown = 50)
  (h3 : pop.spotted_brown * 4 = pop.brown) : 
  brown_to_green_ratio pop = 25 := by
  sorry

end toad_ratio_is_25_to_1_l2222_222218


namespace min_coach_handshakes_correct_l2222_222254

/-- The minimum number of handshakes by coaches in a basketball tournament --/
def min_coach_handshakes : ℕ := 60

/-- Total number of handshakes in the tournament --/
def total_handshakes : ℕ := 495

/-- Number of teams in the tournament --/
def num_teams : ℕ := 2

/-- Function to calculate the number of player-to-player handshakes --/
def player_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the minimum number of handshakes by coaches --/
theorem min_coach_handshakes_correct :
  ∃ (n : ℕ), n % num_teams = 0 ∧
  player_handshakes n + (n / num_teams) * num_teams = total_handshakes ∧
  (n / num_teams) * num_teams = min_coach_handshakes :=
sorry

end min_coach_handshakes_correct_l2222_222254


namespace lcm_of_ratio_and_hcf_l2222_222296

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → 
  Nat.gcd a b = 3 → 
  Nat.lcm a b = 36 := by
sorry

end lcm_of_ratio_and_hcf_l2222_222296


namespace sachins_age_l2222_222244

theorem sachins_age (sachin rahul : ℝ) 
  (h1 : rahul = sachin + 7)
  (h2 : sachin / rahul = 7 / 9) : 
  sachin = 24.5 := by
sorry

end sachins_age_l2222_222244


namespace younger_person_age_l2222_222245

theorem younger_person_age (elder_age younger_age : ℕ) : 
  elder_age = younger_age + 20 →
  elder_age = 32 →
  elder_age - 7 = 5 * (younger_age - 7) →
  younger_age = 12 := by
sorry

end younger_person_age_l2222_222245


namespace solution_set_f_positive_range_of_m_l2222_222223

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for part I
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = Set.Iio (-1/3) ∪ Set.Ioi 3 := by sorry

-- Theorem for part II
theorem range_of_m (h : ∃ x₀ : ℝ, f x₀ + 2*m^2 < 4*m) :
  m ∈ Set.Ioo (-1/2) (5/2) := by sorry

end solution_set_f_positive_range_of_m_l2222_222223


namespace largest_cube_in_sphere_l2222_222266

theorem largest_cube_in_sphere (a b c : ℝ) (ha : a = 22) (hb : b = 2) (hc : c = 10) :
  let cuboid_diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let cube_side := Real.sqrt ((a^2 + b^2 + c^2) / 3)
  cube_side = 14 :=
sorry

end largest_cube_in_sphere_l2222_222266


namespace min_sum_reciprocals_l2222_222284

theorem min_sum_reciprocals (x y : ℕ+) (hxy : x ≠ y) (h : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ (↑a + ↑b : ℕ) = 64 ∧
    ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 15 → (↑a + ↑b : ℕ) ≤ (↑c + ↑d : ℕ) :=
by sorry

end min_sum_reciprocals_l2222_222284


namespace fish_length_difference_l2222_222203

theorem fish_length_difference : 
  let first_fish_length : Real := 0.3
  let second_fish_length : Real := 0.2
  first_fish_length - second_fish_length = 0.1 := by
  sorry

end fish_length_difference_l2222_222203


namespace triangle_max_perimeter_l2222_222236

theorem triangle_max_perimeter (a b c : ℝ) : 
  1 ≤ a ∧ a ≤ 3 ∧ 3 ≤ b ∧ b ≤ 5 ∧ 5 ≤ c ∧ c ≤ 7 →
  ∃ (p : ℝ), p = 8 + Real.sqrt 34 ∧ 
  ∀ (a' b' c' : ℝ), 1 ≤ a' ∧ a' ≤ 3 ∧ 3 ≤ b' ∧ b' ≤ 5 ∧ 5 ≤ c' ∧ c' ≤ 7 →
  (a' + b' + c' ≤ p ∧ 
   ∃ (s : ℝ), s = (a' + b' + c') / 2 ∧ 
   a' * b' * c' / (4 * s * (s - a') * (s - b') * (s - c')).sqrt ≤ 
   3 * 5 / 2) :=
by sorry

end triangle_max_perimeter_l2222_222236


namespace sea_turtle_count_sea_turtle_count_proof_l2222_222221

theorem sea_turtle_count : ℕ → Prop :=
  fun total_turtles =>
    (total_turtles : ℚ) * (1 : ℚ) / (3 : ℚ) + (28 : ℚ) = total_turtles ∧
    total_turtles = 42

-- Proof
theorem sea_turtle_count_proof : sea_turtle_count 42 := by
  sorry

end sea_turtle_count_sea_turtle_count_proof_l2222_222221


namespace book_reading_time_l2222_222235

/-- Given a book with 500 pages, prove that reading the first half at 10 pages per day
    and the second half at 5 pages per day results in a total of 75 days spent reading. -/
theorem book_reading_time (total_pages : ℕ) (first_half_speed second_half_speed : ℕ) :
  total_pages = 500 →
  first_half_speed = 10 →
  second_half_speed = 5 →
  (total_pages / 2 / first_half_speed) + (total_pages / 2 / second_half_speed) = 75 :=
by sorry


end book_reading_time_l2222_222235


namespace tan_alpha_values_l2222_222211

theorem tan_alpha_values (α : Real) 
  (h : 2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + 5 * Real.cos α ^ 2 = 3) : 
  Real.tan α = 1 ∨ Real.tan α = -2 := by
  sorry

end tan_alpha_values_l2222_222211


namespace sequence_parity_l2222_222210

def T : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => T (n + 2) + T (n + 1) - T n

theorem sequence_parity :
  (T 2021 % 2 = 1) ∧ (T 2022 % 2 = 0) ∧ (T 2023 % 2 = 1) := by
  sorry

end sequence_parity_l2222_222210


namespace hyperbola_equation_l2222_222248

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x y : ℝ) → Prop := fun x y => (x^2 / a^2) - (y^2 / b^2) = 1

/-- The focus of a hyperbola -/
def Focus := ℝ × ℝ

/-- Theorem stating the equation of a specific hyperbola -/
theorem hyperbola_equation (H : Hyperbola) (F : Focus) :
  H.equation = fun x y => (y^2 / 12) - (x^2 / 24) = 1 ↔
    F = (0, 6) ∧
    ∃ (K : Hyperbola), K.a^2 = 2 ∧ K.b^2 = 1 ∧
      (∀ x y, H.equation x y ↔ K.equation x y) :=
sorry

end hyperbola_equation_l2222_222248


namespace larger_number_problem_l2222_222242

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1345) (h3 : L = 6 * S + 15) : L = 1611 := by
  sorry

end larger_number_problem_l2222_222242


namespace dinner_arrangement_count_l2222_222260

def number_of_friends : ℕ := 5
def number_of_cooks : ℕ := 2

theorem dinner_arrangement_count :
  Nat.choose number_of_friends number_of_cooks = 10 := by
  sorry

end dinner_arrangement_count_l2222_222260


namespace quadratic_roots_ratio_l2222_222262

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ 
    (∀ x : ℝ, x^2 + 6*x + k = 0 ↔ (x = 2*r ∨ x = r)) ∧
    (2*r : ℝ) / r = 2) → 
  k = 8 := by
sorry

end quadratic_roots_ratio_l2222_222262


namespace max_value_implies_ratio_l2222_222212

/-- The function f(x) = x³ + ax² + bx - a² - 7a -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

/-- f(x) reaches a maximum value of 10 at x = 1 -/
def max_at_one (a b : ℝ) : Prop :=
  (∀ x, f a b x ≤ f a b 1) ∧ f a b 1 = 10

theorem max_value_implies_ratio (a b : ℝ) (h : max_at_one a b) : a / b = -2 / 3 := by
  sorry

end max_value_implies_ratio_l2222_222212


namespace trigonometric_sum_l2222_222271

theorem trigonometric_sum (a b : ℝ) (θ : ℝ) (h : 0 < a) (k : 0 < b) :
  (Real.sin θ ^ 6 / a + Real.cos θ ^ 6 / b = 1 / (a + b)) →
  (Real.sin θ ^ 12 / a ^ 5 + Real.cos θ ^ 12 / b ^ 5 = 1 / (a + b) ^ 5) := by
  sorry

end trigonometric_sum_l2222_222271


namespace number_puzzle_l2222_222286

theorem number_puzzle : ∃ x : ℝ, (x / 7 - x / 11 = 100) ∧ (x = 1925) := by
  sorry

end number_puzzle_l2222_222286


namespace oil_price_reduction_percentage_l2222_222207

/-- Proves that the percentage reduction in oil price is 25% given the specified conditions --/
theorem oil_price_reduction_percentage (original_price reduced_price : ℚ) : 
  reduced_price = 50 →
  (1000 / reduced_price) - (1000 / original_price) = 5 →
  (original_price - reduced_price) / original_price * 100 = 25 := by
  sorry

end oil_price_reduction_percentage_l2222_222207


namespace correct_average_calculation_l2222_222298

theorem correct_average_calculation (n : ℕ) (incorrect_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 ∧ incorrect_avg = 21 ∧ wrong_num = 26 ∧ correct_num = 36 →
  (n : ℚ) * incorrect_avg + (correct_num - wrong_num) = n * 22 := by
  sorry

end correct_average_calculation_l2222_222298


namespace determine_investment_l2222_222213

/-- Represents the investment and profit share of a person -/
structure Investor where
  investment : ℕ
  profitShare : ℕ

/-- Given two investors with a specific profit sharing ratio and one known investment,
    prove that the other investor's investment can be determined -/
theorem determine_investment (p q : Investor) (h1 : p.profitShare = 2)
    (h2 : q.profitShare = 4) (h3 : p.investment = 500000) :
    q.investment = 1000000 := by
  sorry

end determine_investment_l2222_222213


namespace wendys_cookies_l2222_222269

theorem wendys_cookies (pastries_left pastries_sold num_cupcakes : ℕ) 
  (h1 : pastries_left = 24)
  (h2 : pastries_sold = 9)
  (h3 : num_cupcakes = 4) : 
  (pastries_left + pastries_sold) - num_cupcakes = 29 := by
  sorry

#check wendys_cookies

end wendys_cookies_l2222_222269


namespace smallest_integer_satisfying_inequality_l2222_222209

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (0 : ℤ) ≤ x → x^2 < 2*x + 1 → x = 0 :=
by sorry

end smallest_integer_satisfying_inequality_l2222_222209


namespace binary_1101_equals_13_l2222_222228

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1101₂ -/
def binary_1101 : List Bool := [true, false, true, true]

theorem binary_1101_equals_13 : binary_to_decimal binary_1101 = 13 := by
  sorry

end binary_1101_equals_13_l2222_222228


namespace ellipse_equation_l2222_222216

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given an ellipse E with specific properties, prove its equation -/
theorem ellipse_equation (E : Ellipse) (F A B M : Point) :
  E.a > E.b ∧ E.b > 0 ∧  -- a > b > 0
  F = ⟨3, 0⟩ ∧  -- Right focus at F(3,0)
  (A.y - F.y) / (A.x - F.x) = 1/2 ∧  -- Line through F with slope 1/2
  (B.y - F.y) / (B.x - F.x) = 1/2 ∧  -- intersects E at A and B
  M = ⟨1, -1⟩ ∧  -- Midpoint of AB is (1,-1)
  M.x = (A.x + B.x) / 2 ∧
  M.y = (A.y + B.y) / 2 ∧
  (A.x^2 / E.a^2) + (A.y^2 / E.b^2) = 1 ∧  -- A and B lie on the ellipse
  (B.x^2 / E.a^2) + (B.y^2 / E.b^2) = 1 →
  E.a^2 = 18 ∧ E.b^2 = 9 :=
by sorry

end ellipse_equation_l2222_222216


namespace tangent_two_implies_expression_equals_negative_two_l2222_222200

-- Define the theorem
theorem tangent_two_implies_expression_equals_negative_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / 
  (Real.sin (π / 2 - θ) - Real.sin (π - θ)) = -2 := by
  sorry

end tangent_two_implies_expression_equals_negative_two_l2222_222200


namespace polynomial_factorization_l2222_222274

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^3 - 2 * b * (c - a)^3 + 3 * c * (a - b)^3 =
  (a - b) * (b - c) * (c - a) * (5 * a - 4 * b - 3 * c) := by
sorry

end polynomial_factorization_l2222_222274


namespace mary_lamb_count_l2222_222226

/-- The number of lambs Mary has after a series of events -/
def final_lamb_count (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) 
  (lambs_traded : ℕ) (extra_lambs_found : ℕ) : ℕ :=
  initial_lambs + lambs_with_babies * babies_per_lamb - lambs_traded + extra_lambs_found

/-- Theorem stating that Mary ends up with 14 lambs -/
theorem mary_lamb_count : 
  final_lamb_count 6 2 2 3 7 = 14 := by sorry

end mary_lamb_count_l2222_222226


namespace chloe_points_per_treasure_l2222_222249

/-- The number of treasures Chloe found on the first level -/
def treasures_level1 : ℕ := 6

/-- The number of treasures Chloe found on the second level -/
def treasures_level2 : ℕ := 3

/-- Chloe's total score -/
def total_score : ℕ := 81

/-- The number of points Chloe scores for each treasure -/
def points_per_treasure : ℕ := total_score / (treasures_level1 + treasures_level2)

theorem chloe_points_per_treasure :
  points_per_treasure = 9 := by
  sorry

end chloe_points_per_treasure_l2222_222249


namespace sum_integers_minus20_to_10_l2222_222250

def sum_integers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_minus20_to_10 :
  sum_integers (-20) 10 = -155 := by sorry

end sum_integers_minus20_to_10_l2222_222250


namespace octagon_area_division_l2222_222287

theorem octagon_area_division (CO OM MP PU UT TE : ℝ) (D : ℝ) :
  CO = 1 ∧ OM = 1 ∧ MP = 1 ∧ PU = 1 ∧ UT = 1 ∧ TE = 1 →
  (∃ (COMPUTER_area COMPUTED_area CDR_area : ℝ),
    COMPUTER_area = 6 ∧
    COMPUTED_area = 3 ∧
    CDR_area = 3 ∧
    COMPUTED_area = CDR_area) →
  (∃ (CD DR : ℝ),
    CD = 3 ∧
    CDR_area = 1/2 * CD * DR) →
  DR = 2 :=
sorry

end octagon_area_division_l2222_222287


namespace hiker_distance_at_blast_l2222_222214

/-- The time in seconds for which the timer is set -/
def timer_duration : ℝ := 45

/-- The speed of the hiker in yards per second -/
def hiker_speed : ℝ := 6

/-- The speed of sound in feet per second -/
def sound_speed : ℝ := 1200

/-- The distance the hiker has traveled at time t -/
def hiker_distance (t : ℝ) : ℝ := hiker_speed * t * 3

/-- The distance the sound has traveled at time t (t ≥ timer_duration) -/
def sound_distance (t : ℝ) : ℝ := sound_speed * (t - timer_duration)

/-- The time at which the hiker hears the blast -/
noncomputable def blast_time : ℝ := 
  (sound_speed * timer_duration) / (sound_speed - hiker_speed * 3)

/-- The theorem stating that the hiker's distance when they hear the blast is approximately 275 yards -/
theorem hiker_distance_at_blast : 
  ∃ ε > 0, abs (hiker_distance blast_time / 3 - 275) < ε :=
sorry

end hiker_distance_at_blast_l2222_222214


namespace triangle_inequality_l2222_222229

theorem triangle_inequality (α β γ a b c : ℝ) 
  (h_positive : α > 0 ∧ β > 0 ∧ γ > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : α + β + γ = π) : 
  a * (1/β + 1/γ) + b * (1/γ + 1/α) + c * (1/α + 1/β) ≥ 2 * (a/α + b/β + c/γ) := by
  sorry

end triangle_inequality_l2222_222229


namespace nine_pouches_sufficient_l2222_222279

-- Define the number of coins and pouches
def totalCoins : ℕ := 60
def numPouches : ℕ := 9

-- Define a type for pouch distributions
def PouchDistribution := List ℕ

-- Function to check if a distribution is valid
def isValidDistribution (d : PouchDistribution) : Prop :=
  d.length = numPouches ∧ d.sum = totalCoins

-- Function to check if a distribution can be equally split among a given number of sailors
def canSplitEqually (d : PouchDistribution) (sailors : ℕ) : Prop :=
  ∃ (groups : List (List ℕ)), 
    groups.length = sailors ∧ 
    (∀ g ∈ groups, g.sum = totalCoins / sailors) ∧
    groups.join.toFinset = d.toFinset

-- The main theorem
theorem nine_pouches_sufficient :
  ∃ (d : PouchDistribution),
    isValidDistribution d ∧
    (∀ sailors ∈ [2, 3, 4, 5], canSplitEqually d sailors) :=
sorry

end nine_pouches_sufficient_l2222_222279


namespace sequence_inequality_l2222_222295

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h1 : a 0 = 0 ∧ a (n + 1) = 0)
  (h2 : ∀ k : ℕ, k ≥ 1 ∧ k ≤ n → |a (k - 1) - 2 * a k + a (k + 1)| ≤ 1) :
  ∀ k : ℕ, k ≤ n + 1 → |a k| ≤ k * (n + 1 - k) / 2 :=
by sorry

end sequence_inequality_l2222_222295


namespace book_price_theorem_l2222_222224

/-- The price of a book on Monday when prices are 10% more than normal -/
def monday_price : ℚ := 5.50

/-- The normal price increase factor on Monday -/
def monday_factor : ℚ := 1.10

/-- The normal price decrease factor on Friday -/
def friday_factor : ℚ := 0.90

/-- The price of the book on Friday -/
def friday_price : ℚ := monday_price / monday_factor * friday_factor

theorem book_price_theorem :
  friday_price = 4.50 := by sorry

end book_price_theorem_l2222_222224


namespace arithmetic_expression_equality_l2222_222243

theorem arithmetic_expression_equality : 36 + (120 / 15) + (15 * 19) - 150 - (450 / 9) = 129 := by
  sorry

end arithmetic_expression_equality_l2222_222243


namespace license_plate_count_l2222_222251

def license_plate_combinations : ℕ :=
  (Nat.choose 26 2) * (Nat.choose 5 2) * (Nat.choose 3 2) * 24 * 10 * 9 * 8

theorem license_plate_count : license_plate_combinations = 56016000 := by
  sorry

end license_plate_count_l2222_222251


namespace sam_sandwich_count_l2222_222208

/-- The number of sandwiches Sam eats per day -/
def sandwiches_per_day : ℕ := sorry

/-- The ratio of apples to sandwiches -/
def apples_per_sandwich : ℕ := 4

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of apples eaten in a week -/
def total_apples : ℕ := 280

theorem sam_sandwich_count :
  sandwiches_per_day = 10 ∧
  sandwiches_per_day * apples_per_sandwich * days_in_week = total_apples :=
sorry

end sam_sandwich_count_l2222_222208


namespace fraction_addition_l2222_222206

theorem fraction_addition : (7 : ℚ) / 12 + (3 : ℚ) / 8 = (23 : ℚ) / 24 := by
  sorry

end fraction_addition_l2222_222206


namespace extreme_values_and_roots_l2222_222215

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_values_and_roots (a b c : ℝ) :
  (∀ x : ℝ, f' a b x = 0 ↔ x = 1 ∨ x = 3) →
  (a = -6 ∧ b = 9) ∧
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f (-6) 9 c x = 0 ∧ f (-6) 9 c y = 0 ∧ f (-6) 9 c z = 0) →
  -4 < c ∧ c < 0 :=
by sorry

end extreme_values_and_roots_l2222_222215


namespace infinitely_many_consecutive_sums_of_squares_l2222_222222

/-- A function that checks if a number is the sum of two squares --/
def isSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

/-- The main theorem stating that there are infinitely many n satisfying the condition --/
theorem infinitely_many_consecutive_sums_of_squares :
  ∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ 
    isSumOfTwoSquares n ∧
    isSumOfTwoSquares (n + 1) ∧
    isSumOfTwoSquares (n + 2) :=
sorry

end infinitely_many_consecutive_sums_of_squares_l2222_222222


namespace percentage_difference_l2222_222275

theorem percentage_difference (x y : ℝ) (h : y = x * (1 + 1/3)) :
  x = y * (1 - 1/4) :=
sorry

end percentage_difference_l2222_222275


namespace min_bad_work_percentage_l2222_222264

/-- Represents the grading system for student work -/
inductive Grade
  | Accepted
  | NotAccepted

/-- Represents the true quality of student work -/
inductive Quality
  | Good
  | Bad

/-- Neural network classification result -/
def neuralNetworkClassify (work : Quality) : Grade :=
  sorry

/-- Expert classification result -/
def expertClassify (work : Quality) : Grade :=
  sorry

/-- Probability of neural network error -/
def neuralNetworkErrorRate : ℝ := 0.1

/-- Probability of work being bad -/
def badWorkProbability : ℝ := 0.2

/-- Probability of work being good -/
def goodWorkProbability : ℝ := 1 - badWorkProbability

/-- Percentage of work rechecked by experts -/
def recheckedPercentage : ℝ :=
  badWorkProbability * (1 - neuralNetworkErrorRate) + goodWorkProbability * neuralNetworkErrorRate

/-- Theorem: The minimum percentage of bad works among those rechecked by experts is 66% -/
theorem min_bad_work_percentage :
  (badWorkProbability * (1 - neuralNetworkErrorRate)) / recheckedPercentage ≥ 0.66 := by
  sorry

end min_bad_work_percentage_l2222_222264


namespace base_n_representation_of_b_l2222_222285

theorem base_n_representation_of_b (n a b : ℕ) : 
  n > 9 → 
  n^2 - a*n + b = 0 → 
  a = 2*n + 1 → 
  b = n^2 + n := by
sorry

end base_n_representation_of_b_l2222_222285


namespace composite_solid_surface_area_l2222_222227

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The volume of a rectangular solid is the product of its length, width, and height. -/
def volume (l w h : ℕ) : ℕ := l * w * h

/-- The surface area of a rectangular solid. -/
def surfaceAreaRectangular (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

/-- The surface area of a cube. -/
def surfaceAreaCube (s : ℕ) : ℕ := 6 * s * s

theorem composite_solid_surface_area 
  (l w h : ℕ) 
  (prime_l : isPrime l) 
  (prime_w : isPrime w) 
  (prime_h : isPrime h) 
  (vol : volume l w h = 1001) :
  surfaceAreaRectangular l w h + surfaceAreaCube 13 - 13 * 13 = 1467 := by
  sorry

end composite_solid_surface_area_l2222_222227


namespace b_work_time_l2222_222283

/-- Represents the time taken by A, B, and C to complete the work individually --/
structure WorkTime where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions of the problem --/
def work_conditions (t : WorkTime) : Prop :=
  t.a = 2 * t.b ∧ 
  t.a = 3 * t.c ∧ 
  1 / t.a + 1 / t.b + 1 / t.c = 1 / 6

/-- The theorem stating that B takes 18 days to complete the work alone --/
theorem b_work_time (t : WorkTime) : work_conditions t → t.b = 18 := by
  sorry

end b_work_time_l2222_222283


namespace intersecting_lines_theorem_l2222_222238

/-- Given two lines that intersect at (-7, 9), prove that the line passing through their coefficients as points has equation -7x + 9y = 1 -/
theorem intersecting_lines_theorem (A₁ B₁ A₂ B₂ : ℝ) : 
  (A₁ * (-7) + B₁ * 9 = 1) → 
  (A₂ * (-7) + B₂ * 9 = 1) → 
  ∃ (k : ℝ), k * (A₂ - A₁) = B₂ - B₁ ∧ 
             ∀ (x y : ℝ), y - B₁ = k * (x - A₁) → -7 * x + 9 * y = 1 :=
by sorry

end intersecting_lines_theorem_l2222_222238


namespace f_min_value_l2222_222292

/-- The function f(x) = |2x-1| + |3x-2| + |4x-3| + |5x-4| -/
def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

/-- Theorem: The minimum value of f(x) is 1 -/
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 1) ∧ (∃ x : ℝ, f x = 1) := by
  sorry

end f_min_value_l2222_222292


namespace slope_and_intercept_of_3x_plus_2_l2222_222291

/-- Given a linear function y = mx + b, the slope is m and the y-intercept is b -/
def linear_function (m b : ℝ) : ℝ → ℝ := λ x ↦ m * x + b

theorem slope_and_intercept_of_3x_plus_2 :
  ∃ (f : ℝ → ℝ), f = linear_function 3 2 ∧ 
  (∀ x y : ℝ, f x - f y = 3 * (x - y)) ∧
  f 0 = 2 := by
  sorry

end slope_and_intercept_of_3x_plus_2_l2222_222291


namespace negative_two_and_negative_half_reciprocal_l2222_222237

/-- Two non-zero real numbers are reciprocal if their product is 1 -/
def IsReciprocal (a b : ℝ) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ a * b = 1

/-- -2 and -1/2 are reciprocal -/
theorem negative_two_and_negative_half_reciprocal : IsReciprocal (-2) (-1/2) := by
  sorry

end negative_two_and_negative_half_reciprocal_l2222_222237


namespace expression_simplification_l2222_222297

theorem expression_simplification (x : ℝ) (h : x = 2 * Real.sin (60 * π / 180) - Real.tan (45 * π / 180)) :
  (x / (x - 1) - 1 / (x^2 - x)) / ((x + 1)^2 / x) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l2222_222297


namespace max_peak_consumption_l2222_222239

/-- Proves that the maximum average monthly electricity consumption during peak hours is 118 kw•h -/
theorem max_peak_consumption (original_price peak_price off_peak_price total_consumption : ℝ)
  (h1 : original_price = 0.52)
  (h2 : peak_price = 0.55)
  (h3 : off_peak_price = 0.35)
  (h4 : total_consumption = 200)
  (h5 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ total_consumption →
    (original_price - peak_price) * x + (original_price - off_peak_price) * (total_consumption - x) ≥ 
    total_consumption * original_price * 0.1) :
  ∃ max_peak : ℝ, max_peak = 118 ∧ 
    ∀ x : ℝ, 0 ≤ x ∧ x ≤ total_consumption →
      (original_price - peak_price) * x + (original_price - off_peak_price) * (total_consumption - x) ≥ 
      total_consumption * original_price * 0.1 → 
      x ≤ max_peak :=
sorry

end max_peak_consumption_l2222_222239


namespace inner_quadrilateral_area_ratio_l2222_222204

-- Define the quadrilateral type
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the area function for quadrilaterals
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Define the function to get points on the sides of a quadrilateral
def getInnerPoints (q : Quadrilateral) (p : ℝ) : Quadrilateral := sorry

-- Main theorem
theorem inner_quadrilateral_area_ratio 
  (ABCD : Quadrilateral) (p : ℝ) (h : p < 0.5) :
  let A₁B₁C₁D₁ := getInnerPoints ABCD p
  area A₁B₁C₁D₁ / area ABCD = 1 - 2 * p := by sorry

end inner_quadrilateral_area_ratio_l2222_222204


namespace exists_k_for_special_sequence_l2222_222277

/-- A sequence of non-negative integers satisfying certain conditions -/
def SpecialSequence (c : Fin 1997 → ℕ) : Prop :=
  (c 1 ≥ 0) ∧
  (∀ m n : Fin 1997, m > 0 → n > 0 → m + n < 1998 →
    c m + c n ≤ c (m + n) ∧ c (m + n) ≤ c m + c n + 1)

/-- Theorem stating the existence of k for the special sequence -/
theorem exists_k_for_special_sequence (c : Fin 1997 → ℕ) (h : SpecialSequence c) :
  ∃ k : ℝ, ∀ n : Fin 1997, c n = ⌊n * k⌋ :=
sorry

end exists_k_for_special_sequence_l2222_222277


namespace divisor_problem_l2222_222256

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 158 →
  quotient = 9 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 17 := by
sorry

end divisor_problem_l2222_222256


namespace line_equation_proof_l2222_222280

/-- A line parameterized by t ∈ ℝ -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The specific line given in the problem -/
def given_line : ParametricLine where
  x := λ t => 3 * t + 6
  y := λ t => 5 * t - 7

/-- The equation of a line in slope-intercept form -/
structure LineEquation where
  slope : ℝ
  intercept : ℝ

theorem line_equation_proof (L : ParametricLine) 
    (h : L = given_line) : 
    ∃ (eq : LineEquation), 
      eq.slope = 5/3 ∧ 
      eq.intercept = -17 ∧
      ∀ t, L.y t = eq.slope * (L.x t) + eq.intercept :=
  sorry

end line_equation_proof_l2222_222280


namespace equation_roots_and_solution_l2222_222201

-- Define the equation
def equation (x p : ℝ) : Prop :=
  Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x

-- Theorem statement
theorem equation_roots_and_solution :
  ∀ p : ℝ, (∃ x : ℝ, equation x p) ↔ (0 ≤ p ∧ p ≤ 4/3) ∧
  ∀ p : ℝ, 0 ≤ p → p ≤ 4/3 → equation 1 p :=
sorry

end equation_roots_and_solution_l2222_222201


namespace fourth_root_of_2560000_l2222_222258

theorem fourth_root_of_2560000 : Real.sqrt (Real.sqrt 2560000) = 40 := by
  sorry

end fourth_root_of_2560000_l2222_222258


namespace geometric_sequence_value_l2222_222230

theorem geometric_sequence_value (x : ℝ) : 
  (∃ r : ℝ, x / 12 = r ∧ 3 / x = r) → x = 6 ∨ x = -6 := by
  sorry

end geometric_sequence_value_l2222_222230


namespace dog_bone_collection_l2222_222278

/-- Calculates the final number of bones in a dog's collection after finding and giving away some bones. -/
def final_bone_count (initial_bones : ℕ) (found_multiplier : ℕ) (bones_given_away : ℕ) : ℕ :=
  initial_bones + (initial_bones * found_multiplier) - bones_given_away

/-- Theorem stating that given the specific conditions, the dog ends up with 3380 bones. -/
theorem dog_bone_collection : final_bone_count 350 9 120 = 3380 := by
  sorry

end dog_bone_collection_l2222_222278


namespace sum_reciprocals_of_roots_l2222_222272

theorem sum_reciprocals_of_roots (x : ℝ) : 
  x^2 - 7*x + 2 = 0 → 
  ∃ a b : ℝ, (x = a ∨ x = b) ∧ (1/a + 1/b = 7/2) :=
by sorry

end sum_reciprocals_of_roots_l2222_222272


namespace complex_subtraction_multiplication_l2222_222281

theorem complex_subtraction_multiplication (i : ℂ) : 
  (7 - 3 * i) - 3 * (2 - 5 * i) = 1 + 12 * i := by sorry

end complex_subtraction_multiplication_l2222_222281


namespace functional_equation_implies_linear_l2222_222276

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

/-- The main theorem stating that any function satisfying the functional equation is linear -/
theorem functional_equation_implies_linear (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_equation_implies_linear_l2222_222276


namespace jack_hand_in_amount_l2222_222294

def calculate_amount_to_hand_in (hundred_bills two_hundred_bills fifty_bills twenty_bills ten_bills five_bills one_bills quarters dimes nickels pennies : ℕ) (amount_to_leave : ℝ) : ℝ :=
  let total_notes := 100 * hundred_bills + 50 * fifty_bills + 20 * twenty_bills + 10 * ten_bills + 5 * five_bills + one_bills
  let amount_to_hand_in := total_notes - amount_to_leave
  amount_to_hand_in

theorem jack_hand_in_amount :
  calculate_amount_to_hand_in 2 1 5 3 7 27 42 19 36 47 300 = 142 := by
  sorry

end jack_hand_in_amount_l2222_222294


namespace simplify_absolute_value_l2222_222247

theorem simplify_absolute_value : |(-4)^2 - 3^2 + 2| = 9 := by
  sorry

end simplify_absolute_value_l2222_222247


namespace ginkgo_field_length_l2222_222268

/-- The length of a field with evenly spaced trees -/
def field_length (num_trees : ℕ) (interval : ℕ) : ℕ :=
  (num_trees - 1) * interval

/-- Theorem: The length of a field with 10 ginkgo trees planted at 10-meter intervals, 
    including trees at both ends, is 90 meters. -/
theorem ginkgo_field_length : field_length 10 10 = 90 := by
  sorry

end ginkgo_field_length_l2222_222268


namespace ratio_is_two_to_one_l2222_222299

/-- An isosceles right-angled triangle with an inscribed square -/
structure IsoscelesRightTriangleWithSquare where
  /-- The side length of the isosceles right triangle -/
  x : ℝ
  /-- The distance from O to P on OB -/
  a : ℝ
  /-- The distance from O to Q on OA -/
  b : ℝ
  /-- The side length of the inscribed square PQRS -/
  s : ℝ
  /-- The side length of the triangle is positive -/
  x_pos : 0 < x
  /-- a and b are positive and their sum equals x -/
  ab_sum : 0 < a ∧ 0 < b ∧ a + b = x
  /-- The side length of the square is the sum of a and b -/
  square_side : s = a + b
  /-- The area of the square is 2/5 of the area of the triangle -/
  area_ratio : s^2 = (2/5) * (x^2/2)

/-- 
If an isosceles right-angled triangle AOB has a square PQRS inscribed as described, 
and the area of PQRS is 2/5 of the area of AOB, then the ratio of OP to OQ is 2:1.
-/
theorem ratio_is_two_to_one (t : IsoscelesRightTriangleWithSquare) : 
  t.a / t.b = 2 := by sorry

end ratio_is_two_to_one_l2222_222299


namespace twelfth_prime_l2222_222232

def is_prime (n : ℕ) : Prop := sorry

def nth_prime (n : ℕ) : ℕ := sorry

theorem twelfth_prime :
  (nth_prime 7 = 17) → (nth_prime 12 = 37) := by
  sorry

end twelfth_prime_l2222_222232


namespace triangle_abc_properties_l2222_222205

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  (a * Real.cos B + b * Real.cos A) * Real.cos (2 * C) = c * Real.cos C →
  b = 2 * a →
  S = (Real.sqrt 3 / 2) * Real.sin A * Real.sin B →
  C = 2 * Real.pi / 3 ∧
  Real.sin A = Real.sqrt 21 / 14 ∧
  c = Real.sqrt 6 / 2 :=
by sorry

end triangle_abc_properties_l2222_222205


namespace factorization_equality_l2222_222219

theorem factorization_equality (a b x y : ℝ) :
  8 * a * x - b * y + 4 * a * y - 2 * b * x = (4 * a - b) * (2 * x + y) := by
  sorry

end factorization_equality_l2222_222219


namespace draw_specific_sequence_l2222_222270

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the number of cards of each rank (Ace, King, Queen, Jack) -/
def rank_count : ℕ := 4

/-- Represents the number of cards in the hearts suit -/
def hearts_count : ℕ := 13

/-- Calculates the probability of drawing the specified sequence of cards -/
def draw_probability (d : Deck) : ℚ :=
  (rank_count : ℚ) / 52 *
  (rank_count : ℚ) / 51 *
  (rank_count : ℚ) / 50 *
  (rank_count : ℚ) / 49 *
  ((hearts_count - rank_count) : ℚ) / 48

/-- The theorem stating the probability of drawing the specified sequence of cards -/
theorem draw_specific_sequence (d : Deck) :
  draw_probability d = 2304 / 31187500 := by
  sorry

end draw_specific_sequence_l2222_222270


namespace tangent_line_equations_l2222_222267

/-- The function for which we're finding the tangent line -/
def f (x : ℝ) : ℝ := x^3 - 2*x

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The point through which the tangent line passes -/
def P : ℝ × ℝ := (2, 4)

/-- Theorem: The equations of the tangent lines to y = x³ - 2x passing through (2,4) -/
theorem tangent_line_equations :
  ∃ (m : ℝ), (f m = m^3 - 2*m) ∧
             ((4 - f m = (f' m) * (2 - m)) ∧
              (∀ x, f' m * (x - m) + f m = 10*x - 16 ∨
                    f' m * (x - m) + f m = x + 2)) :=
sorry

end tangent_line_equations_l2222_222267


namespace line_segment_properties_l2222_222290

/-- Given a line segment with endpoints (1, 4) and (7, 18), prove properties about its midpoint and slope -/
theorem line_segment_properties :
  let x₁ : ℝ := 1
  let y₁ : ℝ := 4
  let x₂ : ℝ := 7
  let y₂ : ℝ := 18
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  let slope : ℝ := (y₂ - y₁) / (x₂ - x₁)
  (midpoint_x + midpoint_y = 15) ∧ (slope = 7 / 3) := by
  sorry

end line_segment_properties_l2222_222290


namespace cricket_team_right_handed_players_l2222_222252

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 64)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0)
  : (total_players - throwers) * 2 / 3 + throwers = 55 := by
  sorry

end cricket_team_right_handed_players_l2222_222252


namespace fraction_sum_equality_l2222_222225

theorem fraction_sum_equality : (3 : ℚ) / 8 - 5 / 6 + 9 / 4 = 43 / 24 := by
  sorry

end fraction_sum_equality_l2222_222225


namespace ellipse_m_value_l2222_222255

/-- An ellipse with equation mx^2 + y^2 = 1, foci on the y-axis, and major axis length three times the minor axis length has m = 4/9 --/
theorem ellipse_m_value (m : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), m * x^2 + y^2 = 1 ↔ x^2 / b^2 + y^2 / a^2 = 1) ∧ 
    (∃ (c : ℝ), c > 0 ∧ a^2 = b^2 + c^2) ∧ 
    2 * a = 3 * (2 * b)) →
  m = 4/9 := by
sorry

end ellipse_m_value_l2222_222255


namespace angle_ABC_measure_l2222_222240

-- Define the triangle ABC and point D
variable (A B C D : EuclideanPlane)

-- Define the conditions
def on_line_segment (D A C : EuclideanPlane) : Prop := sorry

-- Angle measures in degrees
def angle_measure (p q r : EuclideanPlane) : ℝ := sorry

-- Sum of angles around a point
def angle_sum_around_point (p : EuclideanPlane) : ℝ := sorry

-- Theorem statement
theorem angle_ABC_measure
  (h1 : on_line_segment D A C)
  (h2 : angle_measure A B D = 70)
  (h3 : angle_sum_around_point B = 200)
  (h4 : angle_measure C B D = 60) :
  angle_measure A B C = 70 := by sorry

end angle_ABC_measure_l2222_222240


namespace square_sum_from_difference_and_product_l2222_222273

theorem square_sum_from_difference_and_product (a b : ℝ) 
  (h1 : a - b = 8) 
  (h2 : a * b = 50) : 
  a^2 + b^2 = 164 := by
sorry

end square_sum_from_difference_and_product_l2222_222273


namespace sum_of_consecutive_odds_l2222_222253

def is_valid_sum (n : ℕ) : Prop :=
  ∃ (k : ℤ), 
    (4 * k + 12 = n) ∧ 
    (k % 2 = 1) ∧ 
    ((4 * k + 4) % 10 = 0)

theorem sum_of_consecutive_odds : 
  is_valid_sum 28 ∧ 
  is_valid_sum 52 ∧ 
  is_valid_sum 84 ∧ 
  is_valid_sum 220 ∧ 
  ¬(is_valid_sum 112) :=
sorry

end sum_of_consecutive_odds_l2222_222253


namespace kth_roots_sum_power_real_l2222_222259

theorem kth_roots_sum_power_real (k : ℕ) (x y : ℂ) 
  (hx : x^k = 1) (hy : y^k = 1) : 
  ∃ (r : ℝ), (x + y)^k = r := by sorry

end kth_roots_sum_power_real_l2222_222259


namespace fraction_multiplication_equals_decimal_l2222_222293

theorem fraction_multiplication_equals_decimal : 
  (1 : ℚ) / 3 * (3 : ℚ) / 7 * (7 : ℚ) / 8 = 0.12499999999999997 := by
  sorry

end fraction_multiplication_equals_decimal_l2222_222293


namespace arithmetic_sequence_difference_l2222_222231

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The sum of specific terms in the sequence equals 80 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 + a 8 + a 10 = 80

theorem arithmetic_sequence_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : SumCondition a) :
  ∃ d : ℝ, a 7 - a 8 = -d ∧ ArithmeticSequence a ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
sorry

end arithmetic_sequence_difference_l2222_222231


namespace profit_percentage_equality_l2222_222233

/-- Represents the discount percentage as a rational number -/
def discount : ℚ := 5 / 100

/-- Represents the profit percentage with discount as a rational number -/
def profit_with_discount : ℚ := 2255 / 10000

/-- Theorem stating that the profit percentage without discount is equal to the profit percentage with discount -/
theorem profit_percentage_equality :
  profit_with_discount = (profit_with_discount * (1 - discount)⁻¹) := by sorry

end profit_percentage_equality_l2222_222233


namespace sqrt_simplification_l2222_222257

theorem sqrt_simplification : 3 * Real.sqrt 20 - 5 * Real.sqrt (1/5) = 5 * Real.sqrt 5 := by
  sorry

end sqrt_simplification_l2222_222257


namespace sqrt_equation_sum_l2222_222263

theorem sqrt_equation_sum (n a t : ℝ) (hn : n ≥ 2) (ha : a > 0) (ht : t > 0) :
  Real.sqrt (n + a / t) = n * Real.sqrt (a / t) → a + t = n^2 + n - 1 := by
  sorry

end sqrt_equation_sum_l2222_222263


namespace mangoes_distribution_l2222_222289

theorem mangoes_distribution (total : ℕ) (neighbors : ℕ) 
  (h1 : total = 560) (h2 : neighbors = 8) : 
  (total / 2) / neighbors = 35 := by
  sorry

end mangoes_distribution_l2222_222289


namespace regular_pentagon_most_symmetric_l2222_222282

/-- Represents a geometric figure -/
inductive Figure
  | EquilateralTriangle
  | NonSquareRhombus
  | NonSquareRectangle
  | IsoscelesTrapezoid
  | RegularPentagon

/-- Returns the number of lines of symmetry for a given figure -/
def linesOfSymmetry (f : Figure) : ℕ :=
  match f with
  | Figure.EquilateralTriangle => 3
  | Figure.NonSquareRhombus => 2
  | Figure.NonSquareRectangle => 2
  | Figure.IsoscelesTrapezoid => 1
  | Figure.RegularPentagon => 5

/-- Theorem stating that the regular pentagon has the greatest number of lines of symmetry -/
theorem regular_pentagon_most_symmetric :
  ∀ f : Figure, f ≠ Figure.RegularPentagon → linesOfSymmetry Figure.RegularPentagon > linesOfSymmetry f :=
by sorry

end regular_pentagon_most_symmetric_l2222_222282


namespace equation_has_one_solution_l2222_222202

theorem equation_has_one_solution :
  ∃! x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 2 :=
by sorry

end equation_has_one_solution_l2222_222202


namespace proposition_equivalence_l2222_222288

theorem proposition_equivalence (a b c : ℝ) :
  (a ≤ b → a * c^2 ≤ b * c^2) ↔ (a * c^2 > b * c^2 → a > b) :=
by sorry

end proposition_equivalence_l2222_222288


namespace complex_equation_solution_l2222_222246

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I + a) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end complex_equation_solution_l2222_222246


namespace sum_of_P_roots_l2222_222261

variable (a b c d : ℂ)

def P (X : ℂ) : ℂ := X^6 - X^5 - X^4 - X^3 - X

theorem sum_of_P_roots :
  (a^4 - a^3 - a^2 - 1 = 0) →
  (b^4 - b^3 - b^2 - 1 = 0) →
  (c^4 - c^3 - c^2 - 1 = 0) →
  (d^4 - d^3 - d^2 - 1 = 0) →
  P a + P b + P c + P d = -2 := by sorry

end sum_of_P_roots_l2222_222261


namespace cone_apex_angle_l2222_222241

theorem cone_apex_angle (R : ℝ) (h : R > 0) :
  let lateral_surface := π * R^2 / 2
  let base_circumference := π * R
  lateral_surface = base_circumference * R / 2 →
  let base_diameter := R
  let apex_angle := 2 * Real.arcsin (base_diameter / (2 * R))
  apex_angle = π / 3 := by
  sorry

end cone_apex_angle_l2222_222241


namespace circle_equation_proof_l2222_222234

/-- Prove that the equation (x-1)^2 + (y-1)^2 = 4 represents the circle that passes through
    points A(1,-1) and B(-1,1), and has its center on the line x+y-2=0. -/
theorem circle_equation_proof (x y : ℝ) : 
  (∀ (cx cy : ℝ), cx + cy = 2 →
    (cx - 1)^2 + (cy - 1)^2 = 4 ∧
    (1 - cx)^2 + (-1 - cy)^2 = 4 ∧
    (-1 - cx)^2 + (1 - cy)^2 = 4) ↔
  (x - 1)^2 + (y - 1)^2 = 4 := by
sorry

end circle_equation_proof_l2222_222234


namespace cricketer_average_score_l2222_222217

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (first_matches : ℕ) 
  (last_matches : ℕ) 
  (first_average : ℚ) 
  (last_average : ℚ) 
  (h1 : total_matches = first_matches + last_matches) 
  (h2 : total_matches = 10) 
  (h3 : first_matches = 6) 
  (h4 : last_matches = 4) 
  (h5 : first_average = 42) 
  (h6 : last_average = 34.25) : 
  (first_average * first_matches + last_average * last_matches) / total_matches = 38.9 := by
sorry

end cricketer_average_score_l2222_222217


namespace cos_54_degrees_l2222_222265

theorem cos_54_degrees : Real.cos (54 * π / 180) = (3 - Real.sqrt 5) / 8 := by
  sorry

end cos_54_degrees_l2222_222265


namespace turtle_race_times_l2222_222220

/-- The time it took for Greta's turtle to finish the race -/
def greta_time : ℕ := sorry

/-- The time it took for George's turtle to finish the race -/
def george_time : ℕ := sorry

/-- The time it took for Gloria's turtle to finish the race -/
def gloria_time : ℕ := 8

theorem turtle_race_times :
  (george_time = greta_time - 2) ∧
  (gloria_time = 2 * george_time) ∧
  (greta_time = 6) := by sorry

end turtle_race_times_l2222_222220
