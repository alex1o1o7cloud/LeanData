import Mathlib

namespace NUMINAMATH_CALUDE_overtime_increase_is_25_percent_l3381_338167

/-- Calculates the percentage increase for overtime pay given basic pay and total wage information. -/
def overtime_percentage_increase (basic_pay : ℚ) (total_wage : ℚ) (basic_hours : ℕ) (total_hours : ℕ) : ℚ :=
  let basic_rate : ℚ := basic_pay / basic_hours
  let overtime_hours : ℕ := total_hours - basic_hours
  let overtime_pay : ℚ := total_wage - basic_pay
  let overtime_rate : ℚ := overtime_pay / overtime_hours
  ((overtime_rate - basic_rate) / basic_rate) * 100

/-- Theorem stating that given the specified conditions, the overtime percentage increase is 25%. -/
theorem overtime_increase_is_25_percent :
  overtime_percentage_increase 20 25 40 48 = 25 := by
  sorry

end NUMINAMATH_CALUDE_overtime_increase_is_25_percent_l3381_338167


namespace NUMINAMATH_CALUDE_impossible_2018_after_2019_l3381_338148

/-- Represents a single step in the room occupancy change --/
inductive Step
  | Enter : Step  -- Two people enter (+2)
  | Exit : Step   -- One person exits (-1)

/-- Calculates the change in room occupancy for a given step --/
def stepChange (s : Step) : Int :=
  match s with
  | Step.Enter => 2
  | Step.Exit => -1

/-- Represents a sequence of steps over time --/
def Sequence := List Step

/-- Calculates the final room occupancy given a sequence of steps --/
def finalOccupancy (seq : Sequence) : Int :=
  seq.foldl (fun acc s => acc + stepChange s) 0

/-- Theorem: It's impossible to have 2018 people after 2019 steps --/
theorem impossible_2018_after_2019 :
  ∀ (seq : Sequence), seq.length = 2019 → finalOccupancy seq ≠ 2018 :=
by
  sorry


end NUMINAMATH_CALUDE_impossible_2018_after_2019_l3381_338148


namespace NUMINAMATH_CALUDE_y_is_75_percent_of_x_l3381_338100

/-- Given that 45% of z equals 96% of y and z equals 160% of x, prove that y equals 75% of x -/
theorem y_is_75_percent_of_x (x y z : ℝ) 
  (h1 : 0.45 * z = 0.96 * y) 
  (h2 : z = 1.60 * x) : 
  y = 0.75 * x := by
sorry

end NUMINAMATH_CALUDE_y_is_75_percent_of_x_l3381_338100


namespace NUMINAMATH_CALUDE_inverse_of_complex_expression_l3381_338120

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem inverse_of_complex_expression :
  i ^ 2 = -1 →
  (3 * i - 2 * i⁻¹)⁻¹ = -i / 5 :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_complex_expression_l3381_338120


namespace NUMINAMATH_CALUDE_max_n_for_factorizable_quadratic_l3381_338197

/-- Given a quadratic expression 5x^2 + nx + 60 that can be factored as the product
    of two linear factors with integer coefficients, the maximum possible value of n is 301. -/
theorem max_n_for_factorizable_quadratic : 
  ∀ n : ℤ, 
  (∃ a b : ℤ, ∀ x : ℤ, 5 * x^2 + n * x + 60 = (5 * x + a) * (x + b)) →
  n ≤ 301 :=
by sorry

end NUMINAMATH_CALUDE_max_n_for_factorizable_quadratic_l3381_338197


namespace NUMINAMATH_CALUDE_power_function_property_l3381_338170

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f 4 / f 2 = 3) :
  f (1/2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l3381_338170


namespace NUMINAMATH_CALUDE_popcorn_distribution_l3381_338110

/-- Given the conditions of the popcorn problem, prove that each of Jared's friends can eat 60 pieces of popcorn. -/
theorem popcorn_distribution (pieces_per_serving : ℕ) (jared_pieces : ℕ) (num_friends : ℕ) (total_servings : ℕ)
  (h1 : pieces_per_serving = 30)
  (h2 : jared_pieces = 90)
  (h3 : num_friends = 3)
  (h4 : total_servings = 9) :
  (total_servings * pieces_per_serving - jared_pieces) / num_friends = 60 :=
by sorry

end NUMINAMATH_CALUDE_popcorn_distribution_l3381_338110


namespace NUMINAMATH_CALUDE_eighth_term_value_l3381_338182

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first six terms
  sum_six : ℚ
  -- The seventh term
  seventh_term : ℚ

/-- Theorem: Given an arithmetic sequence where the sum of the first six terms is 21
    and the seventh term is 8, the eighth term is 65/7 -/
theorem eighth_term_value (seq : ArithmeticSequence)
    (h1 : seq.sum_six = 21)
    (h2 : seq.seventh_term = 8) :
    ∃ (a d : ℚ), a + 7 * d = 65 / 7 ∧
                 6 * a + 15 * d = 21 ∧
                 a + 6 * d = 8 :=
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l3381_338182


namespace NUMINAMATH_CALUDE_factorization_equality_l3381_338192

theorem factorization_equality (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^3 + b^3 + c^3 - 3*a*b*c) :=
by sorry

end NUMINAMATH_CALUDE_factorization_equality_l3381_338192


namespace NUMINAMATH_CALUDE_triangle_inequality_l3381_338157

theorem triangle_inequality (a b c : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_triangle : a^2 + b^2 ≥ c^2 ∧ b^2 + c^2 ≥ a^2 ∧ c^2 + a^2 ≥ b^2) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3381_338157


namespace NUMINAMATH_CALUDE_triangle_theorem_l3381_338193

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  angle_sum : A + B + C = π
  tan_condition : 2 * (Real.tan A + Real.tan B) = Real.tan A / Real.cos B + Real.tan B / Real.cos A

/-- Main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) : 
  t.a + t.b = 2 * t.c ∧ Real.cos t.C ≥ 1/2 ∧ ∃ (t' : Triangle), Real.cos t'.C = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3381_338193


namespace NUMINAMATH_CALUDE_domino_arrangements_count_l3381_338121

structure Grid :=
  (rows : Nat)
  (cols : Nat)

structure Domino :=
  (length : Nat)
  (width : Nat)

def count_arrangements (g : Grid) (d : Domino) (num_dominoes : Nat) : Nat :=
  Nat.choose (g.rows + g.cols - 2) (g.cols - 1)

theorem domino_arrangements_count (g : Grid) (d : Domino) (num_dominoes : Nat) :
  g.rows = 6 → g.cols = 4 → d.length = 2 → d.width = 1 → num_dominoes = 4 →
  count_arrangements g d num_dominoes = 126 := by
  sorry

end NUMINAMATH_CALUDE_domino_arrangements_count_l3381_338121


namespace NUMINAMATH_CALUDE_divisible_by_five_l3381_338151

theorem divisible_by_five (n : ℕ) : ∃ k : ℤ, (n^5 : ℤ) + 4*n = 5*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l3381_338151


namespace NUMINAMATH_CALUDE_union_A_B_when_a_zero_complement_A_intersect_B_nonempty_iff_l3381_338108

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 2 < x ∧ x < a + 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2)*x + 2*a = 0}

-- Theorem 1
theorem union_A_B_when_a_zero :
  A 0 ∪ B 0 = {x | -2 < x ∧ x ≤ 2} := by sorry

-- Theorem 2
theorem complement_A_intersect_B_nonempty_iff (a : ℝ) :
  ((Set.univ \ A a) ∩ B a).Nonempty ↔ a ≤ 0 ∨ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_zero_complement_A_intersect_B_nonempty_iff_l3381_338108


namespace NUMINAMATH_CALUDE_expected_winning_percentage_approx_l3381_338169

/-- Represents the political parties --/
inductive Party
  | Republican
  | Democrat
  | Independent

/-- Represents a candidate in the election --/
inductive Candidate
  | X
  | Y

/-- The ratio of registered voters for each party --/
def partyRatio : Party → ℚ
  | Party.Republican => 3
  | Party.Democrat => 2
  | Party.Independent => 1

/-- The percentage of voters from each party expected to vote for Candidate X --/
def votePercentageForX : Party → ℚ
  | Party.Republican => 85 / 100
  | Party.Democrat => 60 / 100
  | Party.Independent => 40 / 100

/-- The total number of registered voters (assumed to be 6n for some positive integer n) --/
def totalVoters : ℚ := 6

/-- Calculate the expected winning percentage for Candidate X --/
def expectedWinningPercentage : ℚ :=
  let votesForX := (partyRatio Party.Republican * votePercentageForX Party.Republican +
                    partyRatio Party.Democrat * votePercentageForX Party.Democrat +
                    partyRatio Party.Independent * votePercentageForX Party.Independent)
  let votesForY := totalVoters - votesForX
  (votesForX - votesForY) / totalVoters * 100

/-- Theorem stating that the expected winning percentage for Candidate X is approximately 38.33% --/
theorem expected_winning_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 / 100 ∧ |expectedWinningPercentage - 3833 / 100| < ε :=
sorry

end NUMINAMATH_CALUDE_expected_winning_percentage_approx_l3381_338169


namespace NUMINAMATH_CALUDE_hyperbola_satisfies_equation_l3381_338114

/-- A hyperbola with given asymptotes and passing through a specific point -/
structure Hyperbola where
  -- The slope of the asymptotes
  asymptote_slope : ℝ
  -- The point through which the hyperbola passes
  point : ℝ × ℝ

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2 / 14 - y^2 / 7 = 1

/-- Theorem stating that the given hyperbola satisfies the equation -/
theorem hyperbola_satisfies_equation (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = 1/2)
  (h_point : h.point = (4, Real.sqrt 2)) :
  hyperbola_equation h 4 (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_satisfies_equation_l3381_338114


namespace NUMINAMATH_CALUDE_triangle_side_length_l3381_338113

theorem triangle_side_length 
  (A B C : Real) 
  (a b c : Real) 
  (h_area : (1/2) * b * c * Real.sin A = Real.sqrt 3)
  (h_angle : B = 60 * π / 180)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3381_338113


namespace NUMINAMATH_CALUDE_round_trip_time_l3381_338191

/-- Calculates the total time for a round trip boat journey -/
theorem round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 9)
  (h2 : stream_speed = 6)
  (h3 : distance = 170) :
  (distance / (boat_speed - stream_speed)) + (distance / (boat_speed + stream_speed)) = 68 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_time_l3381_338191


namespace NUMINAMATH_CALUDE_almond_salami_cheese_cost_l3381_338133

/-- The cost of Sean's Sunday purchases -/
def sean_sunday_cost (almond_croissant : ℝ) (salami_cheese_croissant : ℝ) : ℝ :=
  almond_croissant + salami_cheese_croissant + 3 + 4 + 2 * 2.5

/-- Theorem stating the combined cost of almond and salami & cheese croissants -/
theorem almond_salami_cheese_cost :
  ∃ (almond_croissant salami_cheese_croissant : ℝ),
    sean_sunday_cost almond_croissant salami_cheese_croissant = 21 ∧
    almond_croissant + salami_cheese_croissant = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_almond_salami_cheese_cost_l3381_338133


namespace NUMINAMATH_CALUDE_no_three_rational_solutions_l3381_338139

theorem no_three_rational_solutions :
  ¬ ∃ (r : ℝ), ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (x^3 - 2023*x^2 - 2023*x + r = 0) ∧
    (y^3 - 2023*y^2 - 2023*y + r = 0) ∧
    (z^3 - 2023*z^2 - 2023*z + r = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_three_rational_solutions_l3381_338139


namespace NUMINAMATH_CALUDE_probability_at_least_half_even_dice_l3381_338147

theorem probability_at_least_half_even_dice (dice : Nat) (p_even : ℝ) :
  dice = 4 →
  p_even = 1/2 →
  let p_two_even := Nat.choose dice 2 * p_even^2 * (1 - p_even)^2
  let p_three_even := Nat.choose dice 3 * p_even^3 * (1 - p_even)
  let p_four_even := p_even^4
  p_two_even + p_three_even + p_four_even = 11/16 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_half_even_dice_l3381_338147


namespace NUMINAMATH_CALUDE_no_primes_satisfying_equation_l3381_338179

theorem no_primes_satisfying_equation : 
  ¬ ∃ (a b c d : ℕ), 
    Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧
    a < b ∧ b < c ∧ c < d ∧
    (1 : ℚ) / a + (1 : ℚ) / d = (1 : ℚ) / b + (1 : ℚ) / c :=
by sorry

end NUMINAMATH_CALUDE_no_primes_satisfying_equation_l3381_338179


namespace NUMINAMATH_CALUDE_largest_base4_3digit_decimal_l3381_338119

/-- The largest three-digit number in base-4 -/
def largest_base4_3digit : ℕ := 3 * 4^2 + 3 * 4 + 3

/-- Conversion from base-4 to base-10 -/
def base4_to_decimal (n : ℕ) : ℕ := n

theorem largest_base4_3digit_decimal :
  base4_to_decimal largest_base4_3digit = 63 := by sorry

end NUMINAMATH_CALUDE_largest_base4_3digit_decimal_l3381_338119


namespace NUMINAMATH_CALUDE_circle_equation_l3381_338187

-- Define the line l: x - 2y - 1 = 0
def line_l (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the circle C
def circle_C (center_x center_y radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center_x)^2 + (p.2 - center_y)^2 = radius^2}

theorem circle_equation :
  ∃ (center_x center_y radius : ℝ),
    (line_l center_x center_y) ∧
    ((2 : ℝ), 1) ∈ circle_C center_x center_y radius ∧
    ((1 : ℝ), 2) ∈ circle_C center_x center_y radius ∧
    center_x = -1 ∧ center_y = -1 ∧ radius^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3381_338187


namespace NUMINAMATH_CALUDE_range_of_m_for_propositions_l3381_338178

theorem range_of_m_for_propositions (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) ∨
  (∀ x : ℝ, 4*x^2 + 4*(m+2)*x + 1 ≠ 0) →
  m < -1 := by sorry

end NUMINAMATH_CALUDE_range_of_m_for_propositions_l3381_338178


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l3381_338188

/-- Represents a student number in the range [1, 1000] -/
def StudentNumber := Fin 1000

/-- The total number of students -/
def totalStudents : Nat := 1000

/-- The number of students to be selected in the sample -/
def sampleSize : Nat := 100

/-- The interval between selected students in systematic sampling -/
def samplingInterval : Nat := totalStudents / sampleSize

/-- Predicate to determine if a student number is selected in the systematic sample -/
def isSelected (n : StudentNumber) : Prop :=
  n.val % samplingInterval = 122 % samplingInterval

theorem systematic_sampling_theorem :
  isSelected ⟨121, by norm_num⟩ → isSelected ⟨926, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l3381_338188


namespace NUMINAMATH_CALUDE_dans_initial_money_l3381_338130

/-- The amount of money Dan has left after buying the candy bar -/
def money_left : ℝ := 3

/-- The cost of the candy bar -/
def candy_cost : ℝ := 2

/-- Dan's initial amount of money -/
def initial_money : ℝ := money_left + candy_cost

theorem dans_initial_money : initial_money = 5 := by sorry

end NUMINAMATH_CALUDE_dans_initial_money_l3381_338130


namespace NUMINAMATH_CALUDE_pressure_force_half_ellipse_l3381_338104

/-- Pressure force on a vertically immersed half-elliptical plate -/
theorem pressure_force_half_ellipse (a b ρ g : ℝ) (ha : a > 0) (hb : b > 0) (hρ : ρ > 0) (hg : g > 0) :
  let F := (2 * b * a^2 / 3) * ρ * g
  ∃ (force : ℝ), force = F ∧ 
    force = ∫ (x : ℝ) in -a..a, ρ * g * x * (b / a * Real.sqrt (a^2 - x^2)) :=
by sorry

end NUMINAMATH_CALUDE_pressure_force_half_ellipse_l3381_338104


namespace NUMINAMATH_CALUDE_a_8_equals_15_l3381_338183

/-- The sum of the first n terms of the sequence {aₙ} -/
def S (n : ℕ) : ℕ := n^2

/-- The n-th term of the sequence {aₙ} -/
def a (n : ℕ) : ℕ := S n - S (n-1)

/-- Theorem stating that the 8th term of the sequence is 15 -/
theorem a_8_equals_15 : a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_a_8_equals_15_l3381_338183


namespace NUMINAMATH_CALUDE_f_max_value_f_min_value_l3381_338115

/-- The function f(x) = 2x³ - 6x² - 18x + 7 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

/-- The maximum value of f(x) is 17 -/
theorem f_max_value : ∃ (x : ℝ), f x = 17 ∧ ∀ (y : ℝ), f y ≤ 17 := by sorry

/-- The minimum value of f(x) is -47 -/
theorem f_min_value : ∃ (x : ℝ), f x = -47 ∧ ∀ (y : ℝ), f y ≥ -47 := by sorry

end NUMINAMATH_CALUDE_f_max_value_f_min_value_l3381_338115


namespace NUMINAMATH_CALUDE_min_sum_squares_l3381_338186

theorem min_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  ∃ (m : ℝ), m = 1/3 ∧ a^2 + b^2 + c^2 ≥ m ∧ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ x^2 + y^2 + z^2 = m :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3381_338186


namespace NUMINAMATH_CALUDE_half_power_inequality_l3381_338181

theorem half_power_inequality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) :
  (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_half_power_inequality_l3381_338181


namespace NUMINAMATH_CALUDE_find_m_l3381_338137

def U : Set Int := {-1, 2, 3, 6}

def A (m : Int) : Set Int := {x ∈ U | x^2 - 5*x + m = 0}

theorem find_m : ∃ m : Int, A m = {-1, 6} ∧ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3381_338137


namespace NUMINAMATH_CALUDE_golf_cost_calculation_l3381_338155

/-- Proves that given the cost of one round of golf and the number of rounds that can be played,
    the total amount of money is correctly calculated. -/
theorem golf_cost_calculation (cost_per_round : ℕ) (num_rounds : ℕ) (total_money : ℕ) :
  cost_per_round = 80 →
  num_rounds = 5 →
  total_money = cost_per_round * num_rounds →
  total_money = 400 := by
sorry

end NUMINAMATH_CALUDE_golf_cost_calculation_l3381_338155


namespace NUMINAMATH_CALUDE_tiffany_bags_on_monday_l3381_338126

/-- The number of bags Tiffany had on Monday -/
def bags_on_monday : ℕ := sorry

/-- The number of bags Tiffany found on Tuesday -/
def bags_on_tuesday : ℕ := 4

/-- The total number of bags Tiffany had -/
def total_bags : ℕ := 8

/-- Theorem: Tiffany had 4 bags on Monday -/
theorem tiffany_bags_on_monday : bags_on_monday = 4 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_on_monday_l3381_338126


namespace NUMINAMATH_CALUDE_petrol_price_increase_l3381_338189

theorem petrol_price_increase (original_price original_consumption : ℝ) 
  (h : original_price > 0) (h2 : original_consumption > 0) :
  let new_consumption := original_consumption * (1 - 1/6)
  let price_increase_factor := (original_price * original_consumption) / (original_price * new_consumption)
  price_increase_factor = 1.2 := by
sorry

end NUMINAMATH_CALUDE_petrol_price_increase_l3381_338189


namespace NUMINAMATH_CALUDE_smallest_multiple_35_with_digit_product_35_l3381_338103

/-- Given a natural number, return the product of its digits. -/
def digit_product (n : ℕ) : ℕ := sorry

/-- Given a natural number, check if it's a multiple of 35. -/
def is_multiple_of_35 (n : ℕ) : Prop := ∃ k : ℕ, n = 35 * k

theorem smallest_multiple_35_with_digit_product_35 :
  ∀ n : ℕ, n > 0 → is_multiple_of_35 n → is_multiple_of_35 (digit_product n) →
  n ≥ 735 ∧ (n = 735 → is_multiple_of_35 (digit_product 735)) := by sorry

end NUMINAMATH_CALUDE_smallest_multiple_35_with_digit_product_35_l3381_338103


namespace NUMINAMATH_CALUDE_jake_sausage_spending_l3381_338180

/-- Represents a type of sausage package -/
structure SausagePackage where
  weight : Real
  price_per_pound : Real

/-- Calculates the total cost for a given number of packages of a specific type -/
def total_cost_for_type (package : SausagePackage) (num_packages : Nat) : Real :=
  package.weight * package.price_per_pound * num_packages

/-- Theorem: Jake spends $52 on sausages -/
theorem jake_sausage_spending :
  let type1 : SausagePackage := { weight := 2, price_per_pound := 4 }
  let type2 : SausagePackage := { weight := 1.5, price_per_pound := 5 }
  let type3 : SausagePackage := { weight := 3, price_per_pound := 3.5 }
  let num_packages : Nat := 2
  total_cost_for_type type1 num_packages +
  total_cost_for_type type2 num_packages +
  total_cost_for_type type3 num_packages = 52 := by
  sorry

end NUMINAMATH_CALUDE_jake_sausage_spending_l3381_338180


namespace NUMINAMATH_CALUDE_unique_divisible_by_19_l3381_338102

/-- Converts a base 7 number of the form 52x3 to decimal --/
def base7ToDecimal (x : ℕ) : ℕ :=
  5 * 7^3 + 2 * 7^2 + x * 7 + 3

/-- Checks if a natural number is a valid base 7 digit --/
def isBase7Digit (x : ℕ) : Prop :=
  x ≤ 6

theorem unique_divisible_by_19 :
  ∃! x : ℕ, isBase7Digit x ∧ (base7ToDecimal x) % 19 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_by_19_l3381_338102


namespace NUMINAMATH_CALUDE_distribution_plans_count_l3381_338174

-- Define the number of awards and schools
def total_awards : ℕ := 7
def num_schools : ℕ := 5
def min_awards_per_special_school : ℕ := 2
def num_special_schools : ℕ := 2

-- Define the function to calculate the number of distribution plans
def num_distribution_plans : ℕ :=
  Nat.choose (total_awards - min_awards_per_special_school * num_special_schools + num_schools - 1) (num_schools - 1)

-- Theorem statement
theorem distribution_plans_count :
  num_distribution_plans = 35 :=
sorry

end NUMINAMATH_CALUDE_distribution_plans_count_l3381_338174


namespace NUMINAMATH_CALUDE_solve_for_s_l3381_338163

theorem solve_for_s (s t : ℤ) (eq1 : 8 * s + 7 * t = 156) (eq2 : s = t - 3) : s = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_s_l3381_338163


namespace NUMINAMATH_CALUDE_average_and_difference_l3381_338168

theorem average_and_difference (y : ℝ) : 
  (35 + y) / 2 = 42 → |35 - y| = 14 := by sorry

end NUMINAMATH_CALUDE_average_and_difference_l3381_338168


namespace NUMINAMATH_CALUDE_expected_hits_value_l3381_338107

/-- The probability of hitting the target -/
def hit_probability : ℝ := 0.97

/-- The total number of shots -/
def total_shots : ℕ := 1000

/-- The expected number of hits -/
def expected_hits : ℝ := hit_probability * total_shots

theorem expected_hits_value : expected_hits = 970 := by
  sorry

end NUMINAMATH_CALUDE_expected_hits_value_l3381_338107


namespace NUMINAMATH_CALUDE_cubic_root_approximation_bound_l3381_338123

theorem cubic_root_approximation_bound :
  ∃ (c : ℝ), c > 0 ∧ ∀ (m n : ℤ), n ≥ 1 →
    |2^(1/3 : ℝ) - (m : ℝ) / (n : ℝ)| > c / (n : ℝ)^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_approximation_bound_l3381_338123


namespace NUMINAMATH_CALUDE_unit_digit_of_product_l3381_338198

-- Define the numbers
def a : ℕ := 7858
def b : ℕ := 1086
def c : ℕ := 4582
def d : ℕ := 9783

-- Define the product
def product : ℕ := a * b * c * d

-- Theorem statement
theorem unit_digit_of_product : product % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_product_l3381_338198


namespace NUMINAMATH_CALUDE_sin_600_degrees_l3381_338162

theorem sin_600_degrees : Real.sin (600 * π / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l3381_338162


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3381_338134

theorem ratio_x_to_y (x y : ℝ) (h : (12*x - 5*y) / (15*x - 3*y) = 4/7) : x/y = 23/24 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3381_338134


namespace NUMINAMATH_CALUDE_distinguishable_cube_colorings_eq_30240_l3381_338196

/-- The number of colors available to paint the cube. -/
def num_colors : ℕ := 10

/-- The number of faces on the cube. -/
def num_faces : ℕ := 6

/-- The number of rotational symmetries of a cube. -/
def cube_rotations : ℕ := 24

/-- Calculates the number of distinguishable ways to paint a cube. -/
def distinguishable_cube_colorings : ℕ :=
  (num_colors * (num_colors - 1) * (num_colors - 2) * (num_colors - 3) * 
   (num_colors - 4) * (num_colors - 5)) / cube_rotations

/-- Theorem stating that the number of distinguishable ways to paint the cube is 30240. -/
theorem distinguishable_cube_colorings_eq_30240 :
  distinguishable_cube_colorings = 30240 := by
  sorry

end NUMINAMATH_CALUDE_distinguishable_cube_colorings_eq_30240_l3381_338196


namespace NUMINAMATH_CALUDE_relationship_abc_l3381_338165

theorem relationship_abc : 
  let a : ℝ := (1/3 : ℝ)^(2/3 : ℝ)
  let b : ℝ := (1/3 : ℝ)^(1/3 : ℝ)
  let c : ℝ := (2/3 : ℝ)^(1/3 : ℝ)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3381_338165


namespace NUMINAMATH_CALUDE_system_solution_l3381_338144

theorem system_solution (x y : ℚ) 
  (eq1 : 5 * x - 3 * y = 27) 
  (eq2 : 3 * x + 5 * y = 1) : 
  x + y = 31 / 17 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3381_338144


namespace NUMINAMATH_CALUDE_special_function_sum_l3381_338156

/-- A function satisfying specific properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x^3) = (f x)^3) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ ≠ f x₂)

/-- Theorem stating the sum of f(0), f(1), and f(-1) for a special function -/
theorem special_function_sum (f : ℝ → ℝ) (h : SpecialFunction f) :
  f 0 + f 1 + f (-1) = 0 := by sorry

end NUMINAMATH_CALUDE_special_function_sum_l3381_338156


namespace NUMINAMATH_CALUDE_circle_C_theorem_l3381_338101

-- Define the circle C
def circle_C (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 5}

-- Define the lines
def line_l1 (x y : ℝ) : Prop := x - y + 1 = 0
def line_l2 (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 - Real.sqrt 3 = 0
def line_l3 (m a : ℝ) (x y : ℝ) : Prop := m * x - y + Real.sqrt a + 1 = 0

-- Define the theorem
theorem circle_C_theorem (center : ℝ × ℝ) (M N : ℝ × ℝ) :
  line_l1 center.1 center.2 →
  M ∈ circle_C center →
  N ∈ circle_C center →
  line_l2 M.1 M.2 →
  line_l2 N.1 N.2 →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 17 →
  (∀ (m : ℝ), ∃ (p : ℝ × ℝ), p ∈ circle_C center ∧ line_l3 m 5 p.1 p.2) →
  (((center.1 = 0 ∧ center.2 = 1) ∨
    (center.1 = 3 + Real.sqrt 3 ∧ center.2 = 4 + Real.sqrt 3)) ∧
   (∀ (a : ℝ), (∀ (m : ℝ), ∃ (p : ℝ × ℝ), p ∈ circle_C center ∧ line_l3 m a p.1 p.2) → 0 ≤ a ∧ a ≤ 5)) :=
by sorry


end NUMINAMATH_CALUDE_circle_C_theorem_l3381_338101


namespace NUMINAMATH_CALUDE_cubic_sum_divisible_by_nine_l3381_338124

theorem cubic_sum_divisible_by_nine (n : ℕ+) : 
  ∃ k : ℤ, (n : ℤ)^3 + (n + 1 : ℤ)^3 + (n + 2 : ℤ)^3 = 9 * k :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_divisible_by_nine_l3381_338124


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3381_338152

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l3381_338152


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3381_338129

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = -x^2 + 2*x + 2}
def B : Set ℝ := {y | ∃ x, y = 2^x - 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {y | -1 < y ∧ y ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3381_338129


namespace NUMINAMATH_CALUDE_long_division_puzzle_l3381_338153

theorem long_division_puzzle : ∃! (a b c d : ℕ), 
  (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧
  (c ≠ 0) ∧ (d ≠ 0) ∧
  (1000 * a + 100 * b + 10 * c + d) / (10 * c + d) = (100 * b + 10 * c + d) ∧
  (10 * c + d) * b = (10 * c + d) ∧
  (a = 3) ∧ (b = 1) ∧ (c = 2) ∧ (d = 5) := by
sorry

end NUMINAMATH_CALUDE_long_division_puzzle_l3381_338153


namespace NUMINAMATH_CALUDE_school_journey_time_l3381_338132

/-- Calculates the remaining time to reach the classroom given the total time available,
    time to reach the school gate, and time to reach the school building from the gate. -/
def remaining_time (total_time gate_time building_time : ℕ) : ℕ :=
  total_time - (gate_time + building_time)

/-- Proves that given 30 minutes total time, 15 minutes to reach the gate,
    and 6 minutes to reach the building, there are 9 minutes left to reach the room. -/
theorem school_journey_time : remaining_time 30 15 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_school_journey_time_l3381_338132


namespace NUMINAMATH_CALUDE_inequality_theorem_l3381_338194

theorem inequality_theorem (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_eq : a * b / (c * d) = (a + b) / (c + d)) :
  (a + b) * (c + d) ≥ (a + c) * (b + d) := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3381_338194


namespace NUMINAMATH_CALUDE_inequality_proof_l3381_338172

theorem inequality_proof (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
  a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3381_338172


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3381_338164

theorem arithmetic_equality : 2^2 * 7 + 5 * 12 + 7^2 * 2 + 6 * 3 = 212 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3381_338164


namespace NUMINAMATH_CALUDE_fractional_exponent_simplification_l3381_338158

theorem fractional_exponent_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a ^ (2 * b ^ (1/4))) / (a ^ (1/2) * b ^ (1/4)) = a ^ (3/2) := by
  sorry

end NUMINAMATH_CALUDE_fractional_exponent_simplification_l3381_338158


namespace NUMINAMATH_CALUDE_linear_equation_natural_solution_l3381_338159

theorem linear_equation_natural_solution (m : ℤ) : 
  (∃ x : ℕ, m * (x : ℤ) - 6 = x) ↔ m ∈ ({2, 3, 4, 7} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_linear_equation_natural_solution_l3381_338159


namespace NUMINAMATH_CALUDE_tan_three_expression_l3381_338184

theorem tan_three_expression (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ ^ 2) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 3 / Real.sqrt 10 - 3 / (Real.sqrt 10 + 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_three_expression_l3381_338184


namespace NUMINAMATH_CALUDE_total_length_of_stationery_l3381_338160

/-- Given the lengths of a rubber, pen, and pencil with specific relationships,
    prove that their total length is 29 centimeters. -/
theorem total_length_of_stationery (rubber pen pencil : ℝ) : 
  pen = rubber + 3 →
  pencil = pen + 2 →
  pencil = 12 →
  rubber + pen + pencil = 29 := by
sorry

end NUMINAMATH_CALUDE_total_length_of_stationery_l3381_338160


namespace NUMINAMATH_CALUDE_round_trip_time_l3381_338199

/-- Calculates the total time for a round trip by boat given the boat's speed in standing water,
    the stream's speed, and the distance to the destination. -/
theorem round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 14) 
  (h2 : stream_speed = 1.2) 
  (h3 : distance = 4864) : 
  (distance / (boat_speed + stream_speed)) + (distance / (boat_speed - stream_speed)) = 700 := by
  sorry

#check round_trip_time

end NUMINAMATH_CALUDE_round_trip_time_l3381_338199


namespace NUMINAMATH_CALUDE_cos_210_degrees_l3381_338141

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l3381_338141


namespace NUMINAMATH_CALUDE_sandy_fish_count_l3381_338131

theorem sandy_fish_count (initial_fish : ℕ) (bought_fish : ℕ) : 
  initial_fish = 26 → bought_fish = 6 → initial_fish + bought_fish = 32 := by
  sorry

end NUMINAMATH_CALUDE_sandy_fish_count_l3381_338131


namespace NUMINAMATH_CALUDE_total_oranges_l3381_338154

theorem total_oranges (children : ℕ) (oranges_per_child : ℕ) 
  (h1 : children = 4) 
  (h2 : oranges_per_child = 3) : 
  children * oranges_per_child = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_l3381_338154


namespace NUMINAMATH_CALUDE_sum_is_six_digit_multiple_of_four_l3381_338125

def sum_of_numbers (A B : Nat) : Nat :=
  98765 + A * 1000 + 532 + B * 100 + 41 + 1021

theorem sum_is_six_digit_multiple_of_four (A B : Nat) 
  (h1 : 1 ≤ A ∧ A ≤ 9) (h2 : 1 ≤ B ∧ B ≤ 9) : 
  ∃ (n : Nat), sum_of_numbers A B = n ∧ 
  100000 ≤ n ∧ n < 1000000 ∧ 
  n % 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_is_six_digit_multiple_of_four_l3381_338125


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l3381_338138

theorem min_sum_dimensions (a b c : ℕ+) : 
  a * b * c = 3003 → a + b + c ≥ 45 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l3381_338138


namespace NUMINAMATH_CALUDE_milk_water_solution_volume_l3381_338173

theorem milk_water_solution_volume 
  (initial_milk_percentage : ℝ) 
  (final_milk_percentage : ℝ) 
  (added_water : ℝ) 
  (initial_milk_percentage_value : initial_milk_percentage = 0.84)
  (final_milk_percentage_value : final_milk_percentage = 0.58)
  (added_water_value : added_water = 26.9) : 
  ∃ (initial_volume : ℝ), 
    initial_volume > 0 ∧ 
    initial_milk_percentage * initial_volume / (initial_volume + added_water) = final_milk_percentage ∧
    initial_volume = 60 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_solution_volume_l3381_338173


namespace NUMINAMATH_CALUDE_complement_of_beta_l3381_338128

-- Define angles α and β
variable (α β : Real)

-- Define the conditions
def complementary : Prop := α + β = 180
def alpha_greater : Prop := α > β

-- Define the complement of an angle
def complement (θ : Real) : Real := 90 - θ

-- State the theorem
theorem complement_of_beta (h1 : complementary α β) (h2 : alpha_greater α β) :
  complement β = (α - β) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_beta_l3381_338128


namespace NUMINAMATH_CALUDE_basketball_score_ratio_l3381_338116

/-- Given the scores of three basketball players, prove that the ratio of Tim's points to Ken's points is 1:2 -/
theorem basketball_score_ratio 
  (joe tim ken : ℕ)  -- Scores of Joe, Tim, and Ken
  (h1 : tim = joe + 20)  -- Tim scored 20 points more than Joe
  (h2 : joe + tim + ken = 100)  -- Total points scored is 100
  (h3 : tim = 30)  -- Tim scored 30 points
  : tim * 2 = ken :=
by sorry

end NUMINAMATH_CALUDE_basketball_score_ratio_l3381_338116


namespace NUMINAMATH_CALUDE_largest_common_term_largest_common_term_exists_l3381_338175

theorem largest_common_term (n : ℕ) : n ≤ 200 ∧ 
  (∃ k : ℕ, n = 8 * k + 2) ∧ 
  (∃ m : ℕ, n = 9 * m + 5) →
  n ≤ 194 := by
  sorry

theorem largest_common_term_exists : 
  ∃ n : ℕ, n = 194 ∧ n ≤ 200 ∧ 
  (∃ k : ℕ, n = 8 * k + 2) ∧ 
  (∃ m : ℕ, n = 9 * m + 5) := by
  sorry

end NUMINAMATH_CALUDE_largest_common_term_largest_common_term_exists_l3381_338175


namespace NUMINAMATH_CALUDE_sum_of_products_formula_l3381_338118

/-- The sum of products resulting from repeatedly dividing n balls into two groups -/
def sumOfProducts (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the sum of products for n balls is n * (n-1) / 2 -/
theorem sum_of_products_formula (n : ℕ) : 
  sumOfProducts n = n * (n - 1) / 2 := by
  sorry

#check sum_of_products_formula

end NUMINAMATH_CALUDE_sum_of_products_formula_l3381_338118


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l3381_338109

def arithmetic_progression (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_progression_problem (a₁ d : ℝ) :
  (arithmetic_progression a₁ d 13 = 3 * arithmetic_progression a₁ d 3) ∧
  (arithmetic_progression a₁ d 18 = 2 * arithmetic_progression a₁ d 7 + 8) →
  d = 4 ∧ a₁ = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_problem_l3381_338109


namespace NUMINAMATH_CALUDE_no_bribed_judges_probability_l3381_338143

def total_judges : ℕ := 14
def valid_scores : ℕ := 7
def bribed_judges : ℕ := 2

def probability_no_bribed_judges : ℚ := 3/13

theorem no_bribed_judges_probability :
  (Nat.choose (total_judges - bribed_judges) valid_scores * Nat.choose bribed_judges 0) /
  Nat.choose total_judges valid_scores = probability_no_bribed_judges := by
  sorry

end NUMINAMATH_CALUDE_no_bribed_judges_probability_l3381_338143


namespace NUMINAMATH_CALUDE_equation_represents_point_l3381_338161

theorem equation_represents_point :
  ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 6*y + 7 = 0 ↔ x = 2 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_represents_point_l3381_338161


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_sum_zero_l3381_338176

theorem sqrt_abs_sum_zero_implies_power_sum_zero (a b : ℝ) :
  Real.sqrt (a + 1) + |b - 1| = 0 → a^2023 + b^2024 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_sum_zero_l3381_338176


namespace NUMINAMATH_CALUDE_rational_fraction_implies_integer_sum_squares_over_sum_l3381_338127

theorem rational_fraction_implies_integer_sum_squares_over_sum (a b c : ℕ+) :
  (∃ (r s : ℤ), (r : ℚ) / s = (a * Real.sqrt 3 + b) / (b * Real.sqrt 3 + c)) →
  ∃ (k : ℤ), (a ^ 2 + b ^ 2 + c ^ 2 : ℚ) / (a + b + c) = k := by
sorry

end NUMINAMATH_CALUDE_rational_fraction_implies_integer_sum_squares_over_sum_l3381_338127


namespace NUMINAMATH_CALUDE_tenth_toss_probability_l3381_338106

/-- A fair coin is a coin with equal probability of landing heads or tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of getting heads on a single toss of a fair coin -/
def prob_heads (p : ℝ) : Prop := p = 1/2

/-- The number of times the coin has been tossed -/
def num_tosses : ℕ := 9

/-- The number of heads obtained in the previous tosses -/
def num_heads : ℕ := 7

/-- The number of tails obtained in the previous tosses -/
def num_tails : ℕ := 2

theorem tenth_toss_probability (p : ℝ) 
  (h_fair : fair_coin p) 
  (h_prev_tosses : num_tosses = num_heads + num_tails) :
  prob_heads p := by sorry

end NUMINAMATH_CALUDE_tenth_toss_probability_l3381_338106


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l3381_338185

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 357000) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 75 / 100 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l3381_338185


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l3381_338122

theorem trig_expression_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l3381_338122


namespace NUMINAMATH_CALUDE_number_of_partitions_l3381_338166

-- Define the set A
def A : Set Nat := {1, 2}

-- Define what a partition is
def is_partition (A₁ A₂ : Set Nat) : Prop :=
  A₁ ∪ A₂ = A

-- Define when two partitions are considered the same
def same_partition (A₁ A₂ : Set Nat) : Prop :=
  A₁ = A₂

-- Define a function to count the number of different partitions
def count_partitions : Nat :=
  sorry

-- The theorem to prove
theorem number_of_partitions :
  count_partitions = 9 :=
sorry

end NUMINAMATH_CALUDE_number_of_partitions_l3381_338166


namespace NUMINAMATH_CALUDE_b5b9_value_l3381_338136

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b5b9_value (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_nonzero : ∀ n : ℕ, a n ≠ 0)
  (h_eq : 2 * a 2 + 2 * a 12 = (a 7)^2)
  (h_b7 : b 7 = a 7) :
  b 5 * b 9 = 16 := by
sorry

end NUMINAMATH_CALUDE_b5b9_value_l3381_338136


namespace NUMINAMATH_CALUDE_mortgage_payment_sum_l3381_338135

theorem mortgage_payment_sum (a₁ : ℝ) (r : ℝ) (n : ℕ) (h₁ : a₁ = 100) (h₂ : r = 3) (h₃ : n = 9) :
  a₁ * (1 - r^n) / (1 - r) = 984100 := by
  sorry

end NUMINAMATH_CALUDE_mortgage_payment_sum_l3381_338135


namespace NUMINAMATH_CALUDE_no_odd_sided_cross_section_polyhedron_l3381_338149

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  convex : Bool

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields here

/-- A polygon -/
structure Polygon where
  sides : ℕ

/-- Represents a cross-section of a polyhedron with a plane -/
def cross_section (p : ConvexPolyhedron) (plane : Plane) : Polygon :=
  sorry

/-- Predicate to check if a plane passes through a vertex of the polyhedron -/
def passes_through_vertex (p : ConvexPolyhedron) (plane : Plane) : Prop :=
  sorry

/-- Main theorem: No such convex polyhedron exists -/
theorem no_odd_sided_cross_section_polyhedron :
  ¬ ∃ (p : ConvexPolyhedron),
    (∀ (plane : Plane),
      ¬passes_through_vertex p plane →
      (cross_section p plane).sides % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_no_odd_sided_cross_section_polyhedron_l3381_338149


namespace NUMINAMATH_CALUDE_prob_tails_heads_heads_l3381_338145

-- Define a coin flip as a type with two possible outcomes
inductive CoinFlip : Type
| Heads : CoinFlip
| Tails : CoinFlip

-- Define a sequence of three coin flips
def ThreeFlips := (CoinFlip × CoinFlip × CoinFlip)

-- Define the probability of getting tails on a single flip
def prob_tails : ℚ := 1 / 2

-- Define the desired outcome: Tails, Heads, Heads
def desired_outcome : ThreeFlips := (CoinFlip.Tails, CoinFlip.Heads, CoinFlip.Heads)

-- Theorem: The probability of getting the desired outcome is 1/8
theorem prob_tails_heads_heads : 
  (prob_tails * (1 - prob_tails) * (1 - prob_tails) : ℚ) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_tails_heads_heads_l3381_338145


namespace NUMINAMATH_CALUDE_small_cuboid_length_l3381_338117

/-- Represents the dimensions of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

/-- Theorem: Given a large cuboid of 16m x 10m x 12m and small cuboids of Lm x 4m x 3m,
    if 32 small cuboids can be formed from the large cuboid, then L = 5m -/
theorem small_cuboid_length
  (large : Cuboid)
  (small : Cuboid)
  (h1 : large.length = 16)
  (h2 : large.width = 10)
  (h3 : large.height = 12)
  (h4 : small.width = 4)
  (h5 : small.height = 3)
  (h6 : volume large = 32 * volume small) :
  small.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_small_cuboid_length_l3381_338117


namespace NUMINAMATH_CALUDE_range_of_x_l3381_338190

theorem range_of_x (x : ℝ) : 
  0 ≤ x → x < 2 * Real.pi → Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x →
  π / 4 ≤ x ∧ x ≤ 5 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l3381_338190


namespace NUMINAMATH_CALUDE_maia_daily_requests_l3381_338142

/-- The number of client requests Maia works on each day -/
def requests_per_day : ℕ := 4

/-- The number of days Maia works -/
def days_worked : ℕ := 5

/-- The number of client requests remaining after the working period -/
def remaining_requests : ℕ := 10

/-- The number of client requests Maia gets every day -/
def daily_requests : ℕ := 6

theorem maia_daily_requests : 
  days_worked * daily_requests = days_worked * requests_per_day + remaining_requests :=
by sorry

end NUMINAMATH_CALUDE_maia_daily_requests_l3381_338142


namespace NUMINAMATH_CALUDE_larry_stickers_l3381_338171

theorem larry_stickers (initial_stickers lost_stickers : ℕ) 
  (h1 : initial_stickers = 93)
  (h2 : lost_stickers = 6) :
  initial_stickers - lost_stickers = 87 := by
  sorry

end NUMINAMATH_CALUDE_larry_stickers_l3381_338171


namespace NUMINAMATH_CALUDE_proposition_p_equivalence_l3381_338146

theorem proposition_p_equivalence :
  (∃ x, x < 1 ∧ x^2 < 1) ↔ ¬(∀ x, x < 1 → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_equivalence_l3381_338146


namespace NUMINAMATH_CALUDE_possible_values_of_y_l3381_338111

theorem possible_values_of_y (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let y := a / |a| + b / |b| + (a * b) / |a * b|
  ∃ (S : Set ℝ), S = {3, -1} ∧ y ∈ S :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_y_l3381_338111


namespace NUMINAMATH_CALUDE_tomato_plants_per_row_is_eight_l3381_338195

/-- Represents the garden planting scenario -/
structure GardenPlanting where
  cucumber_to_tomato_ratio : ℚ
  total_rows : ℕ
  tomatoes_per_plant : ℕ
  total_tomatoes : ℕ

/-- Calculates the number of tomato plants per row -/
def tomato_plants_per_row (g : GardenPlanting) : ℚ :=
  g.total_tomatoes / (g.tomatoes_per_plant * (g.total_rows / (1 + g.cucumber_to_tomato_ratio)))

/-- Theorem stating that the number of tomato plants per row is 8 -/
theorem tomato_plants_per_row_is_eight (g : GardenPlanting) 
  (h1 : g.cucumber_to_tomato_ratio = 2)
  (h2 : g.total_rows = 15)
  (h3 : g.tomatoes_per_plant = 3)
  (h4 : g.total_tomatoes = 120) : 
  tomato_plants_per_row g = 8 := by
  sorry

end NUMINAMATH_CALUDE_tomato_plants_per_row_is_eight_l3381_338195


namespace NUMINAMATH_CALUDE_alison_large_tubs_l3381_338177

/-- The number of large tubs Alison bought -/
def num_large_tubs : ℕ := 3

/-- The number of small tubs Alison bought -/
def num_small_tubs : ℕ := 6

/-- The cost of each large tub in dollars -/
def cost_large_tub : ℕ := 6

/-- The cost of each small tub in dollars -/
def cost_small_tub : ℕ := 5

/-- The total cost of all tubs in dollars -/
def total_cost : ℕ := 48

theorem alison_large_tubs : 
  num_large_tubs * cost_large_tub + num_small_tubs * cost_small_tub = total_cost := by
  sorry

end NUMINAMATH_CALUDE_alison_large_tubs_l3381_338177


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3381_338105

def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 - 2}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1}

theorem intersection_of_A_and_B : A ∩ B = {(1, 1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3381_338105


namespace NUMINAMATH_CALUDE_bill_oranges_count_l3381_338112

/-- The number of oranges Betty picked -/
def betty_oranges : ℕ := 15

/-- The number of oranges Bill picked -/
def bill_oranges : ℕ := sorry

/-- The number of oranges Frank picked -/
def frank_oranges : ℕ := 3 * (betty_oranges + bill_oranges)

/-- The number of seeds Frank planted -/
def seeds_planted : ℕ := 2 * frank_oranges

/-- The number of oranges on each tree -/
def oranges_per_tree : ℕ := 5

/-- The total number of oranges Philip can pick -/
def philip_oranges : ℕ := 810

theorem bill_oranges_count : bill_oranges = 12 := by
  sorry

end NUMINAMATH_CALUDE_bill_oranges_count_l3381_338112


namespace NUMINAMATH_CALUDE_probability_is_24_1107_l3381_338140

/-- Represents a 5x5x5 cube with one face painted red and an internal diagonal painted green -/
structure PaintedCube where
  size : Nat
  size_eq : size = 5

/-- The number of unit cubes with exactly three painted faces -/
def three_painted_faces (cube : PaintedCube) : Nat := 8

/-- The number of unit cubes with exactly one painted face -/
def one_painted_face (cube : PaintedCube) : Nat := 21

/-- The total number of unit cubes in the larger cube -/
def total_cubes (cube : PaintedCube) : Nat := cube.size ^ 3

/-- The probability of selecting one cube with exactly three painted faces
    and one cube with exactly one painted face when choosing two cubes uniformly at random -/
def probability (cube : PaintedCube) : Rat :=
  (three_painted_faces cube * one_painted_face cube : Rat) / (total_cubes cube).choose 2

/-- The main theorem stating the probability is 24/1107 -/
theorem probability_is_24_1107 (cube : PaintedCube) : probability cube = 24 / 1107 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_24_1107_l3381_338140


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3381_338150

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- Given that point A(a,1) and point B(5,b) are symmetric with respect to the origin O, prove that a + b = -6 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin a 1 5 b) : a + b = -6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3381_338150
