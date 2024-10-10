import Mathlib

namespace smallest_positive_integer_with_remainders_l3201_320128

theorem smallest_positive_integer_with_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧ 
  ∀ (y : ℕ), y > 0 → 
    (y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7) → 
    x ≤ y ∧
  x = 167 := by
sorry

end smallest_positive_integer_with_remainders_l3201_320128


namespace ap_length_l3201_320138

/-- Square with inscribed circle -/
structure SquareWithCircle where
  /-- Side length of the square -/
  side_length : ℝ
  /-- The square ABCD -/
  square : Set (ℝ × ℝ)
  /-- The inscribed circle ω -/
  circle : Set (ℝ × ℝ)
  /-- Point A of the square -/
  A : ℝ × ℝ
  /-- Point M where the circle intersects CD -/
  M : ℝ × ℝ
  /-- Point P where AM intersects the circle (different from M) -/
  P : ℝ × ℝ
  /-- The side length is 2 -/
  h_side_length : side_length = 2
  /-- A is a vertex of the square -/
  h_A_in_square : A ∈ square
  /-- M is on the circle and on the side CD -/
  h_M_on_circle_and_CD : M ∈ circle ∧ M.2 = -1
  /-- P is on the circle and on line AM -/
  h_P_on_circle_and_AM : P ∈ circle ∧ P ≠ M ∧ ∃ t : ℝ, P = (1 - t) • A + t • M

/-- The length of AP in a square with inscribed circle is √5/5 -/
theorem ap_length (swc : SquareWithCircle) : Real.sqrt 5 / 5 = ‖swc.A - swc.P‖ := by
  sorry

end ap_length_l3201_320138


namespace divisible_by_45_sum_of_digits_l3201_320101

theorem divisible_by_45_sum_of_digits (a b : ℕ) : 
  a < 10 → b < 10 → (60000 + 1000 * a + 780 + b) % 45 = 0 → a + b = 6 := by
  sorry

end divisible_by_45_sum_of_digits_l3201_320101


namespace square_divisibility_l3201_320147

theorem square_divisibility (a b : ℕ+) (h : (a * b + 1) ∣ (a ^ 2 + b ^ 2)) :
  ∃ k : ℕ, (a ^ 2 + b ^ 2) / (a * b + 1) = k ^ 2 := by
  sorry

end square_divisibility_l3201_320147


namespace alice_savings_this_month_l3201_320115

/-- Alice's sales and earnings calculation --/
def alice_savings (sales : ℝ) (basic_salary : ℝ) (commission_rate : ℝ) (savings_rate : ℝ) : ℝ :=
  let commission := sales * commission_rate
  let total_earnings := basic_salary + commission
  total_earnings * savings_rate

/-- Theorem: Alice's savings this month will be $29 --/
theorem alice_savings_this_month :
  alice_savings 2500 240 0.02 0.10 = 29 := by
  sorry

end alice_savings_this_month_l3201_320115


namespace paving_rate_per_square_metre_l3201_320184

/-- Proves that the rate per square metre for paving a room is Rs. 950 given the specified conditions. -/
theorem paving_rate_per_square_metre
  (length : ℝ)
  (width : ℝ)
  (total_cost : ℝ)
  (h1 : length = 5.5)
  (h2 : width = 4)
  (h3 : total_cost = 20900) :
  total_cost / (length * width) = 950 := by
  sorry

#check paving_rate_per_square_metre

end paving_rate_per_square_metre_l3201_320184


namespace number_square_problem_l3201_320130

theorem number_square_problem : ∃! x : ℝ, x^2 + 64 = (x - 16)^2 ∧ x = 6 := by sorry

end number_square_problem_l3201_320130


namespace negative_fractions_comparison_l3201_320151

theorem negative_fractions_comparison : -1/2 < -1/3 := by
  sorry

end negative_fractions_comparison_l3201_320151


namespace light_could_be_green_l3201_320178

/-- Represents the state of a traffic light -/
inductive TrafficLightState
| Red
| Green
| Yellow

/-- Represents a traffic light with its cycle durations -/
structure TrafficLight where
  total_cycle : ℕ
  red_duration : ℕ
  green_duration : ℕ
  yellow_duration : ℕ
  cycle_valid : total_cycle = red_duration + green_duration + yellow_duration

/-- Defines the specific traffic light from the problem -/
def intersection_light : TrafficLight :=
  { total_cycle := 60
  , red_duration := 30
  , green_duration := 25
  , yellow_duration := 5
  , cycle_valid := by rfl }

/-- Theorem stating that the traffic light could be green at any random observation -/
theorem light_could_be_green (t : ℕ) : 
  ∃ (s : TrafficLightState), s = TrafficLightState.Green :=
sorry

end light_could_be_green_l3201_320178


namespace g_of_five_l3201_320161

/-- Given a function g : ℝ → ℝ satisfying 3g(x) + 4g(1 - x) = 6x^2 for all real x, prove that g(5) = -66/7 -/
theorem g_of_five (g : ℝ → ℝ) (h : ∀ x : ℝ, 3 * g x + 4 * g (1 - x) = 6 * x^2) : g 5 = -66/7 := by
  sorry

end g_of_five_l3201_320161


namespace sum_of_coefficients_l3201_320160

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ + a₀ = 128 :=
by sorry

end sum_of_coefficients_l3201_320160


namespace committee_selection_l3201_320171

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem committee_selection (total : ℕ) (committee_size : ℕ) (bill_karl : ℕ) (alice_jane : ℕ) :
  total = 9 ∧ committee_size = 5 ∧ bill_karl = 2 ∧ alice_jane = 2 →
  (choose (total - bill_karl) (committee_size - bill_karl) - 
   choose (total - bill_karl - alice_jane) 1) +
  (choose (total - bill_karl) committee_size - 
   choose (total - bill_karl - alice_jane) 3) = 41 :=
by sorry

end committee_selection_l3201_320171


namespace divisor_problem_l3201_320170

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 15698 →
  quotient = 89 →
  remainder = 14 →
  dividend = divisor * quotient + remainder →
  divisor = 176 := by
sorry

end divisor_problem_l3201_320170


namespace intersection_condition_l3201_320116

/-- Set M in R^2 -/
def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

/-- Set N in R^2 parameterized by r -/
def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

/-- The theorem stating the condition for M ∩ N = N -/
theorem intersection_condition (r : ℝ) : 
  (M ∩ N r = N r) ↔ (0 < r ∧ r ≤ 2 - Real.sqrt 2) := by
  sorry

end intersection_condition_l3201_320116


namespace equation_solution_l3201_320154

theorem equation_solution : ∃ (x : ℝ), x > 0 ∧ 5 * Real.sqrt (1 + x) + 5 * Real.sqrt (1 - x) = 7 * Real.sqrt 2 ∧ x = 7/25 := by
  sorry

end equation_solution_l3201_320154


namespace geometric_sequence_common_ratio_l3201_320189

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a3 : a 3 = 3/2) 
  (h_S3 : (a 1) + (a 2) + (a 3) = 9/2) :
  a 1 / a 0 = 1 := by
sorry

end geometric_sequence_common_ratio_l3201_320189


namespace right_triangle_PQR_area_l3201_320153

/-- A right triangle PQR in the xy-plane with specific properties -/
structure RightTrianglePQR where
  /-- Point P of the triangle -/
  P : ℝ × ℝ
  /-- Point Q of the triangle -/
  Q : ℝ × ℝ
  /-- Point R of the triangle (right angle) -/
  R : ℝ × ℝ
  /-- The triangle has a right angle at R -/
  right_angle_at_R : (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0
  /-- The length of hypotenuse PQ is 50 -/
  hypotenuse_length : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2
  /-- The median through P lies on the line y = x + 5 -/
  median_P : ∃ t : ℝ, (P.1 + Q.1 + R.1) / 3 = t ∧ (P.2 + Q.2 + R.2) / 3 = t + 5
  /-- The median through Q lies on the line y = 3x + 6 -/
  median_Q : ∃ t : ℝ, (P.1 + Q.1 + R.1) / 3 = t ∧ (P.2 + Q.2 + R.2) / 3 = 3 * t + 6

/-- The area of the right triangle PQR is 104.1667 -/
theorem right_triangle_PQR_area (t : RightTrianglePQR) : 
  abs ((t.P.1 - t.R.1) * (t.Q.2 - t.R.2) - (t.Q.1 - t.R.1) * (t.P.2 - t.R.2)) / 2 = 104.1667 := by
  sorry

end right_triangle_PQR_area_l3201_320153


namespace total_animals_l3201_320123

theorem total_animals (giraffes pigs dogs : ℕ) 
  (h1 : giraffes = 6) 
  (h2 : pigs = 8) 
  (h3 : dogs = 4) : 
  giraffes + pigs + dogs = 18 := by
  sorry

end total_animals_l3201_320123


namespace instantaneous_speed_at_t_1_l3201_320167

/-- The displacement function for the particle's motion --/
def s (t : ℝ) : ℝ := 2 * t^3

/-- The velocity function (derivative of displacement) --/
def v (t : ℝ) : ℝ := 6 * t^2

theorem instantaneous_speed_at_t_1 :
  v 1 = 6 := by sorry

end instantaneous_speed_at_t_1_l3201_320167


namespace runner_distance_l3201_320195

/-- Represents the runner's problem --/
def RunnerProblem (speed time distance : ℝ) : Prop :=
  -- Normal condition
  speed * time = distance ∧
  -- Increased speed condition
  (speed + 1) * (2/3 * time) = distance ∧
  -- Decreased speed condition
  (speed - 1) * (time + 3) = distance

/-- Theorem stating the solution to the runner's problem --/
theorem runner_distance : ∃ (speed time : ℝ), RunnerProblem speed time 6 := by
  sorry


end runner_distance_l3201_320195


namespace volume_depends_on_length_l3201_320152

/-- Represents a rectangular prism with variable length -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ
  length_positive : length > 2
  width_is_two : width = 2
  height_is_one : height = 1
  volume_formula : volume = length * width * height

/-- The volume of a rectangular prism is dependent on its length -/
theorem volume_depends_on_length (prism : RectangularPrism) :
  ∃ f : ℝ → ℝ, prism.volume = f prism.length :=
by sorry

end volume_depends_on_length_l3201_320152


namespace prob_specific_quarter_is_one_eighth_l3201_320120

/-- Represents a piece of paper with two sides, each divided into four quarters -/
structure Paper :=
  (sides : Fin 2)
  (quarters : Fin 4)

/-- The total number of distinct parts (quarters) on the paper -/
def total_parts : ℕ := 8

/-- The probability of a specific quarter being on top after random folding -/
def prob_specific_quarter_on_top : ℚ := 1 / 8

/-- Theorem stating that the probability of a specific quarter being on top is 1/8 -/
theorem prob_specific_quarter_is_one_eighth :
  prob_specific_quarter_on_top = 1 / 8 := by
  sorry

end prob_specific_quarter_is_one_eighth_l3201_320120


namespace line_segment_ratio_l3201_320124

/-- Given points E, F, G, and H on a line in that order, prove that EG:FH = 10:17 -/
theorem line_segment_ratio (E F G H : ℝ) : 
  (F - E = 3) → (G - F = 7) → (H - E = 20) → (G - E) / (H - F) = 10 / 17 := by
sorry

end line_segment_ratio_l3201_320124


namespace min_distance_curve_to_line_l3201_320146

/-- The minimum distance from any point on the curve xy = √3 to the line x + √3y = 0 is √3 -/
theorem min_distance_curve_to_line :
  let C := {P : ℝ × ℝ | P.1 * P.2 = Real.sqrt 3}
  let l := {P : ℝ × ℝ | P.1 + Real.sqrt 3 * P.2 = 0}
  ∃ d : ℝ, d = Real.sqrt 3 ∧
    ∀ P ∈ C, ∀ Q ∈ l, d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end min_distance_curve_to_line_l3201_320146


namespace cos_angle_with_z_axis_l3201_320168

/-- Given a point Q in the first octant of 3D space, prove that if the cosine of the angle between OQ
    and the x-axis is 2/5, and the cosine of the angle between OQ and the y-axis is 1/4, then the
    cosine of the angle between OQ and the z-axis is √(311) / 20. -/
theorem cos_angle_with_z_axis (Q : ℝ × ℝ × ℝ) 
    (h_pos : Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0)
    (h_cos_alpha : Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = 2/5)
    (h_cos_beta : Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = 1/4) :
  Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = Real.sqrt 311 / 20 := by
  sorry

end cos_angle_with_z_axis_l3201_320168


namespace calculator_squaring_l3201_320100

theorem calculator_squaring (n : ℕ) : (1 : ℝ) ^ (2^n) ≤ 100 := by
  sorry

end calculator_squaring_l3201_320100


namespace max_area_rectangle_l3201_320159

/-- Given a rectangle with perimeter 160 feet and integer side lengths, 
    the maximum possible area is 1600 square feet. -/
theorem max_area_rectangle (l w : ℕ) : 
  2 * (l + w) = 160 → l * w ≤ 1600 := by
  sorry

end max_area_rectangle_l3201_320159


namespace alice_expected_games_l3201_320181

/-- Represents a tournament with n competitors -/
structure Tournament (n : ℕ) where
  skillLevels : Fin n → ℕ
  distinctSkills : ∀ i j, i ≠ j → skillLevels i ≠ skillLevels j

/-- The expected number of games played by a competitor with a given skill level -/
noncomputable def expectedGames (t : Tournament 21) (skillLevel : ℕ) : ℚ :=
  sorry

/-- Theorem stating the expected number of games for Alice -/
theorem alice_expected_games (t : Tournament 21) (h : t.skillLevels 10 = 11) :
  expectedGames t 11 = 47 / 42 :=
sorry

end alice_expected_games_l3201_320181


namespace matrix_value_proof_l3201_320164

def matrix_operation (a b c d : ℤ) : ℤ := a * c - b * d

theorem matrix_value_proof : matrix_operation 2 3 4 5 = -7 := by
  sorry

end matrix_value_proof_l3201_320164


namespace tennis_ball_ratio_l3201_320133

theorem tennis_ball_ratio : 
  let total_ordered : ℕ := 64
  let extra_yellow : ℕ := 20
  let white_balls : ℕ := total_ordered / 2
  let yellow_balls : ℕ := total_ordered / 2 + extra_yellow
  let gcd : ℕ := Nat.gcd white_balls yellow_balls
  (white_balls / gcd : ℕ) = 8 ∧ (yellow_balls / gcd : ℕ) = 13 := by
sorry

end tennis_ball_ratio_l3201_320133


namespace parking_lot_capacity_l3201_320158

/-- Calculates the number of vehicles that can still park in a lot -/
def remainingParkingSpaces (totalSpaces : ℕ) (caravanSpaces : ℕ) (caravansParked : ℕ) : ℕ :=
  totalSpaces - (caravanSpaces * caravansParked)

/-- Theorem: Given the conditions, 24 vehicles can still park -/
theorem parking_lot_capacity : remainingParkingSpaces 30 2 3 = 24 := by
  sorry

end parking_lot_capacity_l3201_320158


namespace cosine_equality_l3201_320185

theorem cosine_equality (n : ℤ) : 
  100 ≤ n ∧ n ≤ 280 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 317 :=
by sorry

end cosine_equality_l3201_320185


namespace binomial_distribution_problem_l3201_320142

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

variable (ξ : BinomialDistribution)

/-- The expected value of a binomial distribution -/
def expectation (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: For a binomial distribution with E[ξ] = 300 and D[ξ] = 200, p = 1/3 -/
theorem binomial_distribution_problem (ξ : BinomialDistribution) 
  (h2 : expectation ξ = 300) (h3 : variance ξ = 200) : ξ.p = 1/3 := by
  sorry


end binomial_distribution_problem_l3201_320142


namespace quadratic_to_linear_solutions_l3201_320183

theorem quadratic_to_linear_solutions (x : ℝ) :
  x^2 - 2*x - 1 = 0 ∧ (x - 1 = Real.sqrt 2 ∨ x - 1 = -Real.sqrt 2) →
  (x - 1 = Real.sqrt 2 → x - 1 = -Real.sqrt 2) ∧
  (x - 1 = -Real.sqrt 2 → x - 1 = Real.sqrt 2) :=
sorry

end quadratic_to_linear_solutions_l3201_320183


namespace log4_of_16_equals_2_l3201_320105

-- Define the logarithm function for base 4
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

-- State the theorem
theorem log4_of_16_equals_2 : log4 16 = 2 := by
  sorry

end log4_of_16_equals_2_l3201_320105


namespace gcd_of_168_56_224_l3201_320106

theorem gcd_of_168_56_224 : Nat.gcd 168 (Nat.gcd 56 224) = 56 := by
  sorry

end gcd_of_168_56_224_l3201_320106


namespace not_prime_sum_of_squares_l3201_320119

/-- The equation has exactly two positive integer roots -/
def has_two_positive_integer_roots (a b : ℝ) : Prop :=
  ∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x ≠ y ∧
    (∀ z : ℤ, z > 0 → a * z * (z^2 + a * z + 1) = b * (z^2 + b + 1) ↔ z = x ∨ z = y)

/-- Main theorem -/
theorem not_prime_sum_of_squares (a b : ℝ) :
  ab < 0 →
  has_two_positive_integer_roots a b →
  ¬ Nat.Prime (Int.natAbs (Int.floor (a^2 + b^2))) :=
sorry

end not_prime_sum_of_squares_l3201_320119


namespace prob_ace_then_king_standard_deck_l3201_320165

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_aces : ℕ)
  (num_kings : ℕ)

/-- Calculates the probability of drawing an Ace first and a King second from a standard deck -/
def prob_ace_then_king (d : Deck) : ℚ :=
  (d.num_aces : ℚ) / d.total_cards * (d.num_kings : ℚ) / (d.total_cards - 1)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_aces := 4,
    num_kings := 4 }

theorem prob_ace_then_king_standard_deck :
  prob_ace_then_king standard_deck = 4 / 663 := by
  sorry

end prob_ace_then_king_standard_deck_l3201_320165


namespace bakery_items_l3201_320140

theorem bakery_items (total : ℕ) (bread_rolls : ℕ) (croissants : ℕ) (bagels : ℕ)
  (h1 : total = 90)
  (h2 : bread_rolls = 49)
  (h3 : croissants = 19)
  (h4 : total = bread_rolls + croissants + bagels) :
  bagels = 22 := by
sorry

end bakery_items_l3201_320140


namespace inequalities_proof_l3201_320103

theorem inequalities_proof (a b c d : ℝ) 
  (h1 : b > a) (h2 : a > 1) (h3 : c < d) (h4 : d < -1) : 
  (1/b < 1/a ∧ 1/a < 1) ∧ 
  (1/c > 1/d ∧ 1/d > -1) ∧ 
  (a*d > b*c) := by
sorry


end inequalities_proof_l3201_320103


namespace foci_coordinates_l3201_320125

/-- Given that m is the geometric mean of 2 and 8, prove that the foci of x^2 + y^2/m = 1 are at (0, ±√3) -/
theorem foci_coordinates (m : ℝ) (hm_pos : m > 0) (hm_mean : m^2 = 2 * 8) :
  let equation := fun (x y : ℝ) ↦ x^2 + y^2 / m = 1
  ∃ c : ℝ, c^2 = 3 ∧ 
    (∀ x y : ℝ, equation x y ↔ equation x (-y)) ∧
    equation 0 c ∧ equation 0 (-c) :=
sorry

end foci_coordinates_l3201_320125


namespace cube_edge_sum_l3201_320174

theorem cube_edge_sum (surface_area : ℝ) (edge_sum : ℝ) : 
  surface_area = 150 → edge_sum = 12 * (surface_area / 6).sqrt → edge_sum = 60 := by
  sorry

end cube_edge_sum_l3201_320174


namespace power_of_64_l3201_320199

theorem power_of_64 : (64 : ℝ) ^ (5/6) = 32 := by sorry

end power_of_64_l3201_320199


namespace problem_1_problem_2_l3201_320139

-- Part 1
theorem problem_1 : (2 - Real.sqrt 3) ^ 0 - Real.sqrt 12 + Real.tan (π / 3) = 1 - Real.sqrt 3 := by sorry

-- Part 2
theorem problem_2 (a b : ℝ) (h : a ≠ b) : (a - b) / (a + b) / (b - a) = -1 / (a + b) := by sorry

end problem_1_problem_2_l3201_320139


namespace alternating_sum_2023_l3201_320108

/-- Calculates the sum of the alternating series from 1 to n -/
def alternatingSum (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

/-- The sum of the series 1-2+3-4+5-6+...-2022+2023 equals 1012 -/
theorem alternating_sum_2023 :
  alternatingSum 2023 = 1012 := by
  sorry

#eval alternatingSum 2023

end alternating_sum_2023_l3201_320108


namespace four_different_results_l3201_320193

/-- Represents a parenthesized expression of 3^3^3^3 -/
inductive ParenthesizedExpr
| Single : ParenthesizedExpr
| Left : ParenthesizedExpr → ParenthesizedExpr
| Right : ParenthesizedExpr → ParenthesizedExpr
| Both : ParenthesizedExpr → ParenthesizedExpr → ParenthesizedExpr

/-- Evaluates a parenthesized expression to a natural number -/
def evaluate : ParenthesizedExpr → ℕ
| ParenthesizedExpr.Single => 3^3^3^3
| ParenthesizedExpr.Left e => 3^(evaluate e)
| ParenthesizedExpr.Right e => (evaluate e)^3
| ParenthesizedExpr.Both e1 e2 => (evaluate e1)^(evaluate e2)

/-- All possible parenthesized expressions of 3^3^3^3 -/
def allExpressions : List ParenthesizedExpr := [
  ParenthesizedExpr.Single,
  ParenthesizedExpr.Left (ParenthesizedExpr.Left (ParenthesizedExpr.Single)),
  ParenthesizedExpr.Left (ParenthesizedExpr.Right ParenthesizedExpr.Single),
  ParenthesizedExpr.Right (ParenthesizedExpr.Left ParenthesizedExpr.Single),
  ParenthesizedExpr.Right (ParenthesizedExpr.Right ParenthesizedExpr.Single),
  ParenthesizedExpr.Both ParenthesizedExpr.Single ParenthesizedExpr.Single
]

/-- The theorem stating that there are exactly 4 different results -/
theorem four_different_results :
  (allExpressions.map evaluate).toFinset.card = 4 := by sorry

end four_different_results_l3201_320193


namespace geometric_sequence_a1_l3201_320163

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (q : ℚ), ∀ (n : ℕ), a (n + 1) = a n * q

theorem geometric_sequence_a1 (a : ℕ → ℚ) :
  geometric_sequence a →
  a 2 * a 5 = 2 * a 3 →
  (a 4 + a 6) / 2 = 5/4 →
  a 1 = 16 ∨ a 1 = -16 :=
by sorry

end geometric_sequence_a1_l3201_320163


namespace expression_value_l3201_320126

theorem expression_value (p q r : ℝ) 
  (hp : p ≠ 2) (hq : q ≠ 5) (hr : r ≠ 7) : 
  ((p - 2) / (7 - r)) * ((q - 5) / (2 - p)) * ((r - 7) / (5 - q)) = -1 := by
  sorry

end expression_value_l3201_320126


namespace max_product_decomposition_l3201_320132

theorem max_product_decomposition (n k : ℕ) (h : k ≤ n) :
  ∃ (decomp : List ℕ),
    (decomp.sum = n) ∧
    (decomp.length = k) ∧
    (∀ (other_decomp : List ℕ),
      (other_decomp.sum = n) ∧ (other_decomp.length = k) →
      decomp.prod ≥ other_decomp.prod) ∧
    (decomp = List.replicate (n - n / k * k) (n / k + 1) ++ List.replicate (k - (n - n / k * k)) (n / k)) :=
  sorry

end max_product_decomposition_l3201_320132


namespace price_per_game_l3201_320136

def playstation_cost : ℝ := 500
def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def games_to_sell : ℕ := 20

theorem price_per_game :
  (playstation_cost - (birthday_money + christmas_money)) / games_to_sell = 7.5 := by
  sorry

end price_per_game_l3201_320136


namespace jack_buttons_theorem_l3201_320198

/-- The number of buttons Jack needs for all shirts -/
def total_buttons (shirts_per_kid : ℕ) (num_kids : ℕ) (buttons_per_shirt : ℕ) : ℕ :=
  shirts_per_kid * num_kids * buttons_per_shirt

/-- Theorem stating that Jack needs 63 buttons for all shirts -/
theorem jack_buttons_theorem :
  total_buttons 3 3 7 = 63 := by
  sorry

end jack_buttons_theorem_l3201_320198


namespace cleaning_time_proof_l3201_320150

theorem cleaning_time_proof (total_time : ℝ) (lilly_fraction : ℝ) : 
  total_time = 8 → lilly_fraction = 1/4 → 
  (total_time - lilly_fraction * total_time) * 60 = 360 := by
  sorry

end cleaning_time_proof_l3201_320150


namespace angle_rotation_l3201_320166

theorem angle_rotation (initial_angle rotation : ℝ) (h1 : initial_angle = 25) (h2 : rotation = 350) :
  (initial_angle - (rotation - 360)) % 360 = 15 :=
sorry

end angle_rotation_l3201_320166


namespace no_prime_sum_10001_l3201_320145

/-- A function that returns the number of ways to write n as the sum of two primes -/
def countPrimePairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range n)).card

/-- Theorem stating that 10001 cannot be written as the sum of two primes -/
theorem no_prime_sum_10001 : countPrimePairs 10001 = 0 := by sorry

end no_prime_sum_10001_l3201_320145


namespace coefficient_of_y_l3201_320149

theorem coefficient_of_y (y : ℝ) : 
  let expression := 5 * (y - 6) + 6 * (9 - 3 * y^2 + 7 * y) - 10 * (3 * y - 2)
  ∃ a b c : ℝ, expression = a * y^2 + 17 * y + c :=
by sorry

end coefficient_of_y_l3201_320149


namespace box_width_is_15_l3201_320188

/-- Given a rectangular box with length 8 cm and height 5 cm, built using 10 cubic cm cubes,
    and requiring a minimum of 60 cubes, prove that the width of the box is 15 cm. -/
theorem box_width_is_15 (length : ℝ) (height : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) :
  length = 8 →
  height = 5 →
  cube_volume = 10 →
  min_cubes = 60 →
  (min_cubes : ℝ) * cube_volume / (length * height) = 15 := by
  sorry

end box_width_is_15_l3201_320188


namespace complement_A_intersect_B_l3201_320172

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {2, 3}
def B : Set Int := {0, 1}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {0, 1} := by sorry

end complement_A_intersect_B_l3201_320172


namespace rectangular_prism_surface_area_l3201_320180

/-- The surface area of a rectangular prism with given dimensions -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

/-- Theorem: The surface area of a rectangular prism with length 5, width 4, and height 3 is 94 -/
theorem rectangular_prism_surface_area :
  surface_area 5 4 3 = 94 := by
  sorry

end rectangular_prism_surface_area_l3201_320180


namespace triangle_construction_existence_and_uniqueness_l3201_320135

-- Define the triangle structure
structure Triangle where
  sideA : ℝ
  sideB : ℝ
  angleC : ℝ
  sideA_pos : 0 < sideA
  sideB_pos : 0 < sideB
  angle_valid : 0 < angleC ∧ angleC < π

-- Theorem statement
theorem triangle_construction_existence_and_uniqueness 
  (a b : ℝ) (γ : ℝ) (ha : 0 < a) (hb : 0 < b) (hγ : 0 < γ ∧ γ < π) :
  ∃! t : Triangle, t.sideA = a ∧ t.sideB = b ∧ t.angleC = γ :=
sorry

end triangle_construction_existence_and_uniqueness_l3201_320135


namespace solve_smores_problem_l3201_320186

def smores_problem (graham_crackers_per_smore : ℕ) 
                   (total_graham_crackers : ℕ) 
                   (initial_marshmallows : ℕ) 
                   (additional_marshmallows : ℕ) : Prop :=
  let total_smores := total_graham_crackers / graham_crackers_per_smore
  let total_marshmallows := initial_marshmallows + additional_marshmallows
  (total_marshmallows / total_smores = 1)

theorem solve_smores_problem :
  smores_problem 2 48 6 18 := by
  sorry

end solve_smores_problem_l3201_320186


namespace oak_grove_library_books_l3201_320177

theorem oak_grove_library_books :
  let public_library : ℝ := 1986
  let school_libraries : ℝ := 5106
  let community_college_library : ℝ := 3294.5
  let medical_library : ℝ := 1342.25
  let law_library : ℝ := 2785.75
  public_library + school_libraries + community_college_library + medical_library + law_library = 15514.5 := by
  sorry

end oak_grove_library_books_l3201_320177


namespace tissue_purchase_cost_l3201_320156

/-- Calculate the total cost of tissues with discounts and tax -/
theorem tissue_purchase_cost
  (num_boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (price_per_tissue : ℚ)
  (pack_discount : ℚ)
  (volume_discount : ℚ)
  (tax_rate : ℚ)
  (volume_discount_threshold : ℕ)
  (h_num_boxes : num_boxes = 25)
  (h_packs_per_box : packs_per_box = 18)
  (h_tissues_per_pack : tissues_per_pack = 150)
  (h_price_per_tissue : price_per_tissue = 6 / 100)
  (h_pack_discount : pack_discount = 10 / 100)
  (h_volume_discount : volume_discount = 8 / 100)
  (h_tax_rate : tax_rate = 5 / 100)
  (h_volume_discount_threshold : volume_discount_threshold = 10)
  : ∃ (total_cost : ℚ), total_cost = 3521.07 :=
by
  sorry

#check tissue_purchase_cost

end tissue_purchase_cost_l3201_320156


namespace sales_tax_percentage_l3201_320127

/-- Represents the problem of calculating sales tax percentage --/
theorem sales_tax_percentage
  (total_worth : ℝ)
  (tax_rate : ℝ)
  (tax_free_cost : ℝ)
  (h1 : total_worth = 40)
  (h2 : tax_rate = 0.06)
  (h3 : tax_free_cost = 34.7) :
  (total_worth - tax_free_cost) * tax_rate / total_worth = 0.0075 := by
  sorry

end sales_tax_percentage_l3201_320127


namespace triangle_value_l3201_320117

theorem triangle_value (q : ℝ) (h1 : 2 * triangle + q = 134) (h2 : 2 * (triangle + q) + q = 230) :
  triangle = 43 := by sorry

end triangle_value_l3201_320117


namespace number_divided_by_005_equals_900_l3201_320162

theorem number_divided_by_005_equals_900 (x : ℝ) : x / 0.05 = 900 → x = 45 := by
  sorry

end number_divided_by_005_equals_900_l3201_320162


namespace ceiling_x_squared_values_l3201_320197

theorem ceiling_x_squared_values (x : ℝ) (h : ⌈x⌉ = 9) :
  ∃ (S : Finset ℕ), (∀ n ∈ S, ∃ y : ℝ, ⌈y⌉ = 9 ∧ ⌈y^2⌉ = n) ∧ S.card = 17 :=
sorry

end ceiling_x_squared_values_l3201_320197


namespace circle_constant_l3201_320114

theorem circle_constant (r : ℝ) (k : ℝ) (h1 : r = 36) (h2 : 2 * π * r = 72 * k) : k = π := by
  sorry

end circle_constant_l3201_320114


namespace fraction_calculation_l3201_320144

theorem fraction_calculation : 
  (2 / 5 + 3 / 7) / ((4 / 9) * (1 / 8)) = 522 / 35 := by
  sorry

end fraction_calculation_l3201_320144


namespace bacteria_growth_3hours_l3201_320182

/-- The number of bacteria after a given time, given an initial population and doubling time. -/
def bacteriaPopulation (initialPopulation : ℕ) (doublingTimeMinutes : ℕ) (totalTimeMinutes : ℕ) : ℕ :=
  initialPopulation * 2 ^ (totalTimeMinutes / doublingTimeMinutes)

/-- Theorem stating that after 3 hours, starting with 1 bacterium that doubles every 20 minutes, 
    the population will be 512. -/
theorem bacteria_growth_3hours :
  bacteriaPopulation 1 20 180 = 512 := by
  sorry

end bacteria_growth_3hours_l3201_320182


namespace extremum_implies_a_equals_e_l3201_320192

/-- If f(x) = e^x - ax has an extremum at x = 1, then a = e -/
theorem extremum_implies_a_equals_e (a : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = Real.exp x - a * x) ∧ 
   (∃ ε > 0, ∀ h ≠ 0, |h| < ε → f (1 + h) ≤ f 1)) → 
  a = Real.exp 1 := by
sorry

end extremum_implies_a_equals_e_l3201_320192


namespace solve_square_equation_solve_cubic_equation_l3201_320194

-- Part 1
theorem solve_square_equation :
  ∀ x : ℝ, (x - 1)^2 = 9 ↔ x = 4 ∨ x = -2 :=
by sorry

-- Part 2
theorem solve_cubic_equation :
  ∀ x : ℝ, (1/3) * (x + 3)^3 - 9 = 0 ↔ x = 0 :=
by sorry

end solve_square_equation_solve_cubic_equation_l3201_320194


namespace cone_base_circumference_l3201_320179

theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = 18 * Real.pi ∧ h = 3 ∧ V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 6 * Real.sqrt 2 * Real.pi :=
by sorry

end cone_base_circumference_l3201_320179


namespace math_class_registration_l3201_320122

theorem math_class_registration (total : ℕ) (history : ℕ) (english : ℕ) (all_three : ℕ) (exactly_two : ℕ) :
  total = 68 →
  history = 21 →
  english = 34 →
  all_three = 3 →
  exactly_two = 7 →
  ∃ (math : ℕ), math = 14 ∧ 
    total = history + math + english - (exactly_two - all_three) - all_three :=
by sorry

end math_class_registration_l3201_320122


namespace product_of_roots_l3201_320191

theorem product_of_roots : Real.sqrt 16 * (27 ^ (1/3 : ℝ)) = 12 := by
  sorry

end product_of_roots_l3201_320191


namespace geometric_sequence_26th_term_l3201_320134

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_26th_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_14th : a 14 = 10)
  (h_20th : a 20 = 80) :
  a 26 = 640 := by
  sorry

end geometric_sequence_26th_term_l3201_320134


namespace circle_diameter_theorem_l3201_320137

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define a point inside or on a circle
def PointInOrOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 ≤ c.radius^2

-- Define a diameter of a circle
def Diameter (c : Circle) (d : ℝ × ℝ → ℝ × ℝ → Prop) : Prop :=
  ∀ p q, d p q → PointOnCircle c p ∧ PointOnCircle c q ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = 4 * c.radius^2

-- Define a point being on one side of a diameter
def OnOneSideOfDiameter (c : Circle) (d : ℝ × ℝ → ℝ × ℝ → Prop) (p : ℝ × ℝ) : Prop :=
  ∃ q r, Diameter c d ∧ d q r ∧
    ((p.1 - q.1) * (r.2 - q.2) - (p.2 - q.2) * (r.1 - q.1) ≥ 0 ∨
     (p.1 - q.1) * (r.2 - q.2) - (p.2 - q.2) * (r.1 - q.1) ≤ 0)

theorem circle_diameter_theorem (ω : Circle) (inner_circle : Circle) (points : Finset (ℝ × ℝ)) 
    (h1 : ∀ p ∈ points, PointOnCircle ω p)
    (h2 : inner_circle.radius < ω.radius)
    (h3 : ∀ p ∈ points, PointInOrOnCircle inner_circle p) :
  ∃ d : ℝ × ℝ → ℝ × ℝ → Prop, Diameter ω d ∧ 
    (∀ p ∈ points, OnOneSideOfDiameter ω d p) ∧
    (∀ p q, d p q → p ∉ points ∧ q ∉ points) :=
  sorry

end circle_diameter_theorem_l3201_320137


namespace suresh_completion_time_l3201_320113

theorem suresh_completion_time (ashutosh_time : ℝ) (suresh_partial_time : ℝ) (ashutosh_partial_time : ℝ) 
  (h1 : ashutosh_time = 30)
  (h2 : suresh_partial_time = 9)
  (h3 : ashutosh_partial_time = 12)
  : ∃ (suresh_time : ℝ), 
    suresh_partial_time / suresh_time + ashutosh_partial_time / ashutosh_time = 1 ∧ 
    suresh_time = 15 := by
  sorry

end suresh_completion_time_l3201_320113


namespace integral_x_squared_plus_one_over_x_l3201_320169

open Real MeasureTheory Interval

theorem integral_x_squared_plus_one_over_x :
  ∫ x in (1 : ℝ)..2, (x^2 + 1) / x = 3/2 + Real.log 2 := by
  sorry

end integral_x_squared_plus_one_over_x_l3201_320169


namespace alcohol_solution_problem_l3201_320157

theorem alcohol_solution_problem (initial_alcohol_percentage : ℝ) 
                                 (water_added : ℝ) 
                                 (final_alcohol_percentage : ℝ) : 
  initial_alcohol_percentage = 0.26 →
  water_added = 5 →
  final_alcohol_percentage = 0.195 →
  ∃ (initial_volume : ℝ),
    initial_volume * initial_alcohol_percentage = 
    (initial_volume + water_added) * final_alcohol_percentage ∧
    initial_volume = 15 := by
sorry

end alcohol_solution_problem_l3201_320157


namespace a_4_equals_8_l3201_320129

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem a_4_equals_8 (a : ℕ → ℝ) 
    (h1 : a 1 = 1)
    (h2 : ∀ (n : ℕ), a (n + 1) = 2 * a n) : 
  a 4 = 8 := by
  sorry

end a_4_equals_8_l3201_320129


namespace largest_x_and_ratio_l3201_320148

theorem largest_x_and_ratio (a b c d : ℤ) (x : ℝ) : 
  (7 * x / 8 + 1 = 4 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (x ≤ (-8 + 4 * Real.sqrt 15) / 7) →
  (x = (-8 + 4 * Real.sqrt 15) / 7 → a = -8 ∧ b = 4 ∧ c = 15 ∧ d = 7) →
  (a * c * d / b = -210) :=
by sorry

end largest_x_and_ratio_l3201_320148


namespace fraction_evaluation_l3201_320187

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end fraction_evaluation_l3201_320187


namespace chocolate_chips_per_family_member_l3201_320176

/-- Represents the number of chocolate chips per cookie for each type --/
structure ChocolateChipsPerCookie :=
  (chocolate_chip : ℕ)
  (double_chocolate_chip : ℕ)
  (white_chocolate_chip : ℕ)

/-- Represents the number of cookies per batch for each type --/
structure CookiesPerBatch :=
  (chocolate_chip : ℕ)
  (double_chocolate_chip : ℕ)
  (white_chocolate_chip : ℕ)

/-- Represents the number of batches for each type of cookie --/
structure Batches :=
  (chocolate_chip : ℕ)
  (double_chocolate_chip : ℕ)
  (white_chocolate_chip : ℕ)

def total_chocolate_chips (chips_per_cookie : ChocolateChipsPerCookie) 
                          (cookies_per_batch : CookiesPerBatch) 
                          (batches : Batches) : ℕ :=
  chips_per_cookie.chocolate_chip * cookies_per_batch.chocolate_chip * batches.chocolate_chip +
  chips_per_cookie.double_chocolate_chip * cookies_per_batch.double_chocolate_chip * batches.double_chocolate_chip +
  chips_per_cookie.white_chocolate_chip * cookies_per_batch.white_chocolate_chip * batches.white_chocolate_chip

theorem chocolate_chips_per_family_member 
  (chips_per_cookie : ChocolateChipsPerCookie)
  (cookies_per_batch : CookiesPerBatch)
  (batches : Batches)
  (family_members : ℕ)
  (h1 : chips_per_cookie = ⟨2, 4, 3⟩)
  (h2 : cookies_per_batch = ⟨12, 10, 15⟩)
  (h3 : batches = ⟨3, 2, 1⟩)
  (h4 : family_members = 4)
  : (total_chocolate_chips chips_per_cookie cookies_per_batch batches) / family_members = 49 :=
by
  sorry

end chocolate_chips_per_family_member_l3201_320176


namespace unique_integer_representation_l3201_320107

theorem unique_integer_representation (A m n p : ℕ) : 
  A > 0 ∧ 
  m ≥ n ∧ n ≥ p ∧ p ≥ 1 ∧
  A = (m - 1/n) * (n - 1/p) * (p - 1/m) →
  A = 21 :=
by sorry

end unique_integer_representation_l3201_320107


namespace ratio_p_to_r_l3201_320175

theorem ratio_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 3 / 5)
  (h2 : r / s = 5 / 4)
  (h3 : s / q = 1 / 3) :
  p / r = 36 / 25 := by
  sorry

end ratio_p_to_r_l3201_320175


namespace field_width_l3201_320118

/-- The width of a rectangular field given its area and length -/
theorem field_width (area : ℝ) (length : ℝ) (h1 : area = 143.2) (h2 : length = 4) :
  area / length = 35.8 := by
sorry

end field_width_l3201_320118


namespace scientific_notation_of_11580000_l3201_320131

theorem scientific_notation_of_11580000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 11580000 = a * (10 : ℝ) ^ n ∧ a = 1.158 ∧ n = 7 := by
  sorry

end scientific_notation_of_11580000_l3201_320131


namespace election_majority_l3201_320121

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 6000 → 
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 1200 := by
sorry

end election_majority_l3201_320121


namespace central_figure_diameter_bound_a_central_figure_diameter_bound_l3201_320196

/-- A convex figure with diameter 1 --/
class ConvexFigure (F : Type*) :=
  (diameter : ℝ)
  (is_one : diameter = 1)

/-- A planar convex figure --/
class PlanarConvexFigure (F : Type*) extends ConvexFigure F

/-- A spatial convex figure --/
class SpatialConvexFigure (F : Type*) extends ConvexFigure F

/-- The diameter of the central figure L(F) --/
def central_figure_diameter (F : Type*) [ConvexFigure F] : ℝ := sorry

/-- The diameter of the a-central figure II(a, F) --/
def a_central_figure_diameter (F : Type*) [ConvexFigure F] (a : ℝ) : ℝ := sorry

theorem central_figure_diameter_bound 
  (F : Type*) [ConvexFigure F] : 
  (PlanarConvexFigure F → central_figure_diameter F ≤ 1/2) ∧ 
  (SpatialConvexFigure F → central_figure_diameter F ≤ Real.sqrt 2 / 2) := by sorry

theorem a_central_figure_diameter_bound 
  (F : Type*) [ConvexFigure F] (a : ℝ) : 
  (PlanarConvexFigure F → a_central_figure_diameter F a ≤ 1 - a^2/2) ∧ 
  (SpatialConvexFigure F → a_central_figure_diameter F a ≤ Real.sqrt (1 - a^2/2)) := by sorry

end central_figure_diameter_bound_a_central_figure_diameter_bound_l3201_320196


namespace least_number_remainder_l3201_320111

theorem least_number_remainder : ∃ (r : ℕ), r > 0 ∧ 386 % 35 = r ∧ 386 % 11 = r := by
  sorry

end least_number_remainder_l3201_320111


namespace train_speed_l3201_320143

/-- The speed of a train given its length, the speed of a man running in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_speed (train_length : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_length = 110 →
  man_speed = 4 →
  passing_time = 9 / 3600 →
  (train_length / 1000) / passing_time - man_speed = 40 :=
by sorry

end train_speed_l3201_320143


namespace palindrome_divisible_by_seven_probability_l3201_320110

/-- A function that checks if a number is a palindrome -/
def is_palindrome (n : ℕ) : Prop := sorry

/-- A function that generates all 5-digit palindromes -/
def five_digit_palindromes : Finset ℕ := sorry

/-- A function that counts the number of elements in a finite set satisfying a predicate -/
def count_satisfying {α : Type*} (s : Finset α) (p : α → Prop) : ℕ := sorry

/-- The main theorem -/
theorem palindrome_divisible_by_seven_probability :
  ∃ k : ℕ, (k : ℚ) / 900 = (count_satisfying five_digit_palindromes 
    (λ n => (n % 7 = 0) ∧ (is_palindrome (n / 7)))) / (five_digit_palindromes.card) :=
sorry

end palindrome_divisible_by_seven_probability_l3201_320110


namespace diophantine_equation_solutions_l3201_320112

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, x^2 = 1 + 4*y^3*(y + 2) ↔ 
    (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = -2) := by
  sorry

end diophantine_equation_solutions_l3201_320112


namespace reverse_digits_difference_reverse_digits_difference_proof_l3201_320190

def is_valid_k (k : ℕ) : Prop :=
  100 < k ∧ k < 1000

def reverse_digits (k : ℕ) : ℕ :=
  let h := k / 100
  let t := (k / 10) % 10
  let u := k % 10
  100 * u + 10 * t + h

theorem reverse_digits_difference (n : ℕ) : Prop :=
  ∃ (ks : Finset ℕ), 
    ks.card = 80 ∧ 
    (∀ k ∈ ks, is_valid_k k) ∧
    (∀ k ∈ ks, reverse_digits k = k + n) →
    n = 99

-- The proof goes here
theorem reverse_digits_difference_proof : reverse_digits_difference 99 := by
  sorry

end reverse_digits_difference_reverse_digits_difference_proof_l3201_320190


namespace tuesday_to_monday_work_ratio_l3201_320141

theorem tuesday_to_monday_work_ratio :
  let monday : ℚ := 3/4
  let wednesday : ℚ := 2/3
  let thursday : ℚ := 5/6
  let friday : ℚ := 75/60
  let total : ℚ := 4
  let tuesday : ℚ := total - (monday + wednesday + thursday + friday)
  tuesday / monday = 2/3 := by
  sorry

end tuesday_to_monday_work_ratio_l3201_320141


namespace students_neither_sport_l3201_320104

theorem students_neither_sport (total : ℕ) (football : ℕ) (cricket : ℕ) (both : ℕ) :
  total = 470 →
  football = 325 →
  cricket = 175 →
  both = 80 →
  total - (football + cricket - both) = 50 := by
  sorry

end students_neither_sport_l3201_320104


namespace existence_of_incommensurable_segments_l3201_320109

-- Define incommensurability
def incommensurable (x y : ℝ) : Prop :=
  ∀ k : ℚ, k ≠ 0 → x ≠ k * y

-- State the theorem
theorem existence_of_incommensurable_segments :
  ∃ (a b c d : ℝ),
    a + b + c = d ∧
    incommensurable a d ∧
    incommensurable b d ∧
    incommensurable c d :=
by sorry

end existence_of_incommensurable_segments_l3201_320109


namespace intersection_points_vary_at_least_one_intersection_l3201_320173

/-- The number of intersection points between y = Bx^2 and y^3 + 2 = x^2 + 4y varies with B -/
theorem intersection_points_vary (B : ℝ) (hB : B > 0) :
  ∃ (x y : ℝ), y = B * x^2 ∧ y^3 + 2 = x^2 + 4 * y ∧
  ∃ (B₁ B₂ : ℝ) (hB₁ : B₁ > 0) (hB₂ : B₂ > 0),
    (∀ (x₁ y₁ : ℝ), y₁ = B₁ * x₁^2 → y₁^3 + 2 = x₁^2 + 4 * y₁ →
      ∀ (x₂ y₂ : ℝ), y₂ = B₂ * x₂^2 → y₂^3 + 2 = x₂^2 + 4 * y₂ →
        (x₁, y₁) ≠ (x₂, y₂)) :=
by
  sorry

/-- There is at least one intersection point for any positive B -/
theorem at_least_one_intersection (B : ℝ) (hB : B > 0) :
  ∃ (x y : ℝ), y = B * x^2 ∧ y^3 + 2 = x^2 + 4 * y :=
by
  sorry

end intersection_points_vary_at_least_one_intersection_l3201_320173


namespace cubic_sum_minus_product_l3201_320155

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 13) 
  (sum_prod_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1027 := by
sorry

end cubic_sum_minus_product_l3201_320155


namespace point_in_intersection_l3201_320102

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set A
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + m > 0}

-- Define set B
def B (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - n > 0}

-- Define the complement of B with respect to U
def C_U_B (n : ℝ) : Set (ℝ × ℝ) := U \ B n

-- Define point P
def P : ℝ × ℝ := (2, 3)

-- State the theorem
theorem point_in_intersection (m n : ℝ) :
  P ∈ A m ∩ C_U_B n ↔ m > -1 ∧ n ≥ 5 := by
  sorry

end point_in_intersection_l3201_320102
