import Mathlib

namespace job_completion_time_l4032_403262

/-- Given a job that A and B can complete together in 5 days, and B can complete alone in 10 days,
    prove that A can complete the job alone in 10 days. -/
theorem job_completion_time (rate_A rate_B : ℝ) : 
  rate_A + rate_B = 1 / 5 →  -- A and B together complete the job in 5 days
  rate_B = 1 / 10 →          -- B alone completes the job in 10 days
  rate_A = 1 / 10            -- A alone completes the job in 10 days
:= by sorry

end job_completion_time_l4032_403262


namespace bettys_herb_garden_l4032_403234

theorem bettys_herb_garden (basil oregano : ℕ) : 
  oregano = 2 * basil + 2 →
  basil + oregano = 17 →
  basil = 5 := by sorry

end bettys_herb_garden_l4032_403234


namespace min_value_of_f_min_value_of_sum_squares_l4032_403202

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x - 1) + abs (2 * x + 1)

-- Theorem 1: The minimum value of f(x) is 3
theorem min_value_of_f : ∃ k : ℝ, k = 3 ∧ ∀ x : ℝ, f x ≥ k :=
sorry

-- Theorem 2: Minimum value of a² + b² + c² given the conditions
theorem min_value_of_sum_squares :
  ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  3 * a + 2 * b + c = 3 →
  a^2 + b^2 + c^2 ≥ 9/14 :=
sorry

end min_value_of_f_min_value_of_sum_squares_l4032_403202


namespace smallest_square_multiplier_l4032_403201

def y : ℕ := 2^4 * 3^2 * 4^3 * 5^3 * 6^2 * 7^3 * 8^3 * 9^2

theorem smallest_square_multiplier :
  (∀ k : ℕ, k > 0 ∧ k < 350 → ¬ ∃ m : ℕ, k * y = m^2) ∧
  ∃ m : ℕ, 350 * y = m^2 :=
sorry

end smallest_square_multiplier_l4032_403201


namespace derivative_f_at_zero_l4032_403232

-- Define the function f
def f (x : ℝ) : ℝ := 25 * x^3 + 13 * x^2 + 2016 * x - 5

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 2016 := by sorry

end derivative_f_at_zero_l4032_403232


namespace delta_value_l4032_403206

theorem delta_value : ∀ Δ : ℤ, 5 * (-3) = Δ - 3 → Δ = -12 := by
  sorry

end delta_value_l4032_403206


namespace not_closed_sequence_3_pow_arithmetic_closed_sequence_iff_l4032_403240

/-- Definition of a closed sequence -/
def is_closed_sequence (a : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, ∃ k : ℕ, a m + a n = a k

/-- The sequence a_n = 3^n is not a closed sequence -/
theorem not_closed_sequence_3_pow : ¬ is_closed_sequence (λ n => 3^n) := by sorry

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- Necessary and sufficient condition for an arithmetic sequence to be a closed sequence -/
theorem arithmetic_closed_sequence_iff (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d →
  (is_closed_sequence a ↔ ∃ m : ℤ, m ≥ -1 ∧ a 1 = m * d) := by sorry

end not_closed_sequence_3_pow_arithmetic_closed_sequence_iff_l4032_403240


namespace min_value_expression_min_value_achieved_l4032_403298

theorem min_value_expression (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (h : x * y * z * w = 16) :
  x + 2 * y + 4 * z + 8 * w ≥ 16 :=
sorry

theorem min_value_achieved (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (h : x * y * z * w = 16) :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * b * c * d = 16 ∧ a + 2 * b + 4 * c + 8 * d = 16 :=
sorry

end min_value_expression_min_value_achieved_l4032_403298


namespace initial_average_height_l4032_403215

theorem initial_average_height (n : ℕ) (wrong_height correct_height actual_average : ℝ) 
  (h1 : n = 35)
  (h2 : wrong_height = 166)
  (h3 : correct_height = 106)
  (h4 : actual_average = 179) :
  (n * actual_average + (wrong_height - correct_height)) / n = 181 :=
by sorry

end initial_average_height_l4032_403215


namespace count_triples_eq_12_l4032_403283

/-- Least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a,b,c) satisfying the given conditions -/
def count_triples : ℕ := sorry

theorem count_triples_eq_12 :
  count_triples = 12 := by sorry

end count_triples_eq_12_l4032_403283


namespace probability_red_ball_is_two_fifths_l4032_403286

/-- The probability of drawing a red ball from a bag -/
def probability_red_ball (total_balls : ℕ) (red_balls : ℕ) : ℚ :=
  red_balls / total_balls

/-- The total number of balls in the bag -/
def total_balls : ℕ := 5

/-- The number of red balls in the bag -/
def red_balls : ℕ := 2

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

theorem probability_red_ball_is_two_fifths :
  probability_red_ball total_balls red_balls = 2 / 5 := by
  sorry

end probability_red_ball_is_two_fifths_l4032_403286


namespace sum_of_reciprocals_l4032_403212

theorem sum_of_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / a + 1 / b = (a + b) / (a * b) := by
  sorry

end sum_of_reciprocals_l4032_403212


namespace ellipse_equation_form_l4032_403282

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_axes : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  sorry

theorem ellipse_equation_form (e : Ellipse) 
  (h_center : e.center = (0, 0))
  (h_foci : e.foci_on_axes = true)
  (h_eccentricity : e.eccentricity = Real.sqrt 3 / 2)
  (h_point : e.passes_through = (2, 0)) :
  (ellipse_equation e = fun x y => x^2 + 4*y^2 = 4) ∨
  (ellipse_equation e = fun x y => 4*x^2 + y^2 = 16) :=
sorry

end ellipse_equation_form_l4032_403282


namespace quadrilateral_prism_volume_l4032_403217

/-- A quadrilateral prism with specific properties -/
structure QuadrilateralPrism where
  -- The base is a rhombus with apex angle 60°
  base_is_rhombus : Bool
  base_apex_angle : ℝ
  -- The angle between each face and the base is 60°
  face_base_angle : ℝ
  -- There exists a point inside with distance 1 to base and each face
  interior_point_exists : Bool
  -- Volume of the prism
  volume : ℝ

/-- The volume of a quadrilateral prism with specific properties is 8√3 -/
theorem quadrilateral_prism_volume 
  (P : QuadrilateralPrism) 
  (h1 : P.base_is_rhombus = true)
  (h2 : P.base_apex_angle = 60)
  (h3 : P.face_base_angle = 60)
  (h4 : P.interior_point_exists = true) :
  P.volume = 8 * Real.sqrt 3 := by
  sorry

end quadrilateral_prism_volume_l4032_403217


namespace parkway_soccer_boys_percentage_l4032_403291

/-- Given the student population data for the fifth grade at Parkway Elementary School,
    prove that 86% of the students playing soccer are boys. -/
theorem parkway_soccer_boys_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (soccer_players : ℕ)
  (girls_not_playing : ℕ)
  (h1 : total_students = 470)
  (h2 : boys = 300)
  (h3 : soccer_players = 250)
  (h4 : girls_not_playing = 135)
  : (boys_playing_soccer : ℚ) / soccer_players * 100 = 86 :=
by sorry

end parkway_soccer_boys_percentage_l4032_403291


namespace greatest_integer_prime_quadratic_l4032_403241

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def abs_quadratic (x : ℤ) : ℕ := Int.natAbs (8 * x^2 - 53 * x + 21)

theorem greatest_integer_prime_quadratic :
  ∀ x : ℤ, x > 1 → ¬(is_prime (abs_quadratic x)) ∧
  (is_prime (abs_quadratic 1)) ∧
  (∀ y : ℤ, y ≤ 1 → is_prime (abs_quadratic y) → y = 1) :=
sorry

end greatest_integer_prime_quadratic_l4032_403241


namespace log_expression_equals_zero_l4032_403238

-- Define base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_zero :
  (1/2) * log10 4 + log10 5 - (π + 1)^0 = 0 := by sorry

end log_expression_equals_zero_l4032_403238


namespace wilted_flower_ratio_l4032_403248

theorem wilted_flower_ratio (initial_roses : ℕ) (remaining_flowers : ℕ) :
  initial_roses = 36 →
  remaining_flowers = 12 →
  (initial_roses / 2 - remaining_flowers) / (initial_roses / 2) = 1 / 3 :=
by sorry

end wilted_flower_ratio_l4032_403248


namespace smallest_common_multiple_13_8_lcm_13_8_l4032_403258

theorem smallest_common_multiple_13_8 : 
  ∀ n : ℕ, (13 ∣ n ∧ 8 ∣ n) → n ≥ 104 := by
  sorry

theorem lcm_13_8 : Nat.lcm 13 8 = 104 := by
  sorry

end smallest_common_multiple_13_8_lcm_13_8_l4032_403258


namespace jumping_probabilities_l4032_403216

/-- Probability of an athlete successfully jumping over a 2-meter high bar -/
structure Athlete where
  success_prob : ℝ
  success_prob_nonneg : 0 ≤ success_prob
  success_prob_le_one : success_prob ≤ 1

/-- The problem setup with two athletes A and B -/
def problem_setup (A B : Athlete) : Prop :=
  A.success_prob = 0.7 ∧ B.success_prob = 0.6

/-- The probability that A succeeds on the third attempt -/
def prob_A_third_attempt (A : Athlete) : ℝ :=
  (1 - A.success_prob) * (1 - A.success_prob) * A.success_prob

/-- The probability that at least one of A or B succeeds on the first attempt -/
def prob_at_least_one_first_attempt (A B : Athlete) : ℝ :=
  1 - (1 - A.success_prob) * (1 - B.success_prob)

/-- The probability that A succeeds exactly one more time than B in two attempts for each -/
def prob_A_one_more_than_B (A B : Athlete) : ℝ :=
  2 * A.success_prob * (1 - A.success_prob) * (1 - B.success_prob) * (1 - B.success_prob) +
  A.success_prob * A.success_prob * 2 * B.success_prob * (1 - B.success_prob)

theorem jumping_probabilities (A B : Athlete) 
  (h : problem_setup A B) : 
  prob_A_third_attempt A = 0.063 ∧
  prob_at_least_one_first_attempt A B = 0.88 ∧
  prob_A_one_more_than_B A B = 0.3024 := by
  sorry

end jumping_probabilities_l4032_403216


namespace bucket_volume_proof_l4032_403226

/-- The volume of water (in liters) that Tap A runs per minute -/
def tap_a_rate : ℝ := 3

/-- The time (in minutes) it takes Tap B to fill 1/3 of the bucket -/
def tap_b_third_time : ℝ := 20

/-- The time (in minutes) it takes both taps working together to fill the bucket -/
def combined_time : ℝ := 10

/-- The total volume of the bucket in liters -/
def bucket_volume : ℝ := 36

theorem bucket_volume_proof :
  let tap_b_rate := bucket_volume / (3 * tap_b_third_time)
  tap_a_rate + tap_b_rate = bucket_volume / combined_time := by
  sorry

end bucket_volume_proof_l4032_403226


namespace parabola_equation_hyperbola_equation_l4032_403204

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/64 + y^2/16 = 1

-- Define the parabola focus
def parabola_focus : ℝ × ℝ := (-8, 0)

-- Define the hyperbola asymptotes
def hyperbola_asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem for the parabola equation
theorem parabola_equation : 
  ∃ (x y : ℝ), (x, y) = parabola_focus → y^2 = -32*x := by sorry

-- Theorem for the hyperbola equation
theorem hyperbola_equation :
  (∀ (x y : ℝ), ellipse x y ↔ ellipse (-x) y) → 
  (∀ (x y : ℝ), hyperbola_asymptote x y) →
  ∃ (x y : ℝ), x^2/12 - y^2/36 = 1 := by sorry

end parabola_equation_hyperbola_equation_l4032_403204


namespace ellipse_k_range_l4032_403256

theorem ellipse_k_range (k : ℝ) :
  (∃ (x y : ℝ), 2 * x^2 + k * y^2 = 1 ∧ 
   ∃ (c : ℝ), c > 0 ∧ c^2 = 2 * x^2 + k * y^2 - k * (x^2 + y^2)) ↔ 
  (0 < k ∧ k < 2) :=
sorry

end ellipse_k_range_l4032_403256


namespace reflect_x_minus3_minus5_l4032_403264

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The theorem stating that reflecting P(-3,-5) across the x-axis results in (-3,5) -/
theorem reflect_x_minus3_minus5 :
  let P : Point := { x := -3, y := -5 }
  reflect_x P = { x := -3, y := 5 } := by
  sorry

end reflect_x_minus3_minus5_l4032_403264


namespace reflection_line_equation_l4032_403250

-- Define the triangle vertices and their images
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (8, 7)
def C : ℝ × ℝ := (6, -4)
def A' : ℝ × ℝ := (-5, 2)
def B' : ℝ × ℝ := (-10, 7)
def C' : ℝ × ℝ := (-8, -4)

-- Define the reflection line
def L (x : ℝ) : Prop := x = -1

-- Theorem statement
theorem reflection_line_equation :
  (∀ p p', (p = A ∧ p' = A') ∨ (p = B ∧ p' = B') ∨ (p = C ∧ p' = C') →
    p.2 = p'.2 ∧ L ((p.1 + p'.1) / 2)) →
  L (-1) :=
sorry

end reflection_line_equation_l4032_403250


namespace fifteen_factorial_base_nine_zeros_l4032_403219

/-- The number of trailing zeros in n! when written in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

theorem fifteen_factorial_base_nine_zeros :
  trailingZeros 15 9 = 3 :=
sorry

end fifteen_factorial_base_nine_zeros_l4032_403219


namespace max_value_expression_l4032_403236

theorem max_value_expression (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a + 1) * (b + 1) * (c + 1) / (a * b * c + 1) ≤ 16 / 7 := by
  sorry

end max_value_expression_l4032_403236


namespace finite_solutions_factorial_difference_l4032_403275

theorem finite_solutions_factorial_difference (u : ℕ+) :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), ∀ (n a b : ℕ),
    n! = u^a - u^b → (n, a, b) ∈ S :=
sorry

end finite_solutions_factorial_difference_l4032_403275


namespace correct_linear_system_l4032_403279

-- Define a structure for a system of two equations
structure EquationSystem where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ

-- Define the four systems of equations
def systemA : EquationSystem := {
  eq1 := fun x y => x + 5*y - 2,
  eq2 := fun x y => x*y - 7
}

def systemB : EquationSystem := {
  eq1 := fun x y => 2*x + 1 - 1,
  eq2 := fun x y => 3*x + 4*y
}

def systemC : EquationSystem := {
  eq1 := fun x y => 3*x^2 - 5*y,
  eq2 := fun x y => x + y - 4
}

def systemD : EquationSystem := {
  eq1 := fun x y => x - 2*y - 8,
  eq2 := fun x y => x + 3*y - 12
}

-- Define a predicate for linear equations with two variables
def isLinearSystem (s : EquationSystem) : Prop :=
  ∃ a b c d e f : ℝ, 
    (∀ x y, s.eq1 x y = a*x + b*y + c) ∧
    (∀ x y, s.eq2 x y = d*x + e*y + f)

-- Theorem statement
theorem correct_linear_system : 
  ¬(isLinearSystem systemA) ∧ 
  ¬(isLinearSystem systemB) ∧ 
  ¬(isLinearSystem systemC) ∧ 
  isLinearSystem systemD := by
  sorry

end correct_linear_system_l4032_403279


namespace value_of_b_l4032_403224

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 3) : b = 3 := by
  sorry

end value_of_b_l4032_403224


namespace simplify_square_roots_l4032_403290

theorem simplify_square_roots : Real.sqrt 81 - Real.sqrt 49 = 2 := by
  sorry

end simplify_square_roots_l4032_403290


namespace bank_teller_coin_rolls_l4032_403246

theorem bank_teller_coin_rolls 
  (total_coins : ℕ) 
  (num_tellers : ℕ) 
  (coins_per_roll : ℕ) 
  (h1 : total_coins = 1000) 
  (h2 : num_tellers = 4) 
  (h3 : coins_per_roll = 25) : 
  (total_coins / num_tellers) / coins_per_roll = 10 := by
sorry

end bank_teller_coin_rolls_l4032_403246


namespace smallest_with_18_divisors_l4032_403289

/-- Count the number of positive divisors of a natural number -/
def countDivisors (n : ℕ) : ℕ := sorry

/-- Check if a natural number has exactly 18 positive divisors -/
def has18Divisors (n : ℕ) : Prop := countDivisors n = 18

/-- The smallest positive integer with exactly 18 positive divisors -/
def smallestWith18Divisors : ℕ := 288

theorem smallest_with_18_divisors :
  (has18Divisors smallestWith18Divisors) ∧
  (∀ m : ℕ, m < smallestWith18Divisors → ¬(has18Divisors m)) :=
sorry

end smallest_with_18_divisors_l4032_403289


namespace arithmetic_mean_after_removal_l4032_403237

/-- Given a set of 50 numbers with arithmetic mean 38, prove that removing 45 and 55
    results in a new set with arithmetic mean 37.5 -/
theorem arithmetic_mean_after_removal (S : Finset ℝ) (sum_S : ℝ) : 
  S.card = 50 →
  sum_S = S.sum id →
  sum_S / 50 = 38 →
  45 ∈ S →
  55 ∈ S →
  let S' := S.erase 45 |>.erase 55
  let sum_S' := sum_S - 45 - 55
  sum_S' / S'.card = 37.5 := by
  sorry

end arithmetic_mean_after_removal_l4032_403237


namespace number_of_divisors_180_l4032_403276

theorem number_of_divisors_180 : Nat.card (Nat.divisors 180) = 18 := by
  sorry

end number_of_divisors_180_l4032_403276


namespace balloons_kept_winnie_keeps_balloons_l4032_403299

def total_balloons : ℕ := 22 + 44 + 78 + 90
def num_friends : ℕ := 10

theorem balloons_kept (total : ℕ) (friends : ℕ) (h : friends > 0) :
  total % friends = total - friends * (total / friends) :=
by sorry

theorem winnie_keeps_balloons :
  total_balloons % num_friends = 4 :=
by sorry

end balloons_kept_winnie_keeps_balloons_l4032_403299


namespace some_flying_creatures_are_magical_l4032_403265

-- Define our universe
variable (U : Type)

-- Define our predicates
variable (unicorn : U → Prop)
variable (flying : U → Prop)
variable (magical : U → Prop)

-- State the theorem
theorem some_flying_creatures_are_magical :
  (∀ x, unicorn x → flying x) →  -- All unicorns are capable of flying
  (∃ x, magical x ∧ unicorn x) →  -- Some magical creatures are unicorns
  (∃ x, flying x ∧ magical x) :=  -- Some flying creatures are magical creatures
by
  sorry

end some_flying_creatures_are_magical_l4032_403265


namespace bead_arrangement_probability_l4032_403218

def num_red : ℕ := 4
def num_white : ℕ := 2
def num_blue : ℕ := 2
def total_beads : ℕ := num_red + num_white + num_blue

def total_arrangements : ℕ := Nat.factorial total_beads / (Nat.factorial num_red * Nat.factorial num_white * Nat.factorial num_blue)

def valid_arrangements : ℕ := 27  -- This is an approximation based on the problem's solution

theorem bead_arrangement_probability :
  (valid_arrangements : ℚ) / total_arrangements = 9 / 140 :=
sorry

end bead_arrangement_probability_l4032_403218


namespace johnson_smith_tied_may_l4032_403295

/-- Represents the months of a baseball season --/
inductive Month
| Jan | Feb | Mar | Apr | May | Jul | Aug | Sep

/-- Represents a baseball player --/
structure Player where
  name : String
  homeRuns : Month → Nat

def johnson : Player :=
  { name := "Johnson"
  , homeRuns := fun
    | Month.Jan => 2
    | Month.Feb => 12
    | Month.Mar => 15
    | Month.Apr => 8
    | Month.May => 14
    | Month.Jul => 11
    | Month.Aug => 9
    | Month.Sep => 16 }

def smith : Player :=
  { name := "Smith"
  , homeRuns := fun
    | Month.Jan => 5
    | Month.Feb => 9
    | Month.Mar => 10
    | Month.Apr => 12
    | Month.May => 15
    | Month.Jul => 12
    | Month.Aug => 10
    | Month.Sep => 17 }

def totalHomeRunsUpTo (p : Player) (m : Month) : Nat :=
  match m with
  | Month.Jan => p.homeRuns Month.Jan
  | Month.Feb => p.homeRuns Month.Jan + p.homeRuns Month.Feb
  | Month.Mar => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar
  | Month.Apr => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr
  | Month.May => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr + p.homeRuns Month.May
  | Month.Jul => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr + p.homeRuns Month.May + p.homeRuns Month.Jul
  | Month.Aug => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr + p.homeRuns Month.May + p.homeRuns Month.Jul + p.homeRuns Month.Aug
  | Month.Sep => p.homeRuns Month.Jan + p.homeRuns Month.Feb + p.homeRuns Month.Mar + p.homeRuns Month.Apr + p.homeRuns Month.May + p.homeRuns Month.Jul + p.homeRuns Month.Aug + p.homeRuns Month.Sep

theorem johnson_smith_tied_may :
  totalHomeRunsUpTo johnson Month.May = totalHomeRunsUpTo smith Month.May :=
by sorry

end johnson_smith_tied_may_l4032_403295


namespace arithmetic_geometric_sequence_l4032_403231

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_seq (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_seq a →
  geometric_seq (a 1) (a 2) (a 4) →
  a 2 = 4 := by
  sorry

end arithmetic_geometric_sequence_l4032_403231


namespace arithmetic_sequence_tenth_term_l4032_403257

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first three terms equals 3 -/
def sum_first_three (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 = 3

/-- The sum of the 5th, 6th, and 7th terms equals 9 -/
def sum_middle_three (a : ℕ → ℝ) : Prop :=
  a 5 + a 6 + a 7 = 9

theorem arithmetic_sequence_tenth_term (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : sum_first_three a) 
  (h3 : sum_middle_three a) : 
  a 10 = 5 := by
  sorry

end arithmetic_sequence_tenth_term_l4032_403257


namespace quadratic_roots_relation_l4032_403271

theorem quadratic_roots_relation (a b : ℝ) (p : ℝ) : 
  (3 * a^2 + 7 * a + 6 = 0) →
  (3 * b^2 + 7 * b + 6 = 0) →
  (a^3 + b^3 = -p) →
  (p = -35/27) := by
sorry

end quadratic_roots_relation_l4032_403271


namespace sets_problem_l4032_403288

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3)^2 ≤ 0}
def N : Set ℝ := {x | x^2 + x - 6 = 0}

-- Define set A as (complement_I M) ∩ N
def A : Set ℝ := (Set.univ \ M) ∩ N

-- Define set B
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 5 - a}

-- Theorem statement
theorem sets_problem :
  (A = {2}) ∧
  ({a : ℝ | B a ∪ A = A} = {a : ℝ | a ≥ 3}) := by
  sorry

end sets_problem_l4032_403288


namespace union_of_A_and_B_l4032_403244

def A : Set ℝ := {x | x - 1 > 0}
def B : Set ℝ := {x | x^2 - x - 2 > 0}

theorem union_of_A_and_B : A ∪ B = {x | x < -1 ∨ x > 1} := by sorry

end union_of_A_and_B_l4032_403244


namespace last_monkey_gets_255_l4032_403266

/-- Represents the process of monkeys dividing apples -/
def monkey_division (n : ℕ) : ℕ → ℕ
| 0 => n
| (k + 1) => 
  let remaining := monkey_division n k
  (remaining - 1) / 5

/-- The number of monkeys -/
def num_monkeys : ℕ := 5

/-- The minimum number of apples needed for the division process -/
def min_apples : ℕ := 5^5 - 4

/-- The amount the last monkey gets -/
def last_monkey_apples : ℕ := monkey_division min_apples (num_monkeys - 1)

theorem last_monkey_gets_255 : last_monkey_apples = 255 := by
  sorry

end last_monkey_gets_255_l4032_403266


namespace sale_price_markdown_l4032_403255

theorem sale_price_markdown (original_price : ℝ) (h1 : original_price > 0) : 
  let sale_price := 0.8 * original_price
  let final_price := 0.64 * original_price
  let markdown_percentage := (sale_price - final_price) / sale_price * 100
  markdown_percentage = 20 := by sorry

end sale_price_markdown_l4032_403255


namespace max_shapes_in_grid_l4032_403251

/-- The number of rows in the grid -/
def rows : Nat := 8

/-- The number of columns in the grid -/
def columns : Nat := 14

/-- The number of grid points occupied by each shape -/
def points_per_shape : Nat := 8

/-- The total number of grid points in the grid -/
def total_grid_points : Nat := (rows + 1) * (columns + 1)

/-- The maximum number of shapes that can be placed in the grid -/
def max_shapes : Nat := total_grid_points / points_per_shape

theorem max_shapes_in_grid :
  max_shapes = 16 := by sorry

end max_shapes_in_grid_l4032_403251


namespace max_surrounding_squares_l4032_403242

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Represents an arrangement of squares around a central square -/
structure SquareArrangement where
  centralSquare : Square
  surroundingSquare : Square
  numSurroundingSquares : ℕ

/-- The condition that the surrounding squares fit perfectly around the central square -/
def perfectFit (arrangement : SquareArrangement) : Prop :=
  arrangement.centralSquare.sideLength = arrangement.surroundingSquare.sideLength * (arrangement.numSurroundingSquares / 4 : ℝ)

/-- The theorem stating the maximum number of surrounding squares -/
theorem max_surrounding_squares (centralSquare : Square) (surroundingSquare : Square) 
    (h_central : centralSquare.sideLength = 4)
    (h_surrounding : surroundingSquare.sideLength = 1) :
    ∃ (arrangement : SquareArrangement), 
      arrangement.centralSquare = centralSquare ∧ 
      arrangement.surroundingSquare = surroundingSquare ∧
      arrangement.numSurroundingSquares = 16 ∧
      perfectFit arrangement ∧
      ∀ (otherArrangement : SquareArrangement), 
        otherArrangement.centralSquare = centralSquare → 
        otherArrangement.surroundingSquare = surroundingSquare → 
        perfectFit otherArrangement → 
        otherArrangement.numSurroundingSquares ≤ 16 :=
  sorry

end max_surrounding_squares_l4032_403242


namespace fencing_cost_is_5300_l4032_403239

/-- The cost of fencing per meter -/
def fencing_cost_per_meter : ℝ := 26.50

/-- The length of the rectangular plot in meters -/
def plot_length : ℝ := 57

/-- Calculate the breadth of the plot given the length -/
def plot_breadth : ℝ := plot_length - 14

/-- Calculate the perimeter of the rectangular plot -/
def plot_perimeter : ℝ := 2 * (plot_length + plot_breadth)

/-- Calculate the total cost of fencing the plot -/
def total_fencing_cost : ℝ := plot_perimeter * fencing_cost_per_meter

/-- Theorem stating that the total cost of fencing is 5300 currency units -/
theorem fencing_cost_is_5300 : total_fencing_cost = 5300 := by
  sorry

end fencing_cost_is_5300_l4032_403239


namespace all_pollywogs_gone_pollywogs_present_before_44_l4032_403281

/-- Represents the number of pollywogs in the pond after a given number of days -/
def pollywogs_remaining (days : ℕ) : ℕ :=
  if days ≤ 20 then
    2400 - 60 * days
  else
    2400 - 60 * 20 - 50 * (days - 20)

/-- The theorem states that after 44 days, no pollywogs remain in the pond -/
theorem all_pollywogs_gone : pollywogs_remaining 44 = 0 := by
  sorry

/-- The theorem states that before 44 days, there are still pollywogs in the pond -/
theorem pollywogs_present_before_44 (d : ℕ) (h : d < 44) : pollywogs_remaining d > 0 := by
  sorry

end all_pollywogs_gone_pollywogs_present_before_44_l4032_403281


namespace base6_to_base10_conversion_l4032_403247

-- Define the base 6 number as a list of digits
def base6_number : List Nat := [5, 4, 3, 2, 1]

-- Define the base of the number system
def base : Nat := 6

-- Function to convert a list of digits in base 6 to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

-- Theorem statement
theorem base6_to_base10_conversion :
  to_base_10 base6_number base = 7465 := by
  sorry

end base6_to_base10_conversion_l4032_403247


namespace unique_solution_condition_l4032_403207

-- Define the equation
def equation (k : ℝ) (x : ℝ) : Prop :=
  (x - 3) / (k * x + 2) = x

-- Define the condition for exactly one solution
def has_exactly_one_solution (k : ℝ) : Prop :=
  ∃! x : ℝ, equation k x

-- Theorem statement
theorem unique_solution_condition :
  ∀ k : ℝ, has_exactly_one_solution k ↔ k = -1/12 :=
sorry

end unique_solution_condition_l4032_403207


namespace tangent_line_implies_a_equals_one_l4032_403261

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a

theorem tangent_line_implies_a_equals_one (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = x₀ ∧ (deriv (f a)) x₀ = 1) → a = 1 := by
  sorry

end tangent_line_implies_a_equals_one_l4032_403261


namespace amit_left_after_three_days_l4032_403293

/-- The number of days Amit can complete the work alone -/
def amit_days : ℝ := 15

/-- The number of days Ananthu can complete the work alone -/
def ananthu_days : ℝ := 30

/-- The total number of days taken to complete the work -/
def total_days : ℝ := 27

/-- The number of days Amit worked before leaving -/
def amit_worked_days : ℝ := 3

theorem amit_left_after_three_days :
  amit_worked_days * (1 / amit_days) + (total_days - amit_worked_days) * (1 / ananthu_days) = 1 :=
sorry

end amit_left_after_three_days_l4032_403293


namespace special_sequence_2003_l4032_403277

/-- The sequence formed by removing multiples of 3 and 4 (except multiples of 5) from positive integers -/
def special_sequence : ℕ → ℕ := sorry

/-- The 2003rd term of the special sequence -/
def a_2003 : ℕ := special_sequence 2003

/-- Theorem stating that the 2003rd term of the special sequence is 3338 -/
theorem special_sequence_2003 : a_2003 = 3338 := by sorry

end special_sequence_2003_l4032_403277


namespace intersection_reciprocals_sum_l4032_403273

/-- Circle C with equation x^2 + y^2 + 2x - 3 = 0 -/
def CircleC (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 3 = 0

/-- Line l passing through the origin with slope k -/
def LineL (k x y : ℝ) : Prop := y = k * x

/-- Theorem: For any line passing through the origin and intersecting CircleC, 
    the sum of reciprocals of x-coordinates of intersection points is 2/3 -/
theorem intersection_reciprocals_sum (k : ℝ) (hk : k ≠ 0) : 
  ∃ x₁ x₂ y₁ y₂ : ℝ, 
    CircleC x₁ y₁ ∧ CircleC x₂ y₂ ∧ 
    LineL k x₁ y₁ ∧ LineL k x₂ y₂ ∧
    x₁ ≠ x₂ ∧ 
    1 / x₁ + 1 / x₂ = 2 / 3 := by
  sorry

end intersection_reciprocals_sum_l4032_403273


namespace probability_king_hearts_then_ace_in_standard_deck_l4032_403274

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of King of Hearts in a standard deck -/
def KingOfHearts : ℕ := 1

/-- Number of Aces in a standard deck -/
def Aces : ℕ := 4

/-- Probability of drawing King of Hearts first and any Ace second -/
def probability_king_hearts_then_ace (deck : ℕ) (king_of_hearts : ℕ) (aces : ℕ) : ℚ :=
  (king_of_hearts : ℚ) / deck * (aces : ℚ) / (deck - 1)

theorem probability_king_hearts_then_ace_in_standard_deck :
  probability_king_hearts_then_ace StandardDeck KingOfHearts Aces = 1 / 663 := by
  sorry

end probability_king_hearts_then_ace_in_standard_deck_l4032_403274


namespace initial_mixture_volume_l4032_403267

/-- Proof of initial mixture volume given ratio changes after water addition -/
theorem initial_mixture_volume
  (initial_milk : ℝ)
  (initial_water : ℝ)
  (added_water : ℝ)
  (h1 : initial_milk / initial_water = 4)
  (h2 : added_water = 23)
  (h3 : initial_milk / (initial_water + added_water) = 1.125)
  : initial_milk + initial_water = 45 := by
  sorry

end initial_mixture_volume_l4032_403267


namespace wall_length_given_mirror_area_l4032_403233

/-- Given a square mirror and a rectangular wall, prove the length of the wall
    when the mirror's area is half the wall's area. -/
theorem wall_length_given_mirror_area (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 24 →
  wall_width = 42 →
  (mirror_side ^ 2) * 2 = wall_width * (27.4285714 : ℝ) :=
by sorry

end wall_length_given_mirror_area_l4032_403233


namespace purely_imaginary_modulus_l4032_403287

theorem purely_imaginary_modulus (a : ℝ) :
  (a - 2 : ℂ) + a * I = (0 : ℂ) + (a * I) → Complex.abs (a + I) = Real.sqrt 5 := by
  sorry

end purely_imaginary_modulus_l4032_403287


namespace complement_A_intersect_B_not_equal_l4032_403263

def A : Set ℝ := {x : ℝ | |x - 2| ≤ 2}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

theorem complement_A_intersect_B_not_equal :
  (Aᶜ ∪ Bᶜ) ≠ Set.univ ∧
  (Aᶜ ∪ Bᶜ) ≠ {x : ℝ | x ≠ 0} ∧
  (Aᶜ ∪ Bᶜ) ≠ {0} :=
by sorry

end complement_A_intersect_B_not_equal_l4032_403263


namespace square_area_30cm_l4032_403229

/-- The area of a square with side length 30 centimeters is 900 square centimeters. -/
theorem square_area_30cm (s : ℝ) (h : s = 30) : s * s = 900 := by
  sorry

end square_area_30cm_l4032_403229


namespace circle_center_l4032_403225

/-- The center of the circle defined by x^2 + y^2 + 2y = 1 is (0, -1) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 + 2*y = 1) → (0, -1) = (0, -1) := by sorry

end circle_center_l4032_403225


namespace transformed_curve_is_circle_l4032_403203

-- Define the initial polar equation
def initial_polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * (Real.cos θ)^2 + 4 * (Real.sin θ)^2)

-- Define the scaling transformation
def scaling_transformation (x y x' y' : ℝ) : Prop :=
  x' = (1/2) * x ∧ y' = (Real.sqrt 3 / 3) * y

-- Theorem statement
theorem transformed_curve_is_circle :
  ∀ (x y x' y' : ℝ),
  (∃ (ρ θ : ℝ), initial_polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  scaling_transformation x y x' y' →
  ∃ (r : ℝ), x'^2 + y'^2 = r^2 :=
sorry

end transformed_curve_is_circle_l4032_403203


namespace sulfuric_acid_moles_l4032_403208

/-- Represents the chemical reaction Fe + H₂SO₄ → FeSO₄ + H₂ -/
structure ChemicalReaction where
  iron : ℝ
  sulfuricAcid : ℝ
  hydrogen : ℝ

/-- The stoichiometric relationship in the reaction -/
axiom stoichiometry (r : ChemicalReaction) : r.iron = r.sulfuricAcid ∧ r.iron = r.hydrogen

/-- The theorem to prove -/
theorem sulfuric_acid_moles (r : ChemicalReaction) 
  (h1 : r.iron = 2) 
  (h2 : r.hydrogen = 2) : 
  r.sulfuricAcid = 2 := by
  sorry

end sulfuric_acid_moles_l4032_403208


namespace units_sold_to_A_is_three_l4032_403221

/-- Represents the number of units sold to Customer A in a phone store scenario. -/
def units_sold_to_A (total_phones defective_phones units_sold_to_B units_sold_to_C : ℕ) : ℕ :=
  total_phones - defective_phones - units_sold_to_B - units_sold_to_C

/-- Theorem stating that given the specific conditions of the problem, 
    the number of units sold to Customer A is 3. -/
theorem units_sold_to_A_is_three :
  units_sold_to_A 20 5 5 7 = 3 := by
  sorry

end units_sold_to_A_is_three_l4032_403221


namespace square_fraction_is_perfect_square_l4032_403222

theorem square_fraction_is_perfect_square (a b k : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : k > 0) 
  (h4 : (a^2 + b^2 : ℕ) = k * (a * b + 1)) : 
  ∃ (n : ℕ), k = n^2 := by
  sorry

end square_fraction_is_perfect_square_l4032_403222


namespace exponent_division_l4032_403272

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^7 / a^3 = a^4 := by
  sorry

end exponent_division_l4032_403272


namespace number_of_bowls_l4032_403278

/-- Given a table with bowls of grapes, prove that there are 16 bowls when:
  - 8 grapes are added to each of 12 bowls
  - The average number of grapes in all bowls increases by 6
-/
theorem number_of_bowls : ℕ → Prop := λ n =>
  -- n is the number of bowls
  -- Define the increase in total grapes
  let total_increase : ℕ := 12 * 8
  -- Define the increase in average
  let avg_increase : ℕ := 6
  -- The theorem: if the total increase divided by the average increase equals n, 
  -- then n is the number of bowls
  total_increase / avg_increase = n

-- The proof (skipped with sorry)
example : number_of_bowls 16 := by sorry

end number_of_bowls_l4032_403278


namespace simplify_expression_l4032_403210

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) :
  ((m^2 - 3*m + 1) / m + 1) / ((m^2 - 1) / m) = (m - 1) / (m + 1) := by
  sorry

end simplify_expression_l4032_403210


namespace five_fourths_of_twelve_fifths_times_three_l4032_403270

theorem five_fourths_of_twelve_fifths_times_three (x : ℚ) : x = 12 / 5 → (5 / 4 * x) * 3 = 9 := by
  sorry

end five_fourths_of_twelve_fifths_times_three_l4032_403270


namespace new_year_markup_l4032_403230

theorem new_year_markup (initial_markup : ℝ) (discount : ℝ) (final_profit : ℝ) :
  initial_markup = 0.20 →
  discount = 0.07 →
  final_profit = 0.395 →
  ∃ (new_year_markup : ℝ),
    (1 + initial_markup) * (1 + new_year_markup) * (1 - discount) = 1 + final_profit ∧
    new_year_markup = 0.25 := by
  sorry

end new_year_markup_l4032_403230


namespace chess_tournament_l4032_403294

/-- Represents the number of participants from each city --/
structure Participants where
  moscow : ℕ
  saintPetersburg : ℕ
  kazan : ℕ

/-- Represents the number of games played between cities --/
structure Games where
  moscowSaintPetersburg : ℕ
  moscowKazan : ℕ
  saintPetersburgKazan : ℕ

/-- The theorem stating the conditions and the result to be proved --/
theorem chess_tournament (p : Participants) (g : Games) : 
  p.moscow * 9 = p.saintPetersburg * 6 ∧ 
  p.moscow * g.moscowKazan = p.kazan * 8 ∧ 
  p.saintPetersburg * 2 = p.kazan * 6 →
  g.moscowKazan = 4 := by
  sorry

end chess_tournament_l4032_403294


namespace fraction_simplification_l4032_403285

theorem fraction_simplification (x : ℝ) (h : x = 5) : 
  (x^4 + 12*x^2 + 36) / (x^2 + 6) = 31 := by
  sorry

end fraction_simplification_l4032_403285


namespace min_value_expression_l4032_403260

theorem min_value_expression (x y : ℝ) : 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 ≥ 3 := by
  sorry

end min_value_expression_l4032_403260


namespace P_in_fourth_quadrant_l4032_403211

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant. -/
def FourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point P. -/
def P : Point :=
  { x := 3, y := -2 }

/-- Theorem stating that P is in the fourth quadrant. -/
theorem P_in_fourth_quadrant : FourthQuadrant P := by
  sorry

end P_in_fourth_quadrant_l4032_403211


namespace largest_triangle_perimeter_l4032_403243

theorem largest_triangle_perimeter :
  ∀ x : ℕ,
  x > 0 →
  x < 7 + 9 →
  7 + x > 9 →
  9 + x > 7 →
  ∀ y : ℕ,
  y > 0 →
  y < 7 + 9 →
  7 + y > 9 →
  9 + y > 7 →
  7 + 9 + x ≥ 7 + 9 + y →
  7 + 9 + x = 31 :=
by
  sorry

end largest_triangle_perimeter_l4032_403243


namespace boat_speed_is_18_l4032_403254

/-- The speed of the boat in still water -/
def boat_speed : ℝ := 18

/-- The speed of the stream -/
def stream_speed : ℝ := 6

/-- Theorem stating that the boat speed in still water is 18 kmph -/
theorem boat_speed_is_18 :
  (∀ t : ℝ, t > 0 → 1 / (boat_speed - stream_speed) = 2 / (boat_speed + stream_speed)) →
  boat_speed = 18 :=
by
  sorry

end boat_speed_is_18_l4032_403254


namespace sally_quarters_l4032_403245

/-- The number of quarters Sally spent -/
def quarters_spent : ℕ := 418

/-- The number of quarters Sally has left -/
def quarters_left : ℕ := 342

/-- The initial number of quarters Sally had -/
def initial_quarters : ℕ := quarters_spent + quarters_left

theorem sally_quarters : initial_quarters = 760 := by
  sorry

end sally_quarters_l4032_403245


namespace fraction_comparison_l4032_403249

theorem fraction_comparison : (1 / (Real.sqrt 5 - 2)) < (1 / (Real.sqrt 6 - Real.sqrt 5)) := by
  sorry

end fraction_comparison_l4032_403249


namespace vector_subtraction_l4032_403227

def a : Fin 2 → ℝ := ![-1, 3]
def b : Fin 2 → ℝ := ![2, -1]

theorem vector_subtraction : a - 2 • b = ![-5, 5] := by sorry

end vector_subtraction_l4032_403227


namespace rent_increase_percentage_l4032_403228

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.25 * last_year_earnings
  let this_year_earnings := 1.45 * last_year_earnings
  let this_year_rent := 0.35 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 203 := by
sorry

end rent_increase_percentage_l4032_403228


namespace largest_satisfying_number_l4032_403253

/-- Returns the leading digit of a positive integer -/
def leadingDigit (n : ℕ) : ℕ :=
  if n < 10 then n else leadingDigit (n / 10)

/-- Returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Checks if a number satisfies the condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  n > 0 ∧ n = leadingDigit n * sumOfDigits n

theorem largest_satisfying_number :
  satisfiesCondition 48 ∧ ∀ m : ℕ, m > 48 → ¬satisfiesCondition m :=
sorry

end largest_satisfying_number_l4032_403253


namespace lucy_fish_count_l4032_403269

/-- The number of fish Lucy wants to buy -/
def fish_to_buy : ℕ := 68

/-- The total number of fish Lucy would have after buying more -/
def total_fish_after : ℕ := 280

/-- The current number of fish in Lucy's aquarium -/
def current_fish : ℕ := total_fish_after - fish_to_buy

theorem lucy_fish_count : current_fish = 212 := by
  sorry

end lucy_fish_count_l4032_403269


namespace flour_to_add_l4032_403205

theorem flour_to_add (recipe_amount : ℕ) (already_added : ℕ) (h1 : recipe_amount = 8) (h2 : already_added = 2) :
  recipe_amount - already_added = 6 := by
  sorry

end flour_to_add_l4032_403205


namespace multiply_125_3_2_25_solve_equation_l4032_403280

-- Part 1: Prove that 125 × 3.2 × 25 = 10000
theorem multiply_125_3_2_25 : 125 * 3.2 * 25 = 10000 := by sorry

-- Part 2: Prove that the solution to 24(x-12) = 16(x-4) is x = 28
theorem solve_equation : ∃ x : ℝ, 24 * (x - 12) = 16 * (x - 4) ∧ x = 28 := by sorry

end multiply_125_3_2_25_solve_equation_l4032_403280


namespace trailing_zeros_30_factorial_l4032_403209

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 30! is 7 -/
theorem trailing_zeros_30_factorial :
  trailingZeros 30 = 7 := by
  sorry

end trailing_zeros_30_factorial_l4032_403209


namespace shortest_minor_arc_line_l4032_403252

/-- The point M -/
def M : ℝ × ℝ := (1, -2)

/-- The circle C -/
def C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

/-- A line passing through a point -/
def LineThrough (m : ℝ × ℝ) (a b c : ℝ) : Prop :=
  a * m.1 + b * m.2 + c = 0

/-- The theorem stating the equation of the line that divides the circle into two arcs with the shortest minor arc -/
theorem shortest_minor_arc_line :
  ∃ (a b c : ℝ), LineThrough M a b c ∧
  (∀ (x y : ℝ), C x y → (a * x + b * y + c = 0 → 
    ∀ (a' b' c' : ℝ), LineThrough M a' b' c' → 
      (∃ (x' y' : ℝ), C x' y' ∧ a' * x' + b' * y' + c' = 0) → 
        (∃ (x'' y'' : ℝ), C x'' y'' ∧ a * x'' + b * y'' + c = 0 ∧ 
          ∀ (x''' y''' : ℝ), C x''' y''' ∧ a' * x''' + b' * y''' + c' = 0 → 
            (x'' - M.1)^2 + (y'' - M.2)^2 ≤ (x''' - M.1)^2 + (y''' - M.2)^2))) ∧
  a = 1 ∧ b = 2 ∧ c = 3 :=
sorry

end shortest_minor_arc_line_l4032_403252


namespace sqrt_625_div_5_l4032_403268

theorem sqrt_625_div_5 : Real.sqrt 625 / 5 = 5 := by
  sorry

end sqrt_625_div_5_l4032_403268


namespace ln_inequality_solution_set_l4032_403213

theorem ln_inequality_solution_set (x : ℝ) : 
  Real.log (x^2 - 2*x - 2) > 0 ↔ x > 3 ∨ x < -1 := by
  sorry

end ln_inequality_solution_set_l4032_403213


namespace instantaneous_velocity_at_2s_l4032_403214

-- Define the displacement function
def s (t : ℝ) : ℝ := 3 * t^3 - 2 * t^2 + t + 1

-- Define the velocity function as the derivative of the displacement function
def v (t : ℝ) : ℝ := 9 * t^2 - 4 * t + 1

-- Theorem statement
theorem instantaneous_velocity_at_2s : v 2 = 29 := by
  sorry

end instantaneous_velocity_at_2s_l4032_403214


namespace system_four_solutions_l4032_403220

theorem system_four_solutions (a : ℝ) (ha : a > 0) :
  ∃! (solutions : Finset (ℝ × ℝ)), 
    solutions.card = 4 ∧
    ∀ (x y : ℝ), (x, y) ∈ solutions ↔ 
      (y = a * x^2 ∧ y^2 + 3 = x^2 + 4*y) :=
sorry

end system_four_solutions_l4032_403220


namespace line_equation_l4032_403223

/-- A line passing through a point and intersecting axes -/
structure Line where
  -- Point that the line passes through
  P : ℝ × ℝ
  -- x-coordinate of intersection with positive x-axis
  C : ℝ
  -- y-coordinate of intersection with negative y-axis
  D : ℝ
  -- Condition that P lies on the line
  point_on_line : (P.1 / C) + (P.2 / (-D)) = 1
  -- Condition for positive x-axis intersection
  pos_x_axis : C > 0
  -- Condition for negative y-axis intersection
  neg_y_axis : D > 0
  -- Area condition
  area_condition : (1/2) * C * D = 2

/-- Theorem stating the equation of the line -/
theorem line_equation (l : Line) (h : l.P = (1, -1)) :
  ∃ (a b : ℝ), a * l.P.1 + b * l.P.2 + 2 = 0 ∧ 
               ∀ (x y : ℝ), a * x + b * y + 2 = 0 ↔ (x / l.C) + (y / (-l.D)) = 1 := by
  sorry

end line_equation_l4032_403223


namespace normal_distribution_probability_l4032_403284

variables (μ σ : ℝ) (ξ : ℝ → ℝ)

-- Define the normal distribution
def normal_dist (μ σ : ℝ) (ξ : ℝ → ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, f x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ)^2) / (2 * σ^2))

-- Define the probability function
noncomputable def P (A : Set ℝ) : ℝ := sorry

theorem normal_distribution_probability 
  (h1 : normal_dist μ σ ξ)
  (h2 : P {x | ξ x < -1} = 0.3)
  (h3 : P {x | ξ x > 2} = 0.3) :
  P {x | ξ x < 2*μ + 1} = 0.7 := by sorry

end normal_distribution_probability_l4032_403284


namespace triangle_area_proof_l4032_403292

theorem triangle_area_proof (a b : ℝ × ℝ) : 
  a = (2, -3) → b = (4, -1) → 
  abs (a.1 * b.2 - a.2 * b.1) / 2 = 5 := by sorry

end triangle_area_proof_l4032_403292


namespace ninas_run_l4032_403296

theorem ninas_run (x : ℝ) : x + x + 0.67 = 0.83 → x = 0.08 := by
  sorry

end ninas_run_l4032_403296


namespace johns_shower_duration_johns_shower_theorem_l4032_403200

theorem johns_shower_duration (shower_duration : ℕ) (shower_frequency : ℕ) 
  (water_usage_rate : ℕ) (total_water_usage : ℕ) : ℕ :=
  let water_per_shower := shower_duration * water_usage_rate
  let num_showers := total_water_usage / water_per_shower
  let num_days := num_showers * shower_frequency
  let num_weeks := num_days / 7
  num_weeks

theorem johns_shower_theorem : 
  johns_shower_duration 10 2 2 280 = 4 := by
  sorry

end johns_shower_duration_johns_shower_theorem_l4032_403200


namespace total_cash_realized_proof_l4032_403259

/-- Represents a stock with its value and brokerage rate -/
structure Stock where
  value : ℝ
  brokerage_rate : ℝ

/-- Calculates the cash realized for a single stock after brokerage -/
def cash_realized_single (stock : Stock) : ℝ :=
  stock.value * (1 - stock.brokerage_rate)

/-- Calculates the total cash realized for multiple stocks -/
def total_cash_realized (stocks : List Stock) : ℝ :=
  stocks.map cash_realized_single |>.sum

/-- Theorem stating that the total cash realized for the given stocks is 637.818125 -/
theorem total_cash_realized_proof (stockA stockB stockC : Stock)
  (hA : stockA = { value := 120.50, brokerage_rate := 0.0025 })
  (hB : stockB = { value := 210.75, brokerage_rate := 0.005 })
  (hC : stockC = { value := 310.25, brokerage_rate := 0.0075 }) :
  total_cash_realized [stockA, stockB, stockC] = 637.818125 := by
  sorry

end total_cash_realized_proof_l4032_403259


namespace abc_unique_solution_l4032_403297

/-- Represents a base-7 number with two digits --/
def Base7TwoDigit (a b : ℕ) : ℕ := 7 * a + b

/-- Converts a three-digit decimal number to its numeric value --/
def ThreeDigitToNum (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem abc_unique_solution :
  ∀ A B C : ℕ,
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A < 7 → B < 7 → C < 7 →
    A ≠ B → B ≠ C → A ≠ C →
    Base7TwoDigit A B + C = Base7TwoDigit C 0 →
    Base7TwoDigit A B + Base7TwoDigit B A = Base7TwoDigit B 6 →
    ThreeDigitToNum A B C = 425 :=
by sorry

end abc_unique_solution_l4032_403297


namespace smallest_divisible_by_1_to_10_l4032_403235

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end smallest_divisible_by_1_to_10_l4032_403235
