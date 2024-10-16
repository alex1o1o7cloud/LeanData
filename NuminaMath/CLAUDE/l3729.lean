import Mathlib

namespace NUMINAMATH_CALUDE_school_construction_problem_l3729_372979

/-- School construction problem -/
theorem school_construction_problem
  (total_area : ℝ)
  (demolition_cost : ℝ)
  (construction_cost : ℝ)
  (actual_demolition_ratio : ℝ)
  (actual_construction_ratio : ℝ)
  (greening_cost : ℝ)
  (h1 : total_area = 7200)
  (h2 : demolition_cost = 80)
  (h3 : construction_cost = 700)
  (h4 : actual_demolition_ratio = 1.1)
  (h5 : actual_construction_ratio = 0.8)
  (h6 : greening_cost = 200) :
  ∃ (planned_demolition planned_construction greening_area : ℝ),
    planned_demolition + planned_construction = total_area ∧
    actual_demolition_ratio * planned_demolition + actual_construction_ratio * planned_construction = total_area ∧
    planned_demolition = 4800 ∧
    planned_construction = 2400 ∧
    greening_area = 1488 ∧
    greening_area * greening_cost = 
      (planned_demolition * demolition_cost + planned_construction * construction_cost) -
      (actual_demolition_ratio * planned_demolition * demolition_cost + 
       actual_construction_ratio * planned_construction * construction_cost) :=
by sorry

end NUMINAMATH_CALUDE_school_construction_problem_l3729_372979


namespace NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l3729_372982

-- Define the probabilities
def prob_east_wind : ℚ := 3/10
def prob_rain : ℚ := 11/30
def prob_both : ℚ := 4/15

-- State the theorem
theorem conditional_probability_rain_given_east_wind :
  (prob_both / prob_east_wind : ℚ) = 8/9 := by
sorry

end NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l3729_372982


namespace NUMINAMATH_CALUDE_probability_all_genuine_given_equal_weight_l3729_372969

/-- Represents the total number of coins -/
def total_coins : ℕ := 12

/-- Represents the number of genuine coins -/
def genuine_coins : ℕ := 9

/-- Represents the number of counterfeit coins -/
def counterfeit_coins : ℕ := 3

/-- Event A: All 4 selected coins are genuine -/
def event_A : Set (Fin total_coins × Fin total_coins × Fin total_coins × Fin total_coins) :=
  sorry

/-- Event B: The combined weight of the first pair equals the combined weight of the second pair -/
def event_B : Set (Fin total_coins × Fin total_coins × Fin total_coins × Fin total_coins) :=
  sorry

/-- The probability measure on the sample space -/
def P : Set (Fin total_coins × Fin total_coins × Fin total_coins × Fin total_coins) → ℚ :=
  sorry

/-- Theorem stating the conditional probability of A given B -/
theorem probability_all_genuine_given_equal_weight :
    P (event_A ∩ event_B) / P event_B = 84 / 113 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_genuine_given_equal_weight_l3729_372969


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3729_372960

theorem scientific_notation_equivalence :
  686530000 = 6.8653 * (10 ^ 8) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3729_372960


namespace NUMINAMATH_CALUDE_complex_modulus_product_l3729_372918

theorem complex_modulus_product : Complex.abs (5 - 3*I) * Complex.abs (5 + 3*I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l3729_372918


namespace NUMINAMATH_CALUDE_polynomial_characterization_l3729_372937

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that must be satisfied by a, b, and c -/
def SatisfiesCondition (a b c : ℝ) : Prop :=
  a * b + b * c + c * a = 0

/-- The equation that P must satisfy for all a, b, c satisfying the condition -/
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ (a b c : ℝ), SatisfiesCondition a b c →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

/-- The form of the polynomial we're trying to prove -/
def IsQuarticQuadratic (P : RealPolynomial) : Prop :=
  ∃ (α β : ℝ), ∀ x, P x = α * x^4 + β * x^2

theorem polynomial_characterization (P : RealPolynomial) :
  SatisfiesEquation P → IsQuarticQuadratic P :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l3729_372937


namespace NUMINAMATH_CALUDE_irreducible_fraction_l3729_372933

theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l3729_372933


namespace NUMINAMATH_CALUDE_no_integer_solution_l3729_372984

theorem no_integer_solution : ¬ ∃ (x y z : ℤ), (x - y)^3 + (y - z)^3 + (z - x)^3 = 2021 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3729_372984


namespace NUMINAMATH_CALUDE_smallest_representable_numbers_l3729_372915

def is_representable (c : ℕ) : Prop :=
  ∃ m n : ℕ, c = 7 * m^2 - 11 * n^2

theorem smallest_representable_numbers :
  (is_representable 1 ∧ is_representable 5) ∧
  (∀ c : ℕ, c < 1 → ¬is_representable c) ∧
  (∀ c : ℕ, 1 < c → c < 5 → ¬is_representable c) :=
sorry

end NUMINAMATH_CALUDE_smallest_representable_numbers_l3729_372915


namespace NUMINAMATH_CALUDE_problem_statement_l3729_372999

theorem problem_statement : (-0.125)^2003 * (-8)^2004 = -8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3729_372999


namespace NUMINAMATH_CALUDE_distance_problems_l3729_372994

def distance_point_to_line (p : Fin n → ℝ) (a b : Fin n → ℝ) : ℝ :=
  sorry

theorem distance_problems :
  let d1 := distance_point_to_line (![1, 0]) (![0, 0]) (![0, 1])
  let d2 := distance_point_to_line (![1, 0]) (![0, 0]) (![1, 1])
  let d3 := distance_point_to_line (![1, 0, 0]) (![0, 0, 0]) (![1, 1, 1])
  (d1 = 1) ∧ (d2 = Real.sqrt 2 / 2) ∧ (d3 = Real.sqrt 6 / 3) := by
  sorry

end NUMINAMATH_CALUDE_distance_problems_l3729_372994


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3729_372958

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (F P Q : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  F = (c, 0) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x < -a ∨ x > a)) →
  (∀ (x y : ℝ), x^2 + y^2 = b^2 / 4 → ((P.1 - x) * (F.1 - P.1) + (P.2 - y) * (F.2 - P.2) = 0)) →
  Q.1^2 + Q.2^2 = b^2 / 4 →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = ((P.1 - F.1)^2 + (P.2 - F.2)^2) / 4 →
  c^2 / a^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3729_372958


namespace NUMINAMATH_CALUDE_cricket_results_l3729_372970

/-- Represents the cricket matches data -/
structure CricketData where
  matches1 : ℕ
  average1 : ℕ
  matches2 : ℕ
  average2 : ℕ

/-- Calculates the total number of matches played -/
def totalMatches (data : CricketData) : ℕ :=
  data.matches1 + data.matches2

/-- Calculates the total runs scored in all matches -/
def totalRuns (data : CricketData) : ℕ :=
  data.matches1 * data.average1 + data.matches2 * data.average2

/-- Calculates the average score across all matches -/
def overallAverage (data : CricketData) : ℚ :=
  (totalRuns data : ℚ) / (totalMatches data : ℚ)

/-- Theorem stating the results for the given cricket data -/
theorem cricket_results (data : CricketData) 
  (h1 : data.matches1 = 2) (h2 : data.average1 = 27)
  (h3 : data.matches2 = 3) (h4 : data.average2 = 32) :
  totalMatches data = 5 ∧ overallAverage data = 30 := by
  sorry

#eval totalMatches { matches1 := 2, average1 := 27, matches2 := 3, average2 := 32 }
#eval overallAverage { matches1 := 2, average1 := 27, matches2 := 3, average2 := 32 }

end NUMINAMATH_CALUDE_cricket_results_l3729_372970


namespace NUMINAMATH_CALUDE_k_range_for_negative_sum_l3729_372938

/-- A power function that passes through the point (3, 27) -/
def f (x : ℝ) : ℝ := x^3

/-- The theorem stating the range of k for which f(k^2 + 3) + f(9 - 8k) < 0 holds -/
theorem k_range_for_negative_sum (k : ℝ) :
  f (k^2 + 3) + f (9 - 8*k) < 0 ↔ 2 < k ∧ k < 6 := by
  sorry


end NUMINAMATH_CALUDE_k_range_for_negative_sum_l3729_372938


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l3729_372975

-- Define a line type
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

-- Define a line passing through a point
def passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

-- Define the given line
def given_line : Line := { slope := 2, y_intercept := 4 }

-- Define the point that line b passes through
def given_point : Point := { x := 3, y := 7 }

-- Theorem statement
theorem y_intercept_of_parallel_line :
  ∃ (b : Line),
    parallel b given_line ∧
    passes_through b given_point ∧
    b.y_intercept = 1 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l3729_372975


namespace NUMINAMATH_CALUDE_perpendicular_lines_imply_a_equals_one_l3729_372941

-- Define the lines
def line1 (a x y : ℝ) : Prop := (3*a + 2)*x - 3*y + 8 = 0
def line2 (a x y : ℝ) : Prop := 3*x + (a + 4)*y - 7 = 0

-- Define perpendicularity condition
def perpendicular (a : ℝ) : Prop := 
  (3*a + 2) * 3 + (-3) * (a + 4) = 0

-- Theorem statement
theorem perpendicular_lines_imply_a_equals_one :
  ∀ a : ℝ, perpendicular a → a = 1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_imply_a_equals_one_l3729_372941


namespace NUMINAMATH_CALUDE_b_range_l3729_372974

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x

-- Define the derivative of f(x)
def f_derivative (b : ℝ) (x : ℝ) : ℝ := 3*x^2 + b

-- Theorem statement
theorem b_range (b : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, (f_derivative b x) ≤ 0) →
  b ∈ Set.Iic (-3) :=
by sorry

end NUMINAMATH_CALUDE_b_range_l3729_372974


namespace NUMINAMATH_CALUDE_arithmetic_harmonic_geometric_proportion_l3729_372923

theorem arithmetic_harmonic_geometric_proportion (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / ((a + b) / 2) = (2 * a * b / (a + b)) / b := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_harmonic_geometric_proportion_l3729_372923


namespace NUMINAMATH_CALUDE_p_and_not_q_l3729_372991

-- Define proposition p
def p : Prop := ∀ a : ℝ, a > 1 → a^2 > a

-- Define proposition q
def q : Prop := ∀ a : ℝ, a > 0 → a > 1/a

-- Theorem to prove
theorem p_and_not_q : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_and_not_q_l3729_372991


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3729_372967

theorem reciprocal_of_negative_fraction (n : ℤ) (hn : n ≠ 0) :
  (-(1 : ℚ) / n)⁻¹ = -n :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3729_372967


namespace NUMINAMATH_CALUDE_cubic_root_series_sum_l3729_372911

/-- Given a positive real number s satisfying s³ + (1/4)s - 1 = 0,
    the series s² + 2s⁵ + 3s⁸ + 4s¹¹ + ... converges to 16 -/
theorem cubic_root_series_sum (s : ℝ) (hs : 0 < s) (heq : s^3 + (1/4) * s - 1 = 0) :
  ∑' n, (n + 1) * s^(3*n + 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_series_sum_l3729_372911


namespace NUMINAMATH_CALUDE_perpendicular_line_m_value_l3729_372996

/-- Given a line passing through points (m, 3) and (1, m) that is perpendicular
    to a line with slope -1, prove that m = 2. -/
theorem perpendicular_line_m_value (m : ℝ) : 
  (((m - 3) / (1 - m) = 1) ∧ (1 * (-1) = -1)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_m_value_l3729_372996


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3729_372963

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 4) :
  (2/x + 3/y) ≥ 25/4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 4 ∧ 2/x₀ + 3/y₀ = 25/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3729_372963


namespace NUMINAMATH_CALUDE_right_triangle_area_l3729_372944

/-- A right triangle with hypotenuse 13 and one leg 5 has an area of 30 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 5) 
  (h3 : c^2 = a^2 - b^2) (h4 : a > 0 ∧ b > 0 ∧ c > 0) : (1/2) * b * c = 30 := by
  sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_right_triangle_area_l3729_372944


namespace NUMINAMATH_CALUDE_train_speed_l3729_372926

/-- The speed of a train given its passing times and platform length -/
theorem train_speed (platform_length : ℝ) (platform_time : ℝ) (man_time : ℝ) :
  platform_length = 360.0288 →
  platform_time = 44 →
  man_time = 20 →
  ∃ speed : ℝ, abs (speed - 54.00432) < 0.00001 ∧ 
    speed * man_time = speed * platform_time - platform_length :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l3729_372926


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3729_372908

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k ∣ n) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ ¬(k ∣ m)) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3729_372908


namespace NUMINAMATH_CALUDE_parrots_left_on_branch_l3729_372916

/-- Represents the number of birds on a tree branch -/
structure BirdCount where
  parrots : ℕ
  crows : ℕ

/-- The initial state of birds on the branch -/
def initialState : BirdCount where
  parrots := 7
  crows := 13 - 7

/-- The number of birds that flew away -/
def flownAway : ℕ :=
  initialState.crows - 1

/-- The final state of birds on the branch -/
def finalState : BirdCount where
  parrots := initialState.parrots - flownAway
  crows := 1

theorem parrots_left_on_branch :
  finalState.parrots = 2 :=
sorry

end NUMINAMATH_CALUDE_parrots_left_on_branch_l3729_372916


namespace NUMINAMATH_CALUDE_apple_difference_is_two_l3729_372987

/-- The number of apples Jackie has -/
def jackies_apples : ℕ := 10

/-- The number of apples Adam has -/
def adams_apples : ℕ := 8

/-- The difference in apples between Jackie and Adam -/
def apple_difference : ℕ := jackies_apples - adams_apples

theorem apple_difference_is_two : apple_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_is_two_l3729_372987


namespace NUMINAMATH_CALUDE_union_and_subset_l3729_372914

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x < 1 + 3*m}

-- Define the complement of A
def A_complement : Set ℝ := {x | x ≤ -1 ∨ x > 3}

theorem union_and_subset :
  (∀ m : ℝ, m = 1 → A ∪ B m = {x | -1 < x ∧ x < 4}) ∧
  (∀ m : ℝ, B m ⊆ A_complement ↔ m ≤ -1/2 ∨ m > 3) :=
sorry

end NUMINAMATH_CALUDE_union_and_subset_l3729_372914


namespace NUMINAMATH_CALUDE_draw_balls_theorem_l3729_372920

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def red_score : ℕ := 2
def white_score : ℕ := 1
def balls_to_draw : ℕ := 4
def min_score : ℕ := 5

/-- The number of ways to draw 4 balls from a bag containing 4 red balls and 6 white balls,
    where red balls score 2 points and white balls score 1 point,
    such that the total score is not less than 5 points. -/
def ways_to_draw : ℕ := 195

theorem draw_balls_theorem :
  ways_to_draw = 195 :=
sorry

end NUMINAMATH_CALUDE_draw_balls_theorem_l3729_372920


namespace NUMINAMATH_CALUDE_lecture_average_minutes_heard_l3729_372992

/-- Calculates the average number of minutes heard in a lecture --/
theorem lecture_average_minutes_heard 
  (total_duration : ℝ) 
  (total_attendees : ℕ) 
  (full_lecture_percent : ℝ) 
  (missed_lecture_percent : ℝ) 
  (half_lecture_percent : ℝ) 
  (h1 : total_duration = 90)
  (h2 : total_attendees = 200)
  (h3 : full_lecture_percent = 0.3)
  (h4 : missed_lecture_percent = 0.2)
  (h5 : half_lecture_percent = 0.4 * (1 - full_lecture_percent - missed_lecture_percent))
  (h6 : full_lecture_percent + missed_lecture_percent + half_lecture_percent + 
        (1 - full_lecture_percent - missed_lecture_percent - half_lecture_percent) = 1) :
  (full_lecture_percent * total_duration * total_attendees + 
   0 * missed_lecture_percent * total_attendees + 
   (total_duration / 2) * half_lecture_percent * total_attendees + 
   (3 * total_duration / 4) * (1 - full_lecture_percent - missed_lecture_percent - half_lecture_percent) * total_attendees) / 
   total_attendees = 56.25 := by
sorry

end NUMINAMATH_CALUDE_lecture_average_minutes_heard_l3729_372992


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_800_l3729_372935

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_800 :
  units_digit (factorial_sum 800) = 3 := by
sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_800_l3729_372935


namespace NUMINAMATH_CALUDE_derivative_sin_plus_cos_at_pi_l3729_372990

/-- Given f(x) = sin(x) + cos(x), prove that f'(π) = -1 -/
theorem derivative_sin_plus_cos_at_pi :
  let f := λ x : ℝ => Real.sin x + Real.cos x
  (deriv f) π = -1 := by sorry

end NUMINAMATH_CALUDE_derivative_sin_plus_cos_at_pi_l3729_372990


namespace NUMINAMATH_CALUDE_tips_fraction_l3729_372956

/-- Represents the income structure of a waiter -/
structure WaiterIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the fraction of income from tips -/
def fractionFromTips (income : WaiterIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

/-- Theorem: Given the conditions, the fraction of income from tips is 9/13 -/
theorem tips_fraction (income : WaiterIncome) 
  (h : income.tips = (9 / 4) * income.salary) : 
  fractionFromTips income = 9 / 13 := by
  sorry

#check tips_fraction

end NUMINAMATH_CALUDE_tips_fraction_l3729_372956


namespace NUMINAMATH_CALUDE_skyline_hospital_quadruplets_l3729_372976

theorem skyline_hospital_quadruplets :
  ∀ (twins triplets quads : ℕ),
    triplets = 5 * quads →
    twins = 3 * triplets →
    2 * twins + 3 * triplets + 4 * quads = 1200 →
    4 * quads = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_skyline_hospital_quadruplets_l3729_372976


namespace NUMINAMATH_CALUDE_perimeter_of_square_arrangement_l3729_372906

theorem perimeter_of_square_arrangement (total_area : ℝ) (num_squares : ℕ) 
  (arrangement_width : ℕ) (arrangement_height : ℕ) :
  total_area = 216 →
  num_squares = 6 →
  arrangement_width = 3 →
  arrangement_height = 2 →
  let square_area := total_area / num_squares
  let side_length := Real.sqrt square_area
  let perimeter := 2 * (arrangement_width + arrangement_height) * side_length
  perimeter = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_square_arrangement_l3729_372906


namespace NUMINAMATH_CALUDE_operation_result_l3729_372901

def universal_set : Set ℝ := Set.univ

def operation (M N : Set ℝ) : Set ℝ := M ∩ (universal_set \ N)

def set_M : Set ℝ := {x : ℝ | |x| ≤ 2}

def set_N : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}

theorem operation_result :
  operation set_M set_N = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_operation_result_l3729_372901


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l3729_372998

/-- The number of unique arrangements of n distinct beads on a bracelet, 
    considering only rotational symmetry -/
def bracelet_arrangements (n : ℕ) : ℕ := Nat.factorial n / n

/-- Theorem: The number of unique arrangements of 8 distinct beads on a bracelet, 
    considering only rotational symmetry, is 5040 -/
theorem eight_bead_bracelet_arrangements : 
  bracelet_arrangements 8 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l3729_372998


namespace NUMINAMATH_CALUDE_cube_root_of_negative_27_l3729_372907

theorem cube_root_of_negative_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_27_l3729_372907


namespace NUMINAMATH_CALUDE_three_digit_number_count_l3729_372985

/-- A three-digit number where the hundreds digit equals the units digit -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds = units ∧ hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9

/-- The value of a ThreeDigitNumber -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Predicate for divisibility by 4 -/
def divisible_by_four (n : Nat) : Prop :=
  n % 4 = 0

theorem three_digit_number_count :
  (∃ (s : Finset ThreeDigitNumber), s.card = 90) ∧
  (∃ (s : Finset ThreeDigitNumber), s.card = 20 ∧ ∀ n ∈ s, divisible_by_four n.value) :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_count_l3729_372985


namespace NUMINAMATH_CALUDE_trip_length_proof_average_efficiency_proof_l3729_372988

/-- The total length of the trip in miles -/
def trip_length : ℝ := 180

/-- The distance the car ran on battery -/
def battery_distance : ℝ := 60

/-- The rate of gasoline consumption in gallons per mile -/
def gasoline_rate : ℝ := 0.03

/-- The average fuel efficiency for the entire trip in miles per gallon -/
def average_efficiency : ℝ := 50

/-- Theorem stating that the trip length satisfies the given conditions -/
theorem trip_length_proof :
  trip_length = battery_distance + 
  (trip_length - battery_distance) * gasoline_rate * average_efficiency :=
by sorry

/-- Theorem stating that the average efficiency is correct -/
theorem average_efficiency_proof :
  average_efficiency = trip_length / (gasoline_rate * (trip_length - battery_distance)) :=
by sorry

end NUMINAMATH_CALUDE_trip_length_proof_average_efficiency_proof_l3729_372988


namespace NUMINAMATH_CALUDE_first_few_terms_eighth_term_l3729_372980

/-- Definition of the sequence -/
def a (n : ℕ) : ℕ := n^2 + 2*n - 1

/-- The first few terms of the sequence -/
theorem first_few_terms :
  a 1 = 2 ∧ a 2 = 7 ∧ a 3 = 14 ∧ a 4 = 23 := by sorry

/-- The 8th term of the sequence is 79 -/
theorem eighth_term : a 8 = 79 := by sorry

end NUMINAMATH_CALUDE_first_few_terms_eighth_term_l3729_372980


namespace NUMINAMATH_CALUDE_triangle_inequality_l3729_372955

/-- For any triangle ABC with side lengths a, b, c, circumradius R, and inradius r,
    the inequality (b² + c²) / (2bc) ≤ R / (2r) holds. -/
theorem triangle_inequality (a b c R r : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (hR : 0 < R) (hr : 0 < r) (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) :
    (b^2 + c^2) / (2 * b * c) ≤ R / (2 * r) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3729_372955


namespace NUMINAMATH_CALUDE_polynomial_factorization_sum_l3729_372995

theorem polynomial_factorization_sum (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^8 - 3*x^6 + 3*x^4 - x^2 + 2 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + 2*b₃*x + c₃)) : 
  b₁*c₁ + b₂*c₂ + 2*b₃*c₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_sum_l3729_372995


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l3729_372961

theorem square_perimeters_sum (x y : ℝ) 
  (h1 : x^2 + y^2 = 113) 
  (h2 : x^2 - y^2 = 47) 
  (h3 : x ≥ y) : 
  3 * (4 * x) + 4 * y = 48 * Real.sqrt 5 + 4 * Real.sqrt 33 := by
sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l3729_372961


namespace NUMINAMATH_CALUDE_magnitude_of_2_plus_3i_l3729_372917

theorem magnitude_of_2_plus_3i :
  Complex.abs (2 + 3 * Complex.I) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_2_plus_3i_l3729_372917


namespace NUMINAMATH_CALUDE_chord_intersection_sum_l3729_372983

-- Define the sphere and point S
variable (sphere : Type) (S : sphere)

-- Define the chords
variable (A A' B B' C C' : sphere)

-- Define the lengths
variable (AS BS CS : ℝ)

-- Define the volume ratio
variable (volume_ratio : ℝ)

-- State the theorem
theorem chord_intersection_sum (h1 : AS = 6) (h2 : BS = 3) (h3 : CS = 2)
  (h4 : volume_ratio = 2/9) :
  ∃ (SA' SB' SC' : ℝ), SA' + SB' + SC' = 18 := by
  sorry

end NUMINAMATH_CALUDE_chord_intersection_sum_l3729_372983


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3729_372909

theorem product_of_three_numbers (a b c : ℚ) : 
  a + b + c = 30 →
  a = 3 * (b + c) →
  b = 6 * c →
  a * b * c = 675 / 28 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3729_372909


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3729_372910

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3729_372910


namespace NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3729_372945

/-- Represents the price reduction in yuan -/
def price_reduction : ℕ := 10

/-- Cost to purchase each piece of clothing -/
def purchase_cost : ℕ := 45

/-- Original selling price of each piece of clothing -/
def original_price : ℕ := 65

/-- Original daily sales quantity -/
def original_sales : ℕ := 30

/-- Additional sales for each yuan of price reduction -/
def sales_increase_rate : ℕ := 5

/-- Target daily profit -/
def target_profit : ℕ := 800

/-- Theorem stating that the given price reduction achieves the target profit -/
theorem price_reduction_achieves_target_profit :
  (original_price - price_reduction - purchase_cost) *
  (original_sales + sales_increase_rate * price_reduction) = target_profit :=
sorry

end NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3729_372945


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3729_372934

theorem contrapositive_equivalence :
  (∀ x : ℝ, x ≤ -3 → x < 0) ↔ (∀ x : ℝ, x ≥ 0 → x > -3) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3729_372934


namespace NUMINAMATH_CALUDE_quadrilateral_numbers_multiple_of_14_l3729_372957

def quadrilateral_number (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

def is_multiple_of_14 (n : ℕ) : Prop := ∃ k : ℕ, n = 14 * k

theorem quadrilateral_numbers_multiple_of_14 (t : ℤ) :
  (∀ n : ℤ, (n = 28 * t ∨ n = 28 * t + 6 ∨ n = 28 * t + 7 ∨ n = 28 * t + 12 ∨ 
             n = 28 * t + 14 ∨ n = 28 * t - 9 ∨ n = 28 * t - 8 ∨ n = 28 * t - 2 ∨ 
             n = 28 * t - 1) → 
    is_multiple_of_14 (quadrilateral_number n.toNat)) ∧
  (∀ n : ℕ, is_multiple_of_14 (quadrilateral_number n) → 
    ∃ t : ℤ, n = (28 * t).toNat ∨ n = (28 * t + 6).toNat ∨ n = (28 * t + 7).toNat ∨ 
              n = (28 * t + 12).toNat ∨ n = (28 * t + 14).toNat ∨ n = (28 * t - 9).toNat ∨ 
              n = (28 * t - 8).toNat ∨ n = (28 * t - 2).toNat ∨ n = (28 * t - 1).toNat) :=
by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_numbers_multiple_of_14_l3729_372957


namespace NUMINAMATH_CALUDE_office_paper_sheets_per_pack_l3729_372930

/-- The number of sheets in each pack of printer paper -/
def sheets_per_pack (total_packs : ℕ) (documents_per_day : ℕ) (days_lasted : ℕ) : ℕ :=
  (documents_per_day * days_lasted) / total_packs

/-- Theorem stating the number of sheets in each pack of printer paper -/
theorem office_paper_sheets_per_pack :
  sheets_per_pack 2 80 6 = 240 := by
  sorry

end NUMINAMATH_CALUDE_office_paper_sheets_per_pack_l3729_372930


namespace NUMINAMATH_CALUDE_right_triangle_area_l3729_372940

/-- Given a right-angled triangle with height 5 cm and median to hypotenuse 6 cm, its area is 30 cm². -/
theorem right_triangle_area (h : ℝ) (m : ℝ) (area : ℝ) : 
  h = 5 → m = 6 → area = (1/2) * (2*m) * h → area = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3729_372940


namespace NUMINAMATH_CALUDE_smallest_with_9_factors_7_proper_is_180_l3729_372912

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The number of proper factors of a positive integer -/
def num_proper_factors (n : ℕ+) : ℕ := sorry

/-- The smallest positive integer with exactly 9 factors, 
    at least 7 of which are proper factors -/
def smallest_with_9_factors_7_proper : ℕ+ := sorry

theorem smallest_with_9_factors_7_proper_is_180 : 
  smallest_with_9_factors_7_proper = 180 := by sorry

end NUMINAMATH_CALUDE_smallest_with_9_factors_7_proper_is_180_l3729_372912


namespace NUMINAMATH_CALUDE_y_coordinate_of_C_l3729_372964

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Checks if a quadrilateral has a vertical line of symmetry -/
def hasVerticalSymmetry (q : Quadrilateral) : Prop := sorry

theorem y_coordinate_of_C (q : Quadrilateral) :
  q.A = ⟨0, 0⟩ →
  q.B = ⟨0, 1⟩ →
  q.D = ⟨3, 1⟩ →
  q.C.x = q.B.x →
  hasVerticalSymmetry q →
  area q = 18 →
  q.C.y = 11 := by sorry

end NUMINAMATH_CALUDE_y_coordinate_of_C_l3729_372964


namespace NUMINAMATH_CALUDE_marks_walking_distance_l3729_372925

/-- Given that Mark walks 1 mile in 30 minutes, prove that he walks 0.5 miles in 15 minutes. -/
theorem marks_walking_distance (mark_rate : ℝ) (h1 : mark_rate = 1 / 30) :
  mark_rate * 15 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_marks_walking_distance_l3729_372925


namespace NUMINAMATH_CALUDE_area_ratio_equilateral_triangle_extension_l3729_372946

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Extends a side of a triangle by a given factor -/
def extendSide (t : Triangle) (vertex : ℝ × ℝ) (factor : ℝ) : ℝ × ℝ := sorry

theorem area_ratio_equilateral_triangle_extension
  (ABC : Triangle)
  (h_equilateral : ABC.A.1^2 + ABC.A.2^2 = ABC.B.1^2 + ABC.B.2^2 ∧
                   ABC.B.1^2 + ABC.B.2^2 = ABC.C.1^2 + ABC.C.2^2 ∧
                   ABC.C.1^2 + ABC.C.2^2 = ABC.A.1^2 + ABC.A.2^2)
  (B' : ℝ × ℝ)
  (C' : ℝ × ℝ)
  (A' : ℝ × ℝ)
  (h_BB' : B' = extendSide ABC ABC.B 2)
  (h_CC' : C' = extendSide ABC ABC.C 3)
  (h_AA' : A' = extendSide ABC ABC.A 4)
  : area (Triangle.mk A' B' C') / area ABC = 42 := by sorry

end NUMINAMATH_CALUDE_area_ratio_equilateral_triangle_extension_l3729_372946


namespace NUMINAMATH_CALUDE_manny_money_left_l3729_372932

/-- The cost of a plastic chair in dollars -/
def chair_cost : ℚ := 55 / 5

/-- The cost of a portable table in dollars -/
def table_cost : ℚ := 3 * chair_cost

/-- Manny's initial amount of money in dollars -/
def initial_money : ℚ := 100

/-- The cost of Manny's purchase (one table and two chairs) in dollars -/
def purchase_cost : ℚ := table_cost + 2 * chair_cost

/-- The amount of money left after Manny's purchase -/
def money_left : ℚ := initial_money - purchase_cost

theorem manny_money_left : money_left = 45 := by sorry

end NUMINAMATH_CALUDE_manny_money_left_l3729_372932


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l3729_372927

theorem average_of_a_and_b (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 70)
  (h3 : c - a = 50) :
  (a + b) / 2 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l3729_372927


namespace NUMINAMATH_CALUDE_min_team_size_for_handshake_probability_l3729_372954

theorem min_team_size_for_handshake_probability (n : ℕ) : n ≥ 20 ↔ 
  (2 : ℚ) / (n + 1 : ℚ) < (1 : ℚ) / 10 ∧ 
  ∀ m : ℕ, m < n → (2 : ℚ) / (m + 1 : ℚ) ≥ (1 : ℚ) / 10 :=
by sorry

end NUMINAMATH_CALUDE_min_team_size_for_handshake_probability_l3729_372954


namespace NUMINAMATH_CALUDE_dans_age_problem_l3729_372986

theorem dans_age_problem (x : ℝ) : (8 + 20 : ℝ) = 7 * (8 - x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_dans_age_problem_l3729_372986


namespace NUMINAMATH_CALUDE_bob_oyster_shucking_l3729_372921

/-- Given that Bob can shuck 10 oysters in 5 minutes, this theorem proves
    that he can shuck 240 oysters in 2 hours. -/
theorem bob_oyster_shucking (bob_rate : ℕ) (bob_time : ℕ) (total_time : ℕ) :
  bob_rate = 10 →
  bob_time = 5 →
  total_time = 120 →
  (total_time / bob_time) * bob_rate = 240 :=
by
  sorry

#check bob_oyster_shucking

end NUMINAMATH_CALUDE_bob_oyster_shucking_l3729_372921


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3729_372977

def M : Set ℕ := {1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2*a - 1}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3729_372977


namespace NUMINAMATH_CALUDE_orange_juice_bottles_l3729_372948

theorem orange_juice_bottles (orange_price apple_price total_bottles total_cost : ℚ) 
  (h1 : orange_price = 70/100)
  (h2 : apple_price = 60/100)
  (h3 : total_bottles = 70)
  (h4 : total_cost = 4620/100) :
  ∃ (orange_bottles : ℚ), 
    orange_bottles * orange_price + (total_bottles - orange_bottles) * apple_price = total_cost ∧ 
    orange_bottles = 42 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_bottles_l3729_372948


namespace NUMINAMATH_CALUDE_smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11_l3729_372928

theorem smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11 : 
  ∃ w : ℕ, w > 0 ∧ w % 13 = 0 ∧ (w + 3) % 11 = 0 ∧
  ∀ x : ℕ, x > 0 ∧ x % 13 = 0 ∧ (x + 3) % 11 = 0 → w ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_divisible_by_13_and_w_plus_3_divisible_by_11_l3729_372928


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3729_372936

theorem triangle_angle_calculation (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  A = 60 →           -- Angle A is 60°
  C = 2 * B →        -- Angle C is twice Angle B
  C = 80 :=          -- Conclusion: Angle C is 80°
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3729_372936


namespace NUMINAMATH_CALUDE_chopped_cube_height_chopped_cube_height_value_l3729_372973

/-- The height of a 2x2x2 cube with a corner chopped off -/
theorem chopped_cube_height : ℝ :=
  let cube_side : ℝ := 2
  let cut_face_side : ℝ := 2 * Real.sqrt 2
  let cut_face_area : ℝ := Real.sqrt 3 / 4 * cut_face_side^2
  let removed_pyramid_height : ℝ := Real.sqrt 3 / 9
  cube_side - removed_pyramid_height

/-- Theorem stating that the height of the chopped cube is (17√3)/9 -/
theorem chopped_cube_height_value : chopped_cube_height = (17 * Real.sqrt 3) / 9 := by
  sorry


end NUMINAMATH_CALUDE_chopped_cube_height_chopped_cube_height_value_l3729_372973


namespace NUMINAMATH_CALUDE_cosine_rationality_l3729_372959

theorem cosine_rationality (x : ℝ) 
  (h1 : ∃ q : ℚ, (Real.sin (64 * x) + Real.sin (65 * x)) = ↑q)
  (h2 : ∃ r : ℚ, (Real.cos (64 * x) + Real.cos (65 * x)) = ↑r) :
  ∃ (a b : ℚ), (Real.cos (64 * x) = ↑a ∧ Real.cos (65 * x) = ↑b) :=
sorry

end NUMINAMATH_CALUDE_cosine_rationality_l3729_372959


namespace NUMINAMATH_CALUDE_area_of_triangle_PF₁F₂_l3729_372989

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point on the ellipse
def P : ℝ × ℝ := sorry

-- Assert that P is on the ellipse
axiom P_on_ellipse : is_on_ellipse P.1 P.2

-- Define the distances
def PF₁ : ℝ := sorry
def PF₂ : ℝ := sorry

-- Assert the ratio of distances
axiom distance_ratio : PF₁ / PF₂ = 2

-- Theorem to prove
theorem area_of_triangle_PF₁F₂ : 
  let triangle_area := sorry
  triangle_area = 4 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_PF₁F₂_l3729_372989


namespace NUMINAMATH_CALUDE_system_solution_unique_l3729_372905

theorem system_solution_unique :
  ∃! (x y : ℝ), (4 * x - 3 * y = 11) ∧ (2 * x + y = 13) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3729_372905


namespace NUMINAMATH_CALUDE_cubic_difference_l3729_372939

theorem cubic_difference (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l3729_372939


namespace NUMINAMATH_CALUDE_daily_harvest_l3729_372900

/-- The number of sections in the orchard -/
def num_sections : ℕ := 8

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := num_sections * sacks_per_section

theorem daily_harvest : total_sacks = 360 := by
  sorry

end NUMINAMATH_CALUDE_daily_harvest_l3729_372900


namespace NUMINAMATH_CALUDE_problem_polygon_area_l3729_372971

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a polygon on a 2D grid --/
structure GridPolygon where
  vertices : List GridPoint

/-- Calculates the area of a polygon on a grid --/
def calculateGridPolygonArea (polygon : GridPolygon) : ℕ :=
  sorry

/-- The specific polygon from the problem --/
def problemPolygon : GridPolygon :=
  { vertices := [
    {x := 0, y := 0}, {x := 1, y := 0}, {x := 1, y := 1}, {x := 2, y := 1},
    {x := 3, y := 0}, {x := 3, y := 1}, {x := 4, y := 0}, {x := 4, y := 1},
    {x := 4, y := 3}, {x := 3, y := 3}, {x := 4, y := 4}, {x := 3, y := 4},
    {x := 2, y := 4}, {x := 0, y := 4}, {x := 0, y := 2}, {x := 0, y := 0}
  ] }

theorem problem_polygon_area : calculateGridPolygonArea problemPolygon = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_polygon_area_l3729_372971


namespace NUMINAMATH_CALUDE_building_E_floors_l3729_372913

/-- The number of floors in Building A -/
def floors_A : ℕ := 4

/-- The number of floors in Building B -/
def floors_B : ℕ := floors_A + 9

/-- The number of floors in Building C -/
def floors_C : ℕ := 5 * floors_B - 6

/-- The number of floors in Building D -/
def floors_D : ℕ := 2 * floors_C - (floors_A + floors_B)

/-- The number of floors in Building E -/
def floors_E : ℕ := 3 * (floors_B + floors_C + floors_D) - 10

/-- Theorem stating that Building E has 509 floors -/
theorem building_E_floors : floors_E = 509 := by
  sorry

end NUMINAMATH_CALUDE_building_E_floors_l3729_372913


namespace NUMINAMATH_CALUDE_jenny_total_wins_l3729_372953

/-- The number of games Jenny played against Mark -/
def games_with_mark : ℕ := 10

/-- The number of games Mark won against Jenny -/
def marks_wins : ℕ := 1

/-- The number of games Jenny played against Jill -/
def games_with_jill : ℕ := 2 * games_with_mark

/-- The percentage of games Jill won against Jenny -/
def jills_win_percentage : ℚ := 75 / 100

theorem jenny_total_wins : 
  (games_with_mark - marks_wins) + 
  (games_with_jill - (jills_win_percentage * games_with_jill).num) = 14 := by
sorry

end NUMINAMATH_CALUDE_jenny_total_wins_l3729_372953


namespace NUMINAMATH_CALUDE_gas_price_calculation_l3729_372919

theorem gas_price_calculation (rental_cost mileage_rate total_miles total_expense gas_gallons : ℚ)
  (h1 : rental_cost = 150)
  (h2 : mileage_rate = 1/2)
  (h3 : total_miles = 320)
  (h4 : total_expense = 338)
  (h5 : gas_gallons = 8) :
  (total_expense - rental_cost - mileage_rate * total_miles) / gas_gallons = 7/2 := by
  sorry

#eval (338 : ℚ) - 150 - 1/2 * 320
#eval ((338 : ℚ) - 150 - 1/2 * 320) / 8

end NUMINAMATH_CALUDE_gas_price_calculation_l3729_372919


namespace NUMINAMATH_CALUDE_certain_number_proof_l3729_372966

theorem certain_number_proof (x : ℝ) : (60 / 100 * 500 = 50 / 100 * x) → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3729_372966


namespace NUMINAMATH_CALUDE_altitude_df_length_l3729_372931

/-- Represents a parallelogram ABCD with altitudes DE and DF -/
structure Parallelogram where
  /-- Length of side DC -/
  dc : ℝ
  /-- Length of segment EB -/
  eb : ℝ
  /-- Length of altitude DE -/
  de : ℝ
  /-- Ensures dc is positive -/
  dc_pos : dc > 0
  /-- Ensures eb is positive -/
  eb_pos : eb > 0
  /-- Ensures de is positive -/
  de_pos : de > 0
  /-- Ensures eb is less than dc (as EB is part of AB which is equal to DC) -/
  eb_lt_dc : eb < dc

/-- Theorem stating that under the given conditions, DF = 5 -/
theorem altitude_df_length (p : Parallelogram) (h1 : p.dc = 15) (h2 : p.eb = 3) (h3 : p.de = 5) :
  ∃ df : ℝ, df = 5 ∧ df > 0 := by
  sorry

end NUMINAMATH_CALUDE_altitude_df_length_l3729_372931


namespace NUMINAMATH_CALUDE_smallest_gcd_ef_l3729_372902

theorem smallest_gcd_ef (d e f : ℕ+) (h1 : Nat.gcd d e = 210) (h2 : Nat.gcd d f = 770) :
  ∃ (e' f' : ℕ+), Nat.gcd d e' = 210 ∧ Nat.gcd d f' = 770 ∧ 
  Nat.gcd e' f' = 10 ∧ ∀ (e'' f'' : ℕ+), Nat.gcd d e'' = 210 → Nat.gcd d f'' = 770 → 
  Nat.gcd e'' f'' ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_ef_l3729_372902


namespace NUMINAMATH_CALUDE_sum_of_first_three_terms_l3729_372922

-- Define the sequence a_n
def a (n : ℕ) : ℚ := n * (n + 1) / 2

-- Define S_3 as the sum of the first three terms
def S3 : ℚ := a 1 + a 2 + a 3

-- Theorem statement
theorem sum_of_first_three_terms : S3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_three_terms_l3729_372922


namespace NUMINAMATH_CALUDE_correct_calculation_l3729_372981

theorem correct_calculation (a b : ℝ) : 2 * a^2 * b - 4 * a^2 * b = -2 * a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3729_372981


namespace NUMINAMATH_CALUDE_probability_not_above_x_axis_l3729_372951

/-- Parallelogram ABCD with given vertices -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The specific parallelogram ABCD from the problem -/
def ABCD : Parallelogram :=
  { A := (4, 4)
    B := (-2, -2)
    C := (-8, -2)
    D := (0, 4) }

/-- Function to calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Function to calculate the area of the part of the parallelogram below the x-axis -/
def areaBelowXAxis (p : Parallelogram) : ℝ := sorry

/-- Theorem stating the probability of a point not being above the x-axis -/
theorem probability_not_above_x_axis (p : Parallelogram) :
  p = ABCD →
  (areaBelowXAxis p) / (area p) = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_not_above_x_axis_l3729_372951


namespace NUMINAMATH_CALUDE_floor_frac_equation_solutions_l3729_372965

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ :=
  x - (floor x : ℝ)

-- State the theorem
theorem floor_frac_equation_solutions :
  ∀ x : ℝ, (floor x : ℝ) * frac x = 2019 * x ↔ x = 0 ∨ x = -1/2020 := by
  sorry

end NUMINAMATH_CALUDE_floor_frac_equation_solutions_l3729_372965


namespace NUMINAMATH_CALUDE_a_plus_b_value_l3729_372978

theorem a_plus_b_value (a b : ℝ) : 
  (∀ x, (3 * (a * x + b) - 8) = 4 * x + 5) → 
  a + b = 17/3 := by sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l3729_372978


namespace NUMINAMATH_CALUDE_complex_in_third_quadrant_l3729_372929

theorem complex_in_third_quadrant (z : ℂ) : z * (1 + Complex.I) = 1 - 2 * Complex.I → 
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_third_quadrant_l3729_372929


namespace NUMINAMATH_CALUDE_candy_cost_proof_l3729_372949

/-- The cost of candy A per pound -/
def cost_candy_A : ℝ := 3.20

/-- The cost of candy B per pound -/
def cost_candy_B : ℝ := 1.70

/-- The total weight of the mixture in pounds -/
def total_weight : ℝ := 5

/-- The cost per pound of the mixture -/
def mixture_cost_per_pound : ℝ := 2

/-- The weight of candy A in the mixture -/
def weight_candy_A : ℝ := 1

/-- The weight of candy B in the mixture -/
def weight_candy_B : ℝ := total_weight - weight_candy_A

theorem candy_cost_proof :
  cost_candy_A * weight_candy_A + cost_candy_B * weight_candy_B = mixture_cost_per_pound * total_weight :=
by sorry

end NUMINAMATH_CALUDE_candy_cost_proof_l3729_372949


namespace NUMINAMATH_CALUDE_cone_volume_l3729_372952

/-- The volume of a cone with height equal to its radius, where the radius is √m and m is a rational number -/
theorem cone_volume (m : ℚ) (h : m > 0) : 
  let r : ℝ := Real.sqrt m
  let volume := (1/3 : ℝ) * Real.pi * r^2 * r
  volume = (1/3 : ℝ) * Real.pi * m^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l3729_372952


namespace NUMINAMATH_CALUDE_right_triangle_cosine_l3729_372943

theorem right_triangle_cosine (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 5) (h3 : c = 13) :
  let cos_C := a / c
  cos_C = 5 / 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_cosine_l3729_372943


namespace NUMINAMATH_CALUDE_marbles_problem_l3729_372924

theorem marbles_problem (total : ℕ) (bags : ℕ) (remaining : ℕ) : 
  bags = 4 →
  remaining = 21 →
  (total / bags) * (bags - 1) = remaining →
  total = 28 :=
by sorry

end NUMINAMATH_CALUDE_marbles_problem_l3729_372924


namespace NUMINAMATH_CALUDE_sin_inequality_l3729_372962

theorem sin_inequality : 
  Real.sin (11 * π / 180) < Real.sin (168 * π / 180) ∧ 
  Real.sin (168 * π / 180) < Real.cos (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_l3729_372962


namespace NUMINAMATH_CALUDE_total_combinations_l3729_372942

/-- Represents the number of friends in Victoria's group. -/
def num_friends : ℕ := 35

/-- Represents the minimum shoe size. -/
def min_size : ℕ := 5

/-- Represents the maximum shoe size. -/
def max_size : ℕ := 15

/-- Represents the number of unique designs in the store. -/
def num_designs : ℕ := 20

/-- Represents the number of colors for each design. -/
def colors_per_design : ℕ := 4

/-- Represents the number of colors each friend needs to select. -/
def colors_to_select : ℕ := 3

/-- Calculates the number of ways to choose k items from n items. -/
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Theorem stating the total number of combinations to explore. -/
theorem total_combinations : 
  num_friends * num_designs * combination colors_per_design colors_to_select = 2800 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_l3729_372942


namespace NUMINAMATH_CALUDE_sum_of_divisors_143_l3729_372950

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_143 : sum_of_divisors 143 = 168 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_143_l3729_372950


namespace NUMINAMATH_CALUDE_counsel_probability_l3729_372968

def CANOE : Finset Char := {'C', 'A', 'N', 'O', 'E'}
def SHRUB : Finset Char := {'S', 'H', 'R', 'U', 'B'}
def FLOW : Finset Char := {'F', 'L', 'O', 'W'}
def COUNSEL : Finset Char := {'C', 'O', 'U', 'N', 'S', 'E', 'L'}

def prob_CANOE : ℚ := 1 / (CANOE.card.choose 2)
def prob_SHRUB : ℚ := 3 / (SHRUB.card.choose 3)
def prob_FLOW : ℚ := 1 / (FLOW.card.choose 4)

theorem counsel_probability :
  prob_CANOE * prob_SHRUB * prob_FLOW = 3 / 100 := by
  sorry

end NUMINAMATH_CALUDE_counsel_probability_l3729_372968


namespace NUMINAMATH_CALUDE_square_greater_than_linear_for_less_than_negative_one_l3729_372993

theorem square_greater_than_linear_for_less_than_negative_one (x : ℝ) :
  x < -1 → x^2 > 1 + x := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_linear_for_less_than_negative_one_l3729_372993


namespace NUMINAMATH_CALUDE_triangle_area_proof_l3729_372903

/-- Represents a triangle with parallel lines -/
structure TriangleWithParallelLines where
  /-- The area of the largest part -/
  largest_part_area : ℝ
  /-- The number of parallel lines -/
  num_parallel_lines : ℕ
  /-- The number of equal segments on the other two sides -/
  num_segments : ℕ
  /-- The number of parts the triangle is divided into -/
  num_parts : ℕ

/-- Theorem: If a triangle with 9 parallel lines dividing the sides into 10 equal segments
    has its largest part with an area of 38, then the total area of the triangle is 200 -/
theorem triangle_area_proof (t : TriangleWithParallelLines)
    (h1 : t.largest_part_area = 38)
    (h2 : t.num_parallel_lines = 9)
    (h3 : t.num_segments = 10)
    (h4 : t.num_parts = 10) :
    ∃ (total_area : ℝ), total_area = 200 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3729_372903


namespace NUMINAMATH_CALUDE_x_squared_greater_than_x_root_l3729_372972

theorem x_squared_greater_than_x_root (x : ℝ) : x ^ 2 > x ^ (1 / 2) ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_greater_than_x_root_l3729_372972


namespace NUMINAMATH_CALUDE_decreasing_g_implies_a_bound_f_nonpositive_implies_a_bound_l3729_372904

noncomputable section

variables (a b : ℝ)

def f (x : ℝ) : ℝ := Real.log x + a * x - 1 / x + b

def g (x : ℝ) : ℝ := f a b x + 2 / x

theorem decreasing_g_implies_a_bound :
  (∀ x > 0, ∀ y > 0, x < y → g a b y < g a b x) →
  a ≤ -1/4 := by sorry

theorem f_nonpositive_implies_a_bound :
  (∀ x > 0, f a b x ≤ 0) →
  a ≤ 1 - b := by sorry

end NUMINAMATH_CALUDE_decreasing_g_implies_a_bound_f_nonpositive_implies_a_bound_l3729_372904


namespace NUMINAMATH_CALUDE_existence_of_even_and_odd_composite_functions_l3729_372997

theorem existence_of_even_and_odd_composite_functions :
  ∃ (p q : ℝ → ℝ),
    (∀ x, p (-x) = p x) ∧
    (∀ x, p (q (-x)) = -(p (q x))) ∧
    (∃ x, p (q x) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_even_and_odd_composite_functions_l3729_372997


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3729_372947

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x : ℝ), x = 6 ∧ x^2 = a^2 + b^2) →
  (∃ (x y : ℝ), y = Real.sqrt 3 * x ∧ b / a = Real.sqrt 3) →
  a^2 = 9 ∧ b^2 = 27 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3729_372947
