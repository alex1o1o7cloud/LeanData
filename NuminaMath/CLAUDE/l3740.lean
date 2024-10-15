import Mathlib

namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_geq_one_l3740_374026

/-- The function f(x) = x³ - ax² - x + 6 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - x + 6

/-- f is monotonically decreasing in the interval (0,1) --/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y ∧ y < 1 → f a x > f a y

theorem monotone_decreasing_implies_a_geq_one (a : ℝ) :
  is_monotone_decreasing a → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_geq_one_l3740_374026


namespace NUMINAMATH_CALUDE_odd_solutions_count_l3740_374064

theorem odd_solutions_count (x : ℕ) : 
  (∃ (s : Finset ℕ), 
    (∀ y ∈ s, 20 ≤ y ∧ y ≤ 150 ∧ Odd y ∧ (y + 17) % 29 = 65 % 29) ∧ 
    (∀ y, 20 ≤ y ∧ y ≤ 150 ∧ Odd y ∧ (y + 17) % 29 = 65 % 29 → y ∈ s) ∧
    Finset.card s = 3) := by
  sorry

end NUMINAMATH_CALUDE_odd_solutions_count_l3740_374064


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3740_374037

/-- Defines an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Defines a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a line passing through two points -/
def Line (p1 p2 : Point) :=
  {p : Point | (p.y - p1.y) * (p2.x - p1.x) = (p.x - p1.x) * (p2.y - p1.y)}

theorem ellipse_eccentricity (Γ : Ellipse) 
  (F : Point) 
  (A : Point) 
  (B : Point) 
  (N : Point) :
  F.x = 3 ∧ F.y = 0 →
  A.x = 0 ∧ A.y = Γ.b →
  B.x = 0 ∧ B.y = -Γ.b →
  N.x = 12 ∧ N.y = 0 →
  ∃ (M : Point), M ∈ Line A F ∧ M ∈ Line B N ∧ 
    (M.x^2 / Γ.a^2 + M.y^2 / Γ.b^2 = 1) →
  (Γ.a^2 - Γ.b^2) / Γ.a^2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3740_374037


namespace NUMINAMATH_CALUDE_travel_time_l3740_374041

/-- Given a person's travel rate, calculate the time to travel a certain distance -/
theorem travel_time (distance_to_julia : ℝ) (time_to_julia : ℝ) (distance_to_bernard : ℝ) :
  distance_to_julia = 2 →
  time_to_julia = 8 →
  distance_to_bernard = 5 →
  (distance_to_bernard / distance_to_julia) * time_to_julia = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_travel_time_l3740_374041


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l3740_374000

theorem invalid_votes_percentage (total_votes : ℕ) (valid_votes_winner_percentage : ℚ)
  (valid_votes_loser : ℕ) (h1 : total_votes = 5500)
  (h2 : valid_votes_winner_percentage = 55/100)
  (h3 : valid_votes_loser = 1980) :
  (total_votes - (valid_votes_loser / (1 - valid_votes_winner_percentage))) / total_votes = 1/5 := by
  sorry

#check invalid_votes_percentage

end NUMINAMATH_CALUDE_invalid_votes_percentage_l3740_374000


namespace NUMINAMATH_CALUDE_tianjin_population_scientific_notation_l3740_374009

/-- The population of Tianjin -/
def tianjin_population : ℕ := 13860000

/-- Scientific notation representation of Tianjin's population -/
def tianjin_scientific : ℝ := 1.386 * (10 ^ 7)

/-- Theorem stating that the population of Tianjin in scientific notation is correct -/
theorem tianjin_population_scientific_notation :
  (tianjin_population : ℝ) = tianjin_scientific :=
by sorry

end NUMINAMATH_CALUDE_tianjin_population_scientific_notation_l3740_374009


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l3740_374048

theorem sum_of_fractions_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) :
  (a^3*b^3 / ((a^3 - b*c)*(b^3 - a*c))) + 
  (a^3*c^3 / ((a^3 - b*c)*(c^3 - a*b))) + 
  (b^3*c^3 / ((b^3 - a*c)*(c^3 - a*b))) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l3740_374048


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l3740_374054

theorem gcd_of_polynomial_and_multiple (x : ℤ) : 
  (∃ k : ℤ, x = 34567 * k) → 
  Nat.gcd ((3*x+4)*(8*x+3)*(15*x+11)*(x+15) : ℤ).natAbs x.natAbs = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l3740_374054


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3740_374024

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (2*k - 1)*x₁ + k^2 - 1 = 0 ∧ 
    x₂^2 + (2*k - 1)*x₂ + k^2 - 1 = 0 ∧ 
    x₁^2 + x₂^2 = 16 + x₁*x₂) →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3740_374024


namespace NUMINAMATH_CALUDE_jerrys_age_l3740_374075

/-- Given that Mickey's age is 6 years less than 200% of Jerry's age and Mickey is 20 years old, prove that Jerry is 13 years old. -/
theorem jerrys_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 2 * jerry_age - 6) 
  (h2 : mickey_age = 20) : 
  jerry_age = 13 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l3740_374075


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l3740_374011

theorem fraction_sum_simplification :
  150 / 225 + 90 / 135 = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l3740_374011


namespace NUMINAMATH_CALUDE_number_of_winning_scores_l3740_374002

/-- Represents a cross country meet between two teams -/
structure CrossCountryMeet where
  /-- The number of runners in each team -/
  runnersPerTeam : Nat
  /-- The total number of runners -/
  totalRunners : Nat
  /-- Assertion that there are two teams -/
  twoTeams : totalRunners = 2 * runnersPerTeam

/-- Calculates the total score of all runners -/
def totalScore (meet : CrossCountryMeet) : Nat :=
  (meet.totalRunners * (meet.totalRunners + 1)) / 2

/-- Calculates the minimum possible team score -/
def minTeamScore (meet : CrossCountryMeet) : Nat :=
  (meet.runnersPerTeam * (meet.runnersPerTeam + 1)) / 2

/-- Calculates the maximum possible winning score -/
def maxWinningScore (meet : CrossCountryMeet) : Nat :=
  (totalScore meet) / 2 - 1

/-- The main theorem stating the number of possible winning scores -/
theorem number_of_winning_scores (meet : CrossCountryMeet) 
  (h : meet.runnersPerTeam = 6) : 
  (maxWinningScore meet) - (minTeamScore meet) + 1 = 18 := by
  sorry


end NUMINAMATH_CALUDE_number_of_winning_scores_l3740_374002


namespace NUMINAMATH_CALUDE_exactly_two_valid_pairs_l3740_374036

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def is_valid_pair (n m : ℕ+) : Prop := sum_factorials n.val = m.val ^ 2

theorem exactly_two_valid_pairs :
  ∃! (s : Finset (ℕ+ × ℕ+)), s.card = 2 ∧ ∀ (p : ℕ+ × ℕ+), p ∈ s ↔ is_valid_pair p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_pairs_l3740_374036


namespace NUMINAMATH_CALUDE_equation_solution_l3740_374069

theorem equation_solution :
  ∃ x : ℝ, (Real.sqrt (x + 16) - (8 * Real.cos (π / 6)) / Real.sqrt (x + 16) = 4) ∧
  (x = (2 + 2 * Real.sqrt (1 + Real.sqrt 3))^2 - 16) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3740_374069


namespace NUMINAMATH_CALUDE_union_of_sets_l3740_374049

theorem union_of_sets : 
  let M : Set Int := {1, 0, -1}
  let N : Set Int := {1, 2}
  M ∪ N = {1, 2, 0, -1} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l3740_374049


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l3740_374082

theorem complex_expression_simplification :
  Real.rpow 0.027 (1/3) * Real.rpow (225/64) (-1/2) / Real.sqrt (Real.rpow (-8/125) (2/3)) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l3740_374082


namespace NUMINAMATH_CALUDE_smoking_and_sickness_are_distinct_categorical_variables_l3740_374015

-- Define a structure for categorical variables
structure CategoricalVariable where
  name : String
  values : List String

-- Define the "Whether smoking" categorical variable
def whetherSmoking : CategoricalVariable := {
  name := "Whether smoking"
  values := ["smoking", "not smoking"]
}

-- Define the "Whether sick" categorical variable
def whetherSick : CategoricalVariable := {
  name := "Whether sick"
  values := ["sick", "not sick"]
}

-- Theorem to prove that "Whether smoking" and "Whether sick" are two distinct categorical variables
theorem smoking_and_sickness_are_distinct_categorical_variables :
  whetherSmoking ≠ whetherSick :=
sorry

end NUMINAMATH_CALUDE_smoking_and_sickness_are_distinct_categorical_variables_l3740_374015


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l3740_374066

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, b (n + 1) = b n * r

def increasing_sequence (s : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, s (n + 1) > s n

theorem sequence_sum_theorem (a b : ℕ → ℕ) (k : ℕ) :
  a 1 = 1 →
  b 1 = 1 →
  arithmetic_sequence a →
  geometric_sequence b →
  increasing_sequence a →
  increasing_sequence b →
  (∃ k : ℕ, a (k - 1) + b (k - 1) = 250 ∧ a (k + 1) + b (k + 1) = 1250) →
  a k + b k = 502 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l3740_374066


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3740_374028

theorem quadratic_root_value (a : ℝ) : 
  a^2 - 2*a - 3 = 0 → 2*a^2 - 4*a + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3740_374028


namespace NUMINAMATH_CALUDE_min_value_expression_l3740_374005

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x - 1)^2 / (y - 2) + (y - 1)^2 / (x - 2) ≥ 8 ∧
  (∃ x y : ℝ, x > 2 ∧ y > 2 ∧ (x - 1)^2 / (y - 2) + (y - 1)^2 / (x - 2) = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3740_374005


namespace NUMINAMATH_CALUDE_freshman_class_size_l3740_374020

theorem freshman_class_size :
  ∃! n : ℕ, n < 600 ∧ n % 19 = 15 ∧ n % 17 = 11 ∧ n = 53 := by
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l3740_374020


namespace NUMINAMATH_CALUDE_a_fraction_is_one_third_l3740_374087

-- Define the partnership
structure Partnership :=
  (total_capital : ℝ)
  (a_fraction : ℝ)
  (b_fraction : ℝ)
  (c_fraction : ℝ)
  (d_fraction : ℝ)
  (total_profit : ℝ)
  (a_profit : ℝ)

-- Define the conditions
def partnership_conditions (p : Partnership) : Prop :=
  p.b_fraction = 1/4 ∧
  p.c_fraction = 1/5 ∧
  p.d_fraction = 1 - (p.a_fraction + p.b_fraction + p.c_fraction) ∧
  p.total_profit = 2445 ∧
  p.a_profit = 815 ∧
  p.a_profit / p.total_profit = p.a_fraction

-- Theorem statement
theorem a_fraction_is_one_third (p : Partnership) :
  partnership_conditions p → p.a_fraction = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_a_fraction_is_one_third_l3740_374087


namespace NUMINAMATH_CALUDE_proposition_l3740_374053

theorem proposition (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  (Real.sqrt (b^2 - a*c)) / a < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_proposition_l3740_374053


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3740_374089

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) ∧
  (∀ x : ℝ, (x - 1) * (x - 2) * (x - 3) = f x) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3740_374089


namespace NUMINAMATH_CALUDE_perimeter_of_specific_triangle_l3740_374086

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of DP, where P is the tangency point on DE -/
  dp : ℝ
  /-- The length of PE, where P is the tangency point on DE -/
  pe : ℝ
  /-- The length of the tangent from vertex F to the circle -/
  ft : ℝ

/-- The perimeter of the triangle -/
def perimeter (t : TriangleWithInscribedCircle) : ℝ :=
  2 * (t.dp + t.pe + t.ft)

theorem perimeter_of_specific_triangle :
  let t : TriangleWithInscribedCircle := {
    r := 13,
    dp := 17,
    pe := 31,
    ft := 20
  }
  perimeter t = 136 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_specific_triangle_l3740_374086


namespace NUMINAMATH_CALUDE_distance_after_time_l3740_374052

theorem distance_after_time (adam_speed simon_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  adam_speed = 5 →
  simon_speed = 12 →
  time = 5 →
  distance = 65 →
  (adam_speed * time) ^ 2 + (simon_speed * time) ^ 2 = distance ^ 2 := by
sorry

end NUMINAMATH_CALUDE_distance_after_time_l3740_374052


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3740_374032

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | a * x^2 - (2 + a) * x + 2 > 0} = {x : ℝ | 2 / a < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3740_374032


namespace NUMINAMATH_CALUDE_square_difference_l3740_374057

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 9/20) 
  (h2 : x - y = 1/20) : 
  x^2 - y^2 = 9/400 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l3740_374057


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3740_374062

theorem fractional_equation_solution :
  ∃ x : ℝ, (x + 2) / (x - 1) = 0 ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3740_374062


namespace NUMINAMATH_CALUDE_cone_base_circumference_l3740_374077

theorem cone_base_circumference (r : ℝ) (angle : ℝ) :
  r = 6 →
  angle = 300 →
  let full_circumference := 2 * Real.pi * r
  let remaining_fraction := angle / 360
  let cone_base_circumference := remaining_fraction * full_circumference
  cone_base_circumference = 10 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l3740_374077


namespace NUMINAMATH_CALUDE_range_of_a_l3740_374042

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x - 1 ≥ a^2 ∧ x - 4 < 2*a) → 
  -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3740_374042


namespace NUMINAMATH_CALUDE_arithmetic_ellipse_properties_l3740_374080

/-- An arithmetic ellipse with semi-major axis a, semi-minor axis b, and focal distance c -/
structure ArithmeticEllipse where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  a_gt_b : b < a
  arithmetic_progression : 2 * b = a + c
  ellipse_equation : a^2 - b^2 = c^2

/-- Main theorem about arithmetic ellipses -/
theorem arithmetic_ellipse_properties (Γ : ArithmeticEllipse) :
  -- 1. Eccentricity is 3/5
  (Γ.c / Γ.a = 3/5) ∧
  -- 2. Slope of tangent line at (0, -a) is ±3/5
  (∃ k : ℝ, k^2 = (3/5)^2 ∧
    ∀ x y : ℝ, (x^2 / Γ.a^2 + y^2 / Γ.b^2 = 1) → (y = k * x - Γ.a) →
      x^2 + (k * x - Γ.a)^2 / Γ.b^2 = 1) ∧
  -- 3. Circle with diameter MN passes through (±b, 0)
  (∀ m n : ℝ, (m^2 / Γ.a^2 + n^2 / Γ.b^2 = 1) →
    ∃ y : ℝ, ((-Γ.b)^2 + y^2 + (Γ.a * n / (m + Γ.a) + Γ.a * n / (m - Γ.a)) * y - Γ.b^2 = 0) ∧
              (Γ.b^2 + y^2 + (Γ.a * n / (m + Γ.a) + Γ.a * n / (m - Γ.a)) * y - Γ.b^2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_ellipse_properties_l3740_374080


namespace NUMINAMATH_CALUDE_multiply_mistake_l3740_374063

theorem multiply_mistake (x : ℝ) : 97 * x - 89 * x = 4926 → x = 615.75 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mistake_l3740_374063


namespace NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_l3740_374043

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of corresponding angles
def corresponding_angles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry

-- Define the concept of equal angles
def angle_equal (a1 a2 : Angle) : Prop := sorry

-- The theorem to be proved
theorem parallel_lines_corresponding_angles 
  (l1 l2 : Line) (a1 a2 : Angle) : 
  parallel l1 l2 → corresponding_angles a1 a2 l1 l2 → angle_equal a1 a2 := by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_l3740_374043


namespace NUMINAMATH_CALUDE_intersection_of_M_and_P_l3740_374078

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x, y = 3^x}
def P : Set ℝ := {y | y ≥ 1}

-- State the theorem
theorem intersection_of_M_and_P : M ∩ P = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_P_l3740_374078


namespace NUMINAMATH_CALUDE_rectangle_formation_ways_l3740_374047

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of horizontal lines -/
def num_horizontal_lines : ℕ := 4

/-- The number of vertical lines -/
def num_vertical_lines : ℕ := 4

/-- The number of lines needed to form a side of the rectangle -/
def lines_per_side : ℕ := 2

/-- Theorem stating that the number of ways to choose lines to form a rectangle is 36 -/
theorem rectangle_formation_ways :
  (choose num_horizontal_lines lines_per_side) * (choose num_vertical_lines lines_per_side) = 36 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_formation_ways_l3740_374047


namespace NUMINAMATH_CALUDE_point_not_in_region_l3740_374025

def region (x y : ℝ) : Prop := 3 * x + 2 * y < 6

theorem point_not_in_region :
  ¬(region 2 0) ∧ 
  (region 0 0) ∧ 
  (region 1 1) ∧ 
  (region 0 2) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_region_l3740_374025


namespace NUMINAMATH_CALUDE_unique_special_polynomial_l3740_374004

/-- A polynomial function that satisfies the given conditions -/
structure SpecialPolynomial where
  f : ℝ → ℝ
  is_polynomial : Polynomial ℝ
  degree_ge_one : (Polynomial.degree is_polynomial) ≥ 1
  cond_square : ∀ x, f (x^2) = (f x)^2
  cond_compose : ∀ x, f (x^2) = f (f x)

/-- Theorem stating that there exists exactly one special polynomial -/
theorem unique_special_polynomial :
  ∃! (p : SpecialPolynomial), True :=
sorry

end NUMINAMATH_CALUDE_unique_special_polynomial_l3740_374004


namespace NUMINAMATH_CALUDE_marble_ratio_l3740_374022

theorem marble_ratio (total : ℕ) (blue green yellow : ℕ) 
  (h_total : total = 164)
  (h_blue : blue = total / 2)
  (h_green : green = 27)
  (h_yellow : yellow = 14) :
  (total - (blue + green + yellow)) * 4 = total := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l3740_374022


namespace NUMINAMATH_CALUDE_problem_solution_l3740_374071

theorem problem_solution : 
  (Real.sqrt 4 + abs (-3) + (2 - Real.pi) ^ 0 = 6) ∧ 
  (Real.sqrt 18 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt ((-5)^2) = 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3740_374071


namespace NUMINAMATH_CALUDE_incorrect_subset_l3740_374050

-- Define the sets
def set1 : Set ℕ := {1, 2, 3}
def set2 : Set ℕ := {1, 2}

-- Theorem statement
theorem incorrect_subset : ¬(set1 ⊆ set2) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_subset_l3740_374050


namespace NUMINAMATH_CALUDE_books_read_is_seven_l3740_374017

-- Define the number of movies watched
def movies_watched : ℕ := 21

-- Define the relationship between movies watched and books read
def books_read : ℕ := movies_watched - 14

-- Theorem to prove
theorem books_read_is_seven : books_read = 7 := by
  sorry

end NUMINAMATH_CALUDE_books_read_is_seven_l3740_374017


namespace NUMINAMATH_CALUDE_correct_derivatives_l3740_374044

open Real

theorem correct_derivatives :
  (∀ x : ℝ, deriv (λ x => (2 * x) / (x^2 + 1)) x = (2 - 2 * x^2) / (x^2 + 1)^2) ∧
  (∀ x : ℝ, deriv (λ x => exp (3 * x + 1)) x = 3 * exp (3 * x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_correct_derivatives_l3740_374044


namespace NUMINAMATH_CALUDE_find_k_value_l3740_374065

theorem find_k_value (k : ℝ) (h : 64 / k = 4) : k = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l3740_374065


namespace NUMINAMATH_CALUDE_rectangle_area_l3740_374076

/-- 
A rectangle with diagonal length x and length three times its width 
has an area of (3/10)x^2
-/
theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ w : ℝ, w > 0 ∧ 
    w^2 + (3*w)^2 = x^2 ∧ 
    3 * w^2 = (3/10) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3740_374076


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_one_and_five_l3740_374091

theorem sqrt_equality_implies_one_and_five (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) :
  Real.sqrt (1 + Real.sqrt (45 + 20 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b →
  a = 1 ∧ b = 5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_one_and_five_l3740_374091


namespace NUMINAMATH_CALUDE_all_statements_imply_p_and_q_implies_r_l3740_374033

theorem all_statements_imply_p_and_q_implies_r (p q r : Prop) :
  ((p ∧ q ∧ r) → ((p ∧ q) → r)) ∧
  ((¬p ∧ q ∧ ¬r) → ((p ∧ q) → r)) ∧
  ((p ∧ ¬q ∧ ¬r) → ((p ∧ q) → r)) ∧
  ((¬p ∧ ¬q ∧ r) → ((p ∧ q) → r)) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_imply_p_and_q_implies_r_l3740_374033


namespace NUMINAMATH_CALUDE_A_intersect_C_R_B_eq_interval_l3740_374016

-- Define set A
def A : Set ℝ := {x | x^2 + x - 6 ≤ 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sqrt x ∧ 0 ≤ x ∧ x ≤ 4}

-- Define the complement of B relative to ℝ
def C_R_B : Set ℝ := (Set.univ : Set ℝ) \ B

-- Theorem statement
theorem A_intersect_C_R_B_eq_interval :
  A ∩ C_R_B = Set.Icc (-3 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_A_intersect_C_R_B_eq_interval_l3740_374016


namespace NUMINAMATH_CALUDE_original_class_size_l3740_374095

theorem original_class_size (initial_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  initial_avg = 40 →
  new_students = 12 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ (original_size : ℕ),
    (original_size * initial_avg + new_students * new_avg) / (original_size + new_students) = initial_avg - avg_decrease ∧
    original_size = 12 := by
  sorry

end NUMINAMATH_CALUDE_original_class_size_l3740_374095


namespace NUMINAMATH_CALUDE_odd_function_power_l3740_374034

def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - |x - a|

theorem odd_function_power (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is an odd function
  (∃ x, f a x ≠ 0) →          -- f is not identically zero
  a^2012 = 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_power_l3740_374034


namespace NUMINAMATH_CALUDE_inequality_proof_l3740_374051

theorem inequality_proof (k n : ℕ) (hk : k > 0) (hn : n > 0) (hkn : k ≤ n) :
  1 + (k : ℝ) / n ≤ (1 + 1 / n) ^ k ∧ (1 + 1 / n) ^ k < 1 + (k : ℝ) / n + (k : ℝ)^2 / n^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3740_374051


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l3740_374092

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 1 →
  B = π / 4 →
  (1 / 2) * a * b * Real.sin C = 2 →
  b = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l3740_374092


namespace NUMINAMATH_CALUDE_toy_price_problem_l3740_374056

theorem toy_price_problem (num_toys : ℕ) (sixth_toy_price : ℝ) (new_average : ℝ) :
  num_toys = 5 →
  sixth_toy_price = 16 →
  new_average = 11 →
  (num_toys : ℝ) * (num_toys + 1 : ℝ)⁻¹ * (num_toys * new_average - sixth_toy_price) = 10 :=
by sorry

end NUMINAMATH_CALUDE_toy_price_problem_l3740_374056


namespace NUMINAMATH_CALUDE_polygon_with_60_degree_exterior_angles_has_6_sides_l3740_374007

-- Define a polygon type
structure Polygon where
  sides : ℕ
  exteriorAngle : ℝ

-- Theorem statement
theorem polygon_with_60_degree_exterior_angles_has_6_sides :
  ∀ p : Polygon, p.exteriorAngle = 60 → p.sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_60_degree_exterior_angles_has_6_sides_l3740_374007


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3740_374003

theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 3) * x^2 - 4 * x - 1 = 0 ∧ (a - 3) * y^2 - 4 * y - 1 = 0) ↔
  (a > -1 ∧ a ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3740_374003


namespace NUMINAMATH_CALUDE_equation_satisfied_l3740_374029

theorem equation_satisfied (x y z : ℤ) (h1 : x = z + 1) (h2 : y = z) :
  x * (x - y) + y * (y - z) + z * (z - x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l3740_374029


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l3740_374045

/-- The number of bacteria after a given number of 30-second intervals -/
def bacteria_count (initial : ℕ) (intervals : ℕ) : ℕ :=
  initial * 4^intervals

/-- The theorem stating the initial number of bacteria -/
theorem initial_bacteria_count : 
  ∃ (initial : ℕ), bacteria_count initial 8 = 1048576 ∧ initial = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l3740_374045


namespace NUMINAMATH_CALUDE_east_region_difference_l3740_374031

/-- The difference in square miles between the regions east of two plains -/
def region_difference (total_area plain_B_area : ℕ) : ℕ :=
  plain_B_area - (total_area - plain_B_area)

/-- Theorem stating the difference between regions east of plain B and A -/
theorem east_region_difference :
  ∀ (total_area plain_B_area : ℕ),
  total_area = 350 →
  plain_B_area = 200 →
  region_difference total_area plain_B_area = 50 :=
by
  sorry

#eval region_difference 350 200

end NUMINAMATH_CALUDE_east_region_difference_l3740_374031


namespace NUMINAMATH_CALUDE_compare_sqrt_l3740_374067

theorem compare_sqrt : 2 * Real.sqrt 3 < 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_compare_sqrt_l3740_374067


namespace NUMINAMATH_CALUDE_shortest_paths_count_l3740_374035

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the grid and gas station locations -/
structure Grid where
  width : ℕ
  height : ℕ
  gasStations : List Point

/-- Represents the problem setup -/
structure ProblemSetup where
  grid : Grid
  start : Point
  finish : Point
  refuelDistance : ℕ

/-- Calculates the number of shortest paths between two points on a grid -/
def numberOfShortestPaths (start : Point) (finish : Point) : ℕ :=
  sorry

/-- Checks if a path is valid given the refuel constraints -/
def isValidPath (path : List Point) (gasStations : List Point) (refuelDistance : ℕ) : Bool :=
  sorry

/-- Main theorem: The number of shortest paths from A to B with refueling constraints is 24 -/
theorem shortest_paths_count (setup : ProblemSetup) : 
  (numberOfShortestPaths setup.start setup.finish) = 24 :=
sorry

end NUMINAMATH_CALUDE_shortest_paths_count_l3740_374035


namespace NUMINAMATH_CALUDE_min_output_no_loss_l3740_374099

/-- The total cost function for a product -/
def total_cost (x : ℕ) : ℚ :=
  3000 + 20 * x - 0.1 * x^2

/-- The condition for no loss -/
def no_loss (x : ℕ) : Prop :=
  25 * x ≥ total_cost x

/-- The theorem stating the minimum output for no loss -/
theorem min_output_no_loss :
  ∃ (x : ℕ), x > 0 ∧ x < 240 ∧ no_loss x ∧
  ∀ (y : ℕ), y > 0 ∧ y < 240 ∧ no_loss y → x ≤ y ∧ x = 150 :=
sorry

end NUMINAMATH_CALUDE_min_output_no_loss_l3740_374099


namespace NUMINAMATH_CALUDE_total_tables_proof_l3740_374083

/-- Represents the number of table styles -/
def num_styles : ℕ := 10

/-- Represents the sum of x values for all styles -/
def sum_x : ℕ := 100

/-- Calculates the total number of tables made in both months -/
def total_tables (num_styles : ℕ) (sum_x : ℕ) : ℕ :=
  num_styles * (2 * (sum_x / num_styles) - 3)

theorem total_tables_proof :
  total_tables num_styles sum_x = 170 :=
by sorry

end NUMINAMATH_CALUDE_total_tables_proof_l3740_374083


namespace NUMINAMATH_CALUDE_sqrt_12_div_sqrt_3_equals_2_l3740_374084

theorem sqrt_12_div_sqrt_3_equals_2 : Real.sqrt 12 / Real.sqrt 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_div_sqrt_3_equals_2_l3740_374084


namespace NUMINAMATH_CALUDE_odd_function_m_zero_l3740_374001

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- The function f(x) = 2x^3 + m -/
def f (m : ℝ) : ℝ → ℝ := fun x ↦ 2 * x^3 + m

theorem odd_function_m_zero :
  ∀ m : ℝ, IsOdd (f m) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_m_zero_l3740_374001


namespace NUMINAMATH_CALUDE_ellipse_equation_proof_l3740_374010

/-- The equation of the given ellipse -/
def given_ellipse (x y : ℝ) : Prop := 3 * x^2 + 8 * y^2 = 24

/-- The equation of the ellipse we want to prove -/
def target_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

/-- The foci of an ellipse with equation ax^2 + by^2 = c -/
def foci (a b c : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 = (1/a - 1/b) * c ∧ y = 0}

theorem ellipse_equation_proof :
  (∀ x y, given_ellipse x y ↔ 3 * x^2 + 8 * y^2 = 24) →
  (target_ellipse 3 2) →
  (foci 3 8 24 = foci (1/15) (1/10) 1) →
  ∀ x y, target_ellipse x y ↔ x^2 / 15 + y^2 / 10 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_proof_l3740_374010


namespace NUMINAMATH_CALUDE_jade_lego_tower_level_width_l3740_374085

/-- Calculates the width of each level in Jade's Lego tower -/
theorem jade_lego_tower_level_width 
  (initial_pieces : ℕ) 
  (levels : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : initial_pieces = 100)
  (h2 : levels = 11)
  (h3 : remaining_pieces = 23) :
  (initial_pieces - remaining_pieces) / levels = 7 := by
  sorry

end NUMINAMATH_CALUDE_jade_lego_tower_level_width_l3740_374085


namespace NUMINAMATH_CALUDE_sqrt_five_multiplication_l3740_374055

theorem sqrt_five_multiplication : 2 * Real.sqrt 5 * (3 * Real.sqrt 5) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_multiplication_l3740_374055


namespace NUMINAMATH_CALUDE_only_integer_solution_is_two_l3740_374059

theorem only_integer_solution_is_two :
  ∀ x : ℤ, (0 < (x - 1)^2 / (x + 1) ∧ (x - 1)^2 / (x + 1) < 1) ↔ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_only_integer_solution_is_two_l3740_374059


namespace NUMINAMATH_CALUDE_function_extrema_product_l3740_374038

theorem function_extrema_product (a b : Real) :
  let f := fun x => a - Real.sqrt 3 * Real.tan (2 * x)
  (∀ x ∈ Set.Icc (-Real.pi/6) b, f x ≤ 7) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) b, f x ≥ 3) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) b, f x = 7) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) b, f x = 3) →
  a * b = Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_product_l3740_374038


namespace NUMINAMATH_CALUDE_jackie_has_six_apples_l3740_374013

/-- The number of apples Adam has -/
def adam_apples : ℕ := 9

/-- The difference between Adam's and Jackie's apples -/
def difference : ℕ := 3

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := adam_apples - difference

theorem jackie_has_six_apples : jackie_apples = 6 := by sorry

end NUMINAMATH_CALUDE_jackie_has_six_apples_l3740_374013


namespace NUMINAMATH_CALUDE_smaller_circle_circumference_l3740_374018

theorem smaller_circle_circumference 
  (square_area : ℝ) 
  (larger_radius : ℝ) 
  (smaller_radius : ℝ) 
  (h1 : square_area = 784) 
  (h2 : square_area = (2 * larger_radius)^2) 
  (h3 : larger_radius = (7/3) * smaller_radius) : 
  2 * Real.pi * smaller_radius = 12 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_smaller_circle_circumference_l3740_374018


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3740_374023

theorem divisibility_equivalence (m n : ℕ+) :
  83 ∣ (25 * m + 3 * n) ↔ 83 ∣ (3 * m + 7 * n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3740_374023


namespace NUMINAMATH_CALUDE_angle_bisector_length_l3740_374060

/-- Given a triangle ABC with sides b and c, and angle A between them,
    prove that the length of the angle bisector of A is (2bc cos(A/2)) / (b + c) -/
theorem angle_bisector_length (b c A : ℝ) (hb : b > 0) (hc : c > 0) (hA : 0 < A ∧ A < π) :
  let S := (1/2) * b * c * Real.sin A
  let l_a := (2 * b * c * Real.cos (A/2)) / (b + c)
  ∀ S', S' = S → l_a = (2 * b * c * Real.cos (A/2)) / (b + c) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l3740_374060


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3740_374081

theorem simplify_trig_expression : 
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) / 
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) = 
  Real.tan (45 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3740_374081


namespace NUMINAMATH_CALUDE_max_value_problem_1_l3740_374096

theorem max_value_problem_1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  ∃ (max_y : ℝ), ∀ y : ℝ, y = 1/2 * x * (1 - 2*x) → y ≤ max_y ∧ max_y = 1/16 := by
  sorry


end NUMINAMATH_CALUDE_max_value_problem_1_l3740_374096


namespace NUMINAMATH_CALUDE_range_of_a_l3740_374030

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- State the theorem
theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3740_374030


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3740_374094

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 77) : x^2 - y^2 = 5 / 847 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3740_374094


namespace NUMINAMATH_CALUDE_emilys_skirt_cost_l3740_374021

theorem emilys_skirt_cost (art_supplies_cost total_cost : ℕ) (num_skirts : ℕ) 
  (h1 : art_supplies_cost = 20)
  (h2 : num_skirts = 2)
  (h3 : total_cost = 50) :
  ∃ (skirt_cost : ℕ), skirt_cost * num_skirts + art_supplies_cost = total_cost ∧ skirt_cost = 15 :=
by
  sorry

#check emilys_skirt_cost

end NUMINAMATH_CALUDE_emilys_skirt_cost_l3740_374021


namespace NUMINAMATH_CALUDE_comparison_and_inequality_l3740_374093

theorem comparison_and_inequality (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  (2 * x^2 + y^2 > x^2 + x * y) ∧ (Real.sqrt 6 - Real.sqrt 5 < 2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_comparison_and_inequality_l3740_374093


namespace NUMINAMATH_CALUDE_f_composition_equals_8c_implies_c_equals_1_l3740_374008

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 199^x + 1 else x^2 + 2*c*x

theorem f_composition_equals_8c_implies_c_equals_1 (c : ℝ) :
  f c (f c 0) = 8*c → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_8c_implies_c_equals_1_l3740_374008


namespace NUMINAMATH_CALUDE_cubic_cm_in_cubic_meter_proof_l3740_374061

/-- The number of cubic centimeters in one cubic meter -/
def cubic_cm_in_cubic_meter : ℕ := 1000000

/-- The number of centimeters in one meter -/
def cm_in_meter : ℕ := 100

/-- Theorem stating that the number of cubic centimeters in one cubic meter is 1,000,000,
    given that one meter is equal to one hundred centimeters -/
theorem cubic_cm_in_cubic_meter_proof :
  cubic_cm_in_cubic_meter = cm_in_meter ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_cm_in_cubic_meter_proof_l3740_374061


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l3740_374073

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l3740_374073


namespace NUMINAMATH_CALUDE_triangle_side_length_l3740_374068

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ

-- State the theorem
theorem triangle_side_length (t : Triangle) :
  (t.b - t.c = 2) →
  (1/2 * t.b * t.c * Real.sqrt (1 - (-1/4)^2) = 3 * Real.sqrt 15) →
  (Real.cos t.A = -1/4) →
  t.a = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3740_374068


namespace NUMINAMATH_CALUDE_orange_stack_count_l3740_374072

/-- Calculates the number of oranges in a triangular layer -/
def orangesInLayer (a b : ℕ) : ℕ := (a * b) / 2

/-- Calculates the total number of oranges in the stack -/
def totalOranges (baseWidth baseLength : ℕ) : ℕ :=
  let rec sumLayers (width length : ℕ) : ℕ :=
    if width = 0 ∨ length = 0 then 0
    else orangesInLayer width length + sumLayers (width - 1) (length - 1)
  sumLayers baseWidth baseLength

theorem orange_stack_count :
  totalOranges 6 9 = 78 := by
  sorry

#eval totalOranges 6 9  -- Should output 78

end NUMINAMATH_CALUDE_orange_stack_count_l3740_374072


namespace NUMINAMATH_CALUDE_polynomial_degree_condition_l3740_374070

theorem polynomial_degree_condition (k m : ℝ) : 
  (∀ x, ∃ a b, (k - 1) * x^2 + 4 * x - m = a * x + b) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_condition_l3740_374070


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l3740_374046

/-- Calculate the average speed of a round trip flight with wind vectors -/
theorem round_trip_average_speed 
  (speed_to_mother : ℝ) 
  (tailwind_speed : ℝ) 
  (tailwind_angle : ℝ) 
  (speed_to_home : ℝ) 
  (headwind_speed : ℝ) 
  (headwind_angle : ℝ) 
  (h1 : speed_to_mother = 96) 
  (h2 : tailwind_speed = 12) 
  (h3 : tailwind_angle = 30 * π / 180) 
  (h4 : speed_to_home = 88) 
  (h5 : headwind_speed = 15) 
  (h6 : headwind_angle = 60 * π / 180) : 
  ∃ (average_speed : ℝ), 
    abs (average_speed - 93.446) < 0.001 ∧ 
    average_speed = (
      (speed_to_mother + tailwind_speed * Real.cos tailwind_angle) + 
      (speed_to_home - headwind_speed * Real.cos headwind_angle)
    ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l3740_374046


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l3740_374027

theorem complement_intersection_problem (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5} →
  A = {2, 3, 4} →
  B = {3, 4, 5} →
  (Aᶜ ∩ Bᶜ : Set ℕ) = {1, 2, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l3740_374027


namespace NUMINAMATH_CALUDE_nina_taller_than_lena_probability_zero_l3740_374058

-- Define the set of friends
inductive Friend : Type
| Masha : Friend
| Nina : Friend
| Lena : Friend
| Olya : Friend

-- Define a height ordering relation
def TallerThan : Friend → Friend → Prop :=
  sorry

-- Define the conditions
axiom all_different_heights :
  ∀ (a b : Friend), a ≠ b → (TallerThan a b ∨ TallerThan b a)

axiom nina_shorter_than_masha :
  TallerThan Friend.Masha Friend.Nina

axiom lena_taller_than_olya :
  TallerThan Friend.Lena Friend.Olya

-- Define the probability function
def Probability (event : Prop) : ℚ :=
  sorry

-- The theorem to prove
theorem nina_taller_than_lena_probability_zero :
  Probability (TallerThan Friend.Nina Friend.Lena) = 0 :=
sorry

end NUMINAMATH_CALUDE_nina_taller_than_lena_probability_zero_l3740_374058


namespace NUMINAMATH_CALUDE_portfolio_growth_portfolio_growth_example_l3740_374012

theorem portfolio_growth (initial_investment : ℝ) (first_year_rate : ℝ) 
  (additional_investment : ℝ) (second_year_rate : ℝ) : ℝ :=
  let first_year_value := initial_investment * (1 + first_year_rate)
  let second_year_initial := first_year_value + additional_investment
  let final_value := second_year_initial * (1 + second_year_rate)
  final_value

theorem portfolio_growth_example : 
  portfolio_growth 80 0.15 28 0.10 = 132 := by
  sorry

end NUMINAMATH_CALUDE_portfolio_growth_portfolio_growth_example_l3740_374012


namespace NUMINAMATH_CALUDE_pm25_scientific_notation_l3740_374097

theorem pm25_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 0.0000025 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ n = -6 ∧ a = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_pm25_scientific_notation_l3740_374097


namespace NUMINAMATH_CALUDE_prime_sequence_extension_l3740_374079

theorem prime_sequence_extension (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) := by
sorry

end NUMINAMATH_CALUDE_prime_sequence_extension_l3740_374079


namespace NUMINAMATH_CALUDE_composition_central_symmetries_is_translation_composition_translation_central_symmetry_is_central_symmetry_l3740_374040

-- Define the types for our transformations
def CentralSymmetry (center : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry
def Translation (vector : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

-- Define composition of transformations
def Compose (f g : (ℝ × ℝ) → (ℝ × ℝ)) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

-- Theorem 1: Composition of two central symmetries is a translation
theorem composition_central_symmetries_is_translation 
  (c1 c2 : ℝ × ℝ) : 
  ∃ (v : ℝ × ℝ), Compose (CentralSymmetry c2) (CentralSymmetry c1) = Translation v := by sorry

-- Theorem 2: Composition of translation and central symmetry (both orders) is a central symmetry
theorem composition_translation_central_symmetry_is_central_symmetry 
  (v : ℝ × ℝ) (c : ℝ × ℝ) : 
  (∃ (c1 : ℝ × ℝ), Compose (Translation v) (CentralSymmetry c) = CentralSymmetry c1) ∧
  (∃ (c2 : ℝ × ℝ), Compose (CentralSymmetry c) (Translation v) = CentralSymmetry c2) := by sorry

end NUMINAMATH_CALUDE_composition_central_symmetries_is_translation_composition_translation_central_symmetry_is_central_symmetry_l3740_374040


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3740_374039

theorem nested_fraction_evaluation : 
  let f (x : ℝ) := (x + 2) / (x - 2)
  let g (x : ℝ) := (f x + 2) / (f x - 2)
  g 1 = 1/5 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3740_374039


namespace NUMINAMATH_CALUDE_helium_cost_per_ounce_l3740_374019

-- Define the constants
def total_money : ℚ := 200
def sheet_cost : ℚ := 42
def rope_cost : ℚ := 18
def propane_cost : ℚ := 14
def height_per_ounce : ℚ := 113
def max_height : ℚ := 9492

-- Define the theorem
theorem helium_cost_per_ounce :
  let money_left := total_money - (sheet_cost + rope_cost + propane_cost)
  let ounces_needed := max_height / height_per_ounce
  let cost_per_ounce := money_left / ounces_needed
  cost_per_ounce = 3/2 := by sorry

end NUMINAMATH_CALUDE_helium_cost_per_ounce_l3740_374019


namespace NUMINAMATH_CALUDE_simplified_fourth_root_l3740_374088

theorem simplified_fourth_root (c d : ℕ+) :
  (3^5 * 5^3 : ℝ)^(1/4) = c * d^(1/4) → c + d = 3378 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fourth_root_l3740_374088


namespace NUMINAMATH_CALUDE_final_crayon_count_l3740_374074

def crayon_count (initial : ℕ) (added1 : ℕ) (removed : ℕ) (added2 : ℕ) : ℕ :=
  initial + added1 - removed + added2

theorem final_crayon_count :
  crayon_count 25 15 8 12 = 44 := by
  sorry

end NUMINAMATH_CALUDE_final_crayon_count_l3740_374074


namespace NUMINAMATH_CALUDE_security_deposit_is_1110_l3740_374098

/-- Calculates the security deposit for a cabin rental -/
def calculate_security_deposit (daily_rate : ℚ) (duration : ℕ) (pet_fee : ℚ) 
  (service_fee_rate : ℚ) (deposit_rate : ℚ) : ℚ :=
  let subtotal := daily_rate * duration + pet_fee
  let service_fee := service_fee_rate * subtotal
  let total := subtotal + service_fee
  deposit_rate * total

/-- Theorem stating that the security deposit for the given conditions is $1110.00 -/
theorem security_deposit_is_1110 :
  calculate_security_deposit 125 14 100 (1/5) (1/2) = 1110 := by
  sorry

#eval calculate_security_deposit 125 14 100 (1/5) (1/2)

end NUMINAMATH_CALUDE_security_deposit_is_1110_l3740_374098


namespace NUMINAMATH_CALUDE_specific_parallelogram_area_l3740_374014

/-- A parallelogram in 2D space -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := sorry

/-- The specific parallelogram in the problem -/
def specificParallelogram : Parallelogram := {
  v1 := (0, 0)
  v2 := (4, 0)
  v3 := (1, 5)
  v4 := (5, 5)
}

/-- Theorem: The area of the specific parallelogram is 20 square units -/
theorem specific_parallelogram_area :
  parallelogramArea specificParallelogram = 20 := by sorry

end NUMINAMATH_CALUDE_specific_parallelogram_area_l3740_374014


namespace NUMINAMATH_CALUDE_range_of_a_l3740_374006

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → |x^2 - a| + |x + a| = |x^2 + x|) → 
  a ∈ Set.Icc (-1) 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3740_374006


namespace NUMINAMATH_CALUDE_max_correct_answers_l3740_374090

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 30 →
  correct_points = 4 →
  incorrect_points = -3 →
  total_score = 54 →
  ∃ (correct incorrect blank : ℕ),
    correct + incorrect + blank = total_questions ∧
    correct * correct_points + incorrect * incorrect_points = total_score ∧
    correct ≤ 20 ∧
    ∀ (c : ℕ), c > 20 →
      ¬∃ (i b : ℕ), c + i + b = total_questions ∧
                    c * correct_points + i * incorrect_points = total_score :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l3740_374090
