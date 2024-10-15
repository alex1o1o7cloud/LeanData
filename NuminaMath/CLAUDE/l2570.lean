import Mathlib

namespace NUMINAMATH_CALUDE_mary_remaining_money_l2570_257020

def remaining_money (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 4 * p
  let total_cost := 5 * drink_cost + medium_pizza_cost + large_pizza_cost
  30 - total_cost

theorem mary_remaining_money (p : ℝ) :
  remaining_money p = 30 - 12 * p :=
by sorry

end NUMINAMATH_CALUDE_mary_remaining_money_l2570_257020


namespace NUMINAMATH_CALUDE_table_tennis_matches_l2570_257073

theorem table_tennis_matches (player1_matches player2_matches : ℕ) 
  (h1 : player1_matches = 10) 
  (h2 : player2_matches = 21) : ℕ := by
  -- The number of matches the third player played
  sorry

#check table_tennis_matches

end NUMINAMATH_CALUDE_table_tennis_matches_l2570_257073


namespace NUMINAMATH_CALUDE_paco_cookies_l2570_257036

/-- Calculates the number of cookies Paco bought given the initial, eaten, and final cookie counts. -/
def cookies_bought (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

/-- Proves that Paco bought 37 cookies given the problem conditions. -/
theorem paco_cookies : cookies_bought 40 2 75 = 37 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l2570_257036


namespace NUMINAMATH_CALUDE_perimeter_of_specific_quadrilateral_l2570_257035

structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  h_positive : 0 < AB ∧ 0 < BC ∧ 0 < CD ∧ 0 < DA

def perimeter (q : Quadrilateral) : ℝ :=
  q.AB + q.BC + q.CD + q.DA

theorem perimeter_of_specific_quadrilateral :
  ∃ (q : Quadrilateral), 
    q.DA < q.BC ∧
    q.DA = 4 ∧
    q.AB = 5 ∧
    q.BC = 10 ∧
    q.CD = 7 ∧
    perimeter q = 26 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_specific_quadrilateral_l2570_257035


namespace NUMINAMATH_CALUDE_constant_order_magnitude_l2570_257033

theorem constant_order_magnitude (k : ℕ) (h : k > 4) :
  k + 2 < 2 * k ∧ 2 * k < k^2 ∧ k^2 < 2^k := by
  sorry

end NUMINAMATH_CALUDE_constant_order_magnitude_l2570_257033


namespace NUMINAMATH_CALUDE_train_length_calculation_l2570_257065

-- Define the given values
def train_speed : Real := 63  -- km/hr
def man_speed : Real := 3     -- km/hr
def crossing_time : Real := 29.997600191984642  -- seconds

-- Define the theorem
theorem train_length_calculation :
  let relative_speed := (train_speed - man_speed) * 1000 / 3600  -- Convert to m/s
  let train_length := relative_speed * crossing_time
  ∃ ε > 0, abs (train_length - 500) < ε :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2570_257065


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l2570_257095

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific conditions, 
    the man's speed against the current is 16 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 21 2.5 = 16 := by
  sorry

#eval speed_against_current 21 2.5

end NUMINAMATH_CALUDE_mans_speed_against_current_l2570_257095


namespace NUMINAMATH_CALUDE_subset_implies_membership_l2570_257013

theorem subset_implies_membership (P Q : Set α) 
  (h_nonempty_P : P.Nonempty) (h_nonempty_Q : Q.Nonempty) (h_subset : P ⊆ Q) : 
  ∀ x ∈ P, x ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_membership_l2570_257013


namespace NUMINAMATH_CALUDE_least_number_to_add_for_divisibility_l2570_257098

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem least_number_to_add_for_divisibility :
  ∃ (p : ℕ) (h : is_two_digit_prime p), ∀ (k : ℕ), k < 1 → ¬(∃ (q : ℕ), is_two_digit_prime q ∧ (54321 + k) % q = 0) :=
sorry

end NUMINAMATH_CALUDE_least_number_to_add_for_divisibility_l2570_257098


namespace NUMINAMATH_CALUDE_product_mod_five_l2570_257030

theorem product_mod_five : (2023 * 2024 * 2025 * 2026) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_five_l2570_257030


namespace NUMINAMATH_CALUDE_equation_solution_for_all_y_l2570_257079

theorem equation_solution_for_all_y :
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_for_all_y_l2570_257079


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l2570_257063

theorem sqrt_difference_equality : 3 * Real.sqrt 5 - Real.sqrt 20 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l2570_257063


namespace NUMINAMATH_CALUDE_xiao_hong_books_l2570_257034

/-- Given that Xiao Hong originally had 5 books and bought 'a' more books,
    prove that her total number of books now is 5 + a. -/
theorem xiao_hong_books (a : ℕ) : 5 + a = 5 + a := by sorry

end NUMINAMATH_CALUDE_xiao_hong_books_l2570_257034


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2570_257027

theorem cube_root_equation_solution :
  ∃ y : ℝ, (5 + 2 / y)^(1/3) = -3 ↔ y = -1/16 := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2570_257027


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l2570_257052

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Define the point of tangency
def point : ℝ × ℝ := (2, 2)

-- Define the two possible tangent lines
def line1 (x : ℝ) : ℝ := 2
def line2 (x : ℝ) : ℝ := 9*x - 16

theorem tangent_line_at_point :
  (∀ x, line1 x = f x → x = 2) ∧
  (∀ x, line2 x = f x → x = 2) ∧
  line1 (point.1) = point.2 ∧
  line2 (point.1) = point.2 ∧
  f' (point.1) = 9 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l2570_257052


namespace NUMINAMATH_CALUDE_buffalo_count_l2570_257086

/-- A group of animals consisting of buffaloes and ducks -/
structure AnimalGroup where
  buffaloes : ℕ
  ducks : ℕ

/-- The total number of legs in the group -/
def total_legs (group : AnimalGroup) : ℕ := 4 * group.buffaloes + 2 * group.ducks

/-- The total number of heads in the group -/
def total_heads (group : AnimalGroup) : ℕ := group.buffaloes + group.ducks

/-- The main theorem: there are 12 buffaloes in the group -/
theorem buffalo_count (group : AnimalGroup) : 
  (total_legs group = 2 * total_heads group + 24) → group.buffaloes = 12 := by
  sorry

end NUMINAMATH_CALUDE_buffalo_count_l2570_257086


namespace NUMINAMATH_CALUDE_geometric_arithmetic_geometric_sequence_l2570_257031

theorem geometric_arithmetic_geometric_sequence 
  (a b c : ℝ) 
  (h1 : b ^ 2 = a * c)  -- geometric progression condition
  (h2 : b + 2 = (a + c) / 2)  -- arithmetic progression condition
  (h3 : (b + 2) ^ 2 = a * (c + 16))  -- second geometric progression condition
  : (a = 1 ∧ b = 3 ∧ c = 9) ∨ (a = 1/9 ∧ b = -5/9 ∧ c = 25/9) := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_geometric_sequence_l2570_257031


namespace NUMINAMATH_CALUDE_smallest_number_above_50_with_conditions_fifty_one_satisfies_conditions_fifty_one_is_answer_l2570_257087

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def count_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_number_above_50_with_conditions : 
  ∀ n : ℕ, n > 50 → n < 51 → 
  (¬ is_perfect_square n ∨ count_factors n % 2 = 1 ∨ n % 3 ≠ 0) :=
by sorry

theorem fifty_one_satisfies_conditions : 
  ¬ is_perfect_square 51 ∧ count_factors 51 % 2 = 0 ∧ 51 % 3 = 0 :=
by sorry

theorem fifty_one_is_answer : 
  ∀ n : ℕ, n > 50 → n < 51 → 
  (¬ is_perfect_square n ∨ count_factors n % 2 = 1 ∨ n % 3 ≠ 0) ∧
  (¬ is_perfect_square 51 ∧ count_factors 51 % 2 = 0 ∧ 51 % 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_above_50_with_conditions_fifty_one_satisfies_conditions_fifty_one_is_answer_l2570_257087


namespace NUMINAMATH_CALUDE_solution_to_equation_l2570_257022

theorem solution_to_equation (x : ℝ) : -200 * x = 1600 → x = -8 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2570_257022


namespace NUMINAMATH_CALUDE_toms_balloons_l2570_257091

theorem toms_balloons (sara_balloons : ℕ) (total_balloons : ℕ) 
  (h1 : sara_balloons = 8)
  (h2 : total_balloons = 17) :
  total_balloons - sara_balloons = 9 := by
  sorry

end NUMINAMATH_CALUDE_toms_balloons_l2570_257091


namespace NUMINAMATH_CALUDE_jack_morning_emails_l2570_257047

theorem jack_morning_emails (afternoon_emails : ℕ) (morning_afternoon_difference : ℕ) : 
  afternoon_emails = 2 → 
  morning_afternoon_difference = 4 →
  afternoon_emails + morning_afternoon_difference = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l2570_257047


namespace NUMINAMATH_CALUDE_cost_per_sqm_intersecting_roads_l2570_257080

/-- The cost per square meter for traveling two intersecting roads on a rectangular lawn. -/
theorem cost_per_sqm_intersecting_roads 
  (lawn_length : ℝ) 
  (lawn_width : ℝ) 
  (road_width : ℝ) 
  (total_cost : ℝ) : 
  lawn_length = 80 ∧ 
  lawn_width = 40 ∧ 
  road_width = 10 ∧ 
  total_cost = 3300 → 
  (total_cost / ((lawn_length * road_width + lawn_width * road_width) - road_width * road_width)) = 3 := by
sorry

end NUMINAMATH_CALUDE_cost_per_sqm_intersecting_roads_l2570_257080


namespace NUMINAMATH_CALUDE_frank_final_position_l2570_257050

/-- Calculates Frank's final position after a series of dance moves --/
def frankPosition (initialBackSteps : ℤ) (firstForwardSteps : ℤ) (secondBackSteps : ℤ) : ℤ :=
  -initialBackSteps + firstForwardSteps - secondBackSteps + 2 * secondBackSteps

/-- Proves that Frank ends up 7 steps forward from his original starting point --/
theorem frank_final_position :
  frankPosition 5 10 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_frank_final_position_l2570_257050


namespace NUMINAMATH_CALUDE_sin_cos_sum_l2570_257072

theorem sin_cos_sum (θ : ℝ) (h : Real.sin θ ^ 3 + Real.cos θ ^ 3 = 11/16) : 
  Real.sin θ + Real.cos θ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_l2570_257072


namespace NUMINAMATH_CALUDE_odd_function_value_at_negative_one_l2570_257014

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value_at_negative_one
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_nonneg : ∀ x ≥ 0, f x = 2^x + 2*x + f 0)
  : f (-1) = -3 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_value_at_negative_one_l2570_257014


namespace NUMINAMATH_CALUDE_liya_number_preference_l2570_257076

theorem liya_number_preference (n : ℕ) : 
  (n % 3 = 0) ∧ (n % 10 = 0) → n % 10 = 0 := by
sorry

end NUMINAMATH_CALUDE_liya_number_preference_l2570_257076


namespace NUMINAMATH_CALUDE_min_value_problem_l2570_257003

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, a * x + b * y = 1 → x^2 + y^2 - 2*x - 2*y - 2 = 0 → x = 1 ∧ y = 1) →
  (∀ c d : ℝ, c > 0 → d > 0 → c * 1 + d * 1 = 1 → 1/c + 2/d ≥ 1/a + 2/b) →
  1/a + 2/b = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2570_257003


namespace NUMINAMATH_CALUDE_danny_jane_age_difference_l2570_257042

theorem danny_jane_age_difference : ∃ (x : ℝ), 
  (40 : ℝ) - x = (4.5 : ℝ) * ((26 : ℝ) - x) ∧ x = 22 := by
  sorry

end NUMINAMATH_CALUDE_danny_jane_age_difference_l2570_257042


namespace NUMINAMATH_CALUDE_permutations_of_seven_distinct_objects_l2570_257085

theorem permutations_of_seven_distinct_objects : Nat.factorial 7 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_seven_distinct_objects_l2570_257085


namespace NUMINAMATH_CALUDE_discount_from_profit_l2570_257009

/-- Represents a car sale transaction -/
structure CarSale where
  originalPrice : ℝ
  discountRate : ℝ
  profitRate : ℝ
  sellIncrease : ℝ

/-- Theorem stating the relationship between discount and profit in a specific car sale scenario -/
theorem discount_from_profit (sale : CarSale) 
  (h1 : sale.profitRate = 0.28000000000000004)
  (h2 : sale.sellIncrease = 0.60) : 
  sale.discountRate = 0.5333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_discount_from_profit_l2570_257009


namespace NUMINAMATH_CALUDE_cyclist_problem_l2570_257090

theorem cyclist_problem (v t : ℝ) 
  (h1 : (v + 3) * (t - 1) = v * t)
  (h2 : (v - 2) * (t + 1) = v * t) :
  v * t = 60 ∧ v = 12 ∧ t = 5 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_problem_l2570_257090


namespace NUMINAMATH_CALUDE_smallest_equal_burgers_and_buns_l2570_257096

theorem smallest_equal_burgers_and_buns :
  ∃ n : ℕ+, (∀ k : ℕ+, (∃ m : ℕ+, 5 * k = 7 * m) → n ≤ k) ∧ (∃ m : ℕ+, 5 * n = 7 * m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_equal_burgers_and_buns_l2570_257096


namespace NUMINAMATH_CALUDE_divisibility_of_3_105_plus_4_105_l2570_257089

theorem divisibility_of_3_105_plus_4_105 :
  let n : ℕ := 3^105 + 4^105
  (∃ k : ℕ, n = 13 * k) ∧
  (∃ k : ℕ, n = 49 * k) ∧
  (∃ k : ℕ, n = 181 * k) ∧
  (∃ k : ℕ, n = 379 * k) ∧
  (∀ k : ℕ, n ≠ 5 * k) ∧
  (∀ k : ℕ, n ≠ 11 * k) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_3_105_plus_4_105_l2570_257089


namespace NUMINAMATH_CALUDE_color_film_fraction_l2570_257026

theorem color_film_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw := 20 * x
  let total_color := 8 * y
  let selected_bw := y / 5
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  selected_color / total_selected = 40 / 41 := by
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l2570_257026


namespace NUMINAMATH_CALUDE_point_satisfies_conditions_l2570_257088

/-- A point on a line with equal distance to coordinate axes -/
def point_on_line_equal_distance (x y : ℝ) : Prop :=
  y = -2 * x + 2 ∧ (x = y ∨ x = -y)

/-- The point (2/3, 2/3) satisfies the conditions -/
theorem point_satisfies_conditions : point_on_line_equal_distance (2/3) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_point_satisfies_conditions_l2570_257088


namespace NUMINAMATH_CALUDE_P_in_first_quadrant_l2570_257007

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : CartesianPoint) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point P(3,2) -/
def P : CartesianPoint :=
  { x := 3, y := 2 }

/-- Theorem: Point P(3,2) lies in the first quadrant -/
theorem P_in_first_quadrant : isInFirstQuadrant P := by
  sorry


end NUMINAMATH_CALUDE_P_in_first_quadrant_l2570_257007


namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l2570_257025

theorem existence_of_special_numbers :
  ∃ (a b c : ℕ),
    a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
    (a * b * c) % (a + 2012) = 0 ∧
    (a * b * c) % (b + 2012) = 0 ∧
    (a * b * c) % (c + 2012) = 0 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l2570_257025


namespace NUMINAMATH_CALUDE_women_science_majors_percentage_l2570_257023

theorem women_science_majors_percentage
  (non_science_percentage : Real)
  (men_percentage : Real)
  (men_science_percentage : Real)
  (h1 : non_science_percentage = 0.6)
  (h2 : men_percentage = 0.4)
  (h3 : men_science_percentage = 0.5500000000000001) :
  let women_percentage := 1 - men_percentage
  let total_science_percentage := 1 - non_science_percentage
  let men_science_total_percentage := men_percentage * men_science_percentage
  let women_science_total_percentage := total_science_percentage - men_science_total_percentage
  women_science_total_percentage / women_percentage = 0.29999999999999993 :=
by sorry

end NUMINAMATH_CALUDE_women_science_majors_percentage_l2570_257023


namespace NUMINAMATH_CALUDE_sarah_test_result_l2570_257071

/-- Represents a math test with a number of problems and a score percentage -/
structure MathTest where
  problems : ℕ
  score : ℚ
  score_valid : 0 ≤ score ∧ score ≤ 1

/-- Calculates the number of correctly answered problems in a test -/
def correctProblems (test : MathTest) : ℚ :=
  test.problems * test.score

/-- Calculates the overall percentage of correctly answered problems across multiple tests -/
def overallPercentage (tests : List MathTest) : ℚ :=
  let totalCorrect := (tests.map correctProblems).sum
  let totalProblems := (tests.map (·.problems)).sum
  totalCorrect / totalProblems

theorem sarah_test_result : 
  let test1 : MathTest := { problems := 30, score := 85/100, score_valid := by norm_num }
  let test2 : MathTest := { problems := 50, score := 75/100, score_valid := by norm_num }
  let test3 : MathTest := { problems := 20, score := 80/100, score_valid := by norm_num }
  let tests := [test1, test2, test3]
  overallPercentage tests = 78/100 := by
  sorry

end NUMINAMATH_CALUDE_sarah_test_result_l2570_257071


namespace NUMINAMATH_CALUDE_ram_money_calculation_l2570_257094

theorem ram_money_calculation (ram gopal krishan : ℕ) 
  (h1 : ram * 17 = gopal * 7)
  (h2 : gopal * 17 = krishan * 7)
  (h3 : krishan = 4335) : 
  ram = 735 := by
sorry

end NUMINAMATH_CALUDE_ram_money_calculation_l2570_257094


namespace NUMINAMATH_CALUDE_two_distinct_negative_roots_l2570_257021

/-- The polynomial function for which we're finding roots -/
def f (p : ℝ) (x : ℝ) : ℝ := x^4 + 4*p*x^3 + 2*x^2 + 4*p*x + 1

/-- A root of the polynomial is a real number x such that f p x = 0 -/
def is_root (p : ℝ) (x : ℝ) : Prop := f p x = 0

/-- A function to represent that a real number is negative -/
def is_negative (x : ℝ) : Prop := x < 0

/-- The main theorem stating that for p > 1, there are at least two distinct negative real roots -/
theorem two_distinct_negative_roots (p : ℝ) (hp : p > 1) :
  ∃ (x y : ℝ), x ≠ y ∧ is_negative x ∧ is_negative y ∧ is_root p x ∧ is_root p y :=
sorry

end NUMINAMATH_CALUDE_two_distinct_negative_roots_l2570_257021


namespace NUMINAMATH_CALUDE_quadratic_solution_square_l2570_257004

theorem quadratic_solution_square (y : ℝ) (h : 3 * y^2 + 2 = 7 * y + 15) : (6 * y - 5)^2 = 269 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_square_l2570_257004


namespace NUMINAMATH_CALUDE_solution_set_theorem_range_of_m_l2570_257070

-- Define the function f(x) = |x - 2|
def f (x : ℝ) : ℝ := |x - 2|

-- Theorem for the solution set of f(x) + f(2x + 1) ≥ 6
theorem solution_set_theorem (x : ℝ) :
  f x + f (2 * x + 1) ≥ 6 ↔ x ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
sorry

-- Theorem for the range of m
theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x : ℝ, f (x - m) - (-x) ≤ 4/a + 1/b) →
  -13 ≤ m ∧ m ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_range_of_m_l2570_257070


namespace NUMINAMATH_CALUDE_egypt_tour_promotion_l2570_257002

theorem egypt_tour_promotion (total_tourists : ℕ) (free_tourists : ℕ) : 
  (13 : ℕ) + 4 * free_tourists = total_tourists ∧ 
  free_tourists + (100 : ℕ) = total_tourists →
  free_tourists = 29 := by
sorry

end NUMINAMATH_CALUDE_egypt_tour_promotion_l2570_257002


namespace NUMINAMATH_CALUDE_angle_relationship_l2570_257057

theorem angle_relationship (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = (1/2) * Real.sin (α + β)) : α < β := by
  sorry

end NUMINAMATH_CALUDE_angle_relationship_l2570_257057


namespace NUMINAMATH_CALUDE_bananas_per_box_l2570_257037

/-- Given 40 bananas and 8 boxes, prove that 5 bananas must go in each box. -/
theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) 
  (h1 : total_bananas = 40) (h2 : num_boxes = 8) : 
  total_bananas / num_boxes = 5 := by
  sorry

#check bananas_per_box

end NUMINAMATH_CALUDE_bananas_per_box_l2570_257037


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2570_257081

-- Define the sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {-2, -1, 0, 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2570_257081


namespace NUMINAMATH_CALUDE_cody_money_l2570_257097

def final_money (initial : ℕ) (birthday : ℕ) (game_cost : ℕ) : ℕ :=
  initial + birthday - game_cost

theorem cody_money : final_money 45 9 19 = 35 := by
  sorry

end NUMINAMATH_CALUDE_cody_money_l2570_257097


namespace NUMINAMATH_CALUDE_triangle_classification_l2570_257001

/-- Checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem triangle_classification :
  ¬(is_right_triangle 1.5 2 3) ∧
  (is_right_triangle 7 24 25) ∧
  (is_right_triangle 3 4 5) ∧
  (is_right_triangle 9 12 15) :=
by sorry

end NUMINAMATH_CALUDE_triangle_classification_l2570_257001


namespace NUMINAMATH_CALUDE_inequality_expression_l2570_257016

theorem inequality_expression (x : ℝ) : (x + 4 ≥ -1) ↔ (x + 4 ≥ -1) := by sorry

end NUMINAMATH_CALUDE_inequality_expression_l2570_257016


namespace NUMINAMATH_CALUDE_crazy_silly_school_remaining_books_l2570_257006

/-- Given a book series with a total number of books and a number of books already read,
    calculate the number of books remaining to be read. -/
def books_remaining (total : ℕ) (read : ℕ) : ℕ := total - read

/-- Theorem stating that for the 'crazy silly school' series with 14 total books
    and 8 books already read, there are 6 books remaining to be read. -/
theorem crazy_silly_school_remaining_books :
  books_remaining 14 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_remaining_books_l2570_257006


namespace NUMINAMATH_CALUDE_hamiltonian_cycle_with_at_most_one_color_change_l2570_257043

/-- A complete graph with n vertices where each edge is colored either red or blue -/
structure ColoredCompleteGraph (n : ℕ) where
  vertices : Fin n → Type
  edge_color : ∀ (i j : Fin n), i ≠ j → Bool

/-- A Hamiltonian cycle in the graph -/
def HamiltonianCycle (n : ℕ) (G : ColoredCompleteGraph n) :=
  { cycle : List (Fin n) // cycle.length = n ∧ cycle.Nodup }

/-- The number of color changes in a Hamiltonian cycle -/
def ColorChanges (n : ℕ) (G : ColoredCompleteGraph n) (cycle : HamiltonianCycle n G) : ℕ :=
  sorry

/-- Theorem: There exists a Hamiltonian cycle with at most one color change -/
theorem hamiltonian_cycle_with_at_most_one_color_change (n : ℕ) (G : ColoredCompleteGraph n) :
  ∃ (cycle : HamiltonianCycle n G), ColorChanges n G cycle ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_hamiltonian_cycle_with_at_most_one_color_change_l2570_257043


namespace NUMINAMATH_CALUDE_equation_solution_iff_common_root_l2570_257028

theorem equation_solution_iff_common_root
  (a : ℝ) (f g h : ℝ → ℝ) 
  (ha : a > 1)
  (h_sum_nonneg : ∀ x, f x + g x + h x ≥ 0) :
  (∃ x, a^(f x) + a^(g x) + a^(h x) = 3) ↔ 
  (∃ x, f x = 0 ∧ g x = 0 ∧ h x = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_iff_common_root_l2570_257028


namespace NUMINAMATH_CALUDE_unique_quadruple_solution_l2570_257074

theorem unique_quadruple_solution :
  ∃! (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
    a^2 + b^2 + c^2 + d^2 = 9 ∧
    (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81 ∧
    a + b + c + d = 6 ∧
    a = 1.5 ∧ b = 1.5 ∧ c = 1.5 ∧ d = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadruple_solution_l2570_257074


namespace NUMINAMATH_CALUDE_five_to_five_sum_equals_five_to_six_l2570_257015

theorem five_to_five_sum_equals_five_to_six : 
  5^5 + 5^5 + 5^5 + 5^5 + 5^5 = 5^6 := by
  sorry

end NUMINAMATH_CALUDE_five_to_five_sum_equals_five_to_six_l2570_257015


namespace NUMINAMATH_CALUDE_bruce_age_multiple_l2570_257056

/-- The number of years it takes for a person to become a multiple of another person's age -/
def years_to_multiple (initial_age_older : ℕ) (initial_age_younger : ℕ) (multiple : ℕ) : ℕ :=
  let x := (multiple * initial_age_younger - initial_age_older) / (multiple - 1)
  x

/-- Theorem stating that it takes 6 years for a 36-year-old to become 3 times as old as an 8-year-old -/
theorem bruce_age_multiple : years_to_multiple 36 8 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bruce_age_multiple_l2570_257056


namespace NUMINAMATH_CALUDE_triangle_similarity_l2570_257067

-- Define the points in the plane
variable (A B C A' B' C' S M N : ℝ × ℝ)

-- Define the properties of the triangles and points
def is_equilateral (X Y Z : ℝ × ℝ) : Prop := sorry
def is_center (S X Y Z : ℝ × ℝ) : Prop := sorry
def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry
def are_similar (T1 T2 T3 U1 U2 U3 : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem triangle_similarity 
  (h1 : is_equilateral A B C)
  (h2 : is_equilateral A' B' C')
  (h3 : is_center S A B C)
  (h4 : A' ≠ S)
  (h5 : B' ≠ S)
  (h6 : is_midpoint M A' B)
  (h7 : is_midpoint N A B') :
  are_similar S B' M S A' N := by sorry

end NUMINAMATH_CALUDE_triangle_similarity_l2570_257067


namespace NUMINAMATH_CALUDE_log_equation_implies_sum_l2570_257099

theorem log_equation_implies_sum (u v : ℝ) 
  (hu : u > 1) (hv : v > 1)
  (h : (Real.log u / Real.log 3)^3 + (Real.log v / Real.log 5)^3 + 6 = 
       6 * (Real.log u / Real.log 3) * (Real.log v / Real.log 5)) :
  u^Real.sqrt 3 + v^Real.sqrt 3 = 152 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_implies_sum_l2570_257099


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2570_257093

open Real

noncomputable def series_sum (n : ℕ) : ℝ := 3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

theorem infinite_series_sum :
  (∑' n, series_sum n) = 1/4 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2570_257093


namespace NUMINAMATH_CALUDE_problem_solution_l2570_257075

theorem problem_solution (a b : ℝ) : 
  a = 105 ∧ a^3 = 21 * 49 * 45 * b → b = 12.5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2570_257075


namespace NUMINAMATH_CALUDE_invalid_transformation_l2570_257058

theorem invalid_transformation (x y m : ℝ) : 
  ¬(∀ (x y m : ℝ), x = y → x / m = y / m) :=
sorry

end NUMINAMATH_CALUDE_invalid_transformation_l2570_257058


namespace NUMINAMATH_CALUDE_divisibility_condition_l2570_257019

theorem divisibility_condition (n : ℤ) : (n + 1) ∣ (n^2 + 1) ↔ n = -3 ∨ n = -2 ∨ n = 0 ∨ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2570_257019


namespace NUMINAMATH_CALUDE_prime_representation_mod_24_l2570_257082

theorem prime_representation_mod_24 (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  (∃ x y : ℤ, (p : ℤ) = 2 * x^2 + 3 * y^2) ↔ (p % 24 = 5 ∨ p % 24 = 11) :=
by sorry

end NUMINAMATH_CALUDE_prime_representation_mod_24_l2570_257082


namespace NUMINAMATH_CALUDE_least_with_twelve_factors_l2570_257040

/-- A function that counts the number of positive factors of a natural number -/
def count_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 12 positive factors -/
def has_twelve_factors (n : ℕ) : Prop := count_factors n = 12

/-- Theorem stating that 108 is the least positive integer with exactly 12 positive factors -/
theorem least_with_twelve_factors :
  (∀ m : ℕ, m > 0 → m < 108 → ¬(has_twelve_factors m)) ∧ has_twelve_factors 108 := by
  sorry

end NUMINAMATH_CALUDE_least_with_twelve_factors_l2570_257040


namespace NUMINAMATH_CALUDE_distance_to_point_l2570_257038

/-- The distance from the origin to the point (8, -3, 6) in 3D space is √109. -/
theorem distance_to_point : Real.sqrt 109 = Real.sqrt (8^2 + (-3)^2 + 6^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_l2570_257038


namespace NUMINAMATH_CALUDE_number_of_children_l2570_257049

theorem number_of_children (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 4) (h2 : total_pencils = 32) :
  total_pencils / pencils_per_child = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l2570_257049


namespace NUMINAMATH_CALUDE_odd_prob_greater_than_even_prob_l2570_257084

/-- Represents the number of beads in the bottle -/
def num_beads : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the total number of possible outcomes when pouring beads -/
def total_outcomes : ℕ := choose num_beads 1 + choose num_beads 2 + choose num_beads 3 + choose num_beads 4

/-- Calculates the number of outcomes resulting in an odd number of beads -/
def odd_outcomes : ℕ := choose num_beads 1 + choose num_beads 3

/-- Calculates the number of outcomes resulting in an even number of beads -/
def even_outcomes : ℕ := choose num_beads 2 + choose num_beads 4

/-- Theorem stating that the probability of pouring out an odd number of beads
    is greater than the probability of pouring out an even number of beads -/
theorem odd_prob_greater_than_even_prob :
  (odd_outcomes : ℚ) / total_outcomes > (even_outcomes : ℚ) / total_outcomes :=
sorry

end NUMINAMATH_CALUDE_odd_prob_greater_than_even_prob_l2570_257084


namespace NUMINAMATH_CALUDE_min_quadratic_function_l2570_257041

theorem min_quadratic_function :
  (∀ x : ℝ, x^2 - 2*x ≥ -1) ∧ (∃ x : ℝ, x^2 - 2*x = -1) := by
  sorry

end NUMINAMATH_CALUDE_min_quadratic_function_l2570_257041


namespace NUMINAMATH_CALUDE_remainder_sum_l2570_257029

theorem remainder_sum (n : ℤ) (h : n % 24 = 10) : (n % 4 + n % 6 = 6) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2570_257029


namespace NUMINAMATH_CALUDE_fraction_problem_l2570_257083

theorem fraction_problem (n : ℕ) : 
  (n : ℚ) / (3 * n - 7 : ℚ) = 2 / 5 → n = 14 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l2570_257083


namespace NUMINAMATH_CALUDE_hose_flow_rate_l2570_257045

/-- Given a pool that takes 50 hours to fill, water costs 1 cent for 10 gallons,
    and it costs 5 dollars to fill the pool, the hose runs at a rate of 100 gallons per hour. -/
theorem hose_flow_rate (fill_time : ℕ) (water_cost : ℚ) (fill_cost : ℕ) :
  fill_time = 50 →
  water_cost = 1 / 10 →
  fill_cost = 5 →
  (fill_cost * 100 : ℚ) / (water_cost * fill_time) = 100 := by
  sorry

end NUMINAMATH_CALUDE_hose_flow_rate_l2570_257045


namespace NUMINAMATH_CALUDE_solution_set_f_geq_2_range_of_a_for_full_solution_set_l2570_257053

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem for part I
theorem solution_set_f_geq_2 :
  {x : ℝ | f x ≥ 2} = {x : ℝ | x ≥ 3/2} :=
sorry

-- Theorem for part II
theorem range_of_a_for_full_solution_set :
  {a : ℝ | ∀ x, f x ≤ |a - 2|} = {a : ℝ | a ≤ -1 ∨ a ≥ 5} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_2_range_of_a_for_full_solution_set_l2570_257053


namespace NUMINAMATH_CALUDE_multiplication_closed_in_P_l2570_257061

def P : Set ℕ := {n : ℕ | ∃ m : ℕ, m > 0 ∧ n = m^2}

theorem multiplication_closed_in_P : 
  ∀ a b : ℕ, a ∈ P → b ∈ P → (a * b) ∈ P := by
  sorry

end NUMINAMATH_CALUDE_multiplication_closed_in_P_l2570_257061


namespace NUMINAMATH_CALUDE_rachel_toys_l2570_257008

theorem rachel_toys (jason_toys : ℕ) (john_toys : ℕ) (rachel_toys : ℕ)
  (h1 : jason_toys = 21)
  (h2 : jason_toys = 3 * john_toys)
  (h3 : john_toys = rachel_toys + 6) :
  rachel_toys = 1 := by
  sorry

end NUMINAMATH_CALUDE_rachel_toys_l2570_257008


namespace NUMINAMATH_CALUDE_largest_2010_digit_prime_squared_minus_one_div_24_l2570_257000

/-- The largest prime number with 2010 digits -/
def p : ℕ := sorry

/-- p is prime -/
axiom p_prime : Nat.Prime p

/-- p has 2010 digits -/
axiom p_digits : 10^2009 ≤ p ∧ p < 10^2010

/-- p is the largest prime with 2010 digits -/
axiom p_largest : ∀ q, Nat.Prime q → 10^2009 ≤ q → q < 10^2010 → q ≤ p

theorem largest_2010_digit_prime_squared_minus_one_div_24 : 
  24 ∣ (p^2 - 1) := by sorry

end NUMINAMATH_CALUDE_largest_2010_digit_prime_squared_minus_one_div_24_l2570_257000


namespace NUMINAMATH_CALUDE_remaining_artifacts_correct_l2570_257011

structure MarineArtifacts where
  clam_shells : ℕ
  conch_shells : ℕ
  oyster_shells : ℕ
  coral_pieces : ℕ
  sea_glass_shards : ℕ
  starfish : ℕ

def initial_artifacts : MarineArtifacts :=
  { clam_shells := 325
  , conch_shells := 210
  , oyster_shells := 144
  , coral_pieces := 96
  , sea_glass_shards := 180
  , starfish := 110 }

def given_away (a : MarineArtifacts) : MarineArtifacts :=
  { clam_shells := a.clam_shells / 4
  , conch_shells := 50
  , oyster_shells := a.oyster_shells / 3
  , coral_pieces := a.coral_pieces / 2
  , sea_glass_shards := a.sea_glass_shards / 5
  , starfish := 0 }

def remaining_artifacts (a : MarineArtifacts) : MarineArtifacts :=
  { clam_shells := a.clam_shells - (given_away a).clam_shells
  , conch_shells := a.conch_shells - (given_away a).conch_shells
  , oyster_shells := a.oyster_shells - (given_away a).oyster_shells
  , coral_pieces := a.coral_pieces - (given_away a).coral_pieces
  , sea_glass_shards := a.sea_glass_shards - (given_away a).sea_glass_shards
  , starfish := a.starfish - (given_away a).starfish }

theorem remaining_artifacts_correct :
  (remaining_artifacts initial_artifacts) =
    { clam_shells := 244
    , conch_shells := 160
    , oyster_shells := 96
    , coral_pieces := 48
    , sea_glass_shards := 144
    , starfish := 110 } := by
  sorry

end NUMINAMATH_CALUDE_remaining_artifacts_correct_l2570_257011


namespace NUMINAMATH_CALUDE_inequality_proof_l2570_257066

theorem inequality_proof (a₁ a₂ a₃ : ℝ) 
  (h₁ : 0 ≤ a₁) (h₂ : 0 ≤ a₂) (h₃ : 0 ≤ a₃) 
  (h_sum : a₁ + a₂ + a₃ = 1) : 
  a₁ * Real.sqrt a₂ + a₂ * Real.sqrt a₃ + a₃ * Real.sqrt a₁ ≤ 1 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2570_257066


namespace NUMINAMATH_CALUDE_max_value_theorem_l2570_257051

def feasible_region (x y : ℝ) : Prop :=
  2 * x + y ≤ 4 ∧ x ≤ y ∧ x ≥ 1/2

def objective_function (x y : ℝ) : ℝ :=
  2 * x - y

theorem max_value_theorem :
  ∃ (x y : ℝ), feasible_region x y ∧
  ∀ (x' y' : ℝ), feasible_region x' y' →
  objective_function x y ≥ objective_function x' y' ∧
  objective_function x y = 4/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2570_257051


namespace NUMINAMATH_CALUDE_line_segment_ratios_l2570_257048

/-- Given four points X, Y, Z, W on a straight line in that order,
    with XY = 3, YZ = 4, and XW = 20, prove that
    the ratio of XZ to YW is 7/16 and the ratio of YZ to XW is 1/5. -/
theorem line_segment_ratios
  (X Y Z W : ℝ)  -- Points represented as real numbers
  (h_order : X < Y ∧ Y < Z ∧ Z < W)  -- Order of points
  (h_xy : Y - X = 3)  -- XY = 3
  (h_yz : Z - Y = 4)  -- YZ = 4
  (h_xw : W - X = 20)  -- XW = 20
  : (Z - X) / (W - Y) = 7 / 16 ∧ (Z - Y) / (W - X) = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_line_segment_ratios_l2570_257048


namespace NUMINAMATH_CALUDE_profit_on_10th_day_max_profit_day_max_profit_value_k_for_min_profit_l2570_257039

/-- Represents the day number (1 to 50) -/
def Day := Fin 50

/-- The cost price of a lantern in yuan -/
def cost_price : ℝ := 18

/-- The selling price of a lantern on day x -/
def selling_price (x : Day) : ℝ := -0.5 * x.val + 55

/-- The quantity of lanterns sold on day x -/
def quantity_sold (x : Day) : ℝ := 5 * x.val + 50

/-- The daily sales profit on day x -/
def daily_profit (x : Day) : ℝ := (selling_price x - cost_price) * quantity_sold x

/-- Theorem stating the daily sales profit on the 10th day -/
theorem profit_on_10th_day : daily_profit ⟨10, by norm_num⟩ = 3200 := by sorry

/-- Theorem stating the day of maximum profit between 34th and 50th day -/
theorem max_profit_day (x : Day) (h : 34 ≤ x.val ∧ x.val ≤ 50) :
  daily_profit x ≤ daily_profit ⟨34, by norm_num⟩ := by sorry

/-- Theorem stating the maximum profit value between 34th and 50th day -/
theorem max_profit_value : daily_profit ⟨34, by norm_num⟩ = 4400 := by sorry

/-- The modified selling price with increase k -/
def modified_selling_price (x : Day) (k : ℝ) : ℝ := selling_price x + k

/-- The modified daily profit with price increase k -/
def modified_daily_profit (x : Day) (k : ℝ) : ℝ :=
  (modified_selling_price x k - cost_price) * quantity_sold x

/-- Theorem stating the value of k for minimum daily profit of 5460 yuan from 30th to 40th day -/
theorem k_for_min_profit (k : ℝ) (h : 0 < k ∧ k < 8) :
  (∀ x : Day, 30 ≤ x.val ∧ x.val ≤ 40 → modified_daily_profit x k ≥ 5460) ↔ k = 5.3 := by sorry

end NUMINAMATH_CALUDE_profit_on_10th_day_max_profit_day_max_profit_value_k_for_min_profit_l2570_257039


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2570_257032

theorem inequality_system_solution (m : ℝ) : 
  (∃ x : ℤ, (x : ℝ) ≥ 2 ∧ (x - m) / 2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2) ∧ 
   ∀ y : ℤ, y < x → (y : ℝ) - m < 4 ∨ y - 4 > 3 * (y - 2)) →
  -3 < m ∧ m ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2570_257032


namespace NUMINAMATH_CALUDE_ray_walks_11_blocks_home_l2570_257059

/-- Represents Ray's dog walking routine -/
structure DogWalk where
  trips_per_day : ℕ
  total_blocks_per_day : ℕ
  blocks_to_park : ℕ
  blocks_to_school : ℕ

/-- Calculates the number of blocks Ray walks to get back home -/
def blocks_to_home (dw : DogWalk) : ℕ :=
  (dw.total_blocks_per_day / dw.trips_per_day) - (dw.blocks_to_park + dw.blocks_to_school)

/-- Theorem stating that Ray walks 11 blocks to get back home -/
theorem ray_walks_11_blocks_home :
  ∃ (dw : DogWalk),
    dw.trips_per_day = 3 ∧
    dw.total_blocks_per_day = 66 ∧
    dw.blocks_to_park = 4 ∧
    dw.blocks_to_school = 7 ∧
    blocks_to_home dw = 11 := by
  sorry

end NUMINAMATH_CALUDE_ray_walks_11_blocks_home_l2570_257059


namespace NUMINAMATH_CALUDE_m_range_l2570_257054

/-- A function f satisfying the given conditions -/
def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc (-1) 1 → f (-x) = -f x) ∧
  (∀ a b, a ∈ Set.Ioo (-1) 0 → b ∈ Set.Ioo (-1) 0 → a ≠ b → (f a - f b) / (a - b) > 0)

theorem m_range (f : ℝ → ℝ) (m : ℝ) (hf : f_conditions f) (h : f (m + 1) > f (2 * m)) :
  -1/2 ≤ m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2570_257054


namespace NUMINAMATH_CALUDE_product_remainder_zero_l2570_257017

theorem product_remainder_zero : (2005 * 2006 * 2007 * 2008 * 2009) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_zero_l2570_257017


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_plus_one_l2570_257044

theorem smallest_three_digit_multiple_plus_one : ∃ (n : ℕ), 
  (n = 421) ∧ 
  (100 ≤ n) ∧ 
  (n < 1000) ∧ 
  (∃ (k : ℕ), n = k * 3 + 1) ∧
  (∃ (k : ℕ), n = k * 4 + 1) ∧
  (∃ (k : ℕ), n = k * 5 + 1) ∧
  (∃ (k : ℕ), n = k * 6 + 1) ∧
  (∃ (k : ℕ), n = k * 7 + 1) ∧
  (∀ (m : ℕ), 
    (100 ≤ m) ∧ 
    (m < n) → 
    ¬((∃ (k : ℕ), m = k * 3 + 1) ∧
      (∃ (k : ℕ), m = k * 4 + 1) ∧
      (∃ (k : ℕ), m = k * 5 + 1) ∧
      (∃ (k : ℕ), m = k * 6 + 1) ∧
      (∃ (k : ℕ), m = k * 7 + 1))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_plus_one_l2570_257044


namespace NUMINAMATH_CALUDE_light_bulb_configurations_l2570_257060

/-- The number of light bulbs -/
def num_bulbs : ℕ := 5

/-- The number of states each bulb can have (on or off) -/
def states_per_bulb : ℕ := 2

/-- The total number of possible lighting configurations -/
def total_configurations : ℕ := states_per_bulb ^ num_bulbs

theorem light_bulb_configurations :
  total_configurations = 32 :=
by sorry

end NUMINAMATH_CALUDE_light_bulb_configurations_l2570_257060


namespace NUMINAMATH_CALUDE_remaining_cards_l2570_257055

def initial_cards : ℕ := 87
def sam_cards : ℕ := 8
def alex_cards : ℕ := 13

theorem remaining_cards : initial_cards - (sam_cards + alex_cards) = 66 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cards_l2570_257055


namespace NUMINAMATH_CALUDE_ethans_candles_weight_l2570_257068

/-- The combined weight of Ethan's candles -/
def combined_weight (total_candles : ℕ) (beeswax_per_candle : ℕ) (coconut_oil_per_candle : ℕ) : ℕ :=
  total_candles * (beeswax_per_candle + coconut_oil_per_candle)

/-- Theorem: The combined weight of Ethan's candles is 63 ounces -/
theorem ethans_candles_weight :
  combined_weight (10 - 3) 8 1 = 63 := by
  sorry

end NUMINAMATH_CALUDE_ethans_candles_weight_l2570_257068


namespace NUMINAMATH_CALUDE_hundredth_figure_count_l2570_257005

/-- Represents the number of nonoverlapping unit triangles in the nth figure of the pattern. -/
def triangle_count (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The first four terms of the sequence match the given pattern. -/
axiom first_four_correct : 
  triangle_count 0 = 1 ∧ 
  triangle_count 1 = 7 ∧ 
  triangle_count 2 = 19 ∧ 
  triangle_count 3 = 37

/-- The 100th figure contains 30301 nonoverlapping unit triangles. -/
theorem hundredth_figure_count : triangle_count 100 = 30301 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_figure_count_l2570_257005


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2570_257024

-- Problem 1
theorem simplify_expression_1 (x : ℝ) :
  (x + 3) * (3 * x - 2) + x * (1 - 3 * x) = 8 * x - 6 := by sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) (h1 : m ≠ 2) (h2 : m ≠ -2) :
  (1 - m / (m + 2)) / ((m^2 - 4*m + 4) / (m^2 - 4)) = 2 / (m - 2) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2570_257024


namespace NUMINAMATH_CALUDE_largest_root_of_f_cubed_l2570_257069

/-- The function f(x) = x^2 + 12x + 30 -/
def f (x : ℝ) : ℝ := x^2 + 12*x + 30

/-- The composition of f with itself three times -/
def f_cubed (x : ℝ) : ℝ := f (f (f x))

/-- The largest real root of f(f(f(x))) = 0 -/
noncomputable def largest_root : ℝ := -6 + (6 : ℝ)^(1/8)

theorem largest_root_of_f_cubed :
  (f_cubed largest_root = 0) ∧
  (∀ x : ℝ, f_cubed x = 0 → x ≤ largest_root) :=
by sorry

end NUMINAMATH_CALUDE_largest_root_of_f_cubed_l2570_257069


namespace NUMINAMATH_CALUDE_sniper_B_wins_l2570_257012

/-- Represents a sniper with probabilities of scoring 1, 2, and 3 points -/
structure Sniper where
  prob1 : ℝ
  prob2 : ℝ
  prob3 : ℝ

/-- Calculate the expected score for a sniper -/
def expectedScore (s : Sniper) : ℝ := 1 * s.prob1 + 2 * s.prob2 + 3 * s.prob3

/-- Sniper A with given probabilities -/
def sniperA : Sniper := { prob1 := 0.4, prob2 := 0.1, prob3 := 0.5 }

/-- Sniper B with given probabilities -/
def sniperB : Sniper := { prob1 := 0.1, prob2 := 0.6, prob3 := 0.3 }

/-- Theorem stating that Sniper B has a higher expected score than Sniper A -/
theorem sniper_B_wins : expectedScore sniperB > expectedScore sniperA := by
  sorry

end NUMINAMATH_CALUDE_sniper_B_wins_l2570_257012


namespace NUMINAMATH_CALUDE_largest_valid_pair_l2570_257077

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ 100 ∧ is_integer ((a + b) * (a + b + 1) / (a * b : ℚ))

theorem largest_valid_pair :
  ∀ a b : ℕ, valid_pair a b →
    b ≤ 90 ∧
    (b = 90 → a ≤ 35) ∧
    valid_pair 35 90
  := by sorry

end NUMINAMATH_CALUDE_largest_valid_pair_l2570_257077


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l2570_257092

/-- Given two algebraic terms are like terms, prove that the sum of their exponents is 5 -/
theorem like_terms_exponent_sum (a b : ℝ) (m n : ℕ) : 
  (∃ (k : ℝ), k * a^(2*m) * b^3 = 5 * a^6 * b^(n+1)) → m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l2570_257092


namespace NUMINAMATH_CALUDE_option_D_not_suitable_for_comprehensive_survey_l2570_257078

-- Define the type for survey options
inductive SurveyOption
| A -- Security check for passengers before boarding a plane
| B -- School recruiting teachers and conducting interviews for applicants
| C -- Understanding the extracurricular reading time of seventh-grade students in a school
| D -- Understanding the service life of a batch of light bulbs

-- Define a function to check if an option is suitable for a comprehensive survey
def isSuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.A => True
  | SurveyOption.B => True
  | SurveyOption.C => True
  | SurveyOption.D => False

-- Theorem stating that option D is not suitable for a comprehensive survey
theorem option_D_not_suitable_for_comprehensive_survey :
  ¬(isSuitableForComprehensiveSurvey SurveyOption.D) :=
by sorry

end NUMINAMATH_CALUDE_option_D_not_suitable_for_comprehensive_survey_l2570_257078


namespace NUMINAMATH_CALUDE_equilateral_triangles_area_sum_l2570_257018

/-- Given an isosceles right triangle with leg length 36 units, the sum of the areas
    of an infinite series of equilateral triangles drawn on one leg (with their third
    vertices on the hypotenuse) is equal to half the area of the original right triangle. -/
theorem equilateral_triangles_area_sum (leg_length : ℝ) (h : leg_length = 36) :
  let right_triangle_area := (1 / 2) * leg_length * leg_length
  let equilateral_triangles_area_sum := right_triangle_area / 2
  equilateral_triangles_area_sum = 324 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangles_area_sum_l2570_257018


namespace NUMINAMATH_CALUDE_max_vertices_1000_triangles_l2570_257064

/-- The maximum number of distinct points that can be vertices of 1000 triangles in a quadrilateral -/
def max_distinct_vertices (num_triangles : ℕ) (quadrilateral_angle_sum : ℕ) : ℕ :=
  let triangle_angle_sum := 180
  let total_angle_sum := num_triangles * triangle_angle_sum
  let excess_angle_sum := total_angle_sum - quadrilateral_angle_sum
  let side_vertices := excess_angle_sum / triangle_angle_sum
  let original_vertices := 4
  side_vertices + original_vertices

/-- Theorem stating that the maximum number of distinct vertices is 1002 -/
theorem max_vertices_1000_triangles :
  max_distinct_vertices 1000 360 = 1002 := by
  sorry

end NUMINAMATH_CALUDE_max_vertices_1000_triangles_l2570_257064


namespace NUMINAMATH_CALUDE_work_completion_time_l2570_257010

/-- The time taken for all three workers (p, q, and r) to complete the work together -/
theorem work_completion_time 
  (efficiency_p : ℝ) 
  (efficiency_q : ℝ) 
  (efficiency_r : ℝ) 
  (time_p : ℝ) 
  (h1 : efficiency_p = 1.3 * efficiency_q) 
  (h2 : time_p = 23) 
  (h3 : efficiency_r = 1.5 * (efficiency_p + efficiency_q)) : 
  (time_p * efficiency_p) / (efficiency_p + efficiency_q + efficiency_r) = 5.2 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l2570_257010


namespace NUMINAMATH_CALUDE_square_ratio_side_lengths_l2570_257062

theorem square_ratio_side_lengths : 
  ∃ (a b c : ℕ), 
    (a * a * b : ℚ) / (c * c) = 50 / 98 ∧ 
    a = 5 ∧ 
    b = 1 ∧ 
    c = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_lengths_l2570_257062


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l2570_257046

-- Define the original dimensions
def original_length : ℝ := 140
def original_width : ℝ := 40

-- Define the width decrease percentage
def width_decrease_percent : ℝ := 17.692307692307693

-- Define the expected length increase percentage
def expected_length_increase_percent : ℝ := 21.428571428571427

-- Theorem statement
theorem rectangle_dimension_change :
  let new_width : ℝ := original_width * (1 - width_decrease_percent / 100)
  let new_length : ℝ := (original_length * original_width) / new_width
  let actual_length_increase_percent : ℝ := (new_length - original_length) / original_length * 100
  actual_length_increase_percent = expected_length_increase_percent := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l2570_257046
