import Mathlib

namespace NUMINAMATH_CALUDE_smallest_sum_on_square_corners_l1400_140087

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_not_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b > 1

theorem smallest_sum_on_square_corners (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 → D > 0 →
  is_relatively_prime A C →
  is_relatively_prime B D →
  is_not_relatively_prime A B →
  is_not_relatively_prime B C →
  is_not_relatively_prime C D →
  is_not_relatively_prime D A →
  A + B + C + D ≥ 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_on_square_corners_l1400_140087


namespace NUMINAMATH_CALUDE_classroom_difference_l1400_140066

/-- Proves that the difference between the total number of students and books in 6 classrooms is 90 -/
theorem classroom_difference : 
  let students_per_classroom : ℕ := 20
  let books_per_classroom : ℕ := 5
  let num_classrooms : ℕ := 6
  let total_students : ℕ := students_per_classroom * num_classrooms
  let total_books : ℕ := books_per_classroom * num_classrooms
  total_students - total_books = 90 := by
  sorry


end NUMINAMATH_CALUDE_classroom_difference_l1400_140066


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_numbers_l1400_140018

theorem largest_common_divisor_of_consecutive_odd_numbers (n : ℕ) :
  (n % 2 = 0 ∧ n > 0) →
  ∃ (k : ℕ), k = 45 ∧ 
    (∀ (m : ℕ), m ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) → m ≤ k) ∧
    k ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
by sorry


end NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_numbers_l1400_140018


namespace NUMINAMATH_CALUDE_initial_salt_percentage_l1400_140031

/-- Proves that given a 100 kg solution of salt and water, if adding 20 kg of pure salt
    results in a 25% salt concentration, then the initial salt percentage was 10%. -/
theorem initial_salt_percentage
  (initial_mass : ℝ)
  (added_salt : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_mass = 100)
  (h2 : added_salt = 20)
  (h3 : final_concentration = 0.25)
  (h4 : (initial_mass * x + added_salt) / (initial_mass + added_salt) = final_concentration) :
  x = 0.1 := by
  sorry

#check initial_salt_percentage

end NUMINAMATH_CALUDE_initial_salt_percentage_l1400_140031


namespace NUMINAMATH_CALUDE_mode_better_representation_l1400_140057

/-- Represents the salary distribution of employees in a company -/
structure SalaryDistribution where
  manager_salary : ℕ
  deputy_manager_salary : ℕ
  employee_salary : ℕ
  manager_count : ℕ
  deputy_manager_count : ℕ
  employee_count : ℕ

/-- Calculates the mean salary -/
def mean_salary (sd : SalaryDistribution) : ℚ :=
  (sd.manager_salary * sd.manager_count +
   sd.deputy_manager_salary * sd.deputy_manager_count +
   sd.employee_salary * sd.employee_count) /
  (sd.manager_count + sd.deputy_manager_count + sd.employee_count)

/-- Finds the mode salary -/
def mode_salary (sd : SalaryDistribution) : ℕ :=
  if sd.employee_count > sd.manager_count ∧ sd.employee_count > sd.deputy_manager_count then
    sd.employee_salary
  else if sd.deputy_manager_count > sd.manager_count then
    sd.deputy_manager_salary
  else
    sd.manager_salary

/-- Represents how well a measure describes the concentration trend -/
def concentration_measure (salary : ℕ) (sd : SalaryDistribution) : ℚ :=
  (sd.manager_count * (if salary = sd.manager_salary then 1 else 0) +
   sd.deputy_manager_count * (if salary = sd.deputy_manager_salary then 1 else 0) +
   sd.employee_count * (if salary = sd.employee_salary then 1 else 0)) /
  (sd.manager_count + sd.deputy_manager_count + sd.employee_count)

/-- Theorem stating that the mode better represents the concentration trend than the mean -/
theorem mode_better_representation (sd : SalaryDistribution)
  (h1 : sd.manager_salary = 12000)
  (h2 : sd.deputy_manager_salary = 8000)
  (h3 : sd.employee_salary = 3000)
  (h4 : sd.manager_count = 1)
  (h5 : sd.deputy_manager_count = 1)
  (h6 : sd.employee_count = 8) :
  concentration_measure (mode_salary sd) sd > concentration_measure (Nat.floor (mean_salary sd)) sd :=
  sorry

end NUMINAMATH_CALUDE_mode_better_representation_l1400_140057


namespace NUMINAMATH_CALUDE_special_collection_returned_percentage_l1400_140036

/-- Calculates the percentage of returned books given initial count, final count, and loaned count. -/
def percentage_returned (initial : ℕ) (final : ℕ) (loaned : ℕ) : ℚ :=
  (1 - (initial - final : ℚ) / (loaned : ℚ)) * 100

/-- Theorem stating that the percentage of returned books is 65% given the problem conditions. -/
theorem special_collection_returned_percentage :
  percentage_returned 75 61 40 = 65 := by
  sorry

end NUMINAMATH_CALUDE_special_collection_returned_percentage_l1400_140036


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1400_140093

/-- Given two adjacent points (1,2) and (2,5) on a square in a Cartesian coordinate plane,
    the area of the square is 10. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (2, 5)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 10 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l1400_140093


namespace NUMINAMATH_CALUDE_range_of_a_l1400_140026

-- Define the sets A, B, and C
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 3) - 1 / Real.sqrt (7 - x)}
def B : Set ℝ := {y | ∃ x, y = -x^2 + 2*x + 8}
def C (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- Define the theorem
theorem range_of_a (a : ℝ) : A ∪ C a = C a → a ≥ 7 ∨ a + 1 < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1400_140026


namespace NUMINAMATH_CALUDE_measure_45_seconds_l1400_140013

/-- Represents a fuse that can be lit from either end -/
structure Fuse :=
  (burn_time : ℝ)
  (is_uniform : Bool)

/-- Represents the state of burning a fuse -/
inductive BurnState
  | Unlit
  | LitOneEnd
  | LitBothEnds

/-- Represents the result of burning fuses -/
structure BurnResult :=
  (time : ℝ)
  (fuse1 : BurnState)
  (fuse2 : BurnState)

/-- Function to simulate burning fuses -/
def burn_fuses (f1 f2 : Fuse) : BurnResult :=
  sorry

theorem measure_45_seconds (f1 f2 : Fuse) 
  (h1 : f1.burn_time = 60)
  (h2 : f2.burn_time = 60) :
  ∃ (result : BurnResult), result.time = 45 :=
sorry

end NUMINAMATH_CALUDE_measure_45_seconds_l1400_140013


namespace NUMINAMATH_CALUDE_initial_money_correct_l1400_140034

/-- The amount of money Little John had initially -/
def initial_money : ℚ := 10.50

/-- The amount Little John spent on sweets -/
def sweets_cost : ℚ := 2.25

/-- The amount Little John gave to each friend -/
def money_per_friend : ℚ := 2.20

/-- The number of friends Little John gave money to -/
def number_of_friends : ℕ := 2

/-- The amount of money Little John had left -/
def money_left : ℚ := 3.85

/-- Theorem stating that the initial amount of money is correct given the conditions -/
theorem initial_money_correct : 
  initial_money = sweets_cost + (money_per_friend * number_of_friends) + money_left :=
by sorry

end NUMINAMATH_CALUDE_initial_money_correct_l1400_140034


namespace NUMINAMATH_CALUDE_sum_of_first_three_coefficients_l1400_140079

theorem sum_of_first_three_coefficients (b : ℝ) : 
  let expansion := (1 + 2/b)^7
  let first_term_coeff := 1
  let second_term_coeff := 7 * 2 / b
  let third_term_coeff := (7 * 6 / 2) * (2 / b)^2
  first_term_coeff + second_term_coeff + third_term_coeff = 211 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_three_coefficients_l1400_140079


namespace NUMINAMATH_CALUDE_difference_of_squares_l1400_140010

theorem difference_of_squares (m n : ℝ) : m^2 - n^2 = (m + n) * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1400_140010


namespace NUMINAMATH_CALUDE_heart_ratio_l1400_140099

-- Define the ♡ operation
def heart (n m : ℕ) : ℚ := 3 * (n^3 : ℚ) * (m^2 : ℚ)

-- State the theorem
theorem heart_ratio : (heart 3 5) / (heart 5 3) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_l1400_140099


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l1400_140085

theorem unique_root_quadratic (k : ℝ) :
  (∃! x : ℝ, k * x^2 - 3 * x + 2 = 0) → k = 0 ∨ k = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l1400_140085


namespace NUMINAMATH_CALUDE_inequality_proof_l1400_140071

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1400_140071


namespace NUMINAMATH_CALUDE_total_spent_on_fruits_l1400_140080

def total_fruits : ℕ := 32
def plum_cost : ℕ := 2
def peach_cost : ℕ := 1
def plums_bought : ℕ := 20

theorem total_spent_on_fruits : 
  plums_bought * plum_cost + (total_fruits - plums_bought) * peach_cost = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_fruits_l1400_140080


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l1400_140005

theorem divisibility_by_eleven (B : Nat) : 
  (B = 5 → 11 ∣ 15675) → 
  (∀ n : Nat, n < 10 → (11 ∣ (15670 + n) ↔ n = B)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l1400_140005


namespace NUMINAMATH_CALUDE_road_width_calculation_l1400_140082

/-- Given a rectangular lawn with two roads running through the middle,
    calculate the width of each road based on the cost of traveling. -/
theorem road_width_calculation (lawn_length lawn_width total_cost cost_per_sqm : ℝ)
    (h1 : lawn_length = 80)
    (h2 : lawn_width = 60)
    (h3 : total_cost = 5625)
    (h4 : cost_per_sqm = 3)
    (h5 : total_cost = (lawn_length + lawn_width) * road_width * cost_per_sqm) :
    road_width = total_cost / (cost_per_sqm * (lawn_length + lawn_width)) :=
by sorry

end NUMINAMATH_CALUDE_road_width_calculation_l1400_140082


namespace NUMINAMATH_CALUDE_polynomial_value_l1400_140008

theorem polynomial_value (a : ℝ) (h : a = Real.sqrt 17 - 1) : 
  a^5 + 2*a^4 - 17*a^3 - a^2 + 18*a - 17 = -1 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_l1400_140008


namespace NUMINAMATH_CALUDE_modular_inverse_89_mod_90_l1400_140095

theorem modular_inverse_89_mod_90 : ∃ x : ℕ, 0 ≤ x ∧ x < 90 ∧ (89 * x) % 90 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_89_mod_90_l1400_140095


namespace NUMINAMATH_CALUDE_coconut_grove_yield_l1400_140067

theorem coconut_grove_yield (yield_group1 yield_group2 yield_group3 : ℕ) : 
  yield_group1 = 60 →
  yield_group2 = 120 →
  (3 * yield_group1 + 2 * yield_group2 + yield_group3) / 6 = 100 →
  yield_group3 = 180 := by
sorry

end NUMINAMATH_CALUDE_coconut_grove_yield_l1400_140067


namespace NUMINAMATH_CALUDE_f_inequality_l1400_140014

noncomputable def f (x : ℝ) : ℝ := Real.log (|x| + 1) / Real.log (1/2) + 1 / (x^2 + 1)

theorem f_inequality (x : ℝ) : f x > f (2*x - 1) ↔ x > 1 ∨ x < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1400_140014


namespace NUMINAMATH_CALUDE_quadratic_properties_l1400_140022

/-- Quadratic function f(x) = x^2 - 4x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem quadratic_properties :
  (∃ (a b : ℝ), ∀ x, f x = (x - a)^2 + b ∧ a = 2 ∧ b = -1) ∧
  (f 1 = 0 ∧ f 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1400_140022


namespace NUMINAMATH_CALUDE_prob_live_to_25_given_20_l1400_140059

/-- The probability of an animal living to 25 years given it has lived to 20 years -/
theorem prob_live_to_25_given_20 (p_20 p_25 : ℝ) 
  (h1 : p_20 = 0.8) 
  (h2 : p_25 = 0.4) 
  (h3 : 0 ≤ p_20 ∧ p_20 ≤ 1) 
  (h4 : 0 ≤ p_25 ∧ p_25 ≤ 1) 
  (h5 : p_25 ≤ p_20) : 
  p_25 / p_20 = 0.5 := by sorry

end NUMINAMATH_CALUDE_prob_live_to_25_given_20_l1400_140059


namespace NUMINAMATH_CALUDE_min_distance_sum_l1400_140030

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -4*x

-- Define the focus F (we don't know its exact coordinates, but we know it exists)
axiom F : ℝ × ℝ

-- Define point A
def A : ℝ × ℝ := (-2, 1)

-- Define a point P on the parabola
structure PointOnParabola where
  P : ℝ × ℝ
  on_parabola : parabola P.1 P.2

-- State the theorem
theorem min_distance_sum (p : PointOnParabola) :
  ∃ (min : ℝ), min = 3 ∧ ∀ (q : PointOnParabola), 
    Real.sqrt ((q.P.1 - F.1)^2 + (q.P.2 - F.2)^2) +
    Real.sqrt ((q.P.1 - A.1)^2 + (q.P.2 - A.2)^2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1400_140030


namespace NUMINAMATH_CALUDE_complex_power_eq_l1400_140047

theorem complex_power_eq (z : ℂ) : 
  (2 * Complex.cos (20 * π / 180) + 2 * Complex.I * Complex.sin (20 * π / 180)) ^ 6 = 
  -32 + 32 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_eq_l1400_140047


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l1400_140028

/-- Given an infinite geometric series with first term a and common ratio r,
    if the sum of the series is 30 and the sum of the squares of its terms is 120,
    then the first term a is equal to 120/17. -/
theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120) :
  a = 120 / 17 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l1400_140028


namespace NUMINAMATH_CALUDE_a_minus_c_equals_three_l1400_140076

theorem a_minus_c_equals_three (a b c d : ℝ) 
  (h1 : a - b = c + d + 9)
  (h2 : a + b = c - d - 3) : 
  a - c = 3 := by
sorry

end NUMINAMATH_CALUDE_a_minus_c_equals_three_l1400_140076


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1400_140086

/-- A positive geometric sequence -/
def PositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∃ r > 0, ∀ n, a (n + 1) = r * a n)

/-- The fourth term of a positive geometric sequence is 2 if the product of the third and fifth terms is 4 -/
theorem fourth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : PositiveGeometricSequence a)
  (h_prod : a 3 * a 5 = 4) :
  a 4 = 2 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1400_140086


namespace NUMINAMATH_CALUDE_inequality_range_l1400_140062

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1400_140062


namespace NUMINAMATH_CALUDE_cosine_equality_l1400_140032

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (812 * π / 180) → n = 88 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l1400_140032


namespace NUMINAMATH_CALUDE_min_distance_parabola_point_l1400_140007

/-- The minimum value of |y| + |PQ| for a point P(x, y) on the parabola x² = -4y and Q(-2√2, 0) -/
theorem min_distance_parabola_point : 
  let Q : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
  ∃ (min : ℝ), min = 2 ∧ 
    ∀ (P : ℝ × ℝ), (P.1 ^ 2 = -4 * P.2) → 
      abs P.2 + Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parabola_point_l1400_140007


namespace NUMINAMATH_CALUDE_problem_solution_l1400_140065

theorem problem_solution (x y : ℝ) : 
  (2*x - 3*y + 5)^2 + |x + y - 2| = 0 → 3*x - 2*y = -3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1400_140065


namespace NUMINAMATH_CALUDE_triangle_problem_l1400_140023

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The cosine of an angle in a triangle -/
def cosine (t : Triangle) (angle : ℕ) : ℝ :=
  sorry

/-- The sine of an angle in a triangle -/
def sine (t : Triangle) (angle : ℕ) : ℝ :=
  sorry

/-- The area of a triangle -/
def area (t : Triangle) : ℝ :=
  sorry

/-- Main theorem -/
theorem triangle_problem (t : Triangle) 
  (h1 : (t.a - t.c)^2 = t.b^2 - (3/4) * t.a * t.c)
  (h2 : t.b = Real.sqrt 13)
  (h3 : ∃ (k : ℝ), sine t 1 = k - (sine t 2) ∧ sine t 3 = k + (sine t 2)) :
  cosine t 2 = 5/8 ∧ area t = (3 * Real.sqrt 39) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1400_140023


namespace NUMINAMATH_CALUDE_calculate_wins_l1400_140020

/-- Given a team's home game statistics, calculate the number of wins -/
theorem calculate_wins (total_games losses : ℕ) (h1 : total_games = 56) (h2 : losses = 12) : 
  total_games - losses - (losses / 2) = 38 := by
  sorry

#check calculate_wins

end NUMINAMATH_CALUDE_calculate_wins_l1400_140020


namespace NUMINAMATH_CALUDE_solve_for_y_l1400_140048

theorem solve_for_y : ∃ y : ℝ, (3 * y) / 4 = 15 ∧ y = 20 := by sorry

end NUMINAMATH_CALUDE_solve_for_y_l1400_140048


namespace NUMINAMATH_CALUDE_parabola_through_point_2_4_l1400_140042

-- Define a parabola type
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define a function to check if a point is on the parabola
def on_parabola (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

-- Theorem statement
theorem parabola_through_point_2_4 :
  ∃ (p : Parabola), 
    (on_parabola p 2 4) ∧ 
    ((∀ x y : ℝ, p.equation x y ↔ y^2 = 8*x) ∨ 
     (∀ x y : ℝ, p.equation x y ↔ x^2 = y)) :=
sorry

end NUMINAMATH_CALUDE_parabola_through_point_2_4_l1400_140042


namespace NUMINAMATH_CALUDE_robie_chocolate_bags_l1400_140054

/-- Calculates the number of chocolate bags left after a series of transactions -/
def chocolateBagsLeft (initialPurchase givingAway additionalPurchase : ℕ) : ℕ :=
  initialPurchase - givingAway + additionalPurchase

/-- Theorem: Given the specific transactions, prove that 4 bags of chocolates are left -/
theorem robie_chocolate_bags :
  chocolateBagsLeft 3 2 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_robie_chocolate_bags_l1400_140054


namespace NUMINAMATH_CALUDE_f_simplification_f_specific_value_l1400_140069

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : ℝ) : f α = -Real.cos α := by
  sorry

theorem f_specific_value : f (-31 * Real.pi / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_simplification_f_specific_value_l1400_140069


namespace NUMINAMATH_CALUDE_horner_v2_value_l1400_140043

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def v0 : ℝ := 3
def v1 (x : ℝ) : ℝ := v0 * x + 5
def v2 (x : ℝ) : ℝ := v1 x * x + 6

theorem horner_v2_value :
  v2 (-4) = 34 := by sorry

end NUMINAMATH_CALUDE_horner_v2_value_l1400_140043


namespace NUMINAMATH_CALUDE_plan_d_cheaper_at_291_l1400_140077

def plan_c_cost (minutes : ℕ) : ℚ := 15 * minutes

def plan_d_cost (minutes : ℕ) : ℚ :=
  if minutes ≤ 100 then
    2500 + 4 * minutes
  else
    2900 + 5 * (minutes - 100)

theorem plan_d_cheaper_at_291 :
  ∀ m : ℕ, m < 291 → plan_c_cost m ≤ plan_d_cost m ∧
  plan_c_cost 291 > plan_d_cost 291 :=
by sorry

end NUMINAMATH_CALUDE_plan_d_cheaper_at_291_l1400_140077


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_l1400_140074

theorem fourth_term_coefficient (a : ℝ) : 
  (Nat.choose 6 3) * a^3 * (-1)^3 = 160 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_l1400_140074


namespace NUMINAMATH_CALUDE_max_value_of_5x_minus_25x_l1400_140025

theorem max_value_of_5x_minus_25x : 
  ∃ (max : ℝ), max = 1/4 ∧ ∀ x : ℝ, 5^x - 25^x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_5x_minus_25x_l1400_140025


namespace NUMINAMATH_CALUDE_min_discount_factor_proof_l1400_140017

/-- Proves the minimum discount factor for a product with given cost and marked prices, ensuring a minimum profit margin. -/
theorem min_discount_factor_proof (cost_price marked_price : ℝ) (min_profit_margin : ℝ) 
  (h_cost : cost_price = 800)
  (h_marked : marked_price = 1200)
  (h_margin : min_profit_margin = 0.2) :
  ∃ x : ℝ, x = 0.8 ∧ 
    ∀ y : ℝ, (marked_price * y - cost_price ≥ cost_price * min_profit_margin) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_min_discount_factor_proof_l1400_140017


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1400_140041

theorem inscribed_square_area (x y : ℝ) :
  (∃ s : ℝ, s > 0 ∧ x^2 / 4 + y^2 / 8 = 1 ∧ x = s ∧ y = s) →
  (4 * s^2 = 32 / 3) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1400_140041


namespace NUMINAMATH_CALUDE_balloon_count_l1400_140056

def total_balloons (joan_initial : ℕ) (popped : ℕ) (jessica : ℕ) : ℕ :=
  (joan_initial - popped) + jessica

theorem balloon_count : total_balloons 9 5 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l1400_140056


namespace NUMINAMATH_CALUDE_common_chord_circles_l1400_140024

/-- Given two circles with equations x^2 + (y - 3/2)^2 = 25/4 and x^2 + y^2 = m,
    if they have a common chord passing through the point (0, 3/2), then m = 17/2. -/
theorem common_chord_circles (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + (y - 3/2)^2 = 25/4 ∧ x^2 + y^2 = m) ∧ 
  (∃ (x : ℝ), x^2 + (3/2 - 3/2)^2 = 25/4 ∧ x^2 + (3/2)^2 = m) →
  m = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_circles_l1400_140024


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1994_l1400_140090

theorem rightmost_three_digits_of_7_to_1994 : 7^1994 % 1000 = 49 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1994_l1400_140090


namespace NUMINAMATH_CALUDE_triangle_trig_identities_l1400_140035

/-- Given an acute triangle ABC with area 3√3, side lengths AB = 3 and AC = 4, 
    prove the following trigonometric identities involving its angles. -/
theorem triangle_trig_identities 
  (A B C : Real) 
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (h_area : (1/2) * 3 * 4 * Real.sin A = 3 * Real.sqrt 3)
  (h_AB : 3 = 3)
  (h_AC : 4 = 4) :
  Real.sin (π/2 + A) = 1/2 ∧ 
  Real.cos (A - B) = (7 * Real.sqrt 13) / 26 := by
sorry

end NUMINAMATH_CALUDE_triangle_trig_identities_l1400_140035


namespace NUMINAMATH_CALUDE_icosahedron_painting_ways_l1400_140088

/-- Represents a regular icosahedron -/
structure Icosahedron where
  faces : Nat
  rotationalSymmetries : Nat

/-- Represents the number of ways to paint an icosahedron -/
def paintingWays (i : Icosahedron) (colors : Nat) : Nat :=
  Nat.factorial (colors - 1) / i.rotationalSymmetries

/-- Theorem stating the number of distinguishable ways to paint an icosahedron -/
theorem icosahedron_painting_ways (i : Icosahedron) (h1 : i.faces = 20) (h2 : i.rotationalSymmetries = 60) :
  paintingWays i 20 = Nat.factorial 19 / 60 := by
  sorry

#check icosahedron_painting_ways

end NUMINAMATH_CALUDE_icosahedron_painting_ways_l1400_140088


namespace NUMINAMATH_CALUDE_mean_of_four_numbers_with_sum_half_l1400_140015

theorem mean_of_four_numbers_with_sum_half (a b c d : ℚ) 
  (sum_condition : a + b + c + d = 1/2) : 
  (a + b + c + d) / 4 = 1/8 := by
sorry

end NUMINAMATH_CALUDE_mean_of_four_numbers_with_sum_half_l1400_140015


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1400_140044

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def B (a : ℝ) : Set ℝ := {x | x ≥ a}

theorem intersection_implies_a_value (a : ℝ) :
  A ∩ B a = {3} → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1400_140044


namespace NUMINAMATH_CALUDE_cube_order_preserving_l1400_140051

theorem cube_order_preserving (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a^3 < b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_order_preserving_l1400_140051


namespace NUMINAMATH_CALUDE_taxi_trip_length_l1400_140033

/-- Calculates the trip length in miles given the taxi fare parameters and total charge -/
def trip_length (initial_fee : ℚ) (charge_per_segment : ℚ) (segment_length : ℚ) (total_charge : ℚ) : ℚ :=
  let segments := (total_charge - initial_fee) / charge_per_segment
  segments * segment_length

theorem taxi_trip_length :
  let initial_fee : ℚ := 225/100
  let charge_per_segment : ℚ := 35/100
  let segment_length : ℚ := 2/5
  let total_charge : ℚ := 54/10
  trip_length initial_fee charge_per_segment segment_length total_charge = 36/10 := by
  sorry

end NUMINAMATH_CALUDE_taxi_trip_length_l1400_140033


namespace NUMINAMATH_CALUDE_modified_bowling_tournament_distributions_l1400_140072

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 5

/-- Represents the number of possible outcomes for each match -/
def outcomes_per_match : ℕ := 2

/-- Represents the number of matches in the tournament -/
def num_matches : ℕ := 5

/-- Theorem: The number of different prize distributions in the modified bowling tournament -/
theorem modified_bowling_tournament_distributions :
  (outcomes_per_match ^ num_matches : ℕ) = 32 :=
sorry

end NUMINAMATH_CALUDE_modified_bowling_tournament_distributions_l1400_140072


namespace NUMINAMATH_CALUDE_volunteer_distribution_theorem_l1400_140073

/-- The number of ways to distribute volunteers among exits -/
def distribute_volunteers (num_volunteers : ℕ) (num_exits : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements -/
theorem volunteer_distribution_theorem :
  distribute_volunteers 5 4 = 240 :=
sorry

end NUMINAMATH_CALUDE_volunteer_distribution_theorem_l1400_140073


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l1400_140046

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) :
  (2 * r) ^ 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l1400_140046


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l1400_140058

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 120 → 
  a * b + b * c + c * d ≤ 3600 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l1400_140058


namespace NUMINAMATH_CALUDE_sum_equals_200_l1400_140091

theorem sum_equals_200 : 139 + 27 + 23 + 11 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_200_l1400_140091


namespace NUMINAMATH_CALUDE_square_difference_l1400_140045

theorem square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1400_140045


namespace NUMINAMATH_CALUDE_shelter_cat_count_l1400_140096

/-- Calculates the total number of cats and kittens in an animal shelter --/
theorem shelter_cat_count (total_adults : ℕ) (female_ratio : ℚ) (litter_ratio : ℚ) (avg_kittens : ℕ) : 
  total_adults = 100 →
  female_ratio = 1/2 →
  litter_ratio = 1/2 →
  avg_kittens = 4 →
  total_adults + (total_adults * female_ratio * litter_ratio * avg_kittens) = 200 := by
sorry

end NUMINAMATH_CALUDE_shelter_cat_count_l1400_140096


namespace NUMINAMATH_CALUDE_lincoln_high_school_groups_l1400_140016

/-- Represents the number of students in various groups at Lincoln High School -/
structure SchoolGroups where
  total : ℕ
  band : ℕ
  chorus : ℕ
  drama : ℕ
  band_chorus_drama : ℕ

/-- Calculates the number of students in both band and chorus but not in drama -/
def students_in_band_and_chorus_not_drama (g : SchoolGroups) : ℕ :=
  g.band + g.chorus - (g.band_chorus_drama - g.drama)

/-- Theorem stating the number of students in both band and chorus but not in drama -/
theorem lincoln_high_school_groups (g : SchoolGroups) 
  (h1 : g.total = 300)
  (h2 : g.band = 80)
  (h3 : g.chorus = 120)
  (h4 : g.drama = 50)
  (h5 : g.band_chorus_drama = 200) :
  students_in_band_and_chorus_not_drama g = 50 := by
  sorry

end NUMINAMATH_CALUDE_lincoln_high_school_groups_l1400_140016


namespace NUMINAMATH_CALUDE_number_puzzle_l1400_140039

theorem number_puzzle : ∃! x : ℝ, 150 - x = x + 68 :=
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1400_140039


namespace NUMINAMATH_CALUDE_line_intersection_l1400_140021

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line being contained in a plane
variable (contains : Plane → Line → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the property of a line intersecting another line
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem line_intersection
  (a b c : Line) (α β : Plane)
  (h1 : skew a b)
  (h2 : contains α a)
  (h3 : contains β b)
  (h4 : c = intersect α β) :
  intersects c a ∨ intersects c b :=
sorry

end NUMINAMATH_CALUDE_line_intersection_l1400_140021


namespace NUMINAMATH_CALUDE_typhoon_tree_difference_l1400_140004

def initial_trees : ℕ := 3
def dead_trees : ℕ := 13

theorem typhoon_tree_difference : dead_trees - (initial_trees - dead_trees) = 13 := by
  sorry

end NUMINAMATH_CALUDE_typhoon_tree_difference_l1400_140004


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l1400_140084

/-- The number of cakes the baker still has after selling some -/
def remaining_cakes (cakes_made : ℕ) (cakes_sold : ℕ) : ℕ :=
  cakes_made - cakes_sold

/-- Theorem stating that the baker has 139 cakes remaining -/
theorem baker_remaining_cakes :
  remaining_cakes 149 10 = 139 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l1400_140084


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1400_140097

def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 1]
def c : Fin 2 → ℝ := ![-5, 1]

theorem parallel_vectors_k_value (k : ℝ) :
  (∀ i : Fin 2, ∃ t : ℝ, a i + k * b i = t * c i) →
  k = 1/2 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1400_140097


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1400_140012

theorem right_triangle_third_side (a b : ℝ) (ha : a = 6) (hb : b = 10) :
  ∃ c : ℝ, (c = 2 * Real.sqrt 34 ∨ c = 8) ∧
    (c^2 = a^2 + b^2 ∨ b^2 = a^2 + c^2 ∨ a^2 = b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1400_140012


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l1400_140029

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l1400_140029


namespace NUMINAMATH_CALUDE_xe_pow_x_strictly_increasing_l1400_140064

/-- The function f(x) = xe^x is strictly increasing on the interval (-1, +∞) -/
theorem xe_pow_x_strictly_increasing :
  ∀ x₁ x₂, -1 < x₁ → x₁ < x₂ → x₁ * Real.exp x₁ < x₂ * Real.exp x₂ := by
  sorry

end NUMINAMATH_CALUDE_xe_pow_x_strictly_increasing_l1400_140064


namespace NUMINAMATH_CALUDE_computer_price_difference_l1400_140037

/-- The price difference between two stores selling the same computer with different prices and discounts -/
theorem computer_price_difference (price1 : ℝ) (discount1 : ℝ) (price2 : ℝ) (discount2 : ℝ) 
  (h1 : price1 = 950) (h2 : discount1 = 0.06) (h3 : price2 = 920) (h4 : discount2 = 0.05) :
  abs (price1 * (1 - discount1) - price2 * (1 - discount2)) = 19 :=
by sorry

end NUMINAMATH_CALUDE_computer_price_difference_l1400_140037


namespace NUMINAMATH_CALUDE_michael_anna_ratio_is_500_251_l1400_140089

/-- Sum of odd integers from 1 to n -/
def sumOddIntegers (n : ℕ) : ℕ := n^2

/-- Sum of integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The ratio of Michael's sum to Anna's sum -/
def michaelAnnaRatio : ℚ :=
  (sumOddIntegers 500 : ℚ) / (sumIntegers 500 : ℚ)

theorem michael_anna_ratio_is_500_251 :
  michaelAnnaRatio = 500 / 251 := by
  sorry

end NUMINAMATH_CALUDE_michael_anna_ratio_is_500_251_l1400_140089


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_leg_length_l1400_140038

/-- A trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  radius : ℝ
  base1 : ℝ
  base2 : ℝ
  centerInside : Bool

/-- The average length of the legs of the trapezoid squared -/
def averageLegLengthSquared (t : InscribedTrapezoid) : ℝ :=
  sorry

/-- Theorem: For a trapezoid JANE inscribed in a circle of radius 25 with the center inside,
    if the bases are 14 and 30, then the average leg length squared is 2000 -/
theorem inscribed_trapezoid_leg_length
    (t : InscribedTrapezoid)
    (h1 : t.radius = 25)
    (h2 : t.base1 = 14)
    (h3 : t.base2 = 30)
    (h4 : t.centerInside = true) :
  averageLegLengthSquared t = 2000 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_trapezoid_leg_length_l1400_140038


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_exists_largest_n_binomial_sum_largest_n_is_seven_l1400_140027

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) → n ≤ 7 :=
by sorry

theorem exists_largest_n_binomial_sum : 
  ∃ n : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) ∧ 
  (∀ m : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 m) → m ≤ n) :=
by sorry

theorem largest_n_is_seven : 
  (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 7) ∧
  (∀ m : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 m) → m ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_exists_largest_n_binomial_sum_largest_n_is_seven_l1400_140027


namespace NUMINAMATH_CALUDE_square_of_sqrt_three_l1400_140055

theorem square_of_sqrt_three (x : ℝ) : 
  Real.sqrt (x + 3) = 3 → (x + 3)^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sqrt_three_l1400_140055


namespace NUMINAMATH_CALUDE_integral_depends_only_on_ratio_implies_inverse_function_l1400_140009

open Set
open Function
open MeasureTheory
open Interval

theorem integral_depends_only_on_ratio_implies_inverse_function
  (f : ℝ → ℝ) (h_cont : Continuous f) (h_pos : ∀ x, 0 < x → f x ≠ 0) :
  (∀ a b : ℝ, 0 < a → 0 < b →
    ∃ g : ℝ → ℝ, (∫ x in a..b, f x) = g (b / a)) →
  ∃ c : ℝ, ∀ x > 0, f x = c / x :=
sorry

end NUMINAMATH_CALUDE_integral_depends_only_on_ratio_implies_inverse_function_l1400_140009


namespace NUMINAMATH_CALUDE_range_of_a_l1400_140052

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + abs x)

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | f (x^2 + 1) > f (a * x)}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1/2) (1/2) → x ∈ A a) →
  a ∈ Set.Ioo (-5/2) (5/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1400_140052


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1400_140075

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = 3 * x - 10) → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1400_140075


namespace NUMINAMATH_CALUDE_sum_of_cubes_equals_hundred_l1400_140053

theorem sum_of_cubes_equals_hundred : (1 : ℕ)^3 + 2^3 + 3^3 + 4^3 = 10^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equals_hundred_l1400_140053


namespace NUMINAMATH_CALUDE_weight_calculation_l1400_140049

/-- Given a box containing 16 equal weights with a total weight of 17.88 kg,
    and an empty box weighing 0.6 kg, the weight of 7 such weights is 7.56 kg. -/
theorem weight_calculation (total_weight : ℝ) (box_weight : ℝ) (num_weights : ℕ) (target_weights : ℕ)
    (hw : total_weight = 17.88)
    (hb : box_weight = 0.6)
    (hn : num_weights = 16)
    (ht : target_weights = 7) :
    (total_weight - box_weight) / num_weights * target_weights = 7.56 := by
  sorry

end NUMINAMATH_CALUDE_weight_calculation_l1400_140049


namespace NUMINAMATH_CALUDE_no_intersection_points_l1400_140060

/-- The number of intersection points between r = 3 cos θ and r = 6 sin θ is 0 -/
theorem no_intersection_points : ∀ θ : ℝ, 
  ¬∃ r : ℝ, (r = 3 * Real.cos θ ∧ r = 6 * Real.sin θ) :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_points_l1400_140060


namespace NUMINAMATH_CALUDE_star_interior_angle_sum_formula_l1400_140081

/-- An n-pointed star is formed from a convex n-gon by extending each side k
    to intersect with side k+3 (modulo n). This function calculates the
    sum of interior angles at the n vertices of the resulting star. -/
def starInteriorAngleSum (n : ℕ) : ℝ :=
  180 * (n - 6 : ℝ)

/-- Theorem stating that for an n-pointed star (n ≥ 5), the sum of
    interior angles at the n vertices is 180(n-6) degrees. -/
theorem star_interior_angle_sum_formula {n : ℕ} (h : n ≥ 5) :
  starInteriorAngleSum n = 180 * (n - 6 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_star_interior_angle_sum_formula_l1400_140081


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_eq_three_abs_minus_two_l1400_140098

theorem product_of_solutions_abs_eq_three_abs_minus_two (x : ℝ) :
  (∃ x₁ x₂ : ℝ, (|x₁| = 3 * (|x₁| - 2) ∧ |x₂| = 3 * (|x₂| - 2) ∧ x₁ ≠ x₂) →
  x₁ * x₂ = -9) :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_eq_three_abs_minus_two_l1400_140098


namespace NUMINAMATH_CALUDE_rectangles_equal_area_reciprocal_proportion_l1400_140070

/-- Two rectangles with equal areas have reciprocally proportional sides -/
theorem rectangles_equal_area_reciprocal_proportion
  (a b c d : ℝ)
  (h : a * b = c * d)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a / c = d / b :=
by sorry

end NUMINAMATH_CALUDE_rectangles_equal_area_reciprocal_proportion_l1400_140070


namespace NUMINAMATH_CALUDE_max_interior_angles_less_than_120_is_5_l1400_140078

/-- A convex polygon with 532 sides -/
structure ConvexPolygon532 where
  sides : ℕ
  convex : Bool
  sidesEq532 : sides = 532

/-- The maximum number of interior angles less than 120° in a ConvexPolygon532 -/
def maxInteriorAnglesLessThan120 (p : ConvexPolygon532) : ℕ :=
  5

/-- Theorem stating that the maximum number of interior angles less than 120° in a ConvexPolygon532 is 5 -/
theorem max_interior_angles_less_than_120_is_5 (p : ConvexPolygon532) :
  maxInteriorAnglesLessThan120 p = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_interior_angles_less_than_120_is_5_l1400_140078


namespace NUMINAMATH_CALUDE_inscribed_circle_triangle_perimeter_l1400_140068

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of XT, where T is the tangency point on XY -/
  xt : ℝ
  /-- The length of TY, where T is the tangency point on XY -/
  ty : ℝ

/-- Calculate the perimeter of a triangle with an inscribed circle -/
def perimeter (t : InscribedCircleTriangle) : ℝ :=
  sorry

theorem inscribed_circle_triangle_perimeter
  (t : InscribedCircleTriangle)
  (h_r : t.r = 24)
  (h_xt : t.xt = 26)
  (h_ty : t.ty = 31) :
  perimeter t = 345 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_triangle_perimeter_l1400_140068


namespace NUMINAMATH_CALUDE_M_equals_P_l1400_140050

-- Define set M
def M : Set ℝ := {x | ∃ a : ℝ, x = a^2 + 1}

-- Define set P
def P : Set ℝ := {y | ∃ b : ℝ, y = b^2 - 4*b + 5}

-- Theorem stating that M equals P
theorem M_equals_P : M = P := by sorry

end NUMINAMATH_CALUDE_M_equals_P_l1400_140050


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1400_140092

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 90 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ (x : ℚ), x > 0 ∧ total = a * x + b * x + c * x ∧ min (a * x) (min (b * x) (c * x)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1400_140092


namespace NUMINAMATH_CALUDE_robot_position_l1400_140063

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the Cartesian plane defined by y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The robot's path is defined as the set of points equidistant from two given points -/
def RobotPath (A B : Point) : Set Point :=
  {P : Point | (P.x - A.x)^2 + (P.y - A.y)^2 = (P.x - B.x)^2 + (P.y - B.y)^2}

/-- Check if a point is on a line -/
def isOnLine (P : Point) (L : Line) : Prop :=
  P.y = L.m * P.x + L.b

theorem robot_position (a : ℝ) : 
  let A : Point := ⟨a, 0⟩
  let B : Point := ⟨0, 1⟩
  let L : Line := ⟨1, 1⟩  -- y = x + 1
  (∀ P ∈ RobotPath A B, ¬isOnLine P L) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_robot_position_l1400_140063


namespace NUMINAMATH_CALUDE_rain_probability_l1400_140094

/-- The probability of rain on Friday -/
def prob_rain_friday : ℝ := 0.40

/-- The probability of rain on Monday -/
def prob_rain_monday : ℝ := 0.35

/-- The probability of rain on both Friday and Monday -/
def prob_rain_both : ℝ := prob_rain_friday * prob_rain_monday

theorem rain_probability : prob_rain_both = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l1400_140094


namespace NUMINAMATH_CALUDE_ruths_school_schedule_l1400_140006

/-- Ruth's school schedule problem -/
theorem ruths_school_schedule 
  (days_per_week : ℕ) 
  (math_class_percentage : ℚ) 
  (math_class_hours_per_week : ℕ) 
  (h1 : days_per_week = 5)
  (h2 : math_class_percentage = 1/4)
  (h3 : math_class_hours_per_week = 10) :
  let total_school_hours_per_week := math_class_hours_per_week / math_class_percentage
  let school_hours_per_day := total_school_hours_per_week / days_per_week
  school_hours_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_ruths_school_schedule_l1400_140006


namespace NUMINAMATH_CALUDE_x_y_equation_l1400_140003

theorem x_y_equation (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) : 
  (1/3 : ℚ) * x^8 * y^9 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_x_y_equation_l1400_140003


namespace NUMINAMATH_CALUDE_circle_condition_l1400_140083

/-- 
Given a real number a and the equation ax^2 + ay^2 - 4(a-1)x + 4y = 0,
this theorem states that the equation represents a circle if and only if a ≠ 0.
-/
theorem circle_condition (a : ℝ) : 
  (∃ (h k r : ℝ), r > 0 ∧ 
    ∀ (x y : ℝ), ax^2 + ay^2 - 4*(a-1)*x + 4*y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) ↔ 
  a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l1400_140083


namespace NUMINAMATH_CALUDE_linear_function_fits_points_l1400_140040

-- Define the set of points
def points : List (ℝ × ℝ) := [(0, 150), (1, 120), (2, 90), (3, 60), (4, 30)]

-- Define the linear function
def f (x : ℝ) : ℝ := -30 * x + 150

-- Theorem statement
theorem linear_function_fits_points : 
  ∀ (point : ℝ × ℝ), point ∈ points → f point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_fits_points_l1400_140040


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l1400_140019

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x - 1)

theorem f_derivative_at_one : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l1400_140019


namespace NUMINAMATH_CALUDE_problem_solution_l1400_140011

-- Define the propositions
variable (p₁ p₂ p₃ p₄ : Prop)

-- Define the conditions
axiom h1 : p₁
axiom h2 : ¬p₂
axiom h3 : ¬p₃
axiom h4 : p₄

-- Theorem to prove
theorem problem_solution :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1400_140011


namespace NUMINAMATH_CALUDE_common_factor_proof_l1400_140002

theorem common_factor_proof (x y : ℝ) (m n : ℕ) :
  ∃ (k : ℝ), 8 * x^m * y^(n-1) - 12 * x^(3*m) * y^n = k * (4 * x^m * y^(n-1)) ∧
              k ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_common_factor_proof_l1400_140002


namespace NUMINAMATH_CALUDE_select_five_from_eight_l1400_140000

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l1400_140000


namespace NUMINAMATH_CALUDE_vacation_cost_from_dog_walking_vacation_cost_proof_l1400_140061

/-- Calculates the total cost of a vacation based on dog walking earnings --/
theorem vacation_cost_from_dog_walking 
  (start_charge : ℚ)
  (per_block_charge : ℚ)
  (num_dogs : ℕ)
  (total_blocks : ℕ)
  (family_members : ℕ)
  (h1 : start_charge = 2)
  (h2 : per_block_charge = 5/4)
  (h3 : num_dogs = 20)
  (h4 : total_blocks = 128)
  (h5 : family_members = 5)
  : ℚ
  :=
  let total_earnings := start_charge * num_dogs + per_block_charge * total_blocks
  total_earnings

theorem vacation_cost_proof
  (start_charge : ℚ)
  (per_block_charge : ℚ)
  (num_dogs : ℕ)
  (total_blocks : ℕ)
  (family_members : ℕ)
  (h1 : start_charge = 2)
  (h2 : per_block_charge = 5/4)
  (h3 : num_dogs = 20)
  (h4 : total_blocks = 128)
  (h5 : family_members = 5)
  : vacation_cost_from_dog_walking start_charge per_block_charge num_dogs total_blocks family_members h1 h2 h3 h4 h5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_from_dog_walking_vacation_cost_proof_l1400_140061


namespace NUMINAMATH_CALUDE_log_max_min_sum_l1400_140001

theorem log_max_min_sum (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (let f := fun x => Real.log x / Real.log a
   max (f a) (f (2 * a)) + min (f a) (f (2 * a)) = 3) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_log_max_min_sum_l1400_140001
