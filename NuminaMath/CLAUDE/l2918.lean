import Mathlib

namespace NUMINAMATH_CALUDE_gcd_ABC_l2918_291868

-- Define the constants
def a : ℕ := 177
def b : ℕ := 173

-- Define A, B, and C using the given formulas
def A : ℕ := a^5 + (a*b) * b^3 - b^5
def B : ℕ := b^5 + (a*b) * a^3 - a^5
def C : ℕ := b^4 + (a*b)^2 + a^4

-- State the theorem
theorem gcd_ABC : 
  Nat.gcd A C = 30637 ∧ Nat.gcd B C = 30637 := by
  sorry

end NUMINAMATH_CALUDE_gcd_ABC_l2918_291868


namespace NUMINAMATH_CALUDE_zeta_sum_seventh_power_l2918_291818

theorem zeta_sum_seventh_power (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 1)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 3)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 7) :
  ζ₁^7 + ζ₂^7 + ζ₃^7 = 71 := by
  sorry

end NUMINAMATH_CALUDE_zeta_sum_seventh_power_l2918_291818


namespace NUMINAMATH_CALUDE_sqrt_sum_greater_than_one_l2918_291828

theorem sqrt_sum_greater_than_one (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt a + Real.sqrt b > 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_greater_than_one_l2918_291828


namespace NUMINAMATH_CALUDE_partial_square_division_l2918_291835

/-- Represents a square with a side length and a removed portion. -/
structure PartialSquare where
  side_length : ℝ
  removed_fraction : ℝ

/-- Represents a division of the remaining area into parts. -/
structure AreaDivision where
  num_parts : ℕ
  area_per_part : ℝ

/-- Theorem stating that a square with side length 4 and one fourth removed
    can be divided into four equal parts with area 3 each. -/
theorem partial_square_division (s : PartialSquare)
  (h1 : s.side_length = 4)
  (h2 : s.removed_fraction = 1/4) :
  ∃ (d : AreaDivision), 
    d.num_parts = 4 ∧ 
    d.area_per_part = 3 ∧
    d.num_parts * d.area_per_part = s.side_length^2 - s.side_length^2 * s.removed_fraction :=
by sorry

end NUMINAMATH_CALUDE_partial_square_division_l2918_291835


namespace NUMINAMATH_CALUDE_dividend_calculation_l2918_291841

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 20)
  (h2 : quotient = 8)
  (h3 : remainder = 6) :
  divisor * quotient + remainder = 166 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2918_291841


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2918_291863

theorem sum_of_three_numbers (a b c : ℝ) : 
  a ≤ b → b ≤ c → b = 10 → 
  (a + b + c) / 3 = a + 5 → 
  (a + b + c) / 3 = c - 20 → 
  a + b + c = 75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2918_291863


namespace NUMINAMATH_CALUDE_hyperbola_center_correct_l2918_291897

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * x - 8)^2 / 9^2 - (5 * y + 5)^2 / 7^2 = 1

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (2, -1)

/-- Theorem stating that hyperbola_center is the center of the hyperbola defined by hyperbola_equation -/
theorem hyperbola_center_correct :
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    hyperbola_equation (x - hyperbola_center.1) (y - hyperbola_center.2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_correct_l2918_291897


namespace NUMINAMATH_CALUDE_product_of_fractions_l2918_291873

/-- Prove that the product of 2/3 and 1 4/9 is equal to 26/27 -/
theorem product_of_fractions :
  (2 : ℚ) / 3 * (1 + 4 / 9) = 26 / 27 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2918_291873


namespace NUMINAMATH_CALUDE_abs_neg_three_not_pm_three_l2918_291824

theorem abs_neg_three_not_pm_three : ¬(|(-3 : ℤ)| = 3 ∧ |(-3 : ℤ)| = -3) := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_not_pm_three_l2918_291824


namespace NUMINAMATH_CALUDE_table_tennis_tournament_impossibility_l2918_291843

theorem table_tennis_tournament_impossibility (k : ℕ) (h : k > 0) :
  let participants := 2 * k
  let total_matches := k * (2 * k - 1)
  let total_judgements := 2 * total_matches
  ¬ ∃ (judgements_per_participant : ℕ),
    (judgements_per_participant * participants = total_judgements ∧
     judgements_per_participant * 2 = 2 * k - 1) :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_tournament_impossibility_l2918_291843


namespace NUMINAMATH_CALUDE_reflection_problem_l2918_291859

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x - y + 7 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define point N
def N : ℝ × ℝ := (1, 0)

-- Define the intersection point M
def M : ℝ × ℝ := (-2, 1)

-- Define the symmetric point P
def P : ℝ × ℝ := (-2, -1)

-- Define line l₃
def l₃ (x y : ℝ) : Prop := y = (1/3) * x - (1/3)

-- Define the parallel lines at distance √10 from l₃
def parallel_line₁ (x y : ℝ) : Prop := y = (1/3) * x + 3
def parallel_line₂ (x y : ℝ) : Prop := y = (1/3) * x - (11/3)

theorem reflection_problem :
  (∀ x y, l₁ x y ∧ l₂ x y → (x, y) = M) ∧
  P = (-2, -1) ∧
  (∀ x y, l₃ x y ↔ y = (1/3) * x - (1/3)) ∧
  (∀ x y, (parallel_line₁ x y ∨ parallel_line₂ x y) ↔
    ∃ d, d = Real.sqrt 10 ∧ 
    (y - ((1/3) * x - (1/3)))^2 / (1 + (1/3)^2) = d^2) :=
by sorry

end NUMINAMATH_CALUDE_reflection_problem_l2918_291859


namespace NUMINAMATH_CALUDE_zeros_product_greater_than_e_squared_l2918_291823

/-- Given a function f(x) = (ln x) / x - a with two zeros m and n, prove that mn > e² -/
theorem zeros_product_greater_than_e_squared (a : ℝ) (m n : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (Real.log x) / x - a = 0 ∧ (Real.log y) / y - a = 0) →
  (Real.log m) / m - a = 0 →
  (Real.log n) / n - a = 0 →
  m * n > Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_zeros_product_greater_than_e_squared_l2918_291823


namespace NUMINAMATH_CALUDE_namjoon_cookies_l2918_291883

/-- The number of cookies Namjoon had initially -/
def initial_cookies : ℕ := 24

/-- The number of cookies Namjoon ate -/
def eaten_cookies : ℕ := 8

/-- The number of cookies Namjoon gave to Hoseok -/
def given_cookies : ℕ := 7

/-- The number of cookies left after eating and giving away -/
def remaining_cookies : ℕ := 9

theorem namjoon_cookies : 
  initial_cookies - eaten_cookies - given_cookies = remaining_cookies :=
by sorry

end NUMINAMATH_CALUDE_namjoon_cookies_l2918_291883


namespace NUMINAMATH_CALUDE_unique_integer_divisibility_l2918_291899

theorem unique_integer_divisibility (n : ℕ) : 
  n > 1 → (∃ k : ℕ, (2^n + 1) = k * n^2) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisibility_l2918_291899


namespace NUMINAMATH_CALUDE_x_plus_y_values_l2918_291893

theorem x_plus_y_values (x y : ℝ) (h1 : -x = 3) (h2 : |y| = 5) :
  x + y = -8 ∨ x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l2918_291893


namespace NUMINAMATH_CALUDE_expression_domain_l2918_291814

def expression_defined (x : ℝ) : Prop :=
  x + 2 > 0 ∧ 5 - x > 0

theorem expression_domain : ∀ x : ℝ, expression_defined x ↔ -2 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_domain_l2918_291814


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l2918_291822

theorem matrix_equation_proof : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![4, 4; 2, 4]
  N^4 - 3 • N^3 + 3 • N^2 - N = !![16, 24; 8, 12] := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l2918_291822


namespace NUMINAMATH_CALUDE_points_collinear_l2918_291861

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the properties of the triangle
def is_acute_angled (t : Triangle) : Prop := sorry

-- Define the angle A to be 60°
def angle_A_is_60_degrees (t : Triangle) : Prop := sorry

-- Define the orthocenter H
def orthocenter (t : Triangle) : Point := sorry

-- Define point M
def point_M (t : Triangle) (H : Point) : Point := sorry

-- Define point N
def point_N (t : Triangle) (H : Point) : Point := sorry

-- Define the circumcenter O
def circumcenter (t : Triangle) : Point := sorry

-- Define collinearity
def collinear (P Q R S : Point) : Prop := sorry

-- Theorem statement
theorem points_collinear (t : Triangle) (H : Point) (M N O : Point) :
  is_acute_angled t →
  angle_A_is_60_degrees t →
  H = orthocenter t →
  M = point_M t H →
  N = point_N t H →
  O = circumcenter t →
  collinear M N H O :=
sorry

end NUMINAMATH_CALUDE_points_collinear_l2918_291861


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_a_plus_2b_necessary_not_sufficient_l2918_291813

/-- For all x in [0, 1], a+2b>0 is a necessary but not sufficient condition for ax+b>0 to always hold true -/
theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (a * x + b > 0)) ↔ (b > 0 ∧ a + b > 0) :=
by sorry

/-- a+2b>0 is necessary but not sufficient for the above condition -/
theorem a_plus_2b_necessary_not_sufficient (a b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (a * x + b > 0)) → (a + 2*b > 0) ∧
  ¬(∀ a b : ℝ, (a + 2*b > 0) → (∀ x : ℝ, x ∈ Set.Icc 0 1 → (a * x + b > 0))) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_a_plus_2b_necessary_not_sufficient_l2918_291813


namespace NUMINAMATH_CALUDE_michael_twice_jacob_age_l2918_291821

/-- Given that:
    - Jacob is 12 years younger than Michael
    - Jacob will be 13 years old in 4 years
    - At some point in the future, Michael will be twice as old as Jacob
    This theorem proves that Michael will be twice as old as Jacob in 3 years. -/
theorem michael_twice_jacob_age (jacob_age : ℕ) (michael_age : ℕ) (years_until_twice : ℕ) :
  michael_age = jacob_age + 12 →
  jacob_age + 4 = 13 →
  michael_age + years_until_twice = 2 * (jacob_age + years_until_twice) →
  years_until_twice = 3 := by
  sorry

end NUMINAMATH_CALUDE_michael_twice_jacob_age_l2918_291821


namespace NUMINAMATH_CALUDE_total_bulbs_needed_l2918_291878

theorem total_bulbs_needed (medium_lights : ℕ) (small_bulbs : ℕ) (medium_bulbs : ℕ) (large_bulbs : ℕ) :
  medium_lights = 12 →
  small_bulbs = 1 →
  medium_bulbs = 2 →
  large_bulbs = 3 →
  (medium_lights * small_bulbs + 10) * small_bulbs +
  medium_lights * medium_bulbs +
  (2 * medium_lights) * large_bulbs = 118 := by
  sorry

end NUMINAMATH_CALUDE_total_bulbs_needed_l2918_291878


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l2918_291894

theorem triangle_abc_proof (b c : ℝ) (A : Real) (hb : b = 1) (hc : c = 2) (hA : A = 60 * π / 180) :
  ∃ (a : ℝ) (B : Real),
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A ∧
    a = Real.sqrt 3 ∧
    Real.cos B = (a ^ 2 + c ^ 2 - b ^ 2) / (2 * a * c) ∧
    B = 30 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_proof_l2918_291894


namespace NUMINAMATH_CALUDE_min_coach_handshakes_zero_l2918_291858

/-- The total number of handshakes in the gymnastics competition -/
def total_handshakes : ℕ := 325

/-- The number of gymnasts -/
def num_gymnasts : ℕ := 26

/-- The number of handshakes between gymnasts -/
def gymnast_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of handshakes involving coaches -/
def coach_handshakes (total : ℕ) (n : ℕ) : ℕ := total - gymnast_handshakes n

theorem min_coach_handshakes_zero :
  coach_handshakes total_handshakes num_gymnasts = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_zero_l2918_291858


namespace NUMINAMATH_CALUDE_integral_equals_two_minus_three_ln_three_l2918_291898

/-- Given that the solution set of the inequality 1 - 3/(x+a) < 0 is (-1,2),
    prove that the integral from 0 to 2 of (1 - 3/(x+a)) dx equals 2 - 3 * ln 3 -/
theorem integral_equals_two_minus_three_ln_three 
  (a : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 2 = {x : ℝ | 1 - 3 / (x + a) < 0}) : 
  ∫ x in (0:ℝ)..2, (1 - 3 / (x + a)) = 2 - 3 * Real.log 3 := by
  sorry

#check integral_equals_two_minus_three_ln_three

end NUMINAMATH_CALUDE_integral_equals_two_minus_three_ln_three_l2918_291898


namespace NUMINAMATH_CALUDE_farm_sheep_count_l2918_291805

/-- Given a farm with sheep and horses, prove that the number of sheep is 16 -/
theorem farm_sheep_count (sheep horses : ℕ) (horse_food_per_day total_horse_food : ℕ) : 
  (sheep : ℚ) / horses = 2 / 7 →
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  horses * horse_food_per_day = total_horse_food →
  sheep = 16 := by
sorry

end NUMINAMATH_CALUDE_farm_sheep_count_l2918_291805


namespace NUMINAMATH_CALUDE_oliver_shelf_capacity_l2918_291852

/-- The number of books Oliver can fit on a shelf -/
def books_per_shelf (total_books librarian_books shelves : ℕ) : ℕ :=
  (total_books - librarian_books) / shelves

/-- Theorem: Oliver can fit 4 books on a shelf -/
theorem oliver_shelf_capacity :
  books_per_shelf 46 10 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oliver_shelf_capacity_l2918_291852


namespace NUMINAMATH_CALUDE_solution_to_equation_l2918_291857

theorem solution_to_equation (x : ℝ) : 
  (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ 
  (x = 2 ∨ x = -2) := by
sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2918_291857


namespace NUMINAMATH_CALUDE_trucks_needed_l2918_291830

def total_apples : ℕ := 42
def transported_apples : ℕ := 22
def truck_capacity : ℕ := 4

theorem trucks_needed : 
  (total_apples - transported_apples) / truck_capacity = 5 := by
  sorry

end NUMINAMATH_CALUDE_trucks_needed_l2918_291830


namespace NUMINAMATH_CALUDE_expression_evaluation_l2918_291819

theorem expression_evaluation : 
  (((3 : ℚ) + 6 + 9) / ((2 : ℚ) + 5 + 8) - ((2 : ℚ) + 5 + 8) / ((3 : ℚ) + 6 + 9)) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2918_291819


namespace NUMINAMATH_CALUDE_no_non_zero_integer_solution_l2918_291820

theorem no_non_zero_integer_solution :
  ∀ (a b c n : ℤ), 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_non_zero_integer_solution_l2918_291820


namespace NUMINAMATH_CALUDE_football_game_cost_l2918_291881

def total_spent : ℚ := 35.52
def strategy_game_cost : ℚ := 9.46
def batman_game_cost : ℚ := 12.04

theorem football_game_cost :
  total_spent - strategy_game_cost - batman_game_cost = 13.02 := by
  sorry

end NUMINAMATH_CALUDE_football_game_cost_l2918_291881


namespace NUMINAMATH_CALUDE_apples_distribution_l2918_291812

/-- Given 48 apples distributed evenly among 7 children, prove that 1 child receives fewer than 7 apples -/
theorem apples_distribution (total_apples : Nat) (num_children : Nat) 
  (h1 : total_apples = 48) 
  (h2 : num_children = 7) : 
  (num_children - (total_apples % num_children)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_apples_distribution_l2918_291812


namespace NUMINAMATH_CALUDE_new_time_ratio_l2918_291832

-- Define the distances and speed ratio
def first_trip_distance : ℝ := 100
def second_trip_distance : ℝ := 500
def speed_ratio : ℝ := 4

-- Theorem statement
theorem new_time_ratio (v : ℝ) (hv : v > 0) :
  let t1 := first_trip_distance / v
  let t2 := second_trip_distance / (speed_ratio * v)
  t2 / t1 = 1.25 := by
sorry

end NUMINAMATH_CALUDE_new_time_ratio_l2918_291832


namespace NUMINAMATH_CALUDE_a_power_sum_l2918_291816

theorem a_power_sum (a : ℂ) (h : a^2 - a + 1 = 0) : a^10 + a^20 + a^30 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_power_sum_l2918_291816


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l2918_291892

theorem quadratic_root_implies_m (m : ℚ) : 
  ((-2 : ℚ)^2 - m*(-2) - 3 = 0) → m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l2918_291892


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2918_291801

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 3 ∧ x₂ = -5 ∧ 
  x₁^2 + 2*x₁ - 15 = 0 ∧ 
  x₂^2 + 2*x₂ - 15 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2918_291801


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2918_291867

theorem complex_magnitude_problem (z : ℂ) (h : (1 - I) * z - 3 * I = 1) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2918_291867


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l2918_291850

def digit_sum (n : ℕ) : ℕ := sorry

def is_prime (n : ℕ) : Prop := sorry

theorem smallest_prime_with_digit_sum_23 :
  (∀ p : ℕ, is_prime p ∧ digit_sum p = 23 → p ≥ 599) ∧
  is_prime 599 ∧
  digit_sum 599 = 23 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l2918_291850


namespace NUMINAMATH_CALUDE_correct_calculation_l2918_291879

theorem correct_calculation : -7 + 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2918_291879


namespace NUMINAMATH_CALUDE_parkway_elementary_soccer_l2918_291825

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (boys_soccer_percentage : ℚ) :
  total_students = 420 →
  boys = 312 →
  soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  (total_students - boys - (soccer_players - (boys_soccer_percentage * soccer_players).floor)) = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_parkway_elementary_soccer_l2918_291825


namespace NUMINAMATH_CALUDE_probability_of_prime_ball_l2918_291839

def ball_numbers : List Nat := [3, 4, 5, 6, 7, 8, 11, 13]

def is_prime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d ≤ 1 || n % (d + 2) ≠ 0)

def count_primes (numbers : List Nat) : Nat :=
  (numbers.filter is_prime).length

theorem probability_of_prime_ball :
  (count_primes ball_numbers : Rat) / (ball_numbers.length : Rat) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_prime_ball_l2918_291839


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2918_291840

-- Define a sequence type
def Sequence := ℕ → ℝ

-- Define the property of a sequence satisfying a_n = 2a_{n-1} for n ≥ 2
def SatisfiesCondition (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

-- Define a geometric sequence with common ratio 2
def IsGeometricSequenceWithRatio2 (a : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n

-- Theorem statement
theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometricSequenceWithRatio2 a → SatisfiesCondition a) ∧
  (∃ a : Sequence, SatisfiesCondition a ∧ ¬IsGeometricSequenceWithRatio2 a) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2918_291840


namespace NUMINAMATH_CALUDE_lcm_problem_l2918_291808

theorem lcm_problem (a b c : ℕ+) (h1 : b = 30) (h2 : c = 40) (h3 : Nat.lcm (Nat.lcm a.val b.val) c.val = 120) : a = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2918_291808


namespace NUMINAMATH_CALUDE_test_scores_analysis_l2918_291886

def benchmark : ℝ := 85

def deviations : List ℝ := [8, -3, 12, -7, -10, -4, -8, 1, 0, 10]

def actual_scores : List ℝ := deviations.map (λ x => benchmark + x)

theorem test_scores_analysis :
  let max_score := actual_scores.maximum
  let min_score := actual_scores.minimum
  let avg_score := benchmark + (deviations.sum / deviations.length)
  (max_score = 97 ∧ min_score = 75) ∧ avg_score = 84.9 := by
  sorry

end NUMINAMATH_CALUDE_test_scores_analysis_l2918_291886


namespace NUMINAMATH_CALUDE_subtract_negative_self_l2918_291855

theorem subtract_negative_self (a : ℤ) : -a - (-a) = 0 := by sorry

end NUMINAMATH_CALUDE_subtract_negative_self_l2918_291855


namespace NUMINAMATH_CALUDE_exam_scores_difference_l2918_291815

/-- Given five exam scores with specific properties, prove that the absolute difference between two of them is 18. -/
theorem exam_scores_difference (x y : ℝ) : 
  (x + y + 105 + 109 + 110) / 5 = 108 →
  ((x - 108)^2 + (y - 108)^2 + (105 - 108)^2 + (109 - 108)^2 + (110 - 108)^2) / 5 = 35.2 →
  |x - y| = 18 := by
sorry

end NUMINAMATH_CALUDE_exam_scores_difference_l2918_291815


namespace NUMINAMATH_CALUDE_area_between_parabola_and_line_l2918_291895

-- Define the parabola and line functions
def parabola (x : ℝ) : ℝ := x^2 - 1
def line (x : ℝ) : ℝ := x + 1

-- Define the region
def region : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem area_between_parabola_and_line :
  ∫ x in region, (line x - parabola x) = 9/2 := by
  sorry


end NUMINAMATH_CALUDE_area_between_parabola_and_line_l2918_291895


namespace NUMINAMATH_CALUDE_pascals_triangle_51st_row_third_number_l2918_291874

theorem pascals_triangle_51st_row_third_number : 
  (Nat.choose 51 2) = 1275 := by sorry

end NUMINAMATH_CALUDE_pascals_triangle_51st_row_third_number_l2918_291874


namespace NUMINAMATH_CALUDE_lisa_heavier_than_sam_l2918_291800

/-- Proves that Lisa is 7.8 pounds heavier than Sam given the specified conditions -/
theorem lisa_heavier_than_sam (jack sam lisa : ℝ) 
  (total_weight : jack + sam + lisa = 210)
  (jack_weight : jack = 52)
  (sam_jack_relation : jack = sam * 0.8)
  (lisa_jack_relation : lisa = jack * 1.4) :
  lisa - sam = 7.8 := by
  sorry

end NUMINAMATH_CALUDE_lisa_heavier_than_sam_l2918_291800


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l2918_291826

theorem root_sum_reciprocals (p q r s t : ℂ) : 
  p^5 + 10*p^4 + 20*p^3 + 15*p^2 + 8*p + 5 = 0 →
  q^5 + 10*q^4 + 20*q^3 + 15*q^2 + 8*q + 5 = 0 →
  r^5 + 10*r^4 + 20*r^3 + 15*r^2 + 8*r + 5 = 0 →
  s^5 + 10*s^4 + 20*s^3 + 15*s^2 + 8*s + 5 = 0 →
  t^5 + 10*t^4 + 20*t^3 + 15*t^2 + 8*t + 5 = 0 →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 4 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l2918_291826


namespace NUMINAMATH_CALUDE_updated_mean_after_corrections_l2918_291827

/-- Calculates the updated mean of a set of observations after correcting errors -/
theorem updated_mean_after_corrections (n : ℕ) (initial_mean : ℚ) 
  (n1 n2 n3 : ℕ) (error1 error2 error3 : ℚ) : 
  n = 50 → 
  initial_mean = 200 → 
  n1 = 20 → 
  n2 = 15 → 
  n3 = 15 → 
  error1 = -6 → 
  error2 = -5 → 
  error3 = 3 → 
  (initial_mean * n + n1 * error1 + n2 * error2 + n3 * error3) / n = 197 := by
  sorry

#eval (200 * 50 + 20 * (-6) + 15 * (-5) + 15 * 3) / 50

end NUMINAMATH_CALUDE_updated_mean_after_corrections_l2918_291827


namespace NUMINAMATH_CALUDE_area_is_two_l2918_291848

open Real MeasureTheory

noncomputable def area_bounded_by_curves : ℝ :=
  ∫ x in (1/Real.exp 1)..Real.exp 1, (1/x)

theorem area_is_two : area_bounded_by_curves = 2 := by
  sorry

end NUMINAMATH_CALUDE_area_is_two_l2918_291848


namespace NUMINAMATH_CALUDE_perpendicular_sum_l2918_291884

/-- Given vectors a and b in ℝ², if a + b is perpendicular to a, 
    then the second component of b is -1. -/
theorem perpendicular_sum (a b : ℝ × ℝ) (h : a = (1, 0)) :
  (a + b) • a = 0 → b.1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_sum_l2918_291884


namespace NUMINAMATH_CALUDE_violet_hiking_time_l2918_291888

/-- Calculates the maximum hiking time for Violet and her dog given their water consumption rates and Violet's water carrying capacity. -/
theorem violet_hiking_time (violet_rate : ℝ) (dog_rate : ℝ) (water_capacity : ℝ) :
  violet_rate = 800 →
  dog_rate = 400 →
  water_capacity = 4800 →
  (water_capacity / (violet_rate + dog_rate) : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_violet_hiking_time_l2918_291888


namespace NUMINAMATH_CALUDE_six_students_five_lectures_l2918_291871

/-- The number of ways students can choose lectures -/
def lecture_choices (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: 6 students choosing from 5 lectures results in 5^6 possibilities -/
theorem six_students_five_lectures :
  lecture_choices 6 5 = 5^6 := by
  sorry

end NUMINAMATH_CALUDE_six_students_five_lectures_l2918_291871


namespace NUMINAMATH_CALUDE_no_valid_solution_l2918_291869

theorem no_valid_solution : ¬∃ (Y : ℕ), Y > 0 ∧ 2*Y + Y + 3*Y = 14 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_solution_l2918_291869


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2918_291829

theorem quadratic_equation_solution :
  ∃ (m n p : ℕ) (x₁ x₂ : ℚ),
    -- The equation is satisfied by both solutions
    x₁ * (5 * x₁ - 11) = -2 ∧
    x₂ * (5 * x₂ - 11) = -2 ∧
    -- Solutions are in the required form
    x₁ = (m + Real.sqrt n) / p ∧
    x₂ = (m - Real.sqrt n) / p ∧
    -- m, n, and p have a greatest common divisor of 1
    Nat.gcd m (Nat.gcd n p) = 1 ∧
    -- Sum of m, n, and p is 102
    m + n + p = 102 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2918_291829


namespace NUMINAMATH_CALUDE_bert_profit_is_correct_l2918_291833

/-- Represents a product in Bert's shop -/
structure Product where
  price : ℝ
  tax_rate : ℝ

/-- Represents a customer's purchase -/
structure Purchase where
  products : List Product
  discount_rate : ℝ

/-- Calculates Bert's profit given the purchases -/
def calculate_profit (purchases : List Purchase) : ℝ :=
  sorry

/-- The actual purchases made by customers -/
def actual_purchases : List Purchase :=
  [
    { products := [
        { price := 90, tax_rate := 0.1 },
        { price := 50, tax_rate := 0.05 }
      ], 
      discount_rate := 0.1
    },
    { products := [
        { price := 30, tax_rate := 0.12 },
        { price := 20, tax_rate := 0.03 }
      ], 
      discount_rate := 0.15
    },
    { products := [
        { price := 15, tax_rate := 0.09 }
      ], 
      discount_rate := 0
    }
  ]

/-- Bert's profit per item -/
def profit_per_item : ℝ := 10

theorem bert_profit_is_correct : 
  calculate_profit actual_purchases = 50.05 :=
sorry

end NUMINAMATH_CALUDE_bert_profit_is_correct_l2918_291833


namespace NUMINAMATH_CALUDE_music_books_cost_l2918_291838

def total_budget : ℕ := 500
def maths_books : ℕ := 4
def maths_book_price : ℕ := 20
def science_book_price : ℕ := 10
def art_book_price : ℕ := 20

def science_books : ℕ := maths_books + 6
def art_books : ℕ := 2 * maths_books

def maths_cost : ℕ := maths_books * maths_book_price
def science_cost : ℕ := science_books * science_book_price
def art_cost : ℕ := art_books * art_book_price

def total_cost_except_music : ℕ := maths_cost + science_cost + art_cost

theorem music_books_cost (music_cost : ℕ) :
  music_cost = total_budget - total_cost_except_music →
  music_cost = 160 := by
  sorry

end NUMINAMATH_CALUDE_music_books_cost_l2918_291838


namespace NUMINAMATH_CALUDE_elevator_problem_l2918_291896

def elevator_ways (n : ℕ) (k : ℕ) (max_per_floor : ℕ) : ℕ :=
  sorry

theorem elevator_problem : elevator_ways 3 5 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_elevator_problem_l2918_291896


namespace NUMINAMATH_CALUDE_seventh_eleventh_150th_decimal_l2918_291831

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The period length of a rational number's decimal representation -/
def periodLength (q : ℚ) : ℕ := sorry

theorem seventh_eleventh_150th_decimal :
  (7 : ℚ) / 11 ∈ {q : ℚ | periodLength q = 2 ∧ decimalRepresentation q 150 = 3} := by
  sorry

end NUMINAMATH_CALUDE_seventh_eleventh_150th_decimal_l2918_291831


namespace NUMINAMATH_CALUDE_ticket_price_increase_l2918_291802

theorem ticket_price_increase (P V : ℝ) (h1 : P > 0) (h2 : V > 0) : 
  (P + 0.5 * P) * (0.8 * V) = 1.2 * (P * V) := by sorry

#check ticket_price_increase

end NUMINAMATH_CALUDE_ticket_price_increase_l2918_291802


namespace NUMINAMATH_CALUDE_point_M_coordinates_l2918_291854

def f (x : ℝ) : ℝ := 2 * x^2 + 1

theorem point_M_coordinates (x₀ y₀ : ℝ) :
  (∃ M : ℝ × ℝ, M.1 = x₀ ∧ M.2 = y₀ ∧ 
   (deriv f) x₀ = -8 ∧ f x₀ = y₀) →
  x₀ = -2 ∧ y₀ = 9 := by
sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l2918_291854


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2918_291891

def P : Set ℝ := {x | 2 ≤ x ∧ x ≤ 7}
def Q : Set ℝ := {x | x^2 - x - 6 = 0}

theorem intersection_of_P_and_Q : P ∩ Q = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2918_291891


namespace NUMINAMATH_CALUDE_find_x_l2918_291865

theorem find_x (x y z : ℝ) 
  (h1 : x * y / (x + y) = 4)
  (h2 : x * z / (x + z) = 5)
  (h3 : y * z / (y + z) = 6)
  : x = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2918_291865


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l2918_291849

theorem min_value_expression (x : ℝ) : 
  (x^2 + 11) / Real.sqrt (x^2 + 5) ≥ 2 * Real.sqrt 6 := by
  sorry

theorem lower_bound_achievable : 
  ∃ x : ℝ, (x^2 + 11) / Real.sqrt (x^2 + 5) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l2918_291849


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2918_291844

/-- A geometric sequence with common ratio 4 and sum of first three terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 4 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The general term formula for the geometric sequence -/
theorem geometric_sequence_formula (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∀ n : ℕ, a n = 4^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2918_291844


namespace NUMINAMATH_CALUDE_production_theorem_l2918_291889

/-- Represents the production process with recycling --/
def max_parts_and_waste (initial_blanks : ℕ) (efficiency : ℚ) : ℕ × ℚ :=
  sorry

/-- The theorem statement --/
theorem production_theorem :
  max_parts_and_waste 20 (2/3) = (29, 1/3) := by sorry

end NUMINAMATH_CALUDE_production_theorem_l2918_291889


namespace NUMINAMATH_CALUDE_sum_of_powers_l2918_291807

theorem sum_of_powers (x : ℝ) (h1 : x^2023 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^2022 + x^2021 + x^2020 + x^2019 + x^2018 + x^2017 + x^2016 + x^2015 + x^2014 + x^2013 +
  x^2012 + x^2011 + x^2010 + x^2009 + x^2008 + x^2007 + x^2006 + x^2005 + x^2004 + x^2003 +
  x^2002 + x^2001 + x^2000 + x^1999 + x^1998 + x^1997 + x^1996 + x^1995 + x^1994 + x^1993 +
  x^1992 + x^1991 + x^1990 + x^1989 + x^1988 + x^1987 + x^1986 + x^1985 + x^1984 + x^1983 +
  x^1982 + x^1981 + x^1980 + x^1979 + x^1978 + x^1977 + x^1976 + x^1975 + x^1974 + x^1973 +
  -- ... (omitting middle terms for brevity)
  x^50 + x^49 + x^48 + x^47 + x^46 + x^45 + x^44 + x^43 + x^42 + x^41 +
  x^40 + x^39 + x^38 + x^37 + x^36 + x^35 + x^34 + x^33 + x^32 + x^31 +
  x^30 + x^29 + x^28 + x^27 + x^26 + x^25 + x^24 + x^23 + x^22 + x^21 +
  x^20 + x^19 + x^18 + x^17 + x^16 + x^15 + x^14 + x^13 + x^12 + x^11 +
  x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l2918_291807


namespace NUMINAMATH_CALUDE_no_third_quadrant_implies_m_leq_1_l2918_291837

def linear_function (x m : ℝ) : ℝ := -2 * x + 1 - m

theorem no_third_quadrant_implies_m_leq_1 :
  ∀ m : ℝ, (∀ x y : ℝ, y = linear_function x m → (x < 0 → y ≥ 0)) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_third_quadrant_implies_m_leq_1_l2918_291837


namespace NUMINAMATH_CALUDE_fiona_cleaning_time_proof_l2918_291810

/-- Calculates Fiona's cleaning time in minutes given the total cleaning time and Lilly's fraction of work -/
def fiona_cleaning_time (total_time : ℝ) (lilly_fraction : ℝ) : ℝ :=
  (total_time - lilly_fraction * total_time) * 60

/-- Theorem: Given a total cleaning time of 8 hours and Lilly spending 1/4 of the total time, 
    Fiona's cleaning time in minutes is equal to 360. -/
theorem fiona_cleaning_time_proof :
  fiona_cleaning_time 8 (1/4) = 360 := by
  sorry

end NUMINAMATH_CALUDE_fiona_cleaning_time_proof_l2918_291810


namespace NUMINAMATH_CALUDE_special_number_is_perfect_square_l2918_291853

theorem special_number_is_perfect_square (n : ℕ) :
  ∃ k : ℕ, 4 * 10^(2*n+2) - 4 * 10^(n+1) + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_special_number_is_perfect_square_l2918_291853


namespace NUMINAMATH_CALUDE_h1n1_spread_properties_l2918_291803

/-- Represents the spread of H1N1 flu in a community -/
def H1N1Spread (x : ℝ) : Prop :=
  (1 + x)^2 = 36 ∧ x > 0

theorem h1n1_spread_properties (x : ℝ) (hx : H1N1Spread x) :
  x = 5 ∧ (1 + x)^3 > 200 := by
  sorry

#check h1n1_spread_properties

end NUMINAMATH_CALUDE_h1n1_spread_properties_l2918_291803


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2918_291847

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x + 1 ≥ 0) ↔ m ∈ Set.Icc (-2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2918_291847


namespace NUMINAMATH_CALUDE_sqrt_of_nine_l2918_291804

theorem sqrt_of_nine : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_l2918_291804


namespace NUMINAMATH_CALUDE_cookie_selection_count_jamie_cookie_selections_l2918_291846

theorem cookie_selection_count : Nat → Nat → Nat
  | n, k => Nat.choose (n + k - 1) (k - 1)

theorem jamie_cookie_selections :
  cookie_selection_count 7 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_cookie_selection_count_jamie_cookie_selections_l2918_291846


namespace NUMINAMATH_CALUDE_colored_squares_count_l2918_291836

/-- The size of the square grid -/
def gridSize : ℕ := 101

/-- The number of L-shaped layers in the grid -/
def numLayers : ℕ := gridSize / 2

/-- The number of squares colored in the nth L-shaped layer -/
def squaresInLayer (n : ℕ) : ℕ := 8 * n

/-- The total number of colored squares in the grid -/
def totalColoredSquares : ℕ := 1 + (numLayers * (numLayers + 1) * 4)

/-- Theorem stating that the total number of colored squares is 10201 -/
theorem colored_squares_count :
  totalColoredSquares = 10201 := by sorry

end NUMINAMATH_CALUDE_colored_squares_count_l2918_291836


namespace NUMINAMATH_CALUDE_complex_cube_root_magnitude_l2918_291817

theorem complex_cube_root_magnitude (w : ℂ) (h : w^3 = 64 - 48*I) : 
  Complex.abs w = 2 * Real.rpow 10 (1/3) := by
sorry

end NUMINAMATH_CALUDE_complex_cube_root_magnitude_l2918_291817


namespace NUMINAMATH_CALUDE_unique_modulus_of_quadratic_roots_l2918_291862

theorem unique_modulus_of_quadratic_roots :
  ∃! r : ℝ, ∃ z : ℂ, z^2 - 6*z + 34 = 0 ∧ Complex.abs z = r :=
by sorry

end NUMINAMATH_CALUDE_unique_modulus_of_quadratic_roots_l2918_291862


namespace NUMINAMATH_CALUDE_square_problem_l2918_291880

/-- Square with side length 800 -/
structure Square :=
  (side : ℝ)
  (is_800 : side = 800)

/-- Point on the side of the square -/
structure PointOnSide :=
  (x : ℝ)
  (in_range : 0 ≤ x ∧ x ≤ 800)

/-- Expression of the form p + q√r -/
structure SurdExpression :=
  (p q r : ℕ)
  (r_not_perfect_square : ∀ (n : ℕ), n > 1 → ¬(r.gcd (n^2) > 1))

/-- Main theorem -/
theorem square_problem (S : Square) (E F : PointOnSide) (BF : SurdExpression) :
  S.side = 800 →
  E.x < F.x →
  F.x - E.x = 300 →
  Real.cos (60 * π / 180) * (F.x - 400) = Real.sin (60 * π / 180) * 400 →
  800 - F.x = BF.p + BF.q * Real.sqrt BF.r →
  BF.p + BF.q + BF.r = 334 := by
  sorry

end NUMINAMATH_CALUDE_square_problem_l2918_291880


namespace NUMINAMATH_CALUDE_inequalities_for_positive_reals_l2918_291864

theorem inequalities_for_positive_reals (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (-a < -b) ∧ ((b/a + a/b) > 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_positive_reals_l2918_291864


namespace NUMINAMATH_CALUDE_function_equality_l2918_291845

theorem function_equality (f : ℕ+ → ℕ+) :
  (∀ x y : ℕ+, f (x + y * f x) = x * f (y + 1)) →
  (∀ x : ℕ+, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_function_equality_l2918_291845


namespace NUMINAMATH_CALUDE_election_probability_l2918_291887

/-- Represents an election between two candidates -/
structure Election where
  p : ℕ  -- votes for candidate A
  q : ℕ  -- votes for candidate B
  h : p > q  -- condition that p > q

/-- 
The probability that candidate A's vote count is always greater than 
candidate B's throughout the counting process 
-/
noncomputable def winning_probability (e : Election) : ℚ :=
  (e.p - e.q : ℚ) / (e.p + e.q : ℚ)

/-- 
Theorem: The probability that candidate A's vote count is always greater than 
candidate B's throughout the counting process is (p - q) / (p + q) 
-/
theorem election_probability (e : Election) : 
  winning_probability e = (e.p - e.q : ℚ) / (e.p + e.q : ℚ) := by
  sorry

/-- Example for p = 3 and q = 2 -/
example : ∃ (e : Election), e.p = 3 ∧ e.q = 2 ∧ winning_probability e = 1/5 := by
  sorry

/-- Example for p = 1010 and q = 1009 -/
example : ∃ (e : Election), e.p = 1010 ∧ e.q = 1009 ∧ winning_probability e = 1/2019 := by
  sorry

end NUMINAMATH_CALUDE_election_probability_l2918_291887


namespace NUMINAMATH_CALUDE_repetend_of_five_seventeenths_l2918_291856

theorem repetend_of_five_seventeenths :
  ∃ (n : ℕ), (5 : ℚ) / 17 = (n : ℚ) / 999999999999 ∧ 
  n = 294117647058 :=
sorry

end NUMINAMATH_CALUDE_repetend_of_five_seventeenths_l2918_291856


namespace NUMINAMATH_CALUDE_missing_number_in_mean_l2918_291890

theorem missing_number_in_mean (numbers : List ℕ) (missing : ℕ) : 
  numbers = [1, 22, 23, 24, 25, 26, 27] →
  numbers.length = 7 →
  (numbers.sum + missing) / 8 = 20 →
  missing = 12 := by
sorry

end NUMINAMATH_CALUDE_missing_number_in_mean_l2918_291890


namespace NUMINAMATH_CALUDE_clothing_store_problem_l2918_291806

theorem clothing_store_problem (cost_A B : ℕ) (profit_A B : ℕ) :
  3 * cost_A + 2 * cost_B = 450 →
  cost_A + cost_B = 175 →
  profit_A = 30 →
  profit_B = 20 →
  (∀ m : ℕ, m ≤ 100 → profit_A * m + profit_B * (100 - m) ≥ 2400 →
    ∃ n : ℕ, n ≥ m ∧ n ≥ 40) →
  ∃ m : ℕ, m ≥ 40 ∧ ∀ n : ℕ, n < m → profit_A * n + profit_B * (100 - n) < 2400 :=
by
  sorry

end NUMINAMATH_CALUDE_clothing_store_problem_l2918_291806


namespace NUMINAMATH_CALUDE_cost_of_six_books_cost_of_six_books_proof_l2918_291834

/-- Given that two identical books cost $36, prove that six of these books cost $108. -/
theorem cost_of_six_books : ℝ → Prop :=
  fun (cost_of_two_books : ℝ) =>
    cost_of_two_books = 36 →
    6 * (cost_of_two_books / 2) = 108

-- The proof goes here
theorem cost_of_six_books_proof : cost_of_six_books 36 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_six_books_cost_of_six_books_proof_l2918_291834


namespace NUMINAMATH_CALUDE_happy_number_512_l2918_291866

def is_happy_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 1)^2 - (2*k - 1)^2

theorem happy_number_512 :
  is_happy_number 512 ∧
  ¬is_happy_number 285 ∧
  ¬is_happy_number 330 ∧
  ¬is_happy_number 582 :=
sorry

end NUMINAMATH_CALUDE_happy_number_512_l2918_291866


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l2918_291860

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 30 ∧ initial_mean = 180 ∧ incorrect_value = 135 ∧ correct_value = 155 →
  let total_sum := n * initial_mean
  let corrected_sum := total_sum + (correct_value - incorrect_value)
  corrected_sum / n = 180.67 := by
sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l2918_291860


namespace NUMINAMATH_CALUDE_third_dog_food_consumption_l2918_291877

/-- Given information about three dogs' food consumption, prove the amount eaten by the third dog -/
theorem third_dog_food_consumption
  (total_dogs : ℕ)
  (average_consumption : ℝ)
  (first_dog_consumption : ℝ)
  (h_total_dogs : total_dogs = 3)
  (h_average : average_consumption = 15)
  (h_first_dog : first_dog_consumption = 13)
  (h_second_dog : ∃ (second_dog_consumption : ℝ), second_dog_consumption = 2 * first_dog_consumption) :
  ∃ (third_dog_consumption : ℝ),
    third_dog_consumption = total_dogs * average_consumption - (first_dog_consumption + 2 * first_dog_consumption) :=
by sorry

end NUMINAMATH_CALUDE_third_dog_food_consumption_l2918_291877


namespace NUMINAMATH_CALUDE_flour_sack_cost_l2918_291875

/-- Represents the cost and customs scenario for flour sacks --/
structure FlourScenario where
  sack_cost : ℕ  -- Cost of one sack of flour in pesetas
  customs_duty : ℕ  -- Customs duty per sack in pesetas
  truck1_sacks : ℕ := 118  -- Number of sacks in first truck
  truck2_sacks : ℕ := 40   -- Number of sacks in second truck
  truck1_left : ℕ := 10    -- Sacks left by first truck
  truck2_left : ℕ := 4     -- Sacks left by second truck
  truck1_pay : ℕ := 800    -- Additional payment by first truck
  truck2_receive : ℕ := 800  -- Amount received by second truck

/-- The theorem stating the cost of each sack of flour --/
theorem flour_sack_cost (scenario : FlourScenario) : scenario.sack_cost = 1600 :=
  by
    have h1 : scenario.sack_cost * scenario.truck1_left + scenario.truck1_pay = 
              scenario.customs_duty * (scenario.truck1_sacks - scenario.truck1_left) := by sorry
    have h2 : scenario.sack_cost * scenario.truck2_left - scenario.truck2_receive = 
              scenario.customs_duty * (scenario.truck2_sacks - scenario.truck2_left) := by sorry
    sorry  -- The proof goes here

end NUMINAMATH_CALUDE_flour_sack_cost_l2918_291875


namespace NUMINAMATH_CALUDE_prob_not_red_or_purple_is_correct_l2918_291872

-- Define the total number of balls and the number of each color
def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 10
def yellow_balls : ℕ := 7
def red_balls : ℕ := 15
def purple_balls : ℕ := 6

-- Define the probability of choosing a ball that is neither red nor purple
def prob_not_red_or_purple : ℚ := (white_balls + green_balls + yellow_balls) / total_balls

-- Theorem statement
theorem prob_not_red_or_purple_is_correct :
  prob_not_red_or_purple = 13/20 := by sorry

end NUMINAMATH_CALUDE_prob_not_red_or_purple_is_correct_l2918_291872


namespace NUMINAMATH_CALUDE_tangent_line_at_one_inequality_holds_l2918_291842

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*a * Real.log x + (a-2)*x

-- Part 1: Tangent line equation when a = 1
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 4*x + 2*y - 3 = 0 :=
sorry

-- Part 2: Inequality holds for a ≤ -1/2
theorem inequality_holds (a : ℝ) (h : a ≤ -1/2) :
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
    (f a x₂ - f a x₁) / (x₂ - x₁) > a :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_inequality_holds_l2918_291842


namespace NUMINAMATH_CALUDE_dougs_age_l2918_291811

/-- Given the ages of Qaddama, Jack, and Doug, prove Doug's age --/
theorem dougs_age (qaddama jack doug : ℕ) 
  (h1 : qaddama = jack + 6)
  (h2 : doug = jack + 3)
  (h3 : qaddama = 19) : 
  doug = 16 := by
  sorry

end NUMINAMATH_CALUDE_dougs_age_l2918_291811


namespace NUMINAMATH_CALUDE_grid_toothpicks_l2918_291882

/-- Calculates the total number of toothpicks in a rectangular grid. -/
def total_toothpicks (height width : ℕ) : ℕ :=
  let horizontal_lines := height + 1
  let vertical_lines := width + 1
  (horizontal_lines * width) + (vertical_lines * height)

/-- Theorem stating that a 30x15 rectangular grid of toothpicks uses 945 toothpicks. -/
theorem grid_toothpicks : total_toothpicks 30 15 = 945 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpicks_l2918_291882


namespace NUMINAMATH_CALUDE_no_positive_solution_l2918_291885

theorem no_positive_solution :
  ¬ ∃ (x : ℝ), x > 0 ∧ (Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = 2 * Real.log 9 / Real.log 4 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_solution_l2918_291885


namespace NUMINAMATH_CALUDE_hypotenuse_length_l2918_291876

-- Define the triangle
structure RightTriangle where
  a : ℝ  -- Length of one leg
  b : ℝ  -- Length of the other leg
  c : ℝ  -- Length of the hypotenuse
  right_angled : a^2 + b^2 = c^2  -- Pythagorean theorem
  sum_of_squares : a^2 + b^2 + c^2 = 2450
  hypotenuse_relation : c = b + 10

-- Theorem statement
theorem hypotenuse_length (t : RightTriangle) : t.c = 35 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l2918_291876


namespace NUMINAMATH_CALUDE_base_number_proof_l2918_291870

theorem base_number_proof (y : ℝ) (base : ℝ) 
  (h1 : 9^y = base^16) (h2 : y = 8) : base = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l2918_291870


namespace NUMINAMATH_CALUDE_total_distance_rowed_l2918_291851

/-- Calculates the total distance rowed by a man given specific conditions -/
theorem total_distance_rowed (still_water_speed wind_speed river_speed : ℝ)
  (total_time : ℝ) (h1 : still_water_speed = 8)
  (h2 : wind_speed = 1.5) (h3 : river_speed = 3.5) (h4 : total_time = 2) :
  let speed_to := still_water_speed - river_speed - wind_speed
  let speed_from := still_water_speed + river_speed + wind_speed
  let distance := (speed_to * speed_from * total_time) / (speed_to + speed_from)
  2 * distance = 9.75 :=
by sorry

end NUMINAMATH_CALUDE_total_distance_rowed_l2918_291851


namespace NUMINAMATH_CALUDE_function_not_linear_plus_integer_l2918_291809

theorem function_not_linear_plus_integer : 
  ∃ (f : ℚ → ℚ), 
    (∀ x y : ℚ, ∃ z : ℤ, f (x + y) - f x - f y = ↑z) ∧ 
    (¬ ∃ c : ℚ, ∀ x : ℚ, ∃ z : ℤ, f x - c * x = ↑z) := by
  sorry

end NUMINAMATH_CALUDE_function_not_linear_plus_integer_l2918_291809
