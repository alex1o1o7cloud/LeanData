import Mathlib

namespace NUMINAMATH_CALUDE_alternating_arrangement_white_first_arrangement_group_formation_l2748_274800

/- Define the total number of balls -/
def total_balls : ℕ := 12

/- Define the number of white balls -/
def white_balls : ℕ := 6

/- Define the number of black balls -/
def black_balls : ℕ := 6

/- Define the function to calculate factorial -/
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

/- Define the function to calculate combinations -/
def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

/- Theorem for part (a) -/
theorem alternating_arrangement : 
  factorial white_balls * factorial black_balls = 518400 := by sorry

/- Theorem for part (b) -/
theorem white_first_arrangement : 
  factorial white_balls * factorial black_balls = 518400 := by sorry

/- Theorem for part (c) -/
theorem group_formation : 
  choose white_balls 4 * factorial 4 * choose black_balls 3 * factorial 3 = 43200 := by sorry

end NUMINAMATH_CALUDE_alternating_arrangement_white_first_arrangement_group_formation_l2748_274800


namespace NUMINAMATH_CALUDE_system_of_equations_l2748_274876

theorem system_of_equations (x y a b : ℝ) (h1 : 4*x - 2*y = a) (h2 : 6*y - 12*x = b) (h3 : b ≠ 0) : a/b = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l2748_274876


namespace NUMINAMATH_CALUDE_product_of_special_set_l2748_274810

theorem product_of_special_set (n : ℕ) (M : Finset ℝ) : 
  Odd n → 
  n > 1 → 
  Finset.card M = n →
  (∀ x ∈ M, (M.sum id - x) + x = M.sum id) →
  M.prod id = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_special_set_l2748_274810


namespace NUMINAMATH_CALUDE_camp_athlete_difference_l2748_274833

/-- The difference in the total number of athletes in the camp over two nights -/
def athlete_difference (initial : ℕ) (leaving_rate : ℕ) (leaving_hours : ℕ) (arriving_rate : ℕ) (arriving_hours : ℕ) : ℕ :=
  initial - (initial - leaving_rate * leaving_hours + arriving_rate * arriving_hours)

/-- Theorem stating the difference in the total number of athletes in the camp over two nights -/
theorem camp_athlete_difference : athlete_difference 600 35 6 20 9 = 30 := by
  sorry

end NUMINAMATH_CALUDE_camp_athlete_difference_l2748_274833


namespace NUMINAMATH_CALUDE_nilpotent_matrix_powers_l2748_274832

theorem nilpotent_matrix_powers (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A ^ 4 = 0) : A ^ 2 = 0 ∧ A ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nilpotent_matrix_powers_l2748_274832


namespace NUMINAMATH_CALUDE_exists_non_increasing_f_l2748_274863

theorem exists_non_increasing_f :
  ∃ a : ℝ, a < 0 ∧
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧
  let f := fun x => a * x + Real.log x
  f x₁ ≥ f x₂ :=
sorry

end NUMINAMATH_CALUDE_exists_non_increasing_f_l2748_274863


namespace NUMINAMATH_CALUDE_outer_prism_width_is_ten_l2748_274843

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The inner prism dimensions satisfy the given conditions -/
def inner_prism_conditions (d : PrismDimensions) : Prop :=
  d.length * d.width * d.height = 128 ∧
  d.width = 2 * d.length ∧
  d.width = 2 * d.height

/-- The outer prism dimensions are one unit larger in each dimension -/
def outer_prism_dimensions (d : PrismDimensions) : PrismDimensions :=
  { length := d.length + 2
  , width := d.width + 2
  , height := d.height + 2 }

/-- The width of the outer prism is 10 inches -/
theorem outer_prism_width_is_ten (d : PrismDimensions) 
  (h : inner_prism_conditions d) : 
  (outer_prism_dimensions d).width = 10 := by
  sorry

end NUMINAMATH_CALUDE_outer_prism_width_is_ten_l2748_274843


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2748_274850

theorem polynomial_coefficient_sum (a b c d : ℤ) : 
  (∀ x : ℚ, (3*x + 2) * (2*x - 3) * (x - 4) = a*x^3 + b*x^2 + c*x + d) →
  a - b + c - d = 25 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2748_274850


namespace NUMINAMATH_CALUDE_one_fifth_of_seven_times_nine_l2748_274854

theorem one_fifth_of_seven_times_nine : (1 / 5 : ℚ) * (7 * 9) = 12.6 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_seven_times_nine_l2748_274854


namespace NUMINAMATH_CALUDE_total_students_l2748_274859

theorem total_students (girls : ℕ) (boys : ℕ) (total : ℕ) : 
  girls = 160 →
  5 * boys = 8 * girls →
  total = girls + boys →
  total = 416 := by
sorry

end NUMINAMATH_CALUDE_total_students_l2748_274859


namespace NUMINAMATH_CALUDE_hawks_total_points_l2748_274836

theorem hawks_total_points (touchdowns : ℕ) (points_per_touchdown : ℕ) 
  (h1 : touchdowns = 3) 
  (h2 : points_per_touchdown = 7) : 
  touchdowns * points_per_touchdown = 21 := by
  sorry

end NUMINAMATH_CALUDE_hawks_total_points_l2748_274836


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2748_274899

theorem complex_number_quadrant : ∃ (z : ℂ), z = (4 * Complex.I) / (1 + Complex.I) ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2748_274899


namespace NUMINAMATH_CALUDE_power_of_five_l2748_274802

theorem power_of_five (n : ℕ) : 5^n = 5 * 25^2 * 125^3 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_l2748_274802


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_11_l2748_274834

def is_even_digit (d : ℕ) : Prop := d % 2 = 0 ∧ d < 10

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

theorem largest_even_digit_multiple_of_11 :
  ∀ n : ℕ,
    n < 10000 →
    has_only_even_digits n →
    n % 11 = 0 →
    n ≤ 8800 :=
sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_11_l2748_274834


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2748_274869

/-- Given a function f(x) = 2a^x + 3, where a > 0 and a ≠ 1, prove that f(0) = 5 -/
theorem function_passes_through_point
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (f : ℝ → ℝ) 
  (h3 : ∀ x, f x = 2 * a^x + 3) : 
  f 0 = 5 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2748_274869


namespace NUMINAMATH_CALUDE_madeline_work_hours_l2748_274897

def rent : ℕ := 1200
def groceries : ℕ := 400
def medical : ℕ := 200
def utilities : ℕ := 60
def emergency : ℕ := 200
def hourly_wage : ℕ := 15

def total_expenses : ℕ := rent + groceries + medical + utilities + emergency

def hours_needed : ℕ := (total_expenses + hourly_wage - 1) / hourly_wage

theorem madeline_work_hours :
  hours_needed = 138 :=
sorry

end NUMINAMATH_CALUDE_madeline_work_hours_l2748_274897


namespace NUMINAMATH_CALUDE_diana_wins_probability_l2748_274820

def diana_die : ℕ := 6
def apollo_die : ℕ := 4

def favorable_outcomes : ℕ := 14
def total_outcomes : ℕ := diana_die * apollo_die

theorem diana_wins_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 12 :=
sorry

end NUMINAMATH_CALUDE_diana_wins_probability_l2748_274820


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2748_274878

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- The area of the triangle -/
  area : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- Assumption that the hypotenuse is positive -/
  hypotenuse_pos : hypotenuse > 0
  /-- Assumption that the area is positive -/
  area_pos : area > 0
  /-- Assumption that the radius is positive -/
  radius_pos : radius > 0

/-- Theorem stating that for a right-angled triangle with hypotenuse 9 and area 36,
    the radius of the inscribed circle is 3 -/
theorem inscribed_circle_radius
  (triangle : RightTriangleWithInscribedCircle)
  (h1 : triangle.hypotenuse = 9)
  (h2 : triangle.area = 36) :
  triangle.radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2748_274878


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2748_274875

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| ≥ 1} = {x : ℝ | x ≤ -2 ∨ x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2748_274875


namespace NUMINAMATH_CALUDE_correct_sums_l2748_274823

theorem correct_sums (total : ℕ) (wrong : ℕ → ℕ) (h1 : total = 48) (h2 : wrong = λ x => 2 * x) : 
  ∃ x : ℕ, x + wrong x = total ∧ x = 16 := by
sorry

end NUMINAMATH_CALUDE_correct_sums_l2748_274823


namespace NUMINAMATH_CALUDE_division_problem_l2748_274861

theorem division_problem (a : ℝ) : a / 0.3 = 0.6 → a = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2748_274861


namespace NUMINAMATH_CALUDE_amaya_viewing_time_l2748_274813

/-- Represents the total time Amaya spent watching the movie -/
def total_viewing_time (
  segment1 segment2 segment3 segment4 segment5 : ℕ
) (rewind1 rewind2 rewind3 rewind4 : ℕ) : ℕ :=
  segment1 + segment2 + segment3 + segment4 + segment5 +
  rewind1 + rewind2 + rewind3 + rewind4

/-- Theorem stating that the total viewing time is 170 minutes -/
theorem amaya_viewing_time :
  total_viewing_time 35 45 25 15 20 5 7 10 8 = 170 := by
  sorry

end NUMINAMATH_CALUDE_amaya_viewing_time_l2748_274813


namespace NUMINAMATH_CALUDE_weakly_increasing_g_implies_m_eq_4_l2748_274883

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + (4-m)*x + m

-- Define what it means for a function to be increasing on an interval
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- Define what it means for a function to be decreasing on an interval
def IsDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y ≤ b → f x ≥ f y

-- Define what it means for a function to be weakly increasing on an interval
def IsWeaklyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  IsIncreasing f a b ∧ IsDecreasing (fun x => f x / x) a b

-- State the theorem
theorem weakly_increasing_g_implies_m_eq_4 :
  ∀ m : ℝ, IsWeaklyIncreasing (g m) 0 2 → m = 4 :=
by sorry

end NUMINAMATH_CALUDE_weakly_increasing_g_implies_m_eq_4_l2748_274883


namespace NUMINAMATH_CALUDE_mean_of_five_numbers_l2748_274851

theorem mean_of_five_numbers (a b c d e : ℚ) 
  (sum_condition : a + b + c + d + e = 3/4) :
  (a + b + c + d + e) / 5 = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_five_numbers_l2748_274851


namespace NUMINAMATH_CALUDE_min_distance_a_c_l2748_274860

/-- Given vectors a and b in ℝ² satisfying the specified conditions,
    prove that the minimum distance between a and c is (√7 - √2) / 2 -/
theorem min_distance_a_c (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 2)
  (h2 : ‖b‖ = 1)
  (h3 : a • b = 1)
  : (∀ c : ℝ × ℝ, (a - 2 • c) • (b - c) = 0 → 
    ‖a - c‖ ≥ (Real.sqrt 7 - Real.sqrt 2) / 2) ∧ 
    (∃ c : ℝ × ℝ, (a - 2 • c) • (b - c) = 0 ∧ 
    ‖a - c‖ = (Real.sqrt 7 - Real.sqrt 2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_min_distance_a_c_l2748_274860


namespace NUMINAMATH_CALUDE_game_probability_l2748_274809

theorem game_probability (lose_prob : ℚ) (h1 : lose_prob = 5/8) (h2 : lose_prob + win_prob = 1) : win_prob = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_game_probability_l2748_274809


namespace NUMINAMATH_CALUDE_min_sum_squares_min_value_is_two_l2748_274870

theorem min_sum_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → a^2 + b^2 ≤ x^2 + y^2 :=
by sorry

theorem min_value_is_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∃ m : ℝ, m = 2 ∧ a^2 + b^2 ≥ m ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ x^2 + y^2 = m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_value_is_two_l2748_274870


namespace NUMINAMATH_CALUDE_triangle_side_length_l2748_274805

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if a = 2, c = 2√3, and C = π/3, then b = 4 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 →
  C = π / 3 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2748_274805


namespace NUMINAMATH_CALUDE_partnership_profit_l2748_274857

theorem partnership_profit (J M : ℕ) (P : ℚ) : 
  J = 700 →
  M = 300 →
  (P / 6 + (J * 2 * P) / (3 * (J + M))) - (P / 6 + (M * 2 * P) / (3 * (J + M))) = 800 →
  P = 3000 := by
sorry

end NUMINAMATH_CALUDE_partnership_profit_l2748_274857


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l2748_274865

theorem lcm_factor_problem (A B : ℕ+) (Y : ℕ+) : 
  Nat.gcd A B = 23 →
  A = 391 →
  Nat.lcm A B = 23 * 13 * Y →
  Y = 17 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l2748_274865


namespace NUMINAMATH_CALUDE_solve_baseball_cards_problem_l2748_274864

/-- The number of cards Brandon, Malcom, and Ella have, and the combined remaining cards after transactions -/
def baseball_cards_problem (brandon_cards : ℕ) (malcom_extra : ℕ) (ella_less : ℕ) 
  (malcom_fraction : ℚ) (ella_fraction : ℚ) : Prop :=
  let malcom_cards := brandon_cards + malcom_extra
  let ella_cards := malcom_cards - ella_less
  let malcom_remaining := malcom_cards - Int.floor (malcom_fraction * malcom_cards)
  let ella_remaining := ella_cards - Int.floor (ella_fraction * ella_cards)
  malcom_remaining + ella_remaining = 32

/-- Theorem statement for the baseball cards problem -/
theorem solve_baseball_cards_problem : 
  baseball_cards_problem 20 12 5 (2/3) (1/4) := by sorry

end NUMINAMATH_CALUDE_solve_baseball_cards_problem_l2748_274864


namespace NUMINAMATH_CALUDE_last_three_average_l2748_274806

theorem last_three_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 62 →
  (list.take 4).sum / 4 = 54 →
  (list.drop 4).sum / 3 = 72.67 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l2748_274806


namespace NUMINAMATH_CALUDE_unit_digit_of_3_to_58_l2748_274840

theorem unit_digit_of_3_to_58 : 3^58 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_3_to_58_l2748_274840


namespace NUMINAMATH_CALUDE_abc_sum_range_l2748_274886

theorem abc_sum_range (a b c : ℝ) (h : a + b + 2*c = 0) :
  ∃ y : ℝ, y ≤ 0 ∧ a*b + a*c + b*c = y ∧
  ∀ z : ℝ, z ≤ 0 → ∃ a' b' c' : ℝ, a' + b' + 2*c' = 0 ∧ a'*b' + a'*c' + b'*c' = z :=
by sorry

end NUMINAMATH_CALUDE_abc_sum_range_l2748_274886


namespace NUMINAMATH_CALUDE_ashley_exam_result_l2748_274845

/-- The percentage of marks Ashley secured in the exam -/
def ashley_percentage (marks_secured : ℕ) (maximum_marks : ℕ) : ℚ :=
  (marks_secured : ℚ) / (maximum_marks : ℚ) * 100

/-- Theorem stating that Ashley secured 83% in the exam -/
theorem ashley_exam_result : ashley_percentage 332 400 = 83 := by
  sorry

end NUMINAMATH_CALUDE_ashley_exam_result_l2748_274845


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2748_274847

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + y^2 = 4*x*y) : 
  1/x + 1/y = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2748_274847


namespace NUMINAMATH_CALUDE_min_value_of_linear_combination_l2748_274824

theorem min_value_of_linear_combination (x y : ℝ) : 
  3 * x^2 + 3 * y^2 = 20 * x + 10 * y + 10 → 
  5 * x + 6 * y ≥ 122 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_linear_combination_l2748_274824


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l2748_274884

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define a point on a line segment
def PointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

-- Define the angle between two vectors
noncomputable def angle (v w : ℝ × ℝ) : ℝ := sorry

theorem triangle_angle_theorem 
  (A B C : ℝ × ℝ) 
  (E : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_bac : angle (B - A) (C - A) = 30 * π / 180)
  (h_e_on_bc : PointOnSegment E B C)
  (h_be_ec : 3 * ‖E - B‖ = 2 * ‖C - E‖)
  (h_eab : angle (E - A) (B - A) = 45 * π / 180) :
  angle (A - C) (B - C) = 15 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l2748_274884


namespace NUMINAMATH_CALUDE_functional_equation_bijection_l2748_274885

theorem functional_equation_bijection :
  ∃ f : ℕ → ℕ, Function.Bijective f ∧
    ∀ m n : ℕ, f (3*m*n + m + n) = 4*f m*f n + f m + f n :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_bijection_l2748_274885


namespace NUMINAMATH_CALUDE_sarah_reading_capacity_l2748_274898

/-- The number of complete books Sarah can read given her reading speed and available time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) : ℕ :=
  (pages_per_hour * hours_available) / pages_per_book

/-- Theorem: Sarah can read 2 books in 8 hours -/
theorem sarah_reading_capacity :
  books_read 120 360 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sarah_reading_capacity_l2748_274898


namespace NUMINAMATH_CALUDE_evaluate_expression_l2748_274896

theorem evaluate_expression : 2 * ((3^4)^3 - (4^3)^4) = -32471550 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2748_274896


namespace NUMINAMATH_CALUDE_quadratic_sum_l2748_274889

/-- Given a quadratic equation x^2 - 10x + 15 = 0 that can be rewritten as (x + b)^2 = c,
    prove that b + c = 5 -/
theorem quadratic_sum (b c : ℝ) : 
  (∀ x, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → b + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2748_274889


namespace NUMINAMATH_CALUDE_sum_divisors_400_has_one_prime_factor_l2748_274881

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the positive divisors of 400 has exactly one distinct prime factor -/
theorem sum_divisors_400_has_one_prime_factor :
  num_distinct_prime_factors (sum_of_divisors 400) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_divisors_400_has_one_prime_factor_l2748_274881


namespace NUMINAMATH_CALUDE_expansion_coefficients_l2748_274882

theorem expansion_coefficients (m : ℝ) (n : ℕ) :
  m > 0 →
  (1 : ℝ) + n + (n * (n - 1) / 2) = 37 →
  m ^ 2 * (Nat.choose n 6) = 112 →
  n = 8 ∧ m = 2 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l2748_274882


namespace NUMINAMATH_CALUDE_two_noncongruent_triangles_l2748_274814

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  triangle_inequality : a ≤ b + c ∧ b ≤ a + c ∧ c ≤ a + b

/-- Two triangles are congruent if they have the same side lengths (up to permutation) -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.a ∧ t1.b = t2.c ∧ t1.c = t2.b) ∨
  (t1.a = t2.b ∧ t1.b = t2.a ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b) ∨
  (t1.a = t2.c ∧ t1.b = t2.b ∧ t1.c = t2.a)

/-- The set of all triangles with integer side lengths and perimeter 9 -/
def triangles_with_perimeter_9 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 9}

/-- There are exactly 2 non-congruent triangles with integer side lengths and perimeter 9 -/
theorem two_noncongruent_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ triangles_with_perimeter_9 ∧
    t2 ∈ triangles_with_perimeter_9 ∧
    ¬congruent t1 t2 ∧
    ∀ (t : IntTriangle),
      t ∈ triangles_with_perimeter_9 →
      (congruent t t1 ∨ congruent t t2) :=
sorry

end NUMINAMATH_CALUDE_two_noncongruent_triangles_l2748_274814


namespace NUMINAMATH_CALUDE_interest_frequency_proof_l2748_274844

/-- The nominal interest rate per annum -/
def nominal_rate : ℝ := 0.10

/-- The effective annual rate -/
def effective_annual_rate : ℝ := 0.1025

/-- The frequency of interest payment (number of compounding periods per year) -/
def frequency : ℕ := 2

/-- Theorem stating that the given frequency results in the correct effective annual rate -/
theorem interest_frequency_proof :
  (1 + nominal_rate / frequency) ^ frequency - 1 = effective_annual_rate :=
by sorry

end NUMINAMATH_CALUDE_interest_frequency_proof_l2748_274844


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2748_274871

theorem inequality_equivalence (x : ℝ) : 
  (5 * x - 1 < (x + 1)^2 ∧ (x + 1)^2 < 7 * x - 3) ↔ (2 < x ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2748_274871


namespace NUMINAMATH_CALUDE_shirts_to_wash_l2748_274841

theorem shirts_to_wash (total_shirts : ℕ) (rewash_shirts : ℕ) (correctly_washed : ℕ) : 
  total_shirts = 63 → rewash_shirts = 12 → correctly_washed = 29 →
  total_shirts - correctly_washed + rewash_shirts = 46 := by
  sorry

end NUMINAMATH_CALUDE_shirts_to_wash_l2748_274841


namespace NUMINAMATH_CALUDE_azalea_profit_l2748_274808

/-- Calculates the profit from a sheep farm given the number of sheep, shearing cost, wool per sheep, and price per pound of wool. -/
def sheep_farm_profit (num_sheep : ℕ) (shearing_cost : ℕ) (wool_per_sheep : ℕ) (price_per_pound : ℕ) : ℕ :=
  num_sheep * wool_per_sheep * price_per_pound - shearing_cost

/-- Proves that Azalea's profit from her sheep farm is $38,000 -/
theorem azalea_profit : sheep_farm_profit 200 2000 10 20 = 38000 := by
  sorry

end NUMINAMATH_CALUDE_azalea_profit_l2748_274808


namespace NUMINAMATH_CALUDE_computer_knowledge_competition_compositions_l2748_274856

theorem computer_knowledge_competition_compositions :
  let n : ℕ := 8  -- number of people in each group
  let k : ℕ := 4  -- number of people to be selected from each group
  Nat.choose n k * Nat.choose n k = 4900 := by
  sorry

end NUMINAMATH_CALUDE_computer_knowledge_competition_compositions_l2748_274856


namespace NUMINAMATH_CALUDE_tony_paint_area_l2748_274822

/-- The area Tony needs to paint on the wall -/
def area_to_paint (wall_height wall_length door_height door_width window_height window_width : ℝ) : ℝ :=
  wall_height * wall_length - (door_height * door_width + window_height * window_width)

/-- Theorem stating the area Tony needs to paint -/
theorem tony_paint_area :
  area_to_paint 10 15 3 5 2 3 = 129 := by
  sorry

end NUMINAMATH_CALUDE_tony_paint_area_l2748_274822


namespace NUMINAMATH_CALUDE_only_C_not_like_terms_l2748_274807

-- Define a structure for a term
structure Term where
  coefficient : ℚ
  x_exponent : ℕ
  y_exponent : ℕ
  m_exponent : ℕ
  n_exponent : ℕ
  deriving Repr

-- Define a function to check if two terms are like terms
def are_like_terms (t1 t2 : Term) : Prop :=
  t1.x_exponent = t2.x_exponent ∧
  t1.y_exponent = t2.y_exponent ∧
  t1.m_exponent = t2.m_exponent ∧
  t1.n_exponent = t2.n_exponent

-- Define the terms from the problem
def term_A1 : Term := ⟨-1, 2, 1, 0, 0⟩  -- -x²y
def term_A2 : Term := ⟨2, 2, 1, 0, 0⟩   -- 2yx²
def term_B1 : Term := ⟨2, 0, 0, 0, 0⟩   -- 2πR (treating π and R as constants)
def term_B2 : Term := ⟨1, 0, 0, 0, 0⟩   -- π²R (treating π and R as constants)
def term_C1 : Term := ⟨-1, 0, 0, 2, 1⟩  -- -m²n
def term_C2 : Term := ⟨1/2, 0, 0, 1, 2⟩ -- 1/2mn²
def term_D1 : Term := ⟨1, 0, 0, 0, 0⟩   -- 2³ (8)
def term_D2 : Term := ⟨1, 0, 0, 0, 0⟩   -- 3² (9)

-- Theorem stating that only pair C contains terms that are not like terms
theorem only_C_not_like_terms :
  are_like_terms term_A1 term_A2 ∧
  are_like_terms term_B1 term_B2 ∧
  ¬(are_like_terms term_C1 term_C2) ∧
  are_like_terms term_D1 term_D2 :=
sorry

end NUMINAMATH_CALUDE_only_C_not_like_terms_l2748_274807


namespace NUMINAMATH_CALUDE_triangle_inequality_equivalence_l2748_274849

theorem triangle_inequality_equivalence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_equivalence_l2748_274849


namespace NUMINAMATH_CALUDE_existence_of_close_pair_l2748_274812

-- Define a type for numbers between 0 and 1
def UnitInterval := {x : ℝ | 0 < x ∧ x < 1}

-- State the theorem
theorem existence_of_close_pair :
  ∀ (x y z : UnitInterval), ∃ (a b : UnitInterval), |a.val - b.val| ≤ 0.5 :=
sorry

end NUMINAMATH_CALUDE_existence_of_close_pair_l2748_274812


namespace NUMINAMATH_CALUDE_correct_calculation_l2748_274846

theorem correct_calculation : ∃! x : ℤ, (2 - 3 = x ∧ x = -1) ∧
  ¬((-3)^2 = -9) ∧
  ¬(-3^2 = -6) ∧
  ¬(-3 - (-2) = -5) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2748_274846


namespace NUMINAMATH_CALUDE_largest_angle_ABC_l2748_274879

theorem largest_angle_ABC (AC BC : ℝ) (angle_BAC : ℝ) : 
  AC = 5 * Real.sqrt 2 →
  BC = 5 →
  angle_BAC = 30 * π / 180 →
  ∃ (angle_ABC : ℝ), 
    angle_ABC ≤ 135 * π / 180 ∧
    ∀ (other_angle_ABC : ℝ), 
      (AC / Real.sin angle_BAC = BC / Real.sin other_angle_ABC) →
      other_angle_ABC ≤ angle_ABC := by
sorry

end NUMINAMATH_CALUDE_largest_angle_ABC_l2748_274879


namespace NUMINAMATH_CALUDE_composite_has_at_least_three_factors_l2748_274815

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ k ∣ n

/-- The number of factors of a natural number -/
def NumberOfFactors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem composite_has_at_least_three_factors (n : ℕ) (h : IsComposite n) :
    NumberOfFactors n ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_composite_has_at_least_three_factors_l2748_274815


namespace NUMINAMATH_CALUDE_problem_2017_l2748_274842

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n => a + d * (n - 1)

/-- The problem statement -/
theorem problem_2017 : arithmeticSequence 4 3 672 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_problem_2017_l2748_274842


namespace NUMINAMATH_CALUDE_circle_center_l2748_274837

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y = 0

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Theorem stating that (3, -4) is the center of the circle defined by CircleEquation -/
theorem circle_center : 
  ∃ (c : CircleCenter), c.x = 3 ∧ c.y = -4 ∧ 
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - c.x)^2 + (y - c.y)^2 = 25 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l2748_274837


namespace NUMINAMATH_CALUDE_trajectory_of_G_l2748_274853

noncomputable section

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x + Real.sqrt 7)^2 + y^2 = 64

-- Define the fixed point N
def point_N : ℝ × ℝ := (Real.sqrt 7, 0)

-- Define a point P on the circle M
def point_P (x y : ℝ) : Prop := circle_M x y

-- Define point Q on line NP
def point_Q (x y : ℝ) : Prop := ∃ t : ℝ, (x, y) = ((1 - t) * point_N.1 + t * x, (1 - t) * point_N.2 + t * y)

-- Define point G on line segment MP
def point_G (x y : ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (x, y) = ((1 - t) * (-Real.sqrt 7) + t * x, t * y)

-- Define the condition NP = 2NQ
def condition_NP_2NQ (x_p y_p x_q y_q : ℝ) : Prop :=
  (x_p - point_N.1, y_p - point_N.2) = (2 * (x_q - point_N.1), 2 * (y_q - point_N.2))

-- Define the condition GQ ⋅ NP = 0
def condition_GQ_perp_NP (x_g y_g x_q y_q x_p y_p : ℝ) : Prop :=
  (x_g - x_q) * (x_p - point_N.1) + (y_g - y_q) * (y_p - point_N.2) = 0

theorem trajectory_of_G (x y : ℝ) :
  (∃ x_p y_p x_q y_q, 
    point_P x_p y_p ∧
    point_Q x_q y_q ∧
    point_G x y ∧
    condition_NP_2NQ x_p y_p x_q y_q ∧
    condition_GQ_perp_NP x y x_q y_q x_p y_p) →
  x^2/16 + y^2/9 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_G_l2748_274853


namespace NUMINAMATH_CALUDE_team_formation_count_l2748_274839

def male_doctors : ℕ := 5
def female_doctors : ℕ := 4
def team_size : ℕ := 3

def team_formations : ℕ := 
  (Nat.choose male_doctors 2 * Nat.choose female_doctors 1) + 
  (Nat.choose male_doctors 1 * Nat.choose female_doctors 2)

theorem team_formation_count : team_formations = 70 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_count_l2748_274839


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_not_regular_l2748_274828

/-- A pyramid with a regular polygon base and all edges of equal length -/
structure RegularPyramid (n : ℕ) where
  /-- The number of sides of the base polygon -/
  base_sides : n > 2
  /-- The length of each edge of the pyramid -/
  edge_length : ℝ
  /-- The edge length is positive -/
  edge_positive : edge_length > 0

/-- Theorem stating that a hexagonal pyramid cannot have all edges of equal length -/
theorem hexagonal_pyramid_not_regular : ¬∃ (p : RegularPyramid 6), True :=
sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_not_regular_l2748_274828


namespace NUMINAMATH_CALUDE_series_sum_equals_four_ninths_l2748_274855

theorem series_sum_equals_four_ninths :
  (∑' n : ℕ, n / (4 : ℝ)^n) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_four_ninths_l2748_274855


namespace NUMINAMATH_CALUDE_n_has_five_digits_l2748_274831

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 15 -/
axiom n_div_15 : 15 ∣ n

/-- n^2 is a perfect fourth power -/
axiom n_sq_fourth_power : ∃ k : ℕ, n^2 = k^4

/-- n^4 is a perfect square -/
axiom n_fourth_square : ∃ m : ℕ, n^4 = m^2

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ k : ℕ, k > 0 → (15 ∣ k) → (∃ a : ℕ, k^2 = a^4) → (∃ b : ℕ, k^4 = b^2) → n ≤ k

/-- The number of digits in n -/
def digits (m : ℕ) : ℕ := sorry

/-- Theorem stating that n has 5 digits -/
theorem n_has_five_digits : digits n = 5 := sorry

end NUMINAMATH_CALUDE_n_has_five_digits_l2748_274831


namespace NUMINAMATH_CALUDE_elizabeth_pen_purchase_l2748_274892

/-- Calculates the number of pens Elizabeth can buy given her budget and pencil purchase. -/
theorem elizabeth_pen_purchase 
  (total_budget : ℚ)
  (pencil_cost : ℚ)
  (pen_cost : ℚ)
  (pencil_count : ℕ)
  (h1 : total_budget = 20)
  (h2 : pencil_cost = 8/5)  -- $1.60 expressed as a rational number
  (h3 : pen_cost = 2)
  (h4 : pencil_count = 5) :
  (total_budget - pencil_cost * ↑pencil_count) / pen_cost = 6 := by
sorry

end NUMINAMATH_CALUDE_elizabeth_pen_purchase_l2748_274892


namespace NUMINAMATH_CALUDE_ratio_to_percent_l2748_274811

theorem ratio_to_percent (a b : ℕ) (h : a = 2 ∧ b = 3) : (a : ℚ) / (a + b : ℚ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percent_l2748_274811


namespace NUMINAMATH_CALUDE_brick_surface_area_l2748_274868

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 2 cm rectangular prism is 136 cm² -/
theorem brick_surface_area :
  surface_area 10 4 2 = 136 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l2748_274868


namespace NUMINAMATH_CALUDE_first_part_speed_l2748_274866

/-- Proves that given a total trip distance of 255 miles, with the second part being 3 hours at 55 mph,
    the speed S for the first 2 hours must be 45 mph. -/
theorem first_part_speed (total_distance : ℝ) (first_duration : ℝ) (second_duration : ℝ) (second_speed : ℝ) :
  total_distance = 255 →
  first_duration = 2 →
  second_duration = 3 →
  second_speed = 55 →
  ∃ S : ℝ, S = 45 ∧ total_distance = first_duration * S + second_duration * second_speed :=
by sorry

end NUMINAMATH_CALUDE_first_part_speed_l2748_274866


namespace NUMINAMATH_CALUDE_population_increase_rate_l2748_274887

theorem population_increase_rate 
  (initial_population : ℕ) 
  (final_population : ℕ) 
  (increase_rate : ℚ) :
  initial_population = 2000 →
  final_population = 2400 →
  increase_rate = (final_population - initial_population) / initial_population * 100 →
  increase_rate = 20 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_rate_l2748_274887


namespace NUMINAMATH_CALUDE_buffet_meal_combinations_l2748_274862

theorem buffet_meal_combinations : 
  let meat_options : ℕ := 3
  let vegetable_options : ℕ := 5
  let dessert_options : ℕ := 5
  let meat_selections : ℕ := 1
  let vegetable_selections : ℕ := 3
  let dessert_selections : ℕ := 2
  (meat_options.choose meat_selections) * 
  (vegetable_options.choose vegetable_selections) * 
  (dessert_options.choose dessert_selections) = 300 := by
sorry

end NUMINAMATH_CALUDE_buffet_meal_combinations_l2748_274862


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2748_274852

theorem boys_to_girls_ratio (S : ℕ) (G : ℕ) (h1 : S > 0) (h2 : G > 0) 
  (h3 : 2 * G = 3 * (S / 5)) : 
  (S - G : ℚ) / G = 7 / 3 := by sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2748_274852


namespace NUMINAMATH_CALUDE_smallest_square_area_l2748_274825

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The smallest square that can contain two non-overlapping rectangles -/
def smallest_containing_square (r1 r2 : Rectangle) : ℕ := 
  max (r1.width + r2.width) (max r1.height r2.height)

/-- Theorem: The smallest square containing a 3×5 and a 4×6 rectangle has area 49 -/
theorem smallest_square_area : 
  let r1 : Rectangle := ⟨3, 5⟩
  let r2 : Rectangle := ⟨4, 6⟩
  (smallest_containing_square r1 r2)^2 = 49 := by
sorry

#eval (smallest_containing_square ⟨3, 5⟩ ⟨4, 6⟩)^2

end NUMINAMATH_CALUDE_smallest_square_area_l2748_274825


namespace NUMINAMATH_CALUDE_expression_equals_one_l2748_274877

theorem expression_equals_one : 
  (120^2 - 13^2) / (90^2 - 19^2) * ((90-19)*(90+19)) / ((120-13)*(120+13)) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2748_274877


namespace NUMINAMATH_CALUDE_max_quadratic_expression_l2748_274895

theorem max_quadratic_expression :
  ∃ (M : ℝ), M = 67 ∧ ∀ (p : ℝ), -3 * p^2 + 30 * p - 8 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_quadratic_expression_l2748_274895


namespace NUMINAMATH_CALUDE_cube_root_strict_mono_l2748_274818

theorem cube_root_strict_mono {a b : ℝ} (h : a < b) : ¬(a^(1/3) ≥ b^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_strict_mono_l2748_274818


namespace NUMINAMATH_CALUDE_cloth_sale_meters_l2748_274873

/-- Proves that the number of meters of cloth sold is 85 given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 8925)
    (h2 : profit_per_meter = 20)
    (h3 : cost_price_per_meter = 85) :
    (total_selling_price : ℚ) / ((cost_price_per_meter : ℚ) + (profit_per_meter : ℚ)) = 85 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_meters_l2748_274873


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2748_274830

-- Part 1
theorem simplify_expression_1 (a : ℝ) : a - 2*a + 3*a = 2*a := by
  sorry

-- Part 2
theorem simplify_expression_2 (x y : ℝ) : 3*(2*x - 7*y) - (4*x - 10*y) = 2*x - 11*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2748_274830


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2748_274848

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3/5) 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2748_274848


namespace NUMINAMATH_CALUDE_quadratic_negative_roots_l2748_274894

theorem quadratic_negative_roots (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m+2)*x + m + 5 = 0 → x < 0) ↔ m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_roots_l2748_274894


namespace NUMINAMATH_CALUDE_problem_statement_l2748_274826

theorem problem_statement (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h1 : a + 2 / b = b + 2 / c) (h2 : b + 2 / c = c + 2 / a) :
  (a + 2 / b)^2 + (b + 2 / c)^2 + (c + 2 / a)^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2748_274826


namespace NUMINAMATH_CALUDE_cube_dimensions_l2748_274821

-- Define the surface area of the cube
def surface_area : ℝ := 864

-- Theorem stating the side length and diagonal of the cube
theorem cube_dimensions (s d : ℝ) : 
  (6 * s^2 = surface_area) → 
  (s = 12) ∧ 
  (d = 12 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_cube_dimensions_l2748_274821


namespace NUMINAMATH_CALUDE_existence_of_integers_l2748_274880

theorem existence_of_integers : ∃ (a b c d : ℤ), 
  d ≥ 1 ∧ 
  b % d = c % d ∧ 
  a ∣ b ∧ a ∣ c ∧ 
  (b / a) % d ≠ (c / a) % d := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l2748_274880


namespace NUMINAMATH_CALUDE_rica_spent_fraction_l2748_274803

theorem rica_spent_fraction (total_prize : ℝ) (rica_fraction : ℝ) (rica_left : ℝ) : 
  total_prize = 1000 →
  rica_fraction = 3/8 →
  rica_left = 300 →
  (total_prize * rica_fraction - rica_left) / (total_prize * rica_fraction) = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_rica_spent_fraction_l2748_274803


namespace NUMINAMATH_CALUDE_log_ratio_equality_l2748_274827

theorem log_ratio_equality : (Real.log 2 / Real.log 3) / (Real.log 8 / Real.log 9) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_equality_l2748_274827


namespace NUMINAMATH_CALUDE_quadratic_function_value_l2748_274891

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: If f(1) = 7, f(2) = 12, and c = 3, then f(3) = 18 -/
theorem quadratic_function_value (a b : ℝ) 
  (h1 : f a b 3 1 = 7)
  (h2 : f a b 3 2 = 12) :
  f a b 3 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l2748_274891


namespace NUMINAMATH_CALUDE_no_solution_exists_l2748_274838

theorem no_solution_exists (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : 0 < a₁) (h₂ : a₁ < a₂) (h₃ : a₂ < a₃) (h₄ : a₃ < a₄) :
  ¬ ∃ (k : ℝ) (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧
    x₁ + x₂ + x₃ + x₄ = 1 ∧
    a₁ * x₁ + a₂ * x₂ + a₃ * x₃ + a₄ * x₄ = k ∧
    a₁^2 * x₁ + a₂^2 * x₂ + a₃^2 * x₃ + a₄^2 * x₄ = k^2 :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2748_274838


namespace NUMINAMATH_CALUDE_solution_of_inequality_1_solution_of_inequality_2_l2748_274819

-- Define the solution sets
def solution_set_1 : Set ℝ := {x | -5/2 < x ∧ x < 3}
def solution_set_2 : Set ℝ := {x | x < -2/3 ∨ x > 0}

-- Theorem for the first inequality
theorem solution_of_inequality_1 :
  {x : ℝ | 2*x^2 - x - 15 < 0} = solution_set_1 :=
by sorry

-- Theorem for the second inequality
theorem solution_of_inequality_2 :
  {x : ℝ | 2/x > -3} = solution_set_2 :=
by sorry

end NUMINAMATH_CALUDE_solution_of_inequality_1_solution_of_inequality_2_l2748_274819


namespace NUMINAMATH_CALUDE_two_solutions_l2748_274893

/-- The number of positive integers satisfying the equation -/
def solution_count : ℕ := 2

/-- Predicate for integers satisfying the equation -/
def satisfies_equation (n : ℕ) : Prop :=
  (n + 800) / 80 = ⌊Real.sqrt n⌋

/-- Theorem stating that exactly two positive integers satisfy the equation -/
theorem two_solutions :
  (∃ (a b : ℕ), a ≠ b ∧ satisfies_equation a ∧ satisfies_equation b) ∧
  (∀ (n : ℕ), satisfies_equation n → n = a ∨ n = b) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_l2748_274893


namespace NUMINAMATH_CALUDE_exists_even_in_sequence_l2748_274817

/-- A sequence of natural numbers where each subsequent number is obtained by adding one of its non-zero digits to the previous number. -/
def DigitAdditionSequence : Type :=
  ℕ → ℕ

/-- Property that defines the sequence: each subsequent number is obtained by adding one of its non-zero digits to the previous number. -/
def IsValidSequence (seq : DigitAdditionSequence) : Prop :=
  ∀ n : ℕ, ∃ d : ℕ, d > 0 ∧ d < 10 ∧ seq (n + 1) = seq n + d

/-- Theorem stating that there exists an even number in the sequence. -/
theorem exists_even_in_sequence (seq : DigitAdditionSequence) (h : IsValidSequence seq) :
  ∃ n : ℕ, Even (seq n) := by
  sorry

end NUMINAMATH_CALUDE_exists_even_in_sequence_l2748_274817


namespace NUMINAMATH_CALUDE_g_27_is_zero_l2748_274801

/-- A function satisfying the given property -/
def special_function (g : ℕ → ℕ) : Prop :=
  ∀ a b c : ℕ, 3 * g (a^2 + b^2 + c^2) = (g a)^2 + (g b)^2 + (g c)^2

/-- The theorem stating that g(27) = 0 for any function satisfying the special property -/
theorem g_27_is_zero (g : ℕ → ℕ) (h : special_function g) : g 27 = 0 := by
  sorry

#check g_27_is_zero

end NUMINAMATH_CALUDE_g_27_is_zero_l2748_274801


namespace NUMINAMATH_CALUDE_power_of_ten_equation_l2748_274888

theorem power_of_ten_equation (x : ℕ) : (10^x) / (10^650) = 100000 ↔ x = 655 := by
  sorry

end NUMINAMATH_CALUDE_power_of_ten_equation_l2748_274888


namespace NUMINAMATH_CALUDE_slow_train_speed_l2748_274867

-- Define the problem parameters
def total_distance : ℝ := 901
def fast_train_speed : ℝ := 58
def slow_train_departure_time : ℝ := 5.5  -- 5:30 AM in decimal hours
def fast_train_departure_time : ℝ := 9.5  -- 9:30 AM in decimal hours
def meeting_time : ℝ := 16.5  -- 4:30 PM in decimal hours

-- Define the theorem
theorem slow_train_speed :
  let slow_train_travel_time : ℝ := meeting_time - slow_train_departure_time
  let fast_train_travel_time : ℝ := meeting_time - fast_train_departure_time
  let fast_train_distance : ℝ := fast_train_speed * fast_train_travel_time
  let slow_train_distance : ℝ := total_distance - fast_train_distance
  slow_train_distance / slow_train_travel_time = 45 := by
  sorry


end NUMINAMATH_CALUDE_slow_train_speed_l2748_274867


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2748_274829

theorem unique_three_digit_number : ∃! x : ℕ, 
  100 ≤ x ∧ x < 1000 ∧ 
  (∃ k : ℤ, x - 7 = 7 * k) ∧
  (∃ l : ℤ, x - 8 = 8 * l) ∧
  (∃ m : ℤ, x - 9 = 9 * m) :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2748_274829


namespace NUMINAMATH_CALUDE_equation_solution_l2748_274835

theorem equation_solution (x : ℝ) : 
  (((1 - (Real.cos (3 * x))^15 * (Real.cos (5 * x))^2)^(1/4) = Real.sin (5 * x)) ∧ 
   (Real.sin (5 * x) ≥ 0)) ↔ 
  ((∃ n : ℤ, x = π / 10 + 2 * π * n / 5) ∨ 
   (∃ s : ℤ, x = 2 * π * s)) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2748_274835


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2748_274890

theorem smaller_number_proof (L S : ℕ) (h1 : L - S = 2395) (h2 : L = 6 * S + 15) : S = 476 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2748_274890


namespace NUMINAMATH_CALUDE_sequence_a_bounds_l2748_274872

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => sequence_a n + (1 / (n+1)^2) * (sequence_a n)^2

theorem sequence_a_bounds (n : ℕ) : 1 - 1 / (n + 3) < sequence_a (n + 1) ∧ sequence_a (n + 1) < n + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_bounds_l2748_274872


namespace NUMINAMATH_CALUDE_intersection_A_B_l2748_274804

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 2*x)}

-- Define set B
def B : Set ℝ := {x | |x| ≤ 1}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2748_274804


namespace NUMINAMATH_CALUDE_room_length_calculation_l2748_274858

/-- Proves that given a room with specified width, cost per square meter for paving,
    and total paving cost, the length of the room is as calculated. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  width = 3.75 ∧ cost_per_sqm = 600 ∧ total_cost = 12375 →
  (total_cost / cost_per_sqm) / width = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l2748_274858


namespace NUMINAMATH_CALUDE_b_101_mod_49_l2748_274816

/-- The sequence b_n defined as 5^n + 7^n -/
def b (n : ℕ) : ℕ := 5^n + 7^n

/-- Theorem stating that b_101 is congruent to 12 modulo 49 -/
theorem b_101_mod_49 : b 101 ≡ 12 [MOD 49] := by
  sorry

end NUMINAMATH_CALUDE_b_101_mod_49_l2748_274816


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l2748_274874

/-- Given a set of values, proves that the initial mean before correcting an error
    is equal to the expected value. -/
theorem initial_mean_calculation (n : ℕ) (correct_value incorrect_value : ℝ) 
  (correct_mean : ℝ) (expected_initial_mean : ℝ) :
  n = 30 →
  correct_value = 165 →
  incorrect_value = 135 →
  correct_mean = 251 →
  expected_initial_mean = 250 →
  (n * correct_mean - (correct_value - incorrect_value)) / n = expected_initial_mean :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l2748_274874
