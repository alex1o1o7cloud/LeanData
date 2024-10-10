import Mathlib

namespace vodka_alcohol_consumption_l3839_383987

/-- Calculates the amount of pure alcohol consumed by one person when splitting vodka -/
theorem vodka_alcohol_consumption 
  (total_shots : ℕ) 
  (ounces_per_shot : ℚ) 
  (alcohol_percentage : ℚ) 
  (num_people : ℕ) 
  (h1 : total_shots = 8)
  (h2 : ounces_per_shot = 3/2)
  (h3 : alcohol_percentage = 1/2)
  (h4 : num_people = 2) :
  (total_shots : ℚ) * ounces_per_shot * alcohol_percentage / num_people = 3 := by
  sorry

end vodka_alcohol_consumption_l3839_383987


namespace cube_plus_reciprocal_cube_plus_one_l3839_383992

theorem cube_plus_reciprocal_cube_plus_one (m : ℝ) (h : m + 1/m = 10) : 
  m^3 + 1/m^3 + 1 = 971 := by
  sorry

end cube_plus_reciprocal_cube_plus_one_l3839_383992


namespace sum_of_fractions_equals_five_sixths_l3839_383961

theorem sum_of_fractions_equals_five_sixths :
  let sum : ℚ := (1 / (1 * 2)) + (1 / (2 * 3)) + (1 / (3 * 4)) + (1 / (4 * 5)) + (1 / (5 * 6))
  sum = 5 / 6 := by
  sorry

end sum_of_fractions_equals_five_sixths_l3839_383961


namespace solve_for_R_l3839_383938

theorem solve_for_R (R : ℝ) : (R^3)^(1/4) = 64 * 4^(1/16) → R = 256 * 2^(1/6) := by
  sorry

end solve_for_R_l3839_383938


namespace max_t_value_l3839_383947

theorem max_t_value (t : ℝ) (h : t > 0) :
  (∀ u v : ℝ, (u + 5 - 2*v)^2 + (u - v^2)^2 ≥ t^2) →
  t ≤ 2 * Real.sqrt 2 :=
by sorry

end max_t_value_l3839_383947


namespace star_equation_solution_l3839_383908

-- Define the ☆ operation
def star (a b : ℝ) : ℝ := a + b - 1

-- Theorem statement
theorem star_equation_solution :
  ∃ x : ℝ, star 2 x = 1 ∧ x = 0 := by
  sorry

end star_equation_solution_l3839_383908


namespace inequality_equivalence_l3839_383941

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / ((x - 1)^2 + 1) < 0 ↔ x < 3 := by
  sorry

end inequality_equivalence_l3839_383941


namespace arithmetic_sequence_a7_l3839_383964

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  a 3 + a 4 = 9 →
  a 7 = 8 := by sorry

end arithmetic_sequence_a7_l3839_383964


namespace vector_magnitude_problem_l3839_383960

/-- Given two plane vectors satisfying certain conditions, prove that the magnitude of their linear combination is √13. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) : 
  ‖a‖ = Real.sqrt 2 →
  b = (1, 0) →
  a • (a - 2 • b) = 0 →
  ‖2 • a + b‖ = Real.sqrt 13 := by
  sorry

end vector_magnitude_problem_l3839_383960


namespace range_of_function_l3839_383933

theorem range_of_function (a b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → 0 ≤ a * x - b ∧ a * x - b ≤ 1) →
  ∃ y : ℝ, y ∈ Set.Icc (-4/5) (2/7) ∧
    y = (3 * a + b + 1) / (a + 2 * b - 2) :=
sorry

end range_of_function_l3839_383933


namespace problem_statement_l3839_383973

theorem problem_statement (a b : ℝ) (h : 2 * a - 3 * b = 5) :
  4 * a^2 - 9 * b^2 - 30 * b + 1 = 26 := by
  sorry

end problem_statement_l3839_383973


namespace hat_shop_pricing_l3839_383926

theorem hat_shop_pricing (original_price : ℝ) (increase_rate : ℝ) (additional_charge : ℝ) (discount_rate : ℝ) : 
  original_price = 40 ∧ 
  increase_rate = 0.3 ∧ 
  additional_charge = 5 ∧ 
  discount_rate = 0.25 → 
  (1 - discount_rate) * (original_price * (1 + increase_rate) + additional_charge) = 42.75 := by
  sorry

end hat_shop_pricing_l3839_383926


namespace radius_scientific_notation_l3839_383931

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- The given radius in centimeters -/
def radius : ℝ := 0.000012

/-- The scientific notation representation of the radius -/
def radiusScientific : ScientificNotation :=
  { coefficient := 1.2
    exponent := -5
    h1 := by sorry
    h2 := by sorry }

theorem radius_scientific_notation :
  radius = radiusScientific.coefficient * (10 : ℝ) ^ radiusScientific.exponent :=
by sorry

end radius_scientific_notation_l3839_383931


namespace amusement_park_ticket_cost_l3839_383956

/-- The cost of a single ticket to an amusement park -/
def ticket_cost : ℕ := sorry

/-- The number of people in the group -/
def num_people : ℕ := 4

/-- The cost of a set of snacks -/
def snack_cost : ℕ := 5

/-- The total cost for the group, including tickets and snacks -/
def total_cost : ℕ := 92

theorem amusement_park_ticket_cost :
  ticket_cost = 18 ∧
  total_cost = num_people * ticket_cost + num_people * snack_cost :=
sorry

end amusement_park_ticket_cost_l3839_383956


namespace fraction_comparison_l3839_383922

theorem fraction_comparison : 
  (111110 : ℚ) / 111111 < (333331 : ℚ) / 333334 ∧ (333331 : ℚ) / 333334 < (222221 : ℚ) / 222223 := by
  sorry

end fraction_comparison_l3839_383922


namespace brothers_baskets_count_l3839_383997

/-- Represents the number of strawberries in each basket picked by Kimberly's brother -/
def strawberries_per_basket : ℕ := 15

/-- Represents the number of people sharing the strawberries -/
def number_of_people : ℕ := 4

/-- Represents the number of strawberries each person gets when divided equally -/
def strawberries_per_person : ℕ := 168

/-- Represents the number of baskets Kimberly's brother picked -/
def brothers_baskets : ℕ := 3

theorem brothers_baskets_count :
  ∃ (b : ℕ),
    b = brothers_baskets ∧
    (17 * b * strawberries_per_basket - 93 = number_of_people * strawberries_per_person) :=
by sorry

end brothers_baskets_count_l3839_383997


namespace sufficient_not_necessary_condition_l3839_383913

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end sufficient_not_necessary_condition_l3839_383913


namespace f_domain_correct_l3839_383966

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * x^2) / Real.sqrt (1 - x) + Real.log (3 * x + 1)

-- Define the domain of f
def domain_f : Set ℝ := {x | -1/3 < x ∧ x < 1}

-- Theorem stating that domain_f is the correct domain for f
theorem f_domain_correct : 
  ∀ x : ℝ, x ∈ domain_f ↔ (∃ y : ℝ, f x = y) :=
sorry

end f_domain_correct_l3839_383966


namespace exists_n_plus_Sn_eq_1980_consecutive_n_plus_Sn_l3839_383955

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Theorem 1: There exists a natural number n such that n + S(n) = 1980
theorem exists_n_plus_Sn_eq_1980 : ∃ n : ℕ, n + S n = 1980 := by sorry

-- Theorem 2: For any natural number k, either k or k+1 can be expressed as n + S(n)
theorem consecutive_n_plus_Sn : ∀ k : ℕ, (∃ n : ℕ, n + S n = k) ∨ (∃ n : ℕ, n + S n = k + 1) := by sorry

end exists_n_plus_Sn_eq_1980_consecutive_n_plus_Sn_l3839_383955


namespace equation_root_l3839_383925

theorem equation_root : ∃ x : ℝ, 2 * x^2 + 3 * x - 65 = 0 :=
by
  use 5
  sorry

end equation_root_l3839_383925


namespace chemistry_marks_proof_l3839_383967

def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 72
def biology_marks : ℕ := 82
def average_marks : ℕ := 71
def total_subjects : ℕ := 5

theorem chemistry_marks_proof :
  ∃ (chemistry_marks : ℕ),
    chemistry_marks = average_marks * total_subjects - (english_marks + math_marks + physics_marks + biology_marks) ∧
    chemistry_marks = 65 := by
  sorry

end chemistry_marks_proof_l3839_383967


namespace visit_neither_country_l3839_383923

theorem visit_neither_country (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 50 →
  iceland = 25 →
  norway = 23 →
  both = 21 →
  total - (iceland + norway - both) = 23 := by
  sorry

end visit_neither_country_l3839_383923


namespace sallys_pens_l3839_383944

theorem sallys_pens (students : ℕ) (pens_per_student : ℕ) (pens_taken_home : ℕ) :
  students = 44 →
  pens_per_student = 7 →
  pens_taken_home = 17 →
  ∃ (initial_pens : ℕ),
    initial_pens = 342 ∧
    pens_taken_home = (initial_pens - students * pens_per_student) / 2 :=
by
  sorry

end sallys_pens_l3839_383944


namespace proportion_solution_l3839_383935

theorem proportion_solution (x : ℝ) : (0.75 / x = 5 / 7) → x = 1.05 := by
  sorry

end proportion_solution_l3839_383935


namespace chocolate_difference_l3839_383905

theorem chocolate_difference (friend1 friend2 friend3 : ℚ)
  (h1 : friend1 = 5/6)
  (h2 : friend2 = 2/3)
  (h3 : friend3 = 7/9) :
  max friend1 (max friend2 friend3) - min friend1 (min friend2 friend3) = 1/6 :=
by sorry

end chocolate_difference_l3839_383905


namespace divisor_probability_l3839_383989

/-- The number of positive divisors of 10^99 -/
def total_divisors : ℕ := 10000

/-- The number of positive divisors of 10^99 that are multiples of 10^88 -/
def favorable_divisors : ℕ := 144

/-- The probability of a randomly chosen positive divisor of 10^99 being a multiple of 10^88 -/
def probability : ℚ := favorable_divisors / total_divisors

theorem divisor_probability :
  probability = 9 / 625 :=
sorry

end divisor_probability_l3839_383989


namespace fake_to_total_handbags_ratio_l3839_383959

theorem fake_to_total_handbags_ratio
  (total_purses : ℕ)
  (total_handbags : ℕ)
  (authentic_items : ℕ)
  (h1 : total_purses = 26)
  (h2 : total_handbags = 24)
  (h3 : authentic_items = 31)
  (h4 : total_purses / 2 = total_purses - authentic_items + total_handbags - authentic_items) :
  (total_handbags - (authentic_items - total_purses / 2)) / total_handbags = 1 / 4 := by
sorry

end fake_to_total_handbags_ratio_l3839_383959


namespace monomial_degree_6_l3839_383924

def monomial_degree (x_exp y_exp : ℕ) : ℕ := x_exp + y_exp

theorem monomial_degree_6 (a : ℕ) : 
  monomial_degree 2 a = 6 → a = 4 := by
  sorry

end monomial_degree_6_l3839_383924


namespace abs_a_pow_b_eq_one_l3839_383915

/-- Given that (2a+b-1)^2 + |a-b+4| = 0, prove that |a^b| = 1 -/
theorem abs_a_pow_b_eq_one (a b : ℝ) 
  (h : (2*a + b - 1)^2 + |a - b + 4| = 0) : 
  |a^b| = 1 := by
  sorry

end abs_a_pow_b_eq_one_l3839_383915


namespace right_triangle_arctan_sum_l3839_383934

/-- Given a right-angled triangle ABC with ∠A = π/2, 
    prove that arctan(b/(c+a)) + arctan(c/(b+a)) = π/4 -/
theorem right_triangle_arctan_sum (a b c : ℝ) (h_right_angle : a^2 = b^2 + c^2) :
  Real.arctan (b / (c + a)) + Real.arctan (c / (b + a)) = π / 4 := by
  sorry

end right_triangle_arctan_sum_l3839_383934


namespace bucket_capacity_problem_l3839_383900

/-- Given a tank that can be filled by either 18 buckets of 60 liters each or 216 buckets of unknown capacity, 
    prove that the capacity of each bucket in the second case is 5 liters. -/
theorem bucket_capacity_problem (tank_capacity : ℝ) (bucket_count_1 bucket_count_2 : ℕ) 
  (bucket_capacity_1 : ℝ) (bucket_capacity_2 : ℝ) 
  (h1 : tank_capacity = bucket_count_1 * bucket_capacity_1)
  (h2 : tank_capacity = bucket_count_2 * bucket_capacity_2)
  (h3 : bucket_count_1 = 18)
  (h4 : bucket_capacity_1 = 60)
  (h5 : bucket_count_2 = 216) :
  bucket_capacity_2 = 5 := by
  sorry

#check bucket_capacity_problem

end bucket_capacity_problem_l3839_383900


namespace tennis_players_count_l3839_383975

/-- The number of members who play tennis in a sports club -/
def tennis_players (total_members badminton_players neither_players both_players : ℕ) : ℕ :=
  total_members - neither_players - (badminton_players - both_players)

/-- Theorem stating the number of tennis players in the sports club -/
theorem tennis_players_count :
  tennis_players 30 17 3 9 = 19 := by
  sorry

end tennis_players_count_l3839_383975


namespace min_value_theorem_l3839_383962

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) / 2 ∧
  ∀ (z : ℝ), z > 0 → z + y = 1 →
    2 / (z + 3 * y) + 1 / (z - y) ≥ min_val := by
  sorry

end min_value_theorem_l3839_383962


namespace second_number_in_sequence_l3839_383953

theorem second_number_in_sequence (x y z : ℝ) : 
  z = 4 * y →
  y = 2 * x →
  (x + y + z) / 3 = 165 →
  y = 90 := by
sorry

end second_number_in_sequence_l3839_383953


namespace expression_simplification_l3839_383998

theorem expression_simplification (x : ℝ) (h : x^2 - x - 1 = 0) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2 * x^2 - x) / (x^2 + 2 * x + 1)) = 1 := by
  sorry

end expression_simplification_l3839_383998


namespace remainder_theorem_l3839_383906

-- Define the polynomial
def p (x : ℝ) : ℝ := x^5 - x^3 + 3*x^2 + 2

-- State the theorem
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, p = (λ x => (x + 2) * q x + (-10)) :=
sorry

end remainder_theorem_l3839_383906


namespace total_sheets_used_l3839_383949

theorem total_sheets_used (total_classes : ℕ) (first_class_count : ℕ) (last_class_count : ℕ)
  (first_class_students : ℕ) (last_class_students : ℕ)
  (first_class_sheets_per_student : ℕ) (last_class_sheets_per_student : ℕ) :
  total_classes = first_class_count + last_class_count →
  first_class_count = 3 →
  last_class_count = 3 →
  first_class_students = 22 →
  last_class_students = 18 →
  first_class_sheets_per_student = 6 →
  last_class_sheets_per_student = 4 →
  (first_class_count * first_class_students * first_class_sheets_per_student) +
  (last_class_count * last_class_students * last_class_sheets_per_student) = 612 :=
by sorry

end total_sheets_used_l3839_383949


namespace pears_for_20_apples_is_13_l3839_383977

/-- The price of fruits in an arbitrary unit -/
structure FruitPrices where
  apple : ℚ
  orange : ℚ
  pear : ℚ

/-- Given the conditions of the problem, calculate the number of pears
    that can be bought for the price of 20 apples -/
def pears_for_20_apples (prices : FruitPrices) : ℕ :=
  sorry

/-- Theorem stating the result of the calculation -/
theorem pears_for_20_apples_is_13 (prices : FruitPrices) 
  (h1 : 10 * prices.apple = 5 * prices.orange)
  (h2 : 3 * prices.orange = 4 * prices.pear) :
  pears_for_20_apples prices = 13 := by
  sorry

end pears_for_20_apples_is_13_l3839_383977


namespace tank_volume_ratio_l3839_383971

theorem tank_volume_ratio : 
  ∀ (tank1_volume tank2_volume : ℝ), 
  tank1_volume > 0 → tank2_volume > 0 →
  (3/4 : ℝ) * tank1_volume = (5/8 : ℝ) * tank2_volume →
  tank1_volume / tank2_volume = 6/5 := by
  sorry

end tank_volume_ratio_l3839_383971


namespace unique_f_zero_unique_solution_l3839_383952

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (x * y) = x^2 + y^2

/-- The theorem stating that f(0) = 1 is the only valid solution -/
theorem unique_f_zero (f : ℝ → ℝ) (h : FunctionalEq f) : f 0 = 1 := by
  sorry

/-- The theorem stating that f(x) = x² + 1 is the unique solution -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEq f) : 
  ∀ x : ℝ, f x = x^2 + 1 := by
  sorry

end unique_f_zero_unique_solution_l3839_383952


namespace henry_skittles_l3839_383972

theorem henry_skittles (bridget_initial : ℕ) (bridget_final : ℕ) (henry : ℕ) : 
  bridget_initial = 4 → 
  bridget_final = 8 → 
  bridget_final = bridget_initial + henry → 
  henry = 4 := by
sorry

end henry_skittles_l3839_383972


namespace all_diagonal_triangles_multiplicative_l3839_383910

/-- A regular polygon with n sides, all of length 1 -/
structure RegularPolygon (n : ℕ) where
  (n_ge_3 : n ≥ 3)
  (side_length : ℝ := 1)

/-- A triangle formed by diagonals in a regular polygon -/
structure DiagonalTriangle (n : ℕ) (p : RegularPolygon n) where
  (vertex1 : ℝ × ℝ)
  (vertex2 : ℝ × ℝ)
  (vertex3 : ℝ × ℝ)

/-- A triangle is multiplicative if the product of the lengths of two sides equals the length of the third side -/
def is_multiplicative (t : DiagonalTriangle n p) : Prop :=
  ∀ (i j k : Fin 3), i ≠ j → j ≠ k → i ≠ k →
    let sides := [dist t.vertex1 t.vertex2, dist t.vertex2 t.vertex3, dist t.vertex3 t.vertex1]
    sides[i] * sides[j] = sides[k]

/-- The main theorem: all triangles formed by diagonals in a regular polygon are multiplicative -/
theorem all_diagonal_triangles_multiplicative (n : ℕ) (p : RegularPolygon n) :
  ∀ t : DiagonalTriangle n p, is_multiplicative t :=
sorry

end all_diagonal_triangles_multiplicative_l3839_383910


namespace pascals_triangle_51st_row_third_number_l3839_383981

theorem pascals_triangle_51st_row_third_number : 
  (Nat.choose 51 2) = 1275 := by sorry

end pascals_triangle_51st_row_third_number_l3839_383981


namespace paint_color_combinations_l3839_383914

theorem paint_color_combinations (n : ℕ) (h : n = 9) : 
  (n - 1 : ℕ) = 8 := by sorry

end paint_color_combinations_l3839_383914


namespace odd_function_value_l3839_383919

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = -x^3 + (a-2)x^2 + x -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  -x^3 + (a-2)*x^2 + x

theorem odd_function_value (a : ℝ) :
  IsOdd (f a) → f a a = -6 :=
by
  sorry

end odd_function_value_l3839_383919


namespace monomial_properties_l3839_383907

/-- Represents a monomial of the form ax²y -/
structure Monomial where
  a : ℝ
  x : ℝ
  y : ℝ

/-- Checks if two monomials are of the same type -/
def same_type (m1 m2 : Monomial) : Prop :=
  (m1.x ^ 2 * m1.y = m2.x ^ 2 * m2.y)

/-- Returns the coefficient of a monomial -/
def coefficient (m : Monomial) : ℝ := m.a

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ := 3

theorem monomial_properties (m : Monomial) (h : m.a ≠ 0) :
  same_type m { a := -2, x := m.x, y := m.y } ∧
  coefficient m = m.a ∧
  degree m = 3 := by
  sorry

end monomial_properties_l3839_383907


namespace geometric_sequence_sum_l3839_383999

theorem geometric_sequence_sum (a : ℕ → ℚ) :
  a 0 = 4096 ∧ a 1 = 1024 ∧ a 2 = 256 ∧
  a 5 = 4 ∧ a 6 = 1 ∧ a 7 = (1/4 : ℚ) ∧
  (∀ n : ℕ, a (n + 1) = a n * (1/4 : ℚ)) →
  a 3 + a 4 = 80 := by
sorry

end geometric_sequence_sum_l3839_383999


namespace y_intercept_of_line_l3839_383957

/-- The y-intercept of the line 6x - 4y = 24 is (0, -6) -/
theorem y_intercept_of_line (x y : ℝ) : 
  (6 * x - 4 * y = 24) → (x = 0 → y = -6) := by
  sorry

end y_intercept_of_line_l3839_383957


namespace max_distance_ratio_l3839_383995

/-- The maximum ratio of distances from a point on a circle to two fixed points -/
theorem max_distance_ratio : 
  let A : ℝ × ℝ := (0, -2)
  let B : ℝ × ℝ := (1, -1)
  let circle := {P : ℝ × ℝ | P.1^2 + P.2^2 = 2}
  ∃ (max : ℝ), max = (3 * Real.sqrt 2) / 2 ∧ 
    ∀ P ∈ circle, 
      Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) / Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) ≤ max :=
by sorry


end max_distance_ratio_l3839_383995


namespace imaginary_part_of_complex_division_l3839_383951

theorem imaginary_part_of_complex_division :
  let i : ℂ := Complex.I
  (3 + 2*i) / i = Complex.mk 2 (-3) :=
by sorry

end imaginary_part_of_complex_division_l3839_383951


namespace sandy_shopping_percentage_l3839_383988

/-- The percentage of money Sandy spent on shopping -/
def shopping_percentage (initial_amount spent_amount : ℚ) : ℚ :=
  (spent_amount / initial_amount) * 100

/-- Proof that Sandy spent 30% of her money on shopping -/
theorem sandy_shopping_percentage :
  let initial_amount : ℚ := 200
  let remaining_amount : ℚ := 140
  let spent_amount : ℚ := initial_amount - remaining_amount
  shopping_percentage initial_amount spent_amount = 30 := by
sorry

end sandy_shopping_percentage_l3839_383988


namespace cyclic_quadrilaterals_area_l3839_383983

/-- Represents a convex cyclic quadrilateral --/
structure CyclicQuadrilateral where
  diag_angle : Real
  inscribed_radius : Real
  area : Real

/-- The theorem to be proved --/
theorem cyclic_quadrilaterals_area 
  (A B C : CyclicQuadrilateral)
  (h_radius : A.inscribed_radius = 1 ∧ B.inscribed_radius = 1 ∧ C.inscribed_radius = 1)
  (h_sin_A : Real.sin A.diag_angle = 2/3)
  (h_sin_B : Real.sin B.diag_angle = 3/5)
  (h_sin_C : Real.sin C.diag_angle = 6/7)
  (h_equal_area : A.area = B.area ∧ B.area = C.area)
  : A.area = 16/35 := by
  sorry

end cyclic_quadrilaterals_area_l3839_383983


namespace sum_of_squares_and_minimum_l3839_383901

/-- Given an equation and Vieta's formulas, prove the sum of squares and its minimum value -/
theorem sum_of_squares_and_minimum (m : ℝ) (x₁ x₂ : ℝ) 
  (eq : x₁^2 + x₂^2 = (x₁ + x₂)^2 - 2*x₁*x₂)
  (vieta1 : x₁ + x₂ = -(m + 1))
  (vieta2 : x₁ * x₂ = 2*m - 2)
  (D_nonneg : (m + 3)^2 ≥ 0) :
  (x₁^2 + x₂^2 = (m - 1)^2 + 4) ∧ 
  (∀ m', (m' - 1)^2 + 4 ≥ 4) ∧
  (∃ m₀, (m₀ - 1)^2 + 4 = 4) := by
  sorry

end sum_of_squares_and_minimum_l3839_383901


namespace max_product_sum_2000_l3839_383969

theorem max_product_sum_2000 : 
  (∃ (a b : ℤ), a + b = 2000 ∧ a * b = 1000000) ∧
  (∀ (x y : ℤ), x + y = 2000 → x * y ≤ 1000000) := by
  sorry

end max_product_sum_2000_l3839_383969


namespace wendy_full_face_time_l3839_383994

/-- Calculates the total time for Wendy's "full face" routine -/
def fullFaceTime (numProducts : ℕ) (waitTime : ℕ) (makeupTime : ℕ) : ℕ :=
  numProducts * waitTime + makeupTime

/-- Theorem: Wendy's "full face" routine takes 55 minutes -/
theorem wendy_full_face_time :
  fullFaceTime 5 5 30 = 55 := by
  sorry

end wendy_full_face_time_l3839_383994


namespace acute_angles_trig_identities_l3839_383985

theorem acute_angles_trig_identities (α β : Real) 
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_tan_α : Real.tan α = 4/3)
  (h_cos_sum : Real.cos (α + β) = -Real.sqrt 5 / 5) :
  Real.cos (2*α) = -7/25 ∧ Real.tan (α - β) = -2/11 := by
  sorry

end acute_angles_trig_identities_l3839_383985


namespace sum_of_first_100_digits_l3839_383978

/-- The decimal expansion of 1/10101 -/
def decimal_expansion : ℕ → ℕ
| n => sorry

/-- The sum of the first n digits in the decimal expansion of 1/10101 -/
def digit_sum (n : ℕ) : ℕ :=
  (List.range n).map decimal_expansion |>.sum

/-- Theorem: The sum of the first 100 digits after the decimal point in 1/10101 is 450 -/
theorem sum_of_first_100_digits : digit_sum 100 = 450 := by
  sorry

end sum_of_first_100_digits_l3839_383978


namespace banana_apple_worth_l3839_383986

theorem banana_apple_worth (banana_worth : ℚ) :
  (3 / 4 * 12 : ℚ) * banana_worth = 6 →
  (1 / 4 * 8 : ℚ) * banana_worth = 4 / 3 := by
  sorry

end banana_apple_worth_l3839_383986


namespace circle_area_ratio_l3839_383982

theorem circle_area_ratio (r₁ r₂ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 3) 
  (h₃ : (r₁ + r₂)^2 + 40^2 = 41^2) (h₄ : 20 * (r₁ + r₂) = 300) :
  (π * r₁^2) / (π * r₂^2) = 16 := by
sorry

end circle_area_ratio_l3839_383982


namespace fish_ratio_l3839_383940

/-- Proves that the ratio of blue fish to the total number of fish is 1:2 -/
theorem fish_ratio (blue orange green : ℕ) : 
  blue + orange + green = 80 →  -- Total number of fish
  orange = blue - 15 →          -- 15 fewer orange than blue
  green = 15 →                  -- Number of green fish
  blue * 2 = 80                 -- Ratio of blue to total is 1:2
    := by sorry

end fish_ratio_l3839_383940


namespace union_equals_M_l3839_383984

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = 2^x}

-- Define the set S
def S : Set ℝ := {x | ∃ y, y = x - 1}

-- State the theorem
theorem union_equals_M : M ∪ S = M :=
sorry

end union_equals_M_l3839_383984


namespace correct_recommendation_plans_l3839_383948

def total_students : ℕ := 7
def students_to_recommend : ℕ := 4
def sports_talented : ℕ := 2
def artistic_talented : ℕ := 2
def other_talented : ℕ := 3

def recommendation_plans : ℕ := sorry

theorem correct_recommendation_plans : recommendation_plans = 25 := by sorry

end correct_recommendation_plans_l3839_383948


namespace work_time_ratio_l3839_383917

/-- The time it takes for Dev and Tina to complete the task together -/
def T : ℝ := 10

/-- The time it takes for Dev to complete the task alone -/
def dev_time : ℝ := T + 20

/-- The time it takes for Tina to complete the task alone -/
def tina_time : ℝ := T + 5

/-- The time it takes for Alex to complete the task alone -/
def alex_time : ℝ := T + 10

/-- The ratio of time taken by Dev, Tina, and Alex working alone -/
def time_ratio : Prop :=
  ∃ (k : ℝ), k > 0 ∧ dev_time = 6 * k ∧ tina_time = 3 * k ∧ alex_time = 4 * k

theorem work_time_ratio : time_ratio := by
  sorry

end work_time_ratio_l3839_383917


namespace system_solution_l3839_383970

theorem system_solution (x y z : ℝ) : 
  (x^2 + y^2 - z*(x + y) = 2 ∧
   y^2 + z^2 - x*(y + z) = 4 ∧
   z^2 + x^2 - y*(z + x) = 8) ↔
  ((x = 1 ∧ y = -1 ∧ z = 2) ∨ (x = -1 ∧ y = 1 ∧ z = -2)) :=
by sorry

end system_solution_l3839_383970


namespace function_bounds_l3839_383954

/-- Given a function f(x) = 1 - a cos x - b sin x - A cos 2x - B sin 2x,
    where a, b, A, B are real constants, and f(x) ≥ 0 for all real x,
    prove that a² + b² ≤ 2 and A² + B² ≤ 1. -/
theorem function_bounds (a b A B : ℝ) 
    (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end function_bounds_l3839_383954


namespace triangle_properties_l3839_383932

open Real

structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)
  (h5 : a > 0 ∧ b > 0 ∧ c > 0)
  (h6 : a * cos C + Real.sqrt 3 * a * sin C - b - c = 0)
  (h7 : b^2 + c^2 = 2 * a^2)

def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem triangle_properties (t : Triangle) :
  t.A = π / 3 ∧
  isEquilateral t ∧
  ∃ (D : ℝ × ℝ), 
    let B := (0, 0)
    let C := (t.c, 0)
    let A := (t.b * cos t.C, t.b * sin t.C)
    let AC := (A.1 - C.1, A.2 - C.2)
    let AD := (D.1 - A.1, D.2 - A.2)
    2 * (D.1 - B.1, D.2 - B.2) = (C.1 - D.1, C.2 - D.2) ∧
    (AD.1 * AC.1 + AD.2 * AC.2) / Real.sqrt (AC.1^2 + AC.2^2) = 2/3 * Real.sqrt (AC.1^2 + AC.2^2) :=
by sorry

end triangle_properties_l3839_383932


namespace inscribed_squares_ratio_l3839_383909

theorem inscribed_squares_ratio (r : ℝ) (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (2 * a)^2 + (2 * b)^2 = r^2 ∧ 
  (a + 2*b)^2 + b^2 = r^2 → 
  a / b = 5 := by
sorry

end inscribed_squares_ratio_l3839_383909


namespace solve_inequality_range_of_a_l3839_383974

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x - 1|

-- Part I
theorem solve_inequality (x : ℝ) :
  f 2 x < 4 ↔ -1/2 < x ∧ x < 7/2 := by sorry

-- Part II
theorem range_of_a (a : ℝ) :
  (∀ x, f a x ≥ 2) ↔ a ≤ -1 ∨ a ≥ 3 := by sorry

end solve_inequality_range_of_a_l3839_383974


namespace spade_equation_solution_l3839_383903

def spade (A B : ℝ) : ℝ := A^2 + 2*A*B + 3*B + 7

theorem spade_equation_solution :
  ∃ A : ℝ, spade A 5 = 97 ∧ (A = 5 ∨ A = -15) :=
by
  sorry

end spade_equation_solution_l3839_383903


namespace work_completion_time_l3839_383976

theorem work_completion_time (p d : ℕ) (work_left : ℚ) : 
  p = 15 → 
  work_left = 0.5333333333333333 →
  (4 : ℚ) * ((1 : ℚ) / p + (1 : ℚ) / d) = 1 - work_left →
  d = 20 := by sorry

end work_completion_time_l3839_383976


namespace y_intercept_of_line_l3839_383937

/-- The y-intercept of the line 2x - 3y = 6 is -2 -/
theorem y_intercept_of_line (x y : ℝ) : 2 * x - 3 * y = 6 → x = 0 → y = -2 := by
  sorry

end y_intercept_of_line_l3839_383937


namespace line_equation_through_two_points_l3839_383902

/-- The general equation of a line passing through two given points. -/
theorem line_equation_through_two_points :
  ∀ (x y : ℝ), 
  (∃ (t : ℝ), x = -2 * (1 - t) + 0 * t ∧ y = 0 * (1 - t) + 1 * t) ↔ 
  x - 2*y + 2 = 0 :=
by sorry

end line_equation_through_two_points_l3839_383902


namespace no_triangle_with_cube_sum_equal_product_l3839_383958

theorem no_triangle_with_cube_sum_equal_product : ¬∃ (x y z : ℝ), 
  (0 < x ∧ 0 < y ∧ 0 < z) ∧  -- positive real numbers
  (x + y > z ∧ y + z > x ∧ z + x > y) ∧  -- triangle inequality
  (x^3 + y^3 + z^3 = (x+y)*(y+z)*(z+x)) := by
  sorry


end no_triangle_with_cube_sum_equal_product_l3839_383958


namespace quadratic_factorization_l3839_383920

theorem quadratic_factorization :
  ∀ x : ℝ, 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
sorry

end quadratic_factorization_l3839_383920


namespace equation_solution_l3839_383991

theorem equation_solution : ∃ x : ℝ, 20 - 3 * (x + 4) = 2 * (x - 1) ∧ x = 2 := by
  sorry

end equation_solution_l3839_383991


namespace odd_even_sum_difference_l3839_383963

def sum_odd_integers (n : ℕ) : ℕ :=
  let count := (n + 1) / 2
  count * (1 + n) / 2

def sum_even_integers (n : ℕ) : ℕ :=
  let count := n / 2
  count * (2 + n) / 2

theorem odd_even_sum_difference : sum_odd_integers 215 - sum_even_integers 100 = 9114 := by
  sorry

end odd_even_sum_difference_l3839_383963


namespace smallest_X_value_l3839_383945

/-- A function that checks if a natural number consists only of 0s and 1s in its decimal representation -/
def onlyZerosAndOnes (n : ℕ) : Prop := sorry

/-- The smallest positive integer T consisting only of 0s and 1s that is divisible by 15 -/
def T : ℕ := 1110

/-- X is defined as T divided by 15 -/
def X : ℕ := T / 15

theorem smallest_X_value :
  (onlyZerosAndOnes T) ∧ 
  (T % 15 = 0) ∧
  (∀ n : ℕ, n < T → ¬(onlyZerosAndOnes n ∧ n % 15 = 0)) →
  X = 74 := by sorry

end smallest_X_value_l3839_383945


namespace three_digit_sum_property_divisibility_condition_l3839_383946

def three_digit_num (x y z : ℕ) : ℕ := 100 * x + 10 * y + z

theorem three_digit_sum_property (a b c : ℕ) 
  (ha : a ≥ 1 ∧ a ≤ 9) (hb : b ≥ 1 ∧ b ≤ 9) (hc : c ≥ 1 ∧ c ≤ 9) :
  three_digit_num a b c + three_digit_num b c a + three_digit_num c a b = 111 * (a + b + c) :=
sorry

theorem divisibility_condition (a b c : ℕ) 
  (ha : a ≥ 1 ∧ a ≤ 9) (hb : b ≥ 1 ∧ b ≤ 9) (hc : c ≥ 1 ∧ c ≤ 9) :
  (∃ k : ℕ, three_digit_num a b c + three_digit_num b c a + three_digit_num c a b = 7 * k) →
  (a + b + c = 7 ∨ a + b + c = 14 ∨ a + b + c = 21) :=
sorry

end three_digit_sum_property_divisibility_condition_l3839_383946


namespace sixth_segment_length_l3839_383918

def segment_lengths (a : Fin 7 → ℕ) : Prop :=
  a 0 = 1 ∧ a 6 = 21 ∧ 
  (∀ i j, i < j → a i < a j) ∧
  (∀ i j k, i < j ∧ j < k → a i + a j ≤ a k)

theorem sixth_segment_length (a : Fin 7 → ℕ) (h : segment_lengths a) : a 5 = 13 := by
  sorry

end sixth_segment_length_l3839_383918


namespace weight_of_replaced_person_l3839_383929

theorem weight_of_replaced_person 
  (n : ℕ) 
  (average_increase : ℝ) 
  (new_person_weight : ℝ) : 
  n = 10 → 
  average_increase = 3.2 → 
  new_person_weight = 97 → 
  (n : ℝ) * average_increase = new_person_weight - 65 := by
  sorry

end weight_of_replaced_person_l3839_383929


namespace closest_integer_to_cube_root_250_l3839_383936

theorem closest_integer_to_cube_root_250 : 
  ∀ n : ℤ, |n^3 - 250| ≥ |6^3 - 250| := by sorry

end closest_integer_to_cube_root_250_l3839_383936


namespace storage_tubs_price_l3839_383979

/-- Calculates the total price Alison paid for storage tubs after discount -/
def total_price_after_discount (
  large_count : ℕ
  ) (medium_count : ℕ
  ) (small_count : ℕ
  ) (large_price : ℚ
  ) (medium_price : ℚ
  ) (small_price : ℚ
  ) (small_discount : ℚ
  ) : ℚ :=
  let large_medium_total := large_count * large_price + medium_count * medium_price
  let small_total := small_count * small_price * (1 - small_discount)
  large_medium_total + small_total

/-- Theorem stating the total price Alison paid for storage tubs after discount -/
theorem storage_tubs_price :
  total_price_after_discount 4 6 8 8 6 4 (1/10) = 968/10 :=
by
  sorry


end storage_tubs_price_l3839_383979


namespace max_sum_of_squares_l3839_383911

theorem max_sum_of_squares (a b c d : ℕ+) (h : a^2 + b^2 + c^2 + d^2 = 70) :
  a + b + c + d ≤ 16 ∧ ∃ (a' b' c' d' : ℕ+), a'^2 + b'^2 + c'^2 + d'^2 = 70 ∧ a' + b' + c' + d' = 16 := by
  sorry

end max_sum_of_squares_l3839_383911


namespace leap_year_53_sundays_probability_l3839_383930

/-- A leap year has 366 days -/
def leapYearDays : ℕ := 366

/-- A week has 7 days -/
def daysInWeek : ℕ := 7

/-- A leap year has 52 complete weeks and 2 extra days -/
def leapYearWeeks : ℕ := 52
def leapYearExtraDays : ℕ := 2

/-- The probability of a randomly selected leap year having 53 Sundays -/
def probLeapYear53Sundays : ℚ := 2 / 7

theorem leap_year_53_sundays_probability :
  probLeapYear53Sundays = 2 / 7 := by sorry

end leap_year_53_sundays_probability_l3839_383930


namespace geometric_sequence_minimum_value_l3839_383950

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value (a : ℕ → ℝ) (m n : ℕ) :
  GeometricSequence a →
  a 2016 = a 2015 + 2 * a 2014 →
  a m * a n = 16 * (a 1)^2 →
  (4 : ℝ) / m + 1 / n ≥ 3 / 2 :=
by sorry

end geometric_sequence_minimum_value_l3839_383950


namespace remainder_theorem_l3839_383928

-- Define the polynomial q(x)
def q (x D : ℝ) : ℝ := 2 * x^4 - 3 * x^2 + D * x + 6

-- Theorem statement
theorem remainder_theorem (D : ℝ) :
  (q 2 D = 6) → (q (-2) D = 52) := by
  sorry

end remainder_theorem_l3839_383928


namespace vector_parallel_implies_x_y_values_l3839_383912

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ (i : Fin 3), v i = k * w i

/-- Vector a defined as (1, 2, -y) -/
def a (y : ℝ) : Fin 3 → ℝ
  | 0 => 1
  | 1 => 2
  | 2 => -y
  | _ => 0

/-- Vector b defined as (x, 1, 2) -/
def b (x : ℝ) : Fin 3 → ℝ
  | 0 => x
  | 1 => 1
  | 2 => 2
  | _ => 0

/-- The sum of two vectors -/
def vec_add (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ i => v i + w i

/-- The scalar multiplication of a vector -/
def vec_smul (k : ℝ) (v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ i => k * v i

theorem vector_parallel_implies_x_y_values (x y : ℝ) :
  parallel (vec_add (a y) (vec_smul 2 (b x))) (vec_add (vec_smul 2 (a y)) (vec_smul (-1) (b x))) →
  x = 1/2 ∧ y = -4 := by
  sorry

end vector_parallel_implies_x_y_values_l3839_383912


namespace smallest_survey_size_l3839_383943

theorem smallest_survey_size (n : ℕ) : n > 0 ∧ 
  (∃ k₁ : ℕ, n * (140 : ℚ) / 360 = k₁) ∧
  (∃ k₂ : ℕ, n * (108 : ℚ) / 360 = k₂) ∧
  (∃ k₃ : ℕ, n * (72 : ℚ) / 360 = k₃) ∧
  (∃ k₄ : ℕ, n * (40 : ℚ) / 360 = k₄) →
  n ≥ 90 :=
by sorry

end smallest_survey_size_l3839_383943


namespace product_correction_l3839_383942

theorem product_correction (a b : ℕ) : 
  a > 9 ∧ a < 100 ∧ (a - 3) * b = 224 → a * b = 245 := by
  sorry

end product_correction_l3839_383942


namespace a3_range_l3839_383965

/-- A sequence {aₙ} is convex if (aₙ + aₙ₊₂)/2 ≤ aₙ₊₁ for all positive integers n. -/
def is_convex_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a n + a (n + 2)) / 2 ≤ a (n + 1)

/-- The function bₙ = n² - 6n + 10 -/
def b (n : ℕ) : ℝ := (n : ℝ)^2 - 6*(n : ℝ) + 10

theorem a3_range (a : ℕ → ℝ) 
  (h_convex : is_convex_sequence a)
  (h_a1 : a 1 = 1)
  (h_a10 : a 10 = 28)
  (h_bound : ∀ n : ℕ, 1 ≤ n → n < 10 → |a n - b n| ≤ 20) :
  7 ≤ a 3 ∧ a 3 ≤ 19 := by sorry

end a3_range_l3839_383965


namespace largest_angle_cosine_l3839_383927

theorem largest_angle_cosine (t : ℝ) (h : t > 1) :
  let a := t^2 + t + 1
  let b := t^2 - 1
  let c := 2*t + 1
  (a < b + c) ∧ (b < a + c) ∧ (c < a + b) →
  (a^2 + b^2 - c^2) / (2*a*b) = -1/2 :=
sorry

end largest_angle_cosine_l3839_383927


namespace proposition_p_and_not_q_l3839_383968

theorem proposition_p_and_not_q : 
  (∃ x : ℝ, x^2 - x + 1 ≥ 0) ∧ 
  ¬(∀ a b : ℝ, a^2 < b^2 → a < b) :=
by sorry

end proposition_p_and_not_q_l3839_383968


namespace part_one_part_two_l3839_383996

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + a + 3
def g (b : ℝ) (x : ℝ) : ℝ := b*x + 5 - 2*b

-- Part 1
theorem part_one (a : ℝ) :
  (∃ x ∈ Set.Icc (-1) 1, f a x = 0) → -8 ≤ a ∧ a ≤ 0 :=
sorry

-- Part 2
theorem part_two (b : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 4, ∃ x₂ ∈ Set.Icc 1 4, g b x₁ = f 3 x₂) →
  b ∈ Set.Icc (-1) (1/2) :=
sorry

end part_one_part_two_l3839_383996


namespace continuous_compound_interest_interest_rate_problem_l3839_383990

/-- The annual interest rate for a continuously compounded investment --/
noncomputable def annual_interest_rate (initial_investment : ℝ) (final_amount : ℝ) (years : ℝ) : ℝ :=
  (Real.log (final_amount / initial_investment)) / years

/-- Theorem stating the relationship between the initial investment, final amount, time, and interest rate --/
theorem continuous_compound_interest
  (initial_investment : ℝ)
  (final_amount : ℝ)
  (years : ℝ)
  (h1 : initial_investment > 0)
  (h2 : final_amount > initial_investment)
  (h3 : years > 0) :
  final_amount = initial_investment * Real.exp (years * annual_interest_rate initial_investment final_amount years) :=
by sorry

/-- The specific problem instance --/
theorem interest_rate_problem :
  let initial_investment : ℝ := 5000
  let final_amount : ℝ := 8500
  let years : ℝ := 10
  8500 = 5000 * Real.exp (10 * annual_interest_rate 5000 8500 10) :=
by sorry

end continuous_compound_interest_interest_rate_problem_l3839_383990


namespace total_dividend_income_l3839_383921

-- Define the investments for each stock
def investment_A : ℕ := 2000
def investment_B : ℕ := 2500
def investment_C : ℕ := 1500
def investment_D : ℕ := 2000
def investment_E : ℕ := 2000

-- Define the dividend yields for each stock for each year
def yield_A : Fin 3 → ℚ
  | 0 => 5/100
  | 1 => 4/100
  | 2 => 3/100

def yield_B : Fin 3 → ℚ
  | 0 => 3/100
  | 1 => 5/100
  | 2 => 4/100

def yield_C : Fin 3 → ℚ
  | 0 => 4/100
  | 1 => 6/100
  | 2 => 4/100

def yield_D : Fin 3 → ℚ
  | 0 => 6/100
  | 1 => 3/100
  | 2 => 5/100

def yield_E : Fin 3 → ℚ
  | 0 => 2/100
  | 1 => 7/100
  | 2 => 6/100

-- Calculate the total dividend income for a single stock over 3 years
def total_dividend (investment : ℕ) (yield : Fin 3 → ℚ) : ℚ :=
  (yield 0 * investment) + (yield 1 * investment) + (yield 2 * investment)

-- Theorem: The total dividend income from all stocks over 3 years is 1330
theorem total_dividend_income :
  total_dividend investment_A yield_A +
  total_dividend investment_B yield_B +
  total_dividend investment_C yield_C +
  total_dividend investment_D yield_D +
  total_dividend investment_E yield_E = 1330 := by
  sorry


end total_dividend_income_l3839_383921


namespace cryptarithm_solution_l3839_383939

theorem cryptarithm_solution : 
  ∀ y : ℕ, 
    100000 ≤ y ∧ y < 1000000 →
    y * 3 = (y % 100000) * 10 + y / 100000 →
    y = 142857 ∨ y = 285714 :=
by
  sorry

#check cryptarithm_solution

end cryptarithm_solution_l3839_383939


namespace no_mutually_exclusive_sets_l3839_383993

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Yellow

/-- Represents the outcome of drawing two balls -/
def TwoBallDraw := (BallColor × BallColor)

/-- The set of all possible outcomes when drawing two balls from a bag with two white and two yellow balls -/
def SampleSpace : Set TwoBallDraw := sorry

/-- Event: At least one white ball -/
def AtLeastOneWhite (draw : TwoBallDraw) : Prop := 
  draw.1 = BallColor.White ∨ draw.2 = BallColor.White

/-- Event: At least one yellow ball -/
def AtLeastOneYellow (draw : TwoBallDraw) : Prop := 
  draw.1 = BallColor.Yellow ∨ draw.2 = BallColor.Yellow

/-- Event: Both balls are yellow -/
def BothYellow (draw : TwoBallDraw) : Prop := 
  draw.1 = BallColor.Yellow ∧ draw.2 = BallColor.Yellow

/-- Event: Exactly one white ball and one yellow ball -/
def OneWhiteOneYellow (draw : TwoBallDraw) : Prop := 
  (draw.1 = BallColor.White ∧ draw.2 = BallColor.Yellow) ∨
  (draw.1 = BallColor.Yellow ∧ draw.2 = BallColor.White)

/-- The three sets of events -/
def EventSet1 := {draw : TwoBallDraw | AtLeastOneWhite draw ∧ AtLeastOneYellow draw}
def EventSet2 := {draw : TwoBallDraw | AtLeastOneYellow draw ∧ BothYellow draw}
def EventSet3 := {draw : TwoBallDraw | OneWhiteOneYellow draw}

/-- Two events are mutually exclusive if their intersection is empty -/
def MutuallyExclusive (A B : Set TwoBallDraw) : Prop := A ∩ B = ∅

theorem no_mutually_exclusive_sets : 
  ¬(MutuallyExclusive EventSet1 EventSet2) ∧ 
  ¬(MutuallyExclusive EventSet1 EventSet3) ∧ 
  ¬(MutuallyExclusive EventSet2 EventSet3) := by sorry

end no_mutually_exclusive_sets_l3839_383993


namespace range_of_a_l3839_383904

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > a then x + 2 * x else x^2 + 5 * x + 2

/-- Function g(x) defined as f(x) - 2x -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 2 * x

/-- Theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    g a x = 0 ∧ g a y = 0 ∧ g a z = 0 ∧
    (∀ w : ℝ, g a w = 0 → w = x ∨ w = y ∨ w = z)) →
  a ∈ Set.Icc (-1) 2 ∧ a ≠ 2 :=
by sorry

end range_of_a_l3839_383904


namespace distance_ratio_theorem_l3839_383916

/-- Given two points (4,3) and (2,-3) on a coordinate plane, this theorem proves:
    1. The direct distance between them is 2√10
    2. The horizontal distance between them is 2
    3. The ratio of the horizontal distance to the direct distance is not an integer -/
theorem distance_ratio_theorem :
  let p1 : ℝ × ℝ := (4, 3)
  let p2 : ℝ × ℝ := (2, -3)
  let direct_distance := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let horizontal_distance := |p1.1 - p2.1|
  let ratio := horizontal_distance / direct_distance
  (direct_distance = 2 * Real.sqrt 10) ∧
  (horizontal_distance = 2) ∧
  ¬(∃ n : ℤ, ratio = n) :=
by sorry

end distance_ratio_theorem_l3839_383916


namespace like_terms_exponent_sum_l3839_383980

/-- Given that the monomials 2a^(4)b^(-2m+7) and 3a^(2m)b^(n+2) are like terms, prove that m + n = 3 -/
theorem like_terms_exponent_sum (a b : ℝ) (m n : ℤ) 
  (h : ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → 2 * a^4 * b^(-2*m+7) = 3 * a^(2*m) * b^(n+2)) : 
  m + n = 3 := by
  sorry

end like_terms_exponent_sum_l3839_383980
