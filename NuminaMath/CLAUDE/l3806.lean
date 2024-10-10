import Mathlib

namespace x_plus_y_equals_plus_minus_three_l3806_380668

theorem x_plus_y_equals_plus_minus_three (x y : ℝ) 
  (h1 : |x| = 1) 
  (h2 : |y| = 2) 
  (h3 : x * y > 0) : 
  x + y = 3 ∨ x + y = -3 := by
  sorry

end x_plus_y_equals_plus_minus_three_l3806_380668


namespace simple_interest_rate_calculation_l3806_380601

theorem simple_interest_rate_calculation (P A T : ℝ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 10) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 2 := by sorry

end simple_interest_rate_calculation_l3806_380601


namespace bobs_favorite_number_l3806_380632

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem bobs_favorite_number :
  ∃! n : ℕ,
    50 < n ∧ n < 100 ∧
    n % 11 = 0 ∧
    n % 2 ≠ 0 ∧
    sum_of_digits n % 3 = 0 ∧
    n = 99 := by
  sorry

end bobs_favorite_number_l3806_380632


namespace x_expression_l3806_380666

theorem x_expression (m n x : ℝ) (h1 : m ≠ n) (h2 : m ≠ 0) (h3 : n ≠ 0) 
  (h4 : (x + 2*m)^2 - 2*(x + n)^2 = 2*(m - n)^2) : 
  x = 2*m - 2*n := by
sorry

end x_expression_l3806_380666


namespace problem_1_l3806_380649

theorem problem_1 : 96 * 15 / (45 * 16) = 2 := by sorry

end problem_1_l3806_380649


namespace polynomial_equality_l3806_380697

/-- Given a polynomial M such that M + (5x^2 - 4x - 3) = -x^2 - 3x,
    prove that M = -6x^2 + x + 3 -/
theorem polynomial_equality (x : ℝ) (M : ℝ → ℝ) : 
  (M x + (5*x^2 - 4*x - 3) = -x^2 - 3*x) → 
  (M x = -6*x^2 + x + 3) := by
sorry

end polynomial_equality_l3806_380697


namespace share_ratio_l3806_380675

def total_amount : ℕ := 544

def shares (A B C : ℕ) : Prop :=
  A + B + C = total_amount ∧ 4 * B = C

theorem share_ratio (A B C : ℕ) (h : shares A B C) (hA : A = 64) (hB : B = 96) (hC : C = 384) :
  A * 3 = B * 2 := by sorry

end share_ratio_l3806_380675


namespace dart_probability_l3806_380673

/-- The probability of a dart landing in the center square of a regular hexagonal dartboard -/
theorem dart_probability (a : ℝ) (h : a > 0) : 
  let hexagon_side := a
  let square_side := a * Real.sqrt 3 / 2
  let hexagon_area := 3 * Real.sqrt 3 / 2 * a^2
  let square_area := (a * Real.sqrt 3 / 2)^2
  square_area / hexagon_area = 1 / (2 * Real.sqrt 3) :=
by sorry

end dart_probability_l3806_380673


namespace infinite_series_sum_l3806_380652

/-- The sum of the infinite series ∑(k=1 to ∞) k³/3ᵏ is equal to 12 -/
theorem infinite_series_sum : ∑' k, (k : ℝ)^3 / 3^k = 12 := by sorry

end infinite_series_sum_l3806_380652


namespace circle_area_and_circumference_l3806_380669

theorem circle_area_and_circumference (r : ℝ) (h : r > 0) :
  ∃ (A C : ℝ),
    A = π * r^2 ∧
    C = 2 * π * r :=
by sorry

end circle_area_and_circumference_l3806_380669


namespace equidistant_point_x_coordinate_l3806_380671

/-- A point (x, y) in the coordinate plane that is equally distant from the x-axis, y-axis, 
    line x + 2y = 4, and line y = 2x has x-coordinate equal to -4 / (√5 - 7) -/
theorem equidistant_point_x_coordinate (x y : ℝ) : 
  (abs x = abs y) ∧ 
  (abs x = abs (x + 2*y - 4) / Real.sqrt 5) ∧
  (abs x = abs (y - 2*x) / Real.sqrt 5) →
  x = -4 / (Real.sqrt 5 - 7) := by
sorry

end equidistant_point_x_coordinate_l3806_380671


namespace least_multiple_24_greater_500_l3806_380657

theorem least_multiple_24_greater_500 : ∃ n : ℕ, 
  (24 * n > 500) ∧ (∀ m : ℕ, 24 * m > 500 → 24 * n ≤ 24 * m) ∧ (24 * n = 504) := by
  sorry

end least_multiple_24_greater_500_l3806_380657


namespace granola_bar_distribution_l3806_380614

theorem granola_bar_distribution (total : ℕ) (eaten_by_parents : ℕ) (num_children : ℕ) :
  total = 200 →
  eaten_by_parents = 80 →
  num_children = 6 →
  (total - eaten_by_parents) / num_children = 20 :=
by
  sorry

end granola_bar_distribution_l3806_380614


namespace geometric_sequence_sum_l3806_380690

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum (n : ℕ) :
  geometric_sum 1 (1/3) n = 121/81 → n = 5 := by
  sorry

end geometric_sequence_sum_l3806_380690


namespace youngest_brother_age_l3806_380606

theorem youngest_brother_age (a b c : ℕ) : 
  b = a + 1 → c = b + 1 → a + b + c = 96 → a = 31 := by
sorry

end youngest_brother_age_l3806_380606


namespace max_distinct_roots_special_polynomial_l3806_380611

/-- A polynomial with the property that the product of any two distinct roots is also a root -/
def SpecialPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ y → P x = 0 → P y = 0 → P (x * y) = 0

/-- The maximum number of distinct real roots for a special polynomial is 4 -/
theorem max_distinct_roots_special_polynomial :
  ∃ (P : ℝ → ℝ), SpecialPolynomial P ∧
    (∃ (roots : Finset ℝ), (∀ x ∈ roots, P x = 0) ∧ roots.card = 4) ∧
    (∀ (Q : ℝ → ℝ), SpecialPolynomial Q →
      ∀ (roots : Finset ℝ), (∀ x ∈ roots, Q x = 0) → roots.card ≤ 4) :=
sorry

end max_distinct_roots_special_polynomial_l3806_380611


namespace triangle_side_length_l3806_380695

/-- Given a triangle ABC with the following properties:
  * A = 60°
  * a = 6√3
  * b = 12
  * S_ABC = 18√3
  Prove that c = 6 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  A = π / 3 →  -- 60° in radians
  a = 6 * Real.sqrt 3 →
  b = 12 →
  S = 18 * Real.sqrt 3 →
  c = 6 := by
  sorry


end triangle_side_length_l3806_380695


namespace equation_equality_l3806_380672

theorem equation_equality (x y z : ℝ) (h1 : x ≠ y) 
  (h2 : (x^2 - y*z) / (x*(1 - y*z)) = (y^2 - x*z) / (y*(1 - x*z))) : 
  x + y + z = 1/x + 1/y + 1/z := by
sorry

end equation_equality_l3806_380672


namespace fraction_not_simplifiable_l3806_380643

theorem fraction_not_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end fraction_not_simplifiable_l3806_380643


namespace no_modular_inverse_of_3_mod_33_l3806_380637

theorem no_modular_inverse_of_3_mod_33 : ¬ ∃ x : ℕ, x ≤ 32 ∧ (3 * x) % 33 = 1 := by
  sorry

end no_modular_inverse_of_3_mod_33_l3806_380637


namespace solution_set_implies_a_l3806_380692

theorem solution_set_implies_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 6 ≤ 0 ↔ 2 ≤ x ∧ x ≤ 3) → a = -5 :=
by sorry

end solution_set_implies_a_l3806_380692


namespace unique_solution_system_l3806_380602

theorem unique_solution_system (x y : ℝ) :
  (x - 2*y = 1 ∧ 2*x - y = 11) ↔ (x = 7 ∧ y = 3) := by sorry

end unique_solution_system_l3806_380602


namespace exist_distinct_indices_with_difference_not_t_l3806_380629

theorem exist_distinct_indices_with_difference_not_t 
  (n : ℕ+) (t : ℝ) (ht : t ≠ 0) (a : Fin (2*n - 1) → ℝ) :
  ∃ (s : Finset (Fin (2*n - 1))), 
    s.card = n ∧ 
    ∀ (i j : Fin n), i ≠ j → 
      ∃ (x y : Fin (2*n - 1)), x ∈ s ∧ y ∈ s ∧ a x - a y ≠ t :=
by sorry

end exist_distinct_indices_with_difference_not_t_l3806_380629


namespace billy_feeds_twice_daily_l3806_380674

/-- The number of times Billy feeds his horses per day -/
def feedings_per_day (num_horses : ℕ) (oats_per_meal : ℕ) (total_oats : ℕ) (days : ℕ) : ℕ :=
  (total_oats / days) / (num_horses * oats_per_meal)

/-- Theorem: Billy feeds his horses twice a day -/
theorem billy_feeds_twice_daily :
  feedings_per_day 4 4 96 3 = 2 := by
  sorry

end billy_feeds_twice_daily_l3806_380674


namespace triangle_properties_l3806_380608

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  b = a * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin A →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  ((1/4) * a^2 + (1/4) * b^2 - (1/4) * c^2) * 2 / a = 2 →
  -- Conclusions
  A = π/3 ∧ b = Real.sqrt 2 ∧ c = 2 * Real.sqrt 2 := by
sorry

end triangle_properties_l3806_380608


namespace backpacking_cooks_l3806_380685

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of people in the group --/
def total_people : ℕ := 10

/-- The number of people willing to cook --/
def eligible_people : ℕ := total_people - 1

/-- The number of cooks needed --/
def cooks_needed : ℕ := 2

theorem backpacking_cooks : choose eligible_people cooks_needed = 36 := by
  sorry

end backpacking_cooks_l3806_380685


namespace barbara_shopping_expense_l3806_380679

theorem barbara_shopping_expense (tuna_packs : ℕ) (tuna_price : ℚ) 
  (water_bottles : ℕ) (water_price : ℚ) (total_spent : ℚ) :
  tuna_packs = 5 →
  tuna_price = 2 →
  water_bottles = 4 →
  water_price = (3/2) →
  total_spent = 56 →
  total_spent - (tuna_packs * tuna_price + water_bottles * water_price) = 40 := by
sorry

end barbara_shopping_expense_l3806_380679


namespace initial_mixture_volume_l3806_380633

/-- Given a mixture of milk and water with an initial ratio of 3:2, 
    prove that if 66 liters of water are added to change the ratio to 3:4, 
    the initial volume of the mixture was 165 liters. -/
theorem initial_mixture_volume 
  (initial_milk : ℝ) 
  (initial_water : ℝ) 
  (h1 : initial_milk / initial_water = 3 / 2) 
  (h2 : initial_milk / (initial_water + 66) = 3 / 4) : 
  initial_milk + initial_water = 165 := by
  sorry

#check initial_mixture_volume

end initial_mixture_volume_l3806_380633


namespace common_difference_is_half_l3806_380667

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_4_6 : a 4 + a 6 = 6
  sum_5 : (a 1 + a 2 + a 3 + a 4 + a 5 : ℚ) = 10

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℚ :=
  seq.a 2 - seq.a 1

/-- Theorem stating that the common difference is 1/2 -/
theorem common_difference_is_half (seq : ArithmeticSequence) :
  common_difference seq = 1/2 := by
  sorry

end common_difference_is_half_l3806_380667


namespace distance_origin_to_line_l3806_380623

/-- The distance from the origin to the line 4x + 3y - 12 = 0 is 12/5 -/
theorem distance_origin_to_line : 
  let line := {(x, y) : ℝ × ℝ | 4 * x + 3 * y - 12 = 0}
  ∃ d : ℝ, d = 12/5 ∧ ∀ (p : ℝ × ℝ), p ∈ line → Real.sqrt ((p.1)^2 + (p.2)^2) ≥ d := by
  sorry

end distance_origin_to_line_l3806_380623


namespace x_value_l3806_380613

theorem x_value (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 12) : x = 9 := by
  sorry

end x_value_l3806_380613


namespace max_sum_of_squares_of_roots_l3806_380660

theorem max_sum_of_squares_of_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x : ℝ, x^2 - (k-2)*x + (k^2+3*k+5) = 0 ↔ x = x₁ ∨ x = x₂) →
  (∃ k : ℝ, x₁^2 + x₂^2 = 18) ∧
  (∀ k : ℝ, x₁^2 + x₂^2 ≤ 18) :=
by sorry

end max_sum_of_squares_of_roots_l3806_380660


namespace nested_expression_evaluation_l3806_380647

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) = 485 := by
  sorry

end nested_expression_evaluation_l3806_380647


namespace vector_magnitude_l3806_380658

/-- Given two vectors a and b in ℝ², prove that the magnitude of b is 5 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  a = (3, -2) → a + b = (0, 2) → ‖b‖ = 5 := by sorry

end vector_magnitude_l3806_380658


namespace pet_store_dogs_l3806_380689

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
def calculate_dogs (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) : ℕ :=
  (num_cats / cat_ratio) * dog_ratio

/-- Theorem: Given the ratio of cats to dogs is 2:3 and there are 14 cats, there are 21 dogs -/
theorem pet_store_dogs : calculate_dogs 2 3 14 = 21 := by
  sorry

end pet_store_dogs_l3806_380689


namespace three_circles_arrangement_l3806_380642

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The intersection points of two circles --/
def intersectionPoints (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

/-- Three circles have only two common points --/
def haveOnlyTwoCommonPoints (c1 c2 c3 : Circle) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    intersectionPoints c1 c2 ∩ intersectionPoints c2 c3 ∩ intersectionPoints c1 c3 = {p, q}

/-- All three circles intersect at the same two points --/
def allIntersectAtSamePoints (c1 c2 c3 : Circle) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    intersectionPoints c1 c2 = {p, q} ∧
    intersectionPoints c2 c3 = {p, q} ∧
    intersectionPoints c1 c3 = {p, q}

/-- One circle intersects each of the other two circles at two distinct points --/
def oneIntersectsOthersAtDistinctPoints (c1 c2 c3 : Circle) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
    ((intersectionPoints c1 c2 = {p, q} ∧ intersectionPoints c1 c3 = {p, q}) ∨
     (intersectionPoints c2 c1 = {p, q} ∧ intersectionPoints c2 c3 = {p, q}) ∨
     (intersectionPoints c3 c1 = {p, q} ∧ intersectionPoints c3 c2 = {p, q}))

/-- The main theorem --/
theorem three_circles_arrangement (c1 c2 c3 : Circle) :
  haveOnlyTwoCommonPoints c1 c2 c3 →
  allIntersectAtSamePoints c1 c2 c3 ∨ oneIntersectsOthersAtDistinctPoints c1 c2 c3 := by
  sorry


end three_circles_arrangement_l3806_380642


namespace complex_number_modulus_l3806_380684

theorem complex_number_modulus (z : ℂ) : (1 + z) / (1 - z) = Complex.I → Complex.abs z = 1 := by
  sorry

end complex_number_modulus_l3806_380684


namespace exist_decreasing_lcm_sequence_l3806_380696

theorem exist_decreasing_lcm_sequence :
  ∃ (a : Fin 100 → ℕ),
    (∀ i j : Fin 100, i < j → a i < a j) ∧
    (∀ i : Fin 99, Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))) :=
by sorry

end exist_decreasing_lcm_sequence_l3806_380696


namespace factorization_equality_l3806_380626

theorem factorization_equality (a b : ℝ) : 3 * a^2 + 6 * a * b + 3 * b^2 = 3 * (a + b)^2 := by
  sorry

end factorization_equality_l3806_380626


namespace productivity_wage_relation_l3806_380627

/-- Represents the initial workday length in hours -/
def initial_workday : ℝ := 8

/-- Represents the reduced workday length in hours -/
def reduced_workday : ℝ := 7

/-- Represents the wage increase percentage -/
def wage_increase : ℝ := 5

/-- Represents the required productivity increase percentage -/
def productivity_increase : ℝ := 20

/-- Proves that a 20% productivity increase results in a 5% wage increase
    when the workday is reduced from 8 to 7 hours -/
theorem productivity_wage_relation :
  (reduced_workday / initial_workday) * (1 + productivity_increase / 100) = 1 + wage_increase / 100 :=
by sorry

end productivity_wage_relation_l3806_380627


namespace geometric_series_sum_l3806_380620

theorem geometric_series_sum (a : ℝ) (h : |a| < 1) :
  (∑' n, a^n) = 1 / (1 - a) :=
sorry

end geometric_series_sum_l3806_380620


namespace finite_values_l3806_380617

def recurrence (A B : ℕ) (x : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → x (n + 1) = A * Nat.gcd (x n) (x (n - 1)) + B

theorem finite_values (A B : ℕ) (x : ℕ → ℕ) (h : recurrence A B x) :
  ∃ (S : Finset ℕ), ∀ n : ℕ, x n ∈ S :=
sorry

end finite_values_l3806_380617


namespace max_area_is_100_l3806_380646

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Checks if the rectangle satisfies the given conditions -/
def isValidRectangle (r : Rectangle) : Prop :=
  r.length + r.width = 20 ∧ Even r.width

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ :=
  r.length * r.width

/-- Theorem: The maximum area of a valid rectangle is 100 -/
theorem max_area_is_100 :
  ∃ (r : Rectangle), isValidRectangle r ∧
    area r = 100 ∧
    ∀ (s : Rectangle), isValidRectangle s → area s ≤ 100 :=
by sorry

end max_area_is_100_l3806_380646


namespace rectangle_to_hexagon_side_length_l3806_380693

theorem rectangle_to_hexagon_side_length :
  ∀ (rectangle_length rectangle_width : ℝ) (hexagon_side : ℝ),
    rectangle_length = 24 →
    rectangle_width = 8 →
    (3 * Real.sqrt 3 / 2) * hexagon_side^2 = rectangle_length * rectangle_width →
    hexagon_side = 8 * Real.sqrt 3 / 3 := by
  sorry

end rectangle_to_hexagon_side_length_l3806_380693


namespace problem_statement_l3806_380616

theorem problem_statement : (-24 : ℚ) * (5/6 - 4/3 + 5/8) = -3 := by sorry

end problem_statement_l3806_380616


namespace solution_set_f_less_than_2_range_of_a_for_solution_exists_l3806_380612

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 1|

-- Theorem for part I
theorem solution_set_f_less_than_2 :
  {x : ℝ | f x < 2} = Set.Ioo (-4 : ℝ) (2/3) :=
sorry

-- Theorem for part II
theorem range_of_a_for_solution_exists :
  {a : ℝ | ∃ x, f x ≤ a - a^2/2} = Set.Icc (-1 : ℝ) 3 :=
sorry

end solution_set_f_less_than_2_range_of_a_for_solution_exists_l3806_380612


namespace quadratic_real_roots_l3806_380677

/-- A quadratic equation ax^2 - 4x - 2 = 0 has real roots if and only if a ≥ -2 and a ≠ 0 -/
theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 4*x - 2 = 0) ↔ (a ≥ -2 ∧ a ≠ 0) :=
by sorry

end quadratic_real_roots_l3806_380677


namespace seedling_problem_l3806_380670

theorem seedling_problem (x : ℕ) : 
  (x^2 + 39 = (x + 1)^2 - 50) → (x^2 + 39 = 1975) :=
by
  sorry

#check seedling_problem

end seedling_problem_l3806_380670


namespace discriminant_irrational_l3806_380661

/-- A quadratic polynomial without roots -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℚ
  c : ℝ
  no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0

/-- The function f(x) for a QuadraticPolynomial -/
def f (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The discriminant of a QuadraticPolynomial -/
def discriminant (p : QuadraticPolynomial) : ℝ :=
  p.b^2 - 4 * p.a * p.c

/-- Exactly one of c or f(c) is irrational -/
axiom one_irrational (p : QuadraticPolynomial) :
  (¬ Irrational p.c ∧ Irrational (f p p.c)) ∨
  (Irrational p.c ∧ ¬ Irrational (f p p.c))

theorem discriminant_irrational (p : QuadraticPolynomial) :
  Irrational (discriminant p) :=
sorry

end discriminant_irrational_l3806_380661


namespace integer_solutions_of_inequalities_l3806_380628

theorem integer_solutions_of_inequalities :
  let S : Set ℤ := {x | (2 + x : ℝ) > (7 - 4*x) ∧ (x : ℝ) < ((4 + x) / 2)}
  S = {2, 3} := by sorry

end integer_solutions_of_inequalities_l3806_380628


namespace art_gallery_theorem_l3806_380699

/-- A polygon represented by its vertices -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  size : Nat
  h_size : vertices.length = size
  h_size_ge_3 : size ≥ 3

/-- A guard position -/
def Guard := ℝ × ℝ

/-- A point is visible from a guard if the line segment between them doesn't intersect any edge of the polygon -/
def isVisible (p : Polygon) (point guard : ℝ × ℝ) : Prop := sorry

/-- A set of guards covers a polygon if every point in the polygon is visible from at least one guard -/
def covers (p : Polygon) (guards : List Guard) : Prop :=
  ∀ point, ∃ guard ∈ guards, isVisible p point guard

/-- The main theorem: ⌊n/3⌋ guards are sufficient to cover any polygon with n sides -/
theorem art_gallery_theorem (p : Polygon) :
  ∃ guards : List Guard, guards.length ≤ p.size / 3 ∧ covers p guards := by
  sorry

end art_gallery_theorem_l3806_380699


namespace smallest_number_in_sample_l3806_380663

/-- Systematic sampling function that returns the smallest number in the sample -/
def systematicSample (totalProducts : ℕ) (sampleSize : ℕ) (containsProduct : ℕ) : ℕ :=
  containsProduct % (totalProducts / sampleSize)

/-- Theorem: The smallest number in the systematic sample is 10 -/
theorem smallest_number_in_sample :
  systematicSample 80 5 42 = 10 := by
  sorry

#eval systematicSample 80 5 42

end smallest_number_in_sample_l3806_380663


namespace cos_minus_sin_value_l3806_380634

theorem cos_minus_sin_value (α : Real) (h1 : π/4 < α) (h2 : α < π/2) (h3 : Real.sin (2*α) = 24/25) :
  Real.cos α - Real.sin α = -1/5 := by
sorry

end cos_minus_sin_value_l3806_380634


namespace line_inclination_theorem_l3806_380638

theorem line_inclination_theorem (a b c : ℝ) (α : ℝ) : 
  (∃ x y, a * x + b * y + c = 0) →  -- Line exists
  (Real.tan α = -a / b) →           -- Relationship between inclination angle and coefficients
  (Real.sin α + Real.cos α = 0) →   -- Given condition
  a - b = 0 := by
sorry

end line_inclination_theorem_l3806_380638


namespace conditional_probability_rhinitis_cold_l3806_380665

theorem conditional_probability_rhinitis_cold 
  (P_rhinitis : ℝ) 
  (P_rhinitis_and_cold : ℝ) 
  (h1 : P_rhinitis = 0.8) 
  (h2 : P_rhinitis_and_cold = 0.6) : 
  P_rhinitis_and_cold / P_rhinitis = 0.75 := by
  sorry

end conditional_probability_rhinitis_cold_l3806_380665


namespace pie_remainder_l3806_380630

theorem pie_remainder (carlos_share maria_share remainder : ℝ) : 
  carlos_share = 65 ∧ 
  maria_share = (100 - carlos_share) / 2 ∧ 
  remainder = 100 - carlos_share - maria_share →
  remainder = 17.5 := by
  sorry

end pie_remainder_l3806_380630


namespace perfect_square_condition_l3806_380639

theorem perfect_square_condition (n : ℤ) :
  (∃ k : ℤ, 7 * n + 2 = k ^ 2) ↔ 
  (∃ m : ℤ, (n = 7 * m ^ 2 + 6 * m + 1) ∨ (n = 7 * m ^ 2 - 6 * m + 1)) :=
by sorry

end perfect_square_condition_l3806_380639


namespace total_hamburger_combinations_l3806_380682

/-- The number of different condiments available -/
def num_condiments : ℕ := 10

/-- The number of choices for meat patties -/
def meat_patty_choices : ℕ := 3

/-- The number of choices for buns -/
def bun_choices : ℕ := 2

/-- Theorem: The total number of different hamburger combinations -/
theorem total_hamburger_combinations : 
  (2 ^ num_condiments) * meat_patty_choices * bun_choices = 6144 := by
  sorry

end total_hamburger_combinations_l3806_380682


namespace product_of_powers_of_ten_l3806_380619

theorem product_of_powers_of_ten : (10^0.6) * (10^0.4) * (10^0.3) * (10^0.2) * (10^0.5) = 100 := by
  sorry

end product_of_powers_of_ten_l3806_380619


namespace age_ratio_l3806_380650

def sachin_age : ℕ := 63
def age_difference : ℕ := 18

def rahul_age : ℕ := sachin_age + age_difference

theorem age_ratio : 
  (sachin_age : ℚ) / (rahul_age : ℚ) = 7 / 9 := by sorry

end age_ratio_l3806_380650


namespace most_balls_l3806_380621

theorem most_balls (soccerballs basketballs : ℕ) 
  (h1 : soccerballs = 50)
  (h2 : basketballs = 26)
  (h3 : ∃ baseballs : ℕ, baseballs = basketballs + 8) :
  soccerballs > basketballs ∧ soccerballs > basketballs + 8 := by
sorry

end most_balls_l3806_380621


namespace quadratic_root_relation_l3806_380624

/-- Given two quadratic equations with a specific relationship between their roots, 
    prove that the ratio of certain coefficients is 3. -/
theorem quadratic_root_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ (r₁ r₂ : ℝ), (r₁ + r₂ = -p ∧ r₁ * r₂ = m) ∧
                  (3 * r₁ + 3 * r₂ = -m ∧ 9 * r₁ * r₂ = n)) →
  n / p = 3 :=
by sorry

end quadratic_root_relation_l3806_380624


namespace quadratic_equation_roots_l3806_380653

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 + (1/2)*x + a - 2 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 + (1/2)*y + a - 2 = 0 ∧ y = -3/2) :=
by sorry

end quadratic_equation_roots_l3806_380653


namespace total_shot_cost_l3806_380656

/-- Represents the types of dogs Chuck breeds -/
inductive DogBreed
  | GoldenRetriever
  | GermanShepherd
  | Bulldog

/-- Represents the information for each dog breed -/
structure BreedInfo where
  pregnantDogs : Nat
  puppiesPerDog : Nat
  shotsPerPuppy : Nat
  costPerShot : Nat

/-- Calculates the total cost of shots for a specific breed -/
def breedShotCost (info : BreedInfo) : Nat :=
  info.pregnantDogs * info.puppiesPerDog * info.shotsPerPuppy * info.costPerShot

/-- Represents Chuck's dog breeding operation -/
def ChucksDogs : DogBreed → BreedInfo
  | DogBreed.GoldenRetriever => ⟨3, 4, 2, 5⟩
  | DogBreed.GermanShepherd => ⟨2, 5, 3, 8⟩
  | DogBreed.Bulldog => ⟨4, 3, 4, 10⟩

/-- Theorem stating the total cost of shots for all puppies -/
theorem total_shot_cost :
  (breedShotCost (ChucksDogs DogBreed.GoldenRetriever) +
   breedShotCost (ChucksDogs DogBreed.GermanShepherd) +
   breedShotCost (ChucksDogs DogBreed.Bulldog)) = 840 := by
  sorry

end total_shot_cost_l3806_380656


namespace value_of_b_l3806_380676

theorem value_of_b (a b c : ℝ) 
  (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
  (h2 : 6 * b * 7 = 1.5) : 
  b = 15 := by sorry

end value_of_b_l3806_380676


namespace problem_solution_l3806_380662

theorem problem_solution (x : ℚ) : (5 * x - 8 = 15 * x + 4) → (3 * (x + 9) = 129 / 5) := by
  sorry

end problem_solution_l3806_380662


namespace cubes_with_four_neighbors_eq_108_l3806_380640

/-- Represents a parallelepiped with dimensions a, b, and c. -/
structure Parallelepiped where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a > 3
  h2 : b > 3
  h3 : c > 3
  h4 : (a - 2) * (b - 2) * (c - 2) = 429

/-- The number of unit cubes with exactly 4 neighbors in a parallelepiped. -/
def cubes_with_four_neighbors (p : Parallelepiped) : ℕ :=
  4 * ((p.a - 2) + (p.b - 2) + (p.c - 2))

theorem cubes_with_four_neighbors_eq_108 (p : Parallelepiped) :
  cubes_with_four_neighbors p = 108 := by
  sorry

end cubes_with_four_neighbors_eq_108_l3806_380640


namespace cube_sum_and_reciprocal_l3806_380610

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cube_sum_and_reciprocal_l3806_380610


namespace function_composition_identity_l3806_380648

/-- Given two functions f and g defined as f(x) = Ax² + B and g(x) = Bx² + A,
    where A ≠ B, if f(g(x)) - g(f(x)) = 2(B - A) for all x, then A + B = 0 -/
theorem function_composition_identity (A B : ℝ) (h : A ≠ B) :
  (∀ x : ℝ, (A * (B * x^2 + A)^2 + B) - (B * (A * x^2 + B)^2 + A) = 2 * (B - A)) →
  A + B = 0 := by
sorry


end function_composition_identity_l3806_380648


namespace mens_total_wages_l3806_380622

/-- Proves that the men's total wages are 150 given the problem conditions --/
theorem mens_total_wages (W : ℕ) : 
  (12 : ℚ) * W = (20 : ℚ) → -- 12 men equal W women, W women equal 20 boys
  (12 : ℚ) * W * W + W * W * W + (20 : ℚ) * W = (450 : ℚ) → -- Total earnings equation
  (12 : ℚ) * ((450 : ℚ) / ((12 : ℚ) + W + (20 : ℚ))) = (150 : ℚ) -- Men's total wages
:= by sorry

end mens_total_wages_l3806_380622


namespace champagne_bottle_volume_l3806_380659

theorem champagne_bottle_volume
  (hot_tub_volume : ℚ)
  (quarts_per_gallon : ℚ)
  (bottle_cost : ℚ)
  (discount_rate : ℚ)
  (total_spent : ℚ)
  (h1 : hot_tub_volume = 40)
  (h2 : quarts_per_gallon = 4)
  (h3 : bottle_cost = 50)
  (h4 : discount_rate = 0.2)
  (h5 : total_spent = 6400) :
  (hot_tub_volume * quarts_per_gallon) / ((total_spent / (1 - discount_rate)) / bottle_cost) = 1 :=
by sorry

end champagne_bottle_volume_l3806_380659


namespace odd_integer_m_exists_l3806_380609

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 3

theorem odd_integer_m_exists (m : ℤ) (h_odd : m % 2 = 1) (h_g : g (g (g m)) = 14) : m = 121 := by
  sorry

end odd_integer_m_exists_l3806_380609


namespace min_value_theorem_min_value_achieved_l3806_380688

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2) :
  (x + 8*y) / (x*y) ≥ 9 :=
sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2) :
  (x + 8*y) / (x*y) = 9 ↔ x = 4/3 ∧ y = 1/3 :=
sorry

end min_value_theorem_min_value_achieved_l3806_380688


namespace sector_area_l3806_380644

/-- The area of a sector with a central angle of 120° and a radius of 3 is 3π. -/
theorem sector_area (angle : Real) (radius : Real) : 
  angle = 120 * π / 180 → radius = 3 → (1/2) * angle * radius^2 = 3 * π := by
  sorry

end sector_area_l3806_380644


namespace march_text_messages_l3806_380600

def T (n : ℕ) : ℕ := ((n^2) + 1) * n.factorial

theorem march_text_messages : T 5 = 3120 := by
  sorry

end march_text_messages_l3806_380600


namespace greatest_value_when_x_is_negative_six_l3806_380603

theorem greatest_value_when_x_is_negative_six :
  let x : ℝ := -6
  (2 - x > 2 + x) ∧
  (2 - x > x - 1) ∧
  (2 - x > x) ∧
  (2 - x > x / 2) := by
  sorry

end greatest_value_when_x_is_negative_six_l3806_380603


namespace fair_compensation_is_two_l3806_380678

/-- Represents the scenario of two merchants selling cows and buying sheep --/
structure MerchantScenario where
  num_cows : ℕ
  num_sheep : ℕ
  lamb_price : ℕ

/-- The conditions of the problem --/
def scenario_conditions (s : MerchantScenario) : Prop :=
  ∃ (q : ℕ),
    s.num_sheep = 2 * q + 1 ∧
    s.num_cows ^ 2 = 10 * s.num_sheep + s.lamb_price ∧
    s.lamb_price < 10 ∧
    s.lamb_price > 0

/-- The fair compensation amount --/
def fair_compensation (s : MerchantScenario) : ℕ :=
  (10 - s.lamb_price) / 2

/-- Theorem stating that the fair compensation is 2 yuan --/
theorem fair_compensation_is_two (s : MerchantScenario) 
  (h : scenario_conditions s) : fair_compensation s = 2 := by
  sorry


end fair_compensation_is_two_l3806_380678


namespace bcd4_hex_to_dec_l3806_380691

def hex_to_dec (digit : Char) : ℕ :=
  match digit with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | d   => d.toString.toNat!

def hex_string_to_dec (s : String) : ℕ :=
  s.foldr (fun d acc => hex_to_dec d + 16 * acc) 0

theorem bcd4_hex_to_dec :
  hex_string_to_dec "BCD4" = 31444 := by
  sorry

end bcd4_hex_to_dec_l3806_380691


namespace min_stamps_for_47_cents_l3806_380605

def stamps (x y : ℕ) : ℕ := 5 * x + 7 * y

theorem min_stamps_for_47_cents :
  ∃ (x y : ℕ), stamps x y = 47 ∧
  (∀ (a b : ℕ), stamps a b = 47 → x + y ≤ a + b) ∧
  x + y = 7 := by
  sorry

end min_stamps_for_47_cents_l3806_380605


namespace bus_and_walking_problem_l3806_380687

/-- Proof of the bus and walking problem -/
theorem bus_and_walking_problem
  (total_distance : ℝ)
  (walking_speed : ℝ)
  (bus_speed : ℝ)
  (rest_time : ℝ)
  (h1 : total_distance = 21)
  (h2 : walking_speed = 4)
  (h3 : bus_speed = 60)
  (h4 : rest_time = 1/6) -- 10 minutes in hours
  : ∃ (x y : ℝ),
    x + y = total_distance ∧
    x / bus_speed + total_distance / bus_speed = rest_time + y / walking_speed ∧
    x = 19 ∧
    y = 2 := by
  sorry


end bus_and_walking_problem_l3806_380687


namespace coin_identification_possible_l3806_380680

/-- Represents the weight of a coin -/
inductive CoinWeight
| Counterfeit
| Genuine

/-- Represents the result of a weighing -/
inductive WeighingResult
| LeftLighter
| RightLighter
| Equal

/-- Represents a coin -/
structure Coin :=
  (id : Nat)
  (weight : CoinWeight)

/-- Represents a weighing on the balance scale -/
def weighing (left : List Coin) (right : List Coin) : WeighingResult :=
  sorry

/-- Represents the set of all coins -/
def allCoins : List Coin :=
  sorry

/-- The number of coins -/
def numCoins : Nat := 14

/-- The number of counterfeit coins -/
def numCounterfeit : Nat := 7

/-- The number of genuine coins -/
def numGenuine : Nat := 7

/-- The maximum number of allowed weighings -/
def maxWeighings : Nat := 3

theorem coin_identification_possible :
  ∃ (strategy : List (List Coin × List Coin)),
    (strategy.length ≤ maxWeighings) ∧
    (∀ (c : Coin), c ∈ allCoins →
      (c.weight = CoinWeight.Counterfeit ↔ c.id ≤ numCounterfeit) ∧
      (c.weight = CoinWeight.Genuine ↔ c.id > numCounterfeit)) :=
  sorry

end coin_identification_possible_l3806_380680


namespace expand_polynomial_l3806_380694

theorem expand_polynomial (x : ℝ) : (2 + x^2) * (1 - x^4) = -x^6 + x^2 - 2*x^4 + 2 := by
  sorry

end expand_polynomial_l3806_380694


namespace dice_line_probability_l3806_380681

-- Define the dice outcomes
def DiceOutcome : Type := Fin 6

-- Define the probability space
def Ω : Type := DiceOutcome × DiceOutcome

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define the event that (x, y) lies on the line 2x - y = 1
def E : Set Ω :=
  {ω : Ω | 2 * (ω.1.val + 1) - (ω.2.val + 1) = 1}

-- Theorem statement
theorem dice_line_probability :
  P E = 1 / 12 := by sorry

end dice_line_probability_l3806_380681


namespace fib_gcd_consecutive_fib_gcd_identity_l3806_380635

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fib_gcd_consecutive (n : ℕ) : Nat.gcd (fib n) (fib (n + 1)) = 1 := by
  sorry

theorem fib_gcd_identity (m n : ℕ) 
  (h : ∀ a b : ℕ, fib (a + b) = fib b * fib (a + 1) + fib (b - 1) * fib a) : 
  fib (Nat.gcd m n) = Nat.gcd (fib m) (fib n) := by
  sorry

end fib_gcd_consecutive_fib_gcd_identity_l3806_380635


namespace square_difference_equality_l3806_380641

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end square_difference_equality_l3806_380641


namespace circle_center_correct_l3806_380654

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, find its center -/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 1 (-6) 1 10 (-7)
  findCircleCenter eq = CircleCenter.mk 3 (-5) :=
by sorry

end circle_center_correct_l3806_380654


namespace balloon_problem_l3806_380615

theorem balloon_problem (x : ℝ) : x + 5.0 = 12 → x = 7 := by
  sorry

end balloon_problem_l3806_380615


namespace quadratic_set_intersection_l3806_380625

theorem quadratic_set_intersection (p q : ℝ) : 
  let A := {x : ℝ | x^2 + p*x + q = 0}
  let B := {x : ℝ | x^2 - 3*x + 2 = 0}
  (A ∩ B = A) ↔ 
  ((p^2 < 4*q) ∨ (p = -2 ∧ q = 1) ∨ (p = -4 ∧ q = 4) ∨ (p = -3 ∧ q = 2)) :=
by sorry

end quadratic_set_intersection_l3806_380625


namespace blue_ball_weight_is_6_l3806_380664

/-- The weight of the blue ball in pounds -/
def blue_ball_weight : ℝ := 9.12 - 3.12

/-- The weight of the brown ball in pounds -/
def brown_ball_weight : ℝ := 3.12

/-- The total weight of both balls in pounds -/
def total_weight : ℝ := 9.12

theorem blue_ball_weight_is_6 : blue_ball_weight = 6 := by
  sorry

end blue_ball_weight_is_6_l3806_380664


namespace count_valid_pairs_l3806_380636

/-- The number of distinct ordered pairs of positive integers (x,y) satisfying 1/x + 1/y = 1/5 -/
def count_pairs : ℕ := 3

/-- Predicate defining valid pairs -/
def is_valid_pair (x y : ℕ+) : Prop :=
  (1 : ℚ) / x.val + (1 : ℚ) / y.val = (1 : ℚ) / 5

theorem count_valid_pairs :
  ∃ (S : Finset (ℕ+ × ℕ+)), 
    (∀ (p : ℕ+ × ℕ+), p ∈ S ↔ is_valid_pair p.1 p.2) ∧ 
    S.card = count_pairs :=
  sorry


end count_valid_pairs_l3806_380636


namespace product_of_sums_zero_l3806_380631

theorem product_of_sums_zero (x y z w : ℝ) 
  (sum_zero : x + y + z + w = 0)
  (sum_seventh_power_zero : x^7 + y^7 + z^7 + w^7 = 0) : 
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end product_of_sums_zero_l3806_380631


namespace distance_from_y_axis_is_18_l3806_380618

def point_P (x : ℝ) : ℝ × ℝ := (x, -9)

def distance_to_x_axis (p : ℝ × ℝ) : ℝ := |p.2|

def distance_to_y_axis (p : ℝ × ℝ) : ℝ := |p.1|

theorem distance_from_y_axis_is_18 (x : ℝ) :
  let p := point_P x
  distance_to_x_axis p = (1/2) * distance_to_y_axis p →
  distance_to_y_axis p = 18 := by sorry

end distance_from_y_axis_is_18_l3806_380618


namespace D_nec_not_suff_A_l3806_380698

-- Define propositions A, B, C, and D
variable (A B C D : Prop)

-- Define the relationships between the propositions
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom C_nec_and_suff_B : (B ↔ C)
axiom D_nec_not_suff_C : (C → D) ∧ ¬(D → C)

-- Theorem to prove
theorem D_nec_not_suff_A : (A → D) ∧ ¬(D → A) := by
  sorry

end D_nec_not_suff_A_l3806_380698


namespace distance_between_squares_l3806_380686

/-- Given a configuration of two squares where:
    - The smaller square has a perimeter of 8 cm
    - The larger square has an area of 49 cm²
    This theorem states that the distance between point A (top-right corner of the larger square)
    and point B (top-left corner of the smaller square) is approximately 10.3 cm. -/
theorem distance_between_squares (small_square_perimeter : ℝ) (large_square_area : ℝ)
    (h1 : small_square_perimeter = 8)
    (h2 : large_square_area = 49) :
    ∃ (distance : ℝ), abs (distance - Real.sqrt 106) < 0.1 ∧
    distance = Real.sqrt ((large_square_area.sqrt + small_square_perimeter / 4) ^ 2 +
    (large_square_area.sqrt - small_square_perimeter / 4) ^ 2) := by
  sorry

end distance_between_squares_l3806_380686


namespace student_scores_theorem_l3806_380655

/-- A score is a triple of integers, each between 0 and 7 inclusive -/
def Score := { s : Fin 3 → Fin 8 // True }

/-- Given two scores, returns true if the first score is at least as high as the second for each problem -/
def ScoreGreaterEq (s1 s2 : Score) : Prop :=
  ∀ i : Fin 3, s1.val i ≥ s2.val i

theorem student_scores_theorem (scores : Fin 49 → Score) :
  ∃ i j : Fin 49, i ≠ j ∧ ScoreGreaterEq (scores i) (scores j) := by
  sorry

#check student_scores_theorem

end student_scores_theorem_l3806_380655


namespace smallest_product_l3806_380645

def number_list : List Int := [-5, -3, -1, 2, 4, 6]

def is_valid_product (p : Int) : Prop :=
  ∃ (a b c : Int), a ∈ number_list ∧ b ∈ number_list ∧ c ∈ number_list ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ p = a * b * c

theorem smallest_product :
  ∀ p, is_valid_product p → p ≥ -120 :=
by sorry

end smallest_product_l3806_380645


namespace a_share_is_3630_l3806_380651

/-- Calculates the share of profit for an investor in a partnership business. -/
def calculate_share_of_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  ratio_a * total_profit

/-- Theorem stating that A's share of the profit is 3630 given the investments and total profit. -/
theorem a_share_is_3630 :
  calculate_share_of_profit 6300 4200 10500 12100 = 3630 := by
  sorry

end a_share_is_3630_l3806_380651


namespace race_time_proof_l3806_380683

/-- 
Given a race with three participants Patrick, Manu, and Amy:
- Patrick's race time is 60 seconds
- Manu's race time is 12 seconds more than Patrick's
- Amy's speed is twice Manu's speed
Prove that Amy's race time is 36 seconds
-/
theorem race_time_proof (patrick_time manu_time amy_time : ℝ) : 
  patrick_time = 60 →
  manu_time = patrick_time + 12 →
  amy_time * 2 = manu_time →
  amy_time = 36 := by
sorry

end race_time_proof_l3806_380683


namespace percentage_increase_calculation_l3806_380607

def original_earnings : ℝ := 60
def new_earnings : ℝ := 68

theorem percentage_increase_calculation :
  (new_earnings - original_earnings) / original_earnings * 100 = 13.33333333333333 := by
  sorry

end percentage_increase_calculation_l3806_380607


namespace system_solutions_l3806_380604

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * |y| - 4 * |x| = 6
def equation2 (x y a : ℝ) : Prop := x^2 + y^2 - 14*y + 49 - a^2 = 0

-- Define the number of solutions
def has_n_solutions (n : ℕ) (a : ℝ) : Prop :=
  ∃ (solutions : Finset (ℝ × ℝ)), 
    solutions.card = n ∧
    ∀ (x y : ℝ), (x, y) ∈ solutions ↔ equation1 x y ∧ equation2 x y a

-- Theorem statement
theorem system_solutions (a : ℝ) :
  (has_n_solutions 3 a ↔ |a| = 5 ∨ |a| = 9) ∧
  (has_n_solutions 2 a ↔ |a| = 3 ∨ (5 < |a| ∧ |a| < 9)) :=
sorry

end system_solutions_l3806_380604
