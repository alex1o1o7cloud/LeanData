import Mathlib

namespace NUMINAMATH_CALUDE_max_k_inequality_l3838_383807

theorem max_k_inequality (k : ℝ) : 
  (∀ (x y : ℤ), 4 * x^2 + y^2 + 1 ≥ k * x * (y + 1)) ↔ k ≤ 3 := by sorry

end NUMINAMATH_CALUDE_max_k_inequality_l3838_383807


namespace NUMINAMATH_CALUDE_M_on_line_l_line_l_equation_AB_length_l3838_383850

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Define point M
def point_M : ℝ × ℝ := (2, 1)

-- Define that M is on line l
theorem M_on_line_l : line_l point_M.1 point_M.2 := by sorry

-- Define that A and B are on the parabola
axiom A_on_parabola : ∃ (x y : ℝ), parabola x y ∧ line_l x y
axiom B_on_parabola : ∃ (x y : ℝ), parabola x y ∧ line_l x y

-- Define that M is the midpoint of AB
axiom M_midpoint_AB : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
  point_M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

-- Theorem 1: The equation of line l is 2x - y - 3 = 0
theorem line_l_equation : ∀ (x y : ℝ), line_l x y ↔ 2*x - y - 3 = 0 := by sorry

-- Theorem 2: The length of segment AB is √35
theorem AB_length : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
    point_M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = Real.sqrt 35 := by sorry

end NUMINAMATH_CALUDE_M_on_line_l_line_l_equation_AB_length_l3838_383850


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l3838_383809

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 2 * Nat.factorial 6 = 41040 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l3838_383809


namespace NUMINAMATH_CALUDE_angle_sum_at_point_l3838_383842

theorem angle_sum_at_point (y : ℝ) : 
  (170 : ℝ) + y + y = 360 → y = 95 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_at_point_l3838_383842


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l3838_383846

theorem solve_quadratic_equation (x : ℝ) : 2 * (x - 1)^2 = 8 → x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l3838_383846


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l3838_383839

/-- Given a school with 460 students, where 325 play football, 175 play cricket, 
    and 50 play neither, prove that 90 students play both sports. -/
theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) 
  (h1 : total = 460)
  (h2 : football = 325)
  (h3 : cricket = 175)
  (h4 : neither = 50)
  : total = football + cricket - 90 + neither := by
  sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l3838_383839


namespace NUMINAMATH_CALUDE_ordered_pairs_satisfying_conditions_l3838_383873

theorem ordered_pairs_satisfying_conditions :
  ∀ a b : ℕ+,
  (a.val^2 + b.val^2 + 25 = 15 * a.val * b.val) ∧
  (Nat.Prime (a.val^2 + a.val * b.val + b.val^2)) →
  ((a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ordered_pairs_satisfying_conditions_l3838_383873


namespace NUMINAMATH_CALUDE_sequence_sum_lower_bound_l3838_383823

theorem sequence_sum_lower_bound (n : ℕ) (a : ℕ → ℝ) 
  (h1 : a 1 = 0)
  (h2 : ∀ i ∈ Finset.range n, i ≥ 2 → |a i| = |a (i-1) + 1|) :
  (Finset.range n).sum a ≥ -n / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_lower_bound_l3838_383823


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3838_383829

theorem complex_equation_solution (z : ℂ) : 
  (3 + 4*I) / I = z / (1 + I) → z = 7 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3838_383829


namespace NUMINAMATH_CALUDE_sum_from_difference_and_squares_l3838_383874

theorem sum_from_difference_and_squares (m n : ℤ) 
  (h1 : m^2 - n^2 = 18) 
  (h2 : m - n = 9) : 
  m + n = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_from_difference_and_squares_l3838_383874


namespace NUMINAMATH_CALUDE_inequality_proof_l3838_383838

theorem inequality_proof (a : ℝ) : 
  (a^2 + 5)^2 + 4*a*(10 - a) - 8*a^3 ≥ 0 ∧ 
  ((a^2 + 5)^2 + 4*a*(10 - a) - 8*a^3 = 0 ↔ a = 5 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3838_383838


namespace NUMINAMATH_CALUDE_range_of_m_l3838_383893

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def q (x m : ℝ) : Prop := (x - 1 - m) * (x - 1 + m) ≤ 0

-- Define the sufficient condition relationship
def sufficient_condition (m : ℝ) : Prop :=
  ∀ x, q x m → p x

-- Define the not necessary condition relationship
def not_necessary_condition (m : ℝ) : Prop :=
  ∃ x, p x ∧ ¬(q x m)

-- Main theorem
theorem range_of_m :
  ∀ m : ℝ, m > 0 ∧ sufficient_condition m ∧ not_necessary_condition m
  → m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3838_383893


namespace NUMINAMATH_CALUDE_unique_solution_diophantine_equation_l3838_383895

theorem unique_solution_diophantine_equation :
  ∀ x y : ℕ+, 2 * x^2 + 5 * y^2 = 11 * (x * y - 11) ↔ x = 14 ∧ y = 27 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_diophantine_equation_l3838_383895


namespace NUMINAMATH_CALUDE_pet_owners_proof_l3838_383835

theorem pet_owners_proof (total_pet_owners : Nat) 
                         (only_dog_owners : Nat)
                         (only_cat_owners : Nat)
                         (cat_dog_snake_owners : Nat)
                         (total_snakes : Nat)
                         (h1 : total_pet_owners = 99)
                         (h2 : only_dog_owners = 15)
                         (h3 : only_cat_owners = 10)
                         (h4 : cat_dog_snake_owners = 3)
                         (h5 : total_snakes = 69) : 
  total_pet_owners = only_dog_owners + only_cat_owners + cat_dog_snake_owners + (total_snakes - cat_dog_snake_owners) + 5 :=
by
  sorry

#check pet_owners_proof

end NUMINAMATH_CALUDE_pet_owners_proof_l3838_383835


namespace NUMINAMATH_CALUDE_unique_shapes_count_l3838_383877

-- Define a rectangle
structure Rectangle where
  vertices : Fin 4 → Point

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define an ellipse
structure Ellipse where
  foci : Point × Point
  major_axis : ℝ

-- Function to count unique shapes
def count_unique_shapes (R : Rectangle) : ℕ :=
  let circles := sorry
  let ellipses := sorry
  circles + ellipses

-- Theorem statement
theorem unique_shapes_count (R : Rectangle) :
  count_unique_shapes R = 6 :=
sorry

end NUMINAMATH_CALUDE_unique_shapes_count_l3838_383877


namespace NUMINAMATH_CALUDE_range_of_m_l3838_383897

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  ∀ m : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m) ↔ m < 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3838_383897


namespace NUMINAMATH_CALUDE_largest_number_bound_l3838_383843

theorem largest_number_bound (a b c : ℝ) (sum_zero : a + b + c = 0) (product_eight : a * b * c = 8) :
  max a (max b c) ≥ 2 * Real.rpow 4 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_bound_l3838_383843


namespace NUMINAMATH_CALUDE_darkest_cell_value_l3838_383820

/-- Represents the grid structure -/
structure Grid :=
  (white1 white2 white3 white4 : Nat)
  (gray1 gray2 : Nat)
  (dark : Nat)

/-- The grid satisfies the problem conditions -/
def valid_grid (g : Grid) : Prop :=
  g.white1 > 1 ∧ g.white2 > 1 ∧ g.white3 > 1 ∧ g.white4 > 1 ∧
  g.white1 * g.white2 = 55 ∧
  g.white3 * g.white4 = 55 ∧
  g.gray1 = g.white1 * g.white3 ∧
  g.gray2 = g.white2 * g.white4 ∧
  g.dark = g.gray1 * g.gray2

theorem darkest_cell_value (g : Grid) :
  valid_grid g → g.dark = 245025 := by
  sorry

#check darkest_cell_value

end NUMINAMATH_CALUDE_darkest_cell_value_l3838_383820


namespace NUMINAMATH_CALUDE_min_value_inequality_l3838_383856

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1 / x + 1 / y) ≥ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (a + b) * (1 / a + 1 / b) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3838_383856


namespace NUMINAMATH_CALUDE_radius_of_larger_circle_l3838_383863

/-- Given two identical circles touching each other from the inside of a third circle,
    prove that the radius of the larger circle is 9 when the perimeter of the triangle
    formed by connecting the three centers is 18. -/
theorem radius_of_larger_circle (r R : ℝ) : r > 0 → R > r →
  (R - r) + (R - r) + 2 * r = 18 → R = 9 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_larger_circle_l3838_383863


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l3838_383824

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) (h_positive : L > 0 ∧ W > 0) :
  (1.10 * L) * (W * (1 - x / 100)) = L * W * 1.045 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l3838_383824


namespace NUMINAMATH_CALUDE_stamp_arrangement_count_l3838_383860

/-- Represents a stamp with its denomination -/
structure Stamp where
  denomination : Nat
  deriving Repr

/-- Represents an arrangement of stamps -/
def Arrangement := List Stamp

/-- Checks if an arrangement is valid (sums to 15 cents) -/
def isValidArrangement (arr : Arrangement) : Bool :=
  (arr.map (·.denomination)).sum = 15

/-- Checks if two arrangements are considered equivalent -/
def areEquivalentArrangements (arr1 arr2 : Arrangement) : Bool :=
  sorry  -- Implementation details omitted

/-- The set of all possible stamps -/
def allStamps : List Stamp :=
  (List.range 12).map (λ i => ⟨i + 1⟩) ++ (List.range 12).map (λ i => ⟨i + 1⟩)

/-- Generates all valid arrangements -/
def generateValidArrangements (stamps : List Stamp) : List Arrangement :=
  sorry  -- Implementation details omitted

/-- Counts distinct arrangements after considering equivalence -/
def countDistinctArrangements (arrangements : List Arrangement) : Nat :=
  sorry  -- Implementation details omitted

theorem stamp_arrangement_count :
  countDistinctArrangements (generateValidArrangements allStamps) = 213 := by
  sorry

end NUMINAMATH_CALUDE_stamp_arrangement_count_l3838_383860


namespace NUMINAMATH_CALUDE_photograph_perimeter_l3838_383841

/-- 
Given a rectangular photograph with a border, this theorem proves that 
if the total area with a 1-inch border is m square inches, and 
the total area with a 3-inch border is (m + 52) square inches, 
then the perimeter of the photograph is 10 inches.
-/
theorem photograph_perimeter 
  (w l m : ℝ) 
  (h1 : (w + 2) * (l + 2) = m) 
  (h2 : (w + 6) * (l + 6) = m + 52) : 
  2 * (w + l) = 10 :=
sorry

end NUMINAMATH_CALUDE_photograph_perimeter_l3838_383841


namespace NUMINAMATH_CALUDE_expand_binomials_l3838_383821

theorem expand_binomials (a : ℝ) : (a + 3) * (-a + 1) = -a^2 - 2*a + 3 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l3838_383821


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l3838_383816

theorem minimum_value_theorem (a b m n : ℝ) : 
  (∀ x, (x + 2) / (x + 1) < 0 ↔ a < x ∧ x < b) →
  m * a + n * b + 1 = 0 →
  m * n > 0 →
  (∀ m' n', m' * n' > 0 → m' * a + n' * b + 1 = 0 → 2 / m' + 1 / n' ≥ 2 / m + 1 / n) →
  2 / m + 1 / n = 9 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l3838_383816


namespace NUMINAMATH_CALUDE_product_expansion_l3838_383872

theorem product_expansion (x : ℝ) : 5*(x-6)*(x+9) + 3*x = 5*x^2 + 18*x - 270 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3838_383872


namespace NUMINAMATH_CALUDE_cary_calorie_deficit_l3838_383888

-- Define the given constants
def miles_walked : ℕ := 3
def calories_per_mile : ℕ := 150
def calories_consumed : ℕ := 200

-- Define the net calorie deficit
def net_calorie_deficit : ℕ := miles_walked * calories_per_mile - calories_consumed

-- Theorem statement
theorem cary_calorie_deficit : net_calorie_deficit = 250 := by
  sorry

end NUMINAMATH_CALUDE_cary_calorie_deficit_l3838_383888


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3838_383804

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3838_383804


namespace NUMINAMATH_CALUDE_age_of_seventh_person_l3838_383852

-- Define the ages and age differences
variable (A1 A2 A3 A4 A5 A6 A7 D1 D2 D3 D4 D5 : ℕ)

-- Define the conditions
axiom age_order : A1 < A2 ∧ A2 < A3 ∧ A3 < A4 ∧ A4 < A5 ∧ A5 < A6

axiom age_differences : 
  A2 = A1 + D1 ∧
  A3 = A2 + D2 ∧
  A4 = A3 + D3 ∧
  A5 = A4 + D4 ∧
  A6 = A5 + D5

axiom sum_of_six : A1 + A2 + A3 + A4 + A5 + A6 = 246

axiom sum_of_seven : A1 + A2 + A3 + A4 + A5 + A6 + A7 = 315

-- The theorem to prove
theorem age_of_seventh_person : A7 = 69 := by
  sorry

end NUMINAMATH_CALUDE_age_of_seventh_person_l3838_383852


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_two_distinct_roots_l3838_383818

theorem at_least_one_equation_has_two_distinct_roots
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (4 * b^2 - 4 * a * c > 0) ∨ (4 * c^2 - 4 * a * b > 0) ∨ (4 * a^2 - 4 * b * c > 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_two_distinct_roots_l3838_383818


namespace NUMINAMATH_CALUDE_power_equality_comparisons_l3838_383803

theorem power_equality_comparisons :
  (-2^3 = (-2)^3) ∧
  (3^2 ≠ 2^3) ∧
  (-3^2 ≠ (-3)^2) ∧
  (-(3 * 2)^2 ≠ -3 * 2^2) := by sorry

end NUMINAMATH_CALUDE_power_equality_comparisons_l3838_383803


namespace NUMINAMATH_CALUDE_cubic_equation_fraction_value_l3838_383851

theorem cubic_equation_fraction_value (a : ℝ) : 
  a^3 + 3*a^2 + a = 0 → 
  (2022*a^2) / (a^4 + 2015*a^2 + 1) = 0 ∨ (2022*a^2) / (a^4 + 2015*a^2 + 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_fraction_value_l3838_383851


namespace NUMINAMATH_CALUDE_smallest_integer_with_16_divisors_l3838_383861

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a given positive integer has exactly 16 positive divisors -/
def has_16_divisors (n : ℕ+) : Prop := num_divisors n = 16

theorem smallest_integer_with_16_divisors :
  (∃ (n : ℕ+), has_16_divisors n) ∧
  (∀ (m : ℕ+), has_16_divisors m → 384 ≤ m) ∧
  has_16_divisors 384 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_16_divisors_l3838_383861


namespace NUMINAMATH_CALUDE_f_3_range_l3838_383840

/-- Given a quadratic function f(x) = ax^2 - c with specific constraints on f(1) and f(2),
    we prove that f(3) lies within a certain range. -/
theorem f_3_range (a c : ℝ) (h1 : -4 ≤ a - c ∧ a - c ≤ -1) (h2 : -1 ≤ 4*a - c ∧ 4*a - c ≤ 5) :
  -1 ≤ 9*a - c ∧ 9*a - c ≤ 20 := by
  sorry

#check f_3_range

end NUMINAMATH_CALUDE_f_3_range_l3838_383840


namespace NUMINAMATH_CALUDE_gcd_4557_5115_l3838_383854

theorem gcd_4557_5115 : Nat.gcd 4557 5115 = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4557_5115_l3838_383854


namespace NUMINAMATH_CALUDE_exam_student_count_l3838_383834

theorem exam_student_count 
  (total_average : ℝ)
  (excluded_count : ℕ)
  (excluded_average : ℝ)
  (remaining_average : ℝ)
  (h1 : total_average = 80)
  (h2 : excluded_count = 5)
  (h3 : excluded_average = 20)
  (h4 : remaining_average = 95)
  : ∃ N : ℕ, N > 0 ∧ 
    N * total_average = 
    (N - excluded_count) * remaining_average + excluded_count * excluded_average :=
by
  sorry

end NUMINAMATH_CALUDE_exam_student_count_l3838_383834


namespace NUMINAMATH_CALUDE_stratified_sampling_population_size_l3838_383892

theorem stratified_sampling_population_size 
  (total_male : ℕ) 
  (sample_size : ℕ) 
  (female_in_sample : ℕ) 
  (h1 : total_male = 570) 
  (h2 : sample_size = 110) 
  (h3 : female_in_sample = 53) :
  let male_in_sample := sample_size - female_in_sample
  let total_population := (total_male * sample_size) / male_in_sample
  total_population = 1100 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_population_size_l3838_383892


namespace NUMINAMATH_CALUDE_thirty_five_power_ab_equals_R_power_b_times_S_power_a_l3838_383812

theorem thirty_five_power_ab_equals_R_power_b_times_S_power_a
  (a b : ℤ) (R S : ℝ) (hR : R = 5^a) (hS : S = 7^b) :
  35^(a*b) = R^b * S^a := by
  sorry

end NUMINAMATH_CALUDE_thirty_five_power_ab_equals_R_power_b_times_S_power_a_l3838_383812


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_one_l3838_383832

theorem factorial_fraction_equals_one : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_one_l3838_383832


namespace NUMINAMATH_CALUDE_girls_in_class_l3838_383813

theorem girls_in_class (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 56 → 
  4 * boys = 3 * girls → 
  total = girls + boys → 
  girls = 32 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l3838_383813


namespace NUMINAMATH_CALUDE_indeterminate_larger_number_l3838_383833

/-- Given two real numbers x and y and a constant k such that
    x * k = y + 1 and x + y = -64, prove that it's not possible
    to determine which of x or y is larger without additional information. -/
theorem indeterminate_larger_number (x y k : ℝ) 
    (h1 : x * k = y + 1) 
    (h2 : x + y = -64) : 
  ¬ (∀ x y : ℝ, (x * k = y + 1 ∧ x + y = -64) → x < y ∨ y < x) :=
by
  sorry


end NUMINAMATH_CALUDE_indeterminate_larger_number_l3838_383833


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3838_383871

theorem polynomial_simplification (p : ℝ) :
  (5 * p^4 + 4 * p^3 - 7 * p^2 + 9 * p - 3) + (-8 * p^4 + 2 * p^3 - p^2 - 3 * p + 4) =
  -3 * p^4 + 6 * p^3 - 8 * p^2 + 6 * p + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3838_383871


namespace NUMINAMATH_CALUDE_find_k_value_l3838_383822

theorem find_k_value (x y z k : ℝ) 
  (eq1 : 9 / (x + y) = k / (x + 2*z))
  (eq2 : k / (x + 2*z) = 14 / (z - y))
  (cond1 : y = 2*x)
  (cond2 : x + z = 10) :
  k = 46 := by
sorry

end NUMINAMATH_CALUDE_find_k_value_l3838_383822


namespace NUMINAMATH_CALUDE_hyperbola_parabola_ratio_l3838_383887

/-- Given a hyperbola and a parabola with specific properties, prove that the ratio of the hyperbola's semi-major and semi-minor axes is equal to √3/3. -/
theorem hyperbola_parabola_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∃ c : ℝ, c^2 = a^2 + b^2) →  -- Relationship between a, b, and c in a hyperbola
  (c / a = 2) →  -- Eccentricity is 2
  (c = 1) →  -- Right focus coincides with the focus of y^2 = 4x
  a / b = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_ratio_l3838_383887


namespace NUMINAMATH_CALUDE_always_sum_21_l3838_383800

theorem always_sum_21 (selection : Finset ℕ) :
  selection ⊆ Finset.range 20 →
  selection.card = 11 →
  ∃ x y, x ∈ selection ∧ y ∈ selection ∧ x ≠ y ∧ x + y = 21 :=
sorry

end NUMINAMATH_CALUDE_always_sum_21_l3838_383800


namespace NUMINAMATH_CALUDE_sequence_uniqueness_l3838_383819

theorem sequence_uniqueness (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, n ≥ 1 → (a (n + 1))^2 = 1 + (n + 2021) * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = n + 2019 := by
  sorry

end NUMINAMATH_CALUDE_sequence_uniqueness_l3838_383819


namespace NUMINAMATH_CALUDE_soccer_season_length_l3838_383801

theorem soccer_season_length (total_games : ℕ) (games_per_month : ℕ) (h1 : total_games = 27) (h2 : games_per_month = 9) :
  total_games / games_per_month = 3 := by
  sorry

end NUMINAMATH_CALUDE_soccer_season_length_l3838_383801


namespace NUMINAMATH_CALUDE_three_halves_equals_one_point_five_l3838_383886

theorem three_halves_equals_one_point_five : (3 : ℚ) / 2 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_three_halves_equals_one_point_five_l3838_383886


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l3838_383811

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 287 ∧ 
  ∀ m : ℕ, m > 0 ∧ m * (m + 1) = 20412 → m + (m + 1) ≥ 287 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l3838_383811


namespace NUMINAMATH_CALUDE_calories_in_one_bar_l3838_383817

/-- The number of calories in 3 candy bars -/
def total_calories : ℕ := 24

/-- The number of candy bars -/
def num_bars : ℕ := 3

/-- The number of calories in one candy bar -/
def calories_per_bar : ℕ := total_calories / num_bars

theorem calories_in_one_bar : calories_per_bar = 8 := by
  sorry

end NUMINAMATH_CALUDE_calories_in_one_bar_l3838_383817


namespace NUMINAMATH_CALUDE_shower_water_usage_l3838_383866

theorem shower_water_usage (roman remy riley ronan : ℝ) : 
  remy = 3 * roman + 1 →
  riley = roman + remy - 2 →
  ronan = riley / 2 →
  roman + remy + riley + ronan = 60 →
  remy = 18.85 := by
sorry

end NUMINAMATH_CALUDE_shower_water_usage_l3838_383866


namespace NUMINAMATH_CALUDE_triangle_area_equality_l3838_383825

theorem triangle_area_equality (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x^2 + y^2 = 49)
  (h5 : y^2 + y*z + z^2 = 36)
  (h6 : x^2 + Real.sqrt 3 * x * z + z^2 = 25) :
  2*x*y + Real.sqrt 3 * y*z + z*x = 24 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_equality_l3838_383825


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3838_383855

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≥ 1) 
  (h4 : ∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ 
    (Real.sqrt ((x + c)^2 + y^2) - Real.sqrt ((x - c)^2 + y^2))^2 = b^2 - 3*a*b) 
  (h5 : c^2 = a^2 + b^2) : 
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3838_383855


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3838_383858

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : ∀ n, a (n + 1) - a n = 2)
  (h3 : a 3 = 4) :
  a 12 = 22 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3838_383858


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l3838_383830

theorem bowling_team_average_weight 
  (original_players : ℕ) 
  (new_player1_weight : ℕ) 
  (new_player2_weight : ℕ) 
  (new_average_weight : ℕ) 
  (h1 : original_players = 7)
  (h2 : new_player1_weight = 110)
  (h3 : new_player2_weight = 60)
  (h4 : new_average_weight = 99) : 
  ∃ (original_average : ℕ), 
    (original_players * original_average + new_player1_weight + new_player2_weight) / 
    (original_players + 2) = new_average_weight ∧ 
    original_average = 103 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l3838_383830


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l3838_383890

theorem smallest_integer_solution (x : ℤ) : 3 * x - 7 ≤ 17 → x ≤ 8 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l3838_383890


namespace NUMINAMATH_CALUDE_pony_speed_l3838_383884

/-- The average speed of a pony given specific conditions of a chase scenario. -/
theorem pony_speed (horse_speed : ℝ) (head_start : ℝ) (chase_time : ℝ) : 
  horse_speed = 35 → head_start = 3 → chase_time = 4 → 
  ∃ (pony_speed : ℝ), pony_speed = 20 ∧ 
  horse_speed * chase_time = pony_speed * (head_start + chase_time) := by
sorry

end NUMINAMATH_CALUDE_pony_speed_l3838_383884


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3838_383828

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 45) →
  (a 2 + a 5 + a 8 = 39) →
  (a 3 + a 6 + a 9 = 33) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3838_383828


namespace NUMINAMATH_CALUDE_digit_120_is_1_l3838_383878

/-- Represents the decimal number formed by concatenating integers 1 to 51 -/
def x : ℚ :=
  0.123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051

/-- Returns the nth digit after the decimal point in a rational number -/
def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem digit_120_is_1 : nthDigitAfterDecimal x 120 = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_120_is_1_l3838_383878


namespace NUMINAMATH_CALUDE_water_rise_in_vessel_l3838_383899

/-- Represents the rise in water level when a cubical box is immersed in a rectangular vessel -/
theorem water_rise_in_vessel 
  (vessel_length : ℝ) 
  (vessel_breadth : ℝ) 
  (box_edge : ℝ) 
  (h : vessel_length = 60 ∧ vessel_breadth = 30 ∧ box_edge = 30) : 
  (box_edge ^ 3) / (vessel_length * vessel_breadth) = 15 := by
  sorry

#check water_rise_in_vessel

end NUMINAMATH_CALUDE_water_rise_in_vessel_l3838_383899


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l3838_383865

theorem range_of_a_minus_b (a b : ℝ) (θ : ℝ) 
  (h1 : |a - Real.sin θ ^ 2| ≤ 1) 
  (h2 : |b + Real.cos θ ^ 2| ≤ 1) : 
  -1 ≤ a - b ∧ a - b ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l3838_383865


namespace NUMINAMATH_CALUDE_baron_munchausen_claim_l3838_383882

theorem baron_munchausen_claim (weights : Finset ℕ) : 
  weights.card = 8 ∧ weights = Finset.range 8 →
  ∃ (A B C : Finset ℕ), 
    A ∪ B ∪ C = weights ∧
    A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
    A.card = 2 ∧ B.card = 5 ∧ C.card = 1 ∧
    (A.sum id = B.sum id) ∧
    (∀ w ∈ C, w = A.sum id - B.sum id) :=
by sorry

end NUMINAMATH_CALUDE_baron_munchausen_claim_l3838_383882


namespace NUMINAMATH_CALUDE_sum_ge_sum_of_sqrt_products_l3838_383831

theorem sum_ge_sum_of_sqrt_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

end NUMINAMATH_CALUDE_sum_ge_sum_of_sqrt_products_l3838_383831


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3838_383896

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27 * x^3) 
  (h3 : a - b = 2 * x) : 
  a = 3.041 * x ∨ a = -1.041 * x := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3838_383896


namespace NUMINAMATH_CALUDE_divisibility_property_l3838_383869

theorem divisibility_property (n : ℕ) (hn : n > 0) :
  ∃ (a b : ℤ), (n : ℤ) ∣ (4 * a^2 + 9 * b^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3838_383869


namespace NUMINAMATH_CALUDE_max_d_value_l3838_383880

def a (n : ℕ) : ℕ := 100 + n^n

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n-1))

theorem max_d_value :
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → d n ≤ 401 ∧ ∃ (m : ℕ), m ≥ N ∧ d m = 401 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l3838_383880


namespace NUMINAMATH_CALUDE_point_b_coordinates_l3838_383881

/-- Given a circle and two points A and B, if the squared distance from any point on the circle to A
    is twice the squared distance to B, then B has coordinates (1, 1). -/
theorem point_b_coordinates (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → (x - 2)^2 + (y - 2)^2 = 2*((x - a)^2 + (y - b)^2)) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l3838_383881


namespace NUMINAMATH_CALUDE_whitney_book_cost_l3838_383867

/-- Proves that given the conditions of Whitney's purchase, each book costs $11. -/
theorem whitney_book_cost (num_books num_magazines : ℕ) (magazine_cost total_cost book_cost : ℚ) : 
  num_books = 16 →
  num_magazines = 3 →
  magazine_cost = 1 →
  total_cost = 179 →
  total_cost = num_books * book_cost + num_magazines * magazine_cost →
  book_cost = 11 := by
sorry

end NUMINAMATH_CALUDE_whitney_book_cost_l3838_383867


namespace NUMINAMATH_CALUDE_marcos_strawberries_weight_l3838_383836

theorem marcos_strawberries_weight 
  (total_weight : ℝ) 
  (dads_weight : ℝ) 
  (h1 : total_weight = 20)
  (h2 : dads_weight = 17) : 
  total_weight - dads_weight = 3 := by
sorry

end NUMINAMATH_CALUDE_marcos_strawberries_weight_l3838_383836


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l3838_383853

/-- Parabola with equation y^2 = 2x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line passing through a point -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- Intersection points of a line and a parabola -/
def Intersection (p : Parabola) (l : Line) : Set (ℝ × ℝ) :=
  {pt | p.equation pt.1 pt.2 ∧ pt.2 = l.slope * (pt.1 - l.point.1) + l.point.2}

theorem parabola_intersection_theorem (p : Parabola) (l : Line) 
    (A B : ℝ × ℝ) (hA : A ∈ Intersection p l) (hB : B ∈ Intersection p l) :
  p.equation 0.5 0 →  -- Focus is on the parabola
  l.point = p.focus →  -- Line passes through the focus
  ‖A - B‖ = 25/12 →  -- Distance between A and B
  ‖A - p.focus‖ < ‖B - p.focus‖ →  -- AF < BF
  ‖A - p.focus‖ = 5/6 :=  -- |AF| = 5/6
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l3838_383853


namespace NUMINAMATH_CALUDE_smallest_number_l3838_383889

theorem smallest_number (a b c d : ℝ) (h1 : a = 3) (h2 : b = -2) (h3 : c = 1/2) (h4 : d = 2) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by sorry

end NUMINAMATH_CALUDE_smallest_number_l3838_383889


namespace NUMINAMATH_CALUDE_fair_coin_five_tosses_l3838_383868

/-- A fair coin is a coin with equal probability of landing on either side. -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of a specific sequence of n tosses for a fair coin. -/
def prob_sequence (n : ℕ) (p : ℝ) : ℝ := p ^ n

/-- The probability of landing on the same side for n tosses of a fair coin. -/
def prob_same_side (n : ℕ) (p : ℝ) : ℝ := 2 * (prob_sequence n p)

theorem fair_coin_five_tosses (p : ℝ) (h : fair_coin p) :
  prob_same_side 5 p = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_five_tosses_l3838_383868


namespace NUMINAMATH_CALUDE_weight_gain_theorem_l3838_383864

def weight_gain_problem (initial_weight first_month_gain second_month_gain : ℕ) : Prop :=
  initial_weight + first_month_gain + second_month_gain = 120

theorem weight_gain_theorem : 
  weight_gain_problem 70 20 30 := by sorry

end NUMINAMATH_CALUDE_weight_gain_theorem_l3838_383864


namespace NUMINAMATH_CALUDE_bananas_left_l3838_383862

/-- The number of bananas in a dozen -/
def dozen : ℕ := 12

/-- The number of bananas Anthony ate -/
def eaten : ℕ := 2

/-- Theorem: The number of bananas left is 10 -/
theorem bananas_left : dozen - eaten = 10 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_l3838_383862


namespace NUMINAMATH_CALUDE_min_value_of_f_l3838_383859

/-- The function f(x) = 3x^2 - 18x + 2205 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2205

theorem min_value_of_f :
  ∃ (min : ℝ), min = 2178 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3838_383859


namespace NUMINAMATH_CALUDE_money_split_proof_l3838_383805

/-- 
Given two people splitting money in a 2:3 ratio where the smaller share is $50,
prove that the total amount shared is $125.
-/
theorem money_split_proof (smaller_share : ℕ) (total : ℕ) : 
  smaller_share = 50 → 
  2 * total = 5 * smaller_share →
  total = 125 := by
sorry

end NUMINAMATH_CALUDE_money_split_proof_l3838_383805


namespace NUMINAMATH_CALUDE_shanes_remaining_gum_is_eight_l3838_383848

/-- The number of pieces of gum Shane has left after a series of exchanges and consumption --/
def shanes_remaining_gum : ℕ :=
  let elyses_initial_gum : ℕ := 100
  let ricks_gum : ℕ := elyses_initial_gum / 2
  let shanes_initial_gum : ℕ := ricks_gum / 3
  let shanes_gum_after_cousin : ℕ := shanes_initial_gum + 10
  let shanes_gum_after_chewing : ℕ := shanes_gum_after_cousin - 11
  let gum_shared_with_sarah : ℕ := shanes_gum_after_chewing / 2
  shanes_gum_after_chewing - gum_shared_with_sarah

theorem shanes_remaining_gum_is_eight :
  shanes_remaining_gum = 8 := by
  sorry

end NUMINAMATH_CALUDE_shanes_remaining_gum_is_eight_l3838_383848


namespace NUMINAMATH_CALUDE_paper_cups_pallets_l3838_383827

theorem paper_cups_pallets (total : ℕ) (towels tissues plates cups : ℕ) : 
  total = 20 ∧
  towels = total / 2 ∧
  tissues = total / 4 ∧
  plates = total / 5 ∧
  total = towels + tissues + plates + cups →
  cups = 1 := by
sorry

end NUMINAMATH_CALUDE_paper_cups_pallets_l3838_383827


namespace NUMINAMATH_CALUDE_total_gold_value_proof_l3838_383857

/-- The value of one gold bar in dollars -/
def gold_bar_value : ℕ := 2200

/-- The number of gold bars Legacy has -/
def legacy_bars : ℕ := 5

/-- The number of gold bars Aleena has -/
def aleena_bars : ℕ := legacy_bars - 2

/-- The total value of gold for Legacy and Aleena -/
def total_gold_value : ℕ := gold_bar_value * (legacy_bars + aleena_bars)

theorem total_gold_value_proof : total_gold_value = 17600 := by
  sorry

end NUMINAMATH_CALUDE_total_gold_value_proof_l3838_383857


namespace NUMINAMATH_CALUDE_divisibility_proof_l3838_383802

theorem divisibility_proof (a b c : ℝ) 
  (h : (a ≠ 0 ∧ b ≠ 0) ∨ (a ≠ 0 ∧ c ≠ 0) ∨ (b ≠ 0 ∧ c ≠ 0)) :
  ∃ k : ℤ, (a + b + c)^7 - a^7 - b^7 - c^7 = k * (7 * (a + b) * (b + c) * (c + a)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_proof_l3838_383802


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3838_383849

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular polygon satisfies the given condition if the number of its diagonals
    plus 6 equals twice the number of its sides -/
def satisfies_condition (n : ℕ) : Prop :=
  num_diagonals n + 6 = 2 * n

theorem regular_polygon_sides :
  ∃ (n : ℕ), n > 2 ∧ satisfies_condition n ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3838_383849


namespace NUMINAMATH_CALUDE_alternate_arrangements_count_l3838_383885

/-- The number of ways to arrange two men and two women alternately in a row -/
def alternateArrangements : ℕ :=
  let menCount := 2
  let womenCount := 2
  let manFirstArrangements := menCount * womenCount
  let womanFirstArrangements := womenCount * menCount
  manFirstArrangements + womanFirstArrangements

theorem alternate_arrangements_count :
  alternateArrangements = 8 := by
  sorry

end NUMINAMATH_CALUDE_alternate_arrangements_count_l3838_383885


namespace NUMINAMATH_CALUDE_base3_sum_theorem_l3838_383844

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/-- Converts a decimal number to its base 3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The main theorem stating the sum of the given base 3 numbers -/
theorem base3_sum_theorem :
  let a := base3ToDecimal [2, 1, 2, 1]
  let b := base3ToDecimal [1, 2, 1, 2]
  let c := base3ToDecimal [2, 1, 2]
  let d := base3ToDecimal [2]
  decimalToBase3 (a + b + c + d) = [2, 2, 0, 1] := by sorry

end NUMINAMATH_CALUDE_base3_sum_theorem_l3838_383844


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l3838_383847

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l3838_383847


namespace NUMINAMATH_CALUDE_some_birds_are_white_l3838_383876

-- Define our universe
variable (U : Type)

-- Define our predicates
variable (Swan : U → Prop)
variable (Bird : U → Prop)
variable (White : U → Prop)

-- State our theorem
theorem some_birds_are_white
  (h1 : ∀ x, Swan x → White x)  -- All swans are white
  (h2 : ∃ x, Bird x ∧ Swan x)   -- Some birds are swans
  : ∃ x, Bird x ∧ White x :=    -- Conclusion: Some birds are white
by sorry

end NUMINAMATH_CALUDE_some_birds_are_white_l3838_383876


namespace NUMINAMATH_CALUDE_max_segments_proof_l3838_383815

/-- Given n consecutive points on a line with total length 1, 
    this function returns the maximum number of segments with length ≥ a,
    where 0 ≤ a ≤ 1/(n-1) -/
def max_segments (n : ℕ) (a : ℝ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating that for n consecutive points on a line with total length 1,
    and 0 ≤ a ≤ 1/(n-1), the maximum number of segments with length ≥ a is n(n-1)/2 -/
theorem max_segments_proof (n : ℕ) (a : ℝ) 
    (h1 : n > 1) 
    (h2 : 0 ≤ a) 
    (h3 : a ≤ 1 / (n - 1)) : 
  max_segments n a = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_segments_proof_l3838_383815


namespace NUMINAMATH_CALUDE_average_questions_correct_l3838_383891

def dongwoos_group : List Nat := [16, 22, 30, 26, 18, 20]

theorem average_questions_correct : 
  (List.sum dongwoos_group) / (List.length dongwoos_group) = 22 := by
  sorry

end NUMINAMATH_CALUDE_average_questions_correct_l3838_383891


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3838_383837

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  x + 1 > 0 ∧ x + 3 ≤ 4

-- Define the solution set
def solution_set : Set ℝ :=
  {x : ℝ | -1 < x ∧ x ≤ 1}

-- Theorem statement
theorem inequality_system_solution_set :
  {x : ℝ | inequality_system x} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3838_383837


namespace NUMINAMATH_CALUDE_floor_abs_sum_l3838_383879

theorem floor_abs_sum : ⌊|(-5.3 : ℝ)|⌋ + |⌊(-5.3 : ℝ)⌋| = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l3838_383879


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3838_383898

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y + 6 = 0 → x + a*(a+1)*y + (a^2-1) = 0 → 
   (a * 1 + 2 * (a*(a+1)) = 0)) → 
  (a = 0 ∨ a = -3/2) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3838_383898


namespace NUMINAMATH_CALUDE_inequality_proof_l3838_383826

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 3*b^3) / (5*a + b) + (b^3 + 3*c^3) / (5*b + c) + (c^3 + 3*a^3) / (5*c + a) ≥ 
  2/3 * (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3838_383826


namespace NUMINAMATH_CALUDE_expression_equality_implies_square_l3838_383870

theorem expression_equality_implies_square (x y : ℕ) 
  (h : (1 : ℚ) / x + 1 / y + 1 / (x * y) = 1 / (x + 4) + 1 / (y - 4) + 1 / ((x + 4) * (y - 4))) :
  ∃ n : ℕ, x * y + 4 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_implies_square_l3838_383870


namespace NUMINAMATH_CALUDE_complement_of_union_is_four_l3838_383883

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem to prove
theorem complement_of_union_is_four :
  (M ∪ N)ᶜ = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_four_l3838_383883


namespace NUMINAMATH_CALUDE_lcm_of_135_and_468_l3838_383845

theorem lcm_of_135_and_468 : Nat.lcm 135 468 = 7020 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_135_and_468_l3838_383845


namespace NUMINAMATH_CALUDE_original_number_l3838_383810

theorem original_number (x : ℝ) : 1 + 1 / x = 9 / 4 → x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l3838_383810


namespace NUMINAMATH_CALUDE_prime_sum_of_powers_l3838_383894

theorem prime_sum_of_powers (n : ℕ) : 
  (∃ (a b c : ℤ), a + b + c = 0 ∧ Nat.Prime (Int.natAbs (a^n + b^n + c^n))) ↔ Even n := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_of_powers_l3838_383894


namespace NUMINAMATH_CALUDE_vector_sum_length_l3838_383814

/-- Given vectors a and b in ℝ², prove that |a + 2b| = √61 under specific conditions. -/
theorem vector_sum_length (a b : ℝ × ℝ) : 
  a = (3, -4) → 
  ‖b‖ = 2 → 
  (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) = 1/2 →
  ‖a + 2 • b‖ = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_length_l3838_383814


namespace NUMINAMATH_CALUDE_prob_two_hearts_is_one_seventeenth_l3838_383806

/-- A standard deck of cards. -/
structure Deck :=
  (cards : Finset Nat)
  (size : cards.card = 52)
  (suits : Finset Nat)
  (suit_size : suits.card = 4)

/-- The number of hearts in a standard deck. -/
def hearts_count : Nat := 13

/-- The probability of drawing two hearts from a well-shuffled standard deck. -/
def prob_two_hearts (d : Deck) : ℚ :=
  (hearts_count * (hearts_count - 1)) / (d.cards.card * (d.cards.card - 1))

/-- Theorem: The probability of drawing two hearts from a well-shuffled standard deck is 1/17. -/
theorem prob_two_hearts_is_one_seventeenth (d : Deck) :
  prob_two_hearts d = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_hearts_is_one_seventeenth_l3838_383806


namespace NUMINAMATH_CALUDE_triangle_max_area_l3838_383808

/-- The maximum area of a triangle with one side of length 3 and the sum of the other two sides equal to 5 is 3. -/
theorem triangle_max_area :
  ∀ (b c : ℝ),
  b > 0 → c > 0 →
  b + c = 5 →
  let a := 3
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area ≤ 3 ∧ ∃ (b₀ c₀ : ℝ), b₀ > 0 ∧ c₀ > 0 ∧ b₀ + c₀ = 5 ∧
    let s₀ := (a + b₀ + c₀) / 2
    Real.sqrt (s₀ * (s₀ - a) * (s₀ - b₀) * (s₀ - c₀)) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3838_383808


namespace NUMINAMATH_CALUDE_harmonic_mean_of_three_fourths_and_five_sixths_l3838_383875

theorem harmonic_mean_of_three_fourths_and_five_sixths :
  let a : ℚ := 3/4
  let b : ℚ := 5/6
  let harmonic_mean (x y : ℚ) := 2 * x * y / (x + y)
  harmonic_mean a b = 15/19 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_of_three_fourths_and_five_sixths_l3838_383875
