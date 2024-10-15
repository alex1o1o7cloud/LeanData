import Mathlib

namespace NUMINAMATH_CALUDE_extremum_point_implies_a_eq_3_f_increasing_when_a_le_2_max_m_value_l2636_263637

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - a*x

theorem extremum_point_implies_a_eq_3 :
  ∀ a : ℝ, (∀ h : ℝ, h ≠ 0 → (f a (1 + h) - f a 1) / h = 0) → a = 3 :=
sorry

theorem f_increasing_when_a_le_2 :
  ∀ a : ℝ, 0 < a → a ≤ 2 → StrictMono (f a) :=
sorry

theorem max_m_value :
  ∃ m : ℝ, m = -(Real.log 2)⁻¹ ∧
  (∀ a x₀ : ℝ, 1 < a → a < 2 → 1 ≤ x₀ → x₀ ≤ 2 → f a x₀ > m * Real.log a) ∧
  (∀ m' : ℝ, m' > m → ∃ a x₀ : ℝ, 1 < a ∧ a < 2 ∧ 1 ≤ x₀ ∧ x₀ ≤ 2 ∧ f a x₀ ≤ m' * Real.log a) :=
sorry

end NUMINAMATH_CALUDE_extremum_point_implies_a_eq_3_f_increasing_when_a_le_2_max_m_value_l2636_263637


namespace NUMINAMATH_CALUDE_rectangle_roots_l2636_263656

/-- The polynomial whose roots we are considering -/
def f (a : ℝ) (z : ℂ) : ℂ := z^4 - 8*z^3 + 13*a*z^2 - 3*(3*a^2 + 2*a - 4)*z + 1

/-- Predicate to check if four complex numbers form vertices of a rectangle -/
def isRectangle (z₁ z₂ z₃ z₄ : ℂ) : Prop := sorry

/-- The theorem stating that a = 3 is the only real value satisfying the condition -/
theorem rectangle_roots (a : ℝ) : 
  (∃ z₁ z₂ z₃ z₄ : ℂ, f a z₁ = 0 ∧ f a z₂ = 0 ∧ f a z₃ = 0 ∧ f a z₄ = 0 ∧ 
    isRectangle z₁ z₂ z₃ z₄) ↔ a = 3 := by sorry

end NUMINAMATH_CALUDE_rectangle_roots_l2636_263656


namespace NUMINAMATH_CALUDE_expression_value_l2636_263602

theorem expression_value (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : abs a ≠ abs b) :
  let expr1 := b^2 / a^2 + a^2 / b^2 - 2
  let expr2 := (a + b) / (b - a) + (b - a) / (a + b)
  let expr3 := (1 / a^2 + 1 / b^2) / (1 / b^2 - 1 / a^2) - (1 / b^2 - 1 / a^2) / (1 / a^2 + 1 / b^2)
  expr1 * expr2 * expr3 = -8 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2636_263602


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l2636_263626

theorem smallest_four_digit_multiple_of_18 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n → 1008 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l2636_263626


namespace NUMINAMATH_CALUDE_expected_value_is_six_point_five_l2636_263698

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of rolling a fair 12-sided die with faces numbered from 1 to 12 -/
def expected_value : ℚ :=
  (Finset.sum twelve_sided_die (λ i => i + 1)) / 12

/-- Theorem: The expected value of rolling a fair 12-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_is_six_point_five :
  expected_value = 13/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_six_point_five_l2636_263698


namespace NUMINAMATH_CALUDE_sum_of_smallest_solutions_l2636_263651

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the equation
def equation (x : ℝ) : Prop := x - floor x = 2 / (floor x : ℝ)

-- Define the set of positive solutions
def positive_solutions : Set ℝ := {x : ℝ | x > 0 ∧ equation x}

-- State the theorem
theorem sum_of_smallest_solutions :
  ∃ (s₁ s₂ s₃ : ℝ),
    s₁ ∈ positive_solutions ∧
    s₂ ∈ positive_solutions ∧
    s₃ ∈ positive_solutions ∧
    (∀ x ∈ positive_solutions, x ≤ s₁ ∨ x ≤ s₂ ∨ x ≤ s₃) ∧
    s₁ + s₂ + s₃ = 13 + 17 / 30 :=
sorry

end NUMINAMATH_CALUDE_sum_of_smallest_solutions_l2636_263651


namespace NUMINAMATH_CALUDE_equation_equivalence_l2636_263659

theorem equation_equivalence (y : ℝ) (Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) :
  10 * (6 * y + 14 * Real.pi) = 4 * Q :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2636_263659


namespace NUMINAMATH_CALUDE_hard_candy_coloring_is_30_l2636_263650

/-- The amount of food colouring used for each lollipop in milliliters -/
def lollipop_coloring : ℕ := 8

/-- The number of lollipops made in a day -/
def lollipops_made : ℕ := 150

/-- The number of hard candies made in a day -/
def hard_candies_made : ℕ := 20

/-- The total amount of food colouring used in a day in milliliters -/
def total_coloring : ℕ := 1800

/-- The amount of food colouring needed for each hard candy in milliliters -/
def hard_candy_coloring : ℕ := (total_coloring - lollipop_coloring * lollipops_made) / hard_candies_made

theorem hard_candy_coloring_is_30 : hard_candy_coloring = 30 := by
  sorry

end NUMINAMATH_CALUDE_hard_candy_coloring_is_30_l2636_263650


namespace NUMINAMATH_CALUDE_rulers_in_drawer_l2636_263625

/-- Given an initial number of rulers and an additional number of rulers,
    calculate the total number of rulers in the drawer. -/
def total_rulers (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that with 46 rulers initially and 25 rulers added,
    the total number of rulers in the drawer is 71. -/
theorem rulers_in_drawer : total_rulers 46 25 = 71 := by
  sorry

end NUMINAMATH_CALUDE_rulers_in_drawer_l2636_263625


namespace NUMINAMATH_CALUDE_existence_of_integers_l2636_263655

theorem existence_of_integers (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ (x y k : ℤ), 0 < 2 * k ∧ 2 * k < p ∧ k * p + 3 = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l2636_263655


namespace NUMINAMATH_CALUDE_expression_simplification_l2636_263652

theorem expression_simplification (a : ℝ) (h : a^2 + 2*a - 8 = 0) :
  ((a^2 - 4) / (a^2 - 4*a + 4) - a / (a - 2)) / ((a^2 + 2*a) / (a - 2)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2636_263652


namespace NUMINAMATH_CALUDE_pet_store_cages_l2636_263643

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 56)
  (h2 : sold_puppies = 24)
  (h3 : puppies_per_cage = 4) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 8 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l2636_263643


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2636_263699

theorem inequality_solution_set (a b c : ℝ) : 
  a > 0 → 
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 2 ↔ 0 ≤ a * x^2 + b * x + c ∧ a * x^2 + b * x + c ≤ 1) →
  4 * a + 5 * b + c = -1/4 ∨ 4 * a + 5 * b + c = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2636_263699


namespace NUMINAMATH_CALUDE_count_valid_numbers_valid_numbers_are_l2636_263686

def digits : List Nat := [2, 3, 0]

def is_valid_number (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits

def valid_numbers : List Nat :=
  (List.range 1000).filter is_valid_number

theorem count_valid_numbers :
  valid_numbers.length = 4 := by sorry

theorem valid_numbers_are :
  valid_numbers = [230, 203, 302, 320] := by sorry

end NUMINAMATH_CALUDE_count_valid_numbers_valid_numbers_are_l2636_263686


namespace NUMINAMATH_CALUDE_triangle_area_l2636_263636

theorem triangle_area (a c : Real) (B : Real) 
  (h1 : a = Real.sqrt 2)
  (h2 : c = 2 * Real.sqrt 2)
  (h3 : B = 30 * π / 180) :
  (1/2) * a * c * Real.sin B = 1 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2636_263636


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2636_263601

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2636_263601


namespace NUMINAMATH_CALUDE_baker_sales_difference_l2636_263600

theorem baker_sales_difference (cakes_made pastries_made cakes_sold pastries_sold : ℕ) : 
  cakes_made = 14 →
  pastries_made = 153 →
  cakes_sold = 97 →
  pastries_sold = 8 →
  cakes_sold - pastries_sold = 89 := by
sorry

end NUMINAMATH_CALUDE_baker_sales_difference_l2636_263600


namespace NUMINAMATH_CALUDE_minimal_hexahedron_volume_l2636_263697

/-- A trihedral angle -/
structure TrihedralAngle where
  planarAngle : ℝ

/-- The configuration of two trihedral angles -/
structure TrihedralAngleConfiguration where
  angle1 : TrihedralAngle
  angle2 : TrihedralAngle
  vertexDistance : ℝ
  isEquidistant : Bool

/-- The volume of the hexahedron bounded by the faces of two trihedral angles -/
def hexahedronVolume (config : TrihedralAngleConfiguration) : ℝ := sorry

/-- The theorem stating the minimal volume of the hexahedron -/
theorem minimal_hexahedron_volume 
  (config : TrihedralAngleConfiguration) 
  (h1 : config.angle1.planarAngle = π/3) 
  (h2 : config.angle2.planarAngle = π/2)
  (h3 : config.isEquidistant = true) :
  hexahedronVolume config = (config.vertexDistance^3 * Real.sqrt 3) / 20 := by
  sorry


end NUMINAMATH_CALUDE_minimal_hexahedron_volume_l2636_263697


namespace NUMINAMATH_CALUDE_solve_equations_l2636_263696

theorem solve_equations :
  (∃ x : ℝ, 2 * (x + 8) = 3 * (x - 1) ∧ x = 19) ∧
  (∃ y : ℝ, (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l2636_263696


namespace NUMINAMATH_CALUDE_committee_formation_count_l2636_263674

def total_students : ℕ := 8
def committee_size : ℕ := 4
def required_students : ℕ := 2
def remaining_students : ℕ := total_students - required_students

theorem committee_formation_count : 
  Nat.choose remaining_students (committee_size - required_students) = 15 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l2636_263674


namespace NUMINAMATH_CALUDE_special_function_properties_l2636_263639

/-- A function satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y - 3) ∧
  (∀ x : ℝ, x > 0 → f x < 3)

theorem special_function_properties (f : ℝ → ℝ) (hf : SpecialFunction f) :
  (f 0 = 3) ∧
  (∀ x y : ℝ, x < y → f y < f x) ∧
  (∀ x : ℝ, (∀ t : ℝ, t ∈ Set.Ioo 2 4 → 
    f ((t - 2) * |x - 4|) + 3 > f (t^2 + 8) + f (5 - 4*t)) →
    x ∈ Set.Icc (-5/2) (21/2)) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l2636_263639


namespace NUMINAMATH_CALUDE_average_of_first_group_l2636_263647

theorem average_of_first_group (n₁ : ℕ) (n₂ : ℕ) (avg₂ : ℝ) (avg_total : ℝ) :
  n₁ = 40 →
  n₂ = 30 →
  avg₂ = 40 →
  avg_total = 34.285714285714285 →
  (n₁ * (n₁ + n₂) * avg_total - n₂ * avg₂ * (n₁ + n₂)) / (n₁ * (n₁ + n₂)) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_average_of_first_group_l2636_263647


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2636_263621

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - 3| < a) → a > 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2636_263621


namespace NUMINAMATH_CALUDE_probability_ratio_l2636_263640

def total_cards : ℕ := 50
def numbers_range : ℕ := 10
def cards_per_number : ℕ := 5
def cards_drawn : ℕ := 5

def probability_all_same (total : ℕ) (range : ℕ) (per_num : ℕ) (drawn : ℕ) : ℚ :=
  (range : ℚ) / (total.choose drawn)

def probability_four_and_one (total : ℕ) (range : ℕ) (per_num : ℕ) (drawn : ℕ) : ℚ :=
  ((range * (range - 1)) * (per_num.choose (drawn - 1)) * (per_num.choose 1) : ℚ) / (total.choose drawn)

theorem probability_ratio :
  (probability_four_and_one total_cards numbers_range cards_per_number cards_drawn) /
  (probability_all_same total_cards numbers_range cards_per_number cards_drawn) = 225 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l2636_263640


namespace NUMINAMATH_CALUDE_apple_percentage_after_removal_l2636_263603

def initial_apples : ℕ := 10
def initial_oranges : ℕ := 23
def oranges_removed : ℕ := 13

def remaining_oranges : ℕ := initial_oranges - oranges_removed
def total_fruit_after : ℕ := initial_apples + remaining_oranges

theorem apple_percentage_after_removal : 
  (initial_apples : ℚ) / total_fruit_after * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_apple_percentage_after_removal_l2636_263603


namespace NUMINAMATH_CALUDE_line_slope_l2636_263685

/-- The slope of the line given by the equation x/4 + y/5 = 1 is -5/4 -/
theorem line_slope (x y : ℝ) : 
  (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l2636_263685


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l2636_263664

def sum_of_digits (n : ℕ) : ℕ := n % 10 + n / 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_fourth_number :
  let known_numbers := [34, 56, 45]
  let sum_known := known_numbers.sum
  let sum_digits_known := (known_numbers.map sum_of_digits).sum
  ∃ x : ℕ,
    is_two_digit x ∧
    (∀ y : ℕ, is_two_digit y →
      sum_digits_known + sum_of_digits x + sum_digits_known + sum_of_digits y = (sum_known + x + sum_known + y) / 3
      → x ≤ y) ∧
    x = 35 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l2636_263664


namespace NUMINAMATH_CALUDE_average_salary_proof_l2636_263630

def workshop_problem (total_workers : ℕ) (technicians : ℕ) (avg_salary_technicians : ℚ) (avg_salary_others : ℚ) : Prop :=
  let non_technicians : ℕ := total_workers - technicians
  let total_salary_technicians : ℚ := technicians * avg_salary_technicians
  let total_salary_others : ℚ := non_technicians * avg_salary_others
  let total_salary : ℚ := total_salary_technicians + total_salary_others
  let avg_salary_all : ℚ := total_salary / total_workers
  avg_salary_all = 8000

theorem average_salary_proof :
  workshop_problem 28 7 14000 6000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_proof_l2636_263630


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l2636_263681

def f (x a : ℝ) := |2*x - 1| + |x - a|

theorem solution_part1 : 
  {x : ℝ | f x 3 ≤ 4} = Set.Icc 0 2 := by sorry

theorem solution_part2 (a : ℝ) :
  (∀ x, f x a = |x - 1 + a|) → 
  (a < 1/2 → {x : ℝ | f x a = |x - 1 + a|} = Set.Icc a (1/2)) ∧
  (a = 1/2 → {x : ℝ | f x a = |x - 1 + a|} = {1/2}) ∧
  (a > 1/2 → {x : ℝ | f x a = |x - 1 + a|} = Set.Icc (1/2) a) := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l2636_263681


namespace NUMINAMATH_CALUDE_factor_210_into_four_l2636_263641

def prime_factors : Multiset ℕ := {2, 3, 5, 7}

/-- The number of ways to partition a multiset of 4 distinct elements into 4 non-empty subsets -/
def partition_count (m : Multiset ℕ) : ℕ := sorry

theorem factor_210_into_four : partition_count prime_factors = 15 := by sorry

end NUMINAMATH_CALUDE_factor_210_into_four_l2636_263641


namespace NUMINAMATH_CALUDE_angle_a_is_sixty_degrees_l2636_263677

/-- In a triangle ABC, if the sum of angles B and C is twice angle A, then angle A is 60 degrees. -/
theorem angle_a_is_sixty_degrees (A B C : ℝ) (h1 : A + B + C = 180) (h2 : B + C = 2 * A) : A = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_a_is_sixty_degrees_l2636_263677


namespace NUMINAMATH_CALUDE_homework_decrease_iff_thirty_percent_l2636_263682

/-- Represents the decrease in homework duration over two reforms -/
def homework_decrease (a : ℝ) (x : ℝ) : Prop :=
  a * (1 - x)^2 = 0.3 * a

/-- Theorem stating that the homework decrease equation holds if and only if
    the final duration is 30% of the initial duration -/
theorem homework_decrease_iff_thirty_percent (a : ℝ) (x : ℝ) (h_a : a > 0) :
  homework_decrease a x ↔ a * (1 - x)^2 = 0.3 * a :=
sorry

end NUMINAMATH_CALUDE_homework_decrease_iff_thirty_percent_l2636_263682


namespace NUMINAMATH_CALUDE_equation_solution_l2636_263675

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  -2 * x^2 = (4 * x + 2) / (x - 2) ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2636_263675


namespace NUMINAMATH_CALUDE_karlsson_candies_l2636_263648

/-- The number of ones initially written on the board -/
def initial_ones : ℕ := 28

/-- The number of minutes the process continues -/
def total_minutes : ℕ := 28

/-- The number of edges in a complete graph with n vertices -/
def complete_graph_edges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The maximum number of candies Karlsson could eat -/
def max_candies : ℕ := complete_graph_edges initial_ones

theorem karlsson_candies :
  max_candies = 378 :=
sorry

end NUMINAMATH_CALUDE_karlsson_candies_l2636_263648


namespace NUMINAMATH_CALUDE_mass_percentage_not_sufficient_for_unique_compound_l2636_263670

/-- Represents a chemical compound -/
structure Compound where
  name : String
  mass_percentage_O : Float

/-- The mass percentage of O in the compound -/
def given_mass_percentage : Float := 36.36

/-- Theorem stating that the given mass percentage of O is not sufficient to uniquely determine a compound -/
theorem mass_percentage_not_sufficient_for_unique_compound :
  ∃ (c1 c2 : Compound), c1.mass_percentage_O = given_mass_percentage ∧ 
                        c2.mass_percentage_O = given_mass_percentage ∧ 
                        c1.name ≠ c2.name :=
sorry

end NUMINAMATH_CALUDE_mass_percentage_not_sufficient_for_unique_compound_l2636_263670


namespace NUMINAMATH_CALUDE_equation_solution_l2636_263662

theorem equation_solution : 
  ∀ x : ℝ, (2 / ((x - 1) * (x - 2)) + 2 / ((x - 2) * (x - 3)) + 2 / ((x - 3) * (x - 4)) = 1 / 3) ↔ 
  (x = 8 ∨ x = -5/2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2636_263662


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l2636_263657

/-- Represents a conic section --/
inductive ConicSection
| Parabola
| Circle
| Ellipse
| Hyperbola
| Point
| Line
| TwoLines
| Empty

/-- Determines if the given equation represents an ellipse --/
def is_ellipse (a b h k : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≠ b

/-- The equation of the conic section --/
def conic_equation (x y : ℝ) : Prop :=
  x^2 + 6*x + 9*y^2 - 36 = 0

/-- Theorem stating that the given equation represents an ellipse --/
theorem conic_is_ellipse : 
  ∃ (a b h k : ℝ), 
    (∀ (x y : ℝ), conic_equation x y ↔ ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)) ∧
    is_ellipse a b h k :=
sorry


end NUMINAMATH_CALUDE_conic_is_ellipse_l2636_263657


namespace NUMINAMATH_CALUDE_rose_group_size_l2636_263629

theorem rose_group_size (n : ℕ+) (h : Nat.lcm n 19 = 171) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_rose_group_size_l2636_263629


namespace NUMINAMATH_CALUDE_expression_simplification_l2636_263658

theorem expression_simplification (x y : ℚ) (hx : x = 1/9) (hy : y = 5) :
  -1/5 * x * y^2 - 3 * x^2 * y + x * y^2 + 2 * x^2 * y + 3 * x * y^2 + x^2 * y - 2 * x * y^2 = 20/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2636_263658


namespace NUMINAMATH_CALUDE_perfect_square_proof_l2636_263654

theorem perfect_square_proof (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_odd_b : Odd b) (h_int : ∃ k : ℤ, (a + b)^2 + 4*a = k * a * b) : 
  ∃ u : ℕ, a = u^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_proof_l2636_263654


namespace NUMINAMATH_CALUDE_isosceles_triangle_same_color_l2636_263644

-- Define a circle
def Circle : Type := Unit

-- Define a color type
inductive Color
| C1
| C2

-- Define a point on the circle
structure Point (c : Circle) where
  color : Color

-- Define an isosceles triangle
structure IsoscelesTriangle (c : Circle) where
  p1 : Point c
  p2 : Point c
  p3 : Point c
  isIsosceles : True  -- We assume this property without proving it

-- State the theorem
theorem isosceles_triangle_same_color (c : Circle) 
  (coloring : Point c → Color) :
  ∃ (t : IsoscelesTriangle c), 
    t.p1.color = t.p2.color ∧ 
    t.p2.color = t.p3.color :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_same_color_l2636_263644


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2636_263695

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 5 * (b + c))
  (second_eq : b = 9 * c) :
  a * b * c = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2636_263695


namespace NUMINAMATH_CALUDE_M_inter_N_eq_M_l2636_263693

def M : Set ℝ := {x | x^2 - x < 0}
def N : Set ℝ := {x | |x| < 2}

theorem M_inter_N_eq_M : M ∩ N = M := by
  sorry

end NUMINAMATH_CALUDE_M_inter_N_eq_M_l2636_263693


namespace NUMINAMATH_CALUDE_sum_of_specific_geometric_series_l2636_263605

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_specific_geometric_series :
  geometric_series_sum (1/4) (1/2) 7 = 127/256 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_geometric_series_l2636_263605


namespace NUMINAMATH_CALUDE_parabola_directrix_l2636_263623

/-- The equation of the directrix of the parabola y^2 = 2x is x = -1/2 -/
theorem parabola_directrix : ∀ x y : ℝ, y^2 = 2*x → (∃ p : ℝ, p > 0 ∧ x = -p/2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2636_263623


namespace NUMINAMATH_CALUDE_inequality_proof_l2636_263684

theorem inequality_proof (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum : x + 2*y + 3*z = 12) : x^2 + 2*y^3 + 3*z^2 > 24 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2636_263684


namespace NUMINAMATH_CALUDE_not_power_of_two_l2636_263646

theorem not_power_of_two (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ¬ ∃ k : ℕ, (36 * a + b) * (a + 36 * b) = 2^k :=
by sorry

end NUMINAMATH_CALUDE_not_power_of_two_l2636_263646


namespace NUMINAMATH_CALUDE_min_sum_of_arithmetic_sequence_l2636_263607

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference

/-- Sum of the first n terms of an arithmetic sequence -/
def sumOfTerms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem min_sum_of_arithmetic_sequence (seq : ArithmeticSequence) :
  seq.a 1 = -7 →
  sumOfTerms seq 3 = -15 →
  ∀ n : ℕ, sumOfTerms seq n ≥ -16 ∧ 
  (∃ m : ℕ, sumOfTerms seq m = -16) := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_arithmetic_sequence_l2636_263607


namespace NUMINAMATH_CALUDE_only_origin_satisfies_l2636_263669

def satisfies_inequality (x y : ℝ) : Prop := x + y - 1 < 0

theorem only_origin_satisfies : 
  satisfies_inequality 0 0 ∧ 
  ¬satisfies_inequality 2 4 ∧ 
  ¬satisfies_inequality (-1) 4 ∧ 
  ¬satisfies_inequality 1 8 :=
by sorry

end NUMINAMATH_CALUDE_only_origin_satisfies_l2636_263669


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2636_263614

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, -x^2 + 2*x + 3 ≥ 0 ↔ x ∈ Set.Icc (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2636_263614


namespace NUMINAMATH_CALUDE_possible_sums_B_l2636_263608

theorem possible_sums_B (a b c d : ℕ+) 
  (h1 : a * b = 2 * (c + d))
  (h2 : c * d = 2 * (a + b))
  (h3 : a + b ≥ c + d) :
  c + d = 13 ∨ c + d = 10 ∨ c + d = 9 ∨ c + d = 8 := by
sorry

end NUMINAMATH_CALUDE_possible_sums_B_l2636_263608


namespace NUMINAMATH_CALUDE_points_in_small_square_l2636_263612

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square region in a 2D plane -/
structure Square where
  center : Point
  side_length : ℝ

/-- Check if a point is inside a square -/
def is_point_in_square (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) ≤ s.side_length / 2 ∧
  abs (p.y - s.center.y) ≤ s.side_length / 2

/-- The main theorem -/
theorem points_in_small_square (points : Finset Point) 
    (h1 : points.card = 51)
    (h2 : ∀ p ∈ points, is_point_in_square p ⟨⟨0.5, 0.5⟩, 1⟩) :
    ∃ (small_square : Square),
      small_square.side_length = 0.2 ∧
      ∃ (p1 p2 p3 : Point),
        p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
        p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
        is_point_in_square p1 small_square ∧
        is_point_in_square p2 small_square ∧
        is_point_in_square p3 small_square :=
  sorry

end NUMINAMATH_CALUDE_points_in_small_square_l2636_263612


namespace NUMINAMATH_CALUDE_mikaela_savings_l2636_263606

/-- Calculates the total savings for Mikaela over two months of tutoring --/
def total_savings (
  hourly_rate_month1 : ℚ)
  (hours_month1 : ℚ)
  (hourly_rate_month2 : ℚ)
  (additional_hours_month2 : ℚ)
  (spending_ratio_month1 : ℚ)
  (spending_ratio_month2 : ℚ) : ℚ :=
  let earnings_month1 := hourly_rate_month1 * hours_month1
  let savings_month1 := earnings_month1 * (1 - spending_ratio_month1)
  let hours_month2 := hours_month1 + additional_hours_month2
  let earnings_month2 := hourly_rate_month2 * hours_month2
  let savings_month2 := earnings_month2 * (1 - spending_ratio_month2)
  savings_month1 + savings_month2

/-- Proves that Mikaela's total savings from both months is $190 --/
theorem mikaela_savings :
  total_savings 10 35 12 5 (4/5) (3/4) = 190 := by
  sorry

end NUMINAMATH_CALUDE_mikaela_savings_l2636_263606


namespace NUMINAMATH_CALUDE_product_65_35_l2636_263667

theorem product_65_35 : 65 * 35 = 2275 := by
  sorry

end NUMINAMATH_CALUDE_product_65_35_l2636_263667


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2636_263615

/-- An arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

/-- The sequence is increasing -/
def IncreasingSequence (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b →
  IncreasingSequence b →
  b 4 * b 5 = 30 →
  (b 3 * b 6 = -1652 ∨ b 3 * b 6 = -308 ∨ b 3 * b 6 = -68 ∨ b 3 * b 6 = 28) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2636_263615


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l2636_263609

theorem polynomial_uniqueness (Q : ℝ → ℝ) :
  (∀ x, Q x = Q 0 + Q 1 * x + Q 2 * x^2) →
  Q (-1) = 3 →
  Q 3 = 15 →
  ∀ x, Q x = -2 * x^2 + 6 * x - 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l2636_263609


namespace NUMINAMATH_CALUDE_value_equals_scientific_notation_l2636_263634

/-- Represents the value in billion yuan -/
def value : ℝ := 24953

/-- Represents the scientific notation coefficient -/
def coefficient : ℝ := 2.4953

/-- Represents the scientific notation exponent -/
def exponent : ℕ := 13

/-- Theorem stating that the given value in billion yuan is equal to its scientific notation representation -/
theorem value_equals_scientific_notation : value * 10^9 = coefficient * 10^exponent := by
  sorry

end NUMINAMATH_CALUDE_value_equals_scientific_notation_l2636_263634


namespace NUMINAMATH_CALUDE_tin_can_equation_l2636_263624

/-- Represents the number of can bodies that can be made from one sheet of tinplate -/
def bodies_per_sheet : ℕ := 15

/-- Represents the number of can bottoms that can be made from one sheet of tinplate -/
def bottoms_per_sheet : ℕ := 42

/-- Represents the total number of available sheets of tinplate -/
def total_sheets : ℕ := 108

/-- Represents the number of can bottoms needed for one complete tin can -/
def bottoms_per_can : ℕ := 2

theorem tin_can_equation (x : ℕ) :
  x ≤ total_sheets →
  (bottoms_per_can * bodies_per_sheet * x = bottoms_per_sheet * (total_sheets - x)) ↔
  (2 * 15 * x = 42 * (108 - x)) :=
by sorry

end NUMINAMATH_CALUDE_tin_can_equation_l2636_263624


namespace NUMINAMATH_CALUDE_eighth_roll_last_probability_l2636_263631

/-- The probability of the 8th roll being the last roll when rolling a standard six-sided die 
    until getting the same number on consecutive rolls -/
def prob_eighth_roll_last : ℚ := (5^6 : ℚ) / (6^7 : ℚ)

/-- The number of sides on a standard die -/
def standard_die_sides : ℕ := 6

/-- Theorem stating that the probability of the 8th roll being the last roll is correct -/
theorem eighth_roll_last_probability : 
  prob_eighth_roll_last = (5^6 : ℚ) / (6^7 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_eighth_roll_last_probability_l2636_263631


namespace NUMINAMATH_CALUDE_rug_on_floor_l2636_263628

theorem rug_on_floor (rug_length : ℝ) (rug_width : ℝ) (floor_area : ℝ) : 
  rug_length = 2 →
  rug_width = 7 →
  floor_area = 64 →
  rug_length * rug_width ≤ floor_area →
  (floor_area - rug_length * rug_width) / floor_area = 25 / 32 := by
  sorry

end NUMINAMATH_CALUDE_rug_on_floor_l2636_263628


namespace NUMINAMATH_CALUDE_largest_sum_of_digits_l2636_263683

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the decimal 0.abc -/
def DecimalABC (a b c : Digit) : ℚ :=
  (a.val * 100 + b.val * 10 + c.val : ℕ) / 1000

theorem largest_sum_of_digits (a b c : Digit) (y : ℕ) 
  (h1 : DecimalABC a b c = 1 / y)
  (h2 : 0 < y) (h3 : y ≤ 16) :
  a.val + b.val + c.val ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_largest_sum_of_digits_l2636_263683


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2636_263672

theorem gcd_of_three_numbers :
  Nat.gcd 188094 (Nat.gcd 244122 395646) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2636_263672


namespace NUMINAMATH_CALUDE_largest_divisible_n_l2636_263619

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > n → ¬((m + 12) ∣ (m^3 + 144))) ∧ 
  ((n + 12) ∣ (n^3 + 144)) ∧ 
  n = 84 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l2636_263619


namespace NUMINAMATH_CALUDE_ten_passengers_five_stops_l2636_263633

/-- The number of ways for passengers to get off a bus -/
def bus_stop_combinations (num_passengers : ℕ) (num_stops : ℕ) : ℕ :=
  num_stops ^ num_passengers

/-- Theorem: 10 passengers and 5 stops result in 5^10 combinations -/
theorem ten_passengers_five_stops :
  bus_stop_combinations 10 5 = 5^10 := by
  sorry

end NUMINAMATH_CALUDE_ten_passengers_five_stops_l2636_263633


namespace NUMINAMATH_CALUDE_cos_2_sum_of_tan_roots_l2636_263642

theorem cos_2_sum_of_tan_roots (α β : ℝ) : 
  (∃ x y : ℝ, x^2 + 5*x - 6 = 0 ∧ y^2 + 5*y - 6 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  Real.cos (2 * (α + β)) = 12 / 37 := by
sorry

end NUMINAMATH_CALUDE_cos_2_sum_of_tan_roots_l2636_263642


namespace NUMINAMATH_CALUDE_intersection_M_N_l2636_263666

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2636_263666


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2636_263653

theorem algebraic_expression_value (a b : ℝ) (h : a + b - 2 = 0) :
  a^2 - b^2 + 4*b = 4 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2636_263653


namespace NUMINAMATH_CALUDE_investment_problem_l2636_263692

/-- Given two investment projects with specific conditions, 
    prove the minimum distance between them and the profitability of the deal. -/
theorem investment_problem 
  (p₁ x₁ p₂ x₂ : ℝ) 
  (h₁ : 4 * x₁ - 3 * p₁ - 44 = 0) 
  (h₂ : p₂^2 - 12 * p₂ + x₂^2 - 8 * x₂ + 43 = 0) 
  (h₃ : p₁ > 0) 
  (h₄ : p₂ > 0) : 
  let d := Real.sqrt ((x₁ - x₂)^2 + (p₁ - p₂)^2)
  ∃ (min_d : ℝ), 
    (∀ p₁' x₁' p₂' x₂', 
      4 * x₁' - 3 * p₁' - 44 = 0 → 
      p₂'^2 - 12 * p₂' + x₂'^2 - 8 * x₂' + 43 = 0 → 
      p₁' > 0 → 
      p₂' > 0 → 
      Real.sqrt ((x₁' - x₂')^2 + (p₁' - p₂')^2) ≥ min_d) ∧ 
    d = min_d ∧ 
    min_d = 6.2 ∧ 
    x₁ + x₂ - p₁ - p₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l2636_263692


namespace NUMINAMATH_CALUDE_m_range_for_p_necessary_not_sufficient_for_q_l2636_263604

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 > 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 > 0

-- Define the sets A and B
def A : Set ℝ := {x | p x}
def B (m : ℝ) : Set ℝ := {x | q x m}

-- Theorem statement
theorem m_range_for_p_necessary_not_sufficient_for_q :
  ∀ m : ℝ, (m > 0 ∧ (∀ x : ℝ, q x m → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x m)) ↔ m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_m_range_for_p_necessary_not_sufficient_for_q_l2636_263604


namespace NUMINAMATH_CALUDE_cos_difference_value_l2636_263668

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_value_l2636_263668


namespace NUMINAMATH_CALUDE_S_min_at_5_l2636_263673

/-- An arithmetic sequence with first term -9 and S_3 = S_7 -/
def ArithSeq : ℕ → ℤ := fun n => 2*n - 11

/-- Sum of first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := n * (ArithSeq 1 + ArithSeq n) / 2

/-- The condition that S_3 = S_7 -/
axiom S3_eq_S7 : S 3 = S 7

/-- The theorem stating that S_n is minimized when n = 5 -/
theorem S_min_at_5 : ∀ n : ℕ, n ≠ 0 → S 5 ≤ S n :=
sorry

end NUMINAMATH_CALUDE_S_min_at_5_l2636_263673


namespace NUMINAMATH_CALUDE_marcos_boat_distance_l2636_263622

/-- The distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proof that given a speed of 30 mph and a time of 10 minutes, the distance traveled is 5 miles -/
theorem marcos_boat_distance :
  let speed : ℝ := 30  -- Speed in miles per hour
  let time : ℝ := 10 / 60  -- Time in hours (10 minutes converted to hours)
  distance speed time = 5 := by
  sorry

end NUMINAMATH_CALUDE_marcos_boat_distance_l2636_263622


namespace NUMINAMATH_CALUDE_curve_tangent_problem_l2636_263680

theorem curve_tangent_problem (a b : ℝ) : 
  (2 * a + b / 2 = -5) →  -- Curve passes through (2, -5)
  (4 * a - b / 4 = -7/2) →  -- Tangent slope at (2, -5) is -7/2
  a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_curve_tangent_problem_l2636_263680


namespace NUMINAMATH_CALUDE_average_speed_is_27_point_5_l2636_263635

-- Define the initial and final odometer readings
def initial_reading : ℕ := 1551
def final_reading : ℕ := 1881

-- Define the total riding time in hours
def total_time : ℕ := 12

-- Define the average speed
def average_speed : ℚ := (final_reading - initial_reading : ℚ) / total_time

-- Theorem statement
theorem average_speed_is_27_point_5 : average_speed = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_is_27_point_5_l2636_263635


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2636_263620

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 4 = 12) : 
  a 1 + a 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2636_263620


namespace NUMINAMATH_CALUDE_equation_solution_and_condition_l2636_263618

theorem equation_solution_and_condition :
  ∃ x : ℝ, (3 * x + 7 = 22) ∧ (2 * x + 1 ≠ 9) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_and_condition_l2636_263618


namespace NUMINAMATH_CALUDE_ellipse_equation_l2636_263665

/-- Given an ellipse with equation x²/a² + 25y²/(9a²) = 1, prove that the equation
    of the ellipse is x² + 25/9 * y² = 1 under the following conditions:
    - Points A and B are on the ellipse
    - F₂ is the right focus of the ellipse
    - |AF₂| + |BF₂| = 8/5 * a
    - Distance from midpoint of AB to left directrix is 3/2 -/
theorem ellipse_equation (a : ℝ) (A B F₂ : ℝ × ℝ) :
  (∀ x y, x^2/a^2 + 25*y^2/(9*a^2) = 1 → (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) →
  (F₂.1 > 0) →
  (Real.sqrt ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) + Real.sqrt ((B.1 - F₂.1)^2 + (B.2 - F₂.2)^2) = 8/5 * a) →
  (((A.1 + B.1)/2 + 5/4*a) = 3/2) →
  (∀ x y, x^2 + 25/9 * y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2636_263665


namespace NUMINAMATH_CALUDE_intersection_points_form_line_l2636_263691

theorem intersection_points_form_line (s : ℝ) :
  let x := s + 15
  let y := 2*s - 8
  (2*x + 3*y = 8*s + 6) ∧ (x + 2*y = 5*s - 1) →
  y = 2*x - 38 := by
sorry

end NUMINAMATH_CALUDE_intersection_points_form_line_l2636_263691


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2636_263627

theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, 1) →
  b = (-1, k) →
  (∃ (t : ℝ), t ≠ 0 ∧ a.1 * t = b.1 ∧ a.2 * t = b.2) →
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2636_263627


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2636_263616

def A : Set ℤ := {x | ∃ k, x = 2*k + 1}
def B : Set ℤ := {x | ∃ k, x = 2*k}

theorem negation_of_universal_proposition :
  (¬ (∀ x ∈ A, (2 * x) ∈ B)) ↔ (∃ x ∈ A, (2 * x) ∉ B) :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2636_263616


namespace NUMINAMATH_CALUDE_statements_proof_l2636_263690

theorem statements_proof :
  (∀ a b c : ℝ, a > b → c < 0 → a^3 * c < b^3 * c) ∧
  (∀ a b c : ℝ, c > a → a > b → b > 0 → a / (c - a) > b / (c - b)) ∧
  (∀ a b : ℝ, a > b → (1 : ℝ) / a > (1 : ℝ) / b → a > 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_statements_proof_l2636_263690


namespace NUMINAMATH_CALUDE_angle_measure_l2636_263661

-- Define the angle
def angle : ℝ := sorry

-- Define the complement of the angle
def complement : ℝ := 90 - angle

-- Define the supplement of the angle
def supplement : ℝ := 180 - angle

-- State the theorem
theorem angle_measure : 
  supplement = 4 * complement + 15 → angle = 65 := by sorry

end NUMINAMATH_CALUDE_angle_measure_l2636_263661


namespace NUMINAMATH_CALUDE_sandals_sold_example_l2636_263649

/-- Given a ratio of shoes to sandals and the number of shoes sold, 
    calculate the number of sandals sold. -/
def sandals_sold (shoe_ratio : ℕ) (sandal_ratio : ℕ) (shoes : ℕ) : ℕ :=
  (shoes / shoe_ratio) * sandal_ratio

/-- Theorem stating that given the specific ratio and number of shoes sold,
    the number of sandals sold is 40. -/
theorem sandals_sold_example : sandals_sold 9 5 72 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sandals_sold_example_l2636_263649


namespace NUMINAMATH_CALUDE_graph_single_point_implies_d_value_l2636_263688

/-- The equation of the graph -/
def graph_equation (x y : ℝ) (d : ℝ) : Prop :=
  x^2 + 3*y^2 + 6*x - 18*y + d = 0

/-- The graph consists of a single point -/
def single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, graph_equation p.1 p.2 d

/-- If the graph of x^2 + 3y^2 + 6x - 18y + d = 0 consists of a single point, then d = -27 -/
theorem graph_single_point_implies_d_value :
  ∀ d : ℝ, single_point d → d = -27 := by
  sorry

end NUMINAMATH_CALUDE_graph_single_point_implies_d_value_l2636_263688


namespace NUMINAMATH_CALUDE_probability_three_heads_in_seven_tosses_l2636_263687

def coin_tosses : ℕ := 7
def heads_count : ℕ := 3

theorem probability_three_heads_in_seven_tosses :
  (Nat.choose coin_tosses heads_count) / (2 ^ coin_tosses) = 35 / 128 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_heads_in_seven_tosses_l2636_263687


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2636_263676

/-- A quadratic function y = (x + m - 3)(x - m) + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := (x + m - 3) * (x - m) + 3

theorem quadratic_function_property (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ < x₂ →
  f m x₁ = y₁ →
  f m x₂ = y₂ →
  x₁ + x₂ < 3 →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2636_263676


namespace NUMINAMATH_CALUDE_correct_observation_value_l2636_263645

theorem correct_observation_value (n : ℕ) (initial_mean corrected_mean wrong_value : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 40)
  (h3 : corrected_mean = 40.66)
  (h4 : wrong_value = 15) :
  let total_sum := n * initial_mean
  let corrected_sum := n * corrected_mean
  let difference := corrected_sum - total_sum
  let actual_value := wrong_value + difference
  actual_value = 48 := by sorry

end NUMINAMATH_CALUDE_correct_observation_value_l2636_263645


namespace NUMINAMATH_CALUDE_firm_employs_80_looms_l2636_263678

/-- Represents a textile manufacturing firm with looms -/
structure TextileFirm where
  totalSales : ℕ
  manufacturingExpenses : ℕ
  establishmentCharges : ℕ
  profitDecreaseOnBreakdown : ℕ

/-- Calculates the number of looms employed by the firm -/
def calculateLooms (firm : TextileFirm) : ℕ :=
  (firm.totalSales - firm.manufacturingExpenses) / firm.profitDecreaseOnBreakdown

/-- Theorem stating that the firm employs 80 looms -/
theorem firm_employs_80_looms (firm : TextileFirm) 
  (h1 : firm.totalSales = 500000)
  (h2 : firm.manufacturingExpenses = 150000)
  (h3 : firm.establishmentCharges = 75000)
  (h4 : firm.profitDecreaseOnBreakdown = 4375) :
  calculateLooms firm = 80 := by
  sorry

#eval calculateLooms { totalSales := 500000, 
                       manufacturingExpenses := 150000, 
                       establishmentCharges := 75000, 
                       profitDecreaseOnBreakdown := 4375 }

end NUMINAMATH_CALUDE_firm_employs_80_looms_l2636_263678


namespace NUMINAMATH_CALUDE_set_operations_l2636_263610

def A : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -2 ∨ x > 4}

theorem set_operations (x : ℝ) :
  (x ∈ A ∩ B ↔ -5 ≤ x ∧ x < -2) ∧
  (x ∈ (Set.univ \ A) ∩ B ↔ x < -5 ∨ x > 4) ∧
  (x ∈ (Set.univ \ A) ∩ (Set.univ \ B) ↔ 3 < x ∧ x ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2636_263610


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2636_263638

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def P : Set Nat := {1, 2, 3, 4}
def Q : Set Nat := {3, 4, 5}

theorem intersection_complement_theorem :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2636_263638


namespace NUMINAMATH_CALUDE_square_difference_equality_l2636_263679

theorem square_difference_equality : 1007^2 - 993^2 - 1005^2 + 995^2 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2636_263679


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2636_263613

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (ca_count : ℕ) (o_count : ℕ) (h_count : ℕ) 
                     (ca_weight : ℝ) (o_weight : ℝ) (h_weight : ℝ) : ℝ :=
  ca_count * ca_weight + o_count * o_weight + h_count * h_weight

/-- Theorem stating that the molecular weight of the given compound is 74.094 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight 1 2 2 40.08 15.999 1.008 = 74.094 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2636_263613


namespace NUMINAMATH_CALUDE_unique_square_of_divisors_l2636_263663

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- n is a positive integer that equals the square of its number of positive divisors -/
def is_square_of_divisors (n : ℕ) : Prop :=
  n > 0 ∧ n = (num_divisors n) ^ 2

theorem unique_square_of_divisors :
  ∃! n : ℕ, is_square_of_divisors n ∧ n = 9 := by sorry

end NUMINAMATH_CALUDE_unique_square_of_divisors_l2636_263663


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2636_263617

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 5 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 5 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2636_263617


namespace NUMINAMATH_CALUDE_noah_garden_larger_by_75_l2636_263611

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ := d.length * d.width

theorem noah_garden_larger_by_75 (liam_garden noah_garden : GardenDimensions) 
  (h1 : liam_garden.length = 30 ∧ liam_garden.width = 50)
  (h2 : noah_garden.length = 35 ∧ noah_garden.width = 45) : 
  gardenArea noah_garden - gardenArea liam_garden = 75 := by
  sorry

#check noah_garden_larger_by_75

end NUMINAMATH_CALUDE_noah_garden_larger_by_75_l2636_263611


namespace NUMINAMATH_CALUDE_quadratic_solution_implication_l2636_263632

theorem quadratic_solution_implication (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0 → 4 * a + 8 * b = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implication_l2636_263632


namespace NUMINAMATH_CALUDE_rational_fraction_value_l2636_263660

theorem rational_fraction_value (x y : ℝ) : 
  (x - y) / (x + y) = 4 → 
  ∃ (q : ℚ), x / y = ↑q →
  x / y = -5/3 := by
sorry

end NUMINAMATH_CALUDE_rational_fraction_value_l2636_263660


namespace NUMINAMATH_CALUDE_rectangle_area_l2636_263671

theorem rectangle_area (width : ℝ) (h1 : width > 0) : 
  let length := 2 * width
  let diagonal := 10
  width ^ 2 + length ^ 2 = diagonal ^ 2 →
  width * length = 40 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2636_263671


namespace NUMINAMATH_CALUDE_matrix_vector_computation_l2636_263689

variable {m n : ℕ}
variable (N : Matrix (Fin 2) (Fin n) ℝ)
variable (a b : Fin n → ℝ)

theorem matrix_vector_computation 
  (ha : N.mulVec a = ![2, -3])
  (hb : N.mulVec b = ![5, 4]) :
  N.mulVec (3 • a - 2 • b) = ![-4, -17] := by sorry

end NUMINAMATH_CALUDE_matrix_vector_computation_l2636_263689


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2636_263694

theorem perfect_square_condition (m n : ℕ) (hm : m > 1) (hn : n > 1) :
  (∃ k : ℕ, 2^m + 3^n = k^2) ↔ 
  (∃ a b : ℕ, m = 2*a ∧ n = 2*b ∧ a ≥ 1 ∧ b ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2636_263694
