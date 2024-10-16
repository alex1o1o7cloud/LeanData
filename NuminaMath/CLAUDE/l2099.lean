import Mathlib

namespace NUMINAMATH_CALUDE_storage_unit_capacity_l2099_209942

/-- A storage unit with three shelves for storing CDs. -/
structure StorageUnit where
  shelf1_racks : ℕ
  shelf1_cds_per_rack : ℕ
  shelf2_racks : ℕ
  shelf2_cds_per_rack : ℕ
  shelf3_racks : ℕ
  shelf3_cds_per_rack : ℕ

/-- Calculate the total number of CDs that can fit in a storage unit. -/
def totalCDs (unit : StorageUnit) : ℕ :=
  unit.shelf1_racks * unit.shelf1_cds_per_rack +
  unit.shelf2_racks * unit.shelf2_cds_per_rack +
  unit.shelf3_racks * unit.shelf3_cds_per_rack

/-- Theorem stating that the specific storage unit can hold 116 CDs. -/
theorem storage_unit_capacity :
  let unit : StorageUnit := {
    shelf1_racks := 5,
    shelf1_cds_per_rack := 8,
    shelf2_racks := 4,
    shelf2_cds_per_rack := 10,
    shelf3_racks := 3,
    shelf3_cds_per_rack := 12
  }
  totalCDs unit = 116 := by
  sorry

end NUMINAMATH_CALUDE_storage_unit_capacity_l2099_209942


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l2099_209983

theorem price_reduction_percentage (P : ℝ) (S : ℝ) (h1 : P > 0) (h2 : S > 0) :
  let new_sales := 1.80 * S
  let new_revenue := 1.08 * (P * S)
  let new_price := new_revenue / new_sales
  (P - new_price) / P = 0.40 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l2099_209983


namespace NUMINAMATH_CALUDE_newspaper_cost_difference_l2099_209908

/-- Grant's yearly newspaper expenditure -/
def grant_yearly_cost : ℝ := 200

/-- Juanita's weekday newspaper cost -/
def juanita_weekday_cost : ℝ := 0.5

/-- Juanita's Sunday newspaper cost -/
def juanita_sunday_cost : ℝ := 2

/-- Number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Number of weekdays in a week -/
def weekdays_per_week : ℕ := 6

/-- Juanita's weekly newspaper cost -/
def juanita_weekly_cost : ℝ := juanita_weekday_cost * weekdays_per_week + juanita_sunday_cost

/-- Juanita's yearly newspaper cost -/
def juanita_yearly_cost : ℝ := juanita_weekly_cost * weeks_per_year

theorem newspaper_cost_difference : juanita_yearly_cost - grant_yearly_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_cost_difference_l2099_209908


namespace NUMINAMATH_CALUDE_fourth_person_height_l2099_209957

theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℕ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- heights in increasing order
  h₂ = h₁ + 2 →                 -- difference between 1st and 2nd
  h₃ = h₂ + 2 →                 -- difference between 2nd and 3rd
  h₄ = h₃ + 6 →                 -- difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 77  -- average height
  → h₄ = 83 := by
sorry

end NUMINAMATH_CALUDE_fourth_person_height_l2099_209957


namespace NUMINAMATH_CALUDE_cubes_with_le_four_neighbors_eq_144_l2099_209931

/-- Represents a parallelepiped constructed from unit cubes. -/
structure Parallelepiped where
  a : ℕ
  b : ℕ
  c : ℕ
  sides_gt_four : min a (min b c) > 4
  internal_cubes : (a - 2) * (b - 2) * (c - 2) = 836

/-- The number of cubes with no more than four neighbors in the parallelepiped. -/
def cubes_with_le_four_neighbors (p : Parallelepiped) : ℕ :=
  4 * (p.a - 2 + p.b - 2 + p.c - 2) + 8

/-- Theorem stating that the number of cubes with no more than four neighbors is 144. -/
theorem cubes_with_le_four_neighbors_eq_144 (p : Parallelepiped) :
  cubes_with_le_four_neighbors p = 144 := by
  sorry

end NUMINAMATH_CALUDE_cubes_with_le_four_neighbors_eq_144_l2099_209931


namespace NUMINAMATH_CALUDE_count_a_values_correct_l2099_209991

/-- The number of integer values of a for which (a-1)x^2 + 2x - a - 1 = 0 has integer roots for x -/
def count_a_values : ℕ := 5

/-- The equation has integer roots for x -/
def has_integer_roots (a : ℤ) : Prop :=
  ∃ x : ℤ, (a - 1) * x^2 + 2 * x - a - 1 = 0

/-- There are exactly 5 integer values of a for which the equation has integer roots -/
theorem count_a_values_correct :
  (∃ S : Finset ℤ, S.card = count_a_values ∧ 
    (∀ a : ℤ, a ∈ S ↔ has_integer_roots a)) ∧
  (∀ T : Finset ℤ, (∀ a : ℤ, a ∈ T ↔ has_integer_roots a) → T.card ≤ count_a_values) :=
sorry

end NUMINAMATH_CALUDE_count_a_values_correct_l2099_209991


namespace NUMINAMATH_CALUDE_professor_seating_count_l2099_209923

/-- The number of chairs in a row --/
def num_chairs : ℕ := 9

/-- The number of professors --/
def num_professors : ℕ := 3

/-- The number of students --/
def num_students : ℕ := 6

/-- Represents the possible seating arrangements for professors --/
def professor_seating_arrangements : ℕ := sorry

/-- Theorem stating the number of ways professors can choose their chairs --/
theorem professor_seating_count :
  professor_seating_arrangements = 238 :=
sorry

end NUMINAMATH_CALUDE_professor_seating_count_l2099_209923


namespace NUMINAMATH_CALUDE_root_product_l2099_209965

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  (lg x)^2 + (lg 2 + lg 3) * lg x + lg 2 * lg 3 = 0

-- State the theorem
theorem root_product (x₁ x₂ : ℝ) :
  equation x₁ ∧ equation x₂ ∧ x₁ ≠ x₂ → x₁ * x₂ = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_root_product_l2099_209965


namespace NUMINAMATH_CALUDE_tan_fraction_equality_l2099_209966

theorem tan_fraction_equality (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_fraction_equality_l2099_209966


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2099_209960

theorem repeating_decimal_subtraction : 
  ∃ (x y : ℚ), (∀ n : ℕ, (10 * x - x.floor) * 10^n % 10 = 4) ∧ 
               (∀ n : ℕ, (10 * y - y.floor) * 10^n % 10 = 6) ∧ 
               (x - y = -2/9) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2099_209960


namespace NUMINAMATH_CALUDE_inequality_proof_l2099_209987

theorem inequality_proof (a b c d : ℝ) (h : a > b ∧ b > c ∧ c > d) :
  1 / (a - b) + 1 / (b - c) + 1 / (c - d) ≥ 9 / (a - d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2099_209987


namespace NUMINAMATH_CALUDE_unique_b_solution_l2099_209970

theorem unique_b_solution (a b : ℕ) : 
  0 ≤ a → a < 2^2008 → 0 ≤ b → b < 8 → 
  (7 * (a + 2^2008 * b)) % 2^2011 = 1 → 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_b_solution_l2099_209970


namespace NUMINAMATH_CALUDE_complex_sum_and_product_l2099_209976

theorem complex_sum_and_product : ∃ (z₁ z₂ : ℂ),
  z₁ = 2 + 5*I ∧ z₂ = 3 - 7*I ∧ z₁ + z₂ = 5 - 2*I ∧ z₁ * z₂ = -29 + I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_and_product_l2099_209976


namespace NUMINAMATH_CALUDE_mollys_age_l2099_209932

theorem mollys_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / (molly_age : ℚ) = 4 / 3 →
  sandy_age + 6 = 30 →
  molly_age = 18 :=
by sorry

end NUMINAMATH_CALUDE_mollys_age_l2099_209932


namespace NUMINAMATH_CALUDE_concatenated_numbers_remainder_l2099_209961

-- Define a function to concatenate numbers from 1 to n
def concatenateNumbers (n : ℕ) : ℕ := sorry

-- Define a function to calculate the remainder when a number is divided by 9
def remainderMod9 (n : ℕ) : ℕ := n % 9

-- Theorem statement
theorem concatenated_numbers_remainder (n : ℕ) (h : n = 2001) :
  remainderMod9 (concatenateNumbers n) = 6 := by sorry

end NUMINAMATH_CALUDE_concatenated_numbers_remainder_l2099_209961


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l2099_209922

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {3, 4, 5}

theorem complement_of_A_wrt_U : (U \ A) = {1, 2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l2099_209922


namespace NUMINAMATH_CALUDE_sixth_term_value_l2099_209900

/-- Represents a geometric sequence --/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio

/-- Properties of the geometric sequence --/
def GeometricSequence.properties (seq : GeometricSequence) : Prop :=
  -- Sum of first four terms is 40
  seq.a * (1 + seq.r + seq.r^2 + seq.r^3) = 40 ∧
  -- Fifth term is 32
  seq.a * seq.r^4 = 32

/-- Sixth term of the geometric sequence --/
def GeometricSequence.sixthTerm (seq : GeometricSequence) : ℝ :=
  seq.a * seq.r^5

/-- Theorem stating that the sixth term is 1280/15 --/
theorem sixth_term_value (seq : GeometricSequence) 
  (h : seq.properties) : seq.sixthTerm = 1280/15 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l2099_209900


namespace NUMINAMATH_CALUDE_column_for_2023_l2099_209947

def column_sequence : Fin 8 → Char
  | 0 => 'B'
  | 1 => 'C'
  | 2 => 'D'
  | 3 => 'E'
  | 4 => 'D'
  | 5 => 'C'
  | 6 => 'B'
  | 7 => 'A'

def column_for_number (n : ℕ) : Char :=
  column_sequence ((n - 2) % 8)

theorem column_for_2023 : column_for_number 2023 = 'C' := by
  sorry

end NUMINAMATH_CALUDE_column_for_2023_l2099_209947


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2099_209913

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 square units. -/
theorem isosceles_right_triangle_area (h : ℝ) (a : ℝ) (A : ℝ) : 
  h = 6 * Real.sqrt 2 →  -- hypotenuse is 6√2
  h = a * Real.sqrt 2 →  -- relationship between hypotenuse and leg in isosceles right triangle
  A = (1/2) * a^2 →      -- area formula for right triangle
  A = 18 := by
    sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2099_209913


namespace NUMINAMATH_CALUDE_g_of_5_l2099_209992

def g (x : ℝ) : ℝ := 5 * x + 2

theorem g_of_5 : g 5 = 27 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l2099_209992


namespace NUMINAMATH_CALUDE_salt_mixture_problem_l2099_209977

/-- Proves that the amount of initial 20% salt solution is 30 ounces when mixed with 30 ounces of 60% salt solution to create a 40% salt solution. -/
theorem salt_mixture_problem (x : ℝ) :
  (0.20 * x + 0.60 * 30 = 0.40 * (x + 30)) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_salt_mixture_problem_l2099_209977


namespace NUMINAMATH_CALUDE_john_star_wars_spending_l2099_209997

/-- Calculates the total money spent on Star Wars toys --/
def total_spent (group_a_cost group_b_cost : ℝ) 
                (group_a_discount group_b_discount : ℝ) 
                (group_a_tax group_b_tax lightsaber_tax : ℝ) : ℝ :=
  let group_a_discounted := group_a_cost * (1 - group_a_discount)
  let group_b_discounted := group_b_cost * (1 - group_b_discount)
  let group_a_total := group_a_discounted * (1 + group_a_tax)
  let group_b_total := group_b_discounted * (1 + group_b_tax)
  let other_toys_total := group_a_total + group_b_total
  let lightsaber_cost := 2 * other_toys_total
  let lightsaber_total := lightsaber_cost * (1 + lightsaber_tax)
  other_toys_total + lightsaber_total

/-- The total amount John spent on Star Wars toys is $4008.312 --/
theorem john_star_wars_spending :
  total_spent 900 600 0.15 0.25 0.06 0.09 0.04 = 4008.312 := by
  sorry

end NUMINAMATH_CALUDE_john_star_wars_spending_l2099_209997


namespace NUMINAMATH_CALUDE_job_candidate_probability_l2099_209914

theorem job_candidate_probability 
  (p_excel : ℝ) 
  (p_day_shift : ℝ) 
  (h_excel : p_excel = 0.2) 
  (h_day_shift : p_day_shift = 0.7) : 
  p_excel * (1 - p_day_shift) = 0.06 := by
sorry

end NUMINAMATH_CALUDE_job_candidate_probability_l2099_209914


namespace NUMINAMATH_CALUDE_base_7_conversion_correct_l2099_209941

/-- Converts a list of digits in base 7 to its decimal (base 10) representation -/
def toDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The decimal number we want to convert -/
def decimalNumber : Nat := 1987

/-- The proposed base 7 representation -/
def base7Digits : List Nat := [6, 3, 5, 3, 5]

/-- Theorem stating that the conversion is correct -/
theorem base_7_conversion_correct :
  toDecimal base7Digits = decimalNumber := by sorry

end NUMINAMATH_CALUDE_base_7_conversion_correct_l2099_209941


namespace NUMINAMATH_CALUDE_river_road_cars_l2099_209990

theorem river_road_cars (buses cars : ℕ) : 
  (buses : ℚ) / cars = 1 / 17 →  -- ratio of buses to cars is 1:17
  cars = buses + 80 →            -- 80 fewer buses than cars
  cars = 85 :=                   -- prove that there are 85 cars
by sorry

end NUMINAMATH_CALUDE_river_road_cars_l2099_209990


namespace NUMINAMATH_CALUDE_sin_two_phi_l2099_209964

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) :
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_phi_l2099_209964


namespace NUMINAMATH_CALUDE_select_two_each_select_at_least_one_each_select_with_restriction_student_selection_methods_l2099_209951

/-- The number of female students -/
def num_females : ℕ := 5

/-- The number of male students -/
def num_males : ℕ := 4

/-- The number of students to be selected -/
def num_selected : ℕ := 4

/-- Theorem for the number of ways to select 2 males and 2 females -/
theorem select_two_each : ℕ := by sorry

/-- Theorem for the number of ways to select at least 1 male and 1 female -/
theorem select_at_least_one_each : ℕ := by sorry

/-- Theorem for the number of ways to select at least 1 male and 1 female, 
    but not both male A and female B -/
theorem select_with_restriction : ℕ := by sorry

/-- Main theorem combining all selection methods -/
theorem student_selection_methods :
  select_two_each = 1440 ∧
  select_at_least_one_each = 2880 ∧
  select_with_restriction = 2376 := by sorry

end NUMINAMATH_CALUDE_select_two_each_select_at_least_one_each_select_with_restriction_student_selection_methods_l2099_209951


namespace NUMINAMATH_CALUDE_calculation_proof_expression_equivalence_l2099_209981

-- First part of the problem
theorem calculation_proof : 28 + 72 + (9 - 8) = 172 := by sorry

-- Second part of the problem
def original_expression : ℚ := 4600 / 23 - 19 * 10

def reordered_expression : ℚ := (4600 / 23) - (19 * 10)

theorem expression_equivalence : original_expression = reordered_expression := by sorry

end NUMINAMATH_CALUDE_calculation_proof_expression_equivalence_l2099_209981


namespace NUMINAMATH_CALUDE_decimal_multiplication_l2099_209946

theorem decimal_multiplication (a b : ℚ) (n m : ℕ) :
  a = 0.125 →
  b = 3.84 →
  (a * 10^3).num * (b * 10^2).num = 48000 →
  a * b = 0.48 := by
sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l2099_209946


namespace NUMINAMATH_CALUDE_parabola_directrix_l2099_209948

/-- The equation of the directrix of the parabola y = -4x^2 - 16x + 1 -/
theorem parabola_directrix : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = -4 * x^2 - 16 * x + 1 ↔ y = a * (x - b)^2 + c) →
    (∃ d : ℝ, d = 273 / 16 ∧ 
      ∀ x y : ℝ, y = d → 
        (x - b)^2 + (y - c)^2 = (y - (c - 1 / (4 * |a|)))^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2099_209948


namespace NUMINAMATH_CALUDE_unique_solution_l2099_209986

theorem unique_solution : 
  ∀ x y : ℤ, (2*x + 5*y + 1)*(2^(Int.natAbs x) + x^2 + x + y) = 105 ↔ x = 0 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2099_209986


namespace NUMINAMATH_CALUDE_max_absolute_value_of_z_l2099_209944

theorem max_absolute_value_of_z (z : ℂ) : 
  Complex.abs (z - (3 + 4*I)) ≤ 2 → Complex.abs z ≤ 7 ∧ ∃ w : ℂ, Complex.abs (w - (3 + 4*I)) ≤ 2 ∧ Complex.abs w = 7 :=
sorry

end NUMINAMATH_CALUDE_max_absolute_value_of_z_l2099_209944


namespace NUMINAMATH_CALUDE_consecutive_cubes_inequality_l2099_209989

theorem consecutive_cubes_inequality (n : ℕ) : (n + 1)^3 ≠ n^3 + (n - 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_cubes_inequality_l2099_209989


namespace NUMINAMATH_CALUDE_min_value_theorem_l2099_209904

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2*x*y) :
  3*x + 4*y ≥ 5 + 2*Real.sqrt 6 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 2*x₀*y₀ ∧ 3*x₀ + 4*y₀ = 5 + 2*Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2099_209904


namespace NUMINAMATH_CALUDE_diamonds_in_F20_l2099_209918

/-- Definition of the number of diamonds in figure F_n -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 9
  else n^2 + (n-1)^2

/-- Theorem: The number of diamonds in F_20 is 761 -/
theorem diamonds_in_F20 :
  num_diamonds 20 = 761 := by sorry

end NUMINAMATH_CALUDE_diamonds_in_F20_l2099_209918


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l2099_209974

/-- There exist infinitely many ordered quadruples (x, y, z, w) of real numbers
    satisfying the given conditions. -/
theorem infinitely_many_solutions :
  ∃ (S : Set (ℝ × ℝ × ℝ × ℝ)), Set.Infinite S ∧
    ∀ (x y z w : ℝ), (x, y, z, w) ∈ S →
      (x + y = 3 ∧ x * y - z^2 = w ∧ w + z = 4) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l2099_209974


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2099_209910

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 1 → x^2 - 1 > 0)) ↔ (∃ x₀ : ℝ, x₀ > 1 ∧ x₀^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2099_209910


namespace NUMINAMATH_CALUDE_complex_product_squared_l2099_209940

theorem complex_product_squared (P R S : ℂ) : 
  P = 3 + 4*I ∧ R = 2*I ∧ S = 3 - 4*I → (P * R * S)^2 = -2500 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_squared_l2099_209940


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l2099_209928

theorem angle_sum_around_point (x : ℝ) : 
  (6*x + 3*x + x + x + 4*x = 360) → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l2099_209928


namespace NUMINAMATH_CALUDE_library_books_count_l2099_209943

theorem library_books_count (num_bookshelves : ℕ) (floors_per_bookshelf : ℕ) (books_per_floor : ℕ) :
  num_bookshelves = 28 →
  floors_per_bookshelf = 6 →
  books_per_floor = 19 →
  num_bookshelves * floors_per_bookshelf * books_per_floor = 3192 :=
by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l2099_209943


namespace NUMINAMATH_CALUDE_king_descendants_comparison_l2099_209929

theorem king_descendants_comparison :
  let pafnutius_sons := 2
  let pafnutius_two_sons := 60
  let pafnutius_one_son := 20
  let zenobius_daughters := 4
  let zenobius_three_daughters := 35
  let zenobius_one_daughter := 35

  let pafnutius_descendants := pafnutius_sons + pafnutius_two_sons * 2 + pafnutius_one_son * 1
  let zenobius_descendants := zenobius_daughters + zenobius_three_daughters * 3 + zenobius_one_daughter * 1

  zenobius_descendants > pafnutius_descendants := by sorry

end NUMINAMATH_CALUDE_king_descendants_comparison_l2099_209929


namespace NUMINAMATH_CALUDE_standard_normal_probability_l2099_209917

/-- Standard normal distribution function -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- Probability density function of the standard normal distribution -/
noncomputable def φ : ℝ → ℝ := sorry

theorem standard_normal_probability (X : ℝ → ℝ) : 
  (∀ (a b : ℝ), a < b → ∫ x in a..b, φ x = Φ b - Φ a) →
  Φ 2 - Φ (-1) = 0.8185 := by
  sorry

end NUMINAMATH_CALUDE_standard_normal_probability_l2099_209917


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l2099_209973

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x > 2/3 ∨ x < -6} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (7/2)*t} = {t : ℝ | 3/2 ≤ t ∧ t ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l2099_209973


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_2018_2013_l2099_209999

/-- A geometric sequence with common ratio q > 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio_2018_2013 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : GeometricSequence a q)
  (h_sum : a 1 + a 6 = 8)
  (h_prod : a 3 * a 4 = 12) :
  a 2018 / a 2013 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_2018_2013_l2099_209999


namespace NUMINAMATH_CALUDE_inequality_proof_l2099_209926

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + y^3 ≥ x^3 + y^4) : x^3 + y^3 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2099_209926


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2099_209985

theorem rectangle_perimeter (l w : ℝ) 
  (sum_two_sides : l + w = 7)
  (sum_three_sides : 2 * l + w = 9.5) :
  2 * (l + w) = 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2099_209985


namespace NUMINAMATH_CALUDE_bus_seats_solution_l2099_209935

/-- Represents the seating arrangement in a bus -/
structure BusSeats where
  left : ℕ  -- Number of seats on the left side
  right : ℕ  -- Number of seats on the right side
  back : ℕ  -- Capacity of the back seat
  capacity_per_seat : ℕ  -- Number of people each regular seat can hold

/-- The total capacity of the bus -/
def total_capacity (bs : BusSeats) : ℕ :=
  bs.capacity_per_seat * (bs.left + bs.right) + bs.back

theorem bus_seats_solution :
  ∃ (bs : BusSeats),
    bs.right = bs.left - 3 ∧
    bs.capacity_per_seat = 3 ∧
    bs.back = 10 ∧
    total_capacity bs = 91 ∧
    bs.left = 15 := by
  sorry

end NUMINAMATH_CALUDE_bus_seats_solution_l2099_209935


namespace NUMINAMATH_CALUDE_negation_of_existence_squared_greater_than_two_l2099_209950

theorem negation_of_existence_squared_greater_than_two :
  (¬ ∃ x : ℝ, x^2 > 2) ↔ (∀ x : ℝ, x^2 ≤ 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_squared_greater_than_two_l2099_209950


namespace NUMINAMATH_CALUDE_power_sum_problem_l2099_209996

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 85) :
  a * x^5 + b * y^5 = 7025 / 29 := by
sorry

end NUMINAMATH_CALUDE_power_sum_problem_l2099_209996


namespace NUMINAMATH_CALUDE_power_three_mod_ten_l2099_209984

theorem power_three_mod_ten : 3^19 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_ten_l2099_209984


namespace NUMINAMATH_CALUDE_gray_trees_count_l2099_209967

/-- Represents a drone photograph of an area --/
structure Photograph where
  visible_trees : ℕ
  total_trees : ℕ

/-- Represents a set of three drone photographs of the same area --/
structure PhotoSet where
  photo1 : Photograph
  photo2 : Photograph
  photo3 : Photograph
  equal_total : photo1.total_trees = photo2.total_trees ∧ photo2.total_trees = photo3.total_trees

/-- Calculates the number of trees in gray areas given a set of three photographs --/
def gray_trees (photos : PhotoSet) : ℕ :=
  (photos.photo1.total_trees - photos.photo1.visible_trees) +
  (photos.photo2.total_trees - photos.photo2.visible_trees)

/-- Theorem stating that for the given set of photographs, the number of trees in gray areas is 26 --/
theorem gray_trees_count (photos : PhotoSet)
  (h1 : photos.photo1.visible_trees = 100)
  (h2 : photos.photo2.visible_trees = 90)
  (h3 : photos.photo3.visible_trees = 82) :
  gray_trees photos = 26 := by
  sorry


end NUMINAMATH_CALUDE_gray_trees_count_l2099_209967


namespace NUMINAMATH_CALUDE_boys_not_in_varsity_clubs_l2099_209934

theorem boys_not_in_varsity_clubs (total_students : ℕ) (girls_percentage : ℚ) (boys_in_clubs_fraction : ℚ) :
  total_students = 150 →
  girls_percentage = 60 / 100 →
  boys_in_clubs_fraction = 1 / 3 →
  (total_students : ℚ) * (1 - girls_percentage) * (1 - boys_in_clubs_fraction) = 40 :=
by sorry

end NUMINAMATH_CALUDE_boys_not_in_varsity_clubs_l2099_209934


namespace NUMINAMATH_CALUDE_books_bought_two_years_ago_l2099_209945

/-- Represents the number of books in a library over time --/
structure LibraryBooks where
  initial : ℕ  -- Initial number of books 5 years ago
  bought_two_years_ago : ℕ  -- Books bought 2 years ago
  bought_last_year : ℕ  -- Books bought last year
  donated : ℕ  -- Books donated this year
  current : ℕ  -- Current number of books

/-- Theorem stating the number of books bought two years ago --/
theorem books_bought_two_years_ago 
  (lib : LibraryBooks) 
  (h1 : lib.initial = 500)
  (h2 : lib.bought_last_year = lib.bought_two_years_ago + 100)
  (h3 : lib.donated = 200)
  (h4 : lib.current = 1000)
  (h5 : lib.current = lib.initial + lib.bought_two_years_ago + lib.bought_last_year - lib.donated) :
  lib.bought_two_years_ago = 300 := by
  sorry

#check books_bought_two_years_ago

end NUMINAMATH_CALUDE_books_bought_two_years_ago_l2099_209945


namespace NUMINAMATH_CALUDE_triangle_area_product_l2099_209972

theorem triangle_area_product (p q : ℝ) : 
  p > 0 → q > 0 → 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ p * x + q * y = 12) →
  (1/2 * (12/p) * (12/q) = 12) →
  p * q = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_product_l2099_209972


namespace NUMINAMATH_CALUDE_five_spheres_max_regions_l2099_209937

/-- The maximum number of regions into which n spheres can divide three-dimensional space -/
def max_regions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => max_regions n + 2 + n + n * (n + 1) / 2

/-- The maximum number of regions into which five spheres can divide three-dimensional space is 47 -/
theorem five_spheres_max_regions :
  max_regions 5 = 47 := by sorry

end NUMINAMATH_CALUDE_five_spheres_max_regions_l2099_209937


namespace NUMINAMATH_CALUDE_scarletts_oil_measurement_l2099_209911

theorem scarletts_oil_measurement (initial_oil : ℝ) : 
  (initial_oil + 0.67 = 0.84) → initial_oil = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_scarletts_oil_measurement_l2099_209911


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2099_209949

theorem quadratic_expression_value (a b : ℝ) : 
  (2 : ℝ)^2 + a * 2 - 6 = 0 ∧ 
  b^2 + a * b - 6 = 0 → 
  (2 * a + b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2099_209949


namespace NUMINAMATH_CALUDE_identity_function_unique_l2099_209916

/-- A function satisfying the given conditions -/
def satisfying_function (f : ℝ → ℝ) : Prop :=
  (f 1 = 1) ∧ 
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ 
  (∀ x : ℝ, x ≠ 0 → f (1 / x) = f x / x^2)

/-- Theorem stating that any function satisfying the conditions is the identity function -/
theorem identity_function_unique (f : ℝ → ℝ) (h : satisfying_function f) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_identity_function_unique_l2099_209916


namespace NUMINAMATH_CALUDE_line_equivalence_l2099_209963

theorem line_equivalence :
  ∀ (x y : ℝ),
  (3 : ℝ) * (x - 2) + (-4 : ℝ) * (y - (-1)) = 0 ↔
  y = (3/4 : ℝ) * x - (5/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_line_equivalence_l2099_209963


namespace NUMINAMATH_CALUDE_tearing_process_l2099_209901

/-- Represents the number of parts after a series of tearing operations -/
def NumParts : ℕ → ℕ
  | 0 => 1  -- Start with one piece
  | n + 1 => NumParts n + 2  -- Each tear adds 2 parts

theorem tearing_process (n : ℕ) :
  ∀ k, Odd (NumParts k) ∧ 
    (¬∃ m, NumParts m = 100) ∧
    (∃ m, NumParts m = 2017) := by
  sorry

#eval NumParts 1008  -- Should evaluate to 2017

end NUMINAMATH_CALUDE_tearing_process_l2099_209901


namespace NUMINAMATH_CALUDE_rectangular_garden_dimensions_l2099_209936

theorem rectangular_garden_dimensions (perimeter area fixed_side : ℝ) :
  perimeter = 60 →
  area = 200 →
  fixed_side = 10 →
  ∃ (adjacent_side : ℝ),
    adjacent_side = 20 ∧
    2 * (fixed_side + adjacent_side) = perimeter ∧
    fixed_side * adjacent_side = area :=
by sorry

end NUMINAMATH_CALUDE_rectangular_garden_dimensions_l2099_209936


namespace NUMINAMATH_CALUDE_tree_age_difference_l2099_209919

/-- The number of rings in one group -/
def rings_per_group : ℕ := 6

/-- The number of ring groups in the first tree -/
def first_tree_groups : ℕ := 70

/-- The number of ring groups in the second tree -/
def second_tree_groups : ℕ := 40

/-- Each ring represents one year of growth -/
axiom ring_year_correspondence : ∀ (n : ℕ), n.succ.pred = n

theorem tree_age_difference : 
  (first_tree_groups * rings_per_group) - (second_tree_groups * rings_per_group) = 180 := by
  sorry

end NUMINAMATH_CALUDE_tree_age_difference_l2099_209919


namespace NUMINAMATH_CALUDE_prob_blue_or_purple_l2099_209982

/-- A bag of jelly beans with different colors -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  purple : ℕ

/-- The probability of selecting either a blue or purple jelly bean -/
def bluePurpleProbability (bag : JellyBeanBag) : ℚ :=
  (bag.blue + bag.purple : ℚ) / (bag.red + bag.green + bag.yellow + bag.blue + bag.purple : ℚ)

/-- Theorem stating the probability of selecting a blue or purple jelly bean from the given bag -/
theorem prob_blue_or_purple (bag : JellyBeanBag) 
    (h : bag = { red := 7, green := 8, yellow := 9, blue := 10, purple := 4 }) : 
    bluePurpleProbability bag = 7 / 19 := by
  sorry

#eval bluePurpleProbability { red := 7, green := 8, yellow := 9, blue := 10, purple := 4 }

end NUMINAMATH_CALUDE_prob_blue_or_purple_l2099_209982


namespace NUMINAMATH_CALUDE_other_root_of_complex_polynomial_l2099_209902

theorem other_root_of_complex_polynomial (m n : ℝ) :
  (Complex.I + 3) ^ 2 + m * (Complex.I + 3) + n = 0 →
  (3 - Complex.I) ^ 2 + m * (3 - Complex.I) + n = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_complex_polynomial_l2099_209902


namespace NUMINAMATH_CALUDE_valentines_remaining_l2099_209962

/-- Given that Mrs. Wong initially had 30 Valentines and gave 8 away,
    prove that she has 22 Valentines left. -/
theorem valentines_remaining (initial : Nat) (given_away : Nat) :
  initial = 30 → given_away = 8 → initial - given_away = 22 := by
  sorry

end NUMINAMATH_CALUDE_valentines_remaining_l2099_209962


namespace NUMINAMATH_CALUDE_range_of_a_l2099_209953

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 5 ≥ a^2 - 3*a) ↔ -1 ≤ a ∧ a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2099_209953


namespace NUMINAMATH_CALUDE_liam_cycling_speed_l2099_209979

/-- Given the cycling speeds of Eugene, Claire, and Liam, prove that Liam's speed is 6 miles per hour. -/
theorem liam_cycling_speed 
  (eugene_speed : ℝ) 
  (claire_speed_ratio : ℝ) 
  (liam_speed_ratio : ℝ) 
  (h1 : eugene_speed = 6)
  (h2 : claire_speed_ratio = 3/4)
  (h3 : liam_speed_ratio = 4/3) :
  liam_speed_ratio * (claire_speed_ratio * eugene_speed) = 6 :=
by sorry

end NUMINAMATH_CALUDE_liam_cycling_speed_l2099_209979


namespace NUMINAMATH_CALUDE_eggs_left_l2099_209907

theorem eggs_left (initial_eggs : ℕ) (taken_eggs : ℕ) (h1 : initial_eggs = 47) (h2 : taken_eggs = 5) :
  initial_eggs - taken_eggs = 42 := by
sorry

end NUMINAMATH_CALUDE_eggs_left_l2099_209907


namespace NUMINAMATH_CALUDE_function_value_theorem_l2099_209912

theorem function_value_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = Real.sqrt (2 * x + 1)) →
  f a = 5 →
  a = 12 := by
  sorry

end NUMINAMATH_CALUDE_function_value_theorem_l2099_209912


namespace NUMINAMATH_CALUDE_least_tablets_extracted_l2099_209921

theorem least_tablets_extracted (tablets_a tablets_b : ℕ) 
  (ha : tablets_a = 10) (hb : tablets_b = 16) :
  ∃ (n : ℕ), n ≤ tablets_a + tablets_b ∧ 
  (∀ (k : ℕ), k < n → 
    (k < tablets_a + 2 → ∃ (x y : ℕ), x + y = k ∧ (x < 2 ∨ y < 2)) ∧
    (k ≥ tablets_a + 2 → ∃ (x y : ℕ), x + y = k ∧ x ≥ 2 ∧ y ≥ 2)) ∧
  n = 12 :=
sorry

end NUMINAMATH_CALUDE_least_tablets_extracted_l2099_209921


namespace NUMINAMATH_CALUDE_closest_to_580_l2099_209980

def problem_value : ℝ := 0.000218 * 5432000 - 500

def options : List ℝ := [520, 580, 600, 650]

theorem closest_to_580 : 
  ∀ x ∈ options, |problem_value - 580| ≤ |problem_value - x| := by
  sorry

end NUMINAMATH_CALUDE_closest_to_580_l2099_209980


namespace NUMINAMATH_CALUDE_equation_solution_l2099_209903

theorem equation_solution : ∃ t : ℝ, t = 9/4 ∧ Real.sqrt (3 * Real.sqrt (t - 1)) = (t + 9) ^ (1/4) :=
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2099_209903


namespace NUMINAMATH_CALUDE_intersection_area_l2099_209927

/-- The area of a region formed by the intersection of four circles -/
theorem intersection_area (r : ℝ) (h : r = 5) : 
  ∃ (A : ℝ), A = 50 * (π - 2) ∧ 
  A = 8 * (((π * r^2) / 4) - ((r^2) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_l2099_209927


namespace NUMINAMATH_CALUDE_complex_power_sum_l2099_209905

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (5 * π / 180)) :
  z^100 + 1 / z^100 = 2 * Real.cos (140 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2099_209905


namespace NUMINAMATH_CALUDE_simplify_expression_l2099_209933

theorem simplify_expression (a b : ℝ) : a * b - (a^2 - a * b + b^2) = -a^2 + 2 * a * b - b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2099_209933


namespace NUMINAMATH_CALUDE_tan_sum_equals_one_l2099_209988

-- Define the line equation
def line_equation (x y : ℝ) (α β : ℝ) : Prop :=
  x * Real.tan α - y - 3 * Real.tan β = 0

-- Define the theorem
theorem tan_sum_equals_one (α β : ℝ) :
  (∃ (x y : ℝ), line_equation x y α β) → -- Line equation exists
  (Real.tan α = 2) →                     -- Slope is 2
  (3 * Real.tan β = -1) →                -- Y-intercept is 1
  Real.tan (α + β) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_equals_one_l2099_209988


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2099_209978

theorem unique_solution_quadratic (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) ↔ (a = 0 ∨ a = 9/8) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2099_209978


namespace NUMINAMATH_CALUDE_negation_existential_to_universal_l2099_209959

theorem negation_existential_to_universal :
  (¬ ∃ x : ℝ, 2 * x - 3 > 1) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_existential_to_universal_l2099_209959


namespace NUMINAMATH_CALUDE_fraction_equality_l2099_209995

theorem fraction_equality (a b c : ℝ) (h1 : b ≠ c) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
  (a - b) / (b - c) = a / c ↔ 1 / b = (1 / a + 1 / c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2099_209995


namespace NUMINAMATH_CALUDE_seeds_per_flower_bed_l2099_209939

theorem seeds_per_flower_bed 
  (total_seeds : ℕ) 
  (num_flower_beds : ℕ) 
  (h1 : total_seeds = 60) 
  (h2 : num_flower_beds = 6) 
  : total_seeds / num_flower_beds = 10 := by
  sorry

end NUMINAMATH_CALUDE_seeds_per_flower_bed_l2099_209939


namespace NUMINAMATH_CALUDE_quadratic_function_range_l2099_209952

/-- A quadratic function with a positive leading coefficient -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_range
  (f : ℝ → ℝ)
  (h1 : QuadraticFunction f)
  (h2 : ∀ x : ℝ, f x = f (4 - x))
  (h3 : ∀ x : ℝ, f (1 - 2*x^2) < f (1 + 2*x - x^2)) :
  ∀ x : ℝ, f (1 - 2*x^2) < f (1 + 2*x - x^2) → -2 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l2099_209952


namespace NUMINAMATH_CALUDE_right_triangle_base_length_l2099_209924

theorem right_triangle_base_length 
  (height : ℝ) 
  (area : ℝ) 
  (hypotenuse : ℝ) 
  (h1 : height = 8) 
  (h2 : area = 24) 
  (h3 : hypotenuse = 10) : 
  ∃ (base : ℝ), base = 6 ∧ area = (1/2) * base * height ∧ hypotenuse^2 = height^2 + base^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_base_length_l2099_209924


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l2099_209925

/-- Given a vector a = (cos α, 1/2) with magnitude √2/2, prove that cos(2α) = -1/2 -/
theorem cos_double_angle_special_case (α : ℝ) :
  let a : ℝ × ℝ := (Real.cos α, 1/2)
  (a.1^2 + a.2^2 = 1/2) →
  Real.cos (2 * α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l2099_209925


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_dot_product_not_sufficient_exp_not_periodic_negation_existential_l2099_209994

-- 1. Contrapositive
theorem contrapositive_equivalence (p q : ℝ) :
  (p^2 + q^2 = 2 → p + q ≤ 2) ↔ (p + q > 2 → p^2 + q^2 ≠ 2) := by sorry

-- 2. Vector dot product
theorem dot_product_not_sufficient (a b c : ℝ × ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  (a.1 * b.1 + a.2 * b.2 = b.1 * c.1 + b.2 * c.2) → (a = c → False) := by sorry

-- 3. Non-periodicity of exponential function
theorem exp_not_periodic (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ¬∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, a^x = a^(x + T) := by sorry

-- 4. Negation of existential proposition
theorem negation_existential :
  (¬∃ x : ℝ, x^2 - 3*x + 2 ≥ 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 < 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_dot_product_not_sufficient_exp_not_periodic_negation_existential_l2099_209994


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l2099_209915

/-- Given a group of families, some with children and some without, 
    calculate the average number of children in families that have children. -/
theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : total_average = 3)
  (h3 : childless_families = 3)
  (h4 : childless_families < total_families) :
  (total_families : ℚ) * total_average / (total_families - childless_families : ℚ) = 3.75 := by
sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l2099_209915


namespace NUMINAMATH_CALUDE_parabola_sum_coefficients_l2099_209956

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum_coefficients (p : Parabola) :
  p.x_coord 2 = -3 →  -- vertex at (-3, 2)
  p.x_coord (-1) = 1 →  -- passes through (1, -1)
  p.a < 0 →  -- opens to the left
  p.a + p.b + p.c = -23/9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_coefficients_l2099_209956


namespace NUMINAMATH_CALUDE_cannot_make_65_cents_l2099_209955

def coin_value (coin : Nat) : Nat :=
  match coin with
  | 0 => 5  -- nickel
  | 1 => 10 -- dime
  | 2 => 25 -- quarter
  | 3 => 50 -- half-dollar
  | _ => 0  -- invalid coin

def is_valid_coin (c : Nat) : Prop := c ≤ 3

theorem cannot_make_65_cents :
  ¬ ∃ (a b c d e : Nat),
    is_valid_coin a ∧ is_valid_coin b ∧ is_valid_coin c ∧ is_valid_coin d ∧ is_valid_coin e ∧
    coin_value a + coin_value b + coin_value c + coin_value d + coin_value e = 65 :=
by sorry

end NUMINAMATH_CALUDE_cannot_make_65_cents_l2099_209955


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l2099_209920

/-- A quadratic function passing through specific points with a given vertex -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficient_sum (a b c : ℝ) :
  (QuadraticFunction a b c 1 = 4) →
  (QuadraticFunction a b c (-2) = -1) →
  (∀ x, QuadraticFunction a b c x ≥ QuadraticFunction a b c (-1)) →
  (QuadraticFunction a b c (-1) = -2) →
  a + b + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l2099_209920


namespace NUMINAMATH_CALUDE_periodic_sin_and_empty_subset_l2099_209998

-- Define a periodic function
def isPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = f x

-- Define the sine function
noncomputable def sin : ℝ → ℝ := Real.sin

-- Define set A
variable (A : Set ℝ)

-- Theorem statement
theorem periodic_sin_and_empty_subset (A : Set ℝ) : 
  (isPeriodic sin) ∧ (∅ ⊆ A) := by sorry

end NUMINAMATH_CALUDE_periodic_sin_and_empty_subset_l2099_209998


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_3_4_5_l2099_209954

theorem least_three_digit_multiple_of_3_4_5 : 
  (∀ n : ℕ, n ≥ 100 ∧ n < 120 → ¬(3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n)) ∧ 
  (120 ≥ 100 ∧ 3 ∣ 120 ∧ 4 ∣ 120 ∧ 5 ∣ 120) := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_3_4_5_l2099_209954


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l2099_209930

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), (2^k : ℤ) ∣ (12^500 - 6^500) ∧ 
  ∀ (m : ℕ), (2^m : ℤ) ∣ (12^500 - 6^500) → m ≤ k :=
by
  use 501
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l2099_209930


namespace NUMINAMATH_CALUDE_reduced_oil_price_is_80_l2099_209975

/-- Represents the price of oil before and after reduction -/
structure OilPrice where
  original : ℝ
  reduced : ℝ
  reduction_percent : ℝ
  h_reduction : reduced = original * (1 - reduction_percent / 100)

/-- Represents the amount of oil that can be purchased for a fixed price -/
structure OilPurchase where
  price : ℝ
  original_quantity : ℝ
  reduced_quantity : ℝ
  h_quantity_increase : reduced_quantity = original_quantity + 5

/-- Theorem stating that given the conditions, the reduced price of oil is 80 -/
theorem reduced_oil_price_is_80 
  (op : OilPrice) 
  (purchase : OilPurchase) 
  (h_reduction : op.reduction_percent = 50)
  (h_price : purchase.price = 800)
  (h_quantity_balance : purchase.price = op.original * purchase.original_quantity)
  (h_new_quantity_balance : purchase.price = op.reduced * purchase.reduced_quantity) :
  op.reduced = 80 := by
  sorry

end NUMINAMATH_CALUDE_reduced_oil_price_is_80_l2099_209975


namespace NUMINAMATH_CALUDE_farmer_wheat_harvest_l2099_209909

theorem farmer_wheat_harvest (estimated_harvest additional_harvest : ℕ) 
  (h1 : estimated_harvest = 213489)
  (h2 : additional_harvest = 13257) :
  estimated_harvest + additional_harvest = 226746 := by
  sorry

end NUMINAMATH_CALUDE_farmer_wheat_harvest_l2099_209909


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l2099_209993

theorem sqrt_product_plus_one : 
  Real.sqrt ((20 : ℝ) * 19 * 18 * 17 + 1) = 341 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l2099_209993


namespace NUMINAMATH_CALUDE_min_a_for_ln_inequality_l2099_209969

/-- The minimum value of a for which ln x ≤ ax + 1 holds for all x > 0 is 1/e^2 -/
theorem min_a_for_ln_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 → Real.log x ≤ a * x + 1) ∧ 
  (∀ (a : ℝ), (∀ (x : ℝ), x > 0 → Real.log x ≤ a * x + 1) → a ≥ 1 / Real.exp 2) ∧
  (∃ (a : ℝ), a = 1 / Real.exp 2 ∧ ∀ (x : ℝ), x > 0 → Real.log x ≤ a * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_ln_inequality_l2099_209969


namespace NUMINAMATH_CALUDE_lowest_cost_plan_l2099_209971

/-- Represents a gardening style arrangement plan -/
structure ArrangementPlan where
  style_a : ℕ
  style_b : ℕ

/-- Represents the gardening problem setup -/
structure GardeningProblem where
  total_sets : ℕ
  type_a_flowers : ℕ
  type_b_flowers : ℕ
  style_a_type_a : ℕ
  style_a_type_b : ℕ
  style_b_type_a : ℕ
  style_b_type_b : ℕ
  style_a_cost : ℕ
  style_b_cost : ℕ

/-- Checks if an arrangement plan is feasible -/
def is_feasible (problem : GardeningProblem) (plan : ArrangementPlan) : Prop :=
  plan.style_a + plan.style_b = problem.total_sets ∧
  plan.style_a * problem.style_a_type_a + plan.style_b * problem.style_b_type_a ≤ problem.type_a_flowers ∧
  plan.style_a * problem.style_a_type_b + plan.style_b * problem.style_b_type_b ≤ problem.type_b_flowers

/-- Calculates the cost of an arrangement plan -/
def cost (problem : GardeningProblem) (plan : ArrangementPlan) : ℕ :=
  plan.style_a * problem.style_a_cost + plan.style_b * problem.style_b_cost

/-- The main theorem to be proved -/
theorem lowest_cost_plan (problem : GardeningProblem) 
  (h_problem : problem = { 
    total_sets := 50,
    type_a_flowers := 2660,
    type_b_flowers := 3000,
    style_a_type_a := 70,
    style_a_type_b := 30,
    style_b_type_a := 40,
    style_b_type_b := 80,
    style_a_cost := 800,
    style_b_cost := 960
  }) :
  ∃ (optimal_plan : ArrangementPlan),
    is_feasible problem optimal_plan ∧
    cost problem optimal_plan = 44480 ∧
    ∀ (other_plan : ArrangementPlan), 
      is_feasible problem other_plan → 
      cost problem other_plan ≥ cost problem optimal_plan :=
sorry

end NUMINAMATH_CALUDE_lowest_cost_plan_l2099_209971


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2099_209968

/-- Given a line y = mx + b, if the reflection of point (2, 3) across this line is (10, 7), then m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    (x - 2)^2 + (y - 3)^2 = (10 - x)^2 + (7 - y)^2 ∧ 
    y = m * x + b ∧
    (y - 3) = -1 / m * (x - 2)) → 
  m + b = 15 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l2099_209968


namespace NUMINAMATH_CALUDE_quartic_comparison_l2099_209958

noncomputable def Q (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 - 4*x + 5

def sum_of_zeros (f : ℝ → ℝ) : ℝ := 2 -- From Vieta's formula for quartic polynomial

def product_of_zeros (f : ℝ → ℝ) : ℝ := f 0

def sum_of_coefficients (f : ℝ → ℝ) : ℝ := 3 -- 1 - 2 + 3 - 4 + 5

theorem quartic_comparison :
  (sum_of_zeros Q)^2 ≤ Q (-1) ∧
  (sum_of_zeros Q)^2 ≤ product_of_zeros Q ∧
  (sum_of_zeros Q)^2 ≤ sum_of_coefficients Q :=
sorry

end NUMINAMATH_CALUDE_quartic_comparison_l2099_209958


namespace NUMINAMATH_CALUDE_max_silver_tokens_l2099_209938

/-- Represents the state of Alex's tokens -/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth -/
structure Booth where
  redIn : ℕ
  blueIn : ℕ
  redOut : ℕ
  blueOut : ℕ
  silverOut : ℕ

/-- Checks if an exchange is possible at a given booth -/
def canExchange (state : TokenState) (booth : Booth) : Prop :=
  state.red ≥ booth.redIn ∧ state.blue ≥ booth.blueIn

/-- Performs an exchange at a given booth -/
def exchange (state : TokenState) (booth : Booth) : TokenState :=
  { red := state.red - booth.redIn + booth.redOut,
    blue := state.blue - booth.blueIn + booth.blueOut,
    silver := state.silver + booth.silverOut }

/-- The theorem to be proved -/
theorem max_silver_tokens : ∃ (finalState : TokenState),
  let initialState : TokenState := { red := 90, blue := 65, silver := 0 }
  let booth1 : Booth := { redIn := 3, blueIn := 0, redOut := 0, blueOut := 2, silverOut := 1 }
  let booth2 : Booth := { redIn := 0, blueIn := 4, redOut := 2, blueOut := 0, silverOut := 1 }
  (∀ state, (canExchange state booth1 ∨ canExchange state booth2) → 
    (finalState.silver ≥ state.silver)) ∧
  (¬ canExchange finalState booth1 ∧ ¬ canExchange finalState booth2) ∧
  finalState.silver = 67 :=
sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l2099_209938


namespace NUMINAMATH_CALUDE_prove_average_marks_l2099_209906

def average_marks (M P C : ℝ) : Prop :=
  M + P = 40 ∧ C = P + 20 → (M + C) / 2 = 30

theorem prove_average_marks :
  ∀ M P C : ℝ, average_marks M P C :=
by
  sorry

end NUMINAMATH_CALUDE_prove_average_marks_l2099_209906
