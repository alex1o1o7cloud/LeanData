import Mathlib

namespace NUMINAMATH_CALUDE_base_number_problem_l4028_402818

theorem base_number_problem (x : ℝ) (k : ℕ) 
  (h1 : x^k = 5) 
  (h2 : x^(2*k + 2) = 400) : 
  x = 5 := by sorry

end NUMINAMATH_CALUDE_base_number_problem_l4028_402818


namespace NUMINAMATH_CALUDE_divisibility_by_30_l4028_402815

theorem divisibility_by_30 : 
  (∃ p : ℕ, Prime p ∧ p ≥ 7 ∧ (30 ∣ p^2 - 1)) ∧ 
  (∃ p : ℕ, Prime p ∧ p ≥ 7 ∧ ¬(30 ∣ p^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_30_l4028_402815


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l4028_402874

theorem unique_integer_divisible_by_14_with_sqrt_between_25_and_25_3 :
  ∃! n : ℕ+, 14 ∣ n ∧ 25 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 25.3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l4028_402874


namespace NUMINAMATH_CALUDE_percentage_difference_l4028_402839

theorem percentage_difference (x y : ℝ) (h : x = 5 * y) : 
  (x - y) / x * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l4028_402839


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4028_402837

theorem right_triangle_hypotenuse (x y h : ℝ) : 
  x > 0 → 
  y = 2 * x + 2 → 
  (1 / 2) * x * y = 72 → 
  x^2 + y^2 = h^2 → 
  h = Real.sqrt 388 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4028_402837


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4028_402851

theorem inequality_solution_set (x : ℝ) :
  -6 * x^2 - x + 2 ≤ 0 ↔ x ≥ (1/2 : ℝ) ∨ x ≤ -(2/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4028_402851


namespace NUMINAMATH_CALUDE_remainder_of_valid_polynomials_l4028_402821

/-- The number of elements in the tuple -/
def tuple_size : ℕ := 2011

/-- The upper bound for each element in the tuple -/
def upper_bound : ℕ := 2011^2

/-- The degree of the polynomial -/
def poly_degree : ℕ := 4019

/-- The modulus for the divisibility conditions -/
def modulus : ℕ := 2011^2

/-- The final modulus for the remainder -/
def final_modulus : ℕ := 1000

/-- The expected remainder -/
def expected_remainder : ℕ := 281

/-- A function representing the conditions on the polynomial -/
def valid_polynomial (f : ℤ → ℤ) : Prop :=
  (∀ n : ℤ, ∃ k : ℤ, f n = k) ∧
  (∀ i : ℕ, i ≤ tuple_size → ∃ k : ℤ, f i - k = modulus * (f i / modulus)) ∧
  (∀ n : ℤ, ∃ k : ℤ, f (n + tuple_size) - f n = modulus * k)

/-- The main theorem -/
theorem remainder_of_valid_polynomials :
  (upper_bound ^ (poly_degree + 1)) % final_modulus = expected_remainder := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_valid_polynomials_l4028_402821


namespace NUMINAMATH_CALUDE_collinearity_ABD_l4028_402899

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two non-zero vectors are not collinear -/
def not_collinear (a b : V) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ ¬∃ (r : ℝ), a = r • b

/-- Three points are collinear if the vector from the first to the third is a scalar multiple of the vector from the first to the second -/
def collinear (A B D : V) : Prop := ∃ (t : ℝ), D - A = t • (B - A)

theorem collinearity_ABD 
  (a b : V) 
  (h_not_collinear : not_collinear a b)
  (h_AB : B - A = a + b)
  (h_BC : C - B = a + 10 • b)
  (h_CD : D - C = 3 • (a - 2 • b)) :
  collinear A B D :=
sorry

end NUMINAMATH_CALUDE_collinearity_ABD_l4028_402899


namespace NUMINAMATH_CALUDE_night_shift_guards_l4028_402896

/-- Represents the number of guards hired for a night shift -/
def num_guards (total_hours middle_guard_hours first_guard_hours last_guard_hours : ℕ) : ℕ :=
  let middle_guards := (total_hours - first_guard_hours - last_guard_hours) / middle_guard_hours
  1 + middle_guards + 1

/-- Theorem stating the number of guards hired for the night shift -/
theorem night_shift_guards : 
  num_guards 9 2 3 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_night_shift_guards_l4028_402896


namespace NUMINAMATH_CALUDE_rachel_reading_homework_l4028_402856

/-- The number of pages of reading homework Rachel had to complete -/
def reading_pages : ℕ := 15 - 2 - 10

/-- The total number of pages Rachel had to complete -/
def total_pages : ℕ := 15

/-- The number of pages of math homework Rachel had to complete -/
def math_pages : ℕ := 2

/-- The number of pages of biology homework Rachel had to complete -/
def biology_pages : ℕ := 10

theorem rachel_reading_homework :
  reading_pages = 3 ∧
  total_pages = math_pages + reading_pages + biology_pages :=
sorry

end NUMINAMATH_CALUDE_rachel_reading_homework_l4028_402856


namespace NUMINAMATH_CALUDE_no_integer_solutions_l4028_402883

theorem no_integer_solutions : ¬∃ (x y : ℤ), (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l4028_402883


namespace NUMINAMATH_CALUDE_max_weight_theorem_l4028_402806

def weight_set : Set ℕ := {2, 5, 10}

def is_measurable (w : ℕ) : Prop :=
  ∃ (a b c : ℕ), w = 2*a + 5*b + 10*c

def max_measurable : ℕ := 17

theorem max_weight_theorem :
  (∀ w : ℕ, is_measurable w → w ≤ max_measurable) ∧
  is_measurable max_measurable :=
sorry

end NUMINAMATH_CALUDE_max_weight_theorem_l4028_402806


namespace NUMINAMATH_CALUDE_pairing_natural_numbers_to_perfect_squares_l4028_402889

theorem pairing_natural_numbers_to_perfect_squares :
  ∃ f : ℕ → (ℕ × ℕ), 
    (∀ n : ℕ, ∃ m : ℕ, (f n).1 + (f n).2 = m^2) ∧ 
    (∀ x : ℕ, ∃! n : ℕ, x = (f n).1 ∨ x = (f n).2) := by
  sorry

end NUMINAMATH_CALUDE_pairing_natural_numbers_to_perfect_squares_l4028_402889


namespace NUMINAMATH_CALUDE_three_minus_one_point_two_repeating_l4028_402897

/-- The decimal representation of 1.2 repeating -/
def one_point_two_repeating : ℚ := 11 / 9

/-- Proof that 3 - 1.2 repeating equals 16/9 -/
theorem three_minus_one_point_two_repeating :
  3 - one_point_two_repeating = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_three_minus_one_point_two_repeating_l4028_402897


namespace NUMINAMATH_CALUDE_matrix_determinant_and_fraction_sum_l4028_402894

theorem matrix_determinant_and_fraction_sum (p q r : ℝ) :
  let M := ![![p, 2*q, r],
             ![q, r, p],
             ![r, p, q]]
  Matrix.det M = 0 →
  (p / (2*q + r) + 2*q / (p + r) + r / (p + q) = -4) ∨
  (p / (2*q + r) + 2*q / (p + r) + r / (p + q) = 11/6) :=
by sorry

end NUMINAMATH_CALUDE_matrix_determinant_and_fraction_sum_l4028_402894


namespace NUMINAMATH_CALUDE_derivative_of_f_l4028_402891

-- Define the function f
def f (x : ℝ) : ℝ := (3*x - 5)^2

-- State the theorem
theorem derivative_of_f :
  deriv f = λ x => 6 * (3*x - 5) := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l4028_402891


namespace NUMINAMATH_CALUDE_cone_base_radius_l4028_402826

/-- Given a sector paper with a central angle of 90° and a radius of 20 cm
    used to form the lateral surface of a cone, the radius of the base of the cone is 5 cm. -/
theorem cone_base_radius (θ : Real) (R : Real) (r : Real) : 
  θ = 90 → R = 20 → 2 * π * r = (θ / 360) * 2 * π * R → r = 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l4028_402826


namespace NUMINAMATH_CALUDE_expression_equality_l4028_402849

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 2/y) :
  (x^2 - 1/x^2) * (y^2 + 1/y^2) = (x^4/4) - (4/x^4) + 3.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l4028_402849


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l4028_402802

/-- Given two plane vectors a and b, prove that |a + 2b| = √7 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  (a.fst = 1 ∧ a.snd = 0) →  -- a = (1,0)
  ‖b‖ = 1 →  -- |b| = 1
  Real.cos (Real.pi / 3) = (a.fst * b.fst + a.snd * b.snd) / (‖a‖ * ‖b‖) →  -- angle between a and b is 60°
  ‖a + 2 • b‖ = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l4028_402802


namespace NUMINAMATH_CALUDE_fifth_from_end_l4028_402859

-- Define the sequence
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 11

-- Define the final term
def final_term (a : ℕ → ℕ) (final : ℕ) : Prop :=
  ∃ k : ℕ, a k = final ∧ ∀ n > k, a n > final

-- Theorem statement
theorem fifth_from_end (a : ℕ → ℕ) :
  arithmetic_sequence a →
  final_term a 89 →
  ∃ k : ℕ, a k = 45 ∧ a (k + 4) = 89 :=
by sorry

end NUMINAMATH_CALUDE_fifth_from_end_l4028_402859


namespace NUMINAMATH_CALUDE_no_real_solution_l4028_402824

theorem no_real_solution : ¬ ∃ (x : ℝ), (3 / (x^2 - x - 6) = 2 / (x^2 - 3*x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l4028_402824


namespace NUMINAMATH_CALUDE_value_of_a_l4028_402809

theorem value_of_a (A B : Set ℝ) (a : ℝ) : 
  A = {0, 2, a} → 
  B = {1, a^2} → 
  A ∪ B = {0, 1, 2, 4, 16} → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l4028_402809


namespace NUMINAMATH_CALUDE_red_tetrahedron_volume_l4028_402843

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem red_tetrahedron_volume (cube_side : ℝ) (h : cube_side = 8) :
  let cube_volume : ℝ := cube_side ^ 3
  let small_tetrahedron_volume : ℝ := (1 / 3) * (cube_side ^ 2 / 2) * cube_side
  let red_tetrahedron_volume : ℝ := cube_volume - 4 * small_tetrahedron_volume
  red_tetrahedron_volume = 536 / 3 :=
by sorry

end NUMINAMATH_CALUDE_red_tetrahedron_volume_l4028_402843


namespace NUMINAMATH_CALUDE_tom_next_birthday_l4028_402858

-- Define the ages as real numbers
def tom_age : ℝ := sorry
def jerry_age : ℝ := sorry
def spike_age : ℝ := sorry

-- Define the relationships between ages
axiom jerry_spike_relation : jerry_age = 1.2 * spike_age
axiom tom_jerry_relation : tom_age = 0.7 * jerry_age

-- Define the sum of ages
axiom age_sum : tom_age + jerry_age + spike_age = 36

-- Theorem to prove
theorem tom_next_birthday : ⌊tom_age⌋ + 1 = 11 := by sorry

end NUMINAMATH_CALUDE_tom_next_birthday_l4028_402858


namespace NUMINAMATH_CALUDE_line_slope_angle_l4028_402833

theorem line_slope_angle (x y : ℝ) : 
  x + Real.sqrt 3 * y = 0 → 
  Real.tan (150 * π / 180) = -(1 / Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_angle_l4028_402833


namespace NUMINAMATH_CALUDE_laborer_income_l4028_402838

/-- Represents the monthly income of a laborer -/
def monthly_income : ℝ := 69

/-- Represents the average expenditure for the first 6 months -/
def first_6_months_expenditure : ℝ := 70

/-- Represents the reduced monthly expenditure for the next 4 months -/
def next_4_months_expenditure : ℝ := 60

/-- Represents the amount saved after 10 months -/
def amount_saved : ℝ := 30

/-- Theorem stating that the monthly income is 69 given the problem conditions -/
theorem laborer_income : 
  (6 * first_6_months_expenditure > 6 * monthly_income) ∧ 
  (4 * monthly_income = 4 * next_4_months_expenditure + (6 * first_6_months_expenditure - 6 * monthly_income) + amount_saved) →
  monthly_income = 69 := by
sorry

end NUMINAMATH_CALUDE_laborer_income_l4028_402838


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l4028_402892

theorem largest_integer_negative_quadratic : 
  ∀ n : ℤ, n^2 - 11*n + 24 < 0 → n ≤ 7 :=
by sorry

theorem seven_satisfies_inequality : 
  (7 : ℤ)^2 - 11*7 + 24 < 0 :=
by sorry

theorem eight_does_not_satisfy_inequality : 
  (8 : ℤ)^2 - 11*8 + 24 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l4028_402892


namespace NUMINAMATH_CALUDE_equation_solution_l4028_402846

theorem equation_solution : 
  ∃ (x : ℚ), x = -3/4 ∧ x/(x+1) = 3/x + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4028_402846


namespace NUMINAMATH_CALUDE_stratified_sampling_difference_l4028_402895

theorem stratified_sampling_difference (total_male : Nat) (total_female : Nat) (sample_size : Nat) : 
  total_male = 56 → 
  total_female = 42 → 
  sample_size = 28 → 
  (sample_size : ℚ) / ((total_male + total_female) : ℚ) = 2 / 7 → 
  (total_male : ℚ) * ((sample_size : ℚ) / ((total_male + total_female) : ℚ)) - 
  (total_female : ℚ) * ((sample_size : ℚ) / ((total_male + total_female) : ℚ)) = 4 := by
  sorry

#check stratified_sampling_difference

end NUMINAMATH_CALUDE_stratified_sampling_difference_l4028_402895


namespace NUMINAMATH_CALUDE_complement_of_union_l4028_402845

def S : Finset Nat := {1,2,3,4,5,6,7,8,9,10}
def A : Finset Nat := {2,4,6,8,10}
def B : Finset Nat := {3,6,9}

theorem complement_of_union (S A B : Finset Nat) :
  S = {1,2,3,4,5,6,7,8,9,10} →
  A = {2,4,6,8,10} →
  B = {3,6,9} →
  S \ (A ∪ B) = {1,5,7} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_union_l4028_402845


namespace NUMINAMATH_CALUDE_win_sectors_area_l4028_402884

theorem win_sectors_area (r : ℝ) (p : ℝ) : 
  r = 15 → p = 3/7 → (p * π * r^2) = 675*π/7 := by sorry

end NUMINAMATH_CALUDE_win_sectors_area_l4028_402884


namespace NUMINAMATH_CALUDE_right_triangle_area_l4028_402860

theorem right_triangle_area (a b c : ℝ) (h : a > 0) : 
  a * a = 2 * b * b →  -- 45-45-90 triangle condition
  b = 4 →              -- altitude to hypotenuse is 4
  c = a / 2 →          -- c is half of hypotenuse
  (1/2) * a * b = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l4028_402860


namespace NUMINAMATH_CALUDE_tiling_cost_difference_l4028_402854

/-- Represents a tiling option with its cost per tile and labor cost per square foot -/
structure TilingOption where
  tileCost : ℕ
  laborCost : ℕ

/-- Calculates the total cost for a tiling option -/
def totalCost (option : TilingOption) (totalArea : ℕ) (tilesPerSqFt : ℕ) : ℕ :=
  option.tileCost * totalArea * tilesPerSqFt + option.laborCost * totalArea

theorem tiling_cost_difference :
  let turquoise := TilingOption.mk 13 6
  let purple := TilingOption.mk 11 8
  let orange := TilingOption.mk 15 5
  let totalArea := 5 * 8 + 7 * 8 + 6 * 9
  let tilesPerSqFt := 4
  let turquoiseCost := totalCost turquoise totalArea tilesPerSqFt
  let purpleCost := totalCost purple totalArea tilesPerSqFt
  let orangeCost := totalCost orange totalArea tilesPerSqFt
  max turquoiseCost (max purpleCost orangeCost) - min turquoiseCost (min purpleCost orangeCost) = 1950 := by
  sorry

end NUMINAMATH_CALUDE_tiling_cost_difference_l4028_402854


namespace NUMINAMATH_CALUDE_max_sum_problem_l4028_402844

def is_valid_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

theorem max_sum_problem (A B C D : ℕ) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_valid : is_valid_digit A ∧ is_valid_digit B ∧ is_valid_digit C ∧ is_valid_digit D)
  (h_integer : ∃ k : ℕ, k * (C + D) = A + B + 1)
  (h_max : ∀ A' B' C' D' : ℕ, 
    A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ B' ≠ C' ∧ B' ≠ D' ∧ C' ≠ D' →
    is_valid_digit A' ∧ is_valid_digit B' ∧ is_valid_digit C' ∧ is_valid_digit D' →
    (∃ k' : ℕ, k' * (C' + D') = A' + B' + 1) →
    (A' + B' + 1) / (C' + D') ≤ (A + B + 1) / (C + D)) :
  A + B + 1 = 18 := by
sorry

end NUMINAMATH_CALUDE_max_sum_problem_l4028_402844


namespace NUMINAMATH_CALUDE_orange_boxes_l4028_402819

theorem orange_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 2650) (h2 : oranges_per_box = 10) :
  total_oranges / oranges_per_box = 265 := by
  sorry

end NUMINAMATH_CALUDE_orange_boxes_l4028_402819


namespace NUMINAMATH_CALUDE_toll_constant_value_l4028_402872

/-- Represents the toll formula for a bridge -/
def toll_formula (constant : ℝ) (x : ℕ) : ℝ := constant + 0.50 * (x - 2)

/-- Calculates the number of axles for a truck given its wheel configuration -/
def calculate_axles (front_wheels : ℕ) (other_wheels : ℕ) : ℕ :=
  1 + (other_wheels / 4)

theorem toll_constant_value :
  ∃ (constant : ℝ),
    let x := calculate_axles 2 16
    toll_formula constant x = 4 ∧ constant = 2.50 := by
  sorry

end NUMINAMATH_CALUDE_toll_constant_value_l4028_402872


namespace NUMINAMATH_CALUDE_father_son_age_sum_l4028_402801

/-- Represents the ages of a father and son pair -/
structure FatherSonAges where
  father : ℕ
  son : ℕ

/-- The sum of father's and son's ages after a given number of years -/
def ageSum (ages : FatherSonAges) (years : ℕ) : ℕ :=
  ages.father + ages.son + 2 * years

theorem father_son_age_sum :
  ∀ (ages : FatherSonAges),
    ages.father + ages.son = 55 →
    ages.father = 37 →
    ages.son = 18 →
    ageSum ages (ages.father - ages.son) = 93 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_sum_l4028_402801


namespace NUMINAMATH_CALUDE_sqrt_fraction_difference_l4028_402841

theorem sqrt_fraction_difference : Real.sqrt (9/4) - Real.sqrt (4/9) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_difference_l4028_402841


namespace NUMINAMATH_CALUDE_three_digit_sum_l4028_402816

theorem three_digit_sum (S X Z : Nat) 
  (h : (100 * S + 10 * X + Z) * (100 * Z + 10 * X + S) = 78445) :
  (100 * S + 10 * X + Z) + (100 * Z + 10 * X + S) = 1372 := by
  sorry

#check three_digit_sum

end NUMINAMATH_CALUDE_three_digit_sum_l4028_402816


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_sequence_l4028_402862

theorem right_triangle_arithmetic_sequence (a b c : ℕ) : 
  a < b ∧ b < c →                        -- sides form an increasing sequence
  a + b + c = 840 →                      -- perimeter is 840
  b - a = c - b →                        -- sides form an arithmetic sequence
  a^2 + b^2 = c^2 →                      -- it's a right triangle (Pythagorean theorem)
  c = 350 := by sorry                    -- largest side is 350

end NUMINAMATH_CALUDE_right_triangle_arithmetic_sequence_l4028_402862


namespace NUMINAMATH_CALUDE_remaining_payment_l4028_402867

theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (h1 : deposit = 150) (h2 : deposit_percentage = 0.1) : 
  (deposit / deposit_percentage) - deposit = 1350 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_l4028_402867


namespace NUMINAMATH_CALUDE_ellipse_equation_l4028_402880

/-- The equation of an ellipse with given properties -/
theorem ellipse_equation (e : ℝ) (c : ℝ) (h1 : e = 1/2) (h2 : c = 1) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), (x^2 / (a^2) + y^2 / (b^2) = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4028_402880


namespace NUMINAMATH_CALUDE_son_age_is_eighteen_l4028_402847

/-- Represents the ages of a father and son -/
structure FatherSonAges where
  fatherAge : ℕ
  sonAge : ℕ

/-- The condition that the father is 20 years older than the son -/
def ageDifference (ages : FatherSonAges) : Prop :=
  ages.fatherAge = ages.sonAge + 20

/-- The condition that in two years, the father's age will be twice the son's age -/
def futureAgeRelation (ages : FatherSonAges) : Prop :=
  ages.fatherAge + 2 = 2 * (ages.sonAge + 2)

/-- Theorem stating that given the conditions, the son's present age is 18 -/
theorem son_age_is_eighteen (ages : FatherSonAges) 
  (h1 : ageDifference ages) (h2 : futureAgeRelation ages) : ages.sonAge = 18 := by
  sorry

end NUMINAMATH_CALUDE_son_age_is_eighteen_l4028_402847


namespace NUMINAMATH_CALUDE_sandy_grew_eight_carrots_l4028_402807

/-- The number of carrots Sandy grew -/
def sandys_carrots : ℕ := sorry

/-- The number of carrots Mary grew -/
def marys_carrots : ℕ := 6

/-- The total number of carrots grown by Sandy and Mary -/
def total_carrots : ℕ := 14

/-- Theorem stating that Sandy grew 8 carrots -/
theorem sandy_grew_eight_carrots : sandys_carrots = 8 := by
  sorry

end NUMINAMATH_CALUDE_sandy_grew_eight_carrots_l4028_402807


namespace NUMINAMATH_CALUDE_reading_time_difference_l4028_402886

/-- Proves the difference in reading time between two people for a given book -/
theorem reading_time_difference
  (xanthia_speed : ℕ)  -- Xanthia's reading speed in pages per hour
  (molly_speed : ℕ)    -- Molly's reading speed in pages per hour
  (book_pages : ℕ)     -- Number of pages in the book
  (h1 : xanthia_speed = 120)
  (h2 : molly_speed = 60)
  (h3 : book_pages = 360) :
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 180 :=
by sorry

end NUMINAMATH_CALUDE_reading_time_difference_l4028_402886


namespace NUMINAMATH_CALUDE_cos_product_range_in_triangle_l4028_402869

theorem cos_product_range_in_triangle (A B C : ℝ) : 
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ 
  A + B + C = π ∧ 
  B = π / 3 → 
  -1/2 ≤ Real.cos A * Real.cos C ∧ Real.cos A * Real.cos C ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_cos_product_range_in_triangle_l4028_402869


namespace NUMINAMATH_CALUDE_prism_volume_l4028_402810

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the sum of all edge lengths of a rectangular prism -/
def sumOfEdges (d : PrismDimensions) : ℝ :=
  4 * (d.length + d.width + d.height)

/-- Calculates the volume of a rectangular prism -/
def volume (d : PrismDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: The volume of a rectangular prism with edges in ratio 3:2:1 
    and sum of all edge lengths 72 cm is 162 cubic centimeters -/
theorem prism_volume (d : PrismDimensions) 
    (h_ratio : d.length = 3 * d.height ∧ d.width = 2 * d.height) 
    (h_sum : sumOfEdges d = 72) : 
    volume d = 162 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l4028_402810


namespace NUMINAMATH_CALUDE_bart_mixtape_second_side_l4028_402885

def mixtape (first_side_songs : ℕ) (song_length : ℕ) (total_length : ℕ) : ℕ :=
  (total_length - first_side_songs * song_length) / song_length

theorem bart_mixtape_second_side :
  mixtape 6 4 40 = 4 := by
  sorry

end NUMINAMATH_CALUDE_bart_mixtape_second_side_l4028_402885


namespace NUMINAMATH_CALUDE_grass_area_in_square_plot_l4028_402800

theorem grass_area_in_square_plot (perimeter : ℝ) (h_perimeter : perimeter = 40) :
  let side_length := perimeter / 4
  let square_area := side_length ^ 2
  let circle_radius := side_length / 2
  let circle_area := π * circle_radius ^ 2
  let grass_area := square_area - circle_area
  grass_area = 100 - 25 * π :=
by sorry

end NUMINAMATH_CALUDE_grass_area_in_square_plot_l4028_402800


namespace NUMINAMATH_CALUDE_vector_decomposition_l4028_402871

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![(-2), 4, 7]
def p : Fin 3 → ℝ := ![0, 1, 2]
def q : Fin 3 → ℝ := ![1, 0, 1]
def r : Fin 3 → ℝ := ![(-1), 2, 4]

/-- Theorem: x can be decomposed as 2p - q + r -/
theorem vector_decomposition :
  x = fun i => 2 * p i - q i + r i := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l4028_402871


namespace NUMINAMATH_CALUDE_point_P_coordinates_l4028_402822

def M : ℝ × ℝ := (2, 2)
def N : ℝ × ℝ := (5, -2)

def is_right_angle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

theorem point_P_coordinates :
  ∀ x : ℝ,
    let P : ℝ × ℝ := (x, 0)
    is_right_angle M P N → x = 1 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l4028_402822


namespace NUMINAMATH_CALUDE_phone_number_guess_probability_l4028_402864

theorem phone_number_guess_probability : 
  ∀ (total_digits : ℕ) (correct_digit : ℕ),
  total_digits = 10 →
  correct_digit < total_digits →
  (1 - 1 / total_digits) * (1 / (total_digits - 1)) = 1 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_phone_number_guess_probability_l4028_402864


namespace NUMINAMATH_CALUDE_second_hour_distance_l4028_402813

/-- Represents a 3-hour bike ride with specific distance relationships --/
structure BikeRide where
  first_hour : ℝ
  second_hour : ℝ
  third_hour : ℝ
  second_hour_condition : second_hour = 1.2 * first_hour
  third_hour_condition : third_hour = 1.25 * second_hour
  total_distance : first_hour + second_hour + third_hour = 37

/-- Theorem stating that the distance traveled in the second hour is 12 miles --/
theorem second_hour_distance (ride : BikeRide) : ride.second_hour = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_hour_distance_l4028_402813


namespace NUMINAMATH_CALUDE_expression_value_l4028_402808

theorem expression_value : 100 * (100 - 3) - (100 * 100 - 3) = -297 := by sorry

end NUMINAMATH_CALUDE_expression_value_l4028_402808


namespace NUMINAMATH_CALUDE_inequality_proof_l4028_402803

theorem inequality_proof (a b c d e f : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e) (h6 : e ≤ f) :
  (a * f + b * e + c * d) * (a * f + b * d + c * e) ≤ (a + b^2 + c^3) * (d + e^2 + f^3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4028_402803


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l4028_402875

theorem arithmetic_evaluation : 68 + (108 * 3) + (29^2) - 310 - (6 * 9) = 869 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l4028_402875


namespace NUMINAMATH_CALUDE_pizza_pieces_per_pizza_pizza_pieces_theorem_l4028_402857

theorem pizza_pieces_per_pizza 
  (num_students : ℕ) 
  (pizzas_per_student : ℕ) 
  (total_pieces : ℕ) : ℕ :=
  let total_pizzas := num_students * pizzas_per_student
  total_pieces / total_pizzas

#check pizza_pieces_per_pizza 10 20 1200 = 6

-- Proof
theorem pizza_pieces_theorem :
  pizza_pieces_per_pizza 10 20 1200 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pieces_per_pizza_pizza_pieces_theorem_l4028_402857


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l4028_402827

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 40*x^2 + 400 = 0 ∧
  x = -2 * Real.sqrt 5 ∧
  ∀ (y : ℝ), y^4 - 40*y^2 + 400 = 0 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l4028_402827


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l4028_402861

-- Define the set G
def G : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.2 ∧ p.2 ≤ 8 ∧ (p.1 - 3)^2 + 31 = (p.2 - 4)^2 + 8 * Real.sqrt (p.2 * (8 - p.2))}

-- Define the tangent line condition
def isTangentLine (m : ℝ) (b : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = m * p.1 + b ∧ p ∈ G ∧
  ∀ q : ℝ × ℝ, q ∈ G → q.2 ≤ m * q.1 + b

-- Theorem statement
theorem tangent_point_coordinates :
  ∃! p : ℝ × ℝ, p ∈ G ∧ 
    ∃ m : ℝ, m < 0 ∧ 
      isTangentLine m 4 p ∧
      p = (12/5, 8/5) := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l4028_402861


namespace NUMINAMATH_CALUDE_set_operations_l4028_402868

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}

def B : Set ℝ := {x | -1 < x ∧ x < 5}

theorem set_operations :
  (A ∪ B = {x | -3 ≤ x ∧ x < 5}) ∧
  (A ∩ B = {x | -1 < x ∧ x ≤ 4}) ∧
  ((U \ A) ∩ B = {x | 4 < x ∧ x < 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l4028_402868


namespace NUMINAMATH_CALUDE_east_northwest_angle_l4028_402898

/-- Given a circle with ten equally spaced rays, where one ray points due North,
    the smaller angle between the rays pointing East and Northwest is 36°. -/
theorem east_northwest_angle (n : ℕ) (ray_angle : ℝ) : 
  n = 10 ∧ ray_angle = 360 / n → 36 = ray_angle := by sorry

end NUMINAMATH_CALUDE_east_northwest_angle_l4028_402898


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l4028_402825

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  Nat.factorial total / Nat.factorial identical

/-- Theorem: Arranging 6 books with 3 identical copies results in 120 ways -/
theorem book_arrangement_theorem :
  arrange_books 6 3 = 120 := by
  sorry

#eval arrange_books 6 3

end NUMINAMATH_CALUDE_book_arrangement_theorem_l4028_402825


namespace NUMINAMATH_CALUDE_specific_frustum_smaller_cone_altitude_l4028_402811

/-- Represents a frustum of a right circular cone. -/
structure Frustum where
  altitude : ℝ
  largerBaseArea : ℝ
  smallerBaseArea : ℝ

/-- Calculates the altitude of the smaller cone removed from a frustum. -/
def smallerConeAltitude (f : Frustum) : ℝ :=
  sorry

/-- Theorem stating that for a specific frustum, the altitude of the smaller cone is 15. -/
theorem specific_frustum_smaller_cone_altitude :
  let f : Frustum := { altitude := 15, largerBaseArea := 64 * Real.pi, smallerBaseArea := 16 * Real.pi }
  smallerConeAltitude f = 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_frustum_smaller_cone_altitude_l4028_402811


namespace NUMINAMATH_CALUDE_profit_percentage_example_l4028_402812

/-- Calculates the percentage of profit given the cost price and selling price -/
def percentage_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that for a cost price of $600 and a selling price of $648, the percentage profit is 8% -/
theorem profit_percentage_example :
  percentage_profit 600 648 = 8 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_example_l4028_402812


namespace NUMINAMATH_CALUDE_part_1_part_2_l4028_402879

-- Define the circle C
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define Line 1
def line_1 (x y : ℝ) : Prop := 3*x + 4*y - 6 = 0

-- Define Line 2
def line_2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Part I
theorem part_1 (x y m : ℝ) (M N : ℝ × ℝ) :
  circle_C x y m →
  line_1 (M.1) (M.2) →
  line_1 (N.1) (N.2) →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 12 →
  m = 1 :=
sorry

-- Part II
theorem part_2 :
  ∃ m : ℝ, m = -2 ∧
  ∃ A B : ℝ × ℝ,
    circle_C A.1 A.2 m ∧
    circle_C B.1 B.2 m ∧
    line_2 A.1 A.2 ∧
    line_2 B.1 B.2 ∧
    (A.1 * B.1 + A.2 * B.2 = 0) :=
sorry

end NUMINAMATH_CALUDE_part_1_part_2_l4028_402879


namespace NUMINAMATH_CALUDE_rational_function_identity_l4028_402835

def is_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n

theorem rational_function_identity 
  (f : ℚ → ℚ) 
  (h1 : ∃ a : ℚ, ¬is_integer (f a))
  (h2 : ∀ x y : ℚ, is_integer (f (x + y) - f x - f y))
  (h3 : ∀ x y : ℚ, is_integer (f (x * y) - f x * f y)) :
  ∀ x : ℚ, f x = x :=
sorry

end NUMINAMATH_CALUDE_rational_function_identity_l4028_402835


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l4028_402893

theorem x_squared_plus_reciprocal (x : ℝ) (h : x^4 + 1/x^4 = 240) : 
  x^2 + 1/x^2 = Real.sqrt 242 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l4028_402893


namespace NUMINAMATH_CALUDE_triple_tangent_identity_l4028_402823

theorem triple_tangent_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  (3 * x - x^3) / (1 - 3 * x^2) + (3 * y - y^3) / (1 - 3 * y^2) + (3 * z - z^3) / (1 - 3 * z^2) =
  ((3 * x - x^3) / (1 - 3 * x^2)) * ((3 * y - y^3) / (1 - 3 * y^2)) * ((3 * z - z^3) / (1 - 3 * z^2)) := by
  sorry

end NUMINAMATH_CALUDE_triple_tangent_identity_l4028_402823


namespace NUMINAMATH_CALUDE_systematic_sample_property_fourth_student_number_l4028_402882

/-- Represents a systematic sample from a population --/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  start : ℕ

/-- Calculates the nth element in a systematic sample --/
def nth_element (s : SystematicSample) (n : ℕ) : ℕ :=
  ((s.start + (n - 1) * s.interval - 1) % s.population_size) + 1

/-- Theorem stating the properties of the given systematic sample --/
theorem systematic_sample_property (s : SystematicSample) : 
  s.population_size = 54 ∧ 
  s.sample_size = 4 ∧ 
  s.start = 2 ∧ 
  nth_element s 2 = 28 ∧ 
  nth_element s 3 = 41 →
  nth_element s 4 = 1 := by
  sorry

/-- Main theorem to prove --/
theorem fourth_student_number : 
  ∃ (s : SystematicSample), 
    s.population_size = 54 ∧ 
    s.sample_size = 4 ∧ 
    s.start = 2 ∧ 
    nth_element s 2 = 28 ∧ 
    nth_element s 3 = 41 ∧
    nth_element s 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_property_fourth_student_number_l4028_402882


namespace NUMINAMATH_CALUDE_prize_cost_l4028_402887

theorem prize_cost (total_cost : ℕ) (num_prizes : ℕ) (cost_per_prize : ℕ) 
  (h1 : total_cost = 120)
  (h2 : num_prizes = 6)
  (h3 : total_cost = num_prizes * cost_per_prize) :
  cost_per_prize = 20 := by
  sorry

end NUMINAMATH_CALUDE_prize_cost_l4028_402887


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_l4028_402888

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- A line passing through the focus -/
def line_through_focus (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x - focus.1) + focus.2

/-- Theorem: For the parabola y^2 = 4x, if a line passing through its focus
    intersects the parabola at points A(x₁, y₁) and B(x₂, y₂), and x₁ + x₂ = 6,
    then the distance between A and B is 8. -/
theorem parabola_intersection_distance
  (x₁ y₁ x₂ y₂ m : ℝ)
  (h₁ : parabola x₁ y₁)
  (h₂ : parabola x₂ y₂)
  (h₃ : line_through_focus m x₁ y₁)
  (h₄ : line_through_focus m x₂ y₂)
  (h₅ : x₁ + x₂ = 6) :
  ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 64 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_l4028_402888


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l4028_402852

theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n → 1008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l4028_402852


namespace NUMINAMATH_CALUDE_counterexample_exists_l4028_402853

theorem counterexample_exists : ∃ (a b c : ℤ), a > b ∧ b > c ∧ a + b ≤ c := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l4028_402853


namespace NUMINAMATH_CALUDE_parabola_c_value_l4028_402877

/-- A parabola with equation y = 2x^2 + bx + c passes through the points (1,5) and (5,5). 
    The value of c is 15. -/
theorem parabola_c_value : ∀ b c : ℝ, 
  (5 = 2 * (1 : ℝ)^2 + b * 1 + c) → 
  (5 = 2 * (5 : ℝ)^2 + b * 5 + c) → 
  c = 15 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l4028_402877


namespace NUMINAMATH_CALUDE_unique_n_divisible_by_11_l4028_402820

theorem unique_n_divisible_by_11 : ∃! n : ℕ, 0 < n ∧ n < 11 ∧ (18888 - n) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_divisible_by_11_l4028_402820


namespace NUMINAMATH_CALUDE_regular_hexagon_diagonal_l4028_402855

theorem regular_hexagon_diagonal (side_length : ℝ) (h : side_length = 10) :
  let diagonal_length := side_length * Real.sqrt 3
  diagonal_length = 10 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_regular_hexagon_diagonal_l4028_402855


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l4028_402834

theorem simple_interest_rate_calculation 
  (principal : ℝ) 
  (time : ℝ) 
  (interest : ℝ) 
  (h1 : principal = 10000) 
  (h2 : time = 1) 
  (h3 : interest = 800) : 
  (interest / (principal * time)) * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l4028_402834


namespace NUMINAMATH_CALUDE_geometric_sum_five_terms_l4028_402832

theorem geometric_sum_five_terms (a r : ℚ) (h1 : a = 1/4) (h2 : r = 1/4) :
  let S := a + a*r + a*r^2 + a*r^3 + a*r^4
  S = 341/1024 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_five_terms_l4028_402832


namespace NUMINAMATH_CALUDE_ferris_wheel_ticket_cost_l4028_402804

theorem ferris_wheel_ticket_cost 
  (initial_tickets : ℕ) 
  (remaining_tickets : ℕ) 
  (total_spent : ℕ) 
  (h1 : initial_tickets = 13)
  (h2 : remaining_tickets = 4)
  (h3 : total_spent = 81) :
  total_spent / (initial_tickets - remaining_tickets) = 9 := by
sorry

end NUMINAMATH_CALUDE_ferris_wheel_ticket_cost_l4028_402804


namespace NUMINAMATH_CALUDE_parabola_dot_product_zero_l4028_402829

/-- A point on the parabola y^2 = 4x -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- The line passing through two points intersects (4,0) -/
def line_through_four (A B : ParabolaPoint) : Prop :=
  ∃ t : ℝ, A.x + t * (B.x - A.x) = 4 ∧ A.y + t * (B.y - A.y) = 0

/-- The dot product of vectors OA and OB -/
def dot_product (A B : ParabolaPoint) : ℝ :=
  A.x * B.x + A.y * B.y

theorem parabola_dot_product_zero (A B : ParabolaPoint) 
  (h : line_through_four A B) : dot_product A B = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_dot_product_zero_l4028_402829


namespace NUMINAMATH_CALUDE_original_apples_in_B_l4028_402881

/-- Represents the number of apples in each basket -/
structure AppleBaskets where
  A : ℕ  -- Number of apples in basket A
  B : ℕ  -- Number of apples in basket B
  C : ℕ  -- Number of apples in basket C

/-- The conditions of the apple basket problem -/
def apple_basket_conditions (baskets : AppleBaskets) : Prop :=
  -- Condition 1: The number of apples in basket C is twice the number of apples in basket A
  baskets.C = 2 * baskets.A ∧
  -- Condition 2: After transferring 12 apples from B to A, A has 24 less than C
  baskets.A + 12 = baskets.C - 24 ∧
  -- Condition 3: After the transfer, B has 6 more than C
  baskets.B - 12 = baskets.C + 6

theorem original_apples_in_B (baskets : AppleBaskets) :
  apple_basket_conditions baskets → baskets.B = 90 := by
  sorry

end NUMINAMATH_CALUDE_original_apples_in_B_l4028_402881


namespace NUMINAMATH_CALUDE_solution_value_l4028_402873

-- Define the function E
def E (a b c : ℝ) : ℝ := a * b^2 + c

-- State the theorem
theorem solution_value : ∃ a : ℝ, 2*a + E a 3 2 = 4 + E a 5 3 ∧ a = -5/14 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l4028_402873


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l4028_402848

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  n > 1 ∧
  n % 3 = 2 ∧
  n % 7 = 2 ∧
  n % 8 = 2 ∧
  (∀ m : ℕ, m > 1 ∧ m % 3 = 2 ∧ m % 7 = 2 ∧ m % 8 = 2 → n ≤ m) ∧
  n = 170 ∧
  131 ≤ n ∧ n ≤ 170 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l4028_402848


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l4028_402817

theorem cubic_equation_real_root (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) * (x + 2) := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_real_root_l4028_402817


namespace NUMINAMATH_CALUDE_point_translation_l4028_402828

def initial_point : ℝ × ℝ := (0, 1)
def downward_translation : ℝ := 2
def leftward_translation : ℝ := 4

theorem point_translation :
  (initial_point.1 - leftward_translation, initial_point.2 - downward_translation) = (-4, -1) := by
  sorry

end NUMINAMATH_CALUDE_point_translation_l4028_402828


namespace NUMINAMATH_CALUDE_no_two_digit_sum_with_reverse_is_cube_l4028_402831

/-- Function to reverse the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Function to check if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

/-- Theorem: No two-digit positive integer N has the property that
    the sum of N and its digit-reversed number is a perfect cube -/
theorem no_two_digit_sum_with_reverse_is_cube :
  ¬∃ N : ℕ, 10 ≤ N ∧ N < 100 ∧ isPerfectCube (N + reverseDigits N) := by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_sum_with_reverse_is_cube_l4028_402831


namespace NUMINAMATH_CALUDE_bike_price_calculation_l4028_402850

theorem bike_price_calculation (upfront_payment : ℝ) (upfront_percentage : ℝ) 
  (h1 : upfront_payment = 200)
  (h2 : upfront_percentage = 0.20) :
  upfront_payment / upfront_percentage = 1000 := by
  sorry

end NUMINAMATH_CALUDE_bike_price_calculation_l4028_402850


namespace NUMINAMATH_CALUDE_girls_walking_time_l4028_402866

/-- The time taken for two girls walking in opposite directions to be 120 km apart -/
theorem girls_walking_time (speed1 speed2 distance : ℝ) (h1 : speed1 = 7)
  (h2 : speed2 = 3) (h3 : distance = 120) : 
  distance / (speed1 + speed2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_girls_walking_time_l4028_402866


namespace NUMINAMATH_CALUDE_next_shared_meeting_l4028_402865

/-- Represents the number of days between meetings for each group -/
def drama_club_cycle : ℕ := 3
def choir_cycle : ℕ := 5
def debate_team_cycle : ℕ := 7

/-- Theorem stating that the next shared meeting will occur in 105 days -/
theorem next_shared_meeting :
  ∃ (n : ℕ), n > 0 ∧ 
  n % drama_club_cycle = 0 ∧
  n % choir_cycle = 0 ∧
  n % debate_team_cycle = 0 ∧
  ∀ (m : ℕ), m > 0 ∧ 
    m % drama_club_cycle = 0 ∧
    m % choir_cycle = 0 ∧
    m % debate_team_cycle = 0 →
    n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_next_shared_meeting_l4028_402865


namespace NUMINAMATH_CALUDE_binomial_60_3_l4028_402842

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l4028_402842


namespace NUMINAMATH_CALUDE_asterisk_replacement_l4028_402836

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 18) * (x / 162) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l4028_402836


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l4028_402878

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem z_in_second_quadrant :
  (1 + i) * z = -1 →
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l4028_402878


namespace NUMINAMATH_CALUDE_prism_volume_l4028_402814

/-- A right rectangular prism with face areas 18, 32, and 48 square inches has a volume of 288 cubic inches. -/
theorem prism_volume (l w h : ℝ) 
  (area1 : l * w = 18) 
  (area2 : w * h = 32) 
  (area3 : l * h = 48) : 
  l * w * h = 288 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l4028_402814


namespace NUMINAMATH_CALUDE_sin_300_degrees_l4028_402830

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l4028_402830


namespace NUMINAMATH_CALUDE_equation_solutions_l4028_402805

theorem equation_solutions : 
  ∀ n m : ℕ+, 3 * 2^(m : ℕ) + 1 = (n : ℕ)^2 ↔ (n = 7 ∧ m = 4) ∨ (n = 5 ∧ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4028_402805


namespace NUMINAMATH_CALUDE_great_wall_scientific_notation_l4028_402890

theorem great_wall_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 6700000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 6.7 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_great_wall_scientific_notation_l4028_402890


namespace NUMINAMATH_CALUDE_hannahs_remaining_money_l4028_402840

/-- The problem of calculating Hannah's remaining money after selling cookies and cupcakes and buying measuring spoons. -/
theorem hannahs_remaining_money :
  let cookie_count : ℕ := 40
  let cookie_price : ℚ := 4/5  -- $0.8 expressed as a rational number
  let cupcake_count : ℕ := 30
  let cupcake_price : ℚ := 2
  let spoon_set_count : ℕ := 2
  let spoon_set_price : ℚ := 13/2  -- $6.5 expressed as a rational number
  
  let total_sales := cookie_count * cookie_price + cupcake_count * cupcake_price
  let total_spent := spoon_set_count * spoon_set_price
  let remaining_money := total_sales - total_spent

  remaining_money = 79 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_remaining_money_l4028_402840


namespace NUMINAMATH_CALUDE_block3_can_reach_target_l4028_402876

-- Define the board
def Board := Fin 3 × Fin 7

-- Define a block
structure Block where
  label : Nat
  position : Board

-- Define the game state
structure GameState where
  blocks : List Block

-- Define a valid move
inductive Move
| Up : Block → Move
| Down : Block → Move
| Left : Block → Move
| Right : Block → Move

-- Define the initial game state
def initialState : GameState := {
  blocks := [
    { label := 1, position := ⟨2, 2⟩ },
    { label := 2, position := ⟨3, 5⟩ },
    { label := 3, position := ⟨1, 4⟩ }
  ]
}

-- Define the target position
def targetPosition : Board := ⟨2, 4⟩

-- Function to check if a move is valid
def isValidMove (state : GameState) (move : Move) : Bool := sorry

-- Function to apply a move to the game state
def applyMove (state : GameState) (move : Move) : GameState := sorry

-- Theorem: There exists a sequence of valid moves to bring Block 3 to the target position
theorem block3_can_reach_target :
  ∃ (moves : List Move), 
    let finalState := moves.foldl (λ s m => applyMove s m) initialState
    (finalState.blocks.find? (λ b => b.label = 3)).map (λ b => b.position) = some targetPosition :=
sorry

end NUMINAMATH_CALUDE_block3_can_reach_target_l4028_402876


namespace NUMINAMATH_CALUDE_jills_age_l4028_402863

/-- Given that the sum of Henry and Jill's present ages is 40,
    and 11 years ago Henry was twice the age of Jill,
    prove that Jill's present age is 17 years. -/
theorem jills_age (henry_age jill_age : ℕ) 
  (sum_ages : henry_age + jill_age = 40)
  (past_relation : henry_age - 11 = 2 * (jill_age - 11)) :
  jill_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_jills_age_l4028_402863


namespace NUMINAMATH_CALUDE_total_letters_in_seven_hours_l4028_402870

def letters_per_hour (nathan jacob emily : ℕ) : ℕ :=
  nathan + jacob + emily

theorem total_letters_in_seven_hours 
  (nathan_speed : ℕ) 
  (h1 : nathan_speed = 25)
  (jacob_speed : ℕ) 
  (h2 : jacob_speed = 2 * nathan_speed)
  (emily_speed : ℕ) 
  (h3 : emily_speed = 3 * nathan_speed)
  : letters_per_hour nathan_speed jacob_speed emily_speed * 7 = 1050 :=
by
  sorry

end NUMINAMATH_CALUDE_total_letters_in_seven_hours_l4028_402870
