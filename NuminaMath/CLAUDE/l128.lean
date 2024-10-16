import Mathlib

namespace NUMINAMATH_CALUDE_weight_of_CCl4_l128_12821

/-- The molar mass of Carbon in g/mol -/
def molar_mass_C : ℝ := 12.01

/-- The molar mass of Chlorine in g/mol -/
def molar_mass_Cl : ℝ := 35.45

/-- The number of Carbon atoms in a CCl4 molecule -/
def num_C_atoms : ℕ := 1

/-- The number of Chlorine atoms in a CCl4 molecule -/
def num_Cl_atoms : ℕ := 4

/-- The number of moles of CCl4 -/
def num_moles : ℝ := 8

/-- Theorem: The weight of 8 moles of CCl4 is 1230.48 grams -/
theorem weight_of_CCl4 : 
  let molar_mass_CCl4 := molar_mass_C * num_C_atoms + molar_mass_Cl * num_Cl_atoms
  num_moles * molar_mass_CCl4 = 1230.48 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_CCl4_l128_12821


namespace NUMINAMATH_CALUDE_equation_solutions_l128_12820

/-- The equation we're solving -/
def equation (x : ℂ) : Prop :=
  x ≠ -2 ∧ (15*x - x^2)/(x + 2) * (x + (15 - x)/(x + 2)) = 48

/-- The set of solutions to the equation -/
def solutions : Set ℂ :=
  { x | x = 12 + 2*Real.sqrt 38 ∨
        x = 12 - 2*Real.sqrt 38 ∨
        x = -1/2 + Complex.I*(Real.sqrt 95)/2 ∨
        x = -1/2 - Complex.I*(Real.sqrt 95)/2 }

/-- Theorem stating that the solutions are correct and complete -/
theorem equation_solutions :
  ∀ x, equation x ↔ x ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l128_12820


namespace NUMINAMATH_CALUDE_matrix_power_difference_l128_12860

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem matrix_power_difference :
  B^20 - 3 • B^19 = !![0, 4 * 2^19; 0, -2^19] := by sorry

end NUMINAMATH_CALUDE_matrix_power_difference_l128_12860


namespace NUMINAMATH_CALUDE_absolute_value_equation_range_l128_12840

theorem absolute_value_equation_range (x : ℝ) : 
  |x - 1| + x - 1 = 0 → x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_range_l128_12840


namespace NUMINAMATH_CALUDE_perfect_square_sum_existence_l128_12827

theorem perfect_square_sum_existence : ∃ (x y z u v w t s : ℕ+), 
  x^2 + y + z + u = (x + v)^2 ∧
  y^2 + x + z + u = (y + w)^2 ∧
  z^2 + x + y + u = (z + t)^2 ∧
  u^2 + x + y + z = (u + s)^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_sum_existence_l128_12827


namespace NUMINAMATH_CALUDE_f_composition_and_domain_l128_12812

def f (x : ℝ) : ℝ := x + 5

theorem f_composition_and_domain :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f x ∈ Set.Icc 2 7) →
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, ∀ y ∈ Set.Icc (-3 : ℝ) 2, x < y → f x < f y) →
  (∀ x ∈ Set.Icc (-8 : ℝ) (-3), f (f x) = x + 10) ∧
  (∀ x, f (f x) ∈ Set.Icc 2 7 ↔ x ∈ Set.Icc (-8 : ℝ) (-3)) := by
  sorry

#check f_composition_and_domain

end NUMINAMATH_CALUDE_f_composition_and_domain_l128_12812


namespace NUMINAMATH_CALUDE_right_triangle_sin_A_l128_12835

/-- In a right triangle ABC with ∠C = 90°, where sides opposite to angles A, B, and C are a, b, and c respectively, sin A = a/c -/
theorem right_triangle_sin_A (A B C : ℝ) (a b c : ℝ) 
  (h_right : A + B + C = Real.pi)
  (h_C : C = Real.pi / 2)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_pythagorean : a^2 + b^2 = c^2) :
  Real.sin A = a / c := by sorry

end NUMINAMATH_CALUDE_right_triangle_sin_A_l128_12835


namespace NUMINAMATH_CALUDE_largest_x_satisfying_equation_l128_12844

theorem largest_x_satisfying_equation : 
  ∃ (x : ℝ), x = 3/25 ∧ 
  (∀ y : ℝ, y ≥ 0 → Real.sqrt (3*y) = 5*y → y ≤ x) ∧
  Real.sqrt (3*(3/25)) = 5*(3/25) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_satisfying_equation_l128_12844


namespace NUMINAMATH_CALUDE_age_difference_l128_12896

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 16) : A - C = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l128_12896


namespace NUMINAMATH_CALUDE_matrix_product_l128_12882

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 4]

theorem matrix_product :
  A * B = !![17, -5; 16, -20] := by sorry

end NUMINAMATH_CALUDE_matrix_product_l128_12882


namespace NUMINAMATH_CALUDE_remainder_5_pow_2021_mod_17_l128_12826

theorem remainder_5_pow_2021_mod_17 : 5^2021 % 17 = 14 := by
  sorry

end NUMINAMATH_CALUDE_remainder_5_pow_2021_mod_17_l128_12826


namespace NUMINAMATH_CALUDE_equation_system_solution_l128_12831

theorem equation_system_solution (a b c : ℝ) : 
  (∀ x y : ℝ, a * x + y = 5 ∧ b * x - c * y = -1) →
  (3 * 2 + 3 = 5 ∧ b * 2 - c * 3 = -1) →
  (a * 1 + 2 = 5 ∧ b * 1 - c * 2 = -1) →
  a + b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_equation_system_solution_l128_12831


namespace NUMINAMATH_CALUDE_trumpets_fraction_in_band_l128_12883

-- Define the total number of each instrument
def total_flutes : ℕ := 20
def total_clarinets : ℕ := 30
def total_trumpets : ℕ := 60
def total_pianists : ℕ := 20

-- Define the fraction of each instrument that got in
def flutes_fraction : ℚ := 4/5  -- 80%
def clarinets_fraction : ℚ := 1/2
def pianists_fraction : ℚ := 1/10

-- Define the total number of people in the band
def total_in_band : ℕ := 53

-- Theorem to prove
theorem trumpets_fraction_in_band : 
  (total_in_band - 
   (flutes_fraction * total_flutes + 
    clarinets_fraction * total_clarinets + 
    pianists_fraction * total_pianists)) / total_trumpets = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_trumpets_fraction_in_band_l128_12883


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l128_12830

-- Define a function to calculate the sum of digits
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

-- Define primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

-- Theorem statement
theorem smallest_prime_with_digit_sum_23 :
  ∀ n : ℕ, is_prime n ∧ digit_sum n = 23 → n ≥ 599 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l128_12830


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l128_12880

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 20 + 7 + y + 9) / 6 = 12 → y = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l128_12880


namespace NUMINAMATH_CALUDE_truncated_tetrahedron_lateral_area_l128_12888

/-- Given a truncated tetrahedron with base area A₁, top area A₂ (where A₂ ≤ A₁),
    and sum of lateral face areas P, if the solid can be cut by a plane parallel
    to the base such that a sphere can be inscribed in each of the resulting sections,
    then P = (√A₁ + √A₂)(⁴√A₁ + ⁴√A₂)² -/
theorem truncated_tetrahedron_lateral_area
  (A₁ A₂ P : ℝ)
  (h₁ : 0 < A₁)
  (h₂ : 0 < A₂)
  (h₃ : A₂ ≤ A₁)
  (h₄ : ∃ (A : ℝ), 0 < A ∧ A < A₁ ∧ A > A₂ ∧
    ∃ (R₁ R₂ : ℝ), 0 < R₁ ∧ 0 < R₂ ∧
      A = Real.sqrt (A₁ * A₂) ∧
      (A / A₂) = (A₁ / A) ∧ (A / A₂) = (R₁ / R₂)^2) :
  P = (Real.sqrt A₁ + Real.sqrt A₂) * (Real.sqrt (Real.sqrt A₁) + Real.sqrt (Real.sqrt A₂))^2 := by
sorry

end NUMINAMATH_CALUDE_truncated_tetrahedron_lateral_area_l128_12888


namespace NUMINAMATH_CALUDE_minimum_fourth_round_score_l128_12854

def minimum_average_score : ℝ := 96
def number_of_rounds : ℕ := 4
def first_round_score : ℝ := 95
def second_round_score : ℝ := 97
def third_round_score : ℝ := 94

theorem minimum_fourth_round_score :
  let total_required_score := minimum_average_score * number_of_rounds
  let sum_of_first_three_rounds := first_round_score + second_round_score + third_round_score
  let minimum_fourth_round_score := total_required_score - sum_of_first_three_rounds
  minimum_fourth_round_score = 98 := by sorry

end NUMINAMATH_CALUDE_minimum_fourth_round_score_l128_12854


namespace NUMINAMATH_CALUDE_new_girl_weight_l128_12889

/-- The weight of the new girl given the conditions of the problem -/
def weight_of_new_girl (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * weight_increase

/-- Theorem stating the weight of the new girl under the given conditions -/
theorem new_girl_weight :
  weight_of_new_girl 8 3 70 = 94 := by
  sorry

end NUMINAMATH_CALUDE_new_girl_weight_l128_12889


namespace NUMINAMATH_CALUDE_greatest_n_value_l128_12898

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 3600) : n ≤ 5 ∧ ∃ (m : ℤ), m > 5 → 101 * m^2 > 3600 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_l128_12898


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l128_12887

def solutions : Set (ℤ × ℤ) := {(6, 9), (7, 3), (8, 1), (9, 0), (11, -1), (17, -2), (4, -15), (3, -9), (2, -7), (1, -6), (-1, -5), (-7, -4)}

theorem diophantine_equation_solutions :
  {(x, y) : ℤ × ℤ | x * y + 3 * x - 5 * y = -3} = solutions := by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l128_12887


namespace NUMINAMATH_CALUDE_boxes_with_neither_l128_12815

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (erasers : ℕ) (both : ℕ)
  (h1 : total = 12)
  (h2 : markers = 8)
  (h3 : erasers = 5)
  (h4 : both = 4) :
  total - (markers + erasers - both) = 3 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l128_12815


namespace NUMINAMATH_CALUDE_math_statements_l128_12894

theorem math_statements :
  (8^0 = 1) ∧
  (|-8| = 8) ∧
  (-(-8) = 8) ∧
  (¬(Real.sqrt 8 = 2 * Real.sqrt 2 ∨ Real.sqrt 8 = -2 * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_math_statements_l128_12894


namespace NUMINAMATH_CALUDE_interval_intersection_l128_12863

theorem interval_intersection (x : ℝ) : 
  (2 < 4*x ∧ 4*x < 3 ∧ 2 < 5*x ∧ 5*x < 3) ↔ (1/2 < x ∧ x < 3/5) :=
sorry

end NUMINAMATH_CALUDE_interval_intersection_l128_12863


namespace NUMINAMATH_CALUDE_fourth_root_of_256_l128_12816

theorem fourth_root_of_256 (m : ℝ) : (256 : ℝ) ^ (1/4) = 4^m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_256_l128_12816


namespace NUMINAMATH_CALUDE_complex_multiplication_l128_12856

theorem complex_multiplication (i : ℂ) :
  i * i = -1 →
  (1 - i) * (2 + i) = 3 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l128_12856


namespace NUMINAMATH_CALUDE_min_value_of_e_l128_12861

def e (x : ℝ) (C : ℝ) : ℝ := (x - 1) * (x - 3) * (x - 4) * (x - 6) + C

theorem min_value_of_e (C : ℝ) : 
  (C = -0.5625) ↔ (∀ x : ℝ, e x C ≥ 1 ∧ ∃ x₀ : ℝ, e x₀ C = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_e_l128_12861


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_when_union_is_real_l128_12884

def A (a : ℝ) : Set ℝ := {x | a - 4 < x ∧ x < a + 4}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem intersection_when_a_is_one :
  A 1 ∩ B = {x | -3 < x ∧ x < -1} := by sorry

theorem range_of_a_when_union_is_real :
  (∃ a, A a ∪ B = Set.univ) ↔ ∃ a, 1 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_when_union_is_real_l128_12884


namespace NUMINAMATH_CALUDE_area_of_PRQ_l128_12852

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)
  (xy_length : Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 15)
  (xz_length : Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) = 14)
  (yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 7)

-- Define the circumcenter P
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the incenter Q
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the point R
def R (t : Triangle) : ℝ × ℝ := sorry

-- Define the condition for R being tangent to XZ, YZ, and the circumcircle
def is_tangent (t : Triangle) (r : ℝ × ℝ) : Prop := sorry

-- Define the area of a triangle given three points
def triangle_area (a b c : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_PRQ (t : Triangle) 
  (h : is_tangent t (R t)) : 
  triangle_area (circumcenter t) (incenter t) (R t) = 245 / 72 := by
  sorry

end NUMINAMATH_CALUDE_area_of_PRQ_l128_12852


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l128_12891

def repeating_decimal_3 : ℚ := 1/3
def repeating_decimal_04 : ℚ := 4/99
def repeating_decimal_005 : ℚ := 5/999

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_04 + repeating_decimal_005 = 742/999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l128_12891


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l128_12834

theorem smallest_integer_with_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
  x % 3 = 2 ∧ 
  x % 4 = 3 ∧ 
  x % 5 = 4 ∧ 
  ∀ (y : ℕ), y > 0 ∧ y % 3 = 2 ∧ y % 4 = 3 ∧ y % 5 = 4 → x ≤ y :=
by
  use 59
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l128_12834


namespace NUMINAMATH_CALUDE_num_finches_is_four_l128_12859

-- Define the constants based on the problem conditions
def parakeet_consumption : ℕ := 2 -- grams per day
def parrot_consumption : ℕ := 14 -- grams per day
def finch_consumption : ℕ := parakeet_consumption / 2 -- grams per day
def num_parakeets : ℕ := 3
def num_parrots : ℕ := 2
def total_birdseed : ℕ := 266 -- grams for a week
def days_in_week : ℕ := 7

-- Theorem to prove
theorem num_finches_is_four :
  ∃ (num_finches : ℕ),
    num_finches = 4 ∧
    total_birdseed = (num_parakeets * parakeet_consumption + 
                      num_parrots * parrot_consumption + 
                      num_finches * finch_consumption) * days_in_week :=
by
  sorry


end NUMINAMATH_CALUDE_num_finches_is_four_l128_12859


namespace NUMINAMATH_CALUDE_smallest_triangle_area_l128_12853

/-- The smallest area of a triangle with given vertices -/
theorem smallest_triangle_area :
  let A : ℝ × ℝ × ℝ := (-1, 1, 2)
  let B : ℝ × ℝ × ℝ := (1, 2, 3)
  let C : ℝ → ℝ → ℝ × ℝ × ℝ := fun t s ↦ (t, s, 1)
  let triangle_area (t s : ℝ) : ℝ :=
    (1 / 2) * Real.sqrt ((s^2) + ((-t-3)^2) + ((2*s-t-2)^2))
  ∃ (min_area : ℝ), min_area = Real.sqrt 58 / 2 ∧
    ∀ (t s : ℝ), triangle_area t s ≥ min_area :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_area_l128_12853


namespace NUMINAMATH_CALUDE_system_has_solution_l128_12877

/-- The system of equations has a solution for the given range of b -/
theorem system_has_solution (b : ℝ) 
  (h : b ∈ Set.Iic (-7/12) ∪ Set.Ioi 0) : 
  ∃ (a x y : ℝ), x = 7/b - |y + b| ∧ 
                 x^2 + y^2 + 96 = -a*(2*y + a) - 20*x := by
  sorry


end NUMINAMATH_CALUDE_system_has_solution_l128_12877


namespace NUMINAMATH_CALUDE_gcd_problem_l128_12822

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 1171) :
  Int.gcd (3 * b^2 + 17 * b + 91) (b + 11) = 11 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l128_12822


namespace NUMINAMATH_CALUDE_arithmetic_expression_result_l128_12808

theorem arithmetic_expression_result : 5 + 12 / 3 - 4 * 2 + 3^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_result_l128_12808


namespace NUMINAMATH_CALUDE_xy_squared_change_l128_12800

/-- Theorem: Change in xy^2 when x increases by 20% and y decreases by 30% --/
theorem xy_squared_change (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let x' := 1.2 * x
  let y' := 0.7 * y
  1 - (x' * y' * y') / (x * y * y) = 0.412 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_change_l128_12800


namespace NUMINAMATH_CALUDE_courses_choice_theorem_l128_12864

/-- The number of courses available -/
def total_courses : ℕ := 4

/-- The number of courses each person chooses -/
def courses_per_person : ℕ := 2

/-- The number of ways to choose courses with at least one difference -/
def ways_with_difference : ℕ := 30

/-- Theorem stating the number of ways to choose courses with at least one difference -/
theorem courses_choice_theorem : 
  (Nat.choose total_courses courses_per_person) * 
  (Nat.choose total_courses courses_per_person) - 
  (Nat.choose total_courses courses_per_person) = ways_with_difference :=
by sorry

end NUMINAMATH_CALUDE_courses_choice_theorem_l128_12864


namespace NUMINAMATH_CALUDE_widgets_per_week_l128_12857

/-- The number of widgets John can make per hour -/
def widgets_per_hour : ℕ := 20

/-- The number of hours John works per day -/
def hours_per_day : ℕ := 8

/-- The number of days John works per week -/
def days_per_week : ℕ := 5

/-- Theorem: John makes 800 widgets in a week -/
theorem widgets_per_week : 
  widgets_per_hour * hours_per_day * days_per_week = 800 := by
  sorry


end NUMINAMATH_CALUDE_widgets_per_week_l128_12857


namespace NUMINAMATH_CALUDE_inequality_solution_l128_12850

theorem inequality_solution (x : ℕ) (h : x > 1) :
  (6 * (9 ^ (1 / x)) - 13 * (3 ^ (1 / x)) * (2 ^ (1 / x)) + 6 * (4 ^ (1 / x)) ≤ 0) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l128_12850


namespace NUMINAMATH_CALUDE_large_square_area_l128_12886

theorem large_square_area (s : ℝ) (h1 : s > 0) (h2 : 2 * s^2 = 14) : (3 * s)^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_large_square_area_l128_12886


namespace NUMINAMATH_CALUDE_jake_papayas_l128_12893

/-- The number of papayas Jake's brother can eat in one week -/
def brother_papayas : ℕ := 5

/-- The number of papayas Jake's father can eat in one week -/
def father_papayas : ℕ := 4

/-- The total number of papayas needed for 4 weeks -/
def total_papayas : ℕ := 48

/-- The number of weeks -/
def num_weeks : ℕ := 4

/-- Theorem: Jake can eat 3 papayas in one week -/
theorem jake_papayas : 
  ∃ (j : ℕ), j = 3 ∧ num_weeks * (j + brother_papayas + father_papayas) = total_papayas :=
by sorry

end NUMINAMATH_CALUDE_jake_papayas_l128_12893


namespace NUMINAMATH_CALUDE_total_squares_16x16_board_l128_12823

/-- The size of the chess board -/
def boardSize : Nat := 16

/-- The total number of squares on a square chess board of given size -/
def totalSquares (n : Nat) : Nat :=
  (n * (n + 1) * (2 * n + 1)) / 6

/-- An irregular shape on the chess board -/
structure IrregularShape where
  size : Nat
  isNonRectangular : Bool

/-- Theorem stating the total number of squares on a 16x16 chess board -/
theorem total_squares_16x16_board (shapes : List IrregularShape) 
  (h1 : ∀ s ∈ shapes, s.size ≥ 4)
  (h2 : ∀ s ∈ shapes, s.isNonRectangular = true) :
  totalSquares boardSize = 1496 := by
  sorry

#eval totalSquares boardSize

end NUMINAMATH_CALUDE_total_squares_16x16_board_l128_12823


namespace NUMINAMATH_CALUDE_unique_integer_value_of_expression_l128_12866

theorem unique_integer_value_of_expression 
  (m n p : ℕ) 
  (hm : 2 ≤ m ∧ m ≤ 9) 
  (hn : 2 ≤ n ∧ n ≤ 9) 
  (hp : 2 ≤ p ∧ p ≤ 9) 
  (hdiff : m ≠ n ∧ m ≠ p ∧ n ≠ p) : 
  (∃ k : ℤ, (m + n + p : ℚ) / (m + n) = k) → (m + n + p : ℚ) / (m + n) = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_integer_value_of_expression_l128_12866


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l128_12892

theorem solution_satisfies_equations :
  let x : ℚ := 67 / 9
  let y : ℚ := 22 / 3
  (3 * x - 4 * y = -7) ∧ (6 * x - 5 * y = 8) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l128_12892


namespace NUMINAMATH_CALUDE_triangle_existence_uniqueness_l128_12838

/-- A point in 2D Euclidean space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Incircle of a triangle -/
structure Incircle where
  center : Point
  radius : ℝ

/-- Excircle of a triangle -/
structure Excircle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point lies on a line segment -/
def lies_on_segment (P Q R : Point) : Prop := sorry

/-- Predicate to check if a point is the tangency point of a circle and a line -/
def is_tangency_point (P : Point) (C : Incircle ⊕ Excircle) (L M : Point) : Prop := sorry

/-- Theorem stating the existence and uniqueness of a triangle given specific tangency points -/
theorem triangle_existence_uniqueness 
  (T_a T_aa T_c T_ac : Point) 
  (h_distinct : T_a ≠ T_aa ∧ T_c ≠ T_ac) 
  (h_not_collinear : ¬ lies_on_segment T_a T_c T_aa) : 
  ∃! (ABC : Triangle) (k : Incircle) (k' : Excircle), 
    is_tangency_point T_a (Sum.inl k) ABC.B ABC.C ∧ 
    is_tangency_point T_aa (Sum.inr k') ABC.B ABC.C ∧
    is_tangency_point T_c (Sum.inl k) ABC.A ABC.B ∧
    is_tangency_point T_ac (Sum.inr k') ABC.A ABC.B := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_uniqueness_l128_12838


namespace NUMINAMATH_CALUDE_blue_shirts_count_l128_12819

/-- Represents the number of people at a school dance --/
structure DanceAttendance where
  boys : ℕ
  girls : ℕ
  teachers : ℕ

/-- Calculates the number of people wearing blue shirts at the dance --/
def blueShirts (attendance : DanceAttendance) : ℕ :=
  let blueShirtedBoys := (attendance.boys * 20) / 100
  let blueShirtedMaleTeachers := (attendance.teachers * 25) / 100
  blueShirtedBoys + blueShirtedMaleTeachers

/-- Theorem stating the number of people wearing blue shirts at the dance --/
theorem blue_shirts_count (attendance : DanceAttendance) :
  attendance.boys * 4 = attendance.girls * 3 →
  attendance.teachers * 9 = attendance.boys + attendance.girls →
  attendance.girls = 108 →
  blueShirts attendance = 21 := by
  sorry

#eval blueShirts { boys := 81, girls := 108, teachers := 21 }

end NUMINAMATH_CALUDE_blue_shirts_count_l128_12819


namespace NUMINAMATH_CALUDE_classroom_size_l128_12802

/-- Represents the hair colors in the classroom -/
inductive HairColor
  | Red
  | Blonde
  | Black
  | Brown

/-- Represents the ratio of hair colors in the classroom -/
def hair_ratio : HairColor → ℕ
  | HairColor.Red => 3
  | HairColor.Blonde => 6
  | HairColor.Black => 7
  | HairColor.Brown => 4

/-- The number of kids with red hair -/
def red_hair_count : ℕ := 9

/-- The total number of kids in the classroom -/
def total_kids : ℕ := 60

/-- Theorem stating the total number of kids in the classroom -/
theorem classroom_size :
  (∀ (c : HairColor), red_hair_count * hair_ratio c = total_kids * hair_ratio HairColor.Red) ∧
  (red_hair_count * hair_ratio HairColor.Blonde = 2 * red_hair_count * hair_ratio HairColor.Black) →
  total_kids = 60 :=
by sorry

end NUMINAMATH_CALUDE_classroom_size_l128_12802


namespace NUMINAMATH_CALUDE_right_triangle_properties_l128_12881

/-- Properties of a specific right triangle -/
theorem right_triangle_properties :
  ∀ (a b c : ℝ),
  a = 24 →
  b = 2 * a + 10 →
  c^2 = a^2 + b^2 →
  (1/2 * a * b = 696) ∧ (c = Real.sqrt 3940) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l128_12881


namespace NUMINAMATH_CALUDE_point_displacement_on_line_l128_12871

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of the line x = (y / 2) - (2 / 5) -/
def onLine (p : Point) : Prop :=
  p.x = p.y / 2 - 2 / 5

theorem point_displacement_on_line (m n p : ℝ) :
  onLine ⟨m, n⟩ ∧ onLine ⟨m + p, n + 4⟩ → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_displacement_on_line_l128_12871


namespace NUMINAMATH_CALUDE_expected_black_pairs_modified_deck_l128_12836

/-- A deck of cards -/
structure Deck :=
  (total : ℕ)
  (black : ℕ)
  (red : ℕ)
  (h_total : total = black + red)

/-- The expected number of pairs of adjacent black cards in a circular deal -/
def expected_black_pairs (d : Deck) : ℚ :=
  (d.black : ℚ) * (d.black - 1) / (d.total - 1)

/-- The main theorem -/
theorem expected_black_pairs_modified_deck :
  ∃ (d : Deck), d.total = 60 ∧ d.black = 30 ∧ d.red = 30 ∧ expected_black_pairs d = 870 / 59 := by
  sorry

end NUMINAMATH_CALUDE_expected_black_pairs_modified_deck_l128_12836


namespace NUMINAMATH_CALUDE_least_valid_k_l128_12801

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_valid_k (k : ℤ) : Prop :=
  (0.00010101 * (10 : ℝ) ^ k > 100) ∧
  (sum_of_digits k.natAbs ≤ 15)

def exists_valid_m : Prop :=
  ∃ m : ℤ, 0.000515151 * (10 : ℝ) ^ m ≤ 500

theorem least_valid_k :
  is_valid_k 7 ∧ exists_valid_m ∧
  ∀ k : ℤ, k < 7 → ¬(is_valid_k k) :=
sorry

end NUMINAMATH_CALUDE_least_valid_k_l128_12801


namespace NUMINAMATH_CALUDE_speed_in_still_water_l128_12855

/-- Given a man's upstream and downstream speeds, calculate his speed in still water -/
theorem speed_in_still_water 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 37) 
  (h2 : downstream_speed = 53) : 
  (upstream_speed + downstream_speed) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l128_12855


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l128_12809

theorem sum_reciprocals_bound (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (sum : a + b = 1) :
  1/a + 1/b > 4 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l128_12809


namespace NUMINAMATH_CALUDE_equation_solution_l128_12865

theorem equation_solution : ∃ x : ℝ, (3 * x - 5 = -2 * x + 10) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l128_12865


namespace NUMINAMATH_CALUDE_matching_socks_probability_l128_12899

def black_socks : ℕ := 12
def white_socks : ℕ := 6
def blue_socks : ℕ := 9

theorem matching_socks_probability :
  let total_socks := black_socks + white_socks + blue_socks
  let total_choices := (total_socks * (total_socks - 1)) / 2
  let matching_choices := (black_socks * (black_socks - 1)) / 2 +
                          (white_socks * (white_socks - 1)) / 2 +
                          (blue_socks * (blue_socks - 1)) / 2
  (matching_choices : ℚ) / total_choices = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_matching_socks_probability_l128_12899


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l128_12874

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -(1 / 2) := by sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l128_12874


namespace NUMINAMATH_CALUDE_equation_solutions_l128_12837

theorem equation_solutions :
  ∀ (x n : ℕ+) (p : ℕ), 
    Prime p → 
    (x^3 + 3*x + 14 = 2*p^(n : ℕ)) → 
    ((x = 1 ∧ n = 2 ∧ p = 3) ∨ (x = 3 ∧ n = 2 ∧ p = 5)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l128_12837


namespace NUMINAMATH_CALUDE_inequality_solution_set_l128_12803

theorem inequality_solution_set (x : ℝ) :
  abs (x - 5) + abs (x + 3) ≥ 10 ↔ x ≤ -4 ∨ x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l128_12803


namespace NUMINAMATH_CALUDE_impossible_transformation_l128_12814

/-- Represents the operation of replacing two numbers with their updated values -/
def replace_numbers (numbers : List ℕ) (x y : ℕ) : List ℕ :=
  (x - 1) :: (y + 3) :: (numbers.filter (λ n => n ≠ x ∧ n ≠ y))

/-- Checks if a list of numbers is valid according to the problem rules -/
def is_valid_list (numbers : List ℕ) : Prop :=
  numbers.length = 10 ∧ numbers.sum % 2 = 1

/-- The initial list of numbers on the board -/
def initial_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/-- The target list of numbers we want to achieve -/
def target_numbers : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 2012]

/-- Theorem stating that it's impossible to transform the initial numbers into the target numbers -/
theorem impossible_transformation :
  ¬ ∃ (n : ℕ) (operations : List (ℕ × ℕ)),
    operations.length = n ∧
    (operations.foldl (λ acc (x, y) => replace_numbers acc x y) initial_numbers) = target_numbers :=
sorry

end NUMINAMATH_CALUDE_impossible_transformation_l128_12814


namespace NUMINAMATH_CALUDE_spinner_probability_l128_12842

/-- Represents a fair spinner with 4 equal sections -/
structure FairSpinner :=
  (sections : Fin 4)

/-- Probability of not getting 'e' in one spin -/
def prob_not_e : ℝ := 0.75

/-- Target probability of not getting 'e' after multiple spins -/
def target_prob : ℝ := 0.5625

/-- Number of spins to achieve the target probability -/
def num_spins : ℕ := 2

theorem spinner_probability (s : FairSpinner) :
  (prob_not_e ^ num_spins : ℝ) = target_prob :=
sorry

end NUMINAMATH_CALUDE_spinner_probability_l128_12842


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l128_12828

/-- 
Given a loan with simple interest where:
- The principal amount is $1200
- The number of years equals the rate of interest
- The total interest paid is $432
Prove that the rate of interest is 6%
-/
theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) 
  (h1 : principal = 1200)
  (h2 : interest_paid = 432) :
  ∃ (rate : ℝ), rate = 6 ∧ interest_paid = principal * (rate / 100) * rate :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l128_12828


namespace NUMINAMATH_CALUDE_permutation_residue_system_bound_l128_12806

/-- A permutation of (1, 2, ..., n) -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The set {pᵢ + i | 1 ≤ i ≤ n} is a complete residue system modulo n -/
def IsSumCompleteResidue (n : ℕ) (p : Permutation n) : Prop :=
  ∀ k : Fin n, ∃ i : Fin n, (p i + i : ℕ) % n = k

/-- The set {pᵢ - i | 1 ≤ i ≤ n} is a complete residue system modulo n -/
def IsDiffCompleteResidue (n : ℕ) (p : Permutation n) : Prop :=
  ∀ k : Fin n, ∃ i : Fin n, ((p i : ℕ) - (i : ℕ) + n) % n = k

/-- Main theorem: If n satisfies the conditions, then n ≥ 4 -/
theorem permutation_residue_system_bound (n : ℕ) :
  (∃ p : Permutation n, IsSumCompleteResidue n p ∧ IsDiffCompleteResidue n p) →
  n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_permutation_residue_system_bound_l128_12806


namespace NUMINAMATH_CALUDE_max_units_of_A_l128_12843

-- Define the variables
variable (x y z : ℕ)

-- Define the conditions
def initial_equation : Prop := 3 * x + 5 * y + 7 * z = 62
def final_equation : Prop := 2 * x + 4 * y + 6 * z = 50

-- Define the theorem
theorem max_units_of_A : 
  ∀ x y z : ℕ, 
  initial_equation x y z → 
  final_equation x y z → 
  x ≤ 5 ∧ ∃ x' y' z' : ℕ, x' = 5 ∧ initial_equation x' y' z' ∧ final_equation x' y' z' :=
sorry

end NUMINAMATH_CALUDE_max_units_of_A_l128_12843


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l128_12832

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := (n + 1) / 2

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 1 / (n * a n)

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := (2 * n) / (n + 1)

theorem arithmetic_sequence_problem :
  (a 7 = 4) ∧ (a 19 = 2 * a 9) ∧
  (∀ n : ℕ, n > 0 → b n = 1 / (n * a n)) ∧
  (∀ n : ℕ, n > 0 → S n = (2 * n) / (n + 1)) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l128_12832


namespace NUMINAMATH_CALUDE_average_of_x_and_y_l128_12870

theorem average_of_x_and_y (x y : ℝ) : 
  (4 + 6 + 8 + x + y) / 5 = 20 → (x + y) / 2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_average_of_x_and_y_l128_12870


namespace NUMINAMATH_CALUDE_cigar_purchase_problem_l128_12810

theorem cigar_purchase_problem :
  ∃ (x y z : ℕ),
    x + y + z = 100 ∧
    (1/2 : ℚ) * x + 3 * y + 10 * z = 100 ∧
    x = 94 ∧ y = 1 ∧ z = 5 := by
  sorry

end NUMINAMATH_CALUDE_cigar_purchase_problem_l128_12810


namespace NUMINAMATH_CALUDE_shop_prices_existence_l128_12897

theorem shop_prices_existence (S : ℕ) (h : S ≥ 100) :
  ∃ (a b c P : ℕ), 
    a > b ∧ b > c ∧ 
    a + b + c = S ∧
    a * b * c = P ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    ∃ (a' b' c' : ℕ), (a' ≠ a ∨ b' ≠ b ∨ c' ≠ c) ∧
      a' > b' ∧ b' > c' ∧
      a' + b' + c' = S ∧
      a' * b' * c' = P ∧
      a' > 0 ∧ b' > 0 ∧ c' > 0 :=
by sorry

end NUMINAMATH_CALUDE_shop_prices_existence_l128_12897


namespace NUMINAMATH_CALUDE_unique_solution_l128_12804

def clubsuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

theorem unique_solution : ∃! A : ℝ, clubsuit A 5 = 80 ∧ A = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l128_12804


namespace NUMINAMATH_CALUDE_crayons_in_drawer_l128_12817

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 9

/-- The number of crayons Benny added to the drawer -/
def added_crayons : ℕ := 3

/-- The total number of crayons in the drawer after Benny adds more -/
def total_crayons : ℕ := initial_crayons + added_crayons

theorem crayons_in_drawer : total_crayons = 12 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_drawer_l128_12817


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l128_12818

theorem distance_between_complex_points :
  let z₁ : ℂ := -3 + I
  let z₂ : ℂ := 1 - I
  Complex.abs (z₂ - z₁) = Real.sqrt 20 := by sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l128_12818


namespace NUMINAMATH_CALUDE_system_solution_existence_l128_12848

theorem system_solution_existence (a b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + x*y = a ∧ x^2 - y^2 = b) ↔ 
  -2*a ≤ Real.sqrt 3 * b ∧ Real.sqrt 3 * b ≤ 2*a :=
by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l128_12848


namespace NUMINAMATH_CALUDE_seashells_given_correct_l128_12885

/-- The number of seashells Tom gave to Jessica -/
def seashells_given : ℕ :=
  5 - 3

theorem seashells_given_correct : seashells_given = 2 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_correct_l128_12885


namespace NUMINAMATH_CALUDE_charlies_lollipops_l128_12879

/-- Given the number of lollipops of each flavor and the number of friends,
    prove that the number of lollipops Charlie keeps is the remainder when
    the total number of lollipops is divided by the number of friends. -/
theorem charlies_lollipops
  (cherry wintergreen grape shrimp_cocktail raspberry : ℕ)
  (friends : ℕ) (friends_pos : friends > 0) :
  let total := cherry + wintergreen + grape + shrimp_cocktail + raspberry
  (total % friends) = total - friends * (total / friends) :=
by sorry

end NUMINAMATH_CALUDE_charlies_lollipops_l128_12879


namespace NUMINAMATH_CALUDE_simplify_expression_l128_12847

theorem simplify_expression (x : ℝ) (h : x ≠ 2) :
  2 - (2 * (1 - (3 - (2 / (2 - x))))) = 6 - 4 / (2 - x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l128_12847


namespace NUMINAMATH_CALUDE_divisible_by_24_l128_12833

theorem divisible_by_24 (n : ℕ+) : ∃ k : ℤ, (n : ℤ) * (n + 2) * (5 * n - 1) * (5 * n + 1) = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l128_12833


namespace NUMINAMATH_CALUDE_tims_interest_rate_l128_12875

/-- Calculates the compound interest after n years -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

theorem tims_interest_rate :
  let tim_principal : ℝ := 600
  let lana_principal : ℝ := 1000
  let lana_rate : ℝ := 0.05
  let years : ℕ := 2
  ∀ tim_rate : ℝ,
    (compoundInterest tim_principal tim_rate years - tim_principal) =
    (compoundInterest lana_principal lana_rate years - lana_principal) + 23.5 →
    tim_rate = 0.1 := by
  sorry

#check tims_interest_rate

end NUMINAMATH_CALUDE_tims_interest_rate_l128_12875


namespace NUMINAMATH_CALUDE_equation_satisfied_at_x_equals_4_l128_12807

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem equation_satisfied_at_x_equals_4 :
  2 * (f 4) - 19 = f (4 - 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_satisfied_at_x_equals_4_l128_12807


namespace NUMINAMATH_CALUDE_problem_statement_l128_12813

theorem problem_statement (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b < 0) 
  (hab : abs a < abs b) 
  (hbc : abs b < abs c) : 
  (abs (a * b) < abs (b * c)) ∧ 
  (a * c < abs (b * c)) ∧ 
  (abs (a + b) < abs (b + c)) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l128_12813


namespace NUMINAMATH_CALUDE_ac_unit_final_price_l128_12851

/-- Calculates the final price of an air-conditioning unit after a series of price changes. -/
def final_price (initial_price : ℝ) : ℝ :=
  let price1 := initial_price * (1 - 0.12)  -- February
  let price2 := price1 * (1 + 0.08)         -- March
  let price3 := price2 * (1 - 0.10)         -- April
  let price4 := price3 * (1 + 0.05)         -- June
  let price5 := price4 * (1 - 0.07)         -- August
  let price6 := price5 * (1 + 0.06)         -- October
  let price7 := price6 * (1 - 0.15)         -- November
  price7

/-- Theorem stating that the final price of the air-conditioning unit is approximately $353.71. -/
theorem ac_unit_final_price : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |final_price 470 - 353.71| < ε :=
sorry

end NUMINAMATH_CALUDE_ac_unit_final_price_l128_12851


namespace NUMINAMATH_CALUDE_contractor_absent_days_l128_12845

/-- Represents the problem of calculating a contractor's absent days. -/
def ContractorProblem (total_days : ℕ) (daily_wage : ℚ) (daily_fine : ℚ) (total_amount : ℚ) : Prop :=
  ∃ (worked_days absent_days : ℕ),
    worked_days + absent_days = total_days ∧
    daily_wage * worked_days - daily_fine * absent_days = total_amount

/-- Theorem stating that given the problem conditions, the number of absent days is 10. -/
theorem contractor_absent_days :
  ContractorProblem 30 25 (15/2) 425 →
  ∃ (worked_days absent_days : ℕ),
    worked_days + absent_days = 30 ∧
    absent_days = 10 := by
  sorry

#check contractor_absent_days

end NUMINAMATH_CALUDE_contractor_absent_days_l128_12845


namespace NUMINAMATH_CALUDE_expected_value_of_coins_l128_12811

/-- Represents a coin with its value in cents and probability of heads -/
structure Coin where
  value : ℝ
  prob_heads : ℝ

/-- The set of coins being flipped -/
def coins : Finset Coin := sorry

/-- The expected value of a single coin -/
def expected_value (c : Coin) : ℝ := c.value * c.prob_heads

/-- The total expected value of all coins -/
def total_expected_value : ℝ := (coins.sum expected_value)

/-- Theorem stating the expected value of coins coming up heads -/
theorem expected_value_of_coins : 
  (coins.card = 5) → 
  (∃ c ∈ coins, c.value = 1 ∧ c.prob_heads = 1/2) →
  (∃ c ∈ coins, c.value = 5 ∧ c.prob_heads = 1/2) →
  (∃ c ∈ coins, c.value = 10 ∧ c.prob_heads = 1/2) →
  (∃ c ∈ coins, c.value = 25 ∧ c.prob_heads = 1/2) →
  (∃ c ∈ coins, c.value = 50 ∧ c.prob_heads = 3/4) →
  total_expected_value = 58 :=
by sorry

end NUMINAMATH_CALUDE_expected_value_of_coins_l128_12811


namespace NUMINAMATH_CALUDE_expand_product_l128_12872

theorem expand_product (x : ℝ) : 5 * (x + 6) * (x^2 + 2*x + 3) = 5*x^3 + 40*x^2 + 75*x + 90 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l128_12872


namespace NUMINAMATH_CALUDE_complex_number_identity_l128_12805

theorem complex_number_identity (a b c : ℂ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : c ≠ 0) 
  (h4 : a + b + c = 15) 
  (h5 : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) : 
  (a^3 + b^3 + c^3) / (a*b*c) = 18 := by
sorry

end NUMINAMATH_CALUDE_complex_number_identity_l128_12805


namespace NUMINAMATH_CALUDE_correct_problems_l128_12873

theorem correct_problems (total : ℕ) (h1 : total = 54) : ∃ (correct : ℕ), 
  correct + 2 * correct = total ∧ correct = 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_problems_l128_12873


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l128_12895

theorem part_to_whole_ratio (N P : ℚ) (h1 : N = 240) (h2 : (1/5) * N + 6 = P - 6) :
  (P - 6) / N = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l128_12895


namespace NUMINAMATH_CALUDE_school_student_count_l128_12890

theorem school_student_count 
  (total : ℕ) 
  (transferred : ℕ) 
  (difference : ℕ) 
  (h1 : total = 432) 
  (h2 : transferred = 16) 
  (h3 : difference = 24) :
  ∃ (a b : ℕ), 
    a + b = total ∧ 
    (a - transferred) = (b + transferred + difference) ∧
    a = 244 ∧ 
    b = 188 := by
  sorry

end NUMINAMATH_CALUDE_school_student_count_l128_12890


namespace NUMINAMATH_CALUDE_picnic_task_division_l128_12841

theorem picnic_task_division (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 3) :
  Nat.choose n k = 20 := by
  sorry

end NUMINAMATH_CALUDE_picnic_task_division_l128_12841


namespace NUMINAMATH_CALUDE_softball_team_size_l128_12839

theorem softball_team_size :
  ∀ (men women : ℕ),
  women = men + 2 →
  (men : ℚ) / (women : ℚ) = 7777777777777778 / 10000000000000000 →
  men + women = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_size_l128_12839


namespace NUMINAMATH_CALUDE_inequality_solution_count_l128_12858

theorem inequality_solution_count : 
  (∃ (S : Finset ℕ), 
    (∀ n ∈ S, (n : ℝ) + 6 * ((n : ℝ) - 1) * ((n : ℝ) - 15) < 0) ∧ 
    (∀ n : ℕ, (n : ℝ) + 6 * ((n : ℝ) - 1) * ((n : ℝ) - 15) < 0 → n ∈ S) ∧
    Finset.card S = 13) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_count_l128_12858


namespace NUMINAMATH_CALUDE_triangle_properties_l128_12876

theorem triangle_properties (A B C : Real) (a b c : Real) 
  (m_x m_y n_x n_y : Real → Real) :
  (∀ θ, m_x θ = 2 * Real.cos θ ∧ m_y θ = 1) →
  (∀ θ, n_x θ = 1 ∧ n_y θ = Real.sin (θ + Real.pi / 6)) →
  (∃ k : Real, k ≠ 0 ∧ ∀ θ, m_x θ * k = n_x θ ∧ m_y θ * k = n_y θ) →
  a = 2 * Real.sqrt 3 →
  c = 4 →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  A = Real.pi / 3 ∧ 
  b = 2 ∧ 
  1/2 * b * c * Real.sin A = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l128_12876


namespace NUMINAMATH_CALUDE_lcm_problem_l128_12868

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l128_12868


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l128_12825

theorem quadratic_solution_range (t : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - t = 0 → -1 < x ∧ x < 4) →
  -1 ≤ t ∧ t < 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l128_12825


namespace NUMINAMATH_CALUDE_orange_juice_fraction_is_three_tenths_l128_12846

/-- Represents the capacity and fill level of a pitcher -/
structure Pitcher where
  capacity : ℚ
  fillLevel : ℚ

/-- Calculates the fraction of orange juice in the mixture -/
def orangeJuiceFraction (pitchers : List Pitcher) : ℚ :=
  let totalJuice := pitchers.foldl (fun acc p => acc + p.capacity * p.fillLevel) 0
  let totalVolume := pitchers.foldl (fun acc p => acc + p.capacity) 0
  totalJuice / totalVolume

/-- Theorem stating that the fraction of orange juice in the mixture is 3/10 -/
theorem orange_juice_fraction_is_three_tenths :
  let pitchers := [
    Pitcher.mk 500 (1/5),
    Pitcher.mk 700 (3/7),
    Pitcher.mk 800 (1/4)
  ]
  orangeJuiceFraction pitchers = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_is_three_tenths_l128_12846


namespace NUMINAMATH_CALUDE_tanner_has_16_berries_l128_12849

/-- The number of berries each person has -/
structure Berries where
  skylar : ℕ
  steve : ℕ
  stacy : ℕ
  tanner : ℕ

/-- Calculate the number of berries each person has based on the given conditions -/
def calculate_berries : Berries :=
  let skylar := 20
  let steve := 4 * (skylar / 3)^2
  let stacy := 2 * steve + 50
  let tanner := (8 * stacy) / (skylar + steve)
  { skylar := skylar, steve := steve, stacy := stacy, tanner := tanner }

/-- Theorem stating that Tanner has 16 berries -/
theorem tanner_has_16_berries : (calculate_berries.tanner) = 16 := by
  sorry

#eval calculate_berries.tanner

end NUMINAMATH_CALUDE_tanner_has_16_berries_l128_12849


namespace NUMINAMATH_CALUDE_prob_heart_king_spade_l128_12862

/-- Represents a standard deck of 52 cards -/
def standardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def numHearts : ℕ := 13

/-- Number of kings in a standard deck -/
def numKings : ℕ := 4

/-- Number of spades in a standard deck -/
def numSpades : ℕ := 13

/-- Probability of drawing a heart, then a king, then a spade from a standard 52-card deck without replacement -/
theorem prob_heart_king_spade : 
  (numHearts : ℚ) / standardDeck * 
  numKings / (standardDeck - 1) * 
  numSpades / (standardDeck - 2) = 13 / 2550 := by sorry

end NUMINAMATH_CALUDE_prob_heart_king_spade_l128_12862


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_z_l128_12867

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem max_imaginary_part_of_z (z : ℂ) 
  (h : is_purely_imaginary ((z - 6) / (z - 8*I))) : 
  (⨆ (z : ℂ), |z.im|) = 9 := by sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_z_l128_12867


namespace NUMINAMATH_CALUDE_minimal_radius_inscribed_triangle_l128_12824

theorem minimal_radius_inscribed_triangle (a b : ℝ) (ha : a = 3) (hb : b = 4) :
  let c := Real.sqrt (a^2 + b^2)
  let R := c / 2
  R = (5 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_minimal_radius_inscribed_triangle_l128_12824


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_S_l128_12829

/-- Definition of S_n as the sum of reciprocals of non-zero digits from 1 to 2·10^n -/
def S (n : ℕ) : ℚ :=
  sorry

/-- Theorem stating that 32 is the smallest positive integer n for which S_n is an integer -/
theorem smallest_n_for_integer_S :
  ∀ k : ℕ, k > 0 → k < 32 → ¬ (S k).isInt ∧ (S 32).isInt := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_S_l128_12829


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l128_12878

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 7) % 25 = 0 ∧ (n + 7) % 49 = 0 ∧ (n + 7) % 15 = 0 ∧ (n + 7) % 21 = 0

theorem smallest_number_divisible_by_all : 
  is_divisible_by_all 3668 ∧ ∀ m : ℕ, m < 3668 → ¬is_divisible_by_all m := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l128_12878


namespace NUMINAMATH_CALUDE_derivative_of_power_function_l128_12869

theorem derivative_of_power_function (a k : ℝ) (x : ℝ) :
  deriv (λ x => (3 * a * x - x^2)^k) x = k * (3 * a - 2 * x) * (3 * a * x - x^2)^(k - 1) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_power_function_l128_12869
