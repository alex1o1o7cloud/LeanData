import Mathlib

namespace NUMINAMATH_CALUDE_coffee_consumption_l178_17865

theorem coffee_consumption (initial_amount : ℝ) (first_fraction : ℝ) (second_fraction : ℝ) (final_amount : ℝ) : 
  initial_amount = 12 →
  first_fraction = 1/4 →
  second_fraction = 1/2 →
  final_amount = 1 →
  initial_amount - (first_fraction * initial_amount + second_fraction * initial_amount + final_amount) = 2 := by
sorry


end NUMINAMATH_CALUDE_coffee_consumption_l178_17865


namespace NUMINAMATH_CALUDE_hotel_rooms_l178_17863

theorem hotel_rooms (total_rooms : ℕ) (single_cost double_cost : ℕ) (total_revenue : ℕ) :
  total_rooms = 260 ∧
  single_cost = 35 ∧
  double_cost = 60 ∧
  total_revenue = 14000 →
  ∃ (single_rooms double_rooms : ℕ),
    single_rooms + double_rooms = total_rooms ∧
    single_cost * single_rooms + double_cost * double_rooms = total_revenue ∧
    single_rooms = 64 :=
by sorry

end NUMINAMATH_CALUDE_hotel_rooms_l178_17863


namespace NUMINAMATH_CALUDE_cookie_count_l178_17868

theorem cookie_count (cookies_per_bag : ℕ) (num_bags : ℕ) : 
  cookies_per_bag = 41 → num_bags = 53 → cookies_per_bag * num_bags = 2173 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l178_17868


namespace NUMINAMATH_CALUDE_triangle_acute_from_inequalities_l178_17836

theorem triangle_acute_from_inequalities (α β γ : Real) 
  (sum_angles : α + β + γ = Real.pi)
  (ineq1 : Real.sin α > Real.cos β)
  (ineq2 : Real.sin β > Real.cos γ)
  (ineq3 : Real.sin γ > Real.cos α) :
  α < Real.pi / 2 ∧ β < Real.pi / 2 ∧ γ < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_acute_from_inequalities_l178_17836


namespace NUMINAMATH_CALUDE_boxes_remaining_l178_17849

theorem boxes_remaining (total : ℕ) (filled : ℕ) (h1 : total = 13) (h2 : filled = 8) :
  total - filled = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_remaining_l178_17849


namespace NUMINAMATH_CALUDE_pure_imaginary_modulus_l178_17850

theorem pure_imaginary_modulus (b : ℝ) : 
  let z : ℂ := (3 + b * Complex.I) * (1 + Complex.I) - 2
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs z = 4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_modulus_l178_17850


namespace NUMINAMATH_CALUDE_octal_2016_to_binary_l178_17824

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ := sorry

/-- Converts a decimal number to binary --/
def decimal_to_binary (decimal : ℕ) : List ℕ := sorry

/-- Converts a list of binary digits to a natural number --/
def binary_list_to_nat (binary : List ℕ) : ℕ := sorry

theorem octal_2016_to_binary :
  let octal := 2016
  let decimal := octal_to_decimal octal
  let binary := decimal_to_binary decimal
  binary_list_to_nat binary = binary_list_to_nat [1,0,0,0,0,0,0,1,1,1,0] := by sorry

end NUMINAMATH_CALUDE_octal_2016_to_binary_l178_17824


namespace NUMINAMATH_CALUDE_namjoon_has_greater_sum_l178_17800

def jimin_numbers : List Nat := [1, 7]
def namjoon_numbers : List Nat := [6, 3]

theorem namjoon_has_greater_sum :
  List.sum namjoon_numbers > List.sum jimin_numbers := by
  sorry

end NUMINAMATH_CALUDE_namjoon_has_greater_sum_l178_17800


namespace NUMINAMATH_CALUDE_solution_set_f_max_value_g_l178_17873

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Define the function g(x)
def g (x : ℝ) : ℝ := f x - x^2 + x

-- Theorem 1: Solution set of f(x) ≥ 1
theorem solution_set_f (x : ℝ) : f x ≥ 1 ↔ x ≥ 1 := by sorry

-- Theorem 2: Maximum value of g(x)
theorem max_value_g : ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 5/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_max_value_g_l178_17873


namespace NUMINAMATH_CALUDE_second_discount_percentage_second_discount_percentage_proof_l178_17826

theorem second_discount_percentage (original_price first_discount final_price : ℝ) 
  (h1 : original_price = 528)
  (h2 : first_discount = 0.2)
  (h3 : final_price = 380.16) : ℝ :=
let price_after_first_discount := original_price * (1 - first_discount)
let second_discount := (price_after_first_discount - final_price) / price_after_first_discount
0.1

theorem second_discount_percentage_proof 
  (original_price first_discount final_price : ℝ) 
  (h1 : original_price = 528)
  (h2 : first_discount = 0.2)
  (h3 : final_price = 380.16) : 
  second_discount_percentage original_price first_discount final_price h1 h2 h3 = 0.1 := by
sorry

end NUMINAMATH_CALUDE_second_discount_percentage_second_discount_percentage_proof_l178_17826


namespace NUMINAMATH_CALUDE_subcommittee_count_l178_17870

theorem subcommittee_count (n m k t : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 5) (h4 : t = 5) :
  (Nat.choose n k) - (Nat.choose (n - t) k) = 771 :=
by sorry

end NUMINAMATH_CALUDE_subcommittee_count_l178_17870


namespace NUMINAMATH_CALUDE_find_N_l178_17852

theorem find_N : ∃ N : ℤ, (10 + 11 + 12) / 3 = (2010 + 2011 + 2012 + N) / 4 → N = -5989 := by
  sorry

end NUMINAMATH_CALUDE_find_N_l178_17852


namespace NUMINAMATH_CALUDE_smallest_number_proof_smallest_number_is_4725_l178_17812

theorem smallest_number_proof (x : ℕ) : 
  (x + 3 = 4728) ∧ 
  (∃ k₁ : ℕ, (x + 3) = 27 * k₁) ∧ 
  (∃ k₂ : ℕ, (x + 3) = 35 * k₂) ∧ 
  (∃ k₃ : ℕ, (x + 3) = 25 * k₃) →
  x ≥ 4725 :=
by sorry

theorem smallest_number_is_4725 : 
  (4725 + 3 = 4728) ∧ 
  (∃ k₁ : ℕ, (4725 + 3) = 27 * k₁) ∧ 
  (∃ k₂ : ℕ, (4725 + 3) = 35 * k₂) ∧ 
  (∃ k₃ : ℕ, (4725 + 3) = 25 * k₃) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_smallest_number_is_4725_l178_17812


namespace NUMINAMATH_CALUDE_henry_collection_cost_l178_17831

/-- The amount of money Henry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem: Henry needs $144 to finish his collection -/
theorem henry_collection_cost :
  money_needed 3 15 12 = 144 := by
  sorry

end NUMINAMATH_CALUDE_henry_collection_cost_l178_17831


namespace NUMINAMATH_CALUDE_retail_price_calculation_l178_17841

/-- The retail price of a machine, given wholesale price, discount, and profit margin. -/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (h1 : wholesale_price = 90)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.2) :
  ∃ w : ℝ, w = 120 ∧ 
    (1 - discount_rate) * w = wholesale_price + profit_rate * wholesale_price :=
by sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l178_17841


namespace NUMINAMATH_CALUDE_benzene_homolog_bonds_l178_17884

/-- Represents the number of bonds in a molecule -/
def num_bonds (n : ℕ) : ℕ := 3 * n - 3

/-- Represents the number of valence electrons in a molecule -/
def num_valence_electrons (n : ℕ) : ℕ := 4 * n + (2 * n - 6)

/-- Theorem stating the relationship between carbon atoms and bonds in benzene homologs -/
theorem benzene_homolog_bonds (n : ℕ) : 
  num_bonds n = (num_valence_electrons n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_benzene_homolog_bonds_l178_17884


namespace NUMINAMATH_CALUDE_line_parallel_plane_neither_necessary_nor_sufficient_l178_17878

/-- Two lines are perpendicular -/
def perpendicular (l₁ l₂ : Line) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem line_parallel_plane_neither_necessary_nor_sufficient
  (m n : Line) (α : Plane) (h : perpendicular m n) :
  ¬(∀ (m n : Line) (α : Plane), perpendicular m n → (line_parallel_plane n α ↔ line_perp_plane m α)) :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_plane_neither_necessary_nor_sufficient_l178_17878


namespace NUMINAMATH_CALUDE_unique_solution_system_l178_17862

theorem unique_solution_system (x y : ℝ) : 
  (3 * x ≥ 2 * y + 16 ∧ 
   x^4 + 2 * x^2 * y^2 + y^4 + 25 - 26 * x^2 - 26 * y^2 = 72 * x * y) ↔ 
  (x = 6 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l178_17862


namespace NUMINAMATH_CALUDE_club_leadership_combinations_l178_17821

/-- Represents the total number of members in the club -/
def total_members : ℕ := 24

/-- Represents the number of boys in the club -/
def num_boys : ℕ := 12

/-- Represents the number of girls in the club -/
def num_girls : ℕ := 12

/-- Represents the number of age groups -/
def num_age_groups : ℕ := 2

/-- Represents the number of members in each gender and age group combination -/
def members_per_group : ℕ := 6

/-- Theorem stating the number of ways to choose a president and vice-president -/
theorem club_leadership_combinations : 
  (num_boys * members_per_group + num_girls * members_per_group) = 144 := by
  sorry

end NUMINAMATH_CALUDE_club_leadership_combinations_l178_17821


namespace NUMINAMATH_CALUDE_digit_sum_problem_l178_17866

theorem digit_sum_problem (A B C D E F : ℕ) :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F →
  A ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  B ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  C ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  D ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  F ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
  2*A + 3*B + 2*C + 2*D + 2*E + 2*F = 47 →
  B = 5 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l178_17866


namespace NUMINAMATH_CALUDE_negation_of_proposition_l178_17893

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l178_17893


namespace NUMINAMATH_CALUDE_scientific_notation_of_234_1_million_l178_17877

theorem scientific_notation_of_234_1_million :
  let million : ℝ := 10^6
  234.1 * million = 2.341 * 10^6 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_234_1_million_l178_17877


namespace NUMINAMATH_CALUDE_students_passing_both_subjects_l178_17882

theorem students_passing_both_subjects (total_english : ℕ) (total_math : ℕ) (diff_only_english : ℕ) :
  total_english = 30 →
  total_math = 20 →
  diff_only_english = 10 →
  ∃ (both : ℕ),
    both = 10 ∧
    total_english = both + (both + diff_only_english) ∧
    total_math = both + both :=
by sorry

end NUMINAMATH_CALUDE_students_passing_both_subjects_l178_17882


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt5_l178_17829

theorem complex_modulus_sqrt5 (a : ℝ) : 
  Complex.abs (1 + a * Complex.I) = Real.sqrt 5 ↔ a = 2 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt5_l178_17829


namespace NUMINAMATH_CALUDE_horner_v2_equals_10_l178_17894

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^7 + x^6 + x^4 + x^2 + 1 -/
def f (x : ℝ) : ℝ := 2*x^7 + x^6 + x^4 + x^2 + 1

/-- Coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [1, 0, 1, 0, 1, 0, 1, 2]

/-- Theorem: V_2 in Horner's method for f(x) when x = 2 is 10 -/
theorem horner_v2_equals_10 : 
  (horner (coeffs.take 3) 2) = 10 := by sorry

end NUMINAMATH_CALUDE_horner_v2_equals_10_l178_17894


namespace NUMINAMATH_CALUDE_sum_fractions_equals_11111_l178_17860

theorem sum_fractions_equals_11111 : 
  4/5 + 9 * (4/5) + 99 * (4/5) + 999 * (4/5) + 9999 * (4/5) + 1 = 11111 := by
  sorry

end NUMINAMATH_CALUDE_sum_fractions_equals_11111_l178_17860


namespace NUMINAMATH_CALUDE_negative_a_cubed_times_negative_a_fourth_l178_17891

theorem negative_a_cubed_times_negative_a_fourth (a : ℝ) : -a^3 * (-a)^4 = -a^7 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_cubed_times_negative_a_fourth_l178_17891


namespace NUMINAMATH_CALUDE_horner_v4_equals_80_l178_17885

/-- Horner's Rule for polynomial evaluation --/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^6 - 12x^5 + 60x^4 - 160x^3 + 240x^2 - 192x + 64 --/
def f (x : ℝ) : ℝ :=
  x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

/-- The coefficients of the polynomial in reverse order --/
def coeffs : List ℝ := [64, -192, 240, -160, 60, -12, 1]

/-- The value of x for which we're evaluating the polynomial --/
def x : ℝ := 2

/-- The intermediate value v_4 in Horner's Rule calculation --/
def v_4 : ℝ := ((-80 * x) + 240)

theorem horner_v4_equals_80 : v_4 = 80 := by
  sorry

#eval v_4

end NUMINAMATH_CALUDE_horner_v4_equals_80_l178_17885


namespace NUMINAMATH_CALUDE_total_orders_filled_l178_17851

/-- Represents the price of a catfish dinner in dollars -/
def catfish_price : ℚ := 6

/-- Represents the price of a popcorn shrimp dinner in dollars -/
def popcorn_shrimp_price : ℚ := 7/2

/-- Represents the total amount collected in dollars -/
def total_collected : ℚ := 267/2

/-- Represents the number of popcorn shrimp dinners sold -/
def popcorn_shrimp_orders : ℕ := 9

/-- Theorem stating that the total number of orders filled is 26 -/
theorem total_orders_filled : ∃ (catfish_orders : ℕ), 
  catfish_price * catfish_orders + popcorn_shrimp_price * popcorn_shrimp_orders = total_collected ∧ 
  catfish_orders + popcorn_shrimp_orders = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_orders_filled_l178_17851


namespace NUMINAMATH_CALUDE_furniture_legs_l178_17818

theorem furniture_legs 
  (total_tables : ℕ) 
  (total_legs : ℕ) 
  (four_leg_tables : ℕ) 
  (h1 : total_tables = 36)
  (h2 : total_legs = 124)
  (h3 : four_leg_tables = 16) :
  (total_legs - 4 * four_leg_tables) / (total_tables - four_leg_tables) = 3 := by
sorry

end NUMINAMATH_CALUDE_furniture_legs_l178_17818


namespace NUMINAMATH_CALUDE_composite_product_division_l178_17890

def first_eight_composites : List Nat := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List Nat := [16, 18, 20, 21, 22, 24, 25, 26]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem composite_product_division :
  (product_of_list first_eight_composites) / 
  (product_of_list next_eight_composites) = 1 / 3120 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_division_l178_17890


namespace NUMINAMATH_CALUDE_heavy_operator_daily_rate_l178_17899

theorem heavy_operator_daily_rate
  (total_workers : ℕ)
  (num_laborers : ℕ)
  (laborer_rate : ℕ)
  (total_payroll : ℕ)
  (h1 : total_workers = 31)
  (h2 : num_laborers = 1)
  (h3 : laborer_rate = 82)
  (h4 : total_payroll = 3952) :
  (total_payroll - num_laborers * laborer_rate) / (total_workers - num_laborers) = 129 := by
sorry

end NUMINAMATH_CALUDE_heavy_operator_daily_rate_l178_17899


namespace NUMINAMATH_CALUDE_multiply_93_107_l178_17834

theorem multiply_93_107 : 93 * 107 = 9951 := by
  sorry

end NUMINAMATH_CALUDE_multiply_93_107_l178_17834


namespace NUMINAMATH_CALUDE_right_triangle_leg_ratio_l178_17804

theorem right_triangle_leg_ratio (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_projection : (c - b^2 / c) / (b^2 / c) = 4) : b / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_ratio_l178_17804


namespace NUMINAMATH_CALUDE_complex_equality_l178_17848

theorem complex_equality (a : ℝ) : 
  (Complex.re ((1 + 2*Complex.I) * (a + Complex.I)) = Complex.im ((1 + 2*Complex.I) * (a + Complex.I))) → 
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_l178_17848


namespace NUMINAMATH_CALUDE_power_equation_solution_l178_17828

theorem power_equation_solution :
  ∀ m : ℤ, 3 * 2^2000 - 5 * 2^1999 + 4 * 2^1998 - 2^1997 = m * 2^1997 → m = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l178_17828


namespace NUMINAMATH_CALUDE_exponent_multiplication_l178_17869

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l178_17869


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l178_17844

theorem geometric_sequence_ratio_sum (k p r : ℝ) (h1 : k ≠ 0) (h2 : p ≠ r) :
  k * p^2 - k * r^2 = 5 * (k * p - k * r) → p + r = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l178_17844


namespace NUMINAMATH_CALUDE_arccos_one_half_l178_17855

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_l178_17855


namespace NUMINAMATH_CALUDE_charles_earnings_l178_17846

/-- Charles' earnings problem -/
theorem charles_earnings (housesitting_rate : ℝ) (housesitting_hours : ℝ) (dogs_walked : ℝ) (total_earnings : ℝ) :
  housesitting_rate = 15 →
  housesitting_hours = 10 →
  dogs_walked = 3 →
  total_earnings = 216 →
  (total_earnings - housesitting_rate * housesitting_hours) / dogs_walked = 22 := by
  sorry

end NUMINAMATH_CALUDE_charles_earnings_l178_17846


namespace NUMINAMATH_CALUDE_complex_number_location_l178_17815

theorem complex_number_location (z : ℂ) (h : (z - 1) * Complex.I = Complex.I + 1) : 
  0 < z.re ∧ z.im < 0 := by
sorry

end NUMINAMATH_CALUDE_complex_number_location_l178_17815


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l178_17820

/-- Given a sphere with surface area 4π, its volume is 4π/3 -/
theorem sphere_volume_from_surface_area :
  ∀ r : ℝ, 4 * Real.pi * r^2 = 4 * Real.pi → (4 / 3) * Real.pi * r^3 = (4 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l178_17820


namespace NUMINAMATH_CALUDE_equation_is_false_l178_17839

theorem equation_is_false : 4.58 - (0.45 + 2.58) ≠ 4.58 - 2.58 + 0.45 ∨ 4.58 - (0.45 + 2.58) ≠ 2.45 := by
  sorry

end NUMINAMATH_CALUDE_equation_is_false_l178_17839


namespace NUMINAMATH_CALUDE_closest_cube_approximation_l178_17813

def x : Real := 0.48017

theorem closest_cube_approximation :
  ∀ y ∈ ({0.011, 1.10, 11.0, 110} : Set Real),
  |x^3 - 0.110| < |x^3 - y| := by sorry

end NUMINAMATH_CALUDE_closest_cube_approximation_l178_17813


namespace NUMINAMATH_CALUDE_quadratic_equation_two_real_roots_l178_17879

theorem quadratic_equation_two_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (k + 1) * x^2 - 2 * x + 1 = 0 ∧ (k + 1) * y^2 - 2 * y + 1 = 0) ↔
  (k ≤ 0 ∧ k ≠ -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_real_roots_l178_17879


namespace NUMINAMATH_CALUDE_impossible_to_empty_pile_l178_17888

/-- Represents the state of three piles of stones -/
structure PileState where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ

/-- Allowed operations on the piles -/
inductive Operation
  | Add : Fin 3 → Operation
  | Remove : Fin 3 → Operation

/-- Applies an operation to a PileState -/
def applyOperation (state : PileState) (op : Operation) : PileState :=
  match op with
  | Operation.Add i => 
      match i with
      | 0 => ⟨state.pile1 + state.pile2 + state.pile3, state.pile2, state.pile3⟩
      | 1 => ⟨state.pile1, state.pile2 + state.pile1 + state.pile3, state.pile3⟩
      | 2 => ⟨state.pile1, state.pile2, state.pile3 + state.pile1 + state.pile2⟩
  | Operation.Remove i =>
      match i with
      | 0 => ⟨state.pile1 - (state.pile2 + state.pile3), state.pile2, state.pile3⟩
      | 1 => ⟨state.pile1, state.pile2 - (state.pile1 + state.pile3), state.pile3⟩
      | 2 => ⟨state.pile1, state.pile2, state.pile3 - (state.pile1 + state.pile2)⟩

/-- Theorem stating that it's impossible to make a pile empty -/
theorem impossible_to_empty_pile (initialState : PileState) 
  (h1 : Odd initialState.pile1) 
  (h2 : Odd initialState.pile2) 
  (h3 : Odd initialState.pile3) :
  ∀ (ops : List Operation), 
    let finalState := ops.foldl applyOperation initialState
    ¬(finalState.pile1 = 0 ∨ finalState.pile2 = 0 ∨ finalState.pile3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_impossible_to_empty_pile_l178_17888


namespace NUMINAMATH_CALUDE_min_groups_for_club_l178_17830

/-- Given a club with 30 members and a maximum group size of 12,
    the minimum number of groups required is 3. -/
theorem min_groups_for_club (total_members : ℕ) (max_group_size : ℕ) :
  total_members = 30 →
  max_group_size = 12 →
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_members ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_members → k ≥ num_groups) →
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_members ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_members → k ≥ num_groups) ∧
  (∀ (num_groups : ℕ),
    num_groups * max_group_size ≥ total_members ∧
    (∀ (k : ℕ), k * max_group_size ≥ total_members → k ≥ num_groups) →
    num_groups = 3) :=
by
  sorry


end NUMINAMATH_CALUDE_min_groups_for_club_l178_17830


namespace NUMINAMATH_CALUDE_flour_needed_for_one_batch_l178_17892

/-- The number of cups of flour needed for one batch of cookies -/
def flour_per_batch : ℝ := 4

/-- The number of cups of sugar needed for one batch of cookies -/
def sugar_per_batch : ℝ := 1.5

/-- The total number of cups of flour and sugar needed for 8 batches -/
def total_for_eight_batches : ℝ := 44

theorem flour_needed_for_one_batch :
  flour_per_batch = 4 :=
by
  have h1 : sugar_per_batch = 1.5 := rfl
  have h2 : total_for_eight_batches = 44 := rfl
  have h3 : 8 * flour_per_batch + 8 * sugar_per_batch = total_for_eight_batches := by sorry
  sorry

end NUMINAMATH_CALUDE_flour_needed_for_one_batch_l178_17892


namespace NUMINAMATH_CALUDE_power_of_four_equality_l178_17889

theorem power_of_four_equality (m : ℕ) : 4^m = 4 * 16^3 * 64^2 → m = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_equality_l178_17889


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l178_17840

/-- Given a square with an inscribed circle, if the perimeter of the square (in inches) 
    equals the area of the circle (in square inches), then the diameter of the circle 
    is 16/π inches. -/
theorem inscribed_circle_diameter (s : ℝ) (r : ℝ) (h : s > 0) :
  (4 * s = π * r^2) → (2 * r = 16 / π) := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l178_17840


namespace NUMINAMATH_CALUDE_boxes_on_pallet_l178_17806

/-- 
Given a pallet of boxes with a total weight and the weight of each box,
calculate the number of boxes on the pallet.
-/
theorem boxes_on_pallet (total_weight : ℕ) (box_weight : ℕ) 
  (h1 : total_weight = 267)
  (h2 : box_weight = 89) :
  total_weight / box_weight = 3 :=
by sorry

end NUMINAMATH_CALUDE_boxes_on_pallet_l178_17806


namespace NUMINAMATH_CALUDE_complement_A_in_U_is_correct_l178_17871

-- Define the universal set U
def U : Set Int := {x | -2 ≤ x ∧ x ≤ 6}

-- Define set A
def A : Set Int := {x | ∃ n : Nat, x = 2 * n ∧ n ≤ 3}

-- Define the complement of A in U
def complement_A_in_U : Set Int := U \ A

-- Theorem to prove
theorem complement_A_in_U_is_correct :
  complement_A_in_U = {-2, -1, 1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_is_correct_l178_17871


namespace NUMINAMATH_CALUDE_vote_alteration_l178_17832

theorem vote_alteration (got twi tad : ℕ) (x : ℚ) : 
  got = 10 →
  twi = 12 →
  tad = 20 →
  2 * got = got + twi / 2 + tad * (1 - x / 100) →
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_vote_alteration_l178_17832


namespace NUMINAMATH_CALUDE_a₁₂_eq_15_l178_17803

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  a₄_eq_1 : a 4 = 1
  a₇_plus_a₉_eq_16 : a 7 + a 9 = 16

/-- The 12th term of the arithmetic sequence is 15 -/
theorem a₁₂_eq_15 (seq : ArithmeticSequence) : seq.a 12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_a₁₂_eq_15_l178_17803


namespace NUMINAMATH_CALUDE_third_stick_length_l178_17847

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem third_stick_length (x : ℕ) 
  (h1 : is_even x)
  (h2 : x + 10 > 2)
  (h3 : x + 2 > 10)
  (h4 : 10 + 2 > x) : 
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_third_stick_length_l178_17847


namespace NUMINAMATH_CALUDE_square_difference_630_570_l178_17895

theorem square_difference_630_570 : 630^2 - 570^2 = 72000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_630_570_l178_17895


namespace NUMINAMATH_CALUDE_opposite_abs_sum_l178_17814

theorem opposite_abs_sum (a m n : ℝ) : 
  (|a - 2| + |m + n + 3| = 0) → (a + m + n = -1) := by
sorry

end NUMINAMATH_CALUDE_opposite_abs_sum_l178_17814


namespace NUMINAMATH_CALUDE_sin_cos_identity_l178_17811

theorem sin_cos_identity : 
  Real.sin (18 * π / 180) * Real.sin (78 * π / 180) - 
  Real.cos (162 * π / 180) * Real.cos (78 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l178_17811


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l178_17845

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 17 = 0 → n ≤ 986 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l178_17845


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l178_17827

theorem largest_integer_with_remainder (n : ℕ) : n < 100 ∧ n % 9 = 5 ∧ ∀ m, m < 100 ∧ m % 9 = 5 → m ≤ n ↔ n = 95 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l178_17827


namespace NUMINAMATH_CALUDE_mirror_pieces_l178_17833

theorem mirror_pieces (total : ℕ) (swept : ℕ) (stolen : ℕ) (picked_fraction : ℚ) : 
  total = 60 →
  swept = total / 2 →
  stolen = 3 →
  picked_fraction = 1 / 3 →
  (total - swept - stolen) * picked_fraction = 9 := by
sorry

end NUMINAMATH_CALUDE_mirror_pieces_l178_17833


namespace NUMINAMATH_CALUDE_k_range_l178_17856

theorem k_range (x y k : ℝ) : 
  (3 * x + y = k + 1) → 
  (x + 3 * y = 3) → 
  (0 < x + y) → 
  (x + y < 1) → 
  (-4 < k ∧ k < 0) := by
sorry

end NUMINAMATH_CALUDE_k_range_l178_17856


namespace NUMINAMATH_CALUDE_cookie_distribution_l178_17883

theorem cookie_distribution (total : ℝ) (blue green red : ℝ) : 
  blue = (1/4) * total ∧ 
  green = (5/9) * (total - blue) → 
  (blue + green) / total = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cookie_distribution_l178_17883


namespace NUMINAMATH_CALUDE_one_positive_integer_satisfies_condition_l178_17898

theorem one_positive_integer_satisfies_condition : 
  ∃! (x : ℕ+), 25 - (5 * x.val) > 15 := by sorry

end NUMINAMATH_CALUDE_one_positive_integer_satisfies_condition_l178_17898


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l178_17843

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is increasing on (0, +∞) if f(x) < f(y) for all 0 < x < y -/
def IsIncreasingOnPositiveReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

theorem solution_set_of_inequality (f : ℝ → ℝ)
    (h_odd : IsOdd f)
    (h_incr : IsIncreasingOnPositiveReals f)
    (h_zero : f (-3) = 0) :
    {x : ℝ | (x - 2) * f x < 0} = Set.Ioo (-3) 0 ∪ Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l178_17843


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l178_17823

theorem arithmetic_sequence_common_difference
  (a : ℝ)  -- first term
  (an : ℝ) -- last term
  (s : ℝ)  -- sum of all terms
  (h1 : a = 7)
  (h2 : an = 88)
  (h3 : s = 570)
  : ∃ n : ℕ, n > 1 ∧ (an - a) / (n - 1) = 81 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l178_17823


namespace NUMINAMATH_CALUDE_apple_picking_ratio_l178_17842

/-- The number of apples Lexie picked -/
def lexie_apples : ℕ := 12

/-- The total number of apples picked by Lexie and Tom -/
def total_apples : ℕ := 36

/-- The number of apples Tom picked -/
def tom_apples : ℕ := total_apples - lexie_apples

/-- The ratio of Tom's apples to Lexie's apples -/
def apple_ratio : ℚ := tom_apples / lexie_apples

theorem apple_picking_ratio :
  apple_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_apple_picking_ratio_l178_17842


namespace NUMINAMATH_CALUDE_allan_balloons_l178_17858

def park_balloon_problem (jake_initial : ℕ) (jake_bought : ℕ) (difference : ℕ) : ℕ :=
  let jake_total := jake_initial + jake_bought
  jake_total - difference

theorem allan_balloons :
  park_balloon_problem 3 4 1 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_allan_balloons_l178_17858


namespace NUMINAMATH_CALUDE_second_number_is_sixty_l178_17838

theorem second_number_is_sixty :
  ∀ (a b : ℝ),
  (a + b + 20 + 60) / 4 = (10 + 70 + 28) / 3 + 4 →
  (a = 60 ∨ b = 60) :=
by
  sorry

end NUMINAMATH_CALUDE_second_number_is_sixty_l178_17838


namespace NUMINAMATH_CALUDE_jane_sequins_count_l178_17886

/-- The number of rows of blue sequins -/
def blue_rows : Nat := 6

/-- The number of blue sequins in each row -/
def blue_per_row : Nat := 8

/-- The number of rows of purple sequins -/
def purple_rows : Nat := 5

/-- The number of purple sequins in each row -/
def purple_per_row : Nat := 12

/-- The number of rows of green sequins -/
def green_rows : Nat := 9

/-- The number of green sequins in each row -/
def green_per_row : Nat := 6

/-- The total number of sequins Jane adds to her costume -/
def total_sequins : Nat := blue_rows * blue_per_row + purple_rows * purple_per_row + green_rows * green_per_row

theorem jane_sequins_count : total_sequins = 162 := by
  sorry

end NUMINAMATH_CALUDE_jane_sequins_count_l178_17886


namespace NUMINAMATH_CALUDE_total_time_is_four_hours_l178_17896

def first_movie_length : ℚ := 3/2 -- 1.5 hours
def second_movie_length : ℚ := first_movie_length + 1/2 -- 30 minutes longer
def popcorn_time : ℚ := 1/6 -- 10 minutes in hours
def fries_time : ℚ := 2 * popcorn_time -- twice as long as popcorn time

def total_time : ℚ := first_movie_length + second_movie_length + popcorn_time + fries_time

theorem total_time_is_four_hours : total_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_four_hours_l178_17896


namespace NUMINAMATH_CALUDE_regression_analysis_considerations_l178_17881

/-- Represents the key considerations in regression analysis predictions -/
inductive RegressionConsideration
  | ApplicabilityToSamplePopulation
  | Temporality
  | InfluenceOfSampleRange
  | PredictionPrecision

/-- Represents a regression analysis model -/
structure RegressionModel where
  considerations : List RegressionConsideration

/-- Theorem stating the key considerations in regression analysis predictions -/
theorem regression_analysis_considerations (model : RegressionModel) :
  model.considerations = [
    RegressionConsideration.ApplicabilityToSamplePopulation,
    RegressionConsideration.Temporality,
    RegressionConsideration.InfluenceOfSampleRange,
    RegressionConsideration.PredictionPrecision
  ] := by sorry


end NUMINAMATH_CALUDE_regression_analysis_considerations_l178_17881


namespace NUMINAMATH_CALUDE_sphere_radius_when_area_equals_volume_l178_17897

theorem sphere_radius_when_area_equals_volume (R : ℝ) : R > 0 →
  (4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3) → R = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_when_area_equals_volume_l178_17897


namespace NUMINAMATH_CALUDE_disc_interaction_conservation_l178_17880

/-- Represents a disc with radius and angular velocity -/
structure Disc where
  radius : ℝ
  angularVelocity : ℝ

/-- Theorem: Conservation of angular momentum for two interacting discs -/
theorem disc_interaction_conservation
  (d1 d2 : Disc)
  (h_positive_radius : d1.radius > 0 ∧ d2.radius > 0)
  (h_same_material : True)  -- Placeholder for identical material property
  (h_same_thickness : True) -- Placeholder for identical thickness property
  (h_halt : True) -- Placeholder for the condition that both discs come to a halt
  : d1.angularVelocity * d1.radius ^ 3 = d2.angularVelocity * d2.radius ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_disc_interaction_conservation_l178_17880


namespace NUMINAMATH_CALUDE_second_agency_cost_per_mile_l178_17822

/-- The cost per mile for the second agency that makes both agencies' costs equal at 25.0 miles -/
theorem second_agency_cost_per_mile :
  let first_agency_daily_rate : ℚ := 20.25
  let first_agency_per_mile : ℚ := 0.14
  let second_agency_daily_rate : ℚ := 18.25
  let miles_driven : ℚ := 25.0
  let second_agency_per_mile : ℚ := (first_agency_daily_rate - second_agency_daily_rate + first_agency_per_mile * miles_driven) / miles_driven
  second_agency_per_mile = 0.22 := by sorry

end NUMINAMATH_CALUDE_second_agency_cost_per_mile_l178_17822


namespace NUMINAMATH_CALUDE_find_second_number_l178_17867

theorem find_second_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + 28 + x) / 3) + 4 → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_find_second_number_l178_17867


namespace NUMINAMATH_CALUDE_boats_by_april_l178_17876

def boats_in_month (n : Nat) : Nat :=
  match n with
  | 0 => 4  -- January
  | 1 => 2  -- February
  | m + 2 => 3 * boats_in_month (m + 1)  -- March onwards

def total_boats (n : Nat) : Nat :=
  match n with
  | 0 => boats_in_month 0
  | m + 1 => boats_in_month (m + 1) + total_boats m

theorem boats_by_april : total_boats 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_boats_by_april_l178_17876


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l178_17807

/-- Given two arithmetic sequences {a_n} and {b_n} with sums S_n and T_n respectively,
    if S_n/T_n = 2n/(3n+1) for all natural numbers n, then a_5/b_5 = 9/14 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n : ℕ, S n = (n : ℚ) * (a 1 + a n) / 2) →
  (∀ n : ℕ, T n = (n : ℚ) * (b 1 + b n) / 2) →
  (∀ n : ℕ, S n / T n = (2 * n : ℚ) / (3 * n + 1)) →
  a 5 / b 5 = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l178_17807


namespace NUMINAMATH_CALUDE_gummy_worms_problem_l178_17805

theorem gummy_worms_problem (x : ℝ) : (x / 2^4 = 4) → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_gummy_worms_problem_l178_17805


namespace NUMINAMATH_CALUDE_sum_of_divisors_91_l178_17825

/-- The sum of all positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all positive divisors of 91 is 112 -/
theorem sum_of_divisors_91 : sum_of_divisors 91 = 112 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_91_l178_17825


namespace NUMINAMATH_CALUDE_jakes_weight_l178_17861

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 33 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 153) : 
  jake_weight = 113 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l178_17861


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l178_17854

theorem magnitude_of_complex_fraction (z : ℂ) : z = (2 - I) / (1 + 2*I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l178_17854


namespace NUMINAMATH_CALUDE_box_volume_formula_l178_17853

/-- The volume of an open box formed from a rectangular sheet --/
def boxVolume (x y : ℝ) : ℝ := (16 - 2*x) * (12 - 2*y) * y

/-- Theorem stating the volume of the box --/
theorem box_volume_formula (x y : ℝ) :
  boxVolume x y = 192*y - 32*y^2 - 24*x*y + 4*x*y^2 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_formula_l178_17853


namespace NUMINAMATH_CALUDE_q_value_l178_17816

/-- The coordinates of point A -/
def A : ℝ × ℝ := (0, 12)

/-- The coordinates of point Q -/
def Q : ℝ × ℝ := (3, 12)

/-- The coordinates of point B -/
def B : ℝ × ℝ := (15, 0)

/-- The coordinates of point O -/
def O : ℝ × ℝ := (0, 0)

/-- The coordinates of point C -/
def C (q : ℝ) : ℝ × ℝ := (0, q)

/-- The area of triangle ABC -/
def area_ABC : ℝ := 36

/-- Theorem: If the area of triangle ABC is 36 and the points have the given coordinates, then q = 9 -/
theorem q_value : ∃ q : ℝ, C q = (0, q) ∧ area_ABC = 36 → q = 9 := by
  sorry

end NUMINAMATH_CALUDE_q_value_l178_17816


namespace NUMINAMATH_CALUDE_root_product_theorem_l178_17837

theorem root_product_theorem (a b : ℂ) : 
  (a^4 + a^3 - 1 = 0) → 
  (b^4 + b^3 - 1 = 0) → 
  ((a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l178_17837


namespace NUMINAMATH_CALUDE_equation_solution_l178_17872

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -4 ∧ x₂ = -2) ∧ 
  (∀ x : ℝ, (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l178_17872


namespace NUMINAMATH_CALUDE_bowl_glass_pairings_l178_17835

/-- The number of possible pairings when choosing one bowl from a set of distinct bowls
    and one glass from a set of distinct glasses -/
def num_pairings (num_bowls : ℕ) (num_glasses : ℕ) : ℕ :=
  num_bowls * num_glasses

/-- Theorem stating that with 5 distinct bowls and 6 distinct glasses,
    the number of possible pairings is 30 -/
theorem bowl_glass_pairings :
  num_pairings 5 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_bowl_glass_pairings_l178_17835


namespace NUMINAMATH_CALUDE_rectangle_area_change_l178_17808

theorem rectangle_area_change (L B : ℝ) (h₁ : L > 0) (h₂ : B > 0) :
  let A := L * B
  let L' := 1.20 * L
  let B' := 0.95 * B
  let A' := L' * B'
  A' = 1.14 * A := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l178_17808


namespace NUMINAMATH_CALUDE_fourth_player_score_zero_l178_17819

/-- Represents the score of a player in the chess tournament -/
structure PlayerScore :=
  (score : ℕ)

/-- Represents the scores of all players in the tournament -/
structure TournamentScores :=
  (players : Fin 4 → PlayerScore)

/-- The total points awarded in a tournament with 4 players -/
def totalPoints : ℕ := 12

/-- Theorem stating that if three players have scores 6, 4, and 2, the fourth must have 0 -/
theorem fourth_player_score_zero (t : TournamentScores) :
  (∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (t.players i).score = 6 ∧ 
    (t.players j).score = 4 ∧ 
    (t.players k).score = 2) →
  (∃ (l : Fin 4), (∀ m : Fin 4, m ≠ l → 
    (t.players m).score = 6 ∨ 
    (t.players m).score = 4 ∨ 
    (t.players m).score = 2) ∧ 
  (t.players l).score = 0) :=
by sorry

end NUMINAMATH_CALUDE_fourth_player_score_zero_l178_17819


namespace NUMINAMATH_CALUDE_jersey_tshirt_cost_difference_l178_17874

/-- Calculates the final cost difference between a jersey and a t-shirt --/
theorem jersey_tshirt_cost_difference :
  let jersey_price : ℚ := 115
  let tshirt_price : ℚ := 25
  let jersey_discount : ℚ := 10 / 100
  let tshirt_discount : ℚ := 15 / 100
  let sales_tax : ℚ := 8 / 100
  let jersey_shipping : ℚ := 5
  let tshirt_shipping : ℚ := 3

  let jersey_discounted := jersey_price * (1 - jersey_discount)
  let tshirt_discounted := tshirt_price * (1 - tshirt_discount)

  let jersey_with_tax := jersey_discounted * (1 + sales_tax)
  let tshirt_with_tax := tshirt_discounted * (1 + sales_tax)

  let jersey_final := jersey_with_tax + jersey_shipping
  let tshirt_final := tshirt_with_tax + tshirt_shipping

  jersey_final - tshirt_final = 90.83 := by sorry

end NUMINAMATH_CALUDE_jersey_tshirt_cost_difference_l178_17874


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l178_17801

-- Define set A
def A : Set ℝ := {x | |x| ≤ 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l178_17801


namespace NUMINAMATH_CALUDE_max_edges_triangle_free_30_max_edges_k4_free_30_l178_17809

/-- The maximum number of edges in a triangle-free graph with 30 vertices -/
def max_edges_triangle_free (n : ℕ) : ℕ :=
  if n = 30 then 225 else 0

/-- The maximum number of edges in a K₄-free graph with 30 vertices -/
def max_edges_k4_free (n : ℕ) : ℕ :=
  if n = 30 then 300 else 0

/-- Theorem stating the maximum number of edges in a triangle-free graph with 30 vertices -/
theorem max_edges_triangle_free_30 :
  max_edges_triangle_free 30 = 225 := by sorry

/-- Theorem stating the maximum number of edges in a K₄-free graph with 30 vertices -/
theorem max_edges_k4_free_30 :
  max_edges_k4_free 30 = 300 := by sorry

end NUMINAMATH_CALUDE_max_edges_triangle_free_30_max_edges_k4_free_30_l178_17809


namespace NUMINAMATH_CALUDE_product_of_solution_l178_17864

open Real

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

-- State the theorem
theorem product_of_solution (x y : ℝ) 
  (eq1 : (floor x : ℝ) + frac y = 1.7)
  (eq2 : frac x + (floor y : ℝ) = 3.6) :
  x * y = 5.92 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solution_l178_17864


namespace NUMINAMATH_CALUDE_divide_multiply_add_subtract_l178_17810

theorem divide_multiply_add_subtract (x n : ℝ) : x = 40 → ((x / n) * 5 + 10 - 12 = 48 ↔ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_divide_multiply_add_subtract_l178_17810


namespace NUMINAMATH_CALUDE_nickels_left_l178_17887

def initial_cents : ℕ := 475
def exchanged_cents : ℕ := 75
def cents_per_nickel : ℕ := 5
def cents_per_dime : ℕ := 10

def peter_proportion : ℚ := 2/5
def randi_proportion : ℚ := 3/5
def paula_proportion : ℚ := 1/10

theorem nickels_left : ℕ := by
  -- Prove that Ray is left with 82 nickels
  sorry

end NUMINAMATH_CALUDE_nickels_left_l178_17887


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l178_17875

theorem simplify_and_evaluate :
  let a : ℝ := (1/2 : ℝ) + Real.sqrt (1/2)
  (a + Real.sqrt 3) * (a - Real.sqrt 3) - a * (a - 6) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l178_17875


namespace NUMINAMATH_CALUDE_range_of_expression_l178_17802

theorem range_of_expression (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) : 
  -π/6 < 2*α - β/3 ∧ 2*α - β/3 < π := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l178_17802


namespace NUMINAMATH_CALUDE_mixed_doubles_pairing_methods_l178_17817

theorem mixed_doubles_pairing_methods (total_players : Nat) (male_players : Nat) (female_players : Nat) 
  (selected_male : Nat) (selected_female : Nat) :
  total_players = male_players + female_players →
  male_players = 5 →
  female_players = 4 →
  selected_male = 2 →
  selected_female = 2 →
  (Nat.choose male_players selected_male) * (Nat.choose female_players selected_female) * 
  (Nat.factorial selected_male) = 120 := by
sorry

end NUMINAMATH_CALUDE_mixed_doubles_pairing_methods_l178_17817


namespace NUMINAMATH_CALUDE_interest_difference_approx_l178_17859

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

/-- The positive difference between compound and simple interest balances -/
def interest_difference (principal : ℝ) (compound_rate : ℝ) (simple_rate : ℝ) (time : ℕ) : ℝ :=
  |simple_interest principal simple_rate time - compound_interest principal compound_rate time|

theorem interest_difference_approx :
  ∃ ε > 0, |interest_difference 10000 0.04 0.06 12 - 1189| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_difference_approx_l178_17859


namespace NUMINAMATH_CALUDE_box_length_with_cubes_l178_17857

/-- Given a box with dimensions L × 15 × 6 inches that can be filled entirely
    with 90 identical cubes leaving no space unfilled, prove that the length L
    of the box is 27 inches. -/
theorem box_length_with_cubes (L : ℕ) : 
  (∃ (s : ℕ), L * 15 * 6 = 90 * s^3 ∧ s ∣ 15 ∧ s ∣ 6) → L = 27 := by
  sorry

end NUMINAMATH_CALUDE_box_length_with_cubes_l178_17857
