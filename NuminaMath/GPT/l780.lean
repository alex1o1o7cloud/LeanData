import Mathlib

namespace NUMINAMATH_GPT_fraction_simplification_l780_78020

theorem fraction_simplification (a b c : ℝ) :
  (4 * a^2 + 2 * c^2 - 4 * b^2 - 8 * b * c) / (3 * a^2 + 6 * a * c - 3 * c^2 - 6 * a * b) =
  (4 / 3) * ((a - 2 * b + c) * (a - c)) / ((a - b + c) * (a - b - c)) :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l780_78020


namespace NUMINAMATH_GPT_largest_number_in_set_l780_78007

theorem largest_number_in_set :
  ∀ (a b c d : ℤ), (a ∈ [0, 2, -1, -2]) → (b ∈ [0, 2, -1, -2]) → (c ∈ [0, 2, -1, -2]) → (d ∈ [0, 2, -1, -2])
  → (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  → max (max a b) (max c d) = 2
  := 
by
  sorry

end NUMINAMATH_GPT_largest_number_in_set_l780_78007


namespace NUMINAMATH_GPT_min_value_of_y_l780_78062

theorem min_value_of_y (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (∃ y : ℝ, y = 1 / a + 4 / b ∧ (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' = 1 → 1 / a' + 4 / b' ≥ y)) ∧ 
  (∀ y : ℝ, y = 1 / a + 4 / b → y ≥ 9) :=
sorry

end NUMINAMATH_GPT_min_value_of_y_l780_78062


namespace NUMINAMATH_GPT_at_least_three_equal_l780_78060

theorem at_least_three_equal (a b c d : ℕ) (h1 : (a + b) ^ 2 ∣ c * d)
                                (h2 : (a + c) ^ 2 ∣ b * d)
                                (h3 : (a + d) ^ 2 ∣ b * c)
                                (h4 : (b + c) ^ 2 ∣ a * d)
                                (h5 : (b + d) ^ 2 ∣ a * c)
                                (h6 : (c + d) ^ 2 ∣ a * b) :
  ∃ x : ℕ, (x = a ∧ x = b ∧ x = c) ∨ (x = a ∧ x = b ∧ x = d) ∨ (x = a ∧ x = c ∧ x = d) ∨ (x = b ∧ x = c ∧ x = d) :=
sorry

end NUMINAMATH_GPT_at_least_three_equal_l780_78060


namespace NUMINAMATH_GPT_max_mn_square_proof_l780_78070

noncomputable def max_mn_square (m n : ℕ) : ℕ :=
m^2 + n^2

theorem max_mn_square_proof (m n : ℕ) (h1 : 1 ≤ m ∧ m ≤ 2005) (h2 : 1 ≤ n ∧ n ≤ 2005) (h3 : (n^2 + 2 * m * n - 2 * m^2)^2 = 1) : 
max_mn_square m n ≤ 702036 :=
sorry

end NUMINAMATH_GPT_max_mn_square_proof_l780_78070


namespace NUMINAMATH_GPT_geese_percentage_non_ducks_l780_78058

theorem geese_percentage_non_ducks :
  let total_birds := 100
  let geese := 0.20 * total_birds
  let swans := 0.30 * total_birds
  let herons := 0.15 * total_birds
  let ducks := 0.25 * total_birds
  let pigeons := 0.10 * total_birds
  let non_duck_birds := total_birds - ducks
  (geese / non_duck_birds) * 100 = 27 := 
by
  sorry

end NUMINAMATH_GPT_geese_percentage_non_ducks_l780_78058


namespace NUMINAMATH_GPT_eight_digit_product_1400_l780_78024

def eight_digit_numbers_count : Nat :=
  sorry

theorem eight_digit_product_1400 : eight_digit_numbers_count = 5880 :=
  sorry

end NUMINAMATH_GPT_eight_digit_product_1400_l780_78024


namespace NUMINAMATH_GPT_total_students_in_school_l780_78033

theorem total_students_in_school 
  (below_8_percent : ℝ) (above_8_ratio : ℝ) (students_8 : ℕ) : 
  below_8_percent = 0.20 → above_8_ratio = 2/3 → students_8 = 12 → 
  (∃ T : ℕ, T = 25) :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_school_l780_78033


namespace NUMINAMATH_GPT_sequence_is_arithmetic_not_geometric_l780_78074

noncomputable def a := Real.log 3 / Real.log 2
noncomputable def b := Real.log 6 / Real.log 2
noncomputable def c := Real.log 12 / Real.log 2

theorem sequence_is_arithmetic_not_geometric : 
  (b - a = c - b) ∧ (b / a ≠ c / b) := 
by
  sorry

end NUMINAMATH_GPT_sequence_is_arithmetic_not_geometric_l780_78074


namespace NUMINAMATH_GPT_degree_odd_of_polynomials_l780_78043

theorem degree_odd_of_polynomials 
  (d : ℕ) 
  (P Q : Polynomial ℝ) 
  (hP_deg : P.degree = d) 
  (h_eq : P^2 + 1 = (X^2 + 1) * Q^2) 
  : Odd d :=
sorry

end NUMINAMATH_GPT_degree_odd_of_polynomials_l780_78043


namespace NUMINAMATH_GPT_problem_I4_1_l780_78051

variable (A D E B C : Type) [Field A] [Field D] [Field E] [Field B] [Field C]
variable (AD DB DE BC : ℚ)
variable (a : ℚ)
variable (h1 : DE = BC) -- DE parallel to BC
variable (h2 : AD = 4)
variable (h3 : DB = 6)
variable (h4 : DE = 6)

theorem problem_I4_1 : a = 15 :=
  by
  sorry

end NUMINAMATH_GPT_problem_I4_1_l780_78051


namespace NUMINAMATH_GPT_length_of_second_dimension_l780_78056

def volume_of_box (w : ℝ) : ℝ :=
  (w - 16) * (46 - 16) * 8

theorem length_of_second_dimension (w : ℝ) (h_volume : volume_of_box w = 4800) : w = 36 :=
by
  sorry

end NUMINAMATH_GPT_length_of_second_dimension_l780_78056


namespace NUMINAMATH_GPT_inequality_solution_l780_78038

theorem inequality_solution {a b x : ℝ} 
  (h_sol_set : -1 < x ∧ x < 1) 
  (h1 : x - a > 2) 
  (h2 : b - 2 * x > 0) : 
  (a + b) ^ 2021 = -1 := 
by 
  sorry 

end NUMINAMATH_GPT_inequality_solution_l780_78038


namespace NUMINAMATH_GPT_symmetric_point_origin_l780_78048

theorem symmetric_point_origin (x y : Int) (hx : x = -(-4)) (hy : y = -(3)) :
    (x, y) = (4, -3) := by
  sorry

end NUMINAMATH_GPT_symmetric_point_origin_l780_78048


namespace NUMINAMATH_GPT_Oshea_needs_50_small_planters_l780_78030

structure Planter :=
  (large : ℕ)     -- Number of large planters
  (medium : ℕ)    -- Number of medium planters
  (small : ℕ)     -- Number of small planters
  (capacity_large : ℕ := 20) -- Capacity of large planter
  (capacity_medium : ℕ := 10) -- Capacity of medium planter
  (capacity_small : ℕ := 4)  -- Capacity of small planter

structure Seeds :=
  (basil : ℕ)     -- Number of basil seeds
  (cilantro : ℕ)  -- Number of cilantro seeds
  (parsley : ℕ)   -- Number of parsley seeds

noncomputable def small_planters_needed (planters : Planter) (seeds : Seeds) : ℕ :=
  let basil_in_large := min seeds.basil (planters.large * planters.capacity_large)
  let basil_left := seeds.basil - basil_in_large
  let basil_in_medium := min basil_left (planters.medium * planters.capacity_medium)
  let basil_remaining := basil_left - basil_in_medium
  
  let cilantro_in_medium := min seeds.cilantro ((planters.medium * planters.capacity_medium) - basil_in_medium)
  let cilantro_remaining := seeds.cilantro - cilantro_in_medium
  
  let parsley_total := seeds.parsley + basil_remaining + cilantro_remaining
  parsley_total / planters.capacity_small

theorem Oshea_needs_50_small_planters :
  small_planters_needed 
    { large := 4, medium := 8, small := 0 }
    { basil := 200, cilantro := 160, parsley := 120 } = 50 := 
sorry

end NUMINAMATH_GPT_Oshea_needs_50_small_planters_l780_78030


namespace NUMINAMATH_GPT_no_right_triangle_with_sqrt_2016_side_l780_78076

theorem no_right_triangle_with_sqrt_2016_side :
  ¬ ∃ (a b : ℤ), (a * a + b * b = 2016) ∨ (a * a + 2016 = b * b) :=
by
  sorry

end NUMINAMATH_GPT_no_right_triangle_with_sqrt_2016_side_l780_78076


namespace NUMINAMATH_GPT_thirty_five_million_in_scientific_notation_l780_78039

def million := 10^6

def sales_revenue (x : ℝ) := x * million

theorem thirty_five_million_in_scientific_notation :
  sales_revenue 35 = 3.5 * 10^7 :=
by
  sorry

end NUMINAMATH_GPT_thirty_five_million_in_scientific_notation_l780_78039


namespace NUMINAMATH_GPT_multiply_or_divide_inequality_by_negative_number_l780_78087

theorem multiply_or_divide_inequality_by_negative_number {a b c : ℝ} (h : a < b) (hc : c < 0) :
  c * a > c * b ∧ a / c > b / c :=
sorry

end NUMINAMATH_GPT_multiply_or_divide_inequality_by_negative_number_l780_78087


namespace NUMINAMATH_GPT_option_C_correct_l780_78068

theorem option_C_correct (x : ℝ) (hx : 0 < x) : x + 1 / x ≥ 2 :=
sorry

end NUMINAMATH_GPT_option_C_correct_l780_78068


namespace NUMINAMATH_GPT_difference_fewer_children_than_adults_l780_78017

theorem difference_fewer_children_than_adults : 
  ∀ (C S : ℕ), 2 * C = S → 58 + C + S = 127 → (58 - C = 35) :=
by
  intros C S h1 h2
  sorry

end NUMINAMATH_GPT_difference_fewer_children_than_adults_l780_78017


namespace NUMINAMATH_GPT_smallest_positive_integer_l780_78041

theorem smallest_positive_integer (N : ℕ) :
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 7 = 4) ∧ (N % 11 = 9) → N = 207 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l780_78041


namespace NUMINAMATH_GPT_probability_of_forming_CHORAL_is_correct_l780_78045

-- Definitions for selecting letters with given probabilities
def probability_select_C_A_L_from_CAMEL : ℚ :=
  1 / 10

def probability_select_H_O_R_from_SHRUB : ℚ :=
  1 / 10

def probability_select_G_from_GLOW : ℚ :=
  1 / 2

-- Calculating the total probability of selecting letters to form "CHORAL"
def probability_form_CHORAL : ℚ :=
  probability_select_C_A_L_from_CAMEL * 
  probability_select_H_O_R_from_SHRUB * 
  probability_select_G_from_GLOW

theorem probability_of_forming_CHORAL_is_correct :
  probability_form_CHORAL = 1 / 200 :=
by
  -- Statement to be proven here
  sorry

end NUMINAMATH_GPT_probability_of_forming_CHORAL_is_correct_l780_78045


namespace NUMINAMATH_GPT_suitable_for_comprehensive_survey_l780_78080

-- Define the conditions
def is_comprehensive_survey (group_size : ℕ) (is_specific_group : Bool) : Bool :=
  is_specific_group ∧ (group_size < 100)  -- assuming "small" means fewer than 100 individuals/items

def is_sampling_survey (group_size : ℕ) (is_specific_group : Bool) : Bool :=
  ¬is_comprehensive_survey group_size is_specific_group

-- Define the surveys
def option_A (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_comprehensive_survey group_size is_specific_group

def option_B (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

def option_C (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

def option_D (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

-- Question: Which of the following surveys is suitable for a comprehensive survey given conditions
theorem suitable_for_comprehensive_survey :
  ∀ (group_size_A group_size_B group_size_C group_size_D : ℕ) 
    (is_specific_group_A is_specific_group_B is_specific_group_C is_specific_group_D : Bool),
  option_A group_size_A is_specific_group_A ↔ 
  ((option_B group_size_B is_specific_group_B = false) ∧ 
   (option_C group_size_C is_specific_group_C = false) ∧ 
   (option_D group_size_D is_specific_group_D = false)) :=
by
  sorry

end NUMINAMATH_GPT_suitable_for_comprehensive_survey_l780_78080


namespace NUMINAMATH_GPT_circle_radius_squared_l780_78064

theorem circle_radius_squared (r : ℝ) 
  (AB CD: ℝ) 
  (BP angleAPD : ℝ) 
  (P_outside_circle: True) 
  (AB_eq_12 : AB = 12) 
  (CD_eq_9 : CD = 9) 
  (AngleAPD_eq_45 : angleAPD = 45) 
  (BP_eq_10 : BP = 10) : r^2 = 73 :=
sorry

end NUMINAMATH_GPT_circle_radius_squared_l780_78064


namespace NUMINAMATH_GPT_subset1_squares_equals_product_subset2_squares_equals_product_l780_78015

theorem subset1_squares_equals_product :
  (1^2 + 3^2 + 4^2 + 9^2 + 107^2 = 1 * 3 * 4 * 9 * 107) :=
sorry

theorem subset2_squares_equals_product :
  (3^2 + 4^2 + 9^2 + 107^2 + 11555^2 = 3 * 4 * 9 * 107 * 11555) :=
sorry

end NUMINAMATH_GPT_subset1_squares_equals_product_subset2_squares_equals_product_l780_78015


namespace NUMINAMATH_GPT_factorial_sum_simplify_l780_78097

theorem factorial_sum_simplify :
  7 * (Nat.factorial 7) + 5 * (Nat.factorial 5) + 3 * (Nat.factorial 3) + (Nat.factorial 3) = 35904 :=
by
  sorry

end NUMINAMATH_GPT_factorial_sum_simplify_l780_78097


namespace NUMINAMATH_GPT_ray_two_digit_number_l780_78006

theorem ray_two_digit_number (a b n : ℕ) (ha : 1 ≤ a) (hb : b ≤ 9) (hn : n = 10 * a + b) (h1 : n = 4 * (a + b) + 3) (h2 : n + 18 = 10 * b + a) : n = 35 := by
  sorry

end NUMINAMATH_GPT_ray_two_digit_number_l780_78006


namespace NUMINAMATH_GPT_arithmetic_sequence_10th_term_l780_78075

theorem arithmetic_sequence_10th_term (a_1 : ℕ) (d : ℕ) (n : ℕ) 
  (h1 : a_1 = 1) (h2 : d = 3) (h3 : n = 10) : (a_1 + (n - 1) * d) = 28 := by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_10th_term_l780_78075


namespace NUMINAMATH_GPT_factor_polynomial_l780_78053

theorem factor_polynomial (n : ℕ) (hn : 2 ≤ n) 
  (a : ℝ) (b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℤ, n < 2 * k + 1 ∧ 2 * k + 1 < 3 * n ∧ 
  a = (-(2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n)))) ^ (2 * n / (2 * n - 1)) ∧ 
  b = (2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n))) ^ (2 / (2 * n - 1)) := sorry

end NUMINAMATH_GPT_factor_polynomial_l780_78053


namespace NUMINAMATH_GPT_percent_with_university_diploma_l780_78008

theorem percent_with_university_diploma (a b c d : ℝ) (h1 : a = 0.12) (h2 : b = 0.25) (h3 : c = 0.40) 
    (h4 : d = c - a) (h5 : ¬c = 1) : 
    d + (b * (1 - c)) = 0.43 := 
by 
    sorry

end NUMINAMATH_GPT_percent_with_university_diploma_l780_78008


namespace NUMINAMATH_GPT_find_n_values_l780_78025

theorem find_n_values (n : ℕ) (h1 : 0 < n) : 
  (∃ (a : ℕ), n * 2^n + 1 = a * a) ↔ (n = 2 ∨ n = 3) := 
by
  sorry

end NUMINAMATH_GPT_find_n_values_l780_78025


namespace NUMINAMATH_GPT_intersection_complement_l780_78054

open Set

noncomputable def U : Set ℝ := univ

def A : Set ℝ := {x | x^2 - 2 * x < 0}

def B : Set ℝ := {x | x > 1}

theorem intersection_complement (x : ℝ) :
  x ∈ (A ∩ (U \ B)) ↔ 0 < x ∧ x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l780_78054


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l780_78072

theorem asymptotes_of_hyperbola :
  ∀ x y : ℝ, (y^2 / 4 - x^2 / 9 = 1) → (y = (2 / 3) * x ∨ y = -(2 / 3) * x) :=
by
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l780_78072


namespace NUMINAMATH_GPT_ellipse_shortest_major_axis_l780_78098

theorem ellipse_shortest_major_axis (P : ℝ × ℝ) (a b : ℝ) 
  (ha : a > b) (hb : b > 0) (hP_on_line : P.2 = P.1 + 2)
  (h_foci_hyperbola : ∃ c : ℝ, c = 1 ∧ a^2 - b^2 = c^2) :
  (∃ a b : ℝ, a^2 = 5 ∧ b^2 = 4 ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1)) :=
sorry

end NUMINAMATH_GPT_ellipse_shortest_major_axis_l780_78098


namespace NUMINAMATH_GPT_rationalize_denominator_l780_78040

theorem rationalize_denominator :
  (35 / Real.sqrt 35) = Real.sqrt 35 :=
sorry

end NUMINAMATH_GPT_rationalize_denominator_l780_78040


namespace NUMINAMATH_GPT_rotate_cd_to_cd_l780_78095

def rotate180 (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, -p.2)

theorem rotate_cd_to_cd' :
  let C := (-1, 2)
  let C' := (1, -2)
  let D := (3, 2)
  let D' := (-3, -2)
  rotate180 C = C' ∧ rotate180 D = D' :=
by
  sorry

end NUMINAMATH_GPT_rotate_cd_to_cd_l780_78095


namespace NUMINAMATH_GPT_head_start_distance_l780_78086

theorem head_start_distance (v_A v_B L H : ℝ) (h1 : v_A = 15 / 13 * v_B)
    (h2 : t_A = L / v_A) (h3 : t_B = (L - H) / v_B) (h4 : t_B = t_A - 0.25 * L / v_B) :
    H = 23 / 60 * L :=
sorry

end NUMINAMATH_GPT_head_start_distance_l780_78086


namespace NUMINAMATH_GPT_qualified_light_bulb_prob_l780_78027

def prob_factory_A := 0.7
def prob_factory_B := 0.3
def qual_rate_A := 0.9
def qual_rate_B := 0.8

theorem qualified_light_bulb_prob :
  prob_factory_A * qual_rate_A + prob_factory_B * qual_rate_B = 0.87 :=
by
  sorry

end NUMINAMATH_GPT_qualified_light_bulb_prob_l780_78027


namespace NUMINAMATH_GPT_sum_abs_eq_pos_or_neg_three_l780_78081

theorem sum_abs_eq_pos_or_neg_three (x y : Real) (h1 : abs x = 1) (h2 : abs y = 2) (h3 : x * y > 0) :
    x + y = 3 ∨ x + y = -3 :=
by
  sorry

end NUMINAMATH_GPT_sum_abs_eq_pos_or_neg_three_l780_78081


namespace NUMINAMATH_GPT_original_cost_of_car_l780_78079

-- Conditions
variables (C : ℝ)
variables (spent_on_repairs : ℝ := 8000)
variables (selling_price : ℝ := 68400)
variables (profit_percent : ℝ := 54.054054054054056)

-- Statement to be proved
theorem original_cost_of_car :
  C + spent_on_repairs = selling_price - (profit_percent / 100) * C :=
sorry

end NUMINAMATH_GPT_original_cost_of_car_l780_78079


namespace NUMINAMATH_GPT_ratio_of_b_to_c_l780_78005

theorem ratio_of_b_to_c (a b c : ℝ) 
  (h1 : a / b = 11 / 3) 
  (h2 : a / c = 0.7333333333333333) : 
  b / c = 1 / 5 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_b_to_c_l780_78005


namespace NUMINAMATH_GPT_train_crossing_pole_time_l780_78055

/-- 
Given the conditions:
1. The train is running at a speed of 60 km/hr.
2. The length of the train is 66.66666666666667 meters.
Prove that it takes 4 seconds for the train to cross the pole.
-/
theorem train_crossing_pole_time :
  let speed_km_hr := 60
  let length_m := 66.66666666666667
  let conversion_factor := 1000 / 3600
  let speed_m_s := speed_km_hr * conversion_factor
  let time := length_m / speed_m_s
  time = 4 :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_pole_time_l780_78055


namespace NUMINAMATH_GPT_triangle_area_correct_l780_78001

noncomputable def area_of_triangle 
  (a b c : ℝ) (ha : a = Real.sqrt 29) (hb : b = Real.sqrt 13) (hc : c = Real.sqrt 34) : ℝ :=
  let cosC := (b^2 + c^2 - a^2) / (2 * b * c)
  let sinC := Real.sqrt (1 - cosC^2)
  (1 / 2) * b * c * sinC

theorem triangle_area_correct : area_of_triangle (Real.sqrt 29) (Real.sqrt 13) (Real.sqrt 34) 
  (by rfl) (by rfl) (by rfl) = 19 / 2 :=
sorry

end NUMINAMATH_GPT_triangle_area_correct_l780_78001


namespace NUMINAMATH_GPT_find_number_l780_78078

/-- 
  Given that 23% of a number x is equal to 150, prove that x equals 15000 / 23.
-/
theorem find_number (x : ℝ) (h : (23 / 100) * x = 150) : x = 15000 / 23 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l780_78078


namespace NUMINAMATH_GPT_coeff_x3_product_l780_78093

open Polynomial

noncomputable def poly1 := (C 3 * X ^ 3) + (C 2 * X ^ 2) + (C 4 * X) + (C 5)
noncomputable def poly2 := (C 4 * X ^ 3) + (C 6 * X ^ 2) + (C 5 * X) + (C 2)

theorem coeff_x3_product : coeff (poly1 * poly2) 3 = 10 := by
  sorry

end NUMINAMATH_GPT_coeff_x3_product_l780_78093


namespace NUMINAMATH_GPT_length_of_room_l780_78035

theorem length_of_room (Area Width Length : ℝ) (h1 : Area = 10) (h2 : Width = 2) (h3 : Area = Length * Width) : Length = 5 :=
by
  sorry

end NUMINAMATH_GPT_length_of_room_l780_78035


namespace NUMINAMATH_GPT_remainder_of_sum_division_l780_78091

theorem remainder_of_sum_division (x y : ℕ) (k m : ℕ) 
  (hx : x = 90 * k + 75) (hy : y = 120 * m + 115) :
  (x + y) % 30 = 10 :=
by sorry

end NUMINAMATH_GPT_remainder_of_sum_division_l780_78091


namespace NUMINAMATH_GPT_sqrt_20n_integer_exists_l780_78022

theorem sqrt_20n_integer_exists : 
  ∃ n : ℤ, 0 ≤ n ∧ ∃ k : ℤ, k * k = 20 * n :=
sorry

end NUMINAMATH_GPT_sqrt_20n_integer_exists_l780_78022


namespace NUMINAMATH_GPT_shiela_used_seven_colors_l780_78069

theorem shiela_used_seven_colors (total_blocks : ℕ) (blocks_per_color : ℕ) 
    (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) : 
    total_blocks / blocks_per_color = 7 :=
by
  sorry

end NUMINAMATH_GPT_shiela_used_seven_colors_l780_78069


namespace NUMINAMATH_GPT_halfway_between_one_third_and_one_eighth_l780_78077

theorem halfway_between_one_third_and_one_eighth : (1/3 + 1/8) / 2 = 11 / 48 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_halfway_between_one_third_and_one_eighth_l780_78077


namespace NUMINAMATH_GPT_minimum_reciprocal_sum_of_roots_l780_78009

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) := 2 * x^2 + b * x + c

theorem minimum_reciprocal_sum_of_roots {b c : ℝ} {x1 x2 : ℝ} 
  (h1: f (-10) b c = f 12 b c)
  (h2: f x1 b c = 0)
  (h3: f x2 b c = 0)
  (h4: 0 < x1)
  (h5: 0 < x2)
  (h6: x1 + x2 = 2) :
  (1 / x1 + 1 / x2) = 2 :=
sorry

end NUMINAMATH_GPT_minimum_reciprocal_sum_of_roots_l780_78009


namespace NUMINAMATH_GPT_number_of_functions_l780_78096

-- Define the set of conditions
variables (x y : ℝ)

def relation1 := x - y = 0
def relation2 := y^2 = x
def relation3 := |y| = 2 * x
def relation4 := y^2 = x^2
def relation5 := y = 3 - x
def relation6 := y = 2 * x^2 - 1
def relation7 := y = 3 / x

-- Prove that there are 4 unambiguous functions of y with respect to x
theorem number_of_functions : 4 = 4 := sorry

end NUMINAMATH_GPT_number_of_functions_l780_78096


namespace NUMINAMATH_GPT_parabola_transformation_l780_78044

def original_parabola (x : ℝ) : ℝ := 3 * x^2

def shifted_left (x : ℝ) : ℝ := original_parabola (x + 1)

def shifted_down (x : ℝ) : ℝ := shifted_left x - 2

theorem parabola_transformation :
  shifted_down x = 3 * (x + 1)^2 - 2 :=
sorry

end NUMINAMATH_GPT_parabola_transformation_l780_78044


namespace NUMINAMATH_GPT_bryan_push_ups_l780_78061

theorem bryan_push_ups (sets : ℕ) (push_ups_per_set : ℕ) (fewer_in_last_set : ℕ) 
  (h1 : sets = 3) (h2 : push_ups_per_set = 15) (h3 : fewer_in_last_set = 5) :
  (sets - 1) * push_ups_per_set + (push_ups_per_set - fewer_in_last_set) = 40 := by 
  -- We are setting sorry here to skip the proof.
  sorry

end NUMINAMATH_GPT_bryan_push_ups_l780_78061


namespace NUMINAMATH_GPT_find_a_l780_78063

theorem find_a (f : ℕ → ℕ) (a : ℕ) 
  (h1 : ∀ x : ℕ, f (x + 1) = x) 
  (h2 : f a = 8) : a = 9 :=
sorry

end NUMINAMATH_GPT_find_a_l780_78063


namespace NUMINAMATH_GPT_Donovan_Mitchell_goal_l780_78059

theorem Donovan_Mitchell_goal 
  (current_avg : ℕ) 
  (current_games : ℕ) 
  (target_avg : ℕ) 
  (total_games : ℕ) 
  (remaining_games : ℕ) 
  (points_scored_so_far : ℕ)
  (points_needed_total : ℕ)
  (points_needed_remaining : ℕ) :
  (current_avg = 26) ∧
  (current_games = 15) ∧
  (target_avg = 30) ∧
  (total_games = 20) ∧
  (remaining_games = 5) ∧
  (points_scored_so_far = current_avg * current_games) ∧
  (points_needed_total = target_avg * total_games) ∧
  (points_needed_remaining = points_needed_total - points_scored_so_far) →
  (points_needed_remaining / remaining_games = 42) :=
by
  sorry

end NUMINAMATH_GPT_Donovan_Mitchell_goal_l780_78059


namespace NUMINAMATH_GPT_stratified_sampling_females_l780_78019

theorem stratified_sampling_females :
  let males := 500
  let females := 400
  let total_students := 900
  let total_surveyed := 45
  let males_surveyed := 25
  ((males_surveyed : ℚ) / males) * females = 20 := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_females_l780_78019


namespace NUMINAMATH_GPT_polar_coordinates_of_point_l780_78028

theorem polar_coordinates_of_point :
  ∀ (x y : ℝ) (r θ : ℝ), x = -1 ∧ y = 1 ∧ r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x) ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi
  → r = Real.sqrt 2 ∧ θ = 3 * Real.pi / 4 := 
by
  intros x y r θ h
  sorry

end NUMINAMATH_GPT_polar_coordinates_of_point_l780_78028


namespace NUMINAMATH_GPT_geometric_sequence_a2_l780_78031

theorem geometric_sequence_a2 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h1 : a 1 = 1/4) 
  (h3_h5 : a 3 * a 5 = 4 * (a 4 - 1)) 
  (h_seq : ∀ n : ℕ, a n = a 1 * q ^ (n - 1)) :
  a 2 = 1/2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a2_l780_78031


namespace NUMINAMATH_GPT_find_m_l780_78090

noncomputable def vector_a : ℝ × ℝ := (1, -3)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 2)
noncomputable def vector_sum (m : ℝ) : ℝ × ℝ := (1 + m, -1)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_m (m : ℝ) : dot_product vector_a (vector_sum m) = 0 → m = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l780_78090


namespace NUMINAMATH_GPT_rectangles_on_grid_l780_78010

-- Define the grid dimensions
def m := 3
def n := 2

-- Define a function to count the total number of rectangles formed by the grid.
def count_rectangles (m n : ℕ) : ℕ := 
  (m * (m - 1) / 2 + n * (n - 1) / 2) * (n * (n - 1) / 2 + m * (m - 1) / 2) 

-- State the theorem we need to prove
theorem rectangles_on_grid : count_rectangles m n = 14 :=
  sorry

end NUMINAMATH_GPT_rectangles_on_grid_l780_78010


namespace NUMINAMATH_GPT_find_p_r_l780_78000

-- Definitions of the polynomials
def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q
def g (x : ℝ) (r s : ℝ) : ℝ := x^2 + r * x + s

-- Lean statement of the proof problem:
theorem find_p_r (p q r s : ℝ) (h1 : p ≠ r) (h2 : g (-p / 2) r s = 0) 
  (h3 : f (-r / 2) p q = 0) (h4 : ∀ x : ℝ, f x p q = g x r s) 
  (h5 : f 50 p q = -50) : p + r = -200 := 
sorry

end NUMINAMATH_GPT_find_p_r_l780_78000


namespace NUMINAMATH_GPT_volume_ratio_proof_l780_78094

-- Definitions:
def height_ratio := 2 / 3
def volume_ratio (r : ℚ) := r^3
def small_pyramid_volume_ratio := volume_ratio height_ratio
def frustum_volume_ratio := 1 - small_pyramid_volume_ratio
def volume_ratio_small_to_frustum (v_small v_frustum : ℚ) := v_small / v_frustum

-- Lean 4 Statement:
theorem volume_ratio_proof
  (height_ratio : ℚ := 2 / 3)
  (small_pyramid_volume_ratio : ℚ := volume_ratio height_ratio)
  (frustum_volume_ratio : ℚ := 1 - small_pyramid_volume_ratio)
  (v_orig : ℚ) :
  volume_ratio_small_to_frustum (small_pyramid_volume_ratio * v_orig) (frustum_volume_ratio * v_orig) = 8 / 19 :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_proof_l780_78094


namespace NUMINAMATH_GPT_magician_starting_decks_l780_78046

def starting_decks (price_per_deck earned remaining_decks : ℕ) : ℕ :=
  earned / price_per_deck + remaining_decks

theorem magician_starting_decks :
  starting_decks 2 4 3 = 5 :=
by
  sorry

end NUMINAMATH_GPT_magician_starting_decks_l780_78046


namespace NUMINAMATH_GPT_train_speed_l780_78002

noncomputable def speed_of_train_kmph (L V : ℝ) : ℝ :=
  3.6 * V

theorem train_speed
  (L V : ℝ)
  (h1 : L = 18 * V)
  (h2 : L + 340 = 35 * V) :
  speed_of_train_kmph L V = 72 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l780_78002


namespace NUMINAMATH_GPT_parity_of_expression_l780_78011

theorem parity_of_expression
  (a b c : ℕ) 
  (h_a_odd : a % 2 = 1) 
  (h_b_odd : b % 2 = 1) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (h_c_pos : c > 0) :
  ((3^a + (b + 2)^2 * c) % 2 = 1 ↔ c % 2 = 0) ∧ 
  ((3^a + (b + 2)^2 * c) % 2 = 0 ↔ c % 2 = 1) :=
by sorry

end NUMINAMATH_GPT_parity_of_expression_l780_78011


namespace NUMINAMATH_GPT_max_mondays_in_51_days_l780_78092

theorem max_mondays_in_51_days : ∀ (first_day : ℕ), first_day ≤ 6 → (∃ mondays : ℕ, mondays = 8) :=
  by
  sorry

end NUMINAMATH_GPT_max_mondays_in_51_days_l780_78092


namespace NUMINAMATH_GPT_aniyah_more_candles_l780_78026

theorem aniyah_more_candles (x : ℝ) (h1 : 4 + 4 * x = 14) : x = 2.5 :=
sorry

end NUMINAMATH_GPT_aniyah_more_candles_l780_78026


namespace NUMINAMATH_GPT_find_speed_second_part_l780_78066

noncomputable def speed_second_part (x : ℝ) (v : ℝ) : Prop :=
  let t1 := x / 65       -- Time to cover the first x km at 65 kmph
  let t2 := 2 * x / v    -- Time to cover the second 2x km at v kmph
  let avg_time := 3 * x / 26    -- Average speed of the entire journey
  t1 + t2 = avg_time

theorem find_speed_second_part (x : ℝ) (v : ℝ) (h : speed_second_part x v) : v = 86.67 :=
sorry -- Proof of the claim

end NUMINAMATH_GPT_find_speed_second_part_l780_78066


namespace NUMINAMATH_GPT_paperboy_problem_l780_78004

noncomputable def delivery_ways (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 2
  else if n = 2 then 4
  else if n = 3 then 8
  else if n = 4 then 15
  else delivery_ways (n - 1) + delivery_ways (n - 2) + delivery_ways (n - 3) + delivery_ways (n - 4)

theorem paperboy_problem : delivery_ways 12 = 2872 :=
  sorry

end NUMINAMATH_GPT_paperboy_problem_l780_78004


namespace NUMINAMATH_GPT_geom_seq_sum_l780_78012

theorem geom_seq_sum {a : ℕ → ℝ} (q : ℝ) (h1 : a 0 + a 1 + a 2 = 2)
    (h2 : a 3 + a 4 + a 5 = 16)
    (h_geom : ∀ n, a (n + 1) = q * a n) :
  a 6 + a 7 + a 8 = 128 :=
sorry

end NUMINAMATH_GPT_geom_seq_sum_l780_78012


namespace NUMINAMATH_GPT_sin_sum_square_gt_sin_prod_l780_78084

theorem sin_sum_square_gt_sin_prod (α β γ : ℝ) (h1 : α + β + γ = Real.pi) 
  (h2 : 0 < Real.sin α) (h3 : Real.sin α < 1)
  (h4 : 0 < Real.sin β) (h5 : Real.sin β < 1)
  (h6 : 0 < Real.sin γ) (h7 : Real.sin γ < 1) :
  (Real.sin α + Real.sin β + Real.sin γ) ^ 2 > 9 * Real.sin α * Real.sin β * Real.sin γ := 
sorry

end NUMINAMATH_GPT_sin_sum_square_gt_sin_prod_l780_78084


namespace NUMINAMATH_GPT_mason_water_intake_l780_78042

theorem mason_water_intake
  (Theo_Daily : ℕ := 8)
  (Roxy_Daily : ℕ := 9)
  (Total_Weekly : ℕ := 168)
  (Days_Per_Week : ℕ := 7) :
  (∃ M : ℕ, M * Days_Per_Week = Total_Weekly - (Theo_Daily + Roxy_Daily) * Days_Per_Week ∧ M = 7) :=
  by
  sorry

end NUMINAMATH_GPT_mason_water_intake_l780_78042


namespace NUMINAMATH_GPT_number_of_partners_equation_l780_78014

variable (x : ℕ)

theorem number_of_partners_equation :
  5 * x + 45 = 7 * x - 3 :=
sorry

end NUMINAMATH_GPT_number_of_partners_equation_l780_78014


namespace NUMINAMATH_GPT_number_subtracted_l780_78052

theorem number_subtracted (x y : ℕ) (h₁ : x = 48) (h₂ : 5 * x - y = 102) : y = 138 :=
by
  rw [h₁] at h₂
  sorry

end NUMINAMATH_GPT_number_subtracted_l780_78052


namespace NUMINAMATH_GPT_total_trees_in_gray_areas_l780_78083

theorem total_trees_in_gray_areas (x y : ℕ) (h1 : 82 + x = 100) (h2 : 82 + y = 90) :
  x + y = 26 :=
by
  sorry

end NUMINAMATH_GPT_total_trees_in_gray_areas_l780_78083


namespace NUMINAMATH_GPT_black_pens_removed_l780_78023

theorem black_pens_removed (initial_blue : ℕ) (initial_black : ℕ) (initial_red : ℕ)
    (blue_removed : ℕ) (pens_left : ℕ)
    (h_initial_pens : initial_blue = 9 ∧ initial_black = 21 ∧ initial_red = 6)
    (h_blue_removed : blue_removed = 4)
    (h_pens_left : pens_left = 25) :
    initial_blue + initial_black + initial_red - blue_removed - (initial_blue + initial_black + initial_red - blue_removed - pens_left) = 7 :=
by
  rcases h_initial_pens with ⟨h_ib, h_ibl, h_ir⟩
  simp [h_ib, h_ibl, h_ir, h_blue_removed, h_pens_left]
  sorry

end NUMINAMATH_GPT_black_pens_removed_l780_78023


namespace NUMINAMATH_GPT_initial_balance_l780_78089

-- Define the conditions given in the problem
def transferred_percent_of_balance (X : ℝ) : ℝ := 0.15 * X
def balance_after_transfer (X : ℝ) : ℝ := 0.85 * X
def final_balance_after_refund (X : ℝ) (refund : ℝ) : ℝ := 0.85 * X + refund

-- Define the given values
def refund : ℝ := 450
def final_balance : ℝ := 30000

-- The theorem statement to prove the initial balance
theorem initial_balance (X : ℝ) (h : final_balance_after_refund X refund = final_balance) : 
  X = 34564.71 :=
by
  sorry

end NUMINAMATH_GPT_initial_balance_l780_78089


namespace NUMINAMATH_GPT_quotient_transformation_l780_78057

theorem quotient_transformation (A B : ℕ) (h1 : B ≠ 0) (h2 : (A : ℝ) / B = 0.514) :
  ((10 * A : ℝ) / (B / 100)) = 514 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_quotient_transformation_l780_78057


namespace NUMINAMATH_GPT_koschei_coins_l780_78029

theorem koschei_coins :
  ∃ a : ℕ, (a % 10 = 7) ∧ (a % 12 = 9) ∧ (300 ≤ a) ∧ (a ≤ 400) ∧ a = 357 :=
by 
  sorry

end NUMINAMATH_GPT_koschei_coins_l780_78029


namespace NUMINAMATH_GPT_obtain_2020_from_20_and_21_l780_78082

theorem obtain_2020_from_20_and_21 :
  ∃ (a b : ℕ), 20 * a + 21 * b = 2020 :=
by
  -- We only need to construct the proof goal, leaving the proof itself out.
  sorry

end NUMINAMATH_GPT_obtain_2020_from_20_and_21_l780_78082


namespace NUMINAMATH_GPT_complement_of_angle_is_acute_l780_78016

theorem complement_of_angle_is_acute (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < 90) : 0 < 90 - θ ∧ 90 - θ < 90 :=
by sorry

end NUMINAMATH_GPT_complement_of_angle_is_acute_l780_78016


namespace NUMINAMATH_GPT_min_sum_of_perpendicular_sides_l780_78034

noncomputable def min_sum_perpendicular_sides (a b : ℝ) (h : a * b = 100) : ℝ :=
a + b

theorem min_sum_of_perpendicular_sides {a b : ℝ} (h : a * b = 100) : min_sum_perpendicular_sides a b h = 20 :=
sorry

end NUMINAMATH_GPT_min_sum_of_perpendicular_sides_l780_78034


namespace NUMINAMATH_GPT_subsets_neither_A_nor_B_l780_78003

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {4, 5, 6, 7, 8}

theorem subsets_neither_A_nor_B : 
  (U.powerset.card - A.powerset.card - B.powerset.card + (A ∩ B).powerset.card) = 196 := by 
  sorry

end NUMINAMATH_GPT_subsets_neither_A_nor_B_l780_78003


namespace NUMINAMATH_GPT_problem_statement_l780_78085

theorem problem_statement (m : ℤ) (h : (m + 2)^2 = 64) : (m + 1) * (m + 3) = 63 :=
sorry

end NUMINAMATH_GPT_problem_statement_l780_78085


namespace NUMINAMATH_GPT_find_number_l780_78050

theorem find_number (x : ℝ) (h : 15 * x = 300) : x = 20 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l780_78050


namespace NUMINAMATH_GPT_canoes_more_than_kayaks_l780_78018

noncomputable def canoes_and_kayaks (C K : ℕ) : Prop :=
  (2 * C = 3 * K) ∧ (12 * C + 18 * K = 504) ∧ (C - K = 7)

theorem canoes_more_than_kayaks (C K : ℕ) (h : canoes_and_kayaks C K) : C - K = 7 :=
sorry

end NUMINAMATH_GPT_canoes_more_than_kayaks_l780_78018


namespace NUMINAMATH_GPT_no_solution_exists_only_solution_is_1963_l780_78099

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else n % 10 + sum_of_digits (n / 10)

-- Proof problem for part (a)
theorem no_solution_exists :
  ¬ ∃ x : ℕ, x + sum_of_digits x + sum_of_digits (sum_of_digits x) = 1993 :=
sorry

-- Proof problem for part (b)
theorem only_solution_is_1963 :
  ∃ x : ℕ, (x + sum_of_digits x + sum_of_digits (sum_of_digits x) + sum_of_digits (sum_of_digits (sum_of_digits x)) = 1993) ∧ (x = 1963) :=
sorry

end NUMINAMATH_GPT_no_solution_exists_only_solution_is_1963_l780_78099


namespace NUMINAMATH_GPT_car_speeds_and_arrival_times_l780_78065

theorem car_speeds_and_arrival_times
  (x y z u : ℝ)
  (h1 : x^2 = (y + z) * u)
  (h2 : (y + z) / 4 = u)
  (h3 : x / u = y / z)
  (h4 : x + y + z + u = 210) :
  x = 60 ∧ y = 80 ∧ z = 40 ∧ u = 30 := 
by
  sorry

end NUMINAMATH_GPT_car_speeds_and_arrival_times_l780_78065


namespace NUMINAMATH_GPT_total_strength_of_college_l780_78088

-- Declare the variables for number of students playing each sport
variables (C B Both : ℕ)

-- Given conditions in the problem
def cricket_players : ℕ := 500
def basketball_players : ℕ := 600
def both_players : ℕ := 220

-- Theorem stating the total strength of the college
theorem total_strength_of_college (h_C : C = cricket_players) 
                                  (h_B : B = basketball_players) 
                                  (h_Both : Both = both_players) : 
                                  C + B - Both = 880 :=
by
  sorry

end NUMINAMATH_GPT_total_strength_of_college_l780_78088


namespace NUMINAMATH_GPT_function_solution_l780_78032

theorem function_solution (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = sorry) → f a = sorry → (a = 1 ∨ a = -1) :=
by
  intros hfa hfb
  sorry

end NUMINAMATH_GPT_function_solution_l780_78032


namespace NUMINAMATH_GPT_plane_distance_l780_78013

theorem plane_distance (D : ℕ) (h₁ : D / 300 + D / 400 = 7) : D = 1200 :=
sorry

end NUMINAMATH_GPT_plane_distance_l780_78013


namespace NUMINAMATH_GPT_martha_initial_juice_pantry_l780_78021

theorem martha_initial_juice_pantry (P : ℕ) : 
  4 + P + 5 - 3 = 10 → P = 4 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_martha_initial_juice_pantry_l780_78021


namespace NUMINAMATH_GPT_determine_b_l780_78067

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x < 1 then 3 * x - b else 2 ^ x

theorem determine_b (b : ℝ) :
  f (f (5 / 6) b) b = 4 ↔ b = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_determine_b_l780_78067


namespace NUMINAMATH_GPT_union_of_M_and_N_l780_78071

def M : Set ℕ := {1, 2, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_union_of_M_and_N_l780_78071


namespace NUMINAMATH_GPT_line_intersects_ellipse_all_possible_slopes_l780_78037

theorem line_intersects_ellipse_all_possible_slopes (m : ℝ) :
  m^2 ≥ 1 / 5 ↔ ∃ x y : ℝ, (y = m * x - 3) ∧ (4 * x^2 + 25 * y^2 = 100) := sorry

end NUMINAMATH_GPT_line_intersects_ellipse_all_possible_slopes_l780_78037


namespace NUMINAMATH_GPT_fencing_required_l780_78049

variable (L W : ℝ)
variable (Area : ℝ := 20 * W)

theorem fencing_required (hL : L = 20) (hArea : L * W = 600) : 20 + 2 * W = 80 := by
  sorry

end NUMINAMATH_GPT_fencing_required_l780_78049


namespace NUMINAMATH_GPT_triangle_right_angle_l780_78073

theorem triangle_right_angle {a b c : ℝ} {A B C : ℝ} (h : a * Real.cos A + b * Real.cos B = c * Real.cos C) :
  (A = Real.pi / 2) ∨ (B = Real.pi / 2) ∨ (C = Real.pi / 2) :=
sorry

end NUMINAMATH_GPT_triangle_right_angle_l780_78073


namespace NUMINAMATH_GPT_average_age_of_persons_l780_78036

theorem average_age_of_persons 
  (total_age : ℕ := 270) 
  (average_age : ℕ := 15) : 
  (total_age / average_age) = 18 := 
by { 
  sorry 
}

end NUMINAMATH_GPT_average_age_of_persons_l780_78036


namespace NUMINAMATH_GPT_hyperbola_eccentricity_proof_l780_78047

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (b ^ 2 + (a / 2) ^ 2 = a ^ 2)

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt ((a ^ 2 + b ^ 2) / a ^ 2)

theorem hyperbola_eccentricity_proof
  (a b : ℝ) (h : a > b ∧ b > 0) (h1 : ellipse_eccentricity a b h) :
  hyperbola_eccentricity a b = Real.sqrt 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_proof_l780_78047
