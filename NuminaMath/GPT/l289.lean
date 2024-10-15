import Mathlib

namespace NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l289_28909

theorem sum_of_reciprocals_of_roots :
  ∀ (c d : ℝ),
  (6 * c^2 + 5 * c + 7 = 0) → 
  (6 * d^2 + 5 * d + 7 = 0) → 
  (c + d = -5 / 6) → 
  (c * d = 7 / 6) → 
  (1 / c + 1 / d = -5 / 7) :=
by
  intros c d h₁ h₂ h₃ h₄
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l289_28909


namespace NUMINAMATH_GPT_relationship_abc_l289_28951

noncomputable def a : ℝ := (0.7 : ℝ) ^ (0.6 : ℝ)
noncomputable def b : ℝ := (0.6 : ℝ) ^ (-0.6 : ℝ)
noncomputable def c : ℝ := (0.6 : ℝ) ^ (0.7 : ℝ)

theorem relationship_abc : b > a ∧ a > c :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_relationship_abc_l289_28951


namespace NUMINAMATH_GPT_number_of_individuals_left_at_zoo_l289_28977

theorem number_of_individuals_left_at_zoo 
  (students_class1 students_class2 students_left : ℕ)
  (initial_chaperones remaining_chaperones teachers : ℕ) :
  students_class1 = 10 ∧
  students_class2 = 10 ∧
  initial_chaperones = 5 ∧
  teachers = 2 ∧
  students_left = 10 ∧
  remaining_chaperones = initial_chaperones - 2 →
  (students_class1 + students_class2 - students_left) + remaining_chaperones + teachers = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_individuals_left_at_zoo_l289_28977


namespace NUMINAMATH_GPT_real_solutions_system_l289_28914

theorem real_solutions_system (x y z : ℝ) : 
  (x = 4 * z^2 / (1 + 4 * z^2) ∧ y = 4 * x^2 / (1 + 4 * x^2) ∧ z = 4 * y^2 / (1 + 4 * y^2)) ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_system_l289_28914


namespace NUMINAMATH_GPT_remainder_of_19_pow_60_mod_7_l289_28916

theorem remainder_of_19_pow_60_mod_7 : (19 ^ 60) % 7 = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_of_19_pow_60_mod_7_l289_28916


namespace NUMINAMATH_GPT_gcd_f_of_x_and_x_l289_28934

theorem gcd_f_of_x_and_x (x : ℕ) (hx : 7200 ∣ x) :
  Nat.gcd ((5 * x + 6) * (8 * x + 3) * (11 * x + 9) * (4 * x + 12)) x = 72 :=
sorry

end NUMINAMATH_GPT_gcd_f_of_x_and_x_l289_28934


namespace NUMINAMATH_GPT_intersection_eq_l289_28931

def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 0 }
def N : Set ℝ := { -1, 0, 1 }

theorem intersection_eq : M ∩ N = { -1, 0 } := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l289_28931


namespace NUMINAMATH_GPT_solve_for_b_l289_28987

theorem solve_for_b (b : ℝ) : 
  let slope1 := -(3 / 4 : ℝ)
  let slope2 := -(b / 6 : ℝ)
  slope1 * slope2 = -1 → b = -8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_b_l289_28987


namespace NUMINAMATH_GPT_Kyle_throws_farther_l289_28910

theorem Kyle_throws_farther (Parker_distance : ℕ) (Grant_ratio : ℚ) (Kyle_ratio : ℚ) (Grant_distance : ℚ) (Kyle_distance : ℚ) :
  Parker_distance = 16 → 
  Grant_ratio = 0.25 → 
  Kyle_ratio = 2 → 
  Grant_distance = Parker_distance + Parker_distance * Grant_ratio → 
  Kyle_distance = Kyle_ratio * Grant_distance → 
  Kyle_distance - Parker_distance = 24 :=
by
  intros hp hg hk hg_dist hk_dist
  subst hp
  subst hg
  subst hk
  subst hg_dist
  subst hk_dist
  -- The proof steps are omitted
  sorry

end NUMINAMATH_GPT_Kyle_throws_farther_l289_28910


namespace NUMINAMATH_GPT_prime_ge_7_div_30_l289_28937

theorem prime_ge_7_div_30 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
sorry

end NUMINAMATH_GPT_prime_ge_7_div_30_l289_28937


namespace NUMINAMATH_GPT_solve_for_x_l289_28905

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l289_28905


namespace NUMINAMATH_GPT_find_a_l289_28921

theorem find_a (a b : ℝ) (h1 : 0 < a ∧ 0 < b) (h2 : a^b = b^a) (h3 : b = 4 * a) : 
  a = (4 : ℝ)^(1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l289_28921


namespace NUMINAMATH_GPT_min_value_of_f_l289_28974

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem min_value_of_f : ∀ (x : ℝ), x > 2 → f x ≥ 4 := by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l289_28974


namespace NUMINAMATH_GPT_correct_proposition_l289_28929

-- Define the propositions as Lean 4 statements.
def PropA (a : ℝ) : Prop := a^4 + a^2 = a^6
def PropB (a : ℝ) : Prop := (-2 * a^2)^3 = -6 * a^8
def PropC (a : ℝ) : Prop := 6 * a - a = 5
def PropD (a : ℝ) : Prop := a^2 * a^3 = a^5

-- The main theorem statement that only PropD is true.
theorem correct_proposition (a : ℝ) : ¬ PropA a ∧ ¬ PropB a ∧ ¬ PropC a ∧ PropD a :=
by
  sorry

end NUMINAMATH_GPT_correct_proposition_l289_28929


namespace NUMINAMATH_GPT_find_num_adults_l289_28998

-- Define the conditions
def total_eggs : ℕ := 36
def eggs_per_adult : ℕ := 3
def eggs_per_girl : ℕ := 1
def eggs_per_boy := eggs_per_girl + 1
def num_girls : ℕ := 7
def num_boys : ℕ := 10

-- Compute total eggs given to girls
def eggs_given_to_girls : ℕ := num_girls * eggs_per_girl

-- Compute total eggs given to boys
def eggs_given_to_boys : ℕ := num_boys * eggs_per_boy

-- Compute total eggs given to children
def eggs_given_to_children : ℕ := eggs_given_to_girls + eggs_given_to_boys

-- Total number of eggs given to children
def eggs_left_for_adults : ℕ := total_eggs - eggs_given_to_children

-- Calculate the number of adults
def num_adults : ℕ := eggs_left_for_adults / eggs_per_adult

-- Finally, we want to prove that the number of adults is 3
theorem find_num_adults (h1 : total_eggs = 36) 
                        (h2 : eggs_per_adult = 3) 
                        (h3 : eggs_per_girl = 1)
                        (h4 : num_girls = 7) 
                        (h5 : num_boys = 10) : 
                        num_adults = 3 := by
  -- Using the given conditions and computations
  sorry

end NUMINAMATH_GPT_find_num_adults_l289_28998


namespace NUMINAMATH_GPT_B_take_time_4_hours_l289_28979

theorem B_take_time_4_hours (A_rate B_rate C_rate D_rate : ℚ) :
  (A_rate = 1 / 4) →
  (B_rate + C_rate = 1 / 2) →
  (A_rate + C_rate = 1 / 2) →
  (D_rate = 1 / 8) →
  (A_rate + B_rate + D_rate = 1 / 1.6) →
  (B_rate = 1 / 4) ∧ (1 / B_rate = 4) :=
by
  sorry

end NUMINAMATH_GPT_B_take_time_4_hours_l289_28979


namespace NUMINAMATH_GPT_problem1_problem2_l289_28993

-- Equivalent proof statement for part (1)
theorem problem1 : 2023^2 - 2022 * 2024 = 1 := by
  sorry

-- Equivalent proof statement for part (2)
theorem problem2 (m : ℝ) (h : m ≠ 1) (h1 : m ≠ -1) : 
  (m / (m^2 - 1)) / ((m^2 - m) / (m^2 - 2*m + 1)) = 1 / (m + 1) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l289_28993


namespace NUMINAMATH_GPT_megatek_manufacturing_percentage_l289_28962

theorem megatek_manufacturing_percentage (total_degrees manufacturing_degrees : ℝ)
    (h_proportional : total_degrees = 360)
    (h_manufacturing_degrees : manufacturing_degrees = 180) :
    (manufacturing_degrees / total_degrees) * 100 = 50 := by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_megatek_manufacturing_percentage_l289_28962


namespace NUMINAMATH_GPT_solve_quadratic_identity_l289_28982

theorem solve_quadratic_identity (y : ℝ) (h : 7 * y^2 + 2 = 5 * y + 13) :
  (14 * y - 5) ^ 2 = 333 :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_identity_l289_28982


namespace NUMINAMATH_GPT_empty_cistern_time_l289_28952

variable (t_fill : ℝ) (t_empty₁ : ℝ) (t_empty₂ : ℝ) (t_empty₃ : ℝ)

theorem empty_cistern_time
  (h_fill : t_fill = 3.5)
  (h_empty₁ : t_empty₁ = 14)
  (h_empty₂ : t_empty₂ = 16)
  (h_empty₃ : t_empty₃ = 18) :
  1008 / (1/t_empty₁ + 1/t_empty₂ + 1/t_empty₃) = 1.31979 := by
  sorry

end NUMINAMATH_GPT_empty_cistern_time_l289_28952


namespace NUMINAMATH_GPT_Mahesh_completes_in_60_days_l289_28950

noncomputable def MaheshWork (W : ℝ) : ℝ :=
    W / 60

variables (W : ℝ)
variables (M R : ℝ)
variables (daysMahesh daysRajesh daysFullRajesh : ℝ)

theorem Mahesh_completes_in_60_days
  (h1 : daysMahesh = 20)
  (h2 : daysRajesh = 30)
  (h3 : daysFullRajesh = 45)
  (hR : R = W / daysFullRajesh)
  (hM : M = (W - R * daysRajesh) / daysMahesh) :
  W / M = 60 :=
by
  sorry

end NUMINAMATH_GPT_Mahesh_completes_in_60_days_l289_28950


namespace NUMINAMATH_GPT_sqrt_subtraction_l289_28960

theorem sqrt_subtraction : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - Real.sqrt 27 := by
  sorry

end NUMINAMATH_GPT_sqrt_subtraction_l289_28960


namespace NUMINAMATH_GPT_apples_rate_per_kg_l289_28925

variable (A : ℝ)

theorem apples_rate_per_kg (h : 8 * A + 9 * 65 = 1145) : A = 70 :=
sorry

end NUMINAMATH_GPT_apples_rate_per_kg_l289_28925


namespace NUMINAMATH_GPT_g_triple_composition_l289_28935

def g (n : ℕ) : ℕ :=
if n < 5 then n^2 + 1 else 2 * n + 3

theorem g_triple_composition : g (g (g 3)) = 49 :=
by
  sorry

end NUMINAMATH_GPT_g_triple_composition_l289_28935


namespace NUMINAMATH_GPT_average_after_12th_inning_revised_average_not_out_l289_28956

theorem average_after_12th_inning (A : ℝ) (H_innings : 11 * A + 92 = 12 * (A + 2)) : (A + 2) = 70 :=
by
  -- Calculation steps are skipped
  sorry

theorem revised_average_not_out (A : ℝ) (H_innings : 11 * A + 92 = 12 * (A + 2)) (H_not_out : 11 * A + 92 = 840) :
  (11 * A + 92) / 9 = 93.33 :=
by
  -- Calculation steps are skipped
  sorry

end NUMINAMATH_GPT_average_after_12th_inning_revised_average_not_out_l289_28956


namespace NUMINAMATH_GPT_diagonal_length_l289_28988

noncomputable def length_of_diagonal (a b c : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2)

theorem diagonal_length
  (a b c : ℝ)
  (h1 : 2 * (a * b + a * c + b * c) = 11)
  (h2 : 4 * (a + b + c) = 24) :
  length_of_diagonal a b c = 5 := by
  sorry

end NUMINAMATH_GPT_diagonal_length_l289_28988


namespace NUMINAMATH_GPT_find_f_of_1_div_8_l289_28939

noncomputable def f (x : ℝ) (a : ℝ) := (a^2 + a - 5) * Real.logb a x

theorem find_f_of_1_div_8 (a : ℝ) (hx1 : x = 1 / 8) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^2 + a - 5 = 1) :
  f x a = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_1_div_8_l289_28939


namespace NUMINAMATH_GPT_odd_divisibility_l289_28990

theorem odd_divisibility (n : ℕ) (k : ℕ) (x y : ℤ) (h : n = 2 * k + 1) : (x^n + y^n) % (x + y) = 0 :=
by sorry

end NUMINAMATH_GPT_odd_divisibility_l289_28990


namespace NUMINAMATH_GPT_farmer_harvest_correct_l289_28902

def estimated_harvest : ℕ := 48097
def additional_harvest : ℕ := 684
def total_harvest : ℕ := 48781

theorem farmer_harvest_correct : estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end NUMINAMATH_GPT_farmer_harvest_correct_l289_28902


namespace NUMINAMATH_GPT_no_integer_pairs_satisfy_equation_l289_28928

theorem no_integer_pairs_satisfy_equation :
  ∀ (m n : ℤ), ¬(m^3 + 10 * m^2 + 11 * m + 2 = 81 * n^3 + 27 * n^2 + 3 * n - 8) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_pairs_satisfy_equation_l289_28928


namespace NUMINAMATH_GPT_abs_e_pi_minus_six_l289_28932

noncomputable def e : ℝ := 2.718
noncomputable def pi : ℝ := 3.14159

theorem abs_e_pi_minus_six : |e + pi - 6| = 0.14041 := by
  sorry

end NUMINAMATH_GPT_abs_e_pi_minus_six_l289_28932


namespace NUMINAMATH_GPT_number_of_valid_paths_l289_28985

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_valid_paths (n : ℕ) :
  let valid_paths := binomial (2 * n) n / (n + 1)
  valid_paths = binomial (2 * n) n - binomial (2 * n) (n + 1) := 
sorry

end NUMINAMATH_GPT_number_of_valid_paths_l289_28985


namespace NUMINAMATH_GPT_sum_of_A_and_B_l289_28986

theorem sum_of_A_and_B (A B : ℕ) (h1 : (1 / 6 : ℚ) * (1 / 3) = 1 / (A * 3))
                       (h2 : (1 / 6 : ℚ) * (1 / 3) = 1 / B) : A + B = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_A_and_B_l289_28986


namespace NUMINAMATH_GPT_fraction_pizza_covered_by_pepperoni_l289_28945

/--
Given that six pepperoni circles fit exactly across the diameter of a 12-inch pizza
and a total of 24 circles of pepperoni are placed on the pizza without overlap,
prove that the fraction of the pizza covered by pepperoni is 2/3.
-/
theorem fraction_pizza_covered_by_pepperoni : 
  (∃ d r : ℝ, 6 * r = d ∧ d = 12 ∧ (r * r * π * 24) / (6 * 6 * π) = 2 / 3) := 
sorry

end NUMINAMATH_GPT_fraction_pizza_covered_by_pepperoni_l289_28945


namespace NUMINAMATH_GPT_price_difference_is_99_cents_l289_28908

-- Definitions for the conditions
def list_price : ℚ := 3996 / 100
def discount_super_savers : ℚ := 9
def discount_penny_wise : ℚ := 25 / 100 * list_price

-- Sale prices calculated based on the given conditions
def sale_price_super_savers : ℚ := list_price - discount_super_savers
def sale_price_penny_wise : ℚ := list_price - discount_penny_wise

-- Difference in prices
def price_difference : ℚ := sale_price_super_savers - sale_price_penny_wise

-- Prove that the price difference in cents is 99
theorem price_difference_is_99_cents : price_difference = 99 / 100 := 
by
  sorry

end NUMINAMATH_GPT_price_difference_is_99_cents_l289_28908


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l289_28967

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : 6 * a 7 = (a 8 + a 9) / 2)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h3 : ∀ n : ℕ, S n = a 1 * (1 - q^n) / (1 - q)) :
  S 6 / S 3 = 28 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l289_28967


namespace NUMINAMATH_GPT_no_perfect_squares_xy_zt_l289_28918

theorem no_perfect_squares_xy_zt
    (x y z t : ℕ) 
    (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < t)
    (h_eq1 : x + y = z + t) 
    (h_eq2 : xy - zt = x + y) : ¬(∃ a b : ℕ, xy = a^2 ∧ zt = b^2) :=
by
  sorry

end NUMINAMATH_GPT_no_perfect_squares_xy_zt_l289_28918


namespace NUMINAMATH_GPT_sequence_properties_l289_28923

theorem sequence_properties (a : ℕ → ℝ)
  (h1 : a 1 = 1 / 5)
  (h2 : ∀ n : ℕ, n > 1 → a (n - 1) / a n = (2 * a (n - 1) + 1) / (1 - 2 * a n)) :
  (∀ n : ℕ, n > 0 → (1 / a n) - (1 / a (n - 1)) = 4) ∧
  (∀ m k : ℕ, m > 0 ∧ k > 0 → a m * a k = a (4 * m * k + m + k)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_properties_l289_28923


namespace NUMINAMATH_GPT_gray_region_correct_b_l289_28965

-- Define the basic conditions
def square_side_length : ℝ := 3
def small_square_side_length : ℝ := 1

-- Define the triangles resulting from cutting a square
def triangle_area : ℝ := 0.5 * square_side_length * square_side_length

-- Define the gray region area for the second figure (b)
def gray_region_area_b : ℝ := 0.25

-- Lean statement to prove the area of the gray region
theorem gray_region_correct_b : gray_region_area_b = 0.25 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_gray_region_correct_b_l289_28965


namespace NUMINAMATH_GPT_evaluate_f_diff_l289_28927

def f (x : ℝ) := x^5 + 2*x^3 + 7*x

theorem evaluate_f_diff : f 3 - f (-3) = 636 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_diff_l289_28927


namespace NUMINAMATH_GPT_tetrahedron_faces_equal_l289_28973

theorem tetrahedron_faces_equal {a b c a' b' c' : ℝ} (h₁ : a + b + c = a + b' + c') (h₂ : a + b + c = a' + b + b') (h₃ : a + b + c = c' + c + a') :
  (a = a') ∧ (b = b') ∧ (c = c') :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_faces_equal_l289_28973


namespace NUMINAMATH_GPT_determine_list_price_l289_28976

theorem determine_list_price (x : ℝ) :
  0.12 * (x - 15) = 0.15 * (x - 25) → x = 65 :=
by 
  sorry

end NUMINAMATH_GPT_determine_list_price_l289_28976


namespace NUMINAMATH_GPT_no_real_x_condition_l289_28989

theorem no_real_x_condition (a : ℝ) :
  (¬ ∃ x : ℝ, |x - 3| + |x - 1| ≤ a) ↔ a < 2 := 
by
  sorry

end NUMINAMATH_GPT_no_real_x_condition_l289_28989


namespace NUMINAMATH_GPT_sum_of_products_l289_28912

def is_positive (x : ℝ) := 0 < x

theorem sum_of_products 
  (x y z : ℝ) 
  (hx : is_positive x)
  (hy : is_positive y)
  (hz : is_positive z)
  (h1 : x^2 + x * y + y^2 = 27)
  (h2 : y^2 + y * z + z^2 = 25)
  (h3 : z^2 + z * x + x^2 = 52) :
  x * y + y * z + z * x = 30 :=
  sorry

end NUMINAMATH_GPT_sum_of_products_l289_28912


namespace NUMINAMATH_GPT_integer_coordinates_midpoint_exists_l289_28980

theorem integer_coordinates_midpoint_exists (P : Fin 5 → ℤ × ℤ) :
  ∃ i j : Fin 5, i ≠ j ∧
    ∃ x y : ℤ, (2 * x = (P i).1 + (P j).1) ∧ (2 * y = (P i).2 + (P j).2) := sorry

end NUMINAMATH_GPT_integer_coordinates_midpoint_exists_l289_28980


namespace NUMINAMATH_GPT_sum_of_solutions_l289_28907

-- Define the quadratic equation as a product of linear factors
def quadratic_eq (x : ℚ) : Prop := (4 * x + 6) * (3 * x - 8) = 0

-- Define the roots of the quadratic equation
def root1 : ℚ := -3 / 2
def root2 : ℚ := 8 / 3

-- Sum of the roots of the quadratic equation
def sum_of_roots : ℚ := root1 + root2

-- Theorem stating that the sum of the roots is 7/6
theorem sum_of_solutions : sum_of_roots = 7 / 6 := by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l289_28907


namespace NUMINAMATH_GPT_total_cats_in_training_center_l289_28901

-- Definitions corresponding to the given conditions
def cats_can_jump : ℕ := 60
def cats_can_fetch : ℕ := 35
def cats_can_meow : ℕ := 40
def cats_jump_fetch : ℕ := 20
def cats_fetch_meow : ℕ := 15
def cats_jump_meow : ℕ := 25
def cats_all_three : ℕ := 11
def cats_none : ℕ := 10

-- Theorem statement corresponding to proving question == answer given conditions
theorem total_cats_in_training_center
    (cjump : ℕ := cats_can_jump)
    (cfetch : ℕ := cats_can_fetch)
    (cmeow : ℕ := cats_can_meow)
    (cjf : ℕ := cats_jump_fetch)
    (cfm : ℕ := cats_fetch_meow)
    (cjm : ℕ := cats_jump_meow)
    (cat : ℕ := cats_all_three)
    (cno : ℕ := cats_none) :
    cjump
    + cfetch
    + cmeow
    - cjf
    - cfm
    - cjm
    + cat
    + cno
    = 96 := sorry

end NUMINAMATH_GPT_total_cats_in_training_center_l289_28901


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l289_28900

def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x < 1} := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l289_28900


namespace NUMINAMATH_GPT_terminal_side_of_610_deg_is_250_deg_l289_28941

theorem terminal_side_of_610_deg_is_250_deg:
  ∃ k : ℤ, 610 % 360 = 250 := by
  sorry

end NUMINAMATH_GPT_terminal_side_of_610_deg_is_250_deg_l289_28941


namespace NUMINAMATH_GPT_intersection_l289_28913

def setA : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def setB : Set ℝ := { x | x^2 - 2 * x ≥ 0 }

theorem intersection: setA ∩ setB = { x : ℝ | x ≤ 0 } := by
  sorry

end NUMINAMATH_GPT_intersection_l289_28913


namespace NUMINAMATH_GPT_route_length_l289_28922

theorem route_length (D : ℝ) (T : ℝ) 
  (hx : T = 400 / D) 
  (hy : 80 = (D / 5) * T) 
  (hz : 80 + (D / 4) * T = D) : 
  D = 180 :=
by
  sorry

end NUMINAMATH_GPT_route_length_l289_28922


namespace NUMINAMATH_GPT_right_triangle_area_l289_28969

theorem right_triangle_area (a b c: ℝ) (h1: c = 2) (h2: a + b + c = 2 + Real.sqrt 6) (h3: (a * b) / 2 = 1 / 2) :
  (1 / 2) * (a * b) = 1 / 2 :=
by
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_right_triangle_area_l289_28969


namespace NUMINAMATH_GPT_greatest_whole_number_satisfying_inequality_l289_28957

theorem greatest_whole_number_satisfying_inequality :
  ∀ (x : ℤ), 3 * x + 2 < 5 - 2 * x → x <= 0 :=
by
  sorry

end NUMINAMATH_GPT_greatest_whole_number_satisfying_inequality_l289_28957


namespace NUMINAMATH_GPT_algebraic_expression_value_l289_28904

theorem algebraic_expression_value (x y : ℝ) (h : 2 * x - y = 2) : 6 * x - 3 * y + 1 = 7 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l289_28904


namespace NUMINAMATH_GPT_largest_integer_n_exists_l289_28903

theorem largest_integer_n_exists :
  ∃ (x y z n : ℤ), (x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 5 * x + 5 * y + 5 * z - 10 = n^2) ∧ (n = 6) :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_n_exists_l289_28903


namespace NUMINAMATH_GPT_certain_number_division_l289_28933

theorem certain_number_division (x : ℝ) (h : x / 3 + x + 3 = 63) : x = 45 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_division_l289_28933


namespace NUMINAMATH_GPT_inversely_proportional_l289_28984

theorem inversely_proportional (X Y K : ℝ) (h : X * Y = K - 1) (hK : K > 1) : 
  (∃ c : ℝ, ∀ x y : ℝ, x * y = c) :=
sorry

end NUMINAMATH_GPT_inversely_proportional_l289_28984


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_l289_28936

-- Definitions based on conditions
def equilateral_triangle_side : ℕ := 8

-- The statement we need to prove
theorem equilateral_triangle_perimeter : 3 * equilateral_triangle_side = 24 := by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_perimeter_l289_28936


namespace NUMINAMATH_GPT_right_triangle_k_value_l289_28911

theorem right_triangle_k_value (x : ℝ) (k : ℝ) (s : ℝ) 
(h_triangle : 3*x + 4*x + 5*x = k * (1/2 * 3*x * 4*x)) 
(h_square : s = 10) (h_eq_apothems : 4*x = s/2) : 
k = 8 / 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_right_triangle_k_value_l289_28911


namespace NUMINAMATH_GPT_probability_of_three_specific_suits_l289_28970

noncomputable def probability_at_least_one_from_each_of_three_suits : ℚ :=
  1 - (1 / 4) ^ 5

theorem probability_of_three_specific_suits (hearts clubs diamonds : ℕ) :
  hearts = 0 ∧ clubs = 0 ∧ diamonds = 0 → 
  probability_at_least_one_from_each_of_three_suits = 1023 / 1024 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_three_specific_suits_l289_28970


namespace NUMINAMATH_GPT_sara_lunch_total_cost_l289_28966

noncomputable def cost_hotdog : ℝ := 5.36
noncomputable def cost_salad : ℝ := 5.10
noncomputable def cost_soda : ℝ := 2.75
noncomputable def cost_fries : ℝ := 3.20
noncomputable def discount_rate : ℝ := 0.15
noncomputable def tax_rate : ℝ := 0.08

noncomputable def total_cost_before_discount_tax : ℝ :=
  cost_hotdog + cost_salad + cost_soda + cost_fries

noncomputable def discount : ℝ :=
  discount_rate * total_cost_before_discount_tax

noncomputable def discounted_total : ℝ :=
  total_cost_before_discount_tax - discount

noncomputable def tax : ℝ := 
  tax_rate * discounted_total

noncomputable def final_total : ℝ :=
  discounted_total + tax

theorem sara_lunch_total_cost : final_total = 15.07 :=
by
  sorry

end NUMINAMATH_GPT_sara_lunch_total_cost_l289_28966


namespace NUMINAMATH_GPT_math_problem_l289_28919

theorem math_problem (a b n r : ℕ) (h₁ : 1853 ≡ 53 [MOD 600]) (h₂ : 2101 ≡ 101 [MOD 600]) :
  (1853 * 2101) ≡ 553 [MOD 600] := by
  sorry

end NUMINAMATH_GPT_math_problem_l289_28919


namespace NUMINAMATH_GPT_probability_of_black_ball_l289_28991

theorem probability_of_black_ball (P_red P_white : ℝ) (h_red : P_red = 0.43) (h_white : P_white = 0.27) : 
  (1 - P_red - P_white) = 0.3 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_black_ball_l289_28991


namespace NUMINAMATH_GPT_intersection_of_squares_perimeter_l289_28959

noncomputable def perimeter_of_rectangle (side1 side2 : ℝ) : ℝ :=
2 * (side1 + side2)

theorem intersection_of_squares_perimeter
  (side_length : ℝ)
  (diagonal : ℝ)
  (distance_between_centers : ℝ)
  (h1 : 4 * side_length = 8) 
  (h2 : (side1^2 + side2^2) = diagonal^2)
  (h3 : (2 - side1)^2 + (2 - side2)^2 = distance_between_centers^2) : 
10 * (perimeter_of_rectangle side1 side2) = 25 :=
sorry

end NUMINAMATH_GPT_intersection_of_squares_perimeter_l289_28959


namespace NUMINAMATH_GPT_friend_decks_l289_28964

-- Definitions for conditions
def price_per_deck : ℕ := 8
def victor_decks : ℕ := 6
def total_spent : ℕ := 64

-- Conclusion based on the conditions
theorem friend_decks : (64 - (6 * 8)) / 8 = 2 := by
  sorry

end NUMINAMATH_GPT_friend_decks_l289_28964


namespace NUMINAMATH_GPT_value_of_a_l289_28996

noncomputable def f (a x : ℝ) : ℝ := a ^ x

theorem value_of_a (a : ℝ) (h : abs ((a^2) - a) = a / 2) : a = 1 / 2 ∨ a = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l289_28996


namespace NUMINAMATH_GPT_parallel_vectors_x_value_l289_28981

noncomputable def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem parallel_vectors_x_value :
  vectors_parallel (1, -2) (x, 1) → x = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_x_value_l289_28981


namespace NUMINAMATH_GPT_proof_problem_l289_28995

noncomputable def f (a b : ℝ) (x : ℝ) := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem proof_problem (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∀ x : ℝ, f a b x ≤ |f a b (π / 6)|) : 
  (f a b (11 * π / 12) = 0) ∧
  (|f a b (7 * π / 12)| < |f a b (π / 5)|) ∧
  (¬ (∀ x : ℝ, f a b x = f a b (-x)) ∧ ¬ (∀ x : ℝ, f a b x = -f a b (-x))) := 
sorry

end NUMINAMATH_GPT_proof_problem_l289_28995


namespace NUMINAMATH_GPT_smallest_pos_int_b_for_factorization_l289_28994

theorem smallest_pos_int_b_for_factorization :
  ∃ b : ℤ, 0 < b ∧ ∀ (x : ℤ), ∃ r s : ℤ, r * s = 4032 ∧ r + s = b ∧ x^2 + b * x + 4032 = (x + r) * (x + s) ∧
    (∀ b' : ℤ, 0 < b' → b' ≠ b → ∃ rr ss : ℤ, rr * ss = 4032 ∧ rr + ss = b' ∧ x^2 + b' * x + 4032 = (x + rr) * (x + ss) → b < b') := 
sorry

end NUMINAMATH_GPT_smallest_pos_int_b_for_factorization_l289_28994


namespace NUMINAMATH_GPT_common_difference_arithmetic_sequence_l289_28924

theorem common_difference_arithmetic_sequence 
    (a : ℕ → ℝ) 
    (S₅ : ℝ)
    (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h₁ : a 4 + a 6 = 6)
    (h₂ : S₅ = (a 1 + a 2 + a 3 + a 4 + a 5))
    (h_S₅_val : S₅ = 10) :
  ∃ d : ℝ, d = (a 5 - a 1) / 4 ∧ d = 1/2 := 
by
  sorry

end NUMINAMATH_GPT_common_difference_arithmetic_sequence_l289_28924


namespace NUMINAMATH_GPT_evaluate_expression_l289_28926

theorem evaluate_expression : 1 - (-2) * 2 - 3 - (-4) * 2 - 5 - (-6) * 2 = 17 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l289_28926


namespace NUMINAMATH_GPT_spherical_to_rectangular_correct_l289_28961

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  sphericalToRectangular ρ θ φ = (5 * Real.sqrt 6 / 4, 5 * Real.sqrt 6 / 4, 5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_correct_l289_28961


namespace NUMINAMATH_GPT_mrs_choi_profit_percentage_l289_28968

theorem mrs_choi_profit_percentage :
  ∀ (original_price selling_price : ℝ) (broker_percentage : ℝ),
    original_price = 80000 →
    selling_price = 100000 →
    broker_percentage = 0.05 →
    (selling_price - (broker_percentage * original_price) - original_price) / original_price * 100 = 20 :=
by
  intros original_price selling_price broker_percentage h1 h2 h3
  sorry

end NUMINAMATH_GPT_mrs_choi_profit_percentage_l289_28968


namespace NUMINAMATH_GPT_parabola_vertex_l289_28917

theorem parabola_vertex (c d : ℝ) (h : ∀ (x : ℝ), (-x^2 + c * x + d ≤ 0) ↔ (x ≤ -5 ∨ x ≥ 3)) :
  (∃ a b : ℝ, a = 4 ∧ b = 1 ∧ (-x^2 + c * x + d = -x^2 + 8 * x - 15)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l289_28917


namespace NUMINAMATH_GPT_simple_interest_rate_l289_28992

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) :
  (T = 20) →
  (SI = P) →
  (SI = P * R * T / 100) →
  R = 5 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l289_28992


namespace NUMINAMATH_GPT_estimate_total_fish_l289_28946

theorem estimate_total_fish (marked : ℕ) (sample_size : ℕ) (marked_in_sample : ℕ) (x : ℝ) 
  (h1 : marked = 50) 
  (h2 : sample_size = 168) 
  (h3 : marked_in_sample = 8) 
  (h4 : sample_size * 50 = marked_in_sample * x) : 
  x = 1050 := 
sorry

end NUMINAMATH_GPT_estimate_total_fish_l289_28946


namespace NUMINAMATH_GPT_find_values_of_a_to_make_lines_skew_l289_28983

noncomputable def lines_are_skew (t u a : ℝ) : Prop :=
  ∀ t u,
    (1 + 2 * t = 4 + 5 * u ∧
     2 + 3 * t = 1 + 2 * u ∧
     a + 4 * t = u) → false

theorem find_values_of_a_to_make_lines_skew :
  ∀ a : ℝ, ¬ a = 3 ↔ lines_are_skew t u a :=
by
  sorry

end NUMINAMATH_GPT_find_values_of_a_to_make_lines_skew_l289_28983


namespace NUMINAMATH_GPT_SusanBooks_l289_28943

-- Definitions based on the conditions of the problem
def Lidia (S : ℕ) : ℕ := 4 * S
def TotalBooks (S : ℕ) : ℕ := S + Lidia S

-- The proof statement
theorem SusanBooks (S : ℕ) (h : TotalBooks S = 3000) : S = 600 :=
by
  sorry

end NUMINAMATH_GPT_SusanBooks_l289_28943


namespace NUMINAMATH_GPT_derivative_at_one_is_three_l289_28963

-- Definition of the function
def f (x : ℝ) := (x - 1)^2 + 3 * (x - 1)

-- The statement of the problem
theorem derivative_at_one_is_three : deriv f 1 = 3 := 
  sorry

end NUMINAMATH_GPT_derivative_at_one_is_three_l289_28963


namespace NUMINAMATH_GPT_tobys_friends_boys_count_l289_28954

theorem tobys_friends_boys_count (total_friends : ℕ) (girls : ℕ) (boys_percentage : ℕ) 
    (h1 : girls = 27) (h2 : boys_percentage = 55) (total_friends_calc : total_friends = 60) : 
    (total_friends * boys_percentage / 100) = 33 :=
by
  -- Proof is deferred
  sorry

end NUMINAMATH_GPT_tobys_friends_boys_count_l289_28954


namespace NUMINAMATH_GPT_factorize_expression_l289_28906

variable {X M N : ℕ}

theorem factorize_expression (x m n : ℕ) : x * m - x * n = x * (m - n) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l289_28906


namespace NUMINAMATH_GPT_find_polynomials_g_l289_28944

-- Assume f(x) = x^2
def f (x : ℝ) : ℝ := x ^ 2

-- Define the condition that f(g(x)) = 9x^2 - 6x + 1
def condition (g : ℝ → ℝ) : Prop := ∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1

-- Prove that the possible polynomials for g(x) are 3x - 1 or -3x + 1
theorem find_polynomials_g (g : ℝ → ℝ) (h : condition g) :
  (∀ x, g x = 3 * x - 1) ∨ (∀ x, g x = -3 * x + 1) :=
sorry

end NUMINAMATH_GPT_find_polynomials_g_l289_28944


namespace NUMINAMATH_GPT_binomial_expansion_problem_l289_28975

noncomputable def binomial_expansion_sum_coefficients (n : ℕ) : ℤ :=
  (1 - 3) ^ n

def general_term_coefficient (n r : ℕ) : ℤ :=
  (-3) ^ r * (Nat.choose n r)

theorem binomial_expansion_problem :
  ∃ (n : ℕ), binomial_expansion_sum_coefficients n = 64 ∧ general_term_coefficient 6 2 = 135 :=
by
  sorry

end NUMINAMATH_GPT_binomial_expansion_problem_l289_28975


namespace NUMINAMATH_GPT_correct_calculation_l289_28930

theorem correct_calculation (x a b : ℝ) : 
  (x^4 * x^4 = x^8) ∧ ((a^3)^2 = a^6) ∧ ((a * (b^2))^3 = a^3 * b^6) → (a + 2*a = 3*a) := 
by 
  sorry

end NUMINAMATH_GPT_correct_calculation_l289_28930


namespace NUMINAMATH_GPT_find_a_l289_28948

-- Definitions and conditions from the problem
def M (a : ℝ) : Set ℝ := {1, 2, a^2 - 3*a - 1}
def N (a : ℝ) : Set ℝ := {-1, a, 3}
def intersection_is_three (a : ℝ) : Prop := M a ∩ N a = {3}

-- The theorem we want to prove
theorem find_a (a : ℝ) (h : intersection_is_three a) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l289_28948


namespace NUMINAMATH_GPT_sequence_formula_l289_28920

theorem sequence_formula (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, n > 0 → a (n + 2) + 2 * a n = 3 * a (n + 1)) :
  (∀ n, a n = 3 * 2^(n-1) - 2) ∧ (S 4 > 21 - 2 * 4) :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l289_28920


namespace NUMINAMATH_GPT_crown_distribution_l289_28978

theorem crown_distribution 
  (A B C D E : ℤ) 
  (h1 : 2 * C = 3 * A)
  (h2 : 4 * D = 3 * B)
  (h3 : 4 * E = 5 * C)
  (h4 : 5 * D = 6 * A)
  (h5 : A + B + C + D + E = 2870) : 
  A = 400 ∧ B = 640 ∧ C = 600 ∧ D = 480 ∧ E = 750 := 
by 
  sorry

end NUMINAMATH_GPT_crown_distribution_l289_28978


namespace NUMINAMATH_GPT_smallest_integer_in_odd_set_l289_28955

theorem smallest_integer_in_odd_set (is_odd: ℤ → Prop)
  (median: ℤ) (greatest: ℤ) (smallest: ℤ) 
  (h1: median = 126)
  (h2: greatest = 153) 
  (h3: ∀ x, is_odd x ↔ ∃ k: ℤ, x = 2*k + 1)
  (h4: ∀ a b c, median = (a+b) / 2 → c = a → a ≤ b)
  : 
  smallest = 100 :=
sorry

end NUMINAMATH_GPT_smallest_integer_in_odd_set_l289_28955


namespace NUMINAMATH_GPT_correct_operation_l289_28972

theorem correct_operation : 
  (3 - Real.sqrt 2) ^ 2 = 11 - 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_correct_operation_l289_28972


namespace NUMINAMATH_GPT_solve_cyclist_return_speed_l289_28949

noncomputable def cyclist_return_speed (D : ℝ) (V : ℝ) : Prop :=
  let avg_speed := 9.5
  let out_speed := 10
  let T_out := D / out_speed
  let T_back := D / V
  2 * D / (T_out + T_back) = avg_speed

theorem solve_cyclist_return_speed : ∀ (D : ℝ), cyclist_return_speed D (20 / 2.1) :=
by
  intro D
  sorry

end NUMINAMATH_GPT_solve_cyclist_return_speed_l289_28949


namespace NUMINAMATH_GPT_cheese_cut_indefinite_l289_28947

theorem cheese_cut_indefinite (w : ℝ) (R : ℝ) (h : ℝ) :
  R = 0.5 →
  (∀ a b c d : ℝ, a > b → b > c → c > d →
    (∃ h, h < min (a - d) (d - c) ∧
     (d + h < a ∧ d - h > c))) →
  ∃ l1 l2 : ℕ → ℝ, (∀ n, l1 (n + 1) > l2 (n) ∧ l1 n > R * l2 (n)) :=
sorry

end NUMINAMATH_GPT_cheese_cut_indefinite_l289_28947


namespace NUMINAMATH_GPT_Jeffs_donuts_l289_28940

theorem Jeffs_donuts (D : ℕ) (h1 : ∀ n, n = 12 * D - 20) (h2 : n = 100) : D = 10 :=
by
  sorry

end NUMINAMATH_GPT_Jeffs_donuts_l289_28940


namespace NUMINAMATH_GPT_unique_real_root_t_l289_28942

theorem unique_real_root_t (t : ℝ) :
  (∃ x : ℝ, 3 * x + 7 * t - 2 + (2 * t * x^2 + 7 * t^2 - 9) / (x - t) = 0 ∧ 
  ∀ y : ℝ, 3 * y + 7 * t - 2 + (2 * t * y^2 + 7 * t^2 - 9) / (y - t) = 0 ∧ x ≠ y → false) →
  t = -3 ∨ t = -7 / 2 ∨ t = 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_real_root_t_l289_28942


namespace NUMINAMATH_GPT_triangle_median_equiv_l289_28971

-- Assuming necessary non-computable definitions (e.g., α for angles, R for real numbers) and non-computable nature of some geometric properties.

noncomputable def triangle (A B C : ℝ) := 
A + B + C = Real.pi

noncomputable def length_a (R A : ℝ) : ℝ := 2 * R * Real.sin A
noncomputable def length_b (R B : ℝ) : ℝ := 2 * R * Real.sin B
noncomputable def length_c (R C : ℝ) : ℝ := 2 * R * Real.sin C

noncomputable def median_a (b c A : ℝ) : ℝ := (2 * b * c) / (b + c) * Real.cos (A / 2)

theorem triangle_median_equiv (A B C R : ℝ) (hA : triangle A B C) :
  (1 / (length_a R A) + 1 / (length_b R B) = 1 / (median_a (length_b R B) (length_c R C) A)) ↔ (C = 2 * Real.pi / 3) := 
by sorry

end NUMINAMATH_GPT_triangle_median_equiv_l289_28971


namespace NUMINAMATH_GPT_SallyMcQueenCostCorrect_l289_28999

def LightningMcQueenCost : ℕ := 140000
def MaterCost : ℕ := (140000 * 10) / 100
def SallyMcQueenCost : ℕ := 3 * MaterCost

theorem SallyMcQueenCostCorrect : SallyMcQueenCost = 42000 := by
  sorry

end NUMINAMATH_GPT_SallyMcQueenCostCorrect_l289_28999


namespace NUMINAMATH_GPT_forty_percent_of_thirty_percent_l289_28953

theorem forty_percent_of_thirty_percent (x : ℝ) 
  (h : 0.3 * 0.4 * x = 48) : 0.4 * 0.3 * x = 48 :=
by
  sorry

end NUMINAMATH_GPT_forty_percent_of_thirty_percent_l289_28953


namespace NUMINAMATH_GPT_find_value_of_fraction_l289_28915

open Real

theorem find_value_of_fraction (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) : 
  (x + y) / (x - y) = -sqrt (5 / 3) :=
sorry

end NUMINAMATH_GPT_find_value_of_fraction_l289_28915


namespace NUMINAMATH_GPT_combined_basketballs_l289_28938

-- Conditions as definitions
def spursPlayers := 22
def rocketsPlayers := 18
def basketballsPerPlayer := 11

-- Math Proof Problem statement
theorem combined_basketballs : 
  (spursPlayers * basketballsPerPlayer) + (rocketsPlayers * basketballsPerPlayer) = 440 :=
by
  sorry

end NUMINAMATH_GPT_combined_basketballs_l289_28938


namespace NUMINAMATH_GPT_complementary_angle_difference_l289_28997

theorem complementary_angle_difference (a b : ℝ) (h1 : a = 4 * b) (h2 : a + b = 90) : (a - b) = 54 :=
by
  -- Proof is intentionally omitted
  sorry

end NUMINAMATH_GPT_complementary_angle_difference_l289_28997


namespace NUMINAMATH_GPT_cara_younger_than_mom_l289_28958

noncomputable def cara_grandmothers_age : ℤ := 75
noncomputable def cara_moms_age := cara_grandmothers_age - 15
noncomputable def cara_age : ℤ := 40

theorem cara_younger_than_mom :
  cara_moms_age - cara_age = 20 := by
  sorry

end NUMINAMATH_GPT_cara_younger_than_mom_l289_28958
