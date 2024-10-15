import Mathlib

namespace NUMINAMATH_GPT_distinct_triple_identity_l1838_183832

theorem distinct_triple_identity (p q r : ℝ) 
  (h1 : p ≠ q) 
  (h2 : q ≠ r) 
  (h3 : r ≠ p)
  (h : (p / (q - r)) + (q / (r - p)) + (r / (p - q)) = 3) : 
  (p^2 / (q - r)^2) + (q^2 / (r - p)^2) + (r^2 / (p - q)^2) = 3 :=
by 
  sorry

end NUMINAMATH_GPT_distinct_triple_identity_l1838_183832


namespace NUMINAMATH_GPT_largest_lcm_value_l1838_183856

theorem largest_lcm_value :
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 := by
sorry

end NUMINAMATH_GPT_largest_lcm_value_l1838_183856


namespace NUMINAMATH_GPT_number_of_Cl_atoms_l1838_183898

/-- 
Given a compound with 1 aluminum atom and a molecular weight of 132 g/mol,
prove that the number of chlorine atoms in the compound is 3.
--/
theorem number_of_Cl_atoms 
  (weight_Al : ℝ) 
  (weight_Cl : ℝ) 
  (molecular_weight : ℝ)
  (ha : weight_Al = 26.98)
  (hc : weight_Cl = 35.45)
  (hm : molecular_weight = 132) :
  ∃ n : ℕ, n = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_Cl_atoms_l1838_183898


namespace NUMINAMATH_GPT_evaluate_f_neg3_l1838_183886

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

theorem evaluate_f_neg3 (a b c : ℝ) (h : f 3 a b c = 11) : f (-3) a b c = -9 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_neg3_l1838_183886


namespace NUMINAMATH_GPT_calculate_material_needed_l1838_183824

theorem calculate_material_needed (area : ℝ) (pi_approx : ℝ) (extra_material : ℝ) (r : ℝ) (C : ℝ) : 
  area = 50.24 → pi_approx = 3.14 → extra_material = 4 → pi_approx * r ^ 2 = area → 
  C = 2 * pi_approx * r →
  C + extra_material = 29.12 :=
by
  intros h_area h_pi h_extra h_area_eq h_C_eq
  sorry

end NUMINAMATH_GPT_calculate_material_needed_l1838_183824


namespace NUMINAMATH_GPT_concert_ratio_l1838_183883

theorem concert_ratio (a c : ℕ) (h1 : 30 * a + 15 * c = 2250) (h2 : a ≥ 1) (h3 : c ≥ 1) :
  a = 50 ∧ c = 50 ∧ a = c := 
sorry

end NUMINAMATH_GPT_concert_ratio_l1838_183883


namespace NUMINAMATH_GPT_pens_solution_exists_l1838_183868

-- Definition of the conditions
def pen_cost_eq (x y : ℕ) : Prop :=
  17 * x + 12 * y = 150

-- Proof problem statement that follows from the conditions
theorem pens_solution_exists :
  ∃ x y : ℕ, pen_cost_eq x y :=
by
  existsi (6 : ℕ)
  existsi (4 : ℕ)
  -- Normally the proof would go here, but as stated, we use sorry.
  sorry

end NUMINAMATH_GPT_pens_solution_exists_l1838_183868


namespace NUMINAMATH_GPT_sum_six_digit_odd_and_multiples_of_3_l1838_183854

-- Definitions based on conditions
def num_six_digit_odd_numbers : Nat := 9 * (10 ^ 4) * 5

def num_six_digit_multiples_of_3 : Nat := 900000 / 3

-- Proof statement
theorem sum_six_digit_odd_and_multiples_of_3 : 
  num_six_digit_odd_numbers + num_six_digit_multiples_of_3 = 750000 := 
by 
  sorry

end NUMINAMATH_GPT_sum_six_digit_odd_and_multiples_of_3_l1838_183854


namespace NUMINAMATH_GPT_num_other_adults_l1838_183871

-- Define the variables and conditions
def num_baskets : ℕ := 15
def eggs_per_basket : ℕ := 12
def eggs_per_person : ℕ := 9
def shonda_kids : ℕ := 2
def kids_friends : ℕ := 10
def num_participants : ℕ := (num_baskets * eggs_per_basket) / eggs_per_person

-- Prove the number of other adults at the Easter egg hunt
theorem num_other_adults : (num_participants - (shonda_kids + kids_friends + 1)) = 7 := by
  sorry

end NUMINAMATH_GPT_num_other_adults_l1838_183871


namespace NUMINAMATH_GPT_maya_lift_increase_l1838_183808

def initial_lift_America : ℕ := 240
def peak_lift_America : ℕ := 300

def initial_lift_Maya (a_lift : ℕ) : ℕ := a_lift / 4
def peak_lift_Maya (p_lift : ℕ) : ℕ := p_lift / 2

def lift_difference (initial_lift : ℕ) (peak_lift : ℕ) : ℕ := peak_lift - initial_lift

theorem maya_lift_increase :
  lift_difference (initial_lift_Maya initial_lift_America) (peak_lift_Maya peak_lift_America) = 90 :=
by
  -- Proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_maya_lift_increase_l1838_183808


namespace NUMINAMATH_GPT_log_prime_factor_inequality_l1838_183853

open Real

-- Define p(n) such that it returns the number of prime factors of n.
noncomputable def p (n: ℕ) : ℕ := sorry  -- This will be defined contextually for now

theorem log_prime_factor_inequality (n : ℕ) (hn : n > 0) : 
  log n ≥ (p n) * log 2 :=
by 
  sorry

end NUMINAMATH_GPT_log_prime_factor_inequality_l1838_183853


namespace NUMINAMATH_GPT_solution_of_r_and_s_l1838_183839

theorem solution_of_r_and_s 
  (r s : ℝ)
  (hr : (r-5)*(r+5) = 25*r - 125)
  (hs : (s-5)*(s+5) = 25*s - 125)
  (h_distinct : r ≠ s)
  (h_order : r > s) : 
  r - s = 15 := by
sorry

end NUMINAMATH_GPT_solution_of_r_and_s_l1838_183839


namespace NUMINAMATH_GPT_find_dividend_l1838_183818

-- Define the conditions
def divisor : ℕ := 20
def quotient : ℕ := 8
def remainder : ℕ := 6

-- Lean 4 statement to prove the dividend
theorem find_dividend : (divisor * quotient + remainder) = 166 := by
  sorry

end NUMINAMATH_GPT_find_dividend_l1838_183818


namespace NUMINAMATH_GPT_minimum_shift_value_l1838_183846

noncomputable def f (x : ℝ) : ℝ := Real.cos x

theorem minimum_shift_value :
  ∃ m > 0, ∀ x, f (x + m) = Real.sin x ∧ m = 3 * Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_shift_value_l1838_183846


namespace NUMINAMATH_GPT_train_time_original_l1838_183840

theorem train_time_original (D : ℝ) (T : ℝ) 
  (h1 : D = 48 * T) 
  (h2 : D = 60 * (2/3)) : T = 5 / 6 := 
by
  sorry

end NUMINAMATH_GPT_train_time_original_l1838_183840


namespace NUMINAMATH_GPT_solution_l1838_183851

def problem (a b : ℝ) : Prop :=
  ∀ (x : ℝ), (x + a) * (x - 3) = x^2 + 2 * x - b

theorem solution (a b : ℝ) (h : problem a b) : a - b = -10 :=
  sorry

end NUMINAMATH_GPT_solution_l1838_183851


namespace NUMINAMATH_GPT_B_age_l1838_183830

-- Define the conditions
variables (x y : ℕ)
variable (current_year : ℕ)
axiom h1 : 10 * x + y + 4 = 43
axiom reference_year : current_year = 1955

-- Define the relationship between the digit equation and the year
def birth_year (x y : ℕ) : ℕ := 1900 + 10 * x + y

-- Birth year calculation
def age (current_year birth_year : ℕ) : ℕ := current_year - birth_year

-- Final theorem: Age of B
theorem B_age (x y : ℕ) (current_year : ℕ) (h1 : 10 * x  + y + 4 = 43) (reference_year : current_year = 1955) :
  age current_year (birth_year x y) = 16 :=
by
  sorry

end NUMINAMATH_GPT_B_age_l1838_183830


namespace NUMINAMATH_GPT_balance_scale_equation_l1838_183836

theorem balance_scale_equation 
  (G Y B W : ℝ)
  (h1 : 4 * G = 8 * B)
  (h2 : 3 * Y = 6 * B)
  (h3 : 2 * B = 3 * W) : 
  3 * G + 4 * Y + 3 * W = 16 * B :=
by
  sorry

end NUMINAMATH_GPT_balance_scale_equation_l1838_183836


namespace NUMINAMATH_GPT_inequality_problem_l1838_183891

theorem inequality_problem (x y a b : ℝ) (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < b) (h5 : b < 1) : (a ^ x < b ^ y) :=
by 
  sorry

end NUMINAMATH_GPT_inequality_problem_l1838_183891


namespace NUMINAMATH_GPT_binom_10_1_eq_10_l1838_183889

theorem binom_10_1_eq_10 : Nat.choose 10 1 = 10 := by
  sorry

end NUMINAMATH_GPT_binom_10_1_eq_10_l1838_183889


namespace NUMINAMATH_GPT_divide_estate_l1838_183894

theorem divide_estate (total_estate : ℕ) (son_share : ℕ) (daughter_share : ℕ) (wife_share : ℕ) :
  total_estate = 210 →
  son_share = (4 / 7) * total_estate →
  daughter_share = (1 / 7) * total_estate →
  wife_share = (2 / 7) * total_estate →
  son_share + daughter_share + wife_share = total_estate :=
by
  intros
  sorry

end NUMINAMATH_GPT_divide_estate_l1838_183894


namespace NUMINAMATH_GPT_cubic_identity_l1838_183866

variable {a b c : ℝ}

theorem cubic_identity (h1 : a + b + c = 13) (h2 : ab + ac + bc = 30) : a^3 + b^3 + c^3 - 3 * a * b * c = 1027 := 
by 
  sorry

end NUMINAMATH_GPT_cubic_identity_l1838_183866


namespace NUMINAMATH_GPT_find_y_when_x_is_7_l1838_183893

theorem find_y_when_x_is_7
  (x y : ℝ)
  (h1 : x * y = 384)
  (h2 : x + y = 40)
  (h3 : x - y = 8)
  (h4 : x = 7) :
  y = 384 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_y_when_x_is_7_l1838_183893


namespace NUMINAMATH_GPT_division_problem_l1838_183847

theorem division_problem : 8900 / 6 / 4 = 1483.3333 :=
by sorry

end NUMINAMATH_GPT_division_problem_l1838_183847


namespace NUMINAMATH_GPT_number_of_small_cubes_l1838_183838

theorem number_of_small_cubes (X : ℕ) (h1 : ∃ k, k = 29 - X) (h2 : 4 * 4 * 4 = 64) (h3 : X + 8 * (29 - X) = 64) : X = 24 :=
by
  sorry

end NUMINAMATH_GPT_number_of_small_cubes_l1838_183838


namespace NUMINAMATH_GPT_percentage_shaded_in_square_l1838_183828

theorem percentage_shaded_in_square
  (EFGH : Type)
  (square : EFGH → Prop)
  (side_length : EFGH → ℝ)
  (area : EFGH → ℝ)
  (shaded_area : EFGH → ℝ)
  (P : EFGH)
  (h_square : square P)
  (h_side_length : side_length P = 8)
  (h_area : area P = side_length P * side_length P)
  (h_small_shaded : shaded_area P = 4)
  (h_large_shaded : shaded_area P + 7 = 11) :
  (shaded_area P / area P) * 100 = 17.1875 :=
by
  sorry

end NUMINAMATH_GPT_percentage_shaded_in_square_l1838_183828


namespace NUMINAMATH_GPT_towel_percentage_decrease_l1838_183869

theorem towel_percentage_decrease
  (L B: ℝ)
  (original_area : ℝ := L * B)
  (new_length : ℝ := 0.70 * L)
  (new_breadth : ℝ := 0.75 * B)
  (new_area : ℝ := new_length * new_breadth) :
  ((original_area - new_area) / original_area) * 100 = 47.5 := 
by 
  sorry

end NUMINAMATH_GPT_towel_percentage_decrease_l1838_183869


namespace NUMINAMATH_GPT_avianna_blue_candles_l1838_183844

theorem avianna_blue_candles (r b : ℕ) (h1 : r = 45) (h2 : r/b = 5/3) : b = 27 :=
by sorry

end NUMINAMATH_GPT_avianna_blue_candles_l1838_183844


namespace NUMINAMATH_GPT_angle_BAC_l1838_183882

theorem angle_BAC (A B C D : Type*) (AD BD CD : ℝ) (angle_BCA : ℝ) 
  (h_AD_BD : AD = BD) (h_BD_CD : BD = CD) (h_angle_BCA : angle_BCA = 40) :
  ∃ angle_BAC : ℝ, angle_BAC = 110 := 
sorry

end NUMINAMATH_GPT_angle_BAC_l1838_183882


namespace NUMINAMATH_GPT_min_value_of_expression_l1838_183870

theorem min_value_of_expression (m n : ℝ) (h1 : m + 2 * n = 2) (h2 : m > 0) (h3 : n > 0) : 
  (1 / (m + 1) + 1 / (2 * n)) ≥ 4 / 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1838_183870


namespace NUMINAMATH_GPT_angelaAgeInFiveYears_l1838_183872

namespace AgeProblem

variables (A B : ℕ) -- Define Angela's and Beth's current age as natural numbers.

-- Condition 1: Angela is four times as old as Beth.
axiom angelaAge : A = 4 * B

-- Condition 2: Five years ago, the sum of their ages was 45 years.
axiom ageSumFiveYearsAgo : (A - 5) + (B - 5) = 45

-- Theorem: Prove that Angela's age in 5 years will be 49.
theorem angelaAgeInFiveYears : A + 5 = 49 :=
by {
  -- proof goes here
  sorry
}

end AgeProblem

end NUMINAMATH_GPT_angelaAgeInFiveYears_l1838_183872


namespace NUMINAMATH_GPT_bisection_method_correctness_l1838_183855

noncomputable def initial_interval_length : ℝ := 1
noncomputable def required_precision : ℝ := 0.01
noncomputable def minimum_bisections : ℕ := 7

theorem bisection_method_correctness :
  ∃ n : ℕ, (n ≥ minimum_bisections) ∧ (initial_interval_length / 2^n ≤ required_precision) :=
by
  sorry

end NUMINAMATH_GPT_bisection_method_correctness_l1838_183855


namespace NUMINAMATH_GPT_integer_solutions_of_cubic_equation_l1838_183850

theorem integer_solutions_of_cubic_equation :
  ∀ (n m : ℤ),
    n ^ 6 + 3 * n ^ 5 + 3 * n ^ 4 + 2 * n ^ 3 + 3 * n ^ 2 + 3 * n + 1 = m ^ 3 ↔
    (n = 0 ∧ m = 1) ∨ (n = -1 ∧ m = 0) :=
by
  intro n m
  apply Iff.intro
  { intro h
    sorry }
  { intro h
    sorry }

end NUMINAMATH_GPT_integer_solutions_of_cubic_equation_l1838_183850


namespace NUMINAMATH_GPT_tony_fever_temperature_above_threshold_l1838_183819

theorem tony_fever_temperature_above_threshold 
  (n : ℕ) (i : ℕ) (f : ℕ) 
  (h1 : n = 95) (h2 : i = 10) (h3 : f = 100) : 
  n + i - f = 5 :=
by
  sorry

end NUMINAMATH_GPT_tony_fever_temperature_above_threshold_l1838_183819


namespace NUMINAMATH_GPT_negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic_l1838_183897

-- Definitions based on the conditions in the problem:
def LinearFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b
def MonotonicFunction (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- The proposition that 'All linear functions are monotonic functions'
def AllLinearAreMonotonic : Prop := ∀ (f : ℝ → ℝ), LinearFunction f → MonotonicFunction f

-- The correct answer to the question:
def SomeLinearAreNotMonotonic : Prop := ∃ (f : ℝ → ℝ), LinearFunction f ∧ ¬ MonotonicFunction f

-- The proof problem:
theorem negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic : 
  ¬ AllLinearAreMonotonic ↔ SomeLinearAreNotMonotonic :=
by
  sorry

end NUMINAMATH_GPT_negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic_l1838_183897


namespace NUMINAMATH_GPT_proof_G_eq_BC_eq_D_eq_AB_AC_l1838_183801

-- Let's define the conditions of the problem first
variables (A B C O D G F E : Type) [Field A] [Field B] [Field C] [Field O] [Field D] [Field G] [Field F] [Field E]

-- Given triangle ABC with circumcenter O
variable {triangle_ABC: Prop}

-- Given point D on line segment BC
variable (D_on_BC : Prop)

-- Given circle Gamma with diameter OD
variable (circle_Gamma : Prop)

-- Given circles Gamma_1 and Gamma_2 are circumcircles of triangles ABD and ACD respectively
variable (circle_Gamma1 : Prop)
variable (circle_Gamma2 : Prop)

-- Given points F and E as intersection points
variable (intersect_F : Prop)
variable (intersect_E : Prop)

-- Given G as the second intersection point of the circumcircles of triangles BED and DFC
variable (second_intersect_G : Prop)

-- Prove that the condition for point G to be equidistant from points B and C is that point D is equidistant from lines AB and AC
theorem proof_G_eq_BC_eq_D_eq_AB_AC : 
  triangle_ABC ∧ D_on_BC ∧ circle_Gamma ∧ circle_Gamma1 ∧ circle_Gamma2 ∧ intersect_F ∧ intersect_E ∧ second_intersect_G → 
  G_dist_BC ↔ D_dist_AB_AC :=
by
  sorry

end NUMINAMATH_GPT_proof_G_eq_BC_eq_D_eq_AB_AC_l1838_183801


namespace NUMINAMATH_GPT_total_pieces_of_clothing_l1838_183811

-- Define the conditions:
def boxes : ℕ := 4
def scarves_per_box : ℕ := 2
def mittens_per_box : ℕ := 6

-- Define the target statement:
theorem total_pieces_of_clothing : (boxes * (scarves_per_box + mittens_per_box)) = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_pieces_of_clothing_l1838_183811


namespace NUMINAMATH_GPT_domain_of_composite_function_l1838_183800

theorem domain_of_composite_function
    (f : ℝ → ℝ)
    (h : ∀ x, -2 ≤ x ∧ x ≤ 3 → f (x + 1) ∈ (Set.Icc (-2:ℝ) (3:ℝ))):
    ∃ s : Set ℝ, s = Set.Icc 0 (5/2) ∧ (∀ x, x ∈ s ↔ f (2 * x - 1) ∈ Set.Icc (-1) 4) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_composite_function_l1838_183800


namespace NUMINAMATH_GPT_minimum_value_a_l1838_183814

noncomputable def f (a b x : ℝ) := a * Real.log x - (1 / 2) * x^2 + b * x

theorem minimum_value_a (h : ∀ b x : ℝ, x > 0 → f a b x > 0) : a ≥ -Real.exp 3 := 
sorry

end NUMINAMATH_GPT_minimum_value_a_l1838_183814


namespace NUMINAMATH_GPT_percentage_of_smoking_teens_l1838_183809

theorem percentage_of_smoking_teens (total_students : ℕ) (hospitalized_percentage : ℝ) (non_hospitalized_count : ℕ) 
  (h_total_students : total_students = 300)
  (h_hospitalized_percentage : hospitalized_percentage = 0.70)
  (h_non_hospitalized_count : non_hospitalized_count = 36) : 
  (non_hospitalized_count / (total_students * (1 - hospitalized_percentage))) * 100 = 40 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_of_smoking_teens_l1838_183809


namespace NUMINAMATH_GPT_max_abc_value_l1838_183896

variables (a b c : ℕ)

theorem max_abc_value : 
  (a > 0) → (b > 0) → (c > 0) → a + 2 * b + 3 * c = 100 → abc ≤ 6171 := 
by sorry

end NUMINAMATH_GPT_max_abc_value_l1838_183896


namespace NUMINAMATH_GPT_quadratic_one_real_root_positive_m_l1838_183842

theorem quadratic_one_real_root_positive_m (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ((6 * m)^2 - 4 * 1 * (2 * m) = 0)) → m = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_one_real_root_positive_m_l1838_183842


namespace NUMINAMATH_GPT_trig_expr_value_l1838_183812

theorem trig_expr_value :
  (Real.cos (7 * Real.pi / 24)) ^ 4 +
  (Real.sin (11 * Real.pi / 24)) ^ 4 +
  (Real.sin (17 * Real.pi / 24)) ^ 4 +
  (Real.cos (13 * Real.pi / 24)) ^ 4 = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_expr_value_l1838_183812


namespace NUMINAMATH_GPT_square_side_length_l1838_183802

theorem square_side_length (π : ℝ) (s : ℝ) :
  (∃ r : ℝ, 100 = π * r^2) ∧ (4 * s = 100) → s = 25 := by
  sorry

end NUMINAMATH_GPT_square_side_length_l1838_183802


namespace NUMINAMATH_GPT_sum_of_coefficients_eq_one_l1838_183825

theorem sum_of_coefficients_eq_one :
  ∀ x y : ℤ, (x - 2 * y) ^ 18 = (1 - 2 * 1) ^ 18 → (x - 2 * y) ^ 18 = 1 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_eq_one_l1838_183825


namespace NUMINAMATH_GPT_trapezium_top_width_l1838_183888

theorem trapezium_top_width (bottom_width : ℝ) (height : ℝ) (area : ℝ) (top_width : ℝ) 
  (h1 : bottom_width = 8) 
  (h2 : height = 50) 
  (h3 : area = 500) : top_width = 12 :=
by
  -- Definitions
  have h_formula : area = 1 / 2 * (top_width + bottom_width) * height := by sorry
  -- Applying given conditions to the formula
  rw [h1, h2, h3] at h_formula
  -- Solve for top_width
  sorry

end NUMINAMATH_GPT_trapezium_top_width_l1838_183888


namespace NUMINAMATH_GPT_no_pairs_satisfy_equation_l1838_183837

theorem no_pairs_satisfy_equation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a^2 + 1 / b^2 = 1 / (a^2 + b^2)) → False :=
by
  sorry

end NUMINAMATH_GPT_no_pairs_satisfy_equation_l1838_183837


namespace NUMINAMATH_GPT_hockey_pads_cost_l1838_183892

theorem hockey_pads_cost
  (initial_money : ℕ)
  (cost_hockey_skates : ℕ)
  (remaining_money : ℕ)
  (h : initial_money = 150)
  (h1 : cost_hockey_skates = initial_money / 2)
  (h2 : remaining_money = 25) :
  initial_money - cost_hockey_skates - 50 = remaining_money :=
by sorry

end NUMINAMATH_GPT_hockey_pads_cost_l1838_183892


namespace NUMINAMATH_GPT_factorize_expr_l1838_183862

theorem factorize_expr (a b : ℝ) : 2 * a^2 - a * b = a * (2 * a - b) := 
by
  sorry

end NUMINAMATH_GPT_factorize_expr_l1838_183862


namespace NUMINAMATH_GPT_Bernardo_wins_with_smallest_M_l1838_183860

-- Define the operations
def Bernardo_op (n : ℕ) : ℕ := 3 * n
def Lucas_op (n : ℕ) : ℕ := n + 75

-- Define the game behavior
def game_sequence (M : ℕ) : List ℕ :=
  [M, Bernardo_op M, Lucas_op (Bernardo_op M), Bernardo_op (Lucas_op (Bernardo_op M)),
   Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))),
   Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M)))),
   Lucas_op (Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))))),
   Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))))))]

-- Define winning condition
def Bernardo_wins (M : ℕ) : Prop :=
  let seq := game_sequence M
  seq.get! 5 < 1200 ∧ seq.get! 6 >= 1200

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- The final theorem statement
theorem Bernardo_wins_with_smallest_M :
  Bernardo_wins 9 ∧ (∀ M < 9, ¬Bernardo_wins M) ∧ sum_of_digits 9 = 9 :=
by
  sorry

end NUMINAMATH_GPT_Bernardo_wins_with_smallest_M_l1838_183860


namespace NUMINAMATH_GPT_setB_forms_right_triangle_l1838_183843

-- Define the sets of side lengths
def setA : (ℕ × ℕ × ℕ) := (2, 3, 4)
def setB : (ℕ × ℕ × ℕ) := (3, 4, 5)
def setC : (ℕ × ℕ × ℕ) := (5, 6, 7)
def setD : (ℕ × ℕ × ℕ) := (7, 8, 9)

-- Define the Pythagorean theorem condition
def isRightTriangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- The specific proof goal
theorem setB_forms_right_triangle : isRightTriangle 3 4 5 := by
  sorry

end NUMINAMATH_GPT_setB_forms_right_triangle_l1838_183843


namespace NUMINAMATH_GPT_cash_after_brokerage_l1838_183849

theorem cash_after_brokerage (sale_amount : ℝ) (brokerage_rate : ℝ) :
  sale_amount = 109.25 → brokerage_rate = 0.0025 →
  (sale_amount - sale_amount * brokerage_rate) = 108.98 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_cash_after_brokerage_l1838_183849


namespace NUMINAMATH_GPT_max_side_length_l1838_183874

theorem max_side_length (a b c : ℕ) (h : a + b + c = 30) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_order : a ≤ b ∧ b ≤ c) (h_triangle_ineq : a + b > c) : c ≤ 14 := 
sorry

end NUMINAMATH_GPT_max_side_length_l1838_183874


namespace NUMINAMATH_GPT_jessica_initial_withdrawal_fraction_l1838_183816

variable {B : ℝ} -- this is the initial balance

noncomputable def initial_withdrawal_fraction (B : ℝ) : Prop :=
  let remaining_balance := B - 400
  let deposit := (1 / 4) * remaining_balance
  let final_balance := remaining_balance + deposit
  final_balance = 750 → (400 / B) = 2 / 5

-- Our goal is to prove the statement given conditions.
theorem jessica_initial_withdrawal_fraction : 
  ∃ B : ℝ, initial_withdrawal_fraction B :=
sorry

end NUMINAMATH_GPT_jessica_initial_withdrawal_fraction_l1838_183816


namespace NUMINAMATH_GPT_diagonals_from_vertex_of_regular_polygon_l1838_183885

-- Definitions for the conditions in part a)
def exterior_angle (n : ℕ) : ℚ := 360 / n

-- Proof problem statement
theorem diagonals_from_vertex_of_regular_polygon
  (n : ℕ)
  (h1 : exterior_angle n = 36)
  : n - 3 = 7 :=
by sorry

end NUMINAMATH_GPT_diagonals_from_vertex_of_regular_polygon_l1838_183885


namespace NUMINAMATH_GPT_dig_eq_conditions_l1838_183884

theorem dig_eq_conditions (n k : ℕ) 
  (h1 : 10^(k-1) ≤ n^n ∧ n^n < 10^k)
  (h2 : 10^(n-1) ≤ k^k ∧ k^k < 10^n) :
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) :=
by
  sorry

end NUMINAMATH_GPT_dig_eq_conditions_l1838_183884


namespace NUMINAMATH_GPT_pen_price_equation_l1838_183841

theorem pen_price_equation
  (x y : ℤ)
  (h1 : 100 * x - y = 100)
  (h2 : 2 * y - 100 * x = 200) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_pen_price_equation_l1838_183841


namespace NUMINAMATH_GPT_plane_triangle_coverage_l1838_183826

noncomputable def percentage_triangles_covered (a : ℝ) : ℝ :=
  let total_area := (4 * a) ^ 2
  let triangle_area := 10 * (1 / 2 * a^2)
  (triangle_area / total_area) * 100

theorem plane_triangle_coverage (a : ℝ) :
  abs (percentage_triangles_covered a - 31.25) < 0.75 :=
  sorry

end NUMINAMATH_GPT_plane_triangle_coverage_l1838_183826


namespace NUMINAMATH_GPT_fraction_walk_is_three_twentieths_l1838_183817

-- Define the various fractions given in the conditions
def fraction_bus : ℚ := 1 / 2
def fraction_auto : ℚ := 1 / 4
def fraction_bicycle : ℚ := 1 / 10

-- Defining the total fraction for students that do not walk
def total_not_walk : ℚ := fraction_bus + fraction_auto + fraction_bicycle

-- The remaining fraction after subtracting from 1
def fraction_walk : ℚ := 1 - total_not_walk

-- The theorem we want to prove that fraction_walk is 3/20
theorem fraction_walk_is_three_twentieths : fraction_walk = 3 / 20 := by
  sorry

end NUMINAMATH_GPT_fraction_walk_is_three_twentieths_l1838_183817


namespace NUMINAMATH_GPT_find_first_number_l1838_183875

/-- Given a sequence of 6 numbers b_1, b_2, ..., b_6 such that:
  1. For n ≥ 2, b_{2n} = b_{2n-1}^2
  2. For n ≥ 2, b_{2n+1} = (b_{2n} * b_{2n-1})^2
And the sequence ends as: b_4 = 16, b_5 = 256, and b_6 = 65536,
prove that the first number b_1 is 1/2. -/
theorem find_first_number : 
  ∃ b : ℕ → ℝ, b 6 = 65536 ∧ b 5 = 256 ∧ b 4 = 16 ∧ 
  (∀ n ≥ 2, b (2 * n) = (b (2 * n - 1)) ^ 2) ∧
  (∀ n ≥ 2, b (2 * n + 1) = (b (2 * n) * b (2 * n - 1)) ^ 2) ∧ 
  b 1 = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_l1838_183875


namespace NUMINAMATH_GPT_range_of_a_l1838_183848

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3 * a) ↔ (a ≤ -1 ∨ a ≥ 4) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1838_183848


namespace NUMINAMATH_GPT_minimum_value_analysis_l1838_183805

theorem minimum_value_analysis
  (a : ℝ) (m n : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : 2 * m + n = 2)
  (h4 : m > 0)
  (h5 : n > 0) :
  (2 / m + 1 / n) ≥ 9 / 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_analysis_l1838_183805


namespace NUMINAMATH_GPT_yellow_balls_count_l1838_183820

theorem yellow_balls_count {R B Y G : ℕ} 
  (h1 : R + B + Y + G = 531)
  (h2 : R + B = Y + G + 31)
  (h3 : Y = G + 22) : 
  Y = 136 :=
by
  -- The proof is skipped, as requested.
  sorry

end NUMINAMATH_GPT_yellow_balls_count_l1838_183820


namespace NUMINAMATH_GPT_max_min_value_of_f_l1838_183822

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem max_min_value_of_f :
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ f (Real.pi / 6)) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f (Real.pi / 2) ≤ f x) :=
by
  sorry

end NUMINAMATH_GPT_max_min_value_of_f_l1838_183822


namespace NUMINAMATH_GPT_triangle_ratio_inequality_l1838_183865

/-- Given a triangle ABC, R is the radius of the circumscribed circle, 
    r is the radius of the inscribed circle, a is the length of the longest side,
    and h is the length of the shortest altitude. Prove that R / r > a / h. -/
theorem triangle_ratio_inequality
  (ABC : Triangle) (R r a h : ℝ)
  (hR : 2 * R ≥ a)
  (hr : 2 * r < h) :
  (R / r) > (a / h) :=
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_triangle_ratio_inequality_l1838_183865


namespace NUMINAMATH_GPT_problem_statement_l1838_183861

theorem problem_statement (a b : ℝ) (h : a^2 + |b + 1| = 0) : (a + b)^2015 = -1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1838_183861


namespace NUMINAMATH_GPT_Sonja_oil_used_l1838_183890

theorem Sonja_oil_used :
  ∀ (oil peanuts total_weight : ℕ),
  (2 * oil + 8 * peanuts = 10) → (total_weight = 20) →
  ((20 / 10) * 2 = 4) :=
by 
  sorry

end NUMINAMATH_GPT_Sonja_oil_used_l1838_183890


namespace NUMINAMATH_GPT_abs_expression_equals_l1838_183823

theorem abs_expression_equals (h : Real.pi < 12) : 
  abs (Real.pi - abs (Real.pi - 12)) = 12 - 2 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_abs_expression_equals_l1838_183823


namespace NUMINAMATH_GPT_kathryn_remaining_money_l1838_183880

/-- Define the conditions --/
def rent := 1200
def salary := 5000
def food_and_travel_expenses := 2 * rent
def new_rent := rent / 2
def total_expenses := food_and_travel_expenses + new_rent
def remaining_money := salary - total_expenses

/-- Theorem to be proved --/
theorem kathryn_remaining_money : remaining_money = 2000 := by
  sorry

end NUMINAMATH_GPT_kathryn_remaining_money_l1838_183880


namespace NUMINAMATH_GPT_fixed_point_of_shifted_exponential_l1838_183858

theorem fixed_point_of_shifted_exponential (a : ℝ) (H : a^0 = 1) : a^(3-3) + 3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_shifted_exponential_l1838_183858


namespace NUMINAMATH_GPT_rational_numbers_include_positives_and_negatives_l1838_183864

theorem rational_numbers_include_positives_and_negatives :
  ∃ (r : ℚ), r > 0 ∧ ∃ (r' : ℚ), r' < 0 :=
by
  sorry

end NUMINAMATH_GPT_rational_numbers_include_positives_and_negatives_l1838_183864


namespace NUMINAMATH_GPT_trig_identity_example_l1838_183810

theorem trig_identity_example : 4 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 := 
by
  -- The statement "π/12" is mathematically equivalent to 15 degrees.
  sorry

end NUMINAMATH_GPT_trig_identity_example_l1838_183810


namespace NUMINAMATH_GPT_simplify_expression_l1838_183835

noncomputable def sqrt' (x : ℝ) : ℝ := Real.sqrt x

theorem simplify_expression :
  (3 * sqrt' 8 / (sqrt' 2 + sqrt' 3 + sqrt' 7)) = (sqrt' 2 + sqrt' 3 - sqrt' 7) := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1838_183835


namespace NUMINAMATH_GPT_trigonometric_identity_l1838_183859

theorem trigonometric_identity (α : ℝ) : 
  (1 + Real.cos (2 * α - 2 * Real.pi) + Real.cos (4 * α + 2 * Real.pi) - Real.cos (6 * α - Real.pi)) /
  (Real.cos (2 * Real.pi - 2 * α) + 2 * Real.cos (2 * α + Real.pi) ^ 2 - 1) = 
  2 * Real.cos (2 * α) :=
by sorry

end NUMINAMATH_GPT_trigonometric_identity_l1838_183859


namespace NUMINAMATH_GPT_slope_probability_l1838_183873

noncomputable def probability_of_slope_gte (x y : ℝ) (Q : ℝ × ℝ) : ℝ :=
  if y - 1 / 4 ≥ (2 / 3) * (x - 3 / 4) then 1 else 0

theorem slope_probability :
  let unit_square_area := 1  -- the area of the unit square
  let valid_area := (1 / 2) * (5 / 8) * (5 / 12) -- area of the triangle above the line
  valid_area / unit_square_area = 25 / 96 :=
sorry

end NUMINAMATH_GPT_slope_probability_l1838_183873


namespace NUMINAMATH_GPT_original_students_count_l1838_183863

theorem original_students_count (N : ℕ) (T : ℕ) :
  (T = N * 85) →
  ((N - 5) * 90 = T - 300) →
  ((N - 8) * 95 = T - 465) →
  ((N - 15) * 100 = T - 955) →
  N = 30 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_original_students_count_l1838_183863


namespace NUMINAMATH_GPT_large_pizza_slices_l1838_183895

variable (L : ℕ)

theorem large_pizza_slices :
  (2 * L + 2 * 8 = 48) → (L = 16) :=
by 
  sorry

end NUMINAMATH_GPT_large_pizza_slices_l1838_183895


namespace NUMINAMATH_GPT_domain_of_function_l1838_183879

variable (x : ℝ)

theorem domain_of_function :
  {x : ℝ | x + 3 ≥ 0 ∧ 2 - x ≠ 0} =
  {x : ℝ | x ≥ -3 ∧ x ≠ 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1838_183879


namespace NUMINAMATH_GPT_total_cookies_l1838_183876

-- Definitions of the conditions
def cookies_in_bag : ℕ := 21
def bags_in_box : ℕ := 4
def boxes : ℕ := 2

-- Theorem stating the total number of cookies
theorem total_cookies : cookies_in_bag * bags_in_box * boxes = 168 := by
  sorry

end NUMINAMATH_GPT_total_cookies_l1838_183876


namespace NUMINAMATH_GPT_smaller_angle_is_70_l1838_183807

def measure_of_smaller_angle (x : ℕ) : Prop :=
  (x + (x + 40) = 180) ∧ (2 * x - 60 = 80)

theorem smaller_angle_is_70 {x : ℕ} : measure_of_smaller_angle x → x = 70 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_is_70_l1838_183807


namespace NUMINAMATH_GPT_interchanged_digits_subtraction_l1838_183803

theorem interchanged_digits_subtraction (a b k : ℤ) (h1 : 10 * a + b = 2 * k * (a + b)) :
  10 * b + a - 3 * (a + b) = (9 - 4 * k) * (a + b) :=
by sorry

end NUMINAMATH_GPT_interchanged_digits_subtraction_l1838_183803


namespace NUMINAMATH_GPT_three_squares_sum_l1838_183852

theorem three_squares_sum (n : ℤ) (h : n > 5) : 
  3 * (n - 1)^2 + 32 = (n - 5)^2 + (n - 1)^2 + (n + 3)^2 :=
by sorry

end NUMINAMATH_GPT_three_squares_sum_l1838_183852


namespace NUMINAMATH_GPT_xy_value_l1838_183899

structure Point (R : Type) := (x : R) (y : R)

def A : Point ℝ := ⟨2, 7⟩ 
def C : Point ℝ := ⟨4, 3⟩ 

def is_midpoint (A B C : Point ℝ) : Prop :=
  (C.x = (A.x + B.x) / 2) ∧ (C.y = (A.y + B.y) / 2)

theorem xy_value (x y : ℝ) (B : Point ℝ := ⟨x, y⟩) (H : is_midpoint A B C) :
  x * y = -6 := 
sorry

end NUMINAMATH_GPT_xy_value_l1838_183899


namespace NUMINAMATH_GPT_consecutive_sum_is_10_l1838_183881

theorem consecutive_sum_is_10 (a : ℕ) (h : a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 50) : a + 2 = 10 :=
sorry

end NUMINAMATH_GPT_consecutive_sum_is_10_l1838_183881


namespace NUMINAMATH_GPT_real_number_identity_l1838_183815

theorem real_number_identity (a : ℝ) (h : a^2 - a - 1 = 0) : a^8 + 7 * a^(-(4:ℝ)) = 48 := by
  sorry

end NUMINAMATH_GPT_real_number_identity_l1838_183815


namespace NUMINAMATH_GPT_train_speed_incl_stoppages_l1838_183821

theorem train_speed_incl_stoppages
  (speed_excl_stoppages : ℝ)
  (stoppage_time_minutes : ℝ)
  (h1 : speed_excl_stoppages = 42)
  (h2 : stoppage_time_minutes = 21.428571428571423)
  : ∃ speed_incl_stoppages, speed_incl_stoppages = 27 := 
sorry

end NUMINAMATH_GPT_train_speed_incl_stoppages_l1838_183821


namespace NUMINAMATH_GPT_solve_for_P_l1838_183834

theorem solve_for_P (P : Real) (h : (P ^ 4) ^ (1 / 3) = 9 * 81 ^ (1 / 9)) : P = 3 ^ (11 / 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_P_l1838_183834


namespace NUMINAMATH_GPT_distance_inequality_l1838_183827

theorem distance_inequality 
  (A B C D : Point)
  (dist : Point → Point → ℝ)
  (h_dist_pos : ∀ P Q : Point, dist P Q ≥ 0)
  (AC BD AD BC AB CD : ℝ)
  (hAC : AC = dist A C)
  (hBD : BD = dist B D)
  (hAD : AD = dist A D)
  (hBC : BC = dist B C)
  (hAB : AB = dist A B)
  (hCD : CD = dist C D) :
  AC^2 + BD^2 + AD^2 + BC^2 ≥ AB^2 + CD^2 := 
by
  sorry

end NUMINAMATH_GPT_distance_inequality_l1838_183827


namespace NUMINAMATH_GPT_a_eq_zero_l1838_183829

theorem a_eq_zero (a : ℝ) (h : ∀ x : ℝ, ax + 2 ≠ 0) : a = 0 := by
  sorry

end NUMINAMATH_GPT_a_eq_zero_l1838_183829


namespace NUMINAMATH_GPT_compare_x_y_l1838_183857

theorem compare_x_y (a b : ℝ) (h1 : a > b) (h2 : b > 1) (x y : ℝ)
  (hx : x = a + 1 / a) (hy : y = b + 1 / b) : x > y :=
by {
  sorry
}

end NUMINAMATH_GPT_compare_x_y_l1838_183857


namespace NUMINAMATH_GPT_ratio_ac_l1838_183878

-- Definitions based on conditions
variables (a b c : ℕ)
variables (x y : ℕ)

-- Conditions
def ratio_ab := (a : ℚ) / (b : ℚ) = 2 / 3
def ratio_bc := (b : ℚ) / (c : ℚ) = 1 / 5

-- Theorem to prove the desired ratio
theorem ratio_ac (h1 : ratio_ab a b) (h2 : ratio_bc b c) : (a : ℚ) / (c : ℚ) = 2 / 15 :=
by
  sorry

end NUMINAMATH_GPT_ratio_ac_l1838_183878


namespace NUMINAMATH_GPT_problem_statement_l1838_183806

open Nat

theorem problem_statement (n a : ℕ) 
  (hn : n > 1) 
  (ha : a > n^2)
  (H : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ k, a + i = (n^2 + i) * k) :
  a > n^4 - n^3 := 
sorry

end NUMINAMATH_GPT_problem_statement_l1838_183806


namespace NUMINAMATH_GPT_even_digit_number_division_l1838_183845

theorem even_digit_number_division (N : ℕ) (n : ℕ) :
  (N % 2 = 0) ∧
  (∃ a b : ℕ, (∀ k : ℕ, N = a * 10^n + b → N = k * (a * b)) ∧
  ((N = (1000^(2*n - 1) + 1)^2 / 7) ∨
   (N = 12) ∨
   (N = (10^n + 2)^2 / 6) ∨
   (N = 1352) ∨
   (N = 15))) :=
sorry

end NUMINAMATH_GPT_even_digit_number_division_l1838_183845


namespace NUMINAMATH_GPT_cleanup_drive_weight_per_mile_per_hour_l1838_183833

theorem cleanup_drive_weight_per_mile_per_hour :
  let duration := 4
  let lizzie_group := 387
  let second_group := lizzie_group - 39
  let third_group := 560 / 16
  let total_distance := 8
  let total_garbage := lizzie_group + second_group + third_group
  total_garbage / total_distance / duration = 24.0625 := 
by {
  sorry
}

end NUMINAMATH_GPT_cleanup_drive_weight_per_mile_per_hour_l1838_183833


namespace NUMINAMATH_GPT_lesser_number_is_21_5_l1838_183877

theorem lesser_number_is_21_5
  (x y : ℝ)
  (h1 : x + y = 50)
  (h2 : x - y = 7) :
  y = 21.5 :=
by
  sorry

end NUMINAMATH_GPT_lesser_number_is_21_5_l1838_183877


namespace NUMINAMATH_GPT_no_solution_inequality_system_l1838_183887

theorem no_solution_inequality_system (m : ℝ) :
  (¬ ∃ x : ℝ, 2 * x - 1 < 3 ∧ x > m) ↔ m ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_inequality_system_l1838_183887


namespace NUMINAMATH_GPT_find_b_l1838_183867

theorem find_b (b c x1 x2 : ℝ)
  (h_parabola_intersects_x_axis : (x1 ≠ x2) ∧ x1 * x2 = c ∧ x1 + x2 = -b ∧ x2 - x1 = 1)
  (h_parabola_intersects_y_axis : c ≠ 0)
  (h_length_ab : x2 - x1 = 1)
  (h_area_abc : (1 / 2) * (x2 - x1) * |c| = 1)
  : b = -3 :=
sorry

end NUMINAMATH_GPT_find_b_l1838_183867


namespace NUMINAMATH_GPT_inequality_a_neg_one_inequality_general_a_l1838_183831

theorem inequality_a_neg_one : ∀ x : ℝ, (x^2 + x - 2 > 0) ↔ (x < -2 ∨ x > 1) :=
by { sorry }

theorem inequality_general_a : 
∀ (a x : ℝ), ax^2 - (a + 2)*x + 2 < 0 ↔ 
  if a = 0 then x > 1
  else if a < 0 then x < (2 / a) ∨ x > 1
  else if 0 < a ∧ a < 2 then 1 < x ∧ x < (2 / a)
  else if a = 2 then False
  else (2 / a) < x ∧ x < 1 :=
by { sorry }

end NUMINAMATH_GPT_inequality_a_neg_one_inequality_general_a_l1838_183831


namespace NUMINAMATH_GPT_minimum_a_for_cube_in_tetrahedron_l1838_183804

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  (Real.sqrt 6 / 12) * a

theorem minimum_a_for_cube_in_tetrahedron (a : ℝ) (r : ℝ) 
  (h_radius : r = radius_of_circumscribed_sphere a)
  (h_diag : Real.sqrt 3 = 2 * r) :
  a = 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_a_for_cube_in_tetrahedron_l1838_183804


namespace NUMINAMATH_GPT_square_area_from_diagonal_l1838_183813

theorem square_area_from_diagonal :
  ∀ (d : ℝ), d = 10 * Real.sqrt 2 → (d / Real.sqrt 2) ^ 2 = 100 :=
by
  intros d hd
  sorry -- Skipping the proof

end NUMINAMATH_GPT_square_area_from_diagonal_l1838_183813
