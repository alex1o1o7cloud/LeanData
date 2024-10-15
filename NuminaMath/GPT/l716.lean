import Mathlib

namespace NUMINAMATH_GPT_books_from_second_shop_l716_71662

theorem books_from_second_shop (x : ℕ) (h₁ : 6500 + 2000 = 8500)
    (h₂ : 85 = 8500 / (65 + x)) : x = 35 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_books_from_second_shop_l716_71662


namespace NUMINAMATH_GPT_shopkeeper_gain_l716_71645

theorem shopkeeper_gain
  (true_weight : ℝ)
  (cheat_percent : ℝ)
  (gain_percent : ℝ) :
  cheat_percent = 0.1 ∧
  true_weight = 1000 →
  gain_percent = 20 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_gain_l716_71645


namespace NUMINAMATH_GPT_no_natural_number_solution_l716_71647

theorem no_natural_number_solution :
  ¬∃ (n : ℕ), ∃ (k : ℕ), (n^5 - 5*n^3 + 4*n + 7 = k^2) :=
sorry

end NUMINAMATH_GPT_no_natural_number_solution_l716_71647


namespace NUMINAMATH_GPT_correct_propositions_l716_71613

noncomputable def f : ℝ → ℝ := sorry

def proposition1 : Prop :=
  ∀ x : ℝ, f (1 + 2 * x) = f (1 - 2 * x) → ∀ x : ℝ, f (2 - x) = f x

def proposition2 : Prop :=
  ∀ x : ℝ, f (x - 2) = f (2 - x)

def proposition3 : Prop :=
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (2 + x) = -f x) → ∀ x : ℝ, f x = f (4 - x)

def proposition4 : Prop :=
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f x = f (-x - 2)) → ∀ x : ℝ, f (2 - x) = f x

theorem correct_propositions : proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4 :=
by sorry

end NUMINAMATH_GPT_correct_propositions_l716_71613


namespace NUMINAMATH_GPT_triangle_shape_isosceles_or_right_l716_71682

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to the angles

theorem triangle_shape_isosceles_or_right (h1 : a^2 + b^2 ≠ 0) (h2 : 
  (a^2 + b^2) * Real.sin (A - B) 
  = (a^2 - b^2) * Real.sin (A + B))
  (h3 : ∀ (A B C : ℝ), A + B + C = π) :
  ∃ (isosceles : Bool), (isosceles = true) ∨ (isosceles = false ∧ A + B = π / 2) :=
sorry

end NUMINAMATH_GPT_triangle_shape_isosceles_or_right_l716_71682


namespace NUMINAMATH_GPT_usual_time_is_60_l716_71617

variable (S T T' D : ℝ)

-- Defining the conditions
axiom condition1 : T' = T + 12
axiom condition2 : D = S * T
axiom condition3 : D = (5 / 6) * S * T'

-- The theorem to prove
theorem usual_time_is_60 (S T T' D : ℝ) 
  (h1 : T' = T + 12)
  (h2 : D = S * T)
  (h3 : D = (5 / 6) * S * T') : T = 60 := 
sorry

end NUMINAMATH_GPT_usual_time_is_60_l716_71617


namespace NUMINAMATH_GPT_notebook_cost_l716_71643

-- Define the cost of notebook (n) and cost of cover (c)
variables (n c : ℝ)

-- Given conditions as definitions
def condition1 := n + c = 3.50
def condition2 := n = c + 2

-- Prove that the cost of the notebook (n) is 2.75
theorem notebook_cost (h1 : condition1 n c) (h2 : condition2 n c) : n = 2.75 := 
by
  sorry

end NUMINAMATH_GPT_notebook_cost_l716_71643


namespace NUMINAMATH_GPT_parallel_line_through_point_l716_71637

theorem parallel_line_through_point (C : ℝ) :
  (∃ P : ℝ × ℝ, P.1 = 1 ∧ P.2 = 2) ∧ (∃ l : ℝ, ∀ x y : ℝ, 3 * x + y + l = 0) → 
  (3 * 1 + 2 + C = 0) → C = -5 :=
by
  sorry

end NUMINAMATH_GPT_parallel_line_through_point_l716_71637


namespace NUMINAMATH_GPT_percentage_increase_in_efficiency_l716_71642

def sEfficiency : ℚ := 1 / 20
def tEfficiency : ℚ := 1 / 16

theorem percentage_increase_in_efficiency :
    ((tEfficiency - sEfficiency) / sEfficiency) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_efficiency_l716_71642


namespace NUMINAMATH_GPT_sum_gcd_lcm_168_l716_71649

def gcd_54_72 : ℕ := Nat.gcd 54 72

def lcm_50_15 : ℕ := Nat.lcm 50 15

def sum_gcd_lcm : ℕ := gcd_54_72 + lcm_50_15

theorem sum_gcd_lcm_168 : sum_gcd_lcm = 168 := by
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_168_l716_71649


namespace NUMINAMATH_GPT_factorization1_factorization2_factorization3_factorization4_l716_71698

-- Question 1
theorem factorization1 (a b : ℝ) :
  4 * a^2 * b - 6 * a * b^2 = 2 * a * b * (2 * a - 3 * b) :=
by 
  sorry

-- Question 2
theorem factorization2 (x y : ℝ) :
  25 * x^2 - 9 * y^2 = (5 * x + 3 * y) * (5 * x - 3 * y) :=
by 
  sorry

-- Question 3
theorem factorization3 (a b : ℝ) :
  2 * a^2 * b - 8 * a * b^2 + 8 * b^3 = 2 * b * (a - 2 * b)^2 :=
by 
  sorry

-- Question 4
theorem factorization4 (x : ℝ) :
  (x + 2) * (x - 8) + 25 = (x - 3)^2 :=
by 
  sorry

end NUMINAMATH_GPT_factorization1_factorization2_factorization3_factorization4_l716_71698


namespace NUMINAMATH_GPT_max_value_inequality_l716_71680

theorem max_value_inequality (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  3 * x + 4 * y + 6 * z ≤ Real.sqrt 53 := by
  sorry

end NUMINAMATH_GPT_max_value_inequality_l716_71680


namespace NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_l716_71688

-- Define the first theorem
theorem solve_quadratic_1 (x : ℝ) : x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the second theorem
theorem solve_quadratic_2 (x : ℝ) : 25*x^2 - 36 = 0 ↔ x = 6/5 ∨ x = -6/5 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the third theorem
theorem solve_quadratic_3 (x : ℝ) : x^2 + 10*x + 21 = 0 ↔ x = -3 ∨ x = -7 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the fourth theorem
theorem solve_quadratic_4 (x : ℝ) : (x-3)^2 + 2*x*(x-3) = 0 ↔ x = 3 ∨ x = 1 := 
by {
  -- We assume this proof is provided
  sorry
}

end NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_l716_71688


namespace NUMINAMATH_GPT_max_value_of_symmetric_function_l716_71653

def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) (h_sym : ∀ x : ℝ, f x a b = f (-4 - x) a b) : 
  ∃ x : ℝ, (∀ y : ℝ, f y a b ≤ f x a b) ∧ f x a b = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_symmetric_function_l716_71653


namespace NUMINAMATH_GPT_determine_m_from_quadratic_l716_71661

def is_prime (n : ℕ) := 2 ≤ n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem determine_m_from_quadratic (x1 x2 m : ℕ) (hx1_prime : is_prime x1) (hx2_prime : is_prime x2) 
    (h_roots : x1 + x2 = 1999) (h_product : x1 * x2 = m) : 
    m = 3994 := 
by 
    sorry

end NUMINAMATH_GPT_determine_m_from_quadratic_l716_71661


namespace NUMINAMATH_GPT_mul_103_97_l716_71626

theorem mul_103_97 : 103 * 97 = 9991 := by
  sorry

end NUMINAMATH_GPT_mul_103_97_l716_71626


namespace NUMINAMATH_GPT_principal_calc_l716_71650

noncomputable def principal (r : ℝ) : ℝ :=
  (65000 : ℝ) / r

theorem principal_calc (P r : ℝ) (h : 0 < r) :
    (P * 0.10 + P * 1.10 * r / 100 - P * (0.10 + r / 100) = 65) → 
    P = principal r :=
by
  sorry

end NUMINAMATH_GPT_principal_calc_l716_71650


namespace NUMINAMATH_GPT_integer_representation_l716_71683

theorem integer_representation (n : ℤ) : ∃ x y z : ℤ, n = x^2 + y^2 - z^2 :=
by sorry

end NUMINAMATH_GPT_integer_representation_l716_71683


namespace NUMINAMATH_GPT_math_club_team_selection_l716_71616

noncomputable def choose (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.descFactorial n k / Nat.factorial k else 0

theorem math_club_team_selection :
  let boys := 10
  let girls := 12
  let team_size := 8
  let boys_selected := 4
  let girls_selected := 4
  choose boys boys_selected * choose girls girls_selected = 103950 := 
by simp [choose]; sorry

end NUMINAMATH_GPT_math_club_team_selection_l716_71616


namespace NUMINAMATH_GPT_total_students_l716_71699

theorem total_students (boys girls : ℕ) (h_ratio : boys / girls = 8 / 5) (h_girls : girls = 120) : boys + girls = 312 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l716_71699


namespace NUMINAMATH_GPT_second_discount_percentage_l716_71636

def normal_price : ℝ := 49.99
def first_discount : ℝ := 0.10
def final_price : ℝ := 36.0

theorem second_discount_percentage : 
  ∃ p : ℝ, (((normal_price - (first_discount * normal_price)) - final_price) / (normal_price - (first_discount * normal_price))) * 100 = p ∧ p = 20 :=
by
  sorry

end NUMINAMATH_GPT_second_discount_percentage_l716_71636


namespace NUMINAMATH_GPT_stratified_sampling_admin_staff_count_l716_71651

theorem stratified_sampling_admin_staff_count
  (total_staff : ℕ)
  (admin_staff : ℕ)
  (sample_size : ℕ)
  (h_total : total_staff = 160)
  (h_admin : admin_staff = 32)
  (h_sample : sample_size = 20) :
  admin_staff * sample_size / total_staff = 4 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_admin_staff_count_l716_71651


namespace NUMINAMATH_GPT_base_conversion_difference_l716_71660

-- Definitions
def base9_to_base10 (n : ℕ) : ℕ := 3 * (9^2) + 2 * (9^1) + 7 * (9^0)
def base8_to_base10 (m : ℕ) : ℕ := 2 * (8^2) + 5 * (8^1) + 3 * (8^0)

-- Statement
theorem base_conversion_difference :
  base9_to_base10 327 - base8_to_base10 253 = 97 :=
by sorry

end NUMINAMATH_GPT_base_conversion_difference_l716_71660


namespace NUMINAMATH_GPT_ball_radius_and_surface_area_l716_71611

theorem ball_radius_and_surface_area (d h : ℝ) (r : ℝ) :
  d = 12 ∧ h = 2 ∧ (6^2 + (r - h)^2 = r^2) → (r = 10 ∧ 4 * Real.pi * r^2 = 400 * Real.pi) := by
  sorry

end NUMINAMATH_GPT_ball_radius_and_surface_area_l716_71611


namespace NUMINAMATH_GPT_distinct_sum_of_five_integers_l716_71677

theorem distinct_sum_of_five_integers 
  (a b c d e : ℤ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) 
  (h_condition : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = -120) : 
  a + b + c + d + e = 25 :=
sorry

end NUMINAMATH_GPT_distinct_sum_of_five_integers_l716_71677


namespace NUMINAMATH_GPT_total_trees_in_gray_regions_l716_71641

theorem total_trees_in_gray_regions (trees_rectangle1 trees_rectangle2 trees_rectangle3 trees_gray1 trees_gray2 trees_total : ℕ)
  (h1 : trees_rectangle1 = 100)
  (h2 : trees_rectangle2 = 90)
  (h3 : trees_rectangle3 = 82)
  (h4 : trees_total = 82)
  (h_gray1 : trees_gray1 = trees_rectangle1 - trees_total)
  (h_gray2 : trees_gray2 = trees_rectangle2 - trees_total)
  : trees_gray1 + trees_gray2 = 26 := 
sorry

end NUMINAMATH_GPT_total_trees_in_gray_regions_l716_71641


namespace NUMINAMATH_GPT_find_difference_of_roots_l716_71664

-- Define the conditions for the given problem
def larger_root_of_eq_1 (a : ℝ) : Prop :=
  (1998 * a) ^ 2 - 1997 * 1999 * a - 1 = 0

def smaller_root_of_eq_2 (b : ℝ) : Prop :=
  b ^ 2 + 1998 * b - 1999 = 0

-- Define the main problem with the proof obligation
theorem find_difference_of_roots (a b : ℝ) (h1: larger_root_of_eq_1 a) (h2: smaller_root_of_eq_2 b) : a - b = 2000 :=
sorry

end NUMINAMATH_GPT_find_difference_of_roots_l716_71664


namespace NUMINAMATH_GPT_infinitely_many_odd_n_composite_l716_71634

theorem infinitely_many_odd_n_composite (n : ℕ) (h_odd : n % 2 = 1) : 
  ∃ (n : ℕ) (h_odd : n % 2 = 1), 
     ∀ k : ℕ, ∃ (m : ℕ) (h_odd_m : m % 2 = 1), 
     (∃ (d : ℕ), d ∣ (2^m + m) ∧ (1 < d ∧ d < 2^m + m))
:=
sorry

end NUMINAMATH_GPT_infinitely_many_odd_n_composite_l716_71634


namespace NUMINAMATH_GPT_max_area_of_triangle_l716_71656

theorem max_area_of_triangle (AB BC AC : ℝ) (ratio : BC / AC = 3 / 5) (hAB : AB = 10) :
  ∃ A : ℝ, (A ≤ 260.52) :=
sorry

end NUMINAMATH_GPT_max_area_of_triangle_l716_71656


namespace NUMINAMATH_GPT_seamless_assembly_with_equilateral_triangle_l716_71612

theorem seamless_assembly_with_equilateral_triangle :
  ∃ (polygon : ℕ → ℝ) (angle_150 : ℝ),
    (polygon 4 = 90) ∧ (polygon 6 = 120) ∧ (polygon 8 = 135) ∧ (polygon 3 = 60) ∧ (angle_150 = 150) ∧
    (∃ (n₁ n₂ n₃ : ℕ), n₁ * 150 + n₂ * 150 + n₃ * 60 = 360) :=
by {
  -- The proof would involve checking the precise integer combination for seamless assembly
  sorry
}

end NUMINAMATH_GPT_seamless_assembly_with_equilateral_triangle_l716_71612


namespace NUMINAMATH_GPT_geometric_seq_neither_necess_nor_suff_l716_71606

theorem geometric_seq_neither_necess_nor_suff (a_1 q : ℝ) (h₁ : a_1 ≠ 0) (h₂ : q ≠ 0) :
  ¬ (∀ n : ℕ, (a_1 * q > 0 → a_1 * q ^ n < a_1 * q ^ (n + 1)) ∧ (∀ n : ℕ, (a_1 * q ^ n < a_1 * q ^ (n + 1)) → a_1 * q > 0)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_neither_necess_nor_suff_l716_71606


namespace NUMINAMATH_GPT_find_k_l716_71619

def f (a b c x : ℤ) : ℤ := a * x * x + b * x + c

theorem find_k : 
  ∃ k : ℤ, 
    ∃ a b c : ℤ, 
      f a b c 1 = 0 ∧
      60 < f a b c 6 ∧ f a b c 6 < 70 ∧
      120 < f a b c 9 ∧ f a b c 9 < 130 ∧
      10000 * k < f a b c 200 ∧ f a b c 200 < 10000 * (k + 1)
      ∧ k = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l716_71619


namespace NUMINAMATH_GPT_tangent_position_is_six_oclock_l716_71635

-- Define constants and initial conditions
def bigRadius : ℝ := 30
def smallRadius : ℝ := 15
def initialPosition := 12 -- 12 o'clock represented as initial tangent position
def initialArrowDirection := 0 -- upwards direction

-- Define that the small disk rolls counterclockwise around the clock face.
def rollsCCW := true

-- Define the destination position when the arrow next points upward.
def diskTangencyPosition (bR sR : ℝ) (initPos initDir : ℕ) (rolls : Bool) : ℕ :=
  if rolls then 6 else 12

theorem tangent_position_is_six_oclock :
  diskTangencyPosition bigRadius smallRadius initialPosition initialArrowDirection rollsCCW = 6 :=
sorry  -- the proof is omitted

end NUMINAMATH_GPT_tangent_position_is_six_oclock_l716_71635


namespace NUMINAMATH_GPT_arithmetic_sequence_9th_term_l716_71630

variables {a_n : ℕ → ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_9th_term
  (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 3 = 6)
  (h2 : a 6 = 3)
  (h_seq : arithmetic_sequence a d) :
  a 9 = 0 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_9th_term_l716_71630


namespace NUMINAMATH_GPT_cosine_sum_of_angles_l716_71601

theorem cosine_sum_of_angles (α β : ℝ) 
  (hα : Complex.exp (Complex.I * α) = (4 / 5) + (3 / 5) * Complex.I)
  (hβ : Complex.exp (Complex.I * β) = (-5 / 13) + (12 / 13) * Complex.I) :
  Real.cos (α + β) = -7 / 13 :=
by
  sorry

end NUMINAMATH_GPT_cosine_sum_of_angles_l716_71601


namespace NUMINAMATH_GPT_averageFishIs75_l716_71666

-- Introduce the number of fish in Boast Pool
def BoastPool : ℕ := 75

-- Introduce the number of fish in Onum Lake
def OnumLake : ℕ := BoastPool + 25

-- Introduce the number of fish in Riddle Pond
def RiddlePond : ℕ := OnumLake / 2

-- Define the total number of fish in all three bodies of water
def totalFish : ℕ := BoastPool + OnumLake + RiddlePond

-- Define the average number of fish in all three bodies of water
def averageFish : ℕ := totalFish / 3

-- Prove that the average number of fish is 75
theorem averageFishIs75 : averageFish = 75 :=
by
  -- We need to provide the proof steps here but using sorry to skip
  sorry

end NUMINAMATH_GPT_averageFishIs75_l716_71666


namespace NUMINAMATH_GPT_ellipse_foci_on_y_axis_l716_71657

theorem ellipse_foci_on_y_axis (theta : ℝ) (h1 : 0 < theta ∧ theta < π)
  (h2 : Real.sin theta + Real.cos theta = 1 / 2) :
  (0 < theta ∧ theta < π / 2) → 
  (0 < theta ∧ theta < 3 * π / 4) → 
  -- The equation x^2 * sin theta - y^2 * cos theta = 1 represents an ellipse with foci on the y-axis
  ∃ foci_on_y_axis : Prop, foci_on_y_axis := 
sorry

end NUMINAMATH_GPT_ellipse_foci_on_y_axis_l716_71657


namespace NUMINAMATH_GPT_fixed_point_of_line_l716_71648

theorem fixed_point_of_line (m : ℝ) : 
  (m - 2) * (-3) - 8 + 3 * m + 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_line_l716_71648


namespace NUMINAMATH_GPT_fraction_div_add_result_l716_71686

theorem fraction_div_add_result : 
  (2 / 3) / (4 / 5) + (1 / 2) = (4 / 3) := 
by 
  sorry

end NUMINAMATH_GPT_fraction_div_add_result_l716_71686


namespace NUMINAMATH_GPT_right_triangle_acute_angle_ratio_l716_71676

theorem right_triangle_acute_angle_ratio (A B : ℝ) (h_ratio : A / B = 5 / 4) (h_sum : A + B = 90) :
  min A B = 40 :=
by
  -- Conditions are provided
  sorry

end NUMINAMATH_GPT_right_triangle_acute_angle_ratio_l716_71676


namespace NUMINAMATH_GPT_log3_cubicroot_of_3_l716_71600

noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem log3_cubicroot_of_3 :
  log_base_3 (3 ^ (1/3 : ℝ)) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_log3_cubicroot_of_3_l716_71600


namespace NUMINAMATH_GPT_robins_hair_cut_l716_71691

theorem robins_hair_cut (x : ℕ) : 16 - x + 12 = 17 → x = 11 := by
  sorry

end NUMINAMATH_GPT_robins_hair_cut_l716_71691


namespace NUMINAMATH_GPT_find_fraction_l716_71693

noncomputable def condition_eq : ℝ := 5
noncomputable def condition_gq : ℝ := 7

theorem find_fraction {FQ HQ : ℝ} (h : condition_eq * FQ = condition_gq * HQ) :
  FQ / HQ = 7 / 5 :=
by
  have eq_mul : condition_eq = 5 := by rfl
  have gq_mul : condition_gq = 7 := by rfl
  rw [eq_mul, gq_mul] at h
  have h': 5 * FQ = 7 * HQ := h
  field_simp [←h']
  sorry

end NUMINAMATH_GPT_find_fraction_l716_71693


namespace NUMINAMATH_GPT_smaller_circle_x_coordinate_l716_71640

theorem smaller_circle_x_coordinate (h : ℝ) 
  (P : ℝ × ℝ) (S : ℝ × ℝ)
  (H1 : P = (9, 12))
  (H2 : S = (h, 0))
  (r_large : ℝ)
  (r_small : ℝ)
  (H3 : r_large = 15)
  (H4 : r_small = 10) :
  S.1 = 10 ∨ S.1 = -10 := 
sorry

end NUMINAMATH_GPT_smaller_circle_x_coordinate_l716_71640


namespace NUMINAMATH_GPT_polynomial_has_integer_root_l716_71618

noncomputable def P : Polynomial ℤ := sorry

theorem polynomial_has_integer_root
  (P : Polynomial ℤ)
  (h_deg : P.degree = 3)
  (h_infinite_sol : ∀ (x y : ℤ), x ≠ y → x * P.eval x = y * P.eval y → 
  ∃ (x y : ℤ), x ≠ y ∧ x * P.eval x = y * P.eval y) :
  ∃ k : ℤ, P.eval k = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_has_integer_root_l716_71618


namespace NUMINAMATH_GPT_geometric_sequence_problem_l716_71671

section 
variables (a : ℕ → ℝ) (r : ℝ) 

-- Condition: {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) := ∀ n : ℕ, a (n + 1) = a n * r

-- Condition: a_4 + a_6 = 8
axiom a4_a6_sum : a 4 + a 6 = 8

-- Mathematical equivalent proof problem
theorem geometric_sequence_problem (h : is_geometric_sequence a r) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 :=
sorry

end

end NUMINAMATH_GPT_geometric_sequence_problem_l716_71671


namespace NUMINAMATH_GPT_investment_amount_l716_71685

theorem investment_amount (R T V : ℝ) (hT : T = 0.9 * R) (hV : V = 0.99 * R) (total_sum : R + T + V = 6936) : R = 2400 :=
by sorry

end NUMINAMATH_GPT_investment_amount_l716_71685


namespace NUMINAMATH_GPT_minimum_value_f_is_correct_l716_71608

noncomputable def f (x : ℝ) := 
  Real.sqrt (15 - 12 * Real.cos x) + 
  Real.sqrt (4 - 2 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (7 - 4 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (10 - 4 * Real.sqrt 3 * Real.sin x - 6 * Real.cos x)

theorem minimum_value_f_is_correct :
  ∃ x : ℝ, f x = (9 / 2) * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_f_is_correct_l716_71608


namespace NUMINAMATH_GPT_conclusion_1_conclusion_2_conclusion_3_conclusion_4_l716_71624

variable (a : ℕ → ℝ)

-- Conditions
def sequence_positive : Prop :=
  ∀ n, a n > 0

def recurrence_relation : Prop :=
  ∀ n, a (n + 1) ^ 2 - a (n + 1) = a n

-- Correct conclusions to prove:

-- Conclusion ①
theorem conclusion_1 (h1 : sequence_positive a) (h2 : recurrence_relation a) :
  ∀ n ≥ 2, a n > 1 := 
sorry

-- Conclusion ②
theorem conclusion_2 (h1 : sequence_positive a) (h2 : recurrence_relation a) :
  ¬∀ n, a n = a (n + 1) := 
sorry

-- Conclusion ③
theorem conclusion_3 (h1 : sequence_positive a) (h2 : recurrence_relation a) (h3 : 0 < a 1 ∧ a 1 < 2) :
  ∀ n, a (n + 1) > a n :=
sorry

-- Conclusion ④
theorem conclusion_4 (h1 : sequence_positive a) (h2 : recurrence_relation a) (h4 : a 1 > 2) :
  ∀ n ≥ 2, 2 < a n ∧ a n < a 1 :=
sorry

end NUMINAMATH_GPT_conclusion_1_conclusion_2_conclusion_3_conclusion_4_l716_71624


namespace NUMINAMATH_GPT_number_of_new_terms_l716_71695

theorem number_of_new_terms (n : ℕ) (h : n > 1) :
  (2^(n+1) - 1) - (2^n - 1) + 1 = 2^n := by
sorry

end NUMINAMATH_GPT_number_of_new_terms_l716_71695


namespace NUMINAMATH_GPT_apple_baskets_l716_71658

theorem apple_baskets (total_apples : ℕ) (apples_per_basket : ℕ) (total_apples_eq : total_apples = 495) (apples_per_basket_eq : apples_per_basket = 25) :
  total_apples / apples_per_basket = 19 :=
by
  sorry

end NUMINAMATH_GPT_apple_baskets_l716_71658


namespace NUMINAMATH_GPT_coefficient_x_squared_l716_71602

variable {a w c d : ℝ}

/-- The coefficient of x^2 in the expanded form of the equation (ax + w)(cx + d) = 6x^2 + x - 12 -/
theorem coefficient_x_squared (h1 : (a * x + w) * (c * x + d) = 6 * x^2 + x - 12)
                             (h2 : abs a + abs w + abs c + abs d = 12) :
  a * c = 6 :=
  sorry

end NUMINAMATH_GPT_coefficient_x_squared_l716_71602


namespace NUMINAMATH_GPT_systematic_sampling_method_l716_71668

-- Define the problem conditions
def total_rows : Nat := 40
def seats_per_row : Nat := 25
def attendees_left (row : Nat) : Nat := if row < total_rows then 18 else 0

-- Problem statement to be proved: The method used is systematic sampling.
theorem systematic_sampling_method :
  (∀ r : Nat, r < total_rows → attendees_left r = 18) →
  (seats_per_row = 25) →
  (∃ k, k > 0 ∧ ∀ r, r < total_rows → attendees_left r = 18 + k * r) →
  True :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_systematic_sampling_method_l716_71668


namespace NUMINAMATH_GPT_find_cost_expensive_module_l716_71633

-- Defining the conditions
def cost_cheaper_module : ℝ := 2.5
def total_modules : ℕ := 22
def num_cheaper_modules : ℕ := 21
def total_stock_value : ℝ := 62.5

-- The goal is to find the cost of the more expensive module 
def cost_expensive_module (cost_expensive_module : ℝ) : Prop :=
  num_cheaper_modules * cost_cheaper_module + cost_expensive_module = total_stock_value

-- The mathematically equivalent proof problem
theorem find_cost_expensive_module : cost_expensive_module 10 :=
by
  unfold cost_expensive_module
  norm_num
  sorry

end NUMINAMATH_GPT_find_cost_expensive_module_l716_71633


namespace NUMINAMATH_GPT_fraction_equality_l716_71652

theorem fraction_equality (x y a b : ℝ) (hx : x / y = 3) (h : (2 * a - x) / (3 * b - y) = 3) : a / b = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l716_71652


namespace NUMINAMATH_GPT_product_of_equal_numbers_l716_71632

theorem product_of_equal_numbers (a b c d : ℕ) (h_mean : (a + b + c + d) / 4 = 20) (h_known1 : a = 12) (h_known2 : b = 22) (h_equal : c = d) : c * d = 529 :=
by
  sorry

end NUMINAMATH_GPT_product_of_equal_numbers_l716_71632


namespace NUMINAMATH_GPT_weight_of_B_l716_71615

variable (W_A W_B W_C W_D : ℝ)

theorem weight_of_B (h1 : (W_A + W_B + W_C + W_D) / 4 = 60)
                    (h2 : (W_A + W_B) / 2 = 55)
                    (h3 : (W_B + W_C) / 2 = 50)
                    (h4 : (W_C + W_D) / 2 = 65) :
                    W_B = 50 :=
by sorry

end NUMINAMATH_GPT_weight_of_B_l716_71615


namespace NUMINAMATH_GPT_big_white_toys_l716_71623

/-- A store has two types of toys, Big White and Little Yellow, with a total of 60 toys.
    The price ratio of Big White to Little Yellow is 6:5.
    Selling all of them results in a total of 2016 yuan.
    We want to determine how many Big Whites there are. -/
theorem big_white_toys (x k : ℕ) (h1 : 6 * x + 5 * (60 - x) = 2016) (h2 : k = 6) : x = 36 :=
by
  sorry

end NUMINAMATH_GPT_big_white_toys_l716_71623


namespace NUMINAMATH_GPT_factorize_correct_l716_71638
noncomputable def factorize_expression (a b : ℝ) : ℝ :=
  (a - b)^4 + (a + b)^4 + (a + b)^2 * (a - b)^2

theorem factorize_correct (a b : ℝ) :
  factorize_expression a b = (3 * a^2 + b^2) * (a^2 + 3 * b^2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_correct_l716_71638


namespace NUMINAMATH_GPT_alfonzo_visit_l716_71644

-- Define the number of princes (palaces) as n
variable (n : ℕ)

-- Define the type of connections (either a "Ruelle" or a "Canal")
inductive Transport
| Ruelle
| Canal

-- Define the connection between any two palaces
noncomputable def connection (i j : ℕ) : Transport := sorry

-- The theorem states that Prince Alfonzo can visit all his friends using only one type of transportation
theorem alfonzo_visit (h : ∀ i j, i ≠ j → ∃ t : Transport, ∀ k, k ≠ i → connection i k = t) :
  ∃ t : Transport, ∀ i j, i ≠ j → connection i j = t :=
sorry

end NUMINAMATH_GPT_alfonzo_visit_l716_71644


namespace NUMINAMATH_GPT_Pooja_speed_3_l716_71684

variable (Roja_speed Pooja_speed : ℝ)
variable (t d : ℝ)

theorem Pooja_speed_3
  (h1 : Roja_speed = 6)
  (h2 : t = 4)
  (h3 : d = 36)
  (h4 : d = t * (Roja_speed + Pooja_speed)) :
  Pooja_speed = 3 :=
by
  sorry

end NUMINAMATH_GPT_Pooja_speed_3_l716_71684


namespace NUMINAMATH_GPT_log_expression_eq_l716_71679

theorem log_expression_eq (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x^2 / Real.log (y^4)) * 
  (Real.log (y^3) / Real.log (x^6)) * 
  (Real.log (x^4) / Real.log (y^3)) * 
  (Real.log (y^4) / Real.log (x^2)) * 
  (Real.log (x^6) / Real.log y) = 
  16 * Real.log x / Real.log y := 
sorry

end NUMINAMATH_GPT_log_expression_eq_l716_71679


namespace NUMINAMATH_GPT_cake_sector_chord_length_l716_71667

noncomputable def sector_longest_chord_square (d : ℝ) (n : ℕ) : ℝ :=
  let r := d / 2
  let theta := (360 : ℝ) / n
  let chord_length := 2 * r * Real.sin (theta / 2 * Real.pi / 180)
  chord_length ^ 2

theorem cake_sector_chord_length :
  sector_longest_chord_square 18 5 = 111.9473 := by
  sorry

end NUMINAMATH_GPT_cake_sector_chord_length_l716_71667


namespace NUMINAMATH_GPT_sum_x_y_z_l716_71605

noncomputable def a : ℝ := -Real.sqrt (9/27)
noncomputable def b : ℝ := Real.sqrt ((3 + Real.sqrt 7)^2 / 9)

theorem sum_x_y_z (ha : a = -Real.sqrt (9 / 27)) (hb : b = Real.sqrt ((3 + Real.sqrt 7) ^ 2 / 9)) (h_neg_a : a < 0) (h_pos_b : b > 0) :
  ∃ x y z : ℕ, (a + b)^3 = (x * Real.sqrt y) / z ∧ x + y + z = 718 := 
sorry

end NUMINAMATH_GPT_sum_x_y_z_l716_71605


namespace NUMINAMATH_GPT_range_of_c_for_two_distinct_roots_l716_71673

theorem range_of_c_for_two_distinct_roots (c : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 3 * x1 + c = x1 + 2) ∧ (x2^2 - 3 * x2 + c = x2 + 2)) ↔ (c < 6) :=
sorry

end NUMINAMATH_GPT_range_of_c_for_two_distinct_roots_l716_71673


namespace NUMINAMATH_GPT_elderly_people_not_set_l716_71665

def is_well_defined (S : Set α) : Prop := Nonempty S

def all_positive_numbers : Set ℝ := {x : ℝ | 0 < x}
def real_numbers_non_zero : Set ℝ := {x : ℝ | x ≠ 0}
def four_great_inventions : Set String := {"compass", "gunpowder", "papermaking", "printing"}

def elderly_people_description : String := "elderly people"

theorem elderly_people_not_set :
  ¬ (∃ S : Set α, elderly_people_description = "elderly people" ∧ is_well_defined S) :=
sorry

end NUMINAMATH_GPT_elderly_people_not_set_l716_71665


namespace NUMINAMATH_GPT_outfit_choices_l716_71663

noncomputable def calculate_outfits : Nat :=
  let shirts := 6
  let pants := 6
  let hats := 6
  let total_outfits := shirts * pants * hats
  let matching_colors := 4 -- tan, black, blue, gray for matching
  total_outfits - matching_colors

theorem outfit_choices : calculate_outfits = 212 :=
by
  sorry

end NUMINAMATH_GPT_outfit_choices_l716_71663


namespace NUMINAMATH_GPT_area_of_original_square_l716_71631

theorem area_of_original_square 
  (x : ℝ) 
  (h0 : x * (x - 3) = 40) 
  (h1 : 0 < x) : 
  x ^ 2 = 64 := 
sorry

end NUMINAMATH_GPT_area_of_original_square_l716_71631


namespace NUMINAMATH_GPT_option_A_correct_l716_71692

theorem option_A_correct (p : ℕ) (h1 : p > 1) (h2 : p % 2 = 1) : 
  (p - 1)^(p/2 - 1) - 1 ≡ 0 [MOD (p - 2)] :=
sorry

end NUMINAMATH_GPT_option_A_correct_l716_71692


namespace NUMINAMATH_GPT_log_x2y2_l716_71674

theorem log_x2y2 (x y : ℝ) (h1 : Real.log (x^2 * y^5) = 2) (h2 : Real.log (x^3 * y^2) = 2) :
  Real.log (x^2 * y^2) = 16 / 11 :=
by
  sorry

end NUMINAMATH_GPT_log_x2y2_l716_71674


namespace NUMINAMATH_GPT_find_f_1988_l716_71621

namespace FunctionalEquation

def f (n : ℕ) : ℕ :=
  sorry -- definition placeholder, since we only need the statement

axiom f_properties (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (f m + f n) = m + n

theorem find_f_1988 (h : ∀ n : ℕ, 0 < n → f n = n) : f 1988 = 1988 :=
  sorry

end FunctionalEquation

end NUMINAMATH_GPT_find_f_1988_l716_71621


namespace NUMINAMATH_GPT_simplify_fraction_l716_71604

theorem simplify_fraction : (180 : ℚ) / 1260 = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l716_71604


namespace NUMINAMATH_GPT_exponent_multiplication_l716_71654

theorem exponent_multiplication :
  (5^0.2 * 10^0.4 * 10^0.1 * 10^0.5 * 5^0.8) = 50 := by
  sorry

end NUMINAMATH_GPT_exponent_multiplication_l716_71654


namespace NUMINAMATH_GPT_students_enthusiasts_both_l716_71655

theorem students_enthusiasts_both {A B : Type} (class_size music_enthusiasts art_enthusiasts neither_enthusiasts enthusiasts_music_or_art : ℕ) 
(h_class_size : class_size = 50)
(h_music_enthusiasts : music_enthusiasts = 30) 
(h_art_enthusiasts : art_enthusiasts = 25)
(h_neither_enthusiasts : neither_enthusiasts = 4)
(h_enthusiasts_music_or_art : enthusiasts_music_or_art = class_size - neither_enthusiasts):
    (music_enthusiasts + art_enthusiasts - enthusiasts_music_or_art) = 9 := by
  sorry

end NUMINAMATH_GPT_students_enthusiasts_both_l716_71655


namespace NUMINAMATH_GPT_coefficient_x3_in_expansion_l716_71628

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_x3_in_expansion :
  let x := 1
  let a := x
  let b := 2
  let n := 50
  let k := 47
  let coefficient := binom n (n - k) * b^k
  coefficient = 19600 * 2^47 := by
  sorry

end NUMINAMATH_GPT_coefficient_x3_in_expansion_l716_71628


namespace NUMINAMATH_GPT_construct_angle_from_19_l716_71694

theorem construct_angle_from_19 (θ : ℝ) (h : θ = 19) : ∃ n : ℕ, (n * θ) % 360 = 75 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_construct_angle_from_19_l716_71694


namespace NUMINAMATH_GPT_average_letters_per_day_l716_71696

theorem average_letters_per_day (letters_tuesday : Nat) (letters_wednesday : Nat) (total_days : Nat) 
  (h_tuesday : letters_tuesday = 7) (h_wednesday : letters_wednesday = 3) (h_days : total_days = 2) : 
  (letters_tuesday + letters_wednesday) / total_days = 5 :=
by 
  sorry

end NUMINAMATH_GPT_average_letters_per_day_l716_71696


namespace NUMINAMATH_GPT_methane_hydrate_scientific_notation_l716_71690

theorem methane_hydrate_scientific_notation :
  (9.2 * 10^(-4)) = 0.00092 :=
by sorry

end NUMINAMATH_GPT_methane_hydrate_scientific_notation_l716_71690


namespace NUMINAMATH_GPT_alyssa_final_money_l716_71687

-- Definitions based on conditions
def weekly_allowance : Int := 8
def spent_on_movies : Int := weekly_allowance / 2
def earnings_from_washing_car : Int := 8

-- The statement to prove
def final_amount : Int := (weekly_allowance - spent_on_movies) + earnings_from_washing_car

-- The theorem expressing the problem
theorem alyssa_final_money : final_amount = 12 := by
  sorry

end NUMINAMATH_GPT_alyssa_final_money_l716_71687


namespace NUMINAMATH_GPT_muffin_cost_relation_l716_71646

variable (m b : ℝ)

variable (S := 5 * m + 4 * b)
variable (C := 10 * m + 18 * b)

theorem muffin_cost_relation (h1 : C = 3 * S) : m = 1.2 * b :=
  sorry

end NUMINAMATH_GPT_muffin_cost_relation_l716_71646


namespace NUMINAMATH_GPT_sequence_sum_property_l716_71625

theorem sequence_sum_property {a S : ℕ → ℚ} (h1 : a 1 = 3/2)
  (h2 : ∀ n : ℕ, 2 * a (n + 1) + S n = 3) :
  (∀ n : ℕ, a n = 3 * (1/2)^n) ∧
  (∃ (n_max : ℕ),  (∀ n : ℕ, n ≤ n_max → (S n = 3 * (1 - (1/2)^n)) ∧ ∀ n : ℕ, (S (2 * n)) / (S n) > 64 / 63 → n_max = 5)) :=
by {
  -- The proof would go here
  sorry
}

end NUMINAMATH_GPT_sequence_sum_property_l716_71625


namespace NUMINAMATH_GPT_minimum_value_is_two_sqrt_two_l716_71603

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (Real.sqrt (x^2 + (2 - x)^2)) + (Real.sqrt ((2 - x)^2 + x^2))

theorem minimum_value_is_two_sqrt_two :
  ∃ x : ℝ, minimum_value_expression x = 2 * Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_value_is_two_sqrt_two_l716_71603


namespace NUMINAMATH_GPT_totalBalls_l716_71610

def jungkookBalls : Nat := 3
def yoongiBalls : Nat := 2

theorem totalBalls : jungkookBalls + yoongiBalls = 5 := by
  sorry

end NUMINAMATH_GPT_totalBalls_l716_71610


namespace NUMINAMATH_GPT_caterpillar_to_scorpion_ratio_l716_71609

theorem caterpillar_to_scorpion_ratio 
  (roach_count : ℕ) (scorpion_count : ℕ) (total_insects : ℕ) 
  (h_roach : roach_count = 12) 
  (h_scorpion : scorpion_count = 3) 
  (h_cricket : cricket_count = roach_count / 2) 
  (h_total : total_insects = 27) 
  (h_non_cricket_count : non_cricket_count = roach_count + scorpion_count + cricket_count) 
  (h_caterpillar_count : caterpillar_count = total_insects - non_cricket_count) : 
  (caterpillar_count / scorpion_count) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_caterpillar_to_scorpion_ratio_l716_71609


namespace NUMINAMATH_GPT_gcd_of_three_numbers_l716_71672

theorem gcd_of_three_numbers :
  Nat.gcd (Nat.gcd 72 120) 168 = 24 :=
sorry

end NUMINAMATH_GPT_gcd_of_three_numbers_l716_71672


namespace NUMINAMATH_GPT_greatest_integer_value_x_l716_71678

theorem greatest_integer_value_x :
  ∃ x : ℤ, (8 - 3 * (2 * x + 1) > 26) ∧ ∀ y : ℤ, (8 - 3 * (2 * y + 1) > 26) → y ≤ x :=
sorry

end NUMINAMATH_GPT_greatest_integer_value_x_l716_71678


namespace NUMINAMATH_GPT_total_horse_food_l716_71669

theorem total_horse_food (ratio_sh_to_h : ℕ → ℕ → Prop) 
    (sheep : ℕ) 
    (ounce_per_horse : ℕ) 
    (total_ounces_per_day : ℕ) : 
    ratio_sh_to_h 5 7 → sheep = 40 → ounce_per_horse = 230 → total_ounces_per_day = 12880 :=
by
  intros h_ratio h_sheep h_ounce
  sorry

end NUMINAMATH_GPT_total_horse_food_l716_71669


namespace NUMINAMATH_GPT_plane_equation_through_point_and_line_l716_71675

theorem plane_equation_through_point_and_line :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd A B = 1 ∧ Int.gcd A C = 1 ∧ Int.gcd A D = 1 ∧
  ∀ (x y z : ℝ),
    (A * x + B * y + C * z + D = 0 ↔ 
    (∃ (t : ℝ), x = -3 * t - 1 ∧ y = 2 * t + 3 ∧ z = t - 2) ∨ 
    (x = 0 ∧ y = 7 ∧ z = -7)) :=
by
  -- sorry, implementing proofs is not required.
  sorry

end NUMINAMATH_GPT_plane_equation_through_point_and_line_l716_71675


namespace NUMINAMATH_GPT_tetrahedron_volume_minimum_l716_71622

theorem tetrahedron_volume_minimum (h1 h2 h3 : ℝ) (h1_pos : 0 < h1) (h2_pos : 0 < h2) (h3_pos : 0 < h3) :
  ∃ V : ℝ, V ≥ (1/3) * (h1 * h2 * h3) :=
sorry

end NUMINAMATH_GPT_tetrahedron_volume_minimum_l716_71622


namespace NUMINAMATH_GPT_dice_probability_l716_71681

noncomputable def probability_event (event_count : ℕ) (total_count : ℕ) : ℚ := 
  event_count / total_count

theorem dice_probability :
  let event_first_die := 3
  let event_second_die := 3
  let total_outcomes_first := 8
  let total_outcomes_second := 8
  probability_event event_first_die total_outcomes_first * probability_event event_second_die total_outcomes_second = 9 / 64 :=
by
  sorry

end NUMINAMATH_GPT_dice_probability_l716_71681


namespace NUMINAMATH_GPT_region_relation_l716_71697

theorem region_relation (A B C : ℝ)
  (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39)
  (h_triangle : a^2 + b^2 = c^2)
  (h_right_triangle : true) -- Since the triangle is already confirmed as right-angle
  (h_A : A = (π * (c / 2)^2 / 2 - 270) / 2)
  (h_B : B = (π * (c / 2)^2 / 2 - 270) / 2)
  (h_C : C = π * (c / 2)^2 / 2) :
  A + B + 270 = C :=
by
  sorry

end NUMINAMATH_GPT_region_relation_l716_71697


namespace NUMINAMATH_GPT_ben_eggs_remaining_l716_71614

def initial_eggs : ℕ := 75

def ben_day1_morning : ℝ := 5
def ben_day1_afternoon : ℝ := 4.5
def alice_day1_morning : ℝ := 3.5
def alice_day1_evening : ℝ := 4

def ben_day2_morning : ℝ := 7
def ben_day2_evening : ℝ := 3
def alice_day2_morning : ℝ := 2
def alice_day2_afternoon : ℝ := 4.5
def alice_day2_evening : ℝ := 1.5

def ben_day3_morning : ℝ := 4
def ben_day3_afternoon : ℝ := 3.5
def alice_day3_evening : ℝ := 6.5

def total_eggs_eaten : ℝ :=
  (ben_day1_morning + ben_day1_afternoon + alice_day1_morning + alice_day1_evening) +
  (ben_day2_morning + ben_day2_evening + alice_day2_morning + alice_day2_afternoon + alice_day2_evening) +
  (ben_day3_morning + ben_day3_afternoon + alice_day3_evening)

def remaining_eggs : ℝ :=
  initial_eggs - total_eggs_eaten

theorem ben_eggs_remaining : remaining_eggs = 26 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_ben_eggs_remaining_l716_71614


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_eq_1853_l716_71629

theorem sum_of_squares_of_roots_eq_1853
  (α β : ℕ) (h_prime_α : Prime α) (h_prime_beta : Prime β) (h_sum : α + β = 45)
  (h_quadratic_eq : ∀ x, x^2 - 45*x + α*β = 0 → x = α ∨ x = β) :
  α^2 + β^2 = 1853 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_eq_1853_l716_71629


namespace NUMINAMATH_GPT_time_to_produce_one_item_l716_71670

-- Definitions based on the conditions
def itemsProduced : Nat := 300
def totalTimeHours : ℝ := 2.0
def minutesPerHour : ℝ := 60.0

-- The statement we need to prove
theorem time_to_produce_one_item : (totalTimeHours / itemsProduced * minutesPerHour) = 0.4 := by
  sorry

end NUMINAMATH_GPT_time_to_produce_one_item_l716_71670


namespace NUMINAMATH_GPT_focal_distance_of_ellipse_l716_71689

theorem focal_distance_of_ellipse : 
  ∀ (θ : ℝ), (∃ (c : ℝ), (x = 5 * Real.cos θ ∧ y = 4 * Real.sin θ) → 2 * c = 6) :=
by
  sorry

end NUMINAMATH_GPT_focal_distance_of_ellipse_l716_71689


namespace NUMINAMATH_GPT_polynomial_remainder_l716_71639

-- Define the polynomial
def poly (x : ℝ) : ℝ := 3 * x^8 - x^7 - 7 * x^5 + 3 * x^3 + 4 * x^2 - 12 * x - 1

-- Define the divisor
def divisor : ℝ := 3

-- State the theorem
theorem polynomial_remainder :
  poly divisor = 15951 :=
by
  -- Proof omitted, to be filled in later
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l716_71639


namespace NUMINAMATH_GPT_problem_solution_l716_71627

noncomputable def set_M (x : ℝ) : Prop := x^2 - 4*x < 0
noncomputable def set_N (m x : ℝ) : Prop := m < x ∧ x < 5
noncomputable def set_intersection (x : ℝ) : Prop := 3 < x ∧ x < 4

theorem problem_solution (m n : ℝ) :
  (∀ x, set_M x ↔ (0 < x ∧ x < 4)) →
  (∀ x, set_N m x ↔ (m < x ∧ x < 5)) →
  (∀ x, (set_M x ∧ set_N m x) ↔ set_intersection x) →
  m + n = 7 :=
by
  intros H1 H2 H3
  sorry

end NUMINAMATH_GPT_problem_solution_l716_71627


namespace NUMINAMATH_GPT_min_value_f_when_a_eq_1_range_of_a_if_f_leq_3_non_empty_l716_71620

-- Condition 1: Define the function f(x)
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 3)

-- Proof Problem 1: Minimum value of f(x) when a = 1
theorem min_value_f_when_a_eq_1 : (∀ x : ℝ, f x 1 ≥ 2) :=
sorry

-- Proof Problem 2: Range of values for a when f(x) ≤ 3 has solutions
theorem range_of_a_if_f_leq_3_non_empty : 
  (∃ x : ℝ, f x a ≤ 3) → abs (3 - a) ≤ 3 :=
sorry

end NUMINAMATH_GPT_min_value_f_when_a_eq_1_range_of_a_if_f_leq_3_non_empty_l716_71620


namespace NUMINAMATH_GPT_product_of_fractions_l716_71607

theorem product_of_fractions :
  (1 / 3) * (3 / 5) * (5 / 7) = 1 / 7 :=
  sorry

end NUMINAMATH_GPT_product_of_fractions_l716_71607


namespace NUMINAMATH_GPT_total_blocks_fallen_l716_71659

def stack_height (n : Nat) : Nat :=
  if n = 1 then 7
  else if n = 2 then 7 + 5
  else if n = 3 then 7 + 5 + 7
  else 0

def blocks_standing (n : Nat) : Nat :=
  if n = 1 then 0
  else if n = 2 then 2
  else if n = 3 then 3
  else 0

def blocks_fallen (n : Nat) : Nat :=
  stack_height n - blocks_standing n

theorem total_blocks_fallen : blocks_fallen 1 + blocks_fallen 2 + blocks_fallen 3 = 33 :=
  by
    sorry

end NUMINAMATH_GPT_total_blocks_fallen_l716_71659
