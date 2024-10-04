import Mathlib

namespace fraction_correct_l198_198014

theorem fraction_correct (x : ℚ) (h : (5 / 6) * 576 = x * 576 + 300) : x = 5 / 16 := 
sorry

end fraction_correct_l198_198014


namespace least_number_to_add_l198_198078

theorem least_number_to_add (k : ℕ) (h : 1019 % 25 = 19) : (1019 + k) % 25 = 0 ↔ k = 6 :=
by
  sorry

end least_number_to_add_l198_198078


namespace find_side_DF_in_triangle_DEF_l198_198357

theorem find_side_DF_in_triangle_DEF
  (DE EF DM : ℝ)
  (h_DE : DE = 7)
  (h_EF : EF = 10)
  (h_DM : DM = 5) :
  ∃ DF : ℝ, DF = Real.sqrt 51 :=
by
  sorry

end find_side_DF_in_triangle_DEF_l198_198357


namespace sum_ABC_eq_7_base_8_l198_198157

/-- Lean 4 statement for the problem.

A, B, C: are distinct non-zero digits less than 8 in base 8, and
A B C_8 + B C_8 = A C A_8 holds true.
-/
theorem sum_ABC_eq_7_base_8 :
  ∃ (A B C : ℕ), A < 8 ∧ B < 8 ∧ C < 8 ∧ 
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  (A * 64 + B * 8 + C) + (B * 8 + C) = A * 64 + C * 8 + A ∧
  A + B + C = 7 :=
by { sorry }

end sum_ABC_eq_7_base_8_l198_198157


namespace find_x4_y4_z4_l198_198534

theorem find_x4_y4_z4
  (x y z : ℝ)
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 5)
  (h3 : x^3 + y^3 + z^3 = 7) :
  x^4 + y^4 + z^4 = 59 / 3 :=
by
  sorry

end find_x4_y4_z4_l198_198534


namespace identity_element_exists_identity_element_self_commutativity_associativity_l198_198036

noncomputable def star_op (a b : ℤ) : ℤ := a + b + a * b

theorem identity_element_exists : ∃ E : ℤ, ∀ a : ℤ, star_op a E = a :=
by sorry

theorem identity_element_self (E : ℤ) (h1 : ∀ a : ℤ, star_op a E = a) : star_op E E = E :=
by sorry

theorem commutativity (a b : ℤ) : star_op a b = star_op b a :=
by sorry

theorem associativity (a b c : ℤ) : star_op (star_op a b) c = star_op a (star_op b c) :=
by sorry

end identity_element_exists_identity_element_self_commutativity_associativity_l198_198036


namespace binomials_product_l198_198272

noncomputable def poly1 (x y : ℝ) : ℝ := 2 * x^2 + 3 * y - 4
noncomputable def poly2 (y : ℝ) : ℝ := y + 6

theorem binomials_product (x y : ℝ) :
  (poly1 x y) * (poly2 y) = 2 * x^2 * y + 12 * x^2 + 3 * y^2 + 14 * y - 24 :=
by sorry

end binomials_product_l198_198272


namespace circle_chairs_adjacent_l198_198425

theorem circle_chairs_adjacent : 
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)) 
    = 1634 :=
by
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    have h := Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)),
    have answer := 1634,
    sorry

end circle_chairs_adjacent_l198_198425


namespace adam_cat_food_vs_dog_food_l198_198094

def cat_packages := 15
def dog_packages := 10
def cans_per_cat_package := 12
def cans_per_dog_package := 8

theorem adam_cat_food_vs_dog_food:
  cat_packages * cans_per_cat_package - dog_packages * cans_per_dog_package = 100 :=
by
  sorry

end adam_cat_food_vs_dog_food_l198_198094


namespace rectangle_length_l198_198748

theorem rectangle_length (L W : ℝ) (h1 : L = 4 * W) (h2 : L * W = 100) : L = 20 :=
by
  sorry

end rectangle_length_l198_198748


namespace price_of_other_pieces_l198_198183

theorem price_of_other_pieces (total_spent : ℕ) (total_pieces : ℕ) (price_piece1 : ℕ) (price_piece2 : ℕ) 
  (remaining_pieces : ℕ) (price_remaining_piece : ℕ) (h1 : total_spent = 610) (h2 : total_pieces = 7)
  (h3 : price_piece1 = 49) (h4 : price_piece2 = 81) (h5 : remaining_pieces = (total_pieces - 2))
  (h6 : total_spent - price_piece1 - price_piece2 = remaining_pieces * price_remaining_piece) :
  price_remaining_piece = 96 := 
by
  sorry

end price_of_other_pieces_l198_198183


namespace graph_squares_count_l198_198923

theorem graph_squares_count :
  let x_intercept := 45
  let y_intercept := 5
  let total_squares := x_intercept * y_intercept
  let diagonal_squares := x_intercept + y_intercept - 1
  let non_diagonal_squares := total_squares - diagonal_squares
  non_diagonal_squares / 2 = 88 :=
by
  let x_intercept := 45
  let y_intercept := 5
  let total_squares := x_intercept * y_intercept
  let diagonal_squares := x_intercept + y_intercept - 1
  let non_diagonal_squares := total_squares - diagonal_squares
  have h : (non_diagonal_squares / 2 = 88) := sorry
  exact h

end graph_squares_count_l198_198923


namespace Allyson_age_is_28_l198_198238

-- Define the conditions of the problem
def Hiram_age : ℕ := 40
def add_12_to_Hiram_age (h_age : ℕ) : ℕ := h_age + 12
def twice_Allyson_age (a_age : ℕ) : ℕ := 2 * a_age
def condition (h_age : ℕ) (a_age : ℕ) : Prop := add_12_to_Hiram_age h_age = twice_Allyson_age a_age - 4

-- Define the theorem to be proven
theorem Allyson_age_is_28 (a_age : ℕ) (h_age : ℕ) (h_condition : condition h_age a_age) (h_hiram : h_age = Hiram_age) : a_age = 28 :=
by sorry

end Allyson_age_is_28_l198_198238


namespace p_sufficient_but_not_necessary_q_l198_198130

theorem p_sufficient_but_not_necessary_q :
  ∀ x : ℝ, (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros x hx
  cases hx with h1 h2
  apply And.intro
  apply lt_of_lt_of_le h1
  linarith
  apply h2

end p_sufficient_but_not_necessary_q_l198_198130


namespace expression_value_l198_198500

theorem expression_value (a b : ℕ) (h₁ : a = 2023) (h₂ : b = 2020) :
  ((
     (3 / (a - b) + (3 * a) / (a^3 - b^3) * ((a^2 + a * b + b^2) / (a + b))) * ((2 * a + b) / (a^2 + 2 * a * b + b^2))
  ) * (3 / (a + b))) = 3 :=
by
  -- Use the provided conditions
  rw [h₁, h₂]
  -- Execute the following steps as per the mathematical solution steps 
  sorry

end expression_value_l198_198500


namespace claudia_total_earnings_l198_198274

-- Definition of the problem conditions
def class_fee : ℕ := 10
def kids_saturday : ℕ := 20
def kids_sunday : ℕ := kids_saturday / 2

-- Theorem stating that Claudia makes $300.00 for the weekend
theorem claudia_total_earnings : (kids_saturday * class_fee) + (kids_sunday * class_fee) = 300 := 
by
  sorry

end claudia_total_earnings_l198_198274


namespace remainder_123456789012_div_252_l198_198487

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l198_198487


namespace max_elements_in_S_l198_198463

open Nat

theorem max_elements_in_S (S : Finset ℕ) 
  (h1 : ∀ a ∈ S, 0 < a ∧ a ≤ 100)
  (h2 : ∀ a b ∈ S, a ≠ b → ∃ c ∈ S, gcd a c = 1 ∧ gcd b c = 1)
  (h3 : ∀ a b ∈ S, a ≠ b → ∃ d ∈ S, d ≠ a ∧ d ≠ b ∧ gcd a d > 1 ∧ gcd b d > 1) :
  S.card ≤ 72 := 
sorry

end max_elements_in_S_l198_198463


namespace solve_textbook_by_12th_l198_198692

/-!
# Problem Statement
There are 91 problems in a textbook. Yura started solving them on September 6 and solves one problem less each subsequent morning. By the evening of September 8, there are 46 problems left to solve.

We need to prove that Yura finishes solving all the problems by September 12.
-/

def initial_problems : ℕ := 91
def problems_left_by_evening_of_8th : ℕ := 46

def problems_solved_by_evening_of_8th : ℕ :=
  initial_problems - problems_left_by_evening_of_8th

def z : ℕ := (problems_solved_by_evening_of_8th / 3)

theorem solve_textbook_by_12th 
    (total_problems : ℕ)
    (problems_left : ℕ)
    (solved_by_evening_8th : ℕ)
    (daily_problem_count : ℕ) :
    (total_problems = initial_problems) →
    (problems_left = problems_left_by_evening_of_8th) →
    (solved_by_evening_8th = problems_solved_by_evening_of_8th) →
    (daily_problem_count = z) →
    ∃ (finishing_date : ℕ), finishing_date = 12 :=
  by
    intros _ _ _ _
    sorry

end solve_textbook_by_12th_l198_198692


namespace smallest_sum_of_two_squares_l198_198603

theorem smallest_sum_of_two_squares :
  ∃ n : ℕ, (∀ m : ℕ, m < n → (¬ (∃ a b c d e f : ℕ, m = a^2 + b^2 ∧  m = c^2 + d^2 ∧ m = e^2 + f^2 ∧ ((a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f))))) ∧
          (∃ a b c d e f : ℕ, n = a^2 + b^2 ∧  n = c^2 + d^2 ∧ n = e^2 + f^2 ∧ ((a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f))) :=
sorry

end smallest_sum_of_two_squares_l198_198603


namespace apples_prepared_l198_198214

variables (n_x n_l : ℕ)

theorem apples_prepared (hx : 3 * n_x = 5 * n_l - 12) (hs : 6 * n_l = 72) : n_x = 12 := 
by sorry

end apples_prepared_l198_198214


namespace sean_and_julie_sums_l198_198386

theorem sean_and_julie_sums :
  let seanSum := 2 * (Finset.range 301).sum
  let julieSum := (Finset.range 301).sum
  seanSum / julieSum = 2 := by
  sorry

end sean_and_julie_sums_l198_198386


namespace find_blue_yarn_count_l198_198559

def scarves_per_yarn : ℕ := 3
def red_yarn_count : ℕ := 2
def yellow_yarn_count : ℕ := 4
def total_scarves : ℕ := 36

def scarves_from_red_and_yellow : ℕ :=
  red_yarn_count * scarves_per_yarn + yellow_yarn_count * scarves_per_yarn

def blue_scarves : ℕ :=
  total_scarves - scarves_from_red_and_yellow

def blue_yarn_count : ℕ :=
  blue_scarves / scarves_per_yarn

theorem find_blue_yarn_count :
  blue_yarn_count = 6 :=
by 
  sorry

end find_blue_yarn_count_l198_198559


namespace go_total_pieces_l198_198341

theorem go_total_pieces (T : ℕ) (h : T > 0) (prob_black : T = (3 : ℕ) * 4) : T = 12 := by
  sorry

end go_total_pieces_l198_198341


namespace subsets_with_at_least_three_adjacent_chairs_l198_198423

theorem subsets_with_at_least_three_adjacent_chairs :
  let n := 12
  in (number of subsets of circularly arranged chairs) ≥ 3 = 1634 :=
sorry

end subsets_with_at_least_three_adjacent_chairs_l198_198423


namespace remainder_123456789012_div_252_l198_198483

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l198_198483


namespace tangent_line_to_parabola_parallel_l198_198922

theorem tangent_line_to_parabola_parallel (m : ℝ) :
  ∀ (x y : ℝ), (y = x^2) → (2*x - y + m = 0 → m = -1) :=
by
  sorry

end tangent_line_to_parabola_parallel_l198_198922


namespace find_angle_C_l198_198361

theorem find_angle_C (A B C : ℝ)
  (h1 : 2 * Real.sin A + 3 * Real.cos B = 4)
  (h2 : 3 * Real.sin B + 2 * Real.cos A = Real.sqrt 3)
  (triangle_ABC : A + B + C = Real.pi) :
  C = Real.pi / 6 :=
begin
  sorry
end

end find_angle_C_l198_198361


namespace fraction_to_decimal_l198_198313

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198313


namespace find_income_l198_198265

-- Definitions of percentages used in calculations
def rent_percentage : ℝ := 0.15
def education_percentage : ℝ := 0.15
def misc_percentage : ℝ := 0.10
def medical_percentage : ℝ := 0.15

-- Remaining amount after all expenses
def final_amount : ℝ := 5548

-- Income calculation function
def calc_income (X : ℝ) : ℝ :=
  let after_rent := X * (1 - rent_percentage)
  let after_education := after_rent * (1 - education_percentage)
  let after_misc := after_education * (1 - misc_percentage)
  let after_medical := after_misc * (1 - medical_percentage)
  after_medical

-- Theorem statement to prove the woman's income
theorem find_income (X : ℝ) (h : calc_income X = final_amount) : X = 10038.46 := by
  sorry

end find_income_l198_198265


namespace rhombus_perimeter_l198_198047

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 40 :=
by
  sorry

end rhombus_perimeter_l198_198047


namespace equivalent_forms_l198_198340

-- Given line equation
def given_line_eq (x y : ℝ) : Prop :=
  (3 * x - 2) / 4 - (2 * y - 1) / 2 = 1

-- General form of the line
def general_form (x y : ℝ) : Prop :=
  3 * x - 8 * y - 2 = 0

-- Slope-intercept form of the line
def slope_intercept_form (x y : ℝ) : Prop := 
  y = (3 / 8) * x - 1 / 4

-- Intercept form of the line
def intercept_form (x y : ℝ) : Prop :=
  x / (2 / 3) + y / (-1 / 4) = 1

-- Normal form of the line
def normal_form (x y : ℝ) : Prop :=
  3 / Real.sqrt 73 * x - 8 / Real.sqrt 73 * y - 2 / Real.sqrt 73 = 0

-- Proof problem: Prove that the given line equation is equivalent to the derived forms
theorem equivalent_forms (x y : ℝ) :
  given_line_eq x y ↔ (general_form x y ∧ slope_intercept_form x y ∧ intercept_form x y ∧ normal_form x y) :=
sorry

end equivalent_forms_l198_198340


namespace greatest_root_of_f_one_is_root_of_f_l198_198322

def f (x : ℝ) : ℝ := 16 * x^6 - 15 * x^4 + 4 * x^2 - 1

theorem greatest_root_of_f :
  ∀ x : ℝ, f x = 0 → x ≤ 1 :=
sorry

theorem one_is_root_of_f :
  f 1 = 0 :=
sorry

end greatest_root_of_f_one_is_root_of_f_l198_198322


namespace least_positive_integer_divisible_by_four_distinct_primes_l198_198068

def is_prime (n : ℕ) : Prop := Nat.Prime n

def distinct_primes_product (p1 p2 p3 p4 : ℕ) : ℕ :=
  if h1 : is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 then
    p1 * p2 * p3 * p4
  else
    0

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n, n > 0 ∧ (∃ p1 p2 p3 p4, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ n = distinct_primes_product p1 p2 p3 p4) ∧ n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_distinct_primes_l198_198068


namespace max_value_of_XYZ_XY_YZ_ZX_l198_198890

theorem max_value_of_XYZ_XY_YZ_ZX (X Y Z : ℕ) (h : X + Y + Z = 15) : 
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 200 := 
sorry

end max_value_of_XYZ_XY_YZ_ZX_l198_198890


namespace find_e_m_l198_198148

variable {R : Type} [Field R]

def matrix_B (e : R) : Matrix (Fin 2) (Fin 2) R :=
  !![3, 4; 6, e]

theorem find_e_m (e m : R) (hB_inv : (matrix_B e)⁻¹ = m • (matrix_B e)) :
  e = -3 ∧ m = (1 / 11) := by
  sorry

end find_e_m_l198_198148


namespace average_sleep_hours_l198_198153

theorem average_sleep_hours (h_monday: ℕ) (h_tuesday: ℕ) (h_wednesday: ℕ) (h_thursday: ℕ) (h_friday: ℕ)
  (h_monday_eq: h_monday = 8) (h_tuesday_eq: h_tuesday = 7) (h_wednesday_eq: h_wednesday = 8)
  (h_thursday_eq: h_thursday = 10) (h_friday_eq: h_friday = 7) :
  (h_monday + h_tuesday + h_wednesday + h_thursday + h_friday) / 5 = 8 :=
by
  sorry

end average_sleep_hours_l198_198153


namespace halogens_have_solid_liquid_gas_l198_198776

def at_25C_and_1atm (element : String) : String :=
  match element with
  | "Li" | "Na" | "K" | "Rb" | "Cs" => "solid"
  | "N" => "gas"
  | "P" | "As" | "Sb" | "Bi" => "solid"
  | "O" => "gas"
  | "S" | "Se" | "Te" => "solid"
  | "F" | "Cl" => "gas"
  | "Br" => "liquid"
  | "I" | "At" => "solid"
  | _ => "unknown"

def family_has_solid_liquid_gas (family : List String) : Prop :=
  "solid" ∈ family.map at_25C_and_1atm ∧
  "liquid" ∈ family.map at_25C_and_1atm ∧
  "gas" ∈ family.map at_25C_and_1atm

theorem halogens_have_solid_liquid_gas :
  family_has_solid_liquid_gas ["F", "Cl", "Br", "I", "At"] :=
by
  sorry

end halogens_have_solid_liquid_gas_l198_198776


namespace ice_cream_depth_l198_198965

theorem ice_cream_depth 
  (r_sphere : ℝ) 
  (r_cylinder : ℝ) 
  (h_cylinder : ℝ) 
  (V_sphere : ℝ) 
  (V_cylinder : ℝ) 
  (constant_density : V_sphere = V_cylinder)
  (r_sphere_eq : r_sphere = 2) 
  (r_cylinder_eq : r_cylinder = 8) 
  (V_sphere_def : V_sphere = (4 / 3) * Real.pi * r_sphere^3) 
  (V_cylinder_def : V_cylinder = Real.pi * r_cylinder^2 * h_cylinder) 
  : h_cylinder = 1 / 6 := 
by 
  sorry

end ice_cream_depth_l198_198965


namespace original_wage_before_increase_l198_198619

theorem original_wage_before_increase (W : ℝ) 
  (h1 : W * 1.4 = 35) : W = 25 := by
  sorry

end original_wage_before_increase_l198_198619


namespace calc_problem1_calc_problem2_l198_198635

-- Proof Problem 1
theorem calc_problem1 : 
  (Real.sqrt 3 + 2 * Real.sqrt 2) - (3 * Real.sqrt 3 + Real.sqrt 2) = -2 * Real.sqrt 3 + Real.sqrt 2 := 
by 
  sorry

-- Proof Problem 2
theorem calc_problem2 : 
  Real.sqrt 2 * (Real.sqrt 2 + 1 / Real.sqrt 2) - abs (2 - Real.sqrt 6) = 5 - Real.sqrt 6 := 
by 
  sorry

end calc_problem1_calc_problem2_l198_198635


namespace cubic_sum_l198_198670

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 := by
  -- Using the provided conditions x + y = 5 and x^2 + y^2 = 13
  sorry

end cubic_sum_l198_198670


namespace rect_length_is_20_l198_198751

-- Define the conditions
def rect_length_four_times_width (l w : ℝ) : Prop := l = 4 * w
def rect_area_100 (l w : ℝ) : Prop := l * w = 100

-- The main theorem to prove
theorem rect_length_is_20 {l w : ℝ} (h1 : rect_length_four_times_width l w) (h2 : rect_area_100 l w) : l = 20 := by
  sorry

end rect_length_is_20_l198_198751


namespace remainder_when_divided_l198_198489

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l198_198489


namespace parabola_intersects_line_segment_range_l198_198333

theorem parabola_intersects_line_segment_range (a : ℝ) :
  (9/9 : ℝ) ≤ a ∧ a < 2 ↔
  ∃ x1 y1 x2 y2 x0,
    (y1 = a * x1^2 - 3 * x1 + 1) ∧
    (y2 = a * x2^2 - 3 * x2 + 1) ∧
    (∀ x, y = a * x^2 - 3 * x + 1) ∧
    (|x1 - x0| > |x2 - x0| → y1 > y2) ∧
    let M := (-1 : ℝ, -2 : ℝ); N := (3 : ℝ, 2 : ℝ) in
    let y_mn := (x : ℝ) → x - 1 in
    ∃ x_1 x_2, x_1 ≠ x_2 ∧
      (a * x_1^2 - 3 * x_1 + 1 = x_1 - 1) ∧
      (a * x_2^2 - 3 * x_2 + 1 = x_2 - 1) :=
begin
  sorry
end

end parabola_intersects_line_segment_range_l198_198333


namespace percentage_of_360_is_165_6_l198_198072

theorem percentage_of_360_is_165_6 :
  (165.6 / 360) * 100 = 46 :=
by
  sorry

end percentage_of_360_is_165_6_l198_198072


namespace number_of_footballs_l198_198233

theorem number_of_footballs (x y : ℕ) (h1 : x + y = 20) (h2 : 6 * x + 3 * y = 96) : x = 12 :=
by {
  sorry
}

end number_of_footballs_l198_198233


namespace range_of_m_l198_198513

theorem range_of_m {a b c x0 y0 y1 y2 m : ℝ} (h1 : a ≠ 0)
    (A_on_parabola : y1 = a * m^2 + 4 * a * m + c)
    (B_on_parabola : y2 = a * (m + 2)^2 + 4 * a * (m + 2) + c)
    (C_on_parabola : y0 = a * (-2)^2 + 4 * a * (-2) + c)
    (C_is_vertex : x0 = -2)
    (y_relation : y0 ≥ y2 ∧ y2 > y1) :
    m < -3 := 
sorry

end range_of_m_l198_198513


namespace alpha_beta_roots_l198_198853

variable (α β : ℝ)

theorem alpha_beta_roots (h1 : α^2 - 7 * α + 3 = 0) (h2 : β^2 - 7 * β + 3 = 0) (h3 : α > β) :
  α^2 + 7 * β = 46 :=
sorry

end alpha_beta_roots_l198_198853


namespace square_side_length_l198_198458

theorem square_side_length (s : ℝ) (h : 8 * s^2 = 3200) : s = 20 :=
by
  sorry

end square_side_length_l198_198458


namespace Allyson_age_is_28_l198_198239

-- Define the conditions of the problem
def Hiram_age : ℕ := 40
def add_12_to_Hiram_age (h_age : ℕ) : ℕ := h_age + 12
def twice_Allyson_age (a_age : ℕ) : ℕ := 2 * a_age
def condition (h_age : ℕ) (a_age : ℕ) : Prop := add_12_to_Hiram_age h_age = twice_Allyson_age a_age - 4

-- Define the theorem to be proven
theorem Allyson_age_is_28 (a_age : ℕ) (h_age : ℕ) (h_condition : condition h_age a_age) (h_hiram : h_age = Hiram_age) : a_age = 28 :=
by sorry

end Allyson_age_is_28_l198_198239


namespace count_possible_values_l198_198779

open Int

theorem count_possible_values (y : ℕ) :
  (∀ y < 20, lcm y 6 / (y * 6) = 1) → (∃! n, n = 7) := by
  sorry

end count_possible_values_l198_198779


namespace count_at_least_three_adjacent_chairs_l198_198418

noncomputable def count_adjacent_subsets (n : ℕ) (chairs : set (fin n)) : ℕ :=
  -- Add a function definition that counts the number of subsets with at least three adjacent chairs
  sorry

theorem count_at_least_three_adjacent_chairs :
  count_adjacent_subsets 12 (set.univ : set (fin 12)) = 2010 :=
sorry

end count_at_least_three_adjacent_chairs_l198_198418


namespace g_value_at_49_l198_198747

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_value_at_49 :
  (∀ x y : ℝ, 0 < x → 0 < y → x * g y - y * g x = g (x^2 / y)) →
  g 49 = 0 :=
by
  -- Assuming the given condition holds for all positive real numbers x and y
  intro h
  -- sorry placeholder represents the proof process
  sorry

end g_value_at_49_l198_198747


namespace am_gm_example_l198_198739

theorem am_gm_example {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^5 + 4) ≥ 30 :=
sorry

end am_gm_example_l198_198739


namespace num_orange_juice_l198_198434

-- Definitions based on the conditions in the problem
def O : ℝ := sorry -- To represent the number of bottles of orange juice
def A : ℝ := sorry -- To represent the number of bottles of apple juice
def cost_orange_juice : ℝ := 0.70
def cost_apple_juice : ℝ := 0.60
def total_cost : ℝ := 46.20
def total_bottles : ℝ := 70

-- Conditions used as definitions in Lean 4
axiom condition1 : O + A = total_bottles
axiom condition2 : cost_orange_juice * O + cost_apple_juice * A = total_cost

-- Proof statement with the correct answer
theorem num_orange_juice : O = 42 := by
  sorry

end num_orange_juice_l198_198434


namespace largest_initial_number_l198_198881

theorem largest_initial_number (n : ℕ) (h : (∃ a b c d e : ℕ, n ≠ 0 ∧ n + a + b + c + d + e = 200 
                                              ∧ n % a ≠ 0 ∧ n % b ≠ 0 ∧ n % c ≠ 0 ∧ n % d ≠ 0 ∧ n % e ≠ 0)) 
: n ≤ 189 :=
sorry

end largest_initial_number_l198_198881


namespace number_of_adjacent_subsets_l198_198416

/-- The number of subsets of a set of 12 chairs arranged in a circle
that contain at least three adjacent chairs is 1634. -/
theorem number_of_adjacent_subsets : 
  let subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
    if n = 3 then 12 else
    if n = 4 then 12 else
    if n = 5 then 12 else
    if n = 6 then 12 else
    if n = 7 then 792 else
    if n = 8 then 495 else
    if n = 9 then 220 else
    if n = 10 then 66 else
    if n = 11 then 12 else
    if n = 12 then 1 else 0
  in
    (subsets_with_adjacent_chairs 3) +
    (subsets_with_adjacent_chairs 4) +
    (subsets_with_adjacent_chairs 5) +
    (subsets_with_adjacent_chairs 6) +
    (subsets_with_adjacent_chairs 7) +
    (subsets_with_adjacent_chairs 8) +
    (subsets_with_adjacent_chairs 9) +
    (subsets_with_adjacent_chairs 10) +
    (subsets_with_adjacent_chairs 11) +
    (subsets_with_adjacent_chairs 12) = 1634 :=
by
  sorry

end number_of_adjacent_subsets_l198_198416


namespace range_of_x_plus_y_l198_198727

theorem range_of_x_plus_y (x y : ℝ) (h : x^2 + 2 * x * y - 1 = 0) : (x + y ≤ -1 ∨ x + y ≥ 1) :=
by
  sorry

end range_of_x_plus_y_l198_198727


namespace seans_sum_divided_by_julies_sum_l198_198380

theorem seans_sum_divided_by_julies_sum : 
  let seans_sum := 2 * (∑ k in Finset.range 301, k)
  let julies_sum := ∑ k in Finset.range 301, k
  seans_sum / julies_sum = 2 :=
by 
  sorry

end seans_sum_divided_by_julies_sum_l198_198380


namespace product_of_two_digit_numbers_is_not_five_digits_l198_198566

theorem product_of_two_digit_numbers_is_not_five_digits :
  ∀ (a b c d : ℕ), (10 ≤ 10 * a + b) → (10 * a + b ≤ 99) → (10 ≤ 10 * c + d) → (10 * c + d ≤ 99) → 
    (10 * a + b) * (10 * c + d) < 10000 :=
by
  intros a b c d H1 H2 H3 H4
  -- proof steps would go here
  sorry

end product_of_two_digit_numbers_is_not_five_digits_l198_198566


namespace supplies_total_cost_l198_198262

-- Definitions based on conditions in a)
def cost_of_bow : ℕ := 5
def cost_of_vinegar : ℕ := 2
def cost_of_baking_soda : ℕ := 1
def students_count : ℕ := 23

-- The main theorem to prove
theorem supplies_total_cost :
  cost_of_bow * students_count + cost_of_vinegar * students_count + cost_of_baking_soda * students_count = 184 :=
by
  sorry

end supplies_total_cost_l198_198262


namespace marys_score_l198_198030

theorem marys_score (C ω S : ℕ) (H1 : S = 30 + 4 * C - ω) (H2 : S > 80)
  (H3 : (∀ C1 ω1 C2 ω2, (C1 ≠ C2 → 30 + 4 * C1 - ω1 ≠ 30 + 4 * C2 - ω2))) : 
  S = 119 :=
sorry

end marys_score_l198_198030


namespace sampling_prob_equal_l198_198323

theorem sampling_prob_equal (N n : ℕ) (P_1 P_2 P_3 : ℝ)
  (H_random : ∀ i, 1 ≤ i ∧ i ≤ N → P_1 = 1 / N)
  (H_systematic : ∀ i, 1 ≤ i ∧ i ≤ N → P_2 = 1 / N)
  (H_stratified : ∀ i, 1 ≤ i ∧ i ≤ N → P_3 = 1 / N) :
  P_1 = P_2 ∧ P_2 = P_3 :=
by
  sorry

end sampling_prob_equal_l198_198323


namespace smallest_palindrome_in_base3_and_base5_l198_198276

def is_palindrome_base (b n : ℕ) : Prop :=
  let digits := n.digits b
  digits = digits.reverse

theorem smallest_palindrome_in_base3_and_base5 :
  ∃ n : ℕ, n > 10 ∧ is_palindrome_base 3 n ∧ is_palindrome_base 5 n ∧ n = 20 :=
by
  sorry

end smallest_palindrome_in_base3_and_base5_l198_198276


namespace solve_x_l198_198325

noncomputable def op (a b : ℝ) : ℝ := (1 / b) - (1 / a)

theorem solve_x (x : ℝ) (h : op (x - 1) 2 = 1) : x = -1 := 
by {
  -- proof outline here...
  sorry
}

end solve_x_l198_198325


namespace red_grapes_in_salad_l198_198869

theorem red_grapes_in_salad {G R B : ℕ} 
  (h1 : R = 3 * G + 7)
  (h2 : B = G - 5)
  (h3 : G + R + B = 102) : R = 67 :=
sorry

end red_grapes_in_salad_l198_198869


namespace fraction_to_decimal_l198_198314

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198314


namespace hybrids_with_full_headlights_l198_198166

theorem hybrids_with_full_headlights
  (total_cars : ℕ) (hybrid_percentage : ℕ) (one_headlight_percentage : ℕ) :
  total_cars = 600 → hybrid_percentage = 60 → one_headlight_percentage = 40 →
  let total_hybrids := (hybrid_percentage * total_cars) / 100 in
  let one_headlight_hybrids := (one_headlight_percentage * total_hybrids) / 100 in
  let full_headlight_hybrids := total_hybrids - one_headlight_hybrids in
  full_headlight_hybrids = 216 :=
by
  intros h1 h2 h3
  sorry

end hybrids_with_full_headlights_l198_198166


namespace square_area_correct_l198_198246

-- Define the length of the side of the square
def side_length : ℕ := 15

-- Define the area calculation for a square
def square_area (side : ℕ) : ℕ := side * side

-- Define the area calculation for a triangle using the square area division
def triangle_area (square_area : ℕ) : ℕ := square_area / 2

-- Theorem stating that the area of a square with given side length is 225 square units
theorem square_area_correct : square_area side_length = 225 := by
  sorry

end square_area_correct_l198_198246


namespace total_money_correct_l198_198206

def shelly_has_total_money : Prop :=
  ∃ (ten_dollar_bills five_dollar_bills : ℕ), 
    ten_dollar_bills = 10 ∧
    five_dollar_bills = ten_dollar_bills - 4 ∧
    (10 * ten_dollar_bills + 5 * five_dollar_bills = 130)

theorem total_money_correct : shelly_has_total_money :=
by
  sorry

end total_money_correct_l198_198206


namespace least_number_to_subtract_l198_198986

theorem least_number_to_subtract :
  ∃ k : ℕ, k = 45 ∧ (568219 - k) % 89 = 0 :=
by
  sorry

end least_number_to_subtract_l198_198986


namespace original_price_of_coffee_l198_198957

/-- 
  Define the prices of the cups of coffee as per the conditions.
  Let x be the original price of one cup of coffee.
  Assert the conditions and find the original price.
-/
theorem original_price_of_coffee (x : ℝ) 
  (h1 : x + x / 2 + 3 = 57) 
  (h2 : (x + x / 2 + 3)/3 = 19) : 
  x = 36 := 
by
  sorry

end original_price_of_coffee_l198_198957


namespace range_of_a_l198_198726

noncomputable def proposition_p (x : ℝ) : Prop := (4 * x - 3)^2 ≤ 1
noncomputable def proposition_q (x : ℝ) (a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) :
  (¬ (∃ x, ¬ proposition_p x) → ¬ (∃ x, ¬ proposition_q x a)) →
  (¬ (¬ (∃ x, ¬ proposition_p x) ∧ ¬ (¬ (∃ x, ¬ proposition_q x a)))) →
  (0 ≤ a ∧ a ≤ 1 / 2) :=
by
  intro h₁ h₂
  sorry

end range_of_a_l198_198726


namespace find_side_DF_in_triangle_DEF_l198_198358

theorem find_side_DF_in_triangle_DEF
  (DE EF DM : ℝ)
  (h_DE : DE = 7)
  (h_EF : EF = 10)
  (h_DM : DM = 5) :
  ∃ DF : ℝ, DF = Real.sqrt 51 :=
by
  sorry

end find_side_DF_in_triangle_DEF_l198_198358


namespace joan_trip_time_l198_198022

-- Definitions of given conditions as parameters
def distance : ℕ := 480
def speed : ℕ := 60
def lunch_break_minutes : ℕ := 30
def bathroom_break_minutes : ℕ := 15
def number_of_bathroom_breaks : ℕ := 2

-- Conversion factors
def minutes_to_hours (m : ℕ) : ℚ := m / 60

-- Calculation of total time taken
def total_time : ℚ := 
  (distance / speed) + 
  (minutes_to_hours lunch_break_minutes) + 
  (number_of_bathroom_breaks * minutes_to_hours bathroom_break_minutes)

-- Statement of the problem
theorem joan_trip_time : total_time = 9 := 
  by 
    sorry

end joan_trip_time_l198_198022


namespace razorback_tshirt_shop_sales_l198_198918

theorem razorback_tshirt_shop_sales :
  let price_per_tshirt := 16 
  let tshirts_sold := 45 
  price_per_tshirt * tshirts_sold = 720 :=
by
  sorry

end razorback_tshirt_shop_sales_l198_198918


namespace discard_sacks_l198_198854

theorem discard_sacks (harvested_sacks_per_day : ℕ) (oranges_per_day : ℕ) (oranges_per_sack : ℕ) :
  harvested_sacks_per_day = 76 → oranges_per_day = 600 → oranges_per_sack = 50 → 
  harvested_sacks_per_day - oranges_per_day / oranges_per_sack = 64 :=
by
  intros h1 h2 h3
  -- Automatically passes the proof as a placeholder
  sorry

end discard_sacks_l198_198854


namespace binomial_expansion_fraction_l198_198721

theorem binomial_expansion_fraction 
    (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
    (h1 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 1)
    (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = 243) :
    (a_0 + a_2 + a_4) / (a_1 + a_3 + a_5) = -122 / 121 :=
by
  sorry

end binomial_expansion_fraction_l198_198721


namespace symmedian_in_triangle_l198_198777

theorem symmedian_in_triangle (A B C A1 B1 C1 B0 Q : Point) (Ω ω : Circle) :
  altitude A B C A1 → altitude B B1 → altitude C C1 →
  (B0 ∈ Ω) → (B0 = line_intersection (line B B1) Ω) →
  (Q ∈ (Ω ∩ ω)) → (Q ≠ B0) → symmedian B Q A B C :=
by
  sorry

end symmedian_in_triangle_l198_198777


namespace speed_of_car_A_l198_198100

variable (V_A V_B T : ℕ)
variable (h1 : V_B = 35) (h2 : T = 10) (h3 : 2 * V_B * T = V_A * T)

theorem speed_of_car_A :
  V_A = 70 :=
by
  sorry

end speed_of_car_A_l198_198100


namespace best_store_is_A_l198_198343

/-- Problem conditions -/
def price_per_ball : Nat := 25
def balls_to_buy : Nat := 58

/-- Store A conditions -/
def balls_bought_per_offer_A : Nat := 10
def balls_free_per_offer_A : Nat := 3

/-- Store B conditions -/
def discount_per_ball_B : Nat := 5

/-- Store C conditions -/
def cashback_rate_C : Nat := 40
def cashback_threshold_C : Nat := 200

/-- Cost calculations -/
def cost_store_A (total_balls : Nat) (price : Nat) : Nat :=
  let full_offers := total_balls / balls_bought_per_offer_A
  let remaining_balls := total_balls % balls_bought_per_offer_A
  let balls_paid_for := full_offers * (balls_bought_per_offer_A - balls_free_per_offer_A) + remaining_balls
  balls_paid_for * price

def cost_store_B (total_balls : Nat) (price : Nat) (discount : Nat) : Nat :=
  total_balls * (price - discount)

def cost_store_C (total_balls : Nat) (price : Nat) (cashback_rate : Nat) (threshold : Nat) : Nat :=
  let cost_before_cashback := total_balls * price
  let full_cashbacks := cost_before_cashback / threshold
  let cashback_amount := full_cashbacks * cashback_rate
  cost_before_cashback - cashback_amount

theorem best_store_is_A :
  cost_store_A balls_to_buy price_per_ball = 1075 ∧
  cost_store_B balls_to_buy price_per_ball discount_per_ball_B = 1160 ∧
  cost_store_C balls_to_buy price_per_ball cashback_rate_C cashback_threshold_C = 1170 ∧
  cost_store_A balls_to_buy price_per_ball < cost_store_B balls_to_buy price_per_ball discount_per_ball_B ∧
  cost_store_A balls_to_buy price_per_ball < cost_store_C balls_to_buy price_per_ball cashback_rate_C cashback_threshold_C :=
by {
  -- placeholder for the proof
  sorry
}

end best_store_is_A_l198_198343


namespace car_distribution_l198_198626

theorem car_distribution :
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let fourth_supplier := (total_cars - (first_supplier + second_supplier + third_supplier)) / 2
  let fifth_supplier := fourth_supplier
  fourth_supplier = 325000 ∧ fifth_supplier = 325000 := 
by 
  sorry

end car_distribution_l198_198626


namespace distance_city_A_to_C_l198_198641

variable (V_E V_F : ℝ) -- Define the average speeds of Eddy and Freddy
variable (time : ℝ) -- Define the time variable

-- Given conditions
def eddy_time : time = 3 := sorry
def freddy_time : time = 3 := sorry
def eddy_distance : ℝ := 600
def speed_ratio : V_E = 2 * V_F := sorry

-- Derived condition for Eddy's speed
def eddy_speed : V_E = eddy_distance / time := sorry

-- Derived conclusion for Freddy's distance
theorem distance_city_A_to_C (time : ℝ) (V_F : ℝ) : V_F * time = 300 := 
by 
  sorry

end distance_city_A_to_C_l198_198641


namespace cara_total_bread_l198_198324

variable (L B : ℕ)  -- Let L and B be the amount of bread for lunch and breakfast, respectively

theorem cara_total_bread :
  (dinner = 240) → 
  (dinner = 8 * L) → 
  (dinner = 6 * B) → 
  (total_bread = dinner + L + B) → 
  total_bread = 310 :=
by
  intros
  -- Here you'd begin your proof, implementing each given condition
  sorry

end cara_total_bread_l198_198324


namespace book_price_l198_198904

theorem book_price (x : ℕ) : 
  9 * x ≤ 1100 ∧ 13 * x ≤ 1500 → x = 123 :=
sorry

end book_price_l198_198904


namespace find_x_value_l198_198525

theorem find_x_value (x : ℝ) (h1 : x^2 + x = 6) (h2 : x^2 - 2 = 1) : x = 2 := sorry

end find_x_value_l198_198525


namespace find_DF_l198_198359

theorem find_DF (D E F M : Point) (DE EF DM DF : ℝ)
  (h1 : DE = 7)
  (h2 : EF = 10)
  (h3 : DM = 5)
  (h4 : midpoint F E M)
  (h5 : M = circumcenter D E F) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l198_198359


namespace verify_graphical_method_l198_198079

variable {R : Type} [LinearOrderedField R]

/-- Statement of the mentioned conditions -/
def poly (a b c d x : R) : R := a * x^3 + b * x^2 + c * x + d

/-- The main theorem stating the graphical method validity -/
theorem verify_graphical_method (a b c d x0 EJ : R) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) (h3 : 0 < d) (h4 : 0 < x0) (h5 : x0 < 1)
: EJ = poly a b c d x0 := by sorry

end verify_graphical_method_l198_198079


namespace total_floor_area_covered_l198_198593

theorem total_floor_area_covered (combined_area : ℝ) (area_two_layers : ℝ) (area_three_layers : ℝ) : 
  combined_area = 200 → 
  area_two_layers = 22 → 
  area_three_layers = 19 → 
  (combined_area - (area_two_layers + 2 * area_three_layers)) = 140 := 
by
  sorry

end total_floor_area_covered_l198_198593


namespace point_in_at_least_15_circles_l198_198234

theorem point_in_at_least_15_circles
  (C : Fin 100 → Set (ℝ × ℝ))
  (h1 : ∀ i j, ∃ p, p ∈ C i ∧ p ∈ C j)
  : ∃ p, ∃ S : Finset (Fin 100), S.card ≥ 15 ∧ ∀ i ∈ S, p ∈ C i :=
sorry

end point_in_at_least_15_circles_l198_198234


namespace min_value_range_of_a_l198_198337

noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.exp (-2 * x) + a * (2 * x + 1) * Real.exp (-x) + x^2 + x

theorem min_value_range_of_a (a : ℝ) (h : a > 0)
  (min_f : ∃ x : ℝ, f a x = Real.log a ^ 2 + 3 * Real.log a + 2) :
  a ∈ Set.Ici (Real.exp (-3 / 2)) :=
by
  sorry

end min_value_range_of_a_l198_198337


namespace polynomial_coeff_sum_l198_198668

theorem polynomial_coeff_sum :
  let f : ℕ → ℕ := λ x => (x^2 + 1) * (x - 1)^8
  let a_0 := f 2
  let a_sum := f 3
  a_sum - a_0 = 2555 :=
by
  let f : ℕ → ℕ := λ x => (x^2 + 1) * (x - 1)^8
  let a_0 := f 2
  let a_sum := f 3
  show a_sum - a_0 = 2555
  sorry

end polynomial_coeff_sum_l198_198668


namespace number_of_divisors_of_2744_l198_198826

-- Definition of the integer and its prime factorization
def two := 2
def seven := 7
def n := two^3 * seven^3

-- Define the property for the number of divisors
def num_divisors (n : ℕ) : ℕ := (3 + 1) * (3 + 1)

-- Main proof statement
theorem number_of_divisors_of_2744 : num_divisors n = 16 := by
  sorry

end number_of_divisors_of_2744_l198_198826


namespace not_polynomial_option_B_l198_198969

-- Definitions
def is_polynomial (expr : String) : Prop :=
  -- Assuming we have a function that determines if a given string expression is a polynomial.
  sorry

def option_A : String := "m+n"
def option_B : String := "x=1"
def option_C : String := "xy"
def option_D : String := "0"

-- Problem Statement
theorem not_polynomial_option_B : ¬ is_polynomial option_B := 
sorry

end not_polynomial_option_B_l198_198969


namespace wyatt_envelopes_fewer_l198_198073

-- Define assets for envelopes
variables (blue_envelopes yellow_envelopes : ℕ)

-- Conditions from the problem
def wyatt_conditions :=
  blue_envelopes = 10 ∧ yellow_envelopes < blue_envelopes ∧ blue_envelopes + yellow_envelopes = 16

-- Theorem: How many fewer yellow envelopes Wyatt has compared to blue envelopes?
theorem wyatt_envelopes_fewer (hb : blue_envelopes = 10) (ht : blue_envelopes + yellow_envelopes = 16) : 
  blue_envelopes - yellow_envelopes = 4 := 
by sorry

end wyatt_envelopes_fewer_l198_198073


namespace B_and_C_complete_task_l198_198610

noncomputable def A_work_rate : ℚ := 1 / 12
noncomputable def B_work_rate : ℚ := 1.2 * A_work_rate
noncomputable def C_work_rate : ℚ := 2 * A_work_rate

theorem B_and_C_complete_task (B_work_rate C_work_rate : ℚ) 
    (A_work_rate : ℚ := 1 / 12) :
  B_work_rate = 1.2 * A_work_rate →
  C_work_rate = 2 * A_work_rate →
  (B_work_rate + C_work_rate) = 4 / 15 :=
by intros; sorry

end B_and_C_complete_task_l198_198610


namespace remainder_div_252_l198_198465

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l198_198465


namespace marks_chemistry_l198_198110

-- Definitions based on conditions
def marks_english : ℕ := 96
def marks_math : ℕ := 98
def marks_physics : ℕ := 99
def marks_biology : ℕ := 98
def average_marks : ℝ := 98.2
def num_subjects : ℕ := 5

-- Statement to prove
theorem marks_chemistry :
  ((marks_english + marks_math + marks_physics + marks_biology : ℕ) + (x : ℕ)) / num_subjects = average_marks →
  x = 100 :=
by
  sorry

end marks_chemistry_l198_198110


namespace factorize_square_difference_l198_198643

open Real

theorem factorize_square_difference (m n : ℝ) :
  m ^ 2 - 4 * n ^ 2 = (m + 2 * n) * (m - 2 * n) :=
sorry

end factorize_square_difference_l198_198643


namespace inversely_varies_y_l198_198609

theorem inversely_varies_y (x y : ℕ) (k : ℕ) (h₁ : 7 * y = k / x^3) (h₂ : y = 8) (h₃ : x = 2) : 
  y = 1 :=
by
  sorry

end inversely_varies_y_l198_198609


namespace expected_squares_under_attack_l198_198760

noncomputable def expected_attacked_squares : ℝ :=
  let p_no_attack_one_rook : ℝ := (7/8) * (7/8) in
  let p_no_attack_any_rook : ℝ := p_no_attack_one_rook ^ 3 in
  let p_attack_least_one_rook : ℝ := 1 - p_no_attack_any_rook in
  64 * p_attack_least_one_rook 

theorem expected_squares_under_attack :
  expected_attacked_squares = 35.33 :=
by
  sorry

end expected_squares_under_attack_l198_198760


namespace yura_finishes_textbook_on_sep_12_l198_198688

theorem yura_finishes_textbook_on_sep_12 :
  ∃ end_date : ℕ,
    let total_problems := 91,
        problems_left_on_sep_8 := 46,
        solved_until_sep_8 := total_problems - problems_left_on_sep_8,
        z := solved_until_sep_8 / 3, -- Since 3z = solved_until_sep_8
        sep6_solved := z + 1,
        sep7_solved := z,
        sep8_solved := z - 1,
        remaining_problems := total_problems - (sep6_solved + sep7_solved + sep8_solved),
        end_date := 8 + (sep8_solved * (sep8_solved + 1)) / 2,
        -- 8th + sum of an arithmetic series starting from 13 down to 1 problem
    end_date = 12 := sorry

end yura_finishes_textbook_on_sep_12_l198_198688


namespace expected_attacked_squares_l198_198762

theorem expected_attacked_squares :
  ∀ (board : Fin 8 × Fin 8) (rooks : Finset (Fin 8 × Fin 8)), rooks.card = 3 →
  (∀ r ∈ rooks, r ≠ board) →
  let attacked_by_r (r : Fin 8 × Fin 8) (sq : Fin 8 × Fin 8) := r.fst = sq.fst ∨ r.snd = sq.snd in
  let attacked (sq : Fin 8 × Fin 8) := ∃ r ∈ rooks, attacked_by_r r sq in
  let expected_num_attacked_squares := ∑ sq in Finset.univ, ite (attacked sq) 1 0 / 64 in
  expected_num_attacked_squares ≈ 35.33 :=
sorry

end expected_attacked_squares_l198_198762


namespace sequence_ratio_l198_198142

theorem sequence_ratio :
  ∀ {a : ℕ → ℝ} (h₁ : a 1 = 1/2) (h₂ : ∀ n, a n = (a (n + 1)) * (a (n + 1))),
  (a 200 / a 300) = (301 / 201) :=
by
  sorry

end sequence_ratio_l198_198142


namespace correct_transformation_l198_198242

theorem correct_transformation (a b c : ℝ) (h : (b / (a^2 + 1)) > (c / (a^2 + 1))) : b > c :=
by {
  -- Placeholder proof
  sorry
}

end correct_transformation_l198_198242


namespace Xiaoyong_age_solution_l198_198946

theorem Xiaoyong_age_solution :
  ∃ (x y : ℕ), 1 ≤ y ∧ y < x ∧ x < 20 ∧ 2 * x + 5 * y = 97 ∧ x = 16 ∧ y = 13 :=
by
  -- You should provide a suitable proof here
  sorry

end Xiaoyong_age_solution_l198_198946


namespace sufficiency_condition_l198_198833

-- Definitions of p and q
def p (a b : ℝ) : Prop := a > |b|
def q (a b : ℝ) : Prop := a^2 > b^2

-- Main theorem statement
theorem sufficiency_condition (a b : ℝ) : (p a b → q a b) ∧ (¬(q a b → p a b)) := 
by
  sorry

end sufficiency_condition_l198_198833


namespace correct_calculation_l198_198606

theorem correct_calculation (m n : ℤ) :
  (m^2 * m^3 ≠ m^6) ∧
  (- (m - n) = -m + n) ∧
  (m * (m + n) ≠ m^2 + n) ∧
  ((m + n)^2 ≠ m^2 + n^2) :=
by sorry

end correct_calculation_l198_198606


namespace total_prep_time_l198_198714

-- Definitions:
def jack_time_to_put_shoes_on : ℕ := 4
def additional_time_per_toddler : ℕ := 3
def number_of_toddlers : ℕ := 2

-- Total time calculation
def total_time : ℕ :=
  let time_per_toddler := jack_time_to_put_shoes_on + additional_time_per_toddler
  let total_toddler_time := time_per_toddler * number_of_toddlers
  total_toddler_time + jack_time_to_put_shoes_on

-- Theorem:
theorem total_prep_time :
  total_time = 18 :=
sorry

end total_prep_time_l198_198714


namespace yura_finishes_on_correct_date_l198_198700

-- Let there be 91 problems in the textbook.
-- Let Yura start solving problems on September 6th.
def total_problems : Nat := 91
def start_date : Nat := 6

-- Each morning, starting from September 7, he solves one problem less than the previous morning.
def problems_solved (n : Nat) : Nat := if n = 0 then 0 else problems_solved (n - 1) - 1

-- On the evening of September 8, there are 46 problems left to solve.
def remaining_problems_sept_8 : Nat := 46

-- The question is to find on which date Yura will finish solving the textbook
def finish_date : Nat := 12

theorem yura_finishes_on_correct_date : ∃ z : Nat, (problems_solved z * 3 = total_problems - remaining_problems_sept_8) ∧ (z = 15) ∧ finish_date = 12 := by sorry

end yura_finishes_on_correct_date_l198_198700


namespace area_of_field_with_tomatoes_l198_198759

theorem area_of_field_with_tomatoes :
  let length := 3.6
  let width := 2.5 * length
  let total_area := length * width
  let area_with_tomatoes := total_area / 2
  area_with_tomatoes = 16.2 :=
by
  sorry

end area_of_field_with_tomatoes_l198_198759


namespace minimum_swaps_to_sort_30_volumes_l198_198411

/-- disorder: a pair of volumes where the volume with the larger number stands to the left of the volume with the smaller number --/
def disorder (v : ℕ) (u : ℕ) : Prop := v > u

noncomputable def count_disorders (arr : list ℕ) : ℕ :=
(arr.zip arr.tail).count (λ (vu : ℕ × ℕ), disorder vu.fst vu.snd)

theorem minimum_swaps_to_sort_30_volumes :
  ∀ (initial_arrangement : list ℕ), initial_arrangement.length = 30 →
  minimum_swaps_to_sort initial_arrangement = 435 := by
  sorry
  
/-- minimum_swaps_to_sort - function to calculate the number of swaps required to sort a list in the correct order --/
noncomputable def minimum_swaps_to_sort (arr : list ℕ) : ℕ :=
-- heuristic to determine the minimum number of swaps needed; to be proved 
435


end minimum_swaps_to_sort_30_volumes_l198_198411


namespace grocery_store_more_expensive_per_can_l198_198252

theorem grocery_store_more_expensive_per_can :
  ∀ (bulk_case_price : ℝ) (bulk_cans_per_case : ℕ)
    (grocery_case_price : ℝ) (grocery_cans_per_case : ℕ),
  bulk_case_price = 12.00 →
  bulk_cans_per_case = 48 →
  grocery_case_price = 6.00 →
  grocery_cans_per_case = 12 →
  (grocery_case_price / grocery_cans_per_case - bulk_case_price / bulk_cans_per_case) * 100 = 25 :=
by
  intros _ _ _ _ h1 h2 h3 h4
  sorry

end grocery_store_more_expensive_per_can_l198_198252


namespace dogs_sold_l198_198804

theorem dogs_sold (cats_sold : ℕ) (h1 : cats_sold = 16) (ratio : ℕ × ℕ) (h2 : ratio = (2, 1)) : ∃ dogs_sold : ℕ, dogs_sold = 8 := by
  sorry

end dogs_sold_l198_198804


namespace fraction_to_decimal_l198_198310

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198310


namespace chairs_subsets_l198_198419

theorem chairs_subsets:
  let n := 12 in
  count_subsets_with_at_least_three_adjacent_chairs n = 2066 :=
sorry

end chairs_subsets_l198_198419


namespace difference_of_values_l198_198174

theorem difference_of_values (num : Nat) : 
  (num = 96348621) →
  let face_value := 8
  let local_value := 8 * 10000
  local_value - face_value = 79992 := 
by
  intros h_eq
  have face_value := 8
  have local_value := 8 * 10000
  sorry

end difference_of_values_l198_198174


namespace hybrids_with_full_headlights_l198_198167

theorem hybrids_with_full_headlights
  (total_cars : ℕ) (hybrid_percentage : ℕ) (one_headlight_percentage : ℕ) :
  total_cars = 600 → hybrid_percentage = 60 → one_headlight_percentage = 40 →
  let total_hybrids := (hybrid_percentage * total_cars) / 100 in
  let one_headlight_hybrids := (one_headlight_percentage * total_hybrids) / 100 in
  let full_headlight_hybrids := total_hybrids - one_headlight_hybrids in
  full_headlight_hybrids = 216 :=
by
  intros h1 h2 h3
  sorry

end hybrids_with_full_headlights_l198_198167


namespace students_per_bench_l198_198581

-- Definitions based on conditions
def num_male_students : ℕ := 29
def num_female_students : ℕ := 4 * num_male_students
def num_benches : ℕ := 29
def total_students : ℕ := num_male_students + num_female_students

-- Theorem to prove
theorem students_per_bench : total_students / num_benches = 5 := by
  sorry

end students_per_bench_l198_198581


namespace find_vector_at_t_0_l198_198088

def vec2 := ℝ × ℝ

def line_at_t (a d : vec2) (t : ℝ) : vec2 :=
  (a.1 + t * d.1, a.2 + t * d.2)

-- Given conditions
def vector_at_t_1 (v : vec2) : Prop :=
  v = (2, 3)

def vector_at_t_4 (v : vec2) : Prop :=
  v = (8, -5)

-- Prove that the vector at t = 0 is (0, 17/3)
theorem find_vector_at_t_0 (a d: vec2) (h1: line_at_t a d 1 = (2, 3)) (h4: line_at_t a d 4 = (8, -5)) :
  line_at_t a d 0 = (0, 17 / 3) :=
sorry

end find_vector_at_t_0_l198_198088


namespace sticks_per_pot_is_181_l198_198590

/-- Define the problem conditions -/
def number_of_pots : ℕ := 466
def flowers_per_pot : ℕ := 53
def total_flowers_and_sticks : ℕ := 109044

/-- Define the function to calculate the number of sticks per pot -/
def sticks_per_pot (S : ℕ) : Prop :=
  (number_of_pots * flowers_per_pot + number_of_pots * S = total_flowers_and_sticks)

/-- State the theorem -/
theorem sticks_per_pot_is_181 : sticks_per_pot 181 :=
by
  sorry

end sticks_per_pot_is_181_l198_198590


namespace racers_meet_at_start_again_l198_198039

-- We define the conditions as given
def RacingMagic_time := 60
def ChargingBull_time := 60 * 60 / 40 -- 90 seconds
def SwiftShadow_time := 80
def SpeedyStorm_time := 100

-- Prove the LCM of their lap times is 3600 seconds,
-- which is equivalent to 60 minutes.
theorem racers_meet_at_start_again :
  Nat.lcm (Nat.lcm (Nat.lcm RacingMagic_time ChargingBull_time) SwiftShadow_time) SpeedyStorm_time = 3600 ∧
  3600 / 60 = 60 := by
  sorry

end racers_meet_at_start_again_l198_198039


namespace cristian_cookie_problem_l198_198281

theorem cristian_cookie_problem (white_cookies_init black_cookies_init eaten_black_cookies eaten_white_cookies remaining_black_cookies remaining_white_cookies total_remaining_cookies : ℕ) 
  (h_initial_white : white_cookies_init = 80)
  (h_black_more : black_cookies_init = white_cookies_init + 50)
  (h_eats_half_black : eaten_black_cookies = black_cookies_init / 2)
  (h_eats_three_fourth_white : eaten_white_cookies = (3 / 4) * white_cookies_init)
  (h_remaining_black : remaining_black_cookies = black_cookies_init - eaten_black_cookies)
  (h_remaining_white : remaining_white_cookies = white_cookies_init - eaten_white_cookies)
  (h_total_remaining : total_remaining_cookies = remaining_black_cookies + remaining_white_cookies) :
  total_remaining_cookies = 85 :=
by
  sorry

end cristian_cookie_problem_l198_198281


namespace probability_of_sum_at_least_10_l198_198943

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 6

theorem probability_of_sum_at_least_10 :
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 1 / 6 := by
  sorry

end probability_of_sum_at_least_10_l198_198943


namespace incorrect_statement_l198_198872

def data_set : List ℤ := [10, 8, 6, 9, 8, 7, 8]

theorem incorrect_statement : 
  let mode := 8
  let median := 8
  let mean := 8
  let variance := 8
  (∃ x ∈ data_set, x ≠ 8) → -- suppose there is at least one element in the dataset not equal to 8
  (1 / 7 : ℚ) * (4 + 0 + 4 + 1 + 0 + 1 + 0) ≠ 8 := -- calculating real variance from dataset
by
  sorry

end incorrect_statement_l198_198872


namespace smallest_n_is_29_l198_198091

noncomputable def smallest_possible_n (r g b : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm (10 * r) (16 * g)) (18 * b) / 25

theorem smallest_n_is_29 (r g b : ℕ) (h : 10 * r = 16 * g ∧ 16 * g = 18 * b) :
  smallest_possible_n r g b = 29 :=
by
  sorry

end smallest_n_is_29_l198_198091


namespace yura_finishes_problems_by_sept_12_l198_198699

def total_problems := 91
def initial_date := 6 -- September 6
def problems_left_date := 8 -- September 8
def remaining_problems := 46
def decreasing_rate := 1

def problems_solved (z : ℕ) (day : ℕ) : ℕ :=
if day = 6 then z + 1 else if day = 7 then z else if day = 8 then z - 1 else z - (day - 7)

theorem yura_finishes_problems_by_sept_12 :
  ∃ z : ℕ, (problems_solved z 6 + problems_solved z 7 + problems_solved z 8 = total_problems - remaining_problems) ∧
           (problems_solved z 6 + problems_solved z 7 + problems_solved z 8 + problems_solved z 9 + problems_solved z 10 + problems_solved z 11 + problems_solved z 12 = total_problems) :=
sorry

end yura_finishes_problems_by_sept_12_l198_198699


namespace cube_sum_l198_198673

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cube_sum_l198_198673


namespace tan_double_angle_l198_198657

theorem tan_double_angle (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (α - π / 2) = 3 / 5) : 
  Real.tan (2 * α) = 24 / 7 :=
by
  sorry

end tan_double_angle_l198_198657


namespace quadratic_no_real_roots_probability_l198_198930

theorem quadratic_no_real_roots_probability :
  (1 : ℝ) - 1 / 4 - 0 = 3 / 4 :=
by
  sorry

end quadratic_no_real_roots_probability_l198_198930


namespace cost_of_adult_ticket_l198_198940

theorem cost_of_adult_ticket
  (A : ℝ) -- Cost of an adult ticket in dollars
  (x y : ℝ) -- Number of children tickets and number of adult tickets respectively
  (hx : x = 90) -- Condition: number of children tickets sold
  (hSum : x + y = 130) -- Condition: total number of tickets sold
  (hTotal : 4 * x + A * y = 840) -- Condition: total receipts from all tickets
  : A = 12 := 
by
  -- Proof is skipped as per instruction
  sorry

end cost_of_adult_ticket_l198_198940


namespace hybrids_with_full_headlights_l198_198169

-- Definitions for the conditions
def total_cars : ℕ := 600
def percentage_hybrids : ℕ := 60
def percentage_one_headlight : ℕ := 40

-- The proof statement
theorem hybrids_with_full_headlights :
  (percentage_hybrids * total_cars) / 100 - (percentage_one_headlight * (percentage_hybrids * total_cars) / 100) / 100 = 216 :=
by
  sorry

end hybrids_with_full_headlights_l198_198169


namespace Frank_has_four_one_dollar_bills_l198_198121

noncomputable def Frank_one_dollar_bills : ℕ :=
  let total_money := 4 * 5 + 2 * 10 + 20 -- Money from five, ten, and twenty dollar bills
  let peanuts_cost := 10 - 4 -- Cost of peanuts (given $10 and received $4 in change)
  let one_dollar_bills_value := 54 - total_money -- Total money Frank has - money from large bills
  (one_dollar_bills_value : ℕ)

theorem Frank_has_four_one_dollar_bills 
   (five_dollar_bills : ℕ := 4) 
   (ten_dollar_bills : ℕ := 2)
   (twenty_dollar_bills : ℕ := 1)
   (peanut_price : ℚ := 3)
   (change : ℕ := 4)
   (total_money : ℕ := 50)
   (total_money_incl_change : ℚ := 54):
   Frank_one_dollar_bills = 4 := by
  sorry

end Frank_has_four_one_dollar_bills_l198_198121


namespace quadratic_real_roots_range_l198_198501

theorem quadratic_real_roots_range (m : ℝ) :
  ∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0 ↔ m ≤ 4 ∧ m ≠ 3 := 
by
  sorry

end quadratic_real_roots_range_l198_198501


namespace value_multiplied_by_l198_198773

theorem value_multiplied_by (x : ℝ) (h : (7.5 / 6) * x = 15) : x = 12 :=
by
  sorry

end value_multiplied_by_l198_198773


namespace sum_of_powers_mod7_l198_198326

theorem sum_of_powers_mod7 (k : ℕ) : (2^k + 3^k) % 7 = 0 ↔ k % 6 = 3 := by
  sorry

end sum_of_powers_mod7_l198_198326


namespace square_of_1031_l198_198105

theorem square_of_1031 : 1031 ^ 2 = 1060961 := by
  calc
    1031 ^ 2 = (1000 + 31) ^ 2       : by sorry
           ... = 1000 ^ 2 + 2 * 1000 * 31 + 31 ^ 2 : by sorry
           ... = 1000000 + 62000 + 961 : by sorry
           ... = 1060961 : by sorry

end square_of_1031_l198_198105


namespace find_other_number_l198_198215

theorem find_other_number (hcf lcm a b: ℕ) (hcf_value: hcf = 12) (lcm_value: lcm = 396) (a_value: a = 36) (gcd_ab: Nat.gcd a b = hcf) (lcm_ab: Nat.lcm a b = lcm) : b = 132 :=
by
  sorry

end find_other_number_l198_198215


namespace pairs_satisfying_condition_l198_198006

theorem pairs_satisfying_condition :
  (∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 1000 ∧ 1 ≤ y ∧ y ≤ 1000 ∧ (x^2 + y^2) % 7 = 0) → 
  (∃ n : ℕ, n = 20164) :=
sorry

end pairs_satisfying_condition_l198_198006


namespace minimum_value_of_f_range_of_t_l198_198339

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 4)

theorem minimum_value_of_f : ∀ x, f x ≥ 6 ∧ ∃ x0 : ℝ, f x0 = 6 := 
by sorry

theorem range_of_t (t : ℝ) : (t ≤ -2 ∨ t ≥ 3) ↔ ∃ x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f x ≤ t^2 - t :=
by sorry

end minimum_value_of_f_range_of_t_l198_198339


namespace original_number_is_45_l198_198264

theorem original_number_is_45 (x y : ℕ) (h1 : x + y = 9) (h2 : 10 * y + x = 10 * x + y + 9) : 10 * x + y = 45 := by
  sorry

end original_number_is_45_l198_198264


namespace sum_of_cubes_l198_198685

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : a^3 + b^3 = 26 :=
sorry

end sum_of_cubes_l198_198685


namespace p_sufficient_not_necessary_q_l198_198128

def p (x : ℝ) := 0 < x ∧ x < 2
def q (x : ℝ) := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l198_198128


namespace find_m_range_l198_198514

theorem find_m_range
  (m y1 y2 y0 x0 : ℝ)
  (a c : ℝ) (h1 : a ≠ 0)
  (h2 : x0 = -2)
  (h3 : ∀ x, (x, ax^2 + 4*a*x + c) = (m, y1) ∨ (x, ax^2 + 4*a*x + c) = (m + 2, y2) ∨ (x, ax^2 + 4*a*x + c) = (x0, y0))
  (h4 : y0 ≥ y2) (h5 : y2 > y1) :
  m < -3 :=
sorry

end find_m_range_l198_198514


namespace digits_of_2_120_l198_198822

theorem digits_of_2_120 (h : ∀ n : ℕ, (10 : ℝ)^(n - 1) ≤ (2 : ℝ)^200 ∧ (2 : ℝ)^200 < (10 : ℝ)^n → n = 61) :
  ∀ m : ℕ, (10 : ℝ)^(m - 1) ≤ (2 : ℝ)^120 ∧ (2 : ℝ)^120 < (10 : ℝ)^m → m = 37 :=
by
  sorry

end digits_of_2_120_l198_198822


namespace fraction_to_decimal_l198_198288

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l198_198288


namespace ellipse_sum_l198_198107

-- Define the givens
def h : ℤ := -3
def k : ℤ := 5
def a : ℤ := 7
def b : ℤ := 4

-- State the theorem to be proven
theorem ellipse_sum : h + k + a + b = 13 := by
  sorry

end ellipse_sum_l198_198107


namespace Alyssa_total_spent_l198_198802

/-- Definition of fruit costs -/
def cost_grapes : ℝ := 12.08
def cost_cherries : ℝ := 9.85
def cost_mangoes : ℝ := 7.50
def cost_pineapple : ℝ := 4.25
def cost_starfruit : ℝ := 3.98

/-- Definition of tax and discount -/
def tax_rate : ℝ := 0.10
def discount : ℝ := 3.00

/-- Calculation of the total cost Alyssa spent after applying tax and discount -/
def total_spent : ℝ := 
  let total_cost_before_tax := cost_grapes + cost_cherries + cost_mangoes + cost_pineapple + cost_starfruit
  let tax := tax_rate * total_cost_before_tax
  let total_cost_with_tax := total_cost_before_tax + tax
  total_cost_with_tax - discount

/-- Statement that needs to be proven -/
theorem Alyssa_total_spent : total_spent = 38.43 := by 
  sorry

end Alyssa_total_spent_l198_198802


namespace positive_difference_of_squares_and_product_l198_198586

theorem positive_difference_of_squares_and_product (x y : ℕ) 
  (h1 : x + y = 60) (h2 : x - y = 16) :
  x^2 - y^2 = 960 ∧ x * y = 836 :=
by sorry

end positive_difference_of_squares_and_product_l198_198586


namespace inequality_proof_l198_198912

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a^3 + b^3 + c^3 = 3) :
  (1 / (a^4 + 3) + 1 / (b^4 + 3) + 1 / (c^4 + 3) >= 3 / 4) :=
by
  sorry

end inequality_proof_l198_198912


namespace remainder_123456789012_div_252_l198_198488

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l198_198488


namespace find_f_at_7_l198_198996

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^7 + b * x^3 + c * x - 5

theorem find_f_at_7 (a b c : ℝ) (h : f a b c (-7) = 7) : f a b c 7 = -17 :=
by
  sorry

end find_f_at_7_l198_198996


namespace polynomial_strictly_monotone_l198_198735

open Polynomial

-- Define strictly monotone function
def strictly_monotone (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the statement of the problem
theorem polynomial_strictly_monotone (P : Polynomial ℝ) 
  (h : strictly_monotone (λ x, P.eval (P.eval x))) : strictly_monotone (λ x, P.eval x) :=
sorry

end polynomial_strictly_monotone_l198_198735


namespace total_prep_time_l198_198715

-- Definitions:
def jack_time_to_put_shoes_on : ℕ := 4
def additional_time_per_toddler : ℕ := 3
def number_of_toddlers : ℕ := 2

-- Total time calculation
def total_time : ℕ :=
  let time_per_toddler := jack_time_to_put_shoes_on + additional_time_per_toddler
  let total_toddler_time := time_per_toddler * number_of_toddlers
  total_toddler_time + jack_time_to_put_shoes_on

-- Theorem:
theorem total_prep_time :
  total_time = 18 :=
sorry

end total_prep_time_l198_198715


namespace David_pushups_l198_198980

-- Definitions and setup conditions
def Zachary_pushups : ℕ := 7
def additional_pushups : ℕ := 30

-- Theorem statement to be proved
theorem David_pushups 
  (zachary_pushups : ℕ) 
  (additional_pushups : ℕ) 
  (Zachary_pushups_val : zachary_pushups = Zachary_pushups) 
  (additional_pushups_val : additional_pushups = additional_pushups) :
  zachary_pushups + additional_pushups = 37 :=
sorry

end David_pushups_l198_198980


namespace solve_trig_problem_l198_198816

-- Definition of the given problem for trigonometric identities
def problem_statement : Prop :=
  (1 - Real.tan (Real.pi / 12)) / (1 + Real.tan (Real.pi / 12)) = Real.sqrt 3 / 3

theorem solve_trig_problem : problem_statement :=
  by
  sorry -- No proof is needed here

end solve_trig_problem_l198_198816


namespace second_chapter_pages_l198_198612

/-- A book has 2 chapters across 81 pages. The first chapter is 13 pages long. -/
theorem second_chapter_pages (total_pages : ℕ) (first_chapter_pages : ℕ) (second_chapter_pages : ℕ) : 
  total_pages = 81 → 
  first_chapter_pages = 13 → 
  second_chapter_pages = total_pages - first_chapter_pages → 
  second_chapter_pages = 68 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end second_chapter_pages_l198_198612


namespace largest_initial_number_l198_198886

-- Let's define the conditions and the result
def valid_addition (n a : ℕ) : Prop := ∃ k : ℕ, n = a * k + r ∧ 0 < r ∧ r < a

def valid_operations (initial : ℕ) (final : ℕ) (steps : ℕ → ℕ → ℕ) : Prop :=
  ∃ (a b c d e : ℕ), valid_addition initial a ∧
                      valid_addition (initial + a) b ∧
                      valid_addition (initial + a + b) c ∧
                      valid_addition (initial + a + b + c) d ∧
                      valid_addition (initial + a + b + c + d) e ∧
                      initial + a + b + c + d + e = final

theorem largest_initial_number :
  ∃ n : ℕ, (valid_operations n 200 (λn a, n + a)) ∧ (∀ m : ℕ, valid_operations m 200 (λn a, n + a) → m ≤ n) :=
sorry

end largest_initial_number_l198_198886


namespace markus_more_marbles_than_mara_l198_198729

variable (mara_bags : Nat) (mara_marbles_per_bag : Nat)
variable (markus_bags : Nat) (markus_marbles_per_bag : Nat)

theorem markus_more_marbles_than_mara :
  mara_bags = 12 →
  mara_marbles_per_bag = 2 →
  markus_bags = 2 →
  markus_marbles_per_bag = 13 →
  (markus_bags * markus_marbles_per_bag) - (mara_bags * mara_marbles_per_bag) = 2 :=
by
  intros
  sorry

end markus_more_marbles_than_mara_l198_198729


namespace part_a_part_b_l198_198256

def bright (n : ℕ) := ∃ a b : ℕ, n = a^2 + b^3

theorem part_a (r s : ℕ) (h₀ : r > 0) (h₁ : s > 0) : 
  ∃ᶠ n in at_top, bright (r + n) ∧ bright (s + n) := 
by sorry

theorem part_b (r s : ℕ) (h₀ : r > 0) (h₁ : s > 0) : 
  ∃ᶠ m in at_top, bright (r * m) ∧ bright (s * m) := 
by sorry

end part_a_part_b_l198_198256


namespace find_DF_l198_198360

theorem find_DF (D E F M : Point) (DE EF DM DF : ℝ)
  (h1 : DE = 7)
  (h2 : EF = 10)
  (h3 : DM = 5)
  (h4 : midpoint F E M)
  (h5 : M = circumcenter D E F) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l198_198360


namespace find_natural_numbers_l198_198985

-- Problem statement: Find all natural numbers x, y, z such that 3^x + 4^y = 5^z
theorem find_natural_numbers (x y z : ℕ) (h : 3^x + 4^y = 5^z) : x = 2 ∧ y = 2 ∧ z = 2 :=
sorry

end find_natural_numbers_l198_198985


namespace fraction_to_decimal_l198_198316

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198316


namespace remainder_div_252_l198_198468

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l198_198468


namespace fraction_to_decimal_l198_198287

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l198_198287


namespace value_of_expression_when_x_eq_4_l198_198070

theorem value_of_expression_when_x_eq_4 : (3 * 4 + 4)^2 = 256 := by
  sorry

end value_of_expression_when_x_eq_4_l198_198070


namespace length_real_axis_hyperbola_l198_198141

theorem length_real_axis_hyperbola :
  (∃ (C : ℝ → ℝ → Prop) (a b : ℝ), (a > 0) ∧ (b > 0) ∧ 
    (∀ x y : ℝ, C x y = ((x ^ 2) / a ^ 2 - (y ^ 2) / b ^ 2 = 1)) ∧
      (∀ x y : ℝ, ((x ^ 2) / 9 - (y ^ 2) / 16 = 1) → ((x ^ 2) / a ^ 2 - (y ^ 2) / b ^ 2 = 1)) ∧
      C (-3) (2 * Real.sqrt 3)) →
  2 * (3 / 2) = 3 :=
by {
  sorry
}

end length_real_axis_hyperbola_l198_198141


namespace car_distribution_l198_198622

theorem car_distribution :
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let total_distributed_first_three := first_supplier + second_supplier + third_supplier
  let remaining_cars := total_cars - total_distributed_first_three
  let fourth_supplier := remaining_cars / 2
  let fifth_supplier := remaining_cars / 2
  fourth_supplier = 325000 :=
by
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let total_distributed_first_three := first_supplier + second_supplier + third_supplier
  let remaining_cars := total_cars - total_distributed_first_three
  let fourth_supplier := remaining_cars / 2
  let fifth_supplier := remaining_cars / 2
  sorry

end car_distribution_l198_198622


namespace both_pipes_opened_together_for_2_minutes_l198_198768

noncomputable def fill_time (t : ℝ) : Prop :=
  let rate_p := 1 / 12
  let rate_q := 1 / 15
  let combined_rate := rate_p + rate_q
  let work_done_by_p_q := combined_rate * t
  let work_done_by_q := rate_q * 10.5
  work_done_by_p_q + work_done_by_q = 1

theorem both_pipes_opened_together_for_2_minutes : ∃ t : ℝ, fill_time t ∧ t = 2 :=
by
  use 2
  unfold fill_time
  sorry

end both_pipes_opened_together_for_2_minutes_l198_198768


namespace trajectory_equation_l198_198588

-- Definitions and conditions
noncomputable def tangent_to_x_axis (M : ℝ × ℝ) := M.snd = 0
noncomputable def internally_tangent (M : ℝ × ℝ) := ∃ (r : ℝ), 0 < r ∧ M.1^2 + (M.2 - r)^2 = 4

-- The theorem stating the proof problem
theorem trajectory_equation (M : ℝ × ℝ) (h_tangent : tangent_to_x_axis M) (h_internal_tangent : internally_tangent M) :
  (∃ y : ℝ, 0 < y ∧ y ≤ 1 ∧ M.fst^2 = 4 * (y - 1)) :=
sorry

end trajectory_equation_l198_198588


namespace maximum_value_of_expression_l198_198556

noncomputable def maxValue (x y z : ℝ) : ℝ :=
(x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2)

theorem maximum_value_of_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  maxValue x y z ≤ 243 / 16 :=
sorry

end maximum_value_of_expression_l198_198556


namespace cost_of_snake_toy_l198_198979

-- Given conditions
def cost_of_cage : ℝ := 14.54
def dollar_bill_found : ℝ := 1.00
def total_cost : ℝ := 26.30

-- Theorem to find the cost of the snake toy
theorem cost_of_snake_toy : 
  (total_cost + dollar_bill_found - cost_of_cage) = 12.76 := 
  by sorry

end cost_of_snake_toy_l198_198979


namespace border_area_is_correct_l198_198788

def framed_area (height width border: ℝ) : ℝ :=
  (height + 2 * border) * (width + 2 * border)

def photograph_area (height width: ℝ) : ℝ :=
  height * width

theorem border_area_is_correct (h w b : ℝ) (h6 : h = 6) (w8 : w = 8) (b3 : b = 3) :
  (framed_area h w b - photograph_area h w) = 120 := by
  sorry

end border_area_is_correct_l198_198788


namespace probability_sum_odd_l198_198228

open_locale big_operators
open finset

-- Define the finset of numbers
def numbers : finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define what it means for a sequence to be an odd position product
def odd_product (x y z : ℕ) : Prop := odd (x * y * z)

-- Define what it means for a sequence to be an even position product
def even_product (x y z : ℕ) : Prop := ¬odd (x * y * z)

-- Define what it means for a product of two sequences to sum to an odd number
def sum_is_odd (a b c d e f : ℕ) : Prop := odd ((a * b * c) + (d * e * f))

-- Probability calculation setup
def favorable_outcomes : finset (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) :=
  {t ∈ numbers ×ˢ numbers ×ˢ numbers ×ˢ numbers ×ˢ numbers ×ˢ numbers |
    sum_is_odd t.1.1 t.1.2 t.1.3 t.2.1 t.2.2 t.2.3}

def total_outcomes : finset (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) :=
  numbers ×ˢ numbers ×ˢ numbers ×ˢ numbers ×ˢ numbers ×ˢ numbers

theorem probability_sum_odd :
  probability (favorable_outcomes : finset (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ)) = 1 / 10 :=
begin
  sorry
end

end probability_sum_odd_l198_198228


namespace infinite_integer_solutions_iff_l198_198570

theorem infinite_integer_solutions_iff
  (a b c d : ℤ) :
  (∃ inf_int_sol : (ℤ → ℤ) → Prop, ∀ (f : (ℤ → ℤ)), inf_int_sol f) ↔ (a^2 - 4*b = c^2 - 4*d) :=
by
  sorry

end infinite_integer_solutions_iff_l198_198570


namespace johns_total_earnings_per_week_l198_198024

def small_crab_baskets_monday := 3
def medium_crab_baskets_monday := 2
def large_crab_baskets_thursday := 4
def jumbo_crab_baskets_thursday := 1

def crabs_per_small_basket := 4
def crabs_per_medium_basket := 3
def crabs_per_large_basket := 5
def crabs_per_jumbo_basket := 2

def price_per_small_crab := 3
def price_per_medium_crab := 4
def price_per_large_crab := 5
def price_per_jumbo_crab := 7

def total_weekly_earnings :=
  (small_crab_baskets_monday * crabs_per_small_basket * price_per_small_crab) +
  (medium_crab_baskets_monday * crabs_per_medium_basket * price_per_medium_crab) +
  (large_crab_baskets_thursday * crabs_per_large_basket * price_per_large_crab) +
  (jumbo_crab_baskets_thursday * crabs_per_jumbo_basket * price_per_jumbo_crab)

theorem johns_total_earnings_per_week : total_weekly_earnings = 174 :=
by sorry

end johns_total_earnings_per_week_l198_198024


namespace first_house_bottles_l198_198793

theorem first_house_bottles (total_bottles : ℕ) 
  (cider_only : ℕ) (beer_only : ℕ) (half : ℕ → ℕ)
  (mixture : ℕ)
  (half_cider_bottles : ℕ)
  (half_beer_bottles : ℕ)
  (half_mixture_bottles : ℕ) : 
  total_bottles = 180 →
  cider_only = 40 →
  beer_only = 80 →
  mixture = total_bottles - (cider_only + beer_only) →
  half c = c / 2 →
  half_cider_bottles = half cider_only →
  half_beer_bottles = half beer_only →
  half_mixture_bottles = half mixture →
  half_cider_bottles + half_beer_bottles + half_mixture_bottles = 90 :=
by
  intros h_tot h_cid h_beer h_mix h_half half_cid half_beer half_mix
  sorry

end first_house_bottles_l198_198793


namespace lucas_payment_l198_198908

noncomputable def payment (windows_per_floor : ℕ) (floors : ℕ) (days : ℕ) 
  (earn_per_window : ℝ) (delay_penalty : ℝ) (period : ℕ) : ℝ :=
  let total_windows := windows_per_floor * floors
  let earnings := total_windows * earn_per_window
  let penalty_periods := days / period
  let total_penalty := penalty_periods * delay_penalty
  earnings - total_penalty

theorem lucas_payment :
  payment 3 3 6 2 1 3 = 16 := by
  sorry

end lucas_payment_l198_198908


namespace seans_sum_divided_by_julies_sum_l198_198381

theorem seans_sum_divided_by_julies_sum : 
  let seans_sum := 2 * (∑ k in Finset.range 301, k)
  let julies_sum := ∑ k in Finset.range 301, k
  seans_sum / julies_sum = 2 :=
by 
  sorry

end seans_sum_divided_by_julies_sum_l198_198381


namespace cost_each_side_is_56_l198_198009

-- Define the total cost and number of sides
def total_cost : ℕ := 224
def number_of_sides : ℕ := 4

-- Define the cost per side as the division of total cost by number of sides
def cost_per_side : ℕ := total_cost / number_of_sides

-- The theorem stating the cost per side is 56
theorem cost_each_side_is_56 : cost_per_side = 56 :=
by
  -- Proof would go here
  sorry

end cost_each_side_is_56_l198_198009


namespace remy_gallons_used_l198_198202

def roman_usage (R : ℕ) : Prop := R + (3 * R + 1) = 33

def remy_usage (R : ℕ) (Remy : ℕ) : Prop := Remy = 3 * R + 1

theorem remy_gallons_used :
  ∃ R Remy : ℕ, roman_usage R ∧ remy_usage R Remy ∧ Remy = 25 :=
  by
    sorry

end remy_gallons_used_l198_198202


namespace bruce_purchased_mangoes_l198_198267

-- Condition definitions
def cost_of_grapes (k_gra kg_cost_gra : ℕ) : ℕ := k_gra * kg_cost_gra
def amount_spent_on_mangoes (total_paid cost_gra : ℕ) : ℕ := total_paid - cost_gra
def quantity_of_mangoes (total_amt_mangoes rate_per_kg_mangoes : ℕ) : ℕ := total_amt_mangoes / rate_per_kg_mangoes

-- Parameters
variable (k_gra rate_per_kg_gra rate_per_kg_mangoes total_paid : ℕ)
variable (kg_gra_total_amt spent_amt_mangoes_qty : ℕ)

-- Given values
axiom A1 : k_gra = 7
axiom A2 : rate_per_kg_gra = 70
axiom A3 : rate_per_kg_mangoes = 55
axiom A4 : total_paid = 985

-- Calculations based on conditions
axiom H1 : cost_of_grapes k_gra rate_per_kg_gra = kg_gra_total_amt
axiom H2 : amount_spent_on_mangoes total_paid kg_gra_total_amt = spent_amt_mangoes_qty
axiom H3 : quantity_of_mangoes spent_amt_mangoes_qty rate_per_kg_mangoes = 9

-- Proof statement to be proven
theorem bruce_purchased_mangoes : quantity_of_mangoes spent_amt_mangoes_qty rate_per_kg_mangoes = 9 := sorry

end bruce_purchased_mangoes_l198_198267


namespace changed_answers_percentage_l198_198257

variables (n : ℕ) (a b c d : ℕ)

theorem changed_answers_percentage (h1 : a + b + c + d = 100)
  (h2 : a + d + c = 50)
  (h3 : a + c = 60)
  (h4 : b + d = 40) :
  10 ≤ c + d ∧ c + d ≤ 90 :=
  by sorry

end changed_answers_percentage_l198_198257


namespace fractional_exponent_representation_of_sqrt_l198_198223

theorem fractional_exponent_representation_of_sqrt (a : ℝ) : 
  Real.sqrt (a * 3 * a * Real.sqrt a) = a ^ (3 / 4) := 
sorry

end fractional_exponent_representation_of_sqrt_l198_198223


namespace largest_initial_number_l198_198884

theorem largest_initial_number : ∃ n : ℕ, (n + 5 ∑ k : ℕ, k ≠ 0 ∧ ¬ (n % k = 0)) = 200 ∧ n = 189 :=
begin
  sorry
end

end largest_initial_number_l198_198884


namespace least_upper_bound_neg_expression_l198_198821

noncomputable def least_upper_bound : ℝ :=
  - (9 / 2)

theorem least_upper_bound_neg_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  ∃ M, M = least_upper_bound ∧
  ∀ x, (∀ a b, 0 < a → 0 < b → a + b = 1 → x ≤ - (1 / (2 * a)) - (2 / b)) ↔ x ≤ M :=
sorry

end least_upper_bound_neg_expression_l198_198821


namespace _l198_198470

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l198_198470


namespace quadratic_discriminant_l198_198116

def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

theorem quadratic_discriminant : discriminant 5 (5 + 1/5) (1/5) = 576 / 25 := by
  sorry

end quadratic_discriminant_l198_198116


namespace rectangle_area_excluding_hole_l198_198456

theorem rectangle_area_excluding_hole (x : ℝ) (h : x > 5 / 3) :
  let A_large := (2 * x + 4) * (x + 7)
  let A_hole := (x + 2) * (3 * x - 5)
  A_large - A_hole = -x^2 + 17 * x + 38 :=
by
  let A_large := (2 * x + 4) * (x + 7)
  let A_hole := (x + 2) * (3 * x - 5)
  sorry

end rectangle_area_excluding_hole_l198_198456


namespace supplies_total_cost_l198_198263

-- Definitions based on conditions in a)
def cost_of_bow : ℕ := 5
def cost_of_vinegar : ℕ := 2
def cost_of_baking_soda : ℕ := 1
def students_count : ℕ := 23

-- The main theorem to prove
theorem supplies_total_cost :
  cost_of_bow * students_count + cost_of_vinegar * students_count + cost_of_baking_soda * students_count = 184 :=
by
  sorry

end supplies_total_cost_l198_198263


namespace remainder_div_13_l198_198244

theorem remainder_div_13 {k : ℤ} (N : ℤ) (h : N = 39 * k + 20) : N % 13 = 7 :=
by
  sorry

end remainder_div_13_l198_198244


namespace area_of_gray_region_is_27pi_l198_198175

-- Define the conditions
def concentric_circles (inner_radius outer_radius : ℝ) :=
  2 * inner_radius = outer_radius

def width_of_gray_region (inner_radius outer_radius width : ℝ) :=
  width = outer_radius - inner_radius

-- Define the proof problem
theorem area_of_gray_region_is_27pi
(inner_radius outer_radius : ℝ) 
(h1 : concentric_circles inner_radius outer_radius)
(h2 : width_of_gray_region inner_radius outer_radius 3) :
π * outer_radius^2 - π * inner_radius^2 = 27 * π :=
by
  -- Proof goes here, but it is not required as per instructions
  sorry

end area_of_gray_region_is_27pi_l198_198175


namespace find_w_l198_198212

theorem find_w (p q r u v w : ℝ)
  (h₁ : (x : ℝ) → x^3 - 6 * x^2 + 11 * x + 10 = (x - p) * (x - q) * (x - r))
  (h₂ : (x : ℝ) → x^3 + u * x^2 + v * x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p)))
  (h₃ : p + q + r = 6) :
  w = 80 := sorry

end find_w_l198_198212


namespace circle_equation_line_equation_l198_198019

noncomputable def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 * x + 6 * y = 0

noncomputable def point_O : ℝ × ℝ := (0, 0)
noncomputable def point_A : ℝ × ℝ := (1, 1)
noncomputable def point_B : ℝ × ℝ := (4, 2)

theorem circle_equation :
  circle_C point_O.1 point_O.2 ∧
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 :=
by sorry

noncomputable def line_l_case1 (x : ℝ) : Prop :=
  x = 3 / 2

noncomputable def line_l_case2 (x y : ℝ) : Prop :=
  8 * x + 6 * y - 39 = 0

noncomputable def center_C : ℝ × ℝ := (4, -3)
noncomputable def radius_C : ℝ := 5

noncomputable def point_through_l : ℝ × ℝ := (3 / 2, 9 / 2)

theorem line_equation : 
(∀ (M N : ℝ × ℝ), circle_C M.1 M.2 ∧ circle_C N.1 N.2 → ∃ C_slave : Prop, 
(C_slave → 
((line_l_case1 (point_through_l.1)) ∨ 
(line_l_case2 point_through_l.1 point_through_l.2)))) :=
by sorry

end circle_equation_line_equation_l198_198019


namespace max_initial_number_l198_198874

noncomputable def verify_addition (n x : ℕ) : Prop := 
  ∀ (a : ℕ), (a ∣ n) → (a ≠ 1) → (n + a = x) → False

theorem max_initial_number :
  ∃ (n : ℕ), 
  (∀ (a1 a2 a3 a4 a5 : ℕ), 
    verify_addition n a1 ∧ verify_addition (n + a1) a2 ∧
    verify_addition (n + a1 + a2) a3 ∧ verify_addition (n + a1 + a2 + a3) a4 ∧
    verify_addition (n + a1 + a2 + a3 + a4) a5 ∧
    (n + a1 + a2 + a3 + a4 + a5 = 200)) ∧
  (∀ m : ℕ, 
    (∃ (a1 a2 a3 a4 a5 : ℕ), 
      verify_addition m a1 ∧ verify_addition (m + a1) a2 ∧
      verify_addition (m + a1 + a2) a3 ∧ verify_addition (m + a1 + a2 + a3) a4 ∧
      verify_addition (m + a1 + a2 + a3 + a4) a5 ∧
      (m + a1 + a2 + a3 + a4 + a5 = 200)) →
    m ≤ 189)
: ∃ n, n = 189 := by
  sorry

end max_initial_number_l198_198874


namespace tangent_y_intercept_range_l198_198093

theorem tangent_y_intercept_range :
  ∀ (x₀ : ℝ), (∃ y₀ : ℝ, y₀ = Real.exp x₀ ∧ (∃ m : ℝ, m = Real.exp x₀ ∧ ∃ b : ℝ, b = Real.exp x₀ * (1 - x₀) ∧ b < 0)) → x₀ > 1 := by
  sorry

end tangent_y_intercept_range_l198_198093


namespace frank_reading_days_l198_198830

-- Define the parameters
def pages_weekdays : ℚ := 5.7
def pages_weekends : ℚ := 9.5
def total_pages : ℚ := 576
def pages_per_week : ℚ := (pages_weekdays * 5) + (pages_weekends * 2)

-- Define the property to be proved
theorem frank_reading_days : 
  (total_pages / pages_per_week).floor * 7 + 
  (total_pages - (total_pages / pages_per_week).floor * pages_per_week) / pages_weekdays 
  = 85 := 
  by
    sorry

end frank_reading_days_l198_198830


namespace cube_sum_l198_198677

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cube_sum_l198_198677


namespace simplify_expression_l198_198742

open Real

theorem simplify_expression (x : ℝ) (hx : 0 < x) : Real.sqrt (Real.sqrt (x^3 * sqrt (x^5))) = x^(11/8) :=
sorry

end simplify_expression_l198_198742


namespace range_of_a_l198_198934

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 - a * x + 1 > 0) ↔ -2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2 :=
by
  sorry

end range_of_a_l198_198934


namespace yura_finish_date_l198_198704

theorem yura_finish_date :
  (∃ z : ℕ, 3 * z = 45 ∧ (∑ i in list.range (z + 2) \ 0, (z + 1 - i)) = 91 ∧ nat.find (λ n, (∑ i in list.range (n+1), (z + 1 - i)) >= 91) + 6 = 12) :=
by
  sorry

end yura_finish_date_l198_198704


namespace seating_arrangements_count_l198_198396

-- Definitions for conditions
def Seats := Finset.range 18

inductive Planet
| Martian
| Venusian
| Earthling

open Planet

def Arrangement := (Seats → Planet)

def fixed_seats (arr : Arrangement) : Prop :=
  arr 0 = Martian ∧ arr 17 = Earthling

def no_E_left_of_M (arr : Arrangement) : Prop :=
  ∀ i, arr i = Earthling → arr ((i - 1) % 18) ≠ Martian

def no_M_left_of_V (arr : Arrangement) : Prop :=
  ∀ i, arr i = Martian → arr ((i - 1) % 18) ≠ Venusian

def no_V_left_of_E (arr : Arrangement) : Prop :=
  ∀ i, arr i = Venusian → arr ((i - 1) % 18) ≠ Earthling

def no_more_than_two_consecutive (arr : Arrangement) : Prop :=
  ∀ i p, arr i = p → arr ((i + 1) % 18) = p → arr ((i + 2) % 18) ≠ p

-- Combining all conditions
def valid_arrangement (arr : Arrangement) : Prop :=
  fixed_seats arr ∧ 
  no_E_left_of_M arr ∧
  no_M_left_of_V arr ∧
  no_V_left_of_E arr ∧
  no_more_than_two_consecutive arr

theorem seating_arrangements_count : ∃ n, n = 27 ∧ 
  (Finset.filter valid_arrangement (Finset.univ : Finset Arrangement)).card = n :=
begin
  sorry
end

end seating_arrangements_count_l198_198396


namespace minimum_dot_product_l198_198137

-- Define the points A and B
def A : ℝ × ℝ × ℝ := (1, 2, 0)
def B : ℝ × ℝ × ℝ := (0, 1, -1)

-- Define the vector AP
def vector_AP (x : ℝ) := (x - 1, -2, 0)

-- Define the vector BP
def vector_BP (x : ℝ) := (x, -1, 1)

-- Define the dot product of vector AP and vector BP
def dot_product (x : ℝ) : ℝ := (x - 1) * x + (-2) * (-1) + 0 * 1

-- State the theorem
theorem minimum_dot_product : ∃ x : ℝ, dot_product x = (x - 1) * x + 2 ∧ 
  (∀ y : ℝ, dot_product y ≥ dot_product (1/2)) := 
sorry

end minimum_dot_product_l198_198137


namespace problem_l198_198346

variable (m n : ℝ)
variable (h1 : m + n = -1994)
variable (h2 : m * n = 7)

theorem problem (m n : ℝ) (h1 : m + n = -1994) (h2 : m * n = 7) : 
  (m^2 + 1993 * m + 6) * (n^2 + 1995 * n + 8) = 1986 := 
by
  sorry

end problem_l198_198346


namespace solve_y_percentage_l198_198679

noncomputable def y_percentage (x y : ℝ) : ℝ :=
  100 * y / x

theorem solve_y_percentage (x y : ℝ) (h : 0.20 * (x - y) = 0.14 * (x + y)) :
  y_percentage x y = 300 / 17 :=
by
  sorry

end solve_y_percentage_l198_198679


namespace altitude_eqn_equidistant_eqn_l198_198145

-- Define the points A, B, and C
def A : (ℝ × ℝ) := (1, 1)
def B : (ℝ × ℝ) := (-1, 3)
def C : (ℝ × ℝ) := (3, 4)

-- Definition of a line in the form Ax + By + C = 0
structure Line :=
  (A B C : ℝ)
  (non_zero : A ≠ 0 ∨ B ≠ 0)

-- Equation of line l1 (altitude to side BC)
def l1 : Line := { A := 4, B := 1, C := -5, non_zero := Or.inl (by norm_num) }

-- Equation of line l2 (passing through C, equidistant from A and B), two possible values
def l2a : Line := { A := 1, B := 1, C := -7, non_zero := Or.inl (by norm_num) }
def l2b : Line := { A := 2, B := -3, C := 6, non_zero := Or.inl (by norm_num) }

-- Prove the equations for l1 and l2 are correct given the points A, B, and C
theorem altitude_eqn (h : A = (1, 1) ∧ B = (-1, 3) ∧ C = (3, 4)) :
  l1 = { A := 4, B := 1, C := -5, non_zero := Or.inl (by norm_num) } := sorry

theorem equidistant_eqn (h : A = (1, 1) ∧ B = (-1, 3) ∧ C = (3, 4)) :
  l2a = { A := 1, B := 1, C := -7, non_zero := Or.inl (by norm_num) } ∨
  l2b = { A := 2, B := -3, C := 6, non_zero := Or.inl (by norm_num) } := sorry

end altitude_eqn_equidistant_eqn_l198_198145


namespace number_of_buses_l198_198784

-- Define the conditions
def vans : ℕ := 9
def people_per_van : ℕ := 8
def people_per_bus : ℕ := 27
def total_people : ℕ := 342

-- Translate the mathematical proof problem
theorem number_of_buses : ∃ buses : ℕ, (vans * people_per_van + buses * people_per_bus = total_people) ∧ (buses = 10) :=
by
  -- calculations to prove the theorem
  sorry

end number_of_buses_l198_198784


namespace electric_energy_consumption_l198_198554

def power_rating_fan : ℕ := 75
def hours_per_day : ℕ := 8
def days_per_month : ℕ := 30
def watts_to_kWh : ℕ := 1000

theorem electric_energy_consumption : power_rating_fan * hours_per_day * days_per_month / watts_to_kWh = 18 := by
  sorry

end electric_energy_consumption_l198_198554


namespace lana_eats_fewer_candies_l198_198733

-- Definitions based on conditions
def canEatNellie : ℕ := 12
def canEatJacob : ℕ := canEatNellie / 2
def candiesBeforeLanaCries : ℕ := 6 -- This is the derived answer for Lana
def initialCandies : ℕ := 30
def remainingCandies : ℕ := 3 * 3 -- After division, each gets 3 candies and they are 3 people

-- Statement to prove how many fewer candies Lana can eat compared to Jacob
theorem lana_eats_fewer_candies :
  canEatJacob = 6 → 
  (initialCandies - remainingCandies = 12 + canEatJacob + candiesBeforeLanaCries) →
  canEatJacob - candiesBeforeLanaCries = 3 :=
by
  intros hJacobEats hCandiesAte
  sorry

end lana_eats_fewer_candies_l198_198733


namespace domain_of_y_l198_198220

noncomputable def domain_of_function (x : ℝ) : Bool :=
  x < 0 ∧ x ≠ -1

theorem domain_of_y :
  {x : ℝ | (∃ y, y = (x + 1) ^ 0 / Real.sqrt (|x| - x)) } =
  {x : ℝ | domain_of_function x} :=
by
  sorry

end domain_of_y_l198_198220


namespace red_chips_drawn_first_probability_l198_198255

noncomputable def prob_red_chips_drawn_first : ℚ :=
  let total_chips := {chip | chip ∈ {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}}
  in let favorable_arrangements := {arrangement | arrangement.take(3).count(= red_chip) = 3}
  in favorable_arrangements.card / total_chips.card

theorem red_chips_drawn_first_probability :
  prob_red_chips_drawn_first = 1 / 2 :=
sorry

end red_chips_drawn_first_probability_l198_198255


namespace third_speed_is_9_kmph_l198_198616

/-- Problem Statement: Given the total travel time, total distance, and two speeds, 
    prove that the third speed is 9 km/hr when distances are equal. -/
theorem third_speed_is_9_kmph (t : ℕ) (d_total : ℕ) (v1 v2 : ℕ) (d1 d2 d3 : ℕ) 
(h_t : t = 11)
(h_d_total : d_total = 900)
(h_v1 : v1 = 3)
(h_v2 : v2 = 6)
(h_d_eq : d1 = 300 ∧ d2 = 300 ∧ d3 = 300)
(h_sum_t : d1 / (v1 * 1000 / 60) + d2 / (v2 * 1000 / 60) + d3 / (v3 * 1000 / 60) = t) 
: (v3 = 9) :=
by 
  sorry

end third_speed_is_9_kmph_l198_198616


namespace total_cost_supplies_l198_198261

-- Definitions based on conditions
def cost_bow : ℕ := 5
def cost_vinegar : ℕ := 2
def cost_baking_soda : ℕ := 1
def cost_per_student : ℕ := cost_bow + cost_vinegar + cost_baking_soda
def number_of_students : ℕ := 23

-- Statement to be proven
theorem total_cost_supplies : cost_per_student * number_of_students = 184 := by
  sorry

end total_cost_supplies_l198_198261


namespace division_and_subtraction_l198_198632

theorem division_and_subtraction :
  (12 : ℚ) / (1 / 6) - (1 / 3) = 215 / 3 :=
by
  sorry

end division_and_subtraction_l198_198632


namespace chess_match_probability_l198_198431

theorem chess_match_probability (p : ℝ) (h0 : 0 < p) (h1 : p < 1) :
  (3 * p^3 * (1 - p) ≤ 6 * p^3 * (1 - p)^2) → (p ≤ 1/2) :=
by
  sorry

end chess_match_probability_l198_198431


namespace calculation_eq_minus_one_l198_198268

noncomputable def calculation : ℝ :=
  (-1)^(53 : ℤ) + 3^((2^3 + 5^2 - 7^2) : ℤ)

theorem calculation_eq_minus_one : calculation = -1 := 
by 
  sorry

end calculation_eq_minus_one_l198_198268


namespace parabola_intersect_line_segment_range_l198_198334

theorem parabola_intersect_line_segment_range (a : ℝ) :
  (∀ (x : ℝ), y = a * x^2 - 3 * x + 1) →
  (∀ (x1 y1 x2 y2 x0 : ℝ), (x0 ∈ {x | ∃ y, y = a * x^2 - 3 * x + 1}) →
                         (x1 - x0).abs > (x2 - x0).abs →
                         ∃ y1 y2, y1 = a * x1^2 - 3 * x1 + 1 ∧ y2 = a * x2^2 - 3 * x2 + 1 → y1 > y2) →
  (∃ (x : ℝ), x ∈ Icc (-1 : ℝ) (3 : ℝ) → y = x - 1) →
  (∃ (x : ℝ), y = a * x^2 - 3 * x + 1 = (x - 1)) →
  (\(a \ge 10/9 ∧ a < 2\)) :=
by
  sorry

end parabola_intersect_line_segment_range_l198_198334


namespace fraction_to_decimal_l198_198298

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198298


namespace addition_of_fractions_l198_198972

theorem addition_of_fractions : (6/7 : ℚ) + (7/9 : ℚ) = 103/63 := by
  sorry

end addition_of_fractions_l198_198972


namespace count_bases_for_last_digit_l198_198003

theorem count_bases_for_last_digit (n : ℕ) : n = 729 → ∃ S : Finset ℕ, S.card = 2 ∧ ∀ b ∈ S, 2 ≤ b ∧ b ≤ 10 ∧ (n - 5) % b = 0 :=
by
  sorry

end count_bases_for_last_digit_l198_198003


namespace smallest_resolvable_debt_l198_198235

theorem smallest_resolvable_debt (p g : ℤ) : 
  ∃ p g : ℤ, (500 * p + 350 * g = 50) ∧ ∀ D > 0, (∃ p g : ℤ, 500 * p + 350 * g = D) → 50 ≤ D :=
by {
  sorry
}

end smallest_resolvable_debt_l198_198235


namespace count_at_least_three_adjacent_chairs_l198_198417

noncomputable def count_adjacent_subsets (n : ℕ) (chairs : set (fin n)) : ℕ :=
  -- Add a function definition that counts the number of subsets with at least three adjacent chairs
  sorry

theorem count_at_least_three_adjacent_chairs :
  count_adjacent_subsets 12 (set.univ : set (fin 12)) = 2010 :=
sorry

end count_at_least_three_adjacent_chairs_l198_198417


namespace value_of_f_neg_a_l198_198663

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end value_of_f_neg_a_l198_198663


namespace diagonal_inequality_l198_198362

theorem diagonal_inequality (A B C D : ℝ × ℝ) (h1 : A.1 = 0) (h2 : B.1 = 0) (h3 : C.2 = 0) (h4 : D.2 = 0) 
  (ha : A.2 < B.2) (hd : D.1 < C.1) : 
  (Real.sqrt (A.2^2 + C.1^2)) * (Real.sqrt (B.2^2 + D.1^2)) > (Real.sqrt (A.2^2 + D.1^2)) * (Real.sqrt (B.2^2 + C.1^2)) :=
sorry

end diagonal_inequality_l198_198362


namespace part1_part2_l198_198521

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|
def h (a x : ℝ) : ℝ := |f x| + g a x

theorem part1 (a : ℝ) : (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ -2 :=
  sorry

theorem part2 (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → h a x ≤ if a ≥ 0 then 3*a + 3 else if -3 ≤ a then a + 3 else 0) :=
  sorry

end part1_part2_l198_198521


namespace fraction_to_decimal_l198_198296

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198296


namespace find_x_l198_198125

variables {x : ℝ}
def vector_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_x
  (h1 : (6, 1) = (6, 1))
  (h2 : (x, -3) = (x, -3))
  (h3 : vector_parallel (6, 1) (x, -3)) :
  x = -18 := by
  sorry

end find_x_l198_198125


namespace fraction_value_l198_198103

theorem fraction_value :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) / 
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400)) = 244.375 :=
by
  -- The proof would be provided here, but we are skipping it as per the instructions.
  sorry

end fraction_value_l198_198103


namespace transformed_sequence_has_large_element_l198_198950

noncomputable def transformed_value (a : Fin 25 → ℤ) (i : Fin 25) : ℤ :=
  a i + a ((i + 1) % 25)

noncomputable def perform_transformation (a : Fin 25 → ℤ) (n : ℕ) : Fin 25 → ℤ :=
  if n = 0 then a
  else perform_transformation (fun i => transformed_value a i) (n - 1)

theorem transformed_sequence_has_large_element :
  ∀ a : Fin 25 → ℤ,
    (∀ i : Fin 13, a i = 1) →
    (∀ i : Fin 12, a (i + 13) = -1) →
    ∃ i : Fin 25, perform_transformation a 100 i > 10^20 :=
by
  sorry

end transformed_sequence_has_large_element_l198_198950


namespace yura_finishes_problems_l198_198708

theorem yura_finishes_problems 
  (total_problems : ℕ) (start_date : ℕ) (remaining_problems : ℕ) (z : ℕ)
  (H_total : total_problems = 91)
  (H_start_date : start_date = 6)
  (H_remaining : remaining_problems = 46)
  (H_solved_by_8th : (z + 1) + z + (z - 1) = total_problems - remaining_problems) :
  ∃ finish_date, finish_date = 12 :=
by
  use 12
  sorry

end yura_finishes_problems_l198_198708


namespace math_problem_l198_198270

theorem math_problem :
  (-1 : ℝ)^(53) + 3^(2^3 + 5^2 - 7^2) = -1 + (1 / 3^(16)) :=
by
  sorry

end math_problem_l198_198270


namespace greatest_identical_snack_bags_l198_198391

-- Defining the quantities of each type of snack
def granola_bars : Nat := 24
def dried_fruit : Nat := 36
def nuts : Nat := 60

-- Statement of the problem: greatest number of identical snack bags Serena can make without any food left over.
theorem greatest_identical_snack_bags :
  Nat.gcd (Nat.gcd granola_bars dried_fruit) nuts = 12 :=
sorry

end greatest_identical_snack_bags_l198_198391


namespace book_price_l198_198905

theorem book_price (x : ℕ) : 
  9 * x ≤ 1100 ∧ 13 * x ≤ 1500 → x = 123 :=
sorry

end book_price_l198_198905


namespace red_grapes_count_l198_198868

theorem red_grapes_count (G : ℕ) (total_fruit : ℕ) (red_grapes : ℕ) (raspberries : ℕ)
  (h1 : red_grapes = 3 * G + 7) 
  (h2 : raspberries = G - 5) 
  (h3 : total_fruit = G + red_grapes + raspberries) 
  (h4 : total_fruit = 102) : 
  red_grapes = 67 :=
by
  sorry

end red_grapes_count_l198_198868


namespace problem_1_problem_2_l198_198124

open BigOperators

-- Question 1
theorem problem_1 (a : Fin 2021 → ℝ) :
  (1 + 2 * x) ^ 2020 = ∑ i in Finset.range 2021, a i * x ^ i →
  (∑ i in Finset.range 2021, (i * a i)) = 4040 * 3 ^ 2019 :=
sorry

-- Question 2
theorem problem_2 (a : Fin 2021 → ℝ) :
  (1 - x) ^ 2020 = ∑ i in Finset.range 2021, a i * x ^ i →
  ((∑ i in Finset.range 2021, 1 / a i)) = 2021 / 1011 :=
sorry

end problem_1_problem_2_l198_198124


namespace area_of_border_l198_198963

theorem area_of_border
  (h_photo : Nat := 9)
  (w_photo : Nat := 12)
  (border_width : Nat := 3) :
  (let area_photo := h_photo * w_photo
    let h_frame := h_photo + 2 * border_width
    let w_frame := w_photo + 2 * border_width
    let area_frame := h_frame * w_frame
    let area_border := area_frame - area_photo
    area_border = 162) := 
  sorry

end area_of_border_l198_198963


namespace percent_of_b_is_50_l198_198682

variable (a b c : ℝ)

-- Conditions
def c_is_25_percent_of_a : Prop := c = 0.25 * a
def b_is_50_percent_of_a : Prop := b = 0.50 * a

-- Proof
theorem percent_of_b_is_50 :
  c_is_25_percent_of_a c a → b_is_50_percent_of_a b a → c = 0.50 * b :=
by sorry

end percent_of_b_is_50_l198_198682


namespace max_cells_intersected_10_radius_circle_l198_198600

noncomputable def max_cells_intersected_by_circle (radius : ℝ) (cell_size : ℝ) : ℕ :=
  if radius = 10 ∧ cell_size = 1 then 80 else 0

theorem max_cells_intersected_10_radius_circle :
  max_cells_intersected_by_circle 10 1 = 80 :=
sorry

end max_cells_intersected_10_radius_circle_l198_198600


namespace tetrahedron_inequality_l198_198363

theorem tetrahedron_inequality
  (a b c d h_a h_b h_c h_d V : ℝ)
  (ha : V = 1/3 * a * h_a)
  (hb : V = 1/3 * b * h_b)
  (hc : V = 1/3 * c * h_c)
  (hd : V = 1/3 * d * h_d) :
  (a + b + c + d) * (h_a + h_b + h_c + h_d) >= 48 * V := 
  by sorry

end tetrahedron_inequality_l198_198363


namespace solution_exists_l198_198743

def age_problem (S F Y : ℕ) : Prop :=
  S = 12 ∧ S = F / 3 ∧ S - Y = (F - Y) / 5 ∧ Y = 6

theorem solution_exists : ∃ (Y : ℕ), ∃ (S F : ℕ), age_problem S F Y :=
by sorry

end solution_exists_l198_198743


namespace coefficient_of_x4_in_expansion_l198_198820

theorem coefficient_of_x4_in_expansion :
  let expanded_expr := (2 - x) * (Polynomial.monomial 1 2 * x + 1)^6,
      term_x4 := expanded_expr.coeff 4
  in term_x4 = 320 := by
  sorry

end coefficient_of_x4_in_expansion_l198_198820


namespace fraction_to_decimal_l198_198306

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198306


namespace train_speed_calculation_l198_198790

noncomputable def speed_of_train_in_kmph
  (length_of_train : ℝ)
  (length_of_bridge : ℝ)
  (time_to_cross_bridge : ℝ) : ℝ :=
(length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6

theorem train_speed_calculation
  (length_of_train : ℝ)
  (length_of_bridge : ℝ)
  (time_to_cross_bridge : ℝ)
  (h_train : length_of_train = 150)
  (h_bridge : length_of_bridge = 225)
  (h_time : time_to_cross_bridge = 30) :
  speed_of_train_in_kmph length_of_train length_of_bridge time_to_cross_bridge = 45 := by
  simp [speed_of_train_in_kmph, h_train, h_bridge, h_time]
  norm_num
  sorry

end train_speed_calculation_l198_198790


namespace exist_infinite_a_l198_198248

theorem exist_infinite_a (n : ℕ) (a : ℕ) (h₁ : ∃ k : ℕ, k > 0 ∧ (n^6 + 3 * a = (n^2 + 3 * k)^3)) : 
  ∃ f : ℕ → ℕ, ∀ m : ℕ, (∃ k : ℕ, k > 0 ∧ f m = 9 * k^3 + 3 * n^2 * k * (n^2 + 3 * k)) :=
by 
  sorry

end exist_infinite_a_l198_198248


namespace part_a_part_b_l198_198044

-- Define what it means for a coloring to be valid.
def valid_coloring (n : ℕ) (colors : Fin n → Fin 3) : Prop :=
  ∀ (i : Fin n),
  ∃ j k : Fin n, 
  ((i + 1) % n = j ∧ (i + 2) % n = k ∧ colors i ≠ colors j ∧ colors i ≠ colors k ∧ colors j ≠ colors k)

-- Part (a)
theorem part_a (n : ℕ) (hn : 3 ∣ n) : ∃ (colors : Fin n → Fin 3), valid_coloring n colors :=
by sorry

-- Part (b)
theorem part_b (n : ℕ) : (∃ (colors : Fin n → Fin 3), valid_coloring n colors) → 3 ∣ n :=
by sorry

end part_a_part_b_l198_198044


namespace stickers_total_l198_198025

def karl_stickers : ℕ := 25
def ryan_stickers : ℕ := karl_stickers + 20
def ben_stickers : ℕ := ryan_stickers - 10
def total_stickers : ℕ := karl_stickers + ryan_stickers + ben_stickers

theorem stickers_total : total_stickers = 105 := by
  sorry

end stickers_total_l198_198025


namespace find_divisor_l198_198237

def div_remainder (a b r : ℕ) : Prop :=
  ∃ k : ℕ, a = k * b + r

theorem find_divisor :
  ∃ D : ℕ, (div_remainder 242 D 15) ∧ (div_remainder 698 D 27) ∧ (div_remainder (242 + 698) D 5) ∧ D = 37 := 
by
  sorry

end find_divisor_l198_198237


namespace perfect_square_trinomial_k_l198_198536

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ a : ℤ, x^2 + k*x + 25 = (x + a)^2 ∧ a^2 = 25) → (k = 10 ∨ k = -10) :=
by
  sorry

end perfect_square_trinomial_k_l198_198536


namespace find_a_of_pure_imaginary_l198_198330

noncomputable def isPureImaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = ⟨0, b⟩  -- complex number z is purely imaginary if it can be written as 0 + bi

theorem find_a_of_pure_imaginary (a : ℝ) (i : ℂ) (ha : i*i = -1) :
  isPureImaginary ((1 - i) * (a + i)) → a = -1 := by
  sorry

end find_a_of_pure_imaginary_l198_198330


namespace g_f_eval_l198_198894

def f (x : ℤ) := x^3 - 2
def g (x : ℤ) := 3 * x^2 + x + 2

theorem g_f_eval : g (f 3) = 1902 := by
  sorry

end g_f_eval_l198_198894


namespace middle_number_of_consecutive_sum_30_l198_198864

theorem middle_number_of_consecutive_sum_30 (n : ℕ) (h : n + (n + 1) + (n + 2) = 30) : n + 1 = 10 :=
by
  sorry

end middle_number_of_consecutive_sum_30_l198_198864


namespace geometric_seq_arith_seq_problem_l198_198711

theorem geometric_seq_arith_seq_problem (a : ℕ → ℝ) (q : ℝ)
  (h : ∀ n, a (n + 1) = q * a n)
  (h_q_pos : q > 0)
  (h_arith : 2 * (1/2 : ℝ) * a 2 = 3 * a 0 + 2 * a 1) :
  (a 2014 - a 2015) / (a 2016 - a 2017) = 1 / 9 := 
sorry

end geometric_seq_arith_seq_problem_l198_198711


namespace books_left_l198_198736

namespace PaulBooksExample

-- Defining the initial conditions as given in the problem
def initial_books : ℕ := 134
def books_given : ℕ := 39
def books_sold : ℕ := 27

-- Proving that the final number of books Paul has is 68
theorem books_left : initial_books - (books_given + books_sold) = 68 := by
  sorry

end PaulBooksExample

end books_left_l198_198736


namespace ratio_of_roses_l198_198430

-- Definitions for conditions
def roses_two_days_ago : ℕ := 50
def roses_yesterday : ℕ := roses_two_days_ago + 20
def roses_total : ℕ := 220
def roses_today : ℕ := roses_total - roses_two_days_ago - roses_yesterday

-- Lean statement to prove the ratio of roses planted today to two days ago is 2
theorem ratio_of_roses :
  roses_today / roses_two_days_ago = 2 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_roses_l198_198430


namespace negation_of_p_l198_198187

def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

def A (x : ℤ) : Prop := is_odd x
def B (x : ℤ) : Prop := is_even x
def p : Prop := ∀ x, A x → B (2 * x)

theorem negation_of_p : ¬ p ↔ ∃ x, A x ∧ ¬ B (2 * x) :=
by
  -- problem statement equivalent in Lean 4
  sorry

end negation_of_p_l198_198187


namespace problem1_problem2_l198_198273

-- Problem 1
theorem problem1 : 23 + (-13) + (-17) + 8 = 1 :=
by
  sorry

-- Problem 2
theorem problem2 : - (2^3) - (1 + 0.5) / (1/3) * (-3) = 11/2 :=
by
  sorry

end problem1_problem2_l198_198273


namespace find_a_l198_198368

noncomputable def f (a x : ℝ) := a * x + 1 / Real.sqrt 2

theorem find_a (a : ℝ) (h_pos : 0 < a) (h : f a (f a (1 / Real.sqrt 2)) = f a 0) : a = 0 :=
by
  sorry

end find_a_l198_198368


namespace quadrant_606_quadrant_minus_950_same_terminal_side_quadrant_minus_97_l198_198952

theorem quadrant_606 (θ : ℝ) : θ = 606 → (180 < (θ % 360) ∧ (θ % 360) < 270) := by
  sorry

theorem quadrant_minus_950 (θ : ℝ) : θ = -950 → (90 < (θ % 360) ∧ (θ % 360) < 180) := by
  sorry

theorem same_terminal_side (α k : ℤ) : (α = -457 + k * 360) ↔ (∃ n : ℤ, α = -457 + n * 360) := by
  sorry

theorem quadrant_minus_97 (θ : ℝ) : θ = -97 → (180 < (θ % 360) ∧ (θ % 360) < 270) := by
  sorry

end quadrant_606_quadrant_minus_950_same_terminal_side_quadrant_minus_97_l198_198952


namespace intersection_A_B_l198_198364

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | ∃ (n : ℤ), (x : ℝ) = n }

theorem intersection_A_B : A ∩ B = {0, 1} := 
by
  sorry

end intersection_A_B_l198_198364


namespace p_sufficient_not_necessary_for_q_l198_198127

-- Definitions based on conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬ (∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l198_198127


namespace tan_alpha_in_third_quadrant_l198_198140

theorem tan_alpha_in_third_quadrant (α : Real) (h1 : Real.sin α = -5/13) (h2 : ∃ k : ℕ, π < α + k * 2 * π ∧ α + k * 2 * π < 3 * π) : 
  Real.tan α = 5/12 :=
sorry

end tan_alpha_in_third_quadrant_l198_198140


namespace smallest_four_digit_palindrome_div7_eq_1661_l198_198435

theorem smallest_four_digit_palindrome_div7_eq_1661 :
  ∃ (A B : ℕ), (A == 1 ∨ A == 3 ∨ A == 5 ∨ A == 7 ∨ A == 9) ∧
  (1000 ≤ 1100 * A + 11 * B ∧ 1100 * A + 11 * B < 10000) ∧
  (1100 * A + 11 * B) % 7 = 0 ∧
  (1100 * A + 11 * B) = 1661 :=
by
  sorry

end smallest_four_digit_palindrome_div7_eq_1661_l198_198435


namespace tetrahedrons_volume_proportional_l198_198567

-- Define the scenario and conditions.
variable 
  (V V' : ℝ) -- Volumes of the tetrahedrons
  (a b c a' b' c' : ℝ) -- Edge lengths emanating from vertices O and O'
  (α : ℝ) -- The angle between vectors OB and OC which is assumed to be congruent

-- Theorem statement.
theorem tetrahedrons_volume_proportional
  (congruent_trihedral_angles_at_O_and_O' : α = α) -- Condition of congruent trihedral angles
  : (V' / V) = (a' * b' * c') / (a * b * c) :=
sorry

end tetrahedrons_volume_proportional_l198_198567


namespace complex_expression_simplification_l198_198951

-- Given: i is the imaginary unit
def i := Complex.I

-- Prove that the expression simplifies to -1
theorem complex_expression_simplification : (i^3 * (i + 1)) / (i - 1) = -1 := by
  -- We are skipping the proof and adding sorry for now
  sorry

end complex_expression_simplification_l198_198951


namespace probability_sum_odd_l198_198442

theorem probability_sum_odd (balls : Finset ℕ) (odd_balls even_balls : Finset ℕ) :
  balls = Finset.range 14 ∧
  odd_balls = {1, 3, 5, 7, 9, 11, 13} ∧
  even_balls = {2, 4, 6, 8, 10, 12} ∧
  (∀ (draw : Finset ℕ), draw.card = 7 → draw ⊆ balls → 
   (finset.card (draw ∩ odd_balls) % 2 = 1 → 
    ∃ p : ℚ, p = 212 / 429)) :=
begin
  sorry
end

end probability_sum_odd_l198_198442


namespace wall_length_l198_198245

theorem wall_length (mirror_side length width : ℝ) (h1 : mirror_side = 21) (h2 : width = 28) 
  (h3 : 2 * mirror_side^2 = width * length) : length = 31.5 := by
  sorry

end wall_length_l198_198245


namespace quadratic_real_roots_l198_198508

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) →
  (m ≤ 4 ∧ m ≠ 3) :=
sorry

end quadratic_real_roots_l198_198508


namespace total_distance_of_the_race_l198_198173

-- Define the given conditions
def A_beats_B_by_56_meters_or_7_seconds : Prop :=
  ∃ D : ℕ, ∀ S_B S_A : ℕ, S_B = 8 ∧ S_A = D / 8 ∧ D = S_B * (8 + 7)

-- Define the question and correct answer
theorem total_distance_of_the_race : A_beats_B_by_56_meters_or_7_seconds → ∃ D : ℕ, D = 120 :=
by
  sorry

end total_distance_of_the_race_l198_198173


namespace swim_time_CBA_l198_198405

theorem swim_time_CBA (d t_down t_still t_upstream: ℝ) 
  (h1 : d = 1) 
  (h2 : t_down = 1 / (6 / 5))
  (h3 : t_still = 1)
  (h4 : t_upstream = (4 / 5) / 2)
  (total_time_down : (t_down + t_still) = 1)
  (total_time_up : (t_still + t_down) = 2) :
  (t_upstream * (d - (d / 5))) / 2 = 5 / 2 :=
by sorry

end swim_time_CBA_l198_198405


namespace roof_length_width_difference_l198_198232

variable (w l : ℕ)

theorem roof_length_width_difference (h1 : l = 7 * w) (h2 : l * w = 847) : l - w = 66 :=
by 
  sorry

end roof_length_width_difference_l198_198232


namespace total_feet_in_garden_l198_198055

def dogs : ℕ := 6
def ducks : ℕ := 2
def cats : ℕ := 4
def birds : ℕ := 7
def insects : ℕ := 10

def feet_per_dog : ℕ := 4
def feet_per_duck : ℕ := 2
def feet_per_cat : ℕ := 4
def feet_per_bird : ℕ := 2
def feet_per_insect : ℕ := 6

theorem total_feet_in_garden :
  dogs * feet_per_dog + 
  ducks * feet_per_duck + 
  cats * feet_per_cat + 
  birds * feet_per_bird + 
  insects * feet_per_insect = 118 := by
  sorry

end total_feet_in_garden_l198_198055


namespace _l198_198472

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l198_198472


namespace olympiad2024_sum_l198_198021

theorem olympiad2024_sum (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h_product : A * B * C = 2310) : 
  A + B + C ≤ 390 :=
sorry

end olympiad2024_sum_l198_198021


namespace fraction_to_decimal_l198_198294

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198294


namespace compare_two_sqrt_three_with_three_l198_198275

theorem compare_two_sqrt_three_with_three : 2 * Real.sqrt 3 > 3 :=
sorry

end compare_two_sqrt_three_with_three_l198_198275


namespace find_percentage_l198_198439

variable (P : ℝ)
variable (num : ℝ := 70)
variable (result : ℝ := 25)

theorem find_percentage (h : ((P / 100) * num) - 10 = result) : P = 50 := by
  sorry

end find_percentage_l198_198439


namespace clothing_price_l198_198181

theorem clothing_price
  (total_spent : ℕ)
  (num_pieces : ℕ)
  (price_piece_1 : ℕ)
  (price_piece_2 : ℕ)
  (num_remaining_pieces : ℕ)
  (total_remaining_pieces_price : ℕ)
  (price_remaining_piece : ℕ) :
  total_spent = 610 →
  num_pieces = 7 →
  price_piece_1 = 49 →
  price_piece_2 = 81 →
  num_remaining_pieces = 5 →
  total_spent = price_piece_1 + price_piece_2 + total_remaining_pieces_price →
  total_remaining_pieces_price = price_remaining_piece * num_remaining_pieces →
  price_remaining_piece = 96 :=
by
  intros h_total_spent h_num_pieces h_price_piece_1 h_price_piece_2 h_num_remaining_pieces h_total_remaining_price h_price_remaining_piece
  sorry

end clothing_price_l198_198181


namespace fraction_to_decimal_l198_198302

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l198_198302


namespace sean_divided_by_julie_is_2_l198_198389

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum as twice the sum of the first 300 natural numbers
def sean_sum : ℕ := 2 * sum_natural 300

-- Define Julie's sum as the sum of the first 300 natural numbers
def julie_sum : ℕ := sum_natural 300

-- Prove that Sean's sum divided by Julie's sum is 2
theorem sean_divided_by_julie_is_2 : sean_sum / julie_sum = 2 := by
  sorry

end sean_divided_by_julie_is_2_l198_198389


namespace squares_difference_l198_198372

theorem squares_difference (n : ℕ) (h : n > 0) : (2 * n + 1)^2 - (2 * n - 1)^2 = 8 * n := 
by 
  sorry

end squares_difference_l198_198372


namespace Adam_smiley_count_l198_198990

theorem Adam_smiley_count :
  ∃ (adam mojmir petr pavel : ℕ), adam + mojmir + petr + pavel = 52 ∧
  petr + pavel = 33 ∧ adam >= 1 ∧ mojmir >= 1 ∧ petr >= 1 ∧ pavel >= 1 ∧
  mojmir > max petr pavel ∧ adam = 1 :=
by
  sorry

end Adam_smiley_count_l198_198990


namespace range_of_f_l198_198338

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2

theorem range_of_f : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f x ∈ Set.Icc (-18 : ℝ) (2 : ℝ) :=
sorry

end range_of_f_l198_198338


namespace smallest_n_for_property_l198_198803

theorem smallest_n_for_property (n x : ℕ) (d : ℕ) (c : ℕ) 
  (hx : x = 10 * c + d) 
  (hx_prop : 10^(n-1) * d + c = 2 * x) :
  n = 18 := 
sorry

end smallest_n_for_property_l198_198803


namespace chairs_subsets_l198_198420

theorem chairs_subsets:
  let n := 12 in
  count_subsets_with_at_least_three_adjacent_chairs n = 2066 :=
sorry

end chairs_subsets_l198_198420


namespace no_valid_x_l198_198844

-- Definitions based on given conditions
variables {m n x : ℝ}
variables (hm : m > 0) (hn : n < 0)

-- Theorem statement
theorem no_valid_x (hm : m > 0) (hn : n < 0) :
  ¬ ∃ x, (x - m)^2 - (x - n)^2 = (m - n)^2 :=
by
  sorry

end no_valid_x_l198_198844


namespace fraction_to_decimal_l198_198289

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l198_198289


namespace tank_A_is_60_percent_of_tank_B_capacity_l198_198213

-- Conditions
def height_A : ℝ := 10
def circumference_A : ℝ := 6
def height_B : ℝ := 6
def circumference_B : ℝ := 10

-- Statement
theorem tank_A_is_60_percent_of_tank_B_capacity (V_A V_B : ℝ) (radius_A radius_B : ℝ)
  (hA : radius_A = circumference_A / (2 * Real.pi))
  (hB : radius_B = circumference_B / (2 * Real.pi))
  (vol_A : V_A = Real.pi * radius_A^2 * height_A)
  (vol_B : V_B = Real.pi * radius_B^2 * height_B) :
  (V_A / V_B) * 100 = 60 :=
by
  sorry

end tank_A_is_60_percent_of_tank_B_capacity_l198_198213


namespace _l198_198471

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l198_198471


namespace wire_ratio_bonnie_roark_l198_198806

-- Definitions from the conditions
def bonnie_wire_length : ℕ := 12 * 8
def bonnie_volume : ℕ := 8 ^ 3
def roark_cube_side : ℕ := 2
def roark_cube_volume : ℕ := roark_cube_side ^ 3
def num_roark_cubes : ℕ := bonnie_volume / roark_cube_volume
def roark_wire_length_per_cube : ℕ := 12 * roark_cube_side
def roark_total_wire_length : ℕ := num_roark_cubes * roark_wire_length_per_cube

-- Statement to prove
theorem wire_ratio_bonnie_roark : 
  ((bonnie_wire_length : ℚ) / roark_total_wire_length) = (1 / 16) :=
by
  sorry

end wire_ratio_bonnie_roark_l198_198806


namespace shelly_total_money_l198_198204

-- Define the conditions
def num_of_ten_dollar_bills : ℕ := 10
def num_of_five_dollar_bills : ℕ := num_of_ten_dollar_bills - 4

-- Problem statement: How much money does Shelly have in all?
theorem shelly_total_money :
  (num_of_ten_dollar_bills * 10) + (num_of_five_dollar_bills * 5) = 130 :=
by
  sorry

end shelly_total_money_l198_198204


namespace find_slower_train_speed_l198_198595

theorem find_slower_train_speed (l : ℝ) (vf : ℝ) (t : ℝ) (v_s : ℝ) 
  (h1 : l = 37.5)   -- Length of each train
  (h2 : vf = 46)   -- Speed of the faster train in km/hr
  (h3 : t = 27)    -- Time in seconds to pass the slower train
  (h4 : (2 * l) = ((46 - v_s) * (5 / 18) * 27))   -- Distance covered at relative speed
  : v_s = 36 := 
sorry

end find_slower_train_speed_l198_198595


namespace average_sleep_is_8_l198_198154

-- Define the hours of sleep for each day
def mondaySleep : ℕ := 8
def tuesdaySleep : ℕ := 7
def wednesdaySleep : ℕ := 8
def thursdaySleep : ℕ := 10
def fridaySleep : ℕ := 7

-- Calculate the total hours of sleep over the week
def totalSleep : ℕ := mondaySleep + tuesdaySleep + wednesdaySleep + thursdaySleep + fridaySleep
-- Define the total number of days
def totalDays : ℕ := 5

-- Calculate the average sleep per night
def averageSleepPerNight : ℕ := totalSleep / totalDays

-- Prove the statement
theorem average_sleep_is_8 : averageSleepPerNight = 8 := 
by
  -- All conditions are automatically taken into account as definitions
  -- Add a placeholder to skip the actual proof
  sorry

end average_sleep_is_8_l198_198154


namespace remainder_of_large_number_l198_198476

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l198_198476


namespace least_red_chips_l198_198412

/--
  There are 70 chips in a box. Each chip is either red or blue.
  If the sum of the number of red chips and twice the number of blue chips equals a prime number,
  proving that the least possible number of red chips is 69.
-/
theorem least_red_chips (r b : ℕ) (p : ℕ) (h1 : r + b = 70) (h2 : r + 2 * b = p) (hp : Nat.Prime p) :
  r = 69 :=
by
  -- Proof goes here
  sorry

end least_red_chips_l198_198412


namespace number_of_orange_marbles_l198_198938

/--
There are 24 marbles in a jar. Half are blue. There are 6 red marbles.
The rest of the marbles are orange.
Prove that the number of orange marbles is 6.
-/
theorem number_of_orange_marbles :
  ∀ (total_marbles blue_marbles red_marbles orange_marbles : ℕ),
  total_marbles = 24 → 
  blue_marbles = total_marbles / 2 →
  red_marbles = 6 → 
  orange_marbles = total_marbles - (blue_marbles + red_marbles) →
  orange_marbles = 6 :=
by 
  intros total_marbles blue_marbles red_marbles orange_marbles h_total h_blue h_red h_orange 
  rw [h_total, h_blue, h_red, h_orange]
  norm_num
  rw [nat.div_self] 
  reflexivity sorry

end number_of_orange_marbles_l198_198938


namespace remainder_when_divided_l198_198491

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l198_198491


namespace yura_finishes_textbook_on_sep_12_l198_198690

theorem yura_finishes_textbook_on_sep_12 :
  ∃ end_date : ℕ,
    let total_problems := 91,
        problems_left_on_sep_8 := 46,
        solved_until_sep_8 := total_problems - problems_left_on_sep_8,
        z := solved_until_sep_8 / 3, -- Since 3z = solved_until_sep_8
        sep6_solved := z + 1,
        sep7_solved := z,
        sep8_solved := z - 1,
        remaining_problems := total_problems - (sep6_solved + sep7_solved + sep8_solved),
        end_date := 8 + (sep8_solved * (sep8_solved + 1)) / 2,
        -- 8th + sum of an arithmetic series starting from 13 down to 1 problem
    end_date = 12 := sorry

end yura_finishes_textbook_on_sep_12_l198_198690


namespace cubic_root_equation_solution_l198_198320

theorem cubic_root_equation_solution (x : ℝ) : ∃ y : ℝ, y = real.cbrt x ∧ y = 15 / (8 - y) ↔ x = 27 ∨ x = 125 :=
by
  sorry

end cubic_root_equation_solution_l198_198320


namespace sum_first_10_common_l198_198649

-- Definition of sequences' general terms
def a_n (n : ℕ) := 5 + 3 * n
def b_k (k : ℕ) := 20 * 2^k

-- Sum of the first 10 elements in both sequences
noncomputable def sum_of_first_10_common_elements : ℕ :=
  let common_elements := List.map (λ k : ℕ, 20 * 4^k) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  List.sum common_elements

-- Proof statement
theorem sum_first_10_common : sum_of_first_10_common_elements = 6990500 :=
  by sorry

end sum_first_10_common_l198_198649


namespace intersection_A_B_l198_198186

-- Define sets A and B
def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | ∃ y ∈ A, |y| = x}

-- Prove that the intersection of A and B is {0, 2}
theorem intersection_A_B :
  A ∩ B = {0, 2} :=
by
  sorry

end intersection_A_B_l198_198186


namespace albert_wins_strategy_l198_198374

theorem albert_wins_strategy (n : ℕ) (h : n = 1999) : 
  ∃ strategy : (ℕ → ℕ), (∀ tokens : ℕ, tokens = n → tokens > 1 → 
  (∃ next_tokens : ℕ, next_tokens < tokens ∧ next_tokens ≥ 1 ∧ next_tokens ≥ tokens / 2) → 
  (∃ k, tokens = 2^k + 1) → strategy n = true) :=
sorry

end albert_wins_strategy_l198_198374


namespace isosceles_triangle_sides_l198_198597

theorem isosceles_triangle_sides (r R : ℝ) (a b c : ℝ) (h1 : r = 3 / 2) (h2 : R = 25 / 8)
  (h3 : a = c) (h4 : 5 = a) (h5 : 6 = b) : 
  ∃ a b c, a = 5 ∧ c = 5 ∧ b = 6 := by 
  sorry

end isosceles_triangle_sides_l198_198597


namespace minimum_votes_for_tall_l198_198544

theorem minimum_votes_for_tall (voters : ℕ) (districts : ℕ) (precincts : ℕ) (precinct_voters : ℕ)
  (vote_majority_per_precinct : ℕ → ℕ) (precinct_majority_per_district : ℕ → ℕ) (district_majority_to_win : ℕ) :
  voters = 135 ∧ districts = 5 ∧ precincts = 9 ∧ precinct_voters = 3 ∧
  (∀ p, vote_majority_per_precinct p = 2) ∧
  (∀ d, precinct_majority_per_district d = 5) ∧
  district_majority_to_win = 3 ∧ 
  tall_won : 
  ∃ min_votes, min_votes = 30 :=
by
  sorry

end minimum_votes_for_tall_l198_198544


namespace walkways_area_l198_198012

theorem walkways_area (rows cols : ℕ) (bed_length bed_width walkthrough_width garden_length garden_width total_flower_beds bed_area total_bed_area total_garden_area : ℝ) 
  (h1 : rows = 4) (h2 : cols = 3) 
  (h3 : bed_length = 8) (h4 : bed_width = 3) 
  (h5 : walkthrough_width = 2)
  (h6 : garden_length = (cols * bed_length) + ((cols + 1) * walkthrough_width))
  (h7 : garden_width = (rows * bed_width) + ((rows + 1) * walkthrough_width))
  (h8 : total_garden_area = garden_length * garden_width)
  (h9 : total_flower_beds = rows * cols)
  (h10 : bed_area = bed_length * bed_width)
  (h11 : total_bed_area = total_flower_beds * bed_area)
  (h12 : total_garden_area - total_bed_area = 416) : 
  True := 
sorry

end walkways_area_l198_198012


namespace question1_l198_198523

def sequence1 (a : ℕ → ℕ) : Prop :=
   a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = 3 * a (n - 1) + 1

noncomputable def a_n1 (n : ℕ) : ℕ := (3^n - 1) / 2

theorem question1 (a : ℕ → ℕ) (n : ℕ) : sequence1 a → a n = a_n1 n :=
by
  sorry

end question1_l198_198523


namespace rectangle_length_l198_198749

theorem rectangle_length (L W : ℝ) (h1 : L = 4 * W) (h2 : L * W = 100) : L = 20 :=
by
  sorry

end rectangle_length_l198_198749


namespace tan_sum_identity_l198_198499

open Real

theorem tan_sum_identity : 
  tan (80 * π / 180) + tan (40 * π / 180) - sqrt 3 * tan (80 * π / 180) * tan (40 * π / 180) = -sqrt 3 :=
by
  sorry

end tan_sum_identity_l198_198499


namespace least_value_m_n_l198_198898

theorem least_value_m_n :
  ∃ m n : ℕ, (m > 0 ∧ n > 0) ∧
            (Nat.gcd (m + n) 231 = 1) ∧
            (n^n ∣ m^m) ∧
            ¬ (m % n = 0) ∧
            m + n = 377 :=
by 
  sorry

end least_value_m_n_l198_198898


namespace heart_digit_proof_l198_198158

noncomputable def heart_digit : ℕ := 3

theorem heart_digit_proof (heartsuit : ℕ) (h : heartsuit * 9 + 6 = heartsuit * 10 + 3) : 
  heartsuit = heart_digit := 
by
  sorry

end heart_digit_proof_l198_198158


namespace pqrs_sum_l198_198594

/--
Given two pairs of real numbers (x, y) satisfying the equations:
1. x + y = 6
2. 2xy = 6

Prove that the solutions for x in the form x = (p ± q * sqrt(r)) / s give p + q + r + s = 11.
-/
theorem pqrs_sum : ∃ (p q r s : ℕ), (∀ (x y : ℝ), x + y = 6 ∧ 2*x*y = 6 → 
  (x = (p + q * Real.sqrt r) / s) ∨ (x = (p - q * Real.sqrt r) / s)) ∧ 
  p + q + r + s = 11 := 
sorry

end pqrs_sum_l198_198594


namespace total_money_correct_l198_198207

def shelly_has_total_money : Prop :=
  ∃ (ten_dollar_bills five_dollar_bills : ℕ), 
    ten_dollar_bills = 10 ∧
    five_dollar_bills = ten_dollar_bills - 4 ∧
    (10 * ten_dollar_bills + 5 * five_dollar_bills = 130)

theorem total_money_correct : shelly_has_total_money :=
by
  sorry

end total_money_correct_l198_198207


namespace container_volume_ratio_l198_198451

theorem container_volume_ratio
  (A B C : ℚ)  -- A is the volume of the first container, B is the volume of the second container, C is the volume of the third container
  (h1 : (8 / 9) * A = (7 / 9) * B)  -- Condition: First container was 8/9 full and second container gets filled to 7/9 after transfer.
  (h2 : (7 / 9) * B + (1 / 2) * C = C)  -- Condition: Mixing contents from second and third containers completely fill third container.
  : A / C = 63 / 112 := sorry  -- We need to prove this.

end container_volume_ratio_l198_198451


namespace abs_difference_lt_2t_l198_198530

/-- Given conditions of absolute values with respect to t -/
theorem abs_difference_lt_2t (x y s t : ℝ) (h₁ : |x - s| < t) (h₂ : |y - s| < t) :
  |x - y| < 2 * t :=
sorry

end abs_difference_lt_2t_l198_198530


namespace quadratic_real_roots_l198_198506

theorem quadratic_real_roots (m : ℝ) : 
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  (discriminant ≥ 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  sorry

end quadratic_real_roots_l198_198506


namespace inequality_solution_l198_198810

theorem inequality_solution (x : ℝ) : 
  (x < -4 ∨ x > 2) ↔ (x^2 + 3 * x - 4) / (x^2 - x - 2) > 0 :=
sorry

end inequality_solution_l198_198810


namespace co_complementary_angles_equal_l198_198225

def co_complementary (A : ℝ) : ℝ := 90 - A

theorem co_complementary_angles_equal (A B : ℝ) (h : co_complementary A = co_complementary B) : A = B :=
sorry

end co_complementary_angles_equal_l198_198225


namespace jade_initial_pieces_l198_198177

theorem jade_initial_pieces (n w l p : ℕ) (hn : n = 11) (hw : w = 7) (hl : l = 23) (hp : p = n * w + l) : p = 100 :=
by
  sorry

end jade_initial_pieces_l198_198177


namespace complement_union_result_l198_198001

open Set

variable (U A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3})
variable (hA : A = {1, 2})
variable (hB : B = {2, 3})

theorem complement_union_result : compl A ∪ B = {0, 2, 3} :=
by
  -- Our proof steps would go here
  sorry

end complement_union_result_l198_198001


namespace largest_initial_number_l198_198885

theorem largest_initial_number : ∃ n : ℕ, (n + 5 ∑ k : ℕ, k ≠ 0 ∧ ¬ (n % k = 0)) = 200 ∧ n = 189 :=
begin
  sorry
end

end largest_initial_number_l198_198885


namespace find_m_n_sum_l198_198660

theorem find_m_n_sum (x y m n : ℤ) 
  (h1 : x = 2)
  (h2 : y = 1)
  (h3 : m * x + y = -3)
  (h4 : x - 2 * y = 2 * n) : 
  m + n = -2 := 
by 
  sorry

end find_m_n_sum_l198_198660


namespace count_integers_l198_198004

def satisfies_conditions (n : ℤ) (r : ℤ) : Prop :=
  200 < n ∧ n < 300 ∧ n % 7 = r ∧ n % 9 = r ∧ 0 ≤ r ∧ r < 5

theorem count_integers (n : ℤ) (r : ℤ) :
  (satisfies_conditions n r) → ∃! n, 200 < n ∧ n < 300 ∧ ∃ r, n % 7 = r ∧ n % 9 = r ∧ 0 ≤ r ∧ r < 5 :=
by
  sorry

end count_integers_l198_198004


namespace jelly_bean_ratio_l198_198732

-- Define the number of jelly beans each person has
def napoleon_jelly_beans : ℕ := 17
def sedrich_jelly_beans : ℕ := napoleon_jelly_beans + 4
def mikey_jelly_beans : ℕ := 19

-- Define the sum of jelly beans of Napoleon and Sedrich
def sum_jelly_beans : ℕ := napoleon_jelly_beans + sedrich_jelly_beans

-- Define the ratio of the sum of Napoleon and Sedrich's jelly beans to Mikey's jelly beans
def ratio : ℚ := sum_jelly_beans / mikey_jelly_beans

-- Prove that the ratio is 2
theorem jelly_bean_ratio : ratio = 2 := by
  -- We skip the proof steps since the focus here is on the correct statement
  sorry

end jelly_bean_ratio_l198_198732


namespace minimize_f_at_a_l198_198655

def f (x a : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

theorem minimize_f_at_a (a : ℝ) (h : a = 82 / 43) :
  ∃ x, ∀ y, f x a ≤ f y a :=
sorry

end minimize_f_at_a_l198_198655


namespace total_production_cost_l198_198441

-- Conditions
def first_season_episodes : ℕ := 12
def remaining_season_factor : ℝ := 1.5
def last_season_episodes : ℕ := 24
def first_season_cost_per_episode : ℝ := 100000
def other_season_cost_per_episode : ℝ := first_season_cost_per_episode * 2

-- Number of seasons
def number_of_seasons : ℕ := 5

-- Question: Calculate the total cost
def total_first_season_cost : ℝ := first_season_episodes * first_season_cost_per_episode
def second_season_episodes : ℕ := (first_season_episodes * remaining_season_factor).toNat
def second_season_cost : ℝ := second_season_episodes * other_season_cost_per_episode
def third_and_fourth_seasons_cost : ℝ := 2 * second_season_cost
def last_season_cost : ℝ := last_season_episodes * other_season_cost_per_episode
def total_cost : ℝ := total_first_season_cost + second_season_cost + third_and_fourth_seasons_cost + last_season_cost

-- Proof
theorem total_production_cost :
  total_cost = 16800000 :=
by
  sorry

end total_production_cost_l198_198441


namespace product_divisible_by_12_l198_198914

theorem product_divisible_by_12 (a b c d : ℤ) : 
  12 ∣ ((b - a) * (c - a) * (d - a) * (b - c) * (d - c) * (d - b)) :=
  sorry

end product_divisible_by_12_l198_198914


namespace number_of_ways_to_partition_22_as_triangle_pieces_l198_198772

theorem number_of_ways_to_partition_22_as_triangle_pieces : 
  (∃ (a b c : ℕ), a + b + c = 22 ∧ a + b > c ∧ a + c > b ∧ b + c > a) → 
  ∃! (count : ℕ), count = 10 :=
by sorry

end number_of_ways_to_partition_22_as_triangle_pieces_l198_198772


namespace fraction_to_decimal_l198_198297

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198297


namespace net_effect_on_sale_l198_198775

variable (P Q : ℝ) -- Price and Quantity

theorem net_effect_on_sale :
  let reduced_price := 0.40 * P
  let increased_quantity := 2.50 * Q
  let price_after_tax := 0.44 * P
  let price_after_discount := 0.418 * P
  let final_revenue := price_after_discount * increased_quantity 
  let original_revenue := P * Q
  final_revenue / original_revenue = 1.045 :=
by
  sorry

end net_effect_on_sale_l198_198775


namespace line_passes_through_point_l198_198201

theorem line_passes_through_point (k : ℝ) :
  ∀ k : ℝ, (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 :=
by
  intro k
  sorry

end line_passes_through_point_l198_198201


namespace inequality_solution_set_l198_198584

theorem inequality_solution_set :
  (∀ x : ℝ, (3 * x - 2 < 2 * (x + 1) ∧ (x - 1) / 2 > 1) ↔ (3 < x ∧ x < 4)) :=
by
  sorry

end inequality_solution_set_l198_198584


namespace hilt_has_2_pennies_l198_198195

-- Define the total value of coins each person has without considering Mrs. Hilt's pennies
def dimes : ℕ := 2
def nickels : ℕ := 2
def hilt_base_amount : ℕ := dimes * 10 + nickels * 5 -- 30 cents

def jacob_pennies : ℕ := 4
def jacob_nickels : ℕ := 1
def jacob_dimes : ℕ := 1
def jacob_amount : ℕ := jacob_pennies * 1 + jacob_nickels * 5 + jacob_dimes * 10 -- 19 cents

def difference : ℕ := 13
def hilt_pennies : ℕ := 2 -- The solution's correct answer

theorem hilt_has_2_pennies : hilt_base_amount - jacob_amount + hilt_pennies = difference := by sorry

end hilt_has_2_pennies_l198_198195


namespace gcd_91_49_l198_198770

theorem gcd_91_49 : Int.gcd 91 49 = 7 := by
  sorry

end gcd_91_49_l198_198770


namespace bags_of_soil_needed_l198_198795

theorem bags_of_soil_needed
  (length width height : ℕ)
  (beds : ℕ)
  (volume_per_bag : ℕ)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_beds : beds = 2)
  (h_volume_per_bag : volume_per_bag = 4) :
  (length * width * height * beds) / volume_per_bag = 16 :=
by
  sorry

end bags_of_soil_needed_l198_198795


namespace B_equals_1_2_3_l198_198150

def A : Set ℝ := { x | x^2 ≤ 4 }
def B : Set ℕ := { x | x > 0 ∧ (x - 1:ℝ) ∈ A }

theorem B_equals_1_2_3 : B = {1, 2, 3} :=
by
  sorry

end B_equals_1_2_3_l198_198150


namespace find_m_range_l198_198515

theorem find_m_range
  (m y1 y2 y0 x0 : ℝ)
  (a c : ℝ) (h1 : a ≠ 0)
  (h2 : x0 = -2)
  (h3 : ∀ x, (x, ax^2 + 4*a*x + c) = (m, y1) ∨ (x, ax^2 + 4*a*x + c) = (m + 2, y2) ∨ (x, ax^2 + 4*a*x + c) = (x0, y0))
  (h4 : y0 ≥ y2) (h5 : y2 > y1) :
  m < -3 :=
sorry

end find_m_range_l198_198515


namespace cube_sum_l198_198674

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cube_sum_l198_198674


namespace isosceles_triangle_l198_198519

theorem isosceles_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h : a + b = (Real.tan (C / 2)) * (a * Real.tan A + b * Real.tan B)) :
  A = B := 
sorry

end isosceles_triangle_l198_198519


namespace total_amount_shared_l198_198443

-- Given John (J), Jose (Jo), and Binoy (B) and their proportion of money
variables (J Jo B : ℕ)
-- John received 1440 Rs.
variable (John_received : J = 1440)

-- The ratio of their shares is 2:4:6
axiom ratio_condition : J * 2 = Jo * 4 ∧ J * 2 = B * 6

-- The target statement to prove
theorem total_amount_shared : J + Jo + B = 8640 :=
by {
  sorry
}

end total_amount_shared_l198_198443


namespace fraction_sum_l198_198647

theorem fraction_sum :
  (7 : ℚ) / 12 + (3 : ℚ) / 8 = 23 / 24 :=
by
  -- Proof is omitted
  sorry

end fraction_sum_l198_198647


namespace arithmetic_sequence_properties_l198_198511

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = n * (a 1 + a n) / 2

def condition_S10_pos (S : ℕ → ℝ) : Prop :=
S 10 > 0

def condition_S11_neg (S : ℕ → ℝ) : Prop :=
S 11 < 0

-- Main statement
theorem arithmetic_sequence_properties {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}
  (ar_seq : is_arithmetic_sequence a d)
  (sum_first_n : sum_of_first_n_terms S a)
  (S10_pos : condition_S10_pos S)
  (S11_neg : condition_S11_neg S) :
  (∀ n, (S n) / n = a 1 + (n - 1) / 2 * d) ∧
  (a 2 = 1 → -2 / 7 < d ∧ d < -1 / 4) :=
by
  sorry

end arithmetic_sequence_properties_l198_198511


namespace yura_finishes_on_september_12_l198_198694

theorem yura_finishes_on_september_12
  (total_problems : ℕ) (initial_problem_solving_date : ℕ) (initial_remaining_problems : ℕ) (problems_on_sep_6 : ℕ)
  (problems_on_sep_7 : ℕ) (problems_on_sep_8 : ℕ):
  total_problems = 91 →
  initial_problem_solving_date = 6 →
  initial_remaining_problems = 46 →
  problems_on_sep_6 = 16 →
  problems_on_sep_7 = 15 →
  problems_on_sep_8 = 14 →
  (∑ i in finset.range 7, problems_on_sep_6 - i) = total_problems :=
begin
  sorry -- Proof to be provided
end

end yura_finishes_on_september_12_l198_198694


namespace find_DF_l198_198356

noncomputable def triangle (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def median (a b : ℝ) : ℝ := a / 2

theorem find_DF {DE EF DM DF : ℝ} (h1 : DE = 7) (h2 : EF = 10) (h3 : DM = 5) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l198_198356


namespace ken_gets_back_16_dollars_l198_198629

-- Given constants and conditions
def price_per_pound_steak : ℕ := 7
def pounds_of_steak : ℕ := 2
def price_carton_eggs : ℕ := 3
def price_gallon_milk : ℕ := 4
def price_pack_bagels : ℕ := 6
def bill_20_dollar : ℕ := 20
def bill_10_dollar : ℕ := 10
def bill_5_dollar_count : ℕ := 2
def coin_1_dollar_count : ℕ := 3

-- Calculate total cost of items
def total_cost_items : ℕ :=
  (pounds_of_steak * price_per_pound_steak) +
  price_carton_eggs +
  price_gallon_milk +
  price_pack_bagels

-- Calculate total amount paid
def total_amount_paid : ℕ :=
  bill_20_dollar +
  bill_10_dollar +
  (bill_5_dollar_count * 5) +
  (coin_1_dollar_count * 1)

-- Theorem statement to be proved
theorem ken_gets_back_16_dollars :
  total_amount_paid - total_cost_items = 16 := by
  sorry

end ken_gets_back_16_dollars_l198_198629


namespace exists_polynomial_for_divisors_l198_198327

open Polynomial

theorem exists_polynomial_for_divisors (n : ℕ) :
  (∃ P : ℤ[X], ∀ d : ℕ, d ∣ n → P.eval (d : ℤ) = (n / d : ℤ)^2) ↔
  (Nat.Prime n ∨ n = 1 ∨ n = 6) := by
  sorry

end exists_polynomial_for_divisors_l198_198327


namespace fraction_to_decimal_l198_198309

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198309


namespace bell_pepper_slices_l198_198917

theorem bell_pepper_slices :
  ∀ (num_peppers : ℕ) (slices_per_pepper : ℕ) (total_slices_pieces : ℕ) (half_slices : ℕ),
  num_peppers = 5 → slices_per_pepper = 20 → total_slices_pieces = 200 →
  half_slices = (num_peppers * slices_per_pepper) / 2 →
  (total_slices_pieces - (num_peppers * slices_per_pepper)) / half_slices = 2 :=
by
  intros num_peppers slices_per_pepper total_slices_pieces half_slices h1 h2 h3 h4
  -- skip the proof with sorry as instructed
  sorry

end bell_pepper_slices_l198_198917


namespace length_segment_MN_l198_198926

open Real

noncomputable def line (x : ℝ) : ℝ := x + 2

def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem length_segment_MN :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
    on_circle x₁ y₁ →
    on_circle x₂ y₂ →
    (line x₁ = y₁ ∧ line x₂ = y₂) →
    dist (x₁, y₁) (x₂, y₂) = 2 * sqrt 3 :=
by
  sorry

end length_segment_MN_l198_198926


namespace red_grapes_count_l198_198867

theorem red_grapes_count (G : ℕ) (total_fruit : ℕ) (red_grapes : ℕ) (raspberries : ℕ)
  (h1 : red_grapes = 3 * G + 7) 
  (h2 : raspberries = G - 5) 
  (h3 : total_fruit = G + red_grapes + raspberries) 
  (h4 : total_fruit = 102) : 
  red_grapes = 67 :=
by
  sorry

end red_grapes_count_l198_198867


namespace reciprocal_neg_one_div_2022_l198_198230

theorem reciprocal_neg_one_div_2022 : (1 / (-1 / 2022)) = -2022 :=
by sorry

end reciprocal_neg_one_div_2022_l198_198230


namespace original_number_is_5_div_4_l198_198198

-- Define the condition in Lean.
def condition (y : ℚ) : Prop :=
  1 - 1 / y = 1 / 5

-- Define the theorem to prove that y = 5 / 4 given the condition.
theorem original_number_is_5_div_4 (y : ℚ) (h : condition y) : y = 5 / 4 :=
by
  sorry

end original_number_is_5_div_4_l198_198198


namespace radius_of_circle_with_chords_l198_198991

theorem radius_of_circle_with_chords 
  (chord1_length : ℝ) (chord2_length : ℝ) (distance_between_midpoints : ℝ) 
  (h1 : chord1_length = 9) (h2 : chord2_length = 17) (h3 : distance_between_midpoints = 5) : 
  ∃ r : ℝ, r = 85 / 8 :=
by
  sorry

end radius_of_circle_with_chords_l198_198991


namespace least_pos_int_divisible_by_four_distinct_primes_l198_198065

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (p1 p2 p3 p4 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ distinct [p1, p2, p3, p4] ∧ n = p1 * p2 * p3 * p4) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l198_198065


namespace yura_finishes_problems_l198_198707

theorem yura_finishes_problems 
  (total_problems : ℕ) (start_date : ℕ) (remaining_problems : ℕ) (z : ℕ)
  (H_total : total_problems = 91)
  (H_start_date : start_date = 6)
  (H_remaining : remaining_problems = 46)
  (H_solved_by_8th : (z + 1) + z + (z - 1) = total_problems - remaining_problems) :
  ∃ finish_date, finish_date = 12 :=
by
  use 12
  sorry

end yura_finishes_problems_l198_198707


namespace construct_rectangle_l198_198109

structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  diagonal : ℝ
  sum_diag_side : ℝ := side2 + diagonal

theorem construct_rectangle (b a d : ℝ) (r : Rectangle) :
  r.side2 = a ∧ r.side1 = b ∧ r.sum_diag_side = a + d :=
by
  sorry

end construct_rectangle_l198_198109


namespace chairs_subset_count_l198_198427

theorem chairs_subset_count (ch : Fin 12 → bool) :
  (∃ i : Fin 12, ch i ∧ ch (i + 1) % 12 ∧ ch (i + 2) % 12) →
  2056 = ∑ s : Finset (Fin 12), if (∃ i : Fin 12, ∀ j : Fin n, (i + j) % 12 ∈ s) then 1 else 0 :=
sorry

end chairs_subset_count_l198_198427


namespace remy_gallons_used_l198_198203

def roman_usage (R : ℕ) : Prop := R + (3 * R + 1) = 33

def remy_usage (R : ℕ) (Remy : ℕ) : Prop := Remy = 3 * R + 1

theorem remy_gallons_used :
  ∃ R Remy : ℕ, roman_usage R ∧ remy_usage R Remy ∧ Remy = 25 :=
  by
    sorry

end remy_gallons_used_l198_198203


namespace add_candies_to_equalize_l198_198591

-- Define the initial number of candies in basket A and basket B
def candiesInA : ℕ := 8
def candiesInB : ℕ := 17

-- Problem statement: Prove that adding 9 more candies to basket A
-- makes the number of candies in basket A equal to that in basket B.
theorem add_candies_to_equalize : ∃ n : ℕ, candiesInA + n = candiesInB :=
by
  use 9  -- The value we are adding to the candies in basket A
  sorry  -- Proof goes here

end add_candies_to_equalize_l198_198591


namespace discount_is_25_l198_198812

def original_price : ℕ := 76
def discounted_price : ℕ := 51
def discount_amount : ℕ := original_price - discounted_price

theorem discount_is_25 : discount_amount = 25 := by
  sorry

end discount_is_25_l198_198812


namespace prime_between_30_40_with_remainder_l198_198403

theorem prime_between_30_40_with_remainder :
  ∃ n : ℕ, Prime n ∧ 30 < n ∧ n < 40 ∧ n % 9 = 4 ∧ n = 31 :=
by
  sorry

end prime_between_30_40_with_remainder_l198_198403


namespace smallest_number_remainder_problem_l198_198118

theorem smallest_number_remainder_problem :
  ∃ N : ℕ, (N % 13 = 2) ∧ (N % 15 = 4) ∧ (∀ n : ℕ, (n % 13 = 2 ∧ n % 15 = 4) → n ≥ N) :=
sorry

end smallest_number_remainder_problem_l198_198118


namespace a_when_a_minus_1_no_reciprocal_l198_198669

theorem a_when_a_minus_1_no_reciprocal (a : ℝ) (h : ¬ ∃ b : ℝ, (a - 1) * b = 1) : a = 1 := 
by
  sorry

end a_when_a_minus_1_no_reciprocal_l198_198669


namespace length_of_route_l198_198769

theorem length_of_route 
  (D vA vB : ℝ)
  (h_vA : vA = D / 10)
  (h_vB : vB = D / 6)
  (t : ℝ)
  (h_va_t : vA * t = 75)
  (h_vb_t : vB * t = D - 75) :
  D = 200 :=
by
  sorry

end length_of_route_l198_198769


namespace min_value_expression_geq_17_div_2_min_value_expression_eq_17_div_2_for_specific_a_b_l198_198331

noncomputable def min_value_expression (a b : ℝ) (hab : 2 * a + b = 1) : ℝ :=
  4 * a^2 + b^2 + 1 / (a * b)

theorem min_value_expression_geq_17_div_2 {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (hab: 2 * a + b = 1) :
  min_value_expression a b hab ≥ 17 / 2 :=
sorry

theorem min_value_expression_eq_17_div_2_for_specific_a_b :
  min_value_expression (1/3) (1/3) (by norm_num) = 17 / 2 :=
sorry

end min_value_expression_geq_17_div_2_min_value_expression_eq_17_div_2_for_specific_a_b_l198_198331


namespace parabola_range_m_l198_198851

noncomputable def parabola (m : ℝ) (x : ℝ) : ℝ := x^2 - (4*m + 1)*x + (2*m - 1)

theorem parabola_range_m (m : ℝ) :
  (∀ x : ℝ, parabola m x = 0 → (1 < x ∧ x < 2) ∨ (x < 1 ∨ x > 2)) ∧
  parabola m 0 < -1/2 →
  1/6 < m ∧ m < 1/4 :=
by
  sorry

end parabola_range_m_l198_198851


namespace necessary_but_not_sufficient_not_sufficient_condition_l198_198029

theorem necessary_but_not_sufficient (a b : ℝ) : (a > 2 ∧ b > 2) → (a + b > 4) :=
sorry

theorem not_sufficient_condition (a b : ℝ) : (a + b > 4) → ¬(a > 2 ∧ b > 2) :=
sorry

end necessary_but_not_sufficient_not_sufficient_condition_l198_198029


namespace neg_exists_equiv_forall_l198_198329

theorem neg_exists_equiv_forall :
  (¬ ∃ x : ℝ, x^2 - x + 4 < 0) ↔ (∀ x : ℝ, x^2 - x + 4 ≥ 0) :=
by
  sorry

end neg_exists_equiv_forall_l198_198329


namespace remainder_m_squared_plus_4m_plus_6_l198_198529

theorem remainder_m_squared_plus_4m_plus_6 (m : ℤ) (k : ℤ) (hk : m = 100 * k - 2) :
  (m ^ 2 + 4 * m + 6) % 100 = 2 := 
sorry

end remainder_m_squared_plus_4m_plus_6_l198_198529


namespace total_spent_is_correct_l198_198680

def cost_of_lunch : ℝ := 60.50
def tip_percentage : ℝ := 0.20
def tip_amount : ℝ := cost_of_lunch * tip_percentage
def total_amount_spent : ℝ := cost_of_lunch + tip_amount

theorem total_spent_is_correct : total_amount_spent = 72.60 := by
  -- placeholder for the proof
  sorry

end total_spent_is_correct_l198_198680


namespace least_positive_integer_divisible_by_four_distinct_primes_l198_198064

theorem least_positive_integer_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, n > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ n ∧ 
             (∀ m : ℕ, (m > 0 ∧ ∀ p ∈ {2, 3, 5, 7}, p ∣ m) → n ≤ m) := 
begin
  use 210,
  split,
  { exact 210 > 0, },
  split,
  { intros p hp,
    fin_cases hp; simp, },
  intro m,
  intros H1 H2,
  apply nat.le_of_dvd H1,
  have H3 := nat.eq_or_lt_of_le (nat.le_mul_of_pos_left (nat.pos_of_mem_primes 2)),
  all_goals { sorry, } -- The detailed proof steps are not specified
end

end least_positive_integer_divisible_by_four_distinct_primes_l198_198064


namespace quadrilateral_perimeter_l198_198767

theorem quadrilateral_perimeter (a b : ℝ) (h₁ : a = 10) (h₂ : b = 15)
  (h₃ : ∀ (ABD BCD ABC ACD : ℝ), ABD = BCD ∧ ABC = ACD) : a + a + b + b = 50 :=
by
  rw [h₁, h₂]
  linarith


end quadrilateral_perimeter_l198_198767


namespace constant_term_expansion_l198_198217

noncomputable def binomial_expansion (x : ℝ) : ℝ :=
  (x - (1 / x)) * (2 * x + (1 / x))^5

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) :
  ∃ c : ℝ, binomial_expansion x = c ∧ c = -40 :=
by
  sorry

end constant_term_expansion_l198_198217


namespace remainder_123456789012_div_252_l198_198485

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l198_198485


namespace biased_die_expected_value_is_neg_1_5_l198_198251

noncomputable def biased_die_expected_value : ℚ :=
  let prob_123 := (1 / 6 : ℚ) + (1 / 6) + (1 / 6)
  let prob_456 := (1 / 2 : ℚ)
  let gain := prob_123 * 2
  let loss := prob_456 * -5
  gain + loss

theorem biased_die_expected_value_is_neg_1_5 :
  biased_die_expected_value = - (3 / 2 : ℚ) :=
by
  -- We skip the detailed proof steps here.
  sorry

end biased_die_expected_value_is_neg_1_5_l198_198251


namespace esperanzas_gross_monthly_salary_l198_198563

variables (Rent FoodExpenses MortgageBill Savings Taxes GrossSalary : ℝ)

def problem_conditions (Rent FoodExpenses MortgageBill Savings Taxes : ℝ) :=
  Rent = 600 ∧
  FoodExpenses = (3 / 5) * Rent ∧
  MortgageBill = 3 * FoodExpenses ∧
  Savings = 2000 ∧
  Taxes = (2 / 5) * Savings

theorem esperanzas_gross_monthly_salary (h : problem_conditions Rent FoodExpenses MortgageBill Savings Taxes) :
  GrossSalary = Rent + FoodExpenses + MortgageBill + Taxes + Savings → GrossSalary = 4840 :=
by
  sorry

end esperanzas_gross_monthly_salary_l198_198563


namespace quadratic_real_roots_l198_198504

theorem quadratic_real_roots (m : ℝ) : 
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  (discriminant ≥ 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  sorry

end quadratic_real_roots_l198_198504


namespace point_inside_circle_l198_198011

theorem point_inside_circle (m : ℝ) : (1 - 2)^2 + (-3 + 1)^2 < m → m > 5 :=
by
  sorry

end point_inside_circle_l198_198011


namespace fg_of_2_l198_198857

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := (x + 1)^2

theorem fg_of_2 : f (g 2) = 29 := by
  sorry

end fg_of_2_l198_198857


namespace last_number_is_two_l198_198042

theorem last_number_is_two (A B C D : ℝ)
  (h1 : A + B + C = 18)
  (h2 : B + C + D = 9)
  (h3 : A + D = 13) :
  D = 2 :=
sorry

end last_number_is_two_l198_198042


namespace tom_steps_l198_198192

theorem tom_steps (matt_rate : ℕ) (tom_extra_rate : ℕ) (matt_steps : ℕ) (tom_rate : ℕ := matt_rate + tom_extra_rate) (time : ℕ := matt_steps / matt_rate)
(H_matt_rate : matt_rate = 20)
(H_tom_extra_rate : tom_extra_rate = 5)
(H_matt_steps : matt_steps = 220) :
  tom_rate * time = 275 :=
by
  -- We start the proof here, but leave it as sorry.
  sorry

end tom_steps_l198_198192


namespace p_implies_q_and_not_q_implies_p_l198_198134

variable {x : ℝ}

def p : Prop := 0 < x ∧ x < 2
def q : Prop := -1 < x ∧ x < 3

theorem p_implies_q_and_not_q_implies_p : (p → q) ∧ ¬ (q → p) := 
by
  sorry

end p_implies_q_and_not_q_implies_p_l198_198134


namespace find_fourth_number_l198_198783

theorem find_fourth_number (x : ℝ) (h : 3 + 33 + 333 + x = 369.63) : x = 0.63 :=
sorry

end find_fourth_number_l198_198783


namespace ryan_bus_meet_exactly_once_l198_198035

-- Define respective speeds of Ryan and the bus
def ryan_speed : ℕ := 6 
def bus_speed : ℕ := 15 

-- Define bench placement and stop times
def bench_distance : ℕ := 300 
def regular_stop_time : ℕ := 45 
def extra_stop_time : ℕ := 90 

-- Initial positions
def ryan_initial_position : ℕ := 0
def bus_initial_position : ℕ := 300

-- Distance function D(t)
noncomputable def distance_at_time (t : ℕ) : ℤ :=
  let bus_travel_time : ℕ := 15  -- time for bus to travel 225 feet
  let bus_stop_time : ℕ := 45  -- time for bus to stop during regular stops
  let extended_stop_time : ℕ := 90  -- time for bus to stop during 3rd bench stops
  sorry -- calculation of distance function

-- Problem to prove: Ryan and the bus meet exactly once
theorem ryan_bus_meet_exactly_once : ∃ t₁ t₂ : ℕ, t₁ ≠ t₂ ∧ distance_at_time t₁ = 0 ∧ distance_at_time t₂ ≠ 0 := 
  sorry

end ryan_bus_meet_exactly_once_l198_198035


namespace smartphone_price_l198_198561

/-
Question: What is the sticker price of the smartphone, given the following conditions?
Conditions:
1: Store A offers a 20% discount on the sticker price, followed by a $120 rebate. Prices include an 8% sales tax applied after all discounts and fees.
2: Store B offers a 30% discount on the sticker price but adds a $50 handling fee. Prices include an 8% sales tax applied after all discounts and fees.
3: Natalie saves $27 by purchasing the smartphone at store A instead of store B.

Proof Problem:
Prove that given the above conditions, the sticker price of the smartphone is $1450.
-/

theorem smartphone_price (p : ℝ) :
  (1.08 * (0.7 * p + 50) - 1.08 * (0.8 * p - 120)) = 27 ->
  p = 1450 :=
by
  sorry

end smartphone_price_l198_198561


namespace sum_reciprocals_of_roots_l198_198899

theorem sum_reciprocals_of_roots :
  let S := ∑ n in (Finset.fin_range 2020), (1 / (1 - a n))
  (∀ (n : ℕ), a n ∈ ({x | polynomial.eval x (polynomial.C 1 * (polynomial.X 2020 + polynomial.X 2019 + ... + polynomial.X) - polynomial.C 2022) = 0 })) →
  S = -2041210 :=
by
  sorry

end sum_reciprocals_of_roots_l198_198899


namespace arithmetic_computation_l198_198975

theorem arithmetic_computation : 65 * 1515 - 25 * 1515 = 60600 := by
  sorry

end arithmetic_computation_l198_198975


namespace geometric_progression_exists_l198_198114

theorem geometric_progression_exists :
  ∃ (b1 b2 b3 b4: ℤ) (q: ℤ), 
    b2 = b1 * q ∧ 
    b3 = b1 * q^2 ∧ 
    b4 = b1 * q^3 ∧  
    b3 - b1 = 9 ∧ 
    b2 - b4 = 18 ∧ 
    b1 = 3 ∧ b2 = -6 ∧ b3 = 12 ∧ b4 = -24 :=
sorry

end geometric_progression_exists_l198_198114


namespace distance_to_directrix_l198_198144

theorem distance_to_directrix (p : ℝ) (h1 : ∃ (x y : ℝ), y^2 = 2 * p * x ∧ (x = 2 ∧ y = 2 * Real.sqrt 2)) :
  abs (2 - (-1)) = 3 :=
by
  sorry

end distance_to_directrix_l198_198144


namespace first_digit_power_l198_198008

theorem first_digit_power (n : ℕ) (h : ∃ k : ℕ, 7 * 10^k ≤ 2^n ∧ 2^n < 8 * 10^k) :
  (∃ k' : ℕ, 1 * 10^k' ≤ 5^n ∧ 5^n < 2 * 10^k') :=
sorry

end first_digit_power_l198_198008


namespace eq_no_sol_l198_198376

open Nat -- Use natural number namespace

theorem eq_no_sol (k : ℤ) (x y z : ℕ) (hk1 : k ≠ 1) (hk3 : k ≠ 3) :
  ¬ (x^2 + y^2 + z^2 = k * x * y * z) := 
sorry

end eq_no_sol_l198_198376


namespace find_a_l198_198664

theorem find_a (a : ℝ) (A : Set ℝ) (hA : A = {a - 2, a^2 + 4*a, 10}) (h : -3 ∈ A) : a = -3 := 
by
  -- placeholder proof
  sorry

end find_a_l198_198664


namespace sales_worth_l198_198789

theorem sales_worth (S: ℝ) : 
  (1300 + 0.025 * (S - 4000) = 0.05 * S + 600) → S = 24000 :=
by
  sorry

end sales_worth_l198_198789


namespace roots_of_quadratic_l198_198369

theorem roots_of_quadratic (p q : ℝ) (h1 : 3 * p^2 + 9 * p - 21 = 0) (h2 : 3 * q^2 + 9 * q - 21 = 0) :
  (3 * p - 4) * (6 * q - 8) = 122 := by
  -- We don't need to provide the proof here, only the statement
  sorry

end roots_of_quadratic_l198_198369


namespace probability_passing_exam_l198_198170

-- Define probabilities for sets A, B, and C, and passing conditions
def P_A := 0.3
def P_B := 0.3
def P_C := 1 - P_A - P_B
def P_D_given_A := 0.8
def P_D_given_B := 0.6
def P_D_given_C := 0.8

-- Total probability of passing
def P_D := P_A * P_D_given_A + P_B * P_D_given_B + P_C * P_D_given_C

-- Proof that the total probability of passing is 0.74
theorem probability_passing_exam : P_D = 0.74 :=
by
  -- (skip the proof steps)
  sorry

end probability_passing_exam_l198_198170


namespace solve_for_first_expedition_weeks_l198_198023

-- Define the variables according to the given conditions.
variables (x : ℕ)
variables (days_in_week : ℕ := 7)
variables (total_days_on_island : ℕ := 126)

-- Define the total number of weeks spent on the expeditions.
def total_weeks_on_expeditions (x : ℕ) : ℕ := 
  x + (x + 2) + 2 * (x + 2)

-- Convert total days to weeks.
def total_weeks := total_days_on_island / days_in_week

-- Prove the equation
theorem solve_for_first_expedition_weeks
  (h : total_weeks_on_expeditions x = total_weeks):
  x = 3 :=
by
  sorry

end solve_for_first_expedition_weeks_l198_198023


namespace sean_and_julie_sums_l198_198387

theorem sean_and_julie_sums :
  let seanSum := 2 * (Finset.range 301).sum
  let julieSum := (Finset.range 301).sum
  seanSum / julieSum = 2 := by
  sorry

end sean_and_julie_sums_l198_198387


namespace cos_half_alpha_l198_198995

open Real -- open the Real namespace for convenience

theorem cos_half_alpha {α : ℝ} (h1 : cos α = 1 / 5) (h2 : 0 < α ∧ α < π) :
  cos (α / 2) = sqrt (15) / 5 :=
by
  sorry -- Proof is omitted

end cos_half_alpha_l198_198995


namespace infinitely_many_n_prime_l198_198724

theorem infinitely_many_n_prime (p : ℕ) [Fact (Nat.Prime p)] : ∃ᶠ n in at_top, p ∣ 2^n - n := 
sorry

end infinitely_many_n_prime_l198_198724


namespace solution_to_equation_l198_198819

theorem solution_to_equation :
  ∃ x : ℝ, x = (11 - 3 * Real.sqrt 5) / 2 ∧ x^2 + 6 * x + 6 * x * Real.sqrt (x + 4) = 31 :=
by
  sorry

end solution_to_equation_l198_198819


namespace range_of_m_l198_198512

theorem range_of_m {a b c x0 y0 y1 y2 m : ℝ} (h1 : a ≠ 0)
    (A_on_parabola : y1 = a * m^2 + 4 * a * m + c)
    (B_on_parabola : y2 = a * (m + 2)^2 + 4 * a * (m + 2) + c)
    (C_on_parabola : y0 = a * (-2)^2 + 4 * a * (-2) + c)
    (C_is_vertex : x0 = -2)
    (y_relation : y0 ≥ y2 ∧ y2 > y1) :
    m < -3 := 
sorry

end range_of_m_l198_198512


namespace target_water_percentage_is_two_percent_l198_198194

variable (initial_milk_volume pure_milk_volume : ℕ)
variable (initial_water_percentage target_water_percentage : ℚ)

-- Conditions: Initial milk contains 5% water and we add 15 liters of pure milk
axiom initial_milk_condition : initial_milk_volume = 10
axiom pure_milk_condition : pure_milk_volume = 15
axiom initial_water_condition : initial_water_percentage = 5 / 100

-- Prove that target percentage of water in the milk is 2%
theorem target_water_percentage_is_two_percent :
  target_water_percentage = 2 / 100 := by
  sorry

end target_water_percentage_is_two_percent_l198_198194


namespace sean_div_julie_l198_198383

-- Define the sum of the first n integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum: sum of even integers from 2 to 600
def sean_sum : ℕ := 2 * sum_first_n 300

-- Define Julie's sum: sum of integers from 1 to 300
def julie_sum : ℕ := sum_first_n 300

-- Prove that Sean's sum divided by Julie's sum equals 2
theorem sean_div_julie : sean_sum / julie_sum = 2 := by
  sorry

end sean_div_julie_l198_198383


namespace fraction_to_decimal_l198_198293

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198293


namespace relationship_C1_C2_A_l198_198027

variables (A B C C1 C2 : ℝ)

-- Given conditions
def TriangleABC : Prop := B = 2 * A
def AngleSumProperty : Prop := A + B + C = 180
def AltitudeDivides := C1 = 90 - A ∧ C2 = 90 - 2 * A

-- Theorem to prove the relationship between C1, C2, and A
theorem relationship_C1_C2_A (h1: TriangleABC A B) (h2: AngleSumProperty A B C) (h3: AltitudeDivides C1 C2 A) : 
  C1 - C2 = A :=
by sorry

end relationship_C1_C2_A_l198_198027


namespace find_number_l198_198746

theorem find_number (N M : ℕ) 
  (h1 : N + M = 3333) (h2 : N - M = 693) :
  N = 2013 :=
sorry

end find_number_l198_198746


namespace fraction_to_decimal_l198_198290

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l198_198290


namespace abc_inequality_l198_198829

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) : 
    (ab / (a^5 + ab + b^5)) + (bc / (b^5 + bc + c^5)) + (ca / (c^5 + ca + a^5)) ≤ 1 := 
sorry

end abc_inequality_l198_198829


namespace emily_seeds_start_with_l198_198112

-- Define the conditions as hypotheses
variables (big_garden_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ)

-- Conditions: Emily planted 29 seeds in the big garden and 4 seeds in each of her 3 small gardens.
def emily_conditions := big_garden_seeds = 29 ∧ small_gardens = 3 ∧ seeds_per_small_garden = 4

-- Define the statement to prove the total number of seeds Emily started with
theorem emily_seeds_start_with (h : emily_conditions big_garden_seeds small_gardens seeds_per_small_garden) : 
(big_garden_seeds + small_gardens * seeds_per_small_garden) = 41 :=
by
  -- Assuming the proof follows logically from conditions
  sorry

end emily_seeds_start_with_l198_198112


namespace max_distance_from_B_to_P_l198_198662

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -4, y := 1 }
def P : Point := { x := 3, y := -1 }

def line_l (m : ℝ) (pt : Point) : Prop :=
  (2 * m + 1) * pt.x - (m - 1) * pt.y - m - 5 = 0

theorem max_distance_from_B_to_P :
  ∃ B : Point, A = { x := -4, y := 1 } → 
               (∀ m : ℝ, line_l m B) →
               ∃ d, d = 5 + Real.sqrt 10 :=
sorry

end max_distance_from_B_to_P_l198_198662


namespace sum_of_digits_second_smallest_mult_of_lcm_l198_198366

theorem sum_of_digits_second_smallest_mult_of_lcm :
  let lcm12345678 := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8
  let M := 2 * lcm12345678
  (Nat.digits 10 M).sum = 15 := by
    -- Definitions from the problem statement
    let lcm12345678 := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8
    let M := 2 * lcm12345678
    sorry

end sum_of_digits_second_smallest_mult_of_lcm_l198_198366


namespace box_volume_l198_198282

structure Box where
  L : ℝ  -- Length
  W : ℝ  -- Width
  H : ℝ  -- Height

def front_face_area (box : Box) : ℝ := box.L * box.H
def top_face_area (box : Box) : ℝ := box.L * box.W
def side_face_area (box : Box) : ℝ := box.H * box.W

noncomputable def volume (box : Box) : ℝ := box.L * box.W * box.H

theorem box_volume (box : Box)
  (h1 : front_face_area box = 0.5 * top_face_area box)
  (h2 : top_face_area box = 1.5 * side_face_area box)
  (h3 : side_face_area box = 72) :
  volume box = 648 := by
  sorry

end box_volume_l198_198282


namespace minimal_pyramid_height_l198_198015

theorem minimal_pyramid_height (r x a : ℝ) (h₁ : 0 < r) (h₂ : a = 2 * r * x / (x - r)) (h₃ : x > 4 * r) :
  x = (6 + 2 * Real.sqrt 3) * r :=
by
  -- Proof steps would go here
  sorry

end minimal_pyramid_height_l198_198015


namespace simplify_expression_l198_198571

theorem simplify_expression : (90 / 150) * (35 / 21) = 1 :=
by
  -- Insert proof here 
  sorry

end simplify_expression_l198_198571


namespace average_sleep_is_8_l198_198155

-- Define the hours of sleep for each day
def mondaySleep : ℕ := 8
def tuesdaySleep : ℕ := 7
def wednesdaySleep : ℕ := 8
def thursdaySleep : ℕ := 10
def fridaySleep : ℕ := 7

-- Calculate the total hours of sleep over the week
def totalSleep : ℕ := mondaySleep + tuesdaySleep + wednesdaySleep + thursdaySleep + fridaySleep
-- Define the total number of days
def totalDays : ℕ := 5

-- Calculate the average sleep per night
def averageSleepPerNight : ℕ := totalSleep / totalDays

-- Prove the statement
theorem average_sleep_is_8 : averageSleepPerNight = 8 := 
by
  -- All conditions are automatically taken into account as definitions
  -- Add a placeholder to skip the actual proof
  sorry

end average_sleep_is_8_l198_198155


namespace regular_admission_ticket_price_l198_198651

theorem regular_admission_ticket_price
  (n : ℕ) (t : ℕ) (p : ℕ)
  (n_r n_s r : ℕ)
  (H1 : n_r = 3 * n_s)
  (H2 : n_s + n_r = n)
  (H3 : n_r * r + n_s * p = t)
  (H4 : n = 3240)
  (H5 : t = 22680)
  (H6 : p = 4) : 
  r = 8 :=
by sorry

end regular_admission_ticket_price_l198_198651


namespace calculation_eq_minus_one_l198_198269

noncomputable def calculation : ℝ :=
  (-1)^(53 : ℤ) + 3^((2^3 + 5^2 - 7^2) : ℤ)

theorem calculation_eq_minus_one : calculation = -1 := 
by 
  sorry

end calculation_eq_minus_one_l198_198269


namespace unique_intersection_l198_198520

def line1 (x y : ℝ) : Prop := 3 * x - 2 * y - 9 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 4 * y - 12 = 0
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = -1

theorem unique_intersection : ∃! p : ℝ × ℝ, 
                             (line1 p.1 p.2) ∧ 
                             (line2 p.1 p.2) ∧ 
                             (line3 p.1) ∧ 
                             (line4 p.2) ∧ 
                             p = (3, -1) :=
by
  sorry

end unique_intersection_l198_198520


namespace largest_initial_number_l198_198880

theorem largest_initial_number (n : ℕ) (h : (∃ a b c d e : ℕ, n ≠ 0 ∧ n + a + b + c + d + e = 200 
                                              ∧ n % a ≠ 0 ∧ n % b ≠ 0 ∧ n % c ≠ 0 ∧ n % d ≠ 0 ∧ n % e ≠ 0)) 
: n ≤ 189 :=
sorry

end largest_initial_number_l198_198880


namespace number_of_adjacent_subsets_l198_198415

/-- The number of subsets of a set of 12 chairs arranged in a circle
that contain at least three adjacent chairs is 1634. -/
theorem number_of_adjacent_subsets : 
  let subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
    if n = 3 then 12 else
    if n = 4 then 12 else
    if n = 5 then 12 else
    if n = 6 then 12 else
    if n = 7 then 792 else
    if n = 8 then 495 else
    if n = 9 then 220 else
    if n = 10 then 66 else
    if n = 11 then 12 else
    if n = 12 then 1 else 0
  in
    (subsets_with_adjacent_chairs 3) +
    (subsets_with_adjacent_chairs 4) +
    (subsets_with_adjacent_chairs 5) +
    (subsets_with_adjacent_chairs 6) +
    (subsets_with_adjacent_chairs 7) +
    (subsets_with_adjacent_chairs 8) +
    (subsets_with_adjacent_chairs 9) +
    (subsets_with_adjacent_chairs 10) +
    (subsets_with_adjacent_chairs 11) +
    (subsets_with_adjacent_chairs 12) = 1634 :=
by
  sorry

end number_of_adjacent_subsets_l198_198415


namespace quiz_points_minus_homework_points_l198_198910

theorem quiz_points_minus_homework_points
  (total_points : ℕ)
  (quiz_points : ℕ)
  (test_points : ℕ)
  (homework_points : ℕ)
  (h1 : total_points = 265)
  (h2 : test_points = 4 * quiz_points)
  (h3 : homework_points = 40)
  (h4 : homework_points + quiz_points + test_points = total_points) :
  quiz_points - homework_points = 5 :=
by sorry

end quiz_points_minus_homework_points_l198_198910


namespace area_ratio_of_squares_l198_198229

theorem area_ratio_of_squares (s t : ℝ) (h : 4 * s = 4 * (4 * t)) : (s ^ 2) / (t ^ 2) = 16 :=
by
  sorry

end area_ratio_of_squares_l198_198229


namespace limit_C_of_f_is_2_l198_198143

variable {f : ℝ → ℝ}
variable {x₀ : ℝ}
variable {f' : ℝ}

noncomputable def differentiable_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ f' : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs (f (x + h) - f x - f' * h) / abs (h) < ε

axiom hf_differentiable : differentiable_at f x₀
axiom f'_at_x₀ : f' = 1

theorem limit_C_of_f_is_2 
  (hf_differentiable : differentiable_at f x₀) 
  (h_f'_at_x₀ : f' = 1) : 
  (∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (x₀ + 2 * Δx) - f x₀) / Δx - 2) < ε) :=
sorry

end limit_C_of_f_is_2_l198_198143


namespace crates_needed_l198_198460

def ceil_div (a b : ℕ) : ℕ := (a + b - 1) / b

theorem crates_needed :
  ceil_div 145 12 + ceil_div 271 8 + ceil_div 419 10 + ceil_div 209 14 = 104 :=
by
  sorry

end crates_needed_l198_198460


namespace remainder_sum_div_40_l198_198558

variable (k m n : ℤ)
variables (a b c : ℤ)
variable (h1 : a % 80 = 75)
variable (h2 : b % 120 = 115)
variable (h3 : c % 160 = 155)

theorem remainder_sum_div_40 : (a + b + c) % 40 = 25 :=
by
  -- Use sorry as we are not required to fill in the proof
  sorry

end remainder_sum_div_40_l198_198558


namespace tommys_family_members_l198_198414

-- Definitions
def ounces_per_member : ℕ := 16
def ounces_per_steak : ℕ := 20
def steaks_needed : ℕ := 4

-- Theorem statement
theorem tommys_family_members : (steaks_needed * ounces_per_steak) / ounces_per_member = 5 :=
by
  -- Proof goes here
  sorry

end tommys_family_members_l198_198414


namespace lambda_phi_relation_l198_198336

-- Define the context and conditions
variables (A B C D M N : Type) -- Points on the triangle with D being the midpoint of BC
variables (AB AC BC BN BM MN : ℝ) -- Lengths
variables (lambda phi : ℝ) -- Ratios given in the problem

-- Conditions
-- 1. M is a point on the median AD of triangle ABC
variable (h1 : M = D ∨ M = A ∨ M = D) -- Simplified condition stating M's location
-- 2. The line BM intersects the side AC at point N
variable (h2 : N = M ∧ N ≠ A ∧ N ≠ C) -- Defining the intersection point
-- 3. AB is tangent to the circumcircle of triangle NBC
variable (h3 : tangent AB (circumcircle N B C))
-- 4. BC = lambda BN
variable (h4 : BC = lambda * BN)
-- 5. BM = phi * MN
variable (h5 : BM = phi * MN)

-- Goal
theorem lambda_phi_relation : phi = lambda ^ 2 :=
sorry

end lambda_phi_relation_l198_198336


namespace total_tv_show_cost_correct_l198_198440

noncomputable def total_cost_of_tv_show : ℕ :=
  let cost_per_episode_first_season := 100000
  let episodes_first_season := 12
  let episodes_seasons_2_to_4 := 18
  let cost_per_episode_other_seasons := 2 * cost_per_episode_first_season
  let episodes_last_season := 24
  let number_of_other_seasons := 4
  let total_cost_first_season := episodes_first_season * cost_per_episode_first_season
  let total_cost_other_seasons := (episodes_seasons_2_to_4 * 3 + episodes_last_season) * cost_per_episode_other_seasons
  total_cost_first_season + total_cost_other_seasons

theorem total_tv_show_cost_correct : total_cost_of_tv_show = 16800000 := by
  sorry

end total_tv_show_cost_correct_l198_198440


namespace number_of_trees_l198_198687

-- Define the yard length and the distance between consecutive trees
def yard_length : ℕ := 300
def distance_between_trees : ℕ := 12

-- Prove that the number of trees planted in the garden is 26
theorem number_of_trees (yard_length distance_between_trees : ℕ) 
  (h1 : yard_length = 300) (h2 : distance_between_trees = 12) : 
  ∃ n : ℕ, n = 26 :=
by
  sorry

end number_of_trees_l198_198687


namespace p_sufficient_not_necessary_q_l198_198133

theorem p_sufficient_not_necessary_q (x : ℝ) :
  (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros h
  cases h with h1 h2
  split
  case left => linarith
  case right => linarith

end p_sufficient_not_necessary_q_l198_198133


namespace rect_length_is_20_l198_198750

-- Define the conditions
def rect_length_four_times_width (l w : ℝ) : Prop := l = 4 * w
def rect_area_100 (l w : ℝ) : Prop := l * w = 100

-- The main theorem to prove
theorem rect_length_is_20 {l w : ℝ} (h1 : rect_length_four_times_width l w) (h2 : rect_area_100 l w) : l = 20 := by
  sorry

end rect_length_is_20_l198_198750


namespace compound_interest_calculation_l198_198533

noncomputable def compoundInterest (P r t : ℝ) : ℝ :=
  P * (1 + r)^t - P

noncomputable def simpleInterest (P r t : ℝ) : ℝ :=
  P * r * t

theorem compound_interest_calculation :
  ∃ P : ℝ, simpleInterest P 0.10 2 = 600 ∧ compoundInterest P 0.10 2 = 630 :=
by
  sorry

end compound_interest_calculation_l198_198533


namespace robot_swap_eventually_non_swappable_l198_198172

theorem robot_swap_eventually_non_swappable (n : ℕ) (a : Fin n → ℕ) :
  ∃ t : ℕ, ∀ i : Fin (n - 1), ¬ (a (⟨i, sorry⟩ : Fin n) > a (⟨i + 1, sorry⟩ : Fin n)) ↔ n > 1 :=
sorry

end robot_swap_eventually_non_swappable_l198_198172


namespace virginia_sweettarts_l198_198433

theorem virginia_sweettarts (total_sweettarts : ℕ) (sweettarts_per_person : ℕ) (friends : ℕ) (sweettarts_left : ℕ) 
  (h1 : total_sweettarts = 13) 
  (h2 : sweettarts_per_person = 3) 
  (h3 : total_sweettarts = sweettarts_per_person * (friends + 1) + sweettarts_left) 
  (h4 : sweettarts_left < sweettarts_per_person) :
  friends = 3 :=
by
  sorry

end virginia_sweettarts_l198_198433


namespace rotated_angle_540_deg_l198_198226

theorem rotated_angle_540_deg (θ : ℝ) (h : θ = 60) : 
  (θ - 540) % 360 % 180 = 60 :=
by
  sorry

end rotated_angle_540_deg_l198_198226


namespace minimize_expression_l198_198410

theorem minimize_expression :
  ∀ n : ℕ, 0 < n → (n = 6 ↔ ∀ m : ℕ, 0 < m → (n ≤ (2 * (m + 9))/(m))) := 
by
  sorry

end minimize_expression_l198_198410


namespace electric_energy_consumption_l198_198553

def power_rating_fan : ℕ := 75
def hours_per_day : ℕ := 8
def days_per_month : ℕ := 30
def watts_to_kWh : ℕ := 1000

theorem electric_energy_consumption : power_rating_fan * hours_per_day * days_per_month / watts_to_kWh = 18 := by
  sorry

end electric_energy_consumption_l198_198553


namespace yura_finishes_on_september_12_l198_198695

theorem yura_finishes_on_september_12
  (total_problems : ℕ) (initial_problem_solving_date : ℕ) (initial_remaining_problems : ℕ) (problems_on_sep_6 : ℕ)
  (problems_on_sep_7 : ℕ) (problems_on_sep_8 : ℕ):
  total_problems = 91 →
  initial_problem_solving_date = 6 →
  initial_remaining_problems = 46 →
  problems_on_sep_6 = 16 →
  problems_on_sep_7 = 15 →
  problems_on_sep_8 = 14 →
  (∑ i in finset.range 7, problems_on_sep_6 - i) = total_problems :=
begin
  sorry -- Proof to be provided
end

end yura_finishes_on_september_12_l198_198695


namespace quadratic_inequality_solution_l198_198849

theorem quadratic_inequality_solution:
  (∃ p : ℝ, ∀ x : ℝ, x^2 + p * x - 6 < 0 ↔ -3 < x ∧ x < 2) → ∃ p : ℝ, p = 1 :=
by
  intro h
  sorry

end quadratic_inequality_solution_l198_198849


namespace total_spent_target_l198_198454

theorem total_spent_target (face_moisturizer_cost : ℕ) (body_lotion_cost : ℕ) (face_moisturizers_bought : ℕ) (body_lotions_bought : ℕ) (christy_multiplier : ℕ) :
  face_moisturizer_cost = 50 →
  body_lotion_cost = 60 →
  face_moisturizers_bought = 2 →
  body_lotions_bought = 4 →
  christy_multiplier = 2 →
  (face_moisturizers_bought * face_moisturizer_cost + body_lotions_bought * body_lotion_cost) * (1 + christy_multiplier) = 1020 := by
  sorry

end total_spent_target_l198_198454


namespace common_point_geometric_lines_l198_198621

-- Define that a, b, c form a geometric progression given common ratio r
def geometric_prog (a b c r : ℝ) : Prop := b = a * r ∧ c = a * r^2

-- Prove that all lines with the equation ax + by = c pass through the point (-1, 1)
theorem common_point_geometric_lines (a b c r x y : ℝ) (h : geometric_prog a b c r) :
  a * x + b * y = c → (x, y) = (-1, 1) :=
by
  sorry

end common_point_geometric_lines_l198_198621


namespace magician_weeks_worked_l198_198740

theorem magician_weeks_worked
  (hourly_rate : ℕ)
  (hours_per_day : ℕ)
  (total_payment : ℕ)
  (days_per_week : ℕ)
  (h1 : hourly_rate = 60)
  (h2 : hours_per_day = 3)
  (h3 : total_payment = 2520)
  (h4 : days_per_week = 7) :
  total_payment / (hourly_rate * hours_per_day * days_per_week) = 2 := 
by
  -- sorry to skip the proof
  sorry

end magician_weeks_worked_l198_198740


namespace units_digit_of_52_cubed_plus_29_cubed_l198_198941

-- Define the units digit of a number n
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions as definitions in Lean
def units_digit_of_2_cubed : ℕ := units_digit (2^3)  -- 8
def units_digit_of_9_cubed : ℕ := units_digit (9^3)  -- 9

-- The main theorem to prove
theorem units_digit_of_52_cubed_plus_29_cubed : units_digit (52^3 + 29^3) = 7 :=
by
  sorry

end units_digit_of_52_cubed_plus_29_cubed_l198_198941


namespace red_candies_count_l198_198056

def total_candies : ℕ := 3409
def blue_candies : ℕ := 3264

theorem red_candies_count : total_candies - blue_candies = 145 := by
  sorry

end red_candies_count_l198_198056


namespace field_size_l198_198371

theorem field_size
  (cost_per_foot : ℝ)
  (total_money : ℝ)
  (cannot_fence : ℝ)
  (cost_per_foot_eq : cost_per_foot = 30)
  (total_money_eq : total_money = 120000)
  (cannot_fence_eq : cannot_fence > 1000) :
  ∃ (side_length : ℝ), side_length * side_length = 1000000 := 
by
  sorry

end field_size_l198_198371


namespace model_tower_height_l198_198956

theorem model_tower_height (h_real : ℝ) (vol_real : ℝ) (vol_model : ℝ) 
  (h_real_eq : h_real = 60) (vol_real_eq : vol_real = 150000) (vol_model_eq : vol_model = 0.15) :
  (h_real * (vol_model / vol_real)^(1/3) = 0.6) :=
by
  sorry

end model_tower_height_l198_198956


namespace radius_increase_l198_198744

theorem radius_increase (ΔC : ℝ) (ΔC_eq : ΔC = 0.628) : Δr = 0.1 :=
by
  sorry

end radius_increase_l198_198744


namespace bus_driver_hours_worked_l198_198607

-- Definitions based on the problem's conditions.
def regular_rate : ℕ := 20
def regular_hours : ℕ := 40
def overtime_rate : ℕ := regular_rate + (3 * (regular_rate / 4))  -- 75% higher
def total_compensation : ℕ := 1000

-- Theorem statement: The bus driver worked a total of 45 hours last week.
theorem bus_driver_hours_worked : 40 + ((total_compensation - (regular_rate * regular_hours)) / overtime_rate) = 45 := 
by 
  sorry

end bus_driver_hours_worked_l198_198607


namespace orange_marbles_l198_198939

-- Definitions based on the given conditions
def total_marbles : ℕ := 24
def blue_marbles : ℕ := total_marbles / 2
def red_marbles : ℕ := 6

-- The statement to prove: the number of orange marbles is 6
theorem orange_marbles : (total_marbles - (blue_marbles + red_marbles)) = 6 := 
  by 
  sorry

end orange_marbles_l198_198939


namespace remainder_of_large_number_l198_198474

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l198_198474


namespace remainder_123456789012_div_252_l198_198484

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l198_198484


namespace num_ordered_pairs_squares_diff_30_l198_198156

theorem num_ordered_pairs_squares_diff_30 :
  ∃ (n : ℕ), n = 0 ∧
  ∀ (m n: ℕ), 0 < m ∧ 0 < n ∧ m ≥ n ∧ m^2 - n^2 = 30 → false :=
by
  sorry

end num_ordered_pairs_squares_diff_30_l198_198156


namespace fraction_to_decimal_l198_198301

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l198_198301


namespace sequence_even_numbers_sequence_odd_numbers_sequence_square_numbers_sequence_arithmetic_progression_l198_198945

-- Problem 1: Prove the general formula for the sequence of all positive even numbers
theorem sequence_even_numbers (n : ℕ) : ∃ a_n, a_n = 2 * n := by 
  sorry

-- Problem 2: Prove the general formula for the sequence of all positive odd numbers
theorem sequence_odd_numbers (n : ℕ) : ∃ b_n, b_n = 2 * n - 1 := by 
  sorry

-- Problem 3: Prove the general formula for the sequence 1, 4, 9, 16, ...
theorem sequence_square_numbers (n : ℕ) : ∃ a_n, a_n = n^2 := by
  sorry

-- Problem 4: Prove the general formula for the sequence -4, -1, 2, 5, ...
theorem sequence_arithmetic_progression (n : ℕ) : ∃ a_n, a_n = 3 * n - 7 := by
  sorry

end sequence_even_numbers_sequence_odd_numbers_sequence_square_numbers_sequence_arithmetic_progression_l198_198945


namespace find_Allyson_age_l198_198241

variable (Hiram Allyson : ℕ)

theorem find_Allyson_age (h : Hiram = 40)
  (condition : Hiram + 12 = 2 * Allyson - 4) : Allyson = 28 := by
  sorry

end find_Allyson_age_l198_198241


namespace hyperbola_eccentricity_l198_198658

noncomputable def hyperbola_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def midpoint (x1 y1 x2 y2 : ℝ) (mx my : ℝ) : Prop :=
  mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2

theorem hyperbola_eccentricity
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (x1 y1 x2 y2 : ℝ)
  (h_intersection1 : hyperbola_equation x1 y1 a b)
  (h_intersection2 : hyperbola_equation x2 y2 a b)
  (h_slope : y2 - y1 = x2 - x1)
  (mx my : ℝ)
  (h_midpoint : midpoint x1 y1 x2 y2 mx my)
  (hmx : mx = 1)
  (hmy : my = 3) :
  (Real.sqrt ((a^2 + b^2) / b^2) = 2) :=
sorry

end hyperbola_eccentricity_l198_198658


namespace min_value_range_l198_198848

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem min_value_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 2 * a^2 - a - 1) → (0 < a ∧ a ≤ 1) :=
by 
  sorry

end min_value_range_l198_198848


namespace digits_in_2_pow_120_l198_198825

theorem digits_in_2_pow_120 {a b : ℕ} (h : 10^a ≤ 2^200 ∧ 2^200 < 10^b) (ha : a = 60) (hb : b = 61) : 
  ∃ n : ℕ, 10^(n-1) ≤ 2^120 ∧ 2^120 < 10^n ∧ n = 37 :=
by {
  sorry
}

end digits_in_2_pow_120_l198_198825


namespace chair_subsets_with_at_least_three_adjacent_l198_198421

-- Define a structure for representing a circle of chairs
def Circle (n : ℕ) := { k : ℕ // k < n }

-- Condition: We have 12 chairs in a circle
def chairs : finset (Circle 12) :=
  finset.univ

-- Defining adjacency in a circle
def adjacent (c : Circle 12) : finset (Circle 12) :=
  if h : c.val = 11 then
    finset.of_list [⟨0, by linarith⟩, ⟨10, by linarith⟩, ⟨11, by linarith⟩]
  else if h : c.val = 0 then
    finset.of_list [⟨11, by linarith⟩, ⟨0, by linarith⟩, ⟨1, by linarith⟩]
  else
    finset.of_list [⟨c.val - 1, by linarith⟩, ⟨c.val, by linarith⟩, ⟨c.val + 1, by linarith⟩]

-- Define a subset of chairs as "valid" if it contains at least 3 adjacent chairs
def has_at_least_three_adjacent (chair_set : finset (Circle 12)) : Prop :=
  ∃ c, (adjacent c) ⊆ chair_set

-- The theorem to state that the number of "valid" subsets is 2040.
theorem chair_subsets_with_at_least_three_adjacent : 
  finset.card {chair_set : finset (Circle 12) // has_at_least_three_adjacent chair_set} = 2040 := 
by sorry

end chair_subsets_with_at_least_three_adjacent_l198_198421


namespace conditional_prob_l198_198617

noncomputable def prob_A := 0.7
noncomputable def prob_AB := 0.4

theorem conditional_prob : prob_AB / prob_A = 4 / 7 :=
by
  sorry

end conditional_prob_l198_198617


namespace sean_divided_by_julie_is_2_l198_198390

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum as twice the sum of the first 300 natural numbers
def sean_sum : ℕ := 2 * sum_natural 300

-- Define Julie's sum as the sum of the first 300 natural numbers
def julie_sum : ℕ := sum_natural 300

-- Prove that Sean's sum divided by Julie's sum is 2
theorem sean_divided_by_julie_is_2 : sean_sum / julie_sum = 2 := by
  sorry

end sean_divided_by_julie_is_2_l198_198390


namespace hockeyPlayers_count_l198_198171

def numPlayers := 50
def cricketPlayers := 12
def footballPlayers := 11
def softballPlayers := 10

theorem hockeyPlayers_count : 
  let hockeyPlayers := numPlayers - (cricketPlayers + footballPlayers + softballPlayers)
  hockeyPlayers = 17 :=
by
  sorry

end hockeyPlayers_count_l198_198171


namespace problem_statement_l198_198352

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- defining conditions
axiom a1_4_7 : a 1 + a 4 + a 7 = 39
axiom a2_5_8 : a 2 + a 5 + a 8 = 33
axiom is_arithmetic : arithmetic_seq a d

theorem problem_statement : a 5 + a 8 + a 11 = 15 :=
by sorry

end problem_statement_l198_198352


namespace smallest_integer_y_l198_198981

theorem smallest_integer_y (y : ℤ) : (∃ (y : ℤ), (y / 4) + (3 / 7) > (4 / 7) ∧ ∀ (z : ℤ), z < y → (z / 4) + (3 / 7) ≤ (4 / 7)) := 
by
  sorry

end smallest_integer_y_l198_198981


namespace tangent_line_parabola_l198_198850

theorem tangent_line_parabola (P : ℝ × ℝ) (hP : P = (-1, 0)) :
  ∀ x y : ℝ, (y^2 = 4 * x) ∧ (P = (-1, 0)) → (x + y + 1 = 0) ∨ (x - y + 1 = 0) := by
  sorry

end tangent_line_parabola_l198_198850


namespace prism_properties_sum_l198_198550

/-- Prove that the sum of the number of edges, corners, and faces of a rectangular box (prism) with dimensions 2 by 3 by 4 is 26. -/
theorem prism_properties_sum :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 := 
by
  -- Provided conditions and definitions
  let edges := 12
  let corners := 8
  let faces := 6
  -- Summing up these values
  exact rfl

end prism_properties_sum_l198_198550


namespace profit_when_x_is_6_max_profit_l198_198253

noncomputable def design_fee : ℝ := 20000 / 10000
noncomputable def production_cost_per_100 : ℝ := 10000 / 10000

noncomputable def P (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x - 0.8
  else 14.7 - 9 / (x - 3)

noncomputable def cost_of_x_sets (x : ℝ) : ℝ :=
  design_fee + x * production_cost_per_100

noncomputable def profit (x : ℝ) : ℝ :=
  P x - cost_of_x_sets x

theorem profit_when_x_is_6 :
  profit 6 = 3.7 := sorry

theorem max_profit :
  ∀ x : ℝ, profit x ≤ 3.7 := sorry

end profit_when_x_is_6_max_profit_l198_198253


namespace average_age_decrease_l198_198041

-- Define the conditions as given in the problem
def original_strength : ℕ := 12
def new_students : ℕ := 12

def original_avg_age : ℕ := 40
def new_students_avg_age : ℕ := 32

def decrease_in_avg_age (O N : ℕ) (OA NA : ℕ) : ℕ :=
  let total_original_age := O * OA
  let total_new_students_age := N * NA
  let total_students := O + N
  let new_avg_age := (total_original_age + total_new_students_age) / total_students
  OA - new_avg_age

theorem average_age_decrease :
  decrease_in_avg_age original_strength new_students original_avg_age new_students_avg_age = 4 :=
sorry

end average_age_decrease_l198_198041


namespace p_sufficient_not_necessary_q_l198_198129

def p (x : ℝ) := 0 < x ∧ x < 2
def q (x : ℝ) := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l198_198129


namespace sean_and_julie_sums_l198_198385

theorem sean_and_julie_sums :
  let seanSum := 2 * (Finset.range 301).sum
  let julieSum := (Finset.range 301).sum
  seanSum / julieSum = 2 := by
  sorry

end sean_and_julie_sums_l198_198385


namespace right_angle_triangle_l198_198401

theorem right_angle_triangle (a b c : ℝ) (h : (a + b) ^ 2 - c ^ 2 = 2 * a * b) : a ^ 2 + b ^ 2 = c ^ 2 := 
by
  sorry

end right_angle_triangle_l198_198401


namespace find_abc_l198_198528

-- Definitions based on given conditions
variables (a b c : ℝ)
variable (h1 : a * b = 30 * (3 ^ (1/3)))
variable (h2 : a * c = 42 * (3 ^ (1/3)))
variable (h3 : b * c = 18 * (3 ^ (1/3)))

-- Formal statement of the proof problem
theorem find_abc : a * b * c = 90 * Real.sqrt 3 :=
by
  sorry

end find_abc_l198_198528


namespace geometry_problem_l198_198351

-- Definitions for points and segments based on given conditions
variables {O A B C D E F G : Type} [Inhabited O] [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F] [Inhabited G]

-- Lengths of segments based on given conditions
variables (DE EG : ℝ)
variable (BG : ℝ)

-- Given lengths
def given_lengths : Prop :=
  DE = 5 ∧ EG = 3

-- Goal to prove
def goal : Prop :=
  BG = 12

-- The theorem combining conditions and the goal
theorem geometry_problem (h : given_lengths DE EG) : goal BG :=
  sorry

end geometry_problem_l198_198351


namespace amount_A_l198_198034

theorem amount_A (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 527) : A = 62 := by
  sorry

end amount_A_l198_198034


namespace min_abs_expr1_min_abs_expr2_l198_198402

theorem min_abs_expr1 (x : ℝ) : |x - 4| + |x + 2| ≥ 6 := sorry

theorem min_abs_expr2 (x : ℝ) : |(5 / 6) * x - 1| + |(1 / 2) * x - 1| + |(2 / 3) * x - 1| ≥ 1 / 2 := sorry

end min_abs_expr1_min_abs_expr2_l198_198402


namespace jackie_walks_daily_l198_198176

theorem jackie_walks_daily (x : ℝ) :
  (∀ t : ℕ, t = 6 →
    6 * x = 6 * 1.5 + 3) →
  x = 2 :=
by
  sorry

end jackie_walks_daily_l198_198176


namespace opposite_reciprocal_abs_value_l198_198839

theorem opposite_reciprocal_abs_value (a b c d m : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : abs m = 3) : 
  (a + b) / m + c * d + m = 4 ∨ (a + b) / m + c * d + m = -2 := by 
  sorry

end opposite_reciprocal_abs_value_l198_198839


namespace total_amount_paid_l198_198370

-- Define the conditions of the problem
def cost_without_discount (quantity : ℕ) (unit_price : ℚ) : ℚ :=
  quantity * unit_price

def cost_with_discount (quantity : ℕ) (unit_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_cost := cost_without_discount quantity unit_price
  total_cost - (total_cost * discount_rate)

-- Define each category's cost after discount
def pens_cost : ℚ := cost_with_discount 7 1.5 0.10
def notebooks_cost : ℚ := cost_without_discount 4 5
def water_bottles_cost : ℚ := cost_with_discount 2 8 0.30
def backpack_cost : ℚ := cost_with_discount 1 25 0.15
def socks_cost : ℚ := cost_with_discount 3 3 0.25

-- Prove the total amount paid is $68.65
theorem total_amount_paid : pens_cost + notebooks_cost + water_bottles_cost + backpack_cost + socks_cost = 68.65 := by
  sorry

end total_amount_paid_l198_198370


namespace min_value_l198_198578

theorem min_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (m n : ℝ) (h3 : m > 0) (h4 : n > 0) 
(h5 : m + 4 * n = 1) : 
  1 / m + 4 / n ≥ 25 :=
by
  sorry

end min_value_l198_198578


namespace yura_finishes_on_september_12_l198_198696

theorem yura_finishes_on_september_12
  (total_problems : ℕ) (initial_problem_solving_date : ℕ) (initial_remaining_problems : ℕ) (problems_on_sep_6 : ℕ)
  (problems_on_sep_7 : ℕ) (problems_on_sep_8 : ℕ):
  total_problems = 91 →
  initial_problem_solving_date = 6 →
  initial_remaining_problems = 46 →
  problems_on_sep_6 = 16 →
  problems_on_sep_7 = 15 →
  problems_on_sep_8 = 14 →
  (∑ i in finset.range 7, problems_on_sep_6 - i) = total_problems :=
begin
  sorry -- Proof to be provided
end

end yura_finishes_on_september_12_l198_198696


namespace no_possible_salary_distribution_l198_198686

theorem no_possible_salary_distribution (x y z : ℕ) (h1 : x + y + z = 13) (h2 : x + 3 * y + 5 * z = 200) : false :=
by {
  -- Proof goes here
  sorry
}

end no_possible_salary_distribution_l198_198686


namespace inequality_proof_l198_198913

theorem inequality_proof (a b c : ℝ) : 
  1 < (a / (Real.sqrt (a^2 + b^2)) + b / (Real.sqrt (b^2 + c^2)) + 
  c / (Real.sqrt (c^2 + a^2))) ∧ 
  (a / (Real.sqrt (a^2 + b^2)) + b / (Real.sqrt (b^2 + c^2)) + 
  c / (Real.sqrt (c^2 + a^2))) ≤ (3 * Real.sqrt 2 / 2) :=
by
  sorry

end inequality_proof_l198_198913


namespace probability_red_even_and_green_gt_3_l198_198429

variable {Ω : Type} [Probability Ω]

/-- Two 6-sided dice, one red and one green are rolled. -/
def roll_red_is_even (ω : Ω) : Prop :=
  let red := (nat % 6) ω in
  red = 2 ∨ red = 4 ∨ red = 6

def roll_green_greater_than_3 (ω : Ω) : Prop :=
  let green := (nat % 6) ω in
  green = 4 ∨ green = 5 ∨ green = 6

theorem probability_red_even_and_green_gt_3 :
  Prob (λ ω, roll_red_is_even ω ∧ roll_green_greater_than_3 ω) = 1 / 4 :=
by
  sorry

end probability_red_even_and_green_gt_3_l198_198429


namespace fraction_to_decimal_l198_198300

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l198_198300


namespace nabla_value_l198_198929

def nabla (a b c d : ℕ) : ℕ := a * c + b * d

theorem nabla_value : nabla 3 1 4 2 = 14 :=
by
  sorry

end nabla_value_l198_198929


namespace perfect_matchings_same_weight_l198_198026

variables {V : Type*} [fintype V] [decidable_eq V]
variables (A B : finset V) (n : ℕ)
variables (G : simple_graph V) [weight_condition: ∀ e ∈ G.edge_set, 0 < G.edge_weight e]
variables (G' : simple_graph V) [fresh_condition: G'.edge_set = {e | ∃ M, is_min_weight_perfect_matching G M ∧ e ∈ M.edge_set}]

open simple_graph 

theorem perfect_matchings_same_weight (h_size : A.card = n) (h_bipartite : is_bipartite G A B) :
  ∀ M1 M2, is_perfect_matching G' M1 → is_perfect_matching G' M2 → M1.weight (G'.edge_weight) = M2.weight (G'.edge_weight) :=
sorry

end perfect_matchings_same_weight_l198_198026


namespace trapezium_area_correct_l198_198437

-- Define the lengths of the parallel sides and the distance between them
def a := 24  -- length of the first parallel side in cm
def b := 14  -- length of the second parallel side in cm
def h := 18  -- distance between the parallel sides in cm

-- Define the area calculation function for the trapezium
def trapezium_area (a b h : ℕ) : ℕ :=
  1 / 2 * (a + b) * h

-- The theorem to prove that the area of the given trapezium is 342 square centimeters
theorem trapezium_area_correct : trapezium_area a b h = 342 :=
  sorry

end trapezium_area_correct_l198_198437


namespace general_formula_no_arithmetic_sequence_l198_198188

-- Given condition
def Sn (a : ℕ → ℤ) (n : ℕ) : ℤ := 2 * a n - 3 * n

-- Theorem 1: General formula for the sequence a_n
theorem general_formula (a : ℕ → ℤ) (n : ℕ) (h : ∀ n, Sn a n = 2 * a n - 3 * n) : 
  a n = 3 * 2^n - 3 :=
sorry

-- Theorem 2: No three terms of the sequence form an arithmetic sequence
theorem no_arithmetic_sequence (a : ℕ → ℤ) (x y z : ℕ) (h : ∀ n, Sn a n = 2 * a n - 3 * n) (hx : x < y) (hy : y < z) :
  ¬ (a x + a z = 2 * a y) :=
sorry

end general_formula_no_arithmetic_sequence_l198_198188


namespace my_op_identity_l198_198535

def my_op (a b : ℕ) : ℕ := a + b + a * b

theorem my_op_identity (a : ℕ) : my_op (my_op a 1) 2 = 6 * a + 5 :=
by
  sorry

end my_op_identity_l198_198535


namespace smaller_acute_angle_is_20_degrees_l198_198016

noncomputable def smaller_acute_angle (x : ℝ) : Prop :=
  let θ1 := 7 * x
  let θ2 := 2 * x
  θ1 + θ2 = 90 ∧ θ2 = 20

theorem smaller_acute_angle_is_20_degrees : ∃ x : ℝ, smaller_acute_angle x :=
  sorry

end smaller_acute_angle_is_20_degrees_l198_198016


namespace haley_collected_cans_l198_198342

theorem haley_collected_cans (C : ℕ) (h : C - 7 = 2) : C = 9 :=
by {
  sorry
}

end haley_collected_cans_l198_198342


namespace smallest_c_value_l198_198723

theorem smallest_c_value (c d : ℝ) (h_nonneg_c : 0 ≤ c) (h_nonneg_d : 0 ≤ d)
  (h_eq_cos : ∀ x : ℤ, Real.cos (c * x - d) = Real.cos (35 * x)) :
  c = 35 := by
  sorry

end smallest_c_value_l198_198723


namespace ext_9_implication_l198_198998

theorem ext_9_implication (a b : ℝ) (h1 : 3 + 2 * a + b = 0) (h2 : 1 + a + b + a^2 = 10) : (2 : ℝ)^3 + a * (2 : ℝ)^2 + b * (2 : ℝ) + a^2 - 1 = 17 := by
  sorry

end ext_9_implication_l198_198998


namespace first_house_gets_90_bottles_l198_198794

def bottles_of_drinks (total_bottles bottles_cider_only bottles_beer_only : ℕ) : ℕ :=
  let bottles_mixture := total_bottles - bottles_cider_only - bottles_beer_only
  let first_house_cider := bottles_cider_only / 2
  let first_house_beer := bottles_beer_only / 2
  let first_house_mixture := bottles_mixture / 2
  first_house_cider + first_house_beer + first_house_mixture
  
theorem first_house_gets_90_bottles :
  bottles_of_drinks 180 40 80 = 90 :=
by
  rw [bottles_of_drinks]
  sorry

end first_house_gets_90_bottles_l198_198794


namespace value_of_b_l198_198958

theorem value_of_b (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x ≠ 0, f x = -1 / x) (h2 : f a = -1 / 3) (h3 : f (a * b) = 1 / 6) : b = -2 :=
sorry

end value_of_b_l198_198958


namespace y_coordinate_of_point_l198_198020

theorem y_coordinate_of_point (x y : ℝ) (m : ℝ)
  (h₁ : x = 10)
  (h₂ : y = m * x + -2)
  (m_def : m = (0 - (-4)) / (4 - (-4)))
  (h₃ : y = 3) : y = 3 :=
sorry

end y_coordinate_of_point_l198_198020


namespace chess_match_duration_l198_198200

def time_per_move_polly := 28
def time_per_move_peter := 40
def total_moves := 30
def moves_per_player := total_moves / 2

def Polly_time := moves_per_player * time_per_move_polly
def Peter_time := moves_per_player * time_per_move_peter
def total_time_seconds := Polly_time + Peter_time
def total_time_minutes := total_time_seconds / 60

theorem chess_match_duration : total_time_minutes = 17 := by
  sorry

end chess_match_duration_l198_198200


namespace remainder_123456789012_div_252_l198_198480

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l198_198480


namespace remainder_div_252_l198_198466

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l198_198466


namespace least_possible_sum_24_l198_198897

noncomputable def leastSum (m n : ℕ) 
  (h1: m > 0)
  (h2: n > 0)
  (h3: Nat.gcd (m + n) 231 = 1)
  (h4: m^m % n^n = 0)
  (h5: ¬ (m % n = 0))
  : ℕ :=
  m + n

theorem least_possible_sum_24 : ∃ (m n : ℕ), 
  m > 0 ∧ 
  n > 0 ∧ 
  Nat.gcd (m + n) 231 = 1 ∧ 
  m^m % n^n = 0 ∧ 
  ¬(m % n = 0) ∧ 
  leastSum m n m_pos n_pos gcd_cond mult_cond not_mult_cond = 24 :=
begin
  sorry
end

end least_possible_sum_24_l198_198897


namespace smallest_pos_int_div_by_four_primes_l198_198069

theorem smallest_pos_int_div_by_four_primes : 
  ∃ (n : ℕ), (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ (∀ m p1 p2 p3 p4, [p1, p2, p3, p4] = [2, 3, 5, 7] → (∀ p ∈ [p1, p2, p3, p4], p ∣ m) → n ≤ m) :=
by
  let n := 2 * 3 * 5 * 7
  use n
  split
  {
    intros p hp
    fin_cases hp <;> simp [n]
  }
  {
    intros m p1 p2 p3 p4 hps div_con
    fin_cases hps <;> simp [n] at div_con
    have : n ≤ m := sorry -- prove that n is the smallest such integer
    exact this
  }

end smallest_pos_int_div_by_four_primes_l198_198069


namespace Joey_age_l198_198717

theorem Joey_age (J B : ℕ) (h1 : J + 5 = B) (h2 : J - 4 = B - J) : J = 9 :=
by 
  sorry

end Joey_age_l198_198717


namespace box_volume_increase_l198_198962

theorem box_volume_increase (l w h : ℝ)
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
  sorry

end box_volume_increase_l198_198962


namespace complement_union_l198_198000

variable (x : ℝ)

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≤ 1}
def P : Set ℝ := {x | x ≥ 2}

theorem complement_union (x : ℝ) : x ∈ U → (¬ (x ∈ M ∨ x ∈ P)) ↔ (1 < x ∧ x < 2) := 
by
  sorry

end complement_union_l198_198000


namespace shortest_side_length_rectangular_solid_geometric_progression_l198_198589

theorem shortest_side_length_rectangular_solid_geometric_progression
  (b s : ℝ)
  (h1 : (b^3 / s) = 512)
  (h2 : 2 * ((b^2 / s) + (b^2 * s) + b^2) = 384)
  : min (b / s) (min b (b * s)) = 8 := 
sorry

end shortest_side_length_rectangular_solid_geometric_progression_l198_198589


namespace find_distance_city_A_B_l198_198888

-- Variables and givens
variable (D : ℝ)

-- Conditions from the problem
variable (JohnSpeed : ℝ := 40) (LewisSpeed : ℝ := 60)
variable (MeetDistance : ℝ := 160)
variable (TimeJohn : ℝ := (D - MeetDistance) / JohnSpeed)
variable (TimeLewis : ℝ := (D + MeetDistance) / LewisSpeed)

-- Lean 4 theorem statement for the proof
theorem find_distance_city_A_B :
  TimeJohn = TimeLewis → D = 800 :=
by
  sorry

end find_distance_city_A_B_l198_198888


namespace find_b_d_l198_198052

theorem find_b_d (b d : ℕ) (h1 : b + d = 41) (h2 : b < d) : 
  (∃! x, b * x * x + 24 * x + d = 0) → (b = 9 ∧ d = 32) :=
by 
  sorry

end find_b_d_l198_198052


namespace remainder_123456789012_mod_252_l198_198495

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l198_198495


namespace solution_set_of_inequality_l198_198407

theorem solution_set_of_inequality :
  { x : ℝ | (x + 3) * (6 - x) ≥ 0 } = { x : ℝ | -3 ≤ x ∧ x ≤ 6 } :=
sorry

end solution_set_of_inequality_l198_198407


namespace triangle_XYZ_PQZ_lengths_l198_198713

theorem triangle_XYZ_PQZ_lengths :
  ∀ (X Y Z P Q : Type) (d_XZ d_YZ d_PQ : ℝ),
  d_XZ = 9 → d_YZ = 12 → d_PQ = 3 →
  ∀ (XY YP : ℝ),
  XY = Real.sqrt (d_XZ^2 + d_YZ^2) →
  YP = (d_PQ / d_XZ) * d_YZ →
  YP = 4 :=
by
  intros X Y Z P Q d_XZ d_YZ d_PQ hXZ hYZ hPQ XY YP hXY hYP
  -- Skipping detailed proof
  sorry

end triangle_XYZ_PQZ_lengths_l198_198713


namespace no_full_conspiracies_in_same_lab_l198_198219

theorem no_full_conspiracies_in_same_lab
(six_conspiracies : Finset (Finset (Fin 10)))
(h_conspiracies : ∀ c ∈ six_conspiracies, c.card = 3)
(h_total : six_conspiracies.card = 6) :
  ∃ (lab1 lab2 : Finset (Fin 10)), lab1 ∩ lab2 = ∅ ∧ lab1 ∪ lab2 = Finset.univ ∧ ∀ c ∈ six_conspiracies, ¬(c ⊆ lab1 ∨ c ⊆ lab2) :=
by
  sorry

end no_full_conspiracies_in_same_lab_l198_198219


namespace luna_badges_correct_l198_198855

-- conditions
def total_badges : ℕ := 83
def hermione_badges : ℕ := 14
def celestia_badges : ℕ := 52

-- question and answer
theorem luna_badges_correct : total_badges - (hermione_badges + celestia_badges) = 17 :=
by
  sorry

end luna_badges_correct_l198_198855


namespace largest_x_value_l198_198037

noncomputable def quadratic_eq (x : ℝ) : Prop :=
  3 * (9 * x^2 + 15 * x + 20) = x * (9 * x - 60)

theorem largest_x_value (x : ℝ) :
  quadratic_eq x → x = - ((35 - Real.sqrt 745) / 12) ∨
  x = - ((35 + Real.sqrt 745) / 12) :=
by
  intro h
  sorry

end largest_x_value_l198_198037


namespace coefficient_of_x31_is_148_l198_198106

theorem coefficient_of_x31_is_148 (x : ℝ) : 
  (coeff ((1 - x ^ 31) * (1 - x ^ 13) ^ 2 / (1 - x) ^ 3) 31 = 148) :=
sorry

end coefficient_of_x31_is_148_l198_198106


namespace spell_theer_incorrect_probability_l198_198446

theorem spell_theer_incorrect_probability :
  let letters := ['h', 'r', 't', 'e', 'e']
  let target_word := "theer"
  let total_arrangements := 5! / (2! * (5 - 2)!) * 3!
  let correct_arrangements := 1
  let incorrect_probability := (total_arrangements - correct_arrangements) / total_arrangements
  incorrect_probability = 59 / 60 := by
  sorry

end spell_theer_incorrect_probability_l198_198446


namespace cyclic_path_1310_to_1315_l198_198636

theorem cyclic_path_1310_to_1315 :
  ∀ (n : ℕ), (n % 6 = 2 → (n + 5) % 6 = 3) :=
by
  sorry

end cyclic_path_1310_to_1315_l198_198636


namespace polynomial_evaluation_l198_198007

theorem polynomial_evaluation (a : ℝ) (h : a^2 + a - 1 = 0) : a^3 + 2 * a^2 + 2 = 3 :=
by
  sorry

end polynomial_evaluation_l198_198007


namespace express_1997_using_elevent_fours_l198_198780

def number_expression_uses_eleven_fours : Prop :=
  (4 * 444 + 44 * 4 + 44 + 4 / 4 = 1997)
  
theorem express_1997_using_elevent_fours : number_expression_uses_eleven_fours :=
by
  sorry

end express_1997_using_elevent_fours_l198_198780


namespace marek_sequence_sum_l198_198249

theorem marek_sequence_sum (x : ℝ) :
  let a := x
  let b := (a + 4) / 4 - 4
  let c := (b + 4) / 4 - 4
  let d := (c + 4) / 4 - 4
  (a + 4) / 4 * 4 + (b + 4) / 4 * 4 + (c + 4) / 4 * 4 + (d + 4) / 4 * 4 = 80 →
  x = 38 :=
by
  sorry

end marek_sequence_sum_l198_198249


namespace carol_pennies_l198_198345

variable (a c : ℕ)

theorem carol_pennies (h₁ : c + 2 = 4 * (a - 2)) (h₂ : c - 2 = 3 * (a + 2)) : c = 62 :=
by
  sorry

end carol_pennies_l198_198345


namespace activity_participants_l198_198123

variable (A B C D : Prop)

theorem activity_participants (h1 : A → B) (h2 : ¬C → ¬B) (h3 : C → ¬D) : B ∧ C ∧ ¬A ∧ ¬D :=
by
  sorry

end activity_participants_l198_198123


namespace least_pos_int_divisible_by_four_smallest_primes_l198_198066

theorem least_pos_int_divisible_by_four_smallest_primes : 
  ∃ n : ℕ, (∀ p ∣ n, prime p → p ∈ {2, 3, 5, 7}) ∧ n = 2 * 3 * 5 * 7 := 
sorry

end least_pos_int_divisible_by_four_smallest_primes_l198_198066


namespace find_c_l198_198681

theorem find_c (a : ℕ) (c : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 5 = 3 ^ 3 * 5 ^ 2 * 7 ^ 2 * 11 ^ 2 * 13 * c) : 
  c = 385875 := by 
  sorry

end find_c_l198_198681


namespace theatre_lost_revenue_l198_198960

def ticket_price (category : String) : Float :=
  match category with
  | "general" => 10.0
  | "children" => 6.0
  | "senior" => 8.0
  | "veteran" => 8.0  -- $10.00 - $2.00 discount
  | _ => 0.0

def vip_price (base_price : Float) : Float :=
  base_price + 5.0

def calculate_revenue_sold : Float :=
  let general_revenue := 12 * ticket_price "general" + 3 * (vip_price $ ticket_price "general") / 2
  let children_revenue := 3 * ticket_price "children" + vip_price (ticket_price "children")
  let senior_revenue := 4 * ticket_price "senior" + (vip_price (ticket_price "senior")) / 2
  let veteran_revenue := 2 * ticket_price "veteran" + vip_price (ticket_price "veteran")
  general_revenue + children_revenue + senior_revenue + veteran_revenue

def potential_total_revenue : Float :=
  40 * ticket_price "general" + 10 * vip_price (ticket_price "general")

def potential_revenue_lost : Float :=
  potential_total_revenue - calculate_revenue_sold

theorem theatre_lost_revenue : potential_revenue_lost = 224.0 :=
  sorry

end theatre_lost_revenue_l198_198960


namespace range_of_a_l198_198537

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) :=
sorry

end range_of_a_l198_198537


namespace find_n_l198_198778

theorem find_n (n : ℕ) : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ n → n = 29 := by
  intros h
  sorry

end find_n_l198_198778


namespace probability_of_rolling_perfect_square_l198_198394

theorem probability_of_rolling_perfect_square :
  (3 / 12 : ℚ) = 1 / 4 :=
by
  sorry

end probability_of_rolling_perfect_square_l198_198394


namespace total_cards_needed_l198_198738

def red_card_credits := 3
def blue_card_credits := 5
def total_credits := 84
def red_cards := 8

theorem total_cards_needed :
  red_card_credits * red_cards + blue_card_credits * (total_credits - red_card_credits * red_cards) / blue_card_credits = 20 := by
  sorry

end total_cards_needed_l198_198738


namespace total_fish_bought_l198_198033

theorem total_fish_bought (gold_fish blue_fish : Nat) (h1 : gold_fish = 15) (h2 : blue_fish = 7) : gold_fish + blue_fish = 22 := by
  sorry

end total_fish_bought_l198_198033


namespace largest_initial_number_l198_198883

theorem largest_initial_number :
  ∃ n : ℕ, 
    (∀ (a1 a2 a3 a4 a5 : ℕ),
      (n + a1).gcd a1 = 1 ∧
      (n + a1 + a2).gcd a2 = 1 ∧
      (n + a1 + a2 + a3).gcd a3 = 1 ∧
      (n + a1 + a2 + a3 + a4).gcd a4 = 1 ∧
      (n + a1 + a2 + a3 + a4 + a5).gcd a5 = 1 ∧
      n + a1 + a2 + a3 + a4 + a5 = 200) 
    → n = 189 :=
begin
  sorry
end

end largest_initial_number_l198_198883


namespace choosing_officers_l198_198086

noncomputable def total_ways_to_choose_officers (members : List String) (boys : ℕ) (girls : ℕ) : ℕ :=
  let total_members := boys + girls
  let president_choices := total_members
  let vice_president_choices := boys - 1 + girls - 1
  let remaining_members := total_members - 2
  president_choices * vice_president_choices * remaining_members

theorem choosing_officers (members : List String) (boys : ℕ) (girls : ℕ) :
  boys = 15 → girls = 15 → members.length = 30 → total_ways_to_choose_officers members boys girls = 11760 :=
by
  intros hboys hgirls htotal
  rw [hboys, hgirls]
  sorry

end choosing_officers_l198_198086


namespace arithmetic_mean_difference_l198_198159

-- Definitions and conditions
variable (p q r : ℝ)
variable (h1 : (p + q) / 2 = 10)
variable (h2 : (q + r) / 2 = 26)

-- Theorem statement
theorem arithmetic_mean_difference : r - p = 32 := by
  -- Proof goes here
  sorry

end arithmetic_mean_difference_l198_198159


namespace correct_multiplication_value_l198_198615

theorem correct_multiplication_value (N : ℝ) (x : ℝ) : 
  (0.9333333333333333 = (N * x - N / 5) / (N * x)) → 
  x = 3 := 
by 
  sorry

end correct_multiplication_value_l198_198615


namespace exists_h_not_divisible_l198_198285

noncomputable def h : ℝ := 1969^2 / 1968

theorem exists_h_not_divisible (h := 1969^2 / 1968) :
  ∃ (h : ℝ), ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
by
  use h
  intro n
  sorry

end exists_h_not_divisible_l198_198285


namespace domain_of_function_l198_198400

theorem domain_of_function :
  {x : ℝ | 4 - x^2 ≥ 0 ∧ x ≠ 0} = {x : ℝ | -2 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_function_l198_198400


namespace fraction_to_decimal_l198_198307

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198307


namespace remainder_of_large_number_l198_198475

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l198_198475


namespace find_m_find_tangent_lines_l198_198953

section problem1
variables (a m : ℝ)
def f (x : ℝ) : ℝ := (1 / 2) * x^2 - (a + m) * x + a * real.log x
def f' (x : ℝ) : ℝ := deriv f x

theorem find_m (h : f' 1 = 0) : m = 1 :=
sorry
end problem1

section problem2
def f2 (x : ℝ) : ℝ := x^3 + 3*x^2 - 9*x - 3
def f2' (x : ℝ) : ℝ := deriv f2 x
def line_perpendicular_slope := -9
def tangent_lines (x0 : ℝ) (y : ℝ) : Prop :=
(9 * x0 + y + 3 = 0) ∨ (9 * x0 + y - 18 = 0)

theorem find_tangent_lines (x0 : ℝ) (h : f2' x0 = line_perpendicular_slope) : tangent_lines x0 (f2 x0) :=
sorry
end problem2

end find_m_find_tangent_lines_l198_198953


namespace average_sleep_hours_l198_198152

theorem average_sleep_hours (h_monday: ℕ) (h_tuesday: ℕ) (h_wednesday: ℕ) (h_thursday: ℕ) (h_friday: ℕ)
  (h_monday_eq: h_monday = 8) (h_tuesday_eq: h_tuesday = 7) (h_wednesday_eq: h_wednesday = 8)
  (h_thursday_eq: h_thursday = 10) (h_friday_eq: h_friday = 7) :
  (h_monday + h_tuesday + h_wednesday + h_thursday + h_friday) / 5 = 8 :=
by
  sorry

end average_sleep_hours_l198_198152


namespace increasing_sequence_a1_range_l198_198863

theorem increasing_sequence_a1_range
  (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = (4 * a n - 2) / (a n + 1))
  (strictly_increasing : ∀ n, a (n + 1) > a n) :
  1 < a 1 ∧ a 1 < 2 :=
sorry

end increasing_sequence_a1_range_l198_198863


namespace time_ratio_xiao_ming_schools_l198_198642

theorem time_ratio_xiao_ming_schools
  (AB BC CD : ℝ) 
  (flat_speed uphill_speed downhill_speed : ℝ)
  (h1 : AB + BC + CD = 1) 
  (h2 : AB / BC = 1 / 2)
  (h3 : BC / CD = 2 / 1)
  (h4 : flat_speed / uphill_speed = 3 / 2)
  (h5 : uphill_speed / downhill_speed = 2 / 4) :
  (AB / flat_speed + BC / uphill_speed + CD / downhill_speed) / 
  (AB / flat_speed + BC / downhill_speed + CD / uphill_speed) = 19 / 16 :=
by
  sorry

end time_ratio_xiao_ming_schools_l198_198642


namespace xiao_ming_fails_the_test_probability_l198_198013

def probability_scoring_above_80 : ℝ := 0.69
def probability_scoring_between_70_and_79 : ℝ := 0.15
def probability_scoring_between_60_and_69 : ℝ := 0.09

theorem xiao_ming_fails_the_test_probability :
  1 - (probability_scoring_above_80 + probability_scoring_between_70_and_79 + probability_scoring_between_60_and_69) = 0.07 :=
by
  sorry

end xiao_ming_fails_the_test_probability_l198_198013


namespace p_sufficient_but_not_necessary_q_l198_198131

theorem p_sufficient_but_not_necessary_q :
  ∀ x : ℝ, (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros x hx
  cases hx with h1 h2
  apply And.intro
  apply lt_of_lt_of_le h1
  linarith
  apply h2

end p_sufficient_but_not_necessary_q_l198_198131


namespace quotient_remainder_base5_l198_198317

theorem quotient_remainder_base5 (n m : ℕ) 
    (hn : n = 3 * 5^3 + 2 * 5^2 + 3 * 5^1 + 2)
    (hm : m = 2 * 5^1 + 1) :
    n / m = 40 ∧ n % m = 2 :=
by
  sorry

end quotient_remainder_base5_l198_198317


namespace quilt_width_l198_198179

-- Definitions according to the conditions
def quilt_length : ℕ := 16
def patch_area : ℕ := 4
def first_10_patches_cost : ℕ := 100
def total_cost : ℕ := 450
def remaining_budget : ℕ := total_cost - first_10_patches_cost
def cost_per_additional_patch : ℕ := 5
def num_additional_patches : ℕ := remaining_budget / cost_per_additional_patch
def total_patches : ℕ := 10 + num_additional_patches
def total_area : ℕ := total_patches * patch_area

-- Theorem statement
theorem quilt_width :
  (total_area / quilt_length) = 20 :=
by
  sorry

end quilt_width_l198_198179


namespace quadratic_real_roots_l198_198507

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) →
  (m ≤ 4 ∧ m ≠ 3) :=
sorry

end quadratic_real_roots_l198_198507


namespace determine_velocities_l198_198236

theorem determine_velocities (V1 V2 : ℝ) (h1 : 60 / V2 = 60 / V1 + 5) (h2 : |V1 - V2| = 1)
  (h3 : 0 < V1) (h4 : 0 < V2) : V1 = 4 ∧ V2 = 3 :=
by
  sorry

end determine_velocities_l198_198236


namespace eval_expr_l198_198459

theorem eval_expr : (3 : ℚ) / (2 - (5 / 4)) = 4 := by
  sorry

end eval_expr_l198_198459


namespace calculate_expression_l198_198634

theorem calculate_expression :
  |-2*Real.sqrt 3| - (1 - Real.pi)^0 + 2*Real.cos (Real.pi / 6) + (1 / 4)^(-1 : ℤ) = 3 * Real.sqrt 3 + 3 :=
by
  sorry

end calculate_expression_l198_198634


namespace hybrids_with_full_headlights_l198_198168

-- Definitions for the conditions
def total_cars : ℕ := 600
def percentage_hybrids : ℕ := 60
def percentage_one_headlight : ℕ := 40

-- The proof statement
theorem hybrids_with_full_headlights :
  (percentage_hybrids * total_cars) / 100 - (percentage_one_headlight * (percentage_hybrids * total_cars) / 100) / 100 = 216 :=
by
  sorry

end hybrids_with_full_headlights_l198_198168


namespace sean_div_julie_l198_198382

-- Define the sum of the first n integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum: sum of even integers from 2 to 600
def sean_sum : ℕ := 2 * sum_first_n 300

-- Define Julie's sum: sum of integers from 1 to 300
def julie_sum : ℕ := sum_first_n 300

-- Prove that Sean's sum divided by Julie's sum equals 2
theorem sean_div_julie : sean_sum / julie_sum = 2 := by
  sorry

end sean_div_julie_l198_198382


namespace value_of_a_l198_198077

theorem value_of_a (a b k : ℝ) (h1 : a = k / b^2) (h2 : a = 40) (h3 : b = 12) (h4 : b = 24) : a = 10 := 
by
  sorry

end value_of_a_l198_198077


namespace remainder_123456789012_mod_252_l198_198496

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l198_198496


namespace yura_finish_date_l198_198703

theorem yura_finish_date :
  (∃ z : ℕ, 3 * z = 45 ∧ (∑ i in list.range (z + 2) \ 0, (z + 1 - i)) = 91 ∧ nat.find (λ n, (∑ i in list.range (n+1), (z + 1 - i)) >= 91) + 6 = 12) :=
by
  sorry

end yura_finish_date_l198_198703


namespace p_sufficient_not_necessary_for_q_l198_198126

-- Definitions based on conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬ (∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l198_198126


namespace find_k_b_l198_198010

noncomputable def symmetric_line_circle_intersection : Prop :=
  ∃ (k b : ℝ), 
    (∀ (x y : ℝ),  (y = k * x) ∧ ((x-1)^2 + y^2 = 1)) ∧ 
    (∀ (x y : ℝ), (x - y + b = 0)) →
    (k = -1 ∧ b = -1)

theorem find_k_b :
  symmetric_line_circle_intersection :=
  by
    -- omitted proof
    sorry

end find_k_b_l198_198010


namespace fraction_to_decimal_l198_198311

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198311


namespace total_valid_colorings_l198_198111

-- Define a function for the valid colorings
def valid_colorings (colors : Fin 9 → Fin 3) (edges : Finset (Fin 9 × Fin 9)) : Prop :=
  ∀ ⦃i j⦄, (i, j) ∈ edges → colors i ≠ colors j

-- Consider vertices of the three connected triangles
def vertices : Fin 9 := sorry -- Vertex positions in the nine-dot figure

-- Define edges of the three connected triangles
def edges : Finset (Fin 9 × Fin 9) := sorry -- Edges among the nine dots representing the figure

-- The sets of vertices for the three triangles
def first_triangle : Finset (Fin 9) := sorry
def second_triangle : Finset (Fin 9) := sorry
def third_triangle : Finset (Fin 9) := sorry

-- The total number of valid colorings
theorem total_valid_colorings (colors : Fin 9 → Fin 3) (h_valid : valid_colorings colors edges) : 
  ∃ n, n = 12 :=
by
  -- Here you would provide the proof steps, but we're skipping it with sorry
  sorry

end total_valid_colorings_l198_198111


namespace fraction_of_income_from_tips_l198_198968

variable (S T I : ℝ)

-- Conditions
def tips_are_fraction_of_salary : Prop := T = (3/4) * S
def total_income_is_sum_of_salary_and_tips : Prop := I = S + T

-- Statement to prove
theorem fraction_of_income_from_tips (h1 : tips_are_fraction_of_salary S T) (h2 : total_income_is_sum_of_salary_and_tips S T I) :
  T / I = 3 / 7 := by
  sorry

end fraction_of_income_from_tips_l198_198968


namespace composite_for_positive_integers_l198_198989

def is_composite (n : ℤ) : Prop :=
  ∃ a b : ℤ, 1 < a ∧ 1 < b ∧ n = a * b

theorem composite_for_positive_integers (n : ℕ) (h_pos : 1 < n) :
  is_composite (3^(2*n+1) - 2^(2*n+1) - 6*n) := 
sorry

end composite_for_positive_integers_l198_198989


namespace mountain_hill_school_absent_percentage_l198_198630

theorem mountain_hill_school_absent_percentage :
  let total_students := 120
  let boys := 70
  let girls := 50
  let absent_boys := (1 / 7) * boys
  let absent_girls := (1 / 5) * girls
  let absent_students := absent_boys + absent_girls
  let absent_percentage := (absent_students / total_students) * 100
  absent_percentage = 16.67 := sorry

end mountain_hill_school_absent_percentage_l198_198630


namespace inequality_proof_l198_198518

theorem inequality_proof (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : m + n = 1) :
  (m + 1/m) * (n + 1/n) ≥ 25/4 :=
by
  sorry

end inequality_proof_l198_198518


namespace bottle_cap_count_l198_198457

theorem bottle_cap_count (price_per_cap total_cost : ℕ) (h_price : price_per_cap = 2) (h_total : total_cost = 12) : total_cost / price_per_cap = 6 :=
by
  sorry

end bottle_cap_count_l198_198457


namespace contrapositive_example_l198_198218

theorem contrapositive_example (x : ℝ) (h : -2 < x ∧ x < 2) : x^2 < 4 :=
sorry

end contrapositive_example_l198_198218


namespace solve_equation_l198_198319

noncomputable def cube_root (x : ℝ) := x^(1 / 3)

theorem solve_equation (x : ℝ) :
  cube_root x = 15 / (8 - cube_root x) →
  x = 27 ∨ x = 125 :=
by
  sorry

end solve_equation_l198_198319


namespace rachel_should_budget_940_l198_198569

-- Define the prices for Sara's shoes and dress
def sara_shoes : ℝ := 50
def sara_dress : ℝ := 200

-- Define the prices for Tina's shoes and dress
def tina_shoes : ℝ := 70
def tina_dress : ℝ := 150

-- Define the total spending for Sara and Tina, and Rachel's budget
def rachel_budget (sara_shoes sara_dress tina_shoes tina_dress : ℝ) : ℝ := 
  2 * (sara_shoes + sara_dress + tina_shoes + tina_dress)

theorem rachel_should_budget_940 : 
  rachel_budget sara_shoes sara_dress tina_shoes tina_dress = 940 := 
by
  -- skip the proof
  sorry 

end rachel_should_budget_940_l198_198569


namespace verify_magic_square_l198_198318

-- Define the grid as a 3x3 matrix
def magic_square := Matrix (Fin 3) (Fin 3) ℕ

-- Conditions for the magic square
def is_magic_square (m : magic_square) : Prop :=
  (∀ i : Fin 3, (m i 0) + (m i 1) + (m i 2) = 15) ∧
  (∀ j : Fin 3, (m 0 j) + (m 1 j) + (m 2 j) = 15) ∧
  ((m 0 0) + (m 1 1) + (m 2 2) = 15) ∧
  ((m 0 2) + (m 1 1) + (m 2 0) = 15)

-- Given specific filled numbers in the grid
def given_filled_values (m : magic_square) : Prop :=
  (m 0 1 = 5) ∧
  (m 1 0 = 2) ∧
  (m 2 2 = 8)

-- The complete grid based on the solution
def completed_magic_square : magic_square :=
  ![![4, 9, 2], ![3, 5, 7], ![8, 1, 6]]

-- The main theorem to prove
theorem verify_magic_square : 
  is_magic_square completed_magic_square ∧ 
  given_filled_values completed_magic_square := 
by 
  sorry

end verify_magic_square_l198_198318


namespace least_pos_int_div_by_four_distinct_primes_l198_198061

/-- 
The least positive integer that is divisible by four distinct primes is 210.
-/
theorem least_pos_int_div_by_four_distinct_primes : 
  ∃ (n : ℕ), (∀ (p : ℕ), Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) ∧ 
             (∀ (m : ℕ), (∀ (p : ℕ), Prime p → p ∣ m → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → n ≤ m) ∧ 
             n = 210 := 
sorry

end least_pos_int_div_by_four_distinct_primes_l198_198061


namespace fraction_to_decimal_l198_198308

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198308


namespace reciprocal_neg_one_div_2022_l198_198231

theorem reciprocal_neg_one_div_2022 : (1 / (-1 / 2022)) = -2022 :=
by sorry

end reciprocal_neg_one_div_2022_l198_198231


namespace minimum_students_per_bench_l198_198582

theorem minimum_students_per_bench (M : ℕ) (B : ℕ) (F : ℕ) (H1 : F = 4 * M) (H2 : M = 29) (H3 : B = 29) :
  ⌈(M + F) / B⌉ = 5 :=
by
  sorry

end minimum_students_per_bench_l198_198582


namespace quadratic_real_roots_l198_198509

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) →
  (m ≤ 4 ∧ m ≠ 3) :=
sorry

end quadratic_real_roots_l198_198509


namespace distance_correct_l198_198964

-- Define geometry entities and properties
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Sphere where
  center : Point
  radius : ℝ

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define conditions
def sphere_center : Point := { x := 0, y := 0, z := 0 }
def sphere : Sphere := { center := sphere_center, radius := 5 }
def triangle : Triangle := { a := 13, b := 13, c := 10 }

-- Define the distance calculation
noncomputable def distance_from_sphere_center_to_plane (O : Point) (T : Triangle) : ℝ :=
  let h := 12  -- height calculation based on given triangle sides
  let A := 60  -- area of the triangle
  let s := 18  -- semiperimeter
  let r := 10 / 3  -- inradius calculation
  let x := 5 * (Real.sqrt 5) / 3  -- final distance calculation
  x

-- Prove the obtained distance matches expected value
theorem distance_correct :
  distance_from_sphere_center_to_plane sphere_center triangle = 5 * (Real.sqrt 5) / 3 :=
by
  sorry

end distance_correct_l198_198964


namespace no_three_digit_whole_number_solves_log_eq_l198_198542

noncomputable def log_function (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem no_three_digit_whole_number_solves_log_eq :
  ¬ ∃ n : ℤ, (100 ≤ n ∧ n < 1000) ∧ log_function (3 * n) 10 + log_function (7 * n) 10 = 1 :=
by
  sorry

end no_three_digit_whole_number_solves_log_eq_l198_198542


namespace isosceles_triangle_perimeter_l198_198994

theorem isosceles_triangle_perimeter (x y : ℝ) (h : |x - 4| + (y - 8)^2 = 0) :
  4 + 8 + 8 = 20 :=
by
  sorry

end isosceles_triangle_perimeter_l198_198994


namespace fraction_of_180_l198_198059

theorem fraction_of_180 : (1 / 2) * (1 / 3) * (1 / 6) * 180 = 5 := by
  sorry

end fraction_of_180_l198_198059


namespace remainder_when_divided_l198_198490

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l198_198490


namespace arithmetic_expression_value_l198_198942

theorem arithmetic_expression_value : 4 * (8 - 3 + 2) - 7 = 21 := by
  sorry

end arithmetic_expression_value_l198_198942


namespace sum_of_first_11_terms_of_arithmetic_seq_l198_198018

noncomputable def arithmetic_sequence_SUM (a d : ℚ) : ℚ :=  
  11 / 2 * (2 * a + 10 * d)

theorem sum_of_first_11_terms_of_arithmetic_seq
  (a d : ℚ)
  (h : a + 2 * d + a + 6 * d = 16) :
  arithmetic_sequence_SUM a d = 88 := 
  sorry

end sum_of_first_11_terms_of_arithmetic_seq_l198_198018


namespace image_digit_sum_l198_198378

theorem image_digit_sum 
  (cat chicken crab bear goat: ℕ)
  (h1 : 5 * crab = 10)
  (h2 : 4 * crab + goat = 11)
  (h3 : 2 * goat + crab + 2 * bear = 16)
  (h4 : cat + bear + 2 * goat + crab = 13)
  (h5 : 2 * crab + 2 * chicken + goat = 17) :
  cat = 1 ∧ chicken = 5 ∧ crab = 2 ∧ bear = 4 ∧ goat = 3 := by
  sorry

end image_digit_sum_l198_198378


namespace clothing_price_l198_198180

theorem clothing_price
  (total_spent : ℕ)
  (num_pieces : ℕ)
  (price_piece_1 : ℕ)
  (price_piece_2 : ℕ)
  (num_remaining_pieces : ℕ)
  (total_remaining_pieces_price : ℕ)
  (price_remaining_piece : ℕ) :
  total_spent = 610 →
  num_pieces = 7 →
  price_piece_1 = 49 →
  price_piece_2 = 81 →
  num_remaining_pieces = 5 →
  total_spent = price_piece_1 + price_piece_2 + total_remaining_pieces_price →
  total_remaining_pieces_price = price_remaining_piece * num_remaining_pieces →
  price_remaining_piece = 96 :=
by
  intros h_total_spent h_num_pieces h_price_piece_1 h_price_piece_2 h_num_remaining_pieces h_total_remaining_price h_price_remaining_piece
  sorry

end clothing_price_l198_198180


namespace integer_roots_of_quadratic_l198_198818

theorem integer_roots_of_quadratic (b : ℤ) :
  (∃ x : ℤ, x^2 + 4 * x + b = 0) ↔ b = -12 ∨ b = -5 ∨ b = 3 ∨ b = 4 :=
sorry

end integer_roots_of_quadratic_l198_198818


namespace quadratic_two_distinct_roots_l198_198928

theorem quadratic_two_distinct_roots :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 2 * x1^2 - 3 = 0 ∧ 2 * x2^2 - 3 = 0) :=
by
  sorry

end quadratic_two_distinct_roots_l198_198928


namespace new_person_weight_l198_198076

theorem new_person_weight (W : ℝ) (N : ℝ) (old_weight : ℝ) (average_increase : ℝ) (num_people : ℕ)
  (h1 : num_people = 8)
  (h2 : old_weight = 45)
  (h3 : average_increase = 6)
  (h4 : (W - old_weight + N) / num_people = W / num_people + average_increase) :
  N = 93 :=
by
  sorry

end new_person_weight_l198_198076


namespace quadratic_integers_pairs_l198_198348

theorem quadratic_integers_pairs (m n : ℕ) :
  (0 < m ∧ m < 9) ∧ (0 < n ∧ n < 9) ∧ (m^2 > 9 * n) ↔ ((m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2)) :=
by {
  -- Insert proof here
  sorry
}

end quadratic_integers_pairs_l198_198348


namespace sum_of_arithmetic_sequence_l198_198873

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a7 : a 7 = 7) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
  sorry

end sum_of_arithmetic_sequence_l198_198873


namespace length_of_interval_l198_198925

theorem length_of_interval (a b : ℝ) (h : 10 = (b - a) / 2) : b - a = 20 :=
by 
  sorry

end length_of_interval_l198_198925


namespace total_gain_percentage_combined_l198_198614

theorem total_gain_percentage_combined :
  let CP1 := 20
  let CP2 := 35
  let CP3 := 50
  let SP1 := 25
  let SP2 := 44
  let SP3 := 65
  let totalCP := CP1 + CP2 + CP3
  let totalSP := SP1 + SP2 + SP3
  let totalGain := totalSP - totalCP
  let gainPercentage := (totalGain / totalCP) * 100
  gainPercentage = 27.62 :=
by sorry

end total_gain_percentage_combined_l198_198614


namespace g_of_f_three_l198_198892

def f (x : ℝ) : ℝ := x^3 - 2
def g (x : ℝ) : ℝ := 3 * x^2 + x + 2

theorem g_of_f_three : g (f 3) = 1902 :=
by
  sorry

end g_of_f_three_l198_198892


namespace find_n_infinitely_many_squares_find_n_no_squares_l198_198720

def is_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P (n k l m : ℕ) : ℕ := n^k + n^l + n^m

theorem find_n_infinitely_many_squares :
  ∃ k, ∃ l, ∃ m, is_square (P 7 k l m) :=
by
  sorry

theorem find_n_no_squares :
  ∀ (k l m : ℕ) n, n ∈ [5, 6] → ¬is_square (P n k l m) :=
by
  sorry

end find_n_infinitely_many_squares_find_n_no_squares_l198_198720


namespace right_triangle_segment_ratio_l198_198862

-- Definitions of the triangle sides and hypotenuse
def right_triangle (AB BC : ℝ) : Prop :=
  AB/BC = 4/3

def hypotenuse (AB BC AC : ℝ) : Prop :=
  AC^2 = AB^2 + BC^2

def perpendicular_segment_ratio (AD CD : ℝ) : Prop :=
  AD / CD = 9/16

-- Final statement of the problem
theorem right_triangle_segment_ratio
  (AB BC AC AD CD : ℝ)
  (h1 : right_triangle AB BC)
  (h2 : hypotenuse AB BC AC)
  (h3 : perpendicular_segment_ratio AD CD) :
  CD / AD = 16/9 := sorry

end right_triangle_segment_ratio_l198_198862


namespace value_of_expression_l198_198858

theorem value_of_expression (x y : ℝ) (h : x + 2 * y = 30) : (x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3) = 16 :=
by
  sorry

end value_of_expression_l198_198858


namespace departure_sequences_l198_198540

-- Definitions for the conditions
def train_set : Type := {x // x ∈ {A, B, C, D, E, F, G, H}}

-- Prove the total number of different departure sequences for 8 trains is 720
theorem departure_sequences (A B C D E F G H : train_set) :
  (∀ A ∉ B) → (first_departs A) → (last_departs B) → 
  (count_departure_sequences {A, B, C, D, E, F, G, H} 4 4) = 720 :=
by
  sorry

end departure_sequences_l198_198540


namespace arccos_sin_3_l198_198973

theorem arccos_sin_3 : Real.arccos (Real.sin 3) = (Real.pi / 2) + 3 := 
by
  sorry

end arccos_sin_3_l198_198973


namespace ratio_equality_l198_198639

def op_def (a b : ℕ) : ℕ := a * b + b^2
def ot_def (a b : ℕ) : ℕ := a - b + a * b^2

theorem ratio_equality : (op_def 8 3 : ℚ) / (ot_def 8 3 : ℚ) = (33 : ℚ) / 77 := by
  sorry

end ratio_equality_l198_198639


namespace sum_gcd_lcm_l198_198604

theorem sum_gcd_lcm (a b : ℕ) (h_a : a = 75) (h_b : b = 4500) :
  Nat.gcd a b + Nat.lcm a b = 4575 := by
  sorry

end sum_gcd_lcm_l198_198604


namespace sum_of_squares_of_roots_l198_198808

theorem sum_of_squares_of_roots 
  (r s t : ℝ) 
  (hr : y^3 - 8 * y^2 + 9 * y - 2 = 0) 
  (hs : y ≥ 0) 
  (ht : y ≥ 0):
  r^2 + s^2 + t^2 = 46 :=
sorry

end sum_of_squares_of_roots_l198_198808


namespace John_meeting_percentage_l198_198718

def hours_to_minutes (h : ℕ) : ℕ := 60 * h

def first_meeting_duration : ℕ := 30
def second_meeting_duration : ℕ := 60
def third_meeting_duration : ℕ := 2 * first_meeting_duration
def total_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

def total_workday_duration : ℕ := hours_to_minutes 12

def percentage_of_meetings (total_meeting_time total_workday_time : ℕ) : ℕ := 
  (total_meeting_time * 100) / total_workday_time

theorem John_meeting_percentage : 
  percentage_of_meetings total_meeting_duration total_workday_duration = 21 :=
by
  sorry

end John_meeting_percentage_l198_198718


namespace quadratic_eq_two_distinct_real_roots_l198_198933

theorem quadratic_eq_two_distinct_real_roots :
    ∃ x y : ℝ, x ≠ y ∧ (x^2 + x - 1 = 0) ∧ (y^2 + y - 1 = 0) :=
by
    sorry

end quadratic_eq_two_distinct_real_roots_l198_198933


namespace older_brother_allowance_l198_198538

theorem older_brother_allowance 
  (sum_allowance : ℕ)
  (difference : ℕ)
  (total_sum : sum_allowance = 12000)
  (additional_amount : difference = 1000) :
  ∃ (older_brother_allowance younger_brother_allowance : ℕ), 
    older_brother_allowance = younger_brother_allowance + difference ∧
    younger_brother_allowance + older_brother_allowance = sum_allowance ∧
    older_brother_allowance = 6500 :=
by {
  sorry
}

end older_brother_allowance_l198_198538


namespace div_condition_l198_198120

theorem div_condition (N : ℤ) : (∃ k : ℤ, N^2 - 71 = k * (7 * N + 55)) ↔ (N = 57 ∨ N = -8) := 
by
  sorry

end div_condition_l198_198120


namespace minimize_f_at_a_l198_198656

def f (x a : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

theorem minimize_f_at_a (a : ℝ) (h : a = 82 / 43) :
  ∃ x, ∀ y, f x a ≤ f y a :=
sorry

end minimize_f_at_a_l198_198656


namespace remainder_div_252_l198_198467

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l198_198467


namespace artist_paintings_in_four_weeks_l198_198095

theorem artist_paintings_in_four_weeks (hours_per_week : ℕ) (hours_per_painting : ℕ) (weeks : ℕ) (total_paintings : ℕ) :
  hours_per_week = 30 → hours_per_painting = 3 → weeks = 4 → total_paintings = ((hours_per_week / hours_per_painting) * weeks) → total_paintings = 40 :=
by
  intros h_week h_painting h_weeks h_total
  rw [h_week, h_painting, h_weeks]
  norm_num
  exact h_total

end artist_paintings_in_four_weeks_l198_198095


namespace bottles_in_one_bag_l198_198080

theorem bottles_in_one_bag (total_bottles : ℕ) (cartons bags_per_carton : ℕ)
  (h1 : total_bottles = 180)
  (h2 : cartons = 3)
  (h3 : bags_per_carton = 4) :
  total_bottles / cartons / bags_per_carton = 15 :=
by sorry

end bottles_in_one_bag_l198_198080


namespace quadratic_real_roots_range_l198_198502

theorem quadratic_real_roots_range (m : ℝ) :
  ∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0 ↔ m ≤ 4 ∧ m ≠ 3 := 
by
  sorry

end quadratic_real_roots_range_l198_198502


namespace find_number_l198_198453

theorem find_number (x : ℝ) (h: 9999 * x = 4690910862): x = 469.1 :=
by
  sorry

end find_number_l198_198453


namespace sum_of_digits_of_9ab_l198_198712

noncomputable def a : ℕ := 10^2023 - 1
noncomputable def b : ℕ := 2*(10^2023 - 1) / 3

def digitSum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_of_9ab :
  digitSum (9 * a * b) = 20235 :=
by
  sorry

end sum_of_digits_of_9ab_l198_198712


namespace ingrid_tax_rate_proof_l198_198889

namespace TaxProblem

-- Define the given conditions
def john_income : ℝ := 56000
def ingrid_income : ℝ := 72000
def combined_income := john_income + ingrid_income

def john_tax_rate : ℝ := 0.30
def combined_tax_rate : ℝ := 0.35625

-- Calculate John's tax
def john_tax := john_tax_rate * john_income

-- Calculate total tax paid
def total_tax_paid := combined_tax_rate * combined_income

-- Calculate Ingrid's tax
def ingrid_tax := total_tax_paid - john_tax

-- Prove Ingrid's tax rate
theorem ingrid_tax_rate_proof (r : ℝ) :
  (ingrid_tax / ingrid_income) * 100 = 40 :=
  by sorry

end TaxProblem

end ingrid_tax_rate_proof_l198_198889


namespace clock_hands_overlap_24_hours_l198_198856

theorem clock_hands_overlap_24_hours : 
  (∀ t : ℕ, t < 12 →  ∃ n : ℕ, (n = 11 ∧ (∃ h m : ℕ, h * 60 + m = t * 60 + m))) →
  (∃ k : ℕ, k = 22) :=
by
  sorry

end clock_hands_overlap_24_hours_l198_198856


namespace rhombus_area_l198_198745

-- Define d1 and d2 as the lengths of the diagonals
def d1 : ℝ := 15
def d2 : ℝ := 17

-- The theorem to prove the area of the rhombus
theorem rhombus_area : (d1 * d2) / 2 = 127.5 := by
  sorry

end rhombus_area_l198_198745


namespace books_on_shelf_l198_198757

-- Step definitions based on the conditions
def initial_books := 38
def marta_books_removed := 10
def tom_books_removed := 5
def tom_books_added := 12

-- Final number of books on the shelf
def final_books : ℕ := initial_books - marta_books_removed - tom_books_removed + tom_books_added

-- Theorem statement to prove the final number of books
theorem books_on_shelf : final_books = 35 :=
by 
  -- Proof for the statement goes here
  sorry

end books_on_shelf_l198_198757


namespace linear_combination_harmonic_l198_198038

-- Define the harmonic property for a function
def is_harmonic (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ x y, f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4

-- The main statement to be proven in Lean
theorem linear_combination_harmonic
  (f g : ℤ × ℤ → ℝ) (a b : ℝ) (hf : is_harmonic f) (hg : is_harmonic g) :
  is_harmonic (fun p => a * f p + b * g p) :=
by
  sorry

end linear_combination_harmonic_l198_198038


namespace remainder_123456789012_mod_252_l198_198497

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l198_198497


namespace yura_finishes_on_correct_date_l198_198702

-- Let there be 91 problems in the textbook.
-- Let Yura start solving problems on September 6th.
def total_problems : Nat := 91
def start_date : Nat := 6

-- Each morning, starting from September 7, he solves one problem less than the previous morning.
def problems_solved (n : Nat) : Nat := if n = 0 then 0 else problems_solved (n - 1) - 1

-- On the evening of September 8, there are 46 problems left to solve.
def remaining_problems_sept_8 : Nat := 46

-- The question is to find on which date Yura will finish solving the textbook
def finish_date : Nat := 12

theorem yura_finishes_on_correct_date : ∃ z : Nat, (problems_solved z * 3 = total_problems - remaining_problems_sept_8) ∧ (z = 15) ∧ finish_date = 12 := by sorry

end yura_finishes_on_correct_date_l198_198702


namespace inverse_contrapositive_l198_198924

theorem inverse_contrapositive (a b c : ℝ) (h : a > b → a + c > b + c) :
  a + c ≤ b + c → a ≤ b :=
sorry

end inverse_contrapositive_l198_198924


namespace paul_books_sold_l198_198375

theorem paul_books_sold:
  ∀ (initial_books friend_books sold_per_day days final_books sold_books: ℝ),
    initial_books = 284.5 →
    friend_books = 63.7 →
    sold_per_day = 16.25 →
    days = 8 →
    final_books = 112.3 →
    sold_books = initial_books - friend_books - final_books →
    sold_books = 108.5 :=
by intros initial_books friend_books sold_per_day days final_books sold_books
   sorry

end paul_books_sold_l198_198375


namespace cylinder_volume_l198_198755

theorem cylinder_volume (r h : ℝ) (h_radius : r = 1) (h_height : h = 2) : (π * r^2 * h) = 2 * π :=
by
  sorry

end cylinder_volume_l198_198755


namespace calculate_difference_l198_198633

theorem calculate_difference : (-3) - (-5) = 2 := by
  sorry

end calculate_difference_l198_198633


namespace sphere_radius_eq_cylinder_radius_l198_198409

theorem sphere_radius_eq_cylinder_radius
  (r h d : ℝ) (h_eq_d : h = 16) (d_eq_h : d = 16)
  (sphere_surface_area_eq_cylinder : 4 * Real.pi * r^2 = 2 * Real.pi * (d / 2) * h) : 
  r = 8 :=
by
  sorry

end sphere_radius_eq_cylinder_radius_l198_198409


namespace product_of_constants_l198_198555

theorem product_of_constants :
  ∃ M₁ M₂ : ℝ, 
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (45 * x - 82) / (x^2 - 5 * x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) ∧ 
    M₁ * M₂ = -424 :=
by
  sorry

end product_of_constants_l198_198555


namespace find_b_value_l198_198896

theorem find_b_value (f : ℝ → ℝ) (f_inv : ℝ → ℝ) (b : ℝ) :
  (∀ x, f x = 1 / (3 * x + b)) →
  (∀ x, f_inv x = (2 - 3 * x) / (3 * x)) →
  b = -3 :=
by
  intros h1 h2
  sorry

end find_b_value_l198_198896


namespace sand_bucket_capacity_l198_198766

theorem sand_bucket_capacity
  (sandbox_depth : ℝ)
  (sandbox_width : ℝ)
  (sandbox_length : ℝ)
  (sand_weight_per_cubic_foot : ℝ)
  (water_per_4_trips : ℝ)
  (water_bottle_ounces : ℝ)
  (water_bottle_cost : ℝ)
  (tony_total_money : ℝ)
  (tony_change : ℝ)
  (tony's_bucket_capacity : ℝ) :
  sandbox_depth = 2 →
  sandbox_width = 4 →
  sandbox_length = 5 →
  sand_weight_per_cubic_foot = 3 →
  water_per_4_trips = 3 →
  water_bottle_ounces = 15 →
  water_bottle_cost = 2 →
  tony_total_money = 10 →
  tony_change = 4 →
  tony's_bucket_capacity = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry -- skipping the proof as per instructions

end sand_bucket_capacity_l198_198766


namespace number_of_routes_600_l198_198983

-- Define the problem conditions
def number_of_routes (total_cities : Nat) (pick_cities : Nat) (selected_cities : List Nat) : Nat := sorry

-- The number of ways to pick and order 3 cities from remaining 5
def num_ways_pick_three (total_cities : Nat) (pick_cities : Nat) : Nat :=
  Nat.factorial total_cities / Nat.factorial (total_cities - pick_cities)

-- The number of ways to choose positions for M and N
def num_ways_positions (total_positions : Nat) (pick_positions : Nat) : Nat :=
  Nat.choose total_positions pick_positions

-- The main theorem to prove
theorem number_of_routes_600 :
  number_of_routes 7 5 [M, N] = num_ways_pick_three 5 3 * num_ways_positions 4 2 :=
  by sorry

end number_of_routes_600_l198_198983


namespace parametric_to_standard_l198_198978

theorem parametric_to_standard (t a b x y : ℝ)
(h1 : x = (a / 2) * (t + 1 / t))
(h2 : y = (b / 2) * (t - 1 / t)) :
  (x^2 / a^2) - (y^2 / b^2) = 1 :=
by
  sorry

end parametric_to_standard_l198_198978


namespace triangle_area_division_l198_198250

theorem triangle_area_division (T T_1 T_2 T_3 : ℝ) 
  (hT1_pos : 0 < T_1) (hT2_pos : 0 < T_2) (hT3_pos : 0 < T_3) (hT : T = T_1 + T_2 + T_3) :
  T = (Real.sqrt T_1 + Real.sqrt T_2 + Real.sqrt T_3) ^ 2 :=
sorry

end triangle_area_division_l198_198250


namespace no_solution_exists_only_solution_is_1963_l198_198367

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

end no_solution_exists_only_solution_is_1963_l198_198367


namespace problem1_expr_eval_l198_198099

theorem problem1_expr_eval : 
  (1:ℤ) - (1:ℤ)^(2022:ℕ) - (3 * (2/3:ℚ)^2 - (8/3:ℚ) / ((-2)^3:ℤ)) = -8/3 :=
by
  sorry

end problem1_expr_eval_l198_198099


namespace min_value_reciprocal_sum_l198_198841

theorem min_value_reciprocal_sum (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_sum : x + y = 1) : 
  ∃ z, z = 4 ∧ (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 1 -> z ≤ (1/x + 1/y)) :=
sorry

end min_value_reciprocal_sum_l198_198841


namespace num_groups_of_consecutive_natural_numbers_l198_198524

theorem num_groups_of_consecutive_natural_numbers (n : ℕ) (h : 3 * n + 3 < 19) : n < 6 := 
  sorry

end num_groups_of_consecutive_natural_numbers_l198_198524


namespace number_of_distinct_values_l198_198959

theorem number_of_distinct_values (n : ℕ) (mode_count : ℕ) (second_count : ℕ) (total_count : ℕ) 
    (h1 : n = 3000) (h2 : mode_count = 15) (h3 : second_count = 14) : 
    (n - mode_count - second_count) / 13 + 2 ≥ 232 :=
by 
  sorry

end number_of_distinct_values_l198_198959


namespace fraction_numerator_exceeds_denominator_l198_198049

theorem fraction_numerator_exceeds_denominator (x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 3) :
  4 * x + 5 > 10 - 3 * x ↔ (5 / 7) < x ∧ x ≤ 3 :=
by 
  sorry

end fraction_numerator_exceeds_denominator_l198_198049


namespace exist_ints_a_b_for_any_n_l198_198737

theorem exist_ints_a_b_for_any_n (n : ℤ) : ∃ a b : ℤ, n = Int.floor (a * Real.sqrt 2) + Int.floor (b * Real.sqrt 3) := by
  sorry

end exist_ints_a_b_for_any_n_l198_198737


namespace inequality_sufficient_condition_l198_198054

theorem inequality_sufficient_condition (x : ℝ) (h : 1 < x ∧ x < 2) : 
  (x+1)/(x-1) > 2 :=
by
  sorry

end inequality_sufficient_condition_l198_198054


namespace robert_time_to_complete_l198_198085

noncomputable def time_to_complete_semicircle_path (length_mile : ℝ) (width_feet : ℝ) (speed_mph : ℝ) (mile_to_feet : ℝ) : ℝ :=
  let diameter_mile := width_feet / mile_to_feet
  let radius_mile := diameter_mile / 2
  let circumference_mile := 2 * Real.pi * radius_mile
  let semicircle_length_mile := circumference_mile / 2
  semicircle_length_mile / speed_mph

theorem robert_time_to_complete :
  time_to_complete_semicircle_path 1 40 5 5280 = Real.pi / 10 :=
by
  sorry

end robert_time_to_complete_l198_198085


namespace problem_solution_l198_198661

variable {f : ℕ → ℕ}
variable (h_mul : ∀ a b : ℕ, f (a + b) = f a * f b)
variable (h_one : f 1 = 2)

theorem problem_solution : 
  (f 2 / f 1) + (f 4 / f 3) + (f 6 / f 5) + (f 8 / f 7) + (f 10 / f 9) = 10 :=
by
  sorry

end problem_solution_l198_198661


namespace total_points_is_400_l198_198709

-- Define the conditions as definitions in Lean 4 
def pointsPerEnemy : ℕ := 15
def bonusPoints : ℕ := 50
def totalEnemies : ℕ := 25
def enemiesLeftUndestroyed : ℕ := 5
def bonusesEarned : ℕ := 2

-- Calculate the total number of enemies defeated
def enemiesDefeated : ℕ := totalEnemies - enemiesLeftUndestroyed

-- Calculate the points from defeating enemies
def pointsFromEnemies := enemiesDefeated * pointsPerEnemy

-- Calculate the total bonus points
def totalBonusPoints := bonusesEarned * bonusPoints

-- The total points earned is the sum of points from enemies and bonus points
def totalPointsEarned := pointsFromEnemies + totalBonusPoints

-- Prove that the total points earned is equal to 400
theorem total_points_is_400 : totalPointsEarned = 400 := by
    sorry

end total_points_is_400_l198_198709


namespace g_g_g_g_2_eq_1406_l198_198901

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 5 * x + 1

theorem g_g_g_g_2_eq_1406 : g (g (g (g 2))) = 1406 := by
  sorry

end g_g_g_g_2_eq_1406_l198_198901


namespace number_of_small_companies_l198_198866

theorem number_of_small_companies
  (large_companies : ℕ)
  (medium_companies : ℕ)
  (inspected_companies : ℕ)
  (inspected_medium_companies : ℕ)
  (total_inspected_companies : ℕ)
  (small_companies : ℕ)
  (inspection_fraction : ℕ → ℚ)
  (proportion : inspection_fraction 20 = 1 / 4)
  (H1 : large_companies = 4)
  (H2 : medium_companies = 20)
  (H3 : inspected_medium_companies = 5)
  (H4 : total_inspected_companies = 40)
  (H5 : inspected_companies = total_inspected_companies - large_companies - inspected_medium_companies)
  (H6 : small_companies = inspected_companies * 4)
  (correct_result : small_companies = 136) :
  small_companies = 136 :=
by sorry

end number_of_small_companies_l198_198866


namespace part1_part2_l198_198831

theorem part1 (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi) (h_trig : Real.sin α + Real.cos α = 1 / 5) :
  Real.sin α - Real.cos α = 7 / 5 := sorry

theorem part2 (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi) (h_trig : Real.sin α + Real.cos α = 1 / 5) :
  Real.sin (2 * α + Real.pi / 3) = -12 / 25 - 7 * Real.sqrt 3 / 50 := sorry

end part1_part2_l198_198831


namespace degree_to_radian_radian_to_degree_l198_198976

theorem degree_to_radian (d : ℝ) (rad : ℝ) (deg : ℝ) :
  (180 * rad = π) → (d = 210) → rad = (π / 180) → d * rad = 7 * π / 6 :=
by sorry 

theorem radian_to_degree (r : ℝ) (rad : ℝ) (deg : ℝ) :
  (180 * rad = π) → (r = -5 * π / 2) → deg = (180 / π) → r * deg = -450 :=
by sorry

end degree_to_radian_radian_to_degree_l198_198976


namespace cake_cubes_with_exactly_two_faces_iced_l198_198716

theorem cake_cubes_with_exactly_two_faces_iced :
  let cake : ℕ := 3 -- cake dimension
  let total_cubes : ℕ := cake ^ 3 -- number of smaller cubes (total 27)
  let cubes_with_two_faces_icing := 4
  (∀ cake icing (smaller_cubes : ℕ), icing ≠ 0 → smaller_cubes = cake ^ 3 → 
    let top_iced := cake - 2 -- cubes with icing on top only
    let front_iced := cake - 2 -- cubes with icing on front only
    let back_iced := cake - 2 -- cubes with icing on back only
    ((top_iced * 2) = cubes_with_two_faces_icing)) :=
  sorry

end cake_cubes_with_exactly_two_faces_iced_l198_198716


namespace fraction_to_decimal_l198_198299

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l198_198299


namespace minimum_voters_for_tall_win_l198_198547

-- Definitions based on the conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3
def majority_precinct_voters : ℕ := 2
def majority_precincts_per_district : ℕ := 5
def majority_districts : ℕ := 3
def tall_won : Prop := true

-- Problem statement
theorem minimum_voters_for_tall_win : 
  tall_won → (∃ n : ℕ, n = 3 * 5 * 2 ∧ n ≤ voters) :=
by
  sorry

end minimum_voters_for_tall_win_l198_198547


namespace max_value_sqrt_abcd_l198_198028

theorem max_value_sqrt_abcd (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (h_sum : a + b + c + d = 1) :
  Real.sqrt (abcd) ^ (1 / 4) + Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1 / 4) ≤ 1 := 
sorry

end max_value_sqrt_abcd_l198_198028


namespace minimum_rotation_angle_of_square_l198_198092

theorem minimum_rotation_angle_of_square : 
  ∀ (angle : ℝ), (∃ n : ℕ, angle = 360 / n) ∧ (n ≥ 1) ∧ (n ≤ 4) → angle = 90 :=
by
  sorry

end minimum_rotation_angle_of_square_l198_198092


namespace yura_finishes_problems_by_sept_12_l198_198697

def total_problems := 91
def initial_date := 6 -- September 6
def problems_left_date := 8 -- September 8
def remaining_problems := 46
def decreasing_rate := 1

def problems_solved (z : ℕ) (day : ℕ) : ℕ :=
if day = 6 then z + 1 else if day = 7 then z else if day = 8 then z - 1 else z - (day - 7)

theorem yura_finishes_problems_by_sept_12 :
  ∃ z : ℕ, (problems_solved z 6 + problems_solved z 7 + problems_solved z 8 = total_problems - remaining_problems) ∧
           (problems_solved z 6 + problems_solved z 7 + problems_solved z 8 + problems_solved z 9 + problems_solved z 10 + problems_solved z 11 + problems_solved z 12 = total_problems) :=
sorry

end yura_finishes_problems_by_sept_12_l198_198697


namespace cristian_cookie_problem_l198_198280

theorem cristian_cookie_problem (white_cookies_init black_cookies_init eaten_black_cookies eaten_white_cookies remaining_black_cookies remaining_white_cookies total_remaining_cookies : ℕ) 
  (h_initial_white : white_cookies_init = 80)
  (h_black_more : black_cookies_init = white_cookies_init + 50)
  (h_eats_half_black : eaten_black_cookies = black_cookies_init / 2)
  (h_eats_three_fourth_white : eaten_white_cookies = (3 / 4) * white_cookies_init)
  (h_remaining_black : remaining_black_cookies = black_cookies_init - eaten_black_cookies)
  (h_remaining_white : remaining_white_cookies = white_cookies_init - eaten_white_cookies)
  (h_total_remaining : total_remaining_cookies = remaining_black_cookies + remaining_white_cookies) :
  total_remaining_cookies = 85 :=
by
  sorry

end cristian_cookie_problem_l198_198280


namespace area_of_circle_l198_198222

theorem area_of_circle (r : ℝ) : 
  (S = π * r^2) :=
sorry

end area_of_circle_l198_198222


namespace C_share_correct_l198_198450

def investment_A := 27000
def investment_B := 72000
def investment_C := 81000
def total_profit := 80000

def gcd_investment : ℕ := Nat.gcd investment_A (Nat.gcd investment_B investment_C)
def ratio_A : ℕ := investment_A / gcd_investment
def ratio_B : ℕ := investment_B / gcd_investment
def ratio_C : ℕ := investment_C / gcd_investment
def total_parts : ℕ := ratio_A + ratio_B + ratio_C

def C_share : ℕ := (ratio_C / total_parts) * total_profit

theorem C_share_correct : C_share = 36000 := 
by sorry

end C_share_correct_l198_198450


namespace hybrids_with_full_headlights_l198_198165

theorem hybrids_with_full_headlights (total_cars hybrids_percentage one_headlight_percentage : ℝ) 
  (hc : total_cars = 600) (hp : hybrids_percentage = 0.60) (ho : one_headlight_percentage = 0.40) : 
  total_cars * hybrids_percentage - total_cars * hybrids_percentage * one_headlight_percentage = 216 := by
  sorry

end hybrids_with_full_headlights_l198_198165


namespace least_pos_int_div_by_four_distinct_primes_l198_198060

theorem least_pos_int_div_by_four_distinct_primes : 
  let p1 := 2
      p2 := 3
      p3 := 5
      p4 := 7
      n := p1 * p2 * p3 * p4
  in n = 210 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let n := p1 * p2 * p3 * p4
  show n = 210
  sorry

end least_pos_int_div_by_four_distinct_primes_l198_198060


namespace eggs_per_chicken_per_day_l198_198813

theorem eggs_per_chicken_per_day (E c d : ℕ) (hE : E = 36) (hc : c = 4) (hd : d = 3) :
  (E / d) / c = 3 := by
  sorry

end eggs_per_chicken_per_day_l198_198813


namespace p_implies_q_and_not_q_implies_p_l198_198135

variable {x : ℝ}

def p : Prop := 0 < x ∧ x < 2
def q : Prop := -1 < x ∧ x < 3

theorem p_implies_q_and_not_q_implies_p : (p → q) ∧ ¬ (q → p) := 
by
  sorry

end p_implies_q_and_not_q_implies_p_l198_198135


namespace fraction_to_decimal_l198_198305

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198305


namespace sum_difference_of_odd_and_even_integers_l198_198988

noncomputable def sum_of_first_n_odds (n : ℕ) : ℕ :=
  n * n

noncomputable def sum_of_first_n_evens (n : ℕ) : ℕ :=
  n * (n + 1)

theorem sum_difference_of_odd_and_even_integers :
  sum_of_first_n_evens 50 - sum_of_first_n_odds 50 = 50 := 
by
  sorry

end sum_difference_of_odd_and_even_integers_l198_198988


namespace probability_perfect_square_l198_198966

theorem probability_perfect_square (rolls : ℕ → ℕ) (cond : ∀ i, 1 ≤ rolls i ∧ rolls i ≤ 8) 
  (at_least_one_4_5_6 : ∃ i, rolls i = 4 ∨ rolls i = 5 ∨ rolls i = 6) : 
  (1 / 256 : ℚ) = 57 / 256 :=
by
  sorry

end probability_perfect_square_l198_198966


namespace contractor_net_amount_l198_198444

-- Definitions based on conditions
def total_days : ℕ := 30
def pay_per_day : ℝ := 25
def fine_per_absence_day : ℝ := 7.5
def days_absent : ℕ := 6

-- Calculate days worked
def days_worked : ℕ := total_days - days_absent

-- Calculate total earnings
def earnings : ℝ := days_worked * pay_per_day

-- Calculate total fine
def fine : ℝ := days_absent * fine_per_absence_day

-- Calculate net amount received by the contractor
def net_amount : ℝ := earnings - fine

-- Problem statement: Prove that the net amount is Rs. 555
theorem contractor_net_amount : net_amount = 555 := by
  sorry

end contractor_net_amount_l198_198444


namespace triangle_perimeter_l198_198017

noncomputable def smallest_perimeter (a b c : ℕ) : ℕ :=
  a + b + c

theorem triangle_perimeter (a b c : ℕ) (A B C : ℝ) (h1 : A = 2 * B) 
  (h2 : C > π / 2) (h3 : a^2 = b * (b + c)) (h4 : ∃ m n : ℕ, b = m^2 ∧ b + c = n^2 ∧ a = m * n) :
  smallest_perimeter 28 16 33 = 77 :=
by sorry

end triangle_perimeter_l198_198017


namespace find_center_and_tangent_slope_l198_198843

theorem find_center_and_tangent_slope :
  let C := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 6 * p.1 + 8 = 0 }
  let center := (3, 0)
  let k := - (Real.sqrt 2 / 4)
  (∃ c ∈ C, c = center) ∧
  (∃ q ∈ C, q.2 < 0 ∧ q.2 = k * q.1 ∧
             |3 * k| / Real.sqrt (k ^ 2 + 1) = 1) :=
by
  sorry

end find_center_and_tangent_slope_l198_198843


namespace triangle_side_length_median_l198_198353

theorem triangle_side_length_median 
  (D E F : Type) 
  (DE : D → E → ℝ) 
  (EF : E → F → ℝ) 
  (DM : D → ℝ)
  (d_e : D) (e_f : F) (m : D)
  (hDE : DE d_e e_f = 7) 
  (hEF : EF e_f e_f = 10)
  (hDM : DM m = 5) :
  ∃ (DF : D → F → ℝ), DF d_e e_f = real.sqrt 149 := 
begin  
  sorry
end

end triangle_side_length_median_l198_198353


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l198_198209

theorem solve_eq1 :
  ∀ x : ℝ, 6 * x - 7 = 4 * x - 5 ↔ x = 1 := by
  intro x
  sorry

theorem solve_eq2 :
  ∀ x : ℝ, 5 * (x + 8) - 5 = 6 * (2 * x - 7) ↔ x = 11 := by
  intro x
  sorry

theorem solve_eq3 :
  ∀ x : ℝ, x - (x - 1) / 2 = 2 - (x + 2) / 5 ↔ x = 11 / 7 := by
  intro x
  sorry

theorem solve_eq4 :
  ∀ x : ℝ, x^2 - 64 = 0 ↔ x = 8 ∨ x = -8 := by
  intro x
  sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l198_198209


namespace least_pawns_required_l198_198184

theorem least_pawns_required (n k : ℕ) (h1 : n > 0) (h2 : k > 0) (h3 : 2 * k > n) (h4 : 3 * k ≤ 2 * n) : 
  ∃ (m : ℕ), m = 4 * (n - k) :=
sorry

end least_pawns_required_l198_198184


namespace mixing_solutions_l198_198392

theorem mixing_solutions (Vx : ℝ) :
  (0.10 * Vx + 0.30 * 900 = 0.25 * (Vx + 900)) ↔ Vx = 300 := by
  sorry

end mixing_solutions_l198_198392


namespace cubic_sum_l198_198672

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 := by
  -- Using the provided conditions x + y = 5 and x^2 + y^2 = 13
  sorry

end cubic_sum_l198_198672


namespace select_student_based_on_variance_l198_198210

-- Define the scores for students A and B
def scoresA : List ℚ := [12.1, 12.1, 12.0, 11.9, 11.8, 12.1]
def scoresB : List ℚ := [12.2, 12.0, 11.8, 12.0, 12.3, 11.7]

-- Define the function to calculate the mean of a list of rational numbers
def mean (scores : List ℚ) : ℚ := (scores.foldr (· + ·) 0) / scores.length

-- Define the function to calculate the variance of a list of rational numbers
def variance (scores : List ℚ) : ℚ :=
  let m := mean scores
  (scores.foldr (λ x acc => acc + (x - m) ^ 2) 0) / scores.length

-- Prove that the variance of student A's scores is less than the variance of student B's scores
theorem select_student_based_on_variance :
  variance scoresA < variance scoresB := by
  sorry

end select_student_based_on_variance_l198_198210


namespace evaluate_f_neg_a_l198_198147

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x^2) - 3 * x) + 1

theorem evaluate_f_neg_a (a : ℝ) (h : f a = 1 / 3) : f (-a) = 5 / 3 :=
by sorry

end evaluate_f_neg_a_l198_198147


namespace find_numbers_l198_198432

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

noncomputable def number1 := 986
noncomputable def number2 := 689

theorem find_numbers :
  is_three_digit_number number1 ∧ is_three_digit_number number2 ∧
  hundreds_digit number1 = units_digit number2 ∧ hundreds_digit number2 = units_digit number1 ∧
  number1 - number2 = 297 ∧ (hundreds_digit number2 + tens_digit number2 + units_digit number2) = 23 :=
by
  sorry

end find_numbers_l198_198432


namespace cube_sum_l198_198678

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cube_sum_l198_198678


namespace remainder_123456789012_div_252_l198_198482

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l198_198482


namespace exists_such_h_l198_198283

noncomputable def exists_h (h : ℝ) : Prop :=
  ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋)

theorem exists_such_h : ∃ h : ℝ, exists_h h := 
  -- Let's construct the h as mentioned in the provided proof
  ⟨1969^2 / 1968, 
    by sorry⟩

end exists_such_h_l198_198283


namespace probability_interval_l198_198754

/-- 
The probability of event A occurring is 4/5, the probability of event B occurring is 3/4,
and the probability of event C occurring is 2/3. The smallest interval necessarily containing
the probability q that all three events occur is [0, 2/3].
-/
theorem probability_interval (P_A P_B P_C q : ℝ)
  (hA : P_A = 4 / 5) (hB : P_B = 3 / 4) (hC : P_C = 2 / 3)
  (h_q_le_A : q ≤ P_A) (h_q_le_B : q ≤ P_B) (h_q_le_C : q ≤ P_C) :
  0 ≤ q ∧ q ≤ 2 / 3 := by
  sorry

end probability_interval_l198_198754


namespace polynomial_degree_bounds_l198_198900

open nat

theorem polynomial_degree_bounds (p : ℕ) (hp : prime p) (f : polynomial ℤ) (hf_degree : f.degree = d)
  (hf_0 : f.eval 0 = 0) (hf_1 : f.eval 1 = 1) 
  (hf_mod : ∀ (n : ℕ), n > 0 → f.eval n % p = 0 ∨ f.eval n % p = 1) :
  d ≥ p - 1 :=
sorry

end polynomial_degree_bounds_l198_198900


namespace remainder_123456789012_div_252_l198_198479

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l198_198479


namespace no_natural_number_divides_Q_by_x_squared_minus_one_l198_198461

def Q (n : ℕ) (x : ℝ) : ℝ := 1 + 5*x^2 + x^4 - (n - 1) * x^(n - 1) + (n - 8) * x^n

theorem no_natural_number_divides_Q_by_x_squared_minus_one :
  ∀ (n : ℕ), n > 0 → ¬ (x^2 - 1 ∣ Q n x) :=
by
  intros n h
  sorry

end no_natural_number_divides_Q_by_x_squared_minus_one_l198_198461


namespace quadratic_complete_square_l198_198932

theorem quadratic_complete_square :
  ∃ a b c : ℝ, (∀ x : ℝ, 4 * x^2 - 40 * x + 100 = a * (x + b)^2 + c) ∧ a + b + c = -1 :=
sorry

end quadratic_complete_square_l198_198932


namespace area_of_square_field_l198_198040

theorem area_of_square_field (s : ℕ) (A : ℕ) (cost_per_meter : ℕ) 
  (total_cost : ℕ) (gate_width : ℕ) (num_gates : ℕ) 
  (h1 : cost_per_meter = 1)
  (h2 : total_cost = 666)
  (h3 : gate_width = 1)
  (h4 : num_gates = 2)
  (h5 : (4 * s - num_gates * gate_width) * cost_per_meter = total_cost) :
  A = s * s → A = 27889 :=
by
  sorry

end area_of_square_field_l198_198040


namespace bird_families_left_l198_198243

theorem bird_families_left (B_initial B_flew_away : ℕ) (h_initial : B_initial = 41) (h_flew_away : B_flew_away = 27) :
  B_initial - B_flew_away = 14 :=
by
  sorry

end bird_families_left_l198_198243


namespace min_value_abs_sum_exists_min_value_abs_sum_l198_198927

theorem min_value_abs_sum (x : ℝ) : |x - 1| + |x - 4| ≥ 3 :=
by sorry

theorem exists_min_value_abs_sum : ∃ x : ℝ, |x - 1| + |x - 4| = 3 :=
by sorry

end min_value_abs_sum_exists_min_value_abs_sum_l198_198927


namespace expr1_simplified_expr2_simplified_l198_198781

variable (a x : ℝ)

theorem expr1_simplified : (-a^3 + (-4 * a^2) * a) = -5 * a^3 := 
by
  sorry

theorem expr2_simplified : (-x^2 * (-x)^2 * (-x^2)^3 - 2 * x^10) = -x^10 := 
by
  sorry

end expr1_simplified_expr2_simplified_l198_198781


namespace part_a_part_b_l198_198719

variable {A : Type} [Ring A] (h : ∀ x : A, x + x^2 + x^3 = x^4 + x^5 + x^6)

-- Part (a)
theorem part_a (x : A) (n : Nat) (hn : n ≥ 2) (hx : x^n = 0) : x = 0 :=
sorry

-- Part (b)
theorem part_b (x : A) : x^4 = x :=
by
  have h : ∀ x : A, x + x^2 + x^3 = x^4 + x^5 + x^6 := h
  sorry

end part_a_part_b_l198_198719


namespace arithmetic_result_l198_198602

theorem arithmetic_result :
  (3 * 13) + (3 * 14) + (3 * 17) + 11 = 143 :=
by
  sorry

end arithmetic_result_l198_198602


namespace g_of_f_three_l198_198893

def f (x : ℝ) : ℝ := x^3 - 2
def g (x : ℝ) : ℝ := 3 * x^2 + x + 2

theorem g_of_f_three : g (f 3) = 1902 :=
by
  sorry

end g_of_f_three_l198_198893


namespace students_passed_both_tests_l198_198081

theorem students_passed_both_tests
    (total_students : ℕ)
    (passed_long_jump : ℕ)
    (passed_shot_put : ℕ)
    (failed_both : ℕ)
    (h_total : total_students = 50)
    (h_long_jump : passed_long_jump = 40)
    (h_shot_put : passed_shot_put = 31)
    (h_failed_both : failed_both = 4) : 
    (total_students - failed_both = passed_long_jump + passed_shot_put - 25) :=
by 
  sorry

end students_passed_both_tests_l198_198081


namespace quadratic_no_real_roots_l198_198138

open Real

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (hpq : p ≠ q)
  (hpositive_p : 0 < p)
  (hpositive_q : 0 < q)
  (hpositive_a : 0 < a)
  (hpositive_b : 0 < b)
  (hpositive_c : 0 < c)
  (h_geo_sequence : a^2 = p * q)
  (h_ari_sequence : b + c = p + q) :
  (a^2 - b * c) < 0 :=
by
  sorry

end quadratic_no_real_roots_l198_198138


namespace avg_remaining_two_l198_198919

theorem avg_remaining_two (avg5 avg3 : ℝ) (h1 : avg5 = 12) (h2 : avg3 = 4) : (5 * avg5 - 3 * avg3) / 2 = 24 :=
by sorry

end avg_remaining_two_l198_198919


namespace circumference_of_base_l198_198084

-- Definitions used for the problem
def radius : ℝ := 6
def sector_angle : ℝ := 300
def full_circle_angle : ℝ := 360

-- Ask for the circumference of the base of the cone formed by the sector
theorem circumference_of_base (r : ℝ) (theta_sector : ℝ) (theta_full : ℝ) :
  (theta_sector / theta_full) * (2 * π * r) = 10 * π :=
by
  sorry

end circumference_of_base_l198_198084


namespace length_of_string_C_l198_198048

theorem length_of_string_C (A B C : ℕ) (h1 : A = 6 * C) (h2 : A = 5 * B) (h3 : B = 12) : C = 10 :=
sorry

end length_of_string_C_l198_198048


namespace range_of_a_l198_198139

theorem range_of_a (x a : ℝ) (hp : x^2 + 2 * x - 3 > 0) (hq : x > a)
  (h_suff : x^2 + 2 * x - 3 > 0 → ¬ (x > a)):
  a ≥ 1 := 
by
  sorry

end range_of_a_l198_198139


namespace car_distribution_l198_198625

theorem car_distribution :
  ∀ (total_cars cars_first cars_second cars_left : ℕ),
    total_cars = 5650000 →
    cars_first = 1000000 →
    cars_second = cars_first + 500000 →
    cars_left = total_cars - (cars_first + cars_second + (cars_first + cars_second)) →
    ∃ (cars_fourth_fifth : ℕ), cars_fourth_fifth = cars_left / 2 ∧ cars_fourth_fifth = 325000 :=
begin
  intros total_cars cars_first cars_second cars_left H_total H_first H_second H_left,
  use (cars_left / 2),
  split,
  { refl, },
  { rw [H_total, H_first, H_second, H_left],
    norm_num, },
end

end car_distribution_l198_198625


namespace distribute_awards_l198_198937

theorem distribute_awards :
  (∑ k in finset.range 11, if k ≤ 3 then 1 else 0) = 84 :=
by
  sorry

end distribute_awards_l198_198937


namespace remainder_div_252_l198_198464

theorem remainder_div_252 :
  let N : ℕ := 123456789012 in
  let mod_4  := N % 4  in
  let mod_9  := N % 9  in
  let mod_7  := N % 7  in
  let x     := 144    in
  N % 252 = x :=
by
  let N := 123456789012
  let mod_4  := N % 4
  let mod_9  := N % 9
  let mod_7  := N % 7
  let x     := 144
  have h1 : mod_4 = 0 := by sorry
  have h2 : mod_9 = 3 := by sorry
  have h3 : mod_7 = 4 := by sorry
  have h4 : N % 252 = x := by sorry
  exact h4

end remainder_div_252_l198_198464


namespace four_digit_number_difference_l198_198834

theorem four_digit_number_difference
    (digits : List ℕ) (h_digits : digits = [2, 0, 1, 3, 1, 2, 2, 1, 0, 8, 4, 0])
    (max_val : ℕ) (h_max_val : max_val = 3840)
    (min_val : ℕ) (h_min_val : min_val = 1040) :
    max_val - min_val = 2800 :=
by
    sorry

end four_digit_number_difference_l198_198834


namespace average_speed_of_train_l198_198792

-- Define conditions
def traveled_distance1 : ℝ := 240
def traveled_distance2 : ℝ := 450
def time_period1 : ℝ := 3
def time_period2 : ℝ := 5

-- Define total distance and total time based on the conditions
def total_distance : ℝ := traveled_distance1 + traveled_distance2
def total_time : ℝ := time_period1 + time_period2

-- Prove that the average speed is 86.25 km/h
theorem average_speed_of_train : total_distance / total_time = 86.25 := by
  -- Here should be the proof, but we put sorry since we only need the statement
  sorry

end average_speed_of_train_l198_198792


namespace minimum_P_ge_37_l198_198725

noncomputable def minimum_P (x y z : ℝ) : ℝ := 
  (x / y + y / z + z / x) * (y / x + z / y + x / z)

theorem minimum_P_ge_37 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10) : 
  minimum_P x y z ≥ 37 :=
sorry

end minimum_P_ge_37_l198_198725


namespace police_catches_thief_in_two_hours_l198_198948

noncomputable def time_to_catch (speed_thief speed_police distance_police_start lead_time : ℝ) : ℝ :=
  let distance_thief := speed_thief * lead_time
  let initial_distance := distance_police_start - distance_thief
  let relative_speed := speed_police - speed_thief
  initial_distance / relative_speed

theorem police_catches_thief_in_two_hours :
  time_to_catch 20 40 60 1 = 2 := by
  sorry

end police_catches_thief_in_two_hours_l198_198948


namespace sum_common_elements_ap_gp_l198_198648

noncomputable def sum_of_first_10_common_elements : ℕ := 20 * (4^10 - 1) / (4 - 1)

theorem sum_common_elements_ap_gp :
  sum_of_first_10_common_elements = 6990500 :=
by
  unfold sum_of_first_10_common_elements
  sorry

end sum_common_elements_ap_gp_l198_198648


namespace sqrt_difference_of_cubes_is_integer_l198_198455

theorem sqrt_difference_of_cubes_is_integer (a b : ℕ) (h1 : a = 105) (h2 : b = 104) :
  (Int.sqrt (a^3 - b^3) = 181) :=
by
  sorry

end sqrt_difference_of_cubes_is_integer_l198_198455


namespace solve_system_of_equations_l198_198572

theorem solve_system_of_equations (x y : ℝ) (h1 : x + y = 7) (h2 : 2 * x - y = 2) :
  x = 3 ∧ y = 4 :=
by
  sorry

end solve_system_of_equations_l198_198572


namespace inverse_proposition_vertical_angles_false_l198_198628

-- Define the statement "Vertical angles are equal"
def vertical_angles_equal (α β : ℝ) : Prop :=
  α = β

-- Define the inverse proposition
def inverse_proposition_vertical_angles : Prop :=
  ∀ α β : ℝ, α = β → vertical_angles_equal α β

-- The proof goal
theorem inverse_proposition_vertical_angles_false : ¬inverse_proposition_vertical_angles :=
by
  sorry

end inverse_proposition_vertical_angles_false_l198_198628


namespace yura_finishes_problems_by_sept_12_l198_198698

def total_problems := 91
def initial_date := 6 -- September 6
def problems_left_date := 8 -- September 8
def remaining_problems := 46
def decreasing_rate := 1

def problems_solved (z : ℕ) (day : ℕ) : ℕ :=
if day = 6 then z + 1 else if day = 7 then z else if day = 8 then z - 1 else z - (day - 7)

theorem yura_finishes_problems_by_sept_12 :
  ∃ z : ℕ, (problems_solved z 6 + problems_solved z 7 + problems_solved z 8 = total_problems - remaining_problems) ∧
           (problems_solved z 6 + problems_solved z 7 + problems_solved z 8 + problems_solved z 9 + problems_solved z 10 + problems_solved z 11 + problems_solved z 12 = total_problems) :=
sorry

end yura_finishes_problems_by_sept_12_l198_198698


namespace number_of_people_in_group_is_21_l198_198032

-- Definitions based directly on the conditions
def pins_contribution_per_day := 10
def pins_deleted_per_week_per_person := 5
def group_initial_pins := 1000
def final_pins_after_month := 6600
def weeks_in_a_month := 4

-- To be proved: number of people in the group is 21
theorem number_of_people_in_group_is_21 (P : ℕ)
  (h1 : final_pins_after_month - group_initial_pins = 5600)
  (h2 : weeks_in_a_month * (pins_contribution_per_day * 7 - pins_deleted_per_week_per_person) = 260)
  (h3 : 5600 / 260 = 21) :
  P = 21 := 
sorry

end number_of_people_in_group_is_21_l198_198032


namespace hyperbola_vertex_distance_l198_198117

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  4 * x^2 + 24 * x - 4 * y^2 + 16 * y + 44 = 0 →
  2 = 2 :=
by
  intros x y h
  sorry

end hyperbola_vertex_distance_l198_198117


namespace decimal_arithmetic_l198_198797

theorem decimal_arithmetic : 0.45 - 0.03 + 0.008 = 0.428 := by
  sorry

end decimal_arithmetic_l198_198797


namespace max_initial_number_l198_198878

theorem max_initial_number (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    200 = n + a + b + c + d + e ∧ 
    ¬ (n % a = 0) ∧ 
    ¬ ((n + a) % b = 0) ∧ 
    ¬ ((n + a + b) % c = 0) ∧ 
    ¬ ((n + a + b + c) % d = 0) ∧ 
    ¬ ((n + a + b + c + d) % e = 0)) → 
  n ≤ 189 := 
sorry

end max_initial_number_l198_198878


namespace eval_expression_l198_198113

theorem eval_expression :
  (3^3 - 3) - (4^3 - 4) + (5^3 - 5) = 84 := 
by
  sorry

end eval_expression_l198_198113


namespace max_initial_number_l198_198875

noncomputable def verify_addition (n x : ℕ) : Prop := 
  ∀ (a : ℕ), (a ∣ n) → (a ≠ 1) → (n + a = x) → False

theorem max_initial_number :
  ∃ (n : ℕ), 
  (∀ (a1 a2 a3 a4 a5 : ℕ), 
    verify_addition n a1 ∧ verify_addition (n + a1) a2 ∧
    verify_addition (n + a1 + a2) a3 ∧ verify_addition (n + a1 + a2 + a3) a4 ∧
    verify_addition (n + a1 + a2 + a3 + a4) a5 ∧
    (n + a1 + a2 + a3 + a4 + a5 = 200)) ∧
  (∀ m : ℕ, 
    (∃ (a1 a2 a3 a4 a5 : ℕ), 
      verify_addition m a1 ∧ verify_addition (m + a1) a2 ∧
      verify_addition (m + a1 + a2) a3 ∧ verify_addition (m + a1 + a2 + a3) a4 ∧
      verify_addition (m + a1 + a2 + a3 + a4) a5 ∧
      (m + a1 + a2 + a3 + a4 + a5 = 200)) →
    m ≤ 189)
: ∃ n, n = 189 := by
  sorry

end max_initial_number_l198_198875


namespace fraction_of_area_above_line_l198_198580

theorem fraction_of_area_above_line :
  let A := (3, 2)
  let B := (6, 0)
  let side_length := B.fst - A.fst
  let square_area := side_length ^ 2
  let triangle_base := B.fst - A.fst
  let triangle_height := A.snd
  let triangle_area := (1 / 2 : ℚ) * triangle_base * triangle_height
  let area_above_line := square_area - triangle_area
  let fraction_above_line := area_above_line / square_area
  fraction_above_line = (2 / 3 : ℚ) :=
by
  sorry

end fraction_of_area_above_line_l198_198580


namespace initial_amount_of_milk_l198_198560

theorem initial_amount_of_milk (M : ℝ) (h : 0 < M) (h2 : 0.10 * M = 0.05 * (M + 20)) : M = 20 := 
sorry

end initial_amount_of_milk_l198_198560


namespace shelly_total_money_l198_198205

-- Define the conditions
def num_of_ten_dollar_bills : ℕ := 10
def num_of_five_dollar_bills : ℕ := num_of_ten_dollar_bills - 4

-- Problem statement: How much money does Shelly have in all?
theorem shelly_total_money :
  (num_of_ten_dollar_bills * 10) + (num_of_five_dollar_bills * 5) = 130 :=
by
  sorry

end shelly_total_money_l198_198205


namespace men_in_first_group_l198_198861

theorem men_in_first_group (x : ℕ) :
  (20 * 48 = x * 80) → x = 12 :=
by
  intro h_eq
  have : x = (20 * 48) / 80 := sorry
  exact this

end men_in_first_group_l198_198861


namespace g_f_eval_l198_198895

def f (x : ℤ) := x^3 - 2
def g (x : ℤ) := 3 * x^2 + x + 2

theorem g_f_eval : g (f 3) = 1902 := by
  sorry

end g_f_eval_l198_198895


namespace find_first_day_speed_l198_198082

theorem find_first_day_speed (t : ℝ) (d : ℝ) (v : ℝ) (h1 : d = 2.5) 
  (h2 : v * (t - 7/60) = d) (h3 : 10 * (t - 8/60) = d) : v = 9.375 :=
by {
  -- Proof omitted for brevity
  sorry
}

end find_first_day_speed_l198_198082


namespace largest_gold_coins_l198_198947

theorem largest_gold_coins (k : ℤ) (h1 : 13 * k + 3 < 100) : 91 ≤ 13 * k + 3 :=
by
  sorry

end largest_gold_coins_l198_198947


namespace cards_given_l198_198562

def initial_cards : ℕ := 304
def remaining_cards : ℕ := 276
def given_cards : ℕ := initial_cards - remaining_cards

theorem cards_given :
  given_cards = 28 :=
by
  unfold given_cards
  unfold initial_cards
  unfold remaining_cards
  sorry

end cards_given_l198_198562


namespace seans_sum_divided_by_julies_sum_l198_198379

theorem seans_sum_divided_by_julies_sum : 
  let seans_sum := 2 * (∑ k in Finset.range 301, k)
  let julies_sum := ∑ k in Finset.range 301, k
  seans_sum / julies_sum = 2 :=
by 
  sorry

end seans_sum_divided_by_julies_sum_l198_198379


namespace muffin_count_l198_198811

theorem muffin_count (doughnuts cookies muffins : ℕ) (h1 : doughnuts = 50) (h2 : cookies = (3 * doughnuts) / 5) (h3 : muffins = (1 * doughnuts) / 5) : muffins = 10 :=
by sorry

end muffin_count_l198_198811


namespace determinant_matrices_equivalence_l198_198838

-- Define the problem as a Lean theorem statement
theorem determinant_matrices_equivalence (p q r s : ℝ) 
  (h : p * s - q * r = 3) : 
  p * (5 * r + 4 * s) - r * (5 * p + 4 * q) = 12 := 
by 
  sorry

end determinant_matrices_equivalence_l198_198838


namespace solve_linear_eq_l198_198053

theorem solve_linear_eq (x : ℝ) : (x + 1) / 3 = 0 → x = -1 := 
by 
  sorry

end solve_linear_eq_l198_198053


namespace average_of_11_numbers_l198_198398

theorem average_of_11_numbers (a b c d e f g h i j k : ℕ) 
  (h₀ : (a + b + c + d + e + f) / 6 = 19)
  (h₁ : (f + g + h + i + j + k) / 6 = 27)
  (h₂ : f = 34) :
  (a + b + c + d + e + f + g + h + i + j + k) / 11 = 22 := 
by
  sorry

end average_of_11_numbers_l198_198398


namespace monthly_energy_consumption_l198_198552

-- Defining the given conditions
def power_fan_kW : ℝ := 0.075 -- kilowatts
def hours_per_day : ℝ := 8 -- hours per day
def days_per_month : ℝ := 30 -- days per month

-- The math proof statement with conditions and the expected answer
theorem monthly_energy_consumption : (power_fan_kW * hours_per_day * days_per_month) = 18 :=
by
  -- Placeholder for proof
  sorry

end monthly_energy_consumption_l198_198552


namespace digits_of_2_120_l198_198823

theorem digits_of_2_120 (h : ∀ n : ℕ, (10 : ℝ)^(n - 1) ≤ (2 : ℝ)^200 ∧ (2 : ℝ)^200 < (10 : ℝ)^n → n = 61) :
  ∀ m : ℕ, (10 : ℝ)^(m - 1) ≤ (2 : ℝ)^120 ∧ (2 : ℝ)^120 < (10 : ℝ)^m → m = 37 :=
by
  sorry

end digits_of_2_120_l198_198823


namespace quadratic_has_two_distinct_real_roots_l198_198583

theorem quadratic_has_two_distinct_real_roots :
  let a := 1
  let b := 1
  let c := -1
  let discriminant := b^2 - 4 * a * c
  discriminant > 0 :=
by
  let a := 1
  let b := 1
  let c := -1
  let discriminant := b^2 - 4 * a * c
  show discriminant > 0
  sorry

end quadratic_has_two_distinct_real_roots_l198_198583


namespace exists_h_not_divisible_l198_198286

noncomputable def h : ℝ := 1969^2 / 1968

theorem exists_h_not_divisible (h := 1969^2 / 1968) :
  ∃ (h : ℝ), ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
by
  use h
  intro n
  sorry

end exists_h_not_divisible_l198_198286


namespace hall_length_l198_198399

theorem hall_length (L B A : ℝ) (h1 : B = 2 / 3 * L) (h2 : A = 2400) (h3 : A = L * B) : L = 60 := by
  -- proof steps here
  sorry

end hall_length_l198_198399


namespace markus_more_marbles_l198_198730

theorem markus_more_marbles :
  let mara_bags := 12
  let marbles_per_mara_bag := 2
  let markus_bags := 2
  let marbles_per_markus_bag := 13
  let mara_marbles := mara_bags * marbles_per_mara_bag
  let markus_marbles := markus_bags * marbles_per_markus_bag
  mara_marbles + 2 = markus_marbles := 
by
  sorry

end markus_more_marbles_l198_198730


namespace tree_difference_l198_198799

-- Given constants
def Hassans_apple_trees : Nat := 1
def Hassans_orange_trees : Nat := 2

def Ahmeds_orange_trees : Nat := 8
def Ahmeds_apple_trees : Nat := 4 * Hassans_apple_trees

-- Total trees computations
def Ahmeds_total_trees : Nat := Ahmeds_apple_trees + Ahmeds_orange_trees
def Hassans_total_trees : Nat := Hassans_apple_trees + Hassans_orange_trees

-- Theorem to prove the difference in total trees
theorem tree_difference : Ahmeds_total_trees - Hassans_total_trees = 9 := by
  sorry

end tree_difference_l198_198799


namespace chairs_subset_count_l198_198428

theorem chairs_subset_count (ch : Fin 12 → bool) :
  (∃ i : Fin 12, ch i ∧ ch (i + 1) % 12 ∧ ch (i + 2) % 12) →
  2056 = ∑ s : Finset (Fin 12), if (∃ i : Fin 12, ∀ j : Fin n, (i + j) % 12 ∈ s) then 1 else 0 :=
sorry

end chairs_subset_count_l198_198428


namespace find_missing_values_l198_198332

theorem find_missing_values :
  (∃ x y : ℕ, 4 / 5 = 20 / x ∧ 4 / 5 = y / 20 ∧ 4 / 5 = 80 / 100) →
  (x = 25 ∧ y = 16 ∧ 4 / 5 = 80 / 100) :=
by
  sorry

end find_missing_values_l198_198332


namespace parameter_range_exists_solution_l198_198462

theorem parameter_range_exists_solution :
  {a : ℝ | ∃ b : ℝ, ∃ x y : ℝ,
    x^2 + y^2 + 2 * a * (a + y - x) = 49 ∧
    y = 15 * Real.cos (x - b) - 8 * Real.sin (x - b)
  } = {a : ℝ | -24 ≤ a ∧ a ≤ 24} :=
sorry

end parameter_range_exists_solution_l198_198462


namespace solution_in_quadrant_IV_l198_198722

theorem solution_in_quadrant_IV (k : ℝ) :
  (∃ x y : ℝ, x + 2 * y = 4 ∧ k * x - y = 1 ∧ x > 0 ∧ y < 0) ↔ (-1 / 2 < k ∧ k < 2) :=
by
  sorry

end solution_in_quadrant_IV_l198_198722


namespace household_savings_regression_l198_198258

-- Define the problem conditions in Lean
def n := 10
def sum_x := 80
def sum_y := 20
def sum_xy := 184
def sum_x2 := 720

-- Define the averages
def x_bar := sum_x / n
def y_bar := sum_y / n

-- Define the lxx and lxy as per the solution
def lxx := sum_x2 - n * x_bar^2
def lxy := sum_xy - n * x_bar * y_bar

-- Define the regression coefficients
def b_hat := lxy / lxx
def a_hat := y_bar - b_hat * x_bar

-- State the theorem to be proved
theorem household_savings_regression :
  (∀ (x: ℝ), y = b_hat * x + a_hat) :=
by
  sorry -- skip the proof

end household_savings_regression_l198_198258


namespace fourth_graders_bought_more_markers_l198_198097

-- Define the conditions
def cost_per_marker : ℕ := 20
def total_payment_fifth_graders : ℕ := 180
def total_payment_fourth_graders : ℕ := 200

-- Compute the number of markers bought by fifth and fourth graders
def markers_bought_by_fifth_graders : ℕ := total_payment_fifth_graders / cost_per_marker
def markers_bought_by_fourth_graders : ℕ := total_payment_fourth_graders / cost_per_marker

-- Statement to prove
theorem fourth_graders_bought_more_markers : 
  markers_bought_by_fourth_graders - markers_bought_by_fifth_graders = 1 := by
  sorry

end fourth_graders_bought_more_markers_l198_198097


namespace proportion_terms_l198_198539

theorem proportion_terms (x v y z : ℤ) (a b c : ℤ)
  (h1 : x + v = y + z + a)
  (h2 : x^2 + v^2 = y^2 + z^2 + b)
  (h3 : x^4 + v^4 = y^4 + z^4 + c)
  (ha : a = 7) (hb : b = 21) (hc : c = 2625) :
  (x = -3 ∧ v = 8 ∧ y = -6 ∧ z = 4) :=
by
  sorry

end proportion_terms_l198_198539


namespace yanna_sandals_l198_198074

theorem yanna_sandals (shirts_cost: ℕ) (sandal_cost: ℕ) (total_money: ℕ) (change: ℕ) (num_shirts: ℕ)
  (h1: shirts_cost = 5)
  (h2: sandal_cost = 3)
  (h3: total_money = 100)
  (h4: change = 41)
  (h5: num_shirts = 10) : 
  ∃ num_sandals: ℕ, num_sandals = 3 :=
sorry

end yanna_sandals_l198_198074


namespace expected_attacked_squares_is_35_33_l198_198764

-- The standard dimension of a chessboard
def chessboard_size := 8

-- Number of rooks
def num_rooks := 3

-- Expected number of squares under attack by at least one rook
def expected_attacked_squares := 
  (1 - ((49.0 / 64.0) ^ 3)) * 64

theorem expected_attacked_squares_is_35_33 :
  expected_attacked_squares ≈ 35.33 :=
by
  sorry

end expected_attacked_squares_is_35_33_l198_198764


namespace incorrect_conclusion_l198_198982

theorem incorrect_conclusion (p q : ℝ) (h1 : p < 0) (h2 : q < 0) : ¬ ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ (x1 * |x1| + p * x1 + q = 0) ∧ (x2 * |x2| + p * x2 + q = 0) ∧ (x3 * |x3| + p * x3 + q = 0) :=
by
  sorry

end incorrect_conclusion_l198_198982


namespace parking_garage_capacity_l198_198786

open Nat

-- Definitions from the conditions
def first_level_spaces : Nat := 90
def second_level_spaces : Nat := first_level_spaces + 8
def third_level_spaces : Nat := second_level_spaces + 12
def fourth_level_spaces : Nat := third_level_spaces - 9
def initial_parked_cars : Nat := 100

-- The proof statement
theorem parking_garage_capacity : 
  (first_level_spaces + second_level_spaces + third_level_spaces + fourth_level_spaces - initial_parked_cars) = 299 := 
  by 
    sorry

end parking_garage_capacity_l198_198786


namespace least_positive_integer_divisible_by_four_primes_l198_198062

open Nat

theorem least_positive_integer_divisible_by_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  (p1 * p2 * p3 * p4 = 210) :=
begin
  let p1 := 2,
  let p2 := 3,
  let p3 := 5,
  let p4 := 7,
  sorry
end

end least_positive_integer_divisible_by_four_primes_l198_198062


namespace tournament_committees_count_l198_198549

theorem tournament_committees_count :
  ∀ (teams : Fin 5 → Fin 7 → Prop) (female_only : Fin 5) (at_least_2_females : ∀ i, 2 ≤ Finset.card {x | teams i x}) (female_only_condition : ∀ j, teams female_only j → female_only = teams female_only j),
  ∃ (count : ℕ), count = 4 * ( Nat.choose 7 3 * (Nat.choose 7 2)^3 * 1 ) ∧ count = 1,296,540 :=
by
  intros teams female_only at_least_2_females female_only_condition
  use 4 * ( Nat.choose 7 3 * (Nat.choose 7 2) ^ 3 * 1 )
  have h_formula : 4 * (35 * 21 ^ 3 * 1) = 1,296,540 := by norm_num
  exact ⟨rfl, h_formula⟩

end tournament_committees_count_l198_198549


namespace paintings_in_four_weeks_l198_198096

def weekly_hours := 30
def hours_per_painting := 3
def weeks := 4

theorem paintings_in_four_weeks (w_hours : ℕ) (h_per_painting : ℕ) (n_weeks : ℕ) (result : ℕ) :
  w_hours = weekly_hours →
  h_per_painting = hours_per_painting →
  n_weeks = weeks →
  result = (w_hours / h_per_painting) * n_weeks →
  result = 40 :=
by
  intros
  sorry

end paintings_in_four_weeks_l198_198096


namespace participants_who_drank_neither_l198_198631

-- Conditions
variables (total_participants : ℕ) (coffee_drinkers : ℕ) (juice_drinkers : ℕ) (both_drinkers : ℕ)

-- Initial Facts from the Conditions
def conditions := total_participants = 30 ∧ coffee_drinkers = 15 ∧ juice_drinkers = 18 ∧ both_drinkers = 7

-- The statement to prove
theorem participants_who_drank_neither : conditions total_participants coffee_drinkers juice_drinkers both_drinkers → 
  (total_participants - (coffee_drinkers + juice_drinkers - both_drinkers)) = 4 :=
by
  intros
  sorry

end participants_who_drank_neither_l198_198631


namespace circles_intersect_l198_198404

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*x - 4*y - 1 = 0

theorem circles_intersect : 
  (∃ (x y : ℝ), circle1 x y ∧ circle2 x y) := 
sorry

end circles_intersect_l198_198404


namespace packages_of_noodles_tom_needs_l198_198765

def beef_weight : ℕ := 10
def noodles_needed_factor : ℕ := 2
def noodles_available : ℕ := 4
def noodle_package_weight : ℕ := 2

theorem packages_of_noodles_tom_needs :
  (beef_weight * noodles_needed_factor - noodles_available) / noodle_package_weight = 8 :=
by
  sorry

end packages_of_noodles_tom_needs_l198_198765


namespace base_conversion_l198_198043

theorem base_conversion (C D : ℕ) (hC : 0 ≤ C) (hC_lt : C < 8) (hD : 0 ≤ D) (hD_lt : D < 5) :
  (8 * C + D = 5 * D + C) → (8 * C + D = 0) :=
by 
  intro h
  sorry

end base_conversion_l198_198043


namespace yura_finishes_on_correct_date_l198_198701

-- Let there be 91 problems in the textbook.
-- Let Yura start solving problems on September 6th.
def total_problems : Nat := 91
def start_date : Nat := 6

-- Each morning, starting from September 7, he solves one problem less than the previous morning.
def problems_solved (n : Nat) : Nat := if n = 0 then 0 else problems_solved (n - 1) - 1

-- On the evening of September 8, there are 46 problems left to solve.
def remaining_problems_sept_8 : Nat := 46

-- The question is to find on which date Yura will finish solving the textbook
def finish_date : Nat := 12

theorem yura_finishes_on_correct_date : ∃ z : Nat, (problems_solved z * 3 = total_problems - remaining_problems_sept_8) ∧ (z = 15) ∧ finish_date = 12 := by sorry

end yura_finishes_on_correct_date_l198_198701


namespace axisymmetric_triangle_is_isosceles_l198_198532

-- Define a triangle and its properties
structure Triangle :=
  (a b c : ℝ) -- Triangle sides as real numbers
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

def is_axisymmetric (T : Triangle) : Prop :=
  -- Here define what it means for a triangle to be axisymmetric
  -- This is often represented as having at least two sides equal
  (T.a = T.b ∨ T.b = T.c ∨ T.c = T.a)

def is_isosceles (T : Triangle) : Prop :=
  -- Definition of an isosceles triangle
  (T.a = T.b ∨ T.b = T.c ∨ T.c = T.a)

-- The theorem to be proven
theorem axisymmetric_triangle_is_isosceles (T : Triangle) (h : is_axisymmetric T) : is_isosceles T :=
by {
  -- Proof would go here
  sorry
}

end axisymmetric_triangle_is_isosceles_l198_198532


namespace BigJoe_is_8_feet_l198_198805

variable (Pepe_height : ℝ) (h1 : Pepe_height = 4.5)
variable (Frank_height : ℝ) (h2 : Frank_height = Pepe_height + 0.5)
variable (Larry_height : ℝ) (h3 : Larry_height = Frank_height + 1)
variable (Ben_height : ℝ) (h4 : Ben_height = Larry_height + 1)
variable (BigJoe_height : ℝ) (h5 : BigJoe_height = Ben_height + 1)

theorem BigJoe_is_8_feet : BigJoe_height = 8 := by
  sorry

end BigJoe_is_8_feet_l198_198805


namespace smallest_period_f_max_value_h_l198_198846

-- Define the functions f and g
def f (x : ℝ) : ℝ := cos (π / 3 + x) * cos (π / 3 - x)
def g (x : ℝ) : ℝ := 1 / 2 * sin (2 * x) - 1 / 4
def h (x : ℝ) : ℝ := f x - g x

-- Statement 1: The smallest positive period of f(x) is π
theorem smallest_period_f : (∀ x : ℝ, f (x + π) = f x) ∧ (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ π) :=
begin
  sorry
end

-- Statement 2: The maximum value of h(x) is sqrt(2)/2, and the set of x values where h(x) attains its maximum
theorem max_value_h :
  (∀ x : ℝ, h x ≤ sqrt 2 / 2) ∧
  (∀ x : ℝ, h x = sqrt 2 / 2 ↔ ∃ k : ℤ, x = 3 * π / 8 + k * π) :=
begin
  sorry
end

end smallest_period_f_max_value_h_l198_198846


namespace cost_of_one_book_l198_198906

theorem cost_of_one_book (x : ℝ) : 
  (9 * x = 11) ∧ (13 * x = 15) → x = 1.23 :=
by sorry

end cost_of_one_book_l198_198906


namespace steps_Tom_by_time_Matt_reaches_220_l198_198191

theorem steps_Tom_by_time_Matt_reaches_220 (rate_Matt rate_Tom : ℕ) (time_Matt_time_Tom : ℕ) (steps_Matt steps_Tom : ℕ) :
  rate_Matt = 20 →
  rate_Tom = rate_Matt + 5 →
  steps_Matt = 220 →
  time_Matt_time_Tom = steps_Matt / rate_Matt →
  steps_Tom = steps_Matt + time_Matt_time_Tom * 5 →
  steps_Tom = 275 :=
by
  intros h1 h2 h3 h4 h5
  rw [h3, h4, h2, h1] at h5
  norm_num at h5
  exact h5

end steps_Tom_by_time_Matt_reaches_220_l198_198191


namespace more_trees_in_ahmeds_orchard_l198_198801

-- Given conditions
def ahmed_orange_trees : ℕ := 8
def hassan_apple_trees : ℕ := 1
def hassan_orange_trees : ℕ := 2
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees
def ahmed_total_trees : ℕ := ahmed_orange_trees + ahmed_apple_trees
def hassan_total_trees : ℕ := hassan_apple_trees + hassan_orange_trees

-- Statement to be proven
theorem more_trees_in_ahmeds_orchard : ahmed_total_trees - hassan_total_trees = 9 :=
by
  sorry

end more_trees_in_ahmeds_orchard_l198_198801


namespace consecutive_integers_product_sum_l198_198931

theorem consecutive_integers_product_sum (a b c d : ℕ) :
  a * b * c * d = 3024 ∧ b = a + 1 ∧ c = b + 1 ∧ d = c + 1 → a + b + c + d = 30 :=
by
  sorry

end consecutive_integers_product_sum_l198_198931


namespace correct_relation_l198_198665

-- Define the set A
def A : Set ℤ := { x | x^2 - 4 = 0 }

-- The statement that 2 is an element of A
theorem correct_relation : 2 ∈ A :=
by 
    -- We skip the proof here
    sorry

end correct_relation_l198_198665


namespace complementary_angle_of_60_l198_198046

theorem complementary_angle_of_60 (a : ℝ) : 
  (∀ (a b : ℝ), a + b = 180 → a = 60 → b = 120) := 
by
  sorry

end complementary_angle_of_60_l198_198046


namespace mary_initial_blue_crayons_l198_198190

/-- **Mathematically equivalent proof problem**:
  Given that Mary has 5 green crayons and gives away 3 green crayons and 1 blue crayon,
  and she has 9 crayons left, prove that she initially had 8 blue crayons. 
  -/
theorem mary_initial_blue_crayons (initial_green_crayons : ℕ) (green_given_away : ℕ) (blue_given_away : ℕ)
  (crayons_left : ℕ) (initial_crayons : ℕ) :
  initial_green_crayons = 5 →
  green_given_away = 3 →
  blue_given_away = 1 →
  crayons_left = 9 →
  initial_crayons = crayons_left + (green_given_away + blue_given_away) →
  initial_crayons - initial_green_crayons = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end mary_initial_blue_crayons_l198_198190


namespace price_of_other_pieces_l198_198182

theorem price_of_other_pieces (total_spent : ℕ) (total_pieces : ℕ) (price_piece1 : ℕ) (price_piece2 : ℕ) 
  (remaining_pieces : ℕ) (price_remaining_piece : ℕ) (h1 : total_spent = 610) (h2 : total_pieces = 7)
  (h3 : price_piece1 = 49) (h4 : price_piece2 = 81) (h5 : remaining_pieces = (total_pieces - 2))
  (h6 : total_spent - price_piece1 - price_piece2 = remaining_pieces * price_remaining_piece) :
  price_remaining_piece = 96 := 
by
  sorry

end price_of_other_pieces_l198_198182


namespace fraction_value_l198_198104

theorem fraction_value :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) / 
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400)) = 244.375 :=
by
  -- The proof would be provided here, but we are skipping it as per the instructions.
  sorry

end fraction_value_l198_198104


namespace subsets_with_at_least_three_adjacent_chairs_l198_198424

theorem subsets_with_at_least_three_adjacent_chairs :
  let n := 12
  in (number of subsets of circularly arranged chairs) ≥ 3 = 1634 :=
sorry

end subsets_with_at_least_three_adjacent_chairs_l198_198424


namespace percentage_increase_from_1200_to_1680_is_40_l198_198785

theorem percentage_increase_from_1200_to_1680_is_40 :
  let initial_value := 1200
  let final_value := 1680
  let percentage_increase := ((final_value - initial_value) / initial_value) * 100
  percentage_increase = 40 := by
  let initial_value := 1200
  let final_value := 1680
  let percentage_increase := ((final_value - initial_value) / initial_value) * 100
  sorry

end percentage_increase_from_1200_to_1680_is_40_l198_198785


namespace expected_prize_money_l198_198865

theorem expected_prize_money :
  let a1 := 1 / 7
  let prob1 := a1
  let prob2 := 2 * a1
  let prob3 := 4 * a1
  let prize1 := 700
  let prize2 := 700 - 140
  let prize3 := 700 - 140 * 2
  let expected_money := prize1 * prob1 + prize2 * prob2 + prize3 * prob3
  expected_money = 500 := 
by
  -- Definitions
  let a1 := 1 / 7
  let prob1 := a1
  let prob2 := 2 * a1
  let prob3 := 4 * a1
  let prize1 := 700
  let prize2 := 700 - 140
  let prize3 := 700 - 140 * 2
  let expected_money := prize1 * prob1 + prize2 * prob2 + prize3 * prob3

  -- Calculate
  sorry -- Proof to show expected_money equals 500

end expected_prize_money_l198_198865


namespace calculate_box_sum_l198_198828

def box (a b c : Int) : ℚ := a^b + b^c - c^a

theorem calculate_box_sum :
  box 2 3 (-1) + box (-1) 2 3 = 16 := by
  sorry

end calculate_box_sum_l198_198828


namespace _l198_198473

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l198_198473


namespace angle_bisector_slope_l198_198397

/-
Given conditions:
1. line1: y = 2x
2. line2: y = 4x
Prove:
k = (sqrt(21) - 6) / 7
-/

theorem angle_bisector_slope :
  let m1 := 2
  let m2 := 4
  let k := (Real.sqrt 21 - 6) / 7
  (1 - m1 * m2) ≠ 0 →
  k = (m1 + m2 - Real.sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)
:=
sorry

end angle_bisector_slope_l198_198397


namespace solve_equation_l198_198208

theorem solve_equation : ∃ x : ℝ, (2 * x - 1) / 3 - (x - 2) / 6 = 2 ∧ x = 4 :=
by
  sorry

end solve_equation_l198_198208


namespace yura_finishes_textbook_on_sep_12_l198_198689

theorem yura_finishes_textbook_on_sep_12 :
  ∃ end_date : ℕ,
    let total_problems := 91,
        problems_left_on_sep_8 := 46,
        solved_until_sep_8 := total_problems - problems_left_on_sep_8,
        z := solved_until_sep_8 / 3, -- Since 3z = solved_until_sep_8
        sep6_solved := z + 1,
        sep7_solved := z,
        sep8_solved := z - 1,
        remaining_problems := total_problems - (sep6_solved + sep7_solved + sep8_solved),
        end_date := 8 + (sep8_solved * (sep8_solved + 1)) / 2,
        -- 8th + sum of an arithmetic series starting from 13 down to 1 problem
    end_date = 12 := sorry

end yura_finishes_textbook_on_sep_12_l198_198689


namespace problem1_problem2_l198_198659

-- Part 1 proof that a_n - a_n+2 = 2 given the conditions
theorem problem1 (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : ∀ n, b n = a n + a (n + 1))
  (h2 : b 1 = -3) (h3 : b 2 + b 3 = -12) : ∀ n, a n - a(n + 2) = 2 :=
  sorry

-- Part 2 proof that S_n = -n^2/2 - n/2 for the sum of the first n terms of an arithmetic sequence
theorem problem2 (a : ℕ → ℝ) (S : ℕ → ℝ) (h : ∀ n, a n - a(n + 2) = 2) 
  (arith_seq : ∀ n, a(n+1) = a(n) + (-1)) : ∀ n, S n = -n^2/2 - n/2 :=
  sorry

end problem1_problem2_l198_198659


namespace sum_of_possible_w_l198_198971

theorem sum_of_possible_w :
  ∃ w x y z : ℤ, w > x ∧ x > y ∧ y > z ∧ w + x + y + z = 44 ∧ 
  {w - x, w - y, w - z, x - y, x - z, y - z} = {1, 3, 4, 5, 6, 9} ∧
  (w = 16 ∨ w = 15) ∧ w + 15 + 16 = 31 :=
by
  -- Here we state the theorem and outline our conditions.
  sorry

end sum_of_possible_w_l198_198971


namespace primes_with_consecutives_l198_198598

-- Define what it means for a number to be prime
def is_prime (n : Nat) := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬ (n % m = 0)

-- Define the main theorem to prove
theorem primes_with_consecutives (p : Nat) : is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4) → p = 3 :=
by
  sorry

end primes_with_consecutives_l198_198598


namespace min_PM_PN_min_PM_squared_PN_squared_l198_198992

noncomputable def min_value_PM_PN := 3 * Real.sqrt 5

noncomputable def min_value_PM_squared_PN_squared := 229 / 10

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨2, 5⟩
def N : Point := ⟨-2, 4⟩

def on_line (P : Point) : Prop :=
  P.x - 2 * P.y + 3 = 0

theorem min_PM_PN {P : Point} (h : on_line P) :
  dist (P.x, P.y) (M.x, M.y) + dist (P.x, P.y) (N.x, N.y) = min_value_PM_PN := sorry

theorem min_PM_squared_PN_squared {P : Point} (h : on_line P) :
  (dist (P.x, P.y) (M.x, M.y))^2 + (dist (P.x, P.y) (N.x, N.y))^2 = min_value_PM_squared_PN_squared := sorry

end min_PM_PN_min_PM_squared_PN_squared_l198_198992


namespace rita_months_needed_l198_198541

def total_hours_required : ℕ := 2500
def backstroke_hours : ℕ := 75
def breaststroke_hours : ℕ := 25
def butterfly_hours : ℕ := 200
def hours_per_month : ℕ := 300

def total_completed_hours : ℕ := backstroke_hours + breaststroke_hours + butterfly_hours
def remaining_hours : ℕ := total_hours_required - total_completed_hours
def months_needed (remaining_hours hours_per_month : ℕ) : ℕ := (remaining_hours + hours_per_month - 1) / hours_per_month

theorem rita_months_needed : months_needed remaining_hours hours_per_month = 8 := by
  -- Lean 4 proof goes here
  sorry

end rita_months_needed_l198_198541


namespace largest_initial_number_l198_198882

theorem largest_initial_number :
  ∃ n : ℕ, 
    (∀ (a1 a2 a3 a4 a5 : ℕ),
      (n + a1).gcd a1 = 1 ∧
      (n + a1 + a2).gcd a2 = 1 ∧
      (n + a1 + a2 + a3).gcd a3 = 1 ∧
      (n + a1 + a2 + a3 + a4).gcd a4 = 1 ∧
      (n + a1 + a2 + a3 + a4 + a5).gcd a5 = 1 ∧
      n + a1 + a2 + a3 + a4 + a5 = 200) 
    → n = 189 :=
begin
  sorry
end

end largest_initial_number_l198_198882


namespace minimum_votes_for_tall_to_win_l198_198546

-- Definitions based on the conditions
def num_voters := 135
def num_districts := 5
def num_precincts_per_district := 9
def num_voters_per_precinct := 3

-- Tall won the contest
def tall_won := True

-- Winning conditions
def majority_precinct_vote (votes_for_tall : ℕ) : Prop :=
  votes_for_tall >= 2

def majority_district_win (precincts_won_by_tall : ℕ) : Prop :=
  precincts_won_by_tall >= 5

def majority_contest_win (districts_won_by_tall : ℕ) : Prop :=
  districts_won_by_tall >= 3

-- Prove the minimum number of voters who could have voted for Tall
theorem minimum_votes_for_tall_to_win : 
  ∃ (votes : ℕ), votes = 30 ∧ majority_contest_win 3 ∧ 
  (∀ d, d < 3 → majority_district_win 5) ∧ 
  (∀ p, p < 5 → majority_precinct_vote 2) :=
by
  sorry

end minimum_votes_for_tall_to_win_l198_198546


namespace k_value_l198_198408

theorem k_value (k : ℝ) (h : (k / 4) + (-k / 3) = 2) : k = -24 :=
by
  sorry

end k_value_l198_198408


namespace evaluate_f_at_t_plus_one_l198_198185

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 1

-- Define the proposition to be proved
theorem evaluate_f_at_t_plus_one (t : ℝ) : f (t + 1) = 3 * t + 2 := by
  sorry

end evaluate_f_at_t_plus_one_l198_198185


namespace cdf_from_pdf_l198_198522

noncomputable def pdf (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x ≤ Real.pi / 2 then Real.cos x
  else 0

noncomputable def cdf (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x ≤ Real.pi / 2 then Real.sin x
  else 1

theorem cdf_from_pdf (x : ℝ) : 
  ∀ x : ℝ, cdf x = 
    if x ≤ 0 then 0
    else if 0 < x ∧ x ≤ Real.pi / 2 then Real.sin x
    else 1 :=
by
  sorry

end cdf_from_pdf_l198_198522


namespace degree_to_radian_l198_198638

theorem degree_to_radian (deg : ℝ) (h : deg = 50) : deg * (Real.pi / 180) = (5 / 18) * Real.pi :=
by
  -- placeholder for the proof
  sorry

end degree_to_radian_l198_198638


namespace solve_textbook_by_12th_l198_198693

/-!
# Problem Statement
There are 91 problems in a textbook. Yura started solving them on September 6 and solves one problem less each subsequent morning. By the evening of September 8, there are 46 problems left to solve.

We need to prove that Yura finishes solving all the problems by September 12.
-/

def initial_problems : ℕ := 91
def problems_left_by_evening_of_8th : ℕ := 46

def problems_solved_by_evening_of_8th : ℕ :=
  initial_problems - problems_left_by_evening_of_8th

def z : ℕ := (problems_solved_by_evening_of_8th / 3)

theorem solve_textbook_by_12th 
    (total_problems : ℕ)
    (problems_left : ℕ)
    (solved_by_evening_8th : ℕ)
    (daily_problem_count : ℕ) :
    (total_problems = initial_problems) →
    (problems_left = problems_left_by_evening_of_8th) →
    (solved_by_evening_8th = problems_solved_by_evening_of_8th) →
    (daily_problem_count = z) →
    ∃ (finishing_date : ℕ), finishing_date = 12 :=
  by
    intros _ _ _ _
    sorry

end solve_textbook_by_12th_l198_198693


namespace degree_of_product_l198_198211

-- Definitions for the conditions
def isDegree (p : Polynomial ℝ) (n : ℕ) : Prop :=
  p.degree = n

variable {h j : Polynomial ℝ}

-- Given conditions
axiom h_deg : isDegree h 3
axiom j_deg : isDegree j 6

-- The theorem to prove
theorem degree_of_product : h.degree = 3 → j.degree = 6 → (Polynomial.degree (Polynomial.comp h (Polynomial.X ^ 4) * Polynomial.comp j (Polynomial.X ^ 3)) = 30) :=
by
  intros h3 j6
  sorry

end degree_of_product_l198_198211


namespace minimum_votes_for_tall_to_win_l198_198545

-- Definitions based on the conditions
def num_voters := 135
def num_districts := 5
def num_precincts_per_district := 9
def num_voters_per_precinct := 3

-- Tall won the contest
def tall_won := True

-- Winning conditions
def majority_precinct_vote (votes_for_tall : ℕ) : Prop :=
  votes_for_tall >= 2

def majority_district_win (precincts_won_by_tall : ℕ) : Prop :=
  precincts_won_by_tall >= 5

def majority_contest_win (districts_won_by_tall : ℕ) : Prop :=
  districts_won_by_tall >= 3

-- Prove the minimum number of voters who could have voted for Tall
theorem minimum_votes_for_tall_to_win : 
  ∃ (votes : ℕ), votes = 30 ∧ majority_contest_win 3 ∧ 
  (∀ d, d < 3 → majority_district_win 5) ∧ 
  (∀ p, p < 5 → majority_precinct_vote 2) :=
by
  sorry

end minimum_votes_for_tall_to_win_l198_198545


namespace fraction_eggs_given_to_Sofia_l198_198098

variables (m : ℕ) -- Number of eggs Mia has
def Sofia_eggs := 3 * m
def Pablo_eggs := 4 * Sofia_eggs
def Lucas_eggs := 0

theorem fraction_eggs_given_to_Sofia (h1 : Pablo_eggs = 12 * m) :
  (1 : ℚ) / (12 : ℚ) = 1 / 12 := by sorry

end fraction_eggs_given_to_Sofia_l198_198098


namespace S21_sum_is_4641_l198_198526

-- Define the conditions and the sum of the nth group
def first_number_in_group (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 1

def last_number_in_group (n : ℕ) : ℕ :=
  first_number_in_group n + (n - 1)

def sum_of_group (n : ℕ) : ℕ :=
  n * (first_number_in_group n + last_number_in_group n) / 2

-- The theorem to prove
theorem S21_sum_is_4641 : sum_of_group 21 = 4641 := by
  sorry

end S21_sum_is_4641_l198_198526


namespace complement_U_B_eq_D_l198_198365

def B (x : ℝ) : Prop := x^2 - 3 * x + 2 < 0
def U : Set ℝ := Set.univ
def complement_U_B : Set ℝ := U \ {x | B x}

theorem complement_U_B_eq_D : complement_U_B = {x | x ≤ 1 ∨ x ≥ 2} := by
  sorry

end complement_U_B_eq_D_l198_198365


namespace car_distribution_l198_198623

theorem car_distribution :
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let total_distributed_first_three := first_supplier + second_supplier + third_supplier
  let remaining_cars := total_cars - total_distributed_first_three
  let fourth_supplier := remaining_cars / 2
  let fifth_supplier := remaining_cars / 2
  fourth_supplier = 325000 :=
by
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let total_distributed_first_three := first_supplier + second_supplier + third_supplier
  let remaining_cars := total_cars - total_distributed_first_three
  let fourth_supplier := remaining_cars / 2
  let fifth_supplier := remaining_cars / 2
  sorry

end car_distribution_l198_198623


namespace circle_chairs_adjacent_l198_198426

theorem circle_chairs_adjacent : 
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)) 
    = 1634 :=
by
    let chairs : Finset ℕ := Finset.range 12 in
    let subsets_containing_at_least_three_adjacent (s : Finset ℕ) : Prop :=
        (∃ i : ℕ, i ∈ Finset.range 12 ∧ s ⊆ Finset.image (λ j, (i + j) % 12) (Finset.range 3)) in
    have h := Finset.card (Finset.filter subsets_containing_at_least_three_adjacent (Finset.powerset chairs)),
    have answer := 1634,
    sorry

end circle_chairs_adjacent_l198_198426


namespace fraction_to_decimal_l198_198304

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l198_198304


namespace sequence_positive_from_26_l198_198835

-- Define the sequence with its general term formula.
def a_n (n : ℕ) : ℤ := 4 * n - 102

-- State the theorem that for all n ≥ 26, a_n > 0.
theorem sequence_positive_from_26 (n : ℕ) (h : n ≥ 26) : a_n n > 0 := by
  sorry

end sequence_positive_from_26_l198_198835


namespace remainder_when_divided_l198_198493

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l198_198493


namespace fraction_to_decimal_l198_198315

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198315


namespace minimize_f_l198_198654

noncomputable def f (x a : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

theorem minimize_f (a : ℝ) : a = 82 / 43 :=
by
  sorry

end minimize_f_l198_198654


namespace markus_more_marbles_l198_198731

theorem markus_more_marbles :
  let mara_bags := 12
  let marbles_per_mara_bag := 2
  let markus_bags := 2
  let marbles_per_markus_bag := 13
  let mara_marbles := mara_bags * marbles_per_mara_bag
  let markus_marbles := markus_bags * marbles_per_markus_bag
  mara_marbles + 2 = markus_marbles := 
by
  sorry

end markus_more_marbles_l198_198731


namespace find_x_minus_y_l198_198587

variables (x y z : ℝ)

theorem find_x_minus_y (h1 : x - (y + z) = 19) (h2 : x - y - z = 7): x - y = 13 :=
by {
  sorry
}

end find_x_minus_y_l198_198587


namespace min_dist_intersection_points_eq_l198_198149

theorem min_dist_intersection_points_eq :
  ∃ m : ℝ, (0 < m) →
  |m^2 - log m| = (1/2) + (1/2) * log 2 :=
by
  sorry

end min_dist_intersection_points_eq_l198_198149


namespace union_sets_l198_198837

def A : Set ℕ := {2, 1, 3}
def B : Set ℕ := {2, 3, 5}

theorem union_sets : A ∪ B = {1, 2, 3, 5} := 
by {
  sorry
}

end union_sets_l198_198837


namespace no_such_function_exists_l198_198984

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x - f y) = 1 + x - y := by
  sorry

end no_such_function_exists_l198_198984


namespace largest_initial_number_l198_198887

-- Let's define the conditions and the result
def valid_addition (n a : ℕ) : Prop := ∃ k : ℕ, n = a * k + r ∧ 0 < r ∧ r < a

def valid_operations (initial : ℕ) (final : ℕ) (steps : ℕ → ℕ → ℕ) : Prop :=
  ∃ (a b c d e : ℕ), valid_addition initial a ∧
                      valid_addition (initial + a) b ∧
                      valid_addition (initial + a + b) c ∧
                      valid_addition (initial + a + b + c) d ∧
                      valid_addition (initial + a + b + c + d) e ∧
                      initial + a + b + c + d + e = final

theorem largest_initial_number :
  ∃ n : ℕ, (valid_operations n 200 (λn a, n + a)) ∧ (∀ m : ℕ, valid_operations m 200 (λn a, n + a) → m ≤ n) :=
sorry

end largest_initial_number_l198_198887


namespace find_angle_B_l198_198163

theorem find_angle_B (A B C : ℝ) (a b c : ℝ) 
  (h1 : A = 45) 
  (h2 : a = 6) 
  (h3 : b = 3 * Real.sqrt 2)
  (h4 : ∀ A' B' C' : ℝ, 
        ∃ a' b' c' : ℝ, 
        (a' = a) ∧ (b' = b) ∧ (A' = A) ∧ 
        (b' < a') → (B' < A') ∧ (A' = 45)) :
  B = 30 :=
by
  sorry

end find_angle_B_l198_198163


namespace no_natural_solution_l198_198568

theorem no_natural_solution :
  ¬ (∃ (x y : ℕ), 2 * x + 3 * y = 6) :=
by
sorry

end no_natural_solution_l198_198568


namespace evaluate_expression_l198_198814

theorem evaluate_expression :
  let a := 3 * 4 * 5
  let b := (1 : ℝ) / 3
  let c := (1 : ℝ) / 4
  let d := (1 : ℝ) / 5
  (a : ℝ) * (b + c - d) = 23 := by
  sorry

end evaluate_expression_l198_198814


namespace first_year_with_sum_of_digits_10_after_2200_l198_198782

/-- Prove that the first year after 2200 in which the sum of the digits equals 10 is 2224. -/
theorem first_year_with_sum_of_digits_10_after_2200 :
  ∃ y, y > 2200 ∧ (List.sum (y.digits 10) = 10) ∧ 
       ∀ z, (2200 < z ∧ z < y) → (List.sum (z.digits 10) ≠ 10) :=
sorry

end first_year_with_sum_of_digits_10_after_2200_l198_198782


namespace probability_at_least_five_l198_198350

open Classical 

def redPackets : list ℝ := [2.63, 1.95, 3.26, 1.77, 0.39]

def allCombinationsSumAtLeastFive : list (ℝ × ℝ) :=
  [(2.63, 3.26), (3.26, 1.95), (3.26, 1.77)]

def validCombinationCount : ℕ := 3
def totalCombinationCount : ℕ := 5.choose 2
def desiredProbability : ℝ := validCombinationCount / totalCombinationCount

theorem probability_at_least_five :
  desiredProbability = 3 / 10 :=
by
  sorry

end probability_at_least_five_l198_198350


namespace problem_part1_problem_part2_l198_198840

theorem problem_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  a + b + c ≥ 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c := 
sorry

theorem problem_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt a + Real.sqrt b + Real.sqrt c :=
sorry

end problem_part1_problem_part2_l198_198840


namespace equal_cost_at_150_miles_l198_198915

def cost_Safety (m : ℝ) := 41.95 + 0.29 * m
def cost_City (m : ℝ) := 38.95 + 0.31 * m
def cost_Metro (m : ℝ) := 44.95 + 0.27 * m

theorem equal_cost_at_150_miles (m : ℝ) :
  cost_Safety m = cost_City m ∧ cost_Safety m = cost_Metro m → m = 150 :=
by
  sorry

end equal_cost_at_150_miles_l198_198915


namespace tile_difference_is_11_l198_198413

-- Define the initial number of blue and green tiles
def initial_blue_tiles : ℕ := 13
def initial_green_tiles : ℕ := 6

-- Define the number of additional green tiles added as border
def additional_green_tiles : ℕ := 18

-- Define the total number of green tiles in the new figure
def total_green_tiles : ℕ := initial_green_tiles + additional_green_tiles

-- Define the total number of blue tiles in the new figure (remains the same)
def total_blue_tiles : ℕ := initial_blue_tiles

-- Define the difference between the total number of green tiles and blue tiles
def tile_difference : ℕ := total_green_tiles - total_blue_tiles

-- The theorem stating that the difference between the total number of green tiles 
-- and the total number of blue tiles in the new figure is 11
theorem tile_difference_is_11 : tile_difference = 11 := by
  sorry

end tile_difference_is_11_l198_198413


namespace fraction_to_decimal_l198_198291

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l198_198291


namespace red_grapes_in_salad_l198_198870

theorem red_grapes_in_salad {G R B : ℕ} 
  (h1 : R = 3 * G + 7)
  (h2 : B = G - 5)
  (h3 : G + R + B = 102) : R = 67 :=
sorry

end red_grapes_in_salad_l198_198870


namespace inequality_proof_l198_198911

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a^3 + b^3 + c^3 = 3) :
  (1 / (a^4 + 3) + 1 / (b^4 + 3) + 1 / (c^4 + 3) >= 3 / 4) :=
by
  sorry

end inequality_proof_l198_198911


namespace remainder_when_divided_82_l198_198347

theorem remainder_when_divided_82 (x : ℤ) (k m : ℤ) (R : ℤ) (h1 : 0 ≤ R) (h2 : R < 82)
    (h3 : x = 82 * k + R) (h4 : x + 7 = 41 * m + 12) : R = 5 :=
by
  sorry

end remainder_when_divided_82_l198_198347


namespace grasshopper_jump_distance_l198_198224

variable (F G M : ℕ) -- F for frog's jump, G for grasshopper's jump, M for mouse's jump

theorem grasshopper_jump_distance (h1 : F = G + 39) (h2 : M = F - 94) (h3 : F = 58) : G = 19 := 
by
  sorry

end grasshopper_jump_distance_l198_198224


namespace parallel_vectors_tan_l198_198002

theorem parallel_vectors_tan (θ : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h₀ : a = (2, Real.sin θ))
  (h₁ : b = (1, Real.cos θ))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  Real.tan θ = 2 := 
sorry

end parallel_vectors_tan_l198_198002


namespace sean_div_julie_l198_198384

-- Define the sum of the first n integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum: sum of even integers from 2 to 600
def sean_sum : ℕ := 2 * sum_first_n 300

-- Define Julie's sum: sum of integers from 1 to 300
def julie_sum : ℕ := sum_first_n 300

-- Prove that Sean's sum divided by Julie's sum equals 2
theorem sean_div_julie : sean_sum / julie_sum = 2 := by
  sorry

end sean_div_julie_l198_198384


namespace smartphone_customers_l198_198160

theorem smartphone_customers (k : ℝ) (p1 p2 c1 c2 : ℝ)
  (h₁ : p1 * c1 = k)
  (h₂ : 20 = p1)
  (h₃ : 200 = c1)
  (h₄ : 400 = c2) :
  p2 * c2 = k  → p2 = 10 :=
by
  sorry

end smartphone_customers_l198_198160


namespace find_two_digit_number_l198_198075

-- A type synonym for digit
def Digit := {n : ℕ // n < 10}

-- Define the conditions
variable (X Y : Digit)
-- The product of the digits is 8
def product_of_digits : Prop := X.val * Y.val = 8

-- When 18 is added, digits are reversed
def digits_reversed : Prop := 10 * X.val + Y.val + 18 = 10 * Y.val + X.val

-- The question translated to Lean: Prove that the two-digit number is 24
theorem find_two_digit_number (h1 : product_of_digits X Y) (h2 : digits_reversed X Y) : 10 * X.val + Y.val = 24 :=
  sorry

end find_two_digit_number_l198_198075


namespace inequality_proof_l198_198993

variable (x1 x2 y1 y2 z1 z2 : ℝ)
variable (h0 : 0 < x1)
variable (h1 : 0 < x2)
variable (h2 : x1 * y1 > z1^2)
variable (h3 : x2 * y2 > z2^2)

theorem inequality_proof :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
sorry

end inequality_proof_l198_198993


namespace find_DF_l198_198355

noncomputable def triangle (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def median (a b : ℝ) : ℝ := a / 2

theorem find_DF {DE EF DM DF : ℝ} (h1 : DE = 7) (h2 : EF = 10) (h3 : DM = 5) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l198_198355


namespace program_output_is_1023_l198_198601

-- Definition placeholder for program output.
def program_output : ℕ := 1023

-- Theorem stating the program's output.
theorem program_output_is_1023 : program_output = 1023 := 
by 
  -- Proof details are omitted.
  sorry

end program_output_is_1023_l198_198601


namespace ellipse_equation_is_standard_form_l198_198921

theorem ellipse_equation_is_standard_form (m n : ℝ) (h_m_pos : m > 0) (h_n_pos : n > 0) (h_mn_neq : m ≠ n) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ (∀ x y : ℝ, mx^2 + ny^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end ellipse_equation_is_standard_form_l198_198921


namespace minimum_route_length_l198_198576

/-- 
Given a city with the shape of a 5 × 5 square grid,
prove that the minimum length of a route that covers each street exactly once and 
returns to the starting point is 68, considering each street can be walked any number of times. 
-/
theorem minimum_route_length (n : ℕ) (h1 : n = 5) : 
  ∃ route_length : ℕ, route_length = 68 := 
sorry

end minimum_route_length_l198_198576


namespace find_product_xyz_l198_198051

-- Definitions for the given conditions
variables (x y z : ℕ) -- positive integers

-- Conditions
def condition1 : Prop := x + 2 * y = z
def condition2 : Prop := x^2 - 4 * y^2 + z^2 = 310

-- Theorem statement
theorem find_product_xyz (h1 : condition1 x y z) (h2 : condition2 x y z) : 
  x * y * z = 11935 ∨ x * y * z = 2015 :=
sorry

end find_product_xyz_l198_198051


namespace cubic_sum_l198_198671

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 := by
  -- Using the provided conditions x + y = 5 and x^2 + y^2 = 13
  sorry

end cubic_sum_l198_198671


namespace snake_price_correct_l198_198178

-- Define the conditions
def num_snakes : ℕ := 3
def eggs_per_snake : ℕ := 2
def total_eggs : ℕ := num_snakes * eggs_per_snake
def super_rare_multiple : ℕ := 4
def total_revenue : ℕ := 2250

-- The question: How much does each regular baby snake sell for?
def price_of_regular_baby_snake := 250

-- The proof statement
theorem snake_price_correct
  (x : ℕ)
  (h1 : total_eggs = 6)
  (h2 : 5 * x + super_rare_multiple * x = total_revenue)
  :
  x = price_of_regular_baby_snake := 
sorry

end snake_price_correct_l198_198178


namespace little_ma_probability_l198_198871

theorem little_ma_probability :
  let options : List Char := ['A', 'B', 'C', 'D']
  let total_outcomes := options.length
  let favorable_outcomes := 1
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 1 / 4 :=
by
  sorry

end little_ma_probability_l198_198871


namespace remainder_of_large_number_l198_198477

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l198_198477


namespace car_distribution_l198_198624

theorem car_distribution :
  ∀ (total_cars cars_first cars_second cars_left : ℕ),
    total_cars = 5650000 →
    cars_first = 1000000 →
    cars_second = cars_first + 500000 →
    cars_left = total_cars - (cars_first + cars_second + (cars_first + cars_second)) →
    ∃ (cars_fourth_fifth : ℕ), cars_fourth_fifth = cars_left / 2 ∧ cars_fourth_fifth = 325000 :=
begin
  intros total_cars cars_first cars_second cars_left H_total H_first H_second H_left,
  use (cars_left / 2),
  split,
  { refl, },
  { rw [H_total, H_first, H_second, H_left],
    norm_num, },
end

end car_distribution_l198_198624


namespace expected_squares_attacked_by_rooks_l198_198761

theorem expected_squares_attacked_by_rooks : 
  let p := (64 - (49/64)^3) in
  64 * p = 35.33 := by
    sorry

end expected_squares_attacked_by_rooks_l198_198761


namespace monotonic_decreasing_interval_l198_198752

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.cos (2 * x + Real.pi / 4))

theorem monotonic_decreasing_interval :
  ∀ (x1 x2 : ℝ), (-Real.pi / 8) < x1 ∧ x1 < Real.pi / 8 ∧ (-Real.pi / 8) < x2 ∧ x2 < Real.pi / 8 ∧ x1 < x2 →
  f x1 > f x2 :=
sorry

end monotonic_decreasing_interval_l198_198752


namespace secret_reaches_2186_students_on_seventh_day_l198_198620

/-- 
Alice tells a secret to three friends on Sunday. The next day, each of those friends tells the secret to three new friends.
Each time a person hears the secret, they tell three other new friends the following day.
On what day will 2186 students know the secret?
-/
theorem secret_reaches_2186_students_on_seventh_day :
  ∃ (n : ℕ), 1 + 3 * ((3^n - 1)/2) = 2186 ∧ n = 7 :=
by
  sorry

end secret_reaches_2186_students_on_seventh_day_l198_198620


namespace find_other_tax_l198_198564

/-- Jill's expenditure breakdown and total tax conditions. -/
def JillExpenditure 
  (total : ℝ)
  (clothingPercent : ℝ)
  (foodPercent : ℝ)
  (otherPercent : ℝ)
  (clothingTaxPercent : ℝ)
  (foodTaxPercent : ℝ)
  (otherTaxPercent : ℝ)
  (totalTaxPercent : ℝ) :=
  (clothingPercent + foodPercent + otherPercent = 100) ∧
  (clothingTaxPercent = 4) ∧
  (foodTaxPercent = 0) ∧
  (totalTaxPercent = 5.2) ∧
  (total > 0)

/-- The goal is to find the tax percentage on other items which Jill paid, given the constraints. -/
theorem find_other_tax
  {total clothingAmt foodAmt otherAmt clothingTax foodTax otherTaxPercent totalTax : ℝ}
  (h_exp : JillExpenditure total 50 10 40 clothingTax foodTax otherTaxPercent totalTax) :
  otherTaxPercent = 8 :=
by
  sorry

end find_other_tax_l198_198564


namespace largest_initial_number_l198_198877

theorem largest_initial_number :
  ∃ n : ℕ, (∀ k : ℕ, (n % k ≠ 0 → k ∈ {2, 2, 2, 2, 3}) ∧ (n + 11 = 200)) ∧ (n = 189) :=
begin
  sorry -- Proof not required per instruction
end

end largest_initial_number_l198_198877


namespace temperature_on_Saturday_l198_198935

theorem temperature_on_Saturday 
  (avg_temp : ℕ)
  (sun_temp : ℕ) 
  (mon_temp : ℕ) 
  (tue_temp : ℕ) 
  (wed_temp : ℕ) 
  (thu_temp : ℕ) 
  (fri_temp : ℕ)
  (saturday_temp : ℕ)
  (h_avg : avg_temp = 53)
  (h_sun : sun_temp = 40)
  (h_mon : mon_temp = 50) 
  (h_tue : tue_temp = 65) 
  (h_wed : wed_temp = 36) 
  (h_thu : thu_temp = 82) 
  (h_fri : fri_temp = 72) 
  (h_week : 7 * avg_temp = sun_temp + mon_temp + tue_temp + wed_temp + thu_temp + fri_temp + saturday_temp) :
  saturday_temp = 26 := 
by
  sorry

end temperature_on_Saturday_l198_198935


namespace remainder_123456789012_mod_252_l198_198494

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l198_198494


namespace cube_sum_l198_198676

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cube_sum_l198_198676


namespace remainder_when_divided_l198_198492

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end remainder_when_divided_l198_198492


namespace train_speed_late_l198_198774

theorem train_speed_late (v : ℝ) 
  (h1 : ∀ (d : ℝ) (s : ℝ), d = 15 ∧ s = 100 → d / s = 0.15) 
  (h2 : ∀ (t1 t2 : ℝ), t1 = 0.15 ∧ t2 = 0.4 → t2 = t1 + 0.25)
  (h3 : ∀ (d : ℝ) (t : ℝ), d = 15 ∧ t = 0.4 → v = d / t) : 
  v = 37.5 := sorry

end train_speed_late_l198_198774


namespace negation_correct_l198_198753

-- Define the initial proposition
def initial_proposition : Prop :=
  ∃ x : ℝ, x < 0 ∧ x^2 - 2 * x > 0

-- Define the negation of the initial proposition
def negated_proposition : Prop :=
  ∀ x : ℝ, x < 0 → x^2 - 2 * x ≤ 0

-- Statement of the theorem
theorem negation_correct :
  (¬ initial_proposition) = negated_proposition :=
by
  sorry

end negation_correct_l198_198753


namespace angle_CBD_is_10_degrees_l198_198891

theorem angle_CBD_is_10_degrees (angle_ABC angle_ABD : ℝ) (h1 : angle_ABC = 40) (h2 : angle_ABD = 30) :
  angle_ABC - angle_ABD = 10 :=
by
  sorry

end angle_CBD_is_10_degrees_l198_198891


namespace solve_problem_statement_l198_198807

def problem_statement : Prop :=
  ∃ n, 3^19 % n = 7 ∧ n = 1162261460

theorem solve_problem_statement : problem_statement :=
  sorry

end solve_problem_statement_l198_198807


namespace aeroplane_distance_l198_198452

theorem aeroplane_distance
  (speed : ℝ) (time : ℝ) (distance : ℝ)
  (h1 : speed = 590)
  (h2 : time = 8)
  (h3 : distance = speed * time) :
  distance = 4720 :=
by {
  -- The proof will contain the steps to show that distance = 4720
  sorry
}

end aeroplane_distance_l198_198452


namespace tree_difference_l198_198798

-- Given constants
def Hassans_apple_trees : Nat := 1
def Hassans_orange_trees : Nat := 2

def Ahmeds_orange_trees : Nat := 8
def Ahmeds_apple_trees : Nat := 4 * Hassans_apple_trees

-- Total trees computations
def Ahmeds_total_trees : Nat := Ahmeds_apple_trees + Ahmeds_orange_trees
def Hassans_total_trees : Nat := Hassans_apple_trees + Hassans_orange_trees

-- Theorem to prove the difference in total trees
theorem tree_difference : Ahmeds_total_trees - Hassans_total_trees = 9 := by
  sorry

end tree_difference_l198_198798


namespace tan_double_angle_l198_198335

theorem tan_double_angle (α : ℝ) (h : Real.tan (Real.pi - α) = 2) : Real.tan (2 * α) = 4 / 3 := 
by 
  sorry

end tan_double_angle_l198_198335


namespace smallest_pos_integer_l198_198902

-- Definitions based on the given conditions
def arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
def sum_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

-- Given conditions
def condition1 (a1 d : ℤ) : Prop := arithmetic_seq a1 d 11 - arithmetic_seq a1 d 8 = 3
def condition2 (a1 d : ℤ) : Prop := sum_seq a1 d 11 - sum_seq a1 d 8 = 3

-- The claim we want to prove
theorem smallest_pos_integer 
  (n : ℕ) (a1 d : ℤ) 
  (h1 : condition1 a1 d) 
  (h2 : condition2 a1 d) : n = 10 :=
by
  sorry

end smallest_pos_integer_l198_198902


namespace train_length_l198_198447

theorem train_length (V L : ℝ) (h₁ : V = L / 18) (h₂ : V = (L + 200) / 30) : L = 300 :=
by
  sorry

end train_length_l198_198447


namespace verify_min_n_for_coprime_subset_l198_198987

def is_pairwise_coprime (s : Finset ℕ) : Prop :=
  ∀ (a b : ℕ) (ha : a ∈ s) (hb : b ∈ s), a ≠ b → Nat.gcd a b = 1

def contains_4_pairwise_coprime (s : Finset ℕ) : Prop :=
  ∃ t : Finset ℕ, t ⊆ s ∧ t.card = 4 ∧ is_pairwise_coprime t

def min_n_for_coprime_subset : ℕ :=
  111

theorem verify_min_n_for_coprime_subset (S : Finset ℕ) (hS : S = Finset.range 151) :
  ∀ (n : ℕ), (∀ s : Finset ℕ, s ⊆ S ∧ s.card = n → contains_4_pairwise_coprime s) ↔ (n ≥ min_n_for_coprime_subset) :=
sorry

end verify_min_n_for_coprime_subset_l198_198987


namespace monthly_energy_consumption_l198_198551

-- Defining the given conditions
def power_fan_kW : ℝ := 0.075 -- kilowatts
def hours_per_day : ℝ := 8 -- hours per day
def days_per_month : ℝ := 30 -- days per month

-- The math proof statement with conditions and the expected answer
theorem monthly_energy_consumption : (power_fan_kW * hours_per_day * days_per_month) = 18 :=
by
  -- Placeholder for proof
  sorry

end monthly_energy_consumption_l198_198551


namespace inequality_problem_l198_198377

theorem inequality_problem (x : ℝ) (h_denom : 2 * x^2 + 2 * x + 1 ≠ 0) : 
  -4 ≤ (x^2 - 2*x - 3)/(2*x^2 + 2*x + 1) ∧ (x^2 - 2*x - 3)/(2*x^2 + 2*x + 1) ≤ 1 :=
sorry

end inequality_problem_l198_198377


namespace fraction_to_decimal_l198_198295

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198295


namespace find_length_of_train_l198_198967

def speed_kmh : Real := 60
def time_to_cross_bridge : Real := 26.997840172786177
def length_of_bridge : Real := 340

noncomputable def speed_ms : Real := speed_kmh * (1000 / 3600)
noncomputable def total_distance : Real := speed_ms * time_to_cross_bridge
noncomputable def length_of_train : Real := total_distance - length_of_bridge

theorem find_length_of_train :
  length_of_train = 109.9640028797695 := 
sorry

end find_length_of_train_l198_198967


namespace remainder_123456789012_div_252_l198_198486

/-- The remainder when $123456789012$ is divided by $252$ is $228$ --/
theorem remainder_123456789012_div_252 : 
    let M := 123456789012 in
    M % 4 = 0 ∧ 
    M % 9 = 3 ∧ 
    M % 7 = 6 → 
    M % 252 = 228 := 
by
    intros M h_mod4 h_mod9 h_mod7
    sorry

end remainder_123456789012_div_252_l198_198486


namespace find_a_for_unique_solution_l198_198650

theorem find_a_for_unique_solution :
  ∃ a : ℝ, (∀ x : ℝ, 0 ≤ x^2 - a * x + a ∧ x^2 - a * x + a ≤ 1) ↔ a = 2 :=
by
  sorry

end find_a_for_unique_solution_l198_198650


namespace ellipse_properties_slope_range_point_P_on_fixed_line_l198_198136

-- Conditions from the problem statement
variables {a b c : ℝ}
variables {λ : ℝ}

-- Ellipse equation and conditions
def ellipse_eq (x y : ℝ) := (x^2) / a^2 + (y^2) / b^2 = 1
axiom a_gt_b : a > b
axiom b_gt_0 : b > 0
axiom a2_eq_2 : a^2 = 2
axiom b2_eq_1 : b^2 = 1

-- Semi-focal length
def semi_focal_length := c = Real.sqrt (a^2 - b^2)

-- Line equation and point conditions
def line_eq (x : ℝ) := x = -a^2 / c
def N := (- (a^2 / c), 0)
def F1 := (-c, 0)
def F2 := (c, 0)
axiom F1F2_length : (2 * c = 2)
axiom F1F2_eq_2NF1 : (2 * N.1 = 2 * F1.1)

-- Collinearity condition
axiom collinear_NAB : ∀ {A B : ℝ × ℝ}, A.2 > 0 ∧ B.2 > 0 → λ ∈ [1/5, 1/3] → (N.1 = λ * N.1)

-- Proof goals
theorem ellipse_properties :
  ellipse_eq = λ (x y : ℝ), (x^2) / 2 + (y^2) / 1 = 1 :=
by sorry

theorem slope_range :
  ∀ (k : ℝ), (0 < k ∧ k < Real.sqrt(2) / 2) → (Real.sqrt(2) / 6 ≤ k ∧ k ≤ 1/2) :=
by sorry

theorem point_P_on_fixed_line :
  ∀ (P : ℝ × ℝ), P.2 = 0 → P.1 = -1 :=
by sorry

end ellipse_properties_slope_range_point_P_on_fixed_line_l198_198136


namespace line_intersection_range_b_l198_198162

theorem line_intersection_range_b 
  (b : ℝ) 
  (line : ℝ → ℝ := λ x, x + b)
  (curve : ℝ × ℝ → Prop := λ p, (p.1 - 2)^2 + (p.2 - 3)^2 = 4)
  (cond1 : ∃ x, 0 ≤ x ∧ x ≤ 4 ∧ ∃ y, 1 ≤ y ∧ y ≤ 3 ∧ line x = y)
  (cond2 : ∃ x y, curve (x, y) ∧ line x = y) : 
  b ∈ (Icc (1 - 2 * real.sqrt 2) 3) :=
sorry

end line_intersection_range_b_l198_198162


namespace find_angle_complement_supplement_l198_198045

theorem find_angle_complement_supplement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end find_angle_complement_supplement_l198_198045


namespace maximize_xyz_l198_198557

theorem maximize_xyz (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 60) :
    (x, y, z) = (20, 40 / 3, 80 / 3) → x^3 * y^2 * z^4 ≤ 20^3 * (40 / 3)^2 * (80 / 3)^4 :=
by
  sorry

end maximize_xyz_l198_198557


namespace perimeter_equal_l198_198579

theorem perimeter_equal (x : ℕ) (hx : x = 4)
    (side_square : ℕ := x + 2) 
    (side_triangle : ℕ := 2 * x) 
    (perimeter_square : ℕ := 4 * side_square)
    (perimeter_triangle : ℕ := 3 * side_triangle) :
    perimeter_square = perimeter_triangle :=
by
    -- Given x = 4
    -- Calculate side lengths
    -- side_square = x + 2 = 4 + 2 = 6
    -- side_triangle = 2 * x = 2 * 4 = 8
    -- Calculate perimeters
    -- perimeter_square = 4 * side_square = 4 * 6 = 24
    -- perimeter_triangle = 3 * side_triangle = 3 * 8 = 24
    -- Therefore, perimeter_square = perimeter_triangle = 24
    sorry

end perimeter_equal_l198_198579


namespace more_trees_in_ahmeds_orchard_l198_198800

-- Given conditions
def ahmed_orange_trees : ℕ := 8
def hassan_apple_trees : ℕ := 1
def hassan_orange_trees : ℕ := 2
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees
def ahmed_total_trees : ℕ := ahmed_orange_trees + ahmed_apple_trees
def hassan_total_trees : ℕ := hassan_apple_trees + hassan_orange_trees

-- Statement to be proven
theorem more_trees_in_ahmeds_orchard : ahmed_total_trees - hassan_total_trees = 9 :=
by
  sorry

end more_trees_in_ahmeds_orchard_l198_198800


namespace compute_expression_l198_198974

theorem compute_expression :
  120 * 2400 - 20 * 2400 - 100 * 2400 = 0 :=
sorry

end compute_expression_l198_198974


namespace pig_count_correct_l198_198758

def initial_pigs : ℝ := 64.0
def additional_pigs : ℝ := 86.0
def total_pigs : ℝ := 150.0

theorem pig_count_correct : initial_pigs + additional_pigs = total_pigs := by
  show 64.0 + 86.0 = 150.0
  sorry

end pig_count_correct_l198_198758


namespace triangle_perimeter_l198_198449

theorem triangle_perimeter {a b c : ℕ} (ha : a = 10) (hb : b = 6) (hc : c = 7) :
    a + b + c = 23 := by
  sorry

end triangle_perimeter_l198_198449


namespace sum_of_infinite_series_eq_four_l198_198259

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 1 ∨ n = 2 then 1 else (1/3) * sequence (n - 1) + (1/4) * sequence (n - 2)

noncomputable def sum_of_sequence : ℝ :=
  ∑' n, sequence n

theorem sum_of_infinite_series_eq_four : sum_of_sequence = 4 := 
  sorry

end sum_of_infinite_series_eq_four_l198_198259


namespace units_digit_6_l198_198997

theorem units_digit_6 (p : ℤ) (hp : 0 < p % 10) (h1 : (p^3 % 10) = (p^2 % 10)) (h2 : (p + 2) % 10 = 8) : p % 10 = 6 :=
by
  sorry

end units_digit_6_l198_198997


namespace find_smaller_number_l198_198585

theorem find_smaller_number (x y : ℤ) (h1 : x + y = 15) (h2 : 3 * x = 5 * y - 11) : x = 8 :=
by
  sorry

end find_smaller_number_l198_198585


namespace absolute_value_equation_solution_l198_198071

-- mathematical problem representation in Lean
theorem absolute_value_equation_solution (y : ℝ) (h : |y + 2| = |y - 3|) : y = 1 / 2 :=
sorry

end absolute_value_equation_solution_l198_198071


namespace hoopit_toes_l198_198909

theorem hoopit_toes (h : ℕ) : 
  (7 * (4 * h) + 8 * (2 * 5) = 164) -> h = 3 :=
by
  sorry

end hoopit_toes_l198_198909


namespace angles_satisfy_system_l198_198944

theorem angles_satisfy_system (k : ℤ) : 
  let x := Real.pi / 3 + k * Real.pi
  let y := k * Real.pi
  x - y = Real.pi / 3 ∧ Real.tan x - Real.tan y = Real.sqrt 3 := 
by 
  sorry

end angles_satisfy_system_l198_198944


namespace case_two_thirds_possible_case_three_fourths_impossible_case_seven_tenths_impossible_l198_198608

open Real

-- Definitions for students and questions
def m : ℕ := 3 -- number of questions
def n : ℕ := 3 -- number of students

-- Definitions for fractions
def two_thirds : ℝ := 2 / 3
def three_fourths : ℝ := 3 / 4
def seven_tenths : ℝ := 7 / 10

-- Each case can be represented as a separate theorem

-- Case 1: α = 2/3
theorem case_two_thirds_possible :
  ∃ (D : ℕ) (A : ℕ), 
  (D ≥ two_thirds * m) ∧
  (∀ (d : ℕ), d ∈ D → ∃ (s : ℕ), s ∈ n ∧ s cannot answer d) ∧
  (A ≥ two_thirds * n) ∧
  (∀ (a : ℕ), a ∈ A → a answers_at_least two_thirds * m) :=
  sorry

-- Case 2: α = 3/4
theorem case_three_fourths_impossible :
  ¬(
  ∃ (D : ℕ) (A : ℕ), 
  (D ≥ three_fourths * m) ∧
  (∀ (d : ℕ), d ∈ D → ∃ (s : ℕ), s ∈ n ∧ s cannot answer d) ∧
  (A ≥ three_fourths * n) ∧
  (∀ (a : ℕ), a ∈ A → a answers_at_least three_fourths * m)) :=
  sorry

-- Case 3: α = 7/10
theorem case_seven_tenths_impossible :
  ¬(
  ∃ (D : ℕ) (A : ℕ), 
  (D ≥ seven_tenths * m) ∧
  (∀ (d : ℕ), d ∈ D → ∃ (s : ℕ), s ∈ n ∧ s cannot answer d) ∧
  (A ≥ seven_tenths * n) ∧
  (∀ (a : ℕ), a ∈ A → a answers_at_least seven_tenths * m)) :=
  sorry

end case_two_thirds_possible_case_three_fourths_impossible_case_seven_tenths_impossible_l198_198608


namespace second_concert_attendance_l198_198196

theorem second_concert_attendance (n1 : ℕ) (h1 : n1 = 65899) (h2 : n2 = n1 + 119) : n2 = 66018 :=
by
  -- proof goes here
  sorry

end second_concert_attendance_l198_198196


namespace remainder_123456789012_div_252_l198_198481

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end remainder_123456789012_div_252_l198_198481


namespace minimum_voters_for_tall_win_l198_198548

-- Definitions based on the conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3
def majority_precinct_voters : ℕ := 2
def majority_precincts_per_district : ℕ := 5
def majority_districts : ℕ := 3
def tall_won : Prop := true

-- Problem statement
theorem minimum_voters_for_tall_win : 
  tall_won → (∃ n : ℕ, n = 3 * 5 * 2 ∧ n ≤ voters) :=
by
  sorry

end minimum_voters_for_tall_win_l198_198548


namespace cola_cost_l198_198189

theorem cola_cost (h c : ℝ) (h1 : 3 * h + 2 * c = 360) (h2 : 2 * h + 3 * c = 390) : c = 90 :=
by
  sorry

end cola_cost_l198_198189


namespace _l198_198469

example : 123456789012 % 252 = 24 :=
by
  have h4 : 123456789012 % 4 = 0 := sorry
  have h9 : 123456789012 % 9 = 3 := sorry
  have h7 : 123456789012 % 7 = 4 := sorry
  -- Applying the Chinese Remainder Theorem for these congruences
  exact (chinese_remainder_theorem h4 h9 h7).resolve_right $ by sorry

end _l198_198469


namespace derivative_of_f_domain_of_f_range_of_f_l198_198845

open Real

noncomputable def f (x : ℝ) := 1 / (x + sqrt (1 + 2 * x^2))

theorem derivative_of_f (x : ℝ) : 
  deriv f x = - ((sqrt (1 + 2 * x^2) + 2 * x) / (sqrt (1 + 2 * x^2) * (x + sqrt (1 + 2 * x^2))^2)) :=
by
  sorry

theorem domain_of_f : ∀ x : ℝ, f x ≠ 0 :=
by
  sorry

theorem range_of_f : 
  ∀ y : ℝ, 0 < y ∧ y ≤ sqrt 2 → ∃ x : ℝ, f x = y :=
by
  sorry

end derivative_of_f_domain_of_f_range_of_f_l198_198845


namespace number_of_baggies_l198_198197

/-- Conditions -/
def cookies_per_bag : ℕ := 9
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41

/-- Question: Prove the total number of baggies Olivia can make is 6 --/
theorem number_of_baggies : (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_bag = 6 := sorry

end number_of_baggies_l198_198197


namespace Mika_water_left_l198_198193

theorem Mika_water_left :
  let initial_amount := 7 / 2
  let used_amount := 9 / 4
  initial_amount - used_amount = 5 / 4 :=
by
  let initial_amount := 7 / 2
  let used_amount := 9 / 4
  show initial_amount - used_amount = 5 / 4
  sorry

end Mika_water_left_l198_198193


namespace sean_divided_by_julie_is_2_l198_198388

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum as twice the sum of the first 300 natural numbers
def sean_sum : ℕ := 2 * sum_natural 300

-- Define Julie's sum as the sum of the first 300 natural numbers
def julie_sum : ℕ := sum_natural 300

-- Prove that Sean's sum divided by Julie's sum is 2
theorem sean_divided_by_julie_is_2 : sean_sum / julie_sum = 2 := by
  sorry

end sean_divided_by_julie_is_2_l198_198388


namespace infinite_solutions_of_linear_eq_l198_198151

theorem infinite_solutions_of_linear_eq (a b : ℝ) : 
  (∃ b : ℝ, ∃ a : ℝ, 5 * a - 11 * b = 21) := sorry

end infinite_solutions_of_linear_eq_l198_198151


namespace functional_linear_solution_l198_198640

variable (f : ℝ → ℝ)

theorem functional_linear_solution (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) : 
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_linear_solution_l198_198640


namespace unique_k_value_l198_198859

noncomputable def findK (k : ℝ) : Prop :=
  ∃ (x : ℝ), (x^2 - k) * (x + k + 1) = x^3 + k * (x^2 - x - 4) ∧ k ≠ 0 ∧ k = -3

theorem unique_k_value : ∀ (k : ℝ), findK k :=
by
  intro k
  sorry

end unique_k_value_l198_198859


namespace smallest_number_divisible_conditions_l198_198436

theorem smallest_number_divisible_conditions :
  (∃ n : ℕ, 
    n % 9 = 0 ∧ 
    n % 2 = 1 ∧ 
    n % 3 = 1 ∧ 
    n % 4 = 1 ∧ 
    n % 5 = 1 ∧ 
    n % 6 = 1 ∧ 
    n % 7 = 1 ∧ 
    n % 8 = 1 ∧ 
    n = 5041) :=
by { sorry }

end smallest_number_divisible_conditions_l198_198436


namespace positive_integer_divisibility_l198_198328

theorem positive_integer_divisibility :
  ∀ n : ℕ, 0 < n → (5^(n-1) + 3^(n-1) ∣ 5^n + 3^n) → n = 1 :=
by
  sorry

end positive_integer_divisibility_l198_198328


namespace Alan_collected_48_shells_l198_198734

def Laurie_shells : ℕ := 36
def Ben_shells : ℕ := Laurie_shells / 3
def Alan_shells : ℕ := 4 * Ben_shells

theorem Alan_collected_48_shells :
  Alan_shells = 48 :=
by
  sorry

end Alan_collected_48_shells_l198_198734


namespace train_speed_l198_198791

theorem train_speed (lt_train : ℝ) (lt_bridge : ℝ) (time_cross : ℝ) (total_speed_kmph : ℝ) :
  lt_train = 150 ∧ lt_bridge = 225 ∧ time_cross = 30 ∧ total_speed_kmph = (375 / 30) * 3.6 → 
  total_speed_kmph = 45 := 
by
  sorry

end train_speed_l198_198791


namespace unique_solutions_l198_198644

noncomputable def func_solution (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, (x - 2) * f y + f (y + 2 * f x) = f (x + y * f x)

theorem unique_solutions (f : ℝ → ℝ) :
  func_solution f → (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x - 1) :=
by
  intro h
  sorry

end unique_solutions_l198_198644


namespace defect_free_product_probability_is_correct_l198_198406

noncomputable def defect_free_probability : ℝ :=
  let p1 := 0.2
  let p2 := 0.3
  let p3 := 0.5
  let d1 := 0.95
  let d2 := 0.90
  let d3 := 0.80
  p1 * d1 + p2 * d2 + p3 * d3

theorem defect_free_product_probability_is_correct :
  defect_free_probability = 0.86 :=
by
  sorry

end defect_free_product_probability_is_correct_l198_198406


namespace intersection_point_of_lines_l198_198321

theorem intersection_point_of_lines :
  ∃ x y : ℝ, 3 * x + 4 * y - 2 = 0 ∧ 2 * x + y + 2 = 0 := 
by
  sorry

end intersection_point_of_lines_l198_198321


namespace sales_difference_l198_198771
noncomputable def max_min_difference (sales : List ℕ) : ℕ :=
  (sales.maximum.getD 0) - (sales.minimum.getD 0)

theorem sales_difference :
  max_min_difference [1200, 1450, 1950, 1700] = 750 :=
by
  sorry

end sales_difference_l198_198771


namespace odd_expression_l198_198741

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

theorem odd_expression (p q : ℕ) (hp : is_odd p) (hq : is_odd q) : is_odd (2 * p * p - q) :=
by
  sorry

end odd_expression_l198_198741


namespace greatest_value_q_minus_r_l198_198227

theorem greatest_value_q_minus_r : ∃ q r : ℕ, 1043 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ (q - r = 37) :=
by {
  sorry
}

end greatest_value_q_minus_r_l198_198227


namespace chair_subsets_with_at_least_three_adjacent_l198_198422

-- Define a structure for representing a circle of chairs
def Circle (n : ℕ) := { k : ℕ // k < n }

-- Condition: We have 12 chairs in a circle
def chairs : finset (Circle 12) :=
  finset.univ

-- Defining adjacency in a circle
def adjacent (c : Circle 12) : finset (Circle 12) :=
  if h : c.val = 11 then
    finset.of_list [⟨0, by linarith⟩, ⟨10, by linarith⟩, ⟨11, by linarith⟩]
  else if h : c.val = 0 then
    finset.of_list [⟨11, by linarith⟩, ⟨0, by linarith⟩, ⟨1, by linarith⟩]
  else
    finset.of_list [⟨c.val - 1, by linarith⟩, ⟨c.val, by linarith⟩, ⟨c.val + 1, by linarith⟩]

-- Define a subset of chairs as "valid" if it contains at least 3 adjacent chairs
def has_at_least_three_adjacent (chair_set : finset (Circle 12)) : Prop :=
  ∃ c, (adjacent c) ⊆ chair_set

-- The theorem to state that the number of "valid" subsets is 2040.
theorem chair_subsets_with_at_least_three_adjacent : 
  finset.card {chair_set : finset (Circle 12) // has_at_least_three_adjacent chair_set} = 2040 := 
by sorry

end chair_subsets_with_at_least_three_adjacent_l198_198422


namespace solve_textbook_by_12th_l198_198691

/-!
# Problem Statement
There are 91 problems in a textbook. Yura started solving them on September 6 and solves one problem less each subsequent morning. By the evening of September 8, there are 46 problems left to solve.

We need to prove that Yura finishes solving all the problems by September 12.
-/

def initial_problems : ℕ := 91
def problems_left_by_evening_of_8th : ℕ := 46

def problems_solved_by_evening_of_8th : ℕ :=
  initial_problems - problems_left_by_evening_of_8th

def z : ℕ := (problems_solved_by_evening_of_8th / 3)

theorem solve_textbook_by_12th 
    (total_problems : ℕ)
    (problems_left : ℕ)
    (solved_by_evening_8th : ℕ)
    (daily_problem_count : ℕ) :
    (total_problems = initial_problems) →
    (problems_left = problems_left_by_evening_of_8th) →
    (solved_by_evening_8th = problems_solved_by_evening_of_8th) →
    (daily_problem_count = z) →
    ∃ (finishing_date : ℕ), finishing_date = 12 :=
  by
    intros _ _ _ _
    sorry

end solve_textbook_by_12th_l198_198691


namespace no_x0_leq_zero_implies_m_gt_1_l198_198161

theorem no_x0_leq_zero_implies_m_gt_1 (m : ℝ) :
  (¬ ∃ x0 : ℝ, x0^2 + 2 * x0 + m ≤ 0) ↔ m > 1 :=
sorry

end no_x0_leq_zero_implies_m_gt_1_l198_198161


namespace minimize_f_l198_198653

noncomputable def f (x a : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

theorem minimize_f (a : ℝ) : a = 82 / 43 :=
by
  sorry

end minimize_f_l198_198653


namespace license_plates_count_l198_198005

theorem license_plates_count :
  let letters := 26
  let digits := 10
  let odd_digits := 5
  let even_digits := 5
  (letters^3) * digits * (odd_digits + even_digits) = 878800 := by
  sorry

end license_plates_count_l198_198005


namespace cube_path_length_l198_198445

noncomputable def path_length_dot_cube : ℝ :=
  let edge_length := 2
  let radius1 := Real.sqrt 5
  let radius2 := 1
  (radius1 + radius2) * Real.pi

theorem cube_path_length :
  path_length_dot_cube = (Real.sqrt 5 + 1) * Real.pi :=
by
  sorry

end cube_path_length_l198_198445


namespace problem_eq_995_l198_198101

theorem problem_eq_995 :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) /
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400))
  = 995 := sorry

end problem_eq_995_l198_198101


namespace fraction_to_decimal_l198_198312

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l198_198312


namespace sage_reflection_day_l198_198373

theorem sage_reflection_day 
  (day_of_reflection_is_jan_1 : Prop)
  (equal_days_in_last_5_years : Prop)
  (new_year_10_years_ago_was_friday : Prop)
  (reflections_in_21st_century : Prop) : 
  ∃ (day : String), day = "Thursday" :=
by
  sorry

end sage_reflection_day_l198_198373


namespace find_num_adults_l198_198613

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

end find_num_adults_l198_198613


namespace smallest_whole_number_greater_than_sum_is_12_l198_198646

-- Definitions of the mixed numbers as improper fractions
def a : ℚ := 5 / 3
def b : ℚ := 9 / 4
def c : ℚ := 27 / 8
def d : ℚ := 25 / 6

-- The target sum and the required proof statement
theorem smallest_whole_number_greater_than_sum_is_12 : 
  let sum := a + b + c + d
  let smallest_whole_number_greater_than_sum := Nat.ceil sum
  smallest_whole_number_greater_than_sum = 12 :=
by 
  sorry

end smallest_whole_number_greater_than_sum_is_12_l198_198646


namespace digits_in_2_pow_120_l198_198824

theorem digits_in_2_pow_120 {a b : ℕ} (h : 10^a ≤ 2^200 ∧ 2^200 < 10^b) (ha : a = 60) (hb : b = 61) : 
  ∃ n : ℕ, 10^(n-1) ≤ 2^120 ∧ 2^120 < 10^n ∧ n = 37 :=
by {
  sorry
}

end digits_in_2_pow_120_l198_198824


namespace quadratic_real_roots_l198_198505

theorem quadratic_real_roots (m : ℝ) : 
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  (discriminant ≥ 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  sorry

end quadratic_real_roots_l198_198505


namespace rook_attack_expectation_correct_l198_198763

open ProbabilityTheory

noncomputable def rook_attack_expectation : ℝ := sorry

theorem rook_attack_expectation_correct :
  rook_attack_expectation = 35.33 := sorry

end rook_attack_expectation_correct_l198_198763


namespace intersecting_parabolas_circle_radius_sq_l198_198050

theorem intersecting_parabolas_circle_radius_sq:
  (∀ (x y : ℝ), (y = (x + 1)^2 ∧ x + 4 = (y - 3)^2) → 
  ((x + 1/2)^2 + (y - 7/2)^2 = 13/2)) := sorry

end intersecting_parabolas_circle_radius_sq_l198_198050


namespace describe_shape_cylinder_l198_198652

-- Define cylindrical coordinates
structure CylindricalCoordinates where
  r : ℝ -- radial distance
  θ : ℝ -- azimuthal angle
  z : ℝ -- height

-- Define the positive constant c
variable (c : ℝ) (hc : 0 < c)

-- The theorem statement
theorem describe_shape_cylinder (p : CylindricalCoordinates) (h : p.r = c) : 
  ∃ (p : CylindricalCoordinates), p.r = c :=
by
  sorry

end describe_shape_cylinder_l198_198652


namespace remainder_123456789012_mod_252_l198_198498

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l198_198498


namespace problem_inequality_A_problem_inequality_B_problem_inequality_D_problem_inequality_E_l198_198916

variable {a b c : ℝ}

theorem problem_inequality_A (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a * b < b * c :=
by sorry

theorem problem_inequality_B (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a * c < b * c :=
by sorry

theorem problem_inequality_D (h1 : a > 0) (h2 : a < b) (h3 : b < c) : a + b < b + c :=
by sorry

theorem problem_inequality_E (h1 : a > 0) (h2 : a < b) (h3 : b < c) : c / a > 1 :=
by sorry

end problem_inequality_A_problem_inequality_B_problem_inequality_D_problem_inequality_E_l198_198916


namespace range_of_a_l198_198683

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x : ℝ, f x = a*x^3 + Real.log x) :
  (∃ x : ℝ, x > 0 ∧ (deriv f x = 0)) → a < 0 :=
by
  sorry

end range_of_a_l198_198683


namespace total_cost_supplies_l198_198260

-- Definitions based on conditions
def cost_bow : ℕ := 5
def cost_vinegar : ℕ := 2
def cost_baking_soda : ℕ := 1
def cost_per_student : ℕ := cost_bow + cost_vinegar + cost_baking_soda
def number_of_students : ℕ := 23

-- Statement to be proven
theorem total_cost_supplies : cost_per_student * number_of_students = 184 := by
  sorry

end total_cost_supplies_l198_198260


namespace f_odd_and_increasing_l198_198847

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end f_odd_and_increasing_l198_198847


namespace rabbit_hid_carrots_l198_198827

theorem rabbit_hid_carrots (h_r h_f : ℕ) (x : ℕ)
  (rabbit_holes : 5 * h_r = x) 
  (fox_holes : 7 * h_f = x)
  (holes_relation : h_r = h_f + 6) :
  x = 105 :=
by
  sorry

end rabbit_hid_carrots_l198_198827


namespace least_pos_int_divisible_by_four_distinct_primes_l198_198063

theorem least_pos_int_divisible_by_four_distinct_primes : 
  ∃ n : ℕ, (∀ p : ℕ, nat.prime p → (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7) → p ∣ n) ∧ n = 210 :=
by
  sorry

end least_pos_int_divisible_by_four_distinct_primes_l198_198063


namespace unique_polynomial_solution_l198_198645

theorem unique_polynomial_solution (P : Polynomial ℝ) :
  (P.eval 0 = 0) ∧ (∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) ↔ (P = Polynomial.X) :=
by {
  sorry
}

end unique_polynomial_solution_l198_198645


namespace intersection_A_B_l198_198666

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := { x | ∃ k : ℤ, x = 3 * k - 1 }

theorem intersection_A_B :
  A ∩ B = {-1, 2} :=
by
  sorry

end intersection_A_B_l198_198666


namespace f_is_odd_l198_198108

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem f_is_odd : ∀ x, f (-x) = -f x := by
  sorry

end f_is_odd_l198_198108


namespace people_left_line_l198_198266

-- Definitions based on the conditions given in the problem
def initial_people := 7
def new_people := 8
def final_people := 11

-- Proof statement
theorem people_left_line (L : ℕ) (h : 7 - L + 8 = 11) : L = 4 :=
by
  -- Adding the proof steps directly skips to the required proof
  sorry

end people_left_line_l198_198266


namespace root_integer_l198_198517

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

def is_root (x_0 : ℝ) : Prop := f x_0 = 0

theorem root_integer (x_0 : ℝ) (h : is_root x_0) : Int.floor x_0 = 2 := by
  sorry

end root_integer_l198_198517


namespace yura_finish_date_l198_198705

theorem yura_finish_date :
  (∃ z : ℕ, 3 * z = 45 ∧ (∑ i in list.range (z + 2) \ 0, (z + 1 - i)) = 91 ∧ nat.find (λ n, (∑ i in list.range (n+1), (z + 1 - i)) >= 91) + 6 = 12) :=
by
  sorry

end yura_finish_date_l198_198705


namespace max_initial_number_l198_198879

theorem max_initial_number (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    200 = n + a + b + c + d + e ∧ 
    ¬ (n % a = 0) ∧ 
    ¬ ((n + a) % b = 0) ∧ 
    ¬ ((n + a + b) % c = 0) ∧ 
    ¬ ((n + a + b + c) % d = 0) ∧ 
    ¬ ((n + a + b + c + d) % e = 0)) → 
  n ≤ 189 := 
sorry

end max_initial_number_l198_198879


namespace min_side_value_l198_198146

-- Definitions based on the conditions provided
variables (a b c : ℕ) (h1 : a - b = 5) (h2 : (a + b + c) % 2 = 0)

theorem min_side_value (h1 : a - b = 5) (h2 : (a + b + c) % 2 = 0) : c ≥ 7 :=
sorry

end min_side_value_l198_198146


namespace service_charge_percentage_is_correct_l198_198090

-- Define the conditions
def orderAmount : ℝ := 450
def totalAmountPaid : ℝ := 468
def serviceCharge : ℝ := totalAmountPaid - orderAmount

-- Define the target percentage
def expectedServiceChargePercentage : ℝ := 4.0

-- Proof statement: the service charge percentage is expectedServiceChargePercentage
theorem service_charge_percentage_is_correct : 
  (serviceCharge / orderAmount) * 100 = expectedServiceChargePercentage :=
by
  sorry

end service_charge_percentage_is_correct_l198_198090


namespace math_problem_l198_198271

theorem math_problem :
  (-1 : ℝ)^(53) + 3^(2^3 + 5^2 - 7^2) = -1 + (1 / 3^(16)) :=
by
  sorry

end math_problem_l198_198271


namespace tiles_difference_l198_198199

-- Definitions based on given conditions
def initial_blue_tiles : Nat := 20
def initial_green_tiles : Nat := 10
def first_border_green_tiles : Nat := 24
def second_border_green_tiles : Nat := 36

-- Problem statement
theorem tiles_difference :
  initial_green_tiles + first_border_green_tiles + second_border_green_tiles - initial_blue_tiles = 50 :=
by
  sorry

end tiles_difference_l198_198199


namespace symmetric_points_x_axis_l198_198860

theorem symmetric_points_x_axis (m n : ℤ) (h1 : m + 1 = 1) (h2 : 3 = -(n - 2)) : m - n = 1 :=
by
  sorry

end symmetric_points_x_axis_l198_198860


namespace value_of_x_in_interval_l198_198936

theorem value_of_x_in_interval :
  (let x := 1 / Real.logb 2 3 + 1
  in x > 1 ∧ x < 2) := by
  sorry

end value_of_x_in_interval_l198_198936


namespace fence_perimeter_l198_198592

noncomputable def posts (n : ℕ) := 36
noncomputable def space_between_posts (d : ℕ) := 6
noncomputable def length_is_twice_width (l w : ℕ) := l = 2 * w

theorem fence_perimeter (n d w l perimeter : ℕ)
  (h1 : posts n = 36)
  (h2 : space_between_posts d = 6)
  (h3 : length_is_twice_width l w)
  : perimeter = 216 :=
sorry

end fence_perimeter_l198_198592


namespace hexagon_area_l198_198221

theorem hexagon_area (s : ℝ) (hex_area : ℝ) (p q : ℤ) :
  s = 3 ∧ hex_area = (3 * Real.sqrt 3 / 2) * s^2 ∧ hex_area = Real.sqrt p + Real.sqrt q → p + q = 545 :=
by
  sorry

end hexagon_area_l198_198221


namespace min_value_l198_198842

theorem min_value (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (h_sum : x1 + x2 = 1) :
  ∃ m, (∀ x1 x2, x1 > 0 ∧ x2 > 0 ∧ x1 + x2 = 1 → (3 * x1 / x2 + 1 / (x1 * x2)) ≥ m) ∧ m = 6 :=
by
  sorry

end min_value_l198_198842


namespace find_Allyson_age_l198_198240

variable (Hiram Allyson : ℕ)

theorem find_Allyson_age (h : Hiram = 40)
  (condition : Hiram + 12 = 2 * Allyson - 4) : Allyson = 28 := by
  sorry

end find_Allyson_age_l198_198240


namespace correct_calculation_l198_198605

theorem correct_calculation (m n : ℤ) :
  (m^2 * m^3 ≠ m^6) ∧
  (- (m - n) = -m + n) ∧
  (m * (m + n) ≠ m^2 + n) ∧
  ((m + n)^2 ≠ m^2 + n^2) :=
by sorry

end correct_calculation_l198_198605


namespace product_in_base_7_l198_198611

def base_7_product : ℕ :=
  let b := 7
  Nat.ofDigits b [3, 5, 6] * Nat.ofDigits b [4]

theorem product_in_base_7 :
  base_7_product = Nat.ofDigits 7 [3, 2, 3, 1, 2] :=
by
  -- The proof is formally skipped for this exercise, hence we insert 'sorry'.
  sorry

end product_in_base_7_l198_198611


namespace age_problem_l198_198089

theorem age_problem (S F : ℕ) (h1 : F = S + 27) (h2 : F + 2 = 2 * (S + 2)) :
  S = 25 := by
  sorry

end age_problem_l198_198089


namespace fraction_to_decimal_l198_198303

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 :=
by sorry

end fraction_to_decimal_l198_198303


namespace tank_fraction_after_adding_water_l198_198527

theorem tank_fraction_after_adding_water 
  (initial_fraction : ℚ) 
  (full_capacity : ℚ) 
  (added_water : ℚ) 
  (final_fraction : ℚ) 
  (h1 : initial_fraction = 3/4) 
  (h2 : full_capacity = 56) 
  (h3 : added_water = 7) 
  (h4 : final_fraction = (initial_fraction * full_capacity + added_water) / full_capacity) : 
  final_fraction = 7 / 8 := 
by 
  sorry

end tank_fraction_after_adding_water_l198_198527


namespace probability_of_both_l198_198949

variable (A B : Prop)

-- Assumptions
def p_A : ℝ := 0.55
def p_B : ℝ := 0.60

-- Probability of both A and B telling the truth at the same time
theorem probability_of_both : p_A * p_B = 0.33 := by
  sorry

end probability_of_both_l198_198949


namespace simplified_expression_l198_198599

noncomputable def simplify_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ :=
  (x⁻¹ - z⁻¹)⁻¹

theorem simplified_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : 
  simplify_expression x z hx hz = x * z / (z - x) := 
by
  sorry

end simplified_expression_l198_198599


namespace deposits_exceed_10_on_second_Tuesday_l198_198903

noncomputable def deposits_exceed_10 (n : ℕ) : ℕ :=
2 * (2^n - 1)

theorem deposits_exceed_10_on_second_Tuesday :
  ∃ n, deposits_exceed_10 n > 1000 ∧ 1 + (n - 1) % 7 = 2 ∧ n < 21 :=
sorry

end deposits_exceed_10_on_second_Tuesday_l198_198903


namespace sum_of_arithmetic_sequence_l198_198836

noncomputable def arithmetic_sequence_sum (a_1 d : ℝ) (n : ℕ) : ℝ :=
n * a_1 + (n * (n - 1) / 2) * d

theorem sum_of_arithmetic_sequence (a_1 d : ℝ) (p q : ℕ) (h₁ : p ≠ q) (h₂ : arithmetic_sequence_sum a_1 d p = q) (h₃ : arithmetic_sequence_sum a_1 d q = p) : 
arithmetic_sequence_sum a_1 d (p + q) = - (p + q) := sorry

end sum_of_arithmetic_sequence_l198_198836


namespace total_number_of_workers_l198_198247

theorem total_number_of_workers (W N : ℕ) 
    (avg_all : ℝ) 
    (avg_technicians : ℝ) 
    (avg_non_technicians : ℝ)
    (h1 : avg_all = 8000)
    (h2 : avg_technicians = 20000)
    (h3 : avg_non_technicians = 6000)
    (h4 : 7 * avg_technicians + N * avg_non_technicians = (7 + N) * avg_all) :
  W = 49 := by
  sorry

end total_number_of_workers_l198_198247


namespace range_of_a_l198_198393

theorem range_of_a (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, y = f x) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, y = f (x - a) + f (x + a)) ↔ -1/2 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l198_198393


namespace hybrids_with_full_headlights_l198_198164

theorem hybrids_with_full_headlights (total_cars hybrids_percentage one_headlight_percentage : ℝ) 
  (hc : total_cars = 600) (hp : hybrids_percentage = 0.60) (ho : one_headlight_percentage = 0.40) : 
  total_cars * hybrids_percentage - total_cars * hybrids_percentage * one_headlight_percentage = 216 := by
  sorry

end hybrids_with_full_headlights_l198_198164


namespace angle_proof_l198_198667

noncomputable def angle_between_vectors (a b : euclidean_space ℝ (fin 2)) (h1 : ∥a∥ = 1) 
  (h2 : ∥b∥ = real.sqrt 3) (h3 : a + b = ![real.sqrt 3, 1]) : 
  real.angle := 
real.angle_of_cos (-1/2)

theorem angle_proof (a b : euclidean_space ℝ (fin 2)) 
    (h1 : ∥a∥ = 1) (h2 : ∥b∥ = real.sqrt 3) (h3 : a + b = ![real.sqrt 3, 1]) :
  let θ := angle_between_vectors a b h1 h2 h3 in
  θ = 2 * real.pi / 3 :=
begin
  sorry
end

end angle_proof_l198_198667


namespace Petya_workout_duration_l198_198031

theorem Petya_workout_duration :
  ∃ x : ℕ, (x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 135) ∧
            (x + 7 > x) ∧
            (x + 14 > x + 7) ∧
            (x + 21 > x + 14) ∧
            (x + 28 > x + 21) ∧
            x = 13 :=
by sorry

end Petya_workout_duration_l198_198031


namespace largest_initial_number_l198_198876

theorem largest_initial_number :
  ∃ n : ℕ, (∀ k : ℕ, (n % k ≠ 0 → k ∈ {2, 2, 2, 2, 3}) ∧ (n + 11 = 200)) ∧ (n = 189) :=
begin
  sorry -- Proof not required per instruction
end

end largest_initial_number_l198_198876


namespace problem_solution_l198_198516

variable (a b c : ℝ)

theorem problem_solution (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) :
  a + b ≤ 3 * c := 
sorry

end problem_solution_l198_198516


namespace cube_volume_in_pyramid_l198_198637

noncomputable def pyramid_base_side : ℝ := 2
noncomputable def equilateral_triangle_side : ℝ := 2 * Real.sqrt 2
noncomputable def equilateral_triangle_height : ℝ := Real.sqrt 6
noncomputable def cube_side : ℝ := Real.sqrt 6 / 2
noncomputable def cube_volume : ℝ := (Real.sqrt 6 / 2) ^ 3

theorem cube_volume_in_pyramid : cube_volume = 3 * Real.sqrt 6 / 4 :=
by
  sorry

end cube_volume_in_pyramid_l198_198637


namespace radius_of_larger_circle_l198_198756

theorem radius_of_larger_circle (r R AC BC AB : ℝ)
  (h1 : R = 4 * r)
  (h2 : AC = 8 * r)
  (h3 : BC^2 + AB^2 = AC^2)
  (h4 : AB = 16) :
  R = 32 :=
by
  sorry

end radius_of_larger_circle_l198_198756


namespace pyramid_base_side_length_l198_198575

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (side_length : ℝ)
  (h1 : area_lateral_face = 144)
  (h2 : slant_height = 24)
  (h3 : 144 = 0.5 * side_length * 24) : 
  side_length = 12 :=
by
  sorry

end pyramid_base_side_length_l198_198575


namespace car_distribution_l198_198627

theorem car_distribution :
  let total_cars := 5650000
  let first_supplier := 1000000
  let second_supplier := first_supplier + 500000
  let third_supplier := first_supplier + second_supplier
  let fourth_supplier := (total_cars - (first_supplier + second_supplier + third_supplier)) / 2
  let fifth_supplier := fourth_supplier
  fourth_supplier = 325000 ∧ fifth_supplier = 325000 := 
by 
  sorry

end car_distribution_l198_198627


namespace p_sufficient_not_necessary_q_l198_198132

theorem p_sufficient_not_necessary_q (x : ℝ) :
  (0 < x ∧ x < 2) → (-1 < x ∧ x < 3) :=
by
  intros h
  cases h with h1 h2
  split
  case left => linarith
  case right => linarith

end p_sufficient_not_necessary_q_l198_198132


namespace at_least_one_closed_l198_198438

theorem at_least_one_closed {T V : Set ℤ} (hT : T.Nonempty) (hV : V.Nonempty) (h_disjoint : ∀ x, x ∈ T → x ∉ V)
  (h_union : ∀ x, x ∈ T ∨ x ∈ V)
  (hT_closed : ∀ a b c, a ∈ T → b ∈ T → c ∈ T → a * b * c ∈ T)
  (hV_closed : ∀ x y z, x ∈ V → y ∈ V → z ∈ V → x * y * z ∈ V) :
  (∀ a b, a ∈ T → b ∈ T → a * b ∈ T) ∨ (∀ x y, x ∈ V → y ∈ V → x * y ∈ V) := sorry

end at_least_one_closed_l198_198438


namespace quadratic_real_roots_range_l198_198503

theorem quadratic_real_roots_range (m : ℝ) :
  ∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0 ↔ m ≤ 4 ∧ m ≠ 3 := 
by
  sorry

end quadratic_real_roots_range_l198_198503


namespace equation_of_l3_line_l1_through_fixed_point_existence_of_T_l198_198852

-- Question 1: The equation of the line \( l_{3} \)
theorem equation_of_l3 
  (F : ℝ × ℝ) 
  (H_focus : F = (2, 0))
  (k : ℝ) 
  (H_slope : k = 1) : 
  (∀ x y : ℝ, y = k * x + -2 ↔ y = x - 2) := 
sorry

-- Question 2: Line \( l_{1} \) passes through the fixed point (8, 0)
theorem line_l1_through_fixed_point 
  (k m1 : ℝ)
  (H_km1 : k * m1 ≠ 0)
  (H_m1lt : m1 < -t)
  (H_condition : ∃ x y : ℝ, y = k * x + m1 ∧ x^2 + (8/k) * x + (8 * m1 / k) = 0 ∧ ((x, y) = A1 ∨ (x, y) = B1))
  (H_dot_product : (x1 - 0)*(x2 - 0) + (y1 - 0)*(y2 - 0) = 0) : 
  ∀ P : ℝ × ℝ, P = (8, 0) := 
sorry

-- Question 3: Existence of point T such that S_i and d_i form geometric sequences
theorem existence_of_T
  (k : ℝ)
  (H_k : k = 1)
  (m1 m2 m3 : ℝ)
  (H_m_ordered : m1 < m2 ∧ m2 < m3 ∧ m3 < -t)
  (t : ℝ)
  (S1 S2 S3 d1 d2 d3 : ℝ)
  (H_S_geom_seq : S2^2 = S1 * S3)
  (H_d_geom_seq : d2^2 = d1 * d3)
  : ∃ t : ℝ, t = -2 :=
sorry

end equation_of_l3_line_l1_through_fixed_point_existence_of_T_l198_198852


namespace notebook_pen_ratio_l198_198961

theorem notebook_pen_ratio (pen_cost notebook_total_cost : ℝ) (num_notebooks : ℕ)
  (h1 : pen_cost = 1.50) (h2 : notebook_total_cost = 18) (h3 : num_notebooks = 4) :
  (notebook_total_cost / num_notebooks) / pen_cost = 3 :=
by
  -- The steps to prove this would go here
  sorry

end notebook_pen_ratio_l198_198961


namespace cannot_determine_if_counterfeit_coin_is_lighter_or_heavier_l198_198920

/-- 
Vasiliy has 2019 coins, one of which is counterfeit (differing in weight). 
Using balance scales without weights and immediately paying out identified genuine coins, 
it is impossible to determine whether the counterfeit coin is lighter or heavier.
-/
theorem cannot_determine_if_counterfeit_coin_is_lighter_or_heavier 
  (num_coins : ℕ)
  (num_counterfeit : ℕ)
  (balance_scale : Bool → Bool → Bool)
  (immediate_payment : Bool → Bool) :
  num_coins = 2019 →
  num_counterfeit = 1 →
  (∀ coins_w1 coins_w2, balance_scale coins_w1 coins_w2 = (coins_w1 = coins_w2)) →
  (∀ coin_p coin_q, (immediate_payment coin_p = true) → ¬ coin_p = coin_q) →
  ¬ ∃ (is_lighter_or_heavier : Bool), true :=
by
  intro h1 h2 h3 h4
  sorry

end cannot_determine_if_counterfeit_coin_is_lighter_or_heavier_l198_198920


namespace train_length_l198_198448

noncomputable def speed_km_hr : ℝ := 60
noncomputable def time_seconds : ℝ := 36
noncomputable def speed_m_s := speed_km_hr * (5/18 : ℝ)
noncomputable def distance := speed_m_s * time_seconds

-- Theorem statement
theorem train_length : distance = 600.12 := by
  sorry

end train_length_l198_198448


namespace evaluate_expression_l198_198815

theorem evaluate_expression : abs (abs (abs (-2 + 2) - 2) * 2) = 4 := 
by
  sorry

end evaluate_expression_l198_198815


namespace probability_of_event_l198_198531

theorem probability_of_event :
  let prob_three_kings := (4 / 52) * (3 / 51) * (2 / 50),
      prob_two_aces := ( (4 / 52) * (3 / 51) * (48 / 50) )
        + ( (4 / 52) * (48 / 51) * (3 / 50) )
        + ( (48 / 52) * (4 / 51) * (3 / 50) ),
      prob_three_aces := (4 / 52) * (3 / 51) * (2 / 50),
      prob_event := prob_three_kings + prob_two_aces + prob_three_aces
  in prob_event = (43 / 33150) :=
sorry

end probability_of_event_l198_198531


namespace area_of_circle_l198_198278

theorem area_of_circle (r θ : ℝ) (h : r = 4 * Real.cos θ - 3 * Real.sin θ) :
  ∃ π : ℝ, π * (5/2)^2 = 25 * π / 4 :=
by 
  sorry

end area_of_circle_l198_198278


namespace cost_of_one_book_l198_198907

theorem cost_of_one_book (x : ℝ) : 
  (9 * x = 11) ∧ (13 * x = 15) → x = 1.23 :=
by sorry

end cost_of_one_book_l198_198907


namespace Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction_l198_198395

noncomputable def Thabo_book_count_problem : Prop :=
  let P := Nat
  let F := Nat
  ∃ (P F : Nat), 
    -- Conditions
    (P > 40) ∧ 
    (F = 2 * P) ∧ 
    (F + P + 40 = 220) ∧ 
    -- Conclusion
    (P - 40 = 20)

theorem Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction : Thabo_book_count_problem :=
  sorry

end Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction_l198_198395


namespace number_square_of_digits_l198_198510

theorem number_square_of_digits (x y : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) (h3 : 0 ≤ y) (h4 : y ≤ 9) :
  ∃ n : ℕ, (∃ (k : ℕ), (1001 * x + 110 * y) = k^2) ↔ (x = 7 ∧ y = 4) :=
by
  sorry

end number_square_of_digits_l198_198510


namespace fraction_to_decimal_l198_198292

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l198_198292


namespace Frank_is_14_l198_198216

variable {d e f : ℕ}

theorem Frank_is_14
  (h1 : d + e + f = 30)
  (h2 : f - 5 = d)
  (h3 : e + 2 = 3 * (d + 2) / 4) :
  f = 14 :=
sorry

end Frank_is_14_l198_198216


namespace geometric_sequence_problem_l198_198710

noncomputable def geom_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r ^ n

theorem geometric_sequence_problem (a r : ℝ) (a4 a8 a6 a10 : ℝ) :
  a4 = geom_sequence a r 4 →
  a8 = geom_sequence a r 8 →
  a6 = geom_sequence a r 6 →
  a10 = geom_sequence a r 10 →
  a4 + a8 = -2 →
  a4^2 + 2 * a6^2 + a6 * a10 = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end geometric_sequence_problem_l198_198710


namespace triangle_side_length_median_l198_198354

theorem triangle_side_length_median 
  (D E F : Type) 
  (DE : D → E → ℝ) 
  (EF : E → F → ℝ) 
  (DM : D → ℝ)
  (d_e : D) (e_f : F) (m : D)
  (hDE : DE d_e e_f = 7) 
  (hEF : EF e_f e_f = 10)
  (hDM : DM m = 5) :
  ∃ (DF : D → F → ℝ), DF d_e e_f = real.sqrt 149 := 
begin  
  sorry
end

end triangle_side_length_median_l198_198354


namespace right_triangle_exists_l198_198122

-- Define the setup: equilateral triangle ABC, point P, and angle condition
def Point (α : Type*) := α 
def inside {α : Type*} (p : Point α) (A B C : Point α) : Prop := sorry
def angle_at {α : Type*} (p q r : Point α) (θ : ℝ) : Prop := sorry
noncomputable def PA {α : Type*} (P A : Point α) : ℝ := sorry
noncomputable def PB {α : Type*} (P B : Point α) : ℝ := sorry
noncomputable def PC {α : Type*} (P C : Point α) : ℝ := sorry

-- Theorem we need to prove
theorem right_triangle_exists {α : Type*} 
  (A B C P : Point α)
  (h1 : inside P A B C)
  (h2 : angle_at P A B 150) :
  ∃ (Q : Point α), angle_at P Q B 90 :=
sorry

end right_triangle_exists_l198_198122


namespace max_profit_at_max_price_l198_198083

-- Definitions based on the given problem's conditions
def cost_price : ℝ := 30
def profit_margin : ℝ := 0.5
def max_price : ℝ := cost_price * (1 + profit_margin)
def min_price : ℝ := 35
def base_sales : ℝ := 350
def sales_decrease_per_price_increase : ℝ := 50
def price_increase_step : ℝ := 5

-- Profit function based on the conditions
def profit (x : ℝ) : ℝ := (-10 * x^2 + 1000 * x - 21000)

-- Maximum profit and corresponding price
theorem max_profit_at_max_price :
  ∀ x, min_price ≤ x ∧ x ≤ max_price →
  profit x ≤ profit max_price ∧ profit max_price = 3750 :=
by sorry

end max_profit_at_max_price_l198_198083


namespace volume_ratio_of_frustum_l198_198254

theorem volume_ratio_of_frustum
  (h_s h : ℝ)
  (A_s A : ℝ)
  (V_s V : ℝ)
  (ratio_lateral_area : ℝ)
  (ratio_height : ℝ)
  (ratio_base_area : ℝ)
  (H_lateral_area: ratio_lateral_area = 9 / 16)
  (H_height: ratio_height = 3 / 5)
  (H_base_area: ratio_base_area = 9 / 25)
  (H_volume_small: V_s = 1 / 3 * h_s * A_s)
  (H_volume_total: V = 1 / 3 * h * A - 1 / 3 * h_s * A_s) :
  V_s / V = 27 / 98 :=
by
  sorry

end volume_ratio_of_frustum_l198_198254


namespace largest_even_sum_1988_is_290_l198_198349

theorem largest_even_sum_1988_is_290 (n : ℕ) 
  (h : 14 * n = 1988) : 2 * n + 6 = 290 :=
sorry

end largest_even_sum_1988_is_290_l198_198349


namespace sales_percentage_l198_198574

theorem sales_percentage (pens_sales pencils_sales notebooks_sales : ℕ) 
  (h1 : pens_sales = 25)
  (h2 : pencils_sales = 20)
  (h3 : notebooks_sales = 30) :
  100 - (pens_sales + pencils_sales + notebooks_sales) = 25 :=
by
  sorry

end sales_percentage_l198_198574


namespace sum_of_squares_of_roots_l198_198809

theorem sum_of_squares_of_roots :
  (∀ y : ℝ, y^3 - 8*y^2 + 9*y - 2 = 0 → y ≥ 0) →
  (∃ r s t : ℝ, (r^3 - 8*r^2 + 9*r - 2 = 0) ∧ (s^3 - 8*s^2 + 9*s - 2 = 0) ∧ (t^3 - 8*t^2 + 9*t - 2 = 0) ∧
          r + s + t = 8 ∧ r * s + s * t + t * r = 9) →
  r^2 + s^2 + t^2 = 46 :=
by transition
by_paths
by tactic
show ex.

Sorry: Sign-of- root {.no.root,y} good.tags -lean.error

end sum_of_squares_of_roots_l198_198809


namespace fraction_of_180_l198_198058

theorem fraction_of_180 : (1 / 2) * (1 / 3) * (1 / 6) * 180 = 5 := by
  sorry

end fraction_of_180_l198_198058


namespace problem_eq_995_l198_198102

theorem problem_eq_995 :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) /
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400))
  = 995 := sorry

end problem_eq_995_l198_198102


namespace exists_such_h_l198_198284

noncomputable def exists_h (h : ℝ) : Prop :=
  ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋)

theorem exists_such_h : ∃ h : ℝ, exists_h h := 
  -- Let's construct the h as mentioned in the provided proof
  ⟨1969^2 / 1968, 
    by sorry⟩

end exists_such_h_l198_198284


namespace phi_eq_pi_div_two_l198_198577

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.cos (x + ϕ)

theorem phi_eq_pi_div_two (ϕ : ℝ) (h1 : 0 ≤ ϕ) (h2 : ϕ ≤ π)
  (h3 : ∀ x : ℝ, f x ϕ = -f (-x) ϕ) : ϕ = π / 2 :=
sorry

end phi_eq_pi_div_two_l198_198577


namespace solve_a_b_powers_l198_198277

theorem solve_a_b_powers :
  ∃ a b : ℂ, (a + b = 1) ∧ 
             (a^2 + b^2 = 3) ∧ 
             (a^3 + b^3 = 4) ∧ 
             (a^4 + b^4 = 7) ∧ 
             (a^5 + b^5 = 11) ∧ 
             (a^10 + b^10 = 93) :=
sorry

end solve_a_b_powers_l198_198277


namespace stratified_sampling_example_l198_198057

theorem stratified_sampling_example
  (students_ratio : ℕ → ℕ) -- function to get the number of students in each grade, indexed by natural numbers
  (ratio_cond : students_ratio 0 = 4 ∧ students_ratio 1 = 3 ∧ students_ratio 2 = 2) -- the ratio 4:3:2
  (third_grade_sample : ℕ) -- number of students in the third grade in the sample
  (third_grade_sample_eq : third_grade_sample = 10) -- 10 students from the third grade
  (total_sample_size : ℕ) -- the sample size n
 :
  total_sample_size = 45 := 
sorry

end stratified_sampling_example_l198_198057


namespace has_real_root_neg_one_l198_198954

theorem has_real_root_neg_one : 
  (-1)^2 - (-1) - 2 = 0 :=
by 
  sorry

end has_real_root_neg_one_l198_198954


namespace range_of_a_l198_198684

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a+3) / (5-a)) → -3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l198_198684


namespace remainder_of_large_number_l198_198478

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l198_198478


namespace minimum_votes_for_tall_l198_198543

theorem minimum_votes_for_tall (voters : ℕ) (districts : ℕ) (precincts : ℕ) (precinct_voters : ℕ)
  (vote_majority_per_precinct : ℕ → ℕ) (precinct_majority_per_district : ℕ → ℕ) (district_majority_to_win : ℕ) :
  voters = 135 ∧ districts = 5 ∧ precincts = 9 ∧ precinct_voters = 3 ∧
  (∀ p, vote_majority_per_precinct p = 2) ∧
  (∀ d, precinct_majority_per_district d = 5) ∧
  district_majority_to_win = 3 ∧ 
  tall_won : 
  ∃ min_votes, min_votes = 30 :=
by
  sorry

end minimum_votes_for_tall_l198_198543


namespace least_positive_integer_divisible_by_four_smallest_primes_l198_198067

theorem least_positive_integer_divisible_by_four_smallest_primes :
  ∃ n : ℕ, n > 0 ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ n) ∧
    (∀ m : ℕ, m > 0 → (∀ p : ℕ, p ∈ {2, 3, 5, 7} → p ∣ m) → n ≤ m) → n = 210 :=
by
  sorry

end least_positive_integer_divisible_by_four_smallest_primes_l198_198067


namespace min_a_condition_l198_198119

-- Definitions of the conditions
def real_numbers (x : ℝ) := true

def in_interval (a m n : ℝ) : Prop := 0 < n ∧ n < m ∧ m < 1 / a

def inequality (a m n : ℝ) : Prop :=
  (n^(1/m) / m^(1/n) > (n^a) / (m^a))

-- Lean statement
theorem min_a_condition (a m n : ℝ) (h1 : real_numbers m) (h2 : real_numbers n)
    (h3 : in_interval a m n) : inequality a m n ↔ 1 ≤ a :=
sorry

end min_a_condition_l198_198119


namespace paper_cups_count_l198_198970

variables (P C : ℝ) (x : ℕ)

theorem paper_cups_count :
  100 * P + x * C = 7.50 ∧ 20 * P + 40 * C = 1.50 → x = 200 :=
sorry

end paper_cups_count_l198_198970


namespace binary_addition_l198_198796

theorem binary_addition :
  (0b1101 : Nat) + 0b101 + 0b1110 + 0b111 + 0b1010 = 0b10101 := by
  sorry

end binary_addition_l198_198796


namespace total_paint_area_eq_1060_l198_198618

/-- Define the dimensions of the stable and chimney -/
def stable_width := 12
def stable_length := 15
def stable_height := 6
def chimney_width := 2
def chimney_length := 2
def chimney_height := 2

/-- Define the area to be painted computation -/

def wall_area (width length height : ℕ) : ℕ :=
  (width * height * 2) * 2 + (length * height * 2) * 2

def roof_area (width length : ℕ) : ℕ :=
  width * length

def ceiling_area (width length : ℕ) : ℕ :=
  width * length

def chimney_area (width length height : ℕ) : ℕ :=
  (4 * (width * height)) + (width * length)

def total_paint_area : ℕ :=
  wall_area stable_width stable_length stable_height +
  roof_area stable_width stable_length +
  ceiling_area stable_width stable_length +
  chimney_area chimney_width chimney_length chimney_height

/-- Goal: Prove that the total paint area is 1060 sq. yd -/
theorem total_paint_area_eq_1060 : total_paint_area = 1060 := by
  sorry

end total_paint_area_eq_1060_l198_198618


namespace range_of_t_l198_198999

theorem range_of_t 
  (k t : ℝ)
  (tangent_condition : (t + 1)^2 = 1 + k^2)
  (intersect_condition : ∃ x y, y = k * x + t ∧ y = x^2 / 4) : 
  t > 0 ∨ t < -3 :=
sorry

end range_of_t_l198_198999


namespace center_of_circle_l198_198115

theorem center_of_circle (x y : ℝ) : x^2 - 8 * x + y^2 - 4 * y = 4 → (x, y) = (4, 2) :=
by
  sorry

end center_of_circle_l198_198115


namespace fraction_of_girls_participated_l198_198955

theorem fraction_of_girls_participated
  (total_students : ℕ)
  (participating_students : ℕ)
  (participating_girls : ℕ)
  (total_boys total_girls : ℕ)
  (fraction_of_boys_participating : ℚ)
  (total_students_eq : total_students = 800)
  (participating_students_eq : participating_students = 550)
  (participating_girls_eq : participating_girls = 150)
  (fraction_of_boys_participating_eq : fraction_of_boys_participating = 2/3)
  (B_plus_G_eq : total_boys + total_girls = total_students)
  (boys_participating : ℚ)
  (boys_participating_eq : boys_participating = (fraction_of_boys_participating * total_boys))
  (boys_participating_value : total_boys = 600)
  (girls_participating : ℚ)
  (fraction_calculation_eq : girls_participating = 150 / 200)
  :
  girls_participating = 3/4 := by
  sorry -- Proof goes here

end fraction_of_girls_participated_l198_198955


namespace preference_is_related_to_gender_expectation_of_X_correct_l198_198573

noncomputable theory

-- Given conditions
def male_students : ℕ := 100
def female_students : ℕ := 100
def group_a_total : ℕ := 96
def group_a_males : ℕ := 36
def alpha : ℝ := 0.001
def chi_square_critical : ℝ := 10.828

-- Values derived from basic arithmetic operations on given conditions
def group_a_females : ℕ := group_a_total - group_a_males
def group_b_males : ℕ := male_students - group_a_males
def group_b_females : ℕ := female_students - group_a_females
def group_b_total : ℕ := male_students + female_students - group_a_total

-- Chi-square formula components
def ad_bc : ℤ := (group_a_males * group_b_females) - (group_a_females * group_b_males)

def chi_square_value : ℝ := 
  let n : ℝ := (male_students + female_students).toReal
  let a_b : ℝ := (group_a_males + group_a_females).toReal
  let c_d : ℝ := (group_b_males + group_b_females).toReal
  let a_c : ℝ := (group_a_males + group_b_males).toReal
  let b_d : ℝ := (group_a_females + group_b_females).toReal
  (n * (ad_bc)^2) / (a_b * c_d * a_c * b_d).toReal

-- Theorem (statement, no proof)
theorem preference_is_related_to_gender : chi_square_value > chi_square_critical := sorry

-- For Part (2): Distribution table and expectation
def X_distribution : ℕ → ℝ
| 0 => (28/115)
| 1 => (54/115)
| 2 => (144/575)
| 3 => (21/575)
| _ => 0 -- Defined for completeness, though not necessary

def E_X : ℝ := 0 * (28/115) + 1 * (54/115) + 2 * (144/575) + 3 * (21/575)

-- Theorem (statement, no proof)
theorem expectation_of_X_correct : E_X = (621/575) := sorry

end preference_is_related_to_gender_expectation_of_X_correct_l198_198573


namespace problem_solution_l198_198344

theorem problem_solution (a0 a1 a2 a3 a4 a5 : ℝ) :
  (1 + 2*x)^5 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 →
  a0 + a2 + a4 = 121 := 
sorry

end problem_solution_l198_198344


namespace cube_sum_l198_198675

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cube_sum_l198_198675


namespace find_x_minus_y_l198_198832

/-
Given that:
  2 * x + y = 7
  x + 2 * y = 8
We want to prove:
  x - y = -1
-/

theorem find_x_minus_y (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : x - y = -1 :=
by
  sorry

end find_x_minus_y_l198_198832


namespace rectangular_field_area_l198_198787

theorem rectangular_field_area (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangular_field_area_l198_198787


namespace markus_more_marbles_than_mara_l198_198728

variable (mara_bags : Nat) (mara_marbles_per_bag : Nat)
variable (markus_bags : Nat) (markus_marbles_per_bag : Nat)

theorem markus_more_marbles_than_mara :
  mara_bags = 12 →
  mara_marbles_per_bag = 2 →
  markus_bags = 2 →
  markus_marbles_per_bag = 13 →
  (markus_bags * markus_marbles_per_bag) - (mara_bags * mara_marbles_per_bag) = 2 :=
by
  intros
  sorry

end markus_more_marbles_than_mara_l198_198728


namespace fractions_arithmetic_lemma_l198_198817

theorem fractions_arithmetic_lemma : (8 / 15 : ℚ) - (7 / 9) + (3 / 4) = 1 / 2 := 
by
  sorry

end fractions_arithmetic_lemma_l198_198817


namespace conversion_correct_l198_198977

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.enum.foldl (λ acc ⟨i, digit⟩ => acc + digit * 2^i) 0

def n : List ℕ := [1, 0, 1, 1, 1, 1, 0, 1, 1]

theorem conversion_correct :
  binary_to_decimal n = 379 :=
by 
  sorry

end conversion_correct_l198_198977


namespace spherical_to_rectangular_coords_l198_198279

theorem spherical_to_rectangular_coords
  (ρ θ φ : ℝ)
  (hρ : ρ = 6)
  (hθ : θ = 7 * Real.pi / 4)
  (hφ : φ = Real.pi / 3) :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = -3 * Real.sqrt 6 ∧ y = -3 * Real.sqrt 6 ∧ z = 3 :=
by
  sorry

end spherical_to_rectangular_coords_l198_198279


namespace price_of_soda_l198_198596

-- Definitions based on the conditions given in the problem
def initial_amount := 500
def cost_rice := 2 * 20
def cost_wheat_flour := 3 * 25
def remaining_balance := 235
def total_cost := cost_rice + cost_wheat_flour

-- Definition to be proved
theorem price_of_soda : initial_amount - total_cost - remaining_balance = 150 := by
  sorry

end price_of_soda_l198_198596


namespace yura_finishes_problems_l198_198706

theorem yura_finishes_problems 
  (total_problems : ℕ) (start_date : ℕ) (remaining_problems : ℕ) (z : ℕ)
  (H_total : total_problems = 91)
  (H_start_date : start_date = 6)
  (H_remaining : remaining_problems = 46)
  (H_solved_by_8th : (z + 1) + z + (z - 1) = total_problems - remaining_problems) :
  ∃ finish_date, finish_date = 12 :=
by
  use 12
  sorry

end yura_finishes_problems_l198_198706


namespace football_team_gain_l198_198087

theorem football_team_gain (G : ℤ) :
  (-5 + G = 2) → (G = 7) :=
by
  intro h
  sorry

end football_team_gain_l198_198087


namespace total_cups_l198_198565

theorem total_cups (t1 t2 : ℕ) (h1 : t2 = 240) (h2 : t2 = t1 - 20) : t1 + t2 = 500 := by
  sorry

end total_cups_l198_198565
