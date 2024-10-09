import Mathlib

namespace smallest_x_l1888_188834

theorem smallest_x (x : ℕ) (M : ℕ) (h : 1800 * x = M^3) :
  x = 30 :=
by
  sorry

end smallest_x_l1888_188834


namespace mutually_exclusive_but_not_complementary_l1888_188839

-- Definitions for the problem conditions
inductive Card
| red | black | white | blue

inductive Person
| A | B | C | D

open Card Person

-- The statement of the proof
theorem mutually_exclusive_but_not_complementary : 
  (∃ (f : Person → Card), (f A = red) ∧ (f B ≠ red)) ∧ (∃ (f : Person → Card), (f B = red) ∧ (f A ≠ red)) :=
sorry

end mutually_exclusive_but_not_complementary_l1888_188839


namespace problem_sol_l1888_188867

theorem problem_sol (a b : ℝ) (h : ∀ x, (x > -1 ∧ x < 1/3) ↔ (ax^2 + bx + 1 > 0)) : a * b = 6 :=
sorry

end problem_sol_l1888_188867


namespace find_n_l1888_188822

-- Define x and y
def x : ℕ := 3
def y : ℕ := 1

-- Define n based on the given expression.
def n : ℕ := x - y^(x - (y + 1))

-- State the theorem
theorem find_n : n = 2 := by
  sorry

end find_n_l1888_188822


namespace food_consumption_reduction_l1888_188849

noncomputable def reduction_factor (n p : ℝ) : ℝ :=
  (n * p) / ((n - 0.05 * n) * (p + 0.2 * p))

theorem food_consumption_reduction (n p : ℝ) (h : n > 0 ∧ p > 0) :
  (1 - reduction_factor n p) * 100 = 12.28 := by
  sorry

end food_consumption_reduction_l1888_188849


namespace cone_base_radius_l1888_188841

/--
Given a cone with the following properties:
1. The surface area of the cone is \(3\pi\).
2. The lateral surface of the cone unfolds into a semicircle (which implies the slant height is twice the radius of the base).
Prove that the radius of the base of the cone is \(1\).
-/
theorem cone_base_radius 
  (S : ℝ)
  (r l : ℝ)
  (h1 : S = 3 * Real.pi)
  (h2 : l = 2 * r)
  : r = 1 := 
  sorry

end cone_base_radius_l1888_188841


namespace calories_consumed_l1888_188845

-- Define the conditions
def pages_written : ℕ := 12
def pages_per_donut : ℕ := 2
def calories_per_donut : ℕ := 150

-- Define the theorem to be proved
theorem calories_consumed (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

-- Ensure the theorem corresponds to the correct answer
example : calories_consumed 12 2 150 = 900 := by
  sorry

end calories_consumed_l1888_188845


namespace derek_percentage_difference_l1888_188846

-- Definitions and assumptions based on conditions
def average_score_first_test (A : ℝ) : ℝ := A

def derek_score_first_test (D1 : ℝ) (A : ℝ) : Prop := D1 = 0.5 * A

def derek_score_second_test (D2 : ℝ) (D1 : ℝ) : Prop := D2 = 1.5 * D1

-- Theorem statement
theorem derek_percentage_difference (A D1 D2 : ℝ)
  (h1 : derek_score_first_test D1 A)
  (h2 : derek_score_second_test D2 D1) :
  (A - D2) / A * 100 = 25 :=
by
  -- Placeholder for the proof
  sorry

end derek_percentage_difference_l1888_188846


namespace cirrus_to_cumulus_is_four_l1888_188847

noncomputable def cirrus_to_cumulus_ratio (Ci Cu Cb : ℕ) : ℕ :=
  Ci / Cu

theorem cirrus_to_cumulus_is_four :
  ∀ (Ci Cu Cb : ℕ), (Cb = 3) → (Cu = 12 * Cb) → (Ci = 144) → cirrus_to_cumulus_ratio Ci Cu Cb = 4 :=
by
  intros Ci Cu Cb hCb hCu hCi
  sorry

end cirrus_to_cumulus_is_four_l1888_188847


namespace find_x_l1888_188875

def otimes (m n : ℝ) : ℝ := m^2 - 2*m*n

theorem find_x (x : ℝ) (h : otimes (x + 1) (x - 2) = 5) : x = 0 ∨ x = 4 := 
by
  sorry

end find_x_l1888_188875


namespace proposition_is_false_l1888_188836

noncomputable def false_proposition : Prop :=
¬(∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), Real.sin x + Real.cos x ≥ 2)

theorem proposition_is_false : false_proposition :=
by
  sorry

end proposition_is_false_l1888_188836


namespace solve_system_eq_l1888_188804

theorem solve_system_eq (x1 x2 x3 x4 x5 : ℝ) :
  (x3 + x4 + x5)^5 = 3 * x1 ∧
  (x4 + x5 + x1)^5 = 3 * x2 ∧
  (x5 + x1 + x2)^5 = 3 * x3 ∧
  (x1 + x2 + x3)^5 = 3 * x4 ∧
  (x2 + x3 + x4)^5 = 3 * x5 →
  (x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨
  (x1 = 1/3 ∧ x2 = 1/3 ∧ x3 = 1/3 ∧ x4 = 1/3 ∧ x5 = 1/3) ∨
  (x1 = -1/3 ∧ x2 = -1/3 ∧ x3 = -1/3 ∧ x4 = -1/3 ∧ x5 = -1/3) :=
by
  sorry

end solve_system_eq_l1888_188804


namespace greatest_possible_mean_BC_l1888_188869

theorem greatest_possible_mean_BC :
  ∀ (A_n B_n C_weight C_n : ℕ),
    (A_n > 0) ∧ (B_n > 0) ∧ (C_n > 0) ∧
    (40 * A_n + 50 * B_n) / (A_n + B_n) = 43 ∧
    (40 * A_n + C_weight) / (A_n + C_n) = 44 →
    ∃ k : ℕ, ∃ n : ℕ, 
      A_n = 7 * k ∧ B_n = 3 * k ∧ 
      C_weight = 28 * k + 44 * n ∧ 
      44 + 46 * k / (3 * k + n) ≤ 59 :=
sorry

end greatest_possible_mean_BC_l1888_188869


namespace inequality_solution_set_inequality_proof_l1888_188819

def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

theorem inequality_solution_set :
  ∀ x : ℝ, -2 < f x ∧ f x < 0 ↔ -1/2 < x ∧ x < 1/2 :=
by
  sorry

theorem inequality_proof (m n : ℝ) (h_m : -1/2 < m ∧ m < 1/2) (h_n : -1/2 < n ∧ n < 1/2) :
  |1 - 4 * m * n| > 2 * |m - n| :=
by
  sorry

end inequality_solution_set_inequality_proof_l1888_188819


namespace simplify_and_sum_coefficients_l1888_188848

theorem simplify_and_sum_coefficients :
  (∃ A B C D : ℤ, (∀ x : ℝ, x ≠ D → (x^3 + 6 * x^2 + 11 * x + 6) / (x + 1) = A * x^2 + B * x + C) ∧ A + B + C + D = 11) :=
sorry

end simplify_and_sum_coefficients_l1888_188848


namespace rearrange_digits_to_perfect_square_l1888_188862

theorem rearrange_digits_to_perfect_square :
  ∃ n : ℤ, 2601 = n ^ 2 ∧ (∃ (perm : List ℤ), perm = [2, 0, 1, 6] ∧ perm.permutations ≠ List.nil) :=
by
  sorry

end rearrange_digits_to_perfect_square_l1888_188862


namespace simplify_expression_l1888_188843

theorem simplify_expression :
  (20^4 + 625) * (40^4 + 625) * (60^4 + 625) * (80^4 + 625) /
  (10^4 + 625) * (30^4 + 625) * (50^4 + 625) * (70^4 + 625) = 7 := 
sorry

end simplify_expression_l1888_188843


namespace megan_pictures_l1888_188818

theorem megan_pictures (pictures_zoo pictures_museum pictures_deleted : ℕ)
  (hzoo : pictures_zoo = 15)
  (hmuseum : pictures_museum = 18)
  (hdeleted : pictures_deleted = 31) :
  (pictures_zoo + pictures_museum) - pictures_deleted = 2 :=
by
  sorry

end megan_pictures_l1888_188818


namespace calculate_drift_l1888_188877

theorem calculate_drift (w v t : ℝ) (hw : w = 400) (hv : v = 10) (ht : t = 50) : v * t - w = 100 :=
by
  sorry

end calculate_drift_l1888_188877


namespace cheese_stick_problem_l1888_188837

theorem cheese_stick_problem (cheddar pepperjack mozzarella : ℕ) (total : ℕ)
    (h1 : cheddar = 15)
    (h2 : pepperjack = 45)
    (h3 : 2 * pepperjack = total)
    (h4 : total = cheddar + pepperjack + mozzarella) :
    mozzarella = 30 :=
by
    sorry

end cheese_stick_problem_l1888_188837


namespace area_of_wall_photo_l1888_188899

theorem area_of_wall_photo (width_frame : ℕ) (width_paper : ℕ) (length_paper : ℕ) 
  (h_width_frame : width_frame = 2) (h_width_paper : width_paper = 8) (h_length_paper : length_paper = 12) :
  (width_paper + 2 * width_frame) * (length_paper + 2 * width_frame) = 192 :=
by
  sorry

end area_of_wall_photo_l1888_188899


namespace determine_m_minus_n_l1888_188808

-- Definitions of the conditions
variables {m n : ℝ}

-- The proof statement
theorem determine_m_minus_n (h_eq : ∀ x y : ℝ, x^(4 - 3 * |m|) + y^(3 * |n|) = 2009 → x + y = 2009)
  (h_prod_lt_zero : m * n < 0)
  (h_sum : 0 < m + n ∧ m + n ≤ 3) : m - n = 4/3 := 
sorry

end determine_m_minus_n_l1888_188808


namespace dacid_average_l1888_188870

noncomputable def average (a b : ℕ) : ℚ :=
(a + b) / 2

noncomputable def overall_average (a b c d e : ℕ) : ℚ :=
(a + b + c + d + e) / 5

theorem dacid_average :
  ∀ (english mathematics physics chemistry biology : ℕ),
  english = 86 →
  mathematics = 89 →
  physics = 82 →
  chemistry = 87 →
  biology = 81 →
  (average english mathematics < 90) ∧
  (average english physics < 90) ∧
  (average english chemistry < 90) ∧
  (average english biology < 90) ∧
  (average mathematics physics < 90) ∧
  (average mathematics chemistry < 90) ∧
  (average mathematics biology < 90) ∧
  (average physics chemistry < 90) ∧
  (average physics biology < 90) ∧
  (average chemistry biology < 90) ∧
  overall_average english mathematics physics chemistry biology = 85 := by
  intros english mathematics physics chemistry biology
  intros h_english h_mathematics h_physics h_chemistry h_biology
  simp [average, overall_average]
  rw [h_english, h_mathematics, h_physics, h_chemistry, h_biology]
  sorry

end dacid_average_l1888_188870


namespace Sam_has_4_French_Bulldogs_l1888_188805

variable (G F : ℕ)

theorem Sam_has_4_French_Bulldogs
  (h1 : G = 3)
  (h2 : 3 * G + 2 * F = 17) :
  F = 4 :=
sorry

end Sam_has_4_French_Bulldogs_l1888_188805


namespace inverse_proportion_function_l1888_188840

theorem inverse_proportion_function (m x : ℝ) (h : (m ≠ 0)) (A : (m, m / 8) ∈ {p : ℝ × ℝ | p.snd = (m / p.fst)}) :
    ∃ f : ℝ → ℝ, (∀ x, f x = 8 / x) :=
by
  use (fun x => 8 / x)
  intros x
  rfl

end inverse_proportion_function_l1888_188840


namespace david_remaining_money_l1888_188852

-- Given conditions
def hourly_rate : ℕ := 14
def hours_per_day : ℕ := 2
def days_in_week : ℕ := 7
def weekly_earnings : ℕ := hourly_rate * hours_per_day * days_in_week
def cost_of_shoes : ℕ := weekly_earnings / 2
def remaining_after_shoes : ℕ := weekly_earnings - cost_of_shoes
def given_to_mom : ℕ := remaining_after_shoes / 2
def remaining_after_gift : ℕ := remaining_after_shoes - given_to_mom

-- Theorem
theorem david_remaining_money : remaining_after_gift = 49 := by
  sorry

end david_remaining_money_l1888_188852


namespace determine_a_values_l1888_188800

theorem determine_a_values (a : ℝ) (A : Set ℝ) (B : Set ℝ)
  (hA : A = { x | abs x = 1 }) 
  (hB : B = { x | a * x = 1 }) 
  (h_superset : A ⊇ B) :
  a = -1 ∨ a = 0 ∨ a = 1 :=
sorry

end determine_a_values_l1888_188800


namespace factor_expression_l1888_188863

variable (x : ℕ)

theorem factor_expression : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end factor_expression_l1888_188863


namespace weight_per_linear_foot_l1888_188878

theorem weight_per_linear_foot 
  (length_of_log : ℕ) 
  (cut_length : ℕ) 
  (piece_weight : ℕ) 
  (h1 : length_of_log = 20) 
  (h2 : cut_length = length_of_log / 2) 
  (h3 : piece_weight = 1500) 
  (h4 : length_of_log / 2 = 10) 
  : piece_weight / cut_length = 150 := 
  by 
  sorry

end weight_per_linear_foot_l1888_188878


namespace percentage_increase_expenditure_l1888_188873

variable (I : ℝ) -- original income
variable (E : ℝ) -- original expenditure
variable (I_new : ℝ) -- new income
variable (S : ℝ) -- original savings
variable (S_new : ℝ) -- new savings

-- a) Conditions
def initial_spend (I : ℝ) : ℝ := 0.75 * I
def income_increased (I : ℝ) : ℝ := 1.20 * I
def savings_increased (S : ℝ) : ℝ := 1.4999999999999996 * S

-- b) Definitions relating formulated conditions
def new_expenditure (I : ℝ) : ℝ := 1.20 * I - 0.3749999999999999 * I
def original_expenditure (I : ℝ) : ℝ := 0.75 * I

-- c) Proof statement
theorem percentage_increase_expenditure :
  initial_spend I = E →
  income_increased I = I_new →
  savings_increased (0.25 * I) = S_new →
  ((new_expenditure I - original_expenditure I) / original_expenditure I) * 100 = 10 := 
by 
  intros h1 h2 h3
  sorry

end percentage_increase_expenditure_l1888_188873


namespace volume_relation_l1888_188815

variable {x y z V : ℝ}

theorem volume_relation
  (top_area : x * y = A)
  (side_area : y * z = B)
  (volume : x * y * z = V) :
  (y * z) * (x * y * z)^2 = z^3 * V := by
  sorry

end volume_relation_l1888_188815


namespace minute_hand_angle_backward_l1888_188872

theorem minute_hand_angle_backward (backward_minutes : ℝ) (h : backward_minutes = 10) :
  (backward_minutes / 60) * (2 * Real.pi) = Real.pi / 3 := by
  sorry

end minute_hand_angle_backward_l1888_188872


namespace helium_min_cost_l1888_188861

noncomputable def W (x : ℝ) : ℝ :=
  if x < 4 then 40 * (4 * x + 16 / x + 100)
  else 40 * (9 / (x * x) - 3 / x + 117)

theorem helium_min_cost :
  (∀ x, W x ≥ 4640) ∧ (W 2 = 4640) :=
by {
  sorry
}

end helium_min_cost_l1888_188861


namespace cost_per_pack_l1888_188829

variable (total_amount : ℕ) (number_of_packs : ℕ)

theorem cost_per_pack (h1 : total_amount = 132) (h2 : number_of_packs = 11) : 
  total_amount / number_of_packs = 12 := by
  sorry

end cost_per_pack_l1888_188829


namespace set_contains_difference_of_elements_l1888_188811

variable {A : Set Int}

axiom cond1 (a : Int) (ha : a ∈ A) : 2 * a ∈ A
axiom cond2 (a b : Int) (ha : a ∈ A) (hb : b ∈ A) : a + b ∈ A

theorem set_contains_difference_of_elements 
  (a b : Int) (ha : a ∈ A) (hb : b ∈ A) : a - b ∈ A := by
  sorry

end set_contains_difference_of_elements_l1888_188811


namespace sum_diameters_eq_sum_legs_l1888_188842

theorem sum_diameters_eq_sum_legs 
  (a b c R r : ℝ)
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_circum_radius : R = c / 2)
  (h_incircle_radius : r = (a + b - c) / 2) :
  2 * R + 2 * r = a + b :=
by 
  sorry

end sum_diameters_eq_sum_legs_l1888_188842


namespace base_height_is_two_inches_l1888_188890

noncomputable def height_sculpture_feet : ℝ := 2 + (10 / 12)
noncomputable def combined_height_feet : ℝ := 3
noncomputable def base_height_feet : ℝ := combined_height_feet - height_sculpture_feet
noncomputable def base_height_inches : ℝ := base_height_feet * 12

theorem base_height_is_two_inches :
  base_height_inches = 2 := by
  sorry

end base_height_is_two_inches_l1888_188890


namespace count_polynomials_with_three_integer_roots_l1888_188868

def polynomial_with_roots (n: ℕ) : Nat :=
  have h: n = 8 := by
    sorry
  if n = 8 then
    -- Apply the combinatorial argument as discussed
    52
  else
    -- Case for other n
    0

theorem count_polynomials_with_three_integer_roots:
  polynomial_with_roots 8 = 52 := 
  sorry

end count_polynomials_with_three_integer_roots_l1888_188868


namespace simplify_expression_l1888_188838

theorem simplify_expression (a : ℝ) (h : a ≠ 2) : 
  (a^2 / (a - 2)) - (4 / (a - 2)) = a + 2 :=
by 
  sorry

end simplify_expression_l1888_188838


namespace right_triangle_345_l1888_188826

theorem right_triangle_345 : 
  (∃ a b c : ℕ, a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 2 ∧ b = 3 ∧ c = 4 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 4 ∧ b = 5 ∧ c = 6 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 6 ∧ b = 8 ∧ c = 9 ∧ a^2 + b^2 = c^2) :=
by {
  sorry
}

end right_triangle_345_l1888_188826


namespace cost_price_l1888_188882

/-- A person buys an article at some price. 
They sell the article to make a profit of 24%. 
The selling price of the article is Rs. 595.2. 
Prove that the cost price (CP) is Rs. 480. -/
theorem cost_price (SP CP : ℝ) (h1 : SP = 595.2) (h2 : SP = CP * (1 + 0.24)) : CP = 480 := 
by sorry 

end cost_price_l1888_188882


namespace solve_for_x_l1888_188817

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 9 / (x / 3)) : x = 15 * Real.sqrt 1.8 ∨ x = -15 * Real.sqrt 1.8 := 
by
  sorry

end solve_for_x_l1888_188817


namespace length_real_axis_hyperbola_l1888_188889

theorem length_real_axis_hyperbola :
  (∃ (C : ℝ → ℝ → Prop) (a b : ℝ), (a > 0) ∧ (b > 0) ∧ 
    (∀ x y : ℝ, C x y = ((x ^ 2) / a ^ 2 - (y ^ 2) / b ^ 2 = 1)) ∧
      (∀ x y : ℝ, ((x ^ 2) / 9 - (y ^ 2) / 16 = 1) → ((x ^ 2) / a ^ 2 - (y ^ 2) / b ^ 2 = 1)) ∧
      C (-3) (2 * Real.sqrt 3)) →
  2 * (3 / 2) = 3 :=
by {
  sorry
}

end length_real_axis_hyperbola_l1888_188889


namespace max_ballpoint_pens_l1888_188813

def ballpoint_pen_cost : ℕ := 10
def gel_pen_cost : ℕ := 30
def fountain_pen_cost : ℕ := 60
def total_pens : ℕ := 20
def total_cost : ℕ := 500

theorem max_ballpoint_pens : ∃ (x y z : ℕ), 
  x + y + z = total_pens ∧ 
  ballpoint_pen_cost * x + gel_pen_cost * y + fountain_pen_cost * z = total_cost ∧ 
  1 ≤ x ∧ 
  1 ≤ y ∧
  1 ≤ z ∧
  ∀ x', ((∃ y' z', x' + y' + z' = total_pens ∧ 
                    ballpoint_pen_cost * x' + gel_pen_cost * y' + fountain_pen_cost * z' = total_cost ∧ 
                    1 ≤ x' ∧ 
                    1 ≤ y' ∧
                    1 ≤ z') → x' ≤ x) :=
  sorry

end max_ballpoint_pens_l1888_188813


namespace combination_addition_l1888_188835

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem combination_addition :
  combination 13 11 + 3 = 81 :=
by
  sorry

end combination_addition_l1888_188835


namespace count_five_digit_numbers_with_digit_8_l1888_188844

theorem count_five_digit_numbers_with_digit_8 : 
    let total_numbers := 99999 - 10000 + 1
    let without_8 := 8 * (9 ^ 4)
    90000 - without_8 = 37512 := by
    let total_numbers := 99999 - 10000 + 1 -- Total number of five-digit numbers
    let without_8 := 8 * (9 ^ 4) -- Number of five-digit numbers without any '8'
    show total_numbers - without_8 = 37512
    sorry

end count_five_digit_numbers_with_digit_8_l1888_188844


namespace min_value_expression_l1888_188802

theorem min_value_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 3) :
  ∃ (M : ℝ), M = (2 : ℝ) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y = 3 → ((y / x) + (3 / (y + 1)) ≥ M)) :=
by
  use 2
  sorry

end min_value_expression_l1888_188802


namespace eval_exp_l1888_188824

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l1888_188824


namespace area_of_smallest_square_containing_circle_l1888_188801

theorem area_of_smallest_square_containing_circle (r : ℝ) (h : r = 7) : ∃ s, s = 14 ∧ s * s = 196 :=
by
  sorry

end area_of_smallest_square_containing_circle_l1888_188801


namespace competition_result_l1888_188825

variables (Olya Oleg Pasha : ℕ)

theorem competition_result 
  (h1 : Olya ≠ 1 → Olya ≠ 3 → False)
  (h2 : (Oleg = 1 ∨ Oleg = 3) → Olya = 3)
  (h3 : (Oleg ≠ 1 → (Olya = 2 ∨ Olya = 3)))
  (h4 : Olya ≠ 1 ∧ Oleg ≠ 2 ∧ Pasha ≠ 3) :
  Oleg = 1 ∧ Pasha = 2 ∧ Olya = 3 := 
by {
  sorry
}

end competition_result_l1888_188825


namespace sum_a4_a6_l1888_188803

variable (a : ℕ → ℝ) (d : ℝ)
variable (h_arith : ∀ n : ℕ, a (n + 1) = a 1 + n * d)
variable (h_sum : a 2 + a 3 + a 7 + a 8 = 8)

theorem sum_a4_a6 : a 4 + a 6 = 4 :=
by
  sorry

end sum_a4_a6_l1888_188803


namespace simplify_expression_l1888_188856

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -1) (h₂: x ≠ -3) :
  (x - 1 - 8 / (x + 1)) / ( (x + 3) / (x + 1) ) = x - 3 :=
by
  sorry

end simplify_expression_l1888_188856


namespace surveys_from_retired_is_12_l1888_188858

-- Define the given conditions
def ratio_retired : ℕ := 2
def ratio_current : ℕ := 8
def ratio_students : ℕ := 40
def total_surveys : ℕ := 300
def total_ratio : ℕ := ratio_retired + ratio_current + ratio_students

-- Calculate the expected number of surveys from retired faculty
def number_of_surveys_retired : ℕ := total_surveys * ratio_retired / total_ratio

-- Lean 4 statement for proof
theorem surveys_from_retired_is_12 :
  number_of_surveys_retired = 12 :=
by
  -- Proof to be filled in
  sorry

end surveys_from_retired_is_12_l1888_188858


namespace book_difference_l1888_188896

def initial_books : ℕ := 75
def borrowed_books : ℕ := 18
def difference : ℕ := initial_books - borrowed_books

theorem book_difference : difference = 57 := by
  -- Proof will go here
  sorry

end book_difference_l1888_188896


namespace tangent_curve_l1888_188886

variable {k a b : ℝ}

theorem tangent_curve (h1 : 3 = (1 : ℝ)^3 + a * 1 + b)
(h2 : k = 2)
(h3 : k = 3 * (1 : ℝ)^2 + a) :
b = 3 :=
by
  sorry

end tangent_curve_l1888_188886


namespace positive_integer_solutions_eq_17_l1888_188883

theorem positive_integer_solutions_eq_17 :
  {x : ℕ // x > 0} × {y : ℕ // y > 0} → 5 * x + 10 * y = 100 ->
  ∃ (n : ℕ), n = 17 := sorry

end positive_integer_solutions_eq_17_l1888_188883


namespace last_year_sales_l1888_188888

-- Define the conditions as constants
def sales_this_year : ℝ := 480
def percent_increase : ℝ := 0.50

-- The main theorem statement
theorem last_year_sales : 
  ∃ sales_last_year : ℝ, sales_this_year = sales_last_year * (1 + percent_increase) ∧ sales_last_year = 320 := 
by 
  sorry

end last_year_sales_l1888_188888


namespace volume_of_prism_l1888_188831

theorem volume_of_prism (a : ℝ) (h_pos : 0 < a) (h_lat : ∀ S_lat, S_lat = a ^ 2) : 
  ∃ V, V = (a ^ 3 * (Real.sqrt 2 - 1)) / 4 :=
by
  sorry

end volume_of_prism_l1888_188831


namespace sum_of_ages_now_l1888_188865

variable (D A Al B : ℝ)

noncomputable def age_condition (D : ℝ) : Prop :=
  D = 16

noncomputable def alex_age_condition (A : ℝ) : Prop :=
  A = 60 - (30 - 16)

noncomputable def allison_age_condition (Al : ℝ) : Prop :=
  Al = 15 - (30 - 16)

noncomputable def bernard_age_condition (B A Al : ℝ) : Prop :=
  B = (A + Al) / 2

noncomputable def sum_of_ages (A Al B : ℝ) : ℝ :=
  A + Al + B

theorem sum_of_ages_now :
  age_condition D →
  alex_age_condition A →
  allison_age_condition Al →
  bernard_age_condition B A Al →
  sum_of_ages A Al B = 70.5 := by
  sorry

end sum_of_ages_now_l1888_188865


namespace ratio_of_x_to_y_l1888_188816

-- Defining the given condition
def ratio_condition (x y : ℝ) : Prop :=
  (3 * x - 2 * y) / (2 * x + y) = 3 / 5

-- The theorem to be proven
theorem ratio_of_x_to_y (x y : ℝ) (h : ratio_condition x y) : x / y = 13 / 9 :=
by
  sorry

end ratio_of_x_to_y_l1888_188816


namespace ashok_total_subjects_l1888_188897

variable (n : ℕ) (T : ℕ)

theorem ashok_total_subjects (h_ave_all : 75 * n = T + 80)
                       (h_ave_first : T = 74 * (n - 1)) :
  n = 6 := sorry

end ashok_total_subjects_l1888_188897


namespace find_complex_Z_l1888_188876

open Complex

theorem find_complex_Z (Z : ℂ) (h : (2 + 4 * I) / Z = 1 - I) : 
  Z = -1 + 3 * I :=
by
  sorry

end find_complex_Z_l1888_188876


namespace tetrahedron_volume_l1888_188857

theorem tetrahedron_volume (h_1 h_2 h_3 : ℝ) (V : ℝ)
  (h1_pos : 0 < h_1) (h2_pos : 0 < h_2) (h3_pos : 0 < h_3)
  (V_nonneg : 0 ≤ V) : 
  V ≥ (1 / 3) * h_1 * h_2 * h_3 := sorry

end tetrahedron_volume_l1888_188857


namespace cost_price_marked_price_ratio_l1888_188827

theorem cost_price_marked_price_ratio (x : ℝ) (hx : x > 0) :
  let selling_price := (2 / 3) * x
  let cost_price := (3 / 4) * selling_price 
  cost_price / x = 1 / 2 := 
by
  let selling_price := (2 / 3) * x 
  let cost_price := (3 / 4) * selling_price 
  have hs : selling_price = (2 / 3) * x := rfl 
  have hc : cost_price = (3 / 4) * selling_price := rfl 
  have ratio := hc.symm 
  simp [ratio, hs]
  sorry

end cost_price_marked_price_ratio_l1888_188827


namespace average_of_a_and_b_l1888_188854

theorem average_of_a_and_b (a b c : ℝ) 
  (h₁ : (b + c) / 2 = 90)
  (h₂ : c - a = 90) :
  (a + b) / 2 = 45 :=
sorry

end average_of_a_and_b_l1888_188854


namespace geese_count_l1888_188874

variables (k n : ℕ)

theorem geese_count (h1 : k * n = (k + 20) * (n - 75)) (h2 : k * n = (k - 15) * (n + 100)) : n = 300 :=
by
  sorry

end geese_count_l1888_188874


namespace largest_x_value_l1888_188884

theorem largest_x_value
  (x : ℝ)
  (h : (17 * x^2 - 46 * x + 21) / (5 * x - 3) + 7 * x = 8 * x - 2)
  : x = 5 / 3 :=
sorry

end largest_x_value_l1888_188884


namespace find_common_difference_l1888_188833

theorem find_common_difference (AB BC AC : ℕ) (x y z d : ℕ) 
  (h1 : AB = 300) (h2 : BC = 350) (h3 : AC = 400) 
  (hx : x = (2 * d) / 5) (hy : y = (7 * d) / 15) (hz : z = (8 * d) / 15) 
  (h_sum : x + y + z = 750) : 
  d = 536 :=
by
  -- Proof goes here
  sorry

end find_common_difference_l1888_188833


namespace SammyFinishedProblems_l1888_188881

def initial : ℕ := 9 -- number of initial math problems
def remaining : ℕ := 7 -- number of remaining math problems
def finished (init rem : ℕ) : ℕ := init - rem -- defining number of finished problems

theorem SammyFinishedProblems : finished initial remaining = 2 := by
  sorry -- placeholder for proof

end SammyFinishedProblems_l1888_188881


namespace xiao_xuan_wins_l1888_188850

def cards_game (n : ℕ) (min_take : ℕ) (max_take : ℕ) (initial_turn : String) : String :=
  if initial_turn = "Xiao Liang" then "Xiao Xuan" else "Xiao Liang"

theorem xiao_xuan_wins :
  cards_game 17 1 2 "Xiao Liang" = "Xiao Xuan" :=
sorry

end xiao_xuan_wins_l1888_188850


namespace initial_number_of_observations_l1888_188821

theorem initial_number_of_observations (n : ℕ) 
  (initial_mean : ℝ := 100) 
  (wrong_obs : ℝ := 75) 
  (corrected_obs : ℝ := 50) 
  (corrected_mean : ℝ := 99.075) 
  (h1 : (n:ℝ) * initial_mean = n * corrected_mean + wrong_obs - corrected_obs) 
  (h2 : n = (25 : ℝ) / 0.925) 
  : n = 27 := 
sorry

end initial_number_of_observations_l1888_188821


namespace min_trips_needed_l1888_188895

noncomputable def min_trips (n : ℕ) (h : 2 ≤ n) : ℕ :=
  6

theorem min_trips_needed
  (n : ℕ) (h : 2 ≤ n) (students : Finset (Fin (2 * n)))
  (trip : ℕ → Finset (Fin (2 * n)))
  (trip_cond : ∀ i, (trip i).card = n)
  (pair_cond : ∀ (s t : Fin (2 * n)),
    s ≠ t → ∃ i, s ∈ trip i ∧ t ∈ trip i) :
  ∃ k, k = min_trips n h :=
by
  use 6
  sorry

end min_trips_needed_l1888_188895


namespace orange_marbles_l1888_188832

-- Definitions based on the given conditions
def total_marbles : ℕ := 24
def blue_marbles : ℕ := total_marbles / 2
def red_marbles : ℕ := 6

-- The statement to prove: the number of orange marbles is 6
theorem orange_marbles : (total_marbles - (blue_marbles + red_marbles)) = 6 := 
  by 
  sorry

end orange_marbles_l1888_188832


namespace common_difference_range_l1888_188893

variable (d : ℝ)

def a (n : ℕ) : ℝ := -5 + (n - 1) * d

theorem common_difference_range (H1 : a 10 > 0) (H2 : a 9 ≤ 0) :
  (5 / 9 < d) ∧ (d ≤ 5 / 8) :=
by
  sorry

end common_difference_range_l1888_188893


namespace least_value_expression_l1888_188809

theorem least_value_expression (x : ℝ) (h : x < -2) :
  2 * x < x ∧ 2 * x < x + 2 ∧ 2 * x < (1 / 2) * x ∧ 2 * x < x - 2 :=
by
  sorry

end least_value_expression_l1888_188809


namespace gain_percent_l1888_188892

variable (C S : ℝ)

theorem gain_percent (h : 50 * C = 28 * S) : ((S - C) / C) * 100 = 78.57 := by
  sorry

end gain_percent_l1888_188892


namespace original_inhabitants_proof_l1888_188866

noncomputable def original_inhabitants (final_population : ℕ) : ℝ :=
  final_population / (0.75 * 0.9)

theorem original_inhabitants_proof :
  original_inhabitants 5265 = 7800 :=
by
  sorry

end original_inhabitants_proof_l1888_188866


namespace longest_side_of_rectangular_solid_l1888_188864

theorem longest_side_of_rectangular_solid 
  (x y z : ℝ) 
  (h1 : x * y = 20) 
  (h2 : y * z = 15) 
  (h3 : x * z = 12) 
  (h4 : x * y * z = 60) : 
  max (max x y) z = 10 := 
by sorry

end longest_side_of_rectangular_solid_l1888_188864


namespace average_and_fourth_number_l1888_188820

theorem average_and_fourth_number {x : ℝ} (h_avg : ((1 + 2 + 4 + 6 + 9 + 9 + 10 + 12 + x) / 9) = 7) :
  x = 10 ∧ 6 = 6 :=
by
  sorry

end average_and_fourth_number_l1888_188820


namespace intersection_M_N_l1888_188880

open Set Real

def M := {x : ℝ | x^2 < 4}
def N := {x : ℝ | ∃ α : ℝ, x = sin α}
def IntersectSet := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = IntersectSet := by
  sorry

end intersection_M_N_l1888_188880


namespace packs_of_gum_bought_l1888_188898

noncomputable def initial_amount : ℝ := 10.00
noncomputable def gum_cost : ℝ := 1.00
noncomputable def choc_bars : ℝ := 5.00
noncomputable def choc_bar_cost : ℝ := 1.00
noncomputable def candy_canes : ℝ := 2.00
noncomputable def candy_cane_cost : ℝ := 0.50
noncomputable def leftover_amount : ℝ := 1.00

theorem packs_of_gum_bought : (initial_amount - leftover_amount - (choc_bars * choc_bar_cost + candy_canes * candy_cane_cost)) / gum_cost = 3 :=
by
  sorry

end packs_of_gum_bought_l1888_188898


namespace simplify_expression_l1888_188860

theorem simplify_expression : 
  (Real.sqrt 12) + (Real.sqrt 4) * ((Real.sqrt 5 - Real.pi) ^ 0) - (abs (-2 * Real.sqrt 3)) = 2 := 
by 
  sorry

end simplify_expression_l1888_188860


namespace distinct_values_f_in_interval_l1888_188871

noncomputable def f (x : ℝ) : ℤ :=
  ⌊x⌋ + ⌊2 * x⌋ + ⌊(5 * x) / 3⌋ + ⌊3 * x⌋ + ⌊4 * x⌋

theorem distinct_values_f_in_interval : 
  ∃ n : ℕ, n = 734 ∧ 
    ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 100 ∧ 0 ≤ y ∧ y ≤ 100 → 
      f x = f y → x = y :=
sorry

end distinct_values_f_in_interval_l1888_188871


namespace kelly_needs_to_give_away_l1888_188855

-- Definition of initial number of Sony games and desired number of Sony games left
def initial_sony_games : ℕ := 132
def desired_remaining_sony_games : ℕ := 31

-- The main theorem: The number of Sony games Kelly needs to give away to have 31 left
theorem kelly_needs_to_give_away : initial_sony_games - desired_remaining_sony_games = 101 := by
  sorry

end kelly_needs_to_give_away_l1888_188855


namespace find_d_vector_l1888_188891

theorem find_d_vector (x y t : ℝ) (v d : ℝ × ℝ)
  (hline : y = (5 * x - 7) / 2)
  (hparam : ∃ t : ℝ, (x, y) = (4, 2) + t • d)
  (hdist : ∀ {x : ℝ}, x ≥ 4 → dist (x, (5 * x - 7) / 2) (4, 2) = t) :
  d = (2 / Real.sqrt 29, 5 / Real.sqrt 29) := 
sorry

end find_d_vector_l1888_188891


namespace trigonometric_identity_l1888_188810

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  1 + Real.sin α * Real.cos α = 7 / 5 :=
by
  sorry

end trigonometric_identity_l1888_188810


namespace wedge_product_correct_l1888_188851

variables {a1 a2 b1 b2 : ℝ}
def a : ℝ × ℝ := (a1, a2)
def b : ℝ × ℝ := (b1, b2)

def wedge_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.2 - v.2 * w.1

theorem wedge_product_correct (a b : ℝ × ℝ) :
  wedge_product a b = a.1 * b.2 - a.2 * b.1 :=
by
  -- Proof is omitted, theorem statement only
  sorry

end wedge_product_correct_l1888_188851


namespace f_prime_at_pi_over_six_l1888_188885

noncomputable def f (f'_0 : ℝ) (x : ℝ) : ℝ := (1/2)*x^2 + 2*f'_0*(Real.cos x) + x

theorem f_prime_at_pi_over_six (f'_0 : ℝ) (h : f'_0 = 1) :
  (deriv (f f'_0)) (Real.pi / 6) = Real.pi / 6 := by
  sorry

end f_prime_at_pi_over_six_l1888_188885


namespace sum_reciprocals_roots_l1888_188879

theorem sum_reciprocals_roots :
  (∃ p q : ℝ, p + q = 10 ∧ p * q = 3) →
  (∃ p q : ℝ, p ≠ 0 ∧ q ≠ 0 → (1 / p) + (1 / q) = 10 / 3) :=
by
  sorry

end sum_reciprocals_roots_l1888_188879


namespace sum_of_fractions_l1888_188812

-- Definition of the fractions given as conditions
def frac1 := 2 / 10
def frac2 := 4 / 40
def frac3 := 6 / 60
def frac4 := 8 / 30

-- Statement of the theorem to prove
theorem sum_of_fractions : frac1 + frac2 + frac3 + frac4 = 2 / 3 := by
  sorry

end sum_of_fractions_l1888_188812


namespace range_of_x_l1888_188828

theorem range_of_x (x : ℝ) : (x^2 - 9*x + 14 < 0) ∧ (2*x + 3 > 0) ↔ (2 < x) ∧ (x < 7) := 
by 
  sorry

end range_of_x_l1888_188828


namespace find_x_l1888_188814

theorem find_x (P0 P1 P2 P3 P4 P5 : ℝ) (y : ℝ) (h1 : P1 = P0 * 1.10)
                                      (h2 : P2 = P1 * 0.85)
                                      (h3 : P3 = P2 * 1.20)
                                      (h4 : P4 = P3 * (1 - x/100))
                                      (h5 : y = 0.15)
                                      (h6 : P5 = P4 * 1.15)
                                      (h7 : P5 = P0) : x = 23 :=
sorry

end find_x_l1888_188814


namespace weighted_average_inequality_l1888_188806

variable (x y z : ℝ)
variable (h1 : x < y) (h2 : y < z)

theorem weighted_average_inequality :
  (4 * z + x + y) / 6 > (x + y + 2 * z) / 4 :=
by
  sorry

end weighted_average_inequality_l1888_188806


namespace zhijie_suanjing_l1888_188887

theorem zhijie_suanjing :
  ∃ (x y: ℕ), x + y = 100 ∧ 3 * x + y / 3 = 100 :=
by
  sorry

end zhijie_suanjing_l1888_188887


namespace minimize_y_l1888_188894

variable (a b x : ℝ)

def y := (x - a)^2 + (x - b)^2

theorem minimize_y : ∃ x : ℝ, (∀ (x' : ℝ), y x a b ≤ y x' a b) ∧ x = (a + b) / 2 := by
  sorry

end minimize_y_l1888_188894


namespace page_numbers_sum_l1888_188807

theorem page_numbers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 136080) : n + (n + 1) + (n + 2) = 144 :=
by
  sorry

end page_numbers_sum_l1888_188807


namespace repeating_decimals_sum_l1888_188853

theorem repeating_decimals_sum :
  let x := (246 : ℚ) / 999
  let y := (135 : ℚ) / 999
  let z := (579 : ℚ) / 999
  x - y + z = (230 : ℚ) / 333 :=
by
  sorry

end repeating_decimals_sum_l1888_188853


namespace inequality_arith_geo_mean_l1888_188859

variable (a k : ℝ)
variable (h1 : 1 ≤ k)
variable (h2 : k ≤ 3)
variable (h3 : 0 < k)

theorem inequality_arith_geo_mean (h1 : 1 ≤ k) (h2 : k ≤ 3) (h3 : 0 < k):
    ( (a + k * a) / 2 ) ^ 2 ≥ ( (a * (k * a)) ^ (1/2) ) ^ 2 :=
by
  sorry

end inequality_arith_geo_mean_l1888_188859


namespace leak_empty_tank_time_l1888_188823

theorem leak_empty_tank_time (A L : ℝ) (hA : A = 1 / 10) (hAL : A - L = 1 / 15) : (1 / L = 30) :=
sorry

end leak_empty_tank_time_l1888_188823


namespace abs_inequality_solution_l1888_188830

theorem abs_inequality_solution (x : ℝ) : |x - 1| + |x - 3| < 8 ↔ -2 < x ∧ x < 6 :=
by sorry

end abs_inequality_solution_l1888_188830
