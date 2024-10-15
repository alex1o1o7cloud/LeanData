import Mathlib

namespace NUMINAMATH_GPT_aaron_already_had_lids_l1388_138865

-- Definitions for conditions
def number_of_boxes : ℕ := 3
def can_lids_per_box : ℕ := 13
def total_can_lids : ℕ := 53
def lids_from_boxes : ℕ := number_of_boxes * can_lids_per_box

-- The statement to be proven
theorem aaron_already_had_lids : total_can_lids - lids_from_boxes = 14 := 
by
  sorry

end NUMINAMATH_GPT_aaron_already_had_lids_l1388_138865


namespace NUMINAMATH_GPT_simplify_expression_l1388_138815

theorem simplify_expression (a : ℝ) : (2 * a - 3)^2 - (a + 5) * (a - 5) = 3 * a^2 - 12 * a + 34 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1388_138815


namespace NUMINAMATH_GPT_trisha_walked_distance_l1388_138853

theorem trisha_walked_distance :
  ∃ x : ℝ, (x + x + 0.67 = 0.89) ∧ (x = 0.11) :=
by sorry

end NUMINAMATH_GPT_trisha_walked_distance_l1388_138853


namespace NUMINAMATH_GPT_find_asterisk_value_l1388_138880

theorem find_asterisk_value : 
  (∃ x : ℕ, (x / 21) * (x / 189) = 1) → x = 63 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_asterisk_value_l1388_138880


namespace NUMINAMATH_GPT_even_poly_iff_a_zero_l1388_138883

theorem even_poly_iff_a_zero (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + 3) = (x^2 - a*x + 3)) → a = 0 :=
by
  sorry

end NUMINAMATH_GPT_even_poly_iff_a_zero_l1388_138883


namespace NUMINAMATH_GPT_no_equilateral_triangle_on_integer_lattice_l1388_138823

theorem no_equilateral_triangle_on_integer_lattice :
  ∀ (A B C : ℤ × ℤ), 
  A ≠ B → B ≠ C → C ≠ A →
  (dist A B = dist B C ∧ dist B C = dist C A) → 
  false :=
by sorry

end NUMINAMATH_GPT_no_equilateral_triangle_on_integer_lattice_l1388_138823


namespace NUMINAMATH_GPT_total_eyes_in_family_l1388_138806

def mom_eyes := 1
def dad_eyes := 3
def num_kids := 3
def kid_eyes := 4

theorem total_eyes_in_family : mom_eyes + dad_eyes + (num_kids * kid_eyes) = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_eyes_in_family_l1388_138806


namespace NUMINAMATH_GPT_proportion_decrease_l1388_138890

open Real

/-- 
Given \(x\) and \(y\) are directly proportional and positive,
if \(x\) decreases by \(q\%\), then \(y\) decreases by \(q\%\).
-/
theorem proportion_decrease (c x q : ℝ) (h_pos : x > 0) (h_q_pos : q > 0)
    (h_direct : ∀ x y, y = c * x) :
    ((x * (1 - q / 100)) = y) → ((y * (1 - q / 100)) = (c * x * (1 - q / 100))) := by
  sorry

end NUMINAMATH_GPT_proportion_decrease_l1388_138890


namespace NUMINAMATH_GPT_joined_toucans_is_1_l1388_138898

-- Define the number of toucans initially
def initial_toucans : ℕ := 2

-- Define the total number of toucans after some join
def total_toucans : ℕ := 3

-- Define the number of toucans that joined
def toucans_joined : ℕ := total_toucans - initial_toucans

-- State the theorem to prove that 1 toucan joined
theorem joined_toucans_is_1 : toucans_joined = 1 :=
by
  sorry

end NUMINAMATH_GPT_joined_toucans_is_1_l1388_138898


namespace NUMINAMATH_GPT_expression_never_prime_l1388_138804

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem expression_never_prime (n : ℕ) (h : is_prime n) : ¬is_prime (n^2 + 75) :=
sorry

end NUMINAMATH_GPT_expression_never_prime_l1388_138804


namespace NUMINAMATH_GPT_manager_salary_is_3600_l1388_138825

noncomputable def manager_salary (M : ℕ) : ℕ :=
  let total_salary_20 := 20 * 1500
  let new_average_salary := 1600
  let total_salary_21 := 21 * new_average_salary
  total_salary_21 - total_salary_20

theorem manager_salary_is_3600 : manager_salary 3600 = 3600 := by
  sorry

end NUMINAMATH_GPT_manager_salary_is_3600_l1388_138825


namespace NUMINAMATH_GPT_points_on_opposite_sides_l1388_138873

-- Definitions and the conditions written to Lean
def satisfies_A (a x y : ℝ) : Prop :=
  5 * a^2 - 6 * a * x - 2 * a * y + 2 * x^2 + 2 * x * y + y^2 = 0

def satisfies_B (a x y : ℝ) : Prop :=
  a^2 * x^2 + a^2 * y^2 - 8 * a^2 * x - 2 * a^3 * y + 12 * a * y + a^4 + 36 = 0

def opposite_sides_of_line (y_A y_B : ℝ) : Prop :=
  (y_A - 1) * (y_B - 1) < 0

theorem points_on_opposite_sides (a : ℝ) (x_A y_A x_B y_B : ℝ) :
  satisfies_A a x_A y_A →
  satisfies_B a x_B y_B →
  -2 > a ∨ (-1 < a ∧ a < 0) ∨ 3 < a →
  opposite_sides_of_line y_A y_B → 
  x_A = 2 * a ∧ y_A = -a ∧ x_B = 4 ∧ y_B = a - 6/a :=
sorry

end NUMINAMATH_GPT_points_on_opposite_sides_l1388_138873


namespace NUMINAMATH_GPT_area_of_smaller_circle_l1388_138858

noncomputable def radius_smaller_circle : ℝ := sorry
noncomputable def radius_larger_circle : ℝ := 3 * radius_smaller_circle

-- Given: PA = AB = 5
def PA : ℝ := 5
def AB : ℝ := 5

-- Final goal: The area of the smaller circle is 5/3 * π
theorem area_of_smaller_circle (r_s : ℝ) (rsq : r_s^2 = 5 / 3) : (π * r_s^2 = 5/3 * π) :=
by
  exact sorry

end NUMINAMATH_GPT_area_of_smaller_circle_l1388_138858


namespace NUMINAMATH_GPT_find_V_l1388_138856

theorem find_V 
  (c : ℝ)
  (R₁ V₁ W₁ R₂ W₂ V₂ : ℝ)
  (h1 : R₁ = c * (V₁ / W₁))
  (h2 : R₁ = 6)
  (h3 : V₁ = 2)
  (h4 : W₁ = 3)
  (h5 : R₂ = 25)
  (h6 : W₂ = 5)
  (h7 : V₂ = R₂ * W₂ / 9) :
  V₂ = 125 / 9 :=
by sorry

end NUMINAMATH_GPT_find_V_l1388_138856


namespace NUMINAMATH_GPT_minimum_score_for_advanced_course_l1388_138897

theorem minimum_score_for_advanced_course (q1 q2 q3 q4 : ℕ) (H1 : q1 = 88) (H2 : q2 = 84) (H3 : q3 = 82) :
  (q1 + q2 + q3 + q4) / 4 ≥ 85 → q4 = 86 := by
  sorry

end NUMINAMATH_GPT_minimum_score_for_advanced_course_l1388_138897


namespace NUMINAMATH_GPT_factorize_x_squared_minus_one_l1388_138864

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_x_squared_minus_one_l1388_138864


namespace NUMINAMATH_GPT_functional_equation_solution_l1388_138884

theorem functional_equation_solution (f : ℝ → ℝ) (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  (f x = (1/2) * (x + 1 - 1/x - 1/(1-x))) →
  (f x + f (1 / (1 - x)) = x) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l1388_138884


namespace NUMINAMATH_GPT_AllieMoreGrapes_l1388_138882

-- Definitions based on conditions
def RobBowl : ℕ := 25
def TotalGrapes : ℕ := 83
def AllynBowl (A : ℕ) : ℕ := A + 4

-- The proof statement that must be shown.
theorem AllieMoreGrapes (A : ℕ) (h1 : A + (AllynBowl A) + RobBowl = TotalGrapes) : A - RobBowl = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_AllieMoreGrapes_l1388_138882


namespace NUMINAMATH_GPT_work_rates_l1388_138877

theorem work_rates (A B : ℝ) (combined_days : ℝ) (b_rate: B = 35) 
(combined_rate: combined_days = 20 / 11):
    A = 700 / 365 :=
by
  have h1 : B = 35 := by sorry
  have h2 : combined_days = 20 / 11 := by sorry
  have : 1/A + 1/B = 11/20 := by sorry
  have : 1/A = 11/20 - 1/B := by sorry
  have : 1/A =  365 / 700:= by sorry
  have : A = 700 / 365 := by sorry
  assumption

end NUMINAMATH_GPT_work_rates_l1388_138877


namespace NUMINAMATH_GPT_shiela_neighbors_l1388_138885

theorem shiela_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) (neighbors : ℕ) 
  (h1 : total_drawings = 54) (h2 : drawings_per_neighbor = 9) : neighbors = total_drawings / drawings_per_neighbor :=
by
  rw [h1, h2]
  norm_num
  exact sorry

end NUMINAMATH_GPT_shiela_neighbors_l1388_138885


namespace NUMINAMATH_GPT_complement_intersection_l1388_138848

-- Defining the universal set U and subsets A and B
def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {2, 3, 4}
def B : Finset ℕ := {3, 4, 5}

-- Proving the complement of the intersection of A and B in U
theorem complement_intersection : (U \ (A ∩ B)) = {1, 2, 5} :=
by sorry

end NUMINAMATH_GPT_complement_intersection_l1388_138848


namespace NUMINAMATH_GPT_building_height_l1388_138889

-- Definitions of the conditions
def wooden_box_height : ℝ := 3
def wooden_box_shadow : ℝ := 12
def building_shadow : ℝ := 36

-- The statement that needs to be proved
theorem building_height : ∃ (height : ℝ), height = 9 ∧ wooden_box_height / wooden_box_shadow = height / building_shadow :=
by
  sorry

end NUMINAMATH_GPT_building_height_l1388_138889


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1388_138843

-- Statement to prove that the line always passes through the point (2, 2)
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, ∃ x y : ℝ, 
  (1 + 4 * k) * x - (2 - 3 * k) * y + (2 - 14 * k) = 0 ∧ x = 2 ∧ y = 2 :=
sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1388_138843


namespace NUMINAMATH_GPT_susan_added_oranges_l1388_138803

-- Conditions as definitions
def initial_oranges_in_box : ℝ := 55.0
def final_oranges_in_box : ℝ := 90.0

-- Define the quantity of oranges Susan put into the box
def susan_oranges := final_oranges_in_box - initial_oranges_in_box

-- Theorem statement to prove that the number of oranges Susan put into the box is 35.0
theorem susan_added_oranges : susan_oranges = 35.0 := by
  unfold susan_oranges
  sorry

end NUMINAMATH_GPT_susan_added_oranges_l1388_138803


namespace NUMINAMATH_GPT_power_modulo_l1388_138854

theorem power_modulo {a : ℤ} : a^561 ≡ a [ZMOD 561] :=
sorry

end NUMINAMATH_GPT_power_modulo_l1388_138854


namespace NUMINAMATH_GPT_derivative_at_neg_one_l1388_138839

theorem derivative_at_neg_one (a b c : ℝ) (h : (4*a*(1:ℝ)^3 + 2*b*(1:ℝ)) = 2) :
  (4*a*(-1:ℝ)^3 + 2*b*(-1:ℝ)) = -2 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_neg_one_l1388_138839


namespace NUMINAMATH_GPT_alpha_beta_roots_l1388_138866

theorem alpha_beta_roots (α β : ℝ) (hαβ1 : α^2 + α - 1 = 0) (hαβ2 : β^2 + β - 1 = 0) (h_sum : α + β = -1) :
  α^4 - 3 * β = 5 :=
by
  sorry

end NUMINAMATH_GPT_alpha_beta_roots_l1388_138866


namespace NUMINAMATH_GPT_derivative_of_y_l1388_138888

open Real

noncomputable def y (x : ℝ) : ℝ := (cos (log 7) * (sin (7 * x)) ^ 2) / (7 * cos (14 * x))

theorem derivative_of_y (x : ℝ) : deriv y x = (cos (log 7) * tan (14 * x)) / cos (14 * x) := sorry

end NUMINAMATH_GPT_derivative_of_y_l1388_138888


namespace NUMINAMATH_GPT_num_solutions_in_interval_l1388_138831

theorem num_solutions_in_interval : 
  ∃ n : ℕ, n = 2 ∧ ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
  2 ^ Real.cos θ = Real.sin θ → n = 2 := 
sorry

end NUMINAMATH_GPT_num_solutions_in_interval_l1388_138831


namespace NUMINAMATH_GPT_volume_relationship_l1388_138828

open Real

theorem volume_relationship (r : ℝ) (A M C : ℝ)
  (hA : A = (1/3) * π * r^3)
  (hM : M = π * r^3)
  (hC : C = (4/3) * π * r^3) :
  A + M + (1/2) * C = 2 * π * r^3 :=
by
  sorry

end NUMINAMATH_GPT_volume_relationship_l1388_138828


namespace NUMINAMATH_GPT_determine_k_for_quadratic_eq_l1388_138845

theorem determine_k_for_quadratic_eq {k : ℝ} :
  (∀ r s : ℝ, 3 * r^2 + 5 * r + k = 0 ∧ 3 * s^2 + 5 * s + k = 0 →
    (|r + s| = r^2 + s^2)) ↔ k = -10/3 := by
sorry

end NUMINAMATH_GPT_determine_k_for_quadratic_eq_l1388_138845


namespace NUMINAMATH_GPT_factorize_polynomial_find_value_l1388_138887

-- Problem 1: Factorize a^3 - 3a^2 - 4a + 12
theorem factorize_polynomial (a : ℝ) :
  a^3 - 3 * a^2 - 4 * a + 12 = (a - 3) * (a - 2) * (a + 2) :=
sorry

-- Problem 2: Given m + n = 5 and m - n = 1, prove m^2 - n^2 + 2m - 2n = 7
theorem find_value (m n : ℝ) (h1 : m + n = 5) (h2 : m - n = 1) :
  m^2 - n^2 + 2 * m - 2 * n = 7 :=
sorry

end NUMINAMATH_GPT_factorize_polynomial_find_value_l1388_138887


namespace NUMINAMATH_GPT_one_div_m_plus_one_div_n_l1388_138816

theorem one_div_m_plus_one_div_n
  {m n : ℕ} 
  (h1 : Nat.gcd m n = 5) 
  (h2 : Nat.lcm m n = 210)
  (h3 : m + n = 75) :
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 14 :=
by
  sorry

end NUMINAMATH_GPT_one_div_m_plus_one_div_n_l1388_138816


namespace NUMINAMATH_GPT_range_of_m_tangent_not_parallel_l1388_138868

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := (1 / 2) * x^2 - k * x
noncomputable def h (x : ℝ) (m : ℝ) : ℝ := f x + g x (m + (1 / m))
noncomputable def M (x : ℝ) (m : ℝ) : ℝ := f x - g x (m + (1 / m))

theorem range_of_m (m : ℝ) (h_extreme : ∃ x ∈ Set.Ioo 0 2, ∀ y ∈ Set.Ioo 0 2, h y m ≤ h x m) : 
  (0 < m ∧ m ≤ 1 / 2) ∨ (m ≥ 2) :=
  sorry

theorem tangent_not_parallel (x1 x2 x0 : ℝ) (m : ℝ) (h_zeros : M x1 m = 0 ∧ M x2 m = 0 ∧ x1 > x2 ∧ 2 * x0 = x1 + x2) :
  ¬ (∃ l : ℝ, ∀ x : ℝ, M x m = l * (x - x0) + M x0 m ∧ l = 0) :=
  sorry

end NUMINAMATH_GPT_range_of_m_tangent_not_parallel_l1388_138868


namespace NUMINAMATH_GPT_cylinder_radius_unique_l1388_138850

theorem cylinder_radius_unique
  (r : ℝ) (h : ℝ) (V : ℝ) (y : ℝ)
  (h_eq : h = 2)
  (V_eq : V = 2 * Real.pi * r ^ 2)
  (y_eq_increase_radius : y = 2 * Real.pi * ((r + 6) ^ 2 - r ^ 2))
  (y_eq_increase_height : y = 6 * Real.pi * r ^ 2) :
  r = 6 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_radius_unique_l1388_138850


namespace NUMINAMATH_GPT_number_of_books_l1388_138881

theorem number_of_books (original_books new_books : ℕ) (h1 : original_books = 35) (h2 : new_books = 56) : 
  original_books + new_books = 91 :=
by {
  -- the proof will go here, but is not required for the statement
  sorry
}

end NUMINAMATH_GPT_number_of_books_l1388_138881


namespace NUMINAMATH_GPT_work_efficiency_ratio_l1388_138861

variable (A B : ℝ)
variable (h1 : A = 1 / 2 * B) 
variable (h2 : 1 / (A + B) = 13)
variable (h3 : B = 1 / 19.5)

theorem work_efficiency_ratio : A / B = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_work_efficiency_ratio_l1388_138861


namespace NUMINAMATH_GPT_fruit_punch_total_l1388_138894

section fruit_punch
variable (orange_punch : ℝ) (cherry_punch : ℝ) (apple_juice : ℝ) (total_punch : ℝ)

axiom h1 : orange_punch = 4.5
axiom h2 : cherry_punch = 2 * orange_punch
axiom h3 : apple_juice = cherry_punch - 1.5
axiom h4 : total_punch = orange_punch + cherry_punch + apple_juice

theorem fruit_punch_total : total_punch = 21 := sorry

end fruit_punch

end NUMINAMATH_GPT_fruit_punch_total_l1388_138894


namespace NUMINAMATH_GPT_find_number_l1388_138847

theorem find_number :
  ∃ n : ℤ,
    (n % 12 = 11) ∧ 
    (n % 11 = 10) ∧ 
    (n % 10 = 9) ∧ 
    (n % 9 = 8) ∧ 
    (n % 8 = 7) ∧ 
    (n % 7 = 6) ∧ 
    (n % 6 = 5) ∧ 
    (n % 5 = 4) ∧ 
    (n % 4 = 3) ∧ 
    (n % 3 = 2) ∧ 
    (n % 2 = 1) ∧
    n = 27719 :=
sorry

end NUMINAMATH_GPT_find_number_l1388_138847


namespace NUMINAMATH_GPT_find_stamps_l1388_138862

def stamps_problem (x y : ℕ) : Prop :=
  (x + y = 70) ∧ (y = 4 * x + 5)

theorem find_stamps (x y : ℕ) (h : stamps_problem x y) : 
  x = 13 ∧ y = 57 :=
sorry

end NUMINAMATH_GPT_find_stamps_l1388_138862


namespace NUMINAMATH_GPT_june_earnings_l1388_138886

theorem june_earnings (total_clovers : ℕ) (percent_three : ℝ) (percent_two : ℝ) (percent_four : ℝ) :
  total_clovers = 200 →
  percent_three = 0.75 →
  percent_two = 0.24 →
  percent_four = 0.01 →
  (total_clovers * percent_three + total_clovers * percent_two + total_clovers * percent_four) = 200 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_june_earnings_l1388_138886


namespace NUMINAMATH_GPT_tank_full_time_l1388_138844

def tank_capacity : ℕ := 900
def fill_rate_A : ℕ := 40
def fill_rate_B : ℕ := 30
def drain_rate_C : ℕ := 20
def cycle_time : ℕ := 3
def net_fill_per_cycle : ℕ := fill_rate_A + fill_rate_B - drain_rate_C

theorem tank_full_time :
  (tank_capacity / net_fill_per_cycle) * cycle_time = 54 :=
by
  sorry

end NUMINAMATH_GPT_tank_full_time_l1388_138844


namespace NUMINAMATH_GPT_pau_total_ordered_correct_l1388_138809

-- Define the initial pieces of fried chicken ordered by Kobe
def kobe_order : ℝ := 5

-- Define Pau's initial order as twice Kobe's order plus 2.5 pieces
def pau_initial_order : ℝ := (2 * kobe_order) + 2.5

-- Define Shaquille's initial order as 50% more than Pau's initial order
def shaq_initial_order : ℝ := pau_initial_order * 1.5

-- Define the total pieces of chicken Pau will have eaten by the end
def pau_total_ordered : ℝ := 2 * pau_initial_order

-- Prove that Pau will have eaten 25 pieces of fried chicken by the end
theorem pau_total_ordered_correct : pau_total_ordered = 25 := by
  sorry

end NUMINAMATH_GPT_pau_total_ordered_correct_l1388_138809


namespace NUMINAMATH_GPT_goose_price_remains_affordable_l1388_138879

theorem goose_price_remains_affordable :
  ∀ (h v : ℝ),
  h + v = 1 →
  h + (v / 2) = 1 →
  h * 1.2 ≤ 1 :=
by
  intros h v h_eq v_eq
  /- Proof will go here -/
  sorry

end NUMINAMATH_GPT_goose_price_remains_affordable_l1388_138879


namespace NUMINAMATH_GPT_calculate_expression_l1388_138899

theorem calculate_expression : (8^5 / 8^2) * 3^6 = 373248 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1388_138899


namespace NUMINAMATH_GPT_piggy_bank_dimes_diff_l1388_138892

theorem piggy_bank_dimes_diff :
  ∃ (a b c : ℕ), a + b + c = 100 ∧ 5 * a + 10 * b + 25 * c = 1005 ∧ (∀ lo hi, 
  (lo = 1 ∧ hi = 101) → (hi - lo = 100)) :=
by
  sorry

end NUMINAMATH_GPT_piggy_bank_dimes_diff_l1388_138892


namespace NUMINAMATH_GPT_no_solution_l1388_138869

theorem no_solution (n : ℕ) (x y k : ℕ) (h1 : n ≥ 1) (h2 : x > 0) (h3 : y > 0) (h4 : k > 1) (h5 : Nat.gcd x y = 1) (h6 : 3^n = x^k + y^k) : False :=
by
  sorry

end NUMINAMATH_GPT_no_solution_l1388_138869


namespace NUMINAMATH_GPT_common_ratio_of_gp_l1388_138859

variable (r : ℝ)(n : ℕ)

theorem common_ratio_of_gp (h1 : 9 * r ^ (n - 1) = 1/3) 
                           (h2 : 9 * (1 - r ^ n) / (1 - r) = 40 / 3) : 
                           r = 1/3 := 
sorry

end NUMINAMATH_GPT_common_ratio_of_gp_l1388_138859


namespace NUMINAMATH_GPT_sum_of_eggs_is_3712_l1388_138826

-- Definitions based on the conditions
def eggs_yesterday : ℕ := 1925
def eggs_fewer_today : ℕ := 138
def eggs_today : ℕ := eggs_yesterday - eggs_fewer_today

-- Theorem stating the equivalence of the sum of eggs
theorem sum_of_eggs_is_3712 : eggs_yesterday + eggs_today = 3712 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_eggs_is_3712_l1388_138826


namespace NUMINAMATH_GPT_alan_total_spending_l1388_138819

-- Define the conditions
def eggs_bought : ℕ := 20
def price_per_egg : ℕ := 2
def chickens_bought : ℕ := 6
def price_per_chicken : ℕ := 8

-- Total cost calculation
def cost_eggs : ℕ := eggs_bought * price_per_egg
def cost_chickens : ℕ := chickens_bought * price_per_chicken
def total_amount_spent : ℕ := cost_eggs + cost_chickens

-- Prove the total amount spent
theorem alan_total_spending : total_amount_spent = 88 := by
  show cost_eggs + cost_chickens = 88
  sorry

end NUMINAMATH_GPT_alan_total_spending_l1388_138819


namespace NUMINAMATH_GPT_kmph_to_mps_l1388_138810

theorem kmph_to_mps (s : ℝ) (h : s = 0.975) : s * (1000 / 3600) = 0.2708 := by
  -- We include the assumption s = 0.975 as part of the problem condition.
  -- Import Mathlib to gain access to real number arithmetic.
  -- sorry is added to indicate a place where the proof should go.
  sorry

end NUMINAMATH_GPT_kmph_to_mps_l1388_138810


namespace NUMINAMATH_GPT_find_t_l1388_138863

theorem find_t (p q r s t : ℤ)
  (h₁ : p - q - r + s - t = -t)
  (h₂ : p - (q - (r - (s - t))) = -4 + t) :
  t = 2 := 
sorry

end NUMINAMATH_GPT_find_t_l1388_138863


namespace NUMINAMATH_GPT_expectation_of_xi_l1388_138807

noncomputable def compute_expectation : ℝ := 
  let m : ℝ := 0.3
  let E : ℝ := (1 * 0.5) + (3 * m) + (5 * 0.2)
  E

theorem expectation_of_xi :
  let m: ℝ := 1 - 0.5 - 0.2 
  (0.5 + m + 0.2 = 1) → compute_expectation = 2.4 := 
by
  sorry

end NUMINAMATH_GPT_expectation_of_xi_l1388_138807


namespace NUMINAMATH_GPT_fraction_of_buttons_l1388_138852

variable (K S M : ℕ)  -- Kendra's buttons, Sue's buttons, Mari's buttons

theorem fraction_of_buttons (H1 : M = 5 * K + 4) 
                            (H2 : S = 6)
                            (H3 : M = 64) :
  S / K = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_fraction_of_buttons_l1388_138852


namespace NUMINAMATH_GPT_product_is_even_l1388_138812

theorem product_is_even (a b c : ℤ) : Even ((a - b) * (b - c) * (c - a)) := by
  sorry

end NUMINAMATH_GPT_product_is_even_l1388_138812


namespace NUMINAMATH_GPT_relation_between_u_and_v_l1388_138800

def diameter_circle_condition (AB : ℝ) (r : ℝ) : Prop := AB = 2*r
def chord_tangent_condition (AD BC CD : ℝ) (r : ℝ) : Prop := 
  AD + BC = 2*r ∧ CD*CD = (2*r)*(AD + BC)
def point_selection_condition (AD AF CD : ℝ) : Prop := AD = AF + CD

theorem relation_between_u_and_v (AB AD AF BC CD u v r: ℝ)
  (h1: diameter_circle_condition AB r)
  (h2: chord_tangent_condition AD BC CD r)
  (h3: point_selection_condition AD AF CD)
  (h4: u = AF)
  (h5: v^2 = r^2):
  v^2 = u^3 / (2*r - u) := by
  sorry

end NUMINAMATH_GPT_relation_between_u_and_v_l1388_138800


namespace NUMINAMATH_GPT_sheep_count_l1388_138860

-- Define the conditions
def TotalAnimals : ℕ := 200
def NumberCows : ℕ := 40
def NumberGoats : ℕ := 104

-- Define the question and its corresponding answer
def NumberSheep : ℕ := TotalAnimals - (NumberCows + NumberGoats)

-- State the theorem
theorem sheep_count : NumberSheep = 56 := by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_sheep_count_l1388_138860


namespace NUMINAMATH_GPT_gear_B_turns_l1388_138834

theorem gear_B_turns (teeth_A teeth_B turns_A: ℕ) (h₁: teeth_A = 6) (h₂: teeth_B = 8) (h₃: turns_A = 12) :
(turn_A * teeth_A) / teeth_B = 9 :=
by  sorry

end NUMINAMATH_GPT_gear_B_turns_l1388_138834


namespace NUMINAMATH_GPT_total_pennies_after_addition_l1388_138867

def initial_pennies_per_compartment : ℕ := 10
def compartments : ℕ := 20
def added_pennies_per_compartment : ℕ := 15

theorem total_pennies_after_addition :
  (initial_pennies_per_compartment + added_pennies_per_compartment) * compartments = 500 :=
by 
  sorry

end NUMINAMATH_GPT_total_pennies_after_addition_l1388_138867


namespace NUMINAMATH_GPT_root_expression_value_l1388_138805

theorem root_expression_value (p q r : ℝ) (hpq : p + q + r = 15) (hpqr : p * q + q * r + r * p = 25) (hpqrs : p * q * r = 10) :
  (p / (2 / p + q * r) + q / (2 / q + r * p) + r / (2 / r + p * q) = 175 / 12) :=
by sorry

end NUMINAMATH_GPT_root_expression_value_l1388_138805


namespace NUMINAMATH_GPT_necessary_not_sufficient_for_circle_l1388_138841

theorem necessary_not_sufficient_for_circle (a : ℝ) :
  (a ≤ 2 → (x^2 + y^2 - 2*x + 2*y + a = 0 → ∃ r : ℝ, r > 0)) ∧
  (a ≤ 2 ∧ ∃ b, b < 2 → a = b) := sorry

end NUMINAMATH_GPT_necessary_not_sufficient_for_circle_l1388_138841


namespace NUMINAMATH_GPT_cid_earnings_l1388_138851

theorem cid_earnings :
  let model_a_oil_change_cost := 20
  let model_a_repair_cost := 30
  let model_a_wash_cost := 5
  let model_b_oil_change_cost := 25
  let model_b_repair_cost := 40
  let model_b_wash_cost := 8
  let model_c_oil_change_cost := 30
  let model_c_repair_cost := 50
  let model_c_wash_cost := 10

  let model_a_oil_changes := 5
  let model_a_repairs := 10
  let model_a_washes := 15
  let model_b_oil_changes := 3
  let model_b_repairs := 4
  let model_b_washes := 10
  let model_c_oil_changes := 2
  let model_c_repairs := 6
  let model_c_washes := 5

  let total_earnings := 
      (model_a_oil_change_cost * model_a_oil_changes) +
      (model_a_repair_cost * model_a_repairs) +
      (model_a_wash_cost * model_a_washes) +
      (model_b_oil_change_cost * model_b_oil_changes) +
      (model_b_repair_cost * model_b_repairs) +
      (model_b_wash_cost * model_b_washes) +
      (model_c_oil_change_cost * model_c_oil_changes) +
      (model_c_repair_cost * model_c_repairs) +
      (model_c_wash_cost * model_c_washes)

  total_earnings = 1200 := by
  sorry

end NUMINAMATH_GPT_cid_earnings_l1388_138851


namespace NUMINAMATH_GPT_is_equilateral_l1388_138808

open Complex

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry

-- Assume the conditions of the problem
axiom z1_distinct_z2 : z1 ≠ z2
axiom z2_distinct_z3 : z2 ≠ z3
axiom z3_distinct_z1 : z3 ≠ z1
axiom z1_unit_circle : abs z1 = 1
axiom z2_unit_circle : abs z2 = 1
axiom z3_unit_circle : abs z3 = 1
axiom condition : (1 / (2 + abs (z1 + z2)) + 1 / (2 + abs (z2 + z3)) + 1 / (2 + abs (z3 + z1))) = 1
axiom acute_angled_triangle : sorry

theorem is_equilateral (A B C : ℂ) (hA : A = z1) (hB : B = z2) (hC : C = z3) : 
  (sorry : Prop) := sorry

end NUMINAMATH_GPT_is_equilateral_l1388_138808


namespace NUMINAMATH_GPT_Yoongi_score_is_53_l1388_138874

-- Define the scores of the three students
variables (score_Yoongi score_Eunji score_Yuna : ℕ)

-- Define the conditions given in the problem
axiom Yoongi_Eunji : score_Eunji = score_Yoongi - 25
axiom Eunji_Yuna  : score_Yuna = score_Eunji - 20
axiom Yuna_score  : score_Yuna = 8

theorem Yoongi_score_is_53 : score_Yoongi = 53 := by
  sorry

end NUMINAMATH_GPT_Yoongi_score_is_53_l1388_138874


namespace NUMINAMATH_GPT_quadrilateral_perimeter_l1388_138840

-- Define the basic conditions
variables (a b : ℝ)

-- Let's define what happens when Xiao Ming selected 2 pieces of type A, 7 pieces of type B, and 3 pieces of type C
theorem quadrilateral_perimeter (a b : ℝ) : 2 * (a + 3 * b + 2 * a + b) = 6 * a + 8 * b :=
by sorry

end NUMINAMATH_GPT_quadrilateral_perimeter_l1388_138840


namespace NUMINAMATH_GPT_standard_deviation_of_data_l1388_138893

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  let m := mean data
  (data.map (fun x => (x - m)^2)).sum / data.length

noncomputable def std_dev (data : List ℝ) : ℝ :=
  Real.sqrt (variance data)

theorem standard_deviation_of_data :
  std_dev [5, 7, 7, 8, 10, 11] = 2 := 
sorry

end NUMINAMATH_GPT_standard_deviation_of_data_l1388_138893


namespace NUMINAMATH_GPT_ratio_of_sam_to_sue_l1388_138895

-- Definitions
def Sam_age (S : ℕ) : Prop := 3 * S = 18
def Kendra_age (K : ℕ) : Prop := K = 18
def total_age_in_3_years (S U K : ℕ) : Prop := (S + 3) + (U + 3) + (K + 3) = 36

-- Theorem statement
theorem ratio_of_sam_to_sue (S U K : ℕ) (h1 : Sam_age S) (h2 : Kendra_age K) (h3 : total_age_in_3_years S U K) :
  S / U = 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_sam_to_sue_l1388_138895


namespace NUMINAMATH_GPT_pelican_speed_l1388_138818

theorem pelican_speed
  (eagle_speed falcon_speed hummingbird_speed total_distance time : ℕ)
  (eagle_distance falcon_distance hummingbird_distance : ℕ)
  (H1 : eagle_speed = 15)
  (H2 : falcon_speed = 46)
  (H3 : hummingbird_speed = 30)
  (H4 : time = 2)
  (H5 : total_distance = 248)
  (H6 : eagle_distance = eagle_speed * time)
  (H7 : falcon_distance = falcon_speed * time)
  (H8 : hummingbird_distance = hummingbird_speed * time)
  (total_other_birds_distance : ℕ)
  (H9 : total_other_birds_distance = eagle_distance + falcon_distance + hummingbird_distance)
  (pelican_distance : ℕ)
  (H10 : pelican_distance = total_distance - total_other_birds_distance)
  (pelican_speed : ℕ)
  (H11 : pelican_speed = pelican_distance / time) :
  pelican_speed = 33 := 
  sorry

end NUMINAMATH_GPT_pelican_speed_l1388_138818


namespace NUMINAMATH_GPT_product_of_solutions_l1388_138842

-- Definitions based on given conditions
def equation (x : ℝ) : Prop := |x| = 3 * (|x| - 2)

-- Statement of the proof problem
theorem product_of_solutions : ∃ x1 x2 : ℝ, equation x1 ∧ equation x2 ∧ x1 * x2 = -9 := by
  sorry

end NUMINAMATH_GPT_product_of_solutions_l1388_138842


namespace NUMINAMATH_GPT_vasya_fraction_is_0_4_l1388_138830

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end NUMINAMATH_GPT_vasya_fraction_is_0_4_l1388_138830


namespace NUMINAMATH_GPT_volume_of_cube_l1388_138870

theorem volume_of_cube (a : ℕ) (h : ((a - 2) * a * (a + 2)) = a^3 - 16) : a^3 = 64 :=
sorry

end NUMINAMATH_GPT_volume_of_cube_l1388_138870


namespace NUMINAMATH_GPT_least_number_to_subtract_l1388_138813

theorem least_number_to_subtract (n : ℕ) (k : ℕ) (h : 1387 = n + k * 15) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l1388_138813


namespace NUMINAMATH_GPT_find_paycheck_l1388_138855

variable (P : ℝ) -- P represents the paycheck amount

def initial_balance : ℝ := 800
def rent_payment : ℝ := 450
def electricity_bill : ℝ := 117
def internet_bill : ℝ := 100
def phone_bill : ℝ := 70
def final_balance : ℝ := 1563

theorem find_paycheck :
  initial_balance - rent_payment + P - (electricity_bill + internet_bill) - phone_bill = final_balance → 
    P = 1563 :=
by
  sorry

end NUMINAMATH_GPT_find_paycheck_l1388_138855


namespace NUMINAMATH_GPT_interest_equality_l1388_138827

theorem interest_equality (total_sum : ℝ) (part1 : ℝ) (part2 : ℝ) (rate1 : ℝ) (time1 : ℝ) (rate2 : ℝ) (n : ℝ) :
  total_sum = 2730 ∧ part1 = 1050 ∧ part2 = 1680 ∧
  rate1 = 3 ∧ time1 = 8 ∧ rate2 = 5 ∧ part1 * rate1 * time1 = part2 * rate2 * n →
  n = 3 :=
by
  sorry

end NUMINAMATH_GPT_interest_equality_l1388_138827


namespace NUMINAMATH_GPT_train_speed_168_l1388_138802

noncomputable def speed_of_train (L : ℕ) (V_man : ℕ) (T : ℕ) : ℚ :=
  let V_man_mps := (V_man * 5) / 18
  let relative_speed := L / T
  let V_train_mps := relative_speed - V_man_mps
  V_train_mps * (18 / 5)

theorem train_speed_168 :
  speed_of_train 500 12 10 = 168 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_168_l1388_138802


namespace NUMINAMATH_GPT_man_l1388_138829

variable (v : ℝ) (speed_with_current : ℝ) (speed_of_current : ℝ)

theorem man's_speed_against_current :
  speed_with_current = 12 ∧ speed_of_current = 2 → v - speed_of_current = 8 :=
by
  sorry

end NUMINAMATH_GPT_man_l1388_138829


namespace NUMINAMATH_GPT_arccos_one_over_sqrt_two_l1388_138832

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_arccos_one_over_sqrt_two_l1388_138832


namespace NUMINAMATH_GPT_abs_neg_two_thirds_l1388_138811

theorem abs_neg_two_thirds : abs (-2/3 : ℝ) = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_two_thirds_l1388_138811


namespace NUMINAMATH_GPT_revenue_from_full_price_tickets_l1388_138835

theorem revenue_from_full_price_tickets (f h p : ℕ) (H1 : f + h = 150) (H2 : f * p + h * (p / 2) = 2450) : 
  f * p = 1150 :=
by 
  sorry

end NUMINAMATH_GPT_revenue_from_full_price_tickets_l1388_138835


namespace NUMINAMATH_GPT_interval_sum_l1388_138875

theorem interval_sum (a b : ℝ) (h : ∀ x,  |3 * x - 80| ≤ |2 * x - 105| ↔ (a ≤ x ∧ x ≤ b)) :
  a + b = 12 :=
sorry

end NUMINAMATH_GPT_interval_sum_l1388_138875


namespace NUMINAMATH_GPT_intersect_in_third_quadrant_l1388_138876

theorem intersect_in_third_quadrant (b : ℝ) : (¬ (∃ x y : ℝ, y = 2*x + 1 ∧ y = 3*x + b ∧ x < 0 ∧ y < 0)) ↔ b > 3 / 2 := sorry

end NUMINAMATH_GPT_intersect_in_third_quadrant_l1388_138876


namespace NUMINAMATH_GPT_ben_final_amount_l1388_138896

-- Definition of the conditions
def daily_start := 50
def daily_spent := 15
def daily_saving := daily_start - daily_spent
def days := 7
def mom_double (s : ℕ) := 2 * s
def dad_addition := 10

-- Total amount calculation based on the conditions
noncomputable def total_savings := daily_saving * days
noncomputable def after_mom := mom_double total_savings
noncomputable def total_amount := after_mom + dad_addition

-- The final theorem to prove Ben's final amount is $500 after the given conditions
theorem ben_final_amount : total_amount = 500 :=
by sorry

end NUMINAMATH_GPT_ben_final_amount_l1388_138896


namespace NUMINAMATH_GPT_files_rem_nat_eq_two_l1388_138838

-- Conditions
def initial_music_files : ℕ := 4
def initial_video_files : ℕ := 21
def files_deleted : ℕ := 23

-- Correct Answer
def files_remaining : ℕ := initial_music_files + initial_video_files - files_deleted

theorem files_rem_nat_eq_two : files_remaining = 2 := by
  sorry

end NUMINAMATH_GPT_files_rem_nat_eq_two_l1388_138838


namespace NUMINAMATH_GPT_division_example_l1388_138849

theorem division_example : 72 / (6 / 3) = 36 :=
by sorry

end NUMINAMATH_GPT_division_example_l1388_138849


namespace NUMINAMATH_GPT_water_fraction_final_l1388_138836

noncomputable def initial_water_volume : ℚ := 25
noncomputable def first_removal_water : ℚ := 5
noncomputable def first_add_antifreeze : ℚ := 5
noncomputable def first_water_fraction : ℚ := (initial_water_volume - first_removal_water) / initial_water_volume

noncomputable def second_removal_fraction : ℚ := 5 / initial_water_volume
noncomputable def second_water_fraction : ℚ := (initial_water_volume - first_removal_water - second_removal_fraction * (initial_water_volume - first_removal_water)) / initial_water_volume

noncomputable def third_removal_fraction : ℚ := 5 / initial_water_volume
noncomputable def third_water_fraction := (second_water_fraction * (initial_water_volume - 5) + 2) / initial_water_volume

theorem water_fraction_final :
  third_water_fraction = 14.8 / 25 := sorry

end NUMINAMATH_GPT_water_fraction_final_l1388_138836


namespace NUMINAMATH_GPT_range_of_a_l1388_138878

noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ :=
  if n <= 7 then (3 - a) * n - 3 else a ^ (n - 6)

def increasing_seq (a : ℝ) (n : ℕ) : Prop :=
  a_n a n < a_n a (n + 1)

theorem range_of_a (a : ℝ) :
  (∀ n, increasing_seq a n) ↔ (9 / 4 < a ∧ a < 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1388_138878


namespace NUMINAMATH_GPT_ratio_is_five_thirds_l1388_138833

noncomputable def ratio_of_numbers (a b : ℝ) : Prop :=
  (a + b = 4 * (a - b)) → (a = 2 * b) → (a / b = 5 / 3)

theorem ratio_is_five_thirds {a b : ℝ} (h1 : a + b = 4 * (a - b)) (h2 : a = 2 * b) :
  a / b = 5 / 3 :=
  sorry

end NUMINAMATH_GPT_ratio_is_five_thirds_l1388_138833


namespace NUMINAMATH_GPT_best_fit_line_slope_l1388_138857

theorem best_fit_line_slope (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) (d : ℝ) 
  (h1 : x2 - x1 = 2 * d) (h2 : x3 - x2 = 3 * d) (h3 : x4 - x3 = d) : 
  ((y4 - y1) / (x4 - x1)) = (y4 - y1) / (x4 - x1) :=
by
  sorry

end NUMINAMATH_GPT_best_fit_line_slope_l1388_138857


namespace NUMINAMATH_GPT_max_area_225_l1388_138891

noncomputable def max_area_rect_perim60 (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x ≥ 10) : ℝ :=
max (x * y) (30 - x)

theorem max_area_225 (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x ≥ 10) :
  max_area_rect_perim60 x y h1 h2 = 225 :=
sorry

end NUMINAMATH_GPT_max_area_225_l1388_138891


namespace NUMINAMATH_GPT_fraction_of_300_greater_than_3_fifths_of_125_l1388_138872

theorem fraction_of_300_greater_than_3_fifths_of_125 (f : ℚ)
    (h : f * 300 = 3 / 5 * 125 + 45) : 
    f = 2 / 5 :=
sorry

end NUMINAMATH_GPT_fraction_of_300_greater_than_3_fifths_of_125_l1388_138872


namespace NUMINAMATH_GPT_weeks_in_semester_l1388_138820

-- Define the conditions and the question as a hypothesis
def annie_club_hours : Nat := 13

theorem weeks_in_semester (w : Nat) (h : 13 * (w - 2) = 52) : w = 6 := by
  sorry

end NUMINAMATH_GPT_weeks_in_semester_l1388_138820


namespace NUMINAMATH_GPT_minimum_m_l1388_138871

/-
  Given that for all 2 ≤ x ≤ 3, 3 ≤ y ≤ 6, the inequality mx^2 - xy + y^2 ≥ 0 always holds,
  prove that the minimum value of the real number m is 0.
-/
theorem minimum_m (m : ℝ) :
  (∀ x y : ℝ, 2 ≤ x ∧ x ≤ 3 → 3 ≤ y ∧ y ≤ 6 → m * x^2 - x * y + y^2 ≥ 0) → m = 0 :=
sorry -- proof to be provided

end NUMINAMATH_GPT_minimum_m_l1388_138871


namespace NUMINAMATH_GPT_point_on_x_axis_l1388_138817

theorem point_on_x_axis (m : ℝ) (h : (2 * m + 3) = 0) : m = -3 / 2 :=
sorry

end NUMINAMATH_GPT_point_on_x_axis_l1388_138817


namespace NUMINAMATH_GPT_max_value_x2_y2_l1388_138837

noncomputable def max_x2_y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y ≥ x^3 + y^2) : ℝ := 2

theorem max_value_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y ≥ x^3 + y^2) : 
  x^2 + y^2 ≤ max_x2_y2 x y hx hy h :=
by
  sorry

end NUMINAMATH_GPT_max_value_x2_y2_l1388_138837


namespace NUMINAMATH_GPT_total_percent_sample_candy_l1388_138822

theorem total_percent_sample_candy (total_customers : ℕ) (percent_caught : ℝ) (percent_not_caught : ℝ)
  (h1 : percent_caught = 0.22)
  (h2 : percent_not_caught = 0.20)
  (h3 : total_customers = 100) :
  percent_caught + percent_not_caught = 0.28 :=
by
  sorry

end NUMINAMATH_GPT_total_percent_sample_candy_l1388_138822


namespace NUMINAMATH_GPT_meaningful_square_root_l1388_138801

theorem meaningful_square_root (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 := 
by {
  -- This part would contain the proof
  sorry
}

end NUMINAMATH_GPT_meaningful_square_root_l1388_138801


namespace NUMINAMATH_GPT_middle_integer_of_sum_is_120_l1388_138814

-- Define the condition that three consecutive integers sum to 360
def consecutive_integers_sum_to (n : ℤ) (sum : ℤ) : Prop :=
  (n - 1) + n + (n + 1) = sum

-- The statement to prove
theorem middle_integer_of_sum_is_120 (n : ℤ) :
  consecutive_integers_sum_to n 360 → n = 120 :=
by
  sorry

end NUMINAMATH_GPT_middle_integer_of_sum_is_120_l1388_138814


namespace NUMINAMATH_GPT_proof_problem_l1388_138824

-- Definition of the condition
def condition (y : ℝ) : Prop := 6 * y^2 + 5 = 2 * y + 10

-- Stating the theorem
theorem proof_problem : ∀ y : ℝ, condition y → (12 * y - 5)^2 = 133 :=
by
  intro y
  intro h
  sorry

end NUMINAMATH_GPT_proof_problem_l1388_138824


namespace NUMINAMATH_GPT_constant_term_expansion_l1388_138846

theorem constant_term_expansion : 
  ∃ r : ℕ, (9 - 3 * r / 2 = 0) ∧ 
  ∀ (x : ℝ) (hx : x ≠ 0), (2 * x - 1 / Real.sqrt x) ^ 9 = 672 := 
by sorry

end NUMINAMATH_GPT_constant_term_expansion_l1388_138846


namespace NUMINAMATH_GPT_volume_of_regular_tetrahedron_l1388_138821

noncomputable def volume_of_tetrahedron (a : ℝ) : ℝ :=
  (a ^ 3 * Real.sqrt 2) / 12

theorem volume_of_regular_tetrahedron (a : ℝ) : 
  volume_of_tetrahedron a = (a ^ 3 * Real.sqrt 2) / 12 := 
by
  sorry

end NUMINAMATH_GPT_volume_of_regular_tetrahedron_l1388_138821
