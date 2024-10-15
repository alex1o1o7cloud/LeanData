import Mathlib

namespace NUMINAMATH_GPT_cube_root_neg_frac_l2096_209695

theorem cube_root_neg_frac : (-(1/3 : ℝ))^3 = - 1 / 27 := by
  sorry

end NUMINAMATH_GPT_cube_root_neg_frac_l2096_209695


namespace NUMINAMATH_GPT_complex_solutions_x2_eq_neg4_l2096_209673

-- Lean statement for the proof problem
theorem complex_solutions_x2_eq_neg4 (x : ℂ) (hx : x^2 = -4) : x = 2 * Complex.I ∨ x = -2 * Complex.I :=
by 
  sorry

end NUMINAMATH_GPT_complex_solutions_x2_eq_neg4_l2096_209673


namespace NUMINAMATH_GPT_rectangular_prism_sum_l2096_209681

-- Definitions based on conditions
def edges := 12
def corners := 8
def faces := 6

-- Lean statement to prove question == answer given conditions.
theorem rectangular_prism_sum : edges + corners + faces = 26 := by
  sorry

end NUMINAMATH_GPT_rectangular_prism_sum_l2096_209681


namespace NUMINAMATH_GPT_combined_cost_price_is_250_l2096_209682

axiom store_selling_conditions :
  ∃ (CP_A CP_B CP_C : ℝ),
    (CP_A = (110 + 70) / 2) ∧
    (CP_B = (90 + 30) / 2) ∧
    (CP_C = (150 + 50) / 2) ∧
    (CP_A + CP_B + CP_C = 250)

theorem combined_cost_price_is_250 : ∃ (CP_A CP_B CP_C : ℝ), CP_A + CP_B + CP_C = 250 :=
by sorry

end NUMINAMATH_GPT_combined_cost_price_is_250_l2096_209682


namespace NUMINAMATH_GPT_new_average_score_after_drop_l2096_209697

theorem new_average_score_after_drop
  (avg_score : ℝ) (num_students : ℕ) (drop_score : ℝ) (remaining_students : ℕ) :
  avg_score = 62.5 →
  num_students = 16 →
  drop_score = 70 →
  remaining_students = 15 →
  (num_students * avg_score - drop_score) / remaining_students = 62 :=
by
  intros h_avg h_num h_drop h_remain
  rw [h_avg, h_num, h_drop, h_remain]
  norm_num

end NUMINAMATH_GPT_new_average_score_after_drop_l2096_209697


namespace NUMINAMATH_GPT_incorrect_statement_A_l2096_209646

-- conditions as stated in the table
def spring_length (x : ℕ) : ℝ :=
  if x = 0 then 20
  else if x = 1 then 20.5
  else if x = 2 then 21
  else if x = 3 then 21.5
  else if x = 4 then 22
  else if x = 5 then 22.5
  else 0 -- assuming 0 for out of range for simplicity

-- questions with answers
-- Prove that statement A is incorrect
theorem incorrect_statement_A : ¬ (spring_length 0 = 20) := by
  sorry

end NUMINAMATH_GPT_incorrect_statement_A_l2096_209646


namespace NUMINAMATH_GPT_sodium_bicarbonate_moles_combined_l2096_209651

theorem sodium_bicarbonate_moles_combined (HCl NaCl NaHCO3 : ℝ) (reaction : HCl + NaHCO3 = NaCl) 
  (HCl_eq_one : HCl = 1) (NaCl_eq_one : NaCl = 1) : 
  NaHCO3 = 1 := 
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_sodium_bicarbonate_moles_combined_l2096_209651


namespace NUMINAMATH_GPT_combine_material_points_l2096_209668

variables {K K₁ K₂ : Type} {m m₁ m₂ : ℝ}

-- Assume some properties and operations for type K
noncomputable def add_material_points (K₁ K₂ : K × ℝ) : K × ℝ :=
(K₁.1, K₁.2 + K₂.2)

theorem combine_material_points (K₁ K₂ : K × ℝ) :
  (add_material_points K₁ K₂) = (K₁.1, K₁.2 + K₂.2) :=
sorry

end NUMINAMATH_GPT_combine_material_points_l2096_209668


namespace NUMINAMATH_GPT_sum_of_first_n_terms_geom_sequence_l2096_209643

theorem sum_of_first_n_terms_geom_sequence (a₁ q : ℚ) (S : ℕ → ℚ)
  (h : ∀ n, S n = a₁ * (1 - q^n) / (1 - q))
  (h_ratio : S 4 / S 2 = 3) :
  S 6 / S 4 = 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_n_terms_geom_sequence_l2096_209643


namespace NUMINAMATH_GPT_h_odd_l2096_209667

variable (f g : ℝ → ℝ)

-- f is odd and g is even
axiom f_odd : ∀ x, -2 ≤ x ∧ x ≤ 2 → f (-x) = -f x
axiom g_even : ∀ x, -2 ≤ x ∧ x ≤ 2 → g (-x) = g x

-- Prove that h(x) = f(x) * g(x) is odd
theorem h_odd : ∀ x, -2 ≤ x ∧ x ≤ 2 → (f x) * (g x) = (f (-x)) * (g (-x)) := by
  sorry

end NUMINAMATH_GPT_h_odd_l2096_209667


namespace NUMINAMATH_GPT_casper_initial_candies_l2096_209612

theorem casper_initial_candies 
  (x : ℚ)
  (h1 : ∃ y : ℚ, y = x - (1/4) * x - 3) 
  (h2 : ∃ z : ℚ, z = y - (1/5) * y - 5) 
  (h3 : z - 10 = 10) : x = 224 / 3 :=
by
  sorry

end NUMINAMATH_GPT_casper_initial_candies_l2096_209612


namespace NUMINAMATH_GPT_diff_of_squares_div_l2096_209637

theorem diff_of_squares_div (a b : ℤ) (h1 : a = 121) (h2 : b = 112) : 
  (a^2 - b^2) / (a - b) = a + b :=
by
  rw [h1, h2]
  rw [sub_eq_add_neg, add_comm]
  exact sorry

end NUMINAMATH_GPT_diff_of_squares_div_l2096_209637


namespace NUMINAMATH_GPT_range_of_x_l2096_209600

variable (x : ℝ)

def p := x^2 - 4 * x + 3 < 0
def q := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

theorem range_of_x : ¬ (p x ∧ q x) ∧ (p x ∨ q x) → (1 < x ∧ x ≤ 2) ∨ x = 3 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_x_l2096_209600


namespace NUMINAMATH_GPT_largest_prime_factor_8250_l2096_209698

-- Define a function to check if a number is prime (using an existing library function)
def is_prime (n: ℕ) : Prop := Nat.Prime n

-- Define the given problem statement as a Lean theorem
theorem largest_prime_factor_8250 :
  ∃ p, is_prime p ∧ p ∣ 8250 ∧ 
    ∀ q, is_prime q ∧ q ∣ 8250 → q ≤ p :=
sorry -- The proof will be filled in later

end NUMINAMATH_GPT_largest_prime_factor_8250_l2096_209698


namespace NUMINAMATH_GPT_infinite_series_sum_l2096_209665

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) / 4^(n + 1)) + (∑' n : ℕ, 1 / 2^(n + 1)) = 13 / 9 := 
sorry

end NUMINAMATH_GPT_infinite_series_sum_l2096_209665


namespace NUMINAMATH_GPT_total_apartments_in_building_l2096_209679

theorem total_apartments_in_building (A k m n : ℕ)
  (cond1 : 5 = A)
  (cond2 : 636 = (m-1) * k + n)
  (cond3 : 242 = (A-m) * k + n) :
  A * k = 985 :=
by
  sorry

end NUMINAMATH_GPT_total_apartments_in_building_l2096_209679


namespace NUMINAMATH_GPT_range_of_a_l2096_209644

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x * |x - a| - 2 < 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2096_209644


namespace NUMINAMATH_GPT_find_ordered_pair_l2096_209604

theorem find_ordered_pair (a b : ℚ) :
  a • (⟨2, 3⟩ : ℚ × ℚ) + b • (⟨-2, 5⟩ : ℚ × ℚ) = (⟨10, -8⟩ : ℚ × ℚ) →
  (a, b) = (17 / 8, -23 / 8) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l2096_209604


namespace NUMINAMATH_GPT_least_n_l2096_209615

theorem least_n (n : ℕ) (h : (1 : ℝ) / n - (1 / (n + 1)) < (1 / 15)) : n = 4 :=
sorry

end NUMINAMATH_GPT_least_n_l2096_209615


namespace NUMINAMATH_GPT_find_m_l2096_209607

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (m : ℕ)

theorem find_m (h1 : ∀ n, a (n + 1) = a n + d) -- arithmetic sequence
               (h2 : S (2 * m - 1) = 39)       -- sum of first (2m-1) terms
               (h3 : a (m - 1) + a (m + 1) - a m - 1 = 0)
               (h4 : m > 1) : 
               m = 20 :=
   sorry

end NUMINAMATH_GPT_find_m_l2096_209607


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sum_l2096_209674

theorem min_value_of_reciprocal_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a + 2 * b = 1) (h2 : c + 2 * d = 1) :
  16 ≤ (1 / a) + 1 / (b * c * d) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_sum_l2096_209674


namespace NUMINAMATH_GPT_sub_base8_l2096_209626

theorem sub_base8 : (1352 - 674) == 1456 :=
by sorry

end NUMINAMATH_GPT_sub_base8_l2096_209626


namespace NUMINAMATH_GPT_set_D_forms_triangle_l2096_209671

theorem set_D_forms_triangle (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 6) : a + b > c ∧ a + c > b ∧ b + c > a := by
  rw [h1, h2, h3]
  show 4 + 5 > 6 ∧ 4 + 6 > 5 ∧ 5 + 6 > 4
  sorry

end NUMINAMATH_GPT_set_D_forms_triangle_l2096_209671


namespace NUMINAMATH_GPT_jacks_speed_l2096_209670

-- Define the initial distance between Jack and Christina.
def initial_distance : ℝ := 360

-- Define Christina's speed.
def christina_speed : ℝ := 7

-- Define Lindy's speed.
def lindy_speed : ℝ := 12

-- Define the total distance Lindy travels.
def lindy_total_distance : ℝ := 360

-- Prove Jack's speed given the conditions.
theorem jacks_speed : ∃ v : ℝ, (initial_distance - christina_speed * (lindy_total_distance / lindy_speed)) / (lindy_total_distance / lindy_speed) = v ∧ v = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_jacks_speed_l2096_209670


namespace NUMINAMATH_GPT_evaluate_expression_l2096_209658

theorem evaluate_expression : 
  70 + (5 * 12) / (180 / 3) = 71 :=
  by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2096_209658


namespace NUMINAMATH_GPT_parts_rate_relation_l2096_209691

theorem parts_rate_relation
  (x : ℝ)
  (total_parts_per_hour : ℝ)
  (master_parts : ℝ)
  (apprentice_parts : ℝ)
  (h_total : total_parts_per_hour = 40)
  (h_master : master_parts = 300)
  (h_apprentice : apprentice_parts = 100)
  (h : total_parts_per_hour = x + (40 - x)) :
  (master_parts / x) = (apprentice_parts / (40 - x)) := 
by
  sorry

end NUMINAMATH_GPT_parts_rate_relation_l2096_209691


namespace NUMINAMATH_GPT_solutions_diff_l2096_209653

theorem solutions_diff (a b : ℝ) (h1: (a-5)*(a+5) = 26*a - 130) (h2: (b-5)*(b+5) = 26*b - 130) (h3 : a ≠ b) (h4: a > b) : a - b = 16 := 
by
  sorry 

end NUMINAMATH_GPT_solutions_diff_l2096_209653


namespace NUMINAMATH_GPT_proposal_spreading_problem_l2096_209639

theorem proposal_spreading_problem (n : ℕ) : 1 + n + n^2 = 1641 := 
sorry

end NUMINAMATH_GPT_proposal_spreading_problem_l2096_209639


namespace NUMINAMATH_GPT_store_loss_l2096_209683

theorem store_loss (x y : ℝ) (hx : x + 0.25 * x = 135) (hy : y - 0.25 * y = 135) : 
  (135 * 2) - (x + y) = -18 := 
by
  sorry

end NUMINAMATH_GPT_store_loss_l2096_209683


namespace NUMINAMATH_GPT_percentage_difference_between_M_and_J_is_34_74_percent_l2096_209650

-- Definitions of incomes and relationships
variables (J T M : ℝ)
variables (h1 : T = 0.80 * J)
variables (h2 : M = 1.60 * T)

-- Definitions of savings and expenses
variables (Msavings : ℝ := 0.15 * M)
variables (Mexpenses : ℝ := 0.25 * M)
variables (Tsavings : ℝ := 0.12 * T)
variables (Texpenses : ℝ := 0.30 * T)
variables (Jsavings : ℝ := 0.18 * J)
variables (Jexpenses : ℝ := 0.20 * J)

-- Total savings and expenses
variables (Mtotal : ℝ := Msavings + Mexpenses)
variables (Jtotal : ℝ := Jsavings + Jexpenses)

-- Prove the percentage difference between Mary's and Juan's total savings and expenses combined
theorem percentage_difference_between_M_and_J_is_34_74_percent :
  M = 1.28 * J → 
  Mtotal = 0.40 * M →
  Jtotal = 0.38 * J →
  ( (Mtotal - Jtotal) / Jtotal ) * 100 = 34.74 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_between_M_and_J_is_34_74_percent_l2096_209650


namespace NUMINAMATH_GPT_reflection_line_slope_l2096_209659

theorem reflection_line_slope (m b : ℝ)
  (h_reflection : ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = 2 ∧ y1 = 3 ∧ x2 = 10 ∧ y2 = 7 → 
    (x1 + x2) / 2 = (10 - 2) / 2 ∧ (y1 + y2) / 2 = (7 - 3) / 2 ∧ 
    y1 = m * x1 + b ∧ y2 = m * x2 + b) :
  m + b = 15 :=
sorry

end NUMINAMATH_GPT_reflection_line_slope_l2096_209659


namespace NUMINAMATH_GPT_expand_product_l2096_209620

variable (x : ℝ)

theorem expand_product :
  (x + 3) * (x^2 + 4 * x + 6) = x^3 + 7 * x^2 + 18 * x + 18 := 
  sorry

end NUMINAMATH_GPT_expand_product_l2096_209620


namespace NUMINAMATH_GPT_exact_time_is_3_07_27_l2096_209686

theorem exact_time_is_3_07_27 (t : ℝ) (H1 : t > 0) (H2 : t < 60) 
(H3 : 6 * (t + 8) = 89 + 0.5 * t) : t = 7 + 27/60 :=
by
  sorry

end NUMINAMATH_GPT_exact_time_is_3_07_27_l2096_209686


namespace NUMINAMATH_GPT_elephants_at_WePreserveForFuture_l2096_209633

theorem elephants_at_WePreserveForFuture (E : ℕ) 
  (h1 : ∀ gest : ℕ, gest = 3 * E)
  (h2 : ∀ total : ℕ, total = E + 3 * E) 
  (h3 : total = 280) : 
  E = 70 := 
by
  sorry

end NUMINAMATH_GPT_elephants_at_WePreserveForFuture_l2096_209633


namespace NUMINAMATH_GPT_num_C_atoms_in_compound_l2096_209606

def num_H_atoms := 6
def num_O_atoms := 1
def molecular_weight := 58
def atomic_weight_C := 12
def atomic_weight_H := 1
def atomic_weight_O := 16

theorem num_C_atoms_in_compound : 
  ∃ (num_C_atoms : ℕ), 
    molecular_weight = (num_C_atoms * atomic_weight_C) + (num_H_atoms * atomic_weight_H) + (num_O_atoms * atomic_weight_O) ∧ 
    num_C_atoms = 3 :=
by
  -- To be proven
  sorry

end NUMINAMATH_GPT_num_C_atoms_in_compound_l2096_209606


namespace NUMINAMATH_GPT_decreasing_f_range_l2096_209623

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log x - 2 * k * x - 1

theorem decreasing_f_range (k : ℝ) (x₁ x₂ : ℝ) (h₁ : 2 ≤ x₁) (h₂ : x₁ < x₂) (h₃ : x₂ ≤ 4) :
  k ≥ 1 / 4 → (x₁ - x₂) * (f x₁ k - f x₂ k) < 0 :=
sorry

end NUMINAMATH_GPT_decreasing_f_range_l2096_209623


namespace NUMINAMATH_GPT_speed_second_half_l2096_209662

theorem speed_second_half (H : ℝ) (S1 S2 : ℝ) (T : ℝ) : T = 11 → S1 = 30 → S1 * T1 = 150 → S1 * T1 + S2 * T2 = 300 → S2 = 25 :=
by
  intro hT hS1 hD1 hTotal
  sorry

end NUMINAMATH_GPT_speed_second_half_l2096_209662


namespace NUMINAMATH_GPT_mean_equals_d_l2096_209657

noncomputable def sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

theorem mean_equals_d
  (a b c d e : ℝ)
  (h_a : a = sqrt 2)
  (h_b : b = sqrt 18)
  (h_c : c = sqrt 200)
  (h_d : d = sqrt 32)
  (h_e : e = sqrt 8) :
  d = (a + b + c + e) / 4 := by
  -- We insert proof steps here normally
  sorry

end NUMINAMATH_GPT_mean_equals_d_l2096_209657


namespace NUMINAMATH_GPT_probability_of_picking_letter_in_mathematics_l2096_209603

-- Definitions and conditions
def total_letters : ℕ := 26
def unique_letters_in_mathematics : ℕ := 8
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Theorem to be proven
theorem probability_of_picking_letter_in_mathematics :
  probability unique_letters_in_mathematics total_letters = 4 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_picking_letter_in_mathematics_l2096_209603


namespace NUMINAMATH_GPT_total_revenue_calculation_l2096_209648

variables (a b : ℕ) -- Assuming a and b are natural numbers representing the number of newspapers

-- Define the prices
def purchase_price_per_copy : ℝ := 0.4
def selling_price_per_copy : ℝ := 0.5
def return_price_per_copy : ℝ := 0.2

-- Define the revenue and cost calculations
def revenue_from_selling (b : ℕ) : ℝ := selling_price_per_copy * b
def revenue_from_returning (a b : ℕ) : ℝ := return_price_per_copy * (a - b)
def cost_of_purchasing (a : ℕ) : ℝ := purchase_price_per_copy * a

-- Define the total revenue
def total_revenue (a b : ℕ) : ℝ :=
  revenue_from_selling b + revenue_from_returning a b - cost_of_purchasing a

-- The theorem we need to prove
theorem total_revenue_calculation (a b : ℕ) :
  total_revenue a b = 0.3 * b - 0.2 * a :=
by
  sorry

end NUMINAMATH_GPT_total_revenue_calculation_l2096_209648


namespace NUMINAMATH_GPT_parts_of_a_number_l2096_209616

theorem parts_of_a_number 
  (a p q : ℝ) 
  (x y z : ℝ)
  (h1 : y + z = p * x)
  (h2 : x + y = q * z)
  (h3 : x + y + z = a) :
  x = a / (1 + p) ∧ y = a * (p * q - 1) / ((p + 1) * (q + 1)) ∧ z = a / (1 + q) := 
by 
  sorry

end NUMINAMATH_GPT_parts_of_a_number_l2096_209616


namespace NUMINAMATH_GPT_find_x_l2096_209687

theorem find_x (x : ℝ) (h : 0.25 * x = 200 - 30) : x = 680 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l2096_209687


namespace NUMINAMATH_GPT_solve_x_l2096_209647

theorem solve_x : ∀ (x y : ℝ), (3 * x - y = 7) ∧ (x + 3 * y = 6) → x = 27 / 10 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_solve_x_l2096_209647


namespace NUMINAMATH_GPT_product_value_4_l2096_209656

noncomputable def product_of_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 5) : ℝ :=
(x - 1) * (y - 1)

theorem product_value_4 (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 5) : ∃ v : ℝ, product_of_values x y h = v ∧ v = 4 :=
sorry

end NUMINAMATH_GPT_product_value_4_l2096_209656


namespace NUMINAMATH_GPT_point_not_on_line_pq_neg_l2096_209605

theorem point_not_on_line_pq_neg (p q : ℝ) (h : p * q < 0) : ¬ (21 * p + q = -101) := 
by sorry

end NUMINAMATH_GPT_point_not_on_line_pq_neg_l2096_209605


namespace NUMINAMATH_GPT_area_of_border_correct_l2096_209664

def height_of_photograph : ℕ := 12
def width_of_photograph : ℕ := 16
def border_width : ℕ := 3
def lining_width : ℕ := 1

def area_of_photograph : ℕ := height_of_photograph * width_of_photograph

def total_height : ℕ := height_of_photograph + 2 * (lining_width + border_width)
def total_width : ℕ := width_of_photograph + 2 * (lining_width + border_width)

def area_of_framed_area : ℕ := total_height * total_width

def area_of_border_including_lining : ℕ := area_of_framed_area - area_of_photograph

theorem area_of_border_correct : area_of_border_including_lining = 288 := by
  sorry

end NUMINAMATH_GPT_area_of_border_correct_l2096_209664


namespace NUMINAMATH_GPT_Danny_caps_vs_wrappers_l2096_209649

def park_caps : ℕ := 58
def park_wrappers : ℕ := 25
def beach_caps : ℕ := 34
def beach_wrappers : ℕ := 15
def forest_caps : ℕ := 21
def forest_wrappers : ℕ := 32
def before_caps : ℕ := 12
def before_wrappers : ℕ := 11

noncomputable def total_caps : ℕ := park_caps + beach_caps + forest_caps + before_caps
noncomputable def total_wrappers : ℕ := park_wrappers + beach_wrappers + forest_wrappers + before_wrappers

theorem Danny_caps_vs_wrappers : total_caps - total_wrappers = 42 := by
  sorry

end NUMINAMATH_GPT_Danny_caps_vs_wrappers_l2096_209649


namespace NUMINAMATH_GPT_total_sentence_l2096_209608

theorem total_sentence (base_rate : ℝ) (value_stolen : ℝ) (third_offense_increase : ℝ) (additional_years : ℕ) : 
  base_rate = 1 / 5000 → 
  value_stolen = 40000 → 
  third_offense_increase = 0.25 → 
  additional_years = 2 →
  (value_stolen * base_rate * (1 + third_offense_increase) + additional_years) = 12 := 
by
  intros
  sorry

end NUMINAMATH_GPT_total_sentence_l2096_209608


namespace NUMINAMATH_GPT_binary_multiplication_correct_l2096_209632

theorem binary_multiplication_correct:
  let n1 := 29 -- binary 11101 is decimal 29
  let n2 := 13 -- binary 1101 is decimal 13
  let result := 303 -- binary 100101111 is decimal 303
  n1 * n2 = result :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_binary_multiplication_correct_l2096_209632


namespace NUMINAMATH_GPT_flight_time_is_10_hours_l2096_209635

def time_watching_TV_episodes : ℕ := 3 * 25
def time_sleeping : ℕ := 4 * 60 + 30
def time_watching_movies : ℕ := 2 * (1 * 60 + 45)
def remaining_flight_time : ℕ := 45

def total_flight_time : ℕ := (time_watching_TV_episodes + time_sleeping + time_watching_movies + remaining_flight_time) / 60

theorem flight_time_is_10_hours : total_flight_time = 10 := by
  sorry

end NUMINAMATH_GPT_flight_time_is_10_hours_l2096_209635


namespace NUMINAMATH_GPT_compute_operation_value_l2096_209624

def operation (a b c : ℝ) : ℝ := b^3 - 3 * a * b * c - 4 * a * c^2

theorem compute_operation_value : operation 2 (-1) 4 = -105 :=
by
  sorry

end NUMINAMATH_GPT_compute_operation_value_l2096_209624


namespace NUMINAMATH_GPT_find_value_of_expression_l2096_209636

theorem find_value_of_expression (x y z : ℚ)
  (h1 : 2 * x + y + z = 14)
  (h2 : 2 * x + y = 7)
  (h3 : x + 2 * y = 10) : (x + y - z) / 3 = -4 / 9 :=
by sorry

end NUMINAMATH_GPT_find_value_of_expression_l2096_209636


namespace NUMINAMATH_GPT_area_triangle_PTS_l2096_209601

theorem area_triangle_PTS {PQ QR PS QT PT TS : ℝ} 
  (hPQ : PQ = 4) 
  (hQR : QR = 6) 
  (hPS : PS = 2 * Real.sqrt 13) 
  (hQT : QT = 12 * Real.sqrt 13 / 13) 
  (hPT : PT = 4) 
  (hTS : TS = (2 * Real.sqrt 13) - 4) : 
  (1 / 2) * PT * QT = 24 * Real.sqrt 13 / 13 := 
by 
  sorry

end NUMINAMATH_GPT_area_triangle_PTS_l2096_209601


namespace NUMINAMATH_GPT_range_of_m_l2096_209652

noncomputable def f (x : ℝ) : ℝ := -x^2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2^x - m

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Set.Icc (-1 : ℝ) 3, ∃ x2 ∈ Set.Icc (0 : ℝ) 2, f x1 ≥ g x2 m) ↔ m ≥ 10 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2096_209652


namespace NUMINAMATH_GPT_spending_spring_months_l2096_209617

variable (s_feb s_may : ℝ)

theorem spending_spring_months (h1 : s_feb = 2.8) (h2 : s_may = 5.6) : s_may - s_feb = 2.8 := 
by
  sorry

end NUMINAMATH_GPT_spending_spring_months_l2096_209617


namespace NUMINAMATH_GPT_discount_comparison_l2096_209641

noncomputable def final_price (P : ℝ) (d1 d2 d3 : ℝ) : ℝ :=
  P * (1 - d1) * (1 - d2) * (1 - d3)

theorem discount_comparison (P : ℝ) (d11 d12 d13 d21 d22 d23 : ℝ) :
  P = 20000 →
  d11 = 0.25 → d12 = 0.15 → d13 = 0.10 →
  d21 = 0.30 → d22 = 0.10 → d23 = 0.10 →
  final_price P d11 d12 d13 - final_price P d21 d22 d23 = 135 :=
by
  intros
  sorry

end NUMINAMATH_GPT_discount_comparison_l2096_209641


namespace NUMINAMATH_GPT_original_price_is_125_l2096_209611

noncomputable def original_price (sold_price : ℝ) (discount_percent : ℝ) : ℝ :=
  sold_price / ((100 - discount_percent) / 100)

theorem original_price_is_125 : original_price 120 4 = 125 :=
by
  sorry

end NUMINAMATH_GPT_original_price_is_125_l2096_209611


namespace NUMINAMATH_GPT_find_m_n_l2096_209685

theorem find_m_n (m n : ℕ) (h : 26019 * m - 649 * n = 118) : m = 2 ∧ n = 80 :=
by 
  sorry

end NUMINAMATH_GPT_find_m_n_l2096_209685


namespace NUMINAMATH_GPT_max_combinations_for_n_20_l2096_209621

def num_combinations (s n k : ℕ) : ℕ :=
if n = 0 then if s = 0 then 1 else 0
else if s < n then 0
else if k = 0 then 0
else num_combinations (s - k) (n - 1) (k - 1) + num_combinations s n (k - 1)

theorem max_combinations_for_n_20 : ∀ s k, s = 20 ∧ k = 9 → num_combinations s 4 k = 12 :=
by
  intros s k h
  cases h
  sorry

end NUMINAMATH_GPT_max_combinations_for_n_20_l2096_209621


namespace NUMINAMATH_GPT_words_count_correct_l2096_209696

def number_of_words (n : ℕ) : ℕ :=
if n % 2 = 0 then
  8 * 3^(n / 2 - 1)
else
  14 * 3^((n - 1) / 2)

theorem words_count_correct (n : ℕ) :
  number_of_words n = if n % 2 = 0 then 8 * 3^(n / 2 - 1) else 14 * 3^((n - 1) / 2) :=
by
  sorry

end NUMINAMATH_GPT_words_count_correct_l2096_209696


namespace NUMINAMATH_GPT_largest_value_f12_l2096_209618

theorem largest_value_f12 (f : ℝ → ℝ) (hf_poly : ∀ x, f x ≥ 0) 
  (hf_6 : f 6 = 24) (hf_24 : f 24 = 1536) :
  f 12 ≤ 192 :=
sorry

end NUMINAMATH_GPT_largest_value_f12_l2096_209618


namespace NUMINAMATH_GPT_train_lengths_combined_l2096_209613

noncomputable def speed_to_mps (kmph : ℤ) : ℚ := (kmph : ℚ) * 5 / 18

def length_of_train (speed : ℚ) (time : ℚ) : ℚ := speed * time

theorem train_lengths_combined :
  let speed1_kmph := 100
  let speed2_kmph := 120
  let time1_sec := 9
  let time2_sec := 8
  let speed1_mps := speed_to_mps speed1_kmph
  let speed2_mps := speed_to_mps speed2_kmph
  let length1 := length_of_train speed1_mps time1_sec
  let length2 := length_of_train speed2_mps time2_sec
  length1 + length2 = 516.66 :=
by
  sorry

end NUMINAMATH_GPT_train_lengths_combined_l2096_209613


namespace NUMINAMATH_GPT_shopkeeper_intended_profit_l2096_209654

noncomputable def intended_profit_percentage (C L S : ℝ) : ℝ :=
  (L / C) - 1

theorem shopkeeper_intended_profit (C L S : ℝ) (h1 : L = C * (1 + intended_profit_percentage C L S))
  (h2 : S = 0.90 * L) (h3 : S = 1.35 * C) : intended_profit_percentage C L S = 0.5 :=
by
  -- We indicate that the proof is skipped
  sorry

end NUMINAMATH_GPT_shopkeeper_intended_profit_l2096_209654


namespace NUMINAMATH_GPT_moles_of_NaCl_formed_l2096_209672

-- Given conditions
def sodium_bisulfite_moles : ℕ := 2
def hydrochloric_acid_moles : ℕ := 2
def balanced_reaction : Prop :=
  ∀ (NaHSO3 HCl NaCl H2O SO2 : ℕ), 
    NaHSO3 + HCl = NaCl + H2O + SO2

-- Target to prove:
theorem moles_of_NaCl_formed :
  balanced_reaction → sodium_bisulfite_moles = hydrochloric_acid_moles → 
  sodium_bisulfite_moles = 2 := 
sorry

end NUMINAMATH_GPT_moles_of_NaCl_formed_l2096_209672


namespace NUMINAMATH_GPT_min_spiders_sufficient_spiders_l2096_209622

def grid_size : ℕ := 2019

noncomputable def min_k_catch (k : ℕ) : Prop :=
∀ (fly spider1 spider2 : ℕ × ℕ) (fly_move spider1_move spider2_move: ℕ × ℕ → ℕ × ℕ), 
  (fly_move fly = fly ∨ fly_move fly = (fly.1 + 1, fly.2) ∨ fly_move fly = (fly.1 - 1, fly.2)
  ∨ fly_move fly = (fly.1, fly.2 + 1) ∨ fly_move fly = (fly.1, fly.2 - 1))
  ∧ (spider1_move spider1 = spider1 ∨ spider1_move spider1 = (spider1.1 + 1, spider1.2) ∨ spider1_move spider1 = (spider1.1 - 1, spider1.2)
  ∨ spider1_move spider1 = (spider1.1, spider1.2 + 1) ∨ spider1_move spider1 = (spider1.1, spider1.2 - 1))
  ∧ (spider2_move spider2 = spider2 ∨ spider2_move spider2 = (spider2.1 + 1, spider2.2) ∨ spider2_move spider2 = (spider2.1 - 1, spider2.2)
  ∨ spider2_move spider2 = (spider2.1, spider2.2 + 1) ∨ spider2_move spider2 = (spider2.1, spider2.2 - 1))
  → (spider1 = fly ∨ spider2 = fly)

theorem min_spiders (k : ℕ) : min_k_catch k → k ≥ 2 :=
sorry

theorem sufficient_spiders : min_k_catch 2 :=
sorry

end NUMINAMATH_GPT_min_spiders_sufficient_spiders_l2096_209622


namespace NUMINAMATH_GPT_supplement_of_angle_l2096_209609

-- Condition: The complement of angle α is 54 degrees 32 minutes
theorem supplement_of_angle (α : ℝ) (h : α = 90 - (54 + 32 / 60)) :
  180 - α = 144 + 32 / 60 := by
sorry

end NUMINAMATH_GPT_supplement_of_angle_l2096_209609


namespace NUMINAMATH_GPT_max_radius_of_circle_in_triangle_inscribed_l2096_209676

theorem max_radius_of_circle_in_triangle_inscribed (ω : Set (ℝ × ℝ)) (hω : ∀ (P : ℝ × ℝ), P ∈ ω → P.1^2 + P.2^2 = 1)
  (O : ℝ × ℝ) (hO : O = (0, 0)) (P : ℝ × ℝ) (hP : P ∈ ω) (A : ℝ × ℝ) 
  (hA : A = (P.1, 0)) : 
  (∃ r : ℝ, r = (Real.sqrt 2 - 1) / 2) :=
by
  sorry

end NUMINAMATH_GPT_max_radius_of_circle_in_triangle_inscribed_l2096_209676


namespace NUMINAMATH_GPT_no_determinable_cost_of_2_pans_l2096_209655

def pots_and_pans_problem : Prop :=
  ∀ (P Q : ℕ), 3 * P + 4 * Q = 100 → ¬∃ Q_cost : ℕ, Q_cost = 2 * Q

theorem no_determinable_cost_of_2_pans : pots_and_pans_problem :=
by
  sorry

end NUMINAMATH_GPT_no_determinable_cost_of_2_pans_l2096_209655


namespace NUMINAMATH_GPT_problem1_l2096_209627

theorem problem1 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
sorry

end NUMINAMATH_GPT_problem1_l2096_209627


namespace NUMINAMATH_GPT_find_sum_of_squares_l2096_209694

theorem find_sum_of_squares (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + x + y = 119) (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := 
by
  sorry

end NUMINAMATH_GPT_find_sum_of_squares_l2096_209694


namespace NUMINAMATH_GPT_log5_x_l2096_209630

theorem log5_x (x : ℝ) (h : x = (Real.log 2 / Real.log 4) ^ (Real.log 16 / Real.log 2) ^ 2) :
    Real.log x / Real.log 5 = -16 / (Real.log 2 / Real.log 5) := by
  sorry

end NUMINAMATH_GPT_log5_x_l2096_209630


namespace NUMINAMATH_GPT_digit_product_equality_l2096_209663

theorem digit_product_equality (x y z : ℕ) (hx : x = 3) (hy : y = 7) (hz : z = 1) :
  x * (10 * x + y) = 111 * z :=
by
  -- Using hx, hy, and hz, the proof can proceed from here
  sorry

end NUMINAMATH_GPT_digit_product_equality_l2096_209663


namespace NUMINAMATH_GPT_max_entanglements_l2096_209666

theorem max_entanglements (a b : ℕ) (h1 : a < b) (h2 : a < 1000) (h3 : b < 1000) :
  ∃ n ≤ 9, ∀ k, k ≤ n → ∃ a' b' : ℕ, (b' - a' = b - a - 2^k) :=
by sorry

end NUMINAMATH_GPT_max_entanglements_l2096_209666


namespace NUMINAMATH_GPT_sector_area_l2096_209629

theorem sector_area (r : ℝ) (alpha : ℝ) (h : r = 2) (h2 : alpha = π / 3) : 
  1/2 * alpha * r^2 = (2 * π) / 3 := by
  sorry

end NUMINAMATH_GPT_sector_area_l2096_209629


namespace NUMINAMATH_GPT_height_of_balcony_l2096_209689

variable (t : ℝ) (v₀ : ℝ) (g : ℝ) (h₀ : ℝ)

axiom cond1 : t = 6
axiom cond2 : v₀ = 20
axiom cond3 : g = 10

theorem height_of_balcony : h₀ + v₀ * t - (1/2 : ℝ) * g * t^2 = 0 → h₀ = 60 :=
by
  intro h'
  sorry

end NUMINAMATH_GPT_height_of_balcony_l2096_209689


namespace NUMINAMATH_GPT_three_b_minus_a_eq_neg_five_l2096_209693

theorem three_b_minus_a_eq_neg_five (a b : ℤ) (h : |a - 2| + (b + 1)^2 = 0) : 3 * b - a = -5 :=
sorry

end NUMINAMATH_GPT_three_b_minus_a_eq_neg_five_l2096_209693


namespace NUMINAMATH_GPT_triangle_c_and_area_l2096_209631

theorem triangle_c_and_area
  (a b : ℝ) (C : ℝ)
  (h_a : a = 1)
  (h_b : b = 2)
  (h_C : C = Real.pi / 3) :
  ∃ (c S : ℝ), c = Real.sqrt 3 ∧ S = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_c_and_area_l2096_209631


namespace NUMINAMATH_GPT_one_leg_divisible_by_3_l2096_209638

theorem one_leg_divisible_by_3 (a b c : ℕ) (h : a^2 + b^2 = c^2) : (3 ∣ a) ∨ (3 ∣ b) :=
by sorry

end NUMINAMATH_GPT_one_leg_divisible_by_3_l2096_209638


namespace NUMINAMATH_GPT_container_capacity_l2096_209661

/-- Given a container where 8 liters is 20% of its capacity, calculate the total capacity of 
    40 such containers filled with water. -/
theorem container_capacity (c : ℝ) (h : 8 = 0.20 * c) : 
    40 * c * 40 = 1600 := 
by
  sorry

end NUMINAMATH_GPT_container_capacity_l2096_209661


namespace NUMINAMATH_GPT_find_divisor_for_multiple_l2096_209680

theorem find_divisor_for_multiple (d : ℕ) :
  (∃ k : ℕ, k * d % 1821 = 710 ∧ k * d % 24 = 13 ∧ k * d = 3024) →
  d = 23 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_divisor_for_multiple_l2096_209680


namespace NUMINAMATH_GPT_no_set_of_9_numbers_l2096_209688

theorem no_set_of_9_numbers (numbers : Finset ℕ) (median : ℕ) (max_value : ℕ) (mean : ℕ) :
  numbers.card = 9 → 
  median = 2 →
  max_value = 13 →
  mean = 7 →
  (∀ x ∈ numbers, x ≤ max_value) →
  (∃ m ∈ numbers, x ≤ median) →
  False :=
by
  sorry

end NUMINAMATH_GPT_no_set_of_9_numbers_l2096_209688


namespace NUMINAMATH_GPT_seq_nonzero_l2096_209669

def seq (a : ℕ → ℤ) : Prop :=
  (a 1 = 1) ∧ (a 2 = 2) ∧ (∀ n, n ≥ 3 → 
    (if (a (n - 2) * a (n - 1)) % 2 = 0 
     then a n = 5 * a (n - 1) - 3 * a (n - 2) 
     else a n = a (n - 1) - a (n - 2)))

theorem seq_nonzero (a : ℕ → ℤ) (h : seq a) : ∀ n, n > 0 → a n ≠ 0 :=
  sorry

end NUMINAMATH_GPT_seq_nonzero_l2096_209669


namespace NUMINAMATH_GPT_BC_total_750_l2096_209675

theorem BC_total_750 (A B C : ℤ) 
  (h1 : A + B + C = 900) 
  (h2 : A + C = 400) 
  (h3 : C = 250) : 
  B + C = 750 := 
by 
  sorry

end NUMINAMATH_GPT_BC_total_750_l2096_209675


namespace NUMINAMATH_GPT_smallest_leading_coefficient_l2096_209677

theorem smallest_leading_coefficient :
  ∀ (P : ℤ → ℤ), (∃ (a b c : ℚ), ∀ (x : ℤ), P x = a * (x^2 : ℚ) + b * (x : ℚ) + c) →
  (∀ x : ℤ, ∃ k : ℤ, P x = k) →
  (∃ a : ℚ, (∀ x : ℤ, ∃ k : ℤ, a * (x^2 : ℚ) + b * (x : ℚ) + c = k) ∧ a > 0 ∧ (∀ a' : ℚ, (∀ x : ℤ, ∃ k : ℤ, a' * (x^2 : ℚ) + b * (x : ℚ) + c = k) → a' ≥ a) ∧ a = 1 / 2) := 
sorry

end NUMINAMATH_GPT_smallest_leading_coefficient_l2096_209677


namespace NUMINAMATH_GPT_polynomial_remainder_l2096_209610

theorem polynomial_remainder :
  ∀ (x : ℂ), (x^1010 % (x^4 - 1)) = x^2 :=
sorry

end NUMINAMATH_GPT_polynomial_remainder_l2096_209610


namespace NUMINAMATH_GPT_minimize_expression_l2096_209619

theorem minimize_expression (a b c d e f : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) (h_sum : a + b + c + d + e + f = 10) :
  (1 / a + 9 / b + 25 / c + 49 / d + 81 / e + 121 / f) ≥ 129.6 :=
by
  sorry

end NUMINAMATH_GPT_minimize_expression_l2096_209619


namespace NUMINAMATH_GPT_find_x_l2096_209642

theorem find_x (x : ℝ) (hx_pos : x > 0) (hx_ceil_eq : ⌈x⌉ = 15) : x = 14 :=
by
  -- Define the condition
  have h_eq : ⌈x⌉ * x = 210 := sorry
  -- Prove that the only solution is x = 14
  sorry

end NUMINAMATH_GPT_find_x_l2096_209642


namespace NUMINAMATH_GPT_triangle_inequality_l2096_209645

theorem triangle_inequality
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ineq : 5 * (a^2 + b^2 + c^2) < 6 * (a * b + b * c + c * a)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l2096_209645


namespace NUMINAMATH_GPT_smaller_cuboid_length_l2096_209692

-- Definitions based on conditions
def original_cuboid_volume : ℝ := 18 * 15 * 2
def smaller_cuboid_volume (L : ℝ) : ℝ := 4 * 3 * L
def smaller_cuboids_total_volume (L : ℝ) : ℝ := 7.5 * smaller_cuboid_volume L

-- Theorem statement
theorem smaller_cuboid_length :
  ∃ L : ℝ, smaller_cuboids_total_volume L = original_cuboid_volume ∧ L = 6 := 
by
  sorry

end NUMINAMATH_GPT_smaller_cuboid_length_l2096_209692


namespace NUMINAMATH_GPT_fractional_eq_solution_l2096_209628

theorem fractional_eq_solution : ∀ x : ℝ, (x ≠ 3) → ((2 - x) / (x - 3) + 1 / (3 - x) = 1) → (x = 2) :=
by
  intros x h_cond h_eq
  sorry

end NUMINAMATH_GPT_fractional_eq_solution_l2096_209628


namespace NUMINAMATH_GPT_sum_of_squares_l2096_209699

theorem sum_of_squares :
  (2^2 + 1^2 + 0^2 + (-1)^2 + (-2)^2 = 10) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l2096_209699


namespace NUMINAMATH_GPT_base_number_pow_k_eq_4_pow_2k_plus_2_eq_64_l2096_209660

theorem base_number_pow_k_eq_4_pow_2k_plus_2_eq_64 (x k : ℝ) (h1 : x^k = 4) (h2 : x^(2 * k + 2) = 64) : x = 2 :=
sorry

end NUMINAMATH_GPT_base_number_pow_k_eq_4_pow_2k_plus_2_eq_64_l2096_209660


namespace NUMINAMATH_GPT_sin_neg_30_eq_neg_half_l2096_209602

/-- Prove that the sine of -30 degrees is -1/2 -/
theorem sin_neg_30_eq_neg_half : Real.sin (-(30 * Real.pi / 180)) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_neg_30_eq_neg_half_l2096_209602


namespace NUMINAMATH_GPT_area_triangle_formed_by_line_l2096_209678

theorem area_triangle_formed_by_line (b : ℝ) (h : (1 / 2) * |b * (-b / 2)| > 1) : b < -2 ∨ b > 2 :=
by 
  sorry

end NUMINAMATH_GPT_area_triangle_formed_by_line_l2096_209678


namespace NUMINAMATH_GPT_students_in_all_sections_is_six_l2096_209684

-- Define the number of students in each section and the total.
variable (total_students : ℕ := 30)
variable (music_students : ℕ := 15)
variable (drama_students : ℕ := 18)
variable (dance_students : ℕ := 12)
variable (at_least_two_sections : ℕ := 14)

-- Define the number of students in all three sections.
def students_in_all_three_sections (total_students music_students drama_students dance_students at_least_two_sections : ℕ) : ℕ :=
  let a := 6 -- the result we want to prove
  a

-- The theorem proving that the number of students in all three sections is 6.
theorem students_in_all_sections_is_six :
  students_in_all_three_sections total_students music_students drama_students dance_students at_least_two_sections = 6 :=
by 
  sorry -- Proof is omitted

end NUMINAMATH_GPT_students_in_all_sections_is_six_l2096_209684


namespace NUMINAMATH_GPT_maximize_box_volume_l2096_209640

noncomputable def volume (x : ℝ) := (16 - 2 * x) * (10 - 2 * x) * x

theorem maximize_box_volume :
  (∃ x : ℝ, volume x = 144 ∧ ∀ y : ℝ, 0 < y ∧ y < 5 → volume y ≤ volume 2) := 
by
  sorry

end NUMINAMATH_GPT_maximize_box_volume_l2096_209640


namespace NUMINAMATH_GPT_total_cost_8_dozen_pencils_2_dozen_notebooks_l2096_209625

variable (P N : ℝ)

def eq1 : Prop := 3 * P + 4 * N = 60
def eq2 : Prop := P + N = 15.512820512820513

theorem total_cost_8_dozen_pencils_2_dozen_notebooks :
  eq1 P N ∧ eq2 P N → (96 * P + 24 * N = 520) :=
by
  sorry

end NUMINAMATH_GPT_total_cost_8_dozen_pencils_2_dozen_notebooks_l2096_209625


namespace NUMINAMATH_GPT_exists_sol_in_naturals_l2096_209634

theorem exists_sol_in_naturals : ∃ (x y : ℕ), x^2 + y^2 = 61^3 := 
sorry

end NUMINAMATH_GPT_exists_sol_in_naturals_l2096_209634


namespace NUMINAMATH_GPT_project_hours_l2096_209614

variable (K P M : ℕ)

theorem project_hours
  (h1 : P + K + M = 144)
  (h2 : P = 2 * K)
  (h3 : P = M / 3) :
  M - K = 80 :=
sorry

end NUMINAMATH_GPT_project_hours_l2096_209614


namespace NUMINAMATH_GPT_paint_area_correct_l2096_209690

-- Definitions for the conditions of the problem
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def door_height : ℕ := 3
def door_length : ℕ := 5

-- Define the total area of the wall (without considering the door)
def wall_area : ℕ := wall_height * wall_length

-- Define the area of the door
def door_area : ℕ := door_height * door_length

-- Define the area that needs to be painted
def area_to_paint : ℕ := wall_area - door_area

-- The proof problem: Prove that Sandy needs to paint 135 square feet
theorem paint_area_correct : area_to_paint = 135 := 
by
  -- Sorry will be replaced with an actual proof
  sorry

end NUMINAMATH_GPT_paint_area_correct_l2096_209690
