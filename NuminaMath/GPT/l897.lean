import Mathlib

namespace factorize_expression_l897_89704

variable (x : ℝ)

theorem factorize_expression : x^2 + x = x * (x + 1) :=
by
  sorry

end factorize_expression_l897_89704


namespace power_mean_inequality_l897_89709

theorem power_mean_inequality (a b : ℝ) (n : ℕ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hn : 0 < n) :
  (a^n + b^n) / 2 ≥ ((a + b) / 2)^n := 
by
  sorry

end power_mean_inequality_l897_89709


namespace total_bill_is_60_l897_89746

def num_adults := 6
def num_children := 2
def cost_adult := 6
def cost_child := 4
def cost_soda := 2

theorem total_bill_is_60 : num_adults * cost_adult + num_children * cost_child + (num_adults + num_children) * cost_soda = 60 := by
  sorry

end total_bill_is_60_l897_89746


namespace cost_of_math_book_l897_89778

-- The definitions based on the conditions from the problem
def total_books : ℕ := 90
def math_books : ℕ := 54
def history_books := total_books - math_books -- 36
def cost_history_book : ℝ := 5
def total_cost : ℝ := 396

-- The theorem we want to prove: the cost of each math book
theorem cost_of_math_book (M : ℝ) : (math_books * M + history_books * cost_history_book = total_cost) → M = 4 := 
by 
  sorry

end cost_of_math_book_l897_89778


namespace sale_percent_saved_l897_89732

noncomputable def percent_saved (P : ℝ) : ℝ := (3 * P) / (6 * P) * 100

theorem sale_percent_saved :
  ∀ (P : ℝ), P > 0 → percent_saved P = 50 :=
by
  intros P hP
  unfold percent_saved
  have hP_nonzero : 6 * P ≠ 0 := by linarith
  field_simp [hP_nonzero]
  norm_num
  sorry

end sale_percent_saved_l897_89732


namespace graph_symmetric_l897_89714

noncomputable def f (x : ℝ) : ℝ := sorry

theorem graph_symmetric (f : ℝ → ℝ) :
  (∀ x y, y = f x ↔ (∃ y₁, y₁ = f (2 - x) ∧ y = - (1 / (y₁ + 1)))) →
  ∀ x, f x = 1 / (x - 3) := 
by
  intro h x
  sorry

end graph_symmetric_l897_89714


namespace min_value_inequality_l897_89766

theorem min_value_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 3 * a + 2 * b = 1) : 
  ∃ (m : ℝ), m = 25 ∧ (∀ x y, (x > 0) → (y > 0) → (3 * x + 2 * y = 1) → (3 / x + 2 / y) ≥ m) :=
sorry

end min_value_inequality_l897_89766


namespace number_of_rows_in_theater_l897_89756

theorem number_of_rows_in_theater 
  (x : ℕ)
  (h1 : ∀ (students : ℕ), students = 30 → ∃ row : ℕ, row < x ∧ ∃ a b : ℕ, a ≠ b ∧ row = a ∧ row = b)
  (h2 : ∀ (students : ℕ), students = 26 → ∃ empties : ℕ, empties ≥ 3 ∧ x - students = empties)
  : x = 29 :=
by
  sorry

end number_of_rows_in_theater_l897_89756


namespace socks_selection_l897_89768

/-!
  # Socks Selection Problem
  Prove the total number of ways to choose a pair of socks of different colors
  given:
  1. there are 5 white socks,
  2. there are 4 brown socks,
  3. there are 3 blue socks,
  is equal to 47.
-/

theorem socks_selection : 
  let white_socks := 5
  let brown_socks := 4
  let blue_socks := 3
  5 * 4 + 4 * 3 + 5 * 3 = 47 :=
by
  let white_socks := 5
  let brown_socks := 4
  let blue_socks := 3
  sorry

end socks_selection_l897_89768


namespace complement_union_l897_89775

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3}

theorem complement_union : U \ (A ∪ B) = {4} := by
  sorry

end complement_union_l897_89775


namespace proof_l897_89776

-- Define the universal set U.
def U : Set ℕ := {x | x > 0 ∧ x < 9}

-- Define set M.
def M : Set ℕ := {1, 2, 3}

-- Define set N.
def N : Set ℕ := {3, 4, 5, 6}

-- The complement of M with respect to U.
def compl_U_M : Set ℕ := {x ∈ U | x ∉ M}

-- The intersection of complement of M and N.
def result : Set ℕ := compl_U_M ∩ N

-- The theorem to be proven.
theorem proof : result = {4, 5, 6} := by
  -- This is where the proof would go.
  sorry

end proof_l897_89776


namespace don_raise_l897_89707

variable (D R : ℝ)

theorem don_raise 
  (h1 : R = 0.08 * D)
  (h2 : 840 = 0.08 * 10500)
  (h3 : (D + R) - (10500 + 840) = 540) : 
  R = 880 :=
by sorry

end don_raise_l897_89707


namespace bonus_trigger_sales_amount_l897_89759

theorem bonus_trigger_sales_amount (total_sales S : ℝ) (h1 : 0.09 * total_sales = 1260)
  (h2 : 0.03 * (total_sales - S) = 120) : S = 10000 :=
sorry

end bonus_trigger_sales_amount_l897_89759


namespace solution_l897_89771

variable (x y z : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (hz : z > 0)

-- Condition 1: 20/x + 6/y = 1
axiom eq1 : 20 / x + 6 / y = 1

-- Condition 2: 4/x + 2/y = 2/9
axiom eq2 : 4 / x + 2 / y = 2 / 9

-- What we need to prove: 1/z = 1/x + 1/y
axiom eq3 : 1 / x + 1 / y = 1 / z

theorem solution : z = 14.4 := by
  -- Omitted proof, just the statement
  sorry

end solution_l897_89771


namespace angle_equivalence_l897_89730

theorem angle_equivalence :
  ∃ k : ℤ, -495 + 360 * k = 225 :=
sorry

end angle_equivalence_l897_89730


namespace ef_plus_e_l897_89751

-- Define the polynomial expression
def polynomial_expr (y : ℤ) := 15 * y^2 - 82 * y + 48

-- Define the factorized form
def factorized_form (E F : ℤ) (y : ℤ) := (E * y - 16) * (F * y - 3)

-- Define the main statement to prove
theorem ef_plus_e : ∃ E F : ℤ, E * F + E = 20 ∧ ∀ y : ℤ, polynomial_expr y = factorized_form E F y :=
by {
  sorry
}

end ef_plus_e_l897_89751


namespace total_money_raised_l897_89744

-- Given conditions:
def tickets_sold : Nat := 25
def ticket_price : ℚ := 2
def num_donations_15 : Nat := 2
def donation_15 : ℚ := 15
def donation_20 : ℚ := 20

-- Theorem statement proving the total amount raised is $100
theorem total_money_raised
  (h1 : tickets_sold = 25)
  (h2 : ticket_price = 2)
  (h3 : num_donations_15 = 2)
  (h4 : donation_15 = 15)
  (h5 : donation_20 = 20) :
  (tickets_sold * ticket_price + num_donations_15 * donation_15 + donation_20) = 100 := 
by
  sorry

end total_money_raised_l897_89744


namespace hydrocarbon_tree_configurations_l897_89701

theorem hydrocarbon_tree_configurations (n : ℕ) 
  (h1 : 3 * n + 2 > 0) -- Total vertices count must be positive
  (h2 : 2 * n + 2 > 0) -- Leaves count must be positive
  (h3 : n > 0) -- Internal nodes count must be positive
  : (n:ℕ) ^ (n-2) = n ^ (n-2) :=
sorry

end hydrocarbon_tree_configurations_l897_89701


namespace proportional_function_decreases_l897_89791

-- Define the function y = -2x
def proportional_function (x : ℝ) : ℝ := -2 * x

-- State the theorem to prove that y decreases as x increases
theorem proportional_function_decreases (x y : ℝ) (h : y = proportional_function x) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → proportional_function x₁ > proportional_function x₂ := 
sorry

end proportional_function_decreases_l897_89791


namespace binomial_19_10_l897_89764

theorem binomial_19_10 :
  ∀ (binom : ℕ → ℕ → ℕ),
  binom 17 7 = 19448 → binom 17 9 = 24310 →
  binom 19 10 = 92378 :=
by
  intros
  sorry

end binomial_19_10_l897_89764


namespace ellipse_x1_x2_squared_sum_eq_4_l897_89713

theorem ellipse_x1_x2_squared_sum_eq_4
  (x₁ y₁ x₂ y₂ : ℝ)
  (a b : ℝ)
  (ha : a = 2)
  (hb : b = 1)
  (hM : x₁^2 / a^2 + y₁^2 = 1)
  (hN : x₂^2 / a^2 + y₂^2 = 1)
  (h_slope_product : (y₁ / x₁) * (y₂ / x₂) = -1 / 4) :
  x₁^2 + x₂^2 = 4 :=
by
  sorry

end ellipse_x1_x2_squared_sum_eq_4_l897_89713


namespace farmer_flax_acres_l897_89799

-- Definitions based on conditions
def total_acres : ℕ := 240
def extra_sunflower_acres : ℕ := 80

-- Problem statement
theorem farmer_flax_acres (F : ℕ) (S : ℕ) 
    (h1 : F + S = total_acres) 
    (h2 : S = F + extra_sunflower_acres) : 
    F = 80 :=
by
    -- Proof goes here
    sorry

end farmer_flax_acres_l897_89799


namespace sum_xyz_l897_89783

theorem sum_xyz (x y z : ℝ) (h1 : x + y = 1) (h2 : y + z = 1) (h3 : z + x = 1) : x + y + z = 3 / 2 :=
  sorry

end sum_xyz_l897_89783


namespace tangent_line_at_one_l897_89772

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_at_one : ∀ (x y : ℝ), y = 2 * Real.exp 1 * x - Real.exp 1 → 
  ∃ m b : ℝ, (∀ x: ℝ, f x = m * x + b) ∧ (m = 2 * Real.exp 1) ∧ (b = -Real.exp 1) :=
by
  sorry

end tangent_line_at_one_l897_89772


namespace total_cost_rental_l897_89717

theorem total_cost_rental :
  let rental_fee := 20.99
  let charge_per_mile := 0.25
  let miles_driven := 299
  let total_cost := rental_fee + charge_per_mile * miles_driven
  total_cost = 95.74 := by
{
  sorry
}

end total_cost_rental_l897_89717


namespace population_present_l897_89734

theorem population_present (P : ℝ) (h : P * (1.1)^3 = 79860) : P = 60000 :=
sorry

end population_present_l897_89734


namespace find_positive_int_sol_l897_89700

theorem find_positive_int_sol (a b c d n : ℕ) (h1 : n > 1) (h2 : a ≤ b) (h3 : b ≤ c) :
  ((n^a + n^b + n^c = n^d) ↔ 
  ((a = b ∧ b = c - 1 ∧ c = d - 1 ∧ n = 2) ∨ 
  (a = b ∧ b = c ∧ c = d - 1 ∧ n = 3))) :=
  sorry

end find_positive_int_sol_l897_89700


namespace decreasing_interval_eqn_l897_89711

def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem decreasing_interval_eqn {a : ℝ} : (∀ x : ℝ, x < 6 → deriv (f a) x < 0) ↔ a ≥ 6 :=
sorry

end decreasing_interval_eqn_l897_89711


namespace convert_to_scientific_notation_9600000_l897_89797

theorem convert_to_scientific_notation_9600000 :
  9600000 = 9.6 * 10^6 := 
sorry

end convert_to_scientific_notation_9600000_l897_89797


namespace gcd_q_r_min_value_l897_89754

theorem gcd_q_r_min_value (p q r : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) : Nat.gcd q r = 10 :=
sorry

end gcd_q_r_min_value_l897_89754


namespace inequality_for_positive_reals_l897_89753

open Real

theorem inequality_for_positive_reals 
  (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  a^3 * b + b^3 * c + c^3 * a ≥ a * b * c * (a + b + c) :=
sorry

end inequality_for_positive_reals_l897_89753


namespace greatest_possible_large_chips_l897_89785

theorem greatest_possible_large_chips 
  (s l : ℕ) 
  (p : ℕ) 
  (h1 : s + l = 72) 
  (h2 : s = l + p) 
  (h_prime : Prime p) : 
  l ≤ 35 :=
sorry

end greatest_possible_large_chips_l897_89785


namespace find_width_of_lawn_l897_89755

noncomputable def width_of_lawn
    (length : ℕ)
    (cost : ℕ)
    (cost_per_sq_m : ℕ)
    (road_width : ℕ) : ℕ :=
  let total_area := cost / cost_per_sq_m
  let road_area_length := road_width * length
  let eq_area := total_area - road_area_length
  eq_area / road_width

theorem find_width_of_lawn :
  width_of_lawn 110 4800 3 10 = 50 :=
by
  sorry

end find_width_of_lawn_l897_89755


namespace nth_equation_pattern_l897_89710

theorem nth_equation_pattern (n : ℕ) : (n + 1) * (n^2 - n + 1) - 1 = n^3 :=
by
  sorry

end nth_equation_pattern_l897_89710


namespace distance_between_countries_l897_89737

theorem distance_between_countries (total_distance : ℕ) (spain_germany : ℕ) (spain_other : ℕ) :
  total_distance = 7019 →
  spain_germany = 1615 →
  spain_other = total_distance - spain_germany →
  spain_other = 5404 :=
by
  intros h_total_distance h_spain_germany h_spain_other
  rw [h_total_distance, h_spain_germany] at h_spain_other
  exact h_spain_other

end distance_between_countries_l897_89737


namespace mixed_number_calculation_l897_89794

/-
  We need to define a proof that shows:
  75 * (2 + 3/7 - 5 * (1/3)) / (3 + 1/5 + 2 + 1/6) = -208 + 7/9
-/
theorem mixed_number_calculation :
  75 * ((17 / 7) - (16 / 3)) / ((16 / 5) + (13 / 6)) = -208 + 7 / 9 := by
  sorry

end mixed_number_calculation_l897_89794


namespace triangles_fit_in_pan_l897_89762

theorem triangles_fit_in_pan (pan_length pan_width triangle_base triangle_height : ℝ)
  (h1 : pan_length = 15) (h2 : pan_width = 24) (h3 : triangle_base = 3) (h4 : triangle_height = 4) :
  (pan_length * pan_width) / (1/2 * triangle_base * triangle_height) = 60 :=
by
  sorry

end triangles_fit_in_pan_l897_89762


namespace f_2000_equals_1499001_l897_89702

noncomputable def f (x : ℕ) : ℝ → ℝ := sorry

axiom f_initial : f 0 = 1

axiom f_recursive (x : ℕ) : f (x + 4) = f x + 3 * x + 4

theorem f_2000_equals_1499001 : f 2000 = 1499001 :=
by sorry

end f_2000_equals_1499001_l897_89702


namespace max_metro_lines_l897_89790

theorem max_metro_lines (lines : ℕ) 
  (stations_per_line : ℕ) 
  (max_interchange : ℕ) 
  (max_lines_per_interchange : ℕ) :
  (stations_per_line >= 4) → 
  (max_interchange <= 3) → 
  (max_lines_per_interchange <= 2) → 
  (∀ s_1 s_2, ∃ t_1 t_2, t_1 ≤ max_interchange ∧ t_2 ≤ max_interchange ∧
     (s_1 = t_1 ∨ s_2 = t_1 ∨ s_1 = t_2 ∨ s_2 = t_2)) → 
  lines ≤ 10 :=
by
  sorry

end max_metro_lines_l897_89790


namespace good_walker_catch_up_l897_89765

theorem good_walker_catch_up :
  ∀ x y : ℕ, 
    (x = (100:ℕ) + y) ∧ (x = ((100:ℕ)/(60:ℕ) : ℚ) * y) := 
by
  sorry

end good_walker_catch_up_l897_89765


namespace g_50_equals_zero_l897_89741

noncomputable def g : ℝ → ℝ := sorry

theorem g_50_equals_zero (h : ∀ (x y : ℝ), 0 < x → 0 < y → x * g y - y * g x = g ((x + y) / y)) : g 50 = 0 :=
sorry

end g_50_equals_zero_l897_89741


namespace polygon_sides_l897_89719

theorem polygon_sides (n : ℕ) (sum_of_angles : ℕ) (missing_angle : ℕ) 
  (h1 : sum_of_angles = 3240) 
  (h2 : missing_angle * n / (n - 1) = 2 * sum_of_angles) : 
  n = 20 := 
sorry

end polygon_sides_l897_89719


namespace original_wage_before_increase_l897_89780

theorem original_wage_before_increase (W : ℝ) 
  (h1 : W * 1.4 = 35) : W = 25 := by
  sorry

end original_wage_before_increase_l897_89780


namespace proportion_of_mothers_full_time_jobs_l897_89777

theorem proportion_of_mothers_full_time_jobs
  (P : ℝ) (W : ℝ) (F : ℝ → Prop) (M : ℝ)
  (hwomen : W = 0.4 * P)
  (hfathers_full_time : ∀ p, F p → p = 0.75)
  (hno_full_time : P - (W + 0.75 * (P - W)) = 0.19 * P) :
  M = 0.9 :=
by
  sorry

end proportion_of_mothers_full_time_jobs_l897_89777


namespace integer_solutions_count_l897_89712

theorem integer_solutions_count (x : ℤ) :
  (75 ^ 60 * x ^ 60 > x ^ 120 ∧ x ^ 120 > 3 ^ 240) → ∃ n : ℕ, n = 65 :=
by
  sorry

end integer_solutions_count_l897_89712


namespace cube_volume_increase_l897_89757

variable (a : ℝ)

theorem cube_volume_increase (a : ℝ) : (2 * a)^3 - a^3 = 7 * a^3 :=
by
  sorry

end cube_volume_increase_l897_89757


namespace math_problem_l897_89735

theorem math_problem : 1999^2 - 2000 * 1998 = 1 := 
by
  sorry

end math_problem_l897_89735


namespace copies_per_person_l897_89798

-- Definitions derived from the conditions
def pages_per_contract : ℕ := 20
def total_pages_copied : ℕ := 360
def number_of_people : ℕ := 9

-- Theorem stating the result based on the conditions
theorem copies_per_person : (total_pages_copied / pages_per_contract) / number_of_people = 2 := by
  sorry

end copies_per_person_l897_89798


namespace average_trees_planted_l897_89742

theorem average_trees_planted 
  (A : ℕ) 
  (B : ℕ) 
  (C : ℕ) 
  (h1 : A = 35) 
  (h2 : B = A + 6) 
  (h3 : C = A - 3) : 
  (A + B + C) / 3 = 36 :=
  by
  sorry

end average_trees_planted_l897_89742


namespace Hazel_shirts_proof_l897_89769

variable (H : ℕ)

def shirts_received_by_Razel (h_shirts : ℕ) : ℕ :=
  2 * h_shirts

def total_shirts (h_shirts : ℕ) (r_shirts : ℕ) : ℕ :=
  h_shirts + r_shirts

theorem Hazel_shirts_proof
  (h_shirts : ℕ)
  (r_shirts : ℕ)
  (total : ℕ)
  (H_nonneg : 0 ≤ h_shirts)
  (R_twice_H : r_shirts = shirts_received_by_Razel h_shirts)
  (T_total : total = total_shirts h_shirts r_shirts)
  (total_is_18 : total = 18) :
  h_shirts = 6 :=
by
  sorry

end Hazel_shirts_proof_l897_89769


namespace spirit_concentration_l897_89738

theorem spirit_concentration (vol_a vol_b vol_c : ℕ) (conc_a conc_b conc_c : ℝ)
(h_a : conc_a = 0.45) (h_b : conc_b = 0.30) (h_c : conc_c = 0.10)
(h_vola : vol_a = 4) (h_volb : vol_b = 5) (h_volc : vol_c = 6) : 
  (conc_a * vol_a + conc_b * vol_b + conc_c * vol_c) / (vol_a + vol_b + vol_c) * 100 = 26 := 
by
  sorry

end spirit_concentration_l897_89738


namespace intersection_A_B_l897_89724

-- Definition of sets A and B based on given conditions
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2 * x - 3 }
def B : Set ℝ := {y | ∃ x : ℝ, x < 0 ∧ y = x + 1 / x }

-- Proving the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {y | -4 ≤ y ∧ y ≤ -2} := 
by
  sorry

end intersection_A_B_l897_89724


namespace solution_proof_l897_89750

variable (A B C : ℕ+) (x y : ℚ)
variable (h1 : A > B) (h2 : B > C) (h3 : A = B * (1 + x / 100)) (h4 : B = C * (1 + y / 100))

theorem solution_proof : x = 100 * ((A / (C * (1 + y / 100))) - 1) :=
by
  sorry

end solution_proof_l897_89750


namespace probability_all_switches_on_is_correct_l897_89708

-- Mechanical declaration of the problem
structure SwitchState :=
  (state : Fin 2003 → Bool)

noncomputable def probability_all_on (initial : SwitchState) : ℚ :=
  let satisfying_confs := 2
  let total_confs := 2 ^ 2003
  let p := satisfying_confs / total_confs
  p

-- Definition of the term we want to prove
theorem probability_all_switches_on_is_correct :
  ∀ (initial : SwitchState), probability_all_on initial = 1 / 2 ^ 2002 :=
  sorry

end probability_all_switches_on_is_correct_l897_89708


namespace red_beads_count_is_90_l897_89792

-- Define the arithmetic sequence for red beads
def red_bead_count (n : ℕ) : ℕ := 2 * n

-- The sum of the first n terms in our sequence
def sum_red_beads (n : ℕ) : ℕ := n * (n + 1)

-- Verify the number of terms n such that the sum of red beads remains under 100
def valid_num_terms : ℕ := Nat.sqrt 99

-- Calculate total number of red beads on the necklace
def total_red_beads : ℕ := sum_red_beads valid_num_terms

theorem red_beads_count_is_90 (num_beads : ℕ) (valid : num_beads = 99) : 
  total_red_beads = 90 :=
by
  -- Proof skipped
  sorry

end red_beads_count_is_90_l897_89792


namespace jason_investing_months_l897_89727

noncomputable def initial_investment (total_amount earned_amount_per_month : ℕ) := total_amount / 3
noncomputable def months_investing (initial_investment earned_amount_per_month : ℕ) := (2 * initial_investment) / earned_amount_per_month

theorem jason_investing_months (total_amount earned_amount_per_month : ℕ) 
  (h1 : total_amount = 90) 
  (h2 : earned_amount_per_month = 12) 
  : months_investing (initial_investment total_amount earned_amount_per_month) earned_amount_per_month = 5 := 
by
  sorry

end jason_investing_months_l897_89727


namespace product_of_x_values_product_of_all_possible_x_values_l897_89787

theorem product_of_x_values (x : ℚ) (h : abs ((18 : ℚ) / x - 4) = 3) :
  x = 18 ∨ x = 18 / 7 :=
sorry

theorem product_of_all_possible_x_values (x1 x2 : ℚ) (h1 : abs ((18 : ℚ) / x1 - 4) = 3) (h2 : abs ((18 : ℚ) / x2 - 4) = 3) :
  x1 * x2 = 324 / 7 :=
sorry

end product_of_x_values_product_of_all_possible_x_values_l897_89787


namespace negation_proposition_l897_89770

theorem negation_proposition (p : Prop) (h : ∀ x : ℝ, 2 * x^2 + 1 > 0) : ¬p ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 :=
sorry

end negation_proposition_l897_89770


namespace mutually_prime_sum_l897_89736

open Real

theorem mutually_prime_sum (A B C : ℤ) (h_prime : Int.gcd A (Int.gcd B C) = 1)
    (h_eq : A * log 5 / log 200 + B * log 2 / log 200 = C) : A + B + C = 6 := 
sorry

end mutually_prime_sum_l897_89736


namespace find_a_plus_c_l897_89763

theorem find_a_plus_c (a b c d : ℝ) (h1 : ab + bc + cd + da = 40) (h2 : b + d = 8) : a + c = 5 :=
by
  sorry

end find_a_plus_c_l897_89763


namespace complete_contingency_table_chi_sq_test_result_expected_value_X_l897_89716

noncomputable def probability_set := {x : ℚ // x ≥ 0 ∧ x ≤ 1}

variable (P : probability_set → probability_set)

-- Conditions from the problem
def P_A_given_not_B : probability_set := ⟨2 / 5, by norm_num⟩
def P_B_given_not_A : probability_set := ⟨5 / 8, by norm_num⟩
def P_B : probability_set := ⟨3 / 4, by norm_num⟩

-- Definitions related to counts and probabilities
def total_students : ℕ := 200
def male_students := P_A_given_not_B.val * total_students
def female_students := total_students - male_students
def score_exceeds_85 := P_B.val * total_students
def score_not_exceeds_85 := total_students - score_exceeds_85

-- Expected counts based on given probabilities
def male_score_not_exceeds_85 := P_A_given_not_B.val * score_not_exceeds_85
def female_score_not_exceeds_85 := score_not_exceeds_85 - male_score_not_exceeds_85
def male_score_exceeds_85 := male_students - male_score_not_exceeds_85
def female_score_exceeds_85 := female_students - female_score_not_exceeds_85

-- Chi-squared test independence 
def chi_squared := (total_students * (male_score_not_exceeds_85 * female_score_exceeds_85 - female_score_not_exceeds_85 * male_score_exceeds_85) ^ 2) / 
                    (male_students * female_students * score_not_exceeds_85 * score_exceeds_85)
def is_related : Prop := chi_squared > 10.828

-- Expected distributions and expectation of X
def P_X_0 := (1 / 4) ^ 2 * (1 / 3) ^ 2
def P_X_1 := 2 * (3 / 4) * (1 / 4) * (1 / 3) ^ 2 + 2 * (2 / 3) * (1 / 3) * (1 / 4) ^ 2
def P_X_2 := (3 / 4) ^ 2 * (1 / 3) ^ 2 + (1 / 4) ^ 2 * (2 / 3) ^ 2 + 2 * (2 / 3) * (1 / 3) * (3 / 4) * (1 / 4)
def P_X_3 := (3 / 4) ^ 2 * 2 * (2 / 3) * (1 / 3) + 2 * (3 / 4) * (1 / 4) * (2 / 3) ^ 2
def P_X_4 := (3 / 4) ^ 2 * (2 / 3) ^ 2
def expectation_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3 + 4 * P_X_4

-- Lean theorem statements for answers using the above definitions
theorem complete_contingency_table :
  male_score_not_exceeds_85 + female_score_not_exceeds_85 = score_not_exceeds_85 ∧
  male_score_exceeds_85 + female_score_exceeds_85 = score_exceeds_85 ∧
  male_students + female_students = total_students := sorry

theorem chi_sq_test_result :
  is_related = true := sorry

theorem expected_value_X :
  expectation_X = 17 / 6 := sorry

end complete_contingency_table_chi_sq_test_result_expected_value_X_l897_89716


namespace Danny_more_wrappers_than_bottle_caps_l897_89758

theorem Danny_more_wrappers_than_bottle_caps
  (initial_wrappers : ℕ)
  (initial_bottle_caps : ℕ)
  (found_wrappers : ℕ)
  (found_bottle_caps : ℕ) :
  initial_wrappers = 67 →
  initial_bottle_caps = 35 →
  found_wrappers = 18 →
  found_bottle_caps = 15 →
  (initial_wrappers + found_wrappers) - (initial_bottle_caps + found_bottle_caps) = 35 :=
by
  intros h1 h2 h3 h4
  sorry

end Danny_more_wrappers_than_bottle_caps_l897_89758


namespace expression_value_l897_89728

theorem expression_value (a : ℝ) (h : a = 1/3) : 
  (4 * a⁻¹ - 2 * a⁻¹ / 3) / a^2 = 90 := by
  sorry

end expression_value_l897_89728


namespace n_fifth_minus_n_divisible_by_30_l897_89748

theorem n_fifth_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end n_fifth_minus_n_divisible_by_30_l897_89748


namespace distance_between_points_l897_89781

theorem distance_between_points :
  let p1 := (3, -5)
  let p2 := (-4, 4)
  dist p1 p2 = Real.sqrt 130 := by
  sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

end distance_between_points_l897_89781


namespace total_cranes_folded_l897_89705

-- Definitions based on conditions
def hyerinCranesPerDay : ℕ := 16
def hyerinDays : ℕ := 7
def taeyeongCranesPerDay : ℕ := 25
def taeyeongDays : ℕ := 6

-- Definition of total number of cranes folded by Hyerin and Taeyeong
def totalCranes : ℕ :=
  (hyerinCranesPerDay * hyerinDays) + (taeyeongCranesPerDay * taeyeongDays)

-- Proof statement
theorem total_cranes_folded : totalCranes = 262 := by 
  sorry

end total_cranes_folded_l897_89705


namespace solve_for_x_l897_89784

theorem solve_for_x (x : ℤ) (h : (-1) * 2 * x * 4 = 24) : x = -3 := by
  sorry

end solve_for_x_l897_89784


namespace tan_theta_value_l897_89788

noncomputable def tan_theta (θ : ℝ) : ℝ :=
  if (0 < θ) ∧ (θ < 2 * Real.pi) ∧ (Real.cos (θ / 2) = 1 / 3) then
    (2 * (2 * Real.sqrt 2) / (1 - (2 * Real.sqrt 2) ^ 2))
  else
    0 -- added default value for well-definedness

theorem tan_theta_value (θ : ℝ) (h₀: 0 < θ) (h₁ : θ < 2 * Real.pi) (h₂ : Real.cos (θ / 2) = 1 / 3) : 
  tan_theta θ = -4 * Real.sqrt 2 / 7 :=
by
  sorry

end tan_theta_value_l897_89788


namespace passenger_drop_ratio_l897_89773

theorem passenger_drop_ratio (initial_passengers passengers_at_first passengers_at_second final_passengers x : ℕ)
  (h0 : initial_passengers = 288)
  (h1 : passengers_at_first = initial_passengers - (initial_passengers / 3) + 280)
  (h2 : passengers_at_second = passengers_at_first - x + 12)
  (h3 : final_passengers = 248)
  (h4 : passengers_at_second = final_passengers) :
  x / passengers_at_first = 1 / 2 :=
by
  sorry

end passenger_drop_ratio_l897_89773


namespace symmetry_line_intersection_l897_89761

theorem symmetry_line_intersection 
  (k : ℝ) (k_pos : k > 0) (k_ne_one : k ≠ 1)
  (k1 : ℝ) (h_sym : ∀ (P : ℝ × ℝ), (P.2 = k1 * P.1 + 1) ↔ P.2 - 1 = k * (P.1 + 1) + 1)
  (H : ∀ M : ℝ × ℝ, (M.2 = k * M.1 + 1) → (M.1^2 / 4 + M.2^2 = 1)) :
  (k * k1 = 1) ∧ (∀ k : ℝ, ∃ P : ℝ × ℝ, (P.fst = 0) ∧ (P.snd = -5 / 3)) :=
sorry

end symmetry_line_intersection_l897_89761


namespace joey_total_study_time_l897_89740

def hours_weekdays (hours_per_night : Nat) (nights_per_week : Nat) : Nat :=
  hours_per_night * nights_per_week

def hours_weekends (hours_per_day : Nat) (days_per_weekend : Nat) : Nat :=
  hours_per_day * days_per_weekend

def total_weekly_study_time (weekday_hours : Nat) (weekend_hours : Nat) : Nat :=
  weekday_hours + weekend_hours

def total_study_time_in_weeks (weekly_hours : Nat) (weeks : Nat) : Nat :=
  weekly_hours * weeks

theorem joey_total_study_time :
  let hours_per_night := 2
  let nights_per_week := 5
  let hours_per_day := 3
  let days_per_weekend := 2
  let weeks := 6
  hours_weekdays hours_per_night nights_per_week +
  hours_weekends hours_per_day days_per_weekend = 16 →
  total_study_time_in_weeks 16 weeks = 96 :=
by 
  intros h1 h2 h3 h4 h5
  have weekday_hours := hours_weekdays h1 h2
  have weekend_hours := hours_weekends h3 h4
  have total_weekly := total_weekly_study_time weekday_hours weekend_hours
  sorry

end joey_total_study_time_l897_89740


namespace calculate_expression_l897_89720

theorem calculate_expression : 103^3 - 3 * 103^2 + 3 * 103 - 1 = 1061208 := by
  sorry

end calculate_expression_l897_89720


namespace total_increase_by_five_l897_89722

-- Let B be the number of black balls
variable (B : ℕ)
-- Let W be the number of white balls
variable (W : ℕ)
-- Initially the total number of balls
def T := B + W
-- If the number of black balls is increased to 5 times the original, the total becomes twice the original
axiom h1 : 5 * B + W = 2 * (B + W)
-- If the number of white balls is increased to 5 times the original 
def k : ℕ := 5
-- The new total number of balls 
def new_total := B + k * W

-- Prove that the new total is 4 times the original total.
theorem total_increase_by_five : new_total = 4 * T :=
by
sorry

end total_increase_by_five_l897_89722


namespace swallow_distance_flew_l897_89782

/-- The TGV departs from Paris at 150 km/h toward Marseille, which is 800 km away, while an intercité departs from Marseille at 50 km/h toward Paris at the same time. A swallow perched on the TGV takes off at that moment, flying at 200 km/h toward Marseille. We aim to prove that the distance flown by the swallow when the two trains meet is 800 km. -/
theorem swallow_distance_flew :
  let distance := 800 -- distance between Paris and Marseille in km
  let speed_TGV := 150 -- speed of TGV in km/h
  let speed_intercite := 50 -- speed of intercité in km/h
  let speed_swallow := 200 -- speed of swallow in km/h
  let combined_speed := speed_TGV + speed_intercite
  let time_to_meet := distance / combined_speed
  let distance_swallow_traveled := speed_swallow * time_to_meet
  distance_swallow_traveled = 800 := 
by
  sorry

end swallow_distance_flew_l897_89782


namespace quadratic_rewrite_l897_89752

theorem quadratic_rewrite :
  ∃ d e f : ℤ, (4 * (x : ℝ)^2 - 24 * x + 35 = (d * x + e)^2 + f) ∧ (d * e = -12) :=
by
  sorry

end quadratic_rewrite_l897_89752


namespace find_a_l897_89725

open Set

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

theorem find_a (a : ℝ) :
  ∅ ⊂ (A a ∩ B) ∧ A a ∩ C = ∅ → a = -2 :=
by
  sorry

end find_a_l897_89725


namespace value_of_x_plus_y_l897_89786

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : 1 / x - 1 / y = -6) : x + y = -4 / 5 :=
sorry

end value_of_x_plus_y_l897_89786


namespace garden_area_l897_89767

-- Definitions for the conditions
def perimeter : ℕ := 36
def width : ℕ := 10

-- Define the length using the perimeter and width
def length : ℕ := (perimeter - 2 * width) / 2

-- Define the area using the length and width
def area : ℕ := length * width

-- The theorem to prove the area is 80 square feet given the conditions
theorem garden_area : area = 80 :=
by 
  -- Here we use sorry to skip the proof
  sorry

end garden_area_l897_89767


namespace compare_abc_l897_89739

noncomputable def a : ℝ := Real.log 4 / Real.log 3
noncomputable def b : ℝ := (1 / 3 : ℝ) ^ (1 / 3 : ℝ)
noncomputable def c : ℝ := (3 : ℝ) ^ (-1 / 4 : ℝ)

theorem compare_abc : b < c ∧ c < a :=
by
  sorry

end compare_abc_l897_89739


namespace range_of_a_l897_89733

-- Define set A
def setA (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a^2 + 1

-- Define set B
def setB (x a : ℝ) : Prop := (x - 2) * (x - (3 * a + 1)) ≤ 0

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, setA x a → setB x a) ↔ (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) :=
sorry

end range_of_a_l897_89733


namespace average_speed_with_stoppages_l897_89789

/--The average speed of the bus including stoppages is 20 km/hr, 
  given that the bus stops for 40 minutes per hour and 
  has an average speed of 60 km/hr excluding stoppages.--/
theorem average_speed_with_stoppages 
  (avg_speed_without_stoppages : ℝ)
  (stoppage_time_per_hour : ℕ) 
  (running_time_per_hour : ℕ) 
  (avg_speed_with_stoppages : ℝ) 
  (h1 : avg_speed_without_stoppages = 60) 
  (h2 : stoppage_time_per_hour = 40) 
  (h3 : running_time_per_hour = 20) 
  (h4 : running_time_per_hour + stoppage_time_per_hour = 60):
  avg_speed_with_stoppages = 20 := 
sorry

end average_speed_with_stoppages_l897_89789


namespace find_constants_l897_89703

def f (x : ℝ) (a : ℝ) : ℝ := 2 * x ^ 3 + a * x
def g (x : ℝ) (b c : ℝ) : ℝ := b * x ^ 2 + c
def f' (x : ℝ) (a : ℝ) : ℝ := 6 * x ^ 2 + a
def g' (x : ℝ) (b : ℝ) : ℝ := 2 * b * x

theorem find_constants (a b c : ℝ) :
  f 2 a = 0 ∧ g 2 b c = 0 ∧ f' 2 a = g' 2 b →
  a = -8 ∧ b = 4 ∧ c = -16 :=
by
  intro h
  sorry

end find_constants_l897_89703


namespace ratio_female_to_male_l897_89729

variable {f m c : ℕ}

/-- 
  The following conditions are given:
  - The average age of female members is 35 years.
  - The average age of male members is 30 years.
  - The average age of children members is 10 years.
  - The average age of the entire membership is 25 years.
  - The number of children members is equal to the number of male members.
  We need to show that the ratio of female to male members is 1.
-/
theorem ratio_female_to_male (h1 : c = m)
  (h2 : 35 * f + 40 * m = 25 * (f + 2 * m)) :
  f = m :=
by sorry

end ratio_female_to_male_l897_89729


namespace multiple_of_six_and_nine_l897_89726

-- Definitions: x is a multiple of 6, y is a multiple of 9.
def is_multiple_of_six (x : ℤ) : Prop := ∃ m : ℤ, x = 6 * m
def is_multiple_of_nine (y : ℤ) : Prop := ∃ n : ℤ, y = 9 * n

-- Assertions: Given the conditions, prove the following.
theorem multiple_of_six_and_nine (x y : ℤ)
  (hx : is_multiple_of_six x) (hy : is_multiple_of_nine y) :
  ((∃ k : ℤ, x - y = 3 * k) ∧
   (∃ m n : ℤ, x = 6 * m ∧ y = 9 * n ∧ (2 * m - 3 * n) % 3 ≠ 0)) :=
by
  sorry

end multiple_of_six_and_nine_l897_89726


namespace relatively_prime_solutions_l897_89793

theorem relatively_prime_solutions  (x y : ℤ) (h_rel_prime : gcd x y = 1) : 
  2 * (x^3 - x) = 5 * (y^3 - y) ↔ 
  (x = 0 ∧ (y = 1 ∨ y = -1)) ∨ 
  (x = 1 ∧ y = 0) ∨
  (x = -1 ∧ y = 0) ∨
  (x = 4 ∧ (y = 3 ∨ y = -3)) ∨ 
  (x = -4 ∧ (y = -3 ∨ y = 3)) ∨
  (x = 1 ∧ y = -1) ∨
  (x = -1 ∧ y = 1) ∨
  (x = 0 ∧ y = 0) :=
by sorry

end relatively_prime_solutions_l897_89793


namespace x_squared_plus_inverse_squared_l897_89743

theorem x_squared_plus_inverse_squared (x : ℝ) (h : x + 1/x = 3.5) : x^2 + (1/x)^2 = 10.25 :=
by sorry

end x_squared_plus_inverse_squared_l897_89743


namespace no_pairs_probability_l897_89723

-- Define the number of socks and initial conditions
def pairs_of_socks : ℕ := 3
def total_socks : ℕ := pairs_of_socks * 2

-- Probabilistic outcome space for no pairs in first three draws
def probability_no_pairs_in_first_three_draws : ℚ :=
  (4/5) * (1/2)

-- Theorem stating that probability of no matching pairs in the first three draws is 2/5
theorem no_pairs_probability : probability_no_pairs_in_first_three_draws = 2/5 := by
  sorry

end no_pairs_probability_l897_89723


namespace remainder_444_444_mod_13_l897_89706

theorem remainder_444_444_mod_13 :
  444 ≡ 3 [MOD 13] →
  3^3 ≡ 1 [MOD 13] →
  444^444 ≡ 1 [MOD 13] := by
  intros h1 h2
  sorry

end remainder_444_444_mod_13_l897_89706


namespace probability_even_sum_5_balls_drawn_l897_89715

theorem probability_even_sum_5_balls_drawn :
  let total_ways := (Nat.choose 12 5)
  let favorable_ways := (Nat.choose 6 0) * (Nat.choose 6 5) + 
                        (Nat.choose 6 2) * (Nat.choose 6 3) + 
                        (Nat.choose 6 4) * (Nat.choose 6 1)
  favorable_ways / total_ways = 1 / 2 :=
by sorry

end probability_even_sum_5_balls_drawn_l897_89715


namespace population_relation_l897_89747

-- Conditions: average life expectancies
def life_expectancy_gondor : ℝ := 64
def life_expectancy_numenor : ℝ := 92
def combined_life_expectancy (g n : ℕ) : ℝ := 85

-- Proof Problem: Given the conditions, prove the population relation
theorem population_relation (g n : ℕ) (h1 : life_expectancy_gondor * g + life_expectancy_numenor * n = combined_life_expectancy g n * (g + n)) : n = 3 * g :=
by
  sorry

end population_relation_l897_89747


namespace rational_solutions_of_quadratic_l897_89774

theorem rational_solutions_of_quadratic (k : ℕ) (h_positive : k > 0) :
  (∃ p q : ℚ, p * p + 30 * p * q + k * (q * q) = 0) ↔ k = 9 ∨ k = 15 :=
sorry

end rational_solutions_of_quadratic_l897_89774


namespace product_of_roots_cubic_eq_l897_89779

theorem product_of_roots_cubic_eq (α : Type _) [Field α] :
  (∃ (r1 r2 r3 : α), (r1 * r2 * r3 = 6) ∧ (r1 + r2 + r3 = 6) ∧ (r1 * r2 + r1 * r3 + r2 * r3 = 11)) :=
by
  sorry

end product_of_roots_cubic_eq_l897_89779


namespace barbie_earrings_l897_89745

theorem barbie_earrings (total_earrings_alissa : ℕ) (alissa_triple_given : ℕ → ℕ) 
  (given_earrings_double_bought : ℕ → ℕ) (pairs_of_earrings : ℕ) : 
  total_earrings_alissa = 36 → 
  alissa_triple_given (total_earrings_alissa / 3) = total_earrings_alissa → 
  given_earrings_double_bought (total_earrings_alissa / 3) = total_earrings_alissa →
  pairs_of_earrings = 12 :=
by
  intros h1 h2 h3
  sorry

end barbie_earrings_l897_89745


namespace basis_of_R3_l897_89760

def e1 : ℝ × ℝ × ℝ := (1, 0, 0)
def e2 : ℝ × ℝ × ℝ := (0, 1, 0)
def e3 : ℝ × ℝ × ℝ := (0, 0, 1)

theorem basis_of_R3 :
  ∀ (u : ℝ × ℝ × ℝ), ∃ (α β γ : ℝ), u = α • e1 + β • e2 + γ • e3 ∧ 
  (∀ (a b c : ℝ), a • e1 + b • e2 + c • e3 = (0, 0, 0) → a = 0 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end basis_of_R3_l897_89760


namespace defective_units_percentage_l897_89749

variables (D : ℝ)

-- 4% of the defective units are shipped for sale
def percent_defective_shipped : ℝ := 0.04

-- 0.24% of the units produced are defective units that are shipped for sale
def percent_total_defective_shipped : ℝ := 0.0024

-- The theorem to prove: the percentage of the units produced that are defective is 0.06
theorem defective_units_percentage (h : percent_defective_shipped * D = percent_total_defective_shipped) : D = 0.06 :=
sorry

end defective_units_percentage_l897_89749


namespace solve_for_x_l897_89796

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.07 * (25 + x) = 15.1) : x = 111.25 :=
by
  sorry

end solve_for_x_l897_89796


namespace distance_between_parallel_lines_l897_89718

/-- Given two parallel lines y=2x and y=2x+5, the distance between them is √5. -/
theorem distance_between_parallel_lines :
  let A := -2
  let B := 1
  let C1 := 0
  let C2 := -5
  let distance := (|C2 - C1|: ℝ) / Real.sqrt (A ^ 2 + B ^ 2)
  distance = Real.sqrt 5 := by
  -- Assuming calculations as done in the original solution
  sorry

end distance_between_parallel_lines_l897_89718


namespace solution_80_percent_needs_12_ounces_l897_89721

theorem solution_80_percent_needs_12_ounces:
  ∀ (x y: ℝ), (x + y = 40) → (0.30 * x + 0.80 * y = 0.45 * 40) → (y = 12) :=
by
  intros x y h1 h2
  sorry

end solution_80_percent_needs_12_ounces_l897_89721


namespace range_of_function_l897_89731

-- Given conditions 
def independent_variable_range (x : ℝ) : Prop := x ≥ 2

-- Proof statement (no proof only statement with "sorry")
theorem range_of_function (x : ℝ) (y : ℝ) (h : y = Real.sqrt (x - 2)) : independent_variable_range x :=
by sorry

end range_of_function_l897_89731


namespace recurring_subtraction_l897_89795

theorem recurring_subtraction (x y : ℚ) (h1 : x = 35 / 99) (h2 : y = 7 / 9) : x - y = -14 / 33 := by
  sorry

end recurring_subtraction_l897_89795
