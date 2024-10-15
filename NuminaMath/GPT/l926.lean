import Mathlib

namespace NUMINAMATH_GPT_supremum_neg_frac_bound_l926_92668

noncomputable def supremum_neg_frac (a b : ℝ) : ℝ :=
  - (1 / (2 * a)) - (2 / b)

theorem supremum_neg_frac_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  supremum_neg_frac a b ≤ - 9 / 2 :=
sorry

end NUMINAMATH_GPT_supremum_neg_frac_bound_l926_92668


namespace NUMINAMATH_GPT_evaluate_g_inv_l926_92691

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 6)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 7)
variable (h_inv1 : g_inv 6 = 4)
variable (h_inv2 : g_inv 7 = 3)
variable (h_inv_eq : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x)

theorem evaluate_g_inv :
  g_inv (g_inv 6 + g_inv 7) = 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_g_inv_l926_92691


namespace NUMINAMATH_GPT_congruence_is_sufficient_but_not_necessary_for_equal_area_l926_92605

-- Definition of conditions
def Congruent (Δ1 Δ2 : Type) : Prop := sorry -- Definition of congruent triangles
def EqualArea (Δ1 Δ2 : Type) : Prop := sorry -- Definition of triangles with equal area

-- Theorem statement
theorem congruence_is_sufficient_but_not_necessary_for_equal_area 
  (Δ1 Δ2 : Type) :
  (Congruent Δ1 Δ2 → EqualArea Δ1 Δ2) ∧ (¬ (EqualArea Δ1 Δ2 → Congruent Δ1 Δ2)) :=
sorry

end NUMINAMATH_GPT_congruence_is_sufficient_but_not_necessary_for_equal_area_l926_92605


namespace NUMINAMATH_GPT_total_carrots_l926_92659

def Joan_carrots : ℕ := 29
def Jessica_carrots : ℕ := 11

theorem total_carrots : Joan_carrots + Jessica_carrots = 40 := by
  sorry

end NUMINAMATH_GPT_total_carrots_l926_92659


namespace NUMINAMATH_GPT_emily_necklaces_l926_92629

theorem emily_necklaces (n beads_per_necklace total_beads : ℕ) (h1 : beads_per_necklace = 8) (h2 : total_beads = 16) : n = total_beads / beads_per_necklace → n = 2 :=
by sorry

end NUMINAMATH_GPT_emily_necklaces_l926_92629


namespace NUMINAMATH_GPT_true_propositions_l926_92660

-- Definitions according to conditions:
def p (x y : ℝ) : Prop := x > y → -x < -y
def q (x y : ℝ) : Prop := x > y → x^2 > y^2

-- Given that p is true and q is false.
axiom p_true {x y : ℝ} : p x y
axiom q_false {x y : ℝ} : ¬ q x y

-- Proving the actual propositions that are true:
theorem true_propositions (x y : ℝ) : 
  (p x y ∨ q x y) ∧ (p x y ∧ ¬ q x y) :=
by
  have h1 : p x y := p_true
  have h2 : ¬ q x y := q_false
  constructor
  · left; exact h1
  · constructor; assumption; assumption

end NUMINAMATH_GPT_true_propositions_l926_92660


namespace NUMINAMATH_GPT_range_of_k_for_ellipse_l926_92673

def represents_ellipse (x y k : ℝ) : Prop :=
  (k^2 - 3 > 0) ∧ 
  (k - 1 > 0) ∧ 
  (k - 1 ≠ k^2 - 3)

theorem range_of_k_for_ellipse (k : ℝ) : 
  represents_ellipse x y k → k ∈ Set.Ioo (-Real.sqrt 3) (-1) ∪ Set.Ioo (-1) 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_for_ellipse_l926_92673


namespace NUMINAMATH_GPT_initial_dogs_count_is_36_l926_92650

-- Conditions
def initial_cats := 29
def adopted_dogs := 20
def additional_cats := 12
def total_pets := 57

-- Calculate total cats
def total_cats := initial_cats + additional_cats

-- Calculate initial dogs
def initial_dogs (initial_dogs : ℕ) : Prop :=
(initial_dogs - adopted_dogs) + total_cats = total_pets

-- Prove that initial dogs (D) is 36
theorem initial_dogs_count_is_36 : initial_dogs 36 :=
by
-- Here should contain the proof which is omitted
sorry

end NUMINAMATH_GPT_initial_dogs_count_is_36_l926_92650


namespace NUMINAMATH_GPT_polynomial_A_polynomial_B_l926_92676

-- Problem (1): Prove that A = 6x^3 + 8x^2 + x - 1 given the conditions.
theorem polynomial_A :
  ∀ (x : ℝ),
  (2 * x^2 * (3 * x + 4) + (x - 1) = 6 * x^3 + 8 * x^2 + x - 1) :=
by
  intro x
  sorry

-- Problem (2): Prove that B = 6x^2 - 19x + 9 given the conditions.
theorem polynomial_B :
  ∀ (x : ℝ),
  ((2 * x - 6) * (3 * x - 1) + (x + 3) = 6 * x^2 - 19 * x + 9) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_polynomial_A_polynomial_B_l926_92676


namespace NUMINAMATH_GPT_evaluate_expression_l926_92643

open BigOperators

theorem evaluate_expression : 
  ∀ (x y : ℤ), x = -1 → y = 1 → 2 * (x^2 * y + x * y) - 3 * (x^2 * y - x * y) - 5 * x * y = -1 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l926_92643


namespace NUMINAMATH_GPT_number_of_moles_of_NaCl_l926_92632

theorem number_of_moles_of_NaCl
  (moles_NaOH : ℕ)
  (moles_Cl2 : ℕ)
  (reaction : 2 * moles_NaOH + moles_Cl2 = 2 * moles_NaOH + 1) :
  2 * moles_Cl2 = 2 := by 
  sorry

end NUMINAMATH_GPT_number_of_moles_of_NaCl_l926_92632


namespace NUMINAMATH_GPT_kamal_age_problem_l926_92696

theorem kamal_age_problem (K S : ℕ) 
  (h1 : K - 8 = 4 * (S - 8)) 
  (h2 : K + 8 = 2 * (S + 8)) : 
  K = 40 := 
by sorry

end NUMINAMATH_GPT_kamal_age_problem_l926_92696


namespace NUMINAMATH_GPT_Elaine_rent_percentage_l926_92638

variable (E : ℝ) (last_year_rent : ℝ) (this_year_rent : ℝ)

def Elaine_last_year_earnings (E : ℝ) : ℝ := E

def Elaine_last_year_rent (E : ℝ) : ℝ := 0.20 * E

def Elaine_this_year_earnings (E : ℝ) : ℝ := 1.25 * E

def Elaine_this_year_rent (E : ℝ) : ℝ := 0.30 * (1.25 * E)

theorem Elaine_rent_percentage 
  (E : ℝ) 
  (last_year_rent := Elaine_last_year_rent E)
  (this_year_rent := Elaine_this_year_rent E) :
  (this_year_rent / last_year_rent) * 100 = 187.5 := 
by sorry

end NUMINAMATH_GPT_Elaine_rent_percentage_l926_92638


namespace NUMINAMATH_GPT_book_cost_l926_92674

variable {b m : ℝ}

theorem book_cost (h1 : b + m = 2.10) (h2 : b = m + 2) : b = 2.05 :=
by
  sorry

end NUMINAMATH_GPT_book_cost_l926_92674


namespace NUMINAMATH_GPT_total_points_l926_92602

def jon_points (sam_points : ℕ) : ℕ := 2 * sam_points + 3
def sam_points (alex_points : ℕ) : ℕ := alex_points / 2
def jack_points (jon_points : ℕ) : ℕ := jon_points + 5
def tom_points (jon_points jack_points : ℕ) : ℕ := jon_points + jack_points - 4
def alex_points : ℕ := 18

theorem total_points : jon_points (sam_points alex_points) + 
                       jack_points (jon_points (sam_points alex_points)) + 
                       tom_points (jon_points (sam_points alex_points)) 
                       (jack_points (jon_points (sam_points alex_points))) + 
                       sam_points alex_points + 
                       alex_points = 117 :=
by sorry

end NUMINAMATH_GPT_total_points_l926_92602


namespace NUMINAMATH_GPT_employee_salary_l926_92699

theorem employee_salary (x y : ℝ) (h1 : x + y = 770) (h2 : x = 1.2 * y) : y = 350 :=
by
  sorry

end NUMINAMATH_GPT_employee_salary_l926_92699


namespace NUMINAMATH_GPT_velocity_at_t_10_time_to_reach_max_height_max_height_l926_92604

-- Define the height function H(t)
def H (t : ℝ) : ℝ := 200 * t - 4.9 * t^2

-- Define the velocity function v(t) as the derivative of H(t)
def v (t : ℝ) : ℝ := 200 - 9.8 * t

-- Theorem: The velocity of the body at t = 10 seconds
theorem velocity_at_t_10 : v 10 = 102 := by
  sorry

-- Theorem: The time to reach maximum height
theorem time_to_reach_max_height : (∃ t : ℝ, v t = 0 ∧ t = 200 / 9.8) := by
  sorry

-- Theorem: The maximum height the body will reach
theorem max_height : H (200 / 9.8) = 2040.425 := by
  sorry

end NUMINAMATH_GPT_velocity_at_t_10_time_to_reach_max_height_max_height_l926_92604


namespace NUMINAMATH_GPT_find_point_B_l926_92664

theorem find_point_B (A B : ℝ × ℝ) (a : ℝ × ℝ)
  (hA : A = (-1, -5)) 
  (ha : a = (2, 3)) 
  (hAB : B - A = 3 • a) : 
  B = (5, 4) := sorry

end NUMINAMATH_GPT_find_point_B_l926_92664


namespace NUMINAMATH_GPT_min_inverse_ab_l926_92666

theorem min_inverse_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 6) : 
  ∃ (m : ℝ), (m = 2 / 9) ∧ (∀ (a b : ℝ), a > 0 → b > 0 → a + 2 * b = 6 → 1/(a * b) ≥ m) :=
by
  sorry

end NUMINAMATH_GPT_min_inverse_ab_l926_92666


namespace NUMINAMATH_GPT_zero_integers_satisfy_conditions_l926_92624

noncomputable def satisfies_conditions (n : ℤ) : Prop :=
  ∃ k : ℤ, n * (25 - n) = k^2 * (25 - n)^2 ∧ n % 3 = 0

theorem zero_integers_satisfy_conditions :
  (∃ n : ℤ, satisfies_conditions n) → False := by
  sorry

end NUMINAMATH_GPT_zero_integers_satisfy_conditions_l926_92624


namespace NUMINAMATH_GPT_difference_in_spending_l926_92651

-- Condition: original prices and discounts
def original_price_candy_bar : ℝ := 6
def discount_candy_bar : ℝ := 0.25
def original_price_chocolate : ℝ := 3
def discount_chocolate : ℝ := 0.10

-- The theorem to prove
theorem difference_in_spending : 
  (original_price_candy_bar * (1 - discount_candy_bar) - original_price_chocolate * (1 - discount_chocolate)) = 1.80 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_spending_l926_92651


namespace NUMINAMATH_GPT_rook_placement_5x5_l926_92612

theorem rook_placement_5x5 :
  ∀ (board : Fin 5 → Fin 5) (distinct : Function.Injective board),
  ∃ (ways : Nat), ways = 120 := by
  sorry

end NUMINAMATH_GPT_rook_placement_5x5_l926_92612


namespace NUMINAMATH_GPT_second_exponent_base_ends_in_1_l926_92646

theorem second_exponent_base_ends_in_1 
  (x : ℕ) 
  (h : ((1023 ^ 3923) + (x ^ 3921)) % 10 = 8) : 
  x % 10 = 1 := 
by sorry

end NUMINAMATH_GPT_second_exponent_base_ends_in_1_l926_92646


namespace NUMINAMATH_GPT_twenty_percent_greater_than_40_l926_92618

theorem twenty_percent_greater_than_40 (x : ℝ) (h : x = 40 + 0.2 * 40) : x = 48 := by
sorry

end NUMINAMATH_GPT_twenty_percent_greater_than_40_l926_92618


namespace NUMINAMATH_GPT_root_in_interval_implies_a_in_range_l926_92625

theorem root_in_interval_implies_a_in_range {a : ℝ} (h : ∃ x : ℝ, x ≤ 1 ∧ 2^x - a^2 - a = 0) : 0 < a ∧ a ≤ 1 := sorry

end NUMINAMATH_GPT_root_in_interval_implies_a_in_range_l926_92625


namespace NUMINAMATH_GPT_fill_pipe_fraction_l926_92611

theorem fill_pipe_fraction (x : ℝ) (h : x = 1 / 2) : x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fill_pipe_fraction_l926_92611


namespace NUMINAMATH_GPT_value_is_correct_l926_92657

-- Define the number
def initial_number : ℝ := 4400

-- Define the value calculation in Lean
def value : ℝ := 0.15 * (0.30 * (0.50 * initial_number))

-- The theorem statement
theorem value_is_correct : value = 99 := by
  sorry

end NUMINAMATH_GPT_value_is_correct_l926_92657


namespace NUMINAMATH_GPT_field_length_l926_92656

theorem field_length 
  (w l : ℝ)
  (pond_area : ℝ := 25)
  (h1 : l = 2 * w)
  (h2 : pond_area = 25)
  (h3 : pond_area = (1 / 8) * (l * w)) :
  l = 20 :=
by
  sorry

end NUMINAMATH_GPT_field_length_l926_92656


namespace NUMINAMATH_GPT_ab_eq_zero_l926_92694

theorem ab_eq_zero (a b : ℤ) (h : ∀ m n : ℕ, ∃ k : ℤ, a * (m^2 : ℤ) + b * (n^2 : ℤ) = k^2) : a * b = 0 :=
by
  sorry

end NUMINAMATH_GPT_ab_eq_zero_l926_92694


namespace NUMINAMATH_GPT_determine_value_of_product_l926_92606

theorem determine_value_of_product (x : ℝ) (h : (x - 2) * (x + 2) = 2021) : (x - 1) * (x + 1) = 2024 := 
by 
  sorry

end NUMINAMATH_GPT_determine_value_of_product_l926_92606


namespace NUMINAMATH_GPT_car_speed_second_hour_l926_92631

theorem car_speed_second_hour (s1 s2 : ℝ) (h1 : s1 = 10) (h2 : (s1 + s2) / 2 = 35) : s2 = 60 := by
  sorry

end NUMINAMATH_GPT_car_speed_second_hour_l926_92631


namespace NUMINAMATH_GPT_students_no_A_l926_92667

def total_students : Nat := 40
def students_A_chemistry : Nat := 10
def students_A_physics : Nat := 18
def students_A_both : Nat := 6

theorem students_no_A : (total_students - (students_A_chemistry + students_A_physics - students_A_both)) = 18 :=
by
  sorry

end NUMINAMATH_GPT_students_no_A_l926_92667


namespace NUMINAMATH_GPT_no_triangle_sides_exist_l926_92607

theorem no_triangle_sides_exist (x y z : ℝ) (h_triangle_sides : x > 0 ∧ y > 0 ∧ z > 0)
  (h_triangle_inequality : x < y + z ∧ y < x + z ∧ z < x + y) :
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
sorry

end NUMINAMATH_GPT_no_triangle_sides_exist_l926_92607


namespace NUMINAMATH_GPT_jia_profits_1_yuan_l926_92665

-- Definition of the problem conditions
def initial_cost : ℝ := 1000
def profit_rate : ℝ := 0.1
def loss_rate : ℝ := 0.1
def resale_rate : ℝ := 0.9

-- Defined transactions with conditions
def jia_selling_price1 : ℝ := initial_cost * (1 + profit_rate)
def yi_selling_price_to_jia : ℝ := jia_selling_price1 * (1 - loss_rate)
def jia_selling_price2 : ℝ := yi_selling_price_to_jia * resale_rate

-- Final net income calculation
def jia_net_income : ℝ := -initial_cost + jia_selling_price1 - yi_selling_price_to_jia + jia_selling_price2

-- Lean statement to be proved
theorem jia_profits_1_yuan : jia_net_income = 1 := sorry

end NUMINAMATH_GPT_jia_profits_1_yuan_l926_92665


namespace NUMINAMATH_GPT_daughter_age_is_10_l926_92613

variable (D : ℕ)

-- Conditions
def father_current_age (D : ℕ) : ℕ := 4 * D
def father_age_in_20_years (D : ℕ) : ℕ := father_current_age D + 20
def daughter_age_in_20_years (D : ℕ) : ℕ := D + 20

-- Theorem statement
theorem daughter_age_is_10 :
  father_current_age D = 40 →
  father_age_in_20_years D = 2 * daughter_age_in_20_years D →
  D = 10 :=
by
  -- Here would be the proof steps to show that D = 10 given the conditions
  sorry

end NUMINAMATH_GPT_daughter_age_is_10_l926_92613


namespace NUMINAMATH_GPT_boys_more_than_girls_l926_92640

-- Definitions of the conditions
def total_students : ℕ := 100
def boy_ratio : ℕ := 3
def girl_ratio : ℕ := 2

-- Statement of the problem
theorem boys_more_than_girls :
  (total_students * boy_ratio) / (boy_ratio + girl_ratio) - (total_students * girl_ratio) / (boy_ratio + girl_ratio) = 20 :=
by
  sorry

end NUMINAMATH_GPT_boys_more_than_girls_l926_92640


namespace NUMINAMATH_GPT_original_radius_new_perimeter_l926_92623

variable (r : ℝ)

theorem original_radius_new_perimeter (h : (π * (r + 5)^2 = 4 * π * r^2)) :
  r = 5 ∧ 2 * π * (r + 5) = 20 * π :=
by
  sorry

end NUMINAMATH_GPT_original_radius_new_perimeter_l926_92623


namespace NUMINAMATH_GPT_prime_15p_plus_one_l926_92671

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_15p_plus_one (p q : ℕ) 
  (hp : is_prime p) 
  (hq : q = 15 * p + 1) 
  (hq_prime : is_prime q) :
  q = 31 :=
sorry

end NUMINAMATH_GPT_prime_15p_plus_one_l926_92671


namespace NUMINAMATH_GPT_sourav_distance_l926_92658

def D (t : ℕ) : ℕ := 20 * t

theorem sourav_distance :
  ∀ (t : ℕ), 20 * t = 25 * (t - 1) → 20 * t = 100 :=
by
  intros t h
  sorry

end NUMINAMATH_GPT_sourav_distance_l926_92658


namespace NUMINAMATH_GPT_sam_average_speed_l926_92633

theorem sam_average_speed :
  let total_time := 7 -- total time from 7 a.m. to 2 p.m.
  let rest_time := 1 -- rest period from 9 a.m. to 10 a.m.
  let effective_time := total_time - rest_time
  let total_distance := 200 -- total miles covered
  let avg_speed := total_distance / effective_time
  avg_speed = 33.3 :=
sorry

end NUMINAMATH_GPT_sam_average_speed_l926_92633


namespace NUMINAMATH_GPT_find_solutions_of_x4_minus_16_l926_92685

noncomputable def solution_set : Set Complex :=
  {2, -2, Complex.I * 2, -Complex.I * 2}

theorem find_solutions_of_x4_minus_16 :
  {x : Complex | x^4 - 16 = 0} = solution_set :=
by
  sorry

end NUMINAMATH_GPT_find_solutions_of_x4_minus_16_l926_92685


namespace NUMINAMATH_GPT_probability_of_one_fork_one_spoon_one_knife_l926_92669

theorem probability_of_one_fork_one_spoon_one_knife 
  (num_forks : ℕ) (num_spoons : ℕ) (num_knives : ℕ) (total_pieces : ℕ)
  (h_forks : num_forks = 7) (h_spoons : num_spoons = 8) (h_knives : num_knives = 5)
  (h_total : total_pieces = num_forks + num_spoons + num_knives) :
  (∃ (prob : ℚ), prob = 14 / 57) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_one_fork_one_spoon_one_knife_l926_92669


namespace NUMINAMATH_GPT_solve_for_x_l926_92692

theorem solve_for_x : 
  ∃ x : ℝ, 7 * (4 * x + 3) - 5 = -3 * (2 - 8 * x) + 1 / 2 ∧ x = -5.375 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l926_92692


namespace NUMINAMATH_GPT_units_digit_of_m_squared_plus_two_to_the_m_is_seven_l926_92655

def m := 2016^2 + 2^2016

theorem units_digit_of_m_squared_plus_two_to_the_m_is_seven :
  (m^2 + 2^m) % 10 = 7 := by
sorry

end NUMINAMATH_GPT_units_digit_of_m_squared_plus_two_to_the_m_is_seven_l926_92655


namespace NUMINAMATH_GPT_inequality_proof_l926_92630

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) : ab < 1 ∧ 1 < (a^2 + b^2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l926_92630


namespace NUMINAMATH_GPT_initial_antifreeze_percentage_l926_92678

-- Definitions of conditions
def total_volume : ℚ := 10
def replaced_volume : ℚ := 2.85714285714
def final_percentage : ℚ := 50 / 100

-- Statement to prove
theorem initial_antifreeze_percentage (P : ℚ) :
  10 * P / 100 - P / 100 * 2.85714285714 + 2.85714285714 = 5 → 
  P = 30 :=
sorry

end NUMINAMATH_GPT_initial_antifreeze_percentage_l926_92678


namespace NUMINAMATH_GPT_find_constants_l926_92697

theorem find_constants (P Q R : ℤ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (2 * x^2 - 5 * x + 6) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) →
  P = -6 ∧ Q = 8 ∧ R = -5 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l926_92697


namespace NUMINAMATH_GPT_line_tangent_ellipse_l926_92684

-- Define the conditions of the problem
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x + 2
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9

-- Prove the statement about the intersection of the line and ellipse
theorem line_tangent_ellipse (m : ℝ) :
  (∀ x y, line m x y → ellipse x y → x = 0.0 ∧ y = 2.0)
  ↔ m^2 = 1 / 3 :=
sorry

end NUMINAMATH_GPT_line_tangent_ellipse_l926_92684


namespace NUMINAMATH_GPT_find_m_l926_92610

-- Circle equation: x^2 + y^2 + 2x - 6y + 1 = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 6 * y + 1 = 0

-- Line equation: x + m * y + 4 = 0
def line_eq (x y m : ℝ) : Prop := x + m * y + 4 = 0

-- Prove that the value of m such that the center of the circle lies on the line is -1
theorem find_m (m : ℝ) : 
  (∃ x y : ℝ, circle_eq x y ∧ (x, y) = (-1, 3) ∧ line_eq x y m) → m = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_l926_92610


namespace NUMINAMATH_GPT_evaluate_expression_l926_92626

theorem evaluate_expression : 
  (1 / (2 - (1 / (2 - (1 / (2 - (1 / 3))))))) = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l926_92626


namespace NUMINAMATH_GPT_prize_amount_l926_92649

theorem prize_amount (P : ℝ) (n : ℝ) (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : n = 40)
  (h2 : a = 40)
  (h3 : b = (2 / 5) * P)
  (h4 : c = (3 / 5) * 40)
  (h5 : b / c = 120) :
  P = 7200 := 
sorry

end NUMINAMATH_GPT_prize_amount_l926_92649


namespace NUMINAMATH_GPT_max_marks_l926_92690

theorem max_marks (M p : ℝ) (h1 : p = 0.60 * M) (h2 : p = 160 + 20) : M = 300 := by
  sorry

end NUMINAMATH_GPT_max_marks_l926_92690


namespace NUMINAMATH_GPT_range_of_a_l926_92645

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0) ↔ a ≤ -2 ∨ a = 1 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l926_92645


namespace NUMINAMATH_GPT_waiter_earned_in_tips_l926_92619

def waiter_customers := 7
def customers_didnt_tip := 5
def tip_per_customer := 3
def customers_tipped := waiter_customers - customers_didnt_tip
def total_earnings := customers_tipped * tip_per_customer

theorem waiter_earned_in_tips : total_earnings = 6 :=
by
  sorry

end NUMINAMATH_GPT_waiter_earned_in_tips_l926_92619


namespace NUMINAMATH_GPT_find_a_l926_92672

theorem find_a (a n : ℝ) (p : ℝ) (hp : p = 2 / 3)
  (h₁ : a = 3 * n + 5)
  (h₂ : a + 2 = 3 * (n + p) + 5) : a = 3 * n + 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_l926_92672


namespace NUMINAMATH_GPT_find_tangent_points_l926_92693

def f (x : ℝ) : ℝ := x^3 + x - 2
def tangent_parallel_to_line (x : ℝ) : Prop := deriv f x = 4

theorem find_tangent_points :
  (tangent_parallel_to_line 1 ∧ f 1 = 0) ∧ 
  (tangent_parallel_to_line (-1) ∧ f (-1) = -4) :=
by
  sorry

end NUMINAMATH_GPT_find_tangent_points_l926_92693


namespace NUMINAMATH_GPT_line_intersections_with_parabola_l926_92603

theorem line_intersections_with_parabola :
  ∃! (L : ℝ → ℝ) (l_count : ℕ),  
    l_count = 3 ∧
    (∀ x : ℝ, (L x) ∈ {x | (L 0 = 2) ∧ ∃ y, y * y = 8 * x ∧ L x = y}) := sorry

end NUMINAMATH_GPT_line_intersections_with_parabola_l926_92603


namespace NUMINAMATH_GPT_anna_correct_percentage_l926_92652

theorem anna_correct_percentage :
  let test1_problems := 30
  let test1_score := 0.75
  let test2_problems := 50
  let test2_score := 0.85
  let test3_problems := 20
  let test3_score := 0.65
  let correct_test1 := test1_score * test1_problems
  let correct_test2 := test2_score * test2_problems
  let correct_test3 := test3_score * test3_problems
  let total_problems := test1_problems + test2_problems + test3_problems
  let total_correct := correct_test1 + correct_test2 + correct_test3
  (total_correct / total_problems) * 100 = 78 :=
by
  sorry

end NUMINAMATH_GPT_anna_correct_percentage_l926_92652


namespace NUMINAMATH_GPT_max_value_expression_l926_92683

theorem max_value_expression : 
  ∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 →
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 256 / 243 :=
by
  intros x y z hx hy hz hsum
  sorry

end NUMINAMATH_GPT_max_value_expression_l926_92683


namespace NUMINAMATH_GPT_evaluate_ratio_l926_92608

theorem evaluate_ratio : (2^3002 * 3^3005 / 6^3003 : ℚ) = 9 / 2 := 
sorry

end NUMINAMATH_GPT_evaluate_ratio_l926_92608


namespace NUMINAMATH_GPT_work_done_l926_92653

noncomputable def F (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 3

theorem work_done (W : ℝ) (h : W = ∫ x in (1:ℝ)..(5:ℝ), F x) : W = 112 :=
by sorry

end NUMINAMATH_GPT_work_done_l926_92653


namespace NUMINAMATH_GPT_adam_initial_books_l926_92698

theorem adam_initial_books (B : ℕ) (h1 : B - 11 + 23 = 45) : B = 33 := 
by
  sorry

end NUMINAMATH_GPT_adam_initial_books_l926_92698


namespace NUMINAMATH_GPT_range_of_function_is_correct_l926_92635

def range_of_quadratic_function : Set ℝ :=
  {y | ∃ x : ℝ, y = -x^2 - 6 * x - 5}

theorem range_of_function_is_correct :
  range_of_quadratic_function = {y | y ≤ 4} :=
by
  -- sorry allows skipping the actual proof step
  sorry

end NUMINAMATH_GPT_range_of_function_is_correct_l926_92635


namespace NUMINAMATH_GPT_find_m_l926_92681

noncomputable def is_power_function (y : ℝ → ℝ) := 
  ∃ (c : ℝ), ∃ (n : ℝ), ∀ x : ℝ, y x = c * x ^ n

theorem find_m (m : ℝ) :
  (∀ x : ℝ, (∃ c : ℝ, (m^2 - 2 * m + 1) * x^(m - 1) = c * x^n) ∧ (∀ x : ℝ, true)) → m = 2 :=
sorry

end NUMINAMATH_GPT_find_m_l926_92681


namespace NUMINAMATH_GPT_solution_set_inequality_l926_92695

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x

axiom mono_increasing (x y : ℝ) (hxy : 0 < x ∧ x < y) : f x < f y

axiom f_2_eq_0 : f 2 = 0

theorem solution_set_inequality :
  { x : ℝ | (x - 1) * f x < 0 } = { x : ℝ | -2 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x ∧ x < 2 } :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_inequality_l926_92695


namespace NUMINAMATH_GPT_original_distance_between_Stacy_and_Heather_l926_92677

theorem original_distance_between_Stacy_and_Heather
  (H_speed : ℝ := 5)  -- Heather's speed in miles per hour
  (S_speed : ℝ := 6)  -- Stacy's speed in miles per hour
  (delay : ℝ := 0.4)  -- Heather's start delay in hours
  (H_distance : ℝ := 1.1818181818181817)  -- Distance Heather walked when they meet
  : H_speed * (H_distance / H_speed) + S_speed * ((H_distance / H_speed) + delay) = 5 := by
  sorry

end NUMINAMATH_GPT_original_distance_between_Stacy_and_Heather_l926_92677


namespace NUMINAMATH_GPT_no_real_solution_condition_l926_92687

def no_real_solution (k : ℝ) : Prop :=
  let discriminant := 25 + 4 * k
  discriminant < 0

theorem no_real_solution_condition (k : ℝ) : no_real_solution k ↔ k < -25 / 4 := 
sorry

end NUMINAMATH_GPT_no_real_solution_condition_l926_92687


namespace NUMINAMATH_GPT_least_number_to_divisible_by_11_l926_92621

theorem least_number_to_divisible_by_11 (n : ℕ) (h : n = 11002) : ∃ k : ℕ, (n + k) % 11 = 0 ∧ ∀ m : ℕ, (n + m) % 11 = 0 → m ≥ k :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_divisible_by_11_l926_92621


namespace NUMINAMATH_GPT_infinitely_many_n_l926_92663

-- Definition capturing the condition: equation \( (x + y + z)^3 = n^2 xyz \)
def equation (x y z n : ℕ) : Prop := (x + y + z)^3 = n^2 * x * y * z

-- The main statement: proving the existence of infinitely many positive integers n such that the equation has a solution
theorem infinitely_many_n :
  ∃ᶠ n : ℕ in at_top, ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z n :=
sorry

end NUMINAMATH_GPT_infinitely_many_n_l926_92663


namespace NUMINAMATH_GPT_simplify_expression_l926_92689

theorem simplify_expression (a : ℝ) (h : a ≠ 1/2) : 1 - (2 / (1 + (2 * a) / (1 - 2 * a))) = 4 * a - 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l926_92689


namespace NUMINAMATH_GPT_initial_men_count_l926_92636

theorem initial_men_count (M : ℕ) (A : ℕ) (H1 : 58 - (20 + 22) = 2 * M) : M = 8 :=
by
  sorry

end NUMINAMATH_GPT_initial_men_count_l926_92636


namespace NUMINAMATH_GPT_units_digit_periodic_10_l926_92622

theorem units_digit_periodic_10:
  ∀ n: ℕ, (n * (n + 1) * (n + 2)) % 10 = ((n + 10) * (n + 11) * (n + 12)) % 10 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_periodic_10_l926_92622


namespace NUMINAMATH_GPT_fraction_of_boxes_loaded_by_day_crew_l926_92639

theorem fraction_of_boxes_loaded_by_day_crew
    (dayCrewBoxesPerWorker : ℚ)
    (dayCrewWorkers : ℚ)
    (nightCrewBoxesPerWorker : ℚ := (3 / 4) * dayCrewBoxesPerWorker)
    (nightCrewWorkers : ℚ := (3 / 4) * dayCrewWorkers) :
    (dayCrewBoxesPerWorker * dayCrewWorkers) / ((dayCrewBoxesPerWorker * dayCrewWorkers) + (nightCrewBoxesPerWorker * nightCrewWorkers)) = 16 / 25 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_boxes_loaded_by_day_crew_l926_92639


namespace NUMINAMATH_GPT_solve_for_x_l926_92627

theorem solve_for_x (x : ℝ) : (|2 * x + 8| = 4 - 3 * x) → x = -4 / 5 :=
  sorry

end NUMINAMATH_GPT_solve_for_x_l926_92627


namespace NUMINAMATH_GPT_remainder_of_n_div_4_is_1_l926_92615

noncomputable def n : ℕ := sorry  -- We declare n as a noncomputable natural number to proceed with the proof complexity

theorem remainder_of_n_div_4_is_1 (n : ℕ) (h : (2 * n) % 4 = 2) : n % 4 = 1 :=
by
  sorry  -- skip the proof

end NUMINAMATH_GPT_remainder_of_n_div_4_is_1_l926_92615


namespace NUMINAMATH_GPT_inequality_example_l926_92628

theorem inequality_example (a b : ℝ) (h : a - b > 0) : a + 1 > b + 1 :=
sorry

end NUMINAMATH_GPT_inequality_example_l926_92628


namespace NUMINAMATH_GPT_temperature_decrease_is_negative_l926_92600

-- Condition: A temperature rise of 3°C is denoted as +3°C.
def temperature_rise (c : Int) : String := if c > 0 then "+" ++ toString c ++ "°C" else toString c ++ "°C"

-- Specification: Prove a decrease of 4°C is denoted as -4°C.
theorem temperature_decrease_is_negative (h : temperature_rise 3 = "+3°C") : temperature_rise (-4) = "-4°C" :=
by
  -- Proof
  sorry

end NUMINAMATH_GPT_temperature_decrease_is_negative_l926_92600


namespace NUMINAMATH_GPT_find_c_value_l926_92616

theorem find_c_value 
  (a b c : ℝ)
  (h_a : a = 5 / 2)
  (h_b : b = 17)
  (roots : ∀ x : ℝ, x = (-b + Real.sqrt 23) / 5 ∨ x = (-b - Real.sqrt 23) / 5)
  (discrim_eq : ∀ c : ℝ, b ^ 2 - 4 * a * c = 23) :
  c = 26.6 := by
  sorry

end NUMINAMATH_GPT_find_c_value_l926_92616


namespace NUMINAMATH_GPT_greatest_product_three_integers_sum_2000_l926_92680

noncomputable def maxProduct (s : ℝ) : ℝ := 
  s * s * (2000 - 2 * s)

theorem greatest_product_three_integers_sum_2000 : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2000 / 2 ∧ maxProduct x = 8000000000 / 27 := sorry

end NUMINAMATH_GPT_greatest_product_three_integers_sum_2000_l926_92680


namespace NUMINAMATH_GPT_infinite_series_equivalence_l926_92682

theorem infinite_series_equivalence (x y : ℝ) (hy : y ≠ 0 ∧ y ≠ 1) 
  (series_cond : ∑' n : ℕ, x / (y^(n+1)) = 3) :
  ∑' n : ℕ, x / ((x + 2*y)^(n+1)) = 3 * (y - 1) / (5*y - 4) := 
by
  sorry

end NUMINAMATH_GPT_infinite_series_equivalence_l926_92682


namespace NUMINAMATH_GPT_solve_for_x_l926_92686

theorem solve_for_x (x : ℤ) (h : 3 * x + 7 = -2) : x = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l926_92686


namespace NUMINAMATH_GPT_volume_rect_prism_l926_92644

variables (a d h : ℝ)
variables (ha : a > 0) (hd : d > 0) (hh : h > 0)

theorem volume_rect_prism : a * d * h = adh :=
by
  sorry

end NUMINAMATH_GPT_volume_rect_prism_l926_92644


namespace NUMINAMATH_GPT_find_m_value_l926_92654

theorem find_m_value (x: ℝ) (m: ℝ) (hx: x > 2) (hm: m > 0) (h_min: ∀ y, (y = x + m / (x - 2)) → y ≥ 6) : m = 4 := 
sorry

end NUMINAMATH_GPT_find_m_value_l926_92654


namespace NUMINAMATH_GPT_inequality_proof_l926_92620

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : 0 < c)
  : a + b ≤ 2 * c ∧ 2 * c ≤ 3 * c :=
sorry

end NUMINAMATH_GPT_inequality_proof_l926_92620


namespace NUMINAMATH_GPT_b_cong_zero_l926_92637

theorem b_cong_zero (a b c m : ℤ) (h₀ : 1 < m) (h : ∀ (n : ℕ), (a ^ n + b * n + c) % m = 0) : b % m = 0 :=
  sorry

end NUMINAMATH_GPT_b_cong_zero_l926_92637


namespace NUMINAMATH_GPT_savings_after_purchase_l926_92617

theorem savings_after_purchase :
  let price_sweater := 30
  let price_scarf := 20
  let num_sweaters := 6
  let num_scarves := 6
  let savings := 500
  let total_cost := (num_sweaters * price_sweater) + (num_scarves * price_scarf)
  savings - total_cost = 200 :=
by
  sorry

end NUMINAMATH_GPT_savings_after_purchase_l926_92617


namespace NUMINAMATH_GPT_third_month_sale_l926_92661

theorem third_month_sale
  (avg_sale : ℕ)
  (num_months : ℕ)
  (sales : List ℕ)
  (sixth_month_sale : ℕ)
  (total_sales_req : ℕ) :
  avg_sale = 6500 →
  num_months = 6 →
  sales = [6435, 6927, 7230, 6562] →
  sixth_month_sale = 4991 →
  total_sales_req = avg_sale * num_months →
  total_sales_req - (sales.sum + sixth_month_sale) = 6855 := by
  sorry

end NUMINAMATH_GPT_third_month_sale_l926_92661


namespace NUMINAMATH_GPT_right_triangle_345_l926_92641

theorem right_triangle_345 :
  ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
by {
  -- Here, we should construct the proof later
  sorry
}

end NUMINAMATH_GPT_right_triangle_345_l926_92641


namespace NUMINAMATH_GPT_diagonal_cannot_be_good_l926_92670

def is_good (table : ℕ → ℕ → ℕ) (i j : ℕ) :=
  ∀ x y, (x = i ∨ y = j) → ∀ x' y', (x' = i ∨ y' = j) → (x ≠ x' ∨ y ≠ y') → table x y ≠ table x' y'

theorem diagonal_cannot_be_good :
  ∀ (table : ℕ → ℕ → ℕ), (∀ i j, 1 ≤ table i j ∧ table i j ≤ 25) →
  ¬ ∀ k, (is_good table k k) :=
by
  sorry

end NUMINAMATH_GPT_diagonal_cannot_be_good_l926_92670


namespace NUMINAMATH_GPT_fifth_grade_soccer_students_l926_92648

variable (T B Gnp GP S : ℕ)
variable (p : ℝ)

theorem fifth_grade_soccer_students
  (hT : T = 420)
  (hB : B = 296)
  (hp_percent : p = 86 / 100)
  (hGnp : Gnp = 89)
  (hpercent_boys_playing_soccer : (1 - p) * S = GP)
  (hpercent_girls_playing_soccer : GP = 35) :
  S = 250 := by
  sorry

end NUMINAMATH_GPT_fifth_grade_soccer_students_l926_92648


namespace NUMINAMATH_GPT_m_gt_n_l926_92688

noncomputable def m : ℕ := 2015 ^ 2016
noncomputable def n : ℕ := 2016 ^ 2015

theorem m_gt_n : m > n := by
  sorry

end NUMINAMATH_GPT_m_gt_n_l926_92688


namespace NUMINAMATH_GPT_common_difference_arithmetic_sequence_l926_92601

theorem common_difference_arithmetic_sequence
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (d : ℕ) 
  (h₁ : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2) -- sum formula for arithmetic sequence
  (h₂ : 2 * S 3 = 3 * S 2 + 6) : 
  d = 2 := 
sorry

end NUMINAMATH_GPT_common_difference_arithmetic_sequence_l926_92601


namespace NUMINAMATH_GPT_problem_value_of_m_l926_92609

theorem problem_value_of_m (m : ℝ)
  (h1 : (m + 1) * x ^ (m ^ 2 - 3) = y)
  (h2 : m ^ 2 - 3 = 1)
  (h3 : m + 1 < 0) : 
  m = -2 := 
  sorry

end NUMINAMATH_GPT_problem_value_of_m_l926_92609


namespace NUMINAMATH_GPT_certain_number_eq_1000_l926_92614

theorem certain_number_eq_1000 (x : ℝ) (h : 3500 - x / 20.50 = 3451.2195121951218) : x = 1000 := 
by
  sorry

end NUMINAMATH_GPT_certain_number_eq_1000_l926_92614


namespace NUMINAMATH_GPT_a_received_share_l926_92679

def a_inv : ℕ := 7000
def b_inv : ℕ := 11000
def c_inv : ℕ := 18000

def b_share : ℕ := 2200

def total_profit : ℕ := (b_share / (b_inv / 1000)) * 36
def a_ratio : ℕ := a_inv / 1000
def total_ratio : ℕ := (a_inv / 1000) + (b_inv / 1000) + (c_inv / 1000)

def a_share : ℕ := (a_ratio / total_ratio) * total_profit

theorem a_received_share :
  a_share = 1400 := 
sorry

end NUMINAMATH_GPT_a_received_share_l926_92679


namespace NUMINAMATH_GPT_total_bills_combined_l926_92675

theorem total_bills_combined
  (a b c : ℝ)
  (H1 : 0.15 * a = 3)
  (H2 : 0.25 * b = 5)
  (H3 : 0.20 * c = 4) :
  a + b + c = 60 := 
sorry

end NUMINAMATH_GPT_total_bills_combined_l926_92675


namespace NUMINAMATH_GPT_total_packs_of_groceries_l926_92642

-- Definitions based on conditions
def packs_of_cookies : Nat := 4
def packs_of_cake : Nat := 22
def packs_of_chocolate : Nat := 16

-- The proof statement
theorem total_packs_of_groceries : packs_of_cookies + packs_of_cake + packs_of_chocolate = 42 :=
by
  -- Proof skipped using sorry
  sorry

end NUMINAMATH_GPT_total_packs_of_groceries_l926_92642


namespace NUMINAMATH_GPT_find_x_l926_92647

noncomputable def a : ℝ := Real.log 2 / Real.log 10
noncomputable def b : ℝ := 1 / a
noncomputable def log2_5 : ℝ := Real.log 5 / Real.log 2

theorem find_x (a₀ : a = 0.3010) : 
  ∃ x : ℝ, (log2_5 ^ 2 - a * log2_5 + x * b = 0) → 
  x = (log2_5 ^ 2 * 0.3010) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l926_92647


namespace NUMINAMATH_GPT_is_factorization_l926_92662

-- Define the conditions
def A_transformation : Prop := (∀ x : ℝ, (x + 1) * (x - 1) = x ^ 2 - 1)
def B_transformation : Prop := (∀ m : ℝ, m ^ 2 + m - 4 = (m + 3) * (m - 2) + 2)
def C_transformation : Prop := (∀ x : ℝ, x ^ 2 + 2 * x = x * (x + 2))
def D_transformation : Prop := (∀ x : ℝ, 2 * x ^ 2 + 2 * x = 2 * x ^ 2 * (1 + (1 / x)))

-- The goal is to prove that transformation C is a factorization
theorem is_factorization : C_transformation :=
by
  sorry

end NUMINAMATH_GPT_is_factorization_l926_92662


namespace NUMINAMATH_GPT_representation_of_1_l926_92634

theorem representation_of_1 (x y z : ℕ) (h : 1 = 1/x + 1/y + 1/z) : 
  (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 3 ∧ y = 3 ∧ z = 3) :=
by
  sorry

end NUMINAMATH_GPT_representation_of_1_l926_92634
