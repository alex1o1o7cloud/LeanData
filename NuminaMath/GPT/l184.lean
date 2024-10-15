import Mathlib

namespace NUMINAMATH_GPT_deposit_is_500_l184_18462

-- Definitions corresponding to the conditions
def janet_saved : ℕ := 2225
def rent_per_month : ℕ := 1250
def advance_months : ℕ := 2
def extra_needed : ℕ := 775

-- Definition that encapsulates the deposit calculation
def deposit_required (saved rent_monthly months_advance extra : ℕ) : ℕ :=
  let total_rent := months_advance * rent_monthly
  let total_needed := saved + extra
  total_needed - total_rent

-- Theorem statement for the proof problem
theorem deposit_is_500 : deposit_required janet_saved rent_per_month advance_months extra_needed = 500 :=
by
  sorry

end NUMINAMATH_GPT_deposit_is_500_l184_18462


namespace NUMINAMATH_GPT_returned_books_percentage_is_correct_l184_18479

-- This function takes initial_books, end_books, and loaned_books and computes the percentage of books returned.
noncomputable def percent_books_returned (initial_books : ℕ) (end_books : ℕ) (loaned_books : ℕ) : ℚ :=
  let books_out_on_loan := initial_books - end_books
  let books_returned := loaned_books - books_out_on_loan
  (books_returned : ℚ) / (loaned_books : ℚ) * 100

-- The main theorem that states the percentage of books returned is 70%
theorem returned_books_percentage_is_correct :
  percent_books_returned 75 57 60 = 70 := by
  sorry

end NUMINAMATH_GPT_returned_books_percentage_is_correct_l184_18479


namespace NUMINAMATH_GPT_solve_equation_l184_18408

theorem solve_equation (x : ℝ) (h : x ≠ 1) :
  (3 * x) / (x - 1) = 2 + 1 / (x - 1) → x = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l184_18408


namespace NUMINAMATH_GPT_least_grapes_in_heap_l184_18457

theorem least_grapes_in_heap :
  ∃ n : ℕ, (n % 19 = 1) ∧ (n % 23 = 1) ∧ (n % 29 = 1) ∧ n = 12209 :=
by
  sorry

end NUMINAMATH_GPT_least_grapes_in_heap_l184_18457


namespace NUMINAMATH_GPT_intersection_M_N_l184_18486

-- Define set M
def M : Set ℝ := {x : ℝ | ∃ t : ℝ, x = 2^(-t) }

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sin x }

-- Theorem stating the intersection of M and N
theorem intersection_M_N :
  (M ∩ N) = {y : ℝ | 0 < y ∧ y ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l184_18486


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l184_18419

theorem parabola_focus_coordinates (x y p : ℝ) (h : y^2 = 8 * x) : 
  p = 2 → (p, 0) = (2, 0) := 
by 
  sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l184_18419


namespace NUMINAMATH_GPT_suff_not_necess_cond_perpendicular_l184_18441

theorem suff_not_necess_cond_perpendicular (m : ℝ) :
  (m = 1 → ∀ x y : ℝ, x - y = 0 ∧ x + y = 0) ∧
  (m ≠ 1 → ∃ (x y : ℝ), ¬ (x - y = 0 ∧ x + y = 0)) :=
sorry

end NUMINAMATH_GPT_suff_not_necess_cond_perpendicular_l184_18441


namespace NUMINAMATH_GPT_average_salary_increase_l184_18430

theorem average_salary_increase 
  (average_salary : ℕ) (manager_salary : ℕ)
  (n : ℕ) (initial_count : ℕ) (new_count : ℕ) (initial_average : ℕ)
  (total_salary : ℕ) (new_total_salary : ℕ) (new_average : ℕ)
  (salary_increase : ℕ) :
  initial_average = 1500 →
  manager_salary = 3600 →
  initial_count = 20 →
  new_count = initial_count + 1 →
  total_salary = initial_count * initial_average →
  new_total_salary = total_salary + manager_salary →
  new_average = new_total_salary / new_count →
  salary_increase = new_average - initial_average →
  salary_increase = 100 := by
  sorry

end NUMINAMATH_GPT_average_salary_increase_l184_18430


namespace NUMINAMATH_GPT_find_f_of_2_l184_18427

theorem find_f_of_2 : ∃ (f : ℤ → ℤ), (∀ x : ℤ, f (x+1) = x^2 - 1) ∧ f 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_2_l184_18427


namespace NUMINAMATH_GPT_max_XYZ_plus_terms_l184_18434

theorem max_XYZ_plus_terms {X Y Z : ℕ} (h : X + Y + Z = 15) :
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 200 :=
sorry

end NUMINAMATH_GPT_max_XYZ_plus_terms_l184_18434


namespace NUMINAMATH_GPT_find_y_l184_18497

theorem find_y (y : ℝ) (h : (15 + 25 + y) / 3 = 23) : y = 29 :=
sorry

end NUMINAMATH_GPT_find_y_l184_18497


namespace NUMINAMATH_GPT_condo_floors_l184_18431

theorem condo_floors (F P : ℕ) (h1: 12 * F + 2 * P = 256) (h2 : P = 2) : F + P = 23 :=
by
  sorry

end NUMINAMATH_GPT_condo_floors_l184_18431


namespace NUMINAMATH_GPT_well_depth_is_2000_l184_18448

-- Given conditions
def total_time : ℝ := 10
def stone_law (t₁ : ℝ) : ℝ := 20 * t₁^2
def sound_velocity : ℝ := 1120

-- Statement to be proven
theorem well_depth_is_2000 :
  ∃ (d t₁ t₂ : ℝ), 
    d = stone_law t₁ ∧ t₂ = d / sound_velocity ∧ t₁ + t₂ = total_time :=
sorry

end NUMINAMATH_GPT_well_depth_is_2000_l184_18448


namespace NUMINAMATH_GPT_ratio_pentagon_rectangle_l184_18478

-- Definitions of conditions.
def pentagon_side_length (p : ℕ) : Prop := 5 * p = 30
def rectangle_width (w : ℕ) : Prop := 6 * w = 30

-- The theorem to prove.
theorem ratio_pentagon_rectangle (p w : ℕ) (h1 : pentagon_side_length p) (h2 : rectangle_width w) :
  p / w = 6 / 5 :=
by sorry

end NUMINAMATH_GPT_ratio_pentagon_rectangle_l184_18478


namespace NUMINAMATH_GPT_jason_borrowed_amount_l184_18492

theorem jason_borrowed_amount :
  let cycle := [1, 3, 5, 7, 9, 11]
  let total_chores := 48
  let chores_per_cycle := cycle.length
  let earnings_one_cycle := cycle.sum
  let complete_cycles := total_chores / chores_per_cycle
  let total_earnings := complete_cycles * earnings_one_cycle
  total_earnings = 288 :=
by
  sorry

end NUMINAMATH_GPT_jason_borrowed_amount_l184_18492


namespace NUMINAMATH_GPT_sphere_surface_area_l184_18490

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : ∀ r, V = (4/3) * π * r^3)
  (h2 : V = 72 * π) : A = 36 * π * 2^(2/3) :=
by 
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l184_18490


namespace NUMINAMATH_GPT_equal_circles_common_point_l184_18449

theorem equal_circles_common_point (n : ℕ) (r : ℝ) 
  (centers : Fin n → ℝ × ℝ)
  (h : ∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k →
    ∃ (p : ℝ × ℝ),
      dist p (centers i) = r ∧
      dist p (centers j) = r ∧
      dist p (centers k) = r) :
  ∃ O : ℝ × ℝ, ∀ i : Fin n, dist O (centers i) = r := sorry

end NUMINAMATH_GPT_equal_circles_common_point_l184_18449


namespace NUMINAMATH_GPT_find_a_b_l184_18415

theorem find_a_b (a b : ℝ) (h : (⟨1, 2⟩ * (a : ℂ) + b = ⟨0, 2⟩)) : a = 1 ∧ b = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_a_b_l184_18415


namespace NUMINAMATH_GPT_max_sides_three_obtuse_l184_18488

theorem max_sides_three_obtuse (n : ℕ) (convex : Prop) (obtuse_angles : ℕ) :
  (convex = true ∧ obtuse_angles = 3) → n ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_max_sides_three_obtuse_l184_18488


namespace NUMINAMATH_GPT_step_of_induction_l184_18400

theorem step_of_induction (k : ℕ) (h : ∃ m : ℕ, 5^k - 2^k = 3 * m) :
  5^(k+1) - 2^(k+1) = 5 * (5^k - 2^k) + 3 * 2^k := 
by
  sorry

end NUMINAMATH_GPT_step_of_induction_l184_18400


namespace NUMINAMATH_GPT_find_present_ratio_l184_18466

noncomputable def present_ratio_of_teachers_to_students : Prop :=
  ∃ (S T S' T' : ℕ),
    (T = 3) ∧
    (S = 50 * T) ∧
    (S' = S + 50) ∧
    (T' = T + 5) ∧
    (S' / T' = 25 / 1) ∧ 
    (T / S = 1 / 50)

theorem find_present_ratio : present_ratio_of_teachers_to_students :=
by
  sorry

end NUMINAMATH_GPT_find_present_ratio_l184_18466


namespace NUMINAMATH_GPT_contradiction_problem_l184_18471

theorem contradiction_problem (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) → False := 
by
  sorry

end NUMINAMATH_GPT_contradiction_problem_l184_18471


namespace NUMINAMATH_GPT_tangent_of_7pi_over_4_l184_18423

   theorem tangent_of_7pi_over_4 : Real.tan (7 * Real.pi / 4) = -1 := 
   sorry
   
end NUMINAMATH_GPT_tangent_of_7pi_over_4_l184_18423


namespace NUMINAMATH_GPT_rectangle_length_l184_18401

theorem rectangle_length (L W : ℝ) 
  (h1 : L + W = 23) 
  (h2 : L^2 + W^2 = 289) : 
  L = 15 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_length_l184_18401


namespace NUMINAMATH_GPT_relationship_between_n_and_m_l184_18420

variable {n m : ℕ}
variable {x y : ℝ}
variable {a : ℝ}
variable {z : ℝ}

def mean_sample_combined (n m : ℕ) (x y z a : ℝ) : Prop :=
  z = a * x + (1 - a) * y ∧ a > 1 / 2

theorem relationship_between_n_and_m 
  (hx : ∀ (i : ℕ), i < n → x = x)
  (hy : ∀ (j : ℕ), j < m → y = y)
  (hz : mean_sample_combined n m x y z a)
  (hne : x ≠ y) : n < m :=
sorry

end NUMINAMATH_GPT_relationship_between_n_and_m_l184_18420


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l184_18443

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₁ : a 1 = -2012)
  (h₂ : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1)))
  (h₃ : (S 12) / 12 - (S 10) / 10 = 2) :
  S 2012 = -2012 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l184_18443


namespace NUMINAMATH_GPT_probability_at_least_two_worth_visiting_l184_18437

theorem probability_at_least_two_worth_visiting :
  let total_caves := 8
  let worth_visiting := 3
  let select_caves := 4
  let worth_select_2 := Nat.choose worth_visiting 2 * Nat.choose (total_caves - worth_visiting) 2
  let worth_select_3 := Nat.choose worth_visiting 3 * Nat.choose (total_caves - worth_visiting) 1
  let total_select := Nat.choose total_caves select_caves
  let probability := (worth_select_2 + worth_select_3) / total_select
  probability = 1 / 2 := sorry

end NUMINAMATH_GPT_probability_at_least_two_worth_visiting_l184_18437


namespace NUMINAMATH_GPT_find_x_squared_plus_one_over_x_squared_l184_18424

theorem find_x_squared_plus_one_over_x_squared (x : ℝ) (h : x + 1/x = 4) : x^2 + 1/x^2 = 14 := by
  sorry

end NUMINAMATH_GPT_find_x_squared_plus_one_over_x_squared_l184_18424


namespace NUMINAMATH_GPT_solve_equation_l184_18456

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (x / (x - 2) - 3 / x = 1) → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l184_18456


namespace NUMINAMATH_GPT_geometric_sequence_and_general_formula_l184_18444

theorem geometric_sequence_and_general_formula (a : ℕ → ℝ) (h : ∀ n, a (n+1) = (2/3) * a n + 2) (ha1 : a 1 = 7) : 
  ∃ r : ℝ, ∀ n, a n = r ^ (n-1) + 6 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_and_general_formula_l184_18444


namespace NUMINAMATH_GPT_ali_ate_half_to_percent_l184_18404

theorem ali_ate_half_to_percent : (1 / 2 : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_ali_ate_half_to_percent_l184_18404


namespace NUMINAMATH_GPT_a_sufficient_not_necessary_for_a_squared_eq_b_squared_l184_18459

theorem a_sufficient_not_necessary_for_a_squared_eq_b_squared
  (a b : ℝ) :
  (a = b) → (a^2 = b^2) ∧ ¬ ((a^2 = b^2) → (a = b)) :=
  sorry

end NUMINAMATH_GPT_a_sufficient_not_necessary_for_a_squared_eq_b_squared_l184_18459


namespace NUMINAMATH_GPT_sin_inequality_l184_18426

open Real

theorem sin_inequality (a b : ℝ) (n : ℕ) (ha : 0 < a) (haq : a < π/4) (hb : 0 < b) (hbq : b < π/4) (hn : 0 < n) :
  (sin a)^n + (sin b)^n / (sin a + sin b)^n ≥ (sin (2 * a))^n + (sin (2 * b))^n / (sin (2 * a) + sin (2* b))^n :=
sorry

end NUMINAMATH_GPT_sin_inequality_l184_18426


namespace NUMINAMATH_GPT_route_down_distance_l184_18487

noncomputable def rate_up : ℝ := 3
noncomputable def time_up : ℝ := 2
noncomputable def time_down : ℝ := 2
noncomputable def rate_down := 1.5 * rate_up

theorem route_down_distance : rate_down * time_down = 9 := by
  sorry

end NUMINAMATH_GPT_route_down_distance_l184_18487


namespace NUMINAMATH_GPT_mass_of_man_l184_18482

theorem mass_of_man (L B : ℝ) (h : ℝ) (ρ : ℝ) (V : ℝ) : L = 8 ∧ B = 3 ∧ h = 0.01 ∧ ρ = 1 ∧ V = L * 100 * B * 100 * h → V / 1000 = 240 :=
by
  sorry

end NUMINAMATH_GPT_mass_of_man_l184_18482


namespace NUMINAMATH_GPT_marker_cost_l184_18491

theorem marker_cost (s n c : ℕ) (h_majority : s > 20) (h_markers : n > 1) (h_cost : c > n) (h_total_cost : s * n * c = 3388) : c = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_marker_cost_l184_18491


namespace NUMINAMATH_GPT_find_x_given_conditions_l184_18465

variables {x y z : ℝ}

theorem find_x_given_conditions (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = (576 : ℝ)^(1/7) := 
sorry

end NUMINAMATH_GPT_find_x_given_conditions_l184_18465


namespace NUMINAMATH_GPT_prism_aligns_l184_18416

theorem prism_aligns (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ prism_dimensions = (a * 5, b * 10, c * 20) :=
by
  sorry

end NUMINAMATH_GPT_prism_aligns_l184_18416


namespace NUMINAMATH_GPT_avg_visitors_other_days_l184_18409

-- Definitions for average visitors on Sundays and average visitors over the month
def avg_visitors_on_sundays : ℕ := 600
def avg_visitors_over_month : ℕ := 300
def days_in_month : ℕ := 30

-- Given conditions
def num_sundays_in_month : ℕ := 5
def total_days : ℕ := days_in_month
def total_visitors_over_month : ℕ := avg_visitors_over_month * days_in_month

-- Goal: Calculate the average number of visitors on other days (Monday to Saturday)
theorem avg_visitors_other_days :
  (avg_visitors_on_sundays * num_sundays_in_month + (total_days - num_sundays_in_month) * 240) = total_visitors_over_month :=
by
  -- Proof expected here, but skipped according to the instructions
  sorry

end NUMINAMATH_GPT_avg_visitors_other_days_l184_18409


namespace NUMINAMATH_GPT_coin_selection_probability_l184_18460

noncomputable def probability_at_least_50_cents : ℚ := 
  let total_ways := Nat.choose 12 6 -- total ways to choose 6 coins out of 12
  let case1 := 1 -- 6 dimes
  let case2 := (Nat.choose 6 5) * (Nat.choose 4 1) -- 5 dimes and 1 nickel
  let case3 := (Nat.choose 6 4) * (Nat.choose 4 2) -- 4 dimes and 2 nickels
  let successful_ways := case1 + case2 + case3 -- total successful outcomes
  successful_ways / total_ways

theorem coin_selection_probability : 
  probability_at_least_50_cents = 127 / 924 := by 
  sorry

end NUMINAMATH_GPT_coin_selection_probability_l184_18460


namespace NUMINAMATH_GPT_triangle_area_and_angle_l184_18455

theorem triangle_area_and_angle (a b c A B C : ℝ) 
  (habc: A + B + C = Real.pi)
  (h1: (2*a + b)*Real.cos C + c*Real.cos B = 0)
  (h2: c = 2*Real.sqrt 6 / 3)
  (h3: Real.sin A * Real.cos B = (Real.sqrt 3 - 1)/4) :
  (C = 2*Real.pi / 3) ∧ (1/2 * b * c * Real.sin A = (6 - 2 * Real.sqrt 3)/9) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_and_angle_l184_18455


namespace NUMINAMATH_GPT_amount_after_a_year_l184_18452

def initial_amount : ℝ := 90
def interest_rate : ℝ := 0.10

theorem amount_after_a_year : initial_amount * (1 + interest_rate) = 99 := 
by
  -- Here 'sorry' indicates that the proof is not provided.
  sorry

end NUMINAMATH_GPT_amount_after_a_year_l184_18452


namespace NUMINAMATH_GPT_beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l184_18406

def beautiful_association_number (x y a t : ℚ) : Prop :=
  |x - a| + |y - a| = t

theorem beautiful_association_number_part1 :
  beautiful_association_number (-3) 5 2 8 :=
by sorry

theorem beautiful_association_number_part2 (x : ℚ) :
  beautiful_association_number x 2 3 4 ↔ x = 6 ∨ x = 0 :=
by sorry

theorem beautiful_association_number_part3 (x0 x1 x2 x3 x4 : ℚ) :
  beautiful_association_number x0 x1 1 1 ∧ 
  beautiful_association_number x1 x2 2 1 ∧ 
  beautiful_association_number x2 x3 3 1 ∧ 
  beautiful_association_number x3 x4 4 1 →
  x1 + x2 + x3 + x4 = 10 :=
by sorry

end NUMINAMATH_GPT_beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l184_18406


namespace NUMINAMATH_GPT_power_addition_rule_l184_18436

variable {a : ℝ}
variable {m n : ℕ}

theorem power_addition_rule (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + n) = 6 := by
  sorry

end NUMINAMATH_GPT_power_addition_rule_l184_18436


namespace NUMINAMATH_GPT_distance_between_A_and_B_l184_18435

theorem distance_between_A_and_B 
  (v t t1 : ℝ)
  (h1 : 5 * v * t + 4 * v * t = 9 * v * t)
  (h2 : t1 = 10 / (4.8 * v))
  (h3 : 10 / 4.8 = 25 / 12):
  (9 * v * t + 4 * v * t1) = 450 :=
by 
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l184_18435


namespace NUMINAMATH_GPT_initial_gasohol_amount_l184_18473

variable (x : ℝ)

def gasohol_ethanol_percentage (initial_gasohol : ℝ) := 0.05 * initial_gasohol
def mixture_ethanol_percentage (initial_gasohol : ℝ) := gasohol_ethanol_percentage initial_gasohol + 3

def optimal_mixture (total_volume : ℝ) := 0.10 * total_volume

theorem initial_gasohol_amount :
  ∀ (initial_gasohol : ℝ), 
  mixture_ethanol_percentage initial_gasohol = optimal_mixture (initial_gasohol + 3) →
  initial_gasohol = 54 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_gasohol_amount_l184_18473


namespace NUMINAMATH_GPT_smallest_value_x_squared_plus_six_x_plus_nine_l184_18495

theorem smallest_value_x_squared_plus_six_x_plus_nine : ∀ x : ℝ, x^2 + 6 * x + 9 ≥ 0 :=
by sorry

end NUMINAMATH_GPT_smallest_value_x_squared_plus_six_x_plus_nine_l184_18495


namespace NUMINAMATH_GPT_all_metals_conduct_electricity_l184_18438

def Gold_conducts : Prop := sorry
def Silver_conducts : Prop := sorry
def Copper_conducts : Prop := sorry
def Iron_conducts : Prop := sorry
def inductive_reasoning : Prop := sorry

theorem all_metals_conduct_electricity (g: Gold_conducts) (s: Silver_conducts) (c: Copper_conducts) (i: Iron_conducts) : inductive_reasoning := 
sorry

end NUMINAMATH_GPT_all_metals_conduct_electricity_l184_18438


namespace NUMINAMATH_GPT_atomic_number_cannot_be_x_plus_4_l184_18481

-- Definitions for atomic numbers and elements in the same main group
def in_same_main_group (A B : Type) (atomic_num_A atomic_num_B : ℕ) : Prop :=
  atomic_num_B ≠ atomic_num_A + 4

-- Noncomputable definition is likely needed as the problem involves non-algorithmic aspects.
noncomputable def periodic_table_condition (A B : Type) (x : ℕ) : Prop :=
  in_same_main_group A B x (x + 4)

-- Main theorem stating the mathematical proof problem
theorem atomic_number_cannot_be_x_plus_4
  (A B : Type)
  (x : ℕ)
  (h : periodic_table_condition A B x) : false :=
  by
    sorry

end NUMINAMATH_GPT_atomic_number_cannot_be_x_plus_4_l184_18481


namespace NUMINAMATH_GPT_pet_store_cages_l184_18429

theorem pet_store_cages 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (puppies_per_cage : ℕ) 
  (h_initial_puppies : initial_puppies = 45) 
  (h_sold_puppies : sold_puppies = 11) 
  (h_puppies_per_cage : puppies_per_cage = 7) 
  : (initial_puppies - sold_puppies + puppies_per_cage - 1) / puppies_per_cage = 5 :=
by sorry

end NUMINAMATH_GPT_pet_store_cages_l184_18429


namespace NUMINAMATH_GPT_negation_exists_equiv_forall_l184_18412

theorem negation_exists_equiv_forall :
  (¬ (∃ x : ℤ, x^2 + 2*x - 1 < 0)) ↔ (∀ x : ℤ, x^2 + 2*x - 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_exists_equiv_forall_l184_18412


namespace NUMINAMATH_GPT_linear_eq_k_l184_18458

theorem linear_eq_k (k : ℝ) : (k - 3) * x ^ (|k| - 2) + 5 = k - 4 → |k| = 3 → k ≠ 3 → k = -3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_linear_eq_k_l184_18458


namespace NUMINAMATH_GPT_Abby_has_17_quarters_l184_18428

theorem Abby_has_17_quarters (q n : ℕ) (h1 : q + n = 23) (h2 : 25 * q + 5 * n = 455) : q = 17 :=
sorry

end NUMINAMATH_GPT_Abby_has_17_quarters_l184_18428


namespace NUMINAMATH_GPT_probability_A_score_not_less_than_135_l184_18440

/-- A certain school organized a competition with the following conditions:
  - The test has 25 multiple-choice questions, each with 4 options.
  - Each correct answer scores 6 points, each unanswered question scores 2 points, and each wrong answer scores 0 points.
  - Both candidates answered the first 20 questions correctly.
  - Candidate A will attempt only the last 3 questions, and for each, A can eliminate 1 wrong option,
    hence the probability of answering any one question correctly is 1/3.
  - A gives up the last 2 questions.
  - Prove that the probability that A's total score is not less than 135 points is equal to 7/27.
-/
theorem probability_A_score_not_less_than_135 :
  let prob_success := 1 / 3
  let prob_2_successes := (3 * (prob_success^2) * (2/3))
  let prob_3_successes := (prob_success^3)
  prob_2_successes + prob_3_successes = 7 / 27 := 
by
  sorry

end NUMINAMATH_GPT_probability_A_score_not_less_than_135_l184_18440


namespace NUMINAMATH_GPT_odd_function_iff_a2_b2_zero_l184_18418

noncomputable def f (x a b : ℝ) : ℝ := x * |x - a| + b

theorem odd_function_iff_a2_b2_zero {a b : ℝ} :
  (∀ x, f x a b = - f (-x) a b) ↔ a^2 + b^2 = 0 := by
  sorry

end NUMINAMATH_GPT_odd_function_iff_a2_b2_zero_l184_18418


namespace NUMINAMATH_GPT_sam_new_crime_books_l184_18403

theorem sam_new_crime_books (used_adventure_books : ℝ) (used_mystery_books : ℝ) (total_books : ℝ) :
  used_adventure_books = 13.0 →
  used_mystery_books = 17.0 →
  total_books = 45.0 →
  total_books - (used_adventure_books + used_mystery_books) = 15.0 :=
by
  intros ha hm ht
  rw [ha, hm, ht]
  norm_num
  -- sorry

end NUMINAMATH_GPT_sam_new_crime_books_l184_18403


namespace NUMINAMATH_GPT_k_less_than_two_l184_18450

theorem k_less_than_two
    (x : ℝ)
    (k : ℝ)
    (y : ℝ)
    (h : y = (2 - k) / x)
    (h1 : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) : k < 2 :=
by
  sorry

end NUMINAMATH_GPT_k_less_than_two_l184_18450


namespace NUMINAMATH_GPT_domain_of_f_l184_18442

noncomputable def f (x : ℝ) := 1 / Real.log (x + 1) + Real.sqrt (9 - x^2)

theorem domain_of_f :
  {x : ℝ | 0 < x + 1} ∩ {x : ℝ | x ≠ 0} ∩ {x : ℝ | 9 - x^2 ≥ 0} = (Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioc 0 (3 : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l184_18442


namespace NUMINAMATH_GPT_min_frac_sum_l184_18433

open Real

noncomputable def minValue (m n : ℝ) : ℝ := 1 / m + 2 / n

theorem min_frac_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  minValue m n = 3 + 2 * sqrt 2 := by
  sorry

end NUMINAMATH_GPT_min_frac_sum_l184_18433


namespace NUMINAMATH_GPT_decreasing_hyperbola_l184_18475

theorem decreasing_hyperbola (m : ℝ) (x : ℝ) (hx : x > 0) (y : ℝ) (h_eq : y = (1 - m) / x) :
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > x₁ → (1 - m) / x₂ < (1 - m) / x₁) ↔ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_hyperbola_l184_18475


namespace NUMINAMATH_GPT_fair_split_adjustment_l184_18461

theorem fair_split_adjustment
    (A B : ℝ)
    (h : A < B)
    (d1 d2 d3 : ℝ)
    (h1 : d1 = 120)
    (h2 : d2 = 150)
    (h3 : d3 = 180)
    (bernardo_pays_twice : ∀ D, (2 : ℝ) * D = d1 + d2 + d3) :
    (B - A) / 2 - 75 = ((d1 + d2 + d3) - 450) / 2 - (A - (d1 + d2 + d3) / 3) :=
by
  sorry

end NUMINAMATH_GPT_fair_split_adjustment_l184_18461


namespace NUMINAMATH_GPT_find_b_l184_18410

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 35 * b) : b = 63 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_l184_18410


namespace NUMINAMATH_GPT_observation_count_l184_18414

theorem observation_count (n : ℤ) (mean_initial : ℝ) (erroneous_value correct_value : ℝ) (mean_corrected : ℝ) :
  mean_initial = 36 →
  erroneous_value = 20 →
  correct_value = 34 →
  mean_corrected = 36.45 →
  n ≥ 0 →
  ∃ n : ℤ, (n * mean_initial + (correct_value - erroneous_value) = n * mean_corrected) ∧ (n = 31) :=
by
  intros h1 h2 h3 h4 h5
  use 31
  sorry

end NUMINAMATH_GPT_observation_count_l184_18414


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l184_18493

theorem sum_of_reciprocals_of_squares (x y : ℕ) (hxy : x * y = 17) : 
  1 / (x:ℚ)^2 + 1 / (y:ℚ)^2 = 290 / 289 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l184_18493


namespace NUMINAMATH_GPT_initial_hair_length_l184_18470

-- Definitions based on the conditions
def hair_cut_off : ℕ := 13
def current_hair_length : ℕ := 1

-- The problem statement to be proved
theorem initial_hair_length : (current_hair_length + hair_cut_off = 14) :=
by
  sorry

end NUMINAMATH_GPT_initial_hair_length_l184_18470


namespace NUMINAMATH_GPT_find_longer_diagonal_l184_18417

-- Define the necessary conditions
variables (d1 d2 : ℝ)
variable (A : ℝ)
axiom ratio_condition : d1 / d2 = 2 / 3
axiom area_condition : A = 12

-- Define the problem of finding the length of the longer diagonal
theorem find_longer_diagonal : ∃ (d : ℝ), d = d2 → d = 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_longer_diagonal_l184_18417


namespace NUMINAMATH_GPT_Edmund_can_wrap_15_boxes_every_3_days_l184_18451

-- We define the conditions as Lean definitions
def inches_per_gift_box : ℕ := 18
def inches_per_day : ℕ := 90

-- We state the theorem to prove the question (15 gift boxes every 3 days)
theorem Edmund_can_wrap_15_boxes_every_3_days :
  (inches_per_day / inches_per_gift_box) * 3 = 15 :=
by
  sorry

end NUMINAMATH_GPT_Edmund_can_wrap_15_boxes_every_3_days_l184_18451


namespace NUMINAMATH_GPT_quadrilateral_possible_with_2_2_2_l184_18469

theorem quadrilateral_possible_with_2_2_2 :
  ∀ (s1 s2 s3 s4 : ℕ), (s1 = 2) → (s2 = 2) → (s3 = 2) → (s4 = 5) →
  s1 + s2 + s3 > s4 :=
by
  intros s1 s2 s3 s4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_quadrilateral_possible_with_2_2_2_l184_18469


namespace NUMINAMATH_GPT_find_f2_l184_18464

-- A condition of the problem is the specific form of the function
def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

-- Given condition
theorem find_f2 (a b : ℝ) (h : f (-2) a b = 3) : f 2 a b = -19 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l184_18464


namespace NUMINAMATH_GPT_simplify_fraction_lemma_l184_18413

noncomputable def simplify_fraction (a : ℝ) (h : a ≠ 5) : ℝ :=
  (a^2 - 5 * a) / (a - 5)

theorem simplify_fraction_lemma (a : ℝ) (h : a ≠ 5) : simplify_fraction a h = a := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_lemma_l184_18413


namespace NUMINAMATH_GPT_proof_problem_l184_18472

-- Given definitions
def A := { y : ℝ | ∃ x : ℝ, y = x^2 + 1 }
def B := { p : ℝ × ℝ | ∃ x : ℝ, p.snd = x^2 + 1 }

-- Theorem to prove 1 ∉ B and 2 ∈ A
theorem proof_problem : 1 ∉ B ∧ 2 ∈ A :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l184_18472


namespace NUMINAMATH_GPT_rectangular_prism_inequalities_l184_18474

variable {a b c : ℝ}

noncomputable def p (a b c : ℝ) := 4 * (a + b + c)
noncomputable def S (a b c : ℝ) := 2 * (a * b + b * c + c * a)
noncomputable def d (a b c : ℝ) := Real.sqrt (a^2 + b^2 + c^2)

theorem rectangular_prism_inequalities (h : a > b) (h1 : b > c) :
  a > (1 / 3) * (p a b c / 4 + Real.sqrt (d a b c ^ 2 - (1 / 2) * S a b c)) ∧
  c < (1 / 3) * (p a b c / 4 - Real.sqrt (d a b c ^ 2 - (1 / 2) * S a b c)) :=
by
  sorry

end NUMINAMATH_GPT_rectangular_prism_inequalities_l184_18474


namespace NUMINAMATH_GPT_parabola_solution_l184_18498

theorem parabola_solution (a b : ℝ) : 
  (∃ y : ℝ, y = 2^2 + 2 * a + b ∧ y = 20) ∧ 
  (∃ y : ℝ, y = (-2)^2 + (-2) * a + b ∧ y = 0) ∧ 
  b = (0^2 + 0 * a + b) → 
  a = 5 ∧ b = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_parabola_solution_l184_18498


namespace NUMINAMATH_GPT_largest_triangle_perimeter_l184_18480

theorem largest_triangle_perimeter (x : ℤ) (hx1 : 7 + 11 > x) (hx2 : 7 + x > 11) (hx3 : 11 + x > 7) (hx4 : 5 ≤ x) (hx5 : x < 18) : 
  7 + 11 + x = 35 :=
sorry

end NUMINAMATH_GPT_largest_triangle_perimeter_l184_18480


namespace NUMINAMATH_GPT_log_max_reciprocal_min_l184_18407

open Real

-- Definitions for the conditions
variables (x y : ℝ)
def conditions (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 2 * x + 5 * y = 20

-- Theorem statement for the first question
theorem log_max (x y : ℝ) (h : conditions x y) : log x + log y ≤ 1 :=
sorry

-- Theorem statement for the second question
theorem reciprocal_min (x y : ℝ) (h : conditions x y) : (1 / x) + (1 / y) ≥ (7 + 2 * sqrt 10) / 20 :=
sorry

end NUMINAMATH_GPT_log_max_reciprocal_min_l184_18407


namespace NUMINAMATH_GPT_Chloe_pairs_shoes_l184_18468

theorem Chloe_pairs_shoes (cost_per_shoe total_cost : ℤ) (h_cost: cost_per_shoe = 37) (h_total: total_cost = 1036) :
  (total_cost / cost_per_shoe) / 2 = 14 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_Chloe_pairs_shoes_l184_18468


namespace NUMINAMATH_GPT_maximize_triangle_areas_l184_18494

theorem maximize_triangle_areas (L W : ℝ) (h1 : 2 * L + 2 * W = 80) (h2 : L ≤ 25) : W = 15 :=
by 
  sorry

end NUMINAMATH_GPT_maximize_triangle_areas_l184_18494


namespace NUMINAMATH_GPT_ineq_one_of_two_sqrt_amgm_l184_18483

-- Lean 4 statement for Question 1
theorem ineq_one_of_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
sorry

-- Lean 4 statement for Question 2
theorem sqrt_amgm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 :=
sorry

end NUMINAMATH_GPT_ineq_one_of_two_sqrt_amgm_l184_18483


namespace NUMINAMATH_GPT_triangle_side_b_l184_18405

-- Define the conditions and state the problem
theorem triangle_side_b (A C : ℕ) (a b c : ℝ)
  (h1 : C = 4 * A)
  (h2 : a = 36)
  (h3 : c = 60) :
  b = 45 := by
  sorry

end NUMINAMATH_GPT_triangle_side_b_l184_18405


namespace NUMINAMATH_GPT_average_income_QR_l184_18425

theorem average_income_QR 
  (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (P + R) / 2 = 6200)
  (h3 : P = 3000) :
  (Q + R) / 2 = 5250 :=
  sorry

end NUMINAMATH_GPT_average_income_QR_l184_18425


namespace NUMINAMATH_GPT_ball_bounce_height_l184_18439

noncomputable def min_bounces (h₀ h_min : ℝ) (bounce_factor : ℝ) := 
  Nat.ceil (Real.log (h_min / h₀) / Real.log bounce_factor)

theorem ball_bounce_height :
  min_bounces 512 40 (3/4) = 8 :=
by
  sorry

end NUMINAMATH_GPT_ball_bounce_height_l184_18439


namespace NUMINAMATH_GPT_find_A_l184_18445

-- Define the four-digit number being a multiple of 9 and the sum of its digits condition
def digit_sum_multiple_of_9 (A : ℤ) : Prop :=
  (3 + A + A + 1) % 9 = 0

-- The Lean statement for the proof problem
theorem find_A (A : ℤ) (h : digit_sum_multiple_of_9 A) : A = 7 :=
sorry

end NUMINAMATH_GPT_find_A_l184_18445


namespace NUMINAMATH_GPT_total_cows_on_farm_l184_18411

-- Defining the conditions
variables (X H : ℕ) -- X is the number of cows per herd, H is the total number of herds
axiom half_cows_counted : 2800 = X * H / 2

-- The theorem stating the total number of cows on the entire farm
theorem total_cows_on_farm (X H : ℕ) (h1 : 2800 = X * H / 2) : 5600 = X * H := 
by 
  sorry

end NUMINAMATH_GPT_total_cows_on_farm_l184_18411


namespace NUMINAMATH_GPT_volume_of_prism_l184_18432

theorem volume_of_prism (a b c : ℝ) (h₁ : a * b = 48) (h₂ : b * c = 36) (h₃ : a * c = 50) : 
    (a * b * c = 170) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l184_18432


namespace NUMINAMATH_GPT_number_of_observations_is_14_l184_18454

theorem number_of_observations_is_14
  (mean_original : ℚ) (mean_new : ℚ) (original_sum : ℚ) 
  (corrected_sum : ℚ) (n : ℚ)
  (h1 : mean_original = 36)
  (h2 : mean_new = 36.5)
  (h3 : corrected_sum = original_sum + 7)
  (h4 : mean_new = corrected_sum / n)
  (h5 : original_sum = mean_original * n) :
  n = 14 :=
by
  -- Here goes the proof
  sorry

end NUMINAMATH_GPT_number_of_observations_is_14_l184_18454


namespace NUMINAMATH_GPT_gather_all_candies_l184_18447

theorem gather_all_candies (n : ℕ) (h₁ : n ≥ 4) (candies : ℕ) (h₂ : candies ≥ 4)
    (plates : Fin n → ℕ) :
    ∃ plate : Fin n, ∀ i : Fin n, i ≠ plate → plates i = 0 :=
sorry

end NUMINAMATH_GPT_gather_all_candies_l184_18447


namespace NUMINAMATH_GPT_tangents_collinear_F_minimum_area_triangle_l184_18485

noncomputable def ellipse_condition : Prop :=
  ∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1

noncomputable def point_P_on_line (P : ℝ × ℝ) : Prop :=
  P.1 = 4

noncomputable def tangent_condition (P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop) : Prop :=
  -- Tangent lines meet the ellipse equation at points A and B
  ellipse A ∧ ellipse B

noncomputable def collinear (A F B : ℝ × ℝ) : Prop :=
  (A.2 - F.2) * (B.1 - F.1) = (B.2 - F.2) * (A.1 - F.1)

noncomputable def minimum_area (P A B : ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((A.1 * B.2 + B.1 * P.2 + P.1 * A.2) - (A.2 * B.1 + B.2 * P.1 + P.2 * A.1))

theorem tangents_collinear_F (F : ℝ × ℝ) (P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop)
  (h_ellipse_F : F = (1, 0))
  (h_point_P_on_line : point_P_on_line P)
  (h_tangent_condition : tangent_condition P A B ellipse)
  (h_ellipse_def : ellipse_condition) :
  collinear A F B :=
sorry

theorem minimum_area_triangle (F P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop)
  (h_ellipse_F : F = (1, 0))
  (h_point_P_on_line : point_P_on_line P)
  (h_tangent_condition : tangent_condition P A B ellipse)
  (h_ellipse_def : ellipse_condition) :
  minimum_area P A B = 9 / 2 :=
sorry

end NUMINAMATH_GPT_tangents_collinear_F_minimum_area_triangle_l184_18485


namespace NUMINAMATH_GPT_inequality_solution_l184_18489

theorem inequality_solution (x : ℝ) : x ^ 2 < |x| + 2 ↔ -2 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l184_18489


namespace NUMINAMATH_GPT_acute_angle_of_parallel_vectors_l184_18467
open Real

theorem acute_angle_of_parallel_vectors (α : ℝ) (h₁ : abs (α * π / 180) < π / 2) :
  let a := (3 / 2, sin (α * π / 180))
  let b := (sin (α * π / 180), 1 / 6) 
  a.1 * b.2 = a.2 * b.1 → α = 30 :=
by
  sorry

end NUMINAMATH_GPT_acute_angle_of_parallel_vectors_l184_18467


namespace NUMINAMATH_GPT_Goat_guilty_l184_18499

-- Condition definitions
def Goat_lied : Prop := sorry
def Beetle_testimony_true : Prop := sorry
def Mosquito_testimony_true : Prop := sorry
def Goat_accused_Beetle_or_Mosquito : Prop := sorry
def Beetle_accused_Goat_or_Mosquito : Prop := sorry
def Mosquito_accused_Beetle_or_Goat : Prop := sorry

-- Theorem: The Goat is guilty
theorem Goat_guilty (G_lied : Goat_lied) 
    (B_true : Beetle_testimony_true) 
    (M_true : Mosquito_testimony_true)
    (G_accuse : Goat_accused_Beetle_or_Mosquito)
    (B_accuse : Beetle_accused_Goat_or_Mosquito)
    (M_accuse : Mosquito_accused_Beetle_or_Goat) : 
  Prop :=
  sorry

end NUMINAMATH_GPT_Goat_guilty_l184_18499


namespace NUMINAMATH_GPT_focus_of_parabola_l184_18421

-- Define the given parabola equation
def given_parabola (x : ℝ) : ℝ := 4 * x^2

-- Define what it means to be the focus of this parabola
def is_focus (focus : ℝ × ℝ) : Prop :=
  focus = (0, 1 / 16)

-- The theorem to prove
theorem focus_of_parabola : ∃ focus : ℝ × ℝ, is_focus focus :=
  by 
    use (0, 1 / 16)
    exact sorry

end NUMINAMATH_GPT_focus_of_parabola_l184_18421


namespace NUMINAMATH_GPT_minimum_effort_to_qualify_l184_18496

def minimum_effort_to_qualify_for_mop (AMC_points_per_effort : ℕ := 6 * 1/3)
                                       (AIME_points_per_effort : ℕ := 10 * 1/7)
                                       (USAMO_points_per_effort : ℕ := 1 * 1/10)
                                       (required_amc_aime_points : ℕ := 200)
                                       (required_usamo_points : ℕ := 21) : ℕ :=
  let max_amc_points : ℕ := 150
  let effort_amc : ℕ := (max_amc_points / AMC_points_per_effort) * 3
  let remaining_aime_points : ℕ := 200 - max_amc_points
  let effort_aime : ℕ := (remaining_aime_points / AIME_points_per_effort) * 7
  let effort_usamo : ℕ := required_usamo_points * 10
  let total_effort : ℕ := effort_amc + effort_aime + effort_usamo
  total_effort

theorem minimum_effort_to_qualify : minimum_effort_to_qualify_for_mop 6 (10 * 1/7) (1 * 1/10) 200 21 = 320 := by
  sorry

end NUMINAMATH_GPT_minimum_effort_to_qualify_l184_18496


namespace NUMINAMATH_GPT_percentage_caught_customers_l184_18477

noncomputable def total_sampling_percentage : ℝ := 0.25
noncomputable def caught_percentage : ℝ := 0.88

theorem percentage_caught_customers :
  total_sampling_percentage * caught_percentage = 0.22 :=
by
  sorry

end NUMINAMATH_GPT_percentage_caught_customers_l184_18477


namespace NUMINAMATH_GPT_sum_divides_exp_sum_l184_18484

theorem sum_divides_exp_sum (p a b c d : ℕ) [Fact (Nat.Prime p)] 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < p)
  (h6 : a^4 % p = b^4 % p) (h7 : b^4 % p = c^4 % p) (h8 : c^4 % p = d^4 % p) :
  (a + b + c + d) ∣ (a^2013 + b^2013 + c^2013 + d^2013) :=
sorry

end NUMINAMATH_GPT_sum_divides_exp_sum_l184_18484


namespace NUMINAMATH_GPT_minimum_value_of_function_l184_18446

noncomputable def y (x : ℝ) : ℝ := 4 * x + 25 / x

theorem minimum_value_of_function : ∃ x > 0, y x = 20 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_function_l184_18446


namespace NUMINAMATH_GPT_salad_dressing_percentage_l184_18402

variable (P Q : ℝ) -- P and Q are the amounts of dressings P and Q in grams

-- Conditions
variable (h1 : 0.3 * P + 0.1 * Q = 12) -- The combined vinegar percentage condition
variable (h2 : P + Q = 100)            -- The total weight condition

-- Statement to prove
theorem salad_dressing_percentage (P_percent : ℝ) 
    (h1 : 0.3 * P + 0.1 * Q = 12) (h2 : P + Q = 100) : 
    P / (P + Q) * 100 = 10 :=
sorry

end NUMINAMATH_GPT_salad_dressing_percentage_l184_18402


namespace NUMINAMATH_GPT_consumption_reduction_l184_18476

variable (P C : ℝ)

theorem consumption_reduction (h : P > 0 ∧ C > 0) : 
  (1.25 * P * (0.8 * C) = P * C) :=
by
  -- Conditions: original price P, original consumption C
  -- New price 1.25 * P, New consumption 0.8 * C
  sorry

end NUMINAMATH_GPT_consumption_reduction_l184_18476


namespace NUMINAMATH_GPT_absolute_sum_value_l184_18453

theorem absolute_sum_value (x1 x2 x3 x4 x5 : ℝ) 
(h : x1 + 1 = x2 + 2 ∧ x2 + 2 = x3 + 3 ∧ x3 + 3 = x4 + 4 ∧ x4 + 4 = x5 + 5 ∧ x5 + 5 = x1 + x2 + x3 + x4 + x5 + 6) :
  |(x1 + x2 + x3 + x4 + x5)| = 3.75 := 
by
  sorry

end NUMINAMATH_GPT_absolute_sum_value_l184_18453


namespace NUMINAMATH_GPT_calc1_calc2_calc3_calc4_l184_18463

theorem calc1 : 327 + 46 - 135 = 238 := by sorry
theorem calc2 : 1000 - 582 - 128 = 290 := by sorry
theorem calc3 : (124 - 62) * 6 = 372 := by sorry
theorem calc4 : 500 - 400 / 5 = 420 := by sorry

end NUMINAMATH_GPT_calc1_calc2_calc3_calc4_l184_18463


namespace NUMINAMATH_GPT_mrs_wilsborough_vip_tickets_l184_18422

theorem mrs_wilsborough_vip_tickets:
  let S := 500 -- Initial savings
  let PVIP := 100 -- Price per VIP ticket
  let preg := 50 -- Price per regular ticket
  let nreg := 3 -- Number of regular tickets
  let R := 150 -- Remaining savings after purchase
  
  -- The total amount spent on tickets is S - R
  S - R = PVIP * 2 + preg * nreg := 
by sorry

end NUMINAMATH_GPT_mrs_wilsborough_vip_tickets_l184_18422
