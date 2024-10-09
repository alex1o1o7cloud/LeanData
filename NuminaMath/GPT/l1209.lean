import Mathlib

namespace a1_minus_2a2_plus_3a3_minus_4a4_eq_48_l1209_120911

theorem a1_minus_2a2_plus_3a3_minus_4a4_eq_48:
  ∀ (a a_1 a_2 a_3 a_4 : ℝ),
  (∀ x : ℝ, (1 + 2 * x) ^ 4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  a_1 - 2 * a_2 + 3 * a_3 - 4 * a_4 = 48 :=
by
  sorry

end a1_minus_2a2_plus_3a3_minus_4a4_eq_48_l1209_120911


namespace min_expression_value_l1209_120956

theorem min_expression_value (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (∃ (min_val : ℝ), min_val = 12 ∧ (∀ (x y : ℝ), (x > 1) → (y > 1) →
  ((x^2 / (y - 1)) + (y^2 / (x - 1)) + (x + y) ≥ min_val))) :=
by
  sorry

end min_expression_value_l1209_120956


namespace third_team_pieces_l1209_120994

theorem third_team_pieces (total_pieces : ℕ) (first_team : ℕ) (second_team : ℕ) (third_team : ℕ) : 
  total_pieces = 500 → first_team = 189 → second_team = 131 → third_team = total_pieces - first_team - second_team → third_team = 180 :=
by
  intros h_total h_first h_second h_third
  rw [h_total, h_first, h_second] at h_third
  exact h_third

end third_team_pieces_l1209_120994


namespace sailboat_rental_cost_l1209_120926

-- Define the conditions
def rental_per_hour_ski := 80
def hours_per_day := 3
def days := 2
def cost_ski := (hours_per_day * days * rental_per_hour_ski)
def additional_cost := 120

-- Statement to prove
theorem sailboat_rental_cost :
  ∃ (S : ℕ), cost_ski = S + additional_cost → S = 360 := by
  sorry

end sailboat_rental_cost_l1209_120926


namespace function_inverse_l1209_120938

theorem function_inverse (x : ℝ) (h : ℝ → ℝ) (k : ℝ → ℝ) 
  (h_def : ∀ x, h x = 6 - 7 * x) 
  (k_def : ∀ x, k x = (6 - x) / 7) : 
  h (k x) = x ∧ k (h x) = x := 
  sorry

end function_inverse_l1209_120938


namespace exists_three_cycle_l1209_120941

variable {α : Type}

def tournament (P : α → α → Prop) : Prop :=
  (∃ (participants : List α), participants.length ≥ 3) ∧
  (∀ x y, x ≠ y → P x y ∨ P y x) ∧
  (∀ x, ∃ y, P x y)

theorem exists_three_cycle {α : Type} (P : α → α → Prop) :
  tournament P → ∃ A B C, P A B ∧ P B C ∧ P C A :=
by
  sorry

end exists_three_cycle_l1209_120941


namespace min_value_expr_l1209_120951

/-- Given x > y > 0 and x^2 - y^2 = 1, we need to prove that the minimum value of 2x^2 + 3y^2 - 4xy is 1. -/
theorem min_value_expr {x y : ℝ} (h1 : x > y) (h2 : y > 0) (h3 : x^2 - y^2 = 1) :
  2 * x^2 + 3 * y^2 - 4 * x * y = 1 :=
sorry

end min_value_expr_l1209_120951


namespace Sara_sister_notebooks_l1209_120935

theorem Sara_sister_notebooks :
  let initial_notebooks := 4 
  let ordered_notebooks := (3 / 2) * initial_notebooks -- 150% more notebooks
  let notebooks_after_order := initial_notebooks + ordered_notebooks
  let notebooks_after_loss := notebooks_after_order - 2 -- lost 2 notebooks
  let sold_notebooks := (1 / 4) * notebooks_after_loss -- sold 25% of remaining notebooks
  let notebooks_after_sales := notebooks_after_loss - sold_notebooks
  let notebooks_after_giveaway := notebooks_after_sales - 3 -- gave away 3 notebooks
  notebooks_after_giveaway = 3 := 
by {
  sorry
}

end Sara_sister_notebooks_l1209_120935


namespace one_third_of_6_3_eq_21_10_l1209_120980

theorem one_third_of_6_3_eq_21_10 : (6.3 / 3) = (21 / 10) := by
  sorry

end one_third_of_6_3_eq_21_10_l1209_120980


namespace linda_total_distance_l1209_120993

theorem linda_total_distance
  (miles_per_gallon : ℝ) (tank_capacity : ℝ) (initial_distance : ℝ) (refuel_amount : ℝ) (final_tank_fraction : ℝ)
  (fuel_used_first_segment : ℝ := initial_distance / miles_per_gallon)
  (initial_fuel_full : fuel_used_first_segment = tank_capacity)
  (total_fuel_after_refuel : ℝ := 0 + refuel_amount)
  (remaining_fuel_stopping : ℝ := final_tank_fraction * tank_capacity)
  (fuel_used_second_segment : ℝ := total_fuel_after_refuel - remaining_fuel_stopping)
  (distance_second_leg : ℝ := fuel_used_second_segment * miles_per_gallon) :
  initial_distance + distance_second_leg = 637.5 := by
  sorry

end linda_total_distance_l1209_120993


namespace sqrt_expression_range_l1209_120916

theorem sqrt_expression_range (x : ℝ) : x + 3 ≥ 0 ∧ x ≠ 0 ↔ x ≥ -3 ∧ x ≠ 0 :=
by
  sorry

end sqrt_expression_range_l1209_120916


namespace joe_money_fraction_l1209_120964

theorem joe_money_fraction :
  ∃ f : ℝ,
    (200 : ℝ) = 160 + (200 - 160) ∧
    160 - 160 * f - 20 = 40 + 160 * f + 20 ∧
    f = 1 / 4 :=
by
  -- The proof should go here.
  sorry

end joe_money_fraction_l1209_120964


namespace min_b_minus_a_l1209_120981

noncomputable def f (x : ℝ) : ℝ := 1 + x - (x^2) / 2 + (x^3) / 3
noncomputable def g (x : ℝ) : ℝ := 1 - x + (x^2) / 2 - (x^3) / 3
noncomputable def F (x : ℝ) : ℝ := f x * g x

theorem min_b_minus_a (a b : ℤ) (h : ∀ x, F x = 0 → a ≤ x ∧ x ≤ b) (h_a_lt_b : a < b) : b - a = 3 :=
sorry

end min_b_minus_a_l1209_120981


namespace problem_part_1_problem_part_2_l1209_120906

theorem problem_part_1 (a b : ℝ) (h1 : a * 1^2 - 3 * 1 + 2 = 0) (h2 : a * b^2 - 3 * b + 2 = 0) (h3 : 1 + b = 3 / a) (h4 : 1 * b = 2 / a) : a = 1 ∧ b = 2 :=
sorry

theorem problem_part_2 (m : ℝ) (h5 : a = 1) (h6 : b = 2) : 
  (m = 2 → ∀ x, ¬ (x^2 - (m + 2) * x + 2 * m < 0)) ∧
  (m < 2 → ∀ x, x ∈ Set.Ioo m 2 ↔ x^2 - (m + 2) * x + 2 * m < 0) ∧
  (m > 2 → ∀ x, x ∈ Set.Ioo 2 m ↔ x^2 - (m + 2) * x + 2 * m < 0) :=
sorry

end problem_part_1_problem_part_2_l1209_120906


namespace problem_solution_l1209_120914

noncomputable def solve_equation (x : ℝ) : Prop :=
  x ≠ 4 ∧ (x + 36 / (x - 4) = -9)

theorem problem_solution : {x : ℝ | solve_equation x} = {0, -5} :=
by
  sorry

end problem_solution_l1209_120914


namespace first_group_people_count_l1209_120917

def group_ice_cream (P : ℕ) : Prop :=
  let total_days_per_person1 := P * 10
  let total_days_per_person2 := 5 * 16
  total_days_per_person1 = total_days_per_person2

theorem first_group_people_count 
  (P : ℕ) 
  (H1 : group_ice_cream P) : 
  P = 8 := 
sorry

end first_group_people_count_l1209_120917


namespace min_students_l1209_120966

theorem min_students (b g : ℕ) 
  (h1 : 3 * b = 2 * g) 
  (h2 : (b + g) % 5 = 2) : 
  b + g = 57 :=
sorry

end min_students_l1209_120966


namespace percentage_problem_l1209_120943

variable (x : ℝ)
variable (y : ℝ)

theorem percentage_problem : 
  (x / 100 * 1442 - 36 / 100 * 1412) + 63 = 252 → x = 33.52 := by
  sorry

end percentage_problem_l1209_120943


namespace probability_neither_nearsighted_l1209_120955

-- Definitions based on problem conditions
def P_A : ℝ := 0.4
def P_not_A : ℝ := 1 - P_A
def event_B₁_not_nearsighted : Prop := true
def event_B₂_not_nearsighted : Prop := true

-- Independence assumption
variables (indep_B₁_B₂ : event_B₁_not_nearsighted) (event_B₂_not_nearsighted)

-- Theorem statement
theorem probability_neither_nearsighted (H1 : P_A = 0.4) (H2 : P_not_A = 0.6)
  (indep_B₁_B₂ : event_B₁_not_nearsighted ∧ event_B₂_not_nearsighted) :
  P_not_A * P_not_A = 0.36 :=
by
  -- Proof omitted
  sorry

end probability_neither_nearsighted_l1209_120955


namespace find_minimal_sum_n_l1209_120971

noncomputable def minimal_sum_n {a : ℕ → ℤ} {S : ℕ → ℤ} (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 = -9) (h3 : S 3 = S 7) : ℕ := 
     5

theorem find_minimal_sum_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 = -9) (h3 : S 3 = S 7) : minimal_sum_n h1 h2 h3 = 5 :=
    sorry

end find_minimal_sum_n_l1209_120971


namespace bucket_capacity_l1209_120967

theorem bucket_capacity :
  (∃ (x : ℝ), 30 * x = 45 * 9) → 13.5 = 13.5 :=
by
  -- proof needed
  sorry

end bucket_capacity_l1209_120967


namespace polygon_sides_l1209_120991

theorem polygon_sides (n : ℕ) (h_sum : 180 * (n - 2) = 1980) : n = 13 :=
by {
  sorry
}

end polygon_sides_l1209_120991


namespace problem_trigonometric_identity_l1209_120933

-- Define the problem conditions
theorem problem_trigonometric_identity
  (α : ℝ)
  (h : 3 * Real.sin (33 * Real.pi / 14 + α) = -5 * Real.cos (5 * Real.pi / 14 + α)) :
  Real.tan (5 * Real.pi / 14 + α) = -5 / 3 :=
sorry

end problem_trigonometric_identity_l1209_120933


namespace total_balls_l1209_120944

theorem total_balls (blue red green yellow purple orange black white : ℕ) 
  (h1 : blue = 8)
  (h2 : red = 5)
  (h3 : green = 3 * (2 * blue - 1))
  (h4 : yellow = Nat.floor (2 * Real.sqrt (red * blue)))
  (h5 : purple = 4 * (blue + green))
  (h6 : orange = 7)
  (h7 : black + white = blue + red + green + yellow + purple + orange)
  (h8 : blue + red + green + yellow + purple + orange + black + white = 3 * (red + green + yellow + purple) + orange / 2)
  : blue + red + green + yellow + purple + orange + black + white = 829 :=
by
  sorry

end total_balls_l1209_120944


namespace probability_two_green_in_four_l1209_120953

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def bag_marbles := 12
def green_marbles := 5
def blue_marbles := 3
def yellow_marbles := 4
def total_picked := 4
def green_picked := 2
def remaining_marbles := bag_marbles - green_marbles
def non_green_picked := total_picked - green_picked

theorem probability_two_green_in_four : 
  (choose green_marbles green_picked * choose remaining_marbles non_green_picked : ℚ) / (choose bag_marbles total_picked) = 14 / 33 := by
  sorry

end probability_two_green_in_four_l1209_120953


namespace car_speed_l1209_120948

theorem car_speed (v : ℝ) : 
  (4 + (1 / (80 / 3600))) = (1 / (v / 3600)) → v = 3600 / 49 :=
sorry

end car_speed_l1209_120948


namespace total_marks_math_physics_l1209_120952

variable (M P C : ℕ)

theorem total_marks_math_physics (h1 : C = P + 10) (h2 : (M + C) / 2 = 35) : M + P = 60 :=
by
  sorry

end total_marks_math_physics_l1209_120952


namespace increasing_iff_a_ge_half_l1209_120929

noncomputable def f (a x : ℝ) : ℝ := (2 / 3) * x ^ 3 + (1 / 2) * (a - 1) * x ^ 2 + a * x + 1

theorem increasing_iff_a_ge_half (a : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → (2 * x ^ 2 + (a - 1) * x + a) ≥ 0) ↔ a ≥ -1 / 2 :=
sorry

end increasing_iff_a_ge_half_l1209_120929


namespace range_of_t_minus_1_over_t_minus_3_l1209_120932

variable {f : ℝ → ℝ}

-- Function conditions: monotonically decreasing and odd
axiom f_mono_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Condition on the real number t
variable {t : ℝ}
axiom f_condition : f (t^2 - 2 * t) + f (-3) > 0

-- Question: Prove the range of (t-1)/(t-3)
theorem range_of_t_minus_1_over_t_minus_3 (h : -1 < t ∧ t < 3) : 
  ((t - 1) / (t - 3)) < 1/2 :=
  sorry

end range_of_t_minus_1_over_t_minus_3_l1209_120932


namespace divisor_is_50_l1209_120997

theorem divisor_is_50 (D : ℕ) (h1 : ∃ n, n = 44 * 432 ∧ n % 44 = 0)
                      (h2 : ∃ n, n = 44 * 432 ∧ n % D = 8) : D = 50 :=
by
  sorry

end divisor_is_50_l1209_120997


namespace range_of_mu_l1209_120940

noncomputable def problem_statement (a b μ : ℝ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (0 < μ) ∧ (1 / a + 9 / b = 1) → (0 < μ ∧ μ ≤ 16)

theorem range_of_mu (a b μ : ℝ) : problem_statement a b μ :=
  sorry

end range_of_mu_l1209_120940


namespace least_number_of_pennies_l1209_120927

theorem least_number_of_pennies (a : ℕ) :
  (a ≡ 1 [MOD 7]) ∧ (a ≡ 0 [MOD 3]) → a = 15 := by
  sorry

end least_number_of_pennies_l1209_120927


namespace lindy_total_distance_l1209_120984

theorem lindy_total_distance (distance_jc : ℝ) (speed_j : ℝ) (speed_c : ℝ) (speed_l : ℝ)
  (h1 : distance_jc = 270) (h2 : speed_j = 4) (h3 : speed_c = 5) (h4 : speed_l = 8) : 
  ∃ time : ℝ, time = distance_jc / (speed_j + speed_c) ∧ speed_l * time = 240 :=
by
  sorry

end lindy_total_distance_l1209_120984


namespace unique_solution_x_y_z_l1209_120974

theorem unique_solution_x_y_z (x y z : ℕ) (h1 : Prime y) (h2 : ¬ z % 3 = 0) (h3 : ¬ z % y = 0) :
    x^3 - y^3 = z^2 ↔ (x, y, z) = (8, 7, 13) := by
  sorry

end unique_solution_x_y_z_l1209_120974


namespace num_cows_l1209_120915

-- Define the context
variable (C H L Heads : ℕ)

-- Define the conditions
axiom condition1 : L = 2 * Heads + 8
axiom condition2 : L = 4 * C + 2 * H
axiom condition3 : Heads = C + H

-- State the goal
theorem num_cows : C = 4 := by
  sorry

end num_cows_l1209_120915


namespace find_coords_of_P_cond1_find_coords_of_P_cond2_find_coords_of_P_cond3_l1209_120905

variables {m : ℝ} 
def point_on_y_axis (P : (ℝ × ℝ)) := P = (0, -3)
def point_distance_to_y_axis (P : (ℝ × ℝ)) := P = (6, 0) ∨ P = (-6, -6)
def point_in_third_quadrant_and_equidistant (P : (ℝ × ℝ)) := P = (-6, -6)

theorem find_coords_of_P_cond1 (P : ℝ × ℝ) (h : 2 * m + 4 = 0) : point_on_y_axis P ↔ P = (0, -3) :=
by {
  sorry
}

theorem find_coords_of_P_cond2 (P : ℝ × ℝ) (h : abs (2 * m + 4) = 6) : point_distance_to_y_axis P ↔ (P = (6, 0) ∨ P = (-6, -6)) :=
by {
  sorry
}

theorem find_coords_of_P_cond3 (P : ℝ × ℝ) (h1 : 2 * m + 4 < 0) (h2 : m - 1 < 0) (h3 : abs (2 * m + 4) = abs (m - 1)) : point_in_third_quadrant_and_equidistant P ↔ P = (-6, -6) :=
by {
  sorry
}

end find_coords_of_P_cond1_find_coords_of_P_cond2_find_coords_of_P_cond3_l1209_120905


namespace total_money_made_l1209_120961

def dvd_price : ℕ := 240
def dvd_quantity : ℕ := 8
def washing_machine_price : ℕ := 898

theorem total_money_made : dvd_price * dvd_quantity + washing_machine_price = 240 * 8 + 898 :=
by
  sorry

end total_money_made_l1209_120961


namespace no_separation_sister_chromatids_first_meiotic_l1209_120919

-- Definitions for the steps happening during the first meiotic division
def first_meiotic_division :=
  ∃ (prophase_I : Prop) (metaphase_I : Prop) (anaphase_I : Prop) (telophase_I : Prop),
    prophase_I ∧ metaphase_I ∧ anaphase_I ∧ telophase_I

def pairing_homologous_chromosomes (prophase_I : Prop) := prophase_I
def crossing_over (prophase_I : Prop) := prophase_I
def separation_homologous_chromosomes (anaphase_I : Prop) := anaphase_I
def separation_sister_chromatids (mitosis : Prop) (second_meiotic_division : Prop) :=
  mitosis ∨ second_meiotic_division

-- Theorem to prove that the separation of sister chromatids does not occur during the first meiotic division
theorem no_separation_sister_chromatids_first_meiotic
  (prophase_I metaphase_I anaphase_I telophase_I mitosis second_meiotic_division : Prop)
  (h1: first_meiotic_division)
  (h2 : pairing_homologous_chromosomes prophase_I)
  (h3 : crossing_over prophase_I)
  (h4 : separation_homologous_chromosomes anaphase_I)
  (h5 : separation_sister_chromatids mitosis second_meiotic_division) : 
  ¬ separation_sister_chromatids prophase_I anaphase_I :=
by
  sorry

end no_separation_sister_chromatids_first_meiotic_l1209_120919


namespace parabola_translation_l1209_120958

theorem parabola_translation :
  (∀ x, y = x^2) →
  (∀ x, y = (x + 1)^2 - 2) :=
by
  sorry

end parabola_translation_l1209_120958


namespace origin_inside_ellipse_l1209_120976

theorem origin_inside_ellipse (k : ℝ) (h : k^2 * 0^2 + 0^2 - 4*k*0 + 2*k*0 + k^2 - 1 < 0) : 0 < |k| ∧ |k| < 1 :=
by
  sorry

end origin_inside_ellipse_l1209_120976


namespace smallest_sum_of_squares_l1209_120937

theorem smallest_sum_of_squares :
  ∃ (x y : ℤ), x^2 - y^2 = 175 ∧ x^2 ≥ 36 ∧ y^2 ≥ 36 ∧ x^2 + y^2 = 625 :=
by
  sorry

end smallest_sum_of_squares_l1209_120937


namespace find_x_5pi_over_4_l1209_120950

open Real

theorem find_x_5pi_over_4 (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * π) (h3 : sin x + cos x = -sqrt 2) : x = 5 * π / 4 := 
sorry

end find_x_5pi_over_4_l1209_120950


namespace problem1_problem2_l1209_120973

theorem problem1 (a b : ℝ) :
  5 * a * b^2 - 2 * a^2 * b + 3 * a * b^2 - a^2 * b - 4 * a * b^2 = 4 * a * b^2 - 3 * a^2 * b := 
by sorry

theorem problem2 (m n : ℝ) :
  -5 * m * n^2 - (2 * m^2 * n - 2 * (m^2 * n - 2 * m * n^2)) = -9 * m * n^2 := 
by sorry

end problem1_problem2_l1209_120973


namespace emma_final_amount_l1209_120913

theorem emma_final_amount
  (initial_amount : ℕ)
  (furniture_cost : ℕ)
  (fraction_given_to_anna : ℚ)
  (amount_left : ℕ) :
  initial_amount = 2000 →
  furniture_cost = 400 →
  fraction_given_to_anna = 3 / 4 →
  amount_left = initial_amount - furniture_cost →
  amount_left - (fraction_given_to_anna * amount_left : ℚ) = 400 :=
by
  intros h_initial h_furniture h_fraction h_amount_left
  sorry

end emma_final_amount_l1209_120913


namespace SummitAcademy_Contestants_l1209_120908

theorem SummitAcademy_Contestants (s j : ℕ)
  (h1 : s > 0)
  (h2 : j > 0)
  (hs : (1 / 3 : ℚ) * s = (3 / 4 : ℚ) * j) :
  s = (9 / 4 : ℚ) * j :=
sorry

end SummitAcademy_Contestants_l1209_120908


namespace set_intersection_complement_l1209_120936

def U := {x : ℝ | x > -3}
def A := {x : ℝ | x < -2 ∨ x > 3}
def B := {x : ℝ | -1 ≤ x ∧ x ≤ 4}

theorem set_intersection_complement :
  A ∩ (U \ B) = {x : ℝ | -3 < x ∧ x < -2 ∨ x > 4} :=
by sorry

end set_intersection_complement_l1209_120936


namespace sequence_problem_l1209_120910

variable {n : ℕ}

-- We define the arithmetic sequence conditions
noncomputable def a_n : ℕ → ℕ
| n => 2 * n + 1

-- Conditions that the sequence must satisfy
axiom a_3_eq_7 : a_n 3 = 7
axiom a_5_a_7_eq_26 : a_n 5 + a_n 7 = 26

-- Define the sum of the sequence
noncomputable def S_n (n : ℕ) := n^2 + 2 * n

-- Define the sequence b_n
noncomputable def b_n (n : ℕ) := 1 / (a_n n ^ 2 - 1 : ℝ)

-- Define the sum of the sequence b_n
noncomputable def T_n (n : ℕ) := (n / (4 * (n + 1)) : ℝ)

-- The main theorem to prove
theorem sequence_problem :
  (a_n n = 2 * n + 1) ∧ (S_n n = n^2 + 2 * n) ∧ (T_n n = n / (4 * (n + 1))) :=
  sorry

end sequence_problem_l1209_120910


namespace find_x_l1209_120922

theorem find_x (x : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 - i) * (Complex.ofReal x + i) = 1 + i) : x = 0 :=
by sorry

end find_x_l1209_120922


namespace john_bathroom_uses_during_movie_and_intermissions_l1209_120975

-- Define the conditions
def uses_bathroom_interval := 50   -- John uses the bathroom every 50 minutes
def walking_time := 5              -- It takes him an additional 5 minutes to walk to and from the bathroom
def movie_length := 150            -- The movie length in minutes (2.5 hours)
def intermission_length := 15      -- Each intermission length in minutes
def intermission_count := 2        -- The number of intermissions

-- Derived condition
def effective_interval := uses_bathroom_interval + walking_time

-- Total movie time including intermissions
def total_movie_time := movie_length + (intermission_length * intermission_count)

-- Define the theorem to be proved
theorem john_bathroom_uses_during_movie_and_intermissions : 
  ∃ n : ℕ, n = 3 + 2 ∧ total_movie_time = 180 ∧ effective_interval = 55 :=
by
  sorry

end john_bathroom_uses_during_movie_and_intermissions_l1209_120975


namespace diameter_of_circle_is_60_l1209_120962

noncomputable def diameter_of_circle (M N : ℝ) : ℝ :=
  if h : N ≠ 0 then 2 * (M / N * (1 / (2 * Real.pi))) else 0

theorem diameter_of_circle_is_60 (M N : ℝ) (h : M / N = 15) :
  diameter_of_circle M N = 60 :=
by
  sorry

end diameter_of_circle_is_60_l1209_120962


namespace initial_milk_amount_l1209_120909

theorem initial_milk_amount (d : ℚ) (r : ℚ) (T : ℚ) 
  (hd : d = 0.4) 
  (hr : r = 0.69) 
  (h_remaining : r = (1 - d) * T) : 
  T = 1.15 := 
  sorry

end initial_milk_amount_l1209_120909


namespace sage_reflection_day_l1209_120969

theorem sage_reflection_day 
  (day_of_reflection_is_jan_1 : Prop)
  (equal_days_in_last_5_years : Prop)
  (new_year_10_years_ago_was_friday : Prop)
  (reflections_in_21st_century : Prop) : 
  ∃ (day : String), day = "Thursday" :=
by
  sorry

end sage_reflection_day_l1209_120969


namespace value_of_d_l1209_120992

theorem value_of_d (d : ℝ) (h : x^2 - 60 * x + d = (x - 30)^2) : d = 900 :=
by { sorry }

end value_of_d_l1209_120992


namespace n_fraction_of_sum_l1209_120901

theorem n_fraction_of_sum (n S : ℝ) (h1 : n = S / 5) (h2 : S ≠ 0) :
  n = 1 / 6 * ((S + (S / 5))) :=
by
  sorry

end n_fraction_of_sum_l1209_120901


namespace sequence_count_l1209_120957

theorem sequence_count (a : ℕ → ℤ) (h₁ : a 1 = 0) (h₂ : a 11 = 4) 
  (h₃ : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → |a (k + 1) - a k| = 1) : 
  ∃ (n : ℕ), n = 120 :=
by
  sorry

end sequence_count_l1209_120957


namespace find_pairs_l1209_120978

theorem find_pairs (m n : ℕ) (h1 : 1 < m) (h2 : 1 < n) (h3 : (mn - 1) ∣ (n^3 - 1)) :
  ∃ k : ℕ, 1 < k ∧ ((m = k ∧ n = k^2) ∨ (m = k^2 ∧ n = k)) :=
sorry

end find_pairs_l1209_120978


namespace negation_of_existential_statement_l1209_120965

variable (A : Set ℝ)

theorem negation_of_existential_statement :
  ¬(∃ x ∈ A, x^2 - 2 * x - 3 > 0) ↔ ∀ x ∈ A, x^2 - 2 * x - 3 ≤ 0 := by
  sorry

end negation_of_existential_statement_l1209_120965


namespace ellipse_eccentricity_l1209_120996

def ellipse {a : ℝ} (h : a^2 - 4 = 4) : Prop :=
  ∃ c e : ℝ, (c = 2) ∧ (e = c / a) ∧ (e = (Real.sqrt 2) / 2)

theorem ellipse_eccentricity (a : ℝ) (h : a^2 - 4 = 4) : 
  ellipse h :=
by
  sorry

end ellipse_eccentricity_l1209_120996


namespace number_one_half_more_equals_twenty_five_percent_less_l1209_120920

theorem number_one_half_more_equals_twenty_five_percent_less (n : ℤ) : 
    (80 - 0.25 * 80 = 60) → ((3 / 2 : ℚ) * n = 60) → (n = 40) :=
by
  intros h1 h2
  sorry

end number_one_half_more_equals_twenty_five_percent_less_l1209_120920


namespace cos_double_angle_l1209_120995

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + 3 * Real.pi / 2) = Real.sqrt 3 / 3) : Real.cos (2 * α) = -1 / 3 := 
  sorry

end cos_double_angle_l1209_120995


namespace Frank_time_correct_l1209_120954

def Dave_time := 10
def Chuck_time := 5 * Dave_time
def Erica_time := 13 * Chuck_time / 10
def Frank_time := 12 * Erica_time / 10

theorem Frank_time_correct : Frank_time = 78 :=
by
  sorry

end Frank_time_correct_l1209_120954


namespace min_value_of_f_l1209_120946

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (Real.log x / Real.log 2) * (Real.log (2 * x) / Real.log 2) else 0

theorem min_value_of_f : ∃ x > 0, f x = -1/4 :=
sorry

end min_value_of_f_l1209_120946


namespace probability_two_cities_less_than_8000_l1209_120987

-- Define the city names
inductive City
| Bangkok | CapeTown | Honolulu | London | NewYork
deriving DecidableEq, Inhabited

-- Define the distance between cities
def distance : City → City → ℕ
| City.Bangkok, City.CapeTown  => 6300
| City.Bangkok, City.Honolulu  => 6609
| City.Bangkok, City.London    => 5944
| City.Bangkok, City.NewYork   => 8650
| City.CapeTown, City.Bangkok  => 6300
| City.CapeTown, City.Honolulu => 11535
| City.CapeTown, City.London   => 5989
| City.CapeTown, City.NewYork  => 7800
| City.Honolulu, City.Bangkok  => 6609
| City.Honolulu, City.CapeTown => 11535
| City.Honolulu, City.London   => 7240
| City.Honolulu, City.NewYork  => 4980
| City.London, City.Bangkok    => 5944
| City.London, City.CapeTown   => 5989
| City.London, City.Honolulu   => 7240
| City.London, City.NewYork    => 3470
| City.NewYork, City.Bangkok   => 8650
| City.NewYork, City.CapeTown  => 7800
| City.NewYork, City.Honolulu  => 4980
| City.NewYork, City.London    => 3470
| _, _                         => 0

-- Prove the probability
theorem probability_two_cities_less_than_8000 :
  let pairs := [(City.Bangkok, City.CapeTown), (City.Bangkok, City.Honolulu), (City.Bangkok, City.London), (City.CapeTown, City.London), (City.CapeTown, City.NewYork), (City.Honolulu, City.London), (City.Honolulu, City.NewYork), (City.London, City.NewYork)]
  (pairs.length : ℚ) / 10 = 4 / 5 :=
sorry

end probability_two_cities_less_than_8000_l1209_120987


namespace plane_equation_intercept_l1209_120945

theorem plane_equation_intercept (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  ∀ x y z : ℝ, ∃ k : ℝ, k = 1 → (x / a + y / b + z / c) = k :=
by sorry

end plane_equation_intercept_l1209_120945


namespace sin_75_deg_l1209_120998

theorem sin_75_deg : Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := 
by sorry

end sin_75_deg_l1209_120998


namespace function_is_constant_and_straight_line_l1209_120988

-- Define a function f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- Condition: The derivative of f is 0 everywhere
axiom derivative_zero_everywhere : ∀ x, deriv f x = 0

-- Conclusion: f is a constant function
theorem function_is_constant_and_straight_line : ∃ C : ℝ, ∀ x, f x = C := by
  sorry

end function_is_constant_and_straight_line_l1209_120988


namespace ones_digit_of_six_power_l1209_120963

theorem ones_digit_of_six_power (n : ℕ) (hn : n ≥ 1) : (6 ^ n) % 10 = 6 :=
by
  sorry

example : (6 ^ 34) % 10 = 6 :=
by
  have h : 34 ≥ 1 := by norm_num
  exact ones_digit_of_six_power 34 h

end ones_digit_of_six_power_l1209_120963


namespace discriminant_eq_13_l1209_120949

theorem discriminant_eq_13 (m : ℝ) (h : (3)^2 - 4*1*(-m) = 13) : m = 1 :=
sorry

end discriminant_eq_13_l1209_120949


namespace labor_cost_per_hour_l1209_120900

theorem labor_cost_per_hour (total_repair_cost part_cost labor_hours : ℕ)
    (h1 : total_repair_cost = 2400)
    (h2 : part_cost = 1200)
    (h3 : labor_hours = 16) :
    (total_repair_cost - part_cost) / labor_hours = 75 := by
  sorry

end labor_cost_per_hour_l1209_120900


namespace problem1_problem2_l1209_120986

-- Definitions based on the conditions
def internal_medicine_doctors : ℕ := 12
def surgeons : ℕ := 8
def total_doctors : ℕ := internal_medicine_doctors + surgeons
def team_size : ℕ := 5

-- Problem 1: Both doctor A and B must join the team
theorem problem1 : ∃ (ways : ℕ), ways = 816 :=
  by
    let remaining_doctors := total_doctors - 2
    let choose := remaining_doctors.choose (team_size - 2)
    have h1 : choose = 816 := sorry
    exact ⟨choose, h1⟩

-- Problem 2: At least one of doctors A or B must join the team
theorem problem2 : ∃ (ways : ℕ), ways = 5661 :=
  by
    let remaining_doctors := total_doctors - 1
    let scenario1 := 2 * remaining_doctors.choose (team_size - 1)
    let scenario2 := (total_doctors - 2).choose (team_size - 2)
    let total_ways := scenario1 + scenario2
    have h2 : total_ways = 5661 := sorry
    exact ⟨total_ways, h2⟩

end problem1_problem2_l1209_120986


namespace alarm_clock_shows_noon_in_14_minutes_l1209_120902

-- Definitions based on given problem conditions
def clockRunsSlow (clock_time real_time : ℕ) : Prop :=
  clock_time = real_time * 56 / 60

def timeSinceSet : ℕ := 210 -- 3.5 hours in minutes
def correctClockShowsNoon : ℕ := 720 -- Noon in minutes (12*60)

-- Main statement to prove
theorem alarm_clock_shows_noon_in_14_minutes :
  ∃ minutes : ℕ, clockRunsSlow (timeSinceSet * 56 / 60) timeSinceSet ∧ correctClockShowsNoon - (480 + timeSinceSet * 56 / 60) = minutes ∧ minutes = 14 := 
by
  sorry

end alarm_clock_shows_noon_in_14_minutes_l1209_120902


namespace solve_equation_1_solve_equation_2_l1209_120921

theorem solve_equation_1 (x : ℝ) :
  x^2 - 10 * x + 16 = 0 → x = 8 ∨ x = 2 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) :
  x * (x - 3) = 6 - 2 * x → x = 3 ∨ x = -2 :=
by
  sorry

end solve_equation_1_solve_equation_2_l1209_120921


namespace count_random_events_l1209_120977

-- Definitions based on conditions in the problem
def total_products : ℕ := 100
def genuine_products : ℕ := 95
def defective_products : ℕ := 5
def drawn_products : ℕ := 6

-- Events definitions
def event_1 := drawn_products > defective_products  -- at least 1 genuine product
def event_2 := drawn_products ≥ 3  -- at least 3 defective products
def event_3 := drawn_products = defective_products  -- all 6 are defective
def event_4 := drawn_products - 2 = 4  -- 2 defective and 4 genuine products

-- Dummy definition for random event counter state in the problem context
def random_events : ℕ := 2

-- Main theorem statement
theorem count_random_events :
  (event_1 → true) ∧ 
  (event_2 ∧ ¬ event_3 ∧ event_4) →
  random_events = 2 :=
by
  sorry

end count_random_events_l1209_120977


namespace factor_expression_l1209_120930

theorem factor_expression (x : ℝ) :
  (7 * x^6 + 36 * x^4 - 8) - (3 * x^6 - 4 * x^4 + 6) = 2 * (2 * x^6 + 20 * x^4 - 7) :=
  sorry

end factor_expression_l1209_120930


namespace Tim_total_expenditure_l1209_120979

theorem Tim_total_expenditure 
  (appetizer_price : ℝ) (main_course_price : ℝ) (dessert_price : ℝ)
  (appetizer_tip_percentage : ℝ) (main_course_tip_percentage : ℝ) (dessert_tip_percentage : ℝ) :
  appetizer_price = 12.35 →
  main_course_price = 27.50 →
  dessert_price = 9.95 →
  appetizer_tip_percentage = 0.18 →
  main_course_tip_percentage = 0.20 →
  dessert_tip_percentage = 0.15 →
  appetizer_price * (1 + appetizer_tip_percentage) + 
  main_course_price * (1 + main_course_tip_percentage) + 
  dessert_price * (1 + dessert_tip_percentage) = 12.35 * 1.18 + 27.50 * 1.20 + 9.95 * 1.15 :=
  by sorry

end Tim_total_expenditure_l1209_120979


namespace units_digit_2_pow_2130_l1209_120903

theorem units_digit_2_pow_2130 : (Nat.pow 2 2130) % 10 = 4 :=
by sorry

end units_digit_2_pow_2130_l1209_120903


namespace ellipse_center_x_coordinate_l1209_120904

theorem ellipse_center_x_coordinate (C : ℝ × ℝ)
  (h1 : C.1 = 3)
  (h2 : 4 ≤ C.2 ∧ C.2 ≤ 12)
  (hx : ∃ F1 F2 : ℝ × ℝ, F1 = (3, 4) ∧ F2 = (3, 12)
    ∧ (F1.1 = F2.1 ∧ F1.2 < F2.2)
    ∧ C = ((F1.1 + F2.1)/2, (F1.2 + F2.2)/2))
  (tangent : ∀ P : ℝ × ℝ, (P.1 - 0) * (P.2 - 0) = 0)
  (ellipse : ∃ a b : ℝ, a > 0 ∧ b > 0
    ∧ ∀ P : ℝ × ℝ,
      (P.1 - C.1)^2/a^2 + (P.2 - C.2)^2/b^2 = 1) :
   C.1 = 3 := sorry

end ellipse_center_x_coordinate_l1209_120904


namespace jeremie_friends_l1209_120942

-- Define the costs as constants.
def ticket_cost : ℕ := 18
def snack_cost : ℕ := 5
def total_cost : ℕ := 92
def per_person_cost : ℕ := ticket_cost + snack_cost

-- Define the number of friends Jeremie is going with (to be solved/proven).
def number_of_friends (total_cost : ℕ) (per_person_cost : ℕ) : ℕ :=
  let total_people := total_cost / per_person_cost
  total_people - 1

-- The statement that we want to prove.
theorem jeremie_friends : number_of_friends total_cost per_person_cost = 3 := by
  sorry

end jeremie_friends_l1209_120942


namespace jacob_total_distance_l1209_120947

/- Jacob jogs at a constant rate of 4 miles per hour.
   He jogs for 2 hours, then stops to take a rest for 30 minutes.
   After the break, he continues jogging for another 1 hour.
   Prove that the total distance jogged by Jacob is 12.0 miles.
-/
theorem jacob_total_distance :
  let joggingSpeed := 4 -- in miles per hour
  let jogBeforeBreak := 2 -- in hours
  let restDuration := 0.5 -- in hours (though it does not affect the distance)
  let jogAfterBreak := 1 -- in hours
  let totalDistance := joggingSpeed * jogBeforeBreak + joggingSpeed * jogAfterBreak
  totalDistance = 12.0 := 
by
  sorry

end jacob_total_distance_l1209_120947


namespace clara_weight_l1209_120912

-- Define the weights of Alice and Clara
variables (a c : ℕ)

-- Define the conditions given in the problem
def condition1 := a + c = 240
def condition2 := c - a = c / 3

-- The theorem to prove Clara's weight given the conditions
theorem clara_weight : condition1 a c → condition2 a c → c = 144 :=
by
  intros h1 h2
  sorry

end clara_weight_l1209_120912


namespace complement_of_A_in_I_is_246_l1209_120999

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def complement_A_in_I : Set ℕ := {2, 4, 6}

theorem complement_of_A_in_I_is_246 :
  (universal_set \ set_A) = complement_A_in_I :=
  by sorry

end complement_of_A_in_I_is_246_l1209_120999


namespace rhombus_diagonal_length_l1209_120939

-- Define a rhombus with one diagonal of 10 cm and a perimeter of 52 cm.
theorem rhombus_diagonal_length (d : ℝ) 
  (h1 : ∃ a b c : ℝ, a = 10 ∧ b = d ∧ c = 13) -- The diagonals and side of rhombus.
  (h2 : 52 = 4 * c) -- The perimeter condition.
  (h3 : c^2 = (d/2)^2 + (10/2)^2) -- The relationship from Pythagorean theorem.
  : d = 24 :=
by
  sorry

end rhombus_diagonal_length_l1209_120939


namespace xy_zero_iff_x_zero_necessary_not_sufficient_l1209_120983

theorem xy_zero_iff_x_zero_necessary_not_sufficient {x y : ℝ} : 
  (x * y = 0) → ((x = 0) ∨ (y = 0)) ∧ ¬((x = 0) → (x * y ≠ 0)) := 
sorry

end xy_zero_iff_x_zero_necessary_not_sufficient_l1209_120983


namespace time_to_cross_lake_one_direction_l1209_120925

-- Definitions for our conditions
def cost_per_hour := 10
def total_cost_round_trip := 80

-- Statement we want to prove
theorem time_to_cross_lake_one_direction : (total_cost_round_trip / cost_per_hour) / 2 = 4 :=
  by
  sorry

end time_to_cross_lake_one_direction_l1209_120925


namespace asymptotes_of_hyperbola_l1209_120924

theorem asymptotes_of_hyperbola 
  (x y : ℝ)
  (h : x^2 / 4 - y^2 / 36 = 1) : 
  (y = 3 * x) ∨ (y = -3 * x) :=
sorry

end asymptotes_of_hyperbola_l1209_120924


namespace ratio_of_boys_to_girls_l1209_120990

open Nat

theorem ratio_of_boys_to_girls
    (B G : ℕ) 
    (boys_avg : ℕ) 
    (girls_avg : ℕ) 
    (class_avg : ℕ)
    (h1 : boys_avg = 90)
    (h2 : girls_avg = 96)
    (h3 : class_avg = 94)
    (h4 : 94 * (B + G) = 90 * B + 96 * G) :
    2 * B = G :=
by
  sorry

end ratio_of_boys_to_girls_l1209_120990


namespace total_valid_votes_l1209_120959

theorem total_valid_votes (V : ℝ) (H_majority : 0.70 * V - 0.30 * V = 188) : V = 470 :=
by
  sorry

end total_valid_votes_l1209_120959


namespace farmhands_work_hours_l1209_120923

def apples_per_pint (variety: String) : ℕ :=
  match variety with
  | "golden_delicious" => 20
  | "pink_lady" => 40
  | _ => 0

def total_apples_for_pints (pints: ℕ) : ℕ :=
  (apples_per_pint "golden_delicious") * pints + (apples_per_pint "pink_lady") * pints

def apples_picked_per_hour_per_farmhand : ℕ := 240

def num_farmhands : ℕ := 6

def total_apples_picked_per_hour : ℕ :=
  num_farmhands * apples_picked_per_hour_per_farmhand

def ratio_golden_to_pink : ℕ × ℕ := (1, 2)

def haley_cider_pints : ℕ := 120

def hours_worked (pints: ℕ) (picked_per_hour: ℕ): ℕ :=
  (total_apples_for_pints pints) / picked_per_hour

theorem farmhands_work_hours :
  hours_worked haley_cider_pints total_apples_picked_per_hour = 5 := by
  sorry

end farmhands_work_hours_l1209_120923


namespace height_of_table_without_book_l1209_120985

-- Define the variables and assumptions
variables (l h w : ℝ) (b : ℝ := 6)

-- State the conditions from the problem
-- Condition 1: l + h - w = 40
-- Condition 2: w + h - l + b = 34

theorem height_of_table_without_book (hlw : l + h - w = 40) (whlb : w + h - l + b = 34) : h = 34 :=
by
  -- Since we are skipping the proof, we put sorry here
  sorry

end height_of_table_without_book_l1209_120985


namespace lucas_pay_per_window_l1209_120972

-- Conditions
def num_floors : Nat := 3
def windows_per_floor : Nat := 3
def days_to_finish : Nat := 6
def penalty_rate : Nat := 3
def penalty_amount : Nat := 1
def final_payment : Nat := 16

-- Theorem statement
theorem lucas_pay_per_window :
  let total_windows := num_floors * windows_per_floor
  let total_penalty := penalty_amount * (days_to_finish / penalty_rate)
  let original_payment := final_payment + total_penalty
  let payment_per_window := original_payment / total_windows
  payment_per_window = 2 :=
by
  sorry

end lucas_pay_per_window_l1209_120972


namespace calculate_value_l1209_120931

theorem calculate_value : 12 * ((1/3 : ℝ) + (1/4) - (1/12))⁻¹ = 24 :=
by
  sorry

end calculate_value_l1209_120931


namespace inverse_proposition_l1209_120989

theorem inverse_proposition (a : ℝ) :
  (a > 1 → a > 0) → (a > 0 → a > 1) :=
by 
  intros h1 h2
  sorry

end inverse_proposition_l1209_120989


namespace unique_solution_positive_integers_l1209_120970

theorem unique_solution_positive_integers :
  ∀ (a b : ℕ), (0 < a ∧ 0 < b ∧ ∃ k m : ℤ, a^3 + 6 * a * b + 1 = k^3 ∧ b^3 + 6 * a * b + 1 = m^3) → (a = 1 ∧ b = 1) :=
by
  -- Proof goes here
  sorry

end unique_solution_positive_integers_l1209_120970


namespace find_T_l1209_120934

theorem find_T : 
  ∃ T : ℝ, (3 / 4) * (1 / 6) * T = (1 / 5) * (1 / 4) * 120 ∧ T = 48 :=
by
  sorry

end find_T_l1209_120934


namespace solve_equation_l1209_120982

theorem solve_equation : ∀ (x : ℝ), (x / 2 - 1 = 3) → x = 8 :=
by
  intro x h
  sorry

end solve_equation_l1209_120982


namespace distance_sum_conditions_l1209_120968

theorem distance_sum_conditions (a : ℚ) (k : ℚ) :
  abs (20 * a - 20 * k - 190) = 4460 ∧ abs (20 * a^2 - 20 * k - 190) = 2755 →
  a = -37 / 2 ∨ a = 39 / 2 :=
sorry

end distance_sum_conditions_l1209_120968


namespace lines_parallel_to_skew_are_skew_or_intersect_l1209_120928

-- Define skew lines conditions in space
def skew_lines (l1 l2 : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ¬ (∀ t1 t2 : ℝ, l1 t1 = l2 t2) ∧ ¬ (∃ d : ℝ × ℝ × ℝ, ∀ t : ℝ, l1 t + d = l2 t)

-- Define parallel lines condition in space
def parallel_lines (m l : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ∃ v : ℝ × ℝ × ℝ, ∀ t1 t2 : ℝ, m t1 = l t2 + v

-- Define the relationship to check between lines
def relationship (m1 m2 : ℝ → ℝ × ℝ × ℝ) : Prop :=
  (∃ t1 t2 : ℝ, m1 t1 = m2 t2) ∨ skew_lines m1 m2

-- The main theorem statement
theorem lines_parallel_to_skew_are_skew_or_intersect
  {l1 l2 m1 m2 : ℝ → ℝ × ℝ × ℝ}
  (h_skew: skew_lines l1 l2)
  (h_parallel_1: parallel_lines m1 l1)
  (h_parallel_2: parallel_lines m2 l2) :
  relationship m1 m2 :=
by
  sorry

end lines_parallel_to_skew_are_skew_or_intersect_l1209_120928


namespace number_of_friends_l1209_120907

def initial_candies : ℕ := 10
def additional_candies : ℕ := 4
def total_candies : ℕ := initial_candies + additional_candies
def candies_per_friend : ℕ := 2

theorem number_of_friends : total_candies / candies_per_friend = 7 :=
by
  sorry

end number_of_friends_l1209_120907


namespace tap_B_time_l1209_120918

-- Define the capacities and time variables
variable (A_rate B_rate : ℝ) -- rates in percentage per hour
variable (T_A T_B : ℝ) -- time in hours

-- Define the conditions as hypotheses
def conditions : Prop :=
  (4 * (A_rate + B_rate) = 50) ∧ (2 * A_rate = 15)

-- Define the question and the target time
def target_time := 7

-- Define the goal to prove
theorem tap_B_time (h : conditions A_rate B_rate) : T_B = target_time := by
  sorry

end tap_B_time_l1209_120918


namespace sum_of_squares_divisibility_l1209_120960

theorem sum_of_squares_divisibility (n : ℤ) : 
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  let S := a^2 + b^2 + c^2
  (S % 4 = 0 ∧ S % 3 ≠ 0) :=
by
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  let S := a^2 + b^2 + c^2
  sorry

end sum_of_squares_divisibility_l1209_120960
